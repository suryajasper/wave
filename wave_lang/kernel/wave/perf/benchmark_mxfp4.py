# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Standalone reproducible benchmark for MXFP4 Wave GEMM.

Uses rocprof for benchmarking, invoking kernels with wave runtime.

Shapes from CSV with required columns M, N, K, MT_M, MT_N, and MT_K:

  python -u wave_lang/kernel/wave/perf/benchmark_mxfp4.py --shapes wave_lang/kernel/wave/perf/mxfp4_shapes_macrotiles.csv -o results.csv

Optional env: ATT_LIBRARY_PATH=/path/to/rocprof-trace-decoder/rocm/lib. If set, passed to rocprofv3 (--att --att-library-path).
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.schedules import (
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds,
)
from wave_lang.kernel.wave.templates import (
    get_tagged_mxfp4_gemm_preshuffle_scales_and_B,
)
from wave_lang.kernel.lang.global_symbols import SHARED_ADDRESS_SPACE
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
    b_preshuffle,
    e8m0_shuffle,
)
from wave_lang.kernel.wave.perf.utils import (
    find_rocprof_outputs,
    get_rocprofv3_cmd,
    parse_rocprof_kernel_stats,
)
import wave_lang.kernel.lang as tkl

# Lazy imports for WaveASM (only needed when --backend asm is used)
_waveasm_compiler = None


def _get_waveasm_compiler():
    global _waveasm_compiler
    if _waveasm_compiler is None:
        from waveasm.waveasm_e2e import WaveASMCompiler

        _waveasm_compiler = WaveASMCompiler(
            target="gfx950", keep_temp_files=True
        )
    return _waveasm_compiler


# ---------------------------------------------------------------------------
# Wave kernel template and compile options
# ---------------------------------------------------------------------------


def _get_8wave_shape_from_block(block):
    """Choose an 8-wave shape (4x2 or 2x4) from block M/N dims.

    If either block M or N is 32, force that corresponding wave dimension to 2.
    """
    m_blk, n_blk = block[0], block[1]
    if m_blk == 32 and n_blk == 32:
        raise ValueError(
            "Cannot satisfy both M and N=32 with an 8-wave shape constrained to (4, 2) or (2, 4)."
        )
    if m_blk == 32:
        return (2, 4)
    if n_blk == 32:
        return (4, 2)
    return (4, 2)


def _patch_wave_asm(asm: str) -> str:
    """Apply compatibility patches to Wave assembly source text."""
    asm = re.sub(
        r"^\s*\.amdhsa_code_object_version\s+\d+\s*\n",
        "",
        asm,
        flags=re.MULTILINE,
    )
    asm = re.sub(
        r"(amdhsa\.version:\s*\n\s*-\s*)1(\s*\n\s*-\s*)\d+",
        r"\g<1>1\g<2>0",
        asm,
    )
    return asm


def _kernel_name(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    dynamic: bool = False,
    num_splits: int = 0,
) -> str:
    mt_m, mt_n, mt_k = macrotiles
    splitk_tag = f"_splitk{num_splits}" if num_splits > 0 else ""
    if dynamic:
        return f"wave_mxfp4_dynamic_gemm_{mt_m}x{mt_n}x{mt_k}{splitk_tag}"
    m, n, k = shape
    return f"wave_mxfp4_static_gemm_{mt_m}x{mt_n}x{mt_k}_{m}x{n}x{k}{splitk_tag}"


def _patch_and_save_asm(
    intermediates_dir: Path,
    asm_dir: Path,
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    dynamic: bool = False,
) -> None:
    """Find the .rocmasm in intermediates_dir, patch it, and write to asm_dir."""
    rocmasm_files = glob.glob(str(intermediates_dir / "*.rocmasm"))
    if not rocmasm_files:
        print(
            f"  Warning: no .rocmasm found in {intermediates_dir} for shape {shape}",
            file=sys.stderr,
        )
        return
    src_path = rocmasm_files[0]
    with open(src_path) as f:
        asm = f.read()

    name = _kernel_name(shape, macrotiles, dynamic=dynamic)
    asm = asm.replace("gemm", name)
    asm = _patch_wave_asm(asm)

    asm_dir.mkdir(parents=True, exist_ok=True)
    dst_path = asm_dir / f"{name}.s"
    dst_path.write_text(asm)
    print(f"  ASM saved: {dst_path}")


def _spawn_compile_worker(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    dump_dir: Path,
    asm_dir: Path,
    dynamic: bool,
    num_splits: int = 0,
    backend: str = "llvm",
) -> tuple[bool, str]:
    """Spawn a subprocess to compile one kernel and save its ASM.

    Returns (success, error_msg).
    """
    m, n, k = shape
    mt_m, mt_n, mt_k = macrotiles
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_compile_only",
        "--_shape",
        str(m),
        str(n),
        str(k),
        "--_tiles",
        str(mt_m),
        str(mt_n),
        str(mt_k),
        "--_co-asm-dir",
        str(asm_dir),
        "--_co-dump-dir",
        str(dump_dir),
        "--num-splits",
        str(num_splits),
        "--backend",
        backend,
    ]
    if dynamic:
        cmd.append("--_dynamic")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd()
        )
        if proc.returncode == 0:
            return (True, "")
        stderr_preview = (proc.stderr or "").strip()[-500:]
        return (False, stderr_preview or f"exit code {proc.returncode}")
    except subprocess.TimeoutExpired:
        return (False, "compilation timed out (600s)")
    except Exception as e:
        return (False, str(e))


def get_mxfp4_gemm_wave(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    intermediates_dir: Optional[Path] = None,
    dynamic: bool = False,
):
    wave_shape = _get_8wave_shape_from_block(macrotiles)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
        shape,
        macrotiles,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
        output_dtype=tkl.bf16,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = False
    options.linearize_shared_access = True
    options.wave_runtime = True

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]

    if intermediates_dir is not None:
        options.dump_intermediates = str(intermediates_dir)

    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds(
        use_stagger=True, shape=shape, block=macrotiles
    )
    UNROLL_FACTOR = tkl.sym.UNROLL_FACTOR
    options.subs[UNROLL_FACTOR] = 2
    options.postprocess = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
            %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
            transform.loop.unroll %0 { factor = %%UNROLL_FACTOR%% } : !transform.any_op
            transform.yield
        }
    }
    """
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm, schedule)
    return compiled_gemm


def get_mxfp4_splitk_gemm_waveasm(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    num_splits: int,
    dynamic: bool = False,
):
    """Compile split-K MXFP4 GEMM kernel via WaveASM backend.

    Returns (kernel_info, compilation_result) where kernel_info has MLIR/grid/block
    metadata and compilation_result has the generated assembly text + binary.
    """
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.constraints import ScaledMMAType
    from wave_lang.kernel.wave.templates.gemm import get_splitk_mxfp4_gemm_kernel
    from wave_lang.kernel.lang.global_symbols import GLOBAL_ADDRESS_SPACE
    from waveasm.waveasm_e2e import capture_wave_kernel_info

    wave_shape = _get_8wave_shape_from_block(macrotiles)

    splitk_gemm, hyperparams = get_splitk_mxfp4_gemm_kernel(
        shape,
        num_splits=num_splits,
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
        block_shape=macrotiles,
        waves_per_block=wave_shape,
        preshuffle_B=True,
        preshuffle_scales=True,
        output_type=tkl.bf16,
    )

    hyperparams[tkl.sym.ADDRESS_SPACE] = SHARED_ADDRESS_SPACE

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=True,
        minimize_shared_allocs=False,
    )
    options.use_buffer_ops = True
    options.specialize = True

    dynamic_values = None
    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        dynamic_values = {}
        for sym in options.dynamic_symbols:
            if sym in options.subs:
                dynamic_values[sym] = options.subs[sym]
                del options.subs[sym]

    options = set_default_run_config(options)

    kernel_info = capture_wave_kernel_info(
        options, splitk_gemm, dynamic_values=dynamic_values,
    )

    compiler = _get_waveasm_compiler()
    cpp_result = compiler.compile_full(
        kernel_info.mlir_text, kernel_info.workgroup_size
    )

    if not cpp_result.success:
        raise RuntimeError(
            f"WaveASM compilation failed for shape {shape}: {cpp_result.error_message}"
        )

    return kernel_info, cpp_result


def _save_waveasm_asm(
    cpp_result,
    asm_dir: Path,
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    dynamic: bool = False,
    num_splits: int = 0,
) -> None:
    """Save WaveASM-generated assembly (already complete) to asm_dir."""
    name = _kernel_name(shape, macrotiles, dynamic=dynamic, num_splits=num_splits)
    asm = cpp_result.asm_text

    old_name = cpp_result.get_kernel_name() or "splitk_mxfp4_gemm"
    if old_name != name:
        asm = asm.replace(old_name, name)

    asm = _patch_wave_asm(asm)

    asm_dir.mkdir(parents=True, exist_ok=True)
    dst_path = asm_dir / f"{name}.s"
    dst_path.write_text(asm)
    print(f"  ASM saved: {dst_path}")


def validate_mxfp4_splitk_gemm_waveasm(
    shape: tuple[int, int, int],
    kernel_info,
    cpp_result,
) -> bool:
    """Validate split-K kernel via wave_runtime. Output is zero-initialized."""
    from waveasm.waveasm_e2e import run_with_wave_runtime

    m, n, k = shape
    device = torch.device("cuda")
    x, w, x_scale, w_scale = generate_gemm_afp4wfp4_inputs(shape, device)
    torch_ref = torchScaledGemmMXFP4(x, w, x_scale, w_scale)

    w_t = w.T.contiguous()
    x_scale_ps = e8m0_shuffle(x_scale)
    w_scale_ps = e8m0_shuffle(w_scale)
    w_t_ps = b_preshuffle(w_t)

    x_gpu = x.to(device)
    w_t_ps_gpu = w_t_ps.to(device)
    x_scale_ps_gpu = x_scale_ps.to(device)
    w_scale_ps_gpu = w_scale_ps.to(device)
    c_gpu = torch.zeros(m, n, dtype=torch.bfloat16, device=device)

    binary_path = cpp_result.binary_path
    kernel_name = cpp_result.get_kernel_name() or kernel_info.kernel_name

    run_with_wave_runtime(
        binary_path=binary_path,
        inputs=[x_gpu, x_scale_ps_gpu, w_t_ps_gpu, w_scale_ps_gpu],
        outputs=[c_gpu],
        grid=kernel_info.grid_size,
        block=kernel_info.workgroup_size,
        shared_memory_bytes=kernel_info.lds_size,
        func_name=kernel_name,
    )

    torch.testing.assert_close(
        c_gpu.cpu().to(torch.float32),
        torch_ref,
        check_device=False,
        check_dtype=False,
        rtol=5e-2,
        atol=2.0,
    )
    return True


# ---------------------------------------------------------------------------
# Helpers (IREE/rocprof, validate, benchmark)
# ---------------------------------------------------------------------------


class BenchmarkError(RuntimeError):
    """Raised when a benchmark step fails (compile, validation, or benchmark run)."""

    def __init__(self, message: str, stage: str):
        super().__init__(message)
        self.stage = (
            stage  # "compile_failed", "validation_failed", or "benchmark_failed"
        )


def get_flops(M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K


def get_byte_count_mxfp4(M: int, N: int, K: int) -> int:
    # Packed mxfp4: K/2 bytes per row for A (M rows), B (N rows)
    # Scales: K/32 per row for A and B. Output: M*N*2 (bf16).
    return M * (K // 2) + M * (K // 32) + N * (K // 2) + N * (K // 32) + M * N * 2


def runtime_us_to_tflops(M: int, N: int, K: int, runtime_us: float) -> float:
    if runtime_us <= 0:
        return 0.0
    flops = get_flops(M, N, K)
    return (flops / 1e12) / (runtime_us / 1e6)


# --- rocprof / worker helpers -------------------------------------------------


def _clear_dir(dir_path: os.PathLike) -> None:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def _parse_worker_stdout_for_mean_us(stdout: str) -> tuple[float, bool]:
    """Parse worker stdout for a line 'MEAN_US: <float>'; return (mean_us, ok)."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("MEAN_US:"):
            try:
                value = float(line.split(":", 1)[1].strip())
                return value, True
            except (ValueError, IndexError):
                pass
    return 0.0, False


def _run_torch_benchmark(
    kernel_func,
    inputs: tuple,
    warmup_iters: int = 0,
    benchmark_iters: int = 1,
) -> float:
    """Run warmup and benchmark loop with torch.cuda.synchronize; return mean runtime in microseconds."""
    for _ in range(warmup_iters):
        kernel_func(*inputs)
    torch.cuda.synchronize()

    if benchmark_iters < 1:
        return 0.0
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(benchmark_iters):
        kernel_func(*inputs)
    end_ev.record()
    torch.cuda.synchronize()
    mean_ms = start_ev.elapsed_time(end_ev) / benchmark_iters
    return mean_ms * 1000.0  # us


def run_worker(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    warmup_iters: int = 0,
    benchmark_iters: int = 1,
    dynamic: bool = False,
) -> None:
    """Worker entry: compile GEMM with wave_runtime, run torch benchmark, print MEAN_US to stdout."""
    m, n, k = shape
    gemm_rt = get_mxfp4_gemm_wave(shape, macrotiles, dynamic=dynamic)

    device = torch.device("cuda")
    x, w, x_scale, w_scale = generate_gemm_afp4wfp4_inputs((m, n, k), device)
    w_t = w.T.contiguous()
    w_t_ps = b_preshuffle(w_t)
    x_scale_ps = e8m0_shuffle(x_scale)
    w_scale_ps = e8m0_shuffle(w_scale)
    wave_out = torch.empty(m, n, device=device, dtype=torch.float32)
    inputs = (x, x_scale_ps, w_t_ps, w_scale_ps, wave_out)

    mean_us = _run_torch_benchmark(
        gemm_rt, inputs, warmup_iters=warmup_iters, benchmark_iters=benchmark_iters
    )
    print(f"MEAN_US: {mean_us}")


def validate_mxfp4_gemm(shape: tuple[int, int, int], compiled_gemm) -> bool:
    """Run compiled Wave GEMM with preshuffled inputs and compare to torch reference."""
    m, n, k = shape
    try:
        device = torch.device("cuda")
        x, w, x_scale, w_scale = generate_gemm_afp4wfp4_inputs(shape, device)
        torch_ref = torchScaledGemmMXFP4(x, w, x_scale, w_scale)

        w_t = w.T.contiguous()
        w_t_ps = b_preshuffle(w_t)
        x_scale_ps = e8m0_shuffle(x_scale)
        w_scale_ps = e8m0_shuffle(w_scale)
        wave_out = torch.zeros(m, n, device=device, dtype=torch.bfloat16)

        compiled_gemm(x, x_scale_ps, w_t_ps, w_scale_ps, wave_out)
        torch.testing.assert_close(
            wave_out, torch_ref, check_device=False, check_dtype=False
        )
        return True
    except Exception as e:
        raise RuntimeError(f"Validation failed for shape {shape}: {e}") from e


def benchmark_mxfp4_gemm_rocprof(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    profiler_dump_path: Path,
    att_library_path: Optional[str],
    kernel_regex: str = "gemm",
    timeout: Optional[float] = None,
    warmup_iters: int = 0,
    benchmark_iters: int = 1,
    dynamic: bool = False,
) -> float:
    """Run self as worker under rocprofv3 (torch benchmark); return mean runtime in microseconds.

    Raises RuntimeError if the subprocess times out, exits non-zero, the worker does not
    output MEAN_US, or rocprof parsing fails.
    """
    m, n, k = shape
    mt_m, mt_n, mt_k = macrotiles
    _clear_dir(profiler_dump_path)
    profile_prefix = get_rocprofv3_cmd(
        profiler_dump_path, None, kernel_regex, att_library_path
    )
    worker_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--_shape",
        str(m),
        str(n),
        str(k),
        "--_tiles",
        str(mt_m),
        str(mt_n),
        str(mt_k),
        "--warmup-iters",
        str(warmup_iters),
        "--benchmark-iters",
        str(benchmark_iters),
    ]
    if dynamic:
        worker_cmd.append("--_dynamic")
    full_cmd = profile_prefix + worker_cmd
    try:
        proc = subprocess.run(
            full_cmd,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"rocprof worker subprocess timed out after {timeout}s for shape {shape}"
        ) from e
    if proc.returncode != 0:
        stderr_preview = (proc.stderr or "").strip()[:500]
        raise RuntimeError(
            f"rocprof worker subprocess exited with code {proc.returncode} for shape {shape}"
            + (f"; stderr: {stderr_preview}" if stderr_preview else "")
        )
    _, worker_ok = _parse_worker_stdout_for_mean_us(proc.stdout)
    if not worker_ok:
        raise RuntimeError(
            f"Torch benchmarking failed within worker for shape {shape}: no MEAN_US in worker stdout"
        )
    stats_path, _ = find_rocprof_outputs(profiler_dump_path, None)
    if stats_path is None:
        raise RuntimeError(
            f"rocprof kernel stats file not found under {profiler_dump_path} for shape {shape}"
        )
    rocprof_stats = parse_rocprof_kernel_stats(stats_path, kernel_regex)
    if not rocprof_stats or "mean_duration_us" not in rocprof_stats:
        raise RuntimeError(
            f"rocprof kernel stats parsing failed for regex {kernel_regex!r} in {stats_path} for shape {shape}"
        )
    return rocprof_stats["mean_duration_us"]


def run_validate_and_benchmark(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    dump_dir: Path,
    att_library_path: Optional[str],
    kernel_regex: str = "gemm",
    warmup_iters: int = 0,
    benchmark_iters: int = 1,
    asm_dir: Optional[Path] = None,
    dynamic: bool = False,
    skip_validate: bool = False,
) -> tuple[Optional[float], Optional[float], str]:
    """
    Compile with wave_runtime and validate; then benchmark via torch worker under rocprofv3.
    Returns (runtime_us, tflops, status) where status is "ok", "compile_failed", "validation_failed",
    or "benchmark_failed".

    If asm_dir is set, patched assembly is saved immediately after compilation
    (before validation), so .s files are always generated for successful compiles.

    If skip_validate is True, skips numerical validation and rocprof benchmarking
    entirely — useful for compile-and-save-ASM-only runs.
    """
    m, n, k = shape
    mt_m, mt_n, mt_k = macrotiles
    gemm_id = f"gemm_{m}_{n}_{k}_MT_{mt_m}_{mt_n}_{mt_k}"

    intermediates_dir: Optional[Path] = None
    if asm_dir is not None:
        intermediates_dir = dump_dir / "intermediates" / gemm_id
        intermediates_dir.mkdir(parents=True, exist_ok=True)

    # Compile for validation (wave_runtime=True)
    try:
        gemm_rt = get_mxfp4_gemm_wave(
            shape,
            macrotiles,
            intermediates_dir=intermediates_dir,
            dynamic=dynamic,
        )
    except Exception as e:
        raise BenchmarkError(
            f"Compilation failed for shape {shape}: {e}", stage="compile_failed"
        ) from e

    # Save MLIR to dump directory
    mlir_dir = dump_dir / "mlir"
    mlir_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = mlir_dir / f"{gemm_id}.mlir"
    mlir_path.write_text(gemm_rt.asm)

    # Save patched assembly immediately after compilation (before validation)
    if asm_dir is not None and intermediates_dir is not None:
        try:
            _patch_and_save_asm(
                intermediates_dir,
                asm_dir,
                shape,
                macrotiles,
                dynamic=dynamic,
            )
        except Exception as e:
            print(
                f"  Warning: failed to save ASM for shape {shape}: {e}",
                file=sys.stderr,
            )

    if skip_validate:
        return None, None, "ok"

    # Validate numerics
    try:
        validate_mxfp4_gemm(shape, gemm_rt)
    except Exception as e:
        raise BenchmarkError(
            f"Validation failed for shape {shape}: {e}", stage="validation_failed"
        ) from e

    # Benchmark via torch worker under rocprofv3
    profiler_dump_path = dump_dir / "rocprof" / gemm_id
    try:
        runtime_us = benchmark_mxfp4_gemm_rocprof(
            shape,
            macrotiles,
            profiler_dump_path,
            att_library_path,
            kernel_regex=kernel_regex,
            warmup_iters=warmup_iters,
            benchmark_iters=benchmark_iters,
            dynamic=dynamic,
        )
    except RuntimeError as e:
        raise BenchmarkError(str(e), stage="benchmark_failed") from e

    tflops = runtime_us_to_tflops(m, n, k, runtime_us)
    return runtime_us, tflops, "ok"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone MXFP4 Wave GEMM benchmark (no kernel_bench)."
    )
    p.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--_shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--_tiles",
        type=int,
        nargs=3,
        metavar=("mt_m", "mt_n", "mt_k"),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--_dynamic",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--_compile_only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--_co-asm-dir",
        type=Path,
        default=None,
        dest="_co_asm_dir",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--_co-dump-dir",
        type=Path,
        default=None,
        dest="_co_dump_dir",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--shapes",
        type=Path,
        metavar="CSV",
        help="CSV path with header M,N,K (optional columns MT_M, MT_N, MT_K for macrotile sizes)",
    )
    p.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        metavar="N",
        help="Warmup iterations for benchmark (default: 0)",
    )
    p.add_argument(
        "--benchmark-iters",
        type=int,
        default=1,
        metavar="N",
        help="Benchmark iterations (default: 1)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV for results (required)",
    )
    p.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("/tmp/bench_mxfp4_dump"),
        help="Directory for rocprof output (default: /tmp/bench_mxfp4_dump)",
    )
    p.add_argument(
        "--kernel-regex",
        type=str,
        default="gemm",
        help="Regex for rocprof kernel filter (default: gemm)",
    )
    p.add_argument(
        "--asm-dir",
        type=Path,
        default=None,
        help="If set, save patched assembly (.s) for each valid kernel to this directory",
    )
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="Compile kernels with dynamic dims (M/N/K as runtime args) instead of static shapes",
    )
    p.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip numerical validation and rocprof benchmarking (compile and save ASM only)",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel compile workers for --skip-validate mode (default: 4)",
    )
    p.add_argument(
        "--num-splits",
        type=int,
        default=0,
        metavar="N",
        help="Number of split-K splits (0 = no split-K). "
        "Requires --backend asm. Each split processes K/N elements.",
    )
    p.add_argument(
        "--backend",
        choices=["llvm", "asm"],
        default="llvm",
        help="Compiler backend: 'llvm' (default) or 'asm' (WaveASM C++ backend)",
    )
    return p.parse_args()


def validate_shape_and_macrotiles(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
) -> None:
    """Validate shape and macrotile combination. Raises ValueError with a reason if invalid."""
    M, N, K = shape
    mt_m, mt_n, mt_k = macrotiles
    if M <= 4 or N <= 4 or K <= 4:
        raise ValueError(f"M, N, K must be > 4 (got M={M}, N={N}, K={K})")
    if mt_m > M or mt_n > N or mt_k > K:
        raise ValueError(
            f"Macrotiles must not exceed shape dimensions: "
            f"MT_M({mt_m})<=M({M}), MT_N({mt_n})<=N({N}), MT_K({mt_k})<=K({K})"
        )
    if K % 32 != 0:
        raise ValueError(f"K must be divisible by 32 for scale matrix size (got K={K})")
    if mt_m % 4 != 0:
        raise ValueError(
            f"MT_M must be divisible by 4 (wave constraint) (got MT_M={mt_m})"
        )
    if mt_n % 2 != 0:
        raise ValueError(
            f"MT_N must be divisible by 2 (wave constraint) (got MT_N={mt_n})"
        )


def load_shapes_csv(
    path: Path,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Load shape (M,N,K) and macrotile sizes (MT_M, MT_N, MT_K) from CSV.
    Validates each row with validate_shape_and_macrotiles; raises ValueError on first invalid row.
    """
    rows: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):  # 2 = header is row 1
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            MT_M = int(row["MT_M"])
            MT_N = int(row["MT_N"])
            MT_K = int(row["MT_K"])
            shape = (M, N, K)
            macrotiles = (MT_M, MT_N, MT_K)
            try:
                validate_shape_and_macrotiles(shape, macrotiles)
            except ValueError as e:
                raise ValueError(f"{path}: row {row_idx}: {e}") from e
            rows.append((shape, macrotiles))
    return rows


def _write_results_csv(output_path: Path, results: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "M",
                "N",
                "K",
                "MT_M",
                "MT_N",
                "MT_K",
                "runtime_us",
                "tflops",
                "ok",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"Results written to {output_path}")


def _run_compile_only_worker(args) -> None:
    """Entry point for --_compile_only subprocess: compile one kernel, save ASM, exit."""
    import wave_lang.kernel.wave.cache as _wave_cache

    _wave_cache.WAVE_CACHE_ON = 0

    shape = tuple(args._shape)
    macrotiles = tuple(args._tiles)
    validate_shape_and_macrotiles(shape, macrotiles)

    asm_dir = args._co_asm_dir.resolve()
    dump_dir = args._co_dump_dir.resolve()
    asm_dir.mkdir(parents=True, exist_ok=True)
    dynamic = args._dynamic
    num_splits = args.num_splits
    backend = args.backend

    m, n, k = shape
    mt_m, mt_n, mt_k = macrotiles
    gemm_id = f"gemm_{m}_{n}_{k}_MT_{mt_m}_{mt_n}_{mt_k}"

    if backend == "asm" and num_splits > 0:
        _, cpp_result = get_mxfp4_splitk_gemm_waveasm(
            shape, macrotiles, num_splits, dynamic=dynamic,
        )
        _save_waveasm_asm(
            cpp_result, asm_dir, shape, macrotiles,
            dynamic=dynamic, num_splits=num_splits,
        )
    else:
        intermediates_dir = dump_dir / "intermediates" / gemm_id
        intermediates_dir.mkdir(parents=True, exist_ok=True)
        get_mxfp4_gemm_wave(
            shape, macrotiles, intermediates_dir=intermediates_dir, dynamic=dynamic
        )
        _patch_and_save_asm(
            intermediates_dir, asm_dir, shape, macrotiles, dynamic=dynamic
        )


def _run_parallel_compile(
    shape_rows: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    asm_dir: Path,
    dump_dir: Path,
    dynamic: bool,
    num_workers: int,
    output_path: Path,
    num_splits: int = 0,
    backend: str = "llvm",
) -> None:
    """Compile all kernels in parallel via subprocesses, build manifest, write CSV."""
    asm_dir.mkdir(parents=True, exist_ok=True)

    if dynamic:
        compile_tasks: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        all_shapes_by_mt: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
        for shape, macrotiles in shape_rows:
            all_shapes_by_mt.setdefault(macrotiles, []).append(shape)
            if not any(mt == macrotiles for _, mt in compile_tasks):
                compile_tasks.append((shape, macrotiles))
    else:
        compile_tasks = list(shape_rows)
        all_shapes_by_mt = None

    print(
        f"Compiling {len(compile_tasks)} kernel(s) with {num_workers} parallel workers..."
    )

    compiled: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    failed: list[tuple[tuple[int, int, int], tuple[int, int, int], str]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(
                _spawn_compile_worker, shape, mt, dump_dir, asm_dir, dynamic,
                num_splits, backend,
            ): (shape, mt)
            for shape, mt in compile_tasks
        }
        for future in as_completed(future_to_task):
            shape, mt = future_to_task[future]
            success, err_msg = future.result()
            tag = f"({shape[0]}, {shape[1]}, {shape[2]}) MT({mt[0]},{mt[1]},{mt[2]})"
            if success:
                compiled.append((shape, mt))
                print(f"  OK: {tag}")
            else:
                failed.append((shape, mt, err_msg))
                print(f"  FAIL: {tag}: {err_msg[:200]}", file=sys.stderr)

    # Build manifest
    manifest_entries = []
    block_size = [256, 2, 1]
    if dynamic:
        compiled_mts = {mt for _, mt in compiled}
        for mt in sorted(compiled_mts):
            name = _kernel_name((0, 0, 0), mt, dynamic=True, num_splits=num_splits)
            entry = {
                "name": name,
                "macrotile": list(mt),
                "wave_shapes": [list(s) for s in all_shapes_by_mt[mt]],
                "block_size": block_size,
                "asm_file": f"{name}.s",
                "dynamic": True,
            }
            if num_splits > 0:
                entry["num_splits"] = num_splits
            manifest_entries.append(entry)
    else:
        for shape, mt in sorted(compiled):
            name = _kernel_name(shape, mt, dynamic=False, num_splits=num_splits)
            entry = {
                "name": name,
                "macrotile": list(mt),
                "wave_shape": list(shape),
                "block_size": block_size,
                "asm_file": f"{name}.s",
                "dynamic": False,
            }
            if num_splits > 0:
                entry["num_splits"] = num_splits
            manifest_entries.append(entry)

    if manifest_entries:
        manifest_path = asm_dir / "wave_kernels_manifest.json"
        with open(manifest_path, "w") as mf:
            json.dump({"kernels": manifest_entries}, mf, indent=2)
        print(f"Manifest written to {manifest_path} ({len(manifest_entries)} kernels)")

    # Build results CSV (no timing data for compile-only)
    if dynamic:
        compiled_mts = {mt for _, mt in compiled}
    compiled_set = {(s, mt) for s, mt in compiled}
    results = []
    for shape, macrotiles in shape_rows:
        ok = (
            (macrotiles in compiled_mts)
            if dynamic
            else ((shape, macrotiles) in compiled_set)
        )
        results.append(
            {
                "M": shape[0],
                "N": shape[1],
                "K": shape[2],
                "MT_M": macrotiles[0],
                "MT_N": macrotiles[1],
                "MT_K": macrotiles[2],
                "runtime_us": 0.0,
                "tflops": 0.0,
                "ok": ok,
            }
        )
    _write_results_csv(output_path, results)

    n_ok = len(compiled)
    n_total = len(compile_tasks)
    print(f"\nCompiled {n_ok}/{n_total} kernels successfully.")
    if failed:
        print(f"Failed compilations ({len(failed)}):", file=sys.stderr)
        for s, mt, err in failed:
            print(
                f"  ({s[0]},{s[1]},{s[2]}) MT({mt[0]},{mt[1]},{mt[2]}): {err[:200]}",
                file=sys.stderr,
            )
    if n_ok == 0:
        sys.exit(1)


def main():
    att_library_path = os.environ.get("ATT_LIBRARY_PATH") or None

    args = parse_args()

    # --- Internal compile-only worker (subprocess entry point) ---
    if args._compile_only:
        if args._shape is None or args._tiles is None:
            print("--_compile_only requires --_shape and --_tiles.", file=sys.stderr)
            sys.exit(1)
        if args._co_asm_dir is None or args._co_dump_dir is None:
            print(
                "--_compile_only requires --_co-asm-dir and --_co-dump-dir.",
                file=sys.stderr,
            )
            sys.exit(1)
        _run_compile_only_worker(args)
        return

    # --- Internal benchmark worker (subprocess entry point) ---
    if args._worker:
        if args._shape is None:
            print("--_worker requires --_shape M N K.", file=sys.stderr)
            sys.exit(1)
        if args._tiles is None:
            print("--_worker requires --_tiles mt_m mt_n mt_k.", file=sys.stderr)
            sys.exit(1)
        shape = tuple(args._shape)
        macrotiles = tuple(args._tiles)
        try:
            validate_shape_and_macrotiles(shape, macrotiles)
        except ValueError as e:
            print(f"Invalid shape/macrotile: {e}", file=sys.stderr)
            sys.exit(1)
        run_worker(
            shape,
            macrotiles,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            dynamic=args._dynamic,
        )
        return

    # --- Main entry: requires --shapes ---
    if args.shapes is None:
        print("--shapes <path/to/csv> is required.", file=sys.stderr)
        sys.exit(1)
    if not args.shapes.exists():
        print(f"Shapes file not found: {args.shapes}", file=sys.stderr)
        sys.exit(1)
    if args.output is None:
        print("--shapes requires -o/--output for the result CSV.", file=sys.stderr)
        sys.exit(1)

    try:
        shape_rows = load_shapes_csv(args.shapes)
    except ValueError as e:
        print(f"Invalid shape/macrotile in CSV: {e}", file=sys.stderr)
        sys.exit(1)
    if not shape_rows:
        print("No shapes found in CSV.", file=sys.stderr)
        sys.exit(1)

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dump_dir = dump_dir / run_id
    run_dump_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dump directory for this run: {run_dump_dir}")
    asm_dir = args.asm_dir
    dynamic = args.dynamic

    num_splits = args.num_splits
    backend = args.backend

    if dynamic:
        print("Dynamic mode: kernels will use runtime M/N/K dimensions")
    if num_splits > 0:
        print(f"Split-K mode: {num_splits} splits, backend={backend}")

    # --- Parallel compile-only path (--skip-validate) ---
    if args.skip_validate:
        if asm_dir is None:
            print("--skip-validate requires --asm-dir.", file=sys.stderr)
            sys.exit(1)
        _run_parallel_compile(
            shape_rows,
            asm_dir,
            run_dump_dir,
            dynamic,
            num_workers=args.jobs,
            output_path=args.output,
            num_splits=num_splits,
            backend=backend,
        )
        return

    # --- Sequential compile + validate + benchmark path ---
    kernel_regex = args.kernel_regex
    warmup_iters = args.warmup_iters
    benchmark_iters = args.benchmark_iters

    results = []
    manifest_entries = []
    failed_compilation = []
    failed_validation = []
    failed_benchmark = []
    saved_dynamic_macrotiles: set[tuple[int, int, int]] = set()
    dynamic_manifest_shapes: dict[tuple[int, int, int], list[list[int]]] = {}

    for shape, macrotiles in shape_rows:
        M, N, K = shape
        mt_m, mt_n, mt_k = macrotiles

        effective_asm_dir = asm_dir
        if dynamic and macrotiles in saved_dynamic_macrotiles:
            effective_asm_dir = None

        if backend == "asm" and num_splits > 0:
            # --- Split-K WaveASM path ---
            status = "ok"
            runtime_us, tflops = None, None
            try:
                kernel_info, cpp_result = get_mxfp4_splitk_gemm_waveasm(
                    shape, macrotiles, num_splits, dynamic=dynamic,
                )
                if effective_asm_dir is not None:
                    _save_waveasm_asm(
                        cpp_result, effective_asm_dir, shape, macrotiles,
                        dynamic=dynamic, num_splits=num_splits,
                    )
                    if dynamic:
                        saved_dynamic_macrotiles.add(macrotiles)
                        dynamic_manifest_shapes.setdefault(macrotiles, []).append(
                            list(shape)
                        )
                try:
                    validate_mxfp4_splitk_gemm_waveasm(shape, kernel_info, cpp_result)
                except Exception as e:
                    raise BenchmarkError(
                        f"Validation failed for shape {shape}: {e}",
                        stage="validation_failed",
                    ) from e
            except BenchmarkError as e:
                print(f"  {e}", file=sys.stderr)
                traceback.print_exc()
                status = e.stage
            except Exception as e:
                print(f"  Compilation failed: {e}", file=sys.stderr)
                traceback.print_exc()
                status = "compile_failed"
            ok = status == "ok"
        else:
            # --- Standard LLVM path ---
            try:
                runtime_us, tflops, status = run_validate_and_benchmark(
                    shape,
                    macrotiles,
                    run_dump_dir,
                    att_library_path,
                    kernel_regex=kernel_regex,
                    warmup_iters=warmup_iters,
                    benchmark_iters=benchmark_iters,
                    asm_dir=effective_asm_dir,
                    dynamic=dynamic,
                )
            except BenchmarkError as e:
                print(f"  {e}", file=sys.stderr)
                traceback.print_exc()
                status = e.stage
                runtime_us, tflops = None, None
            ok = status == "ok"

            if asm_dir is not None and status != "compile_failed":
                name = _kernel_name(shape, macrotiles, dynamic=dynamic)
                asm_file = asm_dir / f"{name}.s"
                if asm_file.exists():
                    if dynamic:
                        saved_dynamic_macrotiles.add(macrotiles)
                        dynamic_manifest_shapes.setdefault(macrotiles, []).append(
                            list(shape)
                        )
                    else:
                        manifest_entries.append(
                            {
                                "name": name,
                                "macrotile": list(macrotiles),
                                "wave_shape": list(shape),
                                "block_size": [256, 2, 1],
                                "asm_file": f"{name}.s",
                                "dynamic": False,
                            }
                        )

        if status == "compile_failed":
            failed_compilation.append((M, N, K))
        elif status == "validation_failed":
            failed_validation.append((M, N, K))
        elif status == "benchmark_failed":
            failed_benchmark.append((M, N, K))
        mean_us = runtime_us if runtime_us is not None else 0.0
        tflops_val = tflops if tflops is not None else 0.0
        results.append(
            {
                "M": M,
                "N": N,
                "K": K,
                "MT_M": mt_m,
                "MT_N": mt_n,
                "MT_K": mt_k,
                "runtime_us": mean_us,
                "tflops": tflops_val,
                "ok": ok,
            }
        )
        status_str = "ok" if ok else status
        print(
            f"  ({M}, {N}, {K}) MT({mt_m},{mt_n},{mt_k}): {mean_us:.2f} us, {tflops_val:.4f} TFLOPs [{status_str}]"
        )

    for mt_key in sorted(dynamic_manifest_shapes):
        name = _kernel_name(
            (0, 0, 0), mt_key, dynamic=True, num_splits=num_splits,
        )
        entry = {
            "name": name,
            "macrotile": list(mt_key),
            "wave_shapes": dynamic_manifest_shapes[mt_key],
            "block_size": [256, 2, 1],
            "asm_file": f"{name}.s",
            "dynamic": True,
        }
        if num_splits > 0:
            entry["num_splits"] = num_splits
        manifest_entries.append(entry)

    if asm_dir is not None and manifest_entries:
        manifest_path = asm_dir / "wave_kernels_manifest.json"
        with open(manifest_path, "w") as mf:
            json.dump({"kernels": manifest_entries}, mf, indent=2)
        print(f"Manifest written to {manifest_path} ({len(manifest_entries)} kernels)")

    if failed_compilation:
        print(f"Kernels that failed compilation: {failed_compilation}", file=sys.stderr)
    if failed_validation:
        print(
            f"Kernels that failed numerical validation: {failed_validation}",
            file=sys.stderr,
        )
    if failed_benchmark:
        print(f"Kernels that failed benchmarking: {failed_benchmark}", file=sys.stderr)

    valid = [r for r in results if r["ok"] and r["runtime_us"] > 0]
    if not valid:
        print("No successful runs.", file=sys.stderr)
        sys.exit(1)

    avg_us = sum(r["runtime_us"] for r in valid) / len(valid)
    avg_tflops = sum(r["tflops"] for r in valid) / len(valid)
    best = max(valid, key=lambda r: r["tflops"])

    print()
    print(
        f"Average across {len(valid)} kernels: {avg_us:.2f} us, {avg_tflops:.4f} TFLOPs"
    )
    print(
        f"Best kernel: M={best['M']}, N={best['N']}, K={best['K']} MT({best['MT_M']},{best['MT_N']},{best['MT_K']}) -> "
        f"{best['runtime_us']:.2f} us, {best['tflops']:.4f} TFLOPs"
    )
    _write_results_csv(args.output, results)


if __name__ == "__main__":
    main()
