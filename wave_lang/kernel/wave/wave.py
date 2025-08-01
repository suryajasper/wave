# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Lang, compiler, ops, constraints
import inspect
import logging
import warnings
from itertools import chain

# Others
from typing import Any, Callable, Optional, Sequence, get_type_hints

import sympy
import torch.fx as fx
from sympy.utilities.lambdify import lambdastr

import wave_lang.kernel.lang as tkl
from wave_lang.support.ir_imports import Context, Module, Operation

from .._support.indexing import IndexExpr, IndexingContext, index_symbol
from .._support.location_config import LocationCaptureConfig
from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    KernelRegionGraph,
    Launchable,
)
from ..compiler import builder, dispatch_codegen, kernel_codegen
from ..lang import Grid, Memory, SymbolBind
from ..lang.global_symbols import *
from ..ops import wave_ops
from ..ops.wave_ops import CustomOp, Iterate, get_custom

# Passes
from .analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from .analysis.partition_strided_operators import (
    partition_ops_with_gpr_offsets,
    partition_strided_operators,
)
from .barriers import add_shared_memory_barriers
from .cache import get_temp_binary_dir
from .codegen import WaveEmitter
from .compile_options import WaveCompileOptions
from .constraints import (
    Constraint,
    HardwareConstraint,
    ReorderingConstraint,
    TilingConstraint,
    WaveConstraint,
    WorkgroupConstraint,
    get_grid_shape,
)
from .debug_log_hoist import (
    debug_log_hoist,
    debug_log_write_replace,
    DebugArgInfo,
)
from .decompose_dot_mma import decompose_dot_mma
from .decompose_reduce_ops import decompose_reduce_ops
from .decompose_scan_ops import decompose_scan_ops
from .decompose_vmma_ops import decompose_vmma_ops
from .expansion.expansion import add_get_results, expand_graph
from .gather_to_shared import gather_to_shared
from .generate_bounds_exprs import generate_bounds_exprs
from .global_to_shared_gathers import global_to_shared_gathers
from .hoisting import hoist_loop_invariant_ops
from .in_thread_transpose import in_thread_transpose
from .memory_analysis.minimize_shared_allocs import minimize_shared_allocs
from .minimize_global_loads import minimize_global_loads
from .promotion import compute_shared_memory_usage, promote_placeholders
from .schedule_reordering import schedule_reordering
from .scheduling.schedule import schedule_graph
from .shared_memory_indexing import apply_shared_memory_indexing_corrections
from .symbolic_constraints import SymbolicAlias
from .type_inference import infer_types
from .utils.compile_utils import canonicalize_module
from .utils.general_utils import (
    delinearize_index,
    get_hardware_constraint,
    partial,
    remove_files_with_extension,
)
from .utils.graph_utils import (
    initialize_iter_args,
    remove_chained_extractslice,
    remove_chained_getresult,
)
from .utils.print_utils import print_trace, try_apply_pass

# Utils
from .utils.symbol_utils import safe_subs, subs_idxc
from .workgroup_reordering import reorder_workgroups

logger = logging.getLogger(__name__)


__all__ = ["wave", "wave_trace_only"]

# Warn only once
_warned = False


def _are_versions_compatible(ver1: "Version", ver2: "Version") -> bool:
    if ver1.is_prerelease or ver2.is_prerelease:
        return ver1 == ver2
    else:
        # For stable releases, it is fine if the patch level mismatches.
        return (ver1.major == ver2.major) and (ver1.minor == ver2.minor)


def _warn_iree_is_too_old():
    """
    Issue a warning if IREE runtime and compiler versions mismatch or IREE
    version is too low.

    Warning is issued only once.
    """
    global _warned
    if _warned:
        return

    _warned = True

    try:
        from importlib.metadata import version

        from packaging.version import Version

        iree_compiler_ver = Version(version("iree-base-compiler"))
        iree_runtime_ver = Version(version("iree-base-runtime"))
    except:
        return

    if not _are_versions_compatible(iree_compiler_ver, iree_runtime_ver):
        warnings.warn(
            f"IREE compiler and runtime versions mismatch: {iree_compiler_ver} and {iree_runtime_ver}"
        )

    # Increment only when IREE has breaking changes.
    # We don't want to enforce it on package level or make it a hard error just yet.
    min_iree_version = Version("3.6.0rc20250721")
    if iree_compiler_ver < min_iree_version:
        warnings.warn(
            f"IREE version is too old: {iree_compiler_ver}, min version: {min_iree_version}"
        )


def wave(constraints: Optional[list[Constraint]] = None):
    def decorator(f: Callable[..., Any]) -> "LaunchableWave":
        return LaunchableWave(constraints, f.__name__, f)

    return decorator


def wave_trace_only(
    constraints: Optional[list[Constraint]] = None,
    *,
    location_capture_config: Optional[LocationCaptureConfig] = None,
):
    def decorator(f: Callable[..., Any]) -> "Callable[[], CapturedTrace]":
        wave = LaunchableWave(constraints, f.__name__, f)
        return lambda: wave._trace(location_capture_config=location_capture_config)  # type: ignore

    return decorator


def _is_symbol_bind(a: Any) -> bool:
    return inspect.isclass(a) and issubclass(a, SymbolBind)


def _is_memory_arg(a: Any) -> bool:
    return inspect.isclass(a) and issubclass(a, Memory)


class LaunchableWave(Launchable):
    def __init__(
        self,
        constraints: Optional[list[Constraint]],
        name: str,
        eager_function: Callable[[Any], Any],
    ):
        super().__init__(eager_function)

        self.constraints = constraints if constraints else []
        self.induction_vars: dict[CustomOp, IndexExpr] = {}
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)

        self.grid_type = Grid[tuple(get_grid_shape(self.workgroup_constraints))]

        # TODO: needed for the wave_runtime grid calculations, we should really
        # just generate host wrapper suitable for wave_runtime instead of doing
        # it in python (and it will be faster as well).
        hints = get_type_hints(eager_function)
        self.bound_scalar_symbols = {
            index_symbol(name): i
            for i, (name, arg) in enumerate(hints.items())
            if _is_symbol_bind(arg)
        }

        # Build a mapping between symbol and tensor arg (index, dim) so we can
        # use it to extract dynamic symbols from the tensor args.
        symbols_args_map = {}
        for arg_idx, arg in enumerate(hints.values()):
            if not _is_memory_arg(arg):
                continue

            for dim, symbol in enumerate(arg.symbolic_shape):
                if symbol in symbols_args_map:
                    continue

                symbols_args_map[symbol] = (arg_idx, dim)
        self.symbols_args_map = symbols_args_map

    @property
    def workgroup_constraints(self) -> list[WorkgroupConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WorkgroupConstraint)
        ]

    @property
    def tiling_constraints(self) -> list[TilingConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, TilingConstraint)
        ]

    @property
    def wave_constraints(self) -> list[WaveConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WaveConstraint)
        ]

    @property
    def hardware_constraints(self) -> list[HardwareConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, HardwareConstraint)
        ]

    @property
    def reordering_constraints(self) -> list[ReorderingConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, ReorderingConstraint)
        ]

    @property
    def symbolic_constraints(self) -> list[HardwareConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, SymbolicAlias)
        ]

    def _trace(
        self, *, location_capture_config: Optional[LocationCaptureConfig] = None
    ) -> CapturedTrace:
        region_graph = KernelRegionGraph(
            location_capture_config=location_capture_config, func=self._f
        )
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            # Get all explictly defined custom ops
            custom_ops: dict[str, wave_ops.CustomOp] = {
                cls.tkw_op_name: cls
                for _, cls in inspect.getmembers(wave_ops, inspect.isclass)
                if issubclass(cls, wave_ops.CustomOp) and hasattr(cls, "tkw_op_name")
            }

            # Register custom ops
            for name, op in custom_ops.items():
                context.register_custom_op(name, op)

            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                trace = CapturedTrace(region_graph, root_name)

        return trace

    def create_induction_vars(self, trace: CapturedTrace) -> None:
        """
        Creates induction variables for all the reductions in the graph
        and associates tiling constraints all the reduction dimensions
        with the appropriate induction variables.

        """

        def is_reduction(node: fx.Node):
            custom = get_custom(node)
            return isinstance(custom, Iterate)

        reduction_nodes = trace.walk(is_reduction)
        for node in reduction_nodes:
            custom = get_custom(node)
            self.induction_vars[custom] = tkl.IndexSymbol(
                "$ARG" + str(custom.axis), integer=True, nonnegative=True
            )
            for tiling_constraint in self.tiling_constraints:
                if tiling_constraint.dim == custom.axis:
                    tiling_constraint.induction_var = self.induction_vars[custom]

    def initialize_wave_constraints(self) -> None:
        """
        For each wave constraint, determines the appropriate wave id by looking
        for workgroup constraints along the same dimension and using information
        from the hardware constraints.

        """

        hardware_constraint = self.hardware_constraints[0]
        for wave_constraint in self.wave_constraints:
            for workgroup_constraint in self.workgroup_constraints:
                if wave_constraint.dim == workgroup_constraint.dim:
                    wave_constraint.set_wave_id_from_hardware_and_workgroup_constraint(
                        hardware_constraint, workgroup_constraint
                    )

        if hardware_constraint.waves_per_block is None:
            waves_per_block = [1, 1, 1]
            for wave_constraint in self.wave_constraints:
                count = subs_idxc(wave_constraint.waves_per_block)
                waves_per_block[wave_constraint.workgroup_dim] = count

            hardware_constraint.waves_per_block = tuple(waves_per_block)

    def initialize_reductions(self, trace: CapturedTrace) -> None:
        """
        For each reduction, initializes the reduction count by looking at the
        tiling constraints associated with the reduction.

        """
        is_reduction = lambda node: isinstance(get_custom(node), Iterate)
        for reduction in trace.walk(is_reduction):
            for tiling_constraint in self.tiling_constraints:
                if tiling_constraint.dim == get_custom(reduction).axis:
                    reduction.count = subs_idxc(tiling_constraint.count)

    def get_workgroup_dims(self) -> list[int]:
        """
        Returns the workgroup dimensions that are not aliased.
        """
        # Ignore aliased variables. They will be handled separately.
        aliased_dims = [
            x.source for x in self.constraints if isinstance(x, SymbolicAlias)
        ]
        workgroup_dims = [
            x for x in self.workgroup_constraints if x.dim not in aliased_dims
        ]
        return workgroup_dims

    def update_aliased_workgroup_constraints(
        self, workgroup_dims: dict[int, int]
    ) -> None:
        """
        This function updates the wg_dim for aliased workgroup constraints.
        """
        aliased_dims = [
            x.source for x in self.constraints if isinstance(x, SymbolicAlias)
        ]
        # Update the workgroup constraints for aliases sources.
        for constraint in self.workgroup_constraints:
            if constraint.dim in aliased_dims:
                constraint.wg_dim = workgroup_dims[constraint.workgroup_dim].wg_dim

    def initialize_workgroup_constraints(self) -> None:
        """
        For kernels that distribute more than three dimensions among workgroups,
        we need to update the workgroup constraints for dimensions >= 2
        with the appropriate workgroup index.
        """

        workgroup_dims = self.get_workgroup_dims()
        # Filter to WG2 and above.
        dims_to_delinearize = [x for x in workgroup_dims if x.workgroup_dim >= 2]
        if all(x.workgroup_dim <= 2 for x in dims_to_delinearize):
            return
        # Only take account primary dim for delinearize shape.
        shape = [subs_idxc(x.count) for x in dims_to_delinearize if x.primary]
        new_workgroup_dims = delinearize_index(WORKGROUP_2, shape)
        for delinearize_dim in dims_to_delinearize:
            delinearize_dim.wg_dim = new_workgroup_dims[
                delinearize_dim.workgroup_dim - 2
            ]
        self.update_aliased_workgroup_constraints(workgroup_dims)

    def initialize_symbolic_constraints(self) -> None:
        """
        For each symbolic constraint, create new constraints for the
        related symbolic values with appropriate substitutions.
        """
        new_wg_constraints, new_wave_constraints, new_tiling_constraints = [], [], []
        for symbolic_constraint in self.symbolic_constraints:
            new_wg_constraints += symbolic_constraint.create_new_constraints(
                self.workgroup_constraints
            )
            new_wave_constraints += symbolic_constraint.create_new_constraints(
                self.wave_constraints
            )
            new_tiling_constraints += symbolic_constraint.create_new_constraints(
                self.tiling_constraints
            )
        # Remove wave constraints with same tile size as workgroup constraints
        for wave_constraint in new_wave_constraints:
            for workgroup_constraint in new_wg_constraints:
                if (
                    wave_constraint.dim == workgroup_constraint.dim
                    and wave_constraint.tile_size == workgroup_constraint.tile_size
                ):
                    new_wave_constraints.remove(wave_constraint)
        self.constraints += (
            new_wg_constraints + new_wave_constraints + new_tiling_constraints
        )
        idxc = IndexingContext.current()
        for constraint in self.symbolic_constraints:
            if subs_idxc(constraint.target).is_number:
                idxc._bind_symbol(
                    constraint.source,
                    subs_idxc(constraint.source_to_target(constraint.target)),
                )

    def infer_grid_shape(self, idxc: IndexingContext):
        self.grid_type.dims = [1, 1, 1]
        max_workgroup_dim = 2
        aliases = [x.source for x in self.constraints if isinstance(x, SymbolicAlias)]
        for constraint in self.workgroup_constraints:
            if constraint.dim in aliases:
                continue
            if not constraint.primary:
                continue
            dim = (
                constraint.workgroup_dim
                if constraint.workgroup_dim < max_workgroup_dim
                else max_workgroup_dim
            )
            self.grid_type.dims[dim] *= safe_subs(constraint.count, idxc.subs)

    def compile_to_mlir(
        self,
        trace: CapturedTrace,
        context: Context,
        module_op: Optional[Module] = None,
        options: WaveCompileOptions = None,
    ):
        entrypoint_name = self._name
        root_graph = trace.get_root_graph()
        kernel_sig = kernel_codegen.KernelSignature()
        kernel_sig.add_from_graph_placeholders(root_graph)
        kernel_sig.add_from_dynamic_symbols(options.dynamic_symbols)
        kernel_sig.add_grid(self.grid_type)
        kernel_sig.determine_input_output_buffers(root_graph)
        if options.print_signature:
            print(kernel_sig)

        mb = builder.ModuleBuilder(context=context, module_op=module_op)
        exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
        workgroup_size = self.hardware_constraints[0].threads_per_block
        subgroup_size = self.hardware_constraints[0].threads_per_wave

        # Setup LLVM func compilation configs.
        llvm_func_config = {}
        if options.denorm_fp_math_f32:
            llvm_func_config["denormal-fp-math-f32"] = options.denorm_fp_math_f32

        if options.waves_per_eu:
            llvm_func_config["amdgpu-waves-per-eu"] = options.waves_per_eu

        dispatch_entrypoint = exe.define_entrypoint(
            entrypoint_name,
            kernel_sig,
            self.grid_type,
            workgroup_size,
            subgroup_size,
            options.dynamic_symbols,
            llvm_func_config,
        )

        emitter = WaveEmitter(
            dispatch_entrypoint, trace, self.constraints, options, self.grid_type
        )
        try:
            emitter.emit(trace.get_root_graph())
        except:
            logger.info("Error in emitter")
            asm = mb.module_op.get_asm()
            logger.info(asm)
            raise
        emitter.finish()

        if options.canonicalize:
            canonicalize_module(mb.module_op)

        return mb, trace, exe, kernel_sig, entrypoint_name

    def build_initial_pass_pipeline(
        self,
        trace: CapturedTrace,
        options: WaveCompileOptions,
        debug_arg_info: list[DebugArgInfo],
        print_ir_before: Sequence[str] = [],
        print_ir_after: Sequence[str] = [],
    ):
        idxc = IndexingContext.current()

        def finalize_indices():
            idxc.finalize()

        def substitute_vector_shapes():
            self.hardware_constraints[0].subs_vector_shapes(idxc.subs)

        return [
            partial(debug_log_hoist, trace),
            partial(initialize_iter_args, trace),
            partial(self.create_induction_vars, trace),
            partial(self.initialize_reductions, trace),
            finalize_indices,
            substitute_vector_shapes,
            partial(add_get_results, trace),
            partial(infer_types, trace),
            partial(debug_log_write_replace, trace, debug_arg_info),
            partial(
                promote_placeholders,
                trace,
                self.constraints,
                options.reorder_allocs,
            ),
            partial(
                set_node_indices,
                trace,
                self.constraints,
                print_ir_before,
                print_ir_after,
            ),
            partial(reorder_workgroups, trace, self.reordering_constraints),
            partial(expand_graph, trace, self.constraints),
            partial(set_post_expansion_indices, trace, self.constraints),
            partial(remove_chained_getresult, trace),
        ]

    def _trace_and_get_kernel_signature(
        self,
        options: WaveCompileOptions,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ) -> tuple[
        builder.ModuleBuilder,
        CapturedTrace,
        dispatch_codegen.StreamExecutable,
        kernel_codegen.KernelSignature,
        str,
        WaveCompileOptions,
        Sequence[DebugArgInfo],
    ]:
        # Issue a warning if IREE ver is too low.
        # Warning will only be issued if we are compiling the kernel and won't
        # if we are using cached kernel as we don't want to add any additional
        # overhead to 'happy' path.
        _warn_iree_is_too_old()

        # Build wave runtime, if specified.
        if options.wave_runtime:
            # Remove any existing hsaco files in this directory.
            # If the kernel is being cached, then it will be referenced from the
            # cache directory. When kernels are not being cached, we remove them
            # to ensure that at any time there is only one hsaco file in this directory.
            remove_files_with_extension(get_temp_binary_dir(), ".hsaco")

        print_ir_after = options.print_ir_after
        print_ir_before = options.print_ir_before
        if options.print_trace_begin:
            print(f"\n***Tracing kernel {self._name}***")

        debug_arg_info = []

        trace = self._trace(location_capture_config=options.location_capture_config)
        if (
            "all" in print_ir_after
            or "all" in print_ir_before
            or "trace" in print_ir_after
            or "first" in print_ir_before
        ):
            print(f"***After trace/Before first pass***\n")
            print_trace(trace)

        # Initial passes, pre-optimization.
        graph_passes = self.build_initial_pass_pipeline(
            trace, options, debug_arg_info, print_ir_before, print_ir_after
        )

        graph_passes += [
            partial(decompose_vmma_ops, trace, self.constraints),
            partial(decompose_dot_mma, trace, self.constraints),
        ]

        # Optimizations.
        if options.optimization_level:
            graph_passes += [
                partial(hoist_loop_invariant_ops, trace, self.constraints),
                partial(gather_to_shared, trace, self.constraints, options),
                partial(in_thread_transpose, trace, self.constraints),
                partial(global_to_shared_gathers, trace, self.constraints),
                partial(minimize_global_loads, trace, self.constraints),
            ]
        graph_passes += [
            partial(apply_shared_memory_indexing_corrections, trace, self.constraints),
        ]

        # Partition strided operators.
        graph_passes += [
            partial(partition_ops_with_gpr_offsets, trace, self.constraints),
            partial(partition_strided_operators, trace, self.constraints),
            partial(remove_chained_extractslice, trace),
        ]

        graph_passes += [
            partial(decompose_reduce_ops, trace, self.constraints),
            partial(decompose_scan_ops, trace, self.constraints),
        ]

        # Schedule the reduction ops.
        scheduling_type = options.schedule
        use_scheduling_barriers = options.use_scheduling_barriers
        graph_passes.append(
            partial(
                schedule_graph,
                trace,
                self.constraints,
                use_scheduling_barriers,
                scheduling_type,
                options.override_schedule,
                options.dump_schedule,
            )
        )

        if options.optimization_level:
            graph_passes += [
                partial(
                    schedule_reordering,
                    trace,
                    self.constraints,
                    scheduling_type,
                ),
                partial(
                    minimize_shared_allocs,
                    trace,
                    options.minimize_shared_allocs,
                ),
            ]
        graph_passes += [
            partial(add_shared_memory_barriers, trace),
            partial(compute_shared_memory_usage, trace, options.kernel_launch_info),
            partial(generate_bounds_exprs, trace, self.constraints),
        ]

        pass_times = {}
        for p in graph_passes:
            try_apply_pass(p, trace, print_ir_before, print_ir_after, pass_times)

        if options.print_pass_times:
            pass_times_list = sorted(
                pass_times.items(), key=lambda x: x[1], reverse=True
            )

            print(f"Pass times:")
            for k, v in pass_times_list:
                print(f"    {k}: {v:.4f}s")

        if "all" in print_ir_after or "last" in print_ir_after:
            # Take advantage of Python leaking loop variables
            print(f"***After final pass {p.__name__}***\n")
            print_trace(trace)

        # Determine grid shape.
        self.infer_grid_shape(IndexingContext.current())
        if options.print_grid:
            print(f"Grid: {self.grid_type}")

        # Add grid and block dims to kernel launch info.
        # Convert the grid into a lambda that we can use to compute the grid dimension.
        hw_constraint = get_hardware_constraint(self.constraints)
        grid_symbols = list(self.bound_scalar_symbols.keys()) + list(
            options.dynamic_symbols
        )
        options.kernel_launch_info.grid = sympy.lambdify(
            [grid_symbols], self.grid_type.dims
        )
        options.kernel_launch_info.grid_str = lambdastr(
            [grid_symbols], self.grid_type.dims
        )
        options.kernel_launch_info.blocks = [
            int(x) for x in hw_constraint.threads_per_block
        ]
        options.kernel_launch_info.func_name = self._name

        idxc = IndexingContext.current()
        for sym, val in zip(
            [THREAD_0, THREAD_1, THREAD_2, WORKGROUP_0, WORKGROUP_1, WORKGROUP_2],
            chain(hw_constraint.threads_per_block, self.grid_type.dims),
        ):
            if safe_subs(val, idxc.subs) == 1:
                idxc.bind_constant(sym, 0)

        return (
            *self.compile_to_mlir(trace, context, module_op, options=options),
            options,
            debug_arg_info,
        )

    def aot_execute(self, args, kwargs):
        raise NotImplementedError("AOT execution for wave not implemented yet.")

    def eager_execute(self, args, kwargs):
        raise NotImplementedError("Eager execution for wave not implemented yet.")

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"
