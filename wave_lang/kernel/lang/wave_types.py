from itertools import chain
from typing import (
    Any,
    ClassVar,
    Iterable,
    Optional,
    Sequence,
    Type,
    TypeAlias,
    TypeVar,
)

from sympy import Symbol
from sympy.core.expr import Expr
from typing_extensions import Self

from .._support.dtype import DataType
from .._support.indexing import IndexExpr, IndexSymbol, index_symbol
from .kernel_buffer import (
    AddressSpace,
    KernelBufferMeta,
    KernelBufferUsage,
    MemoryLayout,
)

__all__ = [
    "IndexMapping",
    "Memory",
    "Register",
]

MemoryTypeT = TypeVar("MemoryTypeT")


class Memory(metaclass=KernelBufferMeta):
    """
    Represents storage anywhere in the memory hierarchy except registers.
    Parameterized by a shape, address space and element type. The allocated
    memory is traversed by an iterator that specifies the offset, stride
    and size along each dimension.

    The symbolic shape specified here can be interpreted as the logical shape
    of the memory buffer that may or may not be the same as the physical shape.
    If the physical shape is different, it can be specified using the
    physical_layout parameter.

    As an example, consider a GEMM output buffer of logical shape (M, N) where
    M and N are the parallel dimensions of the problem. This is logical shape
    of the buffer. However, the physical shape of the buffer may be (M', N')
    and can be specified as

    Memory[(M, N), AddressSpace.GLOBAL_MEMORY, dtype, MemoryLayout(shape=(M', N'))]

    """

    address_space: ClassVar[int]
    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    rank: ClassVar[int]
    dtype: ClassVar[DataType]
    physical_layout: ClassVar[Optional[MemoryLayout]]
    usage: ClassVar[Optional[KernelBufferUsage]]

    def __init__(self) -> None:
        raise NotImplementedError("Memory types are not directly instantiated.")

    def __class_getitem__(
        cls, shape_and_dtype: tuple[IndexExpr | DataType, ...]
    ) -> Type["Memory"]:
        """
        Syntax: `Memory[shape1, ...., shapeN, addressSpace, dtype, Optional[usage]]`
        or `Memory[(shape1, ..., shapeN), addressSpace, dtype, Optional[usage]]`
        """
        if len(shape_and_dtype) < 3:
            raise TypeError(f"Expected at least 3 arguments, got: {shape_and_dtype}")

        usage = KernelBufferUsage.NONE
        shape_and_dtype = list(shape_and_dtype)
        if isinstance(shape_and_dtype[-1], KernelBufferUsage):
            usage = shape_and_dtype.pop()
        physical_layout = None
        if isinstance(shape_and_dtype[-1], MemoryLayout):
            physical_layout = shape_and_dtype.pop()
        dtype = shape_and_dtype.pop()
        addressSpace = shape_and_dtype.pop()
        shape = tuple(shape_and_dtype)
        # allow shape to be provided as a tuple instead of as individual elements, to work around lack of unpacking in subscripts for Python 3.10
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        # Allow constant int expressions in shape
        shape = tuple(IndexExpr(s) if isinstance(s, int) else s for s in shape)
        if not all(isinstance(s, IndexExpr) for s in shape) or len(shape) == 0:
            raise TypeError(f"Expected shape to be a tuple of IndexExpr, got {shape}")
        if not isinstance(dtype, DataType):
            raise TypeError(f"Expected dtype to be a DataType, got {dtype}")
        if not (
            isinstance(addressSpace, IndexExpr)
            or isinstance(addressSpace, AddressSpace)
        ):
            raise TypeError(
                f"Expected addressSpace to be a AddressSpace, got {addressSpace}"
            )
        if addressSpace == AddressSpace.REGISTER:
            raise TypeError(
                f"Memory does not support address space register, use Register instead."
            )

        return cls.new_subtype(
            name="Memory",
            address_space=addressSpace,
            symbolic_shape=shape,
            dtype=dtype,
            physical_layout=physical_layout,
            usage=usage,
        )


class Register(metaclass=KernelBufferMeta):
    """
    Represents virtual registers. Parameterized by a shape and element type.
    Instantiating this class emits a new `register` operation.
    """

    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    rank: ClassVar[int]
    dtype: ClassVar[DataType]
    value: float

    def __new__(cls, value: float) -> "Register":
        from ..ops.wave_ops import register

        return register(cls.symbolic_shape, cls.dtype, value)

    def __class_getitem__(
        cls, shape_and_dtype: tuple[IndexExpr | DataType, ...]
    ) -> Type["Register"]:
        if len(shape_and_dtype) < 2:
            raise TypeError(f"Expected at least 2 arguments, got: {shape_and_dtype}")

        shape = shape_and_dtype[:-1]
        dtype = shape_and_dtype[-1]

        # Allow constant int expressions in shape
        shape = tuple(IndexExpr(s) if isinstance(s, int) else s for s in shape)

        if not isinstance(dtype, DataType):
            raise TypeError(f"Expected dtype to be a DataType, got {dtype}")

        return cls.new_subtype(
            name="Register",
            address_space=AddressSpace.REGISTER,
            symbolic_shape=shape,
            dtype=dtype,
        )


class SymbolBind:
    """
    Represents a binding between a symbol and a kernel argument.
    """

    dtype: DataType

    def __class_getitem__(cls, dt: DataType) -> Type["SymbolBind"]:
        class Subtype(cls):
            dtype = dt

        Subtype.__name__ = cls.__name__
        return Subtype


SymbolsMap: TypeAlias = dict[IndexSymbol, IndexExpr]


def _subs_expr(expr: Any, subs: Iterable[tuple[IndexExpr, IndexExpr]]) -> Any:
    if isinstance(expr, (Symbol, Expr)):
        return expr.subs(subs)

    return expr


def _is_identity_mapping(iters: Iterable[IndexSymbol], mapping: SymbolsMap) -> bool:
    if len(iters) != len(mapping):
        return False

    for it, val in zip(iters, mapping.values()):
        if it != val:
            return False

    return True


def _map_indices(
    mapping: SymbolsMap, symbols: Optional[tuple[IndexSymbol, ...]]
) -> tuple[IndexExpr, ...]:
    if symbols is None:
        return tuple(mapping.values())

    return tuple(mapping[sym] for sym in symbols)


class IndexMapping:
    """
    Represents a mapping between 2 sets of indices.
    """

    iters: dict[IndexSymbol, int]
    input_mapping: SymbolsMap
    output_mapping: SymbolsMap
    iteration_shape: tuple[IndexExpr, ...]
    dynamic_val_mappings: tuple[SymbolsMap, ...]
    dynamic_val_indices: dict[IndexSymbol, int]

    def __init__(
        self,
        num_iterators: int,
        inputs: SymbolsMap,
        outputs: SymbolsMap,
        dynamic_val_mappings: SymbolsMap | Sequence[SymbolsMap] = (),
    ) -> None:
        iters = {self.iterator(i): i for i in range(num_iterators)}
        iter_shape = [None] * num_iterators
        for sym, expr in chain(inputs.items(), outputs.items()):
            i = iters.get(expr, None)
            if i is None:
                continue

            current = iter_shape[i]
            assert (
                current is None or current == sym
            ), f"Iterator conflict: {current} and {sym}"
            iter_shape[i] = sym

        assert all(
            i is not None for i in iter_shape
        ), f"Cannot determine iteration domain: {iter_shape=}"
        self.iters = iters
        self.iteration_shape = iter_shape
        self.input_mapping = inputs
        self.output_mapping = outputs
        if not isinstance(dynamic_val_mappings, Sequence):
            dynamic_val_mappings = (
                (dynamic_val_mappings,) if dynamic_val_mappings else ()
            )

        self.dynamic_val_mappings = tuple(dynamic_val_mappings)
        num_dyn_vals = len(dynamic_val_mappings)
        self.dynamic_val_indices = {self.dynamic_val(i): i for i in range(num_dyn_vals)}

    @property
    def num_iterators(self) -> int:
        return len(self.iters)

    @property
    def num_dynamic_vals(self) -> int:
        return len(self.dynamic_val_indices)

    def substitute(self, subs: Iterable[tuple[IndexExpr, IndexExpr]]) -> Self:
        new_inputs = {
            key: _subs_expr(val, subs) for key, val in self.input_mapping.items()
        }
        new_outputs = {
            key: _subs_expr(val, subs) for key, val in self.output_mapping.items()
        }
        return IndexMapping(self.num_iterators, new_inputs, new_outputs)

    @property
    def input_shape(self) -> tuple[IndexExpr]:
        return tuple(self.input_mapping.keys())

    @property
    def output_shape(self) -> tuple[IndexExpr]:
        return tuple(self.output_mapping.keys())

    @staticmethod
    def iterator(index: int) -> IndexSymbol:
        return index_symbol(f"$index{index}")

    @staticmethod
    def dynamic_val(index: int) -> IndexSymbol:
        return index_symbol(f"$dynamic_val{index}")

    def map_input_indices(
        self, symbols: Optional[tuple[IndexSymbol, ...]] = None
    ) -> tuple[IndexExpr, ...]:
        return _map_indices(self.input_mapping, symbols)

    def map_output_indices(
        self, symbols: Optional[tuple[IndexSymbol, ...]] = None
    ) -> tuple[IndexExpr, ...]:
        return _map_indices(self.output_mapping, symbols)

    def is_input_identity(self) -> bool:
        return _is_identity_mapping(self.iters.keys(), self.input_mapping)

    def is_output_identity(self) -> bool:
        return _is_identity_mapping(self.iters.keys(), self.output_mapping)

    def is_identity(self) -> bool:
        return self.is_input_identity() and self.is_output_identity()

    def __repr__(self) -> str:
        return (
            f"IndexMapping(iters={self.iters}, input_mapping={self.input_mapping}), "
            f"output_mapping={self.output_mapping}, dynamic_val_mappings={self.dynamic_val_mappings}"
        )
