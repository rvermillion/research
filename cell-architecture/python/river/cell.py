#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from enum import Enum
from tensile.common import *
from tensile.infra import RootObject
from tensile.infra.util import noop
from tensile.nn import CompiledModule, Module, ModuleArgs
from tensile.nn.attention.kv import KVBuffer


class Slot(RootObject):

    __slots__ = ('tick', 'value', 'precision')

    tick: int
    value: Array
    precision: Array

    def __init__(self, tick: int, value: Array, precision: Array):
        self.tick = tick
        self.value = value
        self.precision = precision

    def __iter__(self):
        return iter((self.value, self.precision))

    def _repr_args(self, **options) -> str:
        return f'tick={self.tick}, value={self.value.shape}, precision={self.precision.shape}'


class EdgeType(Enum):

    state = 0
    predict = 1
    error = 2

    def reverse(self) -> 'EdgeType':
        if self == EdgeType.state:
            return self
        elif self == EdgeType.predict:
            return EdgeType.error
        elif self == EdgeType.error:
            return EdgeType.predict
        else:
            raise ValueError(f'Invalid edge type: {self}')

    def is_state(self) -> bool:
        return self == EdgeType.state

    def is_predict(self) -> bool:
        return self == EdgeType.predict

    def is_error(self) -> bool:
        return self == EdgeType.error


class CellType(Enum):

    internal = 0
    sensory = 1
    motor = 2
    top = 3

    def is_top(self) -> bool:
        return self == CellType.top

    def is_internal(self) -> bool:
        return self == CellType.internal

    def is_sensory(self) -> bool:
        return self == CellType.sensory

    def is_motor(self) -> bool:
        return self == CellType.motor

    def is_bottom(self) -> bool:
        return self == CellType.sensory or self == CellType.motor

    def is_predict(self) -> bool:
        return self == CellType.internal or self == CellType.top


initial_state_precision: dict[CellType, float] = {
    CellType.internal: 0.5,
    CellType.sensory: 1.,
    CellType.motor: 1.,
    CellType.top: 0.5
}

initial_precision = 0.5


def expand_batch(x: Array, batch_size: int) -> Array:
    B = x.shape[0]
    if B == batch_size:
        return x
    if B == 1:
        return ten.broadcast_to(x, shape=(batch_size,) + x.shape[1:])
    raise ValueError(f'Cannot expand batch from {B} to {batch_size}')


class Edge(Object):

    __slots__ = ('cortex', 'index', 'source', 'target', 'edge_type', 'dim', 'lag', 'mod', 'reverse',
                 'slots')

    cortex: Annotated['Cortex', field(
        doc='The cortex that owns this edge',
        required=True,
    )]
    index: Annotated[int, field(
        doc='The index of this edge within the cortex',
        required=True,
    )]
    source: Annotated['Cell', field(
        doc='The source cell of this edge',
        required=True,
    )]
    target: Annotated['Cell', field(
        doc='The target cell of this edge',
        required=True,
    )]
    edge_type: Annotated[EdgeType, field(
        doc='The type of this edge',
        required=True,
    )]
    dim: Annotated[int, field(
        doc='The dimension of this edge',
        required=True,
    )]
    lag: Annotated[int, field(
        doc='The lag of this edge',
        default=1,
    )]
    mod: Annotated[int, field(
        doc='The mod of this edge for the slots array',
    )]
    slots: Annotated[list[Slot], field(
        doc='The slots of this edge',
    )]
    reverse: Annotated['Edge', field(
        doc='The reverse of this edge',
    )]

    def postinit(self, spec: Spec):
        super().postinit(spec)

        source = self.source
        target = self.target
        edge_type = self.edge_type

        if edge_type.is_state():
            self.dim = source.dim
        elif edge_type.is_error():
            self.dim = source.dim
        elif edge_type.is_predict():
            self.dim = target.dim
        else:
            raise ValueError(f'Invalid edge type: {self.edge_type}')

        if self not in source.outgoing_edges:
            source.add_outgoing_edge(self)

        if self not in target.incoming_edges:
            target.add_incoming_edge(self)

        mod = self.lag + 1

        value = ten.zeros((1, self.dim), dtype=self.dtype)

        # Precision is in [0, 1] and is 1 - uncertainty
        # We start off with perfect precision
        if edge_type.is_state():
            precision = initial_state_precision[source.cell_type] * ten.ones((1, 1), dtype=self.dtype)
        else:
            precision = initial_precision * ten.ones((1, 1), dtype=self.dtype)

        ten.eval(value, precision)

        slots = []
        for i in range(mod):
            slot = Slot(i, value, precision)
            slots.append(slot)

        self.slots = slots
        self.mod = mod

    def _lazy_reverse(self) -> 'Edge':
        if self.edge_type == EdgeType.state:
            return self
        else:
            rev_type = self.edge_type.reverse()

            for rev in self.cortex.edges:
                if rev.edge_type == rev_type and rev.source == self.target and rev.target == self.source and rev.lag == self.lag:
                    return rev

            raise ValueError(f'Could not find a reverse edge for {self}')

    @property
    def dtype(self) -> DType:
        return self.cortex.dtype

    @property
    def value(self) -> Array:
        return self.get_value(self.cortex.tick)

    @property
    def precision(self) -> Array:
        return self.get_precision(self.cortex.tick)

    def get_slot(self, tick: int) -> Slot:
        return self.slots[tick % self.mod]

    def get_value(self, tick: int) -> Array:
        return self.slots[tick % self.mod].value

    def get_precision(self, tick: int) -> Array:
        return self.slots[tick % self.mod].precision

    def set_slot(self, tick: int, value: Array, precision: Array = None):
        slot = self.slots[tick % self.mod]
        slot.tick = tick
        slot.value = value
        if precision is not None:
            slot.precision = precision

    def set_batch_size(self, batch_size: int):
        for slot in self.slots:
            slot.value = expand_batch(slot.value, batch_size)
            slot.precision = expand_batch(slot.precision, batch_size)

    def _repr_args(self, **options) -> str:
        return f'{self.index}: {self.source.index} -> {self.target.index}, dim={self.dim}'

    def _repr_type(self, **options) -> str:
        return str.capitalize(self.edge_type.name) + 'Edge'


class MemoryLayer(Object):

    __slots__ = ('size',)

    size: Annotated[int, field(
        doc='The size of the memory layer',
    )]


class Memory(Object):

    __slots__ = ('cell', 'size', 'q_heads', 'kv_heads', 'head_dim', 'v_head_dim', 'kvs')

    cell: Annotated['Cell', field(
        doc='The cell that owns this memory',
        required=True
    )]
    size: Annotated[int, field(
        doc='The size of the memory',
        required=True,
    )]
    q_heads: Annotated[int, field(
        doc='The nuber of query heads in the memory',
        required=True,
    )]
    kv_heads: Annotated[int, field(
        doc='The number of kv heads in the memory',
        required=True,
    )]
    head_dim: Annotated[int, field(
        doc='The query and key head dim of the memory',
        required=True,
    )]
    v_head_dim: Annotated[int, field(
        doc='The value head dim of the memory',
        required=True,
    )]
    kvs: Annotated[KVBuffer, field(
        doc='The list of keys',
    )]

    def postinit(self, spec: Spec):
        super().postinit(spec)
        self.kvs = KVBuffer(0)

    def append(self, key: Array, value: Array):
        B = key.shape[0]
        assert B == value.shape[0], f'Key and value batch sizes must match, got {B} and {value.shape[0]}'
        key = key.reshape(B, 1, self.kv_heads, 1, self.head_dim)
        value = value.reshape(B, 1, self.kv_heads, 1, self.v_head_dim)
        ten.eval(key, value)
        self.kvs.update(key, value)
        self.kvs.eval()

    def attend(self, query: Array) -> Array:
        B = query.shape[0]
        n_heads = self.q_heads // self.kv_heads
        query = query.reshape(B, n_heads, self.kv_heads, 1, self.head_dim)
        keys, values = self.kvs.fetch_kv()
        ten.eval(query, keys, values)
        scores = ten.matmul(query, ten.swapaxes(keys, -1, -2))
        weights = ten.softmax(scores, axis=-1)
        values = ten.matmul(weights, values)
        out = values.reshape(B, -1)
        ten.eval(out)
        return out

    def _repr_args(self, **options) -> str:
        return f'cell={self.cell.index}, size={self.size}'


class Cell(CompiledModule):

    __slots__ = ('cortex', 'index', 'dim', 'period', 'initial_state', 'next_tick', 'state_edge',
                 'incoming_edges', 'outgoing_edges')

    cortex: Annotated['Cortex', field(
        doc='The cortex that owns this cell',
        required=True,
        tree=False,
    )]
    index: Annotated[int, field(
        doc='The index of this cell',
        required=True,
    )]
    dim: Annotated[int, field(
        doc="The dimension of this cell's state",
        required=True,
    )]
    period: Annotated[int, field(
        doc='The period of this cell',
        default=1,
    )]
    initial_state: Annotated[Array, field(
        doc='The current state vector of this cell',
        parameter=True,
    )]
    next_tick: Annotated[int, field(
        doc='The next tick at which this cell will be updated',
        default=0,
    )]
    state_edge: Annotated[Edge, field(
        doc='The self edge that connects this cell to itself',
    )]
    incoming_edges: Annotated[list[Edge], field(
        doc='The incoming edges from this cell',
        default_factory=list,
    )]
    outgoing_edges: Annotated[list[Edge], field(
        doc='The outgoing edges from this cell',
        default_factory=list,
    )]
    cell_type: ClassVar[Annotated[CellType, field(
        doc='The type of this cell',
    )]]

    def postinit(self, spec: Spec):
        super().postinit(spec)

        self.initial_state = ten.as_type(ten.random.normal(loc=0., scale=0.1, shape=(1, self.dim,)), self.dtype)
        ten.eval(self.initial_state)

    @property
    def dtype(self) -> DType:
        return self.cortex.dtype

    @property
    def state(self) -> Array:
        return self.get_state(self.cortex.tick)

    def get_state(self, tick: int) -> Array:
        return self.state_edge.get_value(tick)

    def get_precision(self, tick: int) -> Array:
        return self.state_edge.get_precision(tick)

    def set_state(self, tick: int, state: Array, precision: Array = None):
        self.state_edge.set_slot(tick, state, precision)

    @property
    def incoming_prediction_edges(self) -> list[Edge]:
        return [edge for edge in self.incoming_edges if edge.edge_type == EdgeType.predict]

    @property
    def incoming_error_edges(self) -> list[Edge]:
        return [edge for edge in self.incoming_edges if edge.edge_type == EdgeType.error]

    @property
    def outgoing_prediction_edges(self) -> list[Edge]:
        return [edge for edge in self.outgoing_edges if edge.edge_type == EdgeType.predict]

    @property
    def outgoing_error_edges(self) -> list[Edge]:
        return [edge for edge in self.outgoing_edges if edge.edge_type == EdgeType.error]

    @property
    def error_edges(self) -> list[Edge]:
        return self.incoming_error_edges + self.outgoing_error_edges

    @property
    def in_dim(self) -> int:
        in_dim = 0
        for edge in self.incoming_edges:
            in_dim += edge.dim
        return in_dim

    def build_graph(self) -> None:
        self.compile()

    def initialize(self, tick: int) -> None:
        self.state_edge.set_slot(tick, self.initial_state)

    def add_incoming_edge(self, edge: Edge):
        self.incoming_edges.append(edge)

    def add_outgoing_edge(self, edge: Edge):
        self.outgoing_edges.append(edge)

    def build_call(self, mode: Module.Mode, **options) -> Callable[[int], Array]:
        raise NotImplementedError(self)

    if TYPE_CHECKING:
        # noinspection PyFinal
        def __call__(self, tick: int) -> Array: ...

    def _extra_structure(self, **options) -> str:
        return f'#{self.index}, dim={self.dim}'

    @classmethod
    def construct(cls, cell_type: CellType, **kwargs) -> 'Cell':
        impl = cls.implementations[cell_type]
        return impl(**kwargs)

    implementations: ClassVar[dict[CellType, type['Cell']]] = {}

    def __init_subclass__(cls, log: int | str = None, **kwargs):
        super().__init_subclass__(log, **kwargs)
        if 'cell_type' in cls.__dict__:
            cls.implementations[cls.cell_type] = cls


class BottomCell(Cell):

    __slots__ = ()

    @property
    def incoming_error_edges(self) -> list[Edge]:
        return []

    @property
    def outgoing_prediction_edges(self) -> list[Edge]:
        return []

    def add_incoming_edge(self, edge: Edge):
        if edge.edge_type.is_error():
            raise ValueError(f'Error edges cannot be incoming for bottom cells')
        super().add_incoming_edge(edge)

    def add_outgoing_edge(self, edge: Edge):
        if edge.edge_type.is_predict():
            raise ValueError(f'Prediction edges cannot be outgoing for bottom cells')
        super().add_outgoing_edge(edge)


class SensoryCell(BottomCell):

    __slots__ = ()

    cell_type = CellType.sensory

    def build_call(self, mode: Module.Mode, **options) -> Callable[[int], Array]:
        error_edges = self.outgoing_error_edges

        state_edge = self.state_edge

        def error_fn(pred: Array, state: Array) -> Array:
            return pred - state

        def call(tick: int) -> Array:
            state, precision = state_edge.get_slot(tick)

            next_tick = tick + 1

            for edge in error_edges:
                pred = edge.reverse.get_slot(tick)
                err = error_fn(pred.value, state)
                edge.set_slot(next_tick, err, precision * pred.precision)

            return state

        return call


class MotorCell(BottomCell):

    __slots__ = ()

    cell_type = CellType.motor

    def build_call(self, mode: Module.Mode, **options) -> Callable[[int], Array]:
        prediction_edges = self.incoming_prediction_edges
        error_edges = self.outgoing_error_edges

        if len(prediction_edges) == 1:
            control_edge = prediction_edges[0]
        else:
            return noop

        state_edge = self.state_edge

        def error_fn(pred: Array, state: Array) -> Array:
            return pred - state

        def constrain(pred: Array, *others: Array) -> Array:
            return pred


        def call(tick: int) -> Array:
            control = control_edge.get_slot(tick)

            state = constrain(control.value)

            next_tick = tick + 1

            for edge in error_edges:
                pred = edge.reverse.get_slot(tick)
                err = error_fn(pred.value, state)
                edge.set_slot(next_tick, err, pred.precision)

            state_edge.set_slot(next_tick, state, control.precision)

            return state

        return call


class CellAttention(CompiledModule):

    __slots__ = ('q_proj', 'k_proj', 'v_proj', 'o_proj')




class PredictCell(Cell):

    __slots__ = ('working_memory', 'q_proj', 'k_projs', 'v_projs', 'o_proj', 'mlp', 'prediction_heads')

    working_memory: Annotated[Memory, field(
        doc='The working memory of this cell',
    )]
    q_proj: Annotated[Module, field(
        doc='The query projection for this cell',
    )]
    k_projs: Annotated[list[Module], field(
        doc='The key projections for this cell',
        default_factory=list,
    )]
    v_projs: Annotated[list[Module], field(
        doc='The value projections for this cell',
        default_factory=list,
    )]
    o_proj: Annotated[Module, field(
        doc='The output projection for this cell',
    )]
    mlp: Annotated[Module, field(
        doc='The attention mlp of this cell',
    )]
    prediction_heads: Annotated[list[Module], field(
        doc='The prediction heads of this cell',
        default_factory=list,
    )]

    def postinit(self, spec: Spec):
        super().postinit(spec)

        q_heads = 8
        kv_heads = 4
        head_dim = 32
        v_head_dim = self.dim // q_heads

        self.working_memory = Memory(cell=self, size=256,
                                     q_heads=q_heads, kv_heads=kv_heads,
                                     head_dim=head_dim, v_head_dim=v_head_dim)

    def build_graph(self) -> None:

        memory = self.working_memory

        q_heads = memory.q_heads
        kv_heads = memory.kv_heads
        head_dim = memory.head_dim
        v_head_dim = memory.v_head_dim

        k_projs = self.k_projs
        v_projs = self.v_projs

        in_dim = 0
        for edge in self.incoming_edges:
            e_dim = edge.dim
            k_projs.append(Module.from_args(
                in_dim=e_dim,
                out_dim=kv_heads * head_dim,
                bias=False,
                kind='linear',
            ))
            v_projs.append(Module.from_args(
                in_dim=e_dim,
                out_dim=kv_heads * v_head_dim,
                bias=False,
                kind='linear',
            ))
            in_dim += e_dim

        self.q_proj = Module.from_args(
            in_dim=in_dim,
            out_dim=q_heads * head_dim,
            bias=False,
            kind='linear',
        )
        self.o_proj = Module.from_args(
            in_dim=q_heads * v_head_dim,
            out_dim=self.dim,
            bias=False,
            kind='linear',
        )
        self.mlp = Module.from_args(
            in_dim=self.dim,
            hidden_dim=2*self.dim,
            out_dim=self.dim,
            bias=False,
            kind='mlp.glu',
        )

        heads = self.prediction_heads

        for edge in self.outgoing_prediction_edges:
            head = self.init_prediction_head(edge)
            heads.append(head)

        super().build_graph()

    def initialize(self, tick: int) -> None:
        super().initialize(tick)

        state = self.state_edge.get_value(tick)

        for head, edge in zip(self.prediction_heads, self.outgoing_prediction_edges):
            p = head(state)
            edge.set_slot(tick, p)

    def init_prediction_head(self, edge: Edge):
        return Module.from_args(
            in_dim=self.dim,
            out_dim=edge.dim,
            bias=False,
            kind='linear',
        )

    def build_call(self, mode: Module.Mode, **options) -> Callable[[int], Array]:
        incoming_edges = self.incoming_edges

        q_proj = self.q_proj
        k_projs = self.k_projs
        v_projs = self.v_projs
        o_proj = self.o_proj
        mlp = self.mlp

        memory = self.working_memory

        prediction_heads = self.prediction_heads
        prediction_edges = self.outgoing_prediction_edges
        in_error_edges = self.incoming_error_edges
        out_error_edges = self.outgoing_error_edges

        state_edge = self.state_edge

        def error_fn(pred: Array, state: Array) -> Array:
            return pred - state

        default_beta = ten.array(0.9, dtype=self.cortex.dtype)

        def beta_fn(h: Array) -> Array:
            return default_beta


        def precision_weighted_error(tick: int, edge: Edge) -> Array:
            error, precision = edge.get_slot(tick)
            return precision * ten.norm(error, axis=-1, keepdims=True)

        dtype = self.dtype

        precision_decay = ten.array(0.9, dtype=dtype)
        one_minus_precision_decay = 1.0 - precision_decay

        precision_scale = ten.array(1.0, dtype=dtype)
        precision_bias = ten.array(0.0, dtype=dtype)

        outgoing_error_weight = ten.array(0.5, dtype=dtype)

        def call(tick: int) -> Array:
            inputs = []
            for edge, k_proj, v_proj in zip(incoming_edges, k_projs, v_projs):
                x = edge.get_value(tick)
                inputs.append(x)
                memory.append(k_proj(x), v_proj(x))

            state, precision = state_edge.get_slot(tick)

            x = ten.concatenate(inputs, axis=-1)

            h = memory.attend(q_proj(x))

            o = o_proj(h)

            r = mlp(o)

            beta = beta_fn(x)

            state = beta * state + (1 - beta) * self.initial_state + r

            next_tick = tick + 1

            # TODO: calculate new precision from moving average of old precision and norm of incoming and outgoing error edges
            weighted_error = ten.array(0., dtype=dtype)
            for edge in in_error_edges:
                weighted_error = weighted_error + precision_weighted_error(tick, edge)

            for edge in out_error_edges:
                weighted_error = weighted_error + outgoing_error_weight * precision_weighted_error(tick, edge)

            new_precision = ten.sigmoid(-precision_scale * weighted_error + precision_bias)
            new_precision = precision_decay * precision + one_minus_precision_decay * new_precision

            for head, edge in zip(prediction_heads, prediction_edges):
                p = head(state)
                edge.set_slot(next_tick, p, new_precision)

            for edge in out_error_edges:
                pred = edge.reverse.get_slot(tick)
                err = error_fn(pred.value, state)
                edge.set_slot(next_tick, err, precision * pred.precision)

            state_edge.set_slot(next_tick, state, new_precision)

            return state

        return call


class InternalCell(PredictCell):

    __slots__ = ()

    cell_type = CellType.internal


class TopCell(PredictCell):

    __slots__ = ()

    cell_type = CellType.top

    @property
    def incoming_prediction_edges(self) -> list[Edge]:
        return []

    @property
    def outgoing_error_edges(self) -> list[Edge]:
        return []

    def add_incoming_edge(self, edge: Edge):
        if edge.edge_type.is_predict():
            raise ValueError(f'Predict edges cannot be incoming for top cells')
        super().add_incoming_edge(edge)

    def add_outgoing_edge(self, edge: Edge):
        if edge.edge_type.is_error():
            raise ValueError(f'Error edges cannot be outgoing for top cells')
        super().add_outgoing_edge(edge)



class Cortex(CompiledModule):

    __slots__ = ('cells', 'edges', 'dtype', 'tick')

    cells: Annotated[list[Cell], field(
        doc='The list of cells in the cortex',
        default_factory=list,
    )]
    edges: Annotated[list[Edge], field(
        doc='The list of edges in the cortex',
        default_factory=list,
    )]
    dtype: Annotated[DType, field(
        doc='The data type used for computations in the cortex',
        default=ten.float32,
    )]
    tick: Annotated[int, field(
        doc='The current tick of the cortex',
        default=0,
    )]

    @property
    def sensory_cells(self) -> tuple[Cell, ...]:
        return tuple(cell for cell in self.cells if cell.cell_type is CellType.sensory)

    @property
    def motor_cells(self) -> tuple[Cell, ...]:
        return tuple(cell for cell in self.cells if cell.cell_type is CellType.motor)

    @property
    def internal_cells(self) -> tuple[Cell, ...]:
        return tuple(cell for cell in self.cells if cell.cell_type is CellType.internal)

    @property
    def top_cells(self) -> tuple[Cell, ...]:
        return tuple(cell for cell in self.cells if cell.cell_type is CellType.top)

    @property
    def error_edges(self) -> tuple[Edge, ...]:
        return tuple(edge for edge in self.edges if edge.edge_type.is_error())

    def add_cell(self, cell_type: CellType, **kwargs) -> 'Cell':
        cell = Cell.construct(cell_type, cortex=self, index=len(self.cells), **kwargs)
        self.cells.append(cell)

        cell.state_edge = self.add_edge(source=cell, target=cell, edge_type=EdgeType.state)

        return cell

    def add_edge(self, **kwargs) -> 'Edge':
        edge = Edge(cortex=self, index=len(self.edges), **kwargs)
        self.edges.append(edge)
        return edge

    def add_prediction(self, source: int, target: int, lag: int = 1):
        source_cell = self.cells[source]
        target_cell = self.cells[target]

        self.add_edge(source=source_cell, target=target_cell, lag=lag, edge_type=EdgeType.predict)
        self.add_edge(source=target_cell, target=source_cell, lag=lag, edge_type=EdgeType.error)

    def build(self) -> None:
        for cell in self.cells:
            cell.build_graph()
        self.compile()

    def initialize(self, batch_size: int) -> None:
        tick = self.tick
        for cell in self.cells:
            cell.initialize(tick)
        for edge in self.edges:
            edge.set_batch_size(batch_size)

    def build_call(self, mode: Module.Mode, **options) -> Callable:
        cells = self.cells

        sensory_cells = self.sensory_cells
        motor_cells = self.motor_cells

        error_edges = self.error_edges

        def call(*senses: Array|None) -> tuple[Array, ...]:
            tick = self.tick

            if len(senses) != len(sensory_cells):
                raise ValueError(f'Expected {len(sensory_cells)} senses, got {len(senses)}')

            for sense, cell in zip(senses, sensory_cells):
                if sense is not None:
                    cell.set_state(tick, sense)

            for cell in cells:
                if cell.next_tick <= tick:
                    cell(tick)
                    cell.next_tick = tick + cell.period

            next_tick = tick + 1

            # for edge in error_edges:
            #     error, precision = edge.get_slot(next_tick)
            #     print(next_tick, edge, 'error:', error, 'precision:', precision)

            out = tuple(cell.get_state(next_tick) for cell in motor_cells)

            self.tick = next_tick

            return out

        return call

    def _extra_structure(self) -> str:
        return f'{len(self.cells)} cells, {len(self.edges)} edges'


def main():
    cortex = Cortex.from_args()

    cortex.add_cell(CellType.sensory, dim=32)
    cortex.add_cell(CellType.motor, dim=16)
    cortex.add_cell(CellType.internal, dim=64)
    cortex.add_cell(CellType.top, dim=24)

    cortex.add_prediction(2, 0)
    cortex.add_prediction(2, 1)
    cortex.add_prediction(3, 2)

    cortex.build()

    B = 2

    cortex.initialize(B)

    cortex.set_mode(Module.Mode.debug)
    print(cortex.structure())

    x = ten.arange(B*32, dtype=cortex.dtype) / (B*32.)
    x = x.reshape(B, -1)
    ten.eval(x)

    for i in range(5):
        y, = cortex(x)

        for edge in cortex.edges:
            print(i, edge, edge.value, edge.precision)

        print(f'out[{i}]:', y)

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
