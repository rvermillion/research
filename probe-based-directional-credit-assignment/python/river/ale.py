#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from tensile.common import *
from tensile.nn import Module
from tensile.util.buffer import ArrayBuffer


class Struct(dict[str, Any]):

    __slots__ = ()

    def __bool__(self) -> bool:
        return True

    def get_array(self, name: str) -> Array:
        val = self.get(name)
        if ten.is_array(val):
            return val
        if isinstance(val, (bool, int, float)):
            return ten.array(val)
        raise ValueError(f"Telemetry value '{name}' is not an array")

    def get_scalar(self, name: str) -> int|float|bool:
        val = self.get(name)
        if ten.is_array(val):
            val = val.item()
        if isinstance(val, (int, float, bool)):
            return val
        raise ValueError(f"Telemetry value '{name}' is not a scalar")

    def get_bool(self, name: str) -> int:
        val = self.get(name)
        return bool(val)

    def get_int(self, name: str) -> int:
        val = self.get(name)
        if ten.is_array(val):
            val = val.item()
        if isinstance(val, int):
            return val
        raise ValueError(f"Telemetry value '{name}' is not an integer")

    def get_float(self, name: str) -> float:
        val = self.get(name)
        if ten.is_array(val):
            val = val.item()
        if isinstance(val, int):
            return float(val)
        if isinstance(val, float):
            return val
        raise ValueError(f"Telemetry value '{name}' is not a float")

    def update(self, data: Mapping[str, Any]):
        update_item = self.update_item
        for item, value in data.items():
            update_item(item, value, merge=False)

    def merge(self, data: Mapping[str, Any]):
        update_item = self.update_item
        for item, value in data.items():
            update_item(item, value, merge=True)

    def update_item(self, item: str, value: Any, merge: bool = False):
        self[item] = value

    def copy(self) -> Self:
        copy = self.__class__()
        dict.update(copy, self)
        return copy

    def describe(self) -> str:
        return ', '.join(f'{name}: {self[name]}' for name in self)

    def eval(self) -> None:
        for name, value in self.items():
            if ten.is_array(value):
                ten.eval(value)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.describe()})'

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        obj = cls()
        obj.update(data)
        return obj

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        obj = cls()
        obj.update(kwargs)
        return obj


class ControlInfo(Struct):

    __slots__ = ()


class Controllable(Object):

    __slots__ = ()

    def control(self, update: ControlInfo):
        pass


class Telemetry(Struct):

    __slots__ = ()

    def update_item(self, item: str, value: Any, merge: bool = False):
        if ten.is_array(value):
            value = ten.detach(value)
        self[item] = value


class TelemetryReceiver(Object):

    __slots__ = ()

    def listens_for_telemetry(self) -> list[str]:
        return []

    def receive_telemetry(self, telemetry: Telemetry, name: str = None):
        raise NotImplementedError()

    def receive_telemetry_item(self, item: str, data: Any):
        self.receive_telemetry(Telemetry.from_dict({item: data}))


class TelemetryRouter(TelemetryReceiver):

    __slots__ = ('telemetry_routes',)

    telemetry_routes: Annotated[dict[str, list[TelemetryReceiver]], field(
        default_factory=dict,
    )]

    def add_route(self, name: str, receiver: TelemetryReceiver) -> None:
        if receivers := self.telemetry_routes.get(name):
            receivers.append(receiver)
        else:
            self.telemetry_routes[name] = [receiver]

    def add_receiver(self, receiver: TelemetryReceiver) -> None:
        for name in receiver.listens_for_telemetry():
            if receivers := self.telemetry_routes.get(name):
                receivers.append(receiver)
            else:
                self.telemetry_routes[name] = [receiver]

    def receive_telemetry(self, telemetry: Telemetry, name: str = None):
        receivers = set()
        for name in telemetry:
            if route := self.telemetry_routes.get(name):
                receivers.update(route)

        for receiver in receivers:
            receiver.receive_telemetry(telemetry, name)

    def receive_telemetry_item(self, item: str, data: Any):
        if receivers := self.telemetry_routes.get(item):
            for receiver in receivers:
                receiver.receive_telemetry_item(item, data)


class ControllerState(TelemetryReceiver):

    __slots__ = ('controller', )

    controller: Annotated['ALEController', field(
        doc="The controller that owns this adapter",
        required=True,
    )]

    @property
    def name(self) -> str:
        return str(id(self))

    def receive_telemetry(self, telemetry: Telemetry, name: str = None):
        self.update(telemetry)

    def update(self, telemetry: Telemetry):
        raise NotImplementedError()

    def describe(self) -> str:
        return str(self)

    def _repr_args(self, **options) -> str:
        return self.name


class SingleArrayState(ControllerState):

    __slots__ = ('name', 'updater', )

    name: Annotated[str, field(
        doc="The name of the array to update from",
    )]
    updater: Annotated[Callable[[Array], Any], field(
        doc="The updater function for the controller",
    )]

    def postinit(self, spec: Spec):
        super().postinit(spec)
        self.updater = self._build_updater()

    def listens_for_telemetry(self) -> list[str]:
        return [self.name]

    def _build_updater(self) -> Callable:
        raise NotImplementedError()

    def update(self, telemetry: Telemetry):
        self.updater(telemetry.get_array(self.name))

    def _repr_args(self, **options) -> str:
        return self.name


@provides(ControllerState, 'ema')
class EMAState(SingleArrayState):

    __slots__ = ('decays', 'ema', 'var', 'max', 'min', 'last', 'buffer')

    decays: Annotated[Array, field(
        doc="The decay rates of the emas",
        default_factory=lambda: ten.array([0.999, 0.99, 0.9]),
    )]
    ema: Annotated[Array, field(
        doc="The EMA of the variable",
    )]
    var: Annotated[Array, field(
        doc="The variance of the variable",
    )]
    max: Annotated[Array, field(
        doc="The maximum value of the variable",
    )]
    min: Annotated[Array, field(
        doc="The minimum value of the variable",
    )]
    last: Annotated[Array, field(
        doc="The last value of the variable",
    )]
    buffer: Annotated[ArrayBuffer, field(
        doc="Buffer for storing intermediate values",
    )]

    def postinit(self, spec: Spec):
        decays = self.decays
        if decays.ndim != 1 or decays.shape[0] < 2:
            raise ValueError("Decay rates must be a 1D array with at least 2 elements")
        self.ema = ten.zeros_like(decays)
        self.var = ten.zeros_like(decays)
        self.max = ten.zeros((1,))
        self.min = ten.zeros((1,))
        self.last = ten.zeros((1,))
        self.buffer = ArrayBuffer()
        super().postinit(spec)

    def _build_updater(self) -> Callable:
        decays = self.decays
        one_minus_decays = 1 - decays

        def update(val: Array):
            ema = self.ema
            delta = val - ema

            self.last = val
            self.ema = decays * ema + one_minus_decays * val
            self.var = decays * self.var + one_minus_decays * ten.square(delta)
            self.max = ten.maximum(self.max, val)
            self.min = ten.minimum(self.min, val)
            self.buffer.append(val[None])

        return update

    @property
    def fast(self) -> Array:
        return self.ema[-1]

    @property
    def slow(self) -> Array:
        return self.ema[0]

    @property
    def delta(self) -> Array:
        ema = self.ema
        return ema[-1] - ema[0]

    @property
    def percent_delta(self) -> Array:
        ema = self.ema
        slow = ema[0]
        return (ema[-1] - slow) / slow

    @property
    def values(self) -> Array:
        return self.buffer.fetch()

    def describe(self) -> str:
        return f'ema={self.ema}, std={ten.sqrt(self.var)}'

    def _repr_args(self, **options) -> str:
        return self.name + ', ' + self.describe()


class Adapter(Object):

    __slots__ = ('controller', )

    controller: Annotated['ALEController', field(
        doc="The controller that owns this adapter",
        required=True,
    )]

    @property
    def name(self) -> str:
        return str(id(self))

    def adapt(self, step: int) -> None:
        if not self.skip_step(step):
            self._adapt(step)

    def skip_step(self, step: int) -> bool:
        return False

    def _adapt(self, step: int) -> None:
        raise NotImplementedError('Adapter class should be subclassed and implement the adapt method')

    def _repr_args(self, **options) -> str:
        return self.name


class ControllableValue(Object):

    __slots__ = ()

    @property
    def name(self) -> str:
        return str(id(self))

    def get_value(self) -> float:
        raise NotImplementedError()

    def set_value(self, value: float) -> None:
        raise NotImplementedError()



class ValueAdapter(Adapter):

    __slots__ = ('value', 'max_value', 'min_value')

    value: Annotated[ControllableValue, field(
        doc="The controllable value to adapt",
        required=True,
    )]
    max_value: Annotated[float, field(
        doc="The max value",
        default=1.0,
    )]
    min_value: Annotated[float, field(
        doc="The min value",
        default=0.01,
    )]

    @property
    def name(self) -> str:
        return self.value.name

    def get_value(self) -> float:
        return self.value.get_value()

    def set_value(self, value: float) -> None:
        if self.min_value <= value <= self.max_value:
            self.value.set_value(value)


@provides(Adapter, 'decay')
class DecayAdapter(ValueAdapter):

    __slots__ = ('decay', 'decay_every')

    decay: Annotated[float, field(
        doc="The decay rate for the value",
        default=1.0,
    )]
    decay_every: Annotated[int, field(
        doc="How often to decay the value, in steps",
        default=1000,
    )]

    def _adapt(self, step: int) -> None:
        if step % self.decay_every == 0:
            ov = self.get_value()
            nv = ov * self.decay
            print(f'  Decayed [{self.name}]: {ov} -> {nv}')
            self.set_value(nv)


@provides(Adapter, 'ema-state')
class EMAStateAdapter(ValueAdapter):

    __slots__ = ('state', 'threshold', 'delta_threshold', 'multplier',
                 )

    state: Annotated[EMAState, field(
        doc="The EMA state of the value",
        required=True,
    )]
    threshold: Annotated[float, field(
        doc="The threshold for loss to trigger adaptation",
        default=1.0,
    )]
    delta_threshold: Annotated[float, field(
        doc="The threshold for loss delta to trigger adaptation",
        default=0.05,
    )]
    multplier: Annotated[float, field(
        doc="The multiplier for the exploration radius",
        default=1.1,
    )]

    def _coerce_state(self, spec: Any) -> ControllerState:
        if isinstance(spec, ControllerState):
            return spec
        if isinstance(spec, str):
            return self.controller.states[spec]
        raise ValueError("Cannot coerce state to a ControllerState")

    def skip_step(self, step: int) -> bool:
        return bool(self.state.last < self.threshold)

    def increase_value(self):
        ov = self.get_value()
        if ov < self.max_value:
            nv = min(ov * self.multplier, self.max_value)
            print(f'  Increased [{self.name}]: {ov} -> {nv}')
            self.set_value(nv)

    def decrease_value(self):
        ov = self.get_value()
        if ov > self.min_value:
            nv = max(ov / self.multplier, self.min_value)
            print(f'  Decreased [{self.name}]: {ov} -> {nv}')
            self.set_value(nv)

    def _adapt(self, step: int) -> None:
        state = self.state
        pct_delta = state.percent_delta

        # print(f'. Adapting [{self.name}] at step {step}: delta: {pct_delta:.4f} {state.describe()}')
        if pct_delta > self.delta_threshold:
            self.increase_value()


class ALEController(TelemetryRouter):

    __slots__ = ('model', 'loss_fn', 'states', 'adapters', 'buffered_telemetry')

    model: Annotated[Module, field(
        doc="The model to train",
        required=True,
    )]
    loss_fn: Annotated[Callable[[Array, Array], Array], field(
        doc="The loss function to evaluate perturbation effectiveness",
    )]
    states: Annotated[dict[str, ControllerState], field(
        doc="The states that the controller tracks",
        default_factory=dict,
    )]
    adapters: Annotated[dict[str, Adapter], field(
        doc="The adapters to apply during training",
        default_factory=dict,
    )]
    buffered_telemetry: Annotated[Telemetry, field(
        doc="Buffered telemetry for delayed processing",
        default_factory=Telemetry,
    )]


