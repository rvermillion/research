#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from tensile.common import *
from tensile.infra.transform import Transforms
from tensile.nn import Module
from tensile.util.buffer import ArrayBuffer
from tensile.util.metric import Metric
from tensile.util.state import StateAware


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
    def construct(cls, **kwargs: Any) -> Self:
        obj = cls()
        obj.update(kwargs)
        return obj


class ControlInfo(Struct):

    __slots__ = ()


class Controllable(StateAware):

    __slots__ = ()

    @property
    def controllable_values(self) -> dict[str, 'ControllableValue']:
        return {}

    def control(self, update: ControlInfo):
        pass


class ALEComponent(Controllable):

    __slots__ = ('controllable_values',)

    controllable_values: Annotated[dict[str, 'ControllableValue'], field(
        doc='Dictionary of controllable values, keyed by name',
    )]

    def _lazy_controllable_values(self) -> dict[str, 'ControllableValue']:
        return {}

    def initialize(self, controller: 'ALEController'):
        pass


class Telemetry(Struct):

    __slots__ = ('step', )

    step: int

    def update_item(self, item: str, value: Any, merge: bool = False):
        if ten.is_array(value):
            value = ten.detach(value)
        self[item] = value

    @classmethod
    def construct(cls, step: int = -1, /, **kwargs: Any) -> Self:
        telemetry = cls()
        telemetry.step = step
        telemetry.update(kwargs)
        return telemetry

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], step: int = -1) -> Self:
        telemetry = cls()
        telemetry.step = step
        telemetry.update(data)
        return telemetry


class TelemetryReceiver(Object):

    __slots__ = ()

    def listens_for_telemetry(self) -> tuple[str, ...]:
        return ()

    def receive_telemetry(self, telemetry: Telemetry, name: str = None):
        raise NotImplementedError()

    # def receive_telemetry_item(self, item: str, data: Any):
    #     self.receive_telemetry(Telemetry.from_dict({item: data}))


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

    # def receive_telemetry_item(self, item: str, data: Any):
    #     if receivers := self.telemetry_routes.get(item):
    #         for receiver in receivers:
    #             receiver.receive_telemetry_item(item, data)


class LearningState(TelemetryReceiver, StateAware):

    __slots__ = ('controller', 'record')

    controller: Annotated['ALEController', field(
        doc="The controller that owns this adapter",
        required=True,
    )]
    record: Annotated[bool, field(
        doc="Whether to record telemetry for this state",
        default=True,
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

    def get_metrics(self, name: str = None, step: int = 0) -> Sequence[Metric]:
        return ()

    def _repr_args(self, **options) -> str:
        return self.name


SingleArrayUpdater = Callable[[int, Array], Any]


class SingleArrayState(LearningState):

    __slots__ = ('name', 'item', 'updater', 'ndim')

    name: Annotated[str, field(
        doc="The name of the state",
    )]
    item: Annotated[str, field(
        doc="The name of the telemetry item to update from",
    )]
    updater: Annotated[SingleArrayUpdater, field(
        doc="The updater function for the controller",
    )]
    ndim: Annotated[int, field(
        doc="Number of dimensions of the variable",
        default=0
    )]

    def _lazy_item(self) -> str:
        return self.name

    def postinit(self, spec: Spec):
        super().postinit(spec)
        self.updater = self._build_updater()

    def listens_for_telemetry(self) -> tuple[str, ...]:
        return self.item,

    def _build_updater(self) -> SingleArrayUpdater:
        raise NotImplementedError()

    def update(self, telemetry: Telemetry):
        self.updater(telemetry.step, telemetry.get_array(self.item))

    def _repr_args(self, **options) -> str:
        return self.name


eps = 1e-6
slow = 0
fast = -1


def default_decays() -> Array:
    return ten.array([0.999, 0.99, 0.9])


@provides(LearningState, 'ema')
class EMAState(SingleArrayState):

    __slots__ = ('decays', 'ema', 'var', 'max', 'min', 'last',
                 'buffer', 'ema_buffer', 'var_buffer', '_initialized')

    decays: Annotated[Array, field(
        doc="The decay rates of the emas",
        default_factory=default_decays,
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
    ema_buffer: Annotated[ArrayBuffer, field(
        doc="Buffer for storing ema values",
    )]
    var_buffer: Annotated[ArrayBuffer, field(
        doc="Buffer for storing var values",
    )]
    _initialized: Annotated[bool, field(
        doc="Whether this has been initialized or not",
        default=False
    )]

    def _coerce_decays(self, spec: Any) -> Array:
        if spec is None: return default_decays()
        if isinstance(spec, list):
            return ten.array(spec, dtype=ten.float32)
        if isinstance(spec, float):
            return ten.array([spec, 0.], dtype=ten.float32)
        raise ValueError(f'Cannot coerce to Array: {spec!r}')

    def postinit(self, spec: Spec):
        decays = self.decays
        if decays.ndim != 1 or decays.shape[0] < 2:
            raise ValueError("Decay rates must be a 1D array with at least 2 elements")
        if self.ndim > 0:
            for _ in range(self.ndim):
                decays = ten.expand_dims(decays, -1)
            self.decays = decays
        self.ema = ten.zeros_like(decays)
        self.var = ten.zeros_like(decays)
        # self.max = ten.full(decays.shape, -ten.inf)
        # self.min = ten.full(decays.shape, ten.inf)
        single = 1,
        self.max = ten.full(single, -ten.inf)
        self.min = ten.full(single, ten.inf)
        self.last = ten.zeros(single)
        self.buffer = ArrayBuffer()
        self.ema_buffer = ArrayBuffer()
        self.var_buffer = ArrayBuffer()
        super().postinit(spec)

    record_all = False

    def _build_updater(self) -> SingleArrayUpdater:
        decays = self.decays
        one_minus_decays = 1 - decays

        buffer = self.buffer
        ema_buffer = self.ema_buffer
        var_buffer = self.var_buffer

        if self.record:
            if self.record_all:
                def append(val: Array):
                    buffer.append(val[None])
                    ema_buffer.append(self.ema[None])
                    var_buffer.append(self.var[None])
            else:
                def append(val: Array):
                    buffer.append(val[None])
        else:
            def append(val: Array):
                pass

        def update(step: int, val: Array):
            ema = self.ema

            self.last = val

            if self._initialized:
                delta = val - ema
                self.ema = decays * ema + one_minus_decays * val
                self.var = decays * self.var + one_minus_decays * ten.square(delta)
            else:
                self.ema = ema + val
                self._initialized = True

            self.max = ten.maximum(self.max, val)
            self.min = ten.minimum(self.min, val)
            append(val)

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
        return ema[fast] - ema[slow]

    @property
    def percent_delta(self) -> Array:
        ema = self.ema
        slow_ema = ema[slow]
        return (ema[fast] - slow_ema) / (ten.abs(slow_ema) + eps)

    @property
    def progress(self) -> Array:
        ema = self.ema
        return ema[slow] - ema[fast]

    @property
    def relative_progress(self) -> Array:
        ema = self.ema
        slow_ema = ema[slow]
        return (slow_ema - ema[fast]) / (ten.abs(slow_ema) + eps)

    @property
    def standardized_progress(self) -> Array:
        ema = self.ema
        var = self.var
        return (ema[slow] - ema[fast]) / (ten.sqrt(var[slow]) + eps)

    @property
    def relative_noise(self) -> Array:
        ema = self.ema
        var = self.var
        return ten.sqrt(var[fast]) / (ten.abs(ema[fast]) + eps)

    @property
    def values(self) -> Array:
        return self.buffer.fetch()

    def describe(self) -> str:
        return f'ema={self.ema}, std={ten.sqrt(self.var)}'

    def get_metrics(self, name: str = None, step: int = 0) -> Sequence[Metric]:
        metrics: list[Metric] = []
        if self.record and self.ndim == 0:
            if name is None: name = self.name
            metrics.append(Metric.from_buffer(name, self.buffer))
            if self.record_all:
                ema = self.ema_buffer.fetch()
                var = self.var_buffer.fetch()
                ema_slow = ema[:, slow] + 0.
                ema_fast = ema[:, fast] + 0.
                var_slow = var[:, slow] + 0.
                var_fast = var[:, fast] + 0.
                progress = (ema_slow - ema_fast) / (ten.abs(ema_slow) + eps)
                std_progress = (ema_slow - ema_fast) / (ten.sqrt(var_slow) + eps)
                metrics += [
                    Metric.from_array(f'{name}.slow', ema_slow),
                    Metric.from_array(f'{name}.fast', ema_fast),
                    Metric.from_array(f'{name}.var_slow', var_slow),
                    Metric.from_array(f'{name}.var_fast', var_fast),
                    Metric.from_array(f'{name}.progress', progress),
                    Metric.from_array(f'{name}.std_progress', std_progress),
                ]
        return metrics

    def _add_state(self, state: dict[str, Any]):
        super()._add_state(state)
        if self.ndim == 0:
            state['last'] = self.last.item()
            state['max'] = self.max.item()
            state['min'] = self.min.item()

    def _repr_args(self, **options) -> str:
        return self.name + ', ' + self.describe()


@provides(LearningState, 'first-step')
class FirstStepState(SingleArrayState):

    __slots__ = ('step', 'duration', 'trigger')

    step: Annotated[int, field(
        doc="The first step where the trigger was true for duration",
        default=-1,
    )]
    duration: Annotated[int, field(
        doc="The duration the trigger must be true",
        default=1,
    )]
    trigger: Annotated[Predicate[Array], field(
        doc="The trigger to watch for",
        coerce=True,
    )]

    def _build_updater(self) -> SingleArrayUpdater:
        trigger = self.trigger
        duration = self.duration

        def update(step: int, val: Array):
            if self.step < 0:
                if trigger(val):
                    self.step = step
            elif step < self.step + duration:
                if not trigger(val):
                    self.step = -1

        return update

    def _add_state(self, state: dict[str, Any]):
        super()._add_state(state)
        state.update(
            step=self.step,
            duration=self.duration,
        )


@provides(LearningState, 'accumulate')
class AccumulateState(SingleArrayState):

    __slots__ = ('count', 'sum', 'sum_squares', 'max', 'min', 'last')

    count: Annotated[Array, field(
        doc="The count of values accumulated",
    )]
    sum: Annotated[Array, field(
        doc="The EMA of the variable",
    )]
    sum_squares: Annotated[Array, field(
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

    def postinit(self, spec: Spec):
        self.clear()
        super().postinit(spec)

    def clear(self):
        self.count = ten.zeros((1,))
        self.sum = ten.zeros((1,))
        self.sum_squares = ten.zeros((1,))
        self.max = ten.full((1,), -ten.inf)
        self.min = ten.full((1,), ten.inf)
        self.last = ten.zeros((1,))

    def _build_updater(self) -> SingleArrayUpdater:
        def update(step: int, val: Array):
            self.count += 1
            self.sum = self.sum + val
            self.sum_squares = self.sum_squares + ten.square(val)
            self.last = val
            self.max = ten.maximum(self.max, val)
            self.min = ten.minimum(self.min, val)

        return update

    @property
    def mean(self) -> Array:
        return self.sum/self.count

    @property
    def var(self) -> Array:
        count = self.count
        mean = self.sum/count
        return self.sum_squares/count - mean*mean

    @property
    def std(self) -> Array:
        return ten.sqrt(self.var)

    def describe(self) -> str:
        return f'count={self.count}, sum={self.sum}'

    def get_metrics(self, name: str = None, step: int = 0) -> Sequence[Metric]:
        return []

    def _repr_args(self, **options) -> str:
        return self.name + ', ' + self.describe()


class LearningAdapter(Object):

    __slots__ = ('controller', 'warmup',)

    controller: Annotated['ALEController', field(
        doc="The controller that owns this adapter",
        required=True,
    )]
    warmup: Annotated[int, field(
        doc="The number of steps to skip before adapting",
        default=0,
    )]

    @property
    def name(self) -> str:
        return str(id(self))

    def adapt(self, step: int) -> None:
        if not self.skip_step(step):
            self._adapt(step)

    def skip_step(self, step: int) -> bool:
        return step < self.warmup

    def _adapt(self, step: int) -> None:
        raise NotImplementedError('Adapter class should be subclassed and implement the adapt method')

    def _repr_args(self, **options) -> str:
        return self.name


Change = tuple[int, float]

class ControllableValue(StateAware):

    __slots__ = ('changes', 'record')

    changes: Annotated[list[Change], field(default_factory=list)]
    record: Annotated[bool, field(default=True)]

    def postinit(self, spec: Spec):
        super().postinit(spec)

        if self.record:
            self.changes.append((0, self.get_value()))

    @property
    def name(self) -> str:
        return str(id(self))

    def get_value(self) -> float:
        raise NotImplementedError()

    def set_value(self, value: float) -> None:
        raise NotImplementedError()

    def change_value(self, step: int, value: float) -> None:
        if self.record:
            self.changes.append((step, value))
        self.set_value(value)

    def get_metrics(self, name: str = None, step: int = 0) -> Sequence[Metric]:
        if self.record:
            return [Metric.from_changes(name or self.name, self.changes, last_step=step)]
        return []

    def _add_state(self, state: dict[str, Any]):
        super()._add_state(state)
        if changes := self.changes:
            state['changes'] = [
                [s for s, v in changes],
                [v for s, v in changes],
            ]

    def _repr_args(self, **options) -> str:
        return self.name

    @classmethod
    def from_attr(cls, controllable: object, attr: str, name: str = None) -> 'ControllableValue':
        return AttributeControllableValue(controllable=controllable, attr=attr, name=name)


@provides(ControllableValue, 'attr')
class AttributeControllableValue(ControllableValue):

    __slots__ = ('controllable', 'attr', 'name')

    controllable: Annotated[object, field(
        doc="The object to control",
        required=True,
    )]
    attr: Annotated[str, field(
        doc="The attr to control on the object",
        required=True,
    )]
    name: Annotated[str, field(
        doc="The attr to control on the object",
    )]

    def _lazy_name(self) -> str:
        return self.attr

    def get_value(self) -> float:
        return getattr(self.controllable, self.attr)

    def set_value(self, value: float) -> None:
        setattr(self.controllable, self.attr, value)

    def _repr_args(self, **options) -> str:
        return f'{self.name} of {self.controllable}'


class ValueAdapter(LearningAdapter):

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

    def _coerce_value(self, spec: Any) -> ControllableValue:
        if isinstance(spec, ControllableValue):
            return spec
        if isinstance(spec, str):
            value = self.controller.controllable_values[spec]
            if isinstance(value, ControllableValue):
                return value
        raise ValueError(f"Cannot coerce state to a ControllableValue: {spec!r}")

    @property
    def name(self) -> str:
        return self.value.name

    def get_value(self) -> float:
        return self.value.get_value()

    def set_value(self, step: int, value: float) -> None:
        if self.min_value <= value <= self.max_value:
            self.value.change_value(step, value)


@provides(LearningAdapter, 'decay')
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
            self.set_value(step, nv)


@provides(LearningAdapter, 'ema-driven')
class EMADrivenAdapter(ValueAdapter):

    __slots__ = ('state', 'threshold',
                 'expand_trigger', 'expand_factor',
                 'shrink_trigger', 'shrink_factor')

    state: Annotated[EMAState, field(
        doc="The EMA state of the value",
        required=True,
    )]
    threshold: Annotated[float, field(
        doc="The threshold for value to trigger adaptation",
        default=1.0,
    )]
    expand_trigger: Annotated[Predicate[EMAState], field(
        doc="The predicate for triggering expansion",
    )]
    expand_factor: Annotated[float, field(
        doc="The expansion factor for the value",
        default=1.02,
    )]
    shrink_trigger: Annotated[Predicate[EMAState], field(
        doc="The predicate for triggering shrinking",
    )]
    shrink_factor: Annotated[float, field(
        doc="The shrinking factor for the value",
        default=0.98,
    )]

    def _coerce_expand_trigger(self, spec: Any) -> Predicate[EMAState]:
        return predicates.coerce(spec)

    def _coerce_shrink_trigger(self, spec: Any) -> Predicate[EMAState]:
        return predicates.coerce(spec)

    def postinit(self, spec: Spec):
        super().postinit(spec)
        if not self.expand_trigger:
            expand_threshold = spec.get('expand_threshold')
            if isinstance(expand_threshold, float):
                @predicates.function
                def gt_threshold(state: EMAState) -> bool:
                    # noinspection PyTypeChecker
                    return bool(state.percent_delta > expand_threshold)
                self.expand_trigger = gt_threshold
            else:
                self.expand_trigger = predicates.never
        if not self.shrink_trigger:
            shrink_threshold = spec.get('shrink_threshold')
            if isinstance(shrink_threshold, float):
                @predicates.function
                def lt_threshold(state: EMAState) -> bool:
                    # noinspection PyTypeChecker
                    return bool(state.percent_delta < shrink_threshold)
                self.shrink_trigger = lt_threshold
            else:
                self.shrink_trigger = predicates.never

    def _coerce_state(self, spec: Any) -> EMAState:
        if isinstance(spec, EMAState):
            return spec
        if isinstance(spec, str):
            state = self.controller.states[spec]
            if isinstance(state, EMAState):
                return state
        raise ValueError(f"Cannot coerce state to a EMAState: {spec!r}")

    # def skip_step(self, step: int) -> bool:
    #     return bool(self.state.last < self.threshold)

    def increase_value(self, step: int) -> None:
        ov = self.get_value()
        if ov < self.max_value:
            nv = min(ov * self.expand_factor, self.max_value)
            print(f'  Increased [{self.name}]: {ov} -> {nv}')
            self.set_value(step, nv)

    def decrease_value(self, step: int):
        ov = self.get_value()
        if ov > self.min_value:
            nv = max(ov * self.shrink_factor, self.min_value)
            print(f'  Decreased [{self.name}]: {ov} -> {nv}')
            self.set_value(step, nv)

    def _adapt(self, step: int) -> None:
        state = self.state

        if self.expand_trigger(state):
            self.increase_value(step)
        elif self.shrink_trigger(state):
            self.decrease_value(step)


def standardized_progress_predicate(progress_pred: PredicateLike[Array]) -> Predicate[EMAState]:
    progress_pred = predicates.coerce(progress_pred)
    @predicates.function
    def pred(state: EMAState) -> bool:
        return progress_pred(state.standardized_progress)
    return pred

predicates.register('ema_standardized_progress', standardized_progress_predicate)


def slow_predicate(slow_pred: PredicateLike[Array]) -> Predicate[EMAState]:
    return predicates.transform(Transforms.get_attr('slow'), slow_pred)
    # slow_pred = predicates.coerce(slow_pred)
    # @predicates.function
    # def pred(state: EMAState) -> bool:
    #     return slow_pred(state.slow)
    # return pred

predicates.register('ema_slow', slow_predicate)


def plateau_predicate(threshold: float) -> Predicate[EMAState]:
    def plateau(state: EMAState) -> bool:
        return ten.abs(state.standardized_progress).item() < threshold
    return predicates.function(plateau, f'plateau[{threshold}]')


predicates.register('ema_plateau', plateau_predicate)


class ALEController(TelemetryRouter, Controllable):

    __slots__ = ('model', 'loss_fn', 'states', 'adapters', 'components', 'controllable_values', 'buffered_telemetry')

    model: Annotated[Module, field(
        doc="The model to train",
        required=True,
    )]
    loss_fn: Annotated[Callable[[Array, Array], Array], field(
        doc="The loss function to evaluate perturbation effectiveness",
    )]
    states: Annotated[dict[str, LearningState], field(
        doc="The states that the controller tracks",
        default_factory=dict,
    )]
    adapters: Annotated[dict[str, LearningAdapter], field(
        doc="The adapters to apply during training",
        default_factory=dict,
    )]
    components: Annotated[dict[str, ALEComponent], field(
        doc="The components of this controller",
        default_factory=dict,
    )]
    controllable_values: Annotated[dict[str, ControllableValue], field(
        doc="The controllable values to apply during training",
        default_factory=dict,
    )]
    buffered_telemetry: Annotated[Telemetry, field(
        doc="Buffered telemetry for delayed processing",
        default_factory=Telemetry,
    )]

    def _lazy_states(self) -> dict[str, LearningState]:
        return {}

    def _coerce_states(self, spec: Any) -> dict[str, LearningState]:
        states = self._lazy_states()
        if spec is None:
            pass
        elif isinstance(spec, Sequence):
            for s in spec:
                state = LearningState.coerce(s, controller=self)
                states[state.name] = state
        elif isinstance(spec, Mapping):
            for k, s in spec.items():
                state = LearningState.coerce(s, controller=self, name=k)
                states[k] = state
        else:
            raise ValueError(f"Invalid spec type: {type(spec)}")
        return states

    def _coerce_adapters(self, spec: Any) -> dict[str, LearningAdapter]:
        return {}

    def postinit(self, spec: Spec):
        super().postinit(spec)

        if not self.components:
            self.components = self.build_components()

        values = self.controllable_values

        for cname, controllable in self.components.items():
            for vname, value in controllable.controllable_values.items():
                values[cname + '.' + vname] = value

        for state in self.states.values():
            self.add_receiver(state)

    def build_components(self) -> dict[str, ALEComponent]:
        return {}

    def get_metrics(self, step: int = 0) -> Sequence[Metric]:
        metrics = []
        for name, state in self.states.items():
            metrics.extend(state.get_metrics(name, step))
        for name, value in self.controllable_values.items():
            metrics.extend(value.get_metrics(name, step))
        return metrics

    def build_step(self) -> Callable[[int, Array, Array], Array]:
        raise NotImplementedError()

    def _add_state(self, state: dict[str, Any]):
        super()._add_state(state)

        components = {}
        for name, component in self.components.items():
            if s := component.state_dict():
                components[name] = s
        if components:
            state['components'] = components

        states = {}
        for name, ls in self.states.items():
            if s := ls.state_dict():
                states[name] = s
        if states:
            state['states'] = states

        values = {}
        for name, value in self.controllable_values.items():
            if s := value.state_dict():
                values[name] = s
        if values:
            state['controllable_values'] = values


