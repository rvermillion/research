#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.


from tensile.common import *
from tensile.infra import RootObject
from tensile.nn import CompiledModule, Module, ModuleArgs
from tensile.nn.common import Activation
from tensile.nn.layers import Linear
from tensile.nn.module import Functional
from tensile.experiment import Params

from river.ale import (ALEComponent, LearningAdapter, ALEController,
                       ControllableValue, LearningState, Telemetry)

#
# if ten.ten_kind == "torch":
#     ten.set_default_device("mps")



T = TypeVar('T')


def xnormalize(x: Array) -> Array:
    return x / ten.norm(x)


class Credit(RootObject):

    __slots__ = ('probes', 'credit', 'advantages')

    probes: Annotated[Array, field(
        doc="Probe index for each module (M, K)",
    )]
    credit: Annotated[Array, field(
        doc="Credit values for each module (M, K)",
    )]
    advantages: Annotated[Array, field(
        doc="The advantages for each branch (P,)",
    )]

    def __init__(self, probes: Array, credit: Array, advantages: Array):
        self.probes = probes
        self.credit = credit
        self.advantages = advantages
        assert probes.shape == credit.shape, 'Probes and credit must have the same shape'

    @property
    def num_modules(self) -> int:
        return self.probes.shape[0]

    @property
    def num_probes(self) -> int:
        return self.probes.shape[1]

    def append(self, probes: Array, credit: Array, advantages: Array, inplace: bool = True) -> 'Credit':
        assert probes.shape == credit.shape, 'Probes and credit must have the same shape'
        assert probes.shape[0] == self.probes.shape[0], 'Probes and credit must have the same number of modules'

        new_probes = ten.concat([self.probes, probes], axis=1)
        new_credit = ten.concat([self.credit, credit], axis=1)
        new_advantages = self.advantages + advantages
        if inplace:
            self.probes = new_probes
            self.credit = new_credit
            self.advantages = new_advantages
            return self
        return Credit(new_probes, new_credit, new_advantages)

    def __add__(self, other: 'Credit') -> 'Credit':
        return self.append(other.probes, other.credit, other.advantages, inplace=False)

    def __iadd__(self, other: 'Credit') -> 'Credit':
        return self.append(other.probes, other.credit, other.advantages, inplace=True)

    def get_probes(self, m: int) -> Array:
        return self.probes[m]

    def get_credit(self, m: int) -> tuple[Array, Array]:
        return self.probes[m], self.credit[m]

    def spread_credit(self, m: int, k: int) -> Array:
        spread = ten.zeros((k, ))
        probes, credit = self.get_credit(m)
        for i in range(probes.shape[0]):
            spread[probes[i]] += credit[i]
        return spread

    def _repr_args(self, **options) -> str:
        return f'probes={self.probes}, credit={self.credit}'

    @classmethod
    def combine(cls, creds: list['Credit']) -> 'Credit':
        if len(creds) == 1:
            return creds[0]
        if creds:
            probes = ten.concat([c.probes for c in creds], axis=1)
            credit = ten.concat([c.credit for c in creds], axis=1)
            advantages = ten.sum(ten.stack([c.advantages for c in creds], axis=0), axis=0)
            return Credit(probes, credit, advantages)
        raise ValueError("Cannot combine empty list of credits")


class BranchFunction(Protocol):

    def __call__(self, x: Array) -> Array: ...


class ProbeBank(Object):

    __slots__ = ('k', 'in_dim', 'out_dim', 'dtype')

    k: Annotated[int, field(
        doc="The number of probe directions"
    )]
    in_dim: Annotated[int, field(
        doc="The dimension of the input"
    )]
    out_dim: Annotated[int, field(
        doc="The dimension of the output"
    )]
    dtype: Annotated[DType, field(
        doc="The dtype to use for the probes",
        default=ten.float32,
    )]

    def build_branch_function(self, assignment: Array, scales: Array = None) -> BranchFunction:
        raise NotImplementedError()

    def update_module(self, module: 'ProbedLinear', probes: Array, coefficients: Array) -> None:
        raise NotImplementedError()

    def _repr_args(self, **options) -> str:
        return f'{self.in_dim} -> {self.out_dim}, k={self.k}'


@provides(ProbeBank, 'lora')
class LoRAProbeBank(ProbeBank):

    __slots__ = ('rank', 'left', 'right', 'bias')

    rank: Annotated[int, field(
        doc="The rank of the adapters",
        default=1,
    )]
    left: Annotated[Array, field(
        doc="The left matrices"
    )]
    right: Annotated[Array, field(
        doc="The right matrices"
    )]
    bias: Annotated[Optional[Array], field(
        doc="The bias probes if any"
    )]

    def build_branch_function(self, assignment: Array, scales: Array = None) -> BranchFunction:
        left = self.left[assignment]
        right = self.right[assignment]

        if self.bias is None:
            bias = None
        else:
            bias = ten.expand_dims(self.bias[assignment], axis=1)

        if scales is None:
            scale = ten.ones(assignment.shape, dtype=self.dtype)
        else:
            assert scales.shape == assignment.shape, "Shape mismatch"
            scale = scales

        num_branches = assignment.shape[0]

        # ten.eval(left, right)

        if bias is None:
            def branch(x: Array) -> Array:
                # ten.eval(x)
                if x.ndim < 3:
                    x = ten.expand_dims(x, -2)
                    squeeze = True
                else:
                    squeeze = False
                assert x.shape[0] == num_branches, f"Expected {num_branches} branches, got {x.shape[0]}"

                p = ten.matmul(x, left)
                y = ten.matmul(p, right)

                if squeeze:
                    out = ten.squeeze(y, axis=-2)
                    out = scale[:, None] * out
                else:
                    out = y
                    out = scale[:, None, None] * out
                # ten.eval(out)
                return out
        else:
            def branch(x: Array) -> Array:
                # ten.eval(x)
                if x.ndim < 3:
                    x = ten.expand_dims(x, -2)
                    squeeze = True
                else:
                    squeeze = False
                assert x.shape[0] == num_branches, f"Expected {num_branches} branches, got {x.shape[0]}"

                p = ten.matmul(x, left)
                y = ten.matmul(p, right) + bias

                if squeeze:
                    out = ten.squeeze(y, axis=-2)
                    out = scale[:, None] * out
                else:
                    out = scale[:, None, None] * y
                # ten.eval(out)
                return out

        return branch

    def update_module(self, module: 'ProbedLinear', probes: Array, coefficients: Array) -> None:

        # Multiply out the specified probes
        probe_weights = ten.matmul(self.left[probes], self.right[probes])

        # Multiply by the credit array (expanded to broadcast correctly) and sum
        weight_update = ten.sum(probe_weights * coefficients[:, None, None], axis=0)

        # Module weights are stored transposed, so swap axes
        weight_update = ten.swapaxes(weight_update, 0, 1)

        # Update the module weights
        module.weight += weight_update

        bias = self.bias
        if bias is not None:
            # Get the specified bias probes
            probe_bias = bias[probes]

            # Multiply by the credit array (expanded to broadcast correctly) and sum
            bias_update = ten.sum(probe_bias * coefficients[:, None], axis=0)

            # Update the module bias
            module.bias += bias_update

    def _repr_args(self, **options) -> str:
        return super()._repr_args(**options) + f', rank={self.rank}'


class ProbeableModule(CompiledModule):

    __slots__ = ()

    @property
    def dtype(self) -> DType:
        raise NotImplementedError()

    @property
    def probes(self) -> ProbeBank:
        raise NotImplementedError()

    # @probes.setter
    # def probes(self, probes: ProbeBank) -> None:
    #     raise NotImplementedError()

    def update_probes(self, probes: ProbeBank) -> None:
        raise NotImplementedError()

    def update_assignments(self, assignments: 'ModuleAssignments') -> None:
        raise NotImplementedError()

    def update_in_directions(self, probes: Array, coefficients: Array) -> None:
        raise NotImplementedError()


class ProbeManager(ALEComponent):

    __slots__ = ('num_probes', 'orthogonal')

    num_probes: Annotated[int, field(
        doc="The number of probes",
        default=10,
    )]
    orthogonal: Annotated[bool, field(
        doc="Whether to use orthogonal probes",
        default=False,
    )]

    def generate(self, controller: 'ModuleController', keep: Array = None) -> ProbeBank:
        raise NotImplementedError()

    def _repr_arg_items(self, **options) -> Iterable[Any]:
        yield f'num_probes={self.num_probes}'
        if self.orthogonal:
            yield '+orthogonal'


@provides(ProbeManager, 'lora')
class LoRAProbeManager(ProbeManager):

    __slots__ = ('rank', 'bias_p')

    rank: Annotated[int, field(
        doc="The rank of the adapters",
        default=1,
    )]
    bias_p: Annotated[float, field(
        doc="The probability of adding bias to the adapter",
        default=0.7,
    )]

    def generate(self, controller: 'ModuleController', keep: Array = None) -> ProbeBank:
        module = controller.module
        k = controller.num_probes
        r = self.rank
        n = module.in_dim
        m = module.out_dim

        dtype = module.dtype

        nk = k if keep is None else k - keep.shape[0]

        left = ten.as_type(ten.random.normal(shape=(nk+1, n, r)), dtype)
        right = ten.as_type(ten.random.normal(shape=(nk+1, r, m)), dtype)
        left = left / ten.norm(left, axis=1, keepdims=True)
        right = right / ten.norm(right, axis=2, keepdims=True)

        left[0] = 0.
        right[0] = 0.

        if controller.needs_bias:
            bias_p = self.bias_p
            bias = ten.as_type(ten.random.normal(shape=(nk+1, m)), dtype)
            bias = bias / ten.norm(bias, axis=1, keepdims=True)
            if bias_p < 1.0:
                bias = ten.random.bernoulli(bias_p, shape=(nk+1, m)) * bias
            bias[0] = 0.
        else:
            bias = None

        if k != nk:
            old_probes = module.probes
            if isinstance(old_probes, LoRAProbeBank):
                left = ten.concat([left, old_probes.left[keep]], axis=0)
                right = ten.concat([right, old_probes.right[keep]], axis=0)
                if bias is not None:
                    bias = ten.concat([bias, old_probes.bias[keep]], axis=0)

        return ProbeBank.coerce(k=k, in_dim=n, out_dim=m, rank=r, left=left, right=right, bias=bias, kind='lora')

    def _repr_arg_items(self, **options) -> Iterable[Any]:
        yield from super()._repr_arg_items(**options)
        yield f'rank={self.rank}'
        yield f'bias_p={self.bias_p}'


@provides(Linear, 'probed')
class ProbedLinear(Linear, ProbeableModule):

    __slots__ = ('probes', 'branch', 'null_branch')

    weight: Annotated[Array, field(
        doc="The learnable weights of the layer.",
        parameter=False,
    )]
    bias: Annotated[Optional[Array], field(
        doc="The learnable bias of the layer.",
        parameter=False,
    )]
    probes: Annotated[ProbeBank, field(
        doc='The probe bank for the module'
    )]
    branch: Annotated[BranchFunction, field(
        doc="The probe function for this module",
    )]
    null_branch: Annotated[BranchFunction, field(
        doc="The null probe function for this module",
    )]

    def _lazy_null_branch(self) -> BranchFunction:
        null = ten.zeros((self.out_dim, ))

        # noinspection PyUnusedLocal
        def null_probe(x: Array) -> Array:
            return null
        return null_probe

    @property
    def dtype(self) -> DType:
        return self.weight.dtype

    def init_from_args(self, args: Linear.Args):
        super().init_from_args(args)
        ten.eval(self.weight)

    def update_probes(self, probes: ProbeBank) -> None:
        self.probes = probes

    def update_assignments(self, assignments: 'ModuleAssignments'):
        self.branch = self.probes.build_branch_function(assignments.probes, scales=assignments.scales)

    def update_in_directions(self, probes: Array, coefficients: Array) -> None:
        self.probes.update_module(self, probes, coefficients)

    # def update_from_credit(self, m: int, credit: 'Credit') -> None:
    #     self.probes.update_module(m, self, credit)

    def build_call(self, mode: Module.Mode, **options) -> Functional:
        if self.bias is None:
            def call(x: Array) -> Array:
                y = ten.matmul(x, self.weight.T)
                out = y + self.branch(x)
                return out
        else:
            def call(x: Array) -> Array:
                y = ten.matmul(x, self.weight.T) + self.bias
                out = y + self.branch(x)
                return out

        return call

    Args = Linear.Args


class ProbesAndScales(RootObject):

    __slots__ = ('probes', 'scales')

    probes: Annotated[Array, field(
        doc="The current probe indexes"
    )]
    scales: Annotated[Array, field(
        doc="The scales for the current probes"
    )]

    def __init__(self, probes: Array, scales: Array):
        super().__init__()
        self.probes = probes
        self.scales = scales

        assert probes.shape == scales.shape, "Probes and scales must have the same shape"


class ModuleAssignment(ProbesAndScales):

    __slots__ = ()


class ModuleAssignments(ProbesAndScales):

    __slots__ = ()

    def concatenate(self, probes: Array, scales: Array, in_place: bool = True) -> 'ModuleAssignments':
        assert probes.shape == scales.shape, "Probes and scales must have the same shape"

        probes = ten.concatenate([self.probes, probes])
        scales = ten.concatenate([self.scales, scales])
        if in_place:
            self.probes = probes
            self.scales = scales
            return self
        else:
            return ModuleAssignments(probes, scales)

    @property
    def num_branches(self) -> int:
        return self.probes.shape[0]


class Branch(ProbesAndScales):

    __slots__ = ()

    @property
    def num_modules(self) -> int:
        return self.probes.shape[0]


class BranchSchedule(ProbesAndScales):

    __slots__ = ()

    @property
    def num_branches(self) -> int:
        return self.probes.shape[1]

    @property
    def num_modules(self) -> int:
        return self.probes.shape[0]

    def get_module_assignments(self, mod: int) -> ModuleAssignments:
        assert 0 <= mod < self.num_modules, "Module index out of range"
        return ModuleAssignments(self.probes[mod], self.scales[mod])

    def get_branch(self, branch: int) -> Branch:
        assert 0 <= branch < self.num_branches, "Branch index out of range"
        return Branch(self.probes[:, branch], self.scales[:, branch])

    def get_probe(self, mod: int, branch: int) -> Array:
        return self.probes[mod, branch]

    def get_scale(self, mod: int, branch: int) -> Array:
        return self.scales[mod, branch]

    def get_module_probes(self, mod: int) -> Array: # (P, )
        return self.probes[mod]

    def get_branch_probes(self, branch: int) -> Array: # (M, )
        return self.probes[:, branch]

    def get_module_scales(self, mod: int) -> Array:  # (P, )
        return self.scales[mod]

    def get_branch_scales(self, branch: int) -> Array: # (M, )
        return self.scales[:, branch]

    def get_probes_and_scales(self, index: Array) -> tuple[Array, Array]:
        return self.probes[:, index], self.scales[:, index]

    def _repr_args(self, **options) -> str:
        return f"{self.num_modules} x {self.num_branches}"

    @classmethod
    def from_assignments(cls, branches: list[ModuleAssignments]) -> 'BranchSchedule':
        probes = ten.stack([p.probes for p in branches], axis=0)
        scales = ten.stack([p.scales for p in branches], axis=0)
        return cls(probes, scales)


class BranchScheduler(ALEComponent):

    __slots__ = ('current', 'history', 'force_null_branch',
                 'schedule_count')

    current: Annotated[BranchSchedule, field(
        doc="The current branch matrix"
    )]
    history: Annotated[list[BranchSchedule], field(
        doc="The history of branch matrices",
        default_factory=list,
    )]
    force_null_branch: Annotated[bool, field(
        doc="Whether to force the first branch to have null probes assigned",
        default=True
    )]
    schedule_count: Annotated[int, field(
        doc="The number of times that schedule has been called",
        default=0,
    )]

    def schedule_called(self) -> int:
        self.schedule_count += 1
        return self.schedule_count

    def schedule(self, controller: 'PALEController', num_branches: int, keep: Array = None) -> BranchSchedule:
        self.schedule_called()
        new_schedules = []
        if keep is None:
            for m in range(controller.num_modules):
                module_schedule = self.generate_module_assignments(controller, m, num_branches)
                new_schedules.append(module_schedule)
        else:
            schedule = self.current
            old_probes = schedule.probes
            old_scales = schedule.scales
            num_branches -= keep.shape[0]
            for m in range(controller.num_modules):
                module_schedule = self.generate_module_assignments(controller, m, num_branches)
                module_schedule.concatenate(old_probes[m, keep], old_scales[m, keep], in_place=True)
                new_schedules.append(module_schedule)

        schedule = BranchSchedule.from_assignments(new_schedules)
        self.current = schedule
        self.history.append(schedule)
        return schedule

    def generate_module_assignments(self, controller: 'PALEController', m: int, num_branches: int) -> ModuleAssignments:
        raise NotImplementedError()

    # def schedule_module(self, module: ProbeableModule, num_branches: int) -> tuple[Array, Array]:
    #     raise NotImplementedError()


@provides(BranchScheduler, 'random')
class RandomBranchScheduler(BranchScheduler):

    __slots__ = ('exploration_radius', 'p', 'mu', 'sigma', 'min_factor', 'max_factor')

    exploration_radius: Annotated[float, field(
        doc="The exploration radius of the probe scales",
        default=1.0
    )]
    # exploration_radius_decay: Annotated[float, field(
    #     doc="The exploration radius decay rate",
    #     default=1.0
    # )]
    # decay_every: Annotated[int, field(
    #     doc="How often to decay the exploration radius",
    #     default=0
    # )]
    mu: Annotated[float, field(
        doc="The mu of the scale log normal distribution",
        default=0.0
    )]
    sigma: Annotated[float, field(
        doc="The sigma of the scale log normal distribution",
        default=1.0
    )]
    min_factor: Annotated[float, field(
        doc="The minimum factor of the exploration radius",
        default=0.05
    )]
    max_factor: Annotated[float, field(
        doc="The maximum factor of the exploration radius",
        default=1.5
    )]
    p: Annotated[Optional[float], field(
        doc="The probability that the module will be perturbed in a given branch",
    )]
    probe_probs: Annotated[list[Array], field(
        doc="The probability that the module will be perturbed in a given branch",
        init=False,
    )]

    def _lazy_controllable_values(self) -> dict[str, 'ControllableValue']:
        values = [
            ControllableValue.from_attr(self, 'exploration_radius'),
        ]
        return {value.name: value for value in values}

    def modify_exploration_radius(self, nv: float, ov: float = None):
        if ov is None: ov = self.exploration_radius
        if nv < ov:
            self.exploration_radius = nv
            # print(f'. Decreased exploration radius: {ov} -> {nv}')
        elif nv > ov:
            self.exploration_radius = nv
            # print(f'. Increased exploration radius: {ov} -> {nv}')

    # def schedule_called(self) -> int:
    #     cnt = super().schedule_called()
    #     if (every := self.decay_every) and cnt % every == 0:
    #         self.modify_exploration_radius(self.exploration_radius * self.exploration_radius_decay)
    #     return cnt

    def generate_module_assignments(self, controller: 'PALEController', m: int, num_branches: int) -> ModuleAssignments:
        k_m = controller.get_num_probes(m)
        p = self.p
        if p is None:
            probes = ten.random.randint(low=0, high=k_m+1, shape=(num_branches,))
        else:
            # Decide whether the module is perturbed in each branch
            perturbed = ten.random.bernoulli(p, shape=(num_branches,))
            # Randomly assign probe indexes between 1 and k_m for the ones that are perturbed
            probes = perturbed * ten.random.randint(low=1, high=k_m+1, shape=(num_branches,))

        # Make sure the null branch is always present
        if self.force_null_branch:
            probes[0] = 0

        # Scale by +/- magnitude
        scales = self.exploration_radius * ten.exp(self.mu + self.sigma * ten.random.normal(shape=(num_branches,)))

        # Clip the scales to be between min and max
        scales = ten.clip(scales, self.min_factor * self.exploration_radius, self.max_factor * self.exploration_radius)

        # Make the scales negative 50% of the time
        scales *= 2. * ten.random.bernoulli(0.5, shape=(num_branches,)) - 1.0

        # Make sure the scale is 0.0 whenever the probe is 0
        scales = ten.where(probes == 0, 0., scales)

        # ten.eval(probes, scales)

        return ModuleAssignments(probes, scales)

    def _repr_arg_items(self, **options) -> Iterable[Any]:
        yield from super()._repr_arg_items(**options)
        yield f'exploration_radius={self.exploration_radius}'
        yield f'p={self.p}'
        yield f'mu={self.mu}'
        yield f'sigma={self.sigma}'
        yield f'max_factor={self.max_factor}'
        yield f'min_factor={self.min_factor}'


@provides(Module, 'simple-mlp')
class SimpleMLP(CompiledModule):

    __slots__ = ('layers', 'activation', 'normalize')

    layers: Annotated[list[Module], field(
        doc="The list of layers",
        default_factory=list,
    )]
    activation: Annotated[Activation, field(
        doc="The activation function",
    )]
    normalize: Annotated[bool, field(
        doc="Whether to normalize after each layer",
        default=False,
    )]

    def init_from_args(self, args: ModuleArgs):
        super().init_from_args(args)

        in_dim = args.get('in_dim')
        out_dim = args.get('out_dim')
        hidden_dim = args.get('hidden_dim', default=max(in_dim, out_dim))
        projection_kind = args.get('projection_kind', default='linear')
        num_layers = args.get('num_layers', default=3)

        self.normalize = args.get('normalize', default=False)

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        act = args.get('activation', default='relu')
        self.activation = coerce(Activation, kind=act)
        self.layers = [
            Module.from_args(in_dim=in_dim, out_dim=hidden_dim, bias=False, kind=projection_kind)
        ]

        for _ in range(num_layers-2):
            self.layers.append(Module.from_args(in_dim=hidden_dim, out_dim=hidden_dim, bias=False, kind=projection_kind))

        self.layers.append(Module.from_args(in_dim=hidden_dim, out_dim=out_dim, bias=False, kind=projection_kind))

    @property
    def in_dim(self) -> int:
        return self.layers[0].in_dim

    @property
    def out_dim(self) -> int:
        return self.layers[-1].out_dim

    def build_call(self, mode: Module.Mode, **options) -> Callable:
        activation = self.activation
        inner_layers = self.layers[:-1]
        last_layer = self.layers[-1]

        if self.normalize:
            eps = 1e-6
            norm = ten.norm

            def call(x: Array) -> Array:
                for layer in inner_layers:
                    x = layer(x)
                    x = activation(x)
                    x = x / (norm(x) + eps)
                return last_layer(x)
        else:
            def call(x: Array) -> Array:
                for layer in inner_layers:
                    x = layer(x)
                    x = activation(x)
                return last_layer(x)

        return call

    def _extra_structure(self) -> str:
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}'


@provides(Module, 'simple-rnn')
class SimpleRNN(CompiledModule):

    __slots__ = ('input_proj', 'hidden_proj', 'output_proj', 'memory_proj', 'activation',
                 'memory', 'sever', 'noise')

    input_proj: Annotated[Module, field(
        doc="The input projection",
    )]
    hidden_proj: Annotated[Module, field(
        doc="The hidden projection",
    )]
    output_proj: Annotated[Module, field(
        doc="The output projection",
    )]
    memory_proj: Annotated[Optional[Module], field(
        doc="The memory projection",
    )]
    activation: Annotated[Activation, field(
        doc="The activation function",
    )]
    memory: Annotated[int, field(
        doc="Size of a discrete scrapbook memory to use or zero if no memory",
        default=0
    )]
    sever: Annotated[bool, field(
        doc="Whether to sever the hidden state from the output projection",
        default=False
    )]
    noise: Annotated[float, field(
        doc="The scale for random Guassian noise to inject into the hidden state at each step",
        default=0.0
    )]

    def init_from_args(self, args: ModuleArgs):
        super().init_from_args(args)

        in_dim = args.get('in_dim')
        out_dim = args.get('out_dim')
        hidden_dim = args.get('hidden_dim', default=max(in_dim, out_dim))
        projection_kind = args.get('projection_kind', default='linear')
        memory = args.get('memory', default=0)
        sever = bool(args.get('sever', default=False))

        act = args.get('activation', default='tanh')
        self.activation = coerce(Activation, kind=act)
        self.memory = memory
        self.sever = sever
        self.noise = args.get('noise', 0.0)

        out_in_dim = memory if sever else hidden_dim + memory

        if memory > 0:
            self.memory_proj = Module.from_args(in_dim=hidden_dim, out_dim=memory, bias=False, kind=projection_kind)
        else:
            self.memory_proj = None

        self.input_proj = Module.from_args(in_dim=in_dim, out_dim=hidden_dim, bias=True, kind=projection_kind)
        self.hidden_proj = Module.from_args(in_dim=hidden_dim, out_dim=hidden_dim, bias=False, kind=projection_kind)
        self.output_proj = Module.from_args(in_dim=out_in_dim, out_dim=out_dim, bias=True, kind=projection_kind)

    @property
    def in_dim(self) -> int:
        return self.input_proj.in_dim

    @property
    def hidden_dim(self) -> int:
        return self.input_proj.out_dim

    @property
    def out_dim(self) -> int:
        return self.output_proj.out_dim

    def build_call(self, mode: Module.Mode, **options) -> Callable:
        activation = self.activation
        input_proj = self.input_proj
        hidden_dim = self.hidden_dim
        hidden_proj = self.hidden_proj
        output_proj = self.output_proj
        memory_proj = self.memory_proj
        memory = self.memory
        noise = self.noise

        if noise > 0.0:
            def hidden(x: Array) -> Array:
                h = activation(x)
                return h + ten.random.normal(0., noise, shape=h.shape)
        else:
            hidden = activation

        if memory_proj is not None:
            if self.sever:
                def call(x: Array) -> Array:
                    h = ten.zeros((*x.shape[:-2], hidden_dim), dtype=x.dtype)
                    m = ten.zeros((*x.shape[:-2], memory), dtype=x.dtype)
                    for t in range(x.shape[-2]):
                        xp_t = input_proj(x[..., t, :])
                        if t == 0:
                            mp = memory_proj(activation(xp_t))
                            # mp = memory_proj(xp_t)
                            i = ten.detach(ten.argmax(mp, axis=-1))
                            m = ten.one_hot(i, num_classes=memory, dtype=x.dtype)
                        hp_t = hidden_proj(h)
                        h = hidden(xp_t + hp_t)
                    out = output_proj(m)
                    return out
            else:
                def call(x: Array) -> Array:
                    h = ten.zeros((*x.shape[:-2], hidden_dim), dtype=x.dtype)
                    m = ten.zeros((*x.shape[:-2], memory), dtype=x.dtype)
                    for t in range(x.shape[-2]):
                        xp_t = input_proj(x[..., t, :])
                        if t == 0:
                            mp = memory_proj(activation(xp_t))
                            # mp = memory_proj(xp_t)
                            i = ten.detach(ten.argmax(mp, axis=-1))
                            m = ten.one_hot(i, num_classes=memory, dtype=x.dtype)
                        hp_t = hidden_proj(h)
                        h = hidden(xp_t + hp_t)
                    hm = ten.concat([h, m], axis=-1)
                    out = output_proj(hm)
                    return out
        else:

            def call(x: Array) -> Array:
                h = ten.zeros((*x.shape[:-2], hidden_dim), dtype=x.dtype)
                for t in range(x.shape[-2]):
                    xp_t = input_proj(x[..., t, :])
                    hp_t = hidden_proj(h)
                    h = hidden(xp_t + hp_t)
                out = output_proj(h)
                return out

        return call

    def _extra_structure(self) -> str:
        return f'in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, out_dim={self.out_dim}'


class CreditAssigner(ALEComponent):

    __slots__ = ()

    def assign_credit(self, controller: 'PALEController', step: int, losses: Array) -> Credit:
        raise NotImplementedError()


class BestCreditAssigner(CreditAssigner):

    __slots__ = ()

    def get_best(self, losses: Array) -> Array:
        raise NotImplementedError()

    def assign_credit(self, controller: 'PALEController', step: int, losses: Array) -> Credit:
        best = self.get_best(losses)

        schedule = controller.schedule

        best_probes, best_scales = schedule.get_probes_and_scales(best)

        baseline_loss = losses[0]
        advantages = baseline_loss - losses

        mean_advantage = ten.mean(advantages)

        num_win = ten.sum(1.0 * (advantages > 0))
        pct_win = num_win / advantages.shape[0]

        losses_std = ten.std(losses)

        best_advantage = advantages[best]
        best_std_advantage = best_advantage  / losses_std

        # advantages = advantages / (best_advantage + advantage_eps)

        credit = Credit(best_probes, best_scales, advantages)

        best_scale = ten.mean(ten.abs(best_scales))
        mean_scale = ten.mean(ten.abs(schedule.scales))

        controller.receive_telemetry(Telemetry.construct(
            best_advantage=best_advantage,
            best_scale=best_scale/mean_scale,
            mean_scale=mean_scale,
            best_std_advantage=best_std_advantage,
            pct_win=pct_win,
            mean_std_advantage=ten.minimum(5., (ten.sum(advantages)/num_win)/losses_std),
            mean_advantage=mean_advantage,
            advantages=advantages,
            advantages_std=ten.std(advantages),
        ))

        return credit


@provides(CreditAssigner, 'greedy')
class GreedyCreditAssigner(BestCreditAssigner):

    __slots__ = ()

    def get_best(self, losses: Array) -> Array:
        return ten.argmin(losses)[None]


@provides(CreditAssigner, 'top_k')
class TopKCreditAssigner(BestCreditAssigner):

    __slots__ = ('top_k', )

    top_k: Annotated[int, field(
        doc="The number of branches to use for credit assignment",
        default=1,
    )]

    def get_best(self, losses: Array) -> Array:
        return ten.argpartition(losses, self.top_k)

    def _repr_arg_items(self, **options) -> Iterable[Any]:
        yield from super()._repr_arg_items(**options)
        yield f'top_k={self.top_k}'


class UpdateRule(ALEComponent):

    __slots__ = ('step_size', )

    step_size: Annotated[float, field(
        doc="The step size for udpates",
        default=0.05,
    )]

    def _lazy_controllable_values(self) -> dict[str, 'ControllableValue']:
        values = [
            ControllableValue.from_attr(self, 'step_size', name='step_size'),
        ]
        return {value.name: value for value in values}

    def update_module(self, controller: 'ModuleController', credit: Credit):
        m = controller.index
        module = controller.module
        probes, creds = credit.get_credit(m)
        coefficients = self.step_size * creds
        module.update_in_directions(probes, coefficients)



class ModuleController(ALEComponent):

    __slots__ = ('owner', 'index', 'module', 'needs_bias',
                 'probe_win_counts', 'probe_count_decay',
                 'num_probes', 'keep_probes')

    owner: Annotated['PALEController', field(
        doc="The owner of this module controller",
    )]
    index: Annotated[int, field(
        doc="The index of this controller in the list of controllers",
    )]
    module: Annotated[ProbeableModule, field(
        doc="The module controlled",
    )]
    needs_bias: Annotated[bool, field(
        doc='Whether the module needs a bias too',
        default=False,
    )]
    probe_win_counts: Annotated[Array, field(
        doc="The number of times each probe was best for each modules",
    )]
    probe_count_decay: Annotated[float, field(
        doc="The amount to decay probe counts by each step",
        default=0.9,
    )]
    num_probes: Annotated[int, field(
        doc="The number of probes for this module",
    )]
    keep_probes: Annotated[int, field(
        doc="The number of probes to keep for this module",
    )]

    def postinit(self, spec: Spec):
        super().postinit(spec)
        if isinstance(self.module, ProbedLinear):
            self.needs_bias = self.module.bias is not None

    def _num_probes_changed(self, val: int, old: int):
        self.probe_win_counts = ten.zeros((val + 1,), dtype=ten.float32)

    @property
    def dtype(self) -> DType:
        return self.module.dtype

    @property
    def in_dim(self) -> int:
        return self.module.in_dim

    @property
    def out_dim(self) -> int:
        return self.module.out_dim

    def generate_probes(self, probe_manager: ProbeManager, step: int):
        if step == 0:
            probes = probe_manager.generate(self)
        else:
            kp = -self.keep_probes
            if kp == 0:
                probes = probe_manager.generate(self)
            else:
                keep = ten.argsort(self.probe_win_counts[1:])[kp:] + 1
                probes = probe_manager.generate(self, keep)
                probe_win_counts = ten.zeros((self.num_probes + 1,), dtype=ten.float32)
                probe_win_counts[kp:] = self.probe_win_counts[keep]
                self.probe_win_counts = probe_win_counts

        self.module.update_probes(probes)

    def update_module(self, credit: Credit, rule: UpdateRule):
        rule.update_module(self, credit)

        probes = credit.get_probes(self.index)
        probe_win_counts = self.probe_win_counts * self.probe_count_decay
        np = probes.shape[-1]
        if np == 1:
            probe_win_counts[probes] += 1.
        else:
            probe_win_counts = ten.index_add(probe_win_counts, probes, ten.ones(probes.shape, dtype=probe_win_counts.dtype))
        self.probe_win_counts = probe_win_counts

    def schedule_module(self, schedule: BranchSchedule):
        self.module.update_assignments(schedule.get_module_assignments(self.index))


trigger_defaults = {
    'credit': 1,
    'update': 1,
    'schedule': 1,
    'generate': 50,
    'adapt': 500,
}

class PALEController(ALEController):

    __slots__ = ('modules', 'params', 'schedule', 'controllers',
                 'credit_assigner', 'probe_manager', 'scheduler',
                 'update_rule',
                 'num_branches',
                 'triggers',
                 )

    modules: Annotated[list[ProbeableModule], field(
        doc="The probable modules to be scheduled",
    )]
    controllers: Annotated[list[ModuleController], field(
        doc="The controllers for each module",
    )]
    params: Annotated[Params, field(
        doc='The parameters for this experiment',
        default_factory=dict,
    )]
    schedule: Annotated[BranchSchedule, field(
        doc="The current branch schedule"
    )]
    credit_assigner: Annotated[CreditAssigner, field(
        doc="The object responsible for managing probe credits",
        coerce=True,
    )]
    probe_manager: Annotated[ProbeManager, field(
        doc="The object responsible for generating probes",
        coerce=True,
    )]
    scheduler: Annotated[BranchScheduler, field(
        doc="The object responsible for scheduling branches",
        coerce=True,
    )]
    update_rule: Annotated[UpdateRule, field(
        doc="The object responsible for updating module parameters",
        coerce=True,
    )]
    triggers: Annotated[dict[str, Predicate[int]], field(
        doc="The triggers for this controller",
        default_factory=dict,
    )]
    num_branches: Annotated[int, field(
        doc="The number of branches to run in parallel",
    )]

    def _lazy_states(self) -> dict[str, LearningState]:
        return {
            'loss': LearningState.coerce(name='loss', record=True, kind='ema', controller=self)
        }

    def _lazy_update_rule(self) -> UpdateRule:
        return UpdateRule()

    def _coerce_triggers(self, spec: Any) -> dict[str, Predicate[int]]:
        if spec is None: return {}
        if isinstance(spec, Mapping):
            def make_trigger(t) -> Predicate[int]:
                if isinstance(t, int): return predicates.every_n(t)
                return predicates.coerce(t)
            return {
                name: make_trigger(trig) for name, trig in spec.items()
            }
        raise ValueError(f'Cannot coerce to dict[str, Predicate[int]]: {spec!r}')

    def postinit(self, spec: Spec):
        super().postinit(spec)

        params = self.params

        if adapters := params.get('adapters'):
            for name, adapter_spec in adapters.items():
                self.add_adapter(LearningAdapter.coerce(controller=self, **adapter_spec), name=name, overwrite=True)

    def build_components(self) -> dict[str, ALEComponent]:
        model = self.model
        params = self.params

        modules = self.modules
        if not modules:
            for path, module in tree.traverse(model, include=tree.value_predicate(predicates.is_instance(ProbeableModule))):
                modules.append(module)

        def get_list(p: Any) -> list:
            if isinstance(p, list):
                if len(p) < len(modules):
                    return p + p[-1:] * (len(modules) - len(p))
                return p
            return [p] * len(modules)

        np = get_list(self.probe_manager.num_probes)
        kp = get_list(params.get('keep_probes', 0))

        controllers = [
            ModuleController(
                owner=self,
                index=m,
                module=module,
                num_probes=np[m],
                keep_probes=kp[m]
            )
            for m, module in enumerate(modules)
        ]

        components: dict[str, ALEComponent] = {
            'probe_manager': self.probe_manager,
            'scheduler': self.scheduler,
            'credit_assigner': self.credit_assigner,
            'update_rule': self.update_rule,
        }
        for i, controller in enumerate(controllers):
            components[f'module.{i}'] = controller

        self.controllers = controllers

        return components

    def build_trigger(self, name: str) -> Predicate[int]:
        trigger = self.triggers.get(name)
        if trigger is None:
            every = trigger_defaults.get(name, 1)
            return predicates.every_n(every)
        return trigger

    @property
    def num_modules(self) -> int:
        return len(self.modules)

    def initialize_training(self):
        self.generate_probes(0)
        self.schedule_branches(0)

    def finalize_training(self):
        pass

    def generate_probes(self, step: int) -> None:
        probe_manager = self.probe_manager
        for controller in self.controllers:
            controller.generate_probes(probe_manager, step)

    def schedule_branches(self, step: int) -> None:
        schedule = self.scheduler.schedule(self, self.num_branches)
        for controller in self.controllers:
            controller.schedule_module(schedule)
        self.schedule = schedule

    def add_state(self, state: LearningState, name: str = None):
        if name is None: name = state.name
        if name not in self.states:
            self.states[name] = state

    def add_ema_state(self, name: str):
        if name not in self.states:
            self.states[name] = LearningState.coerce(name=name, controller=self, kind='ema')

    def add_adapter(self, adapter: LearningAdapter, name: str = None, overwrite: bool = False):
        if name is None: name = adapter.name
        if overwrite or name not in self.adapters:
            self.adapters[name] = adapter

    def get_module(self, m: int) -> ProbeableModule:
        return self.modules[m]

    def get_module_of_type(self, m: int, cls: type[T]) -> T:
        module = self.modules[m]
        if not isinstance(module, cls):
            raise ValueError(f"Module at index {m} is not of type {cls.__name__}")
        return module

    def get_modules_of_type(self, cls: type[T]) -> Iterable[T]:
        for module in self.modules:
            if isinstance(module, cls):
                yield module

    def get_num_probes(self, m: int) -> int:
        return self.controllers[m].num_probes

    def get_probe_counts(self, m: int) -> Optional[Array]:
        return self.controllers[m].probe_win_counts

    def receive_telemetry(self, telemetry: Telemetry, name: str = None, buffer: bool = True):
        if buffer:
            self.buffered_telemetry.merge(telemetry)
        else:
            super().receive_telemetry(telemetry, name)

    # def receive_telemetry_item(self, item: str, data: Any, buffer: bool = True):
    #     if buffer:
    #         self.buffered_telemetry.update_item(item, data, merge=True)
    #     else:
    #         super().receive_telemetry_item(item, data)

    def process_telemetry(self, step: int):
        telemetry = self.buffered_telemetry
        telemetry.step = step
        telemetry.eval()
        self.receive_telemetry(telemetry, buffer=False)
        self.buffered_telemetry = Telemetry()

    def build_step(self) -> Callable[[int, Array, Array], Array]:

        model = self.model
        controllers = self.controllers
        credit_assigner = self.credit_assigner
        update_rule = self.update_rule
        loss_fn = self.loss_fn
        adapters = self.adapters

        cume_branch_losses = ten.zeros((self.num_branches,))

        generate_trigger = self.build_trigger('generate')
        schedule_trigger = self.build_trigger('schedule')
        credit_trigger = self.build_trigger('credit')
        update_trigger = self.build_trigger('update')
        adapt_trigger = self.build_trigger('adapt')
        telemetry_trigger = self.build_trigger('telemetry')

        credit_buffer: list[Credit] = []

        generate_probes = self.generate_probes
        schedule_branches = self.schedule_branches
        send_telemetry = self.receive_telemetry
        process_telemetry = self.process_telemetry

        def update_modules(step):
            credit = Credit.combine(credit_buffer)
            for controller in controllers:
                controller.update_module(credit, update_rule)
            credit_buffer.clear()

        def assign_credit(step):
            # Assign credit and add it to the credits list
            credit = credit_assigner.assign_credit(self, step, cume_branch_losses)
            credit_buffer.append(credit)
            cume_branch_losses[...] = 0.

        def step_fn(step: int, inputs: Array, targets: Array) -> Array:
            nonlocal cume_branch_losses

            # Broadcast the inputs to the correct number of branches
            nb = self.num_branches
            inputs = ten.broadcast_to(inputs[None], (nb, *inputs.shape))
            # ten.eval(inputs, targets)

            # Evaluate the model with the broadcast inputs
            outputs = model(inputs)

            # Calculate the loss for each branch
            branch_losses = loss_fn(outputs, targets[None])

            cume_branch_losses += ten.detach(branch_losses)

            if credit_trigger(step):
                assign_credit(step)

            if update_trigger(step):
                update_modules(step)

            if generate_trigger(step):
                generate_probes(step)

            if schedule_trigger(step):
                schedule_branches(step)

            loss = ten.detach(branch_losses[0])

            send_telemetry(Telemetry.construct(
                loss=loss,
                losses=branch_losses,
                losses_std=ten.std(branch_losses)
            ))

            if telemetry_trigger(step):
                process_telemetry(step)

            if adapt_trigger(step):
                for adapter in adapters.values():
                    adapter.adapt(step)

            return loss

        return step_fn

