#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
import time
from pathlib import Path

from collections.abc import Container

from tensile.common import *
from tensile.infra import RootObject
from tensile.nn import CompiledModule, Module, ModuleArgs
from tensile.nn.common import Activation
from tensile.nn.layers import Linear
from tensile.nn.module import Functional
from tensile.optim import Optimizer
from tensile.util.buffer import ArrayBuffer
from tensile.experiment import CachedInputExperiment, Experiment, Params, StudentTrainingExperiment

#
# if ten.ten_kind == "torch":
#     ten.set_default_device("mps")

from river.ale import Adapter, ALEController, Controllable, ControllerState, Telemetry, ValueAdapter


T = TypeVar('T')


def normalize(x: Array) -> Array:
    return x / ten.norm(x)


class Probe(Protocol):

    def __call__(self, x: Array) -> Array: ...


class Probes(Object):

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

    def build_probe(self, assignment: Array, scales: Array = None) -> Probe:
        raise NotImplementedError()

    def update_module(self, module: 'ProbeableModule', credit: Array, lr: float) -> None:
        raise NotImplementedError()

    def _repr_args(self, **options) -> str:
        return f'{self.in_dim} -> {self.out_dim}, k={self.k}'


@provides(Probes, 'lora')
class LoRAProbes(Probes):

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

    # noinspection PyPep8Naming
    def build_probe(self, assignment: Array, scales: Array = None) -> Probe:
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

        P = left.shape[0]

        # ten.eval(left, right)

        if bias is None:
            def probe(x: Array) -> Array:
                # ten.eval(x)
                if x.ndim < 3:
                    x = ten.expand_dims(x, -2)
                    squeeze = True
                else:
                    squeeze = False
                assert x.shape[0] == P, f"Expected {P} perturbations, got {x.shape[0]}"

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
            def probe(x: Array) -> Array:
                # ten.eval(x)
                if x.ndim < 3:
                    x = ten.expand_dims(x, -2)
                    squeeze = True
                else:
                    squeeze = False
                assert x.shape[0] == P, f"Expected {P} perturbations, got {x.shape[0]}"

                p = ten.matmul(x, left)
                y = ten.matmul(p, right) + bias

                if squeeze:
                    out = ten.squeeze(y, axis=-2)
                    out = scale[:, None] * out
                else:
                    out = scale[:, None, None] * y
                # ten.eval(out)
                return out


        return probe

    def update_module(self, module: 'ProbedLinear', credit: Array, lr: float) -> None:
        expanded = ten.matmul(self.left[1:], self.right[1:])
        # ten.eval(credit, expanded)
        update = lr * ten.sum(expanded * credit[1:, None, None], axis=0)
        module.weight += ten.swapaxes(update, 0, 1)
        bias = self.bias
        if bias is not None:
            bias_update = lr * ten.sum(bias[1:] * credit[1:, None], axis=0)
            module.bias += bias_update

    def _repr_args(self, **options) -> str:
        return super()._repr_args(**options) + f', rank={self.rank}'


class ProbeableModule(CompiledModule):

    __slots__ = ()

    @property
    def is_probing(self) -> bool:
        raise NotImplementedError()

    @property
    def probes(self) -> Probes:
        raise NotImplementedError()

    @probes.setter
    def probes(self, probes: Probes) -> None:
        raise NotImplementedError()

    def perturb(self, perturbations: 'Perturbations') -> None:
        raise NotImplementedError()

    def update_from_credit(self, credit: Array, lr: float) -> Array:
        raise NotImplementedError()


class ProbeGenerator(Controllable):

    __slots__ = ('num_probes', 'orthogonal')

    num_probes: Annotated[int, field(
        doc="The number of probes",
        default=10,
    )]
    orthogonal: Annotated[bool, field(
        doc="Whether to use orthogonal probes",
        default=False,
    )]

    def generate(self, controller: 'ProbeController', m: int, module: ProbeableModule) -> Probes:
        raise NotImplementedError()


@provides(ProbeGenerator, 'lora')
class LoRAProbeGenerator(ProbeGenerator):

    __slots__ = ('rank', 'bias_p')

    rank: Annotated[int, field(
        doc="The rank of the adapters",
        default=1,
    )]
    bias_p: Annotated[float, field(
        doc="The probability of adding bias to the adapter",
        default=0.7,
    )]

    def generate(self, controller: 'ProbeController', m: int, module: 'ProbedLinear') -> Probes:
        k = self.num_probes
        r = self.rank
        n = module.in_dim
        m = module.out_dim

        dtype = module.weight.dtype

        # j = max(r * k, r)
        # lv = ten.random.normal(shape=(1, n, j))
        # rv = ten.random.normal(shape=(1, j, m))
        #
        # if self.orthogonal:
        #     lv = lv / ten.norm(lv, axis=1, keepdims=True)
        #     rv = rv / ten.norm(rv, axis=2, keepdims=True)
        #
        #     # ten.eval(lv, rv)
        #
        #     def reject(xh: Array, xi: Array) -> Array:
        #         xdot = ten.sum(xi * xh)
        #         xh = xh - xdot * xi
        #         return normalize(xh)
        #
        #     for h in range(1, j):
        #         lvh = lv[0, :, h]
        #         rvh = rv[0, h, :]
        #         for i in range(h):
        #             lvi = lv[0, :, i]
        #             rvi = rv[0, i, :]
        #             lvh = reject(lvh, lvi)
        #             rvh = reject(rvh, rvi)
        #         lv[0, :, h] = lvh
        #         rv[0, h, :] = rvh

        left = ten.as_type(ten.random.normal(shape=(k+1, n, r)), dtype)
        right = ten.as_type(ten.random.normal(shape=(k+1, r, m)), dtype)
        left = left / ten.norm(left, axis=1, keepdims=True)
        right = right / ten.norm(right, axis=2, keepdims=True)

        left[0] = 0.
        right[0] = 0.

        # lefts = [ten.zeros((1, n, r))]
        # rights = [ten.zeros((1, r, m))]
        # lt = ten.as_type(ten.random.normal(shape=(k, n, r)), dtype)
        # rt = ten.as_type(ten.random.normal(shape=(k, r, m)), dtype)
        # lefts.append(lt / ten.norm(lt, axis=1, keepdims=True))
        # rights.append(rt / ten.norm(rt, axis=2, keepdims=True))
        # for _ in range(k):
        #     lt = ten.random.normal(shape=(1, n, r))
        #     rt = ten.random.normal(shape=(1, r, m))
        #     lefts.append(lt / ten.norm(lt, axis=1, keepdims=True))
        #     rights.append(rt / ten.norm(rt, axis=2, keepdims=True))
        # if r > 1:
        #     for _ in range(k):
        #         li = ten.random.permutation(j)[:r]
        #         ri = ten.random.permutation(j)[:r]
        #         lt = lv[:, :, li]
        #         rt = rv[:, ri, :]
        #         lt = normalize(lt)
        #         rt = normalize(rt)
        #         lefts.append(lt)
        #         rights.append(rt)
        # else:
        #     for _ in range(k):
        #         li = ten.random.permutation(j)[:r]
        #         ri = ten.random.permutation(j)[:r]
        #         lt = lv[:, :, li]
        #         rt = rv[:, ri, :]
        #         if r > 1:
        #             lt = normalize(lt)
        #             rt = normalize(rt)
        #         lefts.append(lt)
        #         rights.append(rt)
        # left = ten.concatenate(lefts, axis=0)
        # right = ten.concatenate(rights, axis=0)

        # ten.eval(left, right)

        if module.bias is None:
            bias = None
        else:
            bias_p = self.bias_p
            bias = ten.as_type(ten.random.normal(shape=(k+1, m)), dtype)
            bias = bias / ten.norm(bias, axis=1, keepdims=True)
            if bias_p < 1.0:
                bias = ten.random.bernoulli(bias_p, shape=(k+1, m)) * bias
            bias[0] = 0.

        probes = Probes.coerce(k=k, in_dim=n, out_dim=m, rank=r, left=left, right=right, bias=bias, kind='lora')

        module.probes = probes

        return probes


@provides(Linear, 'probed')
class ProbedLinear(Linear, ProbeableModule):

    __slots__ = ('probes', 'probe')

    weight: Annotated[Array, field(
        doc="The learnable weights of the layer.",
        parameter=False,
    )]
    bias: Annotated[Optional[Array], field(
        doc="The learnable bias of the layer.",
        parameter=False,
    )]

    probes: Annotated[Probes, field(
        doc='The probes for the module'
    )]
    probe: Annotated[Optional[Probe], field(
        doc="The probe function for this moudle",
    )]

    @property
    def is_probing(self) -> bool:
        return self.probe is not None

    def init_from_args(self, args: Linear.Args):
        super().init_from_args(args)
        ten.eval(self.weight)

    def perturb(self, perturbations: 'Perturbations'):
        probes = perturbations.probes
        # noinspection PyTypeChecker
        if ten.all(probes == 0):
            self.probe = None
        else:
            self.probe = self.probes.build_probe(probes, scales=perturbations.scales)
        self.calls.clear()
        self.compile()

    def update_from_credit(self, credit: Array, lr: float) -> Array:
        self.probes.update_module(self, credit, lr=lr)
        # update = self.probes.get_update(credit, lr=lr)
        # self.weight += ten.swapaxes(update, 0, 1)
        return self.weight

    def get_probe(self) -> Probe:
        if self.probe is None:
            raise ValueError("Probe function is None, cannot get probe")
        return self.probe

    def build_call(self, mode: Module.Mode, **options) -> Functional:
        if self.bias is None:
            if self.is_probing:
                probe = self.get_probe()

                def call(x: Array) -> Array:
                    # ten.eval(x)

                    y = ten.matmul(x, self.weight.T)

                    # ten.eval(y)

                    out = y + probe(x)

                    # ten.eval(out)

                    return out
            else:
                def call(x: Array) -> Array:
                    # ten.eval(x)

                    out = ten.matmul(x, self.weight.T)

                    # ten.eval(out)

                    return out
        else:
            if self.is_probing:
                probe = self.get_probe()

                def call(x: Array) -> Array:
                    # ten.eval(x)

                    y = ten.matmul(x, self.weight.T) + self.bias

                    # ten.eval(y)

                    out = y + probe(x)

                    # ten.eval(out)

                    return out
            else:
                def call(x: Array) -> Array:
                    # ten.eval(x)

                    out = ten.matmul(x, self.weight.T) + self.bias

                    # ten.eval(out)

                    return out



        return call

    Args = Linear.Args


class Perturbations(RootObject):

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

    def concatenate(self, probes: Array, scales: Array, in_place: bool = True) -> 'Perturbations':
        assert probes.shape == scales.shape, "Probes and scales must have the same shape"

        probes = ten.concatenate([self.probes, probes])
        scales = ten.concatenate([self.scales, scales])
        if in_place:
            self.probes = probes
            self.scales = scales
            return self
        else:
            return Perturbations(probes, scales)

    def __len__(self) -> int:
        return self.probes.shape[0]


class PerturbationMatrix(RootObject):

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

    @property
    def num_perturbations(self) -> int:
        return self.probes.shape[-1]

    @property
    def num_modules(self) -> int:
        return self.probes.shape[0]

    def get_module_perturbations(self, mod: int) -> Perturbations:
        assert 0 <= mod < self.num_modules, "Module index out of range"
        return Perturbations(self.probes[mod], self.scales[mod])

    def _repr_args(self, **options) -> str:
        return f"{self.num_modules} x {self.num_perturbations}"

    @classmethod
    def from_perturbations(cls, perturbations: list[Perturbations]) -> 'PerturbationMatrix':
        probes = ten.stack([p.probes for p in perturbations], axis=0)
        scales = ten.stack([p.scales for p in perturbations], axis=0)
        return cls(probes, scales)


class PerturbationScheduler(Controllable):

    __slots__ = ('perturbations', 'history', 'keep', 'kept', 'force_null_perturbation',
                 'schedule_count')

    perturbations: Annotated[PerturbationMatrix, field(
        doc="The current perturbation matrix"
    )]
    history: Annotated[list[PerturbationMatrix], field(
        doc="The history of perturbation matrices",
        default_factory=list,
    )]
    keep: Annotated[int, field(
        doc="The number perturbations to keep",
    )]
    kept: Annotated[Optional[Array], field(
        doc="The indexes of the perturbations to keep",
    )]
    force_null_perturbation: Annotated[bool, field(
        doc="Whether to force the first perturbation to have null probes assigned",
        default=True
    )]
    schedule_count: Annotated[int, field(
        doc="The number of times that schedule has been called",
        default=0,
    )]

    def perturb_modules(self, modules: list[ProbeableModule], perturbations: PerturbationMatrix = None):
        if perturbations is None: perturbations = self.perturbations

        for m, module in enumerate(modules):
            module.perturb(perturbations.get_module_perturbations(m))

    def set_perturbations(self, modules: list[ProbeableModule], perturbations: PerturbationMatrix):
        self.perturbations = perturbations
        self.history.append(perturbations)
        self.perturb_modules(modules, perturbations)

    def record_advantage(self, advantage: Array):
        if self.keep > 0:
            self.kept = ten.argsort(advantage[1:])[:self.keep] + 1

    def record_credit(self, m: int, credit: Array):
        pass

    def schedule_called(self) -> int:
        self.schedule_count += 1
        return self.schedule_count

    def schedule(self, controller: 'ProbeController', modules: list[ProbeableModule], num_perturbations: int) -> PerturbationMatrix:
        self.schedule_called()
        new_perturbations = []
        kept = self.kept
        if kept is not None:
            perturbations = self.perturbations
            old_probes = perturbations.probes
            old_scales = perturbations.scales
            # ten.eval(kept, old_probes, old_scales)
            num_perturbations -= kept.shape[0]
            for m, module in enumerate(modules):
                mod_pertubrations = self.generate_perturbations(m, module, num_perturbations)
                mod_pertubrations.concatenate(old_probes[m, kept], old_scales[m, kept], in_place=True)
                new_perturbations.append(mod_pertubrations)
        else:
            for m, module in enumerate(modules):
                mod_pertubrations = self.generate_perturbations(m, module, num_perturbations)
                new_perturbations.append(mod_pertubrations)

        perturbations = PerturbationMatrix.from_perturbations(new_perturbations)

        self.set_perturbations(modules, perturbations)
        return perturbations

    def generate_perturbations(self, m: int, module: ProbeableModule, num_perturbations: int) -> Perturbations:
        raise NotImplementedError()

    # def schedule_module(self, module: ProbeableModule, num_perturbations: int) -> tuple[Array, Array]:
    #     raise NotImplementedError()


@provides(PerturbationScheduler, 'random')
class RandomPerturbationScheduler(PerturbationScheduler):

    __slots__ = ('exploration_radius', 'p', 'mu', 'sigma')

    exploration_radius: Annotated[float, field(
        doc="The exploration radius of the probe perturbations",
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
    p: Annotated[Optional[float], field(
        doc="The probability that the module will be perturbed in a given perturbation",
    )]

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

    def generate_perturbations(self, m: int, module: ProbeableModule, num_perturbations: int) -> Perturbations:
        k_m = module.probes.k
        p = self.p
        if p is None:
            probes = ten.random.randint(low=0, high=k_m+1, shape=(num_perturbations,))
        else:
            # Decide whether the module is perturbed in each perturbation
            perturbed = ten.random.bernoulli(p, shape=(num_perturbations,))
            # Randomly assign probe indexes between 1 and k_m for the ones that are perturbed
            probes = perturbed * ten.random.randint(low=1, high=k_m+1, shape=(num_perturbations,))
        # Scale by +/- magnitude
        scales = self.exploration_radius * ten.exp(self.mu + self.sigma * ten.random.normal(shape=(num_perturbations,)))
        scales = ten.clip(scales, 0.005, 0.5)
        scale_sign = (2. * ten.random.bernoulli(0.5, shape=(num_perturbations,)) - 1.0)

        scales = ten.where(probes == 0, 0., scales * scale_sign)
        # ten.eval(probes, scales)
        if self.force_null_perturbation:
            probes[0] = 0
            scales[0] = 0.

        return Perturbations(probes, scales)

    # def schedule_module(self, module: ProbeableModule, num_perturbations: int) -> tuple[Array, Array]:
    #     k_m = module.probes.k
    #     p = self.p
    #     if p is None:
    #         probes = ten.random.randint(low=0, high=k_m+1, shape=(num_perturbations,))
    #     else:
    #         # Decide whether the module is perturbed in each perturbation
    #         perturbed = ten.random.bernoulli(p, shape=(num_perturbations,))
    #         # Randomly assign probe indexes between 1 and k_m for the ones that are perturbed
    #         probes = perturbed * ten.random.randint(low=1, high=k_m+1, shape=(num_perturbations,))
    #     # Scale by +/- magnitude
    #     scales = self.exploration_radius * ten.exp(self.mu + self.sigma * ten.random.normal(shape=(num_perturbations,)))
    #     scales = ten.clip(scales, 0.005, 0.5)
    #     scale_sign = (2. * ten.random.bernoulli(0.5, shape=(num_perturbations,)) - 1.0)
    #
    #     scales = ten.where(probes == 0, 0., scales * scale_sign)
    #     # ten.eval(probes, scales)
    #     if self.force_null_perturbation:
    #         probes[0] = 0
    #         scales[0] = 0.
    #     return probes, scales


@provides(Module, 'simple-mlp')
class SimpleMLP(CompiledModule):

    __slots__ = ('layers', 'activation')

    layers: Annotated[list[Module], field(
        doc="The list of layers",
        default_factory=list,
    )]
    activation: Annotated[Activation, field(
        doc="The activation function",
    )]

    def init_from_args(self, args: ModuleArgs):
        super().init_from_args(args)

        in_dim = args.get('in_dim')
        out_dim = args.get('out_dim')
        hidden_dim = args.get('hidden_dim', default=max(in_dim, out_dim))
        layer_kind = args.get('layer_kind', default='linear')
        num_layers = args.get('num_layers', default=3)

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        act = args.get('activation', default='relu')
        self.activation = coerce(Activation, kind=act)
        self.layers = [Module.from_args(in_dim=in_dim, out_dim=hidden_dim, bias=False, kind=layer_kind)]

        for _ in range(num_layers-2):
            self.layers.append(Module.from_args(in_dim=hidden_dim, out_dim=hidden_dim, bias=False, kind=layer_kind))

        self.layers.append(Module.from_args(in_dim=hidden_dim, out_dim=out_dim, bias=False, kind=layer_kind))

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

    __slots__ = ('input_proj', 'hidden_proj', 'output_proj', 'activation')

    input_proj: Annotated[Module, field(
        doc="The input projection",
    )]
    hidden_proj: Annotated[Module, field(
        doc="The hidden projection",
    )]
    output_proj: Annotated[Module, field(
        doc="The output projection",
    )]
    activation: Annotated[Activation, field(
        doc="The activation function",
    )]

    def init_from_args(self, args: ModuleArgs):
        super().init_from_args(args)

        in_dim = args.get('in_dim')
        out_dim = args.get('out_dim')
        hidden_dim = args.get('hidden_dim', default=max(in_dim, out_dim))
        projection_kind = args.get('projection_kind', default='linear')

        act = args.get('activation', default='tanh')
        self.activation = coerce(Activation, kind=act)

        self.input_proj = Module.from_args(in_dim=in_dim, out_dim=hidden_dim, bias=True, kind=projection_kind)
        self.hidden_proj = Module.from_args(in_dim=hidden_dim, out_dim=hidden_dim, bias=False, kind=projection_kind)
        self.output_proj = Module.from_args(in_dim=hidden_dim, out_dim=out_dim, bias=True, kind=projection_kind)

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

        def call(x: Array) -> Array:
            h = ten.zeros((*x.shape[:-2], hidden_dim))
            for t in range(x.shape[-2]):
                xp_t = input_proj(x[..., t, :])
                hp_t = hidden_proj(h)
                h = activation(xp_t + hp_t)
            out = output_proj(h)
            return out
        return call

    def _extra_structure(self) -> str:
        return f'in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, out_dim={self.out_dim}'



class CreditAssigner(Controllable):

    __slots__ = ('lr', 'lr_decay', 'decay_every')

    lr: Annotated[float, field(
        doc="The learning rate for credit assignment",
        default=0.05,
    )]
    lr_decay: Annotated[float, field(
        doc="The learning rate decay for credit assignment",
        default=1.00,
    )]
    decay_every: Annotated[float, field(
        doc="How often to decay the learning rate for credit assignment",
        default=1000000,
    )]

    def assign_credit(self, controller: 'ProbeController', step: int, losses: Array):
        raise NotImplementedError()


@provides(CreditAssigner, 'greedy')
class GreedyCreditAssigner(CreditAssigner):

    __slots__ = ()

    def assign_credit(self, controller: 'ProbeController', step: int, losses: Array):
        lr = self.lr
        advantage_eps = 1e-6
        perturbations = controller.perturbations

        best = ten.argmin(losses)

        best_probes = perturbations.probes[:, best]

        baseline_loss = losses[0]
        advantage = ten.maximum(0., baseline_loss - losses)

        advantage = advantage / (advantage[best] + advantage_eps)

        controller.receive_telemetry_item('advantage', advantage)

        # if step % self.decay_every == 0:
        #     lr = self.lr = lr * self.lr_decay

        for m, mod in enumerate(controller.modules):
            if mod.is_probing:
                credit = ten.zeros((mod.probes.k + 1,))
                probe = best_probes[m]
                credit[probe] = perturbations.scales[m, best]
                # ten.eval(credit)
                mod.update_from_credit(credit, lr=lr)
                controller.receive_telemetry_item('credit', credit)


@provides(CreditAssigner, 'top_k')
class TopKCreditAssigner(CreditAssigner):

    __slots__ = ('top_k', )

    top_k: Annotated[int, field(
        doc="The number of perturbation to use for credit assignment",
        default=1,
    )]

    def assign_credit(self, controller: 'ProbeController', step: int, losses: Array):
        top_k = self.top_k
        lr = self.lr
        advantage_eps = 1e-6
        perturbations = controller.perturbations

        best = ten.argsort(losses)[:top_k]

        best_probes = perturbations.probes[:, best]

        baseline_loss = losses[0]
        advantage = ten.maximum(0., baseline_loss - losses)

        advantage = advantage / (ten.sum(advantage[best], keepdims=True) + advantage_eps)

        # controller.record_advantage(advantage)
        controller.receive_telemetry_item('advantage', advantage)

        # if step % self.decay_every == 0:
        #     lr = self.lr = lr * self.lr_decay

        for m, mod in enumerate(controller.modules):
            if mod.is_probing:
                credit = ten.zeros((mod.probes.k + 1,))
                i = best_probes[m]
                for k in range(top_k):
                    p = best[k]
                    probe_credit = perturbations.scales[m, p] * advantage[p]
                    credit[i[k]] += probe_credit
                # ten.eval(credit)
                mod.update_from_credit(credit, lr=lr)
                controller.receive_telemetry_item('credit', credit)
                # controller.record_credit(m, credit)


@provides(Experiment, 'sgd')
class GradientDescentExperiment(StudentTrainingExperiment):

    __slots__ = ('optimizer_kind',)

    optimizer_kind: Annotated[str, field(
        doc='The optimizer kind to use'
    )]

    @staticmethod
    def loss_fn(outputs: Array, targets: Array) -> Array:
        return ten.mean(
            ten.square(outputs - targets)
        )

    def _lazy_optimizer_kind(self) -> str:
        return self.name

    def train(self) -> int:
        lr = self.get_param('lr')

        student = self.student

        optimizer_kind = self.optimizer_kind
        if optimizer_kind == 'adam':
            kwargs = dict(weight_decay=self.get_param('weight_decay'))
        elif optimizer_kind == 'sgd':
            kwargs = dict(momentum=self.get_param('momentum'))
        else:
            kwargs = dict()

        optimizer = Optimizer.coerce(model=student, learning_rate=lr, kind=optimizer_kind, **kwargs)

        bucket_desc = self.descriptor

        loss_fn = self.loss_fn
        header = self.header
        log = self.print
        report_every = self.report_every

        def train_fn(batch: tuple[Array, Array]):
            inputs, targets = batch
            outputs = student(inputs)
            return loss_fn(outputs, targets)

        step = optimizer.stepper(train_fn)

        s = 1
        losses = ArrayBuffer()
        header(f'Student[{bucket_desc}] Training')

        for e in range(self.num_epochs):
            header(f'Student[{bucket_desc}] Epoch: {e+1:2d}')
            for b in self.batch_data(self.batch_size):
                loss = step(b)
                # ten.eval(loss)
                losses.append(loss[None])
                if s % report_every == 0:
                    log(f'Step: {s:5d}  Loss: {loss:.6f}')
                s += 1

        for inputs, targets in self.batch_data(1024):
            outputs = student(inputs)
            loss = loss_fn(outputs, targets)
            log(f'Student[{self.descriptor}] Final loss: {loss:.6f}') # 0.028599 0.083547
            break

        self.add_metric('loss', losses)

        return s-1


@provides(Adapter, 'learning-rate')
class LearningRateAdapter(ValueAdapter):

    __slots__ = ('creditor',)

    creditor: Annotated[CreditAssigner, field(
        doc="The creditor to control",
        required=True,
    )]

    @property
    def name(self) -> str:
        return 'creditor.learning_rate'

    def get_value(self) -> float:
        return self.creditor.lr

    def set_value(self, value: float) -> None:
        self.creditor.lr = value


@provides(Adapter, 'exploration-radius')
class ExplorationRadiusAdapter(ValueAdapter):

    __slots__ = ('scheduler',)

    scheduler: Annotated[RandomPerturbationScheduler, field(
        doc="The scheduler to control",
        required=True,
    )]

    @property
    def name(self) -> str:
        return 'scheduler.exploration_radius'

    def get_value(self) -> float:
        return self.scheduler.exploration_radius

    def set_value(self, value: float) -> None:
        self.scheduler.modify_exploration_radius(value)


class ProbeController(ALEController):

    __slots__ = ('modules', 'perturbations', 'creditor', 'generator', 'scheduler', 'num_perturbations')

    modules: Annotated[list[ProbeableModule], field(
        doc="The probable modules to be scheduled",
    )]
    perturbations: Annotated[PerturbationMatrix, field(
        doc="The current perturbation matrix"
    )]
    creditor: Annotated[CreditAssigner, field(
        doc="The object responsible for managing probe credits"
    )]
    generator: Annotated[ProbeGenerator, field(
        doc="The object responsible for generating perturbations"
    )]
    scheduler: Annotated[PerturbationScheduler, field(
        doc="The object responsible for scheduling perturbation application"
    )]
    num_perturbations: Annotated[int, field(
        doc="The number of perturbations to run in parallel",
    )]

    def postinit(self, spec: Spec):
        super().postinit(spec)

        if 'modules' not in spec:
            model = self.model
            modules = []
            for path, module in tree.traverse(model, include=tree.value_predicate(predicates.is_instance(ProbeableModule))):
                modules.append(module)
            self.modules = modules

        states = self.states

        if 'loss' not in states:
            states['loss'] = ControllerState.coerce(name='loss', controller=self, kind='ema')

        for state in self.states.values():
            self.add_receiver(state)

        self.add_adapter(Adapter.coerce(
            controller=self,
            state='loss',
            scheduler=self.scheduler,
            threshold=0.1,
            delta_threshold=-0.05,
            max_value=1.0,
            min_value=0.01,
            multiplier=1.1,
            kind='exploration-radius',
        ))

        # self.add_adapter(Adapter.coerce(
        #     controller=self,
        #     state='loss',
        #     creditor=self.creditor,
        #     threshold=1.5,
        #     delta_threshold=-0.05,
        #     max_value=1.0,
        #     min_value=0.01,
        #     multiplier=1.1,
        #     kind='learning-rate',
        # ))

    @property
    def num_modules(self) -> int:
        return len(self.modules)

    def initialize(self):
        modules = self.modules

        for m, module in enumerate(modules):
            self.generator.generate(self, m, module)

        self.perturbations = self.scheduler.schedule(self, modules, self.num_perturbations)

    def add_adapter(self, adapter: Adapter, name: str = None):
        if name is None: name = adapter.name
        if name not in self.adapters:
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

    def receive_telemetry(self, telemetry: Telemetry, name: str = None, buffer: bool = True):
        if buffer:
            self.buffered_telemetry.merge(telemetry)
        else:
            super().receive_telemetry(telemetry, name)

    def receive_telemetry_item(self, item: str, data: Any, buffer: bool = True):
        if buffer:
            self.buffered_telemetry.update_item(item, data, merge=True)
        else:
            super().receive_telemetry_item(item, data)

    def process_telemetry(self):
        telemetry = self.buffered_telemetry
        telemetry.eval()
        self.receive_telemetry(telemetry, buffer=False)
        self.buffered_telemetry = Telemetry()

    # @property
    # def loss(self) -> EMAState:
    #     loss = self.states['loss']
    #     if isinstance(loss, EMAState):
    #         return loss
    #     else:
    #         raise ValueError(f"Expected 'loss' state to be an EMAState, got {type(loss)}")

    def stepper(self, generate_every: int, schedule_every: int):
        model = self.model
        modules = self.modules
        scheduler = self.scheduler
        generator = self.generator
        creditor = self.creditor
        loss_fn = self.loss_fn

        adaptive_warmup = 3000
        adapt_every = 500
        adapters = self.adapters

        def take_step(step: int, inputs: Array, targets: Array) -> Array:
            np = self.num_perturbations
            inputs = ten.broadcast_to(inputs[None], (np, *inputs.shape))
            # ten.eval(inputs, targets)
            outputs = model(inputs)

            losses = loss_fn(outputs, targets[None])

            creditor.assign_credit(self, step, losses)

            if step % generate_every == 0:
                for m, module in enumerate(modules):
                    generator.generate(self, m, module)

            if step % schedule_every == 0:
                self.perturbations = scheduler.schedule(self, modules, self.num_perturbations)

            loss = ten.detach(losses[0])

            self.receive_telemetry_item('loss', loss)
            self.receive_telemetry_item('losses', losses)

            self.process_telemetry()

            if step > adaptive_warmup and step % adapt_every == 0:
                for adapter in adapters.values():
                    adapter.adapt(step)

            # if step % 250 == 0:
            #     print(f'Step {step}: Loss: {loss_state.describe()}')

            return loss

        return take_step


@provides(Experiment, 'probed')
class ProbedTrainingExperiment(StudentTrainingExperiment):

    __slots__ = ()

    @staticmethod
    def perturbation_loss_fn(outputs: Array, targets: Array) -> Array:
        return ten.mean(
            ten.square(outputs - targets),
            axis=(1, 2)
        )

    def train(self) -> int:

        num_perturbations = self.get_param('num_perturbations')
        schedule_every = self.get_param('schedule_every', default=1)
        generate_every = self.get_param('generate_every', default=100000000000)

        student = self.student

        modules = []
        for path, module in tree.traverse(student, include=tree.value_predicate(predicates.is_instance(ProbeableModule))):
            modules.append(module)

        generator_spec = self.get_params_with_prefix('generator.')
        generator = ProbeGenerator.coerce(**generator_spec)

        scheduler_spec = self.get_params_with_prefix('scheduler.')
        scheduler = PerturbationScheduler.coerce(**scheduler_spec)

        creditor_spec = self.get_params_with_prefix('creditor.')
        creditor = CreditAssigner.coerce(**creditor_spec)

        controller = ProbeController(
            model=student,
            modules=modules,
            creditor=creditor,
            generator=generator,
            scheduler=scheduler,
            num_perturbations=num_perturbations,
            loss_fn=self.perturbation_loss_fn,
        )

        controller.initialize()

        header = self.header
        log = self.print

        def report_perturbations():
            probes = controller.perturbations.probes
            scales = controller.perturbations.scales

            header('probes')
            log(ten.sign(scales) * probes)
            header('scales')
            log(scales)

        report_perturbations()

        report_every = self.report_every

        s = 1

        losses = ArrayBuffer()
        header(f'Student[{self.descriptor}] Training')

        step = controller.stepper(generate_every, schedule_every)

        for e in range(self.num_epochs):
            header(f'Student[{self.descriptor}] Epoch: {e+1:2d}')
            for inputs, targets in self.batch_data(self.batch_size):

                loss = step(s, inputs, targets)
                # inputs = ten.broadcast_to(inputs[None], (num_perturbations, *inputs.shape))
                # ten.eval(inputs, targets)
                # outputs = student(inputs)
                # perturbation_losses = perturbation_loss_fn(outputs, targets[None])
                #
                # if s % eval_every == 0:
                #     ten.eval(perturbation_losses)
                #
                # creditor.assign_credit(s, scheduler, perturbation_losses)
                #
                # loss = perturbation_losses[0]
                losses.append(loss[None])

                # if s % generate_every == 0:
                #     for module in modules:
                #         module.probes = generator.generate(module)
                #
                # if s % schedule_every == 0:
                #     perturbations = scheduler.schedule(num_perturbations)

                if s % report_every == 0:
                    log(f'[{self.name}]: Step: {s:5d}  Loss: {loss:.6f} Learning Rate: {creditor.lr:.4f}')

                s += 1

        for inputs, targets in self.batch_data(1024):
            inputs = ten.broadcast_to(inputs[None], (controller.num_perturbations, *inputs.shape))
            outputs = student(inputs)
            perturbation_losses = controller.loss_fn(outputs, targets[None])
            loss = perturbation_losses[0]
            ten.eval(perturbation_losses)
            log(f'Student[{self.descriptor}] Final loss: {loss:.6f}') # 0.028599 0.083547
            break

        self.add_metric(f'loss', losses)

        return s-1

    def fixed_param_defaults(self) -> Params:
        return {
            'generator.rank': 1,
            'generator.kind': 'lora',
            'scheduler.exploration_radius_decay': 1.0,
            'scheduler.decay_every': 1000,
            'creditor.top_k': 1,
            'creditor.lr_decay': 1.0,
            'creditor.decay_every': 1000,
            'schedule_every': 1,
            'scheduler.kind': 'random',
            'creditor.kind': 'greedy',
        }

    hyperparam_abbrevs = {
        **Experiment.hyperparam_abbrevs,
        'generator.rank': 'gr',
        'generator.num_probes': 'np',
        'num_perturbations': 'P',
        'scheduler.exploration_radius': 'er',
        'scheduler.exploration_radius_decay': 'erd',
        'scheduler.decay_every': 'sde',
        'top_k': 'tk',
        'weight_decay': 'wd',
        'creditor.lr': 'lr',
        'creditor.lr_decay': 'lrd',
        'creditor.decay_every': 'cde',
        'generate_every': 'ge',
    }


@provides(Experiment, 'tmaze')
class TMazeExperiment(CachedInputExperiment):

    __slots__ = ('delay',)

    delay: Annotated[int, field(
        doc='Delay in steps before credit is rewarded',
        default=5,
    )]

    def get_module(self, spec: Any, name: str = None, save: bool | Path = True) -> Module:
        if save is True and name is not None:
            save = self.get_path(name, write=True)
        return super().get_module(spec, name, save)

    def generate_input_chunk(self) -> Array:
        chunk = ten.zeros((self.in_chunk_size, self.delay, self.in_dim))
        chunk[:, 0, 0] = 2. * ten.random.bernoulli(0.5, shape=(self.in_chunk_size,)) - 1.
        chunk[:, 1:, 1] = ten.random.normal(scale=2., shape=(self.in_chunk_size, self.delay-1,))
        return chunk

    def batch_data(self, b: int) -> Iterable[tuple[Array, Array]]:
        chunk_size = self.in_chunk_size
        for i in range(self.chunks_per_epoch):
            chunk = self.get_input_chunk(i)
            for s in range(0, chunk_size, b):
                inputs = chunk[s:s+b]
                targets = inputs[..., 0, 0:1]
                yield inputs, targets


def moving_average(x: Array, window: int, axis: int = -1) -> Array:
    x = ten.swapaxes(x, axis, -1)

    c = ten.cumsum(x, axis=-1)
    c[..., window:] = c[..., window:] - c[..., :-window]
    y = c[..., window - 1:] / window

    return ten.swapaxes(y, -1, axis)


def main():
    seeds = [
        # 10,
        20,
        # 30,
        # 40
    ]

    top = Experiment(
        name=f'test-{time.strftime("%Y%m%d-%H%M%S", time.localtime())}',
    )

    for seed in seeds:
        # run_teacher_regression_experiment(seed)
        run_tmaze_experiment(seed, seed_range=4, delay=30, parent=top)
        # run_tmaze_experiment(10, train='pbdca', seed_range=2)

    top.run()

    return 0



def run_teacher_regression_experiment(seed: int = 10, train: Container[str]|str = None, seed_range: int = 0):
    if isinstance(train, str): train = set(train.split(','))

    train_sgd_student = 'sgd' in train if train else True
    train_adam_student = 'adam' in train if train else True
    train_pbdca_student = 'pbdca' in train if train else True

    in_dim = 20
    hidden_dim = 24
    out_dim = 16
    num_layers = 4
    batch_size = 8
    num_epochs = 1

    arch = f'{in_dim}x' + 'x'.join(str(hidden_dim) for _ in range(num_layers-2)) + f'x{out_dim}'

    experiment = Experiment.coerce(
        kind='teacher',
        name=f'reg-{arch}-seed-{seed}',
        teacher=dict(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kind='simple-mlp',
        ),
        in_dim=in_dim,
        out_dim=out_dim,
        in_chunk_size=102400,
        chunks_per_epoch=1
    )

    if train_sgd_student:

        experiment.add_experiment(
            kind='sgd',
            name='sgd',
            params={
                'lr': 1.5e-3,
                # 'weight_decay': 0.01,
                'momentum': 0.9,
            },
            student=dict(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                kind='simple-mlp',
            ),
            num_epochs=num_epochs,
            batch_size=batch_size
        )

    if train_adam_student:

        experiment.add_experiment(
            kind='sgd',
            name='adam',
            params={
                'lr': 1.5e-3,
                # 'weight_decay': 0.01,
                # 'momentum': 0.9,
            },
            student=dict(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                kind='simple-mlp',
            ),
            num_epochs=num_epochs,
            batch_size=batch_size
        )

    if train_pbdca_student:

        def add_pbdca_experiment(manual_seed: int = None):

            hyperparams = {
                'generator.num_probes': 12,
                'num_perturbations': 32,
                'scheduler.exploration_radius': 0.4,
                'scheduler.exploration_radius_decay': 0.8,
                'scheduler.decay_every': 3000,
                'creditor.lr': 0.05,
                'creditor.lr_decay': 1.0,
                'creditor.decay_every': 3000,
                'generate_every': 20,
            }
            hyperparam_defaults = {}

            student_args = dict(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                layer_kind='linear.probed',
                kind='simple-mlp',
            )

            sweeps = [
                # {'generate_every': 1000},
                # {'generate_every': 500},
                # {'generate_every': 100},
                # {'generate_every': 50},
                # {'generate_every': 25, 'keep': 4},
                # {'generate_every': 25},
                # {'generate_every': 25, 'schedule_every': 2},
                # {'generate_every': 25, 'generator.num_probes': 12, 'num_perturbations': 32},
                # {'generate_every': 25, 'generator.num_probes': 12, 'num_perturbations': 48},
                # {'generate_every': 25, 'generator.num_probes': 16, 'num_perturbations': 32},
                # {'generate_every': 25, 'generator.num_probes': 16, 'num_perturbations': 64},

                # {'top_k': 2, 'generate_every': 50},
                # {'top_k': 2, 'generate_every': 25},
                # {'schedule_every': 1},
                # {'schedule_every': 2},
                # {'schedule_every': 3},
                # {'schedule_every': 4},
                # {'top_k': 1, 'generator.num_probes': 12, 'num_perturbations': 32},
                # {'top_k': 2, 'generator.num_probes': 12, 'num_perturbations': 32},
                # {'top_k': 3, 'generator.num_probes': 12, 'num_perturbations': 32},
                # {'top_k': 4, 'generator.num_probes': 12, 'num_perturbations': 32},
                # {'generator.num_probes': 12, 'num_perturbations': 48},
                # {'lr': 0.10},
                # {'lr': 0.05},
                # {'lr': 0.02},
                # {'lr': 0.01},
                # {'lr': 0.005},
                # {'lr': 0.001},
            ]

            name = 'pbdca' if manual_seed is None else f'pbdca-seed-{manual_seed}'

            if sweeps:
                experiment.add_experiment(
                    kind='sweep',
                    name='sweep',
                    child=dict(
                        kind='probed',
                        name=name,
                        params=hyperparams,
                        param_defaults=hyperparam_defaults,
                        student=student_args,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        seed=manual_seed,
                    ),
                    sweeps=sweeps,
                )
            else:
                experiment.add_experiment(
                    kind='probed',
                    name=name,
                    params=hyperparams,
                    param_defaults=hyperparam_defaults,
                    student=student_args,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    seed=manual_seed,
                )

        if seed_range > 0:
            for s in range(seed, seed+seed_range):
                add_pbdca_experiment(s)
        else:
            add_pbdca_experiment(seed)

    experiment.run()

    return 0


def run_tmaze_experiment(seed: int = 10, train: Container[str]|str = None, seed_range: int = 0,
                         delay: int = 10, parent: Experiment = None):
    if isinstance(train, str): train = set(train.split(','))

    train_sgd_student = 'sgd' in train if train else True
    train_adam_student = 'adam' in train if train else True
    train_pbdca_student = 'pbdca' in train if train else True

    in_dim = 2
    hidden_dim = 16
    out_dim = 1
    batch_size = 8
    num_epochs = 1

    arch = f'tmaze-{in_dim}x{hidden_dim}x{out_dim}'

    experiment = Experiment.coerce(
        kind='tmaze',
        parent=parent,
        name=f'{arch}-delay-{delay}-seed-{seed}',
        in_dim=in_dim,
        out_dim=out_dim,
        in_chunk_size=102400,
        delay=delay,
        seed=seed,
        chunks_per_epoch=2
    )

    student_kind = 'simple-rnn'

    if train_sgd_student:

        experiment.add_experiment(
            kind='sgd',
            name='sgd',
            params={
                'lr': 1.5e-3,
                # 'weight_decay': 0.01,
                'momentum': 0.9,
            },
            student=dict(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                kind=student_kind,
            ),
            num_epochs=num_epochs,
            batch_size=batch_size
        )

    if train_adam_student:

        experiment.add_experiment(
            kind='sgd',
            name='adam',
            params={
                'lr': 1.5e-3,
                # 'weight_decay': 0.01,
                # 'momentum': 0.9,
            },
            student=dict(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                kind=student_kind,
            ),
            num_epochs=num_epochs,
            batch_size=batch_size
        )

    if train_pbdca_student:

        def add_pbdca_experiment(manual_seed: int = None):
            hyperparams = {
                'generator.num_probes': 12,
                'num_perturbations': 32,
                'scheduler.exploration_radius': 0.8,
                'scheduler.exploration_radius_decay': 0.8,
                'scheduler.decay_every': 3000,
                'creditor.lr': 0.15,
                'creditor.lr_decay': 0.9,
                'creditor.decay_every': 3000,
                'generate_every': 20,
            }
            hyperparam_defaults = {
                'generator.rank': 1,
                'creditor.lr_decay': 1.0,
                'scheduler.exploration_radius_decay': 1.0,
                'scheduler.decay_every': 1000,
                'creditor.decay_every': 1000,
                'schedule_every': 1,
            }

            student_args = dict(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                projection_kind='linear.probed',
                kind=student_kind,
            )

            sweeps = [
                # {'generate_every': 1000},
                # {'generate_every': 500},
                # {'generate_every': 100},
                # {'generate_every': 50},
                # {'generate_every': 25, 'keep': 4},
                # {'generate_every': 25},
                # {'generate_every': 25, 'schedule_every': 2},
                # {'generate_every': 25, 'generator.num_probes': 12, 'num_perturbations': 32},
                # {'generate_every': 25, 'generator.num_probes': 12, 'num_perturbations': 48},
                # {'generate_every': 25, 'generator.num_probes': 16, 'num_perturbations': 32},
                # {'generate_every': 25, 'generator.num_probes': 16, 'num_perturbations': 64},

                # {'top_k': 2, 'generate_every': 50},
                # {'top_k': 2, 'generate_every': 25},
                # {'schedule_every': 1},
                # {'schedule_every': 2},
                # {'schedule_every': 3},
                # {'schedule_every': 4},
                # {'top_k': 1, 'generator.num_probes': 12, 'num_perturbations': 32},
                # {'top_k': 2, 'generator.num_probes': 12, 'num_perturbations': 32},
                # {'top_k': 3, 'generator.num_probes': 12, 'num_perturbations': 32},
                # {'top_k': 4, 'generator.num_probes': 12, 'num_perturbations': 32},
                # {'generator.num_probes': 12, 'num_perturbations': 48},
                # {'creditor.lr': 0.20},
                # {'creditor.lr': 0.10},
                # {'creditor.lr': 0.05},
                # {'creditor.lr': 0.02},
                # {'creditor.lr': 0.01},
                # {'creditor.lr': 0.005},
                # {'creditor.lr': 0.001},
            ]

            name = 'pbdca' if manual_seed is None else f'pbdca-seed-{manual_seed}'

            if sweeps:
                experiment.add_experiment(
                    kind='sweep',
                    name='sweep',
                    child=dict(
                        kind='probed',
                        name=name,
                        params=hyperparams,
                        param_defaults=hyperparam_defaults,
                        student=student_args,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        seed=manual_seed,
                    ),
                    sweeps=sweeps,
                )
            else:
                experiment.add_experiment(
                    kind='probed',
                    name=name,
                    params=hyperparams,
                    param_defaults=hyperparam_defaults,
                    student=student_args,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    seed=manual_seed,
                )

        if seed_range > 0:
            for s in range(seed, seed+seed_range):
                add_pbdca_experiment(s)
        else:
            add_pbdca_experiment(seed)

    if parent is None:
        experiment.run()
    else:
        parent.experiments.append(experiment)

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
