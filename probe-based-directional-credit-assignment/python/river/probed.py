#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from tensile.common import *
from tensile.nn import CompiledModule, Module, ModuleArgs
from tensile.nn.common import Activation
from tensile.nn.layers import Linear
from tensile.nn.module import Functional
from tensile.optim import Optimizer
from tensile.util.buffer import ArrayBuffer

from river.experiment import Experiment, StudentTrainingExperiment


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

    def get_update(self, credit: Array, lr: float) -> Array:
        raise NotImplementedError()

    def _repr_args(self, **options) -> str:
        return f'{self.in_dim} -> {self.out_dim}, k={self.k}'


@provides(Probes, 'lora')
class LoRAProbes(Probes):

    __slots__ = ('rank', 'left', 'right')

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

    # noinspection PyPep8Naming
    def build_probe(self, assignment: Array, scales: Array = None) -> Probe:
        left = self.left[assignment]
        right = self.right[assignment]
        if scales is None:
            scale = ten.ones(assignment.shape, dtype=self.dtype)
        else:
            assert scales.shape == assignment.shape, "Shape mismatch"
            scale = scales

        P = left.shape[0]

        ten.eval(left, right)

        def probe(x: Array) -> Array:
            ten.eval(x)
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
            ten.eval(out)
            return out

        return probe

    def get_update(self, credit: Array, lr: float) -> Array:
        expanded = ten.matmul(self.left[1:], self.right[1:])
        ten.eval(expanded)
        return ten.sum(lr * expanded * credit[1:, None, None], axis=0)

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

    def perturb(self, probes: Array, scales: Array = None) -> None:
        raise NotImplementedError()

    def update_from_credit(self, credit: Array, lr: float) -> Array:
        raise NotImplementedError()


class ProbeGenerator(Object):

    __slots__ = ('num_probes', 'orthogonal')

    num_probes: Annotated[int, field(
        doc="The number of probes",
        default=10,
    )]
    orthogonal: Annotated[bool, field(
        doc="Whether to use orthogonal probes",
        default=False,
    )]

    def generate(self, module: ProbeableModule, k: int = None) -> Probes:
        raise NotImplementedError()


@provides(ProbeGenerator, 'lora')
class LoRAProbeGenerator(ProbeGenerator):

    __slots__ = ('rank', )

    rank: Annotated[int, field(
        doc="The rank of the adapters",
        default=1,
    )]

    def generate(self, module: ProbeableModule, k: int = None) -> Probes:
        if k is None: k = self.num_probes
        r = self.rank
        n = module.in_dim
        m = module.out_dim

        j = max(r * k, r)
        lv = ten.random.normal(shape=(1, n, j))
        rv = ten.random.normal(shape=(1, j, m))

        if self.orthogonal:
            lv = lv / ten.norm(lv, axis=1, keepdims=True)
            rv = rv / ten.norm(rv, axis=2, keepdims=True)

            ten.eval(lv, rv)

            def reject(xh: Array, xi: Array) -> Array:
                xdot = ten.sum(xi * xh)
                xh = xh - xdot * xi
                return normalize(xh)

            for h in range(1, j):
                lvh = lv[0, :, h]
                rvh = rv[0, h, :]
                for i in range(h):
                    lvi = lv[0, :, i]
                    rvi = rv[0, i, :]
                    lvh = reject(lvh, lvi)
                    rvh = reject(rvh, rvi)
                lv[0, :, h] = lvh
                rv[0, h, :] = rvh

        lefts = [ten.zeros((1, n, r))]
        rights = [ten.zeros((1, r, m))]
        for _ in range(k):
            li = ten.random.permutation(j)[:r]
            ri = ten.random.permutation(j)[:r]
            lt = lv[:, :, li]
            rt = rv[:, ri, :]
            lt = normalize(lt)
            rt = normalize(rt)
            lefts.append(lt)
            rights.append(rt)
        left = ten.concatenate(lefts, axis=0)
        right = ten.concatenate(rights, axis=0)

        ten.eval(left, right)

        probes = Probes.coerce(k=k, in_dim=n, out_dim=m, rank=r, left=left, right=right, kind='lora')

        return probes


@provides(Linear, 'probed')
class ProbedLinear(Linear, ProbeableModule):

    __slots__ = ('probes', 'probe')

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

    def perturb(self, probes: Array, scales: Array = None):
        # noinspection PyTypeChecker
        if ten.all(probes == 0):
            self.probe = None
        else:
            self.probe = self.probes.build_probe(probes, scales=scales)
        self.calls.clear()
        self.compile()

    def update_from_credit(self, credit: Array, lr: float) -> Array:
        update = self.probes.get_update(credit, lr=lr)
        self.weight += ten.swapaxes(update, 0, 1)
        return update

    def get_probe(self) -> Probe:
        if self.probe is None:
            raise ValueError("Probe function is None, cannot get probe")
        return self.probe

    def build_call(self, mode: Module.Mode, **options) -> Functional:
        if self.is_probing:
            probe = self.get_probe()

            def call(x: Array) -> Array:
                ten.eval(x)

                y = ten.matmul(x, self.weight.T)

                ten.eval(y)

                out = y + probe(x)

                ten.eval(out)

                return out
        else:
            def call(x: Array) -> Array:
                ten.eval(x)

                out = ten.matmul(x, self.weight.T)

                ten.eval(out)

                return out

        return call

    Args = Linear.Args


class PerturbationMatrix(Object):

    __slots__ = ('probes', 'scales')

    probes: Annotated[Array, field(
        doc="The current probe indexes"
    )]
    scales: Annotated[Array, field(
        doc="The scales for the current probes"
    )]

    def postinit(self, spec: Spec):
        super().postinit(spec)
        ten.eval(self.probes, self.scales)

    @property
    def num_perturbations(self) -> int:
        return self.probes.shape[-1]

    @property
    def num_modules(self) -> int:
        return self.probes.shape[0]


class PerturbationScheduler(Object):

    __slots__ = ('modules', 'perturbations', 'history', 'keep', 'kept', 'force_null_perturbation')

    modules: Annotated[list[ProbeableModule], field(
        doc="The probable modules to be scheduled",
        default_factory=list,
    )]
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

    @property
    def num_modules(self) -> int:
        return len(self.modules)

    def perturb_modules(self, perturbations: PerturbationMatrix = None):
        if perturbations is None: perturbations = self.perturbations

        for m, module in enumerate(self.modules):
            module_probes = perturbations.probes[m]
            module_scales = perturbations.scales[m]
            module.perturb(module_probes, module_scales)

    def set_perturbations(self, perturbations: PerturbationMatrix):
        self.perturbations = perturbations
        self.history.append(perturbations)
        self.perturb_modules(perturbations)

    def record_advantage(self, advantage: Array):
        if self.keep > 0:
            self.kept = ten.argsort(advantage[1:])[:self.keep] + 1

    def record_credit(self, m: int, credit: Array):
        pass

    def schedule(self, num_perturbations: int) -> PerturbationMatrix:
        probes = []
        scales = []
        kept = self.kept
        if kept is not None:
            perturbations = self.perturbations
            old_probes = perturbations.probes
            old_scales = perturbations.scales
            ten.eval(kept, old_probes, old_scales)
            new_perturbations = num_perturbations - kept.shape[0]
            for m, module in enumerate(self.modules):
                new_probes, new_scales = self.schedule_module(module, new_perturbations)
                new_probes = ten.concatenate([new_probes, old_probes[m, kept]])
                new_scales = ten.concatenate([new_scales, old_scales[m, kept]])
                probes.append(new_probes)
                scales.append(new_scales)
        else:
            for module in self.modules:
                new_probes, new_scales = self.schedule_module(module, num_perturbations)
                probes.append(new_probes)
                scales.append(new_scales)


        perturbations = PerturbationMatrix(probes=ten.stack(probes, axis=0),
                                           scales=ten.stack(scales, axis=0))

        self.set_perturbations(perturbations)
        return perturbations

    def schedule_module(self, module: ProbeableModule, num_perturbations: int) -> tuple[Array, Array]:
        raise NotImplementedError()


@provides(PerturbationScheduler, 'random')
class RandomPerturbationScheduler(PerturbationScheduler):

    __slots__ = ('magnitude', 'p', 'mu', 'sigma')

    magnitude: Annotated[float, field(
        doc="The magnitude of the probe perturbation",
        default=1.0
    )]
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

    def schedule_module(self, module: ProbeableModule, num_perturbations: int) -> tuple[Array, Array]:
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
        scales = self.magnitude * ten.exp(self.mu + self.sigma * ten.random.normal(shape=(num_perturbations,)))
        scales = ten.clip(scales, 0.005, 0.5)
        scale_sign = (2. * ten.random.bernoulli(0.5, shape=(num_perturbations,)) - 1.0)

        scales = ten.where(probes == 0, 0., scales * scale_sign)
        ten.eval(probes, scales)
        if self.force_null_perturbation:
            probes[0] = 0
            scales[0] = 0.
        return probes, scales


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
    def out_dim(self) -> int:
        return self.output_proj.out_dim

    def build_call(self, mode: Module.Mode, **options) -> Callable:
        activation = self.activation
        input_proj = self.input_proj
        hidden_dim = input_proj.out_dim
        hidden_proj = self.hidden_proj
        output_proj = self.output_proj

        def call(x: Array) -> Array:
            h = ten.zeros((*x.shape[:-1], hidden_dim))
            for t in range(x.shape[-1]):
                xp_t = input_proj(x[..., t, :])
                hp_t = hidden_proj(h)
                h = activation(xp_t + hp_t)
            out = output_proj(h)
            return out
        return call

    def _extra_structure(self) -> str:
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}'



class Creditor(Object):

    __slots__ = ('lr', )

    lr: Annotated[float, field(
        doc="The learning rate for credit assignment",
        default=0.05,
    )]

    def assign_credit(self, step: int, scheduler: PerturbationScheduler, losses: Array):
        raise NotImplementedError()


@provides(Creditor, 'top_k')
class TopKCreditor(Creditor):

    __slots__ = ('top_k', )

    top_k: Annotated[int, field(
        doc="The number of perturbation to use for credit assignment",
        default=1,
    )]

    def assign_credit(self, step: int, scheduler: PerturbationScheduler, losses: Array):
        top_k = self.top_k
        lr = self.lr
        advantage_eps = 1e-6
        perturbations = scheduler.perturbations

        best = ten.argsort(losses)[:top_k]

        best_probes = perturbations.probes[:, best]

        baseline_loss = losses[0]
        advantage = ten.maximum(0., baseline_loss - losses)

        advantage = advantage / (ten.sum(advantage[best], keepdims=True) + advantage_eps)

        scheduler.record_advantage(advantage)

        for m, mod in enumerate(scheduler.modules):
            if mod.is_probing:
                credit = ten.zeros((mod.probes.k + 1,))
                i = best_probes[m]
                for k in range(top_k):
                    p = best[k]
                    probe_credit = perturbations.scales[m, p] * advantage[p]
                    credit[i[k]] += probe_credit
                ten.eval(credit)
                mod.update_from_credit(credit, lr=lr)
                scheduler.record_credit(m, credit)


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

        experiment = self.parent or self

        bucket_desc = self.descriptor

        loss_fn = self.loss_fn
        header = self.header
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
            for b in experiment.batch_data(self.batch_size):
                loss = step(b)
                # ten.eval(loss)
                losses.append(loss[None])
                if s % report_every == 0:
                    print(f'Step: {s:5d}  Loss: {loss:.6f}')
                s += 1

        for inputs, targets in experiment.batch_data(1024):
            outputs = student(inputs)
            loss = loss_fn(outputs, targets)
            print(f'Student[{self.descriptor}] Final loss: {loss:.6f}') # 0.028599 0.083547
            break

        experiment.add_metric(f'{bucket_desc}:loss', losses)

        return s-1


@provides(Experiment, 'probed')
class ProbedTrainingExperiment(StudentTrainingExperiment):

    __slots__ = ()

    @staticmethod
    def perturbation_loss_fn(outputs: Array, targets: Array) -> Array:
        return ten.mean(
            ten.square(outputs - targets),
            axis=(1, 2)
        )

    def _coerce_student(self, spec: Any) -> Module:
        return self.get_module(spec, self.work_dir / "probed.safetensors")

    def train(self) -> int:

        probe_rank = self.get_param('probe_rank')
        num_probes = self.get_param('num_probes')
        num_perturbations = self.get_param('num_perturbations')
        probe_scale = self.get_param('probe_scale')
        top_k = self.get_param('top_k')
        lr = self.get_param('lr')
        keep = self.get_param('keep', 0)
        lr_decay = self.get_param('lr_decay')
        probe_scale_decay = self.get_param('probe_scale_decay')
        decay_every = self.get_param('decay_every')
        schedule_every = self.get_param('schedule_every', default=1)
        generate_every = self.get_param('generate_every', default=100000000000)

        scheduler_kind = self.get_param('scheduler_kind', default='random')
        generator_kind = self.get_param('generator_kind', default='lora')
        creditor_kind = self.get_param('creditor_kind', default='top_k')

        student = self.student

        modules = []
        for path, module in tree.traverse(student, include=tree.value_predicate(predicates.is_instance(ProbeableModule))):
            modules.append(module)

        generator = ProbeGenerator.coerce(rank=probe_rank, num_probes=num_probes, kind=generator_kind)

        scheduler = PerturbationScheduler.coerce(
            magnitude=probe_scale,
            kind=scheduler_kind,
            modules=modules,
            keep=keep,
        )

        creditor = Creditor.coerce(lr=lr, top_k=top_k, kind=creditor_kind)

        for module in modules:
            module.probes = generator.generate(module)

        perturbations = scheduler.schedule(num_perturbations)

        experiment = self.parent or self
        header = self.header

        def report_perturbations():
            probes = perturbations.probes
            scales = perturbations.scales

            header('probes')
            print(ten.sign(scales) * probes)
            header('scales')
            print(scales)

        report_perturbations()

        report_every = self.report_every

        step = 1

        perturbation_loss_fn = self.perturbation_loss_fn

        losses = ArrayBuffer()
        header(f'Student[{self.descriptor}] Training')

        for e in range(self.num_epochs):
            header(f'Student[{self.descriptor}] Epoch: {e+1:2d}')
            for inputs, targets in experiment.batch_data(self.batch_size):
                inputs = ten.broadcast_to(inputs[None], (num_perturbations, *inputs.shape))
                outputs = student(inputs)
                perturbation_losses = perturbation_loss_fn(outputs, targets[None])

                creditor.assign_credit(step, scheduler, perturbation_losses)

                loss = perturbation_losses[0]
                losses.append(loss[None])

                if step % generate_every == 0:
                    for module in modules:
                        module.probes = generator.generate(module)

                if step % schedule_every == 0:
                    perturbations = scheduler.schedule(num_perturbations)

                if step % report_every == 0:
                        print(f'Student[{self.descriptor}] Step: {step:5d}  Loss: {loss:.6f} Learning Rate: {lr:.4f}')

                step += 1

        for inputs, targets in experiment.batch_data(1024):
            inputs = ten.broadcast_to(inputs[None], (num_perturbations, *inputs.shape))
            outputs = student(inputs)
            perturbation_losses = perturbation_loss_fn(outputs, targets[None])
            loss = perturbation_losses[0]
            ten.eval(perturbation_losses)
            print(f'Student[{self.descriptor}] Final loss: {loss:.6f}') # 0.028599 0.083547
            break

        experiment.add_metric(f'{self.descriptor}:loss', losses)

        return step-1


def moving_average(x: Array, window: int, axis: int = -1) -> Array:
    x = ten.swapaxes(x, axis, -1)

    c = ten.cumsum(x, axis=-1)
    c[..., window:] = c[..., window:] - c[..., :-window]
    y = c[..., window - 1:] / window

    return ten.swapaxes(y, -1, axis)


def main():

    train_sgd_student = True
    train_adam_student = False
    train_pbdca_student = True

    in_dim = 20
    hidden_dim = 24
    out_dim = 16
    num_layers = 4
    batch_size = 8
    num_epochs = 1
    seed = 12345678

    arch = f'{in_dim}x' + 'x'.join(str(hidden_dim) for _ in range(num_layers-2)) + f'x{out_dim}'

    experiment = Experiment.coerce(
        kind='teacher',
        name=f'pbdca-{arch}',
        teacher=dict(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kind='simple-mlp',
        ),
        in_dim=in_dim,
        out_dim=out_dim,
        in_chunk_size=10240,
        chunks_per_epoch=10
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

        hyperparams = {
            'probe_rank': 1,
            'num_probes': 12,
            'num_perturbations': 32,
            'probe_scale': 0.2,
            'probe_scale_decay': 1.0,
            'top_k': 1,
            'lr': 0.05,
            'lr_decay': 1.0,
            'decay_every': 1000,
            # 'schedule_every': 2,
        }
        hyperparam_defaults = {
            'probe_rank': 1,
            'lr_decay': 1.0,
            'probe_scale_decay': 1.0,
            'decay_every': 1000,
            'schedule_every': 1,
        }

        student_args = dict(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            layer_kind='linear.probed',
            kind='simple-mlp',
        )

        sweeps = [
            # [('generate_every', 1000)],
            # [('generate_every', 500)],
            # [('generate_every', 100)],
            # [('generate_every', 50)],
            # [('generate_every', 25), ('keep', 4)],
            [('generate_every', 25)],
            # [('generate_every', 25), ('schedule_every', 2)],
            # [('generate_every', 25), ('num_probes', 12), ('num_perturbations', 32)],
            # [('generate_every', 25), ('num_probes', 12), ('num_perturbations', 48)],
            # [('generate_every', 25), ('num_probes', 16), ('num_perturbations', 32)],
            # [('generate_every', 25), ('num_probes', 16), ('num_perturbations', 64)],

            # [('top_k', 2), ('generate_every', 50)],
            # [('top_k', 2), ('generate_every', 25)],
            # [('schedule_every', 1)],
            # [('schedule_every', 2)],
            # [('schedule_every', 3)],
            # [('schedule_every', 4)],
            # [('top_k', 1), ('num_probes', 12), ('num_perturbations', 32)],
            # [('top_k', 2), ('num_probes', 12), ('num_perturbations', 32)],
            # [('top_k', 3), ('num_probes', 12), ('num_perturbations', 32)],
            # [('top_k', 4), ('num_probes', 12), ('num_perturbations', 32)],
            # [('num_probes', 12), ('num_perturbations', 48)],
            # [('lr', 0.10)],
            # [('lr', 0.05)],
            # [('lr', 0.02)],
            # [('lr', 0.01)],
            # [('lr', 0.005)],
            # [('lr', 0.001)],
        ]

        if sweeps:
            experiment.add_experiment(
                kind='sweep',
                name='sweep',
                child=dict(
                    kind='probed',
                    name='pbdca',
                    params=hyperparams,
                    param_defaults=hyperparam_defaults,
                    student=student_args,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    seed=seed,
                ),
                sweeps=sweeps,
            )
        else:
            experiment.add_experiment(
                kind='probed',
                name='pbdca',
                params=hyperparams,
                param_defaults=hyperparam_defaults,
                student=student_args,
                num_epochs=num_epochs,
                batch_size=batch_size,
                seed=seed,
            )

    experiment.run()



if __name__ == '__main__':
    import sys

    sys.exit(main())
