#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import time
from pathlib import Path

from tensile.common import *
from tensile.nn import Module
from tensile.util.buffer import ArrayBuffer


class Experiment(Object):

    __slots__ = ('name', 'descriptor', 'parent', 'params', 'param_defaults', 'experiments', 'metrics', 'seed', 'work_dir')

    name: Annotated[str, field(
        doc='The name of the experiment'
    )]
    descriptor: Annotated[str, field(
        doc='The descriptor of the experiment'
    )]
    parent: Annotated[Optional['Experiment'], field(
        doc='The parent experiment, if any'
    )]
    params: Annotated[dict[str, Any], field(
        default_factory=dict,
    )]
    param_defaults: Annotated[dict[str, Any], field(
        default_factory=dict
    )]
    experiments: Annotated[list['Experiment'], field(
        doc='The list of sub-experiments for this experiment, if any',
        default_factory=list,
    )]
    metrics: Annotated[dict[str, Array], field(
        default_factory=dict
    )]
    seed: Annotated[Optional[int], field(
        doc='The random seed to set before running the bucket',
    )]

    work_dir: Annotated[Path, field()]

    def _lazy_work_dir(self) -> Path:
        if parent := self.parent:
            return parent.work_dir
        return Path('./work')

    def _lazy_descriptor(self) -> str:
        desc_params = self.descriptor_params
        if desc_params:
            return self.name + '(' + ', '.join(f'{k}={v}' for k, v in desc_params.items()) + ')'
        return self.name

    @property
    def descriptor_params(self) -> dict[str, Any]:
        params = self.params
        bucket_keys = params.keys()
        defaults = self.param_defaults
        return {
            default_hyperparam_abbrevs.get(pk, pk): params[pk]
            for pk in bucket_keys
            if pk in params and (pk not in defaults or params[pk] != defaults[pk])
        }

    def get_param(self, name: str, default: Any = None) -> Any:
        if params := self.params:
            if name in params:
                return params[name]
        if default is None:
            default = self.param_defaults.get(name)
        if parent := self.parent:
            return parent.get_param(name, default)
        return default

    def collect_params(self, name: str, params: dict[str, Any]) -> None:
        if defaults := self.param_defaults.get(name):
            if isinstance(defaults, dict):
                for k, v in defaults.items():
                    params.setdefault(k, v)
        if own := self.params.get(name):
            if isinstance(own, dict):
                params.update(own)

    def get_params(self, name: str) -> dict[str, Any]:
        params = {}
        if parent := self.parent:
            parent.collect_params(name, params)
        self.collect_params(name, params)
        return params

    def add_experiment(self, **spec) -> 'Experiment':
        experiment = Experiment.coerce(parent=self, **spec)
        self.experiments.append(experiment)
        return experiment

    def add_metric(self, name: str, metric: Array|ArrayBuffer):
        if isinstance(metric, ArrayBuffer):
            self.metrics[name] = metric.fetch()
        else:
            self.metrics[name] = metric

    def batch_data(self, b: int) -> Iterable[tuple[Array, Array]]:
        if parent := self.parent:
            return parent.batch_data(b)
        raise NotImplementedError()

    def get_module(self, spec: Any, path: Path = None) -> Module:
        if spec is None:
            raise ValueError('module spec cannot be None')
        if isinstance(spec, Module):
            module = spec
        elif isinstance(spec, dict):
            module = Module.from_args(**spec)
            if path is not None:
                if path.exists():
                    module.load_weights(path)
                else:
                    ten.save_tensors(path, tree.flatdict(module.parameters()))
        else:
            raise ValueError(f'unknown module spec type: {type(spec)}')
        return module

    def run(self):
        self.start()
        self.run_self()
        self.run_experiments()
        self.finish()

    def start(self):
        self.metrics.clear()

    def run_self(self):
        if self.seed is not None:
            ten.random.seed(self.seed)

    def run_experiments(self):
        for exp in self.experiments:
            exp.run()

    def finish(self):
        if metrics := self.metrics:
            from tensile.util import chart

            work_dir = self.work_dir
            name = self.name

            chart.plot_metrics(metrics, out=work_dir / f"{name}-metrics.png", smoothing=0.9)
            grid = {}
            for k, v in metrics.items():
                sep = k.index(':')
                if sep < 0:
                    exp = m = k
                else:
                    exp = k[:sep]
                    m = k[sep+1:]
                if exp in grid:
                    grid[exp][m] = v
                else:
                    grid[exp] = {m: v}
            chart.plot_grid(grid, out=work_dir / f"{name}-grid.png", smoothing=0.9)
            ten.save_tensors(work_dir / f"{name}-metrics.safetensors", metrics)

        self.print(f'\nFinished {self.descriptor}.')

    @staticmethod
    def header(h: str, w: int = 120):
        print()
        w -= len(h)
        if w < 0:
            print(h)
        else:
            p = w//2
            s = p+1 if w % 2 else p
            print("="*p, h, "="*s)

    @staticmethod
    def print(*args):
        print(*args)


class CachedInputExperiment(Experiment):

    __slots__ = ('in_chunk_size', 'in_dim', 'chunks_per_epoch')

    in_chunk_size: Annotated[int, field(
        doc='The size of each input chunk',
        default=1024,
    )]
    in_dim: Annotated[int, field(
        doc='The dimensionality of each input vector',
    )]
    chunks_per_epoch: Annotated[int, field(
        doc='The number of chunks to generate per epoch',
        default=10,
    )]

    def get_input_chunk(self, i: int, name: str = 'inputs') -> Array:
        input_file = self.work_dir / f"{name}-{i}.safetensors"
        in_dim = self.in_dim
        if input_file.exists():
            arrays = ten.load_tensors(input_file)
        else:
            arrays = {
                'input': ten.random.normal(scale=2., shape=(self.in_chunk_size, in_dim))
            }
            ten.save_tensors(input_file, arrays)
        ten.eval(arrays)
        inputs = arrays['input']
        if inputs.shape[-1] != in_dim:
             raise ValueError(f'Expected input chunk to have shape (..., {in_dim}), got: {inputs.shape}')
        return inputs


@provides(Experiment, 'teacher')
class TeacherExperiment(CachedInputExperiment):

    __slots__ = ('teacher',)

    teacher: Annotated[Module, field(
        doc='The teacher model to use',
    )]

    def _coerce_teacher(self, spec: Any) -> Module:
        return self.get_module(spec, self.work_dir / 'teacher.safetensors')

    def batch_data(self, b: int) -> Iterable[tuple[Array, Array]]:
        teacher = self.teacher
        chunk_size = self.in_chunk_size
        for i in range(self.chunks_per_epoch):
            chunk = self.get_input_chunk(i)
            for s in range(0, chunk_size, b):
                inputs = chunk[s:s + b]
                outputs = teacher(inputs)
                yield inputs, outputs

    def start(self):
        super().start()

        self.header('teacher')
        self.print(self.teacher.structure())


# class ExperimentBucket(Object):
#
#     __slots__ = ('experiment', 'name', 'descriptor', 'params', 'param_defaults', 'work_dir', 'report_every', 'seed')
#
#     name: Annotated[str, field(
#         doc='The name of the experiment'
#     )]
#     descriptor: Annotated[str, field(
#         doc='The descriptor of the experiment'
#     )]
#     experiment: Annotated[Experiment, field(
#         doc='The experiment this bucket belongs to'
#     )]
#     params: Annotated[dict[str, Any], field(
#         default_factory=dict
#     )]
#
#     param_defaults: Annotated[dict[str, Any], field(
#         default_factory=dict
#     )]
#
#     work_dir: Annotated[Path, field()]
#     report_every: Annotated[int, field(
#         doc='The number of steps between each report',
#         default=100,
#     )]
#     seed: Annotated[Optional[int], field(
#         doc='The random seed to set before running the bucket',
#     )]
#
#     def _lazy_work_dir(self) -> Path:
#         return self.experiment.work_dir
#
#     def _lazy_descriptor(self) -> str:
#         desc_params = self.descriptor_params
#         if desc_params:
#             return self.name + '-' + '-'.join(f'{k}={v}' for k, v in desc_params.items())
#         return self.name
#
#     @property
#     def descriptor_params(self) -> dict[str, Any]:
#         params = self.params
#         bucket_keys = params.keys()
#         defaults = self.param_defaults
#         return {
#             default_hyperparam_abbrevs.get(pk, pk): params[pk]
#             for pk in bucket_keys
#             if pk in params and (pk not in defaults or params[pk] != defaults[pk])
#         }
#
#     def get_param(self, name: str) -> Any:
#         if params := self.params:
#             if name in params:
#                 return params[name]
#         return self.param_defaults.get(name)
#
#     def run(self):
#         if self.seed is not None:
#             ten.random.seed(self.seed)
#
#         start = time.perf_counter()
#         steps = self._run()
#         end = time.perf_counter()
#         ksteps = steps/1000.
#         self.print(f'time elapsed: {(end - start)/ksteps:.4f} seconds per thousand steps')
#
#     def _run(self) -> int:
#         raise NotImplementedError()
#
#     def print(self, *args):
#         print(f'Bucket[{self.descriptor}]:', *args)
#
#     header = staticmethod(Experiment.header)

Sweeps = Iterable[list[tuple[str, Any]]]
Sweeper = Callable[[], Sweeps]


@provides(Experiment, 'sweep')
class SweepExperiment(Experiment):

    __slots__ = ('child', 'sweeps')

    child: Annotated[dict[str, Any], field(
        doc='The specification for the child experiment to sweep over',
        required=True,
    )]
    sweeps: Annotated[Sweeper, field(
        doc='The parameter sweep to perform',
        required=True,
    )]

    def _coerce_sweeps(self, spec: Any) -> Sweeper:
        if callable(spec):
            return spec
        elif isinstance(spec, Iterable):
            return lambda: spec
        else:
            raise ValueError(f'Invalid sweep specification: {spec}')

    def run_self(self):
        super().run_self()
        child_spec = self.child.copy()
        orig_params = child_spec.pop('params', {})
        for sweep in self.sweeps():
            child_params = orig_params.copy()
            for param, value in sweep:
                child_params[param] = value

            child = Experiment.coerce(parent=self, params=child_params, **child_spec)
            self.experiments.append(child)
            child.run()

    def run_experiments(self):
        # They were already run in run_self....
        pass

    def add_metric(self, name: str, metric: Array | ArrayBuffer):
        if parent := self.parent:
            parent.add_metric(name, metric)
        else:
            super().add_metric(name, metric)

    @staticmethod
    def sweep_values(param: str, values: Iterable[Any]) -> Sweeper:
        def sweep():
            for value in values:
                yield [(param, value)]
        return sweep

    @staticmethod
    def sweep_cross(a: Sweeper, b: Sweeper) -> Sweeper:
        def sweep():
            for sa in a():
                for sb in b():
                    yield sa + sb
        return sweep


class TrainingExperiment(Experiment):

    __slots__ = ('num_epochs', 'batch_size', 'report_every')

    num_epochs: Annotated[int, field(
        doc='The number of epochs to train',
        default=1,
    )]
    batch_size: Annotated[int, field(
        doc='The batch size to use for training',
        default=8,
    )]
    report_every: Annotated[int, field(
        doc='The number of steps between each report',
        default=100,
    )]

    def run_self(self):
        super().run_self()

        start = time.perf_counter()
        steps = self.train()
        end = time.perf_counter()
        ksteps = steps/1000.
        self.print(f'time elapsed: {(end - start)/ksteps:.4f} seconds per thousand steps')

    def train(self) -> int:
        raise NotImplementedError()


class StudentTrainingExperiment(TrainingExperiment):

    __slots__ = ('student', )

    student: Annotated[Module, field(
        doc='The student model to train',
    )]

    def _coerce_student(self, spec: Any) -> Module:
        return self.get_module(spec)



default_hyperparam_abbrevs = {
    'num_probes': 'np',
    'num_perturbations': 'P',
    'weight_decay': 'wd',
    'lr_decay': 'lrd',
    'probe_rank': 'pr',
    'probe_scale': 'ps',
    'probe_scale_decay': 'psd',
    'decay_every': 'decay',
}
