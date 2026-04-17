#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from pathlib import Path

from river.ale import ControllableValue
from tensile.common import *

from tensile.experiment import CachedInputExperiment, Experiment, Params, ModelTrainingExperiment
from tensile.optim import Optimizer
from tensile.util.buffer import ArrayBuffer
from tensile.util.metric import Metric

from river.probed import PALEController, ProbeableModule



@provides(Experiment, 'gradient')
class GradientDescentExperiment(ModelTrainingExperiment):

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

        model = self.model

        optimizer_kind = self.optimizer_kind
        if optimizer_kind == 'adam':
            kwargs = dict(weight_decay=self.get_param('weight_decay'))
        elif optimizer_kind == 'sgd':
            kwargs = dict(momentum=self.get_param('momentum'))
        else:
            kwargs = dict()

        optimizer = Optimizer.coerce(model=model, learning_rate=lr, kind=optimizer_kind, **kwargs)

        bucket_desc = self.descriptor

        loss_fn = self.loss_fn
        header = self.header
        log = self.print
        report_every = self.report_every

        def train_fn(batch: tuple[Array, Array]):
            inputs, targets = batch
            outputs = model(inputs)
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

        for inp, targ in self.batch_data(1024):
            out = model(inp)
            loss = loss_fn(out, targ)
            log(f'Student[{self.descriptor}] Final loss: {loss:.6f}') # 0.028599 0.083547
            break

        self.add_metric(Metric.from_buffer('loss', losses))

        return s-1


@provides(Experiment, 'probed')
class ProbedTrainingExperiment(ModelTrainingExperiment):

    __slots__ = ('controller',)

    controller: Annotated[PALEController, field(

    )]

    @staticmethod
    def branch_loss_fn(outputs: Array, targets: Array) -> Array:
        return ten.mean(
            ten.square(outputs - targets),
            axis=(1, 2)
        )

    def _lazy_controller(self) -> PALEController:
        model = self.model

        modules = []
        for path, module in tree.traverse(model, include=tree.value_predicate(predicates.is_instance(ProbeableModule))):
            modules.append(module)

        unflat_params = self.unflat_params

        controller = PALEController(
            model=model,
            modules=modules,
            loss_fn=self.branch_loss_fn,
            params=unflat_params,
            **unflat_params
        )
        return controller

    def get_reporter(self, values: Iterable[ControllableValue] = None):
        log = self.print
        if values is None:
            report_values = tuple(self.controller.controllable_values.values())
        else:
            report_values = tuple(values)

        def report(step: int, loss: Array):
            log(f'Step: {step:5d}  Loss: {loss:.6f} ' + ' '.join([f'{v.name}: {v.get_value():.4f}' for v in report_values]))

        return report

    def train(self) -> int:

        controller = self.controller

        controller.initialize_training()

        header = self.header
        log = self.print
        final_batch_size = 1024

        def report_schedule():
            probes = controller.schedule.probes
            scales = controller.schedule.scales

            header('probes')
            log(ten.sign(scales) * probes)
            header('scales')
            log(scales)

        report_schedule()

        report_trigger = predicates.every_n(self.report_every)

        # losses = ArrayBuffer()
        header(f'Student[{self.descriptor}] Training')

        step_fn = controller.build_step()

        report = self.get_reporter()

        step = 1

        for e in range(self.num_epochs):
            header(f'{self.descriptor} Epoch: {e+1:2d}')
            for inputs, targets in self.batch_data(self.batch_size):

                loss = step_fn(step, inputs, targets)

                if report_trigger(step): report(step, loss)

                step += 1

        for inputs, targets in self.batch_data(final_batch_size):
            inputs = ten.broadcast_to(inputs[None], (controller.num_branches, *inputs.shape))
            outputs = self.model(inputs)
            branch_losses = controller.loss_fn(outputs, targets[None])
            loss = branch_losses[0]
            ten.eval(branch_losses)
            log(f'Final loss: {loss:.6f}') # 0.028599 0.083547
            break

        controller.finalize_training()

        for metric in controller.get_metrics(step):
            self.add_metric(metric)

        return step-1

    def _add_state(self, state: dict[str, Any]):
        super()._add_state(state)
        if controller := self.controller.state_dict():
            state['controller'] = controller

    def fixed_param_defaults(self) -> Params:
        return {
            'probe_manager.rank': 1,
            'probe_manager.kind': 'lora',
            'scheduler.kind': 'random',
            'credit_assigner.kind': 'greedy',
        }

    hyperparam_abbrevs = {
        **Experiment.hyperparam_abbrevs,
        'probe_manager.rank': 'gr',
        'probe_manager.num_probes': 'np',
        'num_branches': 'nb',
        'scheduler.exploration_radius': 'er',
        'credit_assigner.top_k': 'tk',
        'weight_decay': 'wd',
        'update_rule.step_size': 'lr',
        'generate_every': 'ge',
    }


@provides(Experiment, 'tmaze')
class TMazeExperiment(CachedInputExperiment):

    __slots__ = ('delay', 'num_cues')

    delay: Annotated[int, field(
        doc='Delay in steps before credit is rewarded',
        default=10,
    )]
    num_cues: Annotated[int, field(
        doc='The number of cues to use',
        default=2,
    )]

    def _lazy_model_dir(self) -> Path:
        return self.work_dir / f'models/tmaze-{self.arch}-seed-{self.seed}'

    def _lazy_input_dir(self) -> Path:
        if self.num_cues == 2:
            input_dir = f'tmaze-{self.arch}-delay-{self.delay}-seed-{self.seed}'
        else:
            input_dir = f'tmaze-{self.arch}-delay-{self.delay}-cues-{self.num_cues}-seed-{self.seed}'
        return self.work_dir / f'inputs/{input_dir}'

    def generate_input_chunk(self) -> Array:
        chunk = ten.zeros((self.in_chunk_size, self.delay, self.in_dim))
        # chunk[:, 0, 0] = 2. * ten.random.bernoulli(0.5, shape=(self.in_chunk_size,)) - 1.
        num_cues = self.num_cues
        if num_cues == 2:
            chunk[:, 0, 0] = 2. * ten.random.bernoulli(0.5, shape=(self.in_chunk_size,)) - 1.
        else:
            chunk[:, 0, 0] = ten.as_type(ten.random.randint(low=0, high=num_cues, shape=(self.in_chunk_size,)), ten.float32)
        chunk[:, 1:, 1] = ten.random.normal(scale=2., shape=(self.in_chunk_size, self.delay-1,))
        return chunk

    def batch_data(self, b: int) -> Iterable[tuple[Array, Array]]:
        num_cues = self.num_cues
        if num_cues == 2:
            for inputs in self.iter_batch(b):
                targets = ten.sign(inputs[..., 0, 0:1])
                yield inputs, targets
        else:
            for inputs in self.iter_batch(b):
                targets = 2. * (inputs[..., 0, 0:1] % 2.) - 1.
                yield inputs, targets

