#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import sys
import time
from pathlib import Path
from collections.abc import Container

from tensile import ten, Array
from tensile.experiment import Experiment

# noinspection PyUnusedImports
import river.ale
# noinspection PyUnusedImports
import river.probed
# noinspection PyUnusedImports
import river.experiments


def main():
    seeds = [
        10,
        20,
        30,
        40,
        50,
    ]

    # params = 'fixed'
    # params = 'decay'
    # params = 'fixed-er'
    # params = 'reg-fixed-er'
    # params = 'adaptive-er'
    params = 'adaptive'
    config = Path(f'config/params/{params}.yaml')

    top = Experiment(
        name=f'test-{time.strftime("%Y%m%d-%H%M%S", time.localtime())}-{params}',
        params=config,
    )

    for seed in seeds:
        # add_teacher_regression_experiments(
        #     top,
        #     seed=seed,
        #     seed_range=4,
        #     num_layers=7,
        #     # train='pbdca',
        #     # normalize=True,
        # )
        add_tmaze_experiments(
            top,
            seed=seed,
            # seed_range=2,
            seed_range=4,
            delay=10,
            num_cues=4,
            # memory=1,
            # sever=True,
            # noise=1.0,
            # train='adam,sgd',
            # train='pbdca',
            # batch_size=16,
        )

    top.run()

    return 0



def add_teacher_regression_experiments(
    parent: Experiment,
    seed: int = 10,
    train: Container[str]|str = None,
    seed_range: int = 0,
    in_dim: int = 20,
    hidden_dim: int = 24,
    out_dim: int = 16,
    num_layers: int = 4,
    normalize: bool = False,
    batch_size: int = 8,
    chunk_size: int = 102400,
    chunks_per_epoch: int = 1,
    num_epochs: int = 1,
    sgd_lr: float = 1.5e-3,
    sgd_momentum: float = 0.9,
    adam_lr: float = 1.5e-3,
):
    if isinstance(train, str): train = set(train.split(','))

    train_sgd_student = 'sgd' in train if train else True
    train_adam_student = 'adam' in train if train else True
    train_pbdca_student = 'pbdca' in train if train else True

    arch = f'{in_dim}x' + 'x'.join(str(hidden_dim) for _ in range(num_layers-2)) + f'x{out_dim}'

    model_kind = 'simple-mlp'

    base_name = f'reg-{arch}-seed-{seed}'
    if normalize:
        exp_name = f'{base_name}-norm'
    else:
        exp_name = base_name

    model = dict(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        normalize=normalize,
        kind=model_kind,
    )

    experiment = Experiment.coerce(
        kind='teacher',
        name=exp_name,
        teacher=model,
        in_dim=in_dim,
        out_dim=out_dim,
        in_chunk_size=chunk_size,
        chunks_per_epoch=chunks_per_epoch,
        input_dir=f'inputs/{base_name}',
        model_dir=f'models/{base_name}',
        params={
            'seed': seed,
            'num_layers': num_layers,
            'normalize': normalize,
        },
        parent=parent,
    )

    if train_sgd_student:

        experiment.add_experiment(
            kind='gradient',
            name='sgd',
            params={
                'lr': sgd_lr,
                # 'weight_decay': 0.01,
                'momentum': sgd_momentum,
            },
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size
        )

    if train_adam_student:

        experiment.add_experiment(
            kind='gradient',
            name='adam',
            params={
                'lr': adam_lr,
                # 'weight_decay': 0.01,
                # 'momentum': 0.9,
            },
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size
        )

    if train_pbdca_student:

        def add_pbdca_experiment(probe_seed: int = None):

            hyperparams = {}

            pbdca_model = {
                **model,
                'projection_kind': 'linear.probed',
            }

            sweeps = [
                # {'generate_every': 1000},
                # {'generate_every': 500},
                # {'generate_every': 100},
                # {'generate_every': 50},
                # {'generate_every': 25, 'keep': 4},
                # {'generate_every': 25},
                # {'generate_every': 25, 'schedule_every': 2},
                # {'generate_every': 25, 'probe_manager.num_probes': 12, 'num_branches': 32},
                # {'generate_every': 25, 'probe_manager.num_probes': 12, 'num_branches': 48},
                # {'generate_every': 25, 'probe_manager.num_probes': 16, 'num_branches': 32},
                # {'generate_every': 25, 'probe_manager.num_probes': 16, 'num_branches': 64},

                # {'top_k': 2, 'generate_every': 50},
                # {'top_k': 2, 'generate_every': 25},
                # {'schedule_every': 1},
                # {'schedule_every': 2},
                # {'schedule_every': 3},
                # {'schedule_every': 4},
                # {'top_k': 1, 'probe_manager.num_probes': 12, 'num_branches': 32},
                # {'top_k': 2, 'probe_manager.num_probes': 12, 'num_branches': 32},
                # {'top_k': 3, 'probe_manager.num_probes': 12, 'num_branches': 32},
                # {'top_k': 4, 'probe_manager.num_probes': 12, 'num_branches': 32},
                # {'probe_manager.num_probes': 12, 'num_branches': 48},
                # {'update_rule.step_size': 0.10},
                # {'update_rule.step_size': 0.05},
                # {'update_rule.step_size': 0.02},
                # {'update_rule.step_size': 0.01},
                # {'update_rule.step_size': 0.005},
                # {'update_rule.step_size': 0.001},
            ]

            name = 'pbdca' if probe_seed is None else f'pbdca-seed-{probe_seed}'

            if sweeps:
                experiment.add_experiment(
                    kind='sweep',
                    name=f'{name}-sweep',
                    child=dict(
                        kind='probed',
                        name=name,
                        params=hyperparams,
                        model=pbdca_model,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        seed=probe_seed,
                    ),
                    sweeps=sweeps,
                )
            else:
                experiment.add_experiment(
                    kind='probed',
                    name=name,
                    params=hyperparams,
                    model=pbdca_model,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    seed=probe_seed,
                )

        if seed_range > 0:
            for s in range(seed, seed+seed_range):
                add_pbdca_experiment(s)
        else:
            add_pbdca_experiment(seed)

    parent.experiments.append(experiment)

    return 0


def add_tmaze_experiments(
    parent: Experiment,
    seed: int = 10,
    train: Container[str]|str = None,
    seed_range: int = 0,
    delay: int = 10,
    num_cues: int = 2,
    memory: int = 0,
    sever: bool = False,
    noise: float = 0.0,
    in_dim: int = 2,
    hidden_dim: int = 16,
    out_dim: int = 1,
    batch_size: int = 8,
    chunk_size: int = 102400,
    chunks_per_epoch: int = 2,
    time_per_epoch: float = 0.0,
    num_epochs: int = 1,
    sgd_lr: float = 1.5e-3,
    sgd_momentum: float = 0.9,
    adam_lr: float = 1.5e-3,
):
    if isinstance(train, str): train = set(train.split(','))

    train_sgd_student = 'sgd' in train if train else True
    train_adam_student = 'adam' in train if train else True
    train_pbdca_student = 'pbdca' in train if train else True

    arch = f'tmaze-{in_dim}x{hidden_dim}x{out_dim}'

    model_dir = f'{arch}-seed-{seed}'
    if memory:
        ename = f'{arch}-delay-{delay}-mem-{memory}-seed-{seed}'
        if not sever:
            model_dir = f'{arch}-mem-{memory}-seed-{seed}'
    else:
        ename = f'{arch}-delay-{delay}-seed-{seed}'

    experiment = Experiment.coerce(
        kind='tmaze',
        parent=parent,
        name=ename,
        arch=arch,
        in_dim=in_dim,
        out_dim=out_dim,
        in_chunk_size=chunk_size,
        delay=delay,
        num_cues=num_cues,
        chunks_per_epoch=chunks_per_epoch,
        time_per_epoch=time_per_epoch,
        # input_dir=f'inputs/{ename}',
        model_dir=f'models/{model_dir}',
        params={
            'seed': seed,
            'memory': memory,
            'num_cues': num_cues,
            'sever': sever,
            'noise': noise,
        }
    )

    model_kind = 'simple-rnn'
    model = dict(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        memory=memory,
        sever=sever,
        noise=noise,
        kind=model_kind,
    )

    if train_sgd_student:

        experiment.add_experiment(
            kind='gradient',
            name='sgd',
            params={
                'lr': sgd_lr,
                # 'weight_decay': 0.01,
                'momentum': sgd_momentum,
            },
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size
        )

    if train_adam_student:

        experiment.add_experiment(
            kind='gradient',
            name='adam',
            params={
                'lr': adam_lr,
                # 'weight_decay': 0.01,
                # 'momentum': 0.9,
            },
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size
        )

    if train_pbdca_student:

        def add_pbdca_experiment(probe_seed: int = None):
            hyperparams = {}

            if probe_seed is not None:
                hyperparams['seed'] = probe_seed

            pbdca_model = {
                **model,
                'projection_kind': 'linear.probed',
            }

            sweeps = [
                # Sweep.unpack({'credit_every': range(1, 3)})
                # Sweep.unpack({'update_rule.step_size': [0.15, 0.10]})
            ]

            name = 'probed' if probe_seed is None else f'probed-seed-{probe_seed}'

            if sweeps:
                experiment.add_experiment(
                    kind='sweep',
                    name=f'{name}-sweep',
                    params=hyperparams,
                    child=dict(
                        kind='probed',
                        # name=name,
                        model=pbdca_model,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        seed=probe_seed,
                    ),
                    sweeps=sweeps,
                )
            else:
                experiment.add_experiment(
                    kind='probed',
                    name=name,
                    params=hyperparams,
                    model=pbdca_model,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    seed=probe_seed,
                )

        if seed_range > 0:
            for s in range(seed, seed+seed_range):
                add_pbdca_experiment(s)
        else:
            add_pbdca_experiment(seed)

    parent.experiments.append(experiment)

    return 0

def moving_average(x: Array, window: int, axis: int = -1) -> Array:
    x = ten.swapaxes(x, axis, -1)

    c = ten.cumsum(x, axis=-1)
    c[..., window:] = c[..., window:] - c[..., :-window]
    y = c[..., window - 1:] / window

    return ten.swapaxes(y, -1, axis)

def xmain():
    from tensile.util import plot

    work_dir = Path('./work')

    decay_metrics = ten.load_tensors(work_dir / 'test-20260411-150254/metrics.safetensors')
    adaptive_metrics = ten.load_tensors(work_dir / 'test-20260411-151434/metrics.safetensors')

    series: dict[str, Array] = {}
    for name, metric in decay_metrics.items():
        if name.endswith('loss'):
            series[name + '-decay'] = metric

    for name, metric in adaptive_metrics.items():
        if name.endswith('loss'):
            series[name + '-adapt'] = metric

        # chart.plot_metrics(metrics, out=self.get_path(f"metrics.png", write=True), smoothing=0.9)

    grid = {}
    for name, array in series.items():
        sep = name.rfind(':')
        if sep < 0:
            exp = m = name
        else:
            exp = name[:sep]
            m = name[sep+1:]
        if exp in grid:
            grid[exp][m] = array
        else:
            grid[exp] = {m: array}
    if len(grid) > 1:
        plot.plot_grid(grid, out=work_dir / f"decay-adapt-grid.png", smoothing=0.9, rows=4)


sys.exit(main())
