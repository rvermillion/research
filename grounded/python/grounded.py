#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from tensile.nn.common import *
from tensile.nn.module import ModuleArgs
from tensile.nn.attention.attend import Attend, DefaultAttend
from tensile.nn.attention.score import AttentionScores


default_initial_beta = -20.


@provides(Attend, 'grounded')
class AttendWithGround(DefaultAttend):

    __slots__ = ('betas', 'ground_values')

    betas: Annotated[Array, field(
        doc='The beta value to use for the attention mechanism',
        parameter=True,
    )]
    ground_values: Annotated[Array, field(
        doc='The ground value to use for the attention mechanism',
        parameter=True,
    )]

    def init_from_args(self, args: ModuleArgs):
        super().init_from_args(args)

        n_q_heads = args.get('num_attention_heads')
        n_kv_heads = args.get('num_key_value_heads')
        hidden_size = args.get('hidden_dim')
        initial_beta = args.get('initial_beta', default=default_initial_beta)
        dtype = ten.dtype(args.get('dtype', default=ten.float32))

        if args.get('nilpotent', False):
            ground_values = ten.zeros((1, n_kv_heads, 1, 1, hidden_size // n_q_heads), dtype=dtype)
            betas = ten.full((1, n_kv_heads, 1, 1, 1), initial_beta, dtype=dtype)
        else:
            size = n_kv_heads * (hidden_size//n_q_heads)
            scale = size ** 0.5
            ground_values = ten.as_type(ten.random.normal(scale=scale, shape=(1, n_kv_heads, 1, 1, hidden_size // n_q_heads)), dtype)
            betas = ten.full((1, n_kv_heads, 1, 1, 1), 0., dtype=dtype)

        # ten.eval(ground_values, betas)

        self.betas = betas
        self.ground_values = ground_values

    def scores_factory(self, queries: Array, qs: slice, v_dim: int) -> AttentionScores:
        # We simply initialize the attention scores with the learned ground values and betas.
        scores = AttentionScores(queries, qs, v_dim)
        scores.add_unmasked(self.betas, self.ground_values)
        return scores

    default_weight_aliases = {
        'null_values': 'ground_values',
    }

