#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from tensile.common import *
from tensile.nn import CompiledModule, ModuleArgs
from tensile.nn.attention import AttentionScorer


@provides(AttentionScorer, 'gated-logit')
class GatedLogitAttentionScorer(CompiledModule):

    __slots__ = ('beta', 'use_query_beta', 'gate_dim')

    beta: Annotated[Optional[Array], field(
        parameter=True,
    )]
    use_query_beta: Annotated[bool, field(
        default=True,
    )]
    gate_dim: Annotated[int, field(
        default=16,
    )]

    def init_from_args(self, args: ModuleArgs):
        super().init_from_args(args)

        # self.gate_dim = args.get('gate_dim', default=16)
        use_query_beta = args.get('use_query_beta', default=True)
        if use_query_beta:
            self.beta = None
        else:
            self.beta = ten.array(args.get('beta', default=1.))

        self.use_query_beta = use_query_beta
        self.gate_dim = args.get('gate_dim', default=16)

    def build_call(self, mode: CompiledModule.Mode, **options) -> AttentionScorer:
        if self.use_query_beta:

            gate_dim = self.gate_dim + 1

            # noinspection PyUnusedLocal
            def call(queries: Array, keys_t: Array, qs: Optional[slice], ks: Optional[slice], /, offset: int = 0, **extra) -> Array:

                beta = ten.square(queries[..., :1])

                q_gate = queries[..., 1:gate_dim]
                k_gate_t = keys_t[..., 1:gate_dim, :]

                q_seman = queries[..., gate_dim:]
                k_seman_t = keys_t[..., gate_dim:, :]

                score = ten.matmul(q_seman, k_seman_t)

                gate = ten.matmul(q_gate, k_gate_t)

                return score - beta * ten.softplus(-gate)


        else:
            gate_dim = self.gate_dim

            # noinspection PyUnusedLocal
            def call(queries: Array, keys_t: Array, qs: Optional[slice], ks: Optional[slice], /, offset: int = 0, **extra) -> Array:

                q_gate = queries[..., :gate_dim]
                k_gate_t = keys_t[..., :gate_dim, :]

                q_seman = queries[..., gate_dim:]
                k_seman_t = keys_t[..., gate_dim:, :]

                score = ten.matmul(q_seman, k_seman_t)

                gate = ten.matmul(q_gate, k_gate_t)

                return score - self.beta * ten.softplus(-gate)


        return call

