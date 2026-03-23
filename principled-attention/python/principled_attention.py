#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from tensile.common import *
from tensile.nn.module import CompiledModule, ModuleArgs
from tensile.nn.attention.types import AttentionScorer
from tensile.nn.attention.score import AttentionScores


class PrincipledAttentionScores(AttentionScores):

    __slots__ = ('gamma', 'groundexp', 'ground_value')

    gamma: Array
    groundexp: Array     # (B, *H, Q, 1)
    ground_value: Array     # (B, *H, Q, D_v)

    def __init__(self, queries: Array, qs: slice, v_dim: int, gamma: Array, ground_value: Array, dtype: DType = None):
        super().__init__(queries, qs, v_dim, initial_max=gamma, dtype=dtype)
        self.gamma = gamma
        self.groundexp = ten.array(0., dtype=dtype)
        self.ground_value = ground_value

    def add_masked(self, logits: Array, values: Array) -> None:
        """
        Add new logits and values to the score accumulator.

        :param logits: Logits tensor for the current tile (B, *H, Q, K).
        :param values: Values tensor for the current tile (B, *H, K, D_v).
        :return: None
        """

        key_valid = ten.isfinite(logits)

        # ten.eval(logits, values)
        current_max = ten.max(logits, axis=-1, keepdims=True)

        # Since we initialize self.max with gamma, this is guaranteed to not be -inf
        new_max = ten.maximum(current_max, self.max)

        old_exp = ten.exp(self.max - new_max)

        new_exp = ten.where(key_valid, ten.exp(logits - new_max), 0.0)

        new_groundexp = ten.where(key_valid, ten.exp(ten.maximum(self.gamma, logits) - new_max), 0.0)

        self.sumexp = ten.sum(new_exp, axis=-1, keepdims=True) + self.sumexp * old_exp
        self.groundexp = ten.sum(new_groundexp, axis=-1, keepdims=True) + self.groundexp * old_exp
        self.values = ten.matmul(new_exp, values) + self.values * old_exp
        self.max = new_max
        assert ten.all(self.groundexp + 1e-6 >= self.sumexp), "The groundexp should be >= to the sumexp"

    def add_unmasked(self, logits: Array, values: Array) -> None:
        """
        Add new logits and values to the score accumulator.

        :param logits: Logits tensor for the current tile (B, *H, Q, K).
        :param values: Values tensor for the current tile (B, *H, K, D_v).
        :return: None
        """
        current_max = ten.max(logits, axis=-1, keepdims=True)
        new_max = ten.maximum(current_max, self.max)

        old_exp = ten.exp(self.max - new_max)

        new_exp = ten.exp(logits - new_max)

        groundexp = ten.exp(ten.maximum(self.gamma, logits) - new_max)

        new_sumexp = self.sumexp * old_exp + ten.sum(new_exp, axis=-1, keepdims=True)
        new_groundexp = self.groundexp * old_exp + ten.sum(groundexp, axis=-1, keepdims=True)
        new_values = self.values * old_exp + ten.matmul(new_exp, values)

        self.sumexp = new_sumexp
        self.groundexp = new_groundexp
        self.values = new_values
        self.max = new_max
        assert ten.all(self.groundexp + 1e-6 >= self.sumexp), "The groundexp should be >= to the sumexp"

    def out(self):
        ground_weight = (self.groundexp - self.sumexp) / self.groundexp
        return self.values / self.groundexp + ground_weight * self.ground_value



@provides(AttentionScorer, 'principled')
class PrincipledAttentionScorer(CompiledModule):

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
        gate_end = self.gate_dim + 3

        # noinspection PyUnusedLocal
        def call(queries: Array, keys_t: Array, qs: Optional[slice], ks: Optional[slice], /, offset: int = 0, length: Array = None, **extra) -> Array:

            if length is None:
                raise ValueError("Length must be provided for principled attention scoring")

            # length is (B, *H, Q, 1) how many 

            alpha = queries[..., 0:1]   # (B, *H, Q, 1)
            beta = queries[..., 1:2]    # (B, *H, Q, 1)
            gamma = queries[..., 2:3]   # (B, *H, Q, 1)

            q_gate = queries[..., 3:gate_end]          # (B, *H, Q, G)
            k_gate_t = keys_t[..., 3:gate_end, :]      # (B, *H, G, K)

            q_semantic = queries[..., gate_end:]       # (B, *H, Q, D)
            k_semantic_t = keys_t[..., gate_end:, :]   # (B, *H, D, K)

            semantic_score = ten.matmul(q_semantic, k_semantic_t)   # (B, *H, Q, K)

            gate_score = ten.matmul(q_gate, k_gate_t)               # (B, *H, Q, K)

            margin = (1 + ten.softplus(alpha)*ten.log(length))*(gamma - semantic_score)

            return gamma + margin - ten.softplus(beta) * ten.softplus(-gate_score)

        return call

