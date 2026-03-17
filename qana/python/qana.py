#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from tensile.nn.common import *
from tensile.nn.module import CompiledModule, Module, ModuleArgs
from tensile.nn.attention import AttentionArgs
from tensile.nn.attention.mask import AttentionMasker, make_additive_masker
from tensile.nn.attention.attend import Attend, AttentionScorer
from tensile.nn.attention.util import sdpa_attention_scorer


@provides(Attend, 'qana')
class QANAScorer(CompiledModule):

    __slots__ = ('q_head_dim', 'k_head_dim', 'v_head_dim', 'h_dim', 'u_dim', 'v_dim', 'b_dim', 'qana_proj', 'gamma',
                 'activation')

    q_head_dim: int
    k_head_dim: int
    v_head_dim: int

    h_dim: int
    u_dim: int
    v_dim: int
    b_dim: int

    activation: Activation

    qana_proj: Module
    gamma: Array

    def init_from_args(self, args: AttentionArgs):
        super().init_from_args(args)

        head_dim = args.head_dim

        self.q_head_dim = args.q_head_dim or head_dim
        self.k_head_dim = args.k_head_dim or head_dim
        self.v_head_dim = args.v_head_dim or head_dim

        hidden_dim = 4
        input_dim = self.q_head_dim - self.k_head_dim
        u_dim = hidden_dim * self.k_head_dim
        v_dim = hidden_dim
        b_dim = self.k_head_dim
        output_dim = u_dim + v_dim + b_dim

        self.h_dim = hidden_dim
        self.u_dim = u_dim
        self.v_dim = v_dim
        self.b_dim = b_dim
        self.qana_proj = self.build_qana_proj(input_dim, output_dim)
        self.gamma = ten.array(1e-3)
        self.activation = self.build_activation()

    default_activation_spec: ClassVar[str] = 'relu'

    def build_activation(self) -> Activation:
        return coerce(Activation, kind=self.args.get('activation', default=self.default_activation_spec))

    def build_qana_proj(self, in_size: int, out_size: int) -> Module:
        return self.build_proj(in_size, out_size)

    # noinspection PyPep8Naming
    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable:
        D = self.k_head_dim
        dtype = ten.float32
        network_dim = self.q_head_dim - D
        b_dim = self.b_dim
        u_dim = self.u_dim
        D_v = self.v_head_dim
        v_end = self.v_dim + u_dim
        h_dim = self.h_dim
        activation = self.activation

        q_tile_size = 256
        k_tile_size = 256

        # noinspection PyPep8Naming
        def call(queries: Array, keys: Array, values: Array, *, mask: Array = None) -> Array:

            s = D ** -0.5

            # ten.debug_eval(queries, keys)

            B = queries.shape[0]
            H = queries.shape[1:-2]
            Q = queries.shape[-2]
            K = keys.shape[-2]

            queries = ten.as_type(queries, dtype)
            keys = ten.as_type(keys, dtype)
            values = ten.as_type(values, dtype)

            ten.debug_eval(queries, keys, values)

            last_qs = None
            U_i = None
            V_i = None
            b_i = None

            def scorer(queries: Array, keys: Array, qs: slice, ks: slice) -> Array:
                nonlocal last_qs, U_i, V_i, b_i

                Q_i = queries.shape[-2]
                q_i = queries[..., :D]

                if qs != last_qs:
                    last_qs = qs
                    q_i_net = queries[..., D:]
                    net_i = self.qana_proj(q_i_net)

                    U_i = net_i[..., :u_dim].reshape(B, *H, Q_i, 1, h_dim, self.k_head_dim)
                    V_i = net_i[..., u_dim:v_end].reshape(B, *H, Q_i, 1, 1, h_dim)
                    b_i = net_i[..., v_end:].reshape(B, *H, Q_i, 1, b_dim, 1)

                k_jt = ten.swapaxes(keys, -1, -2)

                logits_ij = ten.matmul(q_i, k_jt)

                bk_ij = keys[..., None, :, :, None] + b_i

                h_ij = ten.matmul(U_i, bk_ij)

                h_ij = activation(h_ij)

                ten.debug_eval(h_ij)

                y_ij = ten.matmul(V_i, h_ij)

                logits_ij += self.gamma * ten.squeeze(y_ij, axis=(-1, -2))

                return logits_ij

            masker = make_additive_masker(mask)

            return tile_attention(queries, keys, values,
                                  masker=masker,
                                  dtype=dtype,
                                  tile_size=(q_tile_size, k_tile_size),
                                  scorer=scorer)

            # q_tiles = []
            # for i in range(0, Q, q_tile_size):
            #     ie = min(i + q_tile_size, Q)
            #     Q_i = ie - i
            #
            #     q_i = s * queries[..., i:ie, :D]
            #
            #     q_i_net = queries[..., i:ie, D:]
            #     net_i = self.qana_proj(q_i_net)
            #
            #     U_i = net_i[..., :u_dim].reshape(B, *H, Q_i, 1, h_dim, self.k_head_dim)
            #     V_i = net_i[..., u_dim:v_end].reshape(B, *H, Q_i, 1, 1, h_dim)
            #     b_i = net_i[..., v_end:].reshape(B, *H, Q_i, 1, b_dim, 1)
            #
            #     max_i = ten.full((B, *H, Q_i, 1), -ten.inf, dtype=dtype)
            #     s_i = ten.zeros((B, *H, Q_i, 1), dtype=dtype)
            #     o_i = ten.zeros((B, *H, Q_i, D_v), dtype=dtype)
            #
            #     for j in range(0, K, k_tile_size):
            #         je = min(j + k_tile_size, K)
            #
            #         keys = keys[..., j:je, :]
            #         v_j = values[..., j:je, :]
            #
            #         k_jt = ten.swapaxes(keys, -1, -2)
            #
            #         logits_ij = ten.matmul(q_i, k_jt)
            #
            #         bk_ij = keys[..., None, :, :, None] + b_i
            #
            #         h_ij = ten.matmul(U_i, bk_ij)
            #
            #         h_ij = activation(h_ij)
            #
            #         ten.debug_eval(h_ij)
            #
            #         y_ij = ten.matmul(V_i, h_ij)
            #
            #         logits_ij += self.gamma * ten.squeeze(y_ij, axis=(-1, -2))
            #
            #         if mask is not None:
            #             logits_ij = apply_additive_mask(logits_ij, mask[..., i:ie, j:je])
            #
            #         max_ij = ten.max(logits_ij, axis=-1, keepdims=True)
            #         new_max_i = ten.maximum(max_ij, max_i)
            #         exp_old = ten.exp(max_i - new_max_i)
            #
            #         prob_ij = ten.exp(logits_ij - new_max_i)
            #
            #         s_i = s_i * exp_old + ten.sum(prob_ij, axis=-1, keepdims=True)
            #         o_i = o_i * exp_old + prob_ij @ v_j
            #
            #         max_i = new_max_i
            #
            #     q_tiles.append(o_i / s_i)
            #
            # out = ten.concatenate(q_tiles, axis=-2)
            #
            # ten.debug_eval(out)
            #
            # return out

        return call
