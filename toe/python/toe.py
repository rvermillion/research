#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
import math

from tensile.nn.common import Callable, Optional, Array, ten
from tensile.graph.common import Base
from tensile.graph.util import show_array


def show(name: str, _a: Array, **options):
    msg = f'{name}: {_a.shape} '
    show_array(_a, prefix=msg, **options)


class ToEBase(Base):

    __slots__ = ()

    def prepare(self, x: Array, **options) -> None:
        pass

    def show(self, name: str, x: Array, **options):
        show(name, x, **options)


class Splitter(ToEBase):

    __slots__ = ('expert_count', )

    expert_count: int

    def __init__(self, expert_count: int):
        self.expert_count = expert_count

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError()


class SplitVectorSplitter(Splitter):

    centers: Array
    split_vectors: Array

    def __init__(self, expert_count: int, input_dim: int):
        super().__init__(expert_count)
        self.centers = ten.random.normal(shape=(1, expert_count, input_dim))
        self.split_vectors = ten.random.normal(shape=(1, expert_count, input_dim))

    def prepare(self, x: Array, svd: bool = True, **options) -> None:
        # x has shape (B, M)
        B, M = x.shape

        # split_vectors has shape (1, P, M)
        split_vectors = self.split_vectors
        P = split_vectors.shape[1]

        centers: list[Optional[Array]] = [None] * P
        sv_list: list[Optional[Array]] = [None] * P

        def recenter(p: int, u: Array, s: Array):
            # u has shape (B, M)
            # s has shape (B, 1)
            i = p-1
            n = ten.sum(s)
            if n.item() < 1e-6:
                return
            center = ten.sum(s * u, axis=0, keepdims=True) / ten.sum(s)
            self.show(f'center {i}', center)
            # center has shape (1, M)
            centers[i] = center

            centered = u - center # (B, M)
            if svd:
                U, S, Vt = ten.linalg.svd(s*centered)
                split_vector = ten.reshape(Vt[0], (1, M))  # First singular vector
                self.show(f'split vector {i}', split_vector)
                sv_list[i] = split_vector
            else:
                split_vector = split_vectors[..., i, :]

            c = p << 1
            if c <= P:
                mag = ten.sum(centered * split_vector, axis=-1, keepdims=True)
                splits = ten.functional.sigmoid(mag)
                # splits has shape (B, 1)

                recenter(c, u, splits * s)
                splits = 1. - splits
                recenter(c+1, u, splits * s)

        recenter(1, x, ten.ones((B, 1), dtype=ten.float32))
        new_centers = ten.stack(centers, axis=1)

        self.centers = new_centers
        if svd:
            self.split_vectors = ten.stack(sv_list, axis=1)

    def __call__(self, x: Array) -> Array:
        # x has shape (B, M)
        w = ten.expand_dims(x, axis=1)

        # w has shape (B, 1, M)
        # centers has shape (1, P, M)
        # centered has shape (B, P, M)
        centered = w - self.centers
        self.show('centered', centered)

        # split_vectors has shape (1, P, M)
        # mag has shape (B, P)
        mag = ten.sum(centered * self.split_vectors, axis=-1)
        self.show('mag', mag)
        # mag = ten.random.normal(scale=5, shape=shape)

        splits = ten.functional.sigmoid(mag)

        return splits


class Experts(ToEBase):

    __slots__ = ('expert_count', )

    expert_count: int

    def __init__(self, expert_count: int):
        self.expert_count = expert_count

    def __call__(self, x: Array, experts: Array = None) -> Array:
        raise NotImplementedError()

    @classmethod
    def create(cls, *, expert_count: int, input_dim: int, output_dim: int, hidden_dim: int = None,
               layer_count: int = 1, activation: Callable[[Array], Array] = None):
        if layer_count == 1:
            return LinearExperts(expert_count, input_dim, output_dim, hidden_dim, activation)
        return MultiLayerExperts(expert_count, input_dim, output_dim, hidden_dim,
                                 inner_layer_count=layer_count-1,
                                 activation=activation)


class LinearExperts(Experts):

    input_dim: int
    output_dim: int
    hidden_dim: int
    activation: Callable[[Array], Array]
    input_weights: Array
    output_weights: Array

    def __init__(self, expert_count: int, input_dim: int, output_dim: int, hidden_dim: int = None, activation: Callable[[Array], Array] = None):
        super().__init__(expert_count)
        hidden_dim = hidden_dim or max(input_dim, output_dim)
        std = math.sqrt(1.0 / input_dim)  # Xavier initialization

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = activation or ten.functional.relu
        self.input_weights = ten.random.normal(shape=(expert_count, hidden_dim, input_dim)) * std
        self.output_weights = ten.random.normal(shape=(expert_count, output_dim, hidden_dim)) * std

    def __call__(self, x: Array, experts: Array = None) -> Array:
        # experts has shape (B, K)

        # x_expanded has shape (B, 1, M, 1)
        x_expanded = ten.expand_dims(x, axis=(1, -1))
        # self.show('expanded', x_expanded)

        if experts is None:
            input_weights = self.input_weights
            output_weights = self.output_weights
        else:
            input_weights = self.input_weights[experts]
            output_weights = self.output_weights[experts]

        # o has shape (B, K, H, 1)
        o = input_weights @ x_expanded
        self.show('o0', ten.squeeze(o, axis=-1))

        # a has shape (B, K, H, 1)
        a = self.activation(o)
        self.show('a0', ten.squeeze(a, axis=-1))

        # y has shape (B, K, V, 1)
        y = output_weights @ a
        y = ten.squeeze(y, axis=-1)
        self.show('y0', y)

        # return has shape (B, K, V)
        return y


class MultiLayerExperts(LinearExperts):

    inner_weights: tuple[Array, ...]

    def __init__(
        self, expert_count: int, input_dim: int, output_dim: int, hidden_dim: int = None,
        inner_layer_count: int = 2, activation: Callable[[Array], Array] = None):
        super().__init__(expert_count, input_dim, output_dim, hidden_dim, activation)
        H = self.hidden_dim
        std = math.sqrt(1.0 / input_dim)  # Xavier initialization

        self.inner_weights = tuple(std * ten.random.normal(shape=(expert_count, H, H)) for _ in range(inner_layer_count))

    def __call__(self, x: Array, experts: Array = None) -> Array:
        # x_expanded has shape (B, 1, M, 1)
        x_expanded = ten.expand_dims(x, axis=(1, -1))
        # self.show('expanded', x_expanded)

        if experts is None:
            input_weights = self.input_weights
            output_weights = self.output_weights
        else:
            input_weights = self.input_weights[experts]
            output_weights = self.output_weights[experts]

        i = 0

        # o has shape (B, K, H, 1)
        o = input_weights @ x_expanded
        self.show(f'o{i}', ten.squeeze(o, axis=-1))

        # a has shape (B, K, H, 1)
        a = self.activation(o)
        self.show(f'a{i}', ten.squeeze(a, axis=-1))

        for inner_weights in self.inner_weights:
            i += 1
            if experts is not None:
                inner_weights = inner_weights[experts]
            o = inner_weights @ a
            self.show(f'o{i}', ten.squeeze(o, axis=-1))

            a = self.activation(o)
        self.show(f'a{i}', ten.squeeze(a, axis=-1))

        # y has shape (B, K, V, 1)
        y = output_weights @ a
        y = ten.squeeze(y, axis=-1)
        self.show('y', y)

        # return has shape (B, K, V)
        return y


class ExpertSelector(ToEBase):

    __slots__ = ()

    def __call__(self, log_weights: Array) -> tuple[Array, Optional[Array]]:
        raise NotImplementedError()


class AllSelector(ExpertSelector):

    __slots__ = ()

    def __call__(self, log_weights: Array) -> tuple[Array, Optional[Array]]:
        return log_weights, None


class TopKSelector(ExpertSelector):

    __slots__ = ('top_k',)

    top_k: int

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def __call__(self, log_weights: Array) -> tuple[Array, Optional[Array]]:
        K = self.top_k
        B, N = log_weights.shape

        batch_ind = ten.arange(B).reshape(B, 1)
        sorted_ind = ten.argsort(log_weights, axis=-1)
        self.show('sorted indices', sorted_ind+1, threshold=200)
        experts = sorted_ind[..., -K:]
        self.show('experts', experts+1, threshold=200)
        self.show('leafs selected', ten.sum(experts >= ((N+1)>>1)-1, axis=-1), threshold=200)

        logw = log_weights[batch_ind, experts]
        self.show('top log weights', logw, threshold=200)

        # total_weight = ten.sum(log_weights, axis=-1, keepdims=True)
        # self.show('total weight', total_weight)
        #
        # log_weights = log_weights / total_weight
        # self.show('normalized weights', log_weights)

        logw = ten.expand_dims(logw, axis=-1)

        return logw, experts


class TreeBeamSelector(ExpertSelector):
    """
    Beam search over a full binary tree router, closure-preserving by construction.

    Inputs:
      log_weights: (B, N)  where left[b, p] = P(go_left | parent node p)
    Outputs:
      weights: (B, Ksel, 1)
      experts: (B, Ksel)  node ids in [0..N-1]
    """

    __slots__ = ('top_k', 'depth', 'beam_width', 'eps', 'include_internal')

    top_k: int
    depth: int
    beam_width: int
    eps: float
    include_internal: bool

    def __init__(
        self,
        top_k: int,
        depth: int,
        beam_width: int = None,
        eps: float = 1e-9,
        include_internal: bool = True,
    ):
        super().__init__()
        self.top_k = top_k
        self.depth = depth
        # Default: choose a width so that collecting internal beams yields ~top_k nodes.
        # Collected count ≈ 1 + (depth-1)*beam_width if include_internal.
        if beam_width is None:
            bw = max(1, top_k // max(1, (depth - 1)))
        else:
            bw = int(beam_width)
        self.beam_width = bw
        self.eps = eps
        self.include_internal = include_internal

    def __call__(self, log_weights: Array) -> tuple[Array, Optional[Array]]:
        # log_weights: (B, N)
        B, N = log_weights.shape
        D = self.depth
        K = self.beam_width
        eps = self.eps

        # Root node id is 0. Start beam with the root.
        beam_nodes = ten.zeros((B, 1), dtype=ten.int32)         # (B,1)
        beam_logw  = ten.zeros((B, 1), dtype=ten.float32)       # (B,1)

        batch = ten.arange(B).reshape(B, 1)

        # Optionally collect internal nodes at each depth (closure set).
        if self.include_internal:
            collected_nodes = [beam_nodes]
            collected_logw  = [beam_logw]
        else:
            collected_nodes = []
            collected_logw  = []

        # Expand for D-1 steps to reach depth D-1 (leaves live at that depth in a full tree).
        for _ in range(D - 1):
            # Beam nodes here are always internal until the last expansion.
            # Gather split probs for these parent nodes: left[b, parent_id]
            # logp_left = log_weights[batch, beam_nodes]                    # (B,Kcur)
            # p_left = ten.clip(p_left, eps, 1.0 - eps)
            self.debug(f'beam nodes {_}', beam_nodes)

            # Children ids (full binary tree indexing):
            left_child  = beam_nodes * 2 + 1                    # (B,Kcur)
            right_child = left_child + 1                        # (B,Kcur)

            # Candidate log weights
            logw_left = log_weights[batch, left_child]
            logw_right = log_weights[batch, right_child]

            # Stack candidates (2*Kcur)
            cand_nodes = ten.concatenate([left_child, right_child], axis=1)  # (B,2Kcur)
            cand_logw  = ten.concatenate([logw_left, logw_right], axis=1)    # (B,2Kcur)

            # Keep top beam_width among candidates
            # argsort is only over 2Kcur, which stays small.
            Kcur2 = cand_logw.shape[1]
            if Kcur2 > K:
                order = ten.argsort(cand_logw, axis=-1)[:, -K:]              # (B,K)
                beam_nodes = cand_nodes[batch, order]                        # (B,K)
                beam_logw  = cand_logw[batch, order]                         # (B,K)
            else:
                beam_nodes = cand_nodes
                beam_logw  = cand_logw

            if self.include_internal:
                collected_nodes.append(beam_nodes)
                collected_logw.append(beam_logw)

        # Build the candidate set to return.
        if self.include_internal:
            experts = ten.concatenate(collected_nodes, axis=1)   # (B, 1 + (D-1)*K)
            logw    = ten.concatenate(collected_logw, axis=1)    # same shape
        else:
            experts = beam_nodes
            logw    = beam_logw

        # If we collected more than top_k, trim by top_k logw.
        # This *preserves closure* because along any path, parent logw >= child logw.
        Ksel = self.top_k
        if experts.shape[1] > Ksel:
            self.show('experts', experts+1, threshold=200)
            order = ten.argsort(logw, axis=-1)[:, -Ksel:]         # (B,Ksel)
            experts = experts[batch, order]
            logw    = logw[batch, order]

        self.show('experts', experts+1, threshold=200)
        self.show('leafs selected', ten.sum(experts >= ((N+1)>>1)-1, axis=-1), threshold=200)

        # Softmax over selected log weights (stable)
        # m = ten.max(logw, axis=-1, keepdims=True)
        # w = ten.exp(logw - m)
        # w = w / ten.sum(w, axis=-1, keepdims=True)
        # weights = ten.expand_dims(w, axis=-1)                     # (B,Ksel,1)

        logw = ten.expand_dims(logw, axis=-1)

        return logw, experts

class TopKLeafSelector(TopKSelector):

    __slots__ = ('leaf_count',)

    leaf_count: int

    def __init__(self, top_k: int, leaf_count: int):
        super().__init__(top_k)
        self.leaf_count = leaf_count

    def __call__(self, weights: Array) -> tuple[Array, Optional[Array]]:
        K = self.top_k
        B, N = weights.shape
        P = N - self.leaf_count

        batch_ind = ten.arange(B).reshape(B, 1)
        sorted_ind = ten.argsort(weights[..., P:], axis=-1) + P
        self.show('sorted indices', sorted_ind, threshold=200)
        experts = sorted_ind[..., -K:]
        self.show('experts', experts, threshold=200)

        weights = weights[batch_ind, experts]
        self.show('top weights', weights, threshold=200)

        total_weight = ten.sum(weights, axis=-1, keepdims=True)
        self.show('total weight', total_weight)

        weights = weights / total_weight
        self.show('normalized weights', weights)

        weights = ten.expand_dims(weights, axis=-1)

        return weights, experts


class TreeOfExperts(ToEBase):

    __slots__ = ('depth', 'node_count', 'top_k', 'evaluate', 'splitter', 'selector', 'experts')

    depth: int
    node_count: int
    top_k: int
    parent_count: int
    paths: Array
    evaluate: Callable[[Array], Array]
    experts: Experts
    splitter: Splitter
    selector: ExpertSelector

    def __init__(self, depth: int, input_dim: int, output_dim: int, hidden_dim: int = None, top_k: int = 0,
                 splitter: Splitter = None, selector: ExpertSelector = None, experts: Experts = None,
                 expert_layers: int = 1):
        super().__init__()
        size = 1 << depth
        M = input_dim
        H = hidden_dim or max(input_dim, output_dim)
        D = depth
        N = size - 1
        K = top_k
        leaf_count = size >> 1
        P = N - leaf_count
        V = output_dim

        if splitter is None:
            splitter = SplitVectorSplitter(P, M)

        if experts is None:
            experts = Experts.create(expert_count=N, input_dim=M, output_dim=V, hidden_dim=H,
                                     layer_count=expert_layers)
        elif experts.expert_count != N:
            raise ValueError(f'Expert count {experts.expert_count} does not match tree depth {depth}.')

        if selector is None:
            if K > 0:
                # selector = TopKSelector(K)
                selector = TreeBeamSelector(K, D)
            else:
                selector = AllSelector()

        self.debug('depth:', D, 'size:', size, 'node count:', N,
                   'leaf count:', leaf_count, 'parent count:', P,
                   'top k:', K)

        # idx = ten.arange(size, dtype=ten.int32)
        # nodes = idx[shift:]
        nodes = ten.arange(N, dtype=ten.int32)
        self.show('nodes', nodes)

        parent_nodes = nodes[:P]
        self.show('parent nodes', parent_nodes)
        leaf_nodes = nodes[-leaf_count:]
        self.show('leaf nodes', leaf_nodes)

        node_ids = nodes + 1
        # paths has shape (N, D)
        paths = (node_ids.reshape(-1, 1) >> ten.arange(D-1, -1, -1)) - 1
        self.show('paths', paths+1, threshold=200)

        # parents has shape (N, D-1)
        parents = paths[..., :-1]
        self.show('parents', parents)

        if ten.any(parents >= P).item():
            raise ValueError(
                f'Parent indices {parents} are out of bounds for a tree of depth {D} and size {size}.'
            )

        # children has shape (N, D-1)
        children = paths[:, 1:]
        self.show('children', children)

        # left_children has shape (N, D-1)
        left_children = children % 2 == 1
        self.show('left children', left_children)

        # missing_parents has shape (N, D-1)
        missing_parents = parents < 0
        self.show('missing parents', missing_parents)

        safe_parents = ten.maximum(parents, 0)    # (N, D-1)
        eps = 1e-9

        def calculate_log_weights(left: Array) -> Array:
            self.show('left', left)

            left_splits = left[..., safe_parents]
            left_splits = ten.clip(left_splits, eps, 1.0 - eps)
            right_splits = 1. - left_splits

            # splits has shape (B, N, D-1)
            splits = ten.where(missing_parents, 1., ten.where(left_children, left_splits, right_splits))
            self.show('splits', splits)

            log_splits = ten.log(splits)

            # log_weights has shape (B, N)
            log_weights = ten.sum(log_splits, axis=-1)
            self.show('log weights', log_weights)

            return log_weights

        def evaluate(x: Array) -> Array:
            # x has shape (B, M)
            B = x.shape[0]

            left_probs = splitter(x)

            # log_weights has shape (B, N)
            log_weights = calculate_log_weights(left_probs)

            # log_weights has shape (B, K, 1)
            # selected_experts has shape (B, K)
            log_weights, selected_experts = selector(log_weights)

            max_weights = ten.max(log_weights, axis=-1, keepdims=True)
            weights = ten.exp(log_weights - max_weights)
            weights = weights / ten.sum(weights, axis=-1, keepdims=True)

            if selected_experts is None:
                # node_values has shape (B, K, V)
                node_values = experts(x, experts=selected_experts)
                self.show(f'node values', node_values)

                # weighted_values has shape (B, K, V)
                weighted_values = node_values * weights
                self.show('weighted values', weighted_values)

                # values has shape (B, V)
                values = ten.sum(weighted_values, axis=-2)
                self.show('values', values)
            else:
                values = ten.zeros((B, V), dtype=ten.float32)

                kb = 4  # or 8, tune for memory/speed
                for j in range(0, K, kb):
                    w_blk = weights[:, j:j+kb, :]                 # (B,kb,1)
                    e_blk = selected_experts[:, j:j+kb]           # (B,kb)
                    y_blk = experts(x, experts=e_blk)             # (B,kb,V)
                    values = values + ten.sum(y_blk * w_blk, axis=-2)   # (B,V)

            return values

        # else:
        #     def evaluate(x: Array) -> Array:
        #         # x has shape (B, M)
        #
        #         weights = calculate_weights(x)
        #         # weights has shape (B, N)
        #
        #         total_weight = ten.sum(weights, axis=-1, keepdims=True)
        #         self.debug('total weight', total_weight)
        #         weights = weights / total_weight
        #         self.debug('normalized weights', weights)
        #
        #         weights = ten.expand_dims(weights, axis=-1)
        #         # weights has shape (B, N, 1)
        #
        #         # node_values has shape (B, N, V)
        #         # node_values = ten.random.normal(shape=(B, N, V))
        #         node_values = experts(x)
        #         self.debug('node values', node_values)
        #
        #         # weighted_values has shape (B, N, V)
        #         weighted_values = node_values * weights
        #         self.debug('weighted values', weighted_values)
        #
        #         # values has shape (B, V)
        #         values = ten.sum(weighted_values, axis=-2)
        #         self.debug('values', values)
        #
        #         return values

        self.depth = D
        self.node_count = N
        self.top_k = K
        self.evaluate = evaluate
        self.experts = experts
        self.splitter = splitter
        self.selector = selector

    def prepare(self, x: Array, **options) -> None:
        self.debug('preparing tree of experts...')
        self.splitter.prepare(x, **options)
        self.selector.prepare(x, **options)
        self.experts.prepare(x, **options)

    def __call__(self, x: Array, **kwargs) -> Array:
        return self.evaluate(x)


def test_toe():

    B = 2
    M = 4
    H = 5
    V = 3
    D = 6
    top_k = (2 * (D-1)) + 1  #1 << (D-1)

    tree = TreeOfExperts(depth=D, input_dim=M, output_dim=V, hidden_dim=H, top_k=top_k, expert_layers=1)

    scale = ten.random.uniform(high=2, shape=(M,))
    prep = ten.random.normal(shape=(1000, M)) * scale
    show('prep data', prep)

    # tree.prepare(prep)

    x = ten.random.normal(shape=(B, M))
    show('x', x)


    y = tree(x)

    show('y', y)

    exit(0)

    #                       [ 1  2  4  8]
    #                       [ 1  2  4  9]
    #                       [ 1  2  5 10]
    #                       [ 1  2  5 11]
    #                       [ 1  3  6 12]
    #                       [ 1  3  6 13]
    #                       [ 1  3  7 14]
    #                       [ 1  3  7 15]