#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from tensile import Array, DType, ten
from tensile.infra import meta
from tensile.infra.types import *
from tensile.nn import CompiledModule
from tensile.nn.layers import DecoderLayer, TransformerBlock


def reflect(x: Array, b: Array) -> Array:
    """Computes the reflection of vector x across the plane normal to vector b using Geometric Algebra (-bxb).

    Args:
        x: Input vector to be reflected.
        b: Normal vector to the reflection plane. Must be a unit vector for proper reflection.

    Returns:
        The reflected vector.
    """
    return x - 2.*ten.matmul(ten.expand_dims(x, axis=-2), ten.expand_dims(b, axis=-1))[..., 0] * b


def rotate(x: Array, a: Array, b: Array) -> Array:
    """Computes the rotation using Geometric Algebra via abxba """
    return reflect(reflect(x, b), a)


def ensure_dtype(x: Array, dtype: DType) -> Array:
    return x if x.dtype is dtype else ten.as_type(x, dtype)


def dtype_ensurer(dtype: DType) -> Callable[[Array], Array]:
    def ensure(x: Array) -> Array:
        return x if x.dtype is dtype else ten.as_type(x, dtype)
    return ensure


ensure_f16 = dtype_ensurer(ten.float16)
ensure_f32 = dtype_ensurer(ten.float32)


def rotate_ensure(x: Array, a: Array, b: Array, ensure: Callable[[Array], Array]) -> Array:
    """Computes the rotation using Geometric Algebra via abxba """
    r = reflect(reflect(ensure(x), ensure(b)), ensure(a))
    return ensure_dtype(r, x.dtype)


def rotate_f16(x: Array, a: Array, b: Array) -> Array:
    """Computes the rotation using Geometric Algebra via abxba """
    return rotate_ensure(x, a, b, ensure_f16)


def rotate_f32(x: Array, a: Array, b: Array) -> Array:
    """Computes the rotation using Geometric Algebra via abxba """
    return rotate_ensure(x, a, b, ensure_f32)


precision_rotators = {
    'f16': rotate_f16,
    'f32': rotate_f32,
    'default': rotate_f32,
}


def pct_diff(a: Array, b: Array) -> Array:
    return (b-a)/ten.sqrt(ten.sum(ten.square(a), axis=-1, keepdims=True))


@meta.provides(DecoderLayer, 'transformer.rotational')
@meta.provides(TransformerBlock, 'rotational')
class RotationalTransformerBlock(DecoderLayer):
    """A transformer block that uses geometric rotations instead of additive attention.

    This block modifies the standard transformer by replacing the additive attention
    mechanism with a geometric rotation. Instead of adding the attention output to
    the input, it reflects the input across a plane determined by the attention
    output plus a reference direction (e0 by default, but can be set by reference_dim).

    The rotation is implemented efficiently using vector operations:
    1. Compute attention output b
    2. Add reference direction to b[reference_dim]
    3. Reflect input x across b using h = x - 2(b•x)b
    4. Flip sign of reference dim component h[reference_dim]

    This geometric transformation allows the model to capture relationships that
    would be difficult to represent through pure addition.
    """

    reference_dim: int

    # kind = 'rotational'

    def init_from_args(self, args: DecoderLayer.Args, reference_dim: int = 0, **kwargs):
        super().init_from_args(args)
        self.reference_dim = reference_dim

    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable:
        attention = self.attention
        mlp = self.mlp
        reference_dim = self.reference_dim

        def decode(x: Array) -> Array:
            """Apply rotational transformer block to input.

            Args:
                x: Input tensor of shape [batch, seq_len, hidden_size]

            Returns:
                Output tensor of same shape as input after applying attention,
                rotation, and MLP transformations.
            """

            b = attention(x)

            # Instead of adding the residual, we rotate the input `x` by twice the angle between the reference vector
            # and `b`. We use a trick where we insist on the reference vector being a one hot vector in the
            # `reference_dim` direction, and then we can use a fast reflection instead of a rotate.
            b[..., reference_dim] += 1.

            b_norm = ten.norm(b)

            x_f32 = ensure_f32(x)
            b_f32 = ensure_f32(b_norm)

            h = reflect(x_f32, b_f32)
            h[..., reference_dim] = -h[..., reference_dim]
            x = ensure_dtype(h, x.dtype)

            h = mlp(x)

            return x + h

        return decode


@meta.provides(DecoderLayer, 'transformer.rotational.learned')
@meta.provides(TransformerBlock, 'rotational.learned')
@meta.provides(RotationalTransformerBlock, 'learned')
class LearnedReferenceRotationalTransformerBlock(DecoderLayer):
    """A transformer block that uses geometric rotations instead of additive attention.

    This block modifies the standard transformer by replacing the additive attention
    mechanism with a geometric rotation. Instead of adding the attention output to
    the input, it reflects the input across a plane determined by the attention
    output plus a reference direction (e0 by default, but can be set by reference_dim).

    The rotation is implemented efficiently using vector operations:
    1. Compute attention output b
    2. Add reference direction to b[reference_dim]
    3. Reflect input x across b using h = x - 2(b•x)b
    4. Flip sign of reference dim component h[reference_dim]

    This geometric transformation allows the model to capture relationships that
    would be difficult to represent through pure addition.
    """

    reference_vector: Array
    rotate_dtype: Callable[[Array, Array, Array], Array]

    # kind = 'rotational.learned'

    def init_from_args(self, args: DecoderLayer.Args, rotation_dtype: str = 'default', **kwargs):
        super().init_from_args(args)

        self.reference_vector = ten.random.normal(shape=(self.hidden_dim,))
        self.rotate_dtype = precision_rotators[rotation_dtype]

    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable:
        attention = self.attention
        mlp = self.mlp
        reference_vector = self.reference_vector
        rotate_dtype = self.rotate_dtype

        def decode(x: Array) -> Array:
            """Apply rotational transformer block to input.

            Args:
                x: Input tensor of shape [batch, seq_len, hidden_size]

            Returns:
                Output tensor of same shape as input after applying attention,
                rotation, and MLP transformations.
            """

            ref = ten.norm(reference_vector)

            b_delta = attention(x)

            # Attention gives us back a delta to add to the reference vector (and normalize) to get vector `b`
            b = ten.norm(b_delta + ref)

            # Instead of adding the residual, we rotate the input `x` by twice the angle between `ref` and `b`
            x = rotate_dtype(x, ref, b)

            h = mlp(x)

            return x + h

        return decode
