import jax.numpy as jnp

from transformers import ByT5Tokenizer, FlaxT5EncoderModel, set_seed

from diffusers import (
    FlaxAutoencoderKL,
    FlaxUNet2DConditionModel,
)


def setup_model(
    seed,
    mixed_precision,
    pretrained_text_encoder_model_name_or_path,
    pretrained_diffusion_model_name_or_path,

):

    set_seed(seed)

    if mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16
    else:
        weight_dtype = jnp.float32

    # Load models and create wrapper for stable diffusion
    tokenizer = ByT5Tokenizer()

    text_encoder = FlaxT5EncoderModel.from_pretrained(
        pretrained_text_encoder_model_name_or_path,
        dtype=weight_dtype,
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="vae",
        dtype=weight_dtype,
    )
    unet, _ = FlaxUNet2DConditionModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="unet",
        dtype=weight_dtype,
    )

    return tokenizer, text_encoder, vae, vae_params, unet
