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
    unet = FlaxUNet2DConditionModel.from_config(
        config={
            "act_fn": "silu",
            "attention_head_dim": [5, 10, 20, 20],
            "block_out_channels": [320, 640, 1280, 1280],
            "center_input_sample": False,
            "cross_attention_dim": 1536,
            "down_block_types": [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ],
            "downsample_padding": 1,
            "dropout": 0.0,
            "dual_cross_attention": False,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 4,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 1e-05,
            "norm_num_groups": 32,
            "num_class_embeds": None,
            "only_cross_attention": False,
            "out_channels": 4,
            "sample_size": 64,
            "up_block_types": [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ],
            "use_linear_projection": True,
        },
        dtype=weight_dtype,
    )

    return tokenizer, text_encoder, vae, vae_params, unet
