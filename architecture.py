import os

import jax.numpy as jnp

from transformers import FlaxT5ForConditionalGeneration, set_seed

from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel


def setup_model(
    seed,
    mixed_precision,
    load_pretrained,
    output_dir,
):

    set_seed(seed)

    if mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16
    else:
        weight_dtype = jnp.float32

    # Load models and create wrapper for stable diffusion

    language_model = FlaxT5ForConditionalGeneration.from_pretrained(
        "/data/byt5-base",
        dtype=weight_dtype,
    )

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        "/data/stable-diffusion-2-1-vae",
        dtype=weight_dtype,
    )

    if load_pretrained:

        # find latest epoch output
        pretrained_dir = [
            dir
            for dir in os.listdir(output_dir).sort(reverse=True)
            if os.path.isdir(os.path.join(output_dir, dir))
        ][0]

        unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
            pretrained_dir,
            dtype=weight_dtype,
        )

        print("loaded unet from pre-trained...")
    else:
        unet = FlaxUNet2DConditionModel.from_config(
            config={
                "attention_head_dim": [5, 10, 20, 20],
                "block_out_channels": [320, 640, 1280, 1280],
                "cross_attention_dim": 1536,
                "down_block_types": [
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ],
                "dropout": 0.0,
                "flip_sin_to_cos": True,
                "freq_shift": 0,
                "in_channels": 4,
                "layers_per_block": 2,
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
        unet_params = None
        print("training unet from scratch...")

    return (
        language_model.encode,
        language_model.params,
        vae,
        vae_params,
        unet,
        unet_params,
    )


if __name__ == "__main__":

    language_model = FlaxT5ForConditionalGeneration.from_pretrained(
        "google/byt5-base",
        dtype=jnp.float32,
    )

    language_model.save_pretrained("/data/byt5-base")

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        "flax/stable-diffusion-2-1",
        subfolder="vae",
        dtype=jnp.float32,
    )

    vae.save_pretrained("/data/stable-diffusion-2-1-vae", params=vae_params)
