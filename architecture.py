import os

import jax.numpy as jnp

from transformers import FlaxT5ForConditionalGeneration, set_seed

from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel


def setup_model(
    seed,
    load_pretrained,
    output_dir,
    training_from_scratch_rng_params,
):
    set_seed(seed)

    # Load models and create wrapper for stable diffusion

    language_model = FlaxT5ForConditionalGeneration.from_pretrained(
        "/data/byt5-base",
        dtype=jnp.float32,
    )

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        "/data/stable-diffusion-2-1-vae",
        dtype=jnp.float32,
    )

    if load_pretrained:
        if os.path.isdir(output_dir):
            # find latest epoch output
            pretrained_dir = [
                dir
                for dir in os.listdir(output_dir).sort(reverse=True)
                if os.path.isdir(os.path.join(output_dir, dir))
            ][0]
        else:
            pretrained_dir = output_dir

        unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
            pretrained_dir,
            dtype=jnp.float32,
        )

        print("loaded unet from pre-trained...")
    else:
        unet = FlaxUNet2DConditionModel.from_config(
            config={
                "_diffusers_version": "0.16.0",
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
            dtype=jnp.float32,
        )
        unet_params = unet.init_weights(rng=training_from_scratch_rng_params)
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
