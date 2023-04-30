from __future__ import annotations

import jax
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard

import numpy as np
from PIL import Image

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDPMSolverMultistepScheduler,
    FlaxUNet2DConditionModel,
)
from transformers import ByT5Tokenizer

from architecture import setup_model


# TODO: try half-precision

tokenized_prompt_max_length = 1024


def tokenize_prompts(prompt: list[str]):
    return ByT5Tokenizer()(
        text=prompt,
        max_length=tokenized_prompt_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="jax",
    ).input_ids


def convert_images(images: jnp.ndarray):
    # create PIL image from JAX tensor converted to numpy
    return [Image.fromarray(np.asarray(image), mode="RGB") for image in images]


def get_validation_predictions_lambda(
    text_encoder,
    text_encoder_params,
    vae: FlaxAutoencoderKL,
    vae_params,
    unet: FlaxUNet2DConditionModel,
    tokenized_prompts: jnp.ndarray,
):

    scheduler = FlaxDPMSolverMultistepScheduler.from_config(
        config={
            "_diffusers_version": "0.16.0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
        }
    )
    timesteps = 20

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    image_width = image_height = 256

    encoded_prompts = text_encoder(
        tokenized_prompts,
        params=text_encoder_params,
        train=False,
    )

    # Generating latent shape
    latent_shape = (
        1536,
        unet.in_channels,
        image_width // vae_scale_factor,
        image_height // vae_scale_factor,
    )

    def __predict_images(seed, unet_params):
        def ___timestep(step, step_args):
            latents, scheduler_state = step_args

            t = jnp.asarray(scheduler_state.timesteps, dtype=jnp.int32)[step]

            timestep = jnp.array(jnp.broadcast_to(t, latents.shape[0]), dtype=jnp.int32)

            scaled_latent_input = jnp.array(
                scheduler.scale_model_input(scheduler_state, latents, t)
            )

            # predict the noise residual
            unet_prediction_sample = unet.apply(
                {"params": unet_params},
                scaled_latent_input,
                timestep,
                encoded_prompts,
            ).sample

            # compute the previous noisy sample x_t -> x_t-1
            return scheduler.step(
                scheduler_state, unet_prediction_sample, t, latents
            ).to_tuple()

        # initialize latents
        initial_latents = (
            jax.random.normal(
                jax.random.PRNGKey(seed), shape=latent_shape, dtype=jnp.float32
            )
            * initial_scheduler_state.init_noise_sigma
        )

        # initialize scheduler state
        initial_scheduler_state = scheduler.set_timesteps(
            scheduler.create_state(), num_inference_steps=timesteps, shape=latent_shape
        )

        # get denoises latents
        final_latents, _ = jax.lax.fori_loop(
            0, timesteps, ___timestep, (initial_latents, initial_scheduler_state)
        )

        # get image from latents
        vae_output = vae.apply(
            {"params": vae_params},
            1 / vae.config.scaling_factor * final_latents,
            method=vae.decode,
        ).sample

        # return 8 bit RGB image (width, height, rgb)
        return (
            ((vae_output / 2 + 0.5).transpose(0, 2, 3, 1).clip(0, 1) * 255)
            .round()
            .astype(jnp.uint8)
        )

    return lambda seed, unet_params: __predict_images(seed, unet_params)


if __name__ == "__main__":
    # Pretrained/freezed and training model setup
    text_encoder, text_encoder_params, vae, vae_params, unet, unet_params = setup_model(
        43,  # seed
        None,  # dtype (defaults to float32)
        True,  # load pre-trained
        "character-aware-diffusion/charred",
        None,
    )
    # validation prompts
    validation_prompts = [
        "a white car",
        "une voiture blanche",
        "a running shoe",
        "une chaussure de course",
        "a perfumer and his perfume organ",
        "un parfumeur et son orgue à parfums",
        "two people",
        "deux personnes",
        "a happy cartoon cat",
        "un dessin de chat heureux",
        "a city skyline",
        "un panorama urbain",
        "a Marilyn Monroe portrait",
        "un portrait de Marilyn Monroe",
        "a rainy day in London",
        "Londres sous la pluie",
    ]

    tokenized_prompts = tokenize_prompts(validation_prompts)

    validation_predictions_lambda = get_validation_predictions_lambda(
        text_encoder,
        text_encoder_params,
        vae,
        vae_params,
        unet,
        tokenized_prompts,
    )

    get_validation_predictions = jax.pmap(
        fun=validation_predictions_lambda,
        axis_name=None,
        donate_argnums=(),
    )

    image_predictions = get_validation_predictions(42, unet_params)

    images = convert_images(image_predictions)
