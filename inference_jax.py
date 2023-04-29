import jax
import jax
import jax.numpy as jnp
from monitoring import wandb_close, wandb_inference_init, wandb_inference_log

import numpy as np
from PIL import Image

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDPMSolverMultistepScheduler,
    FlaxUNet2DConditionModel,
)
from transformers import ByT5Tokenizer, FlaxT5ForConditionalGeneration


def get_inference_lambda(seed):

    tokenizer = ByT5Tokenizer()

    language_model = FlaxT5ForConditionalGeneration.from_pretrained(
        "google/byt5-base",
        dtype=jnp.float32,
    )
    text_encoder = language_model.encode
    text_encoder_params = language_model.params
    max_length = 1024
    tokenized_negative_prompt = tokenizer(
        "", padding="max_length", max_length=max_length, return_tensors="np"
    ).input_ids
    negative_prompt_text_encoder_hidden_states = text_encoder(
        tokenized_negative_prompt,
        params=text_encoder_params,
        train=False,
    )[0]

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
    guidance_scale = jnp.array([7.5], dtype=jnp.float32)

    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        "character-aware-diffusion/charred",
        dtype=jnp.float32,
    )

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        "flax/stable-diffusion-2-1",
        subfolder="vae",
        dtype=jnp.float32,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    image_width = image_height = 256

    # Generating latent shape
    latent_shape = (
        negative_prompt_text_encoder_hidden_states.shape[0],  # is th
        unet.in_channels,
        image_width // vae_scale_factor,
        image_height // vae_scale_factor,
    )

    def __tokenize_prompt(prompt: str):

        return tokenizer(
            text=prompt,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="jax",
        ).input_ids

    def __convert_image(image):
        # create PIL image from JAX tensor converted to numpy
        return Image.fromarray(np.asarray(image), mode="RGB")

    def __predict_image(tokenized_prompt: jnp.array):

        # Get the text embedding
        text_encoder_hidden_states = text_encoder(
            tokenized_prompt,
            params=text_encoder_params,
            train=False,
        )[0]
        context = jnp.concatenate(
            [negative_prompt_text_encoder_hidden_states, text_encoder_hidden_states]
        )

        def ___timestep(step, step_args):

            latents, scheduler_state = step_args

            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]

            # For classifier-free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            latent_input = jnp.concatenate([latents] * 2)

            timestep = jnp.broadcast_to(t, latent_input.shape[0])

            scaled_latent_input = scheduler.scale_model_input(
                scheduler_state, latent_input, t
            )

            # predict the noise residual
            unet_prediction_sample = unet.apply(
                {"params": unet_params},
                jnp.array(scaled_latent_input),
                jnp.array(timestep, dtype=jnp.int32),
                context,
            ).sample

            # perform guidance
            unet_prediction_sample_uncond, unet_prediction_text = jnp.split(
                unet_prediction_sample, 2, axis=0
            )
            guided_unet_prediction_sample = (
                unet_prediction_sample_uncond
                + guidance_scale
                * (unet_prediction_text - unet_prediction_sample_uncond)
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents, scheduler_state = scheduler.step(
                scheduler_state, guided_unet_prediction_sample, t, latents
            ).to_tuple()

            return latents, scheduler_state

        # initialize scheduler state
        initial_scheduler_state = scheduler.set_timesteps(
            scheduler.create_state(), num_inference_steps=timesteps, shape=latent_shape
        )

        # initialize latents
        initial_latents = (
            jax.random.normal(
                jax.random.PRNGKey(seed), shape=latent_shape, dtype=jnp.float32
            )
            * initial_scheduler_state.init_noise_sigma
        )

        final_latents, _ = jax.lax.fori_loop(
            0, timesteps, ___timestep, (initial_latents, initial_scheduler_state)
        )

        vae_output = vae.apply(
            {"params": vae_params},
            1 / vae.config.scaling_factor * final_latents,
            method=vae.decode,
        ).sample

        # return 8 bit RGB image (width, height, rgb)
        return (
            ((vae_output / 2 + 0.5).transpose(0, 2, 3, 1).clip(0, 1) * 255)
            .round()
            .astype(jnp.uint8)[0]
        )

    jax_jit_compiled_predict_image = jax.jit(__predict_image)

    return lambda prompt: __convert_image(
        jax_jit_compiled_predict_image(__tokenize_prompt(prompt))
    )


if __name__ == "__main__":

    wandb_inference_init()

    generate_image_for_prompt = get_inference_lambda(87)

    prompts = [
        "a white car",
        "a running shoe",
        "a forest",
        "two people",
        "a happy cartoon cat",
    ]

    log = []

    for prompt in prompts:

        log.append({"prompt": prompt, "image": generate_image_for_prompt(prompt)})

    wandb_inference_log(log)

    wandb_close()
