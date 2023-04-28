import jax
import jax
import jax.numpy as jnp

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
    timesteps = 50
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

    print("all models setup")

    def __tokenize_prompt(prompt: str):

        return tokenizer(
            text=prompt,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="jax",
        ).input_ids.astype(jnp.float32)

    def __convert_image(vae_output):
        print("skipping image conversion...")
        return None
        # return [
        #     Image.fromarray(image)
        #     for image in (np.asarray(vae_output) * 255).round().astype(np.uint8)
        # ]

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
        jax.debug.print("got text encoding...")

        latent_shape = (
            tokenized_prompt.shape[0],
            unet.in_channels,
            image_width // vae_scale_factor,
            image_height // vae_scale_factor,
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

            jax.debug.print("did one step...")

            return latents, scheduler_state

        # initialize scheduler state
        initial_scheduler_state = scheduler.set_timesteps(
            scheduler.create_state(), num_inference_steps=timesteps, shape=latent_shape
        )
        jax.debug.print("initialized scheduler state...")

        # initialize latents
        initial_latents = (
            jax.random.normal(
                jax.random.PRNGKey(seed), shape=latent_shape, dtype=jnp.float32
            )
            * initial_scheduler_state.init_noise_sigma
        )
        jax.debug.print("initialized latents...")

        final_latents, _ = jax.lax.fori_loop(
            0, timesteps, ___timestep, (initial_latents, initial_scheduler_state)
        )
        jax.debug.print("got final latents...")

        jax.debug.print("got final latents...")

        # scale and decode the image latents with vae
        image = (
            (
                vae.apply(
                    {"params": vae_params},
                    1 / vae.config.scaling_factor * final_latents,
                    method=vae.decode,
                ).sample
                / 2
                + 0.5
            )
            .clip(0, 1)
            .transpose(0, 2, 3, 1)
        )
        jax.debug.print("got vae processed image output...")

        jax.debug.print("got vae decoded image output...")

        # return reshaped vae outputs
        return image

    jax_pmap_predict_image = jax.jit(__predict_image)

    return lambda prompt: __convert_image(
        jax_pmap_predict_image(__tokenize_prompt(prompt))
    )


if __name__ == "__main__":

    # wandb_init(None)

    generate_image_for_prompt = get_inference_lambda(87)

    generate_image_for_prompt("a white car")
    # wandb_close()
