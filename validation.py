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

from architecture import setup_model


def get_validation_predictions_lambda(
    seed,
    text_encoder: FlaxT5ForConditionalGeneration,
    text_encoder_params,
    vae: FlaxAutoencoderKL,
    vae_params,
    unet: FlaxUNet2DConditionModel,
    prompts: list[str],
):
    tokenizer = ByT5Tokenizer()
    tokenized_prompt_max_length = 1024
    tokenized_negative_prompt = tokenizer(
        "",
        padding="max_length",
        max_length=tokenized_prompt_max_length,
        return_tensors="jax",
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

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    image_width = image_height = 256

    # Generating latent shape
    latent_shape = (
        negative_prompt_text_encoder_hidden_states.shape[
            0
        ],  # TODO: if is this for the whole context (positive + negative prompts), we should multiply by two
        unet.in_channels,
        image_width // vae_scale_factor,
        image_height // vae_scale_factor,
    )

    def __tokenize_prompts(prompt: list[str]):
        return tokenizer(
            text=prompt,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="jax",
        ).input_ids

    def __convert_images(images):
        # create PIL image from JAX tensor converted to numpy
        return [Image.fromarray(np.asarray(image), mode="RGB") for image in images]

    def __get_context(tokenized_prompt: jnp.array):
        # Get the text embedding
        text_encoder_hidden_states = text_encoder(
            tokenized_prompt,
            params=text_encoder_params,
            train=False,
        )[0]

        # context = empty negative prompt embedding + prompt embedding
        return jnp.concatenate(
            [negative_prompt_text_encoder_hidden_states, text_encoder_hidden_states]
        )

    get_context = jax.jit(__get_context, device=jax.devices(backend="cpu")[0])

    context = get_context(prompts)

    def __predict_images(seed, unet_params):
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
            .astype(jnp.uint8)[0]
        )

    jax_jit_compiled_predict_images = jax.jit(__predict_images)

    return lambda unet_params: zip(
        prompts,
        __convert_images(
            jax_jit_compiled_predict_images(
                seed, unet_params, __tokenize_prompts(prompts)
            )
        ),
    )


if __name__ == "__main__":
    # Pretrained/freezed and training model setup
    text_encoder, text_encoder_params, vae, vae_params, unet, unet_params = setup_model(
        43,  # seed
        None,  # dtype (defaults to float32)
        True,  # load pre-trained
        "character-aware-diffusion/charred",
        None,
    )
