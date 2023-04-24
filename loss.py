import jax.numpy as jnp
import jax

from diffusers import FlaxDDPMScheduler


def get_vae_latent_distribution_samples(
    image_latent_distribution_sampling,
    sample_rng,
    scaling_factor,
    noise_scheduler,
    noise_scheduler_state,
):
    # (NHWC) -> (NCHW)
    latents = jnp.transpose(image_latent_distribution_sampling, (0, 3, 1, 2)) * scaling_factor

    # Sample noise that we'll add to the latents
    noise_rng, timestep_rng = jax.random.split(sample_rng)
    noise = jax.random.normal(noise_rng, latents.shape)

    # Sample a random timestep for each image
    timesteps = jax.random.randint(
        timestep_rng,
        (latents.shape[0],),
        0,
        noise_scheduler.config.num_train_timesteps,
    )

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(
        noise_scheduler_state, latents, noise, timesteps
    )

    return noisy_latents, timesteps, noise


def get_compute_loss_lambda(
    text_encoder,
    vae,
    unet,
):

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    noise_scheduler_state = noise_scheduler.create_state()

    def __compute_loss_lambda(
        state,
        text_encoder_params,
        vae_params,
        batch,
        sample_rng,
    ):

        # Get the text embedding
        text_encoder_hidden_states = text_encoder(
            batch["input_ids"],
            params=text_encoder_params,
            train=False,
        )[0]

        # Get the image embedding
        vae_outputs = vae.apply(
            {"params": vae_params},
            batch["pixel_values"],
            deterministic=True,
            method=vae.encode,
        )
        # vae_outputs.latent_dist.mode() # <--- can this be cached ?
        image_latent_distribution_sampling = vae_outputs.latent_dist.sample(sample_rng)
        (
            image_sampling_noisy_latents,
            image_sampling_timesteps,
            image_sampling_noise,
        ) = get_vae_latent_distribution_samples(
            image_latent_distribution_sampling,
            sample_rng,
            vae.config.scaling_factor,
            noise_scheduler,
            noise_scheduler_state,
        )

        # Predict the noise residual and compute loss
        model_pred = unet.apply(
            {"params": state},
            image_sampling_noisy_latents,
            image_sampling_timesteps,
            text_encoder_hidden_states,
            train=True,
        ).sample

        # Compute loss from noisy target
        return ((image_sampling_noise - model_pred) ** 2).mean()

    return __compute_loss_lambda
