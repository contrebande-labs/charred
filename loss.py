import jax.numpy as jnp
import jax

from diffusers import FlaxDDPMScheduler


# Min-SNR
snr_gamma = 5.0  # SNR weighting gamma to be used when rebalancing the loss with Min-SNR. Recommended value is 5.0.


def compute_snr_loss_weights(noise_scheduler_state, timesteps):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler_state.common.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    alpha = sqrt_alphas_cumprod[timesteps]
    sigma = sqrt_one_minus_alphas_cumprod[timesteps]
    # Compute SNR.
    snr = jnp.array((alpha / sigma) ** 2)

    # Compute SNR loss weights
    return jnp.where(snr < snr_gamma, snr, jnp.ones_like(snr) * snr_gamma) / snr


def get_vae_latent_distribution_samples(
    image_latent_distribution_sampling,
    sample_rng,
    scaling_factor,
    noise_scheduler,
    noise_scheduler_state,
):
    # (NHWC) -> (NCHW)
    latents = (
        jnp.transpose(image_latent_distribution_sampling, (0, 3, 1, 2)) * scaling_factor
    )

    # Sample noise that we'll add to the latents
    noise_rng, timestep_rng = jax.random.split(sample_rng)
    noisy_image_target = jax.random.normal(noise_rng, latents.shape)

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
        noise_scheduler_state, latents, noisy_image_target, timesteps
    )

    return noisy_latents, timesteps, noisy_image_target


def get_compute_loss_lambda(
    text_encoder,
    text_encoder_params,
    vae,
    vae_params,
    unet,
    batch,
    sample_rng,
):
    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        num_train_timesteps=1000,
    )

    noise_scheduler_state = noise_scheduler.create_state()

    def __compute_loss_lambda(
        state_params,
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
            sample=batch["pixel_values"],
            deterministic=True,
            method=vae.encode,
        )
        # vae_outputs.latent_dist.mode() # <--- can this be cached ?
        image_latent_distribution_sampling = vae_outputs.latent_dist.sample(sample_rng)
        (
            image_sampling_noisy_input,
            image_sampling_timesteps,
            image_sampling_noisy_target,
        ) = get_vae_latent_distribution_samples(
            image_latent_distribution_sampling,
            sample_rng,
            vae.config.scaling_factor,
            noise_scheduler,
            noise_scheduler_state,
        )

        # Predict the noise residual and compute loss
        model_pred = unet.apply(
            {"params": state_params},
            sample=image_sampling_noisy_input,
            timesteps=image_sampling_timesteps,
            encoder_hidden_states=text_encoder_hidden_states,
            train=True,
        ).sample

        # Compute Min-SNR loss weights
        snr_loss_weights = compute_snr_loss_weights(
            noise_scheduler_state, image_sampling_timesteps
        ).average(axis=0)

        # Compute loss from noisy target
        loss = ((image_sampling_noisy_target - model_pred) ** 2)

        # Balance loss with Min-SNR
        min_snr_loss = (loss * snr_loss_weights).mean()

        return min_snr_loss

    return __compute_loss_lambda
