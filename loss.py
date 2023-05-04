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
    noise = jax.random.normal(noise_rng, latents.shape)

    # Sample a random timestep for each image
    timesteps = jax.random.randint(
        key=timestep_rng,
        shape=(latents.shape[0],),
        minval=0,
        maxval=noise_scheduler.config.num_train_timesteps,
        dtype=jnp.int32,
    )

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(
        state=noise_scheduler_state,
        original_samples=latents,
        noise=noise,
        timesteps=timesteps,
    )

    return noisy_latents, timesteps, noise


def get_cacheable_samples(text_encoder, text_encoder_params, input_ids, vae, vae_params, pixel_values, rng):

        # Get the text embedding
        # TODO: Cache this
        text_encoder_hidden_states = text_encoder(
            input_ids,
            params=text_encoder_params,
            train=False,
        )[0]

        # Get the image embedding
        vae_outputs = vae.apply(
            {"params": vae_params},
            sample=pixel_values,
            deterministic=True,
            method=vae.encode,
        )

        # Sample the image embedding
        # TODO: Cache this
        image_latent_distribution_sampling = vae_outputs.latent_dist.sample(rng)

        return text_encoder_hidden_states, image_latent_distribution_sampling

def get_compute_losses_lambda(
    text_encoder, # <-- TODO: take this out of here
    text_encoder_params, # <-- TODO: take this out of here
    vae, # <-- TODO: take this out of here
    vae_params, # <-- TODO: take this out of here
    unet,
):
    # Instanciate training noise scheduler
    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        num_train_timesteps=1000,
    )

    vae_scaling_factor = vae.config.scaling_factor # <-- TODO: take this out of here

    def __compute_losses_lambda(
        state_params,
        batch,
        sample_rng,
    ):
        
        # TODO: take this out of here
        text_encoder_hidden_states, image_latent_distribution_sampling = get_cacheable_samples(
             text_encoder,
             text_encoder_params,
             batch["input_ids"],
             vae,
             vae_params,
             batch["pixel_values"],
             sample_rng,
        )

        # initialize scheduler state
        noise_scheduler_state = noise_scheduler.create_state()

        # Get the vae latent distribution samples
        (
            image_sampling_noisy_input,
            image_sampling_timesteps,
            noise,
        ) = get_vae_latent_distribution_samples(
            image_latent_distribution_sampling,
            sample_rng,
            vae_scaling_factor,
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
            noise_scheduler_state,
            image_sampling_timesteps,
        )

        # Compute each batch sample's loss from noisy target
        loss_tensors = (noise - model_pred) ** 2

        # Get one loss scalar per batch sample
        losses = (
            loss_tensors.mean(
                axis=tuple(range(1, loss_tensors.ndim)),
            )
            * snr_loss_weights
        )  # Balance losses with Min-SNR

        # This must be an averaged scalar, otherwise, you get this:TypeError: Gradient only defined for scalar-output functions. Output had shape: (8,).
        return losses.mean(axis=0)

    return __compute_losses_lambda
