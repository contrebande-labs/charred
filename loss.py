import jax.numpy as jnp
import jax


def loss_fn(
    vae,
    vae_params,
    batch,
    sample_rng,
    noise_scheduler,
    noise_scheduler_state,
    text_encoder,
    text_encoder_params,
    unet,
):
    return lambda state_params: _loss_fn(
        state_params,
        vae,
        vae_params,
        batch,
        sample_rng,
        noise_scheduler,
        noise_scheduler_state,
        text_encoder,
        text_encoder_params,
        unet,
    )


def _loss_fn(
    state_params,
    vae,
    vae_params,
    batch,
    sample_rng,
    noise_scheduler,
    noise_scheduler_state,
    text_encoder,
    text_encoder_params,
    unet,
):

    # Get the image and text embeddings
    # TODO: use cached embeddings instead
    vae_outputs = vae.apply(
        {"params": vae_params},
        batch["pixel_values"],
        deterministic=True,
        method=vae.encode,
    )
    encoder_hidden_states = text_encoder(
        batch["input_ids"],
        params=text_encoder_params,
        train=False,
    )[0]

    # Convert image embeddings to latent space
    latent_samples = vae_outputs.latent_dist.sample(sample_rng)
    latents_transposed = jnp.transpose(latent_samples, (0, 3, 1, 2))  # (NHWC) -> (NCHW)
    latents = latents_transposed * vae.config.scaling_factor

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

    # Predict the noise residual and compute loss
    model_pred = unet.apply(
        {"params": state_params},
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        train=True,
    ).sample

    # Compute loss from noisy target
    return ((noise - model_pred) ** 2).mean()
