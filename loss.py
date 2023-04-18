import jax
import jax.numpy as jnp


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
    # Convert images to latent space
    vae_outputs = vae.apply(
        {"params": vae_params},
        batch["pixel_values"],
        deterministic=True,
        method=vae.encode,
    )
    latents = vae_outputs.latent_dist.sample(sample_rng)
    # (NHWC) -> (NCHW)
    latents = jnp.transpose(latents, (0, 3, 1, 2))
    latents = latents * vae.config.scaling_factor

    # Sample noise that we'll add to the latents
    noise_rng, timestep_rng = jax.random.split(sample_rng)
    noise = jax.random.normal(noise_rng, latents.shape)
    # Sample a random timestep for each image
    bsz = latents.shape[0]
    timesteps = jax.random.randint(
        timestep_rng,
        (bsz,),
        0,
        noise_scheduler.config.num_train_timesteps,
    )

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(
        noise_scheduler_state, latents, noise, timesteps
    )

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(
        batch["input_ids"],
        params=text_encoder_params,
        train=False,
    )[0]

    # Predict the noise residual and compute loss
    model_pred = unet.apply(
        {"params": state_params},
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        train=True,
    ).sample

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(
            noise_scheduler_state, latents, noise, timesteps
        )
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        )

    loss = (target - model_pred) ** 2
    loss = loss.mean()

    return loss
