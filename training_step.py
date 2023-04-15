import jax.numpy as jnp
import jax

from diffusers import (
    FlaxDDPMScheduler,
)

def train_step(text_encoder, vae, unet):

  noise_scheduler = FlaxDDPMScheduler(
      beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
  )

  noise_scheduler_state = noise_scheduler.create_state()

  train_step_lambda = lambda state, text_encoder_params, vae_params, batch, train_rng:_train_step(
    text_encoder, vae, unet,
    noise_scheduler, noise_scheduler_state,
    state, text_encoder_params, vae_params, batch, train_rng)

  # Create parallel version of the train step
  return jax.pmap(train_step_lambda, "batch", donate_argnums=(0,))

def _train_step(text_encoder, vae, unet, noise_scheduler, noise_scheduler_state, state, text_encoder_params, vae_params, batch, train_rng):
    sample_rng, new_train_rng = jax.random.split(train_rng, 2)

    def compute_loss(params):
        # Convert images to latent space
        vae_outputs = vae.apply(
            {"params": vae_params}, batch["pixel_values"], deterministic=True, method=vae.encode
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
        noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(
            batch["input_ids"],
            params=text_encoder_params,
            train=False,
        )[0]

        # Predict the noise residual and compute loss
        model_pred = unet.apply(
            {"params": params}, noisy_latents, timesteps, encoder_hidden_states, train=True
        ).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = (target - model_pred) ** 2
        loss = loss.mean()

        return loss

    grad_fn = jax.value_and_grad(compute_loss)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")

    new_state = state.apply_gradients(grads=grad)

    metrics = {"loss": loss}
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    return new_state, metrics, new_train_rng