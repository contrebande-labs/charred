import jax

from diffusers import (
    FlaxDDPMScheduler,
)

from loss import loss_fn

def train_step(text_encoder, vae, unet):

    # TODO: can we cahe the scheduler higher up, maybe in the main function or main training loop init?
    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    noise_scheduler_state = noise_scheduler.create_state()
 
    train_step_lambda = lambda state, text_encoder_params, vae_params, batch, train_rng:_train_step(
    text_encoder, vae, unet,
    noise_scheduler, noise_scheduler_state,
    state, text_encoder_params, vae_params, batch, train_rng)

    # TODO:  can this be cached somewhere for reuse ?
    # Create parallel version of the train step
    return jax.pmap(train_step_lambda, "batch", donate_argnums=(0,))

def _train_step(text_encoder, vae, unet, noise_scheduler, noise_scheduler_state, state, text_encoder_params, vae_params, batch, train_rng):

    sample_rng, new_train_rng = jax.random.split(train_rng, 2)

    # TODO: can we precompile the loss funtion higher up, maybe in the main function or main training loop init?
    loss_lambda = loss_fn(vae, vae_params, batch, sample_rng, noise_scheduler, noise_scheduler_state, text_encoder, text_encoder_params, unet)
    grad_fn = jax.value_and_grad(loss_lambda)

    loss, grad = grad_fn(state.params)
    grad_mean = jax.lax.pmean(grad, "batch")

    new_state = state.apply_gradients(grads=grad_mean)

    metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")

    return new_state, metrics, new_train_rng