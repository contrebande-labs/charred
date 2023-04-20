import jax

from diffusers import (
    FlaxDDPMScheduler,
)

from loss import loss_fn


def train_step(text_encoder, vae, unet):

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    noise_scheduler_state = noise_scheduler.create_state()

    train_step_lambda = lambda state, text_encoder_params, vae_params, batch, train_rng: _train_step(
        text_encoder,
        text_encoder_params,
        vae,
        vae_params,
        unet,
        state,
        noise_scheduler,
        noise_scheduler_state,
        batch,
        train_rng,
    )

    # Create parallel version of the train step
    return jax.pmap(train_step_lambda, "batch", donate_argnums=(0,))


def _train_step(
    text_encoder,
    text_encoder_params,
    vae,
    vae_params,
    unet,
    state,
    noise_scheduler,
    noise_scheduler_state,
    batch,
    rng,
):

    sample_rng, new_rng = jax.random.split(rng, 2)

    # TODO: can we precompile the loss function higher up, maybe in the main function or main training loop init?
    loss_lambda = loss_fn(
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
    grad_fn = jax.value_and_grad(loss_lambda)

    loss, grad = grad_fn(state.params)
    grad_mean = jax.lax.pmean(grad, "batch")

    new_state = state.apply_gradients(grads=grad_mean)

    metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")

    return new_state, new_rng, metrics
