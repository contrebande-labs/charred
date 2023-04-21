import jax

from diffusers import (
    FlaxDDPMScheduler,
)

from loss import get_loss_lambda


def get_train_step_lambda(text_encoder, vae, unet):

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    noise_scheduler_state = noise_scheduler.create_state()

    # TODO: can we precompile the loss function higher up, maybe in the main function or main training loop init?
    loss_lambda = get_loss_lambda(
        text_encoder,
        vae,
        unet,
        noise_scheduler,
        noise_scheduler_state,
    )

    grad_loss = jax.value_and_grad(loss_lambda)

    def __train_step_lambda(
        state,
        text_encoder_params,
        vae_params,
        batch,
        rng,
    ):

        sample_rng, new_rng = jax.random.split(rng, 2)

        loss, grad = grad_loss(state.params, text_encoder_params, vae_params, batch, sample_rng)

        grad_mean = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad_mean)

        metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")

        return new_state, new_rng, metrics

    return __train_step_lambda
