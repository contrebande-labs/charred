import jax

from loss import get_compute_loss_lambda


def get_training_step_lambda(text_encoder, vae, unet):

    loss_lambda = get_compute_loss_lambda(
        text_encoder,
        vae,
        unet,
    )

    jax_value_and_grad_loss = jax.value_and_grad(loss_lambda)

    def __training_step_lambda(
        state,
        text_encoder_params,
        vae_params,
        batch,
        rng,
    ):

        sample_rng, new_rng = jax.random.split(rng, 2)

        loss, grad = jax_value_and_grad_loss(
            state.params, text_encoder_params, vae_params, batch, sample_rng
        )

        grad_mean = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad_mean)

        metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")

        return new_state, new_rng, metrics

    return __training_step_lambda
