import jax

from loss import get_compute_loss_lambda


def get_training_step_lambda(text_encoder, vae, unet):
    def __training_step_lambda(
        state,
        text_encoder_params,
        vae_params,
        batch,
        rng,
    ):
        sample_rng, new_rng = jax.random.split(rng, 2)

        compute_loss_lambda = get_compute_loss_lambda(
            text_encoder,
            text_encoder_params,
            vae,
            vae_params,
            unet,
            batch,
            sample_rng,
        )

        jax_grad_value_loss = jax.value_and_grad(compute_loss_lambda)

        loss, grad = jax_grad_value_loss(state.params)

        grad_mean = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad_mean)

        metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")

        return new_state, new_rng, metrics

    return __training_step_lambda
