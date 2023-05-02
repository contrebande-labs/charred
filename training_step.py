import jax

from loss import get_compute_losses_lambda


def get_training_step_lambda(text_encoder, vae, unet):
    def __training_step_lambda(
        state,
        text_encoder_params,
        vae_params,
        batch,
        rng,
    ):
        sample_rng, new_rng = jax.random.split(rng, 2)

        # TODO: move this compilation higher up
        jax_loss_value_and_gradient = jax.value_and_grad(
            get_compute_losses_lambda(
                text_encoder,
                text_encoder_params,
                vae,
                vae_params,
                unet,
            )
        )

        # Compute loss and gradients
        # TODO: is this correct?
        loss, grad = jax.lax.pmean(
            jax_loss_value_and_gradient(
                state.params, 
                batch,
                sample_rng,
            ),
            axis_name="batch",
        )

        new_state = state.apply_gradients(grads=grad)

        metrics = { "loss": loss }

        return new_state, new_rng, metrics

    return __training_step_lambda
