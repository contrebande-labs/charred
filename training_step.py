import jax

from loss import get_compute_losses_lambda


def get_training_step_lambda(text_encoder, text_encoder_params, vae, vae_params, unet):

    # Compile loss function
    jax_loss_value_and_gradient = jax.value_and_grad(
        get_compute_losses_lambda(
            text_encoder,
            text_encoder_params,
            vae,
            vae_params,
            unet,
        )
    )
 
    def __training_step_lambda(
        batch,
        rng,
        state,
    ):
 
        # Split RNGs
        sample_rng, new_rng = jax.random.split(rng, 2)

        # Compute loss and gradients
        # TODO: is this correct?
        loss, grad = jax.lax.pmean(
            jax_loss_value_and_gradient(
                batch,
                sample_rng,
                state.params,
            ),
            axis_name="batch",
        )

        # Apply gradients to training state
        new_state = state.apply_gradients(grads=grad)

        # Create metrics dict
        # TODO: add more metrics here
        metrics = { "loss": loss }

        return new_state, new_rng, metrics

    return __training_step_lambda
