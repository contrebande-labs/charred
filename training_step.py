import jax

from loss import get_compute_losses_lambda


def get_training_step_lambda(text_encoder, text_encoder_params, vae, vae_params, unet):
    def __training_step_lambda(
        batch,
        rng,
        state,
    ):

        # Split RNGs
        sample_rng, new_rng = jax.random.split(rng, 2)

        # Get loss function lambda
        compute_batch_losses = get_compute_losses_lambda(
            text_encoder,
            text_encoder_params,
            vae,
            vae_params,
            unet,
            batch,
            sample_rng,
        )

        # Compile loss function.
        # NOTE: Can't have this compiled higher up because jax.value_and_grad-compiled functions require real numbers (floating point) dtypes as arguments
        jax_loss_value_and_gradient = jax.value_and_grad(compute_batch_losses)

        # Compute loss and gradients
        loss, grad = jax_loss_value_and_gradient(
            state.params,
        )

        # Apply gradients to training state
        new_state = state.apply_gradients(grads=grad)

        return new_state, new_rng, loss

    return __training_step_lambda
