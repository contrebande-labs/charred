# jax/flax
import jax
from flax import jax_utils
from flax.training import train_state

# hugging face
from diffusers.utils import check_min_version

# internal code
from args import parse_args
from model import setup_model
from optimizer import setup_optimizer
from repository import create_repository, save_to_repository
from training_loop import training_loop

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.16.0.dev0")


def main():
    args = parse_args()

    if jax.process_index() == 0:
        repo_id = create_repository(args)

    # pipeline models setup
    tokenizer, text_encoder, vae, vae_params, unet, unet_params = setup_model(args)

    # Optimization & Scheduling
    optimizer = setup_optimizer(args)

    # State
    state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    # Replicate the train state on each device
    state = jax_utils.replicate(state)
    text_encoder_params = jax_utils.replicate(text_encoder.params)
    vae_params = jax_utils.replicate(vae_params)

    # Train!
    training_loop(args, tokenizer, text_encoder, text_encoder_params, vae, vae_params, unet)

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        save_to_repository(args, tokenizer, text_encoder, text_encoder_params, vae, vae_params, unet, repo_id, state)

if __name__ == "__main__":
    main()