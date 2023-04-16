# jax/flax
import jax
from flax import jax_utils
from flax.training import train_state

# internal code
from args import parse_args
from architecture import setup_model
from optimizer import setup_optimizer
from repository import create_repository, save_to_repository
from training_loop import training_loop

def main():
    args = parse_args()

    if jax.process_index() == 0:
        repo_id = create_repository(args)

    # pipeline models setup
    tokenizer, text_encoder, vae, vae_params, unet, unet_params = setup_model(
        args.seed, args.mixed_precision,
        args.pretrained_text_encoder_model_name_or_path, args.pretrained_text_encoder_model_revision, 
        args.pretrained_diffusion_model_name_or_path, args.pretrained_diffusion_model_revision)

    # Optimization & Scheduling
    optimizer = setup_optimizer(args)

    # State
    state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    # Replicate the train state on each device
    replicated_state = jax_utils.replicate(state)
    replicated_text_encoder_params = jax_utils.replicate(text_encoder.params)
    replicated_vae_params = jax_utils.replicate(vae_params)

    # Train!
    training_loop(args, tokenizer, text_encoder, replicated_text_encoder_params, vae, vae_params, unet)

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        save_to_repository(args, tokenizer, text_encoder, replicated_text_encoder_params, vae, replicated_vae_params, unet, repo_id, replicated_state)

if __name__ == "__main__":
    main()