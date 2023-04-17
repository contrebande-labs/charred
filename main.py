# jax/flax
import jax
from flax import jax_utils
from flax.training import train_state

# internal code
from args import parse_args
from architecture import setup_model
from monitoring import setup_logging
from optimizer import setup_optimizer
from repository import create_repository, save_to_repository
from training_loop import training_loop

def main():
    args = parse_args()

    logger = setup_logging()

    if jax.process_index() == 0:

        repo_id = create_repository(args.output_dir, args.push_to_hub, args.hub_model_id, args.hub_token)

        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # pipeline models setup
    tokenizer, text_encoder, vae, vae_params, unet, unet_params = setup_model(
        args.seed, args.mixed_precision,
        args.pretrained_text_encoder_model_name_or_path, args.pretrained_text_encoder_model_revision, 
        args.pretrained_diffusion_model_name_or_path, args.pretrained_diffusion_model_revision)

    # Optimization & Scheduling
    optimizer = setup_optimizer(args.learning_rate, args.adam_beta1, args.adam_beta2, args.adam_epsilon, args.adam_weight_decay, args.max_grad_norm)

    # State
    state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    # Replicate the train state on each device
    replicated_state = jax_utils.replicate(state)
    replicated_text_encoder_params = jax_utils.replicate(text_encoder.params)
    replicated_vae_params = jax_utils.replicate(vae_params)

    # Train!
    training_loop(tokenizer, text_encoder, replicated_text_encoder_params, vae, replicated_vae_params, unet, replicated_state,
        args.dataset_name, args.dataset_config_name, args.cache_dir, args.train_data_dir, args.caption_column, args.image_column,
        args.resolution, args.center_crop, args.random_flip,
        args.seed, args.max_train_steps, args.num_train_epochs, args.max_train_samples, args.train_batch_size)

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        save_to_repository(args.output_dir, args.push_to_hub, 
            tokenizer, text_encoder, replicated_text_encoder_params, vae, replicated_vae_params, unet, repo_id, replicated_state)

if __name__ == "__main__":
    main()