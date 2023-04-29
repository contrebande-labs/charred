import os

# jax/flax
import jax
from flax import jax_utils
from flax.core.frozen_dict import unfreeze
from flax.training import train_state

from architecture import setup_model

# internal code
from args import parse_args
from optimizer import setup_optimizer
from training_loop import training_loop
from monitoring import wandb_close, wandb_init


def main():
    args = parse_args()

    output_dir = args.output_dir

    load_pretrained = os.path.exists(output_dir) and os.path.isdir(output_dir)

    # Setup WandB for logging & tracking
    log_wandb = args.log_wandb
    if log_wandb:
        wandb_init(args)

    # init random number generator
    seed = args.seed
    seed_rng = jax.random.PRNGKey(seed)
    rng, training_from_scratch_rng_params = jax.random.split(seed_rng)
    print("random generator setup...")

    # Pretrained/freezed and training model setup
    text_encoder, text_encoder_params, vae, vae_params, unet, unet_params = setup_model(
        seed,
        args.mixed_precision,
        load_pretrained,
        output_dir,
        training_from_scratch_rng_params,
    )
    print("models setup...")

    # Optimization & scheduling setup
    optimizer = setup_optimizer(
        args.learning_rate,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_epsilon,
        args.adam_weight_decay,
        args.max_grad_norm,
    )
    print("optimizer setup...")

    # Training state setup
    unet_training_state = train_state.TrainState.create(
        apply_fn=unet,
        params=unfreeze(unet_params),
        tx=optimizer,
    )
    print("training state initialized...")

    if log_wandb:
        log_wandb = False
        get_validation_predictions = None
    else:
        get_validation_predictions = None

    # JAX device data replication
    replicated_state = jax_utils.replicate(unet_training_state)
    replicated_text_encoder_params = jax_utils.replicate(text_encoder_params)
    replicated_vae_params = jax_utils.replicate(vae_params)
    print("states & params replicated to TPUs...")

    # Train!
    print("Training loop init...")
    training_loop(
        text_encoder,
        replicated_text_encoder_params,
        vae,
        replicated_vae_params,
        unet,
        replicated_state,
        rng,
        args.max_train_steps,
        args.num_train_epochs,
        args.train_batch_size,
        output_dir,
        log_wandb,
        get_validation_predictions,
    )
    print("Training loop done...")

    if log_wandb:
        wandb_close()


if __name__ == "__main__":
    main()
