# jax/flax
import jax
from flax import jax_utils
from flax.training import train_state
from flax.core.frozen_dict import unfreeze
import wandb

# internal code
from args import parse_args
from architecture import setup_model
from optimizer import setup_optimizer
from repository import create_repository
from training_loop import training_loop


def main():

    args = parse_args()

    output_dir = args.output_dir

    # Setup WandB for logging & tracking
    log_wandb = args.log_wandb
    if log_wandb:
        wandb.init(
            entity="charred",
            project="charred",
            job_type="train",
            config=args,
        )
        wandb.config.update(
            {
                "num_devices": jax.device_count(),
            }
        )
        wandb.define_metric("*", step_metric="train/step")
        wandb.define_metric("train/step", step_metric="walltime")

    if args.push_to_hub:
        repo_id = create_repository(output_dir, args.hub_model_id)
        print("connected to hugging face model git repo...")
    else:
        repo_id = None

    # init random number generator
    seed = args.seed
    seed_rng = jax.random.PRNGKey(seed)
    rng, rng_params = jax.random.split(seed_rng)
    print("random generator setup...")

    # Pretrained freezed model setup
    text_encoder, text_encoder_params, vae, vae_params, unet = setup_model(
        seed,
        args.mixed_precision,
    )
    print("freezed models setup...")

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
        params=unfreeze(unet.init_weights(rng=rng_params)),
        tx=optimizer,
    )
    print("training state initialized...")

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
        repo_id,
        log_wandb,
    )
    print("Training loop done...")

    if log_wandb:
        wandb.finish()
        print("WandB closed...")


if __name__ == "__main__":
    main()
