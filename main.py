# jax/flax
import jax
from flax import jax_utils
from flax.training import train_state
from flax.core.frozen_dict import unfreeze
import wandb

from diffusers import (
    FlaxUNet2DConditionModel,
)

# internal code
from args import parse_args
from architecture import setup_model
from optimizer import setup_optimizer
from repository import create_repository
from training_loop import training_loop


def main():

    args = parse_args()

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
        repo_id = create_repository(
            args.output_dir, args.push_to_hub, args.hub_model_id, args.hub_token
        )

    # init random number generator
    seed = args.seed
    seed_rng = jax.random.PRNGKey(seed)
    rng, rng_params = jax.random.split(seed_rng)

    # Pretrained freezed model setup
    tokenizer, text_encoder, vae, vae_params, unet = setup_model(
        seed,
        args.mixed_precision,
        args.pretrained_text_encoder_model_name_or_path,
        args.pretrained_diffusion_model_name_or_path,
    )

    # Optimization & scheduling setup
    optimizer = setup_optimizer(
        args.learning_rate,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_epsilon,
        args.adam_weight_decay,
        args.max_grad_norm,
    )

    # State setup
    replicated_state = jax_utils.replicate(
        train_state.TrainState.create(
            apply_fn=unet,
            params=unfreeze(unet.init_weights(rng=rng_params)),
            tx=optimizer,
        )
    )
    replicated_text_encoder_params = jax_utils.replicate(text_encoder.params)
    replicated_vae_params = jax_utils.replicate(vae_params)

    # Train!
    training_loop(
        tokenizer,
        args.tokenizer_max_length,
        text_encoder,
        replicated_text_encoder_params,
        vae,
        replicated_vae_params,
        unet,
        replicated_state,
        args.cache_dir,
        args.resolution,
        rng,
        args.max_train_steps,
        args.num_train_epochs,
        args.train_batch_size,
        args.output_dir,
        args.dataset_output_dir,
        repo_id,
        log_wandb,
    )


if __name__ == "__main__":
    main()
