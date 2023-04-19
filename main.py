# jax/flax
import jax
from flax import jax_utils
from flax.training import train_state
from flax.core.frozen_dict import unfreeze

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

    # init random number generator
    seed = args.seed
    seed_rng = jax.random.PRNGKey(seed)
    rng, rng_params = jax.random.split(seed_rng)

    repo_id = create_repository(args.output_dir, args.push_to_hub, args.hub_model_id, args.hub_token)

    # Pretrained freezed model setup
    tokenizer, text_encoder, vae, vae_params, ref_unet = setup_model(
        seed, args.mixed_precision,
        args.pretrained_text_encoder_model_name_or_path, args.pretrained_text_encoder_model_revision, 
        args.pretrained_diffusion_model_name_or_path, args.pretrained_diffusion_model_revision)
    
    # Create new U-Net to pre-train from scratch
    unet = FlaxUNet2DConditionModel(
        in_channels=ref_unet.config.in_channels,
        down_block_types=ref_unet.config.down_block_types,
        only_cross_attention=ref_unet.config.only_cross_attention,
        block_out_channels=ref_unet.config.block_out_channels,
        layers_per_block=ref_unet.config.layers_per_block,
        attention_head_dim=ref_unet.config.attention_head_dim,
        cross_attention_dim=ref_unet.config.cross_attention_dim,
        use_linear_projection=ref_unet.config.use_linear_projection,
        flip_sin_to_cos=ref_unet.config.flip_sin_to_cos,
        freq_shift=ref_unet.config.freq_shift,
    )
    unet_params = unfreeze(unet.init_weights(rng=rng_params))

    # Optimization & scheduling setup
    optimizer = setup_optimizer(args.learning_rate, args.adam_beta1, args.adam_beta2, args.adam_epsilon, args.adam_weight_decay, args.max_grad_norm)

    # State setup
    # TODO: find out why we are passing params=unet_params. Only for shape?
    replicated_state = jax_utils.replicate(train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer))
    replicated_text_encoder_params = jax_utils.replicate(text_encoder.params)
    replicated_vae_params = jax_utils.replicate(vae_params)

    # Train!
    training_loop(tokenizer, text_encoder, replicated_text_encoder_params, vae, replicated_vae_params, unet, replicated_state,
        args.cache_dir, args.resolution,
        rng, args.max_train_steps, args.num_train_epochs, args.train_batch_size,
        args.output_dir, args.dataset_output_dir, args.push_to_hub, repo_id, log_wandb=args.log_wandb)

if __name__ == "__main__":
    main()
