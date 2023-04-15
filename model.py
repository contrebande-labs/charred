import jax.numpy as jnp

from transformers import ByT5Tokenizer, FlaxT5EncoderModel, set_seed

from diffusers import (
    FlaxAutoencoderKL,
    FlaxUNet2DConditionModel,
)

def setup_model(args):
  if args.seed is not None:
      set_seed(args.seed)

  weight_dtype = jnp.float32

  if args.mixed_precision == "fp16":
      weight_dtype = jnp.float16
  elif args.mixed_precision == "bf16":
      weight_dtype = jnp.bfloat16

  # Load models and create wrapper for stable diffusion
  tokenizer = ByT5Tokenizer.from_pretrained(
      args.pretrained_model_name_or_path, revision=args.revision, subfolder="tokenizer"
  )
  text_encoder = FlaxT5EncoderModel.from_pretrained(

      args.pretrained_model_name_or_path, revision=args.revision, subfolder="text_encoder", dtype=weight_dtype
  )
  vae, vae_params = FlaxAutoencoderKL.from_pretrained(
      args.pretrained_model_name_or_path, revision=args.revision, subfolder="vae", dtype=weight_dtype
  )
  unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
      args.pretrained_model_name_or_path, revision=args.revision, subfolder="unet", dtype=weight_dtype
  )

  return tokenizer, text_encoder, vae, vae_params, unet, unet_params