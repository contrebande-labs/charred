
import os

from pathlib import Path

from huggingface_hub import create_repo

import jax

# hugging face
from huggingface_hub import upload_folder
from diffusers import (
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
)

from preprocessing import ByT5ImageProcessor

def create_repository(args):

  if args.output_dir is not None:
      os.makedirs(args.output_dir, exist_ok=True)

  if args.push_to_hub:
      repo_id = create_repo(
          repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
      ).repo_id

  return repo_id

def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))

def save_to_repository(args, tokenizer, text_encoder, text_encoder_params, vae, vae_params, unet, repo_id, state):

    scheduler = FlaxPNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
    )

    pipeline = FlaxStableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        feature_extractor=ByT5ImageProcessor(),
    )

    pipeline.save_pretrained(
        args.output_dir,
        params={
            "text_encoder": get_params_to_save(text_encoder_params),
            "vae": get_params_to_save(vae_params),
            "unet": get_params_to_save(state.params),
        },
    )

    if args.push_to_hub:
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )