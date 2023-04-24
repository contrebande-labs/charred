import os

from huggingface_hub import create_repo

from transformers import ByT5Tokenizer

import jax

# hugging face
from huggingface_hub import upload_folder
from diffusers import (
    FlaxStableDiffusionPipeline,
)
from transformers import CLIPImageProcessor


def create_repository(output_dir, hub_model_id):

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    repo_id = create_repo(
        repo_id=hub_model_id,
        exist_ok=True,
    ).repo_id

    return repo_id


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def save_to_repository(
    output_dir,
    unet,
    unet_params,
    repo_id,
):

    unet.save_pretrained(
        save_directory=output_dir,
        params=unet_params,
    )

    upload_folder(
        repo_id=repo_id,
        folder_path=output_dir,
        commit_message="End of training epoch.",
        ignore_patterns=["step_*", "epoch_*"],
    )
