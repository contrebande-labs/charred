import os

from threading import Thread

# hugging face
from huggingface_hub import create_repo, upload_folder

import jax


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


def save_to_local_directory(
    output_dir,
    unet,
    unet_params,
):
    print("saving trained weights...")
    unet.save_pretrained(
        save_directory=output_dir,
        params=get_params_to_save(unet_params),
    )
    print("trained weights saved...")


def save_to_repository(
    output_dir,
    unet,
    unet_params,
    repo_id,
):
    print("saving trained weights...")
    unet.save_pretrained(
        save_directory=output_dir,
        params=unet_params,
    )
    print("trained weights saved...")

    Thread(
        target=lambda: upload_to_repository(
            repo_id,
            output_dir,
            "End of training epoch.",
        )
    ).start()


def upload_to_repository(
    output_dir,
    repo_id,
    commit_message,
):
    upload_folder(
        repo_id=repo_id,
        folder_path=output_dir,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    upload_to_repository(
        "/data/output/000270",
        "character-aware-diffusion/charred",
        "Latest training epoch version as of Apr 26 8PM EST.",
    )
