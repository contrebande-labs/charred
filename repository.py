import os

from threading import Thread

# hugging face
from huggingface_hub import upload_folder


def create_output_dir(output_dir):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)


def save_to_local_directory(
    output_dir,
    unet,
    unet_params,
):
    print("saving trained weights...")
    unet.save_pretrained(
        save_directory=output_dir,
        params=unet_params,
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
        "/data/output.bak/000920",
        "character-aware-diffusion/charred",
        "Latest training epoch version as of Apr 28 11:03PM UST.",
    )
