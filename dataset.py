from datasets import load_dataset
import os
from PIL import Image
import requests
import tqdm

from torchvision import transforms
import torch

from transformers import ByT5Tokenizer, FlaxT5ForConditionalGeneration, set_seed
import jax.numpy as jnp
from diffusers import FlaxAutoencoderKL


def _prefilter(sample):

    image_url = sample["URL"]
    caption = sample["TEXT"]
    watermark_probability = sample["pwatermark"]
    unsafe_probability = sample["punsafe"]
    hash = sample["hash"]

    return (
        caption is not None
        and isinstance(caption, str)
        and image_url is not None
        and isinstance(image_url, str)
        and watermark_probability is not None
        and watermark_probability < 0.6
        and unsafe_probability is not None
        and unsafe_probability < 1.0
        and hash is not None
    )


def _download_image(sample):

    is_ok = False

    image_url = sample["URL"]

    cached_image_image_file_path = os.path.join(
        "/data/image-cache", "%s.jpg" % hex(sample["hash"])
    )

    if os.path.isfile(cached_image_image_file_path):
        pass
    else:

        try:

            # get image data from url
            image_bytes = requests.get(image_url, stream=True, timeout=5).raw

            if image_bytes is not None:

                pil_image = Image.open(image_bytes)

                if pil_image.mode == "RGB":

                    pil_rgb_image = pil_image

                else:

                    # Deal with non RGB images
                    if pil_image.mode == "RGBA":
                        pil_rgba_image = pil_rgb_image
                    else:
                        pil_rgba_image = pil_rgb_image.convert("RGBA")

                    pil_rgb_image = Image.alpha_composite(
                        Image.new("RGBA", pil_image.size, (255, 255, 255)),
                        pil_rgba_image,
                    ).convert("RGB")

                is_ok = True

                pil_rgb_image.save(cached_image_image_file_path)

        except:
            with open(cached_image_image_file_path, mode="a"):
                pass

        # save image to disk but do not catch exception. this has to fail because otherwise the mapper will run forever
        if is_ok:
            pil_rgb_image.save(cached_image_image_file_path)

    return is_ok


def _filter_out_unprocessed(sample):

    cached_image_image_file_path = os.path.join(
        "/data/image-cache", "%s.jpg" % hex(sample["hash"])
    )

    if (
        os.path.isfile(cached_image_image_file_path)
        and os.stat(cached_image_image_file_path).st_size > 0
    ):

        try:

            Image.open(cached_image_image_file_path)

            return True

        except:
            pass

    return False


def _get_pixel_values(image_hash):
    cached_image_image_file_path = os.path.join(
        "/data/image-cache", "%s.jpg" % hex(image_hash)
    )

    # get image data from cache
    pil_rgb_image = Image.open(cached_image_image_file_path)

    return (
        transforms.Compose(
            [
                transforms.Resize(
                    512, interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ]
        )(pil_rgb_image)
        .to(memory_format=torch.contiguous_format)
        .float()
    )


def _compute_intermediate_values(sample):

    sample["pass"] = False

    cached_image_image_file_path = os.path.join(
        "/data/image-cache", "%s.jpg" % hex(sample["hash"])
    )

    if (
        os.path.isfile(cached_image_image_file_path)
        and os.stat(cached_image_image_file_path).st_size > 0
    ):

        try:

            # get image data from cache
            pil_rgb_image = Image.open(cached_image_image_file_path)

            sample["pixel_values"] = (
                transforms.Compose(
                    [
                        transforms.Resize(
                            512, interpolation=transforms.InterpolationMode.LANCZOS
                        ),
                        transforms.CenterCrop(512),
                        transforms.ToTensor(),
                    ]
                )(pil_rgb_image)
                .to(memory_format=torch.contiguous_format)
                .float()
            )

            sample["input_ids"] = ByT5Tokenizer(
                text=sample["TEXT"],
                max_length=1024,
                padding="max_length",
                truncation=True,
                return_tensors="jax",
            ).input_ids

            sample["pass"] = True

        except:
            pass

    return sample


def get_compute_embeddings_lambda():

    set_seed(0)

    language_model = FlaxT5ForConditionalGeneration.from_pretrained(
        "/data/byt5-base",
        dtype=jnp.float32,
    )

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        "/data/stable-diffusion-2-1-vae",
        dtype=jnp.float32,
    )

    tokenizer = ByT5Tokenizer()

    def __compute_embeddings(samples):

        # Caption "tokenizing" to vector or size 4096
        input_ids = tokenizer(
            text=samples["TEXT"],
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="jax",
        ).input_ids

        # compute text embedding
        samples["byt5_text_embedding"] = language_model.encode(
            input_ids,
            train=False,
        )[0]

        # load and transform image
        pixel_values = [_get_pixel_values(image_hash) for image_hash in samples["hash"]]

        # compute image embedding
        samples["vae_latent_dist_mean"] = vae.apply(
            {"params": vae_params},
            torch.stack(pixel_values).numpy(),
            deterministic=True,
            method=vae.encode,
        ).latent_dist.mode()

        return samples

    return lambda samples: __compute_embeddings(samples)


def get_image_noisy_sample_lambda():

    def __image_noisy_sample_lambda(samples):
        latent_dist_means = samples["vae_latent_dist_mean"]
        print(latent_dist_means.shape)
        return samples

    return lambda samples: __image_noisy_sample_lambda(samples["vae_latent_dist_mean"])

def preprocess_dataset():

    # loading the dataset
    dataset = (
        # Dataset.from_parquet(
        #     "/data/laion-high-resolution-filtered-shuffled.parquet",
        #     split="train",
        #     cache_dir="/data/cache",
        # )
        load_dataset(
            # "laion/laion-high-resolution"
            "parquet",
            # data_files={"train": "/data/laion-high-resolution-filtered-shuffled.snappy.parquet"},
            # data_files={"train": "/data/laion-high-resolution-filtered-shuffled-processed-split.zstd.parquet"},
            data_files={
                "train": "/data/laion-high-resolution-filtered-shuffled-processed-split-byt5-vae.zstd.parquet"
            },
            split="train",
            cache_dir="/data/cache",
            streaming=True,
        )
        # .filter(
        #     _prefilter,
        #     #num_proc=96,
        # )
        # .filter(
        #     _download_image,
        #     #num_proc=96,
        # )
        # .map(
        #     _compute_pixel_values,
        #     num_proc=96,
        # )
        # .filter(
        #     _filter_out_unprocessed,
        #     num_proc=8,
        # )
        # .map(
        #     get_compute_embeddings_lambda(),
        #     batched=True,
        #     batch_size=16,
        #     #num_proc=4,
        # )
        # .to_parquet(
        #     "/data/laion-high-resolution-filtered-shuffled-processed-split-byt5-vae.zstd.parquet",
        #     batch_size=96,
        #     compression="ZSTD"
        # )
        # .take(samples)
        # .map(
        #     get_compute_embeddings_lambda(),
        #     batched=True,
        #     batch_size=16,
        #     #num_proc=4,
        # )
    )

    return dataset


def setup_dataset(samples):

    # loading the dataset
    dataset = (
        load_dataset(
            "parquet",
            data_files={
                "train": "/data/laion-high-resolution-filtered-shuffled.snappy.parquet"
            },
            split="train",
            cache_dir="/data/cache",
            streaming=True,
        )
        .map(
            _compute_intermediate_values,
            batched=False,
        )
        .filter(
            lambda sample: sample["pass"],
        )
        .map(
            get_compute_embeddings_lambda(),
            batched=True,
            batch_size=16,
        )
        .remove_columns(["pass"])
        .take(samples)
    )

    return dataset


if __name__ == "__main__":

    max_samples = 10_000

    dataset = preprocess_dataset()

    progress = tqdm(total=max_samples)
    for sample in dataset:
        progress.update(1)
    progress.close()
