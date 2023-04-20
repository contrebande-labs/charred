import os
from datasets import load_dataset
import torch
from torchvision import transforms
from PIL import Image
import requests

from flax import jax_utils

from architecture import setup_model


def _prefilter_dataset(example):

    caption = example["TEXT"]
    image_url = example["URL"]
    watermark_probability = example["pwatermark"]
    unsafe_probability = example["punsafe"]

    return (
        caption is not None
        and isinstance(caption, str)
        and image_url is not None
        and isinstance(image_url, str)
        and watermark_probability is not None
        and watermark_probability < 0.6
        and unsafe_probability is not None
        and unsafe_probability < 0.95
    )


def _download_image(
    image_transforms,
    sample,
):

    sample["pass"] = False

    # TODO: if checksum fails, skip this entry and filter out later
    # checksum = hashlib.md5(image_bytes).hexdigest() == example["hash"]

    # get image data
    try:
        image_bytes = requests.get(sample["URL"], stream=True, timeout=5).raw
        if image_bytes is None:
            return sample
        pil_image = Image.open(image_bytes)
        pil_rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
        pil_rgb_image.paste(pil_image, mask=pil_image.split()[3])
        sample["pixel_values"] = image_transforms(pil_rgb_image)
    except:
        return sample

    sample["pass"] = True

    return sample


def download_image(
    resolution,
):

    # TODO: replace with https://jax.readthedocs.io/en/latest/jax.image.html
    image_transforms = transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.RandomCrop(resolution),
            transforms.ToTensor(),
        ]
    )

    return lambda sample: _download_image(
        image_transforms,
        sample,
    )


def _dataset_transforms(
    tokenizer,
    tokenizer_max_length,
    text_encoder,
    text_encoder_params,
    vae,
    vae_params,
    samples,
):
    # TODO: if checksum fails, skip this entry and filter out later
    # checksum = hashlib.md5(image_bytes).hexdigest() == example["hash"]

    # get image data
    stacked_pixel_values = (
        torch.stack(samples["pixel_values"])
        .to(memory_format=torch.contiguous_format)
        .float()
    ).numpy()

    # compute image embeddings
    samples["vae_outputs"] = vae.apply(
        {"params": vae_params},
        stacked_pixel_values,
        deterministic=True,
        method=vae.encode,
    )

    input_ids = tokenizer(
        text=samples["TEXT"],
        max_length=tokenizer_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).input_ids

    stacked_input_ids = torch.stack(input_ids).numpy()

    samples["encoder_hidden_states"] = text_encoder(
        stacked_input_ids,
        params=text_encoder_params,
        train=False,
    )[0]

    return samples


def dataset_transforms(
    tokenizer,
    tokenizer_max_length,
    text_encoder,
    text_encoder_params,
    vae,
    vae_params,
):

    return lambda samples: _dataset_transforms(
        tokenizer,
        tokenizer_max_length,
        text_encoder,
        text_encoder_params,
        vae,
        vae_params,
        samples,
    )


def setup_dataset(
    max_samples,
    cache_dir,
    resolution,
    tokenizer,
    tokenizer_max_length,
    text_encoder,
    text_encoder_params,
    vae,
    vae_params,
):

    # loading the dataset
    dataset = (
        load_dataset(
            path="laion/laion-high-resolution",
            cache_dir=os.path.join(cache_dir, "laion-high-resolution"),
            split="train",
            streaming=True,
        )
        .filter(_prefilter_dataset)
        .shuffle(seed=27, buffer_size=10_000)
        .map(
            function=download_image(
                resolution,
            ),
            batched=False,
        )
        .filter(
            lambda example: example["pass"]
        )  # filter out samples that didn't pass the tests in the transform function
        .remove_columns(["pass"])
        .map(
            function=dataset_transforms(
                tokenizer,
                tokenizer_max_length,
                text_encoder,
                text_encoder_params,
                vae,
                vae_params,
            ),
            batched=True,
            batch_size=16,
        )
        .remove_columns(["pixel_values"])
        .take(n=max_samples)
    )

    return dataset


if __name__ == "__main__":

    # Pretrained freezed model setup
    tokenizer, text_encoder, vae, vae_params, _ = setup_model(
        7,
        None,
        "google/byt5-base",
        "flax/stable-diffusion-2-1",
    )

    text_encoder_params = jax_utils.replicate(text_encoder.params)

    max_samples = 10

    dataset = setup_dataset(
        max_samples,
        "./dataset-cache",
        1024,
        tokenizer,
        1024,
        text_encoder,
        text_encoder_params,
        vae,
        vae_params,
    )

    # TODO: do batches with DataLoader here to use all the CPUs
    # TODO: use TQDM
    for sample in dataset:
        print(sample["URL"])
