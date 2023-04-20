import os
from datasets import load_dataset
import torch
from torchvision import transforms
from PIL import Image
import requests

from flax import jax_utils

from architecture import setup_model


def _get_one_image_pixel_values(image_transforms, sample):
    print(sample)
    url = sample["URL"]
    try:
        image_bytes = requests.get(url, stream=True, timeout=5).raw
    except:
        print("Image url request fails: %s" % url)
        return None
    if image_bytes is None:
        print("Image url respons body is empty: %s" % url)
        return None
    try:
        pil_image = Image.open(image_bytes)
    except:
        print("Image.open fails on image url: %s" % url)
        return None
    try:
        rgb_pil_image = pil_image.convert("RGB")
    except:
        print("Image.convert fails on image url: %s" % url)
        return None
    try:
        pixel_values = image_transforms(rgb_pil_image)
    except:
        print("Image transforms fail on image url: %s" % url)
        return None
    return pixel_values


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


def _dataset_transforms(
    tokenizer,
    tokenizer_max_length,
    text_encoder,
    text_encoder_params,
    vae,
    vae_params,
    image_transforms,
    samples,
):

    samples["pass"] = False

    # TODO: if checksum fails, skip this entry and filter out later
    # checksum = hashlib.md5(image_bytes).hexdigest() == example["hash"]

    # get image data
    stacked_pixel_values = (
        torch.stack(
            [
                _get_one_image_pixel_values(image_transforms, sample)
                for sample in samples
            ]
        )
        .to(memory_format=torch.contiguous_format)
        .float()
    )

    # compute image embeddings
    samples["vae_outputs"] = vae.apply(
        {"params": vae_params},
        stacked_pixel_values,
        deterministic=True,
        method=vae.encode,
    )

    input_ids = tokenizer(
        text=samples,
        max_length=tokenizer_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).input_ids

    stacked_input_ids = torch.stack(input_ids)

    samples["encoder_hidden_states"] = text_encoder(
        stacked_input_ids,
        params=text_encoder_params,
        train=False,
    )[0]

    samples["pass"] = True

    return samples


def dataset_transforms(
    tokenizer,
    tokenizer_max_length,
    text_encoder,
    text_encoder_params,
    vae,
    vae_params,
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
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    return lambda samples: _dataset_transforms(
        tokenizer,
        tokenizer_max_length,
        text_encoder,
        text_encoder_params,
        vae,
        vae_params,
        image_transforms,
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

    # TODO: make sure we use the datatsets library with JAX : https://huggingface.co/docs/datasets/use_with_jax
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
            function=dataset_transforms(
                tokenizer,
                tokenizer_max_length,
                text_encoder,
                text_encoder_params,
                vae,
                vae_params,
                resolution,
            ),
            batched=True,
            batch_size=16,
        )
        .filter(
            lambda example: example["pass"]
        )  # filter out samples that didn't pass the tests in the transform function
        .remove_columns(["pass"])
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

    dataset = setup_dataset(
        10,
        "/data/dataset/cache",
        1024,
        tokenizer,
        1024,
        text_encoder,
        text_encoder_params,
        vae,
        vae_params,
    )

    for sample in dataset:
        print(sample["URL"])
