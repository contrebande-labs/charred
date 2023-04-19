import os

from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import requests

def _prefilter_dataset(example):

    caption = example["TEXT"]
    image_url = example["URL"]
    watermark_probability = example["pwatermark"]

    return (
        caption is not None
        and isinstance(caption, str)
        and image_url is not None
        and isinstance(image_url, str)
        and watermark_probability is not None
        and watermark_probability < 0.6
    )


def _dataset_transforms(tokenizer, tokenizer_max_length, image_transforms, example):

    caption = example["TEXT"]
    image_url = example["URL"]

    # request image data bytes from http url
    try:
        image_bytes = requests.get(image_url, stream=True).raw
    except:
        return example

    # TODO: if checksum fails, skip this entry and filter out later
    # checksum = hashlib.md5(image_bytes).hexdigest() == example["hash"]

    # append image data
    # TODO: apply and cache image embbedings here instead of in the training loop (and don't keep the pixel data)
    example["pixel_values"] = image_transforms(
        Image.open(image_bytes).convert("RGB")
    )

    # append tokenized text
    # TODO: apply and cache text embbedings here instead of in the training loop (and don't keep the tokenized text)
    example["input_ids"] = tokenizer(
        text=caption,
        max_length=tokenizer_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"]

    example["pass"] = True

    return example


def dataset_transforms(tokenizer, tokenizer_max_length, resolution):

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

    return lambda example: _dataset_transforms(tokenizer, tokenizer_max_length, image_transforms, example)


def setup_dataset(max_train_steps, cache_dir, resolution, tokenizer, tokenizer_max_length):

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
        .take(n=max_train_steps)
        .map(
            function=dataset_transforms(tokenizer, tokenizer_max_length, resolution)
        )
        .filter(lambda example: example["pass"])
        .remove_columns(["pass"])
    )

    return dataset
