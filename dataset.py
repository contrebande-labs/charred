import os

from datasets import load_dataset, load_from_disk
from torchvision import transforms
from PIL import Image
import requests

DATASET_OUTPUT_DIR = "/data/dataset/charred"
SAVE_DATASET_TO_DISK = False

def _dataset_transforms(tokenizer, image_transforms, example):

    caption = example['caption']
    image_url = example['url']
    watermark_probability = example['pwatermark']
  
    filter_pass = caption is not None and isinstance(caption, str) and image_url is not None and isinstance(image_url, str) and watermark_probability < 0.6

    example["pass"] = filter_pass

    if filter_pass:

        image_bytes = requests.get(image_url, stream = True).raw

        # TODO: if checksum fails, skip this entry and filter out later 
        #checksum = hashlib.md5(image_bytes).hexdigest() == example["hash"]

        # append image data
        # TODO: apply and cache image embbedings here instead of in the training loop (and don't keep the pixel data)
        example["pixel_values"] = image_transforms(Image.open(image_bytes).convert("RGB"))

        # append tokenized text
        # TODO: apply and cache text embbedings here instead of in the training loop (and don't keep the tokenized text)
        example["input_ids"] = tokenizer(caption, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True).input_ids
 
    return example

def dataset_transforms(tokenizer, resolution):

    # TODO: replace with https://jax.readthedocs.io/en/latest/jax.image.html
    image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
 
    return lambda example: _dataset_transforms(tokenizer, image_transforms, example)

def setup_dataset(max_train_steps, cache_dir, resolution, tokenizer):

    # TODO: make sure we use the datatsets library with JAX : https://huggingface.co/docs/datasets/use_with_jax
    # loading the dataset

    if os.path.isdir(DATASET_OUTPUT_DIR) and SAVE_DATASET_TO_DISK:
        dataset = load_from_disk(DATASET_OUTPUT_DIR)
    else:
        dataset = load_dataset(
            path="laion/laion-high-resolution",
            cache_dir=os.path.join(cache_dir, "laion-high-resolution"),
            split='train',
            streaming=True
        ).shuffle(
            seed=27,
            buffer_size=10_000
        ).take(
            n=max_train_steps
        ).map(
            transforms=dataset_transforms(tokenizer, resolution),
            remove_columns=[],
            batched=False #TODO: maybe batch this?
        ).filter(
            filter=lambda example: example["pass"]
        )
        if SAVE_DATASET_TO_DISK:
            dataset.save_to_disk(DATASET_OUTPUT_DIR)

    return dataset
