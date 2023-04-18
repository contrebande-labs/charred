# TODO: change that for JAX equivalents
import random

import numpy as np
from torchvision import transforms


# Preprocessing the datasets.
# We need to tokenize input captions and transform the images.
def _tokenize_captions(caption_column, tokenizer, examples, is_train):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="do_not_pad",
        truncation=True,
    )
    input_ids = inputs.input_ids
    return input_ids


def _preprocess_train(
    image_column, caption_column, tokenizer, train_transforms, examples, is_train
):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = _tokenize_captions(
        caption_column, tokenizer, examples, is_train
    )
    return examples


def preprocess_train(image_column, caption_column, tokenizer, train_transforms):
    return lambda examples, is_train=True: _preprocess_train(
        image_column, caption_column, tokenizer, train_transforms, examples, is_train
    )


def setup_train_transforms(
    resolution,
):
    return transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.RandomCrop(resolution),
            transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
