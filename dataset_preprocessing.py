from torchvision import transforms
from PIL import Image
import requests


def _dataset_transforms(tokenizer, image_transforms, example):
    # TODO: download the image from the url or from the img_path? ValueError: --image_column' value 'image' needs to be one of: image_path, caption, NSFW, similarity, LICENSE, url, key, status, error_message, width, height, original_width, original_height, exif, md5
    image = Image.open(requests.get(example["url"], stream = True).raw).convert("RGB")
    example["pixel_values"] = image_transforms(image)
    example["input_ids"] = tokenizer(example["text"], max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True).input_ids
    # TODO: apply and cache image embbedings here instead of in the training loop
    # TODO: apply and cache text embbedings here instead of in the training loop
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
