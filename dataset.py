from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import requests

from transformers import ByT5Tokenizer

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
            transforms.CenterCrop(resolution), #TODO: do we need this?
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
    samples,
):
    # TODO: if checksum fails, skip this entry and filter out later
    # checksum = hashlib.md5(image_bytes).hexdigest() == example["hash"]

    # get image data
    # stacked_pixel_values = (
    #     torch.stack(samples["pixel_values"])
    #     .to(memory_format=torch.contiguous_format)
    #     .float()
    # ).numpy()

    # # compute image embeddings
    # samples["vae_image_embedding"] = vae.apply(
    #     {"params": vae_params},
    #     stacked_pixel_values,
    #     deterministic=True,
    #     method=vae.encode,
    # )

    # stacked_input_ids = torch.stack(
    #     tokenizer(
    #         text=samples["TEXT"],
    #         max_length=tokenizer_max_length,
    #         truncation=True,
    #         padding="max_length",
    #         return_tensors="pt",
    #     ).input_ids
    # ).numpy()

    # samples["byt5_text_embedding"] = text_encoder(
    #     stacked_input_ids,
    #     params=text_encoder_params,
    #     train=False,
    # )[0]

    samples["input_ids"] = tokenizer(
        text=samples["TEXT"],
        max_length=tokenizer_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).input_ids

    return samples


def dataset_transforms(
    tokenizer,
    tokenizer_max_length,
):

    return lambda samples: _dataset_transforms(
        tokenizer,
        tokenizer_max_length,
        samples,
    )


def setup_dataset(
    max_samples,
    tokenizer,
):

    # loading the dataset
    dataset = (
        load_dataset(
            path="laion/laion-high-resolution",
            split="train",
            streaming=True,
        )
        .filter(_prefilter_dataset)
        .shuffle(seed=27, buffer_size=10_000)
        .map(
            function=download_image(
                1024,
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
                1024,
            ),
            batched=True,
            batch_size=16,
        )
        .take(n=max_samples)
    )

    return dataset


if __name__ == "__main__":

    tokenizer = ByT5Tokenizer()

    max_samples = 10

    dataset = setup_dataset(
        max_samples,
        tokenizer,
    )

    # TODO: do batches with DataLoader here to use all the CPUs
    # TODO: use TQDM
    for sample in dataset:
        print(len(sample["pixel_values"][0]))
