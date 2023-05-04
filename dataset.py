from datasets import load_dataset
import os
from PIL import Image
import requests

from torchvision import transforms

from transformers import ByT5Tokenizer


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


def get_compute_intermediate_values_lambda():
    tokenizer = ByT5Tokenizer()

    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )

    def __get_pixel_values(image_hash):
        # compute file name
        cached_image_image_file_path = os.path.join(
            "/data/image-cache", "%s.jpg" % hex(image_hash)
        )

        # get image data from cache
        pil_rgb_image = Image.open(cached_image_image_file_path)

        transformed_image = image_transforms(pil_rgb_image)

        return transformed_image

    def __compute_intermediate_values_lambda(samples):
        samples["input_ids"] = tokenizer(
            text=samples["TEXT"],
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        samples["pixel_values"] = [
            __get_pixel_values(image_hash) for image_hash in samples["hash"]
        ]

        return samples

    return __compute_intermediate_values_lambda


def setup_dataset(n):
    # loading the dataset
    dataset = (
        load_dataset(
            "parquet",
            data_files={
                # "train": "/data/laion-high-resolution-filtered-shuffled.snappy.parquet",
                # "train": "/data/laion-high-resolution-filtered-shuffled-processed-split.zstd.parquet",
                # "train": "/data/laion-high-resolution-filtered-shuffled-processed-split-byt5-vae.zstd.parquet",
                # "train": "/data/laion-high-resolution-filtered-shuffled-validated-10k.zstd.parquet",
                "train": "/data/laion-high-resolution-1M.zstd.parquet",
            },
            split="train[:%d]" % n,
            cache_dir="/data/cache",
            num_proc=4,
        )
        .with_format("torch")
        .map(
            get_compute_intermediate_values_lambda(),
            batched=True,
            batch_size=16,
            num_proc=4,
        )
        .select_columns(["input_ids", "pixel_values"])
    )

    return dataset


def prepare_1m_dataset():
    # Gives 1267072 samples to be exact
    (
        load_dataset(
            "laion/laion-high-resolution",
            split="train",
            cache_dir="/data/cache",
        )
        .with_format("torch")
        .select_columns(["TEXT", "hash"])
        .filter(
            function=_filter_out_unprocessed,
            num_proc=96,
        )
        .to_parquet(
            "/data/laion-high-resolution-1M.zstd.parquet",
            batch_size=128,
            compression="ZSTD",
        )
    )


if __name__ == "__main__":
    prepare_1m_dataset()
    # dataset = setup_dataset(64)

    # dataloader = setup_dataloader(dataset, 16)
    # for batch in dataloader:
    #     print(batch["pixel_values"].shape)
