from datasets import Dataset, load_dataset
import os
from PIL import Image
import requests
from tqdm import tqdm

import torch
from torchvision import transforms

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

    if os.path.isfile(cached_image_image_file_path): pass
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
            with open(cached_image_image_file_path, mode='a'): pass
 
        # save image to disk but do not catch exception. this has to fail because otherwise the mapper will run forever
        if is_ok: pil_rgb_image.save(cached_image_image_file_path)

    return is_ok

def _compute_intermediate_values(sample):

    sample["pass"] = False

    cached_image_image_file_path = os.path.join("/data/image-cache", "%s.jpg" % hex(sample["hash"]))

    if os.path.isfile(cached_image_image_file_path) and os.stat(cached_image_image_file_path).st_size > 0:

        try:

            # get image data from cache
            pil_rgb_image = Image.open(cached_image_image_file_path)

            sample["pixel_values"] = transforms.Compose(
                [
                    transforms.Resize(1024, interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.CenterCrop(1024),
                    transforms.ToTensor(),
                ]
            )(pil_rgb_image)

            # Model has 3 special tokens which take up the input ids 0,1,2 of ByT5.
            # => Need to shift utf-8 character encodings by 3 before passing ids to model.
            sample["input_ids"] = torch.tensor([list(sample["TEXT"].encode("utf-8"))]) + 3

            sample["pass"] = True

        except: pass

    return sample

def _compute_embeddings(sample):

    # # compute image embeddings
    # samples["vae_image_embedding"] = vae.apply(
    #     {"params": vae_params},
    #     samples["pixel_values"],
    #     deterministic=True,
    #     method=vae.encode,
    # )

    # samples["byt5_text_embedding"] = text_encoder(
    #     samples["input_ids"],
    #     params=text_encoder_params,
    #     train=False,
    # )[0]
 
    return sample



def preprocess_dataset():

    # loading the dataset
    dataset = (
        Dataset.from_parquet(
            "/data/laion-high-resolution-filtered-shuffled.parquet",
            split="train",
            cache_dir="/data/cache",
        )
        .filter(
            _prefilter,
            num_proc=96,
        )
        .filter(
            _download_image,
            num_proc=96,
        )
        .map(
            _compute_intermediate_values,
            num_proc=96,
        )
        .filter(
            lambda sample: sample["pass"],
            num_proc=96,
        )
        .map(
            _compute_embeddings,
            num_proc=96,
        )
        .remove_columns(["pass", "pixel_values", "input_ids"])
        .to_parquet(
            "/data/laion-high-resolution-filtered-shuffled-processed.parquet",
            batch_size=96
        )
    )

    return dataset


def setup_dataset(samples):

    # loading the dataset
    dataset = (
        load_dataset(
            "parquet",
            data_files={"train": "/data/laion-high-resolution-filtered-shuffled.parquet"},
            split="train",
            cache_dir="/data/cache",
            streaming=True,
        )
        .map(
            _compute_intermediate_values,
        )
        .filter(
            lambda sample: sample["pass"],
        )
        .take(samples)

    )

    return dataset


if __name__ == "__main__":

    max_samples = 1_000_000

    dataset = setup_dataset(max_samples)

    # TODO: do batches with DataLoader here to use all the CPUs
    # TODO: use TQDM
    progress = tqdm(total=max_samples)
    for sample in dataset: progress.update(1)
    progress.close()
