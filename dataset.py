import os

from datasets import load_dataset, load_from_disk
from dataset_preprocessing import dataset_transforms
from dataset_filtering import dataset_filter

DATASET_OUTPUT_DIR = "/data/dataset/charred"

def setup_dataset(cache_dir, resolution, tokenizer):

    # TODO: make sure we use the datatsets library with JAX : https://huggingface.co/docs/datasets/use_with_jax
    # loading the dataset

    if os.path.isdir(DATASET_OUTPUT_DIR):
        load_from_disk = load_from_disk(DATASET_OUTPUT_DIR)
    else:
        dataset = load_dataset(
            "laion/laion-high-resolution",
            cache_dir=os.path.join(cache_dir, "laion-high-resolution"),
            split='train',
            streaming=True
        ).filter(
            dataset_filter()
        ).map(
            transforms=dataset_transforms(tokenizer, resolution),
            remove_columns=[],
            batched=False
        )
        dataset.save_to_disk(DATASET_OUTPUT_DIR)

    return dataset
