import os
from datasets import load_dataset, interleave_datasets

from preprocessing import dataset_transform

def setup_dataset(cache_dir, image_column, caption_column, resolution, tokenizer):

    # loading the datasets
    # TODO: make sure we use the datatsets library with JAX : https://huggingface.co/docs/datasets/use_with_jax
    # TODO: add language column for en and nolang
    # TODO: find out if we can use the image embeddings instead of recomputing them
    # TODO: pre-compute ByT5 embeddings too, if possible : https://huggingface.co/docs/datasets/stream#map
    # TODO: remove duplicates with image embeddings https://arxiv.org/abs/2303.12733 https://github.com/LAION-AI/laion-dedup https://github.com/LAION-AI/image-deduplication-testset
    # TODO: drop images that are smaller than 1024x1024 https://huggingface.co/docs/datasets/stream#filter https://huggingface.co/docs/datasets/process#select-and-filter
    # TODO: drop images with watermarks (joined) https://laion.ai/blog/laion-5b/
    # TODO: drop images columns https://huggingface.co/docs/datasets/stream#rename-remove-and-cast
    laion2b_en = load_dataset("laion/laion2b-en-vit-l-14-embeddings", cache_dir=os.path.join(cache_dir, "laion2B-en"), streaming=True).shuffle(seed=0, buffer_size=10_000)
    laion2b_multi = load_dataset("laion/laion2b-multi-vit-l-14-embeddings", cache_dir=os.path.join(cache_dir, "laion2B-multi"), streaming=True).shuffle(seed=1, buffer_size=10_000)
    laion2b_nolang = load_dataset("laion/laion1b-nolang-vit-l-14-embeddings", cache_dir=os.path.join(cache_dir, "laion1B-nolang"), streaming=True).shuffle(seed=2, buffer_size=10_000)

    # concatenate the datasets
    # TODO: add probabilities https://huggingface.co/docs/datasets/process#interleave https://huggingface.co/docs/datasets/stream#interleave
    dataset = interleave_datasets([laion2b_en, laion2b_multi, laion2b_nolang])["train"]

    # setting up the transform
    dataset.set_transform(dataset_transform(image_column, caption_column, tokenizer, resolution))

    # Verify the column names for input/target.
    column_names = dataset.column_names
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{image_column}' needs to be one of: {', '.join(column_names)}"
        )
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{caption_column}' needs to be one of: {', '.join(column_names)}"
        )
    
    # TODO: save to disk
    #dataset.save_to_disk("/data/dataset/charred")

    return dataset
