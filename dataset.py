import os

from datasets import load_dataset

def setup_dataset(args):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    image_column = args.image_column
    caption_column = args.caption_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{image_column}' needs to be one of: {', '.join(column_names)}"
        )
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
        )
    
    return caption_column, image_column, dataset