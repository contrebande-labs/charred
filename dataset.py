import os
from datasets import load_dataset, concatenate_datasets

from preprocessing import preprocess_train, setup_train_transforms

def setup_dataset(cache_dir, image_column, caption_column, resolution, center_crop, random_flip, tokenizer):

    # loading the datasets    
    laion2b_en = load_dataset("laion/laion2B-en", cache_dir=os.path.join(cache_dir, "laion2B-en"))
    laion2b_multi = load_dataset("laion/laion2B-multi", cache_dir=os.path.join(cache_dir, "laion2B-multi"))
    laion2b_nolang = load_dataset("laion/laion2B-nolang", cache_dir=os.path.join(cache_dir, "laion2B-nolang"))

    # concatenating the dataset and setting up the transforms
    train_transforms = setup_train_transforms(resolution)
    preprocess_train_lambda = preprocess_train(image_column, caption_column, tokenizer, train_transforms)
    dataset = concatenate_datasets([laion2b_en, laion2b_multi, laion2b_nolang])["train"].with_transform(preprocess_train_lambda)

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

    return dataset
