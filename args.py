import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_diffusion_model_name_or_path",
        type=str,
        default="flax/stable-diffusion-2-1",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_diffusion_model_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_model_name_or_path",
        type=str,
        default="google/byt5-base",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_model_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--dataset_output_dir",
        type=str,
        default="/data/dataset/output/charred",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data/dataset/cache",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1_000_000,
        help="Total number of training steps per epoch to perform.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="contrebande/charred",
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--log_wandb",
        type=bool,
        default=True,
        choices=[True, False],
        help=(
            "Whether to use WandB to log the metrics or not"
        ),
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args
