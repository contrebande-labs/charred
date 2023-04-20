import time

# misc. utils
from tqdm.auto import tqdm
import wandb
from datasets import load_dataset

# jax/flax
import jax
from flax.training.common_utils import shard
from flax import jax_utils

from batch import setup_dataloader
from dataset import setup_dataset
from repository import save_to_repository
from training_step import train_step


def training_loop(
    tokenizer,
    text_encoder,
    text_encoder_params,
    vae,
    vae_params,
    unet,
    state,
    rng,
    max_train_steps,
    num_train_epochs,
    train_batch_size,
    output_dir,
    repo_id,
    log_wandb,
):

    # rng setup
    train_rngs = jax.random.split(rng, jax.local_device_count())

    # dataset setup
    train_dataset = setup_dataset(max_train_steps, tokenizer)

    # batch setup
    total_train_batch_size = train_batch_size * jax.local_device_count()
    train_dataloader = setup_dataloader(train_dataset, total_train_batch_size)

    # Precompiled training step setup
    p_train_step = train_step(text_encoder, text_encoder_params, vae, vae_params, unet)

    # Epoch setup
    epochs = tqdm(range(num_train_epochs), desc="Epoch ... ", position=0)
    step_i = 0
    t0 = time.monotonic()
    for epoch in epochs:

        unreplicated_train_metric = None

        train_step_progress_bar = tqdm(
            total=max_train_steps, desc="Training...", position=1, leave=False
        )

        for batch in train_dataloader:

            batch = shard(batch)

            state, train_rngs, train_metric = p_train_step(state, batch, train_rngs)

            train_step_progress_bar.update(1)

            unreplicated_train_metric = jax_utils.unreplicate(train_metric)

            step_i += 1

            if log_wandb:
                walltime = time.monotonic() - t0
                wandb.log(
                    data={
                        "walltime": walltime,
                        "train/step": step_i,
                        "train/epoch": epoch,
                        "train/secs_per_epoch": walltime / (epoch + 1),
                        "train/steps_per_sec": step_i / walltime,
                        **{
                            f"train/{k}": v
                            for k, v in unreplicated_train_metric.items()
                        },
                    },
                )

        wandb.log(data={}, commit=True)

        # Create the pipeline using using the trained modules and save it after every epoch
        if repo_id is not None:
            save_to_repository(
                output_dir,
                tokenizer,
                text_encoder,
                text_encoder_params,
                vae,
                vae_params,
                unet,
                state,
                repo_id,
            )

        train_step_progress_bar.close()
 
        epochs.update(1)

    epochs.close()
