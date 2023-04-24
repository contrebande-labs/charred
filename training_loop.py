import time

# misc. utils
from tqdm.auto import tqdm
import wandb

# jax/flax
import jax
from flax.training.common_utils import shard
from flax import jax_utils

from batch import setup_dataloader
from dataset import setup_dataset
from repository import save_to_repository
from training_step import get_training_step_lambda


def training_loop(
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
    train_dataset = setup_dataset(max_train_steps)

    # batch setup
    total_train_batch_size = train_batch_size * jax.local_device_count()
    train_dataloader = setup_dataloader(train_dataset, total_train_batch_size)

    # Create parallel version of the train step
    jax_pmap_train_step = jax.pmap(
        get_training_step_lambda(text_encoder, vae, unet), "batch", donate_argnums=(0,)
    )

    # Epoch setup
    epochs = tqdm(range(num_train_epochs), desc="Epoch... ", position=0)
    t0 = time.monotonic()
    global_training_steps = 0
    for epoch in epochs:

        unreplicated_train_metric = None

        steps = tqdm(
            total=max_train_steps, desc="Training steps...", position=1, leave=False
        )

        epoch_steps = 0

        for batch in train_dataloader:

            batch = shard(batch)

            state, train_rngs, train_metric = jax_pmap_train_step(
                state, text_encoder_params, vae_params, batch, train_rngs
            )

            unreplicated_train_metric = jax_utils.unreplicate(train_metric)

            epoch_steps += 1
            global_training_steps += 1
            steps.update(1)

            if log_wandb:
                walltime = time.monotonic() - t0
                wandb.log(
                    data={
                        "walltime": walltime,
                        "train/step": epoch_steps,
                        "train/epoch": epoch,
                        "train/secs_per_epoch": walltime / (epoch + 1),
                        "train/steps_per_sec": global_training_steps / walltime,
                        **{
                            f"train/{k}": v
                            for k, v in unreplicated_train_metric.items()
                        },
                    },
                    commit=False,
                )

        if log_wandb:
            wandb.log(data={}, commit=True)

        # Create the pipeline using using the trained modules and save it after every epoch
        if repo_id is not None:
            save_to_repository(
                output_dir,
                unet,
                state.params,
                repo_id,
            )

        epochs.update(1)

    epochs.close()
