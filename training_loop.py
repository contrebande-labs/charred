import time

# misc. utils
from tqdm.auto import tqdm
import wandb

# jax/flax
import jax
import jax.numpy as jnp
from flax.training.common_utils import shard
from flax import jax_utils

from batch import setup_dataloader
from dataset import setup_dataset
from repository import save_to_repository
from training_step import train_step

def training_loop(tokenizer, text_encoder, text_encoder_params, vae, vae_params, unet, state,
    cache_dir,
    resolution,
    rng, max_train_steps, num_train_epochs, train_batch_size,
    output_dir, dataset_output_dir, push_to_hub, repo_id, log_wandb):
  
  # rng setup
  train_rngs = jax.random.split(rng, jax.local_device_count())
  
  # dataset setup
  train_dataset = setup_dataset(max_train_steps, cache_dir, resolution, tokenizer)

  # batch setup
  total_train_batch_size = train_batch_size * jax.local_device_count()
  train_dataloader = setup_dataloader(train_dataset, total_train_batch_size)

  # Precompiled training step setup
  p_train_step = train_step(text_encoder, text_encoder_params, vae, vae_params, unet)
  
  # Epoch setup
  epochs = tqdm(range(num_train_epochs), desc="Epoch ... ", position=0)
  dataset_needs_saving = True
  step_i = 0
  t0 = time.monotonic()
  for epoch in epochs:

      unreplicated_train_metric = None
 
      train_step_progress_bar = tqdm(total=max_train_steps, desc="Training...", position=1, leave=False)

      for batch in train_dataloader:

          batch = shard(batch)

          state, train_rngs, train_metric = p_train_step(state, batch, train_rngs)

          train_step_progress_bar.update(1)

          unreplicated_train_metric = jax_utils.unreplicate(train_metric)

          step_i += 1

          if log_wandb:
            walltime = time.monotonic() - t0
            wandb.log(
                {
                    "walltime": walltime,
                    "train/loss": train_metric["loss"],
                    "train/step": step_i,
                    "train/epoch": epoch,
                    "train/secs_per_epoch": walltime / (epoch + 1),
                    "train/steps_per_sec": step_i / walltime,
                    **{f"train/{k}": v for k, v in unreplicated_train_metric.items()},
                }
            )

      if dataset_needs_saving:
          dataset_needs_saving = False
          train_dataset.save_to_disk(dataset_output_dir)

      # Create the pipeline using using the trained modules and save it after every epoch
      save_to_repository(output_dir, push_to_hub, 
        tokenizer, text_encoder, text_encoder_params, vae, vae_params, unet, repo_id, state)

      train_step_progress_bar.close()

      epochs.write(f"Epoch... ({epoch}/{num_train_epochs} | Loss: {unreplicated_train_metric['loss']})")
