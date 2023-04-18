# python
import math

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
from training_step import train_step

def training_loop(tokenizer, text_encoder, text_encoder_params, vae, vae_params, unet, state,
    cache_dir,
    resolution,
    seed, max_train_steps, num_train_epochs, train_batch_size,
    output_dir, push_to_hub, repo_id, log_wandb):
  
  # setup WandB for logging & tracking
  if log_wandb:
      wandb.init(project="charred")
      wandb_args = {
        "max_train_steps": max_train_steps, 
        "num_train_epochs": num_train_epochs, 
        "train_batch_size": train_batch_size
      }
      wandb.config.update(wandb_args)
  
  # dataset setup
  train_dataset = setup_dataset(cache_dir, resolution, tokenizer)

  # Initialize our training
  rng = jax.random.PRNGKey(seed)
  train_rng = jax.random.split(rng, jax.local_device_count())

  # batch setup
  total_train_batch_size = train_batch_size * jax.local_device_count()
  train_dataloader = setup_dataloader(tokenizer, train_dataset, total_train_batch_size)
  num_update_steps_per_epoch = math.ceil(len(train_dataloader))      
  
  # Precompiled training step setup
  p_train_step = train_step(text_encoder, text_encoder_params, vae, vae_params, unet)

  # Epoch setup, scheduler and math around the number of training steps.
  max_train_steps = num_train_epochs * num_update_steps_per_epoch
  steps_per_epoch = len(train_dataset) // total_train_batch_size
  distributed_num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
  epochs = tqdm(range(distributed_num_train_epochs), desc="Epoch ... ", position=0)
  global_step = 0
  for epoch in epochs:

      train_metrics = []
 
      train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)

      for batch in train_dataloader:

          batch = shard(batch)

          state, train_rng, train_metric = p_train_step(state, batch, train_rng)

          train_metrics.append(train_metric)

          train_step_progress_bar.update(1)
          if jax.process_index() == 0:
            print(metrics)
            if log_wandb:
                wandb.log(
                    {
                        "train_loss": train_metric["loss"],
                    }
                )
          global_step += 1
          if global_step >= max_train_steps:
              break

      train_metric = jax_utils.unreplicate(train_metric)    # NOTE: @imflash217 thinks this would be an error because we should rather be doing this for every batch rather than just for every epoch

      # Create the pipeline using using the trained modules and save it after every epoch
      save_to_repository(output_dir, push_to_hub, 
        tokenizer, text_encoder, text_encoder_params, vae, vae_params, unet, repo_id, state)

      train_step_progress_bar.close()

      epochs.write(f"Epoch... ({epoch + 1}/{num_train_epochs} | Loss: {train_metric['loss']})")
