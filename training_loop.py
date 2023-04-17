# python
import math

# misc. utils
from tqdm.auto import tqdm

# jax/flax
import jax
from flax.training.common_utils import shard
from flax import jax_utils

from batch import setup_dataloader
from dataset import setup_dataset
from preprocessing import preprocess_train, setup_train_transforms
from training_step import train_step

def training_loop(tokenizer, text_encoder, text_encoder_params, vae, vae_params, unet, state,
    dataset_name, dataset_config_name, cache_dir, train_data_dir, caption_column, image_column,
    resolution, center_crop, random_flip,
    seed, max_train_steps, num_train_epochs, max_train_samples, train_batch_size):
  
  dataset = setup_dataset(dataset_name, dataset_config_name, cache_dir, train_data_dir, image_column, caption_column)

  if jax.process_index() == 0 and max_train_samples is not None:
      dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(max_train_samples))

  train_transforms = setup_train_transforms(resolution, center_crop, random_flip)

  # Initialize our training
  rng = jax.random.PRNGKey(seed)
  train_rng = jax.random.split(rng, jax.local_device_count())

  # Train step
  p_train_step = train_step(text_encoder, text_encoder_params, vae, vae_params, unet)

  # Set the training transforms
  preprocess_train_lambda = preprocess_train(image_column, caption_column, tokenizer, train_transforms)
  train_dataset = dataset["train"].with_transform(preprocess_train_lambda)

  # batch setup
  total_train_batch_size = train_batch_size * jax.local_device_count()
  train_dataloader = setup_dataloader(tokenizer, train_dataset, total_train_batch_size)
  num_update_steps_per_epoch = math.ceil(len(train_dataloader))

  # Scheduler and math around the number of training steps.
  if max_train_steps is None:
      max_train_steps = num_train_epochs * num_update_steps_per_epoch

  distributed_num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

  global_step = 0

  epochs = tqdm(range(distributed_num_train_epochs), desc="Epoch ... ", position=0)
 
  for epoch in epochs:
      # ======================== Training ================================

      train_metrics = []

      steps_per_epoch = len(train_dataset) // total_train_batch_size
      train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)

      # train
      for batch in train_dataloader:

          batch = shard(batch)

          state, train_rng, train_metric = p_train_step(state, batch, train_rng)

          train_metrics.append(train_metric)

          train_step_progress_bar.update(1)
 
          global_step += 1
          if global_step >= max_train_steps:
              break

      train_metric = jax_utils.unreplicate(train_metric)

      train_step_progress_bar.close()

      epochs.write(f"Epoch... ({epoch + 1}/{num_train_epochs} | Loss: {train_metric['loss']})")