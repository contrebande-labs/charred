# python
import math

# misc. utils
from tqdm.auto import tqdm

# TODO: change that for JAX equivalents
import torch

# jax/flax
import jax
from flax.training.common_utils import shard
from flax import jax_utils

from batch import collate
from dataset import setup_dataset
from monitoring import setup_logging
from preprocessing import preprocess_train, setup_train_transforms
from training_step import train_step

def training_loop(args, tokenizer, text_encoder, text_encoder_params, vae, vae_params, unet):

  logger = setup_logging()

  caption_column, image_column, dataset = setup_dataset(args)

  if jax.process_index() == 0 and args.max_train_samples is not None:
      dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))

  train_transforms = setup_train_transforms(args)

  # Initialize our training
  rng = jax.random.PRNGKey(args.seed)
  train_rngs = jax.random.split(rng, jax.local_device_count())

  # Train step
  p_train_step = train_step(text_encoder, vae, unet)

  # Set the training transforms
  preprocess_train_lambda = preprocess_train(image_column, caption_column, tokenizer, train_transforms)
  train_dataset = dataset["train"].with_transform(preprocess_train_lambda)

  # batch setup
  total_train_batch_size = args.train_batch_size * jax.local_device_count()
  collate_lambda = collate(tokenizer)
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset, shuffle=True, collate_fn=collate_lambda, batch_size=total_train_batch_size, drop_last=True
  )

  num_update_steps_per_epoch = math.ceil(len(train_dataloader))

  # Scheduler and math around the number of training steps.
  if args.max_train_steps is None:
      args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

  args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

  logger.info("***** Running training *****")
  logger.info(f"  Num examples = {len(train_dataset)}")
  logger.info(f"  Num Epochs = {args.num_train_epochs}")
  logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
  logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
  logger.info(f"  Total optimization steps = {args.max_train_steps}")

  global_step = 0

  epochs = tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
  for epoch in epochs:
      # ======================== Training ================================

      train_metrics = []

      steps_per_epoch = len(train_dataset) // total_train_batch_size
      train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
      # train
      for batch in train_dataloader:
          batch = shard(batch)
          state, train_metric, train_rngs = p_train_step(state, text_encoder_params, vae_params, batch, train_rngs)
          train_metrics.append(train_metric)

          train_step_progress_bar.update(1)

          global_step += 1
          if global_step >= args.max_train_steps:
              break

      train_metric = jax_utils.unreplicate(train_metric)

      train_step_progress_bar.close()
      epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")