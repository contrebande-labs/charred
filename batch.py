import torch

def collate(examples):

    # TODO: replace torch.stack with https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.stack.html

    pixel_values = torch.stack(
        [example["pixel_values"] for example in examples]
    ).to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack(
        [example["input_ids"] for example in examples]
    )

    batch = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
    }

    return {k: v.numpy() for k, v in batch.items()}

def setup_dataloader(train_dataset, train_batch_size):

  return torch.utils.data.DataLoader(
    # we don't shuffle here because the dataset is already shuffle 
    train_dataset, shuffle=False, collate_fn=collate, batch_size=train_batch_size, drop_last=True
  )