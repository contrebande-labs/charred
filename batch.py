import torch

def collate(tokenizer):
    return lambda examples: _collate(tokenizer, examples)

def _collate(tokenizer, examples):

    # TODO: replace with https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.stack.html
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = [example["input_ids"] for example in examples]

    padded_tokens = tokenizer.pad(
        {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    batch = {
        "pixel_values": pixel_values,
        "input_ids": padded_tokens.input_ids,
    }
    batch = {k: v.numpy() for k, v in batch.items()}

    return batch

def setup_dataloader(tokenizer, train_dataset, train_batch_size):

  collate_lambda = collate(tokenizer)

  return torch.utils.data.DataLoader(
    # we don't shuffle here because the dataset is already shuffle 
    train_dataset, shuffle=False, collate_fn=collate_lambda, batch_size=train_batch_size, drop_last=True
  )