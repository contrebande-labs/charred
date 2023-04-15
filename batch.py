# TODO: change that for JAX equivalents
import torch

def collate(tokenizer):
    return lambda examples: _collate(tokenizer, examples)

def _collate(tokenizer, examples):
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