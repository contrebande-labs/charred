import torch


def setup_dataloader(dataset, batch_size):
    def _collate(samples):

        # TODO: replace torch.stack with https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.stack.html

        pixel_values = (
            torch.stack([sample["pixel_values"] for sample in samples])
            .to(memory_format=torch.contiguous_format)
            .float()
            .numpy()
        )

        input_ids = (
            torch.stack([sample["input_ids"] for sample in samples])
            .to(memory_format=torch.contiguous_format)
            .float()
            .numpy()
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }

    return torch.utils.data.DataLoader(
        # we don't shuffle here because the dataset is already shuffled
        dataset,
        collate_fn=_collate,
        batch_size=batch_size,
        drop_last=True,
    )
