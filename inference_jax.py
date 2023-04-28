import jax
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import jax.numpy as jnp

import torch

from transformers import ByT5Tokenizer, FlaxT5ForConditionalGeneration


def get_inference_lambda():

    tokenizer = ByT5Tokenizer()

    language_model = FlaxT5ForConditionalGeneration.from_pretrained(
        "google/byt5-base",
        dtype=jnp.float32,
    )
    text_encoder = language_model.encode
    text_encoder_params = replicate(language_model.params)

    def __tokenize_prompt(prompt):

        return shard(
            torch.stack(
                [
                    tokenizer(
                        text=[prompt],
                        max_length=1024,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids
                ]
            )
            .to(memory_format=torch.contiguous_format)
            .float()
            .numpy()
        )

    def __predict_image(tokenized_prompt):

        # Get the text embedding
        text_encoder(
            tokenized_prompt,
            params=text_encoder_params,
            train=False,
        )[0]

    jax_pmap_predict_image = jax.pmap(__predict_image)

    return lambda prompt: jax_pmap_predict_image(__tokenize_prompt(prompt))


if __name__ == "__main__":

    infer = get_inference_lambda("character-aware-diffusion/charred", 87)

    infer(["a running shoe"])
