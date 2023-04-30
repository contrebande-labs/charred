Dataset
  DONE: Integrate LAION-HR: https://huggingface.co/datasets/laion/laion-high-resolution
  Integrate WiT dataset: https://huggingface.co/datasets/google/wit
  Augment all datasets to match ByT5's C4 training language distribution: https://huggingface.co/datasets/allenai/c4 https://huggingface.co/datasets/mc4
  Integrate synthetic HTML/SVG/CSS graphic/layout/typography dataset
  Integrate document layout segmentation and understanding dataset
  Integrate calligraphy dataset
  Integrate grapheme-in-the-wild dataset
  Filter out samples with low-aesthetic image score
  Filter out samples with meaningless captions
  Cache freezed models (ByT5 and VAE) embeddings


Training
  DONE: Implement JAX/FLAX SD 2.1 training pipeline with ByT5-Base instead of CLIP: https://github.com/patil-suraj/stable-diffusion-jax https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_flax.py https://huggingface.co/google/byt5-base https://huggingface.co/blog/stable_diffusion_jax
  DONE: WandB monitoring
  OpenTelemetry monitoring including JAX profiler tracing artifact uploading
  Implement Mini-SNR loss rebalancing: https://arxiv.org/abs/2303.09556
  Implement on-the-fly validation
  Integrate Big Vision optimizaitions: https://github.com/google-research/big_vision
  Use ByT5-Large instead of ByT5-Base: https://huggingface.co/google/byt5-large
  Implement streaming, mini-batching and gradient accumulation with image aspect ratio and tokenized caption size bucketing
  Use ByT5-XXL instead of ByT5-Large: https://huggingface.co/google/byt5-xxl https://github.com/google-research/t5x/blob/main/docs/models.md#byt5-checkpoints https://github.com/google-research/t5x/blob/main/t5x/scripts/convert_tf_checkpoint.py
  Port to JAX and Integrate Imagen, SDXL and Deep Floyd improvements: https://github.com/lucidrains/imagen-pytorch https://github.com/deep-floyd/IF https://stable-diffusion-art.com/sdxl-beta/


Inference
  DONE: Implement JAX/FLAX text-to-image inference pipeline with ByT5-Base instead of CLIP: https://huggingface.co/docs/diffusers/training/text2image https://github.com/patil-suraj/stable-diffusion-jax
  Production AOT with IREE over Java JNI/JNA/Panama: https://github.com/openxla/iree https://github.com/iree-org/iree-jax
  Implement OCR and Document understanging inference pipeline with ByT5 text decoder
  Implement text encoding CPU offloading with int8 precision
  Implement accelerated U-Net prediction and VAE decoding with int8 precision : https://github.com/TimDettmers/bitsandbytes https://huggingface.co/blog/hf-bitsandbytes-integration