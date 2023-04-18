def _dataset_filter(example):
  caption = example['caption']
  image_url = example['url']
  watermark_probability = example['pwatermark']
  return caption is not None and isinstance(caption, str) and image_url is not None and isinstance(image_url, str) and watermark_probability < 0.6

def dataset_filter():
  # TODO: drop duplicates with image embeddings https://arxiv.org/abs/2303.12733 https://github.com/LAION-AI/laion-dedup https://github.com/LAION-AI/image-deduplication-testset
  return lambda example: _dataset_filter(example)