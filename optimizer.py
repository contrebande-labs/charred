import optax

def setup_optimizer(args):
  constant_scheduler = optax.constant_schedule(args.learning_rate)

  adamw = optax.adamw(
      learning_rate=constant_scheduler,
      b1=args.adam_beta1,
      b2=args.adam_beta2,
      eps=args.adam_epsilon,
      weight_decay=args.adam_weight_decay,
  )

  return optax.chain(
      optax.clip_by_global_norm(args.max_grad_norm),
      adamw,
  )