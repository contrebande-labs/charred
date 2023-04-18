import optax


def setup_optimizer(
    learning_rate,
    adam_beta1,
    adam_beta2,
    adam_epsilon,
    adam_weight_decay,
    max_grad_norm,
):
    constant_scheduler = optax.constant_schedule(learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=adam_beta1,
        b2=adam_beta2,
        eps=adam_epsilon,
        weight_decay=adam_weight_decay,
    )

    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adamw,
    )
