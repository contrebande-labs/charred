import time

import logger
from args import parse_args


def training_loop(log_wandb):
    # rng setup

    # dataset setup

    print("dataset loaded...")

    # batch setup

    print("dataloader setup...")

    # Create parallel version of the train step

    print("training step compiling...")

    # Epoch setup
    num_train_epochs = 1000
    t0 = time.monotonic()
    global_training_steps = 0
    global_walltime = time.monotonic()
    for epoch in range(num_train_epochs):
        epoch_walltime = time.monotonic()
        epoch_steps = 0

        for i in range(100):
            batch_walltime = time.monotonic()
            if global_training_steps == 0:
                print("training step compiled (process #%d)..." % 99)

            epoch_steps += 1
            global_training_steps += 1

            if log_wandb:
                global_walltime = time.monotonic() - t0
                delta_time = time.monotonic() - batch_walltime
                log_wandb.log(
                    data={
                        "walltime": global_walltime,
                        "train/step": epoch_steps,
                        "train/global_step": global_training_steps,
                        "train/steps_per_sec": 1 / delta_time,
                        "train/epoch": epoch,
                    },
                    commit=True,
                )

        if log_wandb:
            epoch_walltime = global_walltime - epoch_walltime
            log_wandb.log(
                data={
                    "train/secs_per_epoch": epoch_walltime,
                    "train/global_step": global_training_steps,
                },
                commit=True,
            )


def main():
    args = parse_args()

    # Setup WandB for logging & tracking
    log_wandb = args.log_wandb
    if log_wandb:
        log_wandb.init(
            entity="charred",
            project="charred",
            job_type="train",
            config=args,
        )
        log_wandb.define_metric("*", step_metric="train/global_step")
        log_wandb.define_metric("train/global_step", step_metric="walltime")
        log_wandb.define_metric("train/epoch", step_metric="train/global_step")
        log_wandb.define_metric("train/secs_per_epoch", step_metric="train/epoch")

    print("random generator setup...")

    print("models setup...")

    print("optimizer setup...")

    print("training state initialized...")

    print("states & params replicated to TPUs...")

    # Train!
    print("Training loop init...")
    training_loop(True)

    print("Training loop done...")

    if log_wandb:
        log_wandb.finish()
        print("WandB closed...")


if __name__ == "__main__":
    main()
    main()
