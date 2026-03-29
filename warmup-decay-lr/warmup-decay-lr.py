def warmup_decay_schedule(base_lr, warmup_steps, total_steps, current_step):
    """
    Compute the learning rate at a given step using warmup + linear decay.
    """
    # Write code here
    if current_step <= warmup_steps:
        return base_lr * current_step / warmup_steps
    else:
        return base_lr * (total_steps - current_step) / (total_steps - warmup_steps)