def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Write code here
    # print(f"warup_steps: {warmup_steps} step: {step} total_steps: {total_steps}")
    if step < warmup_steps:
        return step * initial_lr / warmup_steps
    elif warmup_steps <= step <= total_steps:
        if total_steps - warmup_steps > 0:
            return final_lr + (initial_lr - final_lr) * (total_steps - step) / (total_steps - warmup_steps)
        else:
            return final_lr
    else:
        return final_lr
