import math 

def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * current_step / total_steps))