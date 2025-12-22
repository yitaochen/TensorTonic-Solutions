import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    #print(f"debug {param}, {grad}, {m}, {v}, {t}")
    # Write code here
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad**2
    m = m_new / (1 - beta1**t)
    v = v_new / (1 - beta2**t)
    # print(f"v value is {v}")
    param_new = param - lr * m / (np.sqrt(v) + eps)
    #print(f"output {param_new}, {m_new}, {v_new}")
    return (param_new, m_new, v_new)