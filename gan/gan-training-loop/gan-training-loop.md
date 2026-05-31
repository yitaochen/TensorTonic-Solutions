# <span style="font-size: 20px;">GAN Training Loop</span>

<span style="font-size: 14px;">The GAN training loop is the alternating optimization procedure at the heart of Generative Adversarial Networks (Goodfellow et al., 2014). Each iteration pits two networks -- a discriminator D and a generator G -- against each other: D is updated to better separate real from fake data, then G is updated to better fool D. Repeated over many iterations, this drives both networks toward equilibrium where G produces samples indistinguishable from real data.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">One iteration of the GAN training loop performs two sequential updates. First, D is trained to maximize its ability to classify real vs. generated data. Then, G is trained to maximize the probability that D misclassifies generated outputs as real. This alternating structure is the defining feature of adversarial training.</span>

<span style="font-size: 14px;">D and G have opposing objectives. D wants to assign high scores to real data and low scores to fake data. G wants D to assign high scores to its outputs. Neither network's loss can be minimized in isolation -- the optimal strategy for each depends on the current state of the other. The training loop addresses this by holding one network fixed while updating the other.</span>

<span style="font-size: 14px;">In this problem, the discriminator uses a sigmoid linear model: a linear transformation followed by sigmoid activation producing a probability score. The training loop logic generalizes to arbitrarily complex architectures.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">Sigmoid Scoring</span>

<span style="font-size: 14px;">The discriminator scores input $x$ via a linear transformation and sigmoid:</span>

$$
D(x) = \sigma(W_D \cdot x) = \frac{1}{1 + e^{-W_D \cdot x}}
$$

<span style="font-size: 14px;">where $W_D$ is the discriminator weight vector. The output $D(x) \in (0, 1)$ represents the probability that $x$ came from the real data distribution.</span>

### <span style="font-size: 14px;">Discriminator Loss</span>

$$
\mathcal{L}_D = -\text{mean}\left[\log(D(x_{\text{real}})) + \log(1 - D(G(z)))\right]
$$

* <span style="font-size: 14px;">**$\log(D(x_{\text{real}}))$:** Rewards D for assigning high probability to real data. Maximized when $D(x_{\text{real}}) \to 1$.</span>
* <span style="font-size: 14px;">**$\log(1 - D(G(z)))$:** Rewards D for assigning low probability to fake data. Maximized when $D(G(z)) \to 0$.</span>
* <span style="font-size: 14px;">**Negative sign:** Converts maximization into minimization for gradient descent.</span>

### <span style="font-size: 14px;">Generator Loss</span>

$$
\mathcal{L}_G = -\text{mean}\left[\log(D(G(z)))\right]
$$

<span style="font-size: 14px;">This is the "non-saturating" variant from Goodfellow et al. Rather than minimizing $\log(1 - D(G(z)))$ (which saturates when D is confident), G maximizes $\log(D(G(z)))$, providing stronger gradients early in training.</span>

### <span style="font-size: 14px;">Epsilon Clipping</span>

<span style="font-size: 14px;">Both losses involve logarithms of values in $(0, 1)$. When $D(x) \approx 0$ or $D(x) \approx 1$, the log approaches $-\infty$. Epsilon clipping prevents this:</span>

$$
\log(\text{clip}(p, \epsilon, 1 - \epsilon)) \quad \text{where } \epsilon = 10^{-8}
$$

### <span style="font-size: 14px;">The Minimax Formulation</span>

<span style="font-size: 14px;">The full game from the paper:</span>

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

<span style="font-size: 14px;">The training loop solves this approximately by alternating: one gradient step on D (ascending), then one on G (descending, using the non-saturating loss).</span>

---

## <span style="font-size: 16px;">The Alternating Optimization</span>

<span style="font-size: 14px;">Why not train D and G jointly? The minimax structure means D wants to maximize $V(D, G)$ while G wants to minimize it. Joint optimization creates conflicting gradients that cancel out or oscillate, preventing either network from learning.</span>

<span style="font-size: 14px;">Alternating optimization decomposes the problem into two cooperative subproblems. Fix G, update D: a standard classification problem. Fix D, update G: a standard generation problem. Each subproblem is well-posed and solvable with gradient descent. This is coordinate descent in function space -- optimizing one "coordinate" (network) while holding the other fixed.</span>

### <span style="font-size: 14px;">$k$ Steps of D per 1 Step of G</span>

<span style="font-size: 14px;">Algorithm 1 in Goodfellow et al. specifies $k$ discriminator steps per single generator step. D must be a reasonably accurate critic before G's gradients become meaningful. However, the paper states: "In practice, we use $k = 1$, the least expensive option." Training D too many steps risks it becoming too strong, causing G's gradients to vanish. The $k = 1$ choice means strict alternation -- one update each per iteration.</span>

---

## <span style="font-size: 16px;">Step-by-Step Training Procedure</span>

### <span style="font-size: 14px;">Step 1: Sample Real Batch</span>

<span style="font-size: 14px;">Draw $m$ samples $\{x^{(1)}, \ldots, x^{(m)}\}$ from the real data distribution $p_{\text{data}}$.</span>

### <span style="font-size: 14px;">Step 2: Sample Noise and Generate Fake Batch</span>

<span style="font-size: 14px;">Draw $m$ noise vectors from $p_z(z)$ (typically Gaussian). Pass through G to produce fake samples $\{G(z^{(1)}), \ldots, G(z^{(m)})\}$.</span>

### <span style="font-size: 14px;">Step 3: Compute Discriminator Scores</span>

<span style="font-size: 14px;">Run both batches through D:</span>

* <span style="font-size: 14px;">$p_{\text{real}}^{(i)} = D(x^{(i)})$ for each real sample</span>
* <span style="font-size: 14px;">$p_{\text{fake}}^{(i)} = D(G(z^{(i)}))$ for each fake sample</span>

### <span style="font-size: 14px;">Step 4: Compute $\mathcal{L}_D$ and Update D</span>

$$
\mathcal{L}_D = -\frac{1}{m}\sum_{i=1}^{m}\left[\log(p_{\text{real}}^{(i)}) + \log(1 - p_{\text{fake}}^{(i)})\right]
$$

<span style="font-size: 14px;">Compute gradients w.r.t. D's parameters only and apply gradient descent. G's parameters are frozen.</span>

### <span style="font-size: 14px;">Step 5: Compute $\mathcal{L}_G$ and Update G</span>

$$
\mathcal{L}_G = -\frac{1}{m}\sum_{i=1}^{m}\log(p_{\text{fake}}^{(i)})
$$

<span style="font-size: 14px;">Compute gradients w.r.t. G's parameters only and apply gradient descent. D's parameters are frozen.</span>

### <span style="font-size: 14px;">Step 6: Return Losses</span>

<span style="font-size: 14px;">Record $\mathcal{L}_D$ and $\mathcal{L}_G$ for monitoring, rounded to 4 decimal places.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The training loop is Algorithm 1 in "Generative Adversarial Nets" (Goodfellow et al., 2014). The algorithm structure:</span>

* <span style="font-size: 14px;">**Outer loop:** For each training iteration...</span>
* <span style="font-size: 14px;">**Inner loop (D):** For $k$ steps, sample real and noise minibatches, compute D's gradient, update D by ascending its stochastic gradient.</span>
* <span style="font-size: 14px;">**Single step (G):** Sample noise, compute G's gradient, update G by descending its stochastic gradient.</span>

<span style="font-size: 14px;">The paper recommends SGD with momentum ($0.5$ initially, increasing to $0.7$). It proves that if D reaches its optimum at each step, the procedure minimizes the Jensen-Shannon divergence between real and generated distributions. In practice with $k = 1$, D does not reach optimality, so training dynamics are approximate.</span>

<span style="font-size: 14px;">The characteristic training pattern: early on, D easily separates real from fake. As G improves, D's task becomes harder, and scores converge toward $0.5$ for both -- indicating D cannot distinguish them. This $D(x) = 0.5$ for all $x$ is the theoretical equilibrium.</span>

---

## <span style="font-size: 16px;">Why Alternating Works</span>

<span style="font-size: 14px;">The GAN objective is a two-player zero-sum game. The solution is a Nash equilibrium: a pair $(D^*, G^*)$ where neither player can improve unilaterally. Finding this requires navigating a saddle point -- maximizing over D while minimizing over G.</span>

<span style="font-size: 14px;">Standard gradient descent cannot find saddle points; it follows gradients downhill for all parameters, pushing D and G in conflicting directions. Simultaneous gradient descent/ascent is known to exhibit rotational dynamics, cycling around the equilibrium rather than converging.</span>

<span style="font-size: 14px;">Alternating optimization resolves this by fully computing one player's best response before the other moves. When D is updated with G fixed, D moves toward its best response. When G is then updated, it responds to the improved D. This best-response dynamic has better convergence properties than simultaneous updates.</span>

<span style="font-size: 14px;">The key requirement is balance: neither player should get too far ahead. If D becomes too strong, G's gradients vanish. If G becomes too strong relative to D, the training signal becomes noisy and unreliable.</span>

---

## <span style="font-size: 16px;">Training Stability</span>

### <span style="font-size: 14px;">D/G Balance</span>

<span style="font-size: 14px;">The most critical factor is maintaining balance between D and G. D should be strong enough to provide meaningful gradients to G, but not so strong that it perfectly classifies all fake samples and kills G's gradients.</span>

### <span style="font-size: 14px;">When D Dominates</span>

<span style="font-size: 14px;">D assigns $D(G(z)) \approx 0$ to all generated samples. The sigmoid saturates near zero, so $\nabla_G \log(D(G(z)))$ vanishes. G receives no useful signal and stops improving -- the vanishing gradient problem.</span>

### <span style="font-size: 14px;">When G Dominates</span>

<span style="font-size: 14px;">G produces outputs that fool D, so $D(G(z)) \approx 1$. D cannot distinguish real from fake, providing no directional gradient. Worse, G may have found a single mode that fools D -- mode collapse -- producing limited variety regardless of input noise.</span>

### <span style="font-size: 14px;">Learning Rate Choices</span>

<span style="font-size: 14px;">A common practice is using a lower learning rate for G than D, ensuring D stays slightly ahead. The Adam optimizer with $\beta_1 = 0.5$ (not default $0.9$) is widely used because lower momentum reduces oscillatory dynamics in the adversarial setting.</span>

### <span style="font-size: 14px;">Label Smoothing</span>

<span style="font-size: 14px;">Instead of hard labels (1 real, 0 fake), one-sided label smoothing uses soft labels (e.g., 0.9 for real, 0 for fake). This prevents D from becoming overconfident, maintaining gradient flow. Only the real label is smoothed -- smoothing the fake label encourages D to assign non-zero probability to obvious fakes.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider 3 real and 3 fake samples in 2D, with discriminator weights $W_D = [1.0, -0.5]$.</span>

### <span style="font-size: 14px;">Data</span>

* <span style="font-size: 14px;">**Real:** $x_1 = [1.5, 0.3]$, $x_2 = [2.0, -0.1]$, $x_3 = [1.8, 0.5]$</span>
* <span style="font-size: 14px;">**Fake:** $g_1 = [0.2, 1.0]$, $g_2 = [-0.3, 0.8]$, $g_3 = [0.5, 1.2]$</span>

### <span style="font-size: 14px;">D Scores on Real Data</span>

* <span style="font-size: 14px;">$W_D \cdot x_1 = 1.5 - 0.15 = 1.35$, $p_{\text{real},1} = \sigma(1.35) \approx 0.7941$</span>
* <span style="font-size: 14px;">$W_D \cdot x_2 = 2.0 + 0.05 = 2.05$, $p_{\text{real},2} = \sigma(2.05) \approx 0.8858$</span>
* <span style="font-size: 14px;">$W_D \cdot x_3 = 1.8 - 0.25 = 1.55$, $p_{\text{real},3} = \sigma(1.55) \approx 0.8250$</span>

### <span style="font-size: 14px;">D Scores on Fake Data</span>

* <span style="font-size: 14px;">$W_D \cdot g_1 = 0.2 - 0.5 = -0.3$, $p_{\text{fake},1} = \sigma(-0.3) \approx 0.4256$</span>
* <span style="font-size: 14px;">$W_D \cdot g_2 = -0.3 - 0.4 = -0.7$, $p_{\text{fake},2} = \sigma(-0.7) \approx 0.3318$</span>
* <span style="font-size: 14px;">$W_D \cdot g_3 = 0.5 - 0.6 = -0.1$, $p_{\text{fake},3} = \sigma(-0.1) \approx 0.4750$</span>

### <span style="font-size: 14px;">Computing $\mathcal{L}_D$</span>

$$
\mathcal{L}_D = -\frac{1}{3}\left[\log(0.7941) + \log(0.8858) + \log(0.8250) + \log(0.5744) + \log(0.6682) + \log(0.5250)\right]
$$

$$
= -\frac{1}{3}\left[-0.2306 - 0.1214 - 0.1924 - 0.5546 - 0.4035 - 0.6444\right]
$$

$$
= -\frac{1}{3}(-2.1469) = 0.7156
$$

<span style="font-size: 14px;">D's loss is positive, indicating room to improve at separating real from fake.</span>

### <span style="font-size: 14px;">Computing $\mathcal{L}_G$</span>

$$
\mathcal{L}_G = -\frac{1}{3}\left[\log(0.4256) + \log(0.3318) + \log(0.4750)\right]
$$

$$
= -\frac{1}{3}\left[-0.8544 - 1.1035 - 0.7445\right] = -\frac{1}{3}(-2.7024) = 0.9008
$$

<span style="font-size: 14px;">G's loss is higher than D's, typical early in training when D outperforms G. G needs the fake scores to move toward 1.0.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Training G and D on the Same Forward Pass</span>

<span style="font-size: 14px;">Computing both losses from a single forward pass and backpropagating simultaneously violates alternating optimization. D's optimizer step must complete before computing G's loss. The computation graphs must be separate.</span>

### <span style="font-size: 14px;">Forgetting to Detach Fake Samples for D Update</span>

<span style="font-size: 14px;">When computing $\mathcal{L}_D$, fake samples $G(z)$ must be treated as fixed data -- D should not backpropagate through G. In PyTorch, call `.detach()` on the generator output before passing to D, or wrap the generator forward pass in `torch.no_grad()`. Failing to detach causes D's gradient update to corrupt G's computation graph.</span>

### <span style="font-size: 14px;">Wrong Alternation Order</span>

<span style="font-size: 14px;">Algorithm 1 specifies: update D first, then G. Reversing this means G optimizes against a stale discriminator. D should always be updated first so G receives gradients from the most current critic.</span>

### <span style="font-size: 14px;">Unbalanced Training Causing Mode Collapse</span>

<span style="font-size: 14px;">If G is updated too many times relative to D, it exploits weaknesses in D's decision boundary, collapsing to a narrow set of outputs that reliably fool D. This is mode collapse -- G ignores the diversity of the real distribution. The fix is ensuring D is updated at least as frequently as G.</span>

### <span style="font-size: 14px;">Vanishing Gradients from the Original Loss</span>

<span style="font-size: 14px;">The original formulation has G minimizing $\log(1 - D(G(z)))$. When D is strong, $D(G(z)) \approx 0$, so $\log(1 - D(G(z))) \approx 0$ and gradients vanish. The non-saturating alternative $-\log(D(G(z)))$ has gradient proportional to $1/D(G(z))$, which grows as $D(G(z)) \to 0$, providing strong signal precisely when G needs it most.</span>

### <span style="font-size: 14px;">Forgetting Epsilon Clipping</span>

<span style="font-size: 14px;">Computing $\log(0)$ produces $-\infty$ or NaN, corrupting all subsequent gradients. Clipping discriminator outputs to $[\epsilon, 1 - \epsilon]$ before taking logarithms is essential. With $\epsilon = 10^{-8}$, the effect on valid probabilities is negligible but prevents catastrophic numerical failures at the boundaries.</span>

---