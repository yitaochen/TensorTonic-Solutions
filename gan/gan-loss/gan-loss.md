# <span style="font-size: 20px;">GAN Loss Functions</span>

<span style="font-size: 14px;">Generative Adversarial Networks pit two networks against each other: a generator G that synthesizes fake data and a discriminator D that tries to tell real from fake. The loss functions that drive this adversarial training are the core mechanism behind GANs. Understanding how L_D and L_G are formulated, why the non-saturating generator loss is preferred, and where numerical instabilities arise is essential for implementing GANs correctly. This theory is grounded in Goodfellow et al. (2014), the paper that introduced the GAN framework.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The GAN loss functions define the adversarial objective that drives GAN training. The discriminator D(x) outputs a probability that input x is real (from the training data) rather than fake (produced by the generator). The generator G(z) takes random noise z and produces synthetic samples. The two networks are trained with competing objectives: D wants to correctly classify real vs fake, while G wants to fool D into classifying fake samples as real.</span>

<span style="font-size: 14px;">This adversarial setup means neither network has a fixed target. The discriminator's target depends on the current generator, and the generator's target depends on the current discriminator. Training alternates between updating D and G, each using its own loss function. The loss functions encode what "winning" means for each player in this two-player game.</span>

<span style="font-size: 14px;">The adversarial objective is what makes GANs fundamentally different from other generative models like VAEs. There is no explicit density estimation or reconstruction loss. Instead, the quality of generated samples emerges purely from the competitive dynamics between the two networks.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The full minimax value function from Goodfellow et al. (2014) is:</span>

$$
\min_G \max_D V(G, D) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

<span style="font-size: 14px;">This single equation encodes the entire adversarial game. D maximizes V, G minimizes V. In practice, we decompose this into two separate losses.</span>

<span style="font-size: 14px;">**Discriminator loss:**</span>

$$
L_D = -\frac{1}{m}\sum_{i=1}^{m}\left[\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))\right]
$$

<span style="font-size: 14px;">The negative sign converts the maximization into a minimization (since optimizers minimize). The two terms reward D for assigning high probability to real samples and low probability to fake samples.</span>

<span style="font-size: 14px;">**Generator loss (non-saturating form):**</span>

$$
L_G = -\frac{1}{m}\sum_{i=1}^{m}\log D(G(z^{(i)}))
$$

<span style="font-size: 14px;">Rather than minimizing $\log(1 - D(G(z)))$ from the original value function, the non-saturating form maximizes $\log D(G(z))$. This is the formulation used in practice and the one this problem implements.</span>

---

## <span style="font-size: 16px;">The Minimax Game</span>

<span style="font-size: 14px;">The GAN objective is a two-player minimax game from game theory. The discriminator is the maximizing player: it wants $V(G, D)$ to be as large as possible. The generator is the minimizing player: it wants $V(G, D)$ to be as small as possible. This is a zero-sum game because any gain for D is a loss for G and vice versa.</span>

<span style="font-size: 14px;">The game-theoretic solution concept is a Nash equilibrium: a pair $(G^*, D^*)$ where neither player can improve by unilaterally changing its strategy. Goodfellow et al. prove that this equilibrium exists and occurs when $p_g = p_{\text{data}}$, meaning the generator perfectly replicates the true data distribution. At this point, the optimal discriminator outputs $D^*(x) = 1/2$ for all x, since real and fake are indistinguishable.</span>

<span style="font-size: 14px;">In practice, training alternates: take k steps of discriminator updates (typically k=1), then one step of generator update. This alternating optimization approximates the theoretical simultaneous game. The discriminator must stay ahead of the generator to provide useful gradient signals, but not so far ahead that the generator receives vanishing gradients.</span>

---

## <span style="font-size: 16px;">Why Non-Saturating Generator Loss</span>

<span style="font-size: 14px;">The original minimax formulation says G should minimize $\log(1 - D(G(z)))$. The problem is gradient saturation. Early in training, the generator produces obviously fake samples, so D confidently outputs values near 0 for fake data. When $D(G(z)) \approx 0$:</span>

<span style="font-size: 14px;">**Saturating form (original):** $\log(1 - D(G(z))) \approx \log(1) = 0$. The gradient with respect to $G$ is nearly zero. The generator receives almost no learning signal precisely when it needs it most.</span>

<span style="font-size: 14px;">**Non-saturating form:** $-\log(D(G(z))) \approx -\log(0) \to +\infty$. The gradient is large and points in a useful direction. The generator gets strong gradients that push it to produce better samples.</span>

<span style="font-size: 14px;">Goodfellow et al. explicitly recommend this in Section 3: "Rather than training G to minimize $\log(1 - D(G(z)))$ we can train G to maximize $\log D(G(z))$. This objective function results in the same fixed point of the dynamics of G and D but provides much stronger gradients early in learning."</span>

<span style="font-size: 14px;">Both forms share the same fixed point: the optimal generator makes $D(G(z)) = 1/2$, at which point both forms yield well-defined, non-zero gradients. The difference is purely in early training dynamics, but this difference is critical for practical training.</span>

---

## <span style="font-size: 16px;">The Discriminator Loss</span>

<span style="font-size: 14px;">The discriminator loss has two terms, each serving a distinct role:</span>

<span style="font-size: 14px;">**Term 1: $-\log D(x)$ for real samples.** Penalizes D for assigning low probability to real data. When D correctly identifies a real sample with $D(x) \approx 1$, the loss is $-\log(1) \approx 0$. When D mistakenly assigns $D(x) \approx 0$, the loss is $-\log(0) \to \infty$. This term trains D to recognize real data.</span>

<span style="font-size: 14px;">**Term 2: $-\log(1 - D(G(z)))$ for fake samples.** Penalizes D for assigning high probability to generated data. When D correctly rejects a fake with $D(G(z)) \approx 0$, the loss is $-\log(1) \approx 0$. When D is fooled with $D(G(z)) \approx 1$, the loss is $-\log(0) \to \infty$. This term trains D to reject fakes.</span>

<span style="font-size: 14px;">Both terms are necessary. Without the real term, D could minimize by outputting 0 for everything. Without the fake term, D could minimize by outputting 1 for everything. Together, they force D to learn the decision boundary between real and fake. The discriminator loss is essentially binary cross-entropy with label 1 for real and label 0 for fake.</span>

---

## <span style="font-size: 16px;">Numerical Stability</span>

<span style="font-size: 14px;">The logarithm is undefined at 0 and goes to $-\infty$. In GAN training, the arguments to $\log$ are discriminator output probabilities that can numerically reach exactly 0.0 or 1.0 due to floating-point limitations.</span>

<span style="font-size: 14px;">**Where the problem occurs:**</span>

* <span style="font-size: 14px;">$\log D(x)$: if $D(x) = 0$ (discriminator completely wrong on a real sample).</span>
* <span style="font-size: 14px;">$\log(1 - D(G(z)))$: if $D(G(z)) = 1$ (discriminator completely fooled).</span>
* <span style="font-size: 14px;">$\log D(G(z))$: if $D(G(z)) = 0$ (discriminator completely rejects fake, used in generator loss).</span>

<span style="font-size: 14px;">**The fix: epsilon clipping.** Before taking the logarithm, clamp the argument to at least $\epsilon = 10^{-8}$: `torch.log(prob + epsilon)`. This bounds the loss at $-\log(10^{-8}) \approx 18.42$ instead of infinity.</span>

<span style="font-size: 14px;">**Why probabilities hit extremes:** The sigmoid $\sigma(x) = 1/(1+e^{-x})$ saturates for large $|x|$. A logit of $+40$ gives sigmoid output $1 - 4.25 \times 10^{-18}$, which rounds to 1.0 in float32. A logit of $-40$ rounds to 0.0. This happens routinely when D is very confident, which is common early in training.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Goodfellow et al. (2014), "Generative Adversarial Nets," established the theoretical foundation for adversarial training. The key results:</span>

<span style="font-size: 14px;">**Proposition 1:** For a fixed generator G, the optimal discriminator is:</span>

$$
D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

<span style="font-size: 14px;">The optimal discriminator outputs the ratio of real data density to total density. When $p_g = p_{\text{data}}$, this gives $D^*(x) = 1/2$ everywhere.</span>

<span style="font-size: 14px;">**Proposition 2 (Theorem 1):** The global minimum of $C(G) = \max_D V(G, D)$ is achieved if and only if $p_g = p_{\text{data}}$, with $C(G) = -\log 4$. The proof shows $C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)$, which is minimized when the two distributions match.</span>

<span style="font-size: 14px;">The paper states: "G and D play a two-player minimax game with value function $V(G,D) = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$." This value function is the heart of the GAN framework, and the loss functions we implement are its practical decomposition.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a mini-batch of size $m = 4$ with $\epsilon = 10^{-8}$.</span>

<span style="font-size: 14px;">**Scenario: Decent discriminator (partially trained)**</span>

<span style="font-size: 14px;">Real samples: $D(x^{(1)}) = 0.9$, $D(x^{(2)}) = 0.8$, $D(x^{(3)}) = 0.7$, $D(x^{(4)}) = 0.85$</span>

<span style="font-size: 14px;">Fake samples: $D(G(z^{(1)})) = 0.3$, $D(G(z^{(2)})) = 0.2$, $D(G(z^{(3)})) = 0.4$, $D(G(z^{(4)})) = 0.15$</span>

<span style="font-size: 14px;">**Computing $L_D$:**</span>

<span style="font-size: 14px;">Real term: $\log(0.9) + \log(0.8) + \log(0.7) + \log(0.85) = -0.1054 + (-0.2231) + (-0.3567) + (-0.1625) = -0.8477$</span>

<span style="font-size: 14px;">Fake term: $\log(0.7) + \log(0.8) + \log(0.6) + \log(0.85) = -0.3567 + (-0.2231) + (-0.5108) + (-0.1625) = -1.2531$</span>

$$
L_D = -\frac{1}{4}[(-0.8477) + (-1.2531)] = -\frac{1}{4}(-2.1008) = 0.5252
$$

<span style="font-size: 14px;">**Computing $L_G$ (non-saturating):**</span>

<span style="font-size: 14px;">$\log(0.3) + \log(0.2) + \log(0.4) + \log(0.15) = -1.2040 + (-1.6094) + (-0.9163) + (-1.8971) = -5.6268$</span>

$$
L_G = -\frac{1}{4}(-5.6268) = 1.4067
$$

<span style="font-size: 14px;">**Interpretation:** $L_D = 0.5252$ is moderate -- D is doing a reasonable job. $L_G = 1.4067$ is high -- D successfully rejects most fakes, so G has room to improve.</span>

<span style="font-size: 14px;">**Scenario: Perfect discriminator (without epsilon clipping)**</span>

<span style="font-size: 14px;">If $D(x) = 1.0$ for all real and $D(G(z)) = 0.0$ for all fake: $\log(0) = -\infty$. With $\epsilon = 10^{-8}$: $\log(10^{-8}) = -18.42$, giving $L_G = 18.42$ -- large but finite. This is why epsilon clipping is essential.</span>

<span style="font-size: 14px;">**Scenario: Optimal equilibrium**</span>

<span style="font-size: 14px;">When $D^*(x) = 0.5$ everywhere: $L_D = -[\log(0.5) + \log(0.5)] = 1.3863 = \log 4$. And $L_G = -\log(0.5) = 0.6931 = \log 2$. These are the loss values at Nash equilibrium.</span>

---

## <span style="font-size: 16px;">Loss Variants</span>

<span style="font-size: 14px;">The original GAN loss has known training instabilities. Several alternatives have been proposed:</span>

### <span style="font-size: 14px;">Wasserstein Loss (WGAN)</span>

<span style="font-size: 14px;">Arjovsky et al. (2017) replaced JS divergence with the Wasserstein-1 distance. The critic outputs an unconstrained real number: $L_D = -\mathbb{E}[D(x)] + \mathbb{E}[D(G(z))]$, $L_G = -\mathbb{E}[D(G(z))]$. No logarithms, no sigmoid, no saturation. Requires weight clipping or gradient penalty (WGAN-GP) to enforce the Lipschitz constraint.</span>

### <span style="font-size: 14px;">Hinge Loss</span>

<span style="font-size: 14px;">Used in SNGAN (Miyato et al., 2018) and BigGAN (Brock et al., 2019): $L_D = -\mathbb{E}[\min(0, -1 + D(x))] - \mathbb{E}[\min(0, -1 - D(G(z)))]$, $L_G = -\mathbb{E}[D(G(z))]$. D stops being rewarded once confident enough (margin of 1), preventing it from becoming too strong.</span>

### <span style="font-size: 14px;">Least-Squares GAN (LSGAN)</span>

<span style="font-size: 14px;">Mao et al. (2017) used squared error: $L_D = \mathbb{E}[(D(x) - 1)^2] + \mathbb{E}[D(G(z))^2]$, $L_G = \mathbb{E}[(D(G(z)) - 1)^2]$. Minimizes Pearson $\chi^2$ divergence. Provides non-vanishing gradients for samples far from the decision boundary.</span>

### <span style="font-size: 14px;">Relativistic GAN (RaGAN)</span>

<span style="font-size: 14px;">Jolicoeur-Martineau (2019) proposed that D should estimate probability that real data is more realistic than fake: $\tilde{D}(x, G(z)) = \sigma(C(x) - C(G(z)))$. Both real and fake contribute to both updates, improving stability.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Using the Saturating Generator Loss</span>

<span style="font-size: 14px;">Implementing $L_G = \log(1 - D(G(z)))$ instead of $L_G = -\log(D(G(z)))$ is the most common GAN mistake. The saturating form produces vanishing gradients early in training when D easily rejects G's output. Training will appear to proceed but G barely improves. Always use the non-saturating form.</span>

### <span style="font-size: 14px;">Forgetting Epsilon Clipping</span>

<span style="font-size: 14px;">Without `epsilon = 1e-8` before the log, training produces NaN losses as soon as D outputs probability near 0 or 1. This can happen within the first few hundred iterations. Both discriminator and generator losses need this protection.</span>

### <span style="font-size: 14px;">Wrong Sign Conventions</span>

<span style="font-size: 14px;">The value function $V(G,D)$ is maximized by D, but optimizers minimize. Forgetting the negative sign in $L_D = -V$ causes D to get worse over time. Similarly, the non-saturating generator loss needs the negative: $L_G = -\log D(G(z))$, not $+\log D(G(z))$. Getting either sign wrong makes the network optimize in the wrong direction.</span>

### <span style="font-size: 14px;">Computing Generator Loss with Detached Discriminator</span>

<span style="font-size: 14px;">When computing $L_G$, gradients must flow through D back into G. If D's computation graph is detached (e.g., `D(fake.detach())` or `torch.no_grad()` on D's forward pass during G's update), no gradients reach G. The correct pattern: generate fakes, pass through D without detaching, compute $L_G$, backpropagate. Detaching is correct during D's update, but wrong during G's update.</span>

### <span style="font-size: 14px;">Not Detaching Fakes During Discriminator Update</span>

<span style="font-size: 14px;">The converse: when computing $L_D$, fake samples should be detached from G's graph (`D(fake.detach())`). Otherwise, D's backward pass computes unnecessary gradients for G, wasting memory. If both optimizers step, G receives an unintended update that destabilizes training.</span>