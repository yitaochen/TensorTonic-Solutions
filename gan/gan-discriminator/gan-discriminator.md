# <span style="font-size: 20px;">GAN Discriminator</span>

<span style="font-size: 14px;">The discriminator is the binary classifier at the heart of a Generative Adversarial Network. Given an input $x$, it outputs a scalar probability $D(x) \in [0, 1]$ estimating whether $x$ came from the real training data or was fabricated by the generator. Values near 1 mean "probably real," values near 0 mean "probably fake." This probability drives the adversarial training dynamic introduced in Goodfellow et al. (2014).</span>

---

## <span style="font-size: 16px;">What It Is: A Binary Classifier</span>

<span style="font-size: 14px;">The discriminator distinguishes between two classes: real samples drawn from $p_{\text{data}}(x)$, and fake samples produced by the generator $G(z)$ where $z \sim p_z(z)$ is a latent noise vector.</span>

<span style="font-size: 14px;">In its simplest form, the discriminator is a single-layer feedforward network. For input $x \in \mathbb{R}^d$:</span>

$$
D(x) = \sigma(x \cdot W)
$$

<span style="font-size: 14px;">where $W \in \mathbb{R}^{d \times 1}$ is a learnable weight matrix and $\sigma$ is the sigmoid function. The weight matrix projects the $d$-dimensional input to a single scalar, and sigmoid squashes it into a probability.</span>

<span style="font-size: 14px;">During training, the discriminator receives inputs from two sources:</span>

* <span style="font-size: 14px;">**Real samples** $x \sim p_{\text{data}}(x)$: drawn from the training dataset. Target: $D(x) \approx 1$.</span>
* <span style="font-size: 14px;">**Fake samples** $\tilde{x} = G(z)$: produced by the generator. Target: $D(G(z)) \approx 0$.</span>

<span style="font-size: 14px;">The discriminator does not know which source a given sample comes from. It must learn to distinguish real from fake purely from the statistical properties of the data.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">The Sigmoid Function</span>

<span style="font-size: 14px;">The sigmoid maps any real-valued input to $(0, 1)$:</span>

$$
\sigma(a) = \frac{1}{1 + e^{-a}}
$$

<span style="font-size: 14px;">Properties relevant to the discriminator:</span>

* <span style="font-size: 14px;">**$\sigma(0) = 0.5$:** The decision boundary. Maximum uncertainty.</span>
* <span style="font-size: 14px;">**$\sigma(a) \to 1$ as $a \to +\infty$:** High confidence the input is real.</span>
* <span style="font-size: 14px;">**$\sigma(a) \to 0$ as $a \to -\infty$:** High confidence the input is fake.</span>
* <span style="font-size: 14px;">**Derivative:** $\sigma'(a) = \sigma(a)(1 - \sigma(a))$, maximized at $a = 0$, vanishing at extremes.</span>

### <span style="font-size: 14px;">Discriminator Forward Pass</span>

<span style="font-size: 14px;">For input $x \in \mathbb{R}^d$ and weight $W \in \mathbb{R}^{d \times 1}$:</span>

<span style="font-size: 14px;">**Step 1.** Compute the logit: $a = x \cdot W$, producing a scalar $a \in \mathbb{R}$.</span>

<span style="font-size: 14px;">**Step 2.** Apply sigmoid: $D(x) = \sigma(a) = \frac{1}{1 + e^{-a}}$.</span>

<span style="font-size: 14px;">The logit $a$ is a log-odds ratio: $a = \log \frac{D(x)}{1 - D(x)}$.</span>

### <span style="font-size: 14px;">Output Interpretation</span>

* <span style="font-size: 14px;">**$D(x) = 0.95$:** 95% confident $x$ is real. Logit $a \approx 2.94$.</span>
* <span style="font-size: 14px;">**$D(x) = 0.50$:** Maximum uncertainty. Logit $a = 0$.</span>
* <span style="font-size: 14px;">**$D(x) = 0.05$:** 95% confident $x$ is fake. Logit $a \approx -2.94$.</span>

---

## <span style="font-size: 16px;">The Discriminator's Role in the Minimax Game</span>

<span style="font-size: 14px;">The GAN objective from Goodfellow et al. (2014) is a two-player minimax game:</span>

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

<span style="font-size: 14px;">The discriminator maximizes $V(D, G)$. The two terms from the discriminator's perspective:</span>

* <span style="font-size: 14px;">**$\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)]$:** For real samples, the discriminator wants $D(x) \to 1$, making $\log D(x) \to 0$ (its maximum). Small $D(x)$ yields a large negative penalty.</span>
* <span style="font-size: 14px;">**$\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$:** For fake samples, the discriminator wants $D(G(z)) \to 0$, making $\log(1 - D(G(z))) \to 0$. Being fooled ($D(G(z))$ large) yields a large negative penalty.</span>

<span style="font-size: 14px;">In practice, this is equivalent to minimizing binary cross-entropy with label 1 for real and 0 for fake:</span>

$$
\mathcal{L}_D = -\frac{1}{m} \sum_{i=1}^{m} \left[ \log D(x^{(i)}) + \log(1 - D(G(z^{(i)}))) \right]
$$

<span style="font-size: 14px;">The discriminator performs gradient descent on $\mathcal{L}_D$, updating weights to better separate real from fake. The generator, conversely, minimizes $V$ by making $D(G(z))$ large (fooling the discriminator). This adversarial tension drives GAN learning.</span>

---

## <span style="font-size: 16px;">Why Sigmoid</span>

### <span style="font-size: 14px;">Probability Output</span>

<span style="font-size: 14px;">Sigmoid maps the raw logit to a valid probability in $(0, 1)$. This is essential because the value function involves $\log D(x)$ and $\log(1 - D(G(z)))$, both requiring $D(x) \in (0, 1)$ to avoid logarithms of non-positive numbers.</span>

### <span style="font-size: 14px;">Connection to Binary Cross-Entropy</span>

<span style="font-size: 14px;">The canonical pairing in logistic regression is sigmoid + BCE loss. When combined, the gradient with respect to the logit $a$ simplifies to:</span>

$$
\frac{\partial \mathcal{L}}{\partial a} = D(x) - y
$$

<span style="font-size: 14px;">where $y \in \{0, 1\}$ is the label. The gradient is simply prediction minus target, with no $\sigma'(a)$ factor that could cause vanishing gradients in logit space. This clean gradient is why sigmoid + BCE is the standard for binary classification.</span>

### <span style="font-size: 14px;">Log-Likelihood Interpretation</span>

<span style="font-size: 14px;">The value function $V(D, G)$ is the expected log-likelihood of the discriminator under a Bernoulli model: $P(\text{real} | x) = D(x)$ and $P(\text{fake} | x) = 1 - D(x)$. Sigmoid ensures this probabilistic interpretation is valid.</span>

---

## <span style="font-size: 16px;">Paper Context: Goodfellow et al. (2014)</span>

<span style="font-size: 14px;">In "Generative Adversarial Nets," the discriminator is described: "The discriminative model D estimates the probability that a sample came from the training data rather than G." The paper frames GANs through an analogy: the generator is a counterfeiter producing fake currency, the discriminator is police detecting counterfeits, and both improve through competition.</span>

<span style="font-size: 14px;">The paper uses multilayer perceptrons for both G and D. The theoretical analysis assumes D has enough capacity to represent the optimal discriminator for any G. Goodfellow et al. prove that for fixed $G$, the optimal discriminator is:</span>

$$
D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

<span style="font-size: 14px;">where $p_g(x)$ is the generator's induced distribution. The optimal discriminator compares data density to generator density: when $p_{\text{data}}(x) > p_g(x)$, $D^*(x) > 0.5$; when $p_g(x) > p_{\text{data}}(x)$, $D^*(x) < 0.5$. This result underpins the paper's main theorem: the global minimum of $V(D^*, G)$ is achieved if and only if $p_g = p_{\text{data}}$, at which point $V = -\log 4$.</span>

---

## <span style="font-size: 16px;">The Optimal Discriminator</span>

### <span style="font-size: 14px;">Derivation Sketch</span>

<span style="font-size: 14px;">For fixed $G$, rewrite $V$ as an integral:</span>

$$
V(D, G) = \int_x \left[ p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx
$$

<span style="font-size: 14px;">At each point $x$, let $a = p_{\text{data}}(x)$ and $b = p_g(x)$. We maximize $f(D) = a \log D + b \log(1 - D)$:</span>

$$
f'(D) = \frac{a}{D} - \frac{b}{1 - D} = 0 \implies D^* = \frac{a}{a + b} = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

<span style="font-size: 14px;">The second derivative $f''(D) = -a/D^2 - b/(1-D)^2 < 0$ for positive $a, b$, confirming a maximum.</span>

### <span style="font-size: 14px;">At Equilibrium: $D^* = 0.5$ Everywhere</span>

<span style="font-size: 14px;">When the generator perfectly matches the data distribution ($p_g = p_{\text{data}}$):</span>

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_{\text{data}}(x)} = \frac{1}{2}
$$

<span style="font-size: 14px;">At Nash equilibrium, the discriminator outputs 0.5 for every input. It cannot distinguish real from fake because the distributions are identical. This is counterintuitive: perfect GAN training makes the discriminator useless. Its purpose is not to be a good classifier at convergence, but to provide a useful training signal to the generator along the way.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a single-layer discriminator with $d = 3$ and weight vector $W = [0.8, -1.2, 0.5]^T$.</span>

### <span style="font-size: 14px;">Real Sample</span>

<span style="font-size: 14px;">Let $x_{\text{real}} = [2.0, -1.0, 1.5]$ from the training data.</span>

<span style="font-size: 14px;">**Step 1.** Logit: $a = (2.0)(0.8) + (-1.0)(-1.2) + (1.5)(0.5) = 1.6 + 1.2 + 0.75 = 3.55$</span>

<span style="font-size: 14px;">**Step 2.** Sigmoid: $D(x_{\text{real}}) = \frac{1}{1 + e^{-3.55}} = \frac{1}{1.0287} \approx 0.972$</span>

<span style="font-size: 14px;">Output 0.972: very high confidence this is real. Correct.</span>

### <span style="font-size: 14px;">Fake Sample</span>

<span style="font-size: 14px;">Let $x_{\text{fake}} = G(z) = [-0.5, 1.8, -0.3]$ from the generator.</span>

<span style="font-size: 14px;">**Step 1.** Logit: $a = (-0.5)(0.8) + (1.8)(-1.2) + (-0.3)(0.5) = -0.4 - 2.16 - 0.15 = -2.71$</span>

<span style="font-size: 14px;">**Step 2.** Sigmoid: $D(x_{\text{fake}}) = \frac{1}{1 + e^{2.71}} = \frac{1}{16.03} \approx 0.062$</span>

<span style="font-size: 14px;">Output 0.062: high confidence this is fake. Correct.</span>

### <span style="font-size: 14px;">Loss Computation</span>

<span style="font-size: 14px;">For this mini-batch of one real and one fake sample:</span>

$$
\mathcal{L}_D = -\frac{1}{2}\left[\log(0.972) + \log(1 - 0.062)\right] = -\frac{1}{2}\left[-0.0284 + (-0.0640)\right] = 0.0462
$$

<span style="font-size: 14px;">This small loss confirms good performance. A perfect discriminator achieves loss 0. A random discriminator (outputting 0.5 for everything) gets $\log 2 \approx 0.693$.</span>

---

## <span style="font-size: 16px;">Discriminator Evolution</span>

<span style="font-size: 14px;">The discriminator architecture has evolved significantly since the original MLP design.</span>

### <span style="font-size: 14px;">Linear and MLP Discriminators (2014)</span>

<span style="font-size: 14px;">The original paper used MLPs with maxout activations. These worked on simple datasets (MNIST) but struggled with complex images because fully connected layers cannot exploit spatial structure and require prohibitively many parameters for large inputs.</span>

### <span style="font-size: 14px;">CNN Discriminators: DCGAN (2015)</span>

<span style="font-size: 14px;">Radford et al. (2015) replaced fully connected layers with strided convolutions. The discriminator uses conv layers with increasing channels, batch normalization, and LeakyReLU, followed by sigmoid. This leverages spatial locality and parameter sharing, enabling realistic 64x64 image generation.</span>

### <span style="font-size: 14px;">PatchGAN (2016)</span>

<span style="font-size: 14px;">Isola et al. (2016) introduced a discriminator that outputs a grid of probabilities instead of a single scalar. Each element classifies whether a local patch is real or fake. This focuses on high-frequency structure (textures, edges) rather than global composition, producing sharper outputs.</span>

### <span style="font-size: 14px;">Spectral Normalization (2018)</span>

<span style="font-size: 14px;">Miyato et al. (2018) constrained the Lipschitz constant by dividing each weight matrix by its largest singular value. This prevents the discriminator from changing too rapidly, ensuring smoother gradients for the generator and stabilizing training.</span>

### <span style="font-size: 14px;">The Critic in WGAN (2017)</span>

<span style="font-size: 14px;">Arjovsky et al. (2017) replaced the discriminator with a "critic" outputting an unbounded real number (no sigmoid). The critic estimates Wasserstein distance between distributions. Lipschitz continuity is enforced through weight clipping or gradient penalty (WGAN-GP, Gulrajani et al., 2017). This addressed training instability and mode collapse.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Discriminator Too Strong: Gradient Vanishing</span>

<span style="font-size: 14px;">If the discriminator becomes too powerful, it outputs values near 1.0 for all real and near 0.0 for all fake. When $D(G(z)) \approx 0$, the generator's gradient through $\log(1 - D(G(z)))$ vanishes because $\log(1 - D(G(z))) \approx \log(1) = 0$ is already near its maximum. The generator receives no learning signal. This is the most common GAN failure mode.</span>

<span style="font-size: 14px;">Mitigations: train the generator more steps per discriminator step, use label smoothing (targets of 0.9 instead of 1.0 for real), or switch to the non-saturating loss $-\log D(G(z))$ which provides stronger gradients early in training.</span>

### <span style="font-size: 14px;">Discriminator Too Weak: No Signal</span>

<span style="font-size: 14px;">If the discriminator is underpowered (too few parameters, unstable learning rate, insufficient training), it provides random gradients. The generator has no meaningful signal to improve against, producing noisy, incoherent outputs. The discriminator must be strong enough to provide a meaningful landscape but not so strong that the landscape becomes flat.</span>

### <span style="font-size: 14px;">Wrong Sigmoid Application</span>

<span style="font-size: 14px;">A common bug is applying sigmoid twice: once in the forward pass and again in the loss. Many frameworks provide `BCEWithLogitsLoss` which internally applies sigmoid. If the network already has sigmoid and you use `BCEWithLogitsLoss`, the double sigmoid compresses output near 0.5, severely degrading training. Either use `BCELoss` with sigmoid in the network, or `BCEWithLogitsLoss` with raw logits. Never both.</span>

### <span style="font-size: 14px;">Forgetting That D Provides G's Training Signal</span>

<span style="font-size: 14px;">The generator never sees real data directly. Its only information about $p_{\text{data}}$ comes from gradients through the discriminator. If the discriminator has blind spots (regions where it cannot distinguish real from fake), the generator has no incentive to improve there. The discriminator's quality directly bounds the generator's potential.</span>

---