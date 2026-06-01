# <span style="font-size: 20px;">Complete GAN System</span>

<span style="font-size: 14px;">A Generative Adversarial Network is two neural networks locked in a minimax game: a Generator that fabricates data and a Discriminator that tries to tell real from fake. Neither network makes sense in isolation. The Generator has no loss function without a Discriminator to fool, and the Discriminator has no adversary without a Generator producing counterfeits. This problem asks you to build both networks as a single class and wire them together into a training step.</span>

<span style="font-size: 14px;">This is the simplest possible GAN: one linear layer per network, no hidden layers, no convolutional filters. Yet it contains every structural element of the original Goodfellow et al. (2014) framework: noise sampling, generator forward pass, discriminator forward pass on both real and fake data, and the two adversarial losses.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The Complete GAN System encapsulates the full adversarial framework as a single object. It holds two weight matrices ($W_G$ and $W_D$), exposes a generator that maps noise to fake data, a discriminator that maps any data to a probability of being real, and a train_step that wires everything together to compute both losses in one forward pass.</span>

<span style="font-size: 14px;">The Generator maps noise $z$ to data space via tanh: $G(z) = \tanh(z \cdot W_G)$. Tanh bounds outputs to $[-1, 1]$, matching the typical range of normalized real data. The Discriminator produces a probability through sigmoid: $D(x) = \sigma(x \cdot W_D)$. The sigmoid squashes the output to $(0, 1)$, representing the probability that $x$ is real.</span>

<span style="font-size: 14px;">The adversarial game is: D wants to output 1 for real data and 0 for fake data. G wants D to output 1 for fake data. They cannot both win. This tension drives generative learning.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Noise sampling:**</span>

$$
z \sim \mathcal{N}(0, I), \quad z \in \mathbb{R}^{n \times \texttt{noise\_dim}}
$$

<span style="font-size: 14px;">**Generator forward pass:**</span>

$$
\hat{x} = \tanh(z \cdot W_G), \quad W_G \in \mathbb{R}^{\texttt{noise\_dim} \times \texttt{data\_dim}}
$$

<span style="font-size: 14px;">**Discriminator forward pass (on any input $x$):**</span>

$$
p = \sigma(x \cdot W_D) = \frac{1}{1 + e^{-x \cdot W_D}}, \quad W_D \in \mathbb{R}^{\texttt{data\_dim} \times 1}
$$

<span style="font-size: 14px;">**Discriminator loss:**</span>

$$
\mathcal{L}_D = -\frac{1}{m}\sum_{i=1}^{m}\bigl[\log D(x_{\text{real}}^{(i)}) + \log(1 - D(G(z^{(i)})))\bigr]
$$

<span style="font-size: 14px;">**Generator loss (non-saturating form):**</span>

$$
\mathcal{L}_G = -\frac{1}{m}\sum_{i=1}^{m}\log D(G(z^{(i)}))
$$

<span style="font-size: 14px;">**Weight initialization:**</span>

$$
W_G \sim \mathcal{N}(0, 0.01), \quad W_D \sim \mathcal{N}(0, 0.01)
$$

---

## <span style="font-size: 16px;">The Two Networks as One System</span>

<span style="font-size: 14px;">G and D are co-dependent. The Generator's loss is defined entirely in terms of the Discriminator's output on fake data: $\mathcal{L}_G = -\mathbb{E}[\log D(G(z))]$. Without D, this expression is meaningless. Conversely, the Discriminator's loss requires fake samples to contrast against real ones: $\mathcal{L}_D = -\mathbb{E}[\log D(x_{\text{real}})] - \mathbb{E}[\log(1 - D(G(z)))]$. Without G, the second term vanishes and D never learns to reject fakes.</span>

<span style="font-size: 14px;">This co-dependence is why the class holds both $W_G$ and $W_D$. They are separate weight matrices with different shapes and different gradient flows, but they must live in the same system because each network's training signal depends on the other's current parameters.</span>

<span style="font-size: 14px;">The adversarial dynamic creates a moving target. As G improves, D must work harder to distinguish fakes. As D improves, G must produce better fakes. Ideally, this arms race converges to an equilibrium where G produces perfect samples and D outputs 0.5 everywhere. In practice, convergence is fragile and depends on learning rates, architecture balance, and training schedule.</span>

---

## <span style="font-size: 16px;">The Forward Pipeline</span>

<span style="font-size: 14px;">The data flow during a training step has four stages:</span>

* <span style="font-size: 14px;">**Stage 1 -- Sample noise.** Draw $z \in \mathbb{R}^{m \times \texttt{noise\_dim}}$ from the standard normal distribution. This is G's raw material.</span>
* <span style="font-size: 14px;">**Stage 2 -- Generator forward.** Compute $\hat{x} = \tanh(z \cdot W_G)$, producing fake samples in $\mathbb{R}^{m \times \texttt{data\_dim}}$.</span>
* <span style="font-size: 14px;">**Stage 3 -- Discriminator on real data.** Compute $p_{\text{real}} = \sigma(x_{\text{real}} \cdot W_D)$. D should push these toward 1.</span>
* <span style="font-size: 14px;">**Stage 4 -- Discriminator on fake data.** Compute $p_{\text{fake}} = \sigma(\hat{x} \cdot W_D)$. D should push these toward 0; G should push them toward 1.</span>

<span style="font-size: 14px;">D uses the same weights $W_D$ in stages 3 and 4. The only difference is the input: real data vs. G's output. This is what makes it adversarial: the same classifier evaluates both sources, and its responses define both losses.</span>

<span style="font-size: 14px;">The forward pipeline is a DAG. G's output feeds into D, but D's output does not feed back into G during the forward pass. The adversarial connection happens through the loss: G's loss is a function of D's output on G's samples, so gradients flow backward through D into G during backpropagation.</span>

---

## <span style="font-size: 16px;">Weight Initialization</span>

<span style="font-size: 14px;">Both weight matrices are drawn from $\mathcal{N}(0, 0.01)$. This small standard deviation is critical:</span>

<span style="font-size: 14px;">**Too large (e.g., $\sigma = 1.0$):** The generator's pre-activation $z \cdot W_G$ will have large magnitude. After tanh, outputs saturate near $\pm 1$ and gradients vanish. For the discriminator, large logits push sigmoid to 0 or 1, where $\log(0) = -\infty$ and learning collapses immediately.</span>

<span style="font-size: 14px;">**Too small (e.g., $\sigma = 10^{-6}$):** Both networks are effectively identity maps. G outputs near-zero values regardless of input. D outputs $\sigma(0) \approx 0.5$ for everything. Gradients exist but are tiny, and training takes impractically long to escape this flat region.</span>

<span style="font-size: 14px;">**The sweet spot ($\sigma = 0.01$):** Pre-activations are small enough that tanh and sigmoid operate in their linear regions (where gradients flow freely) but large enough that the networks produce distinguishable outputs.</span>

<span style="font-size: 14px;">In this implementation, the class receives $W_G$ and $W_D$ as constructor arguments rather than sampling them internally. This makes the system deterministic and testable.</span>

---

## <span style="font-size: 16px;">The Train Step</span>

<span style="font-size: 14px;">The `train_step` method performs one forward pass through the entire system and returns both losses:</span>

<span style="font-size: 14px;">**1. Generate fake data.** Given noise $z$, compute $\hat{x} = \tanh(z \cdot W_G)$.</span>

<span style="font-size: 14px;">**2. Discriminate real data.** Compute $p_{\text{real}} = \sigma(x_{\text{real}} \cdot W_D)$ and clip to $[\epsilon, 1 - \epsilon]$ where $\epsilon = 10^{-8}$.</span>

<span style="font-size: 14px;">**3. Discriminate fake data.** Compute $p_{\text{fake}} = \sigma(\hat{x} \cdot W_D)$ and clip identically.</span>

<span style="font-size: 14px;">**4. Compute D loss.** $\mathcal{L}_D = -\text{mean}[\log(p_{\text{real}}) + \log(1 - p_{\text{fake}})]$. D wants $p_{\text{real}} \to 1$ and $p_{\text{fake}} \to 0$.</span>

<span style="font-size: 14px;">**5. Compute G loss.** $\mathcal{L}_G = -\text{mean}[\log(p_{\text{fake}})]$. G wants $p_{\text{fake}} \to 1$.</span>

<span style="font-size: 14px;">This implementation computes only the forward-pass losses without backpropagation or weight updates. It is a structural demonstration: you wire the components correctly, compute the right losses, and verify the numbers. In a full training loop, you would compute gradients of $\mathcal{L}_D$ w.r.t. $W_D$ (updating D), then gradients of $\mathcal{L}_G$ w.r.t. $W_G$ (updating G).</span>

<span style="font-size: 14px;">The non-saturating G loss ($-\log D(G(z))$) differs from the minimax G loss ($\log(1 - D(G(z)))$). Early in training when D easily rejects fakes, $D(G(z)) \approx 0$. The minimax form gives $\log(1 - 0) \approx 0$: no gradient for G. The non-saturating form gives $-\log(0) \to \infty$: strong gradient signal pushing G to improve. This is the form used in practice and in this implementation.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Goodfellow et al. (2014) introduced GANs in "Generative Adversarial Nets." The paper presents the framework as a minimax game:</span>

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

<span style="font-size: 14px;">Algorithm 1 specifies the training procedure: for each iteration, first update D by ascending its stochastic gradient for $k$ steps, then update G by descending its stochastic gradient for one step. The paper proves that if both networks have enough capacity and D is trained to optimality at each step, then $p_G$ converges to $p_{\text{data}}$. This is a Nash equilibrium: neither player can unilaterally improve.</span>

<span style="font-size: 14px;">The theoretical guarantee relies on D being optimal at every step, which never holds in practice. Real training alternates single gradient steps for D and G, creating a dynamic system that can oscillate, diverge, or mode-collapse. Despite this gap between theory and practice, the basic framework has proven remarkably robust and spawned an entire family of generative models.</span>

<span style="font-size: 14px;">The paper also notes the non-saturating heuristic: instead of G minimizing $\log(1 - D(G(z)))$, G maximizes $\log D(G(z))$. This provides stronger gradients early in training and is universally adopted. This implementation uses the non-saturating form.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace through a concrete GAN with $\texttt{noise\_dim} = 3$, $\texttt{data\_dim} = 2$.</span>

<span style="font-size: 14px;">**Weights:**</span>

$$
W_G = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \\ -0.1 & 0.5 \end{bmatrix} \in \mathbb{R}^{3 \times 2}, \quad W_D = \begin{bmatrix} 0.5 \\ -0.2 \end{bmatrix} \in \mathbb{R}^{2 \times 1}
$$

<span style="font-size: 14px;">**Step 1 -- Sample noise.** $z = [0.5,\; -0.5,\; 1.0]$ (one sample, $\texttt{noise\_dim} = 3$).</span>

<span style="font-size: 14px;">**Step 2 -- Generator forward.** Pre-activation: $z \cdot W_G = [0.5(0.3) + (-0.5)(0.2) + 1.0(-0.1),\;\; 0.5(-0.1) + (-0.5)(0.4) + 1.0(0.5)] = [-0.05,\;\; 0.25]$. Apply tanh: $\hat{x} = [\tanh(-0.05),\;\; \tanh(0.25)] = [-0.0500,\;\; 0.2449]$.</span>

<span style="font-size: 14px;">**Step 3 -- Discriminator on fake data.** $\hat{x} \cdot W_D = (-0.0500)(0.5) + (0.2449)(-0.2) = -0.0250 - 0.0490 = -0.0740$. $p_{\text{fake}} = \sigma(-0.0740) = 1/(1 + e^{0.0740}) \approx 0.4815$.</span>

<span style="font-size: 14px;">**Step 4 -- Discriminator on real data.** $x_{\text{real}} = [0.8,\; -0.3]$. $x_{\text{real}} \cdot W_D = 0.8(0.5) + (-0.3)(-0.2) = 0.40 + 0.06 = 0.46$. $p_{\text{real}} = \sigma(0.46) \approx 0.6130$.</span>

<span style="font-size: 14px;">**Step 5 -- Compute losses.**</span>

$$
\mathcal{L}_D = -[\log(0.6130) + \log(1 - 0.4815)] = -[-0.4894 + (-0.6568)] = 1.1462
$$

$$
\mathcal{L}_G = -\log(0.4815) = 0.7308
$$

<span style="font-size: 14px;">**Interpretation:** D is only 61% confident the real sample is real and assigns 48% probability to the fake being real. G would prefer $p_{\text{fake}} \to 1$, pushing $\mathcal{L}_G \to 0$. At convergence, $p_{\text{fake}} = p_{\text{real}} = 0.5$, giving $\mathcal{L}_D = -2\log(0.5) = 1.3863$ and $\mathcal{L}_G = -\log(0.5) = 0.6931$.</span>

---

## <span style="font-size: 16px;">The GAN Ecosystem</span>

<span style="font-size: 14px;">The basic framework from Goodfellow et al. (2014) has been extended dramatically:</span>

* <span style="font-size: 14px;">**DCGAN (Radford et al., 2015):** Replaced fully-connected layers with convolutional layers. Established guidelines: batch normalization, ReLU in G, LeakyReLU in D, no fully-connected hidden layers.</span>
* <span style="font-size: 14px;">**WGAN (Arjovsky et al., 2017):** Replaced JS divergence with Wasserstein distance. Smoother gradients and a loss that correlates with sample quality. The discriminator (now "critic") outputs an unbounded score, and weights are clipped for a Lipschitz constraint.</span>
* <span style="font-size: 14px;">**WGAN-GP (Gulrajani et al., 2017):** Replaced weight clipping with a gradient penalty, solving capacity underuse and gradient problems caused by hard clipping.</span>
* <span style="font-size: 14px;">**StyleGAN (Karras et al., 2019):** Introduced a mapping network from noise to style space, and adaptive instance normalization (AdaIN) at each layer. Produces photorealistic faces with disentangled control over pose, hair, and age.</span>
* <span style="font-size: 14px;">**BigGAN (Brock et al., 2019):** Scaled GANs to ImageNet resolution with class-conditional generation, orthogonal regularization, and the truncation trick.</span>

<span style="font-size: 14px;">Despite these advances, every variant shares the same core: a generator that transforms noise, a discriminator that classifies, and an adversarial loss that couples them.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Sharing $W_G$ and $W_D$ or using a single weight matrix.**</span>

<span style="font-size: 14px;">$W_G$ has shape $(\texttt{noise\_dim}, \texttt{data\_dim})$ and $W_D$ has shape $(\texttt{data\_dim}, 1)$. They serve different functions and have different shapes. Using one matrix for both means updating G also changes D, destroying the adversarial dynamic.</span>

<span style="font-size: 14px;">**2. Wrong train_step order.**</span>

<span style="font-size: 14px;">The forward pass must run G first (to produce fakes), then D on both real and fake. In full training, the convention is: update D first, then G. Reversing this means G optimizes against a stale D.</span>

<span style="font-size: 14px;">**3. Forgetting to clip probabilities before taking log.**</span>

<span style="font-size: 14px;">If $D(x) = 0$ exactly, $\log(D(x)) = -\infty$ and the loss becomes NaN. Clipping to $[\epsilon, 1-\epsilon]$ with $\epsilon = 10^{-8}$ prevents this without materially changing the loss values.</span>

<span style="font-size: 14px;">**4. Wrong initialization scale.**</span>

<span style="font-size: 14px;">Using $\mathcal{N}(0, 1)$ instead of $\mathcal{N}(0, 0.01)$ causes immediate saturation in both tanh and sigmoid. Training never starts because gradients vanish at the flat regions of both activation functions.</span>

<span style="font-size: 14px;">**5. Confusing G's output with D's output.**</span>

<span style="font-size: 14px;">G outputs data-shaped tensors of shape $(\texttt{batch}, \texttt{data\_dim})$ in $[-1, 1]$. D outputs probabilities of shape $(\texttt{batch}, 1)$ in $(0, 1)$. Treating G's multi-dimensional output as a probability or feeding D's scalar back into the wrong stage is a common shape error.</span>

<span style="font-size: 14px;">**6. Using the wrong G loss form.**</span>

<span style="font-size: 14px;">The minimax form $\log(1 - D(G(z)))$ and the non-saturating form $-\log(D(G(z)))$ are not equivalent. This implementation uses the non-saturating form. The minimax form produces vanishing gradients when D is strong, making G unable to learn.</span>