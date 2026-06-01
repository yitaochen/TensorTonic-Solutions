# <span style="font-size: 20px;">Mode Collapse Detection</span>

<span style="font-size: 14px;">Mode collapse is one of the most common failure modes in GANs. It occurs when the generator produces limited varieties of samples, mapping different noise vectors to nearly identical outputs. Instead of learning the full data distribution, the generator "collapses" onto one or a few modes.</span>

<span style="font-size: 14px;">This problem detects mode collapse through the diversity score: the mean per-feature standard deviation of generated samples. When this score falls below a threshold, the generator is flagged as collapsed.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Mode collapse detection measures output diversity to identify generator failure. A healthy generator produces varied outputs when given different noise inputs. Sample 100 random noise vectors $z_1, \ldots, z_{100}$ and the resulting $G(z_1), \ldots, G(z_{100})$ should span the data distribution. In a collapsed generator, these outputs cluster tightly around one or a few points regardless of the input noise.</span>

<span style="font-size: 14px;">The detection mechanism computes a diversity score from a generated batch. Each feature dimension is examined independently for variation across the batch. If the generator has collapsed, every feature has near-zero variance. The diversity score aggregates per-feature variation into a single number compared against a threshold to produce a boolean is_collapsed signal.</span>

<span style="font-size: 14px;">This is a diagnostic tool, not a training objective. It tells you whether the generator has failed but does not fix the problem. Think of it as a smoke detector for GANs.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Given $N$ generated samples, each with $D$ features, arranged as $X \in \mathbb{R}^{N \times D}$:</span>

<span style="font-size: 14px;">**Per-feature mean:**</span>

$$
\mu_j = \frac{1}{N} \sum_{i=1}^{N} X_{i,j}
$$

<span style="font-size: 14px;">**Per-feature standard deviation:**</span>

$$
\sigma_j = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (X_{i,j} - \mu_j)^2}
$$

<span style="font-size: 14px;">This is the population standard deviation (dividing by $N$, not $N-1$).</span>

<span style="font-size: 14px;">**Diversity score (mean of per-feature stds):**</span>

$$
\text{diversity\_score} = \frac{1}{D} \sum_{j=1}^{D} \sigma_j
$$

<span style="font-size: 14px;">**Collapse detection:**</span>

$$
\text{is\_collapsed} = \begin{cases} \text{True} & \text{if } \text{diversity\_score} < \text{threshold} \\ \text{False} & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">The threshold depends on the data domain. For normalized data (features in $[0, 1]$), a threshold around 0.1 is common. For standardized data, thresholds around 0.5 may be appropriate.</span>

---

## <span style="font-size: 16px;">What Mode Collapse Looks Like</span>

<span style="font-size: 14px;">**All outputs nearly identical despite different noise inputs.** The generator learns to ignore $z$ and maps everything to roughly the same output: $G(z_1) \approx G(z_2) \approx \cdots \approx G(z_N)$ for all noise vectors. The generator becomes essentially a constant function. If generating images, every image looks the same. If generating tabular data, every row has nearly identical values.</span>

<span style="font-size: 14px;">**The generator ignores $z$.** A well-functioning generator uses noise as a source of randomness, mapping different latent regions to different data regions. A collapsed generator develops weights that cancel out $z$'s influence. The hidden layers produce similar activations regardless of input, and the information bottleneck discards diversity rather than carrying it through.</span>

<span style="font-size: 14px;">**Everything maps to one or a few modes.** Real distributions are multimodal. Consider handwritten digits with 10 modes (digits 0-9). A fully collapsed generator might produce only "1" because that single mode temporarily fools the discriminator. A partially collapsed generator might produce only "1" and "7". In both cases the generator fails to capture the full distribution.</span>

---

## <span style="font-size: 16px;">Why Mode Collapse Happens</span>

<span style="font-size: 14px;">Mode collapse arises from the adversarial training dynamics between generator and discriminator:</span>

* <span style="font-size: 14px;">**Generator finds one output that fools D.** If the generator discovers a single point that consistently receives high "real" scores, it is incentivized to map all noise vectors there. From G's perspective, this achieves low loss without modeling the full distribution.</span>
* <span style="font-size: 14px;">**Exploitation over exploration.** The generator can explore the data space to cover all modes (risky, slow) or exploit a known good mode (safe, fast). Gradient descent favors exploitation because gradients point toward the nearest "real-looking" region, not toward uncovered modes.</span>
* <span style="font-size: 14px;">**Discriminator cannot enforce mode coverage.** D evaluates individual samples, not the distribution. It can judge whether one sample looks real but has no mechanism to say "you are repeating yourself." Per-sample feedback cannot express distributional constraints.</span>
* <span style="font-size: 14px;">**Cycling rather than converging.** The minimax game can cycle: G collapses to mode A, D rejects A, G switches to mode B, D rejects B, G switches back. The generator covers one mode at a time, rotating rather than covering all modes simultaneously.</span>

---

## <span style="font-size: 16px;">Types of Mode Collapse</span>

* <span style="font-size: 14px;">**Complete collapse (single output).** The generator maps every noise vector to a single point: $G(z) = c$ for all $z$. Diversity score is near zero. Easy to detect but rare in practice because even a weak discriminator can reject a single repeated sample. Usually indicates a training bug.</span>
* <span style="font-size: 14px;">**Partial collapse (few clusters).** The generator produces samples from $k$ modes where $k$ is much smaller than the true count. For MNIST, it might only produce digits 1, 4, and 7. Diversity score is moderate: higher than complete collapse due to inter-cluster variation but lower than a healthy generator. This is the most common form.</span>
* <span style="font-size: 14px;">**Intra-class collapse (limited variation within each mode).** The generator covers all major modes but with minimal variation within each. For MNIST, all 10 digits appear but each looks nearly identical across samples: same stroke width, slant, position. The diversity score may look acceptable because inter-mode variation masks missing intra-mode variation. Hardest form to detect with simple metrics.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Goodfellow et al. (2014) introduced GANs with the minimax formulation:</span>

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

<span style="font-size: 14px;">The paper proves this objective has a global optimum where $p_G = p_{\text{data}}$. However, the proof assumes infinite capacity and perfect optimization. In practice, a finite neural network may lack capacity to represent $p_{\text{data}}$, and gradient-based optimization may not find the global optimum. Mode collapse is a direct consequence of this gap between theory and practice.</span>

<span style="font-size: 14px;">The paper acknowledges that training requires careful balancing of G and D updates. When D is too strong, G receives vanishing gradients. When D is too weak, G receives noisy gradients. Both scenarios can trigger or worsen mode collapse.</span>

<span style="font-size: 14px;">Subsequent work addressing mode collapse includes minibatch discrimination (Salimans et al., 2016), unrolled GANs (Metz et al., 2017), Wasserstein GANs (Arjovsky et al., 2017), and spectral normalization (Miyato et al., 2018). Each attacks the problem from a different angle but none eliminates it entirely.</span>

---

## <span style="font-size: 16px;">The Standard Deviation Metric</span>

<span style="font-size: 14px;">Standard deviation is a proxy for diversity because it directly measures spread. High std along a feature means that feature takes a wide range of values across the batch, indicating diversity. Near-zero std means all samples share nearly the same value, indicating collapse.</span>

<span style="font-size: 14px;">**Why per feature then average?** Different features may have different natural scales. A single global std would be dominated by high-magnitude features. Computing per-feature stds and averaging gives each feature equal weight in the diversity score.</span>

<span style="font-size: 14px;">**What low std means geometrically.** Each generated sample is a point in $\mathbb{R}^D$. A healthy generator scatters points throughout the region occupied by real data. Low std along all features means all points cluster in a tiny hyperrectangle. In the extreme ($\sigma_j = 0$ for all $j$), all points occupy a single location. Mode collapse shrinks the point cloud from a region to a point or a few isolated clusters.</span>

<span style="font-size: 14px;">**Averaging vs. minimum.** The mean of per-feature stds provides a smoother signal than $\min_j \sigma_j$. The minimum is more conservative (flags collapse if any single feature collapses) but more sensitive to noise. The mean is less prone to false positives from individual noisy features.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Generator with $D = 3$ features, batch of $N = 5$ samples.</span>

<span style="font-size: 14px;">**Collapsed generator output:**</span>

| Sample | Feature 1 | Feature 2 | Feature 3 |
|--------|-----------|-----------|-----------|
| 1      | 0.51      | 0.82      | 0.34      |
| 2      | 0.50      | 0.81      | 0.35      |
| 3      | 0.52      | 0.83      | 0.33      |
| 4      | 0.50      | 0.82      | 0.34      |
| 5      | 0.51      | 0.82      | 0.34      |

<span style="font-size: 14px;">Per-feature means: $\mu_1 = 0.508$, $\mu_2 = 0.820$, $\mu_3 = 0.340$.</span>

<span style="font-size: 14px;">Feature 1 deviations: $[0.002, -0.008, 0.012, -0.008, 0.002]$. Squared: $[0.000004, 0.000064, 0.000144, 0.000064, 0.000004]$. Mean: $0.000056$. $\sigma_1 = \sqrt{0.000056} \approx 0.0075$.</span>

<span style="font-size: 14px;">Feature 2: deviations $[0.00, -0.01, 0.01, 0.00, 0.00]$. Mean of squares: $0.00004$. $\sigma_2 \approx 0.0063$.</span>

<span style="font-size: 14px;">Feature 3: deviations $[0.00, 0.01, -0.01, 0.00, 0.00]$. Mean of squares: $0.00004$. $\sigma_3 \approx 0.0063$.</span>

<span style="font-size: 14px;">**Diversity score:** $(0.0075 + 0.0063 + 0.0063) / 3 \approx 0.0067$.</span>

<span style="font-size: 14px;">With threshold $= 0.1$: $0.0067 < 0.1 \Rightarrow \text{is\_collapsed} = \text{True}$. All five samples are nearly identical.</span>

<span style="font-size: 14px;">**Healthy generator output:**</span>

| Sample | Feature 1 | Feature 2 | Feature 3 |
|--------|-----------|-----------|-----------|
| 1      | 0.23      | 0.91      | 0.67      |
| 2      | 0.78      | 0.15      | 0.42      |
| 3      | 0.45      | 0.63      | 0.89      |
| 4      | 0.12      | 0.47      | 0.31      |
| 5      | 0.89      | 0.72      | 0.55      |

<span style="font-size: 14px;">Per-feature stds: $\sigma_1 \approx 0.299$, $\sigma_2 \approx 0.253$, $\sigma_3 \approx 0.196$.</span>

<span style="font-size: 14px;">**Diversity score:** $(0.299 + 0.253 + 0.196) / 3 \approx 0.249$. With threshold $= 0.1$: $0.249 > 0.1 \Rightarrow \text{is\_collapsed} = \text{False}$. The generator produces varied samples.</span>

---

## <span style="font-size: 16px;">Mitigation Strategies</span>

<span style="font-size: 14px;">Once mode collapse is detected, several techniques can address it:</span>

* <span style="font-size: 14px;">**Minibatch discrimination (Salimans et al., 2016).** The discriminator receives pairwise similarity information across the minibatch. If G produces identical outputs, D sees high similarity and penalizes it. This addresses the limitation that the standard D evaluates samples independently.</span>
* <span style="font-size: 14px;">**Unrolled GANs (Metz et al., 2017).** G's loss is computed by unrolling several D update steps into the future. G optimizes against a future discriminator that has adapted to its outputs, discouraging exploitation of temporary weaknesses at a single mode.</span>
* <span style="font-size: 14px;">**Wasserstein distance (Arjovsky et al., 2017).** Replaces the JS divergence with the Earth Mover's distance, providing meaningful gradients even when generator and real distributions have non-overlapping support. The smoother loss landscape reduces mode collapse.</span>
* <span style="font-size: 14px;">**Spectral normalization (Miyato et al., 2018).** Constrains the Lipschitz constant of D by normalizing spectral norms of weight matrices. Prevents D from becoming too sharp, stabilizing training and reducing cycling behavior.</span>
* <span style="font-size: 14px;">**Diversity-promoting losses.** Add explicit diversity terms to G's loss, e.g., a penalty proportional to $-\text{diversity\_score}$. Mode regularization penalizes small pairwise distances between generated samples, treating diversity as a first-class objective.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Threshold too high gives false positives.**</span>

<span style="font-size: 14px;">If real data naturally has low diversity in some features (binary features, categorical features with few values), the diversity score of a perfect generator would also be low. The threshold must be calibrated relative to the real data's diversity, not set arbitrarily.</span>

<span style="font-size: 14px;">**2. Threshold too low misses partial collapse.**</span>

<span style="font-size: 14px;">A threshold of 0.01 catches complete collapse but misses a generator stuck on a handful of modes. Inter-mode variation can inflate per-feature stds, making a partially collapsed generator look healthy.</span>

<span style="font-size: 14px;">**3. Std of 0 does not always mean collapse.**</span>

<span style="font-size: 14px;">If the real data has a constant feature (e.g., a bias column always equal to 1.0), then $\sigma_j = 0$ for that feature is correct behavior. A perfect generator should reproduce constant features exactly. Interpreting zero std as collapse without checking the real distribution leads to false alarms.</span>

<span style="font-size: 14px;">**4. Confusing low quality with mode collapse.**</span>

<span style="font-size: 14px;">A generator can produce diverse but unrealistic samples (high diversity, low quality) or one very realistic sample repeatedly (low diversity, high quality). The diversity score measures only spread, not realism. A high score does not mean the GAN works well.</span>

<span style="font-size: 14px;">**5. Batch size too small for reliable std.**</span>

<span style="font-size: 14px;">With $N = 5$ samples, the std estimate has high variance. A generator might appear collapsed because a small batch happened to produce similar outputs by chance. Use batches of at least 100-500 samples for reliable detection. The metric is meant for substantial batches, not training minibatches of size 8 or 16.</span>

---