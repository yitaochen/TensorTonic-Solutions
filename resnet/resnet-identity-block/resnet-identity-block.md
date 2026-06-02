# <span style="font-size: 20px;">Identity Block</span>

<span style="font-size: 14px;">The identity block is the simplest residual building block from He et al. (2015), "Deep Residual Learning for Image Recognition." It adds the raw input directly to the output of two linear transformations, with no projection or dimension change on the skip path. This is the block that makes very deep networks trainable by letting gradients flow through an unmodified shortcut connection.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A residual block takes an input $x$, passes it through a stack of layers (the "main path"), and then adds the original $x$ back to the result before a final activation. In an identity block, the skip connection is the identity function: the input passes through unchanged and is added element-wise to the main path's output. No linear projection, no convolution, no learnable parameters on the shortcut. Just $x$ itself.</span>

<span style="font-size: 14px;">For this to work, the input and output dimensions must match exactly. If $x \in \mathbb{R}^d$, then the main path must also produce a vector in $\mathbb{R}^d$. This is why both weight matrices $W_1$ and $W_2$ are square matrices of shape $(d, d)$. The identity block is used within a stage of a ResNet where the spatial dimensions and channel count remain constant. When dimensions need to change, a projection block is used instead.</span>

<span style="font-size: 14px;">In the simplified form used in this problem, the main path consists of two linear layers with a ReLU activation between them. The full computation is: apply $W_1$, apply ReLU, apply $W_2$, add $x$, then apply ReLU. The critical detail is that there is no activation between $W_2$ and the addition. ReLU is applied only after the skip connection has been merged.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Given an input vector $x \in \mathbb{R}^d$ and two weight matrices $W_1, W_2 \in \mathbb{R}^{d \times d}$:</span>

<span style="font-size: 14px;">**Step 1: First linear transformation and activation.**</span>

$$
h = \text{ReLU}(x W_1^T)
$$

<span style="font-size: 14px;">This maps $x$ through $W_1$ and applies ReLU element-wise. The intermediate vector $h \in \mathbb{R}^d$ captures the first layer's learned features, with negative values zeroed out.</span>

<span style="font-size: 14px;">**Step 2: Second linear transformation (no activation).**</span>

$$
F(x) = h W_2^T
$$

<span style="font-size: 14px;">The second linear layer produces $F(x) \in \mathbb{R}^d$. Crucially, there is no ReLU here. The output of $W_2$ is the raw, unactivated result. This is the residual function -- the "difference" the block learns.</span>

<span style="font-size: 14px;">**Step 3: Skip addition and final activation.**</span>

$$
y = \text{ReLU}(F(x) + x)
$$

<span style="font-size: 14px;">The input $x$ is added element-wise to $F(x)$, and then ReLU is applied to the sum. The full expression expanded is:</span>

$$
y = \text{ReLU}(\text{ReLU}(x W_1^T) W_2^T + x)
$$

<span style="font-size: 14px;">The paper writes the block as $y = \text{ReLU}(F(x) + x)$, where $F(x)$ is everything the main path computes. The identity block is defined by the shortcut contributing $x$ with no modification.</span>

* <span style="font-size: 14px;">**$x \in \mathbb{R}^d$:** Input vector. Must have the same dimension as the output.</span>
* <span style="font-size: 14px;">**$W_1 \in \mathbb{R}^{d \times d}$:** Weight matrix for the first linear layer. Square because input and output dimensions match.</span>
* <span style="font-size: 14px;">**$W_2 \in \mathbb{R}^{d \times d}$:** Weight matrix for the second linear layer. Square for the same reason.</span>
* <span style="font-size: 14px;">**$h \in \mathbb{R}^d$:** Hidden activation after the first layer and ReLU.</span>
* <span style="font-size: 14px;">**$F(x) \in \mathbb{R}^d$:** The residual function. This is what the two layers collectively learn.</span>
* <span style="font-size: 14px;">**$y \in \mathbb{R}^d$:** The output of the identity block.</span>

---

## <span style="font-size: 16px;">Why "Identity"</span>

<span style="font-size: 14px;">The name "identity block" refers to the skip connection, not the main path. The shortcut mapping is the identity function $\mathcal{I}(x) = x$. The input passes through with no transformation, no learnable parameters, and no dimensionality change. Compare this to a projection block, where the shortcut applies a linear projection $W_s x$ to match a changed output dimension.</span>

<span style="font-size: 14px;">The identity shortcut introduces zero additional parameters and zero additional computation: the input is simply stored and added. He et al. showed that identity shortcuts are sufficient whenever input and output dimensions match, and that adding unnecessary projections hurts performance.</span>

<span style="font-size: 14px;">In Section 4.1 of the paper, the authors compare three shortcut options: (A) zero-padding for dimension increases with identity otherwise, (B) projection shortcuts only for dimension changes with identity otherwise, and (C) all shortcuts are projections. Option B outperformed A, but C did not meaningfully outperform B. Projecting when dimensions already match adds parameters without improving accuracy. Identity shortcuts are empirically optimal when dimensions match.</span>

---

## <span style="font-size: 16px;">ReLU Placement</span>

<span style="font-size: 14px;">The placement of ReLU activations in the identity block is precise and deliberate. There are exactly two ReLU operations:</span>

<span style="font-size: 14px;">1. **After $W_1$**: The first linear transformation is followed by ReLU. This introduces nonlinearity between the two linear layers. Without it, $x W_1^T W_2^T$ would collapse into a single linear transformation $x (W_1 W_2)^T$, making two layers equivalent to one.</span>

<span style="font-size: 14px;">2. **After the addition $F(x) + x$**: ReLU is applied after the skip connection merges with the main path.</span>

<span style="font-size: 14px;">There is explicitly no ReLU between $W_2$ and the addition. If ReLU were applied after $W_2$ (before addition), the residual $F(x)$ would be non-negative. The block could only add positive values to $x$, never subtract. By keeping $F(x)$ unrestricted, the block can both amplify and suppress features relative to the input.</span>

<span style="font-size: 14px;">The gradient perspective clarifies why this matters. During backpropagation through $y = \text{ReLU}(F(x) + x)$, the gradient with respect to $x$ has two terms: one flowing through $F(x)$ and one flowing directly from the addition. The direct path carries a gradient of 1, multiplied only by the ReLU gate at the output. This unimpeded gradient flow is the core mechanism that lets ResNets train at 100+ layers where plain networks fail.</span>

<span style="font-size: 14px;">This placement was further studied in He et al. (2016), "Identity Mappings in Deep Residual Networks," which found that moving both ReLU and batch normalization before the weight layers improved results in very deep networks. In the original 2015 design used here, the post-addition ReLU is the standard.</span>

---

## <span style="font-size: 16px;">The Residual Function $F(x)$</span>

<span style="font-size: 14px;">He et al.'s central insight is reframing what the layers learn. In a plain network, each block learns a direct mapping $H(x)$. In a residual network, the block instead learns $F(x) = H(x) - x$: the difference between the desired output and the input. The output is then $H(x) = F(x) + x$.</span>

<span style="font-size: 14px;">The paper states: "We explicitly let these layers fit a residual mapping $F(x) := H(x) - x$." This reformulation does not change what the network can represent. Any function $H(x)$ a plain network can learn, a residual network can also learn by setting $F(x) = H(x) - x$. The representational capacity is identical. What changes is the optimization landscape.</span>

<span style="font-size: 14px;">The hypothesis is that learning a residual is easier than learning the full mapping. If the optimal transformation is close to the identity, then $F(x)$ should be close to zero. Pushing a stack of nonlinear layers toward zero is easier than pushing them toward an arbitrary identity-like mapping. At initialization with small weights, $F(x) \approx 0$, so the block starts as an approximate identity. The network then learns small perturbations on top of this baseline.</span>

<span style="font-size: 14px;">In the identity block specifically, $F(x) = \text{ReLU}(x W_1^T) W_2^T$. This is a two-layer network with one hidden nonlinearity. It learns a correction to $x$: positive values in $F(x)$ increase corresponding features; negative values decrease them. The final ReLU then clips any features that end up negative after the addition.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">He et al. (2015) introduced residual learning to solve the degradation problem: empirically, adding more layers to a plain deep network increased training error, not just test error. This was not caused by overfitting but by optimization difficulty. Deeper plain networks were harder to train because gradients either vanished or exploded across many layers, even with batch normalization.</span>

<span style="font-size: 14px;">The authors proposed residual connections as a structural solution. By adding skip connections, they ensured that the gradient could always flow through the identity path. This allowed them to train networks with 152 layers on ImageNet, winning the ILSVRC 2015 classification challenge with a 3.57% top-5 error rate.</span>

<span style="font-size: 14px;">In a ResNet, identity blocks are used within each stage where feature map dimensions stay constant. A typical ResNet-34 has four stages with [3, 4, 6, 3] residual blocks. At stage boundaries, a projection block doubles the channels and halves the spatial resolution. Within each stage, all remaining blocks are identity blocks. In stage 3 of ResNet-34, the first block is a projection block (128 to 256 channels) and the remaining 5 are identity blocks.</span>

<span style="font-size: 14px;">The deeper variants (ResNet-50, ResNet-101, ResNet-152) use "bottleneck" blocks with three layers ($1 \times 1$, $3 \times 3$, $1 \times 1$ convolutions) instead of two. The bottleneck identity block still uses a plain identity shortcut; it just has three layers on the main path. The principle is the same: when dimensions match, skip with identity.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider $d = 3$ with concrete values for $x$, $W_1$, and $W_2$. We trace both the main path and the skip path, then combine them.</span>

<span style="font-size: 14px;">**Input vector:**</span>

$$
x = (1.0, \; -2.0, \; 3.0)
$$

<span style="font-size: 14px;">**Weight matrices:**</span>

$$
W_1 = \begin{pmatrix} 0.5 & -0.1 & 0.3 \\ 0.2 & 0.4 & -0.2 \\ -0.1 & 0.3 & 0.6 \end{pmatrix}, \quad W_2 = \begin{pmatrix} 0.3 & 0.1 & -0.2 \\ -0.1 & 0.5 & 0.2 \\ 0.4 & -0.3 & 0.1 \end{pmatrix}
$$

<span style="font-size: 14px;">**Main path, step 1: Compute $z_1 = x W_1^T$.**</span>

$$
z_{1,1} = 1.0 \times 0.5 + (-2.0) \times (-0.1) + 3.0 \times 0.3 = 0.5 + 0.2 + 0.9 = 1.6
$$

$$
z_{1,2} = 1.0 \times 0.2 + (-2.0) \times 0.4 + 3.0 \times (-0.2) = 0.2 - 0.8 - 0.6 = -1.2
$$

$$
z_{1,3} = 1.0 \times (-0.1) + (-2.0) \times 0.3 + 3.0 \times 0.6 = -0.1 - 0.6 + 1.8 = 1.1
$$

$$
z_1 = (1.6, \; -1.2, \; 1.1)
$$

<span style="font-size: 14px;">**Main path, step 2: Apply ReLU to get $h$.**</span>

$$
h = \text{ReLU}(z_1) = (\max(0, 1.6), \; \max(0, -1.2), \; \max(0, 1.1)) = (1.6, \; 0.0, \; 1.1)
$$

<span style="font-size: 14px;">The second element was negative, so ReLU zeroed it out.</span>

<span style="font-size: 14px;">**Main path, step 3: Compute $F(x) = h W_2^T$ (no ReLU here).**</span>

$$
F(x)_1 = 1.6 \times 0.3 + 0.0 \times 0.1 + 1.1 \times (-0.2) = 0.48 + 0.0 - 0.22 = 0.26
$$

$$
F(x)_2 = 1.6 \times (-0.1) + 0.0 \times 0.5 + 1.1 \times 0.2 = -0.16 + 0.0 + 0.22 = 0.06
$$

$$
F(x)_3 = 1.6 \times 0.4 + 0.0 \times (-0.3) + 1.1 \times 0.1 = 0.64 + 0.0 + 0.11 = 0.75
$$

$$
F(x) = (0.26, \; 0.06, \; 0.75)
$$

<span style="font-size: 14px;">This is the residual: the correction the two layers computed. The values are small relative to $x$, showing the block learns a refinement rather than a complete transformation.</span>

<span style="font-size: 14px;">**Skip path: Identity.**</span>

$$
\text{skip} = x = (1.0, \; -2.0, \; 3.0)
$$

<span style="font-size: 14px;">No computation. The input passes through unchanged.</span>

<span style="font-size: 14px;">**Addition: Merge the two paths.**</span>

$$
F(x) + x = (0.26 + 1.0, \; 0.06 + (-2.0), \; 0.75 + 3.0) = (1.26, \; -1.94, \; 3.75)
$$

<span style="font-size: 14px;">The first and third features increased. The second remains negative because the small positive residual (0.06) could not overcome the large negative input (-2.0).</span>

<span style="font-size: 14px;">**Final ReLU.**</span>

$$
y = \text{ReLU}(F(x) + x) = (\max(0, 1.26), \; \max(0, -1.94), \; \max(0, 3.75)) = (1.26, \; 0.0, \; 3.75)
$$

<span style="font-size: 14px;">The second feature is clipped to zero. The output $y = (1.26, \; 0.0, \; 3.75)$ preserves the structure of the input $x = (1.0, \; -2.0, \; 3.0)$ but with refined magnitudes. The block did not overwrite $x$; it adjusted it.</span>

---

## <span style="font-size: 16px;">Connection to Deeper Architectures</span>

<span style="font-size: 14px;">Identity blocks form the backbone of every ResNet. They are stacked repeatedly within each stage to build depth without changing dimensions. Projection blocks serve as transitions between stages, handling dimension changes. Once the transition is made, identity blocks take over.</span>

<span style="font-size: 14px;">In ResNet-50, there are four stages with [3, 4, 6, 3] bottleneck blocks. In each stage, the first block is a projection block and every subsequent block is an identity block. Out of 16 total blocks, 12 are identity blocks and only 4 are projection blocks. ResNet-152 uses [3, 8, 36, 3] blocks per stage, meaning 46 of 50 total blocks are identity blocks.</span>

<span style="font-size: 14px;">The residual framework was later adopted across deep learning: DenseNet extended skip connections to connect every layer to every other layer, Transformers use residual connections around self-attention and feed-forward sublayers, and U-Net uses skip connections across encoder-decoder boundaries. The identity block from He et al. (2015) demonstrated learning through addition rather than replacement.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Applying ReLU after $W_2$ before the addition.** Writing $y = \text{ReLU}(\text{ReLU}(h W_2^T) + x)$ instead of $y = \text{ReLU}(h W_2^T + x)$ is wrong. If $F(x)$ passes through ReLU before addition, the residual is constrained to be non-negative. The block can only increase features, never decrease them.</span>
* <span style="font-size: 14px;">**Using non-square weight matrices.** If $W_1$ or $W_2$ are not $(d, d)$, the output dimension of the main path will differ from $x$, and the element-wise addition $F(x) + x$ will fail. The identity shortcut requires input and output dimensions to match exactly.</span>
* <span style="font-size: 14px;">**Forgetting the skip connection entirely.** Without the addition of $x$, the block becomes a plain two-layer network: $y = \text{ReLU}(\text{ReLU}(x W_1^T) W_2^T)$. This loses the gradient highway and the residual learning property, and training deep stacks suffers from degradation.</span>
* <span style="font-size: 14px;">**Applying a projection when not needed.** Writing $y = \text{ReLU}(F(x) + W_s x)$ with a learnable $W_s$ when dimensions already match adds unnecessary parameters. He et al. showed that identity shortcuts match or outperform projection shortcuts when dimensions agree.</span>
* <span style="font-size: 14px;">**Wrong order of operations in the main path.** The main path must be: linear, ReLU, linear. Swapping to linear, linear, ReLU collapses two consecutive linear layers into one ($x W_1^T W_2^T = x (W_1 W_2)^T$), wasting parameters and reducing the block to a single effective layer.</span>
* <span style="font-size: 14px;">**Confusing identity blocks with bottleneck blocks.** The two-layer identity block (ResNet-18/34) has two $(d, d)$ weight matrices. The three-layer bottleneck block (ResNet-50/101/152) uses $1 \times 1$ convolutions to reduce and restore channels. Both use identity shortcuts, but the main path differs. This problem uses the two-layer variant.</span>

---