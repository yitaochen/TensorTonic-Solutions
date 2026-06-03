# <span style="font-size: 20px;">Convolutional Block with Projection Shortcut</span>

<span style="font-size: 14px;">The convolutional block (also called the projection shortcut block) is the variant of the residual block from He et al. (2015) used whenever the input and output dimensions differ. Unlike the identity block, which passes the input through unchanged on the skip path, the convolutional block applies a learned projection matrix $W_s$ to the shortcut so that the addition $F(x) + W_s x$ is dimensionally valid. This block appears at every stage transition in ResNet architectures, where the number of channels increases.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A residual block computes $y = F(x) + \text{shortcut}(x)$, where $F(x)$ is the main path through stacked layers and the shortcut provides a direct connection from input to output. In the identity block, the shortcut is simply $x$ itself, which works only when input and output have the same dimensionality. The convolutional block handles the case where they do not match.</span>

<span style="font-size: 14px;">When the channel count changes between stages, the raw input $x$ cannot be added directly to $F(x)$ because their shapes differ. The convolutional block solves this by placing a learned linear projection $W_s$ on the shortcut path. The projection transforms $x$ into the same dimensional space as the main path output, making the element-wise addition possible. The output becomes $y = F(x) + W_s \cdot x$.</span>

<span style="font-size: 14px;">In the original paper, He et al. describe this as "option B": using a $1 \times 1$ convolution on the shortcut to match dimensions. In a simplified fully-connected setting, the projection is a matrix multiply that maps from the input dimension to the output dimension. The key principle is the same: a learned linear transformation aligns the shortcut to the main path.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x \in \mathbb{R}^{d_{in}}$ be the input vector. The main path applies two linear layers with ReLU activations, and the shortcut path applies one linear projection.</span>

<span style="font-size: 14px;">**Main path, layer 1.** Apply the first weight matrix and ReLU:</span>

$$
h = \text{ReLU}(x \cdot W_1)
$$

<span style="font-size: 14px;">Here $W_1 \in \mathbb{R}^{d_{in} \times d_{out}}$ maps from input dimension to output dimension. ReLU introduces nonlinearity. After this step, $h \in \mathbb{R}^{d_{out}}$.</span>

<span style="font-size: 14px;">**Main path, layer 2.** Apply the second weight matrix and ReLU:</span>

$$
z = \text{ReLU}(h \cdot W_2)
$$

<span style="font-size: 14px;">Here $W_2 \in \mathbb{R}^{d_{out} \times d_{out}}$ maps within the output dimension space. Note that $W_1$ changes the dimension (from $d_{in}$ to $d_{out}$) while $W_2$ preserves it. This is the residual mapping $F(x) = z$.</span>

<span style="font-size: 14px;">**Shortcut path.** Project the input to match the output dimension:</span>

$$
s = x \cdot W_s
$$

<span style="font-size: 14px;">Here $W_s \in \mathbb{R}^{d_{in} \times d_{out}}$ is the projection matrix. It performs a pure linear transformation with no activation function. After this step, $s \in \mathbb{R}^{d_{out}}$.</span>

<span style="font-size: 14px;">**Output.** Add the main path and shortcut path:</span>

$$
y = z + s = F(x) + W_s \cdot x
$$

<span style="font-size: 14px;">The element-wise addition is valid because both $z$ and $s$ are in $\mathbb{R}^{d_{out}}$. The projection $W_s$ is what makes the skip connection work across dimension changes.</span>

---

## <span style="font-size: 16px;">When Projection Is Needed</span>

<span style="font-size: 14px;">Deep residual networks are organized into **stages**. Within a stage, all blocks have the same number of channels. When transitioning from one stage to the next, the channel count typically doubles: 64 to 128, 128 to 256, 256 to 512. These transitions are where the convolutional block appears.</span>

<span style="font-size: 14px;">Element-wise addition requires matching shapes. If $x \in \mathbb{R}^{64}$ and $F(x) \in \mathbb{R}^{128}$, the expression $F(x) + x$ is undefined. Without projection, the network must either pad the shortcut with zeros or abandon the skip connection at stage boundaries.</span>

<span style="font-size: 14px;">In a typical ResNet, the majority of blocks are identity blocks. Only the first block of each new stage is a convolutional block. For ResNet-34, there are 16 residual blocks total: 3 identity blocks in stage 1 (all 64 channels), then 1 conv block + 3 identity blocks in stage 2 (64 to 128), 1 conv block + 5 identity blocks in stage 3 (128 to 256), and 1 conv block + 2 identity blocks in stage 4 (256 to 512). Only 3 out of 16 blocks need a projection shortcut.</span>

---

## <span style="font-size: 16px;">Option A vs Option B</span>

<span style="font-size: 14px;">He et al. (2015) compared two strategies for handling dimension mismatches. The paper states: "When the dimensions increase, we consider two options: (A) zero-padding shortcuts, and (B) projection shortcuts."</span>

<span style="font-size: 14px;">**Option A: Zero-padding.** Pad the shortcut with zeros to match the larger dimension. If the input has 64 channels and the output has 128, append 64 zeros. This adds no parameters but the padded dimensions carry no learned information, wasting half the shortcut capacity.</span>

<span style="font-size: 14px;">**Option B: Projection shortcut.** Apply a learned $1 \times 1$ convolution (or linear layer) to project the input into the higher-dimensional space. This adds $d_{in} \times d_{out}$ parameters but allows the network to learn the most useful mapping to the new dimension.</span>

<span style="font-size: 14px;">The paper's experiments found that Option B slightly outperforms Option A. The improvement is marginal, suggesting that the primary benefit comes from the identity mapping principle rather than the shortcut mechanism. In practice, nearly all modern implementations use Option B because the parameter cost at stage transitions is negligible.</span>

<span style="font-size: 14px;">The paper also tested Option C: projection shortcuts on every block, even where dimensions match. Option C performed marginally better, but the extra parameters were not justified. The standard practice became: projection shortcuts only where needed (Option B), identity shortcuts everywhere else.</span>

---

## <span style="font-size: 16px;">The Projection Matrix $W_s$</span>

<span style="font-size: 14px;">The projection matrix $W_s$ serves a single purpose: change the dimensionality of the shortcut to match the main path output. In the convolutional setting, this is a $1 \times 1$ convolution. In the fully-connected setting, it is a matrix multiplication.</span>

<span style="font-size: 14px;">$W_s$ has shape $d_{in} \times d_{out}$. The multiplication $s = x \cdot W_s$ maps from $\mathbb{R}^{d_{in}}$ to $\mathbb{R}^{d_{out}}$. This is a linear transformation with no bias and no activation.</span>

<span style="font-size: 14px;">The absence of a nonlinearity on the shortcut path is deliberate. The paper's central argument is that identity mappings (or as close to identity as possible) on the shortcut enable clean gradient flow. A purely linear projection is the minimal departure from identity needed to handle the dimension change. Adding ReLU to the shortcut would break the clean gradient path, undermining the core benefit of residual learning.</span>

<span style="font-size: 14px;">In the convolutional case, a $1 \times 1$ convolution with stride 2 simultaneously halves spatial resolution and changes channels. In the fully-connected case, $W_s$ only handles the feature dimension change. The parameter count is $d_{in} \times d_{out}$: for a 64-to-128 transition, $8{,}192$ parameters.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">He et al. (2015), "Deep Residual Learning for Image Recognition," introduced residual networks to address the **degradation problem**: deeper plain networks achieve higher training error than shallower ones, not because of overfitting, but because optimization becomes harder. Adding layers should never hurt in theory (the extra layers could learn identity mappings), but SGD struggles to find these identity solutions.</span>

<span style="font-size: 14px;">Residual connections reformulate the learning problem. Instead of learning $H(x)$ directly, each block learns the residual $F(x) = H(x) - x$, and the output is $F(x) + x$. If the optimal transformation is close to identity, pushing $F(x)$ toward zero is easier than learning identity from scratch.</span>

<span style="font-size: 14px;">The convolutional block with projection shortcut is used at stage boundaries in all ResNet variants: ResNet-18, 34, 50, 101, and 152. In ResNet-18/34, the main path uses two $3 \times 3$ convolutions. In ResNet-50/101/152, the main path uses a bottleneck design ($1 \times 1$, $3 \times 3$, $1 \times 1$). Regardless of main path design, the projection shortcut is always a $1 \times 1$ convolution with appropriate stride and output channels.</span>

<span style="font-size: 14px;">ResNet-152 achieved a 3.57% top-5 error rate on ImageNet, winning ILSVRC 2015. Training networks beyond 100 layers was directly enabled by skip connections; without them, networks deeper than roughly 30 layers suffered severe degradation.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a convolutional block with $d_{in} = 4$ and $d_{out} = 6$. The input vector is:</span>

$$
x = \begin{pmatrix} 1.0 & 0.5 & -0.5 & 2.0 \end{pmatrix}
$$

<span style="font-size: 14px;">The first weight matrix $W_1 \in \mathbb{R}^{4 \times 6}$:</span>

$$
W_1 = \begin{pmatrix} 0.3 & -0.1 & 0.2 & 0.0 & 0.4 & -0.2 \\ 0.1 & 0.5 & -0.3 & 0.2 & 0.0 & 0.1 \\ -0.2 & 0.3 & 0.1 & -0.4 & 0.2 & 0.3 \\ 0.4 & 0.0 & 0.3 & 0.1 & -0.1 & 0.2 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1: Main path, layer 1.** Compute $x \cdot W_1$:</span>

<span style="font-size: 14px;">Element 0: $1.0(0.3) + 0.5(0.1) + (-0.5)(-0.2) + 2.0(0.4) = 0.30 + 0.05 + 0.10 + 0.80 = 1.25$</span>

<span style="font-size: 14px;">Element 1: $1.0(-0.1) + 0.5(0.5) + (-0.5)(0.3) + 2.0(0.0) = -0.10 + 0.25 - 0.15 + 0.00 = 0.00$</span>

<span style="font-size: 14px;">Element 2: $1.0(0.2) + 0.5(-0.3) + (-0.5)(0.1) + 2.0(0.3) = 0.20 - 0.15 - 0.05 + 0.60 = 0.60$</span>

<span style="font-size: 14px;">Element 3: $1.0(0.0) + 0.5(0.2) + (-0.5)(-0.4) + 2.0(0.1) = 0.00 + 0.10 + 0.20 + 0.20 = 0.50$</span>

<span style="font-size: 14px;">Element 4: $1.0(0.4) + 0.5(0.0) + (-0.5)(0.2) + 2.0(-0.1) = 0.40 + 0.00 - 0.10 - 0.20 = 0.10$</span>

<span style="font-size: 14px;">Element 5: $1.0(-0.2) + 0.5(0.1) + (-0.5)(0.3) + 2.0(0.2) = -0.20 + 0.05 - 0.15 + 0.40 = 0.10$</span>

<span style="font-size: 14px;">After ReLU: $h = [1.25, 0.00, 0.60, 0.50, 0.10, 0.10]$. Element 1 was exactly zero and remains zero.</span>

<span style="font-size: 14px;">**Step 2: Main path, layer 2.** Let $W_2 = I_6$ (the $6 \times 6$ identity matrix) for simplicity. Then $h \cdot W_2 = h$, and after ReLU: $z = [1.25, 0.00, 0.60, 0.50, 0.10, 0.10]$.</span>

<span style="font-size: 14px;">**Step 3: Shortcut path.** The projection matrix $W_s \in \mathbb{R}^{4 \times 6}$:</span>

$$
W_s = \begin{pmatrix} 0.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 0.5 & 0.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 0.0 & 0.5 & 0.0 & 0.0 & 0.0 \\ 0.0 & 0.0 & 0.0 & 0.5 & 0.0 & 0.0 \end{pmatrix}
$$

<span style="font-size: 14px;">$s = x \cdot W_s = [1.0(0.5), \; 0.5(0.5), \; (-0.5)(0.5), \; 2.0(0.5), \; 0.0, \; 0.0] = [0.50, \; 0.25, \; -0.25, \; 1.00, \; 0.00, \; 0.00]$</span>

<span style="font-size: 14px;">The projection scaled the 4 input values by 0.5 into the first 4 output positions, with zeros in positions 4 and 5.</span>

<span style="font-size: 14px;">**Step 4: Addition.** Combine main path and shortcut:</span>

$$
y = z + s = [1.75, \; 0.25, \; 0.35, \; 1.50, \; 0.10, \; 0.10]
$$

<span style="font-size: 14px;">The shortcut contributed meaningful information. Element 3 went from $0.50$ (main path alone) to $1.50$ (with shortcut), preserving the strong signal from input value $2.0$. The output carries both the learned transformation and a direct trace of the input.</span>

---

## <span style="font-size: 16px;">Identity Block vs Convolutional Block</span>

<span style="font-size: 14px;">**Identity block.** Used when $d_{in} = d_{out}$. The shortcut is $x$ itself with no transformation. All weight matrices are square: $W_1, W_2 \in \mathbb{R}^{d \times d}$. The output is $y = F(x) + x$. No extra parameters on the shortcut. This appears for all layers within a stage where the channel count is constant.</span>

<span style="font-size: 14px;">**Convolutional block.** Used when $d_{in} \neq d_{out}$. The shortcut applies $W_s \in \mathbb{R}^{d_{in} \times d_{out}}$. The first main path matrix $W_1 \in \mathbb{R}^{d_{in} \times d_{out}}$ changes dimension, and $W_2 \in \mathbb{R}^{d_{out} \times d_{out}}$ preserves it. The output is $y = F(x) + W_s \cdot x$. This appears at the first layer of each new stage.</span>

<span style="font-size: 14px;">Gradient flow differs between the two. In the identity block, $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I$. The identity matrix $I$ ensures the gradient magnitude stays at least 1 through the skip path. In the convolutional block, $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + W_s^T$. The gradient through the shortcut is scaled by $W_s^T$ rather than passing through unchanged. This is why projection shortcuts are used only where necessary: they provide a weaker gradient highway than identity shortcuts.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the projection on the shortcut path.** When input and output dimensions differ, directly adding $x$ to $F(x)$ causes a shape mismatch error. The shortcut must be projected via $W_s$ whenever $d_{in} \neq d_{out}$. This is the most common bug when implementing ResNet from scratch: the forward pass crashes at the addition step because the shortcut was left as raw $x$.</span>

* <span style="font-size: 14px;">**Applying ReLU to the shortcut path.** The projection shortcut must be a pure linear transformation: $s = x \cdot W_s$ with no activation. Adding ReLU (i.e., $s = \text{ReLU}(x \cdot W_s)$) breaks the clean gradient flow that makes residual learning effective. The paper keeps the shortcut as close to identity as possible. A nonlinearity on the shortcut defeats this purpose and degrades optimization in very deep networks.</span>

* <span style="font-size: 14px;">**Wrong dimensions for $W_s$.** The projection matrix must have shape $d_{in} \times d_{out}$. A common mistake is initializing $W_s$ as $d_{out} \times d_{out}$ (square in the output space) or $d_{out} \times d_{in}$ (transposed). The first fails because $x \cdot W_s$ requires the first dimension of $W_s$ to equal $d_{in}$. The second transposes the mapping direction entirely.</span>

* <span style="font-size: 14px;">**Applying a projection when dimensions already match.** If $d_{in} = d_{out}$, the block should use an identity shortcut with $\text{shortcut}(x) = x$. Using a projection when dimensions match adds unnecessary parameters and weakens gradient flow compared to a true identity shortcut. The paper tested this (Option C) and found the marginal gain was not worth the cost.</span>

* <span style="font-size: 14px;">**Confusing the shapes of $W_1$ and $W_2$.** In the convolutional block, $W_1 \in \mathbb{R}^{d_{in} \times d_{out}}$ is rectangular (changes dimension), while $W_2 \in \mathbb{R}^{d_{out} \times d_{out}}$ is square (preserves dimension). Swapping these shapes causes either a dimension error or incorrect output shapes. The dimension change happens at the first layer, not the second.</span>

* <span style="font-size: 14px;">**Mixing up activation placement.** In this formulation, ReLU is applied after each linear layer on the main path: $h = \text{ReLU}(x \cdot W_1)$ and $z = \text{ReLU}(h \cdot W_2)$. Some ResNet formulations apply ReLU after the addition instead (post-addition activation). Mixing up which formulation is being used produces different outputs. Always verify whether activation comes before or after the skip connection addition.</span>

* <span style="font-size: 14px;">**Incorrect matrix multiplication order.** The computation is $x \cdot W$, not $W \cdot x$. With $x$ as a row vector $(1, d_{in})$ and $W$ as $(d_{in}, d_{out})$, the product is $(1, d_{out})$. Reversing the order requires transposed weights and column vectors, a different convention entirely.</span>

---