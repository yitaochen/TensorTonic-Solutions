# <span style="font-size: 20px;">Skip Connections / Residual Connections</span>

<span style="font-size: 14px;">Skip connections (also called residual connections or shortcut connections) are the defining architectural innovation of ResNet (He et al., 2015). By adding the input of a block directly to its output via $y = F(x) + x$, they create an additive identity path that preserves gradient flow through arbitrarily deep networks, solving the degradation problem that prevented training of very deep plain networks.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A skip connection is a direct additive pathway that bypasses one or more layers. Instead of learning the full desired mapping $H(x)$ directly, the stacked layers learn only the **residual** $F(x) = H(x) - x$, and the output is reconstructed as $y = F(x) + x$. The input $x$ is added element-wise to the output of the stacked layers, with no additional parameters or computation.</span>

<span style="font-size: 14px;">Given an input $x$ flowing into a block of two or more layers (typically two $3 \times 3$ convolutions with batch normalization and ReLU), the output of those layers is $F(x)$. The network computes $y = F(x) + x$ and passes that forward. The addition is element-wise and requires matching dimensions.</span>

<span style="font-size: 14px;">This transforms the optimization landscape: instead of learning the entire transformation from scratch, the network only needs to learn how to **adjust** the input. If the optimal transformation is close to identity, the residual $F(x)$ is close to zero, and layers need only learn small perturbations.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The core equation of a residual block is:</span>

$$
y = F(x) + x
$$

* <span style="font-size: 14px;">**$x$:** Input to the residual block (e.g., a feature map of shape $C \times H \times W$).</span>
* <span style="font-size: 14px;">**$F(x)$:** The residual function learned by the stacked layers. For a two-layer block: $F(x) = W_2 \cdot \sigma(W_1 \cdot x)$, where $\sigma$ is ReLU.</span>
* <span style="font-size: 14px;">**$y$:** Output of the residual block, passed to the next block or to a final classifier.</span>

<span style="font-size: 14px;">The gradient during backpropagation with a skip connection:</span>

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left( \frac{\partial F(x)}{\partial x} + I \right)
$$

* <span style="font-size: 14px;">**$I$:** The identity matrix, arising from $\frac{\partial x}{\partial x}$. This is the critical term that guarantees gradient flow.</span>
* <span style="font-size: 14px;">**$\frac{\partial F(x)}{\partial x}$:** The Jacobian of the residual function. Even if this has small magnitude, the $+ I$ ensures the total gradient never vanishes.</span>

<span style="font-size: 14px;">The gradient without a skip connection (plain network):</span>

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial F(x)}{\partial x}
$$

<span style="font-size: 14px;">No additive identity term. The gradient depends entirely on $\frac{\partial F(x)}{\partial x}$, which is a product of Jacobians through the stacked layers. If any Jacobian has small singular values, the gradient shrinks.</span>

<span style="font-size: 14px;">When dimensions of $x$ and $F(x)$ do not match (e.g., channel count changes or spatial downsampling), a linear projection $W_s$ is applied to the shortcut:</span>

$$
y = F(x) + W_s x
$$

<span style="font-size: 14px;">$W_s$ is typically a $1 \times 1$ convolution. He et al. showed that identity shortcuts (no $W_s$) outperform projection shortcuts when dimensions already match, confirming the parameter-free identity path is optimal.</span>

---

## <span style="font-size: 16px;">The Vanishing Gradient Problem</span>

<span style="font-size: 14px;">In a plain deep network, the gradient at an early layer is computed as a product of Jacobians through every subsequent layer:</span>

$$
\frac{\partial \mathcal{L}}{\partial \theta_l} = \frac{\partial \mathcal{L}}{\partial h_L} \cdot \prod_{k=l}^{L-1} \frac{\partial h_{k+1}}{\partial h_k} \cdot \frac{\partial h_l}{\partial \theta_l}
$$

<span style="font-size: 14px;">Each factor $\frac{\partial h_{k+1}}{\partial h_k}$ is a layer Jacobian. If the spectral norm of each Jacobian is less than 1, the product shrinks exponentially with depth.</span>

### <span style="font-size: 14px;">Concrete Example: 50-Layer Plain Network</span>

<span style="font-size: 14px;">Suppose each layer's Jacobian has spectral norm $0.9$ (realistic for poorly scaled weights or saturating nonlinearities). The gradient at layer 1 relative to the output is scaled by:</span>

$$
0.9^{49} \approx 0.00515
$$

<span style="font-size: 14px;">Only $0.5\%$ of the output gradient reaches the first layer. For a 100-layer network:</span>

$$
0.9^{99} \approx 0.0000265
$$

<span style="font-size: 14px;">The gradient is effectively zero. This is the **vanishing gradient problem**: the deeper the network, the worse it gets, because multiplicative factors compound exponentially.</span>

<span style="font-size: 14px;">Conversely, if Jacobian norms exceed 1 (say $1.1$), the product explodes: $1.1^{49} \approx 117.4$. This is the **exploding gradient problem**. Both failure modes share the same root cause: gradients in plain networks are products of Jacobians, and products are exponentially sensitive to whether factors are above or below unity.</span>

<span style="font-size: 14px;">Careful initialization (Xavier, He) and batch normalization help keep Jacobian norms near 1 but cannot guarantee it across all layers and training iterations. For very deep networks (50+ layers), these are insufficient: empirically, deeper plain networks have higher training error than shallower ones -- not from overfitting, but from optimization difficulty.</span>

---

## <span style="font-size: 16px;">How Skip Connections Solve It</span>

<span style="font-size: 14px;">Skip connections change gradient computation from a multiplicative chain to an additive one. For a network of $N$ residual blocks, the output is:</span>

$$
x_N = x_0 + \sum_{i=0}^{N-1} F_i(x_i)
$$

<span style="font-size: 14px;">The gradient with respect to any intermediate activation $x_l$:</span>

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_N} \cdot \left( 1 + \frac{\partial}{\partial x_l} \sum_{i=l}^{N-1} F_i(x_i) \right)
$$

<span style="font-size: 14px;">The crucial feature is the **additive $1$** (the identity). Even if $\frac{\partial}{\partial x_l} \sum F_i$ is small or noisy, the gradient is at least $\frac{\partial \mathcal{L}}{\partial x_N}$. The output gradient propagates directly to any earlier layer without attenuation through this "gradient highway."</span>

<span style="font-size: 14px;">In a 50-layer residual network, the gradient at layer 1 is not $0.9^{49}$ of the output gradient. It is at minimum the output gradient itself (through the identity path), plus whatever additional gradient flows through the residual paths. The gradient magnitude is bounded below by a non-vanishing quantity regardless of depth.</span>

---

## <span style="font-size: 16px;">The Residual Learning Insight</span>

<span style="font-size: 14px;">The paper's central hypothesis is that learning a residual $F(x) = H(x) - x$ is easier than learning the full mapping $H(x)$ directly. This is a claim about optimization, not representation: both formulations can represent the same functions, but the residual form is easier to optimize.</span>

<span style="font-size: 14px;">The argument centers on what happens when the optimal mapping is close to identity:</span>

* <span style="font-size: 14px;">**In a plain network:** The layers must learn $H(x) \approx x$, driving weights to reproduce an identity mapping. This is nontrivial, especially for convolutional layers with no natural identity initialization.</span>
* <span style="font-size: 14px;">**In a residual network:** The layers must learn $F(x) \approx 0$, driving weights toward zero. This is much easier because weight decay and initialization both naturally push weights toward small values.</span>

<span style="font-size: 14px;">The authors confirm this empirically: in trained ResNets, the residual functions $F(x)$ have consistently small magnitudes compared to the skip connection $x$. The learned residuals are small perturbations, confirming that layers learn refinements rather than complete transformations.</span>

<span style="font-size: 14px;">This also explains the **degradation problem**: deeper plain networks have higher training error than shallower ones. If additional layers could learn identity mappings, a deeper network should be at least as good as a shallower one. The fact that they perform worse proves optimizing identity-like mappings is genuinely difficult. Residual learning solves this by making identity-like mappings correspond to zero residuals.</span>

---

## <span style="font-size: 16px;">Paper Context: He et al., 2015</span>

<span style="font-size: 14px;">"Deep Residual Learning for Image Recognition" (He, Zhang, Ren, Sun, 2015) from Microsoft Research won first place in ILSVRC 2015 classification, detection, and localization, plus COCO 2015 detection and segmentation.</span>

### <span style="font-size: 14px;">The Degradation Problem</span>

<span style="font-size: 14px;">The paper opens with a counterintuitive observation: a 56-layer plain network has higher training and test error than a 20-layer one. This is not overfitting (training error is also higher), but an optimization failure the authors term the "degradation problem."</span>

### <span style="font-size: 14px;">152 Layers Trained Successfully</span>

<span style="font-size: 14px;">ResNet-152 achieved 3.57% top-5 error on ImageNet, surpassing human-level performance (~5.1%). The paper also demonstrated training up to 1,202 layers on CIFAR-10. The critical achievement was that training error continued to decrease with depth, proving skip connections resolved the degradation problem.</span>

### <span style="font-size: 14px;">Design Choices</span>

<span style="font-size: 14px;">"The shortcut connections introduce neither extra parameter nor computation complexity." Identity shortcuts are preferred over projection shortcuts. Each residual block contains two $3 \times 3$ convolutions (ResNet-34) or a bottleneck with $1 \times 1$, $3 \times 3$, $1 \times 1$ convolutions (ResNet-50/101/152). Batch normalization follows each convolution, and ReLU is applied after the addition.</span>

---

## <span style="font-size: 16px;">Numerical Example: 3 Layers With and Without Skips</span>

<span style="font-size: 14px;">Consider a 3-layer network with scalar input $x_0 = 2.0$, scalar weights $w_k$, and layer function $h_{k+1} = w_k \cdot h_k$. Let $w_1 = 0.5$, $w_2 = 0.6$, $w_3 = 0.7$.</span>

### <span style="font-size: 14px;">Case 1: Plain Network</span>

<span style="font-size: 14px;">**Forward pass:**</span>

$$
h_1 = 0.5 \times 2.0 = 1.0
$$

$$
h_2 = 0.6 \times 1.0 = 0.6
$$

$$
h_3 = 0.7 \times 0.6 = 0.42
$$

<span style="font-size: 14px;">**Gradient at $x_0$:**</span>

$$
\frac{\partial h_3}{\partial x_0} = w_3 \cdot w_2 \cdot w_1 = 0.7 \times 0.6 \times 0.5 = 0.21
$$

<span style="font-size: 14px;">Only 21% of the output gradient reaches the input. With 50 layers at average weight $0.6$: $0.6^{50} \approx 1.3 \times 10^{-11}$, effectively zero.</span>

### <span style="font-size: 14px;">Case 2: Residual Network</span>

<span style="font-size: 14px;">Each layer computes $h_{k+1} = w_k \cdot h_k + h_k = (w_k + 1) \cdot h_k$.</span>

<span style="font-size: 14px;">**Forward pass:**</span>

$$
h_1 = 1.5 \times 2.0 = 3.0
$$

$$
h_2 = 1.6 \times 3.0 = 4.8
$$

$$
h_3 = 1.7 \times 4.8 = 8.16
$$

<span style="font-size: 14px;">**Gradient at $x_0$:**</span>

$$
\frac{\partial h_3}{\partial x_0} = (w_3 + 1)(w_2 + 1)(w_1 + 1) = 1.7 \times 1.6 \times 1.5 = 4.08
$$

<span style="font-size: 14px;">The gradient is $4.08$ vs $0.21$. Skip connections shift each factor from $w_k < 1$ to $w_k + 1 > 1$, preventing gradient vanishing.</span>

* <span style="font-size: 14px;">**Plain network gradient:** $\prod w_k = 0.21$ (shrinks exponentially)</span>
* <span style="font-size: 14px;">**Residual network gradient:** $\prod (w_k + 1) = 4.08$ (stays large)</span>

<span style="font-size: 14px;">At 50 layers: plain gives $0.6^{50} \approx 10^{-11}$, residual gives $1.6^{50} \approx 6.4 \times 10^{10}$. In practice, batch normalization and weight decay keep gradients well-behaved, but the $+1$ fundamentally prevents vanishing.</span>

---

## <span style="font-size: 16px;">Modern Impact</span>

<span style="font-size: 14px;">Skip connections have become the most ubiquitous architectural pattern in deep learning, appearing far beyond their original CNN context.</span>

### <span style="font-size: 14px;">Transformers and the Residual Stream</span>

<span style="font-size: 14px;">Every transformer block uses two skip connections: one around self-attention, one around the feed-forward sub-layer. The "residual stream" interpretation views the sequence of residual additions as a shared communication channel that attention heads and MLPs read from and write to. Without skip connections, transformers cannot be trained at the depths required for modern language models.</span>

### <span style="font-size: 14px;">U-Net</span>

<span style="font-size: 14px;">U-Net (Ronneberger et al., 2015) uses skip connections between corresponding encoder and decoder layers, **concatenating** (not adding) feature maps. These allow the decoder to recover fine spatial details lost during downsampling and are essential for segmentation and modern diffusion models.</span>

### <span style="font-size: 14px;">DenseNet</span>

<span style="font-size: 14px;">DenseNet (Huang et al., 2017) connects every layer to every subsequent layer via concatenation: $x_l = H_l([x_0, x_1, \ldots, x_{l-1}])$. This maximizes feature reuse and gradient flow but increases memory compared to ResNet's additive shortcuts.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Applying Activation After Addition Kills the Gradient Highway</span>

<span style="font-size: 14px;">In the original ResNet ("post-activation"), ReLU is applied **after** the addition: $\text{ReLU}(F(x) + x)$. This means the identity path passes through a ReLU, which clips negative values and partially blocks gradient flow. The "pre-activation ResNet" (He et al., 2016) moves BN and ReLU **before** the weight layers, leaving the identity path completely clean: $y = F(\text{BN}(\text{ReLU}(x))) + x$. For deep ResNets (100+ layers), pre-activation significantly improves performance.</span>

### <span style="font-size: 14px;">Confusing Skip Connections With Dense Connections</span>

<span style="font-size: 14px;">ResNet uses **element-wise addition**: $y = F(x) + x$. DenseNet uses **channel-wise concatenation**: $y = [F(x), x]$. Addition preserves tensor dimensions and adds no parameters. Concatenation increases the channel count, growing memory and computation in subsequent layers. Implementing one when the other is intended produces a fundamentally different architecture.</span>

### <span style="font-size: 14px;">Dimension Mismatch in the Shortcut Path</span>

<span style="font-size: 14px;">Element-wise addition requires $F(x)$ and $x$ to have identical shapes. When a block changes channels or applies strided convolution, the shortcut needs a $1 \times 1$ projection convolution with matching stride. Forgetting this causes shape mismatch errors. Conversely, applying a projection when dimensions already match wastes parameters and performs worse than the identity shortcut, as shown in the paper.</span>

### <span style="font-size: 14px;">Placing Skip Connections Around Single Layers</span>

<span style="font-size: 14px;">A single-layer residual block $y = w \cdot x + x = (w + 1) \cdot x$ is equivalent to a plain linear layer with a shifted weight, providing no representational benefit. The nonlinearity between stacked layers within a block is what gives $F(x)$ the capacity to learn useful transformations. Skip connections around single linear layers are mathematically trivial.</span>

### <span style="font-size: 14px;">Wrong Addition Dimensions</span>

<span style="font-size: 14px;">A common implementation error is adding tensors along the wrong axis or broadcasting incorrectly. The addition in $y = F(x) + x$ must be strictly element-wise across all dimensions ($C \times H \times W$). Broadcasting a $(C \times 1 \times 1)$ tensor across spatial dimensions or summing along the channel axis silently corrupts the gradient flow that skip connections are designed to preserve.</span>

---