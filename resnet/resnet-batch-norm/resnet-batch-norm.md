# <span style="font-size: 20px;">Batch Normalization</span>

<span style="font-size: 14px;">**Batch Normalization** (BatchNorm) normalizes activations across the batch dimension for each channel, stabilizing and accelerating deep network training. Introduced by Ioffe and Szegedy (2015), it became a standard building block after He et al. adopted it in ResNet with the Conv->BN->ReLU ordering that is now ubiquitous in convolutional architectures.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Batch Normalization is a layer inserted between a linear operation (convolution or fully connected) and its activation function. It normalizes the pre-activation values so that, across the current mini-batch, each feature channel has approximately zero mean and unit variance. After normalizing, it applies a learned affine transform that gives the network the ability to recover any distribution it finds useful.</span>

<span style="font-size: 14px;">The core idea: if the inputs to each layer keep shifting in distribution as earlier layers update their weights, training becomes unstable and slow. By re-centering and re-scaling activations at every layer, BatchNorm decouples layers from each other, allowing each one to learn more independently. The original paper calls this problem **internal covariate shift**.</span>

<span style="font-size: 14px;">In a convolutional network, BatchNorm operates per channel. For an input tensor of shape $(B, C, H, W)$, the normalization statistics (mean and variance) are computed over all $B \times H \times W$ values for each of the $C$ channels independently. This preserves the translation equivariance property of convolutions.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">Batch Mean</span>

<span style="font-size: 14px;">For a given channel $c$, compute the mean over all elements in the batch and all spatial positions:</span>

$$
\mu_B^{(c)} = \frac{1}{B \cdot H \cdot W} \sum_{b=1}^{B} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{b,c,h,w}
$$

### <span style="font-size: 14px;">Batch Variance</span>

$$
\sigma_B^{2(c)} = \frac{1}{B \cdot H \cdot W} \sum_{b=1}^{B} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{b,c,h,w} - \mu_B^{(c)})^2
$$

### <span style="font-size: 14px;">Normalize</span>

<span style="font-size: 14px;">Subtract the batch mean and divide by the standard deviation plus a small constant $\epsilon$ for numerical stability:</span>

$$
\hat{x}_{b,c,h,w} = \frac{x_{b,c,h,w} - \mu_B^{(c)}}{\sqrt{\sigma_B^{2(c)} + \epsilon}}
$$

<span style="font-size: 14px;">A typical value for $\epsilon$ is $10^{-5}$. It prevents division by zero when a channel has zero variance.</span>

### <span style="font-size: 14px;">Scale and Shift</span>

<span style="font-size: 14px;">The normalized value is transformed by learnable parameters $\gamma^{(c)}$ and $\beta^{(c)}$:</span>

$$
y_{b,c,h,w} = \gamma^{(c)} \hat{x}_{b,c,h,w} + \beta^{(c)}
$$

<span style="font-size: 14px;">Each channel has its own $\gamma$ (scale) and $\beta$ (shift), so for $C$ channels the layer adds $2C$ learnable parameters.</span>

### <span style="font-size: 14px;">Running Statistics (EMA)</span>

<span style="font-size: 14px;">During training, BatchNorm maintains **running estimates** of the mean and variance using an exponential moving average. After each training step:</span>

$$
\hat{\mu}_{\text{running}}^{(c)} \leftarrow (1 - m) \cdot \hat{\mu}_{\text{running}}^{(c)} + m \cdot \mu_B^{(c)}
$$

$$
\hat{\sigma}_{\text{running}}^{2(c)} \leftarrow (1 - m) \cdot \hat{\sigma}_{\text{running}}^{2(c)} + m \cdot \sigma_B^{2(c)}
$$

<span style="font-size: 14px;">Here $m$ is the **momentum** parameter (default 0.1 in PyTorch). The running statistics are accumulated purely for inference. Note: PyTorch's convention defines momentum so that a larger $m$ gives more weight to the current batch. Some frameworks use the opposite convention, so always check the documentation.</span>

---

## <span style="font-size: 16px;">Training vs. Inference</span>

### <span style="font-size: 14px;">Training Mode</span>

<span style="font-size: 14px;">During training, BatchNorm uses the **current mini-batch statistics** ($\mu_B$, $\sigma_B^2$) to normalize activations. This is essential because the batch statistics provide a differentiable path for backpropagation. Simultaneously, the running mean and running variance are updated via EMA. These buffers are stored in the model's state dict but do not participate in gradient computation.</span>

### <span style="font-size: 14px;">Inference Mode</span>

<span style="font-size: 14px;">At inference time, there may be no batch at all (batch size = 1), or the batch may not represent the training distribution. BatchNorm switches to the **accumulated running statistics**:</span>

$$
\hat{x} = \frac{x - \hat{\mu}_{\text{running}}}{\sqrt{\hat{\sigma}_{\text{running}}^2 + \epsilon}}
$$

<span style="font-size: 14px;">With running statistics and the learned $\gamma$, $\beta$, the entire BatchNorm layer reduces to a fixed affine transform at inference. This means it can be **fused** into the preceding convolution with no computational overhead, a common optimization in deployed models.</span>

---

## <span style="font-size: 16px;">Why Batch Normalize</span>

### <span style="font-size: 14px;">Internal Covariate Shift</span>

<span style="font-size: 14px;">Ioffe and Szegedy (2015) motivated BatchNorm as a solution to **internal covariate shift**: the continuous change in the distribution of a layer's inputs caused by updates to preceding layers. By fixing the first two moments of each layer's inputs, BatchNorm reduces this instability. Later research (Santurkar et al., 2018) argued that the benefit is better explained by **smoothing the loss landscape**, making gradient directions more reliable and allowing larger steps.</span>

### <span style="font-size: 14px;">Enables Higher Learning Rates</span>

<span style="font-size: 14px;">Without BatchNorm, large learning rates cause activations to explode or collapse in deep networks. By keeping activations bounded in a normalized range, BatchNorm allows learning rates that would otherwise diverge. Ioffe and Szegedy reported training BN-Inception with learning rates 10x-30x higher than the baseline while reaching the same accuracy faster.</span>

### <span style="font-size: 14px;">Regularization Effect</span>

<span style="font-size: 14px;">Because batch statistics are computed from a random mini-batch, the normalization introduces noise into activations. Each sample's normalized value depends on what other samples happen to be in the batch. This acts as a mild regularizer, similar in spirit to dropout. Networks trained with BatchNorm often need less dropout or can omit it entirely. The regularization strength decreases with larger batch sizes.</span>

---

## <span style="font-size: 16px;">The Learnable Parameters</span>

<span style="font-size: 14px;">After normalization, every channel's activations have zero mean and unit variance. If this were the final output, the network would lose representational power. The learnable parameters $\gamma$ and $\beta$ restore it.</span>

<span style="font-size: 14px;">If the network learns $\gamma^{(c)} = \sqrt{\sigma_B^{2(c)} + \epsilon}$ and $\beta^{(c)} = \mu_B^{(c)}$, the BatchNorm transformation becomes the identity: $y = x$. This means the network can **undo** normalization entirely if that is optimal. In practice, the network finds an intermediate setting where normalization helps but some deviation from strict zero-mean/unit-variance is beneficial.</span>

<span style="font-size: 14px;">For a conv layer with $C$ output channels, BatchNorm adds $2C$ parameters. This is tiny compared to the conv itself. A $3 \times 3$ conv with 256 input and 256 output channels has $589{,}824$ parameters; its BatchNorm adds only $512$.</span>

---

## <span style="font-size: 16px;">Running Statistics</span>

### <span style="font-size: 14px;">The EMA Update Rule</span>

<span style="font-size: 14px;">With momentum $m = 0.1$ (PyTorch default), after each batch:</span>

$$
\hat{\mu}_{\text{running}} \leftarrow 0.9 \cdot \hat{\mu}_{\text{running}} + 0.1 \cdot \mu_B
$$

$$
\hat{\sigma}_{\text{running}}^2 \leftarrow 0.9 \cdot \hat{\sigma}_{\text{running}}^2 + 0.1 \cdot \sigma_B^2
$$

<span style="font-size: 14px;">The running mean is initialized to 0 and the running variance to 1. Over many training iterations, these converge to the global dataset statistics for each channel. Smaller $m$ means slower adaptation but smoother estimates; larger $m$ tracks recent batches more closely.</span>

### <span style="font-size: 14px;">Why Running Statistics Are Needed</span>

<span style="font-size: 14px;">At inference time, you may process a single image (batch size = 1). Computing a "batch mean" from one sample would just be the sample itself, making normalization meaningless -- the output would always be zero. Running statistics provide a stable, batch-size-independent estimate representing the training distribution. They also ensure **determinism**: two identical inputs always produce identical outputs regardless of batch composition.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Batch Normalization was introduced by Ioffe and Szegedy in "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (2015). They applied BatchNorm to Inception (GoogLeNet) and demonstrated that BN-Inception matched the original Inception's accuracy in 14x fewer training steps.</span>

<span style="font-size: 14px;">He et al. (2015) adopted BatchNorm as a core component of ResNet. In ResNet, every convolutional layer is followed immediately by BatchNorm before the ReLU activation: **Conv -> BN -> ReLU**. The paper states: "We adopt batch normalization right after each convolution and before activation." This ordering ensures normalization operates on the raw convolution output before the non-linearity clips negative values.</span>

<span style="font-size: 14px;">The Conv -> BN -> ReLU pattern has a specific benefit in residual blocks. The shortcut connection adds the input $x$ to the residual branch $F(x)$. BatchNorm keeps the magnitude of the residual branch controlled. Without it, outputs could grow unboundedly as the network deepens to 50, 101, or 152 layers. ResNet's success -- winning ILSVRC 2015 with 3.57% top-5 error -- cemented BatchNorm as indispensable in convolutional architectures.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a single channel ($c = 0$) with a mini-batch of $B = 3$ scalar values (e.g., after global average pooling). Use $\epsilon = 0.00001$, $\gamma = 1.5$, $\beta = 0.5$. Current running mean $= 0.0$, running variance $= 1.0$, momentum $m = 0.1$.</span>

$$
x_1 = 2.0, \quad x_2 = 4.0, \quad x_3 = 6.0
$$

### <span style="font-size: 14px;">Step 1: Batch Mean</span>

$$
\mu_B = \frac{2.0 + 4.0 + 6.0}{3} = \frac{12.0}{3} = 4.0
$$

### <span style="font-size: 14px;">Step 2: Batch Variance</span>

$$
\sigma_B^2 = \frac{(2.0 - 4.0)^2 + (4.0 - 4.0)^2 + (6.0 - 4.0)^2}{3} = \frac{4.0 + 0.0 + 4.0}{3} = \frac{8.0}{3} \approx 2.6667
$$

### <span style="font-size: 14px;">Step 3: Normalize</span>

$$
\sqrt{\sigma_B^2 + \epsilon} = \sqrt{2.6667 + 0.00001} \approx 1.6330
$$

$$
\hat{x}_1 = \frac{2.0 - 4.0}{1.6330} \approx -1.2247, \quad \hat{x}_2 = \frac{4.0 - 4.0}{1.6330} = 0.0, \quad \hat{x}_3 = \frac{6.0 - 4.0}{1.6330} \approx 1.2247
$$

### <span style="font-size: 14px;">Step 4: Scale and Shift</span>

$$
y_1 = 1.5 \times (-1.2247) + 0.5 = -1.8371 + 0.5 = -1.3371
$$

$$
y_2 = 1.5 \times 0.0 + 0.5 = 0.5
$$

$$
y_3 = 1.5 \times 1.2247 + 0.5 = 1.8371 + 0.5 = 2.3371
$$

### <span style="font-size: 14px;">Step 5: Update Running Statistics</span>

$$
\hat{\mu}_{\text{running}} \leftarrow 0.9 \times 0.0 + 0.1 \times 4.0 = 0.4
$$

$$
\hat{\sigma}_{\text{running}}^2 \leftarrow 0.9 \times 1.0 + 0.1 \times 2.6667 = 0.9 + 0.2667 = 1.1667
$$

<span style="font-size: 14px;">After this batch, the running mean moved from 0.0 to 0.4 (toward the batch mean of 4.0) and the running variance from 1.0 to 1.1667 (toward 2.6667). Over thousands of batches, these converge to the true dataset statistics for this channel.</span>

---

## <span style="font-size: 16px;">BatchNorm vs. LayerNorm vs. RMSNorm</span>

* <span style="font-size: 14px;">**Batch Normalization (BN):** Normalizes across the batch and spatial dimensions, per channel. For tensor $(B, C, H, W)$, statistics are over $(B, H, W)$ for each $C$. Requires large batch sizes. Standard in CNNs (ResNet, EfficientNet).</span>
* <span style="font-size: 14px;">**Layer Normalization (LN):** Normalizes across feature dimensions, per sample. For tensor $(B, T, D)$ in a transformer, statistics are over $D$ for each $(B, T)$ position. No batch dependency, so it is the standard for transformers and RNNs. No running statistics needed.</span>
* <span style="font-size: 14px;">**RMSNorm:** Simplified LayerNorm that skips mean subtraction. Normalizes by the root mean square: $\hat{x} = x / \text{RMS}(x)$ where $\text{RMS}(x) = \sqrt{\frac{1}{D}\sum x_i^2}$. Saves 10-15% of normalization cost. Used in LLaMA, Gemma, and other recent LLMs.</span>

<span style="font-size: 14px;">**When to use which:** BatchNorm for convolutional architectures with batch sizes of 16+. LayerNorm for transformers and sequence models. RMSNorm as a drop-in replacement for LayerNorm when training efficiency matters. BatchNorm is rarely used in transformers because sequences in a batch differ in length and content, making batch statistics unreliable.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to switch between train and eval mode.** In PyTorch, `model.train()` makes BatchNorm use batch statistics and update running statistics. `model.eval()` switches to running statistics. Forgetting `model.eval()` at inference causes outputs to depend on batch composition. Forgetting `model.train()` during training means running statistics are never updated.</span>
* <span style="font-size: 14px;">**Batch size too small.** With batch sizes of 1-4, batch statistics become extremely noisy. For batch size = 1, the variance is zero and normalization collapses. Solutions: Group Normalization, Synchronized BatchNorm across GPUs, or LayerNorm.</span>
* <span style="font-size: 14px;">**Running statistics not updated.** If the model stays in eval mode during training, running statistics remain at initialization (mean = 0, variance = 1). Training may proceed using batch stats, but switching to eval at inference yields catastrophic accuracy drops.</span>
* <span style="font-size: 14px;">**Wrong normalization dimension.** BN normalizes over $(B, H, W)$ per channel. Normalizing over the channel dimension instead gives Instance Normalization. In PyTorch, `nn.BatchNorm2d(C)` handles this automatically, but manual implementations must sum over the correct axes.</span>
* <span style="font-size: 14px;">**Placing BN after ReLU instead of before.** The canonical ResNet order is Conv -> BN -> ReLU. BN after ReLU operates on non-negative values only, biasing the mean and distorting the variance. Both the BN paper and ResNet paper specify placing BN before the activation.</span>
* <span style="font-size: 14px;">**Including bias in the preceding convolution.** BatchNorm subtracts the mean and adds its own $\beta$, so any conv bias is redundant -- it gets absorbed into $\mu_B$ and cancelled. Using `bias=False` saves parameters. PyTorch's ResNet implementations follow this convention.</span>
* <span style="font-size: 14px;">**Freezing BatchNorm incorrectly during fine-tuning.** When fine-tuning on a small dataset, batch statistics may not represent the new domain. The safest approach: freeze the entire BN layer by setting `requires_grad=False` for $\gamma$ and $\beta$ and keeping the layer in eval mode.</span>

---