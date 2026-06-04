# <span style="font-size: 20px;">Bottleneck Block</span>

<span style="font-size: 14px;">The bottleneck block is a three-layer residual building block introduced by He et al. (2015) in "Deep Residual Learning for Image Recognition." Instead of stacking two 3x3 convolutions like the BasicBlock used in ResNet-18/34, the bottleneck uses a 1x1-3x3-1x1 sequence that first compresses channels, performs spatial processing in the compressed space, and then expands channels back. This design makes deep networks like ResNet-50, ResNet-101, and ResNet-152 computationally tractable while maintaining representational power.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The bottleneck block is a residual unit consisting of three convolutional layers arranged in a compress-process-expand pattern. The first layer is a 1x1 convolution that reduces the number of channels from the input dimension to a smaller "bottleneck" dimension. The second layer is a 3x3 convolution that performs spatial filtering in this compressed channel space. The third layer is a 1x1 convolution that expands the channels back to the output dimension. A skip connection adds the original input to the output of this three-layer stack, and ReLU is applied after the addition.</span>

<span style="font-size: 14px;">The term "bottleneck" refers to the narrow middle layer. If the input has 256 channels and the bottleneck dimension is 64, the 3x3 convolution operates on only 64 channels instead of 256. This dramatically reduces the number of multiply-add operations while the 1x1 layers handle the channel dimension changes cheaply. The paper states: "The three layers are 1x1, 3x3, and 1x1 convolutions, where the 1x1 layers are responsible for reducing and then increasing (restoring) dimensions."</span>

<span style="font-size: 14px;">The bottleneck block is the standard building block for all ResNet variants with 50 or more layers. ResNet-50 uses 16 bottleneck blocks, ResNet-101 uses 33, and ResNet-152 uses 50. The BasicBlock (two 3x3 convolutions) is only used in the shallower ResNet-18 and ResNet-34.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x \in \mathbb{R}^{H \times W \times C_{in}}$ be the input feature map. Let $b$ denote the bottleneck width and $C_{out}$ denote the output channels.</span>

<span style="font-size: 14px;">**Layer 1 -- 1x1 Reduce.** Compress channels from $C_{in}$ to $b$:</span>

$$
y_1 = \text{ReLU}(\text{BN}(W_1 * x))
$$

<span style="font-size: 14px;">Here $W_1 \in \mathbb{R}^{b \times C_{in} \times 1 \times 1}$ is a pointwise convolution mapping each spatial position from $C_{in}$ to $b$ channels. Batch normalization is applied before ReLU. Output: $y_1 \in \mathbb{R}^{H \times W \times b}$.</span>

<span style="font-size: 14px;">**Layer 2 -- 3x3 Process.** Spatial convolution in the compressed space:</span>

$$
y_2 = \text{ReLU}(\text{BN}(W_2 * y_1))
$$

<span style="font-size: 14px;">Here $W_2 \in \mathbb{R}^{b \times b \times 3 \times 3}$ with padding 1 to preserve spatial dimensions. This is the only layer that performs spatial filtering. Because it operates on $b$ channels rather than $C_{in}$ or $C_{out}$, the computational cost is greatly reduced. Output: $y_2 \in \mathbb{R}^{H \times W \times b}$.</span>

<span style="font-size: 14px;">**Layer 3 -- 1x1 Expand.** Restore channels from $b$ to $C_{out}$:</span>

$$
y_3 = \text{BN}(W_3 * y_2)
$$

<span style="font-size: 14px;">Here $W_3 \in \mathbb{R}^{C_{out} \times b \times 1 \times 1}$. There is no ReLU after this batch normalization. Output: $y_3 \in \mathbb{R}^{H \times W \times C_{out}}$.</span>

<span style="font-size: 14px;">**Shortcut and Addition.** The residual connection adds the input to the transformed output:</span>

$$
\text{output} = \text{ReLU}(y_3 + \text{shortcut}(x))
$$

<span style="font-size: 14px;">If $C_{in} = C_{out}$ and spatial dimensions are unchanged, the shortcut is identity: $\text{shortcut}(x) = x$. If $C_{in} \neq C_{out}$ or stride > 1, a **projection shortcut** is used: $\text{shortcut}(x) = \text{BN}(W_s * x)$ where $W_s \in \mathbb{R}^{C_{out} \times C_{in} \times 1 \times 1}$. ReLU is applied after the addition, ensuring the skip connection provides a clean linear path for gradient flow.</span>

---

## <span style="font-size: 16px;">Why Bottleneck</span>

<span style="font-size: 14px;">The fundamental motivation is computational efficiency. A 3x3 convolution is the most expensive operation in a residual block because its kernel has $3 \times 3 = 9$ elements versus a single element for 1x1. By reducing channels before the 3x3 layer, the bottleneck ensures this expensive operation runs on far fewer channels.</span>

<span style="font-size: 14px;">Consider a block operating on 256 output channels. A BasicBlock with two 3x3 convolutions at 256 channels has $2 \times (256 \times 256 \times 9) = 1{,}179{,}648$ kernel parameters. The bottleneck with $b = 64$ has $256 \times 64 + 64 \times 64 \times 9 + 64 \times 256 = 69{,}632$ kernel parameters, roughly 17 times fewer.</span>

<span style="font-size: 14px;">The paper frames this practically: ResNet-50 has similar computational cost to the 34-layer BasicBlock network despite having 16 more layers and greater accuracy. Without the bottleneck design, building networks of 50+ layers would be prohibitively expensive. The standard compression ratio is 4:1 ($C_{out} = 4b$), balancing computational savings against representational capacity.</span>

---

## <span style="font-size: 16px;">The Three Layers</span>

### <span style="font-size: 14px;">Layer 1: 1x1 Reduce</span>

<span style="font-size: 14px;">The first 1x1 convolution is a channel-mixing operation with no spatial receptive field. It projects each spatial position independently from $C_{in}$ dimensions to $b$ dimensions, where $b$ is typically $C_{in} / 4$. This layer acts as a learned dimensionality reduction. Followed by batch normalization and ReLU, it produces a compact representation at every spatial location. The 1x1 convolution is extremely cheap: its FLOPs scale as $H \times W \times C_{in} \times b$, with no spatial kernel multiplier.</span>

### <span style="font-size: 14px;">Layer 2: 3x3 Process</span>

<span style="font-size: 14px;">The 3x3 convolution is the only layer with a spatial receptive field. It uses padding 1 to preserve spatial dimensions and detects edges, textures, and patterns at the current resolution. Because it operates on $b$ channels rather than $C_{in}$, the cost is reduced by a factor of $(C_{in}/b)^2 = 16$ compared to a 3x3 conv on the full channel width. When the block needs to downsample at a stage transition, stride 2 is applied at this layer, halving the spatial dimensions.</span>

### <span style="font-size: 14px;">Layer 3: 1x1 Expand</span>

<span style="font-size: 14px;">The final 1x1 convolution projects the $b$-channel representation back to $C_{out} = 4b$ channels. This expansion is necessary so the output can be added to the skip connection. Critically, there is no ReLU after this layer's batch normalization. ReLU comes only after the residual addition. Placing ReLU here would mean the residual branch output is always non-negative, preventing the block from learning to subtract from the identity mapping.</span>

---

## <span style="font-size: 16px;">Computational Savings</span>

<span style="font-size: 14px;">Compare a bottleneck block against an equivalent BasicBlock for a $56 \times 56$ feature map with 256 output channels.</span>

<span style="font-size: 14px;">**BasicBlock** (two 3x3 convolutions at 256 channels). FLOPs per layer: $56^2 \times 256 \times 256 \times 9 = 1{,}849{,}688{,}064$. Two layers: $3{,}699{,}376{,}128$ total FLOPs.</span>

<span style="font-size: 14px;">**Bottleneck** (1x1 reduce to 64, 3x3 at 64, 1x1 expand to 256):</span>

* <span style="font-size: 14px;">**1x1 reduce (256 to 64):** $56^2 \times 256 \times 64 = 51{,}380{,}224$ FLOPs</span>
* <span style="font-size: 14px;">**3x3 process (64 to 64):** $56^2 \times 64 \times 64 \times 9 = 115{,}605{,}504$ FLOPs</span>
* <span style="font-size: 14px;">**1x1 expand (64 to 256):** $56^2 \times 64 \times 256 = 51{,}380{,}224$ FLOPs</span>

<span style="font-size: 14px;">**Bottleneck total:** $218{,}365{,}952$ FLOPs, roughly $5.9\%$ of the BasicBlock's cost. This 17x reduction is what makes 50+ layer networks feasible. The savings come from avoiding the 3x3 convolution on full-width channels: the bottleneck's 3x3 layer costs $115M$ FLOPs versus $1{,}850M$ per 3x3 layer in the BasicBlock.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">He et al. (2015) introduced the bottleneck block as a practical solution for building very deep networks. The paper demonstrated that networks with 50, 101, and 152 layers could be trained successfully using residual connections, achieving state-of-the-art results on ImageNet (top-5 error of 3.57% with an ensemble).</span>

<span style="font-size: 14px;">The paper uses two building block types. The **BasicBlock** for ResNet-18 and ResNet-34 contains two 3x3 convolutions with the same channel count. The **Bottleneck Block** for ResNet-50, ResNet-101, and ResNet-152 uses the 1x1-3x3-1x1 sequence. Replacing BasicBlocks with bottlenecks in a 34-layer network produces the 50-layer ResNet, which has more layers yet similar computational cost.</span>

<span style="font-size: 14px;">The channel progression across stages follows a consistent pattern. Stage 1: $C_{out} = 256$, $b = 64$. Stage 2: $C_{out} = 512$, $b = 128$. Stage 3: $C_{out} = 1024$, $b = 256$. Stage 4: $C_{out} = 2048$, $b = 512$. Each stage begins with a stride-2 block to halve spatial dimensions (except stage 1). The number of blocks per stage: ResNet-50 uses [3, 4, 6, 3], ResNet-101 uses [3, 4, 23, 3], and ResNet-152 uses [3, 8, 36, 3].</span>

<span style="font-size: 14px;">The projection shortcut (option B in the paper) is used whenever channel dimensions change between stages. The paper compared three shortcut strategies: (A) zero-padding for dimension increases, (B) projection shortcuts only for dimension changes, (C) projection shortcuts everywhere. Option B was adopted for bottleneck networks as it provides a clean dimension match without excessive parameters.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace through a bottleneck block with input $x$ of shape $(1, 56, 56, 256)$: batch size 1, spatial size $56 \times 56$, $C_{in} = 256$. Bottleneck width $b = 64$, output channels $C_{out} = 256$ (identity shortcut).</span>

<span style="font-size: 14px;">**Layer 1: 1x1 Reduce (256 to 64).** Weight shape: $(64, 256, 1, 1)$. At each spatial position, a $64 \times 256$ matrix multiplies the 256-channel vector to produce a 64-channel vector. Output after conv, BN, ReLU: $(1, 56, 56, 64)$. Parameters: $64 \times 256 = 16{,}384$ weights + 128 BN parameters = $16{,}512$.</span>

<span style="font-size: 14px;">**Layer 2: 3x3 Process (64 to 64).** Weight shape: $(64, 64, 3, 3)$. Each of 64 output channels has a $64 \times 3 \times 3$ filter. Padding 1 preserves spatial size. Output after conv, BN, ReLU: $(1, 56, 56, 64)$. Parameters: $64 \times 64 \times 9 = 36{,}864$ weights + 128 BN parameters = $36{,}992$.</span>

<span style="font-size: 14px;">**Layer 3: 1x1 Expand (64 to 256).** Weight shape: $(256, 64, 1, 1)$. Projects 64 channels back to 256. Output after conv, BN (no ReLU): $(1, 56, 56, 256)$. Parameters: $256 \times 64 = 16{,}384$ weights + 512 BN parameters = $16{,}896$.</span>

<span style="font-size: 14px;">**Shortcut.** $C_{in} = C_{out} = 256$, spatial size unchanged, so $\text{shortcut}(x) = x$ with shape $(1, 56, 56, 256)$. No additional parameters.</span>

<span style="font-size: 14px;">**Addition and ReLU.** Element-wise add $y_3 + x$, both $(1, 56, 56, 256)$. Apply ReLU. Output: $(1, 56, 56, 256)$. Total block parameters: $16{,}512 + 36{,}992 + 16{,}896 = 70{,}400$.</span>

<span style="font-size: 14px;">Now consider the first block of stage 2: $C_{in} = 256$, $C_{out} = 512$, stride 2. Layer 1: 1x1 reduces $(256 \to 128)$, output $(1, 56, 56, 128)$. Layer 2: 3x3 with stride 2 produces $(1, 28, 28, 128)$. Layer 3: 1x1 expands $(128 \to 512)$, output $(1, 28, 28, 512)$. The shortcut requires a projection: 1x1 conv $(256 \to 512)$ with stride 2, producing $(1, 28, 28, 512)$. Both branches now match for element-wise addition.</span>

---

## <span style="font-size: 16px;">BasicBlock vs Bottleneck</span>

<span style="font-size: 14px;">**Structure.** The BasicBlock has two 3x3 convolutions with the same channel count: $\text{Conv}_{3 \times 3}(C, C) \to \text{Conv}_{3 \times 3}(C, C)$. The bottleneck has three layers with a compress-expand pattern: $\text{Conv}_{1 \times 1}(C, b) \to \text{Conv}_{3 \times 3}(b, b) \to \text{Conv}_{1 \times 1}(b, C)$ where $C = 4b$.</span>

<span style="font-size: 14px;">**Depth vs width trade-off.** The BasicBlock uses 2 layers per block, so a 34-layer network has 16 blocks. The bottleneck uses 3 layers per block, so a 50-layer network also has 16 blocks. Despite having 16 more layers, the 50-layer bottleneck network has similar FLOPs to the 34-layer BasicBlock network because the 3x3 convolution in the bottleneck operates on $b = C/4$ channels.</span>

<span style="font-size: 14px;">**When to use which.** The BasicBlock is appropriate for shallower networks (18, 34 layers) where channel counts are modest (64, 128, 256, 512). At these widths, the overhead of two extra 1x1 convolutions per block outweighs the savings from channel compression. The bottleneck becomes advantageous when output channel counts reach 256 or higher, which is the case for deeper variants.</span>

<span style="font-size: 14px;">**Output channel expansion.** In BasicBlock networks, output channels per stage are [64, 128, 256, 512]. In bottleneck networks, they are [256, 512, 1024, 2048], four times larger at every stage. This is because the 1x1 expand layer outputs $4b$ channels. The wider representations in bottleneck networks contribute to their superior accuracy beyond just additional depth.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong channel count in the expand layer.** The output of the third 1x1 conv must match the skip connection's channel count. If you expand to $C_{out} \neq C_{in}$ without a projection shortcut, the element-wise addition fails with a dimension mismatch. The standard pattern is $C_{out} = 4b$, and $C_{in} = C_{out}$ for all blocks except the first in each stage.</span>

* <span style="font-size: 14px;">**Forgetting the skip connection entirely.** Without the residual addition, the block becomes a plain three-layer network that suffers from degradation at depth. The entire purpose of the residual framework is the identity shortcut. Training a 50+ layer network without skip connections leads to higher training error than a shallower network, which is the core problem the paper addresses.</span>

* <span style="font-size: 14px;">**ReLU placement after the third layer instead of after addition.** ReLU must come after the element-wise addition of the main branch and the shortcut, not after the third batch normalization. Placing ReLU before the addition means the residual branch output is non-negative, preventing the block from learning to subtract from the identity mapping. The correct order is: BN on layer 3 output, add shortcut, then ReLU.</span>

* <span style="font-size: 14px;">**Omitting the projection shortcut when dimensions change.** At stage transitions, $C_{in} \neq C_{out}$ and spatial size may halve. A projection shortcut (1x1 conv with appropriate stride and its own BN) is mandatory. Using an identity shortcut when dimensions mismatch causes a runtime shape error.</span>

* <span style="font-size: 14px;">**Applying stride at the wrong layer.** When downsampling, stride 2 should be applied at the 3x3 convolution (layer 2), not at the 1x1 reduce (layer 1). Applying stride at the 1x1 layer discards spatial information before any spatial processing occurs. Some implementations (PyTorch's ResNet-v1.5) apply stride at the first 1x1, which slightly changes the computation, but the original paper applies it at the 3x3.</span>

* <span style="font-size: 14px;">**Confusing bottleneck width with output width.** The bottleneck width $b$ is not the same as the block output width $C_{out}$. In ResNet-50's first stage, $b = 64$ but $C_{out} = 256$. Setting the expand layer to output $b$ channels instead of $4b$ produces a block with the wrong output size and breaks the residual connection for subsequent blocks.</span>

---