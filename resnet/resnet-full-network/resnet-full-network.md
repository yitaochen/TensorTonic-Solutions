# <span style="font-size: 20px;">Full ResNet Assembly</span>

<span style="font-size: 14px;">ResNet-18 is a complete image classification network that stacks residual blocks into a deep pipeline: an initial convolution, four stages of paired BasicBlocks with skip connections, and a final fully connected classifier. Introduced by He et al. (2015), ResNet-18 is the smallest member of the ResNet family and serves as the reference architecture for understanding how residual learning scales from individual blocks to a full network.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The full ResNet assembly is a classification network that takes a raw image and produces class probabilities. Unlike problems that focus on a single residual block in isolation, this problem requires you to wire together the entire pipeline: initial feature extraction via a large convolution, progressive feature refinement through four stages of stacked residual blocks, and a classification head that maps the final feature map to class logits.</span>

<span style="font-size: 14px;">The defining principle is residual learning. Instead of asking each stack of layers to learn the desired mapping $H(x)$ directly, the network learns the residual $F(x) = H(x) - x$, so the output becomes $y = F(x) + x$. The skip connection that adds $x$ back provides a gradient highway, allowing the network to grow much deeper without vanishing gradients. In the full assembly, this principle is applied uniformly across all eight BasicBlocks, but with a critical twist at stage boundaries: when spatial dimensions halve and channel counts double, the skip connection must use a projection shortcut to match dimensions.</span>

<span style="font-size: 14px;">ResNet-18 contains 18 weight layers: 1 initial convolution, 16 convolution layers across 8 BasicBlocks (2 convolutions per block), and 1 fully connected layer. The "18" counts every layer with learnable weight parameters. Batch normalization and ReLU layers are not counted.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Identity shortcut (within a stage).** When input and output have the same dimensions:</span>

$$
y = \text{ReLU}(F(x) + x)
$$

<span style="font-size: 14px;">where $F(x) = W_2 \cdot \text{ReLU}(\text{BN}(W_1 \cdot x))$ represents two stacked 3x3 convolutions with batch normalization. Here $W_1$ and $W_2$ are the weight tensors of the two convolutional layers, and BN denotes batch normalization.</span>

<span style="font-size: 14px;">**Projection shortcut (at stage transitions).** When spatial dimensions halve or channel count changes:</span>

$$
y = \text{ReLU}(F(x) + W_s \cdot x)
$$

<span style="font-size: 14px;">where $W_s$ is a 1x1 convolution with stride 2 that simultaneously halves spatial dimensions and increases channel count.</span>

<span style="font-size: 14px;">**Full network pipeline:**</span>

$$
\text{Input} \xrightarrow{\text{Conv1}} \xrightarrow{\text{MaxPool}} \xrightarrow{\text{Stage1}} \xrightarrow{\text{Stage2}} \xrightarrow{\text{Stage3}} \xrightarrow{\text{Stage4}} \xrightarrow{\text{AvgPool}} \xrightarrow{\text{FC}} \text{Logits}
$$

<span style="font-size: 14px;">Each stage contains exactly 2 BasicBlocks. The first block of stages 2, 3, and 4 uses a projection shortcut with stride 2. All other blocks use identity shortcuts.</span>

* <span style="font-size: 14px;">**$x$:** Input tensor to a residual block, carrying feature maps from the previous block or stage.</span>
* <span style="font-size: 14px;">**$F(x)$:** The residual function: two consecutive 3x3 convolutions with BN and ReLU between them.</span>
* <span style="font-size: 14px;">**$W_s$:** Projection matrix implemented as a 1x1 convolution with stride 2, used only at stage transitions.</span>
* <span style="font-size: 14px;">**$y$:** Output of the residual block, always passed through a final ReLU after the addition.</span>

---

## <span style="font-size: 16px;">The Architecture</span>

<span style="font-size: 14px;">ResNet-18 begins with aggressive spatial reduction. Conv1 is a 7x7 convolution with 64 output channels, stride 2, and padding 3, taking an input of $3 \times 224 \times 224$ to $64 \times 112 \times 112$. Batch normalization and ReLU follow. A 3x3 max pooling layer with stride 2 and padding 1 then reduces spatial dimensions to $64 \times 56 \times 56$.</span>

<span style="font-size: 14px;">After Conv1 and max pooling, the feature map passes through four stages of 2 BasicBlocks each. Each BasicBlock consists of: Conv 3x3, BatchNorm, ReLU, Conv 3x3, BatchNorm, then the skip connection addition, and a final ReLU on the sum. Stage 1 operates at $56 \times 56$ with 64 channels. Stages 2 through 4 each begin by halving spatial resolution and doubling channels.</span>

<span style="font-size: 14px;">After Stage 4, the $512 \times 7 \times 7$ feature map passes through adaptive average pooling, collapsing spatial dimensions to $512 \times 1 \times 1$. This is flattened to a 512-dimensional vector and fed into a fully connected layer mapping to $C$ class logits (1000 for ImageNet).</span>

---

## <span style="font-size: 16px;">Stage Transitions</span>

<span style="font-size: 14px;">Stage transitions are where the most implementation complexity lies. Moving from one stage to the next, spatial dimensions halve and channel count doubles, creating a dimension mismatch between $x$ and $F(x)$ in the first block of the new stage.</span>

<span style="font-size: 14px;">He et al. proposed three options for this mismatch: (A) zero-padding for extra channels with identity spatial downsampling, (B) projection shortcuts only at dimension-changing transitions, and (C) projection shortcuts everywhere. The authors found B outperforms A and C only marginally improves over B at greater cost. The standard implementation uses Option B: projection shortcuts at stage transitions, identity shortcuts elsewhere.</span>

<span style="font-size: 14px;">The projection shortcut is a 1x1 convolution with appropriate output channels and stride 2, followed by batch normalization (no ReLU). At the Stage 1 to Stage 2 boundary, it maps 64 channels to 128 with stride 2, halving spatial size from $56 \times 56$ to $28 \times 28$. The main branch uses stride 2 in its first 3x3 convolution.</span>

<span style="font-size: 14px;">Within a stage, the second BasicBlock uses a plain identity shortcut since dimensions already match. Stage 1 is special: Conv1 already produces 64 channels and max pool sets spatial size to $56 \times 56$, so both blocks in Stage 1 use identity shortcuts with no projection needed.</span>

---

## <span style="font-size: 16px;">Channel Progression</span>

* <span style="font-size: 14px;">**Conv1:** 3 input channels (RGB) to 64 output channels.</span>
* <span style="font-size: 14px;">**Stage 1:** 64 channels throughout. Both BasicBlocks use identity shortcuts. Spatial: $56 \times 56$.</span>
* <span style="font-size: 14px;">**Stage 2:** 128 channels. First block projects 64 to 128 with stride 2. Spatial: $28 \times 28$.</span>
* <span style="font-size: 14px;">**Stage 3:** 256 channels. First block projects 128 to 256 with stride 2. Spatial: $14 \times 14$.</span>
* <span style="font-size: 14px;">**Stage 4:** 512 channels. First block projects 256 to 512 with stride 2. Spatial: $7 \times 7$.</span>
* <span style="font-size: 14px;">**FC layer:** 512 inputs to $C$ outputs.</span>

<span style="font-size: 14px;">This doubling pattern is deliberate. Each time spatial resolution halves (reducing spatial positions by 4x), channel count doubles (increasing feature depth by 2x). This keeps computational cost per layer roughly constant. The pattern $64 \to 128 \to 256 \to 512$ is consistent across all ResNet variants from ResNet-18 to ResNet-152.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">He et al. (2015), "Deep Residual Learning for Image Recognition," introduced the ResNet family to address the degradation problem: adding more layers to a deep network counterintuitively increased training error. The authors showed this was not overfitting but an optimization difficulty, and their solution was the residual learning framework.</span>

<span style="font-size: 14px;">The paper presents five depths: ResNet-18, 34, 50, 101, and 152. ResNet-18 and 34 use BasicBlocks (two 3x3 convolutions per block). ResNet-50, 101, and 152 use Bottleneck blocks (1x1, 3x3, 1x1 convolutions) to manage computation at greater depth. All variants share the four-stage structure with the same channel progression.</span>

<span style="font-size: 14px;">ResNet won ILSVRC 2015 classification with 3.57% top-5 error on ImageNet using a 152-layer ensemble, the first time a network exceeding 100 layers was successfully trained. Beyond classification, the paper won COCO 2015 detection and segmentation, establishing residual connections as a universal building block now found in Transformers, U-Nets, and diffusion models.</span>

---

## <span style="font-size: 16px;">Parameter Count</span>

<span style="font-size: 14px;">ResNet-18 has approximately 11.7 million parameters, remarkably efficient compared to VGG-16's 138 million. Despite having more layers, ResNet-18 uses 12x fewer parameters while achieving better accuracy.</span>

<span style="font-size: 14px;">Three design choices drive this efficiency. First, all residual blocks use 3x3 kernels (9 weights per channel pair) rather than the 7x7 or 11x11 kernels in earlier architectures. Second, global average pooling replaces the massive FC layers that dominated VGG's parameter count: VGG-16 has three FC layers with 4096 hidden units (~120M of 138M parameters), while ResNet-18 has a single FC from 512 to $C$. Third, Conv1's stride-2 plus max pool means residual blocks operate on $56 \times 56$ maps at most.</span>

<span style="font-size: 14px;">Parameter breakdown: Conv1 contributes $3 \times 64 \times 7 \times 7 = 9{,}408$ weights. Each 3x3 conv contributes $C_{in} \times C_{out} \times 9$ weights. Batch normalization adds $2 \times C$ parameters per layer. Projection shortcuts add $C_{in} \times C_{out}$ parameters each. The FC layer adds $512 \times 1000 = 512{,}000$ parameters for ImageNet.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace a single input through the entire ResNet-18, tracking tensor shape at each step. Batch size 1, ImageNet with 1000 classes.</span>

<span style="font-size: 14px;">**Input:**</span>

$$
x: (1, 3, 224, 224)
$$

<span style="font-size: 14px;">**Conv1:** 7x7 conv, 64 filters, stride 2, padding 3. Spatial: $\lfloor(224 - 7 + 6) / 2\rfloor + 1 = 112$.</span>

$$
\text{After Conv1 + BN + ReLU}: (1, 64, 112, 112)
$$

<span style="font-size: 14px;">**MaxPool:** 3x3, stride 2, padding 1. Spatial: $\lfloor(112 - 3 + 2) / 2\rfloor + 1 = 56$.</span>

$$
\text{After MaxPool}: (1, 64, 56, 56)
$$

<span style="font-size: 14px;">**Stage 1 (2 blocks):** Both blocks use 64-channel 3x3 convolutions with identity shortcuts. Shape unchanged.</span>

$$
\text{After Stage 1}: (1, 64, 56, 56)
$$

<span style="font-size: 14px;">**Stage 2, Block 1:** First conv: 64 to 128, stride 2 (halves spatial). Projection shortcut: 1x1, 64 to 128, stride 2.</span>

$$
\text{After Stage 2, Block 1}: (1, 128, 28, 28)
$$

<span style="font-size: 14px;">**Stage 2, Block 2:** 128 to 128, identity shortcut.</span>

$$
\text{After Stage 2}: (1, 128, 28, 28)
$$

<span style="font-size: 14px;">**Stage 3, Block 1:** First conv: 128 to 256, stride 2. Projection: 1x1, 128 to 256, stride 2.</span>

$$
\text{After Stage 3, Block 1}: (1, 256, 14, 14)
$$

<span style="font-size: 14px;">**Stage 3, Block 2:** 256 to 256, identity shortcut.</span>

$$
\text{After Stage 3}: (1, 256, 14, 14)
$$

<span style="font-size: 14px;">**Stage 4, Block 1:** First conv: 256 to 512, stride 2. Projection: 1x1, 256 to 512, stride 2.</span>

$$
\text{After Stage 4, Block 1}: (1, 512, 7, 7)
$$

<span style="font-size: 14px;">**Stage 4, Block 2:** 512 to 512, identity shortcut.</span>

$$
\text{After Stage 4}: (1, 512, 7, 7)
$$

<span style="font-size: 14px;">**Adaptive Average Pooling:** Collapses each $7 \times 7$ feature map to a single value.</span>

$$
\text{After AvgPool}: (1, 512, 1, 1)
$$

<span style="font-size: 14px;">**Flatten + FC:** Reshape to $(1, 512)$, then linear to 1000 classes.</span>

$$
\text{Output logits}: (1, 1000)
$$

<span style="font-size: 14px;">The input is spatially reduced from $224 \times 224$ to $1 \times 1$ across six steps: Conv1 stride 2 ($224 \to 112$), MaxPool ($112 \to 56$), Stage 2 ($56 \to 28$), Stage 3 ($28 \to 14$), Stage 4 ($14 \to 7$), and AvgPool ($7 \to 1$).</span>

---

## <span style="font-size: 16px;">The ResNet Family</span>

<span style="font-size: 14px;">The five original variants share the four-stage structure but differ in depth and block type. ResNet-18 and 34 use BasicBlocks (two 3x3 convolutions). ResNet-50, 101, and 152 use Bottleneck blocks (1x1, 3x3, 1x1 convolutions).</span>

<span style="font-size: 14px;">Blocks per stage by depth: ResNet-18 uses [2, 2, 2, 2]. ResNet-34 uses [3, 4, 6, 3]. ResNet-50 uses [3, 4, 6, 3] with Bottleneck blocks. ResNet-101 uses [3, 4, 23, 3]. ResNet-152 uses [3, 8, 36, 3]. Deeper variants concentrate additional blocks in Stage 3, where intermediate features benefit most from extra depth.</span>

<span style="font-size: 14px;">Bottleneck blocks change internal channel progression. In Stage 3 (256 channels), a Bottleneck reduces from 256 to 64 via 1x1 conv, processes at 64 channels via 3x3 conv, then expands from 64 to 256 via 1x1 conv. The expansion ratio of 4 is fixed, so the final feature dimension for Bottleneck ResNets is $512 \times 4 = 2048$ instead of 512.</span>

<span style="font-size: 14px;">The paper showed clear gains with depth: ResNet-18 achieves 27.88% top-1 error on ImageNet, ResNet-34 achieves 25.03%, and ResNet-152 achieves 21.43%, vindicating the claim that residual connections solve the degradation problem.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong number of blocks per stage.** ResNet-18 uses exactly [2, 2, 2, 2]. Using [3, 4, 6, 3] gives ResNet-34/50 instead. The block count per stage is the primary differentiator between ResNet variants.</span>
* <span style="font-size: 14px;">**Forgetting projection shortcuts at stage transitions.** When spatial dimensions halve and channels double, the identity shortcut $y = F(x) + x$ fails because $F(x)$ and $x$ have different shapes. The first block of stages 2, 3, and 4 must use a 1x1 convolution with stride 2. Omitting this causes a shape mismatch error.</span>
* <span style="font-size: 14px;">**Wrong channel counts.** The correct progression is $64 \to 128 \to 256 \to 512$. Both convolutions within a BasicBlock must use the same channel count; the first conv sets it, and the second maintains it.</span>
* <span style="font-size: 14px;">**Missing the initial Conv1 layer.** ResNet does not start directly with residual blocks. The 7x7 convolution (stride 2) + BN + ReLU + 3x3 max pool (stride 2) is essential. Skipping Conv1 means residual stages receive full-resolution 3-channel input, which is completely wrong.</span>
* <span style="font-size: 14px;">**Forgetting the FC layer or average pooling.** The classifier head requires adaptive average pooling to collapse spatial dimensions, flattening, and a linear layer. Omitting average pooling and feeding $512 \times 7 \times 7 = 25{,}088$ inputs to the FC layer instead of 512 is a common mistake.</span>
* <span style="font-size: 14px;">**Applying stride in the wrong layer.** At stage transitions, stride-2 downsampling goes in the first 3x3 convolution of the first BasicBlock. Applying it in the second convolution or in a separate pooling layer is incorrect. The projection shortcut must also use stride 2 to match.</span>
* <span style="font-size: 14px;">**Missing batch normalization on the projection shortcut.** The projection is a 1x1 conv followed by BN, with no ReLU. The ReLU is applied after the addition $F(x) + W_s x$, not within the shortcut path.</span>
* <span style="font-size: 14px;">**Placing ReLU before the addition.** The correct order is: compute $F(x)$, compute shortcut, add them, then ReLU. Applying ReLU to $F(x)$ before adding the shortcut forces the skip signal through a nonlinearity. The paper is explicit: $y = \text{ReLU}(F(x) + x)$.</span>

---