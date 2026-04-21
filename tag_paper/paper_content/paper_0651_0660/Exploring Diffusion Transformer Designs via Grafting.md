Title: Exploring Diffusion Transformer Designs via Grafting
Abstract: Designing model architectures requires decisions such as selecting operators (e.g., attention, convolution) and configurations (e.g., depth, width). However, evaluating the impact of these decisions on model quality requires costly pretraining, limiting architectural investigation. Inspired by how new software is built on existing code, we ask: can new architecture designs be studied using pretrained models? To this end, we present grafting, a simple approach for editing pretrained diffusion transformers (DiTs) to materialize new architectures under small compute budgets. Informed by our analysis of activation behavior and attention locality, we construct a testbed based on the DiT-XL/2 design to study the impact of grafting on model quality. Using this testbed, we develop a family of hybrid designs via grafting: replacing softmax attention with gated convolution, local attention, and linear attention, and replacing MLPs with variable expansion ratio and convolutional variants. Notably, many hybrid designs achieve good quality (FID: 2.38-2.64 vs. 2.27 for DiT-XL/2) using < 2% pretraining compute. We then graft a text-to-image model (PixArt-Σ), achieving a 1.43× speedup with less than a 2% drop in GenEval score. Finally, we present a case study that restructures DiT-XL/2 by converting every pair of sequential transformer blocks into parallel blocks via grafting. This reduces model depth by 2× and yields better quality (FID: 2.77) than other models of comparable depth. Together, we show that new diffusion model designs can be explored by grafting pretrained DiTs, with edits ranging from operator replacement to architecture restructuring. Code and grafted models: grafting.stanford.edu.

Section: Introduction
Model architecture design plays a central role in machine learning, alongside data, algorithms, compute, and benchmarks. It defines a learnable function and entails key decisions, including the choice of operators (e.g., attention, convolution) and configurations (e.g., model depth, width). Despite this, insight into architectures-what works and what doesn't-is difficult to obtain due to the prohibitive costs of training models from scratch, especially in today's foundation model era. As a result, studying new architectures remains a challenge, particularly for generative models. Much like how new software is built on existing code rather than written from scratch, can pretrained models serve as scaffolds for studying new architectures? In this work, we investigate architectural editing of pretrained models to study new architecture designs. We focus on diffusion transformers (DiTs), a class of generative transformers widely used for image and video generation [1,2,3]. A pretrained model implements a computational graph to perform tasks such as image or video generation. Given a new architectural idea and a pretrained model, we investigate whether the idea can be materialized by modifying its computational graph under small compute budgets. For example, one might hypothesize that a convolutional design could replace Multi-Head Attention (MHA) or Multi-Layer Perceptron (MLP) in a DiT. A simple way to materialize this idea is to replace MHA or MLP operators with a convolutional operator, while preserving model quality. This raises two key questions: (Q1) operator initialization: How to initialize a new operator before integrating it into the computational graph? (Q2) error accumulation: How to mitigate error propagation as multiple operators are integrated into the computational graph?
To address these questions, we present graftingfoot_0 , a simple two-stage approach to architecture editing (Fig. 1). Grafting proceeds as follows: (i) activation distillation: This stage transfers the functionality of the original operator to the new one by distilling its activations using a regression objective. (ii) lightweight finetuning: This stage mitigates error propagation caused by integrating multiple new operators by finetuning using limited data. Architectural editing spans multiple strategies-adding, removing, and replacing [5,6,7] operators. We focus on operator replacement as the core strategy: swapping one operator for another. Other strategies can be viewed as special cases of replacement.
The space of architectural editing is vast, raising a practical question: what types of replacements should we study? We first establish a self-grafting baseline, where we replace all MHA and MLP operators in DiT-XL/2 with randomly initialized counterparts. Despite the scale of this intervention, our grafting procedure achieves near-baseline model quality using under 1% of pretraining compute. Building on this, we focus on replacing existing operators with efficient alternatives, aiming to reduce model FLOPs while preserving quality. We also explore replacements that increase model FLOPs to examine broader design choices. To study this systematically, we construct a testbed based on DiT-XL/2 and define a set of architectural edits to evaluate how different grafting schemes affect model quality. We organize our design space along four axes: (1) which operator to replace (e.g., MHA, MLP); (2) what to replace it with (e.g., convolutions); (3) how to select layers for replacement (e.g., all layers); and (4) replacement ratio (full vs. partial). We focus on replacing MHA and MLP operators, as they account for a large fraction of model FLOPs. Replacements for MHA and MLP operators are motivated by empirical findings and prior architectural designs: our locality analysis supports local operators for MHA, while for MLP, we adopt ideas from prior work [8,9,10].
We validate our grafting approach in increasingly challenging generative modeling setups:
Result I: Grafting yields hybrid architecture designs with good quality for class-conditional image generation (Sec. 4.2). We validate grafting using our testbed. For MHA (softmax attention), we explore several alternatives: local gated convolution (Hyena-SE, and our proposed Hyena-X/ Hyena-Y), local attention (sliding window), and linear attention (Mamba-2). For MLPs, alternatives include MLPs with variable expansion ratio (ratios=3, 6), and a convolutional variant (Hyena-X). Interestingly, several interleaved hybrid architecture designs achieve FID scores between 2.38 and 2.64 (DiT-XL/2 256x256 baseline: 2.27), showing that grafting can construct good quality hybrids (Tab. 4) 2 . Grafting is simple and lightweight: each experiment completes in under 24 hours on 8×H100 GPUs, using less than 2% of pretraining compute.
Result II: We construct efficient hybrid architectures for high-resolution text-to-image (T2I) generation via grafting (Sec. 5). We validate grafting in a challenging, real-world setting: 2048×2048 resolution T2I generation using PixArt-Σ (DiT) [11]. This setting reflects key challenges: it operates on long sequences (16,384 tokens), involves a multimodal setup with text conditioning, and lacks training data. We target MHA operators for grafting, as they account for over 62% of generation latency. Using 12k synthetic data, our grafted model achieves a 1.43× speedup with <2% drop in GenEval score (47.78 vs. 49.75), showing that grafting scales to high-resolution, T2I generation.
Case Study: Converting model depth to width via grafting (Sec. 6). Motivated by our MLP grafting results, we rewire DiT-XL/2 by parallelizing every pair of transformer blocks, as modern GPUs favor parallel over sequential computation. This reduces model depth by 2× (28→14). The grafted model achieves FID=2.77, outperforming other models of comparable depth. To our knowledge, this is the first attempt to convert sequential transformer blocks into parallel in pretrained DiTs, enabling architectures to be restructured.
this section cite: ['b0', 'b1', 'b2', 'b4', 'b5', 'b6', 'b1', 'b7', 'b8', 'b9', 'b10']

Section: Prerequisites
Diffusion models (DMs). DMs generate data samples by iteratively denoising random noise. This sampling process inversely mirrors the forward data corruption mechanism: z t = α t z + σ t ϵ where z = E(x) ∼ q(z) with E representing a pretrained encoder and x the data variable. The noise term ϵ follows the prior distribution N (0, I). The transition kernel from time 0 to t is given by q t (z t |z) = N (z t ; α t z, σ 2 t I). The choice of α t and σ t defines the diffusion variant, such as variancepreserving [12], or flow matching [13]. The training objective [12] is as follows:
L DM (ϕ) = E q(t)q(z,c)N (ϵ;0,I) [∥ϵ -ϵ ϕ (z t , t, c)∥ 2 2 ],(1)
where q(t): time sampling distribution, and q(z, c): joint distribution of latent z and condition c.
this section cite: ['b11', 'b12', 'b11']

Section: Diffusion transformers (DiTs).
DiTs model the diffusion process by patchifying the input-noised images or latent-into a sequence of 1D tokens with positional embeddings. These tokens are processed through transformer blocks comprising self-attention, feedforward layers, residual connections, and normalization layers. DiTs also incorporate conditioning signals, such as noise timestep (t), class labels (c), or natural language prompts, enabling controllable generation [1,14].
this section cite: ['b0', 'b13']

Section: Datasets and evaluation metrics.
For class-conditional image generation, we use ImageNet-1K [15]. We follow [1] and report Inception Score (IS), FID, sFID, Precision, and Recall using 50k generated samples (250 steps DDPM, cfg=1.5). For text-to-image generation, we report GenEval score [16].
this section cite: ['b14', 'b0', 'b15']

Section: Grafting Diffusion Transformers

this section cite: []

Section: Two-Stage Grafting Approach
Grafting aims to materialize new architectures by editing a pretrained model's computational graph.
Given that we focus on replacing existing operators with alternatives, this raises two questions:
(Q1) How should a new operator be initialized before being integrated into the computational graph? Stage 1: Activation distillation. We cast initialization as a regression task. Operators in a DiT block process [B, N, D] inputs (batch, sequence, hidden) and output tensors of the same shape. Given a pretrained operator f l ϕ at layer l, we learn a new operator g l θ that approximates f l ϕ [17]. Since DiT activations are continuous and smooth, this can be posed as a regression problem:
L(θ) = E q(t)q(z,c)qt(zt|z) L reg (g l θ (z t , t, c), f l ϕ (z t , t, c))(2)
where q(z, c) is the joint distribution of latent representation z and condition c, q(t) is the time sampling distribution, and q t (z t |z) is the transition kernel from time 0 to t. L reg is a regression objective such as L 2 . In practice, a good initialization requires as few as 8k samples.
(Q2) How can we mitigate error propagation as multiple operators are integrated into the computational graph? Stage 2: Lightweight finetuning. As more operators are replaced, initialization errors propagate, leading to deviations from the pretrained model's behavior. We apply end-to-end finetuning with limited data to mitigate cumulative errors from stage 1. The fine-tuning objective is given in Equation 1. In practice, we find that competitive performance can be recovered using only 10% of the training data, even when replacing all MHA or MLP layers in DiT-XL/2.
this section cite: ['b16']

Section: Self-grafting Baseline
Prior to studying new architectural designs, we introduce self-grafting, a simple control setup where existing operators (e.g., MHA, MLP) are replaced with identical operators whose weights are randomly initialized. This preserves the computational graph's structure-operator types, receptive fields, and parameter count-while altering the computation performed. Self-grafting serves three purposes: (1) to assess the grafting procedure without architectural changes, (2) to provide a baseline for comparing replacements, and (3) to study factors affecting performance, such as data scale, regression objectives, and hyperparameters.
this section cite: []

Section: Activation Behavior Analysis and Self-grafting Results
We begin by analyzing the activation behavior of MHA and MLP operators across all layers in DiT-XL/2. In both cases, we observe large variance in activation values, particularly in deeper layers (Tab. 1 (i, ii)). When using regression-based distillation for Stage 1, these outliers affect optimization, particularly under the commonly used L 2 objective which penalizes all errors quadratically. This motivates a closer look at regression objectives. We study three regression objectives with different level of sensitivity to outliers-L 2 , L 1 , and Huber [18]-using a self-grafting setup. We select five representative layers (l = 1, 8, 17, 27, 28) for both MHA and MLP, spanning a range of activation  ). This study shows that high-quality initialization can be achieved by choosing operator-specific regression objectives.
values. Each operator is trained with 8K ImageNet-1K [15] samples, for 200 epochs with batch size 64 and learning rate 1e-4. We use δ = 1.0 for Huber objective. We then integrate the initialized operators into the pretrained DiT-XL/2 and evaluate quality without any finetuning.
High-quality initialization can be achieved by choosing operator-specific regression objectives.
Stage 1 Stage 2 IS ↑ FID ↓ sFID ↓ Prec. ↑ Rec. ↑ Baseline 278.20 2.27 4.60 0.83 0.57 MHA (Full Self-grafting) Random Init. 1.66 289.23 154.00 0.00 0.00 0.63% -117.68 16.78 13.69 0.60 0.61 0.63% 0.63% 148.56 11.26 11.10 0.66 0.60 0.63% 5.0% 270.39 2.70 5.46 0.81 0.57 0.63% 10.0% 287.81 2.49 4.71 0.83 0.56 MLP (Full Self-grafting) Random Init. 1.27 314.72 204.99 0.00 0.00 0.63% 10.0% 277.72 2.54 4.52 0.83 0.57
Table 2: Full self-grafting (Stage 2) results (DiT-XL/2).
We report results after replacing all 28 MHA and MLP operators using different amounts of training data. As we increase the training data from 0.63% (8k) to 10.0% (128k), FID improves consistently. Using only 10% of the training data, near-baseline performance is achieved: FID 2.49 for MHA and 2.54 for MLP.
As shown in Tab. 1 (iii,iv), the choice of the regression objective affects performance. For MHA, L 1 achieves the best FID (2.51), followed by Huber (2.55) and L 2 (2.58). For MLPs, L 2 performs best (2.33), while L 1 underperforms (2.83); notably, MLPs have 2× more parameters than MHA which explains its robustness to outliers [19]. This shows that high-quality initialization requires tailored, activation-aware strategies. Further, we evaluate validation loss on held-out samples. For MHA, L 1 achieves the lowest loss; for MLPs, L 2 achieves the lowest loss for all blocks (See Sec. C.3).
Full self-grafting with 10% data achieves near-baseline performance.
We extend our study to replace all MHA and MLP operators in DiT-XL/2 under the self-grafting setup and evaluate the effect of data on recovery (Tab. 2). For MHA, replacing all 28 layers without adaptation results in a noticeable performance drop, but Stage 2 (lightweight fine-tuning) is highly effective: using just 10% of the training data (128k samples), we achieve an FID of 2.53 vs. 2.27 for the baseline. Similarly, full MLP self-grafting with 10% data yields an FID of 2.54. We use batch size 256, learning rate 1e -4 , and 30k iterations. In both cases, the quality is within 0.3 FID of the baseline, showing that full self-grafting is feasible under modest data and compute budgets. MHA scales quadratically with sequence length, making it a computational bottleneck. A natural idea is to replace it with local operators, such as convolution or local attention. However, this will fail if the model relies on long-range dependencies: for example, replacing all MHA operators in DiT-XL/2 with a sliding window attention degrades FID from 2.27 to 53.9. To guide grafting, we quantify MHA locality using a simple band-k metric. Given an attention matrix A ∈ R N ×N and a band size of k, we define a bi-directional band indicator matrix B k ∈ R N ×N as:
this section cite: ['b17', 'b14', 'b18']

Section: Locality Analysis of Self-attention
(B k ) i,j = 1, if |i -j| ≤ k 0, otherwise
Then, locality within a band of size k is computed as:
L k = 1 N i,j A i,j (B k ) i,j(3)
We compute L k for all 28 MHA operators in DiT-XL/2 using 50-step DDIM sampling (250 ImageNet samples, sequence length 256, cfg scale 1.5), averaging across timesteps and samples. As shown in Fig. 2, MHA is largely local: on average, for k=32, 15 out of 28 layers attend to more than 50% attention mass within the band. The first few layers (l=1,2) display non-local attention patterns. Our analysis provides guidance for replacing MHA operators with efficient local operators.
this section cite: []

Section: Experiments I: Hybrid Architectures via Grafting

this section cite: []

Section: Testbed and Experiment setup
Building on our self-grafting results, we now ask: can we maintain model quality when existing operators are replaced with efficient alternatives? To investigate this, we study the grafting procedure along four design axes:
1. operator type to replace -MHA or MLP 2. replacement operator type -such as convolutions 3. layer selection strategy -replace operators in all layers or use heuristic-based selection 4. replacement ratio -full or partial
We construct a testbed to systematically evaluate how design decisions affect generative quality under grafting. We focus on efficient replacements that reduce FLOPs, but also include higher-FLOP variants to explore a broader range of architectural edits. We target MHA and MLP operators, which account for a significant portion of FLOPs in DiTs compared to other operators (e.g., normalization, activation, residuals). The rationale for replacing MHA or MLP operators is grounded in both empirical and architectural considerations: for MHA, our attention locality analysis (Fig. 2) motivates the use of local operators; for MLP, we leverage prior architecture ideas [8,20,21,22,9,10]. Given a replacement operator, the decision to graft it to a model with L transformer layers spans a space of 2 L configurations. To make this tractable, we study two layer selection strategies: full (replace all operators) and interleaved (replace operators in a repeating pattern) strategies. The latter is inspired by striped transformer designs [23,24,25]. Our testbed is detailed in Tab. 3. We introduce Hyena-X and Hyena-Y-two new efficient gated convolution operators designed as drop-in replacements for MHA. While our study includes several off-the-shelf efficient alternatives, we also contribute new operator designs motivated by our MHA locality analysis. This allows us to test novel architectural ideas via grafting, broadening our study. Both Hyena-X and Hyena-Y are local gated convolutions composed of dense, short causal depth-wise 1D convolutions. Fig. 3 (left) illustrates their structure. We also adapt Hyena-X as an MLP alternative by applying it along the channel dimension. Hyena-X and Hyena-Y scale linearly with sequence length, compared to the quadratic scaling of MHA. Operator details are provided in Sec. G. We provide FLOP calculation for both operators in Sec. H.1.
x v k q Hyena-X x v k q Hyena-Y Short Explicit Convolution Dense Hadamard Gating
Experiment setup. For our hybrid experiments, we mostly use the hyperparameters determined from our self-grafting studies (Sec. 3.2).
this section cite: ['b7', 'b19', 'b20', 'b21', 'b8', 'b9', 'b22', 'b23', 'b24']

Section: Stage 1: Operator initialization.
For each new operator, we perform activation distillation using 8K ImageNet-1K samples. Each operator is trained for 200 epochs with a batch size of 64 and an initial learning rate of 1e-4. We pre-extract and store all regression features. All operators can be initialized in parallel. Each operator's training completes in under 30 minutes on a single H100 GPU. Experiment details for stage 1 are included in Sec. C.1.
this section cite: []

Section: Stage 2: Lightweight finetuning.
For all experiments in Table 3, we use 10% of the ImageNet-1K training data and train for 50K steps. We use a batch size of 256, linearly warming up the learning rate to 1e-4 over 1000 steps. Experiments typically complete in under 10 hours on 8×H100 GPUs. For specific ablations on increasing data, such as those involving 20% data or 100K steps, runtimes extend up to 24 hours (<2% pretraining compute). We provide experiment details in Sec. C.1.
Operator Type: Which operator types are we replacing?
this section cite: []

Section: MHA, MLP
Efficient Alternative: What do we replace it with?
this section cite: []

Section: MHA
Convolutions: Hyena-SE [24], Hyena-X/ Hyena-Y (Ours) K=4, causal Local Attention: Sliding Window Attention (SWA) [26,27] w=4, bidirectional Linear Attention: Mamba-2 [28] ds=64, E=2
this section cite: ['b23', 'b25', 'b26', 'b27']

Section: MLP Variable expansion ratio r=3,6
Hyena-X (Ours) r=2, K=4, causal, mix channels
this section cite: []

Section: Layer Selection:
In which layers is the operator replaced?
this section cite: []

Section: Full Replace the operator in all layers
Interleaved Replace the operator in a repeating pattern (e.g., every 2 or 3 out of 4)
Replacement Ratio: What percentage of operators are replaced?
50%, 75%, 100%  MHA results. Replacing MHA operators in DiT-XL/2 via grafting yields strong quality-efficiency tradeoffs. We discuss our key insights below:
• Surprising effectiveness of operators with smaller receptive fields under interleaved grafting. Our findings highlight that at 50% interleaved replacement, several alternatives-including SWA, Hyena-X/Y, and Mamba-2-consistently achieve FID scores within 0.5 of the baseline (2.27). The minimal FID drop observed especially with the SWA and Hyena variants, despite their limited receptive field (K=4, w=4), aligns with our locality analysis (Section 3.4).
• Replacement strategy: Interleaved vs. Full. Performance generally declines when increasing interleaved replacement from 50% to 75%. However, SWA remains effective at 75% interleaved replacement (FID=3.09). At 100% replacement, performance sharply degrades (all FIDs > 75). This trend aligns with our locality analysis, indicating that only a subset of layers are local and amenable to grafting.
• Ablations on data scale and layer selection. We study two factors under 50% MHA replacement. (i) Increasing fine-tuning data from 10% to 20% improves FID across all variants (e.g., Hyena-X: 2.74 → 2.61; SWA: 2.67 → 2.62, Mamba-2: 2.65→ 2.55) (Fig. 4 (a)). (ii) Under 50% replacement, we compare Hyena-X (interleaved) to three targeted heuristics: top-local (layers with highest band-k values), low-local (layers with lowest band-k values), and deep (last 14 layers). Interleaved yields the best FID (2.74), followed by top-local (3.02), low-local (3.18), and deep (4.00). These results confirm that interleaving is effective, and our band-k metric identifies layers that are more amenable to grafting (Fig. 4 (b)).
this section cite: []

Section: MLP results.
Replacing MLP operators via grafting is effective. We discuss our key insights below:
• Variable expansion ratio MLPs are effective under full replacement. MLP alternatives with expansion ratio r=3 and r=6 demonstrate good quality under all replacement ratios. Even under full (100%) replacement, both variants maintain good performance, with r=3 achieving FID=2.66. This highlights that MLP width is a robust dimension for grafting.
• Convolutional alternatives. Hyena-X which combines dense and local channel mixing, performs competitively at 50% replacement (FID=2.63) but degrades at higher ratios, suggesting that such operators are only effective at moderate ratios.
Takeaway 1: Grafting is effective for constructing efficient hybrid architectures with good generative quality under small compute budgets. Interleaved designs are particularly effective.
Ratio IS ↑ FID ↓ sFID ↓ Prec. ↑ Rec. ↑ ∆FLOPs op. ↓ ∆FLOPs f t. ↓ ∆Param ↓ Baseline -278.20 2.27 4.60 0.83 0.57 ---MHA Grafting Random Init. 100% 1.66 289.23 154.00 0.00 0.00 ---Self-grafting 100% 287.81 2.49 4.71 0.83 0.56 ---Hyena-SE (K=4) 50% 274.73 2.73 5.05 0.82 0.56 -49.52% +0.13% +0.22% 75% 231.15 3.62 6.04 0.81 0.54 -74.27% +0.20% +0.33% 100% ✗ ✗ ✗ ✗ ✗ -99.03% +0.26% +0.43% Hyena-X (K=4) 50% 273.30 2.74 5.03 0.83 0.56 -49.90% +0.13% +0.16% 75% 229.11 3.69 6.10 0.81 0.53 -74.85% +0.20% +0.24% 100% ✗ ✗ ✗ ✗ ✗ -99.81% +0.26% +0.33% Hyena-Y (K=4) 50% 273.37 2.72 5.02 0.83 0.55 -49.52% 0.00% +0.05% 75% 228.99 3.66 5.95 0.81 0.53 -74.27% 0.00% +0.08% 100% ✗ ✗ ✗ ✗ ✗ -99.03% 0.00% +0.11% SWA (w=4) 50% 280.62 2.67 4.90 0.83 0.56 -48.24% 0.00% 0.00% 75% 249.99 3.09 5.54 0.82 0.55 -72.36% 0.00% 0.00% 100% ✗ ✗ ✗ ✗ ✗ -96.48% 0.00% 0.00% Mamba-2 (d s =64, E=2) 50% 285.08 2.65 4.84 0.83 0.55 -37.59% +77.89% +28.02% 75% 257.66 3.02 5.48 0.82 0.53 -56.38% +116.83% +42.04% 100% ✗ ✗ ✗ ✗ ✗ -75.17% +155.77% +56.05% MLP Grafting Random Init. 100% 1.27 314.72 204.99 0.00 0.00 ---Self-grafting 100% 277.72 2.54 4.52 0.83 0.57 ---Exp. ratio ↓ (r=3) 50% 272.14 2.53 4.51 0.83 0.57 -12.50% 0.00% -12.50% 75% 279.72 2.61 4.61 0.83 0.56 -18.75% 0.00% -18.75% 100% 252.11 2.66 4.57 0.81 0.57 -25.00% 0.00% -25.00% Exp. ratio ↑ (r=6) 50% 278.00 2.38 4.50 0.83 0.58 +25.00% 0.00% +25.00% 75% 277.94 2.37 4.48 0.82 0.58 +37.50% 0.00% +37.50% 100% 276.86 2.42 4.50 0.82 0.58 +50.00% 0.00% +50.00% Hyena-X (r=2,K=4) 50% 265.60 2.64 4.66 0.83 0.56 +0.01% 0.00% +0.02% 75% 226.13 3.26 4.79 0.81 0.55 +0.02% 0.00% +0.03% 100% ✗ ✗ ✗ ✗ ✗ +0.02% 0.00% +0.03%
Table 4: Generation quality and efficiency metrics for MHA and MLP grafting. We report quality (IS, FID, sFID, Precision, Recall) and efficiency (∆FLOPs and ∆Param) results. Baseline refers to DiT-XL/2. For each alternative, setups that maintain FID within 0.5 of the baseline and offer the largest FLOPs reduction (or smallest FLOPs increase) are highlighted . ✗ denotes setups with poor generation (FID > 50). ∆FLOPs and ∆Param denote the percentage change in operator FLOPs and parameters, respectively. For MHA, total cost is split into ∆FLOPs op. (softmax attention, gating) and ∆FLOPs ft. (QKV/output projections, featurizers). We do not use this decomposition for MLP variants. Mamba-2 incurs higher ∆FLOPs ft. due to additional projections. FLOP expressions are provided in Sec. H.1. Key result: Many interleaved designs achieve good quality generation (FID within 0.5 of baseline). All experiments use 10% training data and <1% pretraining compute.
this section cite: []

Section: Experiments II: Grafting Text-to-Image Diffusion Transformers
We apply grafting to a more challenging setting: high-resolution text-to-image generation with PixArt-Σ [11]. This presents three challenges: (1) long sequences (16,384 tokens for 2048×2048 resolution), (2) a multimodal setup with text conditioning, and (3) lack of publicly available training data. These factors make PixArt-Σ a representative setting for evaluating grafting under real-world constraints. PixArt-Σ contains 28 transformer layers similar to DiT-XL/2.
Experiment setup. We replace MHA operators in PixArt-Σ with Hyena-X via grafting, as MHA accounts for over 62% of generation latency. Hyena-X was chosen based on its good quality-efficiency tradeoff in the ImageNet-1K setup, achieving FID 2.61 with 20% data (see Fig. 4 (b)). Interleaved grafting is applied for layers 8, 10, 12, 14, 16, 18, and 20-27; empirically, we found that layers 20-27 can be replaced without significant quality drop. For grafting, we created a small, uncurated synthetic dataset of 12k image-text pairs. The text prompts for this dataset were sampled from the 30k publicly released evaluation set.  Can we rewire two sequential transformer blocks to run in parallel? Our MLP grafting results showed that MLPs are amenable to grafting, even at 100% replacement with an expansion ratio of r = 6, demonstrating that wider computation within an operator is feasible. This success, combined with the fact that modern GPUs favor parallel over sequential computation, motivates a broader question: can we convert deeper, sequential DiT computations into wider, parallel ones via grafting while maintaining quality? To explore this, we rewire DiT-XL/2 by parallelizing every pair of sequential transformer blocks-each pair receives the same input, and their outputs are merged via a linear projection. This reduces model depth by 2× (28 → 14) with a 6% increase in parameters.
Experiment Setup. The rewiring schematic is shown in Fig. 5. We use DiT-XL/2. Stage 1: Activation distillation. Each parallel pair was initialized via activation distillation using L1 regression. The weights for each block in the parallel pair were initialized from their corresponding pre-trained weights, rather than random initialization. Similar to our previous experiments, 8k ImageNet-1K samples were used for this stage. Stage 2: Lightweight finetuning. Given the architectural restructuring, finetuning was performed using 25% of the training data. The learning rate was linearly warmed up to 1e-4 and halved at 75k and 150k iterations. Additional details can be found in Sec. D.
this section cite: ['b10']

Section: Results.
The goal of this study is to evaluate generative quality (FID) vs. model depth. We report results in Tab. 6. To contextualize our findings, we compare against two categories: (i) DiTs trained from scratch at lower depth, and (ii) pruning methods [31,32]. Our 14-layer grafted model achieves an FID of 2.77-surpassing DiT variants trained from scratch with similar or increased depth, including DiT-L/2 (depth 24, FID 3.73) and U-ViT-L (depth 21, FID 3.44). It also outperforms pruning baselines such as TinyDiT-D14 with masked knowledge distillation (depth 14, FID 2.86) and BK-SDM (depth 14, FID 7.43), though these baselines have fewer parameters (340M) compared to the grafted variants (712M).
Takeaway 3: Grafting enables architectural restructuring at the transformer block level, allowing model depth to be traded for width. 3). For pruning and grafting setups, we report speedup with respect to DiT-XL/2 (depth=28). Off-the-shelf DiT-L/2, U-ViT-L, and DiT-B/2 scores, along with pruning baselines (BK-SDM, TinyDiT-D14, and TinyDiT-D14 w/ MKD), are sourced from [32]. MKD refers to Masked Knowledge Distillation, a recovery method used in [32]. ¶ Speedup is measured for a single forward pass on an Nvidia H100 (batch size=2). (details in Sec. D). Key result. Our grafted models achieve better generative quality at depth=14, surpassing baselines in FID, IS, Precision, and Recall.
this section cite: ['b30', 'b31', 'b31', 'b31']

Section: Method
Depth A.R Iters IS ↑ FID ↓ sFID ↓ Prec. ↑ Recall ↑ Speedup ↑ Params ↓ DiT-L/
this section cite: []

Section: Conclusion and Discussion
In this work, we introduced grafting, a simple approach to architecture editing. We constructed hybrid models by replacing self-attention and MLPs with efficient alternatives, achieving competitive quality (FID 2.38-2.64 vs. 2.27 baseline). We then applied grafting to a high-resolution text-to-image model (PixArt-Σ), yielding a 43% speedup with less than a 2% drop in GenEval score. We then used grafting to restructure DiT-XL/2, converting every pair of sequential transformer blocks into parallel, reducing model depth by half and yielding better quality (FID 2.77) among 14-layer DiTs. These results demonstrate grafting's utility in both short-and long-context settings (e.g., ImageNet-1K and PixArt-Σ, respectively), and for architecture restructuring. Overall, grafting proves to be a lightweight approach for materializing diffusion transformer designs under small compute budgets.
Related work and discussion. Due to page limit, we discuss related work in Supp. A. Further, to demonstrate the generalization of grafting, we graft an LLM (Qwen3-4B [34])-a generative model (autoregressive), model architecture, and data modality distinct from diffusion-based image generation with DiTs (Supp. F). We discuss broader impact, limitations and applications in Supp. I.
this section cite: ['b33']

Section: References
Ref_id:b0 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b1 Title: Video generation models as world simulators Year: (2024)
Ref_id:b2 Title: Photorealistic video generation with diffusion models Year: (2023)
Ref_id:b3 Title: Plant grafting: new mechanisms, evolutionary implications Year: (2002)
Ref_id:b4 Title: On low-rank linearizing of large language models Year: (2025)
Ref_id:b5 Title: The mamba in the llama: Distilling and accelerating hybrid models Year: (2024)
Ref_id:b6 Title: Transformers to ssms: Distilling quadratic knowledge to subquadratic models Year: (2024)
Ref_id:b7 Title: Monarch mixer: A simple sub-quadratic gemmbased architecture Year: (2023)
Ref_id:b8 Title: Sparse upcycling: Training mixture-of-experts from dense checkpoints Year: (2023)
Ref_id:b9 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b10 Title: Pixart-: Weak-to-strong training of diffusion transformer for 4k text-to-image generation Year: (2024)
Ref_id:b11 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b12 Title: Maximilian Nickel, and Matt Le. Flow matching for generative modeling Year: (2022)
Ref_id:b13 Title: Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis Year: (2023)
Ref_id:b14 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b15 Title: Geneval: An object-focused framework for evaluating text-to-image alignment Year: (2023)
Ref_id:b16 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b17 Title: Robust Estimation of a Location Parameter Year: (1964)
Ref_id:b18 Title: Benign overfitting in linear regression Year: (2020)
Ref_id:b19 Title: Solar 10.7 b: Scaling large language models with simple yet effective depth up-scaling Year: (2024)
Ref_id:b20 Title: Compute better spent: Replacing dense layers with structured matrices Year: ()
Ref_id:b21 Title: The impact of depth on compositional generalization in transformer language models Year: (2024)
Ref_id:b22 Title: Mechanistic design and scaling of hybrid architectures Year: (2024)
Ref_id:b23 Title: Systems and algorithms for convolutional multi-hybrid language models at scale Year: (2025)
Ref_id:b24 Title:  Year: (2025)
Ref_id:b25 Title: The long-document transformer Year: (2020)
Ref_id:b26 Title: Generating long sequences with sparse transformers Year: (2019)
Ref_id:b27 Title: Transformers are ssms: Generalized models and efficient algorithms through structured state space duality Year: (2024)
Ref_id:b28 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b29 Title: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b30 Title: Bk-sdm: A lightweight, fast, and cheap version of stable diffusion Year: (2024)
Ref_id:b31 Title: Tinyfusion: Diffusion transformers learned shallow Year: (2024)
Ref_id:b32 Title: All are worth words: A vit backbone for diffusion models Year: (2023)
Ref_id:b33 Title: Qwen3 technical report Year: (2025)
Ref_id:b34 Title: Sana: Efficient high-resolution image synthesis with linear diffusion transformers Year: (2024)
Ref_id:b35 Title: Diffusion models without attention Year: (2024)
Ref_id:b36 Title: Scalable diffusion models with state space backbone Year: (2024)
Ref_id:b37 Title: Zigma: Zigzag mamba diffusion model Year: (2024)
Ref_id:b38 Title: Dim: Diffusion mamba for efficient high-resolution image synthesis Year: (2024)
Ref_id:b39 Title: Dig: Scalable and efficient diffusion models with gated linear attention Year: (2024)
Ref_id:b40 Title:  Year: (2025)
Ref_id:b41 Title: A survey on video diffusion models Year: (2024-11-22)
Ref_id:b42 Title: Matten: Video generation with mamba-attention Year: (2024)
Ref_id:b43 Title: Lingen: Towards high-resolution minute-length text-to-video generation with linear computational complexity Year: (2024)
Ref_id:b44 Title: Scaling diffusion transformers to 16 billion parameters Year: (2024)
Ref_id:b45 Title: Star: Synthesis of tailored architectures Year: (2025)
Ref_id:b46 Title: Scaling inference-efficient language models Year: ()
Ref_id:b47 Title: Clear: Conv-like linearization revs pre-trained diffusion transformers up Year: (2024)
Ref_id:b48 Title: Linfusion: 1 gpu, 1 minute, 16k image Year: (2024)
Ref_id:b49 Title: Edit: Efficient diffusion transformers with linear compressed attention Year: (2025)
Ref_id:b50 Title: Ffn fusion: Rethinking sequential computation in large language models Year: (2025)
Ref_id:b51 Title: Progressive network grafting for few-shot knowledge distillation Year: (2021)
Ref_id:b52 Title: Deep model reassembly Year: (2022)
Ref_id:b53 Title: Piqa: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b54 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b55 Title: Hellaswag: Can a machine really finish your sentence? Year: (2019)
Ref_id:b56 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b57 Title: Measuring massive multitask language understanding Year: ()
Ref_id:b58 Title: Hyena hierarchy: Towards larger convolutional language models Year: ()
Ref_id:b59 Title: Hourvideo: 1-hour video-language understanding Year: (2024)
Ref_id:b60 Title: Eagle 2.5: Boosting long-context post-training for frontier vision-language models Year: (2025)
