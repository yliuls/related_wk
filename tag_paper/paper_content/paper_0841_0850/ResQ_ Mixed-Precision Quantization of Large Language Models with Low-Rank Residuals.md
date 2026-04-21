Title: ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals
Abstract: Post-training quantization (PTQ) of large language models (LLMs) holds the promise in reducing the prohibitive computational cost at inference time. Quantization of all weight, activation and key-value (KV) cache tensors to 4-bit without significantly degrading generalizability is challenging, due to the high quantization error caused by extreme outliers in activations. To tackle this problem, we propose ResQ, a PTQ method that pushes further the state-of-the-art. By means of principal component analysis (PCA), it identifies a low-rank subspace (in practice 1 /8 of the hidden dimension) in which activation variances are highest, and keep the coefficients within this subspace in high precision, e.g. 8-bit, while quantizing the rest to 4-bit. Within each subspace, invariant random rotation is applied to further suppress outliers. We show that this is a provably optimal mixed precision quantization scheme that minimizes error. With the Llama and Qwen2.5 families of models, we demonstrate that ResQ outperforms recent uniform and mixed precision PTQ methods on a variety of benchmarks, achieving up to 33% lower perplexity on Wikitext than the next best method SpinQuant, and upto 5× speedup over 16-bit baseline. Code is available here. 1

Section: Introduction
Growing capabilities of large language models (LLMs) come with an increasing computational cost at inference time. LLM inference has two distinct stages: prefilling, which processes the input prompt and populates the internal 1 Department of Electrical and Computer Engineering, Purdue University, West Lafayette, USA 2 d-Matrix, Santa Clara, USA. Correspondence to: Utkarsh Saxena <saxenau@purdue.edu>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
1 https://github.com/utkarsh-dmx/project-r  esq state called KV (key-value) cache, and generation, where tokens are generated autoregressively. The prefilling stage is compute-bound, requiring trillions of floating-point operations (FLOPs), whereas the generation stage is memorybound due to iterative accesses and updates of the KV cache. These high computational costs are further amplified by modern LLMs' large sizes -some exceeding 400 billion parameters -and the increasingly long context lengths that necessitates large KV caches.
Quantization algorithms are powerful and principled approaches to address the immense computational demands of LLMs at both stages of inference. Quantization of weights reduces parameter storage, KV cache quantization lowers memory usage of KV cache during generation, whereas activation quantization decreases the complexity of floatingpoint operation. However, effective low-precision quantization is difficult due to large outliers in activations, which can be ∼ 20× larger than other values (Dettmers et al., 2022). While post-training methods like KIVI (Liu et al., 2024c) and KVQuant (Hooper et al., 2024) achieve 2-bit KV cache quantization, and techniques like GPTQ (Frantar et al., 2023) and AWQ (Lin et al., 2024c) optimize very low-precision weights, quantizing activations below 8-bit precision remains an open challenge.
Recent LLM activation quantization methods feature two useful strategies: differential treatment of outliers retain outlier channels in high precision, leading to mixed-precision quantization (e.g., Dettmers et al. 2022;Zhao et al. 2024;Ashkboos et al. 2024b; Figure 1a), whereas invariant random rotation suppress outliers, leading to less difficult uniform low-precision quantization (e.g., Ashkboos et al. 2024c;Liu et al. 2025; Figure 1b). Both reduce quantization error and improve signal-to-quantization-noise ratio (Figure 1d,e) locally; yet a notable model performance gap persists from the 16-bit baseline. For example, SpinQuant (Liu et al., 2025) at 4-bit, applied to Meta-Llama-3-8B (Meta, 2024b), exhibits ∼ 20% higher perplexity than the 16-bit floating point baseline, even after nontrivial optimization.
To mend this gap, we introduce ResQ, a novel PTQ method that combines the strengths of both aforementioned strategies and thereby improve model efficiency with aggressive 4bit quantization of activation, weight, and KV cache. Specif- ically, by means of principal component analysis (PCA), we first identify a low-rank subspace that captures highest variances in activation, and mark the coefficients along this subspace for high-precision (8-bit) and the complement subspace for low-precision (4-bit) quantization. Then, ResQ employs invariant random rotations within each subspace before quantization to further suppress outliers (Figure 1c,d,e). We prove that the above treatment minimizes quantization error. Similar to SpinQuant, most projection matrices can be fused into adjacent weights, leading to minimal runtime computational overhead (Section 4.3). Furthermore, ResQ can be applied to KV cache quantization as well, and can be combined with GPTQ (Frantar et al., 2023), resulting in even better generalizing LLMs.
Outlier-based and rotation-based quantization methods can be combined. For example, high-precision outliers could be detected by ℓ ∞ -norm similar to QUIK (Ashkboos et al., 2024b), and random rotations applied within both high-and low-precision quantization groups, as in QuaRot (Ashkboos et al., 2024c). These methods fare less well than ResQ (Figure 1d,e) in practice, in support of ResQ's provably optimal treatment of outlier quantization. When quantizing weight, activation and KV cache to 4-bit with only 1 /8 channels in 8-bit, ResQ achieves 4-33% lower perplexity on Wikitext and 0.1-5.4% 0-shot accuracy improvements over SpinQuant (Liu et al., 2025), the best in practice so far. Unlike SpinQuant, ResQ does not require gradient-based optimization, making it a less demanding and faster PTQ solution. Furthermore, tuning the rank r of ResQ gives rise to Pareto-optimal solutions as a tradeoff between efficiency and accuracy. We claim the following contributions.
1. We propose ResQ, a mixed precision weight, activation, and KV cache quantization method by keeping low-rank, high-variance components in high precision, in combination with random rotation-induced outlier suppression.
2. We theoretically analyze the projection matrices in ResQ and show that using PCA-based projections minimizes quantization error.
3. We conduct extensive experiments on various models and language tasks and show that ResQ outperforms related state-of-the-art approaches.
4. We develop CUDA kernels and achieve runtime speedup on NVIDIA GPUs with our quantized models.
this section cite: ['b12', 'b23', 'b16', 'b12', 'b70', 'b38', 'b38', 'b16', 'b38']

Section: Prior Work

this section cite: []

Section: Quantization of LLMs
Quantization reduces model size and accelerates inference by lowering neural network bit precision (Choi et al., 2018;Hubara et al., 2021;Yao et al., 2022;Gholami et al., 2022;Xi et al., 2023;Park et al., 2024).
It is broadly categorized into two categories: uniform precision quantization (UPQ) and mixed precision quantization (MPQ). Uniform precision quantization (UPQ) applies the same bit-width across all layers, simplifying implementation but neglecting layer-specific sensitivity to quantization. Weight-only UPQ methods reduce storage by compressing weights, using techniques like Hessian-guided rounding (GPTQ, Frantar et al. 2023), adaptive rounding (QuIP, Chee et al. 2023), channelwise scaling (AWQ, Lin et al. 2024c), and multi-codebook quantization (AQLM, Egiazarian et al. 2024). However, these methods struggle with batch processing due to significant activation memory overhead. Weight-activation UPQ compresses both weights and activations to address this. Methods such as SmoothQuant (Xiao et al., 2023) and OmniQuant (Shao et al., 2024) scale activations and weights to handle outliers, while RPTQ (Yuan et al., 2023a), QLLM (Liu et al., 2024a), and QServe (Lin et al., 2025) employ channel-level strategies like clustering and reordering. Rotation-based methods such as QuaRot (Ashkboos et al., 2024c), SpinQuant (Liu et al., 2025) and DuQuant (Lin et al., 2024b) further enhance robustness in low-precision scenarios. KV cache UPQ reduces memory for large batches or long contexts. FlexGen (Sheng et al., 2023) employs 4-bit quantization and memory offloading, while KIVI (Liu et al., 2024c) uses asymmetric 2-bit quantization for compression, enabling efficient inference.
Mixed precision quantization (MPQ) optimizes bitwidths by adapting to the sensitivity of weights and activations, achieving better accuracy than UPQ at similar compression rates. Our proposed method, ResQ, follows the MPQ approach. Weight-only MPQ has advanced efficiency for memory-bound applications with minimal activation demands. Methods like OWQ (Lee et al., 2024) and SpQR (Dettmers et al., 2024) mitigate activation outliers' impact by retaining critical features in full precision, while SqueezeLLM (Kim et al., 2024) employs Dense-and-Sparse decomposition to efficiently store sensitive weights.
Weight-activation MPQ enhances efficiency by addressing activation outliers (e.g. (Guan et al., 2024;Zeng et al., 2025)). Methods like LLM.int8() (Dettmers et al., 2022) and QUIK (Ashkboos et al., 2024b) preserve critical activations with mixed or low-precision decompositions, while Atom (Zhao et al., 2024) and SliM-LLM (Huang et al., 2024) optimize quantization through channel reordering and salience-driven bit allocation. KV cache MPQ reduces memory usage while preserving precision for critical tokens using techniques like non-uniform quantization, importanceaware precision, and salient token compression (Hooper et al., 2024;Yang et al., 2024b;Dong et al., 2024;He et al., 2024). Alternatively, GEAR quantizes all tokens' KV cache and maintains low-rank quantization error (Kang et al., 2024).
this section cite: ['b8', 'b25', 'b64', 'b18', 'b60', 'b44', 'b61', 'b51', 'b34', 'b38', 'b53', 'b28', 'b13', 'b27', 'b20', 'b69', 'b12', 'b70', 'b24', 'b23', 'b14', 'b21', 'b26']

Section: Low-rank Decomposition
Low-rank decomposition techniques have been widely used in model compression, reducing dimensionality while maintaining performance. For instance, SliceGPT (Ashkboos et al., 2024a) projects weight matrices onto principal components for sparsification, while ESPACE (Sakr & Khailany, 2024) reduces activation dimensionality via pre-calibrated projections, achieving inference-time efficiency. Similarly, ASVD (Yuan et al., 2023b) introduces an activation-aware decomposition method that incorporates activation distributions into weight decomposition. Additionally, low-rank decomposition can be applied to reduce KV cache size. For example, Eigen Attention (Saxena et al., 2024) and ASVD (Yuan et al., 2023b) employ low-rank approximations to reduce memory usage in KV caches during attention operations. PALU (Chang et al., 2024) introduces learnable projections to adaptively compress KV caches based on the compression budget. Finally, Matryoshka KV Cache (Lin et al., 2024a) refines this with hierarchical orthogonal projections and knowledge distillation.
this section cite: ['b48', 'b50', 'b6']

Section: Quantization
Quantization of weight, activation or KV cache involves converting component elements to low precision so that they can be represented using fewer bits for more efficient compute and storage. The N -bit integer quantization and dequantization process on matrix X is given as
Q N (X) = X -z X s X • s X + z X ,(1)
where ⌊•⌉ is a round-and-clip function; s X and z X the scale and zero-point;
z X = 0, s X = max(|X|) 2 N -1 -1 for symmetric quantization or z X = min(X), s X = max(X)-min(X) 2 N -1 for asymmetric quantization.
this section cite: []

Section: ResQ
In this section, we introduce ResQ, a mixed-precision quantization approach that projects weights, activations, and the KV cache into an orthogonal space, retaining the low-rank components in high precision (8-bit) and the rest in low precision. We describe the quantization scheme, the generation of the basis space, provide theoretical guarantees, and outline end-to-end LLM inference deployment procedure.
this section cite: []

Section: Quantization Scheme
Given input activation X ∈ R n×d and weight W ∈ R d×d , they are first projected onto an orthogonal basis defined by the vectors U ∈ R d×d . The coefficients of the projections along this basis are then subject to quantization. We seek to quantize some coefficients along certain bases at high precision while those remaining at low precision. Within R d , denote bases of a low-rank space of high-precision components by U h ∈ R d×r and those of its complementary subspace of low-precision components by U l ∈ R d×(d-r) . The rank r controls the amount of components in high precision (in practice we typically choose r = d /8). We have
U h U ⊤ h + U l U ⊤ l = U U ⊤ = I because U is orthogonal. The quantized activation X q is thusly X q = Q(XU ) = [Q L (XU l ) Q H (XU h )] .
(2) Similarly, quantized weights W q is obtained by projecting the inputs space of weights by U ⊤ and quantizing the coefficients,
W q = Q(U ⊤ W ) = Q L (U ⊤ l W ) Q H (U ⊤ h W ) .(3)
And the output of the layer is,
X q W q = Q L (XU l )Q L (U ⊤ l W ) + Q H (XU h )Q H (U ⊤ h W ).(4)
We make two observations due to orthogonality. First, the introduction of the projections do not alter the output of the model at infinite precision. This means that, if quantization operation is removed from Equation 4, the layer output is numerically invariant. Second, multiplication between lowand high-precision components vanishes (Figure 2). This is efficient because only hardware kernels for quantized GEMM between operands of same precision are required.
this section cite: []

Section: Projections and Optimality Thereof
Intuitively, the orthogonal basis vectors U should have two properties: (1) the low-rank space for high-precision quantization should capture the more important components, and
(2) quantization error in both high-and low-precision groups should be minimized. We construct U as a combination of two rotation matrices serving both objectives respectively. We write U i = P i R i , i ∈ {h, l}. Therefore,
U = P R = [P l P h ] R l 0 0 R h ,(5)
where, P l , R l ∈ R d×(d-r) , P h , R h ∈ R d×r . Inspired by prior work (Ashkboos et al., 2024c;Chee et al., 2023), we make R l , R h random orthogonal matrices because random rotation reduces outliers, making the rotated matrices easier to quantize. Furthermore, projection with a random orthogonal matrix increases Gaussianity of activations and weights within high-and low-precision groups, due to Lemma 4.1, conducive to the quantizations applied to these groups.
Lemma 4.1. By Central Limit Theorem, the distribution after multiplication with random orthogonal matrix is approximately Gaussian (Tseng et al., 2024). To determine P , we minimize the activation quantization error ∥X -X q ∥ F . For activations quantized according to Equation 2, we have,
∥X -X q ∥ F = ∥XU l -Q L (XU l )∥ F + ∥XU h -Q H (XU h )∥ F .(6)
Theorem 4.2. For any matrix X quantized to X q according to method described in Equation 2, assuming the values to be quantized in X are normally distributed, we have
E∥X -X q ∥ F ≤ πlog(d -r) 2 L-1 -1 E∥X∥ F - πlog(d -r) 2 L-1 -1 - √ πlog r 2 H-1 -1 E∥XP h ∥ F . (7)
Full proof of Theorem 4.2 is in Appendix A. Theorem 4.2 bounds the quantization error in Equation 6 from above. To lower this upper bound of quantization error is thusly to maximize ∥XP h ∥ F which happens when P h comprises of eigenvectors of the covariance matrix XX ⊤ with its largest eigenvalues. Therefore, the low-rank subspace for highprecision quantization can be obtained by means of PCA, while the subspace for low-precision quantization can be obtained using
U h U ⊤ h + U l U ⊤ l = P h P ⊤ h + P l P ⊤ l = I (be- cause R i is orthogonal).
If we construct P by taking eigenvectors of XX ⊤ arranged in increasing order of eigenvalues, the last r columns of such a P would correspond to P h and the first d -r columns would correspond to P l . The distribution of activation after applying different projection matrices is shown in Figure 3. Projection of activation along P sorts the activation coefficients in increasing order of variance due to increasing eigenvalues of bases vectors. Consequently, the later r channels of the projected activations with higher variance are kept in higher precision. Projection along U = P R smoothes the activations along low precision and high precision groups further reducing quantization error (Figure 3) and improving quantization SNR (Figure 1(d,e)).
this section cite: ['b7', 'b57']

Section: Inference Computation with Optimized Projections
Once the projection matrices are obtained, the operation in Equation 4requires multiplying the weights and activations with U . Weights can be projected and quantized offline. The projection operation on an activation can be merged to the weight of a previous linear layer. Based on the architecture of decoder based LLMs, we introduce four different kinds of projections (Figure 4) :
U A ∈ R d h ×d h , U B , U C ∈ R dhead×dhead , U D ∈ R dFFN×dFFN where d h is hid- den dimension of LLM, d head
this section cite: []

Section: is the attention head dimension and d FFN the hidden dimension of feedforward network (FFN).
Projections at block boundaries Input activations to attention and FFN are projected via U A . Projection is handled by right-multiplying the weight matrix of final linear layer in each block (o proj in attention and down proj in FFN) by U A . Thus, projections of activations is handled at no additional inference cost. To maintain numerical invariance, the first linear layer of each block (q proj|k proj|v proj in attention and up proj|gate proj in FFN) is pre multiplied with U ⊤ A (Figure 4a). Similarly, the weights of the embedding layer and the final head are modified to manage projection of the residual stream.
Projections within the attention block U B , U C ensures that activations within attention block are projected (Figure 4b). Post-multiplication of value projection layer by U B ensures that value vectors in KV cache are projected and quantized optimally. Consequently, the weights of o proj layer need to be pre multiplied by U ⊤ B to ensure numerical invariance. U C ensures that the quantization of key in KV cache is handled optimally. To achieve that, it is required to project both the query and key using the same projection matrix U C . The attention dot product remains invariant under projected inputs,
q proj K ⊤ proj = (qU C )(U ⊤ C K ⊤ ) = qK ⊤ ,(8)
where q and K are query and key after rotary embedding (RoPE), respectively. Because U C cannot be merged into the previous linear layer due to presence of RoPE, the projection is explicitly computed at runtime, but made more efficient by applying uniform precision quantization to U C and corresponding input activations.
Projections within the feedforward block U D ensure improved quantization of activation within FFNs (Figure 4c). U ⊤ D is left-multiplied with weights of down proj, but due to the presence of activation functions within the block, U D cannot be merged to weights of preceding linear layers and is computed at runtime. U D is applied to the hidden dimension of the FFNs (d FFN ) which is typically 3× to 4× the embedding dimension in most LLMs. In this scenario, matrix multiplication with U D is extremely expensive in computation and storage. To minimize the overhead, we choose U D to be a hadamard matrix to leverage fast and efficient hadamard transform kernel. And, we choose weights and activations for down proj layer to be uniformly quantized to low precision.
this section cite: []

Section: Experiments

this section cite: []

Section: Setup
Models, tasks, datasets and baselines We conduct experiments on Llama 2 (Touvron et al., 2023), Llama 3 (Meta, 2024b), and the recently released Llama 3.2 (Meta, 2024a) Table 1: Comparison of perplexity score on Wikitext, average 0-shot common sense reasoning accuracy and average 0-shot MMLU accuracy. Results of all techniques were obtained using their official codebase. Our work ResQ and QUIK (Ashkboos et al., 2024b) keep 1 /8 channels in 8-bit and remaining in 4-bit for W/A/KV = 4.5-bit. Other baselines are uniformly quantized to W/A/KV = 4-bit. All techniques except RTN employ GPTQ (Frantar et al., 2023) for weight quantization. ↑ higher is better, ↓: lower is better. Full results in Appendix D, Tables 10 and 11.
Family Method W/A/KV Meta-Llama-3-8B Meta-Llama-3-70B Wiki (↓) Avg. 0-shot (↑) MMLU (↑) Wiki (↓) Avg. 0-shot (↑)
MMLU (↑) 16-bit baseline 16/16/16 6.1 67.1 63.1 2.9 73.1 75.9 RTN 4/4/4 218.9 39.3 23.6 452.7 45.5 23.2 GPTQ 4/4/4 166.3 39.8 23.3 11.6e3 34.9 25.5 SmoothQuant+ 4/4/4 78.2 42.5 24.7 ---QUIK 4.5/4.5/4.5 14.2 51.6 32.7 8.0 58.2 51.1 QuaRot 4/4/4 7.8 62.1 53.2 5.7 67.6 65.3 SpinQuant 4/4/4 7.4 63.8 56.2 6.2 65.7 59.4 Llama 3 ResQ 4.5/4.5/4.5 7.1 63.9 57.2 4.1 71.1 73.9 Family Method W/A/KV Llama-3.2-1B Llama-3.2-3B Wiki (↓) Avg. 0-shot (↑) MMLU (↑) Wiki (↓) Avg. 0-shot (↑) MMLU (↑) 16-bit baseline 16/16/16 9.8 54.9 36.9 7.8 62.7 54.8 RTN 4/4/4 329.1 38.1 23.8 268.8 38.7 25.7 GPTQ 4/4/4 108.9 38.0 24.9 178.3 40.3 24.8 SmoothQuant+ 4/4/4 228.9 38.0 24.1 96.1 39.0 25.9 QUIK 4.5/4.5/4.5 21.8 44.3 25.1 15.8 48.8 31.1 QuaRot 4/4/4 14.3 49.0 25.5 10.1 56.1 42.0 SpinQuant 4/4/4 13.6 48.8 25.6 9.2 57.9 44.2 Llama 3.2 ResQ 4.5/4.5/4.5 12.4 50.1 29.4 8.8 59.0 49.8 Family Method W/A/KV Qwen2.5-3B Qwen2.5-72B Wiki (↓) Avg. 0-shot (↑) MMLU (↑) Wiki (↓) Avg. 0-shot (↑) MMLU (↑) 16-bit baseline 16/16/16 8.0 63.8 66.1 3.9 73.4 84.3 RTN 4/4/4 39033.0 35.1 23.4 45412.7 34.3 24.0 GPTQ 4/4/4 9977.8 35.1 23.2 37967.2 34.5 23.3 SmoothQuant+ 4/4/4 73306.7 34.8 23.9 ---QUIK 4.5/4.5/4.5 15.5 51.2 39.4 8.3 61.9 69.3 QuaRot 4/4/4 68.8 47.7 28.9 4.9 70.3 80.1 Qwen2.5 ResQ 4.5/4.5/4.5 9.0 61.1 61.2 4.6 72.0 81.5 Table 2: Comparison of performance of quantization approaches on generative tasks. Our work ResQ and QUIK (Ashkboos et al., 2024b) keep 1 /8 of channels in 8-bit and remaining in 4-bit for W/A/KV = 4.5-bit. Other baselines are uniformly quantized to W/A/KV = 4-bit. GSM8K 5-shot (↑) LongBench (↑) Model Method W/A/KV flexible extract strict match qmsum samsum repobench-p 16-bit baseline 16/16/16 51.0 50.6 23.9 44.8 66.4 QUIK 4.5/4.5/4.5 2.3 0.0 10.5 25.2 37.6 QuaRot 4/4/4 27.6 27.1 22.0 43.8 60.6 SpinQuant 4/4/4 29.8 29.6 23.0 43.9 62.6 Meta-Llama-3-8B ResQ 4.5/4.5/4.5 33.6 33.2 23.1 44.1 62.3 16-bit baseline 16/16/16 25.1 24.9 23.1 43.0 64.4 QUIK 4.5/4.5/4.5 2.5 0.0 15.9 31.7 30.9 QuaRot 4/4/4 10.1 9.1 20.6 39.5 56.8 SpinQuant 4/4/4 11.6 11.4 21.7 41.9 59.1 Llama-3.2-3B ResQ 4.5/4.5/4.5 17.1 16.7 21.7 43.0 61.5
and Qwen2.5 (Yang et al., 2024a) models. We also include multi-modal language models belonging to Qwen2 VL family (Wang et al., 2024) for our evaluations. We benchmark our approach against GPTQ (Frantar et al., 2023), QuaRot (Ashkboos et al., 2024c), QUIK (Ashkboos et al., 2024b), SpinQuant (Liu et al., 2025) and SmoothQuant+, a stronger baseline created by combining SmoothQuant (Xiao et al., 2023) with GPTQ following Sharify et al. 2024. We evalu-ate the quantization approaches on a range of tasks which measure the language modeling ability: perplexity on Wikitext (Merity et al., 2017), common sense reasoning ability: average 0-shot accuracy on Arc-c/e (Clark et al., 2018), BoolQ (Clark et al., 2019), HellaSwag (Zellers et al., 2019), Openbook QA (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), WinoGrande (Sakaguchi et al., 2021), language understanding: 0-shot accuracy on MMLU (Hendrycks et al., 2021), mathematical understanding: 5-shot GSM8K (Cobbe et al., 2021), dialogue summarization: samsum (Gliwa et al., 2019) and qmsum (Zhong et al., 2021) from LongBench (Bai et al., 2024), code completion: repobench-p (Liu et al., 2024b) from LongBench, and multi-modal understanding: MMMU (Yue et al., 2024).
this section cite: ['b56', 'b16', 'b58', 'b16', 'b38', 'b61', 'b52', 'b40', 'b10', 'b9', 'b68', 'b43', 'b5', 'b49', 'b47', 'b22', 'b11', 'b19', 'b71', 'b4', 'b67']

Section: Implementation details
We implement ResQ using the HuggingFace Transformers library (Wolf et al., 2020) with PyTorch (Paszke et al., 2019). We share a single U A across all layers, while U B , U C and U D are generated per layer. Following SpinQuant (Liu et al., 2025), we use per-token asymmetric quantization for activations, per-channel symmetric quantization for weights, and per-head asymmetric quantization for the KV cache. We fuse the projection matrices U A , U B , U D into weights and apply GPTQ (Frantar et al., 2023) for weight quantization. To efficiently implement on-the-fly projections, U D is a Hadamard matrix and U C and its activations are quantized to 8-bit. The entire process, including obtaining projections and quantization, runs on a single NVIDIA A100 GPU; for Meta-Llama-3-8B, it takes 35 minutes. Additional details are in Appendix C.
this section cite: ['b59', 'b45', 'b38', 'b16']

Section: Main Results
Language modeling, understanding, and reasoning tasks We evaluate ResQ on tasks that test language modelling ability (perplexity on Wikitext), common sense reasoning ability (average 0-shot accuracy on the eight tasks listed in section 5.1) and language understanding (average 0-shot accuracy on MMLU). The results are presented in Table 1. We see that ResQ reduces the gap to 16-bit performance and outperforms the quantization baselines across all tasks on all models. Particularly, on Llama 3/3.2 family of models, ResQ outperforms SpinQuant by achieving 4-33% lower Wikitext perplexity, 0.1-5.4% better average 0-shot accuracy and a 1-14.5% better accuracy on MMLU benchmark without any additional training. For the Qwen-2.5 model family, all
this section cite: []

Section: Multi-modal understanding
We benchmark the quantization approaches on vision language models (VLMs) by quantizing Qwen2 VL family and evaluating their performance on MMMU (Table 3, Yue et al. 2024). Only the language model is quantized while the vision encoder remains in 16bit as the language model has many more parameters (over 10× for Qwen2-VL-7B-Instruct). ResQ outperforms baselines on both 2B and 7B models, achieving superior accuracy and demonstrating its generalizability. Results for individual MMMU tasks are provided in Appendix F.
Comparison against outliers with rotation baseline A stronger baseline can be created combining existing quanti-
U D U A U B U C U C , U B
Llama-2-7b-hf 5.8 1550 2500 5.8 5.9 5.9 Meta-Llama-3-8B 7.1 1607 37.4 7.2 7.3 7.4 Llama-3.2-3B 8.8 279.2 39.0 9.0 9.2 9.4 zation approaches. Like QUIK, one can find channels which consistently contain outliers and keep them in 8-bit while keep the remaining channels in low precision. And, the quantization of high/low precision groups can be improved using random rotations introduced in QuaRot. Compared with such a baseline which keeps channels with high l ∞norm in 8-bit, ResQ's unique approach involves keeping coefficients along bases with high eigenvalues in 8-bit. We see in Table 4 that ResQ consistently outperforms such a strong baseline across various precisions of W/A/KV highlighting ResQ's PCA driven theoretically optimal approach of choosing high precision components.
this section cite: ['b67']

Section: Iso-bitwidth comparison
We also perform iso-bitwidth comparison of ResQ with SpinQuant and QuaRot at W/A/KV of 4-bit. To enable 4-bits with ResQ, we keep 1 /8 channels corresponding to highest eigen values in P in 6-bit, 1 /8 channels corresponding to lowest eigen values in P in 2-bit and remaining in 4-bit. Within each quantization group, we apply random orthogonal rotations to minimize quantization error. As shown in Table 5, even at same bitwidth of 4-bit, ResQ achieves improved performance on Qwen2.5 family of models. Complete results are provided in Table 14.
Training rotation matrix R To further improve the per-formance of ResQ, the random rotation matrix R can be optimized to minimize final task loss similar to SpinQuant (Liu et al., 2025) albeit at higher computational cost for quantizing the model. We keep identical training hyperparameters as SpinQuant, and learn the rotation R for models upto 8B parameters. The evaluation results are provided in Table 15 in Appendix H. We see upto 5% improvement in Wikitext perplexity, upto 0.6% increase in average 0shot reasoning accuracy and upto 1.1% increase in MMLU accuracy.
this section cite: ['b38']

Section: Hardware Performance
We implement the mixed-precision quantization using CUDA 11.8 and PyTorch.
The weights are quantized into INT4 and INT8 components offline while online quantization of activations into INT4 and INT8 components is handled in a single kernel call. We use CUTLASS (Thakkar et al., 2023) to perform INT4 and INT8 GEMM operations on TensorCore. Further, for KV cache compression, we implement online quantization and packing for memory efficiency. For efficient implementation of online hadamard transform involved in activation quantization in down proj layer, we use the fast hadamard transform library (fas, 2023). Prefill speedup On an NVIDIA RTX 3090 GPU, we achieve a 1.61× to 3.03× speedup with ResQ over the 16-bit baseline for a single decoder block across various language models (Figure 5). Speedups are higher for larger models and shorter sequences. Compared to INT4, ResQ is only 14% slower on average, showing minimal overhead from mixed-precision and on-the-fly projections. Memory usage We evaluate end to end memory usage on NVIDIA RTX 3090 (24GB) for different sequence lengths in Table 7. ResQ consumes 1.84× -3.08× lower memory than the FP16 baseline. Notably, Qwen2.5-14B model leads to out of memory (OOM) error while ResQ is able to support its inference upto sequence length of 8192 tokens. Compared with the INT4 baseline QuaRot, the memory used by ResQ is 4-11% higher. Multi GPU inference We evaluate end-to-end batched inference latency on a GPU server with 3 NVIDIA A100 (82 GB) GPUs running Meta-Llama-3-70B. ResQ's weight, activation, and KV cache quantization enable the 70B model to fit on a single GPU, while FP16 requires model parallelism across all three GPUs. This allows ResQ to support data-parallel inference, unlike FP16. In Table 8, we show time to first token (i.e. end to end prefill latency) at different sequence lengths and batch sizes. Compared to FP16 baseline, ResQ achieves upto 4.98× improvement in end to end latency under batched inference setting. This improvement stems from two factors, first is computational complexity reduction achieved in ResQ due to weight and activation quan-   tization, and, second is the memory compression achieved in ResQ due to weight and kv cache quantization which enables serving the 70B parameter model on a single GPU device. 5.4. Ablation Studies Projection bases We evaluate the impact of different projections employed in ResQ by removing them and evaluating performance in Table 6. We see that removing U D or U A has a catastrophic impact on perplexity highlighting their importance. U B and U C which aid in quantization of KV cache have less severe impact when removed independently. But removing both of them leads to a non trivial increase in perplexity (particularly for Meta-Llama-3-8B and Llama-3.2-3B which employ grouped query attention).
this section cite: []

Section: Rank of high-precision subspace
ResQ allows for seamless trade-off between accuracy and performance by modulating the rank r of high precision subspace (Figure 6-left).
Increasing the rank improves perplexity albeit at the cost of increased computations in high precision.
this section cite: []

Section: Calibration dataset size
We change number of Wikitext calibration samples used to obtain projections and evaluate performance in Figure 6-right. For Meta-Llama-3-8B, MMLU accuracy increases with increasing samples and saturates beyond 128 samples. For Llama-3.2-3B, the trend is unclear with 512 samples achieving best performance.
this section cite: []

Section: Calibration dataset
We evaluate the sensitivity of ResQ's projections to the calibration dataset. While the random rotation matrix R is data-independent, the PCA-based projection matrix P depends on the data. We obtain P using samples from Alpaca (Taori et al., 2023), PTB (Marcus et al., 1993), and C4 (Raffel et al., 2020), and Table 16 in Appendix I shows minimal performance variation, demonstrating the robustness of ResQ's calibration.
this section cite: ['b54', 'b39', 'b46']

Section: Conclusion
We introduce ResQ, a novel mixed-precision, acceleratorfriendly PTQ technique toward 4-bit quantization of large language models. ResQ projects weight, activation, and KV cache tensors to subspaces spanned by principal components, quantizing a low-rank ( 1 /8 of hidden dimension) highvariance subspace to 8-bit and the rest to 4-bit. ResQ outperforms both uniform-and mixed-precision quantization methods. We demonstrate the effectiveness of ResQ across a variety of tasks-including language modeling, language understanding, common-sense reasoning, language generation and multi modal understanding-using the Llama and Qwen models. Compared to SpinQuant, the strongest baseline, ResQ achieves up to 33% lower perplexity on the WikiText dataset without requiring any additional training and offers up to 5× speedup over the 16-bit baseline.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2023)
Ref_id:b1 Title: Compress large language models by deleting rows and columns Year: (2024)
Ref_id:b2 Title: QUIK: Towards end-to-end 4-bit inference on generative large language models Year: (2024)
Ref_id:b3 Title: QuaRot: Outlier-free 4-bit inference in rotated llms Year: (2024)
Ref_id:b4 Title: LongBench: A bilingual, multitask benchmark for long context understanding Year: (2024)
Ref_id:b5 Title: PIQA: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b6 Title: Compressing kv-cache with lowrank projection Year: (2024)
Ref_id:b7 Title: QuIP: 2-bit quantization of large language models with guarantees Year: (2023)
Ref_id:b8 Title: PACT: Parameterized clipping activation for quantized neural networks Year: (2018)
Ref_id:b9 Title: Exploring the surprising difficulty of natural yes/no questions Year: (2019)
Ref_id:b10 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b11 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b12 Title: -bit matrix multiplication for transformers at scale Year: (2022)
Ref_id:b13 Title: SpQR: A sparse-quantized representation for near-lossless llm weight compression Year: (2024)
Ref_id:b14 Title: QAQ: Quality adaptive quantization for llm kv cache Year: (2024)
Ref_id:b15 Title: Extreme compression of large language models via additive quantization Year: (2024)
Ref_id:b16 Title: OPTQ: Accurate post-training quantization for generative pre-trained transformers Year: (2023)
Ref_id:b17 Title: A framework for few-shot language model evaluation Year: ()
Ref_id:b18 Title: A survey of quantization methods for efficient neural network inference Year: (2022)
Ref_id:b19 Title: SAM-Sum Corpus: A human-annotated dialogue dataset for abstractive summarization Year: (2019)
Ref_id:b20 Title: APTQ: Attention-aware post-training mixed-precision quantization for large language models Year: (2024)
Ref_id:b21 Title: ZipCache: Accurate and efficient kv cache quantization with salient token identification Year: (2024)
Ref_id:b22 Title: Measuring massive multitask language understanding Year: (2021)
Ref_id:b23 Title: Towards 10 million context length llm inference with kv cache quantization Year: (2024)
Ref_id:b24 Title: Salience-driven mixed-precision quantization for large language models Year: (2024)
Ref_id:b25 Title: Accurate post training quantization with small calibration sets Year: (2021)
Ref_id:b26 Title: GEAR: An efficient kv cache compression recipefor near-lossless generative inference of llm Year: (2024)
Ref_id:b27 Title: SqueezeLLM: Dense-and-sparse quantization Year: (2024)
Ref_id:b28 Title: OWQ: Outlier-aware weight quantization for efficient finetuning and inference of large language models Year: (2024)
Ref_id:b29 Title: Efficient riemannian optimization on the stiefel manifold via the cayley transform Year: (2020)
Ref_id:b30 Title: Absorbing outliers by low-rank component for 4-bit diffusion models Year: (2025)
Ref_id:b31 Title: Adaptive kv compression via trainable orthogonal projection Year: (2024)
Ref_id:b32 Title: Distributing outliers via dual transformation makes stronger quantized llms Year: (2024)
Ref_id:b33 Title: AWQ: Activation-aware weight quantization for on-device llm compression and acceleration Year: (2024)
Ref_id:b34 Title: QServe: W4a8kv4 quantization and system codesign for efficient llm serving Year: (2025)
Ref_id:b35 Title: QLLM: Accurate and efficient low-bitwidth quantization for large language models Year: (2024)
Ref_id:b36 Title: Benchmarking repository-level code auto-completion systems Year: (2024)
Ref_id:b37 Title: KIVI: A tuning-free asymmetric 2bit quantization for kv cache Year: (2024)
Ref_id:b38 Title: SpinQuant: Llm quantization with learned rotations Year: (2025)
Ref_id:b39 Title: Building a large annotated corpus of English: The Penn Treebank Year: (1993)
Ref_id:b40 Title: Pointer sentinel mixture models Year: (2017)
Ref_id:b41 Title: Llama 3.2: Revolutionizing edge AI and vision with open, customizable models Year: (2024)
Ref_id:b42 Title: Introducing Meta Llama 3: The most capable openly available LLM to date Year: (2024)
Ref_id:b43 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018)
Ref_id:b44 Title: LUT-GEMM: Quantized matrix multiplication based on luts for efficient inference in large-scale generative language models Year: (2024)
Ref_id:b45 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b46 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b47 Title: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b48 Title: Espace: Dimensionality reduction of activations for model compression Year: (2024)
Ref_id:b49 Title: Social iqa: Commonsense reasoning about social interactions Year: (2019)
Ref_id:b50 Title: Eigen Attention: Attention in low-rank space for kv cache compression Year: (2024)
Ref_id:b51 Title: Omnidirectionally calibrated quantization for large language models Year: (2024)
Ref_id:b52 Title: Post training quantization of large language models with microscaling formats Year: (2024)
Ref_id:b53 Title: Flex-Gen: High-throughput generative inference of large language models with a single gpu Year: (2023)
Ref_id:b54 Title: Stanford alpaca: An instruction-following llama model Year: (2023-01)
Ref_id:b55 Title:  Year: ()
Ref_id:b56 Title: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b57 Title: QuIP# even better llm quantization with hadamard incoherence and lattice codebooks Year: (2024)
Ref_id:b58 Title: Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b59 Title: Huggingface's transformers: Stateof-the-art natural language processing Year: (2020)
Ref_id:b60 Title: Training transformers with 4-bit integers Year: (2023)
Ref_id:b61 Title: Accurate and efficient post-training quantization for large language models Year: (2023)
Ref_id:b62 Title: Qwen2.5 technical report Year: (2024)
Ref_id:b63 Title: No Token Left Behind: Reliable kv cache compression via importance-aware mixed precision quantization Year: (2024)
Ref_id:b64 Title: ZeroQuant: Efficient and affordable posttraining quantization for large-scale transformers Year: (2022)
Ref_id:b65 Title: RPTQ: Reorderbased post-training quantization for large language models Year: (2023)
Ref_id:b66 Title: ASVD: Activation-aware singular value decomposition for compressing large language models Year: (2023)
Ref_id:b67 Title: MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b68 Title: Can a machine really finish your sentence? Year: (2019)
Ref_id:b69 Title: ABQ-LLM: Arbitrary-bit quantized inference acceleration for large language models Year: (2025)
Ref_id:b70 Title: Atom: Low-bit quantization for efficient and accurate llm serving Year: (2024)
Ref_id:b71 Title: QM-Sum: A new benchmark for query-based multi-domain meeting summarization Year: (2021)
Ref_id:b72 Title: Qwen2-VL-2B-Instruct W/A/KV (bit) Method 0-shot MMMU tasks Art-Design Business Science Health Humanities Tech Avg Year: ()
Ref_id:b73 Title: Artifact licenses According to their licenses, all language models used in the paper fall under acceptable use case. The licenses for the models are linked for perusal: Llama-2-7b-hf, Llama-2-13b-hf, Meta-Llama-3-8B, Meta-Llama-3-70B, Llama-3.2-1B Year: ()
