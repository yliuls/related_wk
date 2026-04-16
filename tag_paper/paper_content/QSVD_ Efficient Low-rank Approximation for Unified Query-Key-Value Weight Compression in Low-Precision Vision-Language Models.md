Title: QSVD: Efficient Low-rank Approximation for Unified Query-Key-Value Weight Compression in Low-Precision Vision-Language Models
Abstract: Vision-Language Models (VLMs) are integral to tasks such as image captioning and visual question answering, but their high computational cost, driven by large memory footprints and processing time, limits their scalability and real-time applicability. In this work, we propose leveraging Singular-Value Decomposition (SVD) over the joint query (Q), key (K), and value (V) weight matrices to reduce KV cache size and computational overhead. We in addition introduce an efficient rank allocation strategy that dynamically adjusts the SVD rank based on its impact on VLM accuracy, achieving a significant reduction in both memory usage and computational cost. Finally, we extend this approach by applying quantization to both VLM weights and activations, resulting in a highly efficient VLM. Our method outperforms previous approaches that rely solely on quantization or SVD by achieving more than 10% accuracy improvement while consuming less hardware cost, making it better for real-time deployment on resource-constrained devices.We open source our code at https://github.com/SAI-Lab-NYU/QSVD.

Section: Introduction
Vision-Language Models (VLMs) are crucial for advancing artificial intelligence by bridging the gap between visual perception and natural language understanding. By enabling machines to interpret and generate both visual and textual information, VLMs open up a wide range of applications, such as image captioning [63,17,8,11], visual question answering [7,3,47], and content-based search [22,43]. These models are vital for tasks where visual context is needed to fully understand textual queries or vice versa, such as healthcare [34,3,18], education [61], and interactive entertainment [44,26].
Despite their strong performance, Vision-Language Models (VLMs) incur substantial computational costs due to the intensive processing required to integrate high-dimensional visual and textual data. Additionally, their autoregressive token generation places significant pressure on memory bandwidth, becoming a major bottleneck for inference speed. To enable practical deployment in latency-sensitive and resource-constrained environments, it is essential to reduce both the computational overhead and the size of the Key-Value (KV) cache, without compromising model accuracy.
To address this issue, particularly the high memory usage introduced by Multi-Head Attention (MHA), several variants have been proposed, such as Grouped-Query Attention [1] and Multi-Query Attention [42,1], which aim to reduce the number of KV projections while maintaining performance. A recent proposal, Multi-Head Latent Attention (MLA) in the DeepSeek-v3 model [31], offers a novel approach to improving VLM efficiency. It significantly reduces the KV cache size by compressing the KV cache into a latent vector, thereby enhancing inference efficiency. Building on the insights from MLA, this work proposes the application of Singular-Value Decomposition (SVD), which has proven effective in reducing both KV cache size and computational cost in prior research [51,60,52,28,25,5,49], to the joint weight matrices of the query, key, and value. This approach significantly reduces the KV cache size by storing only the latent vectors instead of separate key and value vectors. Additionally, we introduce a novel rank allocation scheme, which investigates the importance of each singular value in relation to VLM accuracy. This results in a minimal SVD rank with the minimal impact on the model accuracy. Finally, we extend this approach by applying quantization to both VLM weights and activations. The proposed framework, termed QSVD, results in an extremely efficient VLM that outperforms previous methods that relied solely on quantization or SVD. Specifically, our contribution can be summarized as follows:
• QSVD proposes applying singular value decomposition to the combined weight matrices of the query, key, and value projections. This technique substantially reduces the size of the KV cache, computational overhead, and weight storage, resulting in significant improvements in hardware efficiency.
• To improve the accuracy of SVD-based compression in VLMs, QSVD proposes a novel importance scoring method that quantifies each singular value's contribution to overall model performance, allowing for rank-based truncation that minimizes accuracy degradation.
• Quantization is applied alongside SVD decomposition to both the weights and activations of the VLM. We propose an efficient method to eliminate outliers under the SVD framework, enabling low-precision operation that reduces the memory footprint of both the KV cache and model weights, while incurring minimal impact on accuracy 2 .
2 Background and Related Work
this section cite: ['b62', 'b16', 'b7', 'b10', 'b6', 'b2', 'b46', 'b21', 'b42', 'b33', 'b2', 'b17', 'b60', 'b43', 'b25', 'b0', 'b41', 'b0', 'b30', 'b50', 'b59', 'b51', 'b27', 'b24', 'b4', 'b48']

Section: Vision Language Model
Vision-Language Models (VLMs) [24,23,33,4,13,48] extend the capabilities of Large Language Models (LLMs) by incorporating visual inputs alongside text, enabling tasks such as visual question answering (VQA) and image captioning. Models such as BLIP and InstructBLIP [24,23] employ data filtering and visual instruction tuning to better align model outputs with human preferences in zero-shot settings. A commonly used architecture, illustrated in Figure 1 (a), processes input images through a visual encoder to produce visual tokens, which are then concatenated with text tokens and passed to a language model for response generation. This concatenation-based design is adopted by widely used models including the LLaVA series [33], SmolVLM [39], PaLI-Gemma [4], and Qwen-VL [48]. Although VLMs demonstrate impressive capabilities, their large size presents challenges in terms of computational efficiency and deployment, particularly on resource-constrained devices. This has led to the development of lightweight alternatives. TinyGPT-V [59] and TinyLLaVA [62] explore efficient designs at smaller scales, while SmolVLM [39] introduces a family of compact models (500M and 2B parameters) that maintain strong performance with much reduced hardware cost.
this section cite: ['b23', 'b22', 'b32', 'b3', 'b12', 'b47', 'b23', 'b22', 'b32', 'b38', 'b3', 'b47', 'b58', 'b61', 'b38']

Section: Singular Value Decomposition for Large Models
Singular Value Decomposition (SVD) [19] is a widely used matrix factorization technique that decomposes a matrix W ∈ R m×n into three components: W = U ΣV T , where U and V are orthogonal matrices containing the left and right singular vectors of W , and Σ is a diagonal matrix of non-negative singular values arranged in descending order. By retaining only the top r singular values and corresponding vectors, we obtain a rank-r approximation:
W ≈ U r Σ r V T r(1)
with U r ∈ R m×r , Σ r ∈ R r×r , and V r ∈ R n×r . Equivalently, the approximation can be expressed as
W ≈ AB by defining A = U r Σ 1 2 r B = Σ 1/2 r V T r and B = Σ 1 2 r V T r .
Such low-rank factorizations preserve the most salient structure of W while reducing its dimensionality, enabling matrix compression and accelerating downstream computations.
SVD has been extensively studied as a compression method for LLMs [51,60,52,28,25,5,49]. Early efforts [40] directly applies standard SVD to weight matrices but encountered significant compression errors. To address this, FWSVD [16] prioritizes parameters based on Fisher information [37], while ASVD [60] incorporates activation outliers into the factorization. SVD-LLM [52] further reduces compression loss by explicitly minimizing the contribution of each truncated singular value. Most of these methods focus on compressing model weights. In contrast, Palu [6] and [58] have shifted attention to compressing the KV-Cache, leveraging SVD and low-rank projections to reduce memory footprint. Recent advances include AdaSVD [28], which adaptively compensates for truncation errors and dynamically allocates compression rates according to layer importance, and SVD-LLM V2 [51], which further optimizes singular value truncation via theoretical loss estimation.
Recently, DeepSeek introduces Multi-Head Latent Attention [31], a novel mechanism that integrates low-rank projections directly into the attention computation. Instead of computing attention over the full key and value matrices, this approach projects them into a lower-dimensional latent space using learned projection matrices, effectively reducing the computational and memory costs of multi-head attention without significantly impacting model performance. This latent factorization can be viewed as an implicit low-rank approximation applied dynamically during inference, offering complementary benefits to static weight compression methods such as SVD.
this section cite: ['b18', 'b50', 'b59', 'b51', 'b27', 'b24', 'b4', 'b48', 'b39', 'b15', 'b36', 'b59', 'b51', 'b5', 'b57', 'b27', 'b50', 'b30']

Section: Quantization for Large Models
Post-training quantization (PTQ) has become one of the most used approaches for enabling efficient inference of large models [38,55,2,29,12,41,30,57,45,53,20,9]. For example, AffineQuant [38] replaces the traditional scaling factor with a learned affine transformation to better align weight with the quantization grid.
Another line of work focuses on smoothing outliers in activation distributions, which have shown that activations in LLMs contain severe outliers at the per-channel level [2,9,29,35,30,55], resulting in substantial quantization errors during activation quantization. To address this issue, SmoothQuant [55] reduces activation outliers by shifting part of the activation outliers into the weights, promoting more balanced quantization. Building upon these ideas, techniques like QuaRot [2], DuQuant [29], and SpinQuant [35] incorporate orthogonal transformations to further enhance quantization performance. These transformations maintain computational invariance by preserving the model's output while effectively suppressing outliers. Specifically, let W and X represent the weight and activation matrices, respectively, where X exhibits channelwise outliers, and let Y = XW denote the output. To eliminate outliers in X, an orthogonal matrix H is introduced, satisfying H ⊤ H = HH ⊤ = I. This transformation yields an equivalent formulation Y = XW = X ′ W ′ , where W ′ = H ⊤ W and X ′ = XH. To minimize runtime overhead, W ′ = H ⊤ W can be precomputed offline, and X ′ = XH can be efficiently integrated into the weight matrices of the previous layer, incurring no additional computational cost. The resulting transformed activation X ′ exhibits a smoother distribution with significantly fewer outliers, thus lowering the quantization errors. In parallel, methods such as GPTQ [12], OmniQuant [41], and AWQ [30] focus on optimizing scaling factors and channel-wise equalization during the calibration process.
In the realm of VLMs, quantization presents unique challenges due to the integration of visual and textual modalities. QSLAW [56] introduces a quantization-aware scale learning method with a multimodal warmup strategy that progressively incorporates linguistic and multimodal samples
≈ ✕ W d qkv E r W u q W u k W u v Wv W d v ≈ W u v r E ✕r E Wk Wv Wq ≈ r ✕ E 3E W d qkv E ≈ ✕ r 3E W u r r ✕ Σ W d q ≈ u q r E ✕r E W d k ≈ u k r E ✕r E W d v ≈ u v r E
✕r E Total MACs: 3LE 2 Total weight parameters: 3E 2 Total KV cache size: 2LE (a) (b) (c to stabilize training. It also emphasizes group-wise scaling to better handle activation outliers. Q-VLM [46] addresses cross-layer dependency in quantization by leveraging activation entropy as a proxy to guide block partitioning. It formulates a quantization strategy that balances discretization error and search cost, and further optimizes the visual encoder to disentangle cross-layer interactions, enabling more efficient calibration. MBQ [27] proposes a modality-balanced quantization approach that accounts for the distinct gradient distributions of visual and textual tokens during calibration. It applies a modality-aware loss to improve the accuracy of scaling factor estimation. However, to the best of our knowledge, no prior work has combined quantization with SVD for efficient VLM processing in the manner proposed by QSVD.
) = ✕ r W d qkv E r L Cqkv = r W d k E r L Ck = r W d v E L Cv X E L E E ✕ E E = L Ck = L Cv E (d) (e) (f) Wconcat Wq Wk Wk Wq Wk Wv Wq Wk Wv W W W W d r E r 3E W u q u k u v W W Wk Wv E X E L ✕ X E L
this section cite: ['b37', 'b54', 'b1', 'b28', 'b11', 'b40', 'b29', 'b56', 'b44', 'b52', 'b19', 'b8', 'b37', 'b1', 'b8', 'b28', 'b34', 'b29', 'b54', 'b54', 'b1', 'b28', 'b34', 'b11', 'b40', 'b29', 'b55', 'b45', 'b26']

Section: Methodology
An overview of QSVD is shown in Figure 1 (b), comprising three key components: joint SVD over the combined QKV weights (Section 3.1), adaptive singular value truncation (Section 3.2), and PTQ over low-rank VLMs (Section 3.3).
this section cite: []

Section: Singular-Value Decomposition over Joint QKV Weights
We introduce an efficient SVD-based approach to reduce computation within the multi-head selfattention block, as illustrated in Figure 2, where each subfigure denotes: (a) Original QKV matrix in a vision-language model (VLM) without SVD. (b) Applying SVD separately to the weight matrices of Q, K, and V, where each of W q , W k , and W v is factorized into a down-and up-projection pair. (c) Our proposed approach: concatenating QKV weights before applying SVD. (d) Standard KV computation during prefilling: the input X is multiplied by W k and W v , and the resulting C k and C v are stored in memory. (e) Computation with per-matrix SVD: during prefilling, X must be read from memory and multiplied with the down-projection matrices W d k and W d v to generate low-rank representation of K and V . (f) Storage and computation in QSVD: since W q , W k , and W v share a common down-projection matrix W d qkv , X is multiplied by W d qkv once to produce the intermediate C qkv , which is stored and later used to reconstruct the KV vectors. Let α, η, and γ denote the weight parameter size, KV cache size, and the computational cost in FLOPs of QKV multiplication, respectively. In the original design, assuming a single-head attention module for simplicity, the combined weight matrices W q , W k , and W v collectively contain a total of α f p = 3E 2 parameters, where E represents the embedding dimension (Figure 2 (a)). The corresponding KV cache requires a memory footprint of η f p = 2LE, where L is the input sequence length (Figure 2 (d)). The total computational cost in FLOPs for generating the key, query, and value vectors is γ f p = 3LE 2 , where three matrix multiplications are required, each with a size of (L × E) and (E × E).
QSVD adopts a more efficient strategy that reduces the number of weight parameters, KV cache size, and overall computational cost. Specifically, the weight matrices W Q , W K , and W V , each of size E × E, are concatenated into a single matrix W concat ∈ R E×3E . A low-rank SVD is then applied to this combined matrix to achieve compression.
[Wq, W k , Wv] = Wconcat ≈ W d r × Σr × W u r (2
)
W d qkv = W d r Σ β r , W u qkv = [W u q , W u k , W u v ] = Σ 1-β r W u r (3
) [Wq, W k , Wv] = W d qkv × [W u q , W u k , W u v ](4)
where
W d qkv ∈ R E×r , W u q , W u k , W u v ∈ R r×E
, and β satisfies 0 ≤ β ≤ 1. After decomposition, the QKV components share a common down-projection matrix W d qkv , while each maintains its own distinct up-projection matrix. This results in a total weight size of α qsvd = 4rE (Figure 2 (c)). The computational cost for generating the query, key, and value vectors is γ qsvd = 4LrE, which arises from two steps: first, multiplying the input X with W d qkv to generate C qkv , and then performing a second multiplication with the concatenated matrices [W u q , W u k , W u v ]. During inference, the intermediate products C qkv between the input and the down-projection matrix W d qkv are buffered to compute the KV vectors, yielding a total buffer size of η qsvd = rL (Figure 2 (f)), and the KV vectors can be easily recomputed as:
K = C qkv W u k , V = C qkv W u v (5
)
In comparison, our method achieves reduced weight size and computational cost when 4rE < 3E 2 and 4LrE < 3LE 2 , which holds when r < 0.75E. This condition is easily satisfied with negligible accuracy loss, as demonstrated by the evaluation results in Section 4. Furthermore, our method consistently reduces the buffered size for the intermediate data, since rE is always smaller than 2E 2 given that r < E. During decoding, the cached intermediate representation C qkv is used to reconstruct the key and value matrices via W u k and W u v , which are then combined with the current query (sequence length l=1) to compute the attention outputs.
Previous methods [60,52] apply SVD individually to the weight matrices, as illustrated in Figure 2(b), resulting in a total of α ind = 6rE parameters. During inference, the intermediate products C k and C v , which is computed from the input X and the down-projection matrices W d k and W d v , must be buffered, leading to a total buffer size of η ind = 2rL (Figure 2(e)). The computational cost for generating the query, key, and value vectors is γ ind = 6LrE, and the buffer size for C k and C v is consistently larger than that required to store the unified C qkv in our method. Finally, our method always achieves a lower weight size, computational cost and intermediate storage.
this section cite: ['b59', 'b51']

Section: Cross-layer Rank Allocation for Low-rank SVD
Performing SVD on the joint QKV weights can lead to hardware efficiency gains in both computation and storage, provided that the rank r of W d qkv is sufficiently reduced without compromising the final accuracy performance. A key challenge, therefore, is determining how to truncate the singular values across all self-attention blocks in the VLM. While prior work has used Fisher information [37] to assess the importance of individual weight matrix or a group of singular values [6,16], QSVD proposes a more effective method that evaluates the importance of each singular value in a way that minimizes degradation in model accuracy.
Given the SVD of a weight matrix W = U ΣV T , it can also be expressed as a summation: W = n i=1 u i σ i v T i , where σ i is the i-th singular value, and u i , v i are the corresponding left and right singular vectors. Truncating a singular value by setting σ i = 0 effectively removes its associated single-rank component from the matrix, resulting in a modified representation W ′ σi of W , we have:
∆Wσ i = W -W ′ σ i = uiσiv T i (6
)
The truncation of σ i will affect the final output of the VLM and lead to an increase in the training loss L t . The corresponding change in training loss can be estimated through first-order expansion:
Lt(W ′ σ i ) = Lt(W -∆Wσ i ) ≈ Lt(W ) - j,k ∆Wσ i [j, k] • ∂Lt ∂W [j, k](7)
where L t (W ′ σi ) denotes the training loss after the weight matrix is modified to W ′ σi , W σi [j, k] represents the (j, k)-th element of the matrix W σi . Let G W represent the gradient of the loss with respect to the original weight matrix W , the changes on the training loss can be expressed as follows:
∆Lσ i = Lt(W ) -Lt(W ′ σ i ) ≈ j,k ∆Wσ i [j, k] • GW [j, k] = ⟨∆Wσ i , GW ⟩F (8
)
where ⟨•, •⟩ F denotes the Frobenius inner product over matrix elements. This formulation enables estimation of each singular value's contribution (e.g., σ i ) to the change in training loss, providing a principled basis for rank selection by measuring the sensitivity of the loss function to each truncated component across all layers, which can be used to evaluate the importance for each singular value. Specifically, by evaluating ∆L σi across multiple calibration samples and computing its squared expectation, we derive the Importance Score Îσi for each singular value σ i , which serves as an empirical approximation of the diagonal Fisher information:
Îσ i = Ex∼D ∆L (n) σ i 2 ≈ 1 N N n=1   j,k ∆Wσ i [j, k] • G (n) W [j, k]   2 (9
)
where D denotes the calibration dataset, n indexes individual samples, and N is the total number of samples in D. However, computing the importance score as defined in Equation 9poses a significant memory burden, primarily due to the need to construct and store ∆W σi [j, k] for all singular values. Since each ∆W σi is a full E × E matrix and there are E such singular values from the joint SVD, the total memory cost scales as O(E 3 ) per layer, making naive computation impractical for large models. To address this, the importance score Îσi can alternatively be computed as follows:
Îσ i = 1 N N n=1 σ 2 i U T G (n) W V 2 (i,i) (10
)
where U and V are the left and right singular vectors from the SVD, σ i is the i-th singular value. The notation (i, i) refers to the i-th diagonal element of the transformed gradient matrix
U T G (n) W V .
The proof is given in Appendix A.1. This form eliminates the need to compute and store ∆W σi for each singular value, requiring only O(E 2 ) memory instead of O(E 3 ) per layer.
The overall SVD procedure is as follows. Starting with the original model, we first concatenate the QKV weight matrices and apply joint SVD, following the method outlined in Section 3.1. We adopt the activation-aware SVD technique from ASVD [60] to extract the singular values across all layers. For each singular value, we then compute its corresponding importance score based on the calibration dataset, as defined in Equation 10. After computing the importance scores, we perform cross-layer ranking by globally sorting all singular values based on their scores. We retain only the top k singular values with the highest importance scores, where k termed rank budget. All remaining singular values are truncated. This global ranking strategy ensures that the most critical components are preserved regardless of the layer they originate from, allowing for an optimal allocation of rank capacity throughout the VLM. For QSVD, we apply the rank selection mechanism to the self-attention layers throughout the entire VLM. Building on the efficient low-rank SVD approach described in Section 3.1 and Section 3.2, this section presents an efficient quantization scheme applied to the resulting low-rank VLMs for further hardware efficiency enhancement. To analyze the outlier distribution in the VLM, we profile the input activation distribution of LLaVA-v1.5 13B [33]. Specifically, we examine the input activations X across the selfattention modules and feed-forward modules within the language model of the VLM, as illustrated in Figure 3 (a) and (b), respectively. Our analysis reveals prominent channelwise outliers in X across all three components, which poses the great challenge when quantizing these VLM activations for low-precision operations.
To address this issue, we adopt the rotational method introduced in [2,54] and outlined in Section 2.3 to smooth channelwise outliers. However, since the self-attention architecture of the VLM has been modified by the application of SVD, we develop an efficient quantization approach that accounts for this change. Let X denote the input to the weight matrices, and the output be expressed as
Y = XW d qkv W u qkv = C qkv W u qkv , where W d qkv = W d r Σ β r and W u qkv = Σ 1-β r W u r
, as notated in Equation 3. The distribution of C qkv , which is buffered for KV vector recomputation, is shown in Figure 3 (c). We observe that C qkv exhibits channelwise outliers, rendering it unsuitable for low-precision quantization. To eliminate the channelwise outliers in both X and the compressed representation C qkv , we introduce orthogonal matrices H 1 and H 2 . The self-attention computation together with its quantized counterpart are then reformulated as follows: where C qkv can be approximately computed using the quantized input and weight as:
Y = (XH ⊤ 1 )(H1W d qkv H ⊤ 2 )(H2W u qkv ) Y ′ = Q(C qkv )Q(H2W u qkv ) (11
)
C qkv ≈ Q(XH ⊤ 1 )Q(H1W d qkv H ⊤ 2 ) (12
)
The computations presented in Equation 11and Equation 12 support low-precision execution while reducing the size of both the weight parameters and the intermediate results C qkv , thereby lowering the overall memory footprint and reduce the processing latency.
Although the introduction of H 1 and H 2 helps mitigate outliers in X and C qkv , respectively, we observe that these transformations do not fully eliminate the severe outliers, particularly those present in C qkv . To analyze this issue, we examine the distribution of W d qkv , which directly influences the distribution of C qkv . We observe that W d qkv = W d r Σ β r is strongly affected by the parameter β. Since Σ r is a diagonal matrix whose entries are singular values that can vary significantly in magnitude, raising them to the power β can amplify the disparity. This, in turn, exacerbates the presence of outliers in C qkv , as shown by the following derivation:
C qkv = XW d qkv = XW d r Σ β r = XW d r diag(σ β 1 , σ β 2 , ..., σ β r ) = [σ β 1 (XW d r )1, ..., σ β r (XW d r )r](13)
where (XW d r ) i denotes the i-th column of XW d r , which can significantly influence the channelwise outlier distribution in C qkv . To address this, we propose learning an optimal value for β by optimizing it over the calibration dataset D, namely:
min β d∈D ||Y d -Y ′ d || 2 (14
)
where Y d and Y ′ d denote the self-attention block outputs with and without quantization for the d-th sample in the calibration dataset D, respectively. The parameter β is optimized individually for each layer within the VLM. Finally, we apply the quantization operations to both the visual encoder and all layers of the language model, resulting in an end-to-end efficient VLM computation.
this section cite: ['b36', 'b5', 'b15', 'b59', 'b32', 'b1', 'b53']

Section: Evaluation Results
We evaluate QSVD on five VLMs: LLaVA-v1.5 7B [33], LLaVA-v1.5 13B, LLaVA-Next 7B, LLaVA-Next 13B, and SmolVLM-Instruct [39]. To determine the optimal rank allocation and β parameters, we use 256 samples from the ScienceQA training dataset [36], following the procedures outlined in Section 3.2 and Section 3.3. For evaluation, we adopt three widely used benchmark datasets, ScienceQA [36], VizWiz [15], and SEED-Bench-IMG [21], in line with prior work such as LLaVA. We compare QSVD against baseline methods by implementing them on the aforementioned VLM models, the baselines include the SVD approaches (ASVD [60], SVD-LLM [52]) and quantization approach (QuaRot [2], DuQuant [29], QVLM [46]). Specifically, for ASVD and SVD-LLM, we follow their official implementations by applying SVD separately to the Key and Value matrices, while avoiding decomposition of the Query matrices to prevent performance degradation. Additionally, SVD is not applied to other linear layers within the VLM. All methods are evaluated using the same calibration samples and random seeds to ensure fairness, and we report their best performance. For QuaRot and DuQuant, we apply them to the various VLMs by strictly following the detailed procedures provided in their respective code repositories. Given that QSVD performs joint SVD with quantization, we first evaluate its SVD-only performance in by comparing it with ASVD and SVD-LLM under equivalent hardware costs, including intermediate data storage for KV recomputation, weight size, and VLM computational cost in FLOPs. Specifically, we express it in terms of the ratio between ours and the FP16 without compression. We denote the SVD-only approach depicted in Section 4.1 as QSVD-noQ. We then apply the quantization techniques introduced in Section 3.3 to the SVD-compressed VLM and compare the results with advanced quantization methods such as DuQuant [29] and QVLM [46], as well as quantized version of ASVD (QASVD). For QASVD, we apply QuaRot [2] to the SVD-truncated VLMs obtained from ASVD. The corresponding results are presented in Section 4.2.
We evaluate all performance results on NVIDIA RTX A6000 GPUs using VLMEvalKit [10], and we report results under three weight-activation quantization configurations: W8A8 (8-bit weights and 8-bit activations), W8A4, and W4A4. For activation quantization, we apply per-token symmetric quantization. For weight quantization, we use round-to-nearest (RTN) with per-channel symmetric quantization and a learnable clipping ratio, determined via linear search to minimize squared error, following [2]. We present ablation studies in Section 4.3 and evaluate the latency improvements of QSVD on GPU in Section 4.4. Additional results are included in Appendix A.2.
this section cite: ['b32', 'b38', 'b35', 'b35', 'b14', 'b20', 'b59', 'b51', 'b1', 'b28', 'b45', 'b28', 'b45', 'b1', 'b9', 'b1']

Section: Accuracy Evaluation on QSVD-noQ
We begin by evaluating the QSVD-noQ performance in FP16 under four different rank budgets k. To ensure a fair comparison, we adjust the rank configurations of all methods such that our approach consistently maintains the lowest hardware cost in terms of intermediate data storage (η), weight size (α), and computational overhead (γ), as mentioned in Section 3.1. Importantly, as noted in Section 3.1, the relative ratios among the weight sizes α fp , α qsvd , and α ind are identical to the ratios among the computational costs γ fp , γ qsvd , and γ ind . This equivalence allows us to report a single normalized metric to represent both weight parameter reduction and computational efficiency. Therefore we have R 1 and R 2 , defined as:
R1 = αi α f p = γi γ f p R2 = ηi η f p (15
)
where i can either be "qsvd" or "ind". The evaluation results are presented in Table 1. Our method outperforms ASVD and SVD-LLM in accuracy while incurring minimal or comparable hardware cost. On LLaVA-v1.5 13B, QSVD-noQ results in less than a 1% drop in ScienceQA-IMG accuracy compared to the original FP16 baseline, and notably even surpasses the FP16 performance on VizWiz. For instance, with R 1 = 46.7% and R 2 = 17.5%, QSVD-noQ achieves an accuracy of 55.79%, exceeding the FP16 counterpart by more than 2%. This may be due to the low-rank approximation effectively mitigating hallucinations [32] in the VLM; however, further investigation is needed to confirm this hypothesis. Moreover, our approach consistently achieves higher accuracy than ASVD and SVD-LLM under reduced parameter and cache ratios (R 1 and R 2 ), with the performance gap widening as these ratios decrease. For instance, in the SmolVLM setting, our method attains over 70% accuracy on ScienceQA-IMG, while both ASVD and SVD-LLM fail to operate effectively.
this section cite: ['b31']

Section: Accuracy Evaluation of QSVD
The low-rank SVD components are subsequently quantized using the techniques described in Section 3.3. We compare QSVD with DuQuant [29] and QVLM [46], as well as QASVD, which integrate QuaRot's quantization approach with the low-rank SVD outputs from ASVD, respectively. Quantization is applied consistently across the entire VLM, including both feed-forward and selfattention layers in the language model and visual encoder. Evaluations are conducted on three benchmark datasets: ScienceQA, VizWiz, and SEEDBench. All methods maintain a similar R 1 of approximately 50%, while R 2 , which has a greater impact on inference latency and cache size, varies across approaches. Notably, QSVD consistently achieves a lower R 2 compared to all other baselines.
As shown in Table 2, under the W8A8 setting, QSVD consistently outperforms other baselines in most scenarios. On large-scale models like LLaVA-1.5 13B, it reaches accuracy comparable to the FP16 baseline while reducing QKV weights and compute by 50%, and cutting intermediate data size to just 18.75%. Under the more aggressive W8A4 setting, QSVD surpasses all baselines and approaches FP16-level performance using as little as 9.38% of the original KV cache. Finally, Table 2 shows the quantization results under the W4A4 setting. Under this configuration, QASVD fail to operate properly (yielding zero accuracy). Despite the challenging conditions, QSVD consistently delivers the highest performance among all models while maintaining the lowest hardware cost in terms of R 1 and R 2 .
this section cite: ['b28', 'b45']

Section: Ablation Study

this section cite: []

Section: Effectiveness of Cross-layer Rank Allocation Scheme
To evaluate the effectiveness of our cross-layer rank allocation strategy (Section 3.2), we compare it with two baseline methods. The first, referred to as the Uniformrank (UR) scheme, applies SVD to the joint QKV weights using the same rank across all VLM blocks. The second, denoted as the Fisher Information-Based (FIB) scheme, also applies SVD to the joint QKV weights but distributes ranks across layers based on Fisher information. This approach has been adopted in prior work for SVD-based compression in LLMs [5]. All methods operate under the same hardware budget, defined by R 1 and R 2 . As shown in Table 3, under aggressive compression, QSVD-noQ consistently outperforms both baselines and maintains accuracy close to the FP16 model. Impact of Learning β As described in Equation 14, we train β to suppress outliers in the intermediate result C qkv . Table 4 presents the impact of the learnable β on VLM accuracy under W4A4 setting over LLaVA 7Bs on Science QA. We compare QSVD with baseline methods using a fixed β across the entire VLM. QSVD consistently achieves the highest accuracy, outperforming all fixed-β baselines, highlighting the importance of learning β for effective low-bit quantization. Long Sequence Scenarios To further evaluate the adaptability of our QSVD method under long sequence conditions, we conduct experiments on the HRBench-4K dataset [50], which consists of 4Kresolution images. We follow the same evaluation setup as mentioned above and use VLMEvalKit [10] to report the "Average All" accuracy metric. Both LLaVA-Next 13B and LLaVA-v1.5 13B are evalu- ated under our QSVD-noQ configuration and compared against ASVD and FP16 baselines. The results are summarized in Table 5.
As shown in Table 5, QSVD-noQ consistently outperforms ASVD in all evaluation settings. Moreover, the relative performance trends on HRBench-4K closely mirror those observed on ScienceQA-IMG and VizWiz, indicating that our rank allocation strategy generalizes effectively to long sequence scenarios resulting from high-resolution visual inputs.
this section cite: ['b4', 'b49', 'b9']

Section: Impact of QSVD on Hallucination
We further evaluate the impact of QSVD on VLM hallucination using HallusionBench [14], following the same evaluation setup as mentioned above. Metrics include aAcc, fAcc, and qAcc from HallusionBench and their overall average score. As shown in Table 6, both LLaVA-v1.5 13B and LLaVA-Next 13B exhibit noticeable improvements in groundedness metrics after QSVD-noQ. For LLaVA-Next 13B, the overall score increases from 36.02 to a peak of 37.53 at R 1 = 70%. Similarly, LLaVA-v1.5 13B improves from 26.70 to 30.30 at R 1 = 70%, marking a clear reduction in hallucination. These findings confirm that QSVD-noQ not only reduces model and cache size but also acts as an effective regularizer against hallucinations. This effect explains why, on certain benchmark datasets such as VizWiz, the QSVD-compressed models occasionally outperform their original FP16 counterparts in terms of end-task accuracy. QSVD leverages both SVD and quantization to jointly compress model weights and KV cache, making it well-suited for deployment on memory-constrained hardware. We evaluate inference latency of the layer-wise LLaVA-v1.5 7B on an NVIDIA RTX 4070 GPU with 12GB memory. The batch size is set to 1 and the token length to 4K. As shown in Figure 4 , under FP16 precision, due to limited GPU memory, both the FP16 baseline and QSVD-noQ require partial offloading to CPU memory. However, QSVD-noQ with 40% and 30% (denoted as noQ-40% and noQ-30%) rank retention benefits from reduced data movement enabled by effective SVD compression, achieving up to a 2.1× speedup over the baseline. Furthermore, QSVD with W8A8 quantization, under 70% and 50% rank retention, completely avoids offloading and achieves a significant speedup of up to 13.1×.
this section cite: ['b13']

Section: Latency Evaluation on VLM

this section cite: []

Section: Conclusion and Limitation
In this work, we proposed QSVD, a unified framework that applies joint singular value decomposition and quantization to compress VLMs efficiently. By decomposing the combined QKV weight matrices and introducing an adaptive cross-layer rank allocation strategy, QSVD significantly reduces computational cost, KV cache size, and model storage with minimal impact on accuracy. Although quantization is applied to all layers of the VLM, compression is mainly focused on the QKV weights in self-attention layers. Future work will explore joint optimization across all model blocks. Additionally, improving VLM efficiency may also make powerful models more accessible, which raises concerns about potential misuse in areas such as surveillance, misinformation, and privacy violations. Further investigation is needed to address these risks.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: Yes, our abstract and introduction accurately reflect the main contributions and scope of the paper. We clearly state that the work proposes a novel application of Singular Value Decomposition (SVD) to the joint QKV matrices in Vision-Language Models (VLMs), with the goal of reducing KV cache size and computational overhead. This aligns with the technical content, which introduces a dynamic SVD rank allocation strategy to balance memory and accuracy. Furthermore, the abstract discusses the extension of this approach through quantization of both weights and activations, which is supported by experimental results in the paper showing better or comparable performance against previous methods.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss the limitation in the last section of the paper. Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: We will inlcude full derivation of our importance score in Appendix A.1.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: Yes, the paper discloses sufficient information to reproduce its main results. It clearly describes the methods for joint QKV SVD, rank allocation, and quantization, with supporting equations and implementation details. The evaluation setup, including models, datasets, and calibration procedures, is well-documented.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] Justification: Our paper does not release new models and assets.
Guidelines
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: Our paper do not involve human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
this section cite: []

Section: Institutional review board (IRB) approvals or equivalent for research with human subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
Answer: [NA] Justification: Our paper does not involve crowdsourcing nor research with human subjects Guidelines: The paper does not involve crowdsourcing nor research with human subjects.
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: This work does not use large language models (LLMs) as an important, original, or non-standard component of the core methodology. LLMs were only used minimally for editing or polishing writing and did not influence the research's scientific content, methodology, or conclusions. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: A Technical Appendices and Supplementary Material
Outline of Appendices A.1: Detailed derivation for Importance Score in Sec. 3.2.
this section cite: []

Section: A.2: Additional experimental results.

this section cite: []

Section: A.3: Case study for QSVD

this section cite: []

Section: A.1 Importance Score derivation
Proof. Recall that the singular value decomposition (SVD) of W is given by:
W = U ΣV T = r i=1 σ i u i v T i
where u i and v i are the i-th left and right singular vectors, and σ i is the i-th singular value.
When truncating σ i , the change in W is:
∆W σi = σ i u i v T i
Then, the inner product between ∆W i and the gradient G
(n) W is: j,k ∆W σi (j, k) • G (n) W (j, k) = ⟨∆W σi , G (n) W ⟩ F
where ⟨•, •⟩ F denotes the Frobenius inner product.
Substitute ∆W σi = σ i u i v T i : ⟨∆W σi , G (n) W ⟩ F = ⟨σ i u i v T i , G (n) W ⟩ F = σ i ⟨u i v T i , G (n) W ⟩ F
Using the property of the Frobenius inner product:
⟨A, B⟩ F = tr(A T B)
we have:
σ i ⟨u i v T i , G (n) W ⟩ F = σ i tr((u i v T i ) T G (n) W ) = σ i tr(v i u T i G (n) W )
By cyclic property of trace:
σ i tr(v i u T i G (n) W ) = σ i tr(u T i G (n) W v i ) Since u T i G (n) W v i is a scalar: σ i tr(u T i G (n) W v i ) = σ i (u T i G (n) W v i )
Note that:
U T G (n) W V ∈ R r×r
and the (i, i)-th diagonal entry is:
[U T G (n) W V ] (i,i) = u T i G (n) W v i Therefore: j,k ∆W σi (j, k) • G (n) W (j, k) = σ i [U T G (n) W V ] (i,i) j,k ∆W σi (j, k) • G W (j, k) = ⟨∆W σi , G W ⟩ F = σ i [U T G W V ] (i,i)
Finally, the importance score Îσi can be computed as follows:
Îσi = 1 N N n=1 σ 2 i U T G (n) W V 2 (i,i)
If a whitening transformation such as ASVD or SVD-LLM is applied prior to SVD, that is, U ΣV T = SVD(W S) where S denotes the whitening matrix, the corresponding importance score can be reformulated as:
Îσi = 1 N N n=1 σ 2 i U T G (n) W S -T V 2 (i,i)
where the term S -T converts the gradient from the original weight space to the whitened space in which the SVD is performed.
this section cite: []

Section: A.2 Full table for experiments
Here we include more detailed experiments table for QSVD method. We evaluate all performance reuslts on NVIDIA RTX A6000 GPUs, we report results under QSVD-noQ and under three weightactivation quantization configurations: W8A8 (8-bit weights and 8-bit activations), W8A4, and W4A4. For activation quantization, we apply per-token symmetric quantization. For weight quantization, we use round-to-nearest (RTN) with per-channel symmetric quantization and a learnable clipping ratio, determined via linear search to minimize squared error, following [2]. For QSVD baseline, we add QuaRot without SVD as an baseline, also for ours method, we use activation clip ratio of 0.85 for vit model and 0.9 for language model, under this setting, we have updated some QSVD accuracy results higher than main paper report.
QSVD-noQ results. Table 7 presents detailed results of QSVD-noQ on SmolVLM [39], LLaVA-v1.5 [33] series, and LLaVA-Next series, along with the corresponding preserved ratios. QSVD-noQ consistently outperforms ASVD and SVD-LLM in accuracy under reduced parameter and cache ratios (R 1 and R 2 ), with the performance gap widening as the compression becomes more aggressive. For instance, in the SmolVLM setting, our method maintains over 70% accuracy on ScienceQA-IMG [36], while both ASVD [60] and SVD-LLM [51] fail to function effectively.
this section cite: ['b1', 'b38', 'b32', 'b35', 'b59', 'b50']

Section: QSVD Results.
We evaluate our proposed QSVD quantization strategy on LLaVA v1.5 and Next series: 7B, 13B, across three benchmarks: ScienceQA, VizWiz, and SEEDBench. Table 8 summarizes performance under two low-bit settings, W8A8 and W8A4.
Under the W8A8 setting, QSVD matches or exceeds prior methods such as Duquant [29], QVLM [46], and QASVD, while reducing the KV cache and intermediate data sizes by up to 50%. Compared to QSVDLLM, our approach avoids the need for manually re-optimizing decomposed matrices while still achieving superior performance.
In the more challenging W8A4 configuration, QSVD continues to deliver robust outputs, reaching levels comparable to the FP16 baseline using just 9.38% of the original KV cache. This demonstrates the scalability of our quantization design under aggressive memory constraints.
Method ScienceQA-IMG ↑ VizWiz ↑ SmolVLM-2B FP16 84.53% 37.07% R1 R2 100% 50.0% 90.0% 42.5% 80.0% 35.0% 70.0% 27.5% 100% 50.0% 90.0% 42.5% 80.0% 35.0% 70.0% 27.5% ASVD 53.84% 7.88% 0.69% 0.10% 6.68% 0.00% 0.00% 0.00% SVDLLM 65.89% 34.61% 9.07% 3.02% 14.86% 1.62% 0.13% 0.00% R1 R2 100% 37.5% 90.0% 33.75% 80.0% 30.0% 70.0% 26.25% 100% 37.5% 90.0% 33.75% 80.0% 30.0% 70.0% 26.25% QSVD-noQ 83.78% 81.70% 79.57% 77.64% 40.67% 39.88% 40.67% 43.84% LLaVA-v1.5 7B FP16 68.01% 50.03% R1 R2 63.3% 22.5% 60.0% 20.0% 56.7% 17.5% 53.3% 15.0% 63.3% 22.5% 60.0% 20.0% 56.7% 17.5% 53.3% 15.0% ASVD 22.36% 19.09% 15.22% 10.81% 47.70% 39.02% 10.10% 8.87% SVDLLM 55.23% 55.03% 49.23% 51.17% 50.10% 50.71% 50.47% 49.99% R1 R2 60.0% 22.5% 53.3% 20.0% 46.7% 17.5% 40.0% 15.0% 60.0% 22.5% 53.3% 20.0% 46.7% 17.5% 40.0% 15.0% QSVD-noQ 66.12% 65.64% 64.06% 61.68% 53.84% 54.19% 53.88% 52.40% LLaVA-v1.5-13B FP16 71.78% 53.63% R1 R2 63.3% 22.5% 60.0% 20.0% 56.7% 17.5% 53.3% 15.0% 63.3% 22.5% 60.0% 20.0% 56.7% 17.5% 53.3% 15.0% ASVD 64.70% 56.92% 46.50% 42.79% 44.48% 40.63% 40.01% 37.87% SVDLLM 71.44% 71.44% 71.29% 70.50% 51.03% 51.15% 49.37% 46.49% R1 R2 60.0% 22.5% 53.3% 20.0% 46.7% 17.5% 40.0% 15.0% 60.0% 22.5% 53.3% 20.0% 46.7% 17.5% 40.0% 15.0% QSVD-noQ 71.79% 71.74% 71.74% 70.80% 56.15% 56.05% 55.79% 54.04% LLaVA-Next 7B FP16 69.51% 54.46% R1 R2 63.3% 22.5% 60.0% 20.0% 56.7% 17.5% 53.3% 15.0% 63.3% 22.5% 60.0% 20.0% 56.7% 17.5% 53.3% 15.0% ASVD 50.72% 47.15% 40.26% 25.73% 47.78% 47.3% 39.41% 6.69% SVDLLM 65.94% 66.14% 64.90% 62.87% 48.01% 48.41% 47.74% 47.73% R1 R2 60.0% 22.5% 53.3% 20.0% 46.7% 17.5% 40.0% 15.0% 60.0% 22.5% 53.3% 20.0% 46.7% 17.5% 40.0% 15.0% QSVD-noQ 69.91% 68.22% 67.03% 65.15% 54.38% 52.31% 51.42% 49.86% LLaVA-Next-13B FP16 73.23% 57.72% R1 R2 63.3% 22.5% 60.0% 20.0% 56.7% 17.5% 53.3% 15.0% 63.3% 22.5% 60.0% 20.0% 56.7% 17.5% 53.3% 15.0% ASVD 69.71% 68.86% 67.43% 64.01% 55.42% 54.97% 54.50% 52.95% SVDLLM 70.30% 69.71% 69.56% 68.52% 53.08% 52.54% 52.52% 51.77% R1 R2 60.0% 22.5% 53.3% 20.0% 46.7% 17.5% 40.0% 15.0% 60.0% 22.5% 53.3% 20.0% 46.7% 17.5% 40.0% 15.0% QSVD-noQ 72.63% 72.29% 72.34% 71.64% 55.48% 55.14% 54.99% 55.77%
Table 7: Accuracy on ScienceQA-IMG and VizWiz datasets. The R 1 , R 2 denotes the proportion of preserved QKV parameters and the corresponding cache ratio.
For completeness, Appendix Tables 8 and 9 report full results across all model variants and bitwidth configurations. Notably, our method consistently ranks highest or near-highest across settings, while maintaining favorable compression ratios of 50%/18.75% in W8A8 and 50%/9.38% in W8A4. These results highlight the ability of QSVD to balance compression and output quality across a diverse range of architectures and tasks.
this section cite: ['b28', 'b45']

Section: A.3 Case study for QSVD
For case study, we randomly selected examples from the ScienceQA [36] test set, which demonstrate our method's superior performance over QVLM and QASVD baselines.
• As shown in Case 2 and 6, where the FP16 model fails to produce the correct answers, QSVD exhibits more consistent responses between W8A8 and W4A4 settings, and does not contradict the FP16 outputs.
• QSVD also demonstrates surprisingly better performance under low-bit settings in Case 1 and 7, where the FP16 model answers incorrectly but QSVD at W4A4 produces the correct response. This may be attributed to the quantization process reducing model noise, as discussed in Dobi-SVD [49], and may partially explain why QSVD sometimes surpasses FP16 accuracy, as observed in Appendix A.2.
• In Case 3, 4, 5, and 8, QSVD successfully answers questions that FP16 answers correctly but where other baselines fail under W4A4, highlighting its robustness under extreme quantization.
• Furthermore, in Case 4, 5, and 7, other baselines output random tokens or repeat content under 4-bit settings, while QSVD maintains coherent and relevant responses.
this section cite: ['b35', 'b48']

Section: 
(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
Answer: [Yes] Justification: We submit the code as the supplementary materials.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes]
this section cite: []

Section: Justification: We include necessary experiments specifications in supplementary materials
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: Yes, we report the average results over 5 random seeds in all our quanization part. And for calculation of our reported accuracy we follow the opensource VLM evaluation framework.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: All the details are provided in the supplementary materials.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: We have read and acknowledge the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: We mentioned it in the last section of the paper.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: Our paper does not involve the release of data or models that carry a high risk of misuse, and therefore no specific safeguards are necessary in this context.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: All code, data, models utilized in this paper are cited and credited.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset. Table 9: Quantization results on W4A4. To determine which jar has the higher temperature, we need to compare the average kinetic energy of the particles in each jar. If the particles in jar A have a higher average kinetic energy, then jar A has a higher temperature. If the particles in jar B have a higher average kinetic energy, then jar B has a higher temperature. However, the image does not provide enough information to determine the average kinetic energy of the particles in each jar. Therefore, we cannot confidently determine which jar has the higher temperature.
this section cite: []

Section: QASVD W8A8

this section cite: []

Section: QVLM W8A8 Response:
The average kinetic energy of the particles in each sample can be determined by calculating the average kinetic energy of each colored ball. The higher the average kinetic energy, the higher the temperature of the sample.
In the image, there are 28 colored balls in each sample. The average kinetic energy of the particles in sample A is calculated as follows: (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/ QASVD W8A8 Response: The average kinetic energy of the particles in sample A is higher than that of sample B. This means that sample A has a higher temperature than sample B. QSVD W8A8 Response: The diagrams show two samples of gas, each in a closed, rigid container. The containers have the same number of colored balls, which represent the gas particles. The samples have different average kinetic energies, which can be determined by comparing the average kinetic energy of the particles in each sample. Sample A has a higher average kinetic energy than Sample B. This is because the balls in Sample A are moving faster than the balls in Sample B. The average kinetic energy of the particles in Sample A is higher than that of Sample B, which means that Sample A has a higher temperature. QSVD W4A4 Response: The average kinetic energy of the particles in sample A is higher than that of sample B. Note that QASVD and QVLM W4A4 output nothing or random number. Case 3: LLaVA-v1.5-13b Question: What is the capital of Louisiana? A. Atlanta B. Frankfort C. New Orleans D. Baton Rouge   In the image, there are two pairs of magnets, each consisting of two magnets. The magnets in different pairs do not affect each other, and all the magnets are made of the same material. The statement "The magnetic force is stronger in Pair 1" suggests that the magnetic force between the magnets in Pair 1 is stronger than the magnetic force between the magnets in Pair 2. This could be due to various factors such as the orientation, size, or shape of the magnets in each pair. However, without more information or a clear visual comparison, it is not possible to definitively determine which pair has a stronger magnetic force.
this section cite: []

Section: QVLM W8A8 Response:
The magnetic force is stronger in Pair 2.
this section cite: []

Section: QVLM W4A4 Response:
1. The magnetic force is stronger in Pair 1.
this section cite: []

Section: QASVD W8A8 Response:
The magnetic force is stronger in Pair 2.
this section cite: []

Section: QSVD W8A8 Response:
1. The magnetic force is stronger in Pair 1.
In the image, there are two pairs of magnets, each consisting of three magnets. The magnets in each pair are arranged in a straight line, and they are made of the same material. The magnetic force between the magnets in each pair is determined by their strength, size, and arrangement. Since the magnets in Pair 1 are larger and more closely spaced, the magnetic force between them is likely to be stronger than the force between the magnets in Pair 2.
this section cite: []

Section: QSVD W4A4 Response:
The magnetic force is stronger in Pair 1.
Note that QASVD W4A4 output a random number.
Case 7: LLaVA-v1.5-13b Question: Which ocean is highlighted? A. the Indian Ocean B. the Atlantic Ocean C. the Pacific Ocean D. the Southern Ocean FP16 Response: D. the Southern Ocean QVLM W8A8 Response: D. the Southern Ocean QVLM W4A4 Response: the Southern Ocean QASVD W8A8 Response: C. the Pacific Ocean QASVD W4A4 Response: 2. C. the Pacific Ocean QSVD W8A8 Response: D. the Southern Ocean QSVD W4A4 Response: B. the Atlantic Ocean Case 8: LLaVA-v1.5-13b
Question: This organism is Fromia monilis. It is a member of the animal kingdom. Fromia monilis is commonly called a tile sea star. Did you notice that this tile sea star is missing one of its arms? If a tile sea star loses an arm, it can regrow the arm over time. This ability is called regeneration. Can Fromia monilis cells make their own food? A. yes B. no FP16 Response: 1. Fromia monilis is a type of sea star, also known as a tile sea star. 2. The sea star is missing one of its arms, which can regrow over time. 3. Fromia monilis cells cannot make their own food , as they are not capable of photosynthesis.
this section cite: []

Section: QVLM W8A8 Response:
1. Fromia monilis is a type of sea star, also known as a tile sea star. 2. The sea star is missing one of its arms, which can regrow over time. 3. Fromia monilis cells cannot make their own food, as they are not capable of photosynthesis.
this section cite: []

Section: QVLM W4A4 Response:
1. Fromia monilis is a type of sea star that belongs to the animal kingdom. 2. Fromia monilis is commonly known as a tile sea star. 3. The tile sea star is missing one of its arms. 4. Fromia monilis can regrow its arms over time, a process called regeneration. 5. Fromia monilis cells can make their own food. QASVD W8A8 Response: Fromia monilis is a type of sea star, which is an invertebrate. Invertebrates are animals that do not have a backbone. They are typically found in aquatic environments, such as oceans and seas. Sea stars, like Fromia monilis, are known for their ability to regrow lost body parts, such as arms. This process is called regeneration. Sea stars are not capable of making their own food, as they are primarily herbivores. They feed on various types of algae and other plant materials found in their environment.
this section cite: []

Section: QASVD W4A4 Response: 1. A. Yes

this section cite: []

Section: QSVD W8A8 Response:
1. Fromia monilis is a type of sea star, also known as a tile sea star. 2. The sea star is missing one of its arms, which can regrow over time. 3. Fromia monilis cells cannot make their own food, as they are not capable of photosynthesis.
this section cite: []

Section: QSVD W4A4 Response:
Fromia monilis is a type of sea star that is commonly found in the ocean. It is a member of the animal kingdom, and it is commonly known as a tile sea star.
Fromia monilis has the ability to regenerate lost body parts, such as an arm. This ability is called regeneration.In terms of whether Fromia monilis cells can make their own food, the answer is no. Fromia monilis is a carnivorous organism, which means it relies on other organisms for food. It feeds on small marine animals such as crabs, clams, and other invertebrates.
this section cite: []

Section: References
Ref_id:b0 Title: Gqa: Training generalized multi-query transformer models from multi-head checkpoints Year: (2023)
Ref_id:b1 Title: Quarot: Outlier-free 4-bit inference in rotated llms Year: (2024)
Ref_id:b2 Title: Vision-language model for visual question answering in medical imagery Year: (2023)
Ref_id:b3 Title: Paligemma: A versatile 3b vlm for transfer Year: (2024)
Ref_id:b4 Title: Kv-cache compression with low-rank projection Year: ()
Ref_id:b5 Title: Compressing kv-cache with low-rank projection Year: (2024)
Ref_id:b6 Title: Prompt-rsvqa: Prompting visual context to a language model for remote sensing visual question answering Year: (2022)
Ref_id:b7 Title: Visualgpt: Data-efficient adaptation of pretrained language models for image captioning Year: (2022)
Ref_id:b8 Title: 8-bit matrix multiplication for transformers at scale. Advances in neural information processing systems Year: (2022)
Ref_id:b9 Title: Vlmevalkit: An open-source toolkit for evaluating large multi-modality models Year: (2024)
Ref_id:b10 Title: Vlrm: Vision-language models act as reward models for image captioning Year: (2024)
Ref_id:b11 Title: Gptq: Accurate post-training quantization for generative pre-trained transformers Year: (2022)
Ref_id:b12 Title: The llama 3 herd of models Year: (2024)
Ref_id:b13 Title: Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models Year: (2024)
Ref_id:b14 Title: Vizwiz grand challenge: Answering visual questions from blind people. CVPR Year: (2018)
Ref_id:b15 Title: Language model compression with weighted low-rank factorization Year: (2022)
Ref_id:b16 Title: Scaling up vision-language pre-training for image captioning Year: (2022)
Ref_id:b17 Title: Vision-language model for generating textual descriptions from clinical images: Model development and validation study Year: (2024)
Ref_id:b18 Title: Principal component analysis: a review and recent developments Year: (2016)
Ref_id:b19 Title: Squeezellm: Dense-and-sparse quantization Year: (2023)
Ref_id:b20 Title: Seed-bench: Benchmarking multimodal large language models Year: (2024)
Ref_id:b21 Title: Searchlvlms: A plug-and-play framework for augmenting large vision-language models by searching up-to-date internet knowledge Year: (2024)
Ref_id:b22 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b23 Title: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation Year: (2022)
Ref_id:b24 Title: Svdqunat: Absorbing outliers by low-rank components for 4-bit diffusion models Year: (2024)
Ref_id:b25 Title: Jarvis-vla: Post-training large-scale vision language models to play visual games with keyboards and mouse Year: (2025)
Ref_id:b26 Title: Modality-balanced quantization for large vision-language models Year: (2024)
Ref_id:b27 Title: Adasvd: Adaptive singular value decomposition for large language models Year: (2025)
Ref_id:b28 Title: Duquant: Distributing outliers via dual transformation makes stronger quantized llms Year: (2024)
Ref_id:b29 Title: Awq: Activation-aware weight quantization for on-device llm compression and acceleration Year: (2024)
Ref_id:b30 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b31 Title: A survey on hallucination in large vision-language models Year: (2024)
Ref_id:b32 Title: Visual instruction tuning Year: (2023)
Ref_id:b33 Title: Qilin-med-vl: Towards chinese large vision-language model for general healthcare Year: (2023)
Ref_id:b34 Title: Llm quantization with learned rotations Year: (2024)
Ref_id:b35 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: ()
Ref_id:b36 Title: A tutorial on fisher information Year: (2017)
Ref_id:b37 Title: Affinequant: Affine transformation quantization for large language models Year: (2024)
Ref_id:b38 Title: Redefining small and efficient multimodal models Year: (2025)
Ref_id:b39 Title: Compressing pre-trained language models by matrix decomposition Year: (2020)
Ref_id:b40 Title: Omniquant: Omnidirectionally calibrated quantization for large language models Year: (2023)
Ref_id:b41 Title: Fast transformer decoding: One write-head is all you need Year: (2019)
Ref_id:b42 Title: Leveraging large vision-language model as user intent-aware encoder for composed image retrieval Year: (2025)
Ref_id:b43 Title: Towards vision assistants for video games Year: (2025)
Ref_id:b44 Title: Quip#: Even better llm quantization with hadamard incoherence and lattice codebooks Year: (2024)
Ref_id:b45 Title: Q-vlm: Post-training quantization for large vision-language models Year: (2024)
Ref_id:b46 Title: Surgical-lvlm: Learning to adapt large vision-language model for grounded visual question answering in robotic surgery Year: (2024)
Ref_id:b47 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b48 Title: Dobi-svd: Differentiable svd for llm compression and some new perspectives Year: (2025)
Ref_id:b49 Title: Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models Year: (2025)
Ref_id:b50 Title: Svd-llm v2: Optimizing singular value truncation for large language model compression Year: (2025)
Ref_id:b51 Title: Svd-llm: Truncation-aware singular value decomposition for large language model compression Year: (2024)
Ref_id:b52 Title: Training transformers with 4-bit integers Year: (2023)
Ref_id:b53 Title: Dfrot: Achieving outlier-free and massive activation-free for rotated llms with refined rotation Year: (2024)
Ref_id:b54 Title: Smoothquant: Accurate and efficient post-training quantization for large language models Year: (2023)
Ref_id:b55 Title: Advancing multimodal large language models with quantization-aware scale learning for efficient adaptation Year: (2024)
Ref_id:b56 Title: Rotated runtime smooth: Training-free activation smoother for accurate INT4 inference Year: (2025)
Ref_id:b57 Title: Effectively compress kv heads for llm Year: (2024)
Ref_id:b58 Title: Tinygpt-v: Efficient multimodal large language model via small backbones Year: (2023)
Ref_id:b59 Title: Asvd: Activation-aware singular value decomposition for compressing large language models Year: (2023)
Ref_id:b60 Title: Vision-language models for vision tasks: A survey Year: (2024)
Ref_id:b61 Title: Tinyllava: A framework of small-scale large multimodal models Year: (2024)
Ref_id:b62 Title: Unified visionlanguage pre-training for image captioning and vqa Year: (2020)
