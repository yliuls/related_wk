Title: SmallKV: Small Model Assisted Compensation of KV Cache Compression for Efficient LLM Inference
Abstract: KV cache eviction has emerged as an effective solution to alleviate resource constraints faced by LLMs in long-context scenarios. However, existing token-level eviction methods often overlook two critical aspects: (1) their irreversible eviction strategy fails to adapt to dynamic attention patterns during decoding (the saliency shift problem), and (2) they treat both marginally important tokens and truly unimportant tokens equally, despite the collective significance of marginal tokens to model performance (the marginal information over-compression problem). To address these issues, we design two compensation mechanisms based on the high similarity of attention matrices between LLMs of different scales. We propose SmallKV, a small model assisted compensation method for KV cache compression. SmallKV can maintain attention matching between different-scale LLMs to: 1) assist the larger model in perceiving globally important information of attention; and 2) use the smaller model's attention scores to approximate those of marginal tokens in the larger model. Extensive experiments on benchmarks including GSM8K, BBH, MT-Bench, and LongBench demonstrate the effectiveness of SmallKV. Moreover, efficiency evaluations show that SmallKV achieves 1.75 -2.56 times higher throughput than baseline methods, highlighting its potential for efficient and performant LLM inference in resource constrained environments.

Section: Introduction
Large language models (LLMs) [29] have emerged with remarkable natural language understanding capabilities and broad application prospects. Despite the advancements, the deployment of LLMs is hindered with significant computational challenges and high GPU memory consumption, particularly when processing long contexts. This issue arises from the intrinsic complexity of their self-attention mechanism, which scales quadratically with the length of the input sequence. Recent developments in reasoning models, exemplified by ChatGPT-o1 [28] and DeepSeek-R1 [7] have exacerbated this issue due to their lengthy reasoning process.
Numerous studies have demonstrated the high degree of sparsity within attention mechanism, leading to the development of various Key-Value cache (KV cache) compression methods such as quantization [15,9], eviction [51,33,43], and merging [17,38,27]. These methods significantly reduce GPU memory usage and enhance the throughput of inference systems. Our work focuses on eviction-based methods, which identify and retain only the critical tokens to reduce KV cache consumption while minimizing performance degradation in a training-free manner.
Current KV cache eviction methods, however, face two key challenges: (1) the saliency shift issue caused by dynamic changes in token importance during decoding, where permanent token removal strategies become suboptimal when the decoding process evolves, and (2) the classification of tokens into critical/unimportant categories fails to account for marginal tokens that collectively contribute significantly to model performance despite their individually modest attention scores. Existing approaches lack mechanisms to adapt to shifting saliency patterns or to apply differentiated treatment to these three distinct token categories (critical, marginal, and unimportant), leading to either excessive memory consumption or unnecessary quality degradation.
A previous study [4] has revealed a notable similarity in attention patterns between small and large models within the BERT architecture. We further discover the similar observation in the decoder-only architecture models, leading to a novel perspective for addressing the aforementioned limitations. We then propose Small Model Assisted Compensation of KV Cache Compression for Efficient LLM Inference (SmallKV), which introduces a small language model (SLM) to perform saliency shift compensation and marginal information compensation for KV Cache Compression (of LLM). Specifically, the saliency shift compensation mechanism leverages the SLM to maintain global critical information and help identify evicted tokens that may regain significance. Meanwhile, the marginal information compensation identifies tokens with relatively lower attention scores, yet still contribute to model performance. These tokens are less sensitive to the approximation of attention, allowing us to leverage attention scores of SLM for compensation. This results in a hierarchical compression strategy that differentiates and compresses tokens based on their varying levels of importance. It should be noted that SmallKV is compatible with efficient attention implementation such as Flash Attention, which significantly enhances the efficiency of inference. In practical deployment, SmallKV can be used with speculative decoding [18] to speed up LLM inference further.
We conduct a comprehensive evaluation of SmallKV across several benchmarks, including GSM8K [6], BBH [34], MT-bench [54], and Longbench [1]. The experimental results indicate that SmallKV consistently delivers superior performance, especially under low KV cache budgets. Experiments on different model series (e.g., the Qwen series and LLaMA series) and model sizes (ranging from 7B to 72B) highlight the robustness and generalizability of our approach. Efficiency evaluations show that SmallKV achieves 1.75 -2.56 times higher throughput compared to previous KV cache compression methods. These experimental results collectively offer compelling evidence of SmallKV's accuracy and efficiency as a compensation plugin for KV cache compression.
this section cite: ['b28', 'b27', 'b6', 'b14', 'b8', 'b50', 'b32', 'b42', 'b16', 'b37', 'b26', 'b3', 'b17', 'b5', 'b33', 'b53', 'b0']

Section: Related Work
The approaches to KV cache compression can be broadly categorized into three classes: eviction, merging, and quantization. Eviction [30,31,19,25,55,49] aims to retain only a small set of critical tokens' KV caches to achieve nearly lossless inference performance. Merging leverages the high angular similarity observed among deep KV caches to reduce both intra-layer [17,38,27,50,37,39] and cross-layer [45,23] redundancies. Quantization compresses the data by mapping the original full-precision tensor values to discrete levels and storing them at lower precision, including model weight quantization [33,46,32] and KV-cache-only quantization [15,9,44,8,16].
Numerous eviction methods have been proposed, which can be categorized into static and dynamic strategies. Static strategies [10,21,47] perform token filtering during prefilling and maintain KV cache of fixed size throughout the subsequent decoding steps (e.g., sliding window attention [2]). A further approach is to maintain both the initial and the recent tokens [41,12]. Dynamic strategies [51,43,52,5] continuously update the critical KV cache during decoding, while the KV cache of unimportant tokens will be permanently removed or offloaded from the GPU. The core of permanent eviction lies in selecting critical tokens to minimize the damage to model accuracy. H 2 O [51] observed that almost all layers exhibit a sparsity exceeding 95%, indicating that maintaining just 5% of the KV cache based on the accumulative attention scores of each token is sufficient for decoding the same output token at each generation step. PyramidInfer [43] employs a pyramid-shaped hierarchical processing approach, where recent tokens are assigned greater weight and the length of KV caches in deeper layers is reduced. Permanent token removal has two main limitations. Irreversible token eviction can degrade model performance in multi-turn dialogue scenarios and long-sequence tasks. Additionally, relying on attention scores prevents the model from adapting to some acceleration techniques (e.g., Flash Attention). Consequently, non-permanent eviction [40,35,48,14,24] has been proposed by dynamic cache offloading based on indexing. These methods manage and access the KV cache offloaded to multi-tier cache systems at the granularity of chunks or clusters. However, achieving fast and accurate retrieval with high precision remains challenging, while the index construction for retrieval introduces significant decoding latency.
Speculative decoding [20,18,3,53] is a widespread technology to accelerate inference. It employs a pair of models, where the smaller one generates candidate tokens in advanced and the larger one verifies them in parallel. By enabling parallel token generation, speculative decoding addresses the inefficiencies inherent in traditional autoregressive approaches, achieving speedups of over three times without compromising the quality of the generated output. This technique has been widely adopted in commercial LLMs [3,22].
this section cite: ['b29', 'b30', 'b18', 'b24', 'b54', 'b48', 'b16', 'b37', 'b26', 'b49', 'b36', 'b38', 'b44', 'b22', 'b32', 'b45', 'b31', 'b14', 'b8', 'b43', 'b7', 'b15', 'b9', 'b20', 'b46', 'b1', 'b40', 'b11', 'b50', 'b42', 'b51', 'b4', 'b50', 'b42', 'b39', 'b34', 'b47', 'b13', 'b23', 'b19', 'b17', 'b2', 'b52', 'b2', 'b21']

Section: Pilot Observation

this section cite: []

Section: Saliency Shift Issue
The dynamic nature of LLM decoding leads to the shift in token saliency, which is the change of the token set with high attention over time. Most existing KV cache eviction strategies, which rely on permanent token removal [51,43], are highly susceptible to this issue. To address this, we leverage a small language model (SLM) from the same model series to approximate the LLM's attention scores. This approach aligns well with speculative decoding [20,18,3,53,22], where the SLM generates candidate tokens in advance and the LLM verifies them in parallel, thereby allowing the combination of these two appealing approaches for further optimization of inference speed.
this section cite: ['b50', 'b42', 'b19', 'b17', 'b2', 'b52', 'b21']

Section: Observation 1.
The saliency shift leads to the discrepancy of the important tokens between the compressed KV cache view and the uncompressed global view, and this gap causes the model to lose critical information during inference. We mimic the continual process of KV cache compression in Figure 1 (a). Here, in "real-drop" procedure, we perform two times compression, mimicking the continual compression in real decoding scenario. In the first compression, the important token (yellow) information in the first half of the dashed line is retained, while the remaining unimportant tokens (blue) are evicted and their KV cache are permanently removed. The important tokens, along with the new ones in the second half of the dotted line, continue to participate in subsequent decoding and the second-time compression. However, at the second time compression, the global view of an uncompressed cache shows differences in the selection of important tokens (red), i.e., the shift in token saliency between the second compression step and the global view-based compression.
Specifically, we quantified the gap caused by saliency shift on the Wikitext-v2 [26] dataset using Qwen2-7B in Figure 1 (b). The importance of tokens is measured by accumulative attention scores, following H 2 O [51]. We measure the discrepancy between sets of important tokens derived from the real drop view and the global view using Jaccard similarity. The relatively low Jaccard similarity values, ranging from 0.55 to 0.77, indicate a significant difference between realistic KV cache eviction methods and the global ground truth, which means a considerable number of globally important tokens are erroneously evicted under the real drop method. In addition, the similarity maintains a clear declining trend as the KV cache budget decreases, indicating that the influence brought by saliency shift is more serious. This is particularly concerning since most current KV cache compression methods claim to operate with a low KV cache budget (e.g., 10%-30%). Therefore, an effective approach to compensate for saliency shift issue is critical and necessary.
this section cite: ['b25', 'b50']

Section: Insights 1.
The core issue with the saliency shift lies in the inability of existing KV cache compression methods to effectively maintain past information after compression. These methods predominantly focus on LLM itself without considering external information. We, however, observe that LLMs of different sizes within the same series tend to exhibit highly consistent attention patterns. Therefore, we propose a collaborative approach during inference, where an assisted SLM works with the LLM in parallel, using the attention matrices of the SLM for saliency shift compensation.  We measure the similarity of the attention patterns between Qwen2-0.5B and Qwen2-7B on Wikitext-v2 [26] dataset. Figure 1 (c) illustrates one example of the trends of accumulative attention scores between SLM and LLM, as more samples and detailed discussion in Appendix B. For each attention head in LLM, we search the most similar one in the SLM for matching. The quantitative analysis shows that average cosine similarity between Qwen2-0.5B and Qwen2-7B after similarity matching reaches 0.947. The result also visualizes a high degree of consistency, particularly among tokens with high attention scores. This finding indicates a promising way to effectively address the saliency shift issue without introducing significant overhead. We evaluate the effectiveness of saliency shift compensation with 10% KV cache budget across six downstream tasks, as shown in Figure 1 (d).
The results demonstrate that using Qwen2-0.5B as the assisted SLM for saliency shift compensation significantly outperforms using only Qwen2-7B without any compensation.
this section cite: ['b25']

Section: Marginal Information Overcompression
Current KV cache eviction methods divide tokens into two categories-critical and unimportant-overlooking the presence of marginal tokens. These tokens, while less impactful than critical ones, still hold significant relevance and collectively contribute far more to model performance than their attention score suggests. Unlike unimportant tokens, marginal tokens are essential for preserving output quality, yet they are currently subjected to the same aggressive compression or eviction strategies as truly negligible tokens. To address this, a dedicated approach is needed to: (1) distinguish marginal tokens from unimportant ones, and (2) apply tailored compression strategies (rather than outright removal) to retain their contribution.
Observation 2. We measure the accumulative attention scores of each token across all heads of Qwen2.5-14B on the BBH dataset [34], and statistically rank these scores in Figure 2 to reveal the necessity of maintaining marginal tokens. Consistent with previous research findings, we also observe that only a small fraction of tokens account for the majority of attention scores. The sum of the accumulative attention score for the top 5% of tokens is approximately 7 times greater than that of the 5%-10% range, while the 5%-10% range is only 1.34 times higher than the 10%-15% range, highlighting the dominant position of the top 5% of tokens in attention scores. However, the compression of tokens in range of 5%-15% has caused the accuracy to drop from 0.482 to 0.148 under H 2 O method.
Insights 2. Previous methods, however, fail to recognize the necessity of maintaining marginal tokens, and conflate them with critical tokens or unimportant tokens. This leads to an underutilization of attention sparsity, inducing a mandatory trade-off between KV cache consumption and model performance degradation. Our observations suggest that maintaining marginal tokens is necessary and that they can tolerate approximation compared to critical tokens (extended analysis can be referred to Appendix C). Therefore, a reasonable approach is to apply different compression strategies for tokens of varying importance levels, as shown in Figure 2. Critical tokens retain the full KV cache to avoid precision loss, marginal tokens retain only the V cache and use SLM to approximate the attention mechanism, and unimportant tokens are evicted completely.
As insight 1 indicates that LLMs of different sizes within the same series exhibit highly consistent attention patterns, we propose compensating for marginal tokens based on the attention scores from the SLM. Specifically, we approximate the attention mechanism of marginal tokens by multiplying their V cache with the corresponding attention scores derived from the SLM. This method reduces the K cache consumption for these tokens while maintaining their contribution to overall model performance. By doing so, we effectively bridge the gap between theoretical attention sparsity and practical performance, thereby enhancing the efficiency and efficacy of KV cache compression.
this section cite: ['b33']

Section: SmallKV Method
We introduce the insight of SmallKV in Section 3, involving saliency shift compensation and marginal tokens compensation assisted by the SLM. In this section, we provide a detailed description.
Similarity Matching. In prefill stage, SmallKV establishes the similarity matching between SLM and LLM. Given an LLM M and the corresponding SLM M s , the prompt is first forward in both models to obtain all attention matrices respectively. The attention matrices of M are denoted as
A i = Sof tmax QiK ⊤ i √ d h ∈ R n×n , 0 ≤ i < L • D,
and those in M s are denoted as A ′ j , 0 ≤ j < l • d. Q ∈ R n×d and K ∈ R n×d denote query matrix and key matrix in attention mechanism. L and l represent the number of layers in the M and M s . D and d denote the number of attention heads per layer in each model. n represents the number of tokens in the prompt. Due to differences in model scaling, we have l • d << L • D. Let C be the KV cache of the context, the accumulative attention score vector in a specific context can be expressed as:
F (A i , C) = (s 1 i , s 2 i , . . . , s n i ),
where
s v i = n u=1 A i [u, v](1)
We then can calculate pairwise similarity between A i and A ′ j according to Jaccard simularity of their TopK indices as:
S(A i , A ′ j ) = T opK (F (A i , C)) ∩ T opK F (A ′ j , C) T opK (F (A i , C)) ∪ T opK F (A ′ j , C)(2)
The mapping function that maps i-th attention matrix of LLM to the j-th one in SLM is obtained by:
f (i) = argmax j S(A i , A ′ j )(3)
Saliency Shift Compensation. Previous eviction strategies remove tokens based on the policy:
E(A i , C) = C \ {v}, where {v} ← argmax {v}∈C F (A i , C) (4
)
The saliency shift problem in the previous methods can be formally defined as:
E(A i , C all ∪ {t}) ̸ = E(A i , C r ∪ {t})(5)
where C all denotes the full KV cache, C r represents the compressed cache obtained after prefilling, and t indicates the recently added token. The equation suggests that the selection of critical tokens from a compressed cache (e.g., after prefilling) is different from that if we have the full cache.  Unlike previous methods, SmallKV maintains the full cache of SLM C s all , and performs eviction for the i cache of LLM based on he f (i)-th (full) cache of SLM and the eviction policy E(A ′ f (i) , C s all ).
this section cite: []

Section: Marginal Information Compensation.
In decoding stage, let the attention be A i ∈ R 1×n , the vanilla attention can be expressed as
O i = A i • V i ∈ R 1×d
, where V i ∈ R n×d denotes the value matrix. The attention mechanism for Marginal Information Compensation can be expressed as
O * i = A * i • V i , where
A * i is calculated as follows: A * [k] =        A i [k], if k ∈ T opK F (A ′ f (i) , C s all ) , A ′ f (i) [k], if k ∈ T op(P -K) F (A ′ f (i) , C s all ) , 0, otherwise.(6)
where K and P -K denote the numbers of critical tokens and marginal tokens, which are determined by KV cache budget.
The formula shows that for marginal tokens that appear in the T op(P -K) importance, we approximate their attention scores using the corresponding elements from SLM M s to avoid using K cache. This approach allows for retaining important information while evicting the K cache of these tokens, effectively implementing a hierarchical compression policy that differentiates between various levels of token importance. The illustration of attention mechanism in SmallKV is shown in Figure 3.
this section cite: []

Section: Method Details.
In practice, to ensure the stability of matching correspondence between SLM and LLM, we dynamically determine the timing of similarity matching based on context length. Specifically, we control the token length for similarity matching within the range of 100 to 200. This is to avoid the matching inaccuracy caused by excessively short token sequences and the distortion of similarity in high-dimensional space caused by overly long token sequences. If the current context length does not meet the minimum threshold, SmallKV delays the timing of similarity matching and KV cache eviction until the requirements are satisfied. Similarly, for context lengths exceeding the threshold, we truncate the prefill token length for similarity calculation.
The detailed procedure of SmallKV can be referred to Algorithm 1. It is important to note that SmallKV does not delete the KV cache but migrates it between GPU HBM and CPU memory, as the process of updating KV cache of critical tokens and V cache of marginal tokens in line 9 of the Algorithm. This migration process executes in parallel with the forward of LLM to prefetch KV cache for latency reduction. Within the attention of LLM, the computation of critical tokens and marginal tokens is also performed in parallel. The computation for critical tokens benefits from acceleration via Flash Attention, whereas the computation for marginal tokens involves only a matrix multiplication. The system architecture of SmallKV is shown in Figure 4.
this section cite: []

Section: Algorithm 1 SmallKV Algorithm
Input: Large language model M ; Assisted small language model M s ; Input x; KV cache budget τ ; 1: if Prefill then 2:
Allocate KV cache budget τ to critical budget τ c , marginal budget τ s 3:
Parallel forward process of M (x) and M s (x) to get attention matrices A i and A ′ j 4:
Establish correspondence of A i and A ′ j by similarity matching (Eq. ( 2) and Eq. ( 3)) 5:
Output new token t i by M (x) and append it to t 6: else 7:
Forward process of M s (x + t) to update A ′ j 8:
In parallel: 9:
-Update KV cache of critical tokens and Update V cache of marginal tokens by E(A ′ f (i) , C s all )
10:
for Attention layer l ∈ layers of LLM do 11:
In parallel:
12:
-Calculate attention output O c of critical tokens using Flash Attention 13:
-Calculate attention output O m of marginal tokens using Eq. ( 6)
14: Attention Output by O c + O m 15: end for 16:
Output new token t i by M (x + t) and append it to t 17: end if Output: Output token sequences t = {t i } L i=1 5 Experiments
this section cite: []

Section: Experiments Setup
Datasets. We comprehensively evaluate the effectiveness of SmallKV in four kinds of scenarios: 1) GSM8K [6] for mathematical reasoning, 2) BBH [34] for language understanding, 3) MT-Bench [54] for multi-turn conversation, and 4) Longbench [1] for long-context scenario.
Baselines. We take two effective KV cache eviction methods as baselines. These methods select critical tokens according to their attention scores, which is the same as SmallKV. H 2 O [51]: employs the accumulative attention score as the metric to evict unimportant KV cache.
PyramidInfer [43]: identifies that deeper layers exhibit greater redundancy and applies differentiated KV cache budgets between layers.
this section cite: ['b5', 'b33', 'b53', 'b0', 'b50', 'b42']

Section: Models.
Our experiments are conducted on the Qwen and LLaMA series of LLMs with different scales. Specifically, in the benchmark results (Section 5.2), we consider four pairwise combinations between the SLM and the LLM. These combinations include: Qwen2-0.5B with Qwen2-7B, Qwen2.5-0.5B with Qwen2.5-14B, Qwen2-7B with Qwen2-72B, LLaMA 3.2-1B with LLaMA 3.1-8B. The Qwen2-72B model is quantized to INT4 data type for efficient computing. The other experiments are primarily performed on the combination of Qwen2-0.5B with Qwen2-7B.
this section cite: []

Section: Implementation Details.
All experiments are conducted on 8 NVIDIA A100 (80GB) GPUs. The configurations of environment include: CUDA (12.0), PyTorch (2.4.0), and huggingFace's Transformersfoot_0 (4.45.1). We use greedy decoding to ensure the stability of the experimental results.
Consistent with previous study, we also allocate a proportion of the KV cache budget to recent tokens. We adopt a fixed ratio of 2:1:2 for allocating resources among critical tokens, recent tokens, and marginal tokens, respectively. For instance, with 20% KV cache budget, we allocate 10%, 5% and 10% budget to critical tokens, recent tokens, and marginal tokens (as marginal tokens only use V cache of LLM, it actually consumes half of budget). When the available KV cache budget becomes limited so that it cannot fully accommodate all critical tokens (e.g. 5%), we proportionally reduce the allocations for recent and marginal tokens to maintain model performance.
this section cite: []

Section: Benchmark Results
Figure 5 shows the performance of four models across a range of KV cache budgets from 100% to 5% on GSM8K, BBH, and MT-Bench. The results indicate that the SmallKV method consistently outperforms baseline approaches across nearly all models and various KV cache budgets, particularly at very low KV cache budgets. The marginal information compensation mechanism enables SmallKV to maintain attention information for a significant proportion of marginal tokens while using few KV cache, thereby sustaining high performance even at low cache budgets. For instance, in the experiment of Qwen2-0.5B paired with Qwen2-7B, with 5% KV cache budget, the scores for H2O and PyramidInfer decline from 79.4 (at full cache) to 36.7 and 47.0, respectively, whereas the SmallKV method maintains the performance score of 73.0.
Table 1 provides a comparative analysis of SmallKV method against baselines on LongBench at three distinct KV cache budgets (τ = 0.3, τ = 0.1, and τ = 0.05). Our method demonstrates superior performance across all five subtasks, underscoring the efficacy of SmallKV in long-context scenarios.
Method Bsz Lenth TPOT (ms) TTFT (s) Thr.(tokens/) Accelerate (Eager) 64 2048+256 59.4 22.15 62.27 (1.00x) Accelerate (Flash) 44.6 1.09 188.23 (3.02x) H2O (20%) 30.5 22.55 76.32 (1.23x) SmallKV (20%) 36.3 2.71 195.50 (3.14x) Accelerate (Eager) 4 16384+512 42.7 14.28 474.21 (1.00x) Accelerate (Flash) 31.3 1.41 990.39 (2.09x) H2O (20%) 20.4 14.32 689.07 (1.45x) SmallKV (20%) 24.2 1.94 1203.42 (2.54x)
Table 2: Evaluation on efficiency of SmallKV. Bsz: batch size. Lenth: prefill length + decode lenth. TPOT: average time per output token in decode stage. TTFT: time to first token. Thr.: End-to-end throughput.
Similar to previous observations, SmallKV performs well with low KV cache budgets. For example, in subtasks such as Multi-Doc QA, Few-shot Learning, and Code Completion, baseline methods exhibit performance degradation with 5% KV cache budget, while the SmallKV method maintains performance comparable to that achieved with full KV cache. This highlights SmallKV's ability to effectively manage cache resources and deliver robust performance even under constrained conditions.
this section cite: []

Section: Efficiency Results

this section cite: []

Section: Setup.
We conduct an efficiency analysis on one NVIDIA A100 GPU. The evaluated model is Qwen2-7B and the SLM used in SmallKV method is Qwen2-0.5B. We use synthetic datasets for testing where all prompts are padded to the same length in the batch and the outputs are also restricted to a fixed length. We consider two common scenarios: 1) Multi-user concurrency: In this scenario, the context is set to a prefill length of 2048 tokens and a decode length of 256 tokens, with a batch size of 64. 2) Extremely long context: the context is set to a prefill length of 16384 tokens and a decode length of 512 tokens, with a batch size of 4. The evaluation metric consists of average time (ms) per output token in decode stage (TPOT), time (s) to first token (TTFT), and end-to-end throughput (token/s). We use Hugging Face Accelerate and H 2 O as baselines. Hugging Face Accelerate employs two attention implementation methods: eager attention and PyTorch SDPA with Flash Attention kernel [11]. H 2 O and SmallKV both employ a KV cache budget of 20%. The times reported in SmallKV include the overhead of the assisted SLM.
this section cite: ['b10']

Section: Results.
The efficiency results are shown in Table 2. It can be observed that compared to Accelerate, KV cache compression techniques (H 2 O and SmallKV) significantly reduce the TPOT during the decoding. This is attributed to their reduction in loading KV cache, thereby alleviating pressure on GPU memory bandwidth and decreasing the computation of attention. Specifically, introducing an additional assisted SLM in SmallKV results in increased latency during the decoding when compared to H 2 O. However, due to its compatibility with memory-efficient attention methods, SmallKV's TTFT is notably lower than that of H 2 O, despite the overhead incurred by similarity matching during the prefill stage. In summary, although the SmallKV method incorporates overhead such as assisted SLM and similarity matching, it benefits both from the advantages of KV cache compression and compatibility with efficient attention methods. These features collectively contribute to its superior efficiency, demonstrating significantly higher throughput than baseline methods across both scenarios.
this section cite: []

Section: Ablation Study
We conduct ablation studies on the BBH benchmark to evaluate the contributions of the saliency shift compensation and marginal information compensation components proposed in the SmallKV.
The experimental results are shown in Figure 6. Firstly, it is evident that the SmallKV method exhibits a notable performance drop in the 10% to 40% KV cache budget range when the marginal information compensation component is absent. This observation aligns with the design purpose of marginal information compensation, which aims to compensate for marginal tokens beyond the small set of critical ones. Additionally, when the KV cache budget is under capacity to cover the critical tokens, the hierarchical compression for marginal tokens fails due to insufficient KV cache budget allocation for those tokens. This is reflected in the performance convergence between the w/o marginal information compensation and the full SmallKV with KV cache budgets below 10%.
If we further remove the saliency shift compensation component from the w/o marginal information compensation setup, as depicted by the w/o both compensation in the figure, there is an additional decline in performance. This highlights the effectiveness of saliency shift compensation, which  mitigates saliency shift issue by utilizing global information from the assisted SLM. This compensation mechanism dynamically depends on context information during generation, thereby enhancing overall performance.
this section cite: []

Section: Impact of SLM Scaling
We also investigate the impact of scaling SLM on the performance of the SmallKV method. Theoretically, as the size of the SLM increases, it can provide more attention matrices, allowing the LLM to match more similar attention matrices for compensation, thereby enhancing the performance of the SmallKV method. Using Qwen2.5-14B as the LLM, we experiment with four differently scaled SLMs: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, and Qwen2.5-7B on the BBH benchmark. The results are presented in Figure 7. Consistent with theoretical expectations, the performance improved gradually as the size of the SLM increased. On average, using Qwen2.5-1.5B, Qwen2.5-3B, and Qwen2.5-7B as SLMs leads to performance improvements of 4.9%, 6.8%, and 8.5% compared to using Qwen2.5-0.5B as SLM.
While the performance improvements are relatively small compared to the increase in parameters of the SLM, we also note that the influence of SLM scaling becomes more pronounced with low KV cache budgets. For instance, with 5% KV cache budget, using Qwen2.5-1.5B, Qwen2.5-3B, and Qwen2.5-7B as SLMs resulted in performance improvements of 30.2%, 63.8%, and 84.4%, respectively, compared to using Qwen2.5-0.5B. This finding indicates that under extremely constrained KV cache budgets, it is crucial to appropriately scale up the SLM to maintain performance levels. In summary, the SmallKV method necessitates a careful trade-off between the overhead introduced by scaling the SLM and the overall performance enhancement in practice.
this section cite: []

Section: Conclusion
We present SmallKV, a novel small-model-assisted compensation framework that addresses two critical limitations in current KV cache eviction methods: the saliency shift problem and the marginal information over-compression issue. By leveraging the high similarity of attention matrices across different model scales, SmallKV introduces two key innovations: (1) preserving globally important attention patterns through SLM assistance, and (2) accurately approximating marginal tokens using the smaller model's attention scores. Our comprehensive evaluation across multiple benchmarks (GSM8K, BBH, MT-Bench, and LongBench) demonstrates that SmallKV consistently maintains model performance even under aggressive KV cache budgets. Notably, SmallKV achieves 1.75×-2.56× throughput improvement over existing methods while remaining compatible with memoryefficient attention implementations like Flash Attention.
this section cite: []

Section: References
Ref_id:b0 Title: LongBench: A bilingual, multitask benchmark for long context understanding Year: (2024-08)
Ref_id:b1 Title: The long-document transformer Year: (2020)
Ref_id:b2 Title: Accelerating large language model decoding with speculative sampling Year: (2023)
Ref_id:b3 Title:  Year: (2021)
Ref_id:b4 Title: Nacl: A general and effective kv cache eviction framework for llms at inference time Year: (2024)
Ref_id:b5 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b6 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b7 Title: Get more with less: Synthesizing recurrence with kv cache compression for efficient llm inference Year: (2024)
Ref_id:b8 Title: Qaq: Quality adaptive quantization for llm kv cache Year: (2024)
Ref_id:b9 Title: Model tells you what to discard: Adaptive kv cache compression for llms Year: (2023)
Ref_id:b10 Title:  Year: ()
Ref_id:b11 Title: Lminfinite: Zero-shot extreme length generalization for large language models Year: (2023)
Ref_id:b12 Title: What matters in transformers? not all attention is needed Year: (2024)
Ref_id:b13 Title: Squeezed attention: Accelerating long context length llm inference Year: (2024)
Ref_id:b14 Title: Towards 10 million context length llm inference with kv cache quantization Year: (2024)
Ref_id:b15 Title: Gear: An efficient kv cache compression recipefor near-lossless generative inference of llm Year: (2024)
Ref_id:b16 Title: Compressed context memory for online language model interaction Year: (2023)
Ref_id:b17 Title: Speculative decoding with big little decoder Year: (2023)
Ref_id:b18 Title: {InfiniGen}: Efficient generative inference of large language models with dynamic {KV} cache management Year: (2024)
Ref_id:b19 Title: Fast inference from transformers via speculative decoding Year: (2023)
Ref_id:b20 Title: Snapkv: Llm knows what you are looking for before generation Year: (2024)
Ref_id:b21 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b22 Title: Minicache: Kv cache compression in depth dimension for large language models Year: (2024)
Ref_id:b23 Title: Retrievalattention: Accelerating longcontext llm inference via vector retrieval Year: (2024)
Ref_id:b24 Title: Clusterkv: Manipulating llm kv cache in semantic space for recallable compression Year: (2024)
Ref_id:b25 Title: Pointer sentinel mixture models Year: (2016)
Ref_id:b26 Title: Dynamic memory compression: Retrofitting llms for accelerated inference Year: (2024)
Ref_id:b27 Title: Introducing openai o1-preview Year: (2024)
Ref_id:b28 Title: OpenAI. Gpt-4 technical report Year: (2024)
Ref_id:b29 Title: Transformers are multi-state rnns Year: (2024)
Ref_id:b30 Title: Sparq attention: Bandwidth-efficient llm inference Year: (2023)
Ref_id:b31 Title: Omniquant: Omnidirectionally calibrated quantization for large language models Year: (2023)
Ref_id:b32 Title: Flexgen: High-throughput generative inference of large language models with a single gpu Year: (2023)
Ref_id:b33 Title: Challenging BIG-bench tasks and whether chain-of-thought can solve them Year: (2023-07)
Ref_id:b34 Title: Quest: Query-aware sparsity for efficient long-context llm inference Year: (2024)
Ref_id:b35 Title: Phi-3 technical report: A highly capable language model locally on your phone Year: (2024)
Ref_id:b36 Title: D2o: Dynamic discriminative operations for efficient generative inference of large language models Year: (2024)
Ref_id:b37 Title: Lossless compressed memory attention Year: (2024)
Ref_id:b38 Title: Model tells you where to merge: Adaptive kv cache merging for llms on long-context tasks Year: (2024)
Ref_id:b39 Title: Infllm: Training-free long-context extrapolation for llms with an efficient context memory Year: (2024)
Ref_id:b40 Title: Efficient streaming language models with attention sinks Year: (2023)
Ref_id:b41 Title: Zhifang Guo, and Zhihao Fan. Qwen2 technical report Year: (2024)
Ref_id:b42 Title: Pyramidinfer: Pyramid kv cache compression for high-throughput llm inference Year: (2024)
Ref_id:b43 Title: No token left behind: Reliable kv cache compression via importance-aware mixed precision quantization Year: (2024)
Ref_id:b44 Title: Efficient inference via layer-wise dissimilar kv cache sharing Year: (2024)
Ref_id:b45 Title: Quantizing weight and key/value cache for large language models gains more Year: (2024)
Ref_id:b46 Title: In-context kv-cache eviction for llms via attention-gate Year: (2024)
Ref_id:b47 Title: Pqcache: Product quantization-based kvcache for long context llm inference Year: (2024)
Ref_id:b48 Title: Unifying kv cache compression for large language models with leankv Year: (2024)
Ref_id:b49 Title: Cam: Cache merging for memory-efficient llms inference Year: (2024)
Ref_id:b50 Title: H2o: Heavy-hitter oracle for efficient generative inference of large language models Year: (2023)
Ref_id:b51 Title: Buzz: Beehive-structured sparse kv cache with segmented heavy hitters for efficient llm inference Year: (2024)
Ref_id:b52 Title: Lookahead: An inference acceleration framework for large language model with lossless generation accuracy Year: (2024)
Ref_id:b53 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b54 Title: Dynamic kv cache compression for long-context modeling based on layer uncertainty Year: (2024)
