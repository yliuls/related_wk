Title: FFN Fusion: Rethinking Sequential Computation in Large Language Models
Abstract: We introduce FFN Fusion, an architectural optimization technique that reduces sequential computation in large language models by identifying and exploiting natural opportunities for parallelization. Our key insight is that sequences of Feed-Forward Network (FFN) layers, particularly those remaining after the removal of specific attention layers, can often be parallelized with minimal accuracy impact. We develop a principled methodology for identifying and fusing such sequences, transforming them into parallel operations that significantly reduce inference latency while preserving model behavior. Applying these techniques to Llama-3.1-405B-Instruct, we create Llama-Nemotron-Ultra-253B-Base (Ultra-253B-Base), an efficient model that achieves a 1.71× speedup in inference latency and 35× lower per token cost while maintaining strong performance across benchmarks. Most intriguingly, we find that even full transformer blocks containing both attention and FFN layers can sometimes be parallelized, suggesting new directions for neural architecture design.

Section: Introduction
Large language models (LLMs) have emerged as one of the most transformative technologies of our time, revolutionizing how we approach artificial intelligence and computation. From powering sophisticated virtual assistants [16] to enabling breakthrough scientific research [1,41], these models have evolved from academic curiosities [35] into indispensable tools that are reshaping entire industries. This transformation has been driven by rapid scaling to hundreds of billions of parameters [33,2,9], enabling unprecedented capabilities in reasoning, generation, and complex problem-solving [5,15]. However, this extraordinary growth in scale and capability comes at a critical cost: the computational demands of running these models have become a fundamental bottleneck. As these models push the boundaries of what is possible with artificial intelligence, their deployment costs and resource requirements severely limit their accessibility, creating an urgent need for innovations that can make their capabilities widely available.
Research in LLM runtime optimization explored diverse approaches to managing computational demands. Traditional techniques like quantization [7,12,8,47]-which reduce memory footprint and accelerate inference through lower-precision arithmetic-and pruning [26,18,17,28,11]-which removes redundant parameters-have become standard tools. A more recent innovation, Mixtureof-Experts (MoE) [38,23], has demonstrated remarkable potential by dynamically activating only a small subset of model parameters during inference. The DeepSeek-V3 architecture [6] pushes this approach to new extremes, employing 256 expert FFN modules at each layer while activating only 8 experts per token (plus one shared expert), effectively achieving the capabilities of a much larger model while maintaining reasonable computational costs through sparse activation. However, each of these approaches faces distinct challenges: quantization encounters precision-accuracy tradeoffs at small bit numbers, pruning struggles to identify significant additional redundancy without compromising accuracy performance (and pruning can be harnessed to efficiency mainly when it is structured), and MoE architectures, while efficient for single-token inference or for very large batches, do not provide optimal throughput at small and intermediate batch sizes due to under-utilization of compute resources, as discussed in Appendix E. Given the above limitations, there is an urgent need for complementary approaches that can unlock new dimensions of efficiency improvement while maintaining the simplicity and predictable scaling characteristics of dense architectures.
In this work, we introduce two major contributions that fundamentally advance the state of LLM efficiency. First, we present FFN Fusion, a novel architectural optimization technique that challenges the conventional sequential nature of transformer computation, and is illustrated in Figure 1. By identifying and exploiting patterns of computational independence in FFN layers, our approach enables parallel execution across multiple GPUs while preserving model functionality. This parallelization is particularly effective on modern GPU nodes, where tensor-parallel implementations often suffer from synchronization delays between consecutive layers. By concentrating the computation into fewer layers and reducing cross-device communication, our method significantly improves hardware utilization. Our findings reveal that substantial portions of LLM computation can be parallelized with minimal accuracy impact, complementing existing runtime optimization techniques like pruning and quantization. Second, we demonstrate the practical impact of these insights through Ultra-253B-Base, a state-of-the-art 253B parameter model derived from Llama-3.1-405B-Instruct (henceforth Llama-405B), derived using FFN Fusion and attention pruning. This model not only showcases the scalability of our approach, but also achieves remarkable efficiency gains while maintaining or exceeding its parent model's capabilities across a comprehensive suite of benchmarks.
this section cite: ['b14', 'b0', 'b38', 'b32', 'b30', 'b1', 'b7', 'b4', 'b13', 'b5', 'b10', 'b6', 'b44', 'b23', 'b16', 'b15', 'b25', 'b9', 'b35', 'b20']

Section: The applicability of FFN Fusion rests on a fundamental insight about modern LLM architectures.
Recent work has shown that LLM display structural redundancy [43,31,10,49,36,25,14], particularly attention mechanisms, which can be selectively removed with minimal accuracy impact [3,19], often leaving models with extended sequences of consecutive FFN layers. We demonstrate that these FFN sequences exhibit surprisingly low inter-layer dependencies, enabling a novel transformation: multiple sequential FFN layers can be fused into a single, wider layer that enables simple parallel execution. This transformation is particularly powerful, eliminating synchronization points while allowing for more efficient hardware utilization without requiring architectural redesign (as visualized in Figure 2). Through analysis and empirical validation, we argue for the conditions under which this fusion preserves model behavior, showing that these conditions are commonly satisfied in practice, especially in larger models where the potential efficiency gains are most significant. Most surprisingly, our preliminary investigations suggest that even complete transformer blocks-containing both attention and FFN components-can sometimes be parallelized, pointing to potential new directions in neural architecture design (see Appendix B). Leveraging these insights, we develop Ultra-253B-Base, derived from Llama-405B through a combination of attention pruning and FFN Fusion. This model achieves remarkable efficiency gains while maintaining or exceeding its parent model's capabilities:
• A 1.71× speedup in inference latency and 35× lower per-token cost at batch size 32.
• State-of-the-art performance on key benchmarks, including 84.92% on Arena Hard, 86.58% on HumanEval, 87.54% on MMLU Instruct, 72.25% on MMLU-Pro, and 9.19 on MT-Bench.
• Improving memory footprint with a 2× reduction in kv-cache memory, and reduced parameter count from 405B to 253B.
Our key contributions are:
• The discovery that some Feed-Forward Networks in transformer architectures can often be parallelized with minimal accuracy loss, challenging the conventional wisdom that sequential processing is necessary for these layers.
• The creation of Ultra-253B-Base, a powerful 253B parameter model that matches or exceeds Llama-405B's capabilities while offering substantially improved efficiency, to be publicly released upon acceptance.
• Preliminary evidence that even complete transformer blocks-containing both attention and FFN components-can sometimes be parallelized, opening new possibilities for architectural innovations in large language models.
this section cite: ['b40', 'b28', 'b8', 'b46', 'b33', 'b22', 'b12', 'b2', 'b17']

Section: Preliminaries
Transformer-based LLMs. The structure of LLMs is typically based on the widely used transformer architecture [42]. It is generally composed of a series of sequential blocks, denoted as f 1 , . . . , f m , each containing an attention layer and an FFN layer. The attention layers capture the contextual relationships between input tokens, while the FFN layers apply transformations independently to each token. Given an input X ∈ R n×d , where n represent the sequence length and d the embedding dimension, a transformer block f : R n×d → R n×d is defined by the following equations:
g(X) = X + Attention(η 1 (X)) (1) f (X) = g(X) + FFN(η 2 (g(X))),(2)
where η k is a token-level normalization module, defined by the equation η k (x i ) = xi ∥xi∥ ⊙ s k with x i ∈ R d being the i-th token, s k ∈ R d being a learnable scale factor, and ⊙ is the element-wise product. The FFN layer used in the models we experiment with is the SwiGLU module [37], which is defined as follows:
SwiGLU(X) = (σ(XW T 2 ) ⊙ XW T 1 )W T 3 ,(3)
where σ is the SiLU activation [20] function, W 1 , W 2 ∈ R d h ×de and W 3 ∈ R de×d h . Here, d e represents the embedding dimension, and d h denotes the hidden dimension of the FFN. Other components of the LLM include an embedding function at the model's input, which maps tokens to vectorized embeddings, an additional normalization function, and a linear function at the output, which normalizes the final block outputs and projects them to the vocabulary dimension.
Puzzle. Puzzle [3] is a neural architecture search (NAS) framework that optimizes a trained LLM for inference efficiency. Starting with a pre-trained model, it prunes or reconfigures each transformer block-often reducing or removing attention layers-while preserving model quality through a distillation process. In particular, applications of Puzzle often show that many attention layers can be removed with minimal accuracy loss, thus leaving sequences of FFNs uninterrupted by attention. Additionally, certain FFN layers experience significant channel pruning, leading to a reduction in their hidden dimension. Step 1 in Figure 1 illustrates a hypothetical outcome of this process.
this section cite: ['b39', 'b34', 'b18', 'b2']

Section: FFN Fusion
Several optimization techniques reduce the number of attention layers to improve inference efficiency [3,19]. An attention-removed transformer block is defined by the equation
f (X) = X + FFN(η 2 (X)).(4)
Given a consecutive sequence of attention-removed blocks f i , . . . , f i+c we define the parallel version of this sequence f [i,i+c] as
f [i,i+c] (X) = X + c j=0 FFN i+j (η 2 (X)),(5)
where FFN i is the FFN layer of block i and η 2 is taken from the last layer in the sequence. While in the sequential form the input for every FFN depends on the output of the previous layers, in this form all the FFN layers share the same input, which makes their computation independent and enables execution in parallel across different GPUs. Another useful property of this formulation is that equation 5 is equivalent to equation 4 with a single wider FFN when the weights are concatenated.
Theorem 3.1. Let n ∈ N, and let FFN 1 , . . . , FFN n be a sequence of FFN functions, where the weights of FFN i are W i 1 , W i 2 , W i 3 . Then, the sum of these FFN functions (equation 5) is equivalent to a single FFN function FFN * , with the weight matrices given by: W * 1 = (W 1 1 ) T , . . . , (W n 1 ) T T W * 2 = (W 1 2 ) T , . . . , (W n 2 ) T T W * 3 = (W 1 3 ), . . . , (W n 3 ) , where [•, . . . , •] denotes the concatenation of matrices along the second axis and the dimensions of the matrices are W * 1 , W * 2
∈ R nd h ×de , W * 3 ∈ R de×nd h .
The theorem (and proof in Appendix A) are written for the simple case where d h is equal for all the FFNs. The extension for the case where the hidden dimension is different for each layer is straightforward. While written in the context of Equation 3, the theorem holds for other common FFN variants [37,39]. See a visualization in Figure 7.
Efficiency motivation and analysis. LLMs are ubiquitously designed in sequential blocks, with the block sizes and the number of blocks increasing as the models grow in size. For bigger models, parallelization techniques such as tensor parallel (TP) are utilized to split work across GPUs and reach acceptable inference latencies. However, this strategy does not result in linear latency improvements with the number of GPUs applied. The first reason, is the communication time required for the all-reduce that follows each TP block. The second reason is more subtle: GPUs are at their best for very large operations. As each atomic General Matrix Multiplication (GEMM) operation becomes smaller, low level overheads (GPU wave quantization) become more apparent and take a larger portion of the latency budget. Therefore, increasing the computation per GPU (by increasing the size of a block) while reducing the number of synchronizations needed (by using fewer blocks) is an effective strategy for low latencies in a TP setting (see Figure 2). Drawing from this motivation, we attempt to fuse many sequential blocks/FFNs in LLMs into fewer, larger blocks/FFNs. Each single reduction in the depth of the computational graph removes one unit of time spent on synchronization, and the bigger fused blocks also enable operating at higher TP numbers with enough computation assigned to each GPU. Figure 6 shows the effects described in this analysis using measurements of different model architectures in the same TP setting.
GPU 0 GPU 1 GPU 2 GPU 3 GPU 0 GPU 1 GPU 2 GPU 3 GPU 0 GPU 1 GPU 2 GPU 3 Sync Sync Sync FFN1 Fusion(FFN1,FFN2) FFN2 Smaller block, lower utilization More sync operations Larger block, higher utilization Less sync operations 0 20 40 60 80 100 120 120 100 80 60 40 20 0 2
-9 2 -8 2 -7 2 -6 2 -5 2 -4 2 -3 2 -2 2 -1 Cosine Dist Measured Block Dropped Block
Figure 3: Block-wise dependency heatmap of Ultra-253B-Base before applying FFN Fusion (logscale). Each coordinate (i, j) encodes how much block j depends on block i, measured by the cosine distance between h j (X) and hj i (X).
this section cite: ['b2', 'b17', 'b34', 'b36']

Section: Pairwise block dependency.
It is reasonable to hypothesize that not every transformer block in a sequential model is equally dependent on all its predecessors. In particular, a given block may depend only on a subset of the blocks that precede it. To investigate this, we perform a pairwise dependency analysis between blocks. Let us define h(X) = f (X) -X as the contribution of f to X, and similarly, h j as the contribution of f j . We define hj i as the contribution of block j when block i is removed from the model. We construct a dependency matrix M ∈ R m×m by computing the following cosine distance:
M ij = CosineDist(h j (X), hj i (X)).
That is, M ij quantifies the dependency of block j on block i. Thus, a small cosine distance indicates that dropping block i has little effect on block j, suggesting relative independence-a characteristic that can be exploited for increasing parallel computation. Conversely, a large cosine distance corresponds to a strong dependency, implying that sequential processing is more critical to maintain performance. We illustrate this approach in Figure 3 by constructing a dependency matrix, M , for Ultra-253B-Base (prior to FFN Fusion; see Section 4). This matrix visually encodes the interdependencies among the model's layers: darker blue hues indicate weaker dependencies-signaling promising opportunities for FFN Fusion-while darker red hues denote strong dependencies that would hinder parallelization or fusion. For example, Figure 3 shows that all layers in the model depend on layers 0 and 5, causing their entire rows to be colored in dark red hues. The dark blue region, marked with a dashed square, shows a sequence of FFNs with low interdependency were selected for FFN Fusion. Due to GPU memory constraints, we are unable to fuse them all at once, so we divide them into fusion sequences based on the maximum size that fits within our devices. Additionally, we applied this metric to explore block parallelization (Apendix B), aiming to parallelize general LLM blocks rather than just FFNs. We constructed the dataset that was used for this evaluation following Distillation Mix [3].
this section cite: ['b2']

Section: Producing Large-Scale Models with FFN Fusion
In this section, we describe how FFN Fusion is applied at large scale to transform a 405B-parameter Llama-based model into our more compact Ultra-253B-Base model. Specifically, we show how identifying and fusing long sequential feed-forward blocks reduces depth without sacrificing accuracy. We utilize a lightweight refinement KD and alignment phase to ensure the fused model retains or even improves upon the performance of its larger predecessor. Finally, we present a comprehensive evaluation of the resulting model, demonstrating significant speedups and strong results on standard benchmarks. We first run the standard Puzzle search on Llama-405B, specifying that the derivative model must obtain a 1.5× latency speedup and fit within a single NVIDIA 8×H100 node (640 GB total), and in a single B100 GPU (192 GB) . This yields a 253B-parameter model whose overall configuration is shown in Appendix D. Many attention layers were removed, resulting in 50 blocks contiguously arranged without interleaved attention layers.
this section cite: []

Section: Ultra-253B-Base pre-CPT Ultra-253B-Base post-CPT Llama 3.3 70B Instruct Llama 3.1 405B Instruct
score_macro) MMLU 87.26 88.09 81.4 88.6 (micro.value) MATH500 68.6 80.4 73.6 69.6 (pass@1) HUMANEVAL 85.36 88.41 84.1 85.97 RULER 128K 58.57 83.21 52.25 73.73 notes mmlu uses 5-shot math instead of math500 we measured humaneval 81.7 RC41-squashed_ffn-diverse_mix iter_0012956 pl8z07xa iter 1800 RC41C-CPT-phase4-2588K-from8K_iter17600-lr1e-7-rope16 FFN Fusion. Next, we apply FFN Fusion ( §3) to 49 of the 50 consecutive FFN layers (See Section 5.3 for ablation on leaving the last layer). Due to per-GPU memory limits, we split the layers into four sequences [66, 73], [74, 85], [86, 100], [101, 114], each fused into a single FFN. Before fusion, the baseline model achieved 84.23 on MMLU and 8.83 on MT-Bench. Remarkably, after fusing all 49 layers-layers that, if removed entirely, would damage performance severely-the model still maintains similar MMLU and MT-Bench scores of 82.76 and 8.35 before any additional training.
Table 1: Comparison of Ultra-253B-Base and its parent Llama-405B after applying FFN Fusion, KD, and alignment.
this section cite: []

Section: Metric Llama-405B Ultra-253B-Base
MMLU Instruct [32] 86.53 87.54 MMLU-Pro [44] 71.92 72.25 Arena Hard [29] 72.56 84.92 HumanEval [4] 85.97 86.58 MT-Bench [50] 9.06 9.19
this section cite: ['b29', 'b41', 'b26', 'b3', 'b47']

Section: Additional Training.
To recover performance, we used KD as described in [3]. This involved a multistage distillation process: 54B tokens at 8k context, followed by 5B tokens each at 16k and 32k, and finally 0.8B tokens at 128k. The KD process improved MMLU and MT-bench scores to 85.17 and 9.10, respectively. Further optimization was explored through two methods. First, inexpensive alignment via RLHF [45]. Table 1 demonstrated that Ultra-253B-Base surpassed Llama-405B's capabilities, particularly on the Arena Hard benchmark. We applied only a longer continual pretraining (CPT), without alignment, following the initial KD. This involved 73B tokens at a context length of 8k and another 15B tokens at a context length of 258k, and also yielded strong performance, even before instruction-based tuning (Figure 4). Even after removing nearly half of the model's attention layers, we still improved its long-context performance, outperforming Llama-405B on RULER-128K [22].
Table 2: User latency under Tensor Parallel (TP) 8 on a single H100 node. A higher tokens/second value indicates lower latency for a single user. Model Tokens/Second Llama-405B 41.44 253B PreFusion 63.38 (1.53×) Ultra-253B-Base 70.92 (1.71×) Llama-3.3-70B
this section cite: ['b2', 'b42', 'b19']

Section: 94.03
Efficiency Improvements. Ultra-253B-Base achieves a 1.71× speedup in user latency (Table 2), a 35× lower per-token cost at batch size 32, and reduced memory footprint with half the attention layers and 253B parameters (down from 405B). Ultra-253B-Base breaks the efficient frontier on a single H100 node, offering a state-of-the-art accuracy and latency under the similar hardware constraints (see Figure 5). Table 2 details the user latency (tokens/second) achieved by Llama-405B, by 253B model (with and without FFN Fusion), and by Llama-3.3-70B, all under identical tensor parallel settings on a single 8×H100 node. Notably, Ultra-253B-Base is 1.71× faster than the parent for single-user decoding. On NVIDIA H200, its rate increases to 90.05 tokens/second. With speculative decoding [27]-using Llama-3.2-1B-Instruct as a draft model with no extra training-Ultra-253B-Base reaches 202 tokens/s on H200. This speculative decoding latency is averaged over all MT-Bench completions. Notably, as TP size increases, the efficiency of the fused FFN layers will increase further, and thus ready to benefit from improving hardware designs, such as GB200 NVL72 nodes. Overall, Ultra-253B-Base demonstrates that FFN Fusion coupled with the Puzzle algorithm can greatly reduce a model's depth at large scale.
this section cite: ['b24']

Section: Additional Empirical Studies
In this section, we present empirical study to validate our approach. In Section 5.1 we present results for FFN Fusion on a 70B model. Next, in Section 5.2, we examine the consequences of removing FFN layers rather than fusing them, demonstrating that these layers are vital to preserving model quality. In Section 5.3, we investigate the sensitivity of certain FFNs to fusion. Section 5.4 demonstrates that the FFN Fusion phenomenon is applicable to a variety of models, spanning different scales and families. Finally, in Section 5.5 we provide a hypothesis as to why FFN Fusion works. We consider a derivative of Llama-3.1-70B-Instruct created using the Puzzle algorithm [3], which reduces the model to 49B parameters while matching the original accuracy. The Puzzle process prunes attention in specific layers, leaving two main sequences of consecutive FFN layers: layers 42-51 (10 layers) and layers 53-70 (18 layers). We evaluate FFN Fusion at four progressively increasing levels of intensity, each reducing the number of FFN layers more than the previous: • Step 3: Fuse the entire first sequence (layers 42-50) into a single FFN, and split the second sequence (53-69) into two fused blocks.
this section cite: ['b2']

Section: FFN Fusion in 70B scale
•
• Step 4: Fuse the entire second sequence (53-69) as well, reducing each main FFN run to a single layer.
Similarly to the Ultra-253B-Base, we chose to exclude the last FFN in each sequence. We perform no additional training when applying these fusions.
Table 3 reports each variant's performance on MMLU and MT-Bench, as well as a combined score Accuracy = (MMLU + 10 × MT-Bench)/2. We observe that step-1 fusion reduces depth with only a 1% overall accuracy drop, while step-4 fusion (collapsing each sequence to a single layer) sees roughly a 4% drop. Step 1 Step 3 15 FFNs removed 20 FFNs removed 5.3 5.4 5.5 5.6 5.7 5.8 5.9 78 79 80 81 82 83 84 85 FFN Fusion Removing FFNs Baseline Step 3 + KD Latency (ms) Accuracy (%)
this section cite: []

Section: Removing FFNs vs. FFN Fusion
An intuitive alternative to fusing FFNs is to remove the same FFN layers outright. However, as shown below, large-scale removal generally leads to significant accuracy degradation, whereas fusion preserves representational capacity by retaining all parameters in a single, parallel module. To determine which FFNs to drop, we rely on Puzzle's block-importance scores and remove the least important FFNs first. We restrict removal only to the FFN-fusion regions. For the 49B model, we observe a clear advantage for fusion over removal. Figure 6 compares the accuracy-latency trade-off for two removal levels (15 or 20 FFNs removed) versus the two fusion steps. As we gradually remove more FFNs, we decrease latency but see larger accuracy drops. By contrast, fusing step 3 yields a 10% latency improvement with only a 3.5% accuracy drop. Removing 15 or 20 FFNs yields comparable or slightly higher latency gains (7.5% or 11%), but at steeper accuracy declines (5% or 8.7%). Moreover, a brief knowledge-distillation phase (25B tokens) on the step-3 fused model actually surpasses the baseline (MMLU: 80.56, MT-Bench: 9.00). With the 49B model, we observe that fusing the final FFN in a long sequence often degrades accuracy more than fusing earlier FFNs. Below, we summarize an ablation study to highlight this phenomenon.
this section cite: []

Section: The Final FFN in Each Sequence is Sensitive to Fusion
Table 4 examines various ways of fusing the two main FFN sequences (layers 42-51 and 53-70) in the 49B model. The first part of the table compares fusing the entire second sequence ([53, 69] vs.
[54, 70]) and the entire first sequence ( [42,50] vs. [43,51]) in isolation, indicating that layer 70 is especially sensitive.
In the second part, we fuse both sequences simultaneously. When either layer 51 or layer 70 is included in these large fused blocks, we see additional performance drops. Notably, skipping the final FFN (e.g. fusing [42,50] , [53, 69]) yields higher MT-Bench scores than including layer 51 or 70. The final FFN in each attention-removed sequence appears uniquely important to the model's representations. While most layers can be safely fused, incorporating the last FFN often triggers a significant accuracy drop. Consequently, omitting this final FFN from the fused groups is typically a more reliable choice for efficient fusion with minimal performance loss. This section explores FFN Fusion on additional models, evaluating its effectiveness across different scales and model families. We applied FFN Fusion to Llama-3.1-8B-Instruct [13] and Mistral Large 2 following a similar procedure as with our 253B and 49B models. Specifically, we used the Puzzle algorithm on both models to remove a portion of their attention layers. Following this, we fused consecutive sequences of FFNs. For Mistral Large 2, we removed 27 attention layers that resulted in 2 sequences of 14 and 13 consecutive FFN layers. We fused each sequence (excluding the last FFN) to a single FFN layer. For Llama-3.1-8B-Instruct (32 layers), we removed 8 consecutive attention layers. We fused the sequence (excluding the last FFN) to a single FFN layer. We also conducted a short KD training on the 8B model over 20B tokens and compared the results to the original model. Table 5 provides a clear demonstration of FFN Fusion's generalizability, showing its effectiveness across additional models.
this section cite: ['b39', 'b47', 'b40', 'b39', 'b47', 'b11']

Section: Additional Models

this section cite: []

Section: Fusion Explainability
Table 6: Reverse order vs Fusion experiment on Puzzle-49B.
this section cite: []

Section: Sequence Fusion Accuracy
Reverse Accuracy [42,49] 83.74 83.52 [53,60] 83.24 83.25 [63,67] 84.22 84. 10 [63, 70] 84.00 83.88
In this section, we aim to further explain the phenomenon of FFN Fusion. We examine the functional structure of LLMs that allows for the fusing of layers, specifically focusing on the consecutive FFN layers after attention removal. First, we rewrite the token-level normalization equation in an alternative form:
η k (x i ) = D s k x i /∥x i ∥,
where we replace the normalization parameters s k ∈ R d with a diagonal matrix D s k ∈ R d×d , with s k on the diagonal. This reformulation allows us to "push" the matrix into its corresponding module-either attention or FFN-and consider equations 1 and 4 as functions of the token direction -→ x i = x i /∥x i ∥, while ignoring the magnitude. Figure 11 (a and b), Similar to the measurements in [30], shows the cosine distance between X and f (X) for each layer of both models. The plot reveals that the FFN fusing areas-[42, 51] , [53, 70] for the 49B model and [66, 115] for the 253B model-exhibit lower cosine distance compared to other regions in the model. Assuming that small changes in the input to an FFN layer lead to small changes in its output, we can conclude that fusing the FFNs with low cosine distance (except for the last one) will not cause significant changes to the model. Specifically, by fusing the FFN layers, we are altering the input for each layer, but the directional difference between the original input (according to the original model's structure) and the new input (resulting from fusion) remains small, and the outputs should remain similar under the smoothness assumption. Another experiment (Table 6) that strengthens this claim shows that even if we reverse the FFNs order instead of fusing them, we get similar performances. The results in Figure 11 (c and d) present the relationship between the layer input X and h(X) = f (X) -X. The small change in token directions may be linked to the low cosine distance between h(X) and X, but they are nearly orthogonal (with a distance of around 0.95 for both models across all layers). However, a more suitable explanation comes from the ratio ∥h(X)∥ / ∥X∥, which is smaller in the fused regions. This suggests that h(X) is small compared to X that it does not significantly affect the direction of X. Although this section can also serves as motivation for FFN removal, we demonstrated in Figure 6 that removing these layers significantly harms the model's performance, to a much greater extent than fusion does. Both metrics were evaluated using the Distillation Mix dataset.
this section cite: ['b39', 'b46', 'b27']

Section: Concluding Remarks
This work introduces FFN Fusion, a novel optimization technique that demonstrates how sequential computation in large language models can be effectively parallelized. Through comprehensive experiments at scales from 49B to 253B parameters, we showed that sequences of FFN layers can be fused with minimal impact on model capabilities. Our work revealed couple of key insights: (1) FFN layers in attention-removed regions exhibit remarkably low inter-layer dependencies, suggesting these computations may be more independent than the sequential architecture implies; and (2) empirically, final FFN layers in parallelizable sequences show higher sensitivity to fusion-an observation that points to potentially meaningful structural patterns in how these models process information.
The practical impact of our work is demonstrated through Ultra-253B-Base, which achieves a 1.71× speedup in inference latency compared to its parent Llama-405B model, while requiring 35× lower per-token cost at batch size 32. Most remarkably, these efficiency gains come with minimal degradation in model capability-in fact, Ultra-253B-Base matches or exceeds its parent's performance across key benchmarks despite using only 253B parameters compared to the original 405B.
Our findings open several promising research directions. First, the clear patterns in inter-layer dependencies we observed could provide a new lens for model interpretability research. The ability to identify which FFN layers operate independently versus those that require sequential processing may offer insights into how these models structure and transform their internal representations. Second, our preliminary finding that even full transformer blocks can sometimes be parallelized suggests possibilities for new architectural designs explicitly optimized for parallel execution. Third, future work could extend FFN Fusion to models incorporating Mixture-of-Experts (MoE) layers. Finding a way to fuse MoE layers while maintaining sparse activation patterns and efficient expert routing could lead to efficiency improvements. Finally, the strong performance of FFN Fusion at larger scales raises intriguing questions about the relationship between model size and natural parallelization.
this section cite: []

Section: Societal Impact
The inference efficiency achieved through FFN Fusion, as demonstrated by its impact on operational costs in Section 4, significantly improves the accessibility of large language models. By reducing these costs, a wider array of users and organizations can now better afford to deploy and leverage these powerful technologies, effectively dismantling economic barriers.
𝐹𝐹𝑁 1 𝑋 + 𝐹𝐹𝑁 2 𝑋 = 𝐹𝐹𝑁 * 𝑋 = 𝜎 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 ⊙ 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑊 1 * 𝑇 𝑊 2 * 𝑇 𝑊 3 * 𝑇 𝑋 𝑋 𝑑 𝑒 𝑑 ℎ 𝑊 1 2𝑇 𝑑 𝑒 𝑑 ℎ 𝑊 3 2𝑇 𝐹𝐹𝑁 1 𝑋 = 𝜎 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 ⊙ 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑊 1 1𝑇 𝑊 2 1𝑇 𝑊 3 1𝑇 𝑋 𝑋 𝑑 𝑒 𝑑 ℎ 𝑊 2 2𝑇 𝑊 1 * 𝑇 𝑑 𝑒 2𝑑 ℎ 𝑊 1 1𝑇 𝑊 1 2𝑇 𝑊 2 * 𝑇 𝑑 𝑒 2𝑑 ℎ 𝑊 2 1𝑇 𝑊 2 2𝑇 2𝑑 ℎ 𝑊 3 1𝑇 𝑊 3 2𝑇 𝑊 3 * 𝑇 𝑑 𝑒 𝑑 𝑒 𝑑 ℎ 𝑊 1 1𝑇 𝑑 𝑒 𝑑 ℎ 𝑊 3 1𝑇 𝑑 𝑒 𝑑 ℎ 𝑊 2 1𝑇 𝐹𝐹𝑁 2 𝑋 = 𝜎 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 ⊙ 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑊 1 2𝑇 𝑊 2 2𝑇 𝑊 3 2𝑇 𝑋 𝑋 𝑇𝑤𝑜 𝑆𝑤𝑖𝐺𝐿𝑈 𝑙𝑎𝑦𝑒𝑟𝑠: 𝑆𝑖𝑛𝑔𝑙𝑒 𝐹𝑢𝑠𝑒𝑑 𝑆𝑤𝑖𝐺𝐿𝑈:
this section cite: []

Section: Method.
We first compute the block-wise dependency matrix exactly as in Section 3, We then search for candidate sequences of 4 consecutive blocks to fuse. The choice of this number of blocks is mostly related to hardware constraints. Arbitrary sizes will be considered in our future work. For each such sequence [i, i + 3], i ∈ [m -4], we extract the corresponding M [i,i+3] ∈ R 4×4 submatrix from the block dependency matrix and compute two statistics:
M [i,i+3] max = max k,j∈[4] M [i,i+3] kj , M [i,i+3] sum = k,j∈[4] M [i,i+3] kj
A lower M max , and M sum indicates weaker dependencies among those blocks, suggesting they may be more amenable to parallelization. We exclude any blocks that lack attention (shaded regions in Figure 8b) because our focus here is on full Transformer blocks that contain both attention and FFN modules. We employ a simple greedy algorithm to choose the best subsequences:
1. Choose the length-4 sequence starting index i * = arg min i M [i,i+3] max .
2. If the argmin is not unique, break the tie using i * = arg min i M [i,i+3] sum (restricted to the set from step 1).
this section cite: []

Section: Parallelize the blocks according to the chose sequence
[i * , i * + 3].
this section cite: []

Section: Remove any overlapping sequences.
5. Repeat until no valid sequences remain.
The motivation behind this algorithm is to avoid choosing sequences with high dependency pairs.
The sum tie breaking is due to step-like characteristic of M max statistic (see Figure 8b).
Table 7 reports the downstream performance (MMLU and MT-Bench) after fusing the sequences chosen by 4 algorithm steps. Parallel Sequences [38,41] 80.51 8.67 [38,41] , [71,74] 80.04 8.67 [38,41] , [71, 74] , [32,35] 77.86 8.20 [38,41] , [71, 74] , [32,35] , [26,29] 56.12 6.62
Results. From Table 7, we observe that parallelizing the first and second sequence of four blocks leads to only a modest drop in MMLU, but once the third sequence is added, the decline becomes much more pronounced. This suggests that full block parallelization is more challenging than FFN Fusion. These observations are supported by the block-wise dependency heatmap and the submatrix statistics (Figure 8), both of which indicate stronger interblock dependencies and higher M max and M sum values in full block sequences compared to attention removed sequences. In addition, Figure 11 shows that the full transformer blocks alter significantly the tokens directions.
this section cite: ['b35', 'b38', 'b35', 'b38', 'b35', 'b38', 'b29', 'b32', 'b35', 'b38', 'b29', 'b32', 'b23', 'b26']

Section: C Datasets
For applying Puzzle throughout our experiments, we used the same dataset mixture used in [3], termed Distillation Mix. This mixture includes source code repositories, Wikipedia articles, books, news websites, and several other domains. The dataset comprises 224 billion tokens collected from three public datasets: FineWeb [34], Dolma [40], and Buzz-V1.2 [21]. For Ultra-253B-Base KD training we used the same data reinforced with synthetic data generated with Llama-405B, following the [48] approach, to help further align Ultra-253B-Base with its parent model.
this section cite: ['b2', 'b31', 'b37', 'b45']

Section: D Ultra-253B-Base and 49B Architecture Overview
In this section, we detail the configuration of each block in the Ultra-253B-Base model and highlight several unique design choices introduced by the Puzzle framework on top of Llama-405B. Notably, the model employs variable FFN widths, with scaling multipliers ranging from 0.4875 up to 4.875 across its 126 blocks. This variability allows the model to dynamically adjust capacity at different depths, balancing performance with computational efficiency. Furthermore, it includes a sequence of 50 consecutive layers that bypass the attention mechanism (denoted by no_op), dedicating these layers entirely to FFN processing and creating an ideal region for FFN Fusion.
We observe a similar pruning pattern in the 49B model, which is derived from the Llama-3.1 -70B model using the Puzzle framework. Like its larger counterpart, the 49B model also features variable FFN multipliers and blocks where attention is removed, although at a smaller scale. Figure 9 compares both architectures.
this section cite: []

Section: E MoE Inference Performance
MoEs show great promise in reducing inference costs and are currently spearheaded by DeepSeek-V3 [6], a highly sparse MoE with a great performance. However, MoEs have their own problems in inference.
this section cite: []

Section: E.1 Problems with Small Blocks
An underlying problem for MoEs paradoxically arises from the small size of their sub-modules. Smaller layers incur large latency than expected for two reasons:
• GPU utilization -low level GPU overheads such as wave quantization produce latency problems that become stronger as the module becomes smaller.
• Communication overhead -The All-Reduce operation that is typical for tensor parallelism creates an overhead for communication. For small modules this overhead increases in proportion.
MoE's experts are smaller than dense MLPs and hence suffer more strongly from the above effects.
In addition the routing mechanism of MoEs is another small module that suffers accordingly. As a result, dense models scale better with the number of GPUs while MoEs reach an early saturation.
It is worth to notice that this is in complete contrast to our FFN Fusing method that creates larger modules that parallelize better, as visualized in Figure 2.
this section cite: []

Section: E.2 Batch Size Effects
MoEs behave very well for very large batch sizes. For such large workloads load balancing issues are less dominant and low level overheads become negligible. However, smaller batch sizes are not as efficient. Concretely, we define an efficient size of a batch through the perspective of analyzing a linear layer. The performance of linear layers varies significantly with batch size. For small batches, performance is limited by the time needed to read weights from DRAM (I/O bottleneck). For large batches, performance becomes compute-bound, limited by the hardware's maximal FLOPs/sec. Notably, the latency of a linear layer remains the same for batch sizes 1 and 64, as both are dominated by weight loading time. This makes small batches extremely inefficient. While other factors exist, we consider a linear layer to operate efficiently once the number of tokens in the batch approaches 64, and inefficient otherwise. MoEs require all experts to be fully utilized for high throughput. in practice, the total batch is split across experts, significantly reducing the effective number of tokens each expert sees. For example, with a sparsity factor of 1/32 (as in [5,6]), a batch of 2048 tokens (64 × 32) is needed to ensure each expert processes around 64 tokens. The case is more extreme in Llama 4 Maverick, where a 1/128 sparsity factor requires a batch of 8192 tokens (64 × 128) to reach this threshold. These batch sizes are often impractical-especially in generation mode-leaving MoEs to operate in a low-efficiency regime, where linear layers are under-utilized. We can thus conclude that MoEs only really live up to their promise for very large batch sizes. For more commonly used intermediate batch sizes they suffer from bad scaling with the number of tokens, larger low level overheads, and worse parallelization scaling than dense models.  • Overall Color Scale. Each entry (i, j) in the heatmap corresponds to the cosine distance between the outputs of block j when block i is dropped versus when the model is intact. A dark red hue signifies a strong dependency, meaning that omitting block i substantially alters block j. Conversely, dark blue indicates a weak dependency, suggesting that dropping block i does not notably affect block j.
this section cite: ['b4']

Section: F Further Pairwise Block Dependency Analysis
• Attention-Removed Region (Dotted Box). The dotted box highlights a region of consistently low dependency values, where most blocks exhibit minimal mutual influence. This area arises from attention pruning in those layers and is particularly well-suited for FFN Fusion, since layers here can be more easily parallelized without significantly harming accuracy.
• Crucial Early Blocks. Some blocks in the upper rows of the matrix (toward the top of the y-axis) display consistently red cells across many columns. This indicates that certain early layers have a strong influence on a wide range of subsequent blocks, making them less amenable to parallelization.
• Sub-region with Sequential Reliance. A noticeable diagonal band of higher values appears in some sub-regions, implying that each block in that band strongly depends on its immediate predecessor. Such sequential reliance reduces the potential for parallelization in those areas, since omitting any single block significantly affects the next one.
• Global Sinks at Later Layers. As we move toward deeper layers (near the bottom-right corner), some blocks again show stronger dependencies, acting as "global sinks" that consolidate information from many earlier blocks. Although the importance of certain blocks may wane in mid-network regions, it can resurface in the final layers.
In summary, the dependency matrix reveals both strongly and weakly coupled sub-regions. Highdependency zones require careful consideration to avoid accuracy loss if parallelization is attempted. Conversely, low-dependency zones, such as the attention-removed region, offer a promising avenue for efficient FFN Fusion. NeurIPS Paper Checklist 1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The claim introduced in the abstract and introduction are supported by theoretical claims (Theorem 3.1) and empirical evidance (Sections 4 & 5)
this section cite: []

Section: G Fusion Explainability
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Justification: The steps that were taken to create the results are fully described in our method & results sections.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: We will release code upon acceptance.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
this section cite: []

Section: A Proof of Theorem 3.1
Proof. The proof will proceed by induction. For the base case, we consider two FFN layers, FFN 1  and FFN 2 . We will demonstrate that FFN * = FFN 1 + FFN 2 . The induction step follows naturally from the associative property of addition.
(FFN 1 (X) + FFN 2 (X)) =(σ(X(W 1 2 ) T ) ⊙ X(W 1 1 ) T )(W 1 3 ) T + (σ(X(W 2 2 ) T ) ⊙ X(W 2 1 ) T )(W 2 3 ) T = σ(X(W 1 2 ) T ) ⊙ X(W 1 1 ) T , σ(X(W 2 2 ) T ) ⊙ X(W 2 1 ) T (W * 3 ) T =( σ(X(W 1 2 ) T ), σ(X(W 2 2 ) T ) ⊙ X(W 1 1 ) T , X(W 2 1 ) T )(W * 3 ) T =(σ(X(W * 2 ) T ) ⊙ X(W * 1 ) T )(W * 3 ) T =FFN * (X)
where [•, •] is the concatenation of matrices along the second dimension.
this section cite: []

Section: B Block Parallelization
In all previous experiments, we focused on fusing sequences of FFNs in attention-removed layers (see Section 3). In that setting, the homogeneous structure of FFN-only layers permits straightforward weight concatenation and fusion. However, when considering full block parallelization-where entire Transformer blocks, each comprising both an attention module and an FFN, are run in parallel-such fusion is not possible. This is because each full block consists of two distinct components, with the attention output explicitly directed to its paired FFN. Running entire blocks completely in parallel is not currently natively supported by heavily optimized inference frameworks such as TensorRT-LLM or vLLM [24]. Nevertheless, one can envision assigning each full block to a different GPU, thereby maximizing parallelism and potentially achieving significant speedups. We use our block-wise dependency analysis to examine the dependency structure between blocks and identify candidate groups whose outputs are relatively independent. Such groups are ideal for parallelization with minimal accuracy loss and could enable higher degrees of tensor parallelism-each full block operating on its own set of GPUs-thereby improving inference throughput. In our experiments, we implement these full block parallelization strategies using a more flexible environment (e.g., HuggingFace Transformers [46]), albeit with reduced optimization for large-scale deployments. A thorough investigation of the actual speedups in specialized frameworks is left for future work.
Answer: [Yes] Justification: The authors indicate that the suggested method is mainly relevant in cases where there is long a sequence of consecutive FFN blocks.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: ['b21', 'b43']

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: Theorem 3.1 proves the equivalence between a parallel computation of FFN blocks and our consolidate formulation.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental setting/details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: The paper describes all the details regarding the main method (FFN Fusion).
Training methods were based on previous work. Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: Error bars are irrelevant for the experiments in the paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification:
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification:
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification:
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: [NA] Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification:
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
this section cite: []

Section: New assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] Justification: [NA] Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: [NA] Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: [NA] Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: [NA] Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: The impact of large language models on scientific discovery: a preliminary study using gpt-4 Year: (2023)
Ref_id:b1 Title:  Year: (2023)
Ref_id:b2 Title:  Year: (2024)
Ref_id:b3 Title:  Year: ()
Ref_id:b4 Title:  Year: ()
Ref_id:b5 Title: 8-bit matrix multiplication for transformers at scale. Advances in neural information processing systems Year: (2022)
Ref_id:b6 Title: The case for 4-bit precision: k-bit inference scaling laws Year: (2023)
Ref_id:b7 Title:  Year: ()
Ref_id:b8 Title: Reducing transformer depth on demand with structured dropout Year: (2019)
Ref_id:b9 Title: The lottery ticket hypothesis: Finding sparse, trainable neural networks Year: (2018)
Ref_id:b10 Title: Gptq: Accurate post-training quantization for generative pre-trained transformers Year: (2022)
Ref_id:b11 Title:  Year: (2024)
Ref_id:b12 Title:  Year: (2024)
Ref_id:b13 Title: rstar-math: Small llms can master math reasoning with self-evolved deep thinking Year: (2025)
Ref_id:b14 Title: Intelligent virtual assistants with llm-based process automation Year: (2023)
Ref_id:b15 Title: Learning both weights and connections for efficient neural network Year: (2015)
Ref_id:b16 Title: Second order derivatives for network pruning: Optimal brain surgeon Year: (1992)
Ref_id:b17 Title: What matters in transformers? not all attention is needed Year: (2024)
Ref_id:b18 Title: Bridging nonlinearities and stochastic regularizers with gaussian error linear units Year: (2016)
Ref_id:b19 Title: Ruler: What's the real context size of your long-context language models Year: (2024)
Ref_id:b20 Title: Mixtral of experts Year: (2024)
Ref_id:b21 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b22 Title: The remarkable robustness of llms: Stages of inference? Year: (2024)
Ref_id:b23 Title: Optimal brain damage Year: (1989)
Ref_id:b24 Title: Fast inference from transformers via speculative decoding Year: (2023-07-29)
Ref_id:b25 Title: Pruning filters for efficient convnets Year: (2016)
Ref_id:b26 Title: From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline Year: (2024)
Ref_id:b27 Title: Deja vu: Contextual sparsity for efficient llms at inference time Year: (2023)
Ref_id:b28 Title: Are sixteen heads really better than one? Advances in neural information processing systems Year: (2019)
Ref_id:b29 Title:  Year: (2024)
Ref_id:b30 Title: GPT-4 technical report Year: (2023)
Ref_id:b31 Title: The fineweb datasets: Decanting the web for the finest text data at scale Year: (2024)
Ref_id:b32 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b33 Title: On the effect of dropping layers of pre-trained transformer models Year: (2023)
Ref_id:b34 Title: Glu variants improve transformer Year: (2020)
Ref_id:b35 Title: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer Year: (2017-04-24)
Ref_id:b36 Title: Primer: Searching for efficient transformers for language modeling Year: (2022)
Ref_id:b37 Title: Dolma: an open corpus of three trillion tokens for language model pretraining research Year: (2024)
Ref_id:b38 Title: Galactica: A large language model for science Year: (2022)
Ref_id:b39 Title:  Year: (2023)
Ref_id:b40 Title: Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned Year: (2019)
Ref_id:b41 Title: Mmlu-pro: A more robust and challenging multi-task language understanding benchmark Year: (2024)
Ref_id:b42 Title: Helpsteer2-preference: Complementing ratings with preferences Year: (2024)
Ref_id:b43 Title:  Year: (2020)
Ref_id:b44 Title: Smoothquant: Accurate and efficient post-training quantization for large language models Year: (2023)
Ref_id:b45 Title: Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing Year: (2024)
Ref_id:b46 Title: Accelerating training of transformer-based language models with progressive layer dropping Year: (2020)
Ref_id:b47 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023-12-10)
