Title: FLASHVID: EFFICIENT VIDEO LARGE LANGUAGE MODELS VIA TRAINING-FREE TREE-BASED SPA-TIOTEMPORAL TOKEN MERGING
Abstract: Although Video Large Language Models (VLLMs) have shown remarkable capabilities in video understanding, they are required to process high volumes of visual tokens, causing significant computational inefficiency. Existing VLLMs acceleration frameworks usually compress spatial and temporal redundancy independently, which overlooks the spatiotemporal relationships, thereby leading to suboptimal spatiotemporal compression. The highly correlated visual features are likely to change in spatial position, scale, orientation, and other attributes over time due to the dynamic nature of video. Building on this insight, we introduce FlashVID, a training-free inference acceleration framework for VLLMs. Specifically, FlashVID utilizes Attention and Diversity-based Token Selection (ADTS) to select the most representative tokens for basic video representation, then applies Tree-based Spatiotemporal Token Merging (TSTM) for fine-grained spatiotemporal redundancy elimination. Extensive experiments conducted on three representative VLLMs across five video understanding benchmarks demonstrate the effectiveness and generalization of our method. Notably, by retaining only 10% of visual tokens, FlashVID preserves 99.1% of the performance of LLaVA-OneVision. Consequently, FlashVID can serve as a training-free and plug-and-play module for extending long video frames, which enables a 10× increase in video frame input to Qwen2.5-VL, resulting in a relative improvement of 8.6% within the same computational budget. Code is available at https://github.com/Fanziyang-v/FlashVID.

Section: INTRODUCTION
Recent advances in Video Large Language Models (VLLMs) (Li et al., 2025a;Zhang et al., 2024;Bai et al., 2025b;Comanici et al., 2025) have demonstrated promising capabilities in video understanding tasks. However, processing large numbers of visual tokens incurs substantial computational and memory overhead, as the attention mechanism scales quadratically with sequence length, limiting practical deployment. To address this challenge, visual token compression (Chen et al., 2024;Yang et al., 2025c;Zhang et al., 2025e) has emerged as a promising approach, leveraging the inherent redundancy in visual inputs to reduce sequence length by removing or merging less informative tokens, thereby enabling efficient inference without significant performance degradation.
While advances have been achieved in visual token compression for images (Bolya et al., 2023;Chen et al., 2024;Yang et al., 2025c;Zhang et al., 2025e), extending these methods to video remains largely underexplored. Videos inherently exhibit both spatial redundancy within frames and temporal redundancy across frames, rendering frame-wise compression strategies suboptimal due to their neglect of temporal dynamics and correlations. This gap highlights the need for compression techniques specifically designed for the spatiotemporal structure of video inputs in VLLMs.
Motivation. Recent VLLM acceleration methods (Huang et al., 2025;Shen et al., 2025;Shao et al., 2025a) typically adopt a three-stage pipeline: (1) video partition, grouping consecutive frames with similar semantics to avoid information mixing;
(2) frame-wise token selection, identifying informative tokens-often guided by [CLS] attention-for basic video representation; and (3) spatiotemporal compression, further reducing redundancy at the segment level.
However, previous methods typically compress temporal and spatial redundancy independently. Such decoupled strategies overlook the intrinsic spatiotemporal relationships in videos. Moreover, temporal redundancy is commonly defined as the consistency of visual features at fixed spatial locations across consecutive frames. Due to the dynamic nature of video, the most semantically similar visual elements are likely to experience changes in spatial position, scale, orientation, and other attributes over time. Consequently, the most correlated visual features in adjacent frames may not reside at the same spatial location. As depicted in Fig. 1a, the Temporal Token Merging (TTM) strategy fails to capture video dynamics, erroneously merging less correlated tokens and distorting the video representations. Relying on such a rigid spatial correspondence for temporal redundancy compression may introduce noise, further misleading the model. So, a natural question arises: "How can we achieve a decent spatiotemporal compression by jointly modeling spatial and temporal redundancy, while accounting for the dynamic characteristics of video?"
Our Solution. To address this challenge, we introduce FlashVID, a novel training-free acceleration framework for VLLMs that effectively reduces spatiotemporal redundancy while preserving critical visual content. Specifically, at the core of FlashVID is the Tree-based Spatiotemporal Token Merging (TSTM) mechanism, which explicitly models both spatial and temporal redundancy through hierarchical spatiotemporal redundancy trees. TSTM enables structured token merging across frames and within frames, allowing for joint spatiotemporal compression that respects the natural structure of video data. However, directly constructing spatiotemporal trees based on the raw video features with excessive noise and redundancy may not focus on the most representative visual information in each frame, or even be biased towards the major but unimportant visual information, thereby affecting the final performance. To alleviate this issue, we further introduce the Attention and Diversity-based Token Selection (ADTS) module, which prioritizes representative tokens in each frame. To this end, through the initial filtering of informative tokens using ADTS and subsequent merging via TSTM, FlashVID accomplishes efficient compression that adjusts to the dynamic attributes of video content while preserving crucial semantics.
this section cite: ['b81', 'b7', 'b6', 'b4', 'b6', 'b21', 'b51']

Section: Vision Encoder Compression
What activities do students engage in within the room?
this section cite: []

Section: Tokenizer
Large Language Model
Visual Tokens Text Tokens Question Extensive experiments have been conducted on five video understanding benchmarks (Fu et al., 2025a;Mangalam et al., 2023;Wu et al., 2024;Li et al., 2024a;Zhou et al., 2025) and three representative VLLMs, i.g., LLaVA-OneVision (Li et al., 2025a), LLaVA-Video (Zhang et al., 2024), and Qwen2.5-VL (Bai et al., 2025b). As shown in Fig. 1b and Fig. 1c, FlashVID outperforms previous state-of-the-art methods by a large margin across all settings. Notably, FlashVID achieves 99.1% relative accuracy to vanilla LLaVA-OneVision while pruning 90% visual tokens. When integrated into Qwen2.5-VL, FlashVID enables processing of up to 10× more video frames, yielding an 8.6% performance gain over the vanilla model with 16 sampled frames under the same computational budget, highlighting its ability to unlock longer temporal context for better video understanding.
To summarize, our main contributions are threefold:
• In this work, we identify that existing token compression methods fail to effectively model the dynamic and evolving nature of video content, leading to suboptimal performance.
• We propose FlashVID, a training-free VLLMs acceleration method that introduces the Tree-based Spatiotemporal Token Merging (TSTM) to jointly model spatial and temporal redundancy across frames, complemented by Attention and Diversity-based Token Selection (ADTS) to obtain the semantically representative content within each frame.
• Extensive experiments show that FlashVID improves inference efficiency with negligible performance drop, and enables the use of longer input sequences for better video understanding within the constrained computational budget.
this section cite: ['b39', 'b66', 'b82', 'b81']

Section: BACKGROUND AND MOTIVATION
In this section, we provide a brief overview of the underlying concepts in this study in Sec. 2.1, and highlight the key observations in Sec. 2.2, which offer valuable insights for our approach.
this section cite: []

Section: PRELIMINARIES
VLLMs inference pipeline. The inference of VLLMs consists of three stages: (1) Encoding.
A vision encoder (e.g., CLIP (Radford et al., 2021) and SigLIP (Zhai et al., 2023)) processes each frame independently, producing N v visual embeddings per frame, which are projected into text space via a modality connector to form H v ∈ R F ×Nv×d . Text queries are embedded as H t ∈ R Nt×d . (2) Prefilling. Each LLM layer l computes self-attention over H via:
Q l = H l W l Q , K l = H l W l K , V l = H l W l V ,(1)
with
W l Q , W l K , W l V ∈ R d×d
. The key-value pairs are stored in the KV Cache for decoding acceleration. (3) Decoding. Response tokens are generated auto-regressively. At step t, only the new #Merged Tokens in TTM #Merged Tokens in TSTM Average Merged Similarities Difference #Merged Tokens in TTM #Merged Tokens in TSTM Frame Index Frame Index token h t is projected to (K t , V t ), which update the cache:
K ← concat[K, K t ], V ← concat[V, V t ].(2)
Such a caching mechanism substantially improves decoding efficiency.
this section cite: ['b46', 'b75']

Section: Efficiency bottleneck analysis.
While VLLMs have achieved remarkable performance on video understanding tasks, their efficiency remains a key challenge due to the heavy computational and memory overhead when processing a large number of visual tokens. Most of this cost stems from the LLM backbone, where the self-attention mechanism and Feed-Forward Networks (FFNs) dominate the computational complexity. Given a model with L Transformer layers, the total Floating Point Operations (FLOPs) can be formulated as:
FLOPs = L × (4nd 2 + 2n 2 d + 2ndm),(3)
with n denoting the sequence length, d the hidden dimension, and m the intermediate dimension of FFNs. In video understanding, the number of visual tokens n v dominates the sequence length n, typically exceeding textual tokens n t by orders of magnitude. This imbalance underscores the necessity to compress visual tokens for efficient inference in VLLMs.
Efficient inference paradigms. As illustrated in Fig. 2, visual token compression frameworks can be grouped into three paradigms: Before-LLM, Inner-LLM, and Hybrid Compression. Compressing tokens only inside the LLM is inefficient, as all visual tokens must still be processed in the shallow layers; thus, reducing tokens before the LLM is critical for reducing overhead. Existing methods (Yang et al., 2025c;Shen et al., 2025) adopt single-stage compression before the LLM, but extreme compression risks losing important visual information. Hybrid compression provides a balance: it retains sufficient tokens as LLM input while further pruning within the LLM to meet computational budget. Training-based approaches (Zhang et al., 2025d;Cai et al., 2025;Hu et al., 2024;Shao et al., 2025b) can mitigate this inefficiency but demand substantial computing resources; in this work, we focus on training-free strategies. A more comprehensive review is provided in Appendix D.
this section cite: ['b51', 'b5', 'b20']

Section: KEY OBSERVATIONS
We summarize two key observations about spatiotemporal redundancy in videos: (1) Temporal redundancy is not bound to fixed spatial locations. Semantically consistent elements in videos often
Large Language Model Vision Encoder Tokenizer What activities do students engage in within the room? Compression Question Text Tokens Compressed Visual Tokens Representative Visual Tokens Calibrate Calibrated Distance Matrices Select Attention and Diversity-based Token Selection (ADTS) Intra-Frame Distance Spatiotemporal Redundancy Tree Adjacent Frame Similarities Construct Aggregate Tree-based Spatiotemporal Token Merging (TSTM) Figure 4: Overview of our FlashVID. FlashVID compresses visual tokens by two synergistic modules: (1) ADTS prioritizes spatiotemporally informative tokens while ensuring feature diversity by solving a calibrated Max-Min Diversity Problem (MMDP); (2) TSTM models redundancy by spatiotemporal redundancy trees, which effectively capture fine-grained video dynamics.
shift in spatial position, scale, or appearance due to motion and scene dynamics, making rigid spatial correspondence across frames unreliable (Huang et al., 2025). (2) Spatial and temporal redundancy are inherently coupled. Redundant regions within a single frame frequently persist across multiple frames. Decoupled spatiotemporal redundancy compression overlooks the intrinsic spatiotemporal relationships, leading to suboptimal compression.
These insights suggest that existing frameworks lack a unified, structure-aware mechanism to capture spatiotemporal relationships under dynamic video conditions, which motivates our hierarchical tree-based spatiotemporal redundancy compression. A conceptual comparison with prior approaches is illustrated in Fig. 3, highlighting the unique advantages of our design.
this section cite: ['b21']

Section: METHODOLOGY
3.1 OVERVIEW As illustrated in Fig. 4, FlashVID integrates two synergistic modules: 1) Attention and Diversitybased Token Selection (ADTS), which first selects informative and diverse tokens for robust video representations; 2) Tree-based Spatiotemporal Token Merging (TSTM), which further minimizes spatiotemporal redundancy while preserving critical visual information.
this section cite: []

Section: TREE-BASED SPATIOTEMPORAL TOKEN MERGING
Videos exhibit dynamic variations in spatial position, scale, and appearance, posing challenges for spatiotemporal redundancy compression. To alleviate this, we propose Tree-based Spatiotemporal Token Merging (TSTM), which models the video redundancy via spatiotemporal redundancy trees.
Construct spatiotemporal redundancy trees. Given video features E v ∈ R F ×Nv×d , TSTM progressively builds spatiotemporal redundancy trees. First, we compute the cosine similarity matrix between visual features in adjacent frames:
S (f ) = cos(E (f ) v , E (f +1) v ) ∈ R Nv×Nv ,(4)
where S (f ) (j, k) measures the feature similarity between j-th token in frame f and k-th token in frame (f +1). Each token links to its most similar counterpart in the previous frame if their similarity exceeds a merging threshold T τ . This gradually forms redundancy trees that capture fine-grained temporal variations while avoiding merging dissimilar tokens.
this section cite: []

Section: Compress spatiotemporal redundancy.
Once the redundancy trees are constructed, tokens within each tree are aggregated:
c (i) = Agg(T (i) ),(5)
Algorithm 1 FlashVID Compression Require: Video features E v ∈ R F ×Nv×d ; similarity function sim(•, •); merging threshold T τ Ensure: Compressed token set X 1: Stage 1: Attention and Diversity-based Token Selection (ADTS) 2: for f = 1 to F do 3: Compute pairwise distance D (f ) , [CLS] attention A (f ) [CLS] , event relevance S(f) e 4: I (f ) ← MMDP(D (f ) , A (f ) [CLS] , S(f) e ) 5: R (f ) ← E (f ) v \ I (f ) 6: end for 7: Stage 2: Tree-based Spatiotemporal Token Merging (TSTM) 8: Initialize each token in R (f ) as a root node and let C be an empty token set. 9: for f = 2 to F do ▷ Build spatiotemporal redundancy trees 10:
for each token r f i ∈ R (f ) do 11:
p * ← arg max p∈R (f -1) sim(r f i , p) 12: if sim(r f i , p * ) ≥ T τ then 13: Connect r f i to p * 14:
end if 15:
end for 16: end for 17: for each tree T do ▷ Aggregate redundancy trees 18:
C ← C ∪ Agg(T ) 19: end for 20: return X ← C ∪ ( F f =1 I (f ) )
where T (i) denotes the i-th spatiotemporal redundancy tree and Agg(•) represents an aggregation function (e.g., mean pooling), producing compact yet informative spatiotemporal representations.
The quality of redundancy trees is critical for fine-grained compression. Although we've explored constraining tree depth and breadth to prevent merging spatiotemporally distant tokens, it yielded negligible gains; thus, no such constraints are applied in practice (see Appendix A.3 for details.)
this section cite: []

Section: ATTENTION AND DIVERSITY-BASED TOKEN SELECTION
Although TSTM effectively compresses spatiotemporal redundancy, it may discard important tokens in noisy and high-volume inputs. To mitigate this, we introduce the Attention and Diversity-based Token Selection (ADTS) module, which prioritizes spatiotemporally informative tokens within each frame while ensuring feature diversity for robust video representations. ADTS formulates token selection as a frame-wise Max-Min Diversity Problem (MMDP) (Alvar et al., 2025). Given video features E v ∈ R F ×Nv×d , we first compute the frame-wise cosine distance matrix:
D (f ) = 1 -cos(E (f ) v , E (f ) v ),(6)
where D (f ) ∈ R Nv×Nv denotes the pairwise feature dissimilarities in frame f . Solving MMDP on D (f ) yields a diverse token subset in frame f with the maximal minimum distance. However, diversity alone may overlook the most informative visual tokens. To address this issue, we introduce two calibration terms: 1) [CLS] attention and 2) event relevance.
[CLS] attention calibration. We extract the attention matrices from the vision encoder. For those encoders without an explicit [CLS] token (e.g., SigLIP (Zhai et al., 2023)), we derive it from the attention matrix:
A = Softmax(QK T / √ d) ∈ R F ×Nv×Nv ,(7)
and compute A [CLS] ∈ R F ×Nv by averaging attention weights each token receives within its frame. This calibration highlights informative tokens in each frame.
this section cite: ['b1', 'b75']

Section: Event relevance calibration.
Event relevance measures a token's correlation with the current video context. We obtain frame embeddings f v = GAP(E v ) ∈ R F ×d by global average pooling and compute the event similarity matrices:
Se = 1 F F i=1 (E v • f v ⊤ )[:, :, i] ∈ R F ×Nv .(8)
This calibration emphasizes the tokens most relevant to the video event. Finally, spatiotemporally informative tokens are selected by solving:
I = MMDP(D, A [CLS] , Se ).(9)
As summarized in Alg. 1, FlashVID compresses video redundancy in two stages: ADTS first selects spatiotemporally informative tokens by solving a calibrated Max-Min Diversity Problem (see Appendix C.2 for details), then TSTM merges redundant tokens across frames through spatiotemporal redundancy trees, yielding compact yet informative visual features.
this section cite: []

Section: EXPERIMENTS
In this section, we conduct extensive experiments across multiple benchmarks and VLLMs. We provide a brief introduction to the experimental settings in Sec. 4.1, present the main experimental results in Sec. 4.2, and discuss essential ablation studies in Sec. 4.3.
this section cite: []

Section: EXPERIMENTAL SETTINGS
Benchmarks. We evaluate our method on five widely-used video understanding benchmarks: VideoMME (Fu et al., 2025a), EgoSchema (Mangalam et al., 2023), LongVideoBench (Wu et al., 2024), MVBench (Li et al., 2024a), and MLVU (Zhou et al., 2025). Notably, these benchmarks cover a wide range of video durations and complex scenarios, providing a comprehensive evaluation of our method's effectiveness and generalization. Additional details can be found in Appendix B.
this section cite: ['b39', 'b66', 'b82']

Section: Compared baselines.
We compare FlashVID with four state-of-the-art training-free VLLM acceleration methods: 1) FastV (Chen et al., 2024), which selects prompt-relevant tokens via textto-visual attention at the prefilling stage; 2) VisionZip (Yang et al., 2025c), pruning tokens using [CLS] attention and spatial merging before the LLM; 3) PruneVID (Huang et al., 2025), combining spatiotemporal token merging with attention-based selection in the LLM; and 4) FastVID (Shen et al., 2025), compressing redundant tokens via density-based spatiotemporal pruning.
this section cite: ['b6', 'b21', 'b51']

Section: Implementation details.
We evaluate our method on three representative VLLMs: LLaVA-OneVision (Li et al., 2025a), LLaVA-Video (Zhang et al., 2024), and Qwen2.5-VL (Bai et al., 2025b), which cover diverse architectures to ensure generality. Following the official setting, LLaVA-OneVision and LLaVA-Video uniformly sample 32 and 64 frames, producing 32 × 196 and 64 × 169 visual tokens, respectively. For LLaVA-Video, we adopt frame token setting, facilitating adaptation for different acceleration frameworks. To ensure a fair comparison, we align the average token budget per transformer layer. Since TSTM compresses redundancy via thresholding, we further apply frame-wise token compression based on DPC-kNN to meet the predefined token budget. Unless otherwise specified, we utilize the same set of hyperparameters for all experiments.
All the experiments are conducted on NVIDIA A800 80G GPUs using LMMs-Eval (Zhang et al., 2025b). Additional implementation details are provided in Appendix C.
this section cite: ['b81']

Section: MAIN RESULTS
We evaluate FlashVID against state-of-the-art baselines on three representative VLLMs with distinct architectures under various retention ratios R. degradation at 15%, 10% due to excessive loss from aggressive spatial compression. FastV shows the weakest performance, as early-layer pruning is unstable. In contrast, FlashVID achieves the best results across all retention ratios, preserving 99.1% of the vanilla model's accuracy even at R = 10%. Moreover, when R ∈ {25%, 20%, 15%}, FlashVID surpasses the vanilla LLaVA-OneVision with full visual tokens input, revealing a "less is more" pattern where excessively redundant tokens may degrade performance.
Results on LLaVA-Video. LLaVA-Video employs a specialized design by inserting newline tokens to inject spatiotemporal positional information. Unlike the official grid token setting, we apply the frame token in LLaVA-Video, which facilitates adaptation for different acceleration frameworks, where we found that these two settings lead to similar performance. In Tab. 1, we evaluate our method against other methods on LLaVA-Video. Notably, our FlashVID outperforms all baselines under various retention ratios.
Results on Qwen2.5-VL In addition to LLaVA-OneVision and LLaVA-Video, we also evaluate our FlashVID against other methods on Qwen2.5-VL, which shows significantly different archi- Results on Qwen2.5-VL under fixed token budget. Due to computational and memory constraints, existing VLLMs typically process only a small number of sampled frames, often missing important visual cues. To assess the benefit of longer temporal context under a fixed computational budget, we apply token compression to enable models to process more frames. As shown in Tab. 3, Qwen2.5-VL achieves consistent improvements over its vanilla 16-frame baseline when equipped with token compression frameworks. Among them, FlashVID delivers the largest performance gains, highlighting its ability to unlock longer video sequences and demonstrating superior efficiency in constrained settings.
this section cite: []

Section: ABLATION STUDIES
In this section, we conduct ablation studies on the ADTS module and the retained ratio α of ADTS and TSTM using LLaVA-OneVision. Additional ablation studies are provided in Appendix. A.3.
this section cite: []

Section: Ablation study on ADTS module.
ADTS is proposed to select both important and diverse tokens. As shown in Tab. 4, we compare our ADTS with ATS and DTS, i.e., attention-based and diversity-based token selection. Our ADTS outperforms other token selection methods based solely on [CLS] attention (ATS) and feature diversity (DTS) by a large margin, demonstrating that ADTS can effectively identify the important visual tokens.
To realize a comprehensive ablation study on ADTS, we further ablate the calibration terms used in ADTS. Tab. 4 reveals that both [CLS] attention and event relevance calibration improve performance, while the optimal performance is yielded at the combination of the two. Ablation study on α. As illustrated in Tab. 5, we conduct an ablation study on merging threshold α, which controls the ratio of visual tokens retained between ADTS and TSTM compression. In particular, α = 0 and α = 1 denote TSTM and ADTS only, respectively. The experimental results show that ADTS alone (α = 1) outperforms TSTM alone (α = 0). However, the peak performance is achieved at α = 0.7, implying that a balanced integration of these two modules (i.e., ADTS and TSTM) is necessary to maintain the model performance.
this section cite: []

Section: EFFICIENCY ANALYSIS
Although token compression can effectively improve the inference efficiency of VLLMs, it can also be a time-consuming operation. In Tab. 6, we conduct an efficiency experiment on LLaVA-OneVision using a single NVIDIA A100 GPU compared to FastVID on VideoMME. Remarkably, FlashVID preserves 99.1% relative accuracy at R = 10%, while FastVID achieves a similar performance at R = 25%. Consequently, FlashVID enables 6.3× prefilling and 2.1× Time-To-First-Token (TTFT) speedups, largely outperforming FastVID. Additional efficiency experimental results on LLaVA-Video can be found in Appendix. A.2.
this section cite: []

Section: CONCLUSION
In this work, we introduce FlashVID, a training-free and plug-and-play acceleration framework for VLLMs. FlashVID combines Attention and Diversity-based Token Selection (ADTS) for representative token filtering with Tree-based Spatiotemporal Token Merging (TSTM) for fine-grained redundancy elimination, effectively compressing spatiotemporal redundancy while preserving essential visual information. Extensive experiments on three VLLMs across five video understanding benchmarks demonstrate that FlashVID achieves superior performance in both efficiency and accuracy. In particular, it can serve as a plug-and-play module, enabling VLLMs to process significantly longer video sequences under a constrained computational budget.
this section cite: []

Section: A MORE EXPERIMENTAL RESULTS
We present comprehensive experimental results of our method. In the Appendix. A.1, we evaluate our FlashVID against previous state-of-the-art-methods on Qwen2.5-VL (Bai et al., 2025b). In the Appendix. A.2, we present additional experimental results on LLaVA-Video. In the Appendix. A.3, we provide additional ablation studies on FlashVID.
this section cite: []

Section: A.1 ADDITIONAL EXPERIMENTS ON QWEN2.5-VL

this section cite: []

Section: Results on Qwen2.5-VL.
To further demonstrate the generalizability of our method, we evaluate it against other methods on Qwen2.5-VL (Bai et al., 2025b), which shows significant differences relative to LLaVA-OneVision and LLaVA-Video. Tab. 2 presents a part of the experimental results on Qwen2.5-VL under retention ratios R ∈ {20%, 10%}. Additional experimental results when R ∈ {25%, 15%} on Qwen2.5-VL are provided in Tab. 7. Notably, our method consistently surpasses previous state-of-the-art methods under various retention ratios, demonstrating strong generalization across different VLLMs.
Results on Qwen2.5-VL under fixed token budget. By applying visual token compression, VLLMs can achieve performance gains by processing more video frames while maintaining the overall computational budget. As discussed in Sec. 4.2, we explore extending the number of input frames under a fixed token budget. Tab. 3 reports results with 5× and 10× frames, demonstrating that VLLMs benefit from longer temporal context without increasing computational cost. Additional results with 3× and 4× frames are presented in Tab. 8, revealing a consistent improvement trend. It highlights that FlashVID effectively compresses visual tokens and preserves compact yet informative representations.
this section cite: []

Section: Additional efficiency analysis
As illustrated in Tab. 6, we test the efficiency of our FlashVID on LLaVA-OneVision (Li et al., 2025a), comparing to FastVID (Shen et al., 2025). In Tab. 10, we further evaluate the efficiency of our FlashVID on LLaVA-Video (Zhang et al., 2024). We report the detailed prefilling time and Time-To-First-Token (TTFT). Notably, our FlashVID enables 5.3× prefilling and 1.9× Time-to-First-Token (TTFT) speedups over the vanilla LLaVA-Video while maintaining 95.9% relative accuracy at 10% retention ratio.
this section cite: ['b51', 'b81']

Section: A.3 ADDITIONAL ABLATION STUDIES
Ablation study on T τ . The merging threshold T τ plays an important role in the Tree-based Spatiotemporal Token Merging (TSTM) module. T τ directly influences the compression quality, in which increasing T τ reduces the merging strength and better preserves temporal details, whereas lowering T τ promotes aggressive compression but may merge less correlated tokens, probably introducing noise to the compact representation. As illustrated in Tab. 13, we conduct an ablation study on merging threshold T τ on LLaVA-OneVision and LLaVA-Video at R = 10%. FlashVID consistently achieves the best performance under different VLLMs when merging threshold T τ = 0.8.
Ablation study on tree depth and breadth constraints. In TSTM, video redundancy is jointly modeled by spatiotemporal redundancy trees, which connect the highly correlated spatiotemporal visual information. Intuitively, applying proper depth and breadth constraints avoids the merge of spatiotemporally distant tokens, which may improve the compression quality. In addition to the threshold parameter T τ , we also test with two extra parameters: 1) depth constraint: aims to maintain the temporal dynamics, preventing tokens from spanning an excessively long temporal range in the same tree; 2) breadth constraint: seeks to preserve the spatial locality, avoiding merging across overly large spatial regions. The detailed implementation of TSTM with depth and breadth constraints is provided in Alg. 2. However, as illustrated in Tab. 11 and Tab. 12, we conduct ablation studies on tree depth and breadth using LLaVA-OneVision. Experimental results show that depth and breadth constraints don't bring performance gains. We hypothesize that the merging threshold T τ delivers a similar effect.
Ablation study on f e . FlashVID retains more visual tokens input to the LLM while pruning within the LLM to satisfy the overall computational budget, avoiding the loss of important visual information. f e controls the expansion ratio, in which a large f e may lead to computational inefficiency, while a low value may lose critical information. In Tab. 14, we conduct an ablation study on f e on LLaVA-OneVision. FlashVID achieves the best performance (99.1% relative accuracy) when the expansion factor f e ∈ {1.25, 1.30}. We adopt f e = 1.25 for better efficiency.
this section cite: []

Section: B EVALUATION BENCHMARKS
The experiments are conducted on the following widely used video understanding benchmarks.
VideoMME. VideoMME (Fu et al., 2025a) is a comprehensive multi-modal evaluation benchmark on video understanding capabilities of VLLMs. It features 900 videos spanning 6 diverse domains and 30 subcategories, with durations ranging from 11 seconds to 1 hour. Each video is accompanied by high-quality human annotations, including 2,700 multiple-choice question-answer pairs.
LongVideoBench. LongVideoBench (Wu et al., 2024) is a comprehensive benchmark designed to evaluate VLLMs on long-context, interleaved video-language understanding. It characterizes 3,763 videos with durations ranging from 8 seconds to 1 hour. This benchmark comprises 6,678 humanannotated multiple-choice questions based on a novel "referring-reasoning" task, where models must retrieve and reason over specific multimodal contexts referenced in the questions, categorized into 17 fine-grained types across perception and relation levels.
MVBench. MVBench (Li et al., 2024a) is a comprehensive benchmark designed to evaluate temporal understanding in multi-modal video tasks, addressing the limitations of existing image-focused
this section cite: ['b66']

Section: Algorithm 2 Tree-based Spatiotemporal Token Merging with Depth and Breadth Constraints
Require: Token sequences {R (foot_0) , R (foot_1) , . . . , R (F ) } from F frames; similarity function sim(•, •); tree depth function depth(•); merging threshold T τ ; max tree depth d max ; neighborhood size k Ensure: Compressed token set C 1: Initialize each token in R (f ) as a root node and let C be an empty token set. 2: for f = 2 to F do ▷ Construct candidate edges 3:
for each token r f i ∈ R (f ) do 4: N (r f i ) ← candidate parents in R (f -1) within neighborhood k 5: p * ← arg max p∈N (r f i ) sim(r f i , p) 6: if sim(r f i , p * ) ≥ T τ then 7: Connect r f i to p * 8:
end if 9:
end for 10: end for 11: for f = F down to 2 do ▷ Backward depth pruning 12:
for each token r f i ∈ R
(f ) do 13: if depth(r f i ) = d max then 14:
Disconnect r f i from its parent 15: Mark r f i as a new root 16: end if 17: end for 18: end for 19: for each tree T do ▷ Aggregate redundancy trees 20:
C ← C ∪ Agg(T ) 21: end for 22: return C benchmarks. It consists of 20 systematically constructed tasks that require complex temporal reasoning skills, generated via static-to-dynamic transformation of static tasks.
EgoSchema. EgoSchema (Mangalam et al., 2023) consists of approximately 5,000 five-choice multiple-choice questions derived from 250 hours of egocentric video. It emphasizes long-form temporal reasoning, as each of its 289 three-minute clips requires tracking objects and actions over time spans that are 5-10× longer than those in previous datasets, thereby posing significant challenges for both spatial perception and extended temporal coherence.
MLVU. MLVU (Zhou et al., 2025) contains 3,102 multiple-choice questions across nine diverse long-video understanding tasks. It challenges models with videos ranging from 3 minutes to 2 hours, requiring reasoning over plot, temporal order, and event retrieval, thereby jointly testing fine-grained spatial recognition and long-range temporal reasoning.
this section cite: ['b39', 'b82']

Section: C IMPLEMENTATION DETAILS

this section cite: []

Section: C.1 REPRODUCTION DETAILS OF COMPARED BASELINES
Unless otherwise specified, all the experiments are conducted on NVIDIA A800 80G GPUs on LMMs-Eval (Zhang et al., 2025b) 1 . We evaluate all methods on three representative VLLMs with distinct architectures and characteristics: LLaVA-OneVision (Li et al., 2025a) and LLaVA-Video (Zhang et al., 2024) 2 , and Qwen2.5-VL (Bai et al., 2025b) foot_2 . All baseline methods are reimplemented in LMMs-Eval, following their official implementations:
• FastV (Chen et al., 2024) foot_3 (ECCV 2024). FastV prunes tokens at the K-th layer of the LLM using cross-modal attention scores, with a pruning ratio r. We follow the official settings with K = 2, using r ∈ {75%, 80%, 85%, 90%} for LLaVA-OneVision in Tab. 1 and Qwen2.5-VL in Tab. 2, while setting r ∈ {80%, 90%} for LLaVA-Video in Tab. 1.
• VisionZip (Yang et al., 2025c) foot_4 (CVPR 2025). VisionZip prunes visual tokens at the output of the vision encoder, conflicting with pooling operations in VLLMs and resulting in performance degradation. Following (Shen et al., 2025), we instead apply pruning after pooling for VisionZip. We follow the official setting by retaining both dominant and contextual ratios at a 54:10 ratio in each frame. We set R to {25%, 20%, 15%, 10%} for LLaVA-OneVision in Tab. 1 and Qwen2.5-VL in Tab. 2, while setting R to {20%, 10%} and {25%, 15%} for LLaVA-Video in Tab. 1 and Tab. 9, respectively.
• PruneVID (Huang et al., 2025) foot_5 (ACL 2025). PruneVID contains both before-LLM compression and inner-LLM pruning during the prefilling stage, along with a KV Cache compression at the decoding stage. Following the official settings, we set the threshold τ = 0.8, the temporal segment ratio γ = 0.25, the token selection ratio α = 0.4, and the pruning layer K = 10. We control the token budget by cluster ratio β. We use β ∈ {40.7%, 32.5%, 24.4%, 16.3%} in Tab. 1.
• FastVID (Shen et al., 2025) foot_6 (NeurIPS 2025). FastVID prunes visual tokens based on spatiotemporal DPC-kNN. It begins with a dynamic segmentation based on transition similarities, followed by a frame-wise salient token selection based on [CLS] attention scores. Finally, it compresses the remaining tokens by spatiotemporal redundancy elimination based on DPC-kNN. Following the official settings, we set the minimum number of segments c = 8, the segment threshold τ = 0.9, the salient token ratio d = 0.4, the anchor frame step p = 4, and the merging factor α = 0.6. We set R to {25%, 20%, 15%, 10%} for LLaVA-OneVision in Tab. 1 and Qwen2.5-VL in Tab. 2, while setting R to {20%, 10%} and {25%, 15%} for LLaVA-Video in Tab. 1 and Tab. 9, respectively.
this section cite: ['b81', 'b6', 'b51', 'b21', 'b51']

Section: C.2 REPRODUCTION DETAILS OF FLASHVID
In addition to ADTS and TSTM modules, FlashVID employs two design choices: 1) video partition and 2) Inner-LLM Pruning for better performance.
Video Partition. State-of-the-art VLLM acceleration methods (Shen et al., 2025;Shao et al., 2025a;Huang et al., 2025;Tao et al., 2025) commonly apply video partitioning before token compression, aiming to avoid information mixing and building upon DySeg (Shen et al., 2025), FlashVID partitions consecutive similar frames into the same segment based on the transition similarities. Instead of using [CLS] token embeddings, we compute transition similarities based on pooled video features. Given video features E v ∈ R F ×Nv×d , we apply global average pooling to obtain the frame embeddings:
f e = GAP(E v ) ∈ R F ×d . (10
) The transition similarities are defined as the cosine similarity of frame embeddings of adjacent frames:
t i = cos(f i , f i+1 ), i = 1, 2, • • • , F -1, T = {t 1 , t 2 , • • • , t F -1 } (11
)
where t i denotes the transition similarity between i-th and (i + 1)-th frame. A low transition similarity indicates a significant scene change. Following DySeg, we set the segment threshold S τ = 0.9 and the minimum number of segments M s = 8.
this section cite: ['b51', 'b21', 'b53', 'b51']

Section: Calibrated Max-Min Diversity Problem.
As discussed in Sec.
3.3, FlashVID utilizes the Attention and Diversity-based Token Selection (ADTS) module to identify the spatiotemporally informative tokens within each frame. Specifically, ADTS formulates frame-wise token selection as Algorithm 3 Calibrated Max-Min Diversity Problem (MMDP) Require: Pairwise distance D (f ) ∈ R Nv×Nv ; [CLS] attention A (f ) [CLS] ∈ R Nv ; event relevance S(f) e ∈ R Nv Ensure: Spatiotemporally informative token indices I (f ) 1: Initialize selected indices I (f ) ← ∅ and R ← {0, 1, ...N v -1} 2: Let 1 Nv ∈ R Nv be an all-ones vector 3: D (f ) ← D (f ) ⊙ A (f ) [CLS] ⊗ 1 Nv ⊙ S(f) e ⊗ 1 Nv ) ▷ Calibrate pairwise distance 4: for i ∈ R do ▷ Select the first token 5: d min [i] ← min j∈R,j̸ =i D (f ) i,j 6: end for 7: k ← arg max d min 8: Move k from R to I (f ) 9: while |I (f ) | < M do ▷ Iteratively add the subsequent tokens 10: Initialize d min ← inf 11: for i ∈ R do 12: d min [i] ← min j∈I (f ) D (f ) i,j 13: end for 14: k ← arg max d min 15: Move k from R to I (f ) 16: end while 17: return I (f )
a Max-Min Diversity Problem (MMDP), calibrated by [CLS] attention and event relevance. The detailed implementation is provided in Alg. 3.
Inner-LLM Pruning. As illustrated in Sec. 2.1, the hybrid compression framework balances efficiency and performance, which preserves sufficient visual information input to the LLM, preventing the loss of important information. FlashVID employs this design for better performance, which retains more visual tokens before the LLM and prunes at a relatively high layer. We set the pruning layer K = 20 and the expansion factor f e = 1.25 without careful tuning for LLaVA-OneVision, LLaVA-Video, and Qwen2.5-VL.
this section cite: []

Section: C.3 TOKEN BUDGET ALIGNMENT
To ensure a fair comparison, we employ a simple and effective strategy that aligns the average number of visual tokens processed by each Transformer layer to meet a similar computational cost, following (Shao et al., 2025b). Eq. 3 presents the Floating Point Operations (FLOPs) formula of the standard Transformer architecture for generality. In this paper, we evaluate three representative VLLMs: LLaVA-OneVision (Li et al., 2025a), LLaVA-Video (Zhang et al., 2024), Qwen2.5-VL (Bai et al., 2025b), which share similar LLM architectures that employ Group Query Attention (Ainslie et al., 2023) and SwiGLU (Dauphin et al., 2017) non-linear activation. The computational FLOPs of these three LLMs can be formulated as:
FLOPs = L × (2nd 2 (1 + g/h) + 2n 2 d + 3ndm), (12
)
where n is sequence length, d the hidden dimension, m the intermediate dimension of FFNs, g the number of key/value heads, and h the number of attention heads. Since the number of visual tokens n v dominates the sequence length n, the sequence length n can be approximated by n v .
To clarify how visual token numbers are determined at each stage. We provide a detailed explanation. Let R be the average retained visual tokens per Transformer layer, M be the number of tokens entering the LLM (after before-LLM compression), K be the pruning layer index, L be the number of Transformer layers in LLM, and R be the number of retained visual tokens (after inner-LLM pruning). Then we have the following equation.
RL = M K + R(L -K).(13)
We introduces an expansion factor f e such that M = f e R ; thus, we have:
R = R(L -f e K) L -K .(14)
And the inner-LLM pruning ratio r becomes:
r = R M = L -f e K f e (L -K) .(15)
Such a simple token budget alignment strategy enables fair comparisons between different acceleration frameworks.
this section cite: ['b81', 'b0']

Section: D RELATED WORK
Multimodal Large Language Models. Recent advances in deep learning (He et al., 2016;Vaswani et al., 2017;Devlin et al., 2019;Dosovitskiy et al., 2021;Radford et al., 2021;He et al., 2022;Cui et al., 2022;2023;Peng et al., 2024a;Yang et al., 2024c;Wang et al., 2024a) have benefited traditional computer vision tasks, such as semantic segmentation and object detection (Tian et al., 2020;Lai et al., 2021;Jiang et al., 2021;Peng et al., 2023;Tian et al., 2022b;Luo et al., 2023;Peng et al., 2024b;Tian et al., 2022a;2019;2023;Ning et al., 2023;Shao et al., 2024;Wang et al., 2025a;b). In particular, transformer-based architectures and large-scale pretraining have increasingly driven the success of Large Language Models (LLMs) (Touvron et al., 2023;Grattafiori et al., 2024;Yang et al., 2024a;b;2025a;Lai et al., 2024b;Peng et al., 2025;Liu et al., 2024a;Guo et al., 2025), exhibiting strong generalization and reasoning capabilities. Building upon LLMs, Multimodal Large Language Models (MLLMs) (Liu et al., 2023;2024b;c;Dai et al., 2023;Li et al., 2023;Comanici et al., 2025;Bai et al., 2025b;a;Wang et al., 2025c;Li et al., 2025c) extend the input modality from text to multimodalities (such as image, audio, and video) by coupling modality encoders with LLM backbones. So far, MLLMs have revolutionized traditional computer vision tasks. For example, representative works like LISA (Lai et al., 2024a) and LISA++ (Yang et al., 2023) study reasoning segmentation powered by MLLMs.
Video Large Language Models. With the rapid advancement of MLLMs, Video Large Language Models (VLLMs) (Li et al., 2025a;Zhang et al., 2024;Bai et al., 2025b;a;Shen et al., 2024;Li et al., 2024b;Maaz et al., 2024;Comanici et al., 2025) have gained increasing attention. Mainstream VLLMs directly process raw video tokens with an optional pooling operation. LLaVA-OneVision (Li et al., 2025a) demonstrates strong video understanding capabilities through task transfer from images. To achieve fine-grained spatiotemporal modeling, some VLLMs employ elaborate designs. LLaVA-Video (Zhang et al., 2024) introduces newline tokens to distinguish spatiotemporal positions. Qwen2-VL (Wang et al., 2024b) and Qwen2.5-VL (Bai et al., 2025b) use Multimodal Rotary Position Embedding (MRoPE). Qwen3-VL (Bai et al., 2025a) employs the Deepstack mechanism (Meng et al., 2024), which extracts visual tokens from intermediate layers of the visual encoder and injects them into the LLM, preserving rich visual information.
To achieve a comprehensive evaluation, we evaluate our method on three representative VLLMs (i.e., LLaVA-OneVision, LLaVA-Video, and Qwen2.5-VL) with significantly different architectures and characteristics.
this section cite: ['b18', 'b60', 'b12', 'b13', 'b46', 'b19', 'b8', 'b55', 'b24', 'b23', 'b42', 'b37', 'b41', 'b49', 'b59', 'b16', 'b45', 'b34', 'b10', 'b28', 'b7', 'b72', 'b81', 'b52', 'b38', 'b7', 'b81', 'b40']

Section: Visual Token Compression.
Token compression has emerged as an effective technique that reduces computational complexity in transformer architectures, such as Vision Transformers (ViTs) (Dosovitskiy et al., 2021) and Large Language Models (LLMs). ToMe (Bolya et al., 2023) gradually merges similar tokens in ViTs. FastV (Chen et al., 2024) identifies text-relevant visual tokens based on text-to-visual attention in the LLM. PyaramidDrop (Xing et al., 2024) and SparseVLM (Zhang et al., 2025e) progressively prunes visual tokens. VisionZip (Yang et al., 2025c), LLaVA-PruMerge (Shang et al., 2025), and VisPruner (Zhang et al., 2025c) filter salient visual tokens via [CLS] attention, while DivPrune (Alvar et al., 2025) selects based on diversity. TopV (Yang et al., 2025b) formulates token selection as an optimization problem. VScan (Zhang et al., 2025a) combines global and local scans for informative visual token selection.
However, the above methods only focus on spatial redundancy compression. To address this, several token compression frameworks for VLLMs have been proposed. DyCoke (Tao et al., 2025) merges redundant tokens in each segment. PruneVID (Huang et al., 2025) distinguishes static and dynamic tokens. STTM (Hyun et al., 2025) models video redundancy by a quadtree. FrameFusion (Fu et al., 2025b) performs both merging and pruning in the LLM. HoliTom (Shao et al., 2025a) combines global redundancy-aware video partition with spatial and inner-LLM compression. FastVID (Shen et al., 2025) employs density-based token pruning. DyTok (Li et al., 2025b) dynamically allocates token budget to each frame or segment, serving as a plug-and-play module for existing token compression methods.
this section cite: ['b13', 'b4', 'b6', 'b67', 'b47', 'b53', 'b21', 'b22', 'b51']

Section: E MORE VISUALIZATIONS E.1 TREE-BASED SPATIOTEMPORAL TOKEN MERGING
Due to the dynamic and evolving nature of video, the most semantically correlated visual features in adjacent frames are likely to experience variation in position, scale, orientation, and other attributes over time. To address this challenge, we propose the Tree-based Spatiotemporal Token Merging (TSTM) mechanism, which models video redundancy by spatiotemporal redundancy trees in a unified way. It enables capturing fine-grained video dynamics. Fig. 5 presents more visualizations of TSTM, highlighting the unique advantages of our TSTM for better spatiotemporal redundancy compression.
this section cite: []

Section: E.2 QUALITATIVE ANALYSIS ON LLAVA-ONEVISION
As illustrated in Tab. 1, we evaluate our FlashVID on LLaVA-OneVision at R ∈ {25%, 20%, 15%, 10%}. Notably, at higher retention ratios (i.e., R = 25%, 20%, 15%), FlashVID surpasses the vanilla LLaVA-OneVision, indicating a "less is more" pattern where excessively redundant tokens may degrade performance. Additionally, FlashVID preserves performance of 99.1% under extreme compression (e.g., R = 10%). In this work, we explore extending VLLMs to process more video frames under a fixed computational budget through visual token compression. As reported in Tab. 3 and Tab. 8, VLLMs benefit from longer temporal context. Fig. 7 presents four qualitative examples comparing Qwen2.5-VL with and without FlashVID, highlighting its ability to capture richer temporal information. FlashVID enables Qwen2.5-VL (Bai et al., 2025b) to process 10× more frames (160 vs. 16) within the same computational cost, providing compact yet informative video representations and improving the model performance.
this section cite: []

Section: E.4 VISUALIZATIONS OF ADTS
As shown in fig. 8, we compare token selection results by ADTS with and without event relevance calibration. Event relevance calibration helps identify the key visual tokens, thereby improving the performance of those tasks requiring fine-grained understanding.
this section cite: []

Section: E.5 VISUALIZATIONS OF FAILURE CASES IN TSTM
As illustrated in Fig. 9, we present visualizations of failure cases in our Tree-based Spatiotemporal Token Merging (TSTM). Although TSTM enables fine-grained spatiotemporal redundancy compression, it might result in merging operations with semantic confusion such as merging tokens from different entities with similar semantic information.
this section cite: []

Section: E.6 VISUAL PERCEPTION LAYERS
We empirically found that certain transformer layers (deep layers) of VLLMs possess strong visual perception capabilities. These visual perception layers can typically identify keyframes. Fig. (a) Visualization of TSTM (Example 1) (b) Visualization of TSTM (Example 2) Figure 9: Visualizations of failure cases in TSTM.
presents several visualizations of visual perception layers. Building upon this insight, we hypothesize that token compression at these layers yields negligible performance degradation.
To balance efficiency and performance, FlashVID adopts a hybrid compression paradigm that retains more visual tokens and prunes visual tokens in the LLM to control the overall computational budget. Hence, we consistently set the pruning layer K = 20 (a relatively high layer for LLaVA-OneVision, LLaVA-Video, and Qwen2.5-VL at 7B scale) without careful tuning.
this section cite: []

Section: F USAGE OF LARGE LANGUAGE MODELS
In this work, Large Language Models (LLMs) are only used for polishing the paper writing. They are not involved in research ideation, experimental design, data analysis, or the formulation of conclusions. All substantive intellectual contributions are made by the authors.
this section cite: []

Section: References
Ref_id:b0 Title: GQA: training generalized multi-query transformer models from multi-head checkpoints Year: (2023)
Ref_id:b1 Title: Divprune: Diversitybased visual token pruning for large multimodal models Year: (2025)
Ref_id:b2 Title:  Year: (2025-06)
Ref_id:b3 Title: Qwen2.5-vl technical report Year: (2025)
Ref_id:b4 Title: Token merging: Your vit but faster Year: (2023)
Ref_id:b5 Title: Matryoshka multimodal models Year: (2025)
Ref_id:b6 Title: An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models Year: (2024)
Ref_id:b7 Title: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities Year: (2025)
Ref_id:b8 Title: Reslt: Residual learning for long-tailed recognition Year: (2022)
Ref_id:b9 Title: Generalized parametric contrastive learning Year: (2023)
Ref_id:b10 Title: Instructblip: Towards general-purpose vision-language models with instruction tuning Year: (2023)
Ref_id:b11 Title: Language modeling with gated convolutional networks Year: (2017)
Ref_id:b12 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b13 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b14 Title: Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis Year: (2025)
Ref_id:b15 Title: Framefusion: Combining similarity and importance for video token reduction on large visual language models Year: (2025)
Ref_id:b16 Title: The llama 3 herd of models Year: (2024)
Ref_id:b17 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b18 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b19 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b20 Title: Matryoshka query transformer for large vision-language models Year: (2024)
Ref_id:b21 Title: Prunevid: Visual token pruning for efficient video large language models Year: (2025)
Ref_id:b22 Title: Multi-granular spatio-temporal token merging for training-free acceleration of video llms Year: (2025)
Ref_id:b23 Title: Guided point contrastive learning for semi-supervised point cloud semantic segmentation Year: (2021)
Ref_id:b24 Title: Semisupervised semantic segmentation with directional context-aware consistency Year: (2021)
Ref_id:b25 Title: Lisa:reasoning segmentation via large language model Year: (2024)
Ref_id:b26 Title: Step-dpo: Stepwise preference optimization for long-chain reasoning of llms Year: (2024)
Ref_id:b27 Title: Llava-onevision: Easy visual task transfer Year: (2025)
Ref_id:b28 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b29 Title: Mvbench: A comprehensive multi-modal video understanding benchmark Year: (2024)
Ref_id:b30 Title: Llama-vid: An image is worth 2 tokens in large language models Year: (2024)
Ref_id:b31 Title: Less is more, but where? dynamic token compression via llm-guided keyframe prior Year: (2025)
Ref_id:b32 Title: Perception, reason, think, and plan: A survey on large multimodal reasoning models Year: (2025)
Ref_id:b33 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b34 Title: Visual instruction tuning Year: (2023)
Ref_id:b35 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b36 Title: Llava-next: Improved reasoning, ocr, and world knowledge, 2024c Year: ()
Ref_id:b37 Title: Pfenet++: Boosting few-shot semantic segmentation with the noise-filtered context-aware prior mask Year: (2023)
Ref_id:b38 Title: Video-chatgpt: Towards detailed video understanding via large vision @ and language models Year: (2024)
Ref_id:b39 Title: Egoschema: A diagnostic benchmark for very long-form video language understanding Year: (2023)
Ref_id:b40 Title: Deeply stacking visual tokens is surprisingly simple and effective for lmms Year: (2024)
Ref_id:b41 Title: Boosting few-shot 3d point cloud segmentation via query-guided enhancement Year: (2023)
Ref_id:b42 Title: Hierarchical dense correlation distillation for few-shot segmentation Year: (2023)
Ref_id:b43 Title: Scalable language model with generalized continual learning Year: (2024)
Ref_id:b44 Title: Oa-cnns: Omni-adaptive sparse cnns for 3d semantic segmentation Year: (2024)
Ref_id:b45 Title: Mitigating object hallucinations via sentence-level early intervention Year: (2025)
Ref_id:b46 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b47 Title: Llava-prumerge: Adaptive token reduction for efficient large multimodal models Year: (2025)
Ref_id:b48 Title: Holitom: Holistic token merging for fast video large language models Year: (2025)
Ref_id:b49 Title: Explore the potential of clip for trainingfree open vocabulary semantic segmentation Year: (2024)
Ref_id:b50 Title: Growing a twig to accelerate large vision-language models Year: (2025)
Ref_id:b51 Title: Fastvid: Dynamic density pruning for fast video large language models Year: (2025)
Ref_id:b52 Title: Spatiotemporal adaptive compression for long video-language understanding Year: (2024)
Ref_id:b53 Title: Dycoke: Dynamic compression of tokens for fast video large language models Year: (2025)
Ref_id:b54 Title: Learning shape-aware embedding for scene text detection Year: (2019)
Ref_id:b55 Title: Prior guided feature enrichment network for few-shot segmentation Year: (2020)
Ref_id:b56 Title: Adaptive perspective distillation for semantic segmentation Year: (2022)
Ref_id:b57 Title: Generalized few-shot semantic segmentation Year: (2022)
Ref_id:b58 Title: Learning context-aware classifier for semantic segmentation Year: (2023)
Ref_id:b59 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b60 Title: Attention is all you need Year: (2017)
Ref_id:b61 Title: Groupcontrast: Semantic-aware self-supervised representation learning for 3d understanding Year: (2024)
Ref_id:b62 Title: Declip: Decoupled learning for open-vocabulary dense perception Year: (2025)
Ref_id:b63 Title: Generalized decoupled learning for enhancing open-vocabulary dense perception Year: (2025)
Ref_id:b64 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b65 Title: Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency Year: (2025)
Ref_id:b66 Title: Longvideobench: A benchmark for long-context interleaved video-language understanding Year: (2024)
Ref_id:b67 Title: Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction Year: (2024)
Ref_id:b68 Title: Qwen2 technical report Year: (2024)
Ref_id:b69 Title: Qwen2.5 technical report Year: (2024)
Ref_id:b70 Title: Qwen3 technical report Year: (2025)
Ref_id:b71 Title: Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model Year: (2025)
Ref_id:b72 Title: Lisa++: An improved baseline for reasoning segmentation with large language model Year: (2023)
Ref_id:b73 Title: Unified language-driven zero-shot domain adaptation Year: (2024)
Ref_id:b74 Title: Visionzip: Longer is better but not necessary in vision language models Year: (2025)
Ref_id:b75 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b76 Title: Vscan: Rethinking visual token reduction for efficient large vision-language models Year: (2025)
Ref_id:b77 Title: Lmms-eval: Reality check on the evaluation of large multimodal models Year: (2025)
Ref_id:b78 Title: Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms Year: (2025)
Ref_id:b79 Title: Llava-mini: Efficient image and video large multimodal models with one vision token Year: (2025)
Ref_id:b80 Title: Sparsevlm: Visual token sparsification for efficient vision-language model inference Year: (2025)
Ref_id:b81 Title: Video instruction tuning with synthetic data Year: (2024)
Ref_id:b82 Title: Mlvu: Benchmarking multi-task long video understanding Year: (2025)
