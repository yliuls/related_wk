Title: LoRATv2: Enabling Low-Cost Temporal Modeling in One-Stream Trackers
Abstract: Transformer-based algorithms, such as LoRAT, have significantly enhanced objecttracking performance. However, these approaches rely on a standard attention mechanism, which incurs quadratic token complexity, making real-time inference computationally expensive. In this paper, we introduce LoRATv2, a novel tracking framework that addresses these limitations with three main contributions. First, LoRATv2 integrates frame-wise causal attention, which ensures full selfattention within each frame while enabling causal dependencies across frames, significantly reducing computational overhead. Moreover, key-value (KV) caching is employed to efficiently reuse past embeddings for further speedup. Second, building on LoRAT's parameter-efficient fine-tuning, we propose Stream-Specific LoRA Adapters (SSLA). As frame-wise causal attention introduces asymmetry in how streams access temporal information, SSLA assigns dedicated LoRA modules to the template and each search stream, with the main ViT backbone remaining frozen. This allows specialized adaptation for each stream's role in temporal tracking. Third, we introduce a two-phase progressive training strategy, which first trains a single-search-frame tracker and then gradually extends it to multi-searchframe inputs by introducing additional LoRA modules. This curriculum-based learning paradigm improves long-term tracking while maintaining training efficiency. In extensive experiments on multiple benchmarks, LoRATv2 achieves state-of-the-art performance, substantially improved efficiency, and a superior performance-to-FLOPs ratio over state-of-the-art trackers. The code is available at https://github.com/LitingLin/LoRATv2.

Section: Introduction
Visual object tracking is a fundamental task in computer vision with broad applications in surveillance, autonomous driving, robotics, augmented reality, and human-computer interaction. Accurate and continuous object localization across frames is crucial for the reliability of these applications. Recent advances in Vision Transformers (ViT) [14] have substantially improved tracking performance, primarily due to the powerful representation capabilities of self-attention. In particular, one-stream trackers, such as MixFormer [9], OSTrack [43], ARTrack [35], SeqTrack [7], and LoRAT [24], unify feature extraction and relation modeling within a single Transformer backbone. Although this design has demonstrated excellent accuracy and efficiency, these methods rely on standard self-attention, which has a quadratic complexity with respect to the number of tokens. Consequently, inference can become prohibitively expensive when dealing with high-resolution input frames or extended temporal sequences, constraining real-time deployment on resource-limited systems.
LoRAT [24], a recently proposed state-of-the-art tracker, employs Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) [19] to significantly reduce the training overhead for largescale Transformer trackers. However, it retains the standard self-attention mechanism at inference time, leading to substantial computational costs as input resolution or frame count increases. Furthermore, while one-stream trackers [43,24] excel with single template-search pairs, simply extending them to multi-frame inputs using standard bidirectional attention often yields only marginal performance gains in our preliminary experiments. We hypothesize this may be partly due to potential rank collapse issues in deep Transformers with bidirectional attention, which can limit their expressive power for complex temporal sequences [13].
To overcome these limitations and better model the sequential nature of tracking, we propose LoRATv2. Visual object tracking can naturally be framed as an autoregressive sequence prediction task, where the target's state in the current frame depends on its history. This perspective aligns well with causal attention mechanisms. Therefore, LoRATv2 introduces the frame-wise causal attention mechanism within the ViT backbone. In this design, tokens within the current frame maintain full self-attention for rich intra-frame representation, while attending causally only to tokens from previous frames, enforcing an autoregressive structure. This approach, coupled with key-value (KV) caching [33,11,31] to reuse past embeddings, not only significantly enhances inference efficiency by reducing MACs and improving FPS, but as our experiments demonstrate (shown in Fig. 1), also leads to improved tracking performance by effectively modeling temporal dependencies.
The adoption of frame-wise causal attention, while beneficial for temporal modeling and efficiency, introduces an inherent asymmetry in how different input streams (e.g., the static template, the first search frame, subsequent search frames) process information. Unlike traditional Siamese trackers or one-stream trackers with full attention where all tokens potentially interact symmetrically, in our causal setup each stream has a distinct view of historical context, as it can only attend to itself and preceding information. To manage this asymmetry effectively, while preserving LoRAT's parameterefficient fine-tuning paradigm, we propose Stream-Specific LoRA Adapters (SSLA). SSLA allocates dedicated LoRA [19] adapters to each input stream, and the main ViT backbone is kept frozen and shared. This allows each stream to develop specialized adaptations. For instance, the template's LoRA can focus on robust initial feature extraction, the first search frame's LoRA might learn to enhance target features while identifying distractors, and LoRAs for subsequent search frames can specialize in precise localization based on the evolving temporal context. This modification maintains the benefits of a shared backbone yet allows minimal, stream-dependent adaptation critical for effective multi-frame tracking.
Finally, we present a two-phase progressive training strategy to incrementally expand from singlesearch-frame to multi-search-frame tracking. This approach significantly reduces the computational resources required for training, as at any given time, only the LoRA modules associated with one or two input streams are trainable, while previously trained LoRA modules and the main backbone remain frozen. This easy-to-hard, curriculum learning-style [2] training paradigm, as demonstrated in our experiments, leads to better tracking performance compared to training all LoRA modules from scratch. Furthermore, this strategy naturally yields a family of trackers capable of processing different numbers of input frames (e.g., a single-search-frame tracker from the initial stage and a multi-search-frame tracker from the extension stage), providing additional flexibility for deployment scenarios with varying computational budgets or accuracy requirements.
Our contributions can be summarized as follows:
1. Frame-wise Causal Attention: We integrate frame-wise causal attention into one-stream visual tracking, where tokens attend fully within their frame and causally to past frames. Combined with KV caching to reuse past embeddings and prevent re-encoding, this mechanism enables computationally efficient long-term temporal modeling for trackers and improves tracking accuracy by robustly capturing temporal context.
this section cite: ['b13', 'b8', 'b42', 'b34', 'b6', 'b23', 'b23', 'b18', 'b42', 'b23', 'b12', 'b32', 'b10', 'b30', 'b18', 'b1']

Section: Stream-Specific LoRA Adapters (SSLA):
To handle the asymmetry introduced by causal connections, we equip each input stream (template or search) with its own low-rank LoRA modules. This preserves a shared backbone for all streams while allowing minimal, streamspecific adaptation for improved tracking accuracy, with zero additional inference cost.
this section cite: []

Section: Two-Phase Progressive Training:
We first train a single-frame tracker (template → one search frame) and then incrementally extend it to multi-frame inputs by adding new LoRA adapters only for the additional search frames. This curriculum-like approach improves long-term performance while greatly reducing memory and training time compared to direct multi-frame training from scratch.
this section cite: []

Section: 4.
Extensive experimental evaluations across multiple benchmark datasets demonstrate that our proposed approach delivers superior tracking performance and significantly improves performance-to-FLOPs ratio compared to existing state-of-the-art self-attention-based methods.
this section cite: []

Section: Related Work
Temporal modeling in tracking. Temporal modeling in visual tracking typically involves leveraging historical information and explicit temporal dependencies to enhance robustness and accuracy. Some methods employ historical prompts to improve tracking stability, such as AQATrack [40] and HipTrack [3]. Approaches like STARK [41] and TATrack [18] dynamically update tracking templates based on prior tracking outcomes to maintain effectiveness over time. Furthermore, autoregressive prediction strategies have been explored extensively in methods such as ARTrack [35], ARTrackV2 [1], and SeqTrack [7], which explicitly model temporal dependencies across frames.
Recent advancements also include multi-frame modeling methods. For example, TCTrack [5] incorporates video-level contextual information through adaptive convolutional techniques, whereas VideoTrack [37] uses transformer-based architectures to integrate broader temporal context. Additionally, ODTrack [45] explicitly propagates token sequences between frames to maintain dense contextual associations.
this section cite: ['b39', 'b2', 'b40', 'b17', 'b34', 'b0', 'b6', 'b4', 'b36', 'b44']

Section: Linear Projection
𝑃 !,! b 𝑃 !,# t 𝑃 #,! b 𝑃 #,# b 𝑃 !,! s 2 𝑃 !,# s 2 𝑃 !,$ s 2 𝑃 #,! s 2 𝑃 #,# s 2 𝑃 #,$ s 2 𝑃 $,! s 2 𝑃 $,# s 2 𝑃 $,$ s 2 Frozen Trainable 𝑃 !,! s 1 𝑃 !,# s 1 𝑃 #,! s 1 𝑃 #,# s 1 MLP LayerNorm LoRA Multi-head Causal Attention LayerNorm LoRA LoRA LoRA LoRA LoRA 𝐿× MLP head patch emb. type emb. pos. emb.
this section cite: []

Section: Causal attention in vision.
Causal attention in vision addresses spurious correlations by explicitly modeling causal relationships. Methods like CATT [42] introduce causal interventions within attention modules to mitigate contextual bias, enforcing object-context separation through samplewise masking. Building upon these foundations, subsequent approaches extend causal attention to autoregressive frameworks. Vision-RWKV [15], for example, adapts causal attention mechanisms for bidirectional global interactions within vision transformers. To enhance scalability and efficiency, some models incorporate hierarchical or sequential structures. VAR [32] employs causal masking strategies for efficient coarse-to-fine image generation. Similarly, Causal Vision Transformers [22] apply sequential causal attention to efficiently handle large-scale images. Moreover, Show-O [39] unifies multimodal generation tasks under a single causal transformer framework. Collectively, these advancements underscore the versatility and efficacy of causal attention methods across diverse visual tasks, spanning recognition to generation, while enabling scalable and controllable visual modeling.
this section cite: ['b41', 'b14', 'b31', 'b21', 'b38']

Section: Method
Our proposed LoRATv2 builds upon LoRAT [24] and introduces additional components for efficient multi-frame, temporal-aware visual object tracking. First, we revisit LoRAT to recap its core design principles (Sec. 3.1). Next, we present our Frame-Wise Causal Attention (Sec. 3.2) and Key-Value Caching (Sec. 3.3), which jointly enable efficient multi-frame modeling without re-encoding past frames. To address the asymmetric dependency introduced by causal attention, we propose Stream-Specific LoRA Adapters (Sec. 3.4), which preserve a unified embedding space while enabling minimal, per-stream adaptation. Finally, Sec. 3.5 details our Two-Phase Progressive Training pipeline for scaling from single-frame to multi-frame scenarios with minimal overhead. Unless otherwise noted, all other design and training settings (e.g., shared positional embeddings, token-type embeddings, anchor-free MLP heads) follow LoRAT [24].
T 1 T 2 S T 1 T 2 S (a) Full Self-Attention T 1 T 2 S T 1 T 2 S (b) Causal Attention T 1 T 2 S T 1 T 2 S (c) Frame-wise Causal Attention
this section cite: ['b23', 'b23']

Section: Revisiting LoRAT
One-Stream Transformer Tracker. LoRAT [24] follows the one-stream Transformer tracker paradigm, where the template and the search region are processed together by a single ViT backbone. First, the images are divided into non-overlapping patches. These patches are then flattened and mapped to patch embeddings via a linear projection, yielding the template tokens {z i } m i=1 and search region tokens {x j } n j=1 . To distinguish these tokens after concatenation, LoRAT incorporates learnable token type embeddings that are added to the patch embeddings of each stream (template or search). Additionally, it uses shared positional embeddings for both streams. The final input sequence is formed by concatenating the enriched tokens:
X = [ z 1 , . . . , z m , x 1 , . . . , x n ].(1)
A Transformer encoder then applies multi-head self-attention and feed-forward layers over X, capturing intra-frame and inter-frame relationships. Finally, the output embeddings corresponding to the search region are fed into an MLP-based head network for target classification and bounding box regression, producing the tracker's output.
this section cite: ['b23']

Section: Self-Attention Complexity.
LoRAT employs standard full self-attention in its ViT backbone. Let X ∈ R (m+n)×d denote the token embeddings. The standard attention mechanism first computes:
Q = XW Q , K = XW K , V = XW V ,(2)
and then:
Attention(Q, K, V) = softmax QK ⊤ √ d V.(3)
This incurs O((m + n) 2 ) operations. Although LoRAT uses Low-Rank Adaptation (LoRA) [19] to reduce training overhead, the inference cost remains governed by full self-attention.
this section cite: ['b18']

Section: Frame-Wise Causal Attention
Standard full self-attention (Eq. 3) grows quadratically with the number of tokens and does not incorporate temporal order. To efficiently model longer sequences in tracking, we introduce a causal mask that enforces an autoregressive dependency across frames while preserving full attention within each frame.
Assume we have T frames, each producing n t tokens {x t 1 , . . . , x t nt } for t = 1, . . . , T . Concatenate them into a single sequence:
X = x 1 1 ,. . . ,x 1 n1 frame 1 , x 2 1 ,. . . ,x 2 n2 frame 2 , . . . , x T 1 ,. . . ,x T n T frame T .(4)
In frame-wise causal attention, queries from frame t can attend to tokens in frames 1, . . . , t but not to frames > t. We implement this via a causal mask M ∈ R N ×N , where N = T t=1 n t , defined by M (p,q) = 0, if frame(q) ≤ frame(p), -∞, otherwise.
(
Let Q = XW Q , K = XW K , and V = XW V be the usual linear projections. Our frame-wise causal attention modifies Eq. 3 by adding M inside the softmax:
Attention(Q, K, V) = softmax QK ⊤ √ d + M V.(6)
Within each frame, M (p,q) = 0 since frame(p) = frame(q), thus preserving full (unmasked) attention. Across frames, any future-to-past connection is disallowed by setting M (p,q) = -∞ if frame(p) < frame(q), enforcing auto-regressive structure. The visualization of frame-wise causal attention mask is shown in Fig. 3c.
this section cite: []

Section: Key-Value Caching
When performing online tracking or processing long sequences, naively recomputing key/value embeddings for past frames is computationally expensive. Key-Value (KV) caching [33,11,31] provides a simple yet powerful mechanism to amortize this cost over time.
At frame t, we first compute queries for the current frame:
Q t = X t W Q .(7)
Here, X t ∈ R nt×d denotes the tokens of frame t. The key/value pairs for frame t are
K t = X t W K , V t = X t W V .(8)
We then cache all past keys and values (frames 1 through t -1):
K 1:t-1 = K 1 , . . . , K t-1 , V 1:t-1 = V 1 , . . . , V t-1 .(9)
For the frame-wise causal attention at step t, we form
K t = K 1:t-1 , K t , V t = V 1:t-1 , V t .(10)
Hence, the attention computation at frame t becomes
Attention Q t , K t , V t ,(11)
where the mask M (Eq. 5) ensures that Q t can only attend to K 1:t-1 (past frames) and K t (the current frame), but not beyond. Crucially, we never re-encode past frames; we simply reuse their cached K t , V t .
this section cite: ['b32', 'b10', 'b30']

Section: Stream-Specific LoRA Adapters (SSLA)
While frame-wise causal attention preserves within-frame symmetry, the Siamese template and search region streams across frames might still diverge in how they attend to historical context. Specifically, the template in frame t remains mostly static (or slowly updated if multiple templates are used), whereas the search region tokens in frame t accumulate knowledge from frames {1, . . . , t -1}. This discrepancy can degrade matching accuracy if not handled carefully.
We therefore introduce Stream-Specific LoRA Adapters. Each input stream s-template, search 1 , search 2 , and so forth-is assigned its own LoRA offset:
s ∈ {template, search 1 , search 2 , . . . }.(12
) Concretely, for the query projection,
Q s = X s W Q + ∆W Q s , (13
)
where X s is the set of tokens from stream s, W Q is the frozen backbone weight, and ∆W Q s is the learnable low-rank offset for that stream. Analogous offsets ∆W K s , ∆W V s attach to W K , W V . Such design brings the following benefits:
• It preserves a shared representation space via the unmodified backbone.
• It applies minimal, stream-dependent adaptations for each set of tokens.
• It allows easy extension by adding new LoRA modules for additional frames.
Table 1: Benchmarking our tracker on four large-scale challenging datasets. For GOT-10k evaluation, all the methods follow the one-shot protocol, training only on the train split of GOT-10k. Bold indicates the best results and underline indicates the second-best.
this section cite: []

Section: Tracker
LaSOT [16] TNL2K [34] GOT-10k [20] VastTrack [30] SUC PNorm P SUC P AO SR0.5 SR0.75 SUC P TransT256 [8] 64.9 73.8 69.0 50.7 51.7 67.1 76.8 60.9 29.9 25.4 AutoMatch255 [44] 58.3 -59.9 47.2 43.5 65.2 76.6 54.3 28.8 26.6 STARK320 [41] 67.1 77.0 ---68.8 78.1 64.1 33.4 30.8 KeepTrack480 [27] 67.1 77.2 70.2 -------MixFormer384 [9] 70.1 79.9 76.3 -------SBT224 [38] 66.7 -71.1 --70.4 80.8 64.7 --AiATrack320 [17] 69.0 79.4 73.8 --69.6 80.0 63.2 --SimTrack384 [6] 70.5 79.7 -55.6 55.7 69.8 78.8 66.0 34.4 30.3 OSTrack384 [43] 71.1 81.1 77.6 55.9 56.7 73.7 83.2 70.8 33.6 31.5 SwinTrack384 [25] 71.3 -76.5 55.9 57.1 72.4 80.5 67.8 33.0 30.3 DropTrack256 [36] 71.8 81.8 78.1 56.9 57.9 75.9 86.8 72.0 37.0 36.5 SeqTrack-B256 [7] 69.9 79.7 76.3 56.4 -74.7 84.7 71.8 --SeqTrack-L384 [7] 72.5 81.5 79.3 57.8 -74.8 81.9 72.2 39.6 40.2 ARTrack-B256 [35] 70.4 79.5 76.6 59.8 -73.5 82.2 70.9 --ARTrack-L384 [35] 73.1 82.2 80.3 60.3 -78.5 87.4 77.8 35.6 32.4 ARTrackV2-B256 [35] 71.6 80.2 77.2 59.2 -75.9 85.4 72.7 --ARTrackV2-L384 [35] 73.6 82.8 81.1 61.6 -79.5 87.8 79.6 --CiteTracker384 [23] 69.7 78.6 75.7 57.7 59.6 74.7 84.3 73.0 --ROMTrack384 [4] 71.4 81.4 78.2 --74.2 84.3 72.4 37.0 36.1 MixViT-L384 [10] 72.4 82.2 80.1 --75.7 85.3 75.1 39.5 39.8 ODTrack-L384 [45] 74.0 84.2 82.3 61.7 -78.2 87.2 77.3 --LoRAT-B224 [24] 71.7 80.9 77.3 58.8 61.3 72.1 81.8 70.7 38.7 37.8 LoRAT-L378 [24] 75.1 84.1 82.0 62.3 67.0 77.5 86.2 78.1 43.9 45.8 LoRATv2-B224 72.0 81.3 77.9 59.6 62.7 74.7 84.7 72.7 39.1 38.7 LoRATv2-B378 74.3 83.6 80.9 60.9 64.9 75.8 85.7 75.4 40.6 40.8 LoRATv2-L224 74.4 83.8 81.2 61.8 66.7 76.9 86.3 76.4 42.0 43.3 LoRATv2-L378 76.1 85.1 83.1 62.4 67.7 78.2 86.8 79.1 44.2 46.7
this section cite: ['b15', 'b33', 'b19', 'b29', 'b7', 'b43']

Section: Two-Phase Progressive Training
Inspired by curriculum learning [2], we adopt a two-phase approach to gradually introduce multiframe complexity. This strategy not only reduces memory requirements compared to training on multiple frames from scratch but also consistently yields better final accuracy in practice.
Phase 1: Single Template + Single Search Region. We follow LoRAT [24] by training only two LoRA adapters, one for the template and one for a single search region, while keeping the ViT backbone frozen. This phase converges quickly and requires minimal memory.
this section cite: ['b1', 'b23']

Section: Phase 2: Extending to an Additional Search Region.
Next, we fix all previously learned parameters and attach a new LoRA adapter for the second search region. Only this newly added adapter is updated, enabling efficient multi-frame modeling without large-scale retraining. The final model supports both single-frame (for speed) and multi-frame inference (for improved robustness), with additional frames accommodated via further adapters as needed.
this section cite: []

Section: Experiments

this section cite: []

Section: Implementation Details
All models are trained on 4×NVIDIA GeForce RTX 4090 GPUs and evaluated on an NVIDIA GeForce RTX 5090 GPU. Basically, we follow LoRAT [24] for fundamental settings (e.g. training datasets, optimization hyperparameters).
this section cite: ['b23']

Section: Model Variants.
We develop four LoRATv2 variants using ViT-Base (B) and ViT-Large (L) backbones, trained progressively as described in Sec. 3.5. The specific configurations are:
• LoRATv2-B/L-224 (Phase 1 Models): -Backbone: ViT-Base/ViT-Large -Template (z): 224 × 224 -Search Region (x 1 ): 224 × 224 • LoRATv2-B/L-378 (Phase 2 Models): -Backbone: ViT-Base/ViT-Large -Template (z): 224 × 224 -Past Search Region (x 1 ): 224 × 224 -Current Search Region (x 2 ): 378 × 378
Training. We use LaSOT [16], TrackingNet [28], GOT-10k [20] (excluding 1k sequences as in [21]), and COCO [26] for training. For the GOT-10k evaluation, models are trained exclusively on the GOT-10k training split.
Phase 1 (-224 variants): Models are trained for 170 epochs (131,072 iterations/epoch) on (z, x 1 ) pairs. The template (z) and search region x 1 are sampled from the same video (up to a 100-frame gap), with strong crop jitter applied to x 1 . The ViT backbone (DINOv2 pre-trained [29,12]) is frozen; two sets of LoRA modules (rank r = 64), one for the template stream and one for the search region stream, are trained.
Phase 2 (-378 variants): Training continues for an additional 170 epochs on (z, x 1 , x 2 ) triplets. The template z is randomly sampled from a video; x 1 , x 2 are sampled from the same video (up to a 100-frame gap) with strong crop jitter. The backbone and previously trained LoRA modules remain frozen. A new set of extra LoRA modules (rank r = 64) and a corresponding prediction head are introduced exclusively for the x 2 stream.
Inference. During inference, all LoRATv2 variants leverage frame-wise causal attention with KV caching. The initial template is encoded only once, and its key/value embeddings are cached to prevent re-computation for past frames. Subsequent search regions are cropped around the prior bounding box (area factor 4 for x 1 , 5 for x 2 ).
this section cite: ['b15', 'b27', 'b19', 'b20', 'b25', 'b28', 'b11']

Section: Phase 1 (-224 variants):
Tracking is performed using only the x 1 stream.
Phase 2 (-378 variants): By default, predictions are derived from the high-resolution x 2 stream. To maintain a temporal context on the past frames, the Key-Value (KV) caches for the x 1 stream are conditionally updated. This update is triggered when a prediction's classification score exceeds a confidence threshold of 0.9. Upon this condition, a new, smaller search region, with dimensions identical to the Phase 1 input, is cropped around the confident prediction and processed to refresh the x 1 stream's cached embeddings.
this section cite: []

Section: State-of-the-Art Comparison
We compare our LoRATv2 with recent Transformer-based trackers on four challenging benchmarks, following their official evaluation protocols. Tab. 1 reports the results.
LaSOT [16] is a large-scale benchmark containing 280 long-term test videos. From Tab. 1 and Tab. 2, our smallest variant, LoRATv2-B-224, achieves a Success (SUC) of 72.0% at 713 fps. Meanwhile, LoRATv2-L-378 sets a new state of the art at 76.1% SUC, outperforming the previous best LoRAT-L 378 [24] (75.1%).
TNL2K [34] is a recently introduced tracking dataset comprising 700 test videos. Our LoRATv2-L-378 attains 62.4% SUC, improving upon LoRAT-L-378 (62.3%) and confirming the advantage of our temporal modeling capability.
GOT-10k [20] consists of 180 test videos and enforces a strict protocol requiring training exclusively on its designated training split. 1 shows that LoRATv2-L-378 achieves a high AO of 78.2%, approaching ARTrackV2-L (79.5%), thus demonstrating our method's strong generalization when limited to a single dataset.
VastTrack [30] is a large-scale benchmark featuring extensive object categories (2,115 classes) to facilitate the development of more general and robust trackers. As shown in Tab. 1, LoRATv2-L-378 MACs) in both speed and computational load, while also achieving higher accuracy on LaSOT (Tab. 1 and Tab. 2). This trend holds for larger models as well; LoRATv2-L-378 runs at 202 fps with 268G MACs, compared to LoRAT-L-378 at 167 fps with 325G MACs. The improved FPS, despite the multi-frame processing and KV cache management, highlights the effectiveness of our architectural optimizations and frame-wise causal attention in reducing redundant computations.
this section cite: ['b15', 'b23', 'b33', 'b19', 'b29']

Section: Ablation Studies
We conduct comprehensive ablation studies (Tab. 3) on key components of LoRATv2 using the ViT-Base backbone. All results are reported on LaSOT and VastTrack datasets, along with FPS and MACs. For more ablation experiments, please touch the appendix. Frame-Wise Causal Attention (FWCA) and SSLA. Table 3 analyzes the interplay between the attention mechanism (FWCA vs. standard Fully Self-Attention, FSA) and the LoRA configuration (Stream-Specific vs. Shared). The results reveal a strong synergy between FWCA and SSLA. When paired with SSLA, FWCA delivers substantial efficiency gains without compromising-and often improving-accuracy. For instance, our final single-frame model (④, FWCA+SSLA) nearly doubles the speed of its FSA counterpart (②, 713 vs. 452 FPS) while halving the MACs (25 vs. 49G) and achieving superior performance on VastTrack. This advantage becomes more critical in the multi-frame setting, where the FWCA+SSLA model (⑧) surpasses its FSA equivalent (⑥) in both accuracy (+1.1% SUC on LaSOT) and efficiency.
The necessity of SSLA is most evident when combined with FWCA. While a shared LoRA configuration converges for a single search frame (③), it suffers from training instability and fails to converge in the more complex two-frame setup (⑦). In contrast, SSLA provides the necessary adaptability to manage the informational asymmetry introduced by the causal structure, ensuring robust training and optimal performance. Conversely, with FSA, the benefits of SSLA are less clear and can be slightly detrimental (⑥ vs. ⑤), as the symmetric nature of full attention does not require such stream-specific adaptation.
this section cite: []

Section: Two-Phase Progressive Training.
To validate our curriculum-based training strategy, we compare it against a standard "one-phase" approach where the multi-frame tracker is trained from scratch. As detailed in Table 4, our two-phase method is superior in both effectiveness and efficiency. The model trained progressively achieves a LaSOT SUC of 74.3%, significantly outperforming the one-phase model even when the latter is trained for twice as long (71.3%). This result highlights the benefit of mastering a simpler task before progressing to a more complex one. Furthermore, our approach is more resource-conscious, reducing training time by 40% (19 vs. 32 hours) and lowering peak GPU memory consumption compared to the extended one-phase baseline. This analysis confirms that progressive training is not merely a heuristic but a more effective and efficient paradigm for developing complex temporal trackers.
this section cite: []

Section: Conclusion
LoRATv2 significantly advances multi-frame object tracking by introducing frame-wise causal attention with KV caching, Stream-Specific LoRA Adapters, and a progressive two-phase training strategy. These innovations lead to state-of-the-art performance with substantially improved computational efficiency and a superior accuracy-FLOPs trade-off, as demonstrated on multiple benchmarks. LoRATv2 offers a powerful and practical solution for real-time tracking with Transformers.
this section cite: []

Section: References
Ref_id:b0 Title: Artrackv2: Prompting autoregressive tracker where to look and how to describe Year: (2024)
Ref_id:b1 Title: Curriculum learning Year: (2009)
Ref_id:b2 Title: Hiptrack: Visual tracking with historical prompts Year: (2024)
Ref_id:b3 Title: Robust object modeling for visual tracking Year: (2023)
Ref_id:b4 Title: Tctrack: Temporal contexts for aerial tracking Year: (2022)
Ref_id:b5 Title: Backbone is all your need: A simplified architecture for visual object tracking Year: (2022)
Ref_id:b6 Title: SeqTrack: Sequence to sequence learning for visual object tracking Year: (2023)
Ref_id:b7 Title: Transformer tracking Year: (2021)
Ref_id:b8 Title: MixFormer: End-to-end tracking with iterative mixed attention Year: (2022)
Ref_id:b9 Title: MixFormer: End-to-end tracking with iterative mixed attention Year: (2024)
Ref_id:b10 Title: Transformer-xl: Attentive language models beyond a fixed-length context Year: (2019)
Ref_id:b11 Title: Vision transformers need registers Year: (2024)
Ref_id:b12 Title: Attention is not all you need: Pure attention loses rank doubly exponentially with depth Year: (2021)
Ref_id:b13 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b14 Title: Vision-rwkv: Efficient and scalable visual perception with rwkv-like architectures Year: (2025)
Ref_id:b15 Title: LaSOT: A high-quality benchmark for large-scale single object tracking Year: (2019)
Ref_id:b16 Title: AiATrack: Attention in attention for transformer visual tracking Year: (2022)
Ref_id:b17 Title: Target-aware tracking with long-term context attention Year: (2023)
Ref_id:b18 Title: LoRA: Low-rank adaptation of large language models Year: (2022)
Ref_id:b19 Title: Got-10k: A large high-diversity benchmark for generic object tracking in the wild Year: (2021)
Ref_id:b20 Title: A novel performance evaluation methodology for single-target trackers Year: (2016)
Ref_id:b21 Title: 2023a. Causal-vit: Robust vision transformer by causal intervention Year: ()
Ref_id:b22 Title: CiteTracker: Correlating image and text for visual tracking Year: (2023)
Ref_id:b23 Title: Tracking meets lora: Faster training, larger model, stronger performance Year: (2024)
Ref_id:b24 Title: SwinTrack: A simple and strong baseline for transformer tracking Year: (2022)
Ref_id:b25 Title: Microsoft COCO: Common objects in context Year: (2014)
Ref_id:b26 Title: Learning target candidate association to keep track of what not to track Year: (2021)
Ref_id:b27 Title: TrackingNet: A large-scale dataset and benchmark for object tracking in the wild Year: (2018)
Ref_id:b28 Title: DINOv2: Learning robust visual features without supervision Year: (2024)
Ref_id:b29 Title: Vasttrack: Vast category visual object tracking Year: (2024)
Ref_id:b30 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b31 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2025)
Ref_id:b32 Title: Attention is all you need Year: (2017)
Ref_id:b33 Title: Towards more flexible and accurate object tracking with natural language: Algorithms and benchmark Year: (2021)
Ref_id:b34 Title: Autoregressive visual tracking Year: (2023)
Ref_id:b35 Title: DropMAE: Masked autoencoders with spatial-attention dropout for tracking tasks Year: (2023)
Ref_id:b36 Title: Videotrack: Learning to track objects via video transformer Year: (2023)
Ref_id:b37 Title: Correlation-aware deep tracking Year: (2022)
Ref_id:b38 Title: Show-o: One single transformer to unify multimodal understanding and generation Year: (2025)
Ref_id:b39 Title: Autoregressive queries for adaptive tracking with spatio-temporal transformers Year: (2024)
Ref_id:b40 Title: Learning spatio-temporal transformer for visual tracking Year: (2021)
Ref_id:b41 Title: Causal attention for vision-language tasks Year: (2021)
Ref_id:b42 Title: Joint feature learning and relation modeling for tracking: A one-stream framework Year: (2022)
Ref_id:b43 Title: Learn to match: Automatic matching network design for visual tracking Year: (2021)
Ref_id:b44 Title: Odtrack: Online dense temporal token learning for visual tracking Year: (2024)
