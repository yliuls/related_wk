Title: Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models
Abstract: We present a neural network structure, FramePack, to train next-frame (or nextframe-section) prediction models for video generation. FramePack compresses input frame contexts with frame-wise importance so that more frames can be encoded within a fixed context length, with more important frames having longer contexts. The frame importance can be measured using time proximity, feature similarity, or hybrid metrics. The packing method allows for inference with thousands of frames and training with relatively large batch sizes. We also present drift prevention methods to address observation bias (error accumulation), including early-established endpoints, adjusted sampling orders, and discrete history representation. Ablation studies validate the effectiveness of the anti-drifting methods in both single-directional video streaming and bi-directional video generation. Finally, we show that existing video diffusion models can be finetuned with FramePack, and analyze the differences between different packing schedules.

Section: Introduction
Forgetting and drifting are the two most critical problems in next-frame-prediction models for video generation. "Forgetting" refers to the fading of memory as the model struggles to remember earlier content and maintain consistent temporal dependencies. "Drifting" refers to the degradation of visual quality due to error accumulation over time (also called exposure/observation bias).
A fundamental dilemma emerges when attempting to simultaneously address both forgetting and drifting: any method that mitigates forgetting by enhancing memory may also increase error accumulation/propagation, thereby exacerbating drifting; any method that reduces drifting by interrupting error propagation needs to weaken temporal dependencies (e.g., masking or re-noising the history), thus worsening the forgetting. This is a fundamental trade-off that hinders the scalability of next-frame prediction models.
A naive solution to forgetting is to encode more frames. But this approach quickly becomes computationally intractable due to the quadratic attention complexity of transformers (even with optimizations like Flash Attention [10], etc.). Moreover, video frames contain significant temporal redundancy, making naive full-context approaches very inefficient. The substantial duplication of visual features across consecutive frames suggests the potential to design effective compression systems to facilitate memorization.
Drifting is influenced by memorizing mechanisms in several ways. The source of drifting lies in the initial errors that occur in individual frames, while the effect is the propagation and accumulation of these errors across subsequent frames, and the eventual drifting out of the train distribution. A stronger memorization mechanism, on one hand, can lead to better temporal consistency and reduce the occurrence of initial errors. On the other hand, it also memorizes more errors and thus accelerates error propagation when errors do occur. This paradoxical relationship between memory mechanisms and drifting necessitates carefully designed training/sampling methods to facilitate error correction or interrupt error propagation.
In this paper, we propose FramePack as an anti-forgetting memory structure along with anti-drifting sampling and training methods. The FramePack structure addresses the forgetting problem by compressing input frames based on their relative importance, ensuring that the total transformer context length converges to a fixed upper bound. We consider both time-proximity-based and feature-similarity-based importance measures. Afterwards, we propose anti-drifting sampling methods that break the causal prediction chain and incorporate bi-directional contexts by planning single or multiple endpoint frames. We also present an anti-drifting training method to convert frame history into discrete tokens so as to reduce the history disparity between training and inference. We show that these methods effectively reduce the occurrence of errors and prevent their propagation.
We demonstrate that existing pretrained video diffusion models (e.g., HunyuanVideo [28], Wan [48], etc.) can be finetuned with FramePack. Our experiments reveal several findings: because next-frame prediction generates smaller tensor sizes per step compared to full-video generation, it enables more balanced diffusion schedulers with less extreme flow shift timesteps. We also show that the efficient implementations of FramePack can process thousands of frames with 13B models even on laptops (e.g., 6GB or 8GB GPU memory).
2 Related Work
this section cite: ['b9', 'b27', 'b47']

Section: Anti-forgetting and Anti-drifting
The trade-off between forgetting and drifting is also evidenced by previous discussions. CausVid [65] shows that when the video generator is causal, the quality degradation appears at the end of the video and the high-quality part may be subject to an upper bound length. DiffusionForcing [6] discussed that the cause of this drift may be related to error accumulation in models' observation disparity between training and inference. Wang et al. [51] discussed that a model with stronger memory may suffer more from drifting and error accumulation.
Noise scheduling and augmentation in history frames modify noise levels at specific timesteps, video times, or image frequencies to mitigate drifting. These methods generally reduce the dependency on past frames. DiffusionForcing [6] and RollingDiffusion [39] are typical examples. Our ablation studies investigate the influence of adding noise to history frames.
Classifier-Free Guidance (CFG) over history frames applies different masks or noise levels to opposite sides of guidance to amplify the forgetting-drifting trade-off. HistoryGuidance [41] demonstrates this approach. Our ablation studies include guidance-based noise scheduling.
Anchor frames can be used as planning elements for video generation. StreamingT2V [20] and ART-V [54] use reference images as anchors. Video planning approaches [32,73,22,60,3,61] use image or video anchors for content planning.
Compressing latent space can improve the efficiency of video diffusion models. FlexTok [2] adjusts token context length to achieve different levels of visual content compression. LTXVideo [17] shows that a highly compressed latent space can be used for diffusing videos efficiently. PyramidFlow [25] diffuses video latents in a pyramid and re-noises downsampled latents in that pyramid to reduce computation costs. FAR [16] proposes a multi-level causal attention structure to establish long-shortterm causal context pacifying and KV caches. HiTVideo [76] uses hierarchical tokenizers to enhance the video generation with autoregressive language models. Memory in world models often involves different modeling of long-term memory. Typical examples are 3D geometry like mesh and proxy [38,50,67,66]. Training diffusion models with domain data can also bake the memory into the model, with full model training [1,46] or low-rank methods [21]. Retrieval-based memory mechanisms like WorldMem [58] are efficient when the task prioritizes reconstructing history contents.
this section cite: ['b64', 'b5', 'b50', 'b5', 'b38', 'b40', 'b19', 'b53', 'b31', 'b72', 'b21', 'b59', 'b2', 'b60', 'b1', 'b16', 'b24', 'b15', 'b75', 'b37', 'b49', 'b66', 'b65', 'b0', 'b45', 'b20', 'b57']

Section: Long Video Generation
Extending video generation beyond short clips remains an open problem. LVDM [19] generates long videos using latent diffusion, while Phenaki [47] creates variable-length videos from sequences of text prompts. Gen-L-Video [49] applies temporal co-denoising for multi-text conditioned videos, and FreeNoise [37] extends pretrained models without additional training via noise rescheduling. NUWA-XL [62] implements a Diffusion-over-Diffusion architecture with coarse-to-fine processing, while Video-Infinity [44] overcomes computational constraints through distributed generation. StreamingT2V [20] produces consistent, dynamic, and extendable videos without hard cuts, and CausVid [65] transforms bidirectional models into fast autoregressive models through distillation. Recent advances include GPT-like architecture (ViD-GPT [15]), multi-event generation (MEVG [36]), attention control for multi-prompt generation (DiTCtrl [5]), precise temporal control (MinT [55]), history-based guidance (HistoryGuidance [41]), unified next-token and full-sequence diffusion (DiffusionForcing [6]), SpectralBlend temporal attention (FreeLong [33]), video autoregressive modeling (FAR [16]), and test-time training (TTT [9]). Harvey et al. [18] proposes a flexible approach for modeling long contexts with dilatation (Hierarchy-2) and propagation. Generating longer videos often requires efficient architectures, e.g., linear attention [4, 59, 52, 8, 68, 26], sparse attention [56,71,72,57], low-bit computation [30,74,29], low-bit attention [70,69], hidden state caching [35,31], distillation [42,34,63,64], etc.
this section cite: ['b18', 'b46', 'b48', 'b36', 'b61', 'b43', 'b19', 'b64', 'b14', 'b35', 'b4', 'b54', 'b40', 'b5', 'b32', 'b15', 'b8', 'b17', 'b55', 'b70', 'b71', 'b56', 'b29', 'b73', 'b28', 'b69', 'b68', 'b34', 'b30', 'b41', 'b33', 'b62', 'b63']

Section: Packing Frame Context
We consider a video generation model that predicts next frames repeatedly to form a video. For simplicity, we consider next-frame-section prediction models using Diffusion Transformers (DiTs) that generate a section X of S unknown frames so that X ∈ R S×h×w×c , conditioned on a section F of T input frames so that F ∈ R T ×h×w×c . All definitions of frames and pixels refer to latent representations, as most modern models operate in latent space.
For next-frame (or next-frame-section) prediction, S is typically 1 (or a small number). We focus on the challenging case where T ≫ S. With per-frame context length L f (typically L f ≈ 1560 for each 480p frame in Hunyuan/Wan/Flux), the vanilla DiT yields total context length L = L f (T + S). This causes a context length explosion when T is large. We observe that the input frames have different importance when predicting the next frame, and we can prioritize them according to their importance.
this section cite: []

Section: Time Proximity Based Packing
We first consider a baseline case where the temporal proximity reflects frame importance (Fig. 1-(a)). More advanced cases involving similarity-based importance are covered in §3.2. With frames temporally closer to the prediction target being more relevant, we enumerate all frames with F 0 being the most important (e.g., the most recent) and F T -1 being the least (e.g., the oldest). We define a length function ϕ(F i ) that determines each frame's context length after VAE encoding and transformer patchifying by applying progressive compression
ϕ(F i ) = L f λ i ,(1)
where λ > 1 is a compression parameter. The frame-wise compression is achieved by manipulating the transformer's patchify kernel size in the input layer (e.g., λ = 2, i = 5 means a kernel size where the product of all dims equals 2 5 = 32 like the 3D kernel 2 × 4 × 4, or 8 × 2 × 2, etc.). The total context length then follows a geometric progression
L = S • L f + L f • T -1 i=0 1 λ i = S • L f + L f • 1 -1/λ T 1 -1/λ ,(2)
and when T → ∞, the total context length converges to lim T →∞ L = (S + λ λ-1 ). This bounded context length makes FramePack's compression bottleneck invariant to the input frame number T .
Since most hardware supports efficient matrix processing by powers of 2, we mainly discuss the case of λ = 2 in this paper. Note that we can represent arbitrary compression rates by duplicating (or dropping) several specific terms in the power-of-2 sequence: considering the accumulation
+∞ i=0 1 2 i = 2
2-1 = 2, if we want to loosen it a bit, for example to 2.625, we can duplicate the terms
1 2 and 1 8 so that 1 1 + 1 2 + ( 1 2 ) + 1 4 + 1 8 + ( 1 8 ) + 1 16 + ... = 1 2 + 1 8 + +∞ i=0 1 2 i = 2.625. Following this, one can cover arbitrary rates by converting the rate value to binary bits and then translating every bit. We present frame packing methods using time proximity or feature similarity. We discuss several typical kernel structures. This list does not necessarily cover all popular variants, and more structures can be developed in a similar way.
Packing schedules The patchifying operations in most DiTs are 3D, and we denote the 3D kernel as (p f , p h , p w ) representing the steps in frame number, height, and width. A same compression rate can be achieved by multiple possible kernel sizes, e.g., the compression rate of 64 can be achieved by (1,8,8), (4, 4, 4), (16, 2, 2), (64, 1, 1), etc. Compression levels can be duplicated and combined with higher compression rates. We discuss more packing ways in Fig. 1-(c): duplication allows for same kernel sizes in frame width and height, making the compression more compact; temporal kernel can compress contiguous frames into a single tensor; symmetric progression treats both beginning and ending frames as equally important.
Independent patchifying parameters Empirical evidence shows that using independent parameters for the different input projections at multiple compression rates facilitates stabilized learning. We assign the most commonly used input compression kernels as independent neural network layers: (2, 4, 4), (4,8,8), and (8, 16, 16). For higher compressions (e.g., at (16, 32, 32)), we first downsample (e.g., with 2 × 2 × 2) and then use the largest kernel (8,16,16). We initialize their separated weights by interpolating from the pretrained patchifying projection (e.g., the (2, 4, 4) projection of HunyuanVideo/Wan).
Tail options While in theory FramePack can process videos of arbitrary length with a fixed, invariant context length, frames may fall below a minimum unit size (e.g., a single latent pixel) when the input length becomes extremely large. We discuss 3 options to process the tail frames: (1) simply delete the tail; (2) allow each tail frame to increase the context length by a single latent pixel; (3) apply global average pooling to all tail frames and process them with the last kernel. In our tests, the visual differences between these options are relatively negligible.
this section cite: ['b0', 'b7', 'b7', 'b3', 'b7', 'b7', 'b7', 'b15', 'b15']

Section: RoPE alignment
When encoding inputs with different compression kernels, the different context lengths require RoPE (Rotary Position Embedding) [43] alignment. RoPE generates complex numbers with real and imaginary parts for each token position across all channels, which we refer to as "phase". We directly downsample (using average pooling) such phases to match the compression kernels.
this section cite: ['b42']

Section: Feature Similarity Based Packing and Hybrid Approach
The aforementioned frame ordering F 0...T -1 can be seen as a result of sorting all history frames using their time positions. We note that such sorting can also use other metrics like feature similarity (Fig. 1-(b)). For instance, consider a typical cosine similarity
sim cos (F i , X) = p (F i ) p • X⊤ p ∥(F i ) p ∥∥ Xp ∥ ,(3)
where the sum is taken over pixels p. This measures the cosine similarity between each history frame and the estimated next frame section. Sorting the history frames using sim cos (•) will produce a We present sampling approaches to generate frames in different temporal orders. The shadowed squares are the generated frames in each iteration, whereas the white squares are the iteration inputs. We also discuss the method to convert the frame history into a discrete representation.
permutation F 0...T -1 with F 0 being the most similar and F T -1 being the least. This permutation can directly replace the aforementioned time proximity. Since similarity-based permutation may change abruptly when processing contiguous frames, we also consider a smooth time proximity modeling
sim time (F i , X) = e -(time(Fi)-time( X)) 2 ,(4)
where time(•) gets the frames' starting time measured in seconds. Consider the weighting
sim hybrid (F i , X) = sim cos (F i , X) + λ time sim time (F i , X) ,(5)
where λ time is a weighting parameter. Sorting the history frames using sim hybrid (•) will produce a permutation that transits relatively smoothly as the generating window moves forward in time. This hybrid approach is suitable for world model datasets (mainly video games) that require returning to previously visited views of scenes or events. Similar sorting can also be applied to facial identity metrics to facilitate movie generation applications that emphasize consistent human actors. This method will be evaluated in ablation experiments in the supplementary materials.
this section cite: []

Section: Drift Prevention
Drifting is a common problem in next-frame prediction models where visual quality degrades as video length increases. We discuss anti-drifting approaches by adjusting the sampling processes and history representations as shown in Fig. 2.
this section cite: []

Section: Planned Endpoints and Adjusted Sampling Order
One possible explanation of drifting is that the modeling of the chained conditional probability P(X t |X t-1 ) fails to approximate P(X t ) due to imperfect estimations. This perspective indicates that a strict causal system is more vulnerable to drifting than bi-directional models that directly approximate P(X t |X t1 , X t2 ) where t 1 < t < t 2 .
Endpoint planning The vanilla sampling method shown in Fig. 2-(a) can be modified into Fig. 2-(b), where the first iteration simultaneously generates both beginning and ending sections, while subsequent iterations fill the gaps between these anchors. This simple method can get rid of drifting in specific cases when the video motions are in a relatively small range, or when the motion patterns are repeated or periodic (e.g., dancing, talking, spinning, etc.), or when the motion content follows some texture patterns (e.g., fire flame, water flow, etc.).
this section cite: []

Section: Inverted sampling (image-to-video)
A variant by inverting the sampling order in Fig. 2-(b) into Fig. 2-(c) is effective for image-to-video generation. In image-to-video, the first frame is a groundtruth user input, whereas the last frame is a generated endpoint that is not guaranteed to perfectly preserve the quality of the user input. All generations in Fig. 2-(c) keep the direction to approximate the high-quality user input, leading to iteratively refined generations.
this section cite: []

Section: Multiple endpoints
The endpoint planning can be repeated with different prompts before filling in the gaps, resulting in a planned sequence of generation (Fig. 2-(d)). Note that though the drifting might still happen in the endpoint-wise, in certain cases, when the prompted sections are distant enough, the error accumulation becomes almost negligible. This method is more flexible than single endpoint planning and supports more dynamic motions and more complicated storytelling.
RoPE with random access These sampling methods require modifications to RoPE to support non-consecutive phases (time indices of frames). This is achieved by skipping the blank phases (indices) in the time dimension.
this section cite: []

Section: History Discretization
Another potential cause of drifting is the difference between training and inference distributions over the history frames. This indicates that drifting can be mitigated if the history representation is not sensitive enough to distinguish between the training and inference frames. Discrete integer tokens are well-suited to reduce the mode gap between training and inference distributions. This perspective is supported by empirical evidence that discrete autoregressive systems (e.g., LLMs) often demonstrate less obvious drifting than continuous autoregressive systems for visual content diffusion.
We discretize the history as in Fig. 2-(e). Consider a dataset Φ ∈ R (B×T ×H×W )×C of precomputed latent videos, a K-Mean over all latent pixels will yield a codebook Ω ∈ R K×C where K ∈ Z + is an adjustable number. Any latent frame F ∈ R T ×H×W ×C can be quantized by Q(•) with
Q(F ) p = arg min k ||F p -Ω k || 2 ,(6)
where p is pixel position, and Q(F ) ∈ [0, K -1] T ×H×W is a matrix of indices over the codebook Ω. The matrix of indices can be converted back to latent videos by Ω Q(F ) ∈ R T ×H×W ×C . We replace all history frames F with Ω Q(F ) during training.
Intuitively, when K = 1, the history becomes meaningless as one single color, and the drifting is eliminated at the cost of giving up memory (errors do not accumulate but sections become unrelated); when K → ∞, the effect is equivalent to no discretization, and the drifting remains. We show with ablation study that a suitable K can minimize the error propagation while simultaneously producing plausible consistency between sections.
this section cite: []

Section: Experiments

this section cite: []

Section: Ablative Naming
To simplify the presentation of the experiments, we use a common naming convention for all ablative structures. A FramePack name is represented as a string such as td_f16k4f4k2f1k1_g9_x_f1k1.
We explain the meaning of this notation:
Kernel: A kernel name is like k1h2w2. The k stands for "kernel", and k1h2w2 indicates a patchify kernel with shape (1, 2, 2), where the temporal size is 1, the height is 2, and the width is 2.
Kernel (simplified): For simplicity, since kernels that are multiples of (1, 2, 2) are commonly used, we use abbreviated notation such as k1 that only denotes the temporal dimension. Specifically, k1 represents k1h2w2 (the kernel (1, 2, 2)), k2 represents k2h4w4 (the kernel (2, 4, 4)), etc.
Encoding frames: The notation f16k4 indicates that 16 frames are encoded by the kernel k4 (the simplified k4h8w8) with kernel size (4, 8, 8).
Packing: The notation f16k4f2k2f1k1 shows a way to encode 19 contiguous frames: the first 16 frames are encoded by the kernel k4 (the kernel (4, 8, 8)), the next 2 frames are encoded by the kernel k2 (the kernel (2, 4, 4)), and the last 1 frame is encoded by the kernel k1 (the kernel (1, 2, 2)).
Tail: We append the notation with td, ta, or tc to indicate the tail frames before or after packing, such as td_f16k4f4k2f1k1. The three options are as discussed in Section 3.1. Herein, the "delete" option td deletes the tail. The "append" option ta compresses each tail frame by performing a 3D pooling of (1, 32, 32) and then encodes with the nearest kernel, and the "compress" option tc uses global average pooling for all tail frames and compresses them with the nearest kernel.
this section cite: []

Section: Skipping:
The notation x skips an arbitrary number of frames (including 0 frames).
this section cite: []

Section: History Discretization:
The notation +d means converting history into discrete space.
this section cite: []

Section: Generating:
The notation g9 means generating 9 frames.
With the above naming convention, we can represent all ablative structures in a compact form. Note that this naming also implies the sampling approach as discussed in Section 4.1:
td_f16k4f4k2f1k1_g9: The vanilla sampling that generates frames in temporal order.
td_f16k4f4k2f1k1_g9+D: The vanilla sampling with history discretization.
td_f16k4f4k2f1k1_g9_x_f1k1: The anti-drifting sampling with an endpoint frame.
f1k1_x_g9_f1k1f4k2f16k4_td: The inverted anti-drifting sampling in inverted temporal order.
this section cite: []

Section: Base Model and Implementation Details
We implement FramePack with Wan and HunyuanVideo. We implement both the text-to-video and image-to-video structures, though both are naturally supported by next-frame-section prediction models and do not need architecture modifications. We report results with HunyuanVideo in the main paper (see also supplementary for Wan results). We conduct all experiments using H100 GPU clusters with training details in the supplementary. Note that FramePack achieves a batch size of about 64 on a single 8×A100-80G node with the 13B HunyuanVideo model at 480p resolution LoRA training with window size 2 or 3 (or batch size 32 of window size 4 or 5), making FramePack suitable for personal or laboratory-scale training and experimentation. We follow the guidelines of LTXVideo [17]'s dataset collection pipeline to gather data at multiple resolutions and quality levels (see also supplementary for more details).
this section cite: ['b16']

Section: Quantitative Evaluation
We discuss the metrics for evaluating ablative architectures. The tested inputs consist of 512 real user prompts for text-to-video and 512 image-prompt pairs for image-to-video tasks. All test samples were curated from real users to ensure diversity and real-world applicability. For quantitative tests, we by default use 30 seconds for long videos and 5 seconds for short videos.
Metrics Multiple metrics for video evaluations are consistent with common benchmarks, e.g., VBench [24], VBench2 [75], etc. Clarity: The MUSIQ [27] image quality predictor trained on SPAQ [14]. This metric measures artifacts like noise and blurring. Aesthetic: The LAION aesthetic predictor [40]. This metric measures the aesthetic values perceived by a CLIP-based estimator.
this section cite: ['b23', 'b74', 'b26', 'b13', 'b39']

Section: Motion:
The video frame interpolation model [23] modified by VBench to measure the smoothness of motion. Dynamic: The RAFT [45] modified by VBench to estimate the degree of dynamics. Note that the "dynamic" metric and "motion" metric represent a trade-off, e.g., a still image may rank high on motion smoothness but will be penalized by low dynamic degrees. Semantic: The video-text score computed by ViCLIP [53]. This metric measures the overall semantic consistency between the generated video and the prompt. Anatomy: The ViT [13] pretrained by VBench for identifying the per-frame presence of hands, faces, bodies, etc. Identity: The facial feature similarity using ArcFace [11] with face detection by RetinaFace [12].
this section cite: ['b22', 'b44', 'b52', 'b12', 'b10', 'b11']

Section: Drifting measurement
We observe that when drifting occurs, a significant difference emerges between the beginning and ending portions of a video across various quality metrics. We define the start-end contrast ∆ M drift for an arbitrary quality metric M as:
∆ M drift (V ) = |M (V start ) -M (V end )| , (7
)
where V is the tested video, V start represents the first 15% of frames, and V end represents the last 15% of frames. This start-end contrast can be applied to different metrics M (e.g., motion score, image quality, etc.). The magnitude of ∆ M drift (V ) directly indicates the severity of drifting. Since video models may generate frames in different temporal orders (either forward or backward), we use the absolute difference to ensure our metric remains direction-agnostic.
this section cite: []

Section: Human assessments
We collect human preferences from A/B tests. Each ablative architecture yields 100 results. The A/B tests are randomly distributed among ablations, and we ensure that each ablation covers at least 100 assessments. We report ELO-K32 score and the relative ranking.
this section cite: []

Section: Ablative results
As shown in Table 1, we note several discoveries. (1) The inverted anti-drifting sampling method achieves the best results in 4 out of 7 metrics, and achieves the best performance in all drifting metrics. (2) However, the inverted anti-drifting sampling has a relatively small dynamic range. (3) While vanilla sampling achieved the highest dynamic score, this is likely attributable to drifting effects rather than genuine quality, evidenced by relatively low ELO scores. (4) The vanilla sampling with discrete history achieves highly competitive human scores while having a much larger dynamic range. (5) We also observe that differences between specific configuration options within the same sampling approach are relatively small and random, suggesting that the overall architecture contributes more to the general difference.
this section cite: []

Section: History discretization parameter
The history discretization is influenced by the parameter K with higher K giving stronger anti-drifting effects but also more challenging learning for smooth transitions between sections. In our tests, K = 128 gives strong drift reduction with relatively minimal training difficulties. We provide more detailed ablations for K in the supplementary materials.
this section cite: []

Section: Additional evaluations
We test feature-similarity-based packing with video game (world model) benchmarks in the supplementary material. See also supplementary material for Wan results and more details of decomposed metrics.
this section cite: []

Section: Comparison to Alternative Architectures
We discuss several relevant alternatives to generate videos in various ways. The involved methods either enable longer video generation, reduce computational bottlenecks, or both. To be specific, we implement these variants on top of HunyuanVideo default architecture (33 latent frames) using a simple naive sliding window with half context length for history inputs.
Repeating image-to-video: Directly repeat the image-to-video inference to make longer videos.
Anchor frames: Use an image as the anchor frame to avoid drifting. We implement a structure that resembles StreamingT2V [20].
Causal attention: Finetune full attention into causal attention for easier KV cache and faster inference. We implement a structure that resembles CausVid [65].
DiffusionForcing: We conduct a detailed ablation study with DiffusionForcing [6]. We focus on the history noise scheduling using the same scheduling as SkyreelV2 [7] that multiplies the diffusion timestep on history frames with σ train during training and σ test in inference to delay the denoising on history latents. Intuitively, σ test = 0 is equivalent to clean latents for history (no noise added to the history). We consider these ablations: (1) σ train being random with σ test = 0.1; (2) σ train being random with σ test = 0.5; (3) σ train being random with σ test = 0; (4) σ train = 0.1 and σ test = 0.1. Usually higher σ test reduces the reliance on the history, which is beneficial for interrupting error accumulation, thus mitigates drifting, but at the cost of aggravating forgetting.
History guidance: Delay the denoising timestep on history latents but also put the completely noised history on the unconditional side of CFG guidance. This will speed up error accumulation, thus aggravating drifting, but also enhance memory to mitigate forgetting. We implement a structure that resembles HistoryGuidance [41].
As shown in Table 2, we observe several findings. (1) The inverted anti-drifting sampling achieves the best results across all drifting metrics, while having a relatively small dynamic range. (2) The vanilla sampling with discrete history is very competitive in drifting measurements, while having a relatively larger dynamic range. (3) Human perception prefers the two proposed candidates as evidenced by the ELO score. (4) See also the supplementary material for more detailed explanations, analysis, and comparisons with DiffusionForcing candidates.
this section cite: ['b19', 'b64', 'b5', 'b6', 'b40']

Section: Conclusion
In this paper, we presented FramePack, a neural network structure that aims to address the forgettingdrifting dilemma in next-frame prediction models for video generation. FramePack applies progressive compression to input frames based on their importance, ensuring the context length converges to a fixed upper bound. We discussed both time-proximity-based packing and feature-similarity-based packing. We also discussed the anti-drifting training methods with history discretization, and the anti-drifting sampling methods using bi-directional context planning and scheduling. Experiments suggest that FramePack can process a large number of frames, improve model responsiveness, and allow for higher batch sizes in training. The approach is compatible with existing video diffusion models and supports various compression variants that can be optimized for wider applications.
this section cite: []

Section: References
Ref_id:b0 Title: Diffusion for world modeling: Visual details matter in atari Year: ()
Ref_id:b1 Title: FlexTok: Resampling images into 1d token sequences of flexible length Year: (2025)
Ref_id:b2 Title: Talc: Time-aligned captions for multi-scene text-to-video generation Year: (2024)
Ref_id:b3 Title: Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction Year: (2023)
Ref_id:b4 Title: Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation Year: (2024)
Ref_id:b5 Title: Diffusion forcing: Next-token prediction meets full-sequence diffusion Year: (2025)
Ref_id:b6 Title: Skyreels-v2: Infinite-length film generative model Year: (2025)
Ref_id:b7 Title: Rethinking attention with performers Year: (2020)
Ref_id:b8 Title: One-minute video generation with test-time training Year: (2025)
Ref_id:b9 Title: Flashattention: Fast and memory-efficient exact attention with io-awareness Year: (2022)
Ref_id:b10 Title: Arcface: Additive angular margin loss for deep face recognition Year: (2019)
Ref_id:b11 Title: Retinaface: Single-stage dense face localisation in the wild Year: (2019)
Ref_id:b12 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b13 Title: Perceptual quality assessment of smartphone photography Year: (2020)
Ref_id:b14 Title: Vid-gpt: Introducing gpt-style autoregressive generation in video diffusion models Year: (2024)
Ref_id:b15 Title: Long-context autoregressive video modeling with next-frame prediction Year: (2025)
Ref_id:b16 Title: Ltx-video: Realtime video latent diffusion Year: (2024)
Ref_id:b17 Title: Flexible diffusion modeling of long videos Year: (2022)
Ref_id:b18 Title: Latent video diffusion models for high-fidelity long video generation Year: (2022)
Ref_id:b19 Title: Streamingt2v: Consistent, dynamic, and extendable long video generation from text Year: (2024)
Ref_id:b20 Title: Slowfast-vgen: Slow-fast learning for action-driven long video generation Year: (2024)
Ref_id:b21 Title: Storyagent: Customized storytelling video generation via multi-agent collaboration Year: (2024)
Ref_id:b22 Title: Real-time intermediate flow estimation for video frame interpolation Year: ()
Ref_id:b23 Title: VBench: Comprehensive benchmark suite for video generative models Year: (2024)
Ref_id:b24 Title: Pyramidal flow matching for efficient video generative modeling Year: (2024)
Ref_id:b25 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b26 Title: Musiq: Multi-scale image quality transformer Year: (2021)
Ref_id:b27 Title: A systematic framework for large video generative models Year: (2024)
Ref_id:b28 Title: Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models Year: (2025)
Ref_id:b29 Title: Q-diffusion: Quantizing diffusion models Year: (2023)
Ref_id:b30 Title: Timestep embedding tells: It's time to cache for video diffusion model Year: (2024)
Ref_id:b31 Title: Videostudio: Generating consistent-content and multi-scene videos Year: (2024)
Ref_id:b32 Title: Freelong: Training-free long video generation with spectralblend temporal attention Year: (2025)
Ref_id:b33 Title: Latent consistency models: Synthesizing high-resolution images with few-step inference Year: (2023)
Ref_id:b34 Title: Fastercache: Training-free video diffusion model acceleration with high quality Year: (2024)
Ref_id:b35 Title: Mevg: Multi-event video generation with text-to-video models Year: (2024)
Ref_id:b36 Title: Freenoise: Tuning-free longer video diffusion via noise rescheduling Year: (2023)
Ref_id:b37 Title: d-informed world-consistent video generation with precise camera control Year: (2025)
Ref_id:b38 Title: Rolling diffusion models Year: (2024)
Ref_id:b39 Title: Laion-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b40 Title: History-guided video diffusion Year: (2025)
Ref_id:b41 Title: Consistency models Year: (2023)
Ref_id:b42 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b43 Title: Video-infinity: Distributed long video generation Year: (2024)
Ref_id:b44 Title: Raft: Recurrent all-pairs field transforms for optical flow Year: (2020)
Ref_id:b45 Title: Diffusion models are real-time game engines Year: (2024)
Ref_id:b46 Title: Phenaki: Variable length video generation from open domain textual description Year: (2022)
Ref_id:b47 Title: Wan: Open and advanced large-scale video generative models Year: (2025)
Ref_id:b48 Title: Gen-l-video: Multi-text to long video generation via temporal co-denoising Year: (2023)
Ref_id:b49 Title: 3d reconstruction with spatial memory Year: (2024)
Ref_id:b50 Title: Error analyses of auto-regressive video diffusion models: A unified framework Year: (2025)
Ref_id:b51 Title: Linformer: Self-attention with linear complexity Year: (2020)
Ref_id:b52 Title: Internvid: A large-scale video-text dataset for multimodal understanding and generation Year: (2023)
Ref_id:b53 Title: Art•v: Auto-regressive text-to-video generation with diffusion models Year: (2023)
Ref_id:b54 Title: Mind the time: Temporally-controlled multi-event video generation Year: (2024)
Ref_id:b55 Title: Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity Year: (2025)
Ref_id:b56 Title: Training-free and adaptive sparse attention for efficient long video generation Year: (2025)
Ref_id:b57 Title: Longterm consistent world simulation with memory Year: (2025)
Ref_id:b58 Title: Efficient high-resolution image synthesis with linear diffusion transformers Year: (2024)
Ref_id:b59 Title: Dreamfactory: Pioneering multiscene long video generation with a multi-agent framework Year: (2024)
Ref_id:b60 Title: Synchronized video storytelling: Generating video narrations with structured storyline Year: (2024)
Ref_id:b61 Title: Nuwa-xl: Diffusion over diffusion for extremely long video generation Year: (2023)
Ref_id:b62 Title: Improved distribution matching distillation for fast image synthesis Year: (2024)
Ref_id:b63 Title: One-step diffusion with distribution matching distillation Year: (2024)
Ref_id:b64 Title: From slow bidirectional to fast causal video generators Year: (2024)
Ref_id:b65 Title: Wonderjourney: Going from anywhere to everywhere Year: (2024)
Ref_id:b66 Title: Wonderworld: Interactive 3d scene generation from a single image Year: (2025)
Ref_id:b67 Title: Metaformer is actually what you need for vision Year: (2022)
Ref_id:b68 Title: Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization Year: (2024)
Ref_id:b69 Title: Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration Year: ()
Ref_id:b70 Title: Spargeattn: Accurate sparse attention accelerating any model inference Year: (2025)
Ref_id:b71 Title: Fast video generation with sliding tile attention Year: (2025)
Ref_id:b72 Title: Moviedreamer: Hierarchical generation for coherent long visual sequence Year: (2024)
Ref_id:b73 Title: Vidit-q: Efficient and accurate quantization of diffusion transformers for image and video generation Year: (2024)
Ref_id:b74 Title: VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness Year: (2025)
Ref_id:b75 Title: Hitvideo: Hierarchical tokenizers for enhancing text-to-video generation with autoregressive large language models Year: (2025)
