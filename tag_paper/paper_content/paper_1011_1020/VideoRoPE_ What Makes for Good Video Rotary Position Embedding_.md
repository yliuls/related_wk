Title: VideoRoPE: What Makes for Good Video Rotary Position Embedding?
Abstract: While Rotary Position Embedding (RoPE) and its variants are widely adopted for their long-context capabilities, the extension of the 1D RoPE to video, with its complex spatio-temporal structure, remains an open challenge. This work first introduces a comprehensive analysis that identifies four key characteristics essential for the effective adaptation of RoPE to video, which have not been fully considered in prior work. As part of our analysis, we introduce a challenging V-NIAH-D (Visual Needle-In-A-Haystack with Distractors) task, which adds periodic distractors into V-NIAH. The V-NIAH-D task demonstrates that previous RoPE variants, lacking appropriate temporal dimension allocation, are easily misled by distractors. Based on our analysis, we introduce VideoRoPE, with a 3D structure designed to preserve spatio-temporal relationships. VideoRoPE features low-frequency temporal allocation to mitigate periodic oscillations, a diagonal layout to maintain spatial symmetry, and adjustable temporal spacing to decouple temporal and spatial indexing. Vide-oRoPE consistently surpasses previous RoPE variants, across diverse downstream tasks such as long video retrieval, video understanding, and video hallucination. Our code is available at https://github.com/Wiselnn570/VideoRoPE.

Section: Introduction
Rotary Position Embedding (RoPE) (Su et al., 2024) helps Transformer models understand word order by assigning each token a unique positional 'marker' calculated using a mathematical rotation matrix. RoPE has advantages in long-
Table 1. Comparison between different RoPE variants for Video Large Language Models (Video LLMs). VideoMME MLVU V-NIAH-D V-NIAH VideoHallucer VideoHallucer (Temporal) VideoHallucer (Spatial) LongVideoBench VideoRoPE(Ours) M-RoPE Vanilla RoPE TAD-RoPE 61.33 57.26 65.56 87.11 91.11 46.20 58.50 57.00 Figure 1. VideoRoPE outperforms RoPE variants on benchmarks.
context understanding (Ding et al., 2024b), and continues to be a default choice in leading Large Language Models (LLMs) like the LLaMA (Touvron et al., 2023a;b;Dubey et al., 2024) and QWen (Yang et al., 2024a;b) series.
The original RoPE implementation (Vanilla RoPE) (Su et al., 2024) is designed for sequential 1D data like text. However, recent Video Large Language Models (Video LLMs) (Li et al., 2023;Lin et al., 2023a;Chen et al., 2024a;Maaz et al., 2024b;Zhang et al., 2024d;Wang et al., 2024c;Chen et al., 2024b;Zhang et al., 2024b) process video, which has a more complex spatio and temporal structure. As shown in Tab. 1, although several RoPE-based approaches (Gao et al., 2024;Wang et al., 2024a) have been proposed to support video inputs, these variants exhibit limitations and do not fully satisfy the following key characteristics:
(1) 2D/3D Structure. Some existing Video LLMs direct flatten the video frame into 1D embeddings and apply the 1D structure RoPE (Su et al., 2024;Gao et al., 2024). These solutions fail to capture video data's inherent 2D or 3D (temporal (t), horizontal (x), and vertical (y)) structure, thus hindering explicit spatial and temporal representation.
this section cite: ['b52', 'b13', 'b52', 'b29', 'b16', 'b52', 'b16']

Section: V-NIAH V-NIAH-D (Ours)
... Video RoPE (Ours) (a) (b) Needle Distractor Distractor (Visual-Needle-In-A-Haystack) Question: Find the frame of a couple in a wedding, what is the color of a balloon on the bridegroom's head? Answer: Yellow Figure 2. Left: To demonstrate the importance of frequential allocation, based on VIAH (a) we present a more challenging V-NIAH-D task (b) that similar images are inserted as distractors. Right: Compared to M-RoPE, our VideoRoPE is more robust in retrieval and is less affected by distractors. See Fig. 7 in the Experiments section for details on the horizontal and vertical axes.
(2) Frequency Allocation. Previous approaches such as M-RoPE used in QWen2-VL (Wang et al., 2024a) employ 3D structure, dividing feature dimensions into distinct subsets for (t, x, y) encoding, respectively. How to determine the optimal allocation of these dimension subsets, and their associated frequenciesfoot_4 are not well studied. Some previous work allocates the lower dimensions corresponding to the high frequency to represent the t. However, the temporal dimension t is significantly tortured by periodic oscillation, and distant positions may have the same embeddings.
We present a simple setting to verify this point. Based on the previous long-video retrieval task V-NIAH (Visual Needle-In-A-Haystack) (Zhang et al., 2024d), we insert several similar images that do not affect the question's answer before and after the needle image as distractor (Hsieh et al., 2024;Yuan et al., 2024), forming a new task, V-NIAH-D (Visual Needle-In-A-Haystack with Distractors). As shown in Fig. 2, we find that previous M-RoPE is misled by distractors, showing a significant performance decline from V-NIAH to V-NIAH-D. Our observation demonstrates that the periodic oscillation reduces Video LLMs' robustness.
(3) Spatial Symmetry. The distance between the end of the precedent textual input and the start of visual input equals the distance between the end of visual input and the start of subsequent textual input (Su, 2024b). Such a symmetry ensures that the visual input receives equal contextual influence from both the preceding and subsequent textual information.
(4) Temporal Index Scaling. Spatial and temporal dimen-sions often exhibit different granularities (e.g., a unit change in x/y differs from a unit change in t) (Gao et al., 2024).
Employing varying index intervals in positional encoding allows for dimension-specific encoding, capturing diverse scales and enhancing efficiency.
Driven by our analysis, we present a new video position embedding strategy, VideoRoPE, which can simultaneously satisfy the four properties in Tab. 1. Specifically, we use a 3D structure to model spatiotemporal information, allocating higher dimensions (lower frequencies), to the temporal axis (Low-frequency Temporal Allocation, LTA) to prioritize temporal modeling. The right panel of Fig. 2 demonstrates that our LTA allocation mitigates oscillations and exhibits robustness to distractors in the V-NIAH-D task. We further employ a Diagonal Layout (DL) design to ensure spatial symmetry and preserve the relative positioning between visual and text tokens. Regarding temporal index scaling, we propose Adjustable Temporal Spacing (ATS), where a hyper-parameter controls the relative temporal spacing of adjacent visual tokens. In summary, our proposed position encoding scheme demonstrates favorable characteristics for modeling video data, yielding a robust and effective representation of positional information.
Overall, the contributions of this work are summarized as:
(1) We present an analysis of four key properties essential for RoPE when applied to video. Motivated by this analysis, we propose VideoRoPE including Low-frequency Temporal Allocation (LTA), Diagonal Layout (DL), and Adjustable Temporal Spacing (ATS) to satisfy all four properties.
(2) We introduce the challenging V-NIAH-D task to expose the drawbacks of current position embedding designs regarding frequency allocation. We reveal that existing Video LLMs are easily misled to frequency-based distractors.
(3) Extensive experiments demonstrate that VideoRoPE consistently achieves superior performance compared to other RoPE variants. For example, VideoRoPE outperforms previous M-RoPE on long video retrieval (+12.4 on V-NIAH, +12.4 on V-NIAH-D), video understanding (+2.9 on LongVideoBench, +4.5 on MLVU, +1.7 on Video-MME) and hallucination (+11.9 on VideoHallucer) benchmarks.
this section cite: ['b19', 'b75', 'b16']

Section: Related Work
RoPE (Rotary Position Embedding). RoPE (Su et al., 2024) is a pivotal mechanism for encoding positional information in LLM long-context modeling. Using a rotation matrix, RoPE unifies the advantages of both absolute and relative positional embedding schemes. In RoPE design, different feature dimensions are embedded with position information based on Trigonometric functions sin and cos with different frequencies (Peng et al., 2023;Liu et al., 2023b). Lower dimensions correspond to higher frequency given larger values of base frequency. The simplicity and effectiveness of RoPE have led to its widespread adoption in leading LLMs (Touvron et al., 2023a;Yang et al., 2024a;Team et al., 2024;Cai et al., 2024;Sun et al., 2024).
this section cite: ['b52', 'b46', 'b55', 'b3', 'b53']

Section: Extending RoPE to Multi-Modal Data.
Extending RoPE to multi-modal or Video LLMs typically follows two approaches. One approach directly applies standard RoPE, flattening visual tokens and treating text and visual tokens as a single 1D sequence. Although variants (e.g., TAD-RoPE (Gao et al., 2024)) introduce enhancements in indexing and attention mechanisms, these 1D RoPE variants overlook the spatiotemporal structure of video and inherent inter-modal differences (Su, 2024a;b;Wang et al., 2024a). In contrast, several studies have explored incorporating structural information to formulate the 2D/3D RoPE. For example, some previous works (Agrawal et al., 2024;Wang et al., 2024a) integrate RoPE-2D into visual encoders to improve spatial representation, particularly for resolution scaling. Based on the RoPE-Tie (Su, 2024a), M-RoPE (Wang et al., 2024a) used in QWen2-VL further generalizes RoPE to three dimensions to model both temporal and spatial dynamics. While effective, M-RoPE exhibits limitations, such as struggles with distractors in our V-NIAH-D task. This work presents a comprehensive analysis of the important characteristics essential for extending RoPE to video and proposes Vide-oRoPE according to our analysis.
this section cite: ['b16', 'b0']

Section: Analysis
3D Structure. The vanilla RoPE defines a matrix A t1,t2 that represents the relative positional encoding between two positions t 1 and t 2 in a 1D sequence:
A t1,t2 = (q t1 R t1 ) (k t2 R t2 ) = q t1 R ∆t k t2 ,(1)
where ∆t = t 1 -t 2 , the symbols q t1 and k t2 are the query and key vectors at positions t 1 and t 2 . The relative rotation matrix R ∆t is defined as R ∆t = exp(∆tiθ n ), while i is the imaginary unit, θ n = β -2n/d is the frequency of rotation applied to a specific n-th pair of d dimensions (n = 0, . . . , d/2 -1), and β is the frequency base parameter. The vanilla RoPE uses d = 128, thus n = 0, . . . , 63. Consequently, the A t1,t2 in Eq. ( 1) can be extended as:
     q (0) q (1) . . . q (126) q (127)          cos θ0∆t -sin θ0∆t • • • 0 0 sin θ0∆t cos θ0∆t • • • 0 0 . . . . . . . . . . . . . . . 0 0 • • • cos θ63∆t sin θ63∆t 0 0 • • • sin θ63∆t cos θ63∆t          k (0) k (1) . . . k (126) k (127)     
(2) While the vanilla RoPE operates on 1D sequences, it can also be applied to higher-dimensional input by flattening the input into a 1-D sequence. However, the flattening process discards crucial neighborhood information, increases the sequence length, and hinders the capture of long-range dependencies. Therefore, preserving the inherent 3D structure is essential when adapting RoPE for video data. Some recent RoPE-variants (e.g., M-RoPE in Qwen2-VL (Wang et al., 2024a)) incorporate the 3D structure. The corresponding relative matrix A (t1,x1,y1) is computed as:
A (t1,x1,y1),(t2,x2,y2) = q (t1,x1,y1) R ∆t,∆x,∆y k (t2,x2,y2) ,(3)
where ∆t = t 1 -t 2 , ∆x = x 1 -x 2 , and ∆y = y 1 -y 2 . M-RoPE divides the d = 128 feature dimensions into 3 groups: the first 32 for temporal positions (t), the middle 48 for horizontal positions (x), and the last 48 for vertical positions (y). As shown in Eq (4), A (t1,x1,y1),(t2,x2,y2) in M-RoPE is extended as:
            q (0) q (1) q (2) q (3) . . . q (30) q (31)                        cos θ0∆t -sin θ0∆t 0 0 • • • 0 0 sin θ0∆t cos θ0∆t 0 0 • • • 0 0 0 0 cos θ1∆t -sin θ1∆t • • • 0 0 0 0 sin θ1∆t cos θ1∆t • • • 0 0 . . . . . . . . . . . . . . . . . . . . . 0 0 0 0 • • • cos θ15∆t -sin θ15∆t 0 0 0 0 • • • sin θ15∆t cos θ15∆t                        k (0) k (1) k (2) k (3) . . . k (30) k (31)            
modeling temporal dependency with higher frequency
+             q (32) q (33) q (34) q (35) . . . q (78) q (79)                        cos θ16∆x -sin θ16∆x 0 0 • • • 0 0 sin θ16∆x cos θ16∆x 0 0 • • • 0 0 0 0 cos θ17∆x -sin θ17∆x • • • 0 0 0 0 sin θ17∆x cos θ17∆x • • • 0 0 . . . . . . . . . . . . . . . . . . . . . 0 0 0 0 • • • cos θ39∆x -sin θ39∆x 0 0 0 0 • • • sin θ39∆x cos θ39∆x                        k (32) k (33) k (34) k (35) . . . k (78) k (79)            
modeling horizontal dependency with intermediate frequency
+             q (80) q (81) q (82) q (83) . . . q (126) q (127)                        cos θ40∆y -sin θ40∆y 0 0 • • • 0 0 sin θ40∆y cos θ40∆y 0 0 • • • 0 0 0 0 cos θ41∆y -sin θ41∆y • • • 0 0 0 0 sin θ41∆y cos θ41∆y • • • 0 0 . . . . . . . . . . . . . . . . . . . . . 0 0 0 0 • • • cos θ63∆y -sin θ63∆y 0 0 0 0 • • • sin θ63∆y cos θ63∆y                        k (80) k (81) k (82) k (83) . . . k (126) k (127)            
modeling vertical dependency with lower frequency (4) Frequency Allocation. Incorporating 3D structure raises the question of how to allocate the temporal (t), horizontal (x), and vertical (y) components within the d dimensions. Note that different allocation strategies are not equivalent Question: what is being transferred to the beaker in the laboratory? M-RoPE: A. Solid substance ... ... needle haystack haystack A. Solid substance B. Gas C. Nothing D. Liquid tester VideoRoPE -x VideoRoPE -y Video RoPE: D. Liquid tester 0 10 20 30 40 50 0 10 20 30 40 50 0 10 20 30 40 50 0 10 20 30 40 50 0 10 20 30 40 50 Figure 3. Attention-based frequential allocation analysis. Middle: M-RoPE's temporal dimension (t) is limited to local information, resulting in a diagonal layout. Bottom: VideoRoPE effectively retrieves the needle using the temporal dimension. The x and y coordinates represent the video frame number, e.g., 50 for 50 frames. For more details see Appendix E.
in the rotation frequency θ n = β -2n/d . As shown in Eq. ( 4), M-RoPE assigns higher frequencies (corresponding to lower dimensions) to the temporal dimension (t).
To highlight the importance of frequency allocation, we introduce a challenging retrieval task Visual Needle-In-A-Hastack-Distractor (V-NIAH-D). V-NIAH-D builds upon V-NIAH (Zhang et al., 2024d), a benchmark designed to evaluate visual long-context understanding. However, the straightforward retrieval-based task has been shown to provide only a superficial form of long-context understanding (Hsieh et al., 2024;Yuan et al., 2024). Therefore, We enhance V-NIAH by incorporating semantically similar distractors, obtained using Google Image Search (Google, 2025) or Flux (Labs, 2023), to mitigate the possibility of correct answers through random chance. These distractors are designed to be unambiguous to the question in Fig. 2.
As shown in Fig. 2, M-RoPE exhibits a clear performance drop from V-NIAH to V-NIAH-D. To investigate this decline, we follow previous works (Xiao et al., 2023;Liu et al., 2023b;Barbero et al., 2024) to visualize the attention scores in Fig. 3. We decompose the attention scores into their corresponding temporal (t), horizontal (x), and vertical (y) components for visualization.
Fig. 3 reveals unusual M-RoPE's attention patterns, despite locating the needle image, it fails to answer the multi-choice question. According to M-RoPE's attention, the needle is located primarily through vertical positional information, rather than temporal features. Thus, the temporal dimension fails to capture long-range semantic dependencies, focusing on local relationships. Conversely, the spatial dimensions capture long-range rather than local semantic information. Lastly, the horizontal and vertical dimensions display distinct characteristics, with the vertical dimension exhibiting phenomena reminiscent of attention sinks (Xiao et al., 2023). These suggest the performance decline primarily results from sub-optimal frequency allocation designs of M-RoPE.
this section cite: ['b19', 'b75', 'b17', 'b27', 'b68', 'b1', 'b68']

Section: Spatial Symmetry.
Given the text tokens T and the visual tokens T v , spatial symmetry (Su, 2024b) claims that the distance between the end of the preceding textual input (T pre ) and the beginning of the visual input (T start v ) is equal to the distance between the end of the visual input (T end v ) and the beginning of the subsequent textual input (T sub ):
T start v -T pre = T sub -T end v . (5
)
The spatial symmetrical structure can potentially simplify the learning process and reduce bias toward input order. However, existing 3D RoPE variants such as M-RoPE do not meet the spatial symmetry, we will elaborate related discussion in Fig. 6.
Temporal Index Scaling. The frame index in video and the token index in text are inherently different (Su, 2024b;Li et al., 2024a). Recognizing this difference, methods like TAD-RoPE, a 1D RoPE adaptation for Video LLMs, introduce distinct step offsets for image and text token indices: γ for image tokens and γ + 1 for text tokens. Consequently, an ideal RoPE design for video data should permit scaling of the temporal index to meet the inherent difference between the frame index and the text index.
this section cite: []

Section: VideoRoPE
Based on some previous research and the above analysis, we claim that a good RoPE design for Video LLMs, especially for long videos, should satisfy four requirements. The first requirement has been solved by RoPE-Tie (Su, 2024a) and the subsequent M-RoPE (Wang et al., 2024a). To solve the last three requirements and mitigate the performance decline observed in V-NIAH-D, we propose our VideoRoPE, comprising the following three key components.
this section cite: []

Section: Low-frequency Temporal Allocation (LTA).
As shown in Eq. ( 2), the vanilla RoPE (Su et al., 2024) uses all dimensions to model the 1D position information. And as indicated in Eq. ( 4), M-RoPE (Wang et al., 2024a) uses different dimensions to model temporal, horizontal, and vertical dimensions sequentially. However, previous frequency allocation strategies are suboptimal because different RoPE dimensions capture dependencies at varying ranges. As shown in Fig. 3, an interesting observation is that the local Based on our analysis, VideoRoPE uses higher dimensions for temporal features in longer contexts and lower dimensions for spatial features, which are limited by resolution and have a fixed range. To avoid the gap between horizontal and vertical positions, we interleave the dimensions responsible for these spatial features. The dimension distribution for VideoRoPE is shown in Eq. ( 6):
            q (96) q (97) q (98) q (99) . . . q (126) q (127)                        cos θ48∆t -sin θ48∆t 0 0 • • • 0 0 sin θ48∆t cos θ48∆t 0 0 • • • 0 0 0 0 cos θ49∆t -sin θ49∆t • • • 0 0 0 0 sin θ49∆t cos θ49∆t • • • 0 0 . . . . . . . . . . . . . . . . . . . . . 0 0 0 0 • • • cos θ63∆t -sin θ63∆t 0 0 0 0 • • • sin θ63∆t cos θ63∆t                        k (96) k (97) k (98) k (99) . . . k (126) k (127)            
modeling temporal dependency with lower frequency
+             q (0) q (1) q (4) q (5) . . . q (92) q (93)                        cos θ0∆x -sin θ0∆x 0 0 • • • 0 0 sin θ0∆x cos θ0∆x 0 0 • • • 0 0 0 0 cos θ2∆x -sin θ2∆x • • • 0 0 0 0 sin θ2∆x cos θ2∆x • • • 0 0 . . . . . . . . . . . . . . . . . . . . . 0 0 0 0 • • • cos θ46∆x -sin θ46∆x 0 0 0 0 • • • sin θ46∆x cos θ46∆x                        k (0) k (1) k (4) k (5) . . . k (92) k (93)            
modeling horizontal dependency with interleaved high frequency
+             q (2) q (3) q (6) q (7) . . . q (94) q (95)                        cos θ1∆y -sin θ1∆y 0 0 • • • 0 0 sin θ1∆y cos θ1∆y 0 0 • • • 0 0 0 0 cos θ3∆y -sin θ3∆y • • • 0 0 0 0 sin θ3∆y cos θ3∆y • • • 0 0 . . . . . . . . . . . . . . . . . . . . . 0 0 0 0 • • • cos θ47∆y -sin θ47∆y 0 0 0 0 • • • sin θ47∆y cos θ47∆y                        k (2) k (3) k (6) k (7) . . . k (94) k (95)            
modeling vertical dependency with interleaved high frequency (6) The horizontal position x and vertical position y are inter-Ok, this is a text.
2 2 2 ... 2 2 ... 2 2 2 ... 2 3 3 3 ... 3 3 ... 3 3 3 ... 3 t dimension 2 2 2 ... 1 1 ... 1 0 0 ... 0 3 3 3 ... 1 1 ... 1 0 0 ... 0 t dimension x dimension y dimension 1 0 1 ... 0 1 ... 0 2 2 ... 2 1 2 ... 1 2 ... 1 2 3 3 ... 3 t dimension x,y dimension Position index for RoPE in text Position index for M-RoPE in video Position index for VideoRoPE (ours) leaved to occupy the lower dimensions, followed by temporal t, which occupies the higher dimensions. We keep the same allocation number for x, y, and t as M-RoPE for a fair comparison, with values of 48, 48, and 32, respectively. The advantages of this distribution are evident in Fig. 4. For a RoPE-based LLM with a 128-dimensional head (64 rotary angles θ n ), we visualize the function of cos θ n t for 3 dimensions using parallel blue planes.
As shown in Fig. 4 (a), M-RoPE's temporal position embeddings are significantly distorted by periodic oscillations (Men et al., 2024), leading to identical embeddings for distant positions. For instance, considering the last three rotary angles, the temporal embeddings are severely affected by these oscillations due to their short monotonic intervals (and even shorter intervals in lower dimensions). This periodicity creates "hash collisions" (red planes), where distant positions share near-identical embeddings, making the model susceptible to distractor influence. Fortunately, our VideoRoPE (Fig. 4 (b)) is free from oscillation and Hash collision in temporal modeling. The relationship between periodicity, monotonicity, and temporal modeling is visualized in Fig 4.
x a x is y a x i s t ax is (a) 3D visualization for Vanilla RoPE.
x a x is y a x i s t ax is (b) 3D visualization for M-RoPE.
x a x is y a x i s t ax is
(c) 3D visualization for VideoRoPE. Diagonal Layout. Fig. 6 provides a visual comparison of spatial symmetry in positional encodings. For vanilla RoPE (Fig. 6a), no spatial relation is considered and the index for every dimension increases directly. While M-RoPE (Fig. 6b), incorporates spatial information within each frame, it introduces two significant discontinuities between textual and visual tokens. This arises from M-RoPE's placement strategy, if the first visual token is at (0, 0), the last token in each frame will always be placed at (W -1, H -1), creating a stack in the bottom-left corner. Furthermore, like vanilla RoPE, M-RoPE's indices increase with input length across all dimensions.
To address these limitations, VideoRoPE arranges the entire input along the diagonal, see Fig. 6c. The central patch's 3D position for each video frame is (t, t, t), with other patches offset in all directions. Our Diagonal Layout has two advantages: (1) our design preserves the relative positions of visual tokens and ensures approximate equidistance from the image corners to the center, preventing text tokens from being overly close to any corner. (2) It maintains the indexing pattern of vanilla RoPE (Fig. 5), as the position index increment between corresponding spatial locations in adjacent frames mirrors that of adjacent textual tokens.
this section cite: ['b52', 'b45']

Section: Adjustable Temporal Spacing.
To scale the temporal index, we introduce a scaling factor δ to better align temporal information between visual and textual tokens.
Suppose the symbol τ denotes the token index, for the starting text (0 ≤ τ < T s ), the temporal, horizontal, and vertical indices are simply set to the raw token index τ . For the video input (T s ≤ τ < T s + T v ), The difference τ -T s represents the index of the current frame relative to the start of the video, which is then scaled by δ to control the space in the temporal dimension. For the ending text (T s +T v ≤ τ < T s +T v +T e ), the temporal, horizontal, and vertical index are the same, creating a linear progression.
According to our adjustable temporal spacing design, for a multi-modal input that consists of a text with T s tokens, a following video with T v frame with W × H patches in each frame, and an ending text with T e tokens, the position indices (t, x, y) of VideoRoPE for τ -th textual token or (τ, w, h)-th visual token are defined as Eq. ( 7):
(t, x, y) =                            (τ, τ, τ ) if 0 ≤ τ < Ts    Ts + δ(τ -Ts), Ts + δ(τ -Ts) + w -W 2 , Ts + δ(τ -Ts) + h -H 2    if Ts ≤ τ < Ts + Tv    τ + (δ -1)Tv, τ + (δ -1)Tv, τ + (δ -1)Tv   
if Ts + Tv ≤ τ < Ts + Tv + Te , (7) where w and h represent the horizontal and vertical indices of the visual patch within the frame, respectively.
In summary, the parameter δ in our adjustable temporal spacing allows for a flexible and consistent way to encode the relative positions of text and video tokens.
this section cite: []

Section: Experiment

this section cite: []

Section: Experimental Setup
Training Data. We use a subset of LLaVA-Video-178k dataset (Zhang et al., 2024e) to train VideoRoPE. The LLaVA-Video-178k dataset covers 178k videos and around 5 million question-answers (QA) pairs from diverse sources such as HD-VILA (Xue et al., 2022), Kinetics (Kay et al., 2017), and ActivityNet (Fabian Caba Heilbron & Niebles, 2015). To balance training efficiency and long-video comprehension, we randomly select 136k videos with durations under 2 minutes and 18k videos with durations between 2 and 3 minutes. This process yielded our training set of approximately 1.3 million pairs. Implementation Details. Using the aforementioned video  Qwen2-7B (Yang et al., 2024a). Our fine-tuning incorporates our VideoRoPE to process the spatiotemporal nature of the video data effectively. We adopt Qwen2-VL's finetuning settings, processing each video at 2 fps with a maximum of 128 frames and dynamically adjusting the image resolution to maintain a consistent token count. However, to prevent memory overflow, we use a context window of 8192 tokens.
Our fine-tuning process employs a batch size of 128, a cosine scheduler with a learning rate of 1e-5, a warm-up ratio of 1e-2, and 704 Nvidia-A100 GPU hours in total. The evaluation involves sampling videos at 2 fps with a minimum of 144 image tokens per frame. We use the vLLM framework (Kwon et al., 2023) to support inference on sequences longer than 32k tokens.
Evaluation Benchmarks. We evaluate our approach using six video benchmarks, including tasks related to long video understanding, long video retrieval, and video hallucination.
For long video understanding, we use LongVideoBench For video hallucination, we use VideoHallucer (Wang et al., 2024d), which evaluates the model's ability to correctly answer both basic and hallucinated questions about video content. Details of these benchmarks can be found in Appendix B.
this section cite: ['b72', 'b25', 'b14', 'b26']

Section: Results on Long Video Understanding
As shown in Tab. 2, we compare our VideoRoPE with existing RoPE variants (vanilla RoPE (Su et al., 2024), TAD-RoPE (Gao et al., 2024), and M-RoPE (Wang et al., 2024a)) across three prominent video understanding benchmarks. Our VideoRoPE consistently outperforms all baseline methods across these benchmarks, demonstrating its robustness and adaptability. Specifically, VideoRoPE achieves improvements of up to 2.91, 4.46, and 1.66 points (64k context length) over the M-RoPE baseline on LongVideoBench, MLVU, and Video-MME, respectively. These results emphasize the superior ability of VideoRoPE to effectively
Table 3. Performance comparison of different RoPEs on V-NIAH and V-NIAH-D. "Acc." refers to the average accuracy across haystack length and frame depth. (Su et al., 2024) 31.78 30.22 TAD-RoPE (Gao et al., 2024) 29.33 29.56 M-RoPE (Wang et al., 2024a) 78.67 74.67 VideoRoPE 91.11 87.11 Table 4. Performance comparison of different RoPEs on VideoHallucer, evaluated at context lengths of 8k, 16k, 32k, and 64k. The maximum result for each RoPE variant across these context lengths is displayed, with bold for the top result and underlined for the second-highest. 'OR' = Object-Relation, 'T' = Temporal, 'SD' = Semantic Detail, 'F' = Factual, 'NF' = Non-factual.
Method V-NIAH Acc. V-NIAH-D Acc. Vanilla RoPE
this section cite: ['b52', 'b16', 'b52', 'b16']

Section: Method OR T SD F NF Avg.
Vanilla RoPE (Su et al., 2024) 51.5 30.0 48.0 8.0 43.0 36.1 TAD-RoPE (Gao et al., 2024) 51.0 37.0 48.0 11.5 47.5 39.0 M-RoPE (Wang et al., 2024a) 39.0 29.0 43.5 12.5 47.5 34.3 VideoRoPE 57.0 58.5 50.5 15.0 50.0 46.2 capture long-range dependencies and maintain performance across various challenging video data tasks.  2) show that both Vanilla RoPE and TAD-RoPE exhibit some extrapolation ability beyond the visual training context. However, both methods fail once they exceed a certain extrapolation limit. In contrast, Fig. 7 (3) and ( 4) highlight the superior performance of VideoRoPE and M-RoPE in extrapolating within the test context range. While both VideoRoPE and M-RoPE successfully handle extrapolation, VideoRoPE consistently outperforms M-RoPE, showcasing the robustness of the task. Tab. 3 provides a quantitative analysis of the retrieval results, demonstrating a 12.44 % performance improvement of our method over M-RoPE on the Video Retrieval task in both settings, confirming the advantages of our proposed method in video retrieval scenarios.
this section cite: ['b52', 'b16']

Section: Results on Long Video Retrieval

this section cite: []

Section: Results on Video Hallucination
As highlighted in Tab. 4, VideoRoPE significantly surpasses current RoPE methods on the VideoHallucer benchmark. In particular, for the Temporal Hallucination task, VideoRoPE demonstrates a substantial performance improvement of 29.5%, indicating its enhanced capability to accurately capture and process temporal dependencies. This improvement suggests that VideoRoPE is better equipped to handle dynamic video sequences, where the understanding of timebased relationships is critical. Similarly, for the Spatial Hallucination task, specifically the Object-Relation Hallucination subtask, VideoRoPE achieves an impressive 18.0% improvement over existing methods, highlighting its ability to better discern complex spatial interactions. These results underscore VideoRoPE's robustness in solving video hallucination and potential for real-world video analysis.
this section cite: []

Section: Ablation Studies Ablation Studies on Module Design.
We conduct ablation experiments on the modules introduced in Section 4, quantitatively evaluating their impact on LongVideoBench and MLVU benchmarks. The experimental results are presented in Tab. 5. The baseline setting, M-RoPE (Wang et al., 2024a), achieves scores of 54.35 on LongVideoBench and 61.10 on MLVU (both using a 64k context length). By progressively integrating the DL (Diagonal Layout), LTA (Low-frequency Temporal Allocation), and ATS (Adjustable Temporal Spacing) modules, our method shows a continuous improvement in performance, achieving enhanced scores of 57.26 on LongVideoBench and 65.56 on MLVU (both using a 64k context length). These results demonstrate the effectiveness of our approach in leveraging spatial-temporal positional information. To refine the analysis of x and y allocation in LTA, we quantitatively evaluate interleaved vs. sequential layouts. We also compare strategies for allocating t, x, and y, including M-RoPE, a uniform interleaved layout, and our VideoRoPE design. Additionally, we explore the optimal ATS scaling factor by varying its value, and further ablate the diagonal layout module to validate the symmetry-based design. See Appendix A.1 for details.
this section cite: []

Section: Conclusion
This paper identifies four key criteria for effective positional encoding: 2D/3D structure, frequency allocation, spatial symmetry, and temporal index scaling. As part of our analysis, through the V-NIAH-D task, we demonstrate that previous RoPE variants are vulnerable to distractors because of lacking proper temporal allocation. As a result, We propose VideoRoPE that uses a 3D structure for spatiotemporal coherence, low-frequency temporal allocation to reduce oscillations, a diagonal layout for spatial symmetry, and adjustable temporal spacing. VideoRoPE outperforms previous RoPE variants in tasks like long video retrieval, video understanding, and video hallucination. We evaluate these approaches on the LongVideoBench benchmark under varying context lengths. This benchmark includes a diverse set of video scenarios, ranging from rapidly changing dynamic scenes to slowly evolving static content.
As shown in the results below, our low-frequency temporal allocation consistently outperforms the interleaved [t t x y x y x y] pattern on average. This suggests that our frequency design more effectively balances global temporal context modeling with local spatial dynamics, making it better suited to handle a wide variety of video conditions.
this section cite: []

Section: References
Ref_id:b0 Title: Pixtral 12b Year: (2024)
Ref_id:b1 Title: Round and round we go! what makes rotary positional encodings useful? arXiv preprint Year: (2024)
Ref_id:b2 Title: Is space-time attention all you need for video understanding? Year: (2021)
Ref_id:b3 Title:  Year: (2024)
Ref_id:b4 Title: Auroracap: Efficient, performant video detailed captioning and a new benchmark Year: (2024)
Ref_id:b5 Title: Open-llava-next: An open-source implementation of llava-next series for facilitating the large multi-modal model community Year: (2024)
Ref_id:b6 Title: Sharegpt4v: Improving large multi-modal models with better captions Year: (2023)
Ref_id:b7 Title: Sharegpt4video: Improving video understanding and generation with better captions Year: (2024)
Ref_id:b8 Title:  Year: ()
Ref_id:b9 Title: Sam2long: Enhancing sam 2 for long video segmentation with a training-free memory tree Year: (2024)
Ref_id:b10 Title: Towards multimodal instruction following Year: (2025)
Ref_id:b11 Title: Extending llm context window beyond 2 million tokens Year: (2024)
Ref_id:b12 Title: Mastering free-form text-image composition and comprehension in vision-language large model Year: (2024)
Ref_id:b13 Title: The llama 3 herd of models Year: (2024)
Ref_id:b14 Title: ActivityNet: A large-scale video benchmark for human activity understanding Year: (2015)
Ref_id:b15 Title: Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis Year: (2024)
Ref_id:b16 Title: Rethinking the transfer from image to video understanding with temporal considerations Year: (2024)
Ref_id:b17 Title: Google image search Year: (2025)
Ref_id:b18 Title: Lm-infinite: Zero-shot extreme length generalization for large language models Year: (2024)
Ref_id:b19 Title: Ruler: What's the real context size of your long-context language models Year: (2024)
Ref_id:b20 Title: Empower llm to grasp video moments Year: (2023)
Ref_id:b21 Title: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation Year: (2024)
Ref_id:b22 Title: Chat-univi: Unified visual representation empowers large language models with image and video understanding Year: (2023)
Ref_id:b23 Title: Needle in a haystack -pressure testing llms Year: ()
Ref_id:b24 Title:  Year: (2023)
Ref_id:b25 Title: The kinetics human action video dataset Year: (2017)
Ref_id:b26 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b27 Title: Multi needle in a haystack Year: (2023)
Ref_id:b28 Title: Less is more: Clipbert for video-and-language learning via sparse sampling Year: (2021)
Ref_id:b29 Title: Chat-centric video understanding Year: (2023)
Ref_id:b30 Title: Temporal reasoning transfer from text to video Year: (2024)
Ref_id:b31 Title: Llama-vid: An image is worth 2 tokens in large language models Year: (2024)
Ref_id:b32 Title: Ovo-bench: How far is your videollms from real-world online video understanding? Year: (2025)
Ref_id:b33 Title: Learning united visual representation by alignment before projection Year: (2023)
Ref_id:b34 Title: Learning united visual representation by alignment before projection Year: (2023)
Ref_id:b35 Title: Mm-vid: Advancing video understanding with gpt-4v(ision), 2023c Year: ()
Ref_id:b36 Title: Visual instruction tuning Year: ()
Ref_id:b37 Title: Scaling laws of rope-based extrapolation Year: (2023)
Ref_id:b38 Title: Towards open-ended event-level video-language understanding Year: (2024)
Ref_id:b39 Title: Mmdu: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms Year: (2024)
Ref_id:b40 Title: Rar: Retrieving and ranking augmented mllms for visual recognition Year: (2024)
Ref_id:b41 Title: Multi-image augmented direct preference optimization for large vision-language models Year: (2024)
Ref_id:b42 Title: Video assistant with large language model enhanced ability Year: (2023)
Ref_id:b43 Title: Videochatgpt: Towards detailed video understanding via large vision and language models Year: ()
Ref_id:b44 Title: Videochatgpt: Towards detailed video understanding via large vision and language models Year: (2024)
Ref_id:b45 Title: Base of rope bounds context length Year: (2024)
Ref_id:b46 Title: Yarn: Efficient context window extension of large language models Year: (2023)
Ref_id:b47 Title: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction Year: (2025)
Ref_id:b48 Title: Streaming long video understanding with large language models Year: (2025)
Ref_id:b49 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b50 Title: Transformer upgrade path: 17. insights into multimodal positional encoding Year: (2024-03)
Ref_id:b51 Title: A brief discussion on multimodal thinking: 3. positional encoding Year: (2024-09)
Ref_id:b52 Title: RoFormer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b53 Title: Moss: An open conversational large language model Year: (2024)
Ref_id:b54 Title: Alpha-clip: A clip model focusing on wherever you want Year: (2023)
Ref_id:b55 Title: Open models based on gemini research and technology Year: (2024)
Ref_id:b56 Title: Llama: Open and efficient foundation language models Year: (2023)
Ref_id:b57 Title: LLaMA 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b58 Title: Multimodal needle in a haystack: Benchmarking long-context capability of multimodal large language models Year: (2025)
Ref_id:b59 Title: A tracklet-centric multimodal and versatile video understanding system Year: (2023)
Ref_id:b60 Title: Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b61 Title: Needle in a multimodal haystack Year: (2024)
Ref_id:b62 Title: Scaling multi-modal llms to 1000 images efficiently via a hybrid architecture Year: (2024)
Ref_id:b63 Title: General video foundation models via generative and discriminative learning Year: (2022)
Ref_id:b64 Title: Videohallucer: Evaluating intrinsic and extrinsic hallucinations in large video-language models. arxiv Year: (2024)
Ref_id:b65 Title: Adaptive treebased video representation for llm reasoning on long videos Year: (2024)
Ref_id:b66 Title: A benchmark for long-context interleaved video-language understanding Year: (2024)
Ref_id:b67 Title: Visual haystacks: A vision-centric needle-in-a-haystack benchmark Year: (2024)
Ref_id:b68 Title: Efficient streaming language models with attention sinks Year: (2023)
Ref_id:b69 Title: Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction Year: (2024)
Ref_id:b70 Title: Videoclip: Contrastive pre-training for zeroshot video-text understanding Year: (2021)
Ref_id:b71 Title: Pllava : Parameter-free llava extension from images to videos for video dense captioning Year: (2024)
Ref_id:b72 Title: Advancing high-resolution videolanguage representation with large-scale video transcriptions Year: (2022)
Ref_id:b73 Title: Qwen2 technical report Year: (2024)
Ref_id:b74 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b75 Title: Lv-eval: A balanced long-context benchmark with 5 length levels up to 256k Year: (2024)
Ref_id:b76 Title: Internlmxcomposer2. 5-reward: A simple yet effective multimodal reward model Year: (2025)
Ref_id:b77 Title: Long-clip: Unlocking the long-text capability of clip Year: (2024)
Ref_id:b78 Title: Video-llama: An instructiontuned audio-visual language model for video understanding Year: (2023)
Ref_id:b79 Title: Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition Year: (2023)
Ref_id:b80 Title: 5-omnilive: A comprehensive multimodal system for long-term streaming video and audio interactions Year: (2024)
Ref_id:b81 Title: Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output Year: (2024)
Ref_id:b82 Title: Long context transfer from language to vision Year: (2024)
Ref_id:b83 Title: Llava-mini: Efficient image and video large multimodal models with one vision token Year: (2025)
Ref_id:b84 Title: Video instruction tuning with synthetic data Year: (2024)
Ref_id:b85 Title: Towards enhanced alignment of mllms with human preference Year: (2024)
Ref_id:b86 Title: Needle in a video haystack: A scalable synthetic framework for benchmarking video mllms Year: (2024)
Ref_id:b87 Title: MLVU: A comprehensive benchmark for multi-task long video understanding Year: (2024)
