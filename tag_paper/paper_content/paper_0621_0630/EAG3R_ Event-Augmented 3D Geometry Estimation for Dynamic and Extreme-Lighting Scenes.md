Title: EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes
Abstract: Robust 3D geometry estimation from videos is critical for applications such as autonomous navigation, SLAM, and 3D scene reconstruction. Recent methods like DUSt3R demonstrate that regressing dense pointmaps from image pairs enables accurate and efficient pose-free reconstruction. However, existing RGB-only approaches struggle under real-world conditions involving dynamic objects and extreme illumination, due to the inherent limitations of conventional cameras. In this paper, we propose EAG3R, a novel geometry estimation framework that augments pointmap-based reconstruction with asynchronous event streams. Built upon the MonST3R backbone, EAG3R introduces two key innovations: (1) a retinex-inspired image enhancement module and a lightweight event adapter with SNR-aware fusion mechanism that adaptively combines RGB and event features based on local reliability; and (2) a novel event-based photometric consistency loss that reinforces spatiotemporal coherence during global optimization. Our method enables robust geometry estimation in challenging dynamic low-light scenes without requiring retraining on night-time data. Extensive experiments demonstrate that EAG3R significantly outperforms state-of-the-art RGB-only baselines across monocular depth estimation, camera pose tracking, and dynamic reconstruction tasks.

Section: Introduction
Estimating geometry from videos or images is a fundamental problem in 3D vision, with broad applications in camera pose estimation, novel view synthesis, geometry reconstruction, and 3D perception. These capabilities are crucial in downstream scenarios such as autonomous driving, SLAM, virtual environments, and robotic navigation. Recent methods like DUSt3R [64] have shown that regressing dense pointmaps from image pairs using transformer-based foundation models enables accurate and efficient pose-free 3D reconstruction. This paradigm has sparked a growing trend toward addressing various challenging scenarios, such as longer image sequences [59,62,60], dynamic scenes [72,10,27,55], and integration with techniques like Gaussian Splatting [16,52,18].
However, in real-world applications such as autonomous driving in the wild, which often involve fast motion and rapidly changing illumination, RGB cameras-dependent on long exposure times for imaging-face significant challenges, including blur, out-of-focus artifacts, overexposure, and underexposure. Consequently, the resulting low-quality images hinder reliable geometry estimation.
Event cameras, on the other hand, provide asynchronous measurements of pixel-level brightness changes with high temporal resolution and dynamic range. They have demonstrated strong resilience in challenging conditions such as fast motion and extreme illumination [17,28,47]. Prior work has leveraged event streams in 3D tasks such as depth estimation [4,79,40], surface reconstruction [8,9], 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
EAG3R Input low light video Input event stream L align L flow L smooth L event Pointmaps Variables of Optimization {X, P, K} Depth Camera Pose Camera Intrinsics Object Motion
this section cite: ['b63', 'b58', 'b61', 'b59', 'b71', 'b9', 'b26', 'b54', 'b15', 'b51', 'b17', 'b16', 'b27', 'b46', 'b3', 'b78', 'b39', 'b7', 'b8']

Section: 4D Reconstruction
Trajectories Figure 1: EAG3R pipeline for event-augmented dynamic 3D reconstruction. EAG3R processes a low-light video and its corresponding event stream within a temporal window, extracting pairwise pointmaps for each frame pair. These pointmaps are jointly optimized under alignment, flow, smoothness, and event-based consistency losses to recover a global dynamic point cloud and per-frame camera poses and intrinsics {X, P, K}. This unified representation enables efficient downstream tasks such as depth estimation and camera pose estimation, under challenging lighting conditions. and neural rendering [48,25], but their integration into modern learning-based geometry pipelines remains limited.
In this paper, we propose EAG3R, an event-augemented MonST3R framework to enhance pointmapbased 3D geometry estimation under dynamic and extremely low-light conditions. Built upon the MonST3R [72] backbone, EAG3R introduces two key innovations: (1) a lightweight event adapter with Signal-to-Noise Ratio (SNR)-aware fusion mechanism that adaptively integrates event and image features based on local reliability, and (2) an event-based photometric consistency loss that enforces alignment between predicted motion-induced brightness changes and event-observed brightness changes during global optimization. These components enable EAG3R to remain robust in scenarios where conventional RGB-only pipelines fail.
We evaluate EAG3R on the MVSEC dataset [78], conducting extensive comparisons on depth estimation, camera pose tracking, and dynamic reconstruction in extreme low-light conditions. Results show that EAG3R significantly outperforms existing baselines, including DUSt3R [64], MonST3R [72], and Easi3R [10] variants, even in a zero-shot nighttime setting.
Our main contributions are as follows:
• We propose EAG3R, the first event-augmented pointmap-based geometry estimation framework, which integrates asynchronous event streams with RGB-based reconstruction to handle dynamic scenes under extreme low-light conditions.
• We design a plug-in event perception module that integrates RGB and event data via: (1) a Retinex-based enhancer for visibility recovery and SNR map prediction; (2) a lightweight Swin-Transformer-based event adapter; and (3) an SNR-aware fusion scheme for adaptive feature integration.
• We develop a novel event-based photometric consistency loss that guides global optimization by aligning predicted motion-induced brightness changes with event-observed measurements, improving spatiotemporal coherence under low light.
• We validate EAG3R across multiple challenging 3D vision tasks-including monocular depth estimation, camera pose tracking, and dynamic reconstruction-and show it significantly outperforms existing RGB-based pose-free methods, even under zero-shot nighttime conditions.
this section cite: ['b47', 'b24', 'b71', 'b77', 'b63', 'b71', 'b9']

Section: Related Work
SfM and SLAM Traditional Structure-from-Motion (SfM) [1,44,45,50,53,54] and Simultaneous Localization and Mapping (SLAM) [11,15,38,41] methods estimate 3D structure and camera motion by establishing 2D correspondences [5,12,36,38,49] or minimizing photometric errors [14,15], followed by bundle adjustment (BA) [2,6,61]. While effective with dense views, these often struggle with sparse or ill-conditioned data. Recent learning-based approaches aim to improve robustness and efficiency: DUSt3R [64], directly regress dense point maps from image pairs using Transformer architectures [13] trained on large-scale 3D datasets. However, DUSt3R and its variants [35,39,56,59,60] are primarily designed for static scenes and their performance degrades with dynamic content.
Pose-free Dynamic Scene Reconstruction Reconstructing dynamic scenes without known poses is a core challenge in SLAM. Classical methods rely on joint pose estimation and dynamic region filtering via semantics [71] or optical flow [75], but depend heavily on accurate segmentation and tracking. Some methods estimate temporally consistent depth using geometric constraints [37] or generative priors [24,51], yet suffer from fragmented reconstructions due to missing camera trajectories. Recent works jointly optimize depth and pose by refining pretrained depth models [46] using flow [65] and masks [23], e.g., in CasualSAM [74], Robust-CVD [30], and MegaSAM [32], the latter integrating DROID-SLAM [58] and diverse priors [43,69]. Recent approaches employ direct pointmap regression, including MonST3R [72], DAS3R [67], CUT3R [62], and Easi3R [10], which leverage optical flow, segmentation, or attention for motion disentanglement.
Low-light Enhancement Low-light image enhancement (LIE) aims to improve image quality under poor illumination. Traditional methods like histogram equalization [3] and Retinex-based algorithms [21] have limited adaptability, while deep learning approaches [66,7,63] achieve better results but still struggle in extreme darkness. Event cameras, with high dynamic range and temporal resolution, enable structural information preservation in very low light [76,73], inspiring eventguided LIE methods [26,34]. However, robust fusion of frame and event data under noise remains challenging. EvLight [33] addresses this with adaptive event-image feature fusion.
this section cite: ['b0', 'b43', 'b44', 'b49', 'b52', 'b53', 'b10', 'b14', 'b37', 'b40', 'b4', 'b11', 'b35', 'b37', 'b48', 'b13', 'b14', 'b1', 'b5', 'b60', 'b63', 'b12', 'b34', 'b38', 'b55', 'b58', 'b59', 'b70', 'b74', 'b36', 'b23', 'b50', 'b45', 'b64', 'b22', 'b73', 'b29', 'b31', 'b57', 'b42', 'b68', 'b71', 'b66', 'b61', 'b9', 'b2', 'b20', 'b65', 'b6', 'b62', 'b75', 'b72', 'b25', 'b33', 'b32']

Section: Event-based 3D Vision
Event cameras have enabled progress in 3D reconstruction under challenging lighting and fast motion [17]. Early work used stereo setups for disparity and multi-view stereo [8,42,77], followed by monocular methods based on geometric priors like camera trajectories [28,47]. Recent approaches apply deep learning to stereo [40] and monocular [4,9] settings, producing dense outputs such as meshes or voxels. Multimodal fusion with structured light [31] or RGB-D sensors [79] further improves robustness. Latest advances adapt NeRF [29,48] and Gaussian Splatting [22,25] to event data, enabling high-fidelity scene reconstruction and novel view synthesis.
this section cite: ['b16', 'b7', 'b41', 'b76', 'b27', 'b46', 'b39', 'b3', 'b8', 'b30', 'b78', 'b28', 'b47', 'b21', 'b24']

Section: Methods
Overview Figure 1 shows the overall pipeline of EAG3R. Our work addresses the critical challenge of robust monocular 3D scene reconstruction-encompassing dynamic geometry, camera pose, and depth estimation-under extreme lighting conditions where traditional RGB-based methods often fail. EAG3R enhances the MonST3R framework by synergistically integrating standard RGB video frames {I t ∈ R H×W ×3 } with asynchronous event streams {E t } (sequences of e k = (x k , y k , t k , p k )). This is achieved through two primary strategies: adaptive event-image feature fusion guided by signal quality, and an event-augmented global optimization that incorporates event-derived cues for static region masking and consistency loss.
The subsequent sections provide a detailed exposition: Section 3.1 reviews the foundational DUSt3R and MonST3R architectures. Section 3.2 then describes our Event-data Integration and Feature Fusion approach, including techniques for RGB image enhancement, the design of a lightweight event adapter, and our core SNR-aware fusion mechanism. Finally, Section 3.3 details the Global Optimization with Event Consistency, explaining how event-based consistency loss are integrated to achieve enhanced spatio-temporal coherence across both RGB and event data.
this section cite: []

Section: Preliminary
Our work builds upon DUSt3R and its dynamic extension MonST3R, which employ pointmaps for direct, dense 3D geometry estimation from images, facilitating pose-free monocular reconstruction.
this section cite: []

Section: Pointmap-based Static Reconstruction.
DUSt3R utilizes pointmaps (X pm ∈ R H×W ×3 ), assigning a 3D coordinate per pixel, predicted for image pairs (I a , I b ) by a Transformer model:
(X a→a pm , X b→a pm ) = Model(I a , I b ).(1)
These encode relative geometry for depth and pose estimation and are refined via global optimization for multi-view consistency into a global point cloud X * .
I t Enhance E evt E img Cross Attention D Head E t F img F evt M SNR Enhance I t L illum I prior N retinex M SNR C Upstream E D Head I t E D Head I t' Weights Sharing Cross Attention Reference View Source View I lu Extension to Dynamic Scenes. MonST3R adapts this for dynamic scenes by finetuning DUSt3R on dynamic datasets, predicting per-frame pointmaps. It optimizes a global scene model X * global (comprising per-frame camera poses {P t }, intrinsics {K t }, and depth maps {D t }) using an objective function:
L MonST3R (X * global ) = L align + w smooth L smooth + w flow L flow ,(2)
guided by alignment, trajectory smoothness, and image-based optical flow (L flow ) terms.
However, the reliance of these image-based methods on clear visual information causes them to struggle in low-light settings: RGB images I t lose crucial detail, and MonST3R's flow estimation with RAFT [57] can become unstable. Our EAG3R addresses these limitations by integrating asynchronous event data E t into both the feature extraction and global optimization stages, aiming for robust 3D reconstruction performance even in such challenging lighting conditions.
this section cite: ['b56']

Section: Event-data Integration and Feature Fusion
To enable robust geometry estimation in low-light scenarios, we redesign the encoding pipeline with a hybrid event-image architecture as is illustrated in Fig. 2. Our improvements begin with a Retinex-inspired enhancement module that operates on the raw input image I t to recover visibility in underexposed regions. This module also estimates a SNR map, M t snr , which serves as a spatial prior for confidence-aware fusion. Next, we introduce a lightweight event adapter based on a Swin Transformer backbone, designed to extract high-fidelity features from the sparse event stream E t . We also establish cross-modal interaction through a cross-attention mechanism between event and image features. Finally, we propose an SNR-aware fusion strategy that adaptively balances image and event features based on local SNR, favoring images in well-lit areas and events in low-visibility regions. This yields a more informative and robust representation F t for downstream 3D reconstruction.
this section cite: []

Section: Retinex-based Image Enhancement.
To enhance image visibility under low-light conditions and provide a spatial reliability prior, we introduce a lightweight Retinex-inspired [7] enhancement module. Given an input image I t , we estimate an illumination map L t illum using a shallow network N retinex with inputs I t and its channel-wise maximum projection I t prior = max c (I t ), and compute the enhanced image via element-wise multiplication:
L t illum = N retinex (I t , I t prior ), I t lu = I t ⊙ L t illum .(3)
To guide adaptive fusion, we compute a SNR map M t snr indicating local image reliability. We convert I t lu to grayscale I t g , apply mean filtering to obtain I t g , and define:
M t snr = I t g I t g -I t g + ϵ ,(4)
where ϵ ensures numerical stability. This SNR map emphasizes high-confidence regions and suppresses noise-dominated areas, enabling reliability-aware feature fusion downstream.
this section cite: ['b6']

Section: Lightweight Event Adapter.
To effectively harness asynchronous event streams E t for dense geometric prediction, we introduce a lightweight event adapter by employing a Swin Transformer backbone initialized with weights from a self-supervised, context-based pre-training regimen on event data [70]. The input events from E t are voxelized into a spatiotemporal grid and processed by the pre-trained Swin Transformer encoder, yielding hierarchical event features {F t evt,l } 4 l=1 . Corresponding hierarchical image features {F t img,l } 4 l=1 are extracted at every 6 layers from intermediate representations within the pre-trained image encoder. At each hierarchical stage l, the event features F t evt,l are spatially aligned and dimension-matched with their respective image counterparts F t img,l . We then apply cross-attention [68], using event features as queries and image features as keys and values:
F ′ evt,l = CrossAttn(Q = F t evt,l , K = F t img,l , V = F t img,l ),(5)
Importantly, the image encoder remains frozen, and only the event adapter is updated. This training strategy ensures efficient adaptation without disrupting the pretrained image backbone, while allowing the event pathway to learn to compensate for degraded or missing visual cues.
this section cite: ['b69', 'b67']

Section: SNR-aware Feature Aggregation.
We combine the final features from the image (F t img-final ) and event (F ′t evt-final ) encoders into a unified representation F t , guided by the normalized SNR map Mt snr . Specifically, we weight image features by Mt snr and event features by its complement (1 -Mt snr ), followed by concatenation:
F t cat = (F t img-final ⊙ Mt snr ) ∥ (F ′t evt-final ⊙ (1 -Mt snr )),(6)
where ⊙ denotes element-wise multiplication with channel-wise broadcasting. The concatenated features F t cat undergo a projection to match the input dimensionality of the downstream decoder. This adaptive feature aggregation dynamically prioritizes image features under high-SNR conditions and event features under low-light scenarios, yielding robust and illumination-invariant representations for effective downstream 3D reconstruction.
this section cite: []

Section: Event-Enhanced Global Optimization
To enhance the performance of MonST3R's 3D reconstruction and camera pose estimation, particularly in challenging low-light environments where image-based cues are compromised, our approach augments its global optimization framework. The primary enhancement is the introduction of an event-based photometric consistency loss, L event . This integration leverages the inherent advantages of event cameras-such as their high dynamic range and ability to capture dynamics even with minimal illumination-to provide robust supervisory signals. The subsequent sections detail the formulation of this L event from raw event data (Section 3.3.1) and its incorporation into the joint optimization process (Section 3.3.2).
this section cite: []

Section: Event-Based Photometric Consistency Loss
The event-based photometric consistency loss, L event , is formulated to evaluate the alignment of brightness change patterns observed within salient image patches. It achieves this by comparing two distinct representations of these patterns: the first, ∆L Pm (u), is derived directly from the raw event stream E; the second, ∆ LPm (u; X global ), is synthesized by integrating photometric information from an intensity image with scene motion inferred from the global state estimate X global . The process of computing this loss is visualized in Fig. 3.
Observed Brightness Increments from Events: Given an event stream E corresponding to the time interval between frames I t and I t ′ , we compute the observed brightness increment within each salient image patch P m by aggregating events e k = (x k , y k , t k , p k ) occurring in both space and time over the patch and interval ∆τ = t ′ -t:  where u = (x, y) denotes local coordinates within the patch. Each patch P m is centered at a Harris corner detected on the reference intensity image I t and covers a small spatial neighborhood around the corner location. This ensures that the selected regions exhibit strong intensity gradients and are thus well-suited for event-based tracking. The aggregation in Equation ( 7) yields a polarity-weighted event accumulation image representing the measured brightness changes within each patch.
∆L Pm (u) = tj ∈[t,t ′ ], (xj ,yj )∈Pm p j δ(u -u j ),(7)
this section cite: []

Section: Brightness Increment Model from Intensity and Motion:
To estimate the brightness change within a salient image patch P m , we adopt a predictive model derived from the principle of brightness constancy. This model synthesizes the expected brightness increment ∆ LPm (u; X global ) by combining photometric and geometric cues, specifically:
• Local Intensity Gradient: The spatial gradient ∂I t grad ∂u (û)| Pm is computed over the patch P m from the intensity image I t at time t.
• Inter-frame Pixel Motion: The motion field ∆u t→t ′ cam (û, X global ) represents the per-pixel displacement between frames t and t ′ , computed by projecting 3D points using the depth map D t and the camera intrinsics and extrinsics (K t , K t ′ , P t , P t ′ ) contained in the global state X global .
Assuming locally constant optical flow and small inter-frame displacements, the predicted brightness increment is expressed as:
∆ LPm (û; X global ) = - ∂I t grad ∂u (û)| Pm • ∆u t→t ′ cam (û, X global )| Pm • ∆τ • C,(8)
where ∆τ denotes the integration interval and C is the contrast sensitivity threshold intrinsic to the event sensor. This expression follows the generative model introduced in [20].
Event-Based Loss Objective: While Equation ( 8) provides an explicit formulation, the scale factor ∆τ • C is unknown and varies across sensors and operating conditions. To eliminate this ambiguity, we normalize both the observed and predicted brightness increment patches to unit L 2 norm before computing the residual. This yields an objective that is invariant to the unknown contrast scale and focuses solely on the alignment of gradient directions:
L event (X global ) = Pm u∈Pm ∆L Pm (u) u∈Pm ∥∆L Pm (u)∥ - ∆ LPm (u; X global ) u∈Pm ∥∆ LPm (u; X global )∥ 2 . (9
)
This loss enforces that brightness variations predicted from image gradients and estimated motion are consistent with real event stream observations, thereby providing a principled supervision signal for optimizing X global .
this section cite: ['b19']

Section: Joint Optimization with Event-Based Constraints
The event-based loss L event is integrated into MonST3R's global optimization objective (Eq. 2). The augmented objective to find the optimal global scene model X * global , pairwise alignments {P * W }, and scales {σ * } becomes:
X * global , {P * W }, {σ * } = arg min Xglobal,{P W },{σ} L align (X global , {P W }, {σ}) + w smooth L smooth (X global ) + w flow L flow (X global ) + w event L event (X global )(10)
where w event is the w event_base scaled by the mean of (1 -S norm ), where S norm are the normalized corner SNR values.
L event provides a more dependable constraint on geometry and motion by leveraging informative event patterns in salient patches. Minimizing this augmented objective refines the state estimate X global by ensuring that brightness changes modeled from intensity and motion, ∆ LPm (u; X global ), align closely with observed event patterns ∆L Pm (u). This synergy enhances the accuracy and robustness of 3D reconstruction and pose estimation, particularly when conventional image quality is poor.
this section cite: []

Section: Experiments
We evaluate our method in a variety of tasks, including depth estimation (Section 4.2), camera pose estimation (Section 4.3) and 4D reconstruction (Section 4.4). We perform ablation studies in Section 4.5. We compare EAG3R with state-of-the-art pose free learning-based reconstruction method, including DUSt3R [64], MonST3R [72], and Easi3R [10].
this section cite: ['b63', 'b71', 'b9']

Section: Experiment Details
For training, we fine-tune the MonST3R baseline by training its ViT-Base decoder, DPT heads, Enhancement Net, and the Event Adapter. The Event Adapter is pre-trained on the ETartanAir dataset. Fine-tuning is performed for 25 epochs, using 8,000 image-event pairs per epoch. We employ the AdamW optimizer with a learning rate of 5 × 10 -5 and a mini-batch size of 4 per GPU. The training process completes in approximately 24 hours on 4 NVIDIA RTX 3090 GPUs. For global optimization, we adopt the same setting as MonST3R, with hyperparameters w smooth = 0.01, w f low = 0.01, and w event_base = 0.01. We use the Adam optimizer for 300 iterations with a learning rate of 0.01.
For dataset selection, we initially attempted to fine-tune the MonST3R baseline using events generated via Video-to-Events (V2E) [19] from MonST3R's fine-tuning datasets. However, the noise in V2Egenerated events led to gradient explosion during training, prompting us to switch to datasets with real event captures and ground truth (GT) depth. Given the scarcity of such data, we selected the Multi Vehicle Stereo Event Camera (MVSEC) dataset [78]. It provides synchronized stereo events and reliable LiDAR-derived depth GT. To ensure a fair zero-shot evaluation of low-light performance, all models were trained exclusively on MVSEC's outdoor_day2 sequence (normal daylight) and tested on the challenging outdoor_night1-3 sequences (extreme low-light).
this section cite: ['b18', 'b77']

Section: Monocular Depth Estimation
We evaluate monocular depth estimation on the MVSEC outdoor_night1-3 sequences, which feature extreme low-light conditions with significant noise and underexposure. All models are trained solely on the MVSEC outdoor_day2 sequence and tested zero-shot on these nighttime scenes to ensure a fair comparison. We report results using standard metrics: Absolute Relative Error (Abs Rel ↓), Scale-invariant RMSE log (RMSE log ↓), and the threshold accuracy δ < 1.25 (↑), where lower is better for error metrics and higher is better for accuracy.
As shown in Tab. 1, DUSt3R performs poorly due to the severe degradation of visual cues at night. However, applying RetinexFormer, a widely used image enhancement network, as a preprocessing light-up step (denoted as (LightUp)) does not yield significant improvements and, in some cases, degrades performance, indicating that image enhancement alone is insufficient for this task without joint optimization with the downstream model. Fine-tuning MonST3R improves its performance across most metrics, demonstrating the benefit of domain adaptation. Our method, EAG3R, outperforms all baselines across all three nighttime sequences, indicating both accurate and reliable depth predictions. EAG3R leverages asynchronous event signals that remain informative in such challenging low-light settings, which enables strong generalization capabilities despite the model never having been trained on nighttime data. These results highlight the distinct advantage of incorporating event-based cues for robust depth estimation under extreme illumination conditions, where conventional RGB-based methods-even when augmented with fine-tuning or pre-enhancement-struggle to perform reliably.
this section cite: []

Section: Camera Pose Estimation
We evaluate camera pose estimation on the challenging MVSEC nighttime sequences using standard metrics (ATE, RPE trans, RPE rot; lower is better), following a consistent zero-shot protocol (trained on outdoor_day2) as in our depth experiments.
As shown in Tab. 2, RGB-only baselines such as DUSt3R fail under extreme low-light conditions, while MonST3R offers improved results. Fine-tuning MonST3R leads to substantial gains, particularly in RPE trans and RPE rot, with further improvements from Easi3R variants. Despite these enhancements, our proposed EAG3R consistently achieves the best performance across most metrics and sequences. This advantage comes from EAG3R's effective use of asynchronous event data, which provides reliable motion cues even when RGB inputs are heavily degraded. As a result, EAG3R maintains robust tracking and delivers more accurate camera trajectories, highlighting the strength of event-based sensing in scenarios where conventional methods often fail. As illustrated in Fig. 4, the predicted trajectory from EAG3R exhibits lower drift and aligns more closely with the ground truth compared to DUSt3R and MonST3R, further demonstrating its superiority in precise pose estimation.
this section cite: []

Section: Dynamic Reconstruction
We evaluate dynamic 3D reconstruction on the MVSEC outdoor_night1-3 sequences. Prior methods such as DUSt3R and MonST3R serve as RGB-based baselines, with MonST3R extending pointmap prediction to dynamic scenes and Easi3R variants incorporating motion-aware masking. However, all remain limited under low-light conditions due to their reliance on degraded RGB inputs.
this section cite: []

Section: MonST3R EAG3R DUSt3R
Figure 4: Comparison of estimated camera trajectories. The predicted trajectories (solid blue) from DUS3R, MonST3R, and EAG3R are evaluated against the ground truth (dashed gray). Notably, EAG3R demonstrates a trajectory that more closely aligns with the ground truth.
Table 3: Ablation study on depth estimation performance on the Night3 sequence. Modules are incrementally added to the MonST3R baseline. Each addition improves performance, with the full EAG3R system achieving the best results.
this section cite: []

Section: Method

this section cite: []

Section: Abs Rel
↓ δ < 1.25 ↑ RMSE log ↓ MonST3R (Baseline) 0.317 0.453 0.418 MonST3R (Finetune) 0.302 0.509 0.401 + Event 0.297 0.518 0.396 + Event + LightUp 0.291 0.523 0.388 + Event + LightUp + SNR Fusion (Full) 0.288 0.533 0.371
In contrast, EAG3R directly integrates asynchronous event streams into the 4D reconstruction pipeline, allowing for improved motion handling and robustness to illumination changes. Qualitative results, provided in the appendix, show that EAG3R produces cleaner, more complete reconstructions and better preserves dynamic scene details compared to purely frame-based methods.
this section cite: []

Section: Ablation Study
To better understand the contribution of each design component in EAG3R, we conduct a systematic ablation study on the MVSEC outdoor_night3 sequence for monocular depth estimation. Starting from the MonST3R baseline, we incrementally add our proposed modules: event inputs, the LightUp enhancement network, and the SNR-aware fusion mechanism. Results are shown in Tab. 3.
Each component contributes positively to the final performance. The introduction of event streams already leads to a substantial improvement, validating the value of asynchronous visual signals in low-light scenarios. Incorporating the LightUp module provides additional gains by improving the quality of underexposed RGB inputs. Finally, the SNR-guided fusion further boosts robustness by adaptively emphasizing reliable features from either modality, particularly in noisy or degraded regions. The combination of these modules leads to the strongest performance, confirming the effectiveness of our full EAG3R design.
this section cite: []

Section: Conclusion
We presented EAG3R, a event-augmented framework for robust 3D geometry estimation under dynamic and low-light conditions. Built on the MonST3R backbone, EAG3R introduces a lightweight event adapter and a retinex-inspired light-up module, an SNR-aware fusion mechanism, and an eventbased photometric consistency loss. These components enable reliable depth and pose estimation where conventional RGB-only methods struggle. EAG3R achieves strong zero-shot generalization to nighttime scenes, consistently outperforming state-of-the-art baselines in depth, camera pose estimation, and dynamic reconstruction tasks. Our results highlight the value of integrating asynchronous event signals into geometry pipelines. We discuss limitations and broader impact in the appendix.
this section cite: []

Section: References
Ref_id:b0 Title: Building Rome in a Day Year: (2011)
Ref_id:b1 Title: Bundle adjustment in the large Year: (2010)
Ref_id:b2 Title: A histogram modification framework and its application for image contrast enhancement Year: (2009)
Ref_id:b3 Title: E3D: event-based 3D shape reconstruction Year: (2020)
Ref_id:b4 Title: Speeded-up robust features (SURF) Year: (2008)
Ref_id:b5 Title: Scene coordinate reconstruction: Posing of image collections via incremental learning of a relocalizer Year: ()
Ref_id:b6 Title: Retinexformer: One-stage Retinex-based transformer for low-light image enhancement Year: (2023)
Ref_id:b7 Title: Event-based 3D reconstruction from neuromorphic retinas Year: (2013)
Ref_id:b8 Title: Dense voxel 3D reconstruction using a monocular event camera Year: (2023)
Ref_id:b9 Title: Easi3R: Estimating disentangled motion from DUSt3R without training Year: (2025)
Ref_id:b10 Title: MonoSLAM: Real-time single camera SLAM Year: (2007)
Ref_id:b11 Title: SuperPoint: Self-supervised interest point detection and description Year: (2018)
Ref_id:b12 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b13 Title: Direct sparse odometry Year: (2017)
Ref_id:b14 Title: LSD-SLAM: Large-scale direct monocular SLAM Year: (2014)
Ref_id:b15 Title: InstantSplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds Year: (2024)
Ref_id:b16 Title: Event-based vision: A survey Year: (2022)
Ref_id:b17 Title: EasySplat: View-adaptive learning makes 3D gaussian splatting easy Year: (2025)
Ref_id:b18 Title: Video to events: Recycling video datasets for event cameras Year: (2020-06)
Ref_id:b19 Title: Asynchronous, photometric feature tracking using events and frames Year: (2018)
Ref_id:b20 Title: LIME: Low-light image enhancement via illumination map estimation Year: (2016)
Ref_id:b21 Title: Event-3DGS: Event-based 3D reconstruction using 3D Gaussian Splatting Year: (2024)
Ref_id:b22 Title: Mask R-CNN Year: (2017)
Ref_id:b23 Title: Generating consistent long depth sequences for open-world videos Year: (2024)
Ref_id:b24 Title: IncEventGS: Pose-free Gaussian Splatting from a single event camera Year: (2024)
Ref_id:b25 Title: Event-based low-illumination image enhancement Year: (2023)
Ref_id:b26 Title: Stereo4D: Learning how things move in 3D from internet stereo videos Year: (2024)
Ref_id:b27 Title: Real-time 3D reconstruction and 6-DOF tracking with an event camera Year: (2016)
Ref_id:b28 Title: Neural radiance fields from a moving event camera Year: (2023)
Ref_id:b29 Title: Robust consistent video depth estimation Year: ()
Ref_id:b30 Title: Event-based structured light for depth reconstruction using frequency tagged light patterns Year: (2018)
Ref_id:b31 Title: MegaSaM: accurate, fast, and robust structure and motion from casual dynamic videos Year: (2024)
Ref_id:b32 Title: Towards robust event-guided low-light image enhancement: A large-scale real-world event-image dataset and novel approach Year: (2024)
Ref_id:b33 Title: Low-light video enhancement with synthetic event guidance Year: (2023)
Ref_id:b34 Title: SLAM3R: real-time dense scene reconstruction from monocular RGB videos Year: (2024)
Ref_id:b35 Title: Distinctive image features from scale-invariant keypoints Year: (2004)
Ref_id:b36 Title: Consistent video depth estimation Year: (2020)
Ref_id:b37 Title: ORB-SLAM: a versatile and accurate monocular SLAM system Year: (2015)
Ref_id:b38 Title: MAS13R-SLAM: real-time dense SLAM with 3D reconstruction priors Year: (2024)
Ref_id:b39 Title: Stereo depth from events cameras: Concentrate and focus on the future Year: (2022)
Ref_id:b40 Title: DTAM: Dense tracking and mapping in real-time Year: (2011)
Ref_id:b41 Title: Asynchronous stereo vision for event-driven dynamic stereo sensor using an adaptive cooperative approach Year: (2013)
Ref_id:b42 Title: UniDepth: universal monocular metric depth estimation Year: ()
Ref_id:b43 Title: Self-calibration and metric reconstruction inspite of varying and unknown intrinsic camera parameters Year: (1999)
Ref_id:b44 Title: Visual modeling with a hand-held camera Year: (2004)
Ref_id:b45 Title: Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer Year: (2020)
Ref_id:b46 Title: EMVS: Eventbased multi-view stereo-3D reconstruction with an event camera in real-time Year: (2018)
Ref_id:b47 Title: EventNeRF: Neural radiance fields from a single colour event camera Year: (2023)
Ref_id:b48 Title: SuperGlue: Learning feature matching with graph neural networks Year: (2020)
Ref_id:b49 Title: Structure-from-Motion Revisited Year: (2016)
Ref_id:b50 Title: Learning temporally consistent video depth from video diffusion priors Year: (2024)
Ref_id:b51 Title: Splatt3R: Zero-shot gaussian splatting from uncalibrated image pairs Year: (2024)
Ref_id:b52 Title: Photo Tourism: exploring photo collections in 3D Year: (2006)
Ref_id:b53 Title: Modeling the world from internet photo collections Year: (2008)
Ref_id:b54 Title: Dynamic point maps: A versatile representation for dynamic 3D reconstruction Year: (2025)
Ref_id:b55 Title: MV-DUSt3R+: single-stage scene reconstruction from sparse views in 2 seconds Year: (2024)
Ref_id:b56 Title: RAFT: Recurrent all-pairs field transforms for optical flow Year: (2020)
Ref_id:b57 Title: Droid-SLAM: Deep visual SLAM for monocular, stereo, and RGB-D cameras Year: ()
Ref_id:b58 Title: 3D reconstruction with spatial memory Year: (2024)
Ref_id:b59 Title: Vggt: Visual geometry grounded transformer Year: (2025)
Ref_id:b60 Title: VGGsfM: Visual geometry grounded deep structure from motion Year: ()
Ref_id:b61 Title: Continuous 3D perception model with persistent state Year: (2025)
Ref_id:b62 Title: Underexposed photo enhancement using deep illumination estimation Year: (2019)
Ref_id:b63 Title: DUSt3R: geometric 3D vision made easy Year: ()
Ref_id:b64 Title: Sea-RAFT: Simple, efficient, accurate RAFT for optical flow Year: ()
Ref_id:b65 Title: Deep Retinex decomposition for low-light enhancement Year: (2018)
Ref_id:b66 Title: R: dynamics-aware gaussian splatting for static scene reconstruction Year: (2024)
Ref_id:b67 Title: SNR-Aware low-light image enhancement Year: (2022)
Ref_id:b68 Title: Depth Anything: Unleashing the power of large-scale unlabeled data Year: ()
Ref_id:b69 Title: Event camera data dense pre-training Year: (2023)
Ref_id:b70 Title: DS-SLAM: A semantic visual SLAM towards dynamic environments Year: (2018)
Ref_id:b71 Title: MonST3R: a simple approach for estimating geometry in the presence of motion Year: (2024)
Ref_id:b72 Title: Learning to see in the dark with events Year: (2020)
Ref_id:b73 Title: Structure and motion from casual videos Year: ()
Ref_id:b74 Title: Exploiting dense point trajectories for localizing moving cameras in the wild Year: ()
Ref_id:b75 Title: Deep learning for event-based vision: A comprehensive survey and benchmarks Year: (2023)
Ref_id:b76 Title: Realtime time synchronized event-based stereo Year: (2018)
Ref_id:b77 Title: The multivehicle stereo event camera dataset: An event camera dataset for 3D perception Year: (2018)
Ref_id:b78 Title: DEVO: Depthevent camera visual odometry in challenging conditions Year: (2022)
