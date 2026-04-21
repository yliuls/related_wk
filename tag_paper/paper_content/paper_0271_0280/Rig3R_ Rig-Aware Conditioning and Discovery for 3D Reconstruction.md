Title: Rig3R: Rig-Aware Conditioning for Learned 3D Reconstruction
Abstract: Estimating agent pose and 3D scene structure from multi-camera rigs is a central task in embodied AI applications such as autonomous driving. Recent learned approaches such as DUSt3R have shown impressive performance in multiview settings. However, these models treat images as unstructured collections, limiting effectiveness in scenarios where frames are captured from synchronized rigs with known or inferable structure. To this end, we introduce Rig3R, a generalization of prior multiview reconstruction models that incorporates rig structure when available, and learns to infer it when not. Rig3R conditions on optional rig metadata including camera IDs, timestamp, and rig calibrations to develop a rig-aware latent space that remains robust to missing information. It jointly predicts global pointmaps and two types of raymaps: a pose raymap relative to a global frame, and a rig raymap relative to a rig-centric frame consistent across time. The global pose raymaps allow the model to reason about the agent's ego-motion, while the rig raymaps allow the model to infer rig structure directly from input images when metadata is missing. Rig3R achieves state-of-the-art performance in 3D reconstruction, camera pose estimation, and rig discovery-outperforming both traditional and learned methods by 17-45% mAA across diverse real-world rig datasets, all in a single forward pass without post-processing or iterative refinement.

Section: 
Figure 1: Rig3R is the first learned 3D vision model to leverage rig constraints when available, and the first method to support rig calibration discovery from unordered images when they are not-achieving strong 3D consistency and performance across diverse settings and rig configurations.
this section cite: []

Section: Introduction
Multi-view scene estimation of camera poses and 3D structure from images is a core capability in computer vision and has enabled spatial understanding for embodied agents, robotic systems, and large-scale visual localization [1]. Accurate estimation of structure and motion is essential for tasks such as simultaneous localization and mapping (SLAM) [2,3], scene relocalization and view synthesis applications [4][5][6][7]. Traditional pipelines based on Structure-from-Motion (SfM) and Multi-View Stereo (MVS) reconstruct scenes by optimizing for geometry via matched image features [8][9][10]. While effective in controlled settings, these methods are brittle in the presence of dynamic objects, visual repetition, or feature poor environments, and often require careful tuning.
Recent learned methods such as DUSt3R [11] have shown impressive capabilities in multiview 3D reconstruction, with many successors [12,13] extending this to single-pass inference. A limitation, however, is that these works treat images as unstructured collections. This overlooks a key structural prior common in real-world applications: images are often captured using synchronized multi-camera rigs with fixed relative configurations. Rig metadata-such as camera ID, timestamp, and relative poses-can provide valuable cues, especially when field-of-view overlap is limited, as is often the case in rig setups. While classical pipelines can exploit this structure [14][15][16], feedforward models currently leave it untapped.
We introduce Rig3R, a transformer-based model for multiview 3D reconstruction and pose estimation that leverages rig metadata when available and learns to infer rig structure when it is not. Rig3R handles unstructured image sets, calibrated rigs, and everything in between, predicting dense pointmaps and raymaps for each image in a single forward pass. These raymaps spatially encode camera intrinsics and extrinsics, which can be recovered in closed form-even in ambiguous regions such as sky or dynamic pixels. To enable this flexibility, Rig3R combines metadata embeddings with dropout training, and includes a dedicated rig prediction head that infers rig structure directly from image content when metadata is unavailable. This allows Rig3R to produce accurate reconstructions, even in cases with minimal view overlap.
Our key contributions are:
• The first learned method that leverages rig constraints to improve 3D reconstruction and pose estimation, while generalizing to inputs with partial or missing metadata (e.g., camera ID, timestamp, rig poses).
• A novel output representation based on global and rig-relative raymaps, enabling closed-form pose estimation and rig structure discovery from unordered image inputs.
• Extensive experiments across diverse real-world driving datasets show that Rig3R achieves state-of-the-art performance in 3D reconstruction, camera pose estimation, and rig discovery, outperforming both traditional and learned methods, all in a single forward pass.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15']

Section: Related Works
Multi-View 3D Reconstruction. Classical pipelines follow a two-stage paradigm: Structure-from-Motion (SfM) for sparse pose and point recovery, followed by Multi-View Stereo (MVS) for densification. Systems like COLMAP [8,17,18] rely on feature matching, triangulation, and bundle adjustment, but remain sensitive to occlusion, motion, and low-texture regions [1]. Early learning-based methods improved robustness by introducing learned features and matching [19][20][21][22][23]. Photometric representations such as NeRFs [4,24,25] and Gaussian splats [5,26] reconstruct scenes via view synthesis but typically require accurate camera poses. More recent approaches bypass both explicit feature matching and known poses, predicting 3D structure directly from RGB images [27][28][29][30]. DUSt3R [11] pioneered pointmap regression from single image pairs without known poses, with follow-up works addressing multi-frame input [31][32][33], dynamic scenes [33][34][35], and downstream tasks [36][37][38]. MV-DUSt3R [13] introduces multi-frame attention, while Fast3R [12] scales to hundreds of views with global consistency. VGGT [39] jointly predicts depth, pose, and structure using a transformer backbone. Pow3R [40] improves flexibility through lightweight conditioning on inputs such as intrinsics, relative pose, or depth. While these models support efficient scene understanding, they treat input views as unordered. Rig3R builds on this single-pass design, conditioning on rig metadata and enabling structure discovery even in the absence of such priors.
this section cite: ['b7', 'b16', 'b17', 'b0', 'b18', 'b19', 'b20', 'b21', 'b22', 'b3', 'b23', 'b24', 'b4', 'b25', 'b26', 'b27', 'b28', 'b29', 'b10', 'b30', 'b31', 'b32', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b12', 'b11', 'b38', 'b39']

Section: Camera Pose Estimation.
Traditional pose estimation relies on geometric solvers such as PnP with RANSAC [41,42] and global optimization via bundle adjustment [9], but performance is brittle under occlusion, dynamic motion, or sparse correspondences. Learned methods such as PoseNet [43] regress 6-DoF poses directly from images [44][45][46], while unsupervised approaches [47] optimize photometric losses to jointly estimate depth and ego-motion. Systems like DROID-SLAM [48] combine differentiable updates with learned features for increased robustness, and several related method adopt a learned SLAM approach [49][50][51]. Several 3D reconstruction models, including those discussed above, also infer poses alongside or through 3D structure [11,52,39,12,30,13,53]. Rig3R extends this trend by predicting dense raymaps that encode per-pixel directions and camera centers, enabling closed-form recovery of intrinsics and extrinsics while enforcing multiview consistency.
this section cite: ['b40', 'b41', 'b8', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b10', 'b51', 'b38', 'b11', 'b29', 'b12', 'b52']

Section: Rig-Aware Multi-View Geometry.
Rig constraints can provide strong geometric cues for multi-view reconstruction, enabling more accurate pose estimation and improved robustness in low-overlap or ambiguous settings. Classical works have leveraged such constraints in various ways. COLMAP [8] incorporates rig structure by modeling the rig as a single moving entity, jointly optimizing global poses while keeping intra-rig calibration fixed through bundle adjustment. Kaess and Dellaert [14] introduced a probabilistic SLAM framework for multi-camera rigs that models cross-camera feature associations under motion. Carrera et al. [16] proposed a SLAM-based method for fully automatic extrinsic calibration of multi-camera systems, even with non-overlapping fields of view. Heng et al. [15] developed an infrastructure-based calibration method using image-based localization and prebuilt maps, requiring no manual intervention. Rig3R takes a markedly different approach from these prior works, providing rig information as optional embeddings on the input, enabling accurate and generalizable 3D reconstruction across both structured and unstructured multi-camera configurations.
this section cite: ['b7', 'b13', 'b15', 'b14']

Section: Rig-aware 3D Reconstruction
We address the task of predicting 3D structure and camera poses from a set of N RGB images {I i } N i=1 , where I i ∈ R 3×H×W . Inputs may range from unordered image collections to temporally distributed views captured by multi-camera rigs. Each image may optionally include metadata M i = {c i , t i , r i }, where c i is a camera ID, t i is a timestamp, and r i is a rig-relative raymap encoding the camera pose. Each metadata field is optional and may be independently omitted during training and inference.
Given inputs {I i , M i } N i=1 , the model predicts for each image: a pointmap P i ∈ R 3×H×W representing per-pixel 3D coordinates in the first image's frame; a confidence map C i ∈ R H×W used to weight the pointmap loss; a pose raymap R pose i ∈ R H×W ×6 encoding camera parameters relative to the first image's frame; and a rig raymap R rig i ∈ R H×W ×6 encoding camera parameters relative to a rig-centric frame, decoupled from ego-motion. Together, these outputs form Rig3R's predictions:
Rig3R : {I i , M i } N i=1 → {P i , C i , R pose i , R rig i } N i=1
The following sections detail our raymap representation, metadata encoding, and model architecture.
this section cite: []

Section: Raymap Representation
Rig3R outputs dense raymaps from both its pose and rig heads. A raymap is a directional field that assigns a unit ray direction to each pixel, with all rays originating from a shared camera center. This representation encodes both camera intrinsics and pose in a unified, geometrically consistent format.
For each pixel (u, v), the viewing ray ruv ∈ S 2 is computed as ruv = R • K -1 [u, v, 1] ⊤ , where K ∈ R 3×3 is the intrinsic matrix, R ∈ SO(3) is the rotation matrix, and the output is unit-normalized. All rays share a common camera center c ∈ R 3 .
Raymaps offer key advantages over alternative representations. They provide spatially aligned, per-pixel supervision and serve as a stable signal even in ambiguous regions such as sky or dynamic objects. Unlike pointmaps, which infer pose indirectly through 3D predictions and often fail in such regions, raymaps offer a more direct and consistent representation for pose estimation. They also encode interpretable geometry, enabling closed-form recovery of camera intrinsics and extrinsics from ray directions and pixel distances. In our implementation, we recover focal lengths using angular constraints derived from pixel-ray correspondences, and estimate rotations in closed form using SVD from aligned camera and world rays [54]. See Section B of supplementary material for more details.
this section cite: ['b53']

Section: Rig-Aware Metadata
Each image may optionally be associated with a metadata tuple M i = {c i , t i , r i }, where c i is a discrete camera identifier shared by all images from the same physical camera, t i ∈ R is a continuous timestamp normalized in seconds, and r i ∈ R H×W ×6 is a rig-relative raymap encoding the camera's pose within the rig. This metadata provides geometric and temporal context for multiview reasoning. The combination of camera ID and timestamp forms a structured decomposition of frame identity, offering strong cues for spatiotemporal alignment. All metadata fields are optional and can be independently dropped during training to encourage robustness to missing information.
this section cite: []

Section: Model Architecture
Image Encoder. Rig3R (see Fig. 2) employs a shared ViT-Large encoder [55,56] to independently patchify and encode each input image using 2D sine-cosine positional encodings. We initialize from DUSt3R [11], though other works [39,13,12] indicate that performance is not sensitive to this choice.
this section cite: ['b54', 'b55', 'b10', 'b38', 'b12', 'b11']

Section: Metadata Embedding.
Each patch is optionally augmented with rig-aware metadata: (1) frame index N , (2) camera ID c i , (3) timestamp t i , and (4) rig raymap patch r i . The discrete IDs N and c i are randomly sampled from a larger index range and encoded using 1D sine-cosine embeddings, following [12], enabling generalization to varying numbers of frames and cameras. The timestamp t i ∈ R is normalized in seconds and encoded similarly. The rig raymap patch r i ∈ R 6 is linearly projected to the model dimension. All components are concatenated and added to the patch tokens. During training, c i , t i , and r i are randomly dropped out to promote robustness, while the frame index N is always included to uniquely identify each image within the transformer.
Transformer Decoder. Patch tokens from all images are passed to a second ViT-Large transformer, trained from scratch, that performs joint self-attention across the full set. This enables Rig3R to aggregate information across views and time, conditioned on metadata when available. Unlike the shared image encoder, the decoder fuses multiview features in a shared latent space.
Prediction Heads. Rig3R uses three multitask heads: one for pointmap prediction and two for raymaps (pose-relative and rig-relative), with shared weights across frames. The pointmap head is a DPT module [57] that predicts a 3D pointmap P i ∈ R 3×H×W and a confidence map C i ∈ R H×W . Each raymap head consists of two MLPs: one predicts per-pixel ray directions, and the other predicts a global camera center via average pooling over patch tokens. This design avoids dedicated query tokens and ensures all gradients flow through the patch tokens, promoting coherence. The prediction head design follows the nature of the output: pointmaps capture fine geometric detail and benefit from the multi-scale DPT head, while raymaps encode global pose information and are efficiently modeled with lightweight MLPs. These three outputs are tightly coupled: pointmaps are expected to lie along rays defined by the pose raymap, and rig and pose raymaps are related through ego-motion. This multitask formulation acts as a structural prior, improving consistency and generalization across diverse multiview settings.
this section cite: ['b11', 'b56']

Section: Data Sampling
To enable Rig3R to reason about diverse rig configurations, the data sampling process plays a critical role. Unlike prior learned methods that treat multi-view inputs as completely unstructured collections of images as shown in Fig. 3A, Rig3R's design acknowledges that real-world embodied system often use synchronized multi-camera rigs with implicit spatial consistency. When introducing rig-aware conditioning, one sampling approach is to explicitly structure data as a C × T tensor, where C is the number of cameras and T is the number of temporal steps, allowing the model to learn from complete spatiotemporal grids as shown in Fig. 3B. However, this formulation restricts flexibility, as it assumes fixed rig configurations and uniform frame availability. Instead, Rig3R flattens all frames into a single sequence as demonstrated in Fig. 3C, preserving the unordered format of prior models while embedding each frame with its own rig metadata (camera ID, timestamp, and rig raymap). This design unifies rig information and unstructured sampling within a single representation, making it a truly general sampling choice; this is conceptually equivalent to learning over dense C × T configurations with stochastic dropout across cameras and timesteps. As a result, Rig3R receives a stronger and more varied learning signal, exposed to all possible sub-rigs and partial observations during training, keeping it flexible to missing views, variable rig geometries, and asynchronous capture setups.
this section cite: []

Section: Training
Training Losses. We train with a multitask loss over pointmaps, pose raymaps, and rig raymaps:
L total = L pmap + λ p L p_rmap + λ r L r_rmap ,
where λ p and λ r are weighting terms for the pose and rig raymap losses.
The pointmap loss L pmap , following [11], is a confidence-weighted regression objective [58,59] with scale-normalized ground truth. For frame v, depth-normalized pointmap error is:
L pmap = i∈D v C v i X v i - 1 z Xv i -α log C v i ,
where X v i is the predicted 3D point at pixel i, Xv i is the ground truth, C v i is the predicted confidence, z is the average scene depth used for normalization, and α is the weight of the regularization term.
The raymap loss L rmap includes terms for both ray directions and camera centers:
L rmap = h,w ∥r v,h,w -rv,h,w ∥ + β c v - 1 z cv .
this section cite: ['b10', 'b57', 'b58']

Section: Rig3R Rig3R
Unstr Calib Here, r v,h,w is the predicted unit ray direction at pixel (h, w), and c v is the predicted camera center for frame v. The ground-truth ray direction and camera center are denoted by rv,h,w and cv , respectively. The average scene depth z is used for scale normalization, and β weights the center loss term. Following [39], we find training is more stable when the model learns scale and direction norms directly-especially for camera centers near the origin.
this section cite: ['b38']

Section: DUSt3R-GA Fast3R
Training Data. We train Rig3R on a diverse data mix: CO3D-v2 [60], BlendedMVS [61], Mapfree [62], ScanNet++ v2 [63], MVImgNet [64], PointOdyssey [65], Virtual KITTI2 [66], TartanAir V2 [67], PandaSet [68], KITTI [69], Argoverse2 [70], nuScenes [71], Waymo [72], and an internal dataset. We process relevant driving datasets following [73]. These cover a broad range of scene types-including indoor, driving, synthetic, and object-centric-with an emphasis on data from multi-camera rigs. For COLMAP datasets, we sample images based on covisibility. For others, we use a random stride within a specified range. In rig-based datasets, we subsample the rig cameras per sequence to increase diversity in rig configurations. Where available, the front-facing camera is always included to ensure overlap and reflect common monocular setups.
this section cite: ['b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b65', 'b66', 'b67', 'b68', 'b69', 'b70', 'b71', 'b72']

Section: Embedding Dropout.
To encourage metadata-aware reasoning and improve robustness, we randomly drop each metadata field (camera ID, timestamp, rig pose) with 50% probability during training. This structured masking teaches the model to leverage metadata when available, and to infer missing context from image content and cross-view relationships when it is not-enabling generalization to diverse input configurations at inference time.
this section cite: []

Section: Training Details.
Rig3R is trained on 24-frame samples with a batch size of 128, using 128 H100 GPUs for 250k steps over 5 days. Images are resized to 512 × 512 with padding. We apply data augmentations including random per-frame color jitter, Gaussian blur, and centered aspect-ratio crops to simulate variation in focal length and image shape. During training, input sequences are randomly shuffled to vary the reference frame and promote generalization. We use the AdamW optimizer with a learning rate of 0.0001 and cosine annealing. While training uses a fixed 24-frame input for consistency and comparability across baselines, all frame IDs other than the first frame are randomly sampled from a shared ID pool from 1 to 100, enabling the model to generalize to sequences much longer than 24 frames.
this section cite: []

Section: Experiments Evaluation Data.
We evaluate Rig3R on the Waymo Open [72] validation set and WayveScenes101 [74], both featuring 5-camera rigs and approximately 200 timesteps per scene at 10 FPS under diverse real-world driving conditions. Waymo provides LiDAR-based ground-truth poses and 3D points, while WayveScenes101 uses COLMAP reconstructions. For each scene, we extract two 24-frame samples, each using the full 5-camera rig spaced approximately 2 seconds apart. Table 2: Multi-view pointmap estimation results. We report accuracy and completeness, and their average as the Chamfer distance. Intrinsics are withheld for all methods.
Baselines. We compare Rig3R to both learned and classical baselines for multi-view 3D reconstruction and pose estimation. MV-DUSt3R [13] and Fast3R [12] are architecturally similar to Rig3R, using transformer-based, feedforward multi-view inference. DUSt3R-GA [11] predicts stereo pointmaps refined via global optimization. As a classical baseline, we evaluate COLMAP [8] in both unstructured and rig-aware modes. While learned models operate per sample, we allow COLMAP to process full scenes for stronger global context. The COLMAP ground truth in WayveScenes101 is generated from full, dense sequences, whereas our evaluations use strided sequences with larger baseline shifts and sparser keypoint overlap. To ensure fairness and consistency, learned models operate on 24-frame chunks as in training, while COLMAP is permitted to process the entire strided sequence jointly, leveraging its inherent ability to optimize globally across frames. We evaluate two variants of Rig3R: Rig3R Unstr , which receives no metadata and treats the sequence as unstructured, and Rig3R Calib , which is given full rig metadata (camera ID, timestamp, and rig raymaps). Note that Rig3R Calib and Rig COLMAP are the only methods that leverage rig constraints, and both receive the same rig calibration information.
this section cite: ['b71', 'b73', 'b12', 'b11', 'b10', 'b7']

Section: Camera Pose Estimation
We evaluate pose estimation using relative rotation and translation accuracy (RRA, RTA) at 15°and 5°thresholds, along with mean average accuracy (mAA) over thresholds up to 30° [75][76][77]. These metrics capture both coarse correctness and fine-grained precision. Full results are shown in Table 1.
On Waymo, Rig3R Calib achieves the best performance (82.1 mAA), and maintains high precision even at 5°thresholds. Rig3R Unstr ranks second overall, despite lacking rig metadata. Learned baselines exhibit sharp drops at 5°. COLMAP improves with rig constraints, and shows smaller differences between thresholds, consistent with classical optimization's binary convergence behavior. We also report wall-clock time per method to estimate poses. Qualitative results in Fig. 4 show Rig3R's improved spatial consistency-especially with embeddings-even under large spatial displacement.
We also evaluate on WayveScenes101, an unseen dataset with a novel rig configuration constructed from scenes where COLMAP reconstruction succeeded-potentially favoring classical methods. Despite this, Rig3R Calib achieves the best performance across all metrics, followed by DUSt3R-
1 Camera Rig 3 Camera Rig 5 Camera Rig 7 Camera Rig GA, which benefits from global optimization. Rig3R Unstr remains competitive, outperforming all feedforward baselines and performing comparably to Rig COLMAP. Rig3R Unstr remains comparable to other baselines, displaying Rig3R's flexibile performance even when rig metadata is missing. Crucially, the large improvement in performance from Rig3R Unstr to Rig3R Calib indicates the value of leveraging rig metadata, suggesting it can enable stronger generalization.
These results confirm that Rig3R Calib achieves the strongest accuracy and precision across datasets, while Rig3R Unstr remains robust without rig metadata. Rig-aware embeddings provide a powerful mechanism for generalization and fine-grained pose estimation in diverse multi-view settings.
this section cite: ['b74', 'b75', 'b76']

Section: Pointmap Estimation
We evaluate 3D reconstruction quality using pointmap accuracy (Acc.), completeness (Comp.), and Chamfer distance (average of the two). Metrics are computed over sparse pointclouds by masking both predictions and ground-truth to valid regions (see Table 2).
On Waymo, Rig3R Calib achieves the lowest error across all metrics. Rig3R Unstr follows closely, significantly outperforming all other baselines and confirming the strength of Rig3R's pointmap predictions even without metadata. Fig. 4 highlights Rig3R's improvements over baselines in 3D reconstruction quality. In particular, with rig-aware embeddings, the model confidently reconstructs side-view cameras with minimal overlap, where geometric cues alone are often insufficient.
On WayveScenes101, Rig3R Calib again leads, demonstrating robust generalization. Interestingly, Fast3R achieves the second-best Chamfer distance, slightly outperforming Rig3R Unstr despite lower pose accuracy. This highlights a key advantage of Rig3R: it estimates pose directly from raymaps rather than pointmaps, enabling more consistent multi-view reasoning. See Section C of the supplementary material for further discussion on pose inference from raymaps versus pointmaps.
These results show that Rig3R learns robust scene structure with strong spatial precision and completeness, and that rig metadata further improves reconstruction quality and generalization.
this section cite: []

Section: Generalization Across Rig Configurations
We assess Rig3R's ability to generalize across rig configurations on the Argoverse2 [70] validation set. We subsample 1, 3, 5, and 7 camera rigs, increasing strides to maintain scene coverage (Fig. 5).
this section cite: ['b69']

Section: Rig Calibration Discovery.
Figure 6a shows results for Rig3R Unstr , which discovers rig calibrations directly from unordered images-without any rig metadata or assumptions about camera configuration. We evaluate two rig-specific metrics: rig ID accuracy and rig-relative pose mAA. Rig ID accuracy reflects how well frames from the same camera are grouped together; we compute it by first clustering the rig raymap outputs, and then evaluating frame assignment accuracy via the Hungarian algorithm. Rig mAA then measures the quality of these predicted clusters, evaluating how accurate the relative orientations and positions of the discovered cameras are in a rig-centeric frame.
Rig3R Unstr achieves strong rig mAA and Rig ID accuracy across all configurations. Performance remains strong even as spatial layout becomes more complex, confirming Rig3R's ability to discover diverse rig structures without supervision. To the best of our knowledge, this is the first attempt, learned or classical, to address rig discovery with unordered images and no timestamps. Notably, Rig3R also handles monocular and unordered inputs by predicting identity rig raymaps and producing a single cluster at inference-correctly signaling the absence of a rig. Fig. 5 visualizes various predicted rig configurations, with camera clusters color-coded across time.
Flexible Rig Performance. We evaluate performance across rig sizes using pose mAA and Chamfer distance (Fig. 6b-c) for Rig3R Calib , Rig3R Unstr , and Fast3R. Rig3R Calib performs best overall, maintaining high pose and reconstruction quality across all settings. Rig3R Unstr also performs strongly, despite receiving no metadata. In contrast, Fast3R degrades as rig size and stride increase-likely due to reduced image overlap, which challenges methods that rely solely on visual correspondence without rig context. These results demonstrate that Rig3R remains robust to spatial variation and generalizes well across diverse rig configurations, as can be seen in Fig. 5.
this section cite: []

Section: Ablation Studies
Metadata Embeddings. In Table 3, we ablate the contribution of each metadata field-camera ID, timestamp, and rig pose-on pose estimation across Waymo and WayveScenes101, which represent previously seen and unseen rig configurations, respectively.
On Waymo, we observe that camera ID and rig pose embeddings provide only modest gains over the unstructured baseline, likely because the rig structure is easily recognized in this familiar setting. In this context, timestamp proves particularly valuable, as it provides dynamic cues for localizing the rig over time. Since the model already reasons about the rig structure implicitly, temporal information becomes the most informative remaining signal.
On WayveScenes101, a previously unseen dataset with a novel rig configuration, we find that camera ID and timestamp embeddings offer only limited gains over the unstructured baseline. This is likely because neither field alone disambiguates the underlying rig layout: camera ID does not indicate motion over time, and timestamp alone does not reveal which camera captured each frame. Classical methods like COLMAP similarly require both to accurately infer rig structure and optimize poses. In contrast, rig pose embeddings provides a direct spatial signal about the novel rig configuration-crucial for reasoning under domain shift. With this input, Rig3R can recognize and adapt to unseen rig geometries, resulting in a substantial boost in generalization and performance (mAA improves from 25.7 to 56.4). These results highlight the unique role of rig pose metadata in enabling generalization to unseen capture setups.
Across datasets, we find that timestamp embeddings primarily improve translation accuracy, as they provide strong cues about velocity and motion magnitude, while rig pose embeddings more strongly influence rotation accuracy, reinforcing consistent relative orientations between cameras within each timestamp. Providing all metadata yields the best performance across both datasets, confirming that Cam Time Rig Waymo WayveScenes101 (unseen) Table 4: Model ablation of pointmap (L pmap ) and rig (L rig ) heads on the pose raymap head (L pose ).
spatial calibration, temporal cues, and view identity are complementary. These results show that Rig3R generalizes well from partial metadata and fully benefits from rig calibration when available.
Multi-task Learning. We evaluate the impact of each auxiliary head by training three Rig3R variants: with only the pose raymap head (L pose ), with pose + rig raymap (L pose + L rig ), and with pose + pointmap (L pose + L pmap ). As this ablation requires training separate models from scratch, we perform it at reduced scale (batch size 32, 5 datasets) for computational efficiency; results are not directly comparable to full-scale evaluations. As shown in Table 4, both auxiliary heads individually improve performance in the unstructured setting. The pointmap head provides the largest gain-raising mAA from 45.8 to 53.5-reflecting the value of 3D grounding. The rig head also improves results over using only the pose raymap head, suggesting that even without metadata, it helps maintain a coherent spatial layout. In the calibrated setting, all metrics are higher across the board, indicating some degree of performance saturation. Still, the rig head yields the highest mAA, likely by reinforcing the structure provided by the rig metadata across timesteps, followed by the pointmap head also helps in this setting. Despite the reduced scale of this setup, we observe a consistent pattern: both heads improve performance, and the rig head is particularly important when rig constraints are available. Importantly, the rig head also enables Rig3R's novel ability to perform rig discovery. For these reasons, we include both auxiliary heads in the full-scale Rig3R model.
this section cite: []

Section: Conclusion
We present Rig3R, a transformer-based model for multiview 3D reconstruction and pose estimation that introduces rig-aware conditioning and rig discovery. Rig3R is the first method to leverage rig metadata in a learned setting and the first to perform rig structure discovery from completely unconstrained image inputs. It jointly predicts pointmaps, global raymaps, and rig-relative raymaps in a single forward pass, achieving strong performance through its spatially-grounded representations.
Limitations and Future Work. Rig3R's main performance limitation is data diversity and quality, particularly regarding variety in rig configurations across existing datasets. One promising direction is to incorporate augmentations that simulate diverse rigs across a continuous configuration space. Future work may also explore balancing structured, rig-based temporal sampling with unordered, general sampling to improve generalization and adaptability across capture settings. While raymaps implicitly downweight dynamic content, explicitly modeling motion could further improve robustness in highly dynamic scenes. Overall, we see rig-aware embeddings as a powerful and generalizable cue, readily applicable to existing and future transformer-based models for multiview reasoning.
this section cite: []

Section: References
Ref_id:b0 Title: Multiple View Geometry in Computer Vision Year: (2003)
Ref_id:b1 Title: ORB-SLAM2: an open-source SLAM system for monocular, stereo and RGB-D cameras Year: (2017)
Ref_id:b2 Title: Loam: Lidar odometry and mapping in real Year: (2014)
Ref_id:b3 Title: Representing scenes as neural radiance fields for view synthesis Year: (2020)
Ref_id:b4 Title: Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b5 Title: Instant neural graphics primitives with a multiresolution hash encoding Year: (2022)
Ref_id:b6 Title: Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction Year: (2021)
Ref_id:b7 Title: Structure-from-motion revisited Year: (2016-06)
Ref_id:b8 Title: Orb-slam: A versatile and accurate monocular slam system Year: (2015)
Ref_id:b9 Title: Accurate, dense, and robust multiview stereopsis Year: (2010)
Ref_id:b10 Title: Dust3r: Geometric 3d vision made easy Year: ()
Ref_id:b11 Title: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass Year: (2025-06)
Ref_id:b12 Title: Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds Year: (2024)
Ref_id:b13 Title: Probabilistic structure matching for visual slam with a multi-camera rig Year: (2010)
Ref_id:b14 Title: Leveraging image-based localization for infrastructure-based calibration of a multi-camera rig Year: (2015)
Ref_id:b15 Title: Slam-based automatic extrinsic calibration of a multi-camera rig Year: (2011)
Ref_id:b16 Title: Global structurefrom-motion revisited Year: (2024)
Ref_id:b17 Title: Fastmap: Revisiting dense and scalable structure from motion Year: (2025)
Ref_id:b18 Title: Superpoint: Self-supervised interest point detection and description Year: (2018)
Ref_id:b19 Title: Superglue: Learning feature matching with graph neural networks Year: (2020)
Ref_id:b20 Title: Back to the feature: Learning robust camera localization from pixels to pose Year: (2021)
Ref_id:b21 Title: Detector-free local feature matching with transformers Year: (2021)
Ref_id:b22 Title: Robust Dense Feature Matching Year: (2023)
Ref_id:b23 Title: Nerf in the wild: Neural radiance fields for unconstrained photo collections Year: (2021)
Ref_id:b24 Title: Das3r: Dynamics-aware gaussian splatting for static scene reconstruction Year: (2024)
Ref_id:b25 Title: MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-view Images Year: (2024-10)
Ref_id:b26 Title: Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision Year: (2024)
Ref_id:b27 Title: Learning robust visual features without supervision Year: (2024)
Ref_id:b28 Title: Depth anything: Unleashing the power of large-scale unlabeled data Year: ()
Ref_id:b29 Title: Vggsfm: Visual geometry grounded deep structure from motion Year: ()
Ref_id:b30 Title: Continuous 3d perception model with persistent state Year: (2025)
Ref_id:b31 Title: 3d reconstruction with spatial memory Year: (2024)
Ref_id:b32 Title: Monst3r: A simple approach for estimating geometry in the presence of motion Year: (2025)
Ref_id:b33 Title: Stereo4d: Learning how things move in 3d from internet stereo videos Year: (2025)
Ref_id:b34 Title: Easi3r: Estimating disentangled motion from dust3r without training Year: (2025)
Ref_id:b35 Title: Slam3r: Real-time dense scene reconstruction from monocular rgb videos Year: (2025)
Ref_id:b36 Title: Grounding image matching in 3d with mast3r, 2024 Year: ()
Ref_id:b37 Title: Mast3r-slam: Real-time dense slam with 3d reconstruction priors Year: (2024)
Ref_id:b38 Title: Vggt: Visual geometry grounded transformer Year: (2025)
Ref_id:b39 Title: Pow3r: Empowering unconstrained 3d reconstruction with camera and scene priors Year: (2025)
Ref_id:b40 Title: Epnp: An accurate o(n) solution to the pnp problem Year: (2009)
Ref_id:b41 Title: Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography Year: (1981)
Ref_id:b42 Title: Posenet: A convolutional network for real-time 6-dof camera relocalization Year: (2016)
Ref_id:b43 Title: Cameras as rays: Pose estimation via ray diffusion Year: (2024)
Ref_id:b44 Title: Reloc3r: Large-scale training of relative camera pose regression for generalizable, fast, and accurate visual localization Year: (2025)
Ref_id:b45 Title: Matching 2d images in 3d: Metric relative pose from metric correspondences Year: (2024)
Ref_id:b46 Title: Unsupervised learning of depth and ego-motion from video Year: (2017)
Ref_id:b47 Title: Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras Year: (2022)
Ref_id:b48 Title: Point-slam: Dense neural point cloud-based slam Year: ()
Ref_id:b49 Title: Nice-slam: Neural implicit scalable encoding for slam Year: (2022-06)
Ref_id:b50 Title: Splatam: Splat, track & map 3d gaussians for dense rgb-d slam Year: (2024)
Ref_id:b51 Title: Grounding image matching in 3d with mast3r Year: (2024)
Ref_id:b52 Title: Scene coordinate reconstruction: Posing of image collections via incremental learning of a relocalizer Year: (2024)
Ref_id:b53 Title: Least-squares fitting of two 3-d point sets Year: (1987)
Ref_id:b54 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b55 Title: Attention is all you need Year: (2023)
Ref_id:b56 Title: Vision transformers for dense prediction Year: (2021)
Ref_id:b57 Title: Modelling uncertainty in deep learning for camera relocalization Year: (2016)
Ref_id:b58 Title: Capturing the geometry of object categories from video supervision Year: (2018)
Ref_id:b59 Title: Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction Year: (2021)
Ref_id:b60 Title: Blendedmvs: A large-scale dataset for generalized multi-view stereo networks Year: (2020)
Ref_id:b61 Title: Map-free visual relocalization: Metric pose relative to a single image Year: (2022)
Ref_id:b62 Title: Scannet: Richly-annotated 3d reconstructions of indoor scenes Year: (2017)
Ref_id:b63 Title: Mvimgnet: A large-scale dataset of multi-view images Year: ()
Ref_id:b64 Title: Pointodyssey: A large-scale synthetic dataset for long-term point tracking Year: (2023)
Ref_id:b65 Title: Virtual kitti 2 Year: (2020)
Ref_id:b66 Title: Tartanair: A dataset to push the limits of visual slam Year: (2020)
Ref_id:b67 Title: Pandaset: Advanced sensor suite dataset for autonomous driving Year: (2021)
Ref_id:b68 Title: Are we ready for autonomous driving? the kitti vision benchmark suite Year: (2012)
Ref_id:b69 Title: Argoverse 2: Next generation datasets for self-driving perception and forecasting Year: (2023)
Ref_id:b70 Title: nuscenes: A multimodal dataset for autonomous driving Year: (2019)
Ref_id:b71 Title: Scalability in perception for autonomous driving: Waymo open dataset Year: (2020)
Ref_id:b72 Title: Omni urban scene reconstruction Year: (2024)
Ref_id:b73 Title: Wayvescenes101: A dataset and benchmark for novel view synthesis in autonomous driving Year: (2024)
Ref_id:b74 Title: Posediffusion: Solving pose estimation via diffusion-aided bundle adjustment Year: (2024)
Ref_id:b75 Title: Relpose: Predicting probabilistic relative rotation for single objects in the wild Year: (2022)
Ref_id:b76 Title: Image matching across wide baselines: From paper to practice Year: (2021)
Ref_id:b77 Title: A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem Year: ()
