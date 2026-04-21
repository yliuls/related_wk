Title: DEPTH ANYTHING 3: RECOVERING THE VISUAL SPACE FROM ANY VIEWS
Abstract: 

Section: ABSTRACT
We present Depth Anything 3 (DA3), a model that predicts spatially consistent geometry from an arbitrary number of visual inputs, with or without known camera poses. In pursuit of minimal modeling, DA3 yields two key insights: a single plain transformer (e.g., vanilla DINO encoder) is sufficient as a backbone without architectural specialization, and a singular depth-ray prediction target obviates the need for complex multi-task learning. Through our teacher-student training paradigm, the model achieves a level of detail and generalization on par with Depth Anything 2 (DA2). We establish a new visual geometry benchmark covering camera pose estimation, any-view geometry and visual rendering. On this benchmark, DA3 sets a new state-of-the-art across all tasks, surpassing prior SOTA VGGT by an average of 35.7% in camera pose accuracy and 23.6% in geometric accuracy. Moreover, it outperforms DA2 in monocular depth estimation. All models are trained exclusively on public academic datasets.
this section cite: []

Section: INTRODUCTION
The ability to perceive and understand 3D spatial information from visual input is a cornerstone of human spatial intelligence (Arterberry and Yonas, 2000) and a critical requirement for applications like robotics and mixed reality. This fundamental capability has inspired a wide array of 3D vision tasks, including monocular depth estimation, Structure from Motion (Snavely et al., 2006), Multi-View Stereo (Seitz et al., 2006) and Simultaneous Localization and Mapping (Mur-Artal et al., 2015). Despite the strong conceptual overlap between these tasks-often differing by only a single factor, such as the number of input views-the prevailing paradigm has been to develop highly specialized models for each one. While recent efforts (Wang et al., 2024c;2025a) have explored unified models to address multiple tasks simultaneously, they typically suffer from several key limitations: they often rely on complex, bespoke architectures, are trained via joint optimization over tasks from scratch, and consequently cannot effectively leverage large-scale pretrained models.
In this work, we step back from established 3D task definitions and return to a more fundamental goal inspired by human spatial intelligence: recovering 3D structure from arbitrary visual inputs, be it a single image, multiple views of a scene, or a video stream. Forsaking intricate architectural engineering, we pursue a minimal modeling strategy guided by two central questions. First, is there a minimal set of prediction targets, or is joint modeling across numerous 3D tasks necessary? Second, can a single plain transformer suffice for this objective? Our work provides an affirmative answer to both. We present Depth Anything 3, a single transformer model trained exclusively for joint any-view depth and pose estimation via a specially chosen ray representation. We demonstrate that this minimal approach is sufficient to reconstruct the visual space from any number of images, with or without known camera poses.
Depth Anything 3 formulates the above geometric reconstruction target as a dense prediction task. For a given set of N input images, the model is trained to output N corresponding depth maps and ray maps, each pixel-aligned with its respective input. The architecture to achieve this begins with a standard pretrained vision transformer (e.g., Oquab et al. 2023), as its backbone, leveraging its powerful feature extraction capabilities. To handle arbitrary view counts, we introduce a key modification: an input-adaptive cross-view self-attention mechanism. This module dynamically rearranges tokens during the forward pass in selected layers, enabling efficient information exchange across all views. For the final prediction, we propose a new dual DPT head designed to jointly outputs both depth and ray values, by processing the same set of features with distinct fusion parameters. To enhance flexibility, the model can optionally incorporate known camera poses via a simple camera encoder, allowing it to adapt to various practical settings. This overall design results in a clean and scalable architecture that directly inherits the scaling properties of its pretrained backbone.
We train Depth Anything 3 via a teacher-student paradigm to unify diverse training data, which is necessary for a generalist model. Our data sources include varied formats like real-world depth camera captures (e.g., Baruch et al. 2021), 3D reconstruction (e.g., Reizenstein et al. 2021), and synthetic data, where real-world depth may be of poor quality (Fig. 7). To resolve this, we adopt a pseudo-labeling strategy inspired by prior works (Yang et al., 2024a;b). Specifically, we train a powerful teacher monocular depth model on synthetic data to generate dense, high-quality pseudodepth for all real-world data. Crucially, to preserve geometric integrity, we align these dense pseudodepth maps with the original sparse or noisy depth. This approach proved remarkably effective, significantly enhancing label detail and completeness without sacrificing the geometric accuracy.
To better evaluate our model and track progress in the field, we establish a comprehensive benchmark for assessing geometry and pose accuracy. The benchmark comprises 5 distinct datasets, totaling over 89 scenes, ranging from object-level to indoor and outdoor environments. By directly evaluating pose accuracy across scenes and fusing the predicted pose and depth into a 3D point cloud for accuracy assessment, the benchmark faithfully measures the pose and depth accuracy of visual geometry estimators. Experiments show that our model achieves state-of-the-art performance on 18 out of 20 settings. Moreover, on standard monocular benchmarks, our model outperforms Depth Anything 2 (Yang et al., 2024b).
To further demonstrate the fundamental capability of Depth Anything 3 in advancing other 3D vision tasks, we introduce a challenging benchmark for feed-forward novel view synthesis (FF-NVS), comprising over 160 scenes. We adhere to the minimal modeling strategy and fine-tune our model with an additional DPT head to predict pixel-aligned 3D Gaussian parameters. Extensive experi- ments yield two key findings: 1) fine-tuning a geometry foundation model for NVS substantially outperforms highly specialized task-specific models (Xu et al., 2025c); 2) enhanced geometric reconstruction capability directly correlates with improved FF-NVS performance, establishing Depth Anything 3 as the optimal backbone for this task.
this section cite: ['b3', 'b51', 'b34', 'b65', 'b37', 'b4', 'b43', 'b78', 'b79', 'b75']

Section: DEPTH ANYTHING 3
We tackle the recovery of consistent 3D geometry from diverse visual inputs-single image, multiview collections, or videos-and optionally incorporate known camera poses when available.
this section cite: []

Section: FORMULATION
We denote the input as I = {I i } Nv i=1 with each I i ∈ R H×W ×3 . For N v = 1 this is a monocular image, and for N v > 1 it represents a video or multi-view set. Each image has depth D i ∈ R H×W , camera pose R i | t i , and intrinsics K i . The camera can also be represented as v i ∈ R 9 with translation t i ∈ R 3 , rotation quaternion q i ∈ R 4 , and FOV parameters
f i ∈ R 2 . A pixel p = (u, v, 1) ⊤ projects to a 3D point P = (X, Y, Z, 1) ⊤ by P = R i D i (u, v) K -1
i p + t i , through which the underlying 3D visual space can be faithfully recovered.
this section cite: []

Section: Depth-ray representation.
Predicting a valid rotation matrix R i is challenging due to the orthogonality constraint. To avoid this, we represent camera pose implicitly with a per-pixel ray map, aligned with the input image and depth map. For each pixel p, the camera ray r ∈ R 6 is defined by its origin t ∈ R 3 and direction d ∈ R 3 : r = (t, d). The direction is obtained by backprojecting p into the camera frame and rotating it to the world frame: d = RK -1 p. The dense ray map M ∈ R H×W ×6 stores these parameters for all pixels. We do not normalize d, so its magnitude preserves the projection scale. Thus, a 3D point in world coordinates is simply P = t + D(u, v) • d. This formulation enables consistent point cloud generation by combining predicted depth and ray maps through element-wise operations.
this section cite: []

Section: Minimal prediction targets.
Recent works aim to build unified models for diverse 3D tasks, often using multitask learning with different targets-for example, point maps alone (Wang et al., 2024b), or combinations of pose, local/global point maps, and depth (Wang et al., 2025a;b;Yang et al., 2025a). However, point maps inherently entangle depth and camera information into a single representation, which makes them less effective than disentangled depth predictions for geometry estimation (Wang et al., 2025a;Yang et al., 2025a). Consequently, prior works introduce additional depth heads alongside point maps, creating redundancy in the prediction targets. In contrast, our experiments (Tab. 5) show that a depth-ray representation forms a minimal yet sufficient disentangled target set for capturing both scene structure and camera motion, outperforming point map-based alternatives. However, recovering camera pose from the ray map at inference is computationally costly. We address this by adding a camera head, D C , which has minimal computational overhead. This transformer operates on camera tokens to predict the field of view (f ∈ R 2 ), rotation as a quaternion (q ∈ R 4 ), and translation (t ∈ R 3 ). Since it processes only one token per view, the added computational cost is negligible.
this section cite: ['b64', 'b59', 'b76', 'b59', 'b76']

Section: ARCHITECTURE
We now detail the architecture of Depth Anything 3, which is illustrated in Fig. 2. The network is composed of three main components: a single transformer model as the backbone, an optional camera encoder for pose conditioning, and a Dual-DPT head for generating predictions.
Single transformer backbone. We use a Vision Transformer with L blocks, pretrained on largescale monocular image corpora (e.g., DINOv2 Oquab et al. 2023). Cross-view reasoning is enabled without architectural changes via an input-adaptive self-attention, implemented by rearranging input tokens. We divide the transformer into two groups of sizes L s and L g . The first L s layers apply self-attention within each image, while the subsequent L g layers alternate between cross-view and within-view attention, operating on all tokens jointly through tensor reordering. In practice, we set L s : L g = 2 : 1 with L = L s + L g . As shown in our ablation study in Tab. 6, this configuration provides the optimal trade-off between performance and efficiency compared to other arrangements. This design is input-adaptive: with a single image, the model naturally reduces to monocular depth estimation without extra cost.
this section cite: ['b37']

Section: Camera condition injection.
To seamlessly handle both posed and unposed inputs, we prepend each view with a camera token c i . If camera parameters (K i , R i , t i ) are available, the token is obtained via a MLP E c :
c i = E c (f i , q i , t i ).
Otherwise, a shared learnable token c l is used. Concatenated with patch tokens, these camera tokens participate in all attention operations, providing either explicit geometric context or a consistent learned placeholder. Dual-DPT head. For the final prediction stage, we introduce a Dual-DPT head that jointly outputs dense depth and ray values, offering both strong and efficient (Tab. 5) results. Backbone features are first processed through shared reassembly modules, then split into two branches with distinct fusion layers for depth and rays, followed by separate output layers. Both branches thus operate on the same processed features, differing only in the fusion stage, which promotes interaction between tasks while avoiding redundant representations. Our model also outputs confidence map for depth following Wang et al. (2024b).
this section cite: ['b64']

Section: TRAINING
Teacher-student learning paradigm. Our training data comes from diverse sources, including real-world depth captures, 3D reconstructions, and synthetic datasets. Real-world depth is often noisy and incomplete (Fig. 7), limiting its supervisory value. To mitigate this, we train a monocular relative depth estimation "teacher" model solely on synthetic data to generate high-quality pseudolabels. These pseudo-depth maps are aligned with the original sparse or noisy ground truth via RANSAC least squares, enhancing label detail and completeness while preserving geometric accuracy. We term this model Depth-Anything-3-Teacher, trained on a large synthetic corpus covering indoor, outdoor, object-centric, and diverse in-the-wild scenes to capture fine geometry. We detail our teacher design in the appendix.
Training objectives. Following the formulation in Sec. 2.1, our model F θ maps an input I to a set of outputs comprising a depth map D, a ray map R, and an optional camera pose ĉ: F θ : I → { D, R, ĉ}. The gray color indicates that ĉ is an optional output, included primarily for practical convenience. Prior to loss computation, all ground-truth signals are normalized by a common scale factor. This scale is defined as the mean ℓ 2 norm of the valid reprojected point maps P, a step that ensures consistent magnitude across different modalities and stabilizes the training process. The overall training objective is defined as a weighted sum of several loss terms:
L = L D ( D, D) + L M ( R, M) + L P ( D ⊙ d + t, P) + βL C (ĉ, v) + αL grad ( D, D).
In practice, we set α = 1 and β = 1. L D is a confidence-aware loss following Wang et al. (2024b). L grad is taken from Yang et al. (2024b) , penalizing the depth gradients. This loss preserves sharp edges while ensuring smoothness in planar regions. We detail the loss function in the appendix.
this section cite: ['b64', 'b79']

Section: FINETUNING FOR FEED-FORWARD NOVEL VIEW SYNTHESIS
Inspired by human spatial intelligence, we believe that consistent depth estimation can greatly enhance downstream 3D vision tasks. We choose feed-forward novel view synthesis (FF-NVS) as the demonstration task, given its growing attention driven by advances in neural 3D representations and its relevance to numerous applications. Adhere to the minimal modeling strategy, we perform FF-NVS by fine-tuning with an added DPT head (GS-DPT) to infer pixel-aligned 3D Gaussians.
this section cite: []

Section: GS-DPT Head.
Given visual tokens for each view extracted via our single transformer backbone (Sec. 2.2), GS-DPT predicts the camera-space 3D Gaussian parameters {σ i , q i , s i , c i } H×W i=1 , where σ i , q i ∈ H, s i ∈ R 3 , c i ∈ R 3 denote the opacity, rotation quaternion, scale, and RGB color of the i-th 3D Gaussian, respectively. Among them, σ i is predicted by the confidence head, while others are predicted by the main GS-DPT head. The estimated depth is unprojected to world coordinates to obtain the global positions P i ∈ R 3 of the 3D Gaussians. These primitives are then rasterized to synthesize novel views from given camera poses. We detail our loss functions in the appendix.
this section cite: []

Section: VISUAL GEOMETRY BENCHMARK
We further introduce a visual geometry benchmark to assess geometry prediction models. It directly evaluates pose accuracy, depth via reconstruction accuracy and visual rendering quality.
Pose accuracy. Our benchmark covers 5 datasets: HiRoom (an internal high-fidelity room dataset), ETH3D (Schops et al., 2017), DTU (Aanaes et al., 2016), 7Scenes (Shotton et al., 2013), and ScanNet++ (Yeshwanth et al., 2023), containing 29, 11, 22, 7, and 20 scenes, respectively. These span object-centric to indoor and outdoor. HiRoom and the benchmark will be released publicly. ScanNet++ is not a zero-shot dataset, as it has been widely used for training since DUSt3R. Although comparisons are biased, we retain it for completeness since subsequent methods also adopt it. We report Auc3 and Auc30, which measure relative rotation and translation score (higher is better).
this section cite: ['b47', 'b0', 'b49', 'b82']

Section: Geometry accuracy.
Using the same datasets, we assess depth accuracy via reconstruction. Unlike Wang et al. (2025a), which aligns predicted depths to ground truth with scale and shift and then reconstructs the scene with ground-truth poses, we reconstruct using both predicted poses and depths. The resulting point cloud is aligned to ground truth by applying evo (Umeyama, 2002) to match predicted poses with ground-truth poses. We report F-Score for all datasets except Chamfer Distance for DTU, following a prior work (Yu et al., 2022).
this section cite: ['b59', 'b56', 'b85']

Section: Visual rendering quality.
We evaluate visual rendering quality on diverse large-scale scenes. We introduce a new NVS benchmark built from three datasets, including DL3DV (Ling et al., 2024) with 140 scenes, Tanks and Temples (Knapitsch et al., 2017a) with 6, and MegaDepth (Li and Snavely, 2018) with 19, each spanning around 300 sampled frames. Ground truth camera poses, estimated with COLMAP, are used directly to ensure accurate and fair comparison across diverse models. We report PSNR, SSIM, and LPIPS metrics on rendered novel views using given camera poses.
this section cite: ['b29', 'b23', 'b28']

Section: EXPERIMENTS
Training datasets, baselines, implementation details and more ablations are provided in the appendix.
Table 1: Comparisons with SOTA methods on pose accuracy. We report both Auc3 ↑ and Auc30 ↑ metrics. The top-3 results are highlighted as first , second , and third .
this section cite: []

Section: Methods

this section cite: []

Section: HiRoom ETH3D DTU 7Scenes ScanNet++

this section cite: []

Section: Auc3 Auc30 Auc3 Auc30 Auc3 Auc30 Auc3 Auc30 Auc3 Auc30
Colmap 13.0 19.0 4.75 41.3 87.2 87.4 22.3 62.0 13.3 19.9 Glomap 31.9 42.6 8.37 21.7 96.8 96.9 24.1 85.0 20.8 42.9 DUSt3R 17.6 54.3 4.30 27.3 4.00 74.3 6.90 61.6 8.10 33.9 Fast3R 25.9 77.0 8.10 44.4 9.50 79.1 19.0 78.6 17.9 72.5 MapAnything 17.9 82.8 19.2 77.4 6.50 72.7 12.6 79.7 20.2 84.1 Pi3 67.0 94.8 35.2 87.3 62.5 94.9 25.5 86.3 50.7 92.1 VGGT 49.1 88.0 26.3 80.8 79.2 99.8 23.9 85.0 62.6 95.1 DA3-Giant 81.7 96.4 39.3 90.6 85.6 94.9 29.2 86.8 83.2 98.0 DA3-Large 37.9 84.5 19.0 81.7 58.4 95.3 25.1 85.4 46.9 92.1 DA3-Base 12.8 79.8 13.6 74.0 31.4 90.8 17.2 81.1 16.2 77.5 DA3-Small 3.40 64.6 4.89 51.9 9.46 82.2 6.19 71.8 2.86 51.8 Classical methods excel on dense, well-textured scenes like DTU, where GLOMAP achieves top performance (Auc3: 96.8). However, they struggle significantly on challenging scenarios with sparse views, textureless regions, or dynamic content. For instance, on HiRoom, COLMAP achieves only 13.0 Auc3 compared to 81.7 from our method. Similarly, on ScanNet++, COLMAP obtains 13.3 Auc3 versus 83.2 from ours.
Our DA3-Giant model establishes new SOTA results across nearly all metrics, with particularly strong performance on challenging datasets. On Auc3, our model delivers at least an 8% relative improvement over all feed-forward competitors, and on ScanNet++ it achieves a 33% relative gain over the second-best feed-forward model. To qualitatively assess pose quality, we visualize predicted camera trajectories on two in-the-wild dynamic videos in Fig. 4. Our trajectories are smooth and closely align with the ground truth, whereas VGGT and Pi3 exhibit substantially noisier paths.
Geometry estimation. As shown in Tab. 2, we compare our method against both classical structure-from-motion pipelines (COLMAP Schönberger and Frahm (2016), GLOMAP Pan et al. (2024b)) and state-of-the-art feed-forward reconstruction methods under two distinct conditions: a pose-free setting where camera parameters are unavailable, and a pose-conditioned setting where they are known.
Classical methods perform competitively on extremely dense datasets like DTU, where GLOMAP+PatchMatchStereo achieves 1.62 mm chamfer distance. However, their performance degrades significantly on datasets that are even slightly sparser. For example, on ETH3D, COLMAP+PM achieves an F-score of only 20.7 (ours: 74.4), and on HiRoom, it drops to 16.8
Table 2: Comparisons with SOTA methods on reconstruction accuracy. For all datasets except DTU, we report the F-Score (F1 ↑). For DTU, we report the chamfer distance (CD ↓, unit: mm). w/o p. and w/ p. denote without pose and with pose, indicating whether ground-truth camera poses are provided for reconstruction. The top-3 results are highlighted as first , second , and third .
Methods HiRoom ETH3D DTU 7Scenes ScanNet++ w/o p. w/ p. w/o p. w/ p. w/o p. w/ p. w/o p. w/ p. w/o p. w/ p. Colmap+PM 16.8 21.9 20.7 25.7 1.67 1.64 40.0 37.3 15.7 19.6 Glomap+PM 30.7 41.1 24.1 36.3 1.62 1.59 43.9 47.6 16.5 22.4 DUSt3R 30.1 39.5 19.7 18.8 7.60 7.97 26.6 39.8 18.9 27.3 Fast3R 40.7 48.2 38.5 50.3 6.88 8.20 41.0 49.8 37.1 53.7 MapAnything 32.4 69.2 54.8 71.9 7.91 3.97 44.8 55.2 39.4 71.3 Pi3 75.8 85.0 72.7 80.6 3.28 1.72 44.2 57.5 63.1 73.3 VGGT 56.7 70.2 57.2 66.7 2.05 1.44 47.9 51.4 66.4 70.7 DA3-Giant 89.3 95.2 74.4 85.8 1.92 0.91 52.0 52.3 76.4 79.2 DA3-Large 48.2 85.7 57.3 79.1 3.45 2.48 48.7 48.7 58.9 72.9 DA3-Base 18.6 71.7 52.8 66.6 5.14 1.99 37.8 47.2 39.7 66.3 DA3-Small 12.9 43.1 39.4 58.2 5.12 4.05 30.8 39.5 24.2 45.7 (ours: 89.3). This highlights the limitations of classical methods in handling sparse-view scenarios, where correspondence matching becomes unreliable.
In contrast, our DA3-Gaint establishes a new SOTA across nearly all scenarios, outperforming all feed-forward competitors in all five pose-free settings. On average, DA3-Gaint achieves a relative improvement of 23.6% over VGGT and 16.7% over Pi3. Even more notably, our much smaller DA3-Large (0.30B parameters) demonstrates remarkable efficiency. Despite being 3× smaller than VGGT (0.90B parameters), it surpasses VGGT in five out of the ten settings, with particularly strong performance on ETH3D.
When camera poses are available, both our method and MapAnything can exploit them for improved results, and other methods also benefit from ground-truth pose fusion. Our model shows clear gains on most datasets except 7Scenes, where the limited video setting already saturates performance and reduces the benefit of pose conditioning. Notably, with pose conditioning, performance gains from scaling model size are smaller than in pose-free models, indicating that pose estimation scales more strongly than depth estimation and requires larger models to fully realize improvements.
this section cite: []

Section: Visual rendering.
To fairly evaluate feedforward novel view synthesis (FF-NVS), we compare against three recent 3DGS models-pixelSplat (Charatan et al., 2024), MVSplat (Chen et al., 2024), and DepthSplat (Xu   (Yang et al., 2025b), MV-DUSt3R (Tang et al., 2025), and VGGT (Wang et al., 2025a). All models are trained on DL3DV-10K training set under a unified protocol and evaluated on our benchmark (Sec. 3). As shown in Tab. 4, all models perform substantially better on DL3DV than on the other datasets, suggesting that 3DGS-based NVS is sensitive to trajectory and pose distributions standardized by DL3DV, rather than scene content. Comparing the two groups, geometry-model-based frameworks consistently outperform specialized feed-forward models, demonstrating that a simple backbone plus DPT head can surpass complex task-specific designs. The advantage stems from large-scale pretraining, which enables better generalization and scalability than approaches relying on epipolar transformers, cost volumes, or cascaded modules. Within this group, NVS performance correlates with geometry estimation capability, making DA3 the strongest backbone. Looking forward, we expect FF-NVS can be effectively addressed with simple architectures leveraging pretrained geometry backbones, and that the strong spatial understanding of DA3 will benefit other 3D vision tasks. We adopt depth + ray + cam as our final configuration. Since the auxiliary cam head incurs negligible computational overhead (∼0.1% of backbone cost) and is significantly faster than extracting camera parameters from ray maps through optimization (0.46ms v.s 8.60ms on an A100 GPU), we include it at no practical cost.
this section cite: ['b7', 'b8', 'b77', 'b54', 'b59']

Section: SUFFICIENCY OF A SINGLE PLAIN TRANSFORMER
We compare a standard ViT-L backbone with a VGGT-style architecture that stacks two distinct transformers, tripling the block count. For fair capacity comparison, the VGGT-style model uses smaller ViT-B backbones, yielding a similar parameter size to our ViT-L. Our backbone supports two attention strategies: Full Alt., which alternates cross-view/within-view attention in all layers (L = L g ), and our default partial alternation. As shown in Table 6, the VGGT-style model drops to 79.8% of our baseline performance, confirming the superiority of a single-transformer design at similar scale. We attribute this gap to full pretraining of our backbone versus two-thirds untrained blocks in VGGT. Moreover, the Full Alt. variant degrades across nearly all metrics-except F1 on 7Scenes-indicating that partial alternation is the more effective and robust strategy.
this section cite: []

Section: RELATED WORK
Multi-view visual geometry estimation. Traditional systems (Schönberger and Frahm, 2016;Schönberger et al., 2016) decompose reconstruction into feature detection and matching, robust relative pose estimation, incremental or global SfM with bundle adjustment, and dense multi-view stereo for per-view depth and fused point clouds. These methods remain strong on well-textured scenes, but their modularity and brittle correspondences complicate robustness under low texture, specularities, or large viewpoint changes. Early learning methods injected robustness at the component level: learned detectors (DeTone et al., 2018), descriptors for matching (Dusmanu et al., 2019), and differentiable optimization layers that expose pose/depth updates to gradient flow (He et al., 2024;Guo et al., 2025;Pan et al., 2024a). On the dense side, cost-volume networks (Yao et al., 2018;Xu et al., 2023;2025b) for MVS replaced hand-crafted regularization with 3D CNNs, improving depth accuracy especially at large baselines and thin structures compared with classical PatchMatch. Early end-to-end approaches (Teed and Deng, 2018;Wang et al., 2024a) moved beyond modular SfM/MVS pipelines by directly regressing camera poses and per-image depths from pairs of images. These approaches reduced engineering complexity and demonstrated the feasibility of learned joint depth pose estimation, but they often struggled with scalability, generalization, and handling arbitrary input cardinalities.
A turning point came with DUSt3R (Wang et al., 2024b), which leveraged transformers to directly predict point map between two views and compute both depth and relative pose in a purely feedforward manner. This work laid the foundation for subsequent transformer-based methods aiming to unify multi-view geometry estimation at scale. Follow-up models extended this paradigm with multi-view inputs (Yang et al., 2025a;Wang et al., 2025b;Tang et al., 2025), video input (Zhang et al., 2025a;Wang et al., 2025b;Murai et al., 2025), robust correspondence modeling (Leroy et al., 2024), camera parameter injection (Jang et al., 2025;Keetha et al., 2025), view synthesis (Zhang et al., 2025b), and multi-modal prior prompting (Liu et al., 2025). Among these, Wang et al. (2025a) push accuracy to a new level through large-scale training, a multi-stage architecture, and redundancy in design. Parallel to this transformer-centric evolution, a complementary line of works (Zhang et al., 2024;Lu et al., 2025) explores diffusion models for ray-based camera estimation. These methods perform inference via iterative sampling, incurring higher computational cost than singlepass predictors. In contrast, we focus on a minimal modeling strategy built around a single, simple transformer.
this section cite: ['b12', 'b17', 'b15', 'b80', 'b71', 'b55', 'b58', 'b64', 'b76', 'b62', 'b54', 'b62', 'b35', 'b26', 'b19', 'b22', 'b88', 'b30', 'b59', 'b86', 'b31']

Section: Monocular depth estimation.
Early monocular depth estimation methods relied on fully supervised learning on single-domain datasets, which often produced models specialized to either indoor rooms (Silberman et al., 2012) or outdoor driving scenes (Geiger et al., 2013). These early deep models achieved good accuracy within their training domain but struggled to generalize to novel environments, highlighting the challenge of cross-domain depth prediction. Modern generalist approaches (Yang et al., 2024a;b;Wang et al., 2025c;Bochkovskii et al., 2024;Yin et al., 2023;Ke et al., 2024) exemplify this trend by leveraging massive multi-dataset training and advanced architectures like vision transformers (Ranftl et al., 2021) or DiT (Peebles and Xie, 2023). Trained on millions of images, they learn broad visual cues and incorporate techniques such as affine-invariant depth normalization. Recent efforts further push toward pixel-perfect and arbitrary-resolution depth estimation (Xu et al., 2025a;2026;Yu et al., 2026), producing fine-grained, flying-pixel-free predictions through pixel-space diffusion or neural implicit representations. In contrast, our method is primarily designed for a unified visual geometry estimation task, yet it still demonstrates competitive monocular depth performance.
this section cite: ['b50', 'b13', 'b78', 'b63', 'b5', 'b83', 'b21', 'b42', 'b41', 'b84']

Section: CONCLUSION AND DISCUSSION
Depth Anything 3 shows that a plain transformer, trained on depth-and-ray targets with teacherstudent supervision, can unify any-view geometry without ornate architectures. Scale-aware depth, per-pixel rays, and adaptive cross-view attention let the model inherit strong pretrained features while remaining lightweight and easy to extend. On the proposed visual geometry benchmark the approach sets new pose and reconstruction records, with both giant and compact variants surpassing prior models, while the same backbone powers efficient feed-forward novel view synthesis model.
We view Depth Anything 3 as a step toward versatile 3D foundation models. Future work can extend its reasoning to dynamic scenes, integrate language and interaction cues, and explore largerscale pretraining to close the loop between geometry understanding and actionable world models. We hope the model and dataset releases, benchmark, and simple modeling principles offered here catalyze broader research on general-purpose 3D perception.
this section cite: []

Section: References
Ref_id:b0 Title: Large-scale data for multiple-view stereopsis Year: (1920)
Ref_id:b1 Title: Direct linear transformation from comparator coordinates into object space coordinates in close-range photogrammetry. Photogrammetric engineering & remote sensing Year: (2015)
Ref_id:b2 Title: Map-free visual relocalization: Metric pose relative to a single image Year: (2022)
Ref_id:b3 Title: Perception of three-dimensional shape specified by optic flow by 8-week-old infants Year: (2000)
Ref_id:b4 Title: Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data Year: (2021-02-21)
Ref_id:b5 Title: Depth pro: Sharp monocular metric depth in less than a second Year: (2024)
Ref_id:b6 Title: Virtual kitti 2 Year: (2020)
Ref_id:b7 Title: pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction Year: (2024)
Ref_id:b8 Title: Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images Year: (2024)
Ref_id:b9 Title: Objaverse: A universe of annotated 3d objects Year: (2023)
Ref_id:b10 Title: Superpoint: Self-supervised interest point detection and description Year: (2018)
Ref_id:b11 Title: Accessed: Sep Year: (2024)
Ref_id:b12 Title: D2-net: A trainable cnn for joint description and detection of local features Year: (2019)
Ref_id:b13 Title: Vision meets robotics: The kitti dataset Year: (2013)
Ref_id:b14 Title: Online training of stereo self-calibration using monocular depth estimation Year: (2021)
Ref_id:b15 Title: Multi-view reconstruction via sfm-guided monocular depth estimation Year: (2025)
Ref_id:b16 Title: All for one, and one for all: Urbansyn dataset, the third musketeer of synthetic driving scenes Year: (2025)
Ref_id:b17 Title: Detector-free structure from motion Year: (2024)
Ref_id:b18 Title: Learning multi-view stereopsis Year: (2018)
Ref_id:b19 Title: Pow3r: Empowering unconstrained 3d reconstruction with camera and scene priors Year: (2025)
Ref_id:b20 Title: Megasynth: Scaling up 3d scene reconstruction with synthesized data Year: (2025)
Ref_id:b21 Title: Repurposing diffusion-based image generators for monocular depth estimation Year: (2024)
Ref_id:b22 Title: MapAnything: Universal feed-forward metric 3D reconstruction Year: (2025)
Ref_id:b23 Title: Tanks and temples: Benchmarking large-scale scene reconstruction Year: (2017-05-21)
Ref_id:b24 Title: Tanks and temples: Benchmarking large-scale scene reconstruction Year: (1920)
Ref_id:b25 Title: Multimodal synthetic dataset of enclosed garden scenes Year: (2021)
Ref_id:b26 Title: Grounding image matching in 3d with mast3r Year: (2024)
Ref_id:b27 Title: Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond Year: (2023)
Ref_id:b28 Title: Megadepth: Learning single-view depth prediction from internet photos Year: (2018)
Ref_id:b29 Title: Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision Year: (2024)
Ref_id:b30 Title: Worldmirror: Universal 3d world reconstruction with any-prior prompting Year: (2025)
Ref_id:b31 Title: Matrix3d: Large photogrammetry model all-in-one Year: (2025)
Ref_id:b32 Title: Scenenet rgb-d: Can 5m synthetic images beat generic imagenet pre-training on indoor segmentation? Year: (2017)
Ref_id:b33 Title: Spring: A high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo Year: (2023)
Ref_id:b34 Title: Orb-slam: A versatile and accurate monocular slam system Year: (2015)
Ref_id:b35 Title: Mast3r-slam: Real-time dense slam with 3d reconstruction priors Year: (2025)
Ref_id:b36 Title: 3d ken burns effect from a single image Year: (2019)
Ref_id:b37 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b38 Title: Global structure-frommotion revisited Year: ()
Ref_id:b39 Title: Global Structure-from-Motion Revisited Year: ()
Ref_id:b40 Title: Aria digital twin: A new benchmark dataset for egocentric 3d machine perception Year: (2023)
Ref_id:b41 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b42 Title: Vision transformers for dense prediction Year: (2021)
Ref_id:b43 Title: Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction Year: (2021)
Ref_id:b44 Title: Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding Year: (2021)
Ref_id:b45 Title: Structure-from-motion revisited Year: (2016)
Ref_id:b46 Title: Pixelwise view selection for unstructured multi-view stereo Year: (2016)
Ref_id:b47 Title: A multi-view stereo benchmark with high-resolution images and multi-camera videos Year: (2017)
Ref_id:b48 Title: A comparison and evaluation of multi-view stereo reconstruction algorithms Year: (2006)
Ref_id:b49 Title: Scene coordinate regression forests for camera relocalization in rgb-d images Year: (2013)
Ref_id:b50 Title: Indoor segmentation and support inference from rgbd images Year: (2012)
Ref_id:b51 Title: Photo tourism: exploring photo collections in 3d Year: (2006)
Ref_id:b52 Title: The Replica dataset: A digital replica of indoor spaces Year: (2018)
Ref_id:b53 Title: The replica dataset: A digital replica of indoor spaces Year: (2019)
Ref_id:b54 Title: Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds Year: (2025)
Ref_id:b55 Title: Deepv2d: Video to depth with differentiable structure from motion Year: (2018)
Ref_id:b56 Title: Least-squares estimation of transformation parameters between two point patterns Year: (2002)
Ref_id:b57 Title: Posediffusion: Solving pose estimation via diffusion-aided bundle adjustment Year: (2023)
Ref_id:b58 Title: Vggsfm: Visual geometry grounded deep structure from motion Year: (2010)
Ref_id:b59 Title: Vggt: Visual geometry grounded transformer Year: (1920)
Ref_id:b60 Title: Flow-motion and depth network for monocular stereo and beyond Year: (2020)
Ref_id:b61 Title: Irs: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation Year: (2019)
Ref_id:b62 Title: Continuous 3d perception model with persistent state Year: (2025)
Ref_id:b63 Title: Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision Year: (2025)
Ref_id:b64 Title: Dust3r: Geometric 3d vision made easy Year: (2006)
Ref_id:b65 Title: Dust3r: Geometric 3d vision made easy Year: (2024-02-22)
Ref_id:b66 Title: Tartanair: A dataset to push the limits of visual slam Year: (2020)
Ref_id:b67 Title: Scalable permutation-equivariant visual geometry learning Year: (2025)
Ref_id:b68 Title: Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation Year: (2023)
Ref_id:b69 Title: Rgbd objects in the wild: Scaling real-world 3d object learning from rgb-d videos Year: (2024)
Ref_id:b70 Title: Structured 3d latents for scalable and versatile 3d generation Year: (2024)
Ref_id:b71 Title: Iterative geometry encoding volume for stereo matching Year: (2023)
Ref_id:b72 Title: Pixel-perfect depth with semantics-prompted diffusion transformers Year: ()
Ref_id:b73 Title: Igev++: Iterative multi-range geometry encoding volumes for stereo matching Year: (2010)
Ref_id:b74 Title: Pixel-perfect visual geometry estimation Year: (2026)
Ref_id:b75 Title: Depthsplat: Connecting gaussian splatting and depth Year: (2007)
Ref_id:b76 Title: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass Year: (2025-03)
Ref_id:b77 Title: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass Year: (2025-08-22)
Ref_id:b78 Title: Depth anything v2 Year: (2024-02-10)
Ref_id:b79 Title: Depth anything v2 Year: (2024)
Ref_id:b80 Title: Mvsnet: Depth inference for unstructured multi-view stereo Year: (2018)
Ref_id:b81 Title: Blendedmvs: A large-scale dataset for generalized multi-view stereo networks Year: (2020)
Ref_id:b82 Title: Scannet++: A highfidelity dataset of 3d indoor scenes Year: (2023)
Ref_id:b83 Title: Metric3d: Towards zero-shot metric 3d prediction from a single image Year: (2023)
Ref_id:b84 Title: Infinidepth: Arbitrary-resolution and fine-grained depth estimation with neural implicit fields Year: (2026)
Ref_id:b85 Title: Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction Year: (2022)
Ref_id:b86 Title: Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as rays: Pose estimation via ray diffusion Year: (2024)
Ref_id:b87 Title: Monst3r: A simple approach for estimating geometry in the presence of motion Year: ()
Ref_id:b88 Title: Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views Year: (2010)
Ref_id:b89 Title: Unrealstereo: Controlling hazardous factors to analyze stereo vision Year: (2018)
Ref_id:b90 Title: Geomvsnet: Learning multi-view stereo with geometry perception Year: (2023)
Ref_id:b91 Title: Structured3d: A large photo-realistic dataset for structured 3d modeling Year: (2020)
Ref_id:b92 Title: Bilateral reference for high-resolution dichotomous image segmentation Year: (2024)
Ref_id:b93 Title: Pointodyssey: A large-scale synthetic dataset for long-term point tracking Year: (2023)
Ref_id:b94 Title: Nicer-slam: Neural implicit scene encoding for rgb slam Year: (1920)
