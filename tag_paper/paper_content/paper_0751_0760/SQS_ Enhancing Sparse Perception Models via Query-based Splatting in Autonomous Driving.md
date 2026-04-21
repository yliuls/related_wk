Title: SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving
Abstract: Sparse Perception Models (SPMs) adopt a query-driven paradigm that forgoes explicit dense BEV or volumetric construction, enabling highly efficient computation and accelerated inference. In this paper, we introduce SQS, a novel query-based splatting pre-training specifically designed to advance SPMs in autonomous driving. SQS introduces a plug-in module that predicts 3D Gaussian representations from sparse queries during pre-training, leveraging self-supervised splatting to learn fine-grained contextual features through the reconstruction of multi-view images and depth maps. During fine-tuning, the pre-trained Gaussian queries are seamlessly integrated into downstream networks via query interaction mechanisms that explicitly connect pre-trained queries with task-specific queries, effectively accommodating the diverse requirements of occupancy prediction and 3D object detection. Extensive experiments on autonomous driving benchmarks demonstrate that SQS delivers considerable performance gains across multiple query-based 3D perception tasks, notably in occupancy prediction and 3D object detection, outperforming prior state-of-the-art pre-training approaches by a significant margin (i.e., +1.3 mIoU on occupancy prediction and +1.0 NDS on 3D detection).

Section: Introduction
Recent advances in vision-centric autonomous driving have driven significant progress in the field [7,60]. From a representation standpoint, existing approaches can be broadly categorized into dense BEVcentric and sparse query-centric paradigms. Dense BEV-centric methods [24,44,13] extract Bird's Eye View (BEV) features from multi-view images for downstream tasks, while Sparse Perception Models (SPMs) [55,33,31] bypass explicit dense representations and directly aggregate features from images using implicit queries, enabling faster inference. Sparse query-centric methods have garnered increasing attention within the community due to their practical advantages for real-world deployment. Despite the dominance of supervised methods, their reliance on precise ground-truth annotations presents a substantial challenge, as acquiring such labels is both costly and labor-intensive. Conversely, the abundance of unlabeled data offers a promising avenue to further enhance model performance. Nevertheless, effectively leveraging this data remains a non-trivial challenge.
To mitigate these challenges, various pre-training strategies have been proposed for autonomous driving. Earlier works leverage contrastive learning [45] and Masked Autoencoders (MAE) [38] for pre-training. However, the coarse supervision they provide limits their capacity to fully capture spatialtemporal geometry. In contrast, NeRF-based approaches such as UniPAD [62] and ViDAR [64] utilize 3D volumetric differentiable rendering to reconstruct and predict 3D scene structures. To reduce memory overhead and improve rendering efficiency, a separate line of research [58,69] introduce 3D Gaussian Splatting (3DGS) [19] for explicit scene representation. By predicting Gaussian parameters for feedforward reconstruction, GaussianPretrain [58] achieves comprehensive scene understanding by integrating geometric and texture representations. Moreover, VisionPAD [69] projects neighboring frames onto the current frame using rendered depths and relative poses, relying solely on RGB image supervision rather than explicit depth annotations. Although existing pre-training paradigms have substantially enhanced the performance of downstream applications, their reliance on dense BEV representations limits their applicability to SPMs (Fig. 1(a)).
In this paper, we present the first attempt to pretrain Sparse Perception Models (SPMs) on unlabeled data to enhance their downstream performance, as shown in Fig. 1(b). Unlike dense BEV-centric perception models, we find out that the latent sparse queries in SPMs lack explicit spatial positions and semantic meanings, making it challenging to directly apply existing rendering-based pre-training methods, which often fail to preserve informative representations during training. To address this challenge, we propose SQS, a novel pre-training framework for SPMs based on query-based splatting. Unlike previous approaches, SQS introduces a small set of adaptive Gaussian queries during pretraining. These queries dynamically predict 3D Gaussians and reconstruct both depth and RGB images via a splatting mechanism, enabling the model to learn fine-grained representations from unlabeled data in a self-supervised manner. After pre-training, the learned Gaussian queries are used for fine-tuning, where they interact and fuse with task-specific queries, resulting in improved downstream performance. We evaluate our approach on tasks such as object detection and 3D occupancy prediction. Experimental results demonstrate that SQS consistently achieves significant performance improvements over state-of-the-art methods without pre-training.
To this end, our contributions can be summarized as follows:
• We propose SQS, the first query-based splatting pre-training technique specifically designed to advance Sparse Perception Models (SPMs).
• We introduce plug-and-play Gaussian queries, which learns fine-grained features in a self-supervised manner during pre-training, and further enhances downstream tasks via interactive feature fusion during fine-tuning.
• SQS significantly enhances performance in both occupancy prediction and 3D object detection, surpassing previous state-of-the-art results on multiple autonomous driving benchmarks (i.e., +1.3 mIoU on occupancy prediction and +1.0 NDS on 3D detection as Fig. 1(c)).
this section cite: ['b6', 'b59', 'b23', 'b43', 'b12', 'b54', 'b32', 'b30', 'b44', 'b37', 'b61', 'b63', 'b57', 'b68', 'b18', 'b57', 'b68']

Section: Related Work
Pre-training in autonomous driving. Pre-training has gained remarkable progress in recent years for autonomous driving. Conventional approaches cover supervised [42,54,50,59], contrastive [23,41,45,65], and masked signal modeling [39,38,2,20] categories. With advances in neural rendering [37], rendering-based pre-training [62,64,74,66] becomes an alternative by rendering images from dense BEV or Volume representation. For example, UniPAD [62] utilizes 3D volumetric differentiable rendering to reconstruct 3D shape structures and appearance characteristics. Meanwhile, ViDAR [64] predicts the future point cloud using a Latent Rendering operator based on historical embeddings. To achieve effective and efficient rendering, 3D Gaussian Splatting [19] is introduced in some recent works. GaussianPretrain [58] considers 3D Gaussian anchors as volumetric LiDAR points for unified geometric and texture representations. Without explicit depth supervision, VisionPAD [69] reconstructs images employing both voxel velocity estimation and multi-frame photometric consistency. These pre-training pipelines successfully model spatial-temporal representation for dense BEV features.
However, the emerging perception methods with sparse route [55,33,34,52,31] are not compatible with the paradigms above. Recently, the query-based pre-training in 2D image has been developed. Frozen-DETR [10] utilizes frozen foundation models with class token and patch token, which provide a compact context and semantic details, respectively. GLID [32] models pre-training pretext task and other downstream tasks as "query-to-answer" problems. Since the sparse pre-training in autonomous driving requires an accurate 3D geometric representation extracted from multi-view images, the existing methods for 2D are inapplicable for Sparse Perception Models (SPMs).
this section cite: ['b41', 'b53', 'b49', 'b58', 'b22', 'b40', 'b44', 'b64', 'b38', 'b37', 'b1', 'b19', 'b36', 'b61', 'b63', 'b73', 'b65', 'b61', 'b63', 'b18', 'b57', 'b68', 'b54', 'b32', 'b33', 'b51', 'b30', 'b9', 'b31']

Section: Sparse Perception Models for 3D Detection and Occupancy Prediction.
For the sparse 3D detection, motivated by DETR [5], DETR3D [55] utilizes a sparse set of 3D object queries to sample the 2D features from images by 3D point projection. To avoid the complex 2D-to-3D projection and feature sampling, PETR series [33,34,52,18,30,67] directly interact with 3D position-aware features by encoding the 3D position into 2D image features. Without relying on dense view transformation nor global attention, Sparse4D [27] iteratively refines anchor boxes via sparsely sampling and fuses spatial-temporal features. In the SparseBEV [31], to adapt the detector in both BEV and image space, a set of sparse pillar queries initialized in BEV space are applied to interact with the image features.
Regarding occupancy prediction, the query-based approaches [51,48,47] are proposed to reduce computational cost. OPUS [51] formulates the task as a streamlined set prediction paradigm. SparseOcc [48] proposes an efficient occupancy network with 3D sparse diffuser and convolutional kernels while OSP [47] presents the Points of Interest (PoIs) to represent the scene. Recently, 3DGS has demonstrated the capacity to adapt flexibly to varying object scales and regional complexities in a deformable manner, thereby enhancing resource allocation and overall efficiency. Based on the aforementioned advantages, another line of works utilize 3DGS for supervised [16,14,75] or self-supervised [1, 68, 17] occupancy prediction. In conclusion, compared to the BEV based methods, the sparse algorithms reduce computational cost and broaden the perception range. This distinctive advantage makes the development of a sparse pre-training algorithm for them particularly imperative.
3D Gaussian Splatting in Autonomous Driving. 3D Gaussian Splatting (3DGS) [19] uses multiple 3D Gaussian primitives for fast radiance field rendering, enabling explicit representation with fewer parameters. For reconstructing driving scenes, several approaches are carefully designed for 3D static [72,61,56] and 4D dynamic [63] scenes. More recently, 3DGS based perception models have been proposed, including occ prediction [16,14,75], bev segmentaiong [6,36] and end-toend tasks [71]. Alternatively, some recent works apply 3DGS for self-supervised occ prediction [1,17,68]. GaussianFlowOcc [1] and TT-GaussOcc [68] model scene dynamics by predicting the temporal flow for each Gaussian throughout the training procedure. Without requiring explicit annotations, GaussTR [17] splats the Gaussians onto 2D perspectives and aligns the extracted features with foundation models. Furthermore, regarding to the self-supervised pre-training, sevaral methods [58,69] adopt 3DGS for explicit geometry representation in the Dense BEV or Volume feature to improve the performance of downstream tasks. Nevertheless, up to now, there is still no pre-training scheme that can effectively adapt to Sparse Perception Models (SPMs).
this section cite: ['b4', 'b54', 'b32', 'b33', 'b51', 'b17', 'b29', 'b66', 'b26', 'b30', 'b50', 'b47', 'b46', 'b50', 'b47', 'b46', 'b15', 'b13', 'b74', 'b18', 'b71', 'b60', 'b55', 'b62', 'b15', 'b13', 'b74', 'b5', 'b35', 'b70', 'b0', 'b16', 'b67', 'b0', 'b67', 'b16', 'b57', 'b68']

Section: Proposed Method
In this section, we introduce our query-based splatting pre-training approach for autonomous driving.
The overall architecture of the proposed SQS framework is depicted in Fig. 2.
We first provide the necessary preliminaries on 3D Gaussian Splatting, which enables the rendering of both RGB images and depth maps from predicted 3D Gaussians. Subsequently, we briefly outline the image encoder employed to extract multi-scale features from multi-view input images. We then detail the query-based Gaussian transformer decoder, which utilizes Gaussian queries to predict 3D Gaussians and facilitates the learning of fine-grained information via self-supervised splatting. Finally, by incorporating the pre-trained Gaussian queries through a query interaction module during finetuning, our approach effectively transfers knowledge from the pre-training stage, thereby enhancing downstream query-based learning performance.
this section cite: []

Section: Preliminaries
3D Gaussian Splatting (3DGS) [19] represents 3D scenes through collections of K Gaussians. Each primitive g k contains 3D position µ k ∈ R 3 , covariance Σ k , opacity α k ∈ [0, 1], and spherical harmonics coefficients c k ∈ R k .
For differentiable optimization, the covariance matrix is parameterized using scaling S ∈ R 3 + and rotation R ∈ R 4 matrices:
Σ = RSS T R T .
(1) Projection to image coordinates employs view transformation W and Jacobian J:
Σ ′ = JWΣW T J T .(2)
Rendering combines ordered Gaussians using an alpha-blend rendering proceduure [37], and color at pixel p is computed as:
C(p) = i∈K c i α i i-1 j=1 (1 -α i ).(3)
To introduce geometric representation [8], depth rendering is computed as:
D(p) = i∈K d i α i i-1 j=1 (1 -α j ),(4)
where d i represents the distance from i-th Gaussian to the camera. Unlike volume rendering [37], 3DGS uses efficient splat-based rasterization that projects 3D Gaussians as 2D image patches.
this section cite: ['b18', 'b36', 'b7', 'b36']

Section: Image Encoder
Given a set of multi-view images
I = {I i ∈ R 3×H×W |i = 1, ..., N }, corresponding intrinsics K = {K i ∈ R 3×3 |i = 1, ..., N } and extrinsics T = {T i ∈ R 4×4 |i = 1, .
.., N } as inputs, where N is the number of cameras. We first need to extract multi-scale multi-view image features for the subsequent decoder. Specifically, we feed multi-view images to the backbone network (e.g., ResNet-101 [11]), and obtain the intermediate multi-level feature F ′ . To further enhance and aggregate these features across different spatial resolutions, we utilize a Feature Pyramid Network (FPN). The FPN processes the multi-level features and produces multi-scale image features F , which effectively captures both high-level semantic information and fine-grained spatial details.
this section cite: ['b10']

Section: Gaussian Transformer Decoder and Gaussian Queries
As illustrated at the top of Fig. 2, SQS employs a Gaussian Transformer Decoder to process 2D image features and reconstruct multi-view RGB and depth images. This reconstruction enables the model to capture the underlying geometry and appearance of Gaussian attributes, providing a strong feature prior. As a result, SQS enhances downstream sparse perception tasks by supplying a pretrained image backbone and enriched Gaussian query representations.
Each Gaussian query is initialized as learnable anchors g k ∈ R K×C , paired with queries q k ∈ R K×D using zero vectors in high-dimensional space, where K is the number of Gaussians, C and D are the dimension of Gaussian primitives and query features respectively. During pre-training, guided by the initialized vectors with learnable Gaussian primitives, these query features interact through self-encoding and deformable cross-attention with image features to predict the Gaussian attributes, enabling the retention of rich and detailed geometric information.
To capture the representation across the entire scene, 3D sparse convolution across Gaussian queries is employed to reduce memory cost with linear computational complexity. Here, the 3D position µ k ∈ R 3 in Gaussian anchor is used to voxelize each Gaussian and the sparse convolution is leveraged on the voxel grid.
Then the multi-level image feature F is aggregated with deformable cross attention. Concretely, for each Gaussian query, multiple 3D reference points are calculated with offsets added to the 3D position µ from the Gaussian anchor. With the intrinsics K and extrinsics T , these 3D points are projected onto 2D image planes to facilitate feature sampling. The resulting set of sampled features serves as keys and values in the subsequent attention mechanism.
Finally, to enable the prediction of Gaussian properties, a dedicated Gaussian head comprising multilayer perceptrons (MLPs) is applied to each Gaussian query. To constrain the predicted parameters within appropriate value ranges, sigmoid activations are applied to the position, scale, and opacity, while the rotation is normalized to unit length. The Gaussian parameters µ, α, c, S and R are iteratively refined across decoder layers, where only the position µ is predicted in the form of a delta, while the remaining parameters are directly replaced at each refinement step.
this section cite: []

Section: Reconstruction Loss for Pre-training
We employ an L1 loss function for both RGB and depth reconstruction. LiDAR points serve as the ground truth (GT) for depth, and the depth loss is supervised exclusively at pixels corresponding to valid LiDAR measurements. The overall loss is formulated as follows:
L = ω 1 L rgb + ω 2 L depth ,(5)
where ω 1 and ω 2 are set to 1.0 and 0.05, respectively, to weight the corresponding terms.
this section cite: []

Section: Query Interaction for Fine-tuning
Through query-based pre-training, both the image backbone and Gaussian queries acquire rich geometric feature. For fine-tuning downstream SPMs, loading the pre-trained image backbone is straightforward; however, reusing pre-trained Gaussian queries remains challenging. Unlike Dense BEV-Centric perception frameworks, which leverage dense BEV representations as a unified intermediate feature, SPMs typically lack such a common structural basis. Instead, these methods often employ task-specific queries and are paired with dedicated decoders designed for different perception tasks. The decoder architectures across various sparse perception algorithms can differ significantly. For instance, SparseBEV [31] initializes queries in 2D BEV space, whereas PETR [33] conducts initialization in 3D space.
To enable consistent reuse of pre-trained queries across diverse sparse perception tasks, as shown in the bottom of Fig. 2, we propose a plug-in framework based on Query Interaction, facilitating greater flexibility and generalization across diverse tasks. This module explicitly bridges pre-trained Gaussian queries with task-specific queries, facilitating effective transfer and adaptation within heterogeneous decoder frameworks.
Specifically, downstream SPMs initially load the weights of image backbone from a sparse pre-trained model. Regarding the reuse of pre-trained Gaussian queries, we fix the parameters of the lightweight pre-trained model. For each perception case, the pre-trained model infers a set of corresponding Gaussian anchors paired with query features. We set opacity threshold α thresh to filter out anchors with low opacity. To leverage pre-trained queries efficiently, spatial-aware local attention [46] is applied. To elaborate, given 3D position µ t of task query q t and µ k from pre-trained anchors g k , we apply k-nearest neighbor algorithm to find k closest 3D Gaussians for each task query. To this end, q t only aggregates features from nearest k Gaussian queries, finally the local query interaction is formulated as follows:
q t = LocalAttn(q t + MLP(µ t ), q k + MLP(g k )).(6)
4 Experiments
this section cite: ['b30', 'b32', 'b45']

Section: Experimental Settings

this section cite: []

Section: Dataset and Metrics.
We conduct experiments on the nuScenes dataset [3], a large-scale benchmark specifically curated for autonomous driving research. The dataset comprises 700 training scenes, 150 validation scenes, and 150 test scenes. Each scene includes synchronized sensor data from six surround-view cameras and LiDAR, enabling comprehensive 3D perception across diverse urban environments. Comprehensive annotations are provided to support multiple tasks, including 3D object detection, LiDAR semantic segmentation, and 3D map segmentation. Building upon the nuScenes dataset, SurroundOcc [57] provides the dense 3D semantic occupancy annotation tailored for the occupancy prediction task. The annotated voxel grid spans [-50m, 50m] along both X and Y axes, and [-5m, 3m] along the Z axis with a resolution of 200 × 200 × 16. Each voxel is assigned one of 18 classes, comprising 16 semantic categories, an empty class, and an unknown class.
The quality of semantic occupancy prediction is evaluated using the mean Intersection-over-Union (mIoU) and Intersection-over-Union (IoU) metrics [49]. For 3D object detection, we adopt the standard nuScenes Detection Score (NDS) and mean Average Precision (mAP) metrics [3]. We also contain five true positive (TP) metrics, including ATE, ASE, AOE, AVE, and AAE for measuring translation, scale, orientation, velocity, and attribute errors, respectively.
this section cite: ['b2', 'b56', 'b48', 'b2']

Section: Implementation Details.
During the pre-training stage, we adopt a ResNet101-DCN [11] backbone initialized from an FCOS3D [54] checkpoint for the occupancy prediction task, while ResNet50 and ResNet101 backbones that are pre-trained with nuImages [3] for the 3D object detection task. The feature extraction employs a feature pyramid network [25] (FPN), producing multi-scale image representations at downsampling factors of 4, 8, 16, and 32. We configure the Gaussian counts to 25,600, and apply two transformer layers to enhance Gaussian attributes. Model training utilizes the AdamW [35] optimizer, with a 0.01 weight decay. The learning rate linearly warms up over the initial 500 steps to 2e-4 and then follows a cosine decay schedule. Pre-training is conducted for 20 epochs using a batch size of 8. Only random horizontal flipping data augmentation is included. Our implementation is based on MMDetection3D [9]. Fine-tuning follows the official downstream model configurations without modification. All experiments are conducted on a server with 8 GPUs. Figure 3: Data efficiency analysis. To assess data efficiency under limited annotation scenarios, we reduce the amount of labeled data used for downstream fine-tuning in the 3D semantic occupancy prediction task. The outcomes demonstrate that our pre-training method significantly enhances performance, even when only a small portion of annotations is available.
this section cite: ['b10', 'b53', 'b2', 'b24', 'b34', 'b8']

Section: Main Results
We evaluate the effectiveness of SQS on two challenging downstream perception tasks: semantic occupancy prediction and 3D object detection.
this section cite: []

Section: Semantic Occupancy Prediction.
In Tab. 1, we present a comprehensive quantitative comparison of various methods for multi-view 3D semantic occupancy prediction on the SurroundOcc validation set. Among these methods, GaussianFormer [16] is a novel query-based occupancy prediction method, performing on par with OccFormer [70] and SurroundOcc [57]. After being pre-trained by our method, the GaussianFormer obtains 1.69 IoU and 1.30 mIoU improvements, achieving 20.40% mIoU, when compared to the 19.10% mIoU for GaussianFormer. These results highlight the effectiveness of SQS for the query-based semantic occupancy prediction task.
this section cite: ['b15', 'b69', 'b56']

Section: 3D Object Detection.
We also have conducted experiments in the 3D object detection task, the results are illustrated in Tab. 2. To validate the generality of SQS, we have performed two different sparse object detection methods, e.g., SparseBEV [31] and Sparse4Dv3 [29] on the nuScenes validation set. When leveraging the ResNet50 as the image backbone, and the input image size is 704 × 256, SparseBEV achieves 55.8 NDS performance, and an impressive 44.8 mAP metric. After being pre-trained by SQS, we reach the 56.6 NDS and 45.2 mAP performance. Meanwhile, we set the new performance record, that is 56.9 NDS and 47.4 mAP for the Sparse4Dv3 being pre-trained by SQS. Then, we upgrade the backbone to ResNet101 and scale the input size to 1408 × 512. Under this setting, the SparseBEV also benefits from our pre-training paradigm with 0.8 mAP and 1.0 NDS improvements. Likewise, Sparse4Dv3 obtains corresponding improvements of 0.7 mAP and 0.8 NDS. The results also validate the effectiveness and generality of our pre-training paradigm.
Table 2: 3D object detection results on the nuScenes val split. † benefits from perspective pre-training [31]. ‡ indicates methods with CBGS [73] which will elongate 1 epoch into 4.5 epochs.
this section cite: ['b30', 'b28', 'b30', 'b72']

Section: Method Backbone Input Size Epochs NDS mAP mATE mASE mAOE mAVE mAAE
PETRv2 [34] ResNet50 704 × 256 60 45.6 34.9 0.700 0.275 0.580 0.437 0.187 BEVStereo [21] ResNet50 704 × 256 90 ‡ 50.0 37.2 0.598 0.270 0.438 0.367 0.190 BEVPoolv2 [12] ResNet50 704 × 256 90 ‡ 52.6 40.6 0.572 0.275 0.463 0.275 0.188 SOLOFusion [43] ResNet50 704 × 256 90 ‡ 53.4 42.7 0.567 0.274 0.511 0.252 0.181 Sparse4Dv2 [28] ResNet50 704 × 256 100 53.9 43.9 0.598 0.270 0.475 0.282 0.179 StreamPETR † [53] ResNet50 704 × 256 60 55.0 45.0 0.613 0.267 0.413 0.265 0.196 SparseBEV [31] ResNet50 704 × 256 36 54.5 43.2 0.606 0.274 0.387 0.251 0.186 SparseBEV † [31] ResNet50 704 × 256 36 55. 8 44.8 0.581 0.271 0.373 0.247 0.190 SparseBEV † + SQS (Ours) ResNet50 704 × 256 36 56.6 45.2 0.564 0.263 0.362 0.232 0.182 Sparse4Dv3 † [29] ResNet50 704 × 256 100 56.1 46.9 0.553 0.274 0.476 0.227 0.200 Sparse4Dv3 † + SQS (Ours) ResNet50 704 × 256 100 56.9 47.4 0.542 0.266 0.458 0.218 0.191 DETR3D † [55] ResNet101-DCN 1600 × 900 24 43.4 34.9 0.716 0.268 0.379 0.842 0.200 BEVFormer † [24] ResNet101-DCN 1600 × 900 24 51.7 41.6 0.673 0.274 0.372 0.394 0.198 BEVDepth [22] ResNet101 1408 × 512 90 ‡ 53.5 41.2 0.565 0.266 0.358 0.331 0.190 Sparse4D † [26] ResNet101-DCN 1600 × 900 48 55.0 44.4 0.603 0.276 0.360 0.309 0.178 SOLOFusion [43] ResNet101 1408 × 512 90 ‡ 58.2 48.3 0.503 0.264 0.381 0.246 0.207 SparseBEV † [31] ResNet101 1408 × 512 24 59.2 50.1 0.562 0.265 0.321 0.243 0.195 SparseBEV † + SQS (Ours) ResNet101 1408 × 512 24 60.2 50.9 0.531 0.251 0.318 0.241 0.185 Sparse4Dv3 † [29] ResNet101 1408 × 512 100 62.3 53.7 0.511 0.255 0.306 0.194 0.192 Sparse4Dv3 † + SQS (Ours) ResNet101 1408 × 512 100 63.1 54.4 0.498 0.241 0.298 0.187 0.188
this section cite: ['b33', 'b20', 'b11', 'b42', 'b27', 'b52', 'b30', 'b30', 'b7']

Section: Render Results

this section cite: []

Section: Ground Truth
Figure 4: Rendering results visualization. Leveraging multi-view images and depth maps projected by the sparse point cloud as supervision, SQS demonstrates compelling depth and image reconstruction after pre-training.
this section cite: []

Section: Data Efficiency.
One of the main advantages of pre-training lies in its ability to improve data efficiency for downstream tasks, especially when annotated data is limited. To further demonstrate the effectiveness of our pre-training strategy under conditions where there is plenty of pre-training data but restricted access to labeled downstream samples, we fine-tune our model-initially pre-trained on the full dataset-using different fractions (10%, 25%, 50%, and 100%) of the SurroundOcc training set.
Fig. 3 showcases how SQS improves data efficiency. With full fine-tuning data, SQS yields improvements of +1.69 IoU and +1.3 mIoU over the baseline. Notably, this benefit is amplified as less fine-tuning data is used: for instance, fine-tuning with only 10% of the data results in a gain of about +3.7 mIoU. These findings highlight the strength of SQS in achieving notable performance improvements through query-based splatting pre-training, particularly when downstream annotated data is scarce.
this section cite: []

Section: Visualization of Renderings.
As shown in Fig. 4, employing 25,600 queries for 3DGS reconstruction through the multi-view RGB images and depth maps as supervision, SQS could predict promising depth and RGB images during the pre-training stage.
this section cite: []

Section: Ablation Studies
In this section, we conduct ablations on the semantic occupancy prediction task on the validation split of the SurroundOcc dataset. In order to reduce the training time, we utilize the quarter of training data during the pre-training and fine-tune stage for all experiments. The results are demonstrated in Tab. 3.
Rendering Objectives. We first investigate the impact of various rendering objectives during the pretraining stage. Specifically, in ModelA, ModelB, and ModelC of Tab. 3, we employ RGB rendering only, depth rendering only, and a combination of both RGB and depth rendering as the pre-training objectives, respectively. The results reveal that utilizing only RGB rendering during pre-training impairs fine-tuning performance, resulting in a reduction of 2.0 in IoU and 3.0 in mIoU. In contrast, incorporating depth rendering alone leads to improvements of 2.1 in both IoU and mIoU metrics. These findings suggest that rendered depth supervision enhances the geometric representation capability of the pre-trained model, thereby facilitating improved fine-tuning performance. Furthermore, when both RGB and depth renderings are jointly applied, we observe a marginal additional improvement. This indicates that rendered RGB supervision provides supplementary benefits in the presence of rendered depth supervision.
this section cite: []

Section: Effects of Query Interaction.
To further assess the impact of query interaction during the fine-tuning stage, we develop Model D, which exclusively incorporates the query interaction mechanism during fine-tuning. As presented in Tab. 3, the pre-trained model is capable of generating meaningful queries for reconstruction, which can be further leveraged to enhance the query learning process through the query interaction module during fine-tuning. This results in improvements of 0.5 IoU and 0.7 mIoU. To eliminate the influence of extra query interaction during the fine-tuning stage, we additionally design Model E to exclusively adapt the query interaction module without pre-training. Its performance remains nearly identical to that of the baseline, indicating that the additional query interaction module offers negligible benefit during fine-tuning. Finally, by initializing with the pre-trained image backbone and FPN neck, we obtain optimal fine-tuning performance, reaching 28.5% IoU and 18.0% mIoU. These results demonstrate the superiority of the query interaction design within our proposed SQS paradigm.
this section cite: []

Section: Limitations and Future Work
While SQS has achieved further improvements across various downstream tasks, becoming a plugand-play general pre-training paradigm for sparse perception models, it still faces several limitations. One limitation is the extra computation burden and memory consumption incurred by the plug-in pretraining model. Another limitation is the insufficient utilization of pre-training queries for different downstream tasks.
In the future, we will explore how to introduce the semantic information during the pre-training stage and then use the semantic information to distinguish the pre-trained queries for various downstream tasks. We will also try to apply the SQS to query-based end-to-end autonomous driving approaches such as SparseAD [67] and GaussianAD [71].
this section cite: ['b66', 'b70']

Section: Conclusion
In this paper, we introduced SQS, a novel query-based splatting pre-training paradigm tailored for autonomous driving SPMs. SQS overcomes the limitations of previous pre-training methods by enabling image backbone and Gaussian queries to learn rich 3D representations through 3D Gaussian prediction and the reconstruction of both images and depth maps. The plug-in design and query interaction strategy further allow seamless transfer and adaptation of the pre-trained model to diverse downstream tasks. Extensive experiments on benchmark datasets validate the effectiveness of SQS, showing promising improvements over various SOTA SPMs.
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [No] Justification: We plan to release the code upon acceptance. Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
this section cite: []

Section: References
Ref_id:b0 Title: Gaussianflowocc: Sparse and weakly supervised occupancy estimation using gaussian splatting and temporal flow Year: (2025)
Ref_id:b1 Title: ALSO: automotive lidar self-supervision by occupancy estimation Year: (2023)
Ref_id:b2 Title: nuscenes: A multimodal dataset for autonomous driving Year: (2020)
Ref_id:b3 Title: Monoscene: Monocular 3d semantic scene completion Year: (2022)
Ref_id:b4 Title: End-to-end object detection with transformers Year: (2020)
Ref_id:b5 Title: Gaussianbev: 3d gaussian representation meets perception models for bev segmentation Year: (2024)
Ref_id:b6 Title: End-to-end autonomous driving: Challenges and frontiers Year: (2024)
Ref_id:b7 Title: Gaussianpro: 3d gaussian splatting with progressive propagation Year: (2024)
Ref_id:b8 Title: MMDetection3D: OpenMMLab next-generation platform for general 3D object detection Year: (2020)
Ref_id:b9 Title: Frozen-detr: Enhancing detr with image understanding from frozen foundation models Year: (2024)
Ref_id:b10 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b11 Title: Bevpoolv2: A cutting-edge implementation of bevdet toward deployment Year: (2022)
Ref_id:b12 Title: Bevdet: High-performance multi-camera 3d object detection in bird-eye-view Year: (2021)
Ref_id:b13 Title: Probabilistic gaussian superposition for efficient 3d occupancy prediction Year: (2024)
Ref_id:b14 Title: Tri-perspective view for vision-based 3d semantic occupancy prediction Year: (2023)
Ref_id:b15 Title: Gaussianformer: Scene as gaussians for vision-based 3d semantic occupancy prediction Year: (2024)
Ref_id:b16 Title: Gausstr: Foundation model-aligned gaussian transformer for self-supervised 3d spatial understanding Year: (2024)
Ref_id:b17 Title: Far3d: Expanding the horizon for surround-view 3d object detection Year: (2024)
Ref_id:b18 Title: 3d gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b19 Title: Masked autoencoder for large-scale lidar point clouds Year: (2024)
Ref_id:b20 Title: Enhancing depth estimation in multi-view 3d object detection with dynamic temporal stereo Year: (2022)
Ref_id:b21 Title: Bevdepth: Acquisition of reliable depth for multi-view 3d object detection Year: (2023)
Ref_id:b22 Title: Simipu: Simple 2d image and 3d point cloud unsupervised pre-training for spatial-aware visual representations Year: (2022)
Ref_id:b23 Title: Bevformer: Learning bird's-eye-view representation from multi-camera images via spatiotemporal transformers Year: (2022)
Ref_id:b24 Title: Feature pyramid networks for object detection Year: (2017)
Ref_id:b25 Title: Sparse4d: Multi-view 3d object detection with sparse spatial-temporal fusion Year: (2022)
Ref_id:b26 Title: Sparse4d v2: Recurrent temporal fusion with sparse model Year: (2023)
Ref_id:b27 Title: Sparse4d v2: Recurrent temporal fusion with sparse model Year: (2023)
Ref_id:b28 Title: Sparse4d v3: Advancing end-to-end 3d detection and tracking Year: (2023)
Ref_id:b29 Title: Ray denoising: Depth-aware hard negative sampling for multi-view 3d object detection Year: (2024)
Ref_id:b30 Title: Sparsebev: High-performance sparse 3d object detection from multi-camera videos Year: (2023)
Ref_id:b31 Title: Glid: Pre-training a generalist encoder-decoder vision model Year: (2024)
Ref_id:b32 Title: Petr: Position embedding transformation for multi-view 3d object detection Year: (2022)
Ref_id:b33 Title: Petrv2: A unified framework for 3d perception from multi-camera images Year: (2023)
Ref_id:b34 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b35 Title: Gaussianlss-toward real-world bev perception: Depth uncertainty estimation via gaussian splatting Year: (2025)
Ref_id:b36 Title: Nerf: Representing scenes as neural radiance fields for view synthesis Year: (2020)
Ref_id:b37 Title: Occupancy-mae: Self-supervised pretraining large-scale lidar point clouds with masked occupancy autoencoders Year: (2023)
Ref_id:b38 Title: Voxel-mae: Masked autoencoders for pre-training large-scale point clouds Year: (2022)
Ref_id:b39 Title: Atlas: End-to-end 3d scene reconstruction from posed images Year: (2020)
Ref_id:b40 Title: Segcontrast: 3d point cloud feature representation learning through self-supervised segment discrimination Year: (2022)
Ref_id:b41 Title: Is pseudo-lidar needed for monocular 3d object detection? Year: (2021)
Ref_id:b42 Title: Time will tell: New outlooks and a baseline for temporal multi-view 3d object detection Year: (2022)
Ref_id:b43 Title: Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d Year: (2020)
Ref_id:b44 Title: BEVContrast: Self-supervision in bev space for automotive lidar point clouds Year: ()
Ref_id:b45 Title: Motion transformer with global intention localization and local movement refinement Year: (2022)
Ref_id:b46 Title: Occupancy as set of points Year: (2024)
Ref_id:b47 Title: Sparseocc: Rethinking sparse latent representation for vision-based semantic occupancy prediction Year: (2024)
Ref_id:b48 Title: Occ3d: A large-scale 3d occupancy prediction benchmark for autonomous driving Year: (2024)
Ref_id:b49 Title: Scene as occupancy Year: (2023)
Ref_id:b50 Title: Opus: Occupancy prediction using a sparse set Year: (2024)
Ref_id:b51 Title: Exploring object-centric temporal modeling for efficient multi-view 3d object detection Year: (2023)
Ref_id:b52 Title: Exploring object-centric temporal modeling for efficient multi-view 3d object detection Year: (2023)
Ref_id:b53 Title: Fcos3d: Fully convolutional one-stage monocular 3d object detection Year: (2021)
Ref_id:b54 Title: Detr3d: 3d object detection from multi-view images via 3d-to-2d queries Year: (2022)
Ref_id:b55 Title: Omni-scene: Omni-gaussian representation for ego-centric sparse-view scene reconstruction Year: (2024)
Ref_id:b56 Title: Surroundocc: Multicamera 3d occupancy prediction for autonomous driving Year: (2023)
Ref_id:b57 Title: Gaussianpretrain: A simple unified 3d gaussian representation for visual pre-training in autonomous driving Year: (2024)
Ref_id:b58 Title: Spot: Scalable 3d pre-training via occupancy prediction for autonomous driving Year: (2023)
Ref_id:b59 Title: Forging vision foundation models for autonomous Year: (2024)
Ref_id:b60 Title: Street gaussians for modeling dynamic urban scenes Year: (2024)
Ref_id:b61 Title: Unipad: A universal pre-training paradigm for autonomous driving Year: (2024)
Ref_id:b62 Title: Storm: Spatio-temporal reconstruction model for large-scale outdoor scenes Year: (2025)
Ref_id:b63 Title: Visual point cloud forecasting enables scalable autonomous driving Year: (2024)
Ref_id:b64 Title: Ad-pt: Autonomous driving pre-training with large-scale point cloud dataset Year: (2024)
Ref_id:b65 Title: A multimodal world model for autonomous driving via unified bev latent space Year: (2024)
Ref_id:b66 Title: Sparsead: Sparse query-centric paradigm for efficient end-to-end autonomous driving Year: (2024)
Ref_id:b67 Title: Tt-gaussocc: Test-time compute for self-supervised occupancy prediction via spatio-temporal gaussian splatting Year: (2025)
Ref_id:b68 Title: Visionpad: A vision-centric pre-training paradigm for autonomous driving Year: (2024)
Ref_id:b69 Title: Occformer: Dual-path transformer for vision-based 3d semantic occupancy prediction Year: (2023)
Ref_id:b70 Title: Gaussian-centric end-to-end autonomous driving Year: (2024)
Ref_id:b71 Title: Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes Year: (2024)
Ref_id:b72 Title: Class-balanced grouping and sampling for point cloud 3d object detection Year: (2019)
Ref_id:b73 Title: Mim4d: Masked modeling with multi-view video for autonomous driving representation learning Year: (2024)
Ref_id:b74 Title: Gaussian world model for streaming 3d occupancy prediction Year: (2024)
