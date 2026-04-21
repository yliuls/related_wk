Title: Auto-regressive Transformer Skeleton …
Abstract: Modern interactive applications increasingly demand dynamic 3D content, yet the transformation of static 3D models into animated assets constitutes a significant bottleneck in content creation pipelines. While recent advances in generative AI have revolutionized static 3D model creation, rigging and animation continue to depend heavily on expert intervention. We present Puppeteer, a comprehensive framework that addresses both automatic rigging and animation for diverse 3D objects. Our system first predicts plausible skeletal structures via an auto-regressive transformer that introduces a joint-based tokenization strategy for compact representation and a hierarchical ordering methodology with stochastic perturbation that enhances bidirectional learning capabilities. It then infers skinning weights via an attention-based

Section: Introduction
From AAA games and animated films to VR/AR experiences and robotic simulations, modern interactive media demands dynamic 3D content. While recent generative AI advances have accelerated the creation of high-fidelity 3D models with intricate geometry and textures, these assets remain predominantly static. Transforming static 3D models into animated versions requires two expertdriven processes: rigging (skeleton setup and skinning weight assignment) and animation. This manual, time-intensive workflow now constitutes a significant impediment to the efficiency of modern content creation pipelines.
Research communities have invested considerable effort in automating the rigging process. Early template-based techniques such as Pinocchio [6] fit predefined skeletal structures to input meshes, achieving satisfactory results on specific categories but failing to generalize to arbitrary shapes. Template-free algorithms [25,3,7,42,71] extract skeletal structures directly from geometric properties but frequently produce excessively dense or topologically incompatible joint configurations unsuitable for practical animation workflows. Deep learning approaches have substantially advanced the field: RigNet [86] pioneered direct skeleton and skinning weight prediction from input shapes using graph neural networks, while MagicArticulate [67] reformulated skeleton generation as an autoregressive problem and introduced a large-scale dataset with detailed rigging annotations. Despite these innovations, significant challenges persist: RigNet struggles with complex mesh topologies due to its reliance on carefully crafted features and restrictive orientation requirements. MagicArticulate suffers from computational inefficiency during inference and limited generalization in its functional diffusion process for skinning weight prediction. Critically, both approaches address only the rigging stage of the pipeline, leaving the equally challenging animation process as a separate manual task that requires substantial expertise.
In this work, we present Puppeteer, a comprehensive framework that integrates automatic rigging and animation into a unified pipeline. To address the data scarcity and limited pose diversity in existing datasets, we expand the Articulation-XL dataset [67] to 59.4k rigged models, including a carefully curated subset of 11.4k diverse pose examples that enhance generalization to varied pose inputs. This expanded dataset serves as the foundation for our learning-based approach. To overcome the limitations of existing rigging approaches in handling diverse shapes and complex topologies, our system introduces key improvements to both fundamental rigging components. For skeleton generation, we employ auto-regressive transformers featuring joint-based tokenization and hierarchical sequence ordering with randomization, creating more compact representations while generating structurally coherent skeletons free from template dependencies. For skinning weight prediction, we propose an attention-based architecture incorporating topology-aware joint attention that explicitly encodes skeletal graph structure, achieving robust weight prediction with enhanced generalization and computational efficiency. Beyond rigging, we address the automatic animation challenge that previous methods have largely overlooked. We introduce a differentiable optimizationbased method that requires no neural network parameters yet produces stable, high-quality animations by combining our generated rigging with reference video guidance easily obtained from off-theshelf video generation models. Our unified framework enables full automation from static meshes to animated assets, transforming the labor-intensive manual workflow into an efficient, accessible pipeline for diverse 3D content creation.
Extensive evaluations demonstrate the effectiveness of our approach across both rigging and animation tasks. For rigging, experiments on the expanded Articulation-XL2.0 dataset and ModelsResource benchmark [76,85] show significant improvements over state-of-the-art methods in skeleton accuracy and skinning weight quality. The robustness of our approach is further validated through successful application to diverse 3D content-from professionally designed game assets to AI-synthesized geometries. For animation, direct comparisons against recent 4D generation techniques [77,59] show that our optimization-based approach produces more temporally consistent and visually faithful results while maintaining computational efficiency. Notably, our method eliminates the jittering artifacts commonly seen in learning-based approaches during complex motion sequences. The clean and stable animation results also highlight the reliability of our automatically generated rigs.
In summary, our work advances automated 3D model rigging and animation through four key contributions: (1) An expanded large-scale articulation dataset with 59.4k rigged models including a diverse-pose subset; (2) A novel auto-regressive skeleton generation approach featuring efficient jointbased tokenization and hierarchical sequence ordering with randomization strategies; (3) An attentionbased architecture for skinning weight prediction incorporating topology-aware joint attention; and (4) A differentiable optimization-based animation method that produces stable, high-quality animation for diverse object categories without requiring extensive computational resources or manual effort.
this section cite: ['b5', 'b24', 'b2', 'b6', 'b41', 'b70', 'b85', 'b66', 'b66', 'b75', 'b84', 'b76', 'b58', 'b2']

Section: Related works
We discuss the related works in automatic rigging here, while related works in 3D animation can be found in the appendix.
this section cite: []

Section: Skeleton generation.
Skeleton generation methods for 3D models fall into two main groups. The first leverages templates or additional inputs. Pinocchio [6] pioneered template-fitting for automatic skeleton extraction, while Li et al. [37] employed deep learning for human joint estimation with a given skeleton template. Some recent works [13,24,69] continue this line for humanoid skeleton generation. A significant limitation of these approaches is their inability to generalize effectively to diverse object categories. Other methods in this group require additional inputs such as point cloud sequences [87], mesh sequences [14,30], manual annotations [26], or video data [88,81,66,98,90,65,68,40]. The second group works without templates or annotations. Traditional approaches [3,7,25,71,42] extract curve skeletons and often produce overly dense joints unsuitable for animation. Modern deep learning methods like Xu et al. [85] and RigNet [86] learn directly from limited datasets containing fewer than 3,000 rigged models. Despite their innovations, these methods depend extensively on carefully crafted features and impose restrictive assumptions regarding shape orientation, substantially constraining their effectiveness when confronted with complex mesh topologies.
With the exponential growth of 3D datasets [15,16] and the success of auto-regressive approaches in 3D generation [62,11,12,72], the field has seen significant advances in skeleton generation. MagicArticulate [67] pioneered the formulation of skeleton generation as an auto-regressive problem and introduced Articulation-XL, a large-scale 3D dataset with rigging information. Several recent works [45,100] have also successfully incorporated auto-regressive transformer architectures for skeleton generation, further validating this approach. In our work, we substantially expand the Articulation-XL dataset from 33k to 59.4k rigged models, including a diverse pose subset containing 11.4k examples. We leverage auto-regressive transformers for skeleton generation, introducing two key innovations: an efficient tokenization method for skeletal structures and a hierarchical sequence ordering strategy with randomization that enhances bidirectional learning capabilities.
this section cite: ['b5', 'b36', 'b12', 'b23', 'b68', 'b86', 'b13', 'b29', 'b25', 'b87', 'b80', 'b65', 'b97', 'b89', 'b64', 'b67', 'b39', 'b2', 'b6', 'b24', 'b70', 'b41', 'b84', 'b85', 'b14', 'b15', 'b61', 'b10', 'b11', 'b71', 'b66', 'b44', 'b99']

Section: Skinning weight prediction.
Following skeleton generation, automatic rigging requires skinning weights prediction to establish joint influence on mesh vertices. Traditional geometric approaches [18,28,19,6] assign weights based on vertex-joint distances-a method that proves inadequate for complex topologies. Learning-based approaches [46,86,54,55] consistently integrate graph neural networks (GNN) with geometric distance cues for skinning weight prediction. However, these GNN-based methodologies face significant limitations in scalability and struggle to generalize effectively across 3D data with diverse spatial orientations. MagicArticulate [67] formulates skinning weight prediction as a functional diffusion problem [97], but suffers from slow inference and limited generalization. We instead introduce an attention-based network that strategically incorporates skeleton graph distances, enabling more robust skinning weight prediction with substantially enhanced generalization across diverse object categories. Concurrent works [100,17] similarly leverage crossattention between surface points and bones to learn skinning weights.
this section cite: ['b17', 'b27', 'b18', 'b5', 'b45', 'b85', 'b53', 'b54', 'b66', 'b96', 'b99', 'b16']

Section: Automatic rigging
Our automatic rigging framework features two sequential modules. First, we deploy an auto-regressive transformer to infer a structurally valid skeleton from a raw 3D mesh (Section 3.2). Subsequently, this skeleton and the original mesh are processed by an attention-based architecture to predict precise per-vertex skinning weights (Section 3.3). To facilitate large-scale learning, we introduce Articulation-XL2.0 (Section 3.1), a comprehensive dataset comprising 59.4k 3D models with high-quality rigging.
this section cite: []

Section: Dataset: Articulation-XL2.0
We present Articulation-XL2.0, an expanded version of Articulation-XL proposed in [67]. Our dataset incorporates multiple geometric data types from Objaverse-XL [15,16] previously excluded, while maintaining the same data filtering process. We further improve quality by eliminating unskinned vertices and conducting manual validation, yielding over 48k high-quality rigged 3D models.
Recognizing that models in our primary dataset are predominantly in rest pose configurations, thus limiting generalization capacity to novel articulations, we have constructed a diverse-pose subset. By identifying the intersection between high-quality animation data from Diffusion4D [41] and our rigged model corpus, we extract 7.3k deformed meshes with corresponding rigging information from animation frames exhibiting maximal deviation from rest pose configurations. To counterbalance the predominance of humanoid morphologies in this subset, we supplement with 4.1k models generated using SMALR [107,108] with parameterizations derived from 41 distinct animal scans and randomized valid poses. The resulting 11.4k diverse-pose dataset significantly enhances performance on unseen poses, as validated in our experiments. We have released Articulation-XL2.0, a comprehensive collection of 59.4k high-quality rigged models, to facilitate future research. Dataset statistics and examples are provided in the appendix.
this section cite: ['b66', 'b14', 'b15', 'b40', 'b106', 'b107']

Section: Auto-regressive skeleton generation
We formulate skeleton generation as a shape-conditioned sequence modeling problem. Given an input mesh M, we employ an auto-regressive framework (Figure 2 top) to predict a skeleton S consisting of 3D joint positions J ∈ R j×3 and topological bone connections B ∈ N b×2 defined by joint indices. Our framework consists of three key components: joint-based skeleton tokenization, hierarchical sequence ordering with randomization, and shape-conditioned auto-regressive generation. Together, these components enable accurate, efficient skeleton generation across varied object structures without relying on predefined templates. Joint-based skeleton tokenization. In [67], skeletons are encoded as bone-based sequences: each of the b bones contributes 6 tokens (the 3D coordinates of its two endpoints), yielding a total sequence length of 6b and redundantly repeating joint positions across multiple connected bones. Inspired by [45], we develop a joint-based tokenization strategy that represents each of the j joints by its 3D coordinates and parent index, producing a sequence of length 4j. Since a tree-structured skeleton satisfies j = b + 1, this yields 4j < 6b whenever j > 3, making the joint-based representation more compact. Unlike [45], which projects joint positions into high-dimensional feature spaces via MLPs, we discretize normalized joint coordinates into a 128 3 grid and append the parent index, producing discretized token sequences that serve as input to our auto-regressive transformer. In practice, we assign the root joint a parent index of 0 and offset all other parent indices by +1 (subtracting 1 during detokenization). Sequence ordering. While our joint-based tokenization provides a compact representation, the sequential ordering of tokens significantly affects skeletal coherence and model performance. For skeleton modeling with joint positions and parent indices, tokens can be sequenced using either spatial ordering (ascending z-y-x coordinates, as in [67]) or hierarchical ordering (breadth-first traversal of the skeletal tree structure). Our experiments demonstrate that spatial ordering frequently produces disconnected skeletons, as child joints generated before their parents create invalid parent references (see Section 5.5 and appendix for comparisons). We therefore adopt hierarchical ordering, applying spatial sorting only among joints at the same hierarchical level.
Additionally, inspired by [92], we enhance bidirectional learning capability through sequence randomization. We group the 4 tokens of each joint together and randomly shuffle these groups, incorporating target-aware positional indicators P = [p 0 , p 1 , ..., p j-1 ] to guide the generation process. Specifically, all tokens within a joint group share a positional indicator signaling which joint will be generated next. To identify the first joint group, we additionally incorporate positional indicators into shape tokens T shape that precede skeleton tokens T skel :
T = [T shape , T skel ] + P = [T shape + p 0 , T 0 skel + p 1 , ..., T j-2 skel + p j-1 , T j-1 skel ].(1)
Shape-conditioned auto-regressive generation. With our tokenization strategy and sequence ordering established, we now describe the auto-regressive generation process. We sample 8,192 points with normals from the input mesh as shape conditioning and encode them using a pre-trained shape encoder [105]. This fixed-length shape token sequence T shape precedes the transformer's skeleton sequence, with < bos > and < eos > tokens marking skeleton boundaries (omitted in Equation ( 1)). We adopt OPT-350M [103] as our decoder-only transformer architecture, training with cross-entropy loss for next-token prediction:
L pred = CE(T, T),(2)
where T and T represent ground truth and predicted token sequences. During inference, generation begins with shape tokens and sequential positional indicators, proceeding auto-regressively until producing the < eos > token, followed by detokenization to recover the complete skeleton.
this section cite: ['b66', 'b44', 'b44', 'b66', 'b91', 'b104', 'b102']

Section: Attention-based skinning weight prediction
In this section, we present an attention-based network for predicting per-vertex skinning weights that determine how the mesh deforms in response to skeleton articulation.
Network architecture. The network architecture is illustrated at the bottom of Figure 2. Our pipeline begins by sampling n points with normals from the input mesh. These points are processed through positional encoding and a part encoder from PartField [47] to obtain part-aware point embeddings F point ∈ R n×d that combine spatial information with part features. We incorporate part-aware features because parts and bones exhibit strong anatomical correspondence, providing valuable structural guidance for skinning weight prediction. In parallel, we construct bone-based coordinates ∈ R j×6 by concatenating each joint's parent position with its own position-for the root joint, its position is duplicated to fill both coordinate slots. These bone coordinates similarly undergo positional encoding to produce bone embeddings F bone ∈ R j×d . Additionally, we feed the sampled points with normals into a pre-trained shape encoder [105] to extract global shape latents F shape ∈ R 257×d .
The architecture then performs a series of attention operations [78]:(1) Bone feature enhancement. We first apply self-attention using the topology-aware joint attention on the bone embedding to obtain enhanced bone features F ′ bone . (2) Global context integration. Cross-attention is performed between global shape latents (as context) and both point and bone features, generating updated features F ′ point and F ′′ bone . (3) Bone-point interaction. Cross-attention uses the updated bone features as queries and F ′ point as keys/values to produce refined bone features F ′′′ bone . (4) Point feature refinement. Final cross-attention between refined bone features F ′′′ bone (as context) and point features F ′ point produces the final point features F ′′ point . Finally, the network computes cosine similarity scores and applies softmax normalization to produce skinning weights:
W = softmax α F ′′ point F ′′′⊤ bone F ′′ point ∥F ′′′ bone ∥ . (3
)
where α is a learnable scaling parameter. We optimize the network using cross-entropy loss during training.
Topology-aware joint attention. While the basic architecture effectively predicts weight, explicitly modeling the skeletal structure significantly enhances performance. Our ablation studies demonstrate that using bone-based coordinates ∈ R j×6 rather than joint coordinates ∈ R j×3 substantially improves performance (see Section 5.5), highlighting the importance of inter-joint relationships within the skeletal structure.
To further leverage topological structure, we propose Topology-aware Joint Attention (TAJA), which augments standard self-attention with relative positional encodings derived from skeletal graph distances. To implement TAJA, we first compute a graph distance matrix D ∈ R j×j from the skeletal structure, then transform these distances into continuous embeddings through quantization and projection operations, yielding position embeddings E dis ∈ R j×j×h , where h is the number of attention heads. The attention mechanism is then modified as:
Attention(Q, K, V, E dis ) = softmax QK T √ d k + λE dis V,(4)
where λ is a learnable scaling parameter. This approach explicitly incorporates inter-joint topological relationships, improving the network's capacity to understand skeletal structure and generate more accurate skinning weights.
this section cite: ['b46', 'b104', 'b77']

Section: Video-guided 3D animation
With the generated skeleton and skinning weights, we transform static meshes into animation-ready assets. This section presents our optimization-based approach for automatically animating rigged 3D models with video guidance. Animation pipeline. Our animation process begins by rendering the rigged mesh as the initial frame I 0 . Using this as a conditioning image, we leverage recent text-to-video generation models [34,1] that can maintain object identity while creating plausible motion sequences. With a text prompt describing the desired animation, these models generate a video sequence V = {I 0 , I 1 , ..., I n-1 } comprising n frames. Given this reference video sequence V , we jointly optimize per-frame joint rotations and global root motion of the 3D mesh to align the resulting animation with the generated video sequence.
this section cite: ['b33', 'b0']

Section: Differentiable optimization framework.
For each frame i ∈ {1, 2, ..., n -1} excluding the first frame, we optimize both root motion parameters (Q i root , T i root ) and joint-specific rotations
Q i joint = {Q i 0 , Q i 1 , ..., Q i j-1 },
where Q ∈ R 4 represents rotation as a unit quaternion and T ∈ R 3 denotes translation. For the first frame (rest pose), we initialize all transformations with identity quaternions and zero translations, which remain fixed during optimization. All subsequent frames are similarly initialized before optimization begins. Our optimization process incorporates rendering losses, tracking losses, and regularization terms:
L = (L rgb + L mask + L f low + L depth ) rendering losses + (L joint_track + L vertex_track ) tracking losses +L reg .(5)
For rendering losses, we utilize differentiable rendering via Pytorch3D [57] to generate predicted frames I ′ i and compute RGB, mask, optical flow, and depth discrepancies between these predictions and the corresponding reference video frames. The optical flow and depth for video frames are extracted using off-the-shelf methods [10,53]. The tracking losses incorporate a 2D joint tracking term and a 2D vertex tracking term that leverage Cotracker3 [35] to trace selected points throughout the video sequence. We project our optimized 3D joints and deformed mesh vertices into 2D space and minimize their distance to the corresponding tracked 2D keypoints. To address occlusion challenges, Table 1: Quantitative comparison of skeleton generation. We evaluate each method on three benchmarks using CD-J2J, CD-J2B, and CD-B2B-all reported in units of 10 -2 . Lower values indicate better alignment. * denotes models trained on Articulation-XL2.0 including the diverse-pose subset; unmarked models were trained without it. Bold and underlined numbers denote the best and second-best results, respectively.
this section cite: ['b56', 'b9', 'b52', 'b34']

Section: Method
Articulation-XL2.0 ModelsResource Diverse-pose
J2J ↓ J2B ↓ B2B ↓ J2J ↓ J2B ↓ B2B ↓ J2J ↓ J2B ↓ B2B ↓Pinocchio
8.324 6.612 5.485 6.852 4.824 4.089 7.967 6.411 5.149 RigNet 7.618 6.076 5.279 7.223 5.987 4.329 7.751 6.392 5.713 MagicArti. 3.172 2.419 2.050 4.129 3.149 2.705 4.525 3.602 3.084 UniRig 3.305 2.611 2.180 3.964 3.021 2.570 3.252 2.569 2.077 Ours 3.062 2.342 1.963 3.843 2.876 2.465 3.276 2.597 2.074 Ours* 3.047 2.337 1.952 3.785 2.847 2.430 2.483 1.922 1.600
we implement visibility detection mechanisms for both joints and vertices. For joints, we define visibility based on ray-mesh intersection: a joint is considered visible if the ray projected from the camera to the joint intersects the mesh surface exactly once. We employ the ray_mesh_intersect function from libigl [29] to compute these joint visibility masks. For vertex visibility, we leverage the rasterization output from Pytorch3D to determine visible surface points. These visibility masks, derived from the first frame, ensure that our tracking losses are applied consistently throughout the sequence based on initial visibility, preventing optimization artifacts from elements that are occluded in the reference pose. We further incorporate regularization terms that enforce frame-to-frame motion smoothness. Complete mathematical formulations of all loss components are provided in the appendix.
this section cite: ['b28']

Section: Experiments

this section cite: []

Section: Experimental setup
Datasets. We train our models on the Articulation-XL2.0 dataset introduced in Section 3.1, which contains over 48k high-quality samples from Objaverse-XL [15,16] as the main set and 11.4k samples from the diverse-pose subset. For model training, we utilize over 46k samples from the main subset and 10.9k from the diverse-pose subset. For evaluation, we employ three distinct test sets: Articulation-XL2.0-test (2k data from the main set), ModelsResource-test [76,86] (270 upright, front-facing models with no overlap with Articulation-XL2.0, enabling assessment of cross-dataset generalization), and a 500-mesh portion of the diverse-pose subset specifically selected to evaluate model performance under varied poses.
this section cite: ['b14', 'b15', 'b75', 'b85']

Section: Implementation details.
To enhance robustness and generalization capabilities, we apply geometric data augmentations (scaling, shifting, rotation transformations) and pose augmentation-articulating the training samples with their ground truth skeleton and skinning weights to simulate diverse poses. Further implementation details are provided in the appendix.
this section cite: []

Section: Skeleton generation results
Baselines and metrics. We include four comparison methods as baselines: Pinocchio [6], which fits predefined skeleton templates to input meshes. RigNet [86], a learning-based model that employs graph convolutions to infer joint locations. MagicArticulate [67], an auto-regressive framework for skeleton generation, and the concurrent method UniRig [100], which similarly uses an auto-regressive transformer approach. All methods are evaluated on Articulation-XL2.0 and ModelsResource test sets, as well as our diverse-pose subset. We evaluate skeleton generation quality using three Chamfer Distance-based metrics from [85,86]: CD-J2J (joint-to-joint), CD-J2B (joint-to-bone) and CD-B2B (bone-to-bone). These metrics measure the spatial alignment between generated and ground truth skeletons, where lower values indicate better performance.
this section cite: ['b5', 'b85', 'b66', 'b99', 'b84', 'b85']

Section: Comparison results.
Qualitative results are shown in Figure 3 for all three benchmarks. RigNet consistently produces invalid skeletons-its graph-convolutional model fails to converge well when trained on our large-scale dataset with highly varied orientations. UniRig presents missing and misaligned skeletons, such as missing bones on the turtle limbs and squirrel tail and misaligned skeletons on human hands, as marked in yellow circles. MagicArticulate matches reference skeletons MagicArticulate Artist-created Ours RigNet UniRig closely on Articulation-XL2.0 and ModelsResource, but exhibits errors in fine details (e.g., missing bones in turtle limbs, incorrect squirrel tail-body junctions) and degrades on the diverse-pose subset, since it was trained only on predominantly rest-pose data without pose augmentation. In contrast, our method yields accurate, structurally correct skeletons across three benchmarks. Importantly, our generated skeletons can even correct omissions in artist-created skeletons, such as a missing turtle head-body connection. Table 1 reports quantitative metrics, where we consistently outperform all baselines on every dataset and metric. Notably, incorporating the diverse-pose subset during training leads to marked improvements on the diverse-pose benchmark.
this section cite: []

Section: Skinning weight prediction results
Baselines and metrics. We compare our method for skinning weight prediction against three baselines: Geodesic Voxel Binding (GVB) [18], a geometry-based technique available in Autodesk Maya [27], RigNet [86], and MagicArticulate [67]. We also evaluate these three methods on Articulation-XL2.0 and ModelsResource test sets, as well as our diverse-pose subset. Skinning weight quality is evaluated using three metrics: precision, recall, and L1-norm error. Precision is the fraction of predicted weights > 1e-4 that are correct, and recall is the fraction of true weights > 1e-4 we recover. The L1-norm error reports the average absolute deviation between predicted and ground truth weights over all vertices. Deformation error results are provided in the appendix.
Comparison results. Figure 4 visualizes each method's predicted skinning weights alongside their L1 error maps. Our method produces more accurate weight distributions with substantially lower errors across all benchmarks. RigNet exhibits large errors on all examples, while MagicArticulate's functional diffusion performs well on Articulation-XL2.0 and the diverse-pose subset but degrades on ModelsResource, revealing limited cross-dataset generalization. Quantitative results in Table 2 confirm these observations, with our method outperforming all baselines on every metric and dataset. Moreover, our approach runs faster-achieving per-example inference speeds that are 1.75×, 45×, and 59× those of RigNet, MagicArticulate, and GVB, respectively (see appendix for details).
this section cite: ['b17', 'b26', 'b85', 'b66']

Section: 3D animation results
Baselines. We compare our animation results with L4GM [59] for video-to-4D generation and MotionDreamer [77] for 3D mesh animation. To ensure a fair evaluation, L4GM is given the same input videos and its multi-view synthesis for the first frame is replaced with ground-truth renderings of the input 3D model. MotionDreamer receives the input 3D model along with the same text prompts used for video generation. In Figure 5, some of its outputs appear untextured because its watertight mesh conversion breaks the UV mappings.
this section cite: ['b58', 'b76']

Section: Artist-painted

this section cite: []

Section: Skinning weights
Error map Skinning weights Error map Skinning weights Error map Ours MagicArticulate RigNet  Comparison results. As shown in Figure 5, we present our generated skeletons and the corresponding video-guided animations. The shapes with skeletons represent the rest poses. Although L4GM's reference views are well aligned with the source video, it repeatedly produces geometric distortions (red highlights), even when provided with ground truth multi-view renderings. MotionDreamer's animations are subtle and can introduce unintended deformations in rigid parts (e.g., the humanoid torso). By contrast, our approach produces accurate, artifact-free animations using fully generated rigging.
this section cite: []

Section: Ablation studies
In this section, we present ablation studies on both skeleton generation and skinning weight prediction. All models are trained on Articulation-XL2.0 without the diverse-pose subset.
Ablation studies on skeleton generation. We ablate four components-pose augmentation, order randomization, tokenization scheme, and skeleton ordering strategy-to measure their effects on skeleton generation (see Table 3). Removing pose augmentation degrades performance across all benchmarks, especially on the diverse-pose test. Disabling order randomization similarly reduces accuracy. Bone-based tokenization matches our method's quality but requires 12 extra training hours and is 1.6× slower at inference. Finally, replacing hierarchical ordering with spatial ordering preserves CD-J2J and CD-J2B but markedly increases CD-B2B error and often produces disconnected skeletons; see the appendix for visualization comparisons.
Ablation studies on skinning weight prediction. We evaluate four key components of our skinning weight prediction framework (see Table 4). First, replacing bone embeddings with joint embeddings Reference Ours L4GM MotionDreamer While L4GM [59] aligns its reference views closely with the input video, it consistently exhibits distortions (highlighted in red). MotionDreamer's [77] animations are subtle and can introduce unintended deformations in rigid parts (e.g., the humanoid torso). In contrast, our method delivers accurate, artifact-free animations using fully generated rigging. Videos are included in the project page. increases the average L1-norm error by 4.0% across all three benchmarks, demonstrating the importance of explicitly modeling bone information. Second, replacing Topology-aware Joint Attention (TAJA) with standard self-attention leads to performance degradation across all benchmarks, highlighting the value of modeling topological relationships between joints. Third, removing part-aware features results in consistent performance drops, confirming their contribution to accurate weight prediction. Finally, eliminating pose augmentation during training increases the L1-norm error on the diverse-pose subset by 9.6%, demonstrating that pose variation is essential for generalization to novel poses. These findings confirm that each component is crucial to our model's overall accuracy.
this section cite: ['b58', 'b76']

Section: Conclusion
In this work, we introduce Puppeteer, a unified rigging-and-animation pipeline built on a dataset with 59.4k high-quality rigged models. Puppeteer first generates skeletons with an autoregressive transformer that uses joint-based tokenization and hierarchical ordering with randomization to capture skeletal structures. An attention-based network with topology-aware features then predicts skinning weights, followed by an efficient optimization module that produces stable, high-quality animations at low computational cost. Across multiple benchmarks, Puppeteer outperforms state-of-the-art methods in skeleton fidelity, skinning accuracy, and animation smoothness.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2025)
Ref_id:b1 Title: Tripo 3d, 2023 Year: ()
Ref_id:b2 Title: Skeleton extraction by mesh contraction Year: (2008)
Ref_id:b3 Title: Tc4d: Trajectory-conditioned text-to-4d generation Year: (2024)
Ref_id:b4 Title: Text-to-4d generation using hybrid score distillation sampling Year: (2024)
Ref_id:b5 Title: Automatic rigging and animation of 3d characters Year: (2007)
Ref_id:b6 Title: Point cloud skeletons via laplacian based contraction Year: (2010)
Ref_id:b7 Title: Ct4d: Consistent text-to-4d generation with animatable meshes Year: (2024)
Ref_id:b8 Title: V2m4: 4d mesh animation reconstruction from a single monocular video Year: (2025)
Ref_id:b9 Title: Video depth anything: Consistent depth estimation for super-long videos Year: (2025)
Ref_id:b10 Title: Meshanything: Artist-created mesh generation with autoregressive transformers Year: (2024)
Ref_id:b11 Title: Meshanything v2: Artist-created mesh generation with adjacent mesh tokenization Year: (2024)
Ref_id:b12 Title: Humanrig: Learning automatic rigging for humanoid character in a large scale dataset Year: (2024)
Ref_id:b13 Title: Automatic conversion of mesh animations into skeleton-based animations Year: (2008)
Ref_id:b14 Title: Objaverse: A universe of annotated 3d objects Year: (2023)
Ref_id:b15 Title: Objaverse-xl: A universe of 10m+ 3d objects Year: (2024)
Ref_id:b16 Title: Anymate: A dataset and baselines for learning 3d object rigging Year: (2025)
Ref_id:b17 Title: Geodesic voxel binding for production character meshes Year: (2013)
Ref_id:b18 Title: Robust biharmonic skinning using geometric fields Year: (2024)
Ref_id:b19 Title: Sync4d: Video guided controllable dynamics for physics-based 4d generation Year: (2024)
Ref_id:b20 Title: Gaussianflow: Splatting gaussian dynamics for 4d content creation Year: (2024)
Ref_id:b21 Title: Anytop: Character animation diffusion with any topology Year: (2025)
Ref_id:b22 Title: Generating diverse and natural 3d human motions from text Year: (2022)
Ref_id:b23 Title: Make-it-animatable: An efficient framework for authoring animation-ready 3d characters Year: ()
Ref_id:b24 Title: L1-medial skeleton of point cloud Year: (2013)
Ref_id:b25 Title:  Year: ()
Ref_id:b26 Title: Autodesk maya Year: (2024)
Ref_id:b27 Title: Bounded biharmonic weights for real-time deformation Year: (2011)
Ref_id:b28 Title: libigl: A simple C++ geometry processing library Year: (2018)
Ref_id:b29 Title: Skinning mesh animations Year: (2005)
Ref_id:b30 Title: Motiongpt: Human motion as a foreign language Year: (2023)
Ref_id:b31 Title: Consistent4d: Consistent 360 {\deg} dynamic object generation from monocular video Year: (2023)
Ref_id:b32 Title: Animate3d: Animating any 3d model with multi-view video diffusion Year: (2024)
Ref_id:b33 Title:  Year: (2025)
Ref_id:b34 Title: Cotracker3: Simpler and better point tracking by pseudo-labelling real videos Year: (2024)
Ref_id:b35 Title: Pose space deformation: a unified approach to shape interpolation and skeleton-driven deformation Year: (2023)
Ref_id:b36 Title: Learning skeletal articulations with neural blend shapes Year: (2021)
Ref_id:b37 Title: Articulated kinematics distillation from video diffusion models Year: (2025)
Ref_id:b38 Title: Dreammesh4d: Video-to-4d generation with sparse-controlled gaussian-mesh hybrid representation Year: (2024)
Ref_id:b39 Title: Learning the 3d fauna of the web Year: (2024)
Ref_id:b40 Title: Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models Year: (2024)
Ref_id:b41 Title: Point2skeleton: Learning skeletal representations from point clouds Year: (2021)
Ref_id:b42 Title: Fast physics-driven 4d content generation from a single image Year: (2024)
Ref_id:b43 Title: Align your gaussians: Text-to-4d with dynamic 3d gaussians and composed diffusion models Year: (2024)
Ref_id:b44 Title: Riganything: Template-free autoregressive rigging for diverse 3d assets Year: (2025)
Ref_id:b45 Title: Automatic skin binding for production characters with deep graph networks Year: (2019)
Ref_id:b46 Title: Partfield: Learning 3d feature fields for part segmentation and beyond Year: (2025)
Ref_id:b47 Title: SMPL: A skinned multiperson linear model Year: (2015-10)
Ref_id:b48 Title: Marching cubes: A high resolution 3d surface construction algorithm Year: (1998)
Ref_id:b49 Title: Pla4d: Pixel-level alignments for text-to-4d gaussian splatting Year: (2024)
Ref_id:b50 Title: Advances in 4d generation: A survey Year: (2025)
Ref_id:b51 Title: Animating the uncaptured: Humanoid mesh animation with video diffusion models Year: (2025)
Ref_id:b52 Title: DPFlow: Adaptive optical flow estimation with a dual-pyramid framework Year: ()
Ref_id:b53 Title: Skinningnet: Two-stream graph convolutional neural network for skinning prediction of synthetic characters Year: (2022)
Ref_id:b54 Title: Heterskinnet: A heterogeneous network for skin weights prediction Year: (2021-04)
Ref_id:b55 Title: Computer animation: algorithms and techniques Year: (2012)
Ref_id:b56 Title:  Year: (2020)
Ref_id:b57 Title: Dreamgaussian4d: Generative 4d gaussian splatting Year: (2023)
Ref_id:b58 Title: L4gm: Large 4d gaussian reconstruction model. Advances in Neural Information Processing Systems Year: (2024)
Ref_id:b59 Title: Hmr-adapter: A lightweight adapter with dual-path cross augmentation for expressive human mesh recovery Year: (2024)
Ref_id:b60 Title: Adhmr: Aligning diffusion-based human mesh recovery via direct preference optimization Year: (2025)
Ref_id:b61 Title: Meshgpt: Generating triangle meshes with decoder-only transformers Year: (2024)
Ref_id:b62 Title: 3d pose transfer with correspondence learning and mesh refinement Year: (2021)
Ref_id:b63 Title: Unsupervised 3d pose transfer with cross consistency and dual reconstruction Year: (2023)
Ref_id:b64 Title: Moda: Modeling deformable 3d objects from casual videos Year: (2024)
Ref_id:b65 Title: Reacto: Reconstructing articulated objects from a single video Year: (2024)
Ref_id:b66 Title: Magicarticulate: Make your 3d models articulation-ready Year: (2025)
Ref_id:b67 Title: Ponymation: Learning articulated 3d animal motions from unlabeled online videos Year: (2024)
Ref_id:b68 Title: Drive: Diffusion-based rigging empowers generation of versatile and expressive characters Year: (2024)
Ref_id:b69 Title: Eg4d: Explicit generation of 4d object without score distillation Year: (2024)
Ref_id:b70 Title: Mean curvature skeletons Year: (2012)
Ref_id:b71 Title: Edgerunner: Auto-regressive auto-encoder for artistic mesh generation Year: (2024)
Ref_id:b72 Title: Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation Year: (2025)
Ref_id:b73 Title: Motionclip: Exposing human motion generation to clip space Year: (2022)
Ref_id:b74 Title: Human motion diffusion model Year: (2022)
Ref_id:b75 Title: The Models-Resource. The models-resource Year: (2019)
Ref_id:b76 Title: Motiondreamer: Exploring semantic video diffusion features for zero-shot 3d mesh animation Year: (2025)
Ref_id:b77 Title: Attention is all you need Year: (2017)
Ref_id:b78 Title: Dual octree graph networks for learning adaptive volumetric shape representations Year: (2022)
Ref_id:b79 Title: Animatabledreamer: Text-guided non-rigid 3d model generation and reconstruction with canonical score distillation Year: (2024)
Ref_id:b80 Title: Learning articulated 3d animals in the wild Year: (2023)
Ref_id:b81 Title: Sc4d: Sparse-controlled video-to-4d generation and motion transfer Year: (2024)
Ref_id:b82 Title: Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency Year: (2024)
Ref_id:b83 Title: Phys4dgen: Physics-compliant 4d generation with multi-material composition perception Year: ()
Ref_id:b84 Title: Predicting animation skeletons for 3d articulated models via volumetric nets Year: (2019)
Ref_id:b85 Title: Neural rigging for articulated characters Year: (2020)
Ref_id:b86 Title: Morig: Motion-aware rigging of character meshes from point clouds Year: (2022)
Ref_id:b87 Title: Building animatable 3d neural models from many casual videos Year: (2022)
Ref_id:b88 Title: Sv4d 2.0: Enhancing spatio-temporal consistency in multi-view video diffusion for high-quality 4d generation Year: (2025)
Ref_id:b89 Title: Riggs: Rigging of 3d gaussians for modeling articulated objects in videos Year: (2025)
Ref_id:b90 Title: Grounded 4d content generation with spatial-temporal consistency Year: (2023)
Ref_id:b91 Title: Randomized autoregressive visual generation Year: (2024)
Ref_id:b92 Title: dynamic: Text-to-4d generation with hybrid priors Year: (2024)
Ref_id:b93 Title: Anymole: Any character motion in-betweening leveraging video diffusion models Year: (2025)
Ref_id:b94 Title: Trans4d: Realistic geometry-aware transition for compositional text-to-4d synthesis Year: (2024)
Ref_id:b95 Title: Stag4d: Spatialtemporal anchored generative 4d gaussians Year: (2024)
Ref_id:b96 Title: Functional diffusion Year: (2024)
Ref_id:b97 Title: Magicpose4d: Crafting articulated models with appearance and motion control Year: (2024)
Ref_id:b98 Title: 4diffusion: Multi-view video diffusion model for 4d generation Year: (2024)
Ref_id:b99 Title: One model to rig them all: Diverse skeleton rigging with unirig Year: (2025)
Ref_id:b100 Title: Motiondiffuse: Text-driven human motion generation with diffusion model Year: (2024)
Ref_id:b101 Title: Large motion model for unified multi-modal motion generation Year: (2024)
Ref_id:b102 Title: Opt: Open pre-trained transformer language models Year: (2022)
Ref_id:b103 Title: Animating one image to 4d dynamic scene Year: (2023)
Ref_id:b104 Title: Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation Year: (2024)
Ref_id:b105 Title: Ar4d: Autoregressive 4d generation from monocular videos Year: (2025)
Ref_id:b106 Title: 3D menagerie: Modeling the 3D shape and pose of animals Year: (2017-07)
Ref_id:b107 Title: Lions and tigers and bears: Capturing non-rigid, 3D, articulated shape from images Year: (2018)
