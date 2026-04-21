Title: GENERATIVE HUMAN GEOMETRY DISTRIBUTION
Abstract: Realistic human geometry generation is an important yet challenging task, requiring both the preservation of fine clothing details and the accurate modeling of clothing-body interactions. To tackle this challenge, we build upon Geometry distributions-a recently proposed representation that can model a single human geometry with high fidelity using a flow matching model. However, extending a single-geometry distribution to a dataset is non-trivial and inefficient for largescale learning. To address this, we propose a new geometry distribution model by two key techniques: (1) encoding distributions as 2D feature maps rather than network parameters, and (2) using SMPL models as the domain instead of Gaussian and refining the associated flow velocity field. We then design a generative framework adopting a two-staged training paradigm analogous to state-of-the-art image and 3D generative models. In the first stage, we compress geometry distributions into a latent space using a diffusion flow model; the second stage trains another flow model on this latent space. We validate our approach on two key tasks: poseconditioned random avatar generation and avatar-consistent novel pose synthesis. Experimental results demonstrate that our method outperforms existing state-ofthe-art methods, achieving a 57% improvement in geometry quality.

Section: INTRODUCTION
3D human geometry generation is a critical yet challenging task. The human body exhibits highfrequency clothing details, which are inherently difficult to synthesize. Therefore, the core of this task is the design of a representation that can capture both the underlying structure and fine details, posing two key challenges: 1) encoding high-frequency details into a low-dimensional manifold without losing fidelity; 2) modeling the relationship between clothing wrinkles and poses to preserve realistic details.
𝐗𝐗 ~𝒩𝒩
this section cite: []

Section: E(𝐗𝐗)
Flow model Existing methods represent the human body in different ways. NeRFs (Men et al., 2024;Zhang et al., 2022;Xiong et al., 2023;Jiang et al., 2023;Xu et al., 2023;Zheng et al., 2024) focus on the rendering results but neglect underlying geometry and are constrained by the rendering speed and resolution. Implicit functions (Xu et al., 2022), such as signed distance functions, struggle to synthesize thin structures and tend to oversmooth results. Point clouds (Zhang et al., 2024c;d) and volume-based representations trade off between memory efficiency and quality. Recently, geometry distributions (Zhang et al., 2025) model single 3D shapes as distributions of points on their surfaces. Samples X from a Gaussian distribution N are transformed via a flow-based diffusion network E(X) into the target geometry, capturing fine-grained surface details (see Fig. 1). This formulation enables infinite point sampling, allowing for a high-fidelity representation of individual shapes. However, extending a singlegeometry distribution to an entire dataset is non-trivial because of two reasons: 1) single-geometry formulations store the geometry in the parameters of a flow network, which leads to prohibitive memory consumption and limits their scalability for generative tasks, and 2) while learning the flow velocity fields from Gaussian distributions to a single shape is feasible, extending this to multiple shapes across a dataset becomes computationally expensive and inefficient. To this end, we propose a novel human geometry distributions model by two key techniques:
• We encode each human geometry distribution into a feature map rather than the network weights of a flow network directly, providing a generalized way to represent geometry distributions.
• We adopt the SMPL (Pavlakos et al., 2019) model distribution instead of a Gaussian and refine the associated flow velocity field for efficient learning. Specifically, we construct spatially efficient mappings from the velocity field and reformulate them into a regularized dense space.
Based on these designs, we propose the first 3D generative method for geometry distributions, modeling the distribution of individual human geometry distributions. Our framework is formulated in two stages: first, we use a flow model to compress each human geometry distribution into a compact feature map, from which a high-fidelity human geometry can be sampled through a denoising process (Fig. 2 (a)), and then, we train another flow model to generate human geometry distributions (i.e., the feature maps). While the currently top-performing methods (Xu et al., 2022;Zhang et al., 2024d;2023b) obtain visually plausible geometric details by enhanced rendering techniques, our method synthesizes high-fidelity geometry directly, leading to large improvements. We design two generative tasks to evaluate the effectiveness of our method (Fig. 2 (b) and ( c)). The first task generates random 3D human geometries conditioned on a given pose, while the second enables novel pose generation of a given avatar. Our quantitative results show that our method improves the geometry quality by 57% (42.9 to 16.2) and the visual appearance by 7% (17.4 to 16.2) compared to the state-of-the-art.
2 RELATED WORK
this section cite: ['b23', 'b54', 'b39', 'b13', 'b58', 'b59', 'b42', 'b52', 'b25', 'b42', 'b57']

Section: GENERAL 3D GENERATION
3D generation methods employ various representations, such as meshes, voxels, point clouds, and implicit functions, each offering distinct advantages but also facing inherent limitations (Zhang et al., 2025). Here, we focus on compact representations that scale efficiently to large datasets.
Given one or more images, direct inference of pixel-aligned positions (Yu et al., 2021a;Tang et al., 2024;Xu et al., 2024b;a) avoids intermediate representations, which is unsuitable for modeling intrinsic human shapes. A popular alternative is to synthesize tri-planes as intermediate representations (Wang et al., 2023;Li et al., 2023;Hong et al., 2023), from which implicit functions or points (Zou et al., 2024;Han et al., 2024) can be sampled. However, tri-planes are limited by their resolution, restricting their ability to capture fine-grained details. GaussianCube (Zhang et al., 2024a) transfers Gaussians into a regular voxel grid via optimal transport, limiting its flexibility for
Table 1: Comparison of 3D human representation methods.
this section cite: ['b52', 'b49', 'b33', 'b44', 'b34', 'b19', 'b11', 'b61', 'b8', 'b53']

Section: Method Loose Clothing Scalability Fine Details
Tri-plane + NeRF (Noguchi et al., 2022;Zhang et al., 2022) ✓ ✓ ✗ Template-based (Hong et al., 2022;Hu et al., 2024) ✗ ✓ ✗ Primitive Volumes (Chen et al., 2023) ✓ ✓ ✗ Mesh-based (Sanyal et al., 2024;Xu et al., 2024c;Feng et al., 2023) ✗ ✓ ✗ Point-based (Abdal et al., 2024;Zhang et al., 2024d) ✓ ✓ ✗ Implicit Function (Xu et al., 2022;Zhang et al., 2023b)
this section cite: ['b24', 'b54', 'b10', 'b12', 'b4', 'b29', 'b45', 'b7', 'b0', 'b57', 'b42']

Section: ✓ ✓ ✗
Geometry Distributions (Ours) ✓ ✓ ✓ intricate articulation. Gamba (Shen et al., 2024) and VectorSet (Zhang et al., 2023a) employ crossattention mechanisms to inject 3D embeddings or image tokens, albeit constrained by memory consumption. Trellis (Xiang et al., 2024) leverages sparse voxels but lacks mechanisms to incorporate human-specific priors. In contrast, 2D maps (Rai et al., 2025;Elizarov et al., 2024;Yan et al., 2024), offer a memory-efficient alternative that aligns naturally with human priors.
this section cite: ['b32', 'b38', 'b26', 'b6', 'b47']

Section: 3D HUMAN RECONSTRUCTION AND GENERATION
Features extracted from images (Saito et al., 2019;2020;Alldieck et al., 2022) provide rich viewdependent information, enabling high-fidelity human reconstruction. Some works incorporate additional cues such as voxel grids (He et al., 2020), human templates (Zheng et al., 2021), or multiview images (Shao et al., 2022) to mitigate the depth ambiguity of single-view image. For example, ICON (Xiu et al., 2022) and ECON (Xiu et al., 2023) predict the frontal and back normals from the SMPL template and reconstruct the human using these maps. Building on these reconstruction techniques, generative networks-such as GANs (Xiong et al., 2023;Jiang et al., 2023) and diffusion models (Sengupta et al., 2024;Shao et al., 2022;Zhang et al., 2024b)-have been employed to synthesize pixel-aligned features or generate multi-view images (Kolotouros et al., 2024) for human generation. However, these methods are inherently limited by their reliance on view-dependent features, sensitivity to multi-view inconsistencies, and often require post-processing to fuse information from different views (Zhang et al., 2024b), failing to capture the intrinsic 3D shape.
To learn intrinsic 3D representations, a popular approach (Noguchi et al., 2022;Zhang et al., 2022;Yang et al., 2024;Wu et al., 2023;2024) is to train a GAN model that synthesizes tri-planes and uses NeRF for rendering. However, these methods are constrained by rendering speed and resolution, often requiring super-resolution modules (Bergman et al., 2022;Dong et al., 2023), multi-part structures (Xu et al., 2023), or refinement stages (Men et al., 2024;Zheng et al., 2024) to achieve high-fidelity results. In addition, Gaussian3Diff (Lan et al., 2024) employs a diffusion model to learn the tri-plane representation for 3D heads, providing an alternative to GAN-based approaches. Rather than employing tri-planes, alternative methods incorporate human templates, alleviating sparse spatial occupancy, but struggle with loose clothing (Hong et al., 2022) or still rely on super-resolution modules (Hu et al., 2024). Explicit representations, such as primitive volumes (Chen et al., 2023), provide efficient rendering but face challenges in extracting detailed geometry. Mesh-based methods, using displacement (Sanyal et al., 2024) or layered surface volumes (Xu et al., 2024c), are limited by fidelity and struggle with loose clothing. Point-based representations (Abdal et al., 2024;Zhang et al., 2024d;c) offer flexibility for modeling thin structures but are constrained by point density. Furthermore, it can be difficult to produce accurate 3D geometry under indirect supervision from rendering. In contrast, gDNA (Xu et al., 2022) proposes learning implicit functions directly from 3D data, but synthesizing high-fidelity geometry remains challenging due to inherent representation limitations (Zhang et al., 2025). Our approach directly learns from 3D data while enabling "infinite" sampling, leading to high-fidelity geometry synthesis. See Tab. 1 for comparison.
this section cite: ['b27', 'b1', 'b9', 'b60', 'b31', 'b40', 'b41', 'b39', 'b13', 'b30', 'b31', 'b17', 'b24', 'b54', 'b48', 'b36', 'b2', 'b5', 'b58', 'b23', 'b59', 'b18', 'b10', 'b12', 'b4', 'b29', 'b45', 'b0', 'b57', 'b42', 'b52']

Section: PRELIMINARIES
Flow Matching Flow matching (Lipman et al., 2023) is a variant of diffusion models. It constructs a flow that transforms samples from a source distribution p to a target distribution q. Specifically, the training objective loss is defined as follow:
arg min θ E x0∼p,x1∼q,t∈[0,1] ∥u θ (x t , t) -(x 1 -x 0 )∥(1)
where x t = (1 -t)x 0 + x 1 . After training, an ordinary differential equation is solved to transition from x 0 to x 1 , allowing us to sample x 1 ∼ q by first sampling x 0 ∼ p.
Based on the flow matching formulation, the core of our modeling is to identify a suitable source distribution and a target distribution that aligns well with our problem. For brevity, we omit t ∈ [0, 1] in subsequent equations.
Geometry Distributions Geometry distributions model a surface M ⊂ R 3 as a probability distribution Φ M , where any sample x 1 ∼ Φ M corresponds to a point on the surface. To achieve this, Zhang et al. (2025) adapt a diffusion model to learn the transformation from a Gaussian distribution N (0, 1) to the target distribution of surface points. A point x 1 ∈ M on the target surface can then be obtained by solving an ODE with an initial point x 0 ∼ N (0, 1).
this section cite: ['b20', 'b52']

Section: METHOD
4.1 OVERVIEW
In this section, we first introduce our human geometry distribution formulation by identifying suitable source and target distributions (Sec. 4.2). Next, we describe encoding this distribution into a feature map using an auto-decoder architecture, as depicted in Fig.
this section cite: []

Section: HUMAN GEOMETRY DISTRIBUTION
Our insight is to replace the Gaussian distribution N (0, 1) in the prior geometry distributions (Zhang et al., 2025) with the SMPL template shape distribution Φ S , aligning the source distribution closer to the target geometry distribution Φ T . A naive approach can be formulated as:
arg min θ E x0∼Φ S ,x1∼Φ T ∥u θ (x t , t) -(x 1 -x 0 )∥ , (2
)
where x t = (1 -t)x 0 + x 1 . Building upon this, we propose two strategies to reduce the modeling complexity and improve training efficiency, including explicitly constructing training pairs and normalizing samples from distributions.
this section cite: ['b52']

Section: Training Pair Construction.
Since the learned probability flow by Eq. 2 approximates a conditional optimal transport (Lipman et al., 2024), one strategy for improving training efficiency is avoiding learning extraneous paths between distant points. To achieve this, we prioritize short flow by constructing training pairs set {(x ′ 0 , x 1 )} T where x ′ 0 ∼ Φ S is geometrically close to x 1 ∼ Φ T . Specifically, we first sample a set of points {x 0 } S on the SMPL template and a set of points {x 1 } T on the target geometry. Then, for each x 1 ∈ {x 1 } T , the corresponding x ′ 0 is obtained by:
x ′ 0 = arg min x0∈{x0} S ||x 1 -x 0 || 2 .
Notably, because multiple points x 1 may share the same nearest SMPL points x ′ 0 , especially in loose or wrinkled regions, directly learning a path from x ′ 0 to x 1 leads to under-sampled geometry (similar to the artifacts shown in the second row of Fig. 6). Therefore, we add a perturbation drawn from N (0, σ) to x ′ 0 , introducing randomness to enhance sample diversity. In this formulation, the source distribution becomes N (x ′ 0 , σ) and the target distribution is the human geometry. Sampling from the constructed training pairs, the flow matching optimization objective is denoted by:
arg min θ E x0∼N (x ′ 0 ,σ),(x ′ 0 ,x1)∈{(x ′ 0 ,x1)} T ∥•∥ (3
)
SMPL templates
this section cite: ['b21']

Section: E(𝐗𝐗)
Human geometries Distribution Normalization. When training the transformation from the SMPL template to the human geometry of a dataset (Eq. 3), the network receives spatially imbalanced supervision due to unevenly sampling points, which arises from two factors. First, points lie only on the surfaces, making them extremely sparse and unevenly distributed relative to the full 3D space. Second, variations in human pose and body shape further exacerbate this spatial imbalance, as some regions are more dynamic and thus sampled less consistently across SMPL vertex map Decompression (b) Human Geometry Generation Bilinear sample (a) Conditional Distribution Encoding t 𝐱𝐱 0 ′ Denoiser 𝐱𝐱 𝑡𝑡 (ℝ 3 ) Δ𝐱𝐱 -𝐧𝐧(ℝ 3 ) Feature map (ℝ 𝐶𝐶×𝐻𝐻×𝑊𝑊 ) (ℝ 𝐾𝐾×16𝐻𝐻×16𝑊𝑊 ) Images Text Representations … Conditioning 𝐱𝐱 𝑡𝑡 (ℝ 𝐶𝐶×𝐻𝐻×𝑊𝑊 ) Feature map -𝐧𝐧 (ℝ 𝐶𝐶×𝐻𝐻×𝑊𝑊 ) SMPL vertex map (ℝ 3 ) (ℝ 1 ) (ℝ 𝐾𝐾 ) geometries. Fig. 3 visualizes this effect by aggregating sampled points from multiple training iterations, showing that some spatial regions have high point density while others remain sparsely sampled. This spatial imbalance leads to inefficient training, as the model pays more attention to high-density regions while neglecting low-density regions. To address this, we normalize both the source and target distribution by subtracting x ′ 0 . The source distribution becomes a zero-centered Gaussian distribution N (0, 1), with σ set to 1, and the target distribution is modeled as a regularized dense displacement field ∆x = x 1 -x ′ 0 . Note that this subtraction removes the positional information of x ′ 0 . To retain this information without suffering from imbalanced sampling, we reintroduce x ′ 0 as a conditional signal that scales the network hidden feature, indirectly influencing the feature representation. By building flow between regularized dense spaces, we reduce the modeling complexity and achieve improved training efficiency, as demonstrated in Sec 5.2. The optimization objective is denoted by:
arg min θ E n,(x ′ 0 ,x1) ∥u θ (x t , t|x ′ 0 ) -(∆x -n)∥ ,(4)
where n ∼ N (0, 1), (x ′ 0 , x 1 ) ∈ {(x ′ 0 , x 1 )} T and x t = (1 -t)n + t∆x.
this section cite: []

Section: CONDITIONAL DISTRIBUTION ENCODING
The dataset for our task consists of a set of pairs D = {(S, T )}, each associated with a latent representation z T |S , which is used to recover T (target geometry) by conditioning on S (SMPL). This design aligns with our generative task, i.e. , S-conditioned T generation. The training loss can be written as,
arg min θ,{z T |S } E (S,T )∈D E n,(x ′ 0 ,x1) u θ (x t , t|x ′ 0 , z T |S ) -(∆x -n) ,(5)
To enhance the expressiveness, we model z T |S ∈ R C×H×W as a compressed 2D feature map (height H and width W ), as shown in Fig. 4 (a). The representation aligns with recent advancements in UV representations for 3D objects (Yan et al., 2024). Furthermore, we use a decoder network Dec ϕ (•) to decompress z T |S to a higher resolution. By leveraging the UV coordinates associated with x ′ 0 , we can then obtain per-point latents on the surface S by sampling the high-resolution map, denoted as Dec ϕ (z T |S )(x ′ 0 ). Thus, the revised optimization can be written as
arg min θ,ϕ,{z T |S } E (S,T )∈D E n,(x ′ 0 ,x1) u θ x t , t|x ′ 0 , Dec ϕ (z T |S )(x ′ 0 ) -(∆x -n) .(6)
Decoder. As illustrated in Fig. 4 (a), the network Dec ϕ (•) is a UNet-style network that contains several downsampling and upsampling layers. We render SMPL vertex positions into UV maps and concatenate them with hidden features in the convolution layers.
Denoising network. The design of the denoiser u θ is adapted from (Zhang et al., 2025). In this case, we have two additional conditioning signals, SMPL point x ′ 0 and latent Dec ϕ (z T |S )(x ′ 0 ). Additionally, we augment x ′ 0 by concatenating normals and canonical coordinates. Surface normals provide directional cues for clothing inference, while canonical coordinates encode body part semantics (e.g., distinguishing limbs from the torso).
this section cite: ['b47', 'b52']

Section: GENERATIVE TASKS
After finishing learning the latents {z T |S }, we train generative models in the latent space. We investigate two tasks: pose-conditioned random generation and novel pose generation of a given avatar, both built on a U-Net (Karras et al., 2024;2025) architecture, as illustrated in Fig. 4 (b).
Pose-conditioned random generation synthesizes diverse human geometries conditioned on an SMPL template mesh. To encode pose information, we render SMPL vertex positions into UV maps and inject them as residual connections into the U-Net architecture. For novel pose generation, the model additionally takes a frontal normal image as an extra condition to indicate the avatar identity. Specifically, for each animation sequence in the dataset, we randomly select one frame to provide the normal image condition and another frame to provide the SMPL pose condition. We leverage the frozen DINO-ViT model (Caron et al., 2021) to extract image features, which are then fused into the U-Net via cross-attention layers.
The pose-conditioned generation networks are trained on the THuman2 dataset (Yu et al., 2021b), while the novel pose generation is trained on 4DDress (Wang et al., 2023). See the Appendix for implementation details.
this section cite: ['b14', 'b50', 'b34']

Section: EXPERIMENTS AND RESULTS
In this section, we first conduct ablation studies on the training pair construction strategy and network architecture. Next, we compare different formulations of geometry distribution, including the formulation proposed by Zhang et al. (2025) and our proposed alternatives. Finally, we compare our method with state-of-the-art generative methods regarding the two selected generative tasks.
this section cite: ['b52']

Section: ABLATION STUDY

this section cite: []

Section: TRAINING PAIRS CONSTRUCTION
Rather than searching for the nearest points on the dense SMPL mesh, we first sample a relatively sparse set of points from the SMPL mesh and then determine the nearest points x ′ 0 from this sampled set. It distributes the mapping workload across different x ′ 0 , which improves sampling efficiency. To validate this, we train our auto-decoder under both strategies and present the results in Fig. 6.
The first row shows the results using the sparse points set sampling strategy, while the second row corresponds to directly identifying the nearest points on the SMPL mesh. The normal maps in the second row exhibit noticeable holes, indicating insufficient sampling. These artifacts are particularly pronounced for loose clothing (middle case in Fig. 6) but also appear in regions of relatively tight clothing (left and right case in Fig. 6).
this section cite: []

Section: NETWORK ARCHITECTURE
Zhang et al. w/o Pairs 𝒩𝒩(𝐱𝐱 𝟎𝟎 ′, 𝝈𝝈) Ours w/o DistNorm We experiment to validate that the auto-decoder outperforms the existing auto-encoders for our task. The auto-encoder learns features from the input, requiring more computations and being potentially constrained by the input's representation capacity. We adopt the 3D representation network VecSet (Zhang et al., 2023a) as the encoder for the auto-encoder and take 50, 000 input points, balancing representation capability and memory consumption. Since VecSet uses independent embeddings rather than a feature map, we treat each pixel of the feature map R 24×24×8 as an individual embedding, yielding 576 embeddings (FeatureMap). To further ensure a fair comparison, we introduce an additional model where we replace our feature map decompression module (Fig. 4 (b)) with the one proposed by VecSet, maintaining as much of the VecSet architecture as possible (VecSet).
The experiment is conducted on the THuman2 dataset (Yu et al., 2021b). As shown in Tab. 3, we report the average distance between the synthesized points and the surface. While converting independent embeddings to a feature map improves reconstruction accuracy, the results remain inferior to our proposed auto-decoder.
this section cite: ['b50']

Section: COMPARISONS OF GEOMETRY DISTRIBUTION FORMULATIONS
To evaluate the training efficiency of different geometry distributions, we first conduct a singlegeometry fitting task and then extend our analysis to a dataset. We compare our method with several alternatives, including the geometry distribution proposed by Zhang et al. (2025), the naive formulation in Eq. 2 (w/o Pairs), and the formulation sampling from N (x ′ 0 , σ) without distribution normalization (w/o DistNorm), proposed in Eq. 3.
For the single-geometry fitting task, we train each formulation for 10, 000 iterations and visualize the synthesized points at the final step in Fig. 5. Our method produces the smoothest and most accurate reconstructions, while all other methods generate points that deviate from the surface. The Chamfer distances at the final step are presented in Tab. 2 (Single). The distribution proposed by Zhang et al. (2025) performs the worst because it samples from Gaussian noise and requires more iterations to capture the coarse human shape. The "w/o Pairs" formulation also exhibits suboptimal performance as it must learn mappings between distant points, leading to slower convergence. The approach that samples from N (x ′ 0 , σ) (w/o DistNorm) achieves the lowest Chamfer distance in this task. With a fixed pose, each sample can focus on learning specified local features, thereby simplifying the optimization. However, as shown in Fig. 5, its reliance on dispersed Gaussian centers leads to noisy output for some regions. Furthermore, this approach negatively impacts convergence when scaling to datasets with diverse poses, as reflected in its degraded performance in Tab. 2 (Dataset).
For dataset-scale evaluation, we train each model on the THuman2 (Yu et al., 2021b) dataset for 30, 000 iterations. Our method consistently achieves significantly higher fidelity compared to the other approaches within the same number of training iterations, as shown in Fig. 7. Notably, the results do not represent the final outcome, as training is not fully completed. The evolution of Chamfer distance over iterations is provided in the Appendix.
this section cite: ['b52', 'b52', 'b50']

Section: POSE-CONDITIONED RANDOM GENERATION
This experiment measures the geometry quality of pose-conditioned random generation. Consistent with previous work (Zhang et al., 2023b;2024d), we utilize the THuman2 Dataset (Yu et al., 2021b) for training and evaluation. We calculate the Fréchet Inception Distance (FID) metrics between the rendered normal images and ground truth normal maps. Notably, some works (Xu et al., 2022;Zhang et al., 2023b;2024d) enhance their rendering results using information such as normal  field/map or predicted points' rotations. Since our goal is to compare geometry quality, we primarily evaluate these methods based on their raw geometry outputs. To ensure fairness, we also report their performance on their enhanced rendered results. For each subject, we render 50 images from different views and synthesize 25, 000 normal images for the dataset.
We compare our method with representative human generation methods employing different 3D representations, including Gaussian splatting (E3Gen (Zhang et al., 2024d)), implicit function (GetAvatar (Feng et al., 2023), gDNA (Xu et al., 2022)), and Nerf (ENARF (Noguchi et al., 2022), GNARF (Bergman et al., 2022) and Eva3D (Hong et al., 2022)). As shown in Tab. 4, our method significantly outperforms other approaches in terms of raw geometry outputs. In comparison to the state-of-the-art (gDNA), our method improves geometry quality by 57% (42.9 to 16.2). Additionally, when compared to the enhanced rendered results from other methods, our raw geometry shows improvements as well (7% from 17.4 to 16.2). As shown in Fig. 8, E3Gen (Zhang et al., 2024d) synthesizes unnatural shapes with inconsistent normal colors. GetAvatar (Zhang et al., 2023b) generates cloth wrinkles with unnatural directional patterns, while gDNA (Xu et al., 2022)'s normals produce wrinkles that appear random, unrealistic, and not coordinated with the human pose. In contrast, our method generates realistic and pose-consistent clothing details, demonstrating superior geometric fidelity and detail preservation. Previous methods (Zhang et al., 2023b;Xu et al., 2022;Zhang et al., 2024d) synthesize 3D humans in canonical space and deform them via rigging, which limits pose-dependent clothing details (e.g., wrinkles and folds). In contrast, Figure 9: Generating different poses for a given avatar. We showcase challenging cases, including exaggerated poses, skirts, and loose outfits.
this section cite: ['b57', 'b50', 'b42', 'b57', 'b57', 'b7', 'b42', 'b24', 'b2', 'b10', 'b57', 'b42', 'b42', 'b57']

Section: NOVEL POSE GENERATION
our approach directly synthesizes points on the deformed human body given the pose-aware feature map, enabling the generation of pose-dependent clothing details, as shown in Fig. 9. More visual results can be found in the Appendix.
For quantitative comparisons, FID metrics reflecting geometric quality are provided in Tab. 4. Note that since prior methods' geometric details remain static across poses, their FID results correspond closely to those reported in the above table. In terms of physical plausibility of geometric details, due to the absence of standard metrics, we conduct a user study to demonstrate that our method uniquely enables pose-aware deformations that appear physically reasonable. For each method, we generate 2 identities under 8 poses and ask 25 participants to rate geometry quality and physical plausibility (1-5 scale), as shown in Tab. 5.
Our superior physical plausibility mainly stems from our synthesized pose-aware feature maps. Nevertheless, even when using a feature map that is incompatible with the body pose, our method still generates visually reasonable results, which demonstrates the robustness of our model in handling novel poses (see the Appendix for details).
this section cite: []

Section: LIMITATIONS
A limitation of our method lies in non-uniform sampling on the target geometry surface, as the number of target points x 1 associated with each SMPL point x ′ 0 varies. Although sufficient sampling and removing points that are too close mitigate this issue, we encourage future work to develop more advanced strategies for training pair construction. Besides, similar to other generative methods, our approach is constrained by the diversity of the training datasets. Specifically, our model generalizes well to various body types, as the dataset includes a reasonable diversity in this aspect. However, it cannot generate clothing styles that are entirely absent from the training data. Furthermore, the use of UV maps can cause seam artifacts in some randomly generated results due to discontinuity segmentation. A better approach would use UV segmentation aligned with real-world garment patches. Since our focus is on a modeling representation, addressing this issue is left for future work.
this section cite: []

Section: CONCLUSIONS
In conclusion, we introduce the generative human geometry distribution, the first method that integrates geometry distributions into generative modeling. This distribution-over-distribution design is novel, unique, and particularly effective, making our approach the only one capable of producing fine-grained geometry in a generative framework. Specifically, using flow matching, we learn a flow that transforms from the SMPL template to the target geometry, significantly improving training efficiency. Following this, our method encodes geometry into compact feature maps, facilitating various downstream generative tasks. Experimental results demonstrate that our approach achieves state-of-the-art quantitative performance compared to existing generative methods. Furthermore, our method is able to synthesize fine-grained clothing details that conform to human poses and exhibits strong robustness, generating plausible results even when provided with a feature map mismatched to the target pose. These results highlight the effectiveness of our representation and its potential for advancing 3D human modeling and synthesis. 9.3 NOVEL POSE GENERATION GT Ours Figure 14: This figure shows the results of our method after full training, using the same identities and poses as in Fig. 7.
Previous methods (Zhang et al., 2023b;Xu et al., 2022;Zhang et al., 2024d) synthesize 3D humans in canonical space and obtain novel pose appearances via deformation. Their limitations are illustrated in Fig. 12, where we deform avatars to different poses, resulting in identical clothing details across all poses. In the main paper, we have showcased that our method can generate pose-aware deformations by synthesizing a pose-compatible feature map. Fig. 12 demonstrates that, although the wrinkle details may not be entirely natural, our method still produces reasonable results even when using an incompatible feature map.
Furthermore, we present additional novel pose generation results in Fig. 13. Specifically, we showcase challenging cases featuring intricate poses and complex clothing deformations, such as loose garments and skirts, which significantly deviate from the SMPL surface.
this section cite: ['b42', 'b57']

Section: References
Ref_id:b0 Title: Gaussian shell maps for efficient 3d human generation Year: (2024)
Ref_id:b1 Title: Photorealistic monocular 3d reconstruction of humans wearing clothing Year: (2022)
Ref_id:b2 Title: Generative neural articulated radiance fields Year: (2008)
Ref_id:b3 Title: Emerging properties in self-supervised vision transformers Year: ()
Ref_id:b4 Title: Primdiffusion: Volumetric primitives diffusion for 3d human generation Year: (2023)
Ref_id:b5 Title: Ag3d: Learning to generate 3d avatars from 2d image collections Year: (2023)
Ref_id:b6 Title: Geometry image diffusion: Fast and data-efficient text-to-3d with image-based surface representation Year: (2024)
Ref_id:b7 Title: Learning disentangled avatars with hybrid 3d representations Year: (2008)
Ref_id:b8 Title: Flex3d: Feedforward 3d generation with flexible reconstruction model and input view curation Year: (2024)
Ref_id:b9 Title: Geo-pifu: Geometry and pixel aligned implicit functions for single-view human reconstruction Year: (2020)
Ref_id:b10 Title: Eva3d: Compositional 3d human generation from 2d image collections Year: (2008)
Ref_id:b11 Title: Lrm: Large reconstruction model for single image to 3d Year: (2023)
Ref_id:b12 Title: Structldm: Structured latent diffusion for 3d human generation Year: (2024)
Ref_id:b13 Title: Humangen: Generating human radiance fields with explicit priors Year: (2023)
Ref_id:b14 Title: Analyzing and improving the training dynamics of diffusion models Year: (2024)
Ref_id:b15 Title: Guiding a diffusion model with a bad version of itself Year: (2025)
Ref_id:b16 Title: 3d gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b17 Title: Eduard Gabriel Bazavan, and Cristian Sminchisescu. Instant 3d human avatar generation using image diffusion models Year: (2024)
Ref_id:b18 Title: Gaussian3diff: 3d gaussian diffusion for 3d full head synthesis and editing Year: (2024)
Ref_id:b19 Title: Instant-3d: Instant neural radiance field training towards on-device ar/vr 3d reconstruction Year: (2023)
Ref_id:b20 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b21 Title: Flow matching guide and code Year: (2024)
Ref_id:b22 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b23 Title: En3d: An enhanced generative model for sculpting 3d humans from 2d synthetic data Year: (2024)
Ref_id:b24 Title: Unsupervised learning of efficient geometry-aware neural articulated representations Year: (2008)
Ref_id:b25 Title: Expressive body capture: 3d hands, face, and body from a single image Year: (2019)
Ref_id:b26 Title: Uvgs: Reimagining unstructured 3d gaussian splatting using uv mapping Year: (2025)
Ref_id:b27 Title: Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization Year: (2019)
Ref_id:b28 Title: Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization Year: (2020)
Ref_id:b29 Title: Sculpt: Shape-conditioned unpaired learning of pose-dependent clothed and textured human meshes Year: (2024)
Ref_id:b30 Title: Diffhuman: probabilistic photorealistic 3d reconstruction of humans Year: (2024)
Ref_id:b31 Title: Diffustereo: High quality human reconstruction via diffusion-based stereo using sparse cameras Year: (2022)
Ref_id:b32 Title: Marry gaussian splatting with mamba for single view 3d reconstruction Year: (2024)
Ref_id:b33 Title: Lgm: Large multi-view gaussian model for high-resolution 3d content creation Year: (2024)
Ref_id:b34 Title: Pf-lrm: Pose-free large reconstruction model for joint pose and shape prediction Year: (2023)
Ref_id:b35 Title: dress: A 4d dataset of real-world human clothing with semantic annotations Year: ()
Ref_id:b36 Title: 3dportraitgan: Learning onequarter headshot 3d gans from a single-view portrait dataset with diverse body poses Year: (2023)
Ref_id:b37 Title: Portrait3d: Text-guided high-quality 3d portrait generation using pyramid representation and gans prior Year: (2024)
Ref_id:b38 Title: Structured 3d latents for scalable and versatile 3d generation Year: (2024)
Ref_id:b39 Title: Get3dhuman: Lifting stylegan-human into a 3d generative model using pixel-aligned reconstruction priors Year: (2023)
Ref_id:b40 Title: Icon: Implicit clothed humans obtained from normals Year: (2022)
Ref_id:b41 Title: Econ: Explicit clothed humans optimized via normal integration Year: (2023)
Ref_id:b42 Title: Gdna: towards generative detailed neural avatars Year: (2022)
Ref_id:b43 Title: Agg: Amortized generative 3d gaussians for single image to 3d Year: (2024-02)
Ref_id:b44 Title: Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation Year: (2024-02)
Ref_id:b45 Title: Efficient 3d articulated human generation with layered surface volumes Year: (2024-03)
Ref_id:b46 Title: Xagen: 3d expressive human avatars generation Year: (2023)
Ref_id:b47 Title: An object is worth 64x64 pixels: Generating 3d object via image diffusion Year: (2024)
Ref_id:b48 Title: Attrihuman-3d: Editable 3d human avatar generation with attribute decomposition and indexing Year: (2024)
Ref_id:b49 Title: pixelnerf: Neural radiance fields from one or few images Year: (2021-02)
Ref_id:b50 Title: Function4d: Real-time human volumetric capture from very sparse consumer rgbd sensors Year: (2006)
Ref_id:b51 Title: 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models Year: (2023-07)
Ref_id:b52 Title: Geometry distributions Year: (2007)
Ref_id:b53 Title: Structuring gaussian splatting using optimal transport for 3d generative modeling Year: (2024-02)
Ref_id:b54 Title: Avatargen: a 3d generative model for animatable human avatars Year: (2022)
Ref_id:b55 Title: Joint2Human: Highquality 3D human generation via compact spherical embedding of 3D joints Year: (2024-06)
Ref_id:b56 Title: HQavatar: Towards high-quality 3D avatar generation via point-based representation Year: (2024-07)
Ref_id:b57 Title: Xingdong Sheng, and Xiaokang Yang. $Eˆ{3}$gen: Efficient, expressive and editable avatars generation Year: (2024)
Ref_id:b58 Title: GETAvatar: Generative textured meshes for animatable human avatars Year: (2023-10)
Ref_id:b59 Title: Semantichuman-hd: High-resolution semantic disentangled 3d human generation Year: (2024)
Ref_id:b60 Title: Pamir: Parametric model-conditioned implicit representation for image-based human reconstruction Year: (2021)
Ref_id:b61 Title: Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers Year: (2024)
