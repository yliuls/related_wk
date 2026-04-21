Title: Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning
Abstract: Existing pretrained models for 3D mesh generation often suffer from data biases and produce low-quality results, while global reinforcement learning (RL) methods rely on object-level rewards that struggle to capture local structure details. To address these challenges, we present Mesh-RFT, a novel fine-grained reinforcement finetuning framework that employs Masked Direct Preference Optimization (M-DPO) to enable localized refinement via quality-aware face masking. To facilitate efficient quality evaluation, we introduce an objective topology-aware scoring system to evaluate geometric integrity and topological regularity at both object and face levels through two metrics: Boundary Edge Ratio (BER) and Topology Score (TS). By integrating these metrics into a fine-grained RL strategy, Mesh-RFT becomes the first method to optimize mesh quality at the granularity of individual * Equal Contribution. † Corresponding Author. 39th Conference on Neural Information Processing Systems (NeurIPS 2025).faces, resolving localized errors while preserving global coherence. Experiment results show that our M-DPO approach reduces Hausdorff Distance (HD) by 24.6% and improves Topology Score (TS) by 3.8% over pre-trained models, while outperforming global DPO methods with a 17.4% HD reduction and 4.9% TS gain. These results demonstrate Mesh-RFT's ability to improve geometric integrity and topological regularity, achieving new state-of-the-art performance in productionready mesh generation.

Section: Introduction
3D polygonal meshes serve as the foundational representation for digital assets in industries such as gaming, film, and product design. Despite their ubiquity, high-quality, topologically optimized meshes-essential for downstream tasks like editing, rigging, and animation-are still predominantly handcrafted by skilled artists. Recent advances in generative models have enabled automated mesh synthesis, significantly reducing the time and expertise required to produce production-ready 3D assets. This democratization of mesh generation broadens access to 3D content creation, empowering non-experts to produce geometrically precise and artistically viable models for applications ranging from immersive media to industrial design.
Existing 3D generative models often use intermediate representations like voxels [1,2], point clouds [3,4,5], latent space [6,7] or implicit fields [8,9]. While these avoid direct mesh generation complexities, post-processing (e.g., Marching Cubes [10]) often introduces topological issues and smoothing. Native mesh generation [11] is more direct, with recent work using autoregressive models and neural compression (e.g., VQ-VAE [12,13,14]) or geometric serialization tokenizers (e.g., [15,16,17,18,19]) for sequence-based generation. However, long sequences for highresolution meshes can cause structural ambiguities and hallucinations (inconsistent edges, nonmanifold vertices, distortions, holes), deviating from geometric constraints or artistic intent, ultimately leading to results that may not align with human aesthetic preferences or intended design. Though truncated training [20] helps, autoregressive methods still lack stable generation and high fidelity.
Recently, reinforcement learning [21,22] has emerged as a compelling approach for aligning mesh generation more closely with human preferences. For example, DeepMesh [23] leverages Direct Preference Optimization (DPO) [24], a simple yet effective preference alignment technique that has also found utility in various other domains [25,26,27]. Nevertheless, directly applying reinforcement fine-tuning to mesh generation using this method encounters two primary challenges. Firstly, objectively quantifying mesh quality is difficult. DeepMesh relies on manual annotation of preference pairs, which is expensive, time-consuming, introduces subjective bias, and limits the training data to only 5,000 samples, hindering generalization. Secondly, its use of global reward signals fails to capture the local topological variations inherent in 3D meshes. As illustrated in Figure 2, high-quality and low-quality structures often coexist within a single mesh, leading to training noise due to this mismatch in supervision.
Figure 2: High-quality, artist-like structures often coexist with messy, low-quality regions within the same mesh.
To overcome these limitations, we introduce Mesh-RFT, a novel framework that combines Masked Direct Preference Optimization (M-DPO) with fine-grained mesh quality evaluation for both global and localized refinement. Unlike prior work using subjective global rewards as supervision signals [23], we employ a topology-aware scoring system with automated metrics-Boundary Edge Ratio (BER) and Topology Score (TS)-to objectively evaluate mesh quality at both object and face levels, circumventing the laborious manual annotation efforts. Mesh-RFT further employs a localized optimization mechanism utilizing M-DPO and qualityaware masks to specifically refine defective regions, thereby addressing the coarse supervision of global rewards. Extensive experiments across diverse meshes demonstrate Mesh-RFT's superior performance, achieving significant improvements over both the pretrain baseline (24.6% HD reduction, 3.8% TS improvement) and global DPO (17.4% HD reduction, 4.9% TS improvement), establishing a new benchmark for accuracy and fidelity in generative mesh modeling. In summary, our contributions are as follows:
• We introduce the first fine-grained reinforcement fine-tuning framework, that integrates Masked Direct Preference Optimization (M-DPO) with fine-grained mesh quality evaluation.
• We devise an objective topology-aware scoring system for evaluating mesh quality, eliminating dependency on manual annotation and addressing subjectivity and scalability limitations.
• We propose a novel localized alignment mechanism that optimizes deficient regions geometrically and topologically via quality-aware masks, bridging the gap between global and local supervision.
• Experiments demonstrate that our method achieves state-of-the-art performance in high-fidelity 3D mesh generation.
2 Related work
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b22']

Section: 3D Generation via Alternative Representations
Many 3D generative models avoid direct mesh modeling by using intermediate representations like voxels, point clouds, or implicit fields. Early voxel methods [1,2] using grids faced memory issues. Point cloud methods [3,4,28,29] with networks like PointNet [5,30] struggle with consistency and detail. Implicit fields, especially neural fields [8,9,31,32], offer efficient representations. These include score distillation with 2D diffusion models [33,34,35,36,37,38] and 3D Transformer models like LRM [39,40,41,42,43], alongside recent latent diffusion methods [44,45,46,47,48,49,50,51] that have demonstrated good scalability and performance. However, these approaches often rely on post-processing via Marching Cubes [10], which can cause topological issues, smoothing, and artifacts.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b27', 'b28', 'b4', 'b29', 'b7', 'b8', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b9']

Section: Native Mesh Generation
While neural shape representations such as implicit fields have been extensively studied, native mesh generation is an emerging area of research. Early approaches leveraging surface patches [52] or mesh graphs [53] often suffered from quality limitations. Diffusion-based methods [54,55] have seen limited exploration in this domain, potentially due to inherent difficulties in directly processing meshes. PolyGen [11] demonstrated promise by autoregressively generating mesh vertices and faces. MeshGPT [12] encoded meshes into quantized tokens using VQ-VAE [56] for autoregressive generation. Subsequently, MeshXL [15] proposed a one-stage autoregressive model operating on coordinate-level mesh sequences. Various tokenization techniques [16,17,19,57] and efficient training strategies [20,58] have been explored to address the challenges of long sequences in highresolution generation; however, achieving stable and high-fidelity results remains a significant hurdle.
this section cite: ['b51', 'b52', 'b53', 'b54', 'b10', 'b11', 'b55', 'b14', 'b15', 'b16', 'b18', 'b56', 'b19', 'b57']

Section: Reinforcement Learning for Mesh Generation
Reinforcement Learning (RL) [59] has gained traction for 3D generation [60] using human feedback. Reinforcement Learning from Human Feedback (RLHF) aligns models with preferences by training a reward model, then fine-tuning with RL. However, RLHF is costly and unstable for 3D tasks. Direct Preference Optimization (DPO) [24] offers a more efficient, stable alternative by removing the reward model. Despite success in language and image domains [25,26], DPO's application to 3D meshes is limited. Closely related, DeepMesh [23] uses global rewards for alignment but struggles with 3D mesh heterogeneity, over-optimizing some regions and under-optimizing others. Thus, RL methods addressing local mesh structures are crucial for better 3D mesh quality and consistency.
this section cite: ['b58', 'b59', 'b23', 'b24', 'b25', 'b22']

Section: Method
This section details the Mesh-RFT framework. As illustrated in Figure 3, our pipeline consists of three stages: First, supervised pretraining is performed by feeding point clouds and ground truth mesh sequences into the model. Second, the pretrained model generates candidates, and a topologyaware score system builds a preference dataset. Third, topology-aware Masked Direct Preference Optimization is applied to post-train the model using this preference dataset to refine its performance.
this section cite: []

Section: Mesh Generation Pre-training
Firstly, we discuss mesh tokenization. Prior works [16,17,19] compress mesh sequences to manage sequence growth with increasing faces, but such techniques embed excessive geometric information per token, causing cascading face errors when a single token is incorrect (e.g., BPT [19] often introduces patch-level holes). To avoid these issues, we adopt the uncompressed mesh sequence method introduced from MeshXL [15]. Specifically, for a given mesh M, we first quantize the vertex coordinates of each face, and then flatten them in XY Z order to construct a complete token sequence.
this section cite: ['b15', 'b16', 'b18', 'b18', 'b14']

Section: Model Architecture.
To better capture the structure of the mesh, rather than framing mesh generation as a generic sequence task, we utilize Hourglass Transformer architecture [20,61]. Our model processes inputs hierarchically and incorporates two shorten and two upsample operations. The shorten operations reduce the token sequence length using techniques such as linear or attention-based pooling, while the upsample operations expand the sequence back to its original length through linear or attention-based methods. This design enables the model to efficiently capture both high-level patterns and fine-grained details. In point-cloud conditioned mesh generation, achieving fine-grained and complex structures requires not only a powerful decoder but also high-quality point cloud features.
To this end, we adopt the point cloud encoder pretrained in Hunyuan3D 2.0 [48] to do this. These features are injected into our autoregressive decoder as keys and values via cross-attention [62].
this section cite: ['b19', 'b60', 'b47', 'b61']

Section: Truncated Training and Sliding-Window Inference.
To reduce memory and computational costs, we employ truncated training with fixed-length segments. This approach involves extracting smaller, fixed-length segments from the mesh sequence for training, rather than using the entire sequence. When a segment does not contain the start-of-sequence (SOS) token, we pad a small prefix portion to avoid misleading the model. During inference, we use a sliding window approach to enhance both speed and generation quality. The sliding process begins once 40% of the training window size is covered, and only the most recent 30% of tokens are retained. This method reduces computational load by focusing on the most relevant tokens, as distant tokens typically have less influence on each
good bad bad good good bad BER=0 TS =74.7 HD =0.0996 BER=0.0016 TS =80.4 HD =0.1375 BER=0.0025 TS =65.3 HD =0.1016 BER=0.0001 TS =79.3 HD =0.1238 BER=0.0020 TS =75.7 HD =0.1453 BER=0.0044 TS =79.2 HD =0.1457
Figure 4: Examples of collected preference pairs. Meshes are annotated as preferred using our scoring system. For certain pairs, the selected "good" meshes may exhibit inferior local performance in specific regions compared to the rejected "bad" meshes.
other. Additionally, it helps mitigate high perplexity at the tail of each window, leading to more accurate and efficient generation.
this section cite: []

Section: Preference Dataset Construction
We establish a systematic pipeline for constructing the preference dataset, which is used for RLHF fine-tuning in the second stage. This pipeline consists of three key components: candidate generation, multi-metric evaluation, and preference ranking. The process is described as follows.
Candidate Generation. For each input point cloud P, we generate eight candidate meshes M 1 P , M 2 P , • • • , M 8 P using the pre-trained model G pre θ .
this section cite: []

Section: Multi-Metric Evaluation.
We evaluate each candidate mesh using a comprehensive set of criteria to assess both geometric consistency and topological quality. In addition to measuring the geometric alignment with the input data, we introduce two topology-oriented metrics that specifically aim to capture the structural integrity and coherence of the generated meshes. These three metrics are: Boundary Edge Ratio (BER) and Topology Score (TS) for evaluating topology, and Hausdorff Distance (HD) for evaluating geometric consistency.
• Boundary Edge Ratio (BER): This metric, defined as BER(M) = E ∂M E M , quantifies the integrity of the mesh by calculating the proportion of its boundary edges (E ∂M ) to the total number of edges (E M ). Boundary edges are those connected to only one face, and a high BER value (typically above 0.002 in our dataset, which consists mostly of closed meshes) suggests potential issues like surface discontinuities, holes, or mesh damage. Ideally, a closed, manifold mesh should have a BER of 0.
• Topology Score (TS): The Topology Score, T S(M) = reconstructed mesh M i P and the input point cloud P by measuring the maximum distance between their respective point samples. A lower HD value indicates a better geometric reconstruction.
Preference Ranking. To construct the preference dataset, we generate pairwise comparisons through exhaustive combinations of the eight candidate meshes for each input point cloud P, resulting in a total of 8  2 = 28 pairs. For each pair (M i P , M j P ), we define a preference relation M i P ≻ M j P if and only if M i P outperforms M j P across all three evaluation metrics:
BER(M i P ) < BER(M j P ) ∧ M i P ≻ M j P ⇐⇒ T S(M i P ) > T S(M j P ) ∧ HD(M i P ) < HD(M j P )(1)
We refer to M i P as the positive sample (denoted M + P ) and M j P as the negative sample (denoted M - P ) for the pair. Using this rule, we construct a set of preference triplets of the form (P, M + P , M - P ), which constitutes our preference dataset for reinforcement learning with human feedback.
this section cite: []

Section: Mesh Generation Post-training
While our pre-trained model produces topologically valid meshes, two persistent challenges remain: (1) localized geometric imperfections in high-curvature regions, and (2) inconsistent face density distribution causing aesthetic artifacts. Although DeepMesh [23] adopts RLHF for mesh refinement, its reward function is primarily based on global mesh structure, making it insufficient for fine-grained control over local mesh quality. To address these limitations, we propose Masked Direct Preference Optimization (M-DPO)-a spatially aware extension of DPO) [24]. M-DPO introduces quality localization masks to guide learning toward problematic regions, enabling more targeted and effective mesh refinement.
this section cite: ['b22', 'b23']

Section: Quality-Aware Local Masking.
The goal of local masking is to differentiate high-quality regions of a mesh from those of lower quality. Given a triangular mesh M, we assess each triangle face individually. A face is labeled as good if it satisfies the following two conditions: (1) it can be successfully merged into a quadrilateral, and (2) the resulting quad has a quality score above a predefined threshold. The quad quality is evaluated using a weighted combination of three metrics introduced in Section 3.2: Angle Quality, Aspect Ratio, and Adjacent Consistency. For each triangle face labeled as good, we assign a value of 1 to all corresponding token positions in the mesh sequence (typically 9 tokens per face). Conversely, faces that do not meet the criteria are considered bad, and their associated tokens are assigned a value of 0. We define the local masking function as ϕ, such that ϕ(M) ∈ {0, 1}
|M| , where |M| denotes the length of the token sequence representing mesh M.
this section cite: []

Section: Masked Direct Preference Optimization.
Standard DPO tends to optimize global reward signals uniformly across the entire mesh sequence, which can lead to over-smoothed results and the loss of fine-grained geometric details. In contrast, our Masked Direct Preference Optimization (M-DPO) addresses this limitation by applying element-wise importance weighting guided by local quality masks, allowing the model to focus refinement specifically on low-quality regions. As illustrated in Figure 3, we designate the pretrained model from the first stage as the reference model, denoted as G ref := G pre θ , whose parameters are frozen during training. A trainable policy model G ψ is then initialized with the parameters of G pre θ , and subsequently fine-tuned to better align with human preferences by encouraging it to generate outputs closer to the positive examples in our preference dataset. The objective of M-DPO is to maximize the likelihood of preferred (positive) samples over less-preferred (negative) ones, with a focus on quality-critical regions identified via local masks:
L M-DPO (π ψ ; π ref ) = -E (P,M + P ,M - P )∼D log σ βL + (P, M + P ) -βL -(P, M - P )(2)
Artist Mesh Point Cloud MeshAnythingv2 BPT DeepMesh Ours
Figure 5: Qualitative comparison on artist-designed meshes. Our method generates more coherent and visually plausible surfaces with finer structural details and fewer topological artifacts compared to baseline approaches.
where the positive and negative log-ratio terms are computed as:
L + (P, M + P ) = log ∥π ψ (M + P |P) ⊙ ϕ(M + P )∥ 1 ∥π ref (M + P |P) ⊙ ϕ(M + P )∥ 1 L -(P, M - P ) = log ∥π ψ (M - P |P) ⊙ 1 -ϕ(M - P ) ∥ 1 ∥π ref (M - P |P) ⊙ 1 -ϕ(M - P ) ∥ 1 (3
)
Here, D denotes the preference dataset, and π is the token-level probability distribution produced by the model. The operator ⊙ indicates element-wise (Hadamard) multiplication, and ∥•∥ 1 denotes the ℓ 1 norm over the token sequence. The hyperparameter β controls the sharpness of preference separation, and σ is the standard sigmoid function. M-DPO effectively preserves satisfactory regions while actively refining low-quality areas identified by the local quality mask. This targeted optimization strategy not only maintains the global structure but also enhances local geometric fidelity, offering a finer control over mesh generation quality compared to standard DPO.
this section cite: []

Section: Experiments

this section cite: []

Section: Experiment Settings
Datasets Our model is pretrained on 2M meshes from large-scale datasets including ShapeNetV2 [63], 3D-FUTURE [64], Objaverse [65], Objaverse-XL [66], and licensed assets. After filtering low-quality scans and poorly topologized CAD models, 800K meshes form the fine-tuning subset. For preference alignment, we construct a specialized dataset of 10,000 generated meshes, each paired with 8 topological variations derived from the same input point cloud. To enhance geometric generalization, meshes are perturbed at the vertex level and subsampled from an initial 50K-point cloud to 16,384 points, without enforcing watertightness. For evaluation, we employ two test sets: (1) 100 high-quality, artist-designed meshes for qualitative analysis, and (2) 100 dense, out-of-distribution meshes generated by Hunyuan2.5 [48], providing rigorous real-world validation.
More data details can be seen in Supplementary A.1.
this section cite: ['b62', 'b63', 'b64', 'b65', 'b47']

Section: Dense Mesh
Point Cloud MeshAnythingv2 BPT DeepMesh Ours
Figure 6: Generalization results on dense, out-of-distribution meshes. Our model demonstrates superior geometric fidelity and surface continuity, maintaining high-quality reconstruction even under complex and unseen input conditions.
this section cite: []

Section: Implementation Details
We pretrained on 256 NVIDIA H20 GPUs (2/GPU) for 10 days with AdamW [67] (β 1 = 0.9, β 2 = 0.99) and Flash Attention, following a 100-step linear warm-up. M-DPO post-training took 8 hours on 64 GPUs with a 5e -7 learning rate. See supplementary material A.2 for full details.
Baselines. We benchmark our approach against leading mesh generation methods, including MeshAnythingV2 [16], BPT [19], and DeepMesh [23]. Since DeepMesh only publicly provides inference code and a 512M parameter version, we use this configuration for comparison.
this section cite: ['b15', 'b18', 'b22']

Section: Qualitative Results
We qualitatively compare our method with existing baselines. As shown in Figure 5, our model produces meshes that are significantly more coherent, artistically plausible, and faithful to the input geometry, particularly in challenging regions such as fine-grained structures and curved surfaces. These results highlight our model's ability to preserve detail and maintain topological regularity. In contrast, baseline methods often exhibit structural artifacts such as incomplete regions, broken connectivity, or excessive smoothing, especially in geometrically intricate areas. To further evaluate generalization beyond the training distribution, we conduct experiments on a set of dense, highresolution meshes not seen during training. As illustrated in Figure 6, our method consistently outperforms prior approaches in reconstructing complex geometry and maintaining surface continuity under high-resolution inputs. These results demonstrate that our model not only performs well on curated artistic data but also generalizes effectively to challenging, real-world examples.
this section cite: []

Section: Quantitative Results
Table 1 presents a quantitative comparison of our method against baselines on artist-designed meshes and dense meshes derived from AI-generated representations.We report both geometric and topological metrics, including Hausdorff Distance (HD), Topology Score (TS), and Boundary Error Rate (BER). Our method consistently outperforms competing approaches across all metrics, demonstrating superior geometric fidelity and topological coherence. To further validate perceptual quality, we conducted a user study(US) in which participants were asked to compare mesh outputs based on visual plausibility and structural integrity. The results indicate a strong preference for our  method, confirming that its advantages are not only quantitatively measurable but also perceptually significant. We evaluate the efficacy of our score-based preference system within the domain of dense mesh generation. As demonstrated in Table 2, employing only Hausdorff Distance to differentiate between high-and low-quality meshes (denoted as N-DPO) yields marginal improvements in geometric consistency over the pretrained model (Pretrain) and exhibits a decrease in the TS score. Conversely, leveraging our proposed composite scoring system (denoted as S-DPO) for the construction of preference data facilitates a substantial performance gain.
this section cite: []

Section: Ablation Study

this section cite: []

Section: Score System

this section cite: []

Section: Mask DPO
Figure 4 illustrates that standard global DPO often fails to capture local variations in mesh quality. Our proposed topology-aware local mask mechanism effectively addresses this limitation by enabling the model to learn from spatially localized preference signals. Built on the preference dataset derived from our scoring system, the Mask-DPO model (denoted as M-DPO) demonstrates a clear advantage over the global score-based DPO baseline (S-DPO), as shown in Figure 7. This localized learning strategy leads to significant improvements in both quantitative metrics and human preference, as confirmed in Table 2. Notably, M-DPO produces outputs that are not only closer to the ground truth but also more consistently favored by human evaluators, providing strong empirical support for localized preference learning.
this section cite: []

Section: Conclusion
Generating high-quality 3D meshes remains a significant challenge. We introduced Mesh-RFT, a novel framework employing topology-aware scoring and Masked Direct Preference Optimization (M-DPO) for fine-grained refinement. By leveraging objective metrics and localized optimization, Mesh-RFT advances the state-of-the-art in automated mesh generation. Our approach significantly improves both the geometric accuracy and topological fidelity of generated meshes compared to previous methods. This work offers a substantial step forward in creating production-ready 3D assets for a wide range of applications. Limitations and future work are discussed in appendix C. Justification: We have discussed the limitations of our work in the appendix, which include the inability to generalize well to snake-like data due to the lack of such samples in the training dataset.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: References
Ref_id:b0 Title: Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling Year: (2016)
Ref_id:b1 Title: O-cnn: Octree-based convolutional neural networks for 3d shape analysis Year: (2017)
Ref_id:b2 Title: Diffusion probabilistic models for 3d point cloud generation Year: (2021)
Ref_id:b3 Title: Shap-e: Generating conditional 3d implicit functions Year: (2023)
Ref_id:b4 Title: Pointnet: Deep learning on point sets for 3d classification and segmentation Year: (2017)
Ref_id:b5 Title: 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models Year: (2023)
Ref_id:b6 Title: Shapellm-omni: A native multimodal llm for 3d generation and understanding Year: (2025)
Ref_id:b7 Title: Learning implicit fields for generative shape modeling Year: (2019)
Ref_id:b8 Title: Deepsdf: Learning continuous signed distance functions for shape representation Year: (2019)
Ref_id:b9 Title: Marching cubes: A high resolution 3d surface construction algorithm Year: (1998)
Ref_id:b10 Title: Polygen: An autoregressive generative model of 3d meshes Year: (2020)
Ref_id:b11 Title: Meshgpt: Generating triangle meshes with decoder-only transformers Year: (2024)
Ref_id:b12 Title: Meshanything: Artist-created mesh generation with autoregressive transformers Year: (2024)
Ref_id:b13 Title: Generic 3d mesh generation via pivot vertices guidance Year: (2024)
Ref_id:b14 Title: Meshxl: Neural coordinate field for generative 3d foundation models Year: (2025)
Ref_id:b15 Title: Meshanything v2: Artist-created mesh generation with adjacent mesh tokenization Year: (2024)
Ref_id:b16 Title: Edgerunner: Auto-regressive auto-encoder for artistic mesh generation Year: (2024)
Ref_id:b17 Title: Treemeshgpt: Artistic mesh generation with autoregressive tree sequencing Year: (2025)
Ref_id:b18 Title: Scaling mesh generation via compressive tokenization Year: (2024)
Ref_id:b19 Title: Meshtron: High-fidelity, artist-like 3d mesh generation at scale Year: (2024)
Ref_id:b20 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b21 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b22 Title: Deepmesh: Auto-regressive artist-mesh creation with reinforcement learning Year: (2025)
Ref_id:b23 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b24 Title: Advancing multilingual reasoning through multilingual alignment-as-preference optimization Year: (2024)
Ref_id:b25 Title: Enhancing llm safety via constrained direct preference optimization Year: (2024)
Ref_id:b26 Title: Aligning modalities in vision large language models via preference fine-tuning Year: (2024)
Ref_id:b27 Title: Rangeldm: Fast realistic lidar point cloud generation Year: (2024)
Ref_id:b28 Title: Pointllm: Empowering large language models to understand point clouds Year: (2024)
Ref_id:b29 Title: Pointnet++: Deep hierarchical feature learning on point sets in a metric space Year: (2017)
Ref_id:b30 Title: Locally attentional sdf diffusion for controllable 3d shape generation Year: (2023)
Ref_id:b31 Title: Sdfusion: Multimodal 3d shape completion, reconstruction, and generation Year: (2023)
Ref_id:b32 Title: Dreamfusion: Text-to-3d using 2d diffusion. arXiv Year: (2022)
Ref_id:b33 Title: Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation Year: (2023)
Ref_id:b34 Title: Magic3d: High-resolution text-to-3d content creation Year: ()
Ref_id:b35 Title: Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation Year: ()
Ref_id:b36 Title: Generative gaussian splatting for efficient 3d content creation Year: (2023)
Ref_id:b37 Title: Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models Year: (2024)
Ref_id:b38 Title: Lrm: Large reconstruction model for single image to 3d Year: (2023)
Ref_id:b39 Title: Imagereward: Learning and evaluating human preferences for text-to-image generation Year: (2023)
Ref_id:b40 Title: Single image to 3d textured mesh with convolutional reconstruction model Year: (2024)
Ref_id:b41 Title: Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models Year: (2024)
Ref_id:b42 Title: Large multi-view gaussian model for high-resolution 3d content creation Year: (2024)
Ref_id:b43 Title: Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation Year: (2023)
Ref_id:b44 Title: Clay: A controllable large-scale generative model for creating high-quality 3d assets Year: (2024)
Ref_id:b45 Title: Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer Year: (2024)
Ref_id:b46 Title: Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner Year: (2024)
Ref_id:b47 Title: Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation Year: (2025)
Ref_id:b48 Title: Sparseflex: High-resolution and arbitrary-topology 3d shape modeling Year: (2025)
Ref_id:b49 Title: High-fidelity 3d shape synthesis using large-scale rectified flow models Year: (2025)
Ref_id:b50 Title: Step1x-3d: Towards high-fidelity and controllable generation of textured 3d assets Year: (2025)
Ref_id:b51 Title: Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation Year: (2024)
Ref_id:b52 Title: Scan2mesh: From unstructured range scans to 3d meshes Year: (2019)
Ref_id:b53 Title: Generating 3d polygonal meshes with diffusion models Year: (2023)
Ref_id:b54 Title: Meshcraft: Exploring efficient and controllable mesh generation with flow-based dits Year: (2025)
Ref_id:b55 Title: Neural discrete representation learning Year: (2017)
Ref_id:b56 Title: Nautilus: Locality-aware autoencoder for scalable mesh generation Year: (2025)
Ref_id:b57 Title: iflame: Interleaving full and linear attention for efficient mesh generation Year: (2025)
Ref_id:b58 Title: Rrhf: Rank responses to align language models with human feedback Year: (2023)
Ref_id:b59 Title: Dreamreward: Text-to-3d generation with human preference Year: (2024)
Ref_id:b60 Title: Hierarchical transformers are more efficient language models Year: (2021)
Ref_id:b61 Title: Attention is all you need Year: (2017)
Ref_id:b62 Title: An information-rich 3d model repository Year: (2015)
Ref_id:b63 Title: 3d-future: 3d furniture shape with texture Year: (2021)
Ref_id:b64 Title: Objaverse: A universe of annotated 3d objects Year: (2023)
Ref_id:b65 Title: Objaverse-xl: A universe of 10m+ 3d objects Year: (2023)
Ref_id:b66 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b67 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b68 Title: Dapo: An open-source llm reinforcement learning system at scale Year: (2025)
Ref_id:b69 Title: Efficient and reliable reinforcement learning for advanced reasoning tasks Year: (2025)
