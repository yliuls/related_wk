Title: InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective
Abstract: The Segment Anything Model (SAM), a vision foundation model, exhibits impressive zero-shot capabilities in general tasks but struggles in specialized domains. Parameter-efficient fine-tuning (PEFT) is a promising approach to unleash the potential of SAM in novel scenarios. However, existing PEFT methods for SAM neglect the domain-invariant relations encoded in the pretrained model. To bridge this gap, we propose InfoSAM, an information-theoretic approach that enhances SAM fine-tuning by distilling and preserving its pre-trained segmentation knowledge. Specifically, we formulate the knowledge transfer process as two novel mutual information-based objectives: (i) to compress the domain-invariant relation extracted from pre-trained SAM, excluding pseudo-invariant information as possible, and (ii) to maximize mutual information between the relational knowledge learned by the teacher (pretrained SAM) and the student (fine-tuned model). The proposed InfoSAM establishes a robust distillation framework for PEFT of SAM. Extensive experiments across diverse benchmarks validate InfoSAM's effectiveness in improving SAM family's performance on real-world tasks, demonstrating its adaptability and superiority in handling specialized scenarios. The code and models are available at InfoSAM project page.

Section: Introduction
Recently, the Segment Anything Model (SAM) (Kirillov et al., 2023;Ravi et al., 2022) emerged as a prominent foundation model for image segmentation. While SAM demonstrates exceptional zero-shot performance on generic object segmentation, it often struggles with domain-specific real-world segmentation tasks (Chen et al., 2023;Zhong et al., 2024). Therefore, Parameter-Efficient Fine-Tuning (PEFT) for SAM (Song et al., 2024;Zhang et al., 2024a;Peng et al., 2024b) has gained attention as a promising solution, significantly reducing the fine-tuning costs associated with SAM's large pre-trained parameter set. Existing PEFT methods for SAM primarily focus on fine-tuning the heavy image encoder (Song et al., 2024;Peng et al., 2024b) or aligning domain-specific features between the mask decoder and image encoder (Xiao et al., 2025). However, a promising improvement avenue is overlooked: preserving the beneficial information in pre-trained models.
Notably, SAM follows an encoder-decoder architecture, where the mask decoder refines the image embeddings extracted by the image encoder to localize objects. Unified training or fine-tuning methods (Shu et al., 2025;Xiao et al., 2025) have demonstrated effectiveness within this framework. This suggests that preserving the implicit relationship between the encoder and decoder could be beneficial for model fine-tuning. This relationship may stem from extensive pre-training and be embedded in the feature distributions, making it delicate and easily disrupted by unrefined PEFT methods (Wang et al., 2024). We argue that this is because task-specific tuning tends to override or suppress the universal visual features learned during pre-training.
To enhance PEFT by leveraging implicit relationships, a natural approach is to extract these relationships from foundation models and inject them into fine-tuned models tailored for specific domains. However, not all implicit relationships are beneficial for downstream tasks-only the key domaininvariant relationships learned from across domains (Hoffman et al., 2018;Xu et al., 2022) contribute positively to every fine-tuned model. While knowledge distillation serves as a flexible bridge for transferring information between models (Gou et al., 2021), we propose to adopt a distillation approach between the pre-trained model and fine-tuned model to retain domain-invariant relationships.
Therefore, this brings us to two key challenges: 1) How can we extract the domain-invariant relationship from pretrained foundation models? 2) How can we effectively transfer the extracted information to fine-tuned models?
To address these challenges, we propose InfoSAM, a novel information-theoretical distillation method specifically designed for SAM PEFT. In order for the teacher to provide a good amount of the domain-invariant information, first we have to find out how this information could be quantified. To this aim, we introduce a robust and efficient Rényi's entropybased quantification from information theory (Ahn et al., 2019) to measure such a relation. However, not all the relations in the pre-trained SAM are domain-invariant, there exists some pseudo-invariant information (e.g., color), which may negatively impact the generalization ability during the fine-tuning process (Li et al., 2022a). Therefore, to address the first challenge, we propose an attention-driven relation module specifically designed to extract critical structural patterns from the pre-trained SAM. By minimizing mutual information between the module's outputs and both encoderdecoder embeddings of SAM, it constructs an effective bottleneck that forces the module to maintain compressed yet domain-invariant representations. Furthermore, to tackle the second challenge, we effectively distill the valuable relational knowledge from pre-trained SAM to the fine-tuned SAM by maximizing the mutual information between their extracted relations. This ensures faithful propagation of compressed semantic dependencies, thereby facilitating a more effective fine-tuning process. Our experiments on SAM and SAM2, evaluated across 4 diverse domains and 8 datasets, show that InfoSAM achieves superior adaptation and segmentation performance.
Overall, our contribution can be summarized as follows:
• We present InfoSAM, the first information-theoretic framework for SAM adaptation, introducing an innovative distillation approach tailored for SAM PEFT to enhance performance in new scenarios.
• InfoSAM proposes novel dual complementary mechanisms for SAM adaptation: a relational bottleneck that strategically compresses task-irrelevant dependencies while preserving domain-invariant semantics, coupled with adaptive cross-model mutual information maximization ensuring provable preservation of essential structural knowledge.
• We conduct a comprehensive benchmark across diverse domains, including natural images, medical imaging, agriculture, and remote sensing. InfoSAM consistently demonstrates superior performance compared to other PEFT and distillation techniques across various downstream tasks.
this section cite: ['b25', 'b37', 'b10', 'b52', 'b41', 'b41', 'b45', 'b39', 'b45', 'b43', 'b18', 'b46', 'b17', 'b0']

Section: Related Work

this section cite: []

Section: Parameter Efficient Fine-Tuning for SAM
Parameter-Efficient Fine-Tuning (PEFT) alleviates the challenges of task-specific deployment in large foundation models by fine-tuning only a minimal subset of parameters while keeping the majority frozen. Several prior works explore fine-tuning SAM for downstream tasks. SAM-Adapter (Chen et al., 2023) is one of the pioneering works applying the PEFT method to SAM, incorporating taskspecific prompts for each adapter. SU-SAM (Song et al., 2024) presents a simple framework to efficiently fine-tune the SAM with Adapter or LoRA. SAM-COBOT (Peng et al., 2024b) boosts existing PEFT techniques for fine-tuning SAM through cross-block orchestration. BLO-SAM (Zhang et al., 2024a) finetunes SAM based on bi-level optimization, eliminating the need for manual prompts by a learnable prompt embedding. Conv-LoRA (Zhong et al., 2024) integrates ultra-lightweight convolutional parameters into LoRA, injecting image-related inductive biases into the plain ViT encoder.
However, the above methods overlook preserving pretrained information in foundation models during fine-tuning.
Our work explores enhancing fine-tuning methods for SAM
Mask Decoder Prompt Encoder Layer-1 Patch Embed … Image Encoder Layer-2 Layer-N Tunable Adapters Relation Module Relation Module Mask Decoder Prompt Encoder Layer-1 Patch Embed … Image Encoder Layer-2 Layer-N Imagefeature Imagefeature Mask token Mask token 𝑧 𝑖 𝑇 𝑧 𝑖 𝑆 𝑧 𝑚 𝑇 𝑧 𝑚 𝑆 Bottleneck 𝐼 𝛼 𝑧 𝑖 𝑇 , 𝑧 𝑚 𝑇 ; 𝑟 𝑇 ≤ 𝐼 𝑐 𝑚𝑎𝑥 𝜔 𝐼 𝛼 𝑟 𝑇 ; 𝑟 𝑆 Optimization Objective Relation Information Relation Information 𝑋~𝐷 Distillation Frozen Parameters Full Training 𝐼 𝛼 𝑧 𝑖 𝑇 , 𝑧 𝑚 𝑇 ; 𝑟 𝑇 𝑧 𝑖 𝑇 , 𝑧 𝑚 𝑇 𝑟 𝑇 Compress Domain-invariant Information ℒ 𝑟 𝐼 𝛼 𝑟 𝑇 ; 𝑟 𝑆 ℒ 𝑑 Distill Domain-invariant Information 𝑟 𝑇 𝑟 𝑆 from a novel perspective: leveraging information-based distillation to maintain domain-invariant relationships.
this section cite: ['b10', 'b41', 'b52']

Section: Knowledge Distillation for SAM
Knowledge distillation (KD) (Gou et al., 2021) effectively transfers knowledge from a large, well-trained model (teacher) to a smaller or simpler one (student). When applying KD to SAM, most efforts focus on compressing and transferring representations for downstream tasks. Mobile-SAM (Zhang et al., 2023a) distills SAM's ViT encoder into a TinyViT, while TinySAM (Shu et al., 2025) uses full-stage KD. Other approaches distill SAM's semantic priors for tasks like medical segmentation (Dong et al., 2024;Shen et al., 2024) and image restoration (Zhang et al., 2024b). However, these methods focus on paired feature maps, neglecting the inter-module relationships within the teacher SAM. To address this, our approach utilizes informationtheoretic principles to extract and transfer compact intermodule relationships to the student model.
this section cite: ['b17', 'b39', 'b12', 'b38']

Section: Domain-invariant Information in SAM
The concept of domain-invariant information was first introduced in prior works on domain adaptive segmentation (DAS), which explored cross-domain invariant features such as edge and structural information (Hoffman et al., 2018). DAS aims to learn domain-invariant representations across multiple domains and follows two main approaches: (i) extraction and refinement of domain-invariant features,where methods like feature disentanglement (Chang et al., 2019) or analysis (Xu et al., 2022) decompose images into domaininvariant (e.g., shapes, edges) and domain-specific (e.g., textures, colors) components, aiming to enhance the former while suppressing the latter; (2) GAN-based domaininvariant feature generation, which employs adversarial training to align domains at different levels: image (Li et al., 2022b), feature (Ma et al., 2024), and output (Huang et al., 2022). For example, GLGAN (Ma et al., 2024) integrates multi-scale global and local features to improve cross-domain transferability in remote sensing. SAM's large-scale pretraining encodes domain-invariant patterns for strong zero-shot generalization. Recent works leverage these universal visual patterns for downstream tasks (Peng et al., 2024a). However, these methods rely on complex designs or external data to learn representations. In contrast, we focus on preserving the domain-invariant information in pre-trained SAM for fine-tuning.
this section cite: ['b18', 'b6', 'b46', 'b30', 'b20', 'b30']

Section: Preliminaries

this section cite: []

Section: Rényi's α-entropy and Mutual Information
In information theory, matrix-based Rényi's α-entropy provides a novel way to quantify single-variable information or interactions across variables directly from samples. Unlike Shannon entropy, it leverages the eigenspectrum of a Gram matrix in reproducing kernel Hilbert space (RKHS), avoiding costly distribution evaluations (Gong et al., 2022).
Definition 1. Let κ : X × X → R be an infinitely divisible positive kernel (Bhatia, 2006). Given {x i } n i=1 ⊂ X , each x i being a real-valued scalar or vector, and the Gram matrix K obtained from K ij = κ(x i , x j ), a matrix-based analog to Rényi's entropy can be defined as:
S α (A) = 1 1 -α log 2 n i=1 λ α i (A)(1)
where the kernel matrix
A ij = 1 n Kij √ KiiKjj
is the normalized version of K and tr(A) = 1. The λ i (A) denotes the ith eigenvalue of A.
Definition 2. Given n pairs of samples {z i = (x i , y i )} n i=1 , and two positive definite kernels κ 1 : X × X → R and κ 2 : Y × Y → R. After computing the Gram matrix A and B, a joint Rényi's entropy can be defined as:
S α (A, B) = S α A • B tr(A • B)(2)
where (A • B) denotes the Hadamard product between the matrices A and B. The mutual information I α (A; B) can be computed as:
I α (A; B) = S α (A) + S α (B) -S α (A, B)(3)
The matrix-based Rényi's mutual information eliminates the need for high-dimensional probability density estimation of Shannon entropy, offering a more accurate and computationally efficient solution (Dong et al., 2023).
this section cite: ['b16', 'b3', 'b13']

Section: Methodology

this section cite: []

Section: Background and Notions
The overview of InfoSAM is illustrated in Fig. 2. Given a teacher model and a student model, we denote the pretrained SAM as ϕ T and the fine-tuned SAM as ϕ S , which are parameterized by ω. Let X ∼ D be an input sampled from the downstream dataset D. The representations produced by ϕ T (X) and ϕ S (X) are defined as follows: The output features of the image encoder are denoted as z T i and z S i , where z T i , z S i ∈ R B×H×W ×D . Here, B is the batch size, H and W represent the height and width, respectively, and D is the dimension of the image embeddings. Similarly, the output tokens from the two-way transformer in the mask decoder are denoted as z T m and z S m , where z T m , z S m ∈ R B×N ×D . These tokens encode the target mask information in a more abstract manner. Here, N represents the number of masks, and the output token shares the same dimension D as the image embeddings.
The goal of PEFT is to fine-tune ϕ S (X; ω) for adaptation to a new downstream task under the supervision of the teacher model ϕ T , where ω denotes the trainable PEFT parameters.
To enhance the PEFT process using the frozen pre-trained teacher SAM, the loss can be formulated as:
ω * = arg min ω L(X, Y | T ),(4)
where T represents intermediate features extracted by the teacher model, capturing the relational information within SAM modules, and Y is the full dense label map. Following prior work, the task-specific loss function L(•) is chosen as the structure loss (Zhong et al., 2024).
In this paper, rather than directly aligning paired representations between teacher and student (Zhang et al., 2023a;Shu et al., 2025), we leverage robust prior relational information from the pre-trained SAM to guide the PEFT process.
this section cite: ['b52', 'b39']

Section: An Information View of SAM Distillation
Problem Formulation. Intuitively, the relationships between different modules in a well-trained foundation model are invaluable, as they are learned from extensive datasets. However, traditional PEFT methods for SAM, when finetuned to downstream tasks, risk disrupting these relationships. In this way, we need to address two key questions: how to capture critical relations from the pre-trained SAM and how to effectively transfer it to the fine-tuned model.
Firstly, by treating the teacher model as a mapping function, we argue that the critical relation information resides in multivariate mutual information I α (z T i , z T m ; r T ), where z T i , z T m and r T represent the image embedding, mask token and relational interactions of the teacher, respectively. As such, I α (z T i , z T m ; r T ) quantifies how much information r T can tell about (z T i , z T m ). Crucially, this mutual information constitutes a learnable bottleneck that fundamentally constrains the knowledge transfer process. The bottleneck mechanism enforces selective attention by restricting the information flow to a compressed representation, where only the most salient teacher-student interactions can be preserved. This is particularly vital as not all relations in the pre-trained SAM are universally transferable. For example, invariant features (e.g., geometric outlines) that exhibit cross-domain consistency are effective, while pseudo-invariant features (e.g., color distributions) that carry domain-specific biases need to be suppressed (Li et al., 2022a).
To prioritize domain-invariant relations, we constrain the information flow via an upper bound I c :
I α (z T i , z T m ; r T ) ≤ I c (5
)
where r T = f T (z T i , z T m ; θ) represents the teacher's relational mapping between image embeddings and mask tokens. The relation module is defined as W ) . The θ represents the parameters of the relation module. This compression forces the module to retain only essential information for distillation.
f T (z T i , z T m ; θ) : R B×H×W ×D × R B×N ×D → R B×N ×(H•
After that, we employ a distillation approach to transfer the core relationships by maximizing their mutual information:
max ω I α (r T ; r S ) subject to I α (z T i , z T m ; r T ) ≤ I c(6)
where r S = f S (z S i , z S m ; θ) denotes the student's relation in fine-tuned SAM, with f S sharing the same parameters as f T . The information bottleneck principle operates through two coupled mechanisms: (i) compression via minimizing I α (z T i , z T m ; r T ) to extract minimal sufficient statistics r T from (z T i , z T m ), and (ii) distillation via maximizing I α (r T ; r S ) to preserve maximal predictive information. The Lagrangian formulation explicitly implements this trade-off:
max ω I α (r T ; r S ) -βI α (z T i , z T m ; r T ) (7
)
where β is a hyper-parameter for trade-off.
this section cite: []

Section: Compressing Intra-SAM Relations.
To efficiently capture the relationship within pre-trained SAM, we propose an attention-based module designed for extraction, illustrated in Fig. 3. Given z T i , the output of the image encoder of SAM, and z T m , the mask token embedding, as the input of relation module f T . It mainly uses a combination of attention mechanisms and residual connections.
First, both z T i and z T m are passed through a Layer Normalization step to stabilize the features. After that, z T m and z T i are linearly projected into a query vector Q ∈ R B×N ×D and a key vector K ∈ R B×(H•W )×D , respectively:
Q = W Q • LayerN orm(z T m ), K = W K • LayerN orm(z T i )(8)
where W Q , W K ∈ R D×D are learnable projection matrices. The attention scores are computed by combining two components: the scaled dot product of Q and K and the residuals from the dot product of z T m and z T i . The scores are summed to produce the final attention map:
S α = QK ⊤ √ D + z T m • z T ⊤ i (9
)
where S α is the attention score. To ensure consistency and comparability, α is flattened and normalized using ℓ 2normalization, resulting in the final output of f T , denoted as r T . To encourage the relation encoding process to focus on domain-invariant information, the first loss for relation compression can be expressed as:
L r =I α (z T i , z T m ; r T ) = S α (G T i , G T m ) + S α (G T r ) -S α (G T i , G T m , G T r )(10)
where It is designed to capture the relationship between image encoder and mask decoder, facilitating effective interaction between these components in SAM.
G T i , G T m , G T r ∈ R N ×N
output of r T with a polynomial kernel of degree 1. Notably, the teacher entropy term in this loss is excluded, as the teacher's weights remain fixed during PEFT.
According to Eq.( 1), computing eigenvalues of large matrices is computationally intensive (Kerr et al., 2009;Yu et al., 2019). To mitigate this, we set α = 2, allowing us to compute matrix-based Rényi's α-entropy via the Frobenius norm:
|A∥ 2 F = tr(AA H ) = n i=1 λ 2 i (A).
Consequently, L r can be reformulated as:
L r = -log 2 ∥G T r ∥ 2 F + log 2 ∥G T imr ∥ 2 F (11
)
where
G T imr = G T i • G T m • G T r .
The • is Hadamard product. The first term in L r acts as a spectral compression regularizer that constrains the relation module and encourages it to learn more compact and refined representations. The second term minimizes the joint entropy of the feature interactions across the image encoder, mask decoder, and relation module, effectively filtering spurious relationships and preserving domain-invariant interactions critical for cross-domain adaptation.
Maximizing Inter-SAM Relations. After extracting the essential relationships between the image encoder and the mask decoder, we transfer the relationships by minimizing their distance. A natural choice to accomplish this is by maximizing the mutual information between the two representations. While most existing works (Ahn et al., 2019;Kuang et al., 2023) focus on minimizing a lower bound of mutual information, we directly maximize the matrix-based Rényi's mutual information itself to avoid the expensive evaluation of underlying distribution for distillation loss:
L d = -I α (r T ; r S ) = -S α (G T r ) -S α (G S r ) + S α (G T r , G S r )(12)
Similarly, G T r and G S r denote the Gram matrices corresponding to the student and teacher relations, respectively. We denote the G T S r = G T r • G S r , then the distillation loss can be expressed as:
L d = log 2 ∥G T r ∥ 2 F + log 2 ∥G S r ∥ 2 F -log 2 ∥G T S r ∥ 2 F (13
)
Consistent with L r , the L d also sets the entropy order α to 2 and utilizes the Frobenius norm for equivalent transformation. From this perspective, the components of L d can be viewed as regularization terms (i.e., the first two terms) and a relation alignment (i.e., the third term) between two models, while log 2 improving robustness to relations.
Overall, combining Eq.( 11) and Eq.( 13), the final objective of relation compression and transfer can be defined as:
L inf o = λ 1 * L r + λ 2 * L d (14
)
where λ 1 and λ 2 are hyper-parameters to trade-off between sufficiency (domain-invariant information transmitted from r T to r S ) and minimality (the complexity of r T ). Further details and PyTorch-style pseudocode for InfoSAM are provided in Appendix A.2 and A.3.
this section cite: ['b24', 'b47', 'b0', 'b26']

Section: Applying information theory to SAM
Overall Loss Function. Following previous works (Zhong et al., 2024), we incorporate the proposed informationtheoretic distillation loss L inf o with a structure loss L ce (Fan et al., 2020b), which combines the weighted IoU loss and binary cross-entropy loss. The overall loss function is derived as:
L = L ce + L inf o (15
)
Finally, we employ this new loss function for fine-tuning SAM. During fine-tuning, we first learn robust relations and then transfer this knowledge. The L inf o regulates information flow between SAM's hierarchical representations, avoiding over-retention of low-level details while enhancing geometrically critical features. This aligns with the ratedistortion tradeoff in information bottleneck theory (Tishby & Zaslavsky, 2015), where information is compressed and then generalized.
this section cite: ['b52', 'b42']

Section: Experiments
Settings. We conduct experiments using SAM (Kirillov et al., 2023) (with a ViT-B backbone) and SAM2 (Ravi et al., 2022) (with a Hiera-B+ backbone) with Adapter (Chen et al., 2022;Song et al., 2024), and LoRA (Hu et al., 2022) across four real-world domains: medical imaging, natural images, agriculture, and remote sensing. We fine-tune SAM's image encoder by adding adapters or LoRA, while fully training the decoder directly. We use a batch size of 4 and the Adam optimizer with an initial learning rate of 2 × 10 -4 , utilizing a CosineAnnealing scheduler that decays to a final learning rate of 2 × 10 -5 . All the methods are trained for 10 epochs with structure loss (i.e., the combination of weighted IoU loss and binary cross entropy loss) unless otherwise specified. During training, prompts are randomly selected from noised ground truth boxes and points at a 1:1 ratio. During evaluation, ground truth boxes are used as the default geometric input prompts to ensure a fair comparison and minimize randomness. More implementation details are provided in Appendix B.
this section cite: ['b25', 'b37', 'b9', 'b41', 'b19']

Section: Datasets.
In the natural image domain, we focus on camouflaged object segmentation (Skurowski et al., 2018;Le et al., 2019;Fan et al., 2020a). For medical imaging, we investigate polyp segmentation (Bernal et al., 2015;Jha et al., 2020) and skin lesion segmentation (Codella et al., 2018). In agriculture and remote sensing, we use leaf disease segmentation (Rath, 2023) and road segmentation datasets (Mnih, 2013) as representative examples, respectively. For further details on the tasks and datasets, please refer to Appendix C.
To verify the effectiveness of our approach, we compare it with two categories of methods: PEFT methods and distillation methods.
PEFT Baselines. The PEFT baselines encompass three types of methods: the direct application of SAM, PEFT methods from the NLP or CV domain, and PEFT methods designed for SAM. These are as follows: 1) The zeroshot performance of the original SAM. 2) Fine-tune SAM's mask decoder only. 3) BitFit (Ben Zaken et al., 2022), which only fine-tunes bias terms in the pre-trained model. 4) AdaptFormer (Chen et al., 2022), which inserts the trainable bottleneck layers into the MLP block of the transformer. 5) LoRA (Hu et al., 2022) inserts trainable bottleneck layers parallel to the frozen linear weight. 6) HQSAM (Ke et al., 2024), which introduces a learnable high-quality output token and enhances mask details by fusing mask decoder features with both early and final ViT features. 7) SU-SAM (Song et al., 2024) presents a simple framework that efficiently fine-tunes the SAM using Adapter or LoRA. 8) ConvLoRA-SAM (Zhong et al., 2024) injects image-related inductive biases into the image encoder of SAM by integrating ultra-lightweight convolutional parameters into LoRA.
Distillation Baselines. In this study, we compare our method with the following baselines: 1) Logit-based distillation (Zhu et al., 2018). 2) single-layer paired feature distillation (i.e., PKD (Cao et al., 2022), PKT (Passalis et al., 2020)), which uses one-stage feature to distill knowledge, with MobileSAM (Zhang et al., 2023a) belonging to this category. 3) multiple-layers paired feature distillation (i.e., VID (Ahn et al., 2019), IBD (Kuang et al., 2023)), which utilizes multi-stage information to transfer knowledge, with each layer aligned separately. Similarly, TinySAM (Shu  (Chen et al., 2021a), Re-viewKD (Chen et al., 2021b)), which utilizes knowledge from multiple layers of the teacher model to supervise the student, by leveraging diverse information extracted from these layers. Currently, no work in SAM has explored crosslayer fusion distillation for PEFT. InfoSAM is the first to address this.
We report the main experimental results on representative datasets from different domains. Additional experimental results are provided in Appendix D, and visualization results are available in Appendix G. All experiments are conducted three times to mitigate randomness, with both average values and standard errors reported.
this section cite: ['b40', 'b27', 'b2', 'b21', 'b11', 'b36', 'b32', 'b1', 'b9', 'b19', 'b23', 'b41', 'b52', 'b53', 'b5', 'b33', 'b0', 'b26']

Section: Segment Anything Across Diverse Domains
We compare InfoSAM with two categories of methods: PEFT methods and distillation methods. The results are presented in Table 1 and Table 2, respectively. In Table 1, all PEFT methods outperform both the zero-shot performance and decoder-only fine-tuning, highlighting the importance of unified fine-tuning for SAM. Additionally, InfoSAM outperforms other PEFT techniques across various datasets from different domains. Compared to other PEFT methods, InfoSAM preserves the pre-trained, domain-invariant knowledge through information-based distillation, which proves effective in enhancing segmentation performance.
In Table 2, it is noteworthy that most distillation methods are detrimental during PEFT, leading to worse performance compared to fine-tuning without distillation. Specifically, TinySAM employs full-stage distillation, requiring the stu-   dent's features to fully mimic the teacher at every stage. However, this becomes catastrophic when the teacher performs poorly (e.g., achieving only 7.2% IoU on the Road dataset). In contrast, InfoSAM further enhances PEFT performance in this challenging scenario, likely due to the relation compression process during distillation, which ensures the student model learns only the essential information from the teacher model.
this section cite: []

Section: Extended Experiment with SAM2
Note that our method is orthogonal to model development, making it easily transferable to SAM2 backbones. As shown in Table 3, InfoSAM demonstrates consistent effectiveness with SAM2. This transferability is attributed to InfoSAM's foundation in information-theoretic derivation, which is structure-independent.
this section cite: []

Section: Distillation Across Models of Different Sizes
We also verify the effectiveness of InfoSAM when scaling the teacher model to larger sizes. As shown in Fig. 4, InfoSAM shows comparable improvements to distillation methods specifically designed for SAM compression. It indicates that InfoSAM is better suited for PFET, even in traditional large-to-small knowledge distillation scenarios.
this section cite: []

Section: Ablation Study
Ablation of Main Components. We conduct an ablation study to evaluate the impact of InfoSAM's components: relation compression loss L r and relation distillation loss L d on three datasets: Kvasir, Leaf, and Road. Using SAM with an adapter as the baseline (Table 4, row 1), row 2 introduces a fixed-dot relation module with mutual information-based distillation loss. Results show that incorporating simple relations improves performance, e.g., 0.8% on Leaf, and mutual information effectively transfers relational features. Combining both losses yields further gains, e.g., 1.4% on Leaf.
this section cite: []

Section: Effects of Relation Module (RM).
To investigate the impact of the proposed relation module, we first conduct experiments to verify its effectiveness in enhancing various distillation methods. Specifically, we evaluate and compare the performance of two compact models, MobileSAM and TinySAM, with and without integrating the relation module, as shown in Table 5. We can observe a significant improvement with RM, e.g., 1.9% IoU on the Leaf dataset. These results suggest that the relation module can effectively capture and leverage high-level semantic information, thereby providing complementary benefits to existing distillation strategies during the fine-tuning stage.
Furthermore, Table 6 illustrates the effectiveness of domaininvariance dependencies in the relation module, we directly apply the module trained on one specific domain to another domain with entirely different knowledge. The results show that it still maintains satisfying results. Moreover, we conduct experiments to explore the nature of domain-invariant information. We use the Boundary F1 Score (Zhang et al., 2023b) to evaluate such universal patterns. The results show that our methods employing the relation module perform better in preserving structural edge features. More results and analysis are available in Appendix F.
this section cite: []

Section: Conclusion
We introduce InfoSAM, an information-theoretic tuning framework designed for SAM adaptation. From an information bottleneck perspective, we extract domain-invariant knowledge from the pre-trained SAM and inject it into the fine-tuned SAM to enhance adaptation efficiency. Specifically, we first propose an attention-based module to capture structural relations while minimizing mutual information to retain the most essential ones. These relations are then transferred by maximizing their mutual information. Extensive evaluations across eight segmentation datasets spanning diverse domains and tasks strongly validate the effectiveness of InfoSAM.
this section cite: []

Section: References
Ref_id:b0 Title: Variational information distillation for knowledge transfer Year: (2019)
Ref_id:b1 Title: BitFit: Simple parameter-efficient fine-tuning for transformer-based masked language-models Year: (2022)
Ref_id:b2 Title: Wm-dova maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians Year: (2015)
Ref_id:b3 Title: Infinitely divisible matrices Year: (2006)
Ref_id:b4 Title: Convex optimization Year: (2004)
Ref_id:b5 Title: General distillation framework for object detectors via pearson correlation coefficient Year: (2022)
Ref_id:b6 Title: All about structure: Adapting structural information across domains for boosting semantic segmentation Year: (2019)
Ref_id:b7 Title: Cross-layer distillation with semantic calibration Year: (2021)
Ref_id:b8 Title: Distilling knowledge via knowledge review Year: (2021)
Ref_id:b9 Title: Adaptformer: Adapting vision transformers for scalable visual recognition Year: (2022)
Ref_id:b10 Title: Adapting segment anything in underperformed scenes Year: (2023)
Ref_id:b11 Title: Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic) Year: (2018)
Ref_id:b12 Title: An efficient segment anything model for the segmentation of medical images Year: (2024)
Ref_id:b13 Title: Optimal randomized approximations for matrix-based rényi's entropy Year: (2023)
Ref_id:b14 Title: Camouflaged object detection Year: (2020)
Ref_id:b15 Title: Pranet: Parallel reverse attention network for polyp segmentation Year: (2020)
Ref_id:b16 Title: Computationally efficient approximations for matrix-based rényi's entropy Year: (2022)
Ref_id:b17 Title: Knowledge distillation: A survey Year: (2021)
Ref_id:b18 Title: Cycada: Cycle-consistent adversarial domain adaptation Year: (2018)
Ref_id:b19 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b20 Title: Multi-level adversarial network for domain adaptive semantic segmentation Year: (2022)
Ref_id:b21 Title: Kvasirseg: A segmented polyp dataset Year: (2020)
Ref_id:b22 Title: Segment anything is not always perfect: An investigation of sam on different real-world applications Year: (2024)
Ref_id:b23 Title: Segment anything in high quality Year: (2024)
Ref_id:b24 Title: Qr decomposition on gpus Year: (2009)
Ref_id:b25 Title: Segment anything Year: (2023)
Ref_id:b26 Title: Improving adversarial robustness via information bottleneck distillation Year: (2023)
Ref_id:b27 Title: Anabranch network for camouflaged object segmentation Year: (2019)
Ref_id:b28 Title: Invariant information bottleneck for domain generalization Year: (2022)
Ref_id:b29 Title: A stepwise domain adaptive segmentation network with covariate shift alleviation for remote sensing imagery Year: (2022)
Ref_id:b30 Title: Decomposition-based unsupervised domain adaptation for remote sensing image semantic segmentation Year: (2024)
Ref_id:b31 Title: Mobilevos: Real-time video object segmentation contrastive learning meets knowledge distillation Year: (2023)
Ref_id:b32 Title: Machine learning for aerial image labeling Year: (2013)
Ref_id:b33 Title: Probabilistic knowledge transfer for lightweight deep representation learning Year: (2020)
Ref_id:b34 Title: Learning to adapt sam for segmenting cross-domain point clouds Year: (2024)
Ref_id:b35 Title: Parameter efficient fine-tuning via cross block orchestration for segment anything model Year: (2024)
Ref_id:b36 Title: Leaf disease segmentation dataset Year: (2023-01-18)
Ref_id:b37 Title: Segment anything in images and videos. Proceedings of the International Conference on Learning Representations Year: (2022)
Ref_id:b38 Title: Fastsam3d: An efficient segment anything model for 3d volumetric medical images Year: (2024)
Ref_id:b39 Title: Pushing the envelope for efficient segment anything model Year: (2025)
Ref_id:b40 Title: Animal camouflage analysis: Chameleon database Year: (2018)
Ref_id:b41 Title: A simple unified framework for adapting segment anything model in underperformed scenes Year: (2024)
Ref_id:b42 Title: Deep learning and the information bottleneck principle Year: (2015)
Ref_id:b43 Title: Empowering sam to continually learn from dynamic domains Year: (2024)
Ref_id:b44 Title: Medical sam adapter: Adapting segment anything model for medical image segmentation Year: (2025)
Ref_id:b45 Title: Cat-sam: Conditional tuning for fewshot adaptation of segment anything model Year: (2025)
Ref_id:b46 Title: Dirl: Domain-invariant representation learning for generalizable semantic segmentation Year: (2022)
Ref_id:b47 Title: Multivariate extension of matrix-based rényi's α-order entropy functional Year: (2019)
Ref_id:b48 Title: Faster segment anything: Towards lightweight sam for mobile applications Year: (2023)
Ref_id:b49 Title: Blo-sam: Bi-level optimization based finetuning of the segment anything model for overfitting-preventing semantic segmentation Year: (2024)
Ref_id:b50 Title: Distilling semantic priors from sam to efficient image restoration models Year: (2024)
Ref_id:b51 Title: Learning shape-invariant representation for generalizable semantic segmentation Year: (2023)
Ref_id:b52 Title: Convolution meets loRA: Parameter efficient finetuning for segment anything model Year: (2024)
Ref_id:b53 Title: Knowledge distillation by onthe-fly native ensemble Year: (2018)
