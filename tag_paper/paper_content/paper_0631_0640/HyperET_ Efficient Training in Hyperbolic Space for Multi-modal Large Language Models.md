Title: HyperET: Efficient Training in Hyperbolic Space for Multi-modal Large Language Models
Abstract: Multi-modal large language models (MLLMs) have emerged as a transformative approach for aligning visual and textual understanding. They typically require extremely high computational resources (e.g., thousands of GPUs) for training to achieve cross-modal alignment at multi-granularity levels. We argue that a key source of this inefficiency lies in the vision encoders they widely equip with, e.g., CLIP and SAM, which lack the alignment with language at multi-granularity levels. To address this issue, in this paper, we leverage hyperbolic space, which inherently models hierarchical levels and thus provides a principled framework for bridging the granularity gap between visual and textual modalities at an arbitrary granularity level. Concretely, we propose an efficient training paradigm for MLLMs, dubbed as HyperET, which can optimize visual representations to align with their textual counterparts at an arbitrary granularity level through dynamic hyperbolic radius adjustment in hyperbolic space. HyperET employs learnable matrices with Möbius multiplication operations, implemented via three effective configurations: diagonal scaling matrices, block-diagonal matrices, and banded matrices, providing a flexible yet efficient parametrization strategy. Comprehensive experiments across multiple MLLM benchmarks demonstrate that HyperET consistently improves both existing pre-training and fine-tuning MLLMs clearly with less than 1% additional parameters. Code is available at https://github.com/godlin-sjtu/HyperET.

Section: Introduction
Thanks to advancements in pre-trained foundation models in both computer vision and natural language processing [60,61,58,1,29,50,14,18,50,15], researchers have been inspired to explore the alignment of visual and language models, leading to the development of multi-modal large language models (MLLMs). This alignment is often achieved through adapters, e.g., Q-former [16]. As a result, MLLMs [2,10,16,40,49,4,9,65] rapidly develop in recent years, demonstrating strong performance in tasks that require both visual and textual understanding, e.g., image captioning and visual question answering (VQA).
Although modern multimodal large language models (MLLMs), e.g., Qwen-VL series [5,64,4] and Intern-VL series [13,12,11], achieve cross-modal alignment across a wide range of MLLM tasks that inherently involve multi-granularity levels, their success heavily depends on extensive data scaling and massive computational resources. For example, InternVL [13] requires training on hundreds of millions of image-text pairs using up to 640 GPUs. Such a resource-intensive training scheme raises serious concerns about efficiency, reproducibility, and long-term sustainability in the MLLM community, especially for researchers and institutions constrained by limited computational resources. To address these concerns, it is essential to identify the underlying cause of this inefficiency. We argue that a key factor lies in the vision encoders that are commonly used, e.g., CLIP [54], SAM [29], and DINOv2 [46]. These encoders are typically aligned with language at a single granularity level, e.g., either pixel-level or object-level, and are thus insufficient to deal with tasks required for alignments at different granularity levels. This mismatch in granularity during training significantly impedes the optimization process, leading to inefficient cross-modal alignment and increased reliance on large-scale computational resources.
To solve this issue, we propose to directly quantify the granularity levels by leveraging hyperbolic space [7]. Empirical observations in prior works demonstrate that visual representations at different hierarchical levels (e.g., image-level and object-level) naturally stratified in hyperbolic space [20,52,47]. This property enables the use of hyperbolic radius [57]-defined as the distance from a point to the origin in hyperbolic space-to quantify granularity levels [47]. Specifically, points in hyperbolic space closer to the origin (smaller radius) encode low-level visual features (e.g., pixel-level information), while points near the boundary (larger radius) represent high-level visual semantics (e.g., image-level concepts). This hierarchical level suggests that adjusting the hyperbolic radius of visual representations can effectively align them with language models at arbitrary granularity levels.
Building on this insight, we propose an efficient training paradigm (HyperET) for MLLMs that is capable of optimizing visual representations to align with their textual counterparts at arbitrary granularity levels. This is achieved through learnable matrices equipped with Möbius multiplication operations [62], which enable direct and continuous adjustment of the hyperbolic radius of visual representations, thereby facilitating cross-modal alignment. In practice, we introduce three parameterefficient forms of the learnable matrices for adjusting the hyperbolic radius: (1) diagonal scaling matrices, (2) block-diagonal scaling matrices, and (3) banded scaling matrices. These designs significantly reduce the number of trainable parameters while retaining the capacity to address granularity mismatch. To further enhance parameter flexibility, the matrices can be extended to a dense version with fully populated learnable elements. This expansion increases the expressive capacity of our proposed training paradigm, facilitating more effective alignment across diverse cross-modal scenarios.
To evaluate the generalization capability and effectiveness of HyperET, we conduct extensive experiments across multiple pre-trained MLLM benchmarks and various downstream MLLM tasks, e.g., ScienceQA [42]. Results demonstrate that the proposed HyperET can be easily plugged-and-play and consistently improve various MLLMs, including LLaVA-1.5 [39] and LLaVA-Next [32] for pre-training, as well as MemVP [26] and LaVIN [43] for fine-tuning on downstream tasks. Notably, HyperET introduces less than 1% of the total trainable parameters, ensuring high parameter efficiency.
this section cite: ['b59', 'b60', 'b57', 'b0', 'b28', 'b49', 'b13', 'b17', 'b49', 'b14', 'b15', 'b1', 'b9', 'b15', 'b39', 'b48', 'b3', 'b8', 'b64', 'b4', 'b63', 'b3', 'b12', 'b11', 'b10', 'b12', 'b53', 'b28', 'b45', 'b6', 'b19', 'b51', 'b46', 'b56', 'b46', 'b61', 'b41', 'b38', 'b31', 'b25', 'b42']

Section: Related Work

this section cite: []

Section: Multi-modal Large Language Models
Multi-modal large language models (MLLMs) [31,10,33,38,2,55,36,70] make significant breakthroughs in recent advancements, aiming to equip large language models [73,1,60,61,3] with the capability to process and interpret visual information. Most MLLMs achieve this goal by integrating a CLIP's vision encoder [69,54] into pre-trained large language models through adapters, e.g., MLPs [40,39], Q-Former [16,35], and attention mechanisms [2]. Despite its effectiveness, this straightforward connection between the visual and language modalities still fails to align pre-trained models from different modalities, thereby resulting in inferior performance, e.g., hallucinations, across various downstream tasks.
this section cite: ['b30', 'b9', 'b32', 'b37', 'b1', 'b54', 'b35', 'b69', 'b72', 'b0', 'b59', 'b60', 'b2', 'b68', 'b53', 'b39', 'b38', 'b15', 'b34', 'b1']

Section: Towards Alignment in MLLM
This failure mainly stems from the fact that the CLIP's vision encoder [50] is designed for standard classification tasks and is not equipped to handle the more fine-grained visual understanding tasks required by the language modality [59,51,27]. To better align the vision and language modalities in MLLMs, recent works explore parameter-efficient fine-tuning methods [43,56,71,26,44]. For example, LaVIN [43] introduces adapters in both the vision encoder and LLaMA [60] to achieve better alignment of the modalities. Another line of research [66,59,27] seeks to overcome this bottleneck by incorporating additional vision models, such as DINOv2 [46], to construct a more powerful and capable vision branch. Despite the advancements in the field, few studies focus on understanding the changes in the representation of vision encoders that enable MLLMs to tackle more complex vision tasks. In this work, we aim to bridge this gap by leveraging hyperbolic space to directly model the granularity levels and adapt the granularity of visual representations to an appropriate level.
this section cite: ['b49', 'b58', 'b50', 'b26', 'b42', 'b55', 'b70', 'b25', 'b43', 'b42', 'b59', 'b65', 'b58', 'b26', 'b45']

Section: Learning in Hyperbolic Space
Unlike Euclidean space, hyperbolic space can be viewed as the continuous analog of a tree [6], making it inherently suitable for capturing hierarchical levels among various data types. Since visual and textual concepts are inherently hierarchical, recent works [17,30,52,47,48] show that hyperbolic space serves as a promising manifold for preserving granularity levels in vision-language model representations, leading to strong performance across downstream tasks. Most existing methods directly reshape the original hierarchical structure to align different modalities. In contrast, our method preserves the original hierarchical structure and specifically leverages the intrinsic property of hyperbolic radius to enable visual representations to adapt their hierarchical level, aligning them with the language modality. This approach provides a complementary perspective to existing efforts in the MLLM community.
this section cite: ['b5', 'b16', 'b29', 'b51', 'b46', 'b47']

Section: Preliminary Concepts
Hyperbolic Geometry. In contrast to Euclidean or spherical geometries, hyperbolic geometry is characterized by a constant negative curvature, which fundamentally distinguishes its geometric properties and computational behaviors. Following prior studies [8,57], we utilize the classical Poincaré ball model-one of five principal analytic models for constructing hyperbolic space [7]-due to its demonstrated efficacy in representing hierarchical levels [23,19,45]. The Poincaré ball model (D n c , g Dc ), characterized by a radius of 1/ √ c and constant negative curvature -c (c > 0) is formally defined as follows:
D n c := {X ∈ R n : c∥X∥ < 1} g Dc := λ 2 c,X g E ,(1)
where λ c,X = 2 1-c∥X∥ 2 and g E = I n denotes the Euclidean metric tensor, serving as the foundation for the hyperbolic space construction.
this section cite: ['b7', 'b56', 'b6', 'b22', 'b18', 'b44']

Section: Hyperbolicity.
Building upon the theoretical framework of gyrovector spaces [62,63], we incorporate Möbius operations into hyperbolic space, specifically the Möbius addition operation "⊕ c " and Möbius multiplication operation "⊗ c ". In hyperbolic geometry, the tangent space T c X D n c at any point X ∈ D n c serves as a first-order approximation of D n c , representing an n-dimensional Euclidean space that locally approximates the hyperbolic structure. The tangent space T c X D n c and D n c are mapped to each other by exponential (T c X D n c → D n c : exp D,c X (•)) and logarithmic (D n c → T c X D n c : log D,c X (•)) maps, respectively. Detailed mathematical definitions and derivations are provided in the supplementary material.
this section cite: ['b61', 'b62']

Section: Methodology
This section first provides the necessary background on the conventional training paradigm from the perspective of parameter space tuning (Sec. 4.1) to contextualize our approach, followed by the detailed presentation of HyperET in Sec. 4.2. Then, Sec. 4.3 provides a theoretical analysis on adjusting the hyperbolic radius of visual representations.
this section cite: []

Section: Parameter Space Tuning
Parameter space tuning in multi-modal large language models (MLLMs), whether through full fine-tuning or parameter-efficient fine-tuning, aims to adapt pre-trained visual and language models to target multi-modal scenarios. However, these tuning methods, which rely solely on gradient updates, operate as constraint-free adjustments in Euclidean space. They often implicitly assume that visual representations can sufficiently adapt to the required granularity level (e.g., transitioning
Efficient Flexible Diagonal Block-Diagonal Banded Full ⊗ 𝒄 Number of Trainable Params Pretrained Weight Matrix Zero Elements Tunable Elements Scaling Matrix Hyperbolic Radius Adjustment Module HRA Input Output 𝑾 ! " 𝑾 ! #$" 𝑾 ! # 𝑾 ! 𝑾 % 𝔻 𝑾 % Euclidean Space: ℝ Hyperbolic Space: 𝔻 Logarithmic Map Exponential Map 𝑛 𝑛 𝑛 𝑛 ! ⟵ ⟶ ⊗ 𝒄 Möbius Multiplication HRA: Hyperbolic Radius Adjustment from image-level to pixel-level), which may lead to inefficiency in alignment at a certain granularity level. In contrast, the core technique of our proposed HyperET involves hyperbolic radius adjustment, which explicitly adjusts the granularity level of visual representations in MLLMs. HyperET provides a simple yet effective solution to granularity mismatch challenges.
this section cite: []

Section: General Hyperbolic Radius Adjustment
As previously discussed, hyperbolic radius adjustment provides a direct mechanism for optimizing the granularity level of visual representations, effectively bridging the granularity gap between visual and language modalities. In practice, we introduce a radius adjustment constraint into the weight update process, generally defined as follows:
Rad W D /Rad W D 0 = s ⇔ Rad W D = s • Rad W D 0 ,(2)
where Rad W D and Rad W D 0 represent the hyperbolic radii of W D and W D 0 , respectively, quantifying their granularity levels. The hyperbolic weight matrices W D and W D 0 are derived by projecting their Euclidean counterparts W and W 0 into hyperbolic space through exponential mapping operations defined in Sec. 3. The scaling coefficient s is task-adaptive, dynamically adjusting to optimize performance. However, this modification requires a two-step procedure: (1) computing the hyperbolic radius of W D and (2) subsequently adjusting it. To streamline this process, we propose a more efficient approach that directly optimizes W D without intermediate radius computations.
Hyperbolic Radius. Without loss of generality, for any point X ∈ D n c in hyperbolic space, its hyperbolic radius is formally defined as follows:
Rad X := d D c (X, 0) = ( 2 √ c )tanh -1 ( √ c∥X∥),(3)
where 0 denotes the origin point in hyperbolic space. In the subsequent analysis, we demonstrate that Mobiüs multiplication operations ⊗ c defined in Sec. 3 can enable precise control over the hyperbolic radius. This critical property is formally stated in the following theorems:
Theorem 1 (Hyperbolic Radius Scaling) For a point X ∈ D n c in hyperbolic space, the hyperbolic radius adjustment function is expanded as follows:
s • Rad X = 2 √ c (s √ c 2 Rad X ) = 2 √ c tanh -1 ( √ c∥s ⊗ c X∥) = Rad s⊗cX .(4)
where ⊗ c here is instantiated as a Mobiüs scalar multiplication operation. Therefore, hyperbolic radius adjustment can be precisely controlled through ⊗ c between s and X, with the scaling coefficient s serving as a primary learnable parameter.
this section cite: []

Section: ■
According to Theorem 1, ⊗ c can provide an equivalent mechanism during parameter space tuning. Consequently, hyperbolic radius adjustment in Eq. ( 2) can be easily achieved as follows:
W D = s ⊗ c W D 0 .(5)
Then, building upon the exponential mapping exp D,c 0 (•) and the logarithmic mapping log D,c 0 (•) defined in Sec. 3, we achieve general hyperbolic radius adjustment via the following reformulation of Eq. ( 5):
W = log D,c 0 (s ⊗ c exp D,c 0 (W 0 )),(6)
where s is a learnable parameter for adjustment. Given that a constrained parameter set-particularly when limited to a single learnable parameter-may ineffectively adjust hyperbolic radius during restricted training iterations, we propose a more flexible parameterization strategy through matrixbased formulations, termed flexible adjustment, which further enables precise control over hyperbolic radius optimization.
this section cite: []

Section: Flexible Adjustment for Hyperbolic Radius.
To achieve this, we adopt a scaling matrix W s to replace the scaling coefficient s, which is satisfied:
s = ∥W s W D 0 ∥/∥W D 0 ∥.
Consequently, the right-hand side of Eq. ( 2) can be reformulated as follows:
Rad W D = s • Rad W D 0 = ∥W s W D 0 ∥ ∥W D 0 ∥ • Rad W D 0 .(7)
In this framework, the learnable parameters transition from scalar values to matrix-based formulations, significantly enhancing flexibility within constrained training iterations. The following theorem demonstrates that hyperbolic radius can be dynamically scaled through Mobiüs matrix multiplication operations, a generalized instantiation of ⊗ c .
Theorem 2 (Hyperbolic Radius Flexibility Scaling) For a point X ∈ D n c in hyperbolic space and a scaling matrix X s ∈ R n×n , the flexible radius adjustment function is formally defined through Eq. ( 7) and Mobiüs matrix multiplication operations as follows:
∥X s X∥ ∥X∥ • Rad X = 2 √ c ( ∥X s X∥ ∥X∥ √ c 2 Rad X ) = 2 √ c tanh -1 ( √ c∥X s ⊗ c X∥) = Rad Xs⊗cX .(8)
Analogically, the scaling matrix W s is able to direct adjust the hyperbolic radius Rad X through Mobiüs matrix multiplication operations, providing precise control over representation granularity. ■ Building upon Theorem 2, we introduce a matrix-based formulation by replacing the scaling coefficient s with W s , resulting in the following reformulation of Eq. ( 6):
W = log D,c 0 (W s ⊗ c exp D,c 0 (W 0 )).(9)
With up to O(n 2 ) learnable elements, W s in Eq. ( 9) offers significantly enhanced flexibility for hyperbolic radius adjustment. Notably, Eqs. ( 5) and ( 9) become equivalent when W s is constrained to a diagonal matrix with uniform scaling factors, i.e., W s = diag(ω 1 , ω 2 , • • • , ω n ) ∈ R n×n and ω i = ω j , i ̸ = j. Moreover, in alignment with the current paradigm of parameter-efficient fine-tuning, we also introduce a parameter-efficient variant of W s , designed to maintain flexibility while reducing computational overhead.
this section cite: []

Section: Efficient Adjustment for Hyperbolic Radius.
To realize this strategy, we define a diagonal scaling matrix W D s = diag(ω 1 , ω 2 , . . . , ω n ) ∈ R n×n , where each ω i is a learnable scalar and ω i ̸ = ω j when i ̸ = j, significantly reducing the number of learnable parameters. In this form, only the diagonal elements are learnable parameters, making the diagonal scaling matrix W D s the most parameterefficient configuration for hyperbolic radius adjustment. Building upon this foundation, we extend W D s into two more flexible variants by incrementally introducing learnable off-diagonal elements, balancing enhanced adjustment capability with parameter efficiency: (1) Block-diagonal scaling matrix W B-D s = diag(R 1 , R 2 , . . . , R r ), where R i ∈ R n r × n r . Here, r is the block size (we assume n is divisible by r). This formulation allows intra-block interactions while maintaining sparsity across blocks; (2) Banded scaling matrix
W B s = ω11 ••• ω1n . . . . . . . . . ωn1 ••• ωnn ∈ R n×n
, where ω ij = 0 for all i, j such that |i -j| > d. Here, d denotes the bandwidth, indicating that nonzero elements are allowed within d entries above and below the main diagonal. These two variants provide mechanisms to capture localized interactions while preserving parameter efficiency. Notably, both W B-D s and W B s degenerate to W D s under specific configurations: when r = n for the block-diagonal matrix and d = 0 for the banded matrix. This highlights the inherent hierarchical flexibility enabled by our parametrization strategy and reflects a insightful way of designing fine-tuning methods.
To summarize these fine-tuning strategies, Fig. 1 provides a unified illustration of different variants of the scaling matrix W s . Our introduced hyperbolic radius adjustment, i.e., HRA, adjusts visual representations by applying learnable scaling matrices to the frozen pre-trained weights W 0 in hyperbolic space, enabling precise adjustment over arbitrary granularity levels.
this section cite: []

Section: Theoretical Analysis
To demonstrate the impact of hyperbolic radius adjustment, we provide a theoretical analysis of the Möbius multiplication operation. The following deduction shows that the proposed Möbius multiplication operation can directly adjust the granularity level of visual representations.
Deduction. For Y 0 = W 0 X, where X ∈ R d×k is the input embedding, W 0 is the pre-trained weight and Y 0 is the pre-trained visual representation, the forward pass of HyperET is as follows:
Y 0 = WX = log D,c 0 (W s ⊗ c exp D,c 0 (W 0 ))X.(10)
Then, according to the definition of hyperbolic space, Y 0 is firstly projected into the hyperbolic space D n c and our HyperET is applied to adjust the hyperbolic radius, resulting in the new visual representation Y, expressed as:
Y D n c 0 = exp D,c 0 (Y 0 ) = W D n c 0 X D n c (11
)
Y D n c = exp D,c 0 (Y) = W s ⊗ c W D n c 0 X D n c , (12
)
where
W D n c 0 = exp D,c 0 (W 0 ) and X D n c = exp D,c 0 (X).
According to the Theorem 2, the hyperbolic radius of Y D n c is obtained as:
Rad Y = Rad Ws⊗cW D n c 0 X D n c = ∥W s W D n c 0 X D n c ∥ ∥W D n c 0 X D n c ∥ • Rad W D n c 0 X D n c = ∥W s W D n c 0 X D n c ∥ ∥W D n c 0 X D n c ∥ Rad Y0 = s • Rad Y0 , (13
)
where s is a scaling coefficient. Therefore, our hyperbolic radius adjustment method can directly adjust the hyperbolic radius of visual representations, thus is able to bridge the granularity gap between visual and textual modalities at an arbitrary granularity level.
this section cite: []

Section: Experiments
Our experimental evaluation encompasses two MLLM scenarios, (1) MLLM's fine-tuning (Sec. 5.1), and (2) MLLM's pre-training (Sec. 5.2). A detailed ablation study is presented in Sec. 5.3 and 5.4.
this section cite: []

Section: MLLM's Fine-tuning
Experimental Setting. We evaluate our method on ScienceQA [42], a challenging large-scale VQA benchmark encompassing diverse scientific domains. Our comparative analysis includes MemVP and other LLaMA-based models with input-space visual prompting: LLaVA [40], and LaVIN [43]. We follow the experiment setting in [43]. All models utilize a CLIP pre-trained ViT-L/14 visual encoder. The weights of HyperET in this task are implemented using the three parameter-efficient scaling matrices, i.e., W D s , W B-D s and W B s , and are adapted in the attention layer, consistent with most parameter-efficient tuning methods, e.g., LoRA [24]. The curvature c is 0.01. All experiments are conducted using a maximum of 8 NVIDIA H800 GPUs.
Comparing to SOTA. Our experimental evaluation compares the proposed approach with state-ofthe-art parameter-efficient fine-tuning (PEFT) methods, including LaVIN [43] and MemVP [26]. As demonstrated in Table 4, which presents both baseline and HyperET enhanced results, our method establishes new state-of-the-art performance. HyperET achieves this breakthrough with minimal parameter overhead (fewer than 1%), delivering substantial improvements to both LaVIN and MemVP frameworks. Particularly noteworthy are the gains observed with LLaMA-13B as the backbone language model, where HyperET enhances average cross-domain accuracy by 0.96% for LaVIN and 0.94% for MemVP. Notably, when integrated with MemVP using LLaMA-7B, HyperET achieves performance on par with the more computationally intensive MemVP-LLaMA-13B configuration, i.e., 93.78. This result demonstrates that HyperET enhances visual representation and achieves comparable performance improvements while utilizing 100, 000 × fewer parameters (0.05M vs 6B) than MemVP using LLaMA-13B, highlighting its remarkable parameter efficiency.
this section cite: ['b41', 'b39', 'b42', 'b42', 'b23', 'b42', 'b25']

Section: MLLM's Pre-training
Experimental Setting. Our pre-training evaluation framework builds upon LLaVA-1.5 [39], employing identical datasets to evaluate HyperET's effectiveness in MLLM pre-training. Our comparative analysis includes LLaVA-1.5 [39] and LLaVA-Next [39] and train and fine-tune our HyperET with the same experiment setting. The weights of HyperET in this task are implemented using the highest flexible scaling matrices, i.e., W s , and are adapted in the attention layer, consistent with most parameter-efficient training methods, e.g., LoRA [24].
Comparing to SOTA. Our experimental framework evaluates the proposed method against stateof-the-art pre-trained MLLMs across 12 standard visual language benchmarks. We implement HyperET using the most flexible matrix configuration, i.e., W s , maximizing the model's adaptability. While this approach resembles full fine-tuning in structure, the actual parameter count remains remarkably efficient at approximately 50M-less than 1% of the language model's 13B parameters-maintaining parameter efficiency. As shown in Table 2, our approach demonstrates significant performance improvements over LLaVA-1.5 [39], particularly in mitigating the limitations of CLIP-based encoders. Notably, on the POPE benchmark [37] for object hallucination detection, HyperET substantially reduces visual hallucinations, providing empirical evidence that hyperbolic radius optimization effectively enhances the cross-modal alignment at an arbitrary level.
Table 2: Comparison with SoTA pre-trained methods on 12 MLLM benchmarks, including VQAv2 [21], GQA [25], VW: VisWiZ [22], SQA: ScienceQA-IMG [42], TVQA: TextVQA [53], PE: POPE [37], ME: MME [67], MB: MMBench [41], MB CN : MMBench-Chinese [41], SD: SEED-Bench [34], LVA W : LLaVA-Bench (In-the-Wild) [40] and M-Vet [68]. Top-1 accuracy is reported (Best in bold, second best is underlined). Lan. Model: Language model. Benchmark names are abbreviated due to space limits. "Ours": we here realize the extra learnable parameters as full matrices, i.e., W s . Vision encoder: CLIP.
Method Lan. Model VQAv2 GQA VW SQA TVQA PE ME MB MB CN SD LVA W M-Vet LLaVA-1.5 Vicuna-7B 78.5 62.0 50.0 66.8 58.2 85.9 1510.7 64.3 58.3 58.6 63.4 30.5 LLaVA-1.5+Ours Vicuna-7B 80.3 63.7 51.9 69.1 60.8 87.7 1536.2 66.8 60.5 60.2 65.6 32.4 LLaVA-1.5 Vicuna-13B 80.0 63.3 53.6 71.6 61.3 85.9 1531.3 67.7 63.6 61.6 70.7 35.4 LLaVA-1.5+Ours Vicuna-13B 82.3 65 .7 55.2 73.7 63.9 88.7 1584.7 69.8 65.2 63.4 72.6 38.3LLaVA-Next Vicuna-7B 81.8 64.2 57.6 70.1 64.9 86.5 1519 67.4 60.6 70.2 81.6 43.9 LLaVA-Next+Ours Vicuna-7B 82.9 65.4 58.9 70.8 65.1 88.9 1551 69.9 62.5 71.0 82.9 44. 8 Table 3: Comparative analysis of fine-tuning spaces and flexibility levels on ScienceQA test set [42]. All experiments utilize MemVP [26] with LLaMA-13B as the backbone language model. The notation is defined as follows:
this section cite: ['b38', 'b38', 'b38', 'b23', 'b38', 'b36', 'b20', 'b24', 'b21', 'b41', 'b52', 'b36', 'b66', 'b40', 'b40', 'b33', 'b39', 'b67', 'b41', 'b25']

Section: Ablation Study on Fine-tuning
Here, we do an ablation study on the ScienceQA [42], employing MemVP [26] as the backbone.
this section cite: ['b41', 'b25']

Section: Introducing Different Flexibility into HyperET.
We systematically investigate our proposed three distinct parameterization strategies for HyperET 's learnable components. These strategies include diagonal scaling matrices W D s and their two extended variants: banded scaling matrices W B s and block-diagonal scaling matrices W B-D s , which systematically increase parameter flexibility through progressively relaxed structural constraints. As evidenced in Table 3, enhanced flexibility-achieved through W B s fine-tuning-provides only marginal performance improvements (0.17% accuracy gain over W D s ) while introducing over-fitting risks when further increasing learnable parameters via expanded banded sizes. These findings demonstrate that our fine-tuning method effectively accomplishes two key objectives: (1) efficient adaptation to optimal hyperbolic radii and (2) robust cross-modal alignment for downstream tasks, all while maintaining parameter efficiency.
this section cite: []

Section: Effectiveness of Training in Hyperbolic Space.
To isolate the performance gains attributable to hyperbolic space rather than parameter increases, we conduct controlled experiments by adding an equivalent number of parameters through matrix multiplication in Euclidean space, as shown in Table 3. The results demonstrate that such parameter augmentation either yields no significant performance improvement or even degrades model performance. This empirical evidence strongly validates the necessity of employing hyperbolic space for adjusting vision encoder in MLLMs, as the observed improvements in Table 3 cannot be explained by mere parameter increases.
this section cite: []

Section: Necessity of Mobiüs matrix multiplication.
We conduct ablation studies to evaluate the impact of the Möbius matrix multiplication operation. As shown in Table 3, a comparison between rows 6 and 7 reveals that standard matrix multiplication (denoted by" ✗") underperforms 0.81% compared to the Möbius matrix multiplication, (represented by "✓"). This performance gap underscores the essential role of Möbius operations in precisely adjusting the hyperbolic radius of visual representations.
this section cite: []

Section: Discussion of Different Vision Encoder.
To further clearly demonstrate the primary contributing factor behind the performance improvements, we present additional experimental results designed to isolate and eliminate the influence of merely increasing the number of trainable parameters. As shown in Table 4, fine-tuning SAM [29] and DINOv2 [46] in Euclidean space with the same number of additional parameters as HyperET yields only marginal gains. In contrast, fine-tuning with HyperET results in substantial performance improvements. These findings clearly illustrate that our main contribution arises specifically from the proposed hyperbolic radius adjustment mechanism, rather than simply from introducing more learnable parameters. To further assess the contribution of each key component in the proposed HyperET, we isolate each component in separate experiments, and the results are presented in Table 5. Comparing row 2 and row 3, training in Euclidean space instead of hyperbolic space leads to significantly lower performance, indicating that the granularity gap mismatch cannot be effectively addressed in Euclidean space. The observed improvement over the baseline is primarily attributable to the increased number of parameters rather than the alignment of granularity levels. Intriguingly, comparing row 2 and row 4, simply transitioning from Euclidean space to hyperbolic space yields a slight improvement, which we attribute to the inherent ability of hyperbolic space to capture granularity levels. However, this strategy lacks explicit mechanisms for hyperbolic radius adjustment, limiting its effectiveness in addressing granularity mismatches. Finally, by adjusting the hyperbolic radius through Möbius matrix multiplication and integrating it with learnable matrices W s , the results show significant performance improvements, e.g., a 2.1% gain on the TextVQA [53], demonstrating the effectiveness of HyperET. Additionally, we visualize the changes in the hyperbolic radius in Fig. 2 after training, and the observed change in the hyperbolic radius indicates the MLLM's requirement for multi-granularity levels, demonstrating the necessity of our proposed HyperET.
this section cite: ['b28', 'b45', 'b52']

Section: Ablation Study on Pre-training

this section cite: []

Section: Conclusion
This work proposes an efficient training paradigm (HyperET) for multi-modal large language models in hyperbolic space. By dynamically adjusting the hyperbolic radius of visual representations through learnable matrices and Möbius multiplication operations, HyperET effectively bridges the granularity gap between visual and textual modalities at an arbitrary granularity level. Our experiments across multiple MLLM benchmarks demonstrate that HyperET consistently improve existing pre-training and fine-tuning baselines by large margins with less than 1% additional parameters.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [NA] Justification: Those are not discussed in the paper.
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

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: All assumptions are proved clearly in the main paper or the supplemental material.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: Detailed experimental settings and information are provided in Sec. 4 and Sec. 5, which are sufficient to reproduce the results reported in this paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [No] Justification: The code will be made public after acceptance.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: All the experimental settings are clarified in Sec. 5 and supplementary materials.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [No] Justification: The paper does not report error bars or statistical significance measures mainly because the experiments were conducted in a stable and controlled environment, where repeated trials showed minimal variability. Additionally, due to computational resources and time constraints, extensive repeated runs were not feasible. Moreover, omitting such measures aligns with common practice in this research area.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources

this section cite: []

Section: NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
• You should answer [Yes] , [No] , or [NA] .
• [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• Please provide a short (1-2 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
IMPORTANT, please:
• Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist",
• Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: The abstract clearly states the paper's main contributions and scope. We propose an efficient training paradigm for MLLMs, dubbed as HyperET, which can optimize visual representations to align with their textual counterparts at an arbitrary granularity level through dynamic hyperbolic radius adjustment in hyperbolic space. Our method consistently improves both existing pre-training and fine-tuning MLLMs by large margins with less than 1% additional parameters.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: Computing resource requirements are detailed in Sec. 5. Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We ensure our research adheres to the guidelines. Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: There is no societal impact of the work performed.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes]
Justification: The paper properly credits all original papers, states their versions, and respects the copyright and terms of use.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Flamingo: a visual language model for few-shot learning Year: (2022)
Ref_id:b2 Title: Palm 2 technical report Year: (2023)
Ref_id:b3 Title: Qwen-vl: A frontier large vision-language model with versatile abilities Year: (2023)
Ref_id:b4 Title: Qwen2.5-vl technical report Year: (2025)
Ref_id:b5 Title: Metric spaces of non-positive curvature Year: (2013)
Ref_id:b6 Title: Hyperbolic geometry. Flavors of geometry Year: (1997)
Ref_id:b7 Title: Learning attentive and hierarchical representations for 3d shape recognition Year: (2020)
Ref_id:b8 Title: Shikra: Unleashing multimodal llm's referential dialogue magic Year: (2023)
Ref_id:b9 Title: A jointly-scaled multilingual language-image model Year: (2022)
Ref_id:b10 Title: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling Year: (2024)
Ref_id:b11 Title: How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites Year: (2024)
Ref_id:b12 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: ()
Ref_id:b13 Title: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality Year: (2023-04)
Ref_id:b14 Title: Scaling instruction-finetuned language models Year: (2024)
Ref_id:b15 Title: Instructblip: Towards general-purpose vision-language models with instruction tuning Year: (2023)
Ref_id:b16 Title: Hyperbolic image-text representations Year: (2023)
Ref_id:b17 Title: Exploring the limits of masked visual representation learning at scale Year: (2023)
Ref_id:b18 Title: Hyperbolic neural networks Year: (2018)
Ref_id:b19 Title: Hyperbolic contrastive learning for visual representations beyond objects Year: (2023)
Ref_id:b20 Title: Making the v in vqa matter: Elevating the role of image understanding in visual question answering Year: (2017)
Ref_id:b21 Title: Vizwiz grand challenge: Answering visual questions from blind people Year: (2018)
Ref_id:b22 Title: Rethinking generalizable face anti-spoofing via hierarchical prototype-guided distribution refinement in hyperbolic space Year: (2024)
Ref_id:b23 Title: Low-rank adaptation of large language models Year: (2021)
Ref_id:b24 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b25 Title: Memory-space visual prompting for efficient vision-language fine-tuning Year: (2024)
Ref_id:b26 Title: Brave: Broadening the visual encoding of vision-language models Year: (2024)
Ref_id:b27 Title: Hyperbolic image embeddings Year: (2020)
Ref_id:b28 Title: Segment anything Year: (2023)
Ref_id:b29 Title: Hyperbolic learning with synthetic captions for open-world detection Year: (2024)
Ref_id:b30 Title: Reasoning segmentation via large language model Year: (2024)
Ref_id:b31 Title: Llava-next: Stronger llms supercharge multimodal capabilities in the wild Year: (2024)
Ref_id:b32 Title: Mimic-it: Multi-modal in-context instruction tuning Year: (2023)
Ref_id:b33 Title: Seedbench: Benchmarking multimodal llms with generative comprehension Year: (2023)
Ref_id:b34 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b35 Title: Videochat: Chat-centric video understanding Year: (2023)
Ref_id:b36 Title: Evaluating object hallucination in large vision-language models Year: (2023)
Ref_id:b37 Title: Monkey: Image resolution and text label are important things for large multi-modal models Year: (2024)
Ref_id:b38 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b39 Title: Visual instruction tuning Year: (2023)
Ref_id:b40 Title: Is your multi-modal model an all-around player? arXiv preprint Year: (2023)
Ref_id:b41 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b42 Title: Cheap and quick: Efficient vision-language instruction tuning for large language models Year: (2023)
Ref_id:b43 Title: Mike Zheng Shou, and Xiaoyan Sun. Visual perception by large language model's weights Year: (2024)
Ref_id:b44 Title: Learning continuous hierarchies in the lorentz model of hyperbolic geometry Year: (2018)
Ref_id:b45 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b46 Title: Compositional entailment learning for hyperbolic vision-language models Year: (2024)
Ref_id:b47 Title: Understanding fine-tuning clip for open-vocabulary semantic segmentation in hyperbolic space Year: (2025)
Ref_id:b48 Title: Kosmos-2: Grounding multimodal large language models to the world Year: (2023)
Ref_id:b49 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b50 Title: Vision language models are blind Year: (2024)
Ref_id:b51 Title: Accept the modality gap: An exploration in the hyperbolic space Year: (2024)
Ref_id:b52 Title: Towards vqa models that can read Year: (2019)
Ref_id:b53 Title: Eva-clip: Improved training techniques for clip at scale Year: (2023)
Ref_id:b54 Title: Generative pretraining in multimodality Year: (2023)
Ref_id:b55 Title: Vl-adapter: Parameter-efficient transfer learning for vision-and-language tasks Year: (2022)
Ref_id:b56 Title: Learning the predictability of the future Year: (2021)
Ref_id:b57 Title: Rethinking optimization and architecture for tiny language models Year: (2024)
Ref_id:b58 Title: Eyes wide shut? exploring the visual shortcomings of multimodal llms Year: (2024)
Ref_id:b59 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b60 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b61 Title: Hyperbolic trigonometry and its application in the poincaré ball model of hyperbolic geometry Year: (2001)
Ref_id:b62 Title: A gyrovector space approach to hyperbolic geometry Year: (2008)
Ref_id:b63 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b64 Title: Large language model is also an open-ended decoder for vision-centric tasks Year: (2023)
Ref_id:b65 Title: Diffusion feedback helps clip see better Year: (2024)
Ref_id:b66 Title: A survey on multimodal large language models Year: (2024)
Ref_id:b67 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2023)
Ref_id:b68 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b69 Title: Video-llama: An instruction-tuned audio-visual language model for video understanding Year: (2023)
Ref_id:b70 Title: Llama-adapter: Efficient fine-tuning of language models with zero-init attention Year: (2023)
Ref_id:b71 Title: Cobra: Extending mamba to multi-modal large language model for efficient inference Year: (2025)
Ref_id:b72 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
