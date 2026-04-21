Title: Hyperbolic Fine-Tuning for Large Language Models
Abstract: Large language models (LLMs) have demonstrated remarkable performance across various tasks. However, it remains an open question whether the default Euclidean space is the most suitable choice for LLMs. In this study, we investigate the geometric characteristics of LLMs, focusing specifically on tokens and their embeddings. Our findings reveal that token frequency follows a power-law distribution, where high-frequency tokens (e.g., "the," "that") constitute the minority, while low-frequency tokens (e.g., "apple," "dog") constitute the majority. Furthermore, high-frequency tokens cluster near the origin, whereas low-frequency tokens are positioned farther away in the embedding space. Additionally, token embeddings exhibit hyperbolic characteristics, indicating a latent tree-like structure within the embedding space. Motivated by these observations, we propose HypLoRA, an efficient fine-tuning approach that operates in hyperbolic space to exploit these underlying hierarchical structures better. HypLoRA performs low-rank adaptation directly in hyperbolic space, thereby preserving hyperbolic modeling capabilities throughout the fine-tuning process. Extensive experiments across various base models and reasoning benchmarks, specifically arithmetic and commonsense reasoning tasks, demonstrate that HypLoRA substantially improves LLM performance.Recent advancements suggest that non-Euclidean geometries, particularly hyperbolic spaces [11,14,15,16,17,18,19,20,21,22,23], offer promising alternatives for modeling hierarchical data. Hyperbolic space, distinguished by its negative curvature, is especially well-suited for representing tree-like hierarchical data due to its exponential volume growth and geometric prior. This geometric property makes hyperbolic space particularly capable for tasks involving complex, hierarchically structured information.

Section: Introduction
Large language models (LLMs) such as GPT-4 [1], LLaMA [2], Gemma [3], and Qwen [4] have demonstrated remarkable capabilities in understanding and generating human-like text [5,6,7]. Despite their impressive capabilities, these models often rely on Euclidean geometry for token representation, which may inadequately capture the inherently complex and hierarchical nature of real-world data structures [8,9,10,11,12,13]. Consider how words naturally organize into nested categories with varying levels of abstraction: abstract concepts like "fruit" occupy higher positions in the semantic hierarchy, while specific instances such as "apple" or "banana" populate the lower levels. Representing such structures effectively is crucial for understanding the semantics of language in LLMs.
10 0 10 1 10foot_0 10foot_1 Token Frequency (log scale) 10 0 10 1 10 2 10 3 Count (log scale) GSM8K = 1.87 0.4 0.6 Token Norm 10 0 10 1 10 2 10 3 Token Frequency (log scale) GSM8K 10 0 10 1 10 2 10 3 Token Frequency (log scale) 10 0 10 1 10 2 10 3 Count (log scale) AQuA = 1.89 0.3 0.4 0.5 0.6 0.7 Token Norm 10 0 10 1 10 2 10 3
Token Frequency (log scale) AQuA Figure 1: Token frequency distribution and token frequency vs. norm analysis for GSM8K (Group 1) and AQuA (Group 2) datasets in LLaMA3-8B. For each group, the left panels show the token frequency distributions (power-law distribution), while the right panels illustrate the relationship between token frequency and the corresponding norms. This visualization reveals the underlying geometric structure of the token embeddings. For additional data analysis and visualizations, please refer to Appendix A.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12']

Section: Proposed Analysis Framework.
In this work, we first delve into how LLMs interact with token embeddings and explore the extent to which these embeddings exhibit non-Euclidean characteristics. We approach this from both a global and local perspective. At the global level, we analyze the overall distribution of tokens by frequency and investigate how these frequencies are arranged across the embedding space. At the local level, we measure the hyperbolicity of the metric space spanned by each input prompt, where the hyperbolicity serves as a proxy for evaluating the distance or dissimilarity between the underlying embedding structure and a tree-like hierarchy [24,25,16].
Our analysis in Section 4 reveals several key insights. Globally, token frequency follows a powerlaw distribution, where high-frequency tokens (e.g., "the," "that") constitute the minority, while lowfrequency tokens (e.g., "apple," "dog") constitute the majority. Power-law distributions are consistent with, and can naturally arise from, underlying hierarchical or branching generative mechanisms [12,26,27]. 2 Besides, high-frequency tokens (e.g., abstract concepts, function words) tend to be located near the origin of the embedding space, while low-frequency tokens (e.g., specific terms) are farther away, as demonstrated in Table 1. Locally, our investigation of hyperbolicity (δ values) in Table 2 demonstrates that LLM token embeddings in each prompt exhibit significant tree-like properties.
Based on our findings above, a natural consideration is to develop hyperbolic LLMs that explicitly incorporate a hyperbolic inductive bias 3 . However, training LLMs from scratch is resourceintensive [29,30,31]. As a more resource-efficient alternative, we propose to build the first low-rank adaptation fine-tuning method in hyperbolic space. This approach is particularly advantageous given that existing LLMs are all Euclidean, and not all downstream tasks require hyperbolic geometry in their fine-tuning. By employing hyperbolic adapters on Euclidean LLMs for specific tasks, we can leverage the benefits of both geometries while maintaining computational efficiency.
Challenges. Adapting LLMs in non-Euclidean embedding spaces with classic techniques, i.e., applying exponential and logarithmic maps within tangent space [32,33,34,35,36] for weight adaptation is problematic in this case. This approach fails to fully capture the hyperbolic geometry, as the exponential and logarithmic maps are mutually inverse and can be canceled with consecutive operations 4 . Consequently, the inherent properties of the hyperbolic space are not effectively preserved, limiting the potential benefits of incorporating non-Euclidean geometries into the adaptation process.
Proposed Method. To address this limitation, we introduce HypLoRA to perform low-rank adaptation directly on the hyperbolic manifold without transformation to the tangent space, thus preserving hyperbolic modeling capabilities and counteracting the cancellation effect. HypLoRA integrates hyperbolic geometry into existing LLMs, implicitly introducing high-order interactions and accounting for token hierarchies, enabling them to benefit from hyperbolic characteristics while minimizing additional computational costs.
To summarize, our main contributions are twofold: (1) We conduct a comprehensive investigation into the geometric characteristics of token embeddings in LLMs, revealing their inherent tree-like structure and strong hyperbolic properties. (2) We propose HypLoRA, a parameter-efficient fine-tuning method that integrates hyperbolic geometry into LLMs while keeping it aligned with the Euclidean LLM framework. We conduct extensive experiments on various models and different tasks, specifically arithmetic reasoning and commonsense reasoning, demonstrating clear advantages over competitive baselines.
2 Related Work Hyperbolic Representation Learning and Foundation Models. Hyperbolic geometry has been successfully applied to various neural network architectures and models [19,21,18], including shallow hyperbolic neural networks [15,33,37,38,39,40], hyperbolic CNNs [41,42,43], hyperbolic GNNs [32,44,45,46], and hyperbolic attention networks or Transformers [47,37,38,48]. These models leverage the inductive biases of hyperbolic geometry to achieve remarkable performance on various tasks and applications [32,22,23,49,16,17,50,51,52,53]. Recent efforts have focused on adapting LLMs and CLIP [54] to hyperbolic spaces. Key advancements include developing more expressive hyperbolic image-text representations [55], enabling compositional entailment learning for deeper vision-language understanding [56], designing safety-aware hyperbolic frameworks for content moderation [57], and creating core modules to facilitate the construction of novel hyperbolic foundation models [58]. While these adaptations show promise, training LLMs from scratch remains computationally expensive [59,60]. The computational complexity increases further when considering Riemannian optimization [59,60,61] and additional hyperbolic operations, like Möbius addition.
this section cite: ['b23', 'b24', 'b15', 'b11', 'b25', 'b26', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b0', 'b1', 'b18', 'b20', 'b17', 'b14', 'b32', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b31', 'b43', 'b44', 'b45', 'b46', 'b36', 'b37', 'b47', 'b31', 'b21', 'b22', 'b48', 'b15', 'b16', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b55', 'b56', 'b57', 'b58', 'b59', 'b58', 'b59', 'b60']

Section: Geometric Analysis of Language Model Embeddings.
Prior work has made important observations about the geometry of embeddings that helped shape and motivate our research. Reif et al. [62] demonstrated that BERT embeddings contain distinct syntactic and semantic subspaces and showed evidence of tree-like parse structures, while Gao et al. [63] revealed that token embeddings tend to cluster in a narrow cone during training, leading to representation degeneration. Building on these geometric insights, Rudman et al. [64] introduced IsoScore to formally quantify how uniformly embeddings utilize the ambient vector space. Additionally, Puccetti et al. [65] analyzed outlier dimensions in Transformers and showed their correlation with token frequencies. While these works provide crucial foundations for understanding embedding geometry, our work differs in that we specifically quantify and leverage the natural hyperbolicity of token embeddings.
Parameter-Efficient Fine-Tuning (PEFT) and LoRA. Fine-tuning LLMs [66,1,2] for downstream tasks poses significant challenges due to their massive number of parameters. To address this issue, PEFT methods have been proposed, which aim to train a small subset of parameters while achieving comparable or even better performance compared to full fine-tuning. PEFT methods can be broadly categorized into prompt-based methods [67,68,69], adapter-based methods [70,71], and reparameterization-based methods [31,72,73]. Among these, the reparameterization-based LoRA [31] has gained significant attention due to its simplicity, effectiveness, and compatibility with existing model architectures. Variants of LoRA, such as LoRA+ [74], DoRA [75], and AdaLoRA [76], have been proposed to improve its performance and efficiency. Recent research has also investigated ensembles of multiple LoRAs [77,78] and quantization techniques [79,80,81]. The proposed method is a foundational algorithm that is orthogonal to existing approaches and can potentially be combined with various LoRA variants to exploit their complementary strengths and achieve superior performance.
this section cite: ['b61', 'b62', 'b63', 'b64', 'b65', 'b0', 'b1', 'b66', 'b67', 'b68', 'b69', 'b70', 'b30', 'b71', 'b72', 'b30', 'b73', 'b74', 'b75', 'b76', 'b77', 'b78', 'b79', 'b80']

Section: Preliminary
This section introduces the key concepts used in our study, including the Lorentz model of hyperbolic geometry and the LoRA adapter.
this section cite: []

Section: Hyperbolic Geometry.
Unlike flat Euclidean geometry, hyperbolic geometry is characterized by a constant negative curvature. We utilize the Lorentz model, also known as the hyperboloid model due to its ability to effectively capture hierarchical structures and maintain numerical stability [14,37,82]. The Lorentz model in n dimensions with curvature -1/K(K > 0) is defined as:
L n K = {x ∈ R n+1 : ⟨x, x⟩ L = -K, x 0 > 0},(1)
where ⟨•, •⟩ L is the Lorentzian inner product, given by: ⟨x, y⟩ L = -x 0 y 0 + n i=1 x i y i . Tangent Space. In the Lorentz model L n K , the tangent space at a point x is denoted T x L n K . It is defined as the set of all vectors u that are orthogonal to x under the Lorentzian inner product:
T x L n K := {u ∈ R n+1 : ⟨u, x⟩ L = 0}.(2)
To facilitate projection between the hyperboloid and its tangent spaces at x, one can utilize two critical mappings: the exponential and logarithmic maps. The exponential map at x, denoted exp K x , projects a vector from the tangent space T x L n K back onto the hyperboloid. Conversely, the logarithmic map, denoted log K
x , maps a point on the hyperboloid to the tangent space at x. The detailed formulas are given in Appendix C.
this section cite: ['b13', 'b36', 'b81']

Section: LoRA Adapter.
The LoRA adapter provides an efficient approach for modifying LLMs with minimal computational overhead. Instead of retraining the entire model, LoRA focuses on adjusting specific components within the model's architecture to transform an input x ∈ R d into an output z ∈ R k . In practice, LoRA targets the weight matrices found in each Transformer layer of an LLM. Typically, the weight W of the Transformer, which resides in the dimensions R k×d , is adapted through a low-rank approximation. This is achieved by introducing an additional term, ∆W , to the original weight matrix:
z = W LoRA (x) = W x + ∆W x = W x + BAx.(3)
Here, A ∈ R r×d and B ∈ R k×r represent two smaller, learnable matrices where r is the rank of these matrices, which is significantly less than either d or k. This design choice ensures that r ≪ min(d, k), thereby reducing the complexity of the model adaptation. During the fine-tuning process, only the matrices A and B are adjusted, while the pre-existing weights W are kept frozen.
This method significantly decreases the number of parameters that need to be trained, from d • k to (d + k) • r, enhancing the efficiency of the fine-tuning process. As a result, LoRA enables the targeted adaptation of LLMs, allowing them to transform an input x into an output z while maintaining high performance and adapting to new tasks or datasets with a fraction of the computational resources typically required.
this section cite: []

Section: Investigation
In this section, we present an in-depth investigation of token embeddings in LLMs from both global and local perspectives. Our goal is to uncover the geometric structures underlying pretrained token representations, specifically examining the global distribution of token frequencies and their spatial arrangement, as well as the local hyperbolicity of token embeddings across various datasets.
this section cite: []

Section: Global Token Statistics
We begin by investigating the global distribution of token frequencies in the context of arithmetic reasoning datasets, focusing on datasets such as GSM8K [83], AQuA [84], MAWPS [85], and SVAMP [86]. We also provide a broader analysis across different types of datasets and LLMs in Appendix A. Figure 1 (left) presents the distribution of token frequencies, with a power-law exponent γ ≈ 1.9, as estimated by the powerlaw package [87]. In such distributions, the exponent γ controls how quickly token frequencies decline: smaller values of γ (closer to 1) indicate a more gradual decay where frequent tokens dominate, while larger values signify a sharper decline, with most tokens being rare.
This power-law behavior is consistent with the tree-like hierarchical nature of language [11,16,88,52,23]. High-frequency tokens often correspond to more abstract or general concepts, while lowfrequency tokens represent specific or rare terms. This pattern aligns with a hierarchical organization of the token space: abstract, high-frequency tokens cluster near the origin, while specific terms are positioned farther out, mirroring how general concepts sit at the core of a semantic hierarchy with specialized terms at the periphery.
Table 1: Mean, Minimum, and Maximum frequency and norm values of token embedding in different base models and groups. Group 1: to, in, have, that, and, is, for, Group 2: how, much, many, time, cost, Group 3: animal, fruit, number, color, size, Group 4: dog, cow, apple, banana, 380, 480, purple, red, medium, small, large.
Model Group Frequency (Mean [Min∼Max]) Norm (Mean [Min∼Max]) Gemma-7B Group 1 4934.4 [1838 ∼ 8539] 3.160 [3.060 ∼ 3.299] Group 2 2709.4 [474 ∼ 6681] 3.561 [3.488 ∼ 3.627] Group 3 292.0 [34 ∼ 1191] 3.765 [3.623 ∼ 3.887] Group 4 114.3 [25 ∼ 284] 3.998 [3.660 ∼ 4.520] LLaMA-7B Group 1 4993.9 [1838 ∼ 8547] 0.951 [0.793 ∼ 1.060] Group 2 2712.6 [474 ∼ 6683] 1.222 [1.118 ∼ 1.299] Group 3 299.8 [34 ∼ 1200] 1.325 [1.274 ∼ 1.428] Group 4 139.1 [26 ∼ 286] 1.364 [1.326 ∼ 1.417] LLaMA3-8B Group 1 4937.4 [1838 ∼ 8547] 0.353 [0.330 ∼ 0.396] Group 2 2710.0 [474 ∼ 6683] 0.456 [0.394 ∼ 0.499] Group 3 292.6 [34 ∼ 1191] 0.499 [0.452 ∼ 0.549] Group 4 97.1 [13 ∼ 284] 0.569 [0.499 ∼ 0.675] LLaMA-13B Group 1 4993.9 [1838 ∼ 8547] 1.027 [0.833 ∼ 1.255] Group 2 2712.6 [474 ∼ 6683] 1.429 [1.346 ∼ 1.489] Group 3 299.8 [34 ∼ 1200] 1.494 [1.453 ∼ 1.532] Group 4 139.1 [26 ∼ 286] 1.501 [1.470 ∼ 1.526]
Empirical Observation. To better understand the relationship between token frequency and their spatial arrangement within the embedding space, we calculate the average token frequency as a function of their distance from the origin. As shown in Figure 1 (right), high-frequency tokens (e.g., "the," "that") tend to have smaller norms, while low-frequency tokens (e.g., "apple," "dog") have larger norms. Table 1 presents representative tokens across different frequencies and norm ranges within the embedding space of different base models. We categorize tokens into four groups based on their linguistic function and specificity: Group 1 contains high-frequency function words (e.g., to, is, and), Group 2 contains common question/quantity words (e.g., how, much, many), Group 3 contains general category nouns (e.g., animal, fruit, color), and Group 4 contains specific instances (e.g., dog, apple, purple).
The results presented in Table 1 demonstrate several critical findings. First, we observe a statistically significant separation between functional/abstract words (Group 1) and specific terms (Group 4) across all models, with Group 1 consistently exhibiting the smallest embedding norms and highest frequencies, while Group 4 shows the largest norms and lowest frequencies. Second, the relative ordering of groups remains consistent across all examined models, with Group 1 < Group 2 < Group 3 < Group 4 in terms of embedding norms, despite absolute magnitude variations. Most notably, even across different architectural families (LLaMA vs. Gemma), the hierarchical organization principle remains preserved, though with different absolute scales, where Gemma-7B exhibits systematically larger embedding norms (mean Group 1 norm: 3.160) compared to LLaMA models (mean Group 1 norm: 0.353 ∼ 1.027), yet maintains the same relative hierarchical structure.
this section cite: ['b82', 'b83', 'b84', 'b85', 'b86', 'b10', 'b15', 'b87', 'b51', 'b22']

Section: Conclusion (1)
These findings suggest that the spatial organization of token embeddings reflects the inherent hierarchical relationships in language, supporting the hypothesis that token embedding in LLMs exhibits a tree-like structure, with spatial positioning aligned with token frequency and specificity. It is worth noting, however, that a power-law distribution of token frequency alone does not guarantee the emergence of a hierarchical token embedding, as it also depends on the training objectives. Our analysis demonstrates that the hierarchy is strongly correlated with token frequencies, which can be understood through the lens of LLMs' tokenization and co-occurrence pattern learning during training [89]. While the exact mechanisms underlying this relationship require further investigation in future work, the spatial distribution of token embeddings remains crucial as it provides the primary motivation for our methodological approach.
this section cite: ['b88']

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. GPT-4 Technical Report Year: (2023)
Ref_id:b1 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b2 Title: Open models based on gemini research and technology Year: (2024)
Ref_id:b3 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b4 Title: Is chatgpt a general-purpose natural language processing task solver Year: (2023)
Ref_id:b5 Title: HuggingGPT: Solving AI tasks with ChatGPT and its friends in Hugging Face Year: (2024)
Ref_id:b6 Title: A survey of personalized large language models: Progress and future directions Year: (2025)
Ref_id:b7 Title: Geometric deep learning: going beyond euclidean data Year: (2017)
Ref_id:b8 Title: Constant curvature graph convolutional networks Year: (2020)
Ref_id:b9 Title: Generalization error bound for hyperbolic ordinal embedding Year: (2021)
Ref_id:b10 Title: Poincaré embeddings for learning hierarchical representations Year: (2017)
Ref_id:b11 Title: Hyperbolic geometry of complex networks Year: (2010)
Ref_id:b12 Title: Low distortion delaunay embedding of trees in hyperbolic plane Year: (2011)
Ref_id:b13 Title: Learning continuous hierarchies in the lorentz model of hyperbolic geometry Year: (2018)
Ref_id:b14 Title: Hyperbolic entailment cones for learning hierarchical embeddings Year: (2018)
Ref_id:b15 Title: Hyperbolic image embeddings Year: (2020)
Ref_id:b16 Title: Hyperbolic deep reinforcement learning Year: (2022)
Ref_id:b17 Title: Hyperbolic deep neural networks: A survey Year: (2021)
Ref_id:b18 Title: Hyperbolic graph neural networks: A review of methods and applications Year: (2022)
Ref_id:b19 Title: Client-specific hyperbolic federated learning Year: ()
Ref_id:b20 Title: Hyperbolic deep learning in computer vision: A survey Year: (2023)
Ref_id:b21 Title: Hrcf: Enhancing collaborative filtering via hyperbolic geometric regularization Year: (2022)
Ref_id:b22 Title: Hicf: Hyperbolic informative collaborative filtering Year: (2022)
Ref_id:b23 Title: Hyperbolicity measures democracy in real-world networks Year: (2015)
Ref_id:b24 Title: On the hyperbolicity of large-scale networks Year: (2013)
Ref_id:b25 Title: Scale-free and hierarchical structures in complex networks Year: (2003)
Ref_id:b26 Title: Hierarchical structures induce long-range dynamical correlations in written texts Year: (2006)
Ref_id:b27 Title: Rethinking the relationship between the power law and hierarchical structures Year: (2025)
Ref_id:b28 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b29 Title: Zero: Memory optimizations toward training trillion parameter models Year: (2020)
Ref_id:b30 Title: Low-rank adaptation of large language models Year: (2021)
Ref_id:b31 Title: Hyperbolic graph convolutional neural networks Year: (2019)
Ref_id:b32 Title: Hyperbolic neural networks Year: (2018)
Ref_id:b33 Title: Hyperbolic temporal network embedding Year: ()
Ref_id:b34 Title: Curve your attention: Mixed-curvature transformers for graph representation learning Year: (2023)
Ref_id:b35 Title: Hyperbolic geometric latent diffusion model for graph generation Year: (2024)
Ref_id:b36 Title: Fully hyperbolic neural networks Year: (2021)
Ref_id:b37 Title: Hyperbolic neural networks++ Year: (2020)
Ref_id:b38 Title: Curvature learning for generalization of hyperbolic neural networks Year: (2025)
Ref_id:b39 Title: Klein model for hyperbolic neural networks Year: (2024)
Ref_id:b40 Title: Hyperbolic geometry in computer vision: A novel framework for convolutional neural networks Year: (2023)
Ref_id:b41 Title: Poincaré resnet Year: (2023)
Ref_id:b42 Title: Philippe Chlenski, and Itsik Pe'er. Hyperbolic genome embeddings Year: (2025)
Ref_id:b43 Title: Hyperbolic graph neural networks Year: (2019)
Ref_id:b44 Title: Hyperbolic graph attention network Year: (2021)
Ref_id:b45 Title: κhgcn: Tree-likeness modeling via continuous and discrete curvature learning Year: (2023)
Ref_id:b46 Title: Hyperbolic attention networks Year: (2018)
Ref_id:b47 Title: Hypformer: Exploring efficient transformer fully in hyperbolic space Year: (2024)
Ref_id:b48 Title: HGCF: Hyperbolic graph convolution networks for collaborative filtering Year: (2021)
Ref_id:b49 Title: Unsupervised discovery of the long-tail in instance segmentation using hierarchical self-supervision Year: (2021)
Ref_id:b50 Title: Hyperbolic embedding inference for structured multi-label prediction Year: (2022)
Ref_id:b51 Title: Discrete-time temporal network embedding via implicit hierarchical learning in hyperbolic space Year: (2021)
Ref_id:b52 Title: Curvature generation in curved spaces for few-shot learning Year: (2021)
Ref_id:b53 Title: Learning transferable visual models from natural language supervision Year: ()
Ref_id:b54 Title: Hyperbolic image-text representations Year: ()
Ref_id:b55 Title: Compositional entailment learning for hyperbolic visionlanguage models Year: (2024)
Ref_id:b56 Title: Hyperbolic safety-aware vision-language models Year: ()
Ref_id:b57 Title: Hypercore: The core framework for building hyperbolic foundation models with comprehensive modules Year: ()
Ref_id:b58 Title: Riemannian optimization in pytorch Year: (2020)
Ref_id:b59 Title: Optimization techniques on riemannian manifolds Year: (2014)
Ref_id:b60 Title: Riemannian adaptive optimization methods Year: (2018)
Ref_id:b61 Title: Visualizing and measuring the geometry of bert Year: (2019)
Ref_id:b62 Title: Representation degeneration problem in training natural language generation models Year: (2019)
Ref_id:b63 Title: Isoscore: Measuring the uniformity of embedding space utilization Year: (2021)
Ref_id:b64 Title: Aleksandr Drozd, and Felice Dell'Orletta. Outliers dimensions that disrupt transformers are driven by frequency Year: (2022)
Ref_id:b65 Title:  Year: (2022-11)
Ref_id:b66 Title: The power of scale for parameter-efficient prompt tuning Year: (2021)
Ref_id:b67 Title: Prefix-tuning: Optimizing continuous prompts for generation Year: (2021)
Ref_id:b68 Title: Exploring universal intrinsic task subspace via prompt tuning Year: (2021)
Ref_id:b69 Title: Parameter-efficient transfer learning for nlp Year: (2019)
Ref_id:b70 Title: Counterinterference adapter for multilingual machine translation Year: (2021)
Ref_id:b71 Title: Intrinsic dimensionality explains the effectiveness of language model fine-tuning Year: (2020)
Ref_id:b72 Title: Parameter efficient tuning with kronecker adapter Year: (2022)
Ref_id:b73 Title: Lora+: Efficient low rank adaptation of large models Year: (2024)
Ref_id:b74 Title: Weight-decomposed low-rank adaptation Year: (2024)
Ref_id:b75 Title: Adaptive budget allocation for parameter-efficient fine-tuning Year: ()
Ref_id:b76 Title: Lora ensembles for large language model fine-tuning Year: (2023)
Ref_id:b77 Title: Mini-ensemble low-rank adapters for parameter-efficient fine-tuning Year: (2024)
Ref_id:b78 Title: Qlora: Efficient finetuning of quantized llms Year: (2024)
Ref_id:b79 Title: Qa-lora: Quantization-aware low-rank adaptation of large language models Year: (2023)
Ref_id:b80 Title: Loftq: Lora-fine-tuning-aware quantization for large language models Year: (2023)
Ref_id:b81 Title: The numerical stability of hyperbolic representation learning Year: (2023)
Ref_id:b82 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b83 Title: Program induction by rationale generation: Learning to solve and explain algebraic word problems Year: (2017)
Ref_id:b84 Title: Mawps: A math word problem repository Year: (2016)
Ref_id:b85 Title: Are nlp models really able to solve simple math word problems? arXiv preprint Year: (2021)
Ref_id:b86 Title: Powerlaw: a python package for analysis of heavy-tailed distributions Year: (2014)
Ref_id:b87 Title: Hierarchical organization in complex networks Year: (2003)
Ref_id:b88 Title: Neural machine translation of rare words with subword units Year: (2016)
Ref_id:b89 Title: Hyperbolic groups Year: (1987)
Ref_id:b90 Title: Computing the gromov hyperbolicity of a discrete metric space Year: (2015)
Ref_id:b91 Title: Semi-supervised classification with graph convolutional networks Year: (2016)
Ref_id:b92 Title: Curvature and temperature of complex networks Year: (2009)
Ref_id:b93 Title: Greedy forwarding in dynamic scale-free networks embedded in hyperbolic metric spaces Year: (2010)
Ref_id:b94 Title: Hyperbolic geometry. Flavors of geometry Year: (1997)
Ref_id:b95 Title: Foundations of hyperbolic manifolds Year: (2006)
Ref_id:b96 Title: A hyperbolic-to-hyperbolic graph convolutional network Year: (2021)
Ref_id:b97 Title: Llm-adapters: An adapter family for parameter-efficient fine-tuning of large language models Year: (2023)
Ref_id:b98 Title: Collective classification in network data Year: (2008)
Ref_id:b99 Title: Exploring network structure, dynamics, and function using networkx Year: (2008)
Ref_id:b100 Title: Mixed-curvature variational autoencoders Year: (2019)
Ref_id:b101 Title: Methods of theoretical physics Year: (1946)
Ref_id:b102 Title: The interplay of the polar decomposition theorem and the lorentz group Year: (2002)
Ref_id:b103 Title: Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. Gsm-symbolic: Understanding the limitations of mathematical reasoning in large language models Year: (2024)
