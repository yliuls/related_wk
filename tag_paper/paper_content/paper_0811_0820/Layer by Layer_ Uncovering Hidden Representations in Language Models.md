Title: Layer by Layer: Uncovering Hidden Representations in Language Models
Abstract: From extracting features to generating text, the outputs of large language models (LLMs) typically rely on the final layers, following the conventional wisdom that earlier layers capture only low-level cues. However, our analysis shows that intermediate layers can encode even richer representations, often improving performance on a range of downstream tasks. To explain and quantify these hidden-layer properties, we propose a unified framework of representation quality metrics based on information theory, geometry, and invariance to input perturbations. Our framework highlights how each layer balances information compression and signal preservation, revealing why mid-depth embeddings can exceed the last layer's performance. Through extensive experiments on 32 text-embedding tasks across various architectures (transformers, state-space models) and domains (language, vision), we demonstrate that intermediate layers consistently provide stronger features, challenging the standard view on final-layer embeddings and opening new directions on using mid-layer representations for more robust and accurate representations. Layer by Layer: Uncovering Hidden Representations in Language Models • Invariance: Are embeddings robust to input perturbations (e.g., InfoNCE (Oord et al., 2018), LiDAR (Thilak et al., 2024) and DiME (Skean et al., 2023))?We show that these perspectives can be viewed under a single lens, which clarifies how intermediate layers strike a balance between retaining features and discarding noise.Key findings and contributions. Our investigation leads to several important insights:• Intermediate layers consistently outperform final layers. This pattern is evident in both transformers and SSMs, suggesting a broad architecture-agnostic effect.• Autoregressive vs. masked-language training. Autoregressive models exhibit a pronounced mid-layer "compression valley," whereas masked or bidirectional models show milder intermediate changes.• Domain-general effect. We extend these results to vision models and find that autoregressive image transformers display the same mid-depth bottleneck, indicating that the training objective, rather than the data modality, is the key driver.• CoT finetuning. Analyzing chain-of-thought (CoT) reveals that finetuning can reshape mid-layer entropy, preserving latent context for multi-step reasoning.

Section: Introduction
Large Language Models (LLMs) have driven remarkable progress in natural language processing (NLP), achieving state-of-the-art results on many tasks (Brown et al., 2020;Devlin et al., 2019;Li et al., 2022). At the heart of most applications lies a common assumption: final-layer representations are the most useful for downstream tasks. Yet a fundamental question remains: does the final layer always yield the best representation?
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). The average score of 32 MTEB tasks using the outputs of every model layer as embeddings for three different model architectures. The x-axis is the depth percentage of the layer, rather than the layer number which varies across models.
In this paper, we conduct a layer-wise analysis of LLMs across diverse architectures-including transformer-based ones (Vaswani et al., 2017), state-space models (SSMs) (Gu & Dao, 2024), and encoder-based models like BERT (Devlin et al., 2019)-spanning parameter scales from tens of millions to billions. Through systematic evaluation on 32 embedding tasks from the Massive Text Embedding Benchmark (MTEB) (Muennighoff et al., 2022), we find that intermediate layers often surpass the final layer by up to 16% in downstream accuracy. Figure 1 illustrates this phenomenon, where mid-depth layers provide particularly strong representations while the very last layer can become overly specialized to the pretraining objective.
A unified framework. To better understand intermediate layers' effectiveness, we combine three complementary perspectives (Section 3):
• Information-theoretic: How much do layers compress or preserve semantic information (Shwartz-Ziv & Tishby, 2019;Shwartz-Ziv, 2022)?
• Geometric: How do token embeddings unfold in highdimensional space (Hosseini & Fedorenko, 2023))?
Overall, our results challenge the default reliance on finallayer embeddings and highlight intermediate layers as potentially underutilized sources of meaningful features. In this paper, we detail our unified framework (Section 3), present extensive experiments in both language and vision (Section 4, 5, 6), and conclude with a discussion of our findings, their implications, and future directions.foot_8
this section cite: ['b10', 'b18', 'b34', 'b64', 'b26', 'b18', 'b41', 'b54', 'b53', 'b31']

Section: Related Work
Understanding Neural Representations. A long line of research has aimed to understand how deep neural networks encode and organize information. Early studies employed linear probes for intermediate layers (Alain & Bengio, 2017), while subsequent efforts introduced more sophisticated techniques such as SVCCA (Raghu et al., 2017) to compare learned features across architectures and training regimes. Although these approaches have shed light on representation dynamics, most focus on vision models or shallow networks. In contrast, our work contributes to a growing body of literature extending layer-wise analysis to large-scale language models, emphasizing specific behaviors of intermediate layers across diverse architectures. Complementing our empirical findings, Saponati et al. (2025) present a theoretical analysis of how different pretext
this section cite: ['b1', 'b47', 'b51']

Section: Layer-wise Analysis in Language Models.
Recent work has increasingly focused on identifying which transformer layers encode different types of information. For example, linguistic features such as part-of-speech tags or semantic roles are best encoded by the middle layers of a BERT (Liu et al., 2019;Tenney et al., 2019;Voita et al., 2019). More recent work has shown that mid-depth layers sometimes hold surprisingly robust features, challenging the usual emphasis on final layer representations (Jin et al., 2024;Gurnee & Tegmark, 2023;Fan et al., 2024). A related line of work investigates the attention sink phenomenon (Xiao et al., 2024;Brunner et al., 2020;Gu et al., 2025), in which attention disproportionately concentrates on a single token. Notably, intermediate decoder layers have been shown to not exhibit these extreme attention sinks (Barbero et al., 2025), suggesting they engage in more distributed and meaningful information processing than the shallow or deep layers.
this section cite: ['b35', 'b59', 'b65', 'b32', 'b28', 'b22', 'b67', 'b11', 'b27', 'b5']

Section: Compression and Generalization.
Multiple lines of research link compression and generalization performance (Deletang et al., 2024). For instance, Bordes et al. (2023) demonstrated that discarding certain layers in selfsupervised encoders can even improve downstream accuracy, while Park et al. (2024a) found that LLM embeddings often lie in low-dimensional manifolds. Our empirical study reinforces these ideas by demonstrating that many networks-especially autoregressive transformers-naturally develop a mid-layer bottleneck that appears crucial for balancing "signal" versus "noise." We show how intermediate layers can achieve optimal trade-offs between preserving task-relevant information and discarding superfluous detail.
this section cite: ['b17', 'b9']

Section: Representation Quality Metrics.
A variety of metrics have been proposed to quantify the "quality" of learned representations. We group them into three main categories:
• Information-theoretic measures capture how much a model's internal representations compress or preserve relevant information. For example, the Information Bottleneck (Shwartz-Ziv & Tishby, 2019; Shwartz-Ziv, 2022) analyzes whether intermediate layers discard noise while retaining essential features. Intrinsic dimensionality, which describes the minimum number of features to represent data, has also been used to analyze intermediate layers in LLMs (Cheng et al., 2025;Valeriani et al., 2023;Razzhigaev et al., 2024). This line of work has shown semantic abstractions useful for downstream tasks are better encoded in middle layers than last layers in large transformer models. While we do not study intrinsic dimensionality in our subsequent analysis, it would make a promising direction for future work.
• Geometric measures focus on the structure of embeddings in high-dimensional space. Classical approaches include analyzing singular values and effective rank of the representation matrix (Garrido et al., 2023). The anisotropy metric of Razzhigaev et al. (2024) has been used to study compression in intermediate model layers and we compare our results with their findings in Section 4.2. Anisotropy fits in well with our proposed framework, though we leave a formal integration to future work. Recent work explores curvature (Hosseini & Fedorenko, 2023) to quantify how smoothly tokens are mapped across consecutive positions or time steps.
• Task-based or invariance metrics evaluate how well representations support downstream goals. For instance, augmentations-based approaches such as In-foNCE (Oord et al., 2018) and LiDAR (Thilak et al., 2024) estimate invariance to perturbations, while methods like NESum or Self-Cluster (Agrawal et al., 2022) link closely to entropy. In computer vision, these scores often correlate strongly with downstream accuracy, highlightFing how robust the embeddings are.
Although these representation quality metric categories may appear distinct, we show (Section 3) that many can be unified under a single lens. This unification illuminates why certain intermediate layers balance compression, geometry, and invariance so effectively, leading to better representations for downstream tasks.
Overall, our work bridges these overlapping threads by evaluating a range of architectures and training paradigms via a unified set of metrics. Beyond merely confirming that intermediate layers can be effective, we elucidate why this happens, tying it to fundamental properties such as entropy, invariance, and geometry. This novel perspective provides an avenue for both finer-grained diagnostics of large language models and more deliberate design of mid-layer representations for downstream tasks.
this section cite: ['b14', 'b63', 'b48', 'b24', 'b48', 'b31', 'b42', 'b60', 'b0']

Section: A Unified Framework for Neural Representations
Key Takeaway: Matrix-based entropy unifies seemingly disparate metrics of representation quality, providing a single theoretical lens for analyzing compression, geometry, and invariance.
A central challenge in analyzing internal representations is determining how to assess their quality. Although existing work draws on numerous ideas-from mutual information to geometric manifold analysis to invariance under augmentations-these threads can seem disparate. In this section, we consolidate them into a unified theoretical framework that shows how these seemingly different metrics connect and why they collectively measure "representation quality."
this section cite: []

Section: Notation and Motivation
Consider a neural network that maps inputs x (e.g., tokens in a sequence) to internal hidden states Z. We denote Z ∈ R N ×D as a matrix of N data samples (or tokens) in D dimensions. Some key questions arise:
1. How compressed are these representations?
2. How robust are they to perturbations or augmentations?
3. How do they geometrically organize different inputs?
Answers to these questions can illuminate which layers strike the right balance between preserving relevant features and discarding noise.
this section cite: []

Section: Matrix-Based Entropy: A Common Theoretical Thread
We focus on a key quantity known as matrix-based entropy (Giraldo et al., 2014;Skean et al., 2023), which applies directly to the Gram matrix K = ZZ ⊤ . Let {λ i (K)} be the (nonnegative) eigenvalues of K. For any order α > 0, define:
S α (Z) = 1 1 -α log r i=1 λi(K) tr(K) α ,(1)
where r = rank(K) ≤ min(N, D). Intuitively, if only a few eigenvalues dominate, S α (Z) is small-indicating a highly compressed representation. Conversely, if Z is spread out across many principal directions, S α (Z) is large.
By varying α, one smoothly transitions between notions like collision entropy (α = 2) and von Neumann entropy (α → 1). We will typically use α = 1 for simplicity.
this section cite: ['b25', 'b56']

Section: Bridging geometry, invariance, and feature locality.
A key benefit of matrix-based entropy is that it unifies multiple representational perspectives:
• Compression or information content: A handful of large eigenvalues in K = ZZ ⊤ indicates that Z is lowrank, i.e. the model has collapsed much of the input variation into fewer dimensions. In contrast, a more uniform eigenvalue spectrum implies higher-entropy, more diverse features.
• Geometric smoothness: If tokens within a prompt follow a trajectory in embedding space with sharp turns, that curvature can manifest as skewed eigenvalue spectra (Hosseini & Fedorenko, 2023). Curvature also differentiates local transitions (token-to-token) from global structural patterns across longer segments or entire prompts.
• Invariance under augmentations: Metrics like In-foNCE (Oord et al., 2018) and LiDAR (Thilak et al., 2024) effectively measure whether augmentations of the same sample (e.g. character swaps) map to similar embeddings. Strong invariance corresponds to stable clustering in ZZ ⊤ , which again depends on the distribution of eigenvalues and how local vs. global features are retained or discarded.
Thus, evaluating S α (Z) provides a single lens for assessing "representation quality" across compression, geometric structure, and invariance-and highlights how both local details and global patterns are organized.
this section cite: ['b31', 'b42', 'b60']

Section: Representation Evaluation Metrics
Key Takeaway: Information-theoretic, geometric, and invariance-based metrics offer complementary perspectives on representation quality that can all be understood through matrix-based entropy.
We now introduce the seven representation evaluation metrics used in our experiments, grouped into three broad categories: (1) information-theoretic, (2) geometric, and (3) augmentation-invariance. All relate back to the Gram matrix K and hence to Eq. ( 1).
this section cite: []

Section: INFORMATION-THEORETIC METRICS
Prompt Entropy. Following Wei et al. (2024), we apply matrix-based entropy (Eq. 1) to the token embeddings within a single prompt. This prompt entropy quantifies how widely tokens are spread in the embedding space. Higher entropy indicates more diverse, less redundant token-level features; lower entropy implies stronger compression.
this section cite: ['b66']

Section: Dataset Entropy.
We can also aggregate embeddings across N prompts by taking the mean token embedding of each prompt to form Z ∈ R N ×D . Applying entropy to Z yields a dataset-level measure of global diversity-revealing how distinctly the model separates different inputs.
Effective Rank. (Roy & Vetterli, 2007) can be shown to be a lower bound to exp(S 1 (Z)), highlighting how dimensionality effectively shrinks if the representation is strongly compressed. We prove this connection in Theorem 1. This has implications for popular representation evaluation metrics such as RankMe (Garrido et al., 2023) and LiDAR (Thilak et al., 2024), which are both inspired by Effective Rank.
this section cite: ['b50', 'b24', 'b60']

Section: GEOMETRIC METRICS
Curvature. Proposed by Hosseini & Fedorenko (2023), curvature captures how sharply the token embeddings turn when viewed as a sequence in R D . For a prompt of length L, let v k = z k+1 -z k be the difference between consecutive tokens. The average curvature is:
C = 1 L -2 L-2 k=1 arccos v ⊤ k+1 v k ∥v k+1 ∥∥v k ∥ .
Higher curvature means consecutive tokens shift direction abruptly and more local level features; lower curvature suggests a smoother trajectory and global level features.
this section cite: ['b31']

Section: AUGMENTATION INVARIANCE METRICS
Lastly, we assess how stable the model's representations are to small perturbations of the same input (e.g., random character swaps, keyboard-level changes; see Appendix). Suppose a prompt p i is augmented into p
(a) i and p (b)
i . After embedding these, we compare the row vectors in Z 1 , Z 2 ∈ R N ×D under different scoring criteria:
InfoNCE. This self-supervised objective (Oord et al., 2018) encourages matched samples to lie close in embedding space while pushing unmatched samples away. A lower InfoNCE loss indicates stronger invariance to augmentation.
LiDAR. LiDAR (Thilak et al., 2024) uses a linear discriminant approach that measures within-class versus betweenclass scatter. Treating each prompt as its own class, LiDAR checks how well augmentations form tight clusters.
this section cite: ['b42', 'b60']

Section: DiME.
Similarly, DiME (Skean et al., 2023) is grounded in matrix-based entropy. It compares real paired samples against random pairings to estimate how uniquely aligned correct augmentations are.
this section cite: ['b56']

Section: Core Theoretical Results
Key Takeaway: Our theoretical framework establishes concrete connections between representation entropy and downstream performance through properties like effective rank and invariance.
Here, we summarize key statements that justify why these metrics meaningfully measure representation quality. We refer to the appendix G for details and proofs. Beyond serving as a unifying view, matrix-based entropy also connects to foundational concepts like majorization, Schur concavity, and mutual information. Furthermore, we can directly relate the eigenvalue entropy to the matrix entropy, most naturally via the Effective Rank (Roy & Vetterli, 2007). The following theorem makes this connection explicit.
Theorem 1 (Lower Bound via Effective Rank). For Shannon-based entropy (α → 1), EffRank(Z) ≤ exp S 1 (Z) , meaning a large effective rank implies a high entropy.
Under appropriate conditions on the data distribution and model, we can show connections between prompt entropy and dataset entropy via the following scaling behaviors: Theorem 2 (Informal).
1. If prompt entropy remains near its maximum for all prompts, then the dataset entropy S 2 Z Z ⊤ grows on the order of log L 2 N . 2. If prompt entropy instead stays near its minimum for all prompts, then dataset entropy grows more slowly, on the order of log L 2 N 3 .
In short, high token-level (prompt) diversity encourages broader global diversity in the dataset-level embeddings, whereas over-compressing token representations can limit how effectively different prompts separate. Our subsequent analysis connects these ideas to self-supervised objectives like InfoNCE, which also tie higher entropy to stronger robustness and discriminability in the learned representations.
Theorem 3 (Dataset Entropy Bounds InfoNCE). For data X and representation Z(X), the InfoNCE loss on N samples satisfies:
log(N ) -InfoNCE ≤ I(X; Z) ≤ H(Z),
where H(Z) is interpretable as matrix-based entropy at the dataset level. Hence, reducing InfoNCE implies learning a higher-entropy (and thus often more robust) representation.
Practical outlook. Overall, our theoretical analysis shows that compression (entropy), geometry (curvature, rank), and invariance (e.g. InfoNCE) are all facets of how the Gram matrix ZZ ⊤ distributes variance. Examining these metrics across different layers reveals exactly where a network "prunes" redundancy (low entropy) versus preserving essential distinctions (high entropy). This unified perspective also facilitates cross-architecture comparisons (e.g. transformers vs. SSMs) by highlighting how each architecture organizes information internally. Beyond offering a theoretical foundation, it provides a practical blueprint for diagnosing, tuning, and improving hidden-layer representations.
this section cite: ['b50']

Section: Empirical Results
In this section, we empirically test our theoretical framework through extensive experiments across architectures, scales, and training regimes. We focus on three key questions:
• Do intermediate layers consistently outperform final layers across diverse downstream tasks?
• How do these intermediate representations differ across architectures, training stages, and scales?
• How does post-training methods (e.g., fine-tuning and chain-of-thought) reshape representations? 4.1. Downstream Task Performance Key Takeaway: Intermediate layers of language models consistently outperform final layers across all architectures and tasks, challenging the conventional wisdom of using final-layer representations.
In this section, we use intermediate layers for downstream embedding tasks and employ our unified framework from Section 3, measuring all the embeddings across all layers.
this section cite: []

Section: EXPERIMENTAL SETUP

this section cite: []

Section: Models
We evaluate three distinct architectural families: Pythia and Llama3 (decoder-only transformers) (Biderman et al., 2023;Dubey et al., 2024), Mamba (state space model) (Gu & Dao, 2024), BERT (encoder-only transformer) (Devlin et al., 2019) and LLM2Vec models (bidirectional attention) (Behnam Ghader et al., 2024).
Tasks We test each layer's embeddings on 32 tasks from the Massive Text Embedding Benchmark (MTEB) (Muennighoff et al., 2022), spanning classification, clustering, and reranking dor a comprehensive evaluation across various tasks. We refer to the Appendix for details.
this section cite: ['b7', 'b20', 'b26', 'b18', 'b6', 'b41']

Section: INTERMEDIATE LAYERS OFTEN OUTPERFORM FINAL LAYERS
Are final-layer embeddings indeed optimal for downstream tasks? In Figure 1, we compare average performance on MTEB tasks across all layers of the three models.
this section cite: []

Section: Key observation.
In nearly every task, some intermediate layer outperforms the final layer. The absolute improvement ranges from 2% to as high as 16% on average, and the best layer often resides around the mid-depth of the network. This phenomena is consistent across all the different architectures. This confirms emerging observations in recent work for generation tasks (Bordes et al., 2023;El-Nouby et al., 2024;Chen et al., 2020;Fan et al., 2024) and extends them to a wider range of benchmarks and tasks.
Why do these layers matter? From our theoretical perspective, intermediate layers appear to strike a balance between retaining sufficient information (avoiding overcompression) and discarding low-level noise. Later in Section 4.2, we show that these sweet spots are not random but tied to how intermediate layers are processing information.
this section cite: ['b9', 'b21', 'b13', 'b22']

Section: LAYER-WISE METRICS CORRELATE WITH DOWNSTREAM PERFORMANCE
To validate our framework, we analyze how each evaluation metric correlates with downstream performance. Figures 3 and 8 show distance correlations between metrics and task scores for Pythia-410M. We find that all metrics exhibit strong relationships with downstream performance. Among them, curvature, DiME, and InfoNCE stand out with particularly high correlations. These associations remain robust across different correlation measures, including Spearman and Kendall, reinforcing the reliability of our findings.
Our results suggest that our metrics capture some aspects of intermediate representations that contribute to downstream utility. In Appendix E, we leverage these strong correlations to select high-performing layers in an unsupervised manner, following (Agrawal et al., 2022;Garrido et al., 2023;Thilak et al., 2024). In short, we can identify an intermediate layer that surpasses the final layer in downstream performance-without using any task-specific labels. For instance, using DiME-based layer selection leads to a 3% average improvement in MTEB scores for the Pythia-410M model.
this section cite: ['b0', 'b24', 'b60']

Section: Architectural and Scale Differences
Key Takeaway: Different architectures exhibit distinct patterns of information compression. Autoregressive models show mid-layer bottlenecks while bidirectional models maintain more uniform trends.
Aside from strong correlations with downstream performance, we can use our evaluation framework to assess the internal behaviors of LLMs. In both this section and Section 4.3, we use WikiText-103 (Merity et al., 2017) for analyzing our representation metrics on standard textual data.
To investigate how architecture and model size influ-ence representation quality, we compare three fundamentally different LLM variants-BERT (encoder-only), Pythia (decoder-only), and Mamba (state-space model)-and then scale up Pythia to observe emerging trends. Encoder vs. Decoder vs. SSM. Figure 2 shows how prompt entropy, curvature, and augmentation metrics evolve across each model's layers. BERT, which encodes the entire input bidirectionally, generally maintains high entropy across layers, suggesting minimal compression: the model can see all tokens at once and need not discard as much information. By contrast, the decoder-only Pythia exhibits a strong mid-layer entropy dip, reflecting its autoregressive objective's tendency to filter or prune non-local details in the middle of the network. As a result, Pythia's "sweet spot" for downstream tasks often lies around mid-depth, where it LiDAR Dataset Entropy Prompt Entropy InfoNCE Curvature DiME Representation Metrics 0.00 0.25 0.50 0.75 1.00 Distance Correlation (dCor)
Figure 3: Relationship between representation metrics and task performance averaged across layers for Pythia 410M. Using distance correlation (dCor), we see strong associative relationships across the board with DiME exhibiting the strongest relationship with downstream performance. We use dCor due to its robustness and ability to measure both linear and non-linear relationships (dCor ∈ [0, 1] with 0 indicating statistical independence and 1 indicating strong dependency). We defer additional results to the Appendix.
balances essential context and compression. Mamba, meanwhile, processes sequences through a state-space approach that yields flatter, more uniform curves across depth: it neither retains as much information as BERT nor compresses as aggressively as Pythia's mid-layers. These conclusions align with Razzhigaev et al. (2024) which showed a flat layer-wise anisotropy for encoder models and a spike in intermediate layer anisotropy for decoder models.
this section cite: ['b40', 'b48']

Section: Scaling Size Effects.
In Figure 12, we analyze Pythia models ranging from 14M to 1B parameters. Larger models display more pronounced intermediate compression (entropy dips), indicating a heightened ability to distill relevant features. We also observe smoother token trajectories (lower curvature) and stronger invariance (higher LiDAR), consistent with findings that bigger models more effectively filter noise and capture long-range dependencies. These trends reinforce why performance peaks in the middle of the network: larger models hold more capacity to compress intermediate representations, yet still preserve crucial semantic details.
Finetuning Effects In Figure 13, we study how finetuning affects the internal representations of Llama3 (Dubey et al., 2024). We compare the baseline Llama3-8B to two finetuned LLM2Vec models (Behnam Ghader et al., 2024). The LLM2Vec-mntp-unsup-simcse model enables bidirectional attention in Llama3 and goes through two unsupervised training phases to improve Llama3's performance on embedding tasks. The LLM2Vec-mntp-supervised adds an additional supervised finetuning phase. It is clear that both finetuned models have improved augmentation invariance. Furthermore, the unsupervised model has higher prompt entropy than Llama3 while the supervised model has less.
this section cite: ['b20', 'b6']

Section: Layer-Level Analysis of Transformer Sub-Components.
While our experiments treat each transformer layer as a single unit, transformer blocks are composed of multiple sub-layers (pre-attention normalization, self-attention, residuals, MLPs). By measuring entropy after each sub-layer, we find in Figure 15 that residual connections drive the midnetwork compression observed in Section 4.2. Specifically:
• Sub-layers before residuals (e.g. pre-attention, attention scores, or MLP pre-residual outputs) often show only mild compression.
• Residual sub-layers exhibit a pronounced drop in entropy, indicating a significant filtering of information.
A concurrent study (Csordás et al., 2025) observed a decrease in the residual stream norm in the second half of decoder models, reinforcing our findings.
The strong entropy "valley" at intermediate layers is tied to how residual paths merge new signals with the existing hidden state. This aligns with prior work indicating that residuals act as a regularizer (Marion et al., 2024), smoothing out spurious components in hidden representations.
this section cite: ['b15', 'b39']

Section: Impact of Training Progression
Takeaway: Significant changes during training occur in intermediate layers and early layers stabilize quickly, supporting the detokenization hypothesis.
We measure Pythia's metrics at multiple checkpoints to understand how layer-wise representations evolve throughout training (Figures 4 and 11). Two main observations emerge:
Intermediate Layers Undergo the Most Change. The largest shifts in representation quality occur in mid-depth layers. Specifically, prompt entropy steadily decreases there as training progresses, implying that intermediate layers increasingly compress and abstract the input. Meanwhile, LiDAR scores are minimal in these same layers. Likewise, curvature becomes smoother in the middle of the network, suggesting the model refines its internal structure to capture longer-range or more nuanced patterns in language.
Early Layers Stabilize Quickly. In contrast to intermediate layers, the earliest layers change very little after the initial phase of training. This observation aligns with the "detokenization" hypothesis of Lad et al. (2024), which posits that the main functional role of early layers is to convert raw tokens into a basic embedding space. This idea is closely related to the "shared task" layers of Zhao et al. (2024), introduced in the context of instruction tuning on diverse tasks. In particular, they show that the first nine layers of LlaMA 2 7B (Touvron et al., 2023) perform general task-agnostic operations. As a result, the most substantial changes to representations, such as enhanced compression, are driven primarily by the intermediate layers, reinforcing their importance for learning robust, high-level features.
this section cite: ['b33', 'b69', 'b62']

Section: Impact of Chain-of-Thought Finetuning
Key Takeaway: CoT finetuning enables models to maintain richer context throughout their layers.
Recent work has highlighted Chain-of-Thought (CoT) finetuning as a powerful strategy for improving reasoning capabilities (Arefin et al., 2025;DeepSeek-AI, 2025). To examine its effects on representations, in Figure 5 we compare Qwen 2.5 and Qwen 2.5-Math (Yang et al., 2024), where the latter underwent additional math pretraining and CoT finetuning. Measuring token-level prompt entropy across sequence length reveals that the finetuned model maintains higher entropy with lower variance across examples.  These findings suggest that CoT finetuning encourages models to preserve more context throughout their hidden layers, enabling better multi-step reasoning. Our framework provides a quantitative lens into how CoT fine-tuning pushes models to maintain richer internal representations across sequences, explaining its effectiveness in multi-step tasks.
While CoT traces can be inspected directly in these models, our approach is particularly valuable for analyzing models that reason in continuous latent space (Hao et al., 2024).
this section cite: ['b2', 'b16', 'b68', 'b29']

Section: Extreme Input Conditions
To better probe the underlying factors affecting representation quality, we inspect each layer's responsiveness to different input types. We use Pythia-410M on three types of extreme prompts and measure prompt entropy across layers (Figure 6). We prove examples of these prompts in Appendix F. Overall, we find that:
1. Token repetition compresses intermediate layers.
As p increases (i.e., more repeated tokens), prompt entropy decreases sharply in mid-depth layers, suggesting that the model recognizes/encodes repetitive patterns and discards redundancy in its internal representation.
2. Random tokens inflate early-layer entropy. Adding token-level randomness, increases entropy significantly in early layers, revealing their sensitivity to noise. In contrast, deeper layers are more robust.
Overall, these results confirm that intermediate layers play a major role in handling complex or unusual inputs, selectively compressing or filtering out repetitive patterns while retaining crucial distinctions. Early layers are more sensitive to noise and the incremental benefit of adding more tokens diminishes with prompt length. This behavior highlights the diverse ways in which different layers balance the trade-off between preserving and discarding information, underscoring the significance of intermediate representations.
this section cite: []

Section: Comparison to Vision Transformers
Do our findings extend to other domains like computer vision? Vision models employ diverse architectures and training objectives from fully supervised learning to selfsupervised methods, and from bidirectional to autoregressive encoders. Their diversity provides an ideal testbed to examine how well our findings generalize and how different training objectives shape internal representations.
We examine several representative vision approaches: ViT (Dosovitskiy et al., 2021), a supervised transformer trained on labeled data; CLIP (Radford et al., 2021), a weakly supervised image encoder; BEiT (Bao et al., 2022), a self-supervised encoder that reconstructs masked patches; DINOv2 (Oquab et al., 2024), a self-supervised approach leveraging augmentations and exponential moving average teachers; MAE (He et al., 2022), a self-supervised approach that reconstructs images from masked patches; AIM (El-Nouby et al., 2024), an autoregressive transformer that predicts the next patch in an image sequence (GPT-style nexttoken prediction); and AIMv2 (Fini et al., 2025), which extends AIM with a multimodal next-token prediction task. In Figure 14, we evaluate every model layer on ImageNet-1k with attention probing and our suite of metrics.
AIM exhibits behavior similar to language models. AIM, which predicts image patches sequentially, exhibits the same entropy "valley" and accuracy peak at intermediate layers that we observed in language models like Pythia. This pattern suggests that autoregressive training, whether over text tokens or image patches, consistently creates a mid-depth information bottleneck. The sequential prediction constraint forces models to compress non-local contextual information early in processing, then selectively re-expand the most relevant features for accurate prediction. AIM's strong intermediate performance was first noted in (El-Nouby et al., 2024). Interestingly, while the AIMv2 model does not show improved intermediate accuracy, it still produces an entropy valley. We hypothesize this difference is due to the multimodal text-vision pretext task, which may alter information compression dynamics.
Vision transformers behave differently from language models. All models except for AIM exhibit strictly increasing downstream accuracy toward final layers. Similar trends have been shown for ResNets (Sorscher et al., 2022), where few-shot classification error is strictly decreasing across layers. Most non-autoregressive vision models show steadily increasing dataset entropy. The notable exception is BEIT, which exhibits a substantial intermediate dip. Taken together, the results suggest that without an autoregressive objective, vision transformers have less need for drastic transformations at mid-depth.
Autoregression as the driving factor. The strong midlayer compression observed in LLMs seems to be not purely a property of "sequential token data" vs. "image patch data," but rather a byproduct of pretraining. While various selfsupervised (or fully supervised) objectives in vision foster more uniform feature building across layers, autoregressive vision models develop similar mid-layer bottlenecks that we see in language. Thus, the objective design-whether or not a model is autoregressive-appears crucial in shaping layer-wise representation quality, regardless of domain.
this section cite: ['b19', 'b46', 'b4', 'b43', 'b30', 'b21', 'b23', 'b21', 'b58']

Section: Discussion and Conclusion
We investigated the representation quality of intermediate layers in LLMs and their role in downstream task performance. We introduced a unified framework of evaluation metrics, establish theoretical connections among them, and apply these metrics to analyze transformer-based architectures, SSMs, and vision models. A key phenomenon unveiled by prompt entropy was an information bottleneck in the middle layers of autoregressive transformers in both vision and language domains. Furthermore, we show that intermediate layers often surpass final layers in representation quality, holding implications for feature relevance and extraction. DiME, curvature, and infoNCE correlate well with downstream performance, suggesting a fundamental connection between representation and generalizability.
In conclusion, our work studies the internal representation dynamics in LLMs, offering theoretical and empirical insights as well as practical implications for optimizing model design and training strategies. Future work should further investigate the underlying causes of intermediate layer compression and do explicit finetuning to control compression.
this section cite: []

Section: References
Ref_id:b0 Title: Assessing representation quality in selfsupervised learning by measuring eigenspectrum decay Year: (2022)
Ref_id:b1 Title: Understanding intermediate layers using linear classifier probes Year: (2017)
Ref_id:b2 Title: Seq-VCR: Preventing collapse in intermediate transformer representations for enhanced reasoning Year: (2025)
Ref_id:b3 Title: Information theory with kernel methods Year: (2022)
Ref_id:b4 Title: BeIT: Bert pretraining of image transformers Year: (2022)
Ref_id:b5 Title:  Year: (2025)
Ref_id:b6 Title: LLM2Vec: Large language models are secretly powerful text encoders Year: (2024)
Ref_id:b7 Title: Pythia: A suite for analyzing large language models across training and scaling Year: (2023)
Ref_id:b8 Title: Von neumann entropy from unitarity Year: (2019)
Ref_id:b9 Title: Guillotine regularization: Why removing layers is needed to improve generalization in self-supervised learning Year: (2023)
Ref_id:b10 Title: Language models are few-shot learners Year: (2020)
Ref_id:b11 Title: On identifiability in transformers Year: (2020)
Ref_id:b12 Title: Discovering latent knowledge in language models without supervision Year: (2023)
Ref_id:b13 Title: Generative pretraining from pixels. ICML Year: (2020)
Ref_id:b14 Title: Emergence of a highdimensional abstraction phase in language transformers Year: (2025)
Ref_id:b15 Title: Do language models use their depth efficiently? arXiv Year: (2025)
Ref_id:b16 Title: Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning Year: (2025)
Ref_id:b17 Title: Language modeling is compression Year: (2024)
Ref_id:b18 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b19 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b20 Title: The Llama 3 herd of models Year: (2024)
Ref_id:b21 Title: Scalable pre-training of large autoregressive image models Year: (2024)
Ref_id:b22 Title: Not all layers of LLMs are necessary during inference Year: (2024)
Ref_id:b23 Title: Multimodal autoregressive pre-training of large vision encoders Year: (2025)
Ref_id:b24 Title: Assessing the downstream performance of pretrained self-supervised representations by their rank Year: (2023)
Ref_id:b25 Title: Measures of entropy from data using infinitely divisible kernels Year: (2014)
Ref_id:b26 Title: Linear-time sequence modeling with selective state spaces Year: (2024)
Ref_id:b27 Title: When attention sink emerges in language models: An empirical view Year: (2025)
Ref_id:b28 Title: Language models represent space and time Year: (2023)
Ref_id:b29 Title: Training large language models to reason in a continuous latent space Year: (2024)
Ref_id:b30 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b31 Title: Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language Year: (2023)
Ref_id:b32 Title: Exploring concept depth: How large language models acquire knowledge at different layers? arXiv Year: (2024)
Ref_id:b33 Title: The remarkable robustness of LLMs: Stages of inference? arXiv Year: (2024)
Ref_id:b34 Title: Competition-level code generation with alphacode Year: (2022)
Ref_id:b35 Title: Linguistic knowledge and transferability of contextual representations Year: (2019)
Ref_id:b36 Title:  Year: (2019)
Ref_id:b37 Title: Eliciting latent knowledge from quirky language models Year: (2024)
Ref_id:b38 Title: Emergence of separable manifolds in deep language representations. ICML Year: (2020)
Ref_id:b39 Title: Implicit regularization of deep residual networks towards neural odes Year: (2024)
Ref_id:b40 Title: Pointer sentinel mixture models Year: (2017)
Ref_id:b41 Title: MTEB: Massive text embedding benchmark Year: (2022)
Ref_id:b42 Title: Representation learning with contrastive predictive coding Year: (2018)
Ref_id:b43 Title: DINOv2: Learning robust visual features without supervision Year: (2024)
Ref_id:b44 Title: The geometry of categorical and hierarchical concepts in large language models Year: (2024)
Ref_id:b45 Title: The linear representation hypothesis and the geometry of large language models Year: (2024)
Ref_id:b46 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b47 Title: SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability Year: (2017)
Ref_id:b48 Title: The shape of learning: Anisotropy and intrinsic dimensions in transformer-based models Year: (2024)
Ref_id:b49 Title: On measures of entropy and information. Proceedings of the fourth Berkeley symposium on mathematical statistics and probability Year: (1961)
Ref_id:b50 Title: The effective rank: A measure of effective dimensionality Year: (2007)
Ref_id:b51 Title: The underlying structures of self-attention: symmetry, directionality, and emergent dynamics in transformer training Year: (2025)
Ref_id:b52 Title: Learning with kernels: support vector machines, regularization, optimization, and beyond Year: (2018)
Ref_id:b53 Title: Information flow in deep neural networks Year: (2022)
Ref_id:b54 Title: Opening the black box of deep neural networks via information Year: (2019)
Ref_id:b55 Title: An information theory perspective on variance-invariance-covariance regularization Year: (2023)
Ref_id:b56 Title: Maximizing mutual information by a difference of matrix-based entropies Year: (2023)
Ref_id:b57 Title: Frobenius norm minimization for selfsupervised learning Year: (2024)
Ref_id:b58 Title: Neural representational geometry underlies few-shot concept learning Year: (2022)
Ref_id:b59 Title: BERT rediscovers the classical nlp pipeline Year: (2019)
Ref_id:b60 Title: Sensing linear probing performance in joint embedding ssl architectures Year: (2024)
Ref_id:b61 Title: Contrastive multiview coding. ECCV Year: (2020)
Ref_id:b62 Title: Llama 2: Open foundation and finetuned chat models Year: (2023)
Ref_id:b63 Title: The geometry of hidden representations of large transformer models Year: (2023)
Ref_id:b64 Title: Attention is all you need Year: (2017)
Ref_id:b65 Title: The bottom-up evolution of representations in the transformer: A study with machine translation and language modeling objectives Year: (2019)
Ref_id:b66 Title: Diff-eRank: A novel rank-based metric for evaluating large language models Year: (2024)
Ref_id:b67 Title: Efficient streaming language models with attention sinks Year: (2024)
Ref_id:b68 Title:  Year: (2024)
Ref_id:b69 Title: Layer by layer: Uncovering where multi-task learning happens in instructiontuned large language models Year: (2024)
Ref_id:b70 Title: Understanding neural networks with logarithm determinant entropy estimator Year: (2021)
