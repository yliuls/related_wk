Title: The Primacy of Magnitude in Low-Rank Adaptation
Abstract: Low-Rank Adaptation (LoRA) offers a parameter-efficient paradigm for tuning large models. While recent spectral initialization methods improve convergence and performance over the naive "Noise & Zeros" scheme, their extra computational and storage overhead undermines efficiency. In this paper, we establish update magnitude as the fundamental driver of LoRA performance and propose LoRAM, a magnitude-driven "Basis & Basis" initialization scheme that matches spectral methods without their inefficiencies 1 . Our key contributions are threefold: (i) Magnitude of weight updates determines convergence. We prove low-rank structures intrinsically bound update magnitudes, unifying hyperparameter tuning in learning rate, scaling factor, and initialization as mechanisms to optimize magnitude regulation. (ii) Spectral initialization succeeds via magnitude amplification. We demystify that the presumed knowledge-driven benefit of the spectral component essentially arises from the boost in the weight update magnitude. (iii) A novel and compact initialization strategy, LoRAM, scales deterministic orthogonal bases using pretrained weight magnitudes to simulate spectral gains. Extensive experiments show that LoRAM serves as a strong baseline, retaining the full efficiency of LoRA while matching or outperforming spectral initialization across benchmarks.

Section: Introduction
The rise of large pretrained models [1,2,3,4,5] has driven urgent needs for parameter-efficient fine-tuning (PEFT) methods [6,7,8,9,10,11]. Among these, Low-Rank Adaptation (LoRA) [7] stands out for its efficiency, flexibility, and stability. By freezing pretrained weights and injecting trainable low-rank matrices, LoRA enables the update of less than 1% of the parameters, significantly reducing memory and compute costs. Its plug-and-play nature achieves easy integration into diverse models, facilitating model sharing and federated learning [12,13]. Additionally, LoRA helps prevent catastrophic forgetting [14], making it well-suited for continual learning. These advantages have led to its wide adoption in multilingual NLP [15,16,17,18] and multimodal applications [19,20,21,22].
Despite achieved efficiency, the low-rank reparameterization constrains practical performance and convergence [14]. Besides the well-known "representation bottleneck" [23,10,24,25,26,27,28,29], LoRA is highly sensitive to hyperparameters due to the non-convex and non-smooth loss landscape [30]. Effective training relies on careful tuning of rank [31,32,33], scaling factor [34], learning rate [35], initialization strategies [36,37,30], and preconditioning [30,38,39]. Recent works increasingly leverage information from pretrained weights [40,41,42] or task-specific data [43,44,39] to improve the "Noise & Zeros" baseline. Among these, PiSSA [40] pioneers the use of Singular Value Decomposition (SVD) for LoRA initialization, employing spectral components of pretrained Figure 1: We propose LoRAM, a magnitude-driven initialization method that enhances both the convergence and performance of LoRA while maintaining its efficiency. Unlike spectral initialization, which precomputes and stores singular components (U, V, S) [40], LoRAM uses deterministic orthogonal bases and derives scaling from pretrained weight statistics. This elegant simplification is grounded in our analysis of LoRA through a novel lens of magnitude dynamics, where we show that the benefits of spectral values in scaling weight update magnitude can be effectively approximated. weights to significantly enhance convergence and performance. Subsequent works [41,44,42,39,43] extend to diverse matrix decompositions, fostering a wave of knowledge-driven initialization schemes.
While spectral initialization [40,41,44,39,43] showcases considerable promise in convergence and performance, two fundamental challenges persist. First, they introduce complexity by requiring additional matrix decomposition and storage overhead, undermining usage efficiency in resourceconstrained settings [45,25,46] and hindering seamless integration with deep learning libraries [6,47]. Second, their success remains poorly understood. The common justification [40,41,43,48] that spectral components preserve features better than random alternatives lacks theoretical grounding. Although recent works [44,39] suggest that some specific initialization may approximate fullparameter gradients, the non-convex nature of LoRA renders training dynamics unpredictable.
In this paper, we demystify the knowledge-driven intuition behind spectral initialization and reveal that its effectiveness primarily stems from the magnitude scaling of weight updates. We design a minimal "Basis & Basis" initialization strategy, which demonstrates comparable performance without the overhead of SVD operations. Specifically, our key contributions can be summarized as:
• We identify weight update magnitude as a fundamental principle for analyzing and improving LoRA's training dynamics. This principle unifies previously independent factors, such as learning rate, scaling, and initialization, revealing their shared ability to control weight update strength and achieve comparable amplification effects when properly configured.
• We demonstrate that initialization scheme critically shapes LoRA's weight magnitude dynamics. Theoretically, we prove that LoRA naturally produces smaller updates than full fine-tuning, which limits its convergence and expressiveness. Moreover, we show that spectral initialization amplifies updates, providing a principled explanation for its effectiveness beyond knowledge-driven intuition.
• Guided by the magnitude principle, we propose Magnitude-driven Initialization (LoRAM) to make LoRA initialization efficient again. LoRAM employs a logarithmic magnitude factor to retain the benefits of spectral scaling, while directly scaling deterministic orthogonal bases to eliminate the need for decomposition and storage. Extensive experiments on language and vision-language tasks establish LoRAM as a strong and practical baseline, surpassing prior initialization schemes.
2 Magnitude Principle for Characterizing LoRA Dynamics
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b6', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b13', 'b22', 'b9', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b29', 'b29', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b38', 'b39', 'b39', 'b40', 'b43', 'b41', 'b38', 'b42', 'b39', 'b40', 'b43', 'b38', 'b42', 'b44', 'b24', 'b45', 'b5', 'b46', 'b39', 'b40', 'b42', 'b47', 'b43', 'b38']

Section: Preliminaries and Notations
Given a pretrained weight matrix W ∈ R n×m , LoRA [7] reparameterizes the forward pass as
y = W x + W LoRA x = W x + α(BA -B (0) A (0) )x,(1)
where B ∈ R n×r , A ∈ R r×m are trainable low-rank matrices with r ≪ min(n, m), and α scales the update magnitude. The initialization term B (0) A (0) can be absorbed into W for the convenience, as illustrated in Figure 1(b). Given a loss function L, LoRA updates are computed as:
∇ A L = αB ⊤ ∂L ∂y x ⊤ = αB ⊤ (∇ W L), ∇ B L = α ∂L ∂y x ⊤ A ⊤ = α(∇ W L)A ⊤ .(2)
Magnitude metric. To elucidate how LoRA affects the training process, we analyze the dynamics of the parameters A, B, and the resulting weight update W LoRA . Specifically, we define the weight magnitude as ν[W LoRA ] = 1 mn ∥W LoRA ∥foot_0 F , which serves as a central metric in our study. Assuming independent and zero-mean entries [49], the expected weight magnitude is given by E[ν[BA]] = r E[ν [B]] E[ν [A]]. In the asymptotic regime where m and n are large, ν[W LoRA ] is approximated with the variance of W LoRA , and ν[BA] ≈ r ν[B] ν[A]. Our analysis is motivated by the fact that LoRA introduces no change to the pretrained weights initially, i.e., ν[W (0) LoRA ] = 0, while its effect emerges gradually through training. Therefore, we use the term "magnitude" instead of "variance" to highlight the cumulative growth of W LoRA . In Appendix B, we also present a theoretical insight showing that parameter magnitude is a key determinant of LoRA's expressiveness.
this section cite: ['b6', 'b48']

Section: Effect of Hyperparameters on Update Magnitude
Let ∆W LoRA using gradient update rules, leading to the following formulations for the update magnitude 2 :
∆W (t) LoRA = αη B (t) ∇ A L (t) + ∇ B L (t) A (t) + η∇ B L (t) ∇ A L (t) ,(3)
and
ν[∆W (t) LoRA ] ≈ rα 2 η 2 ν[B (t) ]ν[∇ A L (t) ] + ν[∇ B L (t) ]ν[A (t) ] .(4)
These equations indicate the complex interplay of multiple hyperparameters, distinct from the fullparameter updates given by ∆W (t) = -η∇ W L (t) . We investigate the interplay among the learning rate η, scaling factor α, and initialization magnitude, revealing a quantifiable equivalence relationship. Proposition 1 (Parameter Scaling Equivalence). For LoRA layers defined in Eq. (1), consider decomposing the scaling factor α = α ′ α A α B , where α ′ , α A , α B ∈ R + . Under the commonly used optimization frameworks with negligible numerical errors, the following parametrization schemes exhibit dynamical equivalence throughout training: For all iterations t ≥ 0, ∆W
LoRA = ∆ W (t) LoRA(t)
W (t) LoRA = W (t)and
LoRA , where Ã(t) , B(t) and W (t) represent the re-parameterized versions.
this section cite: []

Section: Original SGD Adam
Representation αBAx α ′ B Ãx α ′ B Ãx Initialization A (0) = Ainit, B (0) = Binit Ã(0) = αAAinit, B(0) = αBBinit Ã(0) = αAAinit, B(0) = αBBinit Learning Rates ηA > 0, ηB > 0 η Ã = α 2 A ηA, η B = α 2 B ηB η Ã = αAηA, η B = αBηB
Remarks. This equivalence underscores how hyperparameters collectively regulate update magnitude, effectively reducing the search space for optimal configurations. A striking implication is that, increasing η B in LoRA+ [35] is identical to scaling α in RsLoRA [34] under the "Noise & Zeros" initialization, highlighting the critical role of initialization magnitude in shaping LoRA's training dynamics, which is rarely discussed in prior works. For the non-zero initialization, it is advisable to first adjust the initialization magnitudes and learning rates for moderate improvements, as modifying α inherently combines the effects of both and may result in drastic and unpredictable changes.
To demonstrate the joint effect of hyperparameters on LoRA dynamics, we conduct a controlled experiment using a 5-layer MLP with "Noise & Zeros" initialized LoRA layers, setting the intermediate dimension to 400 and the LoRA rank to 25. The network is trained on synthetic data under various hyperparameter settings, consequently using SGD and Adam optimizers with η = 5 × 10 -5 . As shown in Figure 2(a), all settings with α = 16 result in identical loss trajectories and parameter evolution, confirming the theoretical predictions. In contrast, weight updates deviate significantly when using α = 1 with η scaled by 4, indicating that equivalence holds only under specific rules.
this section cite: ['b34', 'b33']

Section: Magnitude Limitation Rooted in Low-Rank Structure
Guided by the established equivalence framework, we fix α = 1 to eliminate the interference of scaling factors, which is the most commonly-used configuration in practical implementation. In the following, we prove initialization magnitudes and other factors critically influence the weight update.
E[⟨A (t) , ∇ A L (t) ⟩] = E[⟨B (t) , ∇ B L (t) ⟩] = 0. Under these conditions, the parameter magnitudes ν t = E[ν[A (t) ]], E[ν[B (t) ]]
T evolve as a linear dynamical system. Its exponential solution admits the linearized approximation under small-η regime:
ν t = I + 0 γ B γ A 0 ν t-1 ≈ σ 2 A + tγ B σ 2 B σ 2 B + tγ A σ 2 A ,(5)
where
γ A = mη 2 σ 2 L , γ B = nη 2 σ 2 L .
This further yields the evolution of weight update magnitude:
ν[W (t) LoRA ] ≈ k 1 γt + O(γ 2 t 2 ), where γ = η 2 σ 2 L , k 1 = r(mσ 4 A + nσ 4 B ).(6)
Remarks. This analysis uncovers essential properties of LoRA initialization and optimization trajectory. First, since γ A and γ B are very small values (≪ 1), the magnitudes of parameters A (t) and B (t) remains nearly unchanged throughout training, potentially constraining the representation capacity of the learned model. Moreover, unlike full-parameter tuning, which evolves at a linear rate of γ, the low-rank structure introduces a proportional factor k 1 , significantly slowing updates. For instance, the naive "Noise & Zeros" initialization yields k 1 = r m , while the dimension m in large models like LLaMA [3] is in the thousands or more. Despite the quadratic term accelerates growth, small gradients may temper this effect in the later training stages. 20 . We explore a regular MLP with the same magnitude denoted as "Linear", and a group of networks with larger initialization magnitudes. Notably, while the loss decreases significantly, the magnitudes of A and B remain nearly unchanged throughout training. The magnitude evolution of W reveals that the basic LoRA with theoretically k 1 = 1  16 , grows substantially slower than the regular MLP. Increasing the initialization magnitude effectively accelerates the growth of W , aligning with our theoretical analysis. Integrating analyses in this section, we derive a magnitude principle for LoRA development:
A valid improvement to LoRA convergence will enhance weight update magnitude ν[W LoRA ].
As shown in Proposition 2, magnitude principle could unify and explain improvement factors in existing works, including the learning rate, scaling factor, gradients, rank and initialization schemes.
this section cite: ['b2']

Section: Demystifying Spectral Gains with Magnitude Principle
The sheer scale of modern neural networks complicates the determination of optimal LoRA initialization across layers. Drawing inspiration from recent spectral initialization methods [40,41,43,44,39], which have demonstrated improved convergence and task performance, we reinterpret their effectiveness through the lens of the magnitude principle and introduce a magnitude-driven initialization Due to its concave nature, we approximate the spectral gain factor using a logarithmic function.
method called LoRAM. Notably, these methods requires extra SVD computations and storage, leading to increased resource overhead and implementation complexity. In contrast, LoRAM mitigates these drawbacks and achieves even better performance. In the following, we take the seminal work PiSSA [40] as a representative baseline and ablate other methods in experiments (see Section 4.3).
this section cite: ['b39', 'b40', 'b42', 'b43', 'b38', 'b39']

Section: Magnitude Gain in Spectral Initialization
The PiSSA method [40] initializes LoRA using the spectral decomposition of pretrained weight matrix W = U SV ⊤ , which has a rank of R[W ]. The spectral initialization is defined as:
A (0) = A SVD = √ S r V ⊤ :,:r , B (0) = B SVD = U :,:r √ S r ,(7)
where S r ∈ R r×r contains the top-r singular values, and U ∈ R n×n , V ∈ R m×m are the left and right singular vector matrices. While prior works [40,41,43,48] attribute PiSSA's success to its ability to preserve principal components, we show that its key advantage lies in singular value weighting. By redistributing dominant variance into the initialization, PiSSA facilitates adaptive magnitude updates across layers, accelerating convergence.
Consider the statistics of the top-r singular values, we define the spectral concentration factor as:
ρ[r] ≜ E r [s] 2 E R[W ] [s 2 ] = ( 1 r r i=1 s i) 2 1 R[W ] R[W ] i=1 s 2 i . (8
)
This factor captures the concentration of energy in the top-r singular values. We then reformulate
ν(A SVD ) = E r [s]ν[V :,:r ] = nρ[r]ν[W ] mR[W ] , ν(B SVD ) = E r [s]ν[U :,:r ] = mρ[r]ν[W ] nR[W ] . (9) Essentially, ρ[r]
acts as a scaling factor that redistributes variance from the pretrained weight matrix, influencing the magnitude of updates during training. Since ρ[r] monotonically decreases with r, its impact is more pronounced for smaller r, making it particularly relevant for LoRA applications.
this section cite: ['b39', 'b39', 'b40', 'b42', 'b47']

Section: Spectral Gain Factor.
Taking the above magnitudes into the dynamics in Eq. ( 6) further derives:
k 1 = Q[r](m + n)ν[W ], 0 ≤ Q[r] ≜ ρ[r]r R[W ] ≤ 1.(10)
Given that ν[W ] ∼ O(min( 1 m , 1 n )), this results in a gain factor of at least Q[r], which we term as the "spectral gain factor". As shown in Figure 3, the spectral gain factor Q[r] exhibits bounded variation in [0, 1] with characteristic concavity, which can be formally derived via Jensen's inequality. Although SVD components are not completely independent preventing the theoretical monotonic increase in Q[r], this concavity also suggests that the gain effect is more pronounced when r is small, reinforcing the effectiveness of spectral initialization for LoRA in the parameter-efficient manner.
this section cite: []

Section: Efficient Magnitude-driven Initialization with LoRAM
We propose LoRAM to achieve similar magnitude update rate in Eq. ( 10) like PiSSA while eliminating spectral computation. As depicted in Algorithm 1, LoRAM initializes the parameter matrices as:
A (0) = A LoRAM = β • Φ ⊤ m , B (0) = B LoRAM = β • Φ n , β = Q[r]•ν[W ] ν[ΦnΦ ⊤ m ] 1 4 .(11)
Algorithm 1 LoRAM Initialization Procedure Input: Pretrained weight W ∈ R n×m , target rank r Output: Initialized parameters A (0) , B (0) , W Φ n , Φ m ← get_basis(n, r), get_basis(m, r) ▷ Generate basis matrices, e.g., Eq (12)
β ← log min(n,m) (r)•ν[W ] ν[ΦnΦ ⊤ m ] 1/4 ▷ Compute magnitude gain factor B (0) , A (0) , W ← β • Φ n , β • Φ ⊤ m , W -β 2 • Φ n Φ ⊤ m ▷ Initialize parameters
Here Φ n and Φ m denote the first r columns of an nand m-dimensional orthogonal basis matrices, respectively. Given that ν[Φ n ] = 1 n and ν[Φ m ] = 1 m , LoRAM achieves the similar magnitude as PiSSA with
k 1 = Q[r](m + n)ν[W ]. We can also derive ν[B (0) A (0) ] = Q[r]ν[W ]
, implying LoRAM inherently ensures numerical stability and moderate corrections to the pretrained weight.
this section cite: []

Section: Logarithmic Gain Factor.
Due to the concave nature of Q[r], we approximate its analytical form using an asymptotic expansion: Q[r] ≈ log min(n,m) (r). As illustrated in Figure 3, this logarithmic function effectively captures the monotonic increase nature, providing a predictable improvement than LoRA and PiSSA particularly when using a small rank r.
this section cite: []

Section: Deterministic Basis Matrix.
To eliminate the need for storing initialization buffers, we adopt an analytic approach instead of random initialization. Specifically, we employ the Discrete Sine Transform (DST) basis due to its simplistic mathematical definition:
Φ m [i, j] = 2 m+1 sin (i+1)(j+1)π m+1 , 0 ≤ i, j < m. (12
)
This formulation constructs orthogonal matrices of arbitrary dimensions, ensuring reproducibility across different devices while providing provable statistical properties: E[Φ m ] = 0 and ν[Φ m ] = 1 m . One may wonder if randomness is required for initialization, we find it unnecessary in LoRA. In fact, DST even slightly outperforms random strategies in our ablation experiments (see Section 4.3).
Efficiency and Compatibility. Since β and Φ avoid complex matrix operations, LoRAM retains the efficiency and storage footprint of naive LoRA. As it only modifies initialization, LoRAM remains plug-and-play, integrating seamlessly into any pipeline that supports standard LoRA. This is especially valuable for modern large models built on fixed and highly optimized frameworks. In contrast, other initialization methods require costly preprocessing, such as matrix generation [44,43] or decomposition [40,42,41], which complicates adoption in standard workflows.
this section cite: ['b43', 'b42', 'b39', 'b41', 'b40']

Section: Experiments
We conduct comprehensive experiments to evaluate LoRAM efficiently implemented via the PEFT library [6]. Following conventional settings [40,41,35], we assess performance on language tasks and extend the evaluation to vision-language tasks, demonstrating LoRAM's generalization across diverse models and modalities. All experiments are run on servers with 8 NVIDIA H800 GPUs.
Baselines. While extensive research on LoRA has explored aspects like structural modifications and rank control, these directions are largely orthogonal to our focus on hyperparameter analysis within the naive LoRA framework. In line with this, we compare LoRAM with the naive LoRA (ICLR 2022) [7], as well as several representative hyperparameter tuning strategies. We first consider weight-driven initialization (marked " §"), including PiSSA (NeurIPS 2024) [40], which uses the top-r singular vectors and values of pre-trained weights; MiLoRA (NAACL 2025) [41], which utilizes the last r singular vectors and values; and OLoRA [42], which applies orthogonal initialization via QR decomposition. All these methods adopt a fixed scaling factor α = 1. We then include RsLoRA [34] (marked " †"), which enhances performance by setting α = √ r, and LoRA+ (ICML 2024) [35] (marked " ‡"), which increases the learning rate with the recommended η B = 4η A . We also evaluate data-driven initialization in the ablation study, including LoRA-GA (NeurIPS 2024) [52] and CorDA (NeurIPS 2024) [43], which require extra pipeline to leverage training data information. Most of these baselines have been integrated and validated in the PEFT library.
this section cite: ['b5', 'b39', 'b40', 'b34', 'b6', 'b39', 'b40', 'b41', 'b33', 'b34', 'b50', 'b42']

Section: Evaluating the Performance on Natural Language Tasks
Nature Language Generation (NLG). As shown in Table 1, we conduct supervised fine-tuning of LLaMA 2-7B [3] on math, coding, and commonsense reasoning tasks. Our setup strictly follows PiSSA [40], using the AdamW optimizer with a batch size of 128, a learning rate of 2 × 10 -5 , a warmup ratio of 0.03, and no weight decay. All experiments are performed on subsets containing 100K data points for one epoch to minimize training overhead. For math tasks, the model is tuned on MetaMathQA [53] and evaluated on GSM8K [54] and MATH [53] validation sets. For coding tasks, we use CodeFeedback [55] as training dataset, with evaluations on HumanEval [56] and MBPP [57].
For commonsense tasks, model is tuned on Commonsense170K [58], and we report averaged accuracy on eight sub-datasets. The results in Table 1 demonstrate that LoRAM consistently outperforms LoRA variants across diverse tasks and rank settings, without requiring matrix decomposition.
this section cite: ['b39', 'b51', 'b52', 'b51', 'b53', 'b54', 'b55', 'b56']

Section: Nature Language Understanding (NLU).
We evaluate the NLU performance by fine-tuning the DeBERTa-v3-base model [50] with a rank of 8 on eight tasks in the GLUE benchmark [59]. We utilize scripts from the Transformers Library [47] to ensure a fair comparison. All methods are trained with a learning rate of 1 × 10 -4 for 3 training epochs, except for MRPC, which uses 5 epochs due to its smaller size. We report overall matched and mismatched accuracy on MNLI, Matthew's correlation on CoLA, Pearson correlation on STS-B, and accuracy on the other datasets. As shown in Table 2, LoRAM achieves competitive performance against PiSSA across most tasks.
this section cite: ['b49', 'b57', 'b46']

Section: Evaluating the Performance on Vision-Language Tasks
Text-to-image synthesis. We adapt the advanced FLUX.1-12B [51] to address the image customization task, implementing LoRA, PiSSA, and LoRAM under identical configurations: a learning rate of 1 × 10 -4 , a batch size of 1 and 1,000 iterations. We set the rank to 8, optimizing 9.3 million  parameters while maintaining computational efficiency on a single GPU. The training data and prompt template adhere to DreamBooth's protocol [22]. Qualitative results in Figure 4 demonstrate that LoRAM exhibits marginally superior performance in detail fidelity compared to PiSSA.
Image-to-text generation. Following the pipeline of LLaVA [60], we employ CLIP-ViT-L/14 [61] as the visual encoder, Vicuna-13B [62] as the text decoder, and a new visual resampler [63] as the connector. In the pre-training stage, we fine-tune only the perceiver resampler using the CC-595K dataset [60] for one epoch. During the subsequent instruction-tuning stage, we fine-tune both Vicuna and the resampler using a 656K mixture dataset [60]. The learning rate is set to 2 × 10 -5 , and the batch size is 128. We follow the official implementation and use a rank of 64. As shown in Table 3, LoRAM achieves favorable performance across multiple multimodal benchmarks.
Training curves. We provide representative training loss curves of diverse initialization methods in Figure 5. It can be noticed that LoRAM is able to have a faster convergence rate in the early stages compared to other LoRA variants and incur smaller losses in the end.
this section cite: ['b21', 'b58', 'b59', 'b60', 'b61', 'b58', 'b58']

Section: Ablating the Magnitude Principle in LoRA improvements
We conduct ablation experiments on LLaMA-2-7B [3] under the NLG setting, focusing on the effect of magnitude gain factor, the choice of basis matrix, and the validation of the magnitude principle.
this section cite: []

Section: Magnitude Gain Factor.
As shown in Table 4, we first evaluate different values of Q[r] and observe that increasing its value slightly improves performance. We further introduce a "tracking mode" that adjusts β in Algorithm 1 based on the initialization magnitudes from reference methods. Specifically,
we set β = V[BrefAref] V[ΦnΦ ⊤ m ]
, where B ref and A ref are initialization matrix using weight-driven [40,41] or data-driven approaches [43,44]. Under this mode, LoRAM essentially matches the performance of prior methods, confirming that magnitudes govern its performance. We also observe that PiSSA,  which selects the top-r singular values, outperforms MiLoRA, which selects the last r singular values. This indicates that leveraging the large dominant singular values enhances performance.
Basis Matrix. We find that the choice of basis matrix generally has limited impact. For instance, replacing the DST basis with a random orthogonal or Gaussian matrix just results in only minor performance degradation. In tracking mode, substituting the SVD-derived basis with DST does not significantly impact performance. A notable exception is LoRA-GA [44], which approximates the full-parameter gradient at initialization. Nonetheless, we emphasize that tracking mode fails not due to incorrectness of magnitude principle, but because the LoRA-GA matrix form maximizes gradient magnitudes of Eq. ( 2), making it irreplaceable by alternatives and validating magnitude principle 3 .
this section cite: ['b39', 'b40', 'b42', 'b43', 'b43']

Section: Upper Bound of Magnitude Scaling.
While increasing update magnitude generally improves performance, the benefit is not unlimited. For example, applying RsLoRA to LoRA-GA yields a clear gain at rank 8, but the improvement diminishes and may even reverse at rank 128. This suggests that magnitude scaling should be applied conservatively at higher ranks, since larger ranks inherently amplify updates, as demonstrated in our Proposition 2. Given that data-driven methods involve costly gradient and SVD computations, we recommend LoRAM with RsLoRA as a more efficient and scalable alternative for accelerating LoRA convergence and performance.
this section cite: []

Section: Conclusion
In this paper, we explore the magnitude principle of Low-Rank Adaptation (LoRA) and introduce a novel magnitude-driven initialization strategy, LoRAM, that bridges the gap between efficiency and performance. Our work demystifies the prevailing awareness surrounding spectral initialization methods, demonstrating that their success primarily stems from the amplification of weight update magnitudes. By focusing on magnitude regulation as the key driver of convergence, we provide a unified perspective that connects seemingly disparate hyperparameter adjustments, such as learning rate, scaling factor, and initialization schemes, under a single framework.
Limitations and Future Work. Despite the advancements introduced in this work, several challenges remain open for future research. First, LoRAM mimics spectral initialization magnitudes rather than seeking optimal ones; exploring alternative strategies could yield further gains. Additionally, different layers may benefit from tailored magnitude settings, motivating joint optimization with learning rate and rank. Finally, our work does not explicitly address optimization dynamics and convergence properties. These directions remain valuable for advancing parameter-efficient fine-tuning.
this section cite: []

Section: References
Ref_id:b0 Title: Language models are few-shot learners Year: (2020)
Ref_id:b1 Title: OpenAI Team. Gpt-4 technical report Year: (2023)
Ref_id:b2 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b3 Title: Qwen technical report Year: (2023)
Ref_id:b4 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b5 Title: Peft: state-of-the-art parameterefficient fine-tuning methods Year: (2023)
Ref_id:b6 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b7 Title: The power of scale for parameter-efficient prompt tuning Year: (2021)
Ref_id:b8 Title: Towards a unified view of parameter-efficient transfer learning Year: (2022)
Ref_id:b9 Title: Krona: Parameter efficient tuning with kronecker adapter Year: (2023)
Ref_id:b10 Title: Parameter-efficient fine-tuning for foundation models Year: (2025)
Ref_id:b11 Title: Slora: Federated parameter efficient fine-tuning of language models Year: (2023)
Ref_id:b12 Title: FeDeRA: Efficient fine-tuning of language models in federated learning leveraging weight decomposition Year: (2024)
Ref_id:b13 Title: Lora learns less and forgets less Year: (2024)
Ref_id:b14 Title: Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning Year: (2022)
Ref_id:b15 Title: Parameter-efficient fine-tuning of large-scale pre-trained language models Year: (2023)
Ref_id:b16 Title: A comparative study between full-parameter and lora-based fine-tuning on chinese instruction data for instruction following large language model Year: (2023)
Ref_id:b17 Title: Lora land: 310 fine-tuned llms that rival gpt-4, a technical report Year: (2024)
Ref_id:b18 Title: mplug-owl: Modularization empowers large language models with multimodality Year: (2023)
Ref_id:b19 Title: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning Year: (2024)
Ref_id:b20 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b21 Title: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation Year: (2023)
Ref_id:b22 Title: Fedpara: Low-rank hadamard product for communication-efficient federated learning Year: (2022)
Ref_id:b23 Title: Delta-lora: Fine-tuning high-rank parameters with the delta of low-rank matrices Year: (2022)
Ref_id:b24 Title: Parameterefficient fine-tuning with discrete fourier transform Year: (2022)
Ref_id:b25 Title: High-rank updating for parameter-efficient fine-tuning Year: (2024)
Ref_id:b26 Title: Relora: High-rank training through low-rank updates Year: (2024)
Ref_id:b27 Title: Chain of lora: Efficient fine-tuning of language models via residual learning Year: (2024)
Ref_id:b28 Title: Breaking the low-rank bottleneck in lora optimization Year: (2024)
Ref_id:b29 Title: On the crucial role of initialization for matrix factorization Year: (2025)
Ref_id:b30 Title: The expressive power of low-rank adaptation Year: (2024)
Ref_id:b31 Title: Adaptive budget allocation for parameter-efficient fine-tuning Year: (2023)
Ref_id:b32 Title: Sparse low-rank adaptation of pre-trained language models Year: (2023)
Ref_id:b33 Title: A rank stabilization scaling factor for fine-tuning with lora Year: (2023)
Ref_id:b34 Title: Lora+: Efficient low rank adaptation of large models Year: (2024)
Ref_id:b35 Title: Asymmetry in low-rank adapters of foundation models Year: (2024)
Ref_id:b36 Title: The impact of initialization on lora finetuning dynamics Year: (2024)
Ref_id:b37 Title: Riemannian preconditioned lora for fine-tuning foundation models Year: (2024)
Ref_id:b38 Title: One-step full gradient suffices for low-rank fine-tuning, provably and efficiently Year: (2025)
Ref_id:b39 Title: Pissa: Principal singular values and singular vectors adaptation of large language models Year: (2024)
Ref_id:b40 Title: Harnessing minor singular components for parameter-efficient llm finetuning Year: (2025)
Ref_id:b41 Title: Olora: Orthonormal low-rank adaptation of large language models Year: (2024)
Ref_id:b42 Title: Corda: Context-oriented decomposition adaptation of large language models Year: (2024)
Ref_id:b43 Title: Lora-ga: Low-rank adaptation with gradient approximation Year: (2024)
Ref_id:b44 Title: Vector-based random matrix adaptation Year: (2024)
Ref_id:b45 Title: Qlora: Efficient finetuning of quantized llms Year: (2023)
Ref_id:b46 Title: Transformers: State-of-the-art natural language processing Year: (2020)
Ref_id:b47 Title: A survey on lora of large language models Year: (2025)
Ref_id:b48 Title: Delving deep into rectifiers: Surpassing human-level performance on imagenet classification Year: (2015)
Ref_id:b49 Title: Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing Year: (2023)
Ref_id:b50 Title: Lora-pro: Are low-rank adapters properly optimized? Year: (2025)
Ref_id:b51 Title: Metamath: Bootstrap your own mathematical questions for large language models Year: (2024)
Ref_id:b52 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b53 Title: Integrating code generation with execution and refinement Year: (2024)
Ref_id:b54 Title: Codegeex: A pre-trained model for code generation with multilingual evaluations on humaneval-x Year: (2023)
Ref_id:b55 Title: Program synthesis with large language models Year: (2021)
Ref_id:b56 Title: LLM-adapters: An adapter family for parameter-efficient fine-tuning of large language models Year: (2023)
Ref_id:b57 Title: Glue: A multi-task benchmark and analysis platform for natural language understanding Year: (2018)
Ref_id:b58 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b59 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b60 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2024)
Ref_id:b61 Title: Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images Year: (2024)
Ref_id:b62 Title: Analyzing and improving the image quality of stylegan Year: (2020)
Ref_id:b63 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b64 Title: Generalized one-shot domain adaptation of generative adversarial networks Year: (2022)
Ref_id:b65 Title: Towards consistent video editing with text-to-image diffusion models Year: (2023)
Ref_id:b66 Title: Top-down compression: Revisit efficient vision token projection for visual instruction tuning Year: (2025)
Ref_id:b67 Title: Joyagentsr1: Joint evolution dynamics for versatile multi-llm agents with reinforcement learning Year: (2025)
Ref_id:b68 Title: Low-rank adaptation for foundation models: A comprehensive review Year: (2024)
Ref_id:b69 Title: Mini-ensemble low-rank adapters for parameter-efficient fine-tuning Year: (2024)
Ref_id:b70 Title: Loran: Improved low-rank adaptation by a non-linear transformation Year: (2024)
Ref_id:b71 Title: Neat: Nonlinear parameter-efficient adaptation of pre-trained models Year: (2024)
Ref_id:b72 Title: Batched low-rank adaptation of foundation models Year: (2024)
Ref_id:b73 Title: Weight-decomposed low-rank adaptation Year: (2024)
Ref_id:b74 Title: Practical tips for finetuning llms using lora Year: (2023)
Ref_id:b75 Title: Lora dropout as a sparsity regularizer for overfitting control Year: (2024)
Ref_id:b76 Title: Lora-fa: Memory-efficient low-rank adaptation for large language models fine-tuning Year: (2023)
Ref_id:b77 Title: Lora-xs: Low-rank adaptation with extremely small number of parameters Year: (2024)
Ref_id:b78 Title: Loftq: Lora-fine-tuning-aware quantization for large language models Year: (2024)
Ref_id:b79 Title: Understanding the difficulty of training deep feedforward neural networks Year: (2010)
Ref_id:b80 Title: One initialization to rule them all: Fine-tuning via explained variance adaptation Year: (2024)
Ref_id:b81 Title: Lora training in the NTK regime has no spurious local minima Year: (2024)
Ref_id:b82 Title: A kernel-based view of language model fine-tuning Year: (2023)
