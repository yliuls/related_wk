Title: ModHiFi: Identifying High Fidelity predictive components for Model Modification
Abstract: Open weight models, which are ubiquitous, rarely provide access to their training data or loss function. This makes modifying such models for tasks such as pruning or unlearning, which are constrained by this unavailability, an active area of research. Existing techniques typically require gradients or ground-truth labels, rendering them infeasible in settings with limited computational resources. In this work, we investigate the fundamental question of identifying components that are critical to the model's predictive performance, without access to either gradients or the loss function, and with only distributional access such as synthetic data. We theoretically demonstrate that the global error is linearly bounded by local reconstruction errors for Lipschitz-continuous networks such as CNNs and well-trained Transformers (which, contrary to existing literature, we find exhibit Lipschitz continuity). This motivates using the locally reconstructive behavior of component subsets to quantify their global importance, via a metric that we term Subset Fidelity. In the uncorrelated features setting, selecting individual components based on their Subset Fidelity scores is optimal, which we utilize to propose ModHiFi, an algorithm for model modification that requires neither training data nor access to a loss function. ModHiFi-P, for structured pruning, achieves an 11% speedup over the current state of the art on ImageNet models and competitive performance on language models. ModHiFi-U, for classwise unlearning, achieves complete unlearning on CIFAR-10 without fine-tuning and demonstrates competitive performance on Swin Transformers. 2 * Author primarily contributed to this work before joining Google. 2 Our code is available at https://github.com/DhruvaKashyap/modhifi 39th Conference on Neural Information Processing Systems (NeurIPS 2025).related to privacy and security [88], and also the use of synthetic data, which has become critical in a variety of language modeling settings [10,72]. Thus, we address the challenging problem of altering well-trained models without training data or the loss function, and only with distributional access to the original training distribution in the form of synthetic data, focusing specifically on structured pruning and classwise unlearning.Modifying open weight models without the loss function and only synthetic data requires answering a fundamental question: which components in a model contribute significantly to its predictive performance 3 ? However, most methods that identify critical components for specific modifications (e.g., pruning) cannot be applied to others (e.g., unlearning) [53], often require expensive fine-tuning, and are architecture-specific. Moreover, most methods utilize gradients to assess the impact of a component on the loss objective, which is not feasible in the absence of the loss function and the training data. While the LLM pruning literature uses calibration datasets to mitigate the problem of the absence of datasets [2,46], the problem of achieving sparsity in vision models without original training data is hard and unsolved [27]. Moreover, the issue of performing classwise unlearning without access to the original training data has not been addressed [32,53].Towards enabling the modification of well-trained open weight models amidst these challenges, we make the following contributions:(C1) Local-to-Global with Lipschitzness. An open question is the extent to which local model modifications impact the predictive performance of the model. In the absence of loss functions and training sets, estimating the impact of component modification by using gradients (as done in [32,43,46]) is infeasible. To address this, in Theorem 3.6, we show that for Lipschitz continuous networks, the reconstruction error at the final layer is at most linear in the local reconstruction errors. Moreover, contrary to the assertion that transformers are not Lipschitz continuous [60], in Corollary B.4, we show that this is not the case for well-trained transformers, allowing us to apply Theorem 3.6 to not just CNNs, but well-trained ViTs and LLMs as well.(C2) Identifying Subsets of Important Components. Contrary to prior work, which usually infers saliencies for single components, we propose measuring the importance of sets of components to understand the cumulative effects of groups of components on a model's predictive performance.Leveraging Theorem 3.6, we propose Subset Fidelity, which quantifies the extent to which a subset of components can reconstruct the output after modifying their weights. However, computing optimal subsets is NP-complete, motivating us to compute Subset Fidelity scores for singleton sets. Theorem 3.9 establishes that selecting singletons with the highest subset fidelity scores is optimal when the features are uncorrelated.(C3) Modifying Models with ModHiFi-X. Motivated by Theorem 3.9, we propose the ModHiFi algorithm, which uses the subset fidelity of singletons to modify models for pruning and classwise unlearning; the algorithm identifies important components using the singleton scores, and removes them (for classwise unlearning, ModHiFi-U) or retains them (for structured pruning, ModHiFi-P). We demonstrate that ModHiFi-P achieves state-of-the-art speedup for ImageNet models and consistently competes with current baselines for language models. For classwise unlearning, ModHiFi-U achieves complete unlearning on all CIFAR-10 classes without finetuning and is competitive with baselines on Swin-Transformers that require fine-tuning. When allowing for a similar fine-tuning budget as said baselines, ModHiFi-U outperforms, particularly when given access to training data. These empirical results demonstrate the practical effectiveness of Subset Fidelity.2 Background, Setup, and Related WorkIn this section, we review the background relevant to our study, establish the notation, and formalize the model modification problem. We also unify Convolutional Networks (CNNs) and Transformers under a single abstraction that underpins our theoretical results in Section 3. c l in i=1 A l ci (X), where A l ci represents either the spatial convolution (Equation (CONV)) or the token-wise linear projection (Equation (LIN)). This decomposition is central to our analysis of component importance. This additive structure allows us to analyze component fidelity in an architecture-agnostic manner. Modifying Open Weight Models without Training Data or the Loss Function via Distributional AccessWe formally define model modification as the process of selectively altering parameters of a pretrained model, without retraining from scratch, to satisfy constraints such as efficiency, privacy, or safety [27,32,63,65]. This includes tasks including structured pruning, unlearning [37], debiasing [31], continual or life-long learning [21,61]. A major impediment in real-world modification is the unavailability of the original training data and loss function [27,53]. To address this, we operate under the constraint of distributional access, specifically utilizing synthetic data [10,72], to proxy the underlying data distribution without requiring the original corpus.These considerations motivate the central question addressed in this work: Can we effectively modify trained models, for tasks such as structured pruning or unlearning, using only distributional access provided through synthetic data?

Section: Introduction
Modern deep learning has made significant strides in a wide variety of tasks, such as classification [35,36], image generation [22], and natural language processing [49]; moreover, well-trained open weight models for such tasks are easily accessible. However, significant challenges remain in their deployment, such as inference in resource-constrained settings [59,68], inference with unbalanced or biased data [28,29], and interpretable inference [93]. These challenges have increased interest in methods that modify the parameters of well-trained models to alter their behavior [61,63,65]. These methods include pruning [27], classwise unlearning [31,34,65], and debiasing [50,65], among other model modifications. Moreover, recent work has studied model modification in the setting where the original training data and loss function are unavailable [53]; this is motivated by concerns
this section cite: ['b36', 'b37', 'b22', 'b50', 'b60', 'b69', 'b28', 'b29', 'b94', 'b62', 'b64', 'b66', 'b27', 'b31', 'b35', 'b66', 'b51', 'b66', 'b54']

Section: Background and Notation
Notation Let [p] = {1, . . . , p} for p ∈ N. We denote vectors by v ∈ R n with entries v i , and matrices by B ∈ R n×m with rows b ⊤ i and columns B :,j . The vectors 1 d and 0 d denote the all-ones and all-zeros vectors in R d , respectively. We use ∥v∥ 2 for the Euclidean norm. For matrices C, D, the inner product is ⟨C, D⟩ = Tr(C ⊤ D), the Frobenius norm is ∥C∥ = ⟨C, C⟩, and the spectral norm ∥C∥ 2 is the largest singular value. For index sets A ⊆ [n] and B ⊆ [m], C[A, B] denotes the submatrix of C defined by these indices. Expectations of a random variable X are written E X [•], omitting the subscript when clear from context.
this section cite: []

Section: 2D Convolution
Consider the l-th layer of a convolutional network. It transforms an input (the output of preceding layers)
Φ l (X) ∈ R c l in ×h l-1 ×w l-1 into output Y l (X) ∈ R c l out ×h l ×w l
. The layer is parameterized by a weight tensor W l ∈ R c l out ×c l in ×k l ×k l
. Each output channel c ∈ [c l out ] is computed as the sum of convolved input channels:
Y l c (X) = c l in i=1 Φ l i (X) ⋆ W l ci = c l in i=1 A l ci (X),(CONV)
where ⋆ denotes the standard 2D convolution. We define A l ci (X) := Φ l i (X) ⋆ W l ci (∈ R h l ×w l ) as the input contribution from channel i to output channel c. For notational simplicity, we omit explicit bias terms and stride/padding specifications, as our analysis generalizes to these standard configurations without loss of generality.
Transformers Transformer blocks consist of Multi-Head Attention (MHA) and Feed-Forward Networks (FFN), with pre-normalization (LayerNorm or RMSNorm) [2,3]. Our analysis focuses on the FFN; we leave attention-specific analysis for future work. Let the input to the l-th layer be ϕ l (X) ∈ R T ×d , where T is sequence length and d is model dimension. The FFN comprises two linear transformations, W l U ∈ R d×d ff and W l D ∈ R d ff ×d , and an elementwise nonlinearity σ(•) and it's output, FFN l (ϕ l (X)) = σ(ϕ l (X)W l U ) W l D . Defining the intermediate activation Φ l (X) := σ(ϕ l (X)W l U ) ∈ R T ×d ff , the contribution from intermediate neuron i ∈ [d ff ] to output coordinate c ∈ [d] is:
A l ci (X) := Φ l :,i (X) W l D,ci .(LIN)
Unified Notation To unify these architectures, we define a common abstraction used in our theoretical results. Let N θ = f L • • • • • f 1 be a network composed of L layers. Each layer l maps an input Φ l (X) to an output Y l (X). Crucially, for both CNNs and Transformers, the output channel c can be decomposed as a sum of atomic input contributions: Y l c (X) = Formulating Model Modification Let θ ⋆ ∈ R D be the parameters of a well-trained model. We seek a modification mask m ⋆ ∈ M (where M defines permissible modifications, e.g., binary masks for pruning) to produce modified parameters θ E = θ ⋆ ⊙ m ⋆ . Given data distributions {D i } K i=1 and weights α ∈ R K , the optimal modification mask is defined as:
m ⋆ = arg min m∈M i α i E X∼Di [L(N θ ⋆ ⊙m (X))] .(MODIFY)
We instantiate this framework for two distinct tasks:
Structured Pruning The goal is to maximize performance subject to sparsity. Let the parameters be partitioned into G disjoint structured groups {G g } G g=1 (e.g., filters, channels, rows, or columns), with θ ⋆ = (θ ⋆ G1 , . . . , θ ⋆ G G ). The admissible set enforces a sparsity budget B, M SP := {m ∈ R D | ∃ z ∈ {0,
1} G , δ ∈ R D s.t. m j = z g δ j ∀j ∈ G g , G g=1 z g ≤ B}.
Structured pruning is recovered from (MODIFY) by setting K = 1, D 1 = D (original task distribution), α 1 = 1, and M = M SP , yielding m ⋆ = arg min
m∈M SP E D [L(N θ ⋆ ⊙m (X))] .
this section cite: ['b1', 'b2']

Section: (STRUCT-PRUNE)
Classwise Unlearning The goal is to degrade performance on a forget distribution D f while preserving performance on a retain distribution D r . We impose no additional structural constraints on the modification and set M U = R D . Classwise unlearning is obtained from (MODIFY) by setting K = 2, (D 1 , D 2 ) = (D r , D f ), (α 1 , α 2 ) = (1, -1), and M = M U , yielding m ⋆ = arg min
m∈M U E X∼Dr [L(N θ ⋆ ⊙m (X))] -E X∼D f [L(N θ ⋆ ⊙m (X))].(UNLEARN)
Our core challenge is to solve (MODIFY) using only synthetic samples, without access to ground-truth labels or the original loss.
this section cite: []

Section: Related Work
We briefly situate our work within the literature on vision and language model modification. A comprehensive survey is provided in Appendix A.
this section cite: []

Section: Vision Model Modification
While structured pruning is well-established for CNNs and ViTs [14,27,89,91], and classwise unlearning has seen recent progress [12,32], these tasks are typically treated in isolation. Crucially, prior methods for jointly addressing these problems rely heavily on access to labeled data [53]. Our work presents the first unified framework for both pruning and unlearning, which operates effectively using only unlabeled synthetic data.
this section cite: ['b14', 'b27', 'b90', 'b92', 'b12', 'b32', 'b54']

Section: LLM Modification
Efficiency in LLMs is primarily addressed via structured pruning [46,47] or sparsification [2]. However, these methods are often architecture-specific and do not extend to unlearning. By validating our method on both LLMs and vision models, we demonstrate a generalized approach to model modification that bridges the gap between these distinct domains.
this section cite: ['b47', 'b48', 'b1']

Section: Which Components Are Important for Modifying Well-Trained Models?
We now address the problem of identifying model components critical to predictive performance. We introduce Subset Fidelity, a metric that quantifies the local reconstructive capacity of component groups and High-Fidelity (HiFi) components. We show theoretically that maximizing local fidelity minimizes a linear upper bound on the global predictive error.
this section cite: []

Section: High-Fidelity Components and the Subset Fidelity Score
Our objective is to estimate the impact of removing a subset of input contributions on the model's output, after optimally compensating for this removal. Directly quantifying this effect is difficult, so we introduce the Subset Fidelity, a measure of how well a subset of components can locally approximate the layer's output. Definition 3.1 (Subset Fidelity). The fidelity of a subset of components C ⊆ [c l in ] in layer l for output channel c is defined as where δ l c is the compensation term.
FS l c (C) := max δ l c ∈R c l in 1 - E ∥Y l c (X) -i∈C δ l ci A l ci (X)∥ 2 E [∥Y l c (X)∥ 2 ] ,(1)
The following properties (proved in Appendix B.2) justify its use as an importance measure. A larger Subset Fidelity indicates that the subset more effectively reconstructs the output, thereby reducing the error of approximating the sum of components with components from a subset. Lemma 3.2 implies two key insights: (1) Fidelity serves as a principled measure of component importance, and
(2) Monotonicity suggests that greedy selection strategies may be effective. Remark 3.3. Equation ( 1) is a generalizes the formulation of El Halabi et al. [11]. In this work, we focus only on the case where the subset fidelities are measured with the expected squared difference.
We leave to future work an exploration of other possible measures of distributional similarity.
To capture the tradeoff between the size of a subset and its fidelity, we define HIFI Sets. Definition 3.4 ((k, η)-HIFI Set). Given a target subset size k and a fidelity threshold η ∈ (0, 1), the
(k, η)-HIFI Set S k,η c for output channel c is any subset in [c l in ] satisfying FS l c (S k,η c ) ≥ η, |S k,η c | ≤ k. (HIFI)
Thus, attributing predictive performance to components reduces to finding the HIFI set for a given (k, η). We can reduce the identification of HIFI sets to solving an optimization problem, the solution of which yields the Maximum Fidelity Subset, which contains the components that best recover the layer's output. Definition 3.5 (k-Maximum Fidelity Subset). Given a target subset size k for layer l, the Maximum Fidelity Subset S l⋆ c for channel c is defined as
S l⋆ c = arg max S⊆[c l in ], |S|=k
this section cite: ['b11']

Section: FS l c (S). (K-MFS)
A simple algorithm for identifying a (k, η)-HIFI set is to solve Equation (K-MFS) for the given k and check whether its fidelity exceeds η. If it does not, no such (k, η)-HIFI set exists. Before proceeding to our theoretical analysis, we empirically verify whether small HiFi sets actually exist in standard models. Our experiments in Section 5.2 empirically establish the existence of a small subset of components that can achieve high fidelity. Moreover, in Section 5.3, we validate the effectiveness of HiFi components with the model's predictive performance. Figure 1 indicates a sample of the results indicating that fewer than 20% of components can achieve high fidelity (≥ 0.8).
this section cite: []

Section: Local Distributional Measures of Component Importance
Finding HIFI subsets corresponds to finding subsets that minimize the l 2 reconstruction error while accounting for weight compensation. Additionally, it enables the derivation of a closed-form expression for weight compensation, allowing for accuracy recovery without requiring fine-tuning.
this section cite: []

Section: Bounding Global Error via Local Modification
We now show that the influence of a component on its immediate layer output provides a tractable proxy for its overall effect on model predictions. The global error is the expectation of the squared difference in the predictions of a network under a modification.
this section cite: []

Section: Theorem 3.6 (Local to Global).
Consider a network N θ as defined in Section 2.1. Let M l be a mask modifying parameters at layer l, and let m l c be the mask vector for output channel c. Assume there exist scalars r ℓ > 0 for all layers ℓ > l such that ∥Φ ℓ c (X)∥ F ≥ r ℓ almost surely. Then,
E ∥N θ (X) -N θ⊙M l (X)∥ 2 ≤ O   c l out c=1 E ∥Y l c (X) - i∈C m l ci A l ci (X)∥ 2  (2)
Sketch. The proof relies on the propagation of error through Lipschitz-continuous layers. See Appendix B.1.
Theorem 3.6 upper-bounds the global error, given by the left-hand side, by a linear function of the local reconstruction errors for each channel in layer l. This implies that global error grows at most linearly with local error, making local fidelity a practical, architecture-agnostic proxy for component influence. The theorem requires that the networks discussed in this work are Lipschitz continuous under suitable conditions. While CNNs are known to be Lipschitz continuous [90], transformers are not [60]. In Corollary B.4, we show that this is not the case for well-trained transformers. Remark 3.7. The leading constant in the order notation quantifies the amplification of local errors through subsequent layers and activations, and is independent of the data distribution, depending only on the model's architecture. Empirical estimates of the constant reported in Appendix C.2 demonstrate the practicality of these constants.
Subset Fidelity for Individual Components Next, we show that both the compensation term and the singleton fidelity scores admit closed-form expressions, thus motivating their use in this work. A derivation is provided in Appendix B.3. Proposition 3.8 (Compensation and Singleton Fidelity). For the l 2 reconstruction error, the optimal compensation term δ ⋆ c , which is the value at which the fidelity score is computed according to Equation (1) for a subset C, is given by,
δ l⋆ ci (C) = 1 + ((Q l c [C, C]) -1 ) ⊤ i Q l c [C, C]1 n-k if i ∈ C 0 if i / ∈ C (FS)
where
Q l c ∈ R c l in ×c l
in is the component similarity matrix (CSM) for channel c, with entries (
Q l c ) ij = E[⟨A l ci (X), A l cj (X)⟩].
The singleton fidelity scores are:
s l ci = FS l c ({i}) = 1 - E[∥Y l c (X) -α l ci A l ci (X)∥ 2 ] E[∥Y l c (X)∥ 2 ] , α l ci = E[⟨Y l c (X), A l ci (X)⟩] E[∥A l ci (X)∥ 2 ] .(3)
Note that solving Equation (K-MFS) exactly is still equivalent to a constrained binary quadratic optimization problem, known to be NP-hard [1]. Viewing Q c as the adjacency matrix of a weighted graph, maximizing Equation (K-MFS) corresponds to identifying a clique of size k, the decision version of the MAXIMUM CLIQUE problem. Intuitively, such cliques correspond to groups of components whose joint removal maximally increases the reconstruction error.
Computing the k-MFS Since fidelity is monotonic, a natural heuristic selects the k components with the highest singleton fidelities s l ci ; we call this strategy: NAIVE. To compute the set of highest fidelity, the k-MFS, we identify conditions under which the NAIVE selection strategy is optimal. Theorem 3.9. Consider output channel c in the l th layer of a network described in Section 2.1. Let the s l ci be defined according to Equation (3) and S l⋆ c be defined according to Definition 3.5. Let Ŝl c = {i | s l ci ≥ s (k) } where s (k) is the k th largest value of s l c . Assuming that there are no ties,
| Ŝl c | = k. If E[⟨A l ci (X), A l cj (X)⟩] = 0 ∀i ̸ = j, then Ŝl c = S l⋆ c .
Sketch. Under the assumptions, the objective simplifies from quadratic to linear. See Appendix B.4.
Remark 3.10. Theorem 3.9 connects a statistical property of the representations to the efficient discovery of HIFI components. It states that when the input contributions are pairwise uncorrelated, the optimal subset is the set of components with the highest fidelity score.
Although the assumption of uncorrelated features rarely holds exactly in practice, it offers a sound theoretical justification for NAIVE HIFI selection. We demonstrate the practical effectiveness of NAIVE HIFI selection through our experiments in Section 5.
this section cite: ['b91', 'b61', 'b0']

Section: Modifying Model Behavior using HiFi Sets
We now propose MODHIFI, a unified algorithmic framework for model modification using only distributional access. We apply this framework to two distinct tasks: structured pruning (MODHIFI-P) and classwise unlearning (MODHIFI-U). The central idea is to identify high-fidelity (HIFI) components and then modify them in a targeted manner using a unified algorithmic procedure, as shown in Algorithm 1. The two tasks operate as duals: pruning retains the high-fidelity components necessary for general performance, while unlearning removes the high-fidelity components most discriminative for a specific target class. Additional details, including complexity and implementation specifics, are provided in Appendix D.
this section cite: []

Section: Structured Pruning
To address Equation (STRUCT-PRUNE), where the objective is to remove entire input channels (or features) that contribute minimally to the model's predictive performance.
In convolutional architectures, we identify and remove input channels across all layers that do not appear in the HIFI sets of any output channel of the residual-coupled layers. For CNNs, pruning is applied to the input channels of convolutional layers. For LLMs, we target the input features of the MLP down-projection matrices (W D ). After pruning, we compute the optimal compensation term δ ⋆ (derived in Proposition 3.8) using the remaining weights. This step restores the fidelity of the layer output without requiring gradient-based fine-tuning.
Algorithm 1 ModHiFi-X Require: Model parameters θ, layer l, k components, threshold η, data D Ensure: Modified parameters θ E 1: Estimate Fidelity: Compute singleton scores s l on D via Equation (3). 2: Select HiFi Set: H l ← Top-k indices of s l . 3: if X = Prune then 4: for i ∈ [c l in ] \ {i | (c, i) ∈ H l } do 5: W l c,i ← 0 ∀c ∈ [c l out ] 6:
Apply compensation δ ⋆ to remaining weights. 7: else if X = Unlearn then 8:
for (c, i) ∈ H l do 9:
W l c,i ← 0 10: return θ
this section cite: []

Section: Class Unlearning
The goal of Equation (UNLEARN) is to erase the influence of a specific forget class. To perform unlearning, we first compute HIFI sets using only samples from the class we wish to forget. The components in these sets are then zeroed out, effectively erasing the influence of that class. This causes the model's predictive performance on the forgotten class to degrade, without significantly impacting the performance of other classes.
this section cite: []

Section: Fidelity Estimation
For vision models, the singleton fidelity score FS l c (•) can be estimated efficiently using distributional access to the input data, i.e., synthetic samples. In practice, for vision models, we estimate the scalar coefficients α l ci directly via batched forward passes on synthetic samples. A large α l ci indicates a high-fidelity component. For LLMs, we develop a tractable Cholesky-based heuristic to estimate the score, providing details in Appendix D.2.
this section cite: []

Section: Experiments
We empirically validate our framework by addressing four central questions: (Q1) Existence of HiFi components. Do a small subset of components exist that can achieve high fidelity? (Q2) Effectiveness of HIFI components. Do HIFI components accurately represent those components important for the predictive performance? (Q3) Effectiveness of using HIFI components for pruning using ModHiFi-P. Does ModHiFi-P result in better accuracy-sparsity tradeoff compared to structured pruning algorithms for vision tasks and language modeling tasks? (Q4) Effectiveness of using HIFI components for machine unlearning using ModHiFi-U. Is it possible to perform machine unlearning, as posed by Jia et al. [32], without finetuning? If so, how does ModHiFi-U compare to their method?
0 10 20 30 40 50 60 Input channel index 0.0 0.1 0.2 0.3 0.4 0.5 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0 0 10 20 30 40 50 60 Input channel index 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0 0 10 20 30 40 50 60 Input channel index 0.0 0.1 0.2 0.3 0.4 0.5 0.6 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0 0 20 40 60 80 100 120 Input channel index 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0
this section cite: ['b32']

Section: Details of the experimental setup
Models, Datasets, and Evaluation We conduct experiments on ResNet-50/101 [25], VGG19 [69], Swin-Transformer [44] and Llama-2-7B [76], benchmarking against relevant experiments from related literature [2,46]. For vision tasks, we measure the classification accuracy, and for NLP tasks, we use EleutherAI's lm-eval-harness [19].
Distributional Access For CIFAR10/100 [35], we use synthetically generated images as detailed in Appendix C.3. We use Alpaca [74] (a synthetic dataset) and WikiText-2 [48] as calibration data for NLP tasks following related literature [2,46]. We provide ablations to measure the impact of synthetic data quality in Appendix C.3.3.
this section cite: ['b25', 'b70', 'b45', 'b77', 'b1', 'b47', 'b19', 'b36', 'b75', 'b49', 'b1', 'b47']

Section: Compute platform and implementation details
We discuss the compute platform, implementation details, and hyperparameters used for our experiments in Appendix C.6.
this section cite: []

Section: Existence of HIFI components: Exploring (Q1)
To empirically assess whether small subsets can achieve high fidelity, we estimate S ⋆ c by sampling random subsets of size k across different architectures and selecting the subset with the highest fidelity. Detailed results are presented in Appendix C.1. Observation 1. Across all evaluated models, each layer typically contains a small subset of input channels (fewer than 20%) that achieves high subset fidelity (≥ 0.8).
This empirical observation suggests that in trained models, only a small subset of components in each layer is responsible for the model's prediction. This observation aligns with the success of structured pruning algorithms in constructing small subnetworks with high statistical performance.
this section cite: []

Section: Effectiveness of HIFI components: Exploring (Q2)
To answer (Q2), and verify whether HIFI components are the components that matter for the final predictive performance, we measure the effects of the fidelity of a component getting destroyed by noising. For a ResNet-50 on CIFAR-10, when 20% of the HIFI components are perturbed with a zero mean Gaussian noise with standard deviation of 0.01, the accuracy of the model drops by around 12%. In contrast, perturbing 80% of the non-HIFI components identically results in an accuracy drop of only 1%. At 50% of components with a noise of standard deviation 0.02, the accuracy drops by 85% when HIFI components are noised compared to only around 1.4% when non-HIFI components are noised. In Appendix C.1.3, we make similar observations across various models and tasks. In Appendix C.1.2, we additionally performed experiments where we compare the removal of HIFI, non HIFI, and random sets of the same size and make similar observations.
this section cite: []

Section: Structured Pruning Experiments: (Q3)

this section cite: []

Section: Vision Models
Baselines We compare against the state-of-the-art structured pruning algorithms specialized for pruning vision models [8,45,56,79], and present additional results on other architectures and datasets in Appendix C.4 where we make similar observations. Following [16], we update the batch norm statistics using the data from distributional access.  Observations We find that our method yields a better accuracy-vs-sparsity tradeoff compared to other algorithms across various datasets. We also train a model obtained with L 2 norm-based structured pruning using the synthetic set based on CIFAR10 for comparison. In Table 2, we observe that for the same FLOP sparsity, our method obtains higher accuracy than the model finetuned on synthetic samples, indicating that our method can outperform finetuning in some cases using synthetic samples for the same sparsity. For the ImageNet dataset, we compare our approach against various state-of-the-art structured pruning algorithms for networks with complex interconnections, including those trained on the ImageNet training set. In Table 1, we observe that for models of similar accuracy, our algorithm obtains the best accuracy-speedup tradeoff with fewer epochs of finetuning. Details of pre-trained networks and post-training are given in Appendix C.7.2. Our study of the effect of the quality of synthetic samples on our algorithm in Appendix C.3.3 indicates that the sparsity-accuracy tradeoff of our algorithm degrades with lower quality samples, but it does not degrade as much as L 2 pruning + finetuning on synthetic samples.
this section cite: ['b7', 'b46', 'b57', 'b80', 'b16']

Section: Large Language Models
Baselines We evaluate ModHiFi on Llama-2-7B, comparing it against state-of-the-art algorithms for structured pruning [2,47]. The use of calibration datasets to compute statistics aligns with our framing of distributional access to data, as LLMs do not make their training data openly accessible. Unless otherwise specified, the algorithms use WikiText-2 for calibration, with 128 samples of length 1024 4 . None of the algorithms performs post-pruning recovery finetuning. Additional details about our choice of baselines can be found in Appendix C.4.3.
this section cite: ['b1', 'b48']

Section: Evaluation
We also measure the performance of the model via its zero-shot accuracy on a suite of standard NLP tasks [5,9,62,92] and WikiText perplexity. In Table 3, we observe that our method is competitive, with consistently high average and task-specific performance, and outperforms at moderate sparsity levels. We find that the quality of the calibration set plays a crucial role, with the performance of ModHiFi-P-Alpaca outperforming that of ModHiFi-P-WikiText. This indicates that retaining only HIFI components provides a model-agnostic approach to structured pruning, with its application to LLMs requiring no modifications beyond its application to vision models.
this section cite: ['b4', 'b9', 'b63', 'b93']

Section: Class Unlearning Experiments: (Q4)
Baselines and Metrics We report the forget and retain accuracy averaged across 10 classes of the CIFAR10 dataset on ResNet-50 and Swin-T models. We benchmark against Gradient Ascent and Jia et al. [32], which are both retraining-based techniques for Unlearning.
this section cite: ['b32']

Section: Unlearning Results
We report the results of our algorithm in Table 4. To answer (Q4), we observe that it is possible to perform unlearning without finetuning in a general editing framework 10× faster than our baseline. In Appendix C.5, we compare results with finetuning using synthetic and training data. We note that the results for Swin-Transformer without finetuning fail to achieve the state of the art. However, as reported in Appendix C.5, we observe a drastic improvement with only three epochs of finetuning on synthetic samples. After 10 epochs of finetuning with our algorithm, we find that our forget accuracy is superior to that of [32] (who use full training) when using synthetic samples. Both forget and remain accuracy are superior when using training samples. Experiments with VGG-19 are present in Appendix C.5 where we make similar observations.
this section cite: ['b32']

Section: Discussion and Conclusion
We have addressed the challenge of modifying well-trained deep networks without access to gradients, loss functions, or original training data. By theoretically connecting local layer-wise reconstruction to global predictive error, we established Subset Fidelity as a rigorous proxy for component importance.
Our empirical analysis reveals a fundamental property of modern networks: predictive performance is concentrated in sparse HIFI substructures that are robust to noise and identifiable via synthetic data. Leveraging this insight, we proposed MODHIFI, a unified framework for model modification. Unlike prior architecture-specific heuristics, MODHIFI is domain-agnostic, effectively handling both structured pruning and classwise unlearning across CNNs and Transformers. Crucially, our method is designed for the regime of distributional access, making it uniquely suited for modern deployments where privacy or scale necessitates the use of synthetic data.
Limitations and Future Work Our theoretical bounds in Theorem 3.6 rely on the local Lipschitz continuity of the network. While we demonstrate that this property holds for well-trained models (including Transformers on bounded domains), it is not guaranteed at initialization. This suggests that the emergence of High-Fidelity components is a consequence of the training dynamics. In this work, we use the expected square loss as a measure of distributional similarity, and we leave for future work the exploration of other metrics of distributional similarity, like the TV distance or Wasserstein metric.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss limitations in Section 6.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: Yes, for each theoretical result, we provide a complete set of assumptions and a correct proof. Proofs are attached in the appendix, Appendix B. We link the relevant appendices in the main body for the reader.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: Yes, we fully disclose all the information to reproduce the main experimental results in Section 5 and the appendices mentioned within that section.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The code and the instructions to reproduce the experiments are provided at the GitHub: https://github.com/DhruvaKashyap/modhifi. Moreover, the data sets used are open-sourced, and details on how to obtain them are provided on our GitHub. Guidelines:
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
Answer: [Yes] Justification: All details to understand the results and reproduce the results are provided in Section 5 and the appendices mentioned therein.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: For experiments where error bars are relevant and computationally feasible, we report them.
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
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: We sufficiently describe the compute resources used for our experiments in Section 5 and the appendices referred to within the section, specifically, Appendix C.6.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: We go through the NeurIPS Code of Ethics and confirm that we adhere to them.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [No] Justification: We do not discuss the societal impact of the work performed in this manuscript since this is foundational research and not tied to any particular applications.
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
Answer: [NA] Justification: We do not release generative models or data in this work.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: We credit original owners of assets used in this work appropriately through citations.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release any new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: We do not use LLMs to develop any methods presented in this work. We clarify further in Appendix E. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
• Appendix D.3 presents the computational complexity of estimating fidelity. The fidelity is linear in number of layers as opposed to the component identification module of SoTA in data-free structured pruning, which is quadratic in the number of layers.
this section cite: []

Section: A Related Work
We present a short literature review in Section 2. In this section, we discuss recent related work on structured pruning and unlearning not discussed in the main body.
Structured Pruning Structured pruning has been widely researched, with a wide variety of methods proposed for it [27]. Unlike unstructured pruning, which sparsifies the weight matrices without changing the architecture of the model [6,15,16,38,73], structured pruning enables immediate improvements in real-world performance measures such as inference time and memory footprint without requiring specialized hardware or software [27,51,59]. A variety of methods have been proposed for structured pruning of convolutional networks, including using norms of weight tensors [39], directional derivative scores [51,52,66], feature map ranks [41,70], coresets [4,40,77], discriminative ability of filters [42,54], and reconstruction error [11,67,90]. However, modern neural networks possess complex interconnections, making them difficult to structurally compress [13,43,56], for which some recent algorithms have been proposed that use gradient information [43] or bounds on the reconstruction error [56,90]. Moreover, pruning without access to either the training data or the loss function is an increasingly important area of research, for which some works have been proposed that use the discriminative ability of filters as a saliency [42,54]. However, none of these works address the problem of pruning large language models.
Pruning of Large Language Models (LLMs) has garnered significant interest in recent years [94].
A variety of unstructured pruning methods have been proposed, such as [17,71]. However, these methods do not provide direct improvements on inference time and memory footprint. Thus, the problem of pruning models with structural interconnections has naturally been applied to pruning LLMs as well, in works such as [2,46,47,84]. A key drawback of these works is that most are not applicable to CNNs or other kinds of models. Our work proposes a unified framework for both pruning models with complex interconnections, including transformers and ResNets, as well as classwise unlearning.
Classwise Unlearning Machine unlearning has gained significant interest in recent years, both for data privacy concerns as well as connections to continual learning [7,29,57,81]. Machine unlearning is typically categorized into exact and approximate unlearning [86]. Exact unlearning involves training models from scratch without the forget data (the data to be forgotten), or by training modules or experts on subsets of data [85,87]. Approximate unlearning, on the other hand, refers to techniques that approximate exact unlearning via various approaches [30,86]. Machine unlearning can be further classified into sample unlearning (wherein individual samples or random subsets of samples are unlearned) [64,78] or classwise unlearning (where classes or concepts are unlearned) [18,24]. In this work, we focus on classwise unlearning.
A variety of approaches have been proposed for classwise unlearning [24]. Popular methods include fine-tuning the model without data from the forget class [20,82], gradient ascent on the forget set [24,75], distillation-based approaches [37], and influence function based methods [30]. More recent work studies using sparsity for machine unlearning, such as [32], which first sparsifies the model, and then applies a fine-tuning-based unlearning algorithm, or [53,80], which identify class-discriminative filters in CNNs, and removes them for unlearning. Two key drawbacks of prior art, however, are: first they exclusively address classwise unlearning, and do not address wider problems of model modification. Second, all prior art assumes access to the original training data. Our proposed approach for classwise unlearning differs from prior art because it only requires synthetic class data, uses a variety of granularities for sparsity in unlearning, and is part of a unified approach to model modification.
this section cite: ['b27', 'b5', 'b15', 'b16', 'b39', 'b74', 'b27', 'b52', 'b60', 'b40', 'b52', 'b53', 'b67', 'b42', 'b71', 'b3', 'b41', 'b78', 'b43', 'b55', 'b11', 'b68', 'b91', 'b13', 'b44', 'b57', 'b44', 'b57', 'b91', 'b43', 'b55', 'b95', 'b17', 'b72', 'b1', 'b47', 'b48', 'b85', 'b6', 'b29', 'b58', 'b82', 'b87', 'b86', 'b88', 'b30', 'b87', 'b65', 'b79', 'b18', 'b24', 'b24', 'b20', 'b83', 'b24', 'b76', 'b38', 'b30', 'b32', 'b54', 'b81']

Section: B Proofs
In this section, we restate the formal statements made in the main body of the paper and present the proofs omitted in the main body. We follow the notation defined in Section 2.
this section cite: []

Section: B.1 Proof of Theorem 3.6
We now provide a proof of Theorem 3.6. We first state the properties of the normalization layer and provide empirical evidence to justify their validity, followed by a restatement of the theorem and its proof. Definition B.1 (RMSNorm and LayerNorm). Consider the l-th layer parameterized by γ l , β l ∈ R d . For an input ϕ(x) ∈ R d , the output y ∈ R d is defined as:
NM(ϕ(x)) = γ l ⊙ z ∥z∥ 2 + β l , where z = Mϕ(x).
Here, ⊙ denotes the Hadamard product. For RMSNorm, M = I d . For LayerNorm,
M = I d -1 d 1 d 1 ⊤ d (the centering matrix).
Normalization layers are not globally Lipschitz continuous due to the singularity at zero. However, they are locally Lipschitz on domains bounded away from the origin. Definition B.2. A function f : R m → R n is Lipschitz continuous in its domain if there exists a positive scalar constant L such that
∥f (x) -f (y)∥ 2 ≤ L∥x -y∥ 2 ∀x, y ∈ R m for all x, y in the domain of f . Lemma B.3. Let X r = {x ∈ R d | ∥x∥ 2 ≥ r > 0}. Define the map f : X r → S d-1 as f (x) = x ∥x∥2 . Then f is Lipschitz continuous on X r with constant L f = 1/r. That is, ∥f (x) -f (y)∥ 2 ≤ 1 r ∥x -y∥ Proof. For any x, y ∈ X r , ∥f (x) -f (y)∥ 2 2 = x ∥x∥ - y ∥y∥ 2 2 = 2 -2 x ⊤ y ∥x∥y∥ Simultaneously, ∥x -y∥ 2 2 = ∥x∥ 2 + ∥y∥ 2 -2x ⊤ y = ∥x∥∥y∥ ∥x∥ ∥y∥ + ∥y∥ ∥x∥ -2 x ⊤ y ∥x∥∥y∥ .
Using the AM-GM inequality a + 1/a ≥ 2 for a > 0, and noting that ∥x∥, ∥y∥ ≥ r:
∥x -y∥ 2 2 ≥ ∥x∥∥y∥ 2 -2 x ⊤ y ∥x∥∥y∥ = ∥x∥∥y∥ ∥f (x) -f (y)∥ 2 2 ≥ r 2 ∥f (x) -f (y)∥ 2 2 .
Rearranging terms completes the proof.
Using Lemma B.3, we can show that the operation performed by normalization layers is Lipschitz continuous in Corollary B.4. Corollary B.4. Let the input to the l-th normalization layer satisfy ∥MΦ(x)∥ 2 ≥ r > 0 for all x.
Then, the normalization layer satisfies
∥NM(ϕ(x)) -NM(ϕ(y))∥ ≤ ∥γ l ∥ ∞ r ∥M∥ 2 ∥ϕ(x) -ϕ(y)∥.
Note that ∥M∥ 2 = 1 for both RMSNorm and LayerNorm.
Proof. Let u(x) = Mx. Then ∥NM(x) -NM(y)∥ 2 = ∥γ l ⊙ u(x) ∥u(x)∥ -u(y) ∥u(y)∥ ∥ 2 ≤ ∥γ l ∥ ∞ 1 r ∥M(x -y)∥ 2 by applying Lemma B.3 to complete the proof.
While the lower bound assumption ∥z∥ ≥ r is technically not guaranteed for all x ∈ R d , we empirically verify that for trained networks, activation norms are strictly bounded away from zero. This validates the local Lipschitz property in the region of interest. We show the layer-wise minimum norm of the pre-LayerNorm representations in Figure 3, estimated on 100 samples from the Alpaca dataset. For various models, we observe the lower bound to be between 0.2 and 60. For clarity of exposition, we only show the layers with the largest and smallest values, along with 5 randomly selected layers. Code for generating these plots can be found in Appendix C. We also observe that this value tends to increase for layers deeper in the network, and leave the utilization of this observation to future work.
this section cite: []

Section: B.1.1 Main Proof
We first state a well-known fact about Lipschitz functions. We then restate and prove Theorem 3.6.
Fact 1. A function f = f L • f L-1 • . . . • f 1 where each f i is Lipschitz continuous with Lipschitz constant L i , is Lipschitz continuous with Lipschitz constant L i=1 L i .
this section cite: []

Section: Theorem 3.6 (Local to Global).
Consider a network N θ as defined in Section 2.1. Let M l be a mask modifying parameters at layer l, and let m l c be the mask vector for output channel c. Assume there exist scalars r ℓ > 0 for all layers ℓ > l such that ∥Φ ℓ c (X)∥ F ≥ r ℓ almost surely. Then,
E ∥N θ (X) -N θ⊙M l (X)∥ 2 ≤ O   c l out c=1 E ∥Y l c (X) - i∈C m l ci A l ci (X)∥ 2  (2)
Proof. Consider a network as defined in Section 2.1.
Let N θ = f L • . . . • f l • f l-1:1 where f l-1:1 = f l-1 • . . . • f 1 .
Under standard assumptions on the smoothness of activations [90], each layer f l is Lipschitz continuous with Lipschitz constant
L l f . From Fact 1, E ∥N θ (X) -N θ⊙M l (X)∥ 2 ≤ ( L ℓ>l L ℓ f ) c l out c=1 E[∥Y l (X) - i m ci A ci (X)]∥ 2
By taking an upper bound on the Lipschitz constants of each layer in the composition, we see that the subnetwork after layer l has a Lipschitz constant of at least C l = L ℓ>l L ℓ f . Where, for convolution-based networks,
C l = max i γ l i σ l i η L-l ℓ>l ∥W ℓ ∥ 2 • max i |γ ℓ i | σ ℓ i
and for transformer models,
C l = η L-l ℓ>l ∥W ℓ ∥ 2 • max i |γ ℓ i | r ℓ
The expected squared error at the final output is:
E[∥N θ (X) -N θ⊙M (X)∥ 2 ] ≤ C 2 l E[∥Y l (X) -Ỹl (X)∥ 2 ].
We decompose the layer output by channels c ∈ [c l out ]. The masked output for channel c is Ỹl
c = i m ci A l ci , where m ci ∈ {0, 1} are entries of M l . E[∥Y l (X) -Ỹl (X)∥ 2 ] = c l out c=1 E   cin i=1 A l ci (X) - cin i=1 m ci A l ci (X) 2   = c l out c=1 E   cin i=1 (1 -m ci )A l ci (X) 2   .
Let v c = 1 -m c be the indicator vector of removed components. Expanding the squared norm:
E   i v ci A l ci (X) 2   = E   i j v ci v cj ⟨A l ci (X), A l cj (X)⟩   = i,j v ci (Q l c ) ij v cj = v ⊤ c Q l c v c .
Substituting this back completes the proof.
this section cite: ['b91']

Section: B.2 Proof of Lemma 3.2
In this section, we prove the properties of Subset Fidelity stated in Lemma 3.2. We restate the definition of fidelity score and state a proposition. We then restate the proposition and provide a proof. Definition 3.1 (Subset Fidelity). The fidelity of a subset of components C ⊆ [c l in ] in layer l for output channel c is defined as
FS l c (C) := max δ l c ∈R c l in 1 - E ∥Y l c (X) -i∈C δ l ci A l ci (X)∥ 2 E [∥Y l c (X)∥ 2 ] ,(1)
where
δ l c is the compensation term. Lemma 3.2 (Properties of Subset Fidelity). For any subset C ⊆ [c l in ] in layer l, (Boundedness) 0 ≤ FS l c (C) ≤ 1 and (Monotonicity) If D ⊆ C, then FS l c (D) ≤ FS l c (C). Proof. Let L(δ; C) := E[∥Y l c (X) -i∈C δ i A l ci (X)∥ 2 ]. The fidelity is FS l c (C) = 1 -min δ L(δ;C) E[∥Y l c (X)∥ 2 ] . 1. Boundedness: Since the norm is non-negative, L(δ; C) ≥ 0 =⇒ FS ≤ 1. Selecting δ = 0 yields L(0; C) = E[∥Y l c ∥ 2 ].
Since the minimum is bounded by this value, the ratio is ≤ 1, so FS ≥ 0.
2. Monotonicity: Let D ⊂ C. The optimization for D is equivalent to optimizing over C with the constraint δ i = 0 ∀i ∈ C \ D. Since D ⊂ C, the feasible set for D is a subset of the feasible set for C. Therefore, min δ L(δ; C) ≤ min δ ′ L(δ ′ ; D). A lower minimum error implies a higher fidelity score. Thus, FS l c (C) ≥ FS l c (D).
this section cite: []

Section: B.3 Proof of Proposition 3.8
Proposition 3.8 (Compensation and Singleton Fidelity). For the l 2 reconstruction error, the optimal compensation term δ ⋆ c , which is the value at which the fidelity score is computed according to Equation (1) for a subset C, is given by,
δ l⋆ ci (C) = 1 + ((Q l c [C, C]) -1 ) ⊤ i Q l c [C, C]1 n-k if i ∈ C 0 if i / ∈ C(FS)
where
Q l c ∈ R c l in ×c l in is the component similarity matrix (CSM) for channel c, with entries (Q l c ) ij = E[⟨A l ci (X), A l cj (X)⟩].
The singleton fidelity scores are:
s l ci = FS l c ({i}) = 1 - E[∥Y l c (X) -α l ci A l ci (X)∥ 2 ] E[∥Y l c (X)∥ 2 ] , α l ci = E[⟨Y l c (X), A l ci (X)⟩] E[∥A l ci (X)∥ 2 ] . (3
) Proof. Define the error e(X) = Y l c (X) -i∈C δ i A l ci (X). We minimize J(δ) = E[∥e(X)∥ 2 ]. Recall that Y l c (X) = cin i=1 A l ci (X). Let u = 1 -δ,
where u is supported on the full set of indices but we constrain δ i = 0 (so u i = 1) for i / ∈ C. The objective is:
J(u) = E   cin i=1 u i A l ci (X) 2   = u ⊤ Q l c u.
We partition indices into C and C. Decompose u as [u C ; u C ]. The constraint δ i = 0 for i / ∈ C implies u C = 1 C . We optimize with respect to u C :
J(u C ) = u ⊤ C 1 ⊤ C Q CC Q CC Q CC Q CC u C 1 C = u ⊤ C Q CC u C + 2u ⊤ C Q CC 1 C + const.
This is a convex quadratic function, whose optima can be computed by taking the gradient w.r.t u C and setting to zero:
2Q CC u C + 2Q CC 1 C = 0 =⇒ u ⋆ C = -Q -1 CC Q CC 1 C .
Recalling δ C = 1 C -u C , we obtain:
δ ⋆ C = 1 C + Q -1 CC Q CC 1 C .
this section cite: []

Section: This matches Equation (3).

this section cite: []

Section: B.4 Proof of Theorem 3.9
In this section, we prove the optimality of the naive algorithm in selecting the k-MFS Optimal set. Theorem 3.9. Consider output channel c in the l th layer of a network described in Section 2.1. Let the s l ci be defined according to Equation (3) and S l⋆ c be defined according to Definition 3.5. Let Ŝl c = {i | s l ci ≥ s (k) } where s (k) is the k th largest value of s l c . Assuming that there are no ties,
| Ŝl c | = k. If E[⟨A l ci (X), A l cj (X)⟩] = 0 ∀i ̸ = j, then Ŝl c = S l⋆ c .
this section cite: []

Section: Proof.
The assumption E[⟨A l ci , A l cj ⟩] = 0 for i ̸ = j implies that the Component Similarity Matrix
Q l c is diagonal. Let q ii = (Q l c ) ii = E[∥A l ci ∥ 2 ] ≥ 0.
This implies that the component similarity matrix is diagonal. For any subset S, the optimal compensation δ ⋆ for diagonal Q simplifies. The reconstruction error for subset S is minimized when we perfectly reconstruct the components in S (since they are orthogonal to components in S). Thus, the residual error comes purely from the removed components S:
min δ E   Y l c (X) - i∈S δ i A l ci (X) 2   = E    j∈S A l cj (X) 2    = j / ∈S q jj .
The Subset Fidelity is:
FS l c (S) = 1 - j / ∈S q jj k q kk = i∈S q ii Tr(Q l c ) .
Similarly, the singleton fidelity score for component i is s l ci = qii Tr(Q l c ) . The optimization problem:
S ⋆ = arg max |S|=k FS l c (S) = arg max |S|=k i∈S q ii .
This linear objective is trivially maximized by selecting the k indices with the largest q ii values. Since s l ci ∝ q ii , this is equivalent to selecting the top-k singleton fidelity scores. Remark B.5. While the assumption of uncorrelated features might not hold under practical scenarios, this result provides an indication that the method could result in effective identification of critical model components in practical settings. Our experiments in Section 5 and Appendix C practically demonstrate the empirical efficacy of the methodology. 2. We discuss the synthetic samples used in our vision experiments and the effect of their quality on the algorithm.
3. We provide additional pruning and unlearning experiments for a variety of architectures.
4. We provide details of our compute platform, hyperparameters, and training procedure for our experiments.
Our code is available at https://github.com/DhruvaKashyap/modhifi. In this section, we provided worst case and average case estimates of the constant C l in Theorem 3.6.
In Figure 11a, we plot the constants obtained in the proof of Theorem 3.6 in Appendix B.1 for a ResNet-50 trained on ImageNet, and observe that the values can be very large (38 orders of magnitude). However, it is important to note that these are worst case guarantees, and that these constants are much smaller in practice. In Figure 11b, we compute the ratio between the global error, E ∥y(X) -y(X; M l )∥ 2 and the local error, 2 and observe that these values are indeed much smaller (10-50) for random values of M l for the expected square loss.
c l out c=1 E Y l c (X) -i∈C m l ci A l ci (X)
this section cite: []

Section: C.3 Discussion on the synthetic samples used in the experiments
We describe the synthetic datasets used in our vision experiments to simulate distributional access.
Randomly selected example images are provided in Figure 12. For NLP tasks, we use WikiText and Alpaca datasets [48,74] which are standard in this field.
this section cite: ['b49', 'b75']

Section: C.3.1 CIFAR5M
For experiments with the CIFAR10 dataset, we use CIFAR5M, a dataset containing 6 million synthetic CIFAR-10-like images sampled from a Diffusion model and labeled by a Big-Transfer model [55], which we randomly sample 10,000 samples from each of the 10 classes to create our dataset. This dataset has an FID [26] of 15.95 with respect to the CIFAR10 training set. This dataset is obtained from here.
this section cite: ['b56', 'b26']

Section: C.3.2 CIFAR100-DDPM
For experiments with the CIFAR100 dataset, we use CIFAR100-DDPM [23], which we randomly downsample to contain 1,000 samples from each of the 100 classes. This dataset has an FID of 4.74 with respect to the CIFAR100 training set. We randomly sample 1,000 samples from each of the 100 classes to create our dataset. This dataset is obtained from here.
this section cite: ['b23']

Section: C.3.3 Effect of Data Quality
To study the effect of data quality on the performance of our algorithm in vision tasks, we apply the pruning algorithm using synthetic datasets based on CIFAR10 generated with different FIDs. We use a diffusion model [33] to generate 3 datasets of differing quality by changing the number of diffusion steps (4,5, and 6). We report the results of our pruning algorithm with different quality datasets in Table 5. We observe that higher quality data leads to an improved sparsity -accuracy tradeoff.
this section cite: ['b33']

Section: C.4 Additional Pruning Experiments
We present additional pruning experiments in Tables 6 and 7.
this section cite: []

Section: C.4.1 Ablation of weight compensation and BatchNorm correction
In this section, we perform ablations for each component of our pruning algorithm, simple pruning, correcting batch norm statistics and weight compensation. We report our results for pruning ResNet-50 on CIFAR 10 in Table 8. We observe that each component allows for a better accuracy sparsity trade-off.
this section cite: []

Section: C.4.2 Final ImageNet Pruned Model
In Figures 13 and 14 we compare the final pruned models for ResNet-50 on ImageNet with DFPC [56]. We observe that our pruning algorithm removes more channels in later coupled channels than DFPC leading to higher gains in sparsity.
this section cite: ['b57']

Section: C.4.3 Baseline selection for LLM Pruning
We choose ShortGPT [47] and SliceGPT [2] as baselines against which we compare ModHiFi. We do so for two broad reasons: all three methods together represent three different granularities for conducting structured pruning for LLMs, and both ShortGPT and SliceGPT are the state-of-the-art within their respective lanes.
The three different granularities are 1. Layer pruning: Entire layers (i.e. transformer decoder blocks) are removed from the network. This is viable since transformers are constant width networks, i.e., there are no architectural restrictions to the ordering or number of layers. ShortGPT falls within this granularity. 2. Embedding pruning: The width of the network (i.e. the embedding dimension) is pruned at a uniform rate across the entire network. This entails a form of feature selection: along with weight matrix pruning, one also has to prune the corresponding dimensions from the feature matrix being fed into every layer. SliceGPT falls within this granularity. 3. Hidden dimension pruning: Here, the number of layers and the width of the embedding are left unchanged. Instead, one prunes the hidden dimensions within the modules that constitute a transformer decoder block. ModHiFi falls within this granularity.
We would like to emphasize that both SliceGPT and ShortGPT are designed to operate on Transformer models, and as such are able to leverage specifics of the architecture to their advantage. In return for this specificity, however, they trade off the ability to generalize to CNNs, something that ModHiFi does with ease due to its architecture-agnostic nature; the only assumption made by the Fidelity Score is that the components being scored belong to linear layers.
this section cite: ['b48', 'b1']

Section: C.5 Additional Unlearning Experiments
We report additional experiments in Table 9 on class unlearning on different architectures. For VGG-19 networks, we remove the HiFi channels for the forget class of the last 12 convolution layers. We also compare our work with DisCEdit-U from [53] wherein we remove discriminative components from the last 8 convolutional layers. We use a custom implementation of the algorithm for our VGG19 and ResNet50 models for CIFAR10, as those models are unavailable in the codebase of [53].
We also compare our work with DisCEdit-U on ResNet50 trained on CIFAR10 as well, which we present in Table 10 We show that our unlearning method achieves similar or superior performance to that of [53] without fine-tuning. Moreover, unlike [53], our approach uses only synthetic samples, showing the efficacy of our work in classwise unlearning, even in the absence of training data.
this section cite: ['b54', 'b54', 'b54', 'b54']

Section: Unlearning with finetuning
Here we compare our method with 3 additional epochs of finetuning on synthetic samples of the remaining class data. Although this setup does not fall into the setup of the work since we do not assume access to the loss function, we provide these results to indicate that even using very few synthetic samples we can perform perfect unlearning. We present these results in Table 11, where we observe almost perfect unlearning for both ResNet-50 and Swin-Transformers.
this section cite: []

Section: Unlearning with baseline budgets
In this section, we compare our method when allowing for the same amount of finetuning as [32], with both synthetic data and training data access. While this violates our assumptions about loss function and training data access, we present these results to provide a fair comparison of our algorithm when run within the same constraints as our baselines.
Our results can be found in Table 12 for the Swin Transformer.
this section cite: ['b32']

Section: C.6 Compute Platform
Implementation Details We implement our proposed methods in PyTorch [58] and use Huggingface's transformers [83] for LLM implementations.
this section cite: ['b59', 'b84']

Section: Inference time measurements
We follow the inference time measurement setting of [56,66]. Inference time is the time taken for a model to compute the forward pass for an input and does not account for loading data into memory. We compute the inference time for a batch of 640 random tensors for GPU and 64 for CPU. 100 iterations are used for warm up, after which the inference
Table 9: Class unlearning on CIFAR10 for VGG19 Model Algorithm Forget Accuracy Remain Accuracy VGG19 -93.50 93.50 DisCEdit-U [53] 2.39 84.2 Ours 0.86 77.85
this section cite: ['b57', 'b67']

Section: JIT Compilation
We present inference time numbers with JIT compilation on Pytorch [58].
Hardware Table 13 details the hardware we use to conduct our experiments. Values in (*) indicate reported values obtained from https://www.amd.com/en/products/accelerators/  instinct/mi200/mi210.html. This machine runs Ubuntu 22.04.3 LTS with kernel 6.8.0-40generic with the hardware in Table 13. Our software stack comprises of Python 3.12.8, PyTorch 2.5.1 built for ROCm 6.2, and torchvision version 0.20.1 built for ROCm 6.2. Inference times are measured on a machine running Ubuntu 20.04.1 LTS with kernel 5.15.0-91generic on the hardware specified in Table 14. The software stack used for inference consists of Python 3.12.8, PyTorch 2.5.1, and Torchvision 0.20.1 for CUDA 12.3.
this section cite: ['b59']

Section: C.6.1 Module-level Time Consumption
In this section, we break down the time each component of our algorithm takes. For 2000 samples batched into batches of size 64, when running the algorithm on a ResNet-50:
• Computation of fidelity scores takes between 32GB to 51GB of VRAM, and between 2 minutes to 5 minutes, on 1 GPU of machine 13, across data from CIFAR10, CIFAR100, and ImageNet.
• Computing δ ⋆ c across 4 GPUs using an average of 60GB per GPU takes 60 minutes for CIFAR10/100, and 90 minutes for ImageNet, averaging to roughly 1 minute per layer.
this section cite: []

Section: C.7 Hyperparameters and Training Procedure

this section cite: []

Section: C.7.1 Hyperparameters for Experiments
We typically set the percentile of removed components to be between 0.01 to 0.2. We randomly select 2% of our synthetic samples to select data for vision tasks and select 128 samples for NLP tasks. • Structured Pruning: We retain HiFi components and discard the rest. While we cannot guarantee a fixed sparsity level in the output model (since HiFi components may span all inputs), we observe in practice that reasonable sparsity emerges naturally. For more aggressive pruning, the algorithm is applied iteratively.
• Class Unlearning: Simply discarding low-fidelity components is insufficient. Instead, we aim to remove or disrupt the influence of HiFi components that are specific to the forget class. The editing strategy depends on the network type:
-In BatchNorm networks, we zero out the weights of HiFi components computed as per the forget class samples. -In LayerNorm-based networks with residual connections (e.g., Swin-T), we negate the weights of HiFi components. This rotates the forget-class representation in the opposite direction due to the residual path.
The unlearning strategy for Transformer-based architectures is captured in the following procedure: We use the Cholesky decomposition since it is efficient to compute.
Algorithm
this section cite: []

Section: D.3 Computational cost
Let N be the number of data points used to estimate the saliency and M l be the complexity of computing the input contribution at layer l for a single sample in a set of coupled channels with m layers. The complexity to compute the set of retained channels for an output channel of a layer is, t l sal = O(N M l C l in d l ). To select the components for the coupled channels, the top p elements for each layer and output channel in them are collected, this costs O( m l=1 C l out (C l in log C l in + t l sal )). The algorithm shows a linear dependence on the number of layers in the network, compared with the BGSC algorithm [56] which has a quadratic dependence.
this section cite: ['b57']

Section: E Full LLM Disclosure
We employed Large Language Models (LLMs) to refine the text for grammar and clarity. Additionally, LLMs were used to generate auxiliary scripts for data visualization (plots). We confirm that LLMs were not used to implement any of the core algorithms or methodologies proposed in this work.      0 200 400 600 800 1000 Input channel index 0.000 0.005 0.010 0.015 0.020 0.025 0.030 0.035 0.040 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0 0 50 100 150 200 250 Input channel index 0.00 0.05 0.10 0.15 0.20 0.25 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0 0 20 40 60 80 100 120 Input channel index 0.00 0.02 0.04 0.06 0.08 0.10 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0 0 50 100 150 200 250 Input channel index 0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0 0 100 200 300 400 500 Input channel index 0.00 0.02 0.04 0.06 0.08 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0 0 10 20 30 40 50 60 Input channel index 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0 0 50 100 150 200 250 Input channel index 0.00 0.01 0.02 0.03 0.04 0.05 0.06 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0 0 10 20 30 40 50 60 Input channel index 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 Fidelity score Noise trained avg_acc=94.99 noise=0.005 avg_acc=93.4325 noise=0.01 avg_acc=24.715 noise=0.05 avg_acc=9.9975 untrained avg_acc=0.0
this section cite: []

Section: References
Ref_id:b0 Title: Computational Complexity: A Modern Approach Year: (2009)
Ref_id:b1 Title: Slicegpt: Compress large language models by deleting rows and columns Year: (2024)
Ref_id:b2 Title: Layer normalization Year: (2016)
Ref_id:b3 Title: Datadependent coresets for compressing neural networks with applications to generalization bounds Year: (2019)
Ref_id:b4 Title: Piqa: Reasoning about physical commonsense in natural language Year: (2020-04)
Ref_id:b5 Title: What is the state of neural network pruning? Year: (2020)
Ref_id:b6 Title: Machine unlearning Year: ()
Ref_id:b7 Title: Only train once: A one-shot neural network training and pruning framework Year: ()
Ref_id:b8 Title: a376033f78e144f494bfc743c0be3330-Paper.pdf Year: (2021)
Ref_id:b9 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b10 Title: Data augmentation using LLMs: Data perspectives, learning paradigms and challenges Year: (2024-08)
Ref_id:b11 Title: Data-efficient structured pruning via submodular optimization Year: (2022)
Ref_id:b12 Title: Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation Year: (2024)
Ref_id:b13 Title: Depgraph: Towards any structural pruning Year: (2023)
Ref_id:b14 Title: Isomorphic pruning for vision models Year: (2025)
Ref_id:b15 Title: The lottery ticket hypothesis: Finding sparse, trainable neural networks Year: (2019)
Ref_id:b16 Title: Optimal brain compression: A framework for accurate post-training quantization and pruning Year: (2022)
Ref_id:b17 Title: Massive language models can be accurately pruned in one-shot Year: (2023-07)
Ref_id:b18 Title: Erasing concepts from diffusion models Year: (2023-10)
Ref_id:b19 Title: A framework for few-shot language model evaluation Year: ()
Ref_id:b20 Title: Eternal sunshine of the spotless net: Selective forgetting in deep networks Year: (2020-06)
Ref_id:b21 Title: Continual learning via neural pruning Year: (2019)
Ref_id:b22 Title: Generative adversarial nets Year: (2014)
Ref_id:b23 Title: Improving robustness using generated data Year: (2021)
Ref_id:b24 Title: Amnesiac machine learning Year: (2021-05)
Ref_id:b25 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b26 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b27 Title: Sparsity in deep learning: pruning and growth for efficient inference and training in neural networks Year: (2021-01)
Ref_id:b28 Title: Characterising bias in compressed models Year: (2020)
Ref_id:b29 Title: What do compressed deep neural networks forget? Year: (2021)
Ref_id:b30 Title: Approximate data deletion from machine learning models Year: (2008)
Ref_id:b31 Title: Distilling model failures as directions in latent space Year: (2023)
Ref_id:b32 Title: Model sparsity can simplify machine unlearning Year: (2023)
Ref_id:b33 Title:  Year: ()
Ref_id:b34 Title: a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf Year: (2022)
Ref_id:b35 Title: Deep unlearning: Fast and efficient gradient-free class forgetting Year: (2024)
Ref_id:b36 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b37 Title: Imagenet classification with deep convolutional neural networks Year: (2012)
Ref_id:b38 Title: Towards unbounded machine unlearning Year: (2023)
Ref_id:b39 Title: Optimal brain damage Year: (1989)
Ref_id:b40 Title: Pruning filters for efficient convnets Year: (2017)
Ref_id:b41 Title: Provable filter pruning for efficient neural networks Year: (2020)
Ref_id:b42 Title: Hrank: Filter pruning using high-rank feature map Year: (2020-06)
Ref_id:b43 Title: Discrimination-aware network pruning for deep model compression Year: (2022)
Ref_id:b44 Title: Group fisher pruning for practical network compression Year: (2021-07)
Ref_id:b45 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021-10)
Ref_id:b46 Title: Thinet: A filter level pruning method for deep neural network compression Year: (2017-10)
Ref_id:b47 Title: On the structural pruning of large language models Year: (2023)
Ref_id:b48 Title: ShortGPT: Layers in large language models are more redundant than you expect Year: (2025-07)
Ref_id:b49 Title: Pointer sentinel mixture models Year: (2017)
Ref_id:b50 Title: Recent advances in natural language processing via large pre-trained language models: A survey Year: (2023-09)
Ref_id:b51 Title: Fast model editing at scale Year: (2022)
Ref_id:b52 Title: Pruning convolutional neural networks for resource efficient inference Year: (2017)
Ref_id:b53 Title: Importance estimation for neural network pruning Year: (2019-06)
Ref_id:b54 Title: DisCEdit: Model editing by identifying discriminative components Year: (2024)
Ref_id:b55 Title: TVSPrune -pruning nondiscriminative filters via total variation separability of intermediate representations without fine tuning Year: (2023)
Ref_id:b56 Title: The deep bootstrap framework: Good online learners are good offline generalizers Year: (2021)
Ref_id:b57 Title: DFPC: Data flow driven pruning of coupled channels without data Year: (2023)
Ref_id:b58 Title: A survey of machine unlearning Year: (2025-09)
Ref_id:b59 Title: Pytorch: An imperative style, highperformance deep learning library Year: (2019)
Ref_id:b60 Title: Optimizing dnn architectures for high speed autonomous navigation in gps denied environments on edge devices Year: (2019)
Ref_id:b61 Title: Lipsformer: Introducing lipschitz continuity to vision transformers Year: (2023)
Ref_id:b62 Title: A layer selection approach to test time adaptation Year: (2025)
Ref_id:b63 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2019)
Ref_id:b64 Title: Editing a classifier by rewriting its prediction rules Year: (2021)
Ref_id:b65 Title: Remember what you want to forget: Algorithms for machine unlearning Year: (2021)
Ref_id:b66 Title: Decomposing and editing predictions by modeling model computation Year: (2024-07-27)
Ref_id:b67 Title: Structural pruning via latency-saliency knapsack Year: (2022)
Ref_id:b68 Title: Numerical pruning for efficient autoregressive models Year: (2025-04)
Ref_id:b69 Title: Efficient acceleration of deep learning inference on resource-constrained edge devices: A review Year: (2023)
Ref_id:b70 Title: Very deep convolutional networks for large-scale image recognition Year: (2015)
Ref_id:b71 Title: CHIP: CHannel Independence-based pruning for compact neural networks Year: (2021)
Ref_id:b72 Title: A simple and effective pruning approach for large language models Year: (2024)
Ref_id:b73 Title: Large language models for data annotation and synthesis: A survey Year: (2024-11)
Ref_id:b74 Title: Pruning neural networks without any data by iteratively conserving synaptic flow Year: (2020)
Ref_id:b75 Title: Stanford alpaca: An instruction-following llama model Year: (2023)
Ref_id:b76 Title: Unrolling sgd: Understanding factors influencing machine unlearning Year: (2022)
Ref_id:b77 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b78 Title: Pruning neural networks via coresets and convex geometry: Towards no assumptions Year: (2022)
Ref_id:b79 Title: From adaptive query release to machine unlearning Year: (2023-07)
Ref_id:b80 Title: Neural pruning via growing regularization Year: ()
Ref_id:b81 Title: Federated unlearning via class-discriminative pruning Year: (2022)
Ref_id:b82 Title: A comprehensive survey of continual learning: Theory, method and application Year: (2024)
Ref_id:b83 Title: Machine unlearning of features and labels Year: (2023)
Ref_id:b84 Title: Huggingface's transformers: State-of-the-art natural language processing Year: (2020)
Ref_id:b85 Title: Sheared llama: Accelerating language model pre-training via structured pruning Year: (2024)
Ref_id:b86 Title: Exact-fun: An exact and efficient federated unlearning approach Year: (2023)
Ref_id:b87 Title: Machine unlearning: Solutions and challenges Year: (2024)
Ref_id:b88 Title: Arcane: An efficient architecture for exact machine unlearning Year: ()
Ref_id:b89 Title: Dreaming to distill: Data-free knowledge transfer via deepinversion Year: (2020-06)
Ref_id:b90 Title: Width & depth pruning for vision transformers Year: (2022-06)
Ref_id:b91 Title: Nisp: Pruning networks using neuron importance score propagation Year: (2018)
Ref_id:b92 Title: Unified visual transformer compression Year: (2022)
Ref_id:b93 Title: HellaSwag: Can a machine really finish your sentence? Year: (2019-07)
Ref_id:b94 Title: A survey on neural network interpretability Year: (2021)
Ref_id:b95 Title: A survey on model compression for large language models Year: (2024)
Ref_id:b96 Title: NeurIPS Paper Checklist 1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We link each contribution in the paper's contents when we theoretically or empirically justify the claims. size of 128 using SGD optimizer with momentum factor of 0.9 with initial learning rate of 0.01 and a MultiStepLR learning rate scheduler with milestones at 60 and 80 epochs Year: ()
