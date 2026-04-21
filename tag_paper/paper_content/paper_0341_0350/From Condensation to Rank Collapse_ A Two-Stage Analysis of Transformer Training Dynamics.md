Title: From Condensation to Rank Collapse: A Two-Stage Analysis of Transformer Training Dynamics
Abstract: Although transformer-based models have shown exceptional empirical performance, the fundamental principles governing their training dynamics are inadequately characterized beyond configuration-specific studies. Inspired by empirical evidence showing improved reasoning capabilities under small initialization scales in language models, we employ the gradient flow analytical framework established in Zhou et al. [2022] to systematically investigate linearized Transformer training dynamics. Our theoretical analysis dissects the dynamics of attention modules into two distinct stages. In the first stage, asymmetric weight perturbations from random initialization sustain non-degenerate gradient dynamics in parameter matrices, facilitating systematic escape from small initialization regimes. Subsequently, these matrices undergo condensation, progressively aligning toward the target orientation. In the second stage, the previously static key-query matrices actively participate in training, driving the normalized matrices toward asymptotic rank collapse. This two-stage framework generalizes classical directional convergence results.

Section: Introduction
The transformer-based models Vaswani et al. [2017] have achieved remarkable breakthroughs in various fields, with the successful application of large language models. However, the theoretical analysis of the transformer still remains in specific tasks, such as in-context learning settings Brown et al. [2020], Olsson et al. [2022], Bietti et al. [2023] or single attention block with reparameterization Tian et al. [2023]. The use of linear regression tasks Zhang et al. [2024a] and Markov chain tasks Ildiz et al. [2024] has provided highly interpretable theoretical analyses, but a crucial question still remains: Can we analyze the characteristics of the training dynamics of transformers independently of specific tasks? Meanwhile, small initialization has been increasingly shown to hold promise in the training process of large models, especially for reasoning tasks. Numerous studies Zhang et al. [2024bZhang et al. [ , 2025b]], Yao et al. [2025] suggest that the implicit regularization effect of small initialization is still effective in large language models. This effectiveness is particularly significant in the context of modern large models, which are characterized by extreme overparameterization. In these regimes, where explicit regularization techniques like weight decay or dropout may prove insufficient on their own, implicit regularization becomes pivotal. It operates by imposing intrinsic constraints on the training dynamics and the resulting parameter space, effectively guiding the model towards solutions with good generalization properties despite the vast hypothesis space. This implicit bias is key to understanding how models with such immense capacity manage to avoid severe overfitting and achieve remarkable performance on unseen data.
Motivated by these observations, we propose to investigate the training dynamics of transformers under a small initialization setting. Leveraging the gradient flow theme similarly to Zhou et al. [2022], We delineate different training dynamics for outer parameters versus attention parameters W Q and W K in Transformers.
We dissect the dynamics of attention modules into two distinct stages. In the first stage, the core attention mechanism, softmax(QK ⊺ ), remains nearly stagnant, as asymmetric weight perturbations from random initialization drive non-degenerate gradient dynamics in parameter matrices, particularly W V , facilitating escape from small initialization regimes. During this escape, the parameter matrix converges row-wise toward the target orientation, a process we term condensation. We theoretically prove that condensation is guaranteed under small initialization, and experimentally observe that it stabilizes without significant fluctuations.
In the second stage, after the outer parameters, such as W V , reach a quasi-steady state, the previously static key-query matrices, W Q and W K , begin to actively participate in training, driving their collapse. This two-stage framework not only elucidates the training dynamics but also generalizes classical directional convergence results, offering a robust theoretical foundation for Transformer optimization.
To sum up, our contribution can be summarized as follows.
1. Blow-up Dynamics: We prove the blow-up property (Theorem 1) holds for measure-theoretically generic initializations, eliminating reliance on dichotomy assumptions while ensuring model nondegeneracy. 2. Condensation Mechanism: By introducing a condensation condition (Assumption 1), we establish theoretical guarantees for condensation emergence (Theorem 2). 3. Key-Query Collapse: After outer parameters stabilize in a quasi-steady state (Assumption 2), the key-query matrices begin active training, leading to asymptotic rank collapse of the normalized key-query matrices (Theorem 3). 4. Experimental evidence: We validate our hypotheses and theoretical predictions on both synthetic and real datasets with one and multi-layer Transformers, consistently observing two-stage dynamics marked by condensation and an eventual rank collapse of the normalized key-query matrices (Figure 1, 2, 3).
this section cite: ['b40', 'b7', 'b30', 'b6', 'b37', 'b16', 'b43', 'b52']

Section: Related Works
Training dynamics of transformer. Given the scale of modern models and the complexity of optimizers, studying the training dynamics of Transformers is a challenging problem. Prior works have primarily investigated the optimization dynamics of a single attention layer Lu et al. [2021], Li et al. [2023], Snell et al. [2021]. However, these studies mainly focused on specific tasks, such as topic structure prediction and translation.
Recently, the dynamics of in-context learning (ICL) has emerged as a prominent research area within Transformer dynamics, particularly given ICL's ability to solve novel tasks without parameter updates. Many works Mahankali et al. [2024], Zhang et al. [2024a], Huang et al. [2023], Collins et al. [2024] have focused on the linear regression setup to theoretically investigate the mechanism of ICL in single-layer Transformers, a line of work that has also informed algorithmic development Akyürek et al. [2023], Bai et al. [2023], Guo et al. [2024]. Another line of research investigates how specific structures within attention emerge during training, notably starting with studies on induction heads Olsson et al. [2022], Reddy [2024], Edelman et al. [2024], Zhang et al. [2025a], memory recall mechanisms Bietti et al. [2023], Cabannes et al. [2024], and even causal structure Nichani et al. [2024].
Despite the sophisticated structure of realistic Transformers, Tian et al. [2024] proposed a novel mathematical framework for analyzing the joint dynamics of MLP and attention blocks and successfully explained the sparsity of attention score matrices. Meanwhile, Chen et al. [2024a] provides a rigorous proof for the convergence of the ICL linear regression task using gradient flow with sufficiently small initialization.
this section cite: ['b22', 'b21', 'b34', 'b25', 'b15', 'b11', 'b0', 'b4', 'b13', 'b30', 'b31', 'b12', 'b6', 'b8', 'b29', 'b38']

Section: Small initialization and its applications
The initialization of a neural network significantly affects its learning outcomes Arora et al. [2019b], Williams et al. [2019], Mei et al. [2018], Jacot et al. [2018], Rotskoff and Vanden-Eijnden [2018], Zhang et al. [2020]. Small initialization is a common setting investigated in the study of neural network optimization dynamics, which is different with the Neural Tangent Kernel (NTK) perspective in infinitely wide networks. For linear model, Ji and Telgarsky [2019] establish matrix alignment results theoretically. For nonlinear model, Zhou et al. [2022] found that small initialization can similarly promote parameter condensation, thereby reducing model complexity. Theoretically, Luo et al. [2021], Chen et al. [2024b], Zhou et al. [2023], Kumar and Haupt [2024] have deepened the understanding of this phenomenon. The recent survey article Xu et al. [2025] systematically synthesizes empirical and theoretical findings. . More recently, many researchers have adopted small initialization settings to simplify the analysis of training dynamics in more complex models. From a theoretical perspective, Zhang et al. [2025a] applied small initialization to ICL tasks to analyze the behavior of linear attention. Yao et al. [2025] considered the training dynamics of the embedding space under small initialization using a synthetic dataset designed for reasoning and memorization. From an applied perspective, Zhang et al. [2019], Huang et al. [2020], Zhu et al. [2021] highlighted the importance of initialization in Transformers, while Bachlechner et al. [2021] combined zero-initialization with residual blocks in Transformers. Some research Zhang et al. [2024b], Yao et al. [2025] shows that small initialization helps Transformers learn the reasoning aspects of data rather than just memorization, a principle already applied in realistic LLM training Yin et al. [2025].
this section cite: ['b41', 'b26', 'b17', 'b32', 'b47', 'b18', 'b52', 'b23', 'b53', 'b19', 'b43', 'b45', 'b14', 'b54', 'b3', 'b43', 'b44']

Section: Preliminaries

this section cite: []

Section: Basic Notations
First, we introduce some notations that will be used in the rest of this paper. Let n and d m be the number of samples and the width of hidden layers, respectively. Let [n] denote the set of integers from 1 to n. Denote vector L 2 norm as ∥ • ∥ 2 and matrix Frobenius norm as ∥ • ∥ F . Let ⟨•, •⟩ represent standard inner product between two vectors. For a vector v, denote its k-th entry as v k . For a matrix A, denote the element in the k-th row and k ′ -th column as A kk ′ . And denote k-th row as A k and k ′ -th column as A k ′ . Unless otherwise specified, summation ' ' is performed over the network width.
this section cite: []

Section: Classification Task
Binary classification: For decision tasks, the network produces a scalar output f θ (X) ∈ R. The predicted class assignment is determined by the sign of the output. The dataset is denoted by D = {(X i , y i )} n i=1 where X i ∈ R s×dm stands for input sequence in which s represents the sequence length and d m represents the hidden dimension, and y i ∈ {±1} stands for label. For a loss function ℓ : R → R + , we define the empirical risk as
L(θ) = 1 n n i=1 ℓ (y i f θ (X i )).
Multi-class Classification: For probabilistic tasks, the network outputs logit vectors f θ (X) ∈ R dv that parameterize a categorical distribution via the softmax transformation P(y
= i|X; θ) = exp(f θ (X)i) ∑ dv j=1 exp(f θ (X)j )
where d v denotes the vocabulary size. For cross-entropy loss, we define the empirical risk as
L(θ) = -1 n n i=1 log P(y = y i |X i ; θ).
this section cite: []

Section: Condensation and Rank Collapse
We formalize the two geometric phenomena that will recur throughout our analysis. Definition 1 (Condensation). Let W (t) be a matrix with rows W k (t) (or columns W k (t)). We say W condenses to a direction v if, as t → T ,
W k (t) ∥W k (t)∥2 , v → ±1 for every index k with ∥W k (t)∥ 2 ̸ = 0 (equivalently,
this section cite: []

Section: the same holds columnwise).
Condensation is a directional notion and implies rank-1 collapse when a unique direction emerges. Rank collapse is a spectral notion and allows k > 1 when multiple top singular directions are tied.
Definition 2 (Asymptotic rank collapse). Let W (t) be a matrix. We say W exhibits rank-k collapse if the limit
W ∞ := lim t→T W (t) ∥W (t)∥ F exists and rank(W ∞ ) ≤ k.
4 Theoretical Results
this section cite: []

Section: Problem Formulation
To analyze condensation phenomenon in transformers, we begin by formulating the problem. Specifically, we consider the following one-layer transformer model: Definition 3 (One-layer transformer). Let X ∈ R s×dm be an input sequence of length s with model dimension d m . The Transformer function f θ : R s×dm → R s is defined by the composition of attention and feed-forward operations: [2] .
f θ (X) := FFN(Attn(X)) = σ Attn(X)W [1] W
(
)1
The attention sublayer Attn : R s×dm → R s×dm is computed as:
Attn(X) = softmax XW Q W ⊺ K X ⊺ √ d m XW V , (2
)
where parameter matrices satisfy W Q , W K , W V , W [1] ∈ R dm×dm and W [2] ∈ R dm . The activation function σ : R → R is tanh.
We use one-layer transformer f θ to solve binary classification tasks and take the last dimension of the output f θ (X i ) s as the output. So the empirical risk to be minimized is given by
L(θ) = 1 n n i=1 ℓ(y i f θ (X i ) s ).(3)
For simplicity of presentation, we employ the exponential loss function ℓ(q) = e -q , which is commonly used in the analysis of classification tasks Lyu and Li [2020]. The analysis can be readily extended to other loss functions such as the logistic loss.
The model parameters are initialized with Gaussian distributions scaled by a small perturbation parameter ε:
W [2] k ∼ N (0, ε 2 ), W [1] kk ′ ∼ N (0, ε 2 ), W Q,kk ′ , W K,kk ′ , W V,kk ′ ∼ N (0, ε 2 ),(4)
where ε ≪ 1 controls initialization magnitude. To analyze training dynamics, we adopt the gradient flow (GF) framework-the continuous-time limit of gradient descent. Given the small initialization scale, we derive effective dynamics through a perturbative expansion of the empirical risk L(θ) in powers of ε.
First, we normalize parameters by absorbing the initialization scale:
W [2] = ε -1 W [2] , W [1] = ε -1 W [1] , WQ = ε -1 W Q , WK = ε -1 W K , WV = ε -1 W V .
Performing a Taylor expansion of L(θ) about ε = 0 yields the leading-order asymptotic form:
L(θ) = 1 2n n i=1   1 -ε 3   s j=1 1 s y i X i,j WV W [1] W [2]   + o(ε 3 )   . (5
)
This expansion induces simplified gradient dynamics characterized by the following proposition.
Proposition 1 (Effective training dynamics). Given a binary dataset {(X i , y i )} n i=1 , we define condensation direction v and rescaled time coordinate t as follows:
v := n i=1 y i s j=1 X i,j n i=1 y i s j=1 X i,j 2 , t := ε ns n i=1 y i   s j=1 X i,j   2 t. (6
)
Then, normalized parameters θ follow leading-order dynamics after rescaling:
d θ d t = ∇ θ v WV W [1] W [2] .(7)
Proposition 1 reveals a hierarchical learning mechanism: During initial training phases, the fullyconnected layers W [1] , W [2] and value projection matrix W V exhibit substantial updates, while the query/key matrices W Q and W K in the self-attention module remain quasi-static.
For subsequent analysis, we define the projection of W V onto v as W v := vW V (omitting bar notation for simplicity) and introduce the energy functional:
E := W v W [1] W [2] . (8
)
The effective dynamics can thus be interpreted as gradient ascent on this energy landscape.
this section cite: ['b24']

Section: Blow Up Dynamics
We first elucidate why Transformers with small initialization can successfully train and eventually escape the small initialization regime. This phenomenon emerges from the interplay between two fundamental mechanisms:
1. Effective dynamics driving: The parameter evolution governed by the effective dynamics exhibits remarkable symmetry, manifested through strict conservation laws that preserve key quantities during training. 2. Random normal initialization: While degenerate cases theoretically exist under Gaussian initialization, they occur with vanishing probability (measure zero in parameter space). Consequently, the dynamics almost surely demonstrate non-degenerate characteristics, ensuring stable training trajectories.
To preserve dynamical symmetry, we invoke the following proposition following the approach established in prior works Ji and Telgarsky [2019]: Proposition 2 (Conservation laws). Under the gradient flow dynamics prescribed by system Eq. ( 7), the following system of conservation laws emerges:
d dt W 2 v,k - k ′ W [1] kk ′ 2 = 0 and d dt W [2] k 2 - k ′ W [1] k ′ k 2 = 0. (9
)
We now analyze the non-symmetric property arising from Gaussian random initialization, with particular focus on the degeneracy mechanism. Crucially, we establish that degeneracy exclusively occurs when initialization violates the following non-degenerate initialization: Definition 4 (Non-degenerate initialization). Let θ = W V , W [1] , W [2] denote parameters initialized from a Gaussian distribution. The initialization is called non
-degenerate if ∥W v ∥ 2 2 ̸ = ∥W [2] ∥ 2 2 and ∥ Ẇv ∥ 2 2 -∥ Ẇ [2] ∥ 2 2 + min ∥W v ∥ 2 2 , ∥W [2] ∥ 2 2 ∥W v ∥ 2 2 -∥W [2] ∥ 2 2 ̸ = 0.(10)
Having clarified the definition of non-degenerate initialization, we present the following theorem that reveals the non-degeneracy property of effective training dynamics.
Theorem 1 (Blow-up in finite time).
Let the parameters be initialized randomly as above from a Gaussian distribution. Then, almost surely, the initialization is non-degenerate in the sense of Definition 4, and the effective training dynamics Eq. (7) blows up in finite time. That is, there exists T * > 0 such that lim t→T * E(t) = +∞. Proof sketch. We prove finitetime blow-up via a Riccati-type differential inequality for the energy E(t). Full technical details are provided in Appendix A.1. (1) Superlinear growth. A direct computation gives Ė(t) ≥ 3E(t) 4/3 , hence ∂ t E(t) -1/3 ≤ -1 and
E(t) ≥ 1 E(0) -1/3 -t 3 . (11
)
For E(0) > 0 this yields T * ≤ E(0) -1/3 .
(2) Negative initial energy. If E(0) ≤ 0, then
E(t) ≥ - 1 (-E(0)) -1/3 + t 3 , (12
)
so E is increasing and cannot remain negative indefinitely. Assuming E(t) ≤ 0 for all t leads to contradictions with (i) standard continuation at finite T * , or (ii) monotone limits at T * = ∞, reducing to the borderline case E(t) ↑ 0.
(3) Borderline exclusion. In the regime T * = ∞ and E(t) ↑ 0, structural identities and conservation give ∥W v (t)∥ 2 2 ∥W [2] (t)∥ 2 2 → 0 and Ė(t) → 0. Under the non-degenerate initialization (Def. 4), this forces a contradiction, since the limiting Ė must be strictly positive.
this section cite: ['b18']

Section: Condensation Dynamics
We have proved that energy and parameters norm will blow up almost surely. It implies the effective dynamics drive parameters escape small initialization area in finite time. The next question is how the effective dynamics affects the emergence of condensation and whether there exist observables to help us characterize condensation.
We propose a condition of condensation and verify its effectiveness using experimental and theoretical methods. In particular, we theoretically prove that the solution of the effective dynamics has specific properties, which is to some extent a sufficiency argument. The necessity argument is quite difficult in theory. But experimental results provide us a strong implication that this condition maybe also necessary.
this section cite: []

Section: Assumption 1 (Condensation condition). The parameters satisfy the condensation condition at time
t. That is 1. For each index i ∈ [d m ] , W [2] i W v W [1],i > 0 and W v,i W [1] i W [2] > 0. 2. For each pair i, j ∈ [d m ], ⟨W [2] i W [1],i , W [2] j W [1],j ⟩ > 0, and ⟨W v,i W [1] i , W v,j W [1] j ⟩ > 0.
This hypothesis can be verified experimentally in Sec. 5.1.2. Then, based on Assumption 1, we formalize the statement of the culminating theorem as follows:
Theorem 2 (Condensation). Under Assumption 1, the effective dynamical system governed by Eq. ( 7) drives the parameter matrix W V to undergo condensation in the sense of Definition 1.
This section gives a highlevel proof sketch; full details appear in Appendix A.2.
Proof sketch. We establish finitetime directional convergence (condensation) via geometric propagation and twosided energy control.
(1) Geometric consistency and alignment dynamics. Under Assumption 1, Proposition 4 shows that once the alignment condition holds at some t 0 < T * , it propagates throughout (t 0 , T * ). Proposition 5 yields a structural dichotomy of the columns of W V into a condensing class C 1 and a uniformly bounded class C 2 . Propositions 6 and 7 further establish dynamical alignment between W [2] and its time derivative, ensuring coherence of the evolving direction.
(2) Singularity structure and condensation. Proposition 8 supplies an energy upper bound which, combined with the lower bound in Eq. ( 11), furnishes a bilateral estimate on E(t). A telescopingintegral argument then proves that the condensing indices dominate in finite time, completing the proof of condensation via Theorem 2.
this section cite: []

Section: Key-Query Dynamics
Following the initial training stage, the parameter matrices W V , W [1] and W [2] exhibit substantial growth in magnitude, effectively escaping the small-initialization regime. In contrast, the key-query matrices W Q and W K demonstrate remarkable stability in scale. This separation phenomenon is fundamentally governed by the effective dynamics of the learning system.
A pivotal question arises: Under what conditions do the key-query matrices become dynamically activated, thereby enabling the attention mechanism to exert its structural influence? We hypothesize that during early training, W V , W [1] and W [2] converge to a critical point where W Q and W K almost vanish, temporarily stabilizing in this dormant state. The following analysis provides mechanistic insights into this dynamical freezing phenomenon. The final activation function is omitted from our analysis. This is justified because the layer's pre-activations consistently operate within the linear regime of the function. Furthermore, empirical results confirm that its inclusion does not alter the model's learning dynamics (refer to B.2). Now empirical loss L(θ) has the following decomposition:
L(θ) ≈ 1 n n i=1 L 1,i (θ)L 2,i (θ),(13)
where
L 1,i (θ) =exp -y i s j=1 1 s X i,j W V W [1] W [2] , L 2,i (θ) =exp -y i s j=1 1 s Xi,sW Q W ⊺ K X ⊺ i,j √ dm -1 s 2 s l=1 Xi,sW Q W ⊺ K X ⊺ i,l √ dm X i,j W V W [1] W [2] .
Based on the above discussion, we now formalize the following assumption.
Assumption 2 (Dynamics separation stage). After the breakdown of the effective dynamics in Eq. ( 7), let δ denote a small parameter. The gradient flow subsequently enters a stage characterized by:
1. Criticality conditions: The outer parameters (W V , W [1] , W [2] ) converge to a quasi-stationary configuration such that
∇ W V L 1 = ∇ W [1] L 1 = ∇ W [2] L 1 = O(δ 2 ), where L 1 = 1 n i L 1,i . 2. Key-query stunting: The attention parameters remain small, satisfying |W Q ij |, |W K ij | = O(δ),
until their norms ∥W Q ∥ and ∥W K ∥ exceed a critical scale.
To facilitate empirical validation, we introduce a modified version of the basic equivalence of the first part of Assumption 2, denoted as Assumption 2*. Assumption 2*. The outer parameters (W V , W [1] , W [2] ) reach a quasi-stationary state whose directions vary negligibly over time, i.e., d dt
W V ∥W V ∥ = d dt W [1] ∥W [1] ∥ = d dt W [2]
∥W [2] ∥ ≈ 0, and the loss evolution satisfies dL1,i dt = O(δ 2 ) for all i. Since we assume the small parameter δ is still relatively small, we get the following Proposition which illustrate evident dynamics separation and the leading order dynamics of key-query matrices. Proposition 3 (Effective dynamics during dynamics separation stage). Under Assumption 2 or Assumption 2*, the empirical risk L(θ) exhibits the following properties:
1. Dynamics separation The gradients of the empirical risk with respect to W V , W [1] and W [2]  are of order O(δ 2 ), while the gradients with respect to the query matrix W Q and key W K are of order O(δ).
this section cite: []

Section: Key-query dynamics
Treating W V , W [1] and W [2] as fixed due to dynamics separation, the leading-order dynamics of key-query matrices are given by
dW Q dt = F W K , dW K dt = F ⊺ W Q , (14
)
where F is defined as follows
F = 1 ns √ d m n i=1 y i L 1,i X ⊺ i,s W [2] ⊺ W [1] ⊺ W V ⊺   s j=1 X ⊺ i,j X i,j - 1 s s l=1 X i,l   . (15
)
Since the dynamics governing W Q and W K form a linear ordinary differential equation system in this context, we can rigorously establish the subsequent conclusions.
Theorem 3 (Asymptotic rank collapse). Given the key-query dynamics governed by Eq. ( 14), the normalized key and query matrices exhibit rank collapse as Definition 2. Specifically, when F possesses a unique largest singular value, both normalized matrices asymptotically become rank 1.
The detailed proofs of Proposition 3 and Theorem 3 can be found in the Appendix A.3.
this section cite: []

Section: Experimental Results
In this section, we first demonstrate the phenomena of cohesion and rank collapse using synthetic data and confirm the assumptions required for our theoretical analysis of the one-layer Transformer model. We then present experiments on natural language processing tasks to demonstrate the generality of our theoretical findings with respect to various datasets and network architectures.
this section cite: []

Section: Synthetic Dataset
We employ the concept of the anchor function Zhang et al. [2024c] to construct a synthetic dataset that simulates a simplified language modeling scenario. The model is a one-layer Transformer with tanh activation, trained using cross-entropy loss and the AdamW optimizer. Further experimental settings are detailed in Appendix B.1.
this section cite: []

Section: Phenomenon: Condensation and Rank Collapse
To dissect the learning dynamics, we visualize the training process through three complementary lenses: the cosine similarity of parameters (Calculation method refer to Sec. B.1), the relative change of norms, and the effective rank of weight matrices. As shown in Figure 1, these analyses collectively reveal a distinct three-stage training trajectory, which we characterize as Condensation, Key-Query Rank Collapse, and the further training. The training process begins with a rapid decrease in loss, driven almost exclusively by the outerlayer parameters since the relative change of the outer parameters far exceed those of the attention parameters (Fig. 1(b)) during this initial phase. This intense optimization leads to the condensation phenomenon, where the initially random outer parameters organize into a low-rank configuration. This is visually evident from the emergence of block structures in their cosine similarity matrices (Fig. 1(a)) and is quantified by a monotonic and significant decrease in their effective rank (Fig. 1(c)). Throughout this stage, the attention parameters remain largely static and unstructured.
Following the initial phase, the training loss enters a prolonged plateau. This signals a critical transition in the learning dynamics, marked by the gray dashed line in Fig. 1 (b). At this stage, a clear dynamics separation occurs: the updates to the outer parameters subside, and the attention parameters become the primary focus of optimization. This empirical observation validates our theoretical framework, particularly Proposition 3. As the changes in the outer parameters become slower (supporting Assumption 2), the attention parameters begin to learn their specialized roles. This is characterized by a rank collapse, confirmed visually by the sudden formation of structure in their similarity matrices and quantitatively by a precipitous drop in their effective rank (Fig. 1(a), 1(c)).
this section cite: []

Section: Experimental Validation of Key Assumptions
To ground our theoretical analysis in the observed dynamics, we now provide direct empirical validation for the key assumptions that underpin our framework: Assumption 1 and Assumption 2.
1 dm dm i=1 cos(u i t , u i t+1 ) (or 1 dm dm i=1 cos(v i t , v i t+1 )). (c) Frobenius norms of parameter groups.
First, we examine the condensation condition. Figure 2(a) plots the proportion of satisfied conditions in Assumption 1. The proportion rapidly approaches 1 within the first 200 training steps, confirming that the outer parameters quickly converge to a state where this assumption holds.
Next, we validate the assumption of dynamics separation. As discussed in the previous section, our observation that the gradual change of outer parameters during Stage 2 and flat loss curve (often means a critical point has appeared) already provide strong qualitative support for the first part of Assumption 2. To analyze this more rigorously, we examine its empirical variant, Assumption 2*. This assumption points that the direction of parameters remains unchanged and the leading-order loss changes very slowly.
Figure 2(b) shows the cosine similarity between the singular vectors of the outer parameter matrices at adjacent time steps. The similarity for all outer parameters remains extremely close to 1 after the first stage. This indicates that the subspace spanned by these parameters is highly stable, meaning their directional structure is effectively frozen. This stability, combined with the flat loss curve observed in Stage 2, provides compelling evidence for Assumption 2*. Figure 2(c) validates the scale separation implied by the assumption. It shows that by the onset of Stage 2, the Frobenius norms of the outer parameters have grown significantly, while the norms of the attention parameters remain small and close to their initialization values. This confirms the expected scale difference between the two parameter groups, where outer parameters are O(1) and attention parameters are O(δ).
this section cite: []

Section: Real Task
We further validate our theoretical predictions on a real-world language modeling benchmark, Wiki-Text Merity et al. [2017]. Unlike the synthetic setup, where anchor functions are explicitly defined, WikiText provides natural linguistic dependencies and high distributional variability. This allows us to test whether the proposed two-stage dynamics, early condensation of outer parameters followed by attention-driven rank collapse, persist in realistic Transformer training. In this setting, we employ a two-layer transformer with GeLU activation and residual connections. To keep the consistency of architecture and focused on the core dynamics, layer normalization is omitted. Further experimental settings are provided in Appendix B.3.
As shown in Figure 3, the two-layer Transformer on WikiText exhibits the same stage-wise dynamics observed in the synthetic experiments. During the initial phase, the outer parameters (W V , W [1] , W [2] ) in both layers undergo rapid condensation, while the attention weights (W Q , W K ) remain largely unchanged. As training proceeds and the loss enters a plateau, the attention parameters begin to evolve, displaying a sharp rank collapse that reorganizes internal representations.
This empirical observation confirms that the separation between outer-parameter condensation and attention-driven rank reduction is not an artifact of the synthetic dataset but also emerges naturally in real-world text modeling. The consistent appearance of this two-stage dynamic across both synthetic and natural settings suggests that implicit regularization, first through low-rank condensation and then through targeted attention adaptation, may serve as a general mechanism underlying the emergence of structured representations in Transformer models.
this section cite: ['b27']

Section: Discussion

this section cite: []

Section: Conclusion
This work advances the theoretical understanding of transformer training dynamics by establishing a two-stage analytical framework. Through gradient flow analysis, we show that small initialization helps models escape degenerate regions via asymmetric weight updates, leading to condensation of parameter matrices toward task-relevant directions. In the subsequent stage, the key-query matrices undergo a coordinated collapse that further refines the learned representations. Together, these results clarify the mechanisms underlying the condensation and rank collapse phenomena, providing a principled foundation for future studies on Transformer optimization and generalization.
this section cite: []

Section: Limitations
While this work provides valuable theoretical insights, its most significant constraint stems from analyzing exclusively binary classification scenarios: a simplification dictated by technical barriers in gradient flow analysis. This narrow scope inherently precludes insights into transformers' dynamics in practical multi-class classification or sequence-to-sequence learning contexts, where complex interactions between multiple prediction targets and attention mechanisms likely emerge. Though focused theoretical simplification is methodologically justified, extending this framework to broader problem domains remains critical for unifying theory with real-world transformer optimization. Future work should prioritize overcoming these technical limitations to theoretically verify whether our conclusions hold true beyond binary settings.
this section cite: []

Section: NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
• You should answer • [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• Please provide a short (12 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
IMPORTANT, please:
• Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist", • Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We elaborate on our setups and contribution in the abstract and introduction, especially in the last paragraph of the introduction.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The limitations could be found at Sec.6.2. Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: We provide the definitions, assumptions and proofs at Sec. 4 and Appendix A.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We show the experiment setup in Sec. B. Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The code is provided in the supplementary materials.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so No is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental setting/details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We show the experiment setup in Sec. B. Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We show the error bar in Fig. 1 for the anchor function experiments and Fig. 3 for WikiText task.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We provide the information of compute resources in Appendix. C. Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The research conducted in the paper conform with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10. Broader impacts Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This work is a phenomenological study, therefore, there is no societal impact of the work performed.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11. Safeguards Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: Justification: This work is a phenomenological study, therefore, this work poses no such risks.
this section cite: []

Section: A Theory Details
A.1 Theory Details for Blow up Dynamics
this section cite: []

Section: A.1.1 Proof for Proposition 2
Proof. Taking advantage of the inherent symmetry of the system, the proof focuses on analyzing the coupled dynamics of W v and W [1] :
d dt (W v,k ) 2 = 2 k ′ W v,k W [1] kk ′ W [2] k ′ = d dt k ′ W [1] kk ′ 2 .
This finished the proof of the first two equations. We also derive the relation between energy E and the evolution of parameters
d dt ∥W v ∥ 2 2 = d dt k W 2 v,k = 2 k,k ′ W v,k W [1] kk ′ W [2] k ′ = 2E.
This is just the third equation.
this section cite: []

Section: A.1.2 Proof for Theorem 1
Proof. Since we use Gaussian random initialization, the initialization satisfies Definition 4 almost surely. Therefore, we establish our results under the assumptions specified in Definition 4. By local Lipshcitz condition on the right hand side of dynamical system Eq. ( 7), it has a solution for t ∈ (0, T * ) where T * is maximum existence time of solution and can be infinity. Taking derivative of E, we obtain
Ė = d dt W v W [1] W [2] = ∥ Ẇv ∥ 2 2 + ∥ Ẇ [2] ∥ 2 2 + ∥W v ∥ 2 2 ∥W [2] ∥ 2 2 . (16
)
The inequality of arithmetic and geometric means leads to
Ė ≥ 3(∥ Ẇv ∥ 2 2 ∥ Ẇ [2] ∥ 2 2 ∥W v ∥ 2 2 ∥W [2] ∥ 2 2 ) 1 3 ≥ [⟨ Ẇv W v ⟩ 2 ⟨ Ẇ [2] W [2] ⟩ 2 ] 1 3 = 3E 4 3 .
This implies that energy E increase monotonically. If E(0) > 0,
d dt E -1 3 ≤ -1.
Integrating both sides of the inequality yields a lower bound for the energy E
E(t) ≥ 1 (E(0) -1 3 -t) 3 . (17
)
Thus, in the case where E(0) > 0, the dynamical system explodes before T * ≤ E(0) -1 3 . In the case where E(0) ≤ 0, we consider -E(t) instead, and obtain
E(t) ≥ - 1 (t + (-E(0)) -1 3 ) 3 . (18
)
We claim that there exists some time t 0 > 0, such that E(t 0 ) > 0. This claim can be proved by contradiction.
Suppose that E(t) ≤ 0 for all 0 < t < T * . Recall that Ė(t) ≥ 0 throughout this interval. The boundedness of W v , W [1] , and W [2] , together with the monotonicity of energy E, implies that E(T * ) = lim t→T * E(t) exists and satisfies -∞ < E(T * ) ≤ 0. We now consider different cases separately.
(i) The case of T * < +∞. The solutions can be extended to a time larger than T * since ∥W v ∥ 2 , ∥W [1] ∥ F , ∥W [2] ∥ 2 are bounded due to the conservation law. This contradicts the definition of T * .
(ii) The case of T * = +∞ and E(T * ) < 0. That is lim t→+∞ E(t) < 0. However, this contradicts Eq. ( 18).
(iii) The case of T * = +∞ and E(T * ) = 0. That is lim t→+∞ E(t) = 0. We prove this case in three steps.
Step 1: We show that lim t→∞ ∥W v (t)∥ 2 2 ∥W [2] (t)∥ 2 2 = 0. Since E(t) ≤ 0 for all t, the quantities ∥W v ∥ 2 2 , ∥W [1] ∥ 2 F , ∥W [2] ∥ 2 2 are monotonically decreasing. However, each of them is bounded below by zero, and hence they all converge to finite limits and remain uniformly bounded.
Moreover, note that Ė ≥ 0 for all t. If lim inf t→∞ Ė(t) > 0, it contradicts the fact that lim t→∞ E(t) = 0. Therefore, it must hold that
lim t→∞ ∥W v (t)∥ 2 2 ∥W [2] (t)∥ 2 2 = 0.
This implies that either
lim t→∞ ∥W v (t)∥ 2 2 = 0 or lim t→∞ ∥W [2] (t)∥ 2 2 = 0. We can also obtain lim t→∞ ∥ Ẇ [1] (t)∥ 2 F = 0 since Ẇ [1] = W ⊺ v W [2] . Step 2: We show lim t→∞ Ė(t) = 0.
Without loss of generality, we assume that lim t→∞ ∥W [2] (t)∥ 2 2 = 0. The case of
lim t→∞ ∥W v (t)∥ 2 2 = 0 is similar. That is ∥W v (0)∥ 2 > ∥W [2] ∥ 2 . By conservation law, we have lim t→∞ ∥ Ẇv (t)∥ 2 2 = 0.
Considering the second derivative of W [2] k ′ , we obtain
Ẅ [2] k ′ = k l W [1] kl W [2] l W [1] kk ′ + W 2 v,k W [1] k ′ . Thus lim t→∞ ∥ Ẅ [1] (t)∥ 2 2 = 0 since lim t→∞ ∥W [1] (t)∥ 2 2 = 0. Note that Ė(t) ≥ 0 and lim t→∞ E(t) = 0. Recall that lim inf t→∞ Ė(t) = 0 and Ė are bounded. And we also have ∥ Ẇ [2] ∥ 2 2 ≤ M . We claim that lim t→∞ ∥ Ẇ [2] (t)∥ 2 2 = 0, which implies lim t→∞ Ė(t) = 0.(19)
Since lim t→∞ ∥W [2] (t)∥ 2 = 0 and lim t→∞ ∥ Ẅ [2] (t)∥ 2 = 0. Using Taylor expansion, we have for some φ ∈ [t, t + 1]
W [2] k (t + 1) = W [2] k (t) + Ẇ [2] k (t) + 1 2 Ẅ [2] k (φ), ∀k, which implies lim t→∞ ∥ Ẇv (t)∥ 2 = 0.
Therefore, the assertion holds.
Step 3: We show that Eq. ( 19) contradicts the condition in Definition 4. By direct calculation, we
obtain d dt ∥ Ẇv ∥ 2 2 = d dt (W [2] ⊺ W [1] ⊺ W [1] W [2] ) = Ẇ [2] ⊺ W [1] ⊺ W [1] W [2] + W [2] ⊺ Ẇ [1] ⊺ W [1] W [2] + W [2] ⊺ W [1] ⊺ Ẇ [1] W [2] + W [2] ⊺ W [1] ⊺ W [1] Ẇ [2] = W v W [1] W [1] ⊺ W [1] W [2] + W [2] ⊺ W [2] W v W [1] W [2] + W [2] ⊺ W [1] ⊺ W ⊺ v W [2] ⊺ W [2] + W [2] ⊺ W [1] ⊺ W [1] W [1] ⊺ W ⊺ v = 2E∥W [1] ∥ 2 2 + 2 Ẇv W [1] Ẇ [2] and d dt ∥ Ẇ [2] ∥ 2 2 = d dt (W v W [1] W [1] ⊺ W ⊺ v ) = Ẇv W [1] W [1] ⊺ W ⊺ v + W v Ẇ [1] W [1] ⊺ W ⊺ v + W v W [1] Ẇ [1] ⊺ W ⊺ v + W v W [1] W [1] ⊺ Ẇ ⊺ v = W [2] ⊺ W [1] ⊺ W [1] W [1] ⊺ W ⊺ v + W v W ⊺ v W [2] ⊺ W [1] ⊺ W ⊺ v + W v W [1] W [2] W v W ⊺ v + W v W [1] W [1] ⊺ W [1] W [2] = 2E∥W v ∥ 2 2 + 2 Ẇv W [1] Ẇ [2] .
Therefore,
d dt ∥ Ẇv ∥ 2 2 - d dt ∥ Ẇ [2] ∥ 2 2 = 2E(∥W [2] (0)∥ 2 2 -∥W v (0)∥ 2 2
). Integrating both sides of the equality, we obtain
lim t→∞ ∥ Ẇv (t)∥ 2 2 -∥ Ẇ [2] (t)∥ 2 2 = ∥ Ẇv (0)∥ 2 2 -∥ Ẇ [2] (0)∥ 2 2 -∥W [2] (0)∥ 2 2 (∥W [2] (0)∥ 2 2 -∥W v (0)∥ 2 2 ).
However, according to Definition 4 and the fact that lim t→∞ ∥ Ẇ [2] (t)∥ 2 = 0, we have that
lim t→∞ ∥ Ẇv (t)∥ 2 2 ̸ = 0.
Based on Eq. ( 16), we have
lim t→∞ Ė(t) = lim t→∞ ∥ Ẇv (t)∥ 2 2 + ∥ Ẇ [2] (t)∥ 2 2 + ∥W v (t)∥ 2 2 ∥W [2] (t)∥ 2 2 = lim t→∞ ∥ Ẇv (t)∥ 2 2 ̸ = 0.
It contradicts with Eq. ( 19) which claims lim t→∞ Ė(t) = 0. This completes the proof.
this section cite: []

Section: A.2 Theory Details for Condensation
In this section, we prove the main theorems which characterize the condensation. In retrospect of the proof of Theorem 1, Eq. ( 17) provides a lower bound that leads to the presence of explosion. However, this inequality leaves the precise growth rate of energy E undetermined. The key idea here is that Assumption 1 can give us an upper limit on how fast the energy can grow. Once we understand this growth rate, we can then move forward with proving the main theorems.
We begin our proof by the following proposition. Proposition 4 (induction). Consider dynamical system Eq. ( 7). If Assumption 1 holds at some time t 0 with t 0 < T * , then Assumption 1 will hold at t ∈ (t 0 , T * ).
Proof. First, we consider the second condition in Assumption 1. By direct calculation, we have
d dt W [2] i W [1],i , W [2] j W [1],j = W [2] j W v W [1],j    W [2] i 2 + 1 W [2] j 2 W [2] i W [2] j W [1],i ⊺ W [1],j    + W [2] i W v W [1],i    W [2] j 2 + 1 W [2] i 2 W [2] i W [2] j W [1],i ⊺ W [1],j    ,and
d dt W v,i W [1] i , W v,j W [1] j = W v,j W [1] j W [2] W 2 v,i + 1 W 2 v,j W v,i W v,j W [1] i W [1] j ⊺ + W v,i W [1] i W [2] W 2 v,j + 1 W 2 v,i W v,i W v,j W [1] i W [1] j ⊺ .
By Assumption 1, we know the above equations are larger than 0. So W
[2] i W [1],i , W [2] j W [1],j and W v,i W [1] i , W v,j W [1] j
will be monotonically increasing since t 0 .
Calculating the derivative of left hand side of first condition, we have
d dt W [2] i W v W [1],i = Ẇ 2 v,i + dm j=1 W v,i W v,j W [1],j ⊺ W [1],i + W [2] i 2 ∥W v ∥ 2 2 and d dt W v,i W [1] i W [2] = Ẇ 2 v,i + dm j=1 W v,i W v,j W [1] i W [1] j ⊺ + (W v,i ) 2 ∥W [2] ∥ 2 2 .
Hence, W
[
i W v W [1],i and W v,i W [1] i W [2] will also increase monotonically since t 0 . Therefore the condensation condition will hold until T * .
Next, we analyze the angle relation between W [2] and its derivative Ẇ [2] . For the simplicity of proof and description, we adopt a standardized notation to represent angles between distinct vectors throughout the ensuing discussion.
this section cite: []

Section: Definition 5. Let ξ ij (t) denote the angle between the vectors W v,i (t)W
[1] i (t) and W v,j (t)W
[1] [1]
i (t). Let φ i (t) denote the angle between W [2] (t) and W v,i (t)W
[1] i (t), while ζ(t) denote the angle between W [2] (t) and Ẇ [2] (t). In subsequent expressions, the variable t will be omitted unless there is a specific emphasis on the temporal change of angles.
We divide the entries of vector W v into two classes according to whether their limit is finite. Proposition 5. Suppose that Assumption 1 holds. Consider the effective dynamics Eq. ( 7). The indices [d m ] can be partitioned into two disjoint classes, denoted by C 1 = {i 1 , . . . , i k } ̸ = ∅ and C 2 = [d m ] \ C 1 . The partition satisfies the following properties:
(i) For each i ∈ [m], the limits of W v,i exist. In particular, lim t→T * W v,i = ± ∞, i ∈ C 1 , W * v,i , i ∈ C 2 . (20
) (ii) The angle ξ ij between the vectors W v,i (t)W [1] i (t) and W v,j (t)W [1] i (t)
, as defined in Definition 5, fulfills the condition:
lim t→T * cos ξ ij = 1, for i, j ∈ C 1 .(21)
(iii) The following limits exist
lim t→T * ∥ Ẇ [2] ∥ 2 ∥W v ∥ 2 2 = lim t→T * ∥ Ẇ [2] ∥ 2 ∥W [2] ∥ 2 2 = 1.(22)
Proof. 1. First, we find that for every index i the (W v,i ) 2 increases monotonically. So their limits exist. We define the index set of the parameters that tend to infinity as C 1 and the others as C 2 . Based on Theorem 1 and conservation laws, we know that C 1 ̸ = ∅. Property 1 is automatically satisfied due to our partition.
2. We introduce new variables
p = ⟨W v,i W [1] i , W v,j W [1] j ⟩, q = W 2 v,i W 2 v,j .
According to the proof of Proposition 4, we find that
dp dt = W v,j W [1] j W [2] W 2 v,i + 1 W 2 v,j W v,i W v,j W [1] i W [1] j ⊺ + W v,i W [1] i W [2] W 2 v,j + 1 W 2 v,i W v,i W v,j W [1] i W [1] j ⊺ = W v,j W [1] j W [2] W 2 v,i + W v,i W [1] i W [2] W 2 v,j 1 + p q . (23
)
Thanks to Eq. ( 7), we obtain
dq dt = 2 W v,j W [1] j W [2] W 2 v,i + W v,i W [1] i W [2] W 2 v,j . (24
)
Combining Eq. ( 23) and Eq. ( 24), we obtain
dp dq = 1 2 p q + 1 2 . (25
)
Let u = p/q. Note that dp dq = d dq (uq) = q du dq + u. Combining this with the right hand of Eq. ( 25), we get
dq q = 2du 1 -u . (26
)
The Eq. ( 26) can be solved explicitly.
ln |q(t)| -ln |q(t 0 )| = -2 ln |u(t) -1| + 2 ln |u(t 0 ) -1|.(27)
For i, j ∈ C 1 , we have lim t→T * u(t) = 1 since q tends to infinite as t tends to T * .
this section cite: []

Section: By definition,
u = p q = ⟨W v,i W [1] i , W v,j W [1] j ⟩ W 2 v,i W 2 v,j = ∥W [1] i ∥ 2 ∥W [1] j ∥ 2 |W v,i ||W v,j | cos ξ ij . (28
)
Using the conservation laws, we have
∥W [1] i ∥ 2 2 (t) -∥W [1] i ∥ 2 2 (0) = W 2 v,i (t) -W 2 v,i (0). For i ∈ C 1 , we have lim t→T * ∥W [1] i ∥ 2 2 W 2 v,i = 1.(29)
Combining Equations ( 28) and ( 29), we get
lim t→T * cos ξ ij = 1, i, j ∈ C 1 . (30
)
This finishes the proof of statement (ii).
3. Finally, we calculate the norm ∥ Ẇ [2] ∥ 2 2 . By definition, we obtain
⟨ Ẇ [2] , Ẇ [2] ⟩ = dm i=1 dm j=1 ⟨W v,i W [1] i , W v,j W [1] j ⟩.
We divide the sum into three parts due to the boundedness of entries of
W v . i∈C1 j∈C1 W 2 v,i W 2 v,j ∥W [1] i ∥ 2 |W v,i | ∥W [1] j ∥ 2 |W v,j | cos ξ ij + 2 i∈C1 j∈C2 W v,i W v,j ∥W [1] i ∥ 2 ∥W [1] j ∥ 2 cos ξ ij + i∈C2 j∈C2 W v,i W v,j ∥W [1] i ∥ 2 ∥W [1] j ∥ 2 cos ξ ij = i∈C1 j∈C1 W 2 v,i W 2 v,j + i∈C1 j∈C1 W 2 v,i W 2 v,j ∥W [1] i ∥ 2 |W v,i | ∥W [1] j ∥ 2 |W v,j | cos ξ ij -1 + 2 i∈C1 j∈C2 W v,i W v,j ∥W [1] i ∥ 2 ∥W [1] j ∥ 2 cos ξ ij + i∈C2 j∈C2 W v,i W v,j ∥W [1] i ∥ 2 ∥W [1] j ∥ 2 cos ξ ij . Since lim t→T * ∥W [1] i ∥2 |Wv,i| ∥W [1] j ∥2 |Wv,j | cos ξ ij = 1, we have lim t→T * ⟨ Ẇ [2] , Ẇ [2] ⟩ i∈C1 W 2 v,i 2 = 1.(31)
Based on statement (i), we obtain
lim t→T * i∈C1 W 2 v,i ∥W v ∥ 2 2 = 1.(32)
Combining Equations ( 31), (32) and conservation law, we have
lim t→T * ∥ Ẇ [2] ∥ 2 ∥W v ∥ 2 2 = lim t→T * ∥ Ẇ [2] ∥ 2 ∥W [2] ∥ 2 2 = 1.(33)
This finishes the proof of statement (iii).
Proposition 5 describes the angle between W v,i W [1] i and W v,j W
[1] j . Since Ẇ [2] is a linear combination of W v,i W [1] i , we immediately have following corollary. Corollary 1. Suppose that Assumption 1 holds. Consider the effective dynamics Eq. ( 7) and recall index class defined in Proposition 5. The angle ψ i between the vectors Ẇ [2] and W v,i W [1] i , as defined in Definition 5, satisfies:
lim t→T * cos ψ i = 1, i ∈ C 1 . (34
)
Proof. By definition,
cos ψ i = ⟨W v,i W [1] i , m j=1 W v,j W [1] j ⟩ ∥W v,i W [1] i ∥ 2 ∥ Ẇ [2] ∥ 2 .
Recall the definition of ξ ij , the above equation can be reformulated as follows:
cos ψ i = m j=1 |W v,i ||W v,j |∥W [1] i ∥ 2 ∥W [1] j ∥ 2 cos ξ ij ∥W v,i W [1] i ∥ 2 ∥ Ẇ [2] ∥ 2 = m j=1 |W v,j |∥W [1] j ∥ 2 cos ξ ij ∥ Ẇ [2] ∥ 2 = j∈C1 W 2 v,j ∥W [1] j ∥2 |Wv,j | cos ξ ij + j∈C2 |W v,j |∥W [1] j ∥ 2 cos ξ ij ∥ Ẇ [2] ∥ 2 .
According to Equations ( 21) and ( 22), we have lim t→T * cos ψ i = 1. This completes the proof.
So far, we have characterized some properties of W v,i W
[1] i which is component of Ẇ [2] . We have shown that some of them will have the same direction when t tends to T * . However, it is not enough for our seek for a upper bound for energy E. Luckily, based on Corollary 1, we can analyze the angle between W [2] and W v,i W [1] i which provides an upper bound. Before this, we give the following proposition. The subsequent proposition demonstrates an extension of statement 1 of Assumption 1, going beyond the condition of W v,i W [1] i W [2] being greater than zero to include additional angle-related information. Proposition 6. Suppose that Assumption 1 holds. Consider the effective dynamics Eq. ( 7) and recall index class defined in Proposition 5. There exists constants T 1 ∈ (t 0 , T * ) and Θ 1 ∈ [0, π 2 ) such that for each index i ∈ C 1 , the follow inequality holds:
cos φ i ≥ cos Θ 1 , t ∈ (T 1 , T * ).
Proof. It is sufficient to prove the statement for any fixed i ∈ C 1 due to the finiteness of |C 1 |. In Proposition 4, we have shown that ⟨W v,i W [1] i , W [2] ⟩ > 0 for t ∈ (t 0 , T * ), which implies cos φ i > 0, t ∈ (t 0 , T * ).
(35)
Hence we can focus on its square, i.e., cos 2 φ i = ⟨W [2] ,W
[1]
i ⟩ 2 ∥W [2] ∥ 2 2 ∥W [1] i ∥ 2 2
. By direct calculation, the deriva-
tion of cos 2 φ i is 2 (∥W [2] ∥ 2 2 ∥W [1] i ∥ 2 2 ) 2 ⟨W [2] , W [1] i ⟩(⟨ Ẇ [2] , W [1] i ⟩ + ⟨W [2] , ḃi ⟩)∥W [2] ∥ 2 2 ∥W [1] i ∥ 2 2 - 2 (∥W [2] ∥ 2 2 ∥W [1] i ∥ 2 2 ) 2 ⟨W [2] , W [1] i ⟩ 2 (⟨ Ẇ [2] , W [2] ⟩∥W [1] i ∥ 2 2 + ∥W [2] ∥ 2 2 ⟨ ḃi , W [1] i ⟩)
We can rewrite the numerator as
2∥W [1] i ∥ 2 2 ⟨W [2] , W [1] i ⟩ ⟨ Ẇ [2] , W [1] i ⟩∥W [2] ∥ 2 2 -⟨W [2] , W [1] i ⟩⟨ Ẇ [2] , W [2] ⟩ + 2∥W [2] ∥ 2 2 ⟨W [2] , W v,i W [1] i ⟩ ∥W [2] ∥ 2 2 ∥W [1] i ∥ 2 2 -⟨W [2] , W [1] i ⟩ 2 .
The second term of above expression is obviously greater than zero by inequality of arithmetic and geometric means. Also we can rewrite the first term as
2∥W [1] i ∥ 2 2 W 2 v,i ⟨W [2] , W v,i W [1] i ⟩ ⟨ Ẇ [2] , W v,i W [1] i ⟩∥W [2] ∥ 2 2 -⟨W [2] , W v,i W [1] i ⟩⟨ Ẇ [2] , W [2] ⟩ .
We find that the first two factors
2∥W [1] i ∥ 2 2 W 2 v,i ⟨W [2] , W v,i W [1]
i ⟩ of above expression are positive. According to Definition 5, the difference term can be reformulated as
⟨ Ẇ [2] , W v,i W [1] i ⟩∥W [2] ∥ 2 -⟨W [2] , W v,i W [1] i ⟩⟨ Ẇ [2] , W [2] ⟩ = ∥W [2] ∥ 2 2 ∥ Ẇ [2] ∥ 2 ∥W v,i W [1] i ∥ 2 (cos ψ i -cos ζ cos φ i ). Note that lim t→T * cos ψ i = 1 for i ∈ C 1 . So for every ε > 0, there exists δ > 0 such that 1 -ε ≤ cos ψ i ≤ 1, t ∈ (T * -δ, T * ).
Set ti = T * -δ and θi = arccos(1 -ε). Then we have either cos φ i ≥ cos θi , t ∈ ( ti , T * ), or there exists t ∈ ( ti , T * ) such that cos φ i ≤ cos θi , then it will increase monotonically until it goes up to θi . No matter in which case, we can find θi ∈ [0, π 2 ) such that cos φ i ≥ cos θi . Let T 1 = max i∈C1 ti and Θ 1 = max i∈C1 θi . Thus, for each i ∈ C 1 , the following inequality holds
cos φ i ≥ cos Θ 1 , t ∈ (T 1 , T * ).
This completes the proof.
In fact, Proposition 6 provides the angular relationship between W [2] and its derivative Ẇ [2] . We summarize it as follows. Proposition 7. Suppose that Assumption 1 holds. Consider the effective dynamics Eq. ( 7). There exists T 2 ∈ (T 1 , T * ) and Θ 2 ∈ [0, π 2 ) such that cos ζ ≥ cos Θ 2 , t ∈ (T 2 , T * ).
Moreover, recall the definition of energy E, the following inequality holds
E = ⟨W [2] , Ẇ [2] ⟩ ≥ cos Θ 2 ∥W [2] ∥ 2 ∥ Ẇ [2] ∥ 2 , t ∈ (T 2 , T * ).(37)
Proof. By definition of ζ, we obtain
cos ζ = ⟨ Ẇ [2] , W [2] ⟩ ∥W [2] ∥ 2 ∥ Ẇ [2] ∥ 2 = ⟨ m j=1 W v,i W [1] i , W [2] ⟩ ∥W [2] ∥ 2 ∥ Ẇ [2] ∥ 2 .
According to Proposition 6, we have
cos ζ ≥ cos Θ 1 i∈C1 ∥W v,i W [1] i ∥ 2 ∥ Ẇ [2] ∥ 2 , t ∈ (T 1 , T * ). Since we have shown that lim t→T * ∑ i∈C 1 ∥Wv,iW [1] i ∥2 ∥ Ẇ [2] ∥2
= 1 according to the proof of Proposition 5, there exists T 2 ∈ (T 1 , T * ) and Θ 2 ∈ [0, π 2 ) such that cos ζ ≥ cos Θ 2 , t ∈ (T 2 , T * ).
Recall the definition of energy E, we have
E = ⟨W [2] , Ẇ [2] ⟩ ≥ (cos Θ 2 ) ∥W [2] ∥ 2 ∥ Ẇ [2] ∥ 2 , t ∈ (T 2 , T * ).
This completes the proof. Now we prove the proposition with all preparations above. Proposition 8 (energy upper bound and blow up estimate). Suppose that Assumption 1 holds. Consider the effective dynamics Eq. ( 7). There exist T 3 ∈ (T 2 , T * ) and C ≥ 1 such that the following upper bound of Energy E holds
E(t) ≤ 1 E(s) -1 3 -C(t -s) 3 , T 3 ≤ s < t < T * . (38
)
Moreover, the blow up time T * is bounded below by
T * ≥ t + E(t) -1 3 C , t < T * . (39
)
Proof. First, by calculating the derivative of energy E, we have
Ė = ∥ Ẇ [2] ∥ 2 2 + ∥ Ẇv ∥ 2 2 + ∥W [2] ∥ 2 2 ∥W v ∥ 2 2 . We rewrite the derivative of energy E as Ė = ∥ Ẇ [2] ∥ 2 2 1 + ∥ Ẇv ∥ 2 2 ∥ Ẇ [2] ∥ 2 2 + ∥W [2] ∥ 2 2 ∥W v ∥ 2 2 ∥ Ẇ [2] ∥ 2 2 = ∥ Ẇ [2] ∥ 4 3 2 ∥W [2] ∥ 4 3 2 ∥ Ẇ [2] ∥ 2 3 2 ∥W [2] ∥ 4 3 2 1 + ∥ Ẇv ∥ 2 2 ∥ Ẇ [2] ∥ 2 2 + ∥W [2] ∥ 2 2 ∥W v ∥ 2 2 ∥ Ẇ [2] ∥ 2 2 .
According to conservation laws and Equation ( 22), there exists T 3 > T 2 such that
∥ Ẇ [2] ∥ 2 3 2 ∥W [2] ∥ 4 3 2 1 + ∥ Ẇv ∥ 2 2 ∥ Ẇ [2] ∥ 2 2 + ∥W [2] ∥ 2 2 ∥W v ∥ 2 2 ∥ Ẇ [2] ∥ 2 2 ≤ 4.
Then we have
Ė ≤ 4∥ Ẇ [2] ∥ 4 3 2 ∥W [2] ∥ 4 3
2 , ∀t > T 3 . Based on Proposition 7, we have
∥W [2] ∥ 2 ∥ Ẇ [2] ∥ 2 ≤ 1 cos Θ 2 E, ∀t > T 3 .
Thus we obtain Ė ≤ 4( 1 cos Θ 2 )
4 3 E, ∀t > T 3 .
We denote 4( 1 cos Θ2 ) 4 3 as C 0 . Then we have C 0 ≥ 4 and
Ė ≤ C 0 E 4 3 .
Opposite to proof of Theorem 1, we obtain
d dt E -1 3 ≥ - 1 3 C 0 , ∀t > T 3 .
We denote C0 3 as C. Thus C ≥ 1 and we have
E(t) ≤ 1 E(s) -1 3 -C(t -s) 3 , T 3 < s < t < T * . (40
)
Hence, for each time t < T * , the time of blow up is bounded below by
T * ≥ t + E(t) -1 3 C . (41
)
This completes the proof.
Now we begin the proof for Theorem 2.
Proof. We just prove the case for W v,i > 0. And the case for W v,i < 0 follows by similar argument. Because the derivative of W v,i is W [1] i W [2] , we only need to show that
T * T3 W [1] i W [2] dt = +∞. (42) Since lim t→T * ∥W [2] ∥ 2 2 ∥ Ẇ [2] ∥2 = 1 due to the statement (iii) of Proposition 5, there exists T 4 > T 3 such
that ∥ Ẇ [2] ∥ 2 ≤ 2 √ 2∥W [2] ∥ 2 2 , which implies E = ⟨ Ẇ [2] , W [2] ⟩ ≤ ∥ Ẇ [2] ∥ 2 ∥W [2] ∥ 2 ≤ 2 √ 2∥W [2] ∥ 3 2 . (43
)
The idea is we can find a infinite division of (T 4 , T * ) such that the integral of W i W [2] on each sub-interval is larger than positive constant which is an independent constant. Then we consider the integral t2 t1 W [1] i W [2] dt. By direct calculation of the derivative of W [1] i W [2] , we have
(W [1] i W [2] ) = W v,i W [2] ⊺ W [2] + W [1] i   j W v,j W [1] j   .
Integrating both sides of the equality, we have (W
[1] i W [2] )(t) = (W [1] i W [2] )(t 1 ) + t t1 W v,i W [2] ⊺ W [2] + W [1] i   j W v,j W [1] j   ds ≥ W v,i (T 4 ) t t1
W [2] ⊺ W [2] ds.
Note that Eq. ( 43) implies t t1 W [2] ⊺ W [2] ds ≥ 1 2
t t1 E 2 3 (s)ds ≥ 1 2 t t1 1 (E(t 1 ) -1 3 -(s -t 1 )) 2 ds = 1 2 1 E(t 1 ) -1 3 -(t -t 1 ) - 1 E(t 1 ) -1 3 .
Thus the integral satisfies
t2 t1 W [1] i W [2] dt ≥ W v,i (T 4 ) 1 2 t2 t1 1 E(t 1 ) -1 3 -(t -t 1 ) - 1 E(t 1 ) -1 3 dt = W v,i (T 4 ) 1 2 -ln(E(t 1 ) -1 3 -(t 2 -t 1 )) + ln(E(t 1 ) -1 3 ) - t 2 -t 1 E(t 1 ) -1 3 .
According to Theorem 8, we can choose t
2 -t 1 = E(t1) -1 3 2C
. Thus we obtain
t2 t1 W [1] i W [2] dt ≥ W v,i (T 4 ) 1 2 ln 1 1 -1 2C - 1 2C . (44
)
Then we introduce auxiliary function
f (t) = ln 1 1 -t -t = -ln(1 -t) -t.
We have f (0) = 0 and ḟ (t) > 0. Then we have ln 1
1-1 2C -1 2C > 0.
Since there are infinitely many such sub-intervals, the proof of the theorem is completed.
this section cite: []

Section: A.3 Theory details for Key-Query Dynamics
This appendix provides the detailed derivations for Proposition 3 and Theorem 3, assuming a linear activation function and holding to Assumption 2 (or its empirical variant, Assumption 2*). Our approach begins with a standard asymptotic analysis to decompose the loss function, as shown in Eq. ( 13).
Starting from the definition of the empirical risk, we have:
L(θ) = 1 n n i=1 e -yif θ (Xi)s = 1 n n i=1 exp   -y i   s j=1 exp Xi,sW Q W ⊺ K X ⊺ i,j √ dm s l=1 exp Xi,sW Q W ⊺ K X ⊺ i,l √ dm X i,j W V W [1] W [2]     = 1 n n i=1 exp -y i s j=1 1 s + 1 s X i,s W Q W ⊺ K X ⊺ i,j √ d m - 1 s 2 s l=1 X i,s W Q W ⊺ K X ⊺ i,l √ d m + O(δ 4 ) X i,j W V W [1] W [2] . (45
)
The third equality results from applying a Taylor expansion to the softmax function, which is justified by Assumption 2. The higher-order term, O(δ 4 ), is subsequently omitted as it does not affect the leading-order training dynamics. Given the property of the exponential loss function, ℓ(q) = e -q , the empirical loss can be decomposed as follows:
L(θ) = 1 n n i=1 exp   -y i   s j=1 1 s X i,j W V W [1] W [2]     • exp   -y i   s j=1 1 s X i,s W Q W ⊺ K X ⊺ i,j √ d m - 1 s 2 s l=1 X i,s W Q W ⊺ K X ⊺ i,l √ d m X i,j W V W [1] W [2]     = 1 n n i=1 L 1,i (θ)L 2,i (θ),(46)
where L 1,i (θ) and L 2,i (θ) correspond to the two exponential factors in the preceding expression.
this section cite: []

Section: A.3.1 Proof for Proposition 3 under Assumption 2
Proof. The proof is structured in two parts. First, we demonstrate the separation of dynamics by showing that the gradients with respect to different sets of weights have different orders of magnitude. Second, we derive the specific dynamics for the key and query matrices.
Dynamics Separation: By symmetry, we will detail the calculations for the partial derivatives with respect to W V and W Q . The derivations for the other weight matrices follow a similar procedure. We begin by computing the partial derivative of the loss L with respect to W V . Applying the product rule to the decomposed loss from Eq. ( 46), we obtain:
∂L ∂W V = ∂ ∂W V 1 n n i=1 L 1,i (θ)L 2,i (θ) = 1 n n i=1 ∂L 1,i ∂W V L 2,i + L 1,i ∂L 2,i ∂W V = 1 n n i=1 ∂L 1,i ∂W V 1 + O(δ 2 ) + L 1,i • O(δ 2 ) .(47)
The third equality holds based on Assumption 2, which implies that L 2,i = 1 + O(δ 2 ) and its derivative ∂L2,i ∂W V is also of order O(δ 2 ). Furthermore, Assumption 2 states that the leading-order term of the loss, L 1,i , is independent of the attention mechanism weights at initialization. Consequently, the term ∂L1,i ∂W V is of order O(δ 2 ). This implies that the entire gradient ∂L ∂W V is dominated by terms of order O(δ 2 ).
Next, we consider the partial derivative with respect to W Q :
∂L ∂W Q = ∂ ∂W Q 1 n n i=1 L 1,i (θ)L 2,i (θ) = 1 n n i=1 L 1,i (θ) ∂L 2,i ∂W Q ,(48)
since L 1,i is independent of W Q . Based on Assumption 2, the term ∂L2,i ∂W Q is of order O(δ), which establishes that the overall gradient ∂L ∂W Q is also of order O(δ). Key-Query Dynamics: The principle of dynamics separation, established above, shows that the gradients with respect to W Q and W K (order O(δ)) are significantly larger than those for W V , W [1] , and W [2] (order O(δ 2 )). Therefore, during the initial phase of training, the dynamics are dominated by the updates to W Q and W K . We can thus analyze their leading-order dynamics by treating the other weight matrices as effectively constant.
To derive these dynamics, we employ matrix calculus with differentials. For a scalar function f (X) of a matrix variable X, the differential is given by df = tr ∂f ∂X ⊺ dX . We apply this to the argument of the exponential in L 2,i , which we denote as A i (θ). The differential of A i with respect to W Q is:
d   -y i   s j=1 1 s X i,s W Q W ⊺ K X ⊺ i,j √ d m - 1 s 2 s l=1 X i,s W Q W ⊺ K X ⊺ i,l √ d m X i,j W V W [1] W [2]     = - y i s √ d m X i,s (dW Q )W ⊺ K   s j=1 X ⊺ i,j - 1 s s l=1 X ⊺ i,l X i,j   W V W [1] W [2] = tr   - y i s √ d m W K   s j=1 X ⊺ i,j X i,j - 1 s s l=1 X i,l   W V W [1] W [2] X ⊺ i,s (dW Q ) ⊺   .
(49) By identifying the coefficient of dW Q from the trace form, we obtain the gradient. Consequently, after neglecting higher-order terms, the leading-order dynamics for W Q under the gradient flow dW Q dt = -∂L ∂W Q are given by:
dW Q dt = 1 ns √ d m n i=1 y i L 1,i X ⊺ i,s W V W [1] W [2] ⊺   s j=1 X ⊺ i,j X i,j - 1 s s l=1 X i,l   ⊺ W K .
(50) A symmetric argument yields the corresponding dynamics for W K . This completes the proof.
this section cite: []

Section: B Experimental Details
In this section, we present more experimental details to supplement the main text.
this section cite: []

Section: B.1 Experimental Setting of Synthetic Dataset
We introduce the dataset construction method and training hyperparameters used to train the synthetic dataset.
First, we give some calculation methods for experimental pictures.
Satisfaction rate. We denote the conditions in Assumption 1 as follows:
A 1 = i ∈ [d m ] W [2] i W v W [1],i > 0, W v,i W [1] i W [2] > 0 , A 2 = (i, j) ∈ [d m ] × [d m ] ⟨W [2] i W [1],i , W [2] j W [1],j ⟩ > 0, ⟨W v,i W [1],i , W v,j W [1],j ⟩ > 0 .
Cosine similarity. To visualize the internal structure of a weight matrix W, we generate a heatmap of its reordered row-wise cosine similarity matrix. The procedure is as follows: first, the row-wise cosine similarity matrix S is computed, where each entry S ij = cos(w i , w j ) measures the similarity between row vectors w i and w j .
To reveal underlying block structures, we then employ a spectral reordering technique. This involves finding the principal eigenvector v max (the one corresponding to the largest eigenvalue) of the similarity matrix S. The sorted order of this eigenvector's components, P = argsort(v max ), provides a permutation index. By applying this permutation to both the rows and columns of S, we group highly correlated row vectors together, making low-rank patterns visually apparent in the final heatmap.
this section cite: []

Section: Dataset Construction.
We construct synthetic datasets using the concept of the anchor function Zhang et al. [2024c], which enables controlled simulation of linguistic relationships. Let the set of prompt anchors be A = {a ∈ N + | α min ≤ a ≤ α max } and the set of keys be Z = {z ∈ N + | ζ min ≤ z ≤ ζ max }, where A and Z are disjoint, i.e., A ∩ Z = ∅.
We define an anchor function F(X) : N s → N, where X = (x 1 , x 2 , . . . , x s ) is a sequence of length s. Each sequence contains exactly one anchor token a ∈ A among the first s -1 positions, and the function outputs the token immediately following the anchor, shifted by a:
F(x 1 , . . . , x s ) = x i+1 + a, where x i = a.(59)
In our experiments, we set A = {1, 2, 3, 4}, Z = {5, . . . , 100}, and s = 10. To introduce synonymy among anchors, we modify the mapping as F(x 1 , . . . , x s ) = x i+1 + (a mod 2), where
x i = a,(60)
so that anchors {1, 2} and {3, 4} produce equivalent outputs, mimicking synonymous relationships observed in natural language.
Model and training hyperparameters. Our model is a decoder-only Transformer with a single layer and a single attention head. The architecture follows the standard GPT design, consisting of a multi-head self-attention block and a position-wise feed-forward network. The Tanh activation function is used in the feed-forward network. A key aspect of our experimental setup is that both the token embedding layer and the final output projection layer are fixed and not updated during training. This allows us to isolate the learning dynamics exclusively within the Transformer's attention and feed-forward weights.
All trainable weights in the model are initialized from a normal distribution with a mean of 0. The standard deviation for different components is set based on the model dimension d model as σ = d -0.85 model . The loss is computed only on the prediction of the last token in the sequence. The model was trained for 30 epochs using the AdamW optimizer. We employed a learning rate scheduler that combines a gradual warmup phase for the first 10 epochs followed by a cosine annealing schedule. The specific hyperparameters are detailed in Table 1.
this section cite: []

Section: 
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: All the assets mentioned in paper is open-sourced and properly cited.
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
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: These type of studys are not involved in this paper. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [Yes] Justification: LLMs are not used for any core methods in this paper. Therefore, we are not required to make such a declaration. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: A.4 Proof for Proposition 3 under Assumption 2*
Proof. Dynamics Separation: We use another scheme to estimate the gradient. Consider
∂L ∂W V = ∂ ∂W V 1 n n i=1 L 1,i (θ)L 2,i (θ) = 1 n n i=1 ∂L 1,i ∂W V L 2,i + L 1,i ∂L 2,i ∂W V = 1 n n i=1 ∂L 1,i ∂W V 1 + O(δ 2 ) + L 1,i • O(δ 2 ) = 1 n n i=1 ∂L 1,i ∂W V + O(δ 2 ).(51)
Take the inner product of both sides of the equation with ∂L ∂W V , we have
∂L ∂W V , ∂L ∂W V = 1 n n i=1 ∂L 1,i ∂W V , ∂L ∂W V + O(δ 2 ).(52)
However, note that
d dt L 1 = - 1 n n i=1 ∂L 1,i ∂W V , ∂L ∂W V - 1 n n i=1 ∂L 1,i ∂W [1] , ∂L ∂W [1] - 1 n n i=1 ∂L 1,i ∂W [2]
, ∂L ∂W [2] . (53) Based on Assumption 2*, we have ∂L ∂W V is O(δ 2 ). The rest of the proof is similar to the previous one.
Key-Query Dynamics: This part is almost the same. Just need to note that the condition
d dt W V ∥W V ∥ = d dt W [1] ∥W [1] ∥ = d dt W [2]
∥W [2] ∥ ≈ 0 gives the same F after normalization.
this section cite: []

Section: A.5 Proof for Theorem 3
Proof. Based on the leading-order dynamics of key-query matrices, we prove the theorem for W Q and the technique for W K is similar. Differentiating the dynamics again, we obtain:
d 2 dt 2 W Q = F F ⊺ W Q .(54)
Let the singular value decomposition of F be F = U ΣV ⊺ , then the dynamics can be rewritten as
d 2 dt 2 W Q = U Σ 2 U ⊺ W Q . (55
) Let WQ = U ⊺ W Q U , we have d 2 dt 2 WQ = Σ 2 WQ . (56
)
The evolutions of entries are
WQ,ij (t) = C 1,ij e λit +C 2,ij e -λit . (57
)
As a result,
rank lim t→∞ W Q ∥W Q ∥ F ≤ k, (58
)
where k is the multiplicity of the largest singular value. This result finishes the proof. In this section, we show similar result for model without activation function, as a supplement to the synthetic data experiments. This shows that it is reasonable to ignore activation in our analysis
Average W t F (c) W Q & W K W V & W [1] &W [2]
this section cite: []

Section: B.3 Experimental Setting of Real Task
To validate that our theoretical insights generalize beyond simplified settings, we conducted experiments on the WikiText dataset, a standard benchmark for language modeling. This experimental setup intentionally incorporates more complex and commonly used architectural features.
Dataset and Task. We use the WikiText dataset, which consists of high-quality articles from Wikipedia. The task is next-token prediction, where the model is trained to predict the next word in a sequence. Consistent with our synthetic experiments, the training objective is calculated exclusively based on the prediction loss for the final token of each input sequence. The sequence length is set to 2048.
Model and Training Hyperparameters. We use a 2-layer decoder-only Transformer. To test the robustness of our findings, this model's architecture includes standard components that were abstracted away in the synthetic setup. Specifically, it incorporates residual connections after both the self-attention and feed-forward sub-layers, and it utilizes the GeLU activation function in the feed-forward network. This more realistic configuration allows us to demonstrate that our theory holds even in the presence of such non-linearities and standard architectural features.
All model weights are initialized from a normal distribution with a standard deviation of σ = d -1.2 model . The model was trained for 5 epochs using the AdamW optimizer with an initial learning rate of 2 × 10 -4 , which was managed by a cosine decay schedule with a warmup phase. The detailed hyperparameters for this experiment are listed in Table 2.
this section cite: []

Section: C Experiments Compute Resources
The experiments were conducted on a server with the following configuration:
this section cite: []

Section: References
Ref_id:b0 Title: What learning algorithm is in-context learning? investigations with linear models Year: (2023)
Ref_id:b1 Title: Implicit regularization in deep matrix factorization Year: (2019)
Ref_id:b2 Title: On exact computation with an infinitely wide neural net Year: (2019)
Ref_id:b3 Title: Rezero is all you need: fast convergence at large depth. In Cassio de Campos and Marloes H. Maathuis Year: (2021-07-30)
Ref_id:b4 Title: Transformers as statisticians: Provable in-context learning with in-context algorithm selection Year: (2023)
Ref_id:b5 Title: Connectivity shapes implicit regularization in matrix factorization models for matrix completion Year: (2024)
Ref_id:b6 Title: Birth of a transformer: A memory viewpoint Year: (2023)
Ref_id:b7 Title: Language models are few-shot learners Year: (2020)
Ref_id:b8 Title: Learning associative memories with gradient descent Year: (2024)
Ref_id:b9 Title: Training dynamics of multi-head softmax attention for in-context learning: Emergence, convergence, and optimality Year: (2024)
Ref_id:b10 Title: Phase diagram of initial condensation for two-layer neural networks Year: (2024)
Ref_id:b11 Title: In-context learning with transformers: Softmax attention adapts to function lipschitzness Year: (2024)
Ref_id:b12 Title: The evolution of statistical induction heads: In-context learning markov chains Year: (2024)
Ref_id:b13 Title: How do transformers learn in-context beyond simple functions? a case study on learning with representations Year: (2024)
Ref_id:b14 Title: Improving transformer optimization through better initialization Year: (2020)
Ref_id:b15 Title: In-context convergence of transformers Year: (2023)
Ref_id:b16 Title: From selfattention to markov models: Unveiling the dynamics of generative transformers Year: (2024)
Ref_id:b17 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b18 Title: Gradient descent aligns the layers of deep linear networks Year: (2019)
Ref_id:b19 Title: Early directional convergence in deep homogeneous neural networks for small initializations Year: (2024)
Ref_id:b20 Title: Algorithmic regularization in over-parameterized matrix sensing and neural networks with quadratic activations Year: (2018)
Ref_id:b21 Title: How do transformers learn topic structure: Towards a mechanistic understanding Year: (2023)
Ref_id:b22 Title: On the dynamics of training attention models Year: (2021)
Ref_id:b23 Title: Phase diagram for two-layer relu neural networks at infinite-width limit Year: (2021)
Ref_id:b24 Title: Gradient descent maximizes the margin of homogeneous neural networks Year: (2020)
Ref_id:b25 Title: One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention Year: (2024)
Ref_id:b26 Title: A mean field view of the landscape of twolayer neural networks Year: (2018)
Ref_id:b27 Title: Pointer sentinel mixture models Year: (2017)
Ref_id:b28 Title: On the explicit role of initialization on the convergence and implicit bias of overparametrized linear networks Year: (2021-07)
Ref_id:b29 Title: How transformers learn causal structure with gradient descent Year: (2024)
Ref_id:b30 Title: -context learning and induction heads Year: (2022)
Ref_id:b31 Title: The mechanistic basis of data dependence and abrupt learning in an in-context classification task Year: (2024)
Ref_id:b32 Title: Parameters as interacting particles: long time convergence and asymptotic error scaling of neural networks Year: (2018)
Ref_id:b33 Title: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks Year: (2013)
Ref_id:b34 Title: Approximating how single head attention learns Year: (2021)
Ref_id:b35 Title: Implicit balancing and regularization: Generalization and convergence guarantees for overparameterized asymmetric matrix sensing Year: (2023)
Ref_id:b36 Title: Small random initialization is akin to spectral learning: Optimization and generalization guarantees for overparameterized low-rank matrix reconstruction Year: (2021)
Ref_id:b37 Title: Scan and snap: Understanding training dynamics and token composition in 1-layer transformer Year: (2023)
Ref_id:b38 Title: JoMA: Demystifying multilayer transformers via joint dynamics of MLP and attention Year: (2024)
Ref_id:b39 Title: On the spectral bias of two-layer linear networks Year: (2023)
Ref_id:b40 Title: Attention is all you need Year: (2017)
Ref_id:b41 Title: Gradient dynamics of shallow univariate relu networks Year: (2019)
Ref_id:b42 Title: An overview of condensation phenomenon in deep learning Year: (2025)
Ref_id:b43 Title: An analysis for reasoning bias of language models with small initialization Year: (2025)
Ref_id:b44 Title:  Year: (2025)
Ref_id:b45 Title: Improving deep transformer with depth-scaled initialization and merged attention Year: (2019-11)
Ref_id:b46 Title: Trained transformers learn linear models in-context Year: (2024)
Ref_id:b47 Title: A type of generalization error induced by initialization in deep neural networks Year: (2020)
Ref_id:b48 Title: Training dynamics of in-context learning in linear attention Year: (2025)
Ref_id:b49 Title: Initialization is critical to whether transformers fit composite functions by inference or memorizing Year: (2024)
Ref_id:b50 Title: Anchor function: a type of benchmark functions for studying language models Year: (2024)
Ref_id:b51 Title: Complexity control facilitates reasoning-based compositional generalization in transformers Year: (2025)
Ref_id:b52 Title: Towards understanding the condensation of neural networks at initial training Year: (2022)
Ref_id:b53 Title: Understanding the initial condensation of convolutional neural networks Year: (2023)
Ref_id:b54 Title: Gradinit: Learning to initialize neural networks for stable and efficient training Year: (2021)
