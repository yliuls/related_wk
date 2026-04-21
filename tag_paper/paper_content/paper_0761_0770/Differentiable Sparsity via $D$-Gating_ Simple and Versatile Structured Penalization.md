Title: Differentiable Sparsity via D-Gating: Simple and Versatile Structured Penalization
Abstract: Structured sparsity regularization offers a principled way to compact neural networks, but its non-differentiability breaks compatibility with conventional stochastic gradient descent and requires either specialized optimizers or additional post-hoc pruning without formal guarantees. In this work, we propose D-Gating, a fully differentiable structured overparameterization that splits each group of weights into a primary weight vector and multiple scalar gating factors. We prove that any local minimum under D-Gating is also a local minimum using non-smooth structured L 2,2/D penalization, and further show that the D-Gating objective converges at least exponentially fast to the L 2,2/D -regularized loss in the gradient flow limit. Together, our results show that D-Gating is theoretically equivalent to solving the original group sparsity problem, yet induces distinct learning dynamics that evolve from a non-sparse regime into sparse optimization. We validate our theory across vision, language, and tabular tasks, where D-Gating consistently delivers strong performance-sparsity tradeoffs and outperforms both direct optimization of structured penalties and conventional pruning baselines.

Section: Introduction
Sparsity in deep learning models has received considerable attention in recent years. On the one hand, unstructured sparsity methods remove individual weights to reduce parameter counts and achieve high compression ratios, but they produce irregular connectivity patterns that are hard to accelerate on standard hardware. On the other hand, structured sparsity targets entire groups of parameters, such as neurons [49,62], convolutional filters [38,40], or attention heads [12,42,53,60,71], yielding coarser sparsity patterns that translate directly into reductions in floating-point operations and memory requirements on existing hardware, which results in more efficient deployment of large models [6,14]. Beyond computational advantages, introducing sparsity can also improve generalization performance [10] and increase model interpretability [20]. Nevertheless, most popular sparsification techniques in deep learning are not based on the non-smooth L 1 and L 2,1 penalties widely used in classical statistics and machine learning [55,56], but rather constitute iterative pruning and retraining pipelines [3,20,34], whose main sparsification mechanism is defined by heuristic pruning criteria like parameter magnitudes. In these methods, the pruning step is decoupled from training, making it difficult to characterize precisely what overall objective is optimized and to provide principled guarantees. Further, the decision space is vast-pruning at initialization [10,11,18,36,54,61], after training [38,40,41,69], or sparsification during training [33,46,48], each with their own subtleties and tradeoffs [6,14,20]-making it cumbersome for practitioners to select a method that balances efficiency, accuracy, and theoretical soundness.
this section cite: ['b48', 'b61', 'b37', 'b39', 'b11', 'b41', 'b52', 'b59', 'b70', 'b5', 'b13', 'b9', 'b19', 'b54', 'b55', 'b2', 'b19', 'b33', 'b9', 'b10', 'b17', 'b35', 'b53', 'b60', 'b37', 'b39', 'b40', 'b68', 'b32', 'b45', 'b47', 'b5', 'b13', 'b19']

Section: Sparsity penalties
Structured sparsity penalties such as the L 2,1 norm are, in theory, capable of eliminating unimportant parameter groups, but in deep learning, they have mostly served as heuristics to steer post-hoc pruning rather than achieve exact sparsity [18,40,62]. Directly enforcing nondifferentiable structured sparsity regularization requires solvers that can cope with its non-smooth nature; if this non-differentiability is ignored, optimization may oscillate or converge to dense, suboptimal solutions, as shown in Fig. 1. Replacing standard stochastic gradient descent (SGD) with specialized procedures such as proximal-type algorithms (e.g. [8,23,43]) introduces substantial complexity, demands non-standard hyperparameter configurations, and often the routines are adapted to specific use cases or model classes, thereby foregoing modularity. This renders such approaches cumbersome to implement and inhibits their adoption for large-scale deep learning.
With these obstacles in mind, we ask and positively answer the following main question:
Research question: Can we design a modular structured sparsity regularization method integrable into any architecture, amenable to SGD, with theoretical guarantees and little practical overhead?
this section cite: ['b17', 'b39', 'b61', 'b7', 'b22', 'b42']

Section: Related literature
Structured sparsity A range of methods has been proposed to induce block-wise zeros in neural networks, yet they often rely on either post-hoc pruning or non-standard optimizers. Early work applies the convex L 2,1 penalty directly to weight groups but optimizes it with vanilla (sub-)gradient descent [49,62], which fails to find sparse solutions and must be followed by an explicit pruning step [8]. To remedy the non-differentiability at zero, [8] proposes the use of a proximal algorithm, which we aim to avoid in favor of compatibility with SGD. [5] further generalizes the L 2,1 regularizer to non-convex L 2,q , q < 1, penalties using a custom optimizer. Rather than penalizing weight structures directly, [4,23,40] introduce shared scaling factors for each group and impose sparsity on those factors instead of the whole parameter group, but still require careful tuning. Although these competitors can yield exact sparsity under certain settings, they either fall back on pruning or abandon standard SGD, motivating our search for a fully differentiable, SGD-compatible alternative.
Differentiable sparsity A possible solution to incorporate sparsity-inducing penalties while retaining differentiability are approaches that split the parameters into multiple components and impose smooth L 2 regularization on the factors, which can be shown to induce the desired non-smooth sparsity penalty on the reconstructed parameters. This idea dates back to [15,21] and has recently been adopted to incorporate differentiable L 1 -type sparsity regularization in neural networks [25,30,28,29,57,72,73]. In the case of matrix (instead of parameter) factorization using two factors, a low-rank bias, given by the trace norm, is induced on the product [27,52]. The implicit bias literature also investigates such parameter decompositions without L 2 regularization, which can also induce sparsity under certain conditions-such as impractically small initialization scales [16,58,63,70]. An extension to implicit group sparsity for linear models is presented by [39]. However, existing proposals either focus on unstructured sparsity, are constrained to only two factors, or do not constitute modular approaches applicable to arbitrary architectures. This leaves a gap in the current literature on whether extensions to arbitrary structures are possible, how such an approach can be implemented in practice, and to what extent there are theoretical guarantees to back this method. For simplicity, we show D-Gating visually for a single fully-connected layer with input-wise grouping (colors), but our approach extends to arbitrary network components such as convolutional filters or attention heads. We proceed by applying D-Gating (red nodes and their connections) to the neural network weight and running SGD on the gating parameters with weight decay. After training, the weights are collapsed again and the zero structures removed, with the resulting sparse minimizers also being minimizers of the non-smooth L 2,2/D -regularized objective.
this section cite: ['b48', 'b61', 'b7', 'b7', 'b4', 'b3', 'b22', 'b39', 'b14', 'b20', 'b24', 'b29', 'b27', 'b28', 'b56', 'b71', 'b72', 'b26', 'b51', 'b15', 'b57', 'b62', 'b69', 'b38']

Section: Our contributions
Inspired by prior work on differentiable sparse regularization, we propose a new approach called D-Gating, which constitutes a structured sparsity-inducing penalty approach. It can be modularly incorporated in "arbitrary" architectures and neither incurs a notable overhead in additional parameters nor requires additional pruning steps, and is compatible with off-the-shelf SGD optimization. We further establish novel theoretical results that show the equivalence of our proposed differentiable regularization method and non-differentiable sparsity-inducing penalties (cf. Fig. 2), akin to what has been shown for approaches with unstructured sparsity penalties. Apart from theoretically and practically studying the loss landscape and training dynamics of our approach, we also validate our theory on an array of experiments to showcase its versatility in diverse deep learning applications.
this section cite: []

Section: Problem statement
In this paper, we propose a general structured sparsity approach for neural networks f (•, w) that allows penalizing excessive network components without placing restrictions on the type of architecture f or the position of the unit within the weight vector w ∈ R p that is targeted with the regularization. Structures such as filters naturally arise in neural networks, yielding a partition G = {G 1 , . . . , G J } of a subset of the indices [p] := {1, . . . , p} of w into J groups w Gj with elements w j,g , g ∈ G j . For filter sparsity in convolutional neural networks, G would be all indices for weights in the convolutional layers, and each G j the indices of weights of one of the J filters.
Given this structure, we seek to optimize a general optimization problem minimize
w∈R p L w (w) := L 0 (w) + λ∥w∥ 2/D 2,2/D(1)
for D ≥ 2, where the unregularized objective 1) constitutes a generalization of what is often referred to as the group lasso [67], or L 2,1 penalty, which is recovered for D = 2. For D > 2, we obtain the more general and non-convex group penalty: ∥w∥ [22]. As this penalty is neither differentiable for D = 2 nor D > 2, SGD optimization of (1) yields unfavorable optimization dynamics and does not achieve exact sparsity (cf. Fig. 1). We therefore either require specialized optimization routines or a surrogate objective which induces the solution to (1). We choose the latter to minimize the overhead (cf. App. E.2) and changes to established training procedures.
L 0 = n i=1 ℓ(y i , f x i , w) is the sum of n observed loss contributions with loss ℓ : Y × Y → R + 0 evaluated on independent data points {(x i , y i )} n i=1 ∈ (X × Y) n . The regularization term in Eq. (
2/D 2,2/D := J j=1 ( g∈Gj |w j,g | 2 ) 1/D
this section cite: ['b66', 'b21', 'b0']

Section: Differentiable structured sparsity via D-Gating
To solve Eq. ( 1) while enabling practitioners to use standard SGD optimizers, we derive a fully differentiable method that implicitly tackles Eq. ( 1) by employing an overparameterized model and a smooth surrogate penalty compatible with SGD optimization.
this section cite: []

Section: Model structure and gating variables
As we are interested in a general method for structured sparsity where arbitrary subsets of network weights can be sparsified, we assume a parametric learning model
f : X × R q+p → R c , (x, w) → f (x; w),(2)
with inputs x ∈ X and parameters w = (v, w), where w ∈ R p contains the penalized parameters of interest and v ∈ R q the remaining parameters. Further, w is a partitioned (structured) weight object comprising J groups, w = (w j ) J j=1 ∈ R p1+...+p J = R p . To convert the non-smooth optimization problem into a smooth optimization problem, we subdivide w into two parts using the following gating operation: Definition 1 (D-Gating). Let w ∈ R p and G = {G 1 , . . . , G J } be a partition of the indices [p] of w into J groups. Further let ω ∈ R p be the primary weight of the same size as w, γ d ∈ R J one of D -1 vectors containing group-wise gating factors, and let γ ⊙ := γ 1 ⊙ . . . ⊙ γ D-1 denote the element-wise product of the gating factors with entries γ ⊙ j = D-1 d=1 γ j,d for j ∈ [J]. For brevity, we collect the scaling factors in the matrix
Γ = γ 1 , . . . , γ D-1 ∈ R J×(D-1) . The D-Gating operation ▶ : R p × R J×(D-1) → R p , (ω, Γ) → ω ▶ γ ⊙ , decomposes w as w = ω ▶ γ ⊙ := ω j D-1 d=1 γ j,d J j=1 = ω j • γ ⊙ j J j=1 ,(3)
and we call w D-Gated if it is parametrized as ω ▶ γ ⊙ .
Intuitively, the D-Gating operation splits each group weight w j into D factors: the vector ω j , corresponding to the original group weights, and the D -1 additional gating factors γ j,d ∈ R, which are applied multiplicatively to all entries of ω j .
this section cite: []

Section: Differentiable penalty
Given the gated formulation, we can now impose surrogate L 2 regularization on (ω, Γ), defined as
L(v, ω, Γ) = L 0 (v, ω ▶ γ ⊙ ) + λ R(ω, Γ) (4
) = n i=1 ℓ y i , f x i , (v, ω ▶ γ ⊙ ) + λ D J j=1 ∥ω j ∥ 2 2 + D-1 d=1 γ 2 j,d ∥ω∥ 2 2 +∥Γ∥ 2 F (5
)
where L 0 (v, w) denotes the unregularized, differentiable loss function with per-sample loss ℓ. In the following, we will denote local minimizers of the D-Gating objective as
(v, ω, Γ) ∈ arg loc min (v,ω,Γ)∈R q+p+J(D-1) L(v, ω, Γ).(6)
While the function values of Eq. ( 4) are not necessarily equal to those of Eq. ( 1), our next section provides a theoretical guarantee of the equivalence of both objectives with regards to their minima and shows that the difference between both objectives is vanishing at least exponentially fast.
this section cite: []

Section: Theoretical results
Because the presence of ungated parameters v is inconsequential for our analyses and all results directly carry over, they will be omitted from now on, and we assume for simplicity of exposition that all model parameters are D-Gated.
this section cite: []

Section: Stationarity condition and loss simplification
The following result establishes that all stationary points of L(ω, Γ) correspond to balanced D-Gating parameters. Otherwise, one could continuously perturb the D-Gating parameters toward a more balanced configuration without altering the effective parameter w = ω ▶ γ ⊙ while strictly decreasing the L 2 penalty R(ω, Γ).
Lemma 1 (Balancedness at stationary points). Let (ω, Γ) be D-Gating parameters satisfying w j = ω j D-1 d=1 γ j,d for j ∈ [J]. If (ω, Γ) is a stationary point of the L 2 -regularized objective L(ω, Γ) with λ > 0, then the gating factors are group-wise balanced in the sense that
∥ω j ∥ 2 2 = γ 2 j,1 = • • • = γ 2 j,D-1 = ∥w j ∥ 2/D 2 ∀ j ∈ [J].(7)
Notably, the loss evaluated at balanced parameters simplifies to reveal its sparsity-inducing nature: Corollary 1 (Loss simplification at balanced gating parameters). Let (ω, Γ) be balanced D-Gating parameters satisfying w j = ω j D-1 d=1 γ j,d and Eq. (7) for j ∈ [J]. The D-Gated objective L(ω, Γ) in Eq. ( 5) then simplifies to
L0(ω ▶ γ ⊙ ) + λ D (∥ω∥ 2 2 + ∥Γ∥ 2 F ) = L0(w) + λ J j=1 ∥wj∥ 2/D 2 = L0(w) + λ ∥w∥ 2/D 2,2/D :=Rw(w) =: Lw(w) (8
)
this section cite: []

Section: Equivalence of optimization problems
The previous result is reassuring as it demonstrates the equivalence of objectives at balanced gating parameters. It does, however, not guarantee that optimizing one objective provides a meaningful solution for the other objective. The following result establishes equivalence at the solution level. Theorem 1 (Equivalence of D-Gating and L 2,2/D regularization). The two optimization problems
minimize ω∈R p ,Γ∈R J×(D-1) L(ω, Γ) := L 0 (ω ▶ γ ⊙ ) + λ D (∥ω∥ 2 2 + ∥Γ∥ 2 F ) (9
) minimize w∈R p L w (w) := L 0 (w) + λ∥w∥ 2/D 2,2/D (10
)
are equivalent in the sense that their local minima are identical. If ( ω, Γ) ∈ arg loc min L(ω, Γ), then ω ▶ γ⊙ = ŵ ∈ arg loc min L w (w), and likewise, if ŵ ∈ arg loc min L w (w), then any balanced gating representation ( ω, Γ) such that ŵ = ω ▶ γ⊙ is a local minimizer of L(ω, Γ).
Specifically, there is a bijective mapping between the local minimizers of L w (w) and the equivalence class of local minimizers of L(ω, Γ), resulting in the same effective parameter w.
this section cite: []

Section: Optimization dynamics
Under ubiquitous (S)GD-based optimization of the D-Gated objective in Eq. ( 5), as well as its theoretically simpler continuous-time gradient flow (GF) limit with infinitesimal learning rate η, we can additionally establish results characterizing the evolution of parameter balancedness, i.e., quantify how fast the D-Gated objective converges to the original L 2,2/D regularized loss.
this section cite: []

Section: Evolution of imbalance and loss convergence for D-Gated models under GF dynamics
The group-wise continuous-time gradient flow dynamics for j ∈ [J] are given by ωj = -∇ ωj L, γj,d = -∂ γ j,d L.
(11) The gradients with respect to the D-Gating parameters ω j and the γ j,d are, using the chain rule,
∇ω j L = γ ⊙ j ∇w j L0 + 2λ D ωj, ∂γ j,d L = γ ⊙ j γ j,d ω ⊤ j ∇w j L0 + 2λ D γ j,d , d ∈ [D -1] .(12)
We define the pair-wise imbalance I between any two group-wise factors d ̸ = d ′ ∈ [D] and show it vanishes exponentially in time:
I j,d,d ′ (t) := ∥ω j (t)∥ 2 2 -γ j,d ′ (t) 2 , if d = 1, γ j,d (t) 2 -γ j,d ′ (t) 2 , if d ̸ = 1.(13)
Lemma 2 (Exponential decay of imbalance under continuous-time GF). Under the gradient flow dynamics Eqs. (11) and (12), the pair-wise imbalance I j,d,d ′ (t) (13) between two gating parameters d, d ′ of group j satisfies
d dt I j,d,d ′ (t) = - 4λ D I j,d,d ′ (t) ∀d ̸ = d ′ , j ∈ [J].(14)
Solving the ODE shows I j,d,d ′ (t) decays exponentially for λ ≥ 0: I j,d,d ′ (t) = I j,d,d ′ (0)e -4λ D t . This result further shows that for λ = 0, I j,d,d ′ (t) is a conserved quantity [32,74], i.e., imbalances decay with a 0 rate. The difference of losses is determined by the difference of regularizers, termed misalignment M(ω, Γ), and thus depends on the overall degree of balancedness.
L(ω, Γ) -L w (ω ▶ γ ⊙ ) = λM(ω, Γ) := λ R(ω, Γ) -R w (ω ▶ γ ⊙ ) (15
) = λ D -1 (∥ω∥ 2 2 + ∥Γ∥ 2 F ) -∥ω ▶ γ ⊙ ∥ 2/D 2,2/D ≥ 0 .(16)
Using this, the previous result can be extended to show |L(ω, Γ) -L w (ω ▶ γ ⊙ )| → 0: Lemma 3 (Convergence of D-gated loss to L 2,2/D regularized loss under GF). Assume that the model parameters of Eq. (4) with effective weight w(t) = ω(t) ▶ γ ⊙ (t) as in Eq. (3) evolve with time t according to the gradient flow in Eqs. (11) and (12). Then, the D-Gated loss L(ω(t), Γ(t)) in Eq. (9) converges to the non-smooth L 2,2/D regularized loss L w (w(t)) in Eq. (10) at least exponentially fast given an initialization-dependent constant C ≥ 0:
L(ω(t), Γ(t)) -L w (w(t)) ≤ Ce -4λ D t ,(17)
Intuitively, this is because balancedness in D-Gating is precisely the condition required for L(ω, Γ) to simplify to L w (w) (cf. Corollary 1). Hence, as the pair-wise imbalances vanish, the balancedness condition becomes increasingly true, and the two losses converge.
this section cite: ['b31', 'b73']

Section: Evolution of imbalance for D-Gated models under (S)GD dynamics
For an analysis of the discrete-time evolution of imbalances, the dynamics becomes more convoluted, but we can establish geometric decay up to first-order in η, and find symmetry-induced absorbing SGD states [72] for balanced gating configurations.
this section cite: ['b71']

Section: Lemma 4 (Imbalance evolution under discrete-time GD).
Consider the D-Gated objective Eq. ( 5). Then, under (S)GD, for any j ∈ [J], (i) the pair-wise imbalances in Eq. ( 13) evolve as
I (t+1) j,d,d ′ = 1 -4λη/D I (t) j,d,d ′ + η 2 ∆ (t) j,d,d ′ , η > 0, d, d ′ ∈ [D], d ̸ = d ′ ,(18)
with separate second-order terms ∆ j,d,d ′ for d = 1 and d, d ′ > 1. For sufficiently small η or near stationarity, the imbalance
I (t+1) j,d,d ′ ≈ (1 -4λη/D) • I (t)
j,d,d ′ exhibits discrete exponential decay. (ii) Balancedness is conserved between any two scalar factors d, d ′ > 1, i.e.,
I (t) j,d,d ′ = 0 ⇒ I (t ′ ) j,d,d ′ = 0 ∀ t ′ > t, and (iii), for balanced zero representations (ω (t) j , {γ (t) j,d } D-1 d=1 ) = 0, it holds (ω (t ′ ) j , {γ (t ′ ) j,d } D-1 d=1 ) = 0 ∀ t ′ > t.
this section cite: []

Section: Numerical experiments
In the following, we empirically investigate the learning dynamics of our approach in Section 5.1 and then showcase various applications in Section 5.2 to demonstrate our method's modularity.
this section cite: []

Section: Learning dynamics and misalignment 5.1.1 Exponential decay of imbalance
We first validate our theoretical results on learning dynamics and loss convergence of D-Gating from Section 4.3. For this, we apply D-Gating with D ∈ {2, 3, 4} to a LeNet-300-100 at the neuron level and train the model on the MNIST dataset using SGD. We use a grid of λ values and measure the loss convergence as defined in Eq. ( 17). Fig. 3 visualizes the results and confirms our theoretical findings on the exponential decay of the loss difference. Appendix E.4 contains further results, e.g., for Adam.
this section cite: []

Section: D-Gating and misalignment for group lasso
To further validate the equivalence of optimization problems as established in the previous section, we run a sparse linear regression where direct L 2,1 -regularized optimization is more accessible due to the availability of specialized optimization routines. For this, we simulate data as described in Appendix C.1 with 40 feature groups of 5 features each, of which 7 are informative (with truly non-zero effects). We use accelerated proximal gradient descent [50] to directly optimize the original L 2,1 -penalized linear model and apply our approach for D ∈ {2, 3, 4} over the same grid of λ values. In addition, we also perform direct GD optimization of the L 2,1 -regularized linear model and compare all methods against an oracle (a linear model using only the signal variables). Results in Fig. 4 confirm the established equivalence between the original and D-Gated objective for D = 2, but also demonstrate the improvement in the performance-sparsity tradeoff for D > 2.
this section cite: ['b49']

Section: Modularity
Next, we demonstrate the flexibility of our method. To this end, we study various types of structured sparsity problems that arise in neural networks. In these experiments, we focus on demonstrating the broad applicability of our method rather than an exhaustive benchmark comparison. Our method supports any form of structured sparsity in neural networks, enabling diverse applications. Selected use cases are shown below. Further results are presented in Appendix E.
this section cite: []

Section: Feature selection in non-linear models
We start by investigating input feature selection, i.e., by individually gating the first-layer weights outgoing from each input feature. We follow the setup of [37] by running the proposed method, LassoNet, as well as HSIC [66] on six diverse datasets as done in [37]. We use the same LeNet-300-100 architecture as a backbone for LassoNet, HSIC and our D-Gating approach (cf. Appendix C.2).
Fig. 5 depicts the comparison results, showing that 2-Gating is often inferior to LassoNet and HSIC. 3-and 4-Gating, however, dominate all other methods for almost all possible input sparsity configurations and, hence, are the favorable options among these methods for feature selection.
this section cite: ['b36', 'b65', 'b36']

Section: Filter sparsity in convolutional neural networks
In the next experiment, we investigate filter sparsification -one of the most prominent applications of structured sparsity in neural networks. As comparison methods, we select three commonly used methods in the filter sparsity literature [20]: Global magnitude pruning, L 2,1 -penalization with naïve optimization followed by magnitude pruning (MP) [62], and network slimming [40]. A more detailed description can be found in Appendix C.3. We run experiments on CIFAR-10, CIFAR-100, and SVHN, using a VGG-16 [51] and ResNet-18 model [19]. D-Gating is implemented by adding gating parameters on the filter level, which, given the size of these models, has a negligible parameter overhead (see Table 5). As filter sparsity can be used to construct a smaller model, potentially deployable on edge devices or similar, we also measure the theoretical speed-up of the sparsified model using floating-point operations (FLOPs). Similar to previous results, Fig. 6 unveils a superior performance of D > 2-Gating compared to gating with D = 2. However, all gating approaches outperform the filter sparsity baselines despite our approach not requiring post-hoc pruning as the main sparsification mechanism. This is the case both in terms of the accuracy-sparsity tradeoff provided by our method as well as the theoretical speed-up implied (cf. Table 1 and Fig. 11).
this section cite: ['b19', 'b61', 'b39', 'b50', 'b18']

Section: Structured sparsity in language modeling
Our next application considers the effect of D-Gating in an attention-based language model [59]. For this, we use NanoGPT and apply D-Gating to the attention heads of all attention layers. A natural comparison is again the direct optimization of the L 2,1 -penalty, i.e., without first transforming the objective into a differentiable one through D-Gating. To highlight the shortcomings of this naïve approach, we also follow the direct optimization with an explicit pruning step. We train NanoGPT on TinyShakespeare (details in Appendix C.4) and evaluate the model's validation accuracy as well as validation perplexity for different regularization strengths and hence levels of attention head sparsity.
Fig. 7 confirms our hypothesis that direct optimization does not result in structured sparsity. Notably, the min. and max. norms of the attention heads converge for large λ under direct L 2,1 penalization. In contrast, D-Gating shows the desired effect of increased sparsity for higher regularization and provides a smooth tradeoff between accuracy and sparsity. Even when combining direct optimization of the L 2,1 regularized objective with additional post-hoc pruning, and taking, at each pruning ratio, the best performance over a grid of λ values, we see that D-Gating achieves much higher head sparsity values before performance degrades significantly.  As a final application, we investigate the sparsification of neural trees. More specifically, we propose a novel modification of Neural Oblivious Decision Ensembles (NODE) [47], a neural network-based decision tree ensemble model. While there are multiple options to apply our approach within this architecture, we demonstrate the efficacy of D-Gating by inducing sparsity on the tree-level, i.e., using the different trees as groups. We test our approach on the Wine data set [7] using a range of λ values for a single tree-layer as suggested by [47]. The tree layer consists of 500 trees, and the weights corresponding to each tree are gated with D = 2 to induce differentiable L 2,1 group sparsity. This is already lower than the default hyperparameter for the tree count of 2048 reported in [47], but raises the question whether 500 trees are in fact necessary to obtain reported performances. Fig. 8 shows the test root mean squared error (RMSE) and the number of active trees as functions of λ. While the full non-sparse model reaches baseline RMSE values, we observe that it is possible to achieve a very similar performance by using less than 5 trees, e.g., for λ = 3 × 10 -3 . This suggests that, at least for the Wine data set, a much simpler and notably less expensive configuration is sufficient.
this section cite: ['b58', 'b46', 'b6', 'b46', 'b46']

Section: Tree sparsity in neural trees

this section cite: []

Section: Further experiments and ablation studies
Additional experimental results are provided in Appendix E. Appendices E.5 and E.6 demonstrate the effectiveness of D-Gating beyond the model classes studied above. First, D-Gating is implemented for multi-modal subnetwork selection in late-fusion architectures, where it consistently succeeds in removing irrelevant data modalities and retaining only informative information.
Next, Appendix E.6 studies a variant of Neural Additive Models (NAMs) [1] with differentiable shape function sparsity, termed D-SNAMs. NAMs combine the inherent interpretability of additive models with the expressivity of neural networks by processing each input independently through its own shape function subnetwork before summing the outputs. Here, D-Gating is applied to the first-layer weights of each feature-specific subnetwork to enforce shape function sparsity, effectively removing uninformative inputs while maintaining the flexibility to model non-linear effects of informative features. Our differentiable approach outperforms competing methods in terms of predictive performance, such as sparse NAMs (SNAMs) [65], which are based on non-differentiable (group lasso) penalties, and does not require post-hoc pruning. In particular, we observe for D > 2, i.e., non-convex induced regularization, that D-Gating produces increasingly sparse solutions while maintaining low generalization error, explainable by the the more aggressive sparsification capabilities of non-convex over convex sparsity penalties [22]. These experiments substantiate D-Gating as a promising approach for subnetwork sparsification, which is amenable to differentiable optimization, whether in multi-modal settings or for attaining shape function sparsity in neural additive modeling. Finally, Appendix E.2 includes further information and experiments on the negligible parameter, runtime, and memory overheads incurred by overparameterization using D-Gating, while Appendix E.3 contains ablation studies on performance and numerical stability with respect to the gating depth D, supporting the recommendation that D ∈ {3, 4} typically yields the best tradeoff.
this section cite: ['b0', 'b64', 'b21']

Section: Discussion
In this paper, we introduce D-Gating, a differentiable structured sparsity method compatible with SGD and applicable to arbitrary differentiable architectures, addressing limitations of non-differentiable penalties in deep learning. We thereby positively answer our initial research question on whether it is possible to design a modular structured sparsity routine, integrable into any architecture, amenable to SGD, with theoretical sparsity guarantees and little practical overhead.
Limitations and future work Due to the flexibility of D-Gating, our approach can provide structured sparsity penalties for arbitrary grouping structures. We systematically demonstrate this flexibility through a diverse set of applications in Section 5.2 and Appendix E. While our theoretical results guarantee equivalence to the original sparse but non-smooth optimization problem, future work could further explore the benefits of this formulation or assess its performance when combined with sophisticated pruning and retraining schedules. Secondly, although the gradient flow limit admits clear analysis, it remains an open question how discrete-time SGD with large learning rates or scheduling impacts the learning dynamics. Finally, integrating D-Gating into the complex training pipelines of modern large-scale foundation models, where sparse training from scratch is often impractical, presents another promising direction.
this section cite: []

Section: References
Ref_id:b0 Title: Neural additive models: Interpretable machine learning with neural nets Year: (2021)
Ref_id:b1 Title: A public domain dataset for human activity recognition using smartphones Year: (2013)
Ref_id:b2 Title: What is the state of neural network pruning? Proceedings of machine learning and systems Year: (2020)
Ref_id:b3 Title: Improving network slimming with nonconvex regularization Year: (2021)
Ref_id:b4 Title: Structured sparsity of convolutional neural networks via nonconvex sparse group regularization Year: (2021)
Ref_id:b5 Title: A survey on deep neural network pruning: Taxonomy, comparison, analysis, and recommendations Year: (2024)
Ref_id:b6 Title: Wine Quality. UCI Machine Learning Repository Year: (2009)
Ref_id:b7 Title: Structured sparsity inducing adaptive optimizers for deep learning Year: (2021)
Ref_id:b8 Title: Spoken letter recognition Year: (1990)
Ref_id:b9 Title: The lottery ticket hypothesis: Finding sparse, trainable neural networks Year: (2019)
Ref_id:b10 Title: Pruning neural networks at initialization: Why are we missing the mark? Year: (2020)
Ref_id:b11 Title: Sparsegpt: Massive language models can be accurately pruned in one-shot Year: (2023)
Ref_id:b12 Title: Early vs late fusion in multimodal convolutional neural networks Year: (2020)
Ref_id:b13 Title: The state of sparsity in deep neural networks Year: (2019)
Ref_id:b14 Title: Least absolute shrinkage is equivalent to quadratic penalization Year: (1998)
Ref_id:b15 Title: Implicit bias of gradient descent on linear convolutional networks Year: (2018)
Ref_id:b16 Title: Design of experiments of the nips 2003 variable selection benchmark Year: (2003)
Ref_id:b17 Title: Learning both weights and connections for efficient neural network Year: (2015)
Ref_id:b18 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b19 Title: Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks Year: (2021)
Ref_id:b20 Title: Lasso, fractional norm and structured sparse estimation using a hadamard product parametrization Year: (2017)
Ref_id:b21 Title: Group sparse optimization via ℓ p,q regularization Year: (2017)
Ref_id:b22 Title: Data-driven sparse structure selection for deep neural networks Year: (2018)
Ref_id:b23 Title: Batch normalization: Accelerating deep network training by reducing internal covariate shift Year: (2015)
Ref_id:b24 Title: Mask in the mirror: Implicit sparsification Year: (2025)
Ref_id:b25 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b26 Title: Weight decay induces low-rank attention layers Year: (2024)
Ref_id:b27 Title: Differentiable attention sparsity via structured d-gating Year: (2025)
Ref_id:b28 Title: Deep weight factorization: Sparse learning through the lens of artificial symmetries Year: (2025)
Ref_id:b29 Title: Smoothing the edges: a general framework for smooth optimization in sparse regularization using hadamard overparametrization Year: (2026)
Ref_id:b30 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b31 Title: Neural mechanics: Symmetry and broken conservation laws in deep learning dynamics Year: (2020)
Ref_id:b32 Title: Soft threshold weight reparameterization for learnable sparsity Year: (2020)
Ref_id:b33 Title: Optimal brain damage Year: (1989)
Ref_id:b34 Title: Gradient-based learning applied to document recognition Year: (1998)
Ref_id:b35 Title: Snip: single-shot network pruning based on connection sensitivity Year: (2019)
Ref_id:b36 Title: Lassonet: Neural networks with feature sparsity Year: (2021)
Ref_id:b37 Title: Pruning filters for efficient convnets Year: (2017)
Ref_id:b38 Title: Implicit regularization for group sparsity Year: (2023)
Ref_id:b39 Title: Learning efficient convolutional networks through network slimming Year: (2017)
Ref_id:b40 Title: Learning pruning-friendly networks via frank-wolfe: One-shot, any-sparsity, and no retraining Year: (2022)
Ref_id:b41 Title: Llm-pruner: On the structural pruning of large language models Year: (2023)
Ref_id:b42 Title: Neural proximal gradient descent for compressive imaging Year: (2018)
Ref_id:b43 Title: Columbia object image library Year: (1996)
Ref_id:b44 Title: Reading digits in natural images with unsupervised feature learning Year: (2011)
Ref_id:b45 Title: Ac/dc: Alternating compressed/decompressed training of deep neural networks Year: (2021)
Ref_id:b46 Title: Neural oblivious decision ensembles for deep learning on tabular data Year: (2020)
Ref_id:b47 Title: Winning the lottery with continuous sparsification Year: (2020)
Ref_id:b48 Title: Group sparse regularization for deep neural networks Year: (2017)
Ref_id:b49 Title: A sparse-group lasso Year: (2013)
Ref_id:b50 Title: Very deep convolutional networks for large-scale image recognition Year: (2015)
Ref_id:b51 Title: Learning with Matrix Factorizations Year: (2004)
Ref_id:b52 Title: A simple and effective pruning approach for large language models Year: (2023)
Ref_id:b53 Title: Pruning neural networks without any data by iteratively conserving synaptic flow Year: (2020)
Ref_id:b54 Title: A comprehensive survey on regularization strategies in machine learning Year: (2022)
Ref_id:b55 Title: Regression shrinkage and selection via the lasso Year: (1996)
Ref_id:b56 Title: Equivalences between sparse models and neural networks Year: (2021)
Ref_id:b57 Title: Implicit regularization for optimal sparse recovery Year: (2019)
Ref_id:b58 Title: Attention is all you need Year: (2017)
Ref_id:b59 Title: Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned Year: (2019)
Ref_id:b60 Title: Picking winning tickets before training by preserving gradient flow Year: (2020)
Ref_id:b61 Title: Learning structured sparsity in deep neural networks Year: (2016)
Ref_id:b62 Title: Kernel and rich regimes in overparametrized models Year: (2020)
Ref_id:b63 Title: Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms Year: (2017)
Ref_id:b64 Title: Sparse neural additive model: Interpretable deep learning with feature selection via group sparsity Year: (2023)
Ref_id:b65 Title: Highdimensional feature selection by feature-wise kernelized lasso Year: (2014)
Ref_id:b66 Title: Model selection and estimation in regression with grouped variables Year: (2006)
Ref_id:b67 Title: 92.45% on cifar-10 in torch Year: (2015)
Ref_id:b68 Title: How sparse can we prune a deep network: A fundamental limit perspective Year: (2024)
Ref_id:b69 Title: High-dimensional linear regression via implicit regularization Year: (2022)
Ref_id:b70 Title: Learn to be efficient: Build structured sparsity in large language models Year: (2024)
Ref_id:b71 Title: Symmetry induces structure and constraint of learning Year: (2023)
Ref_id:b72 Title: Solving l1 penalty with sgd Year: (2023)
Ref_id:b73 Title: Parameter symmetry and noise equilibrium of stochastic gradient descent Year: (2024)
