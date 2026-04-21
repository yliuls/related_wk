Title: Neural Collapse beyond the Unconstrained Features Model: Landscape, Dynamics, and Generalization in the Mean-Field Regime
Abstract: Neural Collapse is a phenomenon where the lastlayer representations of a well-trained neural network converge to a highly structured geometry. In this paper, we focus on its first (and most basic) property, known as NC1: the within-class variability vanishes. While prior theoretical studies establish the occurrence of NC1 via the dataagnostic unconstrained features model, our work adopts a data-specific perspective, analyzing NC1 in a three-layer neural network, with the first two layers operating in the mean-field regime and followed by a linear layer. In particular, we establish a fundamental connection between NC1 and the loss landscape: we prove that points with small empirical loss and gradient norm (thus, close to being stationary) approximately satisfy NC1, and the closeness to NC1 is controlled by the residual loss and gradient norm. We then show that (i) gradient flow on the mean squared error converges to NC1 solutions with small empirical loss, and (ii) for well-separated data distributions, both NC1 and vanishing test loss are achieved simultaneously. This aligns with the empirical observation that NC1 emerges during training while models attain near-zero test error. Overall, our results demonstrate that NC1 arises from gradient training due to the properties of the loss landscape, and they show the co-occurrence of NC1 and small test error for certain data distributions.

Section: Introduction
Neural Collapse (NC), first identified by Papyan et al. (2020), describes a phenomenon observed during the final 1 Institute of Science and Technology Austria (ISTA), Klosterneuburg, Austria.
Correspondence to: Diyuan Wu <diyuan.wu@ist.ac.at>, Marco Mondelli <marco.mondelli@ist.ac.at>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
stages of training where: (i) the penultimate-layer features converge to their respective class means (NC1), (ii) these class means form an equiangular tight frame (ETF) or an orthogonal frame (NC2), and (iii) the columns of the final layer's classifier matrix similarly form an ETF or orthogonal frame, implementing a nearest class-mean decision rule on the penultimate-layer features (NC3). A popular line of theoretical research has investigated the occurrence of NC via the unconstrained features model (UFM), see (Fang et al., 2021a;Han et al., 2022;Mixon et al., 2022) and the discussion in Section 2. In this framework, the penultimate-layer features are treated as free optimization variables, leading to a benign loss landscape for the resulting optimization problem. The primary justification for adopting the UFM is that the complex feature-learning layers encountered in practice are approximated by a universal learner. While the UFM provides an intriguing theoretical perspective on NC, it has notable limitations. In particular, it neglects the dependence on the data distribution, rendering it unsuitable for theoretically analyzing the relationship between NC during training and the test error (Hui et al., 2022). Furthermore, the training dynamics under the UFM framework is not equivalent to the actual training dynamics of neural networks, which makes it challenging to investigate the occurrence of NC from a dynamical perspective.
To address the limitations of UFM, we consider training a three-layer network via gradient flow on the standard mean squared error (MSE) loss. Specifically, we employ a two-layer neural network in the mean-field regime (Mei et al., 2018) as the feature-learning component, and then concatenate it with a linear layer as the final predictor. Our main results both (i) establish sufficient conditions on the loss landscape for the first -and most basic -property of neural collapse, i.e., NC1, to hold, and (ii) show that such conditions are in fact satisfied by training the architecture above. This differentiates our paper from recent studies aiming to theoretically explain the NC phenomenon beyond unconstrained features, as existing work either provides only sufficient conditions for NC to occur (Seleznova et al., 2024), focuses on the NTK regime (Jacot et al., 2024), relies on specific training algorithms (Beaglehole et al., 2024) or on a specific regularization (Hong & Ling, 2024a), see Section 2 for a discussion of related work. Specifically, our contributions are summarized below:
• First, we connect the emergence of NC1, i.e., the fact that the within-class variability vanishes, with properties of the loss landscape: we show that all approximately stationary points with small empirical loss are roughly NC1 solutions, and the degree to which the within-class variability vanishes is controlled by gradient norm and loss. This implies the prevalence of NC1 during training, as practical training procedures typically converge to such points with small gradient and loss.
• Next, we prove that gradient flow on a three-layer network operating in the mean-field regime satisfies the two conditions above (small gradient norm and small loss) and, therefore, it converges to an NC1 solution. While achieving approximately stationary points is expected, the primary challenge lies in controlling the empirical loss due to the model's non-convex nature.
• Finally, we show that, for certain well-separated data distributions, it is possible to achieve NC1 during training as well as vanishing test error, which corroborates the empirical finding that NC1 and strong generalization occur simultaneously.
this section cite: ['b25', 'b7', 'b21', 'b11', 'b19', 'b27', 'b12', 'b3']

Section: Related Work
Neural collapse: UFM and beyond. The introduction of the UFM in (Mixon et al., 2022;Fang et al., 2021a) has prompted a line of work studying the emergence of neural collapse for that model. Specifically, Zhou et al. (2022) focus on the two-layer UFM model, showing that all its stationary points satisfy neural collapse. Han et al. (2022) prove convergence of gradient flow on UFM to NC solutions. Tirer & Bruna (2022) demonstrate that the global minimizers also satisfy neural collapse when the UFM has multiple linear layers or it incorporates the ReLU activation. Súkeník et al. (2023) extend the results to a deep UFM model for binary classification. Súkeník et al. (2024) then show that, for the deep UFM and multi-class classification, all the global optima still satisfy NC1, but not NC2 and NC3, due to the low-rank bias of the model. We also refer to (Kothapalli, 2023) for a rather recent and detailed review.
Going beyond the UFM, Seleznova et al. (2024) study the connection between NC and the neural tangent kernel (NTK), showing NC under certain block structure assumptions on the NTK matrix. However, the occurrence of such a block structure during training is unclear. Beaglehole et al. (2024) establish NC both empirically and theoretically for Deep Recursive Feature Machine training -a method that constructs a neural network by iteratively mapping the data through the average gradient outer product and then applying an untrained random feature map. Pan & Cao (2023) consider classification with cross-entropy loss, providing a quantitative bound for NC. Kothapalli & Tirer (2024) focus on two-layer neural networks in both the NNGP and the NTK limit, proving neural collapse for 1-dimensional Gaussian data. Hong & Ling (2024a) study NC for shallow and deep neural networks, also characterizing the generalization error. However, they regularize the loss by the L 2 -norm of the features rather than the weights, which is different from the weight decay used in practice. Jacot et al. (2024) establish the occurrence of NC for deep neural networks with multiple linear layers, given a balancedness assumption on all the linear layers; in addition, they also prove that balancedness is achieved via gradient descent training using NTK tools. Compared to (Jacot et al., 2024), our proof does not rely on any balancedness condition, and it only requires the gradient norm to be small, which is naturally achievable via gradient flow. In fact, the stationary points to which our results apply may not be balanced, see the discussions at the end of Section 4.1 and 4.2.
Mean-field analysis for networks with more than two layers. While the properties of the loss landscape and training dynamics of two-layer neural networks in the mean-field regime have been extensively studied (Mei et al., 2018;Chen et al., 2020;Javanmard et al., 2020;Shevchenko et al., 2022;Hu et al., 2021;Suzuki et al., 2024a;Takakura & Suzuki, 2024), networks with more than two layers still prove to be challenging to analyze. Prior works (Lu et al., 2020;Araújo et al., 2019;Shevchenko & Mondelli, 2020;Fang et al., 2021b;Pham & Nguyen, 2021;Nguyen & Pham, 2023) have investigated the mean-field regime for deep neural networks, where the widths of all layers tend to infinity. In contrast, we let only the width of the first layer tend to infinity, while the width of the second layer remains of constant order. A closely related paper is by Kim & Suzuki (2024), which studies the in-context loss landscape of a twolayer linear transformer with a formulation similar to ours. However, the global convergence results in (Kim & Suzuki, 2024) rely on assumptions such as absence of weight decay, taking a two time-scale limit, and using a birth-death process (rather than the widely-used gradient flow), which are not applicable to our setting.
this section cite: ['b21', 'b38', 'b7', 'b36', 'b30', 'b31', 'b15', 'b27', 'b3', 'b24', 'b16', 'b12', 'b12', 'b19', 'b4', 'b13', 'b29', 'b10', 'b34', 'b17', 'b1', 'b28', 'b26', 'b22', 'b14', 'b14']

Section: Problem Setting
Notation. Given an integer n, we use the shorthand [n] := {1, . . . , n}. Given a vector v ∈ R d , let v[i] be its i-th entry and Diag(v) ∈ R d×d the diagonal matrix with v on the diagonal. Let 1 d ∈ R d be the all-one vector of dimension d. Given a matrix A, let [A] i,j be its (i, j)-th element. We denote by ∥ • ∥ F , ∥ • ∥ op the Frobenius and operator norms of a matrix, and by ⟨A, B⟩ F = Tr A ⊤ B the Frobenius inner product. Let ⊗ be the Kronecker product and vec(•) the vectorization of the matrix obtained by stacking columns. Given a vector valued function f : R d → R d , we denote by ∇ • f : R d → R its divergence. Given a real valued function g, we denote by ∥g∥ ∞ = sup |g| its infinity norm. Let P 2 (R d ) be the space of probability measures on R d with finite second moment, and W 1 (•, •), W 2 (•, •) the Wasserstein-1 and -2 metrics, respectively.
Three-layer fully connected neural networks. We start by defining the following infinite-width neural network as a feature-learning layer:
h ρ (x) = E ρ [aσ(u ⊤ x)],(1)
where a ∈ R p , u, x ∈ R d , and ρ = Law(a, u) is the joint measure of a, u. The network is parameterized by a probability distribution ρ ∈ P 2 (R p+d ), and it represents the mean-field limit of the finite-width two-layer network below (Mei et al., 2018):
h N (x) = 1 N N j=1 a j σ(u ⊤ j x), θ j = (a j , u j ) i.i.d. ∼ ρ. (2)
For technical convenience, throughout this paper we directly consider the infinite-width network (1). In fact, its difference with the finite-width counterpart (2) can be readily bounded using results from (Mei et al., 2018;2019).
Next, we cascade a linear layer, obtaining a three-layer neural network as follows:
f (x; ρ, W ) = γW ⊤ h ρ (x),(3)
where W ∈ R p×q and γ ∈ R is a (constant) multiplicative factor. Throughout the paper, we will refer to f in (3) as the predictor. Neural networks with two linear layers in the end are also studied by Jacot et al. (2024), and adding a multiplicative factor γ is proposed by Chen et al. (2020) to guarantee the global convergence of the dynamics.
The motivation for considering this model is to explore how training data affects the emergence of neural collapse. While previous studies on UFM offer insights into neural collapse, their key limitation lies in the disregard for the influence of training data. The primary justification for using the UFM is that it functions as a universal learner, thus emulating the complex feature learning layers encountered in practice.
The three-layer network in the mean-field regime defined in (3) not only performs feature learning by taking into account the training data, but the feature layer in (1) is also recognized as a universal learner (Ma et al., 2022).
q-class balanced classification. We consider a q-class balanced classification problem, with each class having m data points. We denote by n = qm the total number of training samples and assume that p ≥ q. The empirical loss function is given by
L n (ρ, W ) = 1 2n ∥γW ⊤ H ρ -Y ∥ 2 F ,
where
H ρ = [h ρ (x 1 ), . . . , h ρ (x n )] ∈ R p×n , Y = [e 1 , . . . e 1 m columns , . . . , e q , . . . e q ] ∈ R q×n .
We also denote X = [x 1 , . . . , x n ] ∈ R d×n . We consider a regularized problem with L 2 and entropy regularization, denoting the regularized loss and the free energy as
L λ,n (ρ, W ) = L n (ρ, W )+ λ W 2 ∥W ∥ 2 F + λ ρ 2 E ρ [∥θ∥ 2 2 ], (4) E n (ρ, W ) = L n (ρ, W ) + λ W 2 ∥W ∥ 2 F + λ ρ 2 E ρ [∥θ∥ 2 2 ] + β -1 E ρ [log ρ],(5)
where the entropic regularization is explicitly added to perform noisy gradient flow, see ( 20). While we focus on balanced classification for technical clarity and brevity, our results extend to unbalanced classification (as considered e.g. in (Thrampoulidis et al., 2022;Hong & Ling, 2024b)) and regression (as considered e.g. in (Andriopoulos et al., 2024)) with minimal modifications.
this section cite: ['b19', 'b19', 'b37', 'b12', 'b4', 'b18', 'b35', 'b0']

Section: Neural collapse metric.
We focus on the first property of neural collapse and, given a feature matrix H ∈ R p×n , we consider the following metric of NC1 as the ratio between in-class variance and total variance:
N C1(H) = Tr ( H -M c ) ⊤ ( H -M c ) Tr H ⊤ H ,
where H ∈ R p×n is the matrix of centered features and M c ∈ R p×n the matrix of in-class means, defined as
M g = 1 n H1 n 1 ⊤ n , H = H -M g , M c = 1 m HY ⊤ Y.
In words, if N C1(H) is small, the within-class variability is negligible compared to the overall variability across classes, capturing the closeness of features to respective class means.
this section cite: []

Section: Within-class Variability Collapse during Training

this section cite: []

Section: Sufficient Conditions for NC1
Throughout the paper, we make the following assumptions that are mild and standard in the related literature, see e.g. (Mei et al., 2018;Chen et al., 2020;Suzuki et al., 2024a).
this section cite: ['b19', 'b4']

Section: Assumption 1. (A1) Regularity of the initialization:
We initialize the training algorithm with W 0 such that W ⊤ 0 W 0 = I q and ρ 0 = N (0, I p+d ).
(A2) Boundedness of the data: for all i, ∥x i ∥ 2 ≤ 1.
(A3) Regularity of the activation function:
∥σ(z)∥ ∞ , ∥σ ′ (z)∥ ∞ , ∥σ ′′ (z)∥ ∞ , ∥σ ′′′ (z)∥ ∞ , ∥(zσ ′ (z)) ′ ∥ ∞ ≤ C 1 for some universal constant C 1 .
We remark that the ReLU activation does not satisfy the regularity conditions above, but we still expect our results to hold by taking the limit of a sequence of approximations of ReLU.
We now define an ϵ S -stationary point of the free energy.
Definition 4.1. We say that (ρ, W ) is an ϵ S -stationary point of E n (ρ, W ) w.r.t. ρ if the following holds:
E ρ ∇ θ δ δρ E n (ρ, W ) 2 2 ≤ ϵ 2 S .
Here, we recall that, given a functional G :
P 2 (R D ) → R, its first variation at ρ is the function δ δρ G(ρ)(•) : R D → R such that, for all ρ ′ ∈ P 2 (R D ), δ δρ G(ρ)(θ) (ρ ′ -ρ)dθ = lim ϵ→0 G((1 -ϵ)ρ + ϵρ ′ ) -G(ρ) ϵ .
The result below (proved in Appendix B.1) characterizes the feature H ρ at any ϵ S -stationary point.
Theorem 4.2. Under Assumption 1, for any ϵ S -stationary point (ρ, W ), we have the following characterization of the learned feature:
vec(H ρ ) = γ K ρ (X, X) n ⊗ W • γ 2 K ρ (X, X) n ⊗ (W ⊤ W ) + λ ρ I nq -1 vec(Y ) + E 1 (ϵ S , λ ρ ; γ, W ),(6)
where
∥E 1 (ϵ S , λ ρ ; γ, W )∥ 2 2 ≤2 λ -4 ρ γ 4 C 2 1 σ max (W ) 4 +λ -2 ρ C 2 1 nϵ 2 S , (7) and the kernel K ρ (X, X) ∈ R n×n induced by ρ is K ρ (X, X) = E ρ [σ(X ⊤ u)σ(u ⊤ X)].(8)
As a consequence, if W is non-singular, we have
H ρ = γ -1 W (W ⊤ W ) -1 Y + E 2 (ϵ S , λ ρ ; γ, ρ, W ), (9
)
where
∥E 2 (ϵ S , λ ρ ; γ, ρ, W )∥ 2 F ≤ 2 λ -4 ρ γ 4 C 2 1 σ max (W ) 4 + λ -2 ρ • C 2 1 nϵ 2 S + 2nγ -2 L n (ρ, W ) + 2λ -2 ρ σ max (W ) 2 C 2 1 nϵ 2 S σ min (W ) 2 . (10
)
Proof sketch. As (ρ, W ) is ϵ S -stationary, the following expression for a holds almost surely w.r.t. the measure ρ:
a + γλ -1 ρ n W (γW ⊤ H ρ -Y )σ(X ⊤ u) + λ -1 ρ β -1 ∇ a log ρ(θ) = O(ϵ S ),(11)
where, with an abuse of notation, the term O(ϵ S ) indicates that the norm of the vector on the LHS is at most of order ϵ S . By plugging (11) into H ρ = E ρ [aσ(u ⊤ X)], we obtain
H ρ = -λ -1 ρ γW (γW ⊤ H ρ -Y ) K ρ (X, X) n +O(ϵ S ). (12
)
Note that (12) is a linear equation in H ρ . Thus, by solving it explicitly and tracking the error in ϵ S , we obtain (6). Finally, the crux of the argument for (9) is to use again stationarity to show that (up to an error of order ϵ S )
(K ′ ⊗ W ′ ) K ′ ⊗ (W ′⊤ W ′ ) + λ ρ I nq -1 vec(Y ) = vec(W ′ (W ′⊤ W ) -1 Y ) + O(L n (ρ, W )),(13)
where K ′ := K ρ (X, X)/n and W ′ := γW .
Note that γ -1 W (W ⊤ W ) -1 Y , i.e., the first term in the decomposition of H ρ in (9), satisfies NC1. Indeed, Y is the one-hot vector of labels and, thus, for two data point x i , x j in the same class k, we have y i = y j , which implies that W (W ⊤ W ) -1 y i = W (W ⊤ W ) -1 y j . The second term E 2 in the decomposition (9) is small, as long as ϵ S and L n (ρ, W ) are small. Hence, the key question is whether we can achieve a nearly stationary point having a small loss L n (ρ, W ) via a certain training dynamics, which is addressed in the next sections.
As the result in (9) requires W not to be too ill-conditioned, we now prove that this is the case, as long as the regularization terms λ W , λ ρ , β -1 and the regularized loss L λ,n are sufficiently small.
Lemma 4.3. Let λ ρ = λ 0 ρ β -1 , λ W = λ 0 W β -1 where λ 0
ρ , λ 0 W are universal constants. Fix any α ≥ 0, 0 < ϵ 0 ≤ 1/2, and assume that
β ≥ max e 4α ϵ 0 log 2α ϵ 0 ,e 4α log(2α) , (2C 2 1 nB(λ 0 ρ ) -1 ) 2 ϵ 0 , 4q n 1 ϵ 0 , 64(qB) 2 ,(14)
for some constant B that doesn't depend on β. Suppose further that (ρ, W ) is any point such that
L λ,n (ρ, W ) ≤ Bβ -1 (log β) α .(15)
Then, we have that
σ min (W ) ≥ β -ϵ0 , σ max (W ) 2 ≤ ∥W ∥ 2 F ≤ 2B(λ 0 W ) -1 (log β) α .
The proof is by contradiction. Suppose that W has a small singular value, then the projection of H ρ in the corresponding left singular space of W needs to be large, since the regularized loss is small and Y is isotropic. However, large component of H ρ in a subspace will in turn lead to large regularized loss due to the second-moment regularization term. The complete argument is deferred to Appendix B.2.
Next, we compute the NC1 metric induced by Theorem 4.2.
Corollary 4.4. Consider the setting of Theorem 4.2 and assume that
∥E 2 ∥ 2 F ≤ 1 8σ 2 max (W ) (q -1)n q , (16
)
where E 2 := E 2 (ϵ S , β; γ, ρ, W ) is bounded as in (10). Then, for any ϵ S -stationary point (ρ, W ) with non-singular W , we have
N C1(H ρ ) ≤ 16∥E 2 ∥ 2 F 1 2σ 2 max (W ) (q-1)n q -4∥E 2 ∥ 2 F . (17
)
The proof of Corollary 4.4 is a direct calculation, and it is provided in Appendix B.3. Note that, in the setting of Lemma 4.3, we have that
1 8σ 2 max (W ) (q -1)n q = Ω((log β) -α ), ∥E 2 ∥ 2 F = O(β -1+2ϵ0 (log β) α + β 4+2ϵ0 (log β) 4α ϵ 2 S ).(18)
Now, let us pick a sufficiently small β -1 (corresponding to small regularization) and then a sufficiently small ϵ S (corresponding to reaching a stationary point). Then, (18) implies that (16) holds and the upper bound on the NC1 metric in (17) vanishes.
Imbalancedness of stationary point. The recent work by Jacot et al. (2024) shows that, for any network with at least two consecutive linear layers in the end, sufficiently small loss and approximate balancedness of the linear layers suffice to guarantee NC1. Our network defined in (3) has two final linear layers, but due to the entropic regularization, all stationary points of the free energy are not balanced, which means that the techniques in (Jacot et al., 2024) cannot be applied to our setup. To demonstrate this, we prove in Appendix B.4 the following result.
Lemma 4.5. Let (ρ, W ) be a stationary point of the free energy, i.e.,
∇ θ δ δρ E n (ρ, W ) = 0, ∇ W E n (ρ, W ) = 0,
with E ρ [a] < ∞. Then, any stationary point satisfies
λ ρ E ρ [aa ⊤ ] -λ W W W ⊤ = β -1 I p .(19)
The result in (19) implies that the network cannot be balanced, i.e., E ρ [aa ⊤ ] cannot be proportional to W W ⊤ . In fact, assume that λ ρ and λ W are of same order as β -1 , i.e., λ ρ = λ 0 ρ β -1 and λ W = λ 0 W β -1 for universal constants λ 0 ρ , λ 0 W . Then, as W W ⊤ is of rank q < p, (19) gives that, for any constant c, ∥E ρ [aa ⊤ ] -cW W ⊤ ∥ op ≥ (λ 0 ρ ) -1 . We complement the theoretical result in Lemma 4.5 with numerical simulations, discussed at the end of Section 4.2, showing that for there are settings such that gradient-based training over standard datasets (MNIST, CIFAR-100) the neural network achieves NC1 without converging to a balanced solution.
this section cite: ['b12', 'b12']

Section: Achieving NC1 via Gradient-based Training
From Theorem 4.2 and Corollary 4.4, we know that NC1 is achieved at any ϵ S -stationary point w.r.t. ρ having small empirical loss. We now consider training ρ and W with gradient flow, i.e.,
dW t = -∇ W L λ,n (ρ t , W t )dt; dθ t = -∇ θ δ δρ L λ,n (ρ t , W t )(θ t )dt + 2β -1 dB t ,(20)
and we show that, having trained long enough, one ensures that both ϵ S and the empirical loss are sufficiently small.
The convergence to an ϵ S -stationary point with arbitrary small ϵ S is a direct consequence of the fact that, under gradient flow, the gradient norm vanishes.
Lemma 4.6. Under Assumption 1, fix β, γ > 0 and consider an initialization (ρ 0 , W 0 ) with finite free energy. For t ≥ 0, let
ϵ t S = E ρt ∇ θ δ δρ E n (ρ t , W t )(θ t ) 2 2 ,
which is equivalent to (ρ t , W t ) being an ϵ t S -stationary point. Then, for any ϵ S > 0, there exists T (ϵ S ) > 0 s.t. for all t > T (ϵ S ) except a finite Lebesgue measure set,
ϵ t S ≤ ϵ S .
The proof of Lemma 4.6 is provided in Appendix C.1. This result directly implies that lim inf t→+∞ ϵ t S = 0. Next, Theorem 4.8 shows that, by picking large enough γ and training long enough, we achieve O(β -1 ) empirical loss. This requires the following mild assumptions that imply the positive definiteness of the kernel K ρ in (8) at initialization, as showed in Lemma 4.7.
Assumption 2. Assume σ (2k) ̸ = 0 for all k > 0 and that there exist s ∈
[d] s.t. (i) x i [s] ̸ = x j [s] for all i ̸ = j ∈ [n], and (ii) x i [s] ̸ = 0 for all i ∈ [n].
In words, the activation function is required to be smooth and have non-zero even derivatives, which is satisfied by e.g. tanh or the sigmoid (also fulfilling Assumption 1). As for the training data, we assume that it is non-degenerate and not parallel, which holds for most practical data sets.
The next technical lemma, which comes from (Nguyen & Mondelli, 2020, Lemma 3.4)foot_0 , shows the required positive definiteness of the kernel. Lemma 4.7.
Under Assumption 2, let K(X, X) = E u∼γ d [σ(X ⊤ u)σ(u ⊤ X)], where γ d = N (0, I d ). Then, λ * := λ min (K(X, X)) > 0.
We are now ready to state our result showing the convergence of the empirical loss to a low loss manifold by running gradient flow for long enough time. Theorem 4.8. Let Assumptions 1, 2 hold, set λ ρ = λ W = β -1 and γ > C 3 , t 0 = βC 5 .
Then, for any β and any t ≥ t 0 , we have
L λ,n (ρ t , W t ) ≤ β -1 C 4 ,(22)
where C 3 , C 4 , C 5 are constants that depends on n, d, p, C 1 , λ * but not on β.
The expression of C 3 , C 4 , C 5 is provided in Theorem C.1, whose statement and proof are in Appendix C.2. We note that the choice
λ ρ = λ W = β -1 is only for technical convenience, what matters here is that λ ρ , λ W = Θ(β -1 ).
Proof sketch. We start by defining a first-hitting time
t * = min{ inf{t : ∥W ⊤ t W t -W ⊤ 0 W 0 ∥ op > R W }, inf{t : D KL (ρ t ||ρ 0 ) > R ρ }},(23)
where D KL (•||•) denotes the KL divergence. Intuitively, (23) means that, for t < t * , the gradient flow stays in a ball around the initialization. The crux of the argument is to show that, with a suitable choice of R W , R ρ , the loss becomes small before the dynamics has exited the ball.
To do so, we first prove in Lemma C.2 that, for t < t * ,
L n (ρ t , W t ) ≤ exp -γ 2 A 1 t + γ -2 β -2 A 2 ,(24)
for some A 1 , A 2 that do not depend on γ, β (but only on λ * , p, d, n). This implies that the empirical loss converges exponentially fast (in t) to an error of order β -2 , as long as the gradient flow is inside the ball. We then show that, by picking a proper γ, t * is large enough so that the first term in ( 24) is of order β -2 for some t 0 < t * .
Finally, by combining the upper bound on the empirical loss with the fact that D KL (ρ t0 , ρ 0 ), ∥W t0 ∥ 2 F are bounded by a constant independent of β, we obtain that E n (ρ t0 , W t0 ) = O(β -1 ). Thus, since the free energy decreases along the gradient flow, the upper bound in (22) follows from the relationship between empirical loss and free energy proved in Lemma A.3.
We remark that we also provide a convergence rate for the loss in ( 24). The problem is challenging due to its nonconvex nature, and to our best knowledge, no explicit rate of convergence is known in the mean-field regime beyond the two-layer case (which has a convex free energy).
Comparison with related work. While the strategy described above is motivated by and similar to that used in (Chen et al., 2020, Theorem 4.4), its technical implementation differs, due to differences in the problem setting. In fact, Chen et al. (2020) consider two-layer neural networks whose free-energy landscape is strongly convex in ρ. In contrast, the presence of the last linear layer W implies that the free-energy landscape is non-convex in (ρ, W ), which makes it less obvious that gradient flow converges to a low free-energy manifold. As a consequence, Chen et al. (2020) show that the gradient flow dynamics stays in a ball around initialization with a certain radius for infinitely long time. In contrast, in our case, the gradient flow dynamics stays in the ball only for finite time, but this finite time suffices to ensure a small enough free energy.
Finally, the combination of Theorem 4.8, Lemma 4.6 and Corollary 4.4 gives that NC1 provably holds under gradient flow training.
Corollary 4.9. Consider the setting of Theorem 4.8 and, for any 0 < δ 0 < 1, let
β > max (2C 2 1 nC 4 ) 6 , 4q n 3 , 64(qC 4 ) 2 , 640C -2 3 C 3 4 1 δ 0 3 ,
where γ, t 0 as chosen as in (21). Then, there exists T (β) > 0 s.t. for all t > max{T (β), t 0 } except a finite Lebesgue measure set, N C1(H ρt ) ≤ δ 0 .
The proof of Corollary 4.9 is deferred to Appendix C.3, and the result implies that lim inf t→+∞ N C1(H ρt ) ≤ δ 0 . Corollary 4.9 implies that, the three-layer model (almost) always achieve NC1 solution, for long enough training, which explains the prevalence of neural collapse in practice.
Imbalancedness after gradient-based training. The numerical results of Figure 1 show that, even if the solution obtained via gradient descent is not balanced, its training loss and gradient norm are still small and, therefore, as predicted by our analysis, it satisfies NC1. As a normalized balancedness measure, we use with
N B(ρ, W ) = ∥E ρ [aa ⊤ ] -c * W W ⊤ ∥ op ∥E ρ [aa ⊤ ]∥ op ,(25)
c * = arg min c ∥E ρ [aa ⊤ ] -cW W ⊤ ∥ 2 F . (26
)
This captures the extent to which E ρ [aa ⊤ ] and W W ⊤ are proportional. We then train the three-layer neural network f N (x) = W ⊤ h N (x), where the output h N (x) of the first two layers is given by ( 2). Specifically, we consider the following two settings.
Setting (a): MNIST. We relabel the dataset into q = 3 classes taking the original label modulo 3, and we randomly pick 10000 samples in each new class for training. The input dimension is d = 784, the number of neurons in the first layer is N = 6272, and the number of neurons in the second one is p = 16. We train the model with SGD of batch size 64 and learning rate η ∈ {0.001, 0.01}, using the smaller (larger) learning rate for the first (second) half of the epochs. We pick weight decay λ W = λ ρ = 10 -4 , but add no noise (β -1 = 0) and fix the last linear layer at initialization, which produces an imbalanced network. In fact, at convergence, N B(ρ, W ) = {0.8093, 0.7149, 0.5895, 0.8531} in our 4 independent experiments. We also plot the evolution of the normalized balancedness metric in (25) as a function of the number of training epochs in Figure 2a of Appendix E, which shows that the network does not achieve balancedness throughout training. However, even if the network is not balanced, Figure 1a still shows that the NC1 metric decreases and flattens to a rather low value, following the same pattern as the training loss and the gradient norm.
Setting (b): CIFAR-100. We perform classification on superclasses using pretrained ResNet50 features. Specifically, we consider the 3 super-classes ["aquatic mammals", "large carnivores", "people"], with each super-class containing 5 original classes and 500 samples in total. We then take a ResNet50 pretrained on ImageNet-1K, extract the penultimate-layer features of the training set, and use such features as training data. The input dimension is d = 2048, the number of neurons in the first layer is N = 16384, and the number of neurons in the second one is p = 64. We train the model with noisy SGD of batch size 64, pick weight decay λ W = λ ρ = β = 10 -4 and learning rate η ∈ {0.001, 0.0001}, using the smaller (larger) learning rate for the first (second) half of the epochs. As in the previous case, the network does not achieve balancedness throughout training: at convergence, N B(ρ, W ) = {0.5386, 0.2779, 0.4141, 0.5257} in the 4 independent experiments; see also Figure 2b in Appendix E for a plot of the metric in (25) as a function of the number of training epochs. Nevertheless, the NC1 metric decreases with the loss and the gradient norm, reaching a small value at the end of training, see Figure 1b.
Similar results are reported for the training of ResNet-18 and VGG-11 in Appendix E. The implementation of the experiments is publicly available at the GitHub repository https:  //github.com/DiyuanWu/icml25_expr.
this section cite: ['b23', 'b4', 'b4']

Section: Within-class Variability Collapse and Generalization
While neural collapse is widely known as a phenomenon occurring at training time, it does not necessarily imply that the test error is small (Hui et al., 2022, Section 4). We now show that, for well-separated datasets, training via gradient flow implies both approximate NC1 and small test error.
Problem setting. We make the following additional assumptions (consistent with Assumption 1).
Assumption 3. We set γ = 1, and assume the activation function σ to be the sigmoid function, i.e., σ(z) = 1 1+e -z . We further assume there are q classes and n data points (x j , y j ) sampled i.i.d. from D, with m points for each class and x j ∼ D(x j |y j ). Each class is balanced, in the sense that D(•, e k ) = 1/q for all k.
dW t = -∇ W L λ,n (ρ t , W t )dt; dθ t = -∇ θ δ δρ L λ,n (ρ t , W t )(θ t )dt + 2β -1 dB t .
For technical reasons, we consider the approximated model obtained by truncating the second layer, i.e.
, h R ρ (x) = E ρ [τ R (a)σ(u ⊤ x)], where τ R : R → R is a smooth function applied component-wise such that τ R (z) =      z, for |z| ≤ R, R + C 0 , for |z| ≥ 2R, smooth interpolation, for R < |z| < 2R.
Notably, for large enough R, the derivative of τ satisfies
τ ′ R (z) =      1, for |z| ≤ R, 0, for |z| ≥ 2R, ≤ C 0 , for R < |z| < 2R. We also denote H R ρ = [h R ρ (x 1 ), . . . , h R ρ (x n )] ∈ R p×n
, and define the loss and free energy w.r.t. the approximated second layer as
L R n (ρ, W ) = 1 2n ∥W ⊤ H R ρ -Y ∥ 2 F , E R n (ρ, W ; β) = L R n (ρ, W ) + β -1 2 ∥W ∥ 2 F + β -1 2 E ρ [∥θ∥ 2 2 ] + β -1 E ρ [log ρ].
We remark that the technical reason for having an approximated second layer is to ensure the uniqueness of the global optimum for the Gibbs minimizer as discussed in Proposition 5.1. Nevertheless, our results in Section 5 hold uniformly for large enough R, which means that we also expect the same conclusion for the original model which corresponds to R = +∞.
Two stage training algorithm. Our result holds for the two-stage training described in Algorithm 1. Specifically, in Stage 1, we aim to find the global optimum of E R n (W 0 , ρ) having fixed W 0 , and in Proposition 5.1 below we show that, for all fixed non-zero W 0 , E R n (W 0 , ρ) has a unique global minimizer in P 2 (R p+d ). Furthermore, such global minimizer is achieved by noisy gradient flow as studied by (Suzuki et al., 2024a). In Stage 2, we run a gradient flow on the free energy, as we did in Section 4.2. Proposition 5.1. Under Assumption 1, for any fixed nonzero W , E R n (ρ, W ) is strongly convex in ρ, and there exist a unique global minimizer ρ with the following Gibbs form:
ρ(θ)∝exp - β n τ R (a) ⊤ W (W ⊤ H R ρ -Y )σ(X ⊤ u)- β 2 ∥θ∥ 2 2 . (27
)
Proof. The result follows from the strong convexity of E R n (ρ, W ). In fact,
1 2n ∥W ⊤ H R ρ -Y ∥ 2 F is convex in H R ρ , H R
ρ is linear in ρ, the L 2 -regularization is convex and the entropic regularization is strongly convex. Then, the claim is a consequence of ( Hu et al., 2021
this section cite: ['b11']

Section: , Proposition 2.5).
Test error analysis. We start by introducing (τ, M )linearly separable data, which intuitively corresponds to each class being linearly separable w.r.t. the others. Definition 5.2. We say that the data distribution D of a qclass classification problem is bounded and (τ, M )-linearly separable if, for each k, there exist ûk s.t. ∥û k ∥ 2 2 ≤ M 2 and
û⊤ k x ≥ τ, if x ∈ supp(D(•|e k )), < -τ if x ∈ supp(D(•|e k ′ )), for all k ′ ̸ = k.
Given a predictor f : R d -→ R q , we aim to bound the mismatch error:
err test (f ; D) = 1 q q k=1 Pr x∼D(•|e k ) [One-Hot(f (x)) ̸ = e k ],
where we define the function One-Hot : R q -→ R q as
[One-Hot(f )] i = 1, if i = arg max i [f ] i , 0, else.
While the (τ, M )-linear separability of the data seem to be restrictive, we remark that in general it is not true that NC1 and vanishing test error co-occur under MSE loss without any assumptions on the data distribution. In fact, the occurrence of NC1 implies that the model overfits the data, and overfitting is not always benign without additional assumptions (Bartlett et al., 2021). In this sense, showing the co-occurrence of NC1 and vanishing test error may be regarded as a harder problem than benign overfitting.
We now show that training on a (τ, M )-linearly separable dataset leads to both neural collapse and test error vanishing in the number of training samples n. Theorem 5.3. Under Assumptions 1 and 3, let the data distribution be bounded and (τ, M )-linear separable as per Definition 5.2. Pick R > 1 large enough, n large enough, and
β = 640C 2 1 nC 2 9 1 δ06
. Then, for any (ρ t , W t ) obtained by Stage 2 of Algorithm 1, we have
err test (f (•; ρ t , W t ); D) ≤ C 10 log(C 11 n/δ 0 ) 1 2n + 6q log(2/δ) n ,
with probability at least 1 -δ. Furthermore, there exists T (β) s.t. for all t > T (β) except a finite Lebesgue measure set, N C1(H ρt ) ≤ δ 0 .
The constants C 9 , C 10 , C 11 depend on d, p, C 0 , C 1 , M, τ, but not on n. Their expression is provided in Theorem D.6, whose statement and proof are in Appendix D.1. The argument uses Rademacher complexity bounds for neural networks in the mean-field regime as in (Chen et al., 2020;Suzuki et al., 2024b;Takakura & Suzuki, 2024), and the key component is to control the dependence of the constant C 10 on n. This is achieved by noting that, for a (τ, M )-separated data distribution, a two-layer network with constant number of neurons approximately interpolates the data.
In a nutshell, Theorem 5.3 provides a sufficient condition on the data distribution to achieve both NC1 and vanishing test error. Although for simplicity in the statement we pick a specific value for β, we note that a similar result would
hold for 640C 2 1 nC 2 9 1 δ0 6 ≤ β ≤ O(poly(n)).
this section cite: ['b2', 'b4', 'b34']

Section: Conclusions and Future Directions
In this work, we consider a three-layer neural network in the mean-field regime and give rather general sufficient conditions for within-class variability collapse (namely, NC1) to occur. We then show that (i) training the three-layer neural network with gradient flow satisfies these conditions, and (ii) a vanishing test error is compatible with neural collapse at training time. Taken together, our results connect representation geometry to loss landscape, gradient flow dynamics and generalization, offering new insights into gradient-based optimization in deep learning.
Three interesting future directions include: (i) establishing more general conditions (either necessary or sufficient) that guarantee both neural collapse during training and vanishing test error; (ii) tackling the challenging case in which there is a non-linearity between the last two layers -a setting where the properties of neural collapse have been proved in the UFM framework for binary classification (Súkeník et al., 2023); and (iii) extending the results to cross-entropy loss, which is more commonly used for classification. The main technical difficulty for the latter is to rule out the possibility that different data points in the same class could have different logits, which appears challenging even when the loss is small.
A. Technical Lemmas 3. Singular space of Kronecker product: given two matrices A ∈ R m×n and B ∈ R p×q , let their SVD be
A = U A S A V ⊤ A , B = U B S B V ⊤ B ,
where
U A ∈ R m×m , S A ∈ R m×n , V A ∈ R n×n and U B ∈ R p×p , S B ∈ R p×q , V B ∈ R q×q . Then, the SVD of A ⊗ B reads A ⊗ B = (U A ⊗ U B )(S A ⊗ S B )(V A ⊗ V B ) ⊤ .
Proof. The first two claims can be easily verified. For the third, we have:
(U A ⊗ U B )(S A ⊗ S B )(V A ⊗ V B ) ⊤ = (U A ⊗ U B )(S A ⊗ S B )(V ⊤ A ⊗ V ⊤ B ) = (U A ⊗ U B )((S A V ⊤ A ) ⊗ (S B V ⊤ B )) = ((U A S A V ⊤ A ) ⊗ (U B S B V ⊤ B )) = A ⊗ B, and (U A ⊗ U B ) ⊤ (U A ⊗ U B ) = (U ⊤ A U A ) ⊗ (U ⊤ B U B ) = I m ⊗ I p = I mp ,
which gives the desired result.
Lemma A.2. Given two matrix A, B ∈ R n×n , assume that A is invertible and A + B is invertible , then we have:
(A + B) -1 = A -1 -(A + B) -1 BA -1 .
Proof. Let (A + B) -1 = A -1 + C where we aim to compute C. Then, we have (A + B)A -1 + (A + B)C = I. This implies that BA -1 + (A + B)C = 0, and we have C = -(A + B) -1 BA -1 , which gives the desired result.
Lemma A.3. Let ρ(θ) ∈ P 2 (R D ) be an absolutely continuous measure, L(ρ) : P 2 -→ R be a non-negative functional, and
E(ρ) = L(ρ) + λ 2 E ρ [∥θ∥ 2 2 ] + β -1 E ρ [log ρ].
Then, for any ρ ∈ P 2 (R D ), we have that
L(ρ) ≤ E(ρ) + β -1 D 2 log 2π λβ , D KL (ρ||ρ 0 ) ≤ β E(ρ) + β -1 D 2 log 2π λβ , E ρ [∥θ∥ 2 2 ] ≤ 4λ -1 E(ρ) + 4λ -1 β -1 1 + D log 8π λβ , L(ρ) + λ 2 E ρ [∥θ∥ 2 2 ] ≤ 3E(ρ) + β -1 D 2 log 2π λβ + 2β -1 1 + D log 8π λβ ,
where ρ 0 ∝ exp -βλ∥θ∥ 2 2 /2 .
Proof. Note that
β -1 βλ 2 E ρ [∥θ∥ 2 2 ] + β -1 E ρ [log ρ] =β -1 E ρ log ρ (2π/(βλ)) -D/2 exp(-βλ∥θ∥ 2 2 /2) -β -1 D 2 log 2π βλ ≥ -β -1 D 2 log 2π λβ ,
where the last passage follows from the non-negativity of the KL divergence. This implies that
E(ρ) = L(ρ) + β -1 D KL (ρ||ρ 0 ) -β -1 D 2 log 2π λβ ≥ L(ρ) -β -1 D 2 log 2π λβ ,
which gives the first two inequalities. The third inequality on the second moment follows from ( Mei First, given ρ, W , we define
∆ a (θ; ρ, W ) := ∇ a δ δρ E n (ρ, W )(θ) = γ n W (γW ⊤ H ρ -Y )σ(X ⊤ u) + λ ρ a + β -1 ∇ a log ρ(θ), ρ a.s.(28)
and we will use the shorthand ∆ a in the rest of the proof. By the definition of ϵ S -stationary point, we have
E ρ [∥∆ a ∥ 2 2 ] ≤ ϵ 2 S .
By rearranging terms in (28), we have
a = - γλ -1 ρ n W (γW ⊤ H ρ -Y )σ(X ⊤ u) -λ -1 ρ β -1 ∇ a log ρ(θ) + λ -1 ρ ∆ a , ρ a.s.
which implies that
H ρ = E ρ [aσ(u ⊤ X)] = -λ -1 ρ γW (γW ⊤ H ρ -Y ) K ρ (X, X) n -λ -1 ρ β -1 E ρ [∇ a log ρ(θ)σ(u ⊤ X)] + λ -1 ρ E ρ [∆ a σ(u ⊤ X)].
We first show that the term E ρ [∇ a log ρ(θ)σ(u ⊤ x)] = 0, for any x. To see this, it is sufficient to show that
∂ a1 log ρ(θ)σ(u ⊤ x) ρ(dθ) = 0.
Indeed, we have
∂ a1 log ρ(θ)σ(u ⊤ x) ρ(dθ) = ∂ a1 ρ(θ)σ(u ⊤ x) dθ = -∂ a1 σ(u ⊤ x) ρ(dθ) = 0, which implies that H ρ = -λ -1 ρ γW (γW ⊤ H ρ -Y ) K ρ (X, X) n + λ -1 ρ E ρ [∆ a σ(u ⊤ X)].(29)
By multiplying both sides of (29) with γW ⊤ and subtracting Y , we get
γW ⊤ H ρ -Y = -λ -1 ρ γ 2 W ⊤ W (γW ⊤ H ρ -Y ) K ρ (X, X) n + λ -1 ρ γW ⊤ E ρ [∆ a σ(u ⊤ X)] -Y.
An application of the first property stated in Lemma A.1 gives that
vec(γW ⊤ H ρ -Y ) = -λ ρ γ 2 K ρ (X, X) n ⊗ (W ⊤ W ) + λ ρ I nq -1 vec(Y ) -λ -1 ρ vec(γW ⊤ E ρ [∆ a σ(u ⊤ X)]) . (30
)
Plugging the expression for vec(γW ⊤ H ρ -Y ) back to ( 29), we have
vec(H ρ ) = -λ -1 ρ γ K ρ (X, X) n ⊗ W vec(γW ⊤ H ρ -Y ) + λ -1 ρ vec(E ρ [∆ a σ(u ⊤ X)]) = γ K ρ (X, X) n ⊗ W γ 2 K ρ (X, X) n ⊗ (W ⊤ W ) + λ ρ I nq -1 vec(Y ) -λ -1 ρ vec(W ⊤ E ρ [∆ a σ(u ⊤ X)]) + λ -1 ρ vec(E ρ [∆ a σ(u ⊤ X)]) = γ K ρ (X, X) n ⊗ W γ 2 K ρ (X, X) n ⊗ (W ⊤ W ) + λ ρ I nq -1 vec(Y ) + E 1 (ϵ S , λ ρ ; γ, W ),
where we define the error vector as
E 1 (ϵ S , λ ρ ; γ, W ) = -λ -1 ρ γ K ρ (X, X) n ⊗ W γ 2 K ρ (X, X) n ⊗ (W ⊤ W ) + λ ρ I nq -1 vec(γW ⊤ E ρ [∆ a σ(u ⊤ X)]) + λ -1 ρ vec(E ρ [∆ a σ(u ⊤ X)]).
Now we aim to upper bound the error. To do so, we write
∥E 1 (ϵ S , λ ρ ; γ, W )∥ 2 2 ≤ 2λ -2 ρ γ K ρ (X, X) n ⊗ W γ 2 K ρ (X, X) n ⊗ (W ⊤ W ) + λ ρ I nq -1 vec(γW ⊤ E ρ [∆ a σ(u ⊤ X)]) 2 2 + 2λ -2 ρ ∥vec(E ρ [∆ a σ(u ⊤ X)])∥ 2 2 ≤ 2λ -2 ρ σ max γ K ρ (X, X) n ⊗ W 2 σ max γ 2 K ρ (X, X) n ⊗ (W ⊤ W ) + λ ρ I nq -1 2 ∥γW ⊤ E ρ [∆ a σ(u ⊤ X)]∥ 2 F + 2λ -2 ρ ∥E ρ [∆ a σ(u ⊤ X)]∥ 2 F ≤ 2λ -4 ρ γ 4 C 2 1 σ max (W ) 4 + 2λ -2 ρ ∥E ρ [∆ a σ(u ⊤ X)]∥ 2 F .(31)
Then, we upper bound ∥E ρ [∆ a σ(u ⊤ X)]∥ 2 F as follows:
∥E ρ [∆ a σ(u ⊤ X)]∥ 2 F ≤ E ρ [∥∆ a σ(u ⊤ X)∥ 2 F ] = E ρ [∥∆ a ∥ 2 2 ∥σ(u ⊤ X)∥ 2 2 ] ≤ C 2 1 nE ρ [∥∆ a ∥ 2 2 ] ≤ C 2 1 nϵ 2 S .(32)
Combining ( 31) and ( 32), ( 6)-(7) readily follow.
To obtain (9), we first show that, when W ⊤ W is full rank, the following equality holds
γ K ρ (X, X) n ⊗ W γ 2 K ρ (X, X) n ⊗ W ⊤ W + λ ρ I nq -1 = γ -1 (I n ⊗ (W (W ⊤ W ) -1 )) -λ ρ γ -1 (I n ⊗ (W (W ⊤ W ) -1 )) K ρ (X, X) n ⊗ W ⊤ W + λ ρ I nq -1 .(33)
To do so, define the eigen-decomposition of Kρ(X,X) n and SVD of W as follows:
K ρ (X, X) n = U K Σ K U ⊤ K , γW = U W S W V ⊤ W ,
where
U K ∈ R n×n , Σ K ∈ R n×n , U W ∈ R p×p , S W ∈ R p×q , V W ∈ R q×q . By Lemma A.1, we have that γ K ρ (X, X) n ⊗W = (U K ⊗U W )(Σ K ⊗S W )(U K ⊗V W ) ⊤ , γ 2 K ρ (X, X) n ⊗W ⊤ W = (U K ⊗V W )(Σ K ⊗(S ⊤ W S W ))(U K ⊗V W ) ⊤ .
Thus, we have the following equalities:
γ K ρ (X, X) n ⊗ W γ 2 K ρ (X, X) n ⊗ W ⊤ W + λ ρ I nq -1 = (U K ⊗ U W )(Σ K ⊗ S W )(U K ⊗ V W ) ⊤ (U K ⊗ V W )(Σ K ⊗ (S ⊤ W S W ) + λ ρ I nq ) -1 (U K ⊗ V W ) ⊤ = (U K ⊗ U W )(Σ K ⊗ S W )(Σ K ⊗ (S ⊤ W S W ) + λ ρ I nq ) -1 (U K ⊗ V W ) ⊤ .
For simplicity, we write the matrix S W = diag(σ 1 , . . . , σ q ) 0 p-q,q and define
S -1 W = diag(σ -1 1 , . . . , σ -1 q ) 0 p-q,q
. Here, given integers n, m, we define 0 n,m as the n × m matrix containing zeros. Clearly, we have that
S -1 W S ⊤ W = I q 0 q,p-q 0 p-q,q 0 p-q,p-q .
Next, we observe that
(I n ⊗ S -1 W S ⊤ W )(Σ K ⊗ S W ) = Σ K ⊗ (S -1 W S ⊤ W S W ) = Σ K ⊗ S W .
Thus, we can write
(Σ K ⊗ S W )(Σ K ⊗ (S ⊤ W S W ) + λ ρ I nq ) -1 = (I n ⊗ S -1 W S ⊤ W )(Σ K ⊗ S W )(Σ K ⊗ (S ⊤ W S W ) + λ ρ I nq ) -1 = (I n ⊗ S -1 W )(I n ⊗ S ⊤ W )(Σ K ⊗ S W )(Σ K ⊗ (S ⊤ W S W ) + λ ρ I nq ) -1 = (I n ⊗ S -1 W )(Σ K ⊗ (S ⊤ W S W ))(Σ K ⊗ (S ⊤ W S W ) + λ ρ I nq ) -1 = (I n ⊗ S -1 W ) I -λ ρ (Σ K ⊗ (S ⊤ W S W ) + λ ρ I nq ) -1 , which implies that γ K ρ (X, X) n ⊗ W γ 2 K ρ (X, X) n ⊗ W ⊤ W + λ ρ I nq -1 = (U K ⊗ U W )(I n ⊗ S -1 W )(U K ⊗ V W ) ⊤ -λ ρ (U K ⊗ U W )(I n ⊗ S -1 W )(Σ K ⊗ (S ⊤ W S W ) + λ ρ I nq ) -1 (U K ⊗ V W ) ⊤ .
Finally, we can verify that:
γ -1 I n ⊗ (W (W ⊤ W ) -1 ) = (U K U ⊤ K ) ⊗ (U W S -1 W V ⊤ W ) = (U K ⊗ U W )(I n ⊗ S -1 W )(U K ⊗ V W ) ⊤ ,and
γ -1 (I n ⊗ (W (W ⊤ W ) -1 )) γ 2 K ρ (X, X) n ⊗ W ⊤ W + λ ρ I nq -1 =(U K ⊗ U W )(I n ⊗ S -1 W )(U K ⊗ V W ) ⊤ (U K ⊗ V W )(Σ K ⊗ (S ⊤ W S W ) + λ ρ I nq ) -1 (U K ⊗ V W ) ⊤ =(U K ⊗ U W )(I n ⊗ S -1 W )(Σ K ⊗ (S ⊤ W S W ) + λ ρ I nq ) -1 (U K ⊗ V W ) ⊤ ,
which gives (33).
From the above decomposition, we know that
γ K ρ (X, X) n ⊗ W γ 2 K ρ (X, X) n ⊗ W ⊤ W + λ ρ I nq -1 vec(Y ) =γ -1 (I n ⊗ (W (W ⊤ W ) -1 ))vec(Y ) -λ ρ γ -1 (I n ⊗ (W (W ⊤ W ) -1 )) γ 2 K ρ (X, X) n ⊗ W ⊤ W + λ ρ I nq -1 vec(Y ) =γ -1 vec(W (W ⊤ W ) -1 Y ) -λ ρ γ -1 (I n ⊗ (W (W ⊤ W ) -1 )) γ 2 K ρ (X, X) n ⊗ W ⊤ W + λ ρ I nq -1 vec(Y ),
where we use the first item of Lemma A.1 in the last passage. Let us now define
E 2 (ϵ S , λ ρ ; γ, ρ, W ) := -λ ρ γ -1 (I n ⊗ (W (W ⊤ W ) -1 )) γ 2 K ρ (X, X) n ⊗ W ⊤ W + λ ρ I nq -1 vec(Y ) =γ -1 (I n ⊗ (W (W ⊤ W ) -1 )) vec(γW ⊤ H ρ -Y ) -γ 2 K ρ (X, X) n ⊗ W ⊤ W + λ ρ I nq -1 vec γW ⊤ E ρ [∆ a σ(u ⊤ X)] ,
where the second passage follows from ( 30). Using (6) (that we proved above), we have
vec(H ρ ) = vec(γ -1 W (W ⊤ W ) -1 Y ) + E 2 (ϵ S , λ ρ ; γ, ρ, W ) + E 1 (ϵ S , λ ρ ; γ, W ). It remains to upper bound ∥ E 2 (ϵ S , λ ρ ; γ, ρ, W )∥ 2 2 .
To this aim, we write
∥ E 2 (ϵ S , λ ρ ; γ, ρ, W )∥ 2 2 ≤ 2γ -2 σ max (I n ⊗ (W (W ⊤ W ) -1 )) 2 (∥γW ⊤ H ρ -Y ∥ 2 F + λ -2 ρ γ 2 ∥W ⊤ E ρ [∆ a σ(u ⊤ X)]∥ 2 F ) ≤ 2nγ -2 L n (ρ, W ) + 2λ -2 ρ ∥W ⊤ E ρ [∆ a σ(u ⊤ X)]∥ 2 F σ min (W ) 2 ≤ 2nγ -2 L n (ρ, W ) + 2λ -2 ρ σ max (W ) 2 ∥E ρ [∆ a σ(u ⊤ X)]∥ 2 F σ min (W ) 2 .
Plugging in the bound in (32) gives
∥ E 2 (ϵ S , λ ρ ; γ, ρ, W )∥ 2 2 ≤ 2nγ -2 L n (ρ, W ) + 2λ -2 ρ σ max (W ) 2 C 2 1 nϵ 2 S σ min (W ) 2 . (34
)
By combining ( 7) and ( 34) with an application of the triangle inequality, the proof is complete.
this section cite: ['b30']

Section: References
Ref_id:b0 Title: The prevalence of neural collapse in neural multivariate regression Year: (2024)
Ref_id:b1 Title: A mean-field limit for certain deep neural networks Year: (2019)
Ref_id:b2 Title: Deep learning: a statistical viewpoint Year: (2021)
Ref_id:b3 Title: Average gradient outer product as a mechanism for deep neural collapse Year: (2024)
Ref_id:b4 Title: A generalized neural tangent kernel analysis for two-layer neural networks Year: (2020)
Ref_id:b5 Title: Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training Year: (2021)
Ref_id:b6 Title: Modeling from features: a mean-field framework for over-parameterized deep neural networks Year: (2021)
Ref_id:b7 Title: Neural collapse under MSE loss: Proximity to and dynamics on the central path Year: (2022)
Ref_id:b8 Title: Beyond unconstrained features: Neural collapse for shallow neural networks with general data Year: (2024)
Ref_id:b9 Title: Neural collapse for unconstrained feature model under cross-entropy loss with imbalanced data Year: (2024)
Ref_id:b10 Title: Mean-field langevin dynamics and energy landscape of neural networks Year: (2021)
Ref_id:b11 Title: Limitations of neural collapse for understanding generalization in deep learning Year: (2022)
Ref_id:b12 Title: Wide neural networks trained with weight decay provably exhibit neural collapse Year: (2024)
Ref_id:b13 Title: Analysis of a two-layer neural network via displacement convexity Year: (2020)
Ref_id:b14 Title: Transformers learn nonlinear features in context: Nonconvex mean-field dynamics on the attention landscape Year: (2024)
Ref_id:b15 Title: Neural collapse: A review on modelling principles and generalization Year: (2023)
Ref_id:b16 Title: Kernel vs. kernel: Exploring how the data structure affects neural collapse Year: (2024)
Ref_id:b17 Title: A mean field analysis of deep resnet and beyond: Towards provably optimization via overparameterization from depth Year: (2020)
Ref_id:b18 Title: The barron space and the flow-induced function spaces for neural network models Year: (2022)
Ref_id:b19 Title: A mean field view of the landscape of two-layer neural networks Year: (2018)
Ref_id:b20 Title: Mean-field theory of two-layers neural networks: dimension-free bounds and kernel limit Year: (2019)
Ref_id:b21 Title: Neural collapse with unconstrained features. Sampling Theory, Signal Processing, and Data Analysis Year: (2022)
Ref_id:b22 Title: A rigorous framework for the mean field limit of multilayer neural networks Year: (2023)
Ref_id:b23 Title: Global convergence of deep networks with one wide layer followed by pyramidal topology Year: (2020)
Ref_id:b24 Title: Towards understanding neural collapse: The effects of batch normalization and weight decay Year: (2023)
Ref_id:b25 Title: Prevalence of neural collapse during the terminal phase of deep learning training Year: (2020)
Ref_id:b26 Title: Global convergence of three-layer neural networks in the mean field regime Year: (2021)
Ref_id:b27 Title: Neural (tangent kernel) collapse. Advances in Neural Information Processing Systems Year: (2024)
Ref_id:b28 Title: Landscape connectivity and dropout stability of sgd solutions for overparameterized neural networks Year: (2020)
Ref_id:b29 Title: Meanfield analysis of piecewise linear solutions for wide relu networks Year: (2022)
Ref_id:b30 Title: Deep neural collapse is provably optimal for the deep unconstrained features model Year: (2023)
Ref_id:b31 Title: Neural collapse vs. low-rank bias: Is deep neural collapse really optimal? Year: (2024)
Ref_id:b32 Title: Mean-field langevin dynamics: Time-space discretization, stochastic gradient, and variance reduction Year: (2024)
Ref_id:b33 Title: Feature learning via mean-field langevin dynamics: classifying sparse parities and beyond Year: (2024)
Ref_id:b34 Title: Mean-field analysis on twolayer neural networks from a kernel perspective Year: (2024)
Ref_id:b35 Title: Imbalance trouble: Revisiting neural-collapse geometry Year: (2022)
Ref_id:b36 Title: Extended unconstrained features model for exploring deep neural collapse Year: (2022)
Ref_id:b37 Title: High-Dimensional Statistics: A Non-Asymptotic Viewpoint Year: (2019)
Ref_id:b38 Title: On the optimization landscape of neural collapse under mse loss: Global optimality with unconstrained features Year: (2022)
