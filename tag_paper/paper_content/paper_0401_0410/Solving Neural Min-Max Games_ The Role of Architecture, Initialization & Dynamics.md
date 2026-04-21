Title: Solving Neural Min-Max Games: The Role of Architecture, Initialization & Dynamics
Abstract: Many emerging applications-such as adversarial training, AI alignment, and robust optimization-can be framed as zero-sum games between neural nets, with von Neumann-Nash equilibria (NE) capturing the desirable system behavior. While such games often involve non-convex non-concave objectives, empirical evidence shows that simple gradient methods frequently converge, suggesting a hidden geometric structure. In this paper, we provide a theoretical framework that explains this phenomenon through the lens of hidden convexity and overparameterization. We identify sufficient conditions-spanning initialization, training dynamics, and network width-that guarantee global convergence to a NE in a broad class of non-convex min-max games. To our knowledge, this is the first such result for games that involve two-layer neural networks. Technically, our approach is twofold: (a) we derive a novel path-length bound for the alternating gradient descent-ascent scheme in min-max games; and (b) we show that the reduction from a hidden convex-concave geometry to two-sided Polyak-Łojasiewicz (PL) min-max condition hold with high probability under overparameterization, using tools from random matrix theory.

Section: Introduction
At the Nobel Symposium marking the centennial of Game Theory [31,83], a key challenge was posed:
the development of a systematic theory for non-convex games spurred by the rapid growth of deep learning in incentive-aware multi-agent systems [104,130]. Indeed, many influential modern AI systems are built upon the fusion of foundational game-theoretic principles-particularly zero-sum games-with the expressive capacity of neural networks. Notable examples include generative adversarial networks (GANs) [51], robust reinforcement learning [89], adversarial attacks [117], domain-invariant representation learning [44], distributionally robust optimization [77,123], and multi-agent environments featuring natural language interactions, such as AI safety debates between large language models and verifier-prover systems [56,16]. In these settings, the game-theoretic framework provides a natural and interpretable objective-typically an equilibrium solution endowed with strong normative appeal, such as the celebrated von Neumann minimax points [48] and Nash-Rosen equilibria [80,94].
At the same time, much of the remarkable progress at the intersection of deep learning and game theory stems from the capacity of deep models to operate effectively in environments with large, often continuous, state and action spaces. Iconic examples include Go [103], autonomous driving [102], Texas Hold'em poker [15], and real-time strategy games such as StarCraft II through AlphaStar [112].
Tackling such large-scale decision-making problems has necessitated the combination of expressive architectures with function-approximation-based learning, replacing high-dimensional reward/value functions and strategy/policy spaces with trainable surrogates. Hence, these surrogates act as flexible intermediaries, enabling generalization across complex environments without exhaustive enumeration of action spaces.While theoretical focus has largely remained on linear approximators [121,26], it is the nonlinear models-such as kernels and deep neural networks-which in practice dramatically expand representational power [67,58], allowing richer strategic behaviors. Thus, agents' policies are encoded through powerful approximators, and equilibrium learning unfolds through iterative parameter tuning (see Figure 1). sequences. Instead of explicitly constructing and searching the full decision tree, a neural network implicitly encodes both the value of paths and the policy for navigation, learning an effective strategy dynamically without ever uncovering the complete structure of the maze.
Despite the empirical success, algorithms with provable convergence guarantees remain scarce. This is unsurprising given that even in finite games, strong computational hardness results [22,23,32,87] and dynamic impossibility theorems [78,53,52,116,46,47] pose significant barriers. Notably, even in two-player zero-sum games-where classical theory guarantees existence and efficient computation of minimax points via LP duality [19] or optimistic first-order methods [120,5]-these assurances collapse when modern deep-learning architectures, with their inherent non-convexity, are introduced [34,10,6]. Specifically:(i) global solution concepts (e.g., von Neumann minimax, Nash equilibria) may fail to exist; (ii) even when they do, tandem gradient-based methods often suffer from instability, cycling, or divergence, resulting in poor solutions. [33,75,35,113].
Thus, the best hope for mitigating the practical impact of these worst-case hardness results lies in focusing on structured subclasses of games. It remains plausible that broad families of nonconcave games-rich enough to capture multi-agent interactions-admit tractable local or even global equilibria.
Hidden convexity: a promising direction. One compelling approach along this path is the emerging theory of hidden convex games [114,79,115,99,29]. In its simplest form, two players interact via a convex-concave zero-sum game Loss(Player 1 , Player 2 ), but control only high-dimensional parameters θ, ϕ, through mappings Player 1 ← M ap 1 θ (•) and Player 2 ← M ap 2 ϕ (•). These mappings are smooth and known, allowing gradient-based training, but typically not efficiently invertible, reflecting the practical irreversibility of neural architectures. Consequently, while the latent game preserves convex-concave structure, the optimization landscape Loss(M ap 1 θ , M ap 2 ϕ ) over control variables becomes highly non-convex [see 99, p. 26] . Although not every non-convex game admits such a structure, many practical applications naturally fit within this framework (see Appendix B). Rank collapse: the fragility of hidden convexity. A major criticism of the hidden convexity paradigm relies critically on the assumption that the Jacobian of the agents' mappings maintain uniformly bounded singular values throughout training. In practice, such uniform bounds often fail, as real-world architectures may suffer from rank collapse or near-singular behavior during optimization (see, e.g., [101,43,37]), undermining theoretical guarantees. When such degeneracies arise, convergence rates can deteriorate exponentially, and worst-case bounds may become vacuous. Even if Jacobian well-conditioning is achieved by a random initialization, there are no assurances that it will be preserved as training evolves.
These limitations underscore the need for explicit, open-box conditions-beyond abstract hidden mappings-that explain the empirical success of efficient training in large-scale min-max settings. Whilst hidden convexity provides significant insights about these systems, it does not answer a fundamental behavioral question:
Can appropriate architectural design, initialization protocols, and training dynamics jointly ensure efficient convergence in large-scale neural min-max games? (⋆)
this section cite: ['b30', 'b82', 'b103', 'b129', 'b50', 'b88', 'b116', 'b43', 'b76', 'b122', 'b55', 'b15', 'b47', 'b79', 'b93', 'b102', 'b101', 'b14', 'b111', 'b120', 'b25', 'b66', 'b57', 'b21', 'b22', 'b31', 'b86', 'b77', 'b52', 'b51', 'b115', 'b45', 'b46', 'b18', 'b119', 'b4', 'b33', 'b9', 'b5', 'b32', 'b74', 'b34', 'b112', 'b113', 'b78', 'b114', 'b98', 'b28', 'b100', 'b42', 'b36']

Section: Setting and Main Contribution
Motivated by the above challenges, we provide-to the best of our knowledge-the first quantitative convergence guarantees addressing the central question (⋆) under minimal assumptions. Formally, given input datasets D F and D G , and latent strategy spaces S F and S G , we consider the hidden min-max problem min
θ∈R d (F ) θ max ϕ∈R d (G) ϕ L D (F θ , G ϕ ),( )
where F θ : R d (F ) 0 → R dim(S F ) and G ϕ : R d (G) 0 → R dim(S G ) are smooth mappings parameterized by θ and ϕ (e.g., neural network weights). While our results extend beyond, we focus on well-studied [127] separable latent minmax objectives of the form
L D (F, G) = I D F 1 (F ) + I D 2 (F, G) -I D G 3 (G),(1)
where D = (D F , D G ) and I D F 1 , I D G 3 -the individual components-are strongly convex and smooth, and I D 2 -the coupling componentis smooth bilinear. As convergence metric, we adopt the Nash gap (also known as the Nikaido-Isoda duality gap [82]):
DG L D (θ, ϕ) := max ϕ ′ L D (F θ , G ϕ ′ ) -min θ ′ L D (F θ ′ , G ϕ )
, and say that ( θ, φ) is an ϵ-saddle (or ϵ-approximate minimax or Nash equilibrium) if DG L D ( θ, φ) ≤ ϵ.
Remarks. Replacing players' actions with neural nets-i.e., F θ = NN θ (•) and G ϕ = NN ϕ (•)renders the end-to-end landscape highly non-convex, although the latent game L remains convex-concave. The separable structure naturally unifies several hidden zero-sum regimes: when I 2 vanishes, it recovers separable strongly-convex-concave games; when I 1 and I 3 vanish, it reduces to bilinear games [114]; and when both components are present, it captures regularized games (e.g., Tikhonov-or entropy-regularized settings), recently used in hidden min-max frameworks, including team and zero-sum Markov games [59,60]. We discuss concrete examples in Section 2 and Appendix B. In these settings, regularization plays a critical role in stabilizing dynamics and mitigating chaotic behaviors, both empirically ([see 99, p. 26]) and theoretically (cf. [115, pp. 7-8], [59]). Before enumerating our techincal contributions, we highlight a key result addressing (⋆): Informal Theorem (Theorem 3.8). There exists a decentralized, gradient-based method (eq. (Alt-GDA)) that computes, with high probability under suitable Gaussian random initialization, an ϵ-approximate Nash equilibrium for any ϵ > 0 in broad class of hidden convex-concave zero-sum games, where each player's strategy is parameterized by a sufficiently wide two-layer neural network.
• The number of iterations required scales as
O poly 1 width 1 , 1 width 2 , 1 n , d input × L 3 µ 3 × log 1 ϵ ,
where width 1 , width 2 are the hidden layer widths, n is the number of training samples, d input is the input dimension, L is the smoothness constant, and µ is the strong convexity modulus of the latent objective.
• This guarantee holds provided the network width 1,2 = Ω µ 2 n 3 dinput . A converse byproduct: input-optimization games. We also uncover a new convergence guarantee in a related but distinct setting: optimizing directly over inputs when the neural network mappings are fixed. This perspective is motivated both by adversarial example generation through min-max formulations (see Section 2, Appendix B & [117]) and by empirical results of [99] for solving normal form zero-sum games using input-optimization at random fixed neural network mappings-without theoretical justification of non-singularity of spectrum trajectory. Formally, the goal is to find input vectors (x Alice , x Bob ) that implement a Nash equilibrium:
min xAlice∈D F max xBob∈D G L (F θ (x Alice ), G ϕ (x Bob )) . (-1 )
for some convex-concave function L, typically referred as attack's loss [117]. In this regard, we formally establish that Algorithm AltGDA converges to an ϵ-Nash equilibrium with iteration complexity Õ 1 ϵ log 1 ϵ under high-probability guarantees (Theorem 3.5). To the best of our knowledge, this provides the first open-box, provable convergence result for input-optimization attacks based on randomly initialized overparameterized neural networks, matching and theoretically explaining the experimental observations of [99] and [117].
this section cite: ['b126', 'b81', 'b113', 'b58', 'b59', 'b58', 'b116', 'b98', 'b116', 'b98', 'b116']

Section: Challenges and Our Approach: Bridging Overparameterization with Strategic Learning
Back to minimization. The optimization of min-max objectives-especially convex-concave or structured non-convex games-has been extensively studied (for an appetizer see Appendix A.1-A.2 and references therein). However, the dynamics of gradient-based methods in games where players are parameterized by neural networks remain far less understood. In minimization of training loss, a powerful lens for analyzing the success of gradient descent (GD) is the theory of overparameterization and the Neural Tangent Kernel (NTK). In the infinite-width limit, GD converges provided the NTK's smallest eigenvalue remains bounded away from zero. For finite-width networks, convergence proofs typically hinge on two ingredients: (i) good NTK conditioning at initialization, and (ii) negligible drift of the NTK during training [85,27,13,106], ensuring that an underlying Polyak-Łojasiewicz (PŁ) condition is maintained.
Extending to Games: The spectrum path. Even simple hidden zero-sum games, where players are parameterized by two-layer neural networks with smooth activations, can cause vanilla GDA to diverge arbitrarily [114]. Although PŁ-based convergence for minimization has been understood since the classical works of Polyak and Łojasiewicz [90,73], analogous results for min-max optimization have only recently emerged [124,125,60]. More recently, hidden convexity has been shown to imply a PŁ structure-both in minimization [41] and in min-max games [60]. However, this reduction to PŁ-condition reveals a key technical obstacle: hidden convexity alone cannot safeguard convergence if the Jacobians of the players' mappings suffer from near-singularities-i.e., if the least singular value approaches zero. In this regime, the effective PŁ-modulus degenerates, the gradient dominance property and convergence guarantees break down. Thus, the evolution of singular values under the employed learning dynamics becomes central challenge.
In this work, we adopt the alternating gradient descent-ascent (AltGDA) method, which mirrors natural sequential play between agents. From a technical standpoint, alternation proves crucial: simultaneous one-timescale GDA (SimGDA) may diverge both in case of hidden convex-concave games [114] and two-sided-PŁ games [124]. Additionally, alternation has been explored as an acceleration and stabilization tool for min-max optimization [66,128].
• AltGDA Path Length: Hence, our first central technical contributions is a tight control of the path length of AltGDA iterates (Lemma 3.3). We show that AltGDA trajectories remain confined within a bounded region around initialization, preventing severe deterioration of hidden convex-concave structure (e.g., Jacobian conditioning). While path-length bounds are relatively straightforward in minimization-by directly unrolling GD iterations-in min-max problems, the alternating structure introduces significant complications for such ad-hoc analysis. To circumvent this, we employ a carefully designed potential function-a weighted interpolation between the two players' Nash gaps-by [124], which may be of independent interest.
Beyond bounding the trajectory, two additional challenges arise relative to standard supervised learning:
• Output Dimension: In games, neural networks output distributions over actions or more generally higher-dimensional vectors, unlike scalar labels in classification tasks. Estimating the singular value spectrum of such vector-output neural networks is more subtle. To address this, we arrive at Lemma 3.7, by adapting techniques from [106] which essentially combines Hermite expansions of hidden layer outputs, first-order Taylor series expansion and Lipschitzness of Jacobians, and high-probability concentration bounds for random Gaussian matrices.
• Average-Case Analysis of Input Min-Max Games: A similar approach is employed for inputoptimization games, where the roles of inputs and weights are reversed. From a worst-case perspective, there exist constructions leading to rank-deficient Jacobians and failure of GDA due to convergence to spurious local optima [114], our analysis takes an average-case view. Specifically, we show that min-max input attacks, solved via AltGDA, succeed with high probability when the neural network mappings are randomly sampled with Gaussian initializations (Theorem 3.5).
• General Loss Structures: Unlike many prior works, which rely on the non-linear least squares structure of supervised losses to control dynamics [106,69,70], we allow general separable latent objectives combining strongly convex regularizers and bilinear couplings. This more general setting requires significantly stronger control on the optimization trajectory and leads to a fundamentally different overparameterization scaling, namely Ω(n 3 ) compared to Ω(n) in pure minimization settings (Theorem 3.8)
this section cite: ['b84', 'b26', 'b12', 'b105', 'b113', 'b89', 'b72', 'b123', 'b124', 'b59', 'b40', 'b59', 'b113', 'b123', 'b65', 'b127', 'b123', 'b105', 'b113', 'b105', 'b68', 'b69']

Section: Preliminaries
We begin by introducing the standard notions of smoothness and Lipschitz continuity that will be used throughout this work. All norms are taken to be the Euclidean (ℓ 2 ) norm unless otherwise stated.
this section cite: []

Section: Lipschitz Continuity, Smoothness, and Strong Convexity.
Let f : R d → R be a differentiable function. We say that f is L f -Lipschitz continuous and L ∇f -smooth if there exist constants
L f , L ∇f > 0 such that |f (u) -f (v)| ≤ L f ∥u -v∥, ∥∇f (u) -∇f (v)∥ ≤ L ∇f ∥u -v∥, ∀u, v ∈ R d .
Moreover, f is µ-strongly convex if there exists µ > 0 such that
f (v) ≥ f (u) + ⟨∇f (u), v -u⟩ + µ 2 ∥v -u∥ 2 , ∀u, v ∈ R d .
Similarly, for a parametrized mapping (e.g., the neural network) M θ (x) : R d0 → R d2 with parameters θ ∈ R M , we say M θ is β M -smooth (w.r.t. θ) at fixed input x if
σ max (∇ θ M θ (x) -∇ θ M θ ′ (x)) ≤ β M ∥θ -θ ′ ∥, ∀θ, θ ′ ∈ R M ,(2)
where σ max (•) denotes the largest singular value and ∇ θ M θ (x) is the Jacobian of M θ (x) with respect to θ.
this section cite: []

Section: Finite-Sample Parametrized Min-Max Setting.
Then, we unroll the general hidden convex-concave model of ( ) to the finite-sample empirical risk minimization (ERM) setting, assuming access to a (possibly labeled) dataset D = (D F , D G ) = {(x i , y i )} n i=1 of size n. Formally, we consider the following optimization problem:
min θ∈R d (F ) θ max ϕ∈R d (G) ϕ L D (F θ , G ϕ ) := I D F 1 (F θ ) + I (D F ,D G ) 2 (F θ , G ϕ ) -I D G 3 (G ϕ ),(⋄)
where the mappings F θ :
d (F ) 0 → R dim(S F ) and G ϕ : d (G) 0 → R dim(S G )
are smooth functions parametrized by θ and ϕ (e.g., neural networks). The individual components and the bilinear coupling expand as:
• I D F 1 (F θ ) = i∈[|D F |] ℓ i (y i , F θ (x i )), I D G 3 (G ϕ ) = j∈[|D G |] ℓ j (y j , G ϕ (x j )), • I (D F ,D G ) 2 (F θ , G ϕ ) = i∈[|D F |] j∈[|D G |] F θ (x i ) ⊤ A(x i , x j , y i , y j )G ϕ (x j ),
where for each sample pair e ij (x i , x j , y i , y j ), the coupling matrix A(x i , x j , y i , y j ) ∈ R dim(S F )×dim(S G ) encodes interactions between players.
this section cite: []

Section: Blanket Assumptions on the Loss and Coupling Terms.
We impose the following structural assumptions on the loss components and bilinear couplings appearing in the finite-sample min-max objective (⋄). and y ∈ Y, the (latent) gradient of each loss ℓ(y, h) satisfies:
∥∇ h ℓ(y, h)∥ ≤ A 1 ∥h∥ + A 2 diam(Y) + A 3 . Remark 2.
2. Item (i) ensures the applicability of gradient-based methods, while Items (i)-(iii) imply that the overall loss L D is (L L , L ∇L )-smooth and (µ θ , µ ϕ )-hidden-strongly convex-concave, with constants determined by the structure of ℓ and A(•). For standard strongly convex losses (e.g., MSE, logistic loss, cross-entropy with ℓ 2 -regularization), the gradient with respect to the network output is controlled as in Item (iv) by an affine function of the output norm, with the leading coefficient proportional to the strong convexity modulus, A 1 = Θ(µ). 1Neural Network and Training Data Model. Definition 2.3 (Two-layer Neural Network). We consider two-layer neural networks (often referred to as shallow networks). Specifically, such a network h is defined by:
h(x) = Map w=(W1,W2) (x) = W (h) 2 ψ W (h) 1 x , where x ∈ R d (h) 0 , W (h) 1 ∈ R d (h) 1 ×d (h) 0 , W (h) 2 ∈ R d (h) 2 ×d (h)
1 , and ψ : R → R is an activation function applied coordinate-wise. Assumption 2.4 (Properties of the Two-layer Neural Network). We assume: • h(•) is twice-differentiable and β h -smooth with respect to (W
(h) 1 , W (h) 2 ).
• ψ is twice-differentiable with ψ(0) = 0, bounded first and second derivatives ( ψmax , ψmax ), and finite Hermite norm ∥ψ∥ H < ∞ 2 .
• The training data (X,
Y ) ∈ R d (h) 0 ×n × R d (h) 2 ×n satisfies ∥x i ∥ = 1 ∀i ∈ [n] and ∥Y ∥ ≤ 1. • σ max (W (h) 2 ) k = O ψmax ψmax for all k ∈ Z ≥0 3 .
Although we assume the activation function ψ to be twice differentiable-thereby excluding nonsmooth activations such as ReLU-our results naturally extend to smooth approximations like the Gaussian Error Linear Unit (GeLU) [55] and the softplus function [39], which have been shown empirically to perform comparably or even better than ReLU in several settings 4  [12,40]. The performance of gradient-based training depends critically on the geometry of the training data. A standard proxy for data diversity is the well-conditioning of sample matrix X (with input vectors as rows), under standard random designs such as isotropic or sub-Gaussian inputs [86,110,111,95]. Assumption 2.5 (Spectral Properties of the Data Matrix). Let X ∈ R n×d denote the data matrix whose rows x i satisfy ∥x i ∥ 2 = 1 for all i. We assume that the number of samples satisfies n ≥ d, and that X is "generic" in the sense that σ min (X * r ) = Ω(1) and σ max (X) = O n/d 5 , where n is the number of samples and d is the ambient input dimension.
For a fair comparison with the minimization literature, in the main body of the paper we adopt the data genericity assumption. Interested readers can refer to appendix for fine-grained width bounds. 6   Solution concept. Note that while our min-max objective L D (F θ , G ϕ ) is not convex-concave in F θ , G ϕ , it is (strongly) convex-concave in the outputs of F θ and G ϕ , i.e., hidden stronglyconvex-concave. Our analysis leverages precisely this hidden structure. Specifically, [41,Proposition 2] states that if min θ f (θ) wheref (θ) = F (H(θ)) and F is strongly convex while H is a smooth map (e.g., a neural net), then f satisfies the PŁ-condition. Thus, hidden strong convexity implies PŁ-condition, even for nonconvex objectives. Utilizing this along with [30, Proposition of 4.1], we can define PŁ-moduli for our min-max objective in terms of the smallest singular values of the neural network Jacobians: Fact 2.6 (Reduction to Two-Sided PŁ-condition [41,30]). The loss function L D satisfies a two-sided Polyak-Łojasiewicz (PL) condition with parameters µ θ σ 2 min (∇ θ F θ ) and µ ϕ σ 2 min (∇ ϕ G ϕ ), where σ min (•) denotes the smallest singular value of the corresponding Jacobian mappings. This reduction resolves several challenges inherent to general nonconvex-nonconcave min-max problems. First, it unifies several optimality notions-namely, global minimax, saddle point, and gradient stationarity-which, in general settings, need not coincide. For formal definitions see Appendix E. In the case where the objective satisfies a two-sided Polyak-Łojasiewicz (PŁ) condition, these notions become equivalent even at their ϵ-approximate versions. We formalize this via the following lemma: Lemma 2.7 (Lemma 2.1 in [124], Appendix C in [60]). If the objective function f satisfies the two-sided PŁ-condition, then all three notions in Definition E.1 are equivalent:
ϵ -(Saddle Point) ⇐⇒ ϵ -(Global Min-Max) ⇐⇒ ϵ -(Stationary Point) ∀ϵ ≥ 0
Second, as discussed in the introduction, saddle points may not exist in general nonconvex-nonconcave problems. Therefore, we explicitly adopt the following benignfoot_2 assumption: Assumption 2.8 (Existence of Saddle Points). The objective function L(θ, ϕ) admits at least one saddle point. Moreover, for any fixed ϕ, min θ∈R m L(θ, ϕ) has a non-empty solution set and a finite minimum value. Similarly, for any fixed θ, max ϕ∈R n L(θ, ϕ) has a non-empty solution set and a finite maximum value. Examples of hidden neural min-max optimization. Due to space limitations, we defer a comprehensive list of examples and references to Appendix B. To build intuition, we present below two representative bilinear examples that highlight the key structural differences. We broadly distinguish two principal types of ML-driven min-max problems • Network Optimization: Problems where optimization is performed over neural network parameters given a fixed dataset (training over weights). This setting captures tasks such as generative modeling or robust adversarial reinforcement learning.
Example : min
θ max ϕ F θ (x) ⊤ AG ϕ (x ′ ).
• Input Optimization: Problems where network parameters are fixed (e.g., random initialization), and optimization occurs over the input space (e.g., adversarial perturbations). This corresponds to input-driven optimization problems such as adversarial attack design.
Example : min
xAlice∈D F max xBob∈D G F θ (x Alice ) ⊤ AG ϕ (x Bob ) .
this section cite: ['b54', 'b38', 'b11', 'b39', 'b85', 'b109', 'b110', 'b94', 'b40', 'b40', 'b29', 'b123', 'b59']

Section: Our Results
Alternating Gradient Descent-Ascent (AltGDA) proceeds by sequentially updating the parameters of the min-player θ and the max-player ϕ, leveraging the most recent gradient information at each step. The updates take the form:
θ (t) = θ (t-1) -η θ ∇ θ L D (θ (t-1) , ϕ (t-1) ), ϕ (t) = ϕ (t-1) + η ϕ ∇ ϕ L D (θ (t) , ϕ (t-1) ) (AltGDA)
where η θ , η ϕ > 0 denote the respective step sizes.
Our analysis builds upon the framework of Yang, Kiyavash, and He [124], which guarantees log(1/ϵ) convergence under a two-sided PL condition. In our setting, the PL moduli depend on the smallest singular values σ min of the Jacobians ∇ θ F θ and ∇ ϕ G ϕ , which must remain bounded away from zero throughout the optimization trajectory (Fact 2.6). This dependence is critical, as both the PL constants and the step sizes in AltGDA scale inversely with σ min .
Hence, we first establish that, under suitable random initialization and sufficient overparameterization, the initialization satisfies σ min (∇ θ F θ ), σ min (∇ ϕ G ϕ ) ≥ cB with high probability (see Lemmas 3.4 and 3.7). Furthermore, by smoothness of the neural mappings, there exists a Euclidean ball B((θ 0 , ϕ 0 ), R) within which the singular values of the Jacobians remain well-conditioned, i.e., σ min > B > 0. The radius is given by R = µJac 2β , where µ Jac := max µ (F ) Jac , µ (G) Jac and β := min {β F , β G }. This result parallels Lemma 1 of Song et al. [106], adapted here to the alternating min-max setting.
However, the optimization trajectory could, in principle, leave this region. To prevent this, we analyze the path length of AltGDA using Yang, Kiyavash, and He [124]'s Lyapunov potential function, rather than directly unrolling the iterates-which would be analytically cumbersome due to alternation: Definition 3.1 (Lyapunov Potential [124]). For a min-max objective function, L(θ, ϕ), we define the Lyapunov potential at time t as P t = (max ϕ L(θ t , ϕ) -L(θ ⋆ , ϕ ⋆ ))+λ (max ϕ L(θ, ϕ t ) -L(θ t , ϕ t )). (Note that the choice of λ will not affect our conclusions about overparameterization in this paper.) Lemma 3.2 (Theorem 3.2 in [124]). Suppose the min-max objective function L(θ, ϕ) is L ∇L -smooth and satisfies the two-sided PŁ-condition with (µ θ , µ ϕ ). Then if we run AltGDA with η θ = µ 2 ϕ 18L 3 ∇L and η ϕ = 1 L ∇L , then ∥θ t+1 -θ t ∥ + ∥ϕ t+1 -ϕ t ∥ ≤ √ αc t/2 √ P 0 where constants α and c ∈ (0, 1) depend only on L ∇L , µ θ , µ ϕ and P 0 is the Lyapunov potential at time t = 0. (Please refer to Remark E.5 for the exact expressions for α and c.) By way of contradiction, let T denote the first iteration such that (θ T , ϕ T ) / ∈ B((θ 0 , ϕ 0 ), R). We will show that, with high probability, the AltGDA trajectory remains within this ball by proving that its total path length is strictly less than R.
Indeed, AltGDA path length satisfies: ℓ(T ) ≜ T -1 t=0 (∥θ t+1 -θ t ∥ + ∥ϕ t+1 -ϕ t ∥) ≤ √ 2α1 1- √ c • √ P 0 .
Therefore, it suffices to show that √ P 0 ≤ R/2 with high probability. The following lemma provides an upper bound on P 0 in terms of the gradient norms: Lemma 3.3 (Upper Bound on Initial Potential P 0 ). Suppose the min-max objective L(θ, ϕ) is L L -Lipschitz and satisfies a two-sided PŁ condition with constants (µ θ , µ ϕ ). Then the initial Lyapunov
potential P 0 ≤ L L (C 1 • ∥∇ θ L(θ 0 , ϕ 0 )∥ + C 2 • ∥∇ ϕ L(θ 0 , ϕ 0 )∥) , where C 1 , C 2 = Θ L L /µ 3 θ .
It is clear that bounding P 0 requires controlling the gradient norms at initialization, which-in our neural setting-requires bounding both the output norm and the spectral norm σ max of the Jacobian via the chain rule. Lemmas 3.4, G.2 and 3.7 provide these bounds under standard overparameterization and Lipschitz stability conditions. As a result, we obtain P 0 ≤ κR 2 for some constant κ < 1 determined by the network width. Thus, with sufficient overparameterization, the iterates remain confined within the well-conditioned region B((θ 0 , ϕ 0 ), R).
Interestingly, this analysis not only ensures that the iterates stay within a region where the PŁcondition holds, but also reveals a beneficial side effect: since the potential function captures a weighted average of Nash gaps and is monotonically decreasing, a small initial value of P 0 implies that the initialization is already mildly close to equilibrium. Consequently, both convergence and geometric stability are maintained throughout training.
this section cite: ['b123', 'b105', 'b123', 'b123', 'b123']

Section: Input-Optimization Min-Max Games
Here, we consider the input-optimization game between two neural networks F θ and G ϕ in hidden bilinear objective with ℓ 2 -regularization defined as follows for a given payoff matrix, A:
L(θ, ϕ) = F (θ) ⊤ AG(ϕ) + ε 2 ∥F (θ)∥ 2 - ε 2 ∥G(ϕ)∥ 2(3)
This game has been proposed by [114] and experimentally analyzed by [99]. Here, F θ and G ϕ are defined similar to Definition 2.3 as
F (θ) = W (F ) 2 ψ(W (F ) 1 θ) and G(ϕ) = W (G) 2 ψ(W (G)
1 ϕ) but with parameters θ, ϕ as inputs and randomly initalized W
(F ) k ∼ N (0, σ 2 k,F ), W (G) k ∼ N (0, (σ 2 k,G
), k ∈ {1, 2} along with differentiable activation function ψ (e.g. GeLU). Therefore, the partial derivatives w.r.t. θ and ϕ will be as follows:
∇ θ f (θ, ϕ) = (∇ θ F θ ) ⊤ AG(ϕ) + ε(∇ θ F θ ) ⊤ F (θ) ∇ ϕ f (θ, ϕ) = (∇ ϕ G ϕ ) ⊤ A ⊤ F (θ) -ε(∇ ϕ G ϕ ) ⊤ G(ϕ)(4)
Using these and Lemma 3.3, we can now bound P 0 as follows:
=⇒ P 0 ≤ ∥F (θ 0 )∥ • (εL L C 1 σ max (∇ θ F θ0 ) + L L C ′ 2 (1 + λ)σ max (A)σ max (∇ ϕ G ϕ0 )) + ∥G(ϕ 0 )∥ • (L L C 1 σ max (A)σ max (∇ θ F θ0 ) + εC ′ 2 (1 + λ)σ max (∇ ϕ G ϕ0 ))(5)
Since we want to stay within the ball B((θ 0 , ϕ 0 ), R), we can ensure P 0 = κR 2 by controlling each term in Equation ( 5) accordingly that ultimately yields Theorem 3.5. For this, we would additionally need to prove Lemma 3.4 as stated below. (Please see Appendix F for proof). 1 ∥θ∥ 2 ) -0.5 . Then, w.p ≥ 1 -e -Ω(d (F ) 1) , (i) the singular values of the Jacobian ∇ θ F θ are bounded as
σ min (∇ θ F θ ) = Ω σ 1,F • σ 2,F • d (F ) 1 and σ max (∇ θ F θ ) = O σ 1,F • σ 2,F • d (F ) 1 (6) (ii) F (θ) is β F -smooth where β F = Θ σ 2 1,F • σ 2,F • (d (F )
1 ) 3/2 . Theorem 3.5. Consider two neural networks F (θ), G(ϕ) as defined in Lemma 3.4 above. For the regularized hidden bilinear min-max objective L(θ, ϕ) as defined in Equation 3 above, AltGDA reaches ε-saddle point w.p. ≥ 1 -e -Ω(d (F ) 1) if (θ 0 , ϕ 0 ) and standard deviations σ k,F and σ k,G ,
k ∈ {1, 2} are chosen such that σ k,F/G = Θ( poly(1/d1) σmax(A) ):
To our knowledge, this is the first fine-grained result for overparametrized networks that establishes an O(ϵ)-approximate minimax solution for the hidden bilinear setting originally proposed by Vlatakis-Gkaragkounis, Flokas, and Piliouras [114].
this section cite: ['b113', 'b98', 'b113']

Section: Neural-Parameters Min-Max Games
Now we analyse the case of Neural-Parameters Min-Max Games as described in Section 2. In particular, when both players are two-layer neural networks, through Lemma 3.7, with high probability the Jacobians are non-singular for random Gaussian initializations which ensures that the games with such networks will satisfy 2-sided PŁ-condition with high probability. Consequently, given appropriate initialization conditions for the networks (Assumption 3.6, Equation ( 9)), just like in Section 3.1, we can show that AltGDA converges to the saddle point by ensuring P 0 = κR 2 via requiring both the networks to have at least cubic overparameterization (Theorem 3.8).
this section cite: []

Section: Initialization Scheme 3.6 (Random Initialization
). We consider the following initialization scheme for a two-layer neural network, F , as defined in Definition 2.3: [106]). Suppose that a two-layer neural network, F , as defined in Definition 2.3, satisfies Assumption 2.4 and
(W (F ) 1 ) 0 ∼ N (0, σ 2 1,F I) (W (F ) 2 ) 0 ∼ N (0, σ 2 2,F I) (7) Lemma 3.7 (Lemma 3 & Appendix E.1-E.4 in
τ r1 |ψ(a)| ≤ |ψ(τ a)| ≤ τ r2 |ψ(a)|,
respectively for all a, 0 < τ < 1, and some constants r 1 , r 2 . Then w.h.p. the neural network (i) Jacobian has following bounds on its singular values
σ min (∇ θ F θ ) = Ω σ r1 1,F d (F ) 1 and σ max (∇ θ F θ ) = Õ σ r2 1,F n • d (F ) 1 (8) (ii) is β F -smooth with β F = √ 2σ max (X)( ψmax + ψmax χ max ) where χ max = sup V σ max (V ).
σ 1,F • σ 2,F ≲ 1 d (F ) 0 d (F ) 1 and σ 1,G • σ 2,G ≲ 1 d (G) 0 d (G) 1(9)
and suppose that the hidden layer widths d (F ) 1and
d (G) 1
for the two networks F and G satisfy
d (F ) 1 = Ω µ 2 θ n 3 d (F ) 0and
d (G) 1 = Ω µ 2 ϕ n 3 d (G) 0(10)
where the datasets (D F , D G ) for both the players are assumed to be of size n. Then game correspond to an (µ θ , µ ϕ )-HSCSC min-max objective as defined in Equation 1 We refer the reader to Appendix G for the proof of Theorem 3.8, and exact expressions for failure probabilities and various quantities stated in both Lemma 3.7 and Theorem 3.8
this section cite: ['b105']

Section: Conclusion & Future Directions
We provide the first convergence guarantees and overparameterization bounds for alternating gradient methods in input games and hidden (strongly) convex-concave neural games. Our analysis tightly links optimization trajectory control with spectral stability, ensuring convergence to near-equilibrium. If the reader would like to look beyond the technicalities around the non-asymptotic bounds, our proof techniques offer several insights for practitioners:
• Interpretation of σ min and Exploration: The smallest singular value of the network Jacobian, σ min , controls how well the model explores the strategy space. When σ min ≈ 0, certain strategies remain unexplored, indicating convergence to spurious subspaces. Our analysis ties this directly to the degree of overparameterization.
• Data Geometry and Regions of Attraction: Our results show that overparameterized networks initialized with sufficiently diverse data are more likely to fall into regions where σ min > 0, ensuring stable convergence under AltGDA. While computing σ min per iteration is impractical, the connection offers design insights for data and architecture. Going beyond the neural networks and training regimes considered in this paper (see Table 1 for a summary) is an important future direction. Among these, the assumption on AltGDA is arguably the most benign. In non-convex/non-concave min-max optimization, stabilization is essential. In practice, double-loop methods (e.g., approximate best-response oracles) are often used for safety, while AltGDA serves as a more parallelizable and simpler single-loop alternative. Similarly, the Gaussian initialization is closely aligned with popular schemes like He or Xavier. The main gap lies in the architecture: practical models are often very deep with fixed-width layers. While recent work has begun to explore overparameterization in deep networks for minimization tasks, our paper focuses on a more analytically tractable setting -explicitly avoiding the NTK regime to provide a non-asymptotic analysis for 1-hidden-layer networks in a game-theoretic context. We view relaxing and extending these assumptions as a promising direction for future work.
Another natural next step is to understand how these techniques extend to non-differentiable activation functions (such as ReLU) or scale to multi-player and non-zero-sum settings -especially in structured environments like polyhedral games, which share connections with extensive-form games. For instance, exploring the analogy between two-sided PŁ-conditions (for two-player games) and hypomonotonicity in multi-agent operator theory may allow us to transfer and generalize some of the intuition and techniques from our current setting. We hope our work and these possible future directions open up rich and technically deep avenues for developing gradient-based methods tailored for structured, non-monotone multiplayer games. (See also Appendix I.)
this section cite: []

Section: References
Ref_id:b0 Title: Last-iterate convergence rates for min-max optimization Year: (2019)
Ref_id:b1 Title: Scaling adversarial training to large perturbation bounds Year: (2022)
Ref_id:b2 Title: A convergence theory for deep learning via over-parameterization Year: ()
Ref_id:b3 Title: Concrete problems in AI safety Year: (2016)
Ref_id:b4 Title: Convergence of log(1/ϵ) for Gradient-Based Algorithms in Zero-Sum Games without the Condition Number: A Smoothed Analysis Year: (2024)
Ref_id:b5 Title: The Complexity of Symmetric Equilibria in Min-Max Optimization and Team Zero-Sum Games Year: (2025)
Ref_id:b6 Title: Finite regret and cycles with fixed step-size via alternating gradient descent-ascent Year: ()
Ref_id:b7 Title: Reinforcement learning with general utilities: Simpler variance reduction and large state-action space Year: ()
Ref_id:b8 Title: Hidden convexity in some nonconvex quadratically constrained quadratic programming Year: (1996)
Ref_id:b9 Title: On the Role of Constraints in the Complexity of Min-Max Optimization Year: (2024)
Ref_id:b10 Title: Descent with Misaligned Gradients and Applications to Hidden Convexity Year: ()
Ref_id:b11 Title: SMU: smooth activation function for deep networks using smoothing maximum technique Year: (2021)
Ref_id:b12 Title: Memorization and optimization in deep neural networks with minimum over-parameterization Year: (2022)
Ref_id:b13 Title: Large Scale GAN Training for High Fidelity Natural Image Synthesis Year: (2019)
Ref_id:b14 Title: Solving imperfect-information games via discounted regret minimization Year: (2019)
Ref_id:b15 Title: Scalable AI safety via doublyefficient debate Year: (2023)
Ref_id:b16 Title: A universal law of robustness via isoperimetry Year: (2021)
Ref_id:b17 Title: On the global convergence of imitation learning: A case for linear quadratic regulator Year: (2019)
Ref_id:b18 Title: Zero-sum polymatrix games: A generalization of minmax Year: (2016)
Ref_id:b19 Title: Faster stochastic algorithms for minimax optimization under polyak-{\L} ojasiewicz condition Year: (2022)
Ref_id:b20 Title: Robust optimization for non-convex objectives Year: (2017)
Ref_id:b21 Title: 3-Nash is PPAD-complete Year: (2005)
Ref_id:b22 Title: Settling the complexity of computing twoplayer Nash equilibria Year: (2009)
Ref_id:b23 Title: Efficient algorithms for a class of stochastic hidden convex optimization and its applications in network revenue management Year: (2024)
Ref_id:b24 Title: Over-parameterization and Adversarial Robustness in Neural Networks: An Overview and Empirical Analysis Year: (2024)
Ref_id:b25 Title: Almost optimal algorithms for two-player markov games with linear function approximation Year: (2021)
Ref_id:b26 Title: On lazy training in differentiable programming Year: (2019)
Ref_id:b27 Title: Cooperating with machines Year: (2018)
Ref_id:b28 Title: Solving hidden monotone variational inequalities with surrogate losses Year: (2024)
Ref_id:b29 Title: Solving hidden monotone variational inequalities with surrogate losses Year: ()
Ref_id:b30 Title: Non-concave games: A challenge for game theory's next 100 years Year: (2022)
Ref_id:b31 Title: The complexity of computing a Nash equilibrium Year: (2009)
Ref_id:b32 Title: The limit points of (optimistic) gradient descent in min-max optimization Year: (2018)
Ref_id:b33 Title: The complexity of constrained min-max optimization Year: (2021)
Ref_id:b34 Title: Training gans with optimism Year: (2017)
Ref_id:b35 Title: Some estimates in minimax problems Year: (1972)
Ref_id:b36 Title: Attention is not all you need: pure attention loses rank doubly exponentially with depth Year: (2021-07-24)
Ref_id:b37 Title: Gradient descent finds global minima of deep neural networks Year: ()
Ref_id:b38 Title: Incorporating second-order functional knowledge for better option pricing Year: (2000)
Ref_id:b39 Title: Sigmoid-weighted linear units for neural network function approximation in reinforcement learning Year: (2018)
Ref_id:b40 Title: Stochastic optimization under hidden convexity Year: (2023)
Ref_id:b41 Title: Supply and demand functions in inventory models Year: (2018)
Ref_id:b42 Title: Rank Diminishing in Deep Neural Networks Year: (2022)
Ref_id:b43 Title: Domain-adversarial training of neural networks Year: (2016)
Ref_id:b44 Title: Convergence of adversarial training in overparametrized neural networks Year: (2019)
Ref_id:b45 Title: Survival of the strictest: Stable and unstable equilibria under regularized learning with partial information Year: ()
Ref_id:b46 Title: On the rate of convergence of regularized learning in games: From bandits and uncertainty to optimism and beyond Year: (2021)
Ref_id:b47 Title: A limited-capacity minimax theorem for non-convex games or: How i learned to stop worrying about mixed-nash and love neural nets Year: ()
Ref_id:b48 Title: A variational inequality perspective on generative adversarial networks Year: (2018)
Ref_id:b49 Title: Negative momentum for improved game dynamics Year: ()
Ref_id:b50 Title: Generative adversarial nets Year: (2014)
Ref_id:b51 Title: Stochastic uncoupled dynamics and Nash equilibrium Year: (2006)
Ref_id:b52 Title: Uncoupled dynamics do not lead to Nash equilibrium Year: (2003)
Ref_id:b53 Title: Provably Efficient Maximum Entropy Exploration Year: (2019)
Ref_id:b54 Title: Gaussian error linear units (gelus) Year: (2016)
Ref_id:b55 Title: AI safety via debate Year: (2018)
Ref_id:b56 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b57 Title: The power of exploiter: Provable multi-agent rl in large state spaces Year: ()
Ref_id:b58 Title: Learning Equilibria in Adversarial Team Markov Games: A Nonconvex-Hidden-Concave Min-Max Optimization Problem Year: ()
Ref_id:b59 Title: Solving Zero-Sum Convex Markov Games Year: (2025)
Ref_id:b60 Title: Progressive Growing of GANs for Improved Quality, Stability, and Variation Year: (2018)
Ref_id:b61 Title: PokéChamp: an Expert-level Minimax Language Agent Year: (2025)
Ref_id:b62 Title: Normalizing flows: An introduction and review of current methods Year: (2020)
Ref_id:b63 Title: The extragradient method for finding saddle points and other problems Year: (1976)
Ref_id:b64 Title: Gradient-Type Methods For Decentralized Optimization Problems With Polyak-{\L} ojasiewicz Condition Over Time-Varying Networks Year: (2022)
Ref_id:b65 Title: Fundamental benefit of alternating updates in minimax optimization Year: (2024)
Ref_id:b66 Title: Learning two-player markov games: Neural function approximation and correlated equilibrium Year: (2022)
Ref_id:b67 Title: Solving weakly-convex-weakly-concave saddle-point problems as successive strongly monotone variational inequalities Year: (2018)
Ref_id:b68 Title: Loss landscapes and optimization in overparameterized non-linear systems and neural networks Year: (2022)
Ref_id:b69 Title: On the linearity of large non-linear models: when and why the tangent kernel is constant Year: (2020)
Ref_id:b70 Title: Convergence Analysis of Randomized SGDA under NC-PL Condition for Stochastic Minimax Optimization Problems Year: (2023)
Ref_id:b71 Title: On the computational efficiency of training neural networks Year: (2014)
Ref_id:b72 Title: Une propriété topologique des sous-ensembles analytiques réels Year: (1963)
Ref_id:b73 Title: Régularisation d'inéquations variationnelles par approximations successives Year: (1970)
Ref_id:b74 Title: Cycles in adversarial regularized learning Year: (2018)
Ref_id:b75 Title: The numerics of gans Year: (2017)
Ref_id:b76 Title: Modeling the Second Player in Distributionally Robust Optimization Year: (2021)
Ref_id:b77 Title: An impossibility theorem in game dynamics Year: (2023)
Ref_id:b78 Title: Generalized natural gradient flows in hidden convex-concave games and gans Year: (2021)
Ref_id:b79 Title: Non-cooperative games Year: (2024)
Ref_id:b80 Title: Cubic regularization of Newton method and its global performance Year: (2006)
Ref_id:b81 Title: Note on non-cooperative convex games Year: (1955)
Ref_id:b82 Title: One Hundred Years of Game Theory Year: (2021-12)
Ref_id:b83 Title: Solving a class of non-convex min-max games using iterative first order methods Year: (2019)
Ref_id:b84 Title: Overparameterized nonlinear learning: Gradient descent takes the shortest path? Year: ()
Ref_id:b85 Title: Toward moderate overparameterization: Global convergence guarantees for training shallow neural networks Year: (2020)
Ref_id:b86 Title: The computational complexity of multi-player concave games and Kakutani fixed points Year: (2022)
Ref_id:b87 Title: The Effect of Model Size on Worst-Group Generalization Year: (2021)
Ref_id:b88 Title: Robust adversarial reinforcement learning Year: ()
Ref_id:b89 Title: Gradient methods for minimizing functionals Year: (1963)
Ref_id:b90 Title: A modification of the Arrow-Hurwitz method of search for saddle points Year: (1980)
Ref_id:b91 Title: Robust optimization over multiple domains Year: (2019)
Ref_id:b92 Title: Monotone operators and the proximal point algorithm Year: (1976)
Ref_id:b93 Title: Existence and uniqueness of equilibrium points for concave n-person games Year: (1965)
Ref_id:b94 Title: Smallest singular value of a random rectangular matrix Year: (2009)
Ref_id:b95 Title: Research priorities for robust and beneficial artificial intelligence Year: (2015)
Ref_id:b96 Title: Depth-width tradeoffs in approximating natural functions with neural networks Year: ()
Ref_id:b97 Title: On the quality of the initial basin in overspecified neural networks Year: ()
Ref_id:b98 Title: Exploiting hidden structures in non-convex games for convergence to Nash equilibrium Year: (2023)
Ref_id:b99 Title: Stylegan-xl: Scaling stylegan to large diverse datasets Year: (2022)
Ref_id:b100 Title: The singular values of convolutional layers Year: (2018)
Ref_id:b101 Title: Safe, multi-agent, reinforcement learning for autonomous driving Year: (2016)
Ref_id:b102 Title: Mastering the game of go without human knowledge Year: (2017)
Ref_id:b103 Title: Collaborative machine learning with incentive-aware model rewards Year: ()
Ref_id:b104 Title: Certifiable Distributional Robustness with Principled Adversarial Training Year: (2017)
Ref_id:b105 Title: Subquadratic overparameterization for shallow neural networks Year: (2021)
Ref_id:b106 Title: Indefinite trust region subproblems and nonsymmetric eigenvalue perturbations Year: (1995)
Ref_id:b107 Title: A convergent O (n) algorithm for off-policy temporal-difference learning with linear function approximation Year: (2008)
Ref_id:b108 Title: Efficient algorithms for smooth minimax optimization Year: (2019)
Ref_id:b109 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b110 Title: Spectral norm of products of random and deterministic matrices Year: (2011)
Ref_id:b111 Title: Grandmaster level in StarCraft II using multi-agent reinforcement learning Year: (2019)
Ref_id:b112 Title: Chaos persists in large-scale multi-agent learning despite adaptive learning rates Year: (2023)
Ref_id:b113 Title: Poincaré recurrence, cycles and spurious equilibria in gradient-descent-ascent for non-convex non-concave zero-sum games Year: (2019)
Ref_id:b114 Title: Solving min-max optimization with hidden structure via gradient descent ascent Year: (2021)
Ref_id:b115 Title: No-regret learning and mixed nash equilibria: They do not mix Year: (2020)
Ref_id:b116 Title: Adversarial attack generation empowered by min-max optimization Year: (2021)
Ref_id:b117 Title: What are the statistical limits of offline RL with linear function approximation Year: (2020)
Ref_id:b118 Title: The hidden convex optimization landscape of regularized two-layer relu networks: an exact characterization of optimal solutions Year: (2021)
Ref_id:b119 Title: Linear last-iterate convergence in constrained saddle-point optimization Year: (2020)
Ref_id:b120 Title: Learning zero-sum simultaneous-move markov games using function approximation and correlated equilibrium Year: ()
Ref_id:b121 Title: Zeroth-Order Alternating Gradient Descent Ascent Algorithms for A Class of Nonconvex-Nonconcave Minimax Problems Year: (2023)
Ref_id:b122 Title: Distributionally Robust Performative Prediction Year: (2024)
Ref_id:b123 Title: Global convergence and variance reduction for a class of nonconvex-nonconcave minimax problems Year: (2020)
Ref_id:b124 Title: Faster single-loop algorithms for minimax optimization without strong concavity Year: ()
Ref_id:b125 Title: Policy-based primal-dual methods for convex constrained Markov decision processes Year: (2023)
Ref_id:b126 Title: Optimal extragradient-based algorithms for stochastic variational inequalities with separable structure Year: (2023)
Ref_id:b127 Title: Near-optimal local convergence of alternating gradient descentascent for minimax optimization Year: ()
Ref_id:b128 Title: Variational policy gradient method for reinforcement learning with general utilities Year: (2020)
Ref_id:b129 Title: Multi-agent reinforcement learning: A selective overview of theories and algorithms Year: (2021)
Ref_id:b130 Title: On the convergence rate of stochastic mirror descent for nonsmooth nonconvex optimization Year: (2018)
Ref_id:b131 Title: Over-parameterized adversarial training: An analysis overcoming the curse of dimensionality Year: (2020)
