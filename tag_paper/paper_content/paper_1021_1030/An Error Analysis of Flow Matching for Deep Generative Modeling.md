Title: An Error Analysis of Flow Matching for Deep Generative Modeling
Abstract: Continuous Normalizing Flows (CNFs) have proven to be a highly efficient technique for generative modeling of complex data since the introduction of Flow Matching (FM). The core of FM is to learn the constructed velocity fields of CNFs through deep least squares regression. Despite its empirical effectiveness, theoretical investigations of FM remain limited. In this paper, we present the first end-to-end error analysis of CNFs built upon FM. Our analysis shows that for general target distributions with bounded support, the generated distribution of FM is guaranteed to converge to the target distribution in the sense of the Wasserstein-2 distance. Furthermore, the convergence rate is significantly improved under an additional mild Lipschitz condition of the target score function.

Section: Introduction
Contemporary generative models have primarily been designed around the construction of a map between two probability distributions that transform samples from the prior distribution to the target distribution. The roots of transportbased sampling and density estimation can be traced back to maximum entropy methods for Gaussianizing data (Tabak & Turner, 2013;Tabak & Vanden-Eijnden, 2010). Normalizing Flows (NFs) provide a neural network implementation of these methods by imposing a structured transformation to make the change of measure tractable in discrete, sequential steps (Dinh et al., 2017;Durkan et al., 2019;Huang et al., 2018;Papamakarios et al., 2017;Rezende & Mohamed, 2015). Continuous Normalizing Flows (CNFs) extend this idea to a continuous-time setting by viewing the map T (x) = X t (x) as the solution of an ordinary differ-ential equation (ODE) (Chen et al., 2018;Grathwohl et al., 2019). However, training neural ODEs at scale is intractable, as it requires simulating the ODE. The introduction of Flow Matching (FM) has made CNFs highly efficient for generative modeling of complex data (Karras et al., 2022;Liu et al., 2023;Albergo & Vanden-Eijnden, 2023;Lipman et al., 2023;Neklyudov et al., 2022;Tong et al., 2023;Chen & Lipman, 2023;Albergo et al., 2023;Shi et al., 2023;De Bortoli et al., 2021).
The success of FM motivates a line of research investigating the generation quality guarantees from the perspective of sampling (Albergo & Vanden-Eijnden, 2023;Albergo et al., 2023;Lu et al., 2022;Chen et al., 2023c). These works assume the underlying velocity field is accurately estimated up to a small error under L 2 -norm and provide generation quality guarantees. However, two issues remain unsolved in these works. The first is to provide guarantees for learning the velocity field of the underlying ODE. The second is to relax the strong assumptions on the underlying velocity field, which may be hard to check. This paper takes a step forward by providing an end-to-end analysisfoot_0 of the deep generative modeling based on FM under mild assumptions. Our main contributions are summarized as follows:
• We provide the first end-to-end analysis for the deep generative models based on FM.
• We prove that the deep generative models built upon FM are guaranteed to converge to the target distribution under mild assumption. Furthermore, the convergence rate gets significantly improved under an additional Lipschitz condition of the target score function.
this section cite: ['b58', 'b59', 'b20', 'b21', 'b32', 'b48', 'b50', 'b12', 'b29', 'b35', 'b41', 'b38', 'b46', 'b60', 'b54', 'b18', 'b42']

Section: Assumptions
Assumption 1.1 (Bounded support). The target distribution π 1 is supported on [0, 1] d .
Assumption 1.2 (Lipschitz score). Let π 1 (dx) =
1 End-to-end learning in generative modeling involves using finite samples from the target distribution as input to learn the underlying distribution, and then generating samples from the learned distribution as output. The goal of end-to-end analysis is to provide guarantees for the accuracy of the learned distribution based on the finite input samples, enabling more reliable generative modeling.
e -V (x) dx. Moreover, the potential V (x) is twice continuously differentiable and satisfies -αI ⪯ ∇ 2 V (x) ⪯ αI with α > 1. Lemma 1.3. Suppose that Assumption 1.1 holds. Then v * (x, t) is ξ-Lipschitz continuous w.r.t. x on R d × [0, T ], where ξ ≤ max 1 1-T , T d
(1-T ) 3 . Further, if 1 2 < T < 1, we have v * is d
(1-T ) 3 -Lipschitz continuous w.r.t. x. Lemma 1.4. Suppose that Assumption 1.1 and Assumption 1.2 hold. Then v * (x, t) is ζ(α, d)-Lipschitz continuous on R d ×[0, 1] w.r.t. x, where ζ(α, d) = d 2 α + α + 2 d 2 scales polynomially with α and d. Remark 1.5. Previous work simply assumes the score function or velocity field to be Lipschitz continuous w.r.t. x for every t (Chen et al., 2023c;a). In this paper, we follow Wibisono & Jog (2018a;b); Mikulincer & Shenfeld (2021;2022); Chewi & Pooladian (2022); Gao et al. (2024) to provide the Lipschitz continuity of the velocity field from the properties of the target distribution.
The proofs in this section are deferred to Appendix D.5.
this section cite: ['b44', 'b39', 'b15', 'b23']

Section: Main Results
All proofs of this section is deferred to Appendix C.3. Theorem 1.6 (Consistency). Suppose Assumption 1.1 holds.
Given n samples from target distribution π 1 and the networks as in Theorem 4.4, with parameter ζ replaced by d (1-T ) 3 , we use the estimated velocity field in (11), to generate samples and choose the maximal step size max k=0,1...,N -1 |t k+1 -t k | = O(n -1 d+5 ) and early stopping time T (n) = 1 -(log n) -1/6 , we have W 2 ( π T (n) , π 1 ) → 0, in probability, where π T (n) denotes the generated distribution at time T (n).
The consistency of FM is mainly based on a mild assumption, i.e. boundedness, which justifies the use of CNFs based on FM. Theorem 1.7 (Improved convergence rate). Suppose Assumption 1.1 and Assumption 1.2 hold. Given n samples from target distribution π 1 and the networks as in Theorem 4.4, with parameter ζ replaced by ζ(α, d) defined in Lemma 1.4, we use the estimated velocity field in (11) to generate samples and choose the maximal step size max k=0,1...,N -1 |t k+1 -t k | = O(n -4 3(d+5) ) and early stopping time T (n) = 1 -n -1 3 (d+5) . Then, with probability of at least 1 -1 n , we have
W 2 ( π T (n) , π 1 ) = O n -1 3(d+5)
, where π T (n) denotes the generated distribution at time T (n).
This result highlights the effectiveness of CNFs based on FM in learning the underlying smooth distribution.
this section cite: []

Section: Related Work
Continuous Normalizing Flows CNFs are proposed by viewing the map T (x) = X t (x) as the solution of an ODE. It is not until the introduction of FM that CNFs have grown to be an efficient method for the generative modeling of complex data (Karras et al., 2022;Liu et al., 2023;Albergo & Vanden-Eijnden, 2023;Lipman et al., 2023;Neklyudov et al., 2022;Tong et al., 2023;Chen & Lipman, 2023;Albergo et al., 2023;Shi et al., 2023;De Bortoli et al., 2021).
The key idea of FM is to learn the constructed velocity fields of CNFs through deep least squares regression. In (Liu et al., 2023), a linear interpolant is proposed with a focus on straight paths. This is employed as a step towards rectifying the transport paths (Liu, 2022) through a procedure which improves sampling efficiency. In (Lipman et al., 2023), the interpolant picture is assembled from the perspective of conditional probability paths connecting to a Gaussian, where a noise convolution is used to improve the learning, at the cost of biasing the method. The paper (Tong et al., 2023) introduces a novel simulation-free objective for learning continuous-time flows conditioned on a general distribution. Further, the authors have shown that lifting the static optimal transport problem to the dynamic setting leads to more efficient training and inference of flow models by lowering the variance of the objective and simplifying flows. FM is extended to the Riemannian setting by Chen & Lipman (2023). Another line of work points out that the probability path of CNFs encompasses that of the Diffusion Models (DMs) (Albergo et al., 2023;Lipman et al., 2023;Albergo & Vanden-Eijnden, 2023). If made to match the performance of their stochastic counterparts, ODE-based methods exhibit a number of desirable characteristics that are absent for SDEs, such as an exact, computationally tractable formula for the likelihood and easy application of well-developed adaptive integration schemes for sampling. Further, one of the most successful techniques of accelerating continuous time process-based sampling, distillation (Liu et al., 2023;Song et al., 2023;Salimans & Ho, 2022;Zheng et al., 2022;Luhman & Luhman, 2021), requires deterministic samplers.
this section cite: ['b35', 'b41', 'b38', 'b46', 'b60', 'b54', 'b18', 'b41', 'b39', 'b38', 'b60', 'b38', 'b41', 'b57', 'b51', 'b68', 'b43']

Section: Lipschitz Score v.s. Lipschitz Velocity Field
In analyzing the convergence of DMs and ODE-based models, the assumption of Lipschitz continuity for the score function or the velocity field has been widely used in previous works (Chen et al., 2023c;Lu et al., 2022;Albergo & Vanden-Eijnden, 2023;Chen et al., 2023a). However, these works simply assume the Lipschitzness. In contrast, our paper takes a step forward and rigorously proves that the velocity field is Lipschitz continuous under mild assumptions on the target distribution. By doing so, we provide a stronger theoretical foundation for the application of CNFs based on FM, and help to bridge the gap between theory and practice.
this section cite: ['b42']

Section: Analysis of ODE-based Models
Significant recent works (Albergo et al., 2023;Chen et al., 2023c;Lu et al., 2022) have put effort into controlling the KL divergence between the generated distribution and the target distribution. These studies have demonstrated that simply regressing the velocity field is insufficient to control the likelihood with ODEbased models. Instead, more advanced learning schemes are required to ensure that the Fisher divergence is kept under control. The work (Albergo & Vanden-Eijnden, 2023) has shown that the Wasserstein-2 distance between the generated distribution and the target distribution can be controlled by the objective of regressing the velocity field, assuming the estimated velocity field is Lipschitz continuous. In our paper, we take a different approach, demonstrating that the true velocity field can be well approximated by a Lipschitz neural network. We compare our work with concurrent analyses for ODE-based models in Table 1 where U (t; δ 1 , δ 2 , δ 3 , C, q) in the third row is an increasing function for δ 1 , δ 2 and δ 3 (Lu et al., 2022), where δ i is an upper bound for the score matching objective of order i, i = 1, 2, 3.
this section cite: ['b42', 'b42']

Section: Preliminaries
Notations We denote [N ] := {0, • • • , N -1}. For matrix A and B, we say A ⪯ B, if B -A is positive semi-definite. We denote the identity matrix in R d×d by I d . For a vector x ∈ R d , we define x ⊗2 := xx T . We denote the ℓ 2 -norm of a vector x by ∥x∥ := d i=1 x 2 i . We define the operator norm of a matrix A as ∥A∥ op := sup ∥x∥≤1 ∥Ax∥. For a twice continuously differentiable function f : R d → R, let ∇f, ∇ 2 f , and ∆f denote its gradient, Hessian, and Laplacian, respectively. For a probability density function π and a measurable function f : R d → R, we define the
L 2 (π)-norm of f as ∥f ∥ L 2 (π) := (f (x)) 2 π(x)dx 1/2 . We define L ∞ (K)-norm as ∥f ∥ L ∞ (K) := sup x∈K |f (x)|. For a vector function v : R d → R d , we define its L 2 (π)-norm as ∥v∥ L 2 (π) := ∥∥v∥∥ L 2 (π) and L ∞ (K)- norm as ∥v∥ L ∞ (K) := ∥∥v∥∥ L ∞ (K) . We use the asymp- totic notation f (x) = O(g(x)
) to denote the statement that f (x) ≤ Cg(x) for some constant C > 0 and O(•) to ignore the logarithm. Given two distributions µ and ν, the Wasserstein-2 distance is defined as W 2 (µ, ν) := inf π∈Π(µ,ν) E (x,y)∼π [∥x -y∥ 2 ] 1/2 , where Π(µ, ν) is the set of all couplings of µ and ν. A coupling is a joint distribution on R d × R d whose marginals are µ and ν on first and second factors, respectively.
Flow Matching Given independent empirical observations of X 0 ∼ π 0 and X 1 ∼ π 1 , we want to find an ordinary differential equation (ODE) on time t ∈ [0, 1],
dZ t = v(Z t , t)dt,(1)
which converts Z 0 from π 0 to Z 1 following π 1 . A line of research (Liu et al., 2023;Liu, 2022;Albergo & Vanden-Eijnden, 2023;Lipman et al., 2023;Neklyudov et al., 2022;Wu et al., 2022;Lee et al., 2023b;Tong et al., 2023;Chen & Lipman, 2023;Albergo et al., 2023;Shi et al., 2023) points out that, the vector field can be found by solving a least square regression problem:
min v L 0 (v) := 1 0 E X0,X1 ∥(X 1 -X 0 ) -v(X t , t)∥ 2 dt, with X t = tX 1 + (1 -t)X 0 ,(2)
where X 0 ∼ π 0 , X 1 ∼ π 1 , and X t is the linear interpolation between X 0 and X 1 . The exact minimum of (2) is achieved by
v * (x, t) = E [X 1 -X 0 |X t = x] .(3)
Velocity Field Approximation In practice, the velocity field v * is approximated by neural networks. To avoid instability, we often clip the integral interval [0, 1] with T . Namely, we consider the following loss function:
min v L(v) := 1 T T 0 E X0,X1 ∥(X 1 -X 0 )-v(X t , t)∥ 2 dt, with X t = tX 1 + (1 -t)X 0 , (4
) Given a family of neural networks NN, we consider the following approximation error,
inf v∈NN T 0 ∥v(•, t)-v * (•, t)∥ 2 L 2 (πt) dt = inf v∈NN L(v)-L(v * ),(5)
where π t is the probability distribution of X t defined in (2). The equivalence in (5) is deferred to Lemma 4.1. We also consider the best approximator in the neural networks
v ∈ argmin v∈NN L(v).(6)
We organize the remaining sections as follows: In Section 3, we show that the true velocity field can be well approximated by a Lipschitz neural network. Section 4 establishes that the optimal neural network can be efficiently estimated. Finally, in Section 5, we analyze the error of distribution recovery using the estimated velocity field.
this section cite: ['b41', 'b39', 'b38', 'b46', 'b66', 'b60', 'b54']

Section: Approximation
In practice, the true velocity field is approximated by neural networks. To ensure effective learning, the network class should be expressive enough to approximate the true velocity field.
this section cite: []

Section: Main Assumptions End-to-end Analysis
Theoretical Results (Albergo & Vanden-Eijnden, 2023)
v is K-Lipschitz in x uniformly on (t, x) ∈ [0, 1] × R d W 2 2 (ρ 1 , ρ 1 ) ≤ e 1+2 K H(v) (Chen et al., 2023c) ∇ ln q ← t (x) is L sc,t -Lipschitz in x and satisfies ∥∇ ln q ← t q ← s (x)∥ ≤ β|t -s| c (1 + ∥x∥ + ∥∇q ← t (x)∥)
KL( p∥q) ≤ ϵ provided ℓ ≥ C 1 and ℓh ≤ C -1 2 , where C 1 and C 2 depends polynomially on parameters in assumptions (Lu et al., 2022)
∥∇ 2 x log p ODE (x t )∥ 2 ≤ C, ∇ log q t is C-Lipschitz, uniformly for t D F (q t ∥p ODE t ) ≤ U (t; δ 1 , δ 2 , δ 3 , C, q)
this section cite: ['b42']

Section: Ours
Bounded support Consistency Bounded support and Lipschitzness of the target score functions
W 2 ( π T (n) , π 1 ) = O n -1 3(d+5)
Table 1. Comparison of existing theoretical results on ODE-based models.
Neural Network Structure We configure the ReLU network v θ in the following way.
NN(L, M, J, K, κ, γ 1 , γ 2 ) = v(x, t) = (W L σ(•) + b L ) • (W L-1 σ(•) + b L-1 ) • • • • • (W 1 σ(•) + b 1 )([x T , t] T ) : network width bounded by M, sup x,t ∥v(x, t)∥ ≤ K, max{∥b i ∥ ∞ , ∥W i ∥ ∞ } ≤ κ for i = 1, • • • , L, L i=1 (∥W i ∥ 0 + ∥b i ∥ 0 ) ≤ J, ∥v(x 1 , t) -v(x 2 , t)∥ ≤ γ 1 ∥x 1 -x 2 ∥ for any t ∈ [0, T ], ∥v(x, t 1 ) -v(x, t 2 )∥ ≤ γ 2 ∥t 1 -t 2 ∥ for any x ,
where the network width refers to the maximum dimensions of the weight matrices, σ is the ReLU activation, and ∥ • ∥ ∞ and ∥ • ∥ 0 denote the maximum magnitude of entries and the number of nonzero entries, respectively. In the sequel, we write the neural network class as NN for brevity.
L = O d + log 1 ε , M = O d 3/2 (log(d/ε)) d+1 2 (1 -T ) 4 ζ d ε -(d+1) , J = O d 3/2 (log(d/ε)) d+1 2 (1 -T ) 4 ζ d ε -(d+1) log 1 ε + d , K = O   d log d ε 1 -T   , κ = O ζ log(d/ε) ∨ d 3 log(d/ε) (1 -T ) 4 , γ 1 = 10dζ, γ 2 = O d 3 log(d/ε) (1 -T ) 4 .
There exists an v θ ∈ NN, such that for any t ∈ [0, T ], we have
∥ v θ (•, t) -v * (•, t)∥ L 2 (πt) ≤ ( √ d + 1)ε,
where π t is the distribution of X t = tX 1 + (1 -t)X 0 .
The proof of Theorem 3.1 can be found in Appendix A.1.
Universal Approximation under the L 2 -norm Many existing universal approximation theory of neural networks focus on approximating target functions on a compact domain under the L ∞ -norm (Yarotsky, 2017;Schmidt-Hieber, 2020;Gühring et al., 2020). Instead, we provide an L 2approximation error bound over the unbounded input domain, where we tackle the unboundedness through a truncation argument.
Lipschitz Neural Network Conventional universal approximation theories of neural networks do not typically provide guarantees on the Lipschitz continuity of the network (Cybenko, 1989;Barron, 1993;Yarotsky, 2017), which is important for effective learning of the true velocity field. A line of research (Jiao et al., 2023;Dahal et al., 2022;Huang et al., 2022) studies Lipschitz neural networks motivated by the Wasserstein Generative Adversarial Network (WGAN) (Arjovsky et al., 2017). The paper (Jiao et al., 2023) studies the approximation capacity of ReLU neural networks with norm constraints on the weights. Meanwhile, (Huang et al., 2022, Lemma 11) provides an explicit bound on the Lipscitz constant required for approximating Hölder functions. In (Dahal et al., 2022), statistical guarantees for WGAN are provided under the Wasserstein 1-distance, assuming that the data distribution is supported on a low-dimensional manifold. These techniques are scalable to our analysis, and for brevity, we adopt the proof of the work (Chen et al., 2023a).
The key difference between our paper and (Chen et al., 2023a) is that they assume the on-support score function is Lipschitz uniformly for t ∈ [t 0 , T ], whereas our paper derives the Lipschitzness of the true velocity field from the assumption on the target distribution. In our construction, the Lipschitz continuity constraints γ 1 and γ 2 do not undermine the approximation power of the neural networks.
In practice, such Lipschitz regularity is often enforced during training by adding regularization (Virmaux & Scaman, 2018;Pauli et al., 2021;Gouk et al., 2021). From a theoretical perspective, the Lipschitz property of the estimated velocity field is crucial in bounding the distribution recovery error, as we demonstrate in Section 5. Moreover, the Lipschitz continuity of the estimated velocity field ensures the existence and uniqueness of the solution of the ODE.
this section cite: ['b67', 'b53', 'b30', 'b16', 'b3', 'b67', 'b34', 'b17', 'b33', 'b2', 'b34', 'b17', 'b61', 'b49', 'b28']

Section: Time as an Additional Input Dimension
In our approach, we introduce time t as an extra input dimension to the neural network, and the network size scales polynomially with the Lipschitz constant τ of the true velocity field with respect to t. In Section D, we derive an upper bound for τ on a clipped time span [0, T ], where T < 1.
Proof Sketch Theorem 3.1 is established by construction.
A noteworthy distinction from the existing universal approximation theories is that the input domain of the velocity field is unbounded. To establish the theorem, we leverage a truncation argument. Let R be a truncation radius.
On the hypercube [-R, R] d × [0, T ], we construct v θ as a piece-wise linear function to approximate v * in the sense of
L ∞ ([-R, R] d × [0, T ]).
Outside the hypercube, we simply set v θ = 0. The L 2 approximation error can be decomposed as
∥v θ (•, t) -v * (•, t)∥ L 2 (πt) = ∥x∥≤R ∥v θ (x, t) -v * (x, t)∥ 2 π t (dx) 1/2 (I) + ∥x∥>R ∥v θ (x, t) -v * (x, t)∥ 2 π t (dx) 1/2 (II)
.
The error term (I) is directly bounded by the approximation error of v θ on the hypercube. It is worth noting that since v θ is bounded and v * (X t , t) has a bounded second moment, the term (II) can be controlled by utilizing the tail behavior of π t .
this section cite: []

Section: Generalization
In this section, we consider the generalization error of estimating the velocity field. We begin with the following connection between the loss function L(v) and the
L 2 ap- proximation error ∥v(•, t) -v * (•, t)∥ L 2 (πt) .
Lemma 4.1. The following holds for any v(x, t):
L(v) -L(v * ) = 1 T T 0 ∥v(•, t) -v * (•, t)∥ 2 L 2 (πt) dt.
Proof. By some calculus, we have
E ∥X 1 -X 0 -v(X t , t)∥ 2 =E ∥X 1 -X 0 -v * (X t , t) + v * (X t , t) -v(X t , t)∥ 2 =E ∥X 1 -X 0 -v * (X t , t)∥ 2 + ∥v(•, t) -v * (•, t)∥ 2 L 2 (πt) + 2E [⟨X 1 -X 0 -v * (X t , t), v * (X t , t) -v(X t , t)⟩] .
(7) By taking expectation conditioned on X t , we have
E [⟨X 1 -X 0 -v * (X t , t), v * (X t , t) -v(X t , t)⟩] =E [E[⟨X 1 -X 0 -v * (X t , t), v * (X t , t) -v(X t , t)⟩|X t ]] =E [⟨E[X 1 -X 0 |X t ] -v * (X t , t), v * (X t , t) -v(X t , t)⟩] =E [⟨v * (X t , t) -v * (X t , t), v * (X t , t) -v(X t , t)⟩] = 0.
Substituting the above identity into (7) and integrating on interval [0, T ], we obtain
L(v) = L(v * ) + 1 T T 0 ∥v(•, t) -v * (•, t)∥ 2 L 2 (πt) dt,
which concludes the proof.
According to Lemma 4.1, minimizing (4) is equivalent to minimizing the difference between the network and the true velocity field in L 2 (π t )-norm.
Empirical Evaluation Let us define
ℓ(x, v) := 1 T T 0 ∥x -x 0 -v(tx + (1 -t)x 0 , t)∥ 2 π 0 (x 0 )dx 0 dt.(8)
In this paper, we choose the standard Gaussian distribution as the prior distribution, i.e., π 0 = N (0, I d ), where d is the dimension of the data. Given n independent and identically distributed (i.i.d.) samples {x 1,i } n
i=1 from π 1 , we have the following empirical version of the least square loss:
L(v) := 1 n n i=1 ℓ(x 1,i , v).(9)
Since our main interest lies in the sample complexity of sampling from π 1 , we consider the situation where ℓ(x, v) can be computed exactly. However, in the usual implementation, the expectation in ( 8) is replaced by empirical evaluation. Given m i.i.d. samples {(t j , x 0,j )} m j=1 from Unif[0, T ] and π 0 , which are cheap to generate, then (8) has the following empirical evaluation:
ℓ(x, v) := 1 m m j=1 ∥x -x 0,j -v(t j x + (1 -t j )x 0,j , t j )∥ 2 .
(10) Due to the efficacy of sampling t and x 0 , ℓ(x, v) can be efficiently approximated by ℓ(x, v) via polynomial-size sample from Unif[0, T ] and π 0 , which will be explained exactly in Section 4.1. Now, we consider the Empirical Risk Minimization (ERM):
v ∈ argmin v∈V L(v) := 1 n n i=1 ℓ(x 1,i , v)(11)
this section cite: []

Section: Error Decomposition
The error of the estimated vector field (11) can be decomposed as:
L( v) -L(v * ) = L( v) -inf v∈NN L(v) Generalization error + inf v∈NN (L(v) -L(v * )) Approximation error (12
)
Further, the generalization error has the following decomposition:
L( v) -inf v∈NN L(v) =L( v) -L( v) + L( v) -L( v) + L( v) -L( v) ≤ L( v) -L( v) + L( v) -L( v),(13)
where the inequality follows from ERM, and v is defined in (6). Note that, for any v ∈ NN, we have
L(v) -L(v) =L(v) -L(v) + L(v) -L(v) = 1 n n i=1 (L(v) -ℓ(x 1,i , v)) + 1 n n i=1 (ℓ(x 1,i , v) -ℓ(x 1,i , v))(14)
By defining H = {ℓ(•, v) : v ∈ NN(L, M, J, K, κ, γ 1 , γ 2 )}, we can apply conventional statistical learning arguments to analyze the first term within the function class H. Due to the unbounded nature of the loss function |x -x 0 -v(tx + (1 -t)x 0 , t)| 2 , controlling the second term requires an additional truncation argument. We will provide further details on our approach at the end of this section.
The complexity of a function class can be measured using the covering number.
Definition 4.2 (Covering number). Let ρ be a pseudo-metric on M and S ⊆ M. For any δ > 0, a set A ⊆ M is called a δ-covering of S if for any x ∈ S there exists y ∈ A such that ρ(x, y) ≤ δ. The δ-covering number of S, denoted by N (δ, S, ρ), is the minimum cardinality of any δ-covering of S.
The function class H exhibits the following properties, which are useful for analyzing the generalization error.
(i) Bounded sup-norm According to Theorem 3.1, the estimated velocity field v(x, t) can be chosen to satisfy
the condition ∥ v∥ L ∞ (R d ×[0,T ]) ≤ K = O √ log(d/ε) 1-T . Then Lemma B.1 shows that sup v∈NN sup x∈[0,1] d ℓ(x, v) ≲ d + K 2 ≲ d + log(d/ε) (1 -T ) 2 .
(ii) Covering number evaluation The covering number of the network class selected in Theorem 3.1 is evaluated as follows:
log N (δ, NN, ∥ • ∥ L ∞ ([-D,D] d ×[0,1]) ) ≲ JL log LM (D ∨ 1)κ δ . (15
)
The above evaluation can be found in (Chen et al., 2022b, Lemma 5.3). Based on the above result, we have the following evaluation for the covering number of the loss function class H:
Lemma 4.3. The covering number of H is evaluated by
log N (δ, H, ∥ • ∥ L ∞ ([0,1] d ) ) ≲ JL log (K + d 1/2 )LM κ log((K 2 + d)/δ) δ .(16)
The proof of Lemma 4.3 is deferred to Appendix B.2. It is worth noting that the evaluation is non-trivial because the evaluation in (15) considers the L ∞ -norm on a bounded subspace, while the region of integration in (8) is unbounded.
To overcome this challenge, we utilize a truncation argument to provide the covering number evaluation for H.
Based on the above discussion, we can now derive the following generalization bound
= n -1 d+5 . Then with probability of at least 1 -1 n , it holds 1 T T 0 ∥ v(•, t) -v * (•, t)∥ 2 L 2 (πt) dt = O ζ d/2 (1 -T ) 4 n -2 d+5 + n d+1 2(d+5) m -1 2 ,
where we omit factors in d, log n, log m, log(1 -T ). By setting m to be of the order O(n), we obtain the convergence
rate of order O ζ d/2 (1-T ) 4 n -2 d+5 .
The proof can be found in Appendix B.3. To the best of our knowledge, Theorem 4.4 provides the first explicit sample complexity bound for FM. Theorem 4.4 becomes vacuous when T tends to 1 with fixed sample size n. This is a consequence of the blowup of the velocity field v * (x, t) as t tends to 1. Although a smaller early stopping time leads to better generalization error, stopping the sampling process at an early time results in a bad distribution recovery. In Section 5, we will show the tradeoff in the choice of stopping time T .
Proof Sketch The generalization error is divided into two terms. The first term's randomness arises from drawing samples from the target distribution π 1 , while the second term's randomness comes from sampling from π 0 and Unif[0, T ].
We encounter two difficulties in deriving the generalization error bound. The first difficulty lies in evaluating the covering number of the loss function class H for the first term. The second difficulty stems from the unboundedness of the term ∥x -x 0 -v(tx + (1 -t)x 0 , t)∥ 2 in the second term. To handle this, we leverage the concentration property of the Gaussian prior distribution and employ a truncation argument to provide an upper bound for the second term with high probability. Specifically, the second term can be decomposed as follows:
ℓ(x 1,i , v) -ℓ(x 1,i , v) = ℓ(x 1,i , v) -ℓ trunc (x 1,i , v) Truncation error (I) + ℓ trunc (x 1,i , v) -ℓ trunc (x 1,i , v) Statistical error + ℓ trunc (x 1,i , v) -ℓ(x 1,i , v) Truncation error (II) ,
where
ℓ trunc (x 1,i , v) := E t,x0 [∥x 1,i -x 0 - v(tx + (1 -t)x 0 , t)∥ 2 1{∥x 0 ∥ ∞ ≤ R}]
and
ℓ trunc (x 1,i , v) := 1 m m j=1 ∥x 1,i -x 0,j -v(t j x 1,i + (1 -t)x 0,j , t j )∥ 2 1{∥x 0,j ∥ ∞ ≤ R}. We can control
Truncation error (I) by utilizing the concentration of Gaussian variables. On the other hand, Statistical error can be controlled using a covering number argument. Furthermore, Truncation error (II) is likely to be equal to zero due to the concentration of Gaussian variables.
this section cite: []

Section: Sampling
This section establishes distribution recovery guarantees using the estimated velocity field.
this section cite: []

Section: Estimated Sampling Dynamics
Given the estimated velocity field v, we can generate samples from an approximation of the continuous flow ODE starting from the prior distribution:
d X t (x) = v( X t (x), t)dt, X 0 (x) = x ∼ π 0 , 0 ≤ t ≤ T.(17)
Proposition 5.1. Suppose Assumption 1.1 holds. For any velocity field v * with Lipschitz constant ζ w.r.t. x, given n samples {x 1,i } n i=1 from π 1 and m samples from π 0 and Unif[0, T ], we choose NN as in Theorem 3.1 with ε = n -1 d+5 . Then with probability of at least 1 -1 n , it holds
W 2 (π T , π T ) = O e γ1 ζ d/4 (1 -T ) 2 n -1 d+5 .(18)
Proof. Note that X t (x) and X t (x) form a coupling of π t and π t , by the definition of Wasserstein-2 distance, we have
W 2 2 (π t , π t ) ≤ R d ∥X t (x) -X t (x)∥ 2 π 0 (x)dx, (19
)
where X t is the flow map solution of (53) with the exact v * defined in (3) and X t is the flow map solution of (54). Now, we consider the evolution of
R t := R d ∥X t (x) -X t (x)∥ 2 π 0 (x)dx.
Differentiating on both sides, we get
dR t dt = 2 R d ⟨v * (X t (x), t)-v( X t (x), t), X t (x)-X t (x)⟩ π 0 (x)dx = 2 R d ⟨v * (X t (x), t) -v(X t (x), t) + v(X t (x), t) -v( X t (x), t), X t (x) -X t (x)⟩π 0 (x)dx.
(20) Using the inequality 2⟨a, b⟩ ≤ ∥a∥ 2 + ∥b∥ 2 , we have
2⟨v * (X t (x), t) -v(X t (x), t), X t (x) -X t (x)⟩ ≤ ∥v * (X t (x), t) -v(X t (x), t)∥ 2 + ∥X t (x) -X t (x)∥ 2 . (21) Note that v ∈ NN defined in Theorem 3.1 is γ 1 -Lipschitz continuous w.r.t. x, the Cauchy-Schwartz inequality implies 2⟨ v(X t (x), t) -v( X t (x), t), X t (x) -X t (x)⟩ ≤ 2γ 1 ∥X t (x) -X t (x)∥ 2 . (22)
Combining ( 20), ( 21) and ( 22), we obtain
dR t dt ≤(1 + 2γ 1 )R t + R d ∥v * (X t (x), t) -v(X t (x), t)∥ 2 π 0 (x)dx.
Therefore, by Lemma C.6 and since R 0 = 0, we deduce
R T ≤ e 1+2γ1 T 0 R d ∥v * (X t (x), t)-v(X t (x), t)∥ 2 π 0 (x)dxdt = e 1+2γ1 T 0 ∥v * (•, t) -v(•, t)∥ 2 L 2 (πt) dt.
By Theorem 4.4 and the fact that v is γ 1 -Lipschitz continuous w.r.t. x since we choose NN as in Theorem 4.4, we get the desired result.
Time Discretization In practice, we need to use a discretetime approximation for the sampling dynamics (17). Let 0 = t 0 < t 1 < • • • < t N = T be the discretization points. We consider the explicit Euler discretization scheme:
d X t (x) = v( X t k (x), t k )dt, t ∈ [t k , t k+1 ),(23)
for k = 0, 1, . . . , N -1 and X 0 (x) = x ∼ π 0 . We denote the distribution of X T (x) by π T .
To establish the distribution recovery guarantees, we need the following discretization error bound:
Lemma 5.2. Let 0 = t 0 < t 1 < • • • < t N = T be the discretization points. For any neural network v in NN(L, M, J, K, κ, γ 1 , γ 2 ), we have:
W 2 ( π T , π T ) = O   e γ1 (γ 1 K + γ 2 ) N -1 k=0 (t k+1 -t k ) 3   ,
where π is the distribution of the final output of the estimated sampling dynamics (17).
The proof of Lemma 5.2 can be found in Appendix C.2.
Tradeoff on Stopping Time T To show the tradeoff, we first present the following lemma:
Lemma 5.3. Suppose Assumption 1.1 holds, we have
W 2 (π T , π 1 ) ≲ (1 -T ) √ d.
The proof of Lemma 5.3 is deferred to Appendix C.2. Proposition 5.1 demonstrates that as the stopping time T tends to 1, the error of using the estimated velocity field in the sampling dynamics increases. Conversely, according to Lemma 5.3, the Wasserstein-2 distance between π T and π 1 decreases as T approaches 1. This reveals a tradeoff in the stopping time T between the error in velocity field estimation and the distribution recovery.
this section cite: []

Section: Conclusion
This paper presents a statistical learning theory perspective on CNFs based on FM. We demonstrate that a Lipschitz neural network can approximate the true velocity field under L 2 (π t )-norm and provide a sample complexity analysis for estimating the velocity field. Furthermore, we prove that under mild assumptions, the generated distribution of CNFs based on FM converges to the target data in Wasserstein-2 distance. Additionally, we show that the convergence rate can be significantly improved by assuming an additional mild Lipschitz condition on the target score function. To the best of our knowledge, this is the first end-to-end analysis of FM.
this section cite: []

Section: References
Ref_id:b0 Title: Building normalizing flows with stochastic interpolants Year: (2023)
Ref_id:b1 Title: Stochastic interpolants: A unifying framework for flows and diffusions Year: (2023)
Ref_id:b2 Title: Wasserstein generative adversarial networks Year: (2017)
Ref_id:b3 Title: Universal approximation bounds for superpositions of a sigmoidal function Year: (1993)
Ref_id:b4 Title: On extensions of the brunn-minkowski and prékopa-leindler theorems, including inequalities for log concave functions, and with an application to the diffusion equation Year: (1976)
Ref_id:b5 Title: Improved analysis of scorebased generative modeling: User-friendly bounds under minimal smoothness assumptions Year: (2022)
Ref_id:b6 Title: Efficient approximation of deep relu networks for functions on low dimensional manifolds Year: (2019)
Ref_id:b7 Title: Distribution approximation and statistical estimation guarantees of generative adversarial networks Year: (2020)
Ref_id:b8 Title: Statistical guarantees of generative adversarial networks for distribution estimation Year: (2020)
Ref_id:b9 Title: Nonparametric regression on low-dimensional manifolds using deep relu networks: Function approximation and statistical recovery. Information and Inference: A Year: (2022)
Ref_id:b10 Title: Score approximation, estimation and distribution recovery of diffusion models on low-dimensional data Year: (2023)
Ref_id:b11 Title: Riemannian flow matching on general geometries Year: (2023)
Ref_id:b12 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b13 Title: Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions Year: (2023)
Ref_id:b14 Title: Restorationdegradation beyond linear diffusions: A non-asymptotic analysis for ddim-type samplers Year: (2023)
Ref_id:b15 Title: An entropic generalization of Caffarelli's contraction theorem via covariance inequalities Year: (2022)
Ref_id:b16 Title: Approximation by superpositions of a sigmoidal function Year: (1989)
Ref_id:b17 Title: On deep generative models for approximation and estimation of distributions on manifolds Year: (2022)
Ref_id:b18 Title: Diffusion schrödinger bridge with applications to score-based generative modeling Year: (2021)
Ref_id:b19 Title: Information theoretic inequalities Year: (1991)
Ref_id:b20 Title: Density estimation using real NVP Year: (2017-04-24)
Ref_id:b21 Title: Neural spline flows Year: (2019)
Ref_id:b22 Title: Deep generative learning via variational gradient flow Year: (2019)
Ref_id:b23 Title: Gaussian interpolation flows Year: (2024)
Ref_id:b24 Title: Discriminative metric learning for partial label learning Year: (2021)
Ref_id:b25 Title: Understanding partial multi-label learning via mutual information Year: (2021)
Ref_id:b26 Title: Partial label learning via label influence function Year: (2022)
Ref_id:b27 Title: A unifying probabilistic framework for partially labeled data learning Year: (2022)
Ref_id:b28 Title: Regularisation of neural networks by enforcing lipschitz continuity Year: (2021)
Ref_id:b29 Title: Scalable reversible generative models with free-form continuous dynamics Year: (2019)
Ref_id:b30 Title: Error bounds for approximations with deep relu neural networks in w s, p norms Year: (2020)
Ref_id:b31 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b32 Title: Neural autoregressive flows Year: (2018)
Ref_id:b33 Title: An error analysis of generative adversarial networks for learning distributions Year: (2022)
Ref_id:b34 Title: Approximation bounds for norm constrained neural networks with applications to regression and gans Year: (2023)
Ref_id:b35 Title: Elucidating the design space of diffusion-based generative models Year: (2022)
Ref_id:b36 Title: Convergence of score-based generative modeling for general data distributions Year: (2023)
Ref_id:b37 Title: Minimizing trajectory curvature of ode-based generative models Year: (2023)
Ref_id:b38 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b39 Title: Rectified flow: A marginal preserving approach to optimal transport Year: (2022)
Ref_id:b40 Title: Hyperspectral imagery classification via stochastic hhsvms Year: (2018)
Ref_id:b41 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2023)
Ref_id:b42 Title: Maximum likelihood training for score-based diffusion odes by high order denoising score matching Year: (2022)
Ref_id:b43 Title: Knowledge distillation in iterative generative models for improved sampling speed Year: (2021)
Ref_id:b44 Title:  Year: (2021)
Ref_id:b45 Title: On the Lipschitz properties of transportation along heat flows Year: (2022)
Ref_id:b46 Title: Action matching: A variational method for learning stochastic dynamics from samples Year: (2022)
Ref_id:b47 Title: Diffusion models are minimax optimal distribution estimators Year: (2023)
Ref_id:b48 Title: Masked autoregressive flow for density estimation Year: (2017)
Ref_id:b49 Title: Training robust neural networks using lipschitz bounds Year: (2021)
Ref_id:b50 Title: Variational inference with normalizing flows Year: (2015)
Ref_id:b51 Title: Progressive distillation for fast sampling of diffusion models Year: (2022)
Ref_id:b52 Title: Log-concavity and strong log-concavity: a review Year: (2014)
Ref_id:b53 Title: Nonparametric regression using deep neural networks with ReLU activation function Year: (2020)
Ref_id:b54 Title: Diffusion Schrödinger bridge matching Year: (2023)
Ref_id:b55 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b56 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b57 Title: Consistency models Year: (2023)
Ref_id:b58 Title: A family of nonparametric density estimation algorithms Year: (2013)
Ref_id:b59 Title: Density estimation by dual ascent of the log-likelihood Year: (2010)
Ref_id:b60 Title: Conditional flow matching: Simulation-free dynamic optimal transport Year: (2023)
Ref_id:b61 Title: Lipschitz regularity of deep neural networks: analysis and efficient estimation. Advances in Neural Information Processing Systems Year: (2018)
Ref_id:b62 Title: High-dimensional statistics: A nonasymptotic viewpoint Year: (2019)
Ref_id:b63 Title: Deep generative learning via schrödinger bridge Year: (2021)
Ref_id:b64 Title: Convexity of mutual information along the heat flow Year: (2018)
Ref_id:b65 Title: Convexity of mutual information along the Ornstein-Uhlenbeck flow Year: (2018)
Ref_id:b66 Title: Fast point cloud generation with straight flows Year: (2022)
Ref_id:b67 Title: Error bounds for approximations with deep relu networks Year: (2017)
Ref_id:b68 Title: Fast sampling of diffusion models via operator learning Year: (2022)
Ref_id:b69 Title: Sample complexity for distributionally robust learning under chi-square divergence Year: (2023)
Ref_id:b70 Title: Sequential kernel goodness-of-fit testing Year: (2024)
