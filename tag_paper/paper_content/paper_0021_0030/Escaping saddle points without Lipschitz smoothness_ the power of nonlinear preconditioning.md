Title: Escaping saddle points without Lipschitz smoothness: the power of nonlinear preconditioning
Abstract: We study generalized smoothness in nonconvex optimization, focusing on (L 0 , L 1 )smoothness and anisotropic smoothness. The former was empirically derived from practical neural network training examples, while the latter arises naturally in the analysis of nonlinearly preconditioned gradient methods. We introduce a new sufficient condition that encompasses both notions, reveals their close connection, and holds in key applications such as phase retrieval and matrix factorization. Leveraging tools from dynamical systems theory, we then show that nonlinear preconditioning -including gradient clipping -preserves the saddle point avoidance property of classical gradient descent. Crucially, the assumptions required for this analysis are actually satisfied in these applications, unlike in classical results that rely on restrictive Lipschitz smoothness conditions. We further analyze a perturbed variant that efficiently attains second-order stationarity with only logarithmic dependence on dimension, matching similar guarantees of classical gradient methods.

Section: Introduction
We consider the unconstrained optimization problem minimize
x∈R n f (x),(1)
where f : R n → R is a twice continuously differentiable nonconvex function. This work studies the nonlinearly preconditioned gradient method, with iterates described by
x k+1 = T γ,λ (x k ) := x k -γ∇ϕ * (λ∇f (x k )), (P-GD)
where ϕ : R n → R ∪ {∞} is referred to as the reference function, and ϕ * , its convex conjugate, is called the dual reference function.
Nonlinear preconditioning provides a flexible framework for constructing and analyzing gradientbased optimization algorithms [36,21,31]. For instance, when ϕ(x) = 1 2 ∥x∥ 2 , the update (P-GD) reduces to classical gradient descent. More broadly, we focus on isotropic reference functions of the form ϕ(x) = h(∥x∥) for some scalar kernel function h : R → R + ∪ {∞}, though our results extend in part to more general settings, including separable reference functions ϕ(x) = n i=1 h(x i ). Some kernel functions of interest include:
h 1 (x) = cosh(x) -1, h 2 (x) = exp(|x|) -|x| -1, h 3 (x) = -|x| -ln(1 -|x|),(2)
each of which upper bounds the quadratic function x 2 /2, as visualized in Fig. 1. These choices induce preconditioners that closely resemble common gradient clipping heuristics, as shown in Fig. 1.
The effectiveness of gradient clipping has been justified using the concept of (L 0 , L 1 )-smoothness, which is empirically motivated by practical neural network training scenarios [42]. However, it 39th Conference on Neural Information Processing Systems (NeurIPS 2025). remains unclear under what precise conditions this smoothness assumption holds in real-world applications. On the other hand, the preconditioned gradient method is naturally analyzed under anisotropic smoothness [36], another generalization of the classical Lipschitz smoothness condition. Rather than imposing a global quadratic upper bound, anisotropic smoothness permits more flexible upper bounds defined in terms of the reference function ϕ. This makes the preconditioned gradient method particularly attractive in settings where the standard Lipschitz condition is too restrictive. This leads us to our first central question:
Can we formally establish anisotropic smoothness and (L 0 , L 1 )-smoothness of practical problems where traditional assumptions fail?
Our second line of inquiry focuses on the behavior of the preconditioned gradient method when applied to nonconvex objectives. Classical gradient descent is known to avoid strict saddle points under the assumption of (global) Lipschitz smoothness [24], a phenomenon which helps explain its strong empirical performance in nonconvex settings. However, for many practical applications Lipschitz smoothness holds only locally or on compact sets around a minimizer, meaning that this assumption is not truly satisfied. This raises the following question:
Does nonlinear preconditioning preserve the saddle point avoidance properties of gradient descent under a possibly less stringent smoothness assumption?
Our results reveal novel connections between different generalizations of smoothness and provide strong theoretical support for nonlinear preconditioning, particularly in nonconvex settings where the classical Lipschitz smoothness assumption may fail.
Contributions Our contributions can be summarized as follows.
• We investigate the classes of problems for which (L 0 , L 1 )-smoothness and anisotropic smoothness -two generalizations of the classical Lipschitz smoothness condition -are applicable. To this end, we propose a novel sufficient condition (Assumption 2.8) that guarantees both anisotropic and (L 0 , L 1 )-smoothness, thereby revealing a structural link between these two frameworks. We further demonstrate in section 2.3 that this condition holds for several prominent nonconvex problems, including phase retrieval, low-rank matrix factorization, and Burer-Monteiro factorizations of MaxCut-type problems.
• We establish that nonlinear preconditioning preserves the saddle point avoidance behavior of gradient descent, and moreover extends results from the classical Lipschitz smoothness framework to the broader setting of anisotropic smoothness. Specifically, we prove asymptotic avoidance of strict saddle points by leveraging the stable-center manifold theorem. By invoking a recent nonsmooth generalization of this theorem, this analysis is then further extended to accommodate hard gradient clipping. Finally, we present a complexity analysis for a perturbed variant of the preconditioned gradient method, showing that it converges to a second-order stationary point with only logarithmic dependence on the problem dimension.
Notation Let S n×n be the set of symmetric n × n matrices. We denote the standard Euclidean inner product on R n by ⟨•, •, ⟩, and the corresponding norm by ∥ • ∥. For X, Y ∈ R m×n , ⟨X, Y ⟩ = trace(X ⊤ Y ) is the standard inner product on R m×n and ∥ • ∥ denotes the spectral norm. The class of k times continuously differentiable functions on an open set O ⊆ R n is denoted by C k (O). We write sgn(x) = x /∥x∥ for x ∈ R n \ {0} and 0 otherwise. A function f ∈ C 2 (R n ) is L-Lipschitz smooth if for all x, y ∈ R n it holds that ∥∇f (x) -∇f (y)∥ ≤ L∥x -y∥, with L ≥ 0, and (L 0 , L 1 )-smooth if ∥∇ 2 f (x)∥ ≤ L 0 + L 1 ∥∇f (x)∥ for all x ∈ R n with L 0 , L 1 ≥ 0. Otherwise, we follow [37].
this section cite: ['b35', 'b20', 'b30', 'b41', 'b35', 'b23', 'b36']

Section: Related work
Generalized smoothness Gradient descent is traditionally analyzed under the assumption of Lipschitz smoothness [34], although many applications violate this condition. Bregman relative smoothness is a popular extension which allows the Hessian to grow unbounded, see e.g. [30] which assumes a certain polynomial growth. More recently, the (L 0 , L 1 )-smoothness condition was proposed by Zhang et al. [42], based on empirical observations in LSTMs, and used to analyze clipped gradient descent and a momentum variant [41]. The framework has since been applied to stochastic normalized gradient descent [43] and generalized SignSGD [12]. Notably, Crawshaw et al. [12] provided empirical evidence that (L 0 , L 1 )-smoothness holds for Transformers [40], albeit with layer-wise variation in constants. Further generalizations include α-symmetric smoothness [9] and ℓ-smoothness [28], and the latter was used to analyze the convergence of Adam [29]. Despite empirical support for these conditions in key applications, theoretical guarantees remain limited.
this section cite: ['b33', 'b29', 'b41', 'b40', 'b42', 'b11', 'b11', 'b39', 'b8', 'b27', 'b28']

Section: Nonlinear preconditioning
The preconditioned gradient method with updates given by (P-GD) was introduced in the convex setting by Maddison et al. [31]. Then, Laude et al. [22,21] studied L-anisotropic smoothness and, under this condition, showed convergence of (P-GD) for nonconvex problems. The method was later extended to measure spaces [4]. Oikonomidis et al. [36] proposed the (L, L)-anisotropic smoothness condition, connected it to (L 0 , L 1 )-smoothness, and analyzed convergence of (P-GD) in both convex and nonconvex settings. We also highlight the works [26,35] that study the concept of Φ-convexity, which is closely related to anisotropic smoothness.
this section cite: ['b30', 'b21', 'b20', 'b3', 'b35', 'b25', 'b34']

Section: Saddle point avoidance
To explain the success of gradient descent on nonconvex problems, much work has focused on its (strict) saddle point avoidance properties [25,24]. It was shown that gradient descent may take exponential time to escape saddle points, even with random initialization [13]. The works [27,33] showed that noise-injected normalized gradient descent escapes them more efficiently. Jin et al. [17,18] demonstrated that perturbed gradient descent escapes saddle points in time polylogarithmic in the problem dimension. Recently, Cao et al. [8] studied saddle point avoidance under a second-order self-bounding regularity condition rather than under classical Lipschitz smoothness.
2 Anisotropic smoothness
this section cite: ['b24', 'b23', 'b12', 'b26', 'b32', 'b16', 'b17', 'b7']

Section: Definition and basic properties
This section introduces (L, L)-anisotropic smoothness as proposed by [36]. The following assumption, which guarantees in particular that ϕ * ∈ C 1 (R n ) and ϕ ≥ 0, is considered valid throughout.
Assumption 2.1. The function ϕ : R n → R is proper, lsc, strongly convex and even with ϕ(0) = 0.
We usually also assume the following condition, which ensures in particular that ϕ * ∈ C 2 (R n ). Assumption 2.2. int dom ϕ ̸ = ∅; ϕ ∈ C 2 (int dom ϕ), and for any sequence {x k } k∈N that converges to some boundary point of int dom ϕ, it follows that ∥∇ϕ(x k )∥ → ∞.
We follow the definition of anisotropic smoothness by [36], which reduces to [21, Def. 3.1] with reference function Lϕ if dom ϕ = R n . If f ∈ C 1 , this concept corresponds to a global version of anisotropic prox-regularity of -f [20, Def. 2.13]. For a geometric intuition, we refer to [26,35,36]. Definition 2.3 ((L, L)-anisotropic smoothness [36]). A function f : R n → R is (L, L)anisotropically smooth relative to a reference function ϕ with constants L,
L > 0 if f (x) ≤ f (x) + LL -1 ϕ(L(x -ȳ)) -LL -1 ϕ(L(x -ȳ))
for all x, x ∈ R n , where
ȳ = T L -1 , L-1 (x) = x -L -1 ∇ϕ * ( L-1 ∇f (x)).
The following proposition provides a sufficient condition for anisotropic smoothness. We consider the case ϕ * ∈ C 2 for simplicity of exposition, but note that a variant for ϕ * / ∈ C 2 can also be formulated. Proposition 2.4 (Second-order characterization of (L, L)-anisotropic smoothness). Suppose that Assumption 2.2 holds, and let f ∈ C 2 be such that for all
x ∈ R n λ max (∇ 2 ϕ * ( L-1 ∇f (x))∇ 2 f (x)) ≤ L L,(3)
and lim ∥x∥→∞ ∥T L -1 , L-1 (x)∥ = ∞. Moreover, assume that either dom ϕ is bounded or that dom ϕ = R n , and that for all x ∈ R n we have f (x) ≤ Lr -foot_0 ϕ(rx)β for some r ∈ (0, L), b ∈ R. Then, f is (δL, L)-anisotropically smooth relative to ϕ for any δ > 1.
We say that f satisfies the second-order characterization of anisotropic smoothness if (3) holds. Note that the growth condition on f is not restrictive when ϕ = dom R n , and that the coercivity assumption on the iteration map T L -1 , L-1 is very mild; we refer the reader to the arguments in [36]. Finally, we connect anisotropic smoothness to some popular smoothness notions. Example 2.5 (Lipschitz-smoothness [36, Proposition 2.3]). Suppose that f ∈ C 2 is L f -Lipschitz smooth. Denote by µ > 0 the parameter of strong convexity of a reference function ϕ. Then f is ( L f/µ, 1)-anisotropically smooth relative to ϕ. Example 2.6 ((L 0 , L 1 )-smoothness). Let f ∈ C 2 be (L 0 , L 1 )-smooth, let L = L 1 , L = L0 /L1, and let ϕ(x) = -∥x∥ln(1 -∥x∥). Then f satisfies the second-order characterization of (L, L)anisotropic smoothness relative to ϕ [36, Proposition 2.6 & Corollary 2.7].
this section cite: ['b35', 'b35', 'b25', 'b34', 'b35', 'b35', 'b35']

Section: A novel sufficient condition for generalized smoothness
Although it is well-known that univariate polynomials are (L 0 , L 1 )-smooth [42, Lemma 2], this is not necessarily the case for multivariate polynomials, as illustrated by the following example. Example 2.7. Consider the polynomial f (x, y) = 1 4 x 4 + 1 4 y 4 -1 2 x 2 y 2 with gradient and Hessian
∇f (x, y) = x 3 -xy 2 y 3 -x 2 y , ∇ 2 f (x, y) = 3x 2 -y 2 -2xy -2xy -x 2 + 3y 2 .
Remark that ∇f (x, -x) = 0 and ∇ 2 f (x, -x) = x 2 2 -2 -2 2 . Clearly, f cannot be (L 0 , L 1 )-smooth since ∥∇ 2 f (x, -x)∥ F = 4∥x∥ 2 grows unbounded, while ∥∇f (x, -x)∥ = 0 for all x ∈ R.
For multivariate polynomials there may exist a path of ∥x∥ → ∞ along which the gradient norm grows slower than the Hessian norm, in which case (L 0 , L 1 )-smoothness cannot hold. More examples are included in appendix A.2. Based on this insight, we propose the following novel condition. Assumption 2.8. There exists an R ∈ N such that for all x ∈ R n ∥∇ 2 f (x)∥ F ≤ p R (∥x∥), and ∥∇f (x)∥ ≥ q R+1 (∥x∥).
Here p R (α) = R i=0 a i α i and q R+1 (α) = R+1 i=0 b i α i are polynomials of degree R and R + 1, respectively, and in particular we assume that b R+1 > 0.
Note that [30] constructs a Bregman distance inducing kernel function under a similar polynomial upper bound to the Hessian norm. Appendix A.1 verifies Assumption 2.8 for univariate polynomials. The following result states that Assumption 2.8 is a sufficient condition for (L 0 , L 1 )-smoothness. 1  Theorem 2.9. Suppose that Assumption 2.8 holds for f ∈ C 2 . Then, for any L 1 > 0 there exists an
L 0 > 0 such that f is (L 0 , L 1 )-smooth.
Under mild conditions on the kernel function h, which appendix A.5 shows hold for all examples in (2), Assumption 2.8 also implies the second-order characterization of anisotropic smoothness. In fact, it implies the stronger condition that ∥∇ 2 ϕ * ( L-1 ∇f (x))∇ 2 f (x)∥ is uniformly bounded. Assumption 2.10. The reference function ϕ is isotropic, i.e., ϕ(x) = h(∥x∥), and such that (i) h * ′ (y) /y is a decreasing function on R + , (ii) lim y→+∞ yh * ′′ (y) = C 2 , for some C 2 ∈ R + , and (iii)
lim y→+∞ h * ′ (s d (y)) y = 0, for any polynomial s d (α) = d i=0 u i α i of degree d.
Theorem 2.11. Suppose that f satisfies Assumption 2.8. If ϕ satisfies Assumption 2.2 and Assumption 2.10, then for any L > 0 there exists an L > 0 such that f satisfies the second-order characterization of (L, L)-anisotropic smoothness relative to ϕ.
this section cite: ['b29']

Section: Applications
We now establish for a number of key applications that Assumption 2.8 holds, thus proving that the objective is (L 0 , L 1 )-smooth and satisfies the second-order characterization of (L, L)-anisotropic smoothness. Remark that for all of these, the classical Lipschitz smoothness assumption is violated.
this section cite: []

Section: Phase retrieval
Consider the real-valued phase retrieval problem with objective and gradient
f (x) = 1 4 m i=1 y 2 i -(a ⊤ i x) 2 2 , ∇f (x) = - m i=1 y 2 i -(a ⊤ i x) 2 a i a ⊤ i x.(4)
Here, a i ∈ R n and y i ∈ R for i ∈ N [1,m] are the measurement vectors and the corresponding measurements, respectively. A relaxed smoothness condition for the phase retrieval problem has been explored in [3] based on Bregman distances. The following theorem establishes that whenever the measurement vectors span R n , the objective f also satisfies our Assumption 2.8. Note that the measurement vectors can only span R n if m ≥ n. Moreover, the assumption on spanning R n is mild compared to well-studied conditions that guarantee signal recovery in the phase retrieval problem. These conditions either require randomly sampled measurement vectors with m on the order of n log n [7], or the so-called complement property [1]. The former ensures the spanning property with high probability, while the latter guarantees it deterministically. Theorem 2.12. Consider the phase retrieval problem with objective (4) and suppose that the vectors
{a i } m i=1 span R n . (i) For any L 1 > 0 there exists L 0 > 0 such that f is (L 0 , L 1 )-smooth.
(ii) If ϕ satisfies Assumptions 2.2 and 2.10, then for any L > 0, there exists an L > 0 such that f satisfies the second-order characterization of (L, L)-anisotropic smoothness.
this section cite: ['b2', 'b6', 'b0']

Section: Symmetric matrix factorization
Consider the symmetric matrix factorization problem with objective and gradient
f (U ) = 1 2 ∥U U ⊤ -Y ∥ 2 F , ∇f (U ) = (U U ⊤ -Y )U.(5)
Here, U ∈ R n×r is the optimization variable, and Y ∈ S n×n is a given symmetric matrix. When r < n, minimizing f yields a low-rank approximation of Y with rank at most r. Such low-rank matrix factorizations are fundamental in a variety of applications, most notably in principal component analysis (PCA) [19], where one seeks to capture the most significant directions of variation in the data. More broadly, symmetric matrix factorization plays a central role across various domains: in machine learning, it underlies techniques such as non-negative matrix factorization for parts-based representation learning [23]; in signal processing, it is employed in matrix completion and compressed sensing to reconstruct structured signals from incomplete or noisy measurements [6]. Theorem 2.13. Consider the symmetric matrix factorization problem with objective (5). Then the following statements hold.
(i) For any L 1 > 0 there exists an L 0 > 0 such that f is (L 0 , L 1 )-smooth.
(ii) If ϕ satisfies Assumptions 2.2 and 2.10, then for any L > 0, there exists an L > 0 such that f satisfies the second-order characterization of (L, L)-anisotropic smoothness.
this section cite: ['b18', 'b22', 'b5']

Section: Asymmetric matrix factorization
Consider the regularized asymmetric matrix factorization problem with objective
f (W, H) = 1 2 ∥W H -Y ∥ 2 F + κ 4 ∥W ∥ 4 F + κ 4 ∥H∥ 4 F ,(6)
where W ∈ R m×r and H ∈ R r×n are the optimization variables, Y ∈ R m×n is a given matrix, and κ ≥ 0 is a regularization parameter. When κ = 0 and r < min{m, n}, this reduces to the classical low-rank matrix factorization problem. Additionally, such objectives have been used to model the training of two-layer linear networks, such as in the case of two-layer autoencoders [15]. We note that the results below also hold for regularization terms of the form κ∥W ⊤ W -HH ⊤ ∥ 2 F as described in [11], and highlight the work of [32], which designed a Bregman proximal-gradient method for similar regularized matrix factorization problems. Theorem 2.14. Consider the asymmetric matrix factorization problem with objective (6) and let κ > 0. Then the following statements hold.
(i) For any L 1 > 0 there exists an L 0 > 0 such that f is (L 0 , L 1 )-smooth.
(ii) If ϕ satisfies Assumptions 2.2 and 2.10, then for any L > 0, there exists an L > 0 such that f satisfies the second-order characterization of (L, L)-anisotropic smoothness.
Note that Theorem 2.14 requires κ > 0. To understand why, observe that the gradient of f is given by
∇ W f (W, H) = (W H -Y )H ⊤ +κ∥W ∥ 2 F W, and ∇ H f (W, H) = W ⊤ (W H -Y )+κ∥H∥ 2 F H.
Let x denote the concatenation of the vectors vec(W ) and vec(H), such that ∥x∥ 2 = ∥W ∥ 2 F +∥H∥ 2 F . In contrast to symmetric matrix factorization, if κ = 0, the gradient norm of f can approach zero as ∥x∥ → ∞, whereas Assumption 2.8 requires an asymptotic growth proportional to ∥x∥ 3 . To see this, consider W ⋆ ∈ R m×r , H ⋆ ∈ R r×n such that W ⋆ H ⋆ = Y . In this case, the gradient norm is zero, and rescaling W ⋆ and H ⋆ with a nonsingular matrix D ∈ R r×r , i.e., W = W ⋆ D and H = D -1 H ⋆ , preserves the gradient norm. As a result, one can construct counterexamples where the gradient norm remains zero while ∥D∥ → ∞, and consequently ∥x∥ → ∞.
Finally, we remark that the key step in proving Theorem 2.14 entails lower bounding ∥∇f (W, H)∥ in terms of the variable V := max(∥W ∥ F , ∥H∥ F ), and exploiting that ∥V ∥ → ∞ if and only if ∥x∥ → ∞. It appears that this strategy can be generalized to the factorization of Y into more than two factors, which is relevant for training deep linear networks.
this section cite: ['b14', 'b10', 'b31']

Section: Burer-Monteiro factorizations of MaxCut-type semidefinite programs
Let us consider so-called MaxCut-type semidefinite programs (SDPs) minimize
X∈S n×n -⟨C, X⟩ subject to X ⪰ 0 diag(X) = 1 n ,(7)
where C ∈ S n×n is the cost matrix. The relaxation (7) provides a precise relaxation to the MaxCut problem, a fundamental combinatorial problem arising in graph optimization [16,14]. In an effort to exploit the typical low-rank structure of the solution, a Burer-Monteiro factorization [5] decomposes
X = V V ⊤ for V ∈ R n×r . This yields minimize V ∈R n×r -⟨C, V V ⊤ ⟩ subject to diag(V V ⊤ ) = 1 n .(8)
Choosing r much smaller than n significantly decreases the number of variables from n 2 to nr. However, the downside of this approach is that convexity is lost. Fortunately, under certain conditions every second-order stationary point of this nonconvex problem is a global minimizer [14]. Let us denote by
x i ∈ R r the i'th row of V , such that V ⊤ = [x 1 , x 2 , . . . , x n ].
We also define the vectorized variable x := [x ⊤ 1 , x ⊤ 2 , . . . , x ⊤ n ] ⊤ ∈ R d where d = nr. Then we denote by f (x) the objective of (8) in terms of x, and likewise by A(x) = 0 the constraint of (8) in terms of x.
As proposed in the seminal work [5], this nonconvex constrained problem can be solved with an augmented Lagrangian method (ALM). Each iteration consists of minimizing with respect to the primal variable x the (unconstrained) augmented Lagrangian with penalty parameter β > 0, i.e.,
L β (x, y) = f (x) + ⟨A(x), y⟩ + β 2 ∥A(x)∥ 2 ,(9)
followed by an update of the multipliers y ∈ R n . A similar strategy was also used in [38] for Burer-Monteiro factorizations of clustering SDPs. The following theorem establishes generalized smoothness of the augmented Lagrangian with respect to the primal variable.
Theorem 2.15. Consider the Burer-Monteiro factorization (8) of the MaxCut-type SDP (7) and let L β denote the augmented Lagrangian with penalty parameter β > 0 of this factorized problem. Then, with respect to the primal variable x ∈ R d and for some fixed multiplier y ∈ R n the following statements hold.
(i) For any L 1 > 0 there exists L 0 > 0 such that L β (•, y) is (L 0 , L 1 )-smooth.
(ii) If ϕ satisfies Assumptions 2.2 and 2.10, then for any L > 0, there exists an L > 0 such that L β (•, y) satisfies the second-order characterization of (L, L)-anisotropic smoothness.
this section cite: ['b15', 'b13', 'b4', 'b13', 'b4', 'b37']

Section: Saddle point avoidance of the preconditioned gradient method
The remarkable performance of simple gradient descent-like methods for minimizing nonconvex functions is often attributed to the fact that they avoid strict saddle points of Lipschitz smooth objectives. This section establishes that nonlinear preconditioning of the gradient preserves this desirable property, and in fact generalizes this result to anisotropically smooth functions.
this section cite: []

Section: Asymptotic results based on the stable-center manifold theorem
Denote by X ⋆ the set of strict saddle points of a function f ∈ C 2 , i.e.,
X ⋆ := x ⋆ | ∇f (x ⋆ ) = 0, λ min (∇ 2 f (x ⋆ )) < 0 .
Classical results like [25,24], which are based on the stable-center manifold theorem [39], exploit the fact that the eigenvalues of the Hessian ∇ 2 f are uniformly bounded. In a similar way, for the preconditioned gradient descent method we require that the second-order characterization of (L, L)-anisotropic smoothness holds. By exploiting the fact that ∇ 2 ϕ * (0) = I, we then obtain the following theorem, which generalizes [25, Theorem 4]. Theorem 3.1. Let f ∈ C 2 and suppose that Assumption 2.2 holds. Consider the iterates (x k ) k∈N generated by the preconditioned gradient method, i.e., x k+1 = T γ, L-1 (x k ), where the initial iterate x 0 ∈ R n is chosen uniformly at random. If f satisfies the second-order characterization of (L, L)anisotropic smoothness, and if γ < 1 L , then
P lim k→∞ x k ∈ X ⋆ = 0. (10
)
Assumption 2.2 ensures ϕ * ∈ C 2 , which in turn guarantees that T γ, L-1 ∈ C 1 , as needed for the stable-center manifold theorem [39,24]. Unfortunately, this means that the reference function ϕ(x) = h(∥x∥) with h = 1 2 ∥ • ∥ 2 + δ [-1,1] , which gives rise to a version of the gradient clipping method [36,Example 1.7], is not covered by Theorem 3.1. Indeed, in this case we have h * ′ (y) = Π [-1,1] (y) = max {min {y, 1} , -1}. Note however that this projection is a piecewise affine function, and therefore h * ′ is continuously differentiable almost everywhere, i.e., except at the points y = ±1.
Based on a recent variant of the stable-center manifold theorem [10] we now establish that also the above clipped gradient variant with ϕ * / ∈ C 2 avoids strict saddle points with probability one. In particular, [10, Proposition 2.5] only requires that the iteration map T γ,λ is continuously differentiable on a set of measure one which contains the set of strict saddle points X ⋆ . We thus have to show that (i) ∇ϕ * ( L-1 ∇f (•)) is differentiable almost everywhere; and that (ii) ∇ϕ * (•) is differentiable around the point L-1 ∇f (x ⋆ ) = 0, with x ⋆ ∈ X ⋆ . Remark that the former requires an additional assumption for guaranteeing that ∇f maps a set of measure one onto a set on which ∇ϕ * is differentiable.
Theorem 3.2. Let f ∈ C 2+ and ϕ(x) = h(∥x∥) with h = 1 2 ∥ • ∥ 2 + δ [-1,1] . Consider the iterates (x k ) k∈N generated by the preconditioned gradient method, i.e., x k+1 = T γ, L-1 (x k ) = x k -γ min( 1 /∥∇f(x k )∥, L-1 )∇f (x k
), where the initial iterate x 0 ∈ R n is chosen uniformly at random. Moreover, suppose that the set
U := x ∈ R n | ∥∇f (x)∥ ̸ = L
is a set of measure one. If f satisfies the second-order sufficient condition for (L, L)-anisotropic smoothness, and if γ < 1 L , then
P lim k→∞ x k ∈ X ⋆ = 0.(11)
this section cite: ['b24', 'b23', 'b38', 'b38', 'b23', 'b35', 'b9']

Section: Efficiently avoiding strict saddle points through perturbations
Despite avoiding strict saddle points asymptotically for almost any initialization, gradient descent may actually be significantly slowed down around saddle points. In fact, gradient descent can take exponential time to escape strict saddle points [13], in the sense that the number of iterations depends exponentially on the dimension n of the optimization variable. Yet, by adding small perturbations, this issue can be mitigated, and the complexity of obtaining a second-order stationary point then depends only polylogarithmically on the dimension n [17,18]. This section establishes a similar result for a perturbed preconditioned gradient method.
Existing works analyzing the complexity of gradient descent for converging to a second-order stationary point require not only Lipschitz continuity of the gradients, but also of the Hessian. This is quite restrictive, since for example any (non-degenerate) polynomial of degree more than 2 violates this assumption. Instead, we require Lipschitz continuity of the mapping
H λ (x) := λ -1 J[∇ϕ * (λ∇f (x))] = ∇ 2 ϕ * (λ∇f (x))∇ 2 f (x).
To ensure well-definedness of H λ , Assumption 2.2 is assumed in the remainder of this section.
this section cite: ['b12', 'b16', 'b17']

Section: Assumption 3.3. The mapping H
λ (x) := ∇ 2 ϕ * (λ∇f (x))∇ 2 f (x) is ρ-Lipschitz-continuous, i.e., ∃ρ > 0 : ∥H λ (x) -H λ (y)∥ ≤ ρ∥x -y∥, ∀x, y ∈ R n .
This new condition appears significantly less restrictive, as illustrated by the following example.
Example 3.4. Let f (x) = 1 4 x 4 -1 2 x 2 and ϕ(x) = cosh(|x|) -1. Since arcsinh is an odd function, ∇ϕ * (λf ′ (x)) = arcsinh(|λf ′ (x)|) sgn(λf ′ (x)) = arcsinh(λf ′ (x)) = arcsinh(λ(x 3 -x)). Therefore, we obtain that H λ (x) = λ -1 d(∇ϕ * (λf ′ (x))) dx = (3x 2 -1) √ 1+λ 2 (x 3 -x) 2 .
One easily verifies that H λ ∈ C 1 with bounded derivative, which implies the required Lipschitz-continuity of H λ . In fact, this reasoning generalizes to any univariate polynomial, regardless of its degree.
Under anisotropic smoothness, it is natural to consider λ -1 ϕ(∇ϕ * (λ∇f (x))) as a first-order stationarity measure, and λ min (H λ (x)) as a second-order stationarity measure. Therefore, we say that a point x ∈ R n is an ϵ-second-order stationary point of an (L, L)-anisotropically smooth function f if
λ -1 ϕ(∇ϕ * (λ∇f (x))) ≤ ϵ 2 ,
and
λ min (∇ 2 ϕ * (λ∇f (x))∇ 2 f (x)) ≥ - √ ρϵ.
For ϕ = 1 2 ∥ • ∥ 2 we recover the classical notion of ϵ-second-order stationarity, with ρ the constant of Lipschitz continuity of ∇ 2 f . Algorithm 1 describes a perturbed preconditioned gradient method that closely resembles perturbation schemes presented in [17,18]. In particular, whenever the first-order stationarity is sufficiently small, then a perturbation is added followed by ⌈T ⌉ > 0 unperturbed iterations.
Algorithm 1 Perturbed preconditioned gradient descent REQUIRE: x 0 ∈ R n , γ, λ > 0, perturbation radius r > 0, time interval T > 0, tolerance G > 0 1: k perturb = 0 2: for k = 0, 1, . . . do 3: if λ -1 ϕ(∇ϕ * (λ∇f (x k ))) ≤ G 2 2
and kk perturb > T then 4:
x k ← x k + γξ k , ξ k ∼ B 0 (r) uniformly, k perturb ← k 5: x k+1 = x k -γ∇ϕ * (λ∇f (x k ))
We analyze the complexity of algorithm 1 under the following assumption. Assumption 3.5. Suppose that Assumption 2.2 holds, such that ϕ * ∈ C 2 , and let ϕ(x) = h(∥x∥) where in particular h ∈ C 2 . Moreover, let h(x) ≥ x 2 /2, and h(x) = x 2 /2 + o(x 2 ) as x → 0.
This assumption holds for kernel functions from (2). Remark that there is no real loss of generality by fixing the scale of h around 0, since a rescaled version of h can be obtained by modifying L.
In our analysis, we specify the parameters of algorithm 1 in terms of L, L, ϵ and some χ ≥ 1,
γ = 1 L , λ = 1 L , r = ϵ 400χ 3 , T = L √ ρϵ χ, G = min 1, 1 √ λ r,(12)
and introduce two additional constants that are used only in the analysis, i.e.,
F = 1 50λχ 3 ϵ 3 ρ , Z = 1 4χ ϵ ρ .(13)
We obtain the following complexity of algorithm 1 for converging to a second-order stationary point.
this section cite: ['b16', 'b17']

Section: Theorem 3.6 (Iteration complexity).
Let f be (L, L)-anisotropically relative to ϕ. Moreover, suppose that Assumptions 3.3 and 3.5 hold, and define constants ∆ f ≥ f (x 0 )inf f and χ = log 2
L 2 √ n∆ f c √ ρ Lϵ 5 /2 δ for some c > 0.
There exists a constant c max > 0 such that if c ≤ c max , then for any ϵ > 0 sufficiently small, and for any δ ∈ (0, 1), Algorithm 1 with parameters as in (12) and (13), visits an ϵ-second-order stationary point in at least T /2 iterations with probability at least 1δ, where
T = 8 max (f (x 0 )-inf f )T F , λ (f (x 0 )-inf f ) 2γG 2 = Õ L(f (x 0 )-inf f ) Lϵ 2 .
The Õ notation hides a factor χ 4 which is polylogarithmic in the dimension n and in the tolerance ϵ.
Theorem 3.6 generalizes [18, Theorem 18], and relies on a similar high-level proof strategy, which goes as follows. If the current iterate x is not an ϵ-second-order stationary point, then either λ -1 ϕ(∇ϕ * (λ∇f (x))) is large, or λ min (H λ ) is sufficiently negative. In either case, we establish a significant decrease in function value after at most ⌈T ⌉ iterations of algorithm 1. Since f (x 0 )inf f is bounded, the number of iterates which are not ϵ-second-order stationary can be bounded.
Nevertheless, the generalization of [18,Theorem 18] to the setting of Algorithm 1 is by no means straightforward. The original proofs rely heavily on Lipschitz smoothness, in a way that often does not generalize directly to the anisotropically smooth setting. Here, we highlight two such difficulties. First, consider a point x ∈ R n and the perturbed point x := x + γξ for some perturbation ξ ∈ B 0 (r). Then, by anisotropic smoothness we can upper bound
f (x) -f (x) ≤ γ λ ϕ(ξ + ∇ϕ * (λ∇f (x))).
While Lipschitz-smoothness with ϕ = 1 2 ∥ • ∥foot_2 readily provides an upper bound in terms of ∥ξ∥ 2 ≤ r 2 and ∥∇f (x)∥ 2 , the reference functions are not typically such that an upper bound in terms of ϕ(ξ) + ϕ(∇ϕ * (λ∇f (x))) can be obtained. Second, unlike in the L-Lipschitz-smooth case where ∥∇ 2 f ∥ ≤ L, the norm ∥H λ ∥ cannot be upper bounded uniformly, even under the second-order characterization of (L, L)anisotropic smoothness. The latter only guarantees that λ max (H λ ) ≤ L L, but it does not lower bound λ min (H λ ). And even if the eigenvalues of H λ were bounded in absolute value, this still would not guarantee boundedness of ∥H λ ∥, since H λ is not a normal matrix in general.
this section cite: ['b17']

Section: Numerical validation
Lastly, we illustrate some merits of nonlinear preconditioning, and validate the complexity result of Theorem 3.6 numerically. The source code is publicly available. 2   Nonlinear preconditioning for symmetric matrix factorization For the symmetric matrix factorization problem (5) with n = 2 and r = 1, Fig. 2 presents a 2D visualization of the level curves of the objective, along with the iterates of both vanilla gradient descent (GD) and the preconditioned variant (P-GD) with ϕ(x) = cosh(∥x∥) -1. Unless GD is initialized close to a stationary point, the stepsize must be chosen very small to prevent the iterates from diverging -as expected, because the quartic objective is not Lipschitz smooth. In contrast, the (P-GD) iterations take the form (for L = 1)
x + = x -γ sinh -1 (∥∇f (x)∥) ∥∇f (x)∥ ∇f (x).
In this case, large gradients are damped -recall the close resemblance to clipping methods, cf. Fig. 1 -resulting in the convergence of (P-GD) for stepsizes γ that are often orders of magnitudes larger than the maximum stepsize of GD. In turn, this causes (P-GD) to often require significantly fewer iterations, and overall outperform GD for fixed stepsize. 0 200 400 600 800 1000 Epochs 600 400 200 0 Objective Value n = 5, L = 1 0 200 400 600 800 1000 Epochs 600 400 200 0 Objective Value n = 5, L = 1.5 0 200 400 600 800 1000 Epochs 600 400 200 0 Objective Value n = 5, L = 2 0 200 400 600 800 1000 Epochs 600 400 200 0 Objective Value n = 5, L = 3 0 500 1000 1500 2000 Epochs 1000 500 0 Objective Value n = 10, L = 1 0 500 1000 1500 2000 Epochs 1000 500 0 Objective Value n = 10, L = 1.5 0 500 1000 1500 2000 Epochs 1000 500 0 Objective Value n = 10, L = 2 0 500 1000 1500 2000 Epochs 1000 500 0 Objective Value n = 10, L = 3
Figure 3: Performance of vanilla GD (blue), perturbed vanilla GD [13, Alg 1] (orange), and Algorithm 1 (green) on the 'octopus' function [13].
Fast avoidance of saddle points Fig 3 validates the fast escape of saddle points by Algorithm 1. We consider the 'octopus' objective [13] which was constructed such that GD takes exponential time to escape saddle points. We select all hyperparameters as in [13, §5], and set the only additional hyperparameter L = 1. We compare against vanilla GD and perturbed vanilla GD [13, Alg 1], and vary the constant L ∈ {1, 1.5, 2, 3} and dimension n ∈ {5, 10}, thus creating counterparts to [11, Figs 3 and 4]. We observe that algorithm 1 performs similar to perturbed vanilla GD, and also scales in a similar way with respect to n and L. This validates the complexity result from Theorem 3.6.
this section cite: ['b12', 'b12']

Section: Conclusion
This work introduced a novel sufficient condition unifying (L 0 , L 1 )-smoothness and anisotropic smoothness. We showed that this condition holds in key applications such as phase retrieval, matrix factorization, and Burer-Monteiro factorizations of MaxCut.
We further analyzed the nonlinearly preconditioned gradient method, which naturally aligns with anisotropic smoothness. Notably, we proved that it preserves the saddle point avoidance properties of gradient descent and extends them to anisotropically smooth settings. This contrasts with prior analyses requiring either global Lipschitz smoothness, or local smoothness combined with compactness, both of which are often unmet in practice.
To our knowledge, this is the first work to rigorously establish saddle point avoidance for problems like phase retrieval and matrix factorization under a smoothness condition that is both practical and verifiable. These results strengthen the theoretical foundations of first-order methods for nonconvex optimization and in particular encourage further study of nonlinear gradient preconditioning.
this section cite: []

Section: References
Ref_id:b0 Title: Saving phase: Injectivity and stability for phase retrieval Year: (2014-07)
Ref_id:b1 Title: Nonlinear Programming. 2nd ed Year: (1999)
Ref_id:b2 Title: First order methods beyond convexity and Lipschitz gradient continuity with applications to quadratic inverse problems Year: (2018)
Ref_id:b3 Title: Mirror and Preconditioned Gradient Descent in Wasserstein Space Year: (2024-12)
Ref_id:b4 Title: A nonlinear programming algorithm for solving semidefinite programs via low-rank factorization Year: (2003-02)
Ref_id:b5 Title: Exact Matrix Completion via Convex Optimization Year: (2009-12)
Ref_id:b6 Title: PhaseLift: Exact and Stable Signal Recovery from Magnitude Measurements via Convex Programming Year: (2013)
Ref_id:b7 Title: Efficiently Escaping Saddle Points under Generalized Smoothness via Self-Bounding Regularity Year: (2025-03)
Ref_id:b8 Title: Generalized-Smooth Nonconvex Optimization is As Efficient As Smooth Nonconvex Optimization Year: (2023-06)
Ref_id:b9 Title: Gradient Descent Provably Escapes Saddle Points in the Training of Shallow ReLU Networks Year: (2024-09)
Ref_id:b10 Title: Nonconvex Optimization Meets Low-Rank Matrix Factorization: An Overview Year: (2019-10)
Ref_id:b11 Title: Robustness to Unbounded Smoothness of Generalized SignSGD Year: (2022-12)
Ref_id:b12 Title: Gradient Descent Can Take Exponential Time to Escape Saddle Points Year: (2017)
Ref_id:b13 Title: Benign landscape for Burer-Monteiro factorizations of MaxCut-type semidefinite programs Year: (2025-03)
Ref_id:b14 Title: Taming Nonconvex Stochastic Mirror Descent with General Bregman Divergence Year: (2024-04)
Ref_id:b15 Title: Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming Year: (1995-11)
Ref_id:b16 Title: How to Escape Saddle Points Efficiently Year: (2017-07)
Ref_id:b17 Title: On Nonconvex Optimization for Machine Learning: Gradients, Stochasticity, and Saddle Points Year: (2021-02)
Ref_id:b18 Title: Principal Component Analysis Year: (2002)
Ref_id:b19 Title: Lower envelopes and lifting for structured nonconvex optimization Year: (2021)
Ref_id:b20 Title: Anisotropic proximal gradient Year: (2025-04)
Ref_id:b21 Title: Dualities for Non-Euclidean Smoothness and Strong Convexity under the Light of Generalized Conjugacy Year: (2023-12)
Ref_id:b22 Title: Learning the parts of objects by non-negative matrix factorization Year: (1999-10)
Ref_id:b23 Title: First-order methods almost always avoid strict saddle points Year: (2019-07)
Ref_id:b24 Title: Gradient Descent Only Converges to Minimizers Year: (2016-06)
Ref_id:b25 Title: Gradient descent with a general cost Year: (2023-06)
Ref_id:b26 Title: The Power of Normalization: Faster Evasion of Saddle Points Year: (2016-11)
Ref_id:b27 Title: Convex and Non-convex Optimization Under Generalized Smoothness Year: (2023-12)
Ref_id:b28 Title: Convergence of Adam Under Relaxed Assumptions Year: (2023-12)
Ref_id:b29 Title: Relatively Smooth Convex Optimization by First-Order Methods, and Applications Year: (2018-01)
Ref_id:b30 Title: Dual Space Preconditioning for Gradient Descent Year: (2021-01)
Ref_id:b31 Title: Beyond Alternating Updates for Matrix Factorization with Inertial Bregman Proximal Gradient Algorithms Year: (2019)
Ref_id:b32 Title: Revisiting Normalized Gradient Descent: Fast Evasion of Saddle Points Year: (2019-11)
Ref_id:b33 Title: Lectures on Convex Optimization Year: (2018)
Ref_id:b34 Title: Forward-backward splitting under the light of generalized convexity Year: (2025-03)
Ref_id:b35 Title: Nonlinearly Preconditioned Gradient Methods under Generalized Smoothness Year: (2025-02)
Ref_id:b36 Title: Wets. Variational Analysis Year: (1998)
Ref_id:b37 Title: An Inexact Augmented Lagrangian Framework for Nonconvex Optimization with Nonlinear Constraints Year: (2019)
Ref_id:b38 Title: Global Stability of Dynamical Systems Year: (1987)
Ref_id:b39 Title: Attention is All you Need Year: (2017)
Ref_id:b40 Title: Improved Analysis of Clipping Algorithms for Non-convex Optimization Year: (2020)
Ref_id:b41 Title: Why gradient clipping accelerates training: A theoretical justification for adaptivity Year: (2020-02)
Ref_id:b42 Title: On the convergence and improvement of stochastic normalized gradient descent Year: (2021-02)
