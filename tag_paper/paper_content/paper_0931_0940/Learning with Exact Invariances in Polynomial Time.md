Title: Learning with Exact Invariances in Polynomial Time
Abstract: We study the statistical-computational trade-offs for learning with exact invariances (or symmetries) using kernel regression. Traditional methods, such as data augmentation, group averaging, canonicalization, and frame-averaging, either fail to provide a polynomial-time solution or are not applicable in the kernel setting. However, with oracle access to the geometric properties of the input space, we propose a polynomial-time algorithm that learns a classifier with exact invariances. Moreover, our approach achieves the same excess population risk (or generalization error) as the original kernel regression problem. To the best of our knowledge, this is the first polynomialtime algorithm to achieve exact (as opposed to approximate) invariances in this setting. In developing our approach, we also resolve a question recently posed by Díaz et al. (2025) on efficient computation of invariant bases and kernels with respect to finite groups, even when the group size is prohibitively large. Our proof leverages tools from differential geometry, spectral theory, and optimization. A key result in our development is a new reformulation of the problem of learning under invariances as optimizing an infinite number of linearly constrained convex quadratic programs, which may be of independent interest.

Section: Introduction
While humans can readily observe symmetries or invariances in systems, it is generally challenging for machines to detect and exploit these properties from data. The objective of machine learning with invariances is to develop approaches that enable models to be trained and utilized under the symmetries inherent in the data. This framework is broadly applicable across various domains in the natural sciences and physics, including atomistic systems (Grisafi et al., 2018), molecular wavefunctions and electronic densities (Unke et al., 2021), interatomic potentials (Batzner et al., 2022), and beyond (Batzner et al., 2023). While many applications involve Euclidean symmetries (Smidt, 2021), the scope of such methods extends well beyond them to other geometries (Bronstein et al., 2017).
Learning with invariances has a longstanding history in machine learning (Hinton, 1987;Kondor, 2008). In recent years, there has been significant interest in the development and analysis of learning methods that account for various types of invariances. This surge in interest is strongly motivated by many models showing considerable success in practice. Empirical evidence suggests the existence of algorithms that can effectively learn under invariances while exhibiting strong generalization and computational efficiency. However, from a theoretical perspective, much of the focus has been on the expressive power of models, generalization bounds, and sample complexity. There remains a relative lack of understanding regarding the statisticalcomputational trade-offs in learning under invariances, even in foundational settings such as kernel regression.
Symmetries can be incorporated into learning in multiple ways. An immediate solution for learning with invariances seems to be data augmentation over the elements of the group. Moreover, some approaches to learning with invariances rely on group averaging, a technique that involves summing over group elements. However, the typically large size of the group can make both of these approaches computationally prohibitive, even super-exponential in the dimension of input data. Alternative approaches, such as canonicalization and frame averaging, also suffer from issues like discontinuities and scalability challenges (Dym et al., 2024).
This paper seeks to address the following question:
Can we obtain an invariant estimator for learning with invariances that achieves both strong generalization and computational efficiency?
The first contribution of this work is a detailed study of the problem of learning with invariances in the context of kernel methods. Kernels, which have been among popular learning approaches, offer both statistical and computational efficiency (Scholkopf & Smola, 2002). We argue that while group averaging fails to produce exactly invariant estimators within a computationally efficient time frame, alternative algorithms can generate invariant estimators for the kernel regression problem in time that is polylogarithmic in the size of the group. In other words, we demonstrate that it is possible to achieve an invariant estimator that is both computationally efficient and exhibits strong generalization. At first glance, this result may seem counterintuitive and even impossible, since it implies that enumerating all possible invariances is not required to design statistically efficient learning algorithms. This provides theoretical support for the empirical observation that computational efficiency and strong generalization are attainable in learning with invariances. To the best of our knowledge, this is the first algorithm that is both statistically and computationally efficient for learning with invariances in the kernel setting, Learning with invariances can be formulated as a nonconvex optimization problem, which is not tractable directly. To design an efficient algorithm, we leverage the spectral theory of the Laplace-Beltrami operator on manifolds. Notably, since this operator commutes with all (isometric) group actions on the manifold, it is possible to find an orthonormal basis of Laplacian eigenfunctions such that each group action on the manifold acts on the eigenspaces of the Laplacian via orthogonal matrices. This theoretical framework allows us to reformulate the original problem of learning with invariances on manifolds as an infinite collection of finite-dimensional convex quadratic programs-one for each eigenspace-each constrained by linear conditions. By truncating the number of quadratic programs solved, we can efficiently approximate solutions to the primary nonconvex optimization problem, thereby approximating kernel solutions to the problem of learning with invariances. This reformulation not only enables us to derive a polynomialtime algorithm for kernel regression under invariances, but it may also have broader applications, the exploration of which we defer to future research.
Finally, we emphasize again that this work is centered on achieving exact invariance, as many applications-especially neural networks with strong empirical performance-are explicitly designed to incorporate exact invariances by construction. In summary, this paper makes the following contributions:
• We initiate the exploration of statistical-computational trade-offs in the context of learning with exact invariances, focusing specifically on kernel regression.
• We reformulate the problem of learning under invariances in kernel methods, leveraging differential geom-etry and spectral theory, and cast it as infinitely many convex quadratic programs with linear constraints, for which we derive an efficient solution in terms of time complexity. We trade off computational and statistical complexity by controlling the number of convex quadratic programs solved to obtain the estimator.
• We introduce the first polynomial algorithm for learning with invariances in the general setting of kernel regression over manifolds.
this section cite: ['b22', 'b2', 'b3', 'b49', 'b8', 'b29', 'b48']

Section: Related Work
Generalization bounds and sample complexity for learning with invariances have been extensively studied, particularly in the context of invariant kernels. Works such as Elesedy (2021), Bietti et al. (2021), Tahmasebi & Jegelka (2023), and Mei et al. (2021) provide insights into this area. Additionally, studies on equivariant kernels (Elesedy & Zaidi, 2021;Petrache & Trivedi, 2023) further our understanding of how equivariances affect learning. PAC-Bayesian methods have also been applied to derive generalization bounds under equivariances (Behboodi et al., 2022). More recently, Kiani et al. (2024) explored the complexity of learning under symmetry constraints for gradient-based algorithms. For studies on the optimization of kernels under invariances, see Teo et al. (2007).
A variety of methods have been proposed to enhance the performance of kernel-based learning models. One prominent approach is the use of random feature models (Rahimi & Recht, 2007), which approximate kernels using randomly selected features. Low-rank kernel approximation techniques, such as the Nyström method (Williams & Seeger, 2000;Drineas et al., 2005), have also been proposed to reduce the computational complexity of kernel methods; see also Bach (2013); Cesa-Bianchi et al. (2015). Divide-and-conquer algorithms offer another pontential avenue for kernel approximation (Zhang et al., 2013). Additionally, the impact of kernel approximation on learning accuracy is well-documented in Cortes et al. (2010).
Our work focuses on learning with invariances, which differs significantly from the tasks of learning invariances and measuring them in neural networks. For example, Benton et al. (2020) address how neural networks can learn invariances, while Goodfellow et al. (2009) study methods to measure the degree of invariance in network architectures. Moreover, in this paper, we assume that the invariances in the formulation are given a priori, meaning that the problem of identifying the appropriate group of invariances in the data-and/or verifying the assumption of invariance-is to be treated as a separate and independent task. For further discussion on testing and learning invariances from data, see Soleymani et al. (2025).
Invariance in kernel methods is not limited to group averag-ing. Other approaches such as frame averaging (Puny et al., 2022), canonicalization (Kaba et al., 2023;Ma et al., 2024), random projections (Dym & Gortler, 2024), and parameter sharing (Ravanbakhsh et al., 2017) have also been proposed to construct invariant function classes. However, canonicalization and frame averaging face challenges, particularly concerning continuity, which has been addressed in recent works like Dym et al. (2024).
In specialized tasks such as graphs, image, and pointcloud data, Graph Neural Networks (GNNs) (Scarselli et al., 2008;Xu et al., 2019), Convolutional Neural Networks (CNNs) (Krizhevsky et al., 2012;Li et al., 2021), and Pointnet (Qi et al., 2017a;b) have demonstrated the effectiveness of leveraging symmetries. Symmetries have also been successfully integrated into generative models (Biloš & Günnemann, 2021;Niu et al., 2020;Köhler et al., 2020). For a broader discussion on various types of invariances and their applications across machine learning tasks, see Bronstein et al. (2017).
this section cite: ['b36', 'b19', 'b41', 'b4', 'b27', 'b45', 'b15', 'b0', 'b10', 'b13', 'b5', 'b21', 'b51', 'b42', 'b26', 'b34', 'b16', 'b46', 'b47', 'b30', 'b33', 'b18', 'b38', 'b28', 'b8']

Section: Background and Problem Statement
Notation. We begin by establishing some frequently used notation. Let M be a smooth, compact, and boundaryless d-dimensional Riemannian manifold. The uniform distribution over the manifold is the normalized volume element corresponding to its metric. We denote the space of squareintegrable functions over M by Lfoot_1 (M) and the space of continuous functions by C(M). Furthermore, H s (M) represents the Sobolev space of functions on M with parameter s, defined as the set of functions with square-integrable derivatives up to order s. Larger values of s correspond to greater smoothness, and it holds that H s (M) ⊆ C(M) if and only if s > d/2, a condition we will assume throughout this paper. For each n ∈ N, we define [n] := {1, 2, . . . , n}.
We use log to denote the logarithm with base 2. We refer to Appendices A.1 and A.2 for a quick review of Riemannian manifolds.
Problem statement. We consider a general learning setup on a smooth, compact, and boundaryless Riemannian manifold M of dimension d. Our objective is to identify an estimator f ∈ F from a feasible space of estimators F ⊆ L 2 (M), based on n independent and uniformlyfoot_0 distributed labeled samples S = {(x i , y i ) : i ∈ [n]} ⊆ (M × R) n drawn from the manifold. Here, the labels y i for i ∈ [n] are produced based on the (unknown) ground truth regression function f ⋆ ∈ C(M), meaning that y i = f ⋆ (x i ) + ϵ i , for each i ∈ [n], where ϵ i , i ∈ [n], is a sequence of independent zero-mean random variables with variance bounded by σ 2 . The population risk (or generalization error) of an estimator f ∈ L 2 (M), which quantifies the quality of the estimation, is defined as:
R( f ) := E ∥ f -f ⋆ ∥ 2 L 2 (M) ,
where the expectation is taken over the randomness of the data and labels.
Given a dataset of size n, finding estimators with minimal population risk can be quite complex, often requiring the resolution of non-convex optimization objectives. However, in scenarios where f ⋆ ∈ H, with H ⊆ L 2 (M) being a Reproducing Kernel Hilbert Space (RKHS), it is feasible to compute kernel-based estimators with low risk efficiently. Specifically, the Kernel Ridge Regression (KRR) estimator for the RKHS H = H s (M), denoted as f KRR , achieves a population risk of R( f KRR ) = O n -s/(s+d/2) while being computable in time O(nfoot_2 ), assuming access to an oracle that computes the kernel associated with the space (Bach, 2024). Note that Sobolev spaces H s (M) with s > d/2 are RKHS. We refer the reader to Appendices A.8 and A.9 for a detailed review of the KRR estimator and related topics on Sobolev spaces.
Learning with invariances. We assume that a finite group G acts smoothly and isometrically 2 on the manifold M, represented by a smooth function θ : G × M → M mapping the product manifold G × M to M. We employ the notation θ(g, x) as gx for any g ∈ G and x ∈ M. In a scenario of learning under invariances, the regression function f ⋆ is invariant under the action of the group G, satisfying f ⋆ (gx) = f ⋆ (x) for each g ∈ G and x ∈ M. Thus, learning under invariances introduces an additional requirement: not only must we compute an estimator with minimal population risk efficiently, but f must also be invariant with respect to G. This additional condition is often satisfied in neural network applications by constructing networks that are invariant by design, such as graph neural networks.
In the context of learning with Sobolev kernels, the KRR estimator f KRR is not G-invariant (see Appendix A for more details). Consequently, the KRR estimator cannot provide a solution for learning under invariances. However, with a shift-invariant Positive Definite Symmetric (PDS) kernel 3 K : M×M → R, one can utilize group averaging to derive a new kernel and a new RKHS holding only G-invariant functions:
K inv x 1 , x 2 := 1 |G| g∈G K gx 1 , x 2 .
Given that the Sobolev space H s (M) adopts a shiftinvariant PDS kernel (Appendix A.8), one can apply the above method to construct and compute a G-invariant kernel (assuming access to evaluating its original kernel). This indicates that the KRR estimator on K inv yields an invariant estimator for f ⋆ with a desirable population risk (see (Tahmasebi & Jegelka, 2023) for a comprehensive study).
However, in terms of computational complexity, this method requires Ω(n 2 |G|) time to compute the new kernel between pairs of input data. In many practical scenarios, |G| can be intolerably large. For instance, for the permutation group
P d , we have |G| = d! ∼ √ d( d e ) d
which is super-exponential in d. Consequently, the group averaging method cannot provide an efficient algorithm for learning with exact invariances. We emphasize "exact invariance" here, as the sum involved in K inv can be approximated by summing over a number of random group transformations. However, this does not guarantee exact invariance, which is the primary goal of this paper.
Other traditional approaches for achieving learning under invariances include data augmentation, canonicalization, and frame averaging. For data augmentation, we need to increase our dataset size by a multiplicative factor of |G|, which is often impractical within efficient time constraints. This is because, for any datapoint x i ∈ S, data augmentation requires adding a new datapoint gx i for each group element g ∈ G to ensure invariance of the underlying learning procedure in a black-box manner, leading to a complexity of Ω(n|G|). Canonicalization involves mapping data onto the quotient space of the group action and subsequently finding an estimator (e.g., a KRR estimator) on the reduced input space. However, this method is also infeasible for kernels due to the unavoidable discontinuities and non-smoothness of the canonicalization maps, which violate RKHS requirements (Dym et al., 2024). Finally, frame averaging is analogous to canonicalization, but it remains unclear how to address continuity issues for efficient frame sizes. Moreover, it requires careful design of frames tailored to the specific problem at hand, making it unsuitable for a generalpurpose algorithm. Thus, motivated by these observations, we pose the following question: Is it possible to obtain a G-invariant estimator for f ⋆ ∈ H s (M) with a desirable population risk (similar to the case without invariances) in poly(n, d, log(|G|)) time?
In the next section, we answer this question affirmatively. This is surprising, as it suggests that enumerating the set G is not required to find statistically efficient G-invariant estimators.
Oracles. To characterize computational complexity, first we need to specify the type of oracle access provided for the estimation. Before doing so, we briefly review the spectral theory of the Laplace-Beltrami operator on manifolds. For further details, we refer the reader to Appendix A.
The Laplace-Beltrami operator generalizes the Laplacian operator to Riemannian manifolds. It has a basis of smooth eigenfunctions ϕ λ,ℓ ∈ L 2 (M), which serve as an orthonormal basis for L 2 (M). The index λ represents the eigenvalue corresponding to the eigenfunction ϕ λ,ℓ , and ℓ ∈ [m λ ] runs over the multiplicity of λ, denoted by m λ . The eigenvalues can be ordered such that
0 = λ 0 < λ 1 ≤ λ 2 ≤ • • • → ∞.
For example, in the case of the sphere S d-1 , the spherical harmonics, which are homogeneous harmonic polynomials, are a natural choice of eigenfunctions.
The sequence of eigenfunctions and their corresponding eigenvalues provide critical information about the geometry of the manifold. In this work, we make use of the following two types of oracles:
• The ability to evaluate any eigenfunction ϕ λ,ℓ (x) at a given point x ∈ M.
• The ability to compute the L 2 (M) inner product between a shifted eigenfunction ϕ λ,ℓ (gx) and another eigenfunction ϕ λ,ℓ ′ (x) for any group element g ∈ G.
For both oracles, we assume free access as long as D λ := λ ′ ≤λ m λ ′ = poly(n, d), where D λ denotes the number of eigenfunctions with eigenvalues less than or equal to λ. This assumption is motivated by the case of S d-1 , where spherical harmonics can be efficiently evaluated or multiplied in low dimensions (in such cases, only a few monomials need to be processed, making the task simplefoot_3 ). The first oracle handles the geometric structure of the manifold, while the second oracle captures the relationship between the group action and the manifold's spectrum.
this section cite: ['b1']

Section: Main Result
In this section, we address the question raised in the previous section by presenting the primary result of the paper, which is encapsulated in the following theorem.
Theorem 1 (Learning with exact invariances in polynomial time). Consider the problem of learning with invariances with respect to a finite group G using a labeled dataset of size n sampled from a manifold of dimension d. Assume that the optimal regression function belongs to the Sobolev space of functions of order s, i.e., f ⋆ ∈ H s (M) for some s > d/2 and let α := 2s/d. Then, there exists an algorithm that, given the data, produces an exactly invariant estimator f such that: 1+α) oracle calls to construct the estimator;
• It runs in time O log 3 (|G|)n 3/(1+α) + n (2+α)/(1+α) ; • It achieves an excess population risk (or generalization error) of R( f ) = O n -s/(s+d/2) ; • It requires O log(|G|)n 2/(1+α) +n (2+α)/(
• For any x ∈ M, the estimator f (x) can be computed in time O n 1/(1+α) using O n 1/(1+α) oracle calls.
The full proof of Theorem 1 is presented in Appendix B.3, while a detailed proof sketch is provided in Section 5, and the algorithm is outlined in Algorithm 1.
Let us interpret the above theorem. Note that without any invariances, the Kernel Ridge Regression (KRR) estimator (details are given in Appendix A.9) provides an estimator f KRR for the Sobolev space H s (M) that is computed in time O n 3 and achieves the risk R( f KRR ) = O n -s/(s+d/2) , which is minimax optimal. Here, while KRR cannot guarantee an exactly invariant estimator, we propose another estimator which is both exactly invariant and also converges with the same rate O n -s/(s+d/2) . As a result, we achieve exact invariances with statistically desirable risk (or sample complexity). In other words, the population risk is the same as the optimal case without invariances, which shows that the algorithm introduces no loss in statistical performance while enforcing group invariances.
We thus come to the following conclusion:
The problem of learning with exact invariances can be efficiently solved in time poly(n, d, log(|G|)) and with excess population risk (or generalization error) O n -s/(s+d/2) which is the same statistical performance as for learning without invariances.
It is worth mentioning that, according to the theorem, the proposed estimator f is not only efficiently achievable but also efficiently computes new predictions on unlabeled data.
Remark 4.1. We notice that in the proof of Theorem 1, the actual time and sample complexity depends only on the size of the minimum generating set of the group G, denoted by ρ(G), instead of log(|G|). We use the logarithm in the theorem just to make the improvement from the naive approach clearer. The actual proof allows to achieve the tighter result with ρ(G) ≤ log(|G|), which holds for any finite group (see Proposition B.1 in Appendix B.1). Note that for some cases (such as cyclic groups) ρ(G) ≪ log(|G|).
this section cite: []

Section: Algorithm and Proof Sketch
In this section, we provide a proof sketch for Theorem 1, introducing several new notations and concepts necessary for achieving the reduction in time complexity.
We begin with the most natural optimization program for obtaining an estimator: the Empirical Risk Minimization (ERM), which proposes the following estimator:
f ERM := arg min f ∈H s (M) 1 n n i=1 (f (x i ) -y i ) 2 ,
where
S = {(x i , y i ) : i ∈ [n]} ⊆ (M × R) n denotes the sampled (labeled) dataset.
However, as discussed, this method does not necessarily produce an estimator that is exactly invariant. A natural idea is to introduce group invariances as constraints into the above optimization, leading to the following constrained ERM solution:
f ERM-C := arg min f ∈H s (M) 1 n n i=1 (f (x i ) -y i ) 2 s.t. ∀(g, x) ∈ G × M : f (gx) = f (x).
While this formulation ensures exact invariance, it introduces |G| functional equations. This is problematic for two reasons: first, |G| constraints are prohibitively many, and second, these constraints require solving functional equalities, which are not easily achievable. Moreover, the functional equations involve non-linear (pointwise) constraints on the estimator function, which at first glance appear intractable due to nonconvexity of the contraints f (gx) = f (x) for general choice of g.
Therefore, it is necessary to reformulate the above optimization program. The goals of the reformulation are to reduce the number of constraints and encode the functional equations into more tractable constraints, ideally linear ones.
Reducing the number of constraints. We begin by using the following basic property (based on the group law):
∀g ∈ {g 1 , g 2 },∀x ∈ M : f (gx) = f (x) =⇒ ∀x ∈ M : f (g 1 g 2 x) = f (x) .
This observation allows us to eliminate many unnecessary constraints. Specifically, we only need constraints over a small subset of G if this subset can generate any group element through arbitrary group multiplications. To formalize this, we introduce the following definition: Definition 5.1. A finite group G is said to be generated by a subset S ⊆ G if for every g ∈ G, there exists a sequence of elements s 1 , s 2 , . . . , s k such that for each i ∈ [k], either
s i ∈ S or s -1 i ∈ S and g = s 1 s 2 • • • s k . The minimum size of such a subset S is denoted by ρ(G).
Clearly, ρ(G) ≤ |G|. However, it can be shown (see Appendix B.1) that ρ(G) ≤ log(|G|), which represents an exponential improvement over the trivial upper bound.
Thus, we can reformulate the constrained ERM as:
f ERM-C := arg min f ∈H s (M) 1 n n i=1 (f (x i ) -y i ) 2 s.t. ∀(g, x) ∈ S × M : f (gx) = f (x),
where |S| ≤ log(|G|). This way we reduce the number of constraints from |G| to log(|G|) by leveraging the concept of minimal group generators. Note that this fact cannot be directly used in data augmentation, group averaging, or canonicalization techniques.
Optimization in the spectral domain. The constrained ERM formulation presented above, while advantageous in terms of reducing the number of constraints, involves optimizing over the infinite-dimensional space H s (M), which is computationally intractable. One way to make this problem tractable is to parametrize the estimator and search for the optimal parameters. To achieve this, we utilize the spectral theory of the Laplace-Beltrami operator over manifolds. While a detailed discussion of spectral theory is provided in Appendix A, we summarize the relevant concepts here.
As mentioned earlier, the Laplace-Beltrami operator yields a sequence of orthonormal eigenfunctions ϕ λ,ℓ ∈ L 2 (M), where λ ∈ {λ 0 , λ 1 , . . .} ⊆ [0, ∞) represents the eigenvalue corresponding to the eigenfunction ϕ λ,ℓ , and ℓ ∈ [m λ ] indexes the multiplicity of λ, denoted by m λ . Therefore, any estimator f ∈ L 2 (M) can be expressed as:
f (x) = λ m λ ℓ=1 f λ,ℓ ϕ λ,ℓ (x), f λ,ℓ := ⟨f, ϕ λ,ℓ ⟩ L 2 (M) .
The idea is to parametrize the problem by finding the best coefficients f λ,ℓ . However, since there are infinitely many eigenvalues, there are infinitely many parameters to estimate, which is not feasible in finite time. Fortunately, we know that f ⋆ ∈ H s (M). From the definition of Sobolev spaces (see Appendix A.8), we have:
∥f ⋆ ∥ 2 H s (M) := λ m λ ℓ=1 (f ⋆ λ,ℓ ) 2 D α λ ,
where D λ = λ ′ ≤λ m λ ′ , and α := 2s d > 1. Thus, we conclude that:
λ:D λ >D m λ ℓ=1 (f ⋆ λ,ℓ ) 2 ≤ D -α ∥f ⋆ ∥ 2 H s (M) = O(D -α ),
for any D > 0. This shows that for Sobolev regression functions f ⋆ ∈ H s (M), we can truncate the estimation of coefficients at a certain cutoff frequency λ, which allows the problem to be parametrized with finitely many parameters.
Although this introduces bias into the estimation (since higher-frequency eigenfunctions will not be captured), the bias is bounded by the above inequality for Sobolev spaces.
Interestingly, this spectral approach yields a more meaningful optimization problem when considering the population risk function rather than ERM. The population risk, which is the primary objective in regression, is given by:
R(f ) = E S ∥f -f ⋆ ∥ 2 L 2 (M) = λ m λ ℓ=1 E[(f λ,ℓ -f ⋆ λ,ℓ ) 2 ].
Constrained spectral method. To review, we introduced an efficient way to impose the constraints related to group invariances in the ERM objective and later presented spectral methods for obtaining estimators. The last step here is to combine these to achieve exact invariances via a constrained spectral method. We use an important property of the Laplace-Beltrami operator to introduce the algorithm.
Let ∆ M denote the Laplace-Beltrami operator on the manifold M, and let G be a group acting isometrically on M.
Define the linear operator T g : f (x) → f (gx) for each group element g ∈ G and any smooth function f on the manifold. Then, we have
∆ M (T g ϕ) = T g (∆ M (ϕ)),
for any smooth function ϕ on the manifold (for a formal proof, please refer to Appendix A.5).
This identity tells us that the Laplace-Beltrami operator ∆ M commutes with the operator T g for each g. Since both operators are linear, spectral theory implies that the commutativity shows the eigenspaces of ∆ M are preserved under the action of the group G, meaning the operators can be simultaneously diagonalized. Specifically, for any λ, ℓ, and any g ∈ G, the function ϕ λ,ℓ (gx) is a linear combination of eigenfunctions ϕ λ,ℓ ′ , ℓ ′ ∈ [m λ ]. In particular, the group G acts via orthogonal matrices on the eigenspace V λ := span(ϕ λ,ℓ : ℓ ∈ [m λ ]) for each λ.
Let D λ (g) denote the m λ × m λ orthogonal matrix corresponding to the action of an element g ∈ G on V λ for each λ. Then, a function
f (x) = λ m λ ℓ=1 f λ,ℓ ϕ λ,ℓ (x)
is G-invariant if and only if
D λ (g)f λ = f λ , ∀g ∈ G ∀λ ∈ {λ 0 , λ 1 , . . .},
where f λ := (f λ,ℓ ) ℓ∈[m λ ] ∈ R m λ for each λ. We can further reduce the number of conditions by passing G to a generator set, which gives only log(|G|) conditions.
Thus, the commutativity of the Laplace-Beltrami operator and any isometric group action allows us to introduce only linear constraints on the spectral method to achieve exact invariances. This leads to the following optimization program:
min f λ,ℓ λ m λ ℓ=1 E[(f λ,ℓ -f ⋆ λ,ℓ ) 2 ], s.t. ∀g ∈ S ∀λ ∈ {λ 0 , λ 1 , . . .} : D λ (g)f λ = f λ .
Here,
f ⋆ λ,ℓ = E x [f ⋆ (x)ϕ λ,ℓ (x)] = E x,y [yϕ λ,ℓ (x)]
is not known a priori; only n samples (x i , y i ) ∈ M × R, i ∈ [n], are given. Furthermore, the constraints are independent for different eigenspaces (i.e., different λ), and the objective is a sum over eigenspaces. This means we can decompose the problem into a set of linearly constrained optimization programs, one for each eigenspace V λ :
min f λ,ℓ m λ ℓ=1 E[(f λ,ℓ -f ⋆ λ,ℓ ) 2 ], s.t. ∀g ∈ S : D λ (g)f λ = f λ .
This reformulation allows us to propose efficient estimators for the problem.
Empirical estimator. In this paper, we suggest the following auxiliary empirical mean estimator from the data for the above optimization program on V λ :
f λ,ℓ = 1 n n i=1 y i ϕ λ,ℓ (x i ), ∀ℓ ∈ [m λ ].(1)
Moreover, we stop estimation and set f λ,ℓ = 0 when D λ > D, where D is a hyperparameter. To obtain a Ginvariant estimator from our primary estimator, we solve the following quadratic program to find a solution satisfying the constraints for each V λ with D λ ≤ D:
f λ,ℓ := arg min f λ,ℓ m λ ℓ=1 (f λ,ℓ -f λ,ℓ ) 2 , s.t. ∀g ∈ S : D λ (g)f λ = f λ .
This optimization problem is a convex quadratic program with linear constraints that can be solved iteratively using the rich convex optimization machinery. Additionally, it has a closed-form solution as noted in Proposition B.2 in Appendix B.2. Let B λ ∈ R |S|m λ ×m λ be defined as the augmented matrix resulting from concatenating D λ (g) -I for all g ∈ S, i.e.,
B λ = [(D λ (g 1 ) -I) ⊤ , (D λ (g 2 ) - I) ⊤ , . . . , (D λ (g |S| ) -I) ⊤ ] ⊤ . Then, f λ,ℓ = f λ,ℓ -B λ ⊤ (B λ B λ ⊤ ) † (B λ f λ )[ℓ],
where † denotes Moore-Penrose inverse.
The final estimator of the algorithm is given by
f (x) = λ:D λ ≤D m λ ℓ=1 f λ,ℓ ϕ λ,ℓ (x).
This meta approach to design a G-invariant estimator f from any primary estimtor f is novel and may be of independent interest. Pseudocode for the method is presented in Algorithm 1. Since the invariance is imposed in the spectral representation, we refer to our proposed algorithm as Spectral Averaging (Spec-Avg).
this section cite: []

Section: Algorithm 1 Learning with Exact Invariances by Spectral Averaging (Spec-Avg)
Input: S = {(x i , y i ) : i ∈ [n]} and α = 2s/d ∈ (1, ∞). Output: f (x).
1: Initialize D ← n 1/(1+α) . 2: for each λ such that D λ ≤ D do 3: for each ℓ ∈ [m λ ] do 4: f λ,ℓ ← 1 n n i=1 y i ϕ λ,ℓ (x i ).
5:
end for 6: end for 7: for each λ such that D λ ≤ D do 8:
Solve the following linearly constrained quadratic program over m λ variables:
f λ,ℓ ← arg min f λ,ℓ m λ ℓ=1 (f λ,ℓ -f λ,ℓ ) 2 , s.t. ∀g ∈ S : D λ (g)f λ = f λ . 9: end for 10: Return: f (x) = λ:D λ ≤D m λ ℓ=1 f λ,ℓ ϕ λ,ℓ (x).
We conclude this section by reviewing how we apply the results from Algorithm 1 to the two following important examples.
Example 5.2. Consider the problem of learning under invariances over the unit sphere S d-1 := {x ∈ R d : ∥x∥ 2 = 1}, where the group G is the group of all permutations of coordinates. Note that |G| = d!, which is prohibitively large for data augmentation or group averaging. However, this group is generated by only two elements: σ 1 = (1 2) and σ 2 = (1 2 . . . d). Here, σ 1 swaps the first and second coordinates, while σ 2 is a cycle that maps 1 → 2, 2 → 3, and so on, cyling with d → 1.
The eigenspaces V λ for the sphere are precisely the sets of homogeneous harmonic polynomials of degree k, where λ = k(k + d -2). The permutation group acts on V λ by permuting the variables of the polynomials. This action is clearly linear, and the matrices D λ (g) can be efficiently computed (using tensor products) as long as k is small. Moreover, homogeneous polynomials of degree k can also be computed efficiently for small k. This shows that the oracles considered in this paper align perfectly with what we observe in the important case of spheres and polynomial regression. In Algorithm 1, we first compute the coefficients of each polynomial for degree k, up to a small k, and then solve a quadratic program with only two linear constraints to obtain an exactly invariant polynomial solution.
Example 5.3. Consider the same setup as the previous example but assume d = 2, i.e., the manifold is the unit circle. In this case, each eigenspace V λ is spanned by sin(kθ) and cos(kθ), where λ = k 2 . Let us assume our task is to find an estimator invariant with respect to rotations by integer multiples of 2π  |G| . This group is cyclic and is generated by only one element g 0 = 2π
|G| . Thus, we have only one constraint for each eigenspace. Indeed, one can observe that D λ (g 0 ) = R(k 2π
|G| ), where R(.) ∈ R 2×2 is the twodimensional rotation matrix. Thus, this example further illustrates how our oracles are defined to solve the problem.
this section cite: []

Section: Experiments
In this section, we provide complementary experiments to support our theoretical results. We first show that, in practice, Kernel Ridge Regression (KRR) is not a Ginvariant estimator. Then, we demonstrate that our algorithm (Spec-Avg) achieves the same rate of population risk as KRR, while enjoying exact invariance properties.
this section cite: []

Section: Problem Statement
We consider the input space (manifold) T d = [-1, 1) d , which represents a flat d-dimensional torus. Additionally, we consider the group of sign-invariances G = {±1} d , acting on this space via coordinate-wise sign inversions. The dataset is generated as n independent and identically distributed (i.i.d.) samples drawn uniformly from this space, with the target function defined as:
f * (x) = 1 d d i=1 ix 2 i . Clearly, this function is invariant w.r.t. group action G.
To analyze estimation via kernels in this setup, we consider a periodic kernel on the torus T d , specifically the von Mises Kernel (von Mises, 1918;Mardia & Jupp, 2009), defined as: K η (x, y) = exp (η cos(π(x -y))) , where η is a positive parameter associated with kernel bandwidth. This kernel function is particularly useful for circular and directional statistics.
Moreover, the kernel admits the following sign-invariant eigenfunctions: ϕ ℓ1,ℓ2,...,ℓ d (x) = d i=1 cos(πℓ i x i ), where ℓ i ∈ N ∪ {0}. The corresponding eigenvalues can be computed as λ = π d i=1 ℓ 2  i , derived from the partial differential equation ∆ϕ ℓ1,ℓ2,...,ℓ d + λϕ ℓ1,ℓ2,...,ℓ d = 0. This formu-lation facilitates the analysis of KRR and Spec-Avg under symmetry constraints, ensuring their compatibility with the underlying group structure. It is worth noting that, in this setting, |G| = 2 d . Consequently, group averaging is computationally inefficient due to the exponential growth of the group size with the dimensionality d.
this section cite: ['b35']

Section: Settings
We conduct our experiments for d = 10. The trained models are evaluated on a test dataset of size 100. Both the test and train datasets are generated uniformly from the interval [-1, 1] d , independently and identically distributed. Each point in our plots represents an average over 10 different random seeds (from 1 to 10) to account for the randomness in the data generation process.
this section cite: []

Section: Results
The results of the experiments are depicted in Figure 1 and Figure 2 in Appendix C. While our algorithm (Spec-Avg) is G-invariant by construction, there is no theoretical guarantee for Kernel Ridge Regression (KRR) to be G-invariant. In Figure 1 in Appendix C, we demonstrate that this is indeed the case in practice, as the estimator KRR is not G-invariant. We define the following measure of Invariance Discrepancy:
ID( f ) def = sup x∈X ,g∈G | f (x) -f (gx)|,
where f is the estimator. We report this value for KRR across different choices of the regularization parameter λ. It is worth noting that ID( f ) is zero for the Spec-Avg estimator, as it is G-invariant by design.
In Figure 2 in Appendix C, we present the empirical excess population risk of KRR and Spec-Avg for different hyperparameters λ and D, respectively. As expected, it is demonstrated that with an appropriate choice of hyperparameters, KRR and Spec-Avg achieve the same order of test error. Higher values of the regularization parameter λ for KRR correspond to lower values of the sparsity parameter D for Spec-Avg, both of which act as mechanisms for regularizing the norm of the estimator. It can be observed that Spec-Avg with D = 176 achieves the same order of performance as KRR with λ = 50.
this section cite: []

Section: Conclusion
In this paper, we explore the statistical-computational tradeoffs in learning with invariances, focusing on kernel regression. We observe that while the Kernel Ridge Regression (KRR) estimator can address this problem, it is not invariant without group averaging, and since group averaging is costly for large groups, we ask whether it is possible to develop statistically sound estimators with efficient time complexity.
Our findings show that by reformulating the problem and reducing the number of constraints using group laws, we can express it as solving an infinite series of quadratic optimization programs under linear constraints. We conclude with an algorithm that achieves an exactly invariant estimator with polynomial time complexity and highlight several additional questions for future research.
this section cite: []

Section: Discussion and Future Directions
We initiated the study on computational-statistical tradeoffs in learning with exact invariances. We designed an algorithm that shows achieving the desirable population risk (the same as kernel regression without invariances) in poly(n, d, log(|G|)) time for the task of kernel regression with invariances on general manifolds. We note that, for simplicity, we have focused on boundaryless manifolds and isometric group actions. However, using standard techniques, the theory can be extended to more general cases as wellfoot_4 . While the proposed spectral algorithm is computationally efficient, it does not offer any improvement in sample complexity over the baseline R( f ) = O n -s/(s+d/2) . It has been observed that without computational constraints, better convergence rates are possible for learning with invariances (Tahmasebi & Jegelka, 2023), which are minimax optimal. Thus, it remains open whether those improved rates are achievable in poly(n, d, log(|G|)) time.
We note that the oracle access we assumed is primarily motivated by the case of the sphere, where polynomials can be evaluated, multiplied, composed by group elements, and integrated efficiently when they are of relatively low degree. We believe this is the most natural oracle access for this problem, as it aligns well with applications involving polynomials. An interesting future work could be to investigate the statistical-computational trade-offs using alternative oracles, e.g., similar to the kernel trick, how to design computationally efficient algorithms that have only access to the inner product of the RKHS. Another interesting future direction is to find whether random feature models as approximations for kernels can significantly improve the statistical-computational trade-off of learning with invariances. At present, our theory does not apply to random feature models.
We also observe that the spectral algorithm used in this paper does not employ the kernel trick, as it requires access to the entire set of features, rather than just their inner products. An interesting question is whether it is possible to utilize kernel tricks and find an alternative (polynomial-time) algorithm for learning under invariances. This approach could potentially improve the statistical efficiency of the spectral algorithm. In the end, we would like to note that captur-
Tahmasebi, B. and Jegelka, S. The exact sample complexity gain from invariances for kernel regression. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 2, 3, 4, 9, 15, 16 Tahmasebi, B. and Jegelka, S. Sample complexity bounds for estimating probability divergences under invariances. In Int. Conference on Machine Learning (ICML), 2024. 9 Teo, C., Globerson, A., Roweis, S., and Smola, A. Convex learning with invariances. In Advances in Neural Information Processing Systems (NeurIPS), 2007. 2 Unke, O., Bogojeski, M., Gastegger, M., Geiger, M., Smidt, T., and Müller, K.-R. Se(3)-equivariant prediction of molecular wavefunctions and electronic densities. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 1 von Mises, R. Über die "ganzzahligkeit" der atomgewichten und ververwandte fragen. Physical Journal, 19:490, 1918. 8 Williams, C. and Seeger, M. Using the nyström method to speed up kernel machines. In Advances in Neural Information Processing Systems (NeurIPS), 2000. 2 Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful are graph neural networks? In Int. Conference on Learning Representations (ICLR), 2019. 3 Zhang, Y., Duchi, J., and Wainwright, M. Divide and conquer kernel ridge regression. In Conference on Learning Theory (COLT), 2013. 2 A. Background A.1. Riemannian Manifolds
In this section, we review some fundamental definitions from differential geometry and refer the reader to Lee (2006); Petersen (2006); Lee (2012) for further details.
Definition A.1 (Manifold). A topological manifold M of dimension dim(M) is a completely separable Hausdorff space that is locally homeomorphic to an open subset of Euclidean space of the same dimension, specifically R dim(M) . More formally, for each point x ∈ M, there exists an open neighborhood U ⊆ M and a homeomorphism ϕ : U → U , where M) .
U ⊆ R dim(
The value dim(M) is referred to as the dimension of the manifold. Examples of manifolds include tori, spheres, R d , and graphs of continuous functions. Manifolds with boundaries differ from boundaryless manifolds in that they may have neighborhoods that locally resemble open subsets of closed dim(M)-dimensional upper half-spaces, denoted as H dim(M) ⊆ R dim(M) , defined as follows:
H d = {(x 1 , x 2 , . . . , x d ) ∈ R d | x d ≥ 0}.
Definition A.2 (Local Coordinates). Given a chart (U, ϕ)-a pair consisting of a local neighborhood U and the corresponding homeomorphism ϕ : U → U -on a manifold M with dimension d, we define local coordinates (x 1 , x 2 , . . . , x d ) such that
ϕ(p) = (x 1 (p), x 2 (p), . . . , x d (p)),
for each point p ∈ U .
Definition A.3 (Tangent Space). At each point x ∈ M, the tangent space T x M is defined as the vector space formed by the tangent vectors to the manifold M at x. A tangent vector v ∈ T x M can be represented as the derivative of a smooth curve γ(t) : (-ϵ, ϵ) → M defined on the manifold with the property that γ(0) = x. It is expressed as
ν = d dt γ(t) t=0 .
The tangent space T x M is a real vector space with dimension dim(M).
Definition A.4 (Riemannian Metric Tensor). A Riemannian metric tensor gfoot_5 on a manifold M is a smooth inner product defined on the tangent space T x M at each point x ∈ M. For any two tangent vectors u, v ∈ T x M, the metric assigns a real number g x (u, v) ∈ R.
Definition A.5 (Riemannian Manifold). A Riemannian manifold is defined as a pair (M, g), where M is a manifold and g is a Riemannian metric tensor defined on the tangent space T x M at each point x ∈ M.
A Riemannian metric tensor provides essential tools for the study of manifolds, which we formalize below. It enables the following:
• the definition of the geodesic distance d(x, y) between any two points x, y ∈ M on the manifold,
• a volume element d vol g (x) over the manifold, serving as the measure for the Borel sigma-algebra over open subsets of the manifold M, and
• the measurement of the angle between any two tangent vectors u, v ∈ T x M, which in turn provides the size of tangent vectors.
Definition A.6 (Geodesic Distance). The geodesic distance d M (x, y) between any two points x, y ∈ M on the manifold is defined as the infimum length among all smooth curves γ : [0, 1] → M connecting x to y (γ(0) = x, γ(1) = y). The length of a curve γ is defined as
L(γ) = 1 0 g γ(t) ( γ, γ) dt,
where γ denotes the derivative dγ dt .
Definition A.7 (Volume Element). The volume element dvol g (x) on a Riemannian manifold (M, g) is defined as
dvol g = det(g ij ) dx 1 ∧ dx 2 ∧ • • • ∧ dx n ,
where g ij are the components of the Riemannian metric tensor, (x 1 , x 2 , . . . , x n ) are the local coordinates, and ∧ denotes the exterior product.
The volume element provides a way to compute the volume of subsets of M by integrating functions over M. Moreover, a Borel measure µ on open subsets of M can be derived form the volume element to form probability measure space, e.g., uniformly over the manifold.
Definition A.8 (Smooth Map). A maping f : M → N is a smooth map if for any charts (U, ϕ) on M, and (V, ψ) on N , the composition function N ) is infinitely differentiable.
ψ • f • ϕ -1 : R dim(M) → R dim(
Definition A.9 (Pullback of the metric tensor). Given Riemannian manifolds M, (N , g) and φ : M → N a smooth map between them. The pullback of the metric tensor g by ϕ, denoted by φ * g is the Riemannian metric tensor on manifold M defined by,
(φ * g) x (u, v) = g φ(x) (dφ x (u), dφ x (v)
), for all points x ∈ M and all u, v ∈ T x M, where dφ x : T x M → T φ(x) N is the differential of the map φ at point x.
Thus, the pullback metric φ * g on M captures the relation between tangent vectors of M in terms of how they are mapped to the manifold N via φ.
Definition A.10 (Connected Manifold). A manifold M is connected if for any two points x, x ′ ∈ M, there is a smooth curve γ : [0, 1] → M such that γ(0) = x and γ(1) = x ′ .
Throughout this paper, we focus on smooth, connected, compact and boundaryless Riemannian manifolds (M, g) unless stated otherwise. For a Riemannian manifolds (M, g), we denoted the dot product induced by the metric tensor g as ⟨u, v⟩ gx = g x (u, v) for all u, v ∈ T x . We drop the subscript x whenever it is clear from the context.
this section cite: ['b32', 'b40', 'b31']

Section: A.2. Functional Spaces over Manifolds
Now equipped with probability measures on manifold discussed in Appendix A.1, we are ready to define functional spaces L p (M) and Sobolev spaces H s (M) on manifold M analogously to their Euclidean counterparts in the following, Definition A.11 (Functional Spaces on Manifolds). The Lebesgue functional spaces L p (M) for p ∈ [1, ∞], and the Sobolev spaces H s (M) for s ≥ 0 on a smooth Manifold M, are defined as follows:
• The Lebesgue space L p (M) consists of measurable functions f : M → R such that ∥f ∥ L p (M) < ∞ where,
∥f ∥ L p (M) = M |f (x)| p dµ(x) 1/p if p ∈ [1, ∞) ess sup x∈M |f (x)| < ∞. if p = ∞ ,
where µ is the uniform measure over the manifold M.
• The Sobolev space H s (M) consists of measurable functions whose derivatives up to order s are in L 2 (M), i.e.,
H s (M) = f ∈ L 2 (M) | D α f ∈ L 2 (M) for all multi-indices α with |α| ≤ s .
this section cite: []

Section: A.3. Lie Group of Isometry Maps
In this section, first we state basic definition of isometric mappings over manifolds and then wrap up by characterizing the isometry group over the manifold.
Definition A.12 (Isometry Map). A bijective mapping τ : M → M is an isometry on the manifold (G, g) if d(τ (x), τ (x ′ )) = d(x, x ′ ).
We also state a brief definition of Lie groups for completeness.
Definition A.13 (Lie group). A group G is a Lie group with smooth group operations (multiplication and inversion) if it is additionally a smooth manifold.
The space of bijective Riemannian isometries defined on the manifold (M, g), denoted by ISO(M, g) constitutes a group with composition operation. The celebrated Myers-Steenrod theorem states that any isometry map τ ∈ ISO(M, g) between connected manifolds is an isometry (Myers & Steenrod, 1939;Palais, 1957). Myers & Steenrod (1939) took it a step further and proved that isometry group of a Riemannian manifold (M, g) is a Lie group.
Alternatively, ISO(M, g) can be charecterized by the pullback of the metric tensor. In terms, τ ∈ ISO(M, g) if and only if g = τ * g (Petersen, 2006).
this section cite: ['b37', 'b39', 'b37', 'b40']

Section: A.4. Laplacian on Manifolds
In this section, we reiterate over definition of Laplace-Beltrami operator on manifolds (which is the generalization of the Laplacian operator
∆ = ∂ 2 1 + ∂ 2 2 + • • • + ∂ 2 d defined on the Euclidean space R d
) and state a several interesting properties that will utilize later. We refer to Chavel (1984) for additional details.
Definition A.14 (Laplace-Beltrami operator). Given a Riemannian manifold (M, g), the Laplace-Beltrami operator ∆ g : H s (M) → H s-2 (M) acts on a smooth function f : M → R by ∆ g f = div g (grad g (f )).
Moreover, ∆ g f has an equivalent weak formulation (Evans, 2022), as the unique continuous linear operator
∆ g : H s (M) → H s-2 (M) which is a solution to the equation, M ψ(x)∆ g ϕ(x)d vol g (x) + M ⟨∇ g ψ(x), ∇ g ϕ(x)⟩ g d vol g (x) = 0, ∀ϕ, ψ ∈ H s (M).(2)
The Laplace-Beltrami operator ∆ g is self-adjoint, eliptic and diagonalizable in L p (M) (Chavel, 1984;Evans, 2022), yielding a sequence of orthonormal eigenfunctions ϕ λ,ℓ ∈ L 2 (M), where λ ∈ {λ 0 , λ 1 , . . .} ⊆ [0, ∞) represents the eigenvalue corresponding to the eigenfunction ϕ λ,ℓ , and ℓ ∈ [m λ ] indexes the multiplicity of λ, denoted by m λ such that ∆ g ϕ λi,ℓ + λ ℓ ϕ λi,ℓ = 0 for all ℓ ∈ {1, . . . , m λi }. Note that the basis starts with the constant function ϕ 0 ≡ 1 and λ 0 = 0. Hence, one can write
∆ g f = - ∞ i=0 m λ i ℓ=1 λ i ⟨f, ϕ λi,ℓ ⟩ϕ λi,ℓ . Lemma A.15. For any function f ∈ L 2 (M), such that f is decomposed into the basis {ϕ λ,ℓ } ∞ λ=1 as f = ∞ i=0 m λ i ℓ=1 ⟨f, ϕ λi,ℓ ⟩ L 2 (M) ϕ λ,ℓ , we know that ∥∇ g f ∥ 2 L 2 (M) = ∞ i=0 m λ i ℓ=1 λ i ⟨f, ϕ λi,ℓ ⟩ 2 L 2 (M) ,
for convergent summations.
Proof. By Equation ( 2),
∥∇ g f ∥ 2 L 2 (M) = M ⟨∇ g f (x), ∇ g f (x)⟩ g d vol g (x) = - M f (x)∆ g f (x) d vol g (x) = ∞ i=0 m λ i ℓ=1 λ i ⟨f, ϕ λi,ℓ ⟩ 2 L 2 (M) .
this section cite: ['b11', 'b20', 'b11', 'b20']

Section: A.5. Commutativity of Laplacian and Isometric Group Actions
Let G be a group acting isometrically on a compact, smooth, boundaryless manifold M. As we stated in the main body of the paper, we have ∆ M (T g ϕ) = T g (∆ M (ϕ)) for each smooth function ϕ on manifold M, where T g ϕ = ϕ(gx). A complete proof of this claim can be found in Canzani (2013). Note that by Equation ( 2), this fact is equivalent to
M h∆ M (T g ϕ)d vol g (x) = M hT g (∆ M (ϕ))d vol g (x),(3)
for each smooth function h on manifold M.
this section cite: ['b9']

Section: A.6. Weyl's Law under Invariances
Weyl's law characterizes the asymptotic distribution of the eigenvalues in a closed-form formula (Hörmander, 1968;Sogge, 1988;Canzani, 2013). Let us denote dimension of the space spanned by the eigenvectors corresponding to eigenvalue of the Laplace-Beltrami operator up to λ as
D λ := λ ′ ≤λ m λ ′ .
Theorem A.16 (Weyl's law (Hörmander, 1968;Sogge, 1988;Canzani, 2013)). Let (M, g) be a compact, boundaryless d-dimensional Riemannian manifold. The asymptotic behavior of dimension count D λ follows
D λ = ω d vol(M) (2π
) d λ d/2 + O(λ (d-1)/2 ),
where
ω d = π d/2 Γ( d 2 +1)
is the volume of the unit d-dimensional ball in R d , vol(M) is the Riemannian volume of M, and O(λ (d-1)/2 ) represents the error term.
Define D λ,G as the dimension of the space induced by projection of the corresponding eigenspaces of D λ into the space of G-invariant functions. Tahmasebi & Jegelka (2023) proved the following characterization over this dimension as λ → ∞.
Theorem A.17 (Dimension counting (Tahmasebi & Jegelka, 2023)). Let (M, g) be a compact, boundaryless d-dimensional Riemannian manifold, and G be a compact finite Lie group acting isometrically on (M, g). Then.
D λ,G = ω d vol(M/G) (2π
) d λ d/2 + O(λ (d-1)/2 ),
as λ → ∞, where again ω d is the volume of the unit d-dimensional ball in R d .
this section cite: ['b25', 'b50', 'b9', 'b25', 'b50', 'b9']

Section: A.7. Sobolev Spaces on Manifolds
The ordinary definition of Sobolev spaces on manifolds deals with having square-integrable derivatives up to an order s.
Here, since our focus is on the spectral approach, we present the spectral definition of Sobolev spaces.
Definition A.18 (Sobolev spaces). The Sobolev space of functions H s (M) on a compact, smooth, boundaryless Riemannian manifold M is defined as:
H s (M) := f = λ m λ ℓ=1 f λ,ℓ ϕ λ,ℓ (x) : ∥f ∥ 2 H s (M) := λ m λ ℓ=1 D α λ f 2 λ,ℓ < ∞ ,
where α := 2s/d.
Note that the above definition is equivalent to the other definition of the Sobolev spaces that involves considering λ s instead of D α λ above. Using Weyl's law (see Appendix A.6), one can show that both definitions are equivalent.
A.8. Sobolev Kernels Sobolev spaces are RKHS when s > d/2. Indeed, the Sobolev kernel can be defined as:
K H s (M) (x, y) := λ m λ ℓ=1 D -α λ ϕ λ,ℓ (x)ϕ λ,ℓ (y).
Note that any group G that acts isometrically on the manifold, also acts on the eigenspace of Laplacian via orthogonal matrices. Since orthogonal matrices preserve the inner product we conclude that
K H s (M) (gx, gy) = K H s (M) (x, y),
for any g ∈ G, which means that the Sobolev kernel is shift-invariant. However, this is clearly not G-invariant since it produces small bump functions, which need not be invariant.
this section cite: []

Section: A.9. Kernel Ridge Regresion (KRR)
Consider a Positive-Definite Symmetric (PDS) kernel K(., .) on a smooth, compact, boundaryless manifold with H denoting its RKHS. The objective of Kernel Ridge Regression (KRR) estimator is to introduce the RKHS norm to the ERM objective to make sure of finding smooth interpolators:
min f ∈H 1 n n i=1 (f (x i ) -y i ) 2 + η∥f ∥ 2 H ,(4)
where η denotes the regularization parameter that balances the bias and variance terms. Here, the objective function takes a closed-form solution to the represented theorem for kernels. This gives an efficient estimator, which is termed KRR in the literature.
However, this estimator need not be G-invariant even when trained on invariant data. To see why, note that as long as the space H includes non-invariant functions, there is a chances that we find a non-invariant function optimizing the above objective due to the observation noise. Thus, the only way to make sure that the KRR estimator is G-invariant is to impose the assumption of having G-invariant kernels, which translated to group averaging over the Sobolev kernel:
K G H s (M) (x, y) := 1 |G| g∈G K H s (M) (gx, y). (5
)
This method is unfortunately not computationally feasible, even though it achieves minimax optimal generalization bounds for learning under invariances with kernels (Tahmasebi & Jegelka, 2023).
this section cite: []

Section: B. Proofs

this section cite: []

Section: B.1. Minimal generating set of a Group
Here, we restate and prove the following lemma on the size of the minimal generating set in group theory for completeness. Proof. Consider the minimal generating set
S = {g 1 , g 2 , . . . , g |S| } of the finite group G. For each k ∈ {1, 2, . . . , |S|}, define G k = ⟨g 1 , g 2 , . . . , g k ⟩.
The identity e is not equal to any of g k , and hence cannot be a member of any G n , since it can always be produced by combining an element with its inverse. Moreover, for all k ∈ {1, 2, . . . , |S|}, we know that
g k+1 / ∈ G k , since otherwise ⟨g 1 , g 2 , . . . , g k , g k+2 , . . . g |S| ⟩ = G which contradicts the minimality of the generating set S for the group G. Therefore, g n+1 G n the left coset of G n is disjoint from G n . Additionally, by definition, we know that g n+1 G n G n ⊆ G n+1 . Hence, |G n+1 | ≥ |g n+1 G n | + |G n | = 2|G n |. By induction, 2 |S| = 2 |S| |G 1 | ≤ |G |S| | = |G| which establishes the claim.
this section cite: []

Section: B.2. Constrained Optimization
In this section, we preset a detailed analysis of the constrained quadratic optimization problem that is used in Algorithm 1.
this section cite: []

Section: Proposition B.2 (Projection into invariant subspace of eigenspaces).
The optimization problem,
f λ := arg min f λ m λ ℓ=1 (f λ,ℓ -f λ,ℓ ) 2 ,(6)
s.t. ∀g ∈ S : D λ (g)f λ = f λ ,
with |S| = m, has a closed form solution,
f λ = f λ -B λ ⊤ (B λ B λ ⊤ ) † (B λ f λ ),where
B λ =      D λ (g 1 ) -I D λ (g 2 ) -I . . . D λ (g m ) -I     
, and † denotes Moore-Penrose inverse.
Proof. For better readability, we define B(g i ) := D λ (g i ) -I, where I ∈ R m λ ×m λ is the identity matrix of size m λ , then,
B λ =      B(g 1 ) B(g 2 ) . . . B(g m )      .
For ease of notation, let a := f λ ∈ R m λ and a * := f λ ∈ R m λ , then the optimization problem (6) can be rewritten as,
a * = min a ′ 1 2 ∥a -a ′ ∥ 2 subject to B λ a ′ = 0.(7)
Now, we need to show that the projection of a onto the subspace defined by {a ′ | Ba ′ = 0} has the following analytical form,
a * = a -B λ ⊤ (B λ B λ ⊤ ) † (B λ a).
We form the Lagrangian,
L(a ′ , λ) = 1 2 ∥a -a ′ ∥ 2 + ξ ⊤ B λ a,
where ξ ∈ R m λ is the vector of Lagrange multipliers. By taking gradients,
∂L ∂a ′ = (a ′ -a) + B λ ⊤ ξ = 0, thus, a * = a -B λ ⊤ ξ. (8) Substituting back into the constraint B λ a * = 0, B λ (a -B λ ⊤ ξ) = B λ a -B λ B λ ⊤ ξ = 0.(9)
Hence, ξ satisfies the above linear system which may have infinite number of solutions. We claim that the choice of ξ * = (B λ B λ ⊤ ) † B λ a leads to the optimal solution of optimization problem (7). The objective of optimization ( 7) is ∥a -a * ∥ 2 = ∥B λ ⊤ ξ∥ 2 2 . Any solution ξ to the linear system (9), can be decomposed as
ξ = ξ * + ξ 0 where ξ 0 is in the nullspace of B λ B λ ⊤ . Hence, ∥B λ ⊤ ξ∥ 2 2 = ∥B λ ⊤ (ξ * + ξ 0 )∥ 2 2 (I) = ∥B λ ⊤ ξ * ∥ 2 2 + ∥B λ ⊤ ξ 0 ∥ 2 2 ≥ ∥B λ ⊤ ξ * ∥ 2 2 . (I) follows since B λ ⊤ ξ * and B λ ⊤ ξ 0 are orthogonal w.r.t. each other. Placing ξ * = (B λ B λ ⊤ ) † B λ a in Equation (8)
concludes the proof.
Remark B.3 (Time complexity of optimization in each eigenspace). We arbitrarily chose to use the closed-form solution of the optimization problem (6) instead of iterative approaches. In the closed form solution, we need to calculate the pseuedoinverse of matrix B λ B λ ⊤ ∈ R |S|m λ ×|S|m λ which can be done via singular value decomposition (SVD) in O(|S| 3 m 3 λ ). The other operations are matrix multiplications that are dominated by this part in terms of computational complexity.
this section cite: []

Section: B.3. Main Theorem
Theorem 1 (Learning with exact invariances in polynomial time). Consider the problem of learning with invariances with respect to a finite group G using a labeled dataset of size n sampled from a manifold of dimension d. Assume that the optimal regression function belongs to the Sobolev space of functions of order s, i.e., f ⋆ ∈ H s (M) for some s > d/2 and let α := 2s/d. Then, there exists an algorithm that, given the data, produces an exactly invariant estimator f such that:
• It runs in time O log 3 (|G|)n 3/(1+α) + n (2+α)/(1+α) ;
• It achieves an excess population risk (or generalization error) of R( f ) = O n -s/(s+d/2) ;
• It requires O log(|G|)n 2/(1+α) + n (2+α)/(1+α) oracle calls to construct the estimator;
• For any x ∈ M, the estimator f (x) can be computed in time O n 1/(1+α) using O n 1/(1+α) oracle calls.
Proof. To prove Theorem 1, we use Algorithm 1. Let us start by calculating the time and oracle complexity of the algorithm. Given a dataset S of size n, we first compute
f λ,ℓ = 1 n n i=1 y i ϕ λ,ℓ (x i ),(10)
for each λ such that D λ ≤ D = n 1/(1+α) , and each ℓ ∈ [m λ ]. This requires O(n 1+1/( 1+α) ) oracle calls and can be accomplished in time O(n 1+1/(1+α) ).
Next, we solve the following constrained quadratic program:
f λ,ℓ ← arg min f λ,ℓ m λ ℓ=1 (f λ,ℓ -f λ,ℓ ) 2 ,(11)
s.t. ∀g ∈ S : D λ (g)f λ = f λ .(12)
This is done for each λ such that D λ ≤ n 1/(1+α) . Note that to even set up this program, we need O(|S|m 2 λ ) oracle calls to find the constraints. We have
(D λ (g)) ℓ,ℓ ′ = ⟨ϕ λ,ℓ (x), ϕ λ,ℓ ′ (gx)⟩ L 2 (M)(13)
for each ℓ, ℓ ′ ∈ [m λ ].
Therefore, the total oracle complexity of the proposed algorithm is
O   λ:D λ ≤n 1/(1+α) |S|m 2 λ + n (2+α)/(1+α)   .(14)
We have already shown in Proposition B.1 that one can use a generator set with ρ(G) ≤ log(|G|). Moreover, note that
λ:D λ ≤n 1/(1+α) m 2 λ = O(n 2/(1+α) ).(15)
Therefore, the oracle complexity is
O log(|G|)n 2/(1+α) + n (2+α)/(1+α) .(16)
Let us now calculate the time complexity of finding the estimator. We have already established that we can compute the empirical estimation in time O(n 1+1/(1+α) ). Next, we need to solve the constrained quadratic program with log(|G|) constraints and m λ variables for each λ such that D λ ≤ n 1/(1+α) . Using the proposed algorithm in Appendix B.2 and also Remark B.3, we can solve each of these constrained quadratic programs in time O(log 3 (|G|)m 3 λ ). Therefore, the total time complexity of this step is bounded by
O   λ:D λ ≤n 1/(1+α) log 3 (|G|)m 3 λ   = O log 3 (|G|)n 3/(1+α) .(17)
This proves that the total time complexity of Algorithm 1 is
O log 3 (|G|)n 3/(1+α) + n (2+α)/(1+α) .(18)
Finally, note that given f , one can evaluate it on new unlabeled data x ∈ M using the formula:
f (x) = λ:D λ ≤D m λ ℓ=1 f λ,ℓ ϕ λ,ℓ (x),(19)
with D = n 1/(1+α) , which requires both time and oracle complexity of O(n 1/(1+α) ).
To complete the proof, we need to study the convergence of the population risk of the proposed estimator. We first note that
R( f ) = E[∥ f -f ⋆ ∥ 2 L 2 (M) ] ≤ 2E[∥ f -f ⋆ ≤D ∥ 2 L 2 (M) ] + 2E[∥f ⋆ >D ∥ 2 L 2 (M) ],(20)
where f ⋆ ≤D denotes the orthogonal projection of the function f ⋆ onto the space of eigenfunctions with eigenvalues satisfying D λ ≤ D. Moreover, f ⋆ >D = f ⋆ -f ⋆ ≤D . First, let us upper bound the second term. Note that, according to the assumption, f ⋆ ∈ H s (M). Thus,
E[∥f ⋆ >D ∥ 2 L 2 (M) ] = λ:D λ >D m λ ℓ=1 (f ⋆ λ,ℓ ) 2(21)
= λ:D λ >D m λ ℓ=1 D -α λ D α λ (f ⋆ λ,ℓ ) 2(22)
≤ D -α λ:D λ >D m λ ℓ=1 D α λ (f ⋆ λ,ℓ ) 2(23)
≤ D -α λ m λ ℓ=1 D α λ (f ⋆ λ,ℓ ) 2(24)
= D -α ∥f ⋆ ∥ 2 H s (M) .(25)
Now we focus on the first term. Note that
E[∥ f -f ⋆ ≤D ∥ 2 L 2 (M) ] = λ:D λ ≤D m λ ℓ=1 E[| f λ,ℓ -f ⋆ λ,ℓ | 2 ].(26)
According to the definition, we have
f ⋆ λ,ℓ = E x [f ⋆ (x)ϕ λ,ℓ (x)] = E x,y [yϕ λ,ℓ (x)],(27)
for each λ, ℓ. Moreover, f λ,ℓ is the empirical estimation obtained from data:
f λ,ℓ = 1 n n i=1 y i ϕ λ,ℓ (x i ).(28)
Thus, we obtain
E[| f λ,ℓ -f ⋆ λ,ℓ | 2 ] = 1 n E |yϕ λ,ℓ (x) -E[yϕ λ,ℓ (x)]| 2 (29) = 1 n E |ϵϕ λ,ℓ (x) + f ⋆ (x)ϕ λ,ℓ (x) -E[f ⋆ (x)ϕ λ,ℓ (x)]| 2 (30) = 1 n σ 2 E[ϕ 2 λ,ℓ ] + E |f ⋆ (x)ϕ λ,ℓ (x) -E[f ⋆ (x)ϕ λ,ℓ (x)]| 2(31)
≤ 1 n σ 2 + E[f ⋆ (x) 2 ϕ 2 λ,ℓ (x)](32)
≤ 1 n σ 2 + ∥f ⋆ ∥ 2 L ∞ (M) ,(33)
where we used the orthonormality of the eigenfunctions ϕ λ,ℓ . Then, summing this up to dimension D gives:
E[∥ f -f ⋆ ≤D ∥ 2 L 2 (M) ] ≤ D n σ 2 + ∥f ⋆ ∥ 2 L ∞ (M) .(34)
Note that, by definition, f = P G f , where P G : L 2 (X ) → L 2 (X ) is the orthogonal projection operator onto the invariant functions. Therefore, we have
E[∥ f -f ⋆ ≤D ∥ 2 L 2 (M) ] = E[∥P G f -f ⋆ ≤D ∥ 2 L 2 (M) ](35)
= E[∥P G f -P G f ⋆ ≤D ∥ 2 L 2 (M) ](36)
≤ E[∥ f -f ⋆ ≤D ∥ 2 L 2 (M) ](37)
≤ D n σ 2 + ∥f ⋆ ∥ 2 L ∞ (M) ,(38)
where the penultimate step follows from P G f ⋆ ≤D = f ⋆ ≤D . Therefore, we can combine the two terms to derive the following population risk bound:
R( f ) = E[∥ f -f ⋆ ∥ 2 L 2 (M) ] ≤ D n σ 2 + ∥f ⋆ ∥ 2 L ∞ (M) + D -α ∥f ⋆ ∥ 2 H s (M) .(39)
We can now specify the above bound to D = n 1/(1+α) , which is used in the algorithm, to get:
R( f ) = E[∥ f -f ⋆ ∥ 2 L 2 (M) ] ≤ n -α/(1+α) σ 2 + ∥f ⋆ ∥ 2 L ∞ (M) + n -α/(1+α) ∥f ⋆ ∥ 2 H s (M) ,(40)
which is equivalent to
R( f ) = O(n -α/(1+α) ).(41)
This completes the proof. Remark B.4. Note that other choices of D may or may not yield better bounds depending on the sparsity of the solution. For this sparsity-unaware upper bound that we use, such a choice of D is optimal. Additionally, since we focus on polynomial time algorithms, we cannot choose exponentially large D even if they deliver gains in sample complexity.
this section cite: []

Section: C. Experimental Results
The plots for the experiments discussed in Section 6 are presented below.  Their almost linear behavior implies that they are polynomial functions of the number of training samples with comparable orders. We note that each point in the plot represents an average over 10 different random seeds.
this section cite: []

Section: Impact Statement
The primary focus of this work is on theoretical problems in machine learning, specifically learning with invariances. As such, it does not have direct societal implications or ethical concerns.
this section cite: []

Section: References
Ref_id:b0 Title: Sharp analysis of low-rank kernel matrix approximations Year: (2013)
Ref_id:b1 Title: Learning theory from first principles Year: (2024)
Ref_id:b2 Title: E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials Year: (2022)
Ref_id:b3 Title: Advancing molecular simulation with equivariant interatomic potentials Year: (2023)
Ref_id:b4 Title: A pacbayesian generalization bound for equivariant networks Year: (2022)
Ref_id:b5 Title: Learning invariances in neural networks from training data Year: (2020)
Ref_id:b6 Title: On the sample complexity of learning under geometric stability Year: ()
Ref_id:b7 Title: Scalable normalizing flows for permutation invariant densities Year: ()
Ref_id:b8 Title: Geometric deep learning: going beyond euclidean data Year: (2017)
Ref_id:b9 Title: Analysis on manifolds via the laplacian Year: (2013)
Ref_id:b10 Title: On the complexity of learning with kernels Year: (2015)
Ref_id:b11 Title: Eigenvalues in Riemannian geometry Year: (1984)
Ref_id:b12 Title: Sample complexity of probability divergences under group symmetry Year: ()
Ref_id:b13 Title: On the impact of kernel approximation on learning accuracy Year: (2010)
Ref_id:b14 Title: Invariant kernels: Rank stabilization and generalization across dimensions Year: (2025)
Ref_id:b15 Title: On the nyström method for approximating a gram matrix for improved kernel-based learning Year: (2005)
Ref_id:b16 Title: Low-dimensional invariant embeddings for universal geometric learning Year: (2024)
Ref_id:b17 Title: Equivariant frames and the impossibility of continuous canonicalization Year: (2004)
Ref_id:b18 Title: Provably strict generalisation benefit for invariance in kernel methods Year: (2021)
Ref_id:b19 Title: Provably strict generalisation benefit for equivariant models Year: (2021)
Ref_id:b20 Title: Partial differential equations Year: (2022)
Ref_id:b21 Title: Measuring invariances in deep networks Year: (2009)
Ref_id:b22 Title: Symmetry-adapted machine learning for tensorial properties of atomistic systems Year: (2018)
Ref_id:b23 Title: Learning translation invariant recognition in a massively parallel networks Year: ()
Ref_id:b24 Title:  Year: (1987)
Ref_id:b25 Title: The spectral function of an elliptic operator Year: (1968)
Ref_id:b26 Title: Equivariance with learned canonicalization functions Year: (2023)
Ref_id:b27 Title: On the hardness of learning under symmetries Year: (2024)
Ref_id:b28 Title: Equivariant flows: exact likelihood generative learning for symmetric densities Year: (2020)
Ref_id:b29 Title: Group theoretical methods in machine learning Year: (2008)
Ref_id:b30 Title: Imagenet classification with deep convolutional neural networks Year: (2012)
Ref_id:b31 Title: Introduction to Smooth Manifolds Year: (2012)
Ref_id:b32 Title: Riemannian manifolds: an introduction to curvature Year: (2006)
Ref_id:b33 Title: A survey of convolutional neural networks: analysis, applications, and prospects Year: (2021)
Ref_id:b34 Title: A canonization perspective on invariant and equivariant learning Year: (2024)
Ref_id:b35 Title: Directional statistics Year: (2009)
Ref_id:b36 Title: Learning with invariances in random features and kernel models Year: (2021)
Ref_id:b37 Title: The group of isometries of a riemannian manifold Year: (1939)
Ref_id:b38 Title: Permutation invariant graph generation via score-based generative modeling Year: (2020)
Ref_id:b39 Title: On the differentiability of isometries Year: (1957)
Ref_id:b40 Title: Riemannian geometry Year: (2006)
Ref_id:b41 Title: Approximation-generalization trade-offs under (approximate) group equivariance Year: (2023)
Ref_id:b42 Title: Frame averaging for invariant and equivariant network design Year: (2022)
Ref_id:b43 Title: Pointnet: Deep learning on point sets for 3d classification and segmentation Year: ()
Ref_id:b44 Title: Deep hierarchical feature learning on point sets in a metric space Year: (2017-03)
Ref_id:b45 Title: Random features for large-scale kernel machines Year: (2007)
Ref_id:b46 Title: Equivariance through parameter-sharing Year: (2017)
Ref_id:b47 Title: The graph neural network model Year: (2008)
Ref_id:b48 Title: Learning with kernels: support vector machines, regularization, optimization, and beyond Year: (2002)
Ref_id:b49 Title: Euclidean symmetry and equivariance in machine learning Year: (2021)
Ref_id:b50 Title: Concerning the lp norm of spectral clusters for second-order elliptic operators on compact manifolds Year: (1988)
Ref_id:b51 Title: A robust kernel statistical test of invariance: Detecting subtle asymmetries Year: (2025)
