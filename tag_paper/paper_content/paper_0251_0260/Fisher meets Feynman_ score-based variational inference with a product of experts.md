Title: Fisher meets Feynman: score-based variational inference with a product of experts
Abstract: We introduce a highly expressive yet distinctly tractable family for black-box variational inference (BBVI). Each member of this family is a weighted product of experts (PoE), and each weighted expert in the product is proportional to a multivariate t-distribution. These products of experts can model distributions with skew, heavy tails, and multiple modes, but to use them for BBVI, we must be able to sample from their densities. We show how to do this by reformulating these products of experts as latent variable models with auxiliary Dirichlet random variables. These Dirichlet variables emerge from a Feynman identity, originally developed for loop integrals in quantum field theory, that expresses the product of multiple fractions (or in our case, t-distributions) as an integral over the simplex. We leverage this simplicial latent space to draw weighted samples from these products of experts-samples which BBVI then uses to find the PoE that best approximates a target density. Given a collection of experts, we derive an iterative procedure to optimize the exponents that determine their geometric weighting in the PoE. At each iteration, this procedure minimizes a regularized Fisher divergence to match the scores of the variational and target densities at a batch of samples drawn from the current approximation. This minimization reduces to a convex quadratic program, and we prove under general conditions that these updates converge exponentially fast to a near-optimal weighting of experts. We conclude by evaluating this approach on a variety of synthetic and real-world target distributions.

Section: Introduction
The goal of variational inference (VI) is to approximate an intractable probability density p by the best-matching density q from some simpler parameterized family Q [4,23,53]. VI is typically used when it is difficult to draw samples from p, but often there exists a "black-box" way to compute the gradient of log p (that is, the score) at any point in the domain of R D [28,42]. Each such gradient evaluation provides a wealth of information-considerably more than what is provided by a mere sample-and for this reason a growing number of researchers have begun to investigate score-based methods for black-box VI (BBVI) [6,7,36,37]. This direction of research is also motivated by the remarkable successes of score-based methods for generative modeling [18,19,[47][48][49].
Despite this allure, score-based BBVI still faces many challenges. There are inherent trade-offs that arise between the expressivity of the variational family Q and its ease of use. For BBVI it must be tractable to evaluate and draw samples from each q ∈ Q; it must also be tractable to optimize over Q and find its best approximation to the target p. These trade-offs must be managed by any practitioner of BBVI. At the same time, researchers need more than methods which are merely practical. BBVI replaces an intractable problem in inference by a more tractable one in optimization, but still the challenge remains to prove theoretical guarantees for the optimizations that arise in this framework [3,11,24,57].
In this paper we introduce a new family for score-based BBVI that navigates these trade-offs in an appealing fashion. The densities in this family are highly expressive and yet manageably tractable, and we are also able to provide certain theoretical guarantees for the optimizations required for scorebased VI. Each density in this family is a product of experts (PoE) [16], and each (weighted) expert is proportional to a multivariate t-distribution [54] over R D . In general, it can be challenging to work with products of experts, but for this family we show how to sample from and (in some cases) evaluate their densities-exactly what is needed to use them for VI.
The full potential of this family is unlocked by a Feynman identity [12,46], originally developed for loop integrals in quantum field theory, that expresses the product of K fractions as an integral over the simplex ∆ K-1 . In particular, we show that the Feynman identity implies a representation of the product of t densities as a continuous mixture of t-distributions. A consequence of this representation is that a PoE in this family is more naturally equipped than a finite mixture model to approximate target densities with a continuum of modes that lie in a convex set. We also leverage this representation to perform two important tasks for BBVI-first, how to draw samples from a PoE in this variational family, and second, how to estimate its normalizing constant. Notably, we transform these problems to samplers and integrals over the simplex ∆ K-1 as opposed to all of R D . Using these techniques, we then introduce a score-based BBVI algorithm to find the member of this PoE with the minimum Fisher divergence to the target density. To do so, we first generate a large pool of experts (i.e., t-distributions with different modes and tails), and then we derive an iterative score-matching procedure to optimize the exponents that determine their geometric weighting in the PoE. In practice this procedure drives many exponents to zero, thus pruning irrelevant experts from the product. More specifically, each iteration updates the expert weights in the PoE by minimizing a regularized Fisher divergence. Score matching is particularly convenient for PoE models because the score is linear in the weights, and the optimization problem reduces to solving a sequence of convex quadratic programs, where each subproblem can be solved efficiently.
In addition, we analyze the convergence of the BBVI algorithm. First, we derive rates of convergence for the expert weights that are iteratively estimated by the algorithm with a finite batch size. We prove that these weights converge exponentially quickly to a neighborhood of an optimal weighting of experts in the PoE, where the size of the neighborhood depends on the amount of misspecification of the variational family. We also demonstrate the benefits of this approach to BBVI empirically on a variety of synthetic and real-world target densities. In particular, we show that with this variational family, we can approximate diverse target densities, including ones that are skewed and heavy-tailed.
this section cite: ['b3', 'b22', 'b52', 'b27', 'b41', 'b5', 'b6', 'b35', 'b36', 'b17', 'b18', 'b46', 'b47', 'b48', 'b2', 'b10', 'b23', 'b56', 'b15', 'b53', 'b11', 'b45']

Section: A latent variable model for products of experts
Our goal is to perform score-based BBVI with a particular PoE variational family by minimizing the Fisher divergence between a variational density q(z) and a target density p(z) supported on R D : D(q; p) = ∥∇ log q(z) -∇ log p(z)∥ 2 q(z) dz.
Before tackling the broader problem of BBVI with products of experts, we begin by studying these models in their own right, showing how to enable the use of the PoE as a variational family. Consider a (weighted) PoE with the density
q(z) = 1 C α K k=1 q k (z) α k ,(2)
where the exponents {α k } K k=1 determine the geometric weighting of the nonnegative functions {q k } K k=1 in the product, and C α is a normalizing constant given by
C α = K k=1 q k (z) α k dz.(3)
We refer to the function q k as the kth expert in the PoE and to the exponentiated function q α k k as the kth weighted expert. We assume that the exponents (or weights) in the PoE are nonnegative (i.e., α k ≥ 0), and later we identify further constraints that ensure the integrability of the product in Eq. 3. When the expert functions q k are Gaussian, the product itself is Gaussian, leading to a closed-form normalizing constant and efficient sampling (see, e.g., Wu and Goodman [56]). However, more generally, it can be difficult to work with products of experts, and in particular to sample from their densities or to estimate their normalizing constants. Because the Fisher divergence in Eq. 1 is generally intractable, we need samples from q to form an empirical estimate of the divergence that can then be minimized. In addition, while the Fisher divergence does not rely on the normalizing constant of q, there are many applications that require it; in this work, we use it to compute the Kullback-Leibler (KL) divergence in Section 4.
In this section, we develop a particular family of products of experts, show how to reformulate the models in this family as latent variable models, and then (most critically) leverage the latent space in these models to draw samples from their densities. The ability to sample efficiently from these models will be at the heart of their use for BBVI in Section 3.
this section cite: ['b55']

Section: Products of multivariate t-distributions
We now focus on the parameterized family of products of experts whose weighted experts are proportional to multivariate t-distributions over R D . Specifically, we suppose that
q k (z) α k = [1 + (z-µ k ) ⊤ Λ k (z-µ k )] -α k ,(4)
where µ k ∈ R D and Λ k ⪰ 0. Recall that a t-distribution is parameterized by its mean µ ∈ R D , inverse scale matrix Ω ≻ 0, and degrees of freedom ν > 0, and it has the probability density function
T (z | µ, Ω, ν) = Γ( ν+D 2 ) |Ω| 1/2 Γ( ν 2 )(πν) D/2 1 + 1 ν (z-µ) ⊤ Ω (z-µ) -(ν+D)/2 ,(5)
where Γ is the gamma function and |Ω| denotes the determinant of Ω. Thus the kth weighted expert in Eq. 4 is proportional to a t-distribution with mean µ k , inverse scale matrix Λ k (when Λ k ≻ 0), and degrees of freedom 2α k -D.
Note that we do not require all of the weighted experts in Eq. 4 to define proper densities; instead we allow some of the inverse scale matrices to be rank-deficient with |Λ k | = 0. Thus these products of experts have more flexibility than finite mixture models whose component densities must be normalizable. The PoE in Eqs. 2 and 4 will be normalizable if 2 k α k > D, provided all inverse scale matrices Λ k are full rank. Consequently, we impose this linear sum constraint on the expert weights. If not all inverse scale matrices are positive definite, then integrability must be verified.
We demonstrate the flexibility of this family in Figure 1, where the first two panels show the product of two t-distributions that take the form q(z) ∝ 2 k=1 1 + (z -µ k ) 2 -α k , where the location parameters are set to either µ = ±3 or µ = ±5. With only two experts they already parameterize a wide range of behaviors. Finally, we plot the mixture of the same two experts as the middle panel, where the mixture weights are formed by normalizing the α k values. These two panels highlight the difference between products-whose "and" relationship yields larger density values when all experts are large-and mixtures-whose "or" relationship yields larger values when any one expert is large.
this section cite: []

Section: Feynman parameterization for products of experts
We show how to rewrite the PoE in Eq. 2 in a particularly revealing form. To do so, we rely on an identity that expresses a product of positive denominators as an integral over the simplex ∆ K-1 := {w ∈ [0, 1] K : k w k = 1}. The simplest form of the identity (for two denominators) is straightforward to verify; it states that 1
A 1 A 2 = 1 0 1 A 1 w + A 2 (1-w) 2 dw(6)
for all A 1 , A 2 > 0. The identity we use generalizes the above to a product of k denominators that are geometrically weighted by exponents α k . In this case, the integral over [0, 1] in Eq. 6 is replaced by an integral over the simplex ∆ K-1 . In particular, it is also true that 1
A α1 1 . . . A α K K = Γ( k α k ) k Γ(α k ) ∆ K-1 k w α k -1 k ( k w k A k ) k α k dw,(7)
This so-called Feynman parameterization is useful to simplify certain loop integrals in quantum field theory [12,46]. Here we make a telling observation about the terms in Eq. 7 that are independent of the denominators A k ; these terms are equal to the probability density function of the Dirichlet distribution. In particular, we can express the Feynman parameterization more compactly as
k A -α k k = E w∼Dir(α) k w k A k -k α k .(8)
We now use the above identity to reinterpret the PoE introduced in the previous section. First we set A k = 1/q k (z) in Eq. 8, where q k (z) is the expert in Eq. 4. Also, as shorthand, let ∥v∥ 2 Λ = v ⊤ Λv denote the quadratic forms that appear in these experts. From these substitutions it follows that
K k=1 q k (z) α k = E w∼Dir(α) 1 + k w k ∥z-µ k ∥ 2 Λ k -k α k .(9)
We now show that the PoE can be re-expressed as a continuous mixture of multivariate t-distributions, all of which have the same number of degrees of freedom but differ in other respects; see Appendix C.1 for a detailed derivation. To do so, we define
Λ(w) = k w k Λ k (10
)
µ(w) = Λ(w) -1 k w k Λ k µ k ,(11)
which can be viewed as location and inverse scale parameters that are continuously indexed by w ∈ ∆ K-1 . Now, after expanding the quadratic form in the denominator of Eq. 9 and completing the square, we can use the definitions of Λ(w) and µ(w) to rewrite Eq. 9 as
K k=1 q k (z) α k = E w∼Dir(α) 1 + k w k ∥µ k -µ(w)∥ 2 Λ k + ∥z-µ(w)∥ 2 Λ(w) -k α k .(12)
Next we supplement Eq. 10 by defining another inverse scale matrix, Ω(w), that is continuously indexed by w ∈ ∆ K-1 and absorbs the terms in Eq. 12 that are independent of z. Let
ν = 2 k α k -D,(13)
σ 2 (w) = k w k ∥µ k -µ(w)∥ 2 Λ k ,(14)
Ω(w) = νΛ(w)/(1+σ 2 (w)).(15)
We can now cast the expectation in Eq. 12 into a more revealing form, arriving at the following representation of the PoE as a continuous mixture of t-distributions. This result is obtained by making the substitutions in Eqs. 13 to 15 and appealing to the form of the t-distribution in Eq. 5.
Result 2.1 (PoE as continuous mixture of t-distributions). Consider the product k q k (z) α k with t-distributed experts (Eq. 4), and let ν = 2 k α k -D. The product can be computed as
K k=1 q k (z) α k = E w∼Dir(α) 1+σ 2 (w) -ν+D 2 1 + 1 ν ∥z-µ(w)∥ 2 Ω(w) -ν+D 2 . (16
)
Thus the PoE can be viewed as a continuous mixture of t-distributions in Eq. 5 with ν degrees of freedom, location parameters µ(w), and inverse scale matrices Ω(w), where w ∈ ∆ K-1 .
This result suggests that this type of PoE may be well-suited for distributions with a continuum of modes that lie in a convex set. Any such regions of high probability will be more naturally modeled by a continuous mixture than a finite one. As an aside, we note that the t-distribution can itself be written as a (continuous) scale mixture of Gaussians [1]. By extension, the above result shows that a product of t-distributed experts can be written even more generally as a continuous location-and-scale mixture of Gaussians. This correspondence provides further motivation for their use in VI.
this section cite: ['b11', 'b45', 'b0']

Section: Joint density and latent variable model
There are many useful results that follow from the Feynman parameterization. One such result arises when revisiting the computation of the PoE's normalizing constant C α in Eq. 3. While C α is initially expressed as an integral over R D , our goal is to re-express it as a (potentially simpler) integral over the simplex ∆ K-1 . Substituting Eq. 16 into Eq. 3, we find that
C α = E w∼Dir(α) 1+σ 2 (w) -ν+D 2 dz 1 + 1 ν ∥z-µ(w)∥ 2 Ω(w) -ν+D 2 , (17
)
where we have applied Fubini's theorem to move the integral over z inside the expectation over w.
We can now perform the integral over z, as it is given exactly by the normalizing constant in Eq. 5 for a t-distribution with inverse scale matrix Ω(w) and ν degrees of freedom. In this way we obtain the following result.
Result 2.2 (Normalizing constant of PoE). Consider the PoE with the t-distributed experts in Eq. 4, and let ν = 2 k α k -D. The normalizing constant of this PoE can alternately be computed as
C α = K k=1 q k (z) α k dz = E w∼Dir(α) Ω(w) -1 2 1+σ 2 (w) -ν+D 2 • Γ( ν 2 )(πν) D 2 Γ( ν+D 2 )(18)
Next we use the Feynman parameterization to show how to sample from q. In particular, we construct a joint density q(w, z) that yields the desired marginal q(z) in Eq. 2, and then we generate samples from the joint q(w, z). Combining Result 2.1 and Result 2.2, we find
q(w, z) = C α (w) C α Dir(w | α) T (z | µ(w), Ω(w), ν),(19)
where the new leading factor C α (w) in the numerator is used to account for all the terms in Eq. 17 that are not absorbed by the (normalized) Dirichlet and t-distributions-namely, C α (w) := Ω(w) 1+σ 2 (w)
ν+D -1 2 Γ( ν 2 ) (πν) D 2 Γ( ν+D 2 ) .(20)
It is then straightforward to verify that q(w, z)dw = q(z), leading to a natural auxiliary-variable sampling procedure. Indeed, we can interpret this joint density as a latent variable model and use its marginal and conditional densities to draw samples from the PoE. Marginalizing over z in Eq. 19, we find that q(w) = q(w, z) dz = Cα(w) Cα Dir(w | α), where C α (w) is given by Eq. 20. Likewise, we recover the t-distribution for the conditional density q(z|w) upon dividing the joint in Eq. 19 by this result. In sum we have shown the following.
Result 2.3 (Latent variable model for PoE). Consider the PoE with the t-distributed experts in Eq. 4. We can draw samples from the PoE by sampling from the generative model
w b ∼ C α (w) C α Dir(w | α),(21)
z b | w b ∼ T (z | µ(w b ), Ω(w b ), ν),(22)
where w b lies in the simplex, the w b -dependent terms on the right are given by Eqs. 10, 11, 13 to 15 and 20, and z b is conditionally t-distributed given w b .
5 0 5 10 15 20 z 0.00 0.05 0.10 0.15 0.20 0.25 density Identify mode(s) target expert (mode) 5 0 5 10 15 20 z Refine mode(s) expert (jitter) 5 0 5 10 15 20 z Fit variational parameters variational approximation Figure 2: To select experts, we identify each mode and then add more experts to refine the fit. The above result is not yet a practical recipe for sampling from a PoE. The difficulty lies in the first step: it is not straightforward to sample from q(w) in Eq. 21 due to its leading dependence on C α (w). But we can circumvent this difficulty by using the Dirichlet distribution that appears in q(w) as a proposal distribution for importance sampling. Suppose we wish to estimate an expected value E q(z) [h(z)]. We can draw a batch of samples
w b ∼ Dir(w | α),(23)
z b | w b ∼ T (z | µ(w b ), Ω(w b ), ν).(24)
from the Dirichlet and t-distributions in Eqs. 21 and 22 and weight these samples by C α (w b ) in the calculation of the expected value. In this way we obtain the estimate
E q(z) [h(z)] ≈ b C α (w b ) h(z b ) b C α (w b ) . (25
)
3 Score-based VI with products of experts
In this section we show how to approximate a target density p by a PoE with t-distributed experts.
To do so, we must specify the number of experts, the parameters of their t-distributions, and the exponents that determine their geometric weighting in the PoE. We discuss these problems in turn.
this section cite: []

Section: Selecting the experts
Of the many ways to select experts, we seek a simple heuristic that works in practice and avoids a complicated, coupled optimization over the expert parameters (µ k , Λ k ) and weights α k in Eq. 4. Our basic strategy is to generate a large, oversaturated pool of experts whose means µ k are concentrated near the modes of the target density. We shall see in later sections that poorly situated (and hence irrelevant) experts are efficiently pruned by the procedure for learning the weights α k .
Our strategy for selecting experts has three steps. The first is to locate the modes of the target density by hill-climbing in log p(z) from randomly chosen starts; if the target density is a Bayesian posterior, then we can choose these starts by sampling from the corresponding prior. The second step is to place an expert at each mode: the expert's mean µ k and inverse scale parameter Λ k to match the mode's location and curvature. The third step is to place additional experts nearby so that the weighted PoE can better model the shape (e.g., skew, kurtosis) of each mode: once a new location µ k is chosen, the corresponding inverse scale Λ k is set to the (negative) Hessian at µ k projected to the cone of positive semidefinite matrices. For further details of this step and a discussion of expert placing cost, see Appendix D.1. Figure 2 illustrates the intuition behind this strategy for a target density with one mode. Overall we found this strategy to be quite effective in conjunction with a score-based procedure for optimizing the expert weights in the PoE. We describe this score-based procedure next.
this section cite: []

Section: Weighting the experts
Given experts q k in the form of Eq. 4, we now consider how to form the weighted PoE in Eq. 2 that best approximates the target density p. Specifically, for q(z) ∝ k q k (z) α k , we seek the weights {α k } K k=1 that minimize the Fisher divergence between q and p, as defined in Eq. 1. Since the PoE in Eq. 2 has support on R D , this divergence vanishes only when q = p.
Though we cannot minimize Eq. 1 directly, in the spirit of BBVI, we can instead attempt to minimize an empirical estimate of the Fisher divergence, one that is based on drawing samples from q. A further simplification is achieved by decoupling the procedures for sampling from q and optimizing q. To do so, we iteratively minimize a closely related objective. In particular, let α (t) ∈ R K be the expert weights at the t th iteration, and let q(z|α) denote the density of the PoE with expert weights α. Rather than minimizing Eq. 1 directly, at the t th iteration we instead solve the simpler problem
α (t+1) = argmin α∈C ∇ log q(z|α) -∇ log p(z) 2 q z|α (t) dz + 1 ηt α -α (t) 2 , (26
)
where η t > 0 is a learning rate and the domain C of the optimization constrains the expert weights to define a normalizable PoE. Note that this update minimizes a biased estimate of the Fisher divergence; the estimate is biased because in the first term of the objective the expectation is performed with respect to q(z|α (t) ) instead of q(z|α). But at the same time, the update attempts to compensate for this bias by penalizing solutions that move too far from one iteration to the next; this penalty is enforced by the regularizer in the second term of the objective. This is the same intuition that is behind a recently proposed "batch-and-match" algorithm for Gaussian BBVI [7].
The rest of this section fleshes out this iterative procedure and highlights its three main advantages. First, when q is a weighted PoE with t-distributed experts, we can use samples to compute an empirical estimate of the objective in Eq. 26. Second, at each iteration, we can minimize this empirical estimate by solving a strongly convex quadratic program. Third, this iterative procedure provably converges under fairly general conditions to a neighborhood of an optimally weighted PoE.
We now construct an empirical estimate E t (α) for the objective in Eq. 26 at the t th iteration. Using the latent variable model in Eqs. 21 and 22, we generate a batch of B weighted samples {(w b , z b )} B b=1 from the PoE with expert weights α (t) . From these samples, we construct the empirical estimate
E t (α) = 1 B B b=1 π b ∇ log q(z b |α) -∇ log p(z b ) 2 + 1 ηt α -α (t) 2 ,(27)
where π b ∝ C α (t) (w b ) is the importance weight of the b th sample from Eq. 25. Note how by sampling the PoE with weights α (t) , we have decoupled these samples from the optimization over α in Eq. 26.
Next we show that the empirical estimate E t (α) is minimized by solving a convex quadratic program in the expert weights α. First, we observe that for any PoE, as defined by q(z|α)
∝ k q k (z) α k in Eq. 2,
the score is linear in the weights of its experts: namely, ∇ log q(z|α) = k α k ∇ log q k (z).
We make this explicit for the t-distributed experts in Eq. 4 by writing
∇ log q(z b |α) = Q b α,(28)
where Q b is the D×K matrix whose k th column is given by
∇ log q k (z b ) = -2q k (z b )Λ k (z b -µ k ).
From the linearity of the scores, it follows at once that E t (α) in Eq. 27 is quadratic in the expert weights α. In particular, we can rewrite Eq. 27 as
E t (α) = α ⊤ 1 B b π b Q ⊤ b Q b + 1 ηt I α -2 1 B b π b Q ⊤ b ∇ log p(z b ) + 1 ηt ⊤ α + E t (0), (29
)
and from the above, we also see that E t (α) is strongly convex in α for all η t ∈ (0, ∞). The expert weights are updated by minimizing this objective subject to a constraint that the newly weighted PoE is normalizable. To satisfy the parameter constraints of the PoE, we define the constraint set, such that α ∈ C , as
C = α ∈ R K : α 1 ≥ 0, α 2 ≥ 0, . . . , α k ≥ 0, k α k ≥ D 2 + ε ,(30)
where ε > 0 is a slack variable added to ensure that the product of experts k q k (z) α k is integrable. Eq. 30 defines a convex set, and hence the overall optimization is convex; in particular, it is a nonnegative least squares (NNLS) problem with linear constraints, for which there exist many efficient solvers [29,Ch. 23]. For our purposes, it is also interesting that problems in NNLS often yield sparse solutions where multiple constraints are active. In our setting, these are solutions in which irrelevant experts are assigned zero weights and do not contribute to the density of the PoE. 6:
Compute matrix Q b ∈ R D×K whose kth column is equal to ∇ log q k (z b ).
this section cite: ['b6', 'b28']

Section: 7:
end for 8:
Compute G t = 1 B B b=1 π b Q ⊤ b Q b + 1 ηt I and h t = 1 B B b=1 π b Q ⊤ b g b + 1 ηt α (t) . 9:
Update expert weights:
α (t+1) = argmin α∈C 1 2 α ⊤ G t α-h ⊤ t α . 10: end for 11: Output: PoE weights α (T ) ∈ C (Eq. 30)
this section cite: []

Section: Convergence theorem
We summarize the overall iterative procedure for learning expert weights in Algorithm 1. To prove convergence we make some basic assumptions about the Fisher divergence in Eq. 1 and its empirical estimate at each iteration of Algorithm 1. To state these assumptions, let
D t (α) = 1 B b π b ∇ log q(z b |α) -∇ log p(z b ) 2 (31)
denote the first term in Eq. 27, and let ∇ D t and H[ D t ] denote the gradient and Hessian of this term with respect to the expert weights α. With these definitions, we can state the following theorem.
Theorem 3.1. Suppose that D(q; p) in Eq. 1 is minimized by a unique α * ∈ C, and also that for all t ≥ 0 there exists some δ ≥ 0 such that E 1 2 ∇ D t (α * ) ≤ δ and some λ > 0 such that H[ D t ] ⪰ λI almost surely. Then for constant learning rates η t ≡ η, the expected error of the iterates satisfies
E α (t) -α * ≤ 1 1+ηλ t α (0) -α * + δ λ . (32
)
This result ensures that the iterates α (t) of Algorithm 1 converge, in expectation, to a neighborhood around the optimal expert weights α * , with the error decaying at a linear (geometric) rate. The proof relies on a few key ideas, most notably that (i) the constrained least-squares problem in Eq. C.6 can be solved by projecting its unconstrained solution onto C, and (ii) this projection is with respect to an induced Mahalanobis norm, and it is nonexpansive [2, Proposition 4.8]. See Appendix E for a complete proof.
The bound in Eq. 32 separates two effects-the transient error, which shrinks exponentially fast with t and depends on the strong convexity parameter λ, and the asymptotic error floor δ λ , which depends also on the misspecification parameter δ. We briefly sketch the intuition behind these terms.
First, the error floor is proportional to the misspecification parameter δ. Let q * denote the optimally weighted PoE. On one hand, if p ∈ Q, then q * = p and D t (α * ) = ∥∇ D t (α * )∥ = 0 for all t; we see in this case that δ = 0. On the other hand, if p ̸ ∈ Q, then q * ̸ = p. In this case we expect that the stochastic gradients will have small norms at α * (and thus δ will be small) whenever D(q * ; p) itself is small. In the right panel of Figure D.2 we highlight on one experiment with an sinh-arcsinh target, that for constant or decreasing step size schedules, the error floor does not go below 10 -1 . However, on the left panel of Figure D.2, we do show that this error floor improves as we increase the batch size. See Appendix D.2 and Appendix D.3 for details.
Second, the transient error depends on the assumption that D t (α) is λ-strongly convex for all t. Observe from Eq. C.6 that H[
D t ] = 1 B B b=1 π b Q ⊤ b Q b ; i.e.
, the Hessian is a sum of B positive semidefinite matrices. Thus for larger batch sizes, the assumption of strong convexity is increasingly likely to be satisfied. We confirm this intuition in Figure E.1, where we show that as we increase the batch size, the eigenvalues of the Hessian also increase and are bounded away from zero. We only encountered one indefinite Hessian in our experiments; this occurred when the number of experts (K = 100) was much larger than the batch size (B = 10).
this section cite: []

Section: Experiments
In this section, we compare VI with this PoE family to Gaussian BBVI (with ADVI [28] and BaM [7]) and normalizing flow-based VI [43]. In Appendix F, we provide more details on these experiments and also results on several additional examples. We also present additional results for estimating PoE normalizing constants and evaluating sampling (Appendix F.1) and expert placement (Appendix D.2).
this section cite: ['b27', 'b6', 'b42']

Section: Synthetic 2D targets
We first consider several synthetic 2D target distributions: 1) Mixture of Gaussians, 2) Product of t-distributions, 3) Diamond, 4) Funnel. We report Monte Carlo estimates of two training-objectiveagnostic metrics, KL(p; q) and E p [∥∇ log q -∇ log p∥ 2 ], using 1000 samples from p. (The VI methods minimize the expectations with respect to q.) We visualize the PoE samples by first drawing a weighted sample and then resampling the generated samples according to their normalized weights. For full details on each target, see Appendix F.
Figure 3 shows each target distribution's contours (gray curves) overlaid with samples from the fitted PoE (blue points). In these examples, the PoE achieves substantially lower values for both divergence metrics than the Gaussian approximation, due to its ability to better capture the tails of the target distributions. The normalizing flow also provides a substantial improvement over the Gaussian fit. Notably, for heavy-tailed (product of experts and diamond) targets, the PoE outperforms the flow-based family as well, yielding lower divergence in these cases.
this section cite: []

Section: Sinh-arcsinh target
We now evaluate a higher-dimensional synthetic target distribution with skew and heavy tails. The sinh-arcsinh distribution [21,22] generalizes the Gaussian distribution with additional parameters ε ∈ R D and τ ∈ R D ++ that control the skewness and the tail-weight. It transforms a Gaussian draw z ∼ N (0, Σ) coordinate-wise via z d = sinh((sinh -1 (z d ) + ε d )/τ d ). We construct the target density in D = 50 dimensions so that it is positively skewed (ε = 0.3) and heavy-tailed (τ d = 0.7).
In Figure 4, we show the KL and Fisher divergences between all approaches, and we found that the PoE had the lowest divergence for both metrics (left). We also plotted the divergences against the number of gradient evaluations for the iterative parameter fitting of the PoE and the normalizing flow. Note that this plot does not show the initial startup costs of each method (selecting the experts for the PoE and selecting a learning rate for the flow), which are not the dominating cost. In addition, this plot only shows the first 15 × 10 4 gradient evaluations. For both divergences, the values decrease slowly for the flow model; the final reported value in the bar graph is after 10 7 gradient evaluations. A D V I B a M F lo w P o E 0.0 0.5 1.0 1.5 2.0 2.5 Negative LLH A D V I B a M F lo w P o E 0 10 20 30 40 50 Fisher divergence (a) garch11 A D V I B a M F lo w P o E 0 5 10 15 Negative LLH A D V I B a M F lo w P o E 0 2 4 6 8 10 Fisher divergence (b) 8schools-noncentered A D V I B a M F lo w P o E 0.00 0.05 0.10 0.15 0.20 Negative LLH A D V I B a M F lo w P o E 0 1 2 3 4 5 Fisher divergence (c) gp-regr Figure 5: posteriordb targets that highlight a range of non-Gaussian posterior properties.
this section cite: ['b20', 'b21']

Section: Posterior inference problems
Next we study several posterior inference test problems from posteriordb [32,33]. For each target, we use reference samples computed using Stan (drawn via Hamiltonian Monte Carlo) [9]. Because we do not have access to the normalized target density p(z), instead of reporting KL(p; q), we report the negative expected log likelihood Neg-LLH(p; q) = -E p [log(q)]. We again also consider the Fisher divergence F (p; q), which does not require normalizing constants. All expectations are estimated using the reference samples from p.
In Figure 5, we highlight three particular examples from posteriordb. First, we consider garch11: this model has (half) uniform priors that skew the posterior when the support is transformed to R D . The skew is not modeled by Gaussian VI, but it is present in the PoE and flow-based approximations.
Next, we consider 8schools-noncentered, which exhibits skew and a heavy-tailed component introduced by a half-Cauchy(0, 5) prior. These properties are best modeled by the variational PoE.
Finally, we study a light-tailed example in gp-regr, where Gaussian BBVI performs well due the near symmetry of the posterior. Nevertheless, despite its heavy tails, the product of t-distributions still outperforms both Gaussian BBVI and the normalizing flow (with a Gaussian base distribution) on this example. This example illustrates how the expert weights α can be optimized to match different tails via the value of ν = 2 k α k -D in the mixture representation of Result 2.1.
this section cite: ['b31', 'b32', 'b8']

Section: Discussion of contributions, limitations, and future work
We have shown that VI with products of t-experts can model a variety of target densities. A key technical insight, via a Feynman identity, was to represent each PoE as a continuous mixture indexed by a Dirichlet variable. We then used this representation to sample from the PoE, a core requirement for BBVI. To optimize the expert weights, we developed a score-based VI algorithm that solves a sequence of convex quadratic programs, and we proved that its iterates converge exponentially to a neighborhood that depends on the degree of misspecification of the variational family.
While our algorithm learns the expert weights, it is limited by relying on a fixed collection of experts selected upfront. In practice, the quality of the approximation depends heavily on how these experts are chosen. While our current heuristic overspecifies the number of experts and prunes away the ineffectual ones, further gains could be achieved with a more refined approach-for instance, a boosting-style approach [15,35] that sequentially adds experts based on (say) score mismatch.
Several promising directions remain. First, the Feynman identity may have broader implications: for example, by providing a semi-analytic procedure to estimate the normalizing constant, it may open a new door to generative models with products of t-experts [26,54]. Second, the PoE construction may be useful in certain sampling methods as an initialization or proposal scheme. Finally, as mentioned above, by choosing experts more carefully we can hope to accelerate every aspect of our approach.
this section cite: ['b14', 'b34', 'b25', 'b53']

Section: References
Ref_id:b0 Title: Scale mixtures of normal distributions Year: (1974)
Ref_id:b1 Title: Convex analysis and monotone operator theory in Hilbert spaces Year: (2017)
Ref_id:b2 Title: Statistical and computational trade-offs in variational inference: A case study in inferential model selection Year: (2022)
Ref_id:b3 Title: Variational inference: A review for statisticians Year: (2017)
Ref_id:b4 Title: composable Bayesian inference in JAX Year: (2024)
Ref_id:b5 Title: EigenVI: score-based variational inference with orthogonal function expansions Year: (2024)
Ref_id:b6 Title: Batch and match: black-box variational inference with a score-based divergence Year: (2024)
Ref_id:b7 Title: Universal boosting variational inference Year: (2019)
Ref_id:b8 Title: Stan: A probabilistic programming language Year: (2017)
Ref_id:b9 Title: Density estimation using real NVP Year: (2017)
Ref_id:b10 Title: Provable convergence guarantees for black-box variational inference Year: (2023)
Ref_id:b11 Title: Space-time approach to quantum electrodynamics Year: (1949)
Ref_id:b12 Title: Handbook of convergence theorems for (stochastic) gradient methods Year: (2023)
Ref_id:b13 Title: Nonparametric variational inference Year: (2012)
Ref_id:b14 Title: Boosting variational inference Year: (2016)
Ref_id:b15 Title: Products of experts Year: (1999)
Ref_id:b16 Title: Training products of experts by minimizing contrastive divergence Year: (2002)
Ref_id:b17 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b18 Title: Estimation of non-normalized statistical models by score matching Year: (2005)
Ref_id:b19 Title: Tails of Lipschitz triangular flows Year: (2020)
Ref_id:b20 Title: Sinh-arcsinh distributions Year: (2009)
Ref_id:b21 Title: The sinh-arcsinh normal distribution Year: (2019)
Ref_id:b22 Title: An introduction to variational methods for graphical models Year: (1999)
Ref_id:b23 Title: On the convergence of black-box variational inference Year: (2023)
Ref_id:b24 Title: Adam: A method for stochastic optimization Year: (2015)
Ref_id:b25 Title: Auto-encoding variational Bayes Year: (2014)
Ref_id:b26 Title: Improved variational inference with inverse autoregressive flow Year: (2016)
Ref_id:b27 Title: Automatic differentiation variational inference Year: (2017)
Ref_id:b28 Title: Solving least squares problems Year: (1995)
Ref_id:b29 Title: Fat-tailed variational inference with anisotropic tail adaptive flows Year: (2022)
Ref_id:b30 Title: Boosting black box variational inference Year: (2018)
Ref_id:b31 Title: Testing, benchmarking and developing Bayesian inference algorithms Year: (2025)
Ref_id:b32 Title: posteriordb: a set of posteriors for Bayesian inference and probabilistic programming Year: (2022)
Ref_id:b33 Title: Effective sample size for importance sampling based on discrepancy measures Year: (2017)
Ref_id:b34 Title: Variational boosting: Iteratively refining posterior approximations Year: (2017)
Ref_id:b35 Title: Variational inference with Gaussian score matching Year: (2023)
Ref_id:b36 Title: Batch, match, and patch: low-rank approximations for score-based variational inference Year: (2025)
Ref_id:b37 Title: Automatic differentiation variational inference with mixtures Year: (2021)
Ref_id:b38 Title: Non-asymptotic analysis of stochastic approximation algorithms for machine learning Year: (2011)
Ref_id:b39 Title: Stochastic gradient descent, weighted sampling, and the randomized Kaczmarz algorithm Year: (2016)
Ref_id:b40 Title: A modern introduction to online learning Year: (2025)
Ref_id:b41 Title: Black box variational inference Year: (2014)
Ref_id:b42 Title: Variational inference with normalizing flows Year: (2015)
Ref_id:b43 Title: BridgeStan: Efficient in-memory access to Stan programs through Python Year: (2023)
Ref_id:b44 Title: Nonparametric automatic differentiation variational inference with spline approximation Year: (2024)
Ref_id:b45 Title: Evaluating Feynman integrals Year: (2004)
Ref_id:b46 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b47 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b48 Title: Sliced score matching: A scalable approach to density and score estimation Year: (2020)
Ref_id:b49 Title: OSQP: An operator splitting solver for quadratic programs Year: (2020)
Ref_id:b50 Title: Unbiased implicit variational inference Year: (2019)
Ref_id:b51 Title: Implicit variational inference for high-dimensional posteriors Year: (2023)
Ref_id:b52 Title: Graphical models, exponential families, and variational inference Year: (2008)
Ref_id:b53 Title: Learning sparse topographic representations with products of student-t distributions Year: (2002)
Ref_id:b54 Title: flowMC: Normalizing flow enhanced sampling package for probabilistic inference in JAX Year: (2023)
Ref_id:b55 Title: Multimodal generative models for scalable weakly-supervised learning Year: (2018)
Ref_id:b56 Title: The computational asymptotics of Gaussian variational inference and the Laplace approximation Year: (2022)
Ref_id:b57 Title: MixFlows: principled variational inference via mixed flows Year: (2023)
Ref_id:b58 Title: Semi-implicit variational inference Year: (2018)
Ref_id:b59 Title: Slice sampling reparameterization gradients Year: (2021)
