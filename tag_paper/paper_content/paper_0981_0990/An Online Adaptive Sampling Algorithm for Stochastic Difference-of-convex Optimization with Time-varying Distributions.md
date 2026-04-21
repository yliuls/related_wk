Title: An Online Adaptive Sampling Algorithm for Stochastic Difference-of-convex Optimization with Time-varying Distributions
Abstract: We propose an online adaptive sampling algorithm for solving stochastic nonsmooth differenceof-convex (DC) problems under time-varying distributions. At each iteration, the algorithm relies solely on data generated from the current distribution and employs distinct adaptive sampling rates for the convex and concave components of the DC function, a novel design guided by our theoretical analysis. We show that, under proper conditions on the convergence of distributions, the algorithm converges subsequentially to DC critical points almost surely. Furthermore, the sample size requirement of our proposed algorithm matches the results achieved in the smooth case or when a measurable subgradient selector is available, both under static distributions. A key element of this analysis is the derivation of a novel O( p/n) pointwise convergence rate (modulo logarithmic factors) for the sample average approximation of subdifferential mappings, where p is the dimension of the variable and n is the sample size -a result of independent interest. Numerical experiments show that the algorithm is efficient for addressing online stochastic nonsmooth problems.

Section: Introduction
We consider the class of stochastic nonsmooth nonconvex optimization problems in the form of where C ⊂ R p is a convex set, ξ, ζ ⊂ Ω are random vectors with probability measures P ξ , P ζ , respectively, and G, H : (R p , Ω) → R are Carathéodory functions, i.e., they are continuous in x for all ξ, ζ ∈ Ω and Borel measurable in ξ and ζ for all x ∈ C. In addition, we assume G and H are convex in x (though not necessarily smooth), making f a difference-of-convex (DC) function.
minimize x∈C f (x) ≜ E ξ∼P ξ [G(x, ξ)] ≜g(x) -E ζ∼P ζ [H(x, ζ)]
When functions g and h are fully accessible, problem (1) can be solved via the classical DC algorithm (DCA). At each iteration, a convex subproblem is solved by linearizing h via the subgradient at the previous point, i.e., x t+1 = argmin x∈C g(x) -y T t (x -x t ) + µ 2 ∥x -x t ∥ 2 for some y t ∈ ∂h(x t ) and µ > 0. Due to the convexity of h, it can be shown that the objective sequence {f (x t )} is non-increasing, and the iterates asymptotically converge to a so-called DC critical point of problem (1).
However, in many applications, functions g and h are not fully known and can only be estimated from sampled data. This challenge is compounded when the underlying datagenerating distribution is time-varying, as in the case of fluctuating demand. The convergence analysis of stochastic DCA is, therefore, significantly more complex than its deterministic counterpart, as it must account for the sample average approximation (SAA) error in both the convex component and the linearized concave component. The latter, in particular, is closely tied to the convergence rate of the SAA error for subdifferential mappings when H is nonsmooth in x, introducing additional difficulty in the analysis.
In this paper, we propose an online adaptive sampling algorithm to solve problem (1). At each iteration, new data from the current distribution is used to construct a stochastic approximation of the linearized DC function, while previous samples are discarded. Unlike stochastic DCAs that aggregate past samples to compute current solutions, our method is more robust to distributional shifts occurring during the data generations along the iterations. The algorithm dynamically determines the sample sizes needed to estimate g and ∂h, adapting to the optimization path throughout the process. Specifically, when the current iterate is far from critical points, less precise yet computationally inexpensive function values and subgradient estimates suffice. However, as the iterates approach the critical points, higher accuracy in function and subgradient estimation becomes crucial for theoretical guarantees and effective practical performance.
We summarize the contribution of the paper as follows:
• We derive a novel O( p/n) convergence rate (modulo logarithmic factors) for the expected pointwise SAA error of set-valued subdifferential mappings (Theorems 3.4 and 3.5), matching the convergence rate of singlevalued gradient mappings in the smooth case, where p is the dimension of the variable and n is the sample size. Our results complement existing work (Davis and Drusvyatskiy, 2022;Ruan, 2024) on the uniform convergence rate for the SAA error for subdifferential mappings. We adopt a new proof technique that analyzes the one-sided deviation of subdifferential setvalued functions through their support functions.
• We propose an online adaptive stochastic framework for DC optimization under time-varying distributions. Unlike existing algorithms in the literature (Le Thi et al., 2024), which require a Borel measurable subgradient selector that is challenging to implement in practice, our algorithm allows the selection of any subgradient from the sampled subdifferential set. Furthermore, our algorithm operates under weak assumptions on the data generation process, allowing the underlying distributions to vary over time without necessarily matching the true distribution. We establish theoretical guarantees under the novel assumption that the cumulative Wasserstein-1 distance between successive distributions over iterations is bounded.
• Assume that we draw N g,t samples to estimate g and N h,t samples to estimate ∂h at time t. For any α g ∈ (0, 1/2) and α h ∈ (0, 1), we establish the almost sure convergence of the iterative sequence to a critical point under the condition that t≥0
1 N αg g,t + 1 N α h h,t < ∞.
We further propose adaptive sampling strategy to adjust sample sizes at each step based on progress from the most recent iteration. In practice, the adaptive strategy enhances performance compared to its non-adaptive counterpart by potentially reducing the number of samples required during the initial stage of the algorithm.
this section cite: ['b13', 'b29', 'b21']

Section: Related Literature
Non-asymptotic convergence analysis of SAAs. An important step in our analysis is the error estimation of SAAs of g and ∂h. The non-asymptotic convergence analysis of SAAs for expected functions has been well-studied in the existing literature; see, for example, the monograph Shapiro (2000). For the SAA convergence rate of subdifferentials, Xu (2010) demonstrates non-asymptotic, dimension-dependent high-probability bounds on the distance between the empirical and population subdifferentials under the Hausdorff metric. However, the population objective is essentially required to be smooth. In Mei et al. (2018), the authors discuss uniform convergence of gradients for smooth objectives under the assumption that the gradient is sub-Gaussian with respect to the population data. In Foster et al. (2018), the authors provide dimension-independent high-probability convergence rates of gradients for smooth Lipschitz generalized linear models, utilizing a "chain rule" for Rademacher complexity. These works do not directly examine the convergence behavior of subdifferential sets. More recently, Ruan (2024) achieves a tight O( p/n) rate (modulo logarithmic factors) for the uniform convergence of weakly convex subdifferential mappings. This complements the O( 4 p/n) uniform convergence rate of subdifferentials in Davis and Drusvyatskiy (2022). However, their result is based on the convex-smooth composite structure, as well as subexponential assumptions for random vector and process, see Assumption C in Ruan (2024).
Stochastic and Online DC Optimization. While deterministic DC algorithms have been extensively studied in existing literature (Le Thi and Pham Dinh, 2018), their stochastic counterparts have only recently gained attention (Thi et al., 2017; Le Thi et al., 2020). The first work that allowed both components in a DC problem to be nonsmooth was presented in Le Thi et al. (2022), where an SDCA scheme was proposed that stores all past information for constructing future subproblems. This approach achieves near-optimal sample size requirement by adding just one sample per DCA subproblem. Le Thi et al. (2024) pioneered the study of DCA in an online setting, eliminating the need to store historical information. Their approach resamples at each iteration and employs SAAs to approximate the linearized DC function using new samples, resulting in adaptive capabilities that offer a significant advantage over those in Le Thi et al. (2022). However, this method relies on the realization of a Borel measurable subgradient selector, as specified in Assumption 1 of Le Thi et al. (2024). Moreover, non-asymptotic convergence of stochastic DC optimization has been studied in Nitanda and Suzuki (2017); Xu et al. (2019), which propose stochastic proximal DC algorithms by adding quadratic terms for DC subproblems.
Nevertheless, these analyses rely on smoothness or Hölder continuity of the gradient, which are often too strong for many nonsmooth functions. Recent work in nonsmooth weakly convex optimization (Davis and Drusvyatskiy, 2018;Sun and Sun, 2022;Moudafi, 2022;Yao et al., 2022) has introduced Moreau envelope smoothing approximations for both components, enabling a non-asymptotic convergence analysis to nearly ϵ-critical points for deterministic problems-a relaxed convergence criterion. These works have yet to establish complete non-asymptotic convergence for non-smooth DC problems since a gap remains between nearly ϵ-critical points and true critical points.
Recent studies have explored online optimization under distribution shifts, particularly within online convex optimization and stochastic approximation methods. Standard approaches typically assess performance through regret bounds relative to a defined measure of distribution shifts (e.g., Besbes et al. (2015); Fahrbach et al. (2023); Sankararaman and Narayanaswamy (2023)). Our proposed algorithm differs due to the nonsmooth nonconvex structure, where regret-based analysis is inapplicable, as our results rely on asymptotic convergence properties instead.
Adaptive Sampling in Stochastic Optimization. Adaptive sampling methods offer advantages over fixed-sample approaches, such as leveraging parallelism and generating iterates with reduced variance due to progressively increasing sample sizes. Adaptive strategies often use gradient approximation tests to regulate accuracy. Examples include norm-based tests (Carter, 1991;Byrd et al., 2012), inner product tests (Bollapragada et al., 2018), and other methods (Cartis and Scheinberg, 2018;Jin et al., 2021). For a comprehensive overview of adaptive sampling techniques, readers are referred to Curtis and Scheinberg (2020).
this section cite: ['b31', 'b37', 'b23', 'b16', 'b13', 'b29', 'b12', 'b25', 'b39', 'b2', 'b15', 'b5', 'b4', 'b3', 'b6', 'b17']

Section: Preliminaries
We first summarize the notation used throughout the paper. We write R p as the p-dimensional Euclidean space equipped with the inner product ⟨x, y⟩ = x ⊤ y and the induced norm ∥x∥ ≜ √ x ⊤ x. The symbol B(x, δ) is used to denote the closed ball of radius δ > 0 centered at a vector x ∈ R p . Let A and C be two nonempty subsets of R p . The distance from a vector x ∈ R p to A is defined as dist(x, A) ≜ inf y∈A ∥y -x∥. The one-sided deviation of A from C is defined as D(A, C) ≜ sup x∈A dist(x, C).
The Hausdorff distance between A and C is defined as H(A, C) := max{D(A, C), D(C, A)}.
We proceed by introducing fundamental concepts from nonsmooth analysis. For detailed discussions, we refer the reader to the monographs (Clarke, 1990;Rockafellar and Wets, 1998;Mordukhovich, 2006). Let r : O → R be a function defined on an open set O ⊆ R p . The classical one-sided directional derivative of r at x ∈ O along the direction
d ∈ R p is defined as r ′ (x; d) ≜ lim t↓0 r(x + td) -r(x) t .
The function r is said to be directionally differentiable at x ∈ O if it is directionally differentiable along any direction d ∈ R p . In contrast, the Clarke directional derivative of r at x ∈ O along the direction d ∈ R p is defined
as r • (x; d) ≜ limsup x→x, t↓0 r(x + td) -r(x) t , which is finite when r is Lipschitz continuous near x.
The Clarke subdifferential of r at x is the set ∂
C r(x) ≜ {v ∈ R p | r • (x; d) ≥ v ⊤ d, ∀d ∈ R p }.
If r is strictly differentiable at x, then ∂ C r(x) = {∇r(x)}. We say that r is Clarke regular at x ∈ O if r is directionally differentiable at x and r • (x; d) = r ′ (x; d) for all d ∈ R p . This Clarke regularity at x is equivalent to have r(x) ≥ r(x) + v⊤ (x -x) + o(∥x -x∥) for any v ∈ ∂ C r(x), which is natural satisfied when r is convex. Moreover, if a function fails to satisfy the Clarke regularity at x, there does not exist an approximate linear lower bound of the original function based on the Clarke subdifferentials with a small o error locally. Since the concept of Clarke subdifferential coincides with the usual subdifferential in convex analysis for a convex function, we simply refer to Clarke subgradient as subgradient in the remainder of the paper.
Let A : R p ⇒ R m be a set-valued mapping. Its outer limit at x ∈ R p is defined as lim sup
x→x A(x) := x ν →x lim sup ν→∞ A(x ν ) = u | ∃ x ν → x, ∃ u ν → u with u ν ∈ A(x ν ) . We say A is outer semicontinuous (osc) at x ∈ R p if lim sup x→x A(x) ⊆ A(x).
Clarke subdifferential is outer semicontinuous, which is necessary in establishing subsequential convergence, by Proposition 6.6 in (Rockafellar and Wets, 1998). In addition, for a Lipschitz r, ∂r(x) is locally bounded, see Theorem 9.13 in (Rockafellar and Wets, 1998).
A point x * ∈ R p is called a DC critical point if 0 ∈ ∂g(x * ) -∂h(x * ), or equivalently ∂g(x * ) ∩ ∂h(x * ) ̸ = ∅.
In this paper, the terminology critical point refers to DC criticality, as defined in the literature on DC programming.
Next, we review some basics of random set-valued mappings and their expectations. Let (Ω, F, P ) be a probability space, and for fixed x, let A(x, ω) : Ω → 2 R p be a general set-valued mapping taking values in closed subsets of R p . The expectation E[A(x, ω)] is defined as the set of E[A(x, ω)] over all integrable selections, where integrability follows Aumann's sense (Aumann, 1965)
. It is well defined if E[H(0, A(x, ω))] < ∞. Let r(x, ξ) : R p × Ξ → R be a random lower semicontinuous function, where ξ : (Ω, F, P ) → Ξ is a random vector with support Ξ ⊂ R m . If r is κ(ξ)-Lipschitz in x,
where E[κ(ξ)] < ∞; and for any x, r(x, ξ) is Clarke regular for a.e. ξ. Then, E[r(x, ξ)] is Clarke regular, and ∂ x E[r(x, ξ)] = E[∂ x r(x, ξ)], by Theorem 2.7.2 in (Clarke, 1990).
Throughout this paper, we assume that the sample space Ω is equipped with a metric d(•, •), making it a metric space. Let P(Ω) denote the set of Radon probability measures on Ω, where each measure P ∈ P(Ω) has a finite first moment. That is, E ξ∼P [d(ξ, ξ 0 )] < ∞ for some ξ 0 ∈ Ω.
For µ, ν ∈ P(Ω), their Wasserstein-1 distance is defined as
W 1 (µ, ν) = sup g∈Lip 1 (Ω) {E X∼µ [g(X)] -E Y ∼ν [g(Y )]} ,
where Lip 1 (Ω) denotes the set of all Lipschitz functions g : Ω → R with the Lipschitz constant 1.
this section cite: ['b9', 'b28', 'b24', 'b28', 'b28', 'b0', 'b9']

Section: The Convergence Rate for the SAA Error of Subdifferential Mappings
In this section, we establish a novel pointwise convergence rate of O( p/n) for subdifferential mappings, where p is the dimension of the variable and n is the sample size. This addresses a major challenge in subgradient-based stochastic nonsmooth problems: analyzing the sampling error of stochastic subgradients regarding the sample size.
For a random function φ(•, ω) : D φ (⊆ R p ) → R and independent and identically distributed (i.i.d.) random variables (ω 1 , . . . , ω n ) ≜ ωn drawn from the same distribution of ω, we could use 1 n n k=1 τ x, ω k as an SAA estimation of the subgradient of E ω [φ(x, ω)], where τ x, ω k is a subgradient selector that satisfies τ x, ω k ∈ ∂ x φ x, ω k .
In the smooth case, each τ x, ω k is an unbiased estimate of the expected gradient at
x, since E ω [∇ x φ(x, ω)] = ∇E ω [φ(x, ω)].
This leads to a straightforward O 1 n convergence rate for the squared error in relation to the sample size n, i.e.,
E ωn 1 n n k=1 ∇ x φ(x, ω k ) -∇E ω [φ(x, ω)] 2 ≤ σ 2 n ,(2)
where σ 2 is the uniform variance of ∇ x ϕ(x, ω k ). However, this result does not directly extend to nonsmooth set-valued subdifferentials. Some studies impose an additional assumption that for any x, τ (x, •) is Borel measurable with respect to ω, enabling a similar convergence rate to (2). In practice, however, implementing a Borel measurable subgradient selector is challenging and often infeasible.
To address this challenge, we analyze the convergence rate of the sample average subdifferential mapping ∂ φ(x) :=
1 n n k=1 ∂ x φ x, ω k to its expected counterpart ∂φ(x) = E ω [∂ x φ(x, ω)]. We define the SAA error for ∂φ(x, •) : Ω → 2 R p as ∆ n (φ, x, ωn ) ≜ H 1 n n i=1 ∂ x φ x, ω i , E ω ∂ x φ(x, ω) .
In the following, we shall develop a novel O( p/n) convergence rate (modulo logarithmic factors) for ∆ n (φ, x, ωn ).
Our results enable algorithms to select any subgradient from the sampled subdifferential set at each iteration while achieving a sampling error bound comparable to the smooth case.
We begin by introducing a lemma regarding the convergence rate of SAAs in expectation. This result is derived from the Rademacher average of the random function ψ(x, ω), as discussed in Corollary 3.2 of (Ermoliev and Norkin, 2013) and further explored in Theorem 10.1.5 of (Cui and Pang, 2021). Let r be any positive scalar. For a random function ψ(•, ω) : D ψ (⊆ [0, r] p ) → R and i.i.d. random variables (ω 1 , . . . , ω n ) = ωn drawn from the distribution of ω, we define the SAA error as δ n (ψ, ωn ) := sup x∈D ψ 1 n n i=1 ψ x, ω i -E ω ψ(x, ω) . We then have the following basic estimates, see, e.g., Theorem 3.1 in (Ermoliev and Norkin, 2013). Lemma 3.1. (Basic Estimates). If functions ψ(•, ω) are bounded by constant M and Lipschitz continuous with constant L ψ in the first variable x uniformly in ω, then for any α ∈ (0, 1/2), s > 0, it holds that
E ωn δ n (ψ, ωn ) ≤ 2 √ p L ψ r + M (1 -2α)e /n α , P √ n |δn (ψ, ωn ) -Eωn δn (ψ, ωn )| ≥ s ≤ 2 exp - s 2 2M 2 .
To analyze the asymptotic behavior of ∆ n (φ, x, ωn ), we need the following assumption.
Assumption 3.2. The function φ( •, ω) is convex and Lipschitz continuous with Lipschitz constant L φ , in terms of the first variable x ∈ D φ , uniformly in ω.
The support function of a set S is defined as σ(u, S) ≜ sup s∈S u T s. It is well known that σ(u, S) = σ(u, conv S), where conv denotes the convex hull of S. Moreover, for any nonempty sets S and S ′ , it follows from (Christian, 2002
) that σ (u, S + S ′ ) = σ(u, S) + σ (u, S ′ ) .(3)
Furthermore, the Hömander's formula, according to Theorem II-18 in (Castaing and Valadier, 1977), states that for any two nonempty convex and compact subsets A and B of R p :
D(A, B) = max ∥u∥⩽1 (σ(u, A) -σ(u, B)).(4)
Using the above formula, we derive the following lemma that converts our targeted quantity ∆ n (φ, x, ωn ) into the SAA error of support functions; see, e.g., (Xu, 2010). Its proof, as well as proofs for Theorems 3.4 and 3.5, can be found in the appendix. Lemma 3.3. Under Assumption 3.2, for any x ∈ D φ ,
∆ n (φ, x, ωn ) = max ∥u∥⩽1 1 n n i=1 σ u, ∂ x φ x, ω i -E ω [σ (u, ∂ x φ(x, ω))] .
We now derive the SAA convergence in expectation.
Theorem 3.4. Under Assumption 3.2, for any α ∈ (0, 1/2),
sup x∈Dφ E ωn [∆ n (φ, x, ωn )] ≤ c n α , where c ≜ 2 √ p(2L φ + L φ / (1 -2α)e).
Moreover, for any s > 0,
P {n α ∆ n (φ, x, ωn ) ≥ c + s} ≤ exp -s 2 / 2L 2 φ .
Remark 1. The concentration-type probabilistic results in Lemma 3.1 and Theorem 3.4 are due to McDiarmid's bounded difference inequality. They will play an important role in the proof of Theorem 3.5.
Next, we strengthen the above theorem to bound the squared SAA error, which is the key result of this section. Theorem 3.5. Under Assumption 3.2, for any α ∈ (0, 1/2), α ′ ∈ (α, 1/2) , we have
sup x∈Dφ E ωn ∆ n (φ, x, ωn ) 2 ≤ c n 2α , where c ≜ ĉ ĉ + L φ √ α ′ √ 2(α ′ -α)e + L 2 φ with ĉ ≜ √ p(2L φ + L φ / (1 -2α ′ )e).
When φ( •, ω) is smooth, Theorem 3.5 simply becomes
sup x∈Dφ E ωn ∆ n (φ, x, ωn ) 2 ≤ L 2 φ n ,
that is, c = L 2 φ and α = 1/2. This demonstrates that our result almost matches the SAA convergence rate in the smooth case. The tools we have developed here can play a crucial role in non-asymptotic convergence analysis of other (subgradient-based) stochastic nonsmooth problems. For example, it enables a "variance reduction" technique similar to that used in smooth optimization. (Bollapragada et al., 2018;Byrd et al., 2012)
this section cite: ['b14', 'b10', 'b14', 'b7', 'b37', 'b3', 'b4']

Section: The Algorithm and Convergence
Before presenting our algorithm, we first list all the needed assumptions for the stochastic functions G and H. Assumption 4.1. (Assumptions for Functions) 1. The feasible region C is convex and closed, and there exists a scalar f such that f (x) > f for all x ∈ C.
2. G( •, ξ) is ρ g -convex (ρ g ≥ 0) and H( •, ζ) is ρ h - convex (ρ h ≥ 0) over C for almost every ξ, ζ ∈ Ω. 3. G( •, ξ) is L g -Lipschitz continuous and H( •, ζ) is L h - Lipschitz continuous over C for almost every ξ, ζ ∈ Ω. 4. For all x ∈ C, G(x, • ) is L ξ -Lipschitz continuous and H(x, • ) is L ζ -Lipschitz continuous over Ω.
this section cite: []

Section: The Algorithmic Framework
We assume that at time t, the data sets S g,t ≜ {ξ t,i } Ng,t i=1 and S h,t ≜ {ζ t,i } N h,t i=1 are generated from the distributions P ξ,t and P ζ,t , respectively, where the latter distributions may not be exactly the same as the true distributions P ξ and
P ζ . Let g t (x) ≜ E ξ∼P ξ,t [G(x, ξ)], h t (x) ≜ E ζ∼P ζ,t [H(x, ζ)], and f t (x) ≜ g t (x) -h t (x).
At time t and iterate x t , we use the data from S g,t to construct a stochastic estimate ḡt (•) of the function g(•), and the data from S h,t to construct a stochastic estimate ht (x t ) of h(x t ), as well as a stochastic estimate ȳt of the subgradient ∂h(x t ). The overall estimation model Mt (•) is given by:
Mt (d) ≜ ḡt (x t + d) -ht (x t ) -ȳT t d + 1 2 µ t ∥d∥ 2 , (5
)
where µ t > 0 is the proximal parameter. The convex subproblem to be solved at iteration t is
minimize d Mt (d) subject to x t + d ∈ C.(6)
The first-order optimality condition of subproblem (6) at the unique optimal solution dt is
zt+1 -ȳt + µ t dt + vt = 0,(7)
where zt+1 ∈ ∂ḡ t (x t + dt ) and vt ∈ ∂i C (x t + dt ) with i C being the indicator function of C. Our proposed online stochastic proximal DC algorithm (ospDCA) framework is presented in Algorithm 1, while the exact rule to update the parameters µ t , N g,t , N h,t will be discussed later.
this section cite: []

Section: Algorithm 1
The ospDCA framework
1: Initialize x 0 , µ 0 , N g,0 , N h,0 . 2: for t = 0, 1, 2, • • • do 3: Generate i.i.d. samples S g,t = {ξ t,i } Ng,t i=1 and S h,t = {ζ t,i } N h,t
i=1 from P ξ,t and P ζ,t , which are independent of the past samples.
x) = 1 Ng,t Ng,t i=1 G x, ξ t,i , ht (x) = 1 N h,t N h,t i=1 H x, ζ t,i , and select ȳt ∈ ∂ ht (x t ) = 1 N h,t N h,t i=1 ∂ x H x t , ζ t,i . 5:
Solve the convex subproblem (6) to obtain dt . 6: Set x t+1 = x t + dt . 7: Update µ t+1 , N g,t+1 , N h,t+1 . 8: end for Under Assumption 4.1, it is trivial to verify that ḡt (x) and g(x) are L g -Lipschitz, ρ g -convex; and ht (x) and h(x) are L h -Lipschitz, ρ h -convex.
Let F t ≜ σ (S g,1 , S h,1 , S g,2 , S h,2 , . . . , S g,t-1 , S h,t-1 ) be a filtration, i.e., an increasing sequence of σ-fields generated by the samples used in the past t -1 iterations.
Remark 2. If there exists an isomorphic mapping ϕ from (Ω, F 1 , P ξ,t ) to (Ω, F 2 , P ζ,t ), Step 3 of Algorithm 1 can be simplified when N g,t ≥ N h,t , as follows:
1. Generate i.i.d. samples S g,t = {ξ t,i } Ng,t i=1 from the distribution of ξ, which are independent of previous samples.
2. For i = 1, 2, . . . , N h,t , set ζ t,i = ϕ(ξ t,i ) and let S h,t = {ζ t,i } N h,t i=1 . A similar procedure applies when N g,t < N h,t .
this section cite: []

Section: Convergence Analysis
In this section, we present the convergence result of Algorithm 1 based on Assumptions 4.1. A brief outline of the convergence analysis is provided in the main text, with detailed proofs available in the appendix.
We first analyze the inexact sufficient descent property at the t-th iteration and derive the following inequality. The result and its proof is similar to the deterministic case, see, e.g., Theorem 3 in (Tao and An, 1997) and Theorem 3.7 in (Tao and An, 1998).
Lemma 4.2. (The Sufficient Descent Property) For any y t ∈ ∂h t (x t ), the step x t+1 from Algorithm 1 satisfies
f t (x t )-f t (x t+1 ) ≥ (y t -ȳt ) T dt + µ t + ρ g + ρ h 2 ∥ dt ∥ 2 + g t (x t ) -ḡt (x t ) -g t (x t+1 ) + ḡt (x t+1 ).
To further the analysis, the SAA error bound derived in Section 3 comes into play. By Lemma 3.1, we could derive the SAA error estimation for g t (x t ) -g t (x t+1 ) as follows.
Corollary 4.3. For any α g ∈ (0, 1/2), we have
E |ḡ t (x t+1 ) -ḡt (x t ) -g t (x t+1 ) + g t (x t )| F t ≤ C g µ t N αg g,t , where C g = 4 √ pL g (L g + L h ) 2 + Lg √ (1-2αg)e .
Remark 3. Note that we relax the assumption that G(x, ξ) is globally uniformly bounded, as posed in Le Thi et al. (2024). Instead, we use the proximal term µ t to ensure that dt does not become too large. This guarantees that G(x t , ξ) -G(x t+1 , ξ) remains uniformly bounded with respect to µ t , which facilitates our SAA error analysis of g t (x t )-g t (x t+1 ) (see the proof of Corollary 4.3 for details).
The SAA error estimation for ∂h t (x t ) is a direct corollary of Theorem 3.5:
Corollary 4.4. For any α h ∈ (0, 1), α ′ h ∈ (α h , 1), we have
sup x∈C E D 2 ∂ ht (x t ), ∂h t (x t ) |F t ≤ C h n α h , where C h = Ĉh Ĉh + L h √ α ′ h √ 2(α ′ h -α h )e + L 2 h with Ĉh = √ p(2L h + L h / (1 -α ′
h )e). Remark 4. With regard to the estimation error from sampling, Liu et al. (2022) assumes that the variance of the stochastic objectives is bounded. Similarly, Berahas et al. ( 2021) needs an unbiased gradient estimation with bounded variance in the study of stochastic sequential quadratic programming. Sequential quadratic programming is extended to the nonsmooth DC problems with smooth convex component in the deterministic Wang and Petra (2023) and stochastic settings Wang et al. (2023), where again a bounded variance of subgradient estimation is required. In Shashaani et al. (2018), the Monte Carlo estimate of the objective is also assumed to be unbiased, and its variance is uniformly bounded. The tools developed in Section 3 provide a tight SAA bound for ∂h, allowing us to derive a result analogous to the one in smooth optimization discussed above.
In the following lemma, we present the sufficient descent property in expectation. Lemma 4.5. At the t-th iteration, the following stands for any c > 0:
E [f t (x t ) -f t+1 (x t+1 ) | F t ] ≥ µ t + ρ g + ρ h 2 -c E dt 2 | F t - C g µ t N αg g,t - C h 4cN α h h,t -L ξ W 1 (P ξ,t+1 , P ξ,t ) -L ζ W 1 (P ζ,t+1 , P ζ,t ), (8)
where α g ∈ (0, 1/2) and α h ∈ (0, 1) with corresponding constants C g and C h defined in Corollaries 4.3 and 4.4.
The following analysis is conducted under the key assumptions stated below. Remark 5. Since the Wasserstein-1 distance of some common distributions is easy to calculate or control, it is not hard to construct examples of time-varying exponential or uniform distributions that satisfy this assumption. A simple example is the regression problem with finite number of outliers or finite times of distribution shifts (due to the change of environment). An example of online sparse robust regression will be provided in Section 6, where time-varying multivariate normal distributions satisfying the above assumption are considered.
Assumption 4.7. (Assumptions for Parameters) (a) There exist 0 < μ < μ such that μ ≤ µ t ≤ μ, ∀t ≥ 0.
(b) There exist α g ∈ (0, 1/2), α h ∈ (0, 1) such that
t≥0 1 N αg g,t + 1 N α h h,t < ∞.(9)
Now, we are ready to present the squared summable property of the iteration step { dt }, and its almost sure convergence to zero. These results are important for the later analysis.
Theorem 4.8. Under Assumptions 4.6 and 4.7, we have
lim t→∞ E   t≥0 dt 2 |F 0   < ∞, hence E dt |F 0 → 0.
Furthermore, lim t→∞ ∥ dt ∥ = 0 with probability 1.
To proceed, we first provide a technical Lemma, which concerns the law of large numbers (LLN) for SAA sequence.
Lemma 4.9. Under Assumptions 4.6 and 4.7, for any fixed R > 0, x ∈ C, x ∈ B(x, R), the following limits hold as t → ∞ with probability 1:
ḡt (x) -ḡt (x) -(g t (x) -g t (x)) → 0, ht (x) -ht (x) -(h t (x) -h t (x)) → 0.
We are ready to present our main convergence result, which is the best that can be achieved under stochastic nonconvex and nonsmooth conditions.
Theorem 4.10. Under Assumptions 4.6 and 4.7, every accumulation point of the sequence {x t } produced by Algorithm 1 is a DC critical point of f with probability 1.
The above theorem only provides the asymptotic convergence of the algorithm, not the non-asymptotic complexity.
The known complexity of the deterministic dc algorithm in (Le Thi et al., 2020) requires a smoothness assumption on either g or h. We left it as a future work to derive the iteration complexity of our proposed algorithm.
The sample size requirement of our algorithm is presented in (9). Notably, the bounds on exponents α g and α h are different. To provide some intuition, this difference arises from the DC structure and the improved convergence rate of the SAA error for the subdifferential mapping. Specifically, linearizing the function h couples the SAA error of ∂h with the stepsize dt , as demonstrated in Lemma 4.2. By applying the Cauchy-Schwarz inequality, we elevate the SAA error of ∂h from first-order to second-order in expectation (see Lemma 4.5), for which Theorem 3.5 establishes the tight convergence rate.
According to Assumption 4.7 (a), the proximal terms µ t for each DC subproblem can be pre-selected arbitrarily, as long as they are upper and lower bounded by positive constants. Regarding the sample size requirement given in Assumption 4.7 (b), this is inherent to our approach and difficult to avoid, as it ensures the necessary accuracy of the algorithm at each step. The choice of sample sizes and step sizes remains an active research topic in stochastic optimization. Even for stochastic gradient descent applied to smooth optimization problems, a non-diminishing step size selection requires sublinearly increasing sampling sizes to guarantee convergence.
this section cite: ['b33', 'b32']

Section: An Adaptive Sampling Algorithm
In this section, we introduce an adaptive sampling strategy for updating µ t , N g,t , N h,t in Algorithm 1. As discussed in the convergence analysis, the key requirement is to ensure that Assumption 4.7 holds. Since Assumption 4.7 (a) is relatively easy to satisfy, we mainly focus on developing strategies to satisfy Assumption 4.7 (b). Given pre-determined constants c l , c µ > 0, a common approach is to increase the sample sizes sublinearly based on the following condition:
Condition 5.1. Suppose that Ng,t and Nh,t are pre-defined such that t≥0 N -α h h,t + N -αg g,t < ∞, we say that the Summable Condition holds at the t-th iteration if the parameters c t , µ t , N g,t , N h,t are chosen to satisfy:
N g,t ≥ Ng,t , N h,t ≥ Nh,t and c l ≤ c t ≤ µ t + ρ g + ρ h 2 -c µ .(10)
However, these pre-determined sample sizes do not adapt to the algorithm's progress at each iteration. In the following, we introduce a practical condition that determines N g,t and N h,t based on the optimization path. Intuitively, a larger stepsize in the early iterations suggests that the current point is far from critical points when less precise but computationally cheaper estimates are sufficient. In contrast, as the algorithm nears the critical points, the stepsize decreases, requiring more accurate estimations to ensure both theoretical guarantees and practical performance. Building on this intuition, we propose a practical Stepsize Norm Condition for adaptive sampling, where the sample size at each iteration is determined by the current stepsize.
Condition 5.2. We say that Stepsize Norm Condition stands at the t-th iteration if parameters c t , µ t , N g,t , N h,t are selected to satisfy:
(µ t-1 -c µ -c t-1 ) ∥ dt-1 ∥ 2 ≥ Cg µtN αg g,t + C h 4ctN α h h,t , c t ≤ µ t + ρg+ρ h 2 -c µ .(11)
Remark 6. Here, c t acts as an intermediate variable for parameter updates, linking others to ensure convergence. These variables serve only to determine µ t , N g,t , and N h,t .
As presented in the following theorem, Assumption 4.7 (b) stands when either condition is satisfied. This plays a critical role in designing a practical adaptive ospDCA with convergence guarantee. Compared to the gradient accuracy condition and other variance-based tests in (Byrd et al., 2012;Bollapragada et al., 2018), our adaptive sampling scheme is not only practically implementable but also backed by rigorous theoretical guarantees.
Theorem 5.3. If either Summable Condition (10) or Stepsize Norm Condition (11) is satisfied for sufficiently large t, and Assumptions 4.7 (a) and 4.6 stand, then Assumption 4.7 (b) stands. To eliminate the intermediate variable c t and adapt the algorithm for any predetermined sequence {µ t } satisfying 0 < μ ≤ µ t ≤ μ, we propose a simplified algorithm by fixing c t = ρg+ρ h 2 + μ 4 = c l and setting c µ = μ 4 , as detailed in Algorithm 2. The complete version of the adaptive sampling ospDCA can be found in the appendix; see Algorithm 3.
Remark 7. Consider the subproblem when updating the sample size N g,t and N h,t . In order to minimize the total number of samples, one could derive that N h,t =
2C h µt+1 Cg(2ρg+2ρ h +μ) N 3/4 g,t .
Hence the optimal order of N h,t is O(N 3/4 g,t ). Furthermore, if the updating rule of N g,t and N h,t is based on this result, then sample size upper bound sequence Nh,t is no longer required.
this section cite: ['b4', 'b3']

Section: An Application: Online Sparse Robust Regression
We consider the online linear regression problem with a robust loss and sparsity-promoting DC regularization. Given streaming data {(x i , y i )} ∞ i=1 drawn from unknown and varying distributions D t , the optimization problem is formulated as minimizing the expected objective:
min β∈R p E (x,y)∼Dt [|y -⟨β, x⟩|] + λ p j=1 min(1, α|β j |).
The regularization term p j=1 min(1, α|β j |) is a cappedℓ 1 penalty, which approximates the sparsity-inducing ℓ 0norm. To facilitate optimization, we use the following DC Algorithm 2 Adaptive ospDCA Require: Initial point x 0 , error estimation parameter α g ∈ (0, 1/2), α h ∈ (0, 1) with corresponding C g , C h defined in Corollaries 4.3 and 4.4, sample size upper bound sequence { Ng,t } and { Nh,t } which satisfy
t≥0 N -α h h,t + N -αg g,t < ∞, predetermined proximal parameters {µ t } with upper bound μ and lower bound μ. 1: for t = 0, 1, 2, • • • do 2: Generate i.i.d. samples {ξ t,i } Ng,t i=1 and {ζ t,i } N h,t i=1
from the distribution of ξ and ζ, which are independent of the past samples.
3: Set ḡt (x) = 1 Ng,t Ng,t i=1 G x, ξ t,i , ht (x) = 1 N h,t N h,t i=1 H x, ζ t,i , and select ȳt ∈ ∂ ht (x t ). 4:
Solve the convex subproblem to obtain dt :
minimize d ḡt (x t + d) -ht (x t ) -ȳT t d + 1 2 µ t ∥d∥ 2 subject to x t + d ∈ C. 5: Set x t+1 = x t + dt . 6:
Update N g,t+1 and N h,t+1 such that one of the followings stands:
1
. µt -μ 2 ∥ dt∥ 2 ≥ Cg µ t+1 N αg g,t+1 + C h (2ρg +2ρ h + μ)N α h h,t+1 , 2. N g,t+1 ≥ Ng,t+1 , and N h,t ≥ Nh,t+1 . 7: end for decomposition: min(1, α|β j |) = 1+α|β j |-max(1, α|β j |).
Thus, the final problem formulation in expectation form is:
min β∈R p E (x,y)∼Dt [G(β, x, y)] -h(β), where G(β, x, y) = |y -⟨β, x⟩| + λ p j=1 (1 + α|β j |) , h(β) = p j=1 max(1, α|β j |).
This expectation-based formulation enables efficient online optimization, making it well-suited for large-scale and streaming data scenarios.
Baselines. We implemented four baselines to compare with our proposed adaptive ospDCA. The first one is ospDCA with a pre-determined, sublinearly growing sample size of t 2.1 per iteration, without adaptivity. The second baseline is S(p)DCA, introduced in Le Thi et al. (2024), where we added an additional proximal term. This algorithm draws one new sample per iteration and uses aggregated samples to construct sample averages. The third and fourth baselines are ospDCA with a fixed sample size per iteration, using 100 and 1000 new samples for SAA, respectively.
this section cite: []

Section: Datasets and Setup.
For the problem, we set α = 1, λ = 0.01, and generate synthetic datasets. Specifically, at each time step t, the feature vector x t is sampled uniformly from  Sample size for experiment (a). Sample size for experiment (b). [-1, 1] p . The corresponding label is given by
y t = x ⊤ t (β opt + δ t ) + ε,
where β opt is a known sparse optimal solution with nonzero entries at specific locations, ε ∼ N (0, 1) represents additive noise, and δ t denotes a time-dependent distribution shift.
It follows that W 1 (D t , D t+1 ) ≤ ∥δ t -δ t+1 ∥ 1 .
In order to ensure that the cumulative Wasserstein-1 distance for D t remains bounded, we set δ t = (-1) t 100t -2 1 p , where 1 p represents a p-dimensional column vector where all entries are equal to 1. We initialize β at zero, set the proximal coefficient µ t = 1, α g = 0.45 ∈ (0, 1/2), and run the experiment until a predefined runtime limit is reached. It is straightforward to verify that G(•, x, y) is 1-Lipschitz for every x, y, and h(•) is λα-Lipschitz. Furthermore, if we impose a bounded constraint on β, then G(β, •, •) is also uniformly Lipschitz in (x, y) for every β.
Results. We evaluate the performance by tracking the distance between the current iterate β t and the optimal solution β opt . We plot the evolution of convergence error and computational time in Figures 1 and 3. Across all experiments, the performance of adaptive ospDCA consistently surpasses the baseline methods. This demonstrates that our proposed algorithm significantly improves convergence efficiency.
During early iterations, the sample size of adaptive ospDCA is relatively small, leading to reduced precision but higher computational efficiency. As the iteration points approach the optimal solution, the sample size increases to enhance estimation accuracy. This transition leads to faster progress in later iterations, ultimately surpassing other algorithms. Compared to its non-adaptive counterpart, adaptive ospDCA invests more time and samples in the later iterations (which are closer to the optimal and thus more important), as illustrated in Figures 2 and 4. The adaptivity makes it more efficient overall and more robust to distribution shifts. Additional experimental results are provided in the appendix.
this section cite: []

Section: Conclusion
In this work, we propose an efficient online adaptive sampling algorithm for stochastic nonsmooth difference-ofconvex (DC) optimization problems with time-varying data distributions. The algorithm relies only on samples drawn from the distribution at the current iterate and adopts distinct adaptive sampling rates for the convex and concave components of the DC objective. We further prove that, under mild convergence conditions on the non-stationary distributions, the generated sequence almost surely has a subsequence that converges to a DC critical point. One of the core contribution of this paper lies in generalizing previous results in the field of stochastic online DC optimization to a broader class of nonsmooth DC problems with time-varying distributions, while maintaining a sampling size requirement comparable to the smooth case. Numerical experiments demonstrate that our algorithm performs well on online sparse robust regression tasks.
this section cite: []

Section: References
Ref_id:b0 Title: Integrals of set-valued functions Year: (1965)
Ref_id:b1 Title: Sequential quadratic optimization for nonlinear equality constrained stochastic optimization Year: (2021)
Ref_id:b2 Title: Nonstationary stochastic optimization Year: (2015)
Ref_id:b3 Title: Adaptive sampling strategies for stochastic optimization Year: (2018)
Ref_id:b4 Title: Sample size selection in optimization methods for machine learning Year: (2012)
Ref_id:b5 Title: On the global convergence of trust region algorithms using inexact gradient information Year: (1991)
Ref_id:b6 Title: Global convergence rate analysis of unconstrained optimization methods based on probabilistic models Year: (2018)
Ref_id:b7 Title: Measurable multifunctions Year: (1977)
Ref_id:b8 Title: Chapter 14 -set-valued integration and set-valued probability theory: An overview Year: (2002)
Ref_id:b9 Title: Optimization and Nonsmooth Analysis Year: (1990)
Ref_id:b10 Title: Modern Nonconvex Nondifferentiable Optimization Year: (2021)
Ref_id:b11 Title: Adaptive stochastic optimization: A framework for analyzing stochastic optimization algorithms Year: (2020)
Ref_id:b12 Title: Stochastic modelbased minimization of weakly convex functions Year: (2018)
Ref_id:b13 Title: Graphical convergence of subgradients in nonconvex optimization and learning Year: (2022)
Ref_id:b14 Title: Sample average approximation method for compound stochastic optimization problems Year: (2013)
Ref_id:b15 Title: Learning rate schedules in the presence of distribution shift Year: (2023-07)
Ref_id:b16 Title: Uniform convergence of gradients for non-convex learning and optimization Year: (2018)
Ref_id:b17 Title: High probability complexity bounds for line search based on stochastic oracles Year: (2021)
Ref_id:b18 Title: Dc programming and dca: thirty years of developments Year: (2018)
Ref_id:b19 Title: Stochastic dca for minimizing a large sum of DC functions with application to multi-class logistic regression Year: (2020)
Ref_id:b20 Title: Stochastic difference-of-convexfunctions algorithms for nonconvex programming Year: (2022)
Ref_id:b21 Title: Online stochastic DCA with applications to principal component analysis Year: (2024)
Ref_id:b22 Title: Solving nonsmooth and nonconvex compound stochastic programs with applications to risk measure minimization Year: ()
Ref_id:b23 Title: The landscape of empirical risk for nonconvex losses Year: (2018)
Ref_id:b24 Title: Variational Analysis and Generalized Differentiation I: Basic Theory Year: (2006)
Ref_id:b25 Title: A Regularization of DC Optimization Year: (2022)
Ref_id:b26 Title: Stochastic Difference of Convex Algorithm and its Application to Training Deep Boltzmann Machines Year: (2017-04)
Ref_id:b27 Title: On the theory of banach space valued multifunctions. 1. integration and conditional expectation Year: (1985)
Ref_id:b28 Title: Variational Analysis Year: (1998)
Ref_id:b29 Title: Subgradient convergence implies subdifferential convergence on weakly convex functions: With uniform rates guarantees Year: (2024)
Ref_id:b30 Title: Online robust non-stationary estimation Year: (2023)
Ref_id:b31 Title: Stochastic Programming by Monte Carlo Simulation Methods Year: (2000)
Ref_id:b32 Title: Kaizhao Sun and Xu Andy Sun. Algorithms for differenceof-convex programs based on difference-of-moreauenvelopes smoothing Year: (1998)
Ref_id:b33 Title: Convex analysis approach to dc programming: theory, algorithms and applications Year: (1997)
Ref_id:b34 Title: Stochastic DCA for the Large-sum of Non-convex Functions Problem and its Application to Group Variable Selection in Classification Year: (2017-08)
Ref_id:b35 Title: A sequential quadratic programming algorithm for nonsmooth problems with upper-C 2 objective Year: (2023)
Ref_id:b36 Title: A sequential quadratic programming method for nonsmooth stochastic optimization with upper-C 2 objective Year: (2023)
Ref_id:b37 Title: Uniform exponential convergence of sample average random functions under general sampling with applications in stochastic programming Year: (2010)
Ref_id:b38 Title: Stochastic optimization for dc functions and non-smooth non-convex regularizers with non-asymptotic convergence Year: (2019)
Ref_id:b39 Title: Large-scale optimization of partial auc in a range of false positive rates Year: (2022)
Ref_id:b40 Title: • • • , 0], δt = (-1) t 5000t -2 1p. Figure 3. Algorithm behavior for online sparse robust regression Year: (2000)
