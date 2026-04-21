Title: BayeSQP: Bayesian Optimization through Sequential Quadratic Programming
Abstract: We introduce BayeSQP, a novel algorithm for general black-box optimization that merges the structure of sequential quadratic programming with concepts from Bayesian optimization. BayeSQP employs second-order Gaussian process surrogates for both the objective and constraints to jointly model the function values, gradients, and Hessian from only zero-order information. At each iteration, a local subproblem is constructed using the GP posterior estimates and solved to obtain a search direction. Crucially, the formulation of the subproblem explicitly incorporates uncertainty in both the function and derivative estimates, resulting in a tractable second-order cone program for high probability improvements under model uncertainty. A subsequent one-dimensional line search via constrained Thompson sampling selects the next evaluation point. Empirical results show that BayeSQP outperforms state-of-the-art methods in specific high-dimensional settings. Our algorithm offers a principled and flexible framework that bridges classical optimization techniques with modern approaches to black-box optimization.

Section: Introduction
In recent years, Bayesian optimization (BO) has emerged as a powerful framework for black-box optimization ranging from applications in robotics [9,6,44] to hyperparameter tuning [55,12] and drug discovery [21,40,10]. To address high-dimensional problems emerging in these fields, a variety of high-dimensional BO (HDBO) approaches have been proposed, including the use of local BO (LBO) methods [15,43] or methods that exploit specific structure in the objective [13]. Recently, a growing debate has emerged over whether such HDBO methods are truly necessary, given that appropriate scaling of the prior can already yield strong performance on certain high-dimensional benchmarks [30,67]. However, as shown by Papenmeier et al. [48], these approaches solve numerical issues in the hyperparameter optimization of the Gaussian process (GP) surrogate but their success can still be attributed to emerging local search behavior. We argue that it is not a matter of choosing either HDBO approaches or standard approaches, but rather of leveraging recent advances in how to achieve numerical stability also for HDBO methods.
Building on this perspective, we aim to integrate the strengths of established classical optimization techniques within the HDBO framework. Specifically, we extend the widely-adopted local method for HDBO GIBO [43,45,65,16,23]-which can be interpreted as combining BO with first-order optimization methods-to LBO with second-order methods. We introduce BayeSQP, a novel algorithm for black-box optimization that merges the structure of sequential quadratic programming (SQP) with concepts from BO. BayeSQP employs GP surrogates for both the objective and constraints that jointly model the function values, gradients, and Hessians from only zero-order information (Figure 1). At each iteration, a local subproblem is constructed using the GP posterior estimates and solved to 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
Sequential Quadratic Programming Bayesian Optimization BayeSQP solve sub-problem f (x) ∇f (x) x ∇ 2 f (x) build sub-problem learn surrogate models Thompson sampling as line search obtain a search direction. Through constrained Thompson sampling, we select the point for the next iteration. In summary, the key contributions of this paper are: C1 A novel algorithm BayeSQP leveraging GP surrogates to utilize the structure of classic SQP within BO for efficient high-dimensional black-box optimization with constraints. C2 An uncertainty-aware subproblem for BayeSQP that accounts for the variance and covariance in function and gradient estimates, resulting in a tractable second-order cone program. C3 Empirical experiments demonstrating that BayeSQP outperforms state-of-the-art BO methods in specific high-dimensional constrained settings.
this section cite: ['b8', 'b5', 'b43', 'b54', 'b11', 'b20', 'b39', 'b9', 'b14', 'b42', 'b12', 'b29', 'b66', 'b47', 'b42', 'b44', 'b64', 'b15', 'b22']

Section: Problem formulation
We consider the problem of finding an optimizer to the general non-convex optimization problem
x * = arg min x∈X f (x) subject to c i (x) ≥ 0, ∀i ∈ I m := {1, . . . , m}
where f : X → R and constraints c i : X → R for all i ∈ I m are black-box functions defined over the compact set X ⊂ R d . At each iteration t ∈ I T where T is the total budget for the optimization, an algorithm selects a query point x t ∈ X and receives noisy zeroth-order feedback following the standard observation model in BO as f t = f (x t ) + ε f for the objective, and c i,t = c i (x t ) + ε ci for all i ∈ I m for the constraints where ε f and ε ci are independent realizations from a zeromean Gaussian distribution with possibly different noise variances. From these observations, we construct independent datasets for the objective function D t f = {(x j , f j )} t j=1 and for each constraint D t ci = {(x j , c i,j )} t j=1 for all i ∈ I m , which any zero-order method can leverage to solve (1).
this section cite: []

Section: Preliminaries

this section cite: []

Section: Sequential quadratic programming
SQP represents a powerful framework for solving nonlinear constrained optimization problems by iteratively solving quadratic subproblems. This method has become one of the most effective techniques for handling a wide range of optimization problems. The foundation of constrained optimization rests on the Lagrangian function, defined as L(x, ξ) = f (x) -m i=1 ξ i c i (x). It combines the objective with the constraints, where each constraint is weighted by its Lagrange multiplier ξ i . Solving the optimization problem involves satisfying the Karush-Kuhn-Tucker conditions for this Lagrangian. To achieve this, at each iteration t, SQP constructs a quadratic approximation of the Lagrangian using the Hessian H t = ∇ 2  xx L(x t , ξ) (or an appropriate approximation thereof) and linearizes the constraints around the current point x t . This generates the following subproblem:
p t = arg min p∈R d 1 2 p ⊤ H t p + ∇f (x t ) ⊤ p + f (x t )
subject to ∇c i (x t ) ⊤ p ≥ -c i (x t ), ∀i ∈ I m
The solution p t provides a search direction, and the next iterate is typically computed as x t+1 = x t + α t p t , where α t is a step size determined by an appropriate line search procedure that ensures adequate progress toward the optimum. For an overview of classical SQP methods, see [46].
Under standard assumptions, SQP exhibits local superlinear convergence when using exact Hessian information, and various quasi-Newton approximation schemes (such as BFGS or SR1 updates, cf. [46]) can maintain good convergence properties while reducing computational overhead. This fast local convergence also makes it interesting for HDBO. However, the key challenge here is that usually only zero-order information on the objective and constraints is available.
this section cite: ['b45', 'b45']

Section: Gaussian processes
GPs are a powerful and flexible framework for modeling functions in a non-parametric way. A GP is defined as a collection of random variables, any finite number of which have a joint Gaussian distribution. Formally, a GP is defined by its mean function m(x) := E[f (x)] and kernel k(x, x ′ ) := Cov[f (x), f (x ′ )] [53]. In BayeSQP, we use GPs to model the objective function f and the constraints c i as standard in BO [18]. Contrary to standard BO, we aim to leverage the following property of GPs: They are closed under linear operations, i.e., the derivative of a GP is again a GP given that the kernel is sufficiently smooth [53]. This enables us to derive a distribution for the gradient and Hessian. We can formulate the following joint prior distribution:
   y f ∇f ∇ 2 f    ∼ N       m(X) m(x) ∇m(x) ∇ 2 m(x)    ,    k(X, X) + σ 2 I • • • k(x, X) k(x, x) • • ∇k(x, X) ∇k(x, x) ∇ 2 k(x, x) • ∇ 2 k(x, X) ∇ 2 k(x, x) ∇ 3 k(x, x) ∇ 4 k(x, x)       (3
)
where y ∈ R n are the n function observations, X = [x 1 , . . . , x n ] ∈ R d×n is the matrix of all training inputs with x i ∈ R d , and x ∈ R d is the test point. 1 Here and in the following, we use • to denote symmetric entries for improved readability. The joint conditional distribution is then:
1   f ∇f ∇ 2 f   x, X, y ∼ N   µ f (x) µ ∇f (x) µ ∇ 2 f (x) ,   σ 2 f (x) • • Σ ∇f,f (x) Σ ∇f (x) • Σ ∇ 2 f,f (x) Σ ∇ 2 f,∇f (x) Σ ∇ 2 f (x)     .(4)
Following standard conditioning of multivariate normal distributions, we can directly compute the mean and covariance functions of the marginals of the posterior as
Marginal GP of f : µ f (x) = m(x) + k(x, X)K -1 (y -m(X)) ∈ R, σ 2 f (x) = k(x, x) -k(x, X)K -1 k(X, x) ∈ R (5
)
Marginal GP of ∇f : µ ∇f (x) = ∇m(x) + ∇k(x, X)K -1 (y -m(X)) ∈ R d , Σ ∇f (x) = ∇ 2 k(x, x) -∇k(x, X)K -1 ∇k(X, x) ∈ R d×d(6)
Here, we defined the Gram matrix as K := k(X, X) + σfoot_1 I with entries [k(X, X)] ij = k(x i , x j ) for i, j ∈ I n , and use the notation that k(X, x) ∈ R n×1 is the vector of kernel evaluations between each training point and the test point, with entries [k(X, x)] i = k(x i , x). Similarly, we can obtain the covariance term Σ ∇f,f (x) as well as the mean estimate of the Hessian. 2 As noted in Müller et al. [43], we must perform the inversion of the Gram matrix K only once. So, while calculating the gradient distribution and the mean of the Hessian is not for free, the additional computational overhead is limited with increasing data set size.
this section cite: ['b52', 'b17', 'b52', 'b42']

Section: Related work
Scalable Bayesian optimization For long, BO has been considered challenging for highdimensional input spaces leading to the development of tailored algorithms for this setting. Such approaches include LBO methods, which we will discuss in more detail in the following, as well as methods that aim to leverage a potential underlying structure or lower dimensional effective dimensionality [33,62,13,47]. Recent results show that some of the core challenges in this highdimensional setting are due to a numerical issues when optimizing the hyperparameters, which can be in part addressed by enforcing larger lengthscales [30,67,48]. These developments do not make scalable approaches obsolete. Rather, we see them as a tool to further improve the modeling also for scalable BO approaches. To address scalability, in the sense of scaling with data, alternative surrogates for BO such as neural networks [56,35,8] or sparse GPs [42,41] have been discussed; addressing these scalability issues, however, is not the focus of this work.
Local Bayesian optimization LBO methods aim to improve the efficiency of the optimization process by focusing on local regions of the search space. Approaches such as TuRBO [15] and SCBO [14] can be classified as pseudo-local methods: their trust-region approach still allows for the exploration of multiple local areas and only over time collapses to one local region. On the contrary, Müller et al. [43] introduced with GIBO a new paradigm of LBO combining gradient-based approaches with BO. Since then, the algorithm has been modified with different acquisition functions to actively learn the gradient [45,58,23,16], theoretically investigated [65], and extended with crash constraints [61]. This class of algorithms operates fully locally. Our algorithm BayeSQP can also be classified as such a local method. In this sense, BayeSQP extends GIBO to second-order optimization by using a Hessian approximation from a GP. Similar ideas have been leveraged in a quasi-Newton methods [11]. However, by incorporating ideas from SQP, BayeSQP is directly applicable to both unconstrained and constrained optimization problems-something which is not possible with GIBO.
this section cite: ['b32', 'b61', 'b12', 'b46', 'b29', 'b66', 'b47', 'b55', 'b34', 'b7', 'b41', 'b40', 'b14', 'b13', 'b42', 'b44', 'b57', 'b22', 'b15', 'b64', 'b60', 'b10']

Section: Bayesian optimization and Gaussian processes in classical optimization
There have been various papers integrating BO with first-order optimization, e.g., for line search [38,57]. GPs have been successfully applied and leveraged in optimization-both for local optimization [25,24] and global optimization (essentially BO) [32,18]. All of these can be classified as a subfield of probabilistic numerics [27,28]. Similar to our approach, Gramacy et al. [20] merged classical methods with BO by lifting the constraints into the objective using an augmented Lagrangian approach which later got extended to a slacked [49] and recently a relaxed version [4]. These approaches are based on expected improvement (EI) and, crucially, Eriksson and Poloczek [14] showed that these approaches do not scale well to high-dimensional problems. BayeSQP differs in the type of acquisition function for the line search as well as the framework as it builds on SQP. To our knowledge, we are the first to leverage a joint GP model of the function, its gradient and Hessian in a classical framework.
this section cite: ['b37', 'b56', 'b24', 'b23', 'b31', 'b17', 'b26', 'b27', 'b19', 'b48', 'b3', 'b13']

Section: BayeSQP: Merging classic SQP and Bayesian optimization
This paper proposes the LBO approach BayeSQP. As described above, the main objects of this approach are GP models of the objective and possible constraints that jointly model the function value, the gradient as a well as the Hessian in a single model. BayeSQP then leverages this model at each iteration to construct a quadratic uncertainty-aware subproblem for a search direction that yields improvement with high probability. In the following, we will first discuss our modeling approach. Based on this, we will construct the subproblem, followed by a discussion on line search. In the end, we touch on further practical extensions and give intuition on the optimization behavior.
this section cite: []

Section: Second-order Gaussian processes as surrogate models for BayeSQP
In BayeSQP, we aim to leverage ideas from both SQP and BO to solve constrained black-box optimization problems as in (1). For this, we will model the objective and all constraints using second-order GP models introduced in Section 3.2 here stated for the objective:
  f ∇f vec(∇ 2 f )   x, X, y ∼ N   µ f (x) µ ∇f (x) vec(µ ∇ 2 f (x)) ,   σ 2 f (x) • × Σ ∇f,f (x) Σ ∇f (x) × × × ×     (7
)
We use surrogate models of the same form for each constraint c i (x). We do not compute the covariance of the Hessian (×) due the scaling issues with dimensions discussed in Section 3.2. Figure 2 demonstrates the effectiveness of such a joint GP model. We can estimate the gradient, identify local optima, and estimate curvature all from only zeroth-order information.
x 0 x 1 Zero-order prediction Training data Test points P1 P2 P3 x 0 Gradient field P1 P2 P3 x 0 Hessian determinant Critical point boundary P1 P2 P3 f (x) = 1.003 ∇f = 0.85 -0.20 λ 1 = -22.079 λ 2 = -17.391 Mean curvature and estimates P1 f (x) = -0.073 ∇f = -3.15 -2.61 λ 1 = -0.316 λ 2 = 3.567 P2 f (x) = -0.965 ∇f = -0.10 0.26 λ 1 = 17.689 λ 2 = 21.351
this section cite: ['b0']

Section: P3
Figure 2: The power of Gaussian processes. Although we only have zeroth-order information about the function, the differentiability of the GP allows us to estimate both the gradient and curvature. All estimates provided are in expectation; the associated uncertainties are not shown.
Crucially, it is not required to always evaluate the full posterior distribution for each test point. In a SQP framework, we can approximate the Hessian of the Lagrangian once at our current iterate as
H t = µ ∇ 2 f (x t ) - m i=1 ξ (t-1) i µ ∇ 2 ci (x t ),(8)
where
ξ (t-1) i
are the Lagrange multipliers from the solution of the last subproblem, but for the subsequent line search, we can directly work with the cheap marginal GP f ∼ N (µ f (x), σ 2 f (x)).
this section cite: []

Section: Deriving the subproblem for BayeSQP
Standard SQP approaches typically require exact knowledge of the objective function, constraints, and their respective gradients. In our case, we only have access to zero-order feedback and the question arises how to formulate a suitable subproblem given our choice of surrogate model.
Expected value SQP subproblem A straightforward approach is to simply formulate a subproblem using expectations, leading to the following expected value subproblem:
p t ∈ arg min p∈R d E 1 2 p ⊤ H t p + ∇f (x t ) ⊤ p + f (x t ) subject to E c i (x t ) + ∇c i (x t ) ⊤ p ≥ 0, ∀i ∈ I m .(9)
While intuitive, this formulation fails to account for the inherent uncertainty in the estimates. As discussed by Nguyen et al. [45] and He et al. [23], taking into account the uncertainty of, e.g., the gradient, can be crucial for improving with high probability for LBO approaches.
this section cite: ['b44', 'b22']

Section: Uncertainty-aware SQP subproblem
To address this limitation, we reformulate the standard QP subproblem into a robust version that explicitly accounts for uncertainty in the estimates:
p t ∈ arg min p∈R d VaR 1-δ f 1 2 p ⊤ H t p + ∇f (x t ) ⊤ p + f (x t )
Objective value-at-risk with confidence level 1-δ f subject to
P c i (x t ) + ∇c i (x t ) ⊤ p ≥ 0 ≥ 1 -δ c
Constraint satisfaction with confidence 1-δc , ∀i ∈ I m .
This formulation accounts for uncertainty through two mechanisms: employing value-at-risk (VaR) for the objective function and enforcing probabilistic feasibility for the constraints. The resulting search direction minimizes the worst-case objective value while ensuring the constraints are satisfied with high probability.
this section cite: []

Section: Tractability through joint Gaussian process
The robust formulation in (10) remains intractable without distributional assumptions. By modeling the objective and constraints as jointly Gaussian with their gradients, we can transform (10) into a deterministic second-order cone program. Next, we derive this tractable reformulation for the constraints; the objective follows analogously.
For the constraints, we aim to ensure that P c i (x t ) + ∇c i (x t ) ⊤ p ≥ 0 ≥ 1 -δ c . Since we have z ⊤ v ∼ N (µ ⊤ z v, v ⊤ Σ z v) for a multivariate Gaussian random variable z ∼ N (µ z , Σ z ) and v is a deterministic vector, we know that c i (x t ) + ∇c i (x t ) ⊤ p is also normal distributed with moments
E c i (x t ) + ∇c i (x t ) ⊤ p = µ ci (x t ) + µ ⊤ ∇ci (x t ) p (11
)
Var c i (x t ) + ∇c i (x t ) ⊤ p = σ 2 ci (x t ) + p ⊤ Σ ∇ci (x t ) p + 2p ⊤ Σ ci,∇ci (x t )(12)
where the last term accounts for the covariance between the function and its gradient. In the following, we drop the explicit evaluation at x t for all moments for notational convenience, i.e., µ ci = µ ci (x t ).
For a Gaussian random variable to remain non-negative with probability at least 1 -δ, we require its mean to exceed its standard deviation multiplied by the corresponding quantile. This yields:
µ ci + µ ⊤ ∇ci p ≥ q 1-δ σ 2 ci + p ⊤ Σ ∇ci p + 2p ⊤ Σ ci,∇ci(13)
where q 1-δ = Φ -1 (1 -δ) denotes the (1 -δ)-quantile of the standard normal distribution. Rearranging the terms and introducing an auxiliary variable t ci to upper-bound the square root term allows the constraint to be reformulated as a set of two inequalities:
-µ ⊤ ∇ci p + q 1-δ b ci ≤ µ ci and σ 2 ci + p ⊤ Σ ∇ci p + 2p ⊤ Σ ci,∇ci ≤ b ci(14)
To express the square root term more compactly, we consider the full covariance matrix associated with the joint Gaussian distribution of c i and its gradient ∇c i . Specifically, we can state
σ 2 ci + p ⊤ Σ ∇ci p + 2p ⊤ Σ ci,∇ci = 1 p ⊤ σ 2 ci • Σ ci,∇ci Σ ∇ci 1 p (15
)
By Cholesky decomposition of the covariance matrix, we can express the square root term as a second-order cone constraint:
σ 2 ci + p ⊤ Σ ∇ci p + 2p ⊤ Σ ci,∇ci = 1 p ⊤ L ci L ⊤ ci 1 p = L ⊤ ci 1 p 2 ≤ b ci (16
)
Using the same reasoning, we can reformulate the objective function by introducing the auxiliary variable b f . In the end, we obtain the following formulation that we refer to as B-SUB.
The uncertainty-aware subproblem of BayeSQP (B-SUB)
p t ∈ arg min p,b f ,{bc i } i∈Im 1 2 p ⊤ H t p + µ ⊤ ∇f p + µ f + q 1-δ f b f (17
) subject to L ⊤ f 1 p 2 ≤ b f , L ⊤ ci 1 p 2 ≤ b ci , ∀i ∈ I m , -µ ⊤ ∇ci p + q 1-δc b ci ≤ µ ci , ∀i ∈ I m .
where L f and L ci are Cholesky factorizations as
L f L ⊤ f = σ 2 f • Σ ∇f,f Σ ∇f , L ci L ⊤ ci = σ 2 ci • Σ ∇ci,ci Σ ∇ci , ∀i ∈ I m . (18
)
We omitted the explicit dependency on xt for clarity but all moments are evaluated at xt.
This formulation also naturally incorporates the subproblem formulation in (9).
this section cite: ['b8']

Section: Corollary 1 (Recovering the expected value formulation).
The solution for the search direction of B-SUB is equivalent to solution of (9) for δ f = 0.5 and δ c = 0.5. (Proof in Appendix C) Remark 1. In practice, the numerical solver will have an influence on the obtained results. So while the cones no longer restrict the search direction, a cone solver might still return a different solution.
this section cite: []

Section: Line search through constrained posterior sampling
With the search direction given as the solution of the B-SUB subproblem, the next step is to decide on a step size α which which we can update the current iterate as x t+1 = x t + α t p t . To implicitly decide on the step size, we perform constrained posterior sampling [14] on the one-dimensional line segment spanned by p t . Specifically, we aim to solve
arg min {xt+αpt | α∈[0,1]} f (x) subject to c i (x) ≥ 0, ∀i ∈ I m .(19)
This is similar to LineBO [34] but for an objective under potentially multiple constraints. However, in contrast to LineBO, our approach does not attempt global convergence along the line. Instead, we aim to select a sufficiently promising α t that yields progress given a limited evaluation budget M for the line search which we set to 3 in all experiments. Similar to [14], we either choose the next point to be the index of the best feasible point, or, if none of the points are feasible, as the point with the least amount of constraint violations as
x k+1 ←    arg min x (j) t ∈F f (x (j) t ), if F ̸ = ∅, arg min 1≤j≤M i∈Im max 0, -c i x (j) t , otherwise,(20)
where
F = x (j) t c i x (j) t
≥ 0, ∀i ∈ I m denotes the set of feasible points among the M samples.
this section cite: ['b13', 'b33', 'b13']

Section: Practical considerations and intuition on optimization behavior
Local sub-sampling Unlike GIBO-style methods [43,45,23], we decide against adaptive subsampling which would require optimizing over the uncertainty of the Hessian which is computationally very expensive. Instead, to approximate local curvature after each line search, we sample K points from a d-dimensional ball of radius ε centered at x t ∈ R d . For this, we first draw a Sobol sequence from the hypercube
[0, 1] d+1 . Each Sobol point ( x, u) ∈ [0, 1] d × [0, 1]
is then transformed such that x approximates a standard normal vector to yield a unit direction x, and u determines the individual radius as r = ε • u 1/d . The final sample is then x = x t + r • x.
this section cite: ['b42', 'b44', 'b22']

Section: Slack variable fallback strategy
The subproblem B-SUB may become infeasible due to constraint linearization or high uncertainty in gradient estimates. To address this, we implement a slack variable version of B-SUB as a fallback, which guarantees feasibility by design (cf. Appendix E). This approach aligns with established practices in classical SQP methods [46]. While the resulting search direction may not provide optimal robustness against uncertainty, the constrained posterior sampling along this direction will still seek to improve upon the current iterate.
this section cite: ['b45']

Section: Intuition on optimization behavior
To gain intuition about the parameters δ f and δ c and their influence on the optimization process, we study BayeSQP on a small toy example. We generate a two-dimensional within-model objective function (cf. Appendix A) with a quadratic constraint, resulting in only a small feasible region in the center. Figure 3 illustrates the optimization paths for different parameterizations. The initial step from the bottom left appears identical for all parameter settings. Subsequently, however, their behaviors differ significantly. In the expected value formulation (δ f , δ c = 0.5), the linearization of the quadratic constraint results in tangential directions p k , leading to limited or no improvement. We observe that incorporating uncertainty into the subproblem pushes the search direction toward the feasible set. Additionally, selecting a very low value for δ c effectively robustifies the constraints, as shown by the resulting directions p k .
this section cite: []

Section: Empirical evaluations
We next quantitatively evaluate our proposed method BayeSQP. Our evaluation first considers unconstrained and then constrained optimization problems using BoTorch [5]. We benchmark against four baselines: logarithmic EI (logEI) [1,32], TuRBO [15], SAASBO [13], and MPD [45]. These baselines are widely used [40,29,51,66] and represent complementary approaches-logEI employs a classic global optimization strategy, TuRBO implements a pseudo-local approach, SAASBO aims to automatically identify and exploit low-dimensional structure within high-dimensional search spaces through a hierarchical sparsity prior, and MPD is a fully local BO approach. Additionally, logEI and TuRBO can be readily adapted for constrained optimization through their respective variants: C-logEI [1,17,19] and SCBO [14] to which we compare on the constrained optimization problems.
In all subsequent plots, we present the median alongside the 5 th to 95 th percentile range (90% inner quantiles) computed across 32 independent random seeds. For BayeSQP, we set the hyperparameters δ f , δ c = 0.2 (unless stated otherwise) and K = d + 1, following Wu et al. [65,Corollary 1].
this section cite: ['b4', 'b0', 'b31', 'b14', 'b12', 'b44', 'b39', 'b28', 'b50', 'b65', 'b0', 'b16', 'b18', 'b13', 'b64']

Section: Unconstrained optimization
We first consider unconstrained within-model problems [26] for which we adapt B-SUB accordingly. We generate the functions using random Fourier features following [50,64] (cf. Appendix A for all details). Optimizing such functions has gained relevance with recent advances in latent space BO [60,22,40], where GP priors are enforced in the latent space [52]. Figure 4 summarizes the results. BayeSQP outperforms the other baselines from dimension 16 onward. Furthermore, we can observe the step-like behavior of BayeSQP resulting from the subsampling followed by solving B-SUB and the subsequent line search which yields the improvement.    Constrained optimization Similarly, we can perform within-model comparisons for the constrained case. Here, also the constraint function c(x) is a sample from an GP. Again, all details are provided in Appendix A. Figure 5 summarizes the constrained within-model results. As in the unconstrained case, BayeSQP outperforms the baselines at high dimensions (Figure 5a), while remaining orders of magnitude faster than SCBO and C-logEI despite computing full Hessians per B-SUB (Figure 5b). However, as we keep increasing dimensions, computing the Hessians of size d × d will results in a computational overhead. Here, low-rank approximations might be useful for balancing the trade-off between computational efficiency and required accuracy of the subproblem-it is likely that especially in the context of BO, the accuracy of the Hessian is not of utmost importance. For a detailed runtime breakdown and discussion we refer to Appendix F. Lastly, in Figure 5c we can observe the influence of the parameters δ f and δ c of B-SUB on the performance for d = 64. We can observe as visualized in Figure 3, not considering uncertainty especially in the constraints (δ c = 0.5) will result in suboptimal performance for such highly non-convex constraints. Including uncertainty results in a small buffer to the boundary, allowing the algorithm to escape local optima with a small region of attraction. The figure further highlights that beyond the decisive factor of taking uncertainty into account the overall sensitivity on how much uncertainty should be incorporated is rather low. The optimal values of these parameters may vary depending on the specific application.
Performance on standard benchmarks Lastly, we also evaluate BayeSQP on standard BO benchmarks. Here, we follow recent best practices and initialize lengthscales with √ d for all baselines [30,67,48]. The results are summarized in Table 1. We can clearly observe that BayeSQP is sensitive to initialization highlighted by the large 90% quantile especially for Ackley. This is to be expected as the algorithm is local and Ackley is very multi-modal. Still, importantly, BayeSQP is able to find feasible solutions for all seeds in all benchmarks contrary to C-logEI. To demonstrate the real-world applicability of BayeSQP, we compare constrained optimization baselines on the 7-dimensional Speed Reducer benchmark [36], which minimizes the weight of a speed reducer subject to 11 mechanical design non-linear constraints (more details in Appendix A.4).
The results are summarized in Table 2. All baselines are able to find feasible solutions for all seeds. C-logEI and BayeSQP show the best performance. In line with previous experiments, BayeSQP demonstrates a clear runtime advantage even in the presence of 11 constraints-each requiring separate Hessian evaluations-and a substantially larger B-SUB.
this section cite: ['b25', 'b49', 'b63', 'b59', 'b21', 'b39', 'b51', 'b29', 'b66', 'b47', 'b35']

Section: Discussion on limitations
While BayeSQP provides a novel framework combining classic optimization methods with BO, there are several limitations and addressing them will be interesting future research. Initialization matters As with any local approach, the initialization of BayeSQP will directly influence its performance (cf. Table 1). This further becomes clear when looking at the flow field of BayeSQP generated from 1000 different initial conditions on the Gramacy benchmark [20] in Figure 6 (details in Appendix A.5). Depending on the initialization, the algorithm converges to a different local optimum of the constrained problem.
Although global approaches can also exhibit sensitivity to initialization, this sensitivity is amplified in LBO approaches, particularly in constrained optimization. However, this sensitivity provides practitioners with the option to incorporate some expert knowledge into the optimization by choosing the initial guess; especially in engineering fields such as robotics, a feasible yet non-optimal solution is often known a-priori. An algorithm like BayeSQP will then become an automatic tool for fine-tuning.
this section cite: ['b19']

Section: Computational considerations
We show that for up to 96 dimensions even with the additional cost of computing the Hessian, BayeSQP demonstrates as very low total runtime. Still, at very large dimensions or high number of constraints, computing as well as storing the Hessian of all constraints will become problematic. In principle, one could also incorporate Hessian uncertainty into B-SUB, for example following efficient schemes such as [2,11]; whether this would lead to empirical performance improvements remains an open question. Future work could focus on evaluating the joint GP over only the most informative Hessian entries, adaptively selected during optimization, or on constructing the Lagrangian Hessian directly from gradient histories using a BFGS-type update scheme.
this section cite: ['b1', 'b10']

Section: Dependency on the kernel and model assumptions
The performance of BayeSQP strongly depends on the choice of kernel and, more generally, on the modeling assumptions underlying the GP surrogate. Since the construction of the B-SUB directly relies on the accuracy of both gradient and Hessian estimates, a poorly chosen kernel can lead to unreliable curvature information and ultimately to suboptimal search directions. While standard kernels such as the squared-exponential kernel perform well for smooth problems, they may struggle in settings with sharp nonlinearities or discontinuous constraints unless handled with additional care. Furthermore, kernel hyperparameters influence the scale and conditioning of the estimated Hessian, which can significantly affect the resulting search direction. Advances in GP modeling and training practices for BO (e.g., [30,67]) are expected to directly improve the robustness and effectiveness of BayeSQP.
In Appendix B, we list possible extensions of BayeSQP which in part address the limitations mentioned above as well as further interesting directions for future work.
this section cite: ['b29', 'b66']

Section: Conclusion
In this paper, we presented BayeSQP as a bridge between classic optimization methods and BO.
BayeSQP uses GP surrogates that jointly model the function, its gradient and its Hessian, which are then used to construct subproblems in an SQP-like fashion. Our results show that BayeSQP can outperform state-of-the-art methods in high-dimensional constrained optimization problems. We believe that BayeSQP provides a promising framework for integrating well-established classical optimization principles with modern black-box optimization techniques.
this section cite: []

Section: References
Ref_id:b0 Title: Unexpected improvements to expected improvement for Bayesian optimization Year: ()
Ref_id:b1 Title: Scalable first-order Bayesian optimization via structured automatic differentiation Year: ()
Ref_id:b2 Title: CVXOPT: a Python package for convex optimization Year: (2013)
Ref_id:b3 Title: ADMMBO: Bayesian optimization with unknown constraints using ADMM Year: (2019)
Ref_id:b4 Title: BoTorch: a framework for efficient Monte-Carlo Bayesian optimization Year: (2020)
Ref_id:b5 Title: Safe controller optimization for quadrotors with Gaussian processes Year: (2016)
Ref_id:b6 Title: Pyro: Deep universal probabilistic programming Year: (2019)
Ref_id:b7 Title: Bayesian optimization via continual variational last layer training Year: ()
Ref_id:b8 Title: Bayesian optimization for learning gaits under uncertainty: An experimental comparison on a dynamic bipedal walker Year: (2016)
Ref_id:b9 Title: Bayesian optimization in drug discovery. High Performance Computing for Drug Discovery and Year: (2023)
Ref_id:b10 Title: High-dimensional Gaussian process inference with derivatives Year: ()
Ref_id:b11 Title: Neural architecture search: A survey Year: (2019)
Ref_id:b12 Title: High-dimensional Bayesian optimization with sparse axis-aligned subspaces Year: ()
Ref_id:b13 Title: Scalable constrained Bayesian optimization Year: ()
Ref_id:b14 Title: Scalable global optimization via local Bayesian optimization Year: (2019)
Ref_id:b15 Title: Minimizing UCB: a better local search strategy in local Bayesian optimization Year: ()
Ref_id:b16 Title: Bayesian optimization with inequality constraints Year: (2014)
Ref_id:b17 Title: Bayesian optimization Year: (2023)
Ref_id:b18 Title: Bayesian optimization with unknown constraints Year: (2014)
Ref_id:b19 Title: Modeling an augmented lagrangian for blackbox constrained optimization Year: (2016)
Ref_id:b20 Title: Constrained Bayesian optimization for automatic chemical design using variational autoencoders Year: (2020)
Ref_id:b21 Title: Highdimensional Bayesian optimisation with variational autoencoders and deep metric learning Year: (2021)
Ref_id:b22 Title: Simulation-aided policy tuning for black-box robot learning Year: ()
Ref_id:b23 Title: Fast probabilistic optimization from noisy gradients Year: (2013)
Ref_id:b24 Title: Quasi-Newton methods: A new direction Year: (2013)
Ref_id:b25 Title: Entropy search for information-efficient global optimization Year: (2012)
Ref_id:b26 Title: Probabilistic numerics and uncertainty in computations Year: (2015)
Ref_id:b27 Title: Probabilistic Numerics: Computation as Machine Learning Year: (2022)
Ref_id:b28 Title: Fine-tuning of neural network approximate MPC without retraining via Bayesian optimization Year: (2024)
Ref_id:b29 Title: Vanilla Bayesian optimization performs great in high dimensions Year: ()
Ref_id:b30 Title: A stopping criterion for Bayesian optimization by the gap of expected minimum simple regrets Year: ()
Ref_id:b31 Title: Efficient global optimization of expensive black-box functions Year: (1998)
Ref_id:b32 Title: High dimensional Bayesian optimisation and bandits via additive models Year: (2015)
Ref_id:b33 Title: Adaptive and safe Bayesian optimization in high dimensions via one-dimensional subspaces Year: (2019)
Ref_id:b34 Title: Promises and pitfalls of the linearized laplace in bayesian optimization Year: ()
Ref_id:b35 Title: Constrained optimization problems in mechanical engineering design using a real-coded steady-state genetic algorithm Year: (2010)
Ref_id:b36 Title: Constrained Bayesian optimization with noisy experiments Year: (2019)
Ref_id:b37 Title: Probabilistic line searches for stochastic optimization Year: (2017)
Ref_id:b38 Title: Automatic termination for hyperparameter optimization Year: (2022)
Ref_id:b39 Title: Local latent space Bayesian optimization over structured inputs Year: ()
Ref_id:b40 Title: Approximation-aware Bayesian optimization Year: ()
Ref_id:b41 Title: Sparse Gaussian processes for Bayesian optimization Year: (2016)
Ref_id:b42 Title: Local policy search with Bayesian optimization Year: ()
Ref_id:b43 Title: Dataefficient autotuning with Bayesian optimization: An industrial control study Year: (2019)
Ref_id:b44 Title: Local Bayesian optimization via maximizing probability of descent Year: (2022)
Ref_id:b45 Title: Numerical optimization Year: (2006)
Ref_id:b46 Title: Increasing the scope as you learn: Adaptive Bayesian optimization in nested subspaces Year: ()
Ref_id:b47 Title: Understanding high-dimensional Bayesian optimization Year: (2025)
Ref_id:b48 Title: Bayesian optimization under mixed constraints with a slack-variable augmented Lagrangian Year: (2016)
Ref_id:b49 Title: Random features for large-scale kernel machines Year: (2007)
Ref_id:b50 Title: The extremely brilliant source storage ring of the european synchrotron radiation facility Year: (2023)
Ref_id:b51 Title: High-dimensional bayesian optimisation with gaussian process prior variational autoencoders Year: ()
Ref_id:b52 Title: Gaussian processes for machine learning Year: (2006)
Ref_id:b53 Title: Combining radial basis function surrogates and dynamic coordinate search in high-dimensional expensive black-box optimization Year: (2013)
Ref_id:b54 Title: Practical Bayesian optimization of machine learning algorithms Year: (2012)
Ref_id:b55 Title: Scalable Bayesian optimization using deep neural networks Year: (2015)
Ref_id:b56 Title: Stochastic gradient line Bayesian optimization for efficient noise-robust optimization of parameterized quantum circuits Year: (2022)
Ref_id:b57 Title: CAGES: Cost-aware gradient entropy search for efficient local multi-fidelity Bayesian optimization Year: (2024)
Ref_id:b58 Title: NeST-BO: Fast local Bayesian optimization via Newton-step targeting of gradient and Hessian information Year: (2025)
Ref_id:b59 Title: Sample-efficient optimization in the latent space of deep generative models via weighted retraining Year: (2020)
Ref_id:b60 Title: Local Bayesian optimization for controller tuning with crash constraints Year: (2024)
Ref_id:b61 Title: Bayesian optimization in a billion dimensions via random embeddings Year: (2016)
Ref_id:b62 Title: Stopping Bayesian optimization with probabilistic regret bounds Year: (2024)
Ref_id:b63 Title: Pathwise conditioning of Gaussian processes Year: ()
Ref_id:b64 Title: The behavior and convergence of local Bayesian optimization Year: ()
Ref_id:b65 Title: ADNNet: Attention-based deep neural network for air quality index prediction Year: (2024)
Ref_id:b66 Title: Standard Gaussian process is all you need for high-dimensional Bayesian optimization Year: ()
