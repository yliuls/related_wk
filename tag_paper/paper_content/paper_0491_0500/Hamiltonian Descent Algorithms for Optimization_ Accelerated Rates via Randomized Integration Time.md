Title: Hamiltonian Descent Algorithms for Optimization: Accelerated Rates via Randomized Integration Time
Abstract: We study the Hamiltonian flow for optimization (HF-opt), which simulates the Hamiltonian dynamics for some integration time and resets the velocity to 0 to decrease the objective function; this is the optimization analogue of the Hamiltonian Monte Carlo algorithm for sampling. For short integration time, HF-opt has the same convergence rates as gradient descent for minimizing strongly and weakly convex functions. We show that by randomizing the integration time in HF-opt, the resulting randomized Hamiltonian flow (RHF) achieves accelerated convergence rates in continuous time, similar to the rates for accelerated gradient flow. We study a discrete-time implementation of RHF as the randomized Hamiltonian gradient descent (RHGD) algorithm. We prove that RHGD achieves the same accelerated convergence rates as Nesterov's accelerated gradient descent (AGD) for minimizing smooth strongly and weakly convex functions. We provide numerical experiments to demonstrate that RHGD is competitive with classical accelerated methods such as AGD across all settings and outperforms them in certain regimes.

Section: Introduction
Optimization plays a central role in machine learning, with algorithms such as gradient descent (GD) and accelerated gradient descent (AGD) [Nesterov, 1983] serving as essential tools for optimizing objective functions. A growing body of work has explored optimization algorithms in the framework of continuous-time dynamical systems, which provides insights into algorithmic behaviors and convergence properties. In this paper, we develop a novel family of accelerated optimization algorithms that are designed based on the Hamiltonian flow.
Hamiltonian flow (HF) originates from classical mechanics, describing the continuous-time evolution of physical systems. At a fundamental level, Hamiltonian flow governs how the positions and momenta of moving bodies evolve while conserving energy. Beyond its roots in physics, Hamiltonian flows have inspired computational algorithms such as Hamiltonian Monte Carlo (HMC) [Duane et al., 1987], a classical method widely employed for sampling from complex, high-dimensional probability distributions [Neal et al., 2011, Betancourt, 2017, Hoffman et al., 2014]. Due to its effectiveness, HMC has found extensive applications in Bayesian inference, statistical physics, and machine learning [Gelman et al., 1995, Kruschke, 2014, Lelievre and Stoltz, 2016].
There has been growing interest in exploring the connections between optimization and sampling, as they share deep theoretical foundations and many algorithmic similarities. Notably, the Langevin dynamics for sampling can be viewed as the gradient flow for minimizing the relative entropy or Kullback-Leibler (KL) divergence in the space of probability distributions [Jordan et al., 1998]. Many works build on this perspective to use techniques from optimization to analyze Langevin-based algorithms [Wibisono, 2018, Bernton, 2018, Durmus et al., 2019, Ma et al., 2021] or develop novel sampling algorithms inspired by optimization [Salim et al., 2020, Lee et al., 2021, Lambert et al., 2022, Chen et al., 2022, Diao et al., 2023, Das and Nagaraj, 2023, Suzuki et al., 2023, Fu and Wilson, 2024, Chen et al., 2025]. In this paper, we strengthen the links between optimization and sampling in the opposite direction, by studying how to translate the Hamiltonian Monte Carlo (HMC), a classical sampling algorithm, to design new optimization algorithms, particularly for accelerated methods.
In the links between optimization and sampling, there is a significant theoretical gap regarding acceleration. In optimization, it is well known that greedy methods such as GD can be outperformed by accelerated methods such as AGD [Nesterov, 1983] which have faster and optimal convergence rates with square-root dependence on the condition number for minimizing smooth and strongly convex functions under the standard first-order oracle model; see Appendix C.1 for a review. A similar acceleration phenomenon in sampling is still elusive, but there are some promising candidates. The underdamped (or kinetic) Langevin dynamics has an accelerated convergence rate in continuous time [Cao et al., 2023], but in discrete time, algorithms based on its discretization still do not have the desired accelerated rates [Ma et al., 2021, Zhang et al., 2023]. Another candidate is the randomized Hamiltonian Monte Carlo (RHMC) [Bou-Rabee and Sanz-Serna, 2017], which is obtained by randomizing the integration time in HMC, and has been conjectured to have an accelerated convergence rate for sampling [Jiang, 2023]. In continuous time, the idealized RHMC indeed has an accelerated convergence rate in χ 2 -divergence for log-concave target distributions [Lu and Wang, 2022]. On the algorithmic side, recent works have shown that for a Gaussian target distribution, HMC with carefully chosen integration time, either determined by the roots of Chebyshev polynomials or randomly drawn from exponential distributions, indeed achieves an accelerated mixing time with square-root dependence on the condition number [Wang and Wibisono, 2023a, Jiang, 2023, Apers et al., 2024]. However, the proof that RHMC achieves acceleration in discrete time for a general target distribution remains missing. In this work, we study the optimization analogue of this question, by designing a new accelerated optimization algorithm based on the randomized Hamiltonian flow.
While Hamiltonian flows have found great success in sampling, their direct use in designing optimization algorithms is still relatively limited. Most existing Hamiltonian-based optimization methods can be seen as the discretization of the accelerated gradient flow (AGF) [Su et al., 2016, Wibisono et al., 2016], which is the combination of HF with a damping term for dissipating energy; this is different from the HF we study in this work which does not have damping. We provide additional discussion of related work in Appendix A. Pure HF without damping terms have rarely been studied for optimization. In fact, it obeys the law of energy conservation rather than dissipation, which is opposite to optimization tasks. Notable prior works include Teel et al. [2019], Diakonikolas and Jordan [2021], De Luca et al. [2023], Wang [2024]. Particularly, Wang [2024] show that Hamiltonian flows with velocity refreshment and Chebyshev-based integration times can achieve accelerated convergence for strongly convex quadratic functions, which is comparable to AGF with refreshment. Beyond quadratic functions, we demonstrate that HF with short-time integration and periodic velocity refreshment achieves the same non-accelerated convergence rates as GD up to constants (see Theorem 1). This naturally prompts a question: Can we develop accelerated optimization methods based on the Hamiltonian flow?
In this work, we answer this question affirmatively and demonstrate that HF with randomized integration time and its discretization yield accelerated convergence rates for minimizing strongly and weakly convex functions. Our principal contributions are:
• We propose the randomized Hamiltonian flow (RHF) for optimization as an analogue of RHMC. We establish its accelerated convergence rates of O(exp(-α/5 t)) for α-strongly convex functions and O(1/t 2 ) for weakly convex functions. These rates match the optimal accelerated convergence rates of AGF [Su et al., 2016, Wibisono et al., 2016] up to constants.
• We study the randomized Hamiltonian gradient descent (RHGD) which is a discretization of RHF.
Under L-smoothness assumption, RHGD achieves the overall iteration complexity of Õ( L/α) for α-strongly convex functions and O( L/ε) for weakly convex functions to generate an εaccurate solution in expectation, matching the optimal accelerated rates of AGD [Nesterov, 1983[Nesterov, , 2018] ] and its randomized variant [Even et al., 2021].
Organization The remainder of this work is organized as follows. Section 2 presents notations, definitions and a review of the Hamiltonian flow (HF) for designing optimization algorithms. Section 3 proposes the definition of the randomized Hamiltonian flow (RHF) and its accelerated convergence rates. Section 4 describes how to discretize continuous-time RHF into an implementable optimization algorithm RHGD and establishes its accelerated convergence rates. Section 5 presents numerical experiments validating the effectiveness of our proposed methods. Section 6 concludes the paper and discusses its limitations.
2 Preliminaries and reviews
this section cite: ['b76', 'b32', 'b75', 'b4', 'b47', 'b40', 'b53', 'b60', 'b50', 'b102', 'b2', 'b35', 'b62', 'b86', 'b57', 'b54', 'b19', 'b30', 'b27', 'b93', 'b38', 'b17', 'b76', 'b15', 'b62', 'b9', 'b49', 'b61', 'b49', 'b0', 'b91', 'b95', 'b29', 'b98', 'b98', 'b91', 'b76', 'b77', 'b37']

Section: Notations and definitions
Let ∥•∥ := ∥•∥ 2 denote the Euclidean norm on the d-dimensional Euclidean space R d . A differentiable function f : R d → R is α-strongly convex if f (y) ≥ f (x) + ⟨∇f (x), y -x⟩ + α 2 ∥y -x∥ 2 for any x, y ∈ R d , where f is (weakly) convex if α = 0. We say f is α-gradient-dominated if ∥∇f (x)∥ 2 ≥ 2α(f (x) -f (x * )) for any x ∈ R d where x * = arg min x∈R d f (x) is a minimizer of f . We say f is L-smooth if f (y) ≤ f (x) + ⟨∇f (x), y -x⟩ + L 2 ∥y -x∥ 2 or equivalently ∥∇f (x) -∇f (y)∥ ≤ L∥x -y∥ for any x, y ∈ R d . We assume α ≤ 1 ≤ L. We say x is an ε-accurate solution in expectation if E[f (x) -f (x * )] ≤ ε. Let κ := L/α denote the condition number. We use ġt := d dt g t to denote the time derivative of a time-dependent quantity g t . We define the flow map
HF η : R d × R d → R d × R d as HF η (X 0 , Y 0 ) = (X η , Y η )
, which is the solution to Hamiltonian flow (HF) at time η starting from (X 0 , Y 0 ). Given a time-dependent random variable Z t , we use ρ Z t to denote its probability distribution. We identify probability distributions with their density functions. Let Exp(γ) denote the exponential distribution with mean 1/γ for γ > 0. Let [n] := {1, 2, ..., n}. We use a = O(b) to denote a ≤ Cb for constant C > 0 and use a = Õ(b) to denote a = O(b) up to logarithmic factors. We use a = Θ(b) to denote a = Cb for constants C > 0.
Problem setting. Our goal is to solve the following optimization problem:
min x∈R d f (x),(1)
where f : R d → R is a differentiable function, and x * = arg min x∈R d f (x) is a minimizer of f .
this section cite: []

Section: Hamiltonian flow for optimization
The Hamiltonian flow (HF) is a system of ordinary differential equations for (X t , Y t ) ∈ R d × R d :
Ẋt = Y t , Ẏt = -∇f (X t ).(HF)
Define the energy (or Hamiltonian) function H(x, y) := f (x) + 1 2 ∥y∥ 2 . A fundamental property of the Hamiltonian flow (HF) is that it conserves energy; see Appendix B.1 for the proof. Lemma 1 (Energy Conservation). Along (HF), H(X t , Y t ) = H(X 0 , Y 0 ) for all t ≥ 0.
We can exploit the conservation property of the Hamiltonian flow to design an optimization algorithm by periodically refreshing the velocity to 0. This idea results in the following Hamiltonian flow for optimization (HF-opt) algorithm, which was also proposed and studied by Teel et al. [2019], Wang [2024]. Below, Π 1 (x, y) = x is the projection operator to the first component.
Algorithm 1 Hamiltonian Flow for Optimization (HF-opt) 1: Initialize x 0 ∈ R d . Choose integration time η k > 0 for k ≥ 0. 2: for k = 0, 1, . . . , K -1 do 3:
x k+1 = Π 1 • HF η k (x k , 0)
▷ (evolve (HF) for time η k and project to first component) 4: end for 5: return x K Lemma 1 implies the following descent lemma of HF-opt; see Appendix B.1 for the proof. Lemma 2. For any k and η k > 0, HF-opt (Algorithm 1) satisfies f (x k+1 ) ≤ f (x k ).
HF-opt is an instance of a new optimization principle, the "Lift-Conserve-Project" (LCP) scheme, which is the same principle that underlies HMC for sampling; see Appendix B for more details.
HF-opt is an idealized algorithm since it assumes we can solve the Hamiltonian flow (HF) exactly. We study its convergence properties in this and the next sections, and we study how to implement it as a concrete discrete-time algorithm in Section 4. When f is smooth, we show that HF-opt with short integration time η k has the following convergence rates under gradient domination and weak convexity in Theorem 1. Note that the first conclusion in Theorem 1 also holds under α-strong convexity since it implies α-gradient domination. We provide the proof in Appendix D.1.
Theorem 1. Assume f is L-smooth. Along Algorithm 1 with η k = h ≤ 1 √ L , from any x 0 ∈ R d : 1. If f is α-gradient-dominated, then f (x k ) -f (x * ) ≤ 1 -1 2 αh 2 k (f (x 0 ) -f (x * )). 2. If f is weakly convex, then f (x k ) -f (x * ) ≤ 34∥x 0 -x * ∥ 2 h 2 k .
Up to constants, the results in Theorems 1 match the convergence rates of GD under the same assumptions (see Theorem 8 in Appendix C.1.1 and Theorem 12 in Appendix C.1.2) and are derived based on the exact simulation of (HF). If we replace HF η k with a one-step leapfrog integrator [Sanz-Serna, 1992] for implementation, then HF-opt recovers exactly the GD algorithm (see Appendix D.2), and thus inherits the same convergence guarantees as GD.
In this paper, we aim to achieve accelerated convergence rates analogous to Nesterov's accelerated gradient descent (AGD). Wang [2024] show that HF-opt achieves the accelerated convergence rate for minimizing strongly convex quadratic functions when the integration time η k is selected based on the roots of Chebyshev polynomials. Inspired by the randomized Hamiltonian Monte Carlo (RHMC) algorithm for sampling [Bou-Rabee and Sanz-Serna, 2017] where the integration time is independently drawn from an exponential distribution, we study its optimization counterpart to explore accelerated convergence rates for a broader class of objectives beyond quadratic functions.
this section cite: ['b95', 'b98', 'b88', 'b98', 'b9']

Section: Randomized Hamiltonian flow for optimization
We propose a new optimization counterpart of RHMC, called the randomized Hamiltonian flow for optimization (RHF-opt), where the integration time is drawn from an exponential distribution.
Algorithm 2 Randomized Hamiltonian Flow for Optimization (RHF-opt)
1: Initialize x 0 ∈ R d . Specify γ(t) > 0 for all t ≥ 0. 2: for k = 0, 1, . . . , K -1 do 3: Set the current time T k = k-1 i=0 τ i (set T 0 = 0) 4: Independently sample τ k ∼ Exp (γ(T k )) 5: Set x k+1 = Π 1 • HF τ k (x k ,0
) ▷ (evolve (HF) for time τ k and project to first component) 6: end for 7: return x K In Algorithm 2, the k-th integration time τ k is a random variable drawn from an exponential distribution with mean 1/γ(T k ), where T k is the current time. Note that γ(t) can depend on time. Below, we choose γ(t) to be a constant when f is strongly convex, and γ(t) ∝ 1/t when f is weakly convex.
this section cite: []

Section: Reformulation of RHF-opt as a continuous-time process
To rigorously state convergence rates of Algorithm 2 (RHF-opt), we first describe an equivalent formulation of RHF-opt as the following piecewise deterministic continuous-time process that we refer to as the randomized Hamiltonian flow (RHF):
1. Evolve (HF) between velocity refreshment events. 2. At random jump times governed by an inhomogeneous Poisson process with rate γ(t), we refresh the velocity to 0, and continue evolving (HF).
In the continuous-time perspective, t ≥ 0 denotes the actual time variable. The sequence {T k } k≥0 in Algorithm 2 (RHF-opt) represents the random refreshment times generated by cumulative sums of independent exponential random variables with rates γ(T k ). Equivalently, the continuous-time process described above can be modeled as the following stochastic process:
dX t = Y t dt, dY t = -∇f (X t ) dt -Y t dN t ,(RHF)
where dN t := k≥1 δ T k (dt) is the Poisson point process with rate γ(t), and T k is the k-th time an event happens. Let Y t -be the left limit of Y t . At each random time T k , the second line in (RHF)
updates Y T k -Y T - k = -Y T - k
, which refreshes the velocity to Y T k = 0. See also Even et al. [2021, Appendix C] for a review of the Poisson point measure and the left limit update described above.
this section cite: []

Section: Accelerated convergence rates of the randomized Hamiltonian flow
We establish the accelerated convergence rates of (RHF) for minimizing strongly and weakly convex functions. Our proofs use the continuity equation along (RHF); see Lemma 13 in Appendix F.
this section cite: []

Section: For strongly convex functions
We show the following convergence rate of (RHF) under strong convexity; see Appendix F.1 for the proof. In the result below, the expectation is over the randomness in (X t , Y t ) ∈ R 2d , which comes from the random integration times in (RHF).
Theorem 2. Assume f is α-strongly convex. Let (X t , Y t ) evolve following (RHF) with the choice γ(t) = 16α
5 , from any X 0 ∈ R d with Y 0 = 0. Then for any t ≥ 0, we have
E [f (X t ) -f (x * )] ≤ exp - α 5 t E f (X 0 ) -f (x * ) + α 10 ∥X 0 -x * ∥ 2 .
Compared with HF-opt with short integration time (Theorem 1), RHF achieves faster convergence for strongly convex functions without smoothness assumption. Recall that the convergence rates for minimizing α-strongly convex functions are O(exp(-2αt)) for the gradient flow (GF) and O(exp(-√ αt)) for the accelerated gradient flow (AGF) [Wibisono et al., 2016] (see Theorems 6 and 7 in Appendix C.1.1). In comparison, RHF achieves a faster convergence rate than GF when α is small, and it matches the accelerated rate of AGF up to constants, albeit in expectation.
this section cite: []

Section: For weakly convex functions
We show the convergence rate of (RHF) under weak convexity; see Appendix F.2 for the proof.
Theorem 3. Assume f is weakly convex. Let (X t , Y t ) evolve following (RHF) with the choice γ(t) = 6 t+1 , from any X 0 ∈ R d with Y 0 = 0. Then for any t ≥ 0, we have
E [f (X t ) -f (x * )] ≤ 5 • E f (X 0 ) -f (x * ) + ∥X 0 -x * ∥ 2 (t + 1) 2 .
Compared with HF-opt with short integration time (Theorem 1), RHF achieves faster convergence for weakly convex functions without smoothness assumption. Recall the convergence rates for minimixing weakly convex functions are O(1/t) for GF and O(1/t 2 ) for AGF [Su et al., 2016, Wibisono et al., 2016] (see Theorems 10 and 11 in Appendix C.1.2). In this case as well, RHF improves upon the rate of GF and matches the accelerated rate of AGF, albeit in expectation.
The convergence guarantees in Theorems 2 and 3 are still idealized because they assume we can exactly simulate Hamiltonian flow (HF). In Section 4, we discuss a practical implementation of RHF.
this section cite: ['b91']

Section: Randomized Hamiltonian gradient descent
We study the discretization and implementation of the randomized Hamiltonian flow (RHF) as a discrete-time algorithm. We consider two sources of approximation in the discretization process.
this section cite: []

Section: Approximate Poisson process.
In RHF, velocity is refreshed at random times governed by a Poisson process with rate γ(t). For a small time increment h > 0, the probability of a refreshment event occurring in [t, t + h) is approximately γ(t) • h, with the probability of multiple events occurring in the same interval being negligible (order o(h)). Thus, given x 0 ∈ R d and y 0 = 0, RHF can be approximated by alternating between a deterministic integration step of (HF) over time h to generate a proposal and a probabilistic accept-refresh step for k ≥ 0:
1. Generate proposal: (x k+1 , ỹk+1 ) = HF h (x k , y k )
2. Accept-refresh: y k+1 = ỹk+1 with probability 1 -min(γ(kh) • h, 1) 0 with probability min(γ(kh) • h, 1)
As h → 0, the process above recovers RHF.
this section cite: []

Section: Approximate Hamiltonian flow.
In practice, we need to simulate the Hamiltonian flow (HF) using a numerical integrator, such as the leapfrog integrator [Leimkuhler and Reich, 2004, Sanz-Serna, 1992, Bou-Rabee and Sanz-Serna, 2018]. Accordingly, we replace the exact flow map HF h with a discrete-time integrator
T h : R d × R d → R d × R d
given stepsize h. As a first step, we consider T h to be the implicit (backward Euler) integrator. The update for (x k+1 , ỹk+1 ) = T h (x k , y k ) satisfies the following system of implicit equations:
x k+1 -x k = hỹ k+1 , (2a) ỹk+1 -y k = -h∇f (x k+1 ).(2b)
By substituting ỹk+1 in (2a) with y k -h∇f (x k+1 ) from ( 2b), updates (2) can be reformulated as
x k+1 = Prox h 2 f (x k + hy k ),(3a)
ỹk+1 = y k -h∇f (x k+1 ).(3b)
where
Prox h 2 f (x) = arg min y∈R d f (y) + 1 2h 2 ∥y -x∥ 2 is the proximal operator.
If we can implement the proximal operator for f , then the updates (3) above yield a concrete algorithm that we call the randomized proximal Hamiltonian descent (RPHD); see Appendix G.1 for more details on RPHD and its convergence analysis. However, the proximal step (3a) is not explicit for general f , and thus we make one further approximation to turn it into a concrete algorithm.
Algorithm. Let x k+ 1 2 := x k + hy k . We approximate the proximal step (3a) by gradient descent:
x k+1 = x k+ 1 2 -h 2 ∇f (x k+ 1 2 ).(4)
This modification leads to a practical algorithm that we call the randomized Hamiltonian gradient descent (RHGD), summarized in Algorithm 3. Note that as h → 0, RHGD recovers RHF.
Algorithm 3 Randomized Hamiltonian Gradient Descent (RHGD)
1: Initialize x 0 ∈ R d and y 0 = 0. Choose stepsize h > 0 and refreshment rate γ k > 0.
2: for k = 0, 1, . . . , K -1 do 3:
x k+ 1 2 = x k + hy k 4:
x k+1 = x k+ 1 2 -h 2 ∇f (x k+ 1 2 ) 5: ỹk+1 = y k -h∇f (x k+1 ) 6:
y k+1 = ỹk+1 with probability 1 -min (γ k • h, 1) 0 with probability min (γ k • h, 1) 7: end for 8: return x K 4.1 Accelerated convergence rates of RHGD RHGD serves as a practical implementation of the randomized Hamiltonian flow (RHF). In the following, we analyze the convergence rates of RHGD under both strong and weak convexity.
this section cite: ['b58', 'b88', 'b9']

Section: For strongly convex functions
We show the following accelerated convergence rate of RHGD for minimizing smooth and strongly convex functions. The proof is deferred to Appendix G.3.1.
Theorem 4. Assume f is α-strongly convex and L-smooth. Then for all k ≥ 0, RHGD (Algorithm 3) with h ≤ 1
4 √ L , γ k = √
α, and from any
x 0 ∈ R d satisfies E[f (x k ) -f (x * )] ≤ 1 + √ αh 6 -k E f (x 0 ) -f (x * ) + α 72 ∥x 0 -x * ∥ 2 . Corollary 1. Assume f is α-strongly convex and L-smooth. To generate x K satisfying E[f (x K ) - f (x * )] ≤ ε, it suffices to run Algorithm 3 with h = 1 4 √ L , γ k = √
α, and from any
x 0 ∈ R d for K ≥ (24 √ κ + 1) • log E f (x 0 ) -f (x * ) + α 72 ∥x 0 -x * ∥ 2 ε .
Corollary 1 shows that RHGD requires O( √ κ log(1/ε)) iterations to generate an ε-accurate solution in expectation under smoothness and strong convexity. Recall under the same assumptions, GD achieves the iteration complexity of O(κ log(1/ε)), whereas AGD achieves the improved iteration complexity of O( √ κ log(1/ε)) (see Corollaries 3 and 4 in Appendix C.1.1). In comparison, RHGD is faster than GD, and matches the accelerated rate of AGD, albeit in expectation.
this section cite: []

Section: For weakly convex functions
We show the following convergence rate of RHGD for minimizing smooth and weakly convex functions. The proof is deferred to Appendix G.3.3. Theorem 5. Assume f is weakly convex and L-smooth. Then for all k ≥ 0, RHGD (Algorithm 3) with h ≤ 1 7 √ L , γ k = 17 2(k+9)h , and from any
x 0 ∈ R d satisfies E[f (x k ) -f (x * )] ≤ 14 • E ∥x 0 -x * ∥ 2 h 2 (k + 8) 2 .
Corollary 2. Assume f is weakly convex and L-smooth. To generate
x K satisfying E[f (x K ) - f (x * )] ≤ ε, it suffices to run Algorithm 3 with h = 1 7 √ L , γ k = 17 2(k+9)h and any x 0 ∈ R d for K ≥ 686L • E [∥x 0 -x * ∥ 2 ] ε .
Corollary 2 shows that RHGD requires O( L/ε) iterations to generate an ε-accurate solution in expectation under L-smoothness and weak convexity. Recall under the same assumptions, GD achieves the iteration complexity of O(L/ε), whereas AGD achieves the improved iteration complexity of O( L/ε) (see Corollaries 5 and 6 in Appendix C.1.2). In comparison, RHGD is faster than GD, and matches the accelerated rate of AGD, albeit in expectation.
this section cite: []

Section: Discussion
Unlike the convergence rates of GD and AGD, which hold deterministically for f (x k ) -f (x * ), the convergence rate of RHGD holds in expectation, i.e., E[f (x k ) -f (x * )] due to the random refreshment. Nevertheless, convergence in expectation can still imply high-probability bounds for f (x k )-f (x * ) via Markov's inequality. We also remark that the continuized version of AGD (CAGD) proposed by Even et al. [2021] and studied by Wang and Wibisono [2023b], where the two variables continuously mix following a linear ordinary differential equation and take gradient steps at random times, similarly achieves an accelerated convergence rate in expectation.
Proof Sketch. We first establish the convergence of the ideal algorithm RPHD (Algorithm 5). Using a Lyapunov function E k , we show that it preserves the accelerated convergence via E k+1 ≤ E k (see Theorems 14 and 15). The analysis for the practical algorithm RHGD follows similarly but accounts for the approximation of the proximal step (3a) using gradient descent (4). We bound the resulting error, which depends on the gradient norm (see Proposition 1), and then incorporate it into the Lyapunov decrease. Unlike prior works (e.g., [Wilson et al., 2021]), our analysis avoids explicitly tracking intermediate iterates, enabling flexibility in the choice of approximation for (3a).
this section cite: ['b37']

Section: References
Ref_id:b0 Title: Hamiltonian Monte Carlo for efficient Gaussian sampling: long and random steps Year: (2024)
Ref_id:b1 Title: First-order optimization algorithms via inertial systems with Hessian driven damping Year: (2022)
Ref_id:b2 Title: Langevin Monte Carlo and JKO splitting Year: (2018-07)
Ref_id:b3 Title: Optimal tuning of the hybrid Monte Carlo algorithm Year: (2013)
Ref_id:b4 Title: A conceptual introduction to Hamiltonian Monte Carlo Year: (2017)
Ref_id:b5 Title: Optimizing the integrator step size for Hamiltonian Monte Carlo Year: (2014)
Ref_id:b6 Title: Two-scale coupling for preconditioned Hamiltonian Monte Carlo in infinite dimensions Year: (2021)
Ref_id:b7 Title: Mixing time guarantees for unadjusted Hamiltonian Monte Carlo Year: (2023)
Ref_id:b8 Title: Unadjusted Hamiltonian MCMC with stratified Monte Carlo time integration Year: (2025)
Ref_id:b9 Title: Nawaf Bou-Rabee and Jesús Maria Sanz-Serna. Geometric integrators and the Hamiltonian Monte Carlo method Year: (2017)
Ref_id:b10 Title: Nonlinear Hamiltonian Monte Carlo & its particle approximation Year: (2023)
Ref_id:b11 Title: Coupling and convergence for Hamiltonian Monte Carlo Year: (2020)
Ref_id:b12 Title: Evaluating the implicit midpoint integrator for Riemannian Hamiltonian Monte Carlo Year: (2021)
Ref_id:b13 Title: Second order quantitative bounds for unadjusted generalized Hamiltonian Monte Carlo Year: (2023)
Ref_id:b14 Title: Complexity of randomized algorithms for underdamped Langevin dynamics Year: (2020)
Ref_id:b15 Title: On explicit L 2 -convergence rate estimate for underdamped Langevin dynamics Year: (2023)
Ref_id:b16 Title: Stan: A probabilistic programming language Year: (2017)
Ref_id:b17 Title: Accelerating optimization over the space of probability measures Year: (2025)
Ref_id:b18 Title: Stochastic gradient Hamiltonian Monte Carlo Year: (2014)
Ref_id:b19 Title: Improved analysis for a proximal algorithm for sampling Year: (2022)
Ref_id:b20 Title: When does Metropolized Hamiltonian Monte Carlo provably outperform Metropolis-adjusted Langevin algorithm Year: (2023)
Ref_id:b21 Title: Fast mixing of Metropolized Hamiltonian Monte Carlo: Benefits of multi-step gradients Year: (2020)
Ref_id:b22 Title: Convergence of Langevin MCMC in KL-divergence Year: (2018)
Ref_id:b23 Title: Underdamped Langevin MCMC: A non-asymptotic analysis Year: (2018)
Ref_id:b24 Title: Analysis of Langevin Monte Carlo from Poincare to log-Sobolev Year: (2024)
Ref_id:b25 Title: Further and stronger analogy between sampling and optimization: Langevin Monte Carlo and gradient descent Year: (2017)
Ref_id:b26 Title: Theoretical guarantees for approximate sampling from smooth and log-concave densities Year: (2017)
Ref_id:b27 Title: Provably fast finite particle variants of SVGD via virtual particle stochastic approximation Year: (2023)
Ref_id:b28 Title: Improving energy conserving descent for machine learning: Theory and practice Year: (2023)
Ref_id:b29 Title: Generalized momentum-based methods: A Hamiltonian perspective Year: (2021)
Ref_id:b30 Title: Forward-backward Gaussian variational inference via JKO in the Bures-Wasserstein space Year: (2023)
Ref_id:b31 Title: Probabilistic path Hamiltonian Monte Carlo Year: (2017)
Ref_id:b32 Title:  Year: (1987)
Ref_id:b33 Title: Nonasymptotic convergence analysis for the unadjusted langevin algorithm Year: (2017)
Ref_id:b34 Title: On the convergence of Hamiltonian Monte Carlo Year: (2017)
Ref_id:b35 Title: Analysis of Langevin Monte Carlo via convex optimization Year: (2019)
Ref_id:b36 Title: Acceleration methods Year: (2021)
Ref_id:b37 Title: A continuized view on Nesterov acceleration for stochastic gradient descent and randomized gossip Year: (2021)
Ref_id:b38 Title: Mean-field underdamped Langevin dynamics and its spacetime discretization Year: (2024-07)
Ref_id:b39 Title: Accelerated stochastic optimization methods under quasar-convexity Year: (2023)
Ref_id:b40 Title: Bayesian data analysis Year: (1995)
Ref_id:b41 Title: Decentralized stochastic gradient Langevin dynamics and Hamiltonian Monte Carlo Year: (2021)
Ref_id:b42 Title: Gradient descent learns linear dynamical systems Year: (2018)
Ref_id:b43 Title: Near-optimal methods for minimizing star-convex functions and beyond Year: (2020)
Ref_id:b44 Title: Entropy-based adaptive Hamiltonian Monte Carlo Year: (2021)
Ref_id:b45 Title: Neutralizing bad geometry in Hamiltonian Monte Carlo using neural transport Year: (2019)
Ref_id:b46 Title: Tuning-free generalized Hamiltonian Monte Carlo Year: (2022)
Ref_id:b47 Title: The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo Year: (2014)
Ref_id:b48 Title: Dissipativity theory for Nesterov's accelerated method Year: (2017)
Ref_id:b49 Title: On the dissipation of ideal Hamiltonian Monte Carlo sampler Year: (2023)
Ref_id:b50 Title: The variational formulation of the Fokker-Planck equation Year: (1998)
Ref_id:b51 Title: Adam: A method for stochastic optimization Year: (2015)
Ref_id:b52 Title: Accelerated mirror descent in continuous and discrete time Year: (2015)
Ref_id:b53 Title: Doing Bayesian Data Analysis: A Tutorial with Year: (2014)
Ref_id:b54 Title: Silvère Bonnabel, and Philippe Rigollet. Variational inference via Wasserstein gradient flows Year: (2022)
Ref_id:b55 Title: Convergence rate of Riemannian Hamiltonian Monte Carlo and faster polytope volume computation Year: (2018)
Ref_id:b56 Title: Algorithmic theory of ODEs and sampling from well-conditioned logconcave densities Year: (2018)
Ref_id:b57 Title: Structured logconcave sampling with a restricted Gaussian oracle Year: (2021)
Ref_id:b58 Title: Simulating Hamiltonian Dynamics Year: (2004)
Ref_id:b59 Title: Contraction and convergence rates for discretized kinetic Langevin dynamics Year: (2024)
Ref_id:b60 Title: Partial differential equations and stochastic methods in molecular dynamics Year: (2016)
Ref_id:b61 Title: On explicit L 2 -convergence rate estimate for piecewise deterministic Markov processes in MCMC algorithms Year: (2022)
Ref_id:b62 Title: Is there an analog of Nesterov acceleration for gradient-based MCMC? Year: (2021)
Ref_id:b63 Title: Hamiltonian descent methods Year: (2018)
Ref_id:b64 Title: Nonparametric Hamiltonian Monte Carlo Year: (2021)
Ref_id:b65 Title: Rapid mixing of Hamiltonian Monte Carlo on strongly log-concave distributions Year: (2017)
Ref_id:b66 Title: Mixing of Hamiltonian Monte Carlo on strongly log-concave distributions 2: Numerical integrators Year: (2019)
Ref_id:b67 Title: Mixing of Hamiltonian Monte Carlo on strongly log-concave distributions: Continuous dynamics Year: (2019)
Ref_id:b68 Title: Dimensionally tight bounds for second-order Hamiltonian Monte Carlo Year: (2018)
Ref_id:b69 Title: Fast Convergence of Φ-Divergence Along the Unadjusted Langevin Algorithm and Proximal Sampler Year: (2025)
Ref_id:b70 Title: Reflection, refraction, and Hamiltonian Monte Carlo Year: (2015)
Ref_id:b71 Title: HMC and Langevin united in the unadjusted and convex case Year: (2022)
Ref_id:b72 Title: An entropic approach for Hamiltonian Monte Carlo: The idealized case Year: (2024)
Ref_id:b73 Title: Faster estimation of Bayesian models in ecology using Hamiltonian Monte Carlo Year: (2017)
Ref_id:b74 Title: A dynamical systems perspective on Nesterov acceleration Year: (2019)
Ref_id:b75 Title: MCMC using Hamiltonian dynamics Year: (2011)
Ref_id:b76 Title: A method for solving the convex programming problem with convergence rate O(1/k 2 ) Year: (1983)
Ref_id:b77 Title: Lectures on convex optimization Year: (2018)
Ref_id:b78 Title: Discontinuous Hamiltonian Monte Carlo for models with discrete parameters and discontinuous likelihoods Year: (2020)
Ref_id:b79 Title: Hamiltonian descent for composite objectives Year: (2019)
Ref_id:b80 Title: The geometry of dissipative evolution equations: The porous medium equation Year: (2001)
Ref_id:b81 Title: Auxiliary-variable exact Hamiltonian Monte Carlo samplers for binary distributions Year: (2013)
Ref_id:b82 Title: Some methods of speeding up the convergence of iteration methods Year: (1964)
Ref_id:b83 Title: Metropolis-adjusted Langevin trajectories: A robust alternative to Hamiltonian Monte Carlo Year: (2022)
Ref_id:b84 Title: Exponential convergence of langevin distributions and their discrete approximations Year: (1996)
Ref_id:b85 Title: Microcanonical Hamiltonian Monte Carlo Year: (2023)
Ref_id:b86 Title: The Wasserstein proximal gradient algorithm Year: (2020)
Ref_id:b87 Title: Probabilistic programming in Python using PyMC3 Year: (2016)
Ref_id:b88 Title: Symplectic integrators for Hamiltonian problems: An overview Year: (1992)
Ref_id:b89 Title: Positive curvature and Hamiltonian Monte Carlo Year: (2014)
Ref_id:b90 Title: Understanding the acceleration phenomenon via high-resolution differential equations Year: (2022)
Ref_id:b91 Title: A differential equation for modeling Nesterov's accelerated gradient method: Theory and insights Year: (2016)
Ref_id:b92 Title: Continuous-time analysis of accelerated gradient methods via conservation laws in dilated coordinate systems Year: (2022)
Ref_id:b93 Title: Convergence of mean-field Langevin dynamics: Time-space discretization, stochastic gradient, and variance reduction Year: (2023)
Ref_id:b94 Title: Stochastic Hamiltonian systems: exponential convergence to the invariant measure, and discretization by the implicit Euler scheme Year: (2002)
Ref_id:b95 Title: First-order optimization algorithms with resets and Hamiltonian flows Year: (2019)
Ref_id:b96 Title: Magnetic Hamiltonian Monte Carlo Year: (2017)
Ref_id:b97 Title: Rapid convergence of the unadjusted langevin algorithm: Isoperimetry suffices Year: (2019)
Ref_id:b98 Title: Frictionless Hamiltonian descent and coordinate Hamiltonian descent for strongly convex quadratic problems and beyond Year: (2024)
Ref_id:b99 Title: Accelerating Hamiltonian Monte Carlo via Chebyshev integration time Year: (2023)
Ref_id:b100 Title: Continuized acceleration for quasar convex functions in non-convex optimization Year: (2023)
Ref_id:b101 Title: Accelerated information gradient flow Year: (2022)
Ref_id:b102 Title: Sampling as optimization in the space of measures: The Langevin dynamics as a composite optimization problem Year: (2018)
