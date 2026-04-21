Title: Go With the Flow: Fast Diffusion for Gaussian Mixture Models
Abstract: Schrödinger Bridges (SBs) are diffusion processes that steer, in finite time, a given initial distribution to another final one while minimizing a suitable cost functional. Although various methods for computing SBs have recently been proposed in the literature, most of these approaches require computationally expensive training schemes, even for solving low-dimensional problems. In this work, we propose an analytic parametrization of a set of feasible policies for steering the distribution of a dynamical system from one Gaussian Mixture Model (GMM) to another. Instead of relying on standard non-convex optimization techniques, the optimal policy within the set can be approximated as the solution of a low-dimensional linear program whose dimension scales linearly with the number of components in each mixture. The proposed method generalizes naturally to more general classes of dynamical systems, such as controllable linear time-varying systems, enabling efficient solutions to multi-marginal momentum SBs between GMMs, a challenging distribution interpolation problem. We showcase the potential of this approach in low-to-moderate dimensional problems such as image-to-image translation in the latent space of an autoencoder, learning of cellular dynamics using multi-marginal momentum SBs, and various other examples. The implementation is publicly available at https://github.com/georgeRapa/GMMflow.

Section: Introduction and Background
The problem of finding mappings between distributions of data, originally known as the Optimal Transport (OT) problem in mathematics, has received significant attention in recent years in multiple research fields, due to its application in problems such as generative AI (Ruthotto & Haber, 2021;Arjovsky et al., 2017), biology (Bunne et al., 2023b;Bunne & Rätsch, 2023;Tong et al., 2020), mean field problems (Liu et al., 2022) and control theory (Chen et al., 2015a,b;Rapakoulias & Tsiotras, 2024) among many others. Despite appearing static in nature, reformulating OT in the context of dynamical systems imbues it with further structure and unlocks tools from the literature on dynamical systems that can be employed for its efficient solution (Benamou & Brenier, 2000).
To set the stage, consider two distributions ρ 0 , ρ 1 , supported on the d-dimensional Euclidean space, denoted by R d , and consider the regularized version of the static OT optimization problem, known as the Entropic Optimal Transport (EOT) problem (Peyré & Cuturi, 2019):
min π∈Π(ρ0,ρ1) R d ×R d 1 2 ∥x 0 -x 1 ∥ 2 dπ(x 0 , x 1 ) -ϵH(π),(1)
where π(x 0 , x 1 ) is the transport plan (also referred to as coupling) between ρ 0 , ρ 1 , Π(ρ 0 , ρ 1 ) is the set of all joint distributions with marginals ρ 0 , ρ 1 , and H is the differential entropy, defined by H(ρ) ≜ -ρ(x) log ρ(x) dx. The corresponding dynamic formulation of the EOT problem is known as the Schrödinger Bridge Problem (SBP) (Léonard, 2014;Chen et al., 2021). When formulated as a stochastic optimal control problem, the SBP is given by
min u∈U J SB ≜ E xt∼ρt 1 0 1 2ϵ ∥u t (x t )∥ 2 dt , s.t. dx t = u t (x t ) dt + √ ϵ dw, x 0 ∼ ρ 0 , x 1 ∼ ρ 1 ,(2)
where the objective is to find an optimal drift function u t (x t ), also referred to as the control policy in the context of control applications, belonging to a set of adapted finite-energy policies U, such that, when applied to the stochastic dynamical system defined by the first constraint in (2), the marginal distribution specified in the second constraint is guaranteed, i.e., for initial conditions sampled at time t = 0 from ρ 0 , the state at time t = 1 will be distributed according to ρ 1 , and the cost J SB in (2) will be minimized.
The increased practical applications of EOT and SBs in multiple machine learning problems, especially in high-dimensional generative applications where the boundary distributions ρ 0 , ρ 1 are only available through a finite number of samples, have led to the development of a multitude of algorithms over recent years. The state-of-the-art methods for solving SBs leverage the properties of problem (2), such as the decomposition of the optimal probability flow into conditional problems that are easier to solve, sometimes even analytically (Chen et al., 2016;Lipman et al., 2023;Liu et al., 2023). In this category of methods, a recent technique known as Diffusion Schrödinger Bridge Matching (DSBM) (Shi et al., 2023;Peluchetti, 2023), or its deterministic counterpart, known as Flow Matching (FM) (Lipman et al., 2023) or Rectified Flow (RF) (Liu et al., 2023), leverages the decomposition of the optimal probability flow to a mixture of flows conditioned on their respective endpoints and retrieves an approximation of the optimal solution to ( 2) as a mixture of conditional policies that are easy to calculate. Theoretically, one needs to combine an infinite number of conditional flows to retrieve the true flow, due to the continuous support of the boundary distributions. To overcome this issue, a neural network is usually trained to approximate this infinite mixture.
While the DSBM and the various Flow Matching algorithms have proven very effective in highdimensional problems, the efficient solution of SBs in simpler problems is hindered by the lack of closed-form expressions in all but very few special cases with Gaussian marginal distributions (Bunne et al., 2023a). To tackle this problem and avoid costly neural network training in smaller problems, recent methods such as Light-SB (LSB) (Korotin et al., 2024) and Light-SB Matching (LSBM) (Gushchin et al., 2024) have been proposed to obtain quick and efficient solutions to SBs within seconds, for problems with low-complexity boundary distributions, such as mixture models. These methods work by an efficient parametrization of the Schrödinger potentials, a key component of the SB. Because this parametrization does not lead to closed-form expressions for the boundary distributions of the SB, the calculation of its parameters is carried out through optimization.
Inspired by the flow decomposition idea behind DSBM and FM methods, and motivated by the need to obtain light-weight and fast SB solvers for a wide class of SB problems, in this paper, we solve the problem of finding a policy that can efficiently steer the distribution of a dynamical system from a Gaussian Mixture Model (GMM) to another one, using a mixture of conditional policies that can each steer the individual components of the initial mixture to the components of the terminal mixture. This approach, which is tailored to GMMs, separates the problem of fitting the boundary distributions to the data and solving the SB, resulting in improved accuracy with regard to the marginal distribution fitting. More specifically, we claim the following main contributions:
1. We present a computationally efficient, training-free method to solve the Schrödinger Bridge and the multi-marginal Momentum Schrödinger Bridge problems in the case where the boundary distributions are Gaussian Mixture Models.
2. In contrast to existing approaches, our method can handle both stochastic and deterministic versions of the problem (2). Based on a control-theoretic formulation, our approach also naturally generalizes to dynamical systems with a general Linear Time-Varying (LTV) structure, with the control input and stochastic component having different dimensions than the state, which could be of interest in Mean Field Games (MFG), multi-agent control applications (Ruthotto et al., 2020;Liu et al., 2022;Chen, 2024), and higher order distribution interpolation such as Wasserstein splines (Chen et al., 2018).
3. We demonstrate the substantial potential of our algorithm in low-dimensional problems, moderate-dimensional image-to-image translation tasks, and multi-marginal diffusion learning problems. Specifically, we show that our approach outperforms state-of-theart lightweight methods for solving the SB problem both in terms of training speed and accuracy of the learned boundary distributions, when these are available through samples (40% better FID scores in the image translation task and one order of magnitude better MMD scores in the multi-marginal diffusion-learning problems). 4. Finally, we extend our method to problems with continuous GMM marginal distributions, a wide class of distributions that can capture multiple useful distributions with heavy tails, and we use our approach to construct upper bounds on the 2-Wasserstein distance and approximate the displacement interpolation between Student-t distributions.
2 Preliminaries
this section cite: ['b54', 'b2', 'b63', 'b40', 'b53', 'b4', 'b50', 'b36', 'b22', 'b19', 'b37', 'b42', 'b59', 'b49', 'b37', 'b42', 'b34', 'b29', 'b55', 'b40', 'b16', 'b20']

Section: Diffusion Schrödinger Bridge Matching and Flow Matching
The composition of diffusion processes as mixtures of processes conditioned on their endpoints was originally proposed by Peluchetti (2021) as a simulation-free algorithm for generative modeling applications. The concept was later tailored to solve the SBP in the DSBM algorithm (Shi et al., 2023), proposed concurrently by Peluchetti (2023). Similar simulation-free methods have also been proposed to solve variants of the same problem in Albergo & Vanden-Eijnden (2023) and in Liu et al. (2024); Theodoropoulos et al. (2025) for the stochastic bridge setting, as well as in Liu et al. (2023) and Lipman et al. (2023) for the deterministic setting. For a more comprehensive overview along with comparisons with other available methods, we refer the reader to (Shi et al., 2023, Section 5) and (Peluchetti, 2023, Section 5).
Given problem (2), the main idea is to decompose the problem into a sequence of elementary conditional subproblems that are easier to solve, and then express the solution as a mixture of the solutions of the conditional subproblems. This idea has an intuitive motivation: Informally, finding a policy that transports the state distribution from an initial density to a target density can be separated into two problems. First, one needs to figure out a transport plan solving the "who goes where" problem and then one needs to compute a point-to-point optimal policy, that solves the "how to get there" problem (Terpin et al., 2024a). In many cases, the two subproblems are decoupled (Chen et al., 2021(Chen et al., , 2016;;Terpin et al., 2024a); most importantly, however, computing the point-to-point optimal policy can be solved analytically for simple dynamical systems, such as the one in (2).
More precisely, the optimal probability flow ρ * t of Problem (2) is known (Föllmer, 1988;Chen et al., 2021) to admit the decomposition
ρ * t (x) = R d ×R d W t|x0,x1 (x) dπ * ϵ (x 0 , x 1 ),(3)
where W t|x0,x1 (x) is the probability density of the unforced dynamics dx t = √ ϵ dw, namely the Brownian motion kernel, pinned at x 0 for t = 0 and at x 1 for t = 1, and π * ϵ (x 0 , x 1 ) is the entropic optimal transport plan between ρ 0 , ρ 1 solving (1). Dai Pra (1991) showed that W t|x0,x1 (x) solves the following optimal control problem
min u t|0,1 ∈U J 0,1 (x 0 , x 1 ) ≜ E 1 0 ∥u t|0,1 (x t )∥ 2 dt , s.t. dx t = u t|0,1 (x t ) dt + √ ϵ dw, x 0 ∼ δ x0 , x 1 ∼ δ x1 ,(4)
where δ x0 , δ x1 are Dirac delta functions centered on x 0 and x 1 , respectively. Assuming ρ t|0,1 (x) and u t|0,1 (x) solve (4), one can construct a feasible solution for the original problem (2) using any transport plan q(x 0 , x 1 ) ∈ Π(ρ 0 , ρ 1 ), i.e., any joint distribution between the desired boundaries ρ 0 , ρ 1 , using the mixtures
ρ t (x) = R d ×R d ρ t|0,1 (x)q(x 0 , x 1 ) dx 0 dx 1 , (5a
) u t (x) = R d ×R d u t|0,1 (x) ρ t|0,1 (x)q(x 0 , x 1 ) ρ t (x) dx 0 dx 1 .(5b)
Showing that the flow (5a) is a feasible solution to (2) for any valid coupling q(x 0 , x 1 ) amounts to verifying that the flow ρ t (x) satisfies the boundary distributions ρ 0 , ρ 1 at times t = 0 and t = 1, respectively. To prove that the policy (5b) produces (5a), it suffices to show that the pair (5a), (5b) satisfies the FPK PDE (Lipman et al., 2023;Liu et al., 2024). When q(x 0 , x 1 ) = π * ϵ (x 0 , x 1 ), (5a) reduces to (3), and (5b) recovers the optimal solution to (2) (Shi et al., 2023;Peluchetti, 2023).
this section cite: ['b48', 'b59', 'b49', 'b0', 'b62', 'b42', 'b37', 'b59', 'b22', 'b19', 'b27', 'b22', 'b24', 'b37', 'b59', 'b49']

Section: Schrödinger Bridges with Gaussian Marginals
The SBP with Gaussian Marginals, henceforth referred to as the Gaussian SB (GSB), has been extensively studied in the literature and can be solved either analytically for simple choices of prior dynamics (Bunne et al., 2023a) or as a convex semidefinite optimization problem for general linear dynamical systems both for continuous and discrete time cases (Chen et al., 2015b;Liu et al., 2025). Because we use the GSB as a building block to construct a policy that works with general GMM boundary distributions, we briefly review the available methods for its solution here. To this end, consider the optimization problem with Gaussian marginals
min u∈U J GSB ≜ E 1 0 ∥u t (x t )∥ 2 dt , s.t. dx t = u t (x t ) dt + √ ϵ dw, x 0 ∼ N (µ 0 , Σ 0 ), x 1 ∼ N (µ 1 , Σ 1 ),(6)
where µ 0 , Σ 0 , µ 1 , Σ 1 are the means and covariances of the initial and final Gaussian boundary distributions, respectively.
Proposition 1. (Bunne et al., 2023a, Theorem 3) The optimal solution to Problem (6) is given by
u t (x) = K t (x -µ t ) + v t with µ t = (1 -t)µ 0 + tµ 1 , v t = µ 1 -µ 0 , and K t = S T t Σ -1 t ,
where
Σ t = (1 -t) 2 Σ 0 + t 2 Σ 1 + (1 -t)t(C ϵ + C T ϵ + ϵI), S t = t(Σ 1 -C T ϵ ) -(1 -t)(Σ 0 -C ϵ ) -ϵtI, with C ϵ = 1 2 (Σ 1 2 0 D ϵ Σ -1 2 0 -ϵI) and D ϵ = (4Σ 1 2 0 Σ 1 Σ 1 2 0 + ϵ 2 I) 1 2 .
Furthermore, the optimal value of the cost J GSB in ( 6) is given by the following proposition.
Proposition 2. Consider Problem (6) with ϵ > 0. Then, the optimal value for the cost J GSB is
J GSB = ∥µ 1 -µ 0 ∥ 2 + tr(Σ 0 ) + tr(Σ 1 ) -ϵ (trM 2ϵ -log det M 2ϵ + log det Σ 1 ) + c, (8
)
where M ϵ = I + (I + 16
ϵ 2 Σ 0 Σ 1 ) 1 2
, and c is a constant independent of the boundary distributions.
In the limit of ϵ → 0, (8) reduces to the well known Bures-Wasserstein distance (Bhatia et al., 2019), defined by
BW(N (µ 0 , Σ 0 )∥N (µ 1 , Σ 1 )) ≜ ∥µ 1 -µ 0 ∥ 2 + tr(Σ 0 ) + tr(Σ 1 ) -2tr Σ 1 2 1 Σ 0 Σ 1 2 1 1 2 .(9)
this section cite: ['b39', 'b6']

Section: Momentum Schrödinger Bridges with multiple Gaussian marginals
Other than Problem (6), we will also make use of the solution to the corresponding Gaussian multimarginal Momentum SB (GMSB) (Chen et al., 2019), which is a variation of Problem (6) where the dynamics include a momentum term, while the goal is to match multiple marginal distributions at regular time intervals. More specifically, the GMSB problem reads
min u∈U E 1 0 ∥u t (x t , v t )∥ 2 dt ,(10a)
s.t. dx t = v t dt, dv t = u t (x t , v t ) dt + √ ϵ dw, (10b) x ti ∼ N (µ ti , Σ ti ), i = 1, . . . , N, (10c
)
where N is the number of marginal constraints and the joint space of position and velocity, namely x t , v t ∈ R d , is referred to as the phase space. Compared to the standard SB problem, and apart from having multiple marginal distributions, the GMSB problem only constrains the position component of the phase space, namely x t . Even when the marginals are Gaussian, a closed-form solution to ( 10) is unknown; however, the problem can be solved efficiently using semidefinite programming. In the special case where the noise parameter ϵ is zero, the problem is known as the Gaussian Wasserstein spline Problem (Chen et al., 2018), and an efficient semidefinite formulation for solving it is given in Chen et al. (2018). Since the semidefinite formulation for solving (10) is a well-studied problem, due to space considerations, we defer it to Appendix D. Finally, we note that a generalization of the Problems ( 6) and ( 10) is achieved by a Linear Time Varying (LTV) structure in the prior dynamics of the bridge, i.e., replacing the first constraint in (6) with
dx t = A t x t dt + B t u t dt + D t dw,(11)
where x t ∈ R d , A t ∈ R d×d , u t ∈ R m , B t ∈ R d×m , D t ∈ R d×q , w t ∈ R q . Bridges with prior dynamics of the form (11) have been extensively studied in the context of control theory, with the corresponding literature known as Covariance Steering (CS) (Chen et al., 2015a,b;Bakolas, 2018;Liu et al., 2025). CS problems can be formulated as convex programs for both continuous and discrete-time cases (Chen et al., 2015b;Liu et al., 2025), and therefore attain an efficient and exact calculation, which we will exploit in the sequel.
3 Fast Diffusion for Mixture Models
this section cite: ['b21', 'b20', 'b20', 'b3', 'b39', 'b39']

Section: Gaussian Mixture Schrödinger Bridge
Equation ( 5b) expresses the policy of Problem (2) as an infinite mixture of conditional, point-to-point policies. In this section, we extend this idea to construct a mixture policy, consisting of conditional policies each solving a Gaussian bridge sub-problem of the form (6). To this end, consider the problem
min u∈U J GMM ≜ E 1 0 ∥u t (x t )∥ 2 dt ,(12a)
s.t. dx t = u t (x t ) dt + √ ϵ dw,(12b)
x 0 ∼ N0 i=1 α i 0 N µ i 0 , Σ i 0 , x 1 ∼ N1 j=1 α j 1 N µ j 1 , Σ j 1 .(12c)
The main result is summarized in the following theorem.
Theorem 1. Consider problem (12), with N 0 components in the initial mixture and N 1 components in the terminal mixture. Assume that u t|ij is the conditional policy that solves the (i, j)-GSB problem, that is, the bridge from the i-th component of the initial mixture, to the j-th component of the terminal mixture and let the resulting probability flow be ρ t|ij . Furthermore, let λ ij ≥ 0 such that, for all j ∈ {1, 2, . . . , N 1 }, i λ ij = α j 1 and such that, for all i ∈ {1, 2, . . . , N 0 }, j λ ij = α i 0 . Then, the policy
u t (x) = i,j u t|ij (x) ρ t|ij (x)λij i,j ρ t|ij (x)λij , (13
)
is a feasible policy for Problem (12), and the corresponding probability flow is
ρ t (x) = i,j ρ t|ij (x)λ ij .(14)
The mixture policy (13) is a weighted average of conditional policies, weighted according to λ ij ρ t|ij (x), while the denominator i,j ρ t|ij (x)λ ij is just a normalizing constant. Since ρ t|ij (x) is a Gaussian distribution centered at the mean of the (i, j)-Gaussian bridge at time t, this weighting scheme prioritizes the conditional policies whose mean is closer to the value of x at the time t.
Equation ( 13) provides a set feasible of solutions to Problem (12), for all values of λ ij satisfying the conditions of Theorem 1. Obtaining the optimal policy within this set is challenging, in general. Alternatively, we can formulate a tractable problem by minimizing an upper bound to the original minimum effort cost function (12a), which is linear with respect to the transport plan λ ij . More formally, we have the following theorem.
Theorem 2. Let J ij be the optimal cost of solving the (i, j)-Gaussian bridge subproblem of the form (6) with marginal distributions the i-th component of the initial and the j-th component of the terminal mixture. Then, the cost function of the linear optimization problem
min λij ≥ 0 J OT ≜ i,j λ ij J ij (15a
) s.t. j λ ij = α i 0 ∀i ∈ {1, 2, . . . , N 0 }, and i λ ij = α j 1 , ∀j ∈ {1, 2, . . . , N 1 }, (15b
)
provides an upper bound for (12a), i.e., J GMM ≤ J OT , for all positive values of λ ij satisfying (15b).
For clarity of exposition, we defer the proofs of Theorems 1, 2 to Appendix A, along with an optimality analysis of the upper bound of Theorem 2.
In practice, to use policy (13) in problems where the boundary distributions are available only through samples, GMMs are first fitted in the samples of ρ 0 , ρ 1 using the Expectation Maximization (EM) algorithm (Bishop & Nasrabadi, 2006), and then (13) is calculated using Theorems 1, 2. For clarity of exposition, we present an overview of this approach in Algorithms 1, 2 for both training and inference, accompanied by the corresponding theoretical complexity analysis in Appendix A.6.
Algorithm 1 GMMflow training Input: Samples from boundary distributions ρ 0 , ρ 1 ; number of GMM components N 0 and N 1 , noise level ϵ ≥ 0.
{α i 0 , µ i 0 , Σ i 0 } N0 i=1 ←EM(ρ 0 , N 0 ) // fits a GMM to initial dataset {α j 1 , µ j 1 , Σ j 1 } N1 j=1 ←EM(ρ 1 , N 1 ) // fits a GMM to final dataset for (i, j) ∈ {1, . . . , N 0 } × {1, . . . , N 1 } do //compute in parallel {u t|ij , ρ t|ij , J ij } ← CS(µ i 0 , Σ i 0 , µ j 1 , Σ j 1 ) //solves the (i, j)-th conditional GSB end for λ ij ← SOLVE (15) USING {J ij , α i 0 , α j 1 } return u t|ij , ρ t|ij , λ ij Algorithm 2 GMMflow inference Input: Component-level solutions u t|ij , ρ t|ij , transport plan λ ij , Initial condition x 0 ∼ ρ 0 , SDE integrator sde int(). u t (x) ← Compute (13) x t ← sde int((12b), x 0 , t ∈ [0, 1]) return x 1
this section cite: ['b7']

Section: Multi-Marginal Problems
In this section, we generalize the results of Section 3.1 to solve the multi-marginal momentum SBs problem (Chen et al., 2023(Chen et al., , 2019) ) with GMM marginal distributions. That is, we solve
min u∈U J GMM ≜ E 1 0 ∥u t (x t , v t )∥ 2 dt (16a
) s.t. dx t = v t dt, dv t = u t (x t , v t ) dt + √ ϵ dw t ,(16b)
x ti ∼ Ni k=1 α k i N (µ k i , Σ k i ), i = 1, . . . M,(16c)
where the i-th marginal mixture is assumed to have N i components. Similarly to Theorem 1, we will combine conditional GMSBs of the form (10) to build a feasible set of policies solving (16).
To facilitate notation, we denote by i = (i 1 , . . . , i M ) the index of the conditional multi-marginal GMSB and use the notation {i|i j = k} to denote the set of all values of i such that i j = k. With this notation in mind, we provide the following generalizations of Theorems 1 and 2:
Theorem 3. Consider problem (16), with M marginal mixture distributions, each having N i Gaussian components, where i = 1, . . . M . Let i = (i 1 . . . i M ) be an M -dimensional index, u t|i be the conditional policy that solves the i-GMSB problem, that is, the Gaussian multi-marginal momentum Schrödinger Bridge going through the (i 1 , . . . , i M ) components of the marginal mixture models, and let the resulting probability flow be ρ t|i . Furthermore, let λ i ≥ 0 be such that, for all j = 1, . . . , M and for all k = 1, . . . , N j ,
{i|ij =k} λ i = α k j .(17)
Then, the policy
u t (x, v) = i u t|i (x, v) ρ t|i (x,v)λ i i ρ t|i (x,v)λ i ,(18)
is a feasible policy for Problem (12), and the corresponding probability flow is
ρ t (x, v) = i ρ t|i (x, v)λ i .(19)
To approximate the optimal multi-marginal transport plan λ i , we use the following upper bound.
Theorem 4. Let J i be the optimal cost of solving the i-GMSB subproblem of the form (10) with marginal distributions (i 1 , . . . , i M ) components of the marginal mixtures. Then, the cost function of the linear optimization problem
min λ i ≥ 0 J OT ≜ i λ i J i(20a)
s.t. {i|ij =k} λ i = α k j , ∀ j = 1, . . . , M, k = 1, . . . , N j(20b)
provides an upper bound for (16a), that is, J GMM ≤ J OT , for all values of λ i satisfying (20b).
In practice, in order to use policy (18) for inference, the initial conditions for the SDE (16b) must be defined. Specifically, to fully define the initial conditions in the state space [x 0 , v 0 ], given an initial position sample x 0 , the corresponding velocity v 0 must be estimated. To this end, note that given λ i , the state distribution ρ t (x t , v t ) is a fully defined GMM given by Equation ( 19). It is easy to show that the conditional distribution ρ t (v t |x t ) is also a GMM, whose parameters can be easily computed. Since this calculation is trivial, we defer it to Appendix C.3. For completeness, we provide the multimarginal inference algorithm in Algorithm 3. The multi-marginal training algorithm is omitted due to its similarity to ( 2).
Algorithm 3 Multi-marginal GMMflow inference Input: Component-level solutions u t|i , ρ t|i , multi-marginal transport plan λ i , sample from x 0 ∼ ρ 0 (x 0 ), SDE integrator sde int(). u t (x) ← Compute (18) Sample v 0 ∼ ρ 0 (v 0 |x 0 ) using Equation (B.19d) [x t ; v t ] ← sde int(Equation (16b), [x 0 ; v 0 ], t ∈ [0, 1]) return x t
this section cite: ['b15', 'b21']

Section: Continuous Gaussian Mixtures
The results of Section 3.1 can be extended to problems with continuous GMM boundary distributions. Specifically, we consider a bridge of the form
min u∈U E 1 0 ∥u t (x t )∥ 2 dt ,(21a)
s.t. dx t = u t (x t ) dt, (21b
) x i ∼ R m N (µ i (w i ), Σ i (w i )) dP i (w i ), i = 0, 1,(21c)
where the boundary distributions (21c) are continuous GMMs with mixing measures P (w 0 ), P (w 1 ) respectively. We keep the dynamics (21b) deterministic to simplify the analysis, although all the results carry over to the general case of stochastic dynamics as well. This more general formulation specializes to problem (12) when P (w 0 ), P (w 1 ) have discrete support; however, it includes many other scenarios where the mixing distributions are continuous. Specifically, the generalization of Theorem 1 is as follows.
Theorem 5. Consider Problem (21) and assume that u t|w0,w1 is the conditional policy that solves the (w 0 , w 1 )-GSB problem, that is the bridge from the initial Gaussian distribution with parameter w 0 to the terminal Gaussian distribution with parameter w 1 and let the resulting probability flow be ρ t|w0,w1 . Furthermore, let Λ(w 0 , w 1 ) be any coupling such that its marginal distributions are P 0 and P 1 respectively. Then, the policy
u t (x) = R m ×R m u t|w0,w1 (x) ρ t|w0,w1 (x) dΛ(w 0 , w 1 ) R m ×R m ρ t|w0,w1 (x) dΛ(w 0 , w 1 ) (22
)
is a feasible policy for Problem (21) and the corresponding probability flow is
ρ t (x) = R m ×R m ρ t|w0,w1 (x) dΛ(w 0 , w 1 ).(23)
Furthermore, Theorem 2 generalizes to the following.
Theorem 6. Let J(w 0 , w 1 ) be the optimal cost of solving the (w 0 , w 1 )-Gaussian bridge subproblem. Then, the optimal transport problem:
J OT ≜ min Λ∈Π(P0,P1) R m ×R m J(w 0 , w 1 ) dΛ(w 0 , w 1 ),(24)
provides an upper bound for Problem (21), that is, J GMM ≤ J OT , where Π(P 0 , P 1 ) represents the set of all couplings with marginals P 0 and P 1 .
Problem ( 24) is challenging to solve in general. However, in many practical cases, such as for Student-t boundary distributions, the parameter spaces for w 0 , w 1 are one-dimensional and ( 24) can be solved in closed form. We explore this interesting direction in Appendix B to approximate the Wasserstein-2 distance and the displacement interpolation between heavy-tail distributions.
this section cite: []

Section: Related Work
Although the idea of creating a mixture policy from elementary point-to-point policies is at the heart of flow-matching, to the best of our knowledge, its benefits for solving problems with mixture models with a finite number of components have not been explored. In an early work, Chen et al. (2016) developed an upper bound on the 2-Wasserstein distance between Gaussian mixture models, i.e., the static, deterministic version of Problem (12), that matches the upper bound of Theorem 2. However, the problem of finding a policy that solves the dynamic problem was not explored. Focusing on works concerning dynamic problems, the concept of constructing stochastic differential equations (SDEs) as mixtures of Gaussian probability flows can be traced back to mathematical finance applications (Brigo, 2002;Brigo et al., 2002). More recently, and in the context of generative applications, Albergo & Vanden-Eijnden (2023) developed similar expressions for finding a policy for steering between mixture models using an alternative conditional solution for the Gaussian-to-Gaussian bridge, focused on the case of deterministic, fully observable dynamical systems with no prior dynamics, and the component-level transport plan was not optimized.
Our work reveals some similarities in scope and structure with LSB (Korotin et al., 2024) and LSBM (Gushchin et al., 2024). Similarly to LSB and LSBM, we aim to provide numerically inexpensive tools to solve the SBP in low-to-moderate dimensional scenarios. Moreover, our feedback policy in Equation ( 13) is a mixture of affine feedback terms weighted with exponential kernels; this is also the case in LSB and LSBM (Korotin et al., 2024, Proposition 3.3). The formulations, however, are otherwise quite distinct. Specifically, LSB/LSBM works by modeling the so-called Schrödinger potentials using a GMM. This results in flows with boundary distributions that have a mixturelike structure and are optimal by construction, but whose marginals are neither amenable to exact calculation, nor are GMMs, in general. Finally, to approximate the optimal flows for boundary distributions available only through samples, both works solve non-linear optimization problems, which are prone to converging to locally optimal solutions. In contrast, our approach works by first pre-fitting GMMs to the data using the Expectation Maximization (EM) (Pedregosa et al., 2011), and then computing an optimal policy by solving exactly a linear program. Finally, being based on a control-theoretic framework, our method generalizes effectively to partially observable and multimarginal distribution matching problems as shown in Section 3.2, while extending LSB and LSBM to handle such problems is non-trivial.
5 Experiments 2D Problems and Benchmarks. We first test the algorithm in various 2D "toy" problems as shown, for example, in Figure 1 for a Gaussian-to-Gaussian Mixture problem for various noise levels. To assess optimality, we evaluate the resulting transport cost for policy (13) for each noise level and compare it with the upper bound from (15a). We also run a series of EOT benchmarks and compare them with state-of-the-art neural approaches such as the DSB (De Bortoli et al., 2021) and DSBM (Shi et al., 2023) algorithms. Due to space considerations, we defer their discussion to Appendix C, along with further experiments studying training and inference time scaling with respect to the problem dimension and the number of GMM components. To evaluate the performance on problems with many GMM components, we tested the algorithm on the distributions depicted in Figure 2, where we first pre-fit 500-component GMMs in the initial and terminal samples.
this section cite: ['b19', 'b9', 'b10', 'b0', 'b34', 'b29', 'b47', 'b25', 'b59']

Section: Image-to-Image Translation.
Following Korotin et al. (2024), we use our algorithm in the latent space of an autoencoder to perform a man-to-woman and adult-to-child image translation task. We   use the pre-trained ALAE autoencoder (Pidhorskyi et al., 2020), trained on the FFHQ dataset (Karras et al., 2021). The latent space of the autoencoder is 512-dimensional. We start by fitting a 10-component mixture model to the embeddings of each image class, with diagonal covariance matrices to facilitate matrix inversions in the Gaussian-to-Gaussian policy calculations summarized in Proposition 1, and then apply Algorithms 1, 2 for ϵ = 0.01. The results are illustrated in Figure 3.
To test how well the generated images match the features of the given target distribution, we calculate the Fréchet inception distance (FID) scores (Heusel et al., 2017) between the actual and the generated images of a given class, using 10,000 samples from each distribution. The FID scores correspond to the empirical Bures-Wasserstein distance between the images of the two classes, evaluated in the latent space of the Inception network. To further test how close the transformed images are to the target class, we also calculate the empirical Bures-Wasserstein distance between the transformed images and the real images of the target class, directly in the latent space of the ALAE autoencoder and report it as ALAE-BW. We compare against two state-of-the-art lightweight SB solvers, namely, LSB and LSBM, with results shown in Tables 1 and 2.
Table 1: Man-to-Woman FID comparison M→W FID ALAE-BW T. Cost LSB 4.94 28.9 8.23 LSBM 4.98 28.3 8.18 GMMflow 3.04 9.3 9.05 Table 2: Adult-to-Child FID comparison A→C FID ALAE-BW T. Cost LSB 6.62 31.00 8.18 LSBM 6.61 30.99 8.19 GMMflow 3.50 8.54 9.33
Qualitatively, our algorithm performs more aggressive feature changes compared to the baseline method, as illustrated in Figure 3. Quantitatively, the features of the transformed images better capture the true distribution of the features of a given target class, given the almost 40% better FID scores and 65% better ALAE-BW scores provided in Tables 1 and 2. We note that the improvement in the FID and ALAE-BW scores comes with a slight increase in the average transport cost, which we measure by E π ∥x 0 -x 1 ∥ 2 , as reported in the last columns of Tables 1 and 2. We attribute this significant performance gain to our use of the EM algorithm for the GMM pre-fitting, which is less prone to converge to locally optimal values, compared to LightSB's maximum likelihood objective, or the LSBM's bridge matching objective. We further remark that our approach takes 63% less time to train, as noted in Table 5 in Appendix C.2.
this section cite: ['b34', 'b51', 'b32', 'b31']

Section: Multi-Marginal Problems.
A key challenge in SBs is learning a system's underlying diffusion process, given samples from partial observations of the distribution of its state, measured at regular time intervals (Chen et al., 2023). This challenge arises, for example, in learning the dynamics of large cell populations throughout their different developmental stages (Bunne et al., 2022;Tong et al., 2020;Terpin et al., 2024b). To showcase the effectiveness of our approach in such problems, we consider the scRNA-seq dataset from ( Moon et al., 2019), with the pre-processing detailed in Tong et al. (2020). The dataset contains samples from the first 100 Principal Components (PC) of individual cell proteins, grouped at 5 regular time intervals, denoted by t 1 , . . . , t 5 .
For our setup, we keep the first 5 PCs from the 5 marginal distributions, and prefit 5-component GMMs in each marginal. We use the second-order model (16b) to capture the prior dynamics and the structure of the system; however, we note that any LTV model with structure of the form (11) would be applicable. To solve the resulting multi-marginal momentum SBs, we first compute the cost of each GMSB and then solve (20). Computing the GMSB cost for all the combinations of components is the most computationally expensive part of our approach. By parallelizing this computation, the total training time is approximately 8 minutes on an i7-12700 CPU.
We visualize the data generated by our method in Figure 4, and provide standard performance metrics in Table 3. Specifically, following Chen et al. ( 2023), we use the Sliced Wasserstein Distance (SWD) and Maximum Mean Discrepancy (MMD) metrics averaged over the 4 predicted time marginals of the dataset. Although there are no light-weight solvers for second-order multi-marginal problems, we compare our method against DMSB (Chen et al., 2023), NSBL (Koshizuka & Sato, 2023), and MIOFlow (Tong et al., 2020). Even though our method requires minutes to compute on a CPU while neural methods take an hour to train on a high-end GPU, the proposed approach outperforms the baselines by one order of magnitude in the MMD metric due to the accurate fitting of the GMMs. The SWD metric, although acceptable, is bounded by the expressivity of the GMMs to capture the higher-order distribution structure. The metrics for MIOFlow, NSBL, and DMSB in Table 3 are taken from (Chen et al., 2023).
this section cite: ['b15', 'b12', 'b63', 'b61', 'b63', 'b15', 'b35', 'b63', 'b15']

Section: Conclusion and Limitations
This paper introduces a novel, efficient method for solving SB problems with GMM boundary distributions by utilizing a mixture of conditional policies, each solving a Gaussian bridge subproblem. In the same way that low-dimensional methods such as Gaussian distributions and mixture models are highly used in statistics and machine learning, we believe that our approach will be a valuable tool in many useful practical problems related to optimal transport, distribution interpolation, distributional control, and related applications, given its very low computational complexity, and excellent empirical performance in complicated statistical problems. The main limitation of our work stems from its dependence on GMM marginal distributions. Since this class of distributions is not designed for use in very high-dimensional problems, we do not expect our method to be applicable in such areas, but rather to work as an efficient tool for obtaining rapid SB solutions to smaller problems.
this section cite: []

Section: References
Ref_id:b0 Title: Building normalizing flows with stochastic interpolants Year: (2023-05)
Ref_id:b1 Title: Scale mixtures of normal distributions Year: (1974)
Ref_id:b2 Title: Wasserstein generative adversarial networks Year: (2017)
Ref_id:b3 Title: Finite-horizon covariance control for discrete-time stochastic linear systems subject to input constraints Year: (2018)
Ref_id:b4 Title: A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem Year: (2000)
Ref_id:b5 Title: Linear-quadratic mean field games Year: (2016)
Ref_id:b6 Title: On the Bures-Wasserstein distance between positive definite matrices Year: (2019)
Ref_id:b7 Title: Pattern Recognition and Machine Learning Year: (2006)
Ref_id:b8 Title: Convex Optimization Year: (2004)
Ref_id:b9 Title: The general mixture diffusion SDE and its relationship with an uncertain-volatility option model with volatility-asset decorrelation Year: (2002)
Ref_id:b10 Title: Lognormal-mixture dynamics under different means Year: (2002)
Ref_id:b11 Title: Neural optimal transport predicts perturbation responses at the single-cell level Year: (2023)
Ref_id:b12 Title: Proximal optimal transport modeling of population dynamics Year: (2022-03)
Ref_id:b13 Title: The Schrödinger bridge between Gaussian measures has a closed form Year: (2023)
Ref_id:b14 Title: Learning singlecell perturbation responses using neural optimal transport Year: (2023)
Ref_id:b15 Title: Deep multi-marginal momentum Schrödinger bridge Year: (2023)
Ref_id:b16 Title: Density control of interacting agent systems Year: (2024)
Ref_id:b17 Title: Optimal steering of a linear stochastic system to a final probability distribution, part I Year: (2015)
Ref_id:b18 Title: Optimal steering of a linear stochastic system to a final probability distribution, part II Year: (2015)
Ref_id:b19 Title: Optimal transport over a linear dynamical system Year: (2016)
Ref_id:b20 Title: Measure-valued spline curves: An optimal transport viewpoint Year: (2018-01)
Ref_id:b21 Title: Multi-marginal Schrödinger bridges Year: (2019-08)
Ref_id:b22 Title: Stochastic control liaisons: Richard Sinkhorn meets Gaspard Monge on a Schrödinger bridge Year: (2021)
Ref_id:b23 Title: Non-asymptotic analysis of diffusion annealed Langevin Monte Carlo for generative modelling Year: (2025)
Ref_id:b24 Title: A stochastic control approach to reciprocal diffusion processes Year: (1991)
Ref_id:b25 Title: Diffusion Schrödinger bridge with applications to score-based generative modeling Year: (2021)
Ref_id:b26 Title: POT: Python optimal transport Year: (2021)
Ref_id:b27 Title: Random fields and diffusion processes Year: (1988)
Ref_id:b28 Title: Building the bridge of Schrödinger: A continuous entropic optimal transport benchmark Year: (2023)
Ref_id:b29 Title: Light and optimal Schrødinger bridge matching Year: (2024-07)
Ref_id:b30 Title: Gaussian error linear units (GELUs) Year: (2016)
Ref_id:b31 Title: GANs trained by a two time-scale update rule converge to a local Nash equilibrium Year: (2017)
Ref_id:b32 Title: A style-based generator architecture for generative adversarial networks Year: (2021)
Ref_id:b33 Title: Hyunjong Lee, and Joong-Ho Won. t 3 -variational autoencoder: Learning heavy-tailed data with student's t and power divergence Year: (2024-05)
Ref_id:b34 Title: Light Schrödinger bridge Year: (2024-05)
Ref_id:b35 Title: Neural Lagrangian Schrödinger bridge: Diffusion modeling for population dynamics Year: (2023-05)
Ref_id:b36 Title: A survey of the Schrödinger problem and some of its connections with optimal transport Year: (2014)
Ref_id:b37 Title: Flow matching for generative modeling Year: (2023-05)
Ref_id:b38 Title: Reachability and controllability analysis of the state covariance for linear stochastic systems Year: (2024)
Ref_id:b39 Title: Optimal covariance steering for discrete-time linear stochastic systems Year: (2025)
Ref_id:b40 Title: Deep generalized Schrödinger bridge Year: (2022)
Ref_id:b41 Title: Generalized Schrödinger bridge matching Year: (2024)
Ref_id:b42 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2023-05)
Ref_id:b43 Title: Entropy-regularized 2-Wasserstein distance between Gaussian measures Year: (2022)
Ref_id:b44 Title: Visualizing structure and transitions in high-dimensional biological data Year: (2019)
Ref_id:b45 Title: Mosek modeling cookbook Year: (2020)
Ref_id:b46 Title: Heavy-tailed diffusion models Year: (2025)
Ref_id:b47 Title: Matthieu Perrot, and Édouard Duchesnay. Scikit-learn: Machine learning in Python Year: (2011)
Ref_id:b48 Title: Non-denoising forward-time diffusions Year: (2021)
Ref_id:b49 Title: Diffusion bridge mixture transports, Schrödinger bridge problems and generative modeling Year: (2023)
Ref_id:b50 Title: Computational Optimal Transport: With Applications to Data Science. Foundations and Trends in Machine Learning Year: (2019)
Ref_id:b51 Title: Adversarial latent autoencoders Year: (2020-06)
Ref_id:b52 Title: Discrete-time optimal covariance steering via semidefinite programming Year: (2023)
Ref_id:b53 Title: Discrete-time maximum likelihood neural distribution steering Year: (2024)
Ref_id:b54 Title: An introduction to deep generative modeling Year: (2021)
Ref_id:b55 Title: A machine learning framework for solving high-dimensional mean field game and mean field control problems Year: (2020)
Ref_id:b56 Title: Optimal Transport for Applied Mathematicians Year: (2015)
Ref_id:b57 Title: Distributed hierarchical distribution control for very-large-scale clustered multi-agent systems Year: (2023-07)
Ref_id:b58 Title: Applied Stochastic Differential Equations Year: (2019)
Ref_id:b59 Title: Diffusion Schrödinger bridge matching Year: (2023)
Ref_id:b60 Title: Dynamic programming in probability spaces via optimal transport Year: (2024)
Ref_id:b61 Title: Learning diffusion at lightspeed Year: (2024)
Ref_id:b62 Title: Feedback Schrödinger bridge matching Year: (2025)
Ref_id:b63 Title: Trajecto-ryNet: A dynamic optimal transport network for modeling cellular dynamics Year: (2020-07)
Ref_id:b64 Title: fundamental algorithms for scientific computing in Python Year: (2020)
Ref_id:b65 Title: Measure and Integral Year: (1977)
