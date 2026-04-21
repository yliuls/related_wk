Title: Parallel Simulation for Log-concave Sampling and Score-based Diffusion Models
Abstract: Sampling from high-dimensional probability distributions is fundamental in machine learning and statistics. As datasets grow larger, computational efficiency becomes increasingly important, particularly in reducing adaptive complexity, namely the number of sequential rounds required for sampling algorithms. While recent works have introduced several parallelizable techniques, they often exhibit suboptimal convergence rates and remain significantly weaker than the latest lower bounds for log-concave sampling. To address this, we propose a novel parallel sampling method that improves adaptive complexity dependence on dimension d reducing it from O(log 2 d) to O(log d).Our approach builds on parallel simulation techniques from scientific computing.

Section: Introduction
We study the problem of sampling from a probability distribution with density π(x) ∝ exp(-f (x)) where f : R d → R is a smooth potential. We consider two types of setting. Problem (a): the distribution is known only up to a normalizing constant (Chewi, 2023), and this kind of problem is fundamental in many fields such as Bayesian inference, randomized algorithms, and machine learning (Robert et al., 1999;Marin et al., 2007;Nakajima et al., 2019). Problem (b): known as the score-based generative models (SGMs) (Song & Ermon, 2019), we are given an approximation of ∇ log π t , where π t is the density of a specific process at time t. The law of this process converges to π over time. SGMs are state-of-the-art in applications like image generation (Ho et al., 2022a;Dhariwal & Nichol, 2021), audio and video generation (Ho et al., 2022b;Yang et al., 2023a), and inverse problems (Song et al., 2022).
For Problem (a), specifically log-concave sampling, starting from the seminal papers of Dalalyan & Tsybakov (2012), Dalalyan (2017), and Durmus & Moulines (2017), there has been a flurry of recent works on proving nonasymptotic guarantees based on simulating a process which converges to π over time (Wibisono, 2018;Vempala & Wibisono, 2019;Mou et al., 2021;Altschuler & Talwar, 2023). Moreover, these processes, such as Langevin dynamics, converge exponentially quickly to π under mild conditions (Dalalyan, 2017;Mou et al., 2021;Bernard et al., 2022). Such dynamics-based algorithms for Problem (a) share a common feature with the inference process of SGMs that they are actually a numerical simulation of an initialvalue problem of differential equations (Hodgkinson et al., 2021). Thanks to the exponentially fast convergence of the process, significant efforts have been conducted on discretizing these processes using numerical methods such as the forward Euler, backward Euler (proximal method), exponential integrator, mid-point, and high-order Runge-Kutta methods (Vempala & Wibisono, 2019;Wibisono, 2019;Shen & Lee, 2019;Li et al., 2019;Oliva & Akyildiz, 2024). Furthermore, in recent years, there have been increasing interest and significant advances in understanding the convergence of inherently dynamics-based SGMs (Bortoli, 2022;Chen et al., 2023c;Lee et al., 2023;Pedrotti et al., 2024;Chen et al., 2023b;Tang & Zhao, 2024;Li & Yan, 2024). Notably, polynomial-time convergence guarantees have been established (Chen et al., 2023c;b;Benton et al., 2024;Liang et al., 2025), and various discretization schemes for SGMs have been analyzed (Lu et al., 2022a;b;Huang et al., 2025).
The algorithms underlying the above results are highly sequential. However, with the increasing size of data sets for sampling, we need to develop a theory for algorithms with limited iterations. For example, the widely-used denoising diffusion probabilistic models (Ho et al., 2020) may take 1000 denoising steps to generate one sample, while the evaluations of a neural network-based score function can be computationally expensive (Song et al., 2021).
As a comparison, recently, the (naturally parallelizable) Picard methods for diffusion models reduced the number of steps to around 50 (Shih et al., 2024). Furthermore, in terms of the dependency on the dimension d and accuracy ε, Picard methods for both Problems (a) and (b) were proven to be able to return an ε-accurate solution within O(log 2 (d/ε 2 )) iterations, improved from previous O(d a /ε b ) with some a, b > 0. However, the O(log 2 (d/ε 2 )) adaptive complexity 1 may not be yet optimal for both Problem (a) and (b). This motivates our investigation into the question:
Can we achieve logarithmic adaptive complexity for both log-concave sampling and sampling for SGMs?
this section cite: ['b14', 'b55', 'b46', 'b48', 'b60', 'b21', 'b61', 'b19', 'b18', 'b23', 'b65', 'b63', 'b47', 'b1', 'b18', 'b47', 'b8', 'b34', 'b63', 'b66', 'b57', 'b42', 'b51', 'b9', 'b13', 'b38', 'b54', 'b12', 'b62', 'b39', 'b13', 'b50', 'b7', 'b43', 'b50', 'b35', 'b31', 'b59', 'b58']

Section: Our Contributions
In this work, we propose a novel sampling method that employs a highly parallel discretization approach for continuous processes, with applications to the overdamped Langevin diffusion (Chewi, 2023) and the stochastic differential equation (SDE) implementation of processes in SGMs (Chen et al., 2024) for Problems (a) and (b), respectively.
Faster parallel log-concave sampling. We first present an improved result for parallel sampling from a strongly log-concave and log-smooth distribution. Specifically, we improve the upper bound from O log 2 d ε 2 (Anari et al., 2024) to O log d ε 2
, with slightly scaling the number of processors and gradient evaluations from
O d ε 2 to O d ε 2 log d ε 2 .
Compared with methods based on underdamped Langevin diffusion (Shen & Lee, 2019;Yu & Dalalyan, 2024;Anari et al., 2024), our method exhibits higher space complexity 2 . This is primarily because underdamped Langevin diffusion typically follows a smoother trajectory than overdamped Langevin diffusion, allowing for larger grid spacing and consequently, a reduced number of grids. We summarize the comparison in Table 1. In this paper, we will focus on the adaptive complexity and discretization schemes for overdamped Langevin diffusion.
Faster parallel sampling for diffusion models. We then present an improved result for diffusion models. Specifically, we propose an efficient algorithm with O log d ε 2
adaptive complexity for SDE implementations of diffusion models (Song & Ermon, 2019). Our method surpasses all the existing parallel methods for diffusion models having O log 2 d ε 2
adaptive complexity (Chen et al., 2024; 1 Adaptive complexity refers to the minimal number of sequential rounds required for an algorithm to achieve a desired accuracy, assuming polynomially many queries can be executed in parallel at each round (Balkanski & Singer, 2018). 2 We note, in this paper, that the space complexity refers to the number of words (Cohen-Addad et al., 2023;Chen et al., 2024) instead of the number of bits (Goldreich, 2008) to denote the approximate required storage.
O d 3/2 ε 2 (Gupta et al., 2025, Theorem B.13) ODE / Parallel midpoint method TV O log 2 d ε 2 O d 3/2 ε 2 (Chen et al., 2024, Theorem 3.3) SDE / Picard method KL O log 2 d ε 2 O d 2 ε 2 Theorem 5.4 SDE / Parallel Picard method KL O log d ε 2 O d 2 ε 2
Gupta et al., 2025), with slightly increasing the number of the processors and gradient evaluations and the space complexity for SDEs. We summarize the comparison in Table 2. Similarly, the better space complexity of the ordinary differential equation (ODE) implementations is attributed to the smoother trajectories of ODEs, which are more readily discretized.
this section cite: ['b14', 'b11', 'b3', 'b57', 'b69', 'b3', 'b60', 'b11', 'b6', 'b17', 'b11', 'b28']

Section: Problem Set-up
In this section, we introduce some preliminaries and key ingredients of log-concave sampling and diffusion models in Sections 2.1 and 2.2, respectively. Following this, Section 2.3 provides an introduction to the fundamentals of Picard iterations.
this section cite: []

Section: Log-concave Sampling
Problem (a) (Sampling task). Given the potential function f : D → R, the goal of the sampling task is to draw a sample from the density π f = Z -1 f exp(-f ), where Z f := D exp(-f (x))dx is the normalizing constant.
Distribution and function class. If f is (strongly) convex, the density π f is said to be (strongly) log-concave. If f is twice-differentiable and ∇ 2 f ⪯ βI (where ⪯ denotes the Loewner order and I is the identity matrix), we say the potential f is β-smooth and the density π f is β-log-smooth.
We define relative Fisher information of probability density ρ w.r.
t. π as FI(ρ∥π) = E ρ [∥∇ log(ρ/π)∥ 2 ] and the Kullback-Leibler (KL) divergence of ρ from π as KL(ρ∥π) = E ρ log(ρ/π). If π is α-strongly log-concave, then the following relation between KL divergence and Fisher informa-tion holds: KL(ρ∥π) ≤ 1 2α FI(ρ∥π) for all probability measures ρ.
Langevin Dynamics. One of the most commonly-used dynamics for sampling is Langevin dynamics (Chewi, 2023), which is the solution to the following SDE,
dx = -∇f (x)dt + √ 2dB t ,
where f ) satisfies strongly log-concavity, then the law of the Langevin diffusion converges exponentially fast to π (Bakry et al., 2014).
(B t ) t∈[0,T ] is a standard Brownian motion in R d . If π ∝ exp(-
this section cite: ['b14', 'b5']

Section: Score function for sampling task.
We assume the score function s : R d → R is a pointwise accurate estimate of ∇f , i.e., ∥s(x) -∇f (x)∥ ≤ δ for all x ∈ R d and some sufficiently small constant δ ∈ R + .
Measures of the output. For two densities ρ and π, we define the total variation (TV) as
TV(ρ, π) = sup{ρ(E) -π(E) | E is an event}.
We have the following relation between the KL divergence and TV distance, known as the Pinsker inequality,
TV(ρ, π) ≤ 1 2 KL(ρ∥π).
We denote by W 2 the Wasserstein distance between ρ and π, which is defined as
W 2 2 (ρ, π) = inf Π E (X,Y )∼Π ∥X -Y ∥ 2 ,
where the infimum is over coupling distributions of (X, Y ) such that X ∼ ρ, Y ∼ π. If π is α-strongly logconcave, the following transport-entropy inequality, known as Talagrand's T 2 inequality, holds (Otto & Villani, 2000) for all ρ ∈ P 2 (R d ), i.e., with finite second moment,
α 2 W 2 2 (ρ, π) ≤ KL(ρ∥π).
Complexity. For any sampling algorithm, we consider the adaptive complexity defined as unparallelizable evaluations of the score function (Chen et al., 2024), and use the notion of the space complexity to denote the approximate required storage during the inference. We note, in this paper, that the space complexity refers to the number of words (Cohen-Addad et al., 2023;Chen et al., 2024) instead of the number of bits (Goldreich, 2008) to denote the approximate required storage.
this section cite: ['b53', 'b11', 'b17', 'b11', 'b28']

Section: Score-based Diffusion Models
Sampling for diffusion models. In score-based diffusion models, one considers forward process (x t ) t∈[0,T ] in R d governed by the canonical Ornstein-Uhlenbeck (OU) process (Ledoux, 2000):
dx t = - 1 2 x t dt + dB t , x 0 ∼ q 0 , t ∈ [0, T ], (1
)
where q 0 is the initial distribution over R d . The corresponding backward process ( ⃗ x t ) t∈[0,T ] in R d follows an SDE defined as
d ⃗ x t = 1 2 ⃗ x t + ∇ log ⃗ p t ( ⃗ x t ) dt + dB t t ∈ [0, T ], ⃗ x 0 ∼ p 0 ≈ N (0 d , I d ) (2) where N (•, •) represents the normal distribution over R d . In practice, the score function ∇ log ⃗ p t ( ⃗ x t ) is estimated by neural network (NN) s θ t : R d → R d
, where θ is the parameters of NN. The backward process is approximated by
dy t = 1 2 y t + s θ t (y t ) dt + dB t t ∈ [0, T ], y 0 ∼ N (0 d , I d ).(3)
Problem (b) (Sampling task for SGMs). Given the learned NN-based score function s θ t , the goal is to simulate the approximated backward process such that the law of the output is close to q 0 .
this section cite: ['b37']

Section: Distribution class.
For SGMs, we assume the data density p 0 has finite second moments and is normalized such that
cov p0 (x 0 ) = E p0 (x 0 -E p0 [x 0 ])(x 0 -E p0 [x 0 ]) ⊤ = I d .
Such a finite moment assumption is standard across previous theoretical works on SGMs (Chen et al., 2023a;b;c) and we adopt the normalization to simplify true score function-related computations as Benton et al. (2024) and Chen et al. (2024) did.
OU process and inverse process The OU process and its inverse process also converge to the target distribution exponentially fast in various divergences and metrics such as the 2-Wasserstein metric W 2 ; see Ledoux (2000). Furthermore, the discrepancy between the terminal distributions of the backward process (2) and its approximation version (3) scales polynomially with respect to the length of the time horizon and the score matching error. (Huang et al. (2025, Theorem 3.5) or setting the step size h → 0 for the results in Chen et al. (2023a;b;c)).
Score function for SGMs. For the NN-based score, we assume the score function is L 2 -accurate, bounded and Lipschitz; we defer the details in Section 5.2.
this section cite: ['b50', 'b7', 'b11', 'b37', 'b50']

Section: Picard Iterations
Consider the integral form of the initial value problem,
x t = x 0 + t 0 f s (x s )ds + √ 2B t .
The main idea of Picard iterations (Clenshaw, 1957) is to approximate the difference over time slice [t n , t n+1 ] as
x tn+1 -x tn = tn+1 tn f s (x s )ds + √ 2(B tn+1 -B tn ) ≈ M i=1 w i f tn+τn,i (x tn+τn,i )ds + √ 2(B tn+1 -B tn ),
with a discrete grids of M collocation points as
x tn = x tn+τn,0 ≤ x tn+τn,1 ≤ x tn+τn,2 ≤ • • • ≤ x tn+τ n,M = x tn+1 .
We update the points in a wave-like fashion, which inherently allows for parallelization: for m ′ = 1, . . . , M ,
x p+1 tn+τn,m = x 0 + m-1 m ′ =1 w m ′ f tn+τ n,m ′ (x p tn+τ n,m ′ ) + √ 2(B tn+τn,m -B tn ).
Various collocation points have been proposed, including uniform points and Chebyshev points (Bai & Junkins, 2011).
In this paper, however, we focus exclusively on the simplest case of uniform points, and extension to other cases is future work. Picard iterations are known to converge exponentially fast and, under certain conditions, even factorially fast for ODEs and backward SDEs (Hutzenthaler et al., 2021).
this section cite: ['b16', 'b4', 'b36']

Section: Technical Overview
We adopt the time splitting for the time horizon used in the existing parallel methods (Shen & Lee, 2019;Gupta et al., 2025;Chen et al., 2024;Anari et al., 2024;Yu & Dalalyan, 2024). With same time grids, our algorithm, however, depart crucially from prior work in the design of parallelism across the time slices, and the modification for controlling the score estimation error. Below we summarize these algorithmic contributions and technical novelties.
Recap of existing parallel sampling methods. Existing works for parallel sampling apply the following generic discretization schemes (Shen & Lee, 2019;Gupta et al., 2025;Chen et al., 2024;Anari et al., 2024;Yu & Dalalyan, 2024). At a high level, these methods divide the time horizon into many large time slices and each slice is further subdivided into grids with a small enough step size. Instead of sequentially updating the grid points, they update all grids at the same time slice simultaneously using exponentially fast converging Picard iterations (Alexander, 1990;Chen et al., 2024;Anari et al., 2024), or randomized midpoint methods (Shen & Lee, 2019;Yu & Dalalyan, 2024;Gupta et al., 2025). With O(log d) Picard iterations for O(log d) time slices, the total adaptive complexity of their algorithms is O(log 2 d). However, while sequential updating of each time slice is not necessary for simulating the process, it remains unclear how to parallelize across time slices for sampling to obtain O(log d) time complexity.
this section cite: ['b57', 'b29', 'b11', 'b3', 'b69', 'b57', 'b29', 'b11', 'b3', 'b69', 'b0', 'b11', 'b3', 'b57', 'b69', 'b29']

Section: Algorithmic novelty: parallel methods across time slices.
Naïvely, if we directly update all the grids simultaneously, the Picard iterations will not converge when the total length is T = O(log d). Instead of updating all time slices together or updating the time slice sequentially, we update the time slices in a diagonal style as illustrated in Figure 1. For any j-th update at the n-th time slice (corresponding the rectangle in the n-th column from the left and the j-th row from the top in Figure 1), there will be two inputs: (a) the right boundary point of the previous time slice, which has been updated j times, and (b) the points on the girds of the same time slice that have been updated j -1 times. Then we perform P times Picard iterations with these inputs, where the hyperparameter P depends on the smoothness of the score function. This repetition is simple but essential for preventing the accumulation of score-matching errors, which could otherwise grow exponentially w.r.t. the length of the time horizon. The main difference compared to the existing Picard methods is that for a fixed time slice, the starting points in our method are updated gradually, whereas in existing methods, the starting points remain fixed once processed. Challenges for convergence. Similar to the arguments for sequential methods or parallel methods with sequentially updating the time slices, we use the standard techniques such as the interpolation method or Girsanov's theorem (Vempala & Wibisono, 2019;Oksendal, 2013;Chewi, 2023) to decompose the total error w.r.t. KL into four components: (i) convergence error of the continuous process, (ii) discretization error, (iii) parallelization error, and (iv) score estimation error (See Eq. ( 4), Lemma B.1, and Lemma C.4). For (i) the convergence error of the continuous processes, their exponential convergence rates allow this error to be effectively controlled by setting the total time length to O(log d ε 2 ), regardless of the specific discretization scheme. For (ii), the discretization error scales approximately as d • h M , where h denotes the time slice length, M the number of discretization points per slice, and h M the grid resolution (See Eq. ( 4), and Lemma C.5). Setting h M ≈ O(ε 2 /d) ensures the discretization error remains within O(ε 2 ). The technical challenges rise from controlling the remaining two errors, which we summarize below.
(iii) Parallelization error: the parallelization error primarily arises from updating with s(x j-1 ) instead of s(x j ) in the Picard iteration, in contrast to the sequential method, where j indexes the steps along the Picard direction. In existing parallel methods, the sequential update across time slices benefits the convergence of truncation errors, E x jx j-1 2 . Assuming the truncation errors in the previous time slice have converged, its right boundary serves as the starting point for all grids in the current O(1)-length time slice which results in an initial bias of O(d). Subsequently, by performing O(log d) exponentially fast Picard iterations, the truncation error will converge. However, in our diagonal-style updating scheme across time, the truncation error interacts with inputs from both the previous time slice and prior updates in the same time slice. Consequently, the bias-convergence loop that holds in sequential updating no longer holds.
(iv) Score estimation error: If the score function itself is Lipschitz continuous (Assumption 5.3 for Problem (b)), no additional score matching error will arise during the Picard iterations. This allows the total score estimation error to remain bounded under mild conditions (Assumption 5.1). However, for Problem (a), since it is the velocity field ∇f instead of the score function s that is Lipschitz, additional score estimation errors will occur during each update. For the sequential algorithm, these additional score estimation errors are contained within the bias-convergence loop, ensuring the total score estimation error remains to be bounded. Conversely, for our diagonal-style updating algorithm, the absence of convergence along the time direction causes these additional score estimation errors to accumulate exponentially over the time direction.
Technical novelty. Our technical contributions address these challenges by the appropriate selection of the number of Picard iterations within each update P and the depth of the Picard iterations J. We outline the details of the choices below.
In the following, we assume that the truncation error at the n-th time slice and the j-th iteration scales with L j n , and that the additional score estimation error for each update scales with δ 2 .
To address the initial challenge related to the truncation error, we choose the Picard depth as J = O(N + log d). We first bound the error of the output for each update with respect to its inputs as L j n ≤ aL j n-1 + bL j-1 n , where a and b are constants. By carefully choosing the length of the time slices, we can ensure that b < 1 along the Picard iteration direction. Consequently, the truncation error will converge if the iteration depth J is sufficiently large, such that a N b J is sufficiently small. This requirement implies that J = O(N + log d).
To mitigate the additional score estimation error for Problem (a), we perform P Picard iterations within each update. The interaction between the truncation error and additional score estimation error can be expressed as L j n ≤ aL j n-1 + bL j-1 n + cδ 2 , where a, b, c are constants. To ensure the total score estimation error remains bounded, it is necessary to have a, b < 1, which guarantees convergence along both the time and Picard directions. By the convergence of the Picard iteration, we can achieve b < 1. For a, the right boundary point of the previous time slice, and prior updates within the same time slice introduce discrepancies in the truncation error. For the impact from the previous time slice, we make use of the contraction of gradient decent to ensure convergence. However, since the grid gap scale as 1/d, the contraction factor is close to 1. Consequently, we have to minimize the impact from prior updates within the same time slice, which scales as O(1) by repeating P = log O(1) Picard iterations for each update.
Balance between time and Picard directions. We note that the Picard method, despite being the simplest approach for time parallelism, has achieved the state of art performance in certain specific settings. On the one hand, the continuous processes need to run for at least O(log d) time. To ensure convergence within every time slice, the time slice length have to be set as O(1), resulting in a necessity for at least O(log d) iterations. On the other hand, with a proper initialization O(d), Picard iterations converge within O(log d) iterations. Our parallelization balances the convergence of the continuous diffusion and the Picard iterations to achieve the improved results.
Related works in scientific computation. Similar parallelism across time slices has also been proposed in scientific computation (Gear, 1991;Gander, 2015;Ong & Schroder, 2020), especially for parallel Picard iterations (Wang, 2023). Compared with prior work in scientific computation, our approach exhibits several significant differences. Firstly, our primary objective differs from that in simulation. In sampling, we aim to ensure that the output distribution closely approximates the target distribution, whereas simulation seeks to make some points on the discrete grid closely match the true dynamics. Second, our algorithm differs sig-nificantly from that of Wang (2023). In our algorithm, each update takes the inputs without the corrector operation. Furthermore, we perform P Picard iterations in each update to prevent error accumulation over time T = O(log d). However, these two fields are connected through the sampling strategies that ensure each discrete point closely approximates the true process at every sampling step.
this section cite: ['b63', 'b50', 'b14', 'b26', 'b25', 'b52', 'b64', 'b64']

Section: Parallel Picard Method for Strongly Log-concave Sampling
In this section, we present parallel Picard methods for strongly log-concave sampling (Algorithm 1) and show it holds improved convergence rate w.r.t. the KL divergence and total variance (Theorem 4.2 and Corollary 4.3). We illustrate the algorithm in Section 4.1, and give a proof sketch in Section 4.3. All the missing proofs can be found in Appendix B.
this section cite: []

Section: Algorithm
Our parallel Picard method for strongly log-concave sampling is summarized in Algorithm 1. In Lines 2-7, we generate the noises and initialize the value at the grid via Langevin Monte Carlo (Chewi, 2023) with a stepsize h = O(1). In Lines 8-26, the time slices are updated in a diagonal manner within the outer loop, as illustrated in Figure 1. In Lines 12-14 and Lines 21-23, we repeat P Picard iterations for each update to ensure convergence. Remark 4.1. Parallelization should be understood as evaluating the score function concurrently, with each time slice potentially being computed in an asynchronous parallel manner, resulting in the overall P (N + J) + N adaptive complexity.
this section cite: ['b14']

Section: Theoretical Guarantees
The following theorem summarizes our theoretical analysis for Algorithm 1. Theorem 4.2. Suppose π is α-strongly log-concave and β-log-smooth, and the score function s is δ-accurate. Let κ = β/α. Suppose
βh = 0.1, M ≥ κd ε 2 , N ≥ 10κ log KL(µ 0 ∥π) ε 2 , δ ≤ 0.2 √ αε, P ≥ 2 log κ 3 + 4, and J -N ≥ log N 3 κδ 2 h + κKL(µ 0 ∥π) + κ 2 d ε 2 .
then Algorithm 1 runs within N + (N + J)P iterations with at most M N queries per iteration and outputs a sample with marginal distribution ρ such that
max √ α 2 W 2 (ρ, π), TV(ρ, π) ≤ KL(ρ, π) 2 ≤ 2ε.
Algorithm 1 Parallel Picard Method for sampling 1: Input: x 0 ∼ µ 0 , approximate score function s ≈ ∇f , the number of the iterations in outer loop J, the number of the iteration in inner loop P , the number of time slices N , the length of time slices h, the number of points on each time slices M . 2: for n = 0, . . . , N -1 do 3:
for m = 0, . . . , M (in parallel) do 4: B nh+m/M h = B nh + N (0, (mh/M )I d ), 5: x j
-1,M = x 0 , for j = 0, . . . , J, 6:
x 0 n,m = x 0 n-1,M -hm M s(x 0 n-1,M ) + √ 2(B nh+mh/M -B nh ), 7:
end for 8: end for 9: for k = 1, . . . , N do 10:
for j = 1, . . . , min{k -1, J} and m = 1, . . . , M (in parallel) do 11: let n = k-j, x j n,0 = x j n-1,M
, and x j,0 n,m = x j-1 n,m , 12:
for p = 1, . . . , P do 13:
x j,p n,m = x j n,0 -h M m-1 m ′ =0 s(x j,p-1 n,m ′ ) + √ 2(B nh+mh/M -B nh ),14:
end for 15:
x j n,m = x j,P n,m , 16:
end for 17: end for 18: for k = N + 1, . . . , N + J -1 do 19:
for n = max{0, k -J}, . . . , N -1 and m = 1, . . . , M (in parallel) do 20: let j = k-n, x j n,0 = x j n-1,M , and x j,0 n,m = x j-1 n,m , 21:
for p = 1, . . . , P do 22:
x j,p n,m = x j n,0 -h M m-1 m ′ =0 s(x j,p-1 n,m ′ ) + √ 2(B nh+mh/M -B nh ),23:
end for 24:
x j n,m = x j,P n,m , 25:
end for 26: end for 27: Return:
x J N -1,M .
To make the guarantee more explicit, we can combine it with the following well-known initialization bound, see, e.g., Dwivedi et al. (2019
this section cite: []

Section: , Section 3.2).
Corollary 4.3. Suppose that π = exp(-f ) is α-strongly log-concave and β-log-smooth, and let κ = β/α. Let x ⋆ be the minimizer of f . Then, for
µ 0 = N (x ⋆ , β -1 ), it holds that KL(µ 0 ∥π) ≤ d 2 log κ. Consequently, setting h = 1 10β , M = κd ε 2 , N = 10κ log d log κ ε 2 , δ ≤ 0.2 √ αε, P ≥ 2 log κ 3 + 4,
and J -N = O log κ 2 d log κ ε 2 , then Algorithm 1 runs within N + (N + J)P = O(κ log d ε 2 ) iterations with at most M N = O( κ 2 d ε 2 log d ε 2 ) queries per iteration and outputs a sample with marginal distribution ρ such that
max √ α 2 W 2 (ρ, π), TV(ρ, π) ≤ KL(ρ, π) 2 ≤ 2ε.
Remark 4.4. The main drawback of our method is the suboptimal space complexity due to its application to overdamped Langevin diffusion which has a less smooth trajectory compared to underdamped Langevin diffusion. However, we anticipate that our method could achieve comparable space complexity when adapted to underdamped Langevin diffusion. Remark 4.5. Regarding the condition number κ, our method achieves the same adaptive complexity of O(κ) as both the state-of-the-art sequential method and existing parallel approaches (Anari et al., 2024;Yu & Dalalyan, 2024;Altschuler & Chewi, 2024). Whether parallelization can improve the dependence on the condition number remains an open question, which we leave for future work. Remark 4.6. When the number of computation cores, denoted by ℓ, is limited, the adaptive complexity of our algorithm is
O κ 2 d ε 2 ℓ log 2 d ε 2 .
this section cite: ['b3', 'b69', 'b2']

Section: Proof Sketch of Theorem 4.2: Performance Analysis of Algorithm 1
The detailed proof of Theorem 4.2 is deferred to Appendix B. As discussed in Section 3, by interpolation methods (Anari et al., 2024), we decompose the error w.r.t. the KL divergence into four error components (corollary B.4):
KL(ρ∥π) ≲ e -Θ(N ) KL(µ 0 ∥π) Convergence of Langevin dynamics + dh M discretization error + N -1 n=1 e -Θ(n) E J N -n parallization error + δ 2 score estimation error ,(4)
where E j n represents the truncation error of the grids at n-th time slice after j update. For the right terms, with the choice of N = O(log d ε 2 ), M = O(dh/ε 2 ) and δ ≤ ε, which ensures a sufficiently long time horizon the sufficiently long time horizon T = N h = O(log d ε 2 ), densely spaced grids with a gap h/M = O(ε 2 /d) and a small score matching error, respectively, we can conclude that
e -Θ(N ) KL(µ 0 ∥π) + dh M + δ 2 ≲ ε 2
Thus, we will focus on proving the convergence of the truncation error E j n in the Picard iterations, and avoiding the additional accumulation of the score estimation error during Picard iterations as discussed before.
Considering that the truncation error expands at most exponentially along the time direction, but diminishes exponentially with an increased depth of the Picard iterations, convergence can be achieved by ensuring that the depth of the Picard iterations surpasses the number of time slices as J ≥ N + O(log d ε 2 ) with initialization error bounded by O(d) (the second part of Corollary B.7 and second part of Corollary B.9).
Due to the non-Lipschitzness of the score function, we can only bound E j n by quantity a∆ j n-1 +bE j-1 n +cδ 2 h 2 (Lemma B.5 and Lemma B.8), where ∆ j n-1 represents the truncation error at the right boundary of the previous time slice.
Here, the coefficients are given by:
a = 1 -0.1 βh κ + O(κ)(3β 2 h 2 ) P , b = O((β 2 h 2 ) P ) and c = O(κδ 2 h 2 ).
Intuitively, a comes from the contraction of the gradient mapping with an additional term from the Picard direction, b reflects convergence along the Picard direction, and c accounts for the accumulation of score estimation error δ over time length h, with an additional scaling by κ due to Young's inequality. To control the growth of the score error, it is essential that the coefficients a and b remain strictly less than one. Setting P = Θ(log κ) is sufficient to ensure this condition.
this section cite: ['b3']

Section: Parallel Picard Method for Sampling of Diffusion Models
In this section, we present parallel Picard methods for diffusion models in Section 5.1 and assumptions in Section 5.2. Then we show it holds improved convergence rate w.r.t. the KL divergence (Theorem 5.4). All the missing details can be found in Appendix C.
this section cite: []

Section: Algorithm
Due to space limitations, the detailed methodology for the parallelization of Picard methods for diffusion models is provided in Appendix C.1 and Algorithm 2. It keeps same parallel structure as that illustrated in Figure 1. Notably, it exhibits the following distinctions in comparison to the parallel Picard methods for strongly log-concave sampling presented in Algorithm 1:
• Since the score function itself is Lipschitz, there will not be additional score matching error during Picard iterations.
As a result, we perform single Picard iteration in one update, i.e., P = 1; • Instead of uniform discrete grids, we employ a shrinking step size discretization scheme towards the data end, and the early stopping technique which is unvoidable to show the convergence for diffusion models (Chen et al., 2024). We show the details in Appendix C.1; • We use an exponential integrator instead of the Euler-Maruyama Integrator in Picard iterations, where an additional high-order discretization error term would emerge (Chen et al., 2023a), which we believe would not affect the overall O(log d) adaptive complexity with parallel sampling.
this section cite: ['b11']

Section: Assumptions
Our theoretical analysis of the algorithm assumes mild conditions regarding the data distribution's regularity and the approximation properties of NNs as discussed in Chen et al. (2024). These assumptions align with those established in previous theoretical works, such as those described by Chen et al. (2023c;a;b;2024).
Assumption 5.1 ((L 2 ([0, t N ]) δ 2 -accurate learned score). The learned NN-based score s θ t is δ 2 -accurate in the sense of E ⃗ p N -1 n=0 Mn-1 m=0 ϵ n,m s θ tn+τn,m ( ⃗ x tn+τn,m ) -∇ log ⃗ p tn+τn,m ( ⃗ x tn+τn,m ) 2 ≤ δ 2 2 .
Assumption 5.2 (Regular and normalized data distribution). The data density p 0 has finite second moments and is normalized such that cov p0 (x 0 ) = I d . Assumption 5.3 (Bounded and Lipschitz learned NN-based score). The learned NN-based score function s θ t has a bounded C 1 norm, i.e. , s θ t (•) L ∞ ([0,T ]) ≤ M s with Lipschitz constant L s .
this section cite: ['b11', 'b13', 'b50']

Section: Theoretical Guarantees
Theorem 5.4. Under Assumptions 5.1, 5.2, and 5.3, given the following choices of the order of the parameters
h = Θ(1), N = O log d ε 2 , M = O d ε 2 log d ε 2 , T = O log d ε 2
, δ ≤ ε, and J = O N +log N d ε 2 the parallel Picard algorithm for diffusion models (Algorithm 2) generates samples from satisfies the following error bound,
KL(p η ∥ q t N ) ≲ de -T + dT M + ε 2 + δ 2 2 ≲ ε 2 ,(5)
with total 2N + J = O log d ε 2
adaptive complexity and dM = O d 2 ε 2 space complexity for parallelizable δ 2accurate score function computations. Remark 5.5. Compared to existing parallel methods, our method improves the adaptive complexity from O(log 2 d ε 2 ) to O(log d ε 2 ). Its main drawback is suboptimal space complexity due to the less smooth trajectory in SDE implementations, but we believe it can achieve comparable space complexity when adapted to ODE implementations. Remark 5.6. When the number of computation cores ℓ is limited, the adaptive complexity is O κ 2 d ε 2 ℓ log 2 d ε 2 , matching the state-of-the-art result of Chen et al. (2024) for ℓ ≤ κd ε 2 . However, employing a large batch size ℓ ≥ κd ε 2 in diffusion models may not always yield substantial benefits (Shih et al., 2024;Li et al., 2024b;a). Remark 5.7. We note that the uniformly Lipschitz assumption (Assumption 5.3) may be too strong. In particular, the required Lipschitz constant can become quite large even becoming unbounded near the zero point (Salmona et al., 2022;Yang et al., 2023b). In this case, to ensure convergence of the Picard iterations under these conditions, the quantity L 2 s e hn h n must be sufficiently small. This requirement implies that the length of each time slice, h n , should scale as O(1/L 2 s ). Consequently, the number of time slices becomes N = O(L 2 s log d), leading to an overall iteration complexity of N = O(L 2 s log d). We also believe our algorithm, which is based on a diagonal-style update, is robust to this assumption by adaptively adjusting the length of the time slices.
this section cite: ['b11', 'b58', 'b56']

Section: Discussion and Conclusion
In this work, we proposed novel parallel Picard methods for various sampling tasks. Notably, we obtain ε 2 -accurate sample w.r.t. the KL divergence within O log d ε 2 , which represents a significant improvement from O log 2 d ε 2 for diffusion models. Furthermore compared with the existing methods applied to the overdamped Langevin dynamics or the SDE implementations for diffusion models, our space complexity only scales by a logarithmic factor.
Our study opens several promising theoretical directions. First, as an analogue to simulation methods in scientific computing, it highlights the potential of leveraging alternative discretization techniques for faster and more efficient sampling. Another direction is exploring smoother dynamics to reduce space complexity in these methods.
Lastly, although our highly parallel methods may introduce engineering challenges, such as the memory bandwidth, we believe our theoretical works will motivates the empirical development of parallel algorithms for both sampling and diffusion models.
this section cite: []

Section: References
Ref_id:b0 Title: Solving Ordinary Differential Equations i: Nonstiff Problems Year: (1990)
Ref_id:b1 Title: Resolving the Mixing Time of the Langevin Algorithm to its Stationary Distribution for Log-Concave Sampling Year: (2023)
Ref_id:b2 Title: Shifted Composition III: Local Error Framework for KL Divergence Year: (2024)
Ref_id:b3 Title: Fast parallel sampling under isoperimetry Year: (2024)
Ref_id:b4 Title: Modified Chebyshev-Picard iteration methods for orbit propagation Year: (2011)
Ref_id:b5 Title: Analysis and geometry of Markov diffusion operators Year: (2014)
Ref_id:b6 Title: The adaptive complexity of maximizing a submodular function Year: (2018)
Ref_id:b7 Title: Nearly d-Linear Convergence Bounds for Diffusion Models via Stochastic Localization Year: (2024)
Ref_id:b8 Title: Hypocoercivity with schur complements Year: (2022)
Ref_id:b9 Title: Convergence of denoising diffusion models under the manifold hypothesis Year: (2022)
Ref_id:b10 Title: Improved Analysis of Scorebased Generative Modeling: User-Friendly Bounds under Minimal Smoothness Assumptions Year: (2023-07)
Ref_id:b11 Title: Accelerating Diffusion Models with Parallel Sampling: Inference at Sub-Linear Time Complexity Year: (2024)
Ref_id:b12 Title: The probability flow ODE is provably fast Year: (2023)
Ref_id:b13 Title: Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions Year: (2023)
Ref_id:b14 Title: Log-concave sampling Year: (2023)
Ref_id:b15 Title: Analysis of Langevin Monte Carlo from Poincaré to Log-Sobolev Year: (2024)
Ref_id:b16 Title: The numerical solution of linear differential equations in Chebyshev series Year: (1957)
Ref_id:b17 Title: Streaming Euclidean k-median and k-means with o(log n) Space Year: (2023)
Ref_id:b18 Title: Theoretical guarantees for approximate sampling from smooth and log-concave densities Year: (2017)
Ref_id:b19 Title: Sparse regression learning by aggregation and Langevin Monte-Carlo Year: (2012)
Ref_id:b20 Title: Parallel MCMC without embarrassing failures Year: (2022)
Ref_id:b21 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b22 Title: Some Gronwall type inequalities and applications Year: (2003)
Ref_id:b23 Title: Nonasymptotic convergence analysis for the unadjusted Langevin algorithm Year: (2017)
Ref_id:b24 Title: Logconcave sampling: Metropolis-Hastings algorithms are fast Year: (2019)
Ref_id:b25 Title: 50 years of time parallel time integration Year: (2013)
Ref_id:b26 Title: Waveform methods for space and time parallelism Year: (1991)
Ref_id:b27 Title: Parallel MCMC algorithms: theoretical foundations, algorithm design, case studies Year: (2024)
Ref_id:b28 Title: Computational complexity: a conceptual perspective Year: (2008)
Ref_id:b29 Title: Faster Diffusion-based Sampling with Randomized Midpoints: Sequential and Parallel. International Conference on Learning Representations Year: (2025)
Ref_id:b30 Title: Parallelizing MCMC sampling via space partitioning Year: (2022)
Ref_id:b31 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b32 Title: Cascaded diffusion models for high fidelity image generation Year: (2022)
Ref_id:b33 Title: Video diffusion models Year: (2022)
Ref_id:b34 Title: Implicit Langevin algorithms for sampling from log-concave densities Year: (2021)
Ref_id:b35 Title: Convergence Analysis of Probability Flow ODE for Score-Based Generative Models Year: (2025)
Ref_id:b36 Title: On the speed of convergence of Picard iterations of backward stochastic differential equations Year: (2021)
Ref_id:b37 Title: The geometry of markov diffusion generators Year: (2000)
Ref_id:b38 Title: Convergence of score-based generative modeling for general data distributions Year: (2023)
Ref_id:b39 Title: Adapting to unknown low-dimensional structures in score-based diffusion models Year: (2024)
Ref_id:b40 Title: Distrifusion: Distributed parallel inference for high-resolution diffusion models Year: (2024)
Ref_id:b41 Title: Efficient Diffusion Model Serving with Add-on Modules Year: (2024)
Ref_id:b42 Title: Stochastic Runge-Kutta Accelerates Langevin Monte Carlo and Beyond Year: (2019)
Ref_id:b43 Title: Broadening Target Distributions for Accelerated Diffusion Models via a Novel Analysis Approach Year: (2025)
Ref_id:b44 Title: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps Year: (2022)
Ref_id:b45 Title: Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models Year: (2022)
Ref_id:b46 Title: Bayesian Core: A Practical Approach to Computational Bayesian statistics Year: (2007)
Ref_id:b47 Title: High-order Langevin diffusion yields an accelerated MCMC algorithm Year: (2021)
Ref_id:b48 Title: Variational Bayesian learning theory Year: (2019)
Ref_id:b49 Title: Parallel MCMC with generalized elliptical slice sampling Year: (2014)
Ref_id:b50 Title: Stochastic differential equations: an introduction with applications Year: (2013)
Ref_id:b51 Title: Kinetic Interacting Particle Langevin Monte Carlo Year: (2024)
Ref_id:b52 Title: Applications of time parallelization. Computing and Visualization in Science Year: (2020)
Ref_id:b53 Title: Generalization of an inequality by Talagrand and links with the logarithmic Sobolev inequality Year: (2000)
Ref_id:b54 Title: Improved Convergence of Score-Based Diffusion Models via Prediction-Correction Year: (2024)
Ref_id:b55 Title: Monte Carlo statistical methods Year: (1999)
Ref_id:b56 Title: Can push-forward generative models fit multimodal distributions? Year: (2022)
Ref_id:b57 Title: The Randomized Midpoint Method for Log-Concave Sampling Year: (2019)
Ref_id:b58 Title: Parallel sampling of diffusion models Year: (2024)
Ref_id:b59 Title: Denoising diffusion implicit models Year: (2021)
Ref_id:b60 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b61 Title: Solving inverse problems in medical imaging with score-based generative models Year: (2022)
Ref_id:b62 Title: Contractive diffusion probabilistic models Year: (2024)
Ref_id:b63 Title: Rapid convergence of the unadjusted langevin algorithm: Isoperimetry suffices Year: (2019)
Ref_id:b64 Title: Parallel Numerical Picard Iteration Methods Year: (2023)
Ref_id:b65 Title: Sampling as optimization in the space of measures: The Langevin dynamics as a composite optimization problem Year: (2018)
Ref_id:b66 Title: Proximal Langevin algorithm: Rapid convergence under isoperimetry Year: (2019)
Ref_id:b67 Title: Diffusion probabilistic modeling for video generation Year: (2023)
Ref_id:b68 Title: Lipschitz singularities in diffusion models Year: (2023)
Ref_id:b69 Title: Parallelized Midpoint Randomization for Langevin Monte Carlo Year: (2024)
Ref_id:b70 Title: Fast sampling of diffusion models with exponential integrator Year: (2023)
