Title: A Unified Framework for Entropy Search and Expected Improvement in Bayesian Optimization
Abstract: Bayesian optimization is a widely used method for optimizing expensive black-box functions, with Expected Improvement being one of the most commonly used acquisition functions. In contrast, information-theoretic acquisition functions aim to reduce uncertainty about the function's optimum and are often considered fundamentally distinct from EI. In this work, we challenge this prevailing perspective by introducing a unified theoretical framework, Variational Entropy Search, which reveals that EI and information-theoretic acquisition functions are more closely related than previously recognized. We demonstrate that EI can be interpreted as a variational inference approximation of the popular information-theoretic acquisition function, named Max-value Entropy Search. Building on this insight, we propose VES-Gamma, a novel acquisition function that balances the strengths of EI and MES. Extensive empirical evaluations across both low-and high-dimensional synthetic and real-world benchmarks demonstrate that VES-Gamma is competitive with state-ofthe-art acquisition functions and in many cases outperforms EI and MES.

Section: Introduction
Bayesian optimization (BO) is a widely used technique for maximizing black-box functions. Given a function f : X → R, BO iteratively refines a probabilistic surrogate of f , typically a Gaussian process (GP), and selects the next evaluation point accordingly. At each iteration, the next sampling point is determined by maximizing an acqui-Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). sition function (AF) α : X → R. An effective AF must balance the exploration-exploitation trade-off, where exploitation prioritizes sampling points predicted by the surrogate to yield high objective values, while exploration targets regions with the potential to uncover even better values.
Expected Improvement (EI) (Mockus, 1998) is one of the most widely used AFs, valued for its simple formulation, computational efficiency, and strong empirical performance. The core idea behind EI is to maximize the expected improvement over the current best observed value, which typically requires a noise-free assumption. More recently, (Villemonteix et al., 2009;Hennig & Schuler, 2012) have introduced the concepts of information-theoretic AFs, which represents a paradigm shift in Bayesian optimization. Unlike EI, which focuses on directly maximizing potential improvement, information-theoretic AFs aim to reduce uncertainty about the function f 's optimal position and/or value, often through entropy-based measures. Due to their fundamentally different underlying philosophies and selection criteria, EI and information-theoretic AFs are widely regarded as distinct methodologies within the BO community (Hennig et al., 2022).
Despite their apparent differences, we argue that EI and information-theoretic AFs share deeper theoretical connections than previously recognized. Understanding this relationship is crucial, as it provides novel insights into designing new acquisition functions. By unifying the perspectives of both sides, we introduce VES-Gamma, a new AF that effectively balances their strengths, resulting in a robust AF that adapts well to diverse optimization problems. VES-Gamma inherits the performance of EI while incorporating information-theoretic considerations.
In summary, we make the following key contributions:
1. We introduce the Variational Entropy Search (VES) framework which shows that EI can be interpreted as a special case of the popular information-theoretic acquisition function Max-value Entropy Search (MES). This unified theoretical perspective reveals that these two types of AFs are more closely related than previously recognized.
2. We propose VES-Gamma as an intermediary between EI and MES, incorporating information-theoretic principles while maintaining EI's strength in performance.
3. We provide an extensive evaluation across a diverse set of low-and high-dimensional synthetic, GP samples, and real-world benchmarks, demonstrating that VES-Gamma consistently performs competitively and, in many cases, outperforms both EI and MES.
this section cite: ['b24', 'b39', 'b11', 'b12']

Section: Background and Related Work

this section cite: []

Section: Gaussian Processes
A Gaussian process is a stochastic process that models an unknown function. It is characterized by the property that any finite set of function evaluations follows a multivariate Gaussian distribution. Assuming that f has a zero mean, a Gaussian process is uniquely determined by the current observations D t := {(x i , y xi )} t i=1 and the kernel function κ(x, x ′ ). Given these, at stage t, the predicted mean of y
x at a new point x is µ t (x) = κ t (x) T (K t ) -1 y t ,
and the predicted covariance between points x and Rasmussen et al. (2006) for more details.
x ′ is Cov t (x, x ′ ) = κ(x, x ′ ) -κ t (x) T (K t ) -1 κ t (x ′ ), where [κ t (x)] i = κ(x i , x), [y t ] i = y xi , and [K t ] i,j = κ(x i , x j ); see
this section cite: ['b30']

Section: Acquisition Functions
Various acquisition functions (AFs) have been proposed to balance exploration and exploitation in optimization tasks, each tailored to different problem characteristics and assumptions. These include Probability of Improvement (PI), Expected Improvement (EI) (Mockus, 1998;Jones et al., 1998), Upper Confidence Bound (UCB) (Srinivas et al., 2010), Knowledge Gradient (KG) (Frazier et al., 2008), and information-theoretic AFs (Villemonteix et al., 2009;Hennig & Schuler, 2012;Hernández-Lobato et al., 2014;Wang & Jegelka, 2017;Hvarfner et al., 2022;Tu et al., 2022). Below, we discuss two types of acquisition functions relevant to this study.
Expected Improvement. Expected Improvement (EI) is one of the most commonly used acquisition functions and is formulated as follows:
α EI (x) = E p(yx|Dt) max{y x , y * t } -y * t ,(1)
where y * t is the maximum observed value in D t , and E p(•) denotes the expectation with respect to the predictive density p(•). The -y * t term at the end can be dropped since it is constant with respect to x.
Information-Theoretic AFs. Information-theoretic AFs form a family of methods designed to select x such that its evaluation reduces uncertainty regarding the optimal points of the objective function. This uncertainty is quantified using differential entropy, defined as H[y] := E p(y) [-log p(y)]. Similarly, the conditional entropy is expressed as H[y|x] := H[x, y] -H[x].
The first information-theoretic AF for BO is Entropy Search (ES) (Hennig & Schuler, 2012), which is formulated as:
α ES (x) = H[x * | D t ] -E p(yx|Dt) [H[x * | D t , y x ]] . (2)
Here, the random variable x * represents the location of the maximum.
Predictive Entropy Search (PES) (Hernández-Lobato et al., 2014) offers a reformulation of ES that is computationally more efficient:
α PES (x) = H[y x | D t ] -E p(x * |Dt) [H[y x | D t , x * ]] . (3)
Since directly estimating the entropy with x * is expensive, following the PES format, Max-value Entropy Search (MES) (Wang & Jegelka, 2017) introduced an alternative approach that focuses on reducing the differential entropy of the 1D maximum value y * :
α MES (x) = H[y * | D t ] -E p(yx|Dt) [H[y * | D t , y x ]] = H[y x | D t ] closed-form -E p(y * |Dt) [H[y x | D t , y * ]] non-closed-form . (4)
Unlike MES and its subsequent extensions (Hvarfner et al., 2022;Takeno et al., 2022) which approximate p(y x | D t , y * ) using a truncated Gaussian, we focus on directly estimating p(y * | D t , y x ) via variational inference.
this section cite: ['b24', 'b19', 'b34', 'b9', 'b39', 'b11', 'b13', 'b40', 'b16', 'b38', 'b11', 'b13', 'b40', 'b16', 'b37']

Section: Related Work Variational Inference and Evidence Lower Bound.
Variational Inference (VI) is a widely used technique in Bayesian modeling to approximate intractable posterior distributions (Paisley et al., 2012;Hoffman et al., 2013;Kingma & Welling, 2014). It relies on maximizing the Evidence Lower Bound (ELBO) to approximate the loglikelihood log p( x) in the presence of latent variables z. The log-likelihood can be decomposed as follows:
log p( x) ≥ E q(z) log p( x | z)p(z) q(z) ,(5)
where p(z) is a fixed prior distribution, and q(z) is a variational approximation to the true posterior p(z | x). The ELBO is formally defined as:
ELBO(p( x | z); q(z)) := E q(z) log p( x | z)p(z) q(z) .(6)
Figure 1. MES aims to optimize x such that the entropy (averaged over all yx) of the maximum values p(y * | Dt, yx) is reduced. The left figure illustrates a noiseless Gaussian process conditioned on the observations Dt with three points (black crosses) and a sample yx at x = 1 drawn from p(yx | Dt) (red star). The mid and right panels illustrate the density p(y * | Dt, yx) (blue curves). When p(y * | Dt, yx) is approximated using an exponential distribution (green dashed curve), this leads to the VES-Exp AF that is equivalent to EI. Furthermore, VES-Gamma, which approximates p(y * | Dt, yx) using a Gamma distribution (red dash-dot curve), leads to a more accurate approximation and a generalized version of EI.
By maximizing the ELBO, VI indirectly maximizes the log-likelihood log p( x), thereby improving the quality of the posterior approximation. In many applications, such as variational autoencoders (VAEs) (Kingma & Welling, 2014) and variational diffusion (Kingma et al., 2021), both the conditional likelihood p( x | z) and the variational distribution q(z) are parameterized using neural networks. Since both the expectation reference probability and the term inside the ELBO are parameterized, one common strategy is to estimate the gradient using finite Monte Carlo samples and the reparameterization trick to optimize the parameters. We adopt this approach, which enables efficient gradient-based optimization and has been widely applied in the BO community (Wilson et al., 2017).
Improving the Expected Improvement. It is widely recognized that EI can be prone to over-exploitation (Qin et al., 2017;Berk et al., 2019;De Ath et al., 2021).
To mitigate this issue, Hoffman et al. (2011) and Kandasamy et al. (2020) propose to use a portfolio of AFs, which assigns probabilities to different AFs at each step. Snoek et al. (2012) proposed a fully-Bayesian treatment on EI to improve empirical performance. Another approach is Weighted EI (WEI), which adaptively adjusts the weights of the components within the EI acquisition function (Sóbester et al., 2005;Benjamins et al., 2023). Similarly, Qin et al. (2017) suggest "weakening" EI using suboptimal points suggested by the AF to mitigate its overexploitative behavior. However, these methods are primarily based on heuristics. Furthermore, information-theoretic acquisition functions are often excluded from these design enhancements, as they are generally considered distinct from heuristic AFs such as PI, EI, UCB, or KG.
this section cite: ['b25', 'b15', 'b22', 'b22', 'b21', 'b42', 'b28', 'b5', 'b7', 'b14', 'b20', 'b32', 'b33', 'b4', 'b28']

Section: Entropy Approximation in Information-theoretic AFs.
Estimating entropy in information-theoretic acquisition functions is computationally expensive and typically requires approximation techniques. Methods such as ES and PES employ sampling-based approaches, including Markov chain Monte Carlo and expectation propagation. In contrast, MES derives an explicit approximation (Wang & Jegelka, 2017, Eq. 6), which was later interpreted as a variational inference formulation by Takeno et al. (2020). This variational perspective has since been extended to multiobjective optimization (Qing et al., 2023). However, this approximation scheme lacks flexibility in tuning the variational distributions. Furthermore, to the best of our knowledge, most MES-based methods focus on approximating p(y x | y * , D t ). An exception is Ma et al. (2023), which approximates p(y * | D t , y x ) using a Gaussian distribution. While this approach provides computational advantages, the inherent symmetry of the Gaussian distribution does not align with the properties of y * .
this section cite: ['b36', 'b29', 'b23']

Section: Variational Entropy Search

this section cite: []

Section: Entropy Search Lower Bound
The idea behind our Variational Entropy Search (VES) framework is to maximize a variational lower bound of MES with a predetermined family of densities to approximate p(y * | D t , y x ). Since we assume noiseless observations, the support is [max{y x , y * t }, +∞). VES is illustrated in Figure 1. The lower bound is formalized in Theorem 3.1 and proven in Appendix A.1.
Theorem 3.1. The MES acquisition function in Eq. (4) adheres to the Barber-Agakov (BA) bound (Barber & Agakov, 2004;Poole et al., 2019) and can be bounded from below as follows: Since the first term on the right-hand side of Eq. ( 7), H[y * | D t ], is independent of both q and x, we can omit it. This leads us to define the remaining term as the Entropy Search Lower Bound (ESLBO):
α MES (x) = H[y * | D t ] -E p(yx|Dt) H[y * | D t , y x ] ≥ H[y * | D t ] + E p(y * ,yx|Dt) log q(y * | D t , y x ) ,(7)
ESLBO(x; q) := E p(y * ,yx|Dt) log q(y * | D t , y x ) ,(8)
where p(y * , y x | D t ) represents a joint density, which can be sampled using Gaussian process path sampling (Hernández-Lobato et al., 2014;Wang & Jegelka, 2017).
To optimize α MES (x), we adopt the VI approach (Paisley et al., 2012), indirectly maximizing α MES (x) by instead maximizing ESLBO. To ensure computational feasibility, the VI method constrains the density q to a predefined family Q. When parameterizing q within Q, the problem becomes tractable by solving for q and x iteratively, as detailed in Algorithm 1.
Notably, this procedure, known as expectation maximization (EM), is analogous to maximizing the ELBO in Eq. ( 5). We conclude our discussion by summarizing the correspondence between ESLBO and ELBO in Table 1.
Algorithm 1 VES Framework Input: Observations D t , variational family Q, number of inner iteration N Output: Next sampling location x t+1 1: initialize x (0) t+1 2: for n = 1 : N do 3: q (n) (y * ) ← arg max q∈Q ESLBO(x (n-1) t+1 ; q) 4: x (n) t+1 ← arg max xt+1 ESLBO(x t+1 ; q (n) ) 5: end for 6: return x (N ) t+1
this section cite: ['b3', 'b26', 'b13', 'b40', 'b25']

Section: EI Through the Lens of the VES Framework
In this section, we aim to establish an explicit connection between the VES and EI acquisition functions, allowing us to see EI through the lens of a VI approximation of the information-theoretical MES AF. We define Q as the set of all exponential density functions, Q exp , parameterized by the λ > 0 exponential density parameter and with support bounded from below by max{y x , y * t }. The variational density function q is given by
q(y * |D t , y x ; λ) = λe -λ(y * -max{yx,y * t }) 1 y * ≥max{yx,y * t } .(9)
For noiseless observations, the indicator function 1 y * ≥max{yx,y * t } always equals one and can be omitted. Plugging in q from Eq. ( 9) into the ESLBO (Eq. ( 8)) yields a new λ-parameterized AF. Since this AF stems from the exponential distribution, we name it VES-Exp. Theorem 3.2 shows that the next sampling point generated from VES-Exp within Algorithm 1 will be the same as for the EI AF; the theorem is proven in Appendix A.2.
Theorem 3.2. When the family Q exp is selected as in Eq. (9) and the function is noiseless, ESLBO in Eq. (8) turns into ESLBO(x; λ) = log λ -λ E p(y * |Dt) [y * ] constant + λ E p(yx|Dt) [max{y x , y * t }] EI . (10) Maximizing ESLBO(x; λ) in Eq. (10) with respect to x and λ yields the same x solution as the maximization of EI in Eq. (1).
The key idea behind the proof is that, following Algorithm 1, the ESLBO in Eq. ( 10) always converges within two iterations. Regardless of the positive value of λ, the value of x that maximizes ESLBO(x; λ) remains the same. Consequently, starting from an arbitrary initial point x (0) , a positive λ (1) is derived, ensuring that ESLBO reaches its maximum value in the next iteration.
Theorem 3.2 reveals that EI can be viewed as a special case of MES, giving a new information-theoretic interpretation of the most popular acquisition function in use today. However, the exponential distribution has a fairly rigid parametric form that does not capture the characteristics of p(y * | D t , y x ). Figure 1 (right) shows an example of the structural limitations of the exponential density in green.
We generate 1000 samples from an example distribution p(y * | D t , y x ), and observe that it significantly deviates from an exponential distribution. Specifically, the density of p(y * | D t , y x ) is non-monotonic, exhibiting a peak before decreasing near max{y x , y * t } (approximately 1.55), while exponential distributions are necessarily monotonic.
This observation motivates the need to enrich the variational distributions Q to allow more flexibility. A natural extension is to use a Gamma distribution, which is a generalization of the exponential distribution. The Gamma density approximation in the previous example is shown in red in Figure 1 (right). The next section introduces VES-Gamma, which is a more general AF that extends VES-Exp and its equivalent EI acquisition function.
this section cite: []

Section: VES-Gamma: A Generalization of EI
VES-Gamma defines Q as the Gamma distribution parameterized by k, β > 0 with its support bounded from below by max{y x , y * t }. The variational density is Table 1. Comparison of key aspects between the ELBO and ESLBO approaches.
q(y * | D t , y x ; k, β) = β k Γ(k) (y * -max{y x , y * t }) k-1 × e -β(y * -max{yx,y * t }) 1 y * ≥max{yx,y * t } ,(11)
this section cite: []

Section: Property

this section cite: []

Section: ELBO Approach ESLBO Approach
Primary Variable p( x | z) x Variational Variable q(z) q(y * | y x , D t ) Lower Bound Formulation ELBO(q(z); p( x | z)) ESLBO(q; x)
the ESLBO is reformulated as
ESLBO(x; k, β) = k log β -log Γ(k) + (k -1)E p(y * ,yx|Dt) [log (y * -max{y x , y * t })] -βE p(y * |Dt) [y * ] + β E p(yx|Dt) [max{y x , y * t }] EI . (12
)
The ESLBO in Eq. ( 12) serves as the primary objective in the VES-Gamma algorithm. Eq. ( 12) consists of five terms, with the last term being the EI AF in Eq. ( 1) scaled by a multiplicative factor. The two hyperparameters, k and β, originally part of the Gamma distribution, dynamically balance different components of the objective. In particular, when k = 1, the Gamma distribution reduces to an exponential distribution, making the ESLBO in Eq. ( 12) equivalent to Eq. ( 10). In the following section, we discuss the approach for determining values for k and β.
this section cite: []

Section: Auto-determination of Tradeoff Hyperparameters.
For any fixed x, the global maximum of the ESLBO in Eq. ( 12) with respect to k and β uniquely exists, as can be demonstrated through derivative analysis. Taking the partial derivatives of ESLBO in Eq. ( 12) and setting them to zero, we obtain:
log β - ∂ log Γ(k) ∂k + E [log z * x ] = 0, k β -E [z * x ] = 0,
where the random variable z * x := y * -max{y x , y * t }. Substituting the second equation into the first yields:
log k -ψ(k) = log E[z * x ] -E[log z * x ],(13)
where ψ(k) := ∂ log Γ(k)/∂k is the digamma function (Abramowitz et al., 1988), which can be efficiently approximated as a series. By Jensen's inequality, log E[z * x ] -E[log z *
x ] ≥ 0. Since log k -ψ(k) is strictly decreasing and approaches zero asymptotically (see Figure 2), the root of Eq. ( 13), k *
x , exists uniquely-except in the degenerate case where z *
x is deterministic. In the practical implementation, we apply a clamping function to ensure that the term log k -ψ(k) does not become zero, and we employ a regularization mechanism to keep the resulting root k *
x close to 1. Specifically, we use L2 regularization when solving log k -
ψ(k) = log E[z * x ] -E[log z * x ]
since the unregularized version is unstable, presumably due to a widely flat landscape. In particular, for ξ(k) := 0 100 200 300 400 500
k 10 3 10 2 10 1 10 0 10 1 log(k) (k) Plot of log(k) (k) Figure 2. Plot of log k -ψ(k) for k ∈ [0, 500].
The function is strictly decreasing and asymptotically approaches zero.
log k -ψ(k) -log E[z * x ] + E[log z * x ]
, we solve the following optimization problem:
min k ξ(k) 2 + λ (k -1) 2 , (14
)
where λ is a regularization parameter which is set to 1 in our experiments.
With this analysis, the value k *
x is determined by minimizing Eq. ( 14) using Brent's method (Brent, 2013), where expectations of z *
x are estimated via Monte Carlo sampling from p(y * , y x | D t ). Once k *
x is obtained, the corresponding β *
x follows as:
β * x ← k * x E [z * x ] .(15)
Notably, the weighting parameters k * x and β * x are locationdependent, as z *
x itself varies with x. The VES-Gamma algorithm, which incorporates these principles, is detailed in Algorithm 2.
Although we provide both a theoretical justification and a practical implementation for the VES-Gamma AF, a deeper interpretation of the ESLBO in Eq. ( 12) remains an open research question. Due to the complex non-linear structure of Eq. ( 12), it is currently uncertain if there is a clear and straightforward interpretation of the various terms and the overall expression. As an example, we hypothesize that the third term acts as an "anti-EI" component, steering the VES-Gamma solution away from the EI recommendation to promote diversity, with the values of β *
x and k * x dynamically balancing its influence. Investigating this hypothesis and further elucidating the role of each term within ESLBO will be the focus of future research.
this section cite: ['b0', 'b6']

Section: Computational Cost of VES-Gamma. Implementing VES-Gamma in Algorithm 2 is computationally intensive.
The number of inner iterations, N , must be sufficiently large for convergence, and each inner iteration requires estimating E[z * x ] by sampling a large number of y * . Consequently, the overall BO loop takes significantly more time than EI and MES, as shown in Table 2 for N = 5. However, since black-box function evaluations are often expensive, the additional computational cost of VES-Gamma is not a major bottleneck in many real-world applications.
this section cite: []

Section: Results

this section cite: []

Section: Experimental Setup
We employ a consistent Gaussian Process (GP) hyperparameter and prior setting across all benchmarks and acquisition functions, evaluating Bayesian optimization (BO) performance using the simple regret r(t) := f *max (xi,yx i )∈Dt y xi , where f * := max x∈X f (x). When f * is unknown, we instead report the negative best function value, -max (xi,yx i )∈Dt y xi .
To warm-start the optimization process, we initialize with 20 random samples drawn uniformly from X and model the GP using a 5 /2-Matérn kernel with automatic relevance determination (ARD) and a dimensionality-scaled lengthscale prior (Hvarfner et al., 2024). Following the theoretical assumption in the VES framework, we only focus on experiments with noise-free observations. Although all benchmarks are noiseless, we allow the GP to accommodate potential non-stationarity or discontinuities in the underlying function.
Each experiment is repeated 10 times to estimate average performance, with results reported as mean ± one standard deviation. For problems with dimension less than 50, we run 100 iterations, otherwise 1000 iterations are computed. For numerical stability in VES-Gamma, we apply clamping: z *
x = max{10 -10 , y * -max {y x , y * t }}. The expectation in Eq. ( 12) is estimated via pathwise conditioning (Wilson et al., 2021) using 128 posterior samples. Additionally, the number of inner iterations N in Algorithm 2 is set to 5, with early stopping applied if ∥x (n-1) -x (n) ∥ < d • 10 -5 , where d denotes the problem dimension. We implement VES-Gamma and our other experiments using BoTorch (Balandat et al., 2020). We always compare against LogEI (Ament et al., 2023) and use EI and LogEI interchangeably. The code is available in https://github.com/NUOJIN/variational-entropy-search.
Benchmarks. To evaluate VES, we consider three distinct categories of benchmark problems: synthetic benchmarks, GP samples, and real-world optimization tasks.
For synthetic benchmarks, we examine commonly used functions that are diverse in dimensionality and landscape complexity. Specifically, we evaluate the 2-dimensional Branin, the 4-dimensional Levy, the 6-dimensional Hartmann, and the 8-dimensional Griewank functions. These benchmarks are widely utilized in optimization studies and provide controlled testbeds for algorithmic comparisons (Surjanovic & Bingham).
For GP sample benchmarks, we draw from a GP prior with a ν = 5 /2 Matérn kernel. These experiments examine the impact of varying length scales (ℓ = {0.5, 1, 2}) and dimensionalities (d = {2, 50, 100}) on algorithmic performance.
For real-world scenarios, we utilize a set of benchmarks reflecting practical high-dimensional problems. These include the 60-dimensional Rover problem (Wang et al., 2018), the 124-dimensional soft-constrained Mopta08 (Jones, 2008) benchmark introduced in Eriksson & Jankowiak (2021), the 180-dimensional Lasso-DNA problem from LassoBench ( Šehić et al., 2022), and the 388-dimensional SVM benchmark, also introduced in Eriksson & Jankowiak (2021). These tasks represent optimization challenges in engineering design, machine learning, and computational biology.
Due to space constraints, additional experiments are provided in Appendix B.
this section cite: ['b17', 'b43', 'b2', 'b1', 'b41', 'b18', 'b8', 'b8']

Section: Comparing VES-Exp and EI
Kolmogorov-Smirnov Test. After establishing the theoretical equivalence of VES-Exp and EI in Section 3.2, we aim to validate this equivalence in our practical implementation. To this end, we employ the Kolmogorov-Smirnov (KS) two-sample test with a significance level of α = 5% to assess statistical similarity. The two samples consist of function values evaluated by each acquisition function (AF) across 10 repeated trials, i.e., Y EI (t) := {y i t } 10 i=1 , where y i t denotes the function evaluation at step t in the i-th trial. The null hypothesis states that the function evaluations from VES-Exp and EI originate from the same distribution.
We collect function values for all 500 iterations and consider a test successful (pass) for each iteration t if the null hypothesis is not rejected. We include six different benchmarks spanning low-dimensional synthetic problems to high-dimensional real-world scenarios. Additional implementation details on KS test are presented in Appendix C.
this section cite: []

Section: Empirical Equivalence Results
Figure 3 illustrates the function values obtained by VES-Exp and EI, while Table 3 reports the passing rates of the KS test across six benchmarks. The results show that all passing rates exceed 90%, with the Hartmann benchmark achieving the highest proportion of accepted tests.
Several factors explain the remaining discrepancies between VES-Exp and EI. First, since both acquisition functions are non-convex, their optimization may yield different next sampling points x t+1 due to variations in initialization. Second, VES methods employ a clamping mechanism to ensure that z * x remains numerically positive, which introduces a dependency between y * and x. In practice, this violates the assumptions used in the proof in Appendix A.2. We also employed Log-EI (Ament et al., 2023) instead of EI in our experiment, which may also explain the difference. Finally, while EI has a closed-form expression, VES-Exp relies on Monte Carlo estimation, introducing numerical inexactness and potential discrepancies.
this section cite: ['b1']

Section: Performance of VES-Gamma
Synthetic Test Functions. Figure 4 illustrates the performance of various methods, including MES, EI, and VES-Gamma, across four synthetic benchmark functions: Branin (d = 2), Levy (d = 4), Hartmann (d = 6), and Griewank (d = 8). The metric shown is the logarithm of the best value (or simple regret), averaged over 10 independent runs. Overall, these results highlight the robustness of VES-Gamma across diverse synthetic benchmarks, consistently ranking among the top-performing methods.
this section cite: []

Section: GP Samples.
Here, we study problem instances where the GP can be fitted without model mismatch. To this end, we sample realizations from an isotropic 100-dimensional GP prior with varying length scale ℓ = 0.05, 0.1, 0.25, 0.5, using the same 5 /2-Matérn covariance function for the GP prior and the GP we fit to the observations.
Figure 5 shows the optimization performance on the 100dimensional GP prior samples. For ℓ = 0.05, 0.1, 0.25, VES-Gamma outperforms EI and MES by a wide margin. EI and MES converge to a suboptimal solution. Only for ℓ = 0.5 does EI reach the same quality as VES-Gamma, outperforming MES.
Real-World Benchmarks. Figure 6 presents the performance of VES-Gamma, EI, and MES across four realworld optimization problems: the 60-dimensional Rover trajectory optimization, the 124-dimensional Mopta08 vehicle optimization, the 180-dimensional weighted Lasso-DNA regression, and the 388-dimensional SVM hyperparameter tuning benchmarks.
Consistent with previous observations, VES-Gamma delivers strong performance, significantly outperforming all other acquisition functions on the SVM benchmark. It also ranks among the top-performing methods, alongside EI, on the Mopta08 and Lasso-DNA benchmarks. On the Rover problem, VES-Gamma performs comparably to EI, while MES achieves the best results in this scenario. MES exhibits mixed performance across the benchmarks, achieving the best results on Rover but falling behind on the Mopta08 and SVM problems.
Overall, VES-Gamma demonstrates robust and consistent performance across all benchmarks, establishing itself as a versatile and reliable acquisition function for highdimensional real-world optimization problems.
this section cite: []

Section: Conclusion
In this work, we introduce Variational Entropy Search (VES), a unified framework that bridges Expected Improvement (EI) and information-theoretic acquisition functions through a variational inference approach. We demonstrate that EI can be interpreted as a special case of Maxvalue Entropy Search (MES), revealing a deeper theoretical connection between these two widely used methodologies in Bayesian optimization. Building on this insight, we propose VES-Gamma, a novel acquisition function that dynamically balances the strengths of EI and MES. Comprehensive benchmark evaluations across a diverse set of lowand high-dimensional optimization problems highlight the robust and consistently high performance of VES-Gamma. These results underscore the potential of the VES framework as a promising foundation for developing more adaptive and efficient acquisition functions in Bayesian optimization.
10 2 10 1 simple regret Branin (d = 2) 10 0 10 2 Levy (d = 4) 0 25 50 75 100 iteration 10 3 10 1 simple regret Hartmann (d = 6) EI MES UCB VES-Gamma 0 25 50 75 100 iteration 10 0 10 1 10 2 Griewank (d = 8) Figure 4. VES-Gamma, EI, and MES on the synthetic Branin (d = 2), Levy (d = 4), Hartmann (d = 6), and Griewank (d = 8) benchmark functions. Average log simple regret: VES-Gamma performs best on Branin and Hartmann, and it is competitive on Levy and Griewank. Limitations and future work. While the Gamma distribution offers flexibility, future work will explore alternative variational distributions to enhance the adaptability of VES-Gamma. Another key direction is improving computational efficiency. Additionally, extending the theoretical framework to noisy settings remains an open challenge, requiring adaptations in variational inference to account for stochastic density supports.  where Q KS (•) represents the survival function of the Kolmogorov distribution:
Q KS (z) = 2 ∞ k=1 (-1) k-1 e -2k 2 z 2 .
Alternatively, the significance level α = 0.05 can be tested using the critical value:
D 0.05 ≈ - 1 2 ln (0.025) • n 1 + n 2 n 1 n 2 .
If D > D 0.05 , we reject the null hypothesis and consider it as failure (not pass).
Detailed p-values for VES-Exp and EI Comparison. We present the p-values obtained from the experiments detailed in Section 4.2. These results are illustrated in Figure 8. It is observed that for the majority of the sample pairs, the calculated p-values are substantially above the 5% significance level.
this section cite: []

Section: References
Ref_id:b0 Title: Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables Year: (1988)
Ref_id:b1 Title: Unexpected Improvements to Expected Improvement for Bayesian Optimization Year: (2023)
Ref_id:b2 Title: BoTorch: A framework for efficient Monte-Carlo Bayesian optimization Year: (2020)
Ref_id:b3 Title: The IM Algorithm: A variational approach to Information Maximization Year: (2004)
Ref_id:b4 Title: Self-Adjusting Weighted Expected Improvement for Bayesian Optimization Year: (2023)
Ref_id:b5 Title: Exploration Enhanced Expected Improvement for Bayesian Optimization Year: (2018)
Ref_id:b6 Title: Algorithms for minimization without derivatives Year: (2013)
Ref_id:b7 Title: Greed is Good: Exploration and Exploitation Trade-offs in Bayesian Optimisation Year: (2021)
Ref_id:b8 Title: High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces Year: (2021)
Ref_id:b9 Title: A Knowledge-Gradient Policy for Sequential Information Collection Year: (2008)
Ref_id:b10 Title: The differentiation of pseudoinverses and nonlinear least squares problems whose variables separate Year: (1973)
Ref_id:b11 Title: Entropy Search for Information-Efficient Global Optimization Year: (2012)
Ref_id:b12 Title: Probabilistic Numerics: Computation as Machine Learning Year: (2022)
Ref_id:b13 Title: Predictive Entropy Search for Efficient Global Optimization of Black-box Functions Year: (2014)
Ref_id:b14 Title: Portfolio Allocation for Bayesian Optimization Year: (2011)
Ref_id:b15 Title: Stochastic Variational Inference Year: (2013)
Ref_id:b16 Title: Joint Entropy Search for Maximally-Informed Bayesian Optimization. Advances in Neural Information Processing Systems Year: (2022)
Ref_id:b17 Title: Vanilla Bayesian Optimization Performs Great in High Dimensions Year: (2024-07)
Ref_id:b18 Title: Large-Scale Multi-Disciplinary Mass Optimization in the Auto Industry Year: (2008-08-20)
Ref_id:b19 Title: Efficient global optimization of expensive black-box functions Year: (1998)
Ref_id:b20 Title: Tuning Hyperparameters without Grad Students: Scalable and Robust Bayesian Optimisation with Dragonfly Year: (2020)
Ref_id:b21 Title: Variational diffusion models Year: (2021)
Ref_id:b22 Title: Auto-Encoding Variational Bayes Year: (2014)
Ref_id:b23 Title: Gaussian Max-Value Entropy Search for Multi-Agent Bayesian Optimization Year: (2023)
Ref_id:b24 Title: The application of Bayesian methods for seeking the extremum Year: (1998)
Ref_id:b25 Title: Variational Bayesian Inference with Stochastic Search Year: (2012)
Ref_id:b26 Title: On Variational Bounds of Mutual Information Year: (2019)
Ref_id:b27 Title: Smooth over-parameterized solvers for non-smooth structured optimization Year: (2023)
Ref_id:b28 Title: Improving the Expected Improvement Algorithm Year: (2017)
Ref_id:b29 Title: {PF} 2 ES: Parallel Feasible Pareto Frontier Entropy Search for Multi-Objective Bayesian Optimization Year: (2023)
Ref_id:b30 Title: Gaussian Processes for Machine Learning Year: (2006)
Ref_id:b31 Title: Las-soBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso Year: (2022)
Ref_id:b32 Title: Practical Bayesian Optimization of Machine Learning Algorithms Year: (2012)
Ref_id:b33 Title: On the Design of Optimization Strategies Based on Global Response Surface Approximation Models Year: (2005)
Ref_id:b34 Title: Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design Year: (2010)
Ref_id:b35 Title: Virtual library of simulation experiments: Test functions and datasets, optimization test problems Year: ()
Ref_id:b36 Title: Multi-fidelity Bayesian Optimization with Max-value Entropy Search and its Parallelization Year: (2020)
Ref_id:b37 Title: Sequential and Parallel Constrained Max-value Entropy Search via Information Lower Bound Year: (2022-06)
Ref_id:b38 Title: Joint Entropy Search for Multi-objective Bayesian Optimization Year: (2022)
Ref_id:b39 Title: An informational approach to the global optimization of expensiveto-evaluate functions Year: (2009)
Ref_id:b40 Title: Max-value Entropy Search for Efficient Bayesian Optimization Year: (2017)
Ref_id:b41 Title: Batched Large-scale Bayesian Optimization in High-dimensional Spaces Year: (2018)
Ref_id:b42 Title: The Reparameterization Trick for Acquisition Functions Year: (2017)
Ref_id:b43 Title: Pathwise Conditioning of Gaussian Processes Year: (2021)
