Title: On the Optimal Construction of Unbiased Gradient Estimators for Zeroth-Order Optimization
Abstract: Zeroth-order optimization (ZOO) is an important framework for stochastic optimization when gradients are unavailable or expensive to compute. A potential limitation of existing ZOO methods is the bias inherent in most gradient estimators unless the perturbation stepsize vanishes. In this paper, we overcome this biasedness issue by proposing a novel family of unbiased gradient estimators based solely on function evaluations. By reformulating directional derivatives as a telescoping series and sampling from carefully designed distributions, we construct estimators that eliminate bias while maintaining favorable variance. We analyze their theoretical properties, derive optimal scaling distributions and perturbation stepsizes of four specific constructions, and prove that SGD using the proposed estimators achieves optimal complexity for smooth non-convex objectives. Experiments on synthetic tasks and language model fine-tuning confirm the superior accuracy and convergence of our approach compared to standard methods.

Section: Introduction
In this paper, we consider the problem of zeroth-order optimization (ZOO), where our goal is to solve the following stochastic optimization problem:
min xPR d f pxq :" E ξ"Ξ f px; ξq,(1)
where f px; ξq is a smooth loss function evaluated on data ξ drawn from a distribution Ξ. In many practical scenarios, gradient information is either unavailable or prohibitively expensive to compute. Due to its versatility, ZOO has been widely adopted across various domains, including black-box adversarial attacks on machine learning models [Chen et al., 2017, Kurakin et al., 2016, Papernot et al., 2017, Cai et al., 2021, Zhao et al., 2020], physics-informed neural networks interfacing with external PDE solvers [Shen et al., 2024, Ma et al., 2025], and reinforcement learning [Choromanski et al., 2018, Lei et al., 2022, Suh et al., 2022]. Recent research on ZOO also focuses on enhancing memory efficiency [Cai et al., 2022a,b, Li et al., 2024, Sugiura and Matsutani, 2025], motivated in large part by fine-tuning large language models [Malladi et al., 2023, Zhang et al., 2024, Gautam et al., 2024, Tang et al., 2024, Wang et al., 2024, 2025].
Unlike first-order methods that rely on stochastic gradients ∇f px; ξq, ZOO uses only function evaluations, without access to gradient information. To approximate gradients, several estimators have been proposed, including the one-point estimate ∇f px; ξq " f px`µv;ξq µ v [Flaxman et al., 2005, Shamir, 2013, Bach and Perchet, 2016, Nesterov and Spokoiny, 2017, Berahas et al., 2022] and two-point estimator ∇f px; ξq " f px`µv;ξq´f px;ξq µ v [Ghadimi and Lan, 2013, Duchi et al., 2015, Nesterov and Spokoiny, 2017] (see Appendix A.1 for further discussions). The random direction v is typically drawn from a Gaussian or uniform spherical distribution, while alternative choices have also gained increasing attention in recent years [Ghadimi and Lan, 2013, Duchi et al., 2015, Ji et al., 2019, Sahu et al., 2019, Coope and Tappenden, 2020, Kozak et al., 2023, Rando et al., 2024a,b, Ma and Huang, 2025, Mi et al., 2025].
However, despite these advancements, a critical limitation arises in zeroth-order gradient estimation; that is, all widely used gradient estimators exhibit inherent bias. Specifically, unless the perturbation step size µ asymptotically tends to zero, these estimators yield persistently biased approximations of the true gradient. This inherent bias motivates a central question explored in this paper: Q1: Is it possible to design an unbiased zeroth-order gradient estimator using only function evaluations?
Contribution 1: In this paper, we answer Q1 affirmatively. Contrary to the belief that zeroth-order gradient estimators must inherently be biased due to finite-step perturbations, we demonstrate that it is indeed possible to construct unbiased gradient estimators using only function evaluations. Our key idea is to express ∇ v f pxq (in the deterministic setting), the directional derivative along the direction v, as a telescoping series:
∇ v f pxq :" lim µnÑ0 f px `µn vq ´f pxq µ n " 8 ÿ n"1 p n " f px `µ1 vq ´f pxq µ 1 `1 p n ˆf px `µn`1 vq ´f pxq µ n`1 ´f px `µn vq ´f pxq µ n ˙ȷ , (2) piq " E n"tpnu 8 n"1 " f px `µ1 vq ´f pxq µ 1 `1 p n ˆf px `µn`1 vq ´f pxq µ n`1 ´f px `µn vq ´f pxq µ n ˙ȷ
where the perturbation stepsize µ n Ñ 0 as n Ñ 8, the sampling distribution tp n u 8 n"1 form a probability distribution (that is, 0 ă p i ă 1 for all i P N and ř 8 i"1 p i " 1), and the expectation representation (i) holds under mild regularity conditions (Proposition 2.1). This formulation allows us to reinterpret the directional derivative as an expectation over n " tp n u 8
n"1 , enabling the construction of a unbiased gradient estimator family P (Definition 2.2):
∇v f pxq :" E n"p Ppn, vq,
where Ppn, vq is an unbiased estimator of f px `µ1 vq ´f pxq µ 1 `1 p n ˆf px `µn`1 vq ´f pxq µ n`1 ´f px `µn vq ´f pxq µ n ˙.
Within this framework, we propose four specific estimators, denoted as P k -estimator for k " 1, 2, 3, 4, corresponding to the number of function evaluations required in each estimation. To the best of our knowledge, unbiased zeroth-order gradient estimators have received little attention in prior literature. The only existing work we are aware of is the four-point estimator proposed by Chen [2020], which shares the same telescoping structure and can be viewed as a special case of our P 4 -estimator.
Contribution 2: Building on our unbiased estimator construction, we conduct a rigorous variance analysis on our proposed P k -estimators. We first present a negative result for the P 1 -estimator; although it requires fewer function evaluations, it may exhibit infinite variance under certain conditions (Theorem 3.1 (a)), which aligns with the one-point estimator in the randomized smoothing [Flaxman et al., 2005]. Next, we characterize the relation among the variance of the P k -estimator (k " 2, 3, 4), the perturbation stepsize sequence tµ n u 8 n"1 , and the sampling distribution tp n u 8 n"1 (Theorem 3.1 (b)). Identifying the optimal choice of tµ n u 8
n"1 and tp n u 8 n"1 leads us to the following non-convex functional optimization problem: min tµnu 8  n"1 ,tpnu 8 n"1
E n"tpnu 8 n"1 ˆµn ´µn`1 p n ˙2(3)
subject to 0 ă p n ă 1;
8 ÿ n"1 p n " 1; 8 ÿ n"1 µ n ă 8.
We present an explicit analytical solution to this optimization problem (Theorem 3.2), which reveals two key insights: (1) our constructed unbiased gradient estimators can achieve the same variance as the classical two-point estimator without introducing additional bias, leading to the best-possible complexity for SGD algorithm (Corollary 3.5); (2) a broad class of sampling distributions can achieve the minimum variance, extending beyond the specific choices considered in prior work [Chen, 2020].
While our theoretical results establish strong guarantees, an important practical question remains:
Q2: Given the optimal choice of tµ n u 8 n"1 and tp n u 8 n"1 , do the proposed unbiased estimators empirically outperform existing zeroth-order methods?
Contribution 3: To address Q2, we empirically validate our proposed approach across both synthetic and practical tasks. On estimating the gradient of mean-square and logistic losses, our method achieves significantly lower gradient estimation error compared to standard zeroth-order methods (Section 4.1). Furthermore, when applied to fine-tuning large language models, the proposed estimators demonstrate faster convergence and higher final accuracy under the same number of function evaluations (Section 4.2). These results confirm the practical advantages of our unbiased construction and underscore its effectiveness in modern zeroth-order optimization tasks.
this section cite: ['b9', 'b34', 'b49', 'b4', 'b66', 'b56', 'b11', 'b35', 'b59', 'b58', 'b44', 'b22', 'b60', 'b18', 'b54', 'b2', 'b47', 'b3', 'b23', 'b16', 'b47', 'b23', 'b16', 'b29', 'b53', 'b12', 'b33', 'b45', 'b7', 'b18', 'b7']

Section: The Derivation of Unbiased Zeroth-Order Estimators
We will start from the deterministic case then turn to the stochastic case in Section 3.3. In this section, we formally derive a class of unbiased estimators for approximating the gradient ∇f pxq using only function evaluations. We also provide a sufficient condition under which the telescoping series in Eq. ( 4) admits the expectation representation. All proofs are provided in the appendix.
this section cite: []

Section: Telescoping Series and Expectation Representation
For a fixed direction v P R d , the directional derivative of a differentiable function f : R d Ñ R at x along the direction v is defined as
∇ v f pxq " lim µÑ0 f px `µvq ´f pxq µ .
Then for any decreasing sequence tµ n u 8 n"1 with lim nÑ8 µ n " 0, one can express this directional derivative as the limit of a convergent sequence
! f px`µnvq´f pxq µn ) : ∇ v f pxq " lim nÑ8 f px `µn vq ´f pxq µ n .
This convergent sequence canonically induces a telescoping series with the same limit:
∇ v f pxq " f px `µ1 vq ´f pxq µ 1 `8 ÿ n"1 " f px `µn`1 vq ´f pxq µ n`1 ´f px `µn vq ´f pxq µ n ȷ . (4)
Next, consider a probability mass function (PMF) tp n u 8 n"1 with p n ą 0 for all n and ř 8 n"1 p n " 1. When the series in Eq. ( 4) is absolutely convergent 2 , we can interpret it as an expectation over a discrete random variable n. That is,
∇ v f pxq " 8 ÿ n"1 p n " f px `µ1 vq ´f pxq µ 1 `1 p n ˆf px `µn`1 vq ´f pxq µ n`1 ´f px `µn vq ´f pxq µ n ˙ȷ " E " f px `µ1 vq ´f pxq µ 1 `1 p n ˆf px `µn`1 vq ´f pxq µ n`1 ´f px `µn vq ´f pxq µ n ˙ȷ .
(5) 2 We follow the standard definition from Spivak [2008]: A series ř 8 n"1 an is called convergent, if the limit of its finite sum limNÑ8 ř N n"1 an exists. A series ř 8 n"1 an is called absolutely convergent, if the series ř 8
n"1 |an| is convergent. See the formal definition in Appendix B.1.
this section cite: []

Section: On the Role of Absolute Convergence.
The absolute convergence of the series in Eq. ( 4) plays a critical role in interpreting the telescoping series as an expectation. This is due to the difference between the series convergence and the existence of expectation:
• The series convergence: Consider the convergent series ř 8 i"1 p i x i . To evaluate its value, we can calculate the finite-sum S n :" ř n i"1 p i x i ; then we have
ř 8 i"1 p i x i " lim nÑ8 S n .
• The existence of expectation: Consider the random variable X with PpX " x i q " p i for i P N. Its expectation ErXs is also written as ř 8 i"1 p i x i . However, the notion of expectation must be well-defined independently of any ordering of outcomes. That is, for an arbitrary permutation σ : N Ñ N, all series ř 8 i"1 p σpiq x σpiq should represent the same value ErXs. As a result, a convergent series can yield different values depending on the order of summation (this result is called the Riemann series theorem [Riemann, 1868, Spivak, 2008]); however, the outcomes of a random variable requires a random variable's expectation to be well-defined regardless of any such ordering. While the expectation representation has been discussed in prior work (e.g., [Chen, 2020]), the lack of attention to absolute convergence has left the conditions ensuring unbiasedness underexplored.
Due to this reason, we provide the following (mild) sufficient condition for ensuring the absolute convergence with adding a slightly stronger requirement on the objective function f : R d Ñ R and the sequence tµ n u 8
n"1 : Proposition 2.1. If the second-order continuously differentiable function f : R d Ñ R has L-Lipschitz continuous gradient and ř 8 n"1 µ n ă 8, then the series
8 ÿ n"1 p n " f px `µ1 vq ´f pxq µ 1 `1 p n ˆf px `µn`1 vq ´f pxq µ n`1 ´f px `µn vq ´f pxq µ n
˙ȷ is absolutely convergent and its limit is ∇ v f pxq.
this section cite: ['b52', 'b7']

Section: The Construction of Unbiased Estimators
With the expectation representation in place, we are ready to define the class of unbiased estimators explicitly.
Definition 2.2. Suppose that the function f : R d Ñ R is continuously differentiable and tµ n u ně1 is a positive sequence with lim nÑ8 µ n " 0 such that the telescoping series
f px `µ1 vq ´f pxq µ 1 `8 ÿ n"1 " f px `µn`1 vq ´f pxq µ n`1 ´f px `µn vq ´f pxq µ n ȷ
is absolutely convergent, the sequence tp n u 8 n"1 forms a PMF, and V is the distribution over R d . Then the family of estimators P :" Ppf, tµ n u 8
n"1 , tp n u 8 n"1 , V q denote the class of random variables such that for every Ppn, vq P P, it satisfies
ErPpn, vq | n, vs " f px `µ1 vq ´f pxq µ 1 `1 p n ˆf px `µn`1 vq ´f pxq µ n`1 ´f px `µn vq ´f pxq µ n ˙,
where v is sampled from V , independent with n " tp n u 8 n"1 .
In the following theorem, we formally prove that our proposed class P is exactly the unbiased estimator of the gradient ∇f pxq. Theorem 2.3 (Unbiasedness). Let P :" Ppf, tµ n u 8 n"1 , tp n u 8 n"1 , V q is defined as Definition 2.2. Then, for any estimator Ppn, vq P P, the following hold:
(a) ErPpn, vq | vs " ∇ v f pxq; that is, Ppn, vq is an unbiased estimator of the directional derivative ∇ v f pxq.
(b) If the random direction v is chosen independently of the sampling n " tp n u 8 n"1 and satisfies Erv v J s " I, then
E n"tpnu 8 n"1 ,v"V " Ppn, vq v ı " ∇f pxq,
so that Ppn, vq v is an unbiased estimator of the gradient ∇f pxq.
this section cite: []

Section: Specific Constructions
In this subsection, we propose four concrete constructions from the estimator family (Definition 2.2) P :" Ppf, tµ n u 8 n"1 , tp n u 8 n"1 , V q based on the number of function evaluations used in estimating the gradient. These constructions are designed to explore two main aspects: (1) the trade-off between the estimator variance and the number of function evaluations, allowing flexibility depending on the computational budget; and (2) a fundamental question purely driven by the theoretical interest: What is the minimum number of function evaluations required to construct an unbiased gradient estimator? P 4 -Estimator. This estimator corresponds to the four-point estimator originally proposed by Chen [2020] with slightly generalizing the choice of the perturbation stepsize sequence tµ n u 8
n"1 and the sampling distribution tp n u 8
n"1 . For a given direction v " V and n " tp n u 8 n"1 , the P 4 -estimator is defined as
P 4 pn, vq " f px `µ1 vq ´f pxq µ 1 `1 p n « f px `µn`1 vq ´f pxq µ n`1 ´f px `µn vq ´f pxq µ n ff . (6)
This construction requires four function evaluations at: x, x `µ1 v, x `µn v, and x `µn`1 v, exhibiting the lowest variance and the most function evaluation counts among all members of P.
P 3 -Estimator. We can reduce one function evaluation by introducing a selection random variable U 2 " Uniform pt0, 1uqfoot_0 . The estimator is then defined as
P 3 pn, vq " f px `µ1 vq ´f pxq µ 1 U 2 `1 p n « f px `µn`1 vq ´f pxq µ n`1 ´f px `µn vq ´f pxq µ n ff p1´U 2 q.
(7) This construction randomly selects one of two pathways: With probability 1{2, it uses the first term only and requires two function evaluations at x `µ1 v and x; otherwise, it uses the second term and requires three function evaluations at x `µn v, x `µn`1 v, and x. This estimator maintains unbiasedness as P 4 , with slightly higher variance.
P 1 -& P 2 -Estimator.
The selection random variable can be naturally extended to construct P 1 -and P 2 -estimators as follows:
P 2 pn, vq " f px `µ1 vq ´f pxq µ 1 I tU3"0u(8)
`1 p n « f px `µn`1 vq ´f pxq µ n`1 I tU3"1u ´f px `µn vq ´f pxq µ n I tU3"2u ff , P 1 pn, vq " f px `µ1 vqI tU4"1u ´f pxqI tU4"0u µ 1 (9) `1 p n « f px `µn`1 vqI tU4"2u ´f pxqI tU4"0u µ n`1 ´f px `µn vqI tU4"3u ´f pxqI tU4"0u µ n ff ,
where U 3 " Uniform pt0, 1, 2uq, U 4 " Uniform pt0, 1, 2, 3uq, and I A is the indicator function, which equals 1 if the event A occurs, and 0 otherwise. of the event A. Remarkably, the construction of P 1 -estimator achieves unbiasedness using only a single function evaluation. However, we will show that in the next section,P 1 -estimator will have infinite variance under certain condition.
this section cite: ['b7']

Section: Variance Analysis of Unbiased Zeroth-Order Estimators
In this section, we provide a theoretical analysis of the variance behavior for the unbiased estimator family P " Ppf, tµ n u 8 n"1 , tp n u 8 n"1 , V q (Definition 2.2). While the unbiasedness has been shown in Theorem 2.3, their variances can differ dramatically depending on the estimator construction. In particular, we prove that the variance becomes unbounded (i.e., infinite) for certain constructions such as P 1 -estimator. We also provide finite-variance bounds for P k -estimators (for k " 2, 3, 4) with matching the optimal variance under specific choices of tp n u and tµ n u.
this section cite: []

Section: Theoretical Analysis
In the following result, we adopt the same condition as Proposition 2.1 to ensure the expectation representation.
Theorem 3.1. Let P :" Ppf, tµ n u 8 n"1 , tp n u 8 n"1 , V q is defined as Definition 2.2. Suppose that f : R d Ñ R is second-order continuously differentiable and has L-Lipschitz continuous gradient, ř 8 n"1 µ n ă 8, and V is the uniform distribution over the sphere with the radius
? d. Define µ :" µ 1 , ϱ :" 8 ÿ n"1 pµ n`1 ´µn q 2 p n , and φ :" 8 ÿ n"1 µ 2 n p n .
Then the following statements hold:
(a) If there exists a point x P R d such that the Hessian ∇ 2 f pxq is positive definite and f pxq ‰ 0, then the variances of the P 1 for estimating ∇f pxq is infinite.
(b) The variance of P k -estimator P k pn, vqv (k " 2, 3, 4) for estimating ∇f pxq is given by
VarrP 2 pn, vq vs ď VarrP 4 pn, vq vs `L2 3 d 3 µ 2 `L2 12 d 3 ϱ `L2 3 d 3 φ. VarrP 3 pn, vqvs ď VarrP 4 pn, vq vs `L2 8 d 3 µ 2 `L2 8 d 3 ϱ. VarrP 4 pn, vqvs ď pd ´1q}∇f pxq} 2 `3L 2 4 d 3 µ 2 `L2 d 3 2 ϱ.
Proof. Part (a) directly follows by analyzing the tail of 1 pn f px`µnvq µn and leveraging the curvature from a positive definite Hessian. For the part (b), we simply decompose the variance of P 2 pn, vqv and P 3 pn, vqv into the variance of estimating P 4 pn, vqv using VarrP vs " dErpP ´P4 pn, vqq 2 s `VarrP 4 pn, vq vs for arbitrary P :" Ppn, vq P P. Then we apply the second-order Taylor expansions with the mean value theorem to control the finite-difference noise. Full details and auxiliary lemmas are provided in Appendix C.
Comparison with Existing Literature. Theorem 3.1 reveals that while P 1 is unbiased, its variance can be infinite under certain conditions, making them unsuitable for SGD. In contrast, P k -estimator (k " 2, 3, 4) offer the finite variance when tµ n u 8
n"1 and tp n u 8 n"1 are appropriately selected. We will show it later that under the optimal setting, their variances match the optimal order of classical two-point estimators [Nesterov and Spokoiny, 2017] but with zero bias:
VarrP k vs " Opd}∇f pxq} 2 `d3 µ 2 q.
This variance will lead to the optimal function query complexity Op d ϵ 4 q for achieving ϵ-accuracy in the gradient norm }∇f pxq} [Duchi et al., 2015].
Comparison with the Noisy Oracle Setup In our work, we consider the exact function evaluation setting with noiseless values. In this case, our variance scales as d 3 µ 2 , which is worse than the d 2 µ 2 of some specific biased estimators, which is mitigated by choosing a small enough µ; the overall sample complexity remains optimal. However, in the noisy function evaluation setting, where each function evaluation may return a noisy value, a smaller µ amplifies the noise, leading to degraded performance. Several recent works have provided more refined analysis under noisy setups with improved variance behavior. Notably, Akhavan et al. [2024] demonstrated that for highly smooth functions, the ℓ 1 -randomization can reduce the variance scaling to d 2 µ 2 with achieving the improved performance for highly smooth objective functions, which extends the existing ℓ 1 -randomization proposed by Akhavan et al. [2022]. Earlier work by Gasnikov et al. [2017] analyzed the variance behavior in single-point and multi-point bandit feedback settings, and more recent developments further explore the impact of first-order smoothness in noisy black-box optimization [Gasnikov et al., 2022]. Notably, all of these results achieve the optimal complexity derived by Duchi et al. [2015].
this section cite: ['b47', 'b16', 'b1', 'b0', 'b21', 'b20', 'b16']

Section: On the Optimal Choices of tµ n u 8
n"1 and tp n u 8
n"1
In previous section, Theorem 3.1 connects the perturbation stepsize sequence tµ n u, the sampling distribution tp n u, and the variance upper bounds of our constructed unbiased estimators, which has received limited discussion in the existing literature. To control the variance term, one must ensure that ϱ :" ř 8
n"1 pµn`1´µnq 2 pn (and φ :" ř 8
n"1 µ 2 n pn for P 2 -estimator) is sufficiently small. This observation naturally raises the question: What are the optimal sequences tµ n u 8
n"1 and tp n u 8 n"1 that minimize this sum? The following theorem addresses this question: Theorem 3.2. Let tµ n u 8
n"1 be a positive, decreasing sequence with ř 8 n"1 µ n ă 8, and let tp n u 8
n"1 be a PMF. Denote µ :" µ 1 . Then the following statements hold:
(a) The lower bound of ϱ is given by ϱ ě µ 2 . Moreover, the equality holds if and only if p n " µn´µn`1 µ .
(b) The lower bound of φ is given by φ ě ´ř8 n"1 µ n ¯2 ą µ 2 . Moreover, the equality holds if and only if p n " µn ř 8 n"1 µn .
This result characterizes the choices of tµ n u 8
n"1 and tp n u 8 n"1 that minimizes ϱ (and φ for the P 2 -estimator), leading to the variance upper bound of the form:
maxtVarrP 2 pn, vqvs, VarrP 3 pn, vqvs, VarrP 4 pn, vqvsu ď Opd}∇f pxq} 2 `d3 µ 2 q.
Here, we can always choose tµ n u 8
n"1 for the P 2 -estimator such that µ 1 « ř 8 n"2 µ n to nearly match the lower bound (« µ 2 1 ).
Sampling from the Optimal Sampling Distribution tp n u 8 n"1 . When the perturbation stepsize sequence tµ n u 8
n"1 is given, sampling the corresponding optimal distribution p n " µn´µn`1 µ1 could be difficult; in most of cases, tp n u 8
n"1 cannot be a ready-to-use distribution naively supported by existing software. Fortunately, we can do it conversely: given an arbitrary PMF tp n u 8
n"1 , the perturbation stepsize takes the form µ n " µ 1 PpN ě nq, where N " tp n u 8
n"1 , providing a practical way to implement the unbiased zeroth-order gradient estimator. To illustrate this point, we provide two concrete examples. Example 3.3 (Geometric P k -Estimators). We consider the geometric distribution n " Geompcq (c P p0, 1q). Then p n " p1 ´cq c n´1 for all n P N. We define µ n by the recursion µ n ´µn`1 "
µ 1 p n " µ 1 p1 ´cq c n´1 . Summing this relation leads to the closed-form solution µ n " µ 1 c n´1
and the optimal value ϱ " µ 2 1 . This construction recovers the geometric sampling scheme used by Chen [2020]. We call the P k -estimator constructed on the geometric distribution as the geometric P k -estimator. It is easy to verify that the corresponding φ is given as φ "
ř 8 n"1 µ 2 n pn " µ 2 1 p1´cq 2 . Example 3.4 (Zipf's P k -Estimators).
We consider the Zipf's distribution n " Zipfpsq (s ą 1). Then p n " 1 ζpsq 1 n s for all n P N, where ζ is the Riemannian zeta function defined as ζpsq "
ř 8 n"1 1 n s . We define µ n by the recursion µ n ´µn`1 " µ 1 p n " µ 1 1 ζpsq 1 n s . Summing this relation leads to the closed-form solution µ n " µ 1 « 1 ´řn´1 j"1 1 j s ζpsq ff .
This construction also leads to the optimal value ϱ " µ 2 1 . When estimating the upper bound of φ, we additionally assume s ą 3. In this case, we have φ "
ř 8 n"1 µ 2 n pn ď ζps´2q ps´1q 2 ζpsq µ 2 1 .
The detailed calculation is put in Example C.7.
In both examples, we start with a well-known easy-to-sample distribution tp n u 8
n"1 , and calculate the associated perturbation stepsize sequence tµ n u 8
n"1 either analytically (Geometric P k -estimators) or iteratively (Zipf's P k -estimators). While all estimators (i.e. P k -estimator with k " 2, 3, 4) achieve the optimal variance in the order with d and µ, these examples indicate a key difference between the P 2 -estimator and the P k -estimator (for k " 3, 4): the variance bound of P 3 -and P 4 -estimator is parameter-agnostic; that is, once tp n u is specified, no additional tuning of distribution parameters is required to attain the optimal bound µ 2 . This distinction highlight the practical advantages of P 3and P 4 -estimators.
this section cite: ['b7']

Section: Convergence of SGD with Unbiased Gradient Estimators
In this subsection, we consider the stochastic optimization setting described in Eq. ( 1), where the goal is to estimate the stochastic gradient ∇f px; ξq rather than the full gradient. Under the optimal sampling distribution tp n u 8
n"1 and the corresponding perturbation stepsize sequence tµ n u 8 n"1 , the convergence upper bound of SGD follows directly from standard results for general unbiased stochastic gradient methods. Corollary 3.5 (Khaled and Richtárik [2022]). Consider the stochastic optimization problem in Eq. ( 1), and suppose that the individual loss f px; ξq is second-order differentiable with L-Lipschitz continuous gradient in x, uniformly over ξ " Ξ. Assume the stochastic gradient is approximated using the P kestimator P k pn, vq v for k " 2, 3, 4. Let the SGD iteration be defined as x t`1 " x t ´ηP k pn t , v t q v t where η P p0, 1 L 2 d s is the stepsize. Then the iterates satisfy the following convergence guarantee:
min 0ďtďT ´1 E}∇f px t q} 2 ď Opd 3 µ 2 η `dη `2 ηT q.
Consequently, choosing η " Θp1{ ? dT q and µ " Op 1 d q yields the optimal complexity T "
Θp d ϵ 4 q of having min 0ďtďT ´1 E}∇f px t q} ď ϵ.
This complexity has matched the lower bound of solving a smooth non-convex optimization problem using zeroth-order gradient-based method [Duchi et al., 2015] and cannot be further improved without adding additional assumptions. Though we directly apply the result from Khaled and Richtárik [2022] (which is applicable for all unbiased estimators), the zeroth-order estimation can result in an additional dependence on the dimension d; this dependence has been reflected in our upper bound.
this section cite: ['b31', 'b16', 'b31']

Section: Experiments
To validate our theoretical results and demonstrate the effectiveness of the proposed unbiased zerothorder gradient estimators, we conduct experiments on two settings: synthetic objectives and language model optimization. Details and hyperparameter configurations are provided in Appendix E.
this section cite: []

Section: Synthetic Examples
We first evaluate our estimators on two classic loss functions [James et al., 2013]: the quadratic loss f reg : R d Ñ R for linear regression and the logistic loss f cls : R d Ñ R for binary classification.
f reg pxq " x J A J Ax, f cls pxq " 1 n n ÿ i"1 logp1 `expp´b i ¨pa J i ¨xqqq,
where each entry of A P R dˆd is independently sampled from the uniform distribution U r´1, 1s, each feature vector a i P R d is sampled from the standard normal distribution Normalp0, I d q, and b i P t´1, 1u are binary labels generated based on a Bernoulli distribution with the fixed sample size n. The gradient of each objective function can be explicitly evaluated; we compare the performance of different zeroth-order gradient estimator using the Mean-Square-Error (MSE), which is defined as MSEp ∇f pxqq :" r ∇f pxq ´∇f pxqs J r ∇f pxq ´∇f pxqs.
(10) We compare the accuracy of estimating the gradient of two loss functions among four different gradient estimators including Zipf's P 3 -estimator (Example 3.4), two-point estimator with Gaussian or uniform random perturbations, and centralized two-point estimator with uniform perturbation (the batch size of two-point estimators is adjusted to exactly 3 function evaluations). For detailed hyper-parameter setting, we put in Appendix E. Several observations can be made from the results shown in Figure 1. First, comparing the same estimator across different dimensions, the MSE error for both objective functions generally increases with the dimension d, which is expected as higherdimensional settings pose greater estimation challenges. Second, comparing different estimators, the Zipf's P 3 -estimator consistently achieves lower MSE compared to others. These results collectively demonstrate the effectiveness of our proposed estimator when estimating the gradient, especially in high-dimensional settings, which will be further validate in the next experiment.
this section cite: ['b28']

Section: Language Model Optimization
In this section, we demonstrate the practical applicability of the unbiased gradient estimators in optimizing the deep neural network. Particularly, we apply it to the task of fine-tuning a pre-trained language model. Using zeroth-order optimization to fine-tune the LLMs has been an active research field in recent years due to its effectiveness in saving memory [Malladi et al., 2023, Zhang et al., 2024, Gautam et al., 2024, Guo et al., 2024]; it allows for fine-tuning model parameters without requiring access to the full computational graph, which can be prohibitively large for modern language models. We conducted experiments using the OPT-1.3b model [Zhang et al., 2022] for sentiment classification on the Stanford Sentiment Treebank (SST-2) dataset [Socher et al., 2013]. To ensure fair comparison, we maintained consistent parameters across experiments: the learning rate η " 10 ´4 and the perturbation stepsize µ " 10 ´3 (corresponding to µ 1 in the proposed unbiased estimators), which is taken from Malladi et al. [2023]'s Table 7 without additional tuning. For two-point estimators, we have adjusted the batch size to align 4 function evaluations. Detailed experimental settings are provided in Appendix E. As shown in Figure 2, zeroth-order optimization using the proposed unbiased zeroth-order estimators achieved superior performance compared to other baseline methods.  Choosing larger batch sizes gives more accurate gradient estimates, leading to lower training loss when measured by the number of updates. However, we also observe that selecting the batch size as b " 1 may also present its own advantage. Therefore, choosing the batch size can be non-trivial and it requires to balance the variance of gradient estimation against the per-step cost.
this section cite: ['b44', 'b22', 'b26', 'b64', 'b57', 'b44']

Section: Conclusion
In this work, we proposed a novel class of unbiased zeroth-order gradient estimators based on a telescoping series expansion of directional derivatives. We established new theoretical results, including a sufficient condition for the expectation representation (Proposition 2.1), the unbiasedness of the proposed estimators (Theorem 2.3), a variance analysis for four specific constructions (Theorem 3.1), and the characterization of the optimal sampling distribution and perturbation stepsize sequence (Theorem 3.2). We further demonstrated that SGD equipped with our estimators achieves optimal sample complexity and empirically outperforms existing mini-batch two-point estimators. These results provide a principled foundation for a new class of estimators in zeroth-order optimization, offering both theoretical insights and practical improvements.
this section cite: []

Section: References
Ref_id:b0 Title: A gradient estimator via l1-randomization for online zero-order optimization with two point feedback Year: (2022)
Ref_id:b1 Title: Gradient-free optimization of highly smooth functions: improved analysis and a new algorithm Year: (2024)
Ref_id:b2 Title: Highly-smooth zero-th order online optimization Year: (2016)
Ref_id:b3 Title: A theoretical and empirical comparison of gradient approximations in derivative-free optimization Year: (2022)
Ref_id:b4 Title: A zeroth-order block coordinate descent algorithm for huge-scale black-box optimization Year: (2021)
Ref_id:b5 Title: A one-bit, comparison-based gradient estimator Year: (2022)
Ref_id:b6 Title: Zeroth-order regularized optimization (zoro): Approximately sparse gradients and adaptive sampling Year: (2022)
Ref_id:b7 Title: Unbiased gradient simulation for zeroth-order optimization Year: (2020)
Ref_id:b8 Title: Faster gradient-free algorithms for nonsmooth nonconvex stochastic optimization Year: (2023)
Ref_id:b9 Title: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models Year: (2017)
Ref_id:b10 Title: Accelerated proximal alternating gradient-descent-ascent for nonconvex minimax machine learning Year: (2022)
Ref_id:b11 Title: Structured evolution with compact architectures for scalable policy optimization Year: (2018)
Ref_id:b12 Title: Gradient and hessian approximations in derivative free optimization Year: (2020)
Ref_id:b13 Title: Convergence rates of finite difference stochastic approximation algorithms Year: (2015)
Ref_id:b14 Title: A gradient sampling method with complexity guarantees for lipschitz functions in high and low dimensions Year: (2022)
Ref_id:b15 Title: Saga: A fast incremental gradient method with support for non-strongly convex composite objectives Year: (2014)
Ref_id:b16 Title: Optimal rates for zero-order convex optimization: The power of two function evaluations Year: (2015)
Ref_id:b17 Title: Spider: Near-optimal non-convex optimization via stochastic path-integrated differential estimator Year: (2018)
Ref_id:b18 Title: Online convex optimization in the bandit setting: Gradient descent without a gradient Year: (2005)
Ref_id:b19 Title: Advanced Calculus Year: (2002)
Ref_id:b20 Title: Randomized gradient-free methods in convex optimization Year: (2022)
Ref_id:b21 Title: Stochastic online optimization. single-point and multi-point non-linear multi-armed bandits. convex and strongly-convex case Year: (2017)
Ref_id:b22 Title: Variance-reduced zeroth-order methods for fine-tuning language models Year: (2024)
Ref_id:b23 Title: Stochastic first-and zeroth-order methods for nonconvex stochastic programming Year: (2013)
Ref_id:b24 Title: Evaluating derivatives: principles and techniques of algorithmic differentiation Year: (2008)
Ref_id:b25 Title: Black-box reductions for zeroth-order gradient algorithms to achieve lower query complexity Year: (2021)
Ref_id:b26 Title: Zeroth-order fine-tuning of llms with extreme sparsity Year: (2024)
Ref_id:b27 Title: Faster gradient-free proximal stochastic methods for nonconvex nonsmooth optimization Year: (2019)
Ref_id:b28 Title: An introduction to statistical learning Year: (2013)
Ref_id:b29 Title: Improved zeroth-order variance reduced algorithms and analysis for nonconvex optimization Year: (2019)
Ref_id:b30 Title: Accelerating stochastic gradient descent using predictive variance reduction Year: (2013)
Ref_id:b31 Title: Better theory for sgd in the nonconvex world Year: (2022)
Ref_id:b32 Title: An algorithm with optimal dimension-dependence for zero-order nonsmooth nonconvex stochastic optimization Year: (2024)
Ref_id:b33 Title: Zeroth-order optimization with orthogonal random directions Year: (2023)
Ref_id:b34 Title: Adversarial machine learning at scale Year: (2016)
Ref_id:b35 Title:  Year: (2022)
Ref_id:b36 Title: Addax: Utilizing zeroth-order gradients to improve memory efficiency and performance of sgd for fine-tuning language models Year: (2024)
Ref_id:b37 Title: Gradient-free methods for deterministic and stochastic nonsmooth nonconvex optimization Year: (2022)
Ref_id:b38 Title: Zeroth-order stochastic variance reduction for nonconvex optimization Year: (2018)
Ref_id:b39 Title: A primer on zeroth-order optimization in signal processing and machine learning: Principals, recent advances, and applications Year: (2020)
Ref_id:b40 Title: Revisiting zeroth-order optimization: Minimum-variance two-point estimators and directionally aligned perturbations Year: (2025)
Ref_id:b41 Title: Understanding the impact of model incoherence on convergence of incremental sgd with random reshuffle Year: (2020)
Ref_id:b42 Title: Data sampling affects the complexity of online sgd over dependent data Year: (2022)
Ref_id:b43 Title: Deep learning of pde correction and mesh adaption without automatic differentiation Year: (2025-02)
Ref_id:b44 Title: Fine-tuning language models with just forward passes Year: (2023)
Ref_id:b45 Title: Kerzoo: Kernel function informed zeroth-order optimization for accurate and accelerated llm fine-tuning Year: (2025)
Ref_id:b46 Title: Random reshuffling: Simple analysis with vast improvements Year: (2020)
Ref_id:b47 Title: Random gradient-free minimization of convex functions Year: (2017)
Ref_id:b48 Title: Sarah: A novel method for machine learning problems using stochastic recursive gradient Year: (2017)
Ref_id:b49 Title: Practical black-box attacks against machine learning Year: (2017)
Ref_id:b50 Title: An optimal structured zeroth-order algorithm for non-smooth optimization Year: (2024)
Ref_id:b51 Title: Stochastic zeroth order descent with structured directions Year: (2024)
Ref_id:b52 Title: Über die darstellbarkeit einer function durch eine trigonometrische reihe Year: (1868)
Ref_id:b53 Title: Towards gradient free and projection free stochastic optimization Year: (2019)
Ref_id:b54 Title: On the complexity of bandit and derivative-free stochastic convex optimization Year: (2013)
Ref_id:b55 Title: An optimal algorithm for bandit and zero-order convex optimization with two-point feedback Year: (2017)
Ref_id:b56 Title: Memory-efficient gradient unrolling for large-scale bi-level optimization Year: (2024)
Ref_id:b57 Title: Recursive deep models for semantic compositionality over a sentiment treebank Year: (2013-10)
Ref_id:b58 Title: Elasticzo: A memory-efficient on-device learning with combined zeroth-and first-order optimization Year: (2025)
Ref_id:b59 Title: Do differentiable simulators give better policy gradients Year: (2022)
Ref_id:b60 Title: Private fine-tuning of large language models with zeroth-order optimization Year: (2024)
Ref_id:b61 Title: Simultaneous computation and memory efficient zeroth-order optimizer for fine-tuning large language models Year: (2024)
Ref_id:b62 Title: Zo2: Scalable zeroth-order fine-tuning for extremely large language models with limited gpu memory Year: (2025)
Ref_id:b63 Title: Complexity of finding stationary points of nonconvex nonsmooth functions Year: (2020)
Ref_id:b64 Title: Opt: Open pre-trained transformer language models Year: (2022)
Ref_id:b65 Title: Revisiting zeroth-order optimization for memory-efficient llm fine-tuning: A benchmark Year: (2024)
Ref_id:b66 Title: Towards query-efficient black-box adversary with zeroth-order natural gradient descent Year: (2020)
