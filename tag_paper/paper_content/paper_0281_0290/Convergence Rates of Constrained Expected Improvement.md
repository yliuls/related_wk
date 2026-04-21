Title: Convergence Rates of Constrained Expected Improvement
Abstract: Constrained Bayesian optimization (CBO) methods have seen significant success in black-box optimization with constraints. One of the most commonly used CBO methods is the constrained expected improvement (CEI) algorithm. CEI is a natural extension of expected improvement (EI) when constraints are incorporated. However, the theoretical convergence rate of CEI has not been established. In this work, we study the convergence rate of CEI by analyzing its simple regret upper bound. First, we show that when the objective function f and constraint function c are assumed to each lie in a reproducing kernel Hilbert space (RKHS), CEI achieves the convergence rates of O t -1 2 log d+1 2 (t) and O t -ν 2ν+d log ν 2ν+d (t) for the commonly used squared exponential and Matérn kernels (ν > 1 2 ), respectively. Second, we show that when f is assumed to be sampled from Gaussian processes (GPs), CEI achieves similar convergence rates with a high probability. Numerical experiments are performed to validate the theoretical analysis.

Section: Introduction
Bayesian optimization (BO) is an efficient method for optimizing expensive black-box functions without derivatives. It leverages probabilistic surrogate models, most commonly Gaussian processes (GPs), to balance exploration and exploitation in the search for optimal solutions [Frazier, 2018]. BO has found widespread success in diverse fields such as structural design [Mathern et al., 2021], machine learning hyperparameter tuning [Wu et al., 2019], robotics [Calandra et al., 2016], fusion design [Wang et al., 2024], etc.
While traditional BO is typically applied to unconstrained settings, many real-world problems involve black-box constraints that must be satisfied. This has motivated growing interest in constrained Bayesian optimization (CBO), where surrogate models are also constructed for constraint functions [Bernardo et al., 2011] that are complex and expensive to evaluate, making CBO especially valuable in applications like engineering design [Song et al., 2024] and automated machine learning [Ungredda and Branke, 2024]. One of the very key difference between unconstrained and constrained optimization is that the feasible region for constrained optimization problem consists of the search space where all constraints must be satisfied. A general form of the constrained BO problem is: minimize x∈C f (x), subject to c(x) ≤ 0,
where f : R d → R is the objective function, and c : R d → R m are the constraint functions. Both are defined on a compact input space C ⊂ R d . The objective and the constraint functions are both expensive black-box functions, that can only be evaluated through expensive physical or computer experiments. Throughout this paper, we consider the noise-free setting for both the objective and the constraints, i.e., the function evaluations are deterministic and the true function values can be observed (see Remark 3.14 for discussion on the noisy case). In addition, a single constraint is considered, i.e., m = 1, for simplicity of presentation. We note that our analysis can be easily extended to multiple constraints (see Remark 3.13 for details).
Broadly, CBO methods can be categorized into implicit and explicit approaches [Amini et al., 2025]. Implicit methods modify standard acquisition functions to incorporate constraints via merit functions or feasibility weights. Explicit methods estimate the feasible region directly and restrict search to this region. Among these, the constrained expected improvement (CEI) [Schonlau et al., 1998, Gelbart et al., 2014, Gardner et al., 2014] stands out as one of the most basic and widely adopted methods. CEI is a natural extension of the well-known expected improvement (EI) function [Jones et al., 1998], where the acquisition function is computed as the product of EI and the probability of feasibility. Thanks to this simple and interpretable formulation, CEI has been successfully applied across domains, and it remains one of the default choices in many constrained BO software packages [Balandat et al., 2020].
Despite its empirical popularity, the theoretical understanding of CEI lags behind. In contrast, unconstrained EI has been more extensively studied. Under a frequentist assumption where the objective f lies in a reproducing kernel Hilbert space (RKHS), Bull [2011] established the convergence rate of EI by deriving the simple regret upper bound. Other works explored the density of sampled sequences [Vazquez and Bect, 2010] or connections between EI and optimal computing budget allocation [Ryzhov, 2016]. However, convergence rates (i.e., simple regret upper bound) for CEI have not been rigorously established-neither under frequentist nor under Bayesian settings. Here, Bayesian setting means the objective f is a function sampled from a GP.
Introducing constraints into EI significantly complicates the theoretical analysis. Unlike in the unconstrained case, the algorithm may need to explore infeasible regions to gain information on the constraint boundary. Furthermore, CEI's acquisition function is inherently more complex and nonconvex, posing challenges for analysis. On the other hand, the presence of constraints in CEI leads to changes in the sampling procedure. As a result, the key challenge to study the convergence rate of CEI lies in analyzing the exploration (searching for feasible regions) and exploitation (optimizing within feasible areas) since the feasibility threshold is unknown in the input space.
In this paper, we provide the first theoretical convergence rates for CEI, focusing on simple regret upper bounds under both the frequentist and Bayesian settings. Our convergence rates provide practitioners theoretical assurance for the practical deployment of CEI. We explain the technical challenges and how we address them in Section 3. Our contributions are summarized as follows:
• Under the frequentist setting, we derive simple regret upper bounds of O t -1 2 log d+1 2 (t)
for the squared exponential (SE) kernel and O t -ν 2ν+d log ν 2ν+d (t) for Matérn kernels (ν > 1  2 ). These bounds are improved upon the direct extension of Bull [2011] to the constrained case for SE kernel with d ≥ 3 and Matérn kernels with d ≥ 3, ν ≥ d d-2 . (see Theorem 3.7).
• Under the Bayesian setting for the objective, we achieve similar simple regret upper bounds with high probabilities. These bounds are established based on the newly derived bounds (see Theorem 3.11) on the difference between the improvement function and its corresponding EI in the Bayesian setting.
This paper is organized as follows. In Section 2, we describe the basics and preliminaries of BO, including the CEI algorithm. In Section 3, the simple regret upper bounds of CEI are established in both settings. Numerical experiments to validate the theoretical results are given in Section 4. Conclusions are made in Section 5. All proof details are presented in the appendix.
this section cite: ['b8', 'b20', 'b33', 'b6', 'b3', 'b25', 'b27', 'b0', 'b24', 'b10', 'b9', 'b15', 'b2', 'b5', 'b30', 'b23', 'b5']

Section: Background
CBO mainly consists of two components: the GP surrogates for the black-box objective function f and constraint function c, and the constrained acquisition function as the sequential sampling rule guiding for the global optimum.
this section cite: []

Section: Gaussian process models for f and c
Without losing generality, let the mean function for the objective GP model prior be 0 and the covariance function (kernel) be k f (x, x ′ ) : R n × R n → R. At sample point x t ∈ C, we denote the objective function value as f (x t ) and the observed constraint function value is c(x t ). Given t sample points, denote x 1:t = [x 1 , . . . , x t ] and f 1:t = [f (x 1 ), . . . , f (x t )]. Moreover, denote the t × t covariance matrix
K f t = [k f (x 1 , x 1 ), . . . , k f (x 1 , x t ); . . . ; k f (x t , x 1 ), . . . , k f (x t , x t )]. The posterior distribution of f (x)|x 1:t , f 1:t ∼ N (µ f t (x), (σ f t (x))
2 ) can then be inferred using Bayes' rule as follows
µ f t (x) = (k f t (x)) T (K f t ) -1 f 1:t , (σ f t ) 2 (x) = k f (x, x) -(k f t (x)) T (K f t ) -1 k f t (x) ,(2)
where
k f t (x) = [k f (x 1 , x), . . . , k f (x t ,
x)] T . Similarly, denote the kernel for c as k c : R n ×R n → R and the covariance matrix
K c t = [k c (x 1 , x 1 ), . . . , k c (x 1 , x t ); . . . ; k c (x t , x 1 ), . . . , k c (x t , x t )]. The posterior distribution for c is µ c t (x) = (k c t (x)) T (K c t ) -1 c 1:t , (σ c t ) 2 (x) = k c (x, x) -(k c t (x)) T (K c t ) -1 k c t (x)
, where k c t (x) = [k c (x 1 , x), . . . , k c (x t , x)] T , and µ c t (x) and (σ c t ) 2 (x) are the posterior mean and variance for c, respectively. Here we use the subscripts f , c and superscripts f , c to distinguish between GPs for f and c. Choices of the kernels k f and k c include the SE and Matérn kernels, which are among the most popular kernels for GP and BO. Their definitions are as follows.
k SE (x, x ′ ) = exp - r 2 2l 2 , k M atérn (x, x ′ ) = 1 Γ(ν)2 ν-1 √ 2νr l ν B ν √ 2νr l ,
where l > 0 is the length hyper-parameters, r = ∥x -x ′ ∥ 2 , ν > 0 is the smoothness parameter of the Matérn kernel, and B ν is the modified Bessel function of the second kind.
this section cite: []

Section: Constrained Expected Improvement
Acquisition functions are critical to the performances of BO algorithms. In the unconstrained setting, one of the most widely adopted acquisition functions is EI [Jones et al., 1998]. Given t samples, the improvement function of f used in EI is defined as
I f t (x) = max{f + t -f (x), 0},(3)
where
f + t = min i=1,...,t f (x i ).
The expectation of (3) conditioned on existing samples is EI, which has a closed form Brochu et al. [2010]:
EI f t (x) = (f + t -µ f t (x))Φ(z f t (x)) + σ f t (x)ϕ(z f t (x)),(4)
where
z f t (x) = f + t -µ f t (x) σ f t (x)
. The functions ϕ and Φ are the probability density function (PDF) and the cumulative distribution function (CDF) of the standard normal distribution, respectively. The t + 1th sample using EI is chosen by
x t+1 = argmax x∈C EI f t (x).(5)
Taking into account the constraint, the constrained improvement function in CEI [Gardner et al., 2014] is defined as
I C t = ∆ c t (x) max{f + t -f (x), 0},(6)
where ∆ c t ∈ {0, 1} is the feasibility indicator function where ∆ c t (x) = 1 if c(x) ≤ 0 and ∆ c t (x) = 0 otherwise. The incumbent f + t in CEI is augmented to be the best feasible observation. CEI assumes that f and c are conditionally independent [Gardner et al., 2014]. Taking the conditional expectation of (6), the CEI function is
EI C t (x) = P t (x)EI f t (x) = Φ - µ c t (x) σ c t (x) EI f t (x),(7)
where P t (•) is the probability of feasibility (POF) function for c(x) ≤ 0. CEI chooses the next sample via
x t+1 = argmax x∈C P t (x)EI f t (x).(8)
The CEI algorithm is given in Algorithm 1.
Algorithm 1 CEI algorithm
1: Choose k f (•, •), k c (•, •), and T 0 initial samples x i , i = 1, . . . , T 0 . Observe f 1:T0 and c 1:T0 . 2: Train the GP surrogate models for f and c respectively conditioned on the initial observations. 3: for t = T 0 + 1, T 0 + 2, . . . do 4:
Find x t+1 based on (8) (CEI).
5:
Observe f (x t+1 ) and c(x t+1 ).
this section cite: ['b15', 'b9', 'b9']

Section: 6:
Update the GP models with the addition of x t+1 , f (x t+1 ), and c(x t+1 ).
this section cite: []

Section: 7:
if Evaluation budget exhausted then 8:
Exit CEI can be extended to multiple constraints assuming conditional independence among the constraints [Gardner et al., 2014]. Our derived convergence rates can also be readily extended to multiple constraints, as we explain in Remark 3.13.
this section cite: ['b9']

Section: Convergence rates of CEI
We present our main results of convergence rates for CEI by establishing the simple regret upper bounds. Denote the optimal solution to the constrained optimization problem (1) as x * . In the unconstrained case, the simple regret of EI is defined as f + t -f (x * ) [Bull, 2011]. In the constrained case, we use the current best feasible observation and compare it to the optimal solution f (x * ), since one could have an infeasible sample point with smaller objective than f (x * ). Given that f + t is already defined as the best feasible observation till iteration t in CEI, we continue to use
r t = f + t -f (x * ),(9)
as the simple regret for CEI. In our analysis, we make the same underlying assumption as CEI that f + t exists. In the following, we first establish the convergence rate under the frequentist assumptions in Section 3.1, including an improved version of the rate under frequentist assumptions in Section 3.1.1. Then, we establish the convergence rate under Bayesian objective assumptions in Section 3.2.
this section cite: ['b5']

Section: Simple regret upper bound under frequentist assumptions
In this section, we present the simple regret upper bound for CEI under the frequentist setting. Moreover, by adopting the information theory-based bounds and techniques in the noise-free cumulative regret bound of upper confidence bound (UCB) [Lyu et al., 2019], we can derive an improved upper bound in some cases compared to Bull [2011]. The definition of RKHS is given below. Definition 3.1. Let k be a positive definite kernel k : X × X → R with respect to a finite Borel measure supported on X . A Hilbert space H k of functions on X with an inner product ⟨•, •⟩ H k is called a RKHS with kernel k if k(•, x) ∈ H k for all x ∈ X , and ⟨f, k(•, x)⟩ H k = f (x) for all x ∈ X , f ∈ H k . The induced RKHS norm ∥f ∥ H k = ⟨f, f ⟩ H k measures the smoothness of f with respect to k.
In this section, we assume the following assumptions on the functions f and c.
r t ≤ c τ B Φ(-B c ) B f 4 t -2 + (0.4 + B f )σ f t k (x t k +1 ) ,(10)
for some t k ∈ [
this section cite: ['b19', 'b5']

Section: Sketch of Proof for Theorem 3.3.
We start by noticing that the sum of the difference between consecutive best feasible observations is bounded, i.e.,
T t=1 f + t-1 -f + t ≤ 2B f . Then, we adopt a technique in Bull [2011] to find t k such that f + t k -f + t k +1 ≤ 2B f k , where k ≤ t k ≤ 2k and 2k ≤ t ≤ 2(k + 1). Next, using the monotonicity of f + t , r t is bounded by r t k . Using the inequality between I f t and EI f t in Lemma B.4, we can bound r t k by the EI on objective: EI f t k (x * ). Then, we transform EI f t k (x * ) into EI f t k (x t k +1 ) by inserting the term P t k (x * ), taking advantage of the multiplicative structure of CEI. The upper bound of r t then consists of the term 1 Pt k (x * ) EI f t k (x t k +1 ). From the confidence interval |f (x) -µ f t (x)| (Lemma B.2) and the fact that f + t k -f + t k +1 ≤ 2B f k , we can bound EI f t k (x t k +1 ). The constraint term 1 Pt k (x * ) remains to be bounded. We use the confidence interval on |c(x) -µ c t (x)| in Lemma B.2 at x * and the fact that x * is a feasible solution to obtain a lower bound for P t k (x * ). This concludes the proof. Remark 3.4 (Constraint in the simple regret upper bound). The terms derived from the constraint function in (10) is 1 Φ(-Bc) , which emerges from the probability of feasibility function and µ c t (x) and σ c t (x) of the GP model of c(x). Thanks to the multiplicative structure between the objective and constraint in I C t (6) and EI c t (7), the simple regret upper bound maintains a similar form.
It is clear from (10) that the convergence of r t relies on the posterior standard deviation σ f t (x t+1 ). Since t k increases with t, as σ f t (x t+1 ) → 0, so does σ f t k (x t k +1 ). In the noise-free setting, the posterior variance can be bounded via the maximum distance between sample points and a given point. To obtain the rate of simple regret bound, we use Assumptions (1)-(4) in Bull [2011] and focus on squared exponential (SE) and Matérn kernels. Recall that the smoothness parameter of the Matérn kernel is ν > 0. Both the SE and Matérn kernels satisfy Assumptions (1)-(4) in Bull [2011], with SE kernel obtained as ν → ∞. Further, define
η = α, ν ≤ 1 0, ν > 1,(11)
where α = 1 2 if ν ∈ N, and α = 0 otherwise. Then, for SE and Matérn kernels, σ f t k (x t k +1 ) can be bounded with the following lemma.
Lemma 3.5 (Bull [2011]). For the SE kernel, there exists constant C ′ > 0 so that given ∀t ∈ N,
σ f i (x i+1 ) ≥ C ′ k -1 d (12
)
holds for at most k times, for ∀k ∈ N, k ≤ t and i = 1, . . . , t -1. For Matérn kernels,
σ f i (x i+1 ) ≥ C ′ k -min{ν,1} d log η (k)(13)
holds at most k times.
In the constrained setting, we are able to obtain the same rates as those in the unconstrained case [Bull, 2011] using Lemma 3.5. Corollary 3.6. Under Assumption 3.2, the CEI algorithm leads to the convergence rates of
O t -1 d and O t -min{ν,1} d log η (t) ,(14)
for SE and Matérn kernels, respectively, where η is from (11).
Corollary 3.6 shows that the CEI algorithm is guaranteed to find the best feasible point asymptotically with the rates elaborated in (14). Also, we point out that the choice of kernels and their parameters affect the convergence rates. Since the SE kernel can be viewed as a Matérn kernel with ν → ∞, its convergence rate is better than Matérn kernels with ν ≤ 1. However, due to the limitations of the kernel analysis in Bull [2011] (see Remark 3.8), for ν ≥ 1, SE and Matérn kernels have similar convergence rates in Corollary 3.6. As we present in the following section, improved rates for both kernels can be obtained in some cases.
this section cite: ['b5', 'b5', 'b5', 'b5', 'b10', 'b13']

Section: Improved simple regret upper bound under frequentist assumptions
Next, we apply maximum information gain and the corresponding information theory to obtain improved simple regret upper bounds. Theorem 3.7. Under Assumption 3.2, the CEI algorithm leads to the improved convergence rates of
O t -1 2 log d+1 2 (t) and O t -ν 2ν+d log ν 2ν+d (t) ,(15)
for SE and Matérn kernels, respectively.
this section cite: []

Section: Sketch of Proof for Theorem 3.7.
The proof follows similar steps to that of Theorem 3.3 but further bounds σ f t k (x t k +1 ) using γ f t . To do so, we first recognize that the bound using γ f t (Lemma A.3) is established in the noisy case where the posterior standard deviation has a different form as in (18). Using Lemma A.4, we can establish that the noise-free posterior standard deviation also satisfies t-1 i=0 σ f i (x i+1 ) ≤ C γ tγ f t . Then, from Lemma A.5, we can find a small enough σ f i (x i+1 ). Specifically, choose k = [t/3], where [x] denotes the largest integer smaller than x. Thus, we have 3k ≤ t ≤ 3(k + 1). Then, there exists k ≤ t k ≤ 3k such that
f + t k -f + t k +1 ≤ 2B f k and σ f t k (x t k +1 ) ≤ √ tγ f t k .
The rest of the proof follows from that of Theorem 3.7.
Remark 3.8 (Improved rate of convergence). As mentioned above, the rates in Corollary 3.6 are the same as the known convergence rates for EI in Bull [2011]. Meanwhile, the rates in Theorem 3.7 is an improvement over those of Bull [2011] for SE kernel with d ≥ 3 and Matérn kernels with
d ≥ 3, ν ≥ d d-2 .
To achieve this, we applied techniques from regret bound analysis on noise-free UCB [Lyu et al., 2019] that allows us to use maximum information gain to bound the sum of σ f t (x t+1 ). Then, we use our techniques in the proof of Theorem 3.3 to bound an individual
σ f t k (x t k +1 ). In Bull [2011], the σ f t k (x t k +1
) is bounded by the Taylor expansion of the kernel functions. Therefore, the rates of decrease are limited to quadratic terms for both SE and Matérn kernels, since their Taylor expansions around 0 for ∥x -x ′ ∥ 2 are quadratic at best. On the other hand, maximum information gain can lead to tight bounds on γ f t that take advantages of the spectral properties of the kernels [Vakili et al., 2021, Iwazaki, 2025]. Hence, using γ f t to bound σ f t k (x t k +1 ) can produce a faster rate. As the open question raised in Vakili [2022] gets answered, further improvement of the convergence rates is possible, e.g., using techniques from Iwazaki [2025].
this section cite: ['b17', 'b5', 'b5', 'b19', 'b29', 'b28']

Section: Simple regret upper bound under Bayesian objective assumption
In this section, we present the simple regret upper bound for CEI under the Bayesian objective assumptions. We again use the maximum information gain to derive the simple regret upper bound. Assumption 3.9. The bound constraint set C ⊂ [0, r] d is compact and convex. The objective function f is sampled from GP (0, k f (x, x ′ )). Further, the objective function f is assumed to be Lipschitz continuous ( of 1 In the remaining of this section we will work under Assumption 3.9.
this section cite: []

Section: Technical Challenges under Bayesian Assumptions.
In addition to the challenges in the frequentist setting, the bounds on EI in the Bayesian setting are not available in current literature, to the best of our knowledge. Starting from the confidence interval on |f (x) -µ f t (x)|, we derive the bounds on |I f t (x) -EI f t (x)| with high probability, an important step towards the bound on r t . Noticeably, under the Bayesian setting, the bounds are satisfied with a given probability, e.g., 1 -δ, where δ ∈ (0, 1).
The simple regret upper bound is given in the following theorem.
Theorem 3.10. Let β = 2 log(6c α /δ) and β t = 2 log(3π t /δ), where c α = 1+2π
2π and π t = π 2 t 2 6 . Under Assumption 3.9, the CEI algorithm leads to the simple regret upper bound
r t ≤c τ (β) 1 Φ(-B c ) 4M f t -2 + 2β 1/2 t t -2 C γ tγ f t + (0.4 + β 1/2 )σ f t k (x t k +1 ) ,(16)
for some t k ∈ [ t 2 -1, t], c τ (β) = τ (β 1/2 ) τ (-β 1/2 )
, and constant M f > 0 with probability ≥ 1 -δ.
The constant M f is from Lemma C.1. The convergence rate is given in the next theorem.
Theorem 3.11. Let β = 2 log(6c α /δ) and β t = 2 log(3π t /δ), where c α = 1+2π 2π and π t = π 2 t 2 6 . Under Assumption 3.9, the CEI algorithm leads to the convergence rates of
O t -1 2 log d+2 2 (t) and O t -ν 2ν+d log 2ν+0.5d 2ν+d (t) ,(17)
for SE and Matérn kernels, respectively, with probability ≥ 1 -δ.
this section cite: []

Section: Sketch of Proof for Theorem 3.10.
Recall that f and c are assumed conditionally independent in CEI. We start from the bound on the confidence interval for f : |f (x) -µ f t (x)| ≤ β 1/2 σ f t (x), with probability ≥ 1 -δ, where β = 2 log(1/δ), as in Lemma C.2. The confidence interval of c remains the same as in the frequentist setting. These are well-known results [Srinivas et al., 2009]. Then, we derive the subsequent bounds
|I f t (x) -EI f t (x)| ≤ √ βσ f t (x), where β = max{1.44, 2 log(c α /δ)} and c α = 1+2π
2π with probability ≥ 1-δ (Lemma C.5). Then, we prove the relationship in Lemma C.6 that
I f t (x) ≤ τ ( √ β) τ (- √ β) EI f t (x) with probability ≥ 1 -δ.
We can now follow the general analysis framework in Section 3.1 and Theorem 3.3 to obtain the simple regret upper bound under Bayesian objective assumptions, while choosing t k with a more defined criterion. Remark 3.12 (Comparison to the frequentist setting). Comparing Theorem 3.7 to Theorem 3.11, the convergence rates in the frequentist and Bayesian settings are the same except for a log 1/2 (t) term. This is partially because simple regret focuses on the best feasible solution f + t and thus many of the parameters in Theorem 3.3 and 3.10 do not depend on t. Remark 3.13 (Multiple constraints). As mentioned in Section 1, our results can be readily applied to CEI with multiple constraints for both frequentist and Bayesian settings. Consider m constraints c i (x) ≤ 0, i = 1, . . . , m. Assuming conditional independence of the constraints, the CEI function
is EI C t (x) = Π m i=1 P i t (x)EI f t (x) = Π m i=1 Φ -µ c i t (x) σ c i t (x) EI f
t (x), where P i t is the probability of feasibility function of constraint c i (x) ≤ 0, and µ ci t (x) and σ ci t (x) are the posterior mean and standard deviation for c i , respectively. By making the assumption that each constraint function lies in its corresponding RKHS of the kernel k ci , we have
|c i (x) -µ ci t (x)| ≤ B ci σ ci t (x),
where B ci is the upper bound of RKHS norm associated with kernel k ci and function c i . We can then apply the analysis framework in this paper to obtain an upper bound similar to that of Theorem 3.3, where the term 1 Φ(-Bc) is replaced with Π m i=1 1 Φ(-Bc i ) . We note that in the Bayesian objective setting, to ensure probability 1 -δ, the parameter β needs to increase with the number of constraints as well, e.g., β = 2 log((m + 5)c α /δ). Remark 3.14 (Extension to the noisy setting). Extending our analysis to the noisy setting is non-trivial, and we discuss the associated challenges for noisy objective and constraint functions separately. A noisy constraint function introduces additional complications in defining feasibility. If only noisy observations of the constraint values are available, the notion of a feasible sample and the definition of f + t becomes ambiguous. As a result, major modifications to the CEI algorithm are required to appropriately handle the uncertainty introduced by noise.
For the noisy objective function, CEI can be adapted similarly to the noisy EI formulation by treating the best feasible noisy observation as the incumbent. However, to the best of our knowledge, a theoretical guarantee on the simple regret bound for the noisy unconstrained setting remains unavailable. Recent work by Wang et al. [2025] provides a framework for deriving noisy simple regret bounds based on the best observed value, r s t = y + t -f (x t ), which can be extended to CEI. Specifically, by defining r s t as the simple regret for CEI with y + t denoting the best feasible noisy observation, a similar proof strategy as in Theorem 3.10 yields an analogous upper bound. In the Bayesian setting with i.i.d. Gaussian noise on the objective and noise-free constraint observations, the convergence rate of the upper bound on r s t can be obtained. However, we note that given the noise, r s t is possibly negative. Remark 3.15 (Infeasible initial sample). It is well known that CEI requires initial feasible sample [Gardner et al., 2014]. That is, f + t exists from the initial samples so that the CEI calculation can proceed. Methods proposed to address this issue typically employ separate strategies when no feasible samples are available and revert to the standard CEI formulation once feasibility is established [Lin et al., 2024, Letham et al., 2019]. In addition, introducing a tolerance parameter in the constraint can further mitigate this problem by allowing near-feasible points when the degree of violation is small. Remark 3.16 (Tolerance in constraints). In gradient-based optimization methods, a tolerance for constraint violation is often used to improve the performance and flexibility of algorithms Wächter and Biegler [2006], Nocedal and Wright [2006]. Motivated by this, we introduce a tolerance parameter λ ≥ 0, where a point x is considered feasible if c(x) ≤ λ and infeasible otherwise. The corresponding CEI with tolerance is defined as
EI C t (x, λ) = P t (x, λ)EI f t (x) = Φ λ-µ c t (x) σ c t (x) EI f t (x).
Clearly, the standard CEI formulation is recovered when λ = 0.
The simple regret bound is affected by λ and should lead to
this section cite: ['b26', 'b32', 'b9', 'b17', 'b16', 'b34', 'b21']

Section: As the sample iteration increases, the inclusion of σ c t k (x *
) is important in balancing -B c that can lead to a large simple regret upper bound. We explain the intuition below. As t → ∞, t k → ∞ and k → ∞. We know σ f t (x t+1 ) → 0, and hence σ f t k (x t k +1 ) → 0 and r t → 0. That is, the simple regret upper bound of CEI with λ > 0 converges. Thus, x t approaches at least one of the optimal solutions. Suppose without losing generality, x t → x * . Then, by definition σ c t k (x * ) → 0. Consequently, we should have λ
σ c t k (x * ) → ∞ for λ > 0. Then, we have Φ λ σ c t k (x * ) → 1. Therefore, 1/Φ( λ σ c t k (x * ) -B c
) → 1 and B c does not affect the simple regret upper bound asymptotically. We note that the convergence rate of CEI with tolerance remains similar since it is dominated by the maximum information gain of f .
this section cite: []

Section: Numerical experiments
Although this paper is primarily theoretical, we conduct numerical experiments to support the theoretical results. We apply the CEI algorithm to eight synthetic problems that are randomly generated from RKHS of kernels and GP priors, and five benchmark problems commonly used in the CBO literature. These numerical experiments are not intended to demonstrate superior performance over the state-of-the-art CBO algorithms. Instead, they serve as empirical evidence for the theoretical analysis presented in this work. All experiments are conducted on M1 (16GB memory) 1 .
this section cite: []

Section: Synthetic problems
In this section, we study objective and constraint functions drawn from reproducing kernel Hilbert spaces (RKHSs) as well as from Gaussian process priors with Matérn (ν = 2.5) and squared exponential (SE) kernels, across input dimensions d ∈ 2, 4. The domain is the hypercube [0, 1] d . For RKHS cases (the frequentist setting), the functions are generated with a similar approach to Chowdhury and Gopalan [2017]. Specifically, both objective f (x) and constraint functions c(x) are generated by sampling from the RKHS associated with a chosen kernel (Matérn/SE kernels with a length scale of 0.2). Each function is constructed as a weighted sum of kernel evaluations at 100 randomly selected basis points, with weights drawn from a standard normal distribution. Formally, the function takes the form f (x) = n i=1 α i k(x, X i ) , where k is the kernel, X i are basis points, and α i are random coefficients; c(x) is generated similarly. For the GP cases (the Bayesian setting), the functions are generated with an approach similar to Srinivas et al. [2009]. Specifically, we uniformly choose 1000 points in the design space and sample randomly from a multivariate Gaussian distribution defined by the GP prior with the chosen kernel.
For each synthetic problem, we conducted 100 independent trials. The number of initial design is set to 10d, and 50 optimization iterations were performed for all cases. We plotted the log-log curve of simple regret against the number of iterations in Figure 1. In all cases, we consistently observed sublinear convergence patterns, which align well with our theoretical guarantees.  4.2 Test problems
this section cite: []

Section: Conclusions
In this paper, we studied the simple regret upper bounds of the CEI algorithm, one of the most widely adopted CBO methods. Under both frequentist setting and Bayesian objective assumptions, we establish for the first time the convergence rates for CEI. Our results provide theoretical support and validation for the empirical success of CEI.
this section cite: []

Section: References
Ref_id:b0 Title: Constrained bayesian optimization: A review Year: (2025)
Ref_id:b1 Title: ADMMBO: Bayesian optimization with unknown constraints using ADMM Year: (2019)
Ref_id:b2 Title: BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization Year: (2020)
Ref_id:b3 Title: Optimization under unknown constraints Year: (2011-10)
Ref_id:b4 Title: A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning Year: (2010-12)
Ref_id:b5 Title: Convergence rates of efficient global optimization algorithms Year: (2011)
Ref_id:b6 Title: Bayesian optimization for learning gaits under uncertainty Year: (2016-02)
Ref_id:b7 Title: On kernelized multi-armed bandits Year: (2017)
Ref_id:b8 Title: Bayesian optimization Year: (2018-10)
Ref_id:b9 Title: Bayesian optimization with inequality constraints Year: (2014)
Ref_id:b10 Title: Bayesian optimization with unknown constraints Year: (2014)
Ref_id:b11 Title: Modeling an augmented lagrangian for blackbox constrained optimization Year: (2016)
Ref_id:b12 Title: Predictive entropy search for Bayesian optimization with unknown constraints Year: (2015)
Ref_id:b13 Title: Gaussian process upper confidence bound achieves nearly-optimal regret in noisefree gaussian process bandits Year: (2025)
Ref_id:b14 Title: Improved regret bounds for gaussian process upper confidence bound in bayesian optimization Year: (2025)
Ref_id:b15 Title: Efficient global optimization of expensive black-box functions Year: (1998)
Ref_id:b16 Title: Constrained Bayesian optimization with noisy experiments Year: (2019)
Ref_id:b17 Title: A multi-fidelity bayesian optimization approach for constrained multi-objective optimization problems Year: (2024)
Ref_id:b18 Title: No-regret bayesian optimization with unknown equality and inequality constraints using exact penalty functions Year: (2022)
Ref_id:b19 Title: Efficient batch black-box optimization with deterministic regret bounds Year: (2019)
Ref_id:b20 Title: Multi-objective constrained Bayesian optimization for structural design Year: (2021-02)
Ref_id:b21 Title: Numerical Optimization Year: (2006)
Ref_id:b22 Title: Bayesian optimization under mixed constraints with a slack-variable augmented lagrangian Year: (2016)
Ref_id:b23 Title: On the convergence rates of expected improvement methods Year: (2016)
Ref_id:b24 Title: Global versus local search in constrained optimization of computer models. Lecture notes-monograph series Year: (1998)
Ref_id:b25 Title: Constrained bayesian optimization algorithms for estimating design points in structural reliability analysis Year: (2024)
Ref_id:b26 Title: Gaussian process optimization in the bandit setting: No regret and experimental design Year: (2009)
Ref_id:b27 Title: Bayesian optimisation for constrained problems Year: (2024-04)
Ref_id:b28 Title: Open problem: Regret bounds for noise-free kernel-based bandits Year: (2022)
Ref_id:b29 Title: On information gain and regret bounds in gaussian process bandits Year: (2021)
Ref_id:b30 Title: Convergence properties of the expected improvement algorithm with fixed mean and covariance functions Year: (2010)
Ref_id:b31 Title: A multifidelity bayesian optimization method for inertial confinement fusion design Year: ()
Ref_id:b32 Title: On the convergence of noisy bayesian optimization with expected improvement Year: (2025)
Ref_id:b33 Title: Hyperparameter optimization for machine learning models based on bayesian optimization Year: (2019)
Ref_id:b34 Title: On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming Year: (2006)
Ref_id:b35 Title: On kernelized multi-armed bandits with constraints Year: (2022)
