Title: Improved Regret Analysis in Gaussian Process Bandits: Optimality for Noiseless Reward, RKHS norm, and Non-Stationary Variance
Abstract: We study the Gaussian process (GP) bandit problem, whose goal is to minimize regret under an unknown reward function lying in some reproducing kernel Hilbert space (RKHS). The maximum posterior variance analysis is vital in analyzing nearoptimal GP bandit algorithms such as maximum variance reduction (MVR) and phased elimination (PE). Therefore, we first show the new upper bound of the maximum posterior variance, which improves the dependence of the noise variance parameters of the GP. By leveraging this result, we refine the MVR and PE to obtain (i) a nearly optimal regret upper bound in the noiseless setting and (ii) regret upper bounds that are optimal with respect to the RKHS norm of the reward function. Furthermore, as another application of our proposed bound, we analyze the GP bandit under the time-varying noise variance setting, which is the kernelized extension of the linear bandit with heteroscedastic noise. For this problem, we show that MVR and PE-based algorithms achieve noise variance-dependent regret upper bounds, which match our regret lower bound.Ω(ln T ) Ω(1) N/A (Vakili, 2022) ness of the underlying reward function depending on the kernel. We focus on the following SE kernel k SE and Matérn kernel k Matérn that are commonly used in the GP bandit:

Section: Introduction
The Gaussian process (GP) bandits (Srinivas et al., 2010) is a powerful framework for sequential decision-making tasks to minimize regret defined by a black-box reward function, which belongs to known reproducing kernel Hilbert space (RKHS). The applications include many fields such as robotics (Berkenkamp et al., 2021), experimental design (Lei et al., 2021), and hyperparameter tuning task (Snoek et al., 2012).
Many existing studies have been conducted to obtain the 1 MI-6 Ltd. Tokyo, Japan 2 Department of Mechanical Engineering, Nagoya University, Aichi, Japan. Correspondence to: Shogo Iwazaki <shogo.iwazaki@gmail.com>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). theoretical guarantee for the regret. Established work by Srinivas et al. (2010) has shown the upper bounds of the cumulative regret for the GP upper confidence bound (GP-UCB) algorithm. Furthermore, Valko et al. (2013) have shown a tighter regret upper bound for the SupKernelUCB algorithm. Scarlett et al. (2017) have shown the lower bound of the regret, which implies that the regret upper bound from (Valko et al., 2013) is near-optimal; that is, the regret upper bound matches the lower bound except for the poly-logarithmic factor. Then, several studies further tackled obtaining a near-optimal GP-bandit algorithm. Vakili et al. (2021a) have proposed maximum variance reduction (MVR), which is shown to be near-optimal for the simple regret incurred by the last recommended action. Furthermore, Li & Scarlett (2022) have shown that phased elimination (PE) is near-optimal for the cumulative regret. The regret analysis of MVR and PE heavily depends on the upper bound for the maximum posterior variance.
We derive the upper bound of the maximum posterior variance in Section 3, by which we tackle tightening the regret upper bound in the settings where room for improvement remains. Our contributions are summarized as follows:
1. In Section 3, we obtain the upper bound of the maximum posterior variance (Lemma 3.1 and Corollary 3.2). Our proposed bound is tighter than the existing bound when the noise variances approach zero.
2. In Section 4, we analyze the GP bandit under the noiseless setting. We show a novel result that PE achieves the cumulative regret upper bound that matches the conjectured lower bound shown by Vakili (2022) under common assumptions in the GP bandit literature. Furthermore, we prove that MVR achieves the exponentially converging and near-optimal simple regret upper bounds for squared exponential (SE) and Matérn kernels, respectively. These results are summarized in Tables 12.
3. In Section 5, we show that the modified PE-and MVRstyle algorithms achieve the near-optimal cumulative and simple regret upper bounds with respect to the RKHS norm upper bound of the reward function under several conditions. These results are summarized in Tables 345.
4. In Section 6, we analyze the GP-bandit problem with the non-stationary noise variance, which is the kernelized extension of the linear bandit with heteroscedastic noise (Zhou et al., 2021). We first study the regret lower bound. Then, we show that the modified PE-and MVR-style algorithms achieve the near-optimal cumulative and simple regret upper bounds, respectively. To our knowledge, our analyses are the first for this setting, though the non-stationary noise is a frequently faced problem.
this section cite: ['b25', 'b1', 'b15', 'b24', 'b25', 'b31', 'b23', 'b31', 'b35']

Section: Related Works
The theoretical assumption of the GP bandit is twofold: Bayesian setting (Srinivas et al., 2010;De Freitas et al., 2012;Russo & Van Roy, 2014;Scarlett, 2018;Takeno et al., 2023;2024) where the reward function follows GPs, and the frequentist setting, where the reward function lies in a known RKHS (Srinivas et al., 2010;Chowdhury & Gopalan, 2017;Vakili et al., 2021a;Li & Scarlett, 2022). Although this paper concentrates on deriving the regret upper bound for the frequentist setting, our Lemma 3.1 and Corollary 3.2 are versatile and can be applied to the Bayesian setting.
Many GP bandit algorithms have been proposed in the frequentist setting (for example, Srinivas et al., 2010;Valko et al., 2013;Chowdhury & Gopalan, 2017;Janz et al., 2020;Vakili et al., 2021a;Li & Scarlett, 2022). Although several existing methods (Valko et al., 2013;Janz et al., 2020;Camilleri et al., 2021;Salgia et al., 2021;Li & Scarlett, 2022) achieve near-optimal regret upper bounds for the ordinary GP bandit setting as summarized in (Li & Scarlett, 2022), we develop PE-and MVR-style algorithms due to their simplicity. On the other hand, although these existing methods are near-optimal regarding the time horizons, the optimality regarding the RKHS norm of the reward function has not been shown as summarized in Tables 345.
The regret analyses are also conducted on the noiseless setting (Bull, 2011;Lyu et al., 2019;Vakili, 2022;Salgia et al., 2024;Kim & Sanz-Alonso, 2024;Flynn & Reeb, 2025). Regarding the cumulative regret, we obtained a tighter upper bound for both SE and Matérn kernels than existing results without the additional assumption for the reward function like Assumption 4.2 in (Salgia et al., 2021). Regarding the simple regret, Kim & Sanz-Alonso (2024) have shown that the random sampling-based algorithm achieves the knownbest regret upper bound in terms of the expectation regarding the algorithm's randomness. Compared with this result, we show the regret upper bounds that always hold with the deterministic MVR-style algorithm. In particular, the regret upper bound is tighter for the Matérn kernel than that from (Kim & Sanz-Alonso, 2024). Tables 1-2 summarize the comparison.
Compared with the regret upper bound, the analysis for the regret lower bound is limited (Bull, 2011;Scarlett et al., 2017;Cai & Scarlett, 2021;Vakili, 2022). From these results, we will confirm the optimality of the GP bandit algorithms in Sections 4 and 5. In Section 6, our regret lower bound for the non-stationary noise variance setting is directly obtained from the proofs of (Bull, 2011;Scarlett et al., 2017).
The linear bandit with heteroscedastic noise, where the noise variance is non-stationary with respect to the time horizons, has been studied (Zhou et al., 2021;Zhang et al., 2021;Kim et al., 2022;Zhou & Gu, 2022;Zhao et al., 2023). These studies aim to obtain the noise variance-dependent regret upper bound, characterized by the sum of noise variances.
To our knowledge, the kernelized extension of this setting has not been investigated. Furthermore, as discussed in Section 6, the direct extension from the linear bandit methods is not near-optimal.
this section cite: ['b25', 'b7', 'b19', 'b22', 'b26', 'b25', 'b5', 'b25', 'b31', 'b5', 'b10', 'b31', 'b10', 'b4', 'b20', 'b2', 'b17', 'b11', 'b8', 'b20', 'b11', 'b11', 'b2', 'b23', 'b3', 'b2', 'b23', 'b35', 'b32', 'b12', 'b34', 'b33']

Section: Preliminaries
Problem Setting. Let f : X → R be an unknown reward function with compact input domain X ⊂ R d . At each step t, a learner chooses the query point x t ∈ X ; after that, the learner observes the corresponding reward y t := f (x t ) + ϵ t , where ϵ t is a mean-zero random variable. Under this setup, the learner's goal is to minimize the following cumulative regret R T or the simple regret r T :
R T = t∈[T ] f (x * ) -f (x t ),(1)
r T = f (x * ) -f ( x T ),(2)
where [T ] = {1, . . . , T } and x * ∈ arg max x∈X f (x). Furthermore, x T ∈ X is the estimated maximizer, which is returned by the algorithm at the end of step T .
this section cite: []

Section: Regularity Assumptions.
To construct an algorithm, we leverage the following assumptions. Assumption 2.1 (Smoothness of f ). Assume that f be an element of RKHS H k with bounded RKHS norm ∥f ∥ H k ≤ B < ∞. Here, H k and ∥f ∥ H k respectively denote RKHS and its norm endowed with a known positive definite kernel k : X × X → R. Furthermore, we assume k(x, x) ≤ 1 holds for all x ∈ X . Assumption 2.2 (Assumption for noise). The noise sequence (ϵ t ) t∈N+ is mutually independent. Furthermore, assume that ϵ t is a sub-Gaussian random variable with variance proxy ρ t ≥ 0; namely, E[exp(λϵ t )] ≤ exp(λ 2 ρ 2 t /2) holds for all λ ∈ R.
In existing works (e.g., Srinivas et al., 2010), Assumption 2.1 is the standard assumption for encoding the smooth-Table 1. Comparison between existing noiseless algorithms' guarantees for cumulative regret and our result. In all algorithms, the smoothness parameter of the Matérn kernel is assumed to be ν > 1/2. Furthermore, d, ℓ, ν, and B are supposed to be Θ(1) here. "Type" column shows that the regret guarantee is (D)eterministic or (P)robabilistic. Throughout this paper, the notation O(•) represents the order notation whose poly-logarithmic dependence is ignored.
this section cite: ['b25']

Section: Algorithm
Regret
(SE) Regret (Matérn) Type Remark ν < d ν = d ν > d GP-UCB O T ln d T O T ν+d 2ν+d D (Lyu et al., 2019) (Kim & Sanz-Alonso, 2024) Explore-then-Commit N/A O T √ 2ν∥x -x∥ 2 ℓ ν J ν √ 2ν∥x -x∥ 2 ℓ ,
where ℓ > 0 and ν > 0 are the lengthscale and smoothness parameter, respectively. Furthermore, Γ(•) and J ν denote Gamma and modified Bessel function, respectively. Assumption 2.2 is also common in existing work (e.g., Vakili et al., 2021a;Li & Scarlett, 2022). In Sections 4-5, we consider the stationary noise variance setting whose variance proxy ρ t is fixed over time, while the non-stationary noise variance setting that allows the time-dependent variance proxies ρ t in Section 6.
Gaussian Process. GP is a fundamental kernel-based model that gives both the prediction and its uncertainty quantification of the underlying function. Let GP(0, k) be a mean-zero GP whose covariance is characterized by the kernel function k. In addition, let X := (x 1 , . . . , x t ) and y := (y 1 , . . . , y t ) ⊤ be training input and output data of GP, respectively. Then, under the Bayesian assumption that f follows the GP prior GP(0, k), the posterior distribution of f (x) given X and y is defined as Gaussian distribution, whose mean µ Σ (x; X, y) and variance
σ 2 Σ (x; X) are µ Σ (x; X, y) = k(x, X) ⊤ (K(X, X) + Σ) -1 y, σ 2 Σ (x; X) = k(x, x) -k(x, X) ⊤ (K(X, X) + Σ) -1 k(x, X),
where K(X, X) := [k(x, x)] x, x∈X ∈ R t×t and K(x, X) = [k(x, x)] x∈X ∈ R t respectively represent the kernel matrix and vector defined by x and X. Furthermore, Σ ∈ R t×t is the positive definite variance parameter matrix that defines the noise structure of the observation of GP. Namely, given the input data X, the corresponding outputs y is assumed to be given as y = f (X) + ϵ under the GP modeling, where ϵ ∼ N (0, Σ) and f (X) ∼ N (0, K(X, X)).
We would like to emphasize that the above modeling assumptions (f ∼ GP(0, k) and ϵ ∼ N (0, Σ)) are "fictional" assumptions that are used only for GP modeling, and are distinct from Assumptions 2.1 and 2.2.
this section cite: []

Section: Maximum Information Gain.
The maximum information gain (MIG) is the kernel-dependent complexity parameter used to characterize the regret and confidence bounds in the GP bandits. Given the variance parameter matrix
Σ T := diag(λ 2 1 , . . . , λ 2 T ), the MIG γ T (Σ T ) is defined as γ T (Σ T ) = max X⊂X T I Σ T (f (X), y),(3)
where
I Σ T (f (X), y) := 1 2 ln det(Σ T +K(X,X)) det(Σ T )
denote the mutual information between y and f (X), under the GP modeling assumpsions f (X) ∼ N (0, K(X, X)), y = f (X) + ϵ, and ϵ ∼ N (0, Σ T ). When Σ T = λ 2 I T with some fixed λ > 0 and the identity matrix I T ∈ R T ×T , the upper bound of MIG γ(T, λ 2 ) is known in several commonly used kernels. For example, γ T (λ 2 I T ) ≤ γ(T, λ 2 ) = O(ln d+1 (T /λ 2 )) and γ T (λ
2 I T ) ≤ γ(T, λ 2 ) = O((T /λ 2 ) d 2ν+d (ln(T /λ 2 )) 2ν 2ν+d
) in SE and Matérn kernels with ν > 1/2, respectively (Vakili et al., 2021b). Furthermore, for general Σ T , we can see γ T (Σ T ) ≤ γ T (λ 2 T I T ) with λ 2 T = min t∈[T ] λ 2 t from the data processing inequality (Theorem 2.8.1 of Cover & Thomas, 2006). We show the proof in Appendix B for completeness.
this section cite: ['b6']

Section: Maximum Variance Reduction and Phased Elimination.
MVR is the algorithm that sequentially chooses the most uncertain action x t = argmax x∈X σ 2 Σ (x; X) as shown in Algorithm 2 in Appendix G. PE is the algorithm that combines MVR and candidate elimination as shown in Algorithm 1 in Appendix G. PE divides the time horizons into batches with appropriately designed lengths and performs MVR in each batch. In PE, after each batch, the inputs whose UCB is lower than the maximum of the lower CB are eliminated from the candidates. MVR and PE achieve near-optimal simple and cumulative regret upper bounds, respectively. Due to their simplicity, we analyze MVR-and PE-style algorithms.
this section cite: []

Section: Uniform Upper Bound of Posterior Variance for Maximum Variance Reduction
In this section, we describe the theoretical core result that gives a new upper bound of the posterior variance for the MVR algorithm. Specifically, our result improves the existing upper bound of posterior variance when decreasing noise variance parameters.
Lemma 3.1 (General posterior variance upper bound for MVR). Fix any compact subset X ⊂ X . Then, the following two statements hold:
1. Stationary var.: Let (λ T ) T ∈N+ be a non-negative sequence, and ( λ T ) T ∈N+ be a strictly positive sequence such that λ T ≤ λ T . Furthermore, for any T ∈ N + , t ∈ [T ], define x T,t ∈ X as x T,t ∈ arg max x∈ X σ λ 2 T It-1 (x; X T,t-1 ), where X T,t-1 = (x T,1 , . . . , x T,t-1 ). Then, for any T ∈ {T ∈ N + | T /2 ≥ 3γ T ( λ 2 T I T )}, the following inequality holds:
max x∈ X σ λ 2 T I T (x; X T,T ) ≤ 4 T λ 2 T T γ T ( λ 2 T I T ). (4)
2. Non-stationary var.: Let (λ t ) t∈N+ be a non-negative sequence, and ( λ t ) t∈N+ be a strictly positive sequence such that λ t ≤ λ t . Furthermore, for any t ∈ N + , define x t ∈ X as x t ∈ arg max x∈ X σ Σt-1 (x; X t-1 ), where X t-1 = (x 1 , . . . , x t-1 ) and Σ t-1 = diag(λ 2 1 , . . . , λ 2 t-1 ). Then, for any T ∈ {T ∈
N + | T /2 ≥ 4γ T ( Σ T )}, max x∈ X σ Σ T (x; X T ) ≤ 4 T T t=1 λ 2 t γ T ( Σ T ), (5
)
where Σ t = diag( λ 2 1 , . . . , λ 2 t ).
To make the above statements explicit, we give the following corollary for k = k SE and k = k Matérn with the stationary variance parameter as a special case of Lemma 3.1. The proof is in Appendix C.
Corollary 3.2. Suppose the assumptions in statement 1 of Lemma 3.1. Then, the following four statements hold:
1. Suppose k = k SE and fix any α > 0. If λ 2 T = Ω(exp(-
T 1 d+1 ln -α (1 + T ))), Eq. (4) holds with λ 2 T = λ 2
T for all T ≥ T , where T < ∞ is the constant that depends on X , α, d, and ℓ.
2. Suppose k = k Matérn with ν > 1/2 and fix any α > 0.
If
λ 2 T = Ω(T -2ν d (ln(1 + T )) 2ν(1+α) d
), Eq. ( 4) holds with λ 2 T = λ 2 T for all T ≥ T , where T < ∞ is the constant that depends on X , α, d, ℓ, and ν.
3. Suppose k = k SE and fix any α > 0 and C > 0. If ∀T ∈ N + , λ
2 T < C exp -T 1 d+1 ln -α (1 + T ) (in-
cluding λ T = 0), the following inequality holds for all T ≥ T :
max x∈ X σ λ 2 T I T (x; X T,T ) ≤ O exp -T 1 d+1 ln -α T ,
where T < ∞ and the implied constant of the above inequality depend on X , α, d, C, and ℓ.
4. Suppose k = k Matérn with ν > 1/2 and fix any α > 0 and C > 0.
If ∀T ∈ N + , λ 2 T < CT -2ν d (ln(1 + T )) 2ν(1+α) d
(including λ T = 0), the following inequality holds for all T ≥ T :
max x∈ X σ λ 2 T I T (x; X T,T ) ≤ O T -ν d (ln T ) ν(1+α) d ,(6)
where T < ∞ and the implied constant of the above inequality depend on X , α, d, C, ℓ, and ν.
In all of the above statements, T increases as decreasing of α, and T → ∞ as α → 0.
Comparison with Existing Upper Bound. Here, we compare statement 1 in Lemma 3.1 with the existing upper bound. By considering the case λ 2 T = λ 2 T , our result shows O T -1 λ 2 T T γ T (λ 2 T I T ) upper bound of posterior standard deviation under the sublinear increasing condition of MIG so that T /2 ≥ 3γ T (λ 2 T I T ) holds. On the other hand, the existing analysis of MVR 1 implies max x∈ X σ λ 2 T I T (x; X T,T ) ≤          O 1 T λ 2 T T γ T (λ 2 T I T ) if λ 2 T = Ω(1), O 1 T T γ T (λ 2 T I T ) ln 1+λ -2 T if λ 2 T = o(1). (7) Since λ 2 T ≤ 1/ ln(1 + λ -2 T ), our analysis improves the noise parameter dependence on the decreasing regime of λ 2 T . Furthermore, several recent noiseless GP bandit works also derive the related result to Lemma 3.1 or Corollary 3.2. Flynn & Reeb (2025) consider the noiseless setting by relying on elliptical potential count lemma (Lemma 3.3 below), and the naive adaptation of their analysis leads 2 max x∈ X σ λ 2 T I T (x; X T,T ) ≤ O   γ T (λ 2 T I T ) T + λ 2 T T γ T (λ 2 T I T ) T   . In the above equation, the decreasing speed of λ 2 T is more restricted than that of Corollary 3.2 to obtain the same order upper bound as Eq. (4). For example, if k = k Matérn and λ 2 T = Ω(T -2ν d (ln(1 + T )) 2ν(1+α) d ) as with the condition of statement 4 in Corollary 3.2, the first term γ T (λ 2 T I T ) T dominates the second term since the existing upper bound of MIG implies γ T (λ 2 T I T ) T
= O(1). Furthermore, Salgia et al. (2024) derives a similar result to our statement 4 with a random sampling algorithm instead of MVR. The main theoretical advantage of our result is that the constant T depends on the entire input space X instead of the subset X , while that of (Salgia et al., 2024) has the dependence on X . This dependence on X raises the requirement of the additional level-set assumption to apply the PE-style 1 Eq. ( 7) also holds for any algorithm by replacing max x∈ X σ λ 2 T I T (x; XT,T ) with 1 T T t=1 σ λ 2 T I T (xT,t; XT,t-1). For example, GP-UCB (Srinivas et al., 2010;Chowdhury & Gopalan, 2017) and GP-TS (Chowdhury & Gopalan, 2017) use the upper bound of 1 T T t=1 σ λ 2 T I T (xT,t; XT,t-1). 2 In some settings of λ 2 T , the first term of r.h.s. γT (λ 2 T IT )/T can be small by putting min γ γT (λ 2 T IT ), λ 2 T , γT (λ 2 T IT ) in the first term, instead of γT (λ 2 T IT ). See Lemma 3.3. algorithm in noiseless feedback (see Assumption 4.2 in (Salgia et al., 2024)). Furthermore, as described in the proof sketch below, our proof mainly relies on the simple extension of the well-known information gain arguments from (Srinivas et al., 2010), not on the technique of (Salgia et al., 2024) that involves the theoretical tools from function approximation literature.
Proof sketch of Lemma 3.1. Here, since statement 2 is derived as the extension of the proof of statement 1, we only describe the proof sketch of statement 1 for simplicity. We leave the full proof, including statement 2, in Appendix C. Our proof is based on the following two observations: 1. For any index set T ⊂ [T ], max x∈ X σ λ 2 T I T (x; X T,T ) can be bounded from above by the average observed posterior standard deviation on T from the definition of MVR. Namely, max x∈ X σ λ 2
T I T (x; X T,T ) ≤ 1 |T | t∈T σ λ 2 T It-1 (x T,t ; X T,t-1 ) holds. 2. If we set T = {t ∈ [T ] | λ -1 T σ λ 2 T It-1 (x T,t ; X T,t-1 ) ≤ 1}, σ 2 λ 2 T It-1 (x T,t ; X T,t-1 ) = λ 2 T min 1, λ -2 T σ 2 λ 2 T It-1 (x T,t ; X T,t-1 ) ≤ 2 λ 2 T ln 1 + λ -2 T σ 2 λ 2 T It-1 (x T,t ; X T,t-1 ) .(8)
for all t ∈ T . We use ∀a ≥ 0, min{1, a} ≤ 2 ln(1+a) in the last line. By relying on the standard MIG-based analysis (e.g., Theorem 5.3, 5.4 in (Srinivas et al., 2010)) and the assumption λ T ≤ λ T , the above inequality implies
t∈T σ λ 2 T It-1 (x T,t ; X T,t-1 ) ≤ 2 λ 2 T T γ T ( λ 2 T I T ).(9)
From the above two observations, the remaining interest is the increasing speed of |T |. We use the following lemma, which we call elliptical potential count lemma in (Flynn & Reeb, 2025), as the analogy of elliptical potential arguments in linear bandits. Lemma 3.3 (Elliptical potential count lemma, Lemma D.9 in (Flynn & Reeb, 2025)). Fix any T ∈ N + , any sequence x 1 , . . . , x T ∈ X , and any λ > 0. Set T c as
T c = {t ∈ [T ] | λ -1 σ λ 2 It-1 (x t ; X t-1 ) > 1},where
X t = (x 1 , . . . , x t ). Then, the number of elements of T c satisfies |T c | ≤ min 3γ 3γ T (λ 2 I T ), λ 2 , 3γ T (λ 2 I T ) . Furthermore, γ(•, •) is any monotonic upper bound of MIG defined on R + × R + , which satisfies ∀T ∈ N + , λ > 0, γ T (λ 2 I T ) ≤ γ(T, λ 2 ) and ∀λ > 0, T ≥ 1, ϵ ≥ 0, γ(T, λ 2 ) ≤ γ(T + ϵ, λ 2 ).
From the above lemma, we obtain the lower bound of T as |T | ≥ T -3γ T ( λ 2 T I T ). Finally, we obtain the desired result by noting |T | ≥ T /2 holds for any T ∈ {T ∈ N + | T /2 ≥ 3γ T ( λ 2 T I T )}.
this section cite: ['b25', 'b5', 'b5', 'b25', 'b25', 'b8', 'b8']

Section: Noiseless Setting
As a first application of our result, we study a noiseless setting; namely, we focus on the setting where ρ t = 0 for all t ∈ N + in Assumption 2.2. The following results show our cumulative and simple regret guarantees for PE and MVR.
Theorem 4.1 (Cumulative Regret Bound for PE.). Suppose Assumptions 2.1 and 2.2 hold with ρ t = 0 for all t ∈ N + . Furthermore, assume B, d, ℓ, and ν are Θ(1). Then, when running Algorithm 1 with β 1/2 = B, λ = 0, and any fixed N 1 ∈ N + , the following statements hold:
• If k = k SE , R T = O(ln T ).
•
If k = k Matérn with ν > 1/2, R T =      O(T d-ν d ) if ν < d, O((ln T ) 2+α ) if ν = d, O(ln T ) if ν > d.(10)
Here, α > 0 is an arbitrarily fixed constant.
Theorem 4.2 (Simple Regret Bound for MVR.). Suppose the same conditions as those of Theorem 4.1. Then, when running Algorithm 2 with λ = 0, the following statements hold:
• If k = k SE , r T = O exp -1 2 T 1 d+1 ln -α T .
•
If k = k Matérn with ν > 1/2, r T = O T -ν d .
Remark 4.3. The above theorems assume that the learner can exactly choose x t in the algorithms, which is unreasonable for a continuous domain X . However, a similar guarantee, which is worse by an additional √ ln T multiplicative factor than the above results, can be obtained by the existing analysis (Li & Scarlett, 2022) under the additional Lipschitz assumption for f . Note that such Lipschitz assumption for f automatically holds under fixed B when we set k = k SE or k = k Matérn with ν > 1 (Lee et al., 2022).
The proofs of Theorems 4.1 and 4.2 are respectively derived by directly following the standard analysis of PE and MVR with statements 3 and 4 of Corollary 3.2. We describe full proofs in Appendix D for completeness.
Discussion. As summarized in Tables 12, our results are the same as or superior to the best-known upper bounds in almost all cases. The only exception is the simple regret with k = k SE , whose polynomial factor in exponential gets worse from -(α + 1/d) into -1/(d + 1), compared to the algorithm of (Kim & Sanz-Alonso, 2024). Roughly speaking, the numerator of the factor -1/(d + 1) in our analysis comes from the exponent of the upper bound of MIG O(ln d+1 (T /λ 2 )). We expect our simple regret has room of improvement from O(exp(-
T 1 d+1 ln -α T )) into O(exp(-T 2 d ln -α T ))
in future work, since the conjectured best upper bound of MIG is O(ln d/2 (T /λ 2 )) from the regret lower bound (Scarlett et al., 2017).
this section cite: ['b14', 'b11', 'b23']

Section: Optimal Dependence of RKHS Norm Upper Bound
As the second application of our result, we consider improving the existing dependence of RKHS norm upper bound B in the regret upper bounds.
this section cite: []

Section: Simple Regret
The following theorem shows our results for simple regret.
Theorem 5.1 (Simple Regret Bound for MVR.). Suppose Assumptions 2.1 and 2.2 hold with ρ t = ρ > 0 for all t ∈ N + . Furthermore, assume ρ, d, ℓ, and ν are Θ(1), and X is finite. Then, when running Algorithm 2 with λ 2 = Θ(B -2 ), the following statements hold for any fixed α > 0 with probability at least 1 -δ:
• If k = k SE , B = O exp T 1 d+1 ln -α (1 + T ) ,and
T ≥ T , then, r T = O ln d+1 (T B 2 ) T .
Here, T is the constant, defined in statement 1 of Corollary 3.2.
•
If k = k Matérn with ν > 1/2, B = O T ν d ln -ν(1+α) d (1 + T ) , and T ≥ T , then, r T = O B d 2ν+d T -ν 2ν+d .
Here, T is the constant defined in statement 2 of Corollary 3.2.
The important point is that the setting of the noise parameter λ 2 = Θ(1/B 2 ) depends on B. We describe the full proof of the above theorem in Appendix E. Remark 5.2. If we allow an additional logarithmic factor, we can eliminate the finiteness assumption of X in Theorem 5.1 by relying on the discretization with 1/T -net as with Remark 4.3. The notable point we have to care about is that the Lipschitz constant is given as B in the existing result (Lee et al., 2022). Therefore, the extension of Theorem 5.1 for continuous domain requires 1/(BT )-net to maintain the order of B, and the resulting regret upper bound suffers from additional ln(T B) factor, instead of √ ln T factor derived from the standard discretizing argument that does not care the dependence of B (Li & Scarlett, 2022).
Discussion. In both kernels, the polynomial dependence of B matches the lower bound in (Scarlett et al., 2017), while there exists room for improvement in the logarithmic factor. On the other hand, there exist some exceptional cases that Theorem 5.1 does not cover, even though its lower bound of the simple regret is guaranteed to converge to 0. For example, when B = Θ(T ν d ln -ν d T ), the lower bound of (Scarlett et al., 2017) suggest that some algorithm find any ϵ-optimal point for sufficiently large T (namely, simple regret converges to 0), while violating our assumption. As with the discussion in Section 4, this limitation can be eliminated in the future if the upper bound of MIG matches the conjectured best upper bound.
this section cite: ['b14', 'b23', 'b23']

Section: Cumulative Regret
The following theorem also shows that the PE algorithm can achieve optimal dependence of B up to a poly-logarithmic factor.
Theorem 5.3 (Cumulative Regret Bound for PE.). Suppose Assumptions 2.1 and 2.2 hold with ρ t = ρ > 0 for all t ∈ N + . Furthermore, assume ρ, d, ℓ, and ν are Θ(1), and X is finite. Then, when running Algorithm 1 with
β 1/2 = (B + ρλ -1 ) 2 ln 2|X |(1+log 2 T ) δ , λ 2 = Θ(B -2
), and any fixed N 1 ∈ N + , the following statements hold with probability at least 1 -δ:
• If k = k SE and B = O( √ T ), then, R T = O (ln T ) T ln d+1 (T B 2 ) ln |X | δ .
•
If k = k Matérn with ν > 1/2 and B = O T 2ν 2 +3νd 4d 2 +4ν 2 +6νd , then, R T = O T ν+d 2ν+d B d 2ν+d .
See Appendix E for the proof. In contrast to the analysis of the MVR, the analysis of PE cannot leverage Corollary 3.2 by setting λ 2 = Θ(B -2 ). Intuitively, this is because the existence of the common constant T over each batch is not guaranteed since λ 2 depends only on T , not the total step size of each batch. Due to this limitation, the above result is proved by leveraging Lemma 3.3 as with the analysis of (Flynn & Reeb, 2025), instead of using Corollary 3.2. As a result, the conditions about B are more restricted than those of Theorem 5.1, due to the fundamental limitation of the analysis of (Flynn & Reeb, 2025) as previously discussed in Section 3. For example, if k = k Matérn , the increasing
speed of B = O T 2ν 2 +3νd 4d 2 +4ν 2 +6νd in Theorem 5.3 is more restricted than B = O T ν d
in Theorem 5.1, regardless the fact that lower bound (Scarlett et al., 2017) suggest the sublinear cumulative regret is achievable when B = o(T ν d ).
We leave future research to break this limitation.
this section cite: ['b8', 'b8', 'b23']

Section: Non-Stationary Variance
As the third application of our result, we consider the nonstationary variance setting, which falls between a noiseless and a noisy regime. In this setting, our goal is to quantify the regret by the cumulative variance proxy V T = t∈[T ] ρ 2 t . That is, we aim to construct an algorithm that achieves better performance than the one for the stationary noise setting if V T increases sublinearly. While the non-stationary variance setting has already been studied and motivated in the linear bandits (Zhou et al., 2021), to our knowledge, no existing GP-bandits literature exists for this problem. Therefore, in Appendix J, we describe some potential applications to motivate non-stationary variance setting in GP-bandits.
By following the existing works (Zhou et al., 2021;Zhang et al., 2021;Zhou & Gu, 2022), we suppose that the learner can access true variance proxy ρ 2 t at the end of step t. We leave the unknown ρ 2 t setting for future research. Note that, as described later, the direct extension of the existing linear bandit algorithm with non-stationary variance does not lead to the near-optimal guarantee.
Lower bound. Since the stationary noise problem with ρ 2 = V T /T is subsumed in the non-stationary problem with the cumulative variance proxy V T , the following lower bounds are obtained as the corollary of the existing stationary variance lower bound (Scarlett et al., 2017) and noiseless lower bound (Bull, 2011).
Corollary 6.1 (Lower bound for cumulative regret). Let X be X = [0, 1] d . Furthermore, assume V T = Ω(1) and V T = O(T 2 ) with sufficiently small implied constant. Then, for any algorithm, there exists a GP bandit problem instance that satisfies Assumptions 2.1 and 2.2 with t∈[T ] ρ 2 t = V T and the following two statements:
• If k = k SE , E[R T ] = Ω V T ln d 2 T 2 V T .
•
If k = k Matérn , E[R T ] = Ω V ν 2ν+d T T d 2ν+d .
Here, we assume B, d, ℓ, and ν are Θ(1).
Corollary 6.2 (Lower bound for simple regret). Let X be X = [0, 1] d . Furthermore, assume V T = Ω(1). If there exists an algorithm such that E[r T ] ≤ ϵ hold for all problem instances that satisfy Assumptions 2.1 and 2.2 with t∈[T ] ρ 2 t = V T , then:
• For k = k SE , the total step size T needs to satisfy
T ≥ Ω V T ϵ 2 ln d 2 1
ϵ .
• For k = k Matérn , the total step size T needs to satisfy
T ≥      Ω V T ϵ 2 1 ϵ d ν if d ≤ 2ν or V T = Ω(T d-2ν d ), Ω 1 ϵ d ν if d > 2ν and V T = O(T d-2ν d
).
Here, we assume B, d, ℓ, and ν are Θ(1). Furthermore, ϵ > 0 is a sufficiently small constant.
The lower bound Ω (1/ϵ)
d ν for k Matérn if d > 2ν and V T = O(T d-2ν d
) in Corollary 6.2 come from (Bull, 2011), and others come from (Scarlett et al., 2017).
Note that the noiseless lower bound for expected regret also holds for noisy settings since an expected regret in the noisy setting can always be reduced to one in the noiseless setting, whose algorithm randomness is induced by observation noise. Interestingly, the above simple regret lower bound indicates that if d > 2ν and V T = O(T
d-2ν d
), the non-stationary variance setting may have the same level of difficulty as that of the noiseless problem. Our VA-MVR algorithm proposed below justifies this fact by providing a simple regret upper bound matching the above lower bound.
Algorithm. Algorithms 3 and 4 in Appendix H show PE and MVR-based algorithms for non-stationary variance problems, which we call variance-aware PE (VA-PE) and MVR (VA-MVR), respectively. The algorithms themselves are the variant of the standard PE or MVR algorithms that directly set the true variance proxy ρ 2 t to the noise variance parameter λ 2 t for the heteroscedastic GP model.
this section cite: ['b35', 'b35', 'b32', 'b34', 'b23', 'b2', 'b2', 'b23']

Section: Theoretical analysis.
The following theorems give the cumulative and simple regret guarantees for VA-PE and VA-MVR, respectively.
Theorem 6.3 (Cumulative regret upper bound for VA-PE). Suppose Assumptions 2.1 and 2.2, and |X | < ∞ holds. Furthermore, assume V T := t∈[T ] ρ 2 t = Ω(1). Then, when running Algorithm 3 with
β 1/2 = B + 2 ln 2|X |(1+log 2 T ) δ , with probability at least 1-δ, R T = O (ln T ) V T ln d+1 T 2 V T ln |X | δ if k = k SE . Furthermore, if k = k Matérn , R T =          O V ν 2ν+d T T d 2ν+d if d ≤ 2ν, O V ν 2ν+d T T d 2ν+d if d > 2ν, V T = Ω T d-2ν d , O T d-ν d if d > 2ν, V T = O T d-2ν d .
Theorem 6.4 (Simple regret upper bound for VA-MVR). Suppose Assumptions 2.1 and 2.2, and |X | < ∞ holds. Furthermore, assume V T = Ω(1). Then, when running Algorithm 4, with probability at least
1 -δ, r T = O V T T 2 ln d+1 T 2 V T ln |X | δ if k = k SE . Further- more, if k = k Matérn , r T =          O V ν 2ν+d T T -2ν 2ν+d if d ≤ 2ν, O V ν 2ν+d T T -2ν 2ν+d if d > 2ν, V T = Ω T d-2ν d , O T -ν d if d > 2ν, V T = O T d-2ν d .
In both results, the regret upper bound matches the lower bound up to the logarithmic factor, except for the cumulative regret guarantee for
d > 2ν, V T = O T d-2ν d in k = k Matérn . However, note that the resulting regret O T d-ν d
in this exceptional case matches the conjectured lower bound (Vakili, 2022); Therefore, as with our simple regret lower bound, O T
d-ν d
upper bound in our analysis has no room for improvement if the conjectured lower bound in (Vakili, 2022) is true.
Comparison with the stationary setting. When V T = Θ(T ), our result matches the existing O(T ν+d 2ν+d ) upper bound for the stationary setting. If we consider the setting that V T increases sublinearly, our algorithm achieves a smaller regret than the existing stationary lower bounds. For example, if V T = Θ(1) in k = k SE , the resulting regret becomes logarithmcally increasing regret R T = O(1), while the regret lower bound for stationary variance setting is Ω( √ T ).
Comparison with the algorithm in heteroscedastic linear bandits. In heteroscedastic linear bandits, the weighted OFUL+ algorithm, which is known to achieve nearly optimal regret with UCB-based algorithm construction, is proposed in the known ρ 2 t setting. We can also consider the extension of weighted OFUL+ to the GP bandits by constructing a UCB-based score with a heteroscedastic GP model. We call this extension variance-aware GP-UCB (VA-GP-UCB), and give the details in Appendix I. However, the regret of VA-GP-UCB becomes strictly worse than VA-PE and VA-MVR due to the following two reasons: Firstly, as with the stationary adaptive confidence bound (e.g., Lemma 3.11 in (Abbasi-Yadkori, 2013)), the existing adaptive confidence bound for heteroscedastic GP-model (Kirschner & Krause, 2018) contains O( γ t (Σ t )) factor in the confidence width parameter, which leads to the sub-optimal order of the regret. Secondly, our analysis of VA-GP-UCB relies on the extension of the elliptical potential count lemma (Lemma C.1) to a heteroscedastic GP model, which could result in worse dependence of the noise variance parameters than that of Lemma 3.1 (See the discussion in Section 3). To our knowledge, the existing technical tools provide no direct way to avoid the above two issues. Finally, we give the summary of our results for the nonstationary variance setting in Tables 6-8 in Appendix A.
this section cite: ['b13']

Section: Conclusion
We study the GP-bandit problem with the following three settings: (i) noiseless observation, (ii) varying RKHS norm, and (iii) non-stationary variance setting. We first propose a new uniform upper bound of the posterior standard deviation of GP in the MVR algorithm. By leveraging this upper bound, we refine the regret guarantee of the existing PE and MVR algorithms. Our derived upper bound matches the lower bound up to the logarithmic factor in the aforementioned three settings.
ln d T T O T -ν 2ν+d D (Lyu et al., 2019) (Kim & Sanz-Alonso, 2024) Kernel-AMM-UCB O ln d+1 T T O T -νd+2ν 2 2ν 2 +2νd+d 2 D (Flynn & Reeb, 2025) GP-UCB+, O exp -CT 1 d -α O T -ν d +α P α > 0 is an arbitrarily EXPLOIT+ fixed constant. (Kim & Sanz-Alonso, 2024) C > 0 is some constant. MVR O exp -1 2 T 1 d+1 ln -α T O T -ν d D α > 0 is an arbitrarily fixed (our analysis) constant. Lower Bound N/A Ω T -ν d N/A (Bull, 2011)
this section cite: []

Section: References
Ref_id:b0 Title: Online learning for linearly parametrized control problems Year: (2013)
Ref_id:b1 Title: Bayesian optimization with safety constraints: safe and automatic parameter tuning in robotics Year: (2021)
Ref_id:b2 Title: Convergence rates of efficient global optimization algorithms Year: (2011)
Ref_id:b3 Title: On lower bounds for standard and robust Gaussian process bandit optimization Year: (2021)
Ref_id:b4 Title: Highdimensional experimental design and kernel bandits Year: (2021)
Ref_id:b5 Title: On kernelized multiarmed bandits Year: (2017)
Ref_id:b6 Title: Elements of Information Theory Year: (2006)
Ref_id:b7 Title: Exponential regret bounds for Gaussian process bandits with deterministic observations Year: (2012)
Ref_id:b8 Title: Tighter confidence bounds for sequential kernel regression Year: (2025)
Ref_id:b9 Title: Rapid, accurate, and precise concentration measurements of a methanol-water mixture using Raman spectroscopy Year: (2018)
Ref_id:b10 Title: Bandit optimisation of functions in the Matérn kernel RKHS Year: (2020)
Ref_id:b11 Title: Enhancing Gaussian process surrogates for optimization and posterior approximation via random exploration Year: (2024)
Ref_id:b12 Title: Improved regret analysis for variance-adaptive linear bandits and horizon-free linear mixture MDPs Year: (2022)
Ref_id:b13 Title: Information directed sampling and bandits with heteroscedastic noise Year: (2018)
Ref_id:b14 Title: Multi-scale zero-order optimization of smooth functions in an RKHS Year: (2022)
Ref_id:b15 Title: Bayesian optimization with adaptive surrogate models for automated experimental design Year: (2021)
Ref_id:b16 Title: Gaussian process bandit optimization with few batches Year: ()
Ref_id:b17 Title: Efficient batch blackbox optimization with deterministic regret bounds Year: (2019)
Ref_id:b18 Title: Risk-averse heteroscedastic Bayesian optimization Year: ()
Ref_id:b19 Title: Learning to optimize via posterior sampling Year: (2014)
Ref_id:b20 Title: A domain-shrinking based Bayesian optimization algorithm with orderoptimal regret performance Year: (2021)
Ref_id:b21 Title: Random exploration in Bayesian optimization: Order-optimal regret and computational efficiency Year: ()
Ref_id:b22 Title: Tight regret bounds for Bayesian optimization in one dimension Year: (2018)
Ref_id:b23 Title: Lower bounds on regret for noisy Gaussian process bandit optimization Year: (2017)
Ref_id:b24 Title: Practical Bayesian optimization of machine learning algorithms Year: (2012)
Ref_id:b25 Title: Gaussian process optimization in the bandit setting: No regret and experimental design Year: (2010)
Ref_id:b26 Title: Randomized Gaussian process upper confidence bound with tighter Bayesian regret bounds Year: (2023)
Ref_id:b27 Title: Posterior sampling-based Bayesian optimization with tighter Bayesian regret bounds Year: (2024)
Ref_id:b28 Title: Open problem: Regret bounds for noise-free kernel-based bandits Year: ()
Ref_id:b29 Title: Optimal order simple regret for Gaussian process bandits Year: ()
Ref_id:b30 Title: On information gain and regret bounds in Gaussian process bandits Year: ()
Ref_id:b31 Title: Finite-time analysis of kernelised contextual bandits Year: (2013)
Ref_id:b32 Title: Improved varianceaware confidence sets for linear bandits and linear mixture MDP Year: (2021)
Ref_id:b33 Title: Variancedependent regret bounds for linear bandits and reinforcement learning: Adaptivity and computational efficiency Year: (2023)
Ref_id:b34 Title: Computationally efficient horizonfree reinforcement learning for linear mixture MDPs Year: (2022)
Ref_id:b35 Title: Nearly minimax optimal reinforcement learning for linear mixture Markov decision processes Year: (2021)
