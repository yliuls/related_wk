Title: Learning Parametric Distributions from Samples and Preferences
Abstract: Recent advances in language modeling have underscored the role of preference feedback in enhancing model performance. This paper investigates the conditions under which preference feedback improves parameter estimation in classes of continuous parametric distributions. In our framework, the learner observes pairs of samples from an unknown distribution along with their relative preferences depending on the same unknown parameter. We show that preference-based M-estimators achieve a better asymptotic variance than sample-only M-estimators, further improved by deterministic preferences. Leveraging the hard constraints revealed by deterministic preferences, we propose an estimator achieving an estimation error scaling of O(1/n)-a significant improvement over the Θ(1/ √ n) rate attainable with samples alone. Next, we establish a lower bound that matches this accelerated rate; up to dimension and problem-dependent constants. While the assumptions underpinning our analysis are restrictive, they are satisfied by notable cases such as Gaussian or Laplace distributions for preferences based on the log-probability reward.

Section: Introduction
Recent progress in language modeling has showcased the effectiveness of preference feedback for fine-tuning (Ziegler et al., 2019;Ouyang et al., 2022;Bai et al., 2022;Touvron et al., 2023;Dubey et al., 2024). Preference dataindicating relative quality between outcomes-consistently outperforms approaches using positive examples only like supervised fine-tuning (Ivison et al., 2024). This empirical success suggests that preference feedback introduces new, complementary information beyond the observed data. Understanding how and why preferences provide this advantage requires connecting the preference model to the data-generating process (Ge et al., 2024).
To understand the role of preference feedback, we focus on a simpler yet illustrative problem: parameter estimation for parametric distributions and preferences. Specifically, the learner observes pairs of samples from an unknown distribution, along with preferences informed by the same parameter. For instance, preferences based on log-probabilities naturally link the preference and probability models, though other formulations are possible (Huang et al., 2024).
For continuous distributions, we uncover a significant statistical learning gap between preference-based and sampleonly estimators. This paper primarily investigates this gap, taking the sample-only maximum likelihood estimator (MLE)-optimal among unbiased estimators-as a baseline. The well-established theory of M-estimators ( Van der Vaart, 2000) suggests that preference-based M-estimators improve asymptotic variance under certain conditions. Yet, this improvement is modest: when samples are of similar quality, preference feedback approaches a fair coin toss, providing minimal additional information. While reducing asymptotic variance is encouraging, it does not fully explain the substantial performance gains observed empirically in large-scale language models. For deterministic preferences, we prove a more striking result: preference-based estimators achieve a statistically significant acceleration in parameter estimation. Specifically, we show that the estimation error scales as O(1/n) instead of the O(1/ √ n) rate achieved by sample-only estimators. This acceleration is supported by a matching lower bound, up to dimension and problem-dependent constants.
While this acceleration might sound surprising, the Θ(1/n) rate can already be observed in a special case of sampleonly parameter estimation. For instance, consider estimating the location parameter θ of a uniform distribution on [θ, θ + 1] based solely on samples (Wainwright, 2019). The minimax rate for estimation error is Θ(1/n). The optimal estimator achieving the accelerated rate is the minimum of uniform observations whose density is positive at θ. This improved rate arises from the accumulation of random variables having a positive density at a specific point through a minimum (or maximum) operator, in contrast to the slower aggregation inherent to averaging. Similarly, for deterministic preferences with log-likelihood rewards, we observe the true ordering between likelihoods. As it enforces hard constraints-through a minimum operator-on the admissible parameters, our preference-based estimator achieves accelerated convergence.
To illustrate this acceleration, consider the standard normal distribution with preferences based on log-probabilities. Let n ∈ N and [n] := {1, • • • , n}. For each i ∈ [n], observe samples (X i , Y i ) ∼ N (0 2 , I 2 ) along with their log-likelihood deterministic preference Z i := sign((Y i -X i )S i ), where S i := (X i + Y i )/2 is their average. The triplet (X i , Y i , Z i ) imposes a hard constraint based on S i on the location of candidate estimators θ that are consistent with this log-likelihood deterministic preference. Specifically, they satisfy θ ≤ S i if S i > 0, and θ ≥ S i otherwise. The set of feasible parameters satisfying all constraints is thus max i∈[n], Si<0 S i , min i∈[n], Si>0 S i . Since the density of N (0, 1/2) is positive near zero, the length of this interval decreases as O(1/n) with high probability.
this section cite: ['b55', 'b31', 'b2', 'b45', 'b7', 'b21', 'b12', 'b17', 'b47', 'b50']

Section: Contributions
For continuous parametric probability distributions, we study the statistical learning gap between preference-based estimators and sample-only estimators.
• First, we show that preference-based M-estimators achieve a better asymptotic variance than sample-only M-estimators. The variance is further improved for deterministic preference.
• Second, we introduce an estimator satisfying the constraints revealed by the deterministic preferences, and prove an accelerated estimation error rate of O(1/n). This constitutes a significant improvement over the Θ(1/ √ n) rate achieved by M-estimators.
• Third, we provide a lower bound of Ω(1/n), matching our upper bound up to problem-specific constants.
Our results are derived under general assumptions on the distributions and the preferences. While restrictive, they are satisfied by notable cases such as Gaussian or Laplace distributions for preferences based on log-probabilities.
1.2. Related Work Learning parametric distributions. Parametric estimation is a central approach in statistics, reducing inference about a distribution to the estimation of a finite-dimensional parameter (Lehmann & Casella, 2006;Wasserman, 2013). The maximum likelihood estimator (MLE) is the most fundamental method in this setting. Its asymptotic properties are well studied (Cramér, 1946;Ibragimov & Has' Minskii, 2013;Van der Vaart, 2000), while non-asymptotic guarantees have been established in Birgé & Massart (1993) and Spokoiny (2012). Lower bounds in parametric estimation rely on techniques such as Le Cam's two-point method (LeCam, 1973), Fano's method (Fano, 1952), and Assouad's method (Assouad, 1983), and provide fundamental limits on estimation accuracy (Tsybakov, 2009).
Learning parametric value/preference functions. In the tabular setting, learning from pairwise comparisons aligns with the ranking problem. The performance of MLE under the Bradley-Terry model (Bradley & Terry, 1952) and its extensions has been extensively studied (Hunter, 2004;Negahban et al., 2012;Hajek et al., 2014;Rajkumar & Agarwal, 2014;Shah et al., 2016;Shah & Wainwright, 2018;Mao et al., 2018). The continuous setting, where generalization beyond observed preferences is required, has received less attention, except for linear utility functions (Zhu et al., 2023;Ge et al., 2024;Yao et al., 2025). Beyond analyzing the sample complexity of reward learning with MLE under the Bradley-Terry noise model, Zhu et al. (2023) study the performance of policies trained on the learned reward model. They show that while MLE may fail, a pessimistic variant can yield a policy with improved performance. Relaxing the noise assumption, Ge et al. (2024) show that utility parameters remain unidentifiable without strong modeling assumptions, even with noise-free query responses. However, they demonstrate that, in the active learning setting, utility can still be learned, even in the absence of noise. Their results highlight that the sampling distribution of observations must be aligned with the utility function to achieve improved sample complexity. Yao et al. (2025) leverages sparsity in the preference model and establish sharp estimation rates depending on the sparsity level. Finally, related estimation problems have also been studied in the contexts of dueling bandits and reinforcement learning (Faury et al., 2020;Saha et al., 2023).
Fine-tuning with preference data. Large language models often go through a post-training phase focusing mainly on learning from preference feedback (Lambert, 2024), to improve capabilities such as summarization, instruction following, and reasoning. The standard approach, reinforcement learning from human feedback (RLHF) (Ziegler et al., 2019), trains a reward model to align with human preferences and then optimizes the policy using reinforcement learning, typically with PPO (Schulman et al., 2017). RLHF follows three main steps: supervised fine-tuning, reward model training, and policy optimization. Another line of work has explored alternatives to PPO to simplify training. One such method, direct preference optimization (DPO), reformulates the reward function to learn a policy directly from preference data, avoiding an explicit reward model.
Other preference optimization objectives have also been proposed (Meng et al., 2024). Finally, while preference data has traditionally been gathered through human annotators, the learning paradigm has recently expanded to include selfplay where the model critiques its own generations (Dubey et al., 2024;Huang et al., 2024).
this section cite: ['b24', 'b52', 'b6', 'b20', 'b47', 'b4', 'b39', 'b23', 'b9', 'b0', 'b46', 'b5', 'b19', 'b30', 'b15', 'b33', 'b38', 'b37', 'b26', 'b54', 'b12', 'b53', 'b54', 'b12', 'b53', 'b10', 'b35', 'b22', 'b55', 'b36', 'b28', 'b7', 'b17']

Section: Problem Statement
Parameter estimation. Let Θ ⊆ R k be a set of parameters for a class of continuous probability distributions F over X ⊆ R d . Let B Θ := max θ∈Θ ∥θ∥ be the bound on Θ for the norm ∥ • ∥ specific to F. Let S k-1 be the unit sphere for this norm. Let p ⊗2 θ be the distribution of two independent observations of p θ .
Let θ ⋆ be an unknown parameter to estimate. Our samples are drawn from p ⊗2 θ ⋆ , i.e., (X, Y ) ∼ p ⊗2 θ ⋆ . We use two archetypal examples satisfying our assumptions. First, the class F N ,Σ of multivariate Gaussian distributions with known covariance Σ, where Θ are the natural parameters with norm ∥ • ∥ Σ where ∥x∥ Σ := Preference feedback. Let ℓ θ : X 2 → R be a parametric preference function. Given a parametric reward function r θ , a reward-based preference function is defined as ℓ θ (x, y) = r θ (x) -r θ (y). As a concrete example for our derivations, we consider preference based on the log-probability reward r θ = log p θ . Given observations (x, y), the true preference z of x over y is governed by sign(ℓ θ (x, y)) ∈ {±1, 0}. In many settings, however, the observed preference Z can be stochastic due to noise or randomness in human feedback.
Conditioned on (X, Y ) ∼ p ⊗2 θ , we denote the p.d.f. of the law of the preference Z by h(ℓ θ (X, Y ), •). On X 2 × {±1, 0}, the p.d.f. of the law of (X, Y, Z) is denoted as q θ,h (x, y, z) := p ⊗2 θ (x, y)h(ℓ θ (x, y), z). Under deterministic feedback, the true preferences are observed:
h det (•, z) := 1 (z = sign(•)) .(1)
Under stochastic feedback, noisy preferences z ∈ {±1} are observed based on the sigmoid link:
h sto (•, z) := σ(z•) with σ(x) := (1 + e -x ) -1 . (2) Informative preferences. A natural question is to see when preference Z ∼ h(ℓ θ ⋆ (X, Y ), •) helps to estimate θ ⋆ compared to using samples (X, Y ) ∼ p ⊗2 θ ⋆ only.
Intuitively, given observations with null preference gradient, parameters close to θ ⋆ could have similar preferences. Therefore, those samples are not sufficient to discriminate between them. For that, let
G 0 (θ ⋆ ) = {(x, y) ∈ X 2 | |ℓ θ ⋆ (x, y)| > 0} (resp. G 1 (θ ⋆ ) = {(x, y) ∈ G 0 (θ ⋆ ) | ∥∇ θ ⋆ ℓ θ ⋆ (x, y)∥ > 0})
be the set of pairs with non-zero preference (resp. gradient) function. For observations in G 0 (θ ⋆ ) ∁ , the preference is zero, hence uninformative. For observations in G 1 (θ ⋆ ) ∁ , the preference is locally independent of the parameter. Therefore, they do not provide gradient information to distinguish θ ⋆ from a neighboring alternative parameter. Only the preferences of samples in G 1 (θ ⋆ ) can provide information on θ ⋆ , hence preference learning is meaningful if these samples are observed, i.e.,
P p ⊗2 θ ⋆ (G 1 (θ ⋆ )) > 0 for all θ ⋆ ∈ Θ.
Negative examples. The above condition is restrictive both on ℓ θ and p θ , even when considering r θ = log p θ . For example, taking p θ as the uniform distribution over [0, θ], we have
P p ⊗2 θ (G 1 (θ)) = 0.
this section cite: []

Section: Sample-only MLE
In the absence of preference observations, a natural baseline is to estimate θ ⋆ directly from the observations. Given
(X i , Y i ) i∈[n] ∼ p ⊗2n θ ⋆ , the sample-only (SO) MLE is θ SO n ∈ arg min θ L SO n (θ) with L SO n (θ) := - i∈[n] log p ⊗2 θ (X i , Y i ) . (SO MLE) Asymptotic normality. Under enough regularity (Van der Vaart, 2000), SO MLE is asymptotically normal, i.e., √ n( θ SO n -θ ⋆ ) ⇝ n→+∞ N (0 k , I(p ⊗2 θ ⋆ ) -1 ) , where I(p θ ) := E p θ [-∇ 2 θ log p θ ]
is the Fisher information matrix of p θ and ⇝ denote the convergence in distribution. Let ⪰ denote the Loewner order on p.s.d. matrices. By the Cramér-Rao bound (Rao, 1992), SO MLE has optimal asymptotic covariance among the class of unbiased sampleonly estimators, i.e., all sample-only unbiased estimator with asymptotic variance V satisfy V ⪰ I(p ⊗2 θ ⋆ ) -1 . While asymptotic guarantees provide insight into estimator behavior as n → ∞, they do not capture performance in the relevant regime of moderate sample sizes. Modern statistics gives meaningful non-asymptotic concentration results on empirical estimators, e.g., for high-dimensional statistics (Vershynin, 2018;Wainwright, 2019).
this section cite: ['b34', 'b48', 'b50']

Section: Regularity conditions.
The asymptotic statistics literature has devised weak regularity conditions under which asymptotic normality holds. "Classical conditions" assume stronger conditions, e.g., θ → log p θ (x) is three times continuously differentiable for every x ∈ X and the integral of its third derivative converges uniformly for all θ (Van der Vaart, 2000, Chapter 5.6). Those "weak" or "classical" conditions ensure that integrals and derivatives can be exchanged, and Taylor approximations around θ ⋆ are well controlled. Throughout this paper, we use "under enough regularity" to refer to these regularity conditions on both p θ and ℓ θ .
For preferences based on the reward r θ = log p θ , the regularity of p θ implies the one of the preference ℓ θ due to the properties of the logarithm. Moreover, those regularity conditions are satisfied for numerous well-known distributions such as F N ,Σ and F Lap,b . When studying deterministic preferences, we introduce general geometric assumptions on p θ and ℓ θ . Since these conditions are inherently more restrictive, our goal is not to identify the weakest possible regularity assumptions under which our derivations hold.
this section cite: []

Section: Preference-based M-estimator
In this section, we investigate when preference-based estimators can improve upon sample-only estimators. Given preference-labeled observations {(X i , Y i , Z i )} i∈[n] , we define the stochastic preferences MLE (SP MLE) as
θ SP n ∈ arg min θ L SP n (θ) with L SP n (θ) := L SO n (θ) - i∈[n] log σ(Z i ℓ θ (X i , Y i )) . (SP MLE)
This objective extends SO MLE by adding a preferencebased term: a binary classification loss using the logistic function (-log σ(x)). When preferences are stochastic, this estimator corresponds to the MLE under a probabilistic preference model, justifying its name. Under sufficient regularity, M-estimators achieve asymptotic normality, so our goal is to obtain lower asymptotic covariance for SP MLE than for SO MLE. In addition, we want to show that SP MLE reaches a lower asymptotic covariance for deterministic preferences than for stochastic preferences.
this section cite: []

Section: Stochastic Preferences
Under stochastic feedback, we are given noisy preference observations (X i , Y i , Z i ) i∈[n] ∼ q ⊗n θ ⋆ ,hsto , where h sto is defined in Equation (2). SP MLE is a specific instance of M-estimator. Under enough regularity (Van der Vaart, 2000, Chapter 5.5), SP MLE is asymptotically normal, i.e., √ n( θ SP n -θ ⋆ ) ⇝ n→+∞ N (0 k , I(q θ ⋆ ,hsto ) -1 ) , where I(q θ,hsto ) := E q θ,h sto [-∇ 2 θ log q θ,hsto ] denotes the Fisher information matrix of q θ,hsto . By the Cramér-Rao bound (Rao, 1992), this variance is optimal among unbiased estimators that rely on stochastic preferences. Lemma 3.1 compares its efficiency to the sample-only MLE.
Lemma 3.1. Let ∆ SP θ := E p ⊗2 θ ⋆ [σ(ℓ θ )σ(-ℓ θ )∇ θ ℓ θ ∇ θ ℓ T θ ]. Then, I(q θ ⋆ ,hsto ) = I(p ⊗2 θ ⋆ ) + ∆ SP θ ⋆ . The p.s.d. matrix ∆ SP θ ⋆ is definite if P p ⊗2 θ ⋆ (|⟨u, ∇ θ ⋆ ℓ θ ⋆ ⟩| > 0) > 0 for all u ∈ S k-1 . Lemma 3.1 shows that I(q θ ⋆ ,hsto ) ⪰ I(p ⊗2
θ ⋆ ) and exhibits a condition under which θ SP n is asymptotically better than θ SO n , meaning that incorporating preference data improves asymptotic efficiency. The condition in Lemma 3.1 ensures that ∇ θ ⋆ ℓ θ ⋆ spans all directions with some probability, making the preference-based estimator asymptotically superior to the sample-only MLE.
For preferences based on the reward r θ = log p θ , this condition holds for both Laplace and Gaussian distributions: ∆ SP θ ⋆ = 4 b 2 ∆ SP Lap(0,1) for F Lap,b (Appendix G), and ∆ SP θ ⋆ = 2Σ 1/2 ∆ SP N (0 d ,I d ) Σ 1/2 for F N ,Σ (Appendix F). Thus, stochastic preferences can improve parameter estimation compared to sample-only estimators. However, nonasymptotic performance can differ, and in practice, the reduction in asymptotic variance may be small, as we investigate empirically in Section 6. Next, we examine whether M-estimators based on deterministic preferences can further improve upon their stochastic counterparts.
this section cite: ['b34']

Section: Deterministic Preferences
We now consider the setting where true preferences are observed, meaning that the preference labels Z i are deterministic. We observe
(X i , Y i , Z i ) i∈[n] ∼ q ⊗[n]
θ ⋆ ,h det , where h det is defined in Equation (1). We use the same M-estimator as in the stochastic setting, θ SP n (θ) ∈ arg min θ L SP n (θ), but now with deterministic preferences. To distinguish this setting, we introduce the notation SP det for the preference-based estimator under deterministic feedback.
Consistency of SP det . Define the population-level objec- tive: M (θ) := E p ⊗2 θ ⋆ [log q θ,hsto (X, Y, sign(ℓ θ ⋆ (X, Y )))]. Under enough regularity, θ SP det n
converges to a maximizer of M (θ) (Van der Vaart, 2000, Chapter 5.2). However, unlike in the stochastic setting, θ ⋆ may not be a maximizer of M since standard regularity conditions on p θ and ℓ θ are insufficient. A sufficient condition for consistency is
E p ⊗2 θ ⋆ [sign(ℓ θ ⋆ )σ(-|ℓ θ ⋆ |)∇ θ ⋆ ℓ θ ⋆ ] = 0 k ,(3)
which holds for F N ,Σ (Appendix F) and F Lap,b (Appendix G) when using reward r θ = log p θ .
Asymptotic variance of SP det . If Equation (3) holds, then under additional regularity conditions (Van der Vaart, 2000, Chapter 5.3) SP det is asymptotically normal with covariance V SP det θ ⋆
given by the following lemma. Lemma 3.2. Let H SP det using reward r θ = log p θ . For Laplace distribution, we have
H SP det θ ⋆ = R SP det θ ⋆ = 0 and ∆ SP det θ ⋆ = 4 b 2 ∆ SP det Lap(0,1) . For Gaussian distributions, we have H SP det θ ⋆ = 0 d×d , ∆ SP det θ ⋆ = 2Σ 1/2 ∆ SP det N (0 d ,I d ) Σ 1/2 and R SP det θ ⋆ = 2Σ 1/2 R SP det N (0 d ,I d ) Σ 1/2 with R SP det
N (0 d ,I d ) ⪰ 0 d×d . Thus, deterministic preferences improve parameter estimation compared to stochastic preferences.
In conclusion, preference-based M-estimators provide asymptotic improvements in estimation efficiency. Next, we explore whether estimators beyond the M-estimation framework can achieve further gains, potentially exceeding the asymptotic normality limitations.
this section cite: []

Section: Beyond M-estimators
While computationally efficient, the SP det estimator does not fully leverage the constraints imposed by deterministic preferences. Unlike in the stochastic setting, deterministic preferences provide separability: there exist parameters that classify training examples perfectly, including θ ⋆ itself. A key limitation of SP det is that, like standard logistic regression, it minimizes a convex surrogate loss (negative log-likelihood). This approach can lead to misclassification of training examples.foot_3 This limitation suggests an opportunity to directly minimize the 0-1 lossfoot_4 , potentially achieving faster rates of convergence.
0-1 loss minimization. Given (X i , Y i , Z i ) i∈[n] ∼ q ⊗[n]
θ ⋆ ,h det , we consider the set C n of parameters that minimize the empirical 0 -1 loss, i.e.,
C n := arg min θ∈Θ i∈[n] 1 (Z i ℓ θ (X i , Y i ) < 0) (4) = {θ ∈ Θ | ∀i ∈ [n], Z i ℓ θ (X i , Y i ) ≥ 0} ,
which is non-empty as θ ⋆ ∈ C n . Parameters θ ∈ C n perfectly classify all training examples. Any estimator θ AE n ∈ C n is referred to as an arbitrary estimator (AE).
Alternatively, we constrain MLE to this feasible set, defining the deterministic preferences MLE (DP MLE), i.e.,
θ DP n ∈ arg min{L SO n (θ) | θ ∈ C n } (DP MLE)
if θ SO n / ∈ C n , and θ DP n := θ SO n otherwise. This estimator minimizes the negative log-likelihood of the samples while ensuring perfect preference classification. For Gaussian with r θ = log p θ , θ DP n estimates θ ⋆ better than θ SO n for all n, i.e., DP MLE dominates SO MLE statistically.
Lemma 4.1. For all n ∈ N and almost surely, we have,
for F N ,Σ , ∥ θ DP n -θ ⋆ ∥ Σ ≤ ∥ θ SO n -θ ⋆ ∥ Σ .
For stochastic preferences, minimizing the 0 -1 loss is generally NP-hard, requiring a convex surrogate like the logistic function. However, for deterministic preferences, computing C n is more tractable. If θ → ℓ θ is affine, then C n is a convex polytope, defined by at most n half-space constraints. For Gaussian-based preferences, i.e., r θ = log p θ and F N ,Σ , we have
Z i ℓ θ (X i , Y i ) ≥ 0 if and only if Z i ⟨X i -Y i , θ -Σ -1 (X i + Y i )/2⟩ ≥ 0.
Consistency of 0 -1 loss minimization. Define the disagreement probability between θ and θ ⋆ as m(θ) :=
P p ⊗2 θ ⋆ (D(θ ⋆ , θ)) where D(θ ⋆ , θ) := {(x, y) ∈ X 2 | ℓ θ ⋆ (x, y)ℓ θ (x, y) < 0} is the set of observations where θ and θ ⋆ assign informative yet opposite preferences. Under enough regularity (Van der Vaart, 2000, Chapter 5.2), θ AE n and θ DP n converge in C(θ ⋆ ) := {θ ∈ Θ | m(θ) = 0}, which is the non-empty set of minimizers of m(θ) as m(θ) ≥ 0 = m(θ ⋆ ). We note the set C(θ ⋆ ) contains θ ⋆ , but possibly others. To ensure consistency ( θ AE n , θ DP n → θ ⋆ ), we impose the following identifiability assumption that guarantees C(θ ⋆ ) = {θ ⋆ }. Assumption 4.2 (Identifiability). For all θ ̸ = θ ⋆ , m(θ) > 0.
When r θ = log p θ , it holds for both Gaussian (F N ,Σ ) (Appendix F) and Laplace (F Lap,b ) (Appendix G) cases.
this section cite: []

Section: Fast estimation rate.
Once consistency is established, the next goal is to analyze the convergence rate of the estimation errors ∥ θ AE n -θ ⋆ ∥ and ∥ θ DP n -θ ⋆ ∥. Since these are not Mestimators, they are not necessarily limited to the typical parametric rate Ω(1/ √ n).
Theorem 4.3 states our main result for Laplace and Gaussian distributions when using log-probability rewards, i.e., a highprobability accelerated rate in O(1/n). Theorem 4.3. Let δ ∈ (0, 1). For F Lap,1 and F N ,1 , we have, for all n ≥ O(log(1/δ)), with probability 1 -δ,
∀ θ n ∈ C n , n| θ n -θ ⋆ | = O (log(1/δ)) .
For F N ,Σ with d > 1, there exists positive
A d = d→+∞ O( √ d) such that, for all n ≥ O(log(1/δ)), with probability 1 -δ, ∀ θ n ∈ C n , n∥ θ n -θ ⋆ ∥ Σ ≤ O (A d log(1/δ) log n) . Theorem 4.3 is a direct corollary of our main result, showing that max θ∈Cn ∥θ -θ ⋆ ∥ = O(1/n) (see Theorem 4.8 below).
It directly guarantees faster convergence rates for both θ AE n and θ DP n . Theorem 4.8 holds under general geometric conditions on p θ and ℓ θ that we introduce with intuitions, while sketching the proof in Section 4.1.
Negative examples. Assumption 4.2 is restrictive both on ℓ θ and p θ , even when considering r θ = log p θ . For example, when all the distributions in F agree on their preferences, sign(ℓ θ (x, y)) is independent of θ. Therefore, we have m(θ) = 0 for all θ ̸ = θ ⋆ , since ℓ θ ⋆ (x, y)ℓ θ (x, y) ≥ 0. Such cases include scenarios where p θ (x) is a monotonic function, e.g., the exponential distribution and the Pareto distribution with a known location, as well as the Laplace distribution with a known location. This motivates later assumptions on the directionality of ∇ θ ⋆ ℓ θ ⋆ for observed samples.
Link to iterative human preference alignment. Many human preference alignment methods build on the Bradley-Terry model for preference, based on rewards. Direct alignment algorithms use variants of the log-likelihood to define the implicit reward of a policy (Rafailov et al., 2023). Choosing ℓ θ (x, y) = log p θ (x) -log p θ (y) coincides with the optimal policy for maximum entropy RL (Swamy et al., 2025). When leveraging offline preference data, the assumption (X, Y ) ∼ p ⊗2 θ ⋆ is unrealistic, as ℓ θ ⋆ is collected from a fixed data set of pairs of observations. However, "online" preference data has become a popular paradigm in the training of recent LLMs. Those iterative alignment procedures rely on the preference data from an earlier model (Dubey et al., 2024). At stage N , the model p θ N is trained based on the preference data for generations by the previous model, i.e., (X, Y ) ∼ p ⊗2
θ N -1 . Under the realizability assumption and without mode collapse, this self-refinement paradigm should converge towards the true model p θ ⋆ . Our setting characterizes the limiting behavior of this iterative process, i.e., preference based on ℓ θ ⋆ for observations from p θ ⋆ . Nonetheless, we do not claim the direct applicability of DP MLE for realistic LLM training.
this section cite: ['b32', 'b41', 'b7']

Section: Upper Bound on the Estimation Error
We establish a high-probability upper bound on the estimation error max θ∈Cn ∥θ -θ ⋆ ∥ in the general case. This requires grasping the geometry of C n relative to θ ⋆ . Linearized feasibility set. Since C n is defined by nonlinear preference constraint, analyzing its geometry is challenging, and we thus consider a linearized approximation of it. We define the linearized constraint set as
C n := {θ ∈ Θ | ∀i ∈ [n], (X i , Y i ) / ∈ D(θ ⋆ , θ)} , where D(θ ⋆ , θ) := {(x, y) ∈ X 2 | ℓ θ ⋆ (x, y) 2 + ℓ θ ⋆ (x, y)⟨θ -θ ⋆ , ∇ℓ θ ⋆ (x, y)⟩ < 0}.
This set replaces ℓ θ with its first-order Taylor expansion around θ ⋆ , neglecting higher-order terms. A key assumption is that the true constraints are at least as strong as the linearized ones. This ensures C n ⊆ C n , allowing us to control C n via C n . Assumption 4.4 (Linearization validity). For all θ ̸ = θ ⋆ , D(θ ⋆ , θ) ⊆ D(θ ⋆ , θ).
this section cite: []

Section: Directional analysis and informative constraints.
To quantify the geometry of C n relative to θ ⋆ , we analyze deviations along directions u ∈ S k-1 . Define the set of informative samples along direction u:
G 1 (θ ⋆ , u) := {(x, y) | ℓ θ ⋆ (x, y)⟨u, ∇ θ ⋆ ℓ θ ⋆ (x, y)⟩ < 0} .
This set contains observations whose preferences give information along the direction u. Assuming that preferences are informative along all directions, we prevent degenerate cases where some directions lack preference information. Assumption 4.5 (Informative Preferences). For all u ∈ S k-1 ,
P p ⊗2 θ ⋆ (G 1 (θ ⋆ , u)) > 0.
Deviation bound via minimum informative sample. Define R n,u as the maximal deviation from θ ⋆ within C n along the direction u, i.e.,
R n,u := max{ε ≥ 0 | θ ⋆ + εu ∈ C n } .
We define the scaling factor
∀(x, y) ∈ G 1 (θ ⋆ , u), V θ ⋆ ,u (x, y) := ℓ θ ⋆ (x, y) -⟨u, ∇ θ ⋆ ℓ θ ⋆ (x, y)⟩ .
The value V θ ⋆ ,u (X i , Y i ) quantifies the amount of information in the preference between X i and Y i to discriminate θ ⋆ from other parameters on the half-line directed by u. The lower V θ ⋆ ,u (X i , Y i ) is, the more discriminative is the preference between X i and
Y i . Since (x, y) ∈ G 1 (θ ⋆ , u) \ D(θ ⋆ , θ ⋆ + εu) if and only if V θ ⋆ ,u (x, y) ≥ ε, we obtain R n,u ≤ min i∈[n] {V θ ⋆ ,u (X i , Y i ) | (X i , Y i ) ∈ G 1 (θ ⋆ , u)} .
Therefore, the maximal deviation R n,u is upper bounded by the minimum of positive random variables. It remains to upper bound the resulting value of this minimum with high probability and conclude provided some regularities hold, e.g., positive density at zero. By analyzing the distribution of V θ ⋆ ,u , we derive the following probabilistic bound. Lemma 4.6. Suppose Assumption 4.5 hold. For all u ∈ S k-1 , with probability 1 -δ, Assumption 4.7 (Positive density at zero and regularity of inverse c.d.f.). For all u ∈ S k-1 , F ′ θ ⋆ ,u (0) ∈ (0, +∞) and there exists (
x θ ⋆ ,u , M θ ⋆ ,u ) ∈ (0, 1) × R + such that sup x∈[0,x θ ⋆ ,u ] |(F -1 θ ⋆ ,u ) ′′ (x)| ≤ M θ ⋆ ,u .
Using this assumption and (F -1 θ ⋆ ,u ) ′ (0) = 1/F ′ θ ⋆ ,u (0), the first-order Taylor expansion with remainder yields
∀x ∈ [0, x θ ⋆ ,u ], |F -1 θ ⋆ ,u (x) -x/F ′ θ ⋆ ,u (0)| ≤ M θ ⋆ ,u x 2 /2 .
This argument leads to our main theorem, directly for k = 1 and using a covering argument for k > 1. Theorem 4.8. Suppose Assumptions 4.2, 4.4, 4.5 and 4.7 hold. Let δ ∈ (0, 1). Let γ > 0 and N (γ) be the γ-covering number of Θ for the norm
∥ • ∥. Let A -1 θ ⋆ = min u∈S k-1 F ′ θ ⋆ ,u (0), B -1 θ ⋆ = min u∈S k-1 x θ ⋆ ,u and C θ ⋆ = max u∈S k-1 M θ ⋆ ,u /2. When k = 1, for all n ≥ B θ ⋆ log(2/δ), max θ∈Cn ∥θ -θ ⋆ ∥ ≤ A θ ⋆ n log(2/δ) + C θ ⋆ n 2 log(2/δ) 2 , with probability 1 -δ. When k > 1, for all n ≥ B θ ⋆ log(N (γ)/δ), with probability 1 -δ, max θ∈Cn ∥θ-θ ⋆ ∥ ≤ γ + A θ ⋆ n log N (γ) δ + C θ ⋆ n 2 log N (γ) δ 2 .
When k > 1, the choice of the optimal parameter γ depends on the norm ∥ • ∥. Since Θ is bounded by B Θ , N (γ) is upper bounded by the covering of the ball having diameter B Θ . As an example, let N 2 (γ) be the ε-covering number of the unit ball in R k for the Euclidean norm. Then, it is known that log
N 2 (ε) ≈ log(ε 2 k)/ε 2 if ε ≳ 1/ √ k and log N 2 (ε) ≈ k log 1 ε 2 k if ε ≲ 1/ √ k. 3 Therefore, optimiz- ing over γ yields an upper bound on max θ∈Cn ∥θ-θ ⋆ ∥ 2 scal- ing as O(A θ ⋆ k/n) when n ≳ A θ ⋆ k 3/2 , and O((A θ ⋆ /n) 1/3
) otherwise, where Õ(•) hides logarithmic terms. For large sample size compared to the dimension, i.e., n ≳ A θ ⋆ kfoot_6/2 , we recover a rate of O(1/n).
In conclusion, we have derived generic assumptions under which the rate of decay of the estimation error of θ AE n and
θ DP n is in O(1/n). This is a significant improvement com- pared to the asymptotic normality of the SP det estimator that implies a rate of O(1/ √ n).
Positive examples. While Assumptions 4.4, 4.5 and 4.7 are restrictive, they hold for F N ,Σ (Appendix F) and F Lap,b (Appendix G) when using reward r θ = log p θ . This yields Theorem 4.3. We have A θ ⋆ = 2b, B θ ⋆ = 8 and C θ ⋆ = 16b for F Lap,b , and
A θ ⋆ = π(d-1)Γ(d/2) 2Γ((d-1)/2) = +∞ O( √ d) for F N ,Σ , hence a rate in O(d 3/2 /n) when n ≫ d 2 .
Extended discussions. In Appendix B, we discuss how to verify or weaken our assumptions (Appendix B.1), the sources of misspecification (Appendix B.2) and other reward models than log-likelihood (Appendix B.3).
this section cite: []

Section: Lower Bound for Deterministic Feedback
In this section, we show that the rate O(1/n) is minimax optimal (up to a logarithmic factor) by deriving a matching lower bound. The standard approach to minimax lower bounds in estimation relies on Fano-type inequalities and hypothesis testing reductions. However, due to Assumption 4.2, the Kullback-Leibler divergence and χ 2 distance between q θ ⋆ and q θ are infinite for θ ̸ = θ ⋆ , making these tools ineffective. Instead, we use Assouad's Lemma (Tsybakov, 2009), which provides lower bounds via the total variation distance (defined as TV(P, Q) := ∥P -Q∥ 1 for distributions P and Q). Since TV is not well-behaved for product distributions, we use the squared Hellinger distance, defined as H 2 (P, Q) := 1 2 ∥ √ P -√ Q∥ 2 2 , which satisfies
TV(P ⊗n , Q ⊗n ) ≤ 2H 2 (P ⊗n , Q ⊗n ) ≤ 2nH 2 (P, Q).
For further analytical convenience, we also employ the Bhattacharyya coefficient, BC(P, Q) := √ PQ 1 , which is related to the Hellinger distance by H 2 (P, Q) = 1-BC(P, Q). Since q θ ⋆ q θ is zero for disagreeing preferences, we define the restricted BC as
BC( θ, θ) := E p ⊗2 θ 1 D( θ, θ) ∪ D 0 ( θ, θ) p ⊗2 θ /p ⊗2 θ , where D 0 ( θ, θ) := G 0 ( θ) ∁ △G 0 (θ) ∁
is the set where the preferences are zero for exactly one parameter.
Lemma 5.1 decomposes the Hellinger distance between two distributions over the preference triplets into the Hellinger distance between sample-only distributions and the disagreement restricted Bhattacharyya coefficient.
Lemma 5.1. H 2 (q θ , q θ ) = BC( θ, θ) + H 2 (p ⊗2 θ , p ⊗2 θ ) for all θ, θ ∈ Θ. As H 2 (p ⊗2 θ , p ⊗2 θ ) ≤ 2H 2 (p θ , p θ )
, deriving a lower bound requires controlling H 2 (p θ , p θ ) and BC( θ, θ), hence, we impose the following assumption. Assumption 5.2. There exists positive constants c 1 , c 2 independent of k and a dimension and problem-dependent scaling function α F (k) such that for all θ, θ ∈ Θ,
BC( θ, θ) + H 2 (p ⊗2 θ , p ⊗2 θ ) ≤ c 1 α F (k) ∥θ -θ∥ + c 2 ∥θ -θ∥ 2 .
Theorem 5.3 bounds the minimax estimation error. Theorem
5.3. Let R max := inf θ sup θ ⋆ ∈Θ E q θ ⋆ [∥ θ -θ ⋆ ∥].
Suppose Assumption 5.2 holds. Then,
R max ≥ Ω min α F (k) √ k n , k n .
This result confirms that the O(1/n) rate is minimax optimal up to logarithmic factors. The scaling α F (k) comes from BC(θ ⋆ , θ) (Assumption 5.2), yet it is challenging to link α F (k) with A θ ⋆ without further assumptions.
Positive examples. While Assumption 5.2 is restrictive, even when using rewards r θ = log p θ , it holds for F Lap,b (Appendix G), i.e.,
BC(θ ⋆ , θ) = |θ ⋆ -θ|/(2b) + O(|θ ⋆ -θ| 2 ) ,
as well as for F N ,Σ (Appendix F), i.e.,
BC(θ ⋆ , θ) = 2e -∥θ ⋆ -θ∥ 2 Σ /4 F θ ⋆ ,u (∥θ ⋆ -θ∥ Σ ) , with F θ ⋆ ,u (ε) ≤ ε/A θ ⋆ and α F (d) = A θ ⋆ .
this section cite: ['b46']

Section: Dimensionality gap.
While the lower bound in Theorem 5.3 scales as Ω(α F (k)
√ k/n) for n ≥ α F (k) 2 , the upper bound in Theorem 4.8 scales as O(A θ ⋆ k/n) for n ≫ A θ ⋆ k 3/2 .
Even for the simple case of Gaussian distributions where A θ ⋆ = α F (d), there is a dimensionality gap. Closing this gap is an important direction for future work. Improvements might come from a tighter analysis, e.g., both for the upper and lower bounds, or the derivation of better estimators based on deterministic preferences.
this section cite: []

Section: Experiments
In this section, we compare the empirical performance of the different estimators introduced in this paper. For preferences based on r θ = log p θ , we conduct a set of experiments for Gaussian distributions, and defer to Appendix H.1 for experiments on Laplace and Rayleigh distributions. In particular, we consider a uniformly drawn mean parameter θ ⋆ ∼ U([1, 2] d ) and the isotropic covariance Σ = I d . For sample size n ∈ [N max ] with N max = 10 4 , we compute the estimation errors ∥ θ n -θ ⋆ ∥ 2 . We repeat this process for N runs different instances and for various choices of d.
For F N ,I d (Appendix F), the M-estimators can be implemented as
θ SO n = 1 2n i∈[n] (X i + Y i ), θ SP n = arg min θ ∥θ -θ SO n ∥ 2 2 - 1 n i∈[n] log σ(Z i ℓ θ (X i , Y i )) , where ℓ θ (X i , Y i ) = ⟨X i -Y i , θ -(X i + Y i )/2⟩. Then, the estimators based on C n = {θ | ∀i ∈ [n], Z i ℓ θ (X i , Y i ) ≥ 0} are θ DP n = arg min θ∈Cn ∥θ -θ SO n ∥ 2
2 and an arbitrary estimator θ AE n ∈ C n . As C n is an interval for d = 1, we use the randomized uniform (RU) estimator, i.e., θ RU n ∼ U (C n ). We also consider the worst-case estimator (WE), defined as θ WE n := arg max θ∈Cn ∥θ -θ ⋆ ∥ 1 . While it is not a valid estimator due to its θ ⋆ dependency, it serves as a proxy for the worst estimation error in C n .  This further highlights the weakness of asymptotic results compared to non-asymptotic guarantees.
this section cite: []

Section: Dependency on sample size.
While Figure 1(a) suggests that RU and WE perform on par with DP MLE, Figures 1(b) and 2 highlight that DP MLE outperforms WE and AE for larger dimensions, where the gap increases when d is nonnegligible compared to n. We conjecture that RU suffers from the same limitation as AE for larger d. As empirical evidence, we study other estimators that disentangle the effect of RU's randomness versus its mean behavior, see Appendix H.2.
this section cite: []

Section: Dependency on dimension.
Figure 2 strengthens the aforementioned empirical observations. For fixed sample size and increasing dimension, DP MLE is the only estimator obtaining the best-of-both world estimation error rate, i.e., O(min{d 3/2 /n, d/n}).
Covariance gap. We show that the covariance gap between SP MLE and SO MLE is relative mild:
(∆ SP Lap(0,1) , ∆ SP det Lap(0,1) ) ≈ (0.16, 0.08) and (∆ SP N (0,1) , ∆ SP det N (0,1) , R SP det N (0,1) ) ≈ (0.17, 0.08, 0.10). Moreover, ∆ SP N (0 d ,I d ) , ∆ SP det N (0 d ,I d ) and R SP det N (0 d ,I d ) are close to α d I d where α d > 0 is decreasing in d (see Figure 3 in Appendix F). In addition to having a small empirical gap for a moderate value of n, the asymptotic gaps between SO MLE and SP MLE are mild. Supplementary experiments. Following the approach of Tang et al. (2024a), we compare estimators using other convex surrogates of the 0-1 loss (Appendix H.3): they all perform similarly. For the logistic loss, we showcase the "mild" impact of normalization and regularization (Appendix H.4).
this section cite: []

Section: Perspectives
This work investigates the role of preference feedback in parameter estimation for continuous parametric distributions. We establish conditions under which preference-based estimators outperform sample-only methods. For stochastic preferences, the preference-based MLE achieves a lower asymptotic variance than its sample-only counterpart. For deterministic preferences, we demonstrate that preferencebased estimators can significantly accelerate parameter estimation, achieving an improved O(1/n) convergence rate compared to the O(1/ √ n) rate of M-estimators. Our lower bound analysis further confirms that this acceleration is minimax optimal up to dimension-dependent constants.
While our results provide a solid theoretical foundation, several open questions remain. A finer analysis of beyond-M-estimators and their constraint set geometry would allow to better quantify the properties of DP MLE, and provide insights for designing improved estimators that better leverage deterministic preferences. Additionally, exploring alternative preference functions beyond the log-probability gap could extend the applicability of our results.
Finally, a key challenge for future work is to quantify the benefits of preference-based estimation for discrete distributions. For distributions with small support, preference feedback may only localize the unknown parameter within a subset of the simplex, leading to diminishing information gains as the sample size increases. However, understanding how preference-based estimators perform in finite-sample settings, particularly in high-dimensional problems, remains an interesting open problem. Addressing these questions could provide further insights into the role of preferences in machine learning and statistical estimation.
this section cite: []

Section: A. Outline
The appendices are organized as follows:
• In Appendix B, we provide detailed discussions on our assumptions, the sources of misspecification and other reward models.
• In Appendix C, we prove the general results presented in Section 3 such as Lemma 3.1.
• In Appendix D, we focus on Section 4 and detail the proofs of Lemma 4.6 and Theorem 4.8
• In Appendix E, we prove the results presented in Section 5.
• In Appendix F, for F N ,Σ and preferences based on r θ = log p θ , we prove all the assumptions introduced in this paper.
• In Appendix G, for F Lap,b and preferences based on r θ = log p θ , we prove all the assumptions introduced in this paper.
• In Appendix H, we provide supplementary experiments to support our theoretical findings.
this section cite: []

Section: B. Extended Discussions
We provide detailed discussions on how to verify or weaken our assumptions (Appendix B.1), the sources of misspecification (Appendix B.2) and other reward models than log-likelihood (Appendix B.3).
this section cite: []

Section: B.1. Verifying or Weakening our Assumptions
Since our assumptions are restrictive, it is natural to wonder how they can be verified or weakened.
this section cite: []

Section: Verifying our assumptions.
Even a closed-form definition of p θ and ℓ θ is given, our assumptions are challenging to verify, hence we suggest using a formal verifier (e.g., Lean (mathlib Community, 2020)) or software (e.g., SageMath (The Sage Developers, 2022)). Empirically, they can be confirmed or rejected by sampling from p ⊗2 θ ⋆ . Assumption 4.4 is rejected by exhibiting (X i , Y i ) ∈ D(θ ⋆ , θ) \ D(θ ⋆ , θ). Assumptions 4.2 and 4.5 are confirmed by finding (X i , Y i ) ∈ D(θ ⋆ , θ) and (X i , Y i ) ∈ G 1 (θ ⋆ , u). Those tests' sampling complexity scales as the inverse event's probability. Using Dvoretzky-Kiefer-Wolfowitz inequality (Dvoretzky et al., 1956), F θ ⋆ ,u can be estimated to verify that Assumption 4.7 holds.
this section cite: ['b8']

Section: Restrictive assumptions.
When studying DP MLE only, we conjecture that the "global" Assumptions 4.2 and 4.4 can be weakened to local versions. Using time-uniform concentration results, we can build a sequence of shrinking confidence regions (R n ) n around SO MLE that contains θ ⋆ for all time n with high probability. Then, we modify DP MLE to be constrained on R n ∩ C n . For n large enough and with high probability, R n ∩ C n will be included in a local neighborhood of θ ⋆ under which the "local" Assumptions 4.2 and 4.4 are satisfied. Given that Assumption 4.4 is based on "ignoring" the reminder term in a first-order Taylor expansion, assuming a local version is a significantly weaker requirement.
this section cite: []

Section: B.2. Sources of Misspecification
There are several possible sources of misspecification not taken into account by our current analysis.
Preference model. The Bradley-Terry model that uses reward-based preferences has limited expressivity as it doesn't allow for intransitive preferences. Even when individuals exhibit transitive preferences, their averaged preferences can be intransitive due to disagreements, see Munos et al. (2024) or Swamy et al. (2024).
Parameter space. When θ ⋆ / ∈ Θ, the deterministic preferences might not provide separability within Θ. The definition of DP MLE should be modified to combine the cross-entropy loss and the classification 0-1 loss, i.e.,
θ DP n ∈ arg min θ∈Θ    L SO n (θ) + λ i∈[n] 1 (Z i ℓ θ (X i , Y i ) < 0)    ,(5)
where λ > 0 is a regularization between those two losses. Equation ( 5) is reminiscent of single-stage alignment procedures such as ORPO (Hong et al., 2024) and ASFT (Wang et al., 2024), see, e.g., Gorbatovski et al. (2025). Without separability, solving Eq. ( 5) can be NP-hard. Under sufficient regularity, θ DP n converges to θ 0 ∈ arg min θ∈Θ {KL(θ ⋆ , θ) + λm(θ)} where m(θ) = P p ⊗2 θ ⋆ (D(θ ⋆ , θ)) and θ 0 ̸ = θ ⋆ . As θ → m(θ) can be non-convex, computing θ 0 might be challenging. Deriving a tractable ELBO method for this optimization is an interesting direction to obtain tractable and robust estimators. As θ 0 lies in the boundary of Θ, we should control the maximal deviation with respect to θ 0 for directions that point towards the interior of Θ to prove an accelerated rate. While parts of our analysis could be used, we believe that finer technical arguments are required.
Parametric model. The true distribution p ⋆ of the observations might not even be a member of our class of distributions F, i.e., p ⋆ / ∈ F. These situations occur when F doesn't contain the true structure, e.g., other parametric or non-parametric class of distributions. Then, L SO n (θ) can be interpreted as a quasi-log-likelihood term. Let us denote by SO quasi-MLE the estimator based on SO MLE for this quasi-log-likelihood. Under sufficient regularity, SO quasi-MLE converges towards θ 0 ∈ arg min θ∈Θ KL(p ⋆ , p θ ) where p ⋆ ̸ = p θ0 ∈ F. Without the separability from well-specified deterministic preference, we define DP quasi-MLE as in Eq. ( 5). Under sufficient regularity, DP quasi-MLE converges towards the minimizer of a similar optimization problem combining the KL term and a misspecified equivalent of m(θ).
this section cite: ['b29', 'b40', 'b16', 'b51', 'b14']

Section: B.3. Reward models
Except for Theorem 4.3, all the derivations in Section 4 hold for general (hence reward-based) preference models provided Assumptions 4.2, 4.4, 4.5 and 4.7 hold. Characterizing the expressivity of parametric rewards satisfying those assumptions is interesting, yet challenging. We provide two positive and one negative examples.
Positive: monotonic reward. Suppose that lθ (x, y) = f (p θ (x)) -f (p θ (x)) where f is increasing on [0, 1]. Since sign( lθ ) = sign(ℓ θ ), hence the parameters with zero classification loss and our estimators are the same. Therefore, our results hold for this class of rewards when our assumptions hold for the log-likelihood reward. When f is decreasing, the preferences are "reversed", and similar arguments can be made. This example includes (1) normalization by a multiplicative constant (e.g., temperature β) and (2) the odds-ratio reward-based preference based on f (x) = log(x/(1 -x)) used by ORPO in Hong et al. (2024).
Positive: margin with Gaussian. Suppose that lθ = ℓ θ + c where c is a constant and ℓ θ is the Gaussian log-likelihood preference. By extending our computations from Appendix F, Assumptions 4.2, 4.4, 4.5 and 4.7 hold with c-dependent positive constants. Margins are used by SimPO from Meng et al. (2024) and IPO from Azar et al. (2024).
Negative: reference model with Gaussian. Suppose that lθ = ℓ θ -ℓ θ0 where θ 0 is known and ℓ θ is the Gaussian log-likelihood preference. Since lθ (x, y) = ⟨x -y, θ -θ 0 ⟩ and ∇ θ lθ (x, y) = x -y, Assumption 4.5 is violated for
u = θ ⋆ -θ 0 , i.e., P p ⊗2 θ ⋆ (G 1 (θ ⋆ , θ ⋆ -θ 0 )) = 0.
Not all direct alignment algorithms rely on a reference model, see, e.g., SimPO and ORPO.
C. Proofs of Section 3 C.1. Stochastic Preferences Under enough regularity, by swapping the integration and the differentiation operators, we can show that
E q θ,h sto [∇ θ log q θ,hsto ] = ∇ θ 1 = 0 k and E q θ,h sto [-∇ 2 θ log q θ,hsto ] = E q θ,h sto [∇ θ log q θ,hsto ∇ θ log q T θ,hsto ] .
Below, we detail the proof of Lemma 3.1.
Proof. Direct computation yields that
∇ θ ⋆ log q θ ⋆ (x, y, z) = ∇ θ ⋆ log p ⊗2 θ ⋆ (x, y) + zσ(-zℓ θ ⋆ (x, y))∇ θ ⋆ ℓ θ ⋆ (x, y) , ∇ 2 θ ⋆ log q θ ⋆ (x, y, z) = ∇ 2 θ ⋆ log p ⊗2 θ ⋆ (x, y) -σ(ℓ θ ⋆ (x, y))σ(-ℓ θ ⋆ (x, y))∇ θ ⋆ ℓ θ ⋆ (x, y)∇ θ ⋆ ℓ θ ⋆ (x, y) T + zσ(-zℓ θ ⋆ (x, y))∇ 2 θ ⋆ ℓ θ ⋆ (x, y) .
where we used that g ′ (x) = σ(-x) and g ′′ (x) = -σ ′ (-x) = -σ(x)σ(-x) with g(x) = log σ(x). By definition of h sto ,
E Z|(X,Y ) Zσ(-Zℓ θ ⋆ (X, Y ))∇ 2 θ ⋆ ℓ θ ⋆ (X, Y ) = (σ(-ℓ θ ⋆ (X, Y ))σ(ℓ θ ⋆ (X, Y )) -σ(ℓ θ ⋆ (X, Y ))σ(-ℓ θ ⋆ (X, Y ))) ∇ 2 θ ⋆ ℓ θ ⋆ (X, Y ) = 0 d×d .
Therefore, we have I(q θ ⋆ ,hsto
) = I(p ⊗2 θ ⋆ ) + ∆ SP θ ⋆ with ∆ SP θ = E p ⊗2 θ ⋆ [σ(ℓ θ )σ(-ℓ θ )∇ θ ℓ θ (∇ θ ℓ θ ) T ].
For all x ∈ R k , we have
x T ∆ SP θ x = ∥x∥ 2 E p ⊗2 θ ⋆ [σ(ℓ θ )σ(-ℓ θ )⟨x/∥x∥, ∇ θ ℓ θ ⟩ 2 ] ≥ 0 .
It is direct to see that this inequality is strict except if
P p ⊗2 θ ⋆ (⟨x/∥x∥, ∇ θ ℓ θ ⟩ 2 = 0) = 1. Therefore, ∆ SP θ ⋆ is a positive definite matrix if P p ⊗2 θ ⋆ (|⟨u, ∇ θ ⋆ ℓ θ ⋆ ⟩| > 0) > 0 for all u ∈ S k-1 .
Note that this condition is implied by Assumption 4.5.
In the following, we consider the special case where H SP det θ ⋆ = 0 d×d . This occurs when θ → ℓ θ is linear, e.g., for F N ,Σ and F Lap,b and preferences based on r θ = log p θ . Then, the condition (6) rewrites as
R SP det θ ⋆ + ∆ SP det θ ⋆ ≻ 0 d×d with ∆ SP det θ ⋆ := 2∆ SP θ ⋆ -M 2,θ ⋆ .
Using that min x∈R σ(|x|) = 1/2 achieved only at x = 0, we have directly that, for all x ∈ R k ,
x T ∆ SP det θ ⋆ x = ∥x∥ 2 E p ⊗2 θ ⋆ [(2σ(|ℓ θ ⋆ |) -1)σ(-|ℓ θ ⋆ |)⟨x/∥x∥, ∇ θ ⋆ ℓ θ ⋆ ⟩ 2 ] ≥ 0 , It is direct to see that this inequality is strict except if P p ⊗2 θ ⋆ (ℓ θ ⋆ ⟨x/∥x∥, ∇ θ ⋆ ℓ θ ⋆ ⟩ = 0) = 1. Therefore, ∆ SP det θ ⋆ is a positive definite matrix if P p ⊗2 θ ⋆ (|ℓ θ ⋆ ⟨u, ∇ θ ⋆ ℓ θ ⋆ ⟩| > 0) > 0 for all u ∈ S k-1 .
Then, a sufficient condition for the condition (6) to hold is that R SP det θ ⋆ is a p.s.d. matrix, i.e., R SP det θ ⋆ ⪰ 0 d×d .
In summary, we have derived sufficient conditions for SP det to be asymptotically better than SP, namely H SP det
θ ⋆ = 0 d×d , R SP det θ ⋆ ⪰ 0 d×d and P p ⊗2 θ ⋆ (|ℓ θ ⋆ ⟨u, ∇ θ ⋆ ℓ θ ⋆ ⟩| > 0) > 0 for all u ∈ S k-1 .
Note that this last condition is implied by Assumption 4.5.
this section cite: ['b16', 'b28', 'b1']

Section: D. Proofs of Section 4 D.1. Proof of Lemma 4.1
Proof. For Gaussian distributions, this is a direct consequence of the following facts: θ ⋆ ∈ C n , θ DP n ∈ arg min θ∈Cn ∥θθ SO n ∥ 2 Σ and C n is convex.
this section cite: []

Section: D.2. Proof of Lemma 4.6
Proof.
Let u ∈ S k-1 . Let F θ ⋆ ,u be the c.d.f. of V θ ⋆ ,u (X, Y ) when (X, Y ) ∼ (p ⊗2 θ ⋆ ) |G1(θ ⋆ ,u) , i.e., p ⊗2 θ ⋆ truncated to G 1 (θ ⋆ , u). Then, F θ ⋆ ,u (ε) = F θ ⋆ ,u (ε)/α θ ⋆ ,u . Let α θ ⋆ ,u = P p ⊗2 θ ⋆ (G 1 (θ ⋆ , u)) and N θ ⋆ ,u = i∈[n] 1 ((X i , Y i ) ∈ G 1 (θ ⋆ , u)) ∼ Bin(n, α θ ⋆ ,u ). Let R n,u = min i∈[n],(Xi,Yi)∈G1(θ ⋆ ,u) V θ ⋆ ,u (X i , Y i ).
Using the derivation in Section 4.1, we have that R n,u ≤ R n,u . Let ε > 0. Conditioned on N θ ⋆ ,u , it is direct to see that
P( R n,u > ε | N θ ⋆ ,u ) = 1 -(1 -(1 -F θ ⋆ ,u (ε)) N θ ⋆ ,u ) = 1 -F θ ⋆ ,u (ε) N θ ⋆ ,u .
Using that N θ ⋆ ,u ∼ Bin(n, α θ ⋆ ,u ), E X∼Bin(n,p) [s X ] = (1 -p + ps) n and 1 -x ≤ exp(-x), we obtain that
P(R n,u > ε) ≤ P( R n,u > ε) ≤ 1 -α θ ⋆ ,u F θ ⋆ ,u (ε) n ≤ exp (-nF θ ⋆ ,u (ε)) . Taking ε = F -1 θ ⋆ ,u (min {1, log(1/δ)/n}) concludes the proof.
this section cite: []

Section: D.3. Proof of Theorem 4.8
Proof. For all u ∈ S k-1 , let (A θ ⋆ , B θ ⋆ , C θ ⋆ ) defined as in Theorem 4.8. Since C n ⊆ C n under Assumption 4.4, we obtain that max θ∈Cn ∥θ -θ ⋆ ∥ ≤ max θ∈ Cn ∥θ -θ ⋆ ∥.
Case k = 1. Since |S 0 | = 2, using Lemma 4.6 with a union bound yield that, with probability at least 1 -δ,
max θ∈ Cn ∥θ -θ ⋆ ∥ ≤ max u∈S0 R n,u ≤ max u∈S0 F -1 θ ⋆ ,u (min{1, log(2/δ)/n}) .
Under Assumption 4.7, for n ≥ B θ ⋆ log(2/δ), we can conclude the proof since
n max θ∈Cn ∥θ -θ ⋆ ∥ ≤ n max θ∈ Cn ∥θ -θ ⋆ ∥ ≤ A θ ⋆ log(2/δ) + C θ ⋆ log(2/δ) 2 /n .
Case k > 1. Let N (γ) be the γ-covering number of Θ for the norm ∥ • ∥. Let {θ j } j∈[N (γ)] be such a γ-covering. For all j ∈ [N (γ)], let ε j = ∥θ j -θ ⋆ ∥ and u j = (θ j -θ ⋆ )/ε j . Using triangular inequality, we obtain
max θ∈ Cn ∥θ -θ ⋆ ∥ ≤ γ + max j∈[N (γ)], θj ∈ Cn ∥θ j -θ ⋆ ∥ ≤ γ + max j∈[N (γ)] 1 θ ⋆ + ε j u j ∈ C n ε j ≤ γ + max j∈[N (γ)]
R n,uj .
Using Lemma 4.6 with a union bound yield that, with probability at least 1 -δ,
n max θ∈Cn ∥θ -θ ⋆ ∥ ≤ nγ + max j∈[N (γ)] nF -1 θ ⋆ ,uj (log(N (γ)/δ)/n) ≤ nγ + A θ ⋆ log(N (γ)/δ) + C θ ⋆ log(N (γ)/δ) 2 /n .
where the last inequality relies on Assumption 4.7 for n ≥ B θ ⋆ log(N (γ)/δ).
this section cite: []

Section: E. Proofs of Section 5 E.1. Proof of Lemma 5.1
Proof. It is direct to see that
q θ ⋆ (x, y, z)q θ (x, y, z) = 0 if (x, y) ∈ D(θ ⋆ , θ) ∪ G 0 (θ ⋆ ) ∁ △G 0 (θ) ∁ p θ ⋆ (x)p θ ⋆ (y)p θ (x)p θ (y) otherwise .
Therefore, we have
BC(q θ ⋆ , q θ ) = (x,y) / ∈D(θ ⋆ ,θ)∪(G0(θ ⋆ ) ∁ △G0(θ) ∁ ) p θ ⋆ (x)p θ ⋆ (y)p θ (x)p θ (y)dxdy = BC(p ⊗2 θ ⋆ , p ⊗2 θ ) -BC(θ ⋆ , θ) .
Using that H 2 (P, Q) = 1 -BC(P, Q), we conclude the proof.
this section cite: []

Section: E.2. Proof of Theorem 5.3
Consider the hypercube
Θ ′ = {θ b = δb : b ∈ {0, 1} d } ⊆ Θ. Note that ∥θ b -θ b ′ ∥ ≥ δ √ k d H (b, b ′ ), where d H (b, b ′
) denotes the Hamming distance between b and b ′ . Then using Assouad's lemma we have
R max ≥ δ √ k 4 1 -max d H (b,b ′ )=1 T V (q ⊗n θ b , q ⊗n θ b ′ ) .
Upper bounding TV with H 2 , Lemma 5.1 yields
T V (q ⊗n θ b , q ⊗n θ b ′ ) ≤ n( BC(θ b , θ b ′ ) + H 2 (p ⊗2 θ b , p ⊗2 θ b ′ )). Then, Assumption 5.2 implies R max ≥ δ √ k 4 1 -n c 1 δ α F (k) + 2c 2 δ 2 . Picking δ = 1 2(c1+2c2) min{ α F (k) n , 1 √ n } ensures that the term in parenthesis is always greater than 1/2, hence R max ≥ √ k 8(c 1 + 2c 2 ) min α F (k) n , 1 √ n .
this section cite: []

Section: F. Multivariate Gaussian with Known Covariance
In the following, θ = Σ -1 µ denote the natural parameter of multivariate Gaussian with known covariance matrix Σ. We have X = R d and k
= d. Let θ ∈ Θ and u ∈ S d-1 for the norm ∥ • ∥ Σ , i.e., ∥u∥ Σ = 1. Let S 2,d-1 = {x ∈ R d | ∥u∥ 2 = 1}.
It is direct to see that
ℓ θ (x, y) = log p θ (x) p θ (y) = ⟨x -y, θ -Σ -1 (x + y)/2⟩ and ∇ θ ⋆ ℓ θ ⋆ (x, y) = x -y .
Therefore, we have Proof that
G 0 (θ ⋆ ) = {(x, y) ∈ (R d ) 2 | |⟨x -y, θ ⋆ -(x + y)/2⟩| > 0} , G 1 (θ ⋆ ) = {(x, y) ∈ G 0 (θ ⋆ ) | ∥x -y∥ > 0} , D(θ ⋆ , θ) = (x, y) ∈ (R d ) 2 | ⟨x -y, θ ⋆ -Σ -1 (x + y)/2⟩ 2 + ⟨x -y, θ ⋆ -Σ -1 (x + y)/2⟩⟨θ -θ ⋆ , x -y⟩ < 0 , G 1 (θ ⋆ , u) = {(x, y) ∈ (R d ) 2 | ⟨x -y, θ ⋆ -Σ -1 (x + y)/2⟩⟨u, x -y⟩ < 0} , ∀(x, y) ∈ G 1 (θ ⋆ , u), V θ ⋆ ,u (x, y) = ⟨x -y, Σ -1 (x + y)/2 -θ ⋆ ⟩ ⟨u, x -y⟩ .
P p ⊗2 θ ⋆ (G 1 (θ ⋆ )) > 0. It is direct to see that dim(G 0 (θ ⋆ ) ∁ ) < 2d and dim(G 0 (θ ⋆ ) \ G 1 (θ ⋆ )) < 2d. Given that p ⊗2 θ ⋆ is a continuous distribution on (R d ) 2 , we obtain that P p ⊗2 θ ⋆ (G 1 (θ ⋆ )) = P p ⊗2 θ ⋆ (G 0 (θ ⋆ )) = 1.
Condition in Lemma 3.1. The condition of Lemma 3.1 is implied by Assumption 4.5, hence we refer to the proof of this result below. Therefore, we have I(q θ ⋆ ,hsto ) ≻ I(p ⊗2 θ ⋆ ).
this section cite: []

Section: Consistency of SP det .
To study SP det for F N ,Σ , we use the change of variable
D = Σ -1/2 (X -Y )/ √ 2 and S = √ 2Σ -1/2 (Σθ ⋆ -(X + Y )/2). Then, we have (D, S) ∼ N (0 2d , I 2d ) and ℓ θ ⋆ (X, Y ) = ⟨S, D⟩ , ∇ θ ⋆ ℓ θ ⋆ (X, Y ) = √ 2Σ 1/2 D , ∇ θ ⋆ log p ⊗2 θ ⋆ (X, Y ) = 2Σθ ⋆ - √ 2Σ 1/2 S . Let M (D, S) = sign(⟨D, S⟩)σ(-|⟨D, S⟩|)D. Then, M (-D, -S) = -M (D, S) for all (D, S) ∈ R 2d
. By integration of an odd function with respect to 0 d with a symmetric distribution around 0 2d , we obtain E (D,S)∼N (0 2d ,I 2d ) [M (D, S)] = 0 d . Therefore, the condition (3) is satisfied and the SP det is a consistent estimator.
Asymptotic variance of SP det . Let H SP det θ ⋆ and R SP det θ ⋆ defined in Lemma 3.2. Since ℓ θ (x, y) = ⟨x -y, θ -(x + y)/2⟩ is linear in θ, we have ∇ 2 θ ⋆ ℓ θ ⋆ = 0 d×d and H SP det θ ⋆ = 0 d×d . The condition P p ⊗2 θ ⋆ (|ℓ θ ⋆ ⟨u, ∇ θ ⋆ ℓ θ ⋆ ⟩| > 0) > 0 for all u ∈ S d-1 is implied by Assumption 4.5, hence we refer to the proof of this result below. Then, the condition R SP det θ ⋆ ⪰ 0 d×d is equivalent to M 3 ⪰ 0 d×d where M 3 = E (D,S)∼N (0 2d ,I 2d ) [sign(⟨D, S⟩)σ (-|⟨D, S⟩|) (DS T + SD T )] . When d = 1, we have M 3 = 2E (D,S)∼N (02,I2) [σ (-|D, S|) |DS|] > 0. When d > 1, for all u ∈ S d-1 , we have u T M 3 u = 2E (D,S)∼N (0 2d ,I 2d ) [sign(⟨D, S⟩)σ (-|⟨D, S⟩|) ⟨u, D⟩⟨u, S⟩] , By rotational symmetry of N (0 2d , I 2d ) and the function to be integrated, showing that min u∈S d-1 u T M 3 u ≥ 0 is equivalent to showing that e T 1 M 3 e 1 ≥ 0, i.e., E (D,S)∼N (0 2d ,I 2d ) [sign(⟨D, S⟩)σ (-|⟨D, S⟩|) D 1 S 1 ] .
By symmetry, we conjecture that ∆ SP N (0 3 validates this conjecture numerically.
d ,I d ) , ∆ SP det N (0 d ,I d ) and R SP det N (0 d ,I d ) are of the form α d I d where α d > 0 is decreasing in d. Figure
Using the sufficient condition derived in Appendix C.2, we have shown that SP det is asymptotically better than SP.
Proof of Assumption 4.4. Since ℓ θ (x, y) = ⟨x -y, θ -(x + y)/2⟩ is linear in θ, we have D(θ ⋆ , θ) = D(θ ⋆ , θ). Proof of Assumption 4.5 For (X, Y ) ∼ p ⊗2 θ ⋆ , let D = Σ -1/2 (X -Y )/ √ 2 and S = √ 2Σ -1/2 ((X + Y )/2 -Σθ ⋆ ). Then, we have (D, S) ∼ N (0 2d , I 2d ). Defining U = D/∥D∥, we have U ∼ U(S 2,d-1 ) is independent of S. Since U = D/∥D∥ ∼ U(S 2,d-1 ) and Σ -1 (x + y)/2 -θ ⋆ = Σ -1/2 S/ √ 2, we obtain P (X,Y )∼p ⊗2 θ ⋆ ((X, Y ) ∈ G 1 (θ ⋆ , u)) = P (U,S)∼U (S 2,d-1 )⊗N (0 d ,I d ) ⟨U, S⟩⟨u, Σ 1/2 U ⟩ > 0 = P U ∼U (S 2,d-1 ) ⟨Σ 1/2 u, U ⟩ > 0 /2 + P U ∼U (S 2,d-1 ) ⟨Σ 1/2 u, U ⟩ < 0 /2 = 1/2 .
where we used that, conditioned on U , ⟨U, S⟩ ∼ N (0, 1) and
P X∼N (0,1) (X < 0) = P X∼N (0,1) (X > 0) = 1/2. The last equality uses that P U ∼U (S 2,d-1 ) ⟨Σ 1/2 u, U ⟩ > 0 = P U ∼U (S 2,d-1 ) ⟨Σ 1/2 u, U ⟩ < 0 = 1/2 by symmetry of the uniform distribution. Therefore, we have shown that P p ⊗2 θ ⋆ (G 1 (θ ⋆ , u)) = 1/2 for all u ∈ S 2,d-1 . Proof of Assumption 4.7. Let us define v = Σ 1/2 u, hence v ∈ U(S 2,d-1 ).
Let Φ denote the c.d.f. of N (0, 1) and erf(x) = 2Φ(x √ 2) -1 be the error function. Let ε > 0. Similarly as above, we obtain that
F θ ⋆ ,u (ε) = P (X,Y )∼p ⊗2 θ ⋆ (0 < V θ ⋆ ,u (X, Y ) ≤ ε) = P (U,S)∼U (S 2,d-1 )⊗N (0 d ,I d ) 0 < ⟨U, S⟩ ⟨v, U ⟩ ≤ √ 2ε = 1 2 E U ∼U (S 2,d-1 ) 2Φ √ 2ε|⟨v, U ⟩| -1 = 1 2 E U ∼U (S 2,d-1 ) [erf (ε|⟨v, U ⟩|)] .
where we use conditioning by U as above. By change of variable, we obtain that
F θ ⋆ ,u (ε) = 1 √ π E U ∼U (S 2,d-1 ) ε|⟨v,U ⟩| 0 e -t 2 dt = ε √ π E U ∼U (S 2,d-1 ) |⟨v, U ⟩| 1 0 e -x 2 ε 2 ⟨v,U ⟩ 2 dx . Using that 1 -x 2 ≤ e -x 2 ≤ 1, we obtain that 0 ≥ √ π ε F θ ⋆ ,u (ε) -E U ∼U (S 2,d-1 ) [|⟨v, U ⟩|] ≥ - ε 2 3 E U ∼U (S 2,d-1 ) |⟨v, U ⟩| 3 . Using that 1 0 x(-2xε 2 ⟨v, U ⟩ 2 )e -x 2 ε 2 ⟨v,U ⟩ 2 dx = e -ε 2 ⟨v,U ⟩ 2 - 1 0 e -x 2 ε 2 ⟨v,U ⟩ 2 dx , we obtain F ′ θ ⋆ ,u (ε) = 1 √ π E U ∼U (S 2,d-1 ) |⟨v, U ⟩|e -ε 2 ⟨v,U ⟩ 2 and F ′′ θ ⋆ ,u (ε) = 2ε √ π E U ∼U (S 2,d-1 ) |⟨v, U ⟩| 3 e -ε 2 ⟨v,U ⟩ 2 .
Therefore, using Lemma F.1, we have
F ′ θ ⋆ ,u (0) = 1 √ π E U ∼U (S 2,d-1 ) [|⟨v, U ⟩|] = 2 d -1 Γ(d/2) πΓ((d -1)/2) = d→+∞ O(1/ √ d) Let us define ε θ ⋆ ,u = E U ∼U (S 2,d-1 ) [|⟨v, U ⟩|] 2E U ∼U (S 2,d-1 ) [|⟨v, U ⟩| 3 ] and M θ ⋆ ,u = 4π E U ∼U (S 2,d-1 ) [|⟨v, U ⟩| 3 ] E U ∼U (S 2,d-1 ) [|⟨v, U ⟩|] 5 .
Then, for all ε ∈ (0, ε θ ⋆ ,u ], we obtain that
F ′′ θ ⋆ ,u (ε) F ′ θ ⋆ ,u (ε) 3 = πε 2 E U ∼U (S 2,d-1 ) |⟨v, U ⟩| 3 e -ε 2 ⟨v,U ⟩ 2 E U ∼U (S 2,d-1 ) |⟨v, U ⟩|e -ε 2 ⟨v,U ⟩ 2 3 ≤ πε 2 E U ∼U (S 2,d-1 ) |⟨v, U ⟩| 3 E U ∼U (S 2,d-1 ) [|⟨v, U ⟩|] -ε 2 E U ∼U (S 2,d-1 ) [|⟨v, U ⟩| 3 ] 3 ≤ M θ ⋆ ,u . Since we have (F -1 θ ⋆ ,u ) ′′ (x) = - F ′′ θ ⋆ ,u (F -1 θ ⋆ ,u (x)) F ′ θ ⋆ ,u (F -1 θ ⋆ ,u (x)) 3 , we obtain sup x∈(0,x θ ⋆ ,u ] |(F -1 θ ⋆ ,u ) ′′ (x)| ≤ M θ ⋆ ,u where x θ ⋆ ,u = F θ ⋆ ,u (ε θ ⋆ ,u ) ≤ E U ∼U (S 2,d-1 ) |⟨Σ 1/2 u, U ⟩| 3 2πE U ∼U (S 2,d-1 ) |⟨Σ 1/2 u, U ⟩| 3 .
Proof of Assumption 4.2. Let ε = ∥θ ⋆ -θ∥ and u = (θ ⋆ -θ)/ε. Then, we have
P p ⊗2 θ ⋆ (D(θ ⋆ , θ)) = P p ⊗2 θ ⋆ (D(θ ⋆ , θ ⋆ + εu)) ≥ P p ⊗2 θ ⋆ (D(θ ⋆ , θ ⋆ + εu) ∩ G 1 (θ ⋆ , u)) = P (X,Y )∼p ⊗2 θ ⋆ (0 < V θ ⋆ ,u (X, Y ) < ε)
Using the above computation, we obtain that
P (X,Y )∼p ⊗2 θ ⋆ (0 < V θ ⋆ ,u (X, Y ) < ε) > 0, hence P p ⊗2 θ ⋆ (D(θ ⋆ , θ)) > 0.
Proof of Assumption 5.2. Using that 1 -e -x ≤ x, we obtain
H 2 (p θ ⋆ , p θ ) = 1 -exp - 1 8 ∥θ ⋆ -θ∥ 2 Σ ≤ 1 8 ∥θ ⋆ -θ∥ 2 Σ .
First, we notice that dim G 0 (θ ⋆ ) ∁ △G 0 (θ) ∁ < 2d, hence we can show that
(x,y)∈G0(θ ⋆ ) ∁ △G0(θ) ∁ p θ ⋆ (x)p θ ⋆ (y)p θ (x)p θ (y)dxdy = 0 .
Second, we see that
∥x -Σθ ⋆ ∥ 2 Σ -1 + ∥y -Σθ ⋆ ∥ 2 Σ -1 + ∥x -Σθ∥ 2 Σ -1 + ∥y -Σθ∥ 2 Σ -1 = ∥x -y∥ 2 Σ -1 + ∥θ -θ ⋆ ∥ 2 Σ + ∥x + y -Σ(θ ⋆ + θ)∥ 2 Σ -1 , D(θ ⋆ , θ) = (x, y) ∈ (R d ) 2 | ⟨x -y, θ ⋆ -Σ -1 (x + y)/2⟩ 2 + ⟨x -y, θ ⋆ -Σ -1 (x + y)/2⟩⟨θ -θ ⋆ , x -y⟩ < 0 ,
Then, we consider the change of variable u = Σ -1/2 (x -y) and v = Σ -1/2 (x + y), whose Jacobian has det(Σ)2 -d as absolute value of its determinant. Therefore, we obtain
e 1 4 ∥θ-θ ⋆ ∥ 2 Σ BC(θ ⋆ , θ) = 1 (4π) d (u,v) 1 0 < - ⟨u, Σ 1/2 θ ⋆ -v/2⟩ ⟨Σ 1/2 (θ -θ ⋆ ), u⟩ < 1 e -1 4 ∥u∥ 2 -1 4 ∥v-Σ 1/2 (θ+θ ⋆ )∥ 2 dudv = 1 (2π) d (ũ,ṽ) 1 ⟨ũ, ṽ⟩ ⟨Σ 1/2 (θ -θ ⋆ ), ũ⟩ < √ 2 e -1 2 ∥ũ∥ 2 -1 2 ∥ṽ∥ 2 dũdṽ = P (X,Y )∼N (0 d ,I d ) ⊗2 ⟨X, Y ⟩ ⟨Σ 1/2 (θ -θ ⋆ ), X⟩ < √ 2 = E U ∼U (S 2,d-1 ) erf |⟨Σ 1/2 (θ -θ ⋆ ), U ⟩| = 2F θ ⋆ ,u (ε)
where the second equality uses the change of variable ũ = u/ √ 2 and ṽ = (v -Σ 1/2 (θ + θ ⋆ ))/ √ 2, whose Jacobian has determinant 2 d . The third and the fourth re-uses computation done previously with ε = ∥θ -θ ⋆ ∥ Σ and u = (θ -θ ⋆ )/ε. Using Lemma F.1 and the above upper bound on F θ ⋆ ,u (ε), we obtain
BC(θ ⋆ , θ) = 2e -ε 2 /4 F θ ⋆ ,u (ε) ≤ 4 d -1 Γ(d/2) πΓ((d -1)/2) ∥θ ⋆ -θ∥ Σ .
Lemma F.1. Let Γ be the Γ function. Then,
∀u ∈ S 2,d-1 , E U ∼U (S 2,d-1 ) [|⟨u, U ⟩|] = 2 d -1 Γ(d/2) √ πΓ((d -1)/2) = d→+∞ O(1/ √ d) .
Proof. Due to rotational symmetry of the distribution, for any unit vector u,
E U ∼U (S 2,d-1 ) [|⟨u, U ⟩|] = E U ∼U (S 2,d-1 ) [|⟨e 1 , U ⟩|] = E U ∼U (S 2,d-1 ) [|U 1 |] .
The density of U 1 is given by
f U1 (x) = Γ d 2 √ π Γ d-1 2 (1 -x 2 ) d-3 2 , x ∈ [-1, 1],
and the expectation can be computed as
E |U 1 | = 1 -1 |x| f U1 (x) dx = 2 1 0 x Γ d 2 √ π Γ d-1 2 (1 -x 2 ) d-3 2 dx = 2Γ d 2 √ π Γ d-1 2 1 0 x (1 -x 2 ) d-3 2 dx = Γ d 2 √ π Γ d-1 2 1 0 (1 -u) d-3 2 du = Γ d 2 √ π Γ d-1 2 • 1 d-1 2 = 2 d -1 Γ d 2 √ π Γ d-1 2 . Therefore, for large d, E |U 1 | = d→+∞ O(1/ √ d).
this section cite: []

Section: G. Laplace with Known Scale
In the following, θ denote the mean parameter of Laplace distribution with known scale b. We have X = R and k = d = 1.
Let θ ∈ Θ and u ∈ {±1}. It is direct to see that
ℓ θ (x, y) = log p θ (x) p θ (y) = |y -θ|/b -|x -θ|/b = 1 b      y -x if θ < min{x, y} x -y if θ > max{x, y} (2θ -(x + y))sign(x -y) if θ ∈ [min{x, y}, max{x, y}] , and ∇ θ ⋆ ℓ θ ⋆ (x, y) = 0 if θ ⋆ < min{x, y} or θ ⋆ > max{x, y} 2 b sign(x -y) if θ ⋆ ∈ [min{x, y}, max{x, y}]
.
Therefore, we have
G 0 (θ ⋆ ) = {(x, y) ∈ R 2 | ||y -θ ⋆ | -|x -θ ⋆ || > 0} , G 1 (θ ⋆ ) = {(x, y) ∈ R 2 | θ ⋆ ∈ [min{x, y}, max{x, y}]} , G 1 (θ ⋆ , u) = {(x, y) ∈ R 2 | θ ⋆ ∈ [min{x, y}, max{x, y}] ∧ u((x + y)/2 -θ ⋆ ) > 0} , D(θ ⋆ , θ) = (x, y) ∈ R 2 | θ ⋆ ∈ [min{x, y}, max{x, y}] ∧ 0 < sign(θ -θ ⋆ )((x + y)/2 -θ ⋆ ) < |θ -θ ⋆ | , ∀(x, y) ∈ G 1 (θ ⋆ , u), V θ ⋆ ,u (x, y) = u((x + y)/2 -θ ⋆ ) .
When θ ⋆ > θ, we have
D(θ ⋆ , θ) = {(x, y) | {θ ⋆ , θ} ⊂ [min{x, y}, max{x, y}] ∧ θ < (x + y)/2 < θ ⋆ } ∪ {(x, y) | θ < min{x, y} ∧ θ ⋆ ∈ ((x + y)/2, max{x, y}]} ∪ {(x, y) | θ < min{x, y} ∧ θ ⋆ > max{x, y}} ∪ {(x, y) | θ ⋆ > max{x, y} ∧ θ ∈ [min{x, y}, (x + y)/2)} .
When θ ⋆ < θ, we have
D(θ ⋆ , θ) = {(x, y) | {θ ⋆ , θ} ⊂ [min{x, y}, max{x, y}] ∧ θ ⋆ < (x + y)/2 < θ} ∪ {(x, y) | θ > max{x, y} ∧ θ ⋆ ∈ [min{x, y}, (x + y)/2)} ∪ {(x, y) | θ ⋆ < min{x, y} ∧ θ > max{x, y}} ∪ {(x, y) | θ ⋆ < min{x, y} ∧ θ ∈ ((x + y)/2, max{x, y}]} . Proof that P p ⊗2 θ ⋆ (G 1 (θ ⋆ )) > 0. It is direct to see that dim(G 0 (θ ⋆ ) ∁ ) < 2. Given that p ⊗2 θ ⋆ is a continuous distribution on (R) 2 , we obtain that P p ⊗2 θ ⋆ (G 0 (θ ⋆ )) = 1.
Using the symmetry of the Laplace distribution around its mean, we have that
P p ⊗2 θ ⋆ (G 1 (θ ⋆ )) = P p ⊗2 θ ⋆ ((-∞, θ ⋆ ) × (θ ⋆ , +∞)) + P p ⊗2 θ ⋆ ((θ ⋆ , +∞) × (-∞, θ ⋆ )) = 1/2 .
Condition in Lemma 3.1. The condition of Lemma 3.1 is implied by Assumption 4.5, hence we refer to the proof of this result below. Therefore, we have I(q θ ⋆ ,hsto ) ≻ I(p ⊗2 θ ⋆ ).
Consistency of SP det . To study SP det for F Lap,b , we use the change of variable D = θ ⋆ -X and S = θ ⋆ -Y . For all (D, S) ∈ G 1 (0), we have
ℓ θ ⋆ (X, Y ) = 1 b (D + S)sign(S -D) , ∇ θ ⋆ ℓ θ ⋆ (X, Y ) = 2 b sign(S -D) , ∇ θ ⋆ log p ⊗2 θ ⋆ (X, Y ) = 0 . For all (D, S) / ∈ G 1 (0), we have ∇ θ ⋆ ℓ θ ⋆ (X, Y ) = 0 and ∇ θ ⋆ log p ⊗2 θ ⋆ (X, Y ) ̸ = 0. Let M (D, S) = 1 ((D, S) ∈ G 1 (0)) σ(-|D + S|/b)sign(D + S)
. Then, M (-D, -S) = -M (D, S) for all (D, S) ∈ R 2 . By integration of an odd function with respect to 0 with a symmetric distribution around 0 2 , we obtain E (D,S)∼N (0 2d ,I 2d ) [M (D, S)] = 0. Therefore, the condition (3) is satisfied and SP det is a consistent estimator.
Asymptotic variance of SP det . Let H SP det θ ⋆ and R SP det θ ⋆ defined in Lemma 3.2. By definition of ℓ θ , we obtain ∇ 2 θ ⋆ ℓ θ ⋆ = 0 and H SP det θ ⋆ = 0. Moreover, using the above formula, we have ∇ θ ⋆ ℓ θ ⋆ (X, Y )∇ θ ⋆ log p ⊗2 θ ⋆ (X, Y ) = 0 for all (D, S) ∈ G 1 (0), hence we obtain R SP det θ ⋆ = 0. The condition P p ⊗2 θ ⋆ (|ℓ θ ⋆ ⟨u, ∇ θ ⋆ ℓ θ ⋆ ⟩| > 0) > 0 for all u ∈ S d-1
is implied by Assumption 4.5, hence we refer to the proof of this result below. Using the sufficient condition derived in Appendix C.2, we have shown that SP det is asymptotically better than SP.
this section cite: []

Section: Proof of Assumption
4.4. Using that D(θ ⋆ , θ) ⊆ G 1 (θ ⋆ ), we simply need to show that D(θ ⋆ , θ) ⊆ G 1 (θ ⋆ ) ∩ D(θ ⋆ , θ).
Let us consider the case θ ⋆ > θ. Then, we have
D(θ ⋆ , θ) = (x, y) ∈ R 2 | θ ⋆ ∈ [min{x, y}, max{x, y}] ∧ θ < (x + y)/2 < θ ⋆ = {(x, y) | {θ ⋆ , θ} ⊂ [min{x, y}, max{x, y}] ∧ θ < (x + y)/2 < θ ⋆ } ∪ {(x, y) | θ < min{x, y} ∧ θ ⋆ ∈ ((x + y)/2, max{x, y}]} = G 1 (θ ⋆ ) ∩ D(θ ⋆ , θ) .
The same result follows when θ ⋆ < θ by using the same argument. In summary, we have shown that D
(θ ⋆ , θ) = G 1 (θ ⋆ ) ∩ D(θ ⋆ , θ) ⊆ D(θ ⋆ , θ).
Proof of Assumption 4.5. Using the symmetry of the Laplace distribution around its mean, we have P
p ⊗2 θ ⋆ (G 1 (θ ⋆ , u)) = P p ⊗2 θ ⋆ (G 1 (θ ⋆ ,1
)) for all u ∈ {±1}. Then, by integrating for x < y, we obtain
P p ⊗2 θ ⋆ (G 1 (θ ⋆ , 1)) = 1 2b 2 x∈(-∞,θ ⋆ ) e x/b y∈(2θ ⋆ -x,+∞) e -y/b dy dx = 1 2b x∈(-∞,θ ⋆ ) e 2x-2θ ⋆ /b dx = 1 4 .
Proof of Assumption 4.7. Let ε > 0. Using the symmetry of the Laplace distribution around its mean, we have F θ ⋆ ,u (ε) = F θ ⋆ ,1 (ε) for all u ∈ {±1}. Similarly as above, by integrating for x < y, we obtain that
F θ ⋆ ,1 (ε) = P (X,Y )∼p ⊗2 θ ⋆ (0 < V θ ⋆ ,1 (X, Y ) ≤ ε) = 1 2b 2 x∈(-∞,θ ⋆ ) e x/b y∈(2θ ⋆ -x,2ε+2θ ⋆ -x) e -y/b dy dx = 1 2b x∈(-∞,θ ⋆ ) e (2x-2θ ⋆ )/b dx - x∈(-∞,θ ⋆ ) e (2x-2θ ⋆ -2ε)/b dx = 1 4 1 -e -2ε/b .
Therefore, we have
F ′ θ ⋆ ,u (x) = 1 2b e -2ε/b , F -1 θ ⋆ ,u (x) = - b 2 log(1 -4x) and (F -1 θ ⋆ ,u ) ′′ (x) = 8b (1 -4x) 2 .
Then, we obtain F ′ θ ⋆ ,u (0) =foot_8 2b and we can take x θ ⋆ ,u = 1/8 and M θ ⋆ ,u = 32b.
Proof of Assumption 4.2. Let ε = |θ ⋆ -θ| and u = sign(θ ⋆ -θ). Using the above computation, we have
P p ⊗2 θ ⋆ (D(θ ⋆ , θ)) ≥ P p ⊗2 θ ⋆ D(θ ⋆ , θ ⋆ + εu) ≥ P p ⊗2 θ ⋆ D(θ ⋆ , θ ⋆ + εu) ∩ G 1 (θ ⋆ , u) = P (X,Y )∼p ⊗2 θ ⋆ (0 < V θ ⋆ ,u (X, Y ) < ε)
Using the above computation, we obtain that P
(X,Y )∼p ⊗2 θ ⋆ (0 < V θ ⋆ ,u (X, Y ) < ε) > 0, hence P p ⊗2 θ ⋆ (D(θ ⋆ , θ)) > 0. Proof of Assumption 5.2. Using that f (x) = x 2 -1 + (1 + x)e -x is positive on R + , we obtain H 2 (p θ ⋆ , p θ ) = 1 -1 + |θ ⋆ -θ| 2b exp - |θ ⋆ -θ| 2b ≤ (θ ⋆ -θ) 2 4b 2 . First, we notice that dim G 0 (θ ⋆ ) ∁ △G 0 (θ) ∁ < 2, hence we can show that (x,y)∈G0(θ ⋆ ) ∁ △G0(θ) ∁ p θ ⋆ (x)p θ ⋆ (y)p θ (x)p θ (y)dxdy = 0 . We consider the case θ ⋆ < θ since θ ⋆ > θ is done similarly as BC(θ ⋆ , θ) = BC(θ, θ ⋆ ). Let ε = θ -θ ⋆ . By integrating for x < y, we have BC(θ ⋆ , θ) = 1 2b 2 x e x/b y 1 (x ≤ θ ⋆ < (x + y)/2 < θ ⋆ + ε ≤ y) e -y/b dy dx + e -(ε+θ ⋆ )/b 2b 2 x e x/b y 1 (y < θ ⋆ + ε ∧ x ≤ θ ⋆ < (x + y)/2) dy dx + e -ε/b 2b 2 x y 1 (θ ⋆ < x < y < θ ⋆ + ε) dy dx + e -θ ⋆ /b 2b 2 y e -y/b x 1 (θ ⋆ < x ∧ (x + y)/2 < θ ⋆ + ε ≤ y) dx dy Direct computation yields x∈(θ ⋆ -ε,θ ⋆ ) e x/b y∈(2θ ⋆ -x,θ ⋆ +ε) 1dy dx = x∈(θ ⋆ -ε,θ ⋆ ) e x/b (x + ε -θ ⋆ ) dx = e (θ ⋆ -ε)/b u∈(0,ε) ue u/b du , u∈(0,ε) ue u/b du = b e ε/b (ε -b) + b , x y
Moreover, we have
x e x/b y 1 (x ≤ θ ⋆ < (x + y)/2 < θ ⋆ + ε ≤ y) e -y/b dy dx = x∈(-∞,θ ⋆ -ε) e x/b y∈(2θ ⋆ -x,2θ ⋆ +2ε-x) e -y/b dy dx + x∈(θ ⋆ -ε,θ ⋆ ) e x/b y∈(θ ⋆ +ε,2θ ⋆ +2ε-x) e -y/b dy dx = b x∈(-∞,θ ⋆ -ε) e -(2θ ⋆ -2x)/b -e -(2θ ⋆ +2ε-2x)/b dx + b x∈(θ ⋆ -ε,θ ⋆ ) e -(θ ⋆ +ε-x)/b -e -(2θ ⋆ +2ε-2x)/b dx = b b 2 e -2ε/b - b 2 e -4ε/b + b e -ε/b -e -2ε/b + b 2 e -4ε/b -e -2ε/b = b 2 e -ε/b -e -2ε/b
Therefore, we have
BC(θ ⋆ , θ ⋆ + ε) = 1 2 e -ε/b -e -2ε/b + 1 2b e -2ε/b + e -2(θ ⋆ +ε)/b e ε/b (ε -b) + b + e -ε/b ε 2 4b 2 = 1 2 e -ε/b -e -2ε/b + 1 2 e -2ε/b + e -2(θ ⋆ +ε)/b ε b e ε/b -e ε/b + 1 + e -ε/b ε 2 4b 2 = 1 2 e -2(θ ⋆ +ε)/b -e -(2θ ⋆ +ε)/b + 1 2 e -ε/b + e -(2θ ⋆ +ε)/b ε b + e -ε/b ε 2 4b 2 = 1 2 e -ε/b e -2θ ⋆ /b (e -ε/b -1 + ε/b) + ε b + ε 2 2b 2 .
Then, we can conclude that
BC(θ ⋆ , θ ⋆ -ε) = BC(θ ⋆ -ε, θ ⋆ ) = 1 2 e -ε/b e -2(θ ⋆ -ε)/b (e -ε/b -1 + ε/b) + ε b + ε 2 2b 2 . Using that f (x) = 1 -x + x 2 /2 -e -x is positive on R + , we obtain BC(θ ⋆ , θ) ≤ |θ ⋆ -θ| 2b 1 + |θ ⋆ -θ| 2b 1 + e -2 min{θ ⋆ ,θ}/b .
this section cite: []

Section: H. Supplementary Experiments
Using the same empirical setup as in Section 6, we conduct additional experiments to support our theoretical claims for other distributions (Appendix H.1), other estimators for Gaussian distributions based on C n (Appendix H.2), other convex surrogates of the 0-1 loss (Appendix H.3) or normalized/regularized versions of the logistic loss (Appendix H.4).
this section cite: []

Section: Reproducibility.
Code for reproducing our empirical results is available at https://github.com/tml-epfl/  learning-parametric-distributions-from-samples-and-preferences. Our code is implemented in Julia (Bezanson et al., 2017), version 1.11.5. The plots are generated with StatsPlots. The optimization problems defining some of our estimators are solved numerically with JuMP (Lubin et al., 2023), by using the Ipopt (Wächter & Biegler, 2006) and HiGHS (Huangfu & Hall, 2018) solvers. Other dependencies are listed in the Readme.md that provides detailed julia instructions to reproduce our experiments, as well as a script.sh to run them all at once. Our experiments are conducted on 12 Intel(R) Core(TM) Ultra 7 165U 4.9GHz CPU.
Gaussian distribution with known variance. For F N ,1 , the SP det and SP estimators are computed with the Ipopt solver. For F N ,I d , the SP det , SP, DP and WE estimators are computed with the Ipopt solver, and the AE estimator uses the HiGHS solver.
this section cite: ['b3', 'b25', 'b49', 'b18']

Section: H.1. Accelerated Rates for Other Distributions

this section cite: []

Section: H.1.1. LAPLACE DISTRIBUTION WITH KNOWN SCALE
Estimators. For F Lap,1 (Appendix G), we have The estimators based on C n are θ AE n ∈ C n , θ WE n := arg max θ∈Cn |θ -θ ⋆ | and θ DP n = arg max θ∈Cn i∈[n]
θ SO n = median({X i } i∈[n] ∪ {Y i } i∈[n] ) and C n = {θ | ∀i ∈ [n], Z i (|Y i -θ| -|X i -θ|) ≥ 0} .
(|Y i -θ| + |X i -θ|) .
Those three estimators are computed with the Ipopt solver.
Experiments. Figure 4(a) confirms empirically the difference in estimation rate between the M-estimators (SO MLE)obtaining O(1/ √ n)-and our estimators based on C n -achieving O(1/n). Moreover, AE and WE perform on par with DP MLE. H.1.2. RAYLEIGH DISTRIBUTION Let σ > 0 be the scale parameter characterizing a Rayleigh distribution. In the following, let θ = -1 2σ 2 < 0 denote the natural parameter of a Rayleigh distribution. We have Θ ⊆ R ⋆ -, X = R + and k = d = 1. The probability density function is defined as ∀x ∈ R + , p θ (x) = exp x 2 θ + log(x) + log(2θ) .
Let θ ∈ Θ and u ∈ {±1}. It is direct to see that, for all (x, y) ∈ R 2 + , ℓ θ (x, y) = log p θ (x) p θ (y) = (x 2 -y 2 )θ + log(x/y) and dℓ θ ⋆ dθ ⋆ (x, y) = x 2 -y 2 = (x -y)(x + y) .
Therefore, we have
G 0 (θ ⋆ ) = {(x, y) ∈ R 2 + | |(x 2 -y 2 )θ ⋆ + log(x/y)| > 0} , G 1 (θ ⋆ ) = {(x, y) ∈ R 2 + | |x -y| > 0} , G 1 (θ ⋆ , u) = {(x, y) ∈ R 2 + | u((x 2 -y 2 ) 2 θ ⋆ + (x 2 -y 2 ) log(x/y)) < 0} , D(θ ⋆ , θ) = {(x, y) ∈ R 2 + | ((x 2 -y 2 )θ ⋆ + log(x/y)) 2 + (x 2 -y 2 )(θ -θ ⋆ ) (x 2 -y 2 )θ ⋆ + log(x/y) } , ∀(x, y) ∈ G 1 (θ ⋆ , u), V θ ⋆ ,u (x, y) = -u θ ⋆ + 1 x + y log(x) -log(y)
x -y .
Proof that Proof of Assumption 4.4 and 4.5. Since ℓ θ (x, y) = (x 2 -y 2 )θ + log(x/y) is linear in θ, we have D(θ ⋆ , θ) = D(θ ⋆ , θ). Let (X, Y ) ∼ p ⊗2 θ ⋆ . Then, we have
P p ⊗2 θ ⋆ (G 1 (θ ⋆ )) > 0. It is direct to see that dim(G 0 (θ ⋆ ) ∁ ) < 2 and dim(G 0 (θ ⋆ ) \ G 1 (θ ⋆ )) < 2. Given that p ⊗2 θ ⋆ is a continuous distribution on (R + ) 2 , we obtain that P p ⊗2 θ ⋆ (G 0 (θ ⋆ )) = P p ⊗2 θ ⋆ (G 1 (θ ⋆ )) = 1.
P p ⊗2 θ ⋆ (G 1 (θ ⋆ , 1)) = P (X,Y )∼p ⊗2 θ ⋆ θ ⋆ < 1 X 2 log(Y /X) 1 -(Y /X) 2 > 0 , P p ⊗2 θ ⋆ (G 0 (θ ⋆ , -1)) = P (X,Y )∼p ⊗2 θ ⋆ θ ⋆ > 1 X 2 log(Y /X) 1 -(Y /X) 2 > 0 .
Estimators. We have
θ SO n = 1 4n i∈[n] (X 2 i + Y 2 i ) and C n = {θ | ∀i ∈ [n], Z i ((X 2 i -Y 2 i )θ + log(X i /Y i )) ≥ 0} .
The estimators based on C n are θ AE n ∈ C n , θ WE n := arg max θ∈Cn |θ -θ ⋆ |. Those two estimators are computed with the Ipopt solver.
Experiments. Figure 4(b) confirms empirically the difference in estimation rate between the M-estimators (SO MLE)obtaining O(1/ √ n)-and our estimators based on C n -achieving O(1/n). Moreover, AE and WE perform similarly.
this section cite: []

Section: H.2. Other Estimators for Gaussian Distributions
To better understand the surprising performance of the RU estimator, we consider other estimators that disentangle the effect of RU's randomness versus its mean behavior.
Univariate Gaussian. The center estimator (CE) returns the center of the interval C n . The truncated Gaussian estimator (TrG) returns a realization from a Gaussian distribution with mean CE and variance 4/n, which is truncated to C n . The truncated MLE (TrMLE) returns the average of the observations ({X i } i∈[n] ∪ {Y i } i∈[n] ) ∩ C n .
Figure 5(a) reveals that TrG performs on par with RU, yet CE and TrMLE outperform both TrG and RU. This suggests that being far away from the boundary of C n improves performance compared to DP that lies on the boundary of C n (as observed empirically). Moreover, randomization on C n worsens performance compared to CE.
Using the derivation in the introduction on univariate Gaussian, it is coherent that CE improves on DP by a multiplicative constant: the average of those two (non-independent) random variables decreases faster. Formally, this could be proven by refining the proof of Lemma 4.6 to account for the property that n = N θ ⋆ ,-1 + N θ ⋆ ,1 . Multivariate Gaussian. For d > 1, multiple centers exist. We use the Chebyshev center estimator (CCE) of C n .
Figures 5(b) and 6 shows that CCE outperforms AE by a constant margin. It only outperforms DP in the regime of large n compared to d and performs worse than SO MLE for small n. Geometrically, for small n and large d, we conjecture that the random polytope C n is more likely to be "spiky" along some directions. Due to those distant vertices, the center would become a worse estimator than DP, since the "average" is intuitively less robust to outliers. In contrast, DP MLE dominates SO MLE statistically (Lemma 4.1), hence it achieves rate O( d/n) when n is small compared to d.
this section cite: []

Section: H.3. Estimators Based on Convex Surrogate of the 0-1 Loss
While DP MLE minimizes an objective that minimizes the 0-1 loss, SP MLE minimizes an objective involving the logistic loss f Log (x) = log(1 + exp(-x)). As in Tang et al. (2024b), we can generalize this approach to f any convex surrogate of the 0-1 loss, see Figure 7(a). For example, we consider the Hinge loss (Hin), i.e., f Hin (x) := max{0, 1 -x}, the square loss (Squ), i.e., f Squ (x) := (1 -x) 2 , the truncated square loss (TrS), i.e., f TrS (x) := max{0, 1 -x} 2 , the Savage loss (Sav), i.e., f Sav (x) := (1 + exp(x)) -2 , and the exponential loss (Exp), i.e., f Exp (x) := exp(-x).
Given (X i , Y i , Z i ) i∈[n] ∼ q ⊗[n]
θ ⋆ ,h det and a loss f , we consider the estimator
θ f n ∈ arg min θ∈Θ    L SO n (θ) + i∈[n] f (Z i ℓ θ (X i , Y i ))    .
All those estimators are computed with the Ipopt solver.
Figure 7(b) shows that all estimators perform on par with SP MLE, i.e., the one based on the logistic loss.
this section cite: []

Section: H.4. Impact of Normalization and Regularization
The estimator defined in Appendix H.3 can be further generalized by introducing a regularization parameter λ ≥ 0 and a normalization parameter β > 0, see, e.g., Gorbatovski et al. (2025).
Given (X i , Y i , Z i ) i∈[n] ∼ q ⊗[n]
θ ⋆ ,h det , a loss f and regularization/normalization (λ, β), we consider the estimator
θ f,λ,β n ∈ arg min θ∈Θ    L SO n (θ) + λ i∈[n] f (βZ i ℓ θ (X i , Y i ))    .
While similar modifications could be made for other losses, we focus on the logistic loss f Log (x) = log(1 + exp(-x)). In particular, we recover SP MLE by taking λ = β = 1.
Figures 8(a) and (b) showcase the "mild" impact of normalization and regularization.
this section cite: ['b14']

Section: 
1.0 0.5 0.0 0.5 1.0 1.5 2.0 2.5 3.0 0.00
this section cite: []

Section: References
Ref_id:b0 Title: Deux remarques sur l'estimation Year: (1983)
Ref_id:b1 Title: A general theoretical paradigm to understand learning from human preferences Year: (2024)
Ref_id:b2 Title: McKinnon Year: (2022)
Ref_id:b3 Title: A fresh approach to numerical computing Year: (2017)
Ref_id:b4 Title: Rates of convergence for minimum contrast estimators. Probability Theory and Related Fields Year: (1993)
Ref_id:b5 Title: Rank analysis of incomplete block designs: I. the method of paired comparisons Year: (1952)
Ref_id:b6 Title:  Year: (1946)
Ref_id:b7 Title: The llama 3 herd of models Year: (2024)
Ref_id:b8 Title: Asymptotic minimax character of the sample distribution function and of the classical multinomial estimator Year: (1956)
Ref_id:b9 Title: Class notes for transmission of information Year: (1952)
Ref_id:b10 Title: Improved optimistic algorithms for logistic bandits Year: (2020)
Ref_id:b11 Title: Agnostic learning of monomials by halfspaces is hard Year: (2012)
Ref_id:b12 Title: Learning linear utility functions from pairwise comparison queries Year: (2024)
Ref_id:b13 Title: A comparison of signalling alphabets. The Bell System Technical Year: (1952)
Ref_id:b14 Title: The differences between direct alignment algorithms are a blur Year: (2025)
Ref_id:b15 Title: Minimax-optimal inference from partial rankings Year: (2014)
Ref_id:b16 Title: Monolithic preference optimization without reference model Year: (2024)
Ref_id:b17 Title: Self-improvement in language models: The sharpening mechanism Year: (2024)
Ref_id:b18 Title: Parallelizing the dual revised simplex method Year: (2018)
Ref_id:b19 Title: Mm algorithms for generalized bradley-terry models. The annals of statistics Year: (2004)
Ref_id:b20 Title: Statistical estimation: asymptotic theory Year: (2013)
Ref_id:b21 Title: Unpacking dpo and ppo: Disentangling best practices for learning from preference feedback Year: (2024)
Ref_id:b22 Title: Reinforcement Learning from Human Feedback Year: (2024)
Ref_id:b23 Title: Convergence of estimates under dimensionality restrictions Year: (1973)
Ref_id:b24 Title: Theory of point estimation Year: (2006)
Ref_id:b25 Title: Jump 1.0: Recent improvements to a modeling language for mathematical optimization Year: (2023)
Ref_id:b26 Title: Minimax rates and efficient algorithms for noisy sorting Year: (2018)
Ref_id:b27 Title: The lean mathematical library Year: (2020-01)
Ref_id:b28 Title: Simple preference optimization with a reference-free reward Year: (2024)
Ref_id:b29 Title: Nash learning from human feedback Year: (2024)
Ref_id:b30 Title: Iterative ranking from pair-wise comparisons Year: (2012)
Ref_id:b31 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b32 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b33 Title: A statistical convergence perspective of algorithms for rank aggregation from pairwise data Year: (2014)
Ref_id:b34 Title: Information and the accuracy attainable in the estimation of statistical parameters Year: (1992)
Ref_id:b35 Title: Dueling rl: Reinforcement learning with trajectory preferences Year: (2023)
Ref_id:b36 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b37 Title: Simple, robust and optimal ranking from pairwise comparisons Year: (2018)
Ref_id:b38 Title: Estimation from pairwise comparisons: Sharp minimax bounds with topology dependence Year: (2016)
Ref_id:b39 Title: Parametric estimation. Finite sample theory Year: (2012)
Ref_id:b40 Title: A minimaximalist approach to reinforcement learning from human feedback Year: (2024)
Ref_id:b41 Title: All roads lead to likelihood: The value of reinforcement learning in fine-tuning Year: (2025)
Ref_id:b42 Title: Generalized preference optimization: A unified approach to offline alignment Year: (2024-07)
Ref_id:b43 Title: Generalized preference optimization: A unified approach to offline alignment Year: (2024)
Ref_id:b44 Title: SageMath, the Sage Mathematics Software System (Version 9.7) Year: ()
Ref_id:b45 Title: Llama 2: Open foundation and finetuned chat models Year: (2023)
Ref_id:b46 Title: Nonparametric estimators. Introduction to Nonparametric Estimation Year: (2009)
Ref_id:b47 Title: Asymptotic statistics Year: (2000)
Ref_id:b48 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b49 Title: On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming Year: (2006)
Ref_id:b50 Title: High-Dimensional Statistics: A Non-Asymptotic Viewpoint Year: (2019)
Ref_id:b51 Title: Aligned supervised fine-tuning through absolute likelihood Year: (2024)
Ref_id:b52 Title: All of statistics: a concise course in statistical inference Year: (2013)
Ref_id:b53 Title: Leveraging sparsity for sample-efficient preference learning: A theoretical perspective Year: (2025)
Ref_id:b54 Title: Principled reinforcement learning with human feedback from pairwise or k-wise comparisons Year: (2023)
Ref_id:b55 Title: Fine-tuning language models from human preferences Year: (2019)
