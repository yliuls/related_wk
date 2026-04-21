Title: Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning
Abstract: Off-policy learning and evaluation leverage logged bandit feedback datasets, which contain context, action, propensity score, and feedback for each data point. These scenarios face significant challenges due to high variance and poor performance with low-quality propensity scores and heavy-tailed reward distributions. We address these issues by introducing a novel estimator based on the log-sum-exponential (LSE) operator, which outperforms traditional inverse propensity score estimators. Our LSE estimator demonstrates variance reduction and robustness under heavy-tailed conditions. For off-policy evaluation, we derive upper bounds on the estimator's bias and variance. In the off-policy learning scenario, we establish bounds on the regret-the performance gap between our LSE estimator and the optimal policy-assuming bounded (1 + ϵ)-th moment of weighted reward. Notably, we achieve a convergence rate of O(n -ϵ/(1+ϵ) ) for the regret bounds, where ϵ ∈ [0, 1] and n is the size of logged bandit feedback dataset. Theoretical analysis is complemented by comprehensive empirical evaluations in both off-policy learning and evaluation scenarios, confirming the practical advantages of our approach. The code for our estimator is available at the following link: https:  //github.com/armin-behnamnia/  lse-offpolicy-learning .

Section: Introduction
Off-policy learning and evaluation from logged data are important problems in reinforcement learning (RL). The logged bandit feedback (LBF) dataset represents interaction logs of a system with its environment, recording context, action, propensity score (i.e., the probability of action selection for a given context under the logging policy), and feedback (reward). It is used in many real applications, e.g., recommendation systems (Aggarwal, 2016;Li et al., 2011), personalized medical treatments (Kosorok & Laber, 2019;Bertsimas et al., 2017), and personalized advertising campaigns (Tang et al., 2013;Bottou et al., 2013). The literature has considered this setting from two perspectives, off-policy evaluation (OPE) and off-policy learning (OPL). In off-policy evaluation, we utilize the LBF dataset from a logging (behavioural) policy and an estimator constructed via e.g., inverse propensity score (IPS) weighting, to evaluate (or estimate) the performance of a different target policy. In off-policy learning, we leverage the estimator and LBF dataset to learn an improved policy with respect to logging policy.
In both scenarios, OPL and OPE, the IPS estimator is proposed (Thomas et al., 2015;Swaminathan & Joachims, 2015a). However, this estimator suffers from significant variance in many cases (Rosenbaum & Rubin, 1983). To address this, some improved IPS estimators have been proposed, such as the IPS estimator with the truncated ratio of policy and logging policy (Ionides, 2008b), IPS estimator with truncated propensity score (Strehl et al., 2010), selfnormalizing estimator (Swaminathan & Joachims, 2015b), exponential smoothing (ES) estimator (Aouali et al., 2023), implicit exploration (IX) estimator (Gabbianelli et al., 2023) and power-mean (PM) estimator (Metelli et al., 2021).
In addition to the significant variance issue of IPS estimators, there are two more challenges in real problems: estimated propensity scores and heavy-tailed behaviour of weighted reward due to noise or outliers. Previous works such as Swaminathan & Joachims (2015a), Metelli et al. (2021), and Aouali et al. (2023) have made assumptions when dealing with LBF datasets. Specifically, these works assume that rewards are not subject to perturbation (noise) and that true propensity scores are available. However, these assumptions may not hold in real-world scenarios.
Noisy or heavy-tailed reward: three primary sources of noise in reward of LBF datasets can be identified as (Wang et al., 2020): (1) inherent noise, arising from physical conditions during feedback collection; (2) application noise, stemming from uncertainty in human feedback; and (3) adversarial noise, resulting from adversarial perturbations in the feedback process. Furthermore, In addition to noisy (perturbed) rewards, a heavy-tailed reward can be observed in many real-life applications, e.g., financial markets (Cont & Bouchaud, 2000) and web advertising (Park et al., 2013), the rewards do not behave bounded and follows heavy-tailed distributions where the variance is not well defined.
Noisy (estimated) propensity scores: The access to the exact values of the propensity scores may not be possible, for example, when human agents annotate the LBF dataset. In this situation, one may settle for training a model to estimate the propensity scores. Then, the propensity score stored in the LBF dataset can be considered a noisy version of the true propensity score.
Therefore, there is a need for an estimator that can effectively manage the heavy-tailed condition and noisy rewards or propensity scores in the LBF dataset.
this section cite: ['b1', 'b58', 'b51', 'b10', 'b13', 'b90', 'b29', 'b72', 'b72', 'b22', 'b77']

Section: Contributions
In this work, we propose a novel estimator for off-policy learning and evaluation from the LBF dataset that outperforms existing estimators when dealing with estimated propensity scores and heavy-tailed or noisy weighted rewards. The contribution of our work is three-fold.
First, we propose a novel non-linear estimator based on the log-sum-exponential (LSE) operator which can be applied to both OPE and OPL scenarios. This LSE estimator effectively reduces variance and is applicable to noisy propensity scores, heavy-tailed reward and noisy reward scenarios.
Second, we provide comprehensive theoretical guarantees for the LSE estimator's performance in OPE and OPL setup. In particular, we first provide bounds on the regret, i.e. the difference between the LSE estimator performance and the true average reward, under mild assumptions. Then, we studied bias and variance of the LSE estimator and its robustness under noisy and heavy-tailed reward scenarios.
Third, we conducted a set of experiments on different datasets to show the performance of the LSE in scenarios with true, estimated propensity scores and noisy reward in comparison with other estimators. We observed an improvement in the performance of the learning policy using LSE in comparison with other state-of-the-art algorithms under different scenarios.
this section cite: []

Section: Log-Sum-Exponential Estimator
Notation: We adopt the following convention for random variables and their distributions in the sequel. A random variable is denoted by an upper-case letter (e.g., Z), an arbitrary value of this variable is denoted with the lower-case letter (e.g., z), and its space of all possible values with the corresponding calligraphic letter (e.g., Z). This way, we can describe generic events like {Z = z} for any z ∈ Z, or events like {g(Z) ≤ 5} for functions g : Z → R. P Z denotes the probability distribution of the random variable Z. The joint distribution of a pair of random variables (Z 1 , Z 2 ) is denoted by P Z1,Z2 . The cardinality of set Z is denoted by |Z|. We denote the set of integer numbers from 1 to n by [n] ≜ {1, • • • , n}. In this work, we consider the natural logarithm, i.e., log(x) := log e (x). For two probability measures P and Q defined on the space Z and a probability measure P define on the space Y, the total variation distance between two densities P and Q, is defined as TV(P, Q) := Z |P -Q|(dz). We also define the conditional total variation distance as TV c (P Z|Y , Q Z|Y ) :=
Y PY =y Z |P Z=z|Y =y -Q Z=z|Y =y |(dz)dy.
Main Idea: Inspired by the log-sum-exponential operator with applications in multinomial linear regression, naive Bayes classifiers and tilted empirical risk (Calafiore et al., 2019;Murphy, 2012;Williams & Barber, 1998;Li et al., 2023), we define the LSE estimator with parameter λ < 0,
LSE λ (Z) = 1 λ log 1 n n i=1 e λzi ,(1)
where Z = {z i } n i=1 are samples from the positive random variable Z. The key property of the LSE operator is its robustness to noisy samples in a limited number of data samples. Here a noisy sample, by intuition, is a point with abnormally large positive z i . Such points vanish in the exponential sum as lim zi→+∞ e λzi = 0 for λ < 0. Therefore the LSE operator ignores terms with large values for negative λ. The robustness of LSE has also been explored in the context of supervised learning by Li et al. (2023) from a practical perspective. Furthermore, in Appendix (App) C, we discuss the connection between the LSE and entropy regularization.
Motivating example: We provide a toy example to investigate the behaviour of LSE as a general estimator and its difference from the Monte-Carlo estimator (a.k.a. simple average) for mean estimation. Suppose that Z is distributed as a Pareto distribution with scale x m and shape ζ. Note that for Z ∼ Pareto(x m , ζ) as a heavy-tailed distribution, we have f Z (z) = ζx ζ m z ζ+1 . Let ζ = 1.5 and x m =foot_0 3 , then we have E[Z] = ζxm ζ-1 = 1. The objective is to estimate E[Z] with n independent samples drawn from the Pareto distribution. We set n ∈ {10, 50, 100, 1000, 10000} and compute the Monte-Carlo (a.k.a. simple average) and LSE estimation of the expectation of Z. Table 1 shows that LSE (with λ = -0.1) effectively keeps the variance and meansquare error (MSE), low without significant side-effects on bias. We also observe that the LSE estimator works well under heavy tail distributions.
Table 1: Bias, variance, and MSE of LSE (with λ = -0.1) and Monte-Carlo estimators. We run the experiment 10000 times and report the variance, bias, and MSE of the estimations.
Estimator n = 10 n = 50 n = 100 n = 1000 n = 10000 Bias Monte-Carlo 0.0154 0.0155 0.0083 0.0061 0.0044 LSE 0.1576 0.1606 0.1616 0.1624 0.1629 Variance Monte-Carlo 1.5406 1.5289 1.3229 1.0203 0.8384 LSE 0.1038 0.0616 0.0443 0.0335 0.0268 MSE Monte-Carlo 1.5409 1.5292 1.3229 1.0203 0.8384 LSE 0.1287 0.0874 0.0704 0.0598 0.0534
this section cite: ['b17', 'b73', 'b59', 'b59']

Section: Related Works
We categorize the estimators based on their approach to reward estimation. Estimators that incorporate reward estimation techniques are classified as model-based estimators.
In contrast, those that work without reward estimation are termed model-free estimators. Below, we review modelbased and model-free estimators. Furthermore, we study the estimators which are designed for unbounded reward (heavy-tailed) scenarios in RL.
Model-free Estimators: In model-free estimators, e.g., IPS estimators, we have many challenges, including, high variance and heavy-tailed scenarios. Recently, many modelfree estimators have been proposed for high variance problems in model-free estimators (Strehl et al., 2010;Ionides, 2008b;Swaminathan & Joachims, 2015b;Aouali et al., 2023;Metelli et al., 2021;Neu, 2015;Aouali et al., 2023;Metelli et al., 2021;Sakhi et al., 2024). However, under heavy-tailed or unbounded reward scenario, the performance of these estimators degrade. In this work, our proposed LSE estimator demonstrates robust performance even under heavy-tailed assumptions, backed by theoretical guarantees.
Model-based Estimators: The direct method for off-policy learning from the LBF datasets is based on the estimation of the reward function, followed by the application of a supervised learning algorithm to the problem. However, this approach does not generalize well, as shown by Beygelzimer & Langford (2009). A different approach where the direct method and the IPS estimator are combined, i.e., doublyrobust, is introduced by Dudík et al. (2014). A different approach based on policy optimization and boosted base learner is proposed to improve the performance in direct methods (London et al., 2023).Furthermore, the optimistic shrinkage (Su et al., 2020) and Dr-Switch (Wang et al., 2017) as other model-based estimators. Our approach differs from this area, as we do not estimate the reward function in the LSE estimator. A combination of the LSE estimator with the direct method is presented in App. G.3 . In this work, we focus on model-free approach.
Unbounded Reward: Unbounded rewards (or returns) have been observed in various domains, including finance (Lu & Rong, 2018) and robotics (Bohez et al., 2019). In the context of multi-arm bandit problems, unbounded rewards can emerge as a result of adversarial attacks on reward distributions (Guan et al., 2020). Within the broader field of RL, researchers have investigated poisoning attacks on rewards and the manipulation of observed rewards (Rakhsha et al., 2020;2021;Rangi et al., 2022). These studies highlight the importance of considering unbounded reward scenarios in RL and bandits algorithms. In particular, in our work, we focus on off-policy learning and evaluation under heavy-tailed (unbounded reward) assumption, employing a bounded (1 + ϵ)-th moment of weighted-reward assumption for ϵ ∈ [0, 1].
this section cite: ['b90', 'b72', 'b74', 'b72', 'b83', 'b11', 'b27', 'b61', 'b91', 'b62', 'b12', 'b32']

Section: Problem Formulation
Let X be the set of contexts and A the set of actions. We consider policies as conditional distributions over actions, given contexts. For each pair of context and action (x, a) ∈ X × A and policy π θ ∈ Π θ , where Π Θ is defined as the set of all policies (policy set) which are parameterized by θ ∈ Θ, where Θ is the set of parameters, e.g., the parameters of a neural network. Furthermore, the π θ (a|x) is defined as the conditional probability of choosing an action given context x under the policy π θ . 1   A reward functionfoot_1 r : X × A → R + , which is unknown, defines the expected reward (feedback) of each observed pair of context and action. In particular, r(x, a) = E P R|X=x,A=a [R] where R ∈ R + is random reward and P R|X=x,A=a is the conditional distribution of reward R given the pair of context and action, (x, a). Note that, in the LBF setting, we only observe the reward (feedback) for the chosen action a in a given context x, under the known logging policy π 0 (a|x). We have access to the LBF dataset S = (x i , a i , p i , r i ) n i=1 with n i.i.d. data points where each 'data point' (x i , a i , p i , r i ) contains the context x i which is sampled from unknown distribution P X , the action a i which is sampled from the known logging policy π 0 (•|x i ), the propensity score p i ≜ π 0 (a i |x i ), and the observed feedback (reward) r i as a sample from distribution P R|X=xi,A=ai under logging policy π 0 (a i |x i ).
We define the expected reward of a learning policy, π θ ∈ Π θ , which is called the value function evaluated at the learning policy, as
V (π θ ) = E P X [E π θ (A|X) [E P R|X,A [R]]] = E P X [E π θ (A|X) [r(A, X)|X]].(2)
We denote the importance weighted reward as w θ (A, X)R, where w θ (A, X) is the weight,
w θ (A, X) = π θ (A|X) π 0 (A|X) .
As discussed by Swaminathan & Joachims (2015b), the IPS estimator is applied over the LBF dataset S (Rosenbaum & Rubin, 1983) to get an unbiased estimator of the value function by considering the weighted reward as,
V (π θ , S) = 1 n n i=1 r i w θ (a i , x i ),(3)
where w θ (a i , x i ) = π θ (ai|xi) π0(ai|xi) . The IPS estimator as an unbiased estimator has bounded variance if the π θ (A|X) is absolutely continuous with respect to π 0 (A|X) (Strehl et al., 2010;Langford et al., 2008). Otherwise, it suffers from a large variance.
LSE in OPE and OPL scenarios: The LSE estimator is defined as ai,xi) , where λ < 0 is a tunable parameter which helps us to recover the IPS estimator for λ → 0. Furthermore, the LSE estimator is an increasing function with respect to λ.
V λ LSE (S, π θ ) := LSE λ (S) = 1 λ log 1 n n i=1 e λriw θ (
OPE scenario: One of the evaluation metrics for an estimator in OPE scenarios is the MSE which is decomposed into squared bias and the variance of the estimator. In particular, for the LSE estimator, we consider the following MSE decomposition in terms of bias and variance,
MSE( V λ LSE (S, π θ )) = B( V λ LSE (S, π θ )) 2 + V( V λ LSE (S, π θ )), B( V λ LSE (S, π θ )) = E[w θ (A, X)R] -E[ V λ LSE (S, π θ )], V( V λ LSE (S, π θ )) = E[( V λ LSE (S, π θ ) -E[ V λ LSE (S, π θ )]) 2 ], where B( V λ LSE (S, π θ )) and V( V λ LSE (S, π θ ))
are bias and variance of the LSE estimator, respectively.
OPL scenario: Our objective in OPL scenario is to find an optimal π θ ⋆ , one which maximize V (π θ ), i.e.,
π θ ⋆ = arg max π θ ∈ΠΘ V (π θ ).(4)
We define the estimation errorfoot_2 , as the difference between the value function and the LSE estimator for a given learning policy π θ ∈ Π θ , i.e.,
Est λ (π θ ) := V (π θ ) -V λ LSE (S, π θ ).(5)
For the OPL scenario, we also define π θ policy as the maximizer of the LSE estimator for a given dataset S,
π θ (S) = arg max π θ ∈ΠΘ V λ LSE (S, π θ ).(6)
Finally, we define regret, as the difference between the value function evaluated at π θ * and π θ ,
R λ (π θ , S) := V (π θ * ) -V (π θ (S)).(7)
More discussion regarding the LSE properties is provided in App. C.
this section cite: ['b90', 'b54']

Section: Theoretical Foundations of the LSE Estimator
In this section, we study the regret, bias-variance and robustness of the LSE estimator. We compare our LSE estimator with other model-free estimators in Table 2. All the proof details are deferred to App.D.
this section cite: []

Section: Non-linearity of LSE:
The LSE estimator is a non-linear model-free estimator with respect to the weighted reward or reward, which is different from linear model-free estimators.
In particular, most estimators can be represented as the weighted average of reward (feedback),
V (π θ , S) = 1 n n i=1 g r i , w θ (a i , x i ) ,(8)
where
g : R + × R + → R is a transformation of r i w θ (a i , x i )
and is defined for each model-free estimator.
For example, we have g(r, y) = ry in the IPS estimator, g(r, y) = r min(y, M ) in the truncated IPS estimator (Ionides, 2008b), g(r, y) = r((1 -λ)y s + λ) 1/s in the PM estimator (Metelli et al., 2021), g(r, y) = ry α for α ∈ (0, 1) in the ES estimator (Aouali et al., 2023) and g(r, y) = r τ y y 2 +τ in the optimistic shrinkage (OS) (Su et al., 2020). For the IX-estimator with parameter η (Gabbianelli et al., 2023), we have g(r, y) = r y 1+η/π0 . Furthermore, recently a logarithmic smoothing (LS) estimator with parameter λ is proposed by Sakhi et al. (2024) where g(r, y) = 1 λ log(1 + λry). However, the LSE estimator is a non-linear function with respect to a whole set of weighted reward samples. Therefore, the previous techniques for regret and bias-variance analysis under linear estimators are not applicable.
Theoretical comparison with other estimators: The comparison of our LSE estimator with other estimators, including, IPS, self-normalized IPS (Swaminathan & Joachims, 2015b), truncated IPS with weight truncation parameter M , ES-estimator with parameter α (Aouali et al., 2023), IXestimator with parameter η, PM-estimator with parameter λ (Metelli et al., 2021), OS-estimator with parameter τ (Su et al., 2020) and LS-estimator with parameter λ (Sakhi et al., 2024) is provided in Table 2.
Note that the truncated IPS (IPS-TR) (Ionides, 2008a) employs truncation, resulting in a non-differentiable estimator. This non-differentiability complicates the optimization phase, often necessitating additional care and sometimes leading to computationally intensive discretizations (Papini et al., 2019). Furthermore, tuning the threshold M in IPS-TR is sensitive (Aouali et al., 2023).
In the following sections, we provide more details regarding heavy-tail assumption and theoretical results for the LSE estimator.
this section cite: ['b72', 'b91', 'b29', 'b83', 'b72', 'b91', 'b83', 'b76']

Section: Heavy-tail Assumption
In this section, the following heavy-tail assumption is made in our theoretical results. Assumption 5.1 (Heavy-tail weighted reward). The reward distribution P R|X,A and P X ⊗ π 0 (A|X) are such that for all learning policy π θ (A|X) ∈ Π θ and some ϵ ∈ [0, 1], the (1 + ϵ)-th moment of the weighted reward is bounded 4 ,
E P X ⊗π0(A|X)⊗P R|X,A w θ (A, X)R 1+ϵ ≤ ν. (9)
We make a few remarks. First, in comparison with the bounded reward function assumption in literature, (Metelli et al., 2021;Aouali et al., 2023), in Assumption 5.1, the reward function can be unbounded. Moreover, our assumptions are weaker with respect to the uniform overlap assumption 5 . In heavy-tailed bandit learning (Bubeck et al., 2013;Shao et al., 2018;Lu et al., 2019), a similar assumption to Assumption 5.1 on (1 + ϵ)-th moment of reward for some ϵ ∈ [0, 1] is assumed. In contrast, in Assumption 5.1, we consider the weighted reward. Note that, under uniform coverage (overlap) assumption, Assumption 5.1 can be interpreted as a heavy-tailed assumption on reward. Furthermore under a bounded reward, Assumption 5.1 would be equivalent with the heavy-tailed assumption on the (1 + ϵ)-th moment of weight function, w θ (a, x). A more detailed theoretical comparison is provided in App. D.1. We also provide a comparison with other estimators under bounded reward assumption in App. D.1.7 where Assumption 5.1 reduces to heavy-tailed assumption on weights.
this section cite: ['b72', 'b15', 'b87', 'b63']

Section: Regret Bounds
In this section, we provide an upper bound on the regret of the LSE estimator as discussed in the OPL scenario. The following novel is a helpful lemma to prove some results.
Lemma 5.2. Consider the random variable
Z > 0. For ϵ ∈ [0, 1], then V e λZ ≤ |λ| 1+ϵ E[Z 1+ϵ ] holds for λ < 0.
In the following Theorem, we provide an upper bound on the regret of learning policy under the LSE estimator.
this section cite: []

Section: Theorem 5.3 (Regret bounds).
For any γ ∈ (0, 1), given Assumption 5.1, assuming finite policy set |Π θ | < ∞ and
n ≥ (2|λ| 1+ϵ ν+ 4 3 γ) log |Π θ | δ γ 2 exp(2λν 1/(1+ϵ) )
for λ < 0, with probability at least (1 -δ), the following upper bound holds on the regret of the LSE estimator,
0 ≤ R λ (π θ , S) ≤ |λ| ϵ 1 + ϵ ν - 4(2 -γ) 3(1 -γ) log 4|Π θ | δ nλ exp(λν 1/(1+ϵ) ) - (2 -γ) (1 -γ)λ 4|λ| 1+ϵ ν log 4|Π θ | δ n exp(2λν 1/(1+ϵ) ) ,
where π θ is defined in equation 6 .
Sketch of Proof: Using Bernstein's inequality, Boucheron et al., 2013 and Lemma 5.2, we provide lower and upper bounds on estimation error for a fixed learning policy π θ . Then, we consider the following decomposition of regret,
V (π θ * ) -V (π θ ) = Est λ (π θ * ) (10
)
+ V λ LSE (S, π θ * ) -V λ LSE (S, π θ ) -Est λ (π θ ).
Note that, the second is negative. We can provide upper and lower bounds on estimation error (Theorem D.2 and Theorem D.3 in App. D.2), respectively.
As the regret bound in Theorem 5.3 depends on λ, we need to select an appropriate λ to study the convergence rate of regret bound with respect to n.
Proposition 5.4 (Convergence rate). Given Assumption 5.1, ) ) and setting λ = -n -1 1+ϵ , then the overall convergence rate of the regret upper bound is O(n -ϵ/(1+ϵ) ).
for any 0 < γ < 1, assuming n ≥ (2ν+ 4 3 γ) log |Π θ | δ γ 2 exp(2ν 1/(1+ϵ
Table 2: Comparison of estimators. We consider the bounded reward function, i.e., R max := sup (a,x)∈A×X r(a, x) for all estimators except LSE. B SN and V SN are the Bias and the Efron-Stein estimate of the variance of selfnormalized IPS. For the ES-estimator, we have T ES = B ES + (1/n) D KL (π θ ∥π 0 ) + log(4/δ) . where D KL (π θ ∥π 0 ) = A π θ (a|x) log(π θ (a|x)/π 0 (a|x))da. We also define power divergence as P α (π θ ∥π 0 ) := A π θ (a|x) α π 0 (a|x) (1-α) da is the power divergence with order α. For the IX-estimator, C η (π) is the smoothed policy coverage ratio. We compare the convergence rate of the concentration (or regret bound) for estimators. B and C are constants. For LS estimator, S λ(π θ ) is the discrepancy between π and π 0 .
Estimator Concentration Convergence Rate Heavy-tailed Regret Bound Noisy Reward Differentiability Subgaussian Like Tail IPS R 2 max P2(πθ∥π0) δn (Metelli et al., 2021) Rmax
O(n -1/2 ) × ✓ × ✓ × SN-IPS (Swaminathan & Joachims, 2015b) Rmax(B SN + V ES log 1 δ ) - × × × ✓ × IPS-TR (M > 0) (Ionides, 2008a) Rmax P2(πθ∥π0) log 1 δ n O(n -1/2 ) × ✓ × × ✓ IX (η > 0) (Gabbianelli et al., 2023) Rmax(2ηCη(πθ) + log(2/δ) ηn ) O(n -1/2 ) × ✓ × ✓ ✓ PM ( λ ∈ [0, 1])
P2(πθ∥π0) log 1 6 n O(n -1/2 ) × × × ✓ ✓ ES (α ∈ [0, 1]) (Aouali et al., 2023) Rmax DKL(πθ∥π0)+log(4 √ n/δ) n + T ES O (log(n)/n) 1/2 × ✓ × ✓ × OS (τ > 0) (Su et al., 2020) max β∈{2,3} β Pβ (πθ∥π0)(log 1 δ ) β-1 n β-1 O n (1-β)/β × × × ✓ × LS ( λ ≥ 0) (Sakhi et al., 2024) λSλ(πθ) + log(2/δ) λn O n -1/2 × ✓ × ✓ ✓ LSE (0 > λ > -∞ and ϵ ∈ [0, 1]) (ours) C 2 log(2|Πθ|/δ) n ϵ/(1+ϵ) O(n -ϵ/(1+ϵ) ) ✓ ✓ ✓ ✓ ✓
Discussion: Note that, if Assumption 5.1 holds for ϵ = 1 where the second moment of weighted reward is bounded, then we have the convergence rate of O(n -1/2 ). Moreover, if higher moments of the weighted reward are bounded, the second moment is also bounded, allowing our results for a bounded second moment to apply. Note that, our theoretical results on regret can be applied to unbounded weighted reward under Assumption 5.1 and other estimators can not guarantee the convergence rate of O(n -ϵ/(1+ϵ) ) under bounded (1 + ϵ)-th moment of weighted reward.
Bounded reward: Our results in Theorem 5.3 also holds under bounded reward and heavy-tailed weights assumption. More discussion is provided in App. D.1.7 .
Finite policy set: The results in this section assumed that the policy set, Π θ , is finite; this is for example the case in off-policy learning problems with a finite number of policies. If this assumption is violated, we can apply the growth function technique which is bounded by VC-dimension (Vapnik, 2013) or Natarajan dimension (Holden & Niranjan, 1995) as discussed in (Jin et al., 2021). Furthermore, we can apply PAC-Bayesian analysis (Gabbianelli et al., 2023) for the LSE estimator. More discussion regarding the PAC-Bayesian approach is provided in App. D.6.
this section cite: ['b14', 'b72', 'b36', 'b43', 'b29']

Section: Subgaussian Concentration:
We also study achieving subgaussian concentration for the LSE estimator where the dependency of regret on δ is subgaussian O log(1/δ) n , in App. D.7.
this section cite: []

Section: Bias and Variance
In this section, we provide an analysis of bias and variance for the LSE estimator.
Proposition 5.5 (Bias bound). Given Assumption 5.1, the following lower and upper bounds hold on the bias of the LSE estimator with λ < 0,
(n -1) 2n|λ| V(e λw θ (A,X)R ) ≤ B( V λ LSE (S, π θ ))(11)
≤ 1 1 + ϵ |λ| ϵ ν + 1 2nλ V(e λw θ (A,X)R ).
Remark 5.6 (Asymptotically Unbiased). By selecting λ as a function of n, which tends to zero as n → ∞, e.g. λ(n) = -n -ς for some ς > 0, the bounds in Proposition 5.5 becomes asymptotically zero. The overall convergence rate for upper bound is O(n -ϵ/(1+ϵ) ) by choosing ς = 1 1+ϵ . For example, if Assumption 5.1 holds for ϵ = 1, then by choosing ς = 1/2, we have the convergence rate of O(n -1/2 ) for the bias of the LSE estimator. Consequently, the LSE estimator is asymptotically unbiased.
For the variance of the LSE estimator, we provide the following upper bound.
Proposition 5.7 (Variance Bound). Assume that E[(w θ (A, X)R) 2 ] ≤ ν 2 6 holds. Then the variance of the LSE estimator with λ < 0, satisfies,
V( V λ LSE (S, π θ )) ≤ 1 n V(w θ (A, X)R) ≤ 1 n ν 2 .(12)
Variance Reduction: We can observe that the variance of the LSE is less than the variance of the IPS estimator for all λ < 0.
Combining the results in Proposition 5.5 and Proposition 5.7, we can derive an upper bound on MSE of the LSE estimator using MSE. The bias and variance trade-off for the LSE estimator and the comparison of different estimators in terms of bias and variance are provided in App. D.1.1.
this section cite: []

Section: Robustness of the LSE Estimator: Noisy Reward
In this section section, we study the robustness of the LSE estimator under noisy reward. We also investigate the performance of the LSE estimator under noisy (estimated) propensity scores in the App. E. To analyze the robustness of the LSE estimator, we extend the approach of tilted empirical risk introduced by Aminian et al. (2025), which provides generalization error bounds 7 under distributional shifts in supervised learning scenario under tilted empirical risk. Our analysis leverages these tools to quantify the robustness of the LSE to noisy rewards.
Suppose that due to an outlier or noise in receiving the feedback (reward), the underlying distribution of the reward given a pair of actions and contexts, P R|X,A is shifted via the distribution of noise or outlier, denoted as P R|X,A .
We model the distributional shift of reward via distribution P R|X,A due to inspiration by the notion of influence function (Marceau & Rioux, 2001;Christmann & Steinwart, 2004). Furthermore, we define the noisy reward LBF dataset as S with n data samples. For our result in this section, the following assumption is made. Assumption 5.8 (Heavy-tailed Weighted Noisy Reward). The P X ⊗ π 0 (A|X) and noisy reward distribution P R|X,A are such that for all learning policy π θ (A|X) ∈ Π θ and some ϵ ∈ [0, 1], the (1 + ϵ)-th moment of the weighted reward is bounded,
E P X ⊗π0(A|X)⊗ P R|X,A w θ (A, X)R 1+ϵ ≤ ν. (13
)
Under the noisy reward LBF dataset, we derive the following learning policy,
π θ ( S) = arg max π θ Π θ V λ LSE (π θ , S) .(14)
In the following theorem, we provide an upper bound on the regret of π θ ( S) as the learning policy under the noisy reward LBF dataset. 7 Generalization error is defined as difference between population and empirical risks in supervised learning scenario.
Theorem 5.9. For any γ ∈ (0, 1), given Assumption 5.1, Assumption 5.8 and assuming n ≥ (2|λ|
1+ϵ ν+ 4 3 γ) log |Π θ | δ γ 2 exp(2λν 1/(1+ϵ) )
for λ < 0, with probability at least (1 -δ), the following upper bound holds on the regret of the LSE estimator under noisy reward logged data,
0 ≤ R λ (π θ ( S), S) ≤ |λ| ϵ 1 + ϵ ν + 2A(γ) |λ| ϵ νC 1 (λ, n) + 2A(γ)C 1 (λ, n) + 2TV(P R|X,A , P R|X,A ) λ 2 D(ν, ν),
where Data-driven λ: For large number of samples, n → ∞, the second and third terms in Theorem 5.9 become negligible. Thus, under a noisy reward setting, λ can be chosen using the following objective function.
A(γ) = (2-γ) (1-γ) , C 1 (λ, n) = log 4|Π θ | δ n|λ| exp(λ ν 1/(1+ϵ) ) , D(ν, ν) = exp(|λ|ν 1/(1+ϵ) )-exp(|λ|ν 1/(1+ϵ) ) ν1/(1+ϵ) -ν 1/(1+ϵ) , π θ ( S) is de- fined in
λ ND := arg min λ∈(-∞,0) |λ| ϵ 1 + ϵ ν+ 2TV c (P R|X,A , P R|X,A ) λ 2 D(ν, ν),
where D(ν, ν) is defined in Theorem 5.9 .
this section cite: ['b68', 'b21']

Section: Robustness: The term
TVc(P R|X,A , P R|X,A ) λ 2
in the upper bound on the LSE regret under noisy reward scenario (Theorem 5.9) can be interpreted as the cost of noise associated with noisy reward. This cost can be reduced by increasing |λ|. However, increasing |λ| also amplifies the term |λ| ϵ 1+ϵ ν in the upper bound on regret. Therefore, there is a trade-off between robustness and regret, particularly for λ < 0 in the LSE estimator. More discussion regarding the robustness of the LSE is provided in App. D.9 .
this section cite: []

Section: Experiments
We present our experiments for OPE and OPL. Our aim is to demonstrate that our proposed estimators not only possess desirable theoretical properties but also compete with baseline estimators in practical scenarios. More details can be found in App.F.
this section cite: []

Section: Off-policy Evaluation
Baselines: For our experiments in OPE setting, we consider truncated IPS estimator (Swaminathan & Joachims, 2015a), PM estimator (Metelli et al., 2021), ES estimator (Aouali et al., 2023), IX estimator (Gabbianelli et al., 2023), SNIPS (Swaminathan & Joachims, 2015b), LS-LIN and LS estimators (Sakhi et al., 2024), and OS (shrinkage) (Su et al., 2020) estimator as baselines. For this purpose, we consider an LBF dataset which has only a single context (state), denoted as x 0 . We consider the learning and logging policies as Gaussian distributions, π θ (•|x 0 ) ∼ N (µ 1 , σ 2 ) and π 0 (•|x 0 ) ∼ N (µ 2 , σ 2 ). The reward function is a positive exponential function e αx 2 which is unbounded. We also set our parameters to observe different tail distributions. We fix µ 1 = 0.5, µ 2 = 1, σ 2 = 0.25 and change the value of α which controls the tail of the weighted reward variable, α ∈ {1.4, 1.6}. We also examine different values of α and the effect of a number of samples for a fixed α in App. G.1. Moreover, we conduct a similar experiment when logging and learning policies are Lomax distributions 8 in App. G.1.
Metrics: We calculate the bias, variance, and MSE of estimators by running the experiments for 10K times each one over 1000 samples.
Discussion: The results presented in Table 4 demonstrate that the LSE estimator has better performance in terms of both MSE and variance when compared to other baselines. We also conducted experiments for OPE under some UCI datasets in App G.12 . More experiments are provided in App. G.11 for comparison of LSE estimator with LS estimator .
this section cite: ['b72', 'b29', 'b83', 'b91']

Section: Off-policy Learning
Baselines: For our experiments in OPL, we compare our LSE estimator against several non-regularized baseline estimators, including, truncated IPS (Swaminathan & Joachims, 2015a), PM (Metelli et al., 2021), ES (Aouali et al., 2023), IX (Gabbianelli et al., 2023), BanditNet (Joachims et al., 2018), LS-LIN (Sakhi et al., 2024) and OS estimator (Su   8 The Lomax distribution is a Pareto Type II distribution which is a heavy-tailed distribution.
this section cite: ['b72', 'b29', 'b45', 'b83']

Section: Datasets:
In off-policy learning scenario, we apply the standard supervised to bandit transformation (Beygelzimer & Langford, 2009) on a classification dataset: Extended-MNIST (EMNIST) (Xiao et al., 2017) to generate the LBF dataset. We also run on FMNIST in App.G.2 . This transformation assumes that each of the classes in the datasets corresponds to an action. Then, a logging policy stochastically selects an action for every sample in the dataset. For each data sample x, action a is sampled by logging policy. For the selected action, propensity score p is determined by the softmax value of that action. If the selected action matches the actual label assigned to the sample, then we have r = 1, and r = 0 otherwise. So, the 4-tuple (x, a, p, r) makes up the LBF dataset.
Noisy (Estimated) propensity score: For noisy propensity score, motivated by Halliwell (2018) and the discussion in App.E.1, we assume a multiplicative inverse Gamma noise 9 on π 0 for b ∈ R + , π 0 = 1 U π 0 , where π(a|x) is the estimated propensity scores and U ∼ Gamma(b, b).
Noisy reward: Inspired by Metelli et al. (2021), we also consider noise in reward samples. In particular, we model noisy reward by a reward-switching probability P f ∈ [0, 1] to simulate noise in the reward samples. For example, a reward sample of r = 1 may switch to r = 0 with probability P f .
Logging policy: To have logging policies with different performances, given inverse temperature 10 τ ∈ {1, 10}, first, we train a linear softmax logging policy on the fullylabeled dataset. Then, when we apply standard supervisedto-bandit transformation on the dataset, the results obtained from the linear logging policy which are weights of each action according to the input, will be multiplied by the inverse temperature τ and then passed to a softmax layer. Thus, as the inverse temperature τ Increases, we will have more uniform and less accurate logging policies.
Metric: We evaluate the performance of the different estimators based on the accuracy of the trained model. Inspired by London & Sandler (2019), we calculate the accuracy for a deterministic policy where the accuracy of the model based on the argmax of the softmax layer output for a given context is computed.
For each value of τ , we apply the LSE estimator and observe the accuracy over three runs on EMNIST. The deterministic accuracies of LSE, PM, ES, IX, BanditNet, OS and LS-LIN for τ ∈ {1, 10} are presented in Table 3.
Discussion: The results presented in Table 3 demonstrate that the LSE estimator achieves maximum accuracy (with less variance) in most scenarios compared to all baselines. Furthermore, an experiment on a real-world dataset, KUAIREC (Gao et al., 2022), is provided in App. G.4 . More discussion and experiments are provided in App. G .
this section cite: ['b11', 'b34', 'b72', 'b60', 'b30']

Section: Conclusion
In this work, inspired by the log-sum-exponential operator, we proposed a novel estimator for off-policy learning and evaluation applications. Subsequently, we conduct a comprehensive theoretical analysis of the LSE estimator, including a study of bias and variance, along with an upper bound on regret under heavy-tailed assumption. Furthermore, we explore the performance of our estimator in sce- 9 If Z ∼ Gamma(α, β), then we have fZ (z) = β α Γ(α) z α-1 e -βz . 10 The inverse temperature τ is defined as π0(ai|x) = exp(h(x,a i )/τ ) k j=1 exp(h(x,a j )/τ ) where h(x, ai) is the i-th input to the softmax layer for context x ∈ X and action ai ∈ A.
narios involving estimated propensity scores or heavy-tailed weighted rewards. Results from our experimental evaluation demonstrate that our estimator, guided by our theoretical framework, performs competitively compared to most baseline estimators in off-policy learning and evaluation.
this section cite: []

Section: Future Works
In this section, we outline several potential directions for future work based on our LSE estimator.
LSE and Regularization: Using regularization for offpolicy learning can improve the performance of estimators (Aminian et al., 2024;Metelli et al., 2021). As future works, we plan to study the effect of regularization on the LSE estimator from both theoretical and practical perspectives.
LSE with Positive λ: Inspired by the application of LSE operator in supervised learning for positive (λ > 0 (Li et al., 2023), exploring the LSE estimator for positive λ in scenarios where the logged dataset is imbalance in terms of rewards or actions, can be an interesting direction. Note that our current theoretical analysis can be applied to negative λ and is not applicable for positive λ.
LSE and RL: We envision extending the application of our estimator to more challenging reinforcement learning settings, such as those considered by Chen & Jiang (2022); Zanette et al. (2021); Xie et al. (2019a), where the i.i.d. assumption does not necessarily hold. In these scenarios, theoretical guarantees must be adapted to account for dependencies in the data-for example, by extending the analysis to martingale difference sequences.
LSE and Missing reward: Note that, in our problem formulation, we assumed that we have access to reward for all logged samples. However, in some applications as discussed in (Aminian et al., 2024), for some data samples the reward (feedback) is missed. In future work, we extend the application of LSE function to these scenarios where the reward (feedback) is partially missed.
this section cite: ['b6', 'b72', 'b59', 'b6']

Section: Impact Statement
This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. Park, M., Lee, S. Y., Hong, J. S., and Kwon, N. K. Deep deterministic policy gradient-based autonomous driving for mobile robots in sparse reward environments. Sensors, 22(24):9574, 2022.
Swaminathan, A. and Joachims, T. Batch learning from logged bandit feedback through counterfactual risk minimization. The Journal of Machine Learning Research, 16(1):1731-1755, 2015a. Swaminathan, A. and Joachims, T. The self-normalized estimator for counterfactual learning. Advances in Neural Information Processing Systems, 28, 2015b. Tang, L., Rosales, R., Singh, A., and Agarwal, D. Automatic ad format selection via contextual bandits. In Proceedings of the 22nd ACM International Conference on Information & Knowledge Management, pp. 1587-1594, 2013. Thomas, P., Theocharous, G., and Ghavamzadeh, M. Highconfidence off-policy evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2015. Tolstikhin, I. O. and Seldin, Y. Pac-bayes-empiricalbernstein inequality. Advances in Neural Information Processing Systems, 26, 2013. Tsiatis, A. A. Semiparametric theory and missing data. Springer, 2006. Vakili, S., Liu, K., and Zhao, Q. Deterministic sequencing of exploration and exploitation for multi-armed bandit problems. IEEE Journal of Selected Topics in Signal Processing, 7(5):759-767, 2013. Vapnik, V. The nature of statistical learning theory. Springer science & business media, 2013. Wang, J., Liu, Y., and Li, B. Reinforcement learning with perturbed rewards. In Proceedings of the AAAI conference on artificial intelligence, 2020. Wang, Y.-X., Agarwal, A., and Dudık, M. Optimal and adaptive off-policy evaluation in contextual bandits. In International Conference on Machine Learning, pp. 3589-3597. PMLR, 2017. Weitzen, S., Lapane, K. L., Toledano, A. Y., Hume, A. L., and Mor, V. Principles for modeling propensity scores in medical research: a systematic literature review. Pharmacoepidemiology and Drug Safety, 13(12):841-853, 2004. Williams, C. K. and Barber, D. Bayesian classification with gaussian processes. IEEE Transactions on pattern analysis and machine intelligence, 20(12):1342-1351, 1998. Xiao, H., Rasul, K., and Vollgraf, R. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017. Xie, T., Ma, Y., and Wang, Y.-X. Towards optimal off-policy evaluation for reinforcement learning with marginalized importance sampling. Advances in Neural Information Processing Systems, 32, 2019a. Xie, Y., Zhu, Y., Cotton, C. A., and Wu, P. A model averaging approach for estimating propensity scores by optimizing balance. Statistical methods in medical research, 28 (1):84-101, 2019b. Xue, B., Wang, Y., Wan, Y., Yi, J., and Zhang, L. Efficient algorithms for generalized linear bandits with heavy-tailed rewards. Advances in Neural Information Processing Systems, 36, 2024. Yan, Y., Li, G., Chen, Y., and Fan, J. The efficacy of pessimism in asynchronous Q-learning. IEEE Transactions on Information Theory, 2023. Yin, M. and Wang, Y.-X. Towards instance-optimal offline reinforcement learning with pessimism. Advances in Neural Information Processing Systems, 34:4065-4078, 2021. Yu, X., Shao, H., Lyu, M. R., and King, I. Pure exploration of multi-armed bandits with heavy-tailed payoffs. In UAI, pp. 937-946, 2018. Zanette, A., Wainwright, M. J., and Brunskill, E. Provable benefits of actor-critic methods for offline reinforcement learning. Advances in neural information processing systems, 34:13626-13640, 2021. Zhang, T. Information-theoretic upper and lower bounds for statistical estimation. IEEE Transactions on Information Theory, 52(4):1307-1321, 2006.
Zhang, X., Chen, J., Wang, H., Xie, H., and Li, H. Uncertainty-aware off-policy learning. arXiv preprint arXiv:2303.06389, 2023a.
Zhang, X., Chen, J., Wang, H., Xie, H., Liu, Y., Lui, J. C., and Li, H. Uncertainty-aware instance reweighting for offpolicy learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023b. URL https:  //openreview.net/forum?id=1pWNhmbllE.
Zhong, H., Huang, J., Yang, L., and Wang, L. Breaking the moments condition barrier: No-regret algorithm for bandits with super heavy-tailed payoffs. Advances in Neural Information Processing Systems, 34:15710-15720, 2021.
Zhu, J., Wan, R., Qi, Z., Luo, S., and Shi, C. Robust offline policy evaluation and optimization with heavytailed rewards. arXiv preprint arXiv:2310.18715, 2023.
Zhu, J., Wan, R., Qi, Z., Luo, S., and Shi, C. Robust offline reinforcement learning with heavy-tailed rewards. In
this section cite: []

Section: References
Ref_id:b0 Title: Goalconstrained sparse reinforcement learning for end-to-end driving Year: (2021)
Ref_id:b1 Title:  Year: (2016)
Ref_id:b2 Title: Online bandit learning with offline preference data Year: (2024)
Ref_id:b3 Title: Bayesian inference of individualized treatment effects using multi-task Gaussian processes Year: (2017)
Ref_id:b4 Title: Simpler pac-bayesian bounds for hostile data Year: (2018)
Ref_id:b5 Title: Learning robust control policies for end-to-end autonomous driving from datadriven simulation Year: (2020)
Ref_id:b6 Title:  Year: (2024)
Ref_id:b7 Title: Generalization and robustness of the tilted empirical risk Year: ()
Ref_id:b8 Title: Exponential smoothing for off-policy learning Year: ()
Ref_id:b9 Title: Generalized random forests Year: (2019)
Ref_id:b10 Title: Personalized diabetes management using electronic medical records Year: (2017)
Ref_id:b11 Title: The offset tree for learning with partial labels Year: (2009)
Ref_id:b12 Title: Value constrained model-free continuous control Year: (2019)
Ref_id:b13 Title: Counterfactual reasoning and learning systems: The example of computational advertising Year: (2013)
Ref_id:b14 Title: Concentration inequalities: A nonasymptotic theory of independence Year: (2013)
Ref_id:b15 Title: Bandits with heavy tail Year: (2013)
Ref_id:b16 Title: The importance of pessimism in fixed-dataset policy optimization Year: (2020)
Ref_id:b17 Title: Log-sumexp neural networks and posynomial models for convex and log-log-convex data Year: (2019)
Ref_id:b18 Title: The master equation and the convergence problem in mean field games Year: (2019)
Ref_id:b19 Title: Offline reinforcement learning under value and density-ratio realizability: the power of gaps Year: (2022)
Ref_id:b20 Title: Surrogate objectives for batch policy optimization in onestep decision making Year: (2019)
Ref_id:b21 Title: On robustness properties of convex risk minimization methods for pattern recognition Year: (2004)
Ref_id:b22 Title: Herd behavior and aggregate fluctuations in financial markets Year: (2000)
Ref_id:b23 Title: Propensity score methods for bias reduction in the comparison of a treatment to a nonrandomized control group Year: (1998)
Ref_id:b24 Title: Handling sparse rewards in reinforcement learning using model predictive control Year: (2023)
Ref_id:b25 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b26 Title: Cooperative multi-agent bandits with heavy tails Year: (2020)
Ref_id:b27 Title: Doubly robust policy evaluation and optimization Year: (2014)
Ref_id:b28 Title: More robust doubly robust off-policy evaluation Year: (2018)
Ref_id:b29 Title: Importanceweighted offline learning done right Year: (2023)
Ref_id:b30 Title: A fullyobserved dataset and insights for evaluating recommender systems Year: (2022)
Ref_id:b31 Title: Alphastable modeling of noise and robust time-delay estimation in the presence of impulsive noise Year: (1999)
Ref_id:b32 Title: Robust stochastic bandit algorithms under probabilistic unbounded adversarial attack Year: (2020)
Ref_id:b33 Title: Pac-bayes generalisation bounds for heavy-tailed losses through supermartingales Year: (2022)
Ref_id:b34 Title: The log-gamma distribution and non-normal error. Variance (Accepted for Publication Year: (2018)
Ref_id:b35 Title: Image denoising: A nonlinear robust statistical approach Year: (2001)
Ref_id:b36 Title: On the practical applicability of vc dimension bounds Year: (1995)
Ref_id:b37 Title: Pac-bayes under potentially heavy tails. Advances in Neural Information Processing Systems Year: (2019)
Ref_id:b38 Title: Sub-gaussian mean estimation in polynomial time Year: (2018)
Ref_id:b39 Title: Tackling heavy-tailed rewards in reinforcement learning with function approximation: Minimax optimal and instancedependent regret bounds Year: (2024)
Ref_id:b40 Title: A new process uncertainty robust student'st based kalman filter for sins/gps integration Year: (2017)
Ref_id:b41 Title: Truncated importance sampling Year: (2008)
Ref_id:b42 Title: Truncated importance sampling Year: (2008)
Ref_id:b43 Title: Is pessimism provably efficient for offline RL Year: (2021)
Ref_id:b44 Title: Policy learning" without"overlap: Pessimism and generalized empirical Bernstein's inequality Year: (2022)
Ref_id:b45 Title: Deep learning with logged bandit feedback Year: (2018)
Ref_id:b46 Title: Learning representations for counterfactual inference Year: (2016)
Ref_id:b47 Title: Balanced policy evaluation and learning. Advances in Neural Information Processing Systems Year: (2018)
Ref_id:b48 Title: Low-rank matrix bandits with heavy-tailed rewards Year: (2024)
Ref_id:b49 Title: Towards optimal doubly robust estimation of heterogeneous causal effects Year: (2020)
Ref_id:b50 Title: Deep reinforcement learning for autonomous driving: A survey Year: (2021)
Ref_id:b51 Title: Precision medicine Year: (2019)
Ref_id:b52 Title:  Year: (2019)
Ref_id:b53 Title: Perturbed-history exploration in stochastic linear bandits Year: (2019)
Ref_id:b54 Title: Exploration scavenging Year: (2008)
Ref_id:b55 Title: Improving propensity score weighting using machine learning Year: (2010)
Ref_id:b56 Title: Weight trimming and propensity score weighting Year: (2011)
Ref_id:b57 Title: Minimax optimal bandits for heavy tail rewards Year: (2022)
Ref_id:b58 Title: Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms Year: (2011)
Ref_id:b59 Title: On tilted losses in machine learning: Theory and applications Year: (2023)
Ref_id:b60 Title: Bayesian counterfactual risk minimization Year: (2019)
Ref_id:b61 Title: Boosted off-policy learning Year: (2023)
Ref_id:b62 Title: Unbounded returns and the possibility of credit rationing: A note on the stiglitz-weiss and arnold-riley models Year: (2018)
Ref_id:b63 Title: Optimal algorithms for lipschitz bandits with heavy-tailed rewards Year: (2019)
Ref_id:b64 Title: Mean estimation and regression under heavy-tailed distributions: A survey Year: (2019)
Ref_id:b65 Title: Robust multivariate mean estimation: The optimality of trimmed mean Year: (2021)
Ref_id:b66 Title: Generalization bounds via convex analysis Year: (2022)
Ref_id:b67 Title: Online-to-pac conversions Year: (2023)
Ref_id:b68 Title: On robustness in risk theory Year: (2001)
Ref_id:b69 Title: Propensity score estimation with boosted regression for evaluating causal effects in observational studies Year: (2004)
Ref_id:b70 Title: No-regret algorithms for heavytailed linear bandits Year: (2016)
Ref_id:b71 Title: Policy optimization via importance sampling Year: (2018)
Ref_id:b72 Title: Subgaussian and differentiable importance sampling for off-policy evaluation and learning Year: (2021)
Ref_id:b73 Title: Machine learning: a probabilistic perspective Year: (2012)
Ref_id:b74 Title: Explore no more: Improved high-probability regret bounds for non-stochastic bandits Year: (2015)
Ref_id:b75 Title: Quasi-oracle estimation of heterogeneous treatment effects Year: (2021)
Ref_id:b76 Title: Optimistic policy optimization via multiple importance sampling Year: (2019)
Ref_id:b77 Title: Ads by whom? ads about what? exploring user influence and contents in social advertising Year: (2013)
Ref_id:b78 Title: Error modelling for multi-sensor measurements in infrastructure-free indoor navigation Year: (2018)
Ref_id:b79 Title: Off-policy evaluation for large action spaces via policy convolution Year: (2023)
Ref_id:b80 Title: Off-policy evaluation for large action spaces via embeddings Year: (2022)
Ref_id:b81 Title: Off-policy evaluation for large action spaces via conjunct effect modeling Year: (2023)
Ref_id:b82 Title: Pac-bayesian offline contextual bandits with guarantees Year: (2023)
Ref_id:b83 Title: Logarithmic smoothing for pessimistic off-policy evaluation, selection and learning Year: (2024)
Ref_id:b84 Title: Evaluating uses of data mining techniques in propensity score estimation: a simulation study Year: (2008)
Ref_id:b85 Title: Understanding machine learning: From theory to algorithms Year: (2014)
Ref_id:b86 Title: Estimating individual treatment effect: generalization bounds and algorithms Year: (2017)
Ref_id:b87 Title: Almost optimal algorithms for linear stochastic bandits with heavy-tailed payoffs Year: (2018)
Ref_id:b88 Title: Robust learning for optimal treatment decision with np-dimensionality Year: (2016)
Ref_id:b89 Title: Adapting neural networks for the estimation of treatment effects Year: (2019)
Ref_id:b90 Title: Learning from logged implicit exploration data Year: (2010)
Ref_id:b91 Title: Doubly robust off-policy evaluation with shrinkage Year: (2020)
Ref_id:b92 Title: Direct importance estimation with model selection and its application to covariate shift adaptation Year: (2007)
