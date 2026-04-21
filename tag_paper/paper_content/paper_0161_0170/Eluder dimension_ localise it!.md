Title: Eluder dimension: localise it!
Abstract: We establish a lower bound on the eluder dimension of generalised linear model classes, showing that standard eluder dimension-based analysis cannot lead to first-order regret bounds. To address this, we introduce a localisation method for the eluder dimension; our analysis immediately recovers and improves on classic results for Bernoulli bandits, and allows for the first genuine first-order bounds for finite-horizon reinforcement learning tasks with bounded cumulative returns.

Section: Introduction
We study decision-making problems where the regret admits a first-order (small-cost) bound of the form R n ≤ nη(a ⋆ )Γ n + Γ ′ n , with η(a ⋆ ) the optimal mean per-round cost and Γ n , Γ ′ n instance and model dependent complexities. The challenge in obtaining first-order bounds is in how we measure the complexity of the task. The (global) eluder dimension (Russo and Van Roy, 2013) is a standard complexity measure used to provide worst-case guarantees, but, as we shall argue, it ignores the local structure of the problem: analyses based on the global eluder dimension often introduce a factor in Γ n that scales like a perstep worst-case information/curvature parameter (denoted κ), which cancels out η(a ⋆ ) and destroys first-order gains. This subtle issue is present in much of the previous work on first-order bounds.
We show that by localising the eluder dimension-restricting it to a small neighbourhood of the optimal model-these κ terms can be moved into the lower-order Γ ′ n term. This tightens the link between exploration and the actual difficulty of the instance and yields genuinely first-order bounds.
Our contributions may be summarised as follows:
1. Localised ℓ 1 -eluder dimension. We define a localised ℓ 1 -eluder dimension (in the sense of Liu et al., 2022) over a small-excess-loss neighbourhood, which avoids the κ dependency in the classic generalised linear bandit setting (Filippi et al., 2010;Faury et al., 2020). 2. Necessity of localisation. We prove lower bounds showing that the global ℓ 1 -and ℓ 2 -eluder dimensions (Russo and Van Roy, 2013;Liu et al., 2022) must scale with κ in the generalised linear setting, and that this negates small-cost or variance-dependent improvements. 3. Stochastic bandits. We propose a version-space optimistic algorithm, ℓ-UCB, which takes a loss ℓ and builds confidence sets for the cost function. Under (a) bounded loss, (b) a Bernstein variance condition, and (c) a triangle condition (Foster and Krishnamurthy, 2021), ℓ-UCB achieves a small-cost bound using analysis based on our localised eluder dimension. 4. Reinforcement learning. We extend our method to online RL, giving ℓ-GOLF, and obtain the first κ-free first-order regret bound for finite-horizon RL with bounded rewards/costs.
this section cite: ['b33', 'b25', 'b14', 'b12', 'b33', 'b25', 'b15']

Section: Related work
Small-cost in bandits. Adversarial small-cost bounds are known (Neu, 2015;Allen-Zhu et al., 2018;Foster and Krishnamurthy, 2021;Ito et al., 2020;Olkhovskaya et al., 2023). However, adversarial algorithms are often conservative in stochastic regimes (Lattimore and Szepesvári, 2020, Ch. 18) and do not transfer cleanly to reinforcement learning, where optimism (Lai and Robbins, 1985) remains central for low regret (Ayoub et al., 2020;Weisz et al., 2023;Wu et al., 2025;Moulin et al., 2025).
In stochastic bandits, first-order bounds typically assume distributional knowledge (e.g., noise/cost models) (Abeille et al., 2021;Faury et al., 2022;Janz et al., 2024;Liu et al., 2024;Lee et al., 2024). We consider stochastic bandits with function approximation and unknown bounded cost distributions, aligning with adversarial-style uncertainty but in a stochastic environment.
Small-cost in reinforcement learning. Online first-order bounds have been shown by Wang et al. (2023) with the (strong) distributional Bellman completeness assumption. This assumption was then removed by Ayoub et al. (2024), but in the offline setting; Wang et al. (2024) extended this result back to the online setting. However, by relying on the global eluder dimension, Wang et al. (2023Wang et al. ( , 2024) ) suffer a (hidden) κ-dependence in the leading term that undermines first-order gains.
Reward-first-order vs cost-first-order. Reward-first-order bounds help when the optimal reward is small (Jin et al., 2020;Wagenmaker et al., 2022); this is very different from the cost-first-order guarantees we target (our results actually hold for both small-cost and small-reward settings). Smallcost results have been previously shown in structured settings such as tabular Markov decision processes (Lee et al., 2020) and linear-quadratic regulators (Kakade et al., 2020).
Instance-optimal exploration. Pure-exploration studies instance-dependent sample complexity, with notable works including policy-difference estimation for tabular reinforcement learning (Narang et al., 2024) and PEDEL for linear function approximation (Wagenmaker and Jamieson, 2022). The algorithm-agnostic lower bounds of Al-Marjani et al. (2022) show that PEDEL is near instanceoptimal for tabular MDPs. These works are complementary to our regret-focused results.
this section cite: ['b30', 'b1', 'b15', 'b16', 'b31', 'b21', 'b3', 'b40', 'b42', 'b28', 'b0', 'b13', 'b17', 'b26', 'b24', 'b39', 'b5', 'b38', 'b39', 'b38', 'b18', 'b23', 'b20', 'b29', 'b27']

Section: Background on generalised linear models & loss functions
We collect the notation and standing assumptions for generalised linear models (GLMs) and losses used throughout. Fix a dimension d ∈ N + , and let A, Θ ⊂ R d be closed sets. Let U ⊂ R be a closed interval and let µ : U → [0, 1] be increasing. The GLM class with link µ and parameter set Θ is GLM(µ, Θ) = {a → µ(⟨a, θ⟩) : a ∈ R d , θ ∈ Θ, ⟨a, θ⟩ ∈ U } .
We also consider losses ℓ : [0, 1] × [0, 1] → R, where ℓ(y, ŷ) evaluates a prediction ŷ against an outcome y. The next assumption records the structural conditions on (A, Θ, µ, ℓ) used in our analysis; the final condition is satisfied when ℓ is the negative log-likelihood of the GLM associated with µ. Assumption 1. We make the following assumptions:
A ⊂ B d 2 (action set bound) (∃S > 0) Θ ⊂ SB d 2 (parameter set bound) (∀(a, θ) ∈ A × Θ) ⟨a, θ⟩ ∈ U (valid domain) (∃L > 0, ∀u, u ′ ∈ U ) |µ(u) -µ(u ′ )| ≤ L|u -u ′ | ( L-Lipschitz link) (∃M ≥ 1, ∀u ∈ U • ) |μ(u)| ≤ M μ(u) (M -self-concordant link) (∃1 ≤ κ < ∞) κ ≥ sup u∈U • 1/ μ(u) (link derivative lower bound)
(∀y ∈ [0, 1], ∀u ∈ U ) ∂ u ℓ(y, µ(u)) = µ(u) -y .
(link and loss are compatible)
Remark 1. The requirement that M, κ ≥ 1 in Assumption 1 is there solely to simplify our bounds.
Examples of loss and link combinations that satisfy our assumptions include the log-loss with the sigmoid link function and the Poisson loss with the exponential link function: Example 1. The log-loss function ℓ X (y, p) = -y log p -(1 -y) log(1 -p) together with the sigmoid link function µ X : [-S, S] → [0, 1] given by u → 1/(1 + e -u ) satisfies Assumption 1 with L = 1/4, M = 1 and κ = 3e S .
Algorithm 1 the ℓ-UCB bandit algorithm input loss function ℓ, model F, nonnegative, nondecreasing confidence widths (β t ) t≥1 for time-step t ∈ N + do let F t be the subset of the model given by
F t = f ∈ F : t-1 i=1 ℓ(Y i , f (A i )) ≤ inf f ∈F t-1 i=1 ℓ(Y i , f (A i )) + β t ,
compute an optimistic function f t ∈ F t and action A t ∈ A that satisfy 3 Bandits with bounded costs and the ℓ-UCB algorithm Our bandit setting comprises a set of actions A and a corresponding set of action-dependent cost distributions P = {P a : a ∈ A} supported on the interval [0, 1] (we will write supp P for the support of a measure P ). At each round t ∈ N + , a learner selects an action A t ∈ A and receives a cost Y t ∼ P At . We measure the learner's performance over n ∈ N + rounds by the n-step regret
R n = n t=1 η(A t ) -η(a ⋆ )
where η : a → yP a (dy) and a ⋆ ∈ arg min a∈A η(a) .
The learner may base its choice of A t on the past observations A 1 , Y 1 , . . . , A t-1 , Y t-1 , any extra randomness independent of the observations (say, for tie-breaking), and prior knowledge in the form of a model class: a set F of functions A → [0, 1] known to contain η. The key assumptions here are: Assumption 2 (Bounded costs). We have ∪ a∈A supp P a ⊂ [0, 1].
Assumption 3 (Realisability). We have that η ∈ F .
Algorithm Our algorithm, ℓ-UCB (Algorithm 1) is an implementation of optimism with empirical risk minimisation-based confidence intervals.
At each time-step t ∈ N + , the algorithm constructs a confidence set F t for η composed of functions in F for which the empirical risk under the loss function ℓ : [0, 1] × [0, 1] → R does not exceed that of the empirical risk minimiser by more than β t , where (β t ) t≥1 is a problem-dependent nonnegative, nondecreasing sequence of confidence widths. The algorithm then computes an optimistic functionaction pair (f t , A t ) ∈ F t × A such that f t (A t ) ≤ f (a), ∀(f, a) ∈ F t × A , and plays A t . Optimising over F t × A is difficult without further assumptions. In Appendix A we detail a standard convex relaxation of this optimisation problem applicable to self-concordant models.
The crucial component to the ℓ-UCB algorithm obtaining small-cost adaptivity is the right choice of the loss function ℓ used to construct the confidence intervals (and well-chosen confidence widths, based on the loss function and model class). Our requirements will be stated in the form of an assumption on the offset versions of the loss functions, and their expectations, which are defined thus: We will write Φ(F) = {φ f : f ∈ F } and Φ(F) = { φf : f ∈ F } for the respective loss classes.
Let ∆ : [0, 1] × [0, 1] → R + be the triangular discrimination function given by ∆(0, 0) = 0 and ∆(p, q) = (p -q) 2 p + q otherwise.
this section cite: []

Section: Assumption 4 (Loss function assumptions).
There exist constants b, c, γ > 0 such that for all (f, a) ∈ F × A, letting Y ∼ P a , the following three bounds hold:
|φ f (Y, a)| ≤ b a.s. , (bounded loss) Var φ f (Y, a) ≤ c φf (a) , (variance condition) ∆(f (a), η(a)) ≤ γ φf (a) .
this section cite: []

Section: (triangle condition)
The first two conditions in Assumption 4, boundedness and the variance condition, allow for a Bernstein-type concentration on the excess loss class. The triangle condition is used in the regret decomposition to move from fast concentration to small-cost bounds. The conditions in Assumption 4 implicitly depend on η, and thus ought to hold uniformly for all η ∈ F . Recall our two losses: The squared loss function fails to satisfy the triangle condition; Theorem 2 of Foster and Krishnamurthy (2021) shows that squared loss cannot lead to the small-cost bounds we seek.
this section cite: ['b15']

Section: The localised eluder dimension & first-order regret bounds for bandits
We define the (global) ℓ 1 -eluder dimension of Liu et al. (2022) as follows; we will henceforth refer to this quantity as the eluder dimension, forgoing the ℓ 1 quantifier.
Definition 2 (Eluder dimension). Let Z be a set and Ψ be a class of real-valued functions on Z, and let z = (z 1 , z 2 , . . . , z n ) be a length n sequence in Z. We define the following:
1. We say x ∈ Z is ε-independent of z with respect to Ψ if there exists a ψ ∈ Ψ such that n t=1 |ψ(z t )| ≤ ε and |ψ(x)| > ε. 2. We say that z is an ε-eluder sequence with respect to Ψ if for all t ≤ n, z t is ε-independent of z 1 , . . . , z t-1 with respect to Ψ.
3. The ε-eluder dimension dim elud (ε; Ψ) of Ψ is the length n of the longest ω-eluder sequence with respect to Ψ for any ω ≥ ε. Remark 2. The ℓ 2 -eluder dimension at scale ε of a function class Ψ is equivalent to the ℓ 1 -eluder dimension of the function class {z → ψ(z) 2 : ψ ∈ Ψ} at scale ε 2 . We think of the ℓ 1 -eluder dimension as loss-agnostic, whereas of the ℓ 2 -eluder dimension as baking in the squared loss.
Our upcoming regret bound will require the choice of a localised model class F ′ ⊂ F, and will depend on it through the following two quantities:
1. The eluder dimension of Φ(F ′ ), the expected excess loss class induced by the localised model class. 2. The number of time-steps t ∈ N + for which the optimistic function f t is not within the localised set F ′ .
The localised eluder dimension will feature in the leading term; by taking F ′ small, we can make it independent of κ. Taking F ′ small may increase the second term, the number of optimistic functions falling outside of F ′ , but this contributes to the regret as an additive term only.
Theorem 1 (Regret bound for ℓ-UCB in bandits). Fix δ ∈ (0, 1), n ∈ N + , bandit instance P, model class F and a loss function ℓ. Suppose that (P, F, ℓ) satisfy Assumptions 2 to 4. Let N n denote the 1/n-covering number of Φ(F) with respect to the uniform metric, and for each t ∈ N + let β t = 5/2 + 15(b + c) log(N n h t /δ) where h t = e + log(1 + t) .
Let F ′ ⊂ F, and denote by d n the 1/n-eluder dimension of Φ(F ′ ). Define
Γ n = γ(1 + (d n + 1)b + 4d n β n log(1 + nb)) .
Suppose a learner uses Algorithm 1, ℓ-UCB, over the course of n-many interactions with P, with model class F, loss function ℓ and confidence widths (β t ) t≥1 . Then, with probability at least 1 -δ, R n ≤ 3 nη(a ⋆ )Γ n + 6Γ n + card{t ≤ n : f t ̸ ∈ F ′ } .
The proof of Theorem 1 is located in Appendix D. Remark 3. The localised model class F ′ does not need to be chosen to run the algorithm. The regret of the algorithm scales automatically with the best possible choice of F ′ . The localised eluder dimension does not need to be computed to run the algorithm; it is an analysis-only quantity. Remark 4. The covering number N n featuring in the confidence widths (β t ) t≥1 does not depend on κ; the confidence widths themselves thus do not bring in any κ dependence.
this section cite: ['b25']

Section: Why localisation matters: eluder dimension lower bound for generalised linear models
The dependence on the ε-eluder dimension of Φ(F ′ ) will be our focus; this can be thought of as measuring the number of times we are 'surprised' at the scale ε > 0, in that there was a model f ∈ F ′ with low expected excess loss on past inputs that has high expected excess loss on some unseen action. The following lower bound shows that it is vital to only consider 'surprises' near a ⋆ .
Theorem 2 (GLM ℓ 1 -eluder dimension lower bound). Let (µ, ℓ) satisfy the last four properties of Assumption 1 (link L-Lipschitz, M -self-concordant, link-derivative lower bound, and link-loss compatibility). Fix S ≥ 4/M and assume that [-S, 0] ⊂ U . Write
κ = μ(0) 2 μ(-S/2) ∈ (0, ∞) , b = min{⌊S⌋, d -1} .
Then, there exist A ⊂ B d 2 and Θ ⊂ SB d 2 such that (A, Θ, µ, ℓ) satisfy Assumption 1 and for every ε ≤ μ(0)/(2M 2 ) the eluder dimension of the expected excess-loss class Φ(F) with F = GLM(µ, Θ) satisfies
dim elud (ε; Φ(F)) ≥ d -1 4b exp min b 16 , log(κ) 2 8SM 2 + 4 log(κ)
, for a sequence of actions taking values in A.
The proof of Theorem 2 is given in Appendix G.
The quantity κ > 0 can be thought of as the ratio of the information gain in the middle of the parameter set and that at a large step in the negative direction; in common GLMs, κ ≈ κ. Corollary 3. Consider the setting of Theorem 2 with the log-loss ℓ X and the sigmoid link function µ(u) = 1/(1 + e -u ). Then, M = 1 and μ(0) = 1/4. Therefore, for S ≥ 4, d ≥ 2 and any ε ≤ 1/8, the ε-eluder dimension of Φ(GLM(µ, Θ)) exceeds d-1 4b exp{b/4300}, where b = min{⌊S⌋, d -1}.
The corollary follows by substituting the relevant quantities into Theorem 2; the proof is omitted.
To understand the implications of Theorem 2, consider the setting of logistic bandits in the usual low-information regime, where ⟨a ⋆ , θ ⋆ ⟩ ≈ -S; think clickthrough rates in online advertising, where even the best adverts rarely get clicked on. Then η(a ⋆ ) ≈ μ(⟨a ⋆ , θ ⋆ ⟩) ≈ exp(-S), which suggests that our regret should be excellent; but at the same time κ ≈ exp(S), and thus the eluder dimension scales as exp(S), completely cancelling out the benefit of the η(a ⋆ ) small-cost term. This results in a bound that fails to truly adapt to the problem instance. We now show how localisation helps.
this section cite: []

Section: Regret upper bound with localisation for the generalised linear model setting
We now instantiate Theorem 1 for generalised linear models; see Appendix F.3 for the relevant proofs.
Consider the GLM setting. For any f t let θ t ∈ Θ be such that f t (•) = µ(⟨•, θ t ⟩). For any r > 0, let Θ ′ (r) = {θ ∈ Θ : ∀a ∈ A , |⟨a, θ -θ ⋆ ⟩| ≤ r} be the r-localised set of parameters and define the corresponding r-localised model class
F ′ (r) = {µ(⟨•, θ⟩) : θ ∈ Θ ′ (r)} .
The eluder dimension of Φ(F ′ (r)) can be upper-bounded as a function of r as follows: Proposition 4. Let Assumption 1 hold. Then, there exists a universal constant C > 0 such that for any r, ε > 0, the ε-eluder dimension of Φ(F ′ (r)) is bounded as
dim elud (ε; Φ(F ′ (r))) ≤ Cd exp(rM ) log(1 + S 2 L exp(rM )/ε) .
In the above bound, taking r = S we end up with an e S dependence, which is an upper bound on κ (up to constant factors; the upper bound is tight in the logistic setting). However, localising to a 1/M neighbourhood, we instead obtain a bound for the ε-eluder
dimension of Φ(F ′ (1/M )) of dim elud (ε; Φ(F ′ (1/M ))) ≤ Cd log(1 + S 2 L/ε) .
Localisation thus allows for first-order bounds where the effect of μ(a ⋆ ) is not overshadowed by κ.
The cost of this 1/M -localisation in terms of the additive term in the regret is bounded as follows: Proposition 5. Under Assumption 1, on the high-probability event of Theorem 1, for any n ∈ N + ,
card{t ≤ n : |⟨A t , θ t -θ ⋆ ⟩| > 1/M } ≤ 64dκM 2 β n log 1 + (64/3)κ 2 M 2 S 2 β n .
That this additive term depends on κ is to be expected; all algorithms for generalised linear models without κ in the leading term have such an additive dependence (Abeille et al., 2021). (Note, our bound depends on the sequence A 1 , . . . , A n rather than holding for all actions-this is fine.)
Using the bounds of Propositions 4 and 5, we obtain the following specialisation of Theorem 1: Proposition 6 (Regret for ℓ-UCB with the logistic model). Let δ ∈ (0, 1), S > 0 and n ∈ N + . Consider the setting of Theorem 1, with the model class F = GLM(µ, Θ) where µ(u) = 1/(1 + e -u ) and the logistic loss function ℓ X . Consider running ℓ-UCB with confidence widths (β t ) t∈N+ given by β t = 5/2 + 60(2S + 1) d log(1 + 8Sn) + log(h t /δ) , h t = e + log(1 + t) .
Then, for a constant C > 0, with probability at least 1 -δ, the resulting regret satisfies the bound
R n ≤ C nη(a ⋆ )dβ n log(1 + Sn) + Cdβ n (log(1 + Sn)) 2 + e 2S d log(1 + β n ) .
The same result of Proposition 6 also holds, up to constant factors, for the Poisson model; each log-loss specific result used in the proof of Proposition 6 has a Poisson equivalent in Appendix C.
Observe that the regret bound of Proposition 6 holds as soon as Assumptions 1 to 3 are met. Importantly, we do not assume that the rewards are generated by a generalised linear model. However, much of the literature does make that assumption, so we make comparisons in that setting.
this section cite: ['b0']

Section: Discussion in the logistic bandit & maximum likelihood estimation settings
The maximum likelihood estimation (MLE) setting is the well-studied setting where the costs are sampled from a known generalised linear model (one might also call this a 'well-specified' setting).
A special case of the MLE setting with bounded rewards is the logistic bandit setting. Here, η is given by a generalised linear model with the sigmoid link function and the responses are given by
Y t ∼ Bernoulli(η(A t )) for each t ∈ N + .
In this setting, the leading term in the regret bound of Proposition 6 nearly matches the lower bound given by Abeille et al. (2021), which states that there exists a C > 0 such that
R n ≥ Cd nv(a ⋆ ) where v(a ⋆ ) = η(a ⋆ )(1 -η(a ⋆ )) .
Likewise, Proposition 6 almost matches the upper bounds of Faury et al. (2022) for their logisticbandit-specific algorithm, which guarantee that for some C > 0, with probability at least
1 -δ, R n ≤ CSd nv(a ⋆ ) log(n/δ) + CS 6 dκ(log(n/δ)) 2 .
The suboptimality of Proposition 6 here is in that it depends on η(a ⋆ ), providing only a small-cost bound, rather than on v ⋆ (a ⋆ ); the latter allows for a simultaneous small-cost and small-reward bound. This is because Proposition 6 only assumes that the triangle condition is met on one side of the reward interval, and so only allows for small-cost bounds; strengthening the assumption to be two-sided (which is satisfied by the logistic model) would allow us to recover the v(a ⋆ ). (We do not do this, as it would rule out, for example, the Poisson GLM, which only gives small-cost bounds.)
Interestingly, while Faury et al. (2022) only consider the logistic bandit setting, their analysis actually shows a regret bound for the wider bounded reward setting. The distinction between our work and that of Faury et al. (2022) is that where we use an analysis-only localisation technique to move κ to an additive term, Faury et al. (2022) use an explicit algorithmic warm-up procedure to do this. That is, they run an approximation of an optimal design at the start of interaction, until their confidence sets have shrunk to a neighbourhood of the true parameter (on the good event where the confidence sets do indeed contain the true parameter). 1 The change from algorithmic localisation to analysis-only localisation is vital for the upcoming reinforcement learning setting where, because we do not have random access to state-action pairs, the solving of an optimal design is not feasible.
The works of Lee et al. (2024) and Emmenegger et al. (2024) also do away with the warm-up employed in Faury et al. (2022), using techniques based on likelihood ratios. However, they rely on their likelihood ratios forming a martingale, which restricts the results to the MLE setting.
this section cite: ['b0', 'b13', 'b13', 'b13', 'b13', 'b24', 'b10', 'b13']

Section: Remark 5 (Bernoullisation).
Any algorithm A that yields first-order regret for the logistic setting can be used to obtain first-order regret for the bounded reward setting using Bernoullisation. The trick is thus: for each time-step t ∈ N + , upon observing Y t ∈ [0, 1], we sample
Y ′ t ∼ Bernoulli(Y t ) and feed Y ′
t to the algorithm A. Since the conditional means of Y t and Y ′ t are the same, firstorder properties are preserved. Bernoullisation, however, destroys any second-order adaptivity of the algorithm. Indeed, consider the case where the (Y t ) t∈N+ are equal to 1/2 almost surely. Then, running empirical loss minimisation with the log-loss on (Y t ) t∈N+ converges to 1/2 after a single observation, but running the same procedure on the corresponding sequence (Y ′ t ) t∈N+ of independent Bernoulli(1/2) random variables leads to an Ω(1/ √ n) absolute error in the estimate.
this section cite: []

Section: First-order regret bounds for online reinforcement learning
We consider the episodic reinforcement learning setting with horizon H ∈ N + . Let M = (S, A, c, P, s 1 ) be a Markov decision process (MDP) with states S, actions A, a cost function c = (c 1 , . . . , c H ) with c h : S × A → [0, 1], a deterministic starting state s 1 ∈ S, and a transition kernel P = (P 1 , . . . , P H ) with P h mapping from S × A to probability measures over S.
The learner interacts with the MDP M for n ∈ N + episodes. At the start of each episode t ∈ [n], the learner specifies a deterministic policy π t = (π t 1 , . . . , π t H ), where π t h : S → A for each h ∈ [H]. We allow the policy π t to depend on the states, actions and costs observed prior to the start of the tth episode, but not on the cost function c or the dynamics P , as these are assumed to be unknown. The learner's aim will be to minimise the expected cumulative cost incurred over the n episodes. To formalise this, let v π h be the value function of policy π in M , given by
v π h (s) = E π H i=h c(S i , π(S i )) | S h = s , for each s ∈ S, where E π [ • | S h = s]
denotes the expectation with respect to the states S h , . . . , S H induced by following the policy π in the MDP M starting at S h = s. Then, letting
v t h := v π t h (h ∈ [H]), the n-episode regret is given by R n = n t=1 v t 1 (s 1 ) -v ⋆ 1 (s 1 )
1 Faury et al. (2022) also propose an online data-rejection procedure that can be used instead of a warm-up. This is, however, again, an algorithmic tool, in contrast to our analysis-only approach.
Algorithm 2 The ℓ-GOLF algorithm input loss function ℓ, models F and G, nonnegative confidence widths (β t ) t for episode t ∈ N + do for each h ∈ [H] let
L t-1 h (f, f ′ ) = t-1 i=1 ℓ 1 ∧ (C i h + f ∧ (S i h+1 )), f ′ (S i h , A i h )
and let F t be the subset of F given by
F t = f ∈ F : L t-1 h (f h+1 , f h ) ≤ inf g∈G h L t-1 h (f h+1 , g) + β t , ∀h ∈ [H] , compute an optimistic function f t ∈ arg min f ∈F t f 1 (s 1 , π f (s 1 ))
and play the policy π t := π ft greedy with respect to f t end for
where v ⋆ h (h ∈ [H]
) is the optimal value function, defined formally just after Eq. ( 1). The key assumption that our learner will be allowed to exploit is the following: Assumption 5. Costs are nonnegative and sum to at most one over each episode.
this section cite: ['b13']

Section: Preliminaries on Q-functions, Bellman optimality operators and greedy policies
Let Q be the set of all maps S ×A → [0, 1] H . For q ∈ Q, we write q h for the map (s, a) → (q(s, a)) h (entry h of q(s, a)), and we write q ∧ for the function S → [0, 1] H defined by
(q ∧ (s)) h = min a∈A q h (s, a) for all s ∈ S and h ∈ [H] .
For convenience, we may augment each q with q H+1 = 0, to reflect the standard boundary condition. We let T : Q → Q denote the Bellman optimality operator for the MDP M , given by
T : q → c + q ∧ dP ,
where, with slight abuse of notation, the integral is to be understood as with respect to the product P = P 1 × • • • × P H . We define the optimal action-value function q ⋆ for M to be the element of Q satisfying
T q ⋆ = q ⋆ ,(1)
and define the value function v ⋆ for M to be v ⋆ = q ⋆∧ . For any function q ∈ Q, we write π q for the policy greedy with respect to q, defined by π q h (s) ∈ arg min a∈A q h (s, a) for all s ∈ S and h ∈ [H] .
this section cite: []

Section: The ℓ-GOLF algorithm, model and loss assumptions & regret bound
Our ℓ-GOLF algorithm (Algorithm 2) is an extension of ℓ-UCB to the episodic online reinforcement learning setting, generalising the GOLF algorithm of Jin et al. (2021) to arbitrary loss functions (GOLF is recovered by taking ℓ to be the squared loss). The algorithm requires the specification of a loss function ℓ : [0, 1] 2 → R, confidence widths (β t ) t∈[n] and function classes G, F ⊂ Q. The model F contains candidate functions for estimating q ⋆ , and the model G contains candidates for estimating T f for f ∈ F. We will use the following two assumptions of Antos et al. (2008): Assumption 6 (Realisability). We assume that q ⋆ ∈ F.
Assumption 7 (Generalised completeness). We assume that T F ⊂ G.
The algorithm proceeds to, in each episode t ∈ N + , construct a confidence set F t ⊂ F containing action-value functions that are close to satisfying the Bellman optimality condition f = T f on the data observed thus far, with errors penalised according to ℓ. It then selects an optimistic function f t ∈ F, and plays the policy π t := π ft greedy with respect to f t . Our analysis will use the following: Definition 3. For any f ∈ Q, h ∈ [H], x ∈ S × A and s ′ ∈ S, we let
y f h : (x, s ′ ) → 1 ∧ (c h (x) + f ∧ h+1 (s ′
)) be the response under the model f . For (f, g) ∈ F × G, we define the excess Bellman loss function
φ f,g h (x, s ′ ) = ℓ(y f h (x, s ′ ), g h (x)) -ℓ(y f h (x, s ′ ), (T f ) h (x))
, and the expected excess Bellman loss function
φf,g h (x) = φ f,g h (x, •)dP h (x) .
We write Φ(F, G) and Φ(F, G) for the classes of excess and expected excess Bellman losses.
Our assumptions on the loss function here mirror those of the bandit setting: Assumption 8 (RL loss function assumptions). There exist constants b, c, γ > 0 such that for all (f, g) ∈ F × G, h ∈ [H], x ∈ S × A, S ′ ∼ P h (x), the following hold:
|φ f,g h (x, S ′ )| ≤ b a.s. , (RL boundedness) Var φ f,g h (x, S ′ ) ≤ c φf,g h (x) , (RL variance condition) ∆(f h (x), (T f ) h (x)) ≤ γ φf,f h (x) . (RL triangle condition)
Our main result is captured by the following theorem. Note, the eluder complexity defined therein is equivalent to an ℓ 1 version of the Bellman eluder dimension of Jin et al. (2021). Theorem 7. Fix δ ∈ (0, 1), n ∈ N + , MDP M , model classes F and G and a loss function ℓ. Suppose that (M, F , G, ℓ) satisfy Assumptions 6 to 8. Let h t = e + log(1 + t) for each t ∈ [n], let N n be the 1/n-covering number of the function class Φ(F, G) with respect to the uniform metric, and let
β t = 5/2 + 15(b + c) log(N n h t /δ) , t ∈ N + .
Let F ′ ⊂ F, and define Z to be the set of functions (S × A) H → R mapping
x → H h=1 φ f,f h (x h ) for some f ∈ F ′ .
Let P f denote the state-action occupancy measure on (S × A) H induced by the interconnection of M and the policy greedy with respect to f ∈ F, and let Ψ be the family of functionals on Z mapping
z → zdP f for each f ∈ F ′ . Let d n denote the 1/n-eluder dimension of Ψ. Define Γ n = γ(1 + (d n + 1)b + d n β n log(1 + nb)) .
Suppose a learner uses Algorithm 2, ℓ-GOLF, over the course of n-many episodes with M , with model classes F and G, loss function ℓ and confidence widths (β t ) t∈N+ .
Then, with probability at least 1 -δ, the learner's regret is bounded as
R n ≤ 3 Hnv ⋆ 1 (s 1 )Γ n + 6HΓ n + card{t ≤ n : f t ̸ ∈ F ′ } .
Theorem 7 is established in Appendix E. For context, the closest results to ours are those of Wang et al. (2023Wang et al. ( , 2024) ) for online RL. Both provide a small-cost regret bound scaling with the Bellman eluder dimension; however, without our notion of a localised dimension, their regret bound scales with κ in the leading term for logistic linear models. This entirely offsets any benefit of their small-cost analysis; the bound is not truly instance-adaptive. Moreover, Wang et al. (2023) assumes that the distributional Bellman operator (Bellemare et al., 2017) lies in their model class, an assumption that is significantly stronger than our Assumption 7 (as discussed in Ayoub et al., 2024). An argument for extending the results from costs to rewards was given in Ayoub et al. (2025).
this section cite: ['b19', 'b2', 'b19', 'b39', 'b38', 'b39', 'b6', 'b5', 'b4']

Section: Conclusion
We have shown that standard eluder dimension analysis inherently fails to achieve first-order regret bounds in generalised linear model settings. By introducing the localised ℓ 1 -eluder dimension, we overcome this limitation, removing problematic worst-case dependencies and achieving genuinely adaptive, first-order regret bounds. Our refined analysis recovers and sharpens classical results in Bernoulli bandit scenarios and demonstrates clear practical advantages through the ℓ-UCB algorithm.
Moreover, our localisation approach successfully extends to finite-horizon reinforcement learning via the ℓ-GOLF algorithm, providing the first genuine first-order regret bounds in this setting. This highlights the crucial role of localisation techniques in developing instance-adaptive algorithms, opening promising avenues for further exploration in broader learning contexts.
this section cite: []

Section: References
Ref_id:b0 Title: Instance-wise minimax-optimal algorithms for logistic bandits Year: (2021)
Ref_id:b1 Title: Make the minority great again: First-order regret bound for contextual bandits Year: (2018)
Ref_id:b2 Title: Learning near-optimal policies with Bellman-residual minimization based fitted policy iteration and a single sample path Year: (2008)
Ref_id:b3 Title: Model-based reinforcement learning with value-targeted regression Year: (2020)
Ref_id:b4 Title: Rectifying Regression in Reinforcement Learning Year: (2025)
Ref_id:b5 Title: Switching the Loss Reduces the Cost in Batch Reinforcement Learning Year: (2024)
Ref_id:b6 Title: A distributional perspective on reinforcement learning Year: (2017)
Ref_id:b7 Title: Concentration Inequalities: A Nonasymptotic Theory of Independence Year: (2013)
Ref_id:b8 Title: On the performance of Thompson sampling on logistic bandits Year: (2019)
Ref_id:b9 Title: Is a Good Representation Sufficient for Sample Efficient Reinforcement Learning? Year: (2020)
Ref_id:b10 Title: Likelihood ratio confidence sets for sequential decisionmaking Year: (2024)
Ref_id:b11 Title: Mixability in Statistical Learning Year: (2012)
Ref_id:b12 Title: Improved optimistic algorithms for logistic bandits Year: (2020)
Ref_id:b13 Title: Jointly Efficient and Optimal Algorithms for Logistic Bandits Year: (2022)
Ref_id:b14 Title: Parametric bandits: The generalized linear case Year: (2010)
Ref_id:b15 Title: Efficient first-order contextual bandits: prediction, allocation, and triangular discrimination Year: (2021)
Ref_id:b16 Title: Tight first-and second-order regret bounds for adversarial linear bandits Year: (2020)
Ref_id:b17 Title: Exploration via linearly perturbed loss minimisation Year: (2024)
Ref_id:b18 Title: Reward-free exploration for reinforcement learning Year: (2020)
Ref_id:b19 Title: Bellman eluder dimension: new rich classes of RL problems, and sample-efficient algorithms Year: (2021)
Ref_id:b20 Title: Information theoretic regret bounds for online nonlinear control Year: (2020)
Ref_id:b21 Title: Asymptotically efficient adaptive allocation rules Year: (1985)
Ref_id:b22 Title: Bandit Algorithms Year: (2020)
Ref_id:b23 Title: Bias no more: High-probability data-dependent regret bounds for adversarial bandits and MDPs Year: (2020)
Ref_id:b24 Title: A Unified Confidence Sequence for Generalized Linear Models, with Applications to Bandits Year: (2024)
Ref_id:b25 Title: When is partially observable reinforcement learning not scary? Year: (2022)
Ref_id:b26 Title: Almost Free: Self-concordance in Natural Exponential Families and an Application to Bandits Year: (2024)
Ref_id:b27 Title: Towards instance-optimality in online PAC reinforcement learning Year: (2022)
Ref_id:b28 Title: Optimistically Optimistic Exploration for Provably Efficient Infinite-Horizon Reinforcement and Imitation Learning Year: (2025)
Ref_id:b29 Title: Sample complexity reduction via policy difference estimation in tabular reinforcement learning Year: (2024)
Ref_id:b30 Title: First-order regret bounds for combinatorial semi-bandits Year: (2015)
Ref_id:b31 Title: First-and Second-Order Bounds for Adversarial Linear Contextual Bandits Year: (2023)
Ref_id:b32 Title: Information Theory: From Coding to Learning Year: (2025)
Ref_id:b33 Title: Eluder dimension and the sample complexity of optimistic exploration Year: (2013)
Ref_id:b34 Title: Generalized self-concordant functions: a recipe for Newton-type methods Year: (2019)
Ref_id:b35 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b36 Title: Instance-dependent near-optimal policy identification in linear MDPs via online experiment design Year: (2022)
Ref_id:b37 Title: First-order regret in reinforcement learning with linear function approximation: A robust estimation approach Year: (2022)
Ref_id:b38 Title: The central role of the loss function in reinforcement learning Year: (2024)
Ref_id:b39 Title: The benefits of being distributional: Small-loss bounds for reinforcement learning Year: (2023)
Ref_id:b40 Title: Online RL in Linearly q π -Realizable MDPs Is as Easy as in Linear MDPs If You Learn What to Ignore Year: (2023)
Ref_id:b41 Title: Time-uniform self-normalized concentration for vectorvalued processes Year: (2023)
Ref_id:b42 Title: Computationally efficient RL under linear Bellman completeness for deterministic dynamics Year: (2025)
