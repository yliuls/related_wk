Title: ONLINE LEARNING AND EQUILIBRIUM COMPUTATION WITH RANKING FEEDBACK
Abstract: Online learning in arbitrary, and possibly adversarial, environments has been extensively studied in sequential decision-making, and it is closely connected to equilibrium computation in game theory. Most existing online learning algorithms rely on numeric utility feedback from the environment, which may be unavailable in human-in-the-loop applications and/or may be restricted by privacy concerns. In this paper, we study an online learning model in which the learner only observes a ranking over a set of proposed actions at each timestep. We consider two ranking mechanisms: rankings induced by the instantaneous utility at the current timestep, and rankings induced by the time-average utility up to the current timestep, under both full-information and bandit feedback settings. Using the standard external-regret metric, we show that sublinear regret is impossible with instantaneous-utility ranking feedback in general. Moreover, when the ranking model is relatively deterministic, i.e., under the Plackett-Luce model with a temperature that is sufficiently small, sublinear regret is also impossible with timeaverage utility ranking feedback. We then develop new algorithms that achieve sublinear regret under the additional assumption that the utility sequence has sublinear total variation. Notably, for full-information time-average utility ranking feedback, this additional assumption can be removed. As a consequence, when all players in a normal-form game follow our algorithms, repeated play yields an approximate coarse correlated equilibrium. We also demonstrate the effectiveness of our algorithms in an online large-language-model routing task.

Section: INTRODUCTION
Online learning has been extensively studied as a model for sequential decision-making in arbitrary and possibly adversarial environments (Shalev-Shwartz et al., 2012;Hazan et al., 2016). At each round, the agent commits to a strategy, takes an action, and then receives feedback from the environment, often in numeric form, such as the utility vector (in the full-information feedback setting) or the realized utility value (in the bandit feedback setting). Numerous algorithms achieve no-regret guarantees, i.e., they ensure that the agent's external regret grows sublinearly with time (Shalev-Shwartz et al., 2012;Hazan et al., 2016). Moreover, online learning is closely connected to equilibrium computation in game theory: when all players employ no-regret learning in the repeated play of a normal-form game (NFG), their time-average play approaches a coarse correlated equilibrium (CCE) (Cesa-Bianchi & Lugosi, 2006).
However, numeric utility feedback may not always be available in real-world applications. For instance, when the feedback is elicited from an environment with humans in the loop, it is typically much easier for them to compare or rank candidate actions than to provide calibrated numerical scores. This phenomenon has been widely recognized and is exemplified by the success of reinforcement learning from human feedback (RLHF) in fine-tuning language models (Ouyang et al., 2022). Moreover, even when well-defined numeric utilities exist, they may be inaccessible to the learning agent due to privacy or security constraints. As a concrete example, consider an online platform (see Figure 1 (a)) that recommends commodities to a stream of customers over time, where t = 1 In (a), an online platform recommends food options to a customer at each timestep and receives a ranking over the proposed items, which it uses to improve recommendation quality. In (b), an online dating app recommends potential matches; users rank the suggested candidates, and the platform leverages these rankings to learn matching equilibria over time.
customers arriving at different timesteps may have different preferences. While the platform aims to improve its recommendations, customers may be unable or unwilling to reveal their true valuations.
Depending on whether customers are one-shot (arrive, rank, and leave forever) or long-lived with memory, the ranking feedback may be induced either by the instantaneous utility at each timestep or by the time-average utility aggregated over historical utility vectors. The platform thus seeks to minimize the regret of its recommendations under ranking-only feedback. Yet, the fundamental limits and algorithmic solutions for this online learning setting remain elusive.
Ranking feedback becomes particularly relevant in game-theoretic settings, where multiple humans repeatedly interact, and the goal is to compute an equilibrium of the underlying game. For example, consider an online dating platform that recommends potential matches (see Figure 1 (b)). In each round, each user may provide only a ranking over the recommended candidates, and the platform aims to compute an equilibrium outcome, i.e., a matching between users, that (approximately) respects everyone's preferences. Related scenarios arise in other matching platforms, e.g., ride-sharing services that match drivers and passengers based on their preferences, such as drivers' preferences over trip lengths and riders' preferences over driving styles (e.g., punctuality or cautiousness). Although these applications may seem reminiscent of the classical stable matching model (Gale & Shapley, 1962), our setting is fundamentally different. See Appendix A for a detailed comparison.
In this paper, we systematically study online learning and equilibrium computation with ranking feedback in a nonstochastic environment, where the loss vectors may be generated arbitrarily, and potentially even adversarially. This setting can also be viewed as a generalization of the stochastic bandits with ranking feedback studied recently in Maran et al. (2024). See Appendix A for a detailed comparison. Our goal is to understand when regret minimization in this setting is possible, and to develop new algorithms that provide both regret-minimization and equilibrium-approximation guarantees. We summarize our contributions as follows.
Contributions. We consider two types of ranking feedback, categorized by how the rankings are generated: one based on the instantaneous utility at each timestep (InstUtil Rank), and one based on the time-average utility up to the current timestep (AvgUtil Rank). We establish the following results: (i) sublinear regret is impossible under InstUtil Rank feedback. Moreover, under AvgUtil Rank feedback, sublinear regret remains unattainable (up to logarithmic factors) when the ranking model is overly deterministic (i.e., when the temperature parameter τ > 0 in (PL) is small); (ii) we propose new algorithms that achieve sublinear regret under an additional assumption that the utility vectors have sublinear variation; (iii) we show that this variation assumption can be removed under full-information AvgUtil Rank feedback when τ is a constant; and (iv) when all players run our no-regret learning algorithms in the repeated play of a normal-form game, the induced play yields an approximate coarse correlated equilibrium. Our contributions are summarized in Table 1.
We present an application to large-language-model routing in Section 8, and provide experimental validation of our algorithms in Appendix C.
this section cite: ['b45', 'b21', 'b45', 'b21', 'b9', 'b19']

Section: PRELIMINARIES
The basic notation and a brief introduction to normal-form games are deferred to Appendix B.
Lower Bound Full-Information Bandit InstUtil Rank Ω(T ) for τ ≤ O (1)
AvgUtil Rank Ω(T ) for τ ≤ O 1 T log T Ω(T ) for τ ≤ O 1 log T Upper Bound (τ = O(1), Sublinear Regret)
this section cite: []

Section: Full-Information Bandit
InstUtil Rank Assumption 4.2 AvgUtil Rank ✓ Assumption 4.2 (q < 1 3 ) Table 1: Summary of our contributions, including the negative results (top) and the positive results (bottom). The bottom table lists the minimal assumptions required to achieve sublinear regret in each setting (✓indicates that no additional assumptions are needed). Here, τ > 0 denotes the temperature parameter of the ranking model in (PL).
this section cite: []

Section: ONLINE LEARNING
We study online learning in a non-stochastic and potentially adversarial environment, where an agent interacts with the environment over multiple timesteps by selecting an action and receiving feedback at each timestep. The agent's action set is finite and denoted as A := a 1 , a 2 , . . . , a |A| with |A| > 1. At each timestep t, the agent commits to a mixed strategy π (t) ∈ ∆ A . In the classical online learning setting (with numeric feedback), with full-information feedback, the agent observes a utility vector u (t) ∈ [-1, 1] A ; with bandit feedback, the agent observes only the realized utility u (t) (a (t) ) for the sampled action a (t) ∼ π (t) . The agent's goal is to minimize her (external) regret, defined as the difference between her cumulative utility and that of the best fixed action in hindsight. Formally, for any integer T > 0, the regret is defined as R (T ),external := max π∈∆ A T t=1 u (t) , π -π (t) .
(2.1) Since our goal is to minimize regret, and regret is unchanged if the utility vector u (t) is shifted by an additive constant at each timestep t, we may assume without loss of generality that u (t) (a |A| ) = 0, i.e., the last action always receives zero utility for any t ∈ [T ].
this section cite: []

Section: ONLINE LEARNING ALGORITHMS WITH NUMERIC FEEDBACK
Our main results later will be modular, in the sense that any standard online learning algorithm with (full-information) numeric feedback, including projected gradient descent (PGD), multiplicativeweights update (MWU), and Follow-The-Regularized-Leader (FTRL) (Hazan et al., 2016), can be used as a deterministic black-box oracle in the algorithms we develop.
As a preliminary, we formally define such deterministic oracles here. Let Alg : ∞ t=0 R A t → ∆ A denote an online learning algorithm, which can be viewed as a mapping from a sequence of utility vectors to a distribution over the action set A. Thus, given utility vectors u (s) t s=1 from timesteps 1 through t, the algorithm outputs the next strategy π (t+1) = Alg u (s) t s=1 to be played at timestep t + 1. Note that this formulation implicitly assumes that Alg is deterministic. Finally, we denote the (external) regret incurred by Alg as
R (T ),external Alg, u (t) T t=1 := max π∈∆ A T t=1 u (t) , π -Alg u (s) t-1 s=1 , (2.2)
which can be made sublinear in T for any sequence of utility vectors u (t) T t=1 (Hazan et al., 2016).
this section cite: ['b21', 'b21']

Section: ONLINE LEARNING WITH RANKING FEEDBACK
In online learning with ranking feedback, at each timestep t, the agent does not have direct access to u (t) , nor the realized utility (the utility of the realized action at timestep t). Instead, at timestep t, she can propose a multiset (which may include repeated elements) of actions o (t) , and receive a permutation σ (t) ∈ Σ o (t) (Σ(•) returns the set of all permutations) from the environment, representing a ranking of those actions in o (t) . In the full-information setting, o (t) = A, i.e., the whole action set is proposed. In the bandit setting, o (t) = K ≤ |A|. Suppose the agent's strategy at timestep t is π (t) ∈ ∆ A , then we assume that in this bandit setting, the actions in o (t) are proposed by sampling from π (t) independently (with replacement). This way, the empirical average of the utilities will be an unbiased estimator of the expected utility, i.e., E t) , π (t) .
a∈o (t) u (t) (a) K = u (
We will adopt this proposal mechanism throughout the paper, which was also used in the literature of adversarial dueling bandits (Saha et al., 2021).
Let σ (t) (k) ∈ A be the k th element of the permutation for any k ∈ [K]. Then, for any
k 1 < k 2 ∈ [K], action σ (t) (k 1 ) is preferred over action σ (t) (k 2 ).
For notational simplicity, we define a i σ > a j if action a i appears ahead of a j (a i is preferred over a j ) in a permutation σ.
For the ranking model, we consider the standard Plackett-Luce (PL) model (Luce, 1959;Plackett, 1975), where at each timestep t, conditioned on the proposed action set o (t) , the ranking σ (t) is sampled according to
P σ (t) o (t) = K k 1 =1 exp 1 τ r (t) σ (t) (k1) K k 2 =k 1 exp 1 τ r (t) (σ (t) (k2)) ,(PL)
where r (t) ∈ R A is some utility vector based on which the ranking is determined, τ > 0 is the temperature parameter that determines how uncertain the ranking model is: when τ → 0 + , the model is absolutely certain, and the action with a larger utility in r (t) will always be ranked in front of the actions with a smaller utility in the permutation (ties are broken in favor of the smaller index).
The utility vector r (t) depends on the problem setting, which we will introduce next.
We consider two types of ranking feedback throughout the paper, based on the choice of r (t) in (PL): (i) ranking by the instantaneous utility (InstUtil Rank); (ii) ranking by the time-average utility (AvgUtil Rank). These two feedback types may be motivated by different applications (cf. Section 1), and have been studied in Yue et al. (2012); Saha & Gaillard (2022); Maran et al. (2024). Both feedback types can also be further separately defined for the full-information and bandit settings, respectively, as follows.
this section cite: ['b29', 'b36', 'b50', 'b38']

Section: InstUtil Rank: Ranking with Instantaneous Utility.
The first type of ranking feedback we consider is based on the instantaneous utility function, i.e., r (t) = u (t) in (PL). This type is relevant when the feedback provider is oblivious or one-shot. For example, a stream of customers arrive in an online fashion, each of whom arrives, ranks, and then leaves, see e.g., Mansour et al. (2015). When the environment is stationary and stochastic, the classical dueling-bandits model likewise uses instantaneous utilities for comparison and ranking (Yue et al., 2012;Du et al., 2020).
Full-information setting. All actions are proposed and evaluated at each timestep t, i.e., o (t) = A.
Hence, the agent's performance can be evaluated by u (t) , π (t) . Note that this does not imply the agent has access to the full vector u (t) , as that would defeat the purpose of our ranking-feedback setting. Accordingly, we evaluate performance using the standard external regret R (T ),external defined in (2.1). Bandit setting. Only the proposed actions at each timestep t can be evaluated and ranked, with the associated elements in the vector u (t) . In particular, the proposed actions are evaluated by the average utility of 1 K a∈o (t) u (t) (a), leading to the following performance metric of regret: R (T ) := max
π∈∆ A T t=1   u (t) , π - 1 K a∈o (t) u (t) (a)   .
(3.1)
Note that such a definition is an external regret, which differs from the regret studied in (multi-)dueling-bandits (Yue et al., 2012;Du et al., 2020;Saha et al., 2021;Saha & Gaillard, 2022). Details can be found in Appendix A.
AvgUtil Rank: Ranking with Time-average Utility.
The second type of ranking-feedback is based on the time-average utility, which differs for the fullinformation and bandit settings, as detailed below. This type is relevant when the feedback provider has memory and can use the history of utilities for ranking. For example, the customers are longlived in the platform, see e.g., Küc ¸ükgül et al. (2022) and Baldwin (2009). Notably, under bandit feedback, when τ → 0 + and the environment is stationary and stochastic, such a model aligns with the one studied in the recent work of Maran et al. (2024).
Full-information setting. The time-average utility vector of u
avg := 1 t t s=1 u (s) will be used as the r (t) in (PL), and the same (external-)regret R (T ),external from (2.1) will be used as the metric. Bandit setting. Only the proposed actions will be given to the environment to evaluate. For instance, the platform (learning agent) may recommend K restaurants among all possibilities to the user (environment) to try out, so that the user will only know her evaluations of those K restaurants. As a result, the average utility is now defined as the empirical mean of the utility vectors over time. Formally, for each timestep t > 0 and action a ∈ A, we define
u (t) empirical (a) := t s=1 u (s) (a) a ′ ∈o (s) 1 (a = a ′ ) t s=1 a ′ ∈o (s) 1 (a = a ′ ) , (3.2) and u (0) empirical (a) = 0. This u (t)
empirical will then be used as the r (t) in (PL) for ranking. The regret metric will still be the one in (3.1). Due to space constraints, we defer the background and formalism of equilibrium computation (with ranking feedback) in the game-theoretic setting to Appendix B.
this section cite: ['b30', 'b50', 'b11', 'b50', 'b11', 'b38', 'b25', 'b3']

Section: HARDNESS RESULTS
In this section, we present hard instances to show that online learning in non-stochastic and potentially adversarial environments can be hard in general, under both InstUtil Rank and AvgUtil Rank, even when there are only two actions.
Theorem 4.1 in the following shows that for any temperature τ in (PL) no larger than a constant, there exists a sequence of utility vectors such that the expected regret is linear under InstUtil Rank, for both full-information and bandit feedback settings.
Theorem 4.1. Consider InstUtil Rank.
For any T > 0, temperature 0 < τ ≤ 0.1, and online learning algorithm, there exists a sequence of utilities u (t) T t=1 such that min E R (T ),external , E R (T ) ≥ Ω (T ) in both full-information and bandit feedback settings. The expectation is taken over the randomness of the algorithm and the ranking.
To prove Theorem 4.1, we need to construct two sequences of utility vectors, which yield the same ranking under InstUtil Rank in expectation. However, being no-regret in one of them will result in linear regret in the other. The detailed proof can be found in Appendix D.
The key challenge in achieving no-regret in the hard instance above is that the utility vectors
u (t) T
t=1 change arbitrarily fast, i.e., the accumulated variation grows linearly in time. Hence, to obtain positive results, we may need to restrict how fast they change over time, as quantitatively characterized by the following assumption.
Assumption 4.2 (Sublinear variation of utility vectors). The utility vectors u (t) T t=1 have a sublinear variation over time, i.e., for some q < 1,
P (T ) := T t=2 u (t) -u (t-1) ≤ O(T q ).
(4.1) Our result stated in Section 5 next will show that with Assumption 4.2, we can achieve sublinear regret, and thus close the gap. Moreover, note that in a game where the opponents all run common no-regret learning algorithms such as Follow-The-Regularized-Leader (cf. Definition L.2), Assumption 4.2 will be satisfied (cf. Lemma L.3).
Next, we show in Theorem 4.3 that when AvgUtil Rank is used, and τ is small enough, the minimal regret is still at least linear in T (up to logarithmic terms).
Theorem 4.3. Consider AvgUtil Rank with full-information feedback. For any T > 0, temperature 0 < τ ≤ O 1 T log T , and online learning algorithm, there exists T ′ ≥ T and a sequence of utilities u (t) T ′ t=1 such that E R (T ′ ),external ≥ Ω (T ′ ) . The expectation is taken over the randomness of the algorithm and the ranking.
To prove Theorem 4.3, we construct log T sequences of utility vectors that induce identical ranking feedback when τ is small. We then show that at least one of these sequences incurs average regret Ω(1). Given Theorem 4.3, it is impossible to achieve o(T ) with AvgUtil Rank when τ is very small. However, in Section 6, we will mitigate the gap by showing that when τ is a constant (i.e., O(1)), we can achieve sublinear regret with AvgUtil Rank, even without Assumption 4.2.
Due to the different instantiations of r (t) in the full-information and the bandit feedback settings under AvgUtil Rank, we have a separate hardness result for the latter, stronger than Theorem 4.3, as it allows a larger τ and avoids logarithmic terms. The result can be viewed as strengthening the hardness result for the adversarial bandit setting in Maran et al. (2024), which corresponds to the case in our model with τ → 0 + . Theorem 4.4. Consider AvgUtil Rank with bandit feedback. For any T > 0, temperature 0 < τ ≤ O 1 log T , and online learning algorithm, there exists a sequence of utilities u (t) 4T  t=1 such that E R (4T ) ≥ Ω (T ) . The expectation is taken over the randomness of the algorithm and the ranking.
To prove Theorem 4.4, we need to construct two utility sequences such that achieving sublinear regret in the first utility sequence will lead to insufficient exploration for the second sequence. As a result, when τ is small, those two sequences cannot be distinguished, and a linear regret must be incurred in one of them. Details of the proof can be found in Appendix D.
this section cite: []

Section: ONLINE LEARNING WITH InstUtil Rank FEEDBACK
We start by introducing a new utility estimation oracle to be used in our later algorithms.
this section cite: []

Section: UTILITY ESTIMATION
A natural approach to learning from ranking feedback is to use the observed permutations to estimate the underlying numeric utility vectors. At each timestep t, we form an estimate of the current utility vector u (t) using a sliding window of the most recent m rounds of observations: when t ≥ m, we use the past m permutations σ (s) t s=t-m+1 to construct an estimate u (t) . Due to the nonconvexity of (PL), the key step is to decompose a length-K ranking into pairwise comparisons. This reduction allows us to exploit the logistic structure of pairwise comparison probabilities and, via the monotonicity (and invertibility) of the logistic map, translate the estimation error of ranking probabilities back into an error bound on estimating the utilities. The full procedure is given in Algorithm 1, and we state its guarantee next.
Theorem 5.1. Consider InstUtil Rank and Algorithm 1. Suppose each action is proposed independently with probability at least p > 0 at each timestep t ∈ [T ] and let u (t) = Estimate σ (s) t s=t-m ′ +1 . Then, for any δ ∈ (0, 1) and t ≥ m ′ , when m ′ p 4 ≥ 2 log 2 δ , with probability at least 1δ, the estimate u (t) satisfies,
u (t) -u (t) ∞ ≤ τ e 1 τ + 1 2 p log 4|A| δ m ′ + t-1 s=t-m ′ +1 u (s+1) -u (s) ∞ .
Treating δ, p, and τ as constants, the accumulated estimation error
T t=1 u (t) -u (t)
∞ will be bounded by O T √ m ′ + m ′ P (T ) , which implies that a sublinear accumulated estimation error can be achievable when P (T ) is sublinear (Assumption 4.2). Moreover, Theorem 5.1 shows that the upper bound diverges as τ → 0 + . This aligns with intuition: when τ → 0 + , the highest-utility action is ranked first almost deterministically, and thus the feedback carries essentially no information about the utility gaps between actions. At the other extreme, when τ → +∞, the bound also diverges because the ranking is nearly sampled uniformly and hence largely independent of the underlying utility vector.
The full proof of Theorem 5.1 is deferred to Appendix E. Next, we show how to achieve sublinear regret in both full-information and bandit settings with InstUtil Rank, based on such an estimator.
this section cite: []

Section: SUBLINEAR REGRET WITH InstUtil Rank
This section shows that for any online learning algorithm that can achieve sublinear external regret with numeric utility feedback, we can construct an online learning algorithm with InstUtil Rank feedback based on it, in a black-box way.
With full-information feedback, the learning agent proposes the full action set A at each timestep. In this case, we can obtain u (t) , an estimate of u (t) , by Algorithm 1, and obtain guarantees using Theorem 5.1 with p = 1, since all actions are proposed at each timestep.
With bandit feedback, the utility of the learning agent at timestep t is 1 K K k=1 u (t) σ (t) (k) , i.e., the average utility of the proposed actions. To achieve sublinear R (T ) (as defined in (3.1)), each proposed action will be sampled from π (t) independently with replacement. In other words, an action may be proposed multiple times at a single timestep. To ensure sufficient exploration, namely, every action is proposed with positive probability, we impose a uniform lower bound π (t) (a) ≥ γ |A| for some γ > 0 and all a ∈ A. We enforce this by updating π (t+1) = (1-γ)Alg u (s) t s=1 +γ 1(A) |A| , i.e., a convex combination of the strategy generated by the no-regret learning algorithm Alg and a uniform probability distribution over A. A diagram of the algorithm can be found in Figure 10, and the details are tabulated in Algorithm 2. Then, we have the following theorem. Theorem 5.2. Consider InstUtil Rank with constant τ > 0. By running Algorithm 2, for any δ ∈ (0, 1), T > 0, and any full-information no-regret learning algorithm with numeric utility feedback, Alg, by choosing the window size m and γ properly, we have that with probability at least 1δ, the following hold:
R (T ),external ≤R (T ),external Alg, u (t) T t=1 + O P (T ) 1 3 T 2 3 log T δ 1 3 (Full-Info) R (T ) ≤R (T ),external Alg, u (t) T t=1 + O P (T ) 1 5 T 4 5 log T δ .
(Bandit)
The proof can be found in Appendices G and H. Theorem 5.2 implies that when P (T ) , the variation of utility vectors, is sublinear (cf. Assumption 4.2), the regret of Algorithm 2 will be sublinear.
6 ONLINE LEARNING WITH AvgUtil Rank FEEDBACK 6.1 UTILITY ESTIMATION Since σ (t) is generated based on u (t) avg := 1 t t s=1 u (s) , we will estimate u (t) avg instead. We will still apply Algorithm 1, which generates u (t) avg , an estimate of u (t) avg , when the permutation is sampled under AvgUtil Rank feedback. Moreover, notice that
u (t) avg -u (t-1) avg ∞ = u (t) + (t -1)u (t-1) avg t -u (t-1) avg ∞ ≤ 1 t u (t) ∞ + u (t-1) avg ∞ ≤ 2 t .
Thus,
t-1 s=t-m ′ +1 u (s+1) avg -u (s) avg ∞
, the counterpart of
t-1 s=t-m ′ +1 u (s+1) -u (s)
∞ in Theorem 5.1, can be bounded by t-1 s=t-m ′ +1 1 s+1 , irrelevant of the accumulated utility variation P (T ) .
this section cite: []

Section: FULL-INFORMATION SETTING
Unlike InstUtil Rank feedback, where we can plug the utility estimates into essentially any (fullinformation) no-regret learning oracle, the AvgUtil Rank setting requires an update rule that is stable with respect to the perturbations in the accumulated (averaged) utilities, such as Follow-The-Regularized-Leader (Hazan et al., 2016). The reason is that we would like the strategies produced from the estimated averages u (1) avg , . . . , u (t) avg to remain close to those that would have been produced from the ground-truth sequence u (1) avg , . . . , u (t) avg . With such stability, the regret of our learner can be controlled by the regret of this "ideal" learner, and hence remains sublinear up to the additional error induced by estimation. Assumption 6.1. The (full-information) online learning algorithm Alg satisfies the following condition: for any T > 0, t ∈ [T ], and sequences of utilities
u (s) t s=1 , u ′ (s) t s=1 ∈ R A t , we have Alg u (s) t s=1 -Alg u ′(s) t s=1 ≤ L t s=1 u (s) - t s=1 u ′(s) ,
where L = Θ (T -c ) for some constant c ∈ (0, 1).
It can be verified that FTRL with any strongly convex regularizer satisfies this assumption (cf. Lemma L.3). Then, similar to Section 5.2, any online learning algorithm satisfying Assumption 6.1 can achieve a sublinear regret when equipped with the utility estimator in Algorithm 1. The overall procedure is summarized in Algorithm 3 and Figure 11, with the following guarantee.
Theorem 6.2. Consider AvgUtil Rank with constant τ > 0 and full-information feedback. By running Algorithm 3, for any δ ∈ (0, 1), T > 0, and any full-information no-regret learning algorithm with numeric utility feedback, Alg, that satisfies Assumption 6.1, by choosing m properly, we have that with probability at least
1 -δ, R (T ),external satisfies R (T ),external ≤R (T ),external Alg, u (t) T t=1 + O LT 5 3 log T δ .
Theorem 6.2 implies that if we choose the stability parameter L = Θ (T -c ) with some c > 2/3, then the external regret R (
T ),external becomes sublinear in T . This choice of L can be realized by instantiating Alg as FTRL with suitable parameters (see Lemma L.3). Hence, the choice of L will also affect R (T ),external Alg, u (t) T t=1 . The formal statement of Theorem 6.2 and its proof can be found in Appendix I. 6.3 BANDIT SETTING By applying Algorithm 1, we can only obtain an estimate of u
(t) empirical (cf. (3.2)) instead of u (t)
avg , since it is the former that is used for ranking. However, almost all no-regret learning algorithms made decisions according to the accumulated utility, such as mirror descent (Hazan et al., 2016), FTRL, and regret matching (Zinkevich et al., 2007). Let n (t) (a) := t s=1 # o (s) (a) for any a ∈ A be the number of times action a has been proposed up to timestep t, where # o (s) (a) is the number of occurrences of a in the proposed choices o (s) at timestep s. A natural idea is to compute a). Nonetheless, the variance of this estimator will be too large due to the multiplication of n (t) (a) ∝ t.
n (t) (a)u (t) empirical (a) -n (t-1) (a)u (t-1) empirical (a) to get an estimate of u (t) (
To address this issue, we divide the timesteps {1, 2, . . . , t} into ⌈t/M ⌉ blocks, with each block containing M timesteps except for the last one.
Then, for each block {s • M + 1, s • M + 2, . . . , (s + 1)M } (for s ≤ t M -1) and a ∈ A, we estimate
1 M (s+1)M s ′ =s•M +1 u (s ′ ) (a) by computing u ((s+1)•M ) empirical (a)n ((s+1)•M ) (a) -u (s•M ) empirical (a)n (s•M ) (a) n ((s+1)•M ) (a) -n (s•M ) (a)
.
In this way, the coefficient multiplying u
((s+1)•M ) empirical (a) is now n ((s+1)•M ) (a) n ((s+1)•M ) (a)-n (s•M ) (a) ∝ t M .
The choice of M therefore induces a bias-variance trade-off: increasing M reduces the variance, but the resulting quantity estimates the average utility of action a conditional on a being selected in that block, which can differ substantially from the block-average utility of a. This discrepancy becomes more pronounced for large M , since more utility variation can accumulate within a longer block. The full algorithm is introduced in Algorithm 3 and Figure 11.
Theorem 6.3. Consider AvgUtil Rank with constant τ > 0 and bandit feedback. By running Algorithm 3, for any δ ∈ (0, 1), T > 0, and any full-information no-regret learning algorithm with numeric utility feedback, Alg, that satisfies Assumption 6.1, by choosing m, γ, and M properly, when P (T ) ≤ O (T q ) for some q < 1 3 and L = Θ (T -c ) with c ∈ 5 6 + q 2 , 1 , with probability at least
1 -δ, R (T ) satisfies R (T ) ≤R (T ),external Alg, u (t) T t=1 + O log 1 δ 2 L 1 3 T 23 18 P (T ) 1 6 ,
where O hides logarithmic dependence on T .
In Theorem 6.3, we require c < 1 because, for many algorithms, one typically has an externalregret bound of the form R (T ),external Alg, u (t) T t=1 ≤ O 1 L + LT . See, e.g., FTRL with any strongly convex regularizer as an example (cf. Lemma L. 3 and Hazan et al. (2016)). In particular, taking c = 1 would make the 1 L term linear in T .
this section cite: ['b21', 'b21', 'b54']

Section: EQUILIBRIUM COMPUTATION WITH RANKING FEEDBACK
For a normal-form game N,
{Ai} N i=1 , {Ui} N i=1 , the external regret of player i ∈ [N ] is defined as R (T ),external i := max πi∈∆ A i T t=1 u (t) i , π i -π (t) i , where π (t) i ∈ ∆ Ai is the strategy of player i at timestep t and u (t) i (a i ) = a ′ ∈× N j=1 Aj U i (a ′ ) 1 (a ′ i = a i ) j ′ ̸ =i π (t) j ′ (a ′ j ′ )
for any a i ∈ A i . Then, it is known that the time-average joint strategy π (T ) avg , where
π (T ) avg (a) := 1 T T t=1 i∈[N ] π (t) i (a i ) for any a ∈ × N i=1 A i , is an ϵ-CCE, with ϵ := max i∈[N ] 1 T R (T ),external i .
Applying the algorithm in Section 5 (for InstUtil Rank feedback) or Section 6 (for AvgUtil Rank feedback), we achieve sublinear R (T ),external i for each player i ∈ [N ]. Note that P (T ) in Assumption 4.2 can be bounded by the summation of all players' strategy variation (see Lemma K.1). Thus, to ensure P (T ) is sublinear in T , Alg needs to additionally satisfy the following assumption.
this section cite: []

Section: Assumption 7.1 (Sublinear variation of strategies). The (full-information) online learning algorithm
Alg needs to satisfy the following condition: for any T > 0, t ∈ [T -1], and sequence of utility
vectors u (s) t+1 s=1 ∈ [-1, 1] A t+1 , we have Alg u (s) t s=1 -Alg u (s) t+1 s=1
≤ η, where η = Θ (T -w ) for some constant w ∈ (0, 1). Theorem 7.2. Consider InstUtil Rank with constant τ > 0 and Algorithm 2. For any δ ∈ (0, 1), T > 0, and any full-information no-regret learning algorithm with numeric utility feedback, Alg, that satisfies Assumption 7.1, by choosing M, m, γ according to Theorem 5.2, we have that with probability at least 1δ, the algorithm finds an ϵ-CCE, with
ϵ ≤ max i∈[N ] 1 T R (T ),external i Alg, u (t) i T t=1 + O η 1 3 log T δ 1 3 (Full-Information) ϵ ≤ max i∈[N ] 1 T R (T ),external i Alg, u (t) i T t=1 + O η 1 5 log T δ .(Bandit)
With AvgUtil Rank feedback, when all the players apply Algorithm 3 and both Assumption 6.1 and Assumption 7.1 are satisfied by the oracle Alg being used, the external regret of each player will be sublinear in T according to Theorem 6.2. Finally, we have the statement below.
Theorem 7.3. Consider AvgUtil Rank with constant τ > 0 and Algorithm 3. For any δ ∈ (0, 1), T > 0, and any full-information no-regret learning algorithm with numeric utility feedback, Alg, that satisfies Assumption 6.1, by choosing M, m, γ according to Theorem 6.2, we have that with probability at least 1δ, the algorithm finds an ϵ-CCE under full-information feedback, with
ϵ ≤ max i∈[N ] 1 T R (T ),external i Alg, u (t) i T t=1 + O LT 5 3 log T δ .
(Full-Information)
When M , m, and γ are chosen as in Theorem 6.3, and both Assumption 6.1 and Assumption 7.1 hold, we have that with probability at least 1δ, the algorithm finds an ϵ-CCE under bandit feedback, with
ϵ ≤ max i∈[N ] 1 T R (T ),external i Alg, u (t) i T t=1 + O log 1 δ 2 L 1 3 η 1 6 + L 1 2 T 4 9 . (Bandit)
Lastly, we would like to remark that although the online learning setting can be hard with a small τ (cf. the hardness results in Theorem 4.3 and Theorem 4.4), computing an equilibrium is still possible even when τ → 0 + . A detailed discussion can be found in Remark K.3.
this section cite: []

Section: EXPERIMENTS
To demonstrate the practical merit of online learning with ranking feedback, we consider the following scenario as an application example, which we term as online large-language-model routing.
A service provider hosts multiple language models, each possessing distinct strengths. For instance, GPT-4o (Hurst et al., 2024) may excel in creative writing, whereas GPT-5 specializes in coding. Consequently, users may exhibit different preferences for different models.
We cast personalized large language model (LLM) routing as an online learning problem with ranking feedback. Fix a particular user and view each candidate model as an action. At each timestep t, the user submits a query, and the server samples K candidate responses by drawing models according to a mixed strategy π (t) and querying the selected models on that prompt. The user then returns a ranking over the proposed responses, providing only relative-preference feedback rather than calibrated scores. This feedback is induced by an underlying, time-varying utility vector u (t) that captures the user's instantaneous (or time-aggregated) satisfaction with each model. The server's objective is to update π (t) online to minimize external regret with respect to the best fixed model in hindsight, while remaining responsive to nonstationary preferences. For instance, a user may prioritize creative writing for a period and later shift toward coding or mathematical problem solving.
We simulate this process in Figure 2. Specifically, we generate responses for the HH-RLHF (Bai et al., 2022) dataset using Qwen3-32B (Yang et al., 2025), Phi-4 (Abdin et al., 2024), GPT-4o (Hurst et al., 2024), and Llama-3.1-70B (Dubey et al., 2024). We evaluate the responses using a reward modelfoot_0 and execute Algorithm 3 with AvgUtil Rank under bandit feedback. At each time step, a prompt is sampled from the dataset along with responses from different models, and the resulting scores form the reward vector. As shown in Figure 2, the average regret decreases over time, suggesting that our router quickly approaches the performance of always serving the best fixed model in hindsight, i.e., the model with the highest cumulative reward under the user's preferences. More experiments about online learning and equilibrium computation can be found in Appendix C.
this section cite: ['b23', 'b2', 'b49', 'b23', 'b13']

Section: A RELATED WORK
Dueling Bandits. Using comparison and/or ranking feedback for sequential decision-making has mostly been studied under the framework of dueling bandits (Yue et al., 2012;Saha & Gaillard, 2022;Saha & Gopalan, 2019;Du et al., 2020;Saha et al., 2021;Dudík et al., 2015), where the agent takes two (or multiple) actions at each timestep, and receives a ranking of them as feedback.
Different from our setting, the ranking feedback in these works was only based on the instantaneous utility at that timestep, while our results can address both settings with instantaneous and timeaverage utilities for ranking. More importantly, the regret notions studied in these works were particularly designed for the dueling-bandit setting, and thus different from the classical external regret we focus on here. Finally, dueling bandits mostly focused on environments that are stationary and stochastic (Yue et al., 2012;Saha & Gaillard, 2022;Saha & Gopalan, 2019;Du et al., 2020), while we focus on the non-stochastic setting where the environment is arbitrary and potentially adversarial, as in online learning (Shalev-Shwartz et al., 2012;Hazan et al., 2016). Due to the last two differences, the implication of these dueling-bandit algorithms in the game-theoretic setting is unclear, while our algorithms find an approximate CCE of the game, as a corollary of the no-(external-)regret guarantee.
Reinforcement Learning from Human Feedback (RLHF) and Preference-Based RL. Inspired by the successes in aligning large language models (LLMs) (Ouyang et al., 2022), reinforcement learning from human feedback has received increasing attention. RLHF is usually instantiated as preference-based learning, where humans rank the model outputs based on their preferences, and a reward model is then estimated from the feedback, which will be further used for model finetuning. This way, RLHF is oftentimes implemented in an offline fashion, where batch feedback data are used for reward model estimation (Ziegler et al., 2019;Bai et al., 2022;Ouyang et al., 2022;Zhu et al., 2023;Park et al., 2024). Recently, online versions of RLHF have also been developed (Dwaracherla et al., 2024;Du et al., 2024;Xie et al., 2025;Cen et al., 2025;Zhang et al., 2025), where the exploration issue was addressed with online feedback. In fact, beyond fine-tuning LLMs, preference-based RL has also been studied in the classical Markov decision process model with online feedback (Novoseller et al., 2020;Saha et al., 2023;Xu et al., 2020). However, the utility/reward functions in these works are again stationary, and the regret notions extend those in the dueling-bandits literature, which are thus different from ours. Hence, these results do not apply to our adversarial online learning and game-theoretic settings.
Learning of Stable Matchings. Some of our motivating scenarios for the game-theoretic setting may also be modeled as the stable matching problem (Gale & Shapley, 1962), which has been extensively studied when the agents have full knowledge of their preferences. Recently, growing efforts have been devoted to learning in stable matching markets with unknown preferences, and through interactions between the agents (Liu et al., 2020;2021;Basu et al., 2021;Jagadeesan et al., 2021;Etesami & Srikant, 2025;Shah et al., 2024b;a) 2024) also provided a hardness result for the adversarial bandit setting (with τ → 0 + in our framework), while our hardness results (with different hard instances) are stronger in the sense that they allow a wider range of τ for the bandit setting, and also cover the full-information setting (cf. Table 1).
this section cite: ['b50', 'b38', 'b39', 'b11', 'b14', 'b50', 'b38', 'b39', 'b11', 'b45', 'b21', 'b53', 'b2', 'b35', 'b51', 'b33', 'b48', 'b19', 'b26']

Section: B ADDITIONAL NOTATION AND PRELIMINARIES
Notation. For any integer N > 0, we define [N ] := {1, ..., N } to denote the set of positive integers no larger than N . We use bold notation x to denote a finite-dimensional vector, and x i to denote the i th element of the vector. For any discrete set S, let |S| denote its cardinality, ∆ S := x ∈ R S : s∈S x s = 1, x s ≥ 0 for all s ∈ S be the probability simplex over S, and 1 (S) be an all-one vector with each index being elements in S. For any ordered discrete set S, we use R S to denote the |S| dimensional real space, where the s th ∈ S element of any x ∈ R S is denoted as x s or x(s). For any vector x ∈ R m , let ∥x∥ p be its L p -norm and we use ∥x∥ to denote the L 2 -norm by default. For any convex compact set C ⊆ R n and x ∈ R n , let Proj C (x) = argmin x ′ ∈C ∥xx ′ ∥. For any event e, let 1 (e) be its indicator, which is equal to one when e holds and zero otherwise. Additionally, for any discrete set S, let Σ (S) be the set containing all the permutations of the elements in S. We will use sig(x) := exp(x)  1+exp(x) : R → R to denote the logistic function.
this section cite: []

Section: B.1 NORMAL-FORM GAMES
An N -player normal-form game can be characterized by a tuple N,
{A i } N i=1 , {U i } N i=1
, where
A i := a 1 i , a 2 i , . . . , a |Ai| i is the (finite) action set for player i ∈ [N ]; U i : × N i=1 A i → [-1, 1] ( ×
denotes the Cartesian product of sets) is the utility function of player i, where U i (a 1 , a 2 , ..., a N ) is the utility of player i when player j ∈ [N ] takes action a j . We call a : = (a 1 , a 2 , ..., a N ) the joint action and let a -i : = (a 1 , ..., a i-1 , a i+1 , ..., a N ). Player i ∈ [N ] can choose a strategy π i ∈ ∆ Ai , and we call × N i=1 ∆ Ai ∋ π = (π 1 , π 2 , . . . , π N ) a strategy profile. When a strategy profile π is implemented, each player i ∈ [N ] has an expected utility of a∈× N j=1 Aj U i (a) j∈[N ] π j (a j ). Lastly, we use the unbold notation π ∈ ∆ × N i=1 Ai to denote the (possibly correlated) joint strategy of all the players, where π(a) is the probability of choosing the joint action a ∈ × N i=1 A i . In this paper, we focus on finding an ϵ-approximate coarse correlated equilibrium (ϵ-CCE) of the NFG, which is a probability distribution over the joint action set. It is formally defined as follows:
Definition B.1 (ϵ-CCE). For any joint strategy π ∈ ∆ × N i=1 Ai , it is an ϵ-CCE for ϵ ≥ 0 if max i∈[N ] max πi∈∆ A i a∈× N j=1 Aj U i (a)   π i (a i ) a ′ i ∈Ai π(a ′ i , a -i ) -π(a)   ≤ ϵ. (ϵ-CCE)
When ϵ = 0, we refer to it as a(n exact) CCE. We also refer to ϵ as the exploitability.
this section cite: []

Section: B.2 EQUILIBRIUM COMPUTATION WITH RANKING FEEDBACK
There is a mediator (platform) in the game that computes strategies for the players, (e.g., Uber recommends the candidate drivers and users to each other), but with only access to the ranking feedback from the players, e.g., humans. Specifically, when the strategy profile π is implemented by the players, player i's utility of taking action
a i ∈ A i is u π i (a i ) := a ′ ∈× N j=1 Aj U i (a ′ ) 1 (a ′ i = a i ) j̸ =i π j (a ′ j ).
However, instead of observing the utility directly, the mediator can only observe the ranking based on it. Therefore, at each timestep t, the mediator will choose a strategy profile π and propose each player i ∈
[N ] a multiset o (t) i = a (t),k i K k=1
consisting of K actions, and in different settings proceed differently as follows:
• Full-information setting. All the actions of each player i ∈ [N ] can be evaluated and ranked at each timestep t based on some utility vector, which is u π
(t) i under InstUtil Rank and u (t) avg := 1 t t s=1 u π (s) i under AvgUtil Rank, where π (t) = π (t) 1 , . . . , π (t) N
is the strategy profile at timestep t.
• Bandit setting. For each player i ∈ [N ], only the K actions in o (t) i that are proposed at timestep t will be evaluated and ranked, with the associated elements in some utility vector. Specifically, under InstUtil Rank, u (t)
i defined below will be used: for each a
i ∈ A i u (t) i (a i ) := 1 |o (t) -i | a ′ -i ∈o (t) -i U i (a i , a ′ -i );
under AvgUtil Rank, the corresponding empirical average utility is as computed in (3.2), with the u (s) therein being replaced by the u (s) i above. As in the online setting, we assume that the actions are proposed in an unbiased way, i.e., E
a -i ∈o (t) -i Ui(ai,a-i) |o (t) -i | = U i (a i , •), π (t) -i , for all a i ∈ A i . In other words, u (t) i is an unbiased estimate of u π (t) i .
The process will be repeated until the mediator finds an (approximate) equilibrium of the game, which is the average of the joint strategy over all timesteps.
this section cite: []

Section: C ADDITIONAL EXPERIMENTS
To assess robustness across learning setups, we study both full-information and bandit feedback under the two ranking-feedback models in InstUtil Rank and AvgUtil Rank. The corresponding regret results are reported in Figures 3 to 6. We also evaluate our methods on randomly generated two-player general-sum games under the same feedback models. For these game experiments, we report the resulting CCE exploitability ϵ across different game parameters in Figures 7 to 9, together with 95% confidence intervals.
We consider both full-information and bandit feedback. Throughout, all experiments run for T = 10 7 iterations, and each player has 10 actions. We evaluate performance across the number of proposed actions K ∈ {3, 5, 10}, the temperature parameter τ ∈ {0.5, 1, 2} in (PL), and the accumulated utility variation P (T ) = T q with q ∈ {0.3, 0.5, 0.7}. For AvgUtil Rank under fullinformation feedback, we consider only the case q = 1.0, which does not restrict the utility variation, since the corresponding guarantee does not rely on Assumption 4.2. Moreover, for AvgUtil Rank under bandit feedback, although 0.5 and 0.7 exceed the theoretical threshold 1 3 required for Algorithm 3, we empirically observe that Algorithm 3 still attains sublinear regret. This suggests that Algorithm 3 is more robust in practice than the current theory indicates.
For hyper-parameter selection, we conduct a separate grid search for each experimental setting. Under InstUtil Rank, we tune the estimation window size m and the exploration rate γ; under AvgUtil Rank, we additionally tune the block size M . Specifically, we search over m ∈ 5 × 10 4 , 10 5 , 1.5 × 10 5 , γ ∈ {0.1, 0.05, 0.01}, and, when applicable, M ∈ 3 × 10 6 , 5 × 10 6 , 10 7 . The learning rate is set to η = 1/ √ T in all experiments except for bandit feedback under AvgUtil Rank, where we use η = 10 -6 . Each hyper-parameter configuration is evaluated over 10 random seeds, namely {0, 1, . . . , 9}. For each figure, we report the best-performing choice of m, M , and γ, namely, the one that achieves the smallest average regret or ϵ over all seeds at horizon T .
Utility estimation is performed using Algorithm 1. As the full-information no-regret learning oracle with numeric feedback, we instantiate Alg as PGD for InstUtil Rank and as FTRL with L 2regularization for AvgUtil Rank.
In Figures 3 to 5, we report the performance of the learning algorithm for InstUtil Rank under both full-information and bandit feedback, and for AvgUtil Rank under bandit feedback. To generate utility sequences with cumulative variation bounded by T q , we first allocate the total variation budget T q uniformly at random across timesteps. At each timestep t, we then sample a perturbation direction n (t) uniformly at random and use binary search to choose a scaling factor α (t) such that -1) equals the variation assigned to round t.
Proj [-1,1] A u (t-1) + α (t) n (t) -u (t
For AvgUtil Rank under full-information feedback, our algorithm can accommodate sequences of utility vectors with unbounded cumulative variation. We therefore additionally conduct an ablation study on the noise model in Figure 6. We first sample an initial utility vector, and at each timestep the utility vector is defined as the initial utility vector plus an independent random shift. For each coordinate, the shift is drawn from one of three noise distributions: Uniform (-σ, σ), N 0, σ 2 , or Γ 1/σ 2 , σ 2 with σ = 0.3. We report the resulting average regret over time in Figure 6. Across all game settings, the exploitability decreases as t increases, suggesting that the timeaveraged joint strategy converges to a CCE.
this section cite: []

Section: D PROOF OF SECTION 4
In this section, we will show the hardness results in Section 4.
this section cite: []

Section: D.1 PROOF OF THEOREM 4.1
Theorem 4.1. Consider InstUtil Rank. For any T > 0, temperature 0 < τ ≤ 0.1, and online learning algorithm, there exists a sequence of utilities u (t) T t=1 such that min E R (T ),external , E R (T ) ≥ Ω (T ) in both full-information and bandit feedback settings. The expectation is taken over the randomness of the algorithm and the ranking.
Proof. Consider an online learning problem with A = {a, b}, so that the utility vector can be represented as (u(a), u(b)). There are two instances with τ = 0.1 and K = 2 under bandit feedback.
In the first instance, there are two types of utility vectors (-0.5, 0) and (0.15, 0). At each timestep, the adversary will choose (-0.5, 0) with probability 4 13 and the other with probability 9 13 . In the second instance, there are two types of utility vectors (-0.02, 0) and (0.1, 0). Recall sig(x) : R → R := exp(x)  1+exp(x) is the logistic function. At each timestep, the adversary will choose
0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 -0.05 0.00 0.05 0.10 0.15 Regret InstUtil Rank with Bandit Feedback (K = 5, q = 0.5) τ =0.5 τ =1 τ =2 1 0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 -0.05 0.00 0.05 0.10 0.15 0.20 Regret InstUtil Rank with Bandit Feedback (K = 5, τ = 1) q =0.3 q =0.5 q =0.7 1 0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 -0.050 -0.025 0.000 0.025 0.050 0.075 0.100 0.125 0.150 Regret InstUtil Rank with Bandit Feedback (τ = 1, q = 0.5) K =3 K =5 K =10 1 Figure 4: The regret for bandit feedback setting under InstUtil Rank feedback in the online learning setting. The performance is evaluated across different temperatures τ , cumulative utility variations P (T ) = T q , and numbers of proposed actions K. Each parameter combination is tested 10 times with different random seeds. 0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 0.0 0.2 0.4 0.6 0.8 1.0 Regret AvgUtil Rank with Bandit Feedback (K = 5, q = 0.5) τ =0.5 τ =1 τ =2 1 0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 0.0 0.2 0.4 0.6 0.8 1.0 Regret AvgUtil Rank with Bandit Feedback (K = 5, τ = 1) q =0.3 q =0.5 q =0.7 1 0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 0.0 0.2 0.4 0.6 0.8 1.0 Regret AvgUtil Rank with Bandit Feedback (τ = 1, q = 0.5) K =3 K =5 K =10 1 Figure 5: The regret for bandit feedback setting under AvgUtil Rank feedback in the online learning setting. The performance is evaluated across different temperatures τ , cumulative utility variations P (T ) = T q , and numbers of proposed actions K. Each parameter combination is tested 10 times with different random seeds. 0.0 0.2 0.4 0.6 0.8 1.0 t ×10 7 0.02 0.04 0.06 0.08 Average Regret Uniform τ = 0.5 τ = 1.0 τ = 2.0 0.0 0.2 0.4 0.6 0.8 1.0 t ×10 7 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 Gaussian 0.0 0.2 0.4 0.6 0.8 1.0 t ×10 7 0.00 0.01 0.02 0.03 0.04 0.05 0.06 Gamma Average Regret under Different Noise Types 1 Figure 6: Regret of Algorithm 3 with AvgUtil Rank under full-information feedback, across different temperatures τ and noise types in the online-learning setting. 0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 0.00 0.02 0.04 0.06 0.08 0.10 0.12 CCE Approximation: ϵ InstUtil Rank Game with Full-information Feedback (K = 10) τ =0.5 τ =1 τ =2 1 0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 0.00 0.05 0.10 0.15 0.20 0.25 0.30 CCE Approximation: ϵ AvgUtil Rank Game with Full-information Feedback (K = 10) τ =0.5 τ =1 τ =2 1 Figure 7: The exploitability for the full-information setting under both InstUtil Rank and AvgUtil Rank feedback in the game-play setting. The performance is tested under different temperatures τ . Each parameter combination is tested 10 times with different random seeds. (-0.02, 0) with probability 4sig(-5)/13+9sig(1.5)/13-sig(1) sig(-0.2)-sig(1) ≈ 0.58 and the other with probability 1 -4sig(-5)/13+9sig(1.5)/13-sig(1) sig(-0.2)-sig(1)
.
The expected utility of action b in both instances is 0. The expected utility of action a in the first instance is -0.05. The expected utility of action a in the second instance is 0.03.
0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 -0.05 0.00 0.05 0.10 0.15 0.20 0.25 0.30 CCE Approximation: ϵ InstUtil Rank Game with Bandit Feedback (K = 5) τ =0.5 τ =1 τ =2 1 0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 -0.05 0.00 0.05 0.10 0.15 0.20 0.25 0.30 CCE Approximation: ϵ InstUtil Rank Game with Bandit Feedback (τ = 1) K =3 K =5 K =10 1 Figure 8: The exploitability for the bandit feedback setting under InstUtil Rank feedback in the game-play setting. Performance is evaluated across different temperatures τ and numbers of proposed actions K. Each parameter combination is tested 10 times with different random seeds. 0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 CCE Approximation: ϵ AvgUtil Rank Game with Bandit Feedback (K = 5) τ =0.5 τ =1 τ =2 1 0.0 0.2 0.4 0.6 0.8 1.0 Timestep ×10 7 0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 CCE Approximation: ϵ AvgUtil Rank Game with Bandit Feedback (τ = 1) K =3 K =5 K =10 1 Figure 9: The exploitability for the bandit feedback setting under AvgUtil Rank feedback in the game-play setting. Performance is evaluated across different temperatures τ and numbers of proposed actions K. Each parameter combination is tested 10 times with different random seeds. Moreover, at any timestep t when both actions a, b are proposed in o (t) , in the ranking σ (t) , the probability of a being preferred over b is Pr a σ (t)
> b = 4 13 sig(-5) + 9 13 sig(1.5).
This coincides with the corresponding probability under the second instance, 4sig(-5)/13 + 9sig(1.5)/13 -sig(1) sig(-0.2) -sig(1) sig(-0.2) + 1 -4sig(-5)/13 + 9sig(1.5)/13 -sig(1) sig(-0.2) -sig(1) sig(1).
Hence, the distribution of the ranking σ (t) is identical in the two instances. When o (t) = {a, b}, the preceding equality shows that Pr a
σ (t)
> b matches across instances. When o (t) = {a, a} (or {b, b}), the ranking is deterministic, so it is trivially the same in both instances.
Therefore, under full-information feedback, for any algorithm that generates π (t) T t=1 , we have t) .
E T t=1 u (t) , π (t) = T t=1 E u (t) , π (t) (i) = T t=1 E u (t) , E π (
The equality (i) holds because u (t) is independent of π (t) given our process of generating both instances. Similarly, under bandit feedback, t) .
E   T t=1 1 K a∈o (t) u (t) (a)   = T t=1 E   1 K a∈o (t) u (t) (a)   (i) = T t=1 E u (t) , E π (
In (i), we define π (t) (a) := 1 K a ′ ∈o (t) 1 (a = a ′ ) for any a ∈ A. Then, for both the full-information feedback and the bandit feedback, we have
E π (t) = σ (1) ,...,σ (t-1) ∈Σ(A) P σ (1) , . . . , σ (t-1) E π (t) | σ (1) , . . . , σ (t-1) .
The first term P σ (1) , . . . , σ (t-1) is equal in the two instances according to the discussion above, and the second term E π (t) | σ (1) , . . . , σ (t-1) is also equal since it only depends on the algorithm. Therefore, E π (t) is the same in both instances.
However, E u (t) = (-0.05, 0) in the first instance but (0.03, 0) in the second. Therefore, whenever achieving sublinear regret in the first instance, the algorithm will suffer a linear regret in the second instance, and vice versa.
this section cite: []

Section: D.2 PROOF OF THEOREM 4.3
Theorem 4.3. Consider AvgUtil Rank with full-information feedback. For any T > 0, temperature 0 < τ ≤ O 1 T log T , and online learning algorithm, there exists T ′ ≥ T and a sequence of utilities
u (t) T ′ t=1 such that E R (T ′ ),external ≥ Ω (T ′ ) .
The expectation is taken over the randomness of the algorithm and the ranking.
Proof. We use (u(a), u(b)) to denote the utility vector when the action set is A = {a, b}. In the following, we will show a hard instance for τ → 0 + , i.e., we always observe the action with higher utility ranks first in the permutation. Then, we will show that τ ≤ O( 1 T log T ) can be reduced to τ → 0 + .
The utility vector at timestep 1 is u (1) = (0.5, 0). We will construct the rest of the utility vectors next.
We call the following an action-a construction, since, except for the last timestep, the observation is that action a is always better. Let K ∈ N be the smallest integer such that 2 K ≥ T , then: Sequence 0 =(0, 1), (0, 0) Sequence 1 =(1, 0), (0, 1), (0, 1), (0, 0) Sequence 2 =(1, 0), (1, 0), (1, 0), (0, 1), (0, 1), (0, 1), (0, 1), (0, 0) . . .
(D.1) Sequence K -1 = (1, 0), ..., (1, 0) 2 K-1 -1 , (0, 1), ..., (0, 1) 2 K-1 , (0, 0) Sequence K = (1, 0), ..., (1, 0) 2 K -1
, (0, 0).
Lemma D.1 in the following shows that at least one of the sequences will incur a low average utility for the algorithm.
Lemma D.1. Consider (D.1). For any online learning algorithm, at least one of the K +1 sequences satisfies that the expected average utility per timestep is less than 0.5 -1 2(K+1) .
By Lemma D.1, there exists a sequence with length 2 k for some k ≤ K such that the average utility per timestep achieved by the algorithm is less than 0.5 -1 2(K+1) . We will pick this sequence as the next 2 k utility vectors. If the current utility vector sequence is no less than T , then the hard instance is completed. Otherwise, we will establish the following action-b construction:
Sequence 0 =(1, 0), (0, 0) Sequence 1 =(0, 1), (1, 0), (1, 0), (0, 0) Sequence 2 =(0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0) . . . Sequence K -1 = (0, 1), ..., (0, 1)
2 K-1 -1 , (1, 0), ..., (1, 0) 2 K-1 , (0, 0) Sequence K = (0, 1), ..., (0, 1) 2 K -1 , (0, 0).
Similarly, except for the last observation, action b is the best action in all the observations. Similar to Lemma D.1, we can show that at least one of the sequences incurs average utility per timestep less than 0.5 -1 2(K+1) . We will add that sequence to the end of our hard instance. Let T ′ ≥ T be the length of the final instance. Therefore, the average regret will be at least
1 2(K+1) - 1 T ≥ Ω( 1 log T ).
Because of the construction, the best action should get at least 0.5 -1 T utility per timestep.
When τ ≤ 1 4T log T , from the construction above, the difference between the cumulative utility of the actions is always 0.5. By the definition in AvgUtil Rank, at any given timestep the probability that an action with lower average utility is preferred over an action with higher average utility is 1sig 0.5 T τ ≤ O 1 T 2 . Applying a union bound over all timesteps, it follows that with probability at least 1 -O 1
T , no such misranking occurs at any timestep. Hence, every permutation ranks the higher-utility action first throughout. In other words, with high probability, the algorithm incurs linear regret.
Lemma D.1. Consider (D.1). For any online learning algorithm, at least one of the K +1 sequences satisfies that the expected average utility per timestep is less than 0.5 -1 2(K+1) .
Proof. Note that in this online learning setting, the strategy π (t) is determined by u (1) , . . . , u (t-1) . Therefore, in all the sequences in the action-a construction, since action a is the best in all the observations, for any two sequences k 1 ≤ k 2 , the expectation of the strategies is the same for the first 2 k1+1 utility vectors. For simplicity, we will use x (t) to denote the probability of choosing action-a at timestep t.
The average utility at sequence 0 is 1-x (1) 2 . The average utility at sequence 1 is
x (1) 4 + 1-x (2) 4 + 1-x (3) 4
. We can see that the utility contributed by x (1) to all the sequences is
1 -x (1) 2 + x (1) 4 + x (1) 8 + ... + x (1) 2 K + x (1) 2 K = 1 2 .
Similarly, the contribution of x (2) , x (3) is 1 4 . The contribution of x (4) , x (5) , . . . , x (7) is 1 8 . Therefore, the total contribution of x (1) , ..., x (2 K -1) is K 2 . There are K + 1 sequences in total, so that at least one of the sequences has average utility per timestep less than K 2(K+1) = 1 2 -1 2K+2 .
this section cite: []

Section: D.3 PROOF OF THEOREM 4.4
Theorem 4.4. Consider AvgUtil Rank with bandit feedback. For any T > 0, temperature 0 < τ ≤ O 1 log T , and online learning algorithm, there exists a sequence of utilities u (t) 4T t=1 such that E R (4T ) ≥ Ω (T ) . The expectation is taken over the randomness of the algorithm and the ranking. We call the first T timesteps as the first phase, the next T timesteps as the second phase, and the last 2T timesteps as the third phase.
For any online learning algorithm to achieve sublinear expected external regret, it must propose action a for at least 0.9T timesteps during the first phase with probability at least 1 2 , since otherwise the expected external regret in the first phase is linear. With probability at least 1 2 , after proposing at least 0.9T timesteps of a in the first phase, the online learner must propose action b for at least 0.2T -0.1T -0.01T 0.2 = 0.45T timesteps during the second phase to achieve sublinear regret. Then, at the end of the second phase, with probability at least 1 4 , u
empirical (b)-u (2T ) empirical (a) ≥ 0.45T •0.2 0.1T +0.45T - 0.1 = 7(2T )
110 . Intuitively, during the third phase, achieving sublinear regret on Instance 1 requires that action a not be proposed too many times. However, if the learner proposes action a too infrequently, it may fail to sample action a sufficiently often to distinguish Instance 1 from Instance 2, which would then lead to linear regret.
Formally, during the third phase of Instance 1, the algorithm needs to propose b for at least 0.2T +2T -0.2T -0.1T -0.01T 1 = 1.89T timesteps with probability at least 1 2 after proposing at least 0.9T a in the first phase and 0.45T b in the second phase. In other words, a is proposed by no more than 0.11T times. Then, in Instance 2, throughout the third phase
u (t) empirical (b) -u (t) empirical (a) (i) ≥u (2T +0.11T ) empirical (b) -u (2T +0.11T ) empirical
(a) ≥ 0.45T • 0.2 0.1T + 0.45T -0.9T • 0.1 + 0.11T • 0.4 0.9T + 0.11T ≥ 0.03.
(i) follows because, among all ways of placing the 0.11T exploration proposals of action a within the third phase, assigning them to the first 0.11T timesteps minimizes the gap between the empirical average utility of actions b and a.
Therefore, when τ → 0 + , the observations of Instance 1 and Instance 2 are the same with probability at least 1 8 . Then, with probability at least 1 8 , according to the discussion above, any learning algorithm will satisfy one of the following:
• Linear regret at timestep T ;
• Linear regret at timestep 2T ;
• Linear regret at timestep 4T in either Instance 1 or Instance 2.
When τ ≤ 200 3 log T , at each timestep the action with the larger empirical average utility is ranked first with probability at least 1-O T -2 . Taking a union bound over all T timesteps, we obtain that this action is ranked first at every timestep with probability at least 1 -O T -1 . Therefore, it suffices to consider the limiting case τ → 0 + , which completes the proof.
this section cite: []

Section: E PROOF OF THEOREM 5.1
In this section, we proved the high probability bound for the utility estimation error, and with that, we gave the regret upper bound of our algorithm under InstUtil Rank feedback. Next, we will introduce the key lemma we used for utility estimation, Lemma E.1, which shows that the ranking of K actions can be decomposed into pair-wise rankings.
this section cite: []

Section: Algorithm 1 Utility Estimation with Action Permutations: Estimate σ
(s) m ′ s=1 1: Input: A set consisting of m ′ permutations of actions : σ (s) m ′ s=1 with |σ (s) | = K for all s ∈ [m ′ ], and temperature τ > 0. 2: for j = 1, 2, . . . , |A| -1 do 3: for s = 1, . . . , m ′ do 4: Calculate n (s) j,1 , n (s) j,2 defined as n (s) j,1 := i,k∈[K] 1 σ (s) (i) = a j , σ (s) (k) = a |A| and i < k , n (s) j,2 := i,k∈[K] 1 σ (s) (i) = a j , σ (s) (k) = a |A| and i > k . 5: end for 6: Let T j := s ∈ [m ′ ] : n (s) j,1 + n (s) j,2 > 0 7: Let sig -1 (x) : (0, 1) → R := log x
1-x be the inverse function of sig(•). The utility of action a j is then estimated as
u(a j ) =    Proj [-1,1] τ sig -1 1 |Tj| • s∈T j n (s) j,1 n (s) j,1 +n (s) j,2 |Tj| > 0 0 |Tj| = 0. 8: end for 9: Return u = u(a 1 ), u(a 2 ), . . . , u a |A|-1 , 0 E.1 PAIR-WISE UTILITY ESTIMATION
Lemma Lemma E.1 shows that, when the number of proposed actions is K > 2, for any two distinct actions a ̸ = b ∈ o (t) , the expected fraction of (a, b)-pairs (i.e., pairs formed by choosing one occurrence of a and one occurrence of b in the multiset
o (t) ) for which a is ranked ahead of b in σ (t) equals sig u (t) (a) -u (t) (b) τ .
Equivalently, the induced pairwise marginal is the same as in the K = 2 case: it matches the probability of observing the permutation (a, b) when only a and b are proposed. Related pairwisemarginal identities appear in Hunter (2004, p. 396); Lemma Lemma E.1 extends them to multisets (allowing repeated actions).
Lemma E.1. Let # S (a) := a ′ ∈S 1 (a ′ = a)
represent the number of elements in a multiset S that are equal to a ∈ A. For any utility vector u, temperature τ > 0, a multiset of proposed actions S with cardinality |S| = K, and any two actions a ̸ = b ∈ S, we have
1 # S (a) • # S (b) E σ K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) S = sig u(a) -u(b) τ .
The expectation is taken over the distribution of the permutation σ under the ranking model (PL).
The proof can be found in Appendix E.2. With Lemma E.1, the general cases where K > 2 actions are proposed can be cast into the case with only two actions being proposed, by enumerating all possible action pairs. Therefore, to estimate u (t) (a j ) for some a j ∈ A, we will first construct an
unbiased estimator of sig u (s) (a j )-u (s) (a |A| ) τ using Lemma E.1, for all timesteps s ∈ [t -m + 1, t]
when both a j , a |A| ∈ o (s) . Since we have assumed without loss of generality that u (s) (a |A| ) = 0, these values coincide with sig u (s) (a j ) τ . Then, by Hoeffding's inequality and the monotonicity of the logistic function sig(•), with high probability, the mean of the logistic function estimators will be bounded between the minimum and maximum of sig u (s) (a j ) τ t s=t-m+1
. By Assumption 4.2, since the utility vectors are changing slowly, that mean can be shown close to sig
u (t) (a j ) τ . With a good estimate of sig u (t) (a j ) τ
, we can then take an inverse of sig(•) to estimate u (t) (a j ). This estimation algorithm is summarized in Algorithm 1 and analyzed in Theorem 5.1 below.
In the following, we will prove Theorem 5.1, which gives the estimation error bound of the utility vector for each timestep.
Theorem 5.1. Consider InstUtil Rank and Algorithm 1. Suppose each action is proposed independently with probability at least p > 0 at each timestep t ∈ [T ] and let u (t) = Estimate σ (s) t s=t-m ′ +1 . Then, for any δ ∈ (0, 1) and t ≥ m ′ , when m ′ p 4 ≥ 2 log 2 δ , with probability at least 1δ, the estimate u (t) satisfies,
u (t) -u (t) ∞ ≤ τ e 1 τ + 1 2 p log 4|A| δ m ′ + t-1 s=t-m ′ +1 u (s+1) -u (s) ∞ .
Proof. Due to the symmetry of timesteps, we will only prove Theorem 5.1 for t = m ′ for notational simplicity.
For any j ∈ [|A| -1], we assume that the probability for action a j being chosen at each timestep is at least p. Let m 1 denote the number of timesteps (out of m ′ ) in which both a j and a |A| occur in the proposed actions. By Hoeffding's Inequality, we have that with probability at least 1 -δ 2 :
m 1 ≥ E [m 1 ] - m ′ 2 log 2 δ ≥ m ′ p 2 - m ′ 2 log 2 δ .
We define
n (s) j,1 := i,k∈[K] 1 σ (s) (i) = a j , σ (s) (k) = a |A| and i < k , n (s) j,2 := i,k∈[K] 1 σ (s) (i) = a j , σ (s) (k) = a |A| and i > k , and the estimation of u (m ′ ) (a j ) as u (m ′ ) (a j ) = Proj [-1,1] τ sig -1 1 m 1 T s=1 1 n (s) j,1 + n (s) j,2 > 0 n (s) j,1 n (s) j,1 + n (s) j,2 , Hence, m 1 = m ′ s=1 1 n (s) j,1 + n (s) j,2 > 0 . By Lemma E.1, since u (s) (a |A| ) = 0 for every s ∈ {1, 2, . . . , m ′ }, and noting that n (s) j,1 + n (s) j,2 = # σ (s) (a j ) • # σ (s) (a |A| ), we have E σ (s) n (s) j,1 n (s) j,1 + n (s) j,2 = sig u (s) (a j ) -u (s) (a |A| ) τ = sig 1 τ u (s) (a j ) .
By Hoeffding's Inequality, we have that with probability at least 1 -δ 2|A| ,
1 m 1 T s=1 1 n (s) j,1 + n (s) j,2 > 0 n (s) j,1 n (s) j,1 + n (s) j,2 -sig 1 τ u (s) (a j ) ≤ 1 2m 1 log 4|A| δ . Let u (m ′ ), * (a j ) ∈ [-1, 1] be the scalar satisfying sig 1 τ u (m ′ ), * (a j ) = 1 m 1 m ′ s=1 1 n (s) j,1 + n (s) j,2 > 0 • sig 1 τ u (s) (a j ) .
Since the logistic function is monotone and continuous, u (m ′ ), * (a j ) is unique and must exist. Then, with probability at least 1
-δ 2|A| , sig 1 τ u (m ′ ) (a j ) -sig 1 τ u (m ′ ), * (a j ) (i) ≤ sig sig -1 1 m 1 T s=1 1 n (s) j,1 + n (s) j,2 > 0 n (s) j,1 n (s) j,1 + n (s) j,2 -sig 1 τ u (m ′ ), * (a j ) = 1 m 1 T s=1 1 n (s) j,1 + n (s) j,2 > 0 n (s) j,1 n (s) j,1 + n (s) j,2 -sig 1 τ u (m ′ ), * (a j ) ≤ 1 2m 1 log 4|A| δ .
(i) uses the monotonicity of sig and the property of projection.
For any u ∈ [-1, 1], we have
dsig u τ du = 1 τ sig u τ 1 -sig - u τ ≥ 1 τ sig - 1 τ 1 -sig 1 τ = 1 τ sig - 1 τ 2 = 1 τ e 1 τ + 1 2 .
Since sig is monotonic, by Taylor expansion and the fact that u
(m ′ ) (a j ), u (m ′ ), * (a j ) ∈ [-1, 1], we get that with probability at least 1 -δ 2|A| , u (m ′ ) (a j ) -u (m ′ ), * (a j ) ≤ τ e 1 τ + 1 2 1 2m 1 log 4|A| δ .
Lemma E.2 in the following shows that u (m ′ ), * (a j ) is bounded between the minimum and maximum
of u (s) (a j ) m ′
s=1 . Then, by further utilizing the assumption that the variation of the utility vectors is small, we can bound the distance between u (m ′ ) and u (m ′ ) .
Lemma E.2. Let x 1 , . . . , x n ∈ [-1, 1] and sig avg := 1 n n i=1 sig(x i ), we have min i∈[n] x i ≤ sig -1 sig avg ≤ max i∈[n] x i .
The proof is postponed to Appendix E.2. Hence,
u (m ′ ), * (a j ) ∈ min u (s) (a j ) m ′ s=1 , max u (s) (a j ) m ′ s=1 . Finally, u (m ′ ) (a j ) -u (m ′ ) (a j ) ≤τ e 1 τ + 1 2 1 2m 1 log 4|A| δ + u (m ′ ), * (a j ) -u (m ′ ) (a j ) ≤τ e 1 τ + 1 2 1 2m 1 log 4|A| δ + max s∈{1,2,...,m ′ -1} u (s) (a j ) -u (m ′ ) (a j ) ≤τ e 1 τ + 1 2 1 2m 1 log 4|A| δ + m ′ -1 s=1 u (s+1) (a j ) -u (s) (a j ) . When m ′ p 4 ≥ 2 log 2 δ , with probability at least 1 -δ 2 , m 1 ≥ m ′ 2 p 2 .
By union bound, with a probability at least 1 -( δ 2 + δ 2|A| ), we have
u (m ′ ) (a j ) -u (m ′ ) (a j ) ≤ τ e 1 τ + 1 2 1 m ′ p 2 log 4|A| δ + m ′ -1 s=1 u (s+1) (a j ) -u (s) (a j ) .
Since a j can be any action other than a |A| , by applying union bound, the following holds with probability at least 1δ,
u (m ′ ) -u (m ′ ) ∞ ≤ τ e 1 τ + 1 2 p log 4|A| δ m ′ + m ′ -1 s=1 u (s+1) -u (s) ∞ .
Remark E.3. Due to the monotonicity of the logistic function sig, the following two projections on sig x τ are equivalent:
Proj [sig(-1 τ ),sig( 1 τ )] sig x τ := min max sig x τ , sig - 1 τ , sig 1 τ , sig Proj [-1,1] (x) τ :=sig min (max (x, -1) , 1) τ . E.2 OMITTED PROOFS Lemma E.1. Let # S (a) := a ′ ∈S 1 (a ′ = a)
represent the number of elements in a multiset S that are equal to a ∈ A. For any utility vector u, temperature τ > 0, a multiset of proposed actions S with cardinality |S| = K, and any two actions a ̸ = b ∈ S, we have
1 # S (a) • # S (b) E σ K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) S = sig u(a) -u(b) τ .
The expectation is taken over the distribution of the permutation σ under the ranking model (PL).
Proof. We will abuse the notion σ > from permutations to subsets of actions. When proposing a set of actions S, let a S > b denote the event that a is ahead of b in the permutation given by the environment.
In PL model, the probability that action a ranks before action b is that
P a {a,b} < b | τ, u = exp 1 τ u(a) exp 1 τ u(a) + exp 1 τ u (b)
.
By definition, let the multiset of the K proposed actions be S. Then, the probability of the K-wise permutation is
P (σ | S, τ, u) = K k1=1 exp 1 τ u (σ (k 1 )) K k2=k1 exp 1 τ u (σ (k 2 )) . (E.1)
Recall that Σ(S) denotes the set that contains all the permutations of the elements in S. Hence, we have
E K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) S = σ∈Σ(S) P (σ | S, τ, u) K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) = σ∈Σ(S) : σ(1)=a P (σ | S, τ, u) K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) (E.2) + σ∈Σ(S) : σ(1)=b P (σ | S, τ, u) K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) (E.3) + σ∈Σ(S) : σ(1)̸ ∈{a,b} P (σ | S, τ, u) K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) . (E.4)
We deal with (E.2) first:
σ∈Σ(S) : σ(1)=a P (σ | S, τ, u) K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) = σ∈Σ(S) : σ(1)=a P (σ | S, τ, u) # S (b) + K k1=2 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) =# S (b)P (σ(1) = a | S, τ, u) + σ∈Σ(S) : σ(1)=a P (σ | S, τ, u) K k1=2 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) =# S (b)P (σ(1) = a | S, τ, u) + P (σ(1) = a | S, τ, u) σ∈Σ(S\{a}) P (σ | S \ {a} , τ, u) K k1=2 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) =# S (b)P (σ(1) = a | S, τ, u) + P (σ(1) = a | S, τ, u) E K-1 k1=1 K-1 k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) | S \ {a} .
Similarly, for (E.3), we have σ∈Σ(S) : σ(1)=b
P (σ | S, τ, u) K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) =P (σ(1) = b | S, τ, u) E K-1 k1=1 K-1 k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) | S \ {b} .
Let Unique (S) be the set of non-repeated elements in S. Then, (E.4) can be written as, σ∈Σ(S) : σ(1)̸ ∈{a,b}
P (σ | S, τ, u) K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) = c∈Unique(S) : c̸ ∈{a,b} P (σ(1) = c | S, τ, u) E K-1 k1=1 K-1 k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) | S \ {c} .
Next, we will use induction to show that for any actions a ̸ = b ∈ S, the following holds:
E K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) S = # S (
E K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) | S =P (σ = ((a, b)) | {a, b} , τ, u) =sig u(a) -u(b) τ =# {a,b} (a)# {a,b} (b)sig u(a) -u(b) τ .
Lemma E.4. For any utility vector u, temperature τ > 0, and a multiset of actions S, the marginal probability of any action a ∈ A ranking at the first place of the permutation can be written as
P (σ(1) = a | S, τ, u) = # S (a) exp 1 τ u (a) a ′ ∈S exp 1 τ u (a ′ ) .
The proof is presented later in this section.
this section cite: []

Section: Induction step.
When (E.5) holds for any S with |S| = K -1. Then, we will show that it still holds for any S with |S| = K. By Lemma E.4, (E.2) is equal to
# S (b)P (σ(1) = a | S, τ, u) + P (σ(1) = a | S, τ, u) σ∈Σ(S\{a}) P (σ | S \ {a} , τ, u) K k1=2 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) =# S (a)# S (b) exp 1 τ u (a) a ′ ∈S exp 1 τ u (a ′ ) + # S (a) exp 1 τ u (a) a ′ ∈S exp 1 τ u (a ′ ) # S\{a} (a) • # S\{a} (b)sig u(a) -u(b) τ =# S (a)# S (b) exp 1 τ u (a) a ′ ∈S exp 1 τ u (a ′ ) + # S (a) exp 1 τ u (a) a ′ ∈S exp 1 τ u (a ′ ) (# S (a) -1) • # S (b)sig u(a) -u(b) τ .
Similarly, (E.3) is equal to
# S (b) exp 1 τ u (b) a ′ ∈S exp 1 τ u (a ′ ) # S (a) • (# S (b) -1) sig u(a) -u(b) τ ,
and (E.4) is equal to
1 - # S (a) exp 1 τ u (a) + # S (b) exp 1 τ u (b) a ′ ∈S exp 1 τ u (a ′ ) # S (a) • # S (b)sig u(a) -u(b) τ .
Lastly, by summing them up, we have
E K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) | S =# S (a)# S (b) exp 1 τ u (a) a ′ ∈S exp 1 τ u (a ′ ) -# S (a) exp 1 τ u (a) + exp 1 τ u (b) a ′ ∈S exp 1 τ u (a ′ ) • # S (b)sig u(a) -u(b) τ + # S (a) • # S (b)sig u(a) -u(b) τ . Note that sig u(a)-u(b) τ = exp( u(a)-u(b) τ ) exp( u(a)-u(b) τ )+1 = exp( u(a) τ ) exp( u(a) τ )+exp( u(b) τ )
. Therefore,
E K k1=1 K k2=k1+1 1 (σ(k 1 ) = a) • 1 (σ(k 2 ) = b) S = # S (a) • # S (b)sig u(a) -u(b) τ ,
and we complete the induction.
Lemma E.4. For any utility vector u, temperature τ > 0, and a multiset of actions S, the marginal probability of any action a ∈ A ranking at the first place of the permutation can be written as
P (σ(1) = a | S, τ, u) = # S (a) exp 1 τ u (a) a ′ ∈S exp 1 τ u (a ′ ) .
Proof. Let Σ (S) be the set containing all permutations of S. By definition, for any action a ∈ A,
P (σ(1) = a | S, τ, u) = σ∈Σ(S) : σ(1)=a P (σ | S, τ, u) = σ∈Σ(S) : σ(1)=a |S| k1=1 exp 1 τ u (σ (k 1 )) |S| k2=k1 exp 1 τ u (σ (k 2 )) .
Since there are # S (a) action a in S, by rearranging the terms, we have
P (σ(1) = a | S, τ, u) =# S (a) exp 1 τ u (a) a ′ ∈S exp 1 τ u (a ′ ) σ∈Σ(S\{a}) |S|-1 k1=1 exp 1 τ u (σ (k 1 )) |S|-1 k2=k1 exp 1 τ u (σ (k 2 )) =# S (a) exp 1 τ u (a) a ′ ∈S exp 1 τ u (a ′ ) σ∈Σ(S\{a}) P (σ | S \ {a} , τ, u) =# S (a) exp 1 τ u (a) a ′ ∈S exp 1 τ u (a ′ )
.
Lemma E.2. Let x 1 , . . . , x n ∈ [-1, 1] and sig avg := 1 n n i=1 sig(x i ), we have min i∈[n]
x i ≤ sig -1 sig avg ≤ max i∈[n] x i .
Proof. The logistic function sig(x) is increasing monotonically with respect to x, since dsig dx = exp(x) (exp(x)+1) 2 > 0. Then, without loss of generality, let
x 1 ≤ x 2 ≤ • • • ≤ x n . Thus, sig(x 1 ) ≤ sig avg ≤ sig(x n ).
Since sig(x) is monotonic and continuous, there exists only one ζ ∈ [x 1 , x n ] such that sig(ζ) = sig avg .
this section cite: []

Section: F ALGORITHMS AND DIAGRAMS
In this section, we present the algorithms' diagrams and pseudo-code of learning with InstUtil Rank and AvgUtil Rank individually.
Published as a conference paper at ICLR 2026 Timestep t (Full-Information) Estimate(•) Oracle Alg(•) σ (s) t s=max{t-m+1,1}
u (t) π (t+1) u (s) (t-1) s=1 Timestep t (Bandit) Estimate(•) Oracle Alg(•) σ (s) t s=max{t-m+1,1} u (t) + •(1 -γ) γ 1(A) |A| π (t+1) u (s) (t-1) s=1
Figure 10: The diagram of Algorithm 2 with InstUtil Rank under full-information feedback (top) and bandit feedback (bottom). + ⃝ represents the addition of (1γ) times the output the Alg and γ times a uniform distribution over A.
this section cite: []

Section: F.1 THE ALGORITHM AND DIAGRAM FOR InstUtil Rank
We present the diagram and the algorithm pseudo-code of learning with InstUtil Rank: Figure 10 and Algorithm 2.
Algorithm 2 Online Learning with InstUtil Rank Feedback 1: Input: Action space A, any full-information no-regret learning algorithm Alg with numeric utility feedback, selected action number K, estimation window size m, and exploration rate γ. 2: Initialize π (1) as uniform distribution 1 |A| over A 3: for timestep t = 1, 2, . . . , T do 4: if Full-information setting then 5: K = |A| in this case. Select all |A| actions. 6: else if Bandit setting then 7: Sample K actions independently with replacement from π (t) . 8: end if 9: Receive a ranking feedback σ (t) = σ (t) (1), σ (t) (2), . . . , σ (t) (K) from the environment. 10: u (t) = Estimate σ (s) t s=max{t-m+1,1} by calling Algorithm 1. 11: if Full-information setting then 12: π (t+1) ← Alg u (s) t s=1 . 13: else if Bandit setting then 14: π (t+1) ← (1γ)Alg u (s) t s=1 + γ 1(A) |A| . 15: end if 16: end for Timestep t (Full-Information) Estimate(•) Oracle Alg(•) σ (s) t s=max{t-m+1,1} u (t) avg u (t) avg t s=1 π (t+1) Timestep t (Bandit) Estimate(•) (F.1) Oracle Alg(•) σ (s) t s=max{t-m+1,1} u (t) avg-est u (t) avgest t s=1 + •(1 -γ) γ 1(A) |A| π (t+1) u (s) empirical (t-1) s=1 Figure 11: The diagram of Algorithm 3 with AvgUtil Rank under full-information feedback (top) and bandit feedback (bottom).
represents copying the estimated utility vector for t times. + ⃝ represents the addition of (1γ) times the output the Alg and γ times a uniform distribution over A.
this section cite: []

Section: F.2 THE ALGORITHM AND DIAGRAM FOR AvgUtil Rank
We present the diagram and the algorithm pseudo-code of learning with AvgUtil Rank: Figure 11 and Algorithm 3.
this section cite: []

Section: G PROOF OF THEOREM 5.2 (FULL-INFORMATION)
In this section, we prove the regret upper bound under InstUtil Rank and full-information feedback.
Theorem G.1 (Formal version of Theorem 5.2 (Full-Information)). Consider Algorithm 2 and fullinformation feedback. For any δ ∈ (0, 1), T > 0, and any no-regret learning algorithm with numeric utility feedback, Alg, with probability at least (1δ), by choosing m = T
P (T ) 2 3 log 4|A|T δ 1 3 , R (T ),external satisfies R (T ),external ≤R (T ),external Alg, u (t) T t=1 + 2τ e 1 τ + 1 2 P (t) 1 3 T 2 3 log 4|A|T δ 1 3 + 2 P (t) 1 3 T 2 3 log 4|A|T δ 1 3 + 4 P (t) -2 3 T 2 3 log 4|A|T δ 1 3 . (G.1)
Published as a conference paper at ICLR 2026 For any given δ, we require the utility estimation bound to hold with probability at least 1 -δ T , then by union bound, with probability at least 1δ, R (T ),external -R (
T ),external Alg, u (t) T t=1 ≤2τ e 1 τ + 1 2 log 4|A|T δ m T + 2m P (T ) + 2 . By choosing m = T P (T ) 2 3 log 4|A|T δ 1 3 , we conclude the proof.
H PROOF OF THEOREM 5.2 (BANDIT)
In this section, we prove the regret upper bound under InstUtil Rank and bandit feedback.
Theorem H.1 (Formal version of Theorem 5.2 (Bandit)). Consider Algorithm 2 and bandit feedback. For any δ ∈ (0, 1), T > 0, and any no-regret learning algorithm with numeric utility feedback, Alg, with probability at least (1-δ), by choosing γ = P (T )
T 1 5 , m = 32|A| 4 K 4 T P (T ) 4 5 log 8|A|T δ , R (T ) satisfies R (T ) ≤R (T ),external Alg, u (t) T t=1 + 2 2T log 2 δ +    τ K e 1 τ + 1 2 |A| + 1    P (
T ) 1 5 T 4 5 (H.1) + 64|A| 4 K 4 P (T ) 1 5 T 4 5 log 8|A|T δ + 128|A| 4 K 4 T P (T ) 4 5 log 8|A|T δ .
Proof. We define: R (T ) := max t) .
π∈∆ A T t=1 u (t) , π -π (
Then, R (T ) ≤ R (T ) -R (T ),external ♥ + R (T ),external -R (T ) ♠ + R (T ) -R (T ),external Alg, u (t) T t=1 ♦ + R (T ),external Alg, u (t) T t=1 ♣ .
Note that ♠ can be bounded by bounding
u (t) -u (t) ∞ as in Appendix G.
♣ is sublinear by the definition of Alg. Next, we will introduce lemmas that individually bound ♥, ♦. The proofs are postponed to Appendices H.1 and H.2.
Lemma H.2 (♥). For any T > 0 and δ ∈ (0, 1), with probability at least 1δ: R (T ) -R (T ),external ≤ 2 2T log 1 δ .
Lemma H.3 (♦). The difference between R (T ) and R (T ),external Alg, u (t) T t=1 satisfies:
R (T ) -R (T ),external Alg, u (t) T t=1 ≤ 2γT.
With Lemma H.2 and Lemma H.3, under the conditions in Theorem 5.2, by letting Theorem 5.1 hold with probability 1 -δ 2T at each timestep and using the union bound, with probability at least 1δ, the regret satisfies
R (T ) ≤R (T ),external Alg, u (t) T t=1 + 2 2T log 2 δ + 2 τ e 1 τ + 1 2 p log 8|A|T δ m T + 2m P (T ) + 2 + γT.
In this case, each action a ∈ A is chosen with probability at least p that satisfies
p ≥1 -1 - γ |A| K ≥ 1 -exp -K γ |A| ≥1 -1 -K γ |A| + 1 2 K γ |A| 2 = K γ |A| - 1 2 K γ |A| 2 . Since K γ |A| ≤ 1, we have 1 2 K γ |A| ≥ 1 2 K γ |A| 2 ⇒ p ≥ Kγ 2 |A| . By letting γ = P (T ) T 1 5 , m = 32|A| 4 K 4 T P (T ) 4 5 log 8|A|T δ , we have R (T ) ≤ R (T ),external Alg, u (t) T t=1 + O T 4 5 P (T ) 1 5 log T δ . The condition m ≥ 2 log( 4T δ ) p 4
is also satisfied since
mp 4 ≥ m K 4 γ 4 16|A| 4 = 2 log 8|A|T δ ≥ 2 log 4T δ .
this section cite: []

Section: H.1 BOUNDING ♥
We will show that R (T ) -R (T ),external is sublinear by using a standard concentration bound.
Lemma H.2 (♥). For any T > 0 and δ ∈ (0, 1), with probability at least 1δ: R (T ) -R (T ),external ≤ 2 2T log 1 δ .
Proof. Let t) , π (t) . t) with replacement and the update rule of π t) is a martingale difference sequence.
d (t) := 1 K a∈o (t) u (t) (a) -u (
By our algorithm design, each element of o (t) is sampled i.i.d. from π (
(t) is deterministic, E d (t) | σ (s) t-1 s=1 , u (s) t-1 s=1 = 0, so that d (
this section cite: []

Section: Due to the bounds of
1 K a∈o (t) u (t) (a) ≤ 1, u (t) , π (t) ≤ 1, we have d (t) = 1 K a∈o (t) u (t) (a) -u (t) , π (t) ≤ 1 K a∈o (t) u (t) (a) + u (t) , π (t) ≤ 2.
Furthermore, we have T ) .
T t=1 d (t) = T t=1   1 K a∈o (t) u (t) (a)   - T t=1 u (t) , π (t) = max π∈∆ A T t=1 u (t) , π -u (t) , π (t) -max π∈∆ A T t=1   u (t) , π - 1 K K j=1 u (t) σ (j)   =R (T ),external -R (
Next, we will introduce Azuma-Hoeffding inequality to finish the concentration bound.
Theorem H.4 (Azuma-Hoeffding inequality). For any martingale difference sequence Y 1 , . . . , Y n such that ∀j ∈ [n] , a j ≤ Y j ≤ b j , the following holds for any w ≥ 0.
P   n j=1 Y j ≥ w   ≤ exp - 2w 2 n j=1 (b j -a j ) 2 .
Then by Theorem H.4, with probability at least t) in Algorithm 2.
1 -δ R (T ) ≤ R (T ),external + 2 2T log 1 δ . H.2 BOUNDING ♦ ♦ can be bounded by O (γT ) by definition of π (
Lemma H.3 (♦). The difference between R (T ) and R (T ),external Alg, u (t) T t=1 satisfies:
R (T ) -R (T ),external Alg, u (t) T t=1 ≤ 2γT. Proof. Let π (t+1) = Alg u (s) t s=1 . Then, R (T ) -R (T ),external Alg, u (t) T t=1 = max π∈∆ A T t=1 u (s) , π -π (t) -max π∈∆ A T t=1 u (t) , π -π (t) = T t=1 u (t) , π (t) -π (t) = T t=1 u (t) , π (t) -(1 -γ)π (t) + γ 1 (A) |A| ≤γ T t=1 u (t) ∞ • π (t) 1 + 1 (A) |A| 1 ≤ 2γT.
I PROOF OF THEOREM 6.2
Theorem I.1 (Formal version of Theorem 6.2). Consider AvgUtil Rank with full-information feedback and Algorithm 3. For any δ ∈ (0, 1), T > 0, and any no-regret learning algorithm with numeric utility feedback Alg that satisfies Assumption 6.1, with probability at least (1δ), by choosing m = 2T 2 3 log 4|A|T δ , R (T ),external satisfies R (T ),external ≤R (T ),external Alg, u (t) T t=1 + |A|τ e 1 τ + 1 2 LT 5 3 + 4T 2 3 log 4|A|T δ (I.1) + 4|A|LT 4 3 (log T + 1) log 4|A|T δ 2 + 4|A|LT 5 3 log 4|A|T δ .
Proof. Let π (t+1) = Alg u (s) t s=1 , i.e., the strategy generated by Alg when the ground-truth utility vectors are given. Then, t) .
R (T ),external -R (T ),external Alg, u (t) T t=1 = max π∈∆ A T t=1 u (t) , π -π (t) -max π∈∆ A T t=1 u (t) , π -π (t) = T t=1 u (t) , π (t) -π (t) ≤ m-1 t=1 u (t) ∞ • π (t) -π (t) 1 + T t=m u (t) • π (t) -π (t) ≤2m + T t=m u (t) • π (t) -π (
By Assumption 6.1 and Theorem 5.1, for any t ≥ m, with probability at least 1δ we have
π (t) -π (t) ≤ Lt u (t) avg -u (t) avg ≤Lt |A|     τ e 1 τ + 1 2 log 4|A|T δ m + t-1 s=t-m+1 2 s + 1     . Then, R (T ),external -R (T ),external Alg, u (t) T t=1 ≤2m + L|A|τ e 1 τ + 1 2 log 4|A|T δ m T 2 + 2L|A| T t=m t t-1 s=t-m+1 1 s + 1 ≤2m + L|A|τ e 1 τ + 1 2 log 4|A|T δ m T 2 + L|A| T t=1 m(2t + m -1) t ≤2m + L|A|τ e 1 τ + 1 2 log 4|A|T δ m T 2 + Lm 2 |A| T t=1 1 t + 2|A|mLT ≤L|A|τ e 1 τ + 1 2 log 4|A|T δ m T 2 + 2m + Lm 2 |A| (log T + 1) + 2|A|mLT.
By choosing m = 2T
2 3 log 4|A|T δ , we have R (T ),external ≤R (T ),external Alg, u (t) T t=1 + O LT 5 3 log 4|A|T δ . Moreover, now m ≥ 2 log( 2T δ ) p 4
, where p = 1 since we are considering full-information feedback.
Therefore, for any t ≥ M and a ∈ A, we have
u (t) avg-est (a) -u (t) avg (a) ∞ ≤ 1 ⌊t/M ⌋ ⌊t/M ⌋ s=1 s•M -1 s ′ =(s-1)M +1 u (s ′ +1) (a) -u (s ′ ) (a) + 2M t .
By combining all the pieces together, we have
u (t) avg-est (a) -u (t) avg (a) ≤ 2C δ ⌊t/M ⌋ ⌊t/M ⌋ s=1 s u (s•M ) empirical (a) -u (s•M ) empirical (a) + u ((s-1)M ) empirical (a) -u ((s-1)M ) empirical (a) + 1 ⌊t/M ⌋ ⌊t/M ⌋ s=1 s•M -1 s ′ =(s-1)M +1 u (s ′ +1) (a) -u (s ′ ) (a) + 2M t .
Then, since
T t=M 1 ⌊t/M ⌋ = ⌊T /M ⌋ s=1 min{(s+1)•M -1,T } t=s•M 1 s ≤ ⌊T /M ⌋ s=1 M s , we have T t=1 u (t) avg-est (a) -u (t) avg (a) = T t=M u (t) avg-est (a) -u (t) avg (a) + M -1 t=1 u (t) avg-est (a) -u (t) avg (a) ≤2C δ ⌊T /M ⌋ s=1 s u (s•M ) empirical (a) -u (s•M ) empirical (a) + u ((s-1)M ) empirical (a) -u ((s-1)M ) empirical (a) ⌊T /M ⌋ s ′ =1 M s ′ + ⌊T /M ⌋ s=1   s•M -1 s ′ =(s-1)M +1 u (s ′ +1) (a) -u (s ′ ) (a)   •   ⌊T /M ⌋ s ′ =1 M s ′   + T t=1 2M t + 2M ≤2C δ M ⌊T /M ⌋ s=1 s u (s•M ) empirical (a) -u (s•M ) empirical (a) + u ((s-1)M ) empirical (a) -u ((s-1)M ) empirical (a) (log (⌊T /M ⌋) + 1) + M ⌊T /M ⌋ s=1   s•M -1 s ′ =(s-1)M +1 u (s ′ +1) (a) -u (s ′ ) (a)   • (log (⌊T /M ⌋) + 1) + 2M (log T + 1) + 2M.
When s = 1, s u
((s-1)M ) empirical (a) -u ((s-1)M )
empirical (a) = 0 by definition. When s > 1, since M ≥ 2m, we have
s (s -1)M -m + 2 ≤ s (s -1)M/2 + 2 ≤ s (s -1)M/2 ≤ 4s s • M = 4 M .
Hence, s u
((s-1)M ) empirical (a) -u ((s-1)M ) empirical (a) ≤s τ |A| e 1 τ + 1 2 γ log 12|A|T δ m + 4KC δ (s-1)M -1 s ′ =(s-1)M -m+1 s s ′ + 1 ≤ T M τ |A| e 1 τ + 1 2 γ log 12|A|T δ m + 16KC δ m M .    
+ M (log T + 1) P (T ) + 2M (log T + 2) .
Lastly, similar to the proof in Appendix I, let π (t+1) = Alg u (s) t s=1 . Then, we have
R (T ),external -R (T ),external Alg, u (t) T t=1 = T t=1 u (t) , π (t) -π (t) ≤ T t=1 u (t) • π (t) -π (t) ≤ |A| T t=1 (1 -γ)Alg u (t) avg-est t s=1 + γ 1 (A) |A| -π (t) ≤(1 -γ) |A| T t=1 Alg u (t) avg-est t s=1 -π (t) + γ |A| T t=1 1 (A) |A| -π (t) ≤ |A| T t=1 Alg u (t) avg-est t s=1 -π (t) + 2γ |A|T.
Further, by Assumption 6.1, we have t)   avg ∞
|A| T t=1 Alg u (t) avg-est t s=1 -π (t) ≤L |A| T t=1 t u (t) avg-est -u (t) avg ≤L|A|T T t=1 u (t) avg-est -u (
.
By combining all the pieces together, we have
R (T ),external ≤R (T ),external Alg, u (t) T t=1 + L|A|T     4C δ (log T + 1)     T 2 M τ |A| e 1 τ + 1 2 γ log 12|A|T δ m + 16KC δ m M T         + L|A|T M (log T + 1) P (T ) + 2M (log T + 2) + 2γ |A|T ≤R (T ),external Alg, u (t) T t=1
+ O   log 1 δ 3 2 γ 2 LT 3 M √ m + m log 1 δ 2 γ 2 M LT 2 + LM P (T ) T + 2γT   ,
where O hides all the log T terms. Lastly, by Lemma H.2, with probability at least 1 -δ 3 , we have R (T ) ≤ R (T ),external + 2 2T log 3 δ .
Moreover, by taking γ = min L 1 3 T 5 18
P (T ) 1 6 , 1 , m = 2T 2 3 |A| 4 log 12|A|T δ
, and M = max 4T 5 6
P (T ) -1 2 |A| 4 log 12|A| 2 T δ , 2m , we can easily verify that m ≥ 2 log( 6 δ ) γ 4
|A| 4 and M ≥ C δ . Finally, by a union bound argument, we complete the proof.
this section cite: []

Section: K PROOF OF SECTION 7
Lemma K.1. For any T > 0 and sequence of strategy profiles π (1) , π (2) , . . . , π (T ) , the variation of utility vectors of any player i ∈ [N ] satisfies that
T t=2 u (t) i -u (t-1) i ≤ √ A N j ′ =1 |A j ′ | T t=2 N j=1 π (t) j -π (t-1) j , (K.1)
where A = max j |A j |.
Proof. For any timestep t, player i ∈ [N ], and joint action a
-i ∈ × j̸ =i A j , let π (t) -i (a -i ) := j̸ =i π(t)
j (a j ). Then, for any timestep t, player i ∈ [N ], and action a i ∈ A i , we have
u (t) i (a i ) -u (t-1) i (a i ) ≤ a ′ ∈× N j=1 Aj U i (a ′ ) 1 (a ′ i = a i ) π (t) -i (a ′ -i ) -π (t-1) -i (a ′ -i ) = U i (a i , a ′ -i ) a ′ -i ∈× j̸ =i Aj , π(t)
-i -π (t-1) -i ≤ U i (a i , a ′ -i ) a ′ -i ∈× j̸ =i Aj ∞ • π (t) -i -π (t-1) -i 1 ≤ π (t) -i -π (t-1) -i 1 .
Further, for any a, b, a ′ , b ′ ∈ [0, 1], we have |ab -
a ′ b ′ | = |ab -ab ′ + ab ′ -a ′ b ′ | ≤ a |b -b ′ | + |a -a ′ | b ′ ≤ |a -a ′ | + |b -b ′ |.
Therefore, by recursively using it, for any a -i ∈ × j̸ =i A j , we have
π (t) -i (a -i ) -π (t-1) -i (a -i ) = j̸ =i π (t) j (a j ) - j̸ =i π (t-1) j (a j ) ≤ j̸ =i π (t) j (a j ) -π (t-1) j (a j ) .
where (i) uses Assumption 6.1 and Assumption 7.1. Then, the accumulated variation of π (t) over time is bounded by
T -1 t=1 π (t+1) -π (t) ≤ ηT + 2 |A|LM T M = ηT + 2 |A|LT,
since there are at most T M timesteps of t ∈ [T ] satisfying t ≡ 0 (mod M ).
With Lemma K.2, we can prove that R (T ),external i is sublinear for any player i ∈ [N ] by Theorem 5.2, Theorem 6.2, and Theorem 6.3. 2 Then, by the folklore result that no-external-regret learning leads to approximate CCE (Hart & Mas-Colell, 2000;Blum & Mansour, 2007), Theorem 7.2 and Theorem 7.3 are proved.
Remark K.3. With the hardness in Theorem 4.3, under AvgUtil Rank feedback, both our noregret result for the online setting and the equilibrium computation result for the game setting hold for a constant τ > 0 (that cannot be arbitrarily small). However, we note that the equilibrium computation result may still be possible when τ → 0 + : with such a deterministic ranking model, the best-response action against the history play of the opponents is now available, precisely leading to the celebrated algorithm of fictitious-play (FP) (Robinson, 1951;Brown, 1951). FP is known to converge to an equilibrium in certain games (Robinson, 1951;Monderer & Shapley, 1996;Sela, 1999;Berger, 2005) (with (slow) convergence rates (Robinson, 1951;Daskalakis & Pan, 2014;Abernethy et al., 2021)), despite that it fails to be no-regret in the online setting (Fudenberg & Levine, 1995;1998).
this section cite: ['b20', 'b6', 'b37', 'b7', 'b37', 'b32', 'b42', 'b5', 'b37', 'b10', 'b1', 'b17', 'b18']

Section: L PROPERTIES OF FOLLOW-THE-REGULARIZED-LEADER (FTRL)
Firstly, we will define the strongly convex function.
Definition L.1. For any integer n and a convex set
X ⊆ R n , a differentiable function ψ(x) : X → R is called c 0 -strongly convex (c 0 > 0) when ψ(x) ≥ ψ(x ′ ) + ⟨∇ψ(x ′ ), x -x ′ ⟩ + c 0 2 ∥x -x ′ ∥ 2 (L.1)
holds for any x, x ′ ∈ X .
Specifically, if (L.1) holds for c 0 = 0, then we call ψ a convex function.
Next, we will introduce the well-known no-regret learning algorithm of Follow-The-Regularized-Leader (Shalev-Shwartz et al., 2012;Hazan et al., 2016).
Definition L.2 (Follow-The-Regularized-Leader (FTRL)). For any T > 0 and at any timestep t ∈ {0} ∪ [T -1], given the utility vectors u (s) t s=1 , the strategy at timestep t + 1, π (t+1) , is defined as,
π (t+1) = argmax π∈∆ A λ t s=1 u (s) , π -ψ(π) ,(FTRL)
for some constant λ > 0. Typically, λ is taken to be Θ (T -r ) for some constant r > 0.
Next, we will introduce the smoothness of (FTRL). Proof. By the first-order optimality, at any timestep t ∈ {0} ∪ [T -1] for any two sequences of utility vectors u (s) t s=1 and u ′ (s) t s=1
, let the corresponding strategy generated by (FTRL) be 2 Inspecting the proofs of these theorems shows that one can readily derive an upper bound on R (T ),external that is tighter than the corresponding bound on R (T ) in the bandit setting. Consequently, R (T ),external i can also be bounded in games under the bandit feedback.
this section cite: ['b45', 'b21']

Section: 
Algorithm 3 Online Learning with AvgUtil Rank Feedback 1: Input: Action space A, any full-information no-regret algorithm Alg under numeric feedback, selected action number K, estimation window size m, exploration rate γ, and block size M . 2: Initialize π (1) as uniform distribution 1  |A| over A 3: for timestep t = 1, 2, . . . , T do 4:
if Full-information setting then 5: K = |A| in this case. Select all |A| actions. 6: else if Bandit setting then 7:
Sample K actions independently with replacement from π (t) . 8:
end if 9:
Receive a ranking feedback σ (t) = σ (t) (1), σ (t) (2), . . . , σ (t) (K) from the environment.
10: if Full-information setting then 11: u (t) avg = Estimate σ (s) t s=max{t-m+1,1} by calling Algorithm 1. 12: π (t+1) ← Alg u (t) avg t s=1 , i.e., the strategy generated by Alg by setting all utility vectors from timestep 1 to t as u (t) avg . 13: else if Bandit setting then 14: u (t) empirical = Estimate σ (s) t s=max{t-m+1,1} by calling Algorithm 1. 15:
Let n (t) (a) := t s=1 # o (s) (a) for any a ∈ A as the number of times action a has been proposed up to timestep t. Then, the estimated average utility is
u (t) avg-est (a) := 1 ⌊t/M ⌋ ⌊t/M ⌋ s=1 u (s•M ) empirical (a)n (s•M ) (a)-u ((s-1)M ) empirical (a)n ((s-1)M ) (a) n (s•M ) (a)-n ((s-1)M ) (a) t ≥ M 0 t < M (F.1) {Let u(0)
empirical (a) = n (0) (a) = 0 for any action a ∈ A} 16:
π (t+1) ← (1 -γ)Alg u (t) avg-est t s=1 + γ 1(A) |A| .
17:
end if 18: end for Proof. By Theorem 5.1, we have
R (T ),external -R (T ),external Alg, u (t) T t=1 = max π∈∆ A T t=1 u (t) , π -π (t) -max π∈∆ A T t=1 u (t) , π -π (t) ≤ max π∈∆ A T t=1 u (t) -u (t) , π -π (t) ≤ T t=1 u (t) -u (t) ∞ • max π∈∆ A π -π (t) 1 .
When t ≥ m, the estimation error between u (t) and u (t) is given by Theorem 5.1 with p = 1 since we are considering full-information feedback. When t < m,
m-1 t=1 u (t) -u (t) ∞ • max π∈∆ A π -π (t) 1 ≤ 4m.
this section cite: []

Section: J PROOF OF THEOREM 6.3
Theorem J.1 (Formal version of Theorem 6.3). Consider AvgUtil Rank with bandit feedback and Algorithm 3. For any δ ∈ (0, 1), T > 0, and any no-regret learning algorithm with numeric utility feedback Alg that satisfies Assumption 6.1, with probability at least (1δ), by choosing m = 2T
2 3 |A| 4 log 12|A|T δ , γ = min L 1 3 T 5 18 P (T ) 1 6 , 1 , and M = max 4T 5 6
P (T ) -1 2 |A| 4 log 12|A| 2 T δ , 2m , R (T ) satisfies R (T ) ≤R (T ),external Alg, u (t) T t=1 + L|A|T W (T ) + 2γ |A|T + 2 2T log 3 δ , (J.1)
where
C δ := |A| log 3|A| 2 T δ γ W (T ) :=4C δ (log T + 1)     T 2 M τ |A| e 1 τ + 1 2 γ log 12|A|T δ m + 16KC δ m M T     + M (log T + 1) P (T ) + 2M (log T + 2) .
Proof. In the first part of the proof, we will bound u
(t) empirical -u (t) empirical ∞
. According to Theorem 5.1 and union bound, since each action is proposed with probability at least γ |A| , with probability at least 1 -δ 3 , for any t ≥ m, we have t) . Then, for any t ∈ [T -1] and a ∈ A, we have
u (t) empirical -u (t) empirical ∞ ≤ τ |A| e 1 τ + 1 2 γ log 12|A|T δ m + t-1 s=t-m+1 u (s+1) empirical -u (s) empirical ∞ . Let # o (t) (a) be the number of action a ∈ A being proposed in o (
u (t+1) empirical (a) -u (t) empirical (a) = u (t) empirical (a) t s=1 # o (s) (a) + u (t+1) (a)# o (t+1) (a) t s=1 # o (s) (a) + # o (t+1) (a) -u (t) empirical (a) ≤ u (t) empirical (a)# o (t+1) (a) t s=1 # o (s) (a) + # o (t+1) (a) + u (t+1) (a)# o (t+1) (a) t s=1 # o (s) (a) + # o (t+1) (a) ≤K u (t) empirical (a) t s=1 # o (s) (a) + # o (t+1) (a) + K u (t+1) (a) t s=1 # o (s) (a) + # o (t+1) (a)
.
Next, we will show that since each action will be proposed with probability at least γ |A| , with high probability, there is a lowerbound for t s=1 # o (s) (a) for any timestep t.
Lemma J.2. Consider the case when actions are proposed with probability at least p > 0 at each timestep. Then, for any δ > 0, any action a ∈ A, and T > 0, with probability at least 1δ, the following holds for any t ≥
log( |A|T δ ) p : ∃t ′ ∈ [T ], such that t - log |A|T δ p ≤ t ′ ≤ t and a ∈ o (t ′ ) . (J.2) Proof. For any t ≥ log( |A|T δ )
p and action a ∈ A, the probability of (J.2) does not hold is at most
(1 -p) log ( |A|T δ ) p ≤ exp -log |A|T δ = δ |A|T .
Therefore, by union bound, with probability 1δ, (J.2) holds for any t ∈ [T ] and any action a ∈ A.
For notational simplicity, let C δ :=
|A| log 3|A| 2 T δ γ
. According to Lemma J.2, with probability at least 1 -δ 3|A| , for any timestep t ≥ C δ , we have
t s=1 # o (s) (a) ≥ t C δ ≥ t 2C δ . (J.3)
By union bound, (J.3) holds for any action a with probability 1 -δ 3 . Therefore, for any t ∈ [T -1] and a ∈ A, we have
u (t+1) empirical (a) -u (t) empirical (a) ≤ 4K C δ t + 1 .
It holds for t < C δ -1 because 4K C δ t+1 ≥ 4K ≥ 2 and all utilities are bounded in [-1, 1]. Finally, by Theorem 5.1 and union bound, with probability at least 1 -2δ 3 , we have
u (t) empirical -u (t) empirical ∞ ≤ τ |A| e 1 τ + 1 2 γ log 12|A|T δ m + 4KC δ t-1 s=t-m+1 1 s + 1 . Let n (t) (a) := t s=1 # o (s) (a)
for any a ∈ A as the number of times action a is proposed up to timestep t. For any a ∈ A and t ≥ M , we define
u (t) avg-est (a) := 1 ⌊t/M ⌋ ⌊t/M ⌋ s=1 u (s•M ) empirical (a)n (s•M ) (a) -u ((s-1)M ) empirical (a)n ((s-1)M ) (a) n (s•M ) (a) -n ((s-1)M ) (a) u (t) avg-est (a) := 1 ⌊t/M ⌋ ⌊t/M ⌋ s=1 u (s•M ) empirical (a)n (s•M ) (a) -u ((s-1)M ) empirical (a)n ((s-1)M ) (a) n (s•M ) (a) -n ((s-1)M ) (a)
.
Note that we define u
(0) empirical (a) = u (0) empirical (a) = n (0) empirical (a) = 0. For t < M , we define u (t) avg-est (a) = u (t)
avg-est (a) = 0 for any action a ∈ A. In the rest of the proof, we will bound u
(t) avg-est -u (t) avg-est ∞ and u (t) avg-est -u (t) avg ∞ individually. J.1 u (t) avg-est -u (t) avg-est ∞ UPPER BOUND
For any a ∈ A, we have
u (t) avg-est (a) -u (t) avg-est (a) ≤ 1 ⌊t/M ⌋ ⌊t/M ⌋ s=1 n (s•M ) (a) n (s•M ) (a) -n ((s-1)M ) (a) u (s•M ) empirical (a) -u (s•M ) empirical (a) + 1 ⌊t/M ⌋ ⌊t/M ⌋ s=1 n ((s-1)M ) (a) n (s•M ) (a) -n ((s-1)M ) (a) u ((s-1)M ) empirical (a) -u ((s-1)M ) empirical(
this section cite: []

Section: a) .
According to Lemma J.2, n (s•M ) (a)n ((s-1)M ) (a) ≥ M 2C δ when M ≥ C δ . Therefore, when M ≥ C δ , since n (s•M ) (a) ≤ s • M , we have
u (t) avg-est (a) -u (t) avg-est (a) ≤ 2C δ ⌊t/M ⌋ ⌊t/M ⌋ s=1 s u (s•M ) empirical (a) -u (s•M ) empirical (a) + u ((s-1)M ) empirical (a) -u ((s-1)M ) empirical (a) . J.2 u (t) avg-est -u (t) avg ∞ UPPER BOUND
For any a ∈ A,
u (t) avg-est (a) -u (t) avg (a) ∞ = 1 ⌊t/M ⌋ ⌊t/M ⌋ s=1 u (s•M ) empirical (a)n (s•M ) (a) -u ((s-1)M ) empirical (a)n ((s-1)M ) (a) n (s•M ) (a) -n ((s-1)M ) (a) -u (t) avg (a) ≤ 1 ⌊t/M ⌋ ⌊t/M ⌋ s=1 u (s•M ) empirical (a)n (s•M ) (a) -u ((s-1)M ) empirical (a)n ((s-1)M ) (a) n (s•M ) (a) -n ((s-1)M ) (a) -u (M ⌊t/M ⌋) avg (a) ♠ + u (t) avg (a) -u (M ⌊t/M ⌋) avg (a) ♣ .
Note that ♣ can be bounded by
♣ = (M ⌊t/M ⌋)u (M ⌊t/M ⌋) avg (a) + t s=M ⌊t/M ⌋+1 u (s) (a) t -u (M ⌊t/M ⌋) avg (a) ≤ M t u (M ⌊t/M ⌋) avg (a) + 1 t t s=M ⌊t/M ⌋+1 u (s) (a) ≤ 2M t .
For ♠, we have
♠ = 1 ⌊t/M ⌋ ⌊t/M ⌋ s=1   u (s•M ) empirical (a)n (s•M ) (a) -u ((s-1)M ) empirical (a)n ((s-1)M ) (a) n (s•M ) (a) -n ((s-1)M ) (a) - 1 M s•M s ′ =(s-1)M +1 u (s ′ ) (a)   ≤ 1 ⌊t/M ⌋ ⌊t/M ⌋ s=1 u (s•M ) empirical (a)n (s•M ) (a) -u ((s-1)M ) empirical (a)n ((s-1)M ) (a) n (s•M ) (a) -n ((s-1)M ) (a) - 1 M s•M s ′ =(s-1)M +1 u (s ′ ) (a) . When n (s•M ) (a) -n ((s-1)M ) (a) > 0, both u (s•M ) empirical (a)n (s•M ) (a)-u ((s-1)M ) empirical (a)n ((s-1)M ) (a) n (s•M ) (a)-n ((s-1)M ) (a) and 1 M s•M s ′ =(s-1)M +1 u (s ′ ) (a) are in the convex hull of u (s ′ ) (a) s•M s ′ =(s-1)M +1 . Therefore, u (s•M ) empirical (a)n (s•M ) (a) -u ((s-1)M ) empirical (a)n ((s-1)M ) (a) n (s•M ) (a) -n ((s-1)M ) (a) - 1 M s•M s ′ =(s-1)M +1 u (s ′ ) (a) ≤ max (s-1)M +1≤s ′ ,s ′′ ≤s•M u (s ′ ) (a) -u (s ′′ ) (a) ≤ s•M -1 s ′ =(s-1)M +1 u (s ′ +1) (a) -u (s ′ ) (a) .
Finally,
u (t) i -u (t-1) i ≤ |A i | π (t) -i -π (t-1) -i 1 ≤ |A i | N j=1 |A j | j ′ ̸ =i π (t) j ′ -π (t-1) j ′ 1 .
K.1 PROOF OF THEOREM 7.2 AND THEOREM 7.3
Before proving Theorem 7.2 and Theorem 7.3, we will show that when Assumption 7.1 is satisfied, the strategy variation is bounded.
Lemma K.2. Suppose Assumption 7.1 is satisfied. For both full-information and bandit settings, Algorithm 2 satisfies the following,
T -1 t=1 π (t) -π (t+1) ≤ ηT.
Suppose Assumption 6.1 is also satisfied, then the following holds for Algorithm 3 in the bandit setting,
T -1 t=1 π (t) -π (t+1) ≤ ηT + 2 |A|LT.
Proof. For Algorithm 2 and the full-information setting, the proof simply follows from the fact that u (t) ∈ [-1, 1] A and Assumption 7.1. For Algorithm 2 and the bandit setting, we have
π (t+1) -π (t) = (1 -γ) Alg u (s) t+1 s=1 -Alg u (s) t s=1 ≤ η.
Thus, we can conclude the proof.
For Algorithm 3 and the bandit setting, for any t ̸ ≡ 0 (mod M ), we have u
avg-est = u (t-1) avg-est . Therefore, by Assumption 7.1, we have
π (t-1) -π (t) ≤ Alg u (t-1) avg-est t-1 s=1 -Alg u (t) avg-est t s=1 = Alg u (t-1) avg-est t-1 s=1 -Alg u (t-1) avg-est t s=1 ≤ η. For any t ≡ 0 (mod M ), let u = u (t) empirical (a)n (t) (a)-u (t-M ) empirical (a)n (t-M ) (a) n (t) (a)-n (t-M ) (a)
, then we have
u (t) avg-est -u (t-1) avg-est = (t/M -1) u (t-1) avg-est + u t/M -u (t-1) avg-est ≤ M t u (t-1) avg-est + ∥u∥ ≤ 2M t |A|.
Therefore,
π (t-1) -π (t) ≤ Alg u (t-1) avg-est t-1 s=1 -Alg u (t) avg-est t s=1 ≤ Alg u (t-1) avg-est t-1 s=1 -Alg u (t-1) avg-est t s=1 + Alg u (t-1) avg-est t s=1 -Alg u (t) avg-est t s=1 (i) ≤η + Lt 2M t |A| =η + 2LM |A|,
Published as a conference paper at ICLR 2026 π (t+1) and π ′ (t+1) respectively, we have
λ t s=1 u (s) -∇ψ π (t+1) , π ′ (t+1) -π (t+1) ≤ 0 λ t s=1 u ′ (s) -∇ψ π ′ (t+1) , π (t+1) -π ′ (t+1) ≤ 0.
By summing them up and rearranging the terms, we have
λ t s=1 u ′ (s) -λ t s=1 u (s) , π ′ (t+1) -π (t+1) ≥ ∇ψ π ′ (t+1) -∇ψ π (t+1) , π ′ (t+1) -π (t+1) .
Since ψ is c 0 -strongly convex, we have
ψ π (t+1) ≥ ψ π ′ (t+1) + ∇ψ π ′ (t+1) , π (t+1) -π ′ (t+1) + c 0 2 π (t+1) -π ′ (t+1) 2 ψ π ′ (t+1) ≥ ψ π (t+1) + ∇ψ π (t+1) , π ′ (t+1) -π (t+1) + c 0 2 π (t+1) -π ′ (t+1) 2 .
By summing them up and rearranging the terms, we have
∇ψ π ′ (t+1) -∇ψ π (t+1) , π ′ (t+1) -π (t+1) ≥ c 0 π (t+1) -π ′ (t+1) 2 .
Therefore,
c 0 π (t+1) -π ′ (t+1) 2 ≤ λ t s=1 u ′ (s) -λ t s=1 u (s) , π ′ (t+1) -π (t+1) (i) ≤ λ t s=1 u ′ (s) -λ t s=1 u (s) • π ′ (t+1) -π (t+1) ,
where (i) is by Hölder's Inequality. Then, s) , so that (FTRL) satisfies Assumption 6.1 with L = λ c0 . Furthermore, note that the results above also hold for sequences of utility vectors of different lengths (not necessarily equal to length t simultaneously). As a result, we have
π (t+1) -π ′ (t+1) ≤ 1 c 0 λ t s=1 u ′ (s) -λ t s=1 u(
π (t+1) -π (t) ≤ λ c 0 t s=1 u (s) - t-1 s=1 u (s) = λ c 0 u (t) ≤ λ c 0 |A|, for any t ∈ {0} ∪ [T -1], which implies that η = λ c0
|A| in Assumption 7.1 for (FTRL).
this section cite: []

Section: M CONCLUSION AND LIMITATIONS
In this paper, we studied online learning and equilibrium computation with ranking feedback, which is particularly relevant to application scenarios with humans in the loop. Focusing on the classical (external-)regret metric, we designed novel hardness instances to show that achieving sublinear regret can be hard in general, in a few different ranking models and feedback settings. We then developed new algorithms to achieve sublinear regret under an additional assumption on the sublinear variation of the utility, leading to an equilibrium computation result in the repeated game setting. Finally, we justify the effectiveness of our approach by simulating routing the user's query to the optimal LLM. We believe our work paves the way for promising avenues of future research. For example, it would be interesting to close the gap between the lower-bound and the positive result for AvgUtil Rank under bandit feedback, i.e., either show the hardness when τ is a constant or achieve sublinear regret for constant τ without Assumption 4.2. Moreover, applying our algorithms to real-world datasets with ranking feedback, such as ride-sharing and match-dating, would also be of great interest.
this section cite: []

Section: References
Ref_id:b0 Title: Phi-4 technical report Year: (2024)
Ref_id:b1 Title: Fast convergence of fictitious play for diagonal payoff matrices Year: (2021)
Ref_id:b2 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b3 Title: The architecture of platforms: A unified view. Platforms, Markets and Innovation Year: (2009)
Ref_id:b4 Title: Beyond log 2 (t) regret for decentralized bandits in matching markets Year: ()
Ref_id:b5 Title: Fictitious play in 2 × n games Year: (2005)
Ref_id:b6 Title: From external to internal regret Year: (2007)
Ref_id:b7 Title: Iterative solution of games by fictitious play Year: (1951)
Ref_id:b8 Title: Value-incentivized preference optimization: A unified approach to online and offline rlhf Year: ()
Ref_id:b9 Title: Prediction, learning, and games Year: (2006)
Ref_id:b10 Title: A counter-example to Karlin's strong conjecture for fictitious play Year: (2014)
Ref_id:b11 Title: Dueling bandits: From two-dueling to multi-dueling Year: (2020)
Ref_id:b12 Title: Exploration-driven policy optimization in RLHF: Theoretical insights on efficient data utilization Year: ()
Ref_id:b13 Title: The Llama 3 herd of models Year: (2024)
Ref_id:b14 Title: Contextual dueling bandits Year: (2015)
Ref_id:b15 Title: Efficient exploration for LLMs Year: ()
Ref_id:b16 Title: Decentralized and uncoordinated learning of stable matchings: A game-theoretic approach Year: ()
Ref_id:b17 Title: Consistency and cautious fictitious play Year: (1995)
Ref_id:b18 Title: The theory of learning in games Year: (1998)
Ref_id:b19 Title: College admissions and the stability of marriage Year: (1962)
Ref_id:b20 Title: A simple adaptive procedure leading to correlated equilibrium Year: (2000)
Ref_id:b21 Title: Introduction to online convex optimization Year: (2016)
Ref_id:b22 Title: Mm algorithms for generalized Bradley-Terry models. The annals of statistics Year: (2004)
Ref_id:b23 Title: Alec Radford, et al. GPT-4o system card Year: (2024)
Ref_id:b24 Title: Learning equilibria in matching markets from bandit feedback Year: ()
Ref_id:b25 Title: Engineering social learning: Information design of time-locked sales campaigns for online platforms Year: (2022)
Ref_id:b26 Title: Competing bandits in matching markets Year: (2020)
Ref_id:b27 Title: Bandit learning in decentralized matching markets Year: (2021)
Ref_id:b28 Title: The power of regularization in solving extensive-form games Year: ()
Ref_id:b29 Title: Individual choice behavior Year: (1959)
Ref_id:b30 Title: Bayesian incentive-compatible bandit exploration Year: (2015)
Ref_id:b31 Title: Bandits with ranking feedback Year: ()
Ref_id:b32 Title: Fictitious play property for games with identical interests Year: (1996)
Ref_id:b33 Title: Dueling posterior sampling for preference-based reinforcement learning Year: (2020)
Ref_id:b34 Title: Training language models to follow instructions with human feedback Year: ()
Ref_id:b35 Title: Principled rlhf from heterogeneous feedback via personalization and preference aggregation Year: (2024)
Ref_id:b36 Title: The analysis of permutations Year: (1975)
Ref_id:b37 Title: An iterative method of solving a game Year: (1951)
Ref_id:b38 Title: Versatile dueling bandits: Best-of-both-world analyses for online learning from preferences Year: (2022)
Ref_id:b39 Title: Combinatorial bandits with relative feedback Year: (2019)
Ref_id:b40 Title: Adversarial dueling bandits Year: ()
Ref_id:b41 Title: Dueling RL: Reinforcement learning with trajectory preferences Year: ()
Ref_id:b42 Title: Fictitious play in "one-against-all" multi-player games Year: (1999)
Ref_id:b43 Title: Learning optimal stable matches in decentralized markets with unknown preferences Year: ()
Ref_id:b44 Title: Two-sided learning in decentralized matching markets Year: (2024)
Ref_id:b45 Title: Online learning and online convex optimization Year: (2012)
Ref_id:b46 Title: Linear last-iterate convergence in constrained saddle-point optimization Year: ()
Ref_id:b47 Title: Exploratory preference optimization: Harnessing implicit Q*-approximation for sample-efficient RLHF Year: ()
Ref_id:b48 Title: Preference-based reinforcement learning with finite-time guarantees Year: (2020)
Ref_id:b49 Title: Qwen3 technical report Year: (2025)
Ref_id:b50 Title: The K-armed dueling bandits problem Year: (2012)
Ref_id:b51 Title: Self-exploring language models: Active preference elicitation for online alignment Year: (2025)
Ref_id:b52 Title: Principled reinforcement learning with human feedback from pairwise or k-wise comparisons Year: ()
Ref_id:b53 Title: Fine-tuning language models from human preferences Year: (2019)
Ref_id:b54 Title: Regret minimization in games with incomplete information Year: (2007)
