Title: MULTIPLAYER NASH PREFERENCE OPTIMIZATION
Abstract: Reinforcement learning from human feedback (RLHF) has emerged as the standard paradigm for aligning large language models with human preferences. However, reward-based methods grounded in the Bradley-Terry assumption struggle to capture the non-transitive and heterogeneous nature of real-world preferences. To address this, recent studies have reframed alignment as a two-player Nash game, giving rise to Nash learning from human feedback (NLHF). While this perspective has inspired algorithms such as INPO, ONPO, and EGPO that offer strong theoretical and empirical guarantees, they remain fundamentally restricted to two-player interactions, introducing a single-opponent bias that fails to capture the full complexity of realistic preference structures. This work introduces Multiplayer Nash Preference Optimization (MNPO), a novel framework that generalizes NLHF to the multiplayer regime. It formulates alignment as an n-player game, where each policy competes against a population of opponents while being regularized toward a reference model. We demonstrate that MNPO inherits the equilibrium guarantees of two-player methods while enabling richer competitive dynamics and improved coverage of diverse preference structures. Comprehensive empirical evaluation shows that MNPO consistently outperforms existing NLHF baselines on instructionfollowing benchmarks, achieving superior alignment quality under heterogeneous annotator conditions and mixed-policy evaluation scenarios. Together, these results establish MNPO as a principled and scalable framework for aligning LLMs with complex, non-transitive human preferences. Code is available at https://  github.com/smiles724/MNPO.

Section: INTRODUCTION
Large language models (LLMs) have achieved remarkable progress in instruction following and open-ended reasoning through reinforcement learning from human feedback (RLHF) (Christiano et al., 2017). Traditional RLHF pipelines built upon the Bradley-Terry (Bradley & Terry, 1952) model have enabled widely deployed systems (e.g., InstructGPT (Ouyang et al., 2022), Claude (Bai et al., 2022), and Gemini (Team et al., 2023)), but they assume transitive preferences and scalar reward functions. Recent empirical studies reveal that human preferences often exhibit non-transitive patterns and heterogeneous structures that violate these assumptions (Ethayarajh et al., 2024;Wu et al., 2024).
This has motivated game-theoretic formulations of alignment that treat preference optimization as finding Nash equilibria in games defined by general preference oracles (Munos et al., 2023). In the Nash learning from human feedback (NLHF) paradigm, a well-aligned policy constitutes an optimal strategy that competing policies cannot exploit, thereby achieving strategic optimality rather than mediocrity. Subsequent work has explored this paradigm through no-regret learning (Zhang et al., 2025b), optimistic mirror descent (Zhang et al., 2025a), and extragradient updates (Zhou et al., 2025), thereby advancing both theoretical guarantees and empirical stability relative to traditional RLHF. Despite these advances, existing NLHF formulations remain constrained to two-player settings, where a single policy competes against one opponent. However, real-world preference alignment rarely resembles a two-agent interaction. Instead, it often involves a mixture of annotators, heterogeneous evaluation criteria, multiple reward models, or even a sequence of historical model checkpoints-creating inherently multi-source, and sometimes conflicting, preference signals (Freund & Schapire, 1999). Reducing this complex landscape to a single opponent introduces a bottleneck: the policy is optimized against only one distribution at a time, resulting in oscillatory behavior, narrow exploration, and a brittle approximation of the broader preference population.
These limitations highlight the need for a framework that explicitly models alignment as competition against an entire population rather than a single synthetic opponent. In this work, we show that extending to the multiplayer setting (Freund & Schapire, 1999) provides a principled mean-field approximation that reduces gradient variance, stabilizes optimization, and more precisely captures diverse preference structures. Crucially, when all policies share the same preference oracle-as naturally occurs when competing against historical versions of a single policy trajectory-the resulting symmetric game admits strong theoretical guarantees via multiplicative weights update.
We address these challenges by proposing Multiplayer Nash Preference Optimization (MNPO), a principled framework that generalizes two-player preference optimization to n-player games. In MNPO, each policy simultaneously competes against a population of other policies while being regularized toward a reference model, producing a competitive equilibrium that balances performance against the population with adherence to a trusted baseline. Our contributions are threefold:
• Theoretical Framework: We establish that MNPO with homogeneous preference oracles admits natural equilibrium characterizations, including well-defined Nash policies and duality gaps that measure alignment quality. We prove that MNPO inherits the desirable convergence properties of existing two-player formulations while enabling richer equilibrium dynamics.
• Algorithmic Innovation: We introduce TD-MNPO, where opponent sets evolve adaptively using weighted combinations of historical policies, naturally yielding provable convergence guarantees. We further propose a heterogeneous extension (HT-MNPO) that, although it lacks formal guarantees, demonstrates strong empirical performance across diverse preference sources.
• Empirical Validation: Through comprehensive experiments on instruction-following and reasoning benchmarks, we demonstrate that MNPO consistently outperforms existing NLHF baselines, particularly excelling in scenarios involving diverse preferences and complex evaluation criteria.
Our analysis reveals that MNPO provides a unifying perspective on RLHF, subsuming many existing methods as special cases while offering improved robustness in multi-agent alignment scenarios. By bridging recent advances in NLHF with the practical demands of aligning LLMs with diverse, potentially non-transitive human preferences, MNPO establishes a scalable foundation for nextgeneration alignment techniques.
this section cite: ['b8', 'b4', 'b37', 'b2', 'b50', 'b18', 'b58', 'b35', 'b73', 'b20', 'b20']

Section: RLHF PRELIMINARIES
Notation. x ∈ X denotes a prompt and X is the prompt space.
x is assumed to be sampled from a fixed but unknown distribution d 0 . An LLM is characterized by a policy π : X → ∆(Y) that takes a prompt x as input and outputs a distribution over the response space Y. The response y ∈ Y is then sampled from the distribution π(• | x).
this section cite: []

Section: Bradley-Terry Model Assumption.
The prevalent RLHF framework (Christiano et al., 2017;Ouyang et al., 2022) assumes the Bradley-Terry model. It assumes that there exists a reward function r * such that for any x ∈ X and y 1 , y 2 ∈ Y, we have
P y 1 ≻ y 2 | x = exp(r * (x,y 1 )) exp(r * (x,y 1 ))+exp(r * (x,y 2 )) = σ r * x, y 1 -r * x, y 2 .
After learning a reward function R(•, •), RLHF algorithms aim to maximize the following KL-regularized objective with preference optimization RL algorithms such as PPO (Schulman et al., 2017):
J(π) = E x∼d0 E y∼π(•|x) [R(x, y)] -τ KL (π(• | x)∥π ref (• | x)) .(1)
Here π ref is the reference policy, which is usually a supervised fine-tuned LLM, and τ > 0 is the regularization parameter. By maximizing the objective, the obtained policy simultaneously achieves a high reward and stays close to π ref , which can mitigate reward hacking (Tien et al., 2022;Skalse et al., 2022) to some extent.
this section cite: ['b8', 'b37', 'b45', 'b52', 'b47']

Section: General Preference Oracle.
The Bradley-Terry model assumption may not always hold in practice.
Recent studies (Munos et al., 2023;Calandriello et al., 2024;Ye et al., 2024;Zhang et al., 2025b;a) have instead directly considered the general preference distribution P without imposing additional assumptions, framing the preference optimization problem as a two-player game. They assume the existence of a preference oracle P :
X × Y × Y → [0, 1].
It can be queried to obtain binary preference signals as z ∼ Ber P y 1 ≻ y 2 | x , where z = 1 indicates that y 1 is preferred to y 2 , and z = 0 indicates the opposite. The preference distribution is introduced as
λ P (x, y, y ′ ) = (y, y ′ ) • I [U < P (y ≻ y ′ | x)] + (y ′ , y) • I [U ≥ P (y ≻ y ′ | x)]
, where U ∼ Uniform(0, 1) is a random variable. Given two policies π 1 and π 2 , LLMs are aligned using this general preference oracle, and the game objective is written as:
J (π 1 , π 2 ) = E x∼d0 [E y1∼π1,y2∼π2 [P (y 1 ≻ y 2 | x)] -τ KL (π 1 ∥π ref ) + τ KL (π 2 ∥π ref )] (2
)
where the max-player π 1 aims to maximize the objective, and the min-player π 2 aims to minimize the objective. The goal of both players is to maximize their winning rates against the opponent while not deviating too far from π ref , which shares a similar spirit with the objective of Eq. 1.
Nash Policy and Duality Gap. Without loss of generality, we restrict our attention to the policy class Π containing the policies with the same support set as π ref .
The Nash equilibrium of the game in Eq. 2 is then defined as:
π * 1 , π * 2 := argmax π1∈Π argmin π2∈Π J (π 1 , π 2 ) .(3)
Due to the symmetry of the two players, their Nash policies are unique and coincide, meaning Ye et al., 2024). Interestingly, J (π * , π) ≥ 0.5 always holds for ∀π ∈ Π, as J (π * , π * ) = 0.5 indicates that π * is the best response against itself. To quantify how well a policy π approximates the Nash policy π * , we define the duality gap as:
π * 1 = π * 2 = π * (
DualGap(π) := max
π1 J (π 1 , π) -min π2 J (π, π 2 ) .(4)
The duality gap is nonnegative and DualGap(π) = 0 if and only if π = π * . Hence, our goal is to find a policy that minimizes the duality gap. Once we achieve DualGap(π) ≤ ϵ, we say that π is an ϵ-approximate Nash policy.
this section cite: ['b35', 'b5', 'b63', 'b63']

Section: METHOD

this section cite: []

Section: RLHF AS MULTIPLAYER GAMES
To extend the two-player preference optimization objective to a multiplayer setting {π i } n i=1 , we consider a framework where each policy seeks to maximize its average preference probability against all other policies while regularizing toward a reference policy. We first focus on the homogeneous multiplayer setting where all players share the same preference oracle. This symmetric structure underpins our theoretical guarantees.
this section cite: []

Section: Plackett-Luce Reward Learning Assumption.
To generalize the maximum-likelihood reward learning objective for the Bradley-Terry model to one-vs-many comparisons, we adopt the Plackett-Luce framework. Specifically, we replace the pairwise logistic term log σ R x, y 1 -R x, y 2 with a softmax over multiple alternatives, extending the Bradley-Terry model to accommodate listwise comparisons. This Plackett-Luce model (Debreu, 1960;Plackett, 1975) maintains the interpretability of reward-based preferences while scaling to more complex decision-making scenarios. Formally, given a learned reward function R(x, y) and a dataset D containing tuples x, y 1 , y 2 , . . . , y k , where y 1 , y 2 , . . . , y k is a pool of k to-be-ranked items, the probability that y i is preferred over the remaining pool of entities y j j̸ =i under the Plackett-Luce model is:
P y i ≻ y j j̸ =i x = exp(R x, y i ) exp(R (x, y i )) + j̸ =i exp(R (x, y j )) .(5)
The corresponding negative log-likelihood for a single comparison is: -log P y i ≻ y j j̸ =i x = log exp R x, y i + j̸ =i exp R x, y j -R x, y i . Consequently, the generalized reward learning objective for learning R becomes:
arg max R∈R E (x,{y 1:k })∼D E y i ∈{y 1:k }   R x, y i -log   exp R x, y i + j̸ =i exp R x, y j     Per-comparison log-likelihood .(6)
Several key observations arise from this formulation. First, when k = 2 for a one-vs-one comparison, Eq. 6 reduces to the vanilla Bradley-Terry objective, given by log σ (R (x, y) -R (x, y ′ )), since σ(a) = e a 1+e a . Second, this objective maximizes the gap between the reward of y i and the log-sumexp (LSE) of all alternatives, effectively applying a soft maximum over competitors. This penalizes cases where y i fails to dominate the collective "strength" of the dispreferred items y j j̸ =i .
Homogeneous Multiplayer Preference Oracle. We consider the case where all n players share the same universal preference oracle P : X × Y × {Y} n-1 → [0, 1]. It directly compares y i with a group of responses {y j } j̸ =i and outputs binary preference signals z ∼ Ber P y i ≻ y j j̸ =i | x . Then, each policy π i competes against the other n -1 players, and the objective function becomes:
J πi, {πj} j̸ =i = E x∼d 0 E y i ∼π i ,{y j |y j ∼π j } j̸ =i P y i ≻ y j j̸ =i x -τ KL (πi(• | x)∥πref(• | x)) .(7)
Here, each policy π i maximizes its expected preference against all other players {π j } j̸ =i while remaining close to the reference policy π ref via a KL penalty. The KL term, weighted by τ , prevents over-optimization and ensures behavioral stability. All policies are updated concurrently: at each step, player i improves its own objective J, leading the population toward a competitive equilibrium that balances performance against opponents and adherence to π ref .
Eq. 7 exhibits several important properties. (i) Symmetric treatment: All policies are treated equally and compete in a symmetric manner, ensuring that π * 1 = π * 2 = • • • = π * n at equilibrium. (ii) Decentralized optimization: Each policy's update depends only on its own actions and the aggregate behavior of its opponents, thereby avoiding complex interdependencies. (iii) Generalization of the two-player case: When n = 2, each policy's objective reduces to maximizing its pairwise preference probability subject to its own KL penalty, as shown in Eq. 2.
Nash Equilibrium and Duality Gap. In this n-player game with homogeneous preference oracles, the Nash equilibrium is a policy π * where no player can improve their respective objectives by unilaterally deviating. Formally, for all π i ∈ Π :
J π * i , {π * j } j̸ =i ≥ J π i , {π * j } j̸ =i , ∀i ∈ {1, . . . , n}.(8)
To quantify how far a given policy π is from the Nash policy π * , we define the unified duality gap in the multiplayer setting. For player i with opponents fixed as
O π = {π j } n-1 j=1 , DualGap(π) = max π ′ ∈Π J(π ′ , O π ) -J(π, O π ).(9)
This measures the maximum gain player i could achieve by unilaterally switching to an alternative strategy while all opponents remain fixed. A Nash equilibrium is characterized by DualGap(π * ) = 0, and DualGap(π) ≤ ϵ indicates that π is an ϵ-approximate Nash policy.
this section cite: ['b13', 'b40']

Section: Multiplayer Nash Preference Optimization.
There are well-known algorithms that approximately solve the Nash equilibrium in a constant-sum multiplayer game. In this work, we follow (Freund & Schapire, 1999) to establish an iterative framework that can asymptotically converge to the optimal policy on average. Given a learning rate η of online mirror descent update, we start with a theoretical analysis that conceptually solves the homogeneous multiplayer game as followsfoot_0 :
π (t+1) i (y | x) ∝   j̸ =i π (t) j (y | x)   1 n-1 exp   η n -1 j̸ =i P y ≻ π (t) j | x   .(10)
This iterative framework starts from a base policy π (0) such as π ref .
In each iteration, the updated policy π (t+1) is obtained from the reference policy π (t) following the multiplicative weight update.
Particularly, a response y should have a higher probability weight if it has a higher average advantage over the current policy π (t) . Notably, Eq. 10 provides convergence guarantees to a Nash equilibrium in this homogeneous circumstance, ensuring that the average policy π t) converges to an ϵ-approximate Nash equilibrium with ϵ = O(1/ √ T ) regret bound (Hart & Mas-Colell, 2000).
(T ) = 1 T T t=1 π (
Eq. 10 is equivalent to π (t+1) i (y | x) = j̸ =i π (t) j (y | x) 1 n-1 exp η n-1 y ≻ π (t) j | x /Z π (t) (x), where Z π (t) (x)
is the normalization factor (a.k.a, the partition function). Then, for any fixed x and y, each ideal update policy π (t+1) i should satisfy:
1 n -1 j̸ =i log π (t+1) i (y | x) π (t) j (y | x) = η n -1 j̸ =i P y ≻ π (t) j | x -log Z π (t) . (11
)
Note that direct computation of π (t+1) involves a normalization factor, which is intractable for the exponentially large response space Y. To avoid computing this normalization factor, we consider the logarithmic ratio between response pair y and y ′ , and define the function h t (π, y, y ′ ) as:
ht π, y, y ′ = log π(y | x) π(y ′ | x) - 1 n -1 j̸ =i log π (t) j (y | x) π (t) j (y ′ | x) .(12)
From Eq. 11, we know that the following equality holds for any response pair y, y ′ ∈ Supp (π ref ):
ht π (t+1) , y, y ′ = η n -1 j̸ =i P y ≻ π (t) j | x -P y ′ ≻ π (t) j | x (13
)
Based on this observation, we define the loss function L t (π) and update the policy π (t+1) as:
π (t+1) ← argmin π Ey w ,y l ∼D t     ht (π, yw, y l ) - η n -1 j̸ =i P y ≻ π (t) j -P y ′ ≻ π (t) j   2   L t (π) (14)
It is clear to see that π (t+1) is the minimizer of L t (π) since L t π (t+1) = 0. Furthermore, in the following lemma, we show that π (t+1) is the unique minimizer of L t within the policy class Π. Lemma 1. For each t ∈ [T ], π t+1 in Eq. 14 is the unique minimizer of L t (π) within Π.
this section cite: ['b20', 'b23']

Section: Published as a conference paper at ICLR 2026
The proof is deferred to Appendix F.1. Moreover, we replace the tricky term P y ≻ π (t) j with a hyperparameter η and propose the following loss to bypass it:
L ′ t (π) = E y,y ′ ∼πt, yw,y l ∼λ P (y,y ′ ) h t (π, y w , y l ) - 1 2η 2 . (15
)
Proposition 1. For any policy π ∈ Π and any iteration t ∈ [T ], L ′ t (π) is equivalent to L t (π), differing only by an additive constant that is independent of π.
See the proof in Appendix F.2. Here, the response pair (y, y ′ ) is directly sampled from the current policy π (t) , which is crucial for the equivalence between L ′ t (π) and L t (π).
this section cite: []

Section: MULTIPLAYER NASH PREFERENCE OPTIMIZATION
Reward-Enhanced MNPO. While our framework is motivated by moving beyond the limitations of purely scalar reward-based approaches, this does not preclude the beneficial incorporation of reward information when available. The key distinction lies in how rewards are utilized: rather than relying solely on reward-based rankings with implicit transitivity assumptions (as in classical RLHF), MNPO can leverage reward information as auxiliary guidance while maintaining the flexibility to handle non-transitive preferences through its game-theoretic structure.
Reward-aware preference optimization (RPO) (Sun et al., 2025) demonstrates how quantitative reward signals can complement qualitative preference comparisons. It aligns learned implicit preference models with explicit reward models that provide graded assessments of response quality. This shift allows MNPO to move beyond binary preference margins and instead minimize discrepancies between the learned reward function r π θ (x, y) and the target reward model r ⋆ (x, y). Concretely, RPO defines the loss over preference pairs as:
L D RPO π θ , x, y 1 , y 2 | r ⋆ , πref, β, η := D rπ θ x, y 1 -rπ θ x, y 2 ∥ηr ⋆ x, y 1 -ηr ⋆ x, y 2 , (16
)
where x, y 1 , y 2 represents a preference pair and the distance metric D : R × R → R * measures alignment between the model's implicit reward differences and the scaled reference reward differences.
The hyperparameters η ∈ R * and β ∈ R * control the reward scale and regularization, respectively. This formulation directly encourages the policy π θ to internalize human-aligned reward values, bridging qualitative preference optimization with quantitative reward modeling.
Importantly, the loss function Eq. 15 can be interpreted as a special case of RPO under a squared distance metric D sq . Specifically,
L ′ t (π) employs a learned reward model r π θ (x, y) := E πj log π(y|x) πj (y|x)
and a reference reward model gap δ r ⋆ := η r ⋆ x, y 1 -r ⋆ x, y 2 = 1 2η . This connection highlights that integrating reward awareness into multiplayer preference games enhances stability, interpretability, and alignment fidelity.
Time-dependent Multiplayers A central challenge in multiplayer optimization lies in defining and updating the set of opponent players π (t) j n-1 j=1 in Eq. 13. Inspired by recent iterative preference optimization methods such as DNO (Rosset et al., 2024), SPIN (Chen et al., 2024), and INPO (Zhang et al., 2025b), which typically rely on past policy iterations (e.g., π ref and π (t-1) ) to construct opponents, we adopt a time-dependent opponent selection mechanism.
At any iteration t, we construct the opponent set from a mixture of recent time-indexed historical policies {π t-j } t j=0 (n ≤ t + 1), weighted by coefficients λ j with λ j ∈ [0, 1] and optionally j λ j ≤ 1. The resulting time-dependent MNPO (TD-MNPO) loss is formulated as follows:
L t,D TD (π | β, {λj}, η) = E y,y ′ ∼π, yw ,y l ∼λ P (y,y ′ ) D log π(yw | x) π(y l | x) - n-2 j=0 λj log πt-j(yw | x) πt-j(y l | x) ηδ ⋆ , (17
)
where δ ⋆ encodes the target reward gap. By blending multiple past policies, this formulation stabilizes training, mitigates overfitting to transient fluctuations, and preserves temporal consistency.
Table 1: Time-dependent MNPO recovers many existing offline or online preference optimization algorithms. We denote the target reward gap as δ r ⋆ := η (r ⋆ (x, y 1 ) -r ⋆ (x, y 2 )) . D sq and D bwd represent the squared distance and backward Bernoulli KL divergence, respectively.
Algorithm Num. Players Opponents Importance Weights Dist. Target Reward Gap
SimPO n = 1 - - D bwd ∞ CPO n = 1 - - D bwd ∞ DPO n = 2 π ref λ j = 1 D bwd ∞ Distill-DPO n = 2 π ref λ j = 1 D sq ∞ DNO n = 2 π t λ j = 1 D bwd ∞ SPIN n = 2 π t λ j = β D bwd ∞ SPPO n = 2 π t λ j = 1 D sq η P (y ≻ π t | x) -1 2 IPO n = 2 π ref λ j = 1 D sq 1 2τ INPO n = 3 π t , π ref λ j = τ η , if j = t η-τ η , if j = 1 D sq 1 2τ
Connections to Existing RLHF. This unified MNPO formulation in Eq. 17 reveals that many preference optimization algorithms can be recovered as special cases by varying the number of players n, choice of opponents O π , distance metric D, and target reward gap δ ⋆ . For instance, DPO emerges by setting n = 2, O π = π ref , and λ j = 1.
Table 1 summarizes these reductions, showing how time-dependent MNPO unifies offline and online preference optimization under one principled framework. We provide a broader overview of RLHF objectives in Appendix E. Compared to static reference-based approaches, the reward-aware multiplayer formulation offers: (i) Smoother policy evolution. Unlike methods that rely solely on the most recent past policy, MNPO gradually incorporates multiple past policies, preventing abrupt shifts and stabilizing policy updates. (ii) Greater robustness. By combining historical opponents with a weighted mixture, MNPO mitigates the risk of overfitting to transient fluctuations in recent iterations. (iii) Unified interpretation. TD-MNPO seamlessly extends existing approaches into a unified formulation, allowing for flexible adaptation to different training scenarios. (iv) Stable convergence. The weighting scheme ensures that recent policies exert greater influence while preserving the broader learning trajectory, thereby leading to more stable convergence. By dynamically leveraging historical policies as opponent players, TD-MNPO enhances preference optimization, making it more adaptable and robust in evolving learning environments. The pseudo-algorithm is described in Appendix B.
this section cite: ['b48', 'b43', 'b7']

Section: MULTIPLAYER GAME WITH HETEROGENEOUS PREFERENCE ORACLES
Multiplayers with Heterogeneous Preference Oracles. In many real-world alignment scenarios, preference signals originate from multiple heterogeneous sources-such as annotators with different evaluation criteria or distinct reward models trained for separate dimensions of quality (e.g., helpfulness (Tan et al., 2025), safety (Dai et al., 2023), conciseness (Dumitru et al., 2025)). We therefore generalize MNPO to the heterogeneous setting by replacing the historical mixture over past policies with a mixture over opponent policies associated with distinct preference oracles.
Concretely, each player π i is paired with a distinct reward model r i (x, y), inducing a player-specific preference oracle
P i : X × Y × {Y} n-1 → [0, 1]. The objective J i π i , {π j } j̸ =i becomes E x∼d0 E y i ∼πi,y j ∼πj P i y i ≻ y j j̸ =i | x -τ KL (π i (• | x)∥π ref (• | x))
, which preserves the multiplayer structure while allowing each agent to learn under its own notion of preference.
Game-Theoretic Properties. When P i ̸ = P j , the resulting game is general-sum and lacks the symmetry needed for formal Nash equilibrium guarantees. In this setting, each player has its own objective J i with its own oracle P i , breaking the constant-sum structure that underpins the convergence guarantees of multiplicative weights update. Consequently, the iterative framework in Eq. 10 does not have formal convergence to the Nash equilibrium in the heterogeneous case (Daskalakis et al., 2009;Hart & Mas-Colell, 2000).
To quantify the quality of a policy profile, we define a player-specific duality gap: DualGap i (π i ) = Published as a conference paper at ICLR 2026
max π ′ i ∈Π J i (π ′ i , O π ) -J i (π i , O π ),
where O π = {π j } j̸ =i are the fixed opponent policies. This measures player i's incentive to deviate from the current strategy. We consider the policy profile {π i } n i=1 to be near a stationary point when max i DualGap i (π i ) ≤ ϵ, indicating that no player has a strong incentive to deviate unilaterally. Nevertheless, the algorithmic framework remains natural and principled: each policy optimizes with respect to the current opponent distribution using its own oracle and empirically yields effective solutions.
Heterogeneous MNPO. Let δ ⋆ i denote the target reward gap induced by reward model r i , following RPO's reward-aware interpretation. The heterogeneous MNPO (HT-MNPO) for player i becomes:
L i,D HT (πi | β, {λj}, η) = E y,y ′ ∼π i ,yw ,y l ∼λ P i (y,y ′ ) D   log πi(yw | x) πi(y l | x) - j̸ =i λj log πj(yw | x) πj(y l | x) ηδ ⋆ i   , (18
)
where the mixture is over opponent policies {π j } j̸ =i and {λ j } are importance weights. Eq. 18 retains the equilibrium-seeking nature of MNPO, but each policy now internalizes a reward-gap signal specific to its own reward model. As a result, MNPO can align with heterogeneous or even conflicting evaluators, potentially yielding a population equilibrium that balances multiple dimensions of quality.
As empirically demonstrated later, this heterogeneous formulation achieves strong performance in multi-reward-model scenarios, suggesting that it can find effective stationary points even without formal equilibrium guarantees. The pseudo-algorithm is illustrated in Appendix B.
this section cite: ['b49', 'b11', 'b17', 'b12', 'b23']

Section: EXPERIMENTAL SETUP
Models and Training Settings. We implement an online RLHF framework (Dong et al., 2024) with Gemma-2-9B-it (Team et al., 2024) as the base model. Our MNPO training consists of T = 3 iterations, where each iteration generates responses from the current policy on a fresh prompt set and updates the policy using preference feedback. To eliminate the need for costly human annotations, we employ the reward model ArmoRM-Llama3-8B-v0.1 (Wang et al., 2024a) to provide preference signals for TD-MNPO. To simulate heterogeneous preference oracles, we select Skywork-Reward-V2-Llama-3.1-8B (Liu et al., 2025) and Athene-RM-8B (Frick et al., 2024) as additional reward models for HT-MNPO. Hyperparameter optimization is critical, as optimal configurations vary across base models and across iterations of the same model. Through empirical analysis, we find that maintaining β within the range [0.01, 10] consistently produces strong results. Furthermore, we observe that gradually increasing β throughout training effectively mitigates training degradation while enabling continued model improvement. Complete implementation details and hyperparameter specifications are provided in Appendix C.
this section cite: ['b14', 'b51', 'b31', 'b21']

Section: Evaluation Benchmarks.
We evaluate primarily on three widely used open-ended instructionfollowing benchmarks: MT-Bench (Zheng et al., 2023), AlpacaEval 2 (Li et al., 2023), and Arena-Hard v0.1 (Li et al., 2024). As suggested by Dubois et al. (2024), we report the win rate (WR) for Arena-Hard and the length-controlled (LC) WR for AlpacaEval 2, as judged by GPT-5-mini rather than the outdated GPT-4 Turbo (Preview-1106).
Since RLHF alignment is known to sometimes degrade reasoning, calibration, and factual accuracy (Ouyang et al., 2022;Dong et al., 2024), we further assess performance on a broader set of 11 academic benchmarks. These benchmarks span multiple abilities, including explicit instruction following (Zhou et al., 2023), general knowledge (Clark et al., 2018;Rein et al., 2024;Hendrycks et al., 2020), commonsense reasoning (Sakaguchi et al., 2021;Lin et al., 2021;Zellers et al., 2019), and math/coding problem-solving (Lewkowycz et al., 2022;Chen et al., 2021).
In addition, we compare MNPO with a group of open-source LLMs, including SmolLM3-3B (Bakouch et al., 2025)
this section cite: ['b71', 'b71', 'b28', 'b16', 'b37', 'b14', 'b72', 'b9', 'b42', 'b24', 'b44', 'b30', 'b67', 'b27', 'b6', 'b3']

Section: EMPIRICAL RESULTS

this section cite: []

Section: Instruction-Following and Preference Alignment.
Table 2 presents the performance of MNPO compared to existing preference optimization methods on three widely used instruction-following benchmarks: AlpacaEval 2.0 (Length-Controlled Win Rate, %), Arena-Hard (Win Rate, %), and MT-Bench (Score/10). All models were evaluated using GPT-5-mini as the judge. MNPO consistently outperforms all baseline methods across all three benchmarks. On AlpacaEval 2.0, MNPO achieves a score of 57.27, improving by 2.92 points over DPO (54.35), 2.11 points over SimPO (55.16), 1.30 points over SPPO (55.97), and 1.18 points over INPO (56.09). The improvements are even more pronounced on Arena-Hard, where MNPO scores 52.26, compared with the next-best method, INPO, at 48.03, representing a 4.23-point improvement. Notably, on this challenging benchmark, MNPO not only outperforms other preference optimization algorithms but also competes favorably with much larger open-source, fine-tuned models and even the latest closed-source models. It surpasses prominent models such as Tulu-2-DPO (70B) and Mixtral-IT (141B). On MT-Bench, MNPO achieves 7.03, outperforming all baselines, with the closest competitor being INPO at 6.95. These results demonstrate that the multiplayer formulation in MNPO provides significant advantages for instruction-following tasks. The consistent improvements across all benchmarks suggest that the framework's ability to accommodate diverse preferences and non-transitive relationships yields better alignment with human expectations in open-ended generation tasks. Knowledge and Reasoning Capabilities. Table 3 evaluates model performance on academic benchmarks covering instruction following, knowledge, and commonsense reasoning. The results show that MNPO maintains strong performance across diverse cognitive tasks while achieving preference alignment. MNPO achieves the highest average score of 71.08 across all benchmarks, outperforming The method also performs competitively on instruction following (IFEval: 73.94) and maintains solid performance on knowledge benchmarks such as MMLU (75.63) and on commonsense reasoning tasks. Importantly, unlike some preference optimization methods that show degradation on certain academic benchmarks (e.g., SimPO's drop to 63.40 on TruthfulQA), MNPO maintains relatively stable performance across all domains. This suggests that the multiplayer framework helps preserve the model's foundational capabilities while improving preference alignment.
Mathematical and Coding Performance. Table 4 presents results on mathematical reasoning and coding benchmarks. MNPO achieves the highest average score of 48.10 across math and coding tasks, outperforming all baseline methods, including SPPO (47.33) and INPO (47.10). On the challenging AIME-24 benchmark, MNPO is the only method to achieve non-zero performance (3.33), while all other methods, including the SFT baseline, score 0. This demonstrates MNPO's superior capability in handling complex mathematical reasoning tasks. On HumanEval, MNPO achieves 61.59, representing the best coding performance among all methods. The results on GSM8K and Minerva-Math show that MNPO maintains competitive performance with existing methods while achieving superior results on the most challenging tasks. This pattern suggests that the multiplayer optimization framework is particularly beneficial for complex reasoning tasks that require handling multiple solution strategies.
this section cite: []

Section: RELATED WORK
Reward-Model-Based RLHF. Classical RLHF trains a reward model on human preference data and optimizes a policy with KL-regularized policy gradients (e.g., PPO) (Christiano et al., 2017;Bai et al., 2022;Schulman et al., 2017). While effective, PPO-style updates can suffer from instability and high memory cost. To address this, GRPO removes the critic to improve training stability and memory efficiency at scale (e.g., DeepSeek-R1) (Shao et al., 2024;Guo et al., 2025), and DAPO further improves sample efficiency through dynamic sampling and refined loss objectives (Yu et al., 2025). VAPO shows that value learning can strengthen RLHF under appropriate design choices (Yuan et al., 2025). Nonetheless, optimizing against imperfect reward proxies remains vulnerable to reward hacking (Weng, 2024;Wen et al., 2024).
RLHF with General Preferences. DPO bypasses reward modeling by directly optimizing a logodds margin between preferred and dispreferred responses (Rafailov et al., 2023). This idea has inspired a family of extensions, including IPO (Azar et al., 2024), KTO (Ethayarajh et al., 2024), SimPO (Meng et al., 2024), and WPO (Zhou et al., 2024), which refine the constraint, scaling, or sampling strategy. Iterative and online forms introduce exploration and continual updates (Xiong et al., 2023;Dong et al., 2024;Xie et al., 2024), but most remain limited to static pairwise supervision.
this section cite: ['b8', 'b2', 'b45', 'b46', 'b22', 'b64', 'b66', 'b57', 'b56', 'b41', 'b18', 'b34', 'b74', 'b60', 'b14', 'b59']

Section: CONCLUSION
We introduce Multiplayer Nash Preference Optimization, extending Nash learning from human feedback to multiplayer settings. Our framework admits or approximates well-defined Nash equilibria and unifies existing preference optimization methods as special cases. Empirically, MNPO outperforms baselines across instruction-following, preference-alignment, and reasoning benchmarks. These results validate that multiplayer formulations better capture heterogeneous human preferences and provide more robust alignment for LLMs.
this section cite: []

Section: References
Ref_id:b0 Title: Theoretical analysis of kl-regularized rlhf with multiple reference models Year: (2025)
Ref_id:b1 Title: A general theoretical paradigm to understand learning from human preferences Year: (2024)
Ref_id:b2 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b3 Title: SmolLM3: smol, multilingual, long-context reasoner Year: (2025)
Ref_id:b4 Title: Rank analysis of incomplete block designs: I. the method of paired comparisons Year: (1952)
Ref_id:b5 Title: Human alignment of large language models through online preference optimisation Year: (2024)
Ref_id:b6 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b7 Title: Self-play fine-tuning converts weak language models to strong language models Year: (2024)
Ref_id:b8 Title: Deep reinforcement learning from human preferences Year: (2017)
Ref_id:b9 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b10 Title: Boosting language models with high-quality feedback Year: (2023)
Ref_id:b11 Title: Safe rlhf: Safe reinforcement learning from human feedback Year: (2023)
Ref_id:b12 Title: The complexity of computing a nash equilibrium Year: (2009)
Ref_id:b13 Title: Individual choice behavior: A theoretical analysis Year: (1960)
Ref_id:b14 Title: Rlhf workflow: From reward modeling to online rlhf Year: (2024)
Ref_id:b15 Title: The llama 3 herd of models Year: (2024)
Ref_id:b16 Title: Length-controlled alpacaeval: A simple way to debias automatic evaluators Year: (2024)
Ref_id:b17 Title: Concisenessguided reinforcement learning for efficient reasoning models Year: (2025)
Ref_id:b18 Title: Model alignment as prospect theoretic optimization Year: (2024)
Ref_id:b19 Title: Robust preference optimization through reward model distillation Year: (2024)
Ref_id:b20 Title: Adaptive game playing using multiplicative weights Year: (1999)
Ref_id:b21 Title: Athene-70b: Redefining the boundaries of post-training for open models Year: (2024-07)
Ref_id:b22 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b23 Title: A simple adaptive procedure leading to correlated equilibrium Year: (2000)
Ref_id:b24 Title: Measuring massive multitask language understanding Year: (2020)
Ref_id:b25 Title: Monolithic preference optimization without reference model Year: (2024)
Ref_id:b26 Title: Tulu 3: Pushing frontiers in open language model post-training Year: (2025)
Ref_id:b27 Title: Solving quantitative reasoning problems with language models Year: (2022)
Ref_id:b28 Title: From live data to high-quality benchmarks: The arena-hard pipeline. Blog post Year: (2024)
Ref_id:b29 Title: Alpacaeval: An automatic evaluator of instruction-following models Year: (2023)
Ref_id:b30 Title: Measuring how models mimic human falsehoods Year: (2021)
Ref_id:b31 Title: Skywork-reward-v2: Scaling preference data curation via human-ai synergy Year: (2025)
Ref_id:b32 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b33 Title: The hidden link between rlhf and contrastive learning Year: (2025)
Ref_id:b34 Title: Simpo: Simple preference optimization with a referencefree reward Year: (2024)
Ref_id:b35 Title: Nash learning from human feedback Year: (2023)
Ref_id:b36 Title:  Year: (2025)
Ref_id:b37 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b38 Title: Pre-dpo: Improving data utilization in direct preference optimization using a guiding reference model Year: (2025)
Ref_id:b39 Title: Disentangling length from quality in direct preference optimization Year: (2024)
Ref_id:b40 Title: The analysis of permutations Year: (1975)
Ref_id:b41 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b42 Title: Gpqa: A graduate-level google-proof q&a benchmark Year: (2024)
Ref_id:b43 Title: Direct nash optimization: Teaching language models to self-improve with general preferences Year: (2024)
Ref_id:b44 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b45 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b46 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b47 Title: Defining and characterizing reward gaming Year: (2022)
Ref_id:b48 Title: Reward-aware preference optimization: A unified mathematical framework for model alignment Year: (2025)
Ref_id:b49 Title: Equilibrate rlhf: Towards balancing helpfulness-safety trade-off in large language models Year: (2025)
Ref_id:b50 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b51 Title: Gemma 2: Improving open language models at a practical size Year: (2024)
Ref_id:b52 Title: Causal confusion and reward misidentification in preference-based reward learning Year: (2022)
Ref_id:b53 Title: Beyond reverse kl: Generalizing direct preference optimization with diverse divergence constraints Year: (2023)
Ref_id:b54 Title: Interpretable preferences via multi-objective reward modeling and mixture-of-experts Year: (2024)
Ref_id:b55 Title: Magnetic preference optimization: Achieving last-iterate convergence for language model alignment Year: (2024)
Ref_id:b56 Title: Language models learn to mislead humans via rlhf Year: (2024)
Ref_id:b57 Title: Reward hacking in reinforcement learning. lilianweng.github.io Year: (2024-11)
Ref_id:b58 Title: Self-play preference optimization for language model alignment Year: (2024)
Ref_id:b59 Title: Exploratory preference optimization: Harnessing implicit q*-approximation for sample-efficient rlhf Year: (2024)
Ref_id:b60 Title: Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint Year: (2023)
Ref_id:b61 Title: Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation Year: (2024)
Ref_id:b62 Title: Qwen3 technical report Year: (2025)
Ref_id:b63 Title: Online iterative reinforcement learning from human feedback with general preference model Year: (2024)
Ref_id:b64 Title: Dapo: An open-source llm reinforcement learning system at scale Year: (2025)
Ref_id:b65 Title: Rrhf: Rank responses to align language models with human feedback Year: (2023)
Ref_id:b66 Title: Efficient and reliable reinforcement learning for advanced reasoning tasks Year: (2025)
Ref_id:b67 Title: Hellaswag: Can a machine really finish your sentence Year: (2019)
Ref_id:b68 Title: Improving llm general preference alignment via optimistic online mirror descent Year: (2025)
Ref_id:b69 Title: Iterative nash policy optimization: Aligning LLMs with general preferences via no-regret learning Year: ()
Ref_id:b70 Title: Slic-hf: Sequence likelihood calibration with human feedback Year: (2023)
Ref_id:b71 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b72 Title: Instruction-following evaluation for large language models Year: (2023)
Ref_id:b73 Title: Extragradient preference optimization (egpo): Beyond last-iterate convergence for nash learning from human feedback Year: (2025)
Ref_id:b74 Title: Silei Xu, and Chenguang Zhu. Wpo: Enhancing rlhf with weighted preference optimization Year: (2024)
