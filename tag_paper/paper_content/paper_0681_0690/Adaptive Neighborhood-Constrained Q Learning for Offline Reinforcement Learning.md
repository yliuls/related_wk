Title: Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning
Abstract: Offline reinforcement learning (RL) suffers from extrapolation errors induced by out-of-distribution (OOD) actions. To address this, offline RL algorithms typically impose constraints on action selection, which can be systematically categorized into density, support, and sample constraints. However, we show that each category has inherent limitations: density and sample constraints tend to be overly conservative in many scenarios, while the support constraint, though least restrictive, faces challenges in accurately modeling the behavior policy. To overcome these limitations, we propose a new neighborhood constraint that restricts action selection in the Bellman target to the union of neighborhoods of dataset actions. Theoretically, the constraint not only bounds extrapolation errors and distribution shift under certain conditions, but also approximates the support constraint without requiring behavior policy modeling. Moreover, it retains substantial flexibility and enables pointwise conservatism by adapting the neighborhood radius for each data point. In practice, we employ data quality as the adaptation criterion and design an adaptive neighborhood constraint. Building on an efficient bilevel optimization framework, we develop a simple yet effective algorithm, Adaptive Neighborhood-constrained Q learning (ANQ), to perform Q learning with target actions satisfying this constraint. Empirically, ANQ achieves state-of-the-art performance on standard offline RL benchmarks and exhibits strong robustness in scenarios with noisy or limited data.

Section: Introduction
Reinforcement learning (RL) tackles sequential decision-making problems and has gained considerable attention in recent years [58,70,67,14]. Despite its promise, RL faces practical challenges, notably the high data collection costs [38] and exploration risks [24]. Offline RL offers a compelling alternative by learning from a static dataset collected by a behavior policy [43,44]. It enables the use of existing large-scale datasets [33,52,63] and reduces the dangers of unsafe exploration. However, it also introduces a key challenge: evaluating out-of-distribution (OOD) actions leads to extrapolation errors [23], which further causes value overestimation and significant performance degradation [44].
To address this issue, offline RL approaches typically impose constraints on the action selection process. A common strategy is to align the probability densities of the trained and behavior policies [83,21], enforcing a density constraint. This is usually achieved using divergence metrics such as reverse Kullback-Leibler (KL) [83,32], forward KL (i.e., behavior cloning) [21,61], Fisher [39], or the implicit CQL divergence [42]. While straightforward, these methods can be overly restrictive both theoretically and empirically [41], in cases where the overall quality of the behavior policy is low. To overcome this limitation, recent work has explored the most relaxed support constraint, which only keeps the selected actions within the support of the behavior policy [82,53,93]. Accomplishing this generally necessitates high-fidelity estimation of the behavior policy through advanced generative modeling techniques [26,82,93,9]. However, such modeling is challenging due to the
this section cite: ['b57', 'b69', 'b66', 'b13', 'b37', 'b23', 'b42', 'b43', 'b32', 'b51', 'b62', 'b22', 'b43', 'b82', 'b20', 'b82', 'b31', 'b20', 'b60', 'b38', 'b41', 'b40', 'b81', 'b52', 'b92', 'b25', 'b81', 'b92', 'b8']

Section: Density
Enforce density proximity between the trained and behavior policies BRAC [83], TD3BC [21], CQL [42] Straightforward but heavily limited by the overall quality of behavior policy Sample Restrict action selection to dataset actions IQL [40], XQL [25], SQL [88] Avoid extrapolation error but lack action generalization beyond the dataset Support Restrict action selection to behavior policy's support BCQ [23], BEAR [41], SPOT [82] Least restrictive but require accurate behavior policy modeling
this section cite: ['b82', 'b20', 'b41', 'b39', 'b24', 'b87', 'b22', 'b40', 'b81']

Section: Neighborhood
Restrict action selection to certain neighborhoods of dataset actions ANQ (Ours)
Flexible and approximate support constraint without behavior modeling high-dimensional and multi-modal nature of real-world data [93], subjecting these methods to heightened error susceptibility and increased computational overhead. Alternatively, the sample constraint has emerged, formulating the Bellman target exclusively using actions in the dataset [6,40,92,88]. These methods are easy to implement and effectively avoid extrapolation errors [40]. However, their performance is inherently limited by a lack of action generalization beyond the offline dataset, often resulting in overly conservative policies when near-optimal actions are rare in the dataset.
This work aims to address the over-conservatism of the density and sample constraints while avoiding complex behavior modeling required by the support constraint. To this end, we introduce a new neighborhood constraint that restricts action selection in the Bellman target to the union of neighborhoods of dataset actions. Theoretically, the constraint not only bounds extrapolation errors and distribution shift under certain conditions, but also approximates the least restrictive support constraint without behavior modeling. Moreover, it maintains high flexibility and can achieve pointwise conservatism by adapting the neighborhood radius for each data point. In light of real-world data patterns, we adopt data quality as the adaptation criterion and develop an adaptive neighborhood constraint in practice. It assigns larger neighborhood radii to low-advantage dataset actions to promote a broader search, and smaller neighborhood radii to high-advantage dataset actions to limit the overall extrapolation error.
To enforce the proposed constraint, we introduce an efficient bilevel optimization framework and, based on it, develop a simple yet effective algorithm, Adaptive Neighborhood-constrained Q learning (ANQ), which performs Q learning with target actions constrained accordingly. Specifically, in the inner optimization, we maximize the Q function separately within each dataset action's neighborhood; in the outer optimization, we implicitly maximize the Q function over all available neighborhoods via expectile regression [40]. With the Q function being trained, the policy is independently extracted by weighted regression toward optimized actions within the neighborhoods, obtained in the inner maximization. Empirically, ANQ achieves state-of-the-art performance on standard offline RL benchmarks [20], including Gym locomotion tasks and challenging AntMaze tasks. Moreover, benefiting from the flexible constraint without behavior modeling errors, ANQ attains superior performance in both noisy and limited data scenarios compared to algorithms with other types of constraints. The code is available at https://github.com/thu-rllab/ANQ.
this section cite: ['b92', 'b5', 'b39', 'b91', 'b87', 'b39', 'b39', 'b19']

Section: Preliminaries
RL. In RL, the environment is typically modeled as a Markov Decision Process (MDP) M = (S, A, P, R, γ, d 0 ), with state space S, action space A, transition dynamics P : S × A → ∆(S), reward function R : S × A → [0, R max ], discount factor γ ∈ [0, 1), and initial state distribution d 0 [73]. The agent seeks a policy π : S → ∆(A) that maximizes the expected return: Offline RL. In offline RL, the agent is able to access a fixed dataset D = {(s i , a i , r i , s ′ i )} n-1 i=0 collected by some behavior policy π β , and aims to learn an optimal policy without further data collection [43,44]. Standard Q learning methods seek to learn the optimal Q function by minimizing:
L Q (θ) = E (s,a,s ′ )∼D (Q θ (s, a) -R(s, a) -γ max a ′ Q θ ′ (s ′ , a ′ )) 2 ,(2)
where Q θ (s, a) denotes a parameterized Q function, and Q θ ′ (s, a) represents a target Q function with parameters updated using Polyak averaging [58].
A central challenge in offline RL is the presence of out-of-distribution (OOD) actions that fall outside the support of the behavior policy. These OOD actions often lead to inaccurate Q value estimates due to extrapolation errors [23]. As a result, maximizing the estimated Q functions tends to favor OOD actions with overestimated values, resulting in significant performance degradation [44].
this section cite: ['b72', 'b42', 'b43', 'b57', 'b22', 'b43']

Section: Adaptive Neighborhood-Constrained Q Learning for Offline RL
This section focuses on developing action-selection constraints to address the OOD issue in offline RL. First, we provide a systematic categorization of existing approaches and analyze the strengths and limitations of each category. To overcome the limitations, we propose a new neighborhood constraint, supported by theoretical analyses that elucidate its properties. Furthermore, we design an adaptive variant of this flexible constraint, achieving pointwise conservatism in practice. Finally, we develop a simple yet effective algorithm to facilitate Q learning and policy extraction under the constraint.
this section cite: []

Section: A Categorization and Analysis of Constraints in Offline RL
To address the OOD issue, offline RL algorithms impose various constraints to prevent either the learned policy (in actor-critic training) or the Bellman target (in Q learning) from selecting OOD actions. In this context, many approaches inherently align the probability density of the trained policy with that of the behavior policy, either explicitly, through divergence measures such as reverse KL [32,83], forward KL (i.e., behavior cloning) [21,61], and Fisher divergence [39], or implicitly, via value penalties that reduce the Q values of trained policy's actions and while increasing those of dataset actions [42,12]. We formalize this concept as the density constraint in Definition 1.
this section cite: ['b31', 'b82', 'b20', 'b60', 'b38', 'b41', 'b11']

Section: Definition 1 (Density constraint).
The trained policy satisfies the density constraint D(π, π β ) ≤ ϵ, where D represents a divergence measure between the trained policy π and the behavior policy π β , e.g., KL, total variation (TV), or Fisher divergence.
While straightforward, this density constraint can be overly restrictive in many scenarios. Lemma 1 provides a theoretical upper bound on policy performance under various forms of density constraints.
this section cite: []

Section: Lemma 1 (Performance bound under density constraints).
If any of the conditions D KL (π∥π β ) ≤ 2ϵ, D KL (π β ∥π) ≤ 2ϵ, or D TV (π, π β ) ≤ √ ϵ holds, then the policy performance η is bounded as follows:
η(π) ≤ η(π β ) + 2R max (1 -γ) 2 √ ϵ.(3)
Lemma 1 demonstrates that policy performance under the density constraints is affected by the overall quality of the behavior policy. Consequently, even if the optimal behavior is present in the dataset, the learned policy can still remain highly suboptimal if the overall behavior policy is of low quality.
To address this limitation inherent in the density constraint, recent studies have explored the more relaxed support constraint, which only requires selected actions to be within the support of the behavior policy [82,26,53,93]. Prior to the formal definition, we introduce a general loss function for constrained Q learning in Eq. ( 4), where C(s) denotes a conditional set of actions for a given state.
L C (θ) = E (s,a,s ′ )∼D (Q θ (s, a) -R(s, a) -γ max a ′ ∈C(s ′ ) Q θ ′ (s ′ , a ′ )) 2 .(4)
Definition 2 (Support constraint). The selected action in the Bellman target is restricted to the support of the behavior policy, which is defined as C Supp (s) := {a ∈ A | π β (a|s) > ϵ}, where π β is the behavior policy and ϵ is a threshold that determines the support.
This support constraint is generally considered the least restrictive for offline RL [82], as the quality of actions outside the behavior policy's support cannot be reliably assessed. To enforce this constraint, existing approaches typically rely on behavior policy modeling, using techniques such as conditional variational autoencoders (CVAEs) [23,41,94,82], autoregressive models [26], flow-GANs [93], and diffusion models [9]. Specifically, these methods either use pre-trained behavior density estimators to explicitly constrain the policy within the behavior support [41,82,93], or employ pre-trained behavior policy samplers to generate in-support actions and select the one with the highest Q value [23,26,94,9]. However, their effectiveness is fundamentally limited by the accuracy of behavior policy modeling [26], which is well-known to be challenging due to the high-dimensional and multi-modal nature of real-world data [93]. Moreover, these methods incur extra computational costs due to behavior model training and, in some cases, extensive action sampling per state.
A parallel research direction has introduced the sample constraint, which constructs the Bellman target exclusively using actions in the dataset [40,85,92,88], and derives policies via weighted behavior cloning [61,81], thereby avoiding querying any out-of-dataset actions.
this section cite: ['b81', 'b25', 'b52', 'b92', 'b81', 'b22', 'b40', 'b93', 'b81', 'b25', 'b92', 'b8', 'b40', 'b81', 'b92', 'b22', 'b25', 'b93', 'b8', 'b25', 'b92', 'b39', 'b84', 'b91', 'b87', 'b60', 'b80']

Section: Definition 3 (Sample constraint).
The selected action in the Bellman target is restricted to the sample set C Samp (s) := {a ∈ A | (s, a) ∈ D}, consisting of actions in the dataset for a given state s ∈ D.
Sample constraint methods are computationally efficient, easy to implement, and effective in avoiding extrapolation errors [40]. However, their performance is inherently constrained by the inability to generalize beyond the offline dataset. This over-conservatism becomes especially problematic when the dataset lacks coverage of near-optimal actions, a common issue in environments with large or continuous action spaces, or when the dataset exhibits low quality or limited diversity. Moreover, to ensure computational stability, these methods often struggle to adequately suppress the impact of suboptimal dataset actions [88], diminishing their efficacy when such actions dominate the dataset.
For extended discussions on related work, we refer the reader to Appendix A.
this section cite: ['b39', 'b87']

Section: Neighborhood Constraint for Offline RL
This work aims to mitigate the over-conservatism inherent in the density and sample constraints, while circumventing the behavior modeling requirement posed by the support constraint. To this end, we introduce a flexible neighborhood constraint for offline RL, which restricts action selection in the Bellman target to the union of neighborhoods of dataset actions on a given state.
Definition 4 (Neighborhood constraint). The selected action in the Bellman target is restricted to the neighborhood set C N (s) := {ã ∈ A | ∥ã -a∥ ≤ ϵ, (s, a) ∈ D}, which comprises actions located within the ϵ-neighborhoods of all dataset actions on a given state s ∈ D.
In contrast to the sample constraint, this neighborhood constraint offers greater freedom, as it allows for seeking better actions beyond the dataset, within a flexible range. As shown in the following Theorem 1, the neighborhood constraint can serve as a viable approximation to the least restrictive support constraint, with the benefit of being achievable without behavior policy modeling. To establish the theorem, we introduce the standardness assumption commonly used in geometric measure theory [13,8], which ensures that the measure does not exhibit "holes" at small scales. Assumption 1 (Standardness). Let S ⊆ R d be the support of a probability distribution ν, and B(x, r) be the closed ball of radius r centered at x ∈ R d . There exist constants r 0 > 0 and C 0 > 0 such that:
∀x ∈ S, ∀r ≤ r 0 , ν(B(x, r)) ≥ C 0 • r d .(5)
Theorem 1 (Support approximation via neighborhoods). Let S ⊆ R d be the compact support of a distribution ν, and let X 1 , . . . , X n be independent and identically distributed samples from ν. Define U n,ϵ = n i=1 B(X i , ϵ) as the union of closed balls of radius ϵ centered at the samples. Let N (S, ϵ/2) denote the covering number of S, i.e., the minimal number of ϵ/2-balls required to cover S. Under the standardness Assumption 1 with constants r 0 , C 0 > 0, for any δ ∈ (0, 1) and ϵ ≤ 2r 0 , if
n ≥ 1 C 0 (ϵ/2) d (log N (S, ϵ/2) + log(1/δ)) ,(6)
then with probability at least 1 -δ, the Hausdorff distance between S and U n,ϵ satisfies
d H (S, U n,ϵ ) := max sup x∈S inf u∈Un,ϵ d(x, u), sup u∈Un,ϵ inf x∈S d(x, u) ≤ ϵ.(7)
Since both the support and neighborhood constraint sets are defined over actions conditioned on a given state, Theorem 1 analyzes their relationship at a fixed state, focusing on their difference in the action space. In Theorem 1, ν represents the behavior policy distribution at a state, and S is defined as its support. X 1 , . . . , X n are i.i.d. samples from ν, corresponding to dataset actions at that state. This theorem ensures that the union of sample neighborhoods U n,ϵ approximates the support S within a controlled Hausdorff distance, capturing the trade-off between sample size n, neighborhood radius ϵ, and the geometric complexity of support S (as reflected by N (S, ϵ/2)). Note that this Hausdorff distance is sensitive to outliers [29], making it well-suited for evaluating approximation quality in our setting, where outlier actions can significantly affect Q function optimization.
In the following, we further investigate several properties of the proposed neighborhood constraint in the context of controlling extrapolation and distribution shift. The definition of this constraint is closely related to the concept of extrapolation, and Lemma 2 provides a theoretical characterization of the extrapolation behavior of deep Q functions under this constraint.
this section cite: ['b12', 'b7', 'b28']

Section: Lemma 2 (Extrapolation behavior).
Under the neural tangent kernel (NTK) regime [30], for any in-sample state-action pair (s, a) ∈ D and in-neighborhood state-action pair (s, ã) such that ∥ã -a∥ ≤ ϵ, the value difference of the deep Q function can be bounded as:
∥Q θ (s, ã) -Q θ (s, a)∥ ≤ C( min(∥s ⊕ a∥, ∥s ⊕ ã∥) √ ϵ + 2ϵ),(8)
where ⊕ denotes the vector concatenation operation, and C is a finite constant.
Lemma 2 is a direct corollary of Theorem 1 in [45], specialized to the case of action extrapolation. It demonstrates that, for any unseen action ã, its Q value Q θ (s, ã) can be effectively controlled by a dataset action's Q value Q θ (s, a) and the distance ∥ã -a∥. Specifically, a smaller neighborhood radius yields tighter control over the output of deep Q functions.
Furthermore, Proposition 1 demonstrates that, under mild continuity conditions on the transition dynamics [15,87], the neighborhood constraint also helps to bound the degree of distribution shift.
this section cite: ['b29', 'b44', 'b14', 'b86']

Section: Proposition 1 (Distribution shift).
Let π 1 be a deterministic policy that satisfies the neighborhood constraint with threshold ϵ. Assume that the transition dynamics P is K P -Lipschitz continuous: ∀s ∈ S, ∀a 1 , a 2 ∈ A, ∥P (s ′ |s, a 1 ) -P (s ′ |s, a 2 )∥ ≤ K P ∥a 1 -a 2 ∥. Then, there exists a policy π 2 satisfying the sample constraint such that:
D TV (d π1 (•), d π2 (•)) ≤ γK P ϵ 2(1 -γ) ,(9)
where
d π (s) = (1 -γ) ∞ t=0 γ t E π [I [s t = s]]
is the state occupancy induced by policy π.
Adaptive neighborhoods. The neighborhood constraint is highly flexible and can, in practice, achieve pointwise conservatism by adapting the neighborhood radius for each data point, enabling the design of an adaptive neighborhood constraint. Considering real-world data patterns, expert data typically clusters within a narrow distribution [3,18], necessitating tighter constraints to mitigate extrapolation errors, while suboptimal data tends to be more dispersed and thus benefits from looser constraints that facilitate policy improvement. Inspired by this idea, we propose a concrete instantiation of adaptive neighborhoods in Definition 5, where per-sample radius is set as ϵ exp(-αA(s, a)).
Definition 5 (Adaptive neighborhood constraint). The selected action in the Bellman target is restricted to the adaptive neighborhood set C AN (s) := {ã ∈ A | ∥ã-a∥ ≤ ϵ exp(-αA(s, a)), (s, a) ∈ D}, where A denotes the advantage function and α is an inverse temperature parameter that modulates the sensitivity of the neighborhood radius to advantage values.
This adaptive neighborhood constraint assigns larger neighborhood radii to dataset actions with low advantage, thereby promoting a broader search over the action space and further mitigating the impact of low-quality data. Conversely, dataset actions with high advantage are assigned smaller neighborhood radii to more effectively reduce the overall extrapolation error. In practice, advantage estimation errors are typically not a concern for two reasons: (1) In-distribution estimation: the advantage is computed only on dataset points (s, a) ∈ D, where estimates are relatively reliable;
(2) Qualitative use: the purpose of using advantage is to distinguish actions qualitatively, and the exponential form is merely a soft heuristic to bias the radius, without requiring precise values.
this section cite: ['b2', 'b17']

Section: Adaptive Neighborhood-Constrained Q Learning
In the following, we develop an efficient bilevel optimization framework to achieve Q learning under the adaptive neighborhood constraint. Specifically, we aim to minimize the following Q learning loss:
L ANQ (θ) = E (s,a,s ′ )∼D Q θ (s, a) -R(s, a) -γ max a ′ ∈CAN(s ′ ) Q θ ′ (s ′ , a ′ ) 2 .(10)
Bilevel optimization. The primary challenge in constrained Q learning lies in enforcing max a ′ ∈C(s ′ ) in the Bellman target. While the support constraint C Supp typically necessitates accurate modeling of the behavior policy, we demonstrate that the adaptive neighborhood constraint C AN can be effectively enforced by decomposing the objective into a bilevel optimization structure: max a∈CAN(s) Q(s, a), ∀s ∈ D ⇐⇒ max a∈D(s) Q(s, a + δ sa ), ∀s ∈ D s.t. δ sa = argmax ∥δ∥≤ϵ exp(-αA(s,a))
Q(s, a + δ), ∀(s, a) ∈ D,
where we use D(s) to denote the empirical action set observed in the dataset for a given state s ∈ D.
The inner maximization in Eq. ( 11) optimizes the Q function separately within each dataset action's neighborhood. To this end, we introduce an auxiliary policy µ ω that takes state-action pairs (s, a) from the dataset as input and outputs action variations δ. This formulation enables straightforward enforcement of the adaptive neighborhood constraint by restricting the norm of µ ω (s, a) to stay within the bound ϵ exp(-αA(s, a)). Practically, we multiply both sides of the constraint inequality by exp(αA(s, a)) to maintain a constant constraint threshold: exp(αA(s, a))∥µ ω (s, a)∥ ≤ ϵ. Consequently, we optimize the Q function with respect to µ ω to seek the optimal action within the adaptive neighborhood of each dataset action, according to the following objective:
We reformulate the constrained optimization problem into an unconstrained one using a Lagrange multiplier λ ∈ R + . In addition, we introduce a state value function V ψ (s), whose training objective will be specified later, and use the difference Q θ ′ -V ψ to compute the advantage function. Accordingly, we optimize the following objective for the inner maximization:
max µω E (s,a)∼D [Q θ (s, a + µ ω (s, a)) -λ exp(α(Q θ ′ (s, a) -V ψ (s)))∥µ ω (s, a)∥] .(13)
The outer maximization in Eq. ( 11) searches over all dataset actions on a given state and seeks the one whose corresponding neighborhood yields the highest Q value. To achieve this objective, we first sample state-action pairs from the dataset and refine the actions by adding the outputs of the trained auxiliary policy, thereby simulating the sampling of the optimized actions across the neighborhoods. We then employ expectile regression [40] to implicitly maximize the Q function over these optimized actions. Specifically, we fit a V function with the following asymmetric squared error loss, treating the Q values of the optimized actions as regression targets:
min V ψ E (s,a)∼D [L τ 2 (Q θ ′ (s, a + µ ω ′ (s, a)) -V ψ (s))] ,(14)
where L τ 2 (x) = |τ -1(x < 0)|x 2 , τ ∈ (0, 1), Q θ ′ and µ ω ′ are the target Q function and target auxiliary policy, whose parameters are updated via Polyak averaging [58].
For τ ≈ 1, V ψ (s) captures the maximum Q value within the adaptive neighborhood set C AN (s). By substituting max a ′ ∈CAN(s ′ ) Q θ ′ (s ′ , a ′ ) in Eq. ( 10) with V ψ (s ′ ), adaptive neighborhood-constrained Q learning is achieved based on the following loss:
min Q θ E (s,a,s ′ )∼D (Q θ (s, a) -R(s, a) -γV ψ (s ′ )) 2 .(15)
A radius-agnostic framework. Although our Q learning algorithm is presented specifically for the adaptive neighborhood in Definition 5, i.e., using the per-sample radius ϵ exp(-αA(s, a)), the overall framework is general and can accommodate arbitrary radius schemes. Specifically, one can simply replace ϵ exp(-αA(s, a)) in Definition 5 (and Eq. ( 11)) with ϵf (s, a) to define a generic per-sample neighborhood radius, where f : S × A → R + is an arbitrary function that modulates the radius. Correspondingly, in Eq. ( 12), exp(αA(s, a)) becomes 1/f (s, a), and Eq. ( 13) becomes:
max µω E (s,a)∼D [Q θ (s, a + µ ω (s, a)) -λ∥µ ω (s, a)∥/f (s, a)].(16)
With all other equations unchanged, the resulting algorithm supports arbitrary neighborhood schemes.
this section cite: ['b39', 'b57']

Section: Policy Extraction via Weighted Regression Toward Optimized Actions
While our algorithm enables Q learning under the adaptive neighborhood constraint, it does not explicitly derive the corresponding policy, thereby requiring a separate policy extraction step. To this end, we employ the weighed behavior cloning method [17,11] and, rather than imitating the actions in the dataset, we instead imitate the actions that have been refined through the auxiliary policy, which represents the optimal actions within the adaptive neighborhoods. Moreover, we set the weights as the exponentiated advantage function [79,61,60,81]. Consequently, the final policy π ϕ : S → A is extracted according to the following loss:
min π ϕ E (s,a)∼D exp(β(Q θ ′ (s, a + µ ω (s, a)) -V ψ (s)))∥a + µ ω (s, a) -π ϕ (s)∥ 2 2 , (17
)
where β is an inverse temperature and Q θ ′ -V ψ computes the advantage function.
Algorithm 1 ANQ 1: Initialize policy π ϕ , auxiliary policy µ ω , target auxiliary policy µ ω ′ , Q-network Q θ , target Q-network Q θ ′ , and V-network V ψ . 2: for each gradient step do 3: Update ψ by minimizing Eq. (14) 4: Update θ by minimizing Eq. (15) 5: Update ω by maximizing Eq. (13) 6: Update ϕ by maximizing Eq. (17) 7:
Update target networks:
θ ′ ← (1 -ξ)θ ′ + ξθ, ω ′ ← (1 -ξ)ω ′ + ξω 8: end for
Remark. This policy extraction step, which does not interfere with the Q learning process described in Section 3.3, also constitutes a key distinction from existing regression-based policy learning objectives, such as those employed in AWR [61], AWAC [60], CRR [81], 10% BC [10], IQL [40], and SQL [88], all of which perform weighted regression toward the dataset actions. In contrast, our policy learning objective performs weighted regression toward the optimized actions within the adaptive neighborhoods. This enables the trained policy to select actions superior to those in the dataset, while also significantly mitigating the adverse effects of suboptimal dataset actions.
Integrating all components, we present our final algorithm in Algorithm 1.
this section cite: ['b16', 'b10', 'b78', 'b60', 'b59', 'b80', 'b60', 'b59', 'b80', 'b9', 'b39', 'b87']

Section: Experiments
We conduct experiments to evaluate the performance and properties of the proposed approach ANQ. Experimental details and extended results are provided in Appendices C and D, respectively.
this section cite: []

Section: Benchmark Results
Tasks. We assess ANQ on two distinct task suites from D4RL [20]: the Gym-MuJoCo locomotion domains and the challenging AntMaze domains. The AntMaze tasks involve sparse rewards and require the ant agent to combine segments of suboptimal trajectories to reach the maze's goal.
Baselines. Our offline RL baselines span various constraint categories. For density constraints, we compare to TD3BC [21], CQL [42], and AWAC [60], where CQL and AWAC essentially enforce a density constraint as analyzed in Theorem 3.5 of [42] and Section 3 of [53], respectively. For support constraints, we include BCQ [23], BEAR [41], and SPOT [82]. For sample constraints, we compare against OneStep RL [6] and IQL [40]. We also include the sequence-modeling approach DT [10].
Table 2: Averaged normalized scores on Gym locomotion and Antmaze tasks over five random seeds. m = medium, m-r = medium-replay, m-e = medium-expert, e = expert, r = random; u = umaze, u-d = umaze-diverse, m-p = medium-play, m-d = medium-diverse, l-p= large-play, l-d = large-diverse.
Dataset-v2 BCQ BEAR DT AWAC OneStep TD3BC CQL IQL SPOT ANQ (Ours) halfcheetah-m 46.6 43.0 42.6 47.9 50.4 48.3 47.0 47.4 58.4 61.8±1.4 hopper-m 59.4 51.8 67.6 59.8 87.5 59.3 53.0 66.2 86.0 100.9±0.6 walker2d-m 71.8 -0.2 74.0 83.1 84.8 83.7 73.3 78.3 86.4 82.9±1.5 halfcheetah-m-r 42.2 36.3 36.6 44.8 42.7 44.6 45.5 44.2 52.2 55.5±1.4 hopper-m-r 60.9 52.2 82.7 69.8 98.5 60.9 88.7 94.7 100.2 101.5±2.7 walker2d-m-r 57.0 7.0 66.6 78.1 61.7 81.8 81.8 73.8 91.6 92.7±3.8 halfcheetah-m-e 95.4 46.0 86.8 64.9 75.1 90.7 75.6 86.7 86.9 94.2±0.8 hopper-m-e 106.9 50.6 107.6 100.1 108.6 98.0 105.6 91.5 99.3 107.0±4.9 walker2d-m-e 107.7 22.1 108.1 110.0 111.3 110.1 107.9 109.6 112.0 111.7±0.2 halfcheetah-e 89.9 92.7 87.7 81.7 88.2 96.7 96.3 95.0 94.8 95.9±0.4 hopper-e 109.0 54.6 94.2 109.5 106.9 107.8 96.5 109.4 111.0 111.4±2.5 walker2d-e 106.3 106.6 108.3 110.1 110.7 110.2 108.5 109.9 109.9 111.8±0.1 halfcheetah-r 2.2 2.3 2.2 6.1 2.3 11.0 17.5 13.1 25.4 24.9±1.0 hopper-r 7.8 3.9 5.4 9.2 5.6 8.5 7.9 7.9 23.4 31.1±0.2 walker2d-r 4.9 12.8 2.2 0.2 6.9 1.6 5.1 5.4 2.4 11.2±9.5 locomotion total 968.0 581.7 972.6 975.3 1041.2 1013.2 1010.2 1033.1 1139.9 1194.5 antmaze-u 78.9 73.0 54.2 80.0 54.0 73.0 82.6 89.6 93.5 96.0±1.6 antmaze-u-d 55.0 61.0 41.2 52.0 57.8 47.0 10.2 65.6 40.7 80.2±1.8 antmaze-m-p 0.0 0.0 0.0 0.0 0.0 0.0 59.0 76.4 74.7 76.2±3.3 antmaze-m-d 0.0 8.0 0.0 0.2 0.6 0.2 46.6 72.8 79.1 77.2±6.1 antmaze-l-p 6.7 0.0 0.0 0.0 0.0 0.0 16.4 42.0 35.3 56.2±4.9 antmaze-l-d 2.2 0.0 0.0 0.0 0.2 0.0 3.2 46.0 36.3 55.8±4.0 antmaze total 142.8 142.0 95.4 132.2 112.6 120.2 218.0 392.4 359.6 441.6
Comparisons. Aggregated results are reported in Table 2. On the Gym locomotion tasks, ANQ outperforms existing methods on most tasks and achieves the highest overall score. On the challenging AntMaze tasks, ANQ surpasses the baselines by a considerable margin, particularly in the most complex large maze settings. The learning curves are provided in Appendix D.5. We also extend our evaluation in Appendix D.2 by comparing ANQ with additional recent SOTA algorithms.
Runtime. We test the runtime of ANQ and some baseline methods on a GeForce RTX 3090. As shown in Appendix D.1, ANQ is among the fastest tier of offline RL algorithms, on par with efficient baselines such as AWAC, IQL, and TD3BC, with a detailed analysis provided in the same section.
this section cite: ['b19', 'b20', 'b41', 'b59', 'b41', 'b52', 'b22', 'b40', 'b81', 'b5', 'b39', 'b9']

Section: Noisy Data Results
We examine the robustness of various constraint types under noisy data conditions. Specifically, we construct noisy datasets by mixing the random and expert datasets at varying ratios, thereby simulating real-world scenarios such as imperfect demonstrations in robotics or suboptimal data collection in autonomous systems. We then evaluate the performance of representative algorithms, including CQL (density), IQL (sample), SPOT (support), and ANQ (neighborhood).
As presented in Figure 1(a), ANQ generally outperforms the other algorithms across expert ratios, and its performance advantage becomes more pronounced as the proportion of expert data decreases. As analyzed in Section 3.1, the density constraint is sensitive to the overall quality of the behavior policy, which tends to be low in noisy datasets, while the support constraint often struggles with modeling the multi-modal behavior policy distribution inherent in such datasets. In contrast, the adaptive neighborhood constraint employed by ANQ exhibits greater robustness to noisy data. Moreover, we evaluate ANQ on such noisy datasets with varying inverse temperature α that controls the adaptiveness of neighborhood radius. The results in Figure 1(b) demonstrate that, compared with the uniform neighborhood constraint (α = 0), the adaptive neighborhood constraint further mitigates the adverse effects of low-quality data in the datasets and exhibits greater robustness to such data. 0.0 0.2 0.4 0.6 0.8 Discard Ratio 0 20 40 60 80 Normalized Return Antmaze-Medium-Play 0.0 0.2 0.4 0.6 0.8 Discard Ratio 0 10 20 30 40 50 60 Antmaze-Large-Play 0.0 0.2 0.4 0.6 0.8 Discard Ratio 0 10 20 30 40 50 60 Antmaze-Large-Diverse ANQ (Neighborhood) SPOT (Support) IQL (Sample) (a) Evaluation of algorithms with various constraint types on reduced datasets 0.2 0.4 0.6 0.8 Discard Ratio 20 40 60 80 Normalized Return ANQ on Antmaze-M-P =2 =5 =10 (b) ANQ with varying λ Figure 2: (a) Evaluation on reduced datasets over five random seeds. (b) Evaluation of ANQ on reduced datasets with varying Lagrange multiplier λ that controls the overall radius of neighborhoods.
this section cite: []

Section: Limited Data Results
We also investigate the robustness of these constraint types in limited data settings. To this end, we create reduced datasets by randomly discarding some portion of transitions from the AntMaze datasets. This setup mimics practical scenarios in which data is rare or partially missing, such as in healthcare applications. Again, we evaluate the performance of IQL, SPOT, and ANQ, omitting CQL due to its consistently inferior performance on Antmaze tasks as reported in Table 2.
As shown in Figure 2(a), ANQ demonstrates superior performance across nearly all discard ratios, with the performance gap widening as the amount of available data decreases. In such limited data settings, support constraint methods face even more difficulties in modeling the behavior policy due to sample scarcity, whereas sample constraint methods risk being overly conservative because of reduced coverage of near-optimal actions. In contrast, ANQ bypasses the need to model the behavior policy and leverages generalization to attain superior performance beyond the offline dataset. Furthermore, we evaluate ANQ on reduced datasets with varying Lagrange multiplier λ, which is inversely proportional to the overall radius of the neighborhoods. The results in Figure 1(b) show that an appropriately large neighborhood is crucial for achieving good performance in such limited data scenarios, further showcasing the benefit and flexibility of ANQ over sample constraint methods.
this section cite: []

Section: Ablation Study
Lagrange multiplier λ. The Lagrange multiplier λ controls the overall neighborhood radius in ANQ. We vary λ and present the learned Q values and performance across various tasks in Figure 3.
As λ decreases from a sufficiently large value, ANQ enables larger neighborhoods, resulting in higher and possibly divergent Q values, and performance also exhibits a rise-then-fall trend. Note that ANQ with λ = 0 and λ = ∞ approximately corresponds to Q learning without any constraint and with the sample constraint, respectively. Therefore, the results provide evidence that ANQ not only effectively suppresses extrapolation error, but also mitigates the over-conservatism of the sample constraint. 0.0 0.1 0.5 1.0 2.0 5.0 10 20 Inverse temperature 40 60 80 100 Normalized Return Return 240 245 250 255 Q Value hopper-medium Q value 0.0 0.1 0.5 1.0 2.0 5.0 10 20 Inverse temperature 54 56 58 60 62 Normalized Return Return 575 600 625 650 675 Q Value halfcheetah-medium Q value 0.0 0.1 0.5 1.0 2.0 5.0 10 20 Inverse temperature 50 52 54 56 Normalized Return Return 440 460 480 500 Q Value halfcheetah-medium-replay Q value 0.0 0.1 0.5 1.0 2.0 5.0 10 20 Inverse temperature 17 20 23 26 29 Normalized Return Return 100 125 150 175 200 Q Value halfcheetah-random Q value Figure 4: Performance and Q values of ANQ with varying inverse temperature α over five random seeds. An appropriately large α (adaptive neighborhoods) yields enhanced performance.
Inverse temperature α. The inverse temperature α determines how the neighborhood radius adapts to the action advantage, where α = 0 corresponds to a fixed neighborhood radius. The results in Figure 4 demonstrate that an appropriately large α leads to enhanced performance, validating our design of advantage-based adaptive neighborhoods. However, an excessively large α (α = 20) may degrade performance, likely due to the increased variance of the learning objective.
this section cite: []

Section: Conclusion and Limitations
This work focuses on developing action-selection constraints to address the OOD issue in offline RL.
To overcome the identified limitations of existing approaches, we propose the flexible neighborhood constraint and the corresponding algorithm ANQ, which mitigates the over-conservatism inherent in the density and sample constraints, and approximates the least restrictive support constraint without challenging behavior modeling. Empirical results demonstrate that ANQ achieves SOTA performance on standard offline RL benchmarks and exhibits enhanced robustness to noisy or limited data.
At the algorithmic level, this work develops a general framework for achieving pointwise conservatism by adapting the neighborhood radius for each data point. In particular, the practical algorithm ANQ represents one instantiation of adaptive neighborhoods, using data point quality as the adaptation criterion. However, this design is not necessarily optimal; incorporating additional information, such as uncertainty quantification, could potentially lead to more effective neighborhood construction.
this section cite: []

Section: References
Ref_id:b0 Title: Constrained policy optimization Year: (2017)
Ref_id:b1 Title: Uncertainty-based offline reinforcement learning with diversified q-ensemble Year: (2021)
Ref_id:b2 Title: A survey of robot learning from demonstration Year: (2009)
Ref_id:b3 Title: Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks Year: (2019)
Ref_id:b4 Title: Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning Year: (2022)
Ref_id:b5 Title: Offline rl without off-policy evaluation Year: (2021)
Ref_id:b6 Title: Neural temporal-difference learning converges to global optima Year: (2019)
Ref_id:b7 Title: Convergence rates for persistence diagram estimation in topological data analysis Year: (2014)
Ref_id:b8 Title: Offline reinforcement learning via high-fidelity generative behavior modeling Year: (2023)
Ref_id:b9 Title: Decision transformer: Reinforcement learning via sequence modeling Year: (2021)
Ref_id:b10 Title: Bail: Bestaction imitation learning for batch deep reinforcement learning Year: (2020)
Ref_id:b11 Title: Adversarially trained actor critic for offline reinforcement learning Year: (2022)
Ref_id:b12 Title: Set estimation: Another bridge between statistics and geometry Year: (2009)
Ref_id:b13 Title: Magnetic control of tokamak plasmas through deep reinforcement learning Year: (2022)
Ref_id:b14 Title: Finite linear programming approximations of constrained discounted markov decision processes Year: (2013)
Ref_id:b15 Title: Approximation of average cost markov decision processes using empirical distributions and concentration inequalities Year: (2015)
Ref_id:b16 Title: Rvs: What is essential for offline rl via supervised learning Year: (2021)
Ref_id:b17 Title: The role of deliberate practice in the acquisition of expert performance Year: (1993)
Ref_id:b18 Title: A theoretical analysis of deep q-learning Year: (2020)
Ref_id:b19 Title: D4rl: Datasets for deep data-driven reinforcement learning Year: (2020)
Ref_id:b20 Title: A minimalist approach to offline reinforcement learning Year: (2021)
Ref_id:b21 Title: Addressing function approximation error in actor-critic methods Year: (2018)
Ref_id:b22 Title: Off-policy deep reinforcement learning without exploration Year: (2019)
Ref_id:b23 Title: A comprehensive survey on safe reinforcement learning Year: (2015)
Ref_id:b24 Title: Extreme q-learning: Maxent RL without entropy Year: (2023)
Ref_id:b25 Title: Emaq: Expected-max q-learning operator for simple yet effective offline and online rl Year: (2021)
Ref_id:b26 Title: A review of safe reinforcement learning: Methods, theory and applications Year: (2022)
Ref_id:b27 Title: Q-value regularized transformer for offline reinforcement learning Year: (2024)
Ref_id:b28 Title: Comparing images using the hausdorff distance Year: (1993)
Ref_id:b29 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b30 Title: When to trust your model: Model-based policy optimization Year: (2019)
Ref_id:b31 Title: Way off-policy batch deep reinforcement learning of implicit human preferences in dialog Year: (2019)
Ref_id:b32 Title:  Year: (2016)
Ref_id:b33 Title: Modelbased reinforcement learning for atari Year: (2019)
Ref_id:b34 Title: Approximately optimal approximate reinforcement learning Year: (2002)
Ref_id:b35 Title: Morel: Model-based offline reinforcement learning Year: (2020)
Ref_id:b36 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b37 Title: Reinforcement learning in robotics: A survey Year: (2013)
Ref_id:b38 Title: Offline reinforcement learning with fisher divergence critic regularization Year: (2021)
Ref_id:b39 Title: Offline reinforcement learning with implicit q-learning Year: (2022)
Ref_id:b40 Title: Stabilizing offpolicy q-learning via bootstrapping error reduction Year: (2019)
Ref_id:b41 Title: Conservative q-learning for offline reinforcement learning Year: (2020)
Ref_id:b42 Title: Batch reinforcement learning. Reinforcement learning: State-of-the-art Year: (2012)
Ref_id:b43 Title: Offline reinforcement learning: Tutorial, review, and perspectives on open problems Year: (2020)
Ref_id:b44 Title: When data geometry meets deep function: Generalizing offline reinforcement learning Year: (2023)
Ref_id:b45 Title: Neural trust region/proximal policy optimization attains globally optimal policy Year: (2019)
Ref_id:b46 Title: Offline reinforcement learning with value-based episodic memory Year: (2021)
Ref_id:b47 Title: Conservative offline distributional reinforcement learning Year: (2021)
Ref_id:b48 Title: Reining generalization in offline reinforcement learning via representation distinction Year: (2023)
Ref_id:b49 Title: Iteratively refined behavior regularization for offline reinforcement learning Year: (2024)
Ref_id:b50 Title: Rethinking decision transformer via hierarchical reinforcement learning Year: (2024)
Ref_id:b51 Title: 1 year, 1000 km: The oxford robotcar dataset Year: (2017)
Ref_id:b52 Title: Supported trust region optimization for offline reinforcement learning Year: (2023)
Ref_id:b53 Title: Supported value regularization for offline reinforcement learning Year: (2023)
Ref_id:b54 Title: Offline reinforcement learning with ood state correction and ood action suppression Year: (2024)
Ref_id:b55 Title: Doubly mild generalization for offline reinforcement learning Year: (2024)
Ref_id:b56 Title: Deployment-efficient reinforcement learning via model-based offline optimization Year: (2021)
Ref_id:b57 Title: Human-level control through deep reinforcement learning Year: (2015)
Ref_id:b58 Title: Model-based reinforcement learning: A survey Year: (2023)
Ref_id:b59 Title: Awac: Accelerating online reinforcement learning with offline datasets Year: (2020)
Ref_id:b60 Title: Advantage-weighted regression: Simple and scalable off-policy reinforcement learning Year: (2019)
Ref_id:b61 Title: A survey of temporal credit assignment in deep reinforcement learning Year: (2023)
Ref_id:b62 Title: Hokoff: Real game dataset from honor of kings and its offline reinforcement learning benchmarks Year: (2023)
Ref_id:b63 Title: Latent reward: Llm-empowered credit assignment in episodic reinforcement learning Year: (2025)
Ref_id:b64 Title: Fast and robust: Task sampling with posterior and diversity synergies for adaptive decision-makers in randomized environments Year: (2025)
Ref_id:b65 Title: Policy regularization with dataset constraint for offline reinforcement learning Year: (2023)
Ref_id:b66 Title: Mastering atari, go, chess and shogi by planning with a learned model Year: (2020)
Ref_id:b67 Title: Trust region policy optimization Year: (2015)
Ref_id:b68 Title: Counterfactual conservative q learning for offline multi-agent reinforcement learning Year: (2023)
Ref_id:b69 Title: Mastering the game of go without human knowledge Year: (2017)
Ref_id:b70 Title: Model-bellman inconsistency for model-based offline reinforcement learning Year: (2023)
Ref_id:b71 Title: an integrated architecture for learning, planning, and reacting Year: (1991)
Ref_id:b72 Title: Reinforcement learning: An introduction Year: (2018)
Ref_id:b73 Title: An efficient algorithm for calculating the exact hausdorff distance Year: (2015)
Ref_id:b74 Title: Revisiting the minimalist approach to offline reinforcement learning Year: (2024)
Ref_id:b75 Title: Mujoco: A physics engine for model-based control Year: (2012)
Ref_id:b76 Title: Learning expressive meta-representations with mixture of expert neural processes Year: (2022)
Ref_id:b77 Title: Model-based meta reinforcement learning using graph structured surrogate models and amortized policy search Year: (2022)
Ref_id:b78 Title: Exponentially weighted imitation learning for batched historical data Year: (2018)
Ref_id:b79 Title: Diffusion policies as an expressive policy class for offline reinforcement learning Year: (2023)
Ref_id:b80 Title: Critic regularized regression Year: (2020)
Ref_id:b81 Title: Supported policy optimization for offline reinforcement learning Year: (2022)
Ref_id:b82 Title: Behavior regularized offline reinforcement learning Year: (2019)
Ref_id:b83 Title: Elastic decision transformer. Advances in neural information processing systems Year: (2023)
Ref_id:b84 Title: The in-sample softmax for offline reinforcement learning Year: (2023)
Ref_id:b85 Title: Bellmanconsistent pessimism for offline reinforcement learning Year: (2021)
Ref_id:b86 Title: Deterministic policy gradient: Convergence analysis Year: (2022)
Ref_id:b87 Title: Offline RL with no OOD actions: In-sample learning via implicit value regularization Year: (2023)
Ref_id:b88 Title: RORL: Robust offline reinforcement learning via conservative smoothing Year: (2022)
Ref_id:b89 Title: Mopo: Model-based offline policy optimization Year: (2020)
Ref_id:b90 Title: Combo: Conservative offline model-based policy optimization Year: (2021)
Ref_id:b91 Title: In-sample actor critic for offline reinforcement learning Year: (2023)
Ref_id:b92 Title: Constrained policy optimization with explicit behavior density for offline reinforcement learning Year: (2024)
Ref_id:b93 Title: Plas: Latent action space for offline reinforcement learning Year: (2021)
Ref_id:b94 Title: Structural features of the fly olfactory circuit mitigate the stability-plasticity dilemma in continual learning Year: (2025)
