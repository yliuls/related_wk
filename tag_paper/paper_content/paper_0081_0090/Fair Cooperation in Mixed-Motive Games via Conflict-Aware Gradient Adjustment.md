Title: Fair Cooperation in Mixed-Motive Games via Conflict-Aware Gradient Adjustment
Abstract: Multi-agent reinforcement learning in mixed-motive settings presents a fundamental challenge: agents must balance individual interests with collective goals, which are neither fully aligned nor strictly opposed. To address this, reward restructuring methods such as gifting and intrinsic motivation have been proposed. However, these approaches primarily focus on promoting cooperation by managing the tradeoff between individual and collective returns, without explicitly addressing fairness with respect to agents' task-specific rewards. In this paper, we propose an adaptive conflict-aware gradient adjustment method that promotes cooperation while ensuring fairness in individual rewards. The proposed method dynamically balances policy gradients derived from individual and collective objectives in situations where the two objectives are in conflict. By explicitly resolving such conflicts, our method improves collective performance while preserving fairness across agents. We provide theoretical results that guarantee monotonic non-decreasing improvement in both the collective and individual objectives and ensure fairness. Empirical results in sequential social dilemma environments demonstrate that our approach outperforms baselines in terms of social welfare, while maintaining fairness.

Section: Introduction
Multi-agent reinforcement learning (MARL) aims to train multiple agents to maximize cumulative rewards in a given task. Depending on the reward structure, MARL is typically categorized into three settings: cooperative, adversarial, and mixed-motive. In the mixed-motive setting, agents' rewards are neither fully aligned (as in cooperative settings) nor entirely opposed (as in adversarial settings), necessitating that each agent balances self-interest with the collective interest. This mixedmotive setting is frequently encountered in real-world applications. For example, in traffic control systems, each agent (e.g., a local intersection controller) may aim to minimize local congestion, which can conflict with global traffic flow optimization if not coordinated. A similar tension happens in sequential social dilemmas (SSDs) such as Cleanup or Harvest [18], where agents must invest in public goods (e.g., cleaning waste or harvesting resources judiciously) that benefit the group but do not yield immediate individual rewards.
However, achieving such a balance in mixed-motive settings is inherently challenging. Excessively selfish behavior by agents can deteriorate collective welfare, which, in turn, negatively impacts each agent's own return-creating a vicious cycle that ultimately harms all participants. Additionally, in some scenarios, certain agents must sacrifice their own returns to improve the collective outcome, potentially leading to unfairness. Conversely, an excessive focus on fairness can hinder learning in tasks that require cooperation. Therefore, it is crucial to enhance collective outcome while ensuring fairness by appropriately balancing individual and collective interests.
In mixed-motive settings, many approaches adopt reward restructuring by incorporating intrinsic rewards such as social influence [12], formal contracts [9], gifting [23,17], and inequity aversion [10]. These methods primarily aim to maximize the collective return by mediating the trade-off between self-interest and collective interests. For example, gifting mechanisms promote cooperation by enabling agents to share a portion of their rewards with others. However, despite their effectiveness in inducing cooperation, such reward restructuring may raise fairness concerns, for example, the gifted reward is intrinsic and not part of the task-defined reward that agents are fundamentally trained to maximize. Consider the Cleanup environment: agents only receive extrinsic rewards for collecting apples, yet apples will only regrow if waste is cleaned. It is often observed that some agents specialize in cleaning waste while others collect apples and subsequently gift a portion of their reward to those who sacrificed their own gain. Although this leads to improved collective performance, the agents engaged in waste cleaning never directly receive task rewards from apple collection. This becomes even worse if the agents are trained with the collective return, since some agents are encouraged to clean the waste all the time. Aside from reward restructuring, an approach has been proposed to align individual and collective objectives by adjusting policy gradients toward stable fixed points of the collective return, while still considering individual interests [20]. However, this method does not adequately consider fairness, as it primarily focuses on stability without explicitly addressing the conflict between individual and collective objectives.
In order to enhance cooperation while ensuring fairness, we propose a fair and conflict-aware gradient adjustment method (FCGrad) that dynamically balances gradients derived from individual and collective objectives by explicitly handling conflicts between them. FCGrad first detects the presence of conflicts, and when conflicts are found, it projects one gradient onto the normal plane of the otherpreserving one objective's direction while avoiding interference with the other. Notably, FCGrad prioritizes the gradient associated with the lower objective value. For example, if the individual objective is lower than the collective objective, indicating that the agent is in an unfair situation, we project the individual gradient onto the normal plane of the collective gradient and use the result as the final update. This enables cooperation to be enhanced while maintaining fairness by resolving conflicts. We provide theoretical results showing that, under certain assumptions, the proposed gradient method guarantees monotonic non-decreasing improvement in both collective and individual objectives. We further show that the two objectives converge to the same value, leading to all agents' objectives aligning-thus ensuring individual fairness. In addition, we empirically demonstrate the effectiveness of FCGrad in terms of α-fairness [25], which captures both performance and fairness, in the Unfair Coin Game and two sequential social dilemma environments: Cleanup and Harvest.
2 Background and Related Works
this section cite: ['b17', 'b11', 'b8', 'b22', 'b16', 'b9', 'b19', 'b24']

Section: Partially Observable Stochastic Game
A Partially Observable Markov Game (POMG) models multi-agent decision-making under uncertainty [21,4]. A POMG is defined as a tuple (N, S, {A i } N i=1 , T, {O i } N i=1 , {R i } N i=1 , γ), where N is the number of agents, S is the set of states, A i is the action set of agent i, T :
S × A 1 × • • • × A N → ∆(S) is the transition function, O i : S → ∆(O i ) is the observation function, R i : S × A 1 × • • • × A N → R
is the reward function for agent i, and γ ∈ [0, 1) is the discount factor. Here, depending on the reward structure, a POMG can represent various types of multi-agent settings: cooperative settings [13,14,15,16], where all agents share an identical reward function (i.e., r 1 = • • • = r N ); adversarial settings [8,31], where agents have directly opposing objectives, often modeled as zero-sum (i.e., N i=1 r i = 0); or mixed-motive settings [24,17], where agents' rewards are neither fully aligned nor strictly opposed, creating simultaneous incentives for both cooperation and competition.
this section cite: ['b20', 'b3', 'b12', 'b13', 'b14', 'b15', 'b7', 'b30', 'b23', 'b16']

Section: Mixed-motive Coordination in Multi-Agent RL
We consider mixed-motive settings, where agents' self-interest often conflicts with collective outcomes. Let us define the collective return as the average of individual returns: R col = 1 N N i=1 R i (s, a), where R i (s, a) is the individual return of Agent i. In the context of gradientbased learning, a conflict occurs when the local and collective return gradients are misaligned, that is, when
∇ θi E R i • ∇ θi E [R col ] < 0, where θ i denotes the parameters of Agent i's policy.
To enhance cooperation (i.e. maximize collective reward) while avoiding conflicts, a variety of approaches have been proposed, including inequity aversion [10,30], social influence [12], reciprocal reward shaping [33], formal contract mechanisms [9], and gifting-based cooperation [23,17]. Many of these approaches are studied in the context of Sequential Social Dilemmas (SSDs) [18], a prominent class of mixed-motive settings in which agents repeatedly arbitrate between short-term selfish actions and long-term collective returns. For example, [17] proposed a gift-based method that balances altrusim and self-interest based based on social relationships between agents. [12] proposed an intrinsic motivation method that rewards agents for exerting causal influence over others' actions, thereby improving coordination in SSDs. [10] introduced inequity-averse agents that learn to cooperate by assigning temporal credit to prosocial behavior and penalizing inequitable outcomes. The aforementioned methods can be broadly viewed as forms of reward shaping, wherein additional intrinsic or socially-informed rewards guide agents toward cooperative behavior.
In contrast to reward shaping approaches, recent work [20] has explored direct optimization in the gradient space to reconcile individual and collective objectives. Specifically, the Altruistic Gradient Adjustment (AgA) method [20] modifies the policy gradients of both the collective and individual objectives, pulling agents toward stable fixed points of the collective objective and pushing them away from unstable ones. The adjusted gradient for Agent i is defined as g i aga = g col + λ(g i ind + H T col g col ), where g col and g ind are the gradients of the collective and individual objectives for Agent i, H T col is the Hessian of the collective return with respect to the policy parameters, and λ is the adjustment coefficient and its sign is determined by sign
[(g col • H T col g col ) (g i ind • H T col g col ) + ∥H T col • g col ∥ 2
]. This adjustment steers the update direction according to the local stability of the collective objective. Despite its effectiveness, AgA incurs additional computational complexity, focuses on the stability of the collective objective rather than directly resolving gradient conflicts, and provides no guarantees of monotonic improvement or fairness.
this section cite: ['b9', 'b29', 'b11', 'b32', 'b8', 'b22', 'b16', 'b17', 'b16', 'b11', 'b9', 'b19', 'b19']

Section: Gradient Adjustment
Gradient adjustment approaches have been actively investigated in multi-task learning [32,22,26,28]. For example, CAGrad [22] formulates a quadratic program to compute a conflict-averse convex combination of gradients, achieving better trade-offs at the cost of increased complexity, and Nash-MTL [26] frames the task-weighting problem as a bargaining game, using the Nash bargaining solution to promote fairness and efficiency across tasks. Another method that inspires our work is PCGrad [32], which mitigates conflicts by projecting each conflicting gradient onto the normal plane of the other, offering a simple yet effective solution with low computational overhead. Specifically, when two gradients g 1 and g 2 are conflicted, PCGrad adjusts them by projecting one onto the normal plane of the other, i.e., gP CGrad
1 = g 1 -g1•g2
∥g2∥ 2 g 2 , and then uses the average of gP CGrad 1 and gP CGrad 2 as the final update.
this section cite: ['b31', 'b21', 'b25', 'b27', 'b21', 'b25', 'b31']

Section: Fairness in Multi-agent RL
Fairness concerns how returns are distributed among agents rather than how large the total return is, making it complementary, but often orthogonal to cooperation and efficiency. Fairness has been considered in multi-agent RL literature in both cooperative and mixed-motive settings [34,6,1,29]. For example, in cooperative settings, [34] formulates fairness as the optimization of a fair social welfare function and [6] proposes a method for achieving team fairness by enforcing permutationequivariant policies, which mitigate emergent unfairness caused by asymmetric role assignment. In mixed-motive settings, [17] shows enhanced fairness when measuring the sum of individual rewards and gifts, whereas in this paper we evaluate fairness using individual rewards only. [10], inspired by the literature on inequality in economics [5], explicitly leverages fairness by adding both disadvantage and advantage inequality terms to the reward of each agent to improve cooperation in SSD. Specifically, the shaped reward for Agent is r i = r i -α IA /(N -1) j̸ =i max(r j -r i , 0)β IA /(N -1) j̸ =i max(r i -r j , 0), where α IA and β IA weight disadvantage and advantage inequity, respectively.
Note that throughout this paper, we define fairness in terms of task-defined extrinsic individual rewards, the quantities that agents are fundamentally trained to maximize, and do not consider intrinsic rewards such as gifting, as they do not directly reflect actual participation in the underlying task. A more detailed discussion on this assumption is provided in Appendix A.
Algorithm 1: FCGrad Input: Policy parameters θ, learning rate η, weighting factor
β 1 Compute g ind := ∇ θ V ind (θ), g col := ∇ θ V col (θ) 2 if ⟨g ind , g col ⟩ ≥ 0 then 3 g FCGrad ← (1 -β)g ind + βg col ; 4 else 5 if V col (θ) ≥ V ind (θ) then 6 g FCGrad ← g ind - ⟨g col , g ind ⟩ ∥g col ∥ 2 g col ; 7 else 8 g FCGrad ← g col - ⟨g ind , g col ⟩ ∥g ind ∥ 2 g ind ; 9 Return θ ← θ + ηg FCGrad ;
(a) Conflict (b) Non-conflict
this section cite: ['b33', 'b5', 'b0', 'b28', 'b33', 'b5', 'b16', 'b9', 'b4']

Section: Methodology
In mixed-motive settings, the individual and collective objectives may be either aligned or in conflict. When they are aligned, optimizing both objectives is sufficient, as neither interferes with the other. In such cases, an appropriately weighted combination of the two can be effective. However, when the objectives are in conflict, it becomes essential to explicitly address the interference between them, as prioritizing one may hinder the other. This is because focusing solely on the individual objective may hinder learning in tasks where cooperative behavior is essential for maximizing individual returns, while focusing solely on the collective objective may compromise fairness among agents. Therefore, it is important to ( 1) recognize when such conflicts arise and (2) correspondingly adjust the individual and collective objectives, appropriately considering both fairness and cooperation.
To this end, we propose a fair and conflict-aware gradient adjustment method, called FCGrad, which guarantees the monotonic non-decrease of both individual and collective objectives, while preserving fairness across individual objectives. Specifically, when the individual and collective gradients are in conflict, FCGrad projects the gradient associated with the lower expected return onto the normal plane of the other. This projected gradient remains a valid ascent direction for its own objective while avoiding interference with the other, and is then used as the final update. For example, if the individual expected return is lower than the collective expected return, indicating that the agent is disadvantaged in terms of fairness, we project the gradient of the individual objective onto the normal plane of the collective gradient and use it as the update direction. The detailed procedure and a visual illustration of FCGrad are provided in Algorithm 1 and Fig. 1, respectively. In the following, we present the detailed method along with its theoretical analysis.
this section cite: []

Section: FCGrad: Fair and Conflict-aware Gradient Adjustment
We now describe how FCGrad operates from the perspective of Agent i. Let θ ∈ R d denote the parameters of the policy π θ for Agent i. Let us define V ind (θ) and V col (θ) as the expected individual and collective returns, respectively, computed under the initial state distribution. Note that V ind (θ) and V col (θ) are the individual and collective objectives, respectively. Let g ind := ∇ θ V ind (θ) and g col := ∇ θ V col (θ) denote the gradients of the individual and collective objectives, respectively. These represent ascent directions for V ind (θ) and V col (θ), meaning that for a sufficiently small η > 0, the following holds:
V ind (θ + ηg ind ) > V ind (θ) and V col (θ + ηg col ) > V col (θ).
FCGrad proceeds as follows: (1) check whether g ind and g col are in conflict by examining the sign of their inner product, where a negative inner product indicates the presence of a conflict.
(2) if ⟨g ind , g col ⟩ ≥ 0 (i.e., non-conflict), FCGrad uses the weighted sum of two gradients: g = (1 -β)g ind + βg col , (3) ⟨g ind , g col ⟩ < 0 (i.e., conflict), FCGrad places more weight on the individual (collective) gradient when the collective (individual) objective is greater, in order to ensure fairness.
The corresponding gradient is given by
g FCGrad = gind if (V col ≥ V ind ) gcol if (V col < V ind )(1)
where gcol and gind are the projections of g col and g ind , respectively, onto the normal plane of another gradient vector, given by
gcol := g col - ⟨g ind , g col ⟩ ∥g ind ∥ 2 g ind , gind := g ind - ⟨g col , g ind ⟩ ∥g col ∥ 2 g col(2)
(4) update the policy parameter with the step size η: θ ← θ + ηg. Note that gcol projects g col onto the normal plane of g ind . Thus, gcol is still a valid ascent direction for the collective objective while preserving the individual reward. This indicates that FCGrad prioritizes the individual objective without compromising the collective one when the agent is in an unfair situation, i.e., when the individual objective is lower. Conversely, when the collective objective is lower, FCGrad prioritizes the collective objective without compromising the individual one.
this section cite: []

Section: Theoretical Analysis
In this section, we prove that FCGrad guarantees monotonically non-decreasing improvements in both the collective and individual objectives, and that both objectives converge to the same value. This ensures that the expected individual returns across agents also converge to the same value.
Theorem 3.1 Assume V ind (θ) and V col (θ) are differentiable and L-smooth. Let the update direction g be defined as in Equation 1. Then, for a sufficiently small step size η > 0, the update θ ← θ + ηg yields monotonically non-decreasing improvements in both V col (θ) and V int (θ).
Proof. See Appendix B.
Theorem 3.1 states that FCGrad ensures monotonic non-decreasing improvements in both V ind (θ t ) and V col (θ t ) under certain assumptions. Note that all agents are updated using FCGrad, so both the individual objectives of all agents and the collective objective, defined as the expected return averaged across agents, are improved accordingly. However, monotonic improvement alone does not guarantee fairness. To establish fairness, it is necessary to further show that the individual and collective values converge to the same value over time, which in turn implies that all agents' individual values also become equal. The next theorem formalizes this result by proving that the gap between V ind (θ t ) and V col (θ t ) vanishes under mild conditions.
Theorem 3.2 Assume V ind (θ) and V col (θ) be L-smooth, and let δ t := V ind (θ t ) -V col (θ t ) denote the value gap at iteration t. Assume the step size satisfies the Robbins-Monro conditions: 0 < η t ≤ |δ t |/L with t η t = ∞ and t η 2 t < ∞. Also assume conflict recurrence, meaning that for any ϵ > 0 and any t, if |δ t | ≥ ϵ, then there exists t ′ ≥ t such that (g ind,t ′ • g col,t ′ ) < 0. Then, the value gap converges to zero:
lim t→∞ |V ind (θ t ) -V col (θ t )| = 0.(3)
Proof. See Appendix B.
Theorem 3.2 states that the gap between the collective and individual objectives converges to zero under certain assumptions, including conflict recurrence, where conflicts occur continuously. This assumption is reasonable in mixed-motive settings, especially near equilibrium, because agents face inherent tensions between cooperation and self-interest, and as they approach equilibrium, misalignments in their objectives can continue to induce conflicts, even with small policy updates. Under the assumption that all agents use FCGrad, the individual objectives of all agents converge to the collective objective, and thus all individual objectives converge to the same value. This, in turn, implies that individual fairness is achieved.
this section cite: []

Section: Practical Algorithm
We now introduce a practical FCGrad-based multi-agent RL algorithm for mixed-motive settings. We consider decentralized training and execution, where each agent does not have access to other agents' information but shares rewards, as commonly assumed in gifting mechanisms [23,17]. Each agent trains its policy and value functions for both individual and collective returns solely based on its own local observations and the shared rewards. For this, we construct two separate value networks for the individual and collective objectives, while sharing a common encoder between them. Each value function is trained using generalized advantage estimation [27] to compute the corresponding advantage estimates. Using the two value functions, we compute the policy gradients of the individual and collective objectives, denoted as g ind and g col , respectively, via the PPO policy gradient. These gradients are then combined using FCGrad to determine the final update direction.
this section cite: ['b22', 'b16', 'b26']

Section: Experimental Results

this section cite: []

Section: Experimental Setup
Environments We conduct our experiments using the JAX-based codebase and environments provided by the SocialJAX suite [7]. We modify the existing environments-Coins, Cleanup, and Harvest-to incorporate a fairness perspective. Specifically, since Cleanup and Harvest already involve inherent fairness dilemmas, we introduce only minor changes by assigning distinct respawn positions to the agents. For the Coin Game, which originally focuses on the conflict between individual and collective objectives, we introduce asymmetry in the potential rewards that agents can obtain, creating a disparity in individual incentives. Fig. 2 illustrates the considered environments. We provide detailed descriptions in the following sections.
Metric As our goal is to maximize returns while ensuring fairness, both performance and fairness metrics should be jointly considered for evaluation. We use α-fairness [25] as the evaluation metric, where, given individual returns (r 1 , • • • , r N ), the fairness utility is defined as
U α (r 1 , • • • , r N ) = N i=1 r 1-α i 1-α , if α ̸ = 1, N i=1 log(r i ), if α = 1.(4)
Notably, the fairness utility recovers several well-known objectives for specific values of α: it corresponds to the collective return when α = 0, the geometric mean of individual rewards-also known as Nash Social Welfare-when α = 1, and the minimum individual reward when α → ∞. Thus, α = 0 reflects no consideration of fairness, and as α increases, the evaluation increasingly prioritizes fairness over aggregate performance. In summary, we consider the following three representative instances of α-fairness return in our evaluation: (i) average return (Mean, α = 0), (ii) geometric mean return (GeoMean, α = 1), and (iii) minimum individual return (Min, α → ∞). Note that α-fairness return considers both performance and fairness, where α determines the trade-off between them. The reported results are averaged over four random seeds.
Baselines We evaluate FCGrad with six baselines: (a) collective reward optimization (Col), (b) individual reward optimization (Ind), (c) inequity aversion reward restructuring (IA) [10], (d) weighted gradient combination of g ind and g col (denoted as Weighted), which corresponds to FCGrad without conflict handling, (e) PCGrad [32], and (f) Altruistic Gradient Adjustment (AgA) [20]. Note that baselines (d)-(f) use the same architecture as FCGrad, where two separate value functions are trained for individual and collective objectives; they differ only in the policy update rule based on g ind and g col . All methods are implemented on top of the IPPO [3].
Hyperparameter We introduce a hyperparameter β for FCGrad, which determines the weight between the collective and individual objectives when there is no conflict. β plays a particularly important role in tasks that require high-level cooperation. We set β to 0.5, 0.7, and 0.8 for the Unfair Coin Game, Cleanup, and Harvest, respectively. The same values of β are used for the baseline method, Weighted. Additional hyperparameters for IPPO are provided in Appendix C.
this section cite: ['b6', 'b24', 'b9', 'b31', 'b19', 'b2']

Section: Unfair Coins
The Coins environment [19] consists of two agents (green and red) and two types of coins, each associated with one of the agents. When a coin appears, it is assigned a color with probabilities p green and p red . An agent receives a reward of 1 for collecting any coin, regardless of its color. However, collecting a coin of the opposite color imposes a penalty of -2 on the other agent, creating a conflict between individual gain and cooperative behavior. In contrast to the original setting [19], where p green and p red are both set to 0.5-so that collecting coins matching each agent's color naturally aligns with fairness and also maximizes the collective reward-we consider an unfair variant where p green = 15/16 and p red = 1/16, introducing an inherent asymmetry in coin appearances. Although optimal collective performance still requires agents to collect coins matching their own color, this setup raises a fairness concern: the green agent receives substantially more rewards due to the higher frequency of green coins. To mitigate this imbalance and achieve a fairer outcome, the green agent must occasionally yield coins to the red agent, sacrificing some collective reward in favor of equity. Results. In the Unfair Coin environment, achieving fairness requires the green agent to yield some of its coins to the red agent, thereby reducing its own reward. In other words, there exists a strong trade-off between collective performance and fairness. Therefore, we particularly focus on the performance trend with respect to α, as well as the Min performance, which places greater emphasis on fairness-the return of the most disadvantaged agent.
Fig. 3 presents the α-fairness returns in the unfair coin environment (top) and the individual return of the green and red agents (bottom). The performance of Col, Ind, and AgA is observed to decrease more dramatically as α increases compared to FCGrad and PCGrad, which are conflict-aware methods. Interestingly, both the collective and individual approaches result in extremely unfair outcomes, but in opposite directions. Col, which maximizes collective reward, trains both agents to collect their own coins. As a result, the green agent, with more coin opportunities, gains higher returns, leading to unfairness. In contrast, with the individual objective, the red agent outperforms the green agent, possibly because the green agent is more frequently penalized by negative rewards due to the abundance of green coins. Meanwhile, the red agent learns without such penalties, accelerating its progress. However, FCGrad shows little variation across agents as α changes, indicating achieved fairness. Notably, FCGrad outperforms the baselines in terms of Min performance. As shown in Fig. 3, both the green and red agents converge to nearly identical returns, showing that fairness is effectively achieved.
this section cite: ['b18', 'b18']

Section: Cleanup
The Cleanup environment consists of N = 4 agents, apples, and waste. Each agent receives a reward of 1 for collecting an apple. Apples grow in an orchard, but their growth depends on the amount of waste present in the environment. Waste accumulates at a constant rate, and beyond a certain threshold, apple growth ceases entirely. Therefore, in order to sustain apple regrowth, some agents must sacrifice their immediate reward by cleaning up the waste. This creates a social dilemma, as the necessary act of cleaning benefits the group but does not provide direct individual reward, thereby generating a tension between self-interest and cooperative behavior. In contrast to the original configuration, where agents are randomly spawned across the map, we fix the spawn positions of agents: some (Agents 2 and 3 in our case) are placed near the apple orchard, while others (Agents 0 and 1) are positioned closer to the waste area. This spatial asymmetry further amplifies the conflict between fairness and efficiency. Note that, unlike the Unfair Coin, Cleanup introduces an intertemporal perspective, involving a trade-off between short-term individual interest and long-term collective interest [10].
Results. Fig. 4 (a) and (b) show the α-fairness performance and individual rewards during training in the Cleanup environment. FCGrad outperforms the baselines in terms of both GeoMean and Min, which reflect not only total return but also fairness. In addition, FCGrad achieves comparable performance to Col in terms of Mean, which is the optimization target of Col. As shown in Fig. 4 (b), under Col, Agent 3 learns to monopolize apple collection, while Agent 0 is trained to sacrifice by primarily cleaning waste. In contrast, FCGrad leads all four agents to obtain reasonably similar returns-demonstrating more fair behavior and achieving the best result in terms of Min. Since using the collective reward is essential in this environment, methods that rely heavily on individual rewards, such as Ind and IA, fail to learn effectively. In addition, AgA fails to properly balance between individual and collective objectives, also struggle to learn successfully.
this section cite: ['b9']

Section: Harvest
The Harvest environment features N = 4 agents and apples distributed across orchard patches. Each agent receives a reward of 1 per apple, but regrowth is stochastic and depends on nearby apples within a fixed radius. Over-harvesting depletes resources, risking environmental collapse, and thus agents must coordinate implicitly to sustain long-term returns. This creates a social dilemma between short-term individual gain and long-term collective benefit. We also introduce spatial asymmetry: Agents 0 and 1 spawn near apples, while Agents 2 and 3 spawn farther away, making collection easier for the former. Similar to the Cleanup, Harvest also poses intertemporal challenges for both cooperation and fairness. Results. Fig. 4 (c) and (d) show the α-fairness returns and individual agent returns during training. FCGrad outperforms the baselines across the considered α values. With the Col, Agents 0 and 1 achieve higher returns than Agents 2 and 3, indicating that they focus solely on collecting apples while accounting for the intertemporal dilemma, but not addressing the resulting unfairness toward Agents 2 and 3. In contrast, FCGrad leads all four agents to achieve similar returns, implying that Agents 0 and 1 take into account the outcomes of Agents 2 and 3. Similar to the results in Cleanup, methods that rely heavily on individual rewards, such as Ind and IA, perform poorly, though they achieve marginal learning. AgA performs better than the individual-reward-based methods, but still underperforms compared to FCGrad. Weighting factor: β determines the balance between the collective and individual objectives when no conflict is detected. It plays a particularly important role in tasks that require highlevel cooperation. For example, in Cleanup, ignoring the collective objective makes it difficult for agents to discover how to improve their individual rewards. We observed this phenomenon in the previous section-solely maximizing individual rewards does not perform well. We present the GeoMean performance of FCGrad in the Harvest environment in Fig. 5, which shows that a β value between 0.7 and 0.8 yields the best performance. Thus, β reflects the required degree of cooperation over self-interest.
this section cite: []

Section: Additional Analysis: Ablation and Fairness Metrics
Additional Fairness metrics: We additionally evaluate fairness using the Gini coefficient [2] and Jain's index [11].
The Gini coefficient is defined as Gini(r 1 , • • • , r N ) = N i=1 N j=1 |ri-rj | 2N N i=1 ri and Jain's index is defined as Jain(r 1 , • • • , r N ) = ( N i=1 ri) 2 N N i=1 r 2 i
, where both metrics range between 0 and 1 and lower Gini and higher Jain values indicate better fairness. Table 1 presents the results, showing that FCGrad generally achieves superior fairness.
this section cite: ['b1', 'b10']

Section: Conclusion
In this work, we address the long-standing challenge of achieving both cooperation and fairness in mixed-motive multi-agent RL. We propose FCGrad, a conflict-aware gradient adjustment method that explicitly resolves gradient-level conflicts between individual and collective objectives. FCGrad dynamically adjusts the update direction based on which objective is more disadvantaged by projecting one gradient onto the normal plane of the other. We theoretically prove that this mechanism guarantees monotonic improvement and convergence of both objectives to the same value. Consequently, individual objectives across agents also converge, ensuring fairness. Extensive experiments in the Unfair Coin environment and sequential social dilemma settings, Cleanup and Harvest, demonstrate that FCGrad not only improves overall performance but also achieves superior fairness, as measured by α-fairness return metrics.
this section cite: []

Section: Limitation
In practice, the recurrence of gradient conflicts, required for our theoretical guarantee, may not hold, as it can be influenced by the weighting factor in non-conflict cases. Understanding this interplay is a promising direction for future work.
Broader Impact Our work promotes fairness in learned behaviors, potentially preventing emergent inequalities in decentralized systems. We believe it has a positive societal impact.
this section cite: []

Section: Acknowledgement
This work was supported by the ONR MURI grant N00014-25-1-2116.
this section cite: []

Section: A Discussion on Fairness
In this appendix, we clarify the modeling assumption underlying our notion of fairness and contrast it with alternative perspectives such as reward-redistribution-based fairness. Our formulation intentionally focuses on a different regime: fairness is grounded purely in task-defined extrinsic rewards that directly reflect each agent's actual behavior, rather than assuming the availability of contract or currency-like mechanisms (e.g., reward exchanges or gifting) for compensating agents.
this section cite: []

Section: A.1 Extrinsic-Reward-Based Fairness vs. Reward Redistribution
Prior approaches such as gifting or incentive mechanisms (e.g., [9,17,23]) allow agents to redistribute rewards among one another-often interpreted as a currency-like signal or contract that enables division of labor. Under such assumptions, reward transfers are considered real, tangible returns and can be used to compensate agents for sacrificial roles (e.g., pollution cleaning without harvesting apples).
In contrast, our work adopts a more primitive perspective of fairness: we consider only task-defined extrinsic rewards that arise directly from environment state-action outcomes (e.g., rewards from collecting apples in the Cleanup environment). We intentionally do not assume the existence of an auxiliary payment mechanism, such as money or transferable reward tokens, that is external to the environment dynamics. From this viewpoint, if one agent continuously cleans while others only harvest apples, such an outcome is deemed unfair unless the cleaner also receives direct extrinsic returns. This modeling choice focuses on fairness that reflects actual participation in the task, rather than contractual compensation.
this section cite: ['b8', 'b16', 'b22']

Section: A.2 Pareto Optimality and the Role of α-Fairness
We emphasize that our objective is not to compute or approximate a game-theoretic equilibrium (e.g., Nash or correlated equilibrium), but rather to learn Pareto-optimal outcomes. In particular, α-fairness is used only as an evaluation metric, not as a training objective. By varying α, one can evaluate different trade-offs between pure efficiency (α = 0), multiplicative balance (α = 1), and max-min fairness (α → ∞). FCGrad yields outcomes that lie on the Pareto frontier across these trade-offs: in terms of collective return it performs comparably to the best baselines, while in terms of α = 1 or α = ∞ it significantly improves fairness without degrading performance.
this section cite: []

Section: B Theoretical Results
Lemma B.1 Let J : R d → R be a continuously differentiable and L-smooth function. Let g 1 = ∇ θ J(θ) be the gradient of J at point θ, and let g 2 ∈ R d be any vector satisfying ⟨g 1 , g 2 ⟩ > 0. Then, for small step size η < 2⟨g1,g2⟩
L∥g2∥ 2 , the update θ ← θ + ηg 2 yields a strict improvement: J(θ + ηg 2 ) > J(θ).
Proof. Since J is L-smooth, for any θ ∈ R d , update direction g 2 ∈ R d , and step size η > 0, the following inequality holds:
J(θ + ηg 2 ) ≥ J(θ) + η⟨∇ θ J(θ), g 2 ⟩ - L 2 η 2 ∥g 2 ∥ 2 .
Let g 1 = ∇ θ J(θ). Then:
J(θ + ηg 2 ) ≥ J(θ) + η⟨g 1 , g 2 ⟩ - L 2 η 2 ∥g 2 ∥ 2 .
Define the right-hand side as a function of η:
∆(η) := η⟨g 1 , g 2 ⟩ - L 2 η 2 ∥g 2 ∥ 2 .
Since ⟨g 1 , g 2 ⟩ > 0, this is a concave quadratic function that is positive for small enough η. Specifically, the inequality ∆(η) > 0 holds when:
η < 2⟨g 1 , g 2 ⟩ L∥g 2 ∥ 2 .
Therefore, for any η ∈ 0, 2⟨g1,g2⟩ L∥g2∥ 2 , we have:
J(θ + ηg 2 ) > J(θ).
Theorem B.2 Assume V ind (θ) and V col (θ) are differentiable and L-smooth. Let the update direction g be defined as in Equation 1. Then, for a sufficiently small step size η > 0, the update θ ← θ + ηg yields monotonically non-decreasing improvements in both V col (θ) and V int (θ).
We consider three cases:
Case 1: (Non-conflict) g ind • g col ≥ 0. Then g = βg ind + (1 -β)g col .
Since g ind , g col are ascent directions for V ind , V col , respectively, their convex combination also satisfies:
g ind • g = β∥g ind ∥ 2 + (1 -β)g ind • g col > 0 (5
) g col • g = βg col • g ind + (1 -β)∥g col ∥ 2 > 0(6)
Since g ind • g and g ind • g are positive, according to Lemma 3.1, g yields a strict improvement in both V ind and V col .
Case 2: (Conflict) g ind • g col < 0 and V ind (θ) < V col (θ). We then use: g = g ind -g col •gind ∥g col ∥ 2 g col . Now, g ind • g = g ind • g ind - (g ind • g col ) ∥g col ∥ 2 (g ind • g col ) = ∥g ind ∥ 2 ∥g col ∥ 2 -(g ind • g col ) 2 ∥g col ∥ 2 > 0 g col • g = g col • g ind - (g ind • g col ) ∥g col ∥ 2 ⟨g col , g col ⟩ = 0 (7
)
Since g ind • g is positive, according to Lemma 3.1, g yields a strict improvement in V ind . In addition, since g col • g is zero, g does not decrease V col .
Case 3: (Conflict) g ind • g col < 0 and V ind (θ) > V col (θ). Symmetric to Case 2: g yields a strict improvement in both V col and does not decrease V ind .
Thus, in all cases, g induces monotonically non-decreasing improvements in V ind and V col .
this section cite: []

Section: Lemma B.3 (Single conflict step)
When the conflict happens (i.e., (g ind • g col ) < 0), then for sufficiently small step size, 0 ≤ η t ≤ ∥δ t ∥/L, we have
L t+1 -L t ≤ - η t 2 |δ t | ∥d t ∥ 2 .(8)
Proof. When δ t < 0 (i.e. V col > V ind ), we use g = g ind -gind•gcol ∥gcol∥ 2 g col . Since L is L-smooth function, the following holds
L t+1 -L t ≤ η t (∇L t • g) + L 2 η 2 t ∥g∥ 2(9)
Here,
(∇L t • g) = δ t (g ind -g col ) • g = δ t (g ind • g -g col • g) = δ t (g ind • g) = δ t ∥g∥ 2(10)
Thus, we have
L t+1 -L t ≤ η t δ t ∥g∥ 2 + L 2 η 2 t ∥g∥ 2 ∥ ≤ η t δ t ∥g∥ 2 + ∥δ t ∥η t 2 ∥g∥ 2 = - η t 2 ∥δ t ∥∥g∥ 2(11)
Lemma B.4 (Single non-conflict step) When the conflict does not happen, (i.e., (g ind • g col ) ≥ 0), the proposed gradient is used. We assume that the step size meets the Robbins-Monro conditions (i.e. ∞ t=0 η t = ∞, ∞ t=0 η 2 t < ∞.) Then, the following holds: t∈N
∥L t+1 -L t ∥ < ∞ (12
)
where N is the set of all non-conflict indices.
Proof. g = βg ind + (1 -β)g col . Let us define G := sup t ∥g 1,t ∥ + ∥g 2,t ∥ (< ∞).
Since L is L-smooth, we have
L t+1 -L t ≤ η t ⟨∇ θ L t , g⟩ + L 2 η 2 t ∥g∥ 2 . (13
) Since ∇ θ L t = δ t (g ind -g col ), ∥⟨∇ θ L t , g⟩∥ = ∥δ t ⟨g ind -g col , βg ind + (1 -β)g col ⟩∥ (14
) ≤ |δ t | β∥g ind ∥∥g ind -g col ∥ + (1 -β)∥g col ∥∥g ind -g col ∥ (Cauchy-Schwarz) (15
)
≤ |δ t | β∥g ind ∥ + (1 -β)∥g col ∥ 2G ≤ 2G 2 |δ t |.(16)
Based on the assumption of the step size η (η t ≤ |δ t |/L), we have
|η t ⟨∇ θ L t , d t ⟩| ≤ 2G 2 |δ t | η t ≤ 2G 2 L η 2 t . (C) Since ∥g∥ = ∥βg ind + g col ∥ ≤ β∥g ind ∥ + (1 -β)∥g col ∥ ≤ G, the following holds. L 2 η 2 t ∥g∥ 2 ≤ L 2 η 2 t G 2(17)
Combined above, we have
∥L t+1 -L t ∥ ≤ (2G 2 L + L 2 G 2 )η 2 t = 5 2 G 2 Lη 2 t (18
) Define C 0 := 5 2 G 2 L to obtain |L t+1 -L t | ≤ C 0 η 2 t . Because ∞ t=0 η 2 t < ∞ (Robbins-Monro assumption), t∈N |L t+1 -L t | ≤ C 0 t∈N η 2 t ≤ C 0 ∞ t=0 η 2 t < ∞.
Theorem B.5 Let V ind and V col be L-smooth. Assume the step size satisfies the Robbins-Monro conditions: 0 < η t ≤ |δ t |/L with t η t = ∞ and t η 2 t < ∞. Also assume conflict recurrence, meaning that for any ϵ > 0 and any t, if |δ t | ≥ ϵ, then there exists t ′ ≥ t such that (g ind,t ′ • g col,t ′ ) < 0. Then, the value gap converges to zero:
lim t→∞ |V ind (θ t ) -V col (θ t )| = 0.(19)
Proof. Denote conflict indices by C and non-conflict by N . Lemma A.3 and Lemma A.4 give for every horizon T L T ≤ L 0 -1 2 t∈C, t<T η t |δ t | ∥d t ∥ 2 + C 0 t∈N , t<T
η 2 t .(20)
According to the assumption of the Robbins-Monro, the following holds:
t∈C η t |δ t | ∥d t ∥ 2 < ∞.(21)
For any conflict step the projection property and bounded gradients imply ∥g t ∥ ≥ σ > 0 with σ := 1 2 min(∥g ind,t ∥, ∥g col,t ∥). Thus, we have
t∈C η t |δ t | ≤ σ -2 t∈C η t |δ t |∥g t ∥ 2 < ∞.(22)
Here, we use contradiction.
this section cite: []

Section: C Implementation Details
All experiments were run on a local server equipped with an AMD EPYC 7713 64-Core CPU and five NVIDIA RTX 6000 Ada Generation GPUs. Each rollout consisted of 64-256 parallel environments depending on the task, and training time per run ranged from 2 to 8 hours. The official implementation of FCGrad is available at: https://github.com/wjkim1202/fcgrad.
this section cite: []

Section: C.1 Unfair Coin
Each agent has a CNN-based actor-critic network. The observation is processed through three convolutional layers with kernel sizes of 5 × 5, 3 × 3, and 3 × 3, each with 32 channels and ReLU activations, followed by a fully connected layer with 64 units. The actor head outputs a categorical distribution over discrete actions, while the critic consists of two separate heads estimating the individual and collective value functions.
We train the networks using the Adam optimizer with a learning rate of 1 × 10 -4 , linearly annealed over time. PPO is used with a clipping threshold of 0.2 and two update epochs per iteration, using 500 minibatches. We collect trajectories from 256 parallel environments, each running for 1000 steps per rollout. The discount factor is set to γ = 0.99 and the GAE parameter to λ = 0.95. The entropy and value loss coefficients are set to 0.1, respectively. Gradients are clipped to a maximum global norm of 0.5.
this section cite: []

Section: C.2 Cleanup
Each agent is equipped with a convolutional actor-critical network. The observation is processed through three convolutional layers with kernel sizes of 5 × 5, 3 × 3, and 3 × 3, each with 32 channels and ReLU activations, followed by a fully connected layer with 64 units. The actor outputs a categorical distribution over discrete actions, and the critic consists of two heads that estimate the individual and collective value functions, respectively.
Training is performed using PPO with a clipping threshold of 0.2 and two update epochs per iteration. A total of 500 minibatches are used per update, with data collected from 64 parallel environments running 1000 steps per rollout. The discount factor is set to γ = 0.99, and the GAE parameter is set to λ = 0.95. We use the Adam optimizer with an initial learning rate of 5 × 10 -4 , which is linearly annealed during training. The value loss coefficient and entropy coefficient are both set to 0.01, and the value function loss is weighted by 0.5. Gradients are clipped with a maximum global norm of 0.5.
this section cite: []

Section: C.3 Harvest
Each agent is equipped with a convolutional actor-critical network. The observation is processed through three convolutional layers with kernel sizes of 5 × 5, 3 × 3, and 3 × 3, each with 32 channels and ReLU activations, followed by a fully connected layer with 64 units. The actor outputs a categorical distribution over discrete actions, and the critic consists of two heads that estimate the individual and collective value functions, respectively.
Training is performed using PPO with a clipping threshold of 0.2 and two update epochs per iteration. A total of 500 minibatches are used per update, with data collected from 64 parallel environments running 1000 steps per rollout. The discount factor is set to γ = 0.99, and the GAE parameter is set to λ = 0.95. We use the Adam optimizer with an initial learning rate of 5 × 10 -4 , which is linearly annealed during training. The entropy and value function loss coefficients are set to 0.01 and 0.5, respectively. Gradients are clipped to a maximum global norm of 0.5.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The abstract and introduction clearly state the core contributions and are consistent with both theoretical and empirical results.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We stated the limitation in the conclusion.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: We included the proof in the Appendix. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We provided the details in the Appendix and the main paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We cited the corresponding paper.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We provided the implementation details in the Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We provided the mean and variance of individual returns.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We stated this in the Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We follow the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [Yes] Justification: We stated the broader impacts in the conclusion. Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: [NA] Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: We have cited the paper. Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] Justification: [NA]
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: [NA]
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: [NA] Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: [NA] Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Cooperation and fairness in multi-agent reinforcement learning Year: (2024)
Ref_id:b1 Title: Miscellanea: Gini's mean difference rediscovered Year: (1968)
Ref_id:b2 Title: Is independent learning all you need in the starcraft multi-agent challenge? arXiv preprint Year: (2020)
Ref_id:b3 Title: Approximate solutions for partially observable stochastic games with common payoffs Year: (2004)
Ref_id:b4 Title: Inequality aversion, efficiency, and maximin preferences in simple distribution experiments Year: (2004)
Ref_id:b5 Title: Cooperative multi-agent fairness and equivariant policies Year: (2022)
Ref_id:b6 Title: Socialjax: An evaluation suite for multi-agent reinforcement learning in sequential social dilemmas Year: (2025)
Ref_id:b7 Title: What is the solution for state-adversarial multi-agent reinforcement learning? Year: ()
Ref_id:b8 Title: Formal contracts mitigate social dilemmas in multi-agent reinforcement learning Year: (2024)
Ref_id:b9 Title: Inequity aversion improves cooperation in intertemporal social dilemmas Year: (2018)
Ref_id:b10 Title: A quantitative measure of fairness and discrimination Year: (1984)
Ref_id:b11 Title: Social influence as intrinsic motivation for multi-agent deep reinforcement learning Year: (2019)
Ref_id:b12 Title: Maser: Multi-agent reinforcement learning with subgoals generated from experience replay buffer Year: (2022)
Ref_id:b13 Title: A variational approach to mutual information-based coordination for multi-agent reinforcement learning Year: (2023)
Ref_id:b14 Title: An adaptive entropy-regularization framework for multiagent reinforcement learning Year: (2023)
Ref_id:b15 Title: Parameter sharing with network pruning for scalable multiagent deep reinforcement learning Year: (2023)
Ref_id:b16 Title: Learning to balance altruism and self-interest based on empathy in mixed-motive games Year: ()
Ref_id:b17 Title: Multi-agent reinforcement learning in sequential social dilemmas Year: (2017)
Ref_id:b18 Title: Maintaining cooperation in complex social dilemmas using deep reinforcement learning Year: (2017)
Ref_id:b19 Title: Aligning individual and collective objectives in multi-agent cooperation Year: ()
Ref_id:b20 Title: Markov games as a framework for multi-agent reinforcement learning Year: (1994)
Ref_id:b21 Title: Conflict-averse gradient descent for multi-task learning Year: (2021)
Ref_id:b22 Title: Gifting in multi-agent reinforcement learning Year: (2020)
Ref_id:b23 Title: Social diversity and social preferences in mixed-motive reinforcement learning Year: (2020)
Ref_id:b24 Title: Fair end-to-end window-based congestion control Year: (2000)
Ref_id:b25 Title: Multi-task learning as a bargaining game Year: (2022)
Ref_id:b26 Title: Highdimensional continuous control using generalized advantage estimation Year: (2015)
Ref_id:b27 Title: Independent component alignment for multi-task learning Year: (2023)
Ref_id:b28 Title: Learning fair cooperation in mixed-motive games with indirect reciprocity Year: (2024)
Ref_id:b29 Title: Evolving intrinsic motivations for altruistic behavior Year: (2019)
Ref_id:b30 Title: Multi-agent adversarial inverse reinforcement learning Year: (2019)
Ref_id:b31 Title: Gradient surgery for multi-task learning Year: (2020)
Ref_id:b32 Title: Reciprocal reward influence encourages cooperation from self-interested agents Year: (2024)
Ref_id:b33 Title: Learning fair policies in decentralized cooperative multi-agent reinforcement learning Year: (2021)
