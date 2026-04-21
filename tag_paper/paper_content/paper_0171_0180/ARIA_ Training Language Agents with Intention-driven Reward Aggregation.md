Title: ARIA: Training Language Agents with Intention-Driven Reward Aggregation
Abstract: Large language models (LLMs) have enabled agents to perform complex reasoning and decision-making through free-form language interactions. However, in openended language action environments (e.g., negotiation or question-asking games), the action space can be formulated as a joint distribution over tokens, resulting in an exponentially large action space. Sampling actions in such a space can lead to extreme reward sparsity, which brings large reward variance, hindering effective reinforcement learning (RL). To address this, we propose ARIA, a method that Aggregates Rewards in Intention space to enable efficient and effective language Agents training. ARIA aims to project natural language actions from the highdimensional joint token distribution space into a low-dimensional intention space, where semantically similar actions are clustered and assigned shared rewards. This intention-aware reward aggregation reduces reward variance by densifying reward signals, fostering better policy optimization. Extensive experiments demonstrate that ARIA not only significantly reduces policy gradient variance, but also delivers substantial performance gains of an average of 9.95% across four downstream tasks, consistently outperforming offline and online RL baselines.

Section: Introduction
Large language models (LLMs) have demonstrated strong capabilities in text comprehension and generation, enabling the development of autonomous agents that operate through natural language, commonly referred to as language agents [1; 2; 3]. Language agents are increasingly expected to interact with environments through language-driven actions to accomplish diverse tasks, such as web navigation [4; 5], text-based games [6; 7; 8], and negotiation [9; 10]. These tasks often require longhorizon planning and reasoning to achieve high-level goals, posing significant challenges for current language agents [11; 12; 13; 14; 15]. According to the structure of the action space, language agent tasks can be broadly categorized into constrained action space tasks and open-ended language action tasks. The former requires agents to perform actions from a predefined, discrete, and verifiable action set, where language serves as a template or command interface to structured environments [16; 17]. In contrast, the action space of open-ended language action tasks comprises free-form natural language utterances without strict validity constraints [18; 19]. These tasks introduce unique challenges: 1) Agents must generate diverse, context-sensitive language actions that dynamically influence other agents or the environment.
2) The open-endedness of language actions gives rise to a vast, unstructured, and highly strategic action space, requiring agents to reason, adapt, and optimize beyond fixed patterns. Given these challenges, we pose the following research question: How can we enhance the performance of language agents in open-ended language action tasks?
Reinforcement learning (RL) is widely used to enhance language agents in complex tasks by enabling them to learn through interaction and feedback [20; 21]. However, in open-ended language action settings, RL faces serious challenges due to extremely sparse rewards caused by exponentially large action space, where actions are represented as token sequences. Given a vocabulary of size V and an average sequence length L, the action space scales as V L , resulting in a combinatorial and exponential explosion. Existing methods directly assign environmental rewards by averaging or decaying. Yet these are inadequate for open-ended tasks, where sampling-based methods such as PPO [22] and REINFORCE [23] must search a vast, unstructured space under sparse and delayed rewards. This leads to high variance in reward estimation and inefficient policy optimization.
To address these challenges, we propose semantic projection, which projects actions from the high-dimensional token space into a low-dimensional intention space, enabling reward aggregation across semantically equivalent actions. LLM agents' actions often reflect underlying intentions, which are far fewer than token combinations. For example, the utterances "I will concede first in order to encourage my opponent to compromise" and "By taking the initiative to compromise, I aim to prompt my counterpart to do the same." convey the same intention of prompting compromise through concession. By grouping such actions under shared intentions, we reduce the action space from V L to intention space C, where |C| ≪ |V L |. This transformation reduces variance by densifying sparse rewards, and facilitates more efficient policy optimization.
Building on semantic projection, we propose ARIA, a method that Aggregates Rewards in Intention space for efficient training of language Agents. ARIA maps natural language actions into a taskspecific intention space via semantic projection, enabling reward aggregation across semantically similar actions to stabilize and improve policy learning. To automatically construct the intention space C, ARIA applies hierarchical clustering [24] over sentence embeddings and adaptively adjusts the clustering granularity. It then aggregates rewards for actions sharing similar intentions and uses REINFORCE [23] to optimize the policy over this compressed space. We evaluate ARIA on four language action tasks, including two single-agent games (Guess My City, 20 Questions) and two adversarial games (Negotiation, Bargaining). Experimental results show that: 1) ARIA significantly reduces reward variance, enabling stable training and improved policy gradient efficiency; 2) It consistently outperforms offline and online RL baselines, achieving an average improvement of 9.95% across all tasks.
In summary, our key contributions are as follows: 1) We propose the operation of semantic projection, which projects actions from the high-dimensional token sequence space into a compact intention space, effectively mitigating reward sparsity in free-form language action tasks; 2) Built upon semantic projection, we design ARIA, a principled approach for training language agents with intention-driven reward aggregation; 3) We conduct extensive experiments on both single-agent and adversarial tasks, showing that ARIA reduces reward variance, accelerates convergence, and outperforms existing offline and online RL baselines.
2 Related Work Natural Language Agent Benchmark Recent studies have introduced evaluation tasks for language agents requiring long-horizon planning and strategic reasoning in multi-turn, goal-driven settings, including social conversations [25], strategy games (e.g., Werewolf [26], Avalon [27]), economics-based scenarios (e.g., bargaining [18; 19], negotiation [19]), and text-based games (e.g., Taboo [28], Guess My City [8], 20 Questions [8], Ask-Guess [29]). In this work, we focus on text-based games (Guess My City, 20 Questions) and adversarial tasks (Bargaining, Negotiation). These settings require dynamic strategy adaptation, balancing short-and long-term goals, and complex reasoning, offering challenging benchmarks for evaluating LLM agents' planning and decision-making.
Semantic Clustering Semantic clustering partitions samples into categories based on semantic similarity, typically by first extracting representations (e.g., embeddings), then applying clustering algorithms such as k-means [30], hierarchical clustering [24], or DBSCAN [31]. In ARIA, actions
Offline REINFORCE R > Aggregated Rewards Objective: ∇ A Log π (a | h) • A Advantage Policy Model a1 o1 a2 o2 a3 Intention Space C k
this section cite: ['b21', 'b22', 'b23', 'b22', 'b24', 'b25', 'b26', 'b18', 'b27', 'b7', 'b7', 'b28', 'b29', 'b23', 'b30']

Section: Reward Aggregation

this section cite: []

Section: Aggregate Rewards

this section cite: []

Section: Exploita)on
Resistance Bargaining Trajectory τ Task Descrip,on: You are Alice. You are selling one product that is worth no less then $80 to you.Bob is a poten)al buyer […] Hello Bob, I'm excited to offer you this amazing product! I will sell it to you for $110 Alice You are being so greedy ! I was hoping to get the product for a lower price. Would you consider selling it for $85 Bob I'm not comfortable selling the product for $85. I'm willing to make a counteroffer. How about we meet in the middle at $92.50? Alice Avg. are embedded and clustered into intentions using hierarchical clustering, which offers flexible post hoc granularity control and captures hierarchical semantic relations for coarse-to-fine strategy modeling.
Training Language Agent with Reinforcement Learning Language agents often face ambiguous goals and sparse rewards, requiring adaptive long-term planning [1; 2; 3], which challenges decisionmaking. Reinforcement learning (RL) provides a principled framework to address these challenges, with existing methods falling into two categories: offline methods [12; 28; 13; 14; 26; 32], which precollect trajectories and apply post-processing (e.g., DPO [33], KTO [34]); and online methods [22; 20; 35; 36], which alternate between sampling and policy updates. However, the high-dimensional action space in free-form language tasks exacerbates reward sparsity and variance, hindering RL training. To mitigate this, we adopt an offline RL setup with reward aggregation and REINFORCE [23], improving learning stability and efficiency.
this section cite: ['b32', 'b33', 'b22']

Section: Method
We present an overview of ARIA in Figure 1. First, we construct the intention space using semantic clustering ( §3.2), where the optimal granularity is determined by Reward-Oriented Granularity Selection ( §3.4). Next, high-dimensional actions and observations are projected into the intention space through semantic projection, enabling reward aggregation ( §3.3). Finally, the aggregated rewards are used to optimize the policy efficiently via offline REINFORCE ( §3.5).
this section cite: []

Section: Task Formulation
In this paper, we select two types of open-ended language action tasks, single-agent and two-agent adversarial games, as the testbed. We formulate the tasks as a partially observable Markov decision process (POMDP) M = (S, A, O, T , R, γ), where S is the global state, A is the action space of natural language actions, O is the observation, T is the transition function, R is the reward function, and γ is the discount factor. In the single-agent setting, an agent P interacts with the environment by performing actions over time. At each step t, the agent receives an observation o t under state s t and maintains a history h t = {o 1 , a 1 , . . . , o t-1 , a t-1 , o t }. The agent then selects an action a t ∼ π θ (• | h t ) conditioned on this history. The state s t subsequently transitions to s t+1 according to the transition function T : S × A → S. When s t reaches the terminal condition, the environment returns a reward R. The objective of the agent is to maximize the expected cumulative reward at the end of the episode based on the policy π θ . In the adversarial setting, two players P ∈ {P 1 , P 2 } take turns performing actions. In state s t , player P i selects an action a i ∼ π i (• | h t ), where h t = {o 1 , a 1 , . . . , o t-1 , a t-1 , o t } is the history of observations and actions, and o t is derived from the state s t and the opponent's action a t . The state s t then transitions to s t+1 according to the transition function T : S × A → S. When the terminal condition is met in s t , the environment returns a reward R to each player. Each player P i aims to maximize the expected reward by the end of the episode based on their policy π i .
this section cite: []

Section: Intention Space Construction
We construct a latent intention space using clustering. Given the action space A and observation space O, each element x ∈ A ∪ O is embedded into a semantic vector using a pre-trained encoder ϕ : A ∪ O → R d . We apply hierarchical agglomerative clustering [37] to partition the embedding space into k clusters, forming the intention space C k (see Appendix D for details). The number of clusters k is selected via reward-oriented granularity selection ( §3.4).
this section cite: ['b36']

Section: Reward Aggregation
Based on the intention space C k , we define a clustering function c k : A ∪ O → [k] that maps each element to a cluster index. At each step t, the action and observation are mapped to cluster labels ãt = c k (a t ) and õt = c k (o t ), respectively. Given the history h t = {a 1 , o 1 , . . . , a t-1 , o t-1 }, the corresponding label sequence is ht = {c k (a 1 ), c k (o 1 ), . . . , c k (a t-1 ), c k (o t-1 )}.
We aggregate rewards across history-action pairs that share the same semantic intention. The trajectory reward R is assigned to intermediate steps using temporal discounting: R(h t , a t ) = γ T -t R, where γ is the discount factor. For each intention pair ( h, ã), we compute the aggregated return by averaging over all history-action pairs that map to it:
R(k) ( h, ã) = 1 |S h,ã | (ht,at)∈Sh ,ã R(h t , a t ),
where S h,ã = {(h t , a t ) : c k (h t ) = h, c k (a t ) = ã} denotes the set of history-action pairs associated with intention ( h, ã). The aggregated return R(k) ( ht , ãt ) is used as the advantage estimate Ã(h t , a t ) for policy optimization.
this section cite: []

Section: Reward-Oriented Granularity Selection
Best Cluster: 2 Score: 1.00
Figure 2: Clustering quality measured by SC, CHI, the reciprocal DBI and the average of three metrics. After normalization and averaging, k = 2 achieves the highest overall score.
Semantic clustering helps compress the free-form, unstructured space of natural language actions and observations. However, selecting the appropriate granularity k remains challenging. For example, in the context of negotiation, we compute standard clustering metrics-Silhouette Score [38], Calinski-Harabasz Index [39], and Davies-Bouldin Index [40]-across different configurations. In Figure 2, these metrics tend to favor overly coarse groupings due to the high similarity among actions, overlooking fine-grained distinctions that are critical for our task (see details of metric calculations in Appendix E).
To address this, we propose a reward-oriented granularity selection mechanism that assesses whether further splitting clusters yields meaningful reward change. Unlike traditional metrics based on geometric structure (i.e., distance in embedding space), our method aligns with the RL objective by directly evaluating the impact on reward aggregation.
SplitScore Let k ∈ [2, K] denote all possible granularity levels. We use SplitScore to select the optimal granularity k * , defined as SplitScore(k) = δ k |D| , where δ k =
(ht,at)∈D R(k+1) (h t , a t ) -R(k) (h t , a t ) represents the reward change for all (h t , a t ) pairs when the number of clusters changes from k to k + 1, and D is the collection of all (h t , a t ) pairs.
this section cite: ['b37', 'b38', 'b39']

Section: Automatic Stopping Criterion
To select the optimal granularity k * , we define an early stopping mechanism based on SplitScore. Given a threshold ϵ > 0 and a window size τfoot_0 , we stop splitting when SplitScore(j) < ϵ for all j ∈ [k, k + τ ] as k increases. The selected k is then taken as k * . We prove in Appendix C that SplitScore is bounded above a monotonically decreasing function. When SplitScore remains below the threshold, further splitting has minimal impact on δ k , indicating that the rewards R(k) (h t , a t ) are nearly unchanged and do not significantly affect the training process. Thus, we select the smallest k that meets the stopping condition to realize better space compression.
this section cite: []

Section: Offline REINFORCE with Aggregated Reward
We use the offline REINFORCE algorithm [23] to optimize the policy. Formally, let π θ (a | s) denote the policy parameterized by θ and assign the aggregated reward R(k) ( ht , ãt ) to Ã(h t , a t ). ARIA optimizes the model by maximizing the following objective:
J(θ) = E τ ∼π θ T t=0 log π θ (a t | h t ) • Ã(h t , a t ) .
this section cite: ['b22']

Section: Theoretical Analysis
In this section, we theoretically show that intention clustering-based aggregation of the rewards in ARIA can reduce the variance of the gradient descent while maintaining a small bound of bias, thus improving training stability and efficiency.
this section cite: []

Section: Background
Let A(h t , a t ) be the original advantage of (h t , a t ) and c k (x) be the cluster label assigned to instance x ∈ A ∪ O under the granularity k, we define ( ht , ãt ) = {(c k (a 1 ), c k (o 1 ), . . . , c k (a t-1 ), c k (o t-1 ), c k (a t )} and calculate the cluster-averaged reward for ( ht , ãt ) as R( ht , ãt ) = 1 |D| ( ht,ãt)∈D R( ht , ãt ), where R( ht , ãt ) is the original reward of ( ht , ãt ). Then we assign R( ht , ãt ) to the advantage of (h t , a t ) as Ã(h t , a t ).
this section cite: []

Section: Main Theorem
We first establish that cluster-based aggregation reduces both the total variance of the policy gradient algorithm and the variance of the policy gradient. We give the following two lemmas.  We leave the proof in Appendix G. Building on Lemma 4.2, we show that the variance reduction by aggregation improves the convergence properties of offline REINFORCE. Proof. Let g = E[g i ] be the expected gradient for the i-th trajectory, where
gi = t ∇ θ log π θ (a i t | h i t ) Ãi t is the gradient estimator. The empirical gradient is ĝ = 1 N N i=1 gi. Let σ 2 = Var t ∇ θ log π θ (at | ht)
• Ãt . By expectation linearity and trajectory independence, the variance of the empirical gradient is E ∥ĝ -g∥ 2 2 = σ 2 N . By Jensen's inequality [41], we get E [∥ĝ -g∥2] ≤ σ √ N .
Intuitively, because clustering reduces σ, supposing we want |ĝ -g| < ϵ, convergence to within ϵ requires fewer samples, or equivalently, enables the use of larger step sizes for the same error tolerance. We then analyze the bias introduced by reward aggregation. To formalize this, we first give the notion of ε-bisimulation. Definition 1 (ε-Bisimulation). Actions a, a ′ are said to be ε-bisimilar if, for all states s, |r(h, a)r(h, a Proof. ε-bisimulation ensures that value differences within a cluster satisfy |Q π (h, a)-Q π (h, a ′ )| ≤ 2ε 1-γ , implying that cluster means differ by at most O(ε). Since ∇ log π is bounded, the inner product bias is O(ε). In summary, by using conditional expectations and variance decomposition, we prove that replacing original advantages A with cluster-averaged advantages Ã removes the intra-cluster variance E[Var(A | C)], lowering the total variance of the policy gradient estimate. Provided that the expectation remains approximately unchanged, this variance reduction leads to more stable training and faster convergence. It allows larger optimization steps without divergence and increases the utility of each sample, explaining why cluster-smoothed advantages yield smoother learning curves.
this section cite: ['b40']

Section: Experiments

this section cite: []

Section: Experimental Setup
Baselines We select both online and offline methods as baselines. For offline methods, we include: 1) Behavior Cloning (BC) that trains the policy using successful trajectories. 2) Trajectory-wise DPO [12], which trains langugae models using successful and failed trajectories. 3) Step-wise DPO [13], which employs success/failure labels at the action level based on simulation outcomes. 4) SPAG [28], which designs a discounted reward and uses offline PPO [22] for optimization of policy gradients. For online methods, we select: 1) Archer [20], which utilizes a hierarchical reinforcement learning framework. 2) StarPO [42], which applies GRPO [35] for policy optimization. Implementation details of baselines are in Appendix I.1.
Tasks We evaluate ARIA in both single-agent and adversarial environments (see Appendix H for details). For the single-agent setting, we consider two tasks: 1) Twenty Questions [8], a dialogue task where the agent plays the role of a guesser, aiming to identify a hidden word selected from a list of 157 candidates by asking up to twenty yes-no questions. The Oracle responds with "Yes" "No" or "Invalid Question". The agent receives a final reward R = 1 upon correctly guessing the target word, ending the episode; otherwise, the reward remains 0. 2) Guess My City [8], a similar multi-turn task where the agent tries to identify a hidden city from a list of 100 candidates within twenty questions. The agent can ask any type of question and receives free-form responses, not limited to yes/no answers. For the adversarial setting, we consider two competitive tasks: 1) Bargaining [43], a two-player game where Alice and Bob take turns proposing how to divide a fixed amount of money over a finite time horizon. As the game progresses, each player's payoff is discounted by a player-specific discount factor. If the game ends without an agreement, both players receive zero payoff. Otherwise, the discounted payoffs for Alice and Bob are given by p A and p B . 2) Negotiation [43], a two-player task where a seller (Alice) and a buyer (Bob) negotiate the price of a product with a true value. Alice and Bob each have subjective valuations. Over a fixed time horizon, the players alternate offers: at odd stages, Alice proposes a price and Bob decides whether to accept; at even stages, Bob proposes and Alice decides. If a price is accepted, the utilities for Alice and Bob are given by u A , u B . If no agreement is reached, both receive zero utility.
Evaluation For the single-agent environments, following ArCHer [20], we evaluate ARIA on a subset of N tasks from Twenty Questions and Guess My City. We report the average final reward, defined as 1 N N i=1 I[R i = 1], where R i denotes the final reward for the i-th trajectory. We set N = 200 for each environment. For the adversarial environments, following GLEE [43], we evaluate ARIA across 48 game configurations. In each configuration, the agent plays as either Alice or Bob against fixed opponents, with each setting repeated N = 25 times. In Bargaining, the goal is to achieve a higher payoff than the opponent. In Negotiation, the objective is to sell at a higher price Table 1: Main results on adversarial games. The best results are bolded, and the second best ones are underlined. The metric is the average win rate.
this section cite: ['b11', 'b12', 'b27', 'b21', 'b19', 'b41', 'b34', 'b7', 'b7', 'b42', 'b42', 'b19', 'b42']

Section: Methods

this section cite: []

Section: Bargaining Negotiation

this section cite: []

Section: GPT-4o Deepseek-V3 Claude-3.5 AVG. GPT-4o Deepseek-V3 Claude-3.5 AVG.
Vanilla Model 30.14 24.05 33.72 29.30 37.92 36.94 40.08 38.31 Offline Baselines BC 46.92 40.64 55.64 47.73 31.92 40.06 32.34 34.77 Traj-wise DPO 46.77 45.58 47.57 46.64 35.57 35.68 35.38 35.54 Step-wise DPO 48.91 55.48 46.00 50.13 36.33 41.56 49.17 42.35 SPAG 30.68 37.26 22.43 30.12 25.83 33.86 33.65 31.11 Online Baselines ArCHer 43.78 47.35 53.94 48.36 35.00 37.84 34.64 35.83 StarPO 33.24 28.77 42.63 34.88 38.55 36.00 43.87 39.47 Ours ARIA ( Iter 1) 51.54 55.26 52.66 53.15 45.65 42.69 49.02 45.79 ARIA ( Iter 2) 53.60 67.33 55.62 58.85 47.46 45.08 48.93 47.16 ARIA ( Iter 3) 58.66 55.83 59.55 58.01 46.48 50.50 49.42 48.80
(as the seller) or buy at a lower price (as the buyer). We let ARIA play both roles (Alice and Bob) against various opponents and compute the average win rate for each role, counting each successful completion of the task objective as a win. Specifically, the average win rate for Alice in Bargaining is defined as W A = 1
this section cite: []

Section: Models
We use Llama-3-8B-Instruct [44] as the policy model. For each language action, we obtain its semantic embedding using text-embedding-3-small [45]. Additional ablation results on alternative embedding models are provided in Appendix K. In single-agent environments, Oracle is simulated with GPT-4. In adversarial settings, we employ opponent models from different families, including GPT-4o (gpt-4o-2024-08-06) [46], Claude 3 (claude-3-5-sonnet-20240620) [47], and DeepSeek-Chat (DeepSeek-V3) [48].
Implementation Details For each scenario, we gather 1,000 games and update the policy using the trajectories. Specifically, in single-agent scenarios, the actor interacts directly with the Oracle (i.e., the environment). For adversarial scenarios, we employ self-play to collect competitive interaction data from both players. To evaluate whether ARIA can consistently improve the policy, we perform three iterations. In each iteration, we collect another 1,000 games using the updated policy and conduct a new round of training. Additional implementation details are provided in Appendix I.
this section cite: ['b43', 'b44', 'b45', 'b46', 'b47']

Section: Results
Table 2: Main results on single-agent games. The best results are bolded, and the second-best ones are underlined. The metric is the average reward.
Methods Twenty. Guess. AVG. Vanilla Model 27.50 13.50 20.50 Offline Baselines BC 27.50 5.50 16.50 Traj-wise DPO 27.00 17.50 22.25 Step-wise DPO 27.50 11.50 19.50 SPAG 26.50 13.00 19.75 Online Baselines ArCHer 26.00 10.00 16.25 StarPO 27.50 10.50 16.00 Ours ARIA (Iter 1) 28.00 29.00 28.50 ARIA (Iter 2) 29.50 32.00 30.75 ARIA (Iter 3) 34.50 36.00 35.25
ARIA significantly improves policy performance. As shown in Table 1, in the adversarial tasks, ARIA achieves the highest average win rate in both Bargaining and Negotiation, surpassing offline and online baselines by 9.67% and 9.83%, respectively. Similarly, in the single-agent tasks (Table 2), ARIA outperforms all baselines by an average of 9.82%. Existing offline and online RL methods both rely on action sampling and reward assignment, where agents interact with the environment, collect action samples, and assign rewards to those actions. This approach works reasonably well in small action spaces, where repeated sampling provides stable and accurate reward estimates. However, in openended language action tasks, where agents act through natural language, the action space  grows exponentially to V L , given a vocabulary of size V and an average sequence length L. In such vast spaces, each sample typically receives only a binary reward signal, and the sample size N is much smaller than the action space, leading to highly sparse and noisy reward signals and making accurate credit assignment challenging. ARIA addresses this by introducing reward aggregation in the intention space, which reduces reward variance and significantly improves learning performance.
ARIA continuously improves policy through iteration. After confirming that ARIA significantly outperforms the baselines, we further investigate its performance under iterative updates. As shown in Table 1 and Table 2, ARIA achieves additional gains of 3.27% and 1.85% after two and three iterations, respectively. This suggests that reward aggregation effectively reduces variance while preserving essential discriminative signals for policy learning, reflecting a favorable bias-variance trade-off. It further enhances sample efficiency and mitigates the risk of premature convergence caused by excessive smoothing, demonstrating that reward aggregation can deliver stable and cumulative performance improvements.
this section cite: []

Section: Extending to Online ARIA Settings
We first perform reward aggregation using pre-collected trajectories. The aggregated rewards are then used to initialize a point-wise reward model (RM), implemented as Llama-3.1-8B-Instruct [44], consistent with the policy model. Subsequently, the policy interacts with the environment to dynamically generate new samples, which are scored by the RM to update the policy. Additionally, the RM is periodically updated with the latest collected data, allowing it to evolve alongside the policy. We conduct the online ARIA on two single-agent games to conveniently observe reward at each iteration. Detailed parameter settings are provided in Appendix I.2.
this section cite: ['b43']

Section: Results
As shown in Figure 3, ARIA achieves faster reward improvement and consistently higher returns across iterations compared to existing online methods (ArCHer and StarPO). This improvement stems from two key advantages: 1) Reward aggregation provides an initial dense and low-variance reward signal, accelerating early-stage policy learning.
2) The dynamic RM update ensures alignment between the reward function and the evolving policy, preventing drift and reward misalignment common in static settings. Together, these factors enhance both sample efficiency and reward shaping accuracy, leading to faster and more stable policy improvement.
this section cite: []

Section: Analysis 6.1 Reward Aggregation Significantly Reduces Reward Variance
We show variance change before and after reward aggregation in Figure 4. As shown in Figure 4a, reward aggregation markedly reduces the fluctuation range of action rewards. The original binary reward distributions are highly polarized, with values mostly concentrated near 0 or 1. In a large action space, most actions are sampled only once, and the corresponding binary reward is directly assigned to each action, resulting in high reward variance. By contrast, after reward aggregation, actions within the same cluster share a common reward, which significantly smooths the distribution and reduces variance. Figure 4b further demonstrates that reward variance decreases across all four tasks, highlighting the effectiveness and necessity of reward aggregation in stabilizing policy learning.
this section cite: []

Section: Reward Aggregation Improves Policy Optimization
To evaluate whether reward aggregation improves training efficiency, we first compare the policy loss curves under different reward shaping strategies in Figure 5b. The results show that ARIA, which applies semantic-level reward aggregation, accelerates loss reduction compared to the vanilla REINFORCE baseline. This indicates that shaping the reward through aggregation provides a stronger learning signal, enabling faster policy updates and improved sample efficiency in offline training. We further observe that, despite converging to similar loss levels, the methods exhibit substantial differences in downstream performance. As shown in Figure 5a, ARIA outperforms other variants by 17.91% and 13.80% on the bargaining and negotiation tasks, respectively. We attribute these gains to the complementary effects of reward decay and reward aggregation: Reward decay introduces temporal structure that helps assign credit to early-stage actions, but plays a limited role in reducing signal noise. In contrast, reward aggregation substantially lowers reward variance by assigning shared signals to semantically similar actions, thereby improving the quality of gradient estimation. This variance reduction enables more stable and efficient optimization and plays a central role in enhancing policy performance in open-ended language action settings.
this section cite: []

Section: Barg.
Nego.  In Section 5.2, we show that ARIA achieves significant improvements on Llama3-8B-Instruct. To further assess the transferability of ARIA, we apply it to the Qwen models (Qwen2.5-7B-Instruct [49] and Qwen2.5-1.5B-Instruct [49]) and conduct comparative experiments on two adversarial games 4 . As shown in Table 3, we observe that altering the base model consistently yields improvements. This suggests that our reward aggregation approach is model-agnostic and independent of specific architectural features or pretraining data of the underlying language models. We attribute this generalizability to the shared structural properties in the semantic spaces learned by large-scale language models. By performing aggregation in the intention space, ARIA leverages these commonalities to reduce reward variance while preserving task-specific discriminative signals.
this section cite: ['b48', 'b48']

Section: Generalization of ARIA to Other Models

this section cite: []

Section: Conclusion
In this paper, we address the core challenges of reinforcement learning in open-ended language action tasks, where agents must operate in exponentially large action spaces and learn from sparse, delayed rewards. To tackle the resulting high variance in policy optimization, we introduce semantic projection, a novel intention-aware framework that maps natural language actions from the high-dimensional token space into a low-dimensional intention space. This projection enables reward aggregation across semantically similar actions, effectively densifying sparse rewards and reducing gradient variance. Built on this idea, we propose ARIA, which automatically discovers task-specific intention structures via hierarchical clustering and integrates the aggregated rewards into REINFORCE for efficient policy learning. We further provide a theoretical analysis showing that replacing original advantages with cluster-averaged advantages reduces intra-cluster variance, thereby lowering the overall variance of the policy gradient and improving learning stability. Extensive experiments across four diverse tasks-including both single-agent and adversarial two-agent games-demonstrate that ARIA improves training stability, accelerates convergence, and consistently outperforms strong offline and online RL baselines. Our findings highlight the importance of structure-aware reward shaping in scaling reinforcement learning for language agents in open-ended environments.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: References
Ref_id:b0 Title: A survey on large language model based autonomous agents Year: (2024)
Ref_id:b1 Title: Large language model agent: A survey on methodology, applications and challenges Year: (2025)
Ref_id:b2 Title: Cognitive architectures for language agents Year: (2023)
Ref_id:b3 Title: A real-world webagent with planning, long context understanding, and program synthesis Year: (2023)
Ref_id:b4 Title: A realistic web environment for building autonomous agents Year: (2023)
Ref_id:b5 Title: Is your agent smarter than a 5th grader? arXiv preprint Year: (2022)
Ref_id:b6 Title: Shaping efficient multitasking language agents in a time-aware simulation Year: (2024)
Ref_id:b7 Title: Lmrl gym: Benchmarks for multi-turn reinforcement learning with language models Year: (2023)
Ref_id:b8 Title: Evaluating language model agency through negotiations Year: (2024)
Ref_id:b9 Title: How well can llms negotiate? negotiationarena platform and analysis Year: (2024)
Ref_id:b10 Title: Clin: A continually learning language agent for rapid task adaptation and generalization Year: (2023)
Ref_id:b11 Title: Trial and error: Exploration-based trajectory optimization for llm agents Year: (2024)
Ref_id:b12 Title: Watch every step! llm agent learning via iterative step-level process refinement Year: (2024)
Ref_id:b13 Title: Agent q: Advanced reasoning and learning for autonomous ai agents Year: (2024)
Ref_id:b14 Title: Selfgoal: Your language agents already know how to achieve high-level goals Year: (2025)
Ref_id:b15 Title: Alfworld: Aligning text and embodied environments for interactive learning Year: (2020)
Ref_id:b16 Title: Towards scalable real-world web interaction with grounded language agents Year: (2023)
Ref_id:b17 Title: Deal or no deal? end-to-end learning for negotiation dialogues Year: (2017)
Ref_id:b18 Title: Glee: A unified framework and benchmark for language-based economic environments Year: (2024)
Ref_id:b19 Title: Training language model agents via hierarchical multi-turn rl Year: (2024)
Ref_id:b20 Title: Sweet-rl: Training multi-turn llm agents on collaborative reasoning tasks Year: (2025)
Ref_id:b21 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b22 Title: Simple statistical gradient-following algorithms for connectionist reinforcement learning Year: (1992)
Ref_id:b23 Title: Hierarchical grouping to optimize an objective function Year: (1963)
Ref_id:b24 Title: Interactive evaluation for social intelligence in language agents Year: (2023)
Ref_id:b25 Title: Multiagent kto: Reinforcing strategic interactions of large language model in language game Year: (2025)
Ref_id:b26 Title: Evaluating llms playing the game of avalon Year: (2023)
Ref_id:b27 Title: Self-playing adversarial language game enhances llm reasoning Year: (2024)
Ref_id:b28 Title: Evaluating llms on conversational games Year: (2023)
Ref_id:b29 Title: Least squares quantization in pcm Year: (1982)
Ref_id:b30 Title: A density-based algorithm for discovering clusters in large spatial databases with noise Year: (1996)
Ref_id:b31 Title: Prism: Self-pruning intrinsic selection method for training-free multimodal data selection Year: (2025)
Ref_id:b32 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b33 Title: Model alignment as prospect theoretic optimization Year: (2024)
Ref_id:b34 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b35 Title: Cot-kinetics: A theoretical modeling assessing lrm reasoning process Year: (2025)
Ref_id:b36 Title: Methods of hierarchical clustering Year: (2011)
Ref_id:b37 Title: Silhouettes: a graphical aid to the interpretation and validation of cluster analysis Year: (1987)
Ref_id:b38 Title: A dendrite method for cluster analysis Year: (1974)
Ref_id:b39 Title: A cluster separation measure Year: (2009)
Ref_id:b40 Title: Sur les fonctions convexes et les inégalités entre les valeurs moyennes Year: (1906)
Ref_id:b41 Title:  Year: (2025)
Ref_id:b42 Title: Glee: A unified framework and benchmark for language-based economic environments Year: (2024)
Ref_id:b43 Title: Llama 3 model card Year: (2024)
Ref_id:b44 Title:  Year: (2022)
Ref_id:b45 Title:  Year: (2023)
Ref_id:b46 Title: Introducing claude 2.1 Year: (2023-11)
Ref_id:b47 Title:  Year: (2025)
Ref_id:b48 Title: Qwen2.5: A party of foundation models Year: (2024-09)
Ref_id:b49 Title: Qwen3 embedding: Advancing text embedding and reranking through foundation models Year: (2025)
Ref_id:b50 Title: Text embeddings by weakly-supervised contrastive pre-training Year: (2022)
