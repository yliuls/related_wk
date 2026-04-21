Title: Monte-Carlo Tree Search with Uncertainty Propagation via Optimal Transport
Abstract: This paper introduces a novel backup strategy for Monte-Carlo Tree Search (MCTS) tailored for highly stochastic and partially observable Markov decision processes. We adopt a probabilistic approach, modeling both value and action-value nodes as Gaussian distributions, to introduce a novel backup operator that computes value nodes as the Wasserstein barycenter of their action-value children nodes; thus, propagating the uncertainty of the estimate across the tree to the root node. We study our novel backup operator when using a novel combination of L 1 -Wasserstein barycenter with α-divergence, by drawing a crucial connection to the generalized mean backup operator. We complement our probabilistic backup operator with two sampling strategies, based on optimistic selection and Thompson sampling, obtaining our Wasserstein MCTS algorithm. We provide theoretical guarantees of asymptotic convergence of O(n -1/2 ), with n as the number of visited trajectories, to the optimal policy and an empirical evaluation on several stochastic and partially observable environments, where our approach outperforms wellknown related baselines.

Section: Introduction
Monte-Carlo Tree Search (MCTS) has become a crucial algorithmic paradigm for tackling challenging planning and 1 Hanoi University of Science and Technology, Hanoi, Vietnam 2 Department of Computer Science, Technical University of Darmstadt, Germany 3 ETHZ -ETH Zurich, Switzerland 4 Department of Electrical Engineering and Automation, Aalto University, Finland 5 Center for Artificial Intelligence and Data Science, University of Würzburg, Germany 6 Hessian Center for Artificial Intelligence (Hessian.ai), Germany 7 Univ. Lille, Inria, CNRS, Centrale Lille, UMR 9189-CRIStAL, F-59000 Lille, France. Correspondence to: Tuan Dam <tuandq@soict.hust.edu.vn>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
Reinforcement Learning (RL) problems, particularly after its widespread success in deterministic games like Go and Chess (Silver et al., 2016a;2017b). However, moving beyond these deterministic settings toward highly stochastic or partially observable Markov Decision Processes (MDPs/POMDPs) reveals major difficulties. In these cases, two key obstacles arise: Uncertainty in Value Estimates: In problems with substantial randomness or limited observability, naive value backups may lead to erroneous or unstable estimates, which propagate through the tree and degrade overall performance. Exploration-Exploitation Balancing: Traditional UCT-based exploration bonuses (Kocsis et al., 2006) can falter under high variance transitions, often causing either over-or under-exploration. Recent works (Tesauro et al., 2012;Bai et al., 2013;2014) have suggested Bayesian or distributional methods for MCTS to better quantify uncertainty. Meanwhile, Metelli et al. (2019) leveraged L 2 -Wasserstein barycenters to propagate distributional information in temporal-difference learning. Yet, several open questions remain on how to unify distribution-based backups and flexible exploration strategies within a single MCTS framework that provably handles high stochasticity and partial observability. Our Approach. In this paper, we propose a new MCTS algorithm, Wasserstein MCTS, that models each node's value as a Gaussian distribution and propagates both mean and variance estimates throughout the tree. Crucially, we introduce a novel backup operator that computes value nodes as L 1 -Wasserstein barycenters of their action-value children, using an α-divergence as the distance measure. This yields:
• Distributional Value Backups: By tracking distributions (rather than point estimates), our method captures the inherent uncertainty of each node's value, especially valuable in stochastic or partially observable domains.
• Generalized Mean Operator: The α-divergence ties naturally to the power-mean backup (Dam et al., 2019;2024a), letting us interpolate between average-like and max-like updates to mitigate the overestimation often seen in RL (Hasselt, 2010).
We complement these distributional backups with two exploration mechanisms-an optimistic UCT bonus, and a Thompson sampling approach that selects actions by sam-pling from the node's Gaussian posterior. Our Key Contributions. 1. Uncertainty Propagation via L 1 -Wasserstein Barycenters. We provide a principled way to back up distributions in an MCTS, unifying L 1 -Wasserstein geometry and α-divergences to handle high variance and partial observability. 2. Connection to Generalized Mean Backup. Our backup operator yields a powermean update for node values, enabling a controllable continuum between overly optimistic (max-like) and riskaverse (average-like) estimates. 3. Polynomial Convergence Analysis. We prove that Wasserstein MCTS with Thompson sampling converges to the optimal policy at a rate O(n -1/2 ), matching known lower bounds. This is in contrast to prior distributional MCTS methods that lacked explicit convergence guarantees. 4. Extensive Empirical Validation. On a suite of highly stochastic MDPs (e.g. River-Swim, Taxi) and partially observable tasks (Pocman, Rocksample), our approach outperforms established baselines, including UCT, Power-UCT, and Bayesian MCTS variants.
Overall, Wasserstein MCTS offers a flexible and theoretically grounded framework for handling uncertainty within MCTS. By combining Gaussian node models, L 1 -Wasserstein barycenters, and α-divergences, it effectively balances exploration and exploitation in domains where noise or partial observability make traditional MCTS methods brittle.
this section cite: ['b17', 'b32', 'b3', 'b4', 'b19', 'b10', 'b15']

Section: Related Work
Metelli et al. (2019) use L 2 -Wasserstein barycenters to propagate uncertainty in temporal-difference learning. In MCTS, Bayesian methods handle uncertainty by treating values as Gaussian distributions (Tesauro et al., 2012) or Dirichlet-NormalGamma posteriors (Bai et al., 2013;2014). Unlike these, we propagate uncertainty throughout the tree via L 1 -Wasserstein barycenters and α-divergences, linking to generalized-mean backups (Dam et al., 2019) and maintaining both mean and variance estimates. This distributional perspective is effective in highly stochastic or partially observable tasks. In multi-armed bandits, optimism (Auer et al., 2002a) and Thompson sampling (Thompson, 1933) are standard; we combine these with our uncertainty propagation scheme to guide action selection in MCTS.
this section cite: ['b32', 'b3', 'b4', 'b10', 'b33']

Section: Background

this section cite: []

Section: Markov Decision Process
We consider an agent in an infinite-horizon discounted Markov decision process (MDP) M = ⟨S, A, R, P, γ⟩, where S is the state space, A is the finite action space, R : S × A × S → R is the reward function, P : S × A → S is the transition kernel, and γ ∈ [0, 1) is the discount factor. A policy π ∈ Π : S → A defines the action selection probabilities based on states. The action-value function Q π is given by
Q π (s, a) ≜ E ∞ k=0 γ k r i+k+1 | s i = s
, a i = a, π , representing the expected cumulative discounted reward for executing action a in state s and following policy π. The objective is to find the optimal policy that maximizes Q π , satisfying the Bellman equation (Bellman, 1954)
: Q * (s, a) ≜ S P(s ′ |s, a) [R(s, a, s ′ ) + γ max a ′ Q * (s ′ , a ′ )] ds ′ , ∀s ∈ S, a ∈ A.
From the optimal action-value function, we derive the optimal value function as V * (s) ≜ max a∈A Q * (s, a), ∀s ∈ S.
this section cite: ['b5']

Section: Monte-Carlo Tree Search
Monte-Carlo Tree Search (MCTS) combines Monte-Carlo sampling, tree search, and exploration strategies from multi-armed bandits (Auer et al., 2002b) to solve MDPs.
It builds a search tree where states are nodes and actions are edges. MCTS involves four key steps: Selection: Navigate from the root to a leaf node using a tree-policy.
Expansion: Expand the reached node based on the tree policy. Simulation: Perform a rollout (Monte-Carlo simulation) from the child node to estimate its value, or use a pretrained neural network (Silver et al., 2016a) for this estimation. Backup: Update the action-values Q(•) along the visited trajectory using the collected rewards.
this section cite: []

Section: Formalization
Problem Setup Monte Carlo Tree Search (MCTS) is an algorithm for exploring and evaluating trajectories in an MDP. Starting from an initial state s 0 , MCTS incrementally builds a planning tree by simulating trajectories. Each trajectory either reaches a leaf node or terminates when a predetermined maximum depth H is reached. At the end of each trajectory, a playout policy (which may be deterministic or stochastic) is executed from the final node reached, allowing the algorithm to evaluate the associated state. After running for t trajectories, the MCTS algorithm provides the following outputs:
• a t : estimate of the optimal action to take in state s 0 , • V t (s 0 ): estimate of the optimal value function at s 0 .
this section cite: []

Section: Evaluating MCTS Performance
The performance of the MCTS algorithm is assessed based on its convergence rate, r(t), which quantifies how quickly the algorithm approaches the optimal policy. Specifically, the following bounds hold:
E [V ⋆ (s 0 ) -Q ⋆ (s 0 , a t )] ⩽ r(t), E V ⋆ (s 0 ) -V t (s 0 ) ⩽ r(t),
where V ⋆ (s 0 ) and Q ⋆ (s 0 , a) are the true optimal value and action-value functions at state s 0 , respectively.
this section cite: []

Section: Recursive Value Estimation
To analyze the MCTS algorithm, we consider a planning horizon H and a playout policy π 0 with an associated value function V 0 . For each node s h at depth h (i.e., the state reached after h steps from s 0 ), we recursively define the value function V (s h ) as follows.
At the leaf nodes (h = H), the value function is simply the playout policy's value:
V (s H ) = V 0 (s H ).
For all other depths h ⩽ H -1, we compute the actionvalue function Q(s h , a) and value function V (s h ) as:
Q(s h , a) = r(s h , a) + γ s h+1 ∈As P(s h+1 | s h , a) V (s h+1 ), V (s h ) = max a Q(s h , a),
where r(s h , a) is the mean immediate reward obtained by taking action a in state s h , P(s h+1 | s h , a) is the probability of transitioning to state s h+1 from s h given action a and γ is the discount factor.
Bounding the Error The recursive structure of the value estimates gives rise to a bound on the error between the true optimal action-value function Q ⋆ (s 0 , a) and the MCTS estimate Q(s 0 , a). Specifically, we have:
Q ⋆ (s 0 , a) -Q(s 0 , a) ⩽ γ H ∥V ⋆ -V 0 ∥ ∞ ,
where the supremum norm ∥V ⋆ -V 0 ∥ ∞ can be restricted to states reachable within H steps from s 0 .
this section cite: []

Section: Goal of MCTS
The ultimate aim of the MCTS algorithm is to minimize the convergence rate r(t) by constructing accurate estimates of Q(s 0 , a) and V (s 0 ), which in turn approach the true optimal functions Q ⋆ (s 0 , a) and V ⋆ (s 0 ), and then identify the best action at the root node:
a ⋆ = arg max a Q ⋆ (s 0 , a).
this section cite: []

Section: Wasserstein Barycenter With α-Divergence
We introduce the key notions behind our distribution-based backups: the Wasserstein barycenter and the α-divergence. Unlike prior works that use L 2 -based Wasserstein distances (Metelli et al., 2019), we adopt an L 1 -Wasserstein distance combined with α-divergences. This combination yields more robust value backups in stochastic and partially observable settings.
this section cite: ['b19']

Section: Wasserstein Barycenter
Let (X , d) be a Polish (complete, separable metric) space. For q ≥ 1, define P q (X ) as the set of probability measures on X whose q-th moment is finite. For two distributions µ, ν ∈ P q (X ), the L q -Wasserstein distance is
W q (µ, ν) = inf ρ∈Γ(µ,ν) E (X,Y )∼ρ d(X, Y ) q 1/q ,
where Γ(µ, ν) is the set of joint couplings whose marginals match µ and ν. Given n distributions {ν i } n i=1 and weights
{w i } summing to 1, the L q -Wasserstein barycenter is ν = arg min ν n i=1 w i W q (ν, ν i ) q .
Our work focuses on q = 1.
this section cite: []

Section: α-divergence and the L 1 Wasserstein Barycenter
In many distribution-based backup schemes, the Wasserstein distance is a natural choice to quantify how "far apart" two distributions are. A commonly used approach (Metelli et al., 2019) is to employ the L 2 -Wasserstein metric. In contrast, we consider an L 1 -Wasserstein formulation coupled with an α-divergence for two main reasons:
• Robustness & Aggregation Control. An L 1 -based metric
can be more robust to outliers and large deviations than L 2 . Furthermore, combining it with the α-divergence allows a continuous interpolation between averaging and max-like backups (through the α parameter).
• Connection to Power-Mean Updates. Modeling nodes as Gaussians (or particle distributions) and relying on L 1 -Wasserstein with an α-divergence yields closed-form updates that coincide with the power-mean operator. This unifies average and maximum backups in a single formula and lets us propagate both means and variances (uncertainty) through the tree.
f -divergences and the α-divergence. An f -divergence (Csiszár, 1964) between two points X and Y over a Manifold M defined as
D fα (X∥Y ) = i ξ (i) Y f α ξ (i) X ξ (i) Y , f α (x) = x α -1-α(x-1) α(α-1) ,
where varying α controls how aggressively or conservatively we measure the "distance" between X and Y . Constructing the L 1 -Wasserstein Barycenter. In our approach, the L 1 -Wasserstein distance between ν and ν i is defined via
W 1 (ν, ν i ) = inf ρ ∈ Γ(ν,νi) E (X,Y )∼ρ D fα (X, Y ) . (1)
The L 1 -Wasserstein barycenter then solves
ν = arg inf ν n i=1 w i W 1 ν, ν i ,
i.e., we seek the single distribution ν that jointly minimizes its L 1 -Wasserstein distance (defined via the α-divergence) to all the ν i .
Why L 1 instead of L 2 . Using the L 1 distance in equation 1 naturally leads to a backup rule resembling the power mean operator Proposition 1. This power-mean update is more robust to high-variance samples and connects smoothly to both the average backup (when α → 0 or p = 1) and the max backup (as α → ∞ or p → ∞). Hence, L 1 -Wasserstein with α-divergences offers a principled way to blend distributions in highly stochastic environments while controlling the balance between underestimation and overestimation in the final backup.
Why Use an α-Divergence Instead of L 2 ? Although αdivergences are not strict metrics (they can be asymmetric and need not satisfy the triangle inequality), their use within an L 1 -Wasserstein framework provides distinct benefits for MCTS under stochastic or partially observable conditions:
• Greater Flexibility via Generalized Means. When combined with the L 1 -Wasserstein distance, an α-divergence naturally yields a power-mean style backup operator (Dam et al., 2019). By adjusting the parameter α, one smoothly interpolates between average-like and max-like backups, allowing precise control over how conservative or aggressive the updates should be. This stands in contrast to L 2 -based distances, which only yield fixed (e.g. purely quadratic) aggregation behavior.
• Robustness to Stochastic Variations. Because αdivergences can emphasize or de-emphasize portions of the distribution differently depending on α, they help mitigate overestimation or underestimation in highly stochastic settings. Empirical studies in distributional RL (Metelli et al., 2019) suggest that more adaptive divergence measures can significantly improve stability and performance when the underlying dynamics involve heavy noise.
• No Need for Symmetry in Backups. MCTS requires a cost functional to aggregate posterior distributions across children nodes, rather than a strict metric. Hence, the lack of symmetry or the triangle inequality does not undermine its validity here. An f -divergence-including αdivergences-is sufficient to drive consistent updates of belief distributions in the tree.
• Unified Framework for Various Divergences. The αdivergence family subsumes and generalizes many standard divergences (e.g. KL, reverse KL). This singleparameter approach enables users to easily switch or finetune the update behavior for different problem characteristics, rather than designing separate algorithms for each divergence.
• Direct Theoretical Connections. Under mild assumptions, L 1 -Wasserstein geometry paired with α-divergences admits closed-form or near-closed-form power-mean formulas (Dam et al., 2019). This not only streamlines theoretical analysis but also simplifies implementation by allowing straightforward computation of mean and variance updates at each node.
In practice, these properties make α-divergences wellsuited for uncertainty propagation within MCTS: despite not being a metric, their adaptability and connection to generalized means allow them to effectively handle complex, high-variance environments.
this section cite: ['b19', 'b9', 'b10', 'b19', 'b10']

Section: V-posterior
It is natural to define a value node as the V-posterior computed with L 1 -Wasserstein barycenters of the children nodes Q-posteriors, following a procedure inspired by Metelli et al. 2019(Metelli et al., 2019) and tailored to MCTS.
Definition 1 (V-posterior). Given a policy π and a state s ∈ S, we define the V-posterior V(s) induced by Qposteriors Q(s, a) with a ∈ A as the L 1 -Wassertein barycenter of the Q(s, a):
V(s) ∈ arg inf V E a∼π(.|s) W 1 (V, Q(s, a)) .
In this work, we model each node in the tree as a Gaussian distribution. We define p = 1 -α and derive the following. Proposition 1. Consider the V-posterior value function V(s) as a Gaussian: N (m(s), σ 2 (s)). Define each Q(s, a) as the action-value function child node of V(s). Each Q(s, a) is assumed as a Gaussian distributions Q(s, a) : N (m(s, a), σ(s, a) 2 ). If the value function V(s) is defined as the Wasserstein barycenter of the action-value function Q(s, a), given the policy π, we have
m(s) = (E a∼π [m(s, a) p ]) 1 p δ(s) = (E a∼π [δ(s, a) p ]) 1 p .
Proposition 1 shows the closed form solutions of the mean and standard deviation of the Gaussian value function V(s) considering it as the L 1 -Wasserstein barycenter Qposteriors. In detail, the mean of V(s) are the power mean of all mean values of all the Q(s, a) function, considering the finite set of actions. When p = 1, we derive the expected form solutions.
We point out that our approach is not restricted to the Gaussian distribution model. We get the following result by considering each tree node as a particle model.
Proposition 2. Consider the V-posterior value function V(s) as an equally weighted Particle model:
x i (s) : i ∈ [1, M ].
M is an integer and M ⩾ 1. Assume each action-value function Q(s, a) has M particles x i (s, a), i ∈ [1, M ]. If the value function V(s) is defined as the Wasserstein barycenter of the action-value function Q(s, a), given the policy π, each particle x i (s), i ∈ [1, M ] can be estimated as
x i (s) = (E a∼π [x i (s, a) p ]) 1/p ,
Proposition 2 shows that each particle of the V-posterior value function V(s) can be derived as the power mean of the respective particles of all the Q(s, a) function. If p = 1, we again get the closed-form solutions as the expectation of the respective particles of all the Q(s, a) functions. The results in Proposition 1, and Proposition 2 can be considered as the generalized result of Proposition A.3 in Metelli et al. (2019). In the next section, we present our Wasserstein Monte-Carlo tree search (W-MCTS ) algorithm, assuming each tree node is a Gaussian distribution.
this section cite: ['b19', 'b19', 'b19']

Section: Wasserstein Monte-Carlo Tree Search
We introduce our Wasserstein Monte-Carlo Tree Search (W-MCTS), where V-posteriors are modeled as Wasserstein barycenters of action-value distributions. With Gaussian distributions at each node, we define backup operators for mean and variance. Additionally, we propose two action selection strategies: optimistic selection and Thompson sampling.
this section cite: []

Section: Backup Operator
We model each V -node and Q-node as a Gaussian with mean and standard deviation:
V m (s), V std (s)
and Q m (s, a), Q std (s, a).
We denote V m (s, N (s)) as the empirical mean estimate of the V -node at state s after N (s) total visits, and Q m (s, a, n(s, a)) as the empirical mean estimate of the Qnode at (s, a) after n(s, a) visits. Likewise, V std (s, N (s)) and Q std (s, a, n(s, a)) are their corresponding empirical standard deviation estimates.
V -nodes. From Proposition 1, the mean and the standard deviation of a V -node is a power-mean aggregation of its Q-children:
V m (s, N (s)) ← a n(s,a) N (s) Q m (s, a, n(s, a)) p 1/p , V std (s, N (s)) ← a n(s,a) N (s) Q std (s, a, n(s, a)) p 1/p ,
where n(s, a) is the visit count of action a at state s, and N (s) = a n(s, a). For p = 1, this reduces to the standard average, whereas p > 1 induces a more "max-like" backup (Dam et al., 2019).
this section cite: ['b10']

Section: Q-nodes.
Under the Bellman-style backup for each Qnode,
Q m (s, a) = E[r(s, a)]+γ E[V m (s ′ )], Q std (s, a) = γ V std (s ′ ),
we replace expectations by empirical sums and visitation counts:
Q m (s, a, n(s, a)) ← r(s,a) + γ s ′ N (s ′ ) V m (s ′ ,N (s ′ )) n(s,a) , Q std (s, a, n(s, a)) ← γ s ′ N (s ′ ) V std (s ′ ) n(s,a)
.
Here, the sums range over transitions and children states s ′ , weighted by their visit counts N (s ′ ). As n(s, a) grows large, both the variance and mean estimators stabilize, eventually converging to deterministic values.
this section cite: []

Section: Action Selection
Monte Carlo Tree Search can adopt a variety of exploration strategies based on the original UCT framework (Kocsis et al., 2006). In practice, multiple refinements exist, such as the variants used in AlphaGo (Silver et al., 2016b), AlphaZero (Silver et al., 2017c;a), MuZero (Schrittwieser et al., 2020), Stochastic MuZero (Antonoglou et al., 2021), and Stochastic-Power-UCT (Dam et al., 2024b). Although different choices of the exploration constant or bonus lead to different performance characteristics, we retain the standard, state-of-the-art designs described below. In our theoretical analysis, however, we focus specifically on Thompson sampling, since the UCT-like optimistic selection can be viewed as a special case of the well-studied Power-UCT algorithm (Dam et al., 2019;2024b).
Optimistic Selection. A classic UCT-style selection picks actions using upper confidence bounds on Q-values, a = argmax ai m(s, a i ) + C log N (s) n(s,ai) , where m(s, a i ) is the empirical mean, n(s, a i ) is the visit count of action a i , and N (s) is the total visit count at state s. Replacing the 1 √ n(s,ai)
term by the empirical standard deviation σ(s, a i ) yields an optimistic variant of Wasserstein MCTS (W-MCTS-OS):
a = argmax ai m(s, a i ) + C σ(s, a i ) log N (s) .
The factor σ(s, a i ) ≈ 1/ n(s, a i ) follows from a CLTbased argument.
Thompson Sampling. In contrast, Thompson sampling stochastically samples an action from the Q-posterior: a = argmax ai θ i ∼ N m(s, a i ), σ 2 (s, a i ) .
this section cite: ['b17', 'b22', 'b0', 'b10']

Section: We refer to this Thompson variant as Wasserstein MCTS-TS (W-MCTS-TS).
In Section 7, we analyze its convergence properties under non-stationary multi-armed bandits and then leverage these results to establish convergence in the planning tree.
this section cite: []

Section: Theoretical Analysis

this section cite: []

Section: Analysis Setup
We define the setting for our theoretical analysis using a class of non-stationary Multi-Armed Bandit (MAB) problems at each state s in the MCTS tree. Consider K arms (actions), each with a mean reward µ k , for k ∈ [K]. At time step t, pulling arm k yields a random reward X k,t , bounded within [0, R]. The average reward for arm k after n trials is:
X k,n = 1 n n t=1 X k,t , with µ k,n = E[X k,n ]
Let ⋆ represent quantities related to the optimal arm, and denote T k (n) as the number of times arm k has been played by step n. We assume the following concentration condition holds: Assumption 1. We assume that the reward sequence, {X k,t : t ⩾ 1}, is a non-stationary process satisfying the assumption: for all 1 > ε > 0, ∃c > 0 that
Pr |X k,n -µ k | > ε ⩽ cn -1 ε -2 , k ∈ [K].
(2)
this section cite: []

Section: Main Results
We show the polynomial convergence of the expected estimated mean value function at the root node in Theorem 1.
this section cite: []

Section: CONVERGENCE OF W-MCTS
We start with an important result as shown below Proposition 3. Applying W-MCTS to an MCTS tree of depth (H), at any depth h of the tree, we have (i) At any depth h, ∃ constant C 0 > 0 that for any 0 < ε < 0, n ⩾ 1, we can derive
Pr V m (s h , a k , n) -V (s h , a k ) ⩾ ε ⩽ C 0 n -1 ε -2 . (ii) At any depth h, ∃ constant C 0 > 0 that for any 0 < ε < 0, n ⩾ 1, we can derive Pr Q m (s h , a k , n) -Q(s h , a k ) ⩾ ε ⩽ C 0 n -1 ε -2 .
Proof Sketch MCTS as a Hierarchical Bandit Structure. The Monte Carlo Tree Search (MCTS) algorithm can be viewed as a hierarchy of multi-armed bandits (MABs), where each node in the search tree represents an independent bandit problem. In this framework, the reward for each node, or current bandit, is influenced by the performance of the bandit algorithms applied to its child nodes. Since the W-MCTS policy adapts dynamically to balance exploitation and exploration, the rewards at each node are inherently non-stationary. The proof of Theorem 1 unfolds through three essential steps:
1. Analyzing Non-stationary Bandits The initial step focuses on the analysis of a non-stationary multi-armed bandit, which reflects the behavior of MABs at each MCTS node. We establish that if the rewards of these nonstationary bandits meet specific concentration properties, the regret induced by the W-MCTS algorithm will exhibit corresponding concentration guarantees. This outcome is formally stated in Theorem 2.
2. Induction Argument Next, we utilize an inductive argument to transfer the convergence and concentration properties from the lower tree levels to the root node. As the rewards from one level inform those of the next, the findings from Step 1 can be recursively applied. We begin at depth H -1 and move upward, demonstrating inductively that the bandit rewards at each level H of the MCTS satisfy the criteria required by Theorem 2. This process propagates the desired properties up to the root node, completing the induction.
this section cite: []

Section: Error Analysis from the Oracle
The final step examines the error introduced by the leaf node estimator, represented by the value function oracle V 0 . With this oracle, the depth-H MCTS can be interpreted as performing H steps of value iteration, starting from V 0 at the leaf nodes (as mentioned in (Dam et al., 2024b)). Importantly, the oracle's error decreases geometrically at a rate of γ due to the contraction mapping property of value iteration, leading to diminishing error as we ascend from the leaf nodes to the root. The complete proof for Proposition 3 can be found in the supplemental material. Finally, we get the main result.
Theorem 1. We have at the root node s 0 ,
E[V m (s 0 , n)] -V (s 0 ) ⩽ O(n -1/2 ).
Our proposed method, W-MCTS, achieves a polynomial convergence rate of O(n -1/2 ), matching the results of Dam et al. (2024b). In contrast, Xiao et al. (2019) introduced MENTS, followed by RENTS and TENTS from Dam et al. (2021), which leverage exponential convergence to a regularized value function through maximum entropy regularization. However, these methods face bias due to errors in the regularized value function, potentially leading to incorrect action selection. Conversely, Painter et al. (2024) employ a similar action selection strategy with a maximum backup operator for value estimation, resulting in exponential reductions in simple regret. However, their method's effectiveness heavily relies on the temperature parameter in Boltzmann exploration, limiting its practical use.
this section cite: ['b34', 'b13', 'b20']

Section: WASSERSTEIN NON-STATIONARY MULTI-ARMED BANDIT
A crucial part of the proof for Theorem 1 is to derive the following result for the W-MCTS in bandit setting. Under the Assumption 1, we consider applying Thompson Sampling strategy as the action selection method for the nonstationary multi-armed bandit (MAB) problems describes above. At each time step n, an action is selected as
a = argmax ai,i∈{1...K} {θ i ∼ N (X k,n , V k /T k (n))}. (3) Let's define X n (p) = K a=1 Ta(n) n X p a,Ta(n) 1/p as the power mean value backup at the root node, T a (n) = n-1 t=1 1(a t = a)
is the number of selections of a prior to round n. We show theoretical results of our method as follows. Under the Assumption 1, we establish the concentration properties of the power mean backup operator X n (p) towards the mean value of the optimal arm µ * = max a {µ a }, a ∈ [K], as shown in Theorem 2.
Theorem 2. Consider a non-stationary bandit problem described as in 7.1 with action selection as Equation (3). Then,
Pr( X n (p) -µ ⋆ ⩾ ε) ⩽ Cn -1 ε -2 .
Theorem 2 states the concentration properties of the power mean estimation by W-MCTS for a non-stationary continuous-armed bandit problem, and play an important role for the induction proof of Proposition 3 leading to the main result presented at Theorem 1.
this section cite: []

Section: Experiments

this section cite: []

Section: Fully Observable, Highly Stochastic Tasks
We compare W-MCTS to UCT (Kocsis et al., 2006), Power-UCT (Dam et al., 2019), and DNG (Bai et al., 2013) in five benchmark environments: FrozenLake, NChain, RiverSwim, SixArms, and Taxi. These tasks all feature significant stochasticity or long-horizon exploration challenges. FrozenLake. A 4 × 4 grid with slippery transitions, implemented in OpenAI Gym (Brockman et al., 2016). The agent aims to reach a goal in the bottom-right corner. Due to frequent slips, each move has high uncertainty. Figure 1 shows that W-MCTS-TS (Thompson sampling) outperforms DNG, UCT, Power-UCT, and W-MCTS (optimistic selection), with W-MCTS at p = 1 performing comparably to W-MCTS-TS.
NChain. An agent can move forward or backward along a chain of length 5. Actions may reverse with 20% probability, making consistent forward progress difficult. In Figure 1, both W-MCTS-TS and W-MCTS-OS exceed UCT and Power-UCT in convergence speed and final returns. RiverSwim. Similar to NChain but more complex transitions: sometimes the agent remains in the same state or only partially moves. This rewards long-term planning to reach high-value states. As in Figure 1, W-MCTS-OS converges fastest and attains the best performance, while Power-UCT eventually reaches similar returns more slowly. SixArms. A 7-state chain with 6 possible arms (actions) leading to different rewards that scale inversely with their success probabilities. This environment demands high exploration. Figure 1 shows that W-MCTS is the only method consistently securing strong returns. Taxi. A 7 × 6 grid where the agent must pick up three passengers, then reach a goal region. Slips occur 10% of the time, adding further uncertainty. Only W-MCTS-TS manages to collect all passengers reliably, outperforming Power-UCT and W-MCTS with optimistic selection.
this section cite: ['b17', 'b10', 'b3', 'b6']

Section: Partially Observable, Highly Stochastic Tasks
We also test W-MCTS against POMCP(UCT), D2NG, and DESPOT in classic POMDP benchmarks: rocksample, pocman, Tag, and LaserTag. Code for POMCP(UCT) (Silver & Veness, 2010b), D2NG (Bai et al., 2014), and DESPOT (Somani et al., 2013) is used as released by the original authors. Rocksample. A robot on an n×n grid can sample or ignore k rocks, then exit. We test three variants: (11,11), (15,15), and (15,35). Figure 2 shows that W-MCTS-TS consistently outperforms both UCT and D2NG. Pocman. A partially observed maze (Silver & Veness, 2010a) where the agent must collect pellets while avoiding ghosts. Table 1 indicates that W-MCTS-TS with p = 100 outperforms UCT and D2NG across most rollout-budget settings, and W-MCTS-OS also matches or surpasses these baselines in some configurations.
this section cite: ['b4', 'b31']

Section: Comparison with DESPOT.
We additionally compare W-MCTS to DESPOT across Tag, LaserTag, rocksample (15 × 15), and Pocman.
Table 2 shows that W-MCTS-OS and W-MCTS-TS achieve higher returns than AB-DESPOT and AR-DESPOT in rocksample. Similarly, W-MCTS-TS surpasses DESPOT in Pocman, Tag, and LaserTag, while W-MCTS-OS outperforms AB-DESPOT in Pocman. Role 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 0.0 0.1 0.2 0.3 0.4 Discounted Return FrozenLake 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 2 3 4 5 6 Discounted Return NChain 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 0.0 0.2 0.4 0.6 0.8 Discounted Return RiverSwim 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 Discounted Return SixArms 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 0.0 0.1 0.2 0.3 0.4 0.5 Discounted Return Taxi DNG Power-UCT UCT W-MCTS-OS W-MCTS-TS W-MCTS-TS, p=1 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 5 0 5 10 15 20 Discounted Return rocksample 11x11 (16 actions) 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 5 0 5 10 15 20 Discounted Return rocksample 15x15 (20 actions) 0.0 0.2 0.4 0.6 0.8 1.0 1.2 Simulations 1e5 5 0 5 10 15 20 Discounted Return rocksample 15x35 (40 actions) D2NG UCT W-MCTS-OS W-MCTS-TS  Table 2: Average total discounted reward. The results for POMCP, and DESPOT are taken from (Somani et al., 2013).
T ag LaserT ag RS(15, 15) P ocman W-MCTS-OS -6.05 ± 0.56 -18.17 ± 0.46 19.76 ± 0.28 297.98 × 2.83 W-MCTS-TS -5.90 ± 0.66 -8.75 ± 0.5 20.29 ± 0.22 315.45 ± 2.15 POMCP -7.14 ± 0.28 -19.58 ± 0.06 12.23 ± 0.32 294.16 ± 4.06 AB-DESPOT -6.57 ± 0.26 -11.13 ± 0.30 18.18 ± 0.30 290.34 ± 4.12 AR-DESPOT -6.26 ± 0.28 -9.34 ± 0.26 18.57 ± 0.30 307.96 ± 4.22 of α-Divergence. We explored several values of α to vary how aggressively our backups shift between average-like and max-like behavior. When α approaches 0 or ∞, the update becomes nearly a pure average (p = 1) or nearly a max backup, respectively. In practice, we found that moderate α values often provide a suitable balance between these extremes, and we report results with the bestperforming choices. Although a more extensive sensitivity analysis could be conducted, the core takeaway is that combining power-mean backups with variance propagation significantly enhances performance in highly stochastic tasks.
this section cite: ['b31']

Section: Key Performance Factors
The superior performance of our method stems from two complementary components that address fundamental lim-itations in existing MCTS approaches for stochastic and partially observable environments: Explicit Variance Propagation. Unlike previous methods that only propagate point estimates or use fixed variance models, our approach dynamically updates both means and variances at each node through the L 1 -Wasserstein barycenter formulation. This capability is particularly crucial in highly stochastic and partially observable environments where uncertainty quantification directly impacts decision quality. Our experimental results demonstrate consistent improvements over Bayesian MCTS methods: we achieve up to 80% improvement over DNG in Frozen-Lake, and significant gains over POMCP across all POMDP environments, with particularly notable improvements of 55.31% in LaserTag and 65.90% in rocksample(15,15). Additionally, we observe improvements of up to 21.38% over AB-DESPOT in LaserTag, highlighting the effectiveness of our distributional approach.
Flexibility in Balancing Exploration-Exploitation. Our approach's ability to interpolate between average-like and max-like backups through the α-divergence parameter allows adaptive behavior across varying levels of stochasticity. In highly stochastic environments such as FrozenLake and NChain, we found that moderate α values (leading to more average-like updates with p closer to 1) performed optimally by preventing overestimation bias. Conversely, in environments with more deterministic regions of the state space, larger α values (yielding more max-like behavior) proved beneficial for faster convergence to optimal policies. This flexibility, combined with our Thompson sampling strategy, enables our algorithm to automatically adapt its exploration-exploitation balance based on the empirical variance observed at each node.
The synergy between these two components-principled uncertainty propagation and adaptive backup operatorsexplains why W-MCTS consistently outperforms both classical MCTS variants and existing Bayesian approaches across our diverse set of benchmark environments.
this section cite: []

Section: Conclusion
We proposed Wasserstein MCTS, an algorithm that models node values as Gaussian distributions and employs L 1 -Wasserstein barycenters with α-divergences to unify average-and max-like backups. Coupled with Thompson sampling or optimistic selection, our method achieves strong empirical performance while offering O(n -1/2 ) convergence guarantees. Experiments in both stochastic MDPs and POMDPs show significant improvements over classic baselines and Bayesian MCTS variants. Future work includes extending these Wasserstein-based ideas to open-loop planning (Leurent & Maillard, 2020;Bubeck & Munos, 2010) for even broader applicability.
this section cite: ['b18', 'b7']

Section: References
Ref_id:b0 Title: Planning in stochastic environments with a learned model Year: (2021)
Ref_id:b1 Title: Finite-time analysis of the multiarmed bandit problem Year: (2002)
Ref_id:b2 Title: Finite-time analysis of the multiarmed bandit problem Year: (2002-05)
Ref_id:b3 Title: Bayesian mixture modelling and inference based thompson sampling in monte-carlo tree search Year: (2013)
Ref_id:b4 Title: Thompson sampling based monte-carlo planning in pomdps Year: (2014)
Ref_id:b5 Title: The theory of dynamic programming Year: (1954)
Ref_id:b6 Title:  Year: (2016)
Ref_id:b7 Title: Open loop optimistic planning Year: (2010)
Ref_id:b8 Title: Families of alpha-betaand gamma-divergences: Flexible and robust measures of similarities Year: (2010)
Ref_id:b9 Title: Eine informationstheoretische ungleichung und ihre anwendung auf beweis der ergodizitaet von markoffschen ketten Year: (1964)
Ref_id:b10 Title: Generalized mean estimation in monte-carlo tree search Year: (2019)
Ref_id:b11 Title: A unified perspective on value backup and exploration in monte-carlo tree search Year: (2024)
Ref_id:b12 Title: Power mean estimation in stochastic monte-carlo tree search Year: (2024)
Ref_id:b13 Title: Convex regularization in monte-carlo tree search Year: (2021)
Ref_id:b14 Title: Fano's inequality for random variables Year: (2020)
Ref_id:b15 Title: Double q-learning Year: (2010)
Ref_id:b16 Title: Finite-time regret of thompson sampling algorithms for exponential family multi-armed bandits Year: (2022)
Ref_id:b17 Title: Improved monte-carlo search Year: (2006)
Ref_id:b18 Title: Practical open-loop optimistic planning Year: (2019)
Ref_id:b19 Title: Propagating uncertainty in reinforcement learning via wasserstein barycenters Year: (2019)
Ref_id:b20 Title: Monte carlo tree search with boltzmann exploration. Advances in Neural Information Processing Systems Year: (2024)
Ref_id:b21 Title: Jensen's inequality for a convex vectorvalued function on an infinite-dimensional space Year: (1974)
Ref_id:b22 Title: Mastering atari, go, chess and shogi by planning with a learned model Year: (2020)
Ref_id:b23 Title: Monte-carlo planning in large pomdps Year: (2010)
Ref_id:b24 Title: Monte-carlo planning in large pomdps Year: (2010)
Ref_id:b25 Title: Mastering the game of Go with deep neural networks and tree search Year: (2016-01)
Ref_id:b26 Title: Mastering the game of go with deep neural networks and tree search Year: (2016)
Ref_id:b27 Title: Mastering chess and shogi by self-play with a general reinforcement learning algorithm Year: (2017)
Ref_id:b28 Title: Mastering the game of go without human knowledge Year: (2017)
Ref_id:b29 Title: Mastering the game of go without human knowledge Year: (2017)
Ref_id:b30 Title: The book of statistical proofs Year: (2020)
Ref_id:b31 Title: Online pomdp planning with regularization Year: (2013)
Ref_id:b32 Title: Bayesian inference in monte-carlo tree search Year: (2012)
Ref_id:b33 Title: On the likelihood that one unknown probability exceeds another in view of the evidence of two samples Year: (1933)
Ref_id:b34 Title: Maximum entropy monte-carlo planning Year: (2019)
