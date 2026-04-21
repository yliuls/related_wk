Title: Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning
Abstract: Designing efficient algorithms for multi-agent reinforcement learning (MARL) is fundamentally challenging because the size of the joint state and action spaces grows exponentially in the number of agents. These difficulties are exacerbated when balancing sequential global decision-making with local agent interactions. In this work, we propose a new algorithm SUBSAMPLE-MFQ (Subsample-Mean-Field-Q-learning) and a decentralized randomized policy for a system with n agents. For any k ≤ n, our algorithm learns a policy for the system in time polynomial in k. We prove that this learned policy converges to the optimal policy on the order of Õ(1/ √ k) as the number of subsampled agents k increases. In particular, this bound is independent of the number of agents n. * Work partially done while a visiting student at Carnegie Mellon University and intern at Cognition AI.

Section: Introduction
Reinforcement Learning (RL) has become a popular framework to solve sequential decision making problems in unknown environments and has achieved tremendous success in a wide array of domains such as playing the game of Go [Silver et al., 2016], robotic control [Kober et al., 2013], and autonomous driving [Kiran et al., 2022, Lin et al., 2023a]. A key feature of most real-world systems is their uncertain nature, and thus, RL has emerged as a powerful tool for learning optimal policies for multi-agent systems to operate in unknown environments [Kim and Giannakis, 2017, Zhang et al., 2021, Lin et al., 2024, Anand and Qu, 2024]. While early RL works focused on the single-agent setting, multi-agent RL (MARL) has recently achieved impressive success in many applications, such as coordinating robotic swarms [Preiss et al., 2017, DeWeese andQu, 2024], real-time bidding [Jin et al., 2018], ride-sharing [Li et al., 2019], and stochastic games [Jin et al., 2020].
Despite growing interest in MARL, extending RL to multi-agent settings poses significant computational challenges due to the curse of dimensionality. MARL is fundamentally difficult as agents in the real-world not only interact with the environment but also with each other [Shapley, 1953]: if each of the n agents has a state space S and action space A, the global state-action space has size (|S||A|) n , which is exponential in n. Thus, many RL algorithms (such as temporal difference learning and tabular Q-learning) require computing and storing an (|S||A|) n -sized Q-table [Sutton et al., 1999, Bertsekas andTsitsiklis, 1996]. This scalability issue has been observed in a variety of MARL settings [Blondel and Tsitsiklis, 2000, Papadimitriou and Tsitsiklis, 1999, Littman, 1994].
An exciting line of work that addresses this intractability is mean-field MARL [Lasry and Lions, 2007, Yang et al., 2018, Gu et al., 2021, 2022, Hu et al., 2023]. The mean-field approach assumes agents are homogeneous in their state-action spaces, enabling their interactions to be approximated by a two-agent setting: here, each agent interacts with a representative "mean agent" which evolves as the empirical distribution of states of all other agents. With these assumptions, mean-field MARL learns an optimal policy with sample complexity O(n |S||A| |S||A|), which is polynomial in the number of agents. However, if n is large, this remains prohibitive even for moderate values of |S| and |A|.
Motivated by this problem, in this paper we study the cooperative setting-where agents work collaboratively to maximize a structured global reward-and ask: Can we design a scalable MARL algorithm for learning an approximately optimal policy in a cooperative multi-agent system?
Contributions. We answer this question affirmatively. Our key contributions are outlined below.
Subsampling algorithm. We model the problem as a Markov Decision Process (MDP) with a global agent and n local agents. We propose SUBSAMPLE-MFQ to address the challenge of MARL with a large number of local agents. SUBSAMPLE-MFQ selects k ≤ n local agents to learn a deterministic policy πest k by applying mean-field value iteration on the k-local-agent subsystem to learn Qest k , which can be viewed as a smaller Q function. SUBSAMPLE-MFQ then deploys a stochastic policy π est k , where the global agent samples k local agents uniformly at each step and uses πest k to determine its action, while each local agent samples k -1 other local agents and uses πest k to determine its action. Sample complexity and theoretical guarantee. As the number of local agents increases, the size of Qk scales polynomially with k, rather than polynomially with n as in mean-field MARL. Analogously, when the size of the local agent's state space grows, the size of Qk scales exponentially with k, rather than exponentially with n, as in traditional Q-learning. The key analytic technique underlying our results is a novel MDP sampling result. This result shows that the performance gap between π est k and the optimal policy π * is at most Õ(1/ √ k), with high probability. The choice of k reveals a fundamental trade-off between the size of the Q-table and the optimality of π est k . For example, if k is set to O(log n), SUBSAMPLE-MFQ is the first centralized MARL algorithm to achieve a polylogarithmic run-time in n, representing an exponential speedup over the previously best-known polytime mean-field MARL methods, while maintaining a decaying optimality gap as n gets large, While our results are theoretical in nature, we hope SUBSAMPLE-MFQ will further exploration of sampling in Markov games, and potentially inspire new practical multi-agent algorithms.
Related work. MARL has a rich history, starting with early works on Markov games [Littman, 1994, Sutton et al., 1999], which are a multi-agent extension of MDPs. MARL has since been actively studied [Zhang et al., 2021] in a broad range of settings. MARL is most similar to the category of "succinctly described" MDPs [Blondel and Tsitsiklis, 2000], where the state/action space is a product space formed by the individual state/action spaces of multiple agents, and where the agents interact to maximize an objective. A recent line of research constrains the problem to sparse networked instances to enforce local interactions between agents [Qu et al., 2020a, Lin et al., 2020, Mondal et al., 2022]. In this formulation, the agents correspond to vertices on a graph who only interact with nearby agents. By exploiting Gamarnik's correlation decay property from combinatorial optimization [Gamarnik et al., 2009], they overcome the curse of dimensionality by simplifying the problem to only search over the policy space derived from the truncated graph to learn approximately optimal solutions. However, as the underlying network structure becomes dense with many local interactions, the neighborhood of each agent gets large, and these algorithms become intractable.
Mean-Field RL. Under assumptions of homogeneity in the state/action spaces of the agents, the problem of densely networked multi-agent RL was studied by Yang et al. [2018], Gu et al. [2021], who approximated the solution in polynomial time with a mean-field approach where the approximation error scales in O(1/ √ n). In contrast, our work achieves subpolynomial runtimes by directly sampling from this mean-field distribution. Cui and Koeppl [2022] introduce heterogeneity to mean-field MARL by modeling non-uniform interactions through graphons; however, these methods crucially assume the existence of graphon sequences that converge in cut-norm to the finite graph. In the cooperative setting, Subramanian et al. [2022], Cui et al. [2023] studies a mean-field setting with q types of homogeneous agents; however, their learned policy does not provably converge to the optimum.
Other related works. Our work is related to factored MDPs, where there is a global action affecting every agent; however, in our case, each agent has its own action [Min et al., 2023, Lauer andRiedmiller, 2000]. Jin et al. [2020] reduces the dependence of the product action space to an additive dependence with V-learning. Our work further reduces the complexity of the joint state space, which has not been previously accomplished. We add to the growing literature on the Centralized Training with Decentralized Execution regime [Zhou et al., 2023], as our algorithm learns a provably approximately optimal policy using centralized information, but makes decisions using only local information during execution. Finally, one can efficiently approximate the Q-table through function approximation [Jin et al., 2021]. However, achieving theoretical bounds on the performance loss due to function approximation is intractable without strong assumptions such as linear Bellman completeness or low Bellman-Eluder dimension [Golowich and Moitra, 2024]. While our work primarily studies the finite tabular setting, we extend it to non-tabular linear MDPs in Appendix J.
this section cite: ['b61', 'b35', 'b33', 'b32', 'b72', 'b45', 'b16', 'b29', 'b41', 'b27', 'b58', 'b7', 'b7', 'b51', 'b46', 'b37', 'b24', 'b26', 'b46', 'b63', 'b72', 'b7', 'b49', 'b21', 'b70', 'b24', 'b14', 'b62', 'b15', 'b39', 'b27', 'b73', 'b28', 'b23']

Section: Preliminaries
Notation. For k, n ∈ N where k ≤ n, let [n]  k denote the set of k-sized subsets of [n] = {1, . . . , n}. For any vector z ∈ R d , let ∥z∥ 1 and ∥z∥ ∞ denote the standard ℓ 1 and ℓ ∞ norms of z respectively. Let ∥A∥ 1 denote the matrix ℓ 1 -norm of A ∈ R n×m . Given variables s 1 , . . . , s n , s ∆ := {s i : i ∈ ∆} for ∆ ⊆ [n]. We use Õ(•) to suppress polylogarithmic factors in all problem parameters except n. For a discrete measurable space (X , F), the total variation distance between probability measures ρ 1 , ρ 2 is given by TV(ρ 1 , ρ 2 ) = 1 2 x∈X |ρ 1 (x) -ρ 2 (x)|. Next, x ∼ D(•) denotes that x is a random element sampled from a distribution D, and we denote that x is a random sample from the uniform distribution over a finite set Ω by x ∼ U(Ω). We include a detailed notation table in Table 1.
this section cite: []

Section: Problem formulation
We consider a system of n + 1 agents, where agent g is a "global decision making agent" and the remaining n agents, denoted by [n], are "local agents." At time t, the agents are in state s(t) = (s g (t), s 1 (t), ..., s n (t)) ∈ S := S g × S n l , where s g (t) ∈ S g denotes the global agent's state, and for each i ∈ [n], s i (t) ∈ S l denotes the state of the i'th local agent. The agents cooperatively select actions a(t) = (a g (t), a 1 (t), ..., a n (t)) ∈ A := A g ×A n l , where a g (t) ∈ A g denotes the global agent's action and a i (t) ∈ A l denotes the i'th local agent's action. At time t + 1, the next state for all the agents is independently generated by stochastic transition kernels P g : S g × S g × A g → ∆(S g ) and P l : S l × S l × S g × A l → ∆(S l ) as follows:
s g (t + 1) ∼ P g (•|s g (t), a g (t)), s i (t + 1) ∼ P l (•|s i (t), s g (t), a i (t)), ∀i ∈ [n].
(1)
The system then collects a structured stage reward r(s(t), a(t)) where the reward r : S × A → R depends on s(t) and a(t) through Equation ( 2), and where r g and r l is typically application specific.
this section cite: []

Section: r(s, a)
= r g (s g , a g ) global component + 1 n i∈[n] r l (s i , s g , a i ) local component(2)
A policy π : S → P(A) maps from states to distributions of actions such that a ∼ π(•|s). Given γ ∈ (0, 1), we seek to learn a policy π that maximizes the (γ-discounted) value for each s ∈ S:
V π (s) = E a(t)∼π(•|s) ∞ t=0 γ t r(s(t), a(t))|s(0) = s .(3)
The cardinality of the search space simplex for the optimal policy is
|S g ||S l | n |A g ||A l | n
, which is exponential in the number of agents, underscoring the need for efficient approximation algorithms.
To efficiently learn policies that maximize the objective, we make the following standard assumptions: Assumption 2.1 (Finite state/action spaces). We assume that the state and action spaces of all the agents in the MARL game are finite:
|S l |, |S g |, |A g |, |A l | < ∞.
Appendix J of the supplementary material relaxes this assumption to the non-tabular setting with infinite continuous sets. Assumption 2.2 (Bounded rewards). The components of the reward function are bounded. Specifically, ∥r g (•, •)∥ ∞ ≤ rg , and ∥r l (•, •, •)∥ ∞ ≤ rl . This implies ∥r(•, •)∥ ∞ ≤ rg + rl := r. Definition 2.3 (ϵ-optimal policy). Given a policy simplex Π, π ∈ Π is an ϵ-optimal policy if
V π (s) ≥ sup π * ∈Π V π * (s) -ϵ.
Motivating examples. Below we give examples of two cooperative MARL settings which are naturally modeled by our setting. Our experiments in Appendix B reveal a monotonic improvement in the learned policies as k → n, while providing a substantial speedup over mean-field Q-learningfoot_0 .
Figure 1: Bounded exploration in warehouse accidents, and traffic congestions with Gaussian squeeze.
• Gaussian squeeze: In this task, n homogeneous agents determine individual actions a i to jointly maximize the objective r(x) = xe -(x-µ) 2 /σ 2 , where x = n i=1 a i , and µ and σ are the predefined mean and variance of the system. In scenarios of traffic congestion, each agent i ∈ [n] is a controller trying to send a i vehicles into the main road, where controllers coordinate with each other to avoid congestion, hence avoiding either over-use or under-use, thereby contributing to the entire system. This GS problem is previously studied in Yang et al. [2018], and serves as an ablation study on the impact of subsampling for MARL.
• Constrained exploration: Consider an M × M grid. Each agent's state is a coordinate in [M ] × [M ]. The state represents the center of a d × d box where the global agent constrains the local agents' movements. Initially, all agents are in the same location. At each time-step, the local agents take actions a i (t) ∈ R 2 (e.g., up, down, left, right) to transition between states and collect rewards. The transition kernel ensures that local agents remain within the d × d box dictated by the global agent, using knowledge of a i (t), s g (t), and s i (t). In warehouse settings where shelves have collapsed, creating hazardous or inaccessible areas, we want agents to clean these areas. However, exploration in these regions may be challenging due to physical constraints or safety concerns, causing exploration in these regions to be disincentivized from the local agents' perspectives. Through an appropriate design of the reward and transition functions, the global agent could guide the local agents to focus on specific d × d grids, allowing efficient cleanup while minimizing risk.
Capturing heterogeneity. Following Mondal et al. [2022], Xu and Klabjan [2023], our model can capture heterogeneity in the local agents by modeling agent types as part of the state: to do this, we assign a type ε i ∈ E to each local agent by letting S l = E × S ′ l , where E is a set that enumerates possible types that are treated as a fixed part of the agent's state, and S ′ l is the latent state space of any local agent. The transition and reward functions can vary depending on the agent's type. The global agent can provide unique signals to local agents of each type by letting s g ∈ S g and a g ∈ A g denote a state/action vector where each element corresponds to a type.
this section cite: ['b70', 'b69']

Section: Algorithmic Approach: Subsampled Value Iteration
Q-learning. Our starting point is the classic Q-learning framework [Watkins and Dayan, 1992] for offline-RL, which seeks to learn the Q-function Q :
S × A → R. For any policy π, Q π (s, a) = E π [ ∞ t=0 γ t r(s(t), a(t))|s(0) = s, a(0) = a].
Initially, Q 0 (s, a) = 0, for all (s, a) ∈ S ×A. Then, for all t ∈ [T ], it is updated as Q t+1 (s, a) ← T Q t (s, a), where the Bellman operator T is
T Q t (s, a) = r(s, a) + γE s ′ g ∼Pg(•|sg,a),s ′ i ∼P l (•|si,sg,ai),∀i∈[n] max a ′ ∈A Q t (s ′ , a ′ ).(4)
It is well known that T is γ-contractive, ensuring that the above updates converge to a unique Q * such that T Q * = Q * . The optimal policy π * : S → A can then be computed greedily as π * (s) = arg max a∈A Q * (s, a). However, the update complexity for the Q-function is
O(|S||A|) = O(|S g ||S l | n |A g ||A l | n ),
which is exponential in the number of local agents increases.
this section cite: ['b68']

Section: Mean-field transformation.
To address this, mean-field MARL [Yang et al., 2018] (under homogeneity assumptions) studies the empirical distribution function F z [n] : Z l → R, for Z l := S l × A l :
F z [n] (z) := 1 n n i=1 1{s i = z s , a i = z a }, ∀z := (z s , z a ) ∈ Z l := S l × A l . (5
) Let µ n (Z l ) = { b n |b ∈ {0, . . . , n}} |Z l | be the space of |Z l |-sized vectors (or |S l | × |A l |-sized tables), where each entry is in {0, 1/n, 2/n, . . . , 1}. Intuitively, µ n (Z l ) is a discrete distribution over (S l , A l )
where each probability assigned is a multiple of 1/n. Then, F z [n] ∈ µ n (Z l ) indicates the proportion of agents in each state-action pair.
As in Q-learning, in mean-field Q-learning initially Q0 (s g , s 1 , a g , a 1 , F z [n]\1 ) = 0. At each time t ∈ [T ], we update Q as Qt+1 = T Qt , where T is the Bellman operator in distribution space:
T Qt (s g , s 1 , a g , a 1 , F z [n]\1 ) = r(s, a) + γE s ′ g ∼Pg(•|sg,ag) s ′ i ∼P l (•|si,sg,ai) ∀i∈[n] max (a ′ g ,a ′ 1 ,a ′ [n]\1 ) ∈Ag×A l ×A n-1 l Qt (s ′ g , s ′ 1 , a ′ g , a ′ 1 , F z ′ [n]\1 )
Since the Q-function is permutation-invariant in the homogeneous local agents, one sees that for each
t ≥ 0, Q t (s g , s [n] , a g , a [n] ) = Qt (s g , s 1 , a g , a 1 , F z [n]\1
). In other words, mean-field Q-learning implements the same updates as standard Q-learning. However, the update complexity of Q is only
O(|S g ||A g ||Z l |n |Z l | ), which scales polynomially in n but exponentially in |Z l |. Remark 3.1. Existing methods use sample complexity Õ(min{|S g ||A g ||Z l | n , |S g ||A g ||Z l |n |Z l | }): one uses Q-learning if |Z l | n-1 < n |Z l |
, and mean-field value iteration otherwise. In each regime, as n scales, the update complexity becomes computationally infeasible.
To further reduce the update complexity, we propose SUBSAMPLE-MFQ to overcome the polynomial (in n) sample complexity of mean-field Q-learning and the exponential (in n) sample complexity of traditional Q-learning. We begin by motivating the intuition behind our algorithms.
this section cite: ['b70']

Section: Overview of approach.
Offline Planning: Algorithm 1. First, the global agent randomly samples a subset of local agents ∆ ⊆ [n] such that |∆| = k, for k ≤ n. It then ignores all other local agents [n] \ ∆, and performs value iteration (using m samples to update the Q-function in each iterate) to approximately learn the Q-function Qest k,m and policy πest k,m for this surrogate subsystem of k local agents. We denote the surrogate reward gained by this subsystem at each time step by r ∆ : S × A → R, where
r ∆ (s, a) = r g (s g , a g ) + 1 |∆| i∈∆ r l (s g , s i , a i ).(6)
In Theorem E.3, we show that ∥ Qest k,m -Q * ∥ ∞ is Lipschitz continuous with respect to the TV-distance between the state/action pairs of the subsampled agents and the full set of n agents. Equipped with this approximation guarantee, we show how to construct an approximately optimal policy on the full system on n agents. In general, converting this policy on k local agents to a policy on the full n-agent system without sacrificing error guarantees can be intractable, and there is a line on works on centralized-training decentralized-execution (CTDE) [Xu andKlabjan, 2023, Zhou et al., 2023] which shows that designing performant decentralized policies can be highly non-trivial.
Online Execution: Algorithm 2. In order to circumvent this obstacle and convert the optimality of each agent's action in the k local-agent subsystem to an approximate optimality guarantee on the full n-agent system, we propose a randomized policy π est k,m , where the global agent samples ∆ ∈ [n]  k at each time-step to derive an action a g ← πest k,m (s g , s ∆ ), and where each local i agent samples k -1 other local agents ∆ i to derive an action a i ← πest k,m (s g , s i , s ∆i ). Finally, Theorem 4.4 shows that the policy π est k,m converges to the optimal policy π * as k → n with rate Õ( 1 √ k ).
this section cite: []

Section: Algorithm description.
We now formally describe the algorithms. Before describing Algorithm 1 (SUBSAMPLE-MFQ: Learning) and Algorithm 2 (SUBSAMPLE-MFQ: Execution) in detail, it will be helpful to first define the empirical distribution function: Definition 3.2 (Empirical Distribution Function). For any population (z 1 , . . . , z n ) ∈ Z n l , where Z l := S l × A l , define the empirical distribution function F z∆ : Z l → R + for all z := (z s , z a ) ∈ S l × A l and for all ∆ ∈ [n]  k by F z∆ (x) := 1
k i∈∆ 1{s i = z s , a i = z a }. Let µ k (Z l ) := b k |b ∈ {0, . . . , k} |Z l | be the space of |Z l |-sized vectors (or |S l | × |A l |-sized tables)
where each entry is in {0, 1/k, 2/k, . . . , 1}.
Intuitively, µ k (Z l ) is a discrete distribution over (S l , A l )
where each probability assigned is a multiple of 1/k. Here, F z∆ ∈ µ k (Z l ) indicates the proportion of agents (in the k-local-agent subsystem) at each state/action pair.
Algorithm 1 (Offline learning). Let m ∈ N denote the sample size for the learning algorithm with sampling parameter k ≤ n. As in Remark 3.1, when
|Z l | k ≤ |Z l |k |Z l |
, the algorithm uses traditional value-iteration, and when
|Z l | k > |Z l |k |Z l |
, it uses mean-field value iteration. We formally describe the procedure for each regime below:
Regime with Large State/Action Space:
When |Z l | k ≤ |Z l |k |Z l |
, we iteratively learn the optimal Q-function for a subsystem with k-local agents denoted by Qt k,m :
S g × S k l × A g × A k l → R, which is initialized to 0. At time t, we update Qt+1 k,m (s g , s ∆ , a g , a ∆ ) = Tk,m Qt k,m (s g , s ∆ , a g , a ∆ ),(7)
where Tk,m is the empirically adapted Bellman operator in Equation ( 8). Since the system is unknown, Tk,m cannot directly perform the Bellman update, so it instead uses m random samples
s j g ∼ P g (•|s g , a g ) and s j i ∼ P l (•|s i , s g , a i ) for each j ∈ [m], i ∈ ∆ to approximate the system: Tk,m Qt k,m (s g , s ∆ , a g , a ∆ ) = r ∆ (s, a) + γ m j∈[m] max a ′ g ∈Ag,a ′ ∆ ∈A k l Qt k,m (s j g , s j ∆ , a ′ g , a ′ ∆ ).
Since Tk,m is γ-contractive, Algorithm 1 applies value iteration with T until Qk,m converges to a fixed point satisfying Tk,m Qest k,m = Qest k,m , yielding a deterministic policy πest k,m (s g , s ∆ ) where
πest k,m (s g , s ∆ ) = arg max ag∈Ag,a∆∈A k l Qest k,m (s g , s ∆ , a g , a ∆ ). (8
)
Regime with Large Number of Agents:
When |Z l | k > |Z l |k |Z l | , we learn the optimal mean-field Q-function for a k local agent system, denoted by Qt k,m : S g × S l × µ k-1 (Z l ) × A g × A l → R, which is initialized to 0. At time t, we update Qt+1 k,m (s g , s 1 , F z ∆ , a 1 , a g ) = Tk,m Qt k,m (s g , s 1 , F z ∆ , a 1 , a g ),(9)
where Tk,m is the empirically adapted mean-field Bellman operator in Equation ( 10). Similarly, as the system is unknown, Tk,m cannot directly perform the Bellman update and instead uses m random samples
s j g ∼ P g (•|s g , a g ) and s j i ∼ P l (•|s i , s g , a i ), ∀j ∈ [m], i ∈ ∆ to approximate the system: Tk,m Qt k,m (s g , s 1 , F z ∆ , a 1 , a g ) = r ∆ (s, a)+ γ m j∈[m] max a ′ g ∈Ag,a ′ 1 ∈A l , F a ′ ∆ ∈µ k-1 (A l ) Qt k,m (s j g , s j 1 , F s j ∆,a ∆′ , a ′ 1 , a ′ g ) (10
)
Since Qt k,m depends on s ∆ and a ∆ through F z∆ , Tk,m is also γ-contractive. So, Algorithm 1 applies value iteration with T until Qk,m converges to a fixed point satisfying Tk,m Qest k,m = Qest k,m , yielding a deterministic policy πest
k,m (s g , s 1 , F s ∆ ): πest k,m (s g , s 1 , F s ∆ ) = arg max ag∈Ag,a1∈A l ,Fa ∆ ∈µ k-1 (A l ) Qest k,m (s g , s 1 , F z ∆ , a 1 , a g ) For πest k,m (s g , s i , s ∆\i ) = a * g , a * i , a * ∆\i , let [π est k,m (s g , s ∆ )] g = a * g and [π est k,m (s g , s i , s ∆\i )] l = a * i .
this section cite: []

Section: Algorithm 1 SUB-SAMPLE-MFQ: Learning
Require: A multi-agent system as in Section 2, number of iterations T , sampling parameters k ∈ [n], m ∈ N, and discount factor γ ∈ (0, 1). 1: Let ∆ = {1, . . . , k}, ∆ = {2, . . . , k}, and
µ k-1 (Z l ) = { b k-1 : b ∈ {0, 1, . . . , k -1}} |S l |×|A l | . 2: if |Z l | k ≤ |Z l |k |Z l | then 3: // Sub-sampled version of standard Q-learning 4: Initialize Q0 k,m (s g , s ∆ , a g , a ∆ ) = 0, ∀(s g , s ∆ , a g , a ∆ ).
5:
for t = 1 to T do 6:
for (s g , s ∆ , a ∆ , a g ) ∈ S g × S k l × A g × A k l do 7: Qt+1 k,m (s g , s ∆ , a g , a ∆ ) = Tk,m Qt k,m (s g , s ∆ , a g , a ∆ ) 8:
Let the greedy policy be πest k,m (s g , s ∆ ) := arg max ag∈Ag,a∆∈A k l QT k,m (s g , s ∆ , a g , a ∆ ). 9: else 10:
// Sub-sampled version of mean-field Q-learning 11:
Initialize Q0 k,m (s g , s 1 , F z ∆ , a 1 , a g ) = 0, ∀(s g , s ∆ , a ∆ , a g ).
12:
for t = 1 to T do 13:
for (s g , s 1 , F z ∆ , a 1 , a g ) ∈ S g × S l × µ k-1 (Z l ) × A l × A g do 14: Qt+1 k,m (s g , s 1 , F z ∆ , a 1 , a g ) = Tk,m Qt k,m (s g , s 1 , F z ∆ , a 1 , a g ) 15:
Let the greedy policy be πest k,m (s g , s i , F s ∆ ) := arg max ag∈Ag,ai∈A l ,
Fa ∆ ∈µ k-1 (A l ) QT k,m (s g , s 1 , F z ∆ , a 1 , a g ).
Remark 3.3. Since mean-field Q-learning maintains the same updates as standard Q-learning (from Lemma C.19 which follows by noting that the Q-function is permutation-invariant in the homogeneous local agents), the deterministic policies πest k,m (s g , s ∆ ) and πest k,m (s g , s 1 , F s ∆ ) are equivalent.
Algorithm 2 (Online implementation). In Algorithm 2, (SUBSAMPLE-MFQ: Execution) the global agent samples local agents ∆(t) ∼ U [n]  k at each step to derive action a g (t) = [π est k,m (s g , s ∆ (t))] g , and each local agent i samples other local agents ∆ i (t) ∼ U [n]\i k-1 to derive action a i (t) = [π est k,m (s g , s i , s ∆ (t))] l . The system then incurs a reward r(s, a). This procedure of first sampling agents and then applying πest k,m is denoted by a stochastic policy π est k,m (a|s), where π est k,m (a g |s) is the global agent's action distribution and π est k,m (a l |s) is the local agent's action distribution:
π est k,m (a g |s) = 1 n k ∆∈( [n] k ) 1(π est k,m (s g , s ∆ ) = a)(11)
π est k,m (a i |s) = 1 n-1 k-1 ∆∈( [n]\i k-1 ) 1(π est k,m (s g , s i , F s ∆ ) = a i )(12)
The agents then transition to their next states.
this section cite: []

Section: Theoretical Guarantees and Analysis Approach
We now show the value of the expected discounted cumulative reward produced by π est k,m is approximately optimal, where the optimality gap decays as k → n and m grows.
Bellman noise. We introduce the notion of Bellman noise, which is used in the main theorem. Note that Tk,m is an unbiased estimator of the adapted Bellman operator Tk ,
Tk Qk (s g , s ∆ , a g , a ∆ ) = r ∆ (s, a)+γE s ′ g ∼Pg(•|sg,ag), s ′ i ∼P l (•|si,sg,ai),∀i∈∆ max a ′ g ∈Ag,a ′ ∆ ∈A k l Qk (s ′ g , s ′ ∆ , a ′ g , a ′ ∆ ). (13) Let Q0 k (s g , s ∆ , a g , a ∆ ) = 0. For t > 0, let Qt+1 k = Tk Qt k ,
where Tk is defined for k ≤ n in Equation (13). Then, Tk is also a γ-contraction with fixed-point Q * k . By the law of large numbers, lim m→∞ Tk,m = Tk and
∥ Qest k,m -Q * k ∥ ∞ → 0 as m → ∞. For finite m, ϵ k,m := ∥ Qest k,m -Q * k ∥ ∞
Algorithm 2 SUBSAMPLE-MFQ: Execution Require: Parameter T ′ for the number of iterations for the decision-making sequence. Sampling parameter k ∈ [n], m ∈ N. Discount factor γ. Policy πest k,m (s g , F s∆ ). 1: Learn πest k,m from Algorithm 1. 2: Sample (s g (0), s [n] (0)) ∼ s 0 , where s 0 is a distribution on the initial global state (s g , s [n] ) 3: Initialize the total reward R 0 = 0. 4: Policy π est k,m (s) is defined as follows: 5: for t = 0 to T ′ do 6:
Choose ∆ uniformly at random from [n]  k and let a g (t) = [π est k,m (s g (t), s ∆ (t))] g . 7:
for i = 1 to n do 8:
Choose ∆ i uniformly at random from [n]\i k-1 and let
a i (t) = [π est k,m (s g (t), s i (t), s ∆i (t))] l . 9: Let s g (t + 1) ∼ P g (•|s g (t), a g (t)). 10: Let s i (t + 1) ∼ P l (•|s i (t), s g (t), a i (t)), ∀i ∈ [n]. 11: R t+1 = R t + γ t • r(s, a)
is the well-studied Bellman noise. To compare the performance between π * and π est k , we define the value function of a policy π: Definition 4.1. For a given policy π, the value function V π : S → R for S := S g × S n l is given by:
V π (s) = E a(t)∼π(•|s(t)) ∞ t=0 γ t r(s(t), a(t)) s(0) = s .(14)
Intuitively, V π (s) is the expected discounted cumulative reward when starting from state s and applying actions from the policy π across an infinite horizon.
With the above preparations, we are primed to present our main result: a high-probability bound on the optimality gap for our learned policy π est k,m that decays with rate Õ(1/ √ k).
Theorem 4.2. Let π est k,m denote the learned policy deployed in SUBSAMPLE-MFQ: Execution. Then, for all s 0 ∈ S := S g × S n l , we have
V π * (s 0 ) -V π est k,m (s 0 ) ≤ r (1 -γ) 2 n -k + 1 2nk ln 40r|S l ||A l ||A g |k |A l |+ 1 2 (1 -γ) 2 + 1 10 √ k + 2ϵ k,m
We prove Theorem 4.4 in Appendix G, and provide a proof sketch in Appendix D. We also generalize the result to stochastic rewards in Appendix H.
To control the Bellman noise ϵ k,m , we show that for sufficiently many samples m * , ϵ k,m * also decays on the order of Õ(1/ √ k) with high probability. For this, we introduce Lemma 4.3: Lemma 4.3 (Controlling the Bellman Noise.). For k ∈ [n], let
m * = 2|S g ||A g ||S l ||A l |k 3.5+|S l ||A l | log(|S g ||A g ||A l ||S l |) (1 -γ) 5 log 1 (1 -γ) 2 be the number of samples in Equation (10). If the number of iterations T satisfies T ≥ 2 1-γ log r√ k 1-γ
then with probability at least 1 -1 100e k , the Bellman error satisfies ϵ k,m * ≤ Õ( 1 √ k ). We defer the proof of Lemma 4.3 to Appendix G.1. To simplify notation, let π est k := π est k,m * . Then, by combining Lemma 4.3 and Theorem 4.2, we obtain our main result in Theorem 4.4: Theorem 4.4. Let π est k denote the learned policy from SUBSAMPLE-MFQ: Execution where the number of samples m is determined in Lemma 4.3. Suppose T ≥ 2 1-γ log r√ k 1-γ . Then, ∀s 0 ∈ S := S g × S n l , with probability at least 1 -1/100e k , we havefoot_1 ,
V π * (s 0 ) -V π est k (s 0 ) ≤ r (1 -γ) 2 n -k + 1 2nk ln 40r|S l ||A l ||A g |k |A l |+ 1 2 (1 -γ) 2 + 4 √ k .
Remark 4.5. One could also derive an alternate bound that retains the T factor via a γ T dependence in the final bound (which shows an exponentially decaying error with the time horizon T ). Moreover, in Lemma G.10, we show that the query complexity is on the order of O(mT
|S g ||S l | k |A g ||A l | k ),
and we bound T and set m = poly(1/(1 -γ)) to attain the 1/ √ k rate.
Remark 4.6. Additionally, our poly(1/(1 -γ))-dependence may be loose since we do not use more complicated variance reduction techniques as in Sidford et al. [2018a,b], Wainwright [2019], Jin et al. [2024] to optimize the number of samples m which is used to bound the Bellman error ϵ k,m . Moreover, incorporating variance reduction would significantly complicate the algorithm and intuition.
Our analysis hinges on two non-trivial technical steps. Firstly, for an intermediary step in Theorem E.3 we establish that TV distance, rather than the stronger KL-divergence, is the correct metric to use (as the KL-divergence between F z∆ and F z [n] decays too slowly as k → n, as we show in Lemma F.6). This requires exploiting a recent extension of the DKW inequality to sampling without replacement [Anand and Qu, 2024], and showing that the transition dynamics we study saturates the dataprocessing inequality for TV-distance. Secondly, we adapt the celebrated performance-difference lemma [Kakade and Langford, 2002] to our multi-agent setting, which entails a principled analysis and careful probabilistic argument (to which we refer the reader to Appendix G).
Remark 4.7. We also extend the formulation of SUBSAMPLE-MFQ to off-policy Q-learning Chen et al. [2021b], Chen and Maguluri [2022a], which replaces the generative oracle assumption with a stochastic approximation scheme that learns an approximately optimal policy using historical data. Appendix I provides theoretical guarantees with a similar decaying optimality gap as in Theorem 4.4.
Remark 4.8. The asymptotic sample complexity of Algorithm 1 for learning πest
k for a fixed k is min{ Õ(|Z l | k |S g ||A g |), Õ(k |Z l | |Z l ||S g ||A g |)}
, which is at least polynomially faster the standard Q-learning or mean-field value iteration as discussed in Remark 3.1. By Theorem 4.4, as k → n, the optimality gap decays, revealing a fundamental trade-off in the choice of k: increasing k improves the performance of the policy, but increases the size of the Q-function. We explore this trade-off further in our experiments. If we set k = O(log n), this leads to a sample complexity of min{ Õ(n
log |Z l | |S g ||A g |), Õ((log n) |Z l | |S g ||A g |)}.
This is an exponential speedup on the complexity from mean-field value iteration (from poly(n) to poly(log n)), as well as over traditional value-iteration (from exp(n) to poly(n)), where the optimality gap decays to 0 with rate O( 1 √ log n ). Remark 4.9. If k = O(log n), SUBSAMPLE-MFQ handles |E| ≤ O(log n/ log log n) types of local agents, since the run-time of the learning algorithm becomes poly(n). This surpasses the previous heterogeneity capacity from Mondal et al. [2022], which only handles constant |E| ≤ O(1). Increasing the agent heterogeneity using this type formulation does make the algorithm somewhat more expensive since it factors into the query complexity via the state space of the local agents; however, since the optimality gap of the learned policy is on the order of Õ(1/ √ k) (modulo very small log |S l ||A l | factors), increasing the amount of agent heterogeneity does not degrade the quality of the learned policy as the theoretical bound does not get worse. Moreover, recent methods from the graphon mean-field MARL community might be able to enable stronger heterogeneity in the system [Cui and Koeppl, 2022, Anand and Liaw, 2025, Hu et al., 2023].
In the non-tabular setting with infinite state/action spaces, one could replace the Q-learning algorithm with any arbitrary value-based RL method that learns Qk with function approximation [Sutton et al., 1999] such as deep Q-networks [Silver et al., 2016]. Doing so raises an additional error that factors into Theorem 4.4. We formalize this below.
Assumption 4.10 (Linear MDP with infinite state spaces). Suppose S g and S l are infinite compact sets. Furthermore, suppose there exists a feature map ϕ : S × A → R d and d unknown (signed) measures µ = (µ 1 , . . . , µ d ) over S and a vector θ ∈ R d such that for any (s, a) ∈ S × A, we have P(•|s, a) = ⟨ϕ(s, a), µ(•)⟩ and r(s, a) = ⟨ϕ(s, a), θ⟩.
The existence of ϕ : S × A → R d implies one can estimate the Q-function of any policy as a linear function. This assumption is commonly used in policy iteration algorithms Lattimore et al. [2020], Wang et al. [2023], and allows one to obtain sample complexity bounds that are independent of |S l | and |A l |. Finally, as is standard in RL, we assume bounded feature-norms [Tkachuk et al., 2023]: Assumption 4.11 (Bounded features). We assume that ∥ϕ(s, a)∥ 2 ≤ 1 for all (s, a) ∈ S × A.
Then, through a reduction from Zhang et al. [2024], Ren et al. [2024] that uses function approximation to learn the spectral features ϕ k for Qk , we derive a performance guarantee for the learned policy π est k , where the optimality gap decays with k. Theorem 4.12. When π est k is derived from the spectral features ϕ k learned in Qk , and M is the number of samples used in the function approximation, then
Pr V π * (s) -V π est k (s) ≤ Õ 1 √ k + ∥ϕ k ∥ 5 log 2k 2 √ M + 2γ r • ∥ϕ k ∥ (1 -γ) √ k ≥ 1 + 200 k - 201 200 √ k
We defer the proof of Theorem 4.12 to Appendix J.
this section cite: ['b30', 'b31', 'b49', 'b14', 'b1', 'b26', 'b63', 'b61', 'b38', 'b67', 'b64', 'b71', 'b56']

Section: Conclusion and Future Works
This work develops subsampling for mean field MARL in a cooperative system with a global decisionmaking agent and n homogeneous local agents. We propose SUBSAMPLE-MFQ which learns each agent's best response to the mean effect from a sample of its neighbors, allowing an exponential reduction on the sample complexity of approximating a solution to the MDP. We provide a theoretical analysis on the optimality gap of the learned policy, showing that (with high probability) the learned policy converges to the optimal policy with the number of agents k sampled at the rate Õ(1/ √ k), and validate our theoretical results through numerical experiments. We show that the decay rate is maintained, on expectation, when the reward functions are stochastic and when agents learn from a single trajectory on historical data through off-policy Q-learning. Finally, we extend this result to the non-tabular setting with infinite state and action spaces under assumptions of a linear MDP model.
Limitations and future work. Our current work assumes that the global and local agents cooperate to optimize a structured reward under a specific dynamic model. While this model is more general than the federated learning setting, one direction would be to extend our algorithms and analysis to weaker network assumptions. We believe our framework, which can handle dense subgraphs, as well as expander-graph decompositions [Reingold, 2008, Anand and Umans, 2023, Anand, 2025] may be amenable for this. Secondly, our current work incorporates mild heterogeneity among agents and assumes they are cooperative; thus, another avenue would be to consider settings with competitive agents or more complex agent heterogeneity. Finally, it would be exciting to generalize this work to the online no-regret setting.
Societal Impacts. This work is theoretical and foundational in nature. As such, while it enables more scalable multi-agent algorithms, it is not tied to any specific applications or deployments.
this section cite: ['b55', 'b3', 'b0']

Section: References
Ref_id:b0 Title: Towards the pseudorandomness of expander random walks for read-once acc0 circuits Year: (2025)
Ref_id:b1 Title: Feel-good thompson sampling for contextual bandits: a markov chain monte carlo showdown Year: (2025)
Ref_id:b2 Title: Efficient reinforcement learning for global decision making in the presence of local agents at scale. arXiV, 2024 Year: ()
Ref_id:b3 Title: Pseudorandomness of the Sticky Random Walk Year: (2023)
Ref_id:b4 Title: The Bit Complexity of Dynamic Algebraic Formulas and Their Determinants Year: (2024)
Ref_id:b5 Title: The structural complexity of matrix-vector multiplication Year: (2025)
Ref_id:b6 Title: Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales Year: (1922)
Ref_id:b7 Title: A Survey of Computational Complexity Results in Systems and Control Year: (1996)
Ref_id:b8 Title: Peer-to-peer learning dynamics of wide neural networks Year: (2024)
Ref_id:b9 Title: Sample complexity of policy-based methods under off-policy sampling and linear function approximation Year: (2022)
Ref_id:b10 Title: Sample complexity of policy-based methods under offpolicy sampling and linear function approximation Year: (2022-03)
Ref_id:b11 Title: Finite-sample analysis of off-policy td-learning via generalized bellman operators Year: (2021)
Ref_id:b12 Title: A lyapunov theory for finite-sample guarantees of asynchronous q-learning and td-learning variants Year: (2021)
Ref_id:b13 Title: Concentration of contractive stochastic approximation: Additive and multiplicative noise Year: (2025)
Ref_id:b14 Title: Learning Graphon Mean Field Games and Approximate Nash Equilibria Year: (2022)
Ref_id:b15 Title: Multi-Agent Reinforcement Learning via Mean Field Control: Common Noise, Major Agents and Approximation Properties Year: (2023)
Ref_id:b16 Title: Locally interdependent multi-agent MDP: Theoretical framework for decentralized agents with dynamic dependencies Year: (2024)
Ref_id:b17 Title: Discovering and removing exogenous state variables and rewards for reinforcement learning Year: (2018)
Ref_id:b18 Title: Asymptotic Minimax Character of the Sample Distribution Function and of the Classical Multinomial Estimator Year: (1956)
Ref_id:b19 Title: On the Complexity of Adversarial Decision Making Year: (2022)
Ref_id:b20 Title: Off-policy deep reinforcement learning without exploration Year: (2019)
Ref_id:b21 Title: Correlation Decay in Random Decision Networks Year: (2009)
Ref_id:b22 Title: Fundamentals of Nonparametric Bayesian Inference: Space of Probability Densities Year: (2017)
Ref_id:b23 Title: The role of inherent bellman error in offline reinforcement learning with linear function approximation Year: (2024)
Ref_id:b24 Title: Mean-Field Controls with Q-Learning for Cooperative MARL: Convergence and Complexity Analysis Year: (2021)
Ref_id:b25 Title: Mean-Field Multi-Agent Reinforcement Learning: A Decentralized Network Approach Year: (2022)
Ref_id:b26 Title: Graphon Mean-Field Control for Cooperative Multi-Agent Reinforcement Learning Year: (2023)
Ref_id:b27 Title: Provably efficient reinforcement learning with linear function approximation Year: (2020-07)
Ref_id:b28 Title: Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms Year: (2021)
Ref_id:b29 Title: Real-time bidding with multi-agent reinforcement learning in display advertising Year: (2018)
Ref_id:b30 Title: Truncated variance reduced value iteration Year: (2024)
Ref_id:b31 Title: Approximately Optimal Approximate Reinforcement Learning Year: (2002)
Ref_id:b32 Title: An Online Convex Optimization Approach to Realtime Energy Pricing for Demand Response Year: (2017)
Ref_id:b33 Title: Deep Reinforcement Learning for Autonomous Driving: A Survey Year: (2022)
Ref_id:b34 Title: A multiple-choice secretary algorithm with applications to online auctions Year: (2005)
Ref_id:b35 Title: Reinforcement Learning in Robotics: A Survey Year: (2013)
Ref_id:b36 Title: Derandomizing multi-distribution learning Year: (2024)
Ref_id:b37 Title:  Year: (2007-03)
Ref_id:b38 Title: Learning with good feature representations in bandits and in RL with a generative model Year: (2020-07)
Ref_id:b39 Title: An algorithm for distributed reinforcement learning in cooperative multi-agent systems Year: (2000)
Ref_id:b40 Title: Sample Complexity of Asynchronous Q-Learning: Sharper Analysis and Variance Reduction Year: (2022)
Ref_id:b41 Title: Efficient ridesharing order dispatching with mean field multi-agent reinforcement learning Year: (2019)
Ref_id:b42 Title: Distributed Reinforcement Learning in Multi-Agent Networked Systems Year: (2020)
Ref_id:b43 Title: Online Adaptive Policy Selection in Time-Varying Systems: No-Regret via Contractive Perturbations Year: ()
Ref_id:b44 Title: Learning-augmented Control via Online Adaptive Policy Selection: No Regret via contractive Perturbations Year: ()
Ref_id:b45 Title: Online policy optimization in unknown nonlinear systems Year: (2024-07-03)
Ref_id:b46 Title: Markov Games as a Framework for Multi-Agent Reinforcement Learning Year: (1994)
Ref_id:b47 Title: The Tight Constant in the Dvoretzky-Kiefer-Wolfowitz Inequality. The Annals of Probability Year: (1990)
Ref_id:b48 Title: Cooperative Multi-Agent Reinforcement Learning: Asynchronous Communication and Linear Function Approximation Year: (2023-07)
Ref_id:b49 Title: On the Approximation of Cooperative Heterogeneous Multi-Agent Reinforcement Learning (MARL) Using Mean Field Control (MFC) Year: (2022-01)
Ref_id:b50 Title: On the Tight Constant in the Multivariate Dvoretzky-Kiefer-Wolfowitz Inequality Year: (2021)
Ref_id:b51 Title: The Complexity of Optimal Queuing Network Control Year: (1999)
Ref_id:b52 Title: Crazyswarm: A large nano-quadcopter swarm Year: (2017)
Ref_id:b53 Title: Scalable Multi-Agent Reinforcement Learning for Networked Systems with Average Reward Year: (2020)
Ref_id:b54 Title: Scalable Reinforcement Learning of Localized Policies for Multi-Agent Networked Systems Year: (2020-06)
Ref_id:b55 Title: Undirected Connectivity in Log-Space Year: (2008-09)
Ref_id:b56 Title: Scalable spectral representations for network multiagent control Year: (2024)
Ref_id:b57 Title: Bayes-ball: The rational pastime (for determining irrelevance and requisite information in belief networks and influence diagrams Year: (2013)
Ref_id:b58 Title: A value for n-person games Year: (1953)
Ref_id:b59 Title: Near-optimal time and sample complexities for solving discounted markov decision process with a generative model Year: (2018)
Ref_id:b60 Title: Variance reduced value iteration and faster algorithms for solving markov decision processes Year: (2018)
Ref_id:b61 Title: Mastering the Game of Go with Deep Neural Networks and Tree Search Year: (2016-01)
Ref_id:b62 Title: Decentralized mean field games Year: (2022)
Ref_id:b63 Title: Policy gradient methods for reinforcement learning with function approximation Year: (1999)
Ref_id:b64 Title: Efficient planning in combinatorial action spaces with applications to cooperative multi-agent reinforcement learning. arXiV Year: (2023)
Ref_id:b65 Title: Introduction to Nonparametric Estimation Year: (2008)
Ref_id:b66 Title: Variance-reduced q-learning is minimax optimal Year: (2019)
Ref_id:b67 Title: More centralized training, still decentralized execution: Multi-agent conditional policy factorization Year: (2023)
Ref_id:b68 Title: Q-learning Year: (1992-05)
Ref_id:b69 Title: Decentralized randomly distributed multi-agent multi-armed bandit with heterogeneous rewards Year: (2023)
Ref_id:b70 Title: Mean Field Multi-Agent Reinforcement Learning Year: (2018-07)
Ref_id:b71 Title: Provable representation with efficient planning for partially observable reinforcement learning Year: (2024)
Ref_id:b72 Title: Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms Year: (2021)
Ref_id:b73 Title: Is centralized training with decentralized execution framework centralized enough for marl? Year: (2023)
