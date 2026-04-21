Title: Provably Efficient RL under Episode-Wise Safety in Constrained MDPs with Linear Function Approximation
Abstract: We study the reinforcement learning (RL) problem in a constrained Markov decision process (CMDP), where an agent explores the environment to maximize the expected cumulative reward while satisfying a single constraint on the expected total utility value in every episode. While this problem is well understood in the tabular setting, theoretical results for function approximation remain scarce. This paper closes the gap by proposing an RL algorithm for linear CMDPs that achieves O( √ K) regret with an episode-wise zero-violation guarantee. Furthermore, our method is computationally efficient, scaling polynomially with problem-dependent parameters while remaining independent of the state space size. Our results significantly improve upon recent linear CMDP algorithms, which either violate the constraint or incur exponential computational costs.

Section: Introduction
Safe decision-making is essential in real-world applications such as plant control and finance [19]. Constrained Markov decision process (CMDP) is a mathematical framework for developing decisionmaking algorithms with formal safety guarantees [2]. This paper studies the reinforcement learning (RL) problem in finite-horizon CMDPs, where an agent explores the environment to maximize the expected cumulative rewards while satisfying a single constraint on the expected total utility value.
Safe exploration has been established in the tabular CMDP settings. Several works [27,8,47] achieve episode-wise zero-violation RL with O( √ K) regret for K number of episodes, ensuring constraint satisfaction in every episode. Their approach consists of two phases: deploying a known strictly safe policy π sf and then updating policies via linear programs (LPs), which optimizes an optimistic objective while satisfying a pessimistic constraint. Deploying π sf is necessary to ensure feasible solutions for the LPs once enough environmental information is collected.
While safe exploration is well-established in tabular CMDPs, extending it to large-scale CMDPs remains a major challenge. LP-based methods are impractical at scale due to their state-dependent computational cost. 2 As a result, even in linear CMDPs, where value functions have linear structure, episode-wise safe RL has not been achieved. The state-of-the-art linear CMDP algorithm [18], which achieves the O( √ K) violation regret, 3 incurs an exponential computational cost of K H , where H is the horizon. Several studies achieve safe RL under instantaneous constraints [4,36], 4 a special subclass of the episode-wise safety that can be overly conservative (e.g., in drone control, temporary high energy consumption is tolerable, but full battery depletion is not). Table 1 summarizes representative algorithms, with additional literature in Appendix B. In short, a fundamental open question remains:
Can we develop a computationally efficient 5 linear CMDP algorithm with sublinear regret and zero episode-wise constraint violation?
Contributions. We propose Optimistic-Pessimistic Softmax Exploration for Linear CMDP (OPSE-LCMDP), the first algorithm for linear CMDPs that achieves O( √ K)-regret and episode-wise safety. Our approach builds on the optimistic-pessimistic exploration framework with two key innovations for large-scale state-space problems: (i) a new deployment rule for π sf , and (ii) a computationally efficient method to implement optimism for the objective and pessimism for the constraint within the softmax policy framework [16,18]. Section 2 first analyzes the linear constrained bandit problem as a "warm-up" for linear CMDPs (H = 1 with an expected instantaneous constraint), highlighting the key role of the π sf deployment rule in avoiding linear regret. Previous instantaneous constraint literature limits π sf deployments by representing the safe action as a vector a sf ∈ R d [32,33,22,3,4]. However, extending this approach to episode-wise safety is non-trivial, as the constraint is imposed on policies rather than actions, and policies may be nonlinear functions (e.g., softmax mapping from value functions) rather than single vectors. We overcome this challenge by showing that if π sf is deployed only when the agent is less confident in π sf 's safety, the number of deployments is logarithmically bounded (Theorem 1). Section 3 then extends the bandit result to RL in CMDPs. To enable optimistic-pessimistic exploration in linear CMDPs, OPSE-LCMDP employs the composite softmax policy (Definition 3), which adjusts optimism and pessimism by controlling a variable λ. OPSE-LCMDP efficiently searches for the best λ through bisection search, achieving a polynomial computational cost in problem parameters, independent of state-space cardinality (Remark 2). Overall, our techniques-the novel π sf deployment rule and softmax-based optimistic-pessimistic exploration-achieve the first episode-wise safe RL with sublinear regret and computational efficiency in linear CMDPs.
this section cite: ['b18', 'b1', 'b26', 'b7', 'b46', 'b17', 'b3', 'b35', 'b15', 'b17', 'b31', 'b32', 'b21', 'b2', 'b3']

Section: Mathematical notations.
The set of probability distributions over a set S is denoted by P(S). For integers a ≤ b, let a, b := {a, . . . , b}, and a, b := ∅ if a > b. For x ∈ R N , its n-th element is x n or x(n). The clipping function clip{x, a, b} returns x ′ with x ′ i = min{max{x i , a}, b} for each i. We define 0 := (0, . . . , 0) ⊤ and 1 := (1, . . . , 1) ⊤ , with dimensions inferred from the context. For a positive definite matrix A ∈ R d×d and x ∈ R d , we denote ∥x∥ A = √
x ⊤ Ax. For positive sequences {a n } and {b n } with n = 1, 2, . . ., we write a n = O (b n ) if there exist C > 0 and N ∈ N 2 While some works (e.g., 30) proposed LP methods for unconstrained linear MDPs, they remain unsuitable for our exploration setting or still incur state-dependent computational costs (see Appendix B). 3 Violation regret denotes the total amount of constraint violation during exploration. 4 Inst. constraint requires u h (s
(k) h , a(k)
h ) ≥ b ∀h, k ∈ 1, H × 1, K (see Section 3 for notations). 5 An algorithm is comp. efficient if its cost is polynomial with problem parameters, excluding the state space.
such that a n ≤ Cb n for all n ≥ N , and a n = Ω(b n ) for the reverse inequality. We use O(•) and Ω(•) to further hide the polylogarithmic factors. Finally, for x ∈ R d , we denote its softmax distribution as SoftMax(x) ∈ P( 1, d ) with its i-th component SoftMax(x) i = exp(x i )/( i exp(x i )).
this section cite: []

Section: Warm Up: Safe Exploration in Linear Constrained Bandit
To better illustrate the core ideas of our CMDP algorithm, this section introduces a contextual linear bandit variant based on Pacchiano et al. [32]. All the proofs in this section are provided in Appendix D. Let A ⊂ R d be the action space, a compact set of bounded d-dimensional vectors. Without loss of generality, we assume ∥a∥ 2 ≤ 1 for any a ∈ A. At each round k, the agent selects a policy π (k) ∈ P(A), samples an action a (k) ∼ π (k) , and observes the reward r (k) = θ ⊤ r a (k) + ε (k) r and utility
u (k) = θ ⊤ u a (k) +ε (k) u .
Here, θ r , θ u ∈ R d are vectors unknown to the agent such that ∥θ r ∥ 2 , ∥θ u ∥ 2 ≤ B, and ε (k) r , ε (k) u are R-sub-Gaussian random noises. For any policy π and g ∈ {r, u}, let g π := E a∼π [⟨θ g , a⟩]. We consider a constraint such that the expected utility must be above the threshold b ∈ R. Formally, let Π sf := {π | u π ≥ b} denote the set of safe policies. The agent's goal is to achieve sublinear regret while satisfying the expected instantaneous constraints defined as follows:
Regret(K) := K k=1 r π ⋆ -r π (k) = o(K) such that π (k) ∈ Π sf ∀k ∈ 1, K ,(1)
where π ⋆ ∈ arg max π∈Π sf r π . A sublinear regret exploration is efficient, as its averaged reward approaches the optimal value, i.e., lim K→∞ 1 K r π (K) → r π ⋆ . Finally, we assume access to a strictly safe policy in Π sf , as deploying arbitrary policies without this assumption risks violating constraints 6 . Assumption 1 (Safe policy). We have access to π sf ∈ Π sf and ξ > 0 such that u π sf -b ≥ ξ .
this section cite: ['b31']

Section: Technical Challenge: Zero-Violation with a Safe Policy
The key to efficient and safe exploration is the optimistic-pessimistic exploration, which constructs an optimistic reward r (k) π ≥ r π and a pessimistic utility u (k) π ≤ u π , and then computes a policy by:
max π∈P(A) r (k) π such that u (k) π ≥ b .(2)
Here, r
π and u (k)
π are designed to quickly approach r π and u π as data accumulates for efficient exploration [1]. However, although Equation (2) can have feasible solutions when u π ≈ u π , the pessimistic constraint may not have any feasible solution in the early stages of exploration.
To ensure that (2) always has a solution, a common bandit approach assumes access to a safe action a sf ∈ A such that θ ⊤ u a sf ≥ b + ξ, and then ensures the feasibility of (2) by leveraging the vector representation of a sf ∈ R d . For example, [32,33,3] designed u (k) π using the orthogonal direction a sf ⊥ := a sfa sf / a sf 2 , while [22] assume a sf = 0 ∈ A with a negative constraint threshold b < 0. Both approaches ensure that a policy playing a sf with probability 1 is always feasible in (2). However, extending this safe action technique to episode-wise safe RL is non-trivial, as the episodewise constraint is imposed on policies rather than actions, and policies in linear CMDPs may be nonlinear functions (e.g., softmax mappings from value functions) rather than single vectors. To address this challenge, we first develop a safe bandit algorithm without relying on safe action techniques.
this section cite: ['b0', 'b31', 'b32', 'b2', 'b21']

Section: Algorithm and Analysis
We summarize the proposed Optimistic-Pessimistic Linear Bandit with Safe Policy (OPLB-SP) in Algorithm 1, which follows the standard linear bandit framework (see Abbasi-Yadkori et al. [1]). Throughout this section, we analyze Algorithm 1 under the parameters listed in its Input line.
Let θ (k) r := (Λ (k) ) -1 k-1 i=1 a (i) r (i) and θ (k) u := (Λ (k) ) -1 k-1 i=1 a (i) u (i) denote the regularized least-squares estimates of θ r and θ u , respectively, where Λ (k) := ρI
+ k-1 i=1 a (i) (a (i) ) ⊤ . Let r (k) π := E a∼π [a ⊤ θ (k) r ] and u (k) π := E a∼π [a ⊤ θ (k)
u ] be the estimated reward and utility functions.
this section cite: ['b0']

Section: Algorithm 1: Optimistic-Pessimistic Linear Bandit with Safe Policy
Input: Regression coefficient ρ = 1, bonus scalers Cu = B + R √ d ln 4Kδ -1 and Cr = Cu(1 + 2Bξ -1 ), safe policy π sf , and iteration length K ∈ N 1 for k = 1, . . . , K do 2 Let β (k) π , r (k) π , and u (k) π be bonus, estimated reward and utility, respectively (see Section 2.2)
3 if Cuβ (k) π sf > ξ 2 then π (k) := π sf / * Deploy π sf if π sf is unconfident * / 4 else π (k) ∈ arg max π∈P(A) r (k) π + Crβ (k) π such that u (k) π -Cuβ (k) π ≥ b 5
Sample an action a (k) ∼ π (k) and observe reward r (k) and utility u (k) .
Using the bonus function
β (k) π := E a∼π ∥a∥ (Λ (k) ) -1
, with the well-established elliptical confidence bound argument for linear bandits [1], the following confidence bounds hold: Lemma 1 (Confidence bounds). For any π and k, with probability (w.p.) at least
1 -δ, r π + 2C u β (k) π ≥ r (k) π + C u β (k) π ≥ r π and u π ≥ u (k) π -C u β (k) π ≥ u π -2C u β (k) π .
Based on Lemma 1, Algorithm 1 solves the following optimistic-pessimistic (Opt-Pes) problem. The optimistic objective promotes efficient exploration, while the pessimism enforces the constraint satisfaction:
Opt-Pes (Line 4) π (k) ∈ arg max π∈P(A) r (k) π + C r β (k) π such that u (k) π -C u β (k) π ≥ b . (3)
This is a convex optimization problem when the set A satisfies certain structural assumptions, such as being discrete or ellipsoidal (see, e.g., Section 19.3 of Lattimore and Szepesvari [26]). To emphasize our approach to the technical challenge in Section 2.1, this section omits the computational details of (3) and focuses instead on the core technique for efficient exploration under episode-wise safety.
this section cite: ['b0', 'b25']

Section: Zero-Violation and Logarithmic Number of π sf Deployments
Since π (k) is either π sf or the solution to Opt-Pes (if feasible), all deployed policies in Algorithm 1 satisfy the constraint with high probability due to the pessimistic constraint. However, as noted in Section 2.1, the pessimistic constraint may render Opt-Pes infeasible, requiring Line 4 to wait until the bonus β (k) π shrinks sufficiently. Yet, waiting too long overuses the suboptimal π sf , leading to poor regret. Thus, exploration must keep the number of iterations where Equation ( 2) is infeasible bounded.
The core technique of Algorithm 1 lies in the π sf deployment trigger based on the confidence of π sf . Specifically, we solve the optimistic-pessimistic optimization whenever β (k) π sf ≤ ξ 2Cu ; otherwise, we correct the data by deploying π sf (see Line 3). Under this trigger, the following Theorem 1 ensures that the number of π sf deployments grows logarithmically with the iteration length K. Definition 1 (π sf unconfident iterations). Let U be the set of iterations when Algorithm 1 is unconfident in π sf , i.e., U := {k ∈ 1,
K | β (k) π sf > ξ/(2C u )}. Let U ∁ := 1, K \ U be its complement. Theorem 1 (Logarithmic |U| bound). It holds w.p. at least 1 -δ that |U| ≤ O dC 2 u ξ -2 ln Kδ -1 .
The proof utilizes the well-known elliptical potential lemma [1]. Intuitively, it ensures that the confidence bounds shrink on average, thereby limiting the number of iterations where the algorithm remains unconfident in π sf . He et al. [21], Zhang et al. [50] employed a similar technique in linear bandits to ensure the suboptimality of policies after sufficient iterations. Moreover, combined with Lemma 1, the following Lemma 2 ensures that, after logarithmic iterations, policies around π sf will become feasible solutions to Opt-Pes and Line 4.
this section cite: ['b0', 'b20', 'b49']

Section: Lemma 2 (Mixture policy feasibility).
Consider k ∈ U ∁ . Let α (k) := ξ-2Cuβ (k) π sf ξ-2Cuβ (k) π sf +2Cuβ (k) π ⋆ . For any α ∈ 0, α (k) , the mixture policy π α := (1 -α)π sf + απ ⋆ satisfies u πα -2C u β (k) πα ≥ b.
Note that the mixture policy π α is introduced only to ensure the feasibility of Opt-Pes; it does not need to be computed in the algorithm.
Finally, Lemma 1 and Lemma 2 directly imply the following zero-violation guarantee: Corollary 1 (Zero-violation). W.p. at least 1 -δ, Algorithm 1 satisfies π (k) ∈ Π sf for any k.
this section cite: []

Section: Regret Analysis
The remaining task is to ensure sublinear regret. By Theorem 1 and Lemma 1, the regret is decomposed as:
Regret(K) ≤ O dBC 2 u ξ -2 + 3C r k∈U ∁ β (k) π 1 + k∈U ∁ r π ⋆ -r (k) π (k) -C r β (k) π 2 .
Using the elliptical potential lemma [1], we can bound 1 ≤ O(C r √ dK).
For the term 2 , when there is no constraint in Opt-Pes, the common strategy is bounding 2 using r π ⋆ -r (k)
π (k) -C r β (k)
π ≤ 0, leveraging the optimism due to Lemma 1 with the maximality of π (k) in Opt-Pes (see, e.g., Abbasi-Yadkori et al. [1]). However, due to the pessimistic constraint in Opt-Pes, π ⋆ may not be a solution to Opt-Pes, necessitating a modification to this approach.
Recall from Lemma 2 that, for k ∈ U ∁ , the mixture policy π α (k
) := (1 -α (k) )π sf + α (k) π ⋆ satisfies u π α (k) -2C u β (k) π α (k) ≥ b.
For this π α (k) , the following optimism with respect to π ⋆ holds:
Lemma 3 (π α (k) optimism). For any k ∈ U ∁ , it holds r π α (k) + (2BC u ξ -1 )β (k) π α (k) ≥ r π ⋆ .
Using Lemmas 1 and 3 with C r = C u (1 + 2Bξ -1 ), we have
2 ≤ k∈U ∁ r π α (k) + C r β (k) π α (k) -r (k) π (k) -C r β (k) π (k) ≤ 0 ,(4)
where the second inequality holds since π α (k) is a feasible solution to Opt-Pes and π (k) is its maximizer. This optimism via a mixture policy technique is adapted from tabular CMDPs [27,8] to the linear bandit setup. By combining all the results, Algorithm 1 archives the following guarantees:
Theorem 2. If OPLB-SP is run with the parameters listed in its Input line, w.p. at least 1 -δ, π (k) ∈ Π sf for any k ∈ 1, K and Regret(K) ≤ O(dBC 2 u ξ -2 + C r √ dK) .
When B = R = 1, the regret bound simplifies to O(
d 2 ξ -2 + ξ -1 √ d 3 K).
In summary, OPLB-SP relies on three components: (i) optimistic-pessimistic updates (Opt-Pes), (ii) a logarithmic number of π sf deployments (Theorem 1), and (iii) compensation for the pessimism (Lemma 3). Building on these components, the next section develops a linear CMDP algorithm.
this section cite: ['b0', 'b0', 'b26', 'b7']

Section: Safe Reinforcement Learning in Linear Constrained MDP
A finite-horizon CMDP is defined as a tuple (S, A, H, P, r, u, b, s 1 ), where S is the finite but potentially exponentially large state space, A is the finite action space (|A| = A),foot_1 H ∈ N is the episode horizon, b ∈ [0, H] is the constrained threshold, and s 1 is the fixed initial state. The reward and utility functions r, u : 1, H × S × A → [0, 1] specify the reward r h (s, a) and constraint utility u h (s, a) when taking action a at state s in step h. For a policy π, transition kernel P , reward function g : 1, H × S × A → R, and entropy coefficient κ ≥ 0, let Q π,g P,h [κ] : S × A → R and V π,g P,h [κ] : S → R denote the entropy-regularized value functions at step h satisfying:
Q π,g P,h [κ] = g h + (P h V π,g h+1,P [κ]), V π,g P,h [κ] = π h (Q π,g P,h [κ] -κ ln π h )
, and V π,g H+1,P [κ] = 0 . For κ = 0, we omit κ, e.g., Q π,g P,h := Q π,g P,h [0]. We denote h κ := h(1 + κ ln A) for h ∈ 1, H . For h ∈ 1, H , let w π P,h ∈ ∆(S × A) denote the occupancy measure of π in P at step h such that
w π P,h (s, a) = P(s h = s, a h = a | π, P ) ∀ (h, s, a) ∈ 1, H × S × A ,(5)
where the expectation is taken over all possible trajectories, in which
a h ∼ π h (• | s h ) and s h+1 ∼ P h (• | s h , a h ).
With a slight abuse of notation, we write w π P,h (s) = a∈A w π P,h (s, a).
this section cite: []

Section: Learning Setup.
An agent interacts with the CMDP for K episodes using policies π (1) , . . . , π (K) ∈ Π. Each episode k starts from s 1 . At step h in episode k, the agent observes a state s
(k) h , selects an action a (k) h ∼ π (k) h (• | s (k) h
), and transitions to s (k)
h+1 ∼ P h (• | s (k) h , a (k) h
). The algorithm lacks knowledge of the transition kernel P , while r and u are known for simplicity. Extending our setting to unknown stochastic reward and utility is straightforward (see, e.g., Efroni et al. [13]).
To handle a potentially large state space, we consider the following linear MDP assumption: Assumption 2 (Linear MDP). We have a known feature map ϕ : S × A → R d satisfying: there exist unknown d (signed) measures µ h := (µ 1 h , . . . ,
µ d h ) ∈ R S×d such that P h (s ′ | s, a) = µ h (s ′ ) ⊤ ϕ(s, a), and known vectors θ r h , θ u h ∈ R d such that r h (s, a) = (θ r h ) ⊤ ϕ(s, a) and u h (s, a) = (θ u h ) ⊤ ϕ(s, a). We assume sup s,a ∥ϕ(s, a)∥ 2 ≤ 1 and V ⊤ µ h 2 ≤ √ d for any V ∈ R S such that ∥V ∥ ∞ ≤ 1.
Let π ⋆ ∈ arg max π∈Π sf V π,r P,1 (s 1 ) be the optimal policy, where Π sf := {π | V π,u P,1 (s 1 ) ≥ b} is the set of safe policies. The goal is to achieve sublinear regret under episode-wise constraints:
Regret(K) := K k=1 V π ⋆ ,r P,1 (s 1 ) -V π (k) ,r P,1 (s 1 ) = o(K) such that π (k) ∈ Π sf ∀k ∈ [K] . (6
)
Finally, we assume the strictly safe policy similar to Section 2.
Assumption 3 (Safe policy). We have access to π sf ∈ Π sf and ξ > 0 such that V π sf ,u P,1 (s 1 ) -b ≥ ξ .
this section cite: ['b12']

Section: Technical Challenge: Optimistic-Pessimistic Optimization in Linear CMDP
Our linear CMDP algorithm builds on OPLB-SP in Section 2: deploying an optimistic-pessimistic policy when confident in π sf ; otherwise, it uses π sf . We will logarithmically bound the number of π sf deployments, similar to Theorem 1, and ensure optimism through a linear mixture of policies, as in Lemma 2. However, computing an optimistic-pessimistic policy in the linear CMDP setting, similar to Opt-Pes, presents a non-trivial challenge. This section outlines the difficulties.
Following standard linear MDP algorithm frameworks (e.g., Jin et al. [23], Lykouris et al. [28]), for each h, k, let β (k) h : (s, a) → ∥ϕ(s, a)∥ (Λ (k) h ) -1 be the bonus, where
Λ (k) h := ρI + k-1 i=1 ϕ(s (i) h , a (i) h )ϕ(s (i) h , a(i)
h ) ⊤ and ρ > 0. For any V : S → R, let P
h V be the next-step value estimation defined as: ( P
(k) h V )(s, a) := ϕ(s, a) ⊤ (Λ (k) h ) -1 k-1 i=1 ϕ(s (i) h , a (i) h )V (s (i)h+1
). We construct the following optimistic and pessimistic value functions for reward and utility, respectively: Definition 2 (Clipped value functions). Let C r , C u , C † , B † > 0. For each k, h, π, and κ ≥ 0, define
Q π,r (k),h [κ], Q π, † (k),h , Q π,u (k),h : S × A → R and V π,r (k),h [κ], V π, † (k),h , V π,u (k),h : S → R such that: Q π,r (k),h [κ] := r h + clip{C r β (k) h + P (k) h V π,r (k),h+1 [κ], 0, H κ -h κ }, V π,r (k),h [κ] := π h (Q π,r (k),h [κ] -κ ln π h ) , Q π, † (k),h := B † β (k) h + clip{C † β (k) h + P (k) h V π, † (k),h+1 , 0, B † (H -h)}, V π, † (k),h := π h Q π, † (k),h , Q π,u (k),h := u h + clip{-C u β (k) h + P (k) h V π,u (k),h+1 , 0, H -h}, and V π,u (k),h := π h Q π,u (k),h . We set V π,r (k),H+1 [κ] = V π, † (k),H+1 = V π,u (k),H+1 = 0. For κ = 0, omit κ, e.g., Q π,r (k),h := Q π,r (k),h [0].
Algorithm 2: Optimistic-Pessimistic Softmax Exploration for Linear CMDP Input: Regr. coeff. ρ = 1, bonus scalers Cr = O(dH), Cu = O(dH), C † = O(d 2 H 3 ξ -1 ), B † = O dH 2 ξ -1 , entropy coeff. κ = Ω ξ 3 H -4 d -1 K -0.5 , search length T = O(H), λ-threshold C λ = O dH 4 ξ -2 , safe policy π sf , and iter. length K ∈ N 1 for k = 1, . . . , K do 2 Let V π,u (k),h be value function (Definition 2) and π (k),λ be softmax policy (Definition 3) / * π sf trigger is implicitly tied to π sf confidence (Lemma 5) * /
3 if V π (k),C λ ,u (k),1 (s1) < b then Set π (k) := π sf 4 else if V π (k),0 ,u (k),1(s1)
≥ b then Set π (k) := π (k),0 5 else / * Do bisection-search to find safe π (k),λ with small λ * / 6 Set λ (k,1) := 0 and λ (k,1) := C λ . Let λ (k,t) := (λ (k,t) + λ (k,t) )/2 7 for t = 1, . . . , T do 8 if V π (k),λ (k,t)
,u (k),1 (s1) ≥ b then λ (k,t+1) := λ (k,t) and λ (k,t+1) := λ (k,t) 9 else λ (k,t+1) := λ (k,t) and λ (k,t+1) := λ (k,t)
10 Set π (k) := π (k),λ (k,T ) 11
Sample a trajectory (s
(k) 1 , a (k) 1 , . . . , s (k) H , a(k)
H ) by deploying π (k)
We will utilize Q π, † (k),h and V π, † (k),h to compensate for the pessimism, similar to the bandit proof in (4). 8 Entropy regularization in Q π,r (k),h [κ] is for the later analysis. The clipping operators are essential to avoid the propagation of unreasonable value estimates [48].
Using these value functions, one might consider extending Opt-Pes to linear CMDPs by solving:
max π∈Π V π,r (k),1 (s 1 ) + V π, † (k),1 (s 1 ) such that V π,u (k),1 (s 1 ) ≥ b .(7)
However, solving this (7) is challenging due to (i) the large state space in the linear CMDP setting (|S| ≫ 1) and (ii) the clipping operators in Q π,r (k),h , Q π, † (k),h , and Q π,u (k),h . In tabular CMDPs with small |S|, Liu et al. [27] and Bura et al. [8] used linear programming (LP) to solve similar optimistic-pessimistic optimization problems, achieving zero violation. However, the computational cost of LP scales with |S|, making it impractical for linear CMDPs.
Another option is the Lagrangian method, which reformulates the constrained optimization as a min-max optimization:
min λ≥0 max π∈Π V π,r (k),1 (s 1 ) + V π, † (k),1 (s 1 ) + λ(V π,u (k),1 (s 1 ) -b).
When the value functions are exact, i.e.,
V π, † (k),h + V π,r (k),h + V π,u (k),h = V π,r+B † β (k) +λu P,h
, this min-max is equivalent to (7), and the inner maximization reduces to a standard policy optimization [2]. Both favorable properties arise due to the linearity of the value function in the occupancy measure (see, e.g., Paternain et al. [34]). However, due to clipping, the value functions in Definition 2 may not be representable via occupancy measures, making the Lagrangian approach inapplicable.
To address this large-scale optimization challenge, instead of directly solving (7), we realize optimism and pessimism through a novel adaptation of the recent softmax policy technique for linear CMDPs [18,16], combined with the π sf deployment technique from Section 2.
this section cite: ['b22', 'b27', 'b47', 'b26', 'b7', 'b6', 'b1', 'b33', 'b17', 'b15']

Section: Algorithm and Analysis
We summarize the proposed OPSE-LCMDP in Algorithm 2 and analyze it under the parameters in its Input line. All formal theorems and proofs in this section are in Appendix E. A key component of our algorithm is the composite softmax policy, which balances optimism and pessimism via λ ≥ 0: Definition 3 (Composite softmax policy). For λ ≥ 0, κ > 0, let π (k),λ ∈ Π be a policy such that
π (k),λ h (• | s) = SoftMax 1 κ Q π (k),λ , † (k),h (s, •) + Q π (k),λ ,r (k),h [κ](s, •) + λQ π (k),λ ,u (k),h (s, •) .
π (k),λ can be computed iteratively in a backward manner for h = H, . . . , 1. For this π (k),λ , using the Lipschitz continuity of SoftMax(•) (see Ghosh et al. [16]), the following confidence bounds hold: Lemma 4 (Confidence bounds). For any (k, h), λ ∈ [0, C λ ], π ∈ {π (k),λ , π sf }, w.p. at least 1 -δ,
V π,r P,h ≤ V π,r (k),h ≤ V π,r+2Cr β (k) P,h , V π,B † β (k) P,h ≤ V π, † (k),h ≤ V π,(B † +2C † )β (k) P,h , V π,u-2Cuβ (k) P,h ≤ V π,u (k),h ≤ V π,u P,h .
Using Lemma 4, analogous to Section 2.2.1, we next establish the zero-violation guarantee.
this section cite: ['b15']

Section: Zero-Violation and Logarithmic Number of π sf Deployments
In the softmax policy (Definition 3), λ balances optimism and pessimism: a small λ promotes exploration, while a large λ prioritizes constraint satisfaction. Building on this, Algorithm 2 conducts a bisection search to find the smallest feasible λ while ensuring the pessimistic constraint holds (Line 4 to Line 10). If a large λ = C λ fails to satisfy the constraint, the algorithm assumes no feasible pessimistic policy exists and deploys π sf (Line 3). Since the softmax policy is only deployed for λ satisfying V π (k),λ ,u (k),1 (s 1 ) ≥ b, Lemma 4 implies the following zero-violation guarantees:
Corollary 2 (Zero-violation). W.p. at least 1 -δ, Algorithm 2 satisfies π (k) ∈ Π sf for any k.
Next, we bound the number of π sf deployments to achieve sublinear regret. To this end, similar to the bandit warm-up (Section 2), we relate π sf deployment to π sf uncertainty level and logarithmically bound the number of uncertain iterations. The following Lemma 5 ensures that, if Algorithm 2 is confident in π sf and runs with appropriate C λ and κ, then π sf is not deployed. Definition 4 (π sf unconfident iterations). Let U be the iterations when Algorithm 2 is unconfident in π sf , i.e., U := {k ∈ 1,
K | V π sf ,β (k) P,1 (s 1 ) > ξ 4Cu }. Let U ∁ := 1, K \ U be its complement. Lemma 5 (Implicit π sf deployment trigger). When C λ ≥ 8H 2 κ (B † +1) ξ
and κ ≤ ξ 2 32H 2 κ (B † +1) , then w.p. at least 1 -δ, it holds that V π (k),C λ ,u (k),1 (s 1 ) ≥ b for all k ∈ U ∁ .
Essentially, the proof of Lemma 5 relies on the following monotonic property of the value function for the softmax policy: if the value estimation is exact, increasing λ monotonically improves safety. Lemma 6 (Softmax value monotonicity). For λ ≥ 0, let π λ be a softmax policy such that
π λ h (• | s) = SoftMax( 1 κ (Q π,r P,h [κ](s, •) + λQ π,u P,h (s, •))). Then, V π λ ,u P,1 (s 1 ) is monotonically increasing in λ.
While the true value function enjoys this monotonicity, the estimated value V π (k),λ ,u (k),1 (s 1 ) may not, as P
h V can take negative values even when V is positive. This complicates the proof of Lemma 5. To address this, we leverage Lemma 4, which sandwiches the estimated values by some true values. We prove Lemma 5 by showing that, for sufficiently large C λ , any sandwiched value satisfies the constraint under pessimism, implying that the estimated value also satisfies it. This novel result enables bisection search to adjust λ, making OPSE-LCMDP more computationally efficient than Ghosh et al. [18]. The detailed proofs of Lemmas 5 and 6 are provided in Appendix E.4.1.
Finally, the following theorem ensures that the number of π sf deployment scales logarithmic to K, as in Theorem 1. The proof follows from extending the bandit's proof of Theorem 1 to CMDPs.
Theorem 3 (Logarithmic |U| bound). It holds w.p. at least 1 -δ that |U| ≤ O d 3 H 4 ξ -2 ln KHδ -1 .
this section cite: ['b17']

Section: Regret Analysis
The remaining task is to ensure sublinear regret. By Theorem 3 and Lemma 4, the regret is decomposed as:
Regret(K) ≤ O d 3 H 4 ξ 2 + k∈U ∁ V π (k) ,2Cr β (k) P,1(s1)
1
+ k∈U ∁ V π ⋆ ,r P,1 (s1) -V π (k) ,r (k),1 [κ](s1)2
+ κKH ln A ,
where the last term arises from the entropy regularization (V π,r P,1 (s 1 )[κ] -V π,r P,1 (s 1 ) ≤ κH ln A). Using the elliptical potential lemma for linear MDPs [23], we obtain 1 ≤ O(C r H √ dK).
We now bound 2 . Note that for any k ∈ U ∁ , due to Lemma 5, π (k) is the softmax policy by Line 10. To bound 2 , following a similar approach to Lemma 3 in the bandit, we replace π ⋆ with a mixture policy that satisfies the pessimistic constraint. To this end, we utilize the following lemmas. Definition 5 (Mixture policy). For α ∈ [0, 1], let π α be a mixture policy such that, for any h, w π α P,h = (1 -α)w π sf P,h + αw π ⋆ P,h . Such a π α is ensured to exists for any α ∈ [0, 1] [7]. Lemma 7 (Safe and optimistic mixture policy). Let α (k) := ξ ξ+2V π ⋆ ,2Cu β (k)
P,1 (s1) . If B † ≥ 4CuH ξ , then for any k ∈ U ∁ , it holds (i) V π α (k) ,u-2Cuβ (k) P,1 (s 1 ) ≥ b and (ii) V π α (k) ,r+B † β (k) P,1 (s 1 ) ≥ V π ⋆ ,r P,1 (s 1 ).
We note that the mixture policy π α (k) is introduced only for the regret analysis; it is not required in the
actual algorithm. Since λ (k,T ) is chosen to satisfy V π (k) ,u (k),1 (s 1 ) < b and b ≤ V π α (k) ,u-2Cuβ (k) P,1 (s 1 ) holds by Lemma 7, 2 ≤ k∈U ∁ V π α (k) ,B † β (k) P,1 (s1) + V π α (k) ,r P,1 [κ](s1) + λ (k,T ) V π α (k) ,u-2Cuβ (k) P,1 (s1) -V π (k) , † (k),1 (s1) -V π (k) ,r (k),1 [κ](s1) -λ (k,T ) V π (k) ,u (k),1 (s1)     3
+ k∈U ∁ V π (k) , † (k),1 (s1) 4 + C λ k∈U ∁ V π (k),λ (k,T ) ,u (k),1 (s1) -V π (k),λ (k,T ) ,u (k),1 (s1) 5 .
Using Lemma 4, similar to 1 , we have 4 ≤ O (B † + C † )H √ dK . The term 5 is controlled by the bisection search width (λ (k,T ) -λ (k,T ) ) and the following sensitivity of V π (k),λ ,u
(k),1 (s 1 ) to λ.
Lemma 8. For any k and λ ∈ [0, C λ ], we have V π (k),λ ,u
(k),1 (s 1 ) -V π (k),λ+ε ,u (k),1 (s 1 ) ≤ O (KH) H ε
Ghosh et al. [18] also derived a similar exponential bound (see their Appendix C). Due to the update rule of the bisection search, setting the search iteration to T = O(H) ensures that 5 ≤ O(1).
For 3 , using a modification of the so-called value-difference lemma [40], we have
3 = k∈U ∁ V π α (k) ,f 1 P,1 (s 1 ) -V π α (k) ,f 2 P,1 (s 1 ) -λ (k,T ) V π α (k) ,2Cuβ (k) P,1 (s 1 ) ,(8)
where f 1 , f 2 : 1, H × S × A → R are functions such that, for any h,
f 1 h = π α (k) h -π (k) h Q π (k) , † (k),h +Q π (k) ,r (k),h [κ]+λ (k) Q π (k) ,u (k),h -κπ α (k) h ln π α (k) h +κπ (k) h ln π (k) h and f 2 h = Q π (k) ,r (k),h [κ]-r h -P h V π (k) ,r (k),h+1 [κ] +λ (k,T ) u h +P h V π (k) ,u (k),h+1 -Q π (k) ,u (k),h + Q π (k) , † (k),h -B † β (k) -P h V π (k) , † (k),h+1
.
Our use of the softmax policy with entropy regularization is crucial for bounding 3 . Since the analytical maximizer of the regularized optimization
max π∈P(A) a∈A π(a)(x(a) -κ ln π(a)) is given by SoftMax 1 κ x(•) , it follows that f 1 is non-positive, implying V π α (k) ,f 1 P,1 (s 1 ) ≤ 0.
Additionally, applying Lemma 4, we derive
f 2 h ≥ -λ (k,T ) 2C u β (k) h , which leads to -V π α (k) ,f 2 P,1 (s 1 )- λ (k,T ) V π α (k)
,2Cuβ (k) P,1 (s 1 ) ≤ 0. By substituting these bounds into Equation (8), we obtain 3 ≤ 0.
By combining all the results, Algorithm 2 achieves the following guarantees: Theorem 4. If OPSE-LCMDP is run with the parameters listed in its Input line, w.p. at least 1 -δ,
π (k) ∈ Π sf ∀k ∈ 1, K and Regret(K) ≤ O(H 2 √ d 3 K) (i) + O(d 3 H 4 ξ -2 ) (ii) + O(H 4 ξ -1 √ d 5 K) (iii) .
Notably, Theorem 4 is the first linear CMDP result achieving zero episode-wise constraint violations and sublinear regret. We conclude this section by discussing the regret bound quality and computational cost of OPSE-LCMDP.
this section cite: ['b22', 'b17', 'b39']

Section: Remark 1 (Can we do better?).
Without the constraint-i.e., removing (ii) and (iii)-our bound matches the O(H 2 √ d 3 K) regret of the fundamental LSVI-UCB algorithm [23]. The ξ -2 term in (ii) is unavoidable [32]. The ξ -1 dependence in (iii) remains unresolved, yet appears in all the existing safe RL literature [27,8,47,36,3,32,22]. These observations suggest that the bound is tight in ξ -1 and K. As for d and H, the term (iii) introduces an extra dH 2 factor over (i). Similar deterioration has been observed in tabular CMDPs [27,42] and partially mitigated via Bernstein-type bonus analysis [47]. While improvement may be possible, it is unclear whether our √ d 5 H 8 dependence is overly loose when compared with existing CMDP regret bounds (e.g.,
√
d 3 H 4 by Ghosh et al. [18]), since none of the existing results achieve episode-wise safe exploration. In general, regret or sample complexity bounds under different safety requirements are not directly comparable, even if the problem settings appear similar. For example, Vaswani et al. [42] shows that the sample complexity lower bound for tabular CMDPs exhibits a worse dependence on the horizon under a strict safe policy requirement than when small violations are allowed. Establishing a formal regret lower bound for our setting would be necessary to assess the tightness of our result, but this is beyond the scope of the current paper.
this section cite: ['b22', 'b31', 'b26', 'b7', 'b46', 'b35', 'b2', 'b31', 'b21', 'b26', 'b41', 'b46', 'b17', 'b41']

Section: Remark 2 (Computational cost). Algorithm 2 requires up to T value evaluations (Definition 2) and policy computation (Definition 3).
Using the bisection search, we bound T = O(H), reducing the computational cost per-iteration to O(H × [value & policy comp.]). As this cost scales polynomially with A, H, and d [28], OPSE-LCMDP runs in polynomial time-an improvement over recent Ghosh et al. [18], which achieves O( √ K) violation regret but incurs an exponential K H cost.
this section cite: ['b27', 'b17']

Section: Conclusion
This paper proposed OPSE-LCMDP, the first RL algorithm achieving both sublinear regret and episode-wise constraint satisfaction in linear CMDPs (Theorem 4). Our approach builds on optimisticpessimistic exploration with two key innovations: (i) a novel deployment rule for π sf and (ii) a softmax-based approach for efficiently implementing optimistic-pessimistic policies in linear CMDPs.
this section cite: []

Section: References
Ref_id:b0 Title: Improved Algorithms for Linear Stochastic Bandits Year: (2011)
Ref_id:b1 Title: Constrained Markov Decision Processes Year: (1999)
Ref_id:b2 Title: Linear Stochastic Bandits under Safety Constraints Year: (2019)
Ref_id:b3 Title: Safe Reinforcement Learning with Linear Function Approximation Year: (2021)
Ref_id:b4 Title: Logistic Q-Learning Year: (2021)
Ref_id:b5 Title:  Year: (1997)
Ref_id:b6 Title: A Convex Analytic Approach to Markov Decision Processes. Probability Theory and Related Fields Year: (1988)
Ref_id:b7 Title: DOPE: Doubly Optimistic and Pessimistic Exploration for Safe Reinforcement Learning Year: (2022)
Ref_id:b8 Title: Unifying PAC and Regret: Uniform PAC Bounds for Episodic Reinforcement Learning Year: (2017)
Ref_id:b9 Title: Provably Efficient Safe Exploration via Primal-Dual Policy Optimization Year: (2021)
Ref_id:b10 Title: Last-Iterate Convergent Policy Gradient Primal-Dual Methods for Constrained MDPs Year: (2024)
Ref_id:b11 Title: On Log-Sum Inequalities Year: (2024)
Ref_id:b12 Title: Exploration-Exploitation in Constrained MDPs Year: (2020)
Ref_id:b13 Title: Offline Primal-Dual Reinforcement Learning for Linear Mdps Year: (2024)
Ref_id:b14 Title: A Theory of Regularized Markov Decision Processes Year: (2019)
Ref_id:b15 Title: Provably Efficient Model-Free Constrained RL with Linear Function Approximation Year: (2022)
Ref_id:b16 Title: Achieving Sub-linear Regret in Infinite Horizon Average Reward Constrained MDP with Linear Function Approximation Year: (2023)
Ref_id:b17 Title: Towards Achieving Sub-linear Regret and Hard Constraint Violation in Model-free RL Year: (2024)
Ref_id:b18 Title: A Review of Safe Reinforcement Learning: Methods, Theory and Applications Year: (2022)
Ref_id:b19 Title: Learning with Safety Constraints: Sample Complexity of Reinforcement Learning for Constrained MDPs Year: (2021)
Ref_id:b20 Title: Uniform-PAC Bounds for Reinforcement Learning with Linear Function Approximation Year: (2021)
Ref_id:b21 Title: Directional Optimism for Safe Linear Bandits Year: (2024)
Ref_id:b22 Title: Provably Efficient Reinforcement Learning with Linear Function Approximation Year: (2020)
Ref_id:b23 Title: A Policy Gradient Primal-Dual Algorithm for Constrained MDPs with Uniform PAC Guarantees Year: (2024)
Ref_id:b24 Title: A Linearly Relaxed Approximate Linear Program for Markov Decision Processes Year: (2017)
Ref_id:b25 Title: Bandit Algorithms Year: (2020)
Ref_id:b26 Title: Learning Policies with Zero or Bounded Constraint Violation for Constrained MDPs Year: (2021)
Ref_id:b27 Title: Corruption-Robust Exploration in Episodic Reinforcement Learning Year: (2021)
Ref_id:b28 Title: Truly No-Regret Learning in Constrained MDPs Year: (2024)
Ref_id:b29 Title: Efficient Global Planning in Large MDPs via Stochastic Primal-Dual Optimization Year: (2023)
Ref_id:b30 Title: A Unifying View of Optimism in Episodic Reinforcement Learning Year: (2020)
Ref_id:b31 Title: Stochastic Bandits with Linear Constraints Year: (2021)
Ref_id:b32 Title: Contextual Bandits with Stage-wise Constraints Year: (2024)
Ref_id:b33 Title: Constrained Reinforcement Learning Has Zero Duality Gap Year: (2019)
Ref_id:b34 Title: Markov Decision Processes: Discrete Stochastic Dynamic Programming Year: (1994)
Ref_id:b35 Title: Provably Efficient RL for Linear MDPs under Instantaneous Safety Constraints in Non-Convex Feature Spaces Year: (2025)
Ref_id:b36 Title: Near-Optimal Regret Bounds for Stochastic Shortest Path Year: (2020)
Ref_id:b37 Title: f -divergence Inequalities Year: (2016)
Ref_id:b38 Title: Understanding Machine Learning: From Theory to Algorithms Year: (2014)
Ref_id:b39 Title: Optimistic Policy Optimization with Bandit Feedback Year: (2020)
Ref_id:b40 Title: A Near-Optimal Algorithm for Safe Reinforcement Learning under Instantaneous Hard Constraints Year: (2023)
Ref_id:b41 Title: Near-Optimal Sample Complexity Bounds for Constrained MDPs Year: (2022)
Ref_id:b42 Title: Introduction to the Non-Asymptotic Analysis of Random Matrices Year: (2010)
Ref_id:b43 Title: A Provably-Efficient Model-Free Algorithm for Constrained Markov Decision Processes Year: (2021)
Ref_id:b44 Title: A Provably-Efficient Model-Free Algorithm for Infinite-Horizon Average-Reward Constrained Markov Decision Processes Year: (2022)
Ref_id:b45 Title: A Dual Approach to Constrained Markov Decision Processes with Entropy Regularization Year: (2022)
Ref_id:b46 Title: Improved Regret Bound for Safe Reinforcement Learning via Tighter Cost Pessimism and Reward Optimism Year: (2024)
Ref_id:b47 Title: Frequentist Regret Bounds for Randomized Least-Squares Value Iteration Year: (2020)
Ref_id:b48 Title: Finite-Time Complexity of Online Primal-Dual Natural Actor-Critic Algorithm for Constrained Markov Decision Processes Year: (2022)
Ref_id:b49 Title: On the Interplay Between Misspecification and Sub-Optimality Gap in Linear Contextual Bandits Year: (2023)
