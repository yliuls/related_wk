Title: To Distill or Decide? The Algorithmic Trade-off in Partially Observable Reinforcement Learning
Abstract: Partial observability is a notorious challenge in reinforcement learning (RL), due to the need to learn complex, history-dependent policies. Recent empirical successes have used privileged expert distillation -which leverages availability of latent state information during training (e.g., from a simulator) to learn and imitate the optimal latent, Markovian policy -to disentangle the task of "learning to see" from "learning to act" [56,12,9]. While expert distillation is more computationally efficient than RL without latent state information, it also has well-documented failure modes. In this paper -through a simple but instructive theoretical model called the perturbed Block MDP, and controlled experiments on challenging simulated locomotion tasks -we investigate the algorithmic trade-off between privileged expert distillation and standard RL without privileged information. Our main findings are: (1) The trade-off empirically hinges on the stochasticity of the latent dynamics, as theoretically predicted by contrasting approximate decodability with belief contraction in the perturbed Block MDP; and (2) The optimal latent policy is not always the best latent policy to distill. Our results suggest new guidelines for effectively exploiting privileged information, potentially advancing the efficiency of policy learning across many practical partially observable domains.

Section: Introduction
Partial observability is a common challenge in applied reinforcement learning: the decision-making agent may not see the true state of the environment at all time-steps, whose information might only be probabilistically inferred from the history of observations. An illustrative task is robot learning for robots with image-based perception [58,68]. A single image of the robot (or, in first-person perspective, of the environment) will not capture important elements of the state such as the robot's velocity, and may miss other features due to e.g. occlusion or limited view.
The canonical theoretical model for such tasks is Partially Observable Markov Decision Process (POMDP). Unfortunately, there are well-documented computational [57] and statistical [28] barriers to planning and learning in POMDPs, which have motivated many theoretical works that seek to bypass these barriers by making additional structural assumptions [28,33,19,24,23,42]. On the empirical side, the standard technique for mitigating partial observability is frame-stacking, which enabled notable successes for learning to play Atari games [50,51]. The idea is to treat the "state" of the environment as the concatenation of a short window of L recent observations, and apply a standard algorithm for fully-observed reinforcement learning (RL). This technique inspired theoretical developments such as L-step decodability [19], and has some theoretical underpinnings for γ-observable POMDPs [24]. Yet frame-stacking is not a silver bullet for partially observable decision-making: sometimes effective planning requires long memory [18]. Also, high-dimensional observations (such as stacks of images) can confound learning complex behaviors [58,78].
this section cite: ['b57', 'b67', 'b56', 'b27', 'b27', 'b32', 'b18', 'b23', 'b22', 'b41', 'b49', 'b50', 'b18', 'b23', 'b17', 'b57', 'b77']

Section: Learning from latent state information.
A common heuristic for planning in known POMDPs is to use the optimal latent policy (also known as the state-based policy or privileged policy)i.e., the optimal policy that is allowed to "cheat" and see the underlying state of the environment -as a starting point for computing an executable policy -i.e. a policy that only depends on the observable history [40,65,12]. More recent works have brought this ansatz to the learning task, where the description of the POMDP is a priori unknown. In the standard theoretical formalization of this task [31], the latent states of the POMDP are never observed (nor even identifiable); however, for applications such as robotics, it is often practically reasonable to construct a simulator of the environment [13], from which the learning agent may draw trajectories that include both the observations as well as the latent states -"privileged" information that is only available at training time, not at test time.
The most prominent paradigm for exploiting this additional information is called privileged expert distillation, 1 which applies methods from imitation learning and structured prediction [15,62,64,8] to learning in POMDPs. Expert distillation has two steps: (1) learn an optimal latent policy, using a standard RL algorithm with the latent state information provided by the simulator; and (2) distill the latent policy to an executable policy, using an imitation learning algorithm such as DAgger [64]. This paradigm has achieved impressive success in applications such as autonomous driving [9], robotics [37,48,85] and LLMs [11].
These successes suggest a fundamental question: when does expert distillation help in realistic decision-making tasks? On the one hand, in controlled experiments, expert distillation uniformly converges faster and more stably than RL without latent state information [53], likely because it disentangles representation learning from decision-making [9]. Moreover, expert distillation enjoys a provable computational advantage in decodable POMDPsfoot_1 [7]. On the other hand, there are welldocumented failure modes of expert distillation -most notably, due to its inability to encourage purely information-gathering actions [2,79] -where more expensive hybrid methods such as Asymmetric Actor-Critic [58] are fundamentally required.
In this paper, motivated by image-based locomotion tasks, we focus on the middle ground where (perfect) decodability may fail, yet the observations are still highly informative of the latent state. In this regime, we ask: (i) when and how is expert distillation as performant as standard RL with frame-stacking, and (ii) are there lightweight improvements to expert distillation? We use simple theoretical models in tandem with controlled experiments to address the preceding questions.
this section cite: ['b39', 'b64', 'b11', 'b30', 'b12', 'b14', 'b61', 'b63', 'b7', 'b63', 'b8', 'b36', 'b47', 'b84', 'b10', 'b52', 'b8', 'b6', 'b1', 'b78', 'b57']

Section: Our contributions.
1. The prior theoretical model for understanding the benefits of latent state information was a perfectly decodable POMDP [7]. We begin by empirically demonstrating that this model is too restrictive for image-based locomotion tasks. 2. We then introduce approximate decodability, and connect it to the success of expert distillation -in analogy with the connection between belief contraction and the success of standard reinforcement learning with frame-stacking. But when are these conditions satisfied? As a theoretical testbed, we introduce the perturbed Block MDP. 3. We show both theoretically (by analyzing the perturbed Block MDP model) and experimentally that the performance of expert distillation compared to standard reinforcement learning depends crucially on the stochasticity of the model dynamics: for deterministic dynamics, distillation is competitive with RL, but as the stochasticity increases, its performance comparatively degrades. 4. Finally, we show that distillation of the optimal latent policy is often a sub-optimal use of latent state information: the simple modification of adding stochasticity to the latent MDP before computing the optimal policy yields robust performance benefits via improved smoothness.
this section cite: ['b6']

Section: Preliminaries
A (finite-horizon, layered) Partially Observable Markov Decision Process (POMDP) is a tuple P = (H, X , S, A, P, O, R), where H ∈ N is the horizon, X = {X h } H h=1 is the observa-tion space, S = {S h } H h=1 is the latent state space, A = {A h } H h=1 is the action space, P = {P h : S h-1 × A h-1 → ∆(S h )} H h=1 describes the latent transitions, O = {O h : S h → ∆(X h )} H h=1 describes the emission distributions, and R = {R h : S h × A h → [0, 1]} describes the rewards. We write A := max h |A h |, S := max h |S h |, and X := max h |X h |. Given any timestep h and L ∈ [H], we denote X h-L:h := X h-L × X h-L+1 × • • • × X h , and similarly for A h-L:h , with the shorthand h -L := max{1, h -L}. Then an L-step executable policy is a collection π = {π h : X h-L+1:h × A h-L:h-1 → ∆(A h )}; we let Π L denote the class of such policies. Given any executable policy π ∈ Π := Π H , a trajectory τ = (s 1 , x 1 , a 1 , r 1 , . . . , s H , x H , a H , r H ) is generated by s h ∼ P h (s h-1 , a h-1 ), x h ∼ O h (s h ), a h ∼ π(x 1:h , a 1:h-1 ), r h = R h (s h , a h ). We use P π and E π to denote the law and expectation under this process. Following convention, we assume H h=1 r h ≤ 1 almost surely under all policies. The value of a policy π is J(π) := E π H h=1 r h .
Note that the POMDP P also defines an underlying Markov Decision Process (MDP) M = {S, A, P, R, H} (which we call the latent MDP) where the state is fully observable. A latent (Markovian) policy is a collection π latent = {π latent h : S h → ∆(A h )}, and we let Π latent denote the class of latent policies. A latent trajectory τ latent = (s 1 , a 1 , . . . , s H , a H ) is generated by s h ∼ P h (s h-1 , a h-1 ), a h ∼ π latent h (s h ), and we define P π latent and E π latent accordingly.
Learning with/without latent state information. In the standard theoretical RL access model (i.e. without latent state information) [30,27], at training time, the learning agent can repeatedly interact with the POMDP P by playing an executable policy π and observing the partial trajectory (x 1:H , a 1:H , r 1:H ). In contrast, in the learning with latent state information model [7], at training time, the learning agent can play any policy, and observes the full trajectory (s 1:H , x 1:H , a 1:H , r 1:H ).
In both settings, the goal is to eventually produce an executable policy π that minimizes J(π ⋆ ) -J( π) (where π ⋆ is the optimal executable policy).
this section cite: ['b29', 'b26', 'b6']

Section: Belief states.
A belief state is a distribution over latent states. For a prior b on the latent state at step h -1, let U h (b; a h-1 , x h ) be the posterior on the latent state at step h after taking action a h-1 and then observing x h (see Definition B.1 for the formal algebraic definition).
Definition 2.1. For any observation/action sequence (x 1:h , a 1:h-1 ), the true belief state b h (x 1:h , a 1:h-1 ) is defined as follows. For h = 1 with observation x 1 , let b 1 (x 1 ) := B 1 (P 1 ; x 1 ). For any 2 ≤ h ≤ H, let b h (x 1:h , a 1:h-1 ) := U h (b h-1 (x 1:h-1 , a 1:h-2 ); a h-1 , x h ).
For any executable policy π, step h, and history (x 1:h , a 1:h-1 ), b h (x 1:h , a 1:h-1 ) is the distribution of the latent state s h under P π , conditioned on (x 1:h , a 1:h-1 ) (Lemma C.2).
Many methods for efficient planning in POMDPs are based on approximate belief states that only depend on a short window of recent actions and observations [29,24]. Informally, the approximate belief state b apx h (x h-L+1:h , a h-L:h-1 ; D) is the posterior on state s h after observing (x h-L+1:h , a h-L:h-1 ) with prior D on state s h-L . See Definition B.2 for the formal definition (analogous to Definition 2.1).
this section cite: ['b28', 'b23']

Section: Additional notation. For distributions b, b
′ ∈ ∆(S h ), the density ratio is ∥b/b ′ ∥ ∞ = sup s∈S h b(s)/b ′ (s) ∈ [1, ∞], with the convention that 0/0 = 1. For a belief state b ∈ ∆(S h ) and conditional distribution π h : S h → ∆(A h ), we let π h • b denote the distribution over A h obtained as (π h • b)(a h ) := s h ∈S h b(s h )π h (a h | s h ).(2)
Experimental Setup. We use three tasks in the Deepmind control suite [73]: walker-run, dog-walk and the challenging humanoid-walk. To implement online (resp., offline) expert distillation, we (1) train an expert on the latent state information using MrQ [22], and (2) imitate the expert via DAgger [64] (resp., Behavior Cloning (BC)) on L-step executable policies. Unless otherwise specified, we use the standard choice of L = 3, and we use mean squared error (MSE) as the loss function: give input X = {x i } N i=1 and target
Y = {y i ∈ R d } N i=1 , the loss of a function f is ℓ(f, X, Y ) = 1 N d N i=1 d j=1 (f (x i ) j -y i j ) 2 .
To implement reinforcement learning (RL), we use MrQ [22] on L-step executable policies. In experiments, we follow the common empirical practice of only stacking observations (rather than both observations and actions).
this section cite: ['b72', 'b21', 'b63', 'b21']

Section: Approximate Decodability and Belief Contraction
Even with access to latent state information during training, the problem of learning a near-optimal policy in a POMDP is as hard as the planning task (where a description of the POMDP is already known), which is well-known to be computationally intractable in the worst case [57]. However, POMDPs encountered in practice will often satisfy additional structural properties that may mitigate this hardness. Some of the most widely-studied properties are decodability [19,7] and belief contraction (also known as filter stability) [29,24].
Privileged information is known to yield a provable computational benefit in decodable POMDPs [7]. However, as we empirically demonstrate in Section 3.1, perfect decodability is an unrealistic assumption in our motivating tasks. For this reason, in Section 3.2 we introduce the notion of approximate decodability. Heuristically, this property governs the success of expert distillation with L-step framestacking, whereas belief contraction governs the success of standard RL (also with L-step framestacking). But when are these properties satisfied? As a clean theoretical testbed for studying this question, in Section 3.3 we introduce the δ-perturbed Block MDP.
this section cite: ['b56', 'b18', 'b6', 'b28', 'b23', 'b6']

Section: Prior Work: Perfectly Decodable POMDPs
In some applications, such as video games, it is plausible that the agent can deduce the latent state from a small number of recent observations. This was empirically substantiated by the success of DQN [50] and its variants, which only use the most recent four observations as policy inputs. Theoretically, this motivated the study of the L-step decodable model [19], which posits that the most recent L observations and actions suffice to fully disambiguate the latent state (Definition B.3).
Without latent state information (i.e. in the standard RL access model), learning a near-optimal policy in an L-step decodable POMDPs requires Ω(A L ) samples [19]. However, with latent state information, [7] show that the sample and time complexity of learning a near-optimal policy π ∈ Π M such that J(π) ≥ arg max π∈Π M J(π) -ε with high probability is only poly(S, A, X, H, 1/ε). Thus, for large L, there is a clear theoretical benefit of latent state information (both statistically and computationally). However, unfortunately, L-step decodability is not always a realistic assumption: Empirical test: does perfect decodability hold? Through controlled experiments on our three chosen locomotion tasks (Section 2), we observe that latent states are not perfectly decodable in practice, especially in early timesteps. We defer details of this experiment to Appendix H.1.
this section cite: ['b49', 'b18', 'b18', 'b6']

Section: Errors in POMDPs
The above empirical result motivates the following theoretical definition of decodability error: Definition 3.1 (Decodability Error). Fix a POMDP P. The decodability error for an executable policy π and timestep h ∈ [H] is
ε decode h (π) := E π [1 -∥b h (x 1:h , a 1:h-1 )∥ ∞ ].
Intuitively, decodability error quantifies stochasticity of the true belief. Below, we show that it upper bounds the misspecification of any latent policy π latent with respect to the class of executable policies.
this section cite: []

Section: Lemma 3.1 (See Lemma E.3).
Let π latent ∈ Π latent be a latent policy and let b 1:H be a collection of functions b h : X h × A h-1 → ∆(S h ). Define executable policies π, π by π(x 1:h , a 1:h-1 ) := π latent • b h (x 1:h , a 1:h-1 ) and π(x 1:h , a 1:h-1 ) := π latent • b h (x 1:h , a 1:h-1 ) (see Eq. ( 2)). Then
TV(P π latent , P π ) ≤ H h=1 2ε decode h (π) + E π b h (x 1:h , a 1:h-1 ) -b h (x 1:h , a 1:h-1 ) 1 .(3)
Low decodability error is not strictly required for low misspecification (see Section 6), but some such assumption is needed to rule out models requiring active information-gathering [79]. As a special case, Lemma 3.1 implies that π latent is 2 H h=1 ε decode h (π)-close to the executable policy π, which evaluates π latent at a random state s ′ h sampled from the true belief b h (x 1:h , a 1:h-1 ); this is because if b h (x 1:h , a 1:h-1 ) is highly concentrated, then s ′ h likely matches the true latent state. The second error term in Eq. ( 3) quantifies error in learning the true belief -e.g., due to using only L-step histories.
Next, it is instructive to contrast decodability error with belief contraction error, the discrepancy between the true belief and the approximate belief induced by the L most recent observations/actions: Definition 3.2 (Belief Contraction Error [24]). Fix a POMDP P. For an executable policy π, and timestep h ∈ [H], the L-step belief contraction error (L ∈
[h -1]) is ε contract h (π; L) := E π ∥b h (x 1:h , a 1:h-1 ) -b apx h (x h-L+1:h , a h-L:h-1 ; unif(S h-L ))∥ 1 .
In the absence of latent state information, bounding the belief contraction error is the standard method of analyzing provably efficient algorithms for RL in POMDPs [29,75,24]. Indeed, belief contraction implies that the POMDP with L-step frame-stacking is approximately Markovian, which heuristically suggests that a standard RL algorithm [30,5] with frame-stacking should achieve low error in time ≈ (AX) O(L) . Due to technical issues with error compounding, this is not formally true, but under an additional observability condition, there is an algorithm that provably achieves that guarantee: Theorem 3.1 (Informal; see Theorem B.1; due to [23]). Suppose the POMDP is γ-observable (Definition B.4), and satisfies L-step belief contraction with error ε. 3 There exists a reinforcement learning algorithm that achieves the sub-optimality bound
J(π ⋆ ) -J(π rl ) ≤ ε • poly(S, X, H, γ -1 ), in time (XA) O(L) • poly(H, S, γ -1 , ε -1 ).
Technically, the explicit result in [23] fixes L ∼ logfoot_3 (SH/ε)/γ (in which case the desired belief contraction bound is implied by γ-observability, but the algorithm requires quasi-polynomial time), but we observe that the proof extends to the result above -see Theorem B.1. Notably, Theorem 3.1 gives a polynomial-time algorithm if belief contraction holds for L = O(1).
this section cite: ['b78', 'b23', 'b28', 'b74', 'b23', 'b29', 'b4', 'b22', 'b22']

Section: The Perturbed Block MDP
Approximate decodability and belief contraction are conditions under which expert distillation and standard RL with frame-stacking, respectively, may be reasonably expected to succeed. But when are these conditions satisfied, and how do they compare? As a theoretical testbed, we introduce the perturbed Block MDP model. Block MDPs [16] are a well-studied abstraction of environments with rich observations yet simple latent dynamics. However, they assume that the latent state is fully determined by the current observation. Below, we generalize Block MDPs by allowing for δ probability that the observation is sampled from an arbitrary conditional distribution. 4  Definition 3.3. Fix a parameter δ > 0. A POMDP P is a δ-perturbed Block MDP if, for each h ∈ [H], there are
O h , E h : S h → ∆(X h ) such that O h : S h → ∆(X h )
satisfies the block property [17], i.e.
O h (• | s h ), O h (• | s ′ h )
have disjoint supports for all s h ̸ = s ′ h , and moreover the emission distribution O h at step h can be decomposed as follows:
O h (x h | s h ) = (1 -δ) O h (x h | s h ) + δE h (x h | s h ).
A simple example is the noisy sensor model where S = X and the true state is observed with probability at least 1 -δ. Later, we will examine the empirical validity of this model; for now we study its theoretical implications. Below, we prove that for any δ-perturbed Block MDP, the belief contraction error decays exponentially as the frame-stack increases, by a factor of O(δ) per frame. Theorem 3.2 (See Theorem D.1). Suppose that the POMDP P is a δ-perturbed Block MDP. There is a universal constant C D.1 > 1 with the following property. Fix an executable policy π, indices 1 ≤ h -L < h ≤ H, and a distribution D ∈ ∆(S h-L ). Then for any partial history (x 1:h-L , a 1:h-L-1 ),
E π [∥b h (x 1:h , a 1:h-1 ) -b apx h (x h-L+1:h , a h-L:h-1 ; D)∥ 1 ] ≤ (C D.1 δ) L/9 b h (x 1:h-L , a 1:h-L-1 ) D ∞
where the expectation is over trajectories drawn from policy π conditioned on the partial history (x 1:h-L , a 1:h-L-1 ). Thus, in particular, ε contract h (π; L) ≤ (C D.1 δ) L/9 S. While prior belief contraction results [24] apply to this model, they only yield contraction by 1 -(1 -2δ)/C per frame, for a large constant C > 1 (Remark D.1), and so are vacuous for L = o(log S), even in the regime δ ≪ 1 (i.e. low observation noise). Theorem 3.2 remedies this limitation; e.g. for δ = 1/S it yields ε contract h (π; L) ≤ O(1/S) with only L = O(1). To prove Theorem 3.2, one might hope that each new observation contracts the TV-distance by poly(δ) in expectation. This is false (Example D.1), but in such cases, it turns out that the density ratio decays, yielding a win-win argument.
Heuristically, Theorem 3.2 suggests that standard RL with L-step frame-stacking should progressively improve as L increases. Formally, Theorem 3.2 and Theorem 3.1 imply the following end-to-end learning guarantee for the RL algorithm of [23] (which does not use latent state information): Corollary 3.1 (Informal; see Corollary F.1). There is a method that, for any δ-perturbed Block MDP, learns a policy π with J(π ⋆ ) -J( π) ≤ (C 3.2 δ) L/9 (SXH) O(1) in time (XA/δ) O(L) (HS) O(1) .
From a theoretical view, it remains to understand the decodability error for the perturbed Block MDP. As we will show, this qualitatively depends on the stochasticity of the transition dynamics.
this section cite: ['b15', 'b16', 'b23', 'b22']

Section: Distillation is Competitive for Deterministic Dynamics
In some environments, it is reasonable to assume that the latent transition dynamics are deterministic (e.g., if the dynamics are governed by simple Newtonian mechanics). Simulation benchmarks with this property include some Atari games as well as MuJoCo tasks. In this section, we theoretically and empirically study the performance of expert distillation, versus standard RL with frame-stacking, in such environments (with deterministic latent transitions, but stochastic initial state and observations).
this section cite: []

Section: Theoretical Analysis under Deterministic Dynamics
Below, we show that for perturbed Block MDPs with deterministic dynamics, the decodability error decays exponentially as the step h ∈ [H] increases. Intuitively, each observation concentrates the true belief state further, and the deterministic transitions cannot "spread out" the belief state. While this intuition is not quite rigorous, it can be proven that most observations concentrate the belief state; the result follows from an appropriate martingale analysis (Lemma C.5).
this section cite: []

Section: Proposition 4.1 (See Proposition D.1).
There is a universal constant C 4.1 > 1 so that the following holds. Suppose that P is a δ-perturbed Block MDP with deterministic transitions. For any executable policy π and index h ∈ [H], it holds that ε decode h (π) ≤ min(δ, (C 4.1 δ) (h-1)/9 ).
From Lemma 3.1, the "ideal" distillation of a latent expert π latent is π imitation := π latent • b, i.e., given any history, query the latent expert based on the true belief. Combining Lemma 3.1 and Proposition 4.1 immediately yields a strong, horizon-independent guarantee for this policy: if π latent is the optimal latent policy, then
J(π ⋆ ) -J(π imitation ) ≤ J(π latent ) -J(π imitation ) ≤ 2 H h=1 min(δ, (C 4.1 δ) (h-1)/9 ) ≤ O(δ),
where the first inequality is by Lemma C.4. Of course, exactly learning the true belief state may be unrealistic, since this would require conditioning on the entire history. However, we can prove that (a bottom: DAgger) with respect to the horizon. We repeat 5 runs for each horizon and task, and perform linear regression on the results from each task. Note that the trajectory rewards for this plot have been normalized by horizon (and by action-prediction error), so linear scaling indicates compounding errors.
slight modification of) the Forward algorithm [62] (the non-stationary version of DAgger) on L-step executable policies learns the following approximation of π imitation ,foot_4 in the infinite-sample limit:
π Forward h (• | x h-L+1:h , a h-L:h-1 ) = π latent h • b apx h (x h-L+1:h , a h-L:h-1 ; d π Forward h-L ) if h > L π latent h • b h (x 1:h , a 1:h-1 )
otherwise
See Appendix E.2 for the algorithm and proof. Applying this derivation to Lemma 3.1, then using Proposition 4.1 to bound the decodability error and Theorem 3.2 to bound the error in approximate beliefs, gives the following guarantee for expert distillation under deterministic latent dynamics: Theorem 4.1 (See Theorem E.1). Suppose that the POMDP P is a δ-perturbed Block MDP with deterministic transitions, and fix L ∈ N. Let π latent ∈ Π latent be the optimal latent policy, and let π Forward be the policy computed by Forward with policy class Π L (i.e. all L-step executable policies) and expert π latent , in the infinite-sample limit. Then J(π ⋆ ) -J(π Forward ) ≤ J(π latent ) -J(π Forward ) ≤ TV(P π latent , P π ) ≤ O(δ) + (C D.1 δ) L/9 SH.
Comparison with RL. While Theorem 4.1 is presented in the infinite-sample limit, the effective sample complexity is only ≈ (XA) O(L) , since the optimization is over L-step executable policies. More concretely, up to additional error ε opt , the above guarantee can be achieved by the same algorithm with only poly((AX) L , H, ε -1 opt ) time and samples (Theorem E.2). Thus, the guarantee for Forward qualitatively matches the guarantee for RL (Corollary 3.1), aside from the additional horizon-independent term of O(δ) incurred above (due to poor decodability in initial steps).
this section cite: ['b61']

Section: Empirical Analysis under Deterministic Dynamics
Theorem 4.1 gives a strong performance guarantee for expert distillation under deterministic latent dynamics, nearly matching that of RL. This suggests that expert distillation may be preferred over standard RL due to its (practical) efficiency. Also, Theorem 4.1 suggests that error may compound with the horizon H. However, the result is only an upper bound, and only for a stylized setting. We now investigate whether these two theoretical implications hold up empirically.
Expert distillation outperforms RL under deterministic dynamics. In this experiment, we compare the (a) asymptotic performance and (b) computational efficiency of expert distillation and standard RL. We train each method until convergence, and we plot the episodic return with respect to the wall clock time in Figure 1. We see that offline expert distillation (i.e., behavior cloning) is competitive in easier tasks such as walker, but is suboptimal in harder tasks such as humanoid and dog. However, online imitation learning (i.e., DAgger) is able to achieve the best performance in all tasks, and with better computational efficiency (i.e., faster convergence) than RL. This supports our theory that under deterministic dynamics, expert distillation can be close to optimal.
Empirical vignette: the source of error compounding? The horizon dependence of the error in imitation learning has received intensive empirical [62,35,3] and theoretical [59,20,60] study, both from the perspective of sample complexity [59,20] and misspecification [60]. It is widely believed that behavior cloning suffers error compounding over the horizon, which is avoided by online methods such as DAgger that are able to recover from mistakes [63,59]. Does this compounding manifest in expert distillation for POMDPs, and is the cause sampling error or misspecification? In Figure 2, we vary the horizon H ∈ [50,450], and measure the sub-optimality of offline and online expert distillation. We normalize rewards so that trajectory reward lies in [0, 1]. We further normalize by mean action-prediction MSE (averaged over choice of H). We see strong horizon dependence for behavior cloning (and weaker for DAgger, likely due to recoverability). This contrasts with empirical results of [20]: they perform well-specified behavior cloning in similar tasks, and find little horizon dependence. Together, our results therefore suggest that misspecification, rather than sampling error, may be the more fundamental source of horizon dependence for behavior cloning.
this section cite: ['b61', 'b34', 'b2', 'b58', 'b19', 'b59', 'b58', 'b19', 'b59', 'b62', 'b58', 'b49', 'b19']

Section: RL Outperforms Distillation for Stochastic Dynamics
While deterministic dynamics are plausible in some applications, there are also many potential sources of stochasticity; in real-world robotics, stochasticity may be required to model e.g. internal motor noise or unknowable features of the external environment. Some robotics simulators [44] also have stochasticity arising from a PDE solver. How does the stochasticity of the environment affect the performance of expert distillation and RL?
this section cite: ['b43']

Section: Theoretical Analysis under Stochastic Dynamics
We show a negative result in the perturbed Block MDP model: for general dynamics, the misspecification of the optimal latent policy with respect to the class of L-step executable policies does not necessarily decay as L increases, in contrast with the case of deterministic dynamics (Lemma 3.1).
Proposition 5.1 (See Proposition D.2). Let δ > 0 and H ∈ N. There is a δ-perturbed Block MDP P with horizon H such that for all L ∈ [H], the optimal latent policy π latent satisfies the following bound, where Π L is the class of L-step executable policies:
min π∈Π L TV(P π latent , P π ) ≥ Ω(min(1, δH)).
This result also highlights the difference between decodability error and belief contraction error, which does decay as L increases, regardless of the transition dynamics (Theorem 3.2). The intuition for Proposition 5.1 is simple: in the extreme case where the dynamics are uniformly mixing at every step, prior observations yield no information about the current state, so the δ error incurred by trying to decode the current observation is irreducible. This decodability error compounds over timesteps, and means that executable policies are unable to simulate the latent policy that plays an action uniquely indexed by the latent state. In contrast, POMDPs with uniform mixing are easy for standard RL, precisely because they reduce to H independent horizon-1 subproblems.
Comparison with RL. The above result, compared with Corollary 3.1, suggests a potential empirical benefit of standard RL over expert distillation: the former may generically be able to trade increased computation (by increasing L) for improved performance (by mitigating observation noise), whereas the latter -at least in the worst case -incurs irreducible error due to stochasticity in the dynamics. To be sure, the uniformly-mixing construction from Proposition 5.1 is practically unrealistic; nevertheless, below we verify that this benefit occurs in more realistic environments.
this section cite: []

Section: Experimental Analysis under Stochastic Dynamics
RL with more computation eventually outperforms distillation. To simulate a POMDP with stochastic latent dynamics, we apply motor noise in the humanoid-walk task. We add 0-mean isotropic Gaussian noise with std-dev ∈ {0.1, 0.2, 0.3} to each action. We compare DAgger and RL with frame-stack L ∈ {2, 3, 4}. We run each method until convergence (with the same number of episodes for all runs with fixed algorithm/noise level) and plot episodic return against wall-clock time (Figure 3). We observe that expert distillation does not benefit from larger L, whereas the performance of RL sometimes benefits (at the cost of longer wall-clock time). This improvement is not as dramatic as the theory predicts, perhaps suggesting that there is theoretically unaccounted-for dependence between observation errors. Nevertheless, the results do corroborate the main prediction: RL robustly outperforms expert distillation for higher noise levels.
this section cite: []

Section: Empirical vignette: does belief contraction error track RL sub-optimality?
We empirically estimate belief contraction error for each task with no motor noise, and for humanoid-walk with std-dev = 0.2. We approximate the (unknown) ground truth belief by training a model b L ⋆ that takes L ⋆ = 10 input frames. We compare against models b L with L ∈ [2, 5] input frames. Each model's output belief is parametrized as a multivariate Gaussian distribution with diagonal covariance. All models are trained on the same 2000 trajectories collected by the latent expert policy. For each L we compute the KL-divergence (a tractable proxy for TV-distance) between outputs of b L and b L ⋆ , and average across 100 episodes of validation data, also collected by the same latent expert policy. We find that the empirical error decreases slightly as L increases (Figure 7), though not as fast as the theory predicts. 6 Adding motor noise has little noticeable effect (Figure 8). Interestingly, the error is not predictive across tasks: dog-walk has highest empirical error among the three tasks, yet RL achieves the lowest sub-optimality on it (Figure 1), indicating a theoretically-unexplained confounder.
this section cite: []

Section: Towards Better Distillation: Imitating a Smoother Expert
In this section, we discuss how the bounds via approximate decodability (e.g., Lemma 3.1) are loose since they fail to capture the smoothness of the latent expert. A tighter bound with smoothness suggests potential benefits of artificially smoothing the latent expert before distillation. We then propose a broadly-applicable method for improving the smoothness, and show that it yields empirical benefits. We view these results as largely a proof-of-concept and leave more detailed investigation to future work.
this section cite: []

Section: Smoothness of the latent policy.
Suppose that the true belief state at some step is always uniform over two particular states {s, s ′ }. Then decodability error is large, and a worst-case latent policy π latent -namely, one that plays different actions on these states -is unavoidably misspecified with respect to the class of executable policies. However, ambiguity between s and s ′ is most likely to occur if these states are somehow similar (e.g., close w.r.t. a metric). If π latent is smooth in the sense that it plays similar action distributions for nearby states, then the misspecification should be mitigated. This phenomenon can be captured more generally by the following variant of Definition 3.1, which measures decodability error of the actions (and hence is adaptive to the latent expert): Definition 6.1 (Action-prediction error). Fix a latent policy π latent . For a fixed executable policy π, and timestep h, the action-prediction error is defined as
ε act;π latent h (π) = E π 1 -π latent • b h (x 1:h , a 1:h-1 ) ∞ .
In Lemma 3.1, the decodability error can indeed be replaced by the action-prediction error -see Lemma E.4. Note that so long as π latent is deterministic (which is without loss of generality for the optimal latent policy), it generically holds that
ε act;π latent h (π) ≤ ε decode h (π).
Algorithmic intervention: smoothing experts with motor noise. One way to construct a smoother expert policy is to pre-or post-compose the optimal latent policy at each step with e.g. a Gaussian convolution kernel (on the state or action space, respectively). However, such approaches ignore the sequential nature of decision-making: smoothing the policy at later steps means that earlier actions may no longer be optimal. We propose instead computing the optimal policy for a modified latent MDP with additional motor noise. This encourages robustness to motor noise, as a tractable proxy for robustness to observation noise-see Appendix G for an example of one potential mechanism by which the former may lead to the latter.
this section cite: []

Section: Experimental results.
For both humanoid-walk and dog-walk, for each σ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}, we train an expert latent policy π σ in the environment with mean-0, std. dev.-σ, Gaussian motor noise on each action. We distill each expert to an executable policy via DAgger in an environment with σ = 0.2. We observe that π 0.2 incurs worse estimated actionprediction error than some higher-noise experts (Figure 9). Moreover, despite being the optimal latent policy for this environment, it is not the best expert to distill (Figure 4): the distillations of policies with lower action-prediction error achieve higher reward (substantially for humanoid-walk and modestly for dog-walk). We also observe that the effect disappears when the true environment has deterministic dynamics (Appendix H.2), likely since it is near-decodable.
this section cite: []

Section: Related methods.
We view this method as a lightweight version of asymmetric RL methods that iteratively refine the expert [78]. It is also closely related to the principle of noise injection in imitation learning [35,4], which has been shown to robustify Behavior Cloning-to match the performance of DAgger-by mitigating out-of-distribution effects. 7 Figure 4 demonstrates that even though DAgger uses online data collection to mitigate out-of-distribution effects, noise injection can still improve its performance in challenging image-based tasks-thus suggesting that there may be a qualitatively different phenomenon at play in highly misspecified settings.
Limitations and future work. Our theoretical results are for discrete tabular models with independent observation noise; weakening these assumptions could yield more precise understanding of the fundamental challenges that arise in applications with rich partial observations. Our experiments use synthetic injected motor noise; extending to more natural sources of stochasticity could be valuable. Also, there is a vast design space of algorithmic interventions for smoothing, of which we have only touched the surface. Finally, our work is motivated by applications like robot learning where near-decodability is plausible, but an important problem -which we did not explore -is to understand the algorithmic trade-offs in applications that require active information-gathering.
this section cite: ['b77', 'b34', 'b3']

Section: Contents of Appendix A Additional Related Work
A.
1 Theoretical literature . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . A.2 Empirical literature . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . B Additional Preliminaries B.1 Belief states . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . B.2 Decodability and γ-Observability . . . . . . . . . . . . . . . . . . . . . . . . . . . C Technical Lemmas D Omitted Proofs for Perturbed Block MDP D.1 Belief Contraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . D.2 Approximate Decodability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . D.3 Misspecification Lower Bound for Stochastic Dynamics . . . . . . . . . . . . . . . . E Omitted Proofs for Expert Distillation E.1 Misspecification Bounds for Composed Policies . . . . . . . . . . . . . . . . . . . . E.2 Analysis of Forward for L-step Executable Policies . . . . . . . . . . . . . . . . . E.3 Finite-sample guarantee . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F Omitted Proofs for Reinforcement Learning G A Motivating Toy Model for Smoothing H Supplemental Materials for Experiments H.1 Misspecification of Decodability in Practice . . . . . . . . . . . . . . . . . . . . . H.2 Imitating a Smoother Expert under Deterministic Latent Dynamics . . . . . . . . . H.3 Omitted Figures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . H.4 Experiment Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
this section cite: []

Section: A Additional Related Work
A.1 Theoretical literature Planning and learning in POMDPs. It is well-known that the planning problem in POMDPs (i.e. finding a near-optimal policy given the description of the POMDP) is computationally intractable [57,39,6,54], and the harder learning problem (i.e. finding a near-optimal policy given interactive sample access to the POMDP) is also statistically intractable [31], without additional assumptions. In light of these results, there has been recent interest in uncovering natural assumptions that allow statistically or computationally efficient algorithms. On the computational side, [19] introduced the L-step decodability assumption, and under this assumption derived a learning algorithm with time complexity poly(X L , A L , H) via frame-stacking. Additionally, [24,23] derived a quasi-polynomial time algorithm for learning in γ-observable POMDPs. Computationally efficient learning algorithms are also known for certain classes of POMDPs with deterministic dynamics [28,76] and certain latent MDPs [33,34], which are a special case of POMDPs with fixed latent information.
On the statistical side, [28] derived a statistically efficient algorithm for POMDPs satisfying a weak observability condition. Recently, [41,75,42,81] proposed statistically efficient algorithms for POMDPs or Predictive State Representations (PSR) satisfying certain low-rank conditions. More tangential to our work, there has also been increasing interest in off-policy evaluation in POMDPs [74,83].
Learning with privileged information in POMDPs. The most relevant theoretical works to ours are recent works that study the problem of learning POMDPs with latent state information (also called hindsight observability) [33,84,36,7]. Of these, [33,84] are focused on a narrow yet interesting special case of POMDPs called latent MDPs, where the unobserved data is fixed and lowdimensional. [36] show that learning in general POMDPs with latent state information is statistically tractable, in contrast with the situation without latent state information. [7] show that with latent state information the sample complexity of the algorithm for learning γ-observable POMDPs [23] can be improved from quasi-polynomial to polynomial, though it is an open question whether this is possible without latent state information. Finally, as mentioned earlier, [7] showed that in perfectly decodable POMDPs (Definition B.3), expert distillation yields a fully polynomial time algorithm for learning (for arbitrarily large window size L). Since there is a statistical lower bound of Ω(A L ) in the absence of latent state information [19], this yields a provable computational benefit of latent state information (and, in particular, for expert distillation), but only for perfectly decodable POMDPs.
Compared to the preceding theoretical works, our work seeks both theoretically and empirically grounded understanding of the relative merits of expert distillation versus standard reinforcement learning. Among works with similar motivations or results, [12] derive expressions for the output of expert distillation (analogous to Lemma 3.1, except they use a slightly different value-based distillation procedure rather than policy-based) and establish sub-optimality bounds for several imitation learning algorithms. However, they do not instantiate these bounds for concrete models, or theoretically contrast with reinforcement learning. [71] establish provable benefits of imitating the optimal policy in a fully-observed MDP (versus learning it via reinforcement learning), but they do not consider partial observability nor the ensuing error due to misspecification. [72] discuss a failure mode of expert distillation in POMDPs when using offline imitation learning to distill the latent expert. The "latching" effect that they discuss is due to conditioning on previous actions (see also [66]) -a technical issue that corresponds to why we analyze Forward with L random actionsthough it is not clear whether this effect is related to the performance gaps between behavior cloning and DAgger in our locomotion experiments. Finally, several works [2,79] give examples of a more fundamental failure mode of expert distillation: in general POMDPs, the optimal policy may need to take information-gathering actions. The classical "Tiger Door" exemplifies this failure mode [40].
Learning with rich observations. There has been extensive recent interest in reinforcement learning with rich observations [31], i.e. where the observation space is too large to enumerate. This line of work has developed largely in parallel with the literature on partial observability, but it is motivated by similar applications as our work (e.g. robotics with image-based perception), and these works formalize the fundamental empirical challenge of representation learning, i.e. "learning to see" [9]. The most well-studied model is the Block MDP [14,17], which corresponds to perfect decodability with L = 1, but is studied in function approximation settings where the observation space is extremely large or infinite, since the problem is computationally easy if the observation space has polynomiallybounded cardinality. While the task of learning in Block MDPs is typically computationally intractable as it inherits the intractable of PAC learning [25], there is by now precise understanding of the computational complexity relative to supervised learning oracles [49,82,46,70,61].
As observed by [7], there is also a provable computational benefit of latent state information in Block MDPs. Recent works [23,61] showed that in the absence of latent state information, learning in Φ-decodable Block MDPs (where Φ is the function approximation class) is strictly harder than the supervised learning task of Φ-decodable one-context regression. In contrast, it is straightforward to see that with latent state information and a one-context regression oracle, the true decoding function ϕ ⋆ ∈ Φ can be learned up to inverse-polynomial error (on average over any exploratory policy). This function, composed with the optimal latent policy, yields the optimal executable policy. [7] formally proved this result with a slightly different (multi-class classification rather than regression) supervised learning oracle.
this section cite: ['b56', 'b38', 'b5', 'b53', 'b30', 'b18', 'b23', 'b22', 'b27', 'b75', 'b32', 'b33', 'b27', 'b40', 'b74', 'b41', 'b80', 'b73', 'b82', 'b32', 'b83', 'b35', 'b6', 'b32', 'b83', 'b35', 'b6', 'b22', 'b6', 'b18', 'b11', 'b70', 'b71', 'b65', 'b1', 'b78', 'b39', 'b30', 'b8', 'b13', 'b16', 'b24', 'b48', 'b81', 'b45', 'b69', 'b60', 'b6', 'b22', 'b60', 'b6']

Section: A.2 Empirical literature
Applied methods that learn with privileged information. Privileged information has been widely used in training policies for real-world POMDPs, such as in robotics and autonomous driving. The most prominent and successful method is expert distillation [56,9,37,48,32,85,80,69,21,26,10]. First, one trains an expert policy with access to privileged information -either the latent state in a simulator [9], or observation data from more expensive sensors that will not be available at deployment [56]. Second, one trains an executable policy by performing offline or online imitation learning with respect to the latent policy. [56] also observe empirical benefits of online imitation learning compared to offline, which is corroborated by our results. While there are also notable successes of using RL without privileged information [1,43], some of the previously-mentioned works observed that RL without privileged information failed to learn locomotion in their environment [37].
Motivated by the theoretical failure modes of expert distillation in the prequel, there is also a line of work in the middle ground between expert distillation and RL without privileged information, that seeks to avoid these failure modes while also improving the convergence of standard RL. These hybrid methods include Asymmetric Actor-Critic [58], and more broadly are described as asymmetric learning [58,79,78,55,77,67,45,52,38]. While there is some evidence that an algorithm inspired by Asymmetric Actor-Critic may enjoy an improved statistical/computational trade-off for γ-observable POMDPs [7] (compared to the best-known algorithm that does not use privileged information [23]), the theoretical foundations for these methods remain otherwise largely unexplored.
Learning with and without privileged information. In addition to the previously-mentioned ablation experiments [37], recent work of [53] conducted controlled comparisons between expert distillation and standard RL on simulated locomotion and manipulation tasks, with the goal of providing heuristic guidance on when to prefer expert distillation over RL without privileged information. They found that expert distillation converges faster. They also classified tasks as "easy" or "hard" based on the convergence speed of standard RL, and suggested that expert distillation performed better on the "hard" tasks.
Improvements to expert distillation. Recall that a key benefit of expert distillation for POMDPs with rich observations (e.g. as found in robotics with image-based perception) was that it avoids performing reinforcement learning on the high-dimensional and complex observation space. In contrast, most of the previously-mentioned works on asymmetric learning use exactly such an algorithm (e.g., as the "Actor" component in Asymmetric Actor-Critic). An exception is the method of [78], which is a variant of expert distillation that iteratively refines the expert with the goal of decreasing misspecification. Our smoothed distillation method (Section 6) can be thought of as a more lightweight approach that refines the expert in one shot. As mentioned in Section 6, it is also similar (though not identical) to several noise injection methods in imitation learning [35,4].
this section cite: ['b55', 'b8', 'b36', 'b47', 'b31', 'b84', 'b79', 'b68', 'b20', 'b25', 'b9', 'b8', 'b55', 'b55', 'b0', 'b42', 'b36', 'b57', 'b57', 'b78', 'b77', 'b54', 'b76', 'b66', 'b44', 'b51', 'b37', 'b6', 'b22', 'b36', 'b52', 'b77', 'b34', 'b3']

Section: B Additional Preliminaries

this section cite: []

Section: B.1 Belief states
The following operators describe how a belief state evolves as more information is revealed. Definition B.1 (Belief state update [24]). For each h ∈ {1, . . . , H}, the Bayes operator is
B h : ∆(S h ) × X h → ∆(S h ) defined by B h (b; x h )(s h ) := O h (x h | s h )b(s h ) z h ∈S h O h (x h | z h )b(z h ) .
For each h ∈ {2, . . . , H}, the belief update operator
U h : ∆(S h-1 ) × A h-1 × X h → ∆(S h ), is defined by U h (b; a h-1 , x h ) := B h (P h (a h-1 ) • b; x h ) where P h (a) denotes the real-valued |S h | × |S h-1 | matrix of latent transition probabilities from step h -1 to step h under action a h-1 .
The following definition of an approximate belief state is analogous to the inductive definition of a true belief state (Definition 2.1); the only difference is that it updates based on a window of the L most recent observations and actions (x h-L+1:h , a h-L:h-1 ) rather than the entire history, and is additionally parametrized by a distribution D (which represents the prior on the latent state at step h -L). Definition B.2. For a window length L > 0, any h > L, and prior D ∈ ∆(S h-L ), the approximate belief state is inductively defined as b apx h (x h-L+1:h , a h-L:h-1 ; D) := U h (b apx h-1 (x h-L+1:h-1 , a h-L:h-2 ; D), a h-1 , x h ) where for L = 0, b apx h (∅; D) := D. For h ≤ L, the approximate belief state is defined to coincide with the true belief state.
this section cite: ['b23']

Section: B.2 Decodability and γ-Observability
For completeness, we include the definition of (perfect) L-step decodability from [19]. Definition B.3 (L-step decodable model [19]). A POMDP is said to be L-step decodable if, for each timestep h ∈ [H], there exists a deterministic mapping ϕ h : X h-L:h × A h-L:h-1 → S h such that for any admissible trajectory τ = (s, x, a) 1:h (i.e., a trajectory that occurs with positive probability under the uniformly random policy), we have s h = ϕ h (x h-L:h , a h-L:h-1 ).
Next, we introduce relevant definitions and results relating to γ-observable POMDPs from [24,23]. Definition B.4 (γ-observability [24]). Let γ ∈ (0, 1). A POMDP is γ-observable if for any h ∈ [H] and distributions b, b ′ ∈ ∆(S h ), it holds that
O ⊤ h b -O ⊤ h b ′ 1 ≥ γ∥b -b ′ ∥ 1 , where O h ∈ R S h ×X h is the observation matrix defined by (O h ) sx := O h (x | s).
The algorithm of [23] for learning γ-observable POMDPs in quasi-polynomial time requires bounding a slightly generalized version of the belief contraction error defined in Definition 3.2. We state this version below. Definition B.5 (Generalized belief contraction [23]). Let ε, ϕ ∈ (0, 1) and L ∈ N. We say that a POMDP P satisfies (ε; ϕ, L)-belief contraction if the following property holds. Let π be an executable policy, let h ∈ {L + 1, . . . , H}, and let D, D ′ ∈ ∆(S h-L ). If D ′ D ∞ ≤ 1/ϕ, then for any fixed history (x 1:h-L , a 1:h-L-1 ) it holds that
E s h-L ∼D ′ E π ∥b apx h (x h-L+1:h , a h-L:h-1 ; D ′ ) -b apx (x h-L+1:h , a h-L:h-1 ; D)∥ 1 ≤ ε
where the inner expectation is over partial trajectories (x h-L+1:h , a h-L:h-1 ) drawn from P by initializing to latent state s h-L at step h -L, and sampling action a k ∼ π(x 1:k , a 1:k-1 ) at each h -L ≤ k < h.
For context, see [23,Theorem 6.2] for the formal statement that γ-observability implies (ε; ϕ, L)belief contraction with L ∼ γ -4 log(1/(εϕ)). Definition 3.2 is a special case of the above definition, with D ′ := b h-L (x 1:h-L , a 1:h-L-1 ) and D := unif(S h-L ).
this section cite: ['b18', 'b18', 'b23', 'b22', 'b23', 'b22', 'b22', 'b22']

Section: Theorem B.1 ([23]
). There is a constant C ⋆ with the following property. Given ε, β, γ > 0, L ∈ N, and a γ-observable POMDP P, set ϕ := γ C ⋆ •H 5 S 7/2 X 2 ε. If P satisfies (ε; ϕ, L)-belief contraction, the algorithm BaSeCAMP [23] produces an executable policy π that satisfies J(π ⋆ ) -J( π) ≤ ε • poly(S, X, H, γ -1 ) with probability at least 1 -β.
Moreover, the time complexity is poly((XA) L , H, S, ε -1 , γ -1 , log(β -1 )).
Proof. Immediate from inspecting the analysis of BaSeCAMP [23]: while their analysis sets L := γ -4 log(1/(εϕ)), the only place this is used in the proof is to invoke [23, Theorem 6.2] (which is the claim that any γ-observable POMDP satisfies (ε; ϕ, L)-belief contraction with that choice of L). Thus, it is sufficient to choose any L for which (ε; ϕ, L)-belief contraction holds.
this section cite: ['b22', 'b22']

Section: C Technical Lemmas
Lemma C.1 (Data processing inequality). Let S, T be sets, let p, q ∈ ∆(S) be distributions, and let K : S → ∆(T ) be a conditional distribution function. Then
TV(K • p, K • q) ≤ TV(p, q). Similarly, if p ≪ q, then K • p K • q ∞ ≤ p q ∞ .
Proof. The first inequality follows from the fact that total variation distance is an f -divergence. The second inequality can be directly checked: for all y ∈ T ,
(K • p)(y) = x∈S K(y | x)p(x) ≤ p q ∞ x∈S K(y | x)q(x) = p q ∞ (K • q)(y
)
as needed.
Recall that a policy is executable if the action distribution at any step is determined by the action/observation history (note that a latent policy therefore may not be executable). The following lemma, which was implicitly used in prior work [24], verifies under any executable policy, the conditional distribution of the latent state given the history is the true belief state. 8Lemma C.2. Fix any step h ∈ [H] and executable policy π. Then
P π [s h | x 1:h , a 1:h-1 ] = b h (x 1:h , a 1:h-1 )(s h ) and, if h > 1, P π [x h | x 1:h-1 , a 1:h-1 ] = (O ⊤ h • P h (a h-1 ) • b h-1 (x 1:h-1 , a 1:h-2 ))(x h
) for any action/observation history (x 1:h , a 1:h-1 ) and latent state s h .
Proof. We prove the first claim by induction on h. It is clear that
P π [s 1 | x 1 ] ∝ O 1 (x 1 | s 1 )P[s 1 ] ∝ B 1 (P 1 ; x 1 ) = b 1 (x 1 )(s 1 )
, where proportionality is up to factors independent of s 1 . Since P π [• | x 1 ] and b 1 (x 1 ) are distributions, it follows from the proportionality that they are equal. Now fix any h ∈ {2, . . . , H} and assume the claim holds for h -1. Let (s 1:h , x 1:h , a 1:h-1 ) be a random trajectory drawn from P π , i.e. generated via s h ∼ P h (s h-1 , a h-1 ), x h ∼ O h (s h ), and a h ∼ π(x 1:h , a 1:h-1 ) (since we assumed that π is executable, the action distribution does not directly depend on s 1:h ). Then,
P π [s h | x 1:h , a 1:h-1 ] ∝ s 1:h-1 P π [s 1:h , x 1:h , a 1:h-1 ] = s 1:h-1 P π [s 1:h-1 , x 1:h-1 , a 1:h-2 ]π(x 1:h-1 , a 1:h-2 )P[s h | s h-1 , a h-1 ]O h (x h | s h ) ∝ s 1:h-1 P π [s 1:h-1 , x 1:h-1 , a 1:h-2 ]P[s h | s h-1 , a h-1 ]O h (x h | s h ) = O h (x h | s h ) s h-1 P[s h | s h-1 , a h-1 ] s 1:h-2 P π [s 1:h-1 , x 1:h-1 , a 1:h-2 ] ∝ O h (x h | s h ) s h-1 P[s h | s h-1 , a h-1 ]P π [s h-1 | x 1:h-1 , a 1:h-2 ] = O h (x h | s h ) s h-1 P[s h | s h-1 , a h-1 ]b h-1 (x 1:h-1 , a 1:h-2 )(s h-1 ) ∝ b h (x 1:h , a 1:h-1 )(s h )
where the penultimate equality uses the induction hypothesis and the final equality uses Eq. ( 1). This proves the first claim. To prove the second claim, observe that by a similar argument to above, for any h > 1,
P π [s h | x 1:h-1 , a 1:h-1 ] ∝ s h-1 P[s h | s h-1 , a h-1 ]b h-1 (x 1:h-1 , a 1:h-2 )(s h-1 ) so that P π [s h | x 1:h-1 , a 1:h-1 ] = (P h (a h-1 ) • b h-1 (x 1:h-1 , a 1:h-2 ))(s h ). But then P π [x h | x 1:h-1 , a 1:h-1 ] ∝ s 1:h P π [s 1:h , x 1:h , a 1:h-1 ] = s h O h (x h | s h ) s 1:h-1 P π [s 1:h , x 1:h-1 , a 1:h-1 ] ∝ s h O h (x h | s h )P π [s h | x 1:h-1 , a 1:h-1 ]. Therefore P π [x h | x 1:h-1 , a 1:h-1 ] = (O ⊤ h • P h (a h-1 ) • b h-1 (x 1:h-1 , a 1:h-2 ))(x h ) as claimed.
We will also need the following variant of Lemma C.2, which shows how approximate belief states arise as conditional probability distributions:
Lemma C.3. Fix any h ∈ [H], L ∈ {0, . . . , h -1}, and executable policy π where π h-t (• | x 1:h-t , a 1:h-t-1 ) is determined by (x h-L+1:h-t , a h-L:h-t-1 ) for all t ∈ [L]. Then
P π [s h | x h-L+1:h , a h-L:h-1 ] = b apx h (x h-L+1:h , a h-L:h-1 ; d π h-L ).
Proof. We induct on L. If L = 0 then, for any h ∈ [H], by definition
P π [s h ] = d π h (s h ) = b apx h (∅; d π h )(s h ).
Fix any L > 0 and suppose the claim holds for L -1 (for all h > L -1). Then for any h > L,
P π [s h | x h-L+1:h , a h-L:h-1 ] ∝ s h-L:h-1 P π [s h-L:h , x h-L+1:h , a h-L:h-1 ] = s h-L:h-1 P π [s h-L:h-1 , x h-L+1:h-1 , a h-L:h-2 ]π(a h-1 | x h-L+1:h-1 , a h-L:h-2 )P h [s h | s h-1 , a h-1 ]O h (x h | s h ) ∝ s h-L:h-1 P π [s h-L:h-1 , x h-L+1:h-1 , a h-L:h-2 ]P h [s h | s h-1 , a h-1 ]O h (x h | s h ) = O h (x h | s h ) s h-1 P h [s h | s h-1 , a h-1 ]P π [s h-1 , x h-L+1:h-1 , a h-L:h-2 ] ∝ O h (x h | s h ) s h-1 P h [s h | s h-1 , a h-1 ]P π [s h-1 | x h-L+1:h-1 , a h-L:h-2 ] = O h (x h | s h ) s h-1 P h [s h | s h-1 , a h-1 ]b apx h-1 (x h-L+1:h-1 , a h-L:h-2 ; d π h-L )(s h-1 ) ∝ b apx h (x h-L+1:h , a h-L:h-1 ; d π h-L )(s h )
this section cite: ['b23']

Section: by the induction hypothesis and the definition of b
apx h (x h-L+1:h , a h-L:h-1 ; d π h-L ).
The following fact is well-known. Lemma C.4. Let π ⋆ be the optimal policy of the POMDP, and let π latent be the optimal policy of the MDP. Then we have that J(π ⋆ ) ≤ J(π latent ).
Proof. We prove by proving a more general result. Consider any POMDP P and its corresponding MDP M, for any latent policy π latent , we use Q M;π latent h : S h × A h → [0, 1] to denote the Q-value of following π latent in the MDP at timestep h. We use Q P;π h : X 1:h × A 1:h → [0, 1] to denote the Q-value at timestep h of following executable policy π. We will use Q P and Q M to denote the optimal POMDP Q-value and optimal MDP Q-value functions. Note that Q P satisfies the following optimality equation, for any x 1:h , a 1:h , we have
Q P h ((x 1:h , a 1:h-1 ), a h ) = s h b h (x 1:h , a 1:h-1 )R h (s h , a h ) + x h+1 P (x h+1 | x 1:h , a 1:h ) max a h+1 Q P h ((x 1:h+1 , a 1:h ), a h+1 ), where P (x h+1 | x 1:h , a 1:h ) = s h ,s h+1 b h (s h | x 1:h , a 1:h )P h (s h+1 | s h , a h )O h+1 (x h+1 | s h+1 ).
Note that in this case, x 1:h , a h-1 can be summarize as b h (x 1:h , a h-1 ); and thus given any belief b h ∈ ∆(S h ), we will abuse the notation and define
Q P h (b h , a h ) = s h b h (s h )R h (s h , a h ) + x h+1 P (x h+1 | b h , a h ) max a h+1 Q P h (b h+1 , a h+1 ),
where b h+1 := U h+1 (b h ; a h , x h+1 ). Similarly, we can define
Q M h (b h , a h ) = s h b h (s h )Q M (s h , a h ).
Note that in this case, let π ⋆ be the optimal executable policy, and let π latent be the optimal MDP policy, we have that
J(π ⋆ ) = E x1 Q P 1 (b 1 (x 1 ), π ⋆ (x 1 )) ,and
J(π latent ) = E s1 [Q M 1 (s 1 , π latent (s 1 ))].
In the following we will prove that, for any timestep h ∈ [H], for any admissible belief b h ∈ ∆(S h ), fix action a h , we have that
Q M h (b h , a h ) ≥ Q P h (b h , a h
). We proceed with induction. For h = H, we have
Q M H (b H , a H ) = s H b H (s H )R H (s H , a H ) = Q P H (b H , a H ).
Then assuming Q M h+1 (b h+1 , a h+1 ) ≥ Q P h (b h+1 , a h+1 ) for any admissible b h+1 , we have for any admissible b h and action a h ,
Q P h (b h , a h ) = s h b h (s h )R h (s h , a h ) + x h+1 P (x h+1 | b h , a h ) max a h+1 Q P h+1 (b h+1 , a h+1 ) ≤ s h b h (s h )R h (s h , a h ) + x h+1 P (x h+1 | b h , a h ) max a h+1 Q M h+1 (b h+1 , a h+1 ) ≤ s h b h (s h )R h (s h , a h ) + x h+1 P (x h+1 | b h , a h )   s h+1 b h+1 (s h+1 ) max a h+1 Q M h+1 (s h+1 , a h+1 )   .
For any function f that only depend on the state, we have
x h+1 P (x h+1 | b h , a h )   s h+1 b h+1 (s h+1 )f (s h+1 )   = x h+1 P (x h+1 | b h , a h )   s h+1 O h (x h+1 | s h+1 ) s h P h (s h+1 | s h , a h )b h (s h ) P (x h+1 | b h , a h ) f (s h+1 )   = x h+1 s h+1 O h+1 (x h+1 | s h+1 ) s h P h (s h+1 | s h , a h )b h (s h )f (s h+1 ) = s h ,s h+1 P h (s h+1 | s h , a h )b h (s h )f (s h+1 ).
This gives that
Q P h (b h , a h ) ≤ s h b h (s h )R h (s h , a h ) + s h ,s h+1 P h (s h+1 | s h , a h )b h (s h ) max a h+1 Q M h+1 (s h+1 , a h+1 ) = s h b h (s h )   R h (s h , a h ) + s h+1 P h (s h+1 | s h , a h ) max a h+1 Q M h+1 (s h+1 , a h+1 )   = s h b h (s h )Q M h (s h , a h ) = Q M h (b h , a h ).
Finally, we conclude the proof by noting that
J(π latent ) = E s1 [Q M 1 (s 1 , π latent (s 1 ))] = E s1 [max a1 Q M 1 (s 1 , a 1 )] ≥ max a1 Q M 1 (b 1 , a 1 ) ≥ max a1 Q P 1 (b 1 , a 1 ) = J(π ⋆ ).
We will need the following martingale bound to analyze belief contraction error and decodability error in the perturbed Block MDP.
Lemma C.5. Fix ε ∈ (0, 3 -6 ) and S > 0. Let X 0 , . . . , X L be a non-negative supermartingale with E[X 0 ] ≤ S and Pr[X n+1 > εX n |X n ] ≤ ε almost surely for all 0 ≤ n < L. Then
E[min(X L , S)] ≤ 2 • 3 L ε L/3 S.
Proof. For any integer 0 ≤ n ≤ L and any integer k, define f (n, k) := Pr[ε k+1 S < X n ≤ ε k S]. We prove by induction on n that f (n, k) ≤ 3 n ε (n-k)/3 . If n = 0, then the claim is trivially true for k ≥ 0 since f (n, k) ≤ 1 always. For any k < 0, by Markov's inequality,
f (0, k) ≤ Pr[X 0 > ε k+1 S] ≤ E[X 0 ] ε k+1 S = ε -k-1 ≤ ε -k/3 .
For any 0 < n ≤ L and integer k, we have
f (n, k) = ∞ ℓ=-∞ Pr[ε k+1 S < X n ≤ ε k S | ε ℓ+1 S < X n-1 ≤ ε ℓ S] • f (n -1, ℓ) ≤ k-1 ℓ=-∞ f (n -1, ℓ) + εf (n -1, k) + εf (n -1, k + 1) + ∞ ℓ=k+2 ε ℓ-k-1 f (n -1, ℓ) (4
)
where the inequality uses the following two facts. First, for any ℓ ≥ k,
Pr[ε k+1 S < X n ≤ ε k S | ε ℓ+1 S < X n-1 ≤ ε ℓ S] ≤ Pr[X n > εX n-1 | ε ℓ+1 S < X n-1 ≤ ε ℓ S] ≤ ε by lemma assumption. Second, for any ℓ ≥ k + 2, Pr[ε k+1 S < X n ≤ ε k S | ε ℓ+1 S < X n-1 ≤ ε ℓ S] ≤ Pr[X n > ε k+1-ℓ X n-1 | ε ℓ+1 S < X n-1 ≤ ε ℓ S] ≤ ε ℓ-k-1 since X 0 , . . . , X L is a supermartingale. Returning to Eq. (4), we get f (n, k) ≤ k-1 ℓ=-∞ f (n -1, ℓ) + εf (n -1, k) + εf (n -1, k + 1) + ∞ ℓ=k+2 ε ℓ-k-1 f (n -1, ℓ) ≤ k-1 ℓ=-∞ 3 n-1 ε (n-1-ℓ)/3 + 3 n-1 ε 1+(n-k-1)/3 + 3 n-1 ε 1+(n-k-2)/3 + ∞ ℓ=k+2 3 n-1 ε ℓ-k-1+(n-ℓ-1)/3 ≤ 3 n-1 ε (n-k)/3 1 1 -ε 1/3 + ε 2/3 + ε 1/3 + 1 1 -ε 2/3 ≤ 3 n ε (n-k)/3
where the final inequality holds since ε ≤ 1/64. This completes the induction. Next,
E[min(X L , S)] ≤ -1 ℓ=-∞ S • f (L, ℓ) + ∞ ℓ=0 ε ℓ S • f (L, ℓ) ≤ 3 L S • -1 ℓ=-∞ ε (L-ℓ)/3 + ∞ ℓ=0 ε ℓ+(L-ℓ)/3 ≤ 3 L S • ε (L+1)/3 1 -ε 1/3 + ε L/3 1 -ε 2/3 ≤ 3 L S • 2ε L/3 as claimed.
this section cite: []

Section: D Omitted Proofs for Perturbed Block MDP
Below, we restate the definition of a δ-perturbed Block MDP. We then prove our main theoretical results for the perturbed Block MDP. In Appendix D.1, we prove Theorem 3.2 (the belief contraction result, restated as Theorem D.1). In Appendix D.2, we prove Proposition 4.1 (the decodability result for deterministic dynamics, restated as Proposition D.1). Finally, in Appendix D.3, we prove Proposition 5.1 (the misspecification lower bound for stochastic dynamics, restated as Proposition D.2).
Definition D.1. Fix a parameter δ > 0. A POMDP P is a δ-perturbed Block MDP if, for each h ∈ [H], there are O h , E h : S h → ∆(X h ) such that O h : S h → ∆(X h ) satisfies the block property [17]
, i.e. O h (• | s h ), O h (• | s ′
h ) have disjoint supports for all s h ̸ = s ′ h , and moreover the emission distribution O h at step h can be decomposed as follows:
O h (x h | s h ) = (1 -δ) O h (x h | s h ) + δE h (x h | s h ).
For notational convenience, for each x ∈ X h let ϕ(x) ∈ S h be the unique state for which O h (x | ϕ(x)) > 0 (or arbitrary, if no such state exists).
Definition D.2. For any h ∈ [H], b ∈ S h , and x h ∈ O h , we write
O h (x h | b) := z h ∈S h O h (x h | z h )b(z h ).
this section cite: ['b16']

Section: We similarly define E
h (x h | b) and O h (x h | b). Notice that O h (x h | b) = b(ϕ(x h )) O h (x h | ϕ(x h )).
this section cite: []

Section: D.1 Belief Contraction
In this section, we prove Theorem D.1, a slight generalization of Theorem 3.2. The proof broadly follows the proof of belief contraction for γ-observable POMDPs [24] (of which δ-perturbed Block MDPs are a special case -see Remark D.1), but since we require a stronger bound, we must modify the argument.
The basic idea (and main technical difficulty) in [24] is to identify a monotonic transform of an f -divergence that multiplicatively contracts in expectation under the Bayes operator (Definition B.1). In their case, they show that KL(B h (b;
x h ) ∥ B h (b ′ ; x h )) contracts by roughly a constant factor (relative to KL(b ∥ b ′ )) in expectation over x h ∼ O h (• | b).
That is, updating the true belief and approximate belief by an observation drawn from the true belief tends to decrease the KL-divergence. Updating the two beliefs by applying a transition matrix cannot increase the KL-divergence since it is an f -divergence, so an iterative argument (alternating between observation updates and transition updates) proves exponential contraction of the belief error.
However, we would like to prove contraction by poly(δ) per step, and the following example seems to present an obstacle to proving such contraction via KL-divergence. It also presents an obstacle to directly analyzing TV-distance.
Example D.1 (Failure of contraction of TV and KL). Fix δ > 0. Let S = X = {0, 1} and let O : S → ∆(X ) be defined by O(x | x) = 1 -δ. Define b = (1, 0) and b ′ = (δ 2 , 1 -δ 2 ). Then it holds almost surely over x ∼ O(• | b) that:
• TV(B(b; x), B(b ′ ; x)) ≥ 1 -δ even though TV(b, b ′ ) ≤ 1. • KL(B(b; x) ∥ B(b ′ ; x)) ≥ log(1/δ) even though KL(b ∥ b ′ ) ≤ 2 log(1/δ).
this section cite: ['b23', 'b23']

Section: ◁
To resolve this, we observe that when the TV-distance fails to decay, the density ratio ∥b/b ′ ∥ ∞ does decay. To formalize this, we study contraction of the following error metric. While we cannot show that it contracts by poly(δ) in expectation, we can show that it contracts with high probability; this is the content of Lemma D.2 below.
Definition D.3. For distributions b, b ′ ∈ ∆(S) with b ≪ b ′ , we define D ⋆ (b∥b ′ ) := TV(b, b ′ ) • b b ′ ∞ .
Note that the above metric upper bounds TV-distance, and is upper bounded by ∥b/b ′ ∥ ∞ . Also, as the product of metrics that satisfy the data processing inequality (Lemma C.1), it also satisfies the same inequality, so it cannot increase under application of (even stochastic) transitions.
Lemma D.1. Let h ∈ [H]. Fix b, b ′ ∈ ∆(S h ) with b ≪ b ′ , and fix x ∈ O h . Then B h (b; x) B h (b ′ ; x) ∞ = O h (x | b ′ ) O h (x | b) • b b ′ ∞ .
Proof. We have
B h (b; x) B h (b ′ ; x) ∞ = max s∈S h O h (x | s)b(s) O h (x | b) • O h (x | b ′ ) O h (x | s)b ′ (s) = max s∈S h b(s) b ′ (s) • O h (x | b ′ ) O h (x | b) = O h (x | b ′ ) O h (x | b) • b b ′ ∞ as claimed. Lemma D.2. Fix h ∈ [H] and b, b ′ ∈ ∆(S h ). Draw x ∼ O h (• | b). Define random variable ξ := D ⋆ (B h (b; x)∥B h (b ′ ; x)) = TV(B(b; x), B(b ′ ; x)) B(b; x) B(b ′ ; x) ∞ .(5)
Then E[ξ] ≤ 4D ⋆ (b∥b ′ ) and
Pr ξ > 4δ 1/3 D ⋆ (b∥b ′ ) ≤ 2δ 1/3 .
Proof. First, we compute that for any fixed x ∈ X h ,
TV(B(b; x), B(b ′ ; x)) = s∈S h O h (x | s) b(s) O h (x | b) - b ′ (s) O h (x | b ′ ) = (1 -δ) O h (x | ϕ(x)) b(ϕ(x)) O h (x | b) - b ′ (ϕ(x)) O h (x | b ′ ) + δ s∈S h E h (x | s) b(s) O h (x | b) - b ′ (s) O h (x | b ′ ) (6
)
by Definition B.1 and Definition D.1. We bound these terms individually. To bound the first term, since
O h (x | b) = (1 -δ)b(ϕ(x)) O h (x | ϕ(x)) + δE h (x | b) and O h (x | b ′ ) = (1 -δ)b ′ (ϕ(x)) O h (x | ϕ(x)) + δE h (x | b ′ ), O h (x | ϕ(x)) b(ϕ(x)) O h (x | b) - b ′ (ϕ(x)) O h (x | b ′ ) = δ O h (x | ϕ(x)) b(ϕ(x))E h (x | b ′ ) -b ′ (ϕ(x))E h (x | b) O h (x | b)O h (x | b ′ ) ≤ δ O h (x | ϕ(x)) |b(ϕ(x)) -b ′ (ϕ(x))| • E h (x | b) O h (x | b)O h (x | b ′ ) + b(ϕ(x)) • |E h (x | b) -E h (x | b ′ )| O h (x | b)O h (x | b ′ ) ≤ δ O h (x | ϕ(x)) |b(ϕ(x)) -b ′ (ϕ(x))| • E h (x | b) O h (x | b)O h (x | b ′ ) + |E h (x | b) -E h (x | b ′ )| (1 -δ)O h (x | b ′ ) . (7
)
where the second inequality uses the fact that
O h (x | b) ≥ (1 -δ)b(ϕ(x)) O h (x | ϕ(x)
). To bound the second term,
s∈S h E h (x | s) b(s) O h (x | b) - b ′ (s) O h (x | b ′ ) ≤ s∈S h E h (x | s) O h (x | b ′ ) |b(s) -b ′ (s)| + s∈S h E h (x | s)b(s) O h (x | b)O h (x | b ′ ) |O h (x | b) -O h (x | b ′ = s∈S h E h (x | s) O h (x | b ′ ) |b(s) -b ′ (s)| + E h (x | b) O h (x | b)O h (x | b ′ ) |O h (x | b) -O h (x | b ′ )| (8
)
Let E be the set of x ∈ X h such that E h (x | b) ≤ δ -1/3 O h (x | b).
Then the quantity ξ defined in Eq. ( 5) satisfies the following bound, where the expectation is over the randomness of
x ∼ O h (• | b): E[ξ1[x ∈ E]] = x∈E O h (x | b)TV(B h (b; x), B h (b ′ ; x)) B h (b; x) B h (b ′ ; x) ∞ = b b ′ ∞ x∈E O h (x | b ′ )TV(B h (b; x), B h (b ′ ; x)) ≤ b b ′ ∞ (1 -δ)δ x∈E O h (x | ϕ(x)) E h (x | b) O h (x | b) |b(ϕ(x)) -b ′ (ϕ(x))| + δ x∈E |E h (x | b) -E h (x | b ′ )| + δ x∈E s∈S h E h (x | s)|b(s) -b ′ (s)| + δ x∈E E h (x | b) O h (x | b) |O h (x | b) -O h (x | b ′ )| ≤ b b ′ ∞ (1 -δ)δ 2/3 x∈E O h (x | ϕ(x))|b(ϕ(x)) -b ′ (ϕ(x))| + δ x∈E |E h (x | b) -E h (x | b ′ )| + δ x∈E s∈S h E h (x | s)|b(s) -b ′ (s)| + δ 2/3 x∈E |O h (x | b) -O h (x | b ′ )| ≤ 4δ 2/3 TV(b, b ′ ) b b ′ ∞ (9
)
where the second equality is by Lemma D.1; the first inequality bounds each term TV(B h (b; x), B h (b ′ ; x)) using Eqs. ( 6) to ( 8); the second inequality uses the definition of E; and the final inequality uses the data processing inequality for kernels E h and O h . Additionally,
Pr[x ̸ ∈ E] = x∈X h \E O h (x | b) = x∈X h O h (x | b)1 O h (x | b) E h (x | b) < δ 1/3 ≤ δ 1/3 x∈X h E h (x | b) = δ 1/3 (10
) since E h (• | b) is a distribution. It follows that Pr ξ > 4δ 1/3 TV(b, b ′ ) b b ′ ∞ ≤ Pr ξ1[x ∈ E] > 4δ 1/3 TV(b, b ′ ) b b ′ ∞ + Pr[x ̸ ∈ E]
≤ 2δ 1/3 where the second inequality applies Markov's inequality to Eq. ( 9) for the first term, and Eq. ( 10) for the second term. This proves the second claim of the lemma statement. To prove the first claim, note that E h (x | b) ≤ δ -1 O h (x | b) for all x ∈ X h . Thus, modifying the calculation from Eq. ( 9) (this time summing over all x ∈ X h instead of x ∈ E) gives
E[ξ] = x∈X h O h (x | b)TV(B h (b; x), B h (b ′ ; x)) B(b; x) B(b ′ ; x) ∞ ≤ 4TV(b, b ′ ) b b ′ ∞ as needed.
The following result, which shows that the error metric decays with high probability under a belief update, is straightforward consequence of Lemma D.2 and the data processing inequality. Corollary D.1. Fix h ∈ {2, . . . , H}. Let b, b ′ ∈ ∆(S h-1 ) with b ≪ b ′ . For any action a h-1 ∈ A h-1 , with expectation over
x h ∼ (O h ) ⊤ P h (a h-1 ) • b, E[D ⋆ (U h (b; a h-1 , x h )∥U h (b ′ ; a h-1 , x h ))] ≤ 4D ⋆ (b∥b ′ ) and Pr[D ⋆ (U h (b; a h-1 , x h )∥U h (b ′ ; a h-1 , x h )) > 4δ 1/3 D ⋆ (b∥b ′ )] ≤ 2δ 1/3 .
Proof. By applying Lemma D.2 with P h (a h-1 ) • b and
P h (a h-1 ) • b ′ , E[D ⋆ (U h (b; a h-1 , x h )∥U h (b ′ ; a h-1 , x h ))] = E x h ∼(O h ) ⊤ P h (a h-1 )•b [D ⋆ (B h (P h (a h-1 ) • b; x h )∥B h (P h (a h-1 ) • b ′ ; x h ))] ≤ 4D ⋆ (P h (a h-1 ) • b∥P h (a h-1 ) • b ′ ) ≤ 4D ⋆ (b∥b ′ )
Pr D ⋆ (U h (b; a h-1 , x h )∥U h (b ′ ; a h-1 , x h )) > 4δ 1/3 D ⋆ (P h (a h-1 ) • b∥P h (a h-1 ) • b ′ ) ≤ 2δ 1/3
and therefore
Pr D ⋆ (U h (b; a h-1 , x h )∥U h (b ′ ; a h-1 , x h )) > 4δ 1/3 D ⋆ (b∥b ′ ) ≤ 2δ 1/3
by again applying the data processing inequality as above.
We now can prove our main belief contraction result (which includes Theorem 3.2 as a special case) by iteratively applying Corollary D.1. The main technical detail is to verify that the observations are indeed drawn from the true belief states, which relies on Lemma C.2. Theorem D.1. There is a universal constant C D.1 > 1 with the following property. Fix an executable policy π, indices 1 ≤ h < h + L ≤ H, and distributions D, D ′ ∈ ∆(S h ). Then for any partial history (x 1:h , a 1:h-1 ) it holds that
E s h ∼D ′ E π [TV(b apx h+L (x h+1:h+L , a h:h+L-1 ; D ′ ), b apx h+L (x h+1:h+L , a h:h+L-1 ; D)) | s h ] ≤ (C D.1 δ) L/9 D ′ D ∞
where the inner expectation is over partial trajectories (x h+1:h+L , a h:h+L-1 ) drawn from policy π with the environment initialized in state s h at step h.
As a consequence, it holds for any partial history (x 1:h , a 1:h-1 ) that
E π [TV(b h+L (x 1:h+L , a 1:h+L-1 ), b apx h+L (x h+1:h+L , a h:h+L-1 ; D))] ≤ (C D.1 δ) L/9 b h (x 1:h , a 1:h-1 ) D ∞
where the expectation is over trajectories drawn from π conditioned on the partial history (x 1:h , a 1:h-1 ).
Proof. We first observe that the second claim follows from the first claim by setting D ′ := b h (x 1:h , a 1:h-1 ). Indeed, conditioned on (x 1:h , a 1:h-1 ), the law of s h is precisely b h (x 1:h , a 1:h-1 ) (Lemma C.2), so drawing (x h+1:h+L , a h:h+L-1 ) conditioned on (x 1:h , a 1:h-1 ) is equivalent to first drawing s h ∼ b h (x 1:h , a 1:h-1 ) and then drawing (x h+1:h+L , a h:h+L-1 ) from the POMDP initialized at s h . Moreover, by the recursive definitions of b and b apx , we have b h+L (x 1:h+L , a 1:h+L-1 ) = b apx h+L (x h+1:h+L , a h:h+L-1 ; b h (x 1:h , a 1:h-1 )).
It remains to prove the first claim. Fix (x 1:h , a 1:h-1 ). For 0 ≤ t ≤ L, define the random variable X t := 4 -t D ⋆ (b apx h+t (x h+1:h+t , a h:h+t-1 ; D ′ )∥b apx h+t (x h+1:h+t , a h:h+t-1 ; D)) where (x h+1:h+L , a h:h+L-1 ) is drawn by sampling s h ∼ D ′ , initializing the POMDP at s h , and then rolling out with policy π (to be precise, the action distribution at step h + t is π(x 1:h+t , a 1:h+t-1 )).
Note that the roll-out does not resample x h , which is already fixed. Recall that D ⋆ (p∥q) := TV(p, q) p q ∞ , so that TV(p, q) ≤ D ⋆ (p∥q) ≤ p q ∞ for any distributions p, q. Then
X 0 = D ⋆ (b apx h (∅; D ′ )∥b apx h (∅; D)) = D ⋆ (D ′ ∥D) ≤ D ′ D ∞ .
Moreover,
TV(b apx h+L (x h+1:h+L , a h:h+L-1 ; D ′ ), b apx h+L (x h+1:h+L , a h:h+L-1 ; D)) ≤ min(D ⋆ (b apx h+L (x h+1:h+L , a h:h+L-1 ; D ′ )∥b apx h+L (x h+1:h+L , a h:h+L-1 ; D)), 1) ≤ 4 L min(X L , 1)(11)
since TV(p, q) ≤ 1 for any distributions p, q. Fix 0 < t ≤ L and condition on (x h+1:h+t-1 , a h:h+t-2 ), which determines X t-1 . The conditional distribution of a h+t-1 is then π(x 1:h+t-1 , a 1:h+t-2 ), and for any fixed a h+t-1 the conditional distribution of x h+t is (O h+t ) ⊤ P h+t (a h+t-1 ) • b apx h+t-1 (x h+1:h+t-1 , a h:h+t-2 ; D ′ ) by Lemma C.2 (applied to the modified POMDP that is initialized to a latent state s h ∼ D ′ immediately before the action a h is taken; for this POMDP b apx h+t-1 (x h+1:h+t-1 , a h:h+t-2 ; D ′ ) is the true belief state). Recall that by definition, b apx h+t (x h+1:h+t , a h:h+t-1 ; D ′ ) = U h+t (b apx h+t-1 (x h+1:h+t-1 , a h:h+t-2 ; D ′ ), a h+t-1 , x h+t ) and b apx h+t (x h+1:h+t , a h:h+t-1 ; D) = U h+t (b apx h+t-1 (x h+1:h+t-1 , a h:h+t-2 ; D), a h+t-1 , x h+t ). By Corollary D.1, it holds in expectation (resp., in probability) over x h+t , conditioned on the prior history, that
E[X t ] = 4 -t E[D ⋆ (b apx h+t (x h+1:h+t , a h:h+t-1 ; D ′ )∥b apx h+t (x h+1:h+t , a h:h+t-1 ; D))] ≤ 4 1-t D ⋆ (b apx h+t-1 (x h+1:h+t-1 , a h:h+t-2 ; D ′ )∥b apx h+t-1 (x h+1:h+t-1 , a h:h+t-2 ; D)) = X t-1
this section cite: []

Section: and similarly
Pr[X t > δ 1/3 X t-1 ] ≤ 2δ 1/3 . Since these bounds hold for any fixed a h+t-1 ∈ A h , they also hold in expectation (resp., in probability) over the joint draws of a h+t-1 and x h+t , conditioned on any realization of
(x h+1:h+t-1 , a h:h+t-2 ). Thus, E[X t | X t-1 ] ≤ X t-1 and Pr[X t > δ 1/3 X t-1 | X t-1 ] ≤ 2δ 1/3
both hold almost surely. We can now apply Lemma C.5 to the sequence (X 0 , . . . , X L ) with parame-
ters S := D ′ D ∞ and ε := 2δ 1/3 ; we get that E[min(X L , 1)] ≤ E[min(X L , S)] ≤ 2 • 3 L 2 L/3 δ L/9 D ′ D ∞ .
Combining this bound with Eq. ( 11), and setting C D.1 to be a sufficiently large constant, completes the proof.
Remark D.1. Any δ-perturbed Block MDP is γ-observable with γ = 1 -2δ (Definition B.4): for any h ∈ [H], we have O h = (1 -δ) O h + δE h , and thus given any b, b ′ , we have
O ⊤ h b -O ⊤ h b ′ 1 = (1 -δ) O ⊤ h (b -b ′ ) + δE ⊤ h (b -b ′ ) 1 ≥ (1 -δ) O ⊤ h (b -b ′ ) 1 -δ E ⊤ h (b -b ′ ) 1 ≥ (1 -δ) ∥b -b ′ ∥ 1 -δ ∥E h ∥ op ∥b -b ′ ∥ 1 ≥ (1 -2δ) ∥b -b ′ ∥ 1
because O h satisfies the block property. It was shown in [24, Theorem 4.7] that, for any γ-observable POMDP P, the belief contraction error can be bounded as
ε contract h (π; L) ≤ (1 -γ 4 /2 40 ) L • O(S).
However, substituting in γ := 1 -2δ, we see that due to the constant factor of 2 40 , this bound does not asymptotically improve as δ decreases -indeed, it is never better than (1 -1/2 40 ) L • O(S)and moreover is vacuous for L = o(log S).
this section cite: []

Section: D.2 Approximate Decodability
In this section we prove Proposition 4.1, restated below as Proposition D.1, which states that for δ-perturbed Block MDPs with deterministic latent transitions, the decodability error decays exponentially. The proof is somewhat analogous to that of Theorem D.1; the key difference is that the claim that the transitions do not increase decodability error is only true for deterministic transitions (whereas the analogous claim for belief contraction error is unconditionally true).
For notational convenience, we make the following definition of the "ℓ ∞ variance" V ∞ (b) for a given distribution b.
Definition D.4. For any set S and distribution b ∈ ∆(S), define V ∞ (b) := 1 -∥b∥ ∞ .
The following lemma shows that the ℓ ∞ variance contracts by poly(δ) with high probability under the Bayes operator.
Lemma D.3. Let δ > 0, and suppose that P is a δ-perturbed Block MDP with deterministic latent transitions (but potentially stochastic initial state). Fix h ∈ [H] and b ∈ ∆(S h ).
Draw x ∼ O h (• | b). Then E[V ∞ (B h (b; x))] ≤ min(V ∞ (b), δ) and Pr[V ∞ (B h (b; x)) > δ 1/3 V ∞ (b)] ≤ 3δ 1/3 . Proof. Pick any s ⋆ ∈ S h such that b s ⋆ = ∥b∥ ∞ , and hence s∈S h \{s ⋆ } b(s) = V ∞ (b). Then E[V ∞ (B h (b; x))] = x∈X h O h (x | b) 1 -max s∈S h B h (b; x)(s) = x∈X h min s∈S h (O h (x | b) -b(s)O h (x | s)) = x∈X h min s∈S h s ′ ∈S h \{s} b(s ′ )O h (x | s ′ )(12)
where the second equality is by the definition
B h (b; x)(s) := b(s)O h (x|s) O h (x|b) (Definition B.1). First, Eq. (12) implies that E[V ∞ (B h (b; x))] ≤ x∈X h s ′ ∈S h \{s ⋆ } b(s ′ )O h (x | s ′ ) = s ′ ∈S h \{s ⋆ } b(s ′ ) = V ∞ (b),
where the first equality uses the fact that O h (• | s ′ ) is a distribution for any fixed s ′ . Next, Eq. ( 12) implies that
E[V ∞ (B h (b; x))] ≤ x∈X h s ′ ∈S h \{ϕ(x)} b(s ′ )O h (x | s ′ ) = δ x∈X h s ′ ∈S h \{ϕ(x)} b(s ′ )E h (x | s ′ ) = δ s ′ ∈S h b(s ′ ) x∈X h :ϕ(x)̸ =s ′ E h (x | s ′ ) ≤ δ s ′ ∈S h b(s ′ ) ≤ δ.(13)
This, together with the preceding bound, proves the first claim of the lemma. Now consider the event that ϕ(
x) = s ⋆ . Since O h (x | b) ≥ (1 -δ)b(ϕ(x)) O h (x | ϕ(x)), we have Pr[ϕ(x) ̸ = s ⋆ ] = 1- x∈X h :ϕ(x)=s ⋆ O h (x | b) ≤ 1-(1-δ)b(s ⋆ ) = 1-(1-δ)(1-V ∞ (b)) ≤ δ+V ∞ (b).(14)
Moreover, by an analogous calculation as Eq. ( 12),
E[V ∞ (B h (b; x))1[ϕ(x) = s ⋆ ]] = x∈X h :ϕ(x)=s ⋆ O h (x | b) 1 -max s∈S h B h (b; x)(s) = x∈X h :ϕ(x)=s ⋆ min s∈S h (O h (x | b) -b(s)O h (x | s)) ≤ x∈X h :ϕ(x)=s ⋆ O h (x | b) -b(s ⋆ )O h (x | s ⋆ ) = x∈X h :ϕ(x)=s ⋆ s∈S h \{s ⋆ } b(s)O h (x | s) = δ x∈X h :ϕ(x)=s ⋆ s∈S h \{s ⋆ } b(s)E h (x | s) ≤ δV ∞ (b).
It follows that
Pr[V ∞ (B h (b; x)) > δ 1/3 V ∞ (b)] ≤ Pr[V ∞ (B h (b; x))1[ϕ(x) = s ⋆ ] > δ 1/3 V ∞ (b)] + Pr[ϕ(x) ̸ = s ⋆ ] ≤ δ 2/3 + δ + V ∞ (b)
by Markov's inequality and Eq. ( 14). To conclude, we distinguish two cases.
If V ∞ (b) ≤ δ 1/3 , then we get Pr[V ∞ (B h (b; x)) > δ 1/3 V ∞ (b)] ≤ δ 2/3 + δ + δ 1/3 ≤ 3δ 1/3 as needed. Otherwise, V ∞ (b) > δ 1/3 , so Pr[V ∞ (B h (b; x)) > δ 1/3 V ∞ (b)] ≤ Pr[V ∞ (B h (b; x)) > δ 2/3 ] ≤ δ 1/3
by Markov's inequality and Eq. ( 13). This completes the proof.
Using Lemma D.3 and the assumption of deterministic latent dynamics, it is straightforward to show that the ℓ ∞ variance contracts by poly(δ) with high probability under the belief update operator: Corollary D.2. Let δ > 0, and suppose that P is a δ-perturbed Block MDP with deterministic latent transitions (but potentially stochastic initial state). Fix h ∈ [H] and b ∈ ∆(S h ).
this section cite: []

Section: For any action a
h-1 ∈ A h-1 , with expectation over x h ∼ (O h ) ⊤ P h (a h-1 ) • b, it holds that E[V ∞ (U h (b; a h-1 , x h ))] ≤ V ∞ (b)and
Pr[V ∞ (U h (b; a h-1 , x h )) > δ 1/3 V ∞ (b)] ≤ 3δ 1/3 .
Proof. From Definition B.1, we have for any
x h that U h (b; a h-1 , x h ) = B h (P h (a h-1 ) • b; x h ).
this section cite: []

Section: Applying Lemma D.3 to the distribution P
h (a h-1 ) • b (observe that x h is indeed distributed according to O h (• | P h (a h-1 ) • b)), we get E[V ∞ (U h (b; a h-1 , x h ))] ≤ V ∞ (P h (a h-1 ) • b)and
Pr[V ∞ (U h (b; a h-1 , x h )) > δ 1/3 V ∞ (P h (a h-1 ) • b)] ≤ 3δ 1/3 . To complete the proof, it suffices to show that V ∞ (P h (a h-1 ) • b) ≤ V ∞ (b)
. Indeed, since the transitions are deterministic, the matrix
P h (a h-1 ) ∈ R |S h |×|S h-1 | satisfies that every column is a standard basis vector. Identify any s ⋆ ∈ S h-1 with b s ⋆ = ∥b∥ ∞ . Then there is some s h ∈ S h with P h (a h-1 ) s h ,s ⋆ = P h (s h | s ⋆ , a h-1 ) = 1. But then the entry of P h (a h-1 ) • b indexed by s h is at least b s ⋆ . So indeed V ∞ (P h (a h-1 ) • b) ≤ V ∞ (b).
We can now prove the following restatement of Proposition 4.1.
Proposition D.1. There is a universal constant C D.1 > 1 so that the following holds. Let δ > 0, and suppose that P is a δ-perturbed Block MDP with deterministic latent transitions (but potentially stochastic initial state). Fix any executable policy π and index h ∈ [H]. It holds that
E π [V ∞ (b h (x 1:h , a 1:h-1 ))] ≤ min(δ, (C D.1 δ) (h-1)/9 ).
Proof. Define a sequence of random variables X t := V ∞ (b t (x 1:t , a 1:t-1 )) for 1 ≤ t ≤ h, where (x 1:h , a 1:h-1 ) is a random trajectory drawn from policy π. We have X 1 ≤ 1 almost surely. By the same argument as in Theorem D.1 (except using Corollary D.2 rather than Corollary D.1), we have for all
1 < t ≤ h that E[X t | X t-1 ] ≤ X t-1 and Pr[X t > δ 1/3 X t-1 | X t-1 ] ≤ 3δ 1/3
hold almost surely. Thus, Lemma C.5 applied to (X 1 , . . . , X h ) with parameters S := 1 and ε := 3δ
1/3 implies that E[X h ] = E[min(X h , 1)] ≤ 2 • 3 h-1 3 (h-1)/3 δ (h-1)/9 ≤ (C D.1 δ) (h-1)/9 so long as C D.1 is sufficiently large. Additionally, we know that b 1 (x 1 ) = B 1 (P 1 ; x 1 ) so E[X 1 ] = E π [V ∞ (B 1 (P 1 ; x 1
))] ≤ δ by Lemma D.3 and the fact that
x 1 has distribution O 1 (• | P 1 ). Thus, E[X t ] ≤ δ for all 1 ≤ t ≤ h.
this section cite: []

Section: D.3 Misspecification Lower Bound for Stochastic Dynamics
In this section we prove the following restatement of Proposition 5.1, which shows that in a δperturbed Block MDP with general (stochastic) latent transitions, the misspecification of the optimal latent policy with respect to the class of executable policies can be as large as Ω(δH) (for δ ≤ 1/H). This implies an analogous lower bound on decodability error, i.e. it cannot improve exponentially as h increases, unlike the case of deterministic latent transitions. Moreover, it shows a fundamental source of (horizon-dependent) error that is not mitigated by increasing the frame-stack L: since the following bound applies to all executable policies, it also applies to the class of L-step executable policies for any L. Proposition D.2. Let δ > 0 and H ∈ N. There is a δ-perturbed Block MDP P with horizon H such that the optimal latent policy π latent satisfies
min π∈Π TV(P π latent , P π ) ≥ Ω(min(1, δH))
where Π is the class of executable policies.
Proof. We define P as follows. For all h ∈ [H], define the latent state space and observation space to be S h := X h := {0, 1}; also define A h := {0, 1}. Let the initial distribution and latent transition dynamics at each step be uniformly random (independent of the previous state and action). For each h ∈ [H], define the reward function R h :
S h × A h → [0, 1] be defined by R h (s, a) = 1 H 1[s = a].
this section cite: []

Section: Define the observation distribution O
h : S h → ∆(S h ) so that O h (s | s) = 1 -δ and O h (1 -s | s) = δ.
It is clear that P is a δ-perturbed Block MDP. Under the trajectory distribution P π latent induced by the optimal latent policy π latent , it holds that a h = s h for all h ∈ [H] with probability 1. However, for any executable policy π, for any step h and history τ 1:h-1 = (s 1:h-1 , x 1:h-1 , a 1:h-1 ), it holds that Pr π [a h = s h | τ 1:h-1 ] ≤ 1 -δ since the prior history is independent of s h , and the conditional distribution s h | x h has only 1 -δ mass on x h . Thus,
Pr π [∀h ∈ [H] : a h = s h ] ≤ (1 -δ) H .
If δ ≥ 1/H then (1 -δ) H ≤ e -1 ≤ 1 -Ω(1). Otherwise, (1 -δ) H ≤ 1 -Ω(δH). Thus, TV(P π , P π latent ) ≥ Ω(min(1, δH)) as claimed.
this section cite: []

Section: E Omitted Proofs for Expert Distillation

this section cite: []

Section: E.1 Misspecification Bounds for Composed Policies
In this section we prove upper bounds on the misspecification of a latent policy (with respect to certain executable policies obtained by composing the latent with some belief state) in terms of instance-dependent error metrics. The first main result is Lemma E.3 (a restatement of Lemma 3.1), where the upper bound is in terms of decodability error (Definition 3.1) and error in approximating the true belief state. The second main result is Lemma E.4, where the decodability error term is replaced by action-prediction error (Definition 6.1).
To prove Lemma E.3, it is convenient to first analyze the policy that samples a state from the true belief state induced by the current history, and then samples an action from π latent accordingly. The key technical observation, below, encapsulates the intuition that resampling a near-deterministic random variable is likely to yield the same realization. Proof. Consider the process where we draw (Y, Y nuis ) ∼ P Y,Y nuis , Z ∼ P Z|Y , and Y ′ ∼ P Y . If Y ′ = Y then we set Z ′ = Z; otherwise we sample Z ′ ∼ P Z|Y ′ . Then (Y, Y nuis , Z) is distributed according to P Y,Y nuis P Z|Y , and (Y, Y nuis , Z ′ ) is distributed according to P Y,Y nuis Q Z . Thus, we have defined a coupling. Moreover,
Pr[(Y, Y nuis , Z) ̸ = (Y, Y nuis , Z ′ )] ≤ Pr[Y ̸ = Y ′ ] ≤ 2 Pr[Y ̸ = arg max y P Y (y)] = 2V ∞ (P Y )
as needed for the first claim. For the second claim,
Pr[(Y, Y nuis , Z) ̸ = (Y, Y nuis , Z ′ )] ≤ Pr[Z ̸ = Z ′ ] = 2V ∞ (P Z )
as needed.
Lemma E.2. Let π latent ∈ Π latent be a latent (Markovian) policy and define the executable policy π by π(x 1:h , a 1:h-1 ) := π latent • b h (x 1:h , a 1:h-1 ). Then
TV(P π latent , P π ) ≤ 2 H h=1 E π [V ∞ (b h (x 1:h , a 1:h-1 ))].
Proof. For each 1 ≤ h ≤ H + 1 let π • h π latent denote the policy that follows π for the first h -1 actions and subsequently follows π latent . Since π • 1 π latent = π latent and π • H+1 π latent = π, it suffices to show that, for each h ∈ [H],
TV(P π• h π latent , P π• h+1 π latent ) ≤ 2E π [V ∞ (b h (x 1:h , a 1:h-1 ))].
Observe that both distributions have the same conditional distributions over (s h+1:H , x h+1:H , a h+1:H ) given (s 1:h , x 1:h , a 1:h ). By this fact and the data processing inequality,
TV(P π• h π latent , P π• h+1 π latent ) = TV(P π• h π latent X,Y,Y nuis ,Z , P π• h+1 π latent X,Y,Y nuis ,Z
) where X = (x 1:h , a 1:h-1 ), Y = s h , Y nuis = s 1:h-1 , and Z = a h . Both distributions have the same marginal over (X, Y, Y nuis ). The distribution of Y |X is precisely b h (x 1:h , a 1:h-1 ) by Lemma C.2 and the fact that π is executable. In the former distribution, a h is generated by sampling from π latent (s h ). In the latter distribution, a h is generated by sampling s ′ h ∼ b h (x 1:h , a 1:h-1 ) and then sampling from π latent (s ′ h ). Thus, the conditions of Lemma E.1 are met (after conditioning out X), and we get
TV(P π• h π latent , P π• h+1 π latent ) ≤ 2E π [V ∞ (b h (x 1:h , a 1:h-1
))] as needed.
We now prove the following restatement of Lemma 3.1.
Lemma E.3. Let π latent ∈ Π latent be a latent (Markovian) policy and let b 1:H be a collection of functions b h : X h × A h-1 → ∆(S h ). Define the executable policy π by π(x 1:h , a 1:h-1 ) := π latent • b h (x 1:h , a 1:h-1 ).
Then
TV(P π latent , P π ) ≤ H h=1 E π 2V ∞ (b h (x 1:h , a 1:h-1 )) + b h (x 1:h , a 1:h-1 ) -b h (x 1:h , a 1:h-1 ) 1and
TV(P π latent , P π ) ≤ H h=1 E π [2V ∞ (b h (x 1:h , a 1:h-1 ))]+E π b h (x 1:h , a 1:h-1 ) -b h (x 1:h , a 1:h-1 )
1 where π(x 1:h , a 1:h-1 ) := π latent • b h (x 1:h , a 1:h-1 ).
Proof. We can couple P π and P π so that at any step h, if the trajectories have thus far been the same sequence (x 1:h , a 1:h-1 ), then the probability that they choose different actions a h is at most
b h (x 1:h , a 1:h-1 ) -b h (x 1:h , a 1:h-1 ) 1 .
By a standard hybrid argument, it follows that
TV(P π , P π ) ≤ H h=1 E π b h (x 1:h , a 1:h-1 ) -b h (x 1:h , a 1:h-1 ) 1
and also
TV(P π , P π ) ≤ H h=1 E π b h (x 1:h , a 1:h-1 ) -b h (x 1:h , a 1:h-1 ) 1 .
Combining with Lemma E.2 completes the proof.
The preceding lemma used the intuition that if the latent state is near-deterministic when conditioned on the observation/action history, then resampling it is unlikely to change it. The following lemma uses the intuition that if the action is near-deterministic (which is likely when the action-prediction is small) when conditioned on the action/observation history, then resampling the latent state -and subsequently resampling the action conditioned on this resampled latent state -is unlikely to change the action, though it may have changed the state.
Lemma E.4. Let π latent ∈ Π latent be a latent (Markovian) policy and let b 1:H be a collection of functions b h :
X h × A h-1 → ∆(S h ).
Define the executable policy π by π(x 1:h , a 1:h-1 ) := π latent • b h (x 1:h , a 1:h-1 ).
Then
TV(P π latent , P π ) ≤ H h=1 E π 2ε act;π latent h (π) + π latent • b h (x 1:h , a 1:h-1 ) -π latent • b h (x 1:h , a 1:h-1 )
1 where π(x 1:h , a 1:h-1 ) := π latent • b h (x 1:h , a 1:h-1 ).
Proof. As with the proof of Lemma E.3, it suffices to show that
TV(P π latent , P π ) ≤ H h=1 E π 2ε act;π latent h (π) .
The proof is identical to that of Lemma E.3 except for using the second claim of Lemma E.1 instead of the first.
this section cite: []

Section: E.2 Analysis of Forward for L-step Executable Policies
In this section we describe the (slightly modified) Forward imitation learning algorithm [62], applied to the problem of distilling an expert latent policy π latent to an L-step executable policy π ∈ π L . We first formally derive the expression for the policy learned in the infinite-sample limit (Lemma E.5), and then prove Theorem E.1 (the regret bound for this policy, restated as Theorem 4.1). Finally, we prove a finite-sample guarantee for the modified Forward algorithm (Theorem E.2), using the same ideas together with a standard finite-sample analysis for Maximum Likelihood Estimation [20].
Forward with L-step random actions. For a latent Markovian policy π latent ∈ Π latent and a parameter L ∈ [H], the (modified) Forward algorithm computes an L-step executable policy π = π 1:H as follows. For h = 1, . . . , H:
1. Draw n trajectories τ i = (s i 1:H , x i 1:H , a i 1:H ) from the policy π 1:h-L-1 • h-L Unif(A) (which follows π until step h -L -1 and subsequently plays uniformly random actions).
this section cite: ['b61', 'b19']

Section: Compute
π h := arg max π h ∈Π L h 1 n n i=1 log π h (π latent h (s h ) | x max(1,h-L+1):h , a max(1,h-L):h-1 ). Above, Π L h is the set of L-step conditional distributions π h : X h-L+1:h × A h-L:h-1 → ∆(A).
Note that the standard Forward algorithm is identical except that it draws data from π 1:h-1 at step h. The modified algorithm is simpler to analyze since the random actions do not bias the conditional distribution of the latent state at step h given the partial history (x h-L+1:h , a h-L:h-1 ), so this distribution can be directly related to an approximate belief state with appropriate prior (Lemma C.3).
For notational convenience, let π denote the policy obtained from the above algorithm in the infinitesample limit, i.e. at step h we define
π h := arg max π h ∈Π L h E π 1:h-L-1 • h-L Unif(A) log π h (π latent h (s h ) | x max(1,h-L+1):h , a max(1,h-L):h-1 ) .
The following lemma gives a closed-form expression for this policy. Lemma E.5. It holds for all h ∈ [H] that
π h (• | x h-L+1:h , a h-L:h-1 ) = π latent h • b apx h (x h-L+1:h , a h-L:h-1 ; d π h-L ) if h > L π latent h • b h (x 1:h , a 1:h-1 )
otherwise so long as Π h contains this conditional distribution.
Proof. We have for each h > L that π h = arg max
π h ∈Π h E π 1:h-L-1 • h-L Unif(A) log π h (π latent h (s h ) | x h-L+1:h , a h-L:h-1 ) .
Since population-level maximum likelihood minimizes KL divergence, we get
π h (a h | x h-L+1:h , a h-L:h-1 ) = s h ∈S h π latent h (a h | s h ) • P π 1:h-L-1 • h-L Unif(A) [s h | x h-L+1:h , a h-L:h-1 ] = s h ∈S h π latent h (a h | s h ) • b apx h (x h-L+1:h , a h-L:h-1 ; d π 1:h-L-1 • h-L Unif(A) h-L )(s h ) = s h ∈S h π latent h (a h | s h ) • b apx h (x h-L+1:h , a h-L:h-1 ; d π h-L )(s h )
where the second equality uses Lemma C.3. The application of this lemma uses the fact that π 1:h-L-1 • h-L Unif(A) plays actions at steps h -L, . . . , h -1 that are uniformly random. This proves the lemma in the case h > L. The case h ≤ L is analogous but uses Lemma C.2 instead of Lemma C.3.
Theorem E.1. Suppose that the POMDP P is a δ-perturbed Block MDP with deterministic transitions, and fix L ∈ N. Let π latent ∈ Π latent be a latent (Markovian) policy, and let π be the policy computed by Forward with L-step random actions, in the infinite-sample limit. Then J(π latent ) -J( π) ≤ TV(P π latent , P π ) ≤ O(δ) + (C D.1 δ) L/9 SH.
this section cite: []

Section: Proof.
Define the executable policy π by π(x 1:h , a 1:h-1 ) := π latent • b h (x 1:h , a 1:h-1 ). By Lemma E.3 and the closed-form expression for π (Lemma E.5), we have
TV(P π latent , P π ) ≤ H h=1 E π [2V ∞ (b h (x 1:h , a 1:h-1 ))] + H h=L+1 E π b h (x 1:h , a 1:h-1 ) -b apx h (x h-L+1:h , a h-L:h-1 ; d π h-L 1 .
By Proposition D.1 and the fact that π is executable, the first term can be bounded as
H h=1 E π [2V ∞ (b h (x 1:h , a 1:h-1 ))] ≤ H h=1 min δ, (C D.1 δ) (h-1)/9
≤ O(δ).
Next, for each h ∈ {L + 1, . . . , H}, we can bound
E π b h (x 1:h , a 1:h-1 ) -b apx h (x h-L+1:h , a h-L:h-1 ; d π h-L ) 1 ≤ (C D.1 δ) L/9 E π b h (x 1:h , a 1:h-1 ) d π h-L ∞ ≤ (C D.1 δ) L/9 E π s∈S h b h (x 1:h , a 1:h-1 )(s) d π h-L (s) = (C D.1 δ) L/9 s∈S h 1 d π h-L (s) E π [b h (x 1:h , a 1:h-1 )(s)] = (C D.1 δ) L/9 s∈S h 1 d π h-L (s) E π P π [s h = s | x 1:h , a 1:h-1 ] = (C D.1 δ) L/9 s∈S h d π h-L (s) d π h-L (s) = (C D.1 δ) L/9 S
where the first inequality uses Theorem D.1 (and the fact that π is an executable policy), and the second equality uses Lemma C.2 (and the fact that π is an executable policy). Putting everything together, we get TV(P π latent , P π ) ≤ O(δ) + (C D.1 δ) L/9 SH as claimed.
this section cite: []

Section: E.3 Finite-sample guarantee
Theorem E.2. There is a constant C E.2 > 0 with the following property. Let δ, η, ε opt > 0 and suppose that the POMDP P is a δ-perturbed Block MDP with deterministic transitions. If n ≥ C E.2 X L A 3L+1 H 2 ε -2 opt log(Hn/η), the output π of the modified Forward algorithm with n samples per step satisfies, with probability at least 1 -η, J(π latent ) -J( π) ≤ TV(P π latent , P π ) ≤ O(δ) + (C D.1 δ) L/9 SH + ε opt .
Moreover, π can be computed in time poly(n, H, X L , A L ).
Proof. By a standard analysis of the log-loss for unconstrained distribution classes, π h (x max(1,h-L+1):h , a max(1,h-L):h-1 ) is precisely the empirical distribution of π latent h (s i h ) over the data i ∈ [n] with (x i max(1,h-L+1):h , a i max(1,h-L):h-1 ) = (x max(1,h-L+1):h , a max(1,h-L):h-1 ). Thus, π can be computed in the stated time complexity.
To prove the claimed statistical bound, fix h ∈ [H]. Let G = {g π h : π h ∈ Π L h } denote the family of distributions indexed by Π L h , where g π h is the joint distribution of τ h = (x max(1,h-L+1):h , a max(1,h-L):h-1 ) and π h (τ h ) over trajectories drawn from π 1:h-L-1 • h-L Unif(A). Also let g ⋆ denote the joint distribution of τ h and π latent h (s h ) over trajectories drawn from π 1:h-L-1 • h-L Unif(A). Then g ⋆ = g π ⋆ h ∈ G where
π ⋆ h (• | x h-L+1:h , a h-L:h-1 ) = π latent h • b apx h (x h-L+1:h , a h-L:h-1 ; d π h-L ) if h > L π latent h • b h (x 1:h , a 1:h-1 )
otherwise by the same argument as in Lemma E.5. Moreover, g π h is precisely the Maximum Likelihood Estimation (MLE) estimate over distribution class G with n samples from g ⋆ . Thus, we can compare π h and π ⋆ h using a standard analysis for MLE, e.g. [20, Proposition B.1]: we can bound the logcovering number of G (with discretization error ε := 1/(Hn)) by X L A L+1 log(1/ε), so we get with probability at least 1 -η/H that
TV(g ⋆ , g π h ) = E π 1:h-L-1 • h-L Unif(A) [TV( π ⋆ h (• | τ h ), π h (• | τ h )] ≤ O X L A L+1 log(Hn/η) n
where τ h := (x max(1,h-L+1):h , a max(1,h-L):h-1 )). By change-of-measure, it follows that
E π 1:h-1 [TV( π ⋆ h (• | τ h ), π h (• | τ h )] ≤ O A L • X L A L+1 log(Hn/η) n ≤ ε opt H (15
)
where the final inequality holds by the theorem assumption that n
≥ C E.2 X L A 3L+1 H 2 ε -2
opt log(Hn/η), so long as C E.2 is a sufficiently large constant. Next, we observe that
TV(P π latent , P π ) ≤ H h=1 E π [TV(π latent h (s h ), π h (τ h ))] ≤ H h=1 E π [TV(π latent h (s h ), π ⋆ h (τ h ))] + H h=1 E π [TV( π ⋆ h (τ h ), π h (τ h ))] ≤ H h=1 E π [2V ∞ (b h (x 1:h , a 1:h-1 ))] + H h=L+1 E π b h (x 1:h , a 1:h-1 ) -b apx h (x h-L+1:h , a h-L:h-1 ; d π h-L ) 1 + H h=1 E π [TV( π ⋆ h (τ h ), π h (τ h ))]
where the final inequality is by Lemma E.3. As in Theorem E.1, the first term can be bounded by O(δ) and the second term can be bounded by (C D.1 δ) L/9 S. By Eq. ( 15) and a union bound over h ∈ [H], the third term is at most ε opt with probability at least 1 -η. Substituting these bounds into the above expression completes the proof.
this section cite: []

Section: F Omitted Proofs for Reinforcement Learning
In this section we prove Corollary 3.1, stated formally below.
Corollary F.1. There is a reinforcement learning algorithm that, for any given δ, β ∈ (0, 1/3) and L ∈ N, and any δ-perturbed Block MDP P, learns a policy π rl satisfying J(π ⋆ ) -J(π rl ) ≤ (C 3.2 δ) L/18 • poly(S, X, H)
with probability at least 1 -β and in time (XA/δ) O(L) • poly(H, S, log(1/β)).
Proof. Recall from Remark D.1 that any δ-perturbed Block MDP P is (1 -2δ)-observable (Definition B.4). Also, by Theorem D.1, P satisfies (ε; ϕ, L)-belief contraction for any ϕ > 0 and L ∈ N with ε := (C D.1 δ) L/9 • ϕ -1 . Thus, we can invoke Theorem B.1 with γ := 1/3 and ε := (C D.1 δ) L/18 √ 3C ⋆ H 5 S 7/2 X 2 , where C ⋆ is as defined in Theorem B.1. The choice of ϕ in Theorem B.1 indeed satisfies ε = (C D.1 δ) L/9 • ϕ -1 , so P satisfies (ε; ϕ, L)-belief contraction, and thus the algorithm BaSeCAMP [23] produces an executable policy π that satisfies J(π ⋆ ) -J( π) ≤ ε • poly(S, X, H) in time poly((AX) L , H, S, ε -1 , log(β -1 )). Substituting in the choice of ε completes the proof.
this section cite: ['b22']

Section: G A Motivating Toy Model for Smoothing
Adding to the discussion from Section 6, we informally discuss a theoretical toy model in which smoothing the expert may decrease (a metric version of) action-prediction error and thus improve final performance. Consider a horizon-1 POMDP where the state and action spaces have metrics d S and d A , and the reward function R is binary-valued. For each latent state s, let G(s) be the set of "good" actions, i.e. G(s) := {a ∈ A : R(s, a) = 1}, and let D(s) be the diameter of G(s). Suppose that the following natural assumptions hold:
1. The map s → G(s) is d S → d A Lipschitz. That is, perturbing s by ε (with respect to metric d S )
only perturbs the set G(s) by O(ε) (with respect to metric d A ).
2. Under any observation, the posterior on states is "ε-local" with respect to d S , i.e. contained in an ε-ball.
With no smoothing, the optimal expert may, for each s, play an arbitrary action in G(s), so the "metric" action-prediction error (i.e. expected dispersion of actions played by the expert, conditioned on an observation) can be as large as O(ε) + max s D(s). However, suppose the environment has motor noise. In particular, suppose the support of the noise is an η-radius ball (with respect to d A ) around the chosen action. Then for each s, the optimal policy is forced to the "interior" of G(s), i.e. not within η of the boundary, effectively equivalent to decreasing the diameter of G(s) by η. Moreover, if η ≳ ε, then for any two states s, s ′ in the posterior of observation o, the actions chosen by the optimal policy will lie in both G(s) and G(s ′ ), so (under mild additional structural assumptions, e.g. convexity of G(s) in Euclidean space) distillation should produce an optimal policy.
Of course, if η is too large, the "interior" of some of the G(s) sets becomes empty, i.e. the optimal policy cannot play a robustly good action. As a result, it may play arbitrary actions for these states, so the action prediction error can become large again (and the policy value decreases).
this section cite: []

Section: H Supplemental Materials for Experiments
In Appendix H.1 we present details for our empirical test of whether perfect decodability holds in image-based locomotion tasks. In Appendix H.3 we present figures omitted from the main body of the paper. In Appendix H.4 we include hyperparameter choices and details about compute resources.
this section cite: []

Section: H.1 Misspecification of Decodability in Practice
For each of the three tasks (walker-run, humanoid-walk, and dog-walk), we collect 2000 trajectories from the expert latent policy, and for each L ∈ {1, 2, 3, 4, 5} we train a model directly mapping from L observations x h-L+1:h to the state s h , by minimizing mean-squared error. We then evaluate the validation error of the model on 100 trajectories collected from the same policy and plot the per-timestep error in Figure 5. We normalize the states with the trajectory mean and standard deviation of the combined dataset. Note that the error is composed with three parts: 1) error due to model capacity; 2) using fixed-length frame-stacks instead of the whole history; 3) inherent failure of perfect decodability. We observe that the trained model is able to achieve a small error in the later timesteps, which suggests that error 1, model capacity error, is (likely) small. However, the error is large in the initial timesteps. As error 2 does not exist for steps h ≤ L, it follows that error 3 is non-trivial, i.e., perfect decodability fails. Ablation. To investigate whether the error at initial timesteps is due to parameter sharing, we also tried to train non-stationary models (i.e., one model for each timestep) or using weighted loss with higher weights for the initial timesteps, but neither approach significantly changed the results.
this section cite: []

Section: Conclusions.
We conjecture that the higher error at early timesteps is due to a nearly uniform distribution for the initial state distribution, where states that induce occlusion may be quite likely. In later time-steps, the observations along the expert trajectory are in a more stable regime and thus may introduce less occlusion (and hence be more likely to correspond to a unique latent state). Finally, we observe that even at later time-steps there is still significant prediction error. While this could potentially be due to model capacity error, it nevertheless demonstrates impracticality of learning a perfect decoder, and it roughly corresponds with task difficulty (humanoid-walk and dog-walk are harder than walker-run).
Source of the error. One may naturally wonder if the error is caused by unpredictable state components that are irrelevant to decision-making (one example could be the absolute position of the agent, but in the environments that we test on, absolute coordinates are actually not part of the latent space). If this is the case, then the error would not negatively impact the performance of the policy distilled from the latent expert.
One piece of evidence against this hypothesis is that the action-prediction error is indeed non-trivial for our tasks of interest (c.r. Figure 9), which suggests that the error is not only due to irrelevant components. In addition, we take a more detailed look at the state prediction error and identify the components that contribute most to the error. With the same setup as in Figure 5, we compute the mean squared error for each coordinate of the state, averaged over the whole trajectories, and present the top 5 and bottom 5 coordinates in Table 1. We see that the coordinates that contribute most to the error are mostly angular velocities of limbs, which are indeed hard to predict from images. On the other hand, the coordinates that contribute least to the error are mostly joint angles or balance point coordinates, which are easier to predict from images. From first principles, all of these coordinates are crucial for the optimal policy, providing additional confirmation that the error is not only due to irrelevant components. In Section 6, we showed that under stochastic latent dynamics, imitating a smoother expert (which is trained under higher motor noise) can lead to better performance. A natural question is whether this phenomenon also exists under deterministic latent dynamics. Theoretically, the answer is no as we showed in Theorem E.1 that with enough framestack, imitating the non-smoothed expert can already be optimal under deterministic latent dynamics. That said, it remains unclear if smoothing the expert can help improve the performance in practice. To answer this question, we conduct experiments in the same setup as in Section 6, but with motor noise σ = 0 when performing expert distillation. We vary the motor noise level used to train the latent expert over {0.1, 0.2, 0.3, 0.4, 0.5}, and we use a framestack of size 3.
We present the results in Figure 6. We see that imitating a smoother expert does not lead to better performance in this case, and the best performance is achieved by imitating the non-smoothed expert (c.r., Figure 1). This corroborates our theoretical findings. H.3 Omitted Figures H.3.1 Belief contraction with/without motor noise Figure 7: Belief contraction error with respect to the framestack L = {2, 3, 4, 5} on all tasks. For each framestack L, we use train a Gaussian parametrized neural network to predict the belief with L framestack input. We compute the KL distance to the output of an L = 10 network (serving as an approximation of the true belief), averaged over a validation dataset with 100 episodes of data.
The orange plot denotes the decrease in KL divergence between two numbers of framestacks. We repeat the experiment for 5 times and plot the mean and standard deviation. We observe that the belief contraction error decreases (although not as fast as predicted by the theory) as the number of framestack increases.
Figure 8: Belief contraction error with respect to the framestack L = {2, 3, 4, 5} on humanoid-walk tasks, with and without motor noise. For each framestack L, we train a Gaussian parametrized neural network to predict the belief with L framestack input. We compute the KL distance to the output of an L = 10 network (serving as an approximation of the true belief), averaged over a validation dataset with 100 episodes of data. The orange plot denotes the decrease in KL divergence between two numbers of framestacks. We repeat the experiment for 5 times and plot the mean and standard deviation. We observe that very similar belief contraction phenomena occur with or without the motor noise.
this section cite: []

Section: References
Ref_id:b0 Title: Legged locomotion in challenging terrains using egocentric vision Year: (2023)
Ref_id:b1 Title: Hindsight is only 50/50: Unsuitability of mdp based approximate pomdp solvers for multi-resolution information gathering Year: (2018)
Ref_id:b2 Title: Butterfly effects of sgd noise: Error amplification in behavior cloning and autoregression Year: (2023)
Ref_id:b3 Title: Provable guarantees for generative behavior cloning: Bridging low-level stability and high-level behavior Year: (2023)
Ref_id:b4 Title: R-max-a general polynomial time algorithm for near-optimal reinforcement learning Year: (2002-10)
Ref_id:b5 Title: On the complexity of partially observed markov decision processes Year: (1996)
Ref_id:b6 Title: Provable partially observable reinforcement learning with privileged information Year: (2024)
Ref_id:b7 Title: Learning to search better than your teacher Year: (2015)
Ref_id:b8 Title: Learning by cheating Year: (2019)
Ref_id:b9 Title: Extreme parkour with legged robots Year: (2024)
Ref_id:b10 Title: Process reward models for llm agents: Practical framework and directions Year: (2025)
Ref_id:b11 Title: Datadriven planning via imitation learning Year: (2018)
Ref_id:b12 Title: Transfer from simulation to real world through learning deep inverse dynamics model Year: (2016)
Ref_id:b13 Title: On oracle-efficient PAC RL with rich observations Year: (2018)
Ref_id:b14 Title: Search-based structured prediction Year: (2009)
Ref_id:b15 Title: Provably efficient RL with rich observations via latent state decoding Year: (2019)
Ref_id:b16 Title: Provably efficient RL with rich observations via latent state decoding Year: (2019)
Ref_id:b17 Title: Partially observable reinforcement learning with memory traces Year: (2025)
Ref_id:b18 Title: Provable reinforcement learning with a short-term memory Year: (2022)
Ref_id:b19 Title: Is behavior cloning all you need? understanding horizon in imitation learning Year: (2024)
Ref_id:b20 Title: Learning deep sensorimotor policies for visionbased autonomous drone racing Year: (2023)
Ref_id:b21 Title: Towards general-purpose model-free reinforcement learning Year: (2025)
Ref_id:b22 Title: Learning in observable pomdps, without computationally intractable oracles Year: (2022)
Ref_id:b23 Title: Planning and learning in partially observable systems via filter stability Year: (2023)
Ref_id:b24 Title: Exploration is harder than prediction: Cryptographically separating reinforcement learning from supervised learning Year: (2024)
Ref_id:b25 Title: Anymal parkour: Learning agile navigation for quadrupedal robots Year: (2024)
Ref_id:b26 Title: Contextual decision processes with low Bellman rank are PAC-learnable Year: (2017)
Ref_id:b27 Title: Sample-efficient reinforcement learning of undercomplete POMDPs Year: (2020)
Ref_id:b28 Title: Near optimality of finite memory feedback policies in partially observed markov decision processes Year: (2022)
Ref_id:b29 Title: Near-optimal reinforcement learning in polynomial time Year: (2002)
Ref_id:b30 Title: PAC reinforcement learning with rich observations Year: (2016)
Ref_id:b31 Title: Rma: Rapid motor adaptation for legged robots Year: (2021)
Ref_id:b32 Title: RL for latent MDPs: Regret guarantees and a lower bound Year: (2021)
Ref_id:b33 Title: RL in latent MDPs is tractable: Online guarantees via off-policy evaluation Year: (2024)
Ref_id:b34 Title: Dart: Noise injection for robust imitation learning Year: (2017)
Ref_id:b35 Title: Learning in pomdps is sample-efficient with hindsight observability Year: (2023)
Ref_id:b36 Title: Learning quadrupedal locomotion over challenging terrain Year: (2020)
Ref_id:b37 Title: Guided policy optimization under partial observability Year: (2025)
Ref_id:b38 Title: Memoryless policies: Theoretical limitations and practical results Year: (1994)
Ref_id:b39 Title: Learning policies for partially observable environments: Scaling up Year: (1995)
Ref_id:b40 Title: When is partially observable reinforcement learning not scary? Year: (2022)
Ref_id:b41 Title: Optimistic mle: A generic model-based algorithm for partially observable sequential decision making Year: (2023)
Ref_id:b42 Title: Perpetual humanoid control for real-time simulated avatars Year: (2023)
Ref_id:b43 Title: Isaac gym: High performance gpu-based physics simulation for robot learning Year: (2021)
Ref_id:b44 Title: Student-informed teacher training Year: (2024)
Ref_id:b45 Title: Efficient model-free exploration in low-rank MDPs Year: (2023)
Ref_id:b46 Title: Representation learning with multi-step inverse kinematics: An efficient and optimal approach to rich-observation rl Year: (2023)
Ref_id:b47 Title: Learning robust perceptive locomotion for quadrupedal robots in the wild Year: (2022)
Ref_id:b48 Title: Kinematic state abstraction and provably efficient rich-observation reinforcement learning Year: (2020)
Ref_id:b49 Title: Playing atari with deep reinforcement learning Year: (2013)
Ref_id:b50 Title: Human-level control through deep reinforcement learning Year: (2015)
Ref_id:b51 Title: Tar: Teacher-aligned representations via contrastive learning for quadrupedal locomotion Year: (2025)
Ref_id:b52 Title: When should we prefer state-to-visual dagger over visual reinforcement learning? Year: (2025)
Ref_id:b53 Title: Complexity of finite-horizon markov decision process problems Year: (2000)
Ref_id:b54 Title: Leveraging fully observable policies for learning under partial observability Year: (2022)
Ref_id:b55 Title: Agile autonomous driving using end-to-end deep imitation learning Year: (2017)
Ref_id:b56 Title: The complexity of markov decision processes Year: (1987)
Ref_id:b57 Title: Asymmetric actor critic for image-based robot learning Year: (2017)
Ref_id:b58 Title: Toward the fundamental limits of imitation learning Year: (2020)
Ref_id:b59 Title: Computational-statistical tradeoffs at the next-token prediction barrier: Autoregressive and imitation learning under misspecification Year: (2025)
Ref_id:b60 Title: Necessary and sufficient oracles: Toward a computational taxonomy for reinforcement learning Year: (2025)
Ref_id:b61 Title: Efficient reductions for imitation learning Year: (2010)
Ref_id:b62 Title: Reinforcement and imitation learning via interactive no-regret learning Year: (2014)
Ref_id:b63 Title: A reduction of imitation learning and structured prediction to no-regret online learning Year: (2011)
Ref_id:b64 Title: Online planning algorithms for pomdps Year: (2008)
Ref_id:b65 Title: Regularized behavior cloning for blocking the leakage of past action information Year: (2023)
Ref_id:b66 Title: Tgrl: An algorithm for teacher guided reinforcement learning Year: (2023)
Ref_id:b67 Title: Cliport: What and where pathways for robotic manipulation Year: (2021)
Ref_id:b68 Title: Learning perception-aware agile flight in cluttered environments Year: (2023)
Ref_id:b69 Title: Rich-observation reinforcement learning with continuous latent dynamics Year: (2024)
Ref_id:b70 Title: Deeply aggrevated: Differentiable imitation learning for sequential prediction Year: (2017)
Ref_id:b71 Title: Sequence model imitation learning with unobserved contexts Year: (2022)
Ref_id:b72 Title: Deepmind control suite Year: (2018)
Ref_id:b73 Title: Future-dependent value-based off-policy evaluation in pomdps Year: (2023)
Ref_id:b74 Title: Provably efficient reinforcement learning in partially observable dynamical systems Year: (2022)
Ref_id:b75 Title: Computationally efficient pac rl in pomdps with latent determinism and conditional embeddings Year: (2023)
Ref_id:b76 Title: Impossibly good experts and how to follow them Year: (2022)
Ref_id:b77 Title: Robust asymmetric learning in pomdps Year: (2021)
Ref_id:b78 Title: Bridging the imitation gap by adaptive insubordination Year: (2021)
Ref_id:b79 Title: Neural volumetric memory for visual locomotion control Year: (2023)
Ref_id:b80 Title: PAC reinforcement learning for predictive state representations Year: (2023)
Ref_id:b81 Title: Efficient reinforcement learning in block MDPs: A model-free representation learning approach Year: (2022)
Ref_id:b82 Title: Statistical tractability of off-policy evaluation of history-dependent policies in pomdps Year: (2025)
Ref_id:b83 Title: Horizon-free reinforcement learning for latent markov decision processes Year: (2022)
Ref_id:b84 Title: Robot parkour learning Year: ()
