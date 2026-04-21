Title: Reinforcement Learning with Imperfect Transition Predictions: A Bellman-Jensen Approach
Abstract: Traditional reinforcement learning (RL) assumes the agents make decisions based on Markov decision processes (MDPs) with one-step transition models. In many real-world applications, such as energy management and stock investment, agents can access multi-step predictions of future states, which provide additional advantages for decision making. However, multi-step predictions are inherently high-dimensional: naively embedding these predictions into an MDP leads to an exponential blow-up in state space and the curse of dimensionality. Moreover, existing RL theory provides few tools to analyze prediction-augmented MDPs, as it typically works on one-step transition kernels and cannot accommodate multi-step predictions with errors or partial action-coverage. We address these challenges with three key innovations: First, we propose the Bayesian value function to characterize the optimal prediction-aware policy tractably. Second, we develop a novel BellmanJensen Gap analysis on the Bayesian value function, which enables characterizing the value of imperfect predictions. Third, we introduce BOLA (Bayesian Offline Learning with Online Adaptation), a two-stage model-based RL algorithm that separates offline Bayesian value learning from lightweight online adaptation to real-time predictions. We prove that BOLA remains sample-efficient even under imperfect predictions. We validate our theory and algorithm on synthetic MDPs and a real-world wind energy storage control problem.

Section: Introduction
Reinforcement Learning (RL) [1] has emerged as a powerful framework for sequential decisionmaking, achieving remarkable success across diverse domains [2][3][4][5]. Classical RL formulates decision-making as a Markov Decision Process (MDP), where an agent seeks to maximize expected cumulative rewards in a stochastic environment. A central premise of this framework is that once the agent has accurately captured the environment model (e.g., the transition dynamics), it can, in principle, compute an optimal policy. In model-based RL, this involves explicitly learning the transition kernel and reward function to solve the optimal policy [6]. Even in model-free RL, agents implicitly learn the environment through value function or policy learning [7].
However, in many real-world applications, agents can access even richer information: the prediction of future transition realizations. Rather than relying solely on expected transition dynamics, these realization-level predictions specify exact future states, which can reduce or even eliminate the environments inherent stochasticity and enable more effective decision-making. For example, in financial markets, accurate multi-step price forecasts can substantially improve trading strategies [8], while in energy systems, reliable predictions of renewable energy generation allow the system operators to schedule the power generation sources more efficiently [9].
Despite their potential, incorporating multi-step transition predictions into MDPs faces three key challenges. First, the predictions over a multi-step horizon are inherently high-dimensional: augmenting the state with these predictions expands the state space exponentially, making standard solutions computationally intractable (see Section 3 for more details). Second, even if the augmented MDP can be solved, existing theory lacks formal tools to quantify the benefits of multi-step transition predictions, particularly when they are inaccurate or only cover a subset of actions. Third, in the absence of strong assumptions on function approximation [1,[10][11][12][13], RL's sample complexity scales at least linearly with the size of the stateaction space [14,15], and the exponential state expansion also induces an exponential blow-up in the required samples [16]. Addressing these challenges is essential for the rigorous integration of transition predictions into RL. See Appendix A for a detailed literature review.
To overcome these challenges, our contributions can be summarized as follows:
Tractable Optimal Policy for MDPs with Transition Predictions. We introduce a low-dimensional Bayesian value function that integrates multi-step transition predictions into the value evaluation, which enables a tractable characterization of the optimal prediction-aware policy.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b0', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15']

Section: Characterization of the Value of Imperfect Predictions using Bellman-Jensen Gap.
We introduce the Bellman-Jensen Gap framework, a novel analytical tool that decomposes the advantage of multistep predictions into a recursive sum of local Jensen gaps in the Bayesian value function. Building on this framework, we characterize the value of imperfect predictions and show how it can close the performance gap to the offline optimal policy in Theorem 4.1.
this section cite: []

Section: Prediction-Aware Algorithm with Improved Sample Complexity.
We propose BOLA, a two-stage model-based RL algorithm that combines offline Bayesian value estimation with online integration of real-time predictions. We prove that BOLA avoids the exponential sample complexity and, given high-quality predictions, is more sample efficient than classical model-based RL [14,15]. Our analysis relies on tailored error-decomposition and telescoping bounds to control multi-step transition errors.
The remainder of this paper proceeds as follows. Section 2 formalizes the prediction-augmented MDP framework. Based on this formulation, Section 3 introduces a tractable Bayesian value function to characterize the optimal policy, circumventing the curse of dimensionality. Subsequently, Section 4 establishes the Bellman-Jensen Gap to theoretically quantify the value of imperfect predictions. This analysis directly motivates Section 5, where we present the BOLA algorithm with provable sample efficiency guarantees. Section 6 then provides empirical validation on a wind energy storage problem. Finally, Section 7 concludes with a discussion of limitations and future work. Complete proofs and additional details are included in the appendices.
this section cite: ['b13', 'b14']

Section: Markov Decision Processes with Transition Predictions
In this section, we formally introduce the framework for MDPs augmented with transition predictions. We begin by introducing the definition of a discounted infinite-horizon MDP, which is specified by the tuple M = (S, A, P, r, γ). Here, S and A are the finite state and action spaces, respectively; P is the transition kernel, i.e., P (• | s, a) denotes the distribution of the next state given that action a is taken at state s; r : S × A → [0, 1] is the reward function; and γ ∈ (0, 1) is the discount factor. Since we focus on finite MDPs, assuming bounded rewards is without loss of generality. The model parameters of the MDP (i.e., the transition kernel P and the reward function r) are unknown to the agent, but the agent can interact with the environment by observing the current state, selecting actions, and receiving the resulting next state and reward.
this section cite: []

Section: MDPs with Transition Predictions
We extend the classical MDP framework by incorporating imperfect predictions of future transition dynamics. Formally, consider an MDP with transition predictions characterized by the tuple M p = (S, A, P, r, γ, K, A -, ε), where (K, A -, ε) capture the prediction structure. Specifically, K denotes the finite prediction horizon; A -⊆ A specifies the subset of actions for which transition outcomes can be predicted; and ε quantifies the associated prediction errors.
We first consider an ideal case that the prediction is accurate. At discrete time steps t = 0, K, 2K, . . ., the agent receives a batch of predicted transitions for the next K steps, denoted as σ * = (σ * 1 , σ * 2 , . . . , σ * K ), and each σ * k is a binary matrix of size |S||A -| × |S|, where each row corresponds to a (s, a) pair with a ∈ A -and is a one-hot vector indicating the predicted next state.
Given an accurate one-step transition prediction σ * k , the conditional transition probabilities depend on whether the action taken falls within the predictable action subset A -. This subset captures actions for which reliable prediction models are available, allowing the agent to exploit future information. In contrast, actions outside A -must rely solely on the underlying transition dynamics P (s |s, a) of the environment. Accordingly, the conditional transition model with predictions is:
P (s | s, a, σ k ) =
σ * k ((s, a), s ), ∀a ∈ A -, s, s ∈ S, P (s | s, a), ∀a / ∈ A -, s, s ∈ S.
To preserve the Markov property of the underlying MDP, we impose two natural conditions on each onestep prediction matrix σ * k within the K-step forecast:
• Independence and stationarity. Each σ * k is drawn i.i.d. from a fixed distribution. This ensures that every predicted transition remains stationary with respect to the transition kernel. Also, the prediction depends only on the current stateaction pair and not on any prior history.
• Consistency. In expectation, the accurate prediction exactly recovers the true transition kernel:
E σ * k ∼P σ * σ * k ((s, a), s ) = P (s | s, a), ∀ k ≤ K, a ∈ A -, s, s ∈ S.(2)
However, exact prediction is not always attainable in practice. Thus, we assume the agent only receives inaccurate predictions denoted as σ = (σ 1 , σ 2 , . . . , σ K ) ∈ Q K , where Q K denotes the space of inaccurate prediction σ, and each σ k ∈ [0, 1] |S||A -|×|S| is a stochastic matrix indicating predicted transition probabilities for the state-action pairs at step k in the future. Each prediction σ k may differ from the true future transition σ * k due to prediction error. We model this discrepancy as σ k = σ * k + ε k , k = 1, . . . , K, where ε k is a random error matrix drawn from a distribution f ε k |σ * k . This formulation captures a broad class of imperfect predictions, allowing us to study how finitehorizon, partial, and inaccurate forecasts can be leveraged for improved decision-making in MDPs.
Remark: Our model differs fundamentally from prior work such as [17,18], which treats predictions as noisy estimates of the transition kernel and aims only to match standard MDP performance. In contrast, we model ideal predictions as concrete, one-hot realizations with certain consistency property in Eq. ( 2), enabling us to leverage realization-level information to surpass classic MDP performance. Furthermore, unlike these methods, we explicitly address partial action predictability, an open challenge identified in [17].
this section cite: ['b16', 'b17', 'b16']

Section: Decision-Making and Optimization Objectives
We adopt a fixed-horizon planning protocol [19], where the agent makes decisions at discrete time points t = 0, K, 2K, . . . . At each decision point, after observing the current state s t and receiving the prediction batch σ, the agent selects an action sequence a = (a 0 , . . . , a K-1 ) ∈ A K according to a policy π : S ×Q K → ∆(A K ), where Q K denotes the prediction space and ∆(A K ) denotes the probability simplex over K-step action sequences. This setting models how agents dynamically plan decisions over a finite prediction horizon. The agent seeks to maximize the expected cumulative reward by selecting an optimal policy, defined as
π * = arg max π E π [ ∞ t=0 γ t r(s t , a t ) | s 0 = s, σ 0 = σ]
for all s ∈ S, σ ∈ Q K .
this section cite: ['b18']

Section: Bayesian Value Function and Prediction-Aware Optimal Policy
In this section, we develop a tractable formulation for decision-making with multi-step imperfect transition predictions.
Motivation: A straightforward strategy for using predictions is to treat (s, σ) as the new state and solve a standard MDP over this extended state space. However, this approach quickly becomes intractable. A K-step prediction σ = (σ 1 , . . . , σ K ) consists of K transition matrices, each of size |S||A| × |S|, resulting in an exponentially large state space size of at least |S| K|S||A| . Moreover, since σ is typically noisy and continuous, its support can be uncountably infinite. As a result, the augmented value function must satisfy an infinite-dimensional Bellman equation, making classical solutions impractical even for K = 1.
In summary, while predictions have the potential to improve performance, naively augmenting the state space with raw prediction vectors leads to intractable computation. To address this issue, we next introduce a Bayesian value function that enables a tractable, prediction-aware characterization of the optimal policy.
this section cite: []

Section: Bayesian Value Function and Optimal Policy Structure
To avoid explicit state augmentation, we instead formulate a Bayesian value function defined over the original state space. The key idea is to take an expectation over the prediction distribution, thereby shifting the complexity into an outer integral while preserving a tractable structure. Formally, we define the Bayesian value function as:
V Bayes,π K,A -,ε (s) := E σ E π ∞ t=0 γ t r(s t , a t ) s 0 = s, σ 0 = σ .(3)
This Bayesian value function represents the expected cumulative reward when each decision is made after drawing a K-step prediction σ. We call it Bayesian because we marginalize over the distribution of σ, thereby accounting for forecast uncertainty in the value estimate. Importantly, the policy can condition on the realized σ, yet the value function itself remains defined solely over the original state space. This preserves tractability by avoiding an explicit statespace augmentation. The optimal Bayesian value is then:
V Bayes, * K,A -,ε (s) = max π V Bayes,π K,A -,ε (s), ∀ s ∈ S.(4)
By constructing an auxiliary MDP that incorporates the predictions and linking it with the optimal Bayesian value function, we derive the corresponding Bellman optimality equation (see Appendix B for the proof).
this section cite: []

Section: Theorem 3.1 (Bellman Optimality Equation for Bayesian Value Function).
The optimal Bayesian value function V Bayes, * K,A -,ε is the unique solution to the following fixed-point equation:
V Bayes, * K,A -,ε (s) = E σ max a K-1 t=0 γ t st P (s t |s, a 0:t-1 , σ 1:t )r(s t , a t ) +γ K s K P (s K |s, a, σ)V Bayes, * K,A -,ε (s K ) , ∀ s ∈ S.(5)
Here in Eq. ( 5), a 0:t-1 = (a 0 , . . . , a t-1 ) and σ 1:t = (σ 1 , . . . , σ t ) denote the sequences of actions and predictions, respectively, and P (s t |s 0 , a 0:t-1 , σ 1:t ) is the multi-step transition probability from initial state s 0 to state s t after t steps under the sequences of actions and predictions a 0:t-1 and σ 1:t , which satisfies the following recursive relation:
P (s t |s 0 , a 0:t-1 , σ 1:t ) = st-1∈S P (s t |s t-1 , a t-1 , σ t )P (s t-1 |s 0 , a 0:t-2 , σ 1:t-1 ), ∀ t. (6
)
The recursive form in Eq. ( 5) captures how predictions guide near-term planning over horizon K, with long-term value rolled into V Bayes, * K,A -,ε (s K ). Importantly, the corresponding Bellman operator is a contraction mapping with parameter γ K under the infinity norm, which guarantees the existence and uniqueness of the solution and enables efficient fixed-point computation.
The optimal Bayesian value function directly yields the optimal policy, as characterized below. The proof is provided in Appendix C.
this section cite: []

Section: Corollary 3.1 (Optimal Policy with Bayesian Value Function and Transition Predictions).
The optimal policy π * (• | s, σ) with K-step transition predictions σ satisfies:
{a ∈ A K | π * (a | s, σ) > 0} ⊆ arg max a∈A K K-1 t=0 γ t st P (s t |s, a 0:t-1 , σ 1:t )r(s t , a t ) + γ K s K P (s K |s, a, σ)V Bayes, * K,A -,ε (s K ) ∀ s ∈ S, σ ∈ Q K .(7)
This result shows that the optimal prediction-aware policy can be computed via a finite-horizon planning over σ, followed by terminal reward using the Bayesian value function V Bayes, * K,A -,ε . In effect, we have reduced the original infinite-horizon problem with high-dimensional predictions to a special form of fixedhorizon planning [19], which is tractable without explicitly augmenting the state space.
this section cite: ['b18']

Section: Analyzing the Value of Predictions
In this section, we examine how access to transition predictions improves decision-making in MDPs. Classical MDPs face a structural limitation: their value functions involve deeply nested max-over-E operations, which force agents to commit to fixed policies based on expected dynamics. We show that transition predictions alleviate this limitation by enabling a localized reordering of the max and E operators, allowing actions to adapt to realized transitions. We use a Bellman-Jensen Gap analysis on the Bayesian value function to characterize the value of predictions.
this section cite: []

Section: Bellman-Jensen Gap
We introduce the Bellman-Jensen Gap by comparing the following value functions.
this section cite: []

Section: Bellman Expansion of Optimal Value Function.
By recursively applying the Bellman optimality equation for classical discounted MDPs, the value function can be expressed in the following nested form [20]:
V * MDP (s 0 ) = max a0 r(s 0 , a 0 ) + γE σ * 1 max a1 r(s 1 , a 1 ) + γE σ * 2 max a2 [r(s 2 , a 2 ) + • • • ] ,(8)
where σ * t denotes the transition realization of s t at time t. Each expectation E σ * t is equivalent to taking the expectation over the next state s t ∼ P (• | s t-1 , a t-1 ), corresponding to the transition dynamics governed by σ * t . This formulation results in a deeply nested max-over-E structure, where the agent must choose an action that is optimal in expectation, without the ability to anticipate and adapt to future information.
In contrast, if the agent had access to perfect predictions of future transitions, it could defer action selection until those transitions are known, which allows a localized reordering of the max and E operators. We use a one-step prediction case to illustrate it:
this section cite: ['b19']

Section: Operator Reordering with One-Step Prediction.
Recall the Bayesian value function with K = 1, which can be expanded into the following recursive form:
V Bayes, * K=1,A,0 (s 0 ) = E σ * 1 max a0 r(s 0 , a 0 )+γE σ * 2 max a1 [r(s 1 , a 1 )+• • • ] .(9)
Observe that, with one-step prediction, each E σ * t operator is moved to the outer side of the neighborhood max at-1 operator. Intuitively, it provides the agent the ability to make decisions according to the transition prediction σ * t at time t. Mathematically, this localized reordering creates a local Jensen gap by exploiting Jensen's inequality due to E σ [max a f (s, a; σ)] ≥ max a E σ [f (s, a; σ)], where discrete maximization is a convex function, and f (s, a; σ) denotes the expected return under state s with action a and prediction σ. Since the Bayesian value function contains infinitely many such operator reordering in a recursive manner, we term it as the Bellman-Jensen Gap.
this section cite: []

Section: Maximal Bellman-Jensen Gap with Infinite-Step Prediction.
Such Bellman-Jensen Gaps reach the maximum when the prediction horizon is infinite. Formally, let V Bayes, * off ∈ R |S| denote the offline optimal Bayesian value function, where the agent has exact knowledge of all future transitions:
V Bayes, * off (s 0 ) = E σ * 1 E σ * 2 • • • max a0 r(s 0 , a 0 )+γ max a1 r(s 1 , a 1 )+γ max a2 [r(s 2 , a 2 )+• • • ] = lim k→∞ E σ * 1:k max a 0:k-1 k-1 t=0 γ t r(s t , a t ) ,(10)
where σ * 1:k is the sequence of realized transition kernels and a 0:k-1 is the action sequence over horizon k. The existence of the limit is shown in Appendix D.
Observe that, with infinitely long accurate prediction, all E σ * operators appear outside of any max a operator, which indicates that the agent can make the decision with full information of all future information, yielding the maximal Bellman-Jensen Gap defined as follows: Definition 4.1 (Maximal Bellman-Jensen Gap). For any state s ∈ S, we define the Maximal Bellman-Jensen Gap as ∆(s) := V Bayes, * off (s) -V * MDP (s), which quantifies the greatest possible performance gain from knowing exact future transitions.
The maximal Bellman-Jensen Gap characterizes the fundamental benefit that predictive information can offer in MDPs. It upper-bounds the value of any prediction by capturing the intrinsic benefits of operator reordering in the value function. Note that, this analytical framework naturally extends to other types of predictions. For example, by redefining σ to represent the prediction on reward realizations, the same Bellman-Jensen Gap analysis applies.
this section cite: []

Section: Closing the BellmanJensen Gap with Imperfect Predictions
We now leverage the BellmanJensen gap framework to analyze how imperfect predictions narrow the performance gap to the offline oracle. In particular, we derive explicit bounds on the suboptimality of a policy that uses finite-horizon, inaccurate, and partial action-coverage predictions.
The following theorem provides a finite-horizon performance bound that decomposes the suboptimality into three interpretable components, each capturing a distinct structural limitation. The proof is provided in Appendix E. Theorem 4.1 (Bellman-Jensen Performance Bound). Given any prediction with horizon K ≥ 1, predictable action set A -⊆ A and prediction errors ε, the performance gap between the predictionaware policy and the offline optimal policy satisfies:
max s∈S V Bayes, * off (s) -V Bayes, * K,A -,ε (s) ≤ C 1 γ K K log |A| (1 -γ) 6 5 (1 -γ 2K )
A1:loss due to finite prediction window
+ K j=1 γ j (1 -γ)(1 -γ K ) j A2:loss due to prediction error + C 2 ∞ t=1 γ t log(|A| t+1 -|A -| t+1 + 1)θ 2 max
A3:loss due to partial action predictability , where C 1 and C 2 are absolute constants, j denotes the prediction error at step j, defined as the Wasserstein-1 distance between the predicted and true transition distributions; parameter θ 2 max = max s,a0:t,t σ 2 (r(s t , a t |s 0 = s, a 0:t )) captures the variability of the reward, and σ(•) denotes the sub-Gaussian parameter.
Interpretation. Theorem 4.1 shows that the performance gap decomposes into three terms. The first term A 1 quantifies the performance loss due to the finite prediction horizon K. The factor γ K reflects that the benefit of predictions decreases exponentially with the horizon length, implying that even short-term predictions can capture significant potential improvement. When K -→ ∞, this loss term diminishes. The (1 -γ) 6/5 exponent arises from a refined dyadic horizon decomposition argument controlling the dependence on the discount factor (see Lemmas E.2 and E.3 for details). The term log |A| shows the number of actions slightly increases this gap, as a larger action space makes it statistically harder to identify the optimal action under uncertainty.
The second term A 2 captures the impact of prediction errors and disappears as the predicted transitions become accurate. Notably, it highlights that errors in subsequent steps have progressively smaller effects on overall performance, aligning with practical intuition. When j = for all j, we have A 2 = O( /(1 -γ) 2 ), which is independent of K, indicating that this term is primarily governed by the average prediction error, rather than the length of the prediction horizon.
The third term A 3 arises from partial action predictability and vanishes when all actions are predictable (i.e., A -= A). It is scaled by θ 2 max , indicating that greater reward uncertainty amplifies the Bellman-Jensen Gap. When A -= ∅, this term will not blow up and simplifies to O( log |A|θ 2 max (1 -γ) -3 2 ). Corollary 4.1. Given any prediction horizon K ≥ 1, if the predictions are perfectly accurate with j = 0 for all 1 ≤ j ≤ K, and all actions are predictable with A -= A, then the maximal performance gap satisfies
max s∈S (V Bayes, * off (s) -V Bayes, * K,A -,ε (s)) ≤ O(γ K √ K).
This result demonstrates that sufficiently accurate predictive informationeven over a finite horizoncan dramatically reduce the fundamental Bellman-Jensen Gap, bringing the agents performance significantly closer to the offline oracle benchmark. It characterizes the theoretical upper bound on the improvement that predictive signals can offer, revealing an exponential decay in the gap with horizon length K, up to a sublinear √ K correction term.
this section cite: []

Section: BOLA: Bayesian Offline Learning with Online Adaptation
Building on the theoretical understanding of the prediction-aware policy, in this section, we present a practical model-based algorithm for implementing the prediction-aware optimal policy.
The key insight from Theorem 3.1 and Corollary 3.1 is that optimal decisions can be achieved by combining short-horizon planning with a precomputed Bayesian value as the terminal function, which can effectively leverage predictive information without explicitly expanding the state space. This motivates the design of BOLA, a two-stage approach that cleanly separates offline learning from online adaptation to predictions.
this section cite: []

Section: BOLA Algorithm Overview
We propose BOLA (Bayesian Offline Learning with Online Adaptation), a model-based reinforcement learning algorithm designed to exploit transition predictions for efficient decision-making. BOLA decomposes learning and planning into two stages: (1) Offline Stage: Estimate the Bayesian value function V Bayes, * K,A -,ε (s) from samples by solving the Bellman equation in Eq. ( 5), and (2) Online Stage: At each decision point, observe real-time transition predictions σ and compute the optimal short-horizon action sequence using Eq. ( 7).
this section cite: []

Section: Offline Bayesian Value Function Learning.
To implement the prediction-aware Bellman operator from Eq. ( 5), we adopt a model-based learning approach inspired by classical MDPs. Specifically, we estimate the key quantities required to compute the Bayesian value function V Bayes, * K,A -,ε (s) via value iteration. These include: (1) the reward function r(s, a), (2) the distribution over K-step transition predictions P (σ), and (3) the multi-step transition kernel P (s | s, a, σ).
Importantly, the recursive structure in Eq. ( 6) allows us to avoid estimating the full K-step transition model directly. Instead, it suffices to estimate one-step transition probabilities P (s | s, a, σ) for predictable actions a ∈ A -, and standard MDP transitions P (s | s, a) for actions a / ∈ A -.
We assume access to a generative model [21,22]. For each state-action pair (s, a) with a / ∈ A -, the generative model allows us to generate N 1 independent next-state samples, denoted by {s i (s,a) } N1 i=1 .
For estimating the prediction distribution, we assume there exists a prediction oracle defined as follows: Assumption 5.1 (Prediction Oracle). The agent has access to a prediction oracle O pred that, upon query, returns independent samples σ i ∼ P (σ), where σ i = (σ i 1 , . . . , σ i K ) represents a K-step transition prediction vector drawn from the underlying distribution P (σ).
Under this assumption, we draw N 2 independent prediction samples from the oracle, denoted by {σ i = (σ i 1 , . . . , σ i K )} N2 i=1 , which are then used to estimate P (σ) via empirical frequencies. Using these samples, we estimate the relevant probabilities by empirical frequency:
P (s | s, a) = 1 N 1 N1 i=1 1(s i (s,a) = s ), ∀a ∈ A \ A -,(11)
P (σ) = 1 N 2 N2 i=1 1(σ i = σ), ∀σ ∈ Q K . (12
)
The reward function r(s, a) is obtained by sampling from each state-action pair (s, a) once.
With all model components estimated, we apply value iteration on the prediction-augmented Bellman operator in Eq. ( 5) to compute an approximate Bayesian value function V Bayes, * K,A -,ε . The contraction property of the Bellman operator ensures that this fixed-point iteration converges.
Online Adaptation. At each decision point, BOLA receives a K-step transition prediction vector σ = (σ 1 , . . . , σ K ). Given the current state s, BOLA first evaluates the multi-step transition probabilities by incorporating the prediction σ, and then solves the optimal action sequence through Eq. ( 7) using the precomputed Bayesian value function V Bayes, * K,A -,ε . This scheme enables real-time adaptation without solving high-dimensional value functions online. By combining offline long-term terminal value estimation with short-horizon prediction-aware planning [23], BOLA avoids state space augmentation and maintains computational tractability. Algorithm 1 summarizes the BOLA procedure. Determine the optimal policy π * (• | s, σ) by solving Eq. ( 7); 11: end for
this section cite: ['b20', 'b21', 'b22']

Section: Sample Complexity Guarantees
This section presents the sample complexity guarantees of Algorithm 1, more specifically, the learning of the Bayesian value function. The proof of the following theorem is presented in Appendix F. Theorem 5.1. For any given MDP, any confidence level δ ∈ (0, 1), any desired accuracy level ∈ (0, 1 1-γ ), a tradeoff parameter α ∈ (0, 1), and prediction horizon K ≥ 1, let D 1 be the number of samples drawn from the generative model, and D 2 be the number of samples drawn from the prediction oracle O pred . If
D 1 = C 1 |S|(|A| -|A -|) log (K|S|(|A| -|A -|)/δ) (1 -γ) 4 (1 -α) 2 2 + |S||A|, D 2 = C 2 log (4|S|/δ) (1 -γ) 2 (1 -γ K ) 2 α 2 2 ,
where C 1 , C 2 are absolute constants, then with probability at least 1 -δ, the learned Bayesian value function satisfies max
s | V Bayes, * K,A -,ε (s) -V Bayes, * K,A -,ε (s)| ≤ .(13)
Theorem 5.1 quantifies the sample complexity of BOLA, explicitly capturing the interplay between environment sampling and predictive adaptation. In particular, BOLAs total sample requirements decompose into two distinct regimes:
Environmentinteraction samples D 1 . The first requirement D 1 , arising from direct environment interactions, scales with |A| -|A -|, the number of unpredictable actions. This leads to a sample complexity which is strictly smaller than the classical dependence of O(|S||A| -2 ) [14,24], since increased predictability reduces the environment sampling burden. In the extreme case where all actions in the prediction horizon are predictable (A -= A), the dominant term of D 1 vanishes altogether (except for the reward function learning cost |S||A|), as the predictive model fully specifies the transition dynamics within the prediction horizon.
Prediction Oracle Samples D 2 . The second term D 2 represents the samples required from the predictive model, which exhibits a distinct scaling behavior: as the prediction horizon K grows, the required number of samples decreases due to the stronger contraction factor γ K in the Bayesian Bellman equation. When K ≥ O(log( 1 γ )), this term improves to (1 -γ) -2 , which is lower than that in model-based RL [15]. This highlights that when predictions are both comprehensive and with long enough horizon, BOLA can achieve lower sample complexity than classical MDP approaches.
Trade-off between the Two Sample Sources. Together, these two sampling regimes reveal a tradeoff parameterized by α: increasing α (more environment interaction) raises the environment sample requirement D 1 to O((1 -α) -2 ), while reducing the required number of samples D 2 from the prediction oracle. One can choose a reasonable α to trade off the sample requirements.
this section cite: ['b13', 'b23', 'b14']

Section: Numerical Studies
Although our emphasis is on theory and finitesample guarantees, we provide a small-scale empirical case study to demonstrate practical relevance. Specifically, we evaluate BOLA on a windfarm storage control task, where the operator minimizes energy imbalance penalties by charging or discharging a battery based on wind mismatch, price signals, and current state of charge (see Figure 1 for the setup, Appendix G for more details, and Appendix H for additional experiments). Specifically, Figure 2(a) illustrates the cumulative cost reduction achieved by BOLA under different prediction horizons, compared against a baseline MDP policy without prediction. As the prediction horizon K increases from 1 to 4, the cost reduction consistently improves, confirming that longer foresight enables the agent to better anticipate upcoming mismatches and price fluctuations. Notably, the largest marginal improvement occurs at K = 1, indicating that even a short look-ahead can significantly enhance decision-making performance. 0 500 1000 1500 2000 2500 3000 Time Slot 0 1000 2000 3000 4000 Cost Reduction ($) Prediction Horizon K=4 Prediction Horizon K=3 Prediction Horizon K=2 Prediction Horizon K=1 No Predictions (a) Cost Reduction with Predictions 0 5 10 15 20 25 30 Relative Prediction Error (%) 0 1000 2000 3000 Cost Reduction ($) Prediction Horizon K=4 Prediction Horizon K=3 Prediction Horizon K=2 Prediction Horizon K=1 No Prediction Baseline (b) Robustness on Prediction Errors Figure 2: Storage Control Performance. (a) Cumulative cost reduction for different prediction horizons: longer prediction horizon yields greater savings over the no-prediction baseline. (b) Robustness to prediction noise: cost savings decline roughly linearly with error, yet all predictive policies outperform the baseline even at 30% noise.
degrade more gradually as noise increases. Nevertheless, all prediction-based policies consistently outperform the no-prediction baseline, even when the relative error reaches 30%.
this section cite: []

Section: Conclusion
In this work, we study the theoretical value of transition predictions in sequential decision-making. We propose a prediction-augmented MDP framework, characterize the benefit of predictions via the Bellman-Jensen Gap, and develop a tractable model-based RL algorithm with sample complexity guarantees. A natural future direction is to extend our results to the model-free setting.
Limitations and Future Directions. In prediction-augmented MDP, we consider fixed-horizon planning where the agent receives a K-step prediction and plans a K-step sequence of actions. For sequential decision-making problems with predictions, another popular framework is called the receding-horizon control, where the agent receives a K-step prediction but only plans a single-step action instead of a sequence of K actions. Intuitively, using receding-horizon control could be more beneficial to the agent than fixed-horizon planning, since the agent does not have to commit to a sequence of actions and can adaptively choose actions based on the new realizations of the states and the predictions. Further investigating the advantage of prediction-augmented MDPs with receding-horizon control is among the future directions of this work. On the theoretical side, we have established the first upper bounds on BOLAs sample complexity, but it remains open whether these can be tightened or matched by lower bounds. In particular, refined variance-based techniques (e.g., refined concentration for the multi-step Bayesian operator) may yield stronger guarantees.
this section cite: []

Section: References
Ref_id:b0 Title: Reinforcement Learning: An Introduction Year: (2018)
Ref_id:b1 Title: Modelbased reinforcement learning for Atari Year: ()
Ref_id:b2 Title: Reinforcement learning in robotics: A survey Year: (2013)
Ref_id:b3 Title: Deep reinforcement learning for intelligent transportation systems: A survey Year: (2020)
Ref_id:b4 Title: Reinforcement learning for selective key applications in power systems: Recent advances and future challenges Year: (2022)
Ref_id:b5 Title: Model-based reinforcement learning: A survey Year: (2023)
Ref_id:b6 Title: Reinforcement learning: A survey Year: (1996)
Ref_id:b7 Title: Reinforcement learning in economics and finance Year: (2021)
Ref_id:b8 Title: A review on the selected applications of forecasting models in renewable power systems Year: (2019)
Ref_id:b9 Title: Analysis of temporal-difference learning with function approximation Year: (1996)
Ref_id:b10 Title: A finite time analysis of temporal difference learning with linear function approximation Year: (2018)
Ref_id:b11 Title: Finite-time error bounds for linear stochastic approximation and TD-learning Year: (2019)
Ref_id:b12 Title: Target network and truncation overcome the deadly triad in Q-learning Year: (2023)
Ref_id:b13 Title: On the sample complexity of reinforcement learning with a generative model Year: (2012)
Ref_id:b14 Title: Breaking the sample size barrier in model-based reinforcement learning with a generative model Year: (2020)
Ref_id:b15 Title: Overcoming the curse of dimensionality in reinforcement learning through approximate factorization Year: ()
Ref_id:b16 Title: Predictive control and regret analysis of non-stationary MDP with look-ahead information Year: ()
Ref_id:b17 Title: Efficiently solving discounted MDPs with predictions on transition matrices Year: (2025)
Ref_id:b18 Title: Model predictive control: Theory and practicea survey Year: (1989)
Ref_id:b19 Title: The bellman equation for minimizing the maximum cost Year: (1989)
Ref_id:b20 Title: A sparse sampling algorithm for nearoptimal planning in large Markov decision processes Year: (2002)
Ref_id:b21 Title: On the sample complexity of reinforcement learning Year: (2003)
Ref_id:b22 Title: Planning theory Year: (2017)
Ref_id:b23 Title: Breaking the sample complexity barrier to regret-optimal model-free reinforcement learning Year: (2021)
Ref_id:b24 Title: Model-based reinforcement learning with a generative model is minimax optimal Year: (2020)
Ref_id:b25 Title: Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model Year: (2013)
Ref_id:b26 Title: Near-optimal time and sample complexities for solving Markov decision processes with a generative model Year: (2018)
Ref_id:b27 Title: Minimax regret bounds for reinforcement learning Year: (2017)
Ref_id:b28 Title: Provably efficient reinforcement learning with linear function approximation Year: (2020)
Ref_id:b29 Title: Towards tight bounds on the sample complexity of average-reward MDPs Year: (2021)
Ref_id:b30 Title: Near-optimal sample complexity bounds for constrained MDPs Year: (2022)
Ref_id:b31 Title: Model-based bayesian reinforcement learning in partially observable domains Year: (2008)
Ref_id:b32 Title: Provably efficient reinforcement learning in partially observable dynamical systems Year: (2022)
Ref_id:b33 Title: Optimistic mle: A generic model-based algorithm for partially observable sequential decision making Year: (2023)
Ref_id:b34 Title: Optimal Learning: Computational procedures for Bayes-adaptive Markov decision processes Year: (2002)
Ref_id:b35 Title: Bayes-adaptive pomdps Year: (2007)
Ref_id:b36 Title: Efficient bayes-adaptive reinforcement learning using sample-based search Year: (2012)
Ref_id:b37 Title: Lookahead-bounded Q-learning Year: (2020)
Ref_id:b38 Title: The role of lookahead and approximate policy evaluation in reinforcement learning with linear value function approximation Year: (2025)
Ref_id:b39 Title: Beyond the one-step greedy approach in reinforcement learning Year: (2018)
Ref_id:b40 Title: Gal Chechik, and Gal Dalal. Planning and learning with adaptive lookahead Year: (2023)
Ref_id:b41 Title: Online planning with lookahead policies Year: (2020)
Ref_id:b42 Title: Can Q-learning be improved with advice? Year: (2022)
Ref_id:b43 Title: Beyond black-box advice: Learningaugmented algorithms for MDPs with Q-value predictions Year: (2024)
Ref_id:b44 Title: Leveraging predictions in smoothed online convex optimization via gradient-based algorithms Year: (2020)
Ref_id:b45 Title: Online optimization with predictions and switching costs: Fast algorithms and the fundamental limit Year: (2020)
Ref_id:b46 Title: Online convex optimization using predictions Year: (2015)
Ref_id:b47 Title: Online optimization with predictions and non-convex losses Year: (2020)
Ref_id:b48 Title: The power of predictions in online control Year: (1994)
Ref_id:b49 Title: On the regret analysis of online LQR control with predictions Year: (2021)
Ref_id:b50 Title: Perturbation-based regret analysis of predictive control in linear time varying systems Year: (2021)
Ref_id:b51 Title: Bounded-regret MPC via perturbation analysis: Prediction error, constraints, and nonlinearity Year: (2022)
Ref_id:b52 Title: Performance analysis of online anticipatory algorithms for large multistage stochastic integer programs Year: (2007)
Ref_id:b53 Title: Markov Decision Processes: Discrete Stochastic Dynamic Programming Year: (2014)
Ref_id:b54 Title: Neuro-Dynamic Programming Year: (1996)
Ref_id:b55 Title: Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales Year: (1922)
Ref_id:b56 Title: Principles of Mathematical Analysis Year: (1964)
Ref_id:b57 Title: High-Dimensional Probability: An Introduction with Applications in Data Science Year: (2018)
Ref_id:b58 Title: Computational optimal transport Year: (2019)
Ref_id:b59 Title:  Year: (2022-12-20)
Ref_id:b60 Title: Self-improving online storage control for stable wind power commitment Year: (2024)
Ref_id:b61 Title: Near-optimal reinforcement learning in factored MDPs Year: (2014)
Ref_id:b62 Title: Reinforcement learning with general value function approximation: Provably efficient approach via bounded Eluder dimension Year: (2020)
Ref_id:b63 Title: Reinforcement learning for UAV attitude control Year: (2019)
Ref_id:b64 Title: An optimal solutions-guided deep reinforcement learning approach for online energy storage control Year: (2024)
Ref_id:b65 Title: Sample-adaptive robust economic dispatch with statistically feasible guarantees Year: (2023)
Ref_id:b66 Title: Deep reinforcement learning for economic dispatch of virtual power plant in internet of energy Year: (2020)
Ref_id:b67 Title: Temperature control of a commercial building with model predictive control techniques Year: (2014)
Ref_id:b68 Title: A control strategy based on deep reinforcement learning under the combined wind-solar storage system Year: (2021)
