Title: Improving LLM General Preference Alignment via Optimistic Online Mirror Descent
Abstract: Reinforcement learning from human feedback (RLHF) has demonstrated remarkable effectiveness in aligning large language models (LLMs) with human preferences. Many existing alignment approaches rely on the Bradley-Terry (BT) model assumption, which assumes the existence of a ground-truth reward for each promptresponse pair. However, this assumption can be overly restrictive when modeling complex human preferences. In this paper, we drop the BT model assumption and study LLM alignment under general preferences, formulated as a two-player game. Drawing on theoretical insights from learning in games, we integrate optimistic online mirror descent into our alignment framework to approximate the Nash policy. Theoretically, we demonstrate that our approach achieves an O(T -1 ) bound on the duality gap, improving upon the previous O(T -1/2 ) result. Meanwhile, it enjoys a linear convergence rate in the last iterate, a property not achieved by previous methods. More importantly, we implement our method and show through experiments that it outperforms state-of-the-art RLHF algorithms across multiple representative benchmarks.

Section: Introduction and Related Works
Reinforcement learning from human feedback (RLHF) has played a pivotal role in aligning large language models (LLMs) with human preferences. The goal of RLHF is to fine-tune LLMs to generate responses that are preferred by humans. It has been successfully deployed in state-of-the-art models, including Instruct-GPT [Ouyang et al., 2022] and Claude [Bai et al., 2022b]. The first RLHF framework for LLMs was developed by Ouyang et al. [2022], where after the pre-training stage, the LLM is fine-tuned to maximize the reward signal from a reward model using the proximal policy optimization (PPO) algorithm [Schulman et al., 2017]. This pipeline requires training both the reward model and the policy model. In addition, policy gradient approaches such as PPO often exhibit high variance and instability during training [Peng et al., 2023], leading to increased computational costs.
To develop a more stable and computationally lightweight alignment approach, Rafailov et al. [2024b] propose the Direct Preference Optimization (DPO) algorithm, which directly trains the LLM on a preference dataset and bypasses the need for a reward model. DPO uses an offline preference dataset, and since its development, a line of research has explored different exploration strategies and proposed online direct preference alignment algorithms [Xiong et al., 2024, Xie et al., 2024, Dong et al., 2024, Yuan et al., 2024]. All these methods assume that human preferences can be modeled using the Bradley-Terry (BT) model, where a reward function R * exists such that, for any prompt x and response pair (y 1 , y 2 ), the preference between y 1 and y 2 satisfies:
P(y 1 ≻ y 2 | x) = σ(R * (x, y 1 ) -R * (x, y 2 )),
where σ(z) = 1 1+exp(-z) is the sigmoid function. However, the existence of a reward function and the BT model are strong assumptions that can be overly restrictive when modeling complex human preferences. For example, the preference signals in the BT model are always transitive: if A is preferred to B and B is preferred to C, then A must always be preferred to C. This transitive property contradicts evidence from human decision-making [May, 1954, Tversky, 1969], especially when preferences are at the population level and aggregated from different human groups [May, 1954, Ye et al., 2024]. Furthermore, the limitations of the BT model have also been observed in RLHF practice. Jiang et al. [2023] show that a preference model with 0.4B parameters achieves performance comparable to Llama-2-13B-based reward models. Ye et al. [2024] train a BT reward model and a preference model separately using the same base model and preference dataset, and their results demonstrate that the preference model consistently outperforms the reward model on Reward-Bench [Lambert et al., 2024] under different base models. These findings motivate us to drop the BT model assumption and instead consider general preferences.
In this work, we study the problem of aligning LLMs with general preferences and formulate it as a two-player zero-sum game. Our objective is to approximate the Nash policy of the game, which ensures a win rate of at least 50% against any other policy. As established in the game theory literature [Bai et al., 2020, Liu et al., 2021], self-play algorithms have proven to be highly effective in approximating Nash policies. Building on this, we aim to propose a novel online RLHF algorithm that further leverages the self-play strcture to enhance general preference alignment for LLMs. Our contributions are summarized as follows.
Contributions. We propose a novel online general preference alignment algorithm, Optimistic Nash Policy Optimization (ONPO). Inspired by recent advancements in game theory, our algorithm integrates optimistic online mirror descent [Rakhlin andSridharan, 2013, Syrgkanis et al., 2015] into the self-play framework. By utilizing a reward predictor in a two-step update strategy, ONPO more effectively leverages the self-play mechanism and achieves a faster convergence rate of O(T -1 ) on the duality gap, improving upon the previous O(T -1/2 ) result. Moreover, ONPO enjoys a linear convergence rate in the last iterate, a property not achieved by previous methods such as INPO [Zhang et al., 2024].
ONPO can be efficiently implemented by directly minimizing a loss objective on a preference dataset, making it computationally lightweight in practice. We evaluate ONPO on several representative benchmarks, comparing it with state-of-the-art general preference alignment algorithms. Experimental results demonstrate that ONPO consistently outperforms or achieves performance comparable to the baselines across different base models and benchmarks. Notably, on the AlpacaEval 2.0 benchmark [Li et al., 2023a], ONPO achieves a 21.2% and 9.9% relative improvement over the strongest baseline when using Mistral-Instruct and Llama-3-8B as the base models, respectively.
this section cite: ['b37', 'b37', 'b46', 'b38', 'b58', 'b34', 'b50', 'b34', 'b18', 'b59', 'b23', 'b1', 'b63']

Section: Preliminary
Problem Setup. We study the contextual formulation which is extensively used in previous RLHF literature [Rafailov et al., 2024b, Xiong et al., 2024]. The prompt x ∈ X is sampled from an unknown prompt distribution d 1 . Y is the response space and an LLM is characterized by a policy π : X → ∆(Y) which outputs the response probability given the context. For any policy π, we use E π to denote the expectations under π.
this section cite: []

Section: General Preferences.
In this work, we drop the BT model assumption [Bradley and Terry, 1952] and focus on directly aligning LLMs with general preferences. To this end, we define a general preference oracle as follows: Definition 1 (General Preference Oracle). There exists a preference oracle P : X × Y → Y → [0, 1], which can be queried to obtain the binary preference signal:
z ∼ Ber P(y 1 ≻ y 2 | x)),
where z = 1 indicates y 1 is preferred to y 2 , and z = 0 indicates the opposite.
Unlike the BT model assumption, which assumes the existence of a reward function R * for each x and y, the general preference oracle always compares y 1 to another y 2 . This setup aligns with practical scenarios, where it is often easier for users to compare two responses than to assign an absolute score to a single response. Since the preference signal always involves two responses, potentially come from two different policies, we formulate the LLM alignment problem as a two-player zero-sum game. The objective of this game is the expected win rate between the two players:
J(π 1 , π 2 ) := E x∼d1 E y 1 ∼π1,y 2 ∼π2 P(y 1 ≻ y 2 | x) .
Here π 1 is the policy of the max-player, aiming to maximize the objective, while π 2 is the policy of the min-player, aiming to minimize it. Nash Policies and Duality Gap. Our learning goal is to find the Nash equilibrium of the game, which is defined as:
π * 1 , π * 2 := argmax π1 argmin π2 J(π 1 , π 2 ).
Due to the symmetric nature of the game, the Nash policies for both players are identical, i.e., π * 1 = π * 2 = π * , and the game value is J(π * , π * ) = 0.5. Since Nash policies are the best responses to each other, for any policy π, we have J(π * , π) ≥ 0.5, indicating that the Nash policy will not lose to any other policy. To quantify how well a policy π approximates π * , we define the duality gap as:
DualGap(π) := max π1 J(π 1 , π) -min π2 J(π, π 2 ).
The duality gap is non-negative and DualGap(π) = 0 if and only if π = π * . Hence, our goal is to find a policy that minimizes the duality gap. Once we achieve DualGap(π) ≤ ϵ, we say that π is an ϵ-approximate Nash policy.
this section cite: ['b4']

Section: Algorithm
In this section, we begin by briefly reviewing the self-play algorithm with online mirror descent (OMD) updates, which is used in previous general preference alignment algorithm [Zhang et al., 2024]. Next, we present our proposed algorithm, which leverages the faster convergence properties of optimistic OMD, inspired by advancements in game theory [Rakhlin andSridharan, 2013, Syrgkanis et al., 2015]. Through theoretical analysis, we show that our approach achieves an improved bound on the duality gap and a linear convergence rate in the last iterate. Finally, we describe the implementation of our algorithm. Following Azar et al. [2024], Zhang et al. [2024], we omit the context x throughout the rest of the paper since each context is independent.
this section cite: ['b63', 'b63']

Section: Self-play Algorithm with OMD Update
Self-play algorithms are widely used in approximating the Nash policy [Bai et al., 2020, Liu et al., 2021]. The key idea is to let the policy play against itself, enabling iterative self-improvement. The algorithm is performed in an online manner, with each iteration using online mirror descent (OMD) to update the policy. Specifically, at iteration t, we find the policy that maximizes the following objective:
π t+1 = argmax π ⟨π, r t ⟩ - 1 η KL(π∥π t ),(1)
where r t (y) = P(y ≻ π t ) = E y ′ ∼πt [P(y ≻ y ′ )] is the expected win rate of response y against the current policy π t , and η > 0 is the learning rate. This objective ensures that π t+1 not only aims to maximize the win rate over π t but also remains close to π t , as measured by the KL divergence term. The stability introduced by the KL regularization is critical for achieving a sublinear regret bound. Without this regularization, one can construct examples where the algorithm suffers from linear regret, which is undesirable [Lattimore and Szepesvári, 2020].
We can show that the uniform mixture of π 1:T achieves an O(T -1/2 ) duality gap, as stated in the following theorem. The proof is deferred to Appendix C.1.
Theorem 1. Let D = max π KL(π∥π 1 ) and π = unif{π 1 , • • • , π T }. Self-play algorithm in Eq. 1with η = D T satisfies:
DualGap(π) ≤ 4 √ D √ T .
In RLHF practice, we typically use a small number of iterations (e.g., T = 3), so the uniform mixture policy π can be directly deployed. Unlike Zhang et al. [2024], which adopts a KL-regularized game formulation, we directly use the win rate between two policies as the game objective. This formulation has two advantages: 1. The Nash policy in our formulation guarantees at least a 50% win rate against any other policy, which aligns directly with the goal of general preference alignment. In contrast, the Nash policy in the KL-regularized game only ensures this when the KL terms are negligible. 2. In the KL-regularized setting, the analysis of the OMD algorithm relies on a coverage assumption that the log-density ratio log π(y)/ log π ref (y) is uniformly bounded for all π, and the regret bound depends linearly on the coverage coefficient. Our analysis avoids this assumption and its associated dependence by directly optimizing the win rate, resulting in improved results.
this section cite: ['b1', 'b24', 'b63']

Section: Optimistic Nash Policy Optimization
While self-play with OMD update already achieves an O( √ T ) regret bound, which is near-optimal in many online learning scenarios, there is still room for improvement by better leveraging the self-play structure. Recent advancements in learning in games [Rakhlin andSridharan, 2013, Syrgkanis et al., 2015] demonstrate that a faster convergence rate of O(T -1 ) can be achieved when both players adopt optimistic OMD update. In this subsection, we introduce how to integrate optimistic OMD into the self-play algorithm, resulting in an algorithm called Optimistic Nash Policy Optimization (ONPO).
The key idea of optimistic OMD is to incorporate a reward or loss predictor at each iteration. Recall that in OMD update, we use the expected win rate over the current policy π t as the reward vector r t to compute π t+1 . While in optimistic OMD, the learner utilizes a reward predictor m t and adopts a two-step update strategy:
π t = argmax π ⟨π, m t ⟩ - 1 η KL(π∥π ′ t ) π ′ t+1 = argmax π ⟨π, r t ⟩ - 1 η KL(π∥π ′ t ).
Here π t aims to maximize the reward predictor m t and the auxiliary policy π ′ t+1 is updated after observing the actual reward r t . The word "optimistic" comes from that the learner believes that the predictor m t provides a good approximation of the true reward r t .
Next, we describe how to apply optimistic OMD in our self-play algorithm. In both OMD and optimistic OMD, the KL regularization term is consistently used to ensure that the next policy remains close to the previous policies. This regularization provides stability, making it reasonable to assume that the change from π t to π t+1 is small. Based on this observation, we directly use the reward information from the previous iteration as the predictor, i.e., let
m t = r t-1 = E y ′ ∼πt-1 [P(y ≻ y ′ )].
In the following theorem, we demonstrate that ONPO achieves an O(1/T ) duality gap, improving over the previous
O(1/ √ T ) result. Theorem 2. Let D = max π KL(π∥π ′ 1 ) and π = unif{π 1 , • • • , π T }, ONPO algorithm with η = min{ 1 2 , √ D} satisfies: DualGap(π) ≤ 4 √ D T .
Here, π ′ 1 = π 1 is the initialization policy. Theoretically, π ′ 1 can be set as a uniform policy, in which case D is bounded by log |Y|. In RLHF practice, π ′ 1 is typically a supervised fine-tuned policy. The proof is provided in Appendix C.2. The key to achieving the O(1/T ) rate lies in the regret bounded by variation in utilities (RVU) property of optimistic OMD. Specifically, the stability terms ∥r t -r t-1 ∥ 2 ∞ are canceled out by the negative term -∥π t -π t-1 ∥ 2 1 , which arises from the selfplay mechanism where r t represents the win rate over π t . Additionally, the stability inherent in optimistic OMD ensures that the learned policy remains close to the initial policy. This aligns with the motivation behind incorporating KL regularization into the game objective in prior works [Munos et al., 2023, Zhang et al., 2024]. Since our update rule already implicitly enforces this stability, explicit regularization in the game objective is unnecessary.
Although the uniform mixture policy is implementable, a more common choice in RLHF practice is to directly deploy the last policy. In the following theorem, we show that ONPO also achieves a linear convergence rate in the last iterate. Theorem 3. Assume that the Nash policy π * is unique, with η ≤ 1 8 , we have KL(π * ∥π t ) ≤ O(C -t ) where C > 1 is a constant.
The proof follows directly from the analysis of Theorem 3 in Wei et al. [2020]. Zhang et al. [2024] also demonstrate that self-play with OMD achieves last-iterate convergence. However, their result relies on the strong convexity induced by the KL regularization terms in their game objective and does not apply to our formulation. This highlights another key advantage of using optimistic OMD: it not only improves the duality gap bound but also ensures last-iterate convergence without requiring explicit regularization.
this section cite: ['b35', 'b54', 'b63']

Section: Implementation of ONPO
In this subsection, we describe the implementation of ONPO with query access to the preference oracle P. The primary challenge in implementing ONPO lies in computing r t (y), which involves taking an expectation over the entire policy π t . Fortunately, this challenge can be addressed by avoiding the direct estimation of r t (y) and instead relying on binary preference feedback between responses.
To achieves this, our goal is to design a loss function that does not involve P(y ≻ π t ) for policy optimization. We focus on obtaining the loss objective for π t here and the derivation for π ′ t is similar. The key observation is that, π t has a closed-form solution which satisfies ∀y, y ′ ∈ Y,
log π t (y) π t (y ′ ) -log π ′ t (y) π ′ t (y ′ ) = η (P(y ≻ π t-1 ) -P(y ′ ≻ π t-1 ).
Therefore, similar to the techniques used in Azar et al. [2024], Zhang et al. [2024], solving π t is equivalent to finding the minimizer of the following loss function:
E y,y ′ ∼π t-1 gt(π, y, y ′ ) -η P(y ≻ πt-1) -P(y ′ ≻ πt-1) 2 .
where g t (π, y, y ′ ) = log π(y) π(y ′ ) -log π ′ t (y) π ′ t (y ′ ) . Since the inside win rate term is with respect to π t-1 and we also have an expectation over π t-1 outside, the loss function can be further written as
E y,y ′ ∼πt-1,yw,y l ∼λp(y,y ′ ) g t (π, y w , y l ) - η 2 2 ,
where λ p is the preference distribution [Calandriello et al., 2024]:
λ p (y, y ′ ) =
(y, y ′ ) with probability P(y ≻ y ′ ) (y ′ , y) with probability 1 -P(y ≻ y ′ ).
To calculate the loss function, we only need the access to sample from the current policy, which is standard and easy to implement in practice. Putting everything together, the implementation of ONPO is summarized in Algorithm 1.
In the beginning, we initialize π ′ 1 and π 1 with the supervised fine-tuned policy π SFT . At each iteration t, we sample responses from the current policy π t and use the preference feedback from the oracle P to construct the dataset D t . Then we can directly minimize the corresponding loss functions on D t to find π ′ t+1 and π t+1 respectively. We use the last iteration policy π T as the output policy, which is consistent with online RLHF practice [Dong et al., 2024, Wu et al., 2024, Zhang et al., 2024].
this section cite: ['b63', 'b5', 'b12']

Section: Discussion
In this section, we discuss the differences between ONPO and other general preference alignment methods.
Algorithm 1 Implementation of ONPO 1: Input: Number of iterations T , learning rate η, preference oracle P, supervised fine-tuned policy π SFT . 2: Initialize π ′ 1 ← π SFT , π 1 ← π SFT . 3: for iteration t = 1, 2, . . . , T -1 do 4: Sample response pairs from the current policy π t : {y (i) 1 , y (i) 2 } n i=1 ∼ π t . 5: Construct preference dataset D t = {y (i) w , y (i) l } n i=1 with feedback from the oracle P. 6:
Calculate π ′ t+1 as:
π ′ t+1 = argmin π E yw,y l ∼Dt g t (π, y w , y l ) - η 2 2 . 7:
Calculate π t+1 as:
π t+1 = argmin π E yw,y l ∼Dt g t+1 (π, y w , y l ) - η 2 2 .
8: end for 9: Output π T .
IPO. Azar et al. [2024] is the first to address general preference alignment in LLMs. The optimization objective of IPO is:
max π E y∼π,y ′ ∼µ [P(y ≻ y ′ )] -τ KL(π∥π ref ),
where µ is a fixed policy. From a game-theoretic perspective, the goal of IPO is to find the best response to µ. However, this approach only ensures that the learned policy outperforms µ, which leaves the possibility that another policy could outperform the learned policy. In contrast, our approach focuses on learning the Nash policy in a two-player game. This provides stronger theoretical guarantees, as the Nash policy will not lose to any other policy.
Nash-MD. Munos et al. [2023] is the first to formulate the alignment problem as a two-player zero-sum game. Their game objective includes KL regularization terms, which ensure that the player's policy remains close to the reference policy π ref .
The KL terms are weighted by a parameter τ . They proposed an iterative algorithm, Nash-MD, to learn the Nash policy of the game. At each iteration t, the policy is updated as:
π t+1 = argmax π P(π ≻ π ′ t ) - 1 η t KL(π, π ′ t ),
where π ′ t is a geometric mixture policy of the current policy π t and the reference policy π ref :
π ′ t (y) = π t (y) 1-ηtτ π ref (y) ηtτ y ′ π t (y ′ ) 1-ηtτ π ref (y ′ ) ηtτ
. Nash-MD requires sampling from the mixture policy π ′ t . However, the response space Y is often exponentially large, making the exact computation of π ′ t intractable. To address this, Munos et al. [2023] propose sampling from an approximate policy. The theoretical guarantees of this approximation remain unclear. In contrast, our approach only requires sampling from the current policy π t , which is straightforward to implement in practice.
Online IPO. Calandriello et al. [2024] propose the online IPO population loss:
E y,y ′ ∼SG[π] yw,y l ∼λp(y,y ′ ) log π(y w )π ref (y l ) π(y l )π ref (y w ) - 1 2τ 2 ,
where SG is the stop-gradient operator, which prevents gradients from propagating through the data-generation process. Unlike the offline IPO approach, which always samples from a fixed policy µ, online IPO leverages responses generated by the current policy π.
Since the policy π is updated throughout training, policy gradient methods are used to minimize the objective. However, as discussed earlier, policy gradient methods in RLHF have limitations, including being resource-intensive and unstable to train. In contrast, ONPO avoids these challenges by directly minimizing a loss function over a preference dataset, offering a more stable and efficient implementation.
this section cite: ['b0', 'b35', 'b35', 'b5']

Section: References
Ref_id:b0 Title: A general theoretical paradigm to understand learning from human preferences Year: (2024)
Ref_id:b1 Title: Near-optimal reinforcement learning with self-play Year: (2020)
Ref_id:b2 Title: Near-optimal learning of extensive-form games with imperfect information Year: (2022)
Ref_id:b3 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b4 Title: Rank analysis of incomplete block designs: I. the method of paired comparisons Year: (1952)
Ref_id:b5 Title: Human alignment of large language models through online preference optimisation Year: (2024)
Ref_id:b6 Title: Hedging in games: Faster convergence of external and swap regrets Year: (2020)
Ref_id:b7 Title: On the weaknesses of reinforcement learning for neural machine translation Year: (2019)
Ref_id:b8 Title: Deep reinforcement learning from human preferences Year: (2017)
Ref_id:b9 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b10 Title: Near-optimal no-regret algorithms for zero-sum games Year: (2011)
Ref_id:b11 Title: Near-optimal no-regret learning in general games Year: (2021)
Ref_id:b12 Title: Rlhf workflow: From reward modeling to online rlhf Year: (2024)
Ref_id:b13 Title: The llama 3 herd of models Year: (2024)
Ref_id:b14 Title: Length-controlled alpacaeval: A simple way to debias automatic evaluators Year: (2024)
Ref_id:b15 Title: Model alignment as prospect theoretic optimization Year: (2024)
Ref_id:b16 Title: Adaptive game playing using multiplicative weights Year: (1999)
Ref_id:b17 Title: Koala: A dialogue model for academic research Year: (2023-04)
Ref_id:b18 Title: Llm-blender: Ensembling large language models with pairwise ranking and generative fusion Year: (2023)
Ref_id:b19 Title: V-learning-a simple, efficient, decentralized algorithm for multiagent rl Year: (2021)
Ref_id:b20 Title: Rl with kl penalties is better viewed as bayesian inference Year: (2022)
Ref_id:b21 Title: Model-free learning for two-player zero-sum partially observable markov games with perfect recall Year: (2021)
Ref_id:b22 Title: Faster algorithms for extensive-form game solving via improved smoothing functions Year: (2020)
Ref_id:b23 Title: Evaluating reward models for language modeling Year: (2024)
Ref_id:b24 Title: Bandit algorithms Year: (2020)
Ref_id:b25 Title: Last-iterate convergence in extensive-form games Year: (2021)
Ref_id:b26 Title: From live data to high-quality benchmarks: The arena-hard pipeline Year: (2024)
Ref_id:b27 Title: Alpacaeval: An automatic evaluator of instruction-following models Year: (2023)
Ref_id:b28 Title: Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models Year: (2023)
Ref_id:b29 Title: Measuring how models mimic human falsehoods Year: (2021)
Ref_id:b30 Title: A sharp analysis of model-based reinforcement learning with self-play Year: (2021)
Ref_id:b31 Title: Comal: A convergent meta-algorithm for aligning llms with general preferences Year: (2024)
Ref_id:b32 Title: Cycles in zero-sum differential games and biological diversity Year: (2018)
Ref_id:b33 Title: Provably efficient reinforcement learning in decentralized generalsum markov games Year: (2023)
Ref_id:b34 Title: Intransitivity, utility, and the aggregation of preference patterns Year: (1954)
Ref_id:b35 Title: Nash learning from human feedback Year: (2023)
Ref_id:b36 Title: Gpt-4 technical report Year: (2023)
Ref_id:b37 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b38 Title: Stabilizing rlhf through advantage model and selective rehearsal Year: (2023)
Ref_id:b39 Title: From r to q * : Your language model is secretly a q-function Year: (2024)
Ref_id:b40 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2024)
Ref_id:b41 Title: Optimization, learning, and games with predictable sequences Year: (2013)
Ref_id:b42 Title: Gpqa: A graduate-level google-proof q&a benchmark Year: (2023)
Ref_id:b43 Title: Direct nash optimization: Teaching language models to self-improve with general preferences Year: (2024)
Ref_id:b44 Title: Online and bandit algorithms for nonstationary stochastic saddle-point optimization Year: (2019)
Ref_id:b45 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b46 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b47 Title: Multi-turn reinforcement learning from preference human feedback Year: (2024)
Ref_id:b48 Title: Fast convergence of regularized learning in games Year: (2015)
Ref_id:b49 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b50 Title: Intransitivity of preferences Year: (1969)
Ref_id:b51 Title: Self-instruct: Aligning language models with self-generated instructions Year: (2022)
Ref_id:b52 Title: Mmlu-pro: A more robust and challenging multi-task language understanding benchmark Year: (2024)
Ref_id:b53 Title: Online reinforcement learning in stochastic games Year: (2017)
Ref_id:b54 Title: Linear last-iterate convergence in constrained saddle-point optimization Year: (2020)
Ref_id:b55 Title: Multi-step alignment as markov games: An optimistic online gradient descent approach with convergence guarantees Year: (2025)
Ref_id:b56 Title: Self-play preference optimization for language model alignment Year: (2024)
Ref_id:b57 Title: Exploratory preference optimization: Harnessing implicit q*-approximation for sample-efficient rlhf Year: (2024)
Ref_id:b58 Title: Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint Year: (2024)
Ref_id:b59 Title: A theoretical analysis of nash learning from human feedback under general kl-regularized preference Year: (2024)
Ref_id:b60 Title: Self-rewarding language models Year: (2024)
Ref_id:b61 Title: Rank responses to align language models with human feedback without tears Year: (2023)
Ref_id:b62 Title: Hellaswag: Can a machine really finish your sentence Year: (2019)
Ref_id:b63 Title: Iterative nash policy optimization: Aligning llms with general preferences via no-regret learning Year: (2024)
Ref_id:b64 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2024)
Ref_id:b65 Title: Regret minimization in games with incomplete information Year: (2007)
