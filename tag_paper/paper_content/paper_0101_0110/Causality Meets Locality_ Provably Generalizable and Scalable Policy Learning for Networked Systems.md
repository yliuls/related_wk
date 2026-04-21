Title: Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems
Abstract: Large-scale networked systems, such as traffic, power, and wireless grids, challenge reinforcement-learning agents with both scale and environment shifts. To address these challenges, we propose GSAC (Generalizable and Scalable Actor-Critic), a framework that couples causal representation learning with meta actor-critic learning to achieve both scalability and domain generalization. Each agent first learns a sparse local causal mask that provably identifies the minimal neighborhood variables influencing its dynamics, yielding exponentially tight approximately compact representations (ACRs) of state and domain factors. These ACRs bound the error of truncating value functions to κ-hop neighborhoods, enabling efficient learning on graphs. A meta actor-critic then trains a shared policy across multiple source domains while conditioning on the compact domain factors; at test time, a few trajectories suffice to estimate the new domain factor and deploy the adapted policy. We establish finite-sample guarantees on causal recovery, actor-critic convergence, and adaptation gap, and show that GSAC adapts rapidly and significantly outperforms learning-from-scratch and conventional adaptation baselines.

Section: Introduction
Large-scale networked systems, such as traffic networks [41], power grids [3], and wireless communication systems [1,44], present significant challenges for reinforcement learning (RL) due to their massive scale, sparse local interactions, and structural heterogeneity. These characteristics impose two fundamental difficulties: scalability and generalizability. On one hand, the joint state-action space grows exponentially with the number of agents, making conventional RL algorithms [29] computationally intractable. On the other hand, real-world networked systems often experience environment shifts and structural changes, necessitating algorithms that can generalize and adapt efficiently across different environments. Therefore, a natural question arises:
Is it feasible to design a provably generalizable and scalable MARL algorithm for networked system?
While this question has attracted increasing attention in single-agent RL domain generalization literature [39,10,6], its resolution remains open in the multi-agent reinforcement learning (MARL) context, especially for learning in networked system. In this work, we provide an affirmative answer by developing a causality-inspired framework, GSAC (Generalizable and Scalable Actor-Critic), which couples causal representation learning with meta actor-critic learning to achieve both scalability and generalizability. We summarize our main contributions as follows.
• We establish structural identifiability in networked MARL, providing the first sample complexity results of causal mask recovery and domain factor estimation.
• We introduce efficient algorithms to construct approximately compact representations (ACRs) of states and domain factors. ACR improves both scalability, by significantly reducing the input dimensionality required for learning and computation, and generalizability, by isolating the minimal and most informative components. This approach may be of independent interest.
• We propose a meta actor-critic algorithm, which jointly learns scalable localized policies across multiple source domains, conditioning on the compact domain factors to generalize effectively.
• We provide rigorous theoretical guarantees on finite-sample convergence and adaptation gap, and empirically validate our method on two benchmarks demonstrating rapid adaptation and superior performance over learning-from-scratch and conventional adaptation baselines.
this section cite: ['b40', 'b2', 'b0', 'b43', 'b28', 'b38', 'b9', 'b5']

Section: Related works
Networked MARL with guarantees. Our work is closely related to the line of research on MARL in networked systems [23,17,24]. These studies leverage local interactions to enable scalability, proposing decentralized policy optimization algorithms that learn local policies for each agent with finite-time convergence guarantees. In particular, [24] provides the first provably efficient MARL framework for networked systems under the discounted reward setting. The works in [23] and [17] extend this framework to the average-reward setting and to stochastic and non-local network structures, respectively. However, to the best of our knowledge, no existing methodology addresses both the design and theoretical analysis of policies that are simultaneously generalizable across domains and scalable in networked MARL. Our work fills this gap by introducing a principled framework that achieves provable generalization and scalability via causal representation learning and domain-conditioned policy optimization.
this section cite: ['b22', 'b16', 'b23', 'b23', 'b22', 'b16']

Section: Domain generalization and adaptation in RL.
RL agents often encounter environmental shifts between training and deployment, prompting recent efforts to improve generalization and adaptation. [10] proposes learning factored representations along with individual change factors across domains, while [7] extends this approach to handle non-stationary environments. [22,33] aim to enhance generalization to unseen states by eliminating redundant dependencies between state and action variables in causal dynamics models. Beyond causal approaches, a substantial body of work focuses on learning domain-invariant representations without causal modeling, such as bisimulation metrics [39] that preserve decision-relevant structure while filtering out nuisance features. Causal representation learning has also been applied to model goal-conditioned transitions [6], offering theoretical guarantees via structure-aware meta-learning [10], and credit assignment [42,32]. However, these works focus solely on single-agent settings or multi-agent settings with a limited number of agents. In contrast, we provide the first sample complexity guarantees for structural identifiability in networked MARL and introduce ACRs, a novel mechanism that enables both scalable learning and provable generalization across domains.
A detailed discussion of related work is provided in Appendix A.
this section cite: ['b9', 'b6', 'b21', 'b32', 'b38', 'b5', 'b9', 'b41', 'b31']

Section: Preliminaries
For clarity, we provide a complete table of notation in Appendix B. Networked MARL. We consider networked MARL represented as a graph G = (N , E ), where N = 1, . . . , n denotes the set of agents and E ⊆ N × N encodes the interaction edges. Each agent i ∈ N observes a local state s i ∈ S i and selects an action a i ∈ A i , forming the global state s = (s 1 , . . . , s n ) ∈ S := S 1 ×• • •×S n and the joint action a = (a 1 , . . . , a n ) ∈ A := A 1 ×• • •×A n . At each time step t, the system evolves according to the following decentralized transition dynamics:
P (s(t + 1) | s(t), a(t)) = n i=1 P i s i (t + 1) | s Ni (t), a i (t) ,(1)
where N i ⊆ N denotes the set of neighbors of agent i in the interaction graph, including i itself, and s Ni (t) collects the states of agents in N i at time t. Each agent i adopts a localized policy π θi i , parameterized by θ i ∈ Θ i , which specifies a distribution over local actions conditioned on its local neighborhood state: π i (a i | s Ni ). Agents act independently according to their respective policies. We write θ := (θ 1 , . . . , θ n ) to denote the tuple of all local policy parameters, and define the joint policy as π θ (a | s) := n i=1 π θi i (a i | s Ni ). We only consider localized policy and use θ and π interchangeably throughout the paper when referring to policies.
Each agent receives a local reward r i (s i , a i ) depending on its local state and action, and the global reward is defined as the average across all agents: r(s, a) := 1 n n i=1 r i (s i , a i ). The goal is to learn a set of localized policies θ that maximize the expected discounted sum of global rewards, starting from an initial state distribution ρ 0 : max θ∈Θ J(θ) := E s∼ρ0 E a(t)∼π θ (•|s(t)), s(t+1)∼P (•|s(t),a(t)) ∞ t=0 γ t r(s(t), a(t)) s(0) = s . (2) For clarity, we assume the reward function is known; however, our framework and analysis readily extend to the setting with unknown rewards.
Truncation as efficient approximation in networked MARL A central challenge in applying reinforcement learning to networked systems is the curse of dimensionality: while each agent's local state and action spaces are relatively small, the global state and action spaces grow exponentially with the number of agents n. This renders standard RL methods computationally intractable at scale. [24] exploit the local interaction structure and demonstrate that the Q-function exhibits exponential decay with respect to graph distance. This property enables a principled truncation of the Q-function for scalable approximation. Specifically, for a state-action pair (s, a) and a policy π:
Q π (s, a) := E a(t)∼π θ (•|s(t)) ∞ t=0 γ t r(s(t), a(t)) s(0) = s, a(0) = a = 1 n n i=1 E a(t)∼π θ (•|s(t)) ∞ t=0 γ t r i (s i (t), a i (t)) s(0) = s, a(0) = a := 1 n n i=1 Q π i (s, a).(3)
For an integer κ ≥ 0, let N κ i denote the κ-hop neighborhood of agent i, and define N κ -i := N \N κ i as the set of agents outside this neighborhood. We write the global state and action as s = (s N κ i , s N κ -i ) and a = (a N κ i , a N κ -i ), respectively. [24] show that for any π, agent i ∈ N , and any tuples
s N κ i , s N κ -i , s ′ N κ -i , a N κ i , a N κ -i , a ′ N κ -i
, Q π i satisfies the exponential decay property:
Q π i (s N κ i , s N κ -i , a N κ i , a N κ -i ) -Q π i (s N κ i , s ′ N κ -i , a N κ i , a ′ N κ -i ) ≤ r 1 -γ γ κ+1 , (4
)
where r is the upper bound on the reward functions. Thus the influence of distant agents on Q π i diminishes rapidly with graph distance. This motivates the use of a truncated Q-function:
Qπ i s N κ i , a N κ i := Q π i s N κ i , sN κ -i , a N κ i , āN κ -i ,(5)
where sN κ -i and āN κ -i are fixed (and arbitrary) placeholders for the unobserved components. Due to exponential decay, the truncated Q-function approximates the true Q-function with bounded error: sup (s,a)∈S×A Qπ i s N κ i , a N κ i -Q π i (s, a) ≤ r 1-γ γ κ+1 . Furthermore, this truncated approximation can be directly used in actor-critic updates, enabling scalable policy gradient estimation. The approximation error in the truncated policy gradient is also exponentially small 3 . Crucially, the truncation technique has significantly reduced input dimensionality, making it much more efficient to compute, store, and optimize in large-scale networked settings.
this section cite: ['b23', 'b23']

Section: Networked MARL under domain generalization
The standard formulation of networked MARL, as reviewed above, assumes a fixed environment or domain. We extend this framework to incorporate domain generalization, where the environment may shift between training and deployment. Each environment E(ω) is characterized by its own transition dynamics (see Equation 6), and is uniquely specified by a latent domain factor ω = (ω i ) i∈N ∈ Ω, where ω i encodes the domain-specific dynamics for agent i. To model structured variation across environments, we define the local generative process for each agent i. Let s i = (s i,1 , . . . , s i,d s i ), where d s i is its dimensionality. The state and reward for agent i evolve as: s i,j (t + 1) = f i,j c s→s Ni,j ⊙ s Ni (t), c a→s i,j ⊙ a i (t), c ω→s i,j
⊙ ω i , ϵ s i,j (t) ,(6)
where ⊙ denotes element-wise multiplication, and ϵ denotes i.i.d. stochastic noise. The binary vectors c •→• are causal masks, indicating structural dependencies between variables. Crucially, the functions f i,j and the causal masks c •→• are shared across all environments, and only ω i varies. This decomposition allows us to disentangle invariant causal structure from domain-specific variations.
We consider a training setup with M source domains ⟨E 1 , . . . , E M ⟩ and a target domain E M +1 , each drawn i.i.d. from an unknown domain distribution ω m ∼ D for m = 1, • • • , M + 1. The goal is to learn networked MARL policies from the source domains that generalize effectively to new, unseen target domains, using as little adaptation data as possible.
this section cite: []

Section: Approximately compact representation
In networked MARL, each agent's truncated value function and localized policy are given by Qπ i :
S N κ i × A N κ i → R and π i : S Ni → ∆(A), respectively. The input dimensionality of the value function is dim(S N κ i × A N κ i ) = j∈N κ i (d s j + d a i ),
and that of the policy is dim(S Ni ) = j∈Ni d s j . While truncation to the κ-hop neighborhood significantly reduces the input size compared to the global space dim(S × A) = j∈N (d s j + d a j ), the input dimensionality can still be large, especially when κ or node degrees are high. To further mitigate this issue, we propose constructing ACRs by leveraging the identifiable causal masks c i for each agent i, as defined in Equation 6. Specifically, we use the causal structure to extract a reduced set of relevant variables from
s N κ i , denoted s • N κ i ⊂ s N κ i
, which preserves the predictive information for the value function. This leads to a strictly smaller input space dim(s
• N κ i ) < dim(s N κ i ),
with approximation error that decays exponentially in κ. Moreover, we extend this ACR framework to include the policy inputs and domain-specific factors, enabling efficient transfer across domains. The resulting ACR framework improves both scalability, by significantly reducing the input dimensionality required for learning and computation, and generalizability, by isolating the minimal and most informative domain-specific components relevant to each agent. In the following subsections, we detail the algorithmic construction of ACRs, beginning with the fixed-environment setting. For clarity, we assume the causal masks and domain factors are given; their identification and the estimation of domain factors are addressed in Section 5.
this section cite: []

Section: ACR for fixed environment
We begin by formalizing the notion of an ACR for truncated Q-functions.
this section cite: []

Section: Definition 1 (Value ACR).
Given the graph G encoded by the binary masks c, for each agent i and its κ-hop neighborhood N κ i , we recursively define the κ-hop ACR s where sN κ i /s • N κ i denotes fixed, arbitrary values for the irrelevant components. The approximation error of Qπ i compared to the full Q-function is exponentially small in κ, as shown below.
Proposition 1 (Approximation error of value ACR). For any agent i, policy π, and κ ≥ 0, the approximation error between Qπ i and Q π i satisfies:
sup (s,a)∈S×A Qπ i s • N κ i , a N κ i -Q π i (s, a) ≤ 2r 1 -γ γ κ+1 .
The proof is deferred to Appendix E. We now define ACRs for policies.
this section cite: []

Section: Definition 2 (Policy ACR).
Given the graph G encoded by binary causal masks c, the ACR for agent i's policy over its neighborhood N i is defined recursively using Algorithm 3 (see Appendix D).
Algorithm 3 takes as input the causal masks c N κ i and the local states s Ni , and outputs a compact representation s
• Ni ⊂ s Ni such that dim(s • Ni ) < dim(s Ni ).
We then define the approximately compact policy πi :
S • Ni → ∆ Ai by πi • | s • Ni := π i • | s • Ni , sNi /s • Ni ,
where sNi /s • Ni are fixed values for the non-influential components. By combining the value and policy ACRs, we define the doubly compact Q-function Qπ i s • N κ i , a N κ i
, which reduces input complexity while retaining performance guarantees. Proposition 2 (Approximation error). For any policy π, let π be the corresponding approximately compact policy constructed using Algorithm 3, for any i, and κ ≥ 0, we have
sup (s,a)∈S×A Qπ i s • N κ i , a N κ i -Q π i (s, a) ≤ 3 r 1 -γ γ κ+1 .
The proof of Proposition 2 is deferred to Appendix E. These results show that identifying ACRs enables dramatic reductions in the state-space dimensionality for each agent, making learning and computation tractable even in large-scale networks. In particular, the effective input dimension satisfies |s
• N κ i | ≪ |s N κ i |.
In the next subsection, we extend ACR construction to domain factors, enabling generalization across multiple environments.
this section cite: []

Section: ACR for efficient domain generalization
We now extend the ACR framework to support generalization across environments characterized by latent domain-specific factors. Given a policy π and environment ω = (ω i ) i∈N , the environmentconditioned Q-function is defined as
Q π i (s, a, ω) := E a(t)∼π(•|s(t)),s(t+1)∼Pω(•|s(t),a(t)) ∞ t=0 γ t r i (s i (t), a i (t)) | s(0) = s, a(0) = a .
We show (Lemma 7) that this ω-conditioned Q-function retains the exponential decay property.This motivates defining a truncated ω-conditioned Q-function by fixing irrelevant variables:
Qπ i s N κ i , a N κ i , ω N κ i := Q π i s N κ i , sN κ -i , a N κ i , āN κ -i , ω N κ i , ωN κ -i .(8)
We consider ω-conditioned policies π = (π 1 , . . . , π n ) such that each π i :
S Ni × Ω Ni → ∆ Ai , i.e., a i (t) ∼ π i (•|s Ni (t), ω Ni ).
To reduce the input space further, we define a domain-factor ACR ω • (Definition 5), which identifies the minimal subset of ω N κ i influencing the reward. Using both the value ACR and domain ACR, we define the approximately compact ω-conditioned Q-function 6). Similarly, we define policy ACRs for ω-conditioned policies (Definition 7), and let π denote the corresponding approximately compact policy. We then obtain a fully compact representation Qπ i of Q-function, which provably approximates the true Q-function. Proposition 3. For any policy π, and any i, we have sup (s,a,ω)∈S×A×Ω
Qπ i s • N κ i , a N κ i , ω • N κ i (Definition
Qπ i s • N κ i , a N κ i , ω • N κ i -Q π i (s, a, ω) ≤ 3r 1 -γ γ κ+1 . (9
)
The proof of Proposition 3 is deferred to Appendix E. Proposition 3 shows that it suffices to operate on the key components of the κ-hop state and latent domain factors, which substantially reduces input dimensionality. As a result, both training and test-time inference become significantly more scalable-without sacrificing theoretical guarantees. In the next section, we present our main algorithm, which leverages these ACRs to achieve generalizable and scalable networked policy learning. For simplicity, we will omit "approximately compact" when referring to the functions and policies Q, knowing that all components operate on the identified ACRs.
4 Generalizable and scalable actor-critic
this section cite: ['b5']

Section: Roadmap
We propose GSAC (Generalizable and Scalable Actor-Critic), a principled framework for scalable and generalizable networked MARL. Our GSAC framework (Algorithm 1) integrates causal discovery, representation learning, and meta actor-critic optimization into a unified pipeline, with each phase supported by theoretical results. Phase 1 (causal discovery and domain factor estimation) is underpinned by Theorem 1 and Propositions 4-5, which establish structural identifiability and sample complexity guarantees for recovering causal masks and latent domain factors. Phase 2 (construction of ACRs) leverages the causal structure to build compact representations of value functions and policies, with bounded approximation errors rigorously characterized by Propositions 1-3. Phase 3 (meta actor-critic learning) performs scalable policy optimization across multiple source domains, with convergence of the critic and actor updates guaranteed by Theorem 2 (critic error bound) and Theorem 3 (policy gradient convergence). Finally, Phase 4 (fast adaptation to new domains) exploits the learned meta-policy and compact domain factors to achieve rapid adaptation, where the adaptation performance gap is formally controlled by Theorem 4. Together, these results demonstrate that each algorithmic component is theoretically justified and collectively leads to provable scalability and generalization in networked MARL. Figure 3 visually illustrates the GSAC pipeline.
this section cite: []

Section: Algorithm overview
GSAC (Algorithm 1) consists of four sequential phases:
Phase 1: Causal discovery and domain factor estimation. In each source environment, agents estimate their local causal masks and latent domain factors to disentangle invariant structure from domain-specific variations; details are deferred to Section 5.
this section cite: []

Section: Phase 2: Constructing ACRs.
Using the recovered causal masks, each agent constructs ACRs for value functions and policies as described in Section 3. These ACRs significantly reduce the input dimensionality while preserving decision-relevant information for the following phases.
Phase 3: Meta-learning via actor-critic optimization. Agents are trained on across M source environments by optimizing policies through local actor-critic updates using ACR-based inputs. We provide a detailed description of this procedure in the following Section 4.3.
this section cite: []

Section: Phase 4: Fast adaptation to target domain.
In a new, unseen environment, each agent collects a few trajectories and estimates its domain factor ωM+1 i . The learned meta-policy π θ(K) i is then conditioned on the adapted ACR input (s •  Ni , ωM+1 Ni ) for immediate deployment. This process allows efficient generalization without requiring further policy training from scratch.
this section cite: []

Section: Key idea: meta-learning via actor-critic optimization
At each outer iteration k, a source domain m(k) is sampled and each agent i roll out trajectories using their current policies π θ(k) i , which are conditioned on the compact ACR inputs (s •  Ni , ω• Ni ). These interactions are used to update both the critic Q and the actor π θ in a decentralized yet coordinated manner, enabling policy generalization across domains.
this section cite: []

Section: Critic update.
Each agent maintains a local tabular critic Qi over its the ACR input space S
• N κ i × A N κ i × Ω • N κ i
. At every inner iteration t, the critic is updated using temporal-difference (TD) learning (Line 16): the Q-value for the current state-action-domain triple is updated toward a bootstrap target composed of the received reward and the next-step value. All other Q-values remain unchanged. This TD update leads to an estimate of a truncated Q-function for current domain.
Actor update. After completing an episode, each agent aggregates the Q-values of all agents within its κ-hop neighborhood and weights them by the gradient of the log-policy at each timestep. The actor parameters θ are then updated using stochastic gradient ascent with stepsize η k = η/ √ k + 1.
We present the finite-time convergence and adaptation error bounds of GSAC in Section 6. Prior to that, we discuss causal discovery and domain factor estimation in Phase 1 in Section 5.
this section cite: []

Section: Algorithm 1 GENERALIZABLE AND SCALABLE ACTOR-CRITIC
1: Input: θ i (0); parameter κ; T , length of each episode; stepsize parameters h, t 0 , η. 2: for source domain index m = 1, 2, . . . , M do ▷ P1: causal recovery and domain estimation
3: Sample ω m ∼ D, each agent i estimate the causal mask c i and domain factor ωm i 4: end for 5: for each agent i do ▷ P2: approximately compact representation 6: Identify s • N κ i ← ACR Q (c, i, κ) and s • Ni ← ACR π (c, i) 7: Identify ω m,• N κ i ← ACR Q (c, i, κ) and ω m,• Ni ← ACR π (c, i) for each m = 1, 2, • • • , M 8: end for 9: for k = 0, 1, 2, . . . , K -1 do ▷ P3: meta-learning 10: Sample domain index m(k) ∼ {1, . . . , K}, set ω• ← ωm(k),• , sample s(0) ∼ ρ 0 11: Each agent i takes action a i (0) ∼ π θi(k) i (• | s • Ni (0), ω• Ni ), and receive reward r i (0) 12: Initialize critic Q0 i ∈ R S • N κ i ×A N κ i ×Ω • N κ i to be all zeros 13:
for t = 1 to T do 14:
Get state s i (t), take action
a i (t) ∼ π θi(k) i (• | s • Ni (t), ω• Ni ), get reward r i (t) 15:
Update Q-function with stepsize α t-1
← h t-1+t0 : Qt i (s • N κ i (t-1), a N κ i (t-1), ω• N κ i ) ← (1 -α t-1 ) Qt-1 i (s • N κ i (t-1), a N κ i (t-1), ω• N κ i ) + α t-1 r i (t-1) + γ Qt-1 i (s • N κ i (t), a N κ i (t), ω• N κ i ) , Qt i (s • N κ i , a N κ i , ω• N κ i ) ← Qt-1 i (s • N κ i , a N κ i , ω• N κ i ), for (s • N κ i , a N κ i ) ̸ = (s • N κ i (t-1), a N κ i (t-1))16:
end for 17:
Each agent i estimates policy gradient:
ĝi (k) ← T t=0 γ t • 1 n j∈N κ i QT j (s • N κ j (t), a N κ j (t), ω• N κ j )∇ θi log π θi(k) i (a i (t) | s • Ni (t), ω• Ni )18
: Update policy: θ i (k + 1) ← θ i (k) + η k ĝi (k) with stepsize η k ← η √ k+1 19: end for 20: Collect few trajectories {(s(t), a(t), s(t + 1))} Ta t=0 in the new domain ▷ P4: generalization 21: Each agent i estimates new domain factor ωM+1 i , and deploy policy π θ(K) i (•|s • Ni , ωM+1 Ni )
this section cite: []

Section: Causal recovery and domain factor estimation
In this section, we first discuss the computational overhead of causal discovery and ACR construction, and then provide the theoretical guarantees for causal recovery and domain factor estimation in Theorem 1 and Proposition 4-5. Phase 1 (causal discovery) and Phase 2 (ACR construction) introduce upfront costs that are only one-time, local, and amortized. In particular, they are one-time preprocessing steps for each source domain and do not need to be repeated during meta-training or adaptation. Both steps are local per agent and over small neighborhoods, parallel across agents, and the results are re-used for the entire meta-training horizon and for adaptation. Theorem 1 (Structural identifiability in networked MARL). Under the standard faithfulness assumption, the structural matrices c i in 6 are identifiable from the observed data.
Theorem 1 guarantees that the underlying structure, that encodes how neighboring states, local actions, and latent domain factors affect local transitions, can be uniquely recovered from trajectories under standard causal discovery assumptions. The proof is deferred to Appendix F. Furthermore, Proposition 4 provides a finite-sample guarantee for recovering the local structural dependencies. Proposition 4 (Informal). Under standard assumptions, including faithfulness, minimum mutual information for true causal links, bounded in-degree d max , sub-Gaussian noise, and Lipschitz continuity of the transition function, the sample complexity to recover the causal masks satisfies
O dim(s Ni ) • d max log(dim(s Ni ) • n/δ) λ 2 ,
with probability at least 1 -δ, where λ quantifies signal strength.
The required sample size scales almost linearly with the number of observed variables dim(s Ni , the sparsity level d max , and decays quadratically with the strength of causal signals λ. The formal statement and proof of Proposition 4, and discussions on the imposed assumptions are deferred to Appendix F. Proposition 5 (Informal). Suppose the causal masks are recovered and the domain-dependent transition dynamics are identifiable. Assume that distinct domain factors induce distinguishable state transitions in total variation, and that Ω i is compact with diameter D Ω . Then, with probability at least 1 -δ, the estimated factor ω given a trajectory of length T e generated from true factor ω * satisfies
∥ ω -ω * ∥ 2 ≤ δ ω (T e ) = O   D Ω log(nT e /δ) T e   .(10)
The estimation error decays as O(1/ √ T e ) with high probability, and depends logarithmically on the number of agents. The result highlights that domain generalization can be performed efficiently with only a few samples. The formal statement and proof of Proposition 5 are deferred to Appendix F.
this section cite: []

Section: Convergence results and adaptation gap
For clarity of analysis, we first establish convergence and adaptation guarantees for an ACR-free variant of GSAC (Algorithm 5, detailed in Appendix D). Theoretical results for GSAC with ACR follow as corollaries. We define the expected return of a policy parameterized by θ using an estimated domain factor ω ′ in a true environment ω:
J(θ, ω ′ ; ω) := E s∼ρ0 E a(t)∼π θ (•|s(t),ω ′ ),s(t+1)∼Pω(s(t),a(t)) ∞ t=0 γ t r(s(t), a(t)) s(0) = s , (11) where π θ (a|s, ω) = n i=1 π θi i (a i |s Ni , ω Ni )
is the joint domain-conditioned policy. For notational simplicity, we write J(θ, ω) := J(θ, ω; ω) when the estimated and true domain factors coincide. To this end, for domain generalization our objective under domain distribution D is: max θ∈Θ J(θ) := E ω∼D [J(θ, ω)].
this section cite: []

Section: Convergence
We begin by introducing standard assumptions (Assumption 1-4) used in networked MARL [24], as well as additional ones (Assumption 5-6) for the domain generalization setting.
θ i , ∥∇ θi log π θi i (a i |s Ni , ω Ni )∥ ≤ L i , ∥∇ θ log π θ (a|s, ω)∥ ≤ L := n i=1 L 2 i , and ∇J(θ) is L ′ -Lipschitz continuous in θ. Assumption 4. Each agent's parameter space Θ i ⊂ R d θ i is compact with diameter bounded by D Θ . Assumption 5. For all i: (i) P i (s i (t + 1)|s Ni (t), a i (t), ω i ) is L P -Lipschitz in ω i ; (ii) Q θ i (s, a, ω) is L Q -Lipschitz in ω; (iii) ∇ θi J(θ, ω) is L J -Lipschitz in ω.
Assumption 6. The domain factor space Ω i is compact with diameter bounded by D Ω . Discussion on assumptions. Assumption 1-4 are standard for proving convergence of networked MARL algorithms, without consideration of domain generalization/adaptation [23,17,24]. To account for generalizability across domains in networked systems, we introduce additional Assumption 5-6 regarding the latent domain factor. Assumption 5 is similar with Assumption 3, which imposes the smoothness w.r.t. the domain factor, while Assumption 3 imposes the smoothness w.r.t. the actor parameter θ. Assumption 6 is a regularity assumption to ensure the compactness of domain factor space. We provide a detailed discussion on these assumption in Appendix G.
this section cite: ['b23', 'b22', 'b16', 'b23']

Section: Critic error bound.
We first analyze the inner-loop critic update. Fixing any outer iteration k, and omitting k from the notation, we establish the following result. Theorem 2 shows that the inner loop converges to an estimate of Q i with steady-state error decaying with 1/ √ T e and exponentially in κ. The proof and formal statement of Theorem 2 are deferred to Appendix G. Theorem 2 (Critic error bound, informal). Under Assumptions 1-6, and for any δ ∈ (0, 1), if the critic stepsize is set as α t = h/(t + t 0 ) and the domain factor ω is estimated from T e trajectories, then with probability at least 1 -δ, the critic estimate satisfies:
|Q i (s, a, ω) -QT i (s N κ i , a N κ i , ωN κ i )| ≤ C a √ T + t 0 + C ′ a T + t 0 + 2cρ κ+1 (1 -γ) 2 + C ′ ω log(nT e /δ) T e ,
where C a , C ′ a , and C ω are constants. These will be further characterized and discussed in Section 6.3, where we analyze the additional benefits of incorporating ACR.
this section cite: []

Section: Actor convergence.
Based on the critic bound, we derive the bound on policy gradient updates. The first term, of order O(1/ √ K + 1), vanishes as the number of outer iterations K increases. The remaining three terms correspond to different sources of error: the second arises from neighborhood truncation and decays exponentially with κ; the third stems from estimation of the domain factor ω, with error decreasing as 1/ √ T e ; and the fourth reflects the approximation of the domain distribution D using only M sampled source domains, decaying with 1/ √ M . The formal statement and proof of Theorem 3 are deferred to Appendix G. Theorem 3 (Policy gradient convergence, informal). Under Assumptions 1-6, for K ≥ 3 and sufficiently large T , if the actor and critic stepsizes are chosen appropriately and domain factors are estimated from T e samples, then with probability at least 1 -δ:
K-1 k=0 η k ∥∇J(θ(k))∥ 2 K-1 k=0 η k ≤ Õ 1 √ K + 1 + ρ κ+1 + 1 T e + 1 M .
this section cite: []

Section: Generalization
In Phase 4, for a new domain ω M +1 ∼ D, ACR-free GSAC collects T a trajectories, estimates ωM+1 , and deploys the policy π θ(K) (•|s, ωM+1 ). The expected return is:
J(θ(K), ωM+1 ; ω M +1 ) = E s(0)∼ρ0 E a(t)∼π θ (•|s(t), ωM+1 ),s(t+1)∼P ω M +1 (s(t),a(t)) ∞ t=0 γ t r t .
Theorem 4 (Adaptation guarantee). Under Assumptions 1-6, with probability at least 1 -δ:
E J(θ(K), ωM+1 ; ω M +1 ) | θ(K) ≥ J(θ(K)) -L ω ′ C ω log(n/δ) T a .
Theorem 4 establishes that the adaptation gap decreases at a rate of O(1/ √ T a ), relative to the return of the meta-trained policy. The proof is deferred to Appendix G.
this section cite: []

Section: Additional gains from ACR
Notably, ACR introduces only a constant-factor increase in approximation error over truncation (multiplicative factor 3), and thus all ACR-free convergence and generalization results naturally extend to GSAC. Beyond computational gains discussed in Section 3, ACR also improves sample efficiency.
In Theorem 2, the explicit constants are defined as C a := 6ε
1- √ γ τ h σ [log( 2τ T 2 δ ) + f (κ) log SA] with ε := 4 r 1-γ + 2r, C ′ a := 2 1- √ γ max( 16εhτ σ , 2r 1-γ (τ + t 0 ))
, and C ′ ω := L Q C ω √ D Ω , and the
stepsize α t = h t+t0 satisfying h ≥ 1 σ max(2, 1 1- √ γ ), t 0 ≥ max(2h, 4σh, τ ).
By leveraging ACR, the effective dimensionality of the κ-hop state space is dramatically reduced, leading to smaller τ and larger σ in Assumption 2. This reduces C a , C ′ a , and t 0 , enabling faster convergence and reducing the required inner-loop iterations T to achieve a given accuracy in Q-value estimation of within critic, thereby decreasing the total sample complexity.
this section cite: []

Section: Numerical experiments
We evaluate GSAC on two standard benchmarks for networked MARL algorithms: wireless communications [44,24,36] and traffic control [30,24] (the latter deferred to Appendix H.2). A detailed description of the experimental setup and comprehensive results are provided in Appendix H. We consider M = 3 source domains, each with domain factors ω ∈ {0.2, 0.5, 0.8}, while the target domain uses ω target = 0.65 unless otherwise specified. In each domain, GSAC is trained for K outer iterations (depending on convergence), with a inner loop horizon T = 10. For domain factor estimation, we collect T e = 20 trajectories per domain. To evaluate adaptation performance in the target domain, we compare GSAC against three baselines:
• GSAC (ours): Learn π θ (a|s, ω) by optimizing θ over source domains. At test time, estimate ω ′ and directly deploy π θ (a|s, ω ′ ) in the target domain without further training.
• SAC-MTL (multi-task): Learn π θ (a|s, z), where z denotes the pre-specified one-hot encoding of each source domain, and jointly optimize θ across domains [37]; deploy π θ (a|s, z ′ ) in the target domain.
• SAC-FT (fine-tune): Train a single policy π θ (a|s) across source domains without domain-factor conditioning and fine-tune θ in the target domain.
• SAC-LFS (learning from scratch): Train π θ (a|s) in the target domain without prior meta-training.
To evaluate both scalability and generalizability, we vary the grid size of the wireless communication network (grid size ∈ {3, 4, 5}), which affects both the number of users and the connectivity structure (see Figure 45). As shown in Figure 2 and Figure 6, GSAC consistently maintains high training and adaptation performance across grid sizes, demonstrating its scalability and generalizability. In particular, GSAC achieves the best few-shot performance (Episodes 1-30), reflecting rapid adaptation from minimal target data. In contrast, SAC-MTL exhibits moderate performance during the early adaptation phase but improves steadily over time. This is because multi-task learning leverages shared structure across source domains, yet lacks explicit domain-factor conditioning, leading to slower adaptation compared to GSAC. SAC-FT performs worse in the early phase due to its need to fine-tune policy parameters directly in the target domain, resulting in higher sample complexity. SAC-LFS suffers the slowest convergence and lowest overall return, underscoring the importance of meta-training and domain factor conditioning for efficient cross-domain adaptation.
this section cite: ['b43', 'b23', 'b35', 'b29', 'b23', 'b36']

Section: Conclusions and future work
We presented GSAC, a causality-aware MARL framework that integrates ACR construction with meta actor-critic learning, achieving provable scalability, fast adaptation, and strong generalization across domains. We established quasi-linear sample complexity and finite-sample convergence guarantees, and showed empirically that GSAC consistently outperforms competitive baselines on challenging networked MARL benchmarks.
The main limitation of our current work is that GSAC has only been evaluated on tabular and fully observed benchmarks. Nonetheless, it establishes a principled foundation for practical control of large-scale networked systems. Promising future directions include: (i) extending GSAC to continuous state and action spaces with function approximation based on ACRs [10]; (ii) broadening empirical evaluation to more diverse and large-scale networked systems [19]; and (iii) incorporating partial observability into the framework.
this section cite: ['b9', 'b18']

Section: References
Ref_id:b0 Title: Wireless powered communication networks: An overview Year: (2016)
Ref_id:b1 Title: Invariant causal imitation learning for generalizable policies Year: (2023)
Ref_id:b2 Title: Reinforcement learning for selective key applications in power systems: Recent advances and future challenges Year: (2022)
Ref_id:b3 Title: An economic and low-carbon dispatch algorithm for microgrids with electric vehicles Year: (2024)
Ref_id:b4 Title: Risk-sensitive and robust decisionmaking: a cvar optimization approach Year: (2015)
Ref_id:b5 Title: Generalizable goalconditioned reinforcement learning via causal world models Year: ()
Ref_id:b6 Title: Factored adaptation for nonstationary reinforcement learning Year: (2022)
Ref_id:b7 Title: Local td-update is more sample-efficient than batching for marl policy evaluation with average reward Year: ()
Ref_id:b8 Title: Real-time coordination of human couriers and drones for ondemand food-delivery platforms: A multi-stage risk-aware multi-agent reinforcement learning framework Year: (2025)
Ref_id:b9 Title: Adarl: What, where, and how to adapt in transfer reinforcement learning Year: ()
Ref_id:b10 Title: Causal discovery from heterogeneous/nonstationary data Year: (2020)
Ref_id:b11 Title: Robust dynamic programming Year: (2005)
Ref_id:b12 Title: Graph neural networks for multi-agent reinforcement learning Year: (2018)
Ref_id:b13 Title: Bridging distributional and risk-sensitive reinforcement learning with provable regret bounds Year: (2024)
Ref_id:b14 Title: Provably efficient multi-agent reinforcement learning with fully decentralized communication Year: ()
Ref_id:b15 Title: A finite-time analysis of distributed q-learning. OpenReview Year: (2024)
Ref_id:b16 Title: Multi-agent reinforcement learning in stochastic networked systems Year: (2021)
Ref_id:b17 Title: Invariant causal representation learning for generalization in imitation and reinforcement learning Year: (2022)
Ref_id:b18 Title: Efficient and scalable reinforcement learning for large-scale network control Year: (2024)
Ref_id:b19 Title: Robust control of markov decision processes with uncertain transition matrices Year: (2005)
Ref_id:b20 Title: Models, reasoning and inference Year: (2000)
Ref_id:b21 Title: Mocoda: Model-based counterfactual data augmentation Year: (2022)
Ref_id:b22 Title: Scalable multi-agent reinforcement learning for networked systems with average reward Year: (2020)
Ref_id:b23 Title: Scalable reinforcement learning for multi-agent networked systems Year: (2022)
Ref_id:b24 Title: Efficient off-policy meta-reinforcement learning via probabilistic context variables Year: (2019)
Ref_id:b25 Title: ProMP: Proximal meta-policy search Year: (2019)
Ref_id:b26 Title: Risk-averse dynamic programming for markov decision processes Year: (2010)
Ref_id:b27 Title: Multi-agent actor-critic multitask reinforcement learning based on gtd(1) with consensus Year: (2022)
Ref_id:b28 Title: Reinforcement learning: An introduction Year: (1998)
Ref_id:b29 Title: Max pressure control of a network of signalized intersections Year: (2013)
Ref_id:b30 Title: Overcoming the sim-to-real gap: Leveraging simulation to learn to explore for real-world rl Year: ()
Ref_id:b31 Title: Offline multi-agent reinforcement learning with causal credit assignment Year: ()
Ref_id:b32 Title: Causal dynamics learning for task-independent state abstraction Year: (2022)
Ref_id:b33 Title: Decentralized adaptive formation via consensus-oriented multi-agent communication Year: (2023)
Ref_id:b34 Title: Mean field multi-agent reinforcement learning Year: (2018)
Ref_id:b35 Title: Scalable primaldual actor-critic method for safe multi-agent rl with general utilities Year: (2023)
Ref_id:b36 Title: Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning Year: (2020)
Ref_id:b37 Title: Fully decentralized multi-agent reinforcement learning with networked agents Year: (2018)
Ref_id:b38 Title: Learning bisimulation metrics for generalization in reinforcement learning Year: ()
Ref_id:b39 Title: Tackling non-stationarity in reinforcement learning via causal-origin representation Year: (2023)
Ref_id:b40 Title: Global convergence of localized policy iteration in networked multi-agent reinforcement learning Year: (2023)
Ref_id:b41 Title: Meng Fang, and Mykola Pechenizkiy. Interpretable reward redistribution in reinforcement learning: A causal approach Year: (2023)
Ref_id:b42 Title: VariBAD: A very good method for bayes-adaptive deep rl via meta-learning Year: (2020)
Ref_id:b43 Title: Temporal starvation in multi-channel csma networks: an analytical framework Year: (2019)
