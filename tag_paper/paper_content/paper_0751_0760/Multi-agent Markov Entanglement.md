Title: Multi-agent Markov Entanglement
Abstract: Value decomposition has long been a fundamental technique in multi-agent reinforcement learning and dynamic programming. Specifically, the value function of a global state (s 1 , s 2 , . . . , s N ) is often approximated as the sum of local functions:This approach has found various applications in modern reinforcement learning systems. However, the theoretical justification for why this decomposition works so effectively remains underexplored. In this paper, we uncover the underlying mathematical structure that enables value decomposition. We demonstrate that a Markov decision process (MDP) permits value decomposition if and only if its transition matrix is not "entangled"-a concept analogous to quantum entanglement in quantum physics. Drawing inspiration from how physicists measure quantum entanglement, we introduce how to measure the "Markov entanglement" and show that this measure can be used to bound the decomposition error in general multi-agent MDPs. Using the concept of Markov entanglement, we proved that a widely-used class of policies, the index policy, is weakly-entangled and enjoys a sublinear O( √ N ) scale of decomposition error for N -agent systems. Finally, we show Markov entanglement can be efficiently estimated, guiding practitioners on the feasibility of value decomposition.

Section: Introduction
Learning the value function given certain policy, or policy evaluation, is one of the most fundamental tasks in RL. Significant attention has been paid to single-agent policy evaluation [39,8,40]. However, when it comes to multi-agent reinforcement learning (MARL), single-agent methodologies typically suffer from the curse of dimensionality: the state space of the system scales exponentially with the number of agents. To tackle this problem, one common technique is value decomposition,
V (s 1 , s 2 , . . . , s N ) ≈ N i=1 V i (s i ) ,
where V i is some local function that can be learned independently by each agent. It quickly follows that this decomposition greatly reduces the computation complexity from exponential to linear dependency on the number of agents N .
The remaining question is whether this decomposition is effective. This is non-trivial due to the coupling of agents-individual agent's action and transition depend on other agents. For example, in a ride-hailing platform, if one driver took the order, then other drivers are not allowed fulfill the same order. As a result, value decomposition may lose information and introduce bias without considering the global constraints.
In the past several decades, both positive and negative results have been reported. Back to the last century, [49,47] apply Lagrange relaxations to decompose the global value and obtain the well-known Whittle index policy. The Lagrange decomposition idea has also been proved successful in many other important multi-agent tasks such as network revenue management [1,50], resource allocation [27,7], and online matching [11,12,36,28]. However, Lagrange decomposition relies on the knowledge of system dynamics and [2] show its decomposition error can be arbitrarily bad for general multi-agent MDPs. In more recent days, practitioners apply online (deep) reinforcement learning to train a local value function for each individual agent. This practice gives birth to state-of-the-art dispatching policies in ride-hailing platforms and has been well recognized by the operations research community, such as DiDi Chuxing [33] (Daniel H. Wagner Prize, 2020) and Lyft [4] (Franz Edelman Laureates, 2024). Intervention policies based on a similar value decomposition idea also demonstrate substantial empirical advantages and have been deployed by a behavioral health platform in Kenya [5] (Pierskalla Award, 2024). In broader MARL literature, value decomposition serves as one key component of centralized training and decentralized execution (CTDE) paradigm, achieving strong empirical performance [38,29,35]. However, recent research has started reflecting on the invalidity and potential flaw of value decomposition in practice [25,16].
Despite all these empirical success and failures, there remains little theoretical understanding of whether and how we can decompose the value function in multi-agent MDPs.
this section cite: ['b38', 'b7', 'b39', 'b48', 'b46', 'b0', 'b49', 'b26', 'b6', 'b10', 'b11', 'b35', 'b27', 'b1', 'b32', 'b3', 'b4', 'b37', 'b28', 'b34', 'b24', 'b15']

Section: This paper
In this paper, we will uncover the underlying mathematical structure that enables/disables value decomposition. Our new theoretical framework quantifies the inter-dependence of agents in multiagent MDPs and systematically characterizes the effectiveness of value decomposition. For simplicity, we will demonstrate the main results through two-agent MDPs indexed by agent A and B. We later extend our results to general N -agent MDPs in Appendix H.
We start with a trivial example where two agents are independent, i.e. each following independent MDPs. It's clear that the global value function can be decomposed as the sum of value functions of local MDPs. As two agents are independent, it holds
P π (s ′ A , s ′ B | s A , s B ) = P π (s ′ A | s A ) • P π (s ′ B | s B )
, or in matrix form, P π AB = P π A ⊗ P π B , where ⊗ is the tensor product or Kronecker product of matrices. The important question is whether we can extend beyond this trivial case of independent subsystems.
this section cite: []

Section: A Sufficient and Necessary Condition
We introduce a new condition called "Markov Entanglement" to describe the intrinsic structure of transition dynamics in multi-agent MDPs.
Definition 1 (Markov Entanglement). Consider a two-agent MDP with transition P π AB . If there exists
P π AB = K j=1 x j P (j) A ⊗ P (j) B ,
then P π AB is separable; otherwise entangled. Compared with the preceding example of independent subsystems, Markov entanglement offers an intuitive interpretation: a two-agent MDP is separable if it can be expressed as a linear combination of independent subsystems. We then demonstrate, separable P π AB ⇐⇒ decomposable V π AB , where V π AB is decomposable if there exist local value functions V A , V B such that V π AB (s A , s B ) = V A (s A ) + V B (s B ) for all (s A , s B ). This result sharply unravels the secret structure of system dynamics governing value decomposition. As a sufficient condition, our finding strictly generalizes the previous independent subsystem example, extending it to scenarios involving interacting and coupled agents. As a necessary condition, we prove that exact value decomposition under any reward kernel requires the system dynamics to be separable. Taken together, this result provides a complete characterization of when exact value function decomposition is possible in multi-agent MDPs.
More interestingly, our Markov entanglement condition turns out be a mathematical counterpart of quantum entanglement in quantum physics, whose definition is provided below.
Definition 2 (Quantum Entanglement). Consider a two-party quantum state ρ AB . If there exists
ρ AB = K j=1 x j ρ (j) A ⊗ ρ (j) B , x ≥ 0 ,
then ρ AB is separable; otherwise entangled.
The quantum state is represented by a density matrix, a positive semidefinite matrix with unit trace, analogous to transition matrix in the Markov world. The concept of quantum entanglement describes the inter-dependence of particles in a quantum system, while Markov entanglement describes that of agents in a Markov system.
Finally, we introduce several novel proof techniques concerning the sufficient and necessary condition, including an "absorbing" technique for separable transition matrices and a novel characterization of the linear space spanned by tensor products of transition matrices. We believe these techniques hold independent interest for the broader RL community.
Decomposition Error in General Multi-agent MDPs Despite the precise characterization of Markov entanglement and exact value decomposition, general multi-agent MDPs can exhibit arbitrary complexity, with agents intricately entangled. This raises a critical question: can value decomposition serve as a meaningful approximation in such scenarios? To address this, we introduce a mathematical quantification to measure the Markov entanglement in general multi-agent MDPs,
E(P π AB ) := min P ∈PSEP d(P π AB , P ) ,(1)
where P SEP is the set of all separable transition matrices and d(•, •) is some distance measure. In other words, the degree of Markov entanglement is determined by its distance to the closest separable transition matrix. This concept can also find its counterpart in quantum physics, with the measure of quantum entanglement defined as
E(ρ AB ) := min ρ∈ρSEP d(ρ AB , ρ) ,
where ρ SEP is the set of all separable quantum states. In quantum physics, various distance measures have been designed for density matrices and capture different physical interpretations [31]. In the Markov world, we analogously design distance measures for transition matrices and relate them to the value decomposition error,
decomposition error of V π AB = O E(P π AB ) .
where ∥•∥ depends on the distance we use to measure Markov entanglement. We explore diverse distance measures including the well-known total variation distance and its stationary distribution weighted variant. We also design a novel agent-wise distance incorporating the multi-agent structure, which may be of independent interest to the MARL community. We further demonstrate how different distance measures give birth to the decomposition error in different norms.
this section cite: ['b30']

Section: Applications of Markov Entanglement
Finally, we leverage our Markov entanglement theory to analyze several structured multi-agent MDPs. We prove that a widely-used class of index policies is asymptotically separable, exhibiting a decomposition error that scales as O( √ N ) with the number of agents N . This result theoretically justifies the practical effectiveness of value decomposition for index-based policies. Our proof builds on innovations that integrate Markov entanglement with mean-field analysis. We also show that Markov entanglement admits an efficient empirical estimation, thus helping practitioners determine when value decomposition is feasible.
this section cite: []

Section: Other related work
In the first section, we have reviewed typical empirical works on value decomposition. Here, we complement that discussion with related literature on theoretical insights.
Prior theoretical research has extensively investigated the decomposition of optimal value functions in multi-agent settings. A prominent area involves Lagrange relaxation, with the Restless Multi-Armed Bandit (RMAB, [49]) as a foundational model. Lagrange relaxation decouples the constraint of agents, yielding a decomposable value that upper bounds the original value. The per-agent decomposition error is proven to decay asymptotically to zero [47,48,41] and enjoys a quadratic or exponential rate [20,21,11,51,52]. Other work generalizes to Weakly-Coupled MDPs (WCMDPs) [6,13,19]. However, [2] showed Lagrange relaxation can have arbitrarily large errors and proposed an alternative decomposition called Approximate Linear Programs (ALP), which is proven to have tighter error [12]. Despite these advancements, characterizing decomposition error for general multi-agent MDPs remains unknown. In contrast, our Markov entanglement theory analyzes value decomposition for general multi-agent MDPs under arbitrary policies, including optimal ones.
Another line of theoretical work has concentrated on policy optimization via value decomposition. Despite reported empirical successes, rigorous theoretical analysis remains challenging. [5] derived an approximation ratio for a specific index policy on a two-state RMAB. [43,16] analyzed the convergence of the CTDE paradigm under strong exploration assumptions, while also highlighting scenarios of divergence. In contrast, our work instead focuses on policy evaluation rather than optimization. This enables us to derive clear and interpretable bounds on the decomposition error for general finite-state multi-agent MDPs that only require the existence of a stationary distribution.
Notations We abbreviate subscripts (s) := (s 1:N ) := (s 1 , s 2 , . . . , s N ). Particularly, for two-agent case, when the context is clear, we abbreviate (s) := (s AB ) := (s A , s B ). Let [N ] = {1, 2, . . . , N } and Z + be the set of positive integers.
this section cite: ['b48', 'b46', 'b47', 'b40', 'b19', 'b20', 'b10', 'b50', 'b51', 'b5', 'b12', 'b18', 'b1', 'b11', 'b4', 'b42', 'b15']

Section: Model
We consider a standard two-agent MDP M AB (S, A, P , r (2) Notice we formally introduce our research question using Q-value instead of V-value function as in the introduction. Q-value decomposition is a stronger result that implies V-value function decomposition. It also turns out that Q-value further incorporates action information enabling more general theoretical analysis. More discussions can be found in Appendix B.
this section cite: []

Section: Local (Q-)value functions
Recent literature offers several algorithms for learning local (Q-)values. In this paper, we use a meta-algorithm framework in 1 to summarize their underlying principles. This meta-algorithm framework is simple and intuitive: each agent independently fits its local Qvalues based on its local observations. Notably, the framework requires no prior knowledge of the MDP, and learning can be performed in a fully decentralized manner. Furthermore, we use term meta in that we do not pose restrictions on how agents estimate their local Q-values. For tabular case, one Meta Algorithm 1: Leaning Local Q-value Functions Require: Global policy π; horizon length T .
1: Execute π for T epochs and obtain D = (s t AB , a t AB , r t AB , s t+1 AB , a t+1 AB ) T -1 t=1 . 2: Each agent i ∈ {A, B} fits Q π i using local observations D i = (s t i , a t i , r t i , s t+1 i , a t+1 i ) T -1 t=1 .
can plug in Temporal Difference (TD) learning [39] or its variants. For large-scale problems, one can apply linear function approximations (e.g. [5,24,8]) or more sophisticated neural networks (e.g. [33,38,29]).
Despite the flexibility in fitting local value functions, it is helpful to call out a particular approach: TD learning for local Q-values in the tabular case, as it facilitates the analysis and reveals the structure of value decomposition in the next section.
this section cite: ['b38', 'b4', 'b23', 'b7', 'b32', 'b37', 'b28']

Section: Local TD learning.
Although each agent's environment is not Markovian in a local sense (it is, more precisely, partially observed Markovian), one can still define its "marginalized" local transition matrix under the stationary distribution. Mathematically, for agent A, we denote P π A ∈ R |S||A|×|S||A| as its local transition where
P π A (s ′ A , a ′ A | s A , a A ) = s ′ B ,a ′ B s B ,a B P π AB (s ′ AB , a ′ AB | s AB , a AB ) µ π AB (s B , a B | s A , a A ) . (3
)
Here, µ π AB ∈ ∆(S) denotes the global stationary distribution under policy π (for convenience, we assume π induces a unichain, i.e. µ π AB is unique and strictly positive). 2 Given this "marginalized" local transition, the local Q-values obtained by Meta Algorithm 1 using tabular TD learning converge to the solution of the following "marginalized" Bellman equation:
Q π A = (I -γP π A ) -1 r A .
By symmetry, we can derive analogous results for agent B, obtaining its transition matrix P π B and local Q-values Q π B . Next, we show how Q π A and Q π B contribute to the exact value decomposition.
this section cite: []

Section: Exact value decomposition
To begin, recall the key condition we identify in the introduction: Markov Entanglement in Definition 1. Our first theorem shows that an MDP with no Markov entanglement is indeed sufficient for the exact value decomposition. More importantly, local TD learning (or Meta Algorithm 1 more generally) is guaranteed to recover such decomposition, i.e. Q π AB = Q π A ⊗ e + e ⊗ Q π B . Theorem 1. Consider a two-agent MDP M AB and policy π : S → ∆(A). If two agents are separable, i.e. there exists K ∈ Z + , measure {x j } j∈[K] , and transition matrices P (j)
A , P (j) B j∈[K]
such that
P π AB = K j=1 x j P (j) A ⊗P (j) B . Then it holds P π A = K i=1 x j P (j)
A and P π B = K j=1 x j P (j) B . Furthermore, the Eq. ( 2) holds
Q π AB = Q π A ⊗ e + e ⊗ Q π B .
This theorem establishes that even when the system is not independent, as long as it can be represented as a linear combination of independent subsystems, the global Q-value admits an exact decomposition.
An illustrative example of coupling and Markov entanglement To elucidate the concept of Markov entanglement, we present an example of two-agent MDP where agents are coupled but not entangled. Consider a two-agent MDP M AB with |A A | = |A B | = 2 , where action 1 means activate and 0 means idle. Each agent i ∈ {A, B} has its own local transition kernel P i . We examine the following policy: at each time-step, we randomly activate one agent and keep another idle, i.e. π(a | s) = 1/2 if a = (0, 1) or a = (1, 0). Consequently, this policy couples the agents through the constraint a A + a B = 1 at each timestep. However, we will demonstrate that despite this coupling, there's no entanglement. Specifically, we construct the following decomposition
P π AB = 1 2 P 0 A ⊗ P 1 B + 1 2 P 1 A ⊗ P 0 B ,(4)
where P a i refers to the transition matrix of agents i ∈ {A, B} taking action a ∈ {0, 1}. Intuitively, the right-hand side of Eq. ( 4) describes how at each time step, the global system randomly selects between two possible transitions: P 0 A ⊗ P 1 B or P 1 A ⊗ P 0 B , each with equal probability (akin to rolling a fair dice). This example thus clearly demonstrates a coupled system can still be separable and thus admits an exact value decomposition.
this section cite: []

Section: Proof of sufficiency
Theorem 1 admits a simple proof based on the several basic properties of tensor product. First of all, given P π AB = K j=1 x j P (j) A ⊗ P (j) B , we can plug this into the formulation of P π A in Eq. (3) and quickly verify P π A = K i=1 x i P (i) A . It remains to show Eq. (2). Notice that
(I -γP π AB ) -1 (r A ⊗ e) = ∞ t=0 γ t   K j=1 x j P (j) A ⊗ P (j) B   t (r A ⊗ e) (i) = ∞ t=0 γ t     K j=1 x j P (j) A   t r A   ⊗ e = (I -γP π A ) -1 r A ⊗ e = Q π A ⊗ e .
where we refer to (i) as an "absorbing" technique based on the bilinearity and mixed-product property of tensor product 3 . Specifically, since P e = e for any transition matrix P , we have for any t,
  K j=1 x j P (j) A ⊗ P (j) B   t (r A ⊗ e) =   K j=1 x j P (j) A ⊗ P (j) B   t-1   K j=1 x j P (j) A r A ⊗ P (j) B e   =   K j=1 x j P (j) A ⊗ P (j) B   t-1   K j=1 x j P (j) A r A   ⊗ e = . . . =     K j=1 x j P (j) A   t r A   ⊗ e .
Similar results can be derived for P π B such that (I -γP π AB ) -1 (e ⊗ r B ) = e ⊗ Q π B . Finally, combining the above results, we have
Q π AB = (I -γP π AB ) -1 r AB = (I -γP π AB ) -1 (r A ⊗ e + e ⊗ r B ) = Q π A ⊗ e + e ⊗ Q π B .
this section cite: []

Section: Necessary condition for the exact value decomposition
We then investigate whether Markov entanglement is necessary for the exact Q-value decomposition. The answer is in general no, since one can construct trivial counterexamples such as r A = r B = 0 or γ = 0, where the decomposition trivially holds. On the other hand, we focus on a stronger and more general concept of the exact value decomposition that holds under any reward kernel given γ > 0.
Formally, we present the following theorem. Theorem 2. Consider a two-agent Markov MDP M AB with discount factor γ > 0 and π : S → ∆(A). Suppose there exists local functions
Q i : r i → R |S||A| for i ∈ {A, B} such that Q π AB = Q A (r A ) ⊗ e + e ⊗ Q B (r B )
holds for any pair of reward r A , r B , then A, B must be separable.
Combined with Theorem 1, we conclude Markov entanglement serves as a sufficient and necessary condition for the exact value decomposition. We also emphasize that Theorem 2 considers general local functions Q i . This generality accommodates all methods for fitting local Q i , such as deep neural networks, provided that the training relies solely on the local observations of agent i.
There exist other possible ways for value decomposition. For example, [38,16] consider
Q π AB (s, a) = L A (s A , a A , r AB ) + L B (s B , a B , r AB )
where L A , L B are learned jointly via minimizing the global Bellman errorfoot_3 ; [35,29,37,42] consider general monotonic operations beyond additive decompositions. These methods introduce possibly richer representations at the cost of more sophisticated implementations and less interpretability, which is beyond the scope of this paper.
Proof sketch of necessity Our proof builds on several novel techniques. Recall P SEP is the set of all separable transition matrices.
this section cite: ['b37', 'b15', 'b34', 'b28', 'b36', 'b41']

Section: Step 1: Understanding the orthogonal complement.
If a transition matrix is entangled, it will have non-zero component in the orthogonal complement of P SEP , which we construct as
P ⊥ SEP =    |S||A|-1 j=1 ε j e ⊤ ⊗ W 1 j + |S||A|-1 j=1 W 2 j ⊗ ε j e ⊤ W 1 1:j , W 2 1:j ∈ R |S||A|×|S||A|    ,
where ε j = (1, 0, . . . , 0, -1, 0, . . . , 0) ⊤ with the first element 1 and (j + 1)-th element -1. Then, we study an intermediate transition matrix (1 -γ)(I -γP π AB ) -1 . We show if it's entangled, we are able to construct r A , r B based on its component in P ⊥ SEP such that Q π AB is not decomposable under this pair of rewards. We thus conclude decomposable
Q π AB =⇒ separable (1 -γ)(I -γP π AB ) -1 .
this section cite: []

Section: Value decomposition error in general two-agent MDPs
In general, the system transition P π AB can be arbitrarily entangled. In these scenarios, we investigate when value decomposition Q π
A ⊗ e + e ⊗ Q π B is an effective approximation of Q π AB . As mentioned in the introduction, we define the measure of Markov entanglement in Eq. ( 1) as certain distance between P π AB and its closet separable transition matrix. We will examine several distance measures for transition matrices and relate them to the decomposition error.
this section cite: []

Section: Entry-wise error bound
Total variation distance One widely used metric for transition matrices is Total Variation (TV) distance. Specifically, for two transition matrices P ,
P ′ ∈ R |S| 2 |A| 2 ×|S| 2 |A| 2 , define ∥P -P ′ ∥ TV := max (s,a)∈S×A D TV (P (•, • | s, a), P ′ (•, • | s, a)) ,(5)
where D TV is the total variation distance between probability measures. While TV distance is straightforward, it does not take into account the inherent multi-agent structure.
this section cite: []

Section: Agent-wise distance
We thus introduce a more refined distance specially designed for multi-agent MDPs. Formally, the Agent-wise Total Variation (ATV) distance between two transition matrices
P , P ′ ∈ R |S| 2 |A| 2 ×|S| 2 |A| 2 w.r.t agent A is defined as ∥P -P ′ ∥ ATVA := max (s,a)∈S×A D TV   s ′ B ,a ′ B P (•, • | s, a), s ′ B ,a ′ B P ′ (•, • | s, a)   .(6)
The ATV distance w.r.t agent B can be defined similarly. Intuitively, compared to TV, ATV focuses on an individual agent and measures the difference between its local transitions. One can also verify ATV is tighter distance, i.e. ∥P -P ′ ∥ ATVA ≤ ∥P -P ′ ∥ TV . We can plug ATV into Eq. ( 1) and obtain the measure of Markov entanglement w.r.t ATV distance E i (P π AB ) := min P ∈PSEP ∥P π AB -P ∥ ATVi for i ∈ {A, B}. In fact, one can also verify
E A (P π AB ) = min P A max (s,a)∈S×A D TV P π AB (•, • | s, a), P A (•, • | s A , a A ) ,(7)
The following theorem connects these measures to the value decomposition error.
Theorem 3. Consider a two-agent Markov system M AB and policy π : S → ∆(A) with the measure of Markov entanglement E A (P π AB ), E B (P π AB ) defined in Eq. ( 7), then the decomposition error is entry-wise bounded by the measure of Markov entanglement,
Q π AB -(Q π A ⊗ e + e ⊗ Q π B ) ∞ ≤ 4γ E A (P π AB )r A max + E B (P π AB )r B max (1 -γ) 2 .
this section cite: []

Section: Error weighted by stationary distribution
Entry-wise error bound is a very strong result for Q-value decomposition. This comes with the entry-wise TV bounds in both TV and ATV distance. An alterative choice is to consider an error weighted by the stationary distribution. Formally, consider
Q π AB -(Q π A ⊗ e + e ⊗ Q π B ) µ π AB := s,a µ π AB (s, a) Q π AB (s, a)-(Q π A (s A , a A )+Q π B (s B , a B )) .
We note that this norm is clearly weaker than the entry-wise norm. Nevertheless, a stationary distribution weighted error bound is sufficient in many practical scenarios. Similar ideas are also quite common in policy evaluation literature [14,40,9].
this section cite: ['b13', 'b39', 'b8']

Section: Distance weighted by stationary distribution
To analyze this µ π AB -weight decomposition error, we analogously propose the µ π AB -weighted distance measure of Markov entanglement. Specifically, we have the following µ π AB -weighted version of Eq. ( 7).
E A (P π AB ) = min P A s,a µ π AB (s, a)D TV P π AB (•, • | s, a), P A (•, • | s A , a A ) .(8)
Eq. ( 8) substitutes the µ π AB -weighted average for the maximum operator in Eq. ( 7). Finally, we have the following variant of Theorem 3. Theorem 4. Under the same setup as Theorem 3 with µ π AB -weighted measure of Markov entanglement E A (P π AB ), E B (P π AB ) defined in Eq. ( 8), the µ π AB -weighted decomposition error is bounded,
Q π AB -(Q π A ⊗ e + e ⊗ Q π B ) µ π AB ≤ 4γ E A (P π AB )r A max + E B (P π AB )r B max (1 -γ) 2 .
Compared to Theorem 3, Theorem 4 measures a weaker µ π AB -weighted decomposition error, while the condition on P π AB is also relaxed, requiring only a weighted average bound in Eq. ( 8).
this section cite: []

Section: Multi-agent Markov entanglement
Finally, we extend the results to multi-agent MDPs with the measure of Markov entanglement E 1:N (P π 1:N ) for an N -agent MDP. The extension is relatively straightforward. We demonstrate the extension of Theorem 4 below and more details can be found in Appendix H. Theorem 5. Consider a N -agent MDP M 1:N with the measure of Markov entanglement E i (P π 1:N ) w.r.t ATV distance, the µ π 1:N -weighted decomposition error is bounded by the measure of Markov entanglement,
Q π 1:N (s, a) - N i=1 Q π i (s i , a i ) µ π 1:N ≤ 4γ N i=1 E i (P π 1:N )r i max (1 -γ) 2 .
this section cite: []

Section: Applications of Markov Entanglement
In this section, we apply Markov entanglement and demonstrate a widely-used class of index policies is asymptotically separable. To begin, we introduce the model of Restless Multi-Armed Bandit (RMAB, [49]). In an N -agent RMAB, each agent follows a homogeneous two-action MDP with action 1 meaning activate and 0 idle. A central decision maker will activate M ≤ N agents at each timestep and leave other agents idle. In other words, agents transit independently but are coupled under constraint N i=1 a i = M . In RMAB, arguably the most classical and widely-used policy is the index policy, which we formally define as Definition 3 (Index Policy). There exists a priority index ν s for each local state s. The decision maker will always activate agents in the descending order of the priority until the budget constraint M is met. Ties are resolved fairly via uniform random sampling of agents at the same state.
The index policy traces back to the well-known Gittins Index [46], Whittle Index [49,47,20], and fluid-based index policies [41,21]. [33,4,5,30,44,3] apply data-driven method to optimize index policies and report great empirical success in industrial implementations. Understanding the mystery behind such success calls for a theory for general index policies. We then present our main theorem. Theorem 6. Consider an N -agent restless multi-armed bandit. For any index policy satisfying mild technical conditions, there exists constant C independent of N , such that for any agent i ∈ [N ], its µ π 1:N -weighted measure of Markov entanglement is bounded,
E i (P π 1:N ) ≤ C/ √ N .
Theorem 6 requires two standard technical conditions for index policies: non-degenerate and uniform global attractor property, which are used in almost all related theoretical work [47,41,20,21] and are detailed in Appendix I. Theorem 6 justifies index polices are asymptotically separable. Combined with an N -agent version of Theorem 4, we obtain the sublinear decomposition error for index policies
Q π 1:N (s, a) - N i=1 Q π i (s i , a i ) µ π 1:N ≤ O( √ N ) .
This sublinear error result explains why the value decomposition in [33,4,5] manages to effectively approximate the global value function in large-scale practical applications.
this section cite: ['b48', 'b45', 'b48', 'b46', 'b19', 'b40', 'b20', 'b32', 'b3', 'b4', 'b29', 'b43', 'b2', 'b46', 'b40', 'b19', 'b20', 'b32', 'b3', 'b4']

Section: Efficient verification of value decomposition
For practitioners, verifying the feasibility of value decomposition is challenging due to the exponential computational complexity of estimating the global Q-value. As a solution, Markov entanglement offers an efficient way to empirically test whether value decomposition can be safely applied. Consider the µ π AB -weighted measure of Markov entanglement in Eq. ( 8), we have
E A (P π AB ) ≈ 1 2 min P A 1 T T t=1 s ′ A ,a ′ A P π AB (s ′ A , a ′ A | s t , a t ) -P A (s ′ A , a ′ A | s t A , a t A ) (9
)
In other words, we can apply a Monte-Carlo estimation for E A (P π AB ). Notice Eq. ( 9) is convex for P A , which enables efficient solutions. As a result, Eq. ( 9) provides an efficient estimation of Markov entanglement via simulation and can be easily extend to N -agent MDPs.
Numerical experiments. Finally, we empirically study the value decomposition for the index policy on a circulant RMAB benchmark [3,52,10,18] that has 4 different states each local agent. As a result, the global state space scales as large as 4 1800 > 10 1000 for N = 1800 agents. The specific transitions and rewards are introduced in Appendix K. For each RMAB instance, we sample a trajectory of length T = 5N and use the collected data to i) solve Eq. ( 9) to estimate the measure of Markov entanglement; ii) train local Q-value decomposition. It quickly follows from the results in Figure 1:
The estimated Markov entanglement decays as O(1/ √ N ) in the left panel, consistent with theoretical predictions. This also implies a low decomposition error scaling of O( √ N ), as seen in the right panel. Furthermore, the simulated trajectory has a length of T = 5N while the global state space has size |S| N , making both entanglement estimation and local Q-value decomposition sample-efficient.
this section cite: ['b2', 'b51', 'b9', 'b17']

Section: Discussions
Comparison with quantum entanglement One notable difference between the definition of Markov and quantum entanglement is that the former does not require coefficients x ≥ 0. In Appendix C, we show there exist separable two-agent MDPs that can only be represented by linear combinations but not convex combinations of independent subsystems, highlighting a structural difference between Markov and quantum entanglement. Finally, we emphasize that our analogy to quantum entanglement is mostly in the mathematical formulation; there is no clear physical interpretation analogy between Markov and quantum entanglement. Relations to Influenced-based MARL There's another line of MARL research that explicitly models the influence of other agents as intrinsic rewards for exploration [45,26]. It turns out the mutual information can be viewed as the measure of Markov entanglement under KL-divergence. Specifically, we can rewrite mutual information in [45] as
I(S ′ 2 , A ′ 2 ; S 2 , A 2 |S 1 , A 1 ) = s,a,s ′ 2 ,a ′ 2 p π (s, a, s ′ 2 , a ′ 2 ) log p π (s ′ 2 , a ′ 2 |s, a) p π (s ′ 2 , a ′ 2 |s 2 , a 2 ) = s,a µ π (s, a)D KL (p π (•|s, a)||P 2 (•|s 2 , a 2 )) .
This is highly related to our measure of Markov entanglement under a µ π -weighted agent-wise KL-divergence, which we can define as
E 2 (P 12 ) = min P2 s,a µ π (s, a)D KL (p π (•|s, a)||P 2 (•|s 2 , a 2 )) .
Intuitively, the measure of Markov entanglement can be viewed as how closely one agent can be approximated as an independent subsystem. This characterization aligns naturally with mutual information. Furthermore, since KL-divergence provides an upper bound for total variation distance, it consequently bounds our Markov entanglement measure relative to the ATV distance introduced in our paper. This connection demonstrates that influence-based MARL methods naturally fit within our theoretical framework, corresponding to a specialized distance measure.
this section cite: ['b44', 'b25', 'b44']

Section: Conclusion
This paper established the mathematical foundation of value decomposition in MARL. Drawing inspiration from quantum physics, we propose the idea of Markov entanglement and prove that it serves as a sufficient and necessary condition for the exact value decomposition. We further characterize the decomposition error in general multi-agent MDPs through the measure of Markov entanglement.
As application examples, we prove widely-used index policies are asymptotically separable and suggest practitioners using Markov entanglement as a proxy for estimating the effectiveness of value decomposition.
this section cite: []

Section: References
Ref_id:b0 Title: Dynamic bid prices in revenue management Year: (2007)
Ref_id:b1 Title: Relaxations of weakly coupled stochastic dynamic programs Year: (2008)
Ref_id:b2 Title: Whittle index based q-learning for restless bandits with average reward Year: (2022)
Ref_id:b3 Title: A better match for drivers and riders: Reinforcement learning at lyft Year: (2024)
Ref_id:b4 Title: Policy optimization for personalized interventions in behavioral health Year: (2023)
Ref_id:b5 Title: Dynamic pricing of relocating resources in large networks Year: (2021)
Ref_id:b6 Title: The best of many worlds: Dual mirror descent for online allocation problems Year: (2023)
Ref_id:b7 Title:  Year: (1996)
Ref_id:b8 Title: A finite time analysis of temporal difference learning with linear function approximation Year: (2021)
Ref_id:b9 Title: Learn to intervene: An adaptive learning policy for restless bandits in application to preventive healthcare Year: (2021)
Ref_id:b10 Title: Dynamic programs with shared resources and signals: Dynamic fluid policies and asymptotic optimality Year: (2022)
Ref_id:b11 Title: Technical note-on the strength of relaxations of weakly coupled stochastic dynamic programs Year: (2023)
Ref_id:b12 Title: Fluid policies, reoptimization, and performance guarantees in dynamic resource allocation Year: (2025)
Ref_id:b13 Title: Neural temporal-difference learning converges to global optima Year: (2019)
Ref_id:b14 Title: The league of robot runners competition: Goals, designs, and implementation Year: (2024)
Ref_id:b15 Title: Understanding value decomposition algorithms in deep cooperative multi-agent reinforcement learning Year: (2022)
Ref_id:b16 Title: Correcting for interference in experiments: A case study at douyin Year: (2023)
Ref_id:b17 Title: Towards q-learning the whittle index for restless bandits Year: (2019)
Ref_id:b18 Title: Reoptimization nearly solves weakly coupled markov decision processes Year: (2022)
Ref_id:b19 Title: Exponential asymptotic optimality of whittle index policy Year: (2023-05)
Ref_id:b20 Title: Linear program-based policies for restless bandits: Necessary and sufficient conditions for (exponentially fast) asymptotic optimality Year: (2024)
Ref_id:b21 Title: Multiagent planning with factored mdps Year: (2001)
Ref_id:b22 Title: Efficient solution algorithms for factored mdps Year: (2003)
Ref_id:b23 Title: Real-time rideshare driver supply values using online reinforcement learning Year: (2022)
Ref_id:b24 Title: Rethinking individual global max in cooperative multi-agent reinforcement learning Year: (2022)
Ref_id:b25 Title: Social influence as intrinsic motivation for multi-agent deep reinforcement learning Year: (2019-06)
Ref_id:b26 Title: Minimizing the age of information in broadcast wireless networks Year: (2016)
Ref_id:b27 Title: Blind dynamic resource allocation in closed networks via mirror backpressure Year: (2024)
Ref_id:b28 Title: Maven: Multi-agent variational exploration Year: (2019)
Ref_id:b29 Title: Neurwin: Neural whittle index network for restless bandits via deep rl Year: (2021)
Ref_id:b30 Title: Quantum computation and quantum information Year: (2010)
Ref_id:b31 Title: Near-optimal reinforcement learning in factored mdps Year: (2014)
Ref_id:b32 Title: Ride-hailing order dispatching at didi via reinforcement learning Year: (2020)
Ref_id:b33 Title: Global rewards in restless multi-armed bandits Year: (2024)
Ref_id:b34 Title: Monotonic value function factorisation for deep multi-agent reinforcement learning Year: (2020)
Ref_id:b35 Title: Weakly coupled deep q-networks Year: (2023)
Ref_id:b36 Title: QTRAN: Learning to factorize with transformation for cooperative multi-agent reinforcement learning Year: (2019)
Ref_id:b37 Title: Value-decomposition networks for cooperative multi-agent learning based on team reward Year: (2018)
Ref_id:b38 Title: Reinforcement Learning: An Introduction Year: (2018)
Ref_id:b39 Title: Analysis of temporal-diffference learning with function approximation Year: (1996)
Ref_id:b40 Title: Asymptotically optimal priority policies for indexable and nonindexable restless bandits Year: (1947)
Ref_id:b41 Title: Duplex dueling multi-agent q-learning Year: (2020)
Ref_id:b42 Title: Towards understanding cooperative multi-agent q-learning with value factorization Year: (2021)
Ref_id:b43 Title: Optimistic whittle index policy: Online learning for restless bandits Year: (2023)
Ref_id:b44 Title: Influence-based multi-agent exploration Year: (2019)
Ref_id:b45 Title: On the gittins index for multiarmed bandits Year: (1992)
Ref_id:b46 Title: On an index policy for restless bandits Year: (1990)
Ref_id:b47 Title: Addendum to 'on an index policy for restless bandits Year: (1991)
Ref_id:b48 Title: Restless bandits: Activity allocation in a changing world Year: (1988)
Ref_id:b49 Title: An approximate dynamic programming approach to network revenue management with customer choice Year: (2009)
Ref_id:b50 Title: Restless bandits with many arms: Beating the central limit theorem Year: (2021)
Ref_id:b51 Title: Near-optimality for infinite-horizon restless bandits with many arms Year: (2022)
