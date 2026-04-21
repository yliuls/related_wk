Title: ON THE SURPRISING EFFECTIVENESS OF A SINGLE GLOBAL MERGING IN DECENTRALIZED LEARNING
Abstract: Decentralized learning provides a scalable alternative to parameter-server-based training, yet its performance is often hindered by limited peer-to-peer communication. In this paper, we study how communication should be scheduled over time, including determining when and how frequently devices synchronize. Counterintuitive empirical results show that concentrating communication budgets in the later stages of decentralized training remarkably improves global test performance. Surprisingly, we uncover that fully connected communication at the final step, implemented by a single global merging, can significantly improve the performance of decentralized learning under high data heterogeneity. Our theoretical contributions, which explain these phenomena, are the first to establish that the globally merged model of decentralized SGD can match the convergence rate of parallel SGD. Technically, we reinterpret part of the discrepancy among local models, which were previously considered as detrimental noise, as constructive components essential for matching this rate. This work provides evidence that decentralized learning is able to generalize under high data heterogeneity and limited communication, while offering broad new avenues for model merging research. Blog post and code are available at Grokking in Decentralized Learning and Code, respectively.

Section: INTRODUCTION
Decentralized learning offers a promising approach to crowdsource computational workloads across geographically distributed compute (Yuan et al., 2022;Borzunov et al., 2023b;Jaghouar et al., 2024). A defining characteristic of this setting is the reliance on peer-to-peer communication during training, involving the peer-level exchange of model parameters or gradients. However, such communication is often constrained in practice due to limited bandwidth between geographically distant nodes, making it a scarce resource. These constraints can significantly degrade the performance of decentralized learning, both theoretically and empirically (Lian et al., 2017;Koloskova et al., 2020;Vogels et al., 2021). As a result, efficiently allocating limited communication resources becomes a fundamental challenge in decentralized learning, especially in heterogeneous environments where varying local data distributions intensify communication demands (Martínez Beltrán et al., 2023).
To date, most efforts addressing this challenge have focused on optimizing communication allocation at the spatial level, particularly through the design of communication graphs (Ying et al., 2021;Li et al., 2022b;Takezawa et al., 2023;Kharrat et al., 2024). In contrast, the temporal allocation of communication, i.e., deciding when and how frequently agents synchronize with others, remains a significant yet underexplored direction for improving decentralized learning. Although temporal communication allocation has been studied in federated learning (FL) (Tang et al., 2020), this problem remains largely untouched in the fully decentralized setting, which is fundamentally different due to the lack of a central server for global aggregation (see discussions in Section 2 and Remark 1).  Question: How to allocate communication budget in decentralized learning over temporal levels?
To answer this question, we design a series of experiments that allocate communication budgets across different time windows during training (see Figure 2). Specifically, we divide the training process into consecutive windows, each with of a fixed number of communication rounds. We assign higher communication budgets to selected windows using global synchronization via AllReduce (Sergeev & Del Balso, 2018), while keeping communication low otherwise by infrequent synchronization with random peer agents. 1 This reveals how the temporal communication allocation affects performance under constrained budgets. We observe that allocating higher communication budgets toward the later stages of training consistently leads to improved final test performance (see Definition 1). More surprisingly, we observe the remarkable effect of a single round of fully-connected communication. 2Surprising Phenomenon: A single global merging of decentralized models, under severely constrained communication and high data heterogeneity, can significantly improve global test performance.
Our Contributions are summarized below.
• Empirical Observations. (1): We highlight the critical role of a single global merging in decentralized training, showing that it can achieve performance close to federated learning, even under severe communication constraints and data heterogeneity (see Figure 1a, Figure 1b). (2):
We observe that limited but non-zero communication preserves a challenging cross-initialization, cross-distribution "mergeability" of local models throughout training (see Definition 2, Figure 1c, and the blue curve in Figure 2c), which does not hold under complete local training (green curve in Figure 1a, Figure 1b). These findings remain consistent across datasets, heterogeneity levels, model architectures, and communication topologies (see Appendix C.3), and provide a first systematic study of global merging in decentralized learning.
• Theoretical Contributions. We investigate the underlying mechanism that enables the mergeability of local models in decentralized learning. Specifically, we provide the first convergence analysis showing that the globally merged model of decentralized SGD can match the rate of parallel SGD (Theorem 1 and Proposition 2). Furthermore, we offer a theoretical explanation for why limited but nonzero communication can ensure mergeability, and why communication should be concentrated in the later stages of training (see Proposition 3).
We anticipate that this work will pave the way for principled decentralized training algorithms capable of generalizing under communication constraints and data heterogeneity, while also advancing model merging research (see discussions in Section 6 and insights in the Q&A Section, Appendix A).
this section cite: ['b107', 'b42', 'b56', 'b46', 'b92', 'b61', 'b106', 'b99', 'b87', 'b44', 'b89']

Section: RELATED WORK

this section cite: []

Section: Temporal Communication Allocation in Parallel, Federated, and Decentralized Learning.
Communication allocation is well-studied in both data-centric parallel learning (Li et al., 2014), and Federated Learning (FL) (McMahan et al., 2017). In parallel learning settings, Gu et al. (2024) proposed a novel strategy for scheduling local steps by analyzing the implicit bias of Local SGD (Gu et al., 2023b). FL extends this server-based paradigm to handle not identically and independently distributed (non-IID) data, but it critically retains a global model. This reliance on a global model has shaped a broad consensus in the FL literature: frequent, early-stage communication is considered essential for aligning local models (Wang et al., 2019;Tang et al., 2020).
In contrast, our work addresses fully decentralized learning, a fundamentally different setting that lacks a central server. Instead of optimizing a generic global model, the goal is to make local models generalize to the global distribution. Despite extensive work focusing on communication allocation at the spatial level in decentralized learning (e.g., designing communication topologies) (Ying et al., 2021;Li et al., 2022b;Takezawa et al., 2023;Kharrat et al., 2024), few studies examine temporal allocation; a pioneering IID study (Kong et al., 2021) showed that stronger early alignment to the global average can modestly improve test performance. These findings do not directly transfer to non-IID settings, where L(•) ≡ L k (•) no longer holds (see Equation (1) and Definition C.2), so they mainly characterize local rather than global generalization (see Definition 1). Due to space constraints, we refer readers to Appendix B.2 and Appendix B.3 for related work on the implicit bias of decentralized learning, and on the topic of model merging.
this section cite: ['b53', 'b64', 'b33', 'b96', 'b89', 'b106', 'b99', 'b87', 'b44', 'b47']

Section: NOTATIONS AND PRELIMINARIES

this section cite: []

Section: NON-IID DECENTRALIZED LEARNING
Decentralized learning formalizes distributed learning as an optimization problem over a connected graph G = (V, E), where V contains m agents and E denotes the communication links. Each agent k ∈ V samples data from a local distribution D k and maintains a local model θ k ∈ R d . The objective is to learn a consensus model θ that minimizes the global population risk (Koloskova et al., 2020):
min θ∈R d L(θ) ≜ 1 m k∈V E ξ k ∼D k L(θ; ξ k ) ,(1)
where
E ξ k ∼D k L(θ; ξ k ) ≜ L k (θ) denotes the local population risk of θ on unseen instance ξ k ∼ D k .
In practice, the optimization of Equation ( 1) is performed under the empirical risk minimization framework, leveraging m local datasets S ≜ m k=1 S k , where S k = {ξ k,1 , . . . , ξ k,ζ } denotes the dataset of agent k sampled from D k . The resulting optimization problem is given by:
min θ∈R d   L S (θ) ≜ 1 m k∈V n k ζ=1 L(θ; ξ k,ζ )   .
(
To solve the optimization problem in Equation ( 2), decentralized algorithms minimize the global empirical risk with only local computations and peer-to-peer communication (Tsitsiklis et al., 1986;Nedic & Ozdaglar, 2009). The communication graph is governed by a weighted adjacency matrix W (t) ∈ [0, 1] m×m , sampled from a distribution W (t) , where each entry W
k,l ≥ 0 reflects the influence of agent l on agent k. 4 Decentralized learning algorithms operate by alternating between local updates and model aggregation through communication with neighbors, as outlined in Algorithm 1.
Algorithm 1 Decentralized Learning input Initialize values θ (0) k ∈ R d on each agent k ∈ V, number of steps T , mixing matrix W 1: in parallel on all agents k ∈ V, for t = 0, . . . , T -1 do 2: Sample training data ξ (t) k from D k , θ (t+1) k ← Optimizer(θ (t) k , ξ (t) k ) ▷ Local update 3: Send θ (t)
k to out-neighbor(s) and receive
{θ (t) l } l∈Nin(k) from in-neighbor(s) ▷ Communication 4: Sample mixing matrix W (t) ∼ W (t) , θ (t+1) k ← l∈Nin(k) W (t) k,l θ (t) l
▷ Gossip averaging 5: end parallel for Practical Evaluation Metrics. In decentralized learning, data heterogeneity and limited training time often prevent a full consensus model θ. We therefore propose to adopt average global test accuracy, a proxy for global population risk, as the primary metric to quantify how well local models generalize to the global data distribution.
Definition 1 (Average Global Test Accuracy). The average accuracy of agents k ∈ V is defined as:
Acc({θ (t) k } k∈V ) = 1 m k∈V Acc(θ (t) k ) Average Accuracy across agents , where Acc(•) ≜ 1 m l∈V E ξ l ∼D l Acc(•; ξ l )
Test accuracy on the global distribution .
Remark 1 (Metric Justification). This metric is specifically designed to address a core question in fully decentralized learning: how well do local models {θ (t) k } k∈V , trained with limited peer-to-peer synchronization, generalize to the global data distribution D? This metric offers a more realistic evaluation for decentralized settings without a global model. See discussions in Appendix C.2.
this section cite: ['b46', 'b90', 'b69']

Section: MERGEABILITY

this section cite: []

Section: Definition 2 (Mergeability under Global Population Risk).
A set of local models {θ k } k∈V is globally mergeable if there exist combination weights {w k } k∈V ∈ [0, 1] such that:
L k∈V w k θ k ≤ k∈V w k L(θ k ),(3)
where L(•) denotes the global population risk.
Definition 2 formalizes the intuition that a linearly interpolated model performs no worse than the original local models. This Definition is inherently non-trivial due to the non-convexity of L.
this section cite: []

Section: EMPIRICAL OBSERVATIONS

this section cite: []

Section: INCREASING IMPACT OF COMMUNICATION IN THE LATER STAGES OF TRAINING
To investigate potential solutions of communication scheduling, we explore a direct strategy: Concentrate communication in a small subset of communication rounds. To this end, we divide the training process into consecutive windows, each consisting of a fixed length of communication rounds. Specifically, the communication scheme is as follows: (1) fully-connected communication (see Figure 1d (b)) is activated only within specific communication windows (i.e., global synchronization via AllReduce (Sergeev & Del Balso, 2018)foot_4 ); (2) while in all other rounds, each agent communicates only with one random peer with a probability of 0.2 (see "Communication Graph" in Appendix C.1).foot_5 As shown in Figure 2, training is divided into 10 (a) and 20 (b) communication windows, respectively. The bars in Figure 2 show both the best global test accuracy achieved during training (lighter-colored bars) and the final test accuracy at the end of training (darker-colored bars). Each bar corresponds to one communication window, where fully connected communication is applied only to the rounds within that window, while random peer communication is used in all other rounds.
For instance, the inset in Figure 2a presents the complete test accuracy trajectory when fully-connected communication is applied during rounds 150 to 180. A consistent trend emerges: allocating communication budgets toward the later stages of training yields substantial improvements, particularly in final test accuracy.
this section cite: []

Section: A SINGLE GLOBAL MERGING SIGNIFICANTLY IMPROVES GLOBAL TEST PERFORMANCE
In Figure 2b, we reduce the fully-connected communication window length to 10 rounds, yet still observe substantial improvements in global test performance. This observation naturally raises the question: What happens if the fully-connected window is reduced to a single round?
To investigate this, we conduct experiments where fully-connected communication is applied only once, implemented by a single global merging. As shown in Figure 1a and Figure 1b, a single global merging is sufficient to significantly improve global test performance. Consistent gains are observed across a wide range of settings, including different datasets, architectures, and communication topologies (see additional experiments in Appendix C.3). The significant increase in performance suggests that the potential of decentralized learning might be considerably underestimated.
Comparisons. D-PSGD (Lian et al., 2017) introduced final global merging under IID settings, but did not analyze the performance gap before and after merging. In contrast, we provide a systematic study of this recovery in challenging non-IID scenarios. Chen et al. (2021) showed the benefit of periodic global averaging, but their method requires frequent global communication (every H = 48 steps), whereas we recover performance with only a single merging. We also compare with Skew-Compensated Sparse Push (SCSP) (Aketi et al., 2021), which also includes a final global merging step. While both works aim to reduce communication, they differ from ours in methodology and setting: (1 ) SCSP uses gradient sparsification (top-k gradients) over a fixed topology, whereas we study mergeability under topological sparsification (sparse gossip); (2) SCSP focuses on one local step (H = 1), whereas we show robust mergeability with multiple local steps (e.g., H = 100) under high heterogeneity. While these works share the broader goal of improving communication efficiency, our work offers a new perspective by investigating the mergeability itself: Why local models retain this property despite extremely limited communication and high data heterogeneity.
Remark 2 (Non-Triviality of the Performance Gain). One may attribute the final improvement to the cumulative effect of sparse gossip. In our setup, each communication round activates one random peer exchange with probability R = 0.2 over T = 300 rounds, yielding about 60 peer exchanges per agent in expectation. This may appear analogous, on average, to multiple implicit global aggregations. If this additive interpretation were sufficient to explain the gain, local models should already be better aligned and achieve performance close to that of the post-merge model. However, our empirical results challenge this interpretation: as shown in Figure 1b, local models under sparse gossip still exhibit poor global test performance, close to the no-communication case, while a single final global merging produces a large performance gain. This gap indicates that the effect of single merging is non-trivial, rather than a simple result of accumulated local gossip.
this section cite: ['b56', 'b13', 'b1']

Section: Communication Cost Comparison.
Let P be the model size, m the number of agents, and T the number of training rounds. A standard Ring-AllReduce-based protocol incurs a total communication cost of O(2mP T ). In contrast, our decentralized setup has a cost of O(mRP T + 2mP ), where R ≪ 2 denotes the expected number of peers per round, and the O(2mP ) term arises from final global merging via Ring-AllReduce. We also note that while a global merging may appear less practical in some cases, it can be effectively approximated via multiple rounds of gossip synchronization among local agents (see our supplementary experiments in Appendix C.3.4).
this section cite: []

Section: MERGEABILITY PERSISTS UNDER LIMITED BUT NONZERO COMMUNICATION
A follow-up question is whether the effectiveness of the global merging is specific to the end of training. To investigate this, we assess the counterfactual performance of the globally averaged model at each training round, as depicted by the light-blue curve in Figure 2c. The experiments are conducted under a lower-communication setting, where each agent communicates with one random peer at each round with probability 0.2 (see "Communication Graph" in Subsection C.1). A consistent superiority of the merged model (light-blue curve) over the local models (dark-blue curve) is observed throughout training, suggesting that local models remain mergeable at all stages (see Definition 2).
As an ablation, we conduct an experiment where all models are trained locally without any communication (see Figure 2c). In this case, the counterfactual test performance of the globally averaged model remains close to zero (light-orange curve), indicating that without communication, local models are not mergeable. This suggests that mergeability does not arise inherently from the local models themselves. Interestingly, under the low-communication setting, the performance of local models before merging (dark-blue curve) remains similar to that in the no-communication case (dark-orange curve). However, after global merging, the models show significant performance improvement. This clear contrast implies that extremely limited but nonzero communication enables mergeability.
this section cite: []

Section: Mergeability without Consensus.
Prior work on gossip algorithms has suggested that local models may converge to a similar state even in minimal communication regimes (Jelasity et al., 2005). In contrast, our work addresses a more challenging heterogeneous data setting where we find that local models do not reach a single consensus point, yet remain mergeable. Specifically, we identify an emergent geometric structure where decentralized training guides local models to a ring-like high-loss region surrounding a central low-loss basin (see Figure 1c). Notably, this corresponds to a more challenging cross-initialization, cross-distribution merging scenario, where local models do not start from a shared pretrained checkpoint and are trained on non-IID data distributions.
this section cite: ['b43']

Section: THEORETICAL ANALYSIS
In this section, we examine the underlying mechanisms that enable the mergeability of local models in decentralized learning. As an initial step, we conduct a fine-grained convergence analysis of the globally merged models trained by Decentralized SGD (DSGD). 8 To substantiate the mergeability of local models, we compare the convergence rate of the merged model trained by DSGD with that of parallel SGD. Remarkably, we prove that the merged model in decentralized learning can match the rate of parallel SGD (Dekel et al., 2012;Li et al., 2014). This supports the empirical findings that the merged model can preserve the performance of individual local models (see Definition 2).
this section cite: ['b18', 'b53']

Section: ASSUMPTIONS
We start by introducing the commonly used assumptions (Kong et al., 2021;Koloskova et al., 2020). Assumption 1 (Mixing matrix). Each sample of the (randomized) mixing matrix W ∈ R m×m is doubly stochastic. Moreover, there exists p > 0 such that
E W ∥ΘW -Θ∥ 2 F ≤ (1 -p)∥Θ -Θ∥ 2 F , ∀ Θ ∈ R d×m .(4)
Here Θ = [θ 1 , . . . , θ m ], Θ = [ θ, . . . , θ] ≡ Θ 1 m 11 ⊤ where θ = 1 m m k=1 θ k . Assumption 2 (Smoothness).
The objective function L is four-times continuously differentiable (i.e., L ∈ C 4 ) and there exist constants L q ≥ 0 for q ∈ {1, . . . , 4} such that:
∥∇ q L(θ)∥ ≤ L q , ∀ θ ∈ R d .(5)
We note that given L ∈ C 2 , the boundedness of the Hessian norm (i.e., the case q = 2) implies that L is L 2 -smooth, thereby recovering Assumption D.1 with (L = L 2 ). Assumption 3 (Bounded noise and diversity). There exist σ 2 , ζ 2 ≥ 0 such that for any θ k ∈ {θ k } m k=1 :
1 m m k=1 E ξ k ∥∇L k (θ k ; ξ k ) -∇L k (θ k )∥ 2 2 ≤ σ 2 , 1 m m k=1 ∥∇L k (θ k ) -∇L(θ k )∥ 2 2 ≤ ζ 2 , (6
)
where
L(θ) = 1 m m k=1 L k (θ).
Here σ measures the local noise level and ζ is a measure of the heterogeneity among agents.
this section cite: ['b47', 'b46']

Section: CONVERGENCE ANALYSIS
Theorem 1 (Non-convex Convergence Rate of DSGD). Suppose Assumption 2 and Assumption 3 hold. Consider decentralized SGD (DSGD) with initializations θ
k = θ (0) for all k ∈ V, and a constant learning rate satisfying η <
2 L2 . Let θ(t) = 1 m m k=1 θ (t)
k denote the averaged model at the t-th step. To achieve an ε-stationary point such that
1 T T -1 t=0 E ∥∇L( θ(t) )∥ 2
2 ≤ ε, the total number of steps T satisfies:
T = O σ 2 mε 2 + 1 ε + T -1 t=0 U (t) • L(θ (0) ) -L ⋆ ,
where
U (t) = (ηL 2 -1)(∇L( θ(t) )) ⊤ ∇ Tr ∇ 2 L( θ(t) ) Γ (t) + Θ(Ξ 3 t ), with Γ (t) = 1 m m k=1 (θ (t) k -θ(t) )(θ (t) k -θ(t) ) ⊤ and the consensus distance Ξ 2 t = Tr(Γ (t) ).
Remark 3. We note that Theorem 1 gives an implicit bound depending on U (t) , t ∈ {1, 2, . . . , T -1}, rather than a closed-form expression. It primarily serves to bridge convergence with the periteration dynamics of U (t) , facilitating the subsequent derivation of the conditions on consensus and communication required to recover the parallel SGD rate (see Proposition 2 and Proposition 3).
Comparison and Technical Novelty. As summarized in Table 1, unified analysis by Koloskova et al. (2020) showed that DSGD suffers from additional terms of order
O 1-p pε + √ p σ+ζ p ε 3/2
in the convergence rate compared to parallel SGD. The core idea behind their analysis is to separate the effects of three key factors: the descent force (i.e., the squared gradient norm), gradient noise, and
) Rate O σ 2 mε 2 + 1 ε O σ 2 mε 2 + 1 pε + √ p σ+ζ p ε 3/2 O σ 2 mε 2 + 1 ε + 1 ε T -1 t=0 U (t)
parameter discrepancy among agents. Each of these components is then analyzed and controlled separately. Among them, both the gradient noise and the model discrepancy are treated as detrimental to convergence. In contrast, we adopt a new proof framework that leverages the implicit bias of decentralized learning (see Proposition D.3 (Zhu et al., 2023b) and Appendix B.2). Rather than treating the discrepancy among agents purely as noise, we partially incorporate it as a constructive component essential for matching the rate of parallel SGD. This intuition is formalized through the convergence guarantee provided in Theorem 1, which introduces an additional term of t) . In what follows, we conduct a fine-grained analysis on the sign of U (t) .
O 1 ε T -1 t=0 U (
Remark 4 (Reduction to Standard Rates). We consider two special cases where the term U (t)  vanishes because the consensus error is identically zero (Ξ t ≡ 0):
• The single-agent case (m = 1);
• The fully synchronous Parallel SGD case, where perfect synchronization ensures identical local models (θ
k ≡ θ(t) for all k). In both settings, the auxiliary term U (t) in Theorem 1 strictly equals zero. Consequently, Theorem 1 naturally recovers the rate of standard (Parallel) SGD, which is of the order O σ 2 mε 2 + 1 ε .
To better characterize how the high-order loss landscape affects the dynamics of U (t) , we introduce a new assumption that is theoretically novel yet empirically supported by prior literature.
this section cite: ['b46', 'b117']

Section: Assumption 4 (Progressive sharpening).
For any positive semi-definite matrix Σ, the gradient of population risk negatively aligns with the gradient of sharpness. Formally, ∀θ ∈ R d ,
∇L(θ) ⊤ ∇ Tr(∇ 2 L(θ)Σ) < 0.(7)
By the density of R, there exists a γ > 0 such that ∇L(θ
) ⊤ ∇ Tr(∇ 2 L(θ)Σ) < -γ∥∇L(θ)∥ < 0 holds strictly. We refer to γ * ≜ sup γ > 0 | ∇L(θ) ⊤ ∇Tr(∇ 2 L(θ)Σ) < -γ∥∇L(θ)∥
as the degree of progressive sharpening. Remark 5. Tr(∇ 2 L(θ)Σ) can be interpreted as an "average sharpness" around θ; see similar metrics in (Gu et al., 2023a;Zhu et al., 2023b). Assumption 4 reflects a widely observed phenomenon in deep learning: The loss gradient exhibits a negative correlation with the gradient of sharpness (Wang et al., 2022;Damian et al., 2023;Cohen et al., 2025). Intuitively, this condition implies that as the optimizer moves to reduce the loss, it simultaneously moves in a direction that increases the sharpness. t) remains negative. In the following, we establish that this term can dominate the other terms in U (t) , thereby ensuring U (t) < 0.
Assumption 4 ensures ∇L( θ(t) ) ⊤ ∇ Tr ∇ 2 L( θ(t) ) Γ (t) in U (
Proposition 2. Suppose Assumption 2 and Assumption 4 hold. Assume η satisfies η > 1/L 2 , and assume ∥∇L( θ(t) )∥ ≥ µ t > 0 for all t. Consider the matrix t) ). Then, for any fixed m > 0, there exists Ξ 2 t > 0 such that
Γ (t) = 1 m m k=1 (θ (t) k -θ(t) )(θ (t) k - θ(t) ) ⊤ and its trace Ξ 2 t = Tr(Γ (
U (t) ≜ 1 2 (ηL 2 -1)∇L( θ(t) ) ⊤ ∇ Tr ∇ 2 L( θ(t) ) Γ (t) + O(Ξ 3 t ) < 0.(8)
Explanations for Assumptions. We assume a lower bound on the global gradient norm evaluated at the averaged parameters θ(t) , i.e., ∥∇L( θ(t) )∥ ≥ µ t > 0. We note that this applies to the gradient on the global data set, which can remain significant even if individual local gradients vanish. The assumption is motivated by the Polyak-Lojasiewicz (PL) condition (Polyak, 1963),
1 2 ∥∇L(θ)∥ 2 ≥ µ(L(θ) -L *
), which ensures the gradient is bounded from zero before reaching the optimum. Our new assumption formalizes this property for the pre-convergence phase by denoting this lower bound at iteration t as 1 2 µ 2 t . We also note that Assumption 2 requires that the norm of the loss derivatives is bounded up to the fourth order, ∥∇ q L(θ)∥ ≤ L q for q = 1, 2, 3, 4. These bounds are necessary to analyze the interaction between the consensus error Ξ t and higher-order landscape.
At a high level, Proposition 2 highlights the potential of leveraging decentralized training to accelerate distributed training beyond communication efficiency. This theoretical insight aligns with the empirical gains observed in Figure C.1a. Specifically, Proposition 2 implies that satisfying U (t) < 0 necessitates careful control of both η and Ξ 2 t . We proceed by discussing the learning rate constraint below, and subsequently analyze the role of Ξ 2 t , based on which we provide a theoretical justification for allocating more communication in the later stages of training in Subsection 5.3. Remark 6 (Acceleration Under Larger Learning Rate). Note that the coefficient of the term
∇L( θ(t) ) ⊤ ∇ Tr ∇ 2 L( θ(t) ) Γ (t) in Inequality (8) is 1 2 (ηL 2 -1).
To ensure the negativity of this term, we require the condition η > 1 L2 . Notably, the resulting interval 1 L2 < η < 2 L2 coincides exactly with the regime of "oscillatory convergence" in classical optimization theory for a quadratic objective L(θ) = 1 2 θ ⊤ Hθ (Polyak, 1987, Chapter 1, p. 26).
To provide deeper intuition, we outline the proof sketch of Theorem 1 below. This analysis establishes a novel descent lemma tailored for decentralized SGD, demonstrating how learning rate control harnesses progressive sharpening to drive acceleration.
E ξ (t) L( θ(t+1) ) ≤ L( θ(t) ) -η - η 2 L 2 2 >0 ∇L( θ(t) ) 2 Standard Descent + η 2 L 2 -η >0 ∇L( θ(t) ) ⊤ ∇Tr ∇ 2 L( θ(t) )Γ (t) <0, progressive sharpening + η 2 L 2 σ 2 2m + O(Ξ 3 t ). (9
)
We note that this third-order effect differs from prior analyses, which were typically established in a near-minima regime (Li et al., 2022c). By contrast, we show that this mechanism emerges whenever local models are inconsistent, i.e., Ξ t > 0.
Equation ( 9) further illuminates the critical role of the consensus violation term Ξ t = Tr(Γ (t) ).
Crucially, the progressive sharpening term scales with O(Ξ 2 t ), whereas the higher-order residual error is O(Ξ 3 t ). Consequently, provided Ξ t is properly controlled such that the O(Ξ 2 t ) gain dominates the O(Ξ 3 t ) error, decentralized SGD can match or even surpass the convergence rate of parallel SGD. According to Corollary D.2, E Ξ 2 t is bounded by
E Ξ 2 t ≤ O (1 -p) p 2 ,(10)
where the parameter p(with p ∈ (0, 1] reflects connectivity in the communication graph (see Assumption 1). A larger p indicates better connectivity and faster consensus, while a smaller p implies a sparse communication graph (i.e., lower communication) and slower information propagation. For example, p = 1 corresponds to a fully connected topology, enabling perfect communication, whereas p = 0 represents the extreme case of complete local training with no communication.
this section cite: ['b117', 'b97', 'b17', 'b14', 'b73']

Section: Why Limited but Nonzero Communication Enables Mergeability.
Notably, random communication graphs can achieve p = Θ(1), striking a favorable trade-off: they require relatively low communication overhead while still maintaining efficient information mixing due to randomized edge sampling, which ensures a rapid decrease of Ξ t (Vos et al., 2023). This is why we adopt random topologies as the primary setup in our experiments: They can satisfy the condition in Proposition 2 even under extremely limited communication, thereby ensuring mergeability (see Figure 1). However, in the case of full local training where p = 0 (see Figure 1d), the right-hand side of Equation ( 10) increases to infinity, indicating that Ξ t may diverge. As a consequence, the condition of Ξ t in Proposition 2 can no longer be satisfied, which explains why local models after complete local training cannot be reliably merged (see the green curve in Figure 1b).
this section cite: ['b94']

Section: A THEORETICAL EXPLANATION FOR COMMUNICATION ALLOCATION
Proposition 2 highlights the importance of Ξ 2 t to satisfy Inequality (8). This motivates the question of how small Ξ 2 t (or how large p) should be, which we answer in the following sufficient condition.
Proposition 3 (Critical Consensus Edge). Suppose Assumption 1-Assumption 4 hold. Assume η > 1 L2 , and the consensus error Ξ t ≤ 1 for all t. Then, the following condition ensures that the critical Inequality (8) is satisfied:
24 1 -p) η 2 p 2 ϕ 2 + σ 2 < min    (ηL 2 -1)γ * µ t 2(ηL 2 + L4 24 ) √ mL 1 , (ηL 2 -1)γ * µ t 2Σ high    , (11
)
where
Σ high = 1 8 ηL 2 L 2 3 + 1 2 η √ mL 2 L 3 + ηmL2L 2 4 1152 .
Here, γ * denotes the degree of progressive sharpening (see Assumption 4), ϕ 2 denotes the uniform upper bound of the averaged squared local gradient norm (i.e.,
1 m m k=1 ∥∇L k (θ (t) k )∥ 2 ≤ ϕ 2
), and µ t is the lower bound on the global gradient norm (i.e., ∥∇L( θ(t) )∥ ≥ µ t > 0).
Practical Guidance. Proposition 3 provides guidance for allocating communication to ensure U (t) ≤ 0, thereby guaranteeing the contribution of each step to the cumulative sum in Theorem 1. From Equation ( 11), note that ϕ, σ 2 , γ * , m, and L q (q ∈ {1, . . . , 4}) are time-independent, while the key time-varying factor is the gradient lower bound µ t , which tracks the optimization status of the averaged parameters θ(t) on the global landscape. Under this interpretation, Equation ( 11) implies that the communication-related term p should be dynamically adjusted in response to the changing landscape geometry captured by µ t . Analytically, the left-hand side of Equation ( 11) is a strictly decreasing function of p, while the right-hand side is an increasing function of µ t . This implies a fundamental trade-off: more frequent communication (larger p) relaxes the condition, whereas a vanishing gradient (smaller µ t ) tightens the allowable error bound. Specifically,
• Early, High-Gradient Regime: In the starting phase of training, when the globally averaged model is far from a minimum, the lower bound on gradient norm µ t is large. This corresponds to a relaxed consensus requirement in Equation ( 11), which permits low-frequency communication (i.e., smaller p) without significantly impacting the performance of the globally merged model.
• Late, Low-Gradient Regime: As models approach a solution and training enters a convergence phase, the gradient norm µ t decreases. This tightens the constraint in Equation ( 11). In this regime, frequent communication (i.e., larger p) becomes critical.
We note that this theoretically motivated guidance aligns well with our empirical findings in Section 4 that more communication should be concentrated in the later stages of training.
this section cite: []

Section: IMPLICATIONS AND DISCUSSIONS
Model Merging. The effectiveness of a single merging of decentralized models has significant implications for model merging. A recent work showed that pre-trained models occupy a large, flat "basic capability basin", within which fine-tuning creates smaller "specific capability basins" (Chen et al., 2025). The observed "mergeability" of local models in our paper implies that decentralized learning may guide agents into connected specific capability basins, allowing simple permutationfree merging to integrate specialized knowledge. This suggests a practical direction: lightweight synchronization during local training may improve basin connectivity and simplify later merging into a more capable model.
Decentralized Learning. Our work provides promising empirical and theoretical evidence that decentralized learning can generalize under high data heterogeneity and limited communication.
More importantly, our findings could directly motivate a new class of adaptive, communicationefficient decentralized learning algorithms, which dynamically allocate their communication budget by monitoring training dynamics to satisfy the critical consensus edge condition in Equation ( 11).
this section cite: ['b10']

Section: LLM USAGE STATEMENT
We use large language models (LLMs) as writing-assistance tools. Their role is confined to proofreading and language polishing.
this section cite: []

Section: IMPACT STATEMENT
This paper studies the problem of temporal communication allocation in decentralized distributed learning, a topic of very high significance in the era of communication-intensive large model training. Specifically, we aim to contribute to the development of communication-efficient decentralized learning without compromising performance. The potential positive social impact are twofold:
• Democratizing Access. For individuals and organizations with constrained infrastructure, our work contributes to the democratization of access to large-scale collaborative training. By reducing communication requirements, we lower the barrier to entry for participating in advanced model development. Such inclusivity can extend the applicability of distributed learning systems to edge environments, thereby promoting more equitable contributions to models trained at scale.
• Reducing Training Costs. In data center environments, our approach can alleviate communication bottlenecks of distributed training. This reduction directly translates to shorter total wall-clock training time, thereby lowering the overall costs and energy consumption associated with large-scale distributed training.
No negative societal impacts are identified.
this section cite: []

Section: ETHICS STATEMENT
Our research strictly adheres to the ICLR Code of Ethics. The work is foundational, focusing on the algorithmic and theoretical properties of decentralized learning, and does not involve human subjects or the collection of new sensitive data. All experiments were conducted on publicly available, standard academic datasets. We foresee no direct negative societal impacts; on the contrary, by reducing communication overhead, our findings may contribute positively by democratizing access to large-scale distributed training and lowering the associated resource footprint.
this section cite: []

Section: REPRODUCIBILITY STATEMENT
We are committed to the reproducibility of our research. Our theoretical claims, including all assumptions and their justifications, are presented in Section 5 with complete, step-by-step proofs provided in Appendix D. Comprehensive details for reproducing our empirical results, including model architectures, data processing, hyperparameter settings, and communication configurations, are well documented in Appendix C.1.
this section cite: []

Section: A LIMITATIONS AND POTENTIAL QUESTIONS
Q: Why use decentralized AdamW in some experiments when the theory is on decentralized SGD?
A:We use decentralized AdamW in some of our experiments for its superior performance in Non-IID settings. Crucially, all reported empirical observations remain fully consistent when using decentralized SGD, directly aligning with our theoretical analysis (see Figure 1 and Subsection C.3).
Q: How does the theory explain local models in decentralized learning are globally mergeable?
A: The theoretical explanation for the "mergeability" of local models in decentralized learning is supported by our result that a globally merged model converges faster to the optimum than individual local models. Specifically, we provide a fine-grained convergence analysis showing that the model merged from decentralized SGD (DSGD) can match the convergence rate of parallel SGD, despite limited communication. Since the rate of m-agent parallel SGD is superior to that of a single local model, this result transitively justifies the merged model's superior performance relative to any individual model, thereby providing theoretical support for mergeability.
Q (Hyperparameter Tuning): How were the baselines tuned in terms of hyperparameters?
A: All hyperparameters were tuned via grid search based on global test performance, with the batch size searched over {64, 128}. For ResNet-18 trained from scratch on Tiny ImageNet, we searched the learning rate over {1 × 10 -4 , 5 × 10 -4 , 1 × 10 -3 } for AdamW and {1 × 10 -3 , 5 × 10 -3 , 1 × 10 -2 } for SGD. For CLIP ViT-B/32 on Tiny ImageNet, we searched the learning rate over {1 × 10 -4 , 5 × 10 -4 , 1 × 10 -3 } for AdamW and {5 × 10 -4 , 1 × 10 -3 , 5 × 10 -3 , 1 × 10 -2 } for SGD. For the optimal hyperparameters selected for our main experiments, please refer to the Implementation Details in Appendix C.1 and the additional empirical results in Subsection C.3).
Q (Comparison with Model Soup): How do initialization schemes affect results? Performance gains from merging have been observed in Model Soup (Wortsman et al., 2022a).
A: We use different initialization schemes and observe consistent performance gains from global merging, whether models start from different random initializations or from a pretrained state.
The majority of our experiments use different initializations, demonstrating that local models in decentralized learning can be effectively merged regardless of their starting points. This is quite surprising, as it contrasts with methods like Model Soup, which require models to be fine-tuned from an identical pretrained state. Furthermore, our experiments with a shared pretrained state confirm that the performance gains hold in that setting as well (see Figure 1a and Subsection C.3).
Q (Methodology for Landscape Visualization): Please clarify the methodology for visualizing the loss landscape in Figure 1c, including the basis for the visualization grid.
A: We adopt the visualization tool from (Crisostomi et al., 2024), positioning 16 trained models at the vertices of a regular hexadecagon. Any point within this polygon is an interpolated model whose parameters are determined by Wachspress barycentric coordinates; we then evaluate its cross-entropy loss to generate the contour map. Unlike methods that use random directions, our visualization grid is deterministically defined by the models themselves, allowing a direct investigation of their geometric connectivity. The full implementation is available in their official code repository https://github.com/crisostomi/cycle-consistent-model- merging/blob/master/notebooks/plots/plot_loss_contours_n_models.ipynb.
this section cite: ['b15']

Section: Q (Experimental Scope):
The empirical findings are restricted to visual tasks.
A: Our empirical findings primarily focus on tasks within the vision domain. We note that this is consistent with most existing decentralized learning literature (Lin et al., 2021;Kong et al., 2021;Ying et al., 2021;Vogels et al., 2021;Li et al., 2022b;Zehtabi et al., 2025). Extending the experimental setup to broader tasks is a meaningful direction for future research.
Q: The finding in Figure 2 (c), that local models eventually converge to a similar state even with limited communication, was also observed in prior work (Jelasity et al., 2005).
A: In our setting, the local models do not, in fact, converge to a similar state or a single consensus point. This is because our work addresses a more challenging heterogeneous data regime, which differs from the setting in the cited prior work. Instead, we identify an emergent geometric structure where decentralized training guides local models to a shared "high-loss ring" surrounding a central low-loss basin (see Figure 1c). Although the models do not reach a consensus, they remain surprisingly mergeable within this region. This geometric arrangement allows their average, i.e., the globally merged model, to fall directly into the low-loss basin. To the best of our knowledge, we are the first to identify this emergent phenomenon in decentralized learning.
this section cite: ['b47', 'b47', 'b106', 'b92', 'b99', 'b111', 'b43']

Section: B ADDITIONAL BACKGROUND AND RELATED WORK B.1 DECENTRALIZED LEARNING
Modern large-scale model training and inference are predominantly conducted within centralized, high-cost data centers. Driven by mounting constraints on computational resources and power availability (Pilz et al., 2025), both academia and industry are increasingly exploring decentralized training approaches (OpenAI, 2025;Grand View Research, 2024). This paradigm, drawing inspiration from swarm intelligence systems (Bonabeau et al., 1999;Mavrovouniotis et al., 2017), offers a more economical and scalable approach by distributing computational tasks across globally distributed nodes, rather than relying solely on a single central server (Yuan et al., 2022;Borzunov et al., 2023b;Jaghouar et al., 2024;Ramasinghe et al., 2025). A notable illustration of the computational potential through decentralization is the Bitcoin system, which sustains workloads equivalent to a 16 GW power draw (CCAF, 2023), surpassing by a factor of three the estimated 5 GW consumption of the largest AI supercluster under development (Gardizy & Efrati, 2024;OpenAI, 2025).
To provide context, we summarize key algorithmic and theoretical advances in decentralized learning.
While our discussion highlights several notable contributions, it is not exhaustive; readers are referred to recent advances and surveys (Zhu et al., 2025;Martínez Beltrán et al., 2023;Singha et al., 2024;Yuan et al., 2024;He et al., 2025;Ramasinghe et al., 2025;Kolehmainen et al., 2025).
this section cite: ['b72', 'b70', 'b3', 'b63', 'b107', 'b42', 'b76', 'b28', 'b70', 'b118', 'b61', 'b84', 'b109', 'b35', 'b76', 'b45']

Section: Algorithmic Progress in Decentralized Learning.
The advancement of decentralized learning algorithms has been primarily driven by the need for communication-efficiency in practical distributed learning. Decentralized algorithms have been refined to handle a variety of realistic scenarios, including time-varying communication topologies (Nedi'c & Olshevsky, 2014;Koloskova et al., 2020;Ying et al., 2021;Takezawa et al., 2023), asynchronous updates (Lian et al., 2018;Xu et al., 2021;Nadiradze et al., 2021;Bornstein et al., 2023;Even et al., 2024), statistical heterogeneity (Tang et al., 2018;Vogels et al., 2021;Le Bars et al., 2023), and robustness to Byzantine failures (He et al., 2022;Ye & Ling, 2025). Moreover, recent works extended beyond standard empirical risk minimization to more structured problem classes, such as compositional (Gao & Huang, 2021), minimax (Xian et al., 2021;Zhu et al., 2023a;Chen et al., 2024), and bi-level optimization (Yang et al., 2022;Gao et al., 2023;Chen et al., 2023). Additionally, privacy concerns in decentralized learning are also critical, with efforts focusing on differentially privacy (Cyffers et al., 2024;Allouah et al., 2024) and data reconstruction attacks (Mrini et al., 2024).
Theoretical Progress in Decentralized Learning. Foundational work on decentralized optimization (Nedic & Ozdaglar, 2009;Sayed, 2014;Yuan et al., 2016;Lian et al., 2017) laid the groundwork for understanding convergence. Building on this, Lu & De Sa (2021) proposed a hierarchical abstraction of decentralization, distinguishing it into three layers, providing a unified view across federated and decentralized paradigms. Koloskova et al. (2020) consolidated synchronous decentralized SGD algorithms with changing communication topologies and local updates, and Even et al. (2024) extended the unifying perspective to asynchronous protocols. More recently, Zehtabi et al. (2025) developed these frameworks further by considering the sporadicity of both communication and computations. On the generalization front, Richards et al. (2020) derived stability-based bounds for decentralized SGD in convex settings, while Sun et al. ( 2021) extended these to non-convex objectives, revealing a dependency on the spectral gap of the communication graph. This dependency was subsequently refined by Zhu et al. (2022), who introduced a Gaussian weight difference assumption to tighten the bound. Complementary results showed that in convex regimes, the generalization of decentralized SGD matches that of centralized SGD (Le Bars et al., 2024), while in non-convex landscapes, decentralization primarily impacts worst-case generalization behavior. To account for unexplained generalization behaviors in decentralized training (Kong et al., 2021;Gurbuzbalaban et al., 2022;Vogels et al., 2023), Zhu et al. (2023b) linked decentralized SGD to random sharpnessaware minimization (SAM), revealing a bias toward flatter minima. Notably, akin to our finding that decentralized learning generalizes when allocated high communication late in training, Zhou et al. (2025) has shown that SAM efficiently selects flatter minima in the later stage of training.
Towards Decentralized Training of Foundation Models. Recent advances have shown the feasibility of training large-scale foundation models in decentralized environments. DT-FM (Yuan et al., 2022) introduced tasklet-based scheduling for Transformer training under bandwidth-constrained settings, enabling efficient resource allocation. SWARM Parallelism (Ryabinin et al., 2023) scaled decentralized training through resilient pipeline design and adaptive load balancing. CocktailSGD (Wang et al., 2023) further improved efficiency via a combination of decentralization, gradient sparsification, and quantization for LLM fine-tuning. On the inference side, Petal (Borzunov et al., 2023a) exploited peer-to-peer networks to amortize computational costs across heterogeneous nodes. Most recently, Intellect (Jaghouar et al., 2024), building on Diloco (Douillard et al., 2023), leveraged hybrid parallelism, i.e., both data and model parallelism, to collaboratively train models with billions of parameters. NoLoCo (Kolehmainen et al., 2025) further extended Diloco to gossip-type decentralized settings. For a broad survey of large-scale deep learning practice, see Shen et al. (2024;2025).
this section cite: ['b68', 'b46', 'b106', 'b87', 'b57', 'b102', 'b66', 'b4', 'b22', 'b88', 'b92', 'b50', 'b37', 'b105', 'b26', 'b101', 'b11', 'b104', 'b27', 'b12', 'b16', 'b2', 'b65', 'b69', 'b81', 'b108', 'b56', 'b116', 'b51', 'b47', 'b34', 'b93', 'b117', 'b114', 'b107', 'b80', 'b95', 'b42', 'b19', 'b45', 'b70']

Section: B.2 IMPLICIT BIAS OF DECENTRALIZED LEARNING
The concept of implicit bias, i.e., the intrinsic preference of learning algorithms for solutions with certain properties, has emerged as a key concept in explaining the empirical success of modern deep learning (Li et al., 2022c;Vardi, 2023;Lyu, 2024). Recent studies have highlighted intriguing distinctions between decentralized stochastic gradient descent (DSGD) and its centralized counterpart (CSGD). Gurbuzbalaban et al. (2022) demonstrated that under certain conditions, DSGD operating on large, sparse topologies exhibits heavier-tailed parameter distributions compared to CSGD. Zhang et al. (2021) showed that decentralization introduces landscape-dependent noise, which can improve tolerance to larger learning rates. This observation aligns with findings by Vogels et al. (2023), who revealed that collaboration in decentralized settings permits the use of larger learning rates. Zhu et al. (2023b) first explicitly characterized the implicit bias of decentralized SGD by establishing its connection with random sharpness-aware minimization, proving the existence of flatness bias in decentralized training. Complementing this, Cao et al. (2024) offered a detailed analysis of the interplay between flatness and optimization in DSGD, particularly its ability to escape local minima. More recently, Wu & Sun (2024) investigated the implicit regularization properties of decentralized optimization in non-convex sparse regression problems, recovering the convergence rates achieved by gradient descent in centralized settings. (2020) discovered that different solutions of deep neural networks can be merged together by simply averaging their parameters. Sonthalia et al. (2025) further showed that the solutions may form a star domain. We note that these phenomena are observed in the following scenarios:
• Shared initialization (Frankle et al., 2020;Fort et al., 2020;Zhou et al., 2023). Models are initialized from a pretrained checkpoint.
• Homogeneous data distribution (Wortsman et al., 2022a). Models are trained on homogeneous data distribution.
• Permutation (Ainsworth et al., 2023;Entezari et al., 2022). Models are independently trained.
The neurons of one model are permuted to match the neurons of the other while maintaining a functionally equivalent network.
These findings have inspired a range of model merging techniques for various applications. Comparisons with Model Merging Literature. Our results show that mode connectivity, or mergeability, can still emerge in decentralized learning, even when the local models are initialized differently, trained on highly heterogeneous data, and merged without any permutation. Our findings offer new insights into both model merging techniques and the geometry of the neural network loss landscape, which we anticipate will motivate further advances in both areas. Dataset. We use three widely adopted image classification datasets: CIFAR-10, CIFAR-100 (Krizhevsky et al., 2009), and Tiny ImageNet (Le & Yang, 2015). CIFAR-10 consists of 60,000 RGB images across 10 classes, while CIFAR-100 contains 60,000 RGB images across 100 classes. The images in both datasets have a spatial resolution of 32 × 32 pixels. Tiny ImageNet is a subset of the ImageNet dataset, comprising 100,000 images drawn from 200 classes, with each image resized to 64 × 64 pixels. It provides a mid-scale benchmark that is more challenging than CIFAR datasets but less computationally demanding than training full ImageNet. To incorporate data augmentation, we employ a combination of RandomCrop with 4-pixel padding, RandomHorizontalFlip, and RandAugment with num_ops=2 and magnitude=9.
Details of Decentralized Learning. We simulate a heterogeneous decentralized learning environment. For our main experiments (Figure 1a and Figure 1b), we use m = 32 agents, while for other experiments, including the sliding window experiments (Figure 2) and the loss landscape visualizations (Figure 1c), we use m = 16 agents. The number of agents for the visualization was chosen as 16 for clarity, as a plot with 32 models would be visually crowded. In all configurations, we employ a Dirichlet distribution characterized by α = 0.1 to partition the data among agents. The Dirichlet distribution is commonly used to partition data in federated learning scenarios, as it allows for the control of label distribution skew among agents (Yurochkin et al., 2019;Hsu et al., 2019). A smaller α results in more imbalanced data distributions, where some agents predominantly receive data from a limited number of classes, while a larger α results in more uniform label distributions across agents. This configuration effectively captures the realistic non-IID nature of decentralized learning, where different agents may have access to personalized data reflective of their local environments.
• Communication Graph. We evaluate three decentralized communication topologies: random graph, ring graph, and exponential graph. In the random graph setting, during each communication round, each agent selects a random subset of its neighbors for gossip averaging. For "R 1", each agent selects exactly one random neighbor in each round. For "R 0.2", each agent selects one neighbor with probability 0.2 and continues local training without communication with probability 0.8. The ring graph enforces a fixed cyclic communication structure, while the exponential graph ensures connectivity by allowing agents to communicate at exponentially increasing distances.
• • Local Data per Agent. Each agent is assigned a subset of the dataset with a fixed size of 4096 samples, drawn according to a Dirichlet distribution to simulate realistic non-IID scenarios.
this section cite: ['b91', 'b60', 'b34', 'b112', 'b93', 'b117', 'b8', 'b100', 'b85', 'b24', 'b23', 'b113', 'b0', 'b21', 'b48', 'b49', 'b110', 'b38']

Section: Model Architecture.
To ensure a representative comparison across different model families, we adopt ResNet-18 (He et al., 2016) and CLIP ViT-B/32 (Radford et al., 2021) as backbone architectures in our experiments. ResNet-18 is a widely used lightweight convolutional neural network that serves as a canonical example of traditional CNN-based architectures. In contrast, CLIP ViT-B/32 is a transformer-based vision model pre-trained on large-scale image-text pairs. For experiments on Tiny ImageNet, where images are resized to 64×64 pixels, we adjust the CLIP visual encoder to handle the lower resolution. With a patch size of 32, each image yields 4 visual tokens arranged in a 2×2 grid, plus a [CLS] token, resulting in a 5-token input sequence.
Implementation Details. All hyperparameters are tuned through grid search based on global test performance (see Definition 1). For experiments using decentralized SGD, the optimal learning rates were found to be 1 × 10 -2 for ResNet-18 (trained from scratch) and 1 × 10 -3 for CLIP ViT-B/32. When using decentralized AdamW, the optimal learning rate is 5 × 10 -4 for ResNet-18 (both when trained from scratch and fine-tuned from ImageNet-pretrained weights) and 1 × 10 -5 for the pretrained CLIP ViT-B/32 on Tiny ImageNet. For all experiments, weight decay is set to 5×10 -4 and the batch size is selected as 128. The key empirical results remain consistent across these optimizer and hyperparameter choices, indicating that our conclusions are stable and not sensitive to specific hyperparameter configurations. 1c. To analyze the geometric connections among models after decentralized training, we visualize the loss landscape spanning their parameter spaces. We adopt the visualization tool from (Crisostomi et al., 2024), which is specifically designed to analyze the interpolation space within the convex hull formed by a given set of models. In our implementation, we position the 16 trained models at the vertices of a regular hexadecagon. Any point within this polygon represents an interpolated model, whose parameters are a weighted sum of the parameters of the 16 vertex models; the weights are determined by the point's Wachspress barycentric coordinates. We then evaluate the cross-entropy loss of each interpolated model on the entire test set to generate the final loss contour map, as shown in Figure 1c. The implementation is available in their notebook https://github.com/crisostomi/cycle-consistent-model- merging/blob/master/notebooks/plots/plot_loss_contours_n_models.ipynb within the official code repository for (Crisostomi et al., 2024). We note two key aspects of this visualization approach:
this section cite: ['b36', 'b75', 'b15', 'b15']

Section: Details of Loss Landscape Visualization in Figure
• Focus on Convex Combinations. For points outside the polygon, one or more of their barycentric coordinates become negative, corresponding to an extrapolation, which is often unstable. This visualization approach is consistent with Definition 2, focusing on the space of convex combinations among the models.
• Deterministic Grid vs. Random Directions. Notably, the visualization method differs from approaches that use random directions to probe the landscape of a single model, as our visualization grid is defined directly by the 16 models themselves. This allows us to directly investigate the geometric connectivity and interpolation properties among this predefined set of models.
this section cite: []

Section: Computational Resource Requirements and Runtime.
To enhance accessibility for researchers working with diverse computational environments, our code includes a centralized simulation of decentralized training. This enables the reproduction and extension of our decentralized learning experiments using fewer GPUs. A single decentralized AdamW training experiment with 16 agents using ResNet-18 on the Tiny ImageNet dataset requires approximately 15 GB of GPU memory and can be conducted on a single GPU with sufficient memory, such as an NVIDIA V100, RTX 3090, RTX 4090, or A100. On an A100 GPU, the typical runtime is approximately 8 hours for 300 communication rounds, each comprising 100 local steps. For the CLIP ViT-B/32 model, the memory demand rises to about 30 GB, yet it remains feasible on a single A100 GPU, with a runtime of approximately 12 hours under the same configuration of 300 communication rounds and 100 local steps per round.
this section cite: []

Section: C.2 PRACTICAL EVALUATION METRICS
The standard evaluation metric of parallel and federated learning is the accuracy of the global model.
Definition C.1 (Test Accuracy of Global Model). The accuracy of the global model θ is defined as:
Acc(θ) ≜ 1 m k∈V E ξ k ∼D k Acc(θ; ξ k ) if IID = E ξ∼D Acc(θ; ξ).
In decentralized learning, models are often evaluated in the absence of a full consensus model θ due to data heterogeneity and limited training time. Two major metrics are adopted in this scenario.
this section cite: []

Section: Definition C.2 (Average Local Test Accuracy).
The average accuracy of agents k ∈ V is defined as:
Acc({θ k } k∈V ) ≜ 1 m k∈V E ξ k ∼D k Acc(θ k ; ξ k )
Average Test accuracy on the local distribution across agents
if IID = 1 m k∈V E ξ∼D Acc(θ k ; ξ).
Remark C.1 (Local Generalization). This metric aims to address the following question in decentralized learning: how well do local models {θ k } k∈V , with the aid of peer-to-peer communication, generalize to their local (personalized) data distribution D l ? This is the standard evaluation metric in personalized decentralized settings, where the goals are to optimize local objectives.
However, in real-world scenarios, local data distributions are often heterogeneous and not guaranteed to be IID across agents. In such settings, an important goal is to understand how well local models, trained on limited local data, generalize to the global data distribution. To account for this, we adopt the following average global test accuracy, a proxy of average global population risk, as the primary evaluation metric, which quantifies how well local models generalize to the global distribution.
this section cite: []

Section: Definition C.3 (Average Global Test Accuracy).
The average accuracy of agents k ∈ V is defined as:
Acc({θ k } k∈V ) = 1 m k∈V Acc(θ k ) Average Accuracy across agents , where Acc(•) ≜ 1 m l∈V E ξ l ∼D l Acc(•; ξ l )
Test accuracy on the global distribution .
Remark C.2 (Global Generalization). This metric is specifically designed to address a core research question in fully decentralized learning with non-IID data: how well do local models {θ k } k∈V , trained with limited peer-to-peer synchronization, generalize to the global data distribution D? We note that this objective is particularly critical in the highly non-IID scenarios we study, where local models drift significantly apart. Unlike federated learning that measures the performance of a global model, this metric offers a more realistic evaluation for decentralized settings where no central server is present.
this section cite: []

Section: C.3 ADDITIONAL EXPERIMENTS C.3.1 DIFFERENT NUMBER OF AGENTS AND OPTIMIZERS
We conduct additional experiments by varying the number of agents (from 16 to 32) and comparing different optimizers (SGD to AdamW). The effect of single merging remains consistent. Decentralized training involves each agent syncing model parameters with a random peer per round with a probability of 0.2, with a single global merging at the final round (see details in Appendix C.1).
this section cite: []

Section: C.3.2 DIFFERENT COMMUNICATION TOPOLOGIES
We also conduct additional experiments with different communication topologies to examine whether the empirical results remain consistent. New observations are summarized below.
• Models remain mergeable under different numbers of peers. We evaluate two settings (random topology with R = 0.2 and R = 1; see "Communication Graph" in Appendix C.1). As shown in -� 0.7 -Federated Training Decentralized Training (R 1) Decentralized Training (R 0.2) u -Local Training u < 0.6 - � rf)_ � 0.5 -� ro ..D 0.4 -0 � d 0.3 - (I) bJ) ro � 0.2 -(I) � 0.1 -0.0 -1:1 �l;:i 1;:il:i �l;:i 1;:il:i �l;:i �l;:i " " ", 'v '? Communication Round (a) Different Number of Peers R (b) Different Topologies Figure C.3: Global test accuracy (see Definition 1) of training ResNet-18 on Tiny ImageNet, distributed across 16 agents with high heterogeneity (Dirichlet α = 0.1; see details in Appendix C.1). We evaluate the effects of different (a) number of peers R, and (b) communication topologies. Pretrained weights are used only in (a). • Models remain mergeable across different communication topologies. We evaluate two topologies: exponential and ring graphs. As shown in Figure C.3b, both topologies preserve the mergeability of local models, with exponential graphs yielding slightly better test performance for both local and merged models. The trend of mergeability persists across topologies throughout training, though performance may vary.
this section cite: []

Section: C.3.3 DIFFERENT HYPERPARAMETERS, DATASET, AND HETEROGENEITY LEVEL
We further conduct supplementary experiments in which the final global merge is approximated by topology-constrained gossip merging on an exponential graph (Ying et al., 2021).
Summary. Consistent test performance improvement of a single global merging across a wide range of settings are observed, including different hyperparameter setups, datasets, degree of data heterogeneity, model architectures, optimizers, initialization schemes, and communication topologies.
this section cite: ['b106']

Section: C.3.4 REALIZING FINAL GLOBAL GOSSIP VIA TOPOLOGY-CONSTRAINED MERGING
We also perform additional experiments where the final global merging is approximated by multiple rounds of gossip merging, i.e., a specific exponential topology (Ying et al., 2021). (a) (b) (c) (d) Figure C.5: Global test accuracy (see Definition 1) for ResNet-18 trained with decentralized AdamW across 32 agents under different levels of data heterogeneity (Dirichlet α = 0.1 (a, c) vs. α = 1.0 (b, d); see Appendix C.1). Results are reported on both CIFAR-100 (a, b) and Tiny ImageNet (c, d). Left: regular decentralized training followed by one round of topologyconstrained final gossip merging on an exponential graph. Middle: the same topology-constrained setting, but with five rounds of final gossip merging to better approximate global aggregation. Right: baseline setting (our original approach), with random communication among all agents during training followed by one perfect global merge.
Summary. We observe that (1) even a single round of topology-constrained final gossip merging substantially improves global test accuracy, and (2) the resulting performance is comparable to the baseline that uses random communication among all agents followed by one perfect global merge.
this section cite: ['b106']

Section: D THEORY
This section provides the proofs of the main theoretical results presented in this paper. For simplicity, and following the setup in the existing literature, we assume that the sample size of local agents is n k = n for all k ∈ V.
Lemma D.1 (Consensus Distance Recursion under Local Updates (Kong et al., 2021)). Suppose Assumption 1-Assumption 3 hold. Let θ (t)
k be the local parameter on client k at t-th step, and denote their average by θ(t) = 1 m m k=1 θ (t) k . Define the consensus distance and the average gradient norm at round t by
Ξ 2 t = 1 m m k=1 ∥θ (t) k -θ(t) ∥ 2 and ϕ 2 t = 1 m m k=1 ∥∇L k (θ (t) k )∥ 2 , where L k (θ) = E ξ k ∼D k [L(θ; ξ k )].
Let η > 0 the learning rate, and σ 2 the variance bound from Assumption 3. Then there exists a constant p > 0 (see Assumption 1) such that for all t ≥ 0, the following inequality holds:
E Ξ 2 t+1 ≤ 1 - p 2 Ξ 2 t + 12(1 -p) p η 2 ϕ 2 t + σ 2 , (D.1)
where the expectation is taken over the stochastic gradients in the t-th update phase.
Proof. For completeness, we provide the proof of Lemma D.1, with minor corrections and additional details. In decentralized SGD (Algorithm 1 with SGD as the local optimizer), each agent k ∈ V performs at each iteration
θ (t+1) k = m l=1 W k,l θ (t) l -η ∇L l (θ (t) l ; ξ (t) l ) .
In matrix form, letting
Θ (t) = [θ (t) 1 , . . . , θ (t) m ] ∈ R d×m , ∇L(Θ (t) ; ξ (t) ) = [∇L 1 (θ (t) 1 ; ξ (t) 1 ), . . . , ∇L m (θ (t) m ; ξ (t) m )], we have Θ (t+1) = Θ (t) -η ∇L(Θ (t) ; ξ (t) ) W.
The consensus matrix after mixing is
Θ(t+1) = Θ (t+1) 1 m 11 ⊤ = Θ (t) -η ∇L(Θ (t) ; ξ (t) ) 1 m 11 ⊤ , since 1 ⊤ W = 1 ⊤ .
Thus the consensus distance satisfies
m Ξ 2 t+1 = Θ (t+1) -Θ(t+1) 2 F = Θ (t) -η ∇L(Θ (t) ; ξ (t) ) (W -1 m 11 ⊤ ) 2 F .
By Assumption 1, for any Θ ∈ R d×m ,
E W ΘW -Θ 2 F ≤ (1 -ρ) Θ -Θ 2 F , we obtain, m Ξ 2 t+1 ≤ (1 -p) Θ (t) (I -1 m 11 ⊤ ) -η ∇L(Θ (t) ; ξ (t) )(I -1 m 11 ⊤ ) 2 F . Applying the inequality ∥A + B∥ 2 F ≤ (1 + α)∥A∥ 2 F + (1 + 1/α)∥B∥ 2 F with α = p 2 gives m Ξ 2 t+1 ≤ (1 -p) (1 + p 2 ) Θ (t) (I -1 m 11 ⊤ ) 2 F + (1 + 2 p ) η 2 ∇L(Θ (t) ; ξ (t) ) 2 F ≤ 1 -p 2 m Ξ 2 t + 6(1 -p) p η 2 ∇L(Θ (t) ; ξ (t) ) 2 F ,
where we use (1 + p/2) ≤ 1 + p and (1 + 2/p) ≤ 6/p for p ∈ (0, 1).
Published as a conference paper at ICLR 2026
We now decompose the stochastic gradient as
∇L(Θ (t) ; ξ (t) ) = ∇L(Θ (t) ) + ∇L(Θ (t) ; ξ (t) ) -∇L(Θ (t) ) ,
so by Young's Inequality, we have
∇L(Θ (t) ; ξ (t) ) 2 F ≤ 2 ∇L(Θ (t) ) 2 F + 2 ∇L(Θ (t) ; ξ (t) ) -∇L(Θ (t) ) 2 F .
Taking expectation over ξ (t) and invoking Assumption 3, we get
E ∥∇L(Θ (t) ; ξ (t) )∥ 2 F ≤ 2∥∇L(Θ (t) )∥ 2 F + 2 σ 2 m.
Substituting back and dividing by m yields
E Ξ 2 t+1 ≤ 1 -p 2 Ξ 2 t + 12(1 -p) p η 2 ϕ 2 t + σ 2 ,
which completes the proof.
Corollary D.2 (Upper Bounds of Consensus Distance (Kong et al., 2021)). Define the consensus distance and the average gradient norm at round t by
Ξ 2 t = 1 m m k=1 ∥θ (t) k -θ(t) ∥ 2 and ϕ 2 t = 1 m m k=1 ∥∇L k (θ (t) k )∥ 2 , where L k (θ) = E ξ k ∼D k [L(θ; ξ k )]
. Under the conditions of Lemma D.1, suppose that for all iterations t, the gradient norms are uniformly bounded by a constant ϕ, i.e. ϕ 2 t ≤ ϕ 2 , ∀t ∈ {1, . . . , T }. Then the expected consensus distance satisfies
E Ξ 2 t ≤ 24 (1 -p) η 2 p 2 ϕ 2 + σ 2 .
In the general case where the gradient-norms change slowly, i.e., ϕ 2 t ≤ (1 + p 4 ) ϕ 2 t+1 , we have
E Ξ 2 t ≤ 48 (1 -p)η 2 p 2 ϕ 2 t-1 + σ 2 .
The expectation here is taken over the stochastic gradients in the t-th update phase.
Proof. Consider the key recursion from Lemma D.1:
E Ξ 2 t+1 ≤ 1 -p 2 Ξ 2 t + 12 (1 -p) p η 2 ϕ 2 t + σ 2 .
(1) Special Case: uniformly bounded gradient norms.
Assume ϕ 2 t ≤ ϕ 2 . Unrolling the above gives
E Ξ 2 t+1 ≤ t-1 i=0 1 -p 2 i 12 (1 -p) p η 2 (ϕ 2 + σ 2 ). Since t-1 i=0 (1 -p 2 ) i ≤ 2 p , we can bound the consensus distance as E Ξ 2 t+1 ≤ 12 (1 -p) p η 2 (ϕ 2 + σ 2 ) × 2 p = 24 (1 -p) η 2 p 2 (ϕ 2 + σ 2 ),
which yields the first claim.
(2) Special Case: slowly changing gradient norms.
If ϕ 2 t ≤ (1 + p 4 ) ϕ 2 t+1 , and since
1 -p 2 i 1 + p 4 i ≤ 1 -p 4 i , the consensus distance satisfies E Ξ 2 t+1 ≤ t-i-1 i=0 1 - p 2 i 12(1 -p)η 2 ϕ 2 t-1 + σ 2 p ≤ t-1 i=0 1 - p 4 i 12(1 -p)η 2 (ϕ 2 t-1 + σ 2 ) p ≤ 48(1 -p)η 2 p 2 ϕ 2 t-1 + σ 2 . (D.2)
Proposition D.3 (Implicit Bias of Decentralized SGD (Zhu et al., 2023b)). Suppose Assumption 2 hold, the globally averaged model of decentralized SGD (DSGD), defined by θ
(t) = 1 m m k=1 θ (t)
k , follows the following gradient descent direction:
E ξ (t) [ θ(t+1) ] = θ(t) -η • E ϵ (t) ∼N (0,Γ (t) ) ∇L( θ(t) + ϵ (t) ) + δ (t) , where Γ (t) = 1 m m k=1 (θ (t) k -θ(t) )(θ (t) k -θ(t) ) ⊤ ∈ R m×m denotes the consensus distance matrix, and δ (t) = O η m m k=1 ∥θ (t) k -θ(t) ∥ 3
2 denotes the high-order terms. The first expectation eliminates the randomness from sampled data ξ
(t) = {ξ (t) k } k∈V at step (t).
We can then control the expected squared distance between two consecutive steps of the globally averaged model with Corollary D.4.
Corollary D.4. Under the assumptions in Proposition D.3, the expected squared distance between two consecutive iterates of decentralized SGD can be bounded as follows:
E ξ (t) θ(t+1) -θ(t) 2 ≤ η 2 σ 2 m + η 2 ∇L θ(t) + 1 2 ∇ Tr ∇ 2 L( θ(t) )Γ (t) + δ (t) 2 . (D.3) Proof. Denote γ (t+1) = E ξ (t) θ(t+1) -θ(t)
. We can expand the expected distance as follows:
E ξ (t) θ(t+1) -θ(t) 2 = E ξ (t) θ(t+1) 2 + θ(t) 2 -2( θ(t) ) ⊤ γ (t+1) = Tr Cov( θ(t+1) ) + E ξ (t) θ(t+1) 2 + θ(t) 2 -2( θ(t) ) ⊤ γ (t+1) = Tr Cov( θ(t+1) ) + E ξ (t) [ θ(t+1) -θ(t) ] 2 = Tr Cov( η m m k=1 ∇L(θ (t) k ; ξ (t) k )) + E ξ (t) [ θ(t+1) -θ(t) ] 2 , (D.4)
where the second equality follows from the definition of the covariance matrix, namely
Tr Cov( θ(t+1) ) = E ξ (t) θ(t+1) 2 -E ξ (t) θ(t+1) 2 ,
and the final equality is derived from the original update of decentralized SGD (without applying Proposition D.3):
θ(t+1) = θ(t) -η • 1 m m k=1 ∇L(θ (t) k ; ξ (t) k ).
According to the convexity of the vector norm and the fact that
Tr Cov( 1 m m k=1 ∇L(θ (t) k ; ξ (t) k )) = E ξ (t) 1 m m k=1 ∇L(θ (t) k ; ξ (t) k ) - 1 m m k=1 E ξ (t) k ∇L(θ (t) k ; ξ (t) k ) 2 , (D.5)
we then complete the proof by applying Proposition D.3 and the bounded noise assumption in Assumption 3.
Published as a conference paper at ICLR 2026
Corollary D.5. Let Γ (t) = 1 m m k=1 (θ (t) k -θ(t) )(θ (t) k -θ(t) ) ⊤ ∈ R d×d , where θ(t) = 1 m m k=1 θ (t)
k ∈ R d denotes the globally averaged model across m agents. Suppose Assumption 2 hold. Then, for ϵ (t) ∼ N (0, Γ (t) ), the expected gradient perturbation satisfies:
E ϵ (t) ∼N (0,Γ (t) ) ∇L( θ(t) + ϵ (t) ) -∇L( θ(t) ) = 1 2 ∇ Tr ∇ 2 L( θ(t) )Γ (t) + E ϵ (t) ∼N (0,Γ (t) ) R 3 (ϵ (t) ) , (D.6)
where ∥R 3 (ϵ (t) )∥ is bounded by L4 24 ∥ϵ (t) ∥ 3 .
Proof. We apply the third-order Taylor expansion to ∇L around θ(t) :
∇L( θ(t) + ϵ (t) ) = ∇L( θ(t) ) + ∇ 2 L( θ(t) )ϵ (t) + 1 2 ∇ 3 L( θ(t) )[ϵ (t) , ϵ (t) ] + R 3 (ϵ (t) ),
with the remainder: t) due to properties of the Gaussian distribution. The remainder bound can be bounded as
R 3 (ϵ (t) ) = 1 0 (1 -τ ) 3 6 ∇ 4 L( θ(t) + τ ϵ (t) )[ϵ (t) , ϵ (t) , ϵ (t) ]dτ. Taking expectations over ϵ (t) ∼ N (0, Γ (t) ), since E[ϵ (t) ] = 0, the linear term vanishes. The quadratic term E ∇ 3 L( θ(t) )[ϵ (t) , ϵ (t) ] simplifies to ∇ Tr ∇ 2 L( θ(t) )Γ (
∥R 3 (ϵ (t) )∥ ≤ 1 0 (1 -τ ) 3 6 L 4 ∥ϵ (t) ∥ 3 dτ = L 4 ∥ϵ (t) ∥ 3 • 1 6 1 0 (1 -τ ) 3 dτ. Since 1 0 (1 -τ ) 3 dτ = 1 4 , we have: ∥R 3 (ϵ (t) )∥ ≤ L 4 ∥ϵ (t) ∥ 3 • 1 6 • 1 4 = L 4 24 ∥ϵ (t) ∥ 3 .
For comparison, we restate the convergence rate of DSGD by Koloskova et al. (2020). Assumption D.1 (L-smoothness). Each population risk L k = E ξ k ∼D k L(θ; ξ k ) for k ∈ {1, . . . , m} is continuously differentiable, and there is a constant L ≥ 0 such that:
∥∇L k (θ) -∇L k (ϑ)∥ ≤ L∥θ -ϑ∥, ∀ θ, ϑ ∈ R d .
(D.7) Theorem D.6 (Non-convex Convergence Rate of DSGD (Koloskova et al., 2020)). Under Assumption 1, Assumption D.1 and Assumption 3, let the learning rate η satisfy η ≤ η
max = O p L let θ(t) = 1 m m k=1 θ (t)
k denote the averaged model at the t-th step. To achieve an ε-stationary point such that
1 T T -1 t=0 E ∥∇L( θ(t) )∥ 2
2 ≤ ε, the total number of steps T satisfies:
T = O σ 2 m ε 2 + √ p σ + ζ p ε 3/2 + 1 pε • L θ 0 -L ⋆ .
We then provide our main theoretical results as follows.
Theorem D.7 (Non-convex Convergence Rate of DSGD). Suppose Assumption 2 and Assumption 3 hold. Consider decentralized SGD (DSGD) with initializations θ
k = θ (0) for all k ∈ V, and a constant learning rate satisfying η <
2 L2 . Let θ(t) = 1 m m k=1 θ (t)
k denote the averaged model at the t-th step. To achieve an ε-stationary point such that
1 T T -1 t=0 E ∥∇L( θ(t) )∥ 2
2 ≤ ε, the total number of steps T satisfies:
T = O σ 2 mε 2 + 1 ε + T -1 t=0 U (t) • L(θ (0) ) -L ⋆ ,
where
U (t) = 1 2 (ηL 2 -1)∇L( θ(t) ) ⊤ ∇ Tr ∇ 2 L( θ(t) ) Γ (t) + Θ(Ξ 3 t ), with Γ (t) = 1 m m k=1 (θ (t) k -θ(t) )(θ (t) k -θ(t) ) ⊤
and the consensus distance Ξ 2 t = Tr(Γ (t) ).
Proof. We structure the proof into several key steps.
this section cite: ['b47', 'b47', 'b117', 'b46', 'b46']

Section: Step (A): Descent Force Decomposition.
Based on the L 2 -smoothness (Assumption D.1) of the loss function L (as implied by Assumption 2)), we can apply the first-order Taylor expansion around θ(t) to establish an upper bound for L( θ(t+1) ):
L θ(t+1) ≤ L θ(t) + ∇L θ(t) ⊤ θ(t+1) -θ(t) + L2 2 θ(t+1) -θ(t) 2 .
According Proposition D.3, we have
E ξ (t) [ θ(t+1) ] = θ(t) -η ∇L θ(t) + 1 2 ∇ Tr(∇ 2 L( θ(t) )Γ (t) ) + δ (t) ,
where Γ (t) denotes the variance matrix of ϵ
(t) ∼ N (0, Γ (t) ) and δ (t) = Θ η m m k=1 ∥θ (t) k -θ(t) ∥ 3 2
denotes the high-order residuals (see Proposition D.3).
Substituting this into the previous bound and taking the expectation with respect to random data sampling yields:
E ξ (t) [L θ(t+1) ] ≤ L θ(t) -η ∇L θ(t) ⊤ ∇L θ(t) + 1 2 ∇ Tr(∇ 2 L( θ(t) )Γ (t) ) -δ (t) + E ξ (t) L2 2 θ(t+1) -θ(t) 2 .
According to Corollary D.4, we obtain
E ξ (t) θ(t+1) -θ(t) 2 ≤ η 2 σ 2 m + η 2 ∇L θ(t) + 1 2 ∇ Tr(∇ 2 L( θ(t) )Γ (t) ) + δ (t) 2 .
We can then decompose the squared norm:
∇L θ(t) + 1 2 ∇ Tr(∇ 2 L( θ(t) )Γ (t) ) 2 = 1 4 ∇ Tr(∇ 2 L( θ(t) )Γ (t) ) 2 + ∇L θ(t) 2 + ∇ Tr(∇ 2 L( θ(t) )Γ (t) ) ⊤ L θ(t) .
Combining the previous steps, we obtain:
E ξ (t) L θ(t+1) ≤ L θ(t) -(η -η 2 L2 2 ) ∇L θ(t) 2 + η 2 L 2 8 ∇ Tr(∇ 2 L( θ(t) )Γ (t) ) 2 T1 1 2 (-η + η 2 L 2 ) • ∇L θ(t) ⊤ ∇ Tr(∇ 2 L( θ(t) )Γ (t) ) T2 +η ∇L θ(t) ⊤ δ (t) T3 + η 2 σ 2 m • L 2 2 + η 2 L 2 ∇L θ(t) + 1 2 ∇ Tr(∇ 2 L( θ(t) )Γ (t) ) ⊤ δ (t) T4 + η 2 L 2 2 δ (t) 2 T5 . (D.8) We subsequently control terms related to E ϵ (t) ∼N (0,Γ (t) ) ∇L θ(t+ 1 2 ) ) -∇L θ(t) in Equation (D.8).
this section cite: []

Section: Step (B): Control Consensus-Related Terms
Combining the residual upper bound in Corollary D.5 and with the concavity of (•)
3/2 , we can derive
δ (t) ≤ L 4 24 • 1 m m k=1 θ (t) k -θ(t) 3 ≤
For clarity, we consolidate the terms involving T 1 to T 2 in Equation (D.8):
Accumulated T -terms =(-η + η 2 L 2 )T 2 + ηT 3 + η 2 L 2 8 (T 1 + 8T 4 + 4T 5 ). (D.9)
Substituting the Upper bounds for the Accumulated T -terms into Equation (D.8) yields the updated descent inequality:
E ξ (t) L θ(t+1) ≤ L θ(t) -η -η 2 L2 2 ∇L θ(t) 2 + η U (t) + η 2 σ 2 L2 2 , (D.10)
where η U (t) denotes the upper bound for Equation (D.9):
η U (t) ≜(η 2 L 2 -η)T 2 + η √ mL 1 L 4 24 Ξ 3 t + 1 8 η 2 L 2 L 2 3 Ξ 4 t + 1 2 η 2 √ mL 2 (2L 1 + L 3 Ξ 2 t )Ξ 3 t + η 2 mL 2 L 2 4 1152 Ξ 6 t , = (η 2 L 2 -η)T 2 ≜ A (t) = Θ(Ξ 2 t ) + η 2 L 2 + ηL 4 24 √ mL 1 + η 2 8 L 2 L 2 3 Ξ t + η 2 2 √ mL 2 L 3 Ξ 2 t + η 2 mL 2 L 2 4 1152 Ξ 3 t Ξ 3 t ≜ H (t) = O(Ξ 3 t ) ,(D.11)
and we recall that the consensus distance Ξ 2 t = 1 m m k=1 θ
k -θ(t) 2 . To facilitate subsequent analysis, we further separate U (t) into an Acceleration term A (t) plus High-order terms H (t) , With U (t) serving as a unified proxy for the consensus errors, we obtain the new rate as follows.
this section cite: []

Section: Step (C): Derive the Convergence Rate
Starting from the descent inequality Equation (D.10):
E ξ (t) L( θ(t+1) ) ≤ L( θ(t) ) -η - η 2 L 2 2 ∇L( θ(t) ) 2 + η U (t) + σ 2 m η 2 L 2 2 .
Taking full expectation and summing over t = 0, . . . , T -1, we obtain
T -1 t=0 η - η 2 L 2 2 E ∇L( θ(t) ) 2 ≤ L(θ (0) ) -E L( θ(T ) ) + η T -1 t=0 U (t) + σ 2 η 2 L 2 T 2m .
To ensure the descent property of -∇L( θ(t) ) 2 , we have to set η -η 2 L2 2 ≥ 0, which in turn implies that η ≤ ℓ L2 , with ℓ < 2. Under this condition, and denoting ∆ = L( θ(0) ) -L * , we obtain
1 T T -1 t=0 E ∇L( θ(t) ) 2 ≤ 2∆ (2 -ℓ) η T + 2 (2 -ℓ) η 1 T T -1 t=0 U (t) + σ 2 ηL 2 (2 -ℓ) m . (D.12)
Published as a conference paper at ICLR 2026
To ensure this is at most ε, it suffices to enforce
σ 2 ηL 2 (2 -ℓ) m ≤ ε 3 , 2 (2 -ℓ) 1 T T -1 t=0 U (t) ≤ ε 3 , and 2∆ (2 -ℓ) η T ≤ ε 3 . (D.13)
To satisfy all three conditions simultaneously, along with a stability condition η ≤ ℓ L2 , we should select η accordingly: t) .
η ≤ min ℓ L 2 , (2 -ℓ)mε 3σ 2 L 2 , and T ≥ max 6 ∆ (2 -ℓ)ηε , 6 (2 -ℓ)ε T -1 t=0 U (
To ensure a valid step-size η exists, we substitute these three upper bounds into the condition for T . This yields three distinct lower bounds on the total number of iterations T that must be satisfied. By rearranging the inequality T η ≥ 6∆ ε , we require:
T ≥ max 6∆L 2 ℓ (2 -ℓ) ε , 18∆σ 2 L 2 (2 -ℓ) 2 mε 2 , 6 (2 -ℓ)ε T -1 t=0 U (t) ,
where the first two bounds are derived directly by substituting the first two terms from the min{•} operation for η into the first lower bound of T . Therefore, the total number of iterations T should be large enough to satisfy all applicable lower bounds. This leads to the sufficient condition:
T = O ∆ ε + ∆ σ 2 m ε 2 + 1 ε T -1 t=0 U (t) ,
This condition is sufficient to guarantee
1 T T -1 t=0 E ∇L( θ(t) ) 2 2 ≤ ε.
The proof is now complete.
Proposition D.8. Suppose Assumption 2 and Assumption 4 hold. Assume η > 1/L 2 , and assume ∥∇L( θ(t) )∥ ≥ µ t > 0 for all t. Consider the matrix
Γ (t) = 1 m m k=1 (θ (t) k -θ(t) )(θ(t)
k -θ(t) ) ⊤ and its trace Ξ 2 t = Tr(Γ (t) ). Then, for any fixed m > 0, there exists Ξ 2 t > 0 such that
U (t) ≜ 1 2 (ηL 2 -1) ∇L( θ(t) ) ⊤ ∇ Tr ∇ 2 L( θ(t) ) Γ (t) =Θ(Ξ 2 t )
+Θ(Ξ 3 t ) < 0. (D.14) Remark D.1. We note that Proposition D.8 does not contradict Equation (D.12) when both ∆ and σ are zero. The condition ∆ = L( θ(0) ) -L * = 0 implies that the models are initialized at an optimal point. In Theorem D.7, we assume that all initializations are identical (θ
k = θ (0) , ∀k ∈ V), so it follows that all models begin at the same optimum. Consequently, the consensus error remains zero throughout all iterations, meaning the model covariance matrix Γ (t) is the zero matrix and its trace Ξ t is also zero. Since every component of the term U (t) , defined as
U (t) ≜ (ηL 2 -1)T 2 ≜ A (t) = Θ(Ξ 2 t ) + ηL 2 + L 4 24 √ mL 1 + 1 8 ηL 2 L 2 3 Ξ t + 1 2 η √ mL 2 L 3 Ξ 2 t + ηmL 2 L 2 4 1152 Ξ 3 t Ξ 3 t ≜ H (t) = O(Ξ 3 t )
.
(D.15)
Proof. The proof relies on establishing that for sufficiently small Ξ t , the negative leading term A (t)  in the decomposition of U (t) dominates the higher-order residual term H (t) . Specifically, A (t) is of order Θ(Ξ 2 t ), while H (t) is of order O(Ξ 3 t ). (t) . Let g (t) = ∇L( θ(t) ) and recall that t) .
this section cite: []

Section: Step (A): Derive Upper Bound on A
T 2 ≜ (g (t) ) ⊤ ∇ Tr ∇ 2 L( θ(t) ) Γ (
Based on the definition of U (t) in Equation (D.15), we have the leading term A (t) = (ηL 2 -1)T 2 . To analyze T 2 , consider the trilinear form defined on the unit sphere. Let δ
k be the deviation vectors such that
Γ (t) = 1 m m k=1 δ (t) k (δ (t) k ) ⊤ and Ξ 2 t = 1 m m k=1 ∥δ(t)
k ∥ 2 . Define the function F (δ) for δ ̸ = 0:
F (δ) = ∇ 3 L( θ(t) )[δ, δ, g (t) ] ∥g (t) ∥∥δ∥ 2 .
Under Assumption 4, we have ∇ 3 L( θ(t) )[δ, δ, g (t) ] < 0, implying F (δ) < 0. Since the unit sphere S = {δ : ∥δ∥ = 1} is compact, F attains a maximum value M < 0. Let γ = -M > 0. It follows that for any vector δ,
∇ 3 L( θ(t) )[δ, δ, g (t) ] ≤ -γ ∥g (t) ∥∥δ∥ 2 .
Substituting this bound into the summation for T 2 :
T 2 = 1 m m k=1 ∇ 3 L( θ(t) )[δ (t) k , δ (t) k , g (t) ] ≤ 1 m m k=1 -γ∥g (t) ∥∥δ (t) k ∥ 2 = -γ∥g (t) ∥Ξ 2 t .
Given the gradient lower bound ∥g (t) ∥ ≥ µ t , we obtain T 2 ≤ -γµ t Ξ 2 t . Consequently, assuming the pre-condition ηL 2 -1 > 0 holds, the leading term A (t) satisfies:
A (t) = (ηL 2 -1)T 2 ≤ -(ηL 2 -1)γµ t Ξ 2 t . (D.16)
This confirms that A (t) provides a strictly negative contribution of order Θ(Ξ 2 t ).
this section cite: []

Section: Step (B): Dominance over Higher-Order Residuals.
We now show that U (t) = A (t) + H (t) < 0 for small Ξ t . According to Equation (D.15), the residual term is given by
H (t) = ηL 2 + L 4 24 √ mL 1 + 1 8 ηL 2 L 2 3 Ξ t + 1 2 η √ mL 2 L 3 Ξ 2 t + ηmL 2 L 2 4 1152 Ξ 3 t ≜P (Ξt) Ξ 3 t .
Here, P (Ξ t ) is a polynomial in Ξ t with positive coefficients, and thus H (t) = O(Ξ 3 t ). The condition U (t) < 0 is equivalent to H (t) < -A (t) . Using the bound from Equation (D.16), it suffices to show:
P (Ξ t )Ξ 3
t < (ηL 2 -1)γµ t Ξ 2 t . For Ξ t > 0, we divide both sides by Ξ 2 t , reducing the condition to: Ξ t • P (Ξ t ) < (ηL 2 -1)γµ t .
Let Q(u) = u • P (u) for u ≥ 0. Since P (u) is bounded in a neighborhood of zero, we have:
lim u→0 + Q(u) = 0 • ηL 2 + L 4 24 √ mL 1 = 0.
The term on the right-hand side, C ≜ (ηL 2 -1)γµ t , is a strictly positive constant. By the continuity of polynomial functions and the intermediate value theorem, there exists a threshold δ > 0 such that for all 0 < Ξ t < δ, the inequality Q(Ξ t ) < C holds. This implies that for a sufficiently small consensus error Ξ t , the negative drift from A (t) dominates the residual H (t) , ensuring U (t) < 0.
Explanation. The high-level intuition of Theorem D.7 and Proposition D.8 is outlined by the following descent lemma.
E ξ (t) L( θ(t+1) ) ≤ L( θ(t) ) -η - η 2 L 2 2 >0 ∇L( θ(t) ) 2 Standard Descent
this section cite: []

Section: 
+ η 2 L 2 -η >0 ∇L( θ(t) ) ⊤ ∇Tr ∇ 2 L( θ(t) )Γ (t) <0, progressive sharpening + η 2 L 2 σ 2 2m + O(Ξ 3 t ).
(D.17) Remark D.2. To simultaneously satisfy this requirement σ 2 ηL2 (2-ℓ)m ≤ ε/3 in Equation (D.13) and the condition η > 1/L 2 needed to maintain the descent property introduced by progressive sharpening, we can appropriately scale the number of agents m. Specifically, for any arbitrarily small target accuracy ε, σ 2 ηL2
(2-ℓ)m ≤ ε/3 is guaranteed to hold as long as the network size satisfies m > 3σ 2
(2-ℓ)ε . In other words, we can achieve the same convergence rate as parallel SGD by adapting the network size m to the desired target accuracy ε. Proposition D.9 (Critical Consensus Edge). Suppose Assumption 1-Assumption 3 hold. Assume η > 1 L2 , and the consensus error Ξ t ≤ 1 for all t. Then, the following condition ensures that the critical Inequality (8) is satisfied:
24(1 -p)η 2 p 2 ϕ 2 + σ 2 < min (ηL 2 -1)γ * µ t 2(ηL 2 + L4 24 ) √ mL 1 , (ηL 2 -1)γ * µ t 2Σ high , (D.18)
where
Σ high = 1 8 ηL 2 L 2 3 + 1 2 η √ mL 2 L 3 + ηmL2L 2 4 1152 .
Here, γ * denotes the degree of progressive sharpening (see Assumption 4), ϕ 2 denotes the uniform upper bound of the averaged squared local gradient norm (i.e.,
1 m m k=1 ∥∇L k (θ (t) k )∥ 2 ≤ ϕ 2
), and µ t is the lower bound on the global gradient norm (i.e., ∥∇L( θ(t) )∥ ≥ µ t > 0).
Proof. The proof establishes that the condition in Equation (11) suffices to guarantee U (t) < 0.
Recall from the decomposition in Equation (D.15) that U (t) = A (t) + H (t) , where A (t) is the leading descent term and H (t) represents higher-order residuals. We aim to show that the negative drift dominates the error, i.e., H (t) < -A (t) .
Using the lower bound on the gradient norm ∥∇L( θ(t) )∥ ≥ µ t derived in Proposition 2, the leading term satisfies A (t) ≤ -(ηL 2 -1)γ * µ t Ξ 2 t . Let K ≜ (ηL 2 -1)γ * µ t . Under the sharpening regime (ηL 2 > 1), we have K > 0. Thus, a sufficient condition for U (t) < 0 is:
H (t) < KΞ 2 t .(
D.19) Step (A): Bounding the residual term. Substituting the expansion of H (t) from Equation (D.15) into Equation (D.19) and dividing both sides by Ξ 2 t (assuming Ξ t > 0), the requirement becomes:
ηL 2 + L4 24 √ mL 1 + 1 8 ηL 2 L 2 3 Ξ t + 1 2 η √ mL 2 L 3 Ξ 2 t + ηmL2L 2 4 1152 Ξ 3 t ≜P (Ξt) Ξ t < K. We define C lin ≜ (ηL 2 + L424
)
√ mL 1 as the coefficient of the linear term. Utilizing the assumption that the consensus error is locally bounded (Ξ t ≤ 1), we have Ξ k t ≤ Ξ t for k ≥ 1. This allows us to upper bound the higher-order polynomial terms using the aggregated coefficient Σ high defined in the proposition: P (Ξ t )Ξ t ≤ C lin Ξ t + Σ high Ξ 2 t . Therefore, it suffices to ensure C lin Ξ t + Σ high Ξ 2 t < K.
this section cite: []

Section: Step (B).
To satisfy the inequality above, we employ a budget splitting strategy, requiring both the linear and quadratic components to be bounded by half of the descent budget K/2. This yields two separate constraints on Ξ t :
C lin Ξ t < K 2 =⇒ Ξ t < K 2C lin
, and
Σ high Ξ 2 t < K 2 =⇒ Ξ t < K 2Σ high . Consequently, if Ξ t < min K 2C lin , K 2Σ high , then U (t) < 0 holds.
Finally, invoking Corollary D.2, which bounds the consensus error as Ξ t ≤ 24(1-p)η 2 p 2 (ϕ 2 + σ 2 ), we see that Equation (11) ensures Ξ t falls within the safety region. This completes the proof.
this section cite: []

Section: References
Ref_id:b0 Title: Git re-basin: Merging models modulo permutation symmetries Year: (2023)
Ref_id:b1 Title: Sparse-push: Communication-& energyefficient decentralized distributed learning over directed & time-varying graphs with non-iid datasets Year: (2021)
Ref_id:b2 Title: The privacy power of correlated noise in decentralized learning Year: (2024)
Ref_id:b3 Title: Swarm Intelligence: From Natural to Artificial Systems Year: (1999)
Ref_id:b4 Title: SWIFT: Rapid decentralized federated learning via wait-free model communication Year: (2023)
Ref_id:b5 Title: Petals: Collaborative inference and fine-tuning of large models Year: (2023)
Ref_id:b6 Title: Distributed inference and fine-tuning of large language models over the internet Year: (2023)
Ref_id:b7 Title: Randomized gossip algorithms Year: (2006)
Ref_id:b8 Title: On the trade-off between flatness and optimization in distributed learning Year: (2024)
Ref_id:b9 Title: Cambridge bitcoin electricity consumption index (CBECI) Year: (2023)
Ref_id:b10 Title: Understanding pre-training and fine-tuning from loss landscape perspectives Year: (2025)
Ref_id:b11 Title: An efficient stochastic algorithm for decentralized nonconvexstrongly-concave minimax optimization Year: (2024)
Ref_id:b12 Title: Decentralized stochastic bilevel optimization with improved per-iteration complexity Year: (2023)
Ref_id:b13 Title: Accelerating gossip sgd with periodic global averaging Year: (2021)
Ref_id:b14 Title: Understanding optimization in deep learning with central flows Year: (2025)
Ref_id:b15 Title: $c^2m^3$: Cycle-consistent multi-model merging Year: (2024)
Ref_id:b16 Title: Differentially private decentralized learning with random walks Year: (2024)
Ref_id:b17 Title: Self-stabilization: The implicit bias of gradient descent at the edge of stability Year: (2023)
Ref_id:b18 Title: Optimal distributed online prediction using mini-batches Year: (2012)
Ref_id:b19 Title: Distributed lowcommunication training of language models Year: (2023)
Ref_id:b20 Title: Essentially no barriers in neural network energy landscape Year: (2018)
Ref_id:b21 Title: The role of permutation invariance in linear mode connectivity of neural networks Year: (2022)
Ref_id:b22 Title: Asynchronous SGD on graphs: a unified framework for asynchronous decentralized and federated optimization Year: (2024)
Ref_id:b23 Title: Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the neural tangent kernel Year: (2020)
Ref_id:b24 Title: Linear mode connectivity and the lottery ticket hypothesis Year: (2020)
Ref_id:b25 Title: Topology and geometry of half-rectified network optimization Year: (2017)
Ref_id:b26 Title: Fast training method for stochastic compositional optimization problems Year: (2021)
Ref_id:b27 Title: On the convergence of distributed stochastic bilevel optimization algorithms over a network Year: (2023)
Ref_id:b28 Title: Microsoft and OpenAI plot $100 billion stargate AI supercomputer. The Information Year: (2024)
Ref_id:b29 Title: Loss surfaces, mode connectivity, and fast ensembling of dnns Year: (2018)
Ref_id:b30 Title: Ai infrastructure market size, share & growth report Year: (2024)
Ref_id:b31 Title: Why (and when) does local SGD generalize better than SGD? Year: (2023)
Ref_id:b32 Title: Why (and when) does local SGD generalize better than SGD? Year: (2023)
Ref_id:b33 Title: A quadratic synchronization rule for distributed deep learning Year: (2024)
Ref_id:b34 Title: Heavy-tail phenomenon in decentralized sgd Year: (2022)
Ref_id:b35 Title: Imagining a democratic, affordable future of foundation models: A decentralised avenue Year: (2025)
Ref_id:b36 Title: Identity mappings in deep residual networks Year: (2016)
Ref_id:b37 Title: Byzantine-robust decentralized learning via clippedgossip Year: (2022)
Ref_id:b38 Title: Measuring the effects of non-identical data distribution for federated visual classification Year: (2019)
Ref_id:b39 Title: Patching open-vocabulary models by interpolating weights Year: (2022)
Ref_id:b40 Title: Editing models with task arithmetic Year: (2023)
Ref_id:b41 Title: Averaging weights leads to wider optima and better generalization Year: (2018)
Ref_id:b42 Title: Intellect-1 technical report Year: (2024)
Ref_id:b43 Title: Gossip-based aggregation in large dynamic networks Year: (2005-08)
Ref_id:b44 Title: Decentralized personalized federated learning Year: (2024)
Ref_id:b45 Title: Noloco: No-all-reduce low communication training method for large models Year: (2025)
Ref_id:b46 Title: A unified theory of decentralized SGD with changing topology and local updates Year: (2020)
Ref_id:b47 Title: Consensus control for decentralized deep learning Year: (2021)
Ref_id:b48 Title: Learning multiple layers of features from tiny images (tech. rep Year: (2009)
Ref_id:b49 Title: Tiny imagenet visual recognition challenge Year: (2015)
Ref_id:b50 Title: Refined convergence and topology learning for decentralized sgd with heterogeneous data Year: (2023)
Ref_id:b51 Title: Improved stability and generalization guarantees of the decentralized SGD algorithm Year: (2024)
Ref_id:b52 Title: Branch-train-merge: Embarrassingly parallel training of expert language models Year: (2022)
Ref_id:b53 Title: Communication efficient distributed machine learning with the parameter server Year: (2014)
Ref_id:b54 Title: Learning to collaborate in decentralized learning of personalized models Year: (2022)
Ref_id:b55 Title: What happens after SGD reaches zero loss? -a mathematical framework Year: (2022)
Ref_id:b56 Title: Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent Year: (2017)
Ref_id:b57 Title: Asynchronous decentralized parallel stochastic gradient descent Year: (2018)
Ref_id:b58 Title: Quasi-global momentum: Accelerating decentralized deep learning on heterogeneous data Year: (2021)
Ref_id:b59 Title: Optimal complexity in decentralized training Year: (2021)
Ref_id:b60 Title: Implicit Bias of Deep Learning Optimization: A Mathematical Examination Year: (2024)
Ref_id:b61 Title: Manuel Gil Pérez, Gregorio Martínez Pérez, and Alberto Huertas Celdrán. Decentralized federated learning: Fundamentals, state of the art, frameworks, trends, and challenges Year: (2023)
Ref_id:b62 Title: Merging models with fisher-weighted averaging Year: (2022)
Ref_id:b63 Title: A survey of swarm intelligence for dynamic optimization: Algorithms and applications Year: (2017)
Ref_id:b64 Title: Communication-Efficient Learning of Deep Networks from Decentralized Data Year: (2017)
Ref_id:b65 Title: Privacy attacks in decentralized learning Year: (2024)
Ref_id:b66 Title: Asynchronous decentralized sgd with quantized and local updates Year: (2021)
Ref_id:b67 Title: Uniform convergence may be unable to explain generalization in deep learning Year: (2019)
Ref_id:b68 Title: Distributed optimization over time-varying directed graphs Year: (2014)
Ref_id:b69 Title: Distributed subgradient methods for multi-agent optimization Year: (2009)
Ref_id:b70 Title: Announcing the stargate project Year: (2025)
Ref_id:b71 Title: Task arithmetic in the tangent space: Improved editing of pre-trained models Year: (2023)
Ref_id:b72 Title: Trends in ai supercomputers Year: (2025)
Ref_id:b73 Title: Gradient methods for minimizing functionals Year: (1963)
Ref_id:b74 Title: Introduction to Optimization Year: (1987)
Ref_id:b75 Title: Learning transferable visual models from natural language supervision Year: (2021-07)
Ref_id:b76 Title: Protocol models: Scaling decentralized training with communication-efficient model parallelism Year: (2025)
Ref_id:b77 Title: Diverse weight averaging for out-of-distribution generalization Year: (2022)
Ref_id:b78 Title: Model ratatouille: Recycling diverse models for out-of-distribution generalization Year: (2023-07)
Ref_id:b79 Title: Graph-dependent implicit regularisation for distributed stochastic subgradient descent Year: (2020)
Ref_id:b80 Title: SWARM parallelism: Training large models can be surprisingly communication-efficient Year: (2023)
Ref_id:b81 Title: Adaptation, Learning, and Optimization over Networks Year: (2014)
Ref_id:b82 Title: On efficient training of large-scale deep learning models Year: ()
Ref_id:b83 Title: Will llms scaling hit the wall? breaking barriers via distributed resources on massive edge devices Year: (2025)
Ref_id:b84 Title:  Year: (2024)
Ref_id:b85 Title: Do deep neural network solutions form a star domain? Year: (2025)
Ref_id:b86 Title: Stability and generalization of decentralized stochastic gradient descent Year: (2021)
Ref_id:b87 Title: Beyond exponential graph: Communication-efficient topologies for decentralized learning via finite-time convergence Year: (2023)
Ref_id:b88 Title: Decentralized training over decentralized data Year: (2018)
Ref_id:b89 Title: Communication-efficient distributed deep learning: A comprehensive survey Year: (2020)
Ref_id:b90 Title: Distributed asynchronous deterministic and stochastic gradient optimization algorithms Year: (1986)
Ref_id:b91 Title: On the implicit bias in deep-learning algorithms Year: (2023)
Ref_id:b92 Title: Relaysum for decentralized deep learning on heterogeneous data Year: (2021)
Ref_id:b93 Title: Beyond spectral gap: The role of the topology in decentralized learning Year: (2023)
Ref_id:b94 Title: Epidemic learning: Boosting decentralized learning with randomized communication Year: (2023)
Ref_id:b95 Title: CocktailSGD: Fine-tuning foundation models over 500Mbps networks Year: (2023)
Ref_id:b96 Title: Adaptive federated learning in resource constrained edge computing systems Year: (2019)
Ref_id:b97 Title: Analyzing sharpness along GD trajectory: Progressive sharpening and edge of stability Year: (2022)
Ref_id:b98 Title: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time Year: (2022)
Ref_id:b99 Title: Robust fine-tuning of zero-shot models Year: (2022-06)
Ref_id:b100 Title: Implicit regularization of decentralized gradient descent for sparse regression Year: (2024)
Ref_id:b101 Title: A faster decentralized algorithm for nonconvex minimax problems Year: (2021)
Ref_id:b102 Title: A(dp) 2 sgd: Asynchronous decentralized parallel stochastic gradient descent with differential privacy Year: (2021)
Ref_id:b103 Title: TIES-merging: Resolving interference when merging models Year: (2023)
Ref_id:b104 Title: Decentralized gossip-based stochastic bilevel optimization over communication networks Year: (2022)
Ref_id:b105 Title: Generalization error matters in decentralized learning under Byzantine attacks Year: (2025)
Ref_id:b106 Title: Exponential graph is provably efficient for decentralized deep training Year: (2021)
Ref_id:b107 Title: Decentralized training of foundation models in heterogeneous environments Year: (2022)
Ref_id:b108 Title: On the convergence of decentralized gradient descent Year: (2016)
Ref_id:b109 Title: Decentralized federated learning: A survey and perspective Year: (2024)
Ref_id:b110 Title: Bayesian nonparametric federated learning of neural networks Year: (2019)
Ref_id:b111 Title: Decentralized sporadic federated learning: A unified algorithmic framework with convergence guarantees Year: (2025)
Ref_id:b112 Title: Loss landscape dependent self-adjusting learning rates in decentralized stochastic gradient descent Year: (2021)
Ref_id:b113 Title: Going beyond linear mode connectivity: The layerwise linear feature connectivity Year: (2023)
Ref_id:b114 Title: Sharpness-aware minimization efficiently selects flatter minima late in training Year: (2025)
Ref_id:b115 Title: Stability and generalization of the decentralized stochastic gradient descent ascent algorithm Year: (2023)
Ref_id:b116 Title: Topologyaware generalization of decentralized SGD Year: (2022)
Ref_id:b117 Title: Decentralized SGD and average-direction SAM are asymptotically equivalent Year: (2023)
Ref_id:b118 Title: DICE: Data influence cascade in decentralized learning Year: (2025)
