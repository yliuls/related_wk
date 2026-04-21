Title: Graph Diffusion for Robust Multi-Agent Coordination
Abstract: Offline multi-agent reinforcement learning (MARL) struggles to estimate out-of-distribution states or actions due to the absence of real-time interactions with the environment. Although diffusion models have shown promising potential in addressing these challenges, they primarily apply independent diffusion to the historical trajectories of individual agents, which overlooks the crucial dynamics in multi-agent coordination and limits the policy robustness in dynamic environments. In this paper, we propose MCGD, a novel Multi-agent Coordination framework based on Graph Diffusion models to improve the effectiveness and robustness of collaborative policies. Specifically, we construct a sparse coordination graph with continuous node attributes and discrete edge attributes to identify the underlying multi-agent dynamics effectively. We then derive the transition probabilities between edge categories and present adaptive categorical diffusion to model the structure diversity of inter-agent coordination. According to the coordination structure, we define the neighbor-dependent forward noise and design anisotropic diffusion to increase the action diversity of each agent. Extensive experiments across various multi-agent environments demonstrate that MCGD significantly outperforms existing state-of-the-art baselines in coordination performance and exhibits superior robustness to dynamic environmental changes.

Section: Introduction
Offline Multi-Agent Reinforcement Learning (MARL) enables policy learning from pre-collected datasets, circumventing the need for real-time interactions with the environment (Lange et al., 2012;Levine et al., 2020). This approach is essential in scenarios where real-time interactions are expensive, unsafe, or infeasible, such as robotics in hazardous environments or autonomous systems (Barde et al., 2024;Wang et al., 2024). Offline MARL offers a path for deploying intelligent agents without requiring continuous interaction, but it introduces significant challenges.
The primary challenges in offline MARL arise from the absence of real-time feedback, which limits the ability to adapt policies dynamically based on ongoing interactions with the environment (Matsunaga et al., 2023). First, generalization to unseen states and actions is difficult because policies are trained on limited data, which may not fully capture the diversity of real-world scenarios. In multi-agent settings, this issue is compounded by the complexity of agent interactions. Second, out-of-distribution (OOD) actions and states present another challenge, as offline methods often struggle to handle situations not represented in the training data. This can lead to unreliable or unsafe behaviors due to the model's inability to properly extrapolate from the limited data available (Kumar et al., 2019;Fujimoto et al., 2019).
Recent advancements have integrated diffusion models (Song & Ermon, 2019;Ho et al., 2020) into both singleagent and multi-agent offline reinforcement learning (RL) to improve policy stability and performance. In single-agent offline RL, diffusion models address issues like overestimation bias by modeling the full distribution over actions, leading to more robust value function estimates (Janner et al., 2022;Ajay et al., 2022). This probabilistic approach helps mitigate common challenges in high-dimensional state spaces by capturing complex dependencies between states and actions, offering a more stable alternative to traditional Q-learning. For multi-agent offline RL, diffusion models like MADIFF (Zhu et al., 2023) extend this framework to model complex agent interactions, leveraging diffusion processes to simulate cooperative dynamics within multiple interacting agents. Additionally, methods such as EAQ (Oh et al.) enhance the training process by incorporating the Q-total function into the diffusion model, improving the estimation of joint action values in multi-agent settings. DOM2 (Li et al., 2023), on the other hand, overcomes the conservatism of offline RL by enabling greater exploration of strategies, thus reducing the tendency to rely on suboptimal, overly cautious policies.
Nevertheless, these methods typically apply diffusion models independently to the historical trajectories of individual agents, overlooking the crucial coordination dynamics between agents. This approach limits policy robustness in dynamic multi-agent settings, as it adapts to changes in individual agent attributes but fails to capture the evolving coordination strategies that emerge as agents interact. Figure 1 illustrates this limitation using a four-agent collaborative hunting scenario, where four vehicles work together to capture an adversary. This scenario mimics real-world challenges in multi-agent systems, where agents must continuously coordinate and adapt to both internal and external changes. The hunting strategy, trained on fixed speed attributes from offline data, fails to generalize to real-time shifts, such as changes in agent speed or the sudden unavailability of an agent. When two agents' speeds change, the diffusion-based method can adjust the strategy through noise addition and denoising, as shown in Figure 1(a), allowing the task to proceed. However, when one agent unexpectedly becomes unavailable, the remaining agents cannot adapt their coordination structure, and the hunting task fails. This example highlights a critical challenge in offline MARL: the need for policies that not only adapt to individual agent attributes but also dynamically adjust to shifts in the coordination structures. In real-world applications, such changes in coordination are inevitable, and robust policies must be able to handle these shifts to ensure successful and efficient multi-agent collaboration.
In this work, we propose a novel generative framework MCGD for Multi-agent Coordination based on Graph Diffusion models to enable diverse and adaptive collaborative policies, as illustrated in Figure 1 (b), with enhanced effectiveness and robustness in dynamic environments. Ini-tially, to identify the collaborative dynamics, we construct a sparse graph with continuous nodes and discrete edges, retaining essential and eliminating ineffective multi-agent interactions. Next, we measure the observational differences to derive an edge transition matrix adapted to multi-agent behaviors, and present categorical diffusion to model the structure diversity in inter-agent coordination. For each agent, we define neighbor-dependent forward noise to capture the dynamic coordination structure and develop anisotropic diffusion to model the diversity in single-agent actions. Finally, extensive comparative evaluations on three well-established benchmarks-MPE, MAMuJoCo, and SMAC-demonstrate that our framework achieves superior coordination performance and robustness, significantly outperforming state-ofthe-art baselines by up to 12.8% and 14.2%, respectively.
Our contributions are summarized as follows:
• We propose the first graph diffusion model for multi-agent coordination with superior effectiveness and robustness in dynamic multi-agent environments.
• We present a categorical diffusion process to simulate transitions between edge categories, modeling the structure diversity in multi-agent coordination.
• We develop an anisotropic diffusion process incorporating neighbor-dependent forward noise to model the diversity in single-agent actions.
• Comparative evaluations across various challenging multiagent environments demonstrate the significant advantages of our framework in coordination performance and policy robustness compared to state-of-the-art baselines.
this section cite: ['b21', 'b22', 'b3', 'b41', 'b26', 'b20', 'b38', 'b12', 'b1', 'b47', 'b23']

Section: Related Work

this section cite: []

Section: Offline Multi-agent Reinforcement Learning
Offline coordination is a significant hurdle that limits advancements and exploration within offline multi-agent reinforcement learning (MARL) (Zhu et al., 2023). To address this, recent studies (Chen et al., 2021;Yang et al., 2021) have extended single-agent offline algorithms to multi-agent scenarios through policy regularization. However, these extensions face persistent extrapolation errors (Fujimoto et al., 2019) in offline environments, which remain challenging to resolve fully. An alternative approach, the centralized training with decentralized execution (CTDE) paradigm (Oliehoek et al., 2008), has driven notable progress through algorithms such as QMIX (Rashid et al., 2020), MADDPG (Lowe et al., 2017), and MAPPO (Yu et al., 2022). Additionally, methods like MA-ICQ (Yang et al., 2021) and OMAR (Pan et al., 2022) have been developed to address distributional shifts in offline settings by adopting conservatism principles. Despite their success in reducing distributional errors, these conservatism-based methods restrict the flex-ibility of agent coordination, limiting adaptability across varied scenarios. To enhance multi-agent coordination, diffusion models capable of capturing complex distributions have recently been introduced into offline MARL (Zhu et al., 2023;Li et al., 2023;Oh et al.). However, these algorithms independently apply diffusion models to the historical trajectories of individual agents, neglecting the inter-agent coordination structure, which reduces their robustness to environmental changes.
In response, this work introduces a graph diffusion-based multi-agent coordination framework to explicitly capture the coordination structure and utilize distinct diffusion processes for structure and action diversity, achieving superior adaptability and performance in dynamic environments.
this section cite: ['b47', 'b5', 'b43', 'b28', 'b33', 'b24', 'b45', 'b43', 'b31', 'b47', 'b23', 'b27']

Section: Diffusion Models
Diffusion models (Sohl-Dickstein et al., 2015;Ho et al., 2020) have emerged as a powerful generative framework for modeling complex data distributions. They have shown significant success in continuous generation tasks such as image synthesis (Dhariwal & Nichol, 2021;Rombach et al., 2022), animation creation (Ho et al., 2022;Luo & Hu, 2021), and molecule design (Corso et al.; Trippe et al.). Recently, two distinct strategies have been proposed to adapt diffusion models for generating graphs with discrete structures (Austin et al., 2021;Hoogeboom et al., 2021). The first approach (Chen et al.;Hoogeboom et al., 2022) embeds graph data into a continuous space by adding Gaussian noise to the node features and adjacency matrix, enabling the model to learn the underlying graph distribution. An alternative strategy (Vignac et al.;Hua et al., 2024) avoids continuous perturbations, which are less effective at capturing the structural properties of graph data. Instead, it directly models categorical diffusion for discrete graph data by estimating the transition probabilities between different categories. However, existing graph diffusion models primarily focus on the independent diffusion of discrete attributes across nodes and edges. This limitation makes them unsuitable for adaptively modeling the diversity of discrete multi-agent structures and continuous single-agent actions in offline multi-agent reinforcement learning.
To address these challenges, we propose the first graph diffusion model for modeling multi-agent collaboration, which combines categorical diffusion for inter-agent structures with continuous diffusion for multi-agent actions, facilitating more effective and robust collaboration.
this section cite: ['b37', 'b12', 'b8', 'b34', 'b13', 'b25', 'b2', 'b14', 'b15', 'b40', 'b16']

Section: Preliminaries
In this section, we formally define the fundamental concepts and provide a summary of the primary notations, as detailed in Appendix 7.1.1.
this section cite: []

Section: Offline Multi-agent Reinforcement Learning
We model the fully cooperative multi-agent task as a decentralized partially observable Markov decision process (Dec-POMDP) (Oliehoek et al., 2016), described as a tuple M = ⟨I, S, A, P, Ω, O, R, γ⟩ with agents I = {n 1 , n 2 , . . . , n |I| }. In this context, S and A denote state and action spaces, respectively, and γ ∈ [0, 1] is the discount factor. At each timestep t, every agent n i observes a local observation o t i ∈ Ω produced by the function O(o t i |S t , a t i ) and chooses an action a t i ∈ A. All chosen actions form a joint action A t ∈ A |I| and lead to a transition to the next global state S t+1 according to the function P(S t+1 |S t , A t ), resulting in a joint reward r t = R(S t , A t ). In offline settings, rather than interacting with the environment in realtime, we have access to a historical dataset D to learn multiagent policies that maximize the discounted cumulative reward. The offline dataset D typically consists of multiagent observation-action trajectories, where each trajectory τ = [O 0 , A 0 , O 1 , A 1 , . . . , O T , A T ] includes the joint observations and actions at each time step.
this section cite: ['b29']

Section: Denoising Diffusion Probabilistic Model
The denoising diffusion probabilistic model (DDPM) (Ho et al., 2020) is a generative model that synthesizes continuous or discrete data through a forward noising process and a corresponding reverse denoising process.
In continuous diffusion, DDPM introduces Gaussian noise to the original data X 0 ∼ q(X) at each iteration k, perturbing it progressively as follows:
q(X k |X k-1 ) = N (X k ; 1 -β k X k-1 , β k I), (1
)
where I denotes the identify covariance matrix, and β k ∈ (0, 1) controls the scale of the Gaussian noise added at iteration k. In the reverse process, DDPM employs a trained, parameterized Gaussian transition kernel p θ to iteratively denoise samples, gradually reconstructing the original data distribution from the noisy samples as follows:
p θ (X k-1 |X k ) = N (X t-1 ; µ θ (X k , k), Σ θ (X k , k)), (2)
where µ θ and Σ θ represent the predicted average value and covariance matrix parameterized by θ.
For discrete diffusion on data X (e.g., graph structures), DDPM computes the transition probabilities between categories to replace the Gaussian noise, yielding the noisy data at iteration k as follows:
q(X k |X k-1 ) = cat(X k |X k-1 Q β k ),(3)
where Q β k represents the transition matrix applied to the discrete data X k-1 at iteration k, and cat denotes the categorical distribution over possible categories of X k . During the denoising process, the posterior distribution p θ is computed using the Bayes rule as follows:
p θ (X k-1 |X k , X) ∝ cat(X k [Q β k ] T ⊙ XQ β k-1 ), (4
)
where
Q β k-1 = Q β1 Q β2 . . . Q β k-1 , [Q β k ]
T represents the transpose of the transition matrix Q β k , and ⊙ denotes the Hadamard product.
this section cite: ['b12']

Section: Methodology
In this work, we propose the first graph-based diffusion framework for offline multi-agent reinforcement learning, including categorical diffusion on discrete edges and anisotropic diffusion on continuous nodes, to model the diversity of inter-agent coordination and single-agent actions, respectively. The proposed framework consists of three primary processes: forward noising, reverse denoising, and policy sampling, as detailed in Figure 2. Specifically, we perturb forward the nearest-neighbor coordination graph constructed from multi-agent historical trajectories, design the graph transformer network to reversely recover the clean attributes of discrete edges and continuous nodes, and employ the trained diffusion model to sample collaborative policies for multi-agent decentralized execution.
For clarity, we denote the timestep in an episode as {t} T t=1 , the diffusion iteration as {k} K k=1 , and the agent as {n i } |I| i=1 .
this section cite: []

Section: Forward Noising Process
To capture the collaborative dynamics among agents, we identify the essential multi-agent interactions to construct the sparse coordination graph G t = (A t , E t ) at timestep t. For each agent n i ∈ I, we establish undirected edges connecting it with its k nearest neighbors defined by their observation difference, resulting in k-nn coordination graph.
In the graph G t , A t ∈ R |I|×d represents the d-dimensional continuous node attributes that encode the actions of each agent, while E t ∈ {0, 1} |I|×|I| is a binary adjacency matrix where each entry indicates the presence (or absence) of an edge between two agents. For discrete action spaces, we represent each agent's action using a one-hot vector of dimension d, where d is the number of discrete actions.
In this subsection, we present a graph diffusion process over the graph G t to simulate the dynamic multi-agent behavior when online coordination, detailed as follows:
q(G t k |G t k-1 ) = ( 1 -β k A t k-1 + β k ϵ, E t k-1 Q β k ), (5
) where ϵ and Q β k are the forward noise and transition matrix applied A t k-1 and E t k-1 , respectively, at iteration k. To guarantee the efficiency and adaptivity of the forward noising process, three desirable properties are required:
• The transition matrix Q β k should be adaptive to the multiagent historical trajectories. • The marginal distribution q(G t K |G t ) should have a closedform expression for efficient calculation.
• The Gaussian noise ϵ should be anisotropic and dependent on the inter-agent coordination E t .
this section cite: []

Section: CATEGORICAL NOISING ON EDGES
Considering the relevance among multiple agents, we define an adaptive transition matrix between edge categories and present a categorical diffusion process on the discrete edges to model the diversity in inter-agent coordination.
To quantify the multi-agent relevance, for each pair of agents n i and n j , we calculate the cosine similarity c i,j between their observations o t i , o t j ∈ O t as a measure of the likelihood of diffusion occurring between n i and n j . Intuitively, a high similarity in the agents' observations reflects a strong alignment of their states and indicates the potential for substitutable coordination, thereby increasing the likelihood of diffusion between them. Leveraging this metric, we construct a similarity matrix C ∈ R |I|×|I| and derive the transition matrix Q t between edge categories as follows:
Q t = e (C-D)t ,(6)
where D is the diagonal degree matrix of C, normalizing the similarity values. Theorem 4.1 establishes the rationality of the transition matrix Q t for categorical diffusion, confirming the necessary properties identified in prior studies (Yi et al., 2024;Shi et al.).
Theorem 4.1. The transition matrix Q t satisfies the following beneficial properties of symmetry, additivity, locality, and convergence:
[Q t ] T = Q t , Q ti+tj = Q ti Q tj , lim t→0 Q t = I, lim t→∞ Q t = 1 |I| 11 T .(7)
The detailed proof is provided in Appendix 7.2.1.
By incorporating the scale parameter β k , the categorical noising of edge discrete E t at diffusion iteration k is formal- ized as follows:
𝑜 1 0 𝑎 1 0 𝑜 1 1 𝑎 1 1 … … 𝑜 1 𝑇 𝑎 1 𝑇 ⋮ ⋱ ⋮ 𝑜
q(E t k |E t k-1 ) = cat(E t k |E t k-1 Q β k ), Q β k = e (C-D)β k , (8
) where cat(•) represents the categorical distribution, parameterized by the product E t k-1 Q β k . Utilizing the symmetry and additivity properties in Theorem 4.1, we derive the closed-form expression for categorical noising on E t :
q(E t K |E t ) = cat(E t K |E t Q β K ), Q β K = Q β1 Q β2 . . . Q β K = e K k=1 [β k (C-D)] . (9) 4.1.2. ANISOTROPIC NOISING ON NODES
Instead of the noising process on historical trajectories independent of inter-agent coordination, we define a forward Gaussian noise conditioned on neighboring agents' actions and then design an anisotropic diffusion on continuous nodes to model the diverse single-agent actions.
For each agent n i ∈ I, we encode the action information of its neighbors in the original coordination graph G t as A t i ⊂ A t , and calculate the covariance matrix Σ i ∈ R d×d of A t i to characterize the anisotropy of its diffusion process. The forward noising over the single-agent action a t i at iteration k is formalized as follows:
q(a t i,k |a t i,k-1 ) = N (a t i,k ; 1 -β k a t i,k-1 , β k Σ i ). (10
)
Given the parameters α k = 1 -β k and α K = K k=1 α k , the closed-form expression of the marginal distribution of a t i,K is derived as follows:
q(a t i,K |a t i ) = N (a t i,K ; √ α K a t i , (1 -α K )Σ i ).(11)
The detailed proof is provided in Appendix 7.2.2.
By integrating the categorical diffusion over discrete edges and anisotropic diffusion over continuous actions, we derive the closed-form expression for the marginal distribution q(G t K |G t ) as follows:
q(G t K |G t ) = ( √ α K A t + (1 -α K )ϵ, E t Q K ), (12
)
where ϵ deontes the multi-agent d-dimensional anisotropic Gaussian noise, with each component ϵ i for agent n i drawn from N (0, Σ i ).
this section cite: ['b44', 'b36']

Section: Reverse Denoising Process
In this subsection, we design a graph diffusion network to reverse denoise the perturbed coordination graph, with the goal of recovering the original attributes of inter-agent coordination and single-agent actions.
Given the edge attributes E t k and E t , we apply the Bayes rule to derive the posterior distribution of E t k-1 as follows:
q(E t k-1 |E t k , E t ) ∝ cat(E t k-1 |E t k Q β k ⊙ E t Q β k-1 ), (13
) where ⊙ denotes the pairwise Hadamard product operation.
To predict the unperturbed edge attributes E t , we employ a graph transformer network (Yun et al., 2019) to construct a graph denoising network f θ . This network takes as input the noisy graph G t K = (A t K , E t K ) and outputs the estimated edge attribute Êt θ and node attribute Ât θ
To optimize the graph diffusion model parameterized by θ, we employ the cross-entropy loss L ce to quantify the discrepancy between the clean edge attribute E t and the predicted edge attribute Êt θ as follows:
L ce = E (Ot,At)∈τ i,j cross-entropy(E t i,j , Êt i,j ), (14
)
where E t i,j and Êt i,j denote the (i, j) entry in the discrete attributes in E t and Êt θ , respectively. To further optimize the denoising network f θ , we compute the Euclidean distance between the clean node attribute A t and the predicted node attribute Ât θ , and incorporate the Q-loss (Wang et al.) to define the anisotropic diffusion loss L ad , as follows:
L ad = E (Ot,At)∈τ i=1 ||a t i -ât i || 2 -λQ ϕi (o t i , ât i ) ,(15)
where ât i ∈ Ât denotes the predicted action of agent n i ∈ I, o t i represents the average observation of agent n i and its neighbors, and λ is the regularization coefficient.
To reduce model complexity and maintain scalability, we process each neighboring observation using a sharedparameter MLP, followed by a mean pooling operation over the resulting features. This replaces concatenation, which can significantly increase the parameter count as the number of neighbors grows. By limiting the neighbors to those with higher similarity, the averaging operation is performed over semantically similar features, thereby mitigating information loss.
The training details of our MCGD framework are provided in Appendix 7.1.2.
this section cite: ['b46']

Section: Policy Sampling Process
In this subsection, we design a policy sampling process based on the trained graph diffusion model to generate multiagent collaborative behaviors, with the aim to progressively denoise the inter-agent coordination and multi-agent actions in a decentralized execution setting. The whole sampling process is summarized in Algorithm 1.
At each timestep t, each agent n i ∈ I obtains its local observation o t i from the environment and selects the optimal action a t i according to its trained Q-value function Q * ϕi (line 5 in Algorithm 1). Instead of the noise initialization used in prior methods (Li et al., 2023;Zhu et al., 2023), for each agent n i , we generate N random action samples from the continuous action space to construct a candidate set. We then evaluate each candidate using the trained Q-function Q * ϕi and select the action with the highest Q-value, providing a more informed and value-guided sampling strategy. Since the optimal action is selected from a finite set of sampled candidates, this approach is naturally applicable to both discrete and continuous action spaces and does not require differentiability or closed-form maximization over actions. For the actions of other agents that are inaccessible, we replace them with a standard Gaussian noise of the same dimensionality, thereby forming the multi-agent actions A t .
Using the actions A t as continuous node attributes, we initialize a fully connected coordination graph G t i = (A t , E t ) (line 6 in Algorithm 1), where the edge attribute E t is defined such that off-diagonal entries are set to 1. The attributes E t and A t are treated as the initial perturbed data E t K and A t K in the context of categorical and anisotropic Algorithm 1 MCGD Sampling Algorithm 1: Input: diffusion parameter K, graph diffusion model f θ * , and Q-value function Q * ϕi for each agent n i 2: Initialize: initial observation o 0 i for each agent n i 3: for each timestep t do 4: for each agent n i do 5: Select action a t i based on Q * ϕi 6: Construct the fully-connected graph G t i 7: for each diffusion iteration k do 8: Predict the clean attributes Êt θ and Ât θ based on the denoising network f θ 9: Sample the edge attributes E t k-1 from the posterior distribution p θ (E t k-1 |E t k , Êt θ ) 10: end for 11: Execute the single-agent action a t i ∈ Ât θ 12: end for 13: end for diffusion processes, respectively. At each diffusion iteration k, we employ the denoising network f θ to predict the denoised edge attribute Êt θ and node attribute Ât θ (line 8 in Algorithm 1). Using the edge attributes Êt θ and E t k , we derive the posterior distribution p θ (E t k-1 |E t k , E t 0 ), from which we sample the edge attribute E t k-1 at iteration k -1 (line 9 in Algorithm 1), as follows:
E t k-1 ∼ cat(E t k-1 |E t k Q β k ⊙ f θ (E t k , k)Q β k-1 ). (16
)
Based on the sampled edge attribute E t k-1 and the predicted node attribute Ât θ , we construct the perturbed coordination graph G t k-1 = ( Ât θ , E t k-1 ), which will be used in the next denoising iteration k -1. After the final denoising iteration, we extract the single-agent action a t i from the final predicted node attribute Ât θ as the sampled action of agent n i ∈ I at timestep t (line 11 in Algorithm 1).
Under decentralized execution, each agent n i extracts its own action by retrieving the i-th row of Ât θ , ensuring consistency with the fully decentralized setting. In discrete action spaces, the denoised continuous output is passed through a softmax layer, and the final discrete action is selected via an argmax operation over the corresponding softmax probabilities.
this section cite: ['b23', 'b47']

Section: Evaluation
In this work, we conduct comparative experiments across different multi-agent environments, aiming to validate our method's ability to model the complex and diverse behaviors of cooperative agents. Specifically, we evaluate whether MCGD outperforms state-of-the-art baselines in coordination performance and demonstrates superior robustness to Table 1: Comparison between MCGD and baselines on offline Expert or Good datasets across the MPE, MAMuJoCo, and SMAC benchmarks: "average value ± standard deviation". Bold: the best performance, underline: the second performance.
this section cite: []

Section: Method
Expert MPE Good MAMuJoCo Good SMAC Spread Tag World 2halfcheetah 2ant 4ant 3m 2s3z 5m6m 8m MA-ICQ 79.2 ± 4.3 92.6 ± 15.5 83.5 ± 20.7 1735.2 ± 748.3 1186.2 ± 653.9 1214.9 ± 738.4 18.8 ± 0.6 19.6 ± 0.3 16.3 ± 0.9 19.6 ± 0.3 MA-CQL 74.1 ± 5.8 68.3 ± 13.2 57.7 ± 20.5 2722.8 ± 1022.6 1394.8 ± 604.3 1039.1 ± 617.5 19.6 ± 0.3 19.0 ± 0.8 13.8 ± 3.1 11.3 ± 6.1 OMAR 82.9 ± 2.4 97.9 ± 16.4 84.8 ± 21.0 2963.8 ± 410.5 1075.3 ± 374.1 954.8 ± 319.7 18.4 ± 0.2 18.8 ± 0.5 15.7 ± 0.3 16.2 ± 0.5 MA-SfBC 87.5 ± 7.3 77.4 ± 13.9 97.3 ± 19.1 2386.6 ± 440.3 1764.1 ± 457.4 1721.8 ± 392.3 19.1 ± 0.3 19.2 ± 0.3 15.2 ± 0.1 18.1 ± 0.3 DOM2 88.7 ± 6.3 98.2 ± 14.4 99.5 ± 17.1 3676.8 ± 248.5 2187.4 ± 190.3 1836.2 ± 241.6 19.4 ± 0.2 19.0 ± 0.4 18.1 ± 0.7 18.2 ± 0.4 MADIFF 82.1 ± 5.9 103.0 ± 12.0 96.4 ± 13.7 3446.5 ± 213.3 2479.3 ± 105.8 2414.5 ± 128.3 19.6 ± 0.7 19.4 ± 0.1 18.0 ± 1.0 19.2 ± 0.1 MCGD 93.8 ± 2.7 109.6 ± 13.3 110.9 ± 11.5 3917.4 ± 193.7 2782.7 ± 203.9 2609.2 ± 165.2 22.1 ± 0.1 20.7 ± 0.1 18.9 ± 0.5 20.1 ± 0.2 Abs.(%) Avg.↑ 5.1(5.7) 6.6(6.4) 11.4(11.5) 240.6(6.5) 303.4(12.2) 194.7(8.1) 2.5(12.8) 1.1(5.6) 0.9(5.0) 0.9(4.7)
Table 2: Comparison between MCGD and baselines in shifted environments including MPE Spread, MPE Tag, MPE World, and MAMuJoCo 2halfcheetah, with dynamic changes in agent attributes and coordination structure: "average value ± standard deviation". Bold: the best performance in each graph, underline: the second performance.
dynamic environmental changes (e.g., agent attributes and coordination structure). For fair evaluation, we run each offline experiment five times with different random seeds and report the average episodic return obtained in online rollout as the performance measure.
this section cite: []

Section: Experimental Setup

this section cite: []

Section: BENCHMARKS
We conduct extensive evaluations on three following wellestablished multi-agent benchmarks:
• Multi-Agent Particle Environments (MPE) (Lowe et al., 2017). Multiple 2D particles work together to achieve shared objectives across various scenarios. In the Spread task, three particles are positioned randomly and must cover three landmarks without colliding. In the Tag task, three predators cooperate to capture a pre-trained, faster-moving prey. In the World task, three predators catch a pre-trained prey, which aims to collect food while evading capture. For the offline datasets, we use four datasets from (Pan et al., 2022), each corresponding to different levels of training quality, Expert, Medium-Replay, Medium, and Random.
• Multi-Agent MuJoCo (MAMuJoCo) (Peng et al., 2021). Independent agents control different robotic joints to maximize forward speed. Three configurations are chosen: 2agent halfcheetah (2halfcheetah), 2-agent ant (2ant), and 4-agent ant (4ant). The offline datasets are sourced from (Formanek et al., 2023), which includes varying quality levels-Good, Medium, and Poor-for each control task.
• StarCraft Multi-Agent Challenge (SMAC) (Samvelyan et al., 2019). A team of agents collaborates to fight against an enemy team controlled by hand-coded AI. Four different maps are explored: three Marines per team (3m), two Stalkers and three Zealots per team (2s3z), five Marines versus six Marines (5m vs 6m), and eight Marines per team (8m).
The off-the-grid offline datasets (Formanek et al., 2023) are used here, with varying quality levels-Good, Medium, and Poor-available for each map.
this section cite: ['b24', 'b31', 'b32', 'b9', 'b35', 'b9']

Section: BASELINE
We compare the MGCD framework with the various stateof-the-art baselines including: offline MARL algorithms such as MA-ICQ (Yang et al., 2021), MA-CQL (Jiang & Lu, 2023), and OMAR (Pan et al., 2022)), the extension of the single-agent diffusion-based policy, MA-SfBC (Chen et al., 2022)), and diffusion-based MARL methods such as DOM2 (Li et al., 2023) and MADIFF (Zhu et al., 2023).
this section cite: ['b43', 'b19', 'b31', 'b4', 'b23', 'b47']

Section: Numerical Results
To evaluate the effectiveness of collaborative policies, we report numerical results for both MCGD and baseline methods, using offline Expert and Good datasets across three environments.
The average values and standard deviations of the episodic returns for each model are summarized in Table 1. Notably, our MCGD framework consistently achieves state-of-the-art performance across all multi-agent cooperative scenarios, with smaller deviations in most settings, which demonstrates its advantages in terms of both effectiveness and stability. The fact that the DOM2 and MADIFF algorithms rank second and third in each task underscores the potential of diffusion models in addressing the out-of-distribution challenge in offline multi-agent reinforcement learning. When compared to diffusion-based D 6WDQGDUG(QYLURQPHQW E 6KLIWHG(QYLURQPHQW$ F 6KLIWHG(QYLURQPHQW&6 /DQGPDUN /DQGPDUN /DQGPDUN $JHQW $JHQW $JHQW baselines, MCGD achieves improvements of at least 5.7%, 6.5%, and 4.7% in the MPE, MAMujoco, and SMAC benchmarks, respectively. These significant advantages further highlight the importance of the underlying multi-agent coordination framework when applying diffusion models to historical trajectories. The experimental results on Medium and Poor datasets are provided in Appendix.
To further validate the policy robustness, we design shifted environments (details are provided in Appendix 7.3.2) by altering agent attributes and coordination structures. Specifically, we randomly modify the particle speed in the MPE environment and adjust the power, density, and friction of joints in MAMuJoCo to simulate dynamic changes in agent attributes. Additionally, we randomly select a particle or joint to fix its speed or power attribute as zero, simulating changes in coordination structures caused by the sudden disconnection of an agent. For each selected task, we train the collaborative policies using the offline datasets generated in standard environments and evaluate them in shifted environments. The corresponding experimental results are shown in Table 2. We observe that MCGD consistently achieves the highest episodic return across all shifted environments, underscoring the robustness of our framework. Furthermore, in more challenging scenarios with dynamic coordination structures, MCGD's performance advantage becomes even more pronounced. This superiority can be attributed to the modeling of diverse edge structures and node actions within the multi-agent collaboration graph, facilitated by our defined graph diffusion process.
this section cite: []

Section: Qualitative Analysis
To investigate the effectiveness and robustness of the MCGD framework, we focus on the MPE spread task involving 3 agents and 3 landmarks, conducted in both standard and shifted environments. The team reward is defined as the sum of the distances between each landmark and its closest agent. When an agent collides with another, it incurs a penalty, which is subtracted from the team reward. Two example episodic trajectories are visualized in Figure 4, illustrating the coordinative behavior of agents in the task.
In the standard environment, each agent quickly approaches one or two landmarks to minimize the distance to the closest landmark. When the minimum speed of Agent 0 decreases, anisotropic diffusion within the MCGD framework dynamically reallocates the target landmarks for all three agents, altering their motion trajectories. Additionally, the diffusion process's neighbor dependence helps prevent collisions caused by excessive action uncertainty when two agents are close to one another. This cooperation ensures the successful completion of the task.
When Agent 0 goes offline and can no longer move, categorical diffusion alters the coordination structure among the remaining agents. With Agent 0's influence removed, the remaining two agents shift their focus from minimizing the distance to a single landmark to minimizing the sum of distances to multiple landmarks.
To illustrate how the learned coordination graph evolves during task execution, we further provide a case study on the MPE Spread task in Figure 5, where the horizontal axis indicates timesteps and the vertical axis shows different experimental settings.
Initially, although a nearest-neighbor graph is used for initialization, large positional differences between agents cause the forward diffusion process to disrupt coordination edges. As a result, the model predicte no edges, and agents act independently. As agents move closer, the learned graph
*RRG 0HGLXP 3RRU (SLVRGLF5HWXUQ P *RRG 0HGLXP 3RRU V] *RRG 0HGLXP 3RRU (SLVRGLF5HWXUQ PP *RRG 0HGLXP 3RRU P 0&*' 0&*'&' 0&*'$' 0$',))
Figure 6: Ablation study on categorical diffusion and anisotropic diffusion within the MCGD framework, evaluated in SMAC environments.
gradually recover the underlying coordination structure, enabling effective collaboration such as landmark assignment and collision avoidance.
In a modified setting where the speed of Agent 0 is reduced, coordination edges emerge primarily between the other agents. Agent 0, due to its reduced speed, require more timesteps to engage in coordination, delaying the full coordination graph reconstruction. In a more extreme case where Agent 0 is inactive, it remains isolated, and coordination is exclusively formed between the other two agents. These visualizations demonstrate the adaptive nature of the learned graph, which dynamically reflects the agents' interaction context and task demands.
this section cite: []

Section: Ablation Studies
To verify the functionality of key modules, categorical diffusion and anisotropic diffusion, we have designed two model variants, MCGD-CD and MCGD-AD. Specifically, in MCGD-CD, we ignore the dynamics of the multi-agent coordination structure and fix it to the k-nearest neighboring graph, whereas in MCGD-AD, we perform independent diffusion on multi-agent trajectories and aggregate the observations from neighboring agents based on the coordination structure. We experimentally compare the MCGD framework and its variants with the best performing baseline, MADIFF, using offline datasets categorized as Good, Medium, and Poor in the SMAC environment. As shown in Figure 6, the full MCGD framework outperforms all datasets, with both variants also surpassing the baseline, highlighting the importance of the two key modules in the MCGD framework. Although MCGD-CD performs satisfactorily in simpler tasks such as 3m and 2s3z, it struggles in other complex scenarios, reflecting the limitations of fixed coordination structures in multi-agent collaboration and highlighting the need for modeling coordination structure diversity in MCGD.
We further validate the design of processing neighboring observations through an ablation study using the SMAC benchmark, comparing two variants: MCGD-AO (with observation averaging) and MCGD-FC (with feature concatenation). As shown in Table 3, MCGD-AO achieves both higher average rewards and lower computational cost across all tasks. The inferior performance of MCGD-FC is largely attributed to increased parameterization and greater optimization difficulty, further supporting the effectiveness and generality of our averaging-based approach.
this section cite: []

Section: Conclusion
This work proposes the first graph diffusion model for offline multi-agent coordination, which employs categorical diffusion on discrete edge attributes and anisotropic diffusion on continuous node attributes to capture structure diversity and action diversity, respectively. Extensive evaluations across three benchmark multi-agent environments demonstrate the effectiveness of the MCGD framework in coordination and its superior robustness to dynamic changes in environmental conditions.
In future work, we aim to introduce more complex and adaptable collaboration graph structures to support a broader range of agent interactions. Real-world validation remains a critical direction to further demonstrate the practical applicability of our framework. Our team is currently working on deploying the proposed method in real-world multi-robot hunting scenarios. Although quantitative results are not yet available for inclusion in this version, we are actively collecting data and refining the deployment process. We plan to report these findings as part of a more extensive evaluation in a future extension of this work.
this section cite: []

Section: References
Ref_id:b0 Title: Reducing overestimation bias in multi-agent domains using double centralized critics Year: (2019)
Ref_id:b1 Title: Is conditional generative modeling all you need for decision-making? Year: (2022)
Ref_id:b2 Title: Structured denoising diffusion models in discrete state-spaces Year: (2021)
Ref_id:b3 Title: A model-based solution to the offline multi-agent reinforcement learning coordination problem Year: (2024)
Ref_id:b4 Title: Offline reinforcement learning via high-fidelity generative behavior modeling Year: (2022)
Ref_id:b5 Title: Decision transformer: Reinforcement learning via sequence modeling Year: (2021)
Ref_id:b6 Title: Analog bits: Generating discrete data using diffusion models with selfconditioning Year: ()
Ref_id:b7 Title: Diffusion steps, twists, and turns for molecular docking Year: ()
Ref_id:b8 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b9 Title: Offthe-grid marl: a framework for dataset generation with baselines for cooperative offline multi-agent reinforcement learning Year: (2023)
Ref_id:b10 Title: Off-policy deep reinforcement learning without exploration Year: ()
Ref_id:b11 Title:  Year: (2019)
Ref_id:b12 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b13 Title: Video diffusion models Year: (2022)
Ref_id:b14 Title: Argmax flows and multinomial diffusion: Learning categorical distributions Year: (2021)
Ref_id:b15 Title: Equivariant diffusion for molecule generation in 3d Year: (2022)
Ref_id:b16 Title: Unified diffusion for complete molecule generation Year: (2024)
Ref_id:b17 Title: Planning with diffusion for flexible behavior synthesis Year: ()
Ref_id:b18 Title:  Year: (2022)
Ref_id:b19 Title: Offline decentralized multi-agent reinforcement learning Year: (2023)
Ref_id:b20 Title: Stabilizing off-policy q-learning via bootstrapping error reduction Year: (2019)
Ref_id:b21 Title: Batch reinforcement learning Year: (2012)
Ref_id:b22 Title: Offline reinforcement learning: Tutorial, review, and perspectives on open problems Year: (2020)
Ref_id:b23 Title: Beyond conservatism: Diffusion policies in offline multi-agent reinforcement learning Year: (2023)
Ref_id:b24 Title: Multi-agent actor-critic for mixed cooperative-competitive environments Year: (2017)
Ref_id:b25 Title: Diffusion probabilistic models for 3d point cloud generation Year: (2021)
Ref_id:b26 Title: Alberdice: addressing out-ofdistribution joint actions in offline multi-agent rl via alternating stationary distribution correction estimation Year: (2023)
Ref_id:b27 Title: Diffusionbased episodes augmentation for offline multi-agent reinforcement learning Year: ()
Ref_id:b28 Title: Optimal and approximate q-value functions for decentralized pomdps Year: (2008)
Ref_id:b29 Title: A concise introduction to decentralized POMDPs Year: (2016)
Ref_id:b30 Title: Assessing generalization in deep reinforcement learning Year: (2018)
Ref_id:b31 Title: Plan better amid conservatism: Offline multi-agent reinforcement learning with actor rectification Year: (2022)
Ref_id:b32 Title: Factored multi-agent centralised policy gradients. Advances in Neural Information Processing Systems Year: (2021)
Ref_id:b33 Title: Monotonic value function factorisation for deep multi-agent reinforcement learning Year: (2020)
Ref_id:b34 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b35 Title: The starcraft multi-agent challenge Year: (2019)
Ref_id:b36 Title: Graph-constrained diffusion for end-to-end path planning Year: ()
Ref_id:b37 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b38 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b39 Title: Diffusion probabilistic modeling of protein backbones in 3d for the motif-scaffolding problem Year: ()
Ref_id:b40 Title: Digress: Discrete denoising diffusion for graph generation Year: ()
Ref_id:b41 Title: Offline multiagent reinforcement learning with implicit global-to-local value regularization Year: (2024)
Ref_id:b42 Title: Diffusion policies as an expressive policy class for offline reinforcement learning Year: (2022)
Ref_id:b43 Title: Believe what you see: Implicit constraint approach for offline multi-agent reinforcement learning Year: (2021)
Ref_id:b44 Title: Graph denoising diffusion for inverse protein folding Year: (2024)
Ref_id:b45 Title: The surprising effectiveness of ppo in cooperative multi-agent games Year: (2022)
Ref_id:b46 Title: Graph transformer networks Year: (2019)
Ref_id:b47 Title: Table 7: Comparison between MCGD and baselines on offline Medium-Replay, Medium, and Random datasets across the MPE benchmark: "average value ± standard deviation Year: (2023)
