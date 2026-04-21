Title: HM3: Hierarchical Multi-Objective Model Merging for Pretrained Models
Abstract: Model merging is a technique that combines multiple large pretrained models into a single model, enhancing performance and broadening task adaptability without original data or additional training. However, most existing model merging methods focus primarily on exploring the parameter space, merging models with identical architectures. Despite its potential, merging in the architecture space remains in its early stages due to the vast search space and challenges related to layer compatibility. This paper designs a hierarchical model merging framework named HM3, formulating a bilevel multi-objective model merging problem across both parameter and architecture spaces. At the parameter level, HM3 integrates existing merging methods to quickly identify optimal parameters. Based on these, an actorcritic strategy with efficient policy discretization is employed at the architecture level to explore inference paths with Markov property in the layer-granularity search space for reconstructing these optimal models. By training reusable policy and value networks, HM3 learns Pareto optimal models to provide customized solutions for various tasks. Experimental results on language and vision tasks demonstrate that HM3 outperforms methods focusing solely on the parameter or architecture space.

Section: Introduction
Recent advancements in large pretrained models and large language models (LLMs) have demonstrated remarkable performance and strong generalization abilities across various domains, such as natural language processing [8,84,68]. Open-source communities have provided many pretrained models for various data types, as well as fine-tuned versions tailored to specific tasks. However, fine-tuning large models is often a complex process that requires vast amounts of high-quality data and computational resources [28,16]. To address the challenge of building foundational models capable of handling diverse tasks under limited computational resources, model merging has gained increasing attention [35,77]. Model merging leverages existing pretrained models to flexibly transfer and integrate knowledge without requiring the original training data or additional model training [66,48,37]. This approach enables the creation of new models with stronger generalization capabilities, suited to multiple tasks and scenarios [73]. In recent years, model merging has become a simple yet powerful approach for large foundational model development, with merged models showing significant potential on the Open LLM leaderboard [44]. Current model merging methods primarily focus on merging models with the same architecture in the parameter space [58,57]. They discard most redundant parameters, and only need to design parameter adjustment strategies in the remaining space, which often obtain moderate performance [72,80,27]. Thus, research in the parameter space has become quite extensive and mature [24,39,20,19,18].
However, focusing solely on merging models within the parameter space significantly limits their practical utility [1,73]. Models with different architectures exhibit broader diversity in representation capabilities and task types [38,43,85], potentially expanding the performance boundaries of merged models beyond those of a single architecture. Some approaches [62,61,74] attempt to unify different architectures via knowledge distillation before performing parameter merging. However, these methods still operate within the parameter-space paradigm and typically incur substantial training costs in distillation, especially for LLMs. Recent work has explored architecture-level merging, such as Franken merging [22] and SOLAR 10.7B [29], which stitches different layers from LLMs. Nevertheless, merging models across different architectures presents several practical challenges [17,57], resulting in limited research in this area. Primarily, architecture-level merging alters the computational logic of the model, necessitating the design of coordination strategies to ensure internal compatibility and seamless information flow within the new architecture. Moreover, jointly exploring both the parameter space and architecture space increases the problem's complexity [82], requiring well-defined search spaces and efficient search strategies to identify the optimal model configuration. Recently, evolutionary algorithm (EA) has been employed to search for optimal architectures [1]. However, they fail to reveal the mapping between architecture sequences and performance, making them unsuitable for handling the complex, high-dimensional problem of merging multiple models. Additionally, evolutionary processes are often one-time fusions, requiring a complete restart when faced with new problems, leading to significant computational consumption [68,64].
To this end, merging large pretrain models in parameter and architecture spaces appears to be a promising approach, which can enhance the representational ability of the merged models while maintaining performance. However, research in this area is scarce, primarily because merging models in both spaces without careful consideration can undermine their internal compatibility, potentially causing a performance collapse. In addition, the complexity of the architecture space further increases the difficulty of model merging and reduces the efficiency of existing search methods [1]. Additionally, users may have diverse preferences and expectations for the merged model, making it crucial to weight tasks differently based on these varying preferences [34,33,36].
To merge models across both parameter and architecture levels and achieve efficient model merging schemes, this paper proposes a hierarchical model merging method (HM3) that builds a bridge for model merging in the parameter and architecture spaces. HM3 first defines a joint optimization problem for model merging that spans both the parameter level and the architecture level. Compared to existing methods, HM3 has also taken extra considerations on conflicts or trade-offs across tasks by extending this problem to a multi-objective optimization perspective. In HM3, we sample diverse preference vectors to decompose the multi-objective problem into multiple subproblems, and simultaneously solve them to identify approximate Pareto-optimal merged models across tasks To relieve the strong coupling between variables and the exponentially large search space of each subproblem, HM3 transforms it into a bilevel optimization problem without compromising theoretical optimality. At the architecture level, an actor-critic reinforcement learning (RL) method is designed to explore inference paths with a Markov property in the layer-granularity search space, enabling the reconstruction of these optimal models. To improve efficiency in the large discrete action space, HM3 incorporates a Wolpertinger strategy for policy discretization. Once training is achieved, the policy and value networks of this actor-critic strategy can be reused to predict optimal merging architectures and parameters for different tasks. The final approximate Pareto merged models meet different preferences based on specific needs and trade-offs. The main contributions of this paper are summarized as follows:
• We propose the hierarchical model merging method (HM3), provide the definition of the joint model merging optimization problem that spans parameter and architect space, and transform it into a bilevel optimization problem without losing theoretical optimality to relieve strong coupling and vast search space.
• The proposed HM3 is the first reusable model merging framework by integrating the current parameter-merging method and designing an actor-critic-based RL method with Wolpertinger policy discretization to guide the search, exploring the optimal model configurations in both the parameter and architecture spaces.
• We propose to incorporate a multi-objective optimization paradigm into model merging processing, which allows users to prioritize the importance of multiple tasks based on task needs by searching for the approximate Pareto front of merging strategy, enabling them to select the most suitable merged model.
this section cite: ['b7', 'b83', 'b67', 'b27', 'b15', 'b34', 'b76', 'b65', 'b47', 'b36', 'b72', 'b43', 'b57', 'b56', 'b71', 'b79', 'b26', 'b23', 'b38', 'b19', 'b18', 'b17', 'b0', 'b72', 'b37', 'b42', 'b84', 'b61', 'b60', 'b73', 'b21', 'b28', 'b16', 'b56', 'b81', 'b0', 'b67', 'b63', 'b0', 'b33', 'b32', 'b35']

Section: Hierarchical Multi-Objective Model Merging Framework
This paper aims to jointly optimize both the parameters and architecture of pretrained models to obtain a set of approximately Pareto-optimal merged models that accommodate diverse preferences under multi-task settings. Detailed related work is provided in the appendix A. Since there is a lack of definition for architecture-level merging, we propose a unified mathematical formulation for multi-objective model merging at both the parameter and architecture levels for the first time.
this section cite: []

Section: Problem Formulation and Challenge Discussion
At the parameter level, existing works already define the merging process via optimization over Θ Θ Θ = {θ mt,lt } T t=1 . At the architecture level, we consider the optimization of model architecture α as an inference path search problem, where a search token traverses layers from multiple fine-tuned or merged models to identify an inference path with total length not exceeding T max . This inference path is represented by a sequence {(m t , l t )} T t=1 , where (m t , l t ) denotes the model index and layer index selected at the t-th search step, and T is the total path length. To this end, the unified optimization problem is defined as: max
Θ∈P⊆R d Θ , α={(mt,lt)} T t=1 ∈M F(Θ, α) = (f 1 (Θ, α), f 2 (Θ, α), . . . , f K (Θ, α))(1a)
s.t. C1 : Θ = {θ mt,lt |θ mt,lt = G K k=1 ϖ k θ k,lt } T t=1 ;(1b)
C2 : ϖ k = 1, if θ k,lt
has the same base model as θ mt,lt ; 0, otherwise;
C3 : |α| = T ≤ T max ;(1c)
this section cite: []

Section: C4
:
T -1 t=1 1 dim out (m t , l t ; θ mt,lt ) ̸ = dim in (m t+1 , l t+1 ; θ mt+1,lt+1 ) = 0. (1e
)
where f k (•) for k ∈ {1, 2, . . . , K} denotes the performance on the k-th task; P is the parameter space; Θ Θ Θ is the parameters of the merged model; M = Tmax T =1 {α = {(m t , l t )} T t=1 | m t ∈ {1, . . . ,K}, l t ∈ {1, . . . , L}} is the architecture space; C1 enforces a maximum inference path length of T max ; and C2 strictly ensures that the output dimension of each selected layer matches the input dimension of the next. Compared to the problem formulations of existing model merging methods, our approach extends the formulation to the architecture level. By jointly considering both space, we elevate model merging from a parameter interpolation problem to a more general structural composition problem.
In (1), P is constructed from multiple pretrained LLMs, which results in a high-dimensional, nonconvex, and piecewise linear geometric structure. Additionally, M is a discrete set of cross-model, cross-layer inference paths, whose size grows exponentially with the number of models K and the number of layers L. The strong coupling between Θ Θ Θ and α leads to an extremely large and complex joint search space. Furthermore, multi-objective function F(Θ Θ Θ, α) = (f 1 , . . . , f K ) exhibits nonsmoothness, non-convexity, and non-differentiability under such coupled variables, making it difficult to solve for traditional convex optimization or multi-objective methods. A final challenge lies in layers from different fine-tuned models must be stitched together while preserving dimensional consistency across the output-input interfaces.
this section cite: []

Section: Transform the Problem Formulation into A Bilevel Framework
To address the strong coupling between Θ Θ Θ and α, we reformulate (1) as a bilevel optimization problem [51,13]. In this problem, the upper level searches for the optimal merged parameters Θ Θ Θ * in P, while the lower level searches for the optimal inference path α * under the static environment by the converged upper-level solution Θ Θ Θ * . This decomposition transforms the original joint search space of size |P| × |M| into two sequential subproblems with complexity |P| + |M|, thereby significantly mitigating the combinatorial explosion in the search process. The bilevel optimization problem is given as:
max Θ∈P⊆R d Θ F Θ, α * (Θ) = f 1 Θ, α * (Θ) , f 2 Θ, α * (Θ) , . . . , f K Θ, α * (Θ) (2a
) s.t. Θ = {θ mt,lt |θ mt,lt = G K k=1 ϖ k θ k,lt } T t=1 ;(2b)
ϖ k = 1, if θ k,lt
has the same base model as θ mt,lt ; 0, otherwise;
α * (Θ) ∈ arg max α={(mt,lt)} T t=1 ∈M F Θ, α(2c)
s.t. |α| = T ≤ T max ,(2d)
T -1 t=1 1[dim out (m t , l t ; θ mt,lt ) ̸ = dim in (m t+1 , l t+1 ; θ mt+1,lt+1 )] = 0. ((2e)
)2f
Lemma 1 (Stackelberg Equilibrium [31, 4, Thm.
this section cite: ['b50', 'b12']

Section: 3.1]).
Assume the follower feasible mapping Ω(Θ Θ Θ) = {α ∈ M | |α| ≤ T max , dim out = dim in } is non-empty for all Θ Θ Θ (since α base ∈ Ω), and its graph is closed due to (A1). Under Assumption 1, the bilevel optimization problem (2) admits at least one Stackelberg equilibrium (Θ Θ Θ * , α * ). The associated leader-follower payoff corresponds to a global optimum of the original problem (1). The proof of Lemma 1 is provided in the appendix B.1.
In the appendix B.1, we further prove that the bilevel optimization problem can be modeled as a Stackelberg game, for which an equilibrium solution exists. This ensures that problem transformation does not incur any loss of optimality. In this bilevel optimization problem, the lower-level optimization searches for the optimal merged model architecture. The resulting architecture determines the length of the inference path (i.e., the number of layers to be merged). The upper-level optimization then operates on the parameter set {θ θ θ mt,lt } T t=1 corresponding to this architecture. Consequently, the optimal architecture found by the lower level dynamically determines the dimensionality and scale of the parameter search space for the upper level. This naturally forms a hierarchical decision-making structure, i.e., first optimizing the model architecture, then optimizing the corresponding model parameters, which embodies the core hierarchical nature of the proposed HM3.
this section cite: []

Section: Model the User Preference into A Multi-Objective Optimization Problem
After mitigating the strong coupling between variables, we focus on the multi-objective property of (2). To accommodate diverse user preferences, we adopt a decomposition-based strategy that explicitly guides the solution set to cover the Pareto front boundary under controllable preference vectors. Due to the nonconvexity of the search space, we employ Tchebycheff decomposition strategy, which effectively approximates non-convex Pareto fronts by transforming the original multi-objective problem into N preference-weighted scalar subproblems. By solving these subproblems in parallel, we obtain a set of approximately Pareto-optimal merged models that satisfy varying user preferences. Specifically, we begin by generating N preference weight vectors {λ
(i) } N i=1 from K-dimensional probability simplex ∆ K = {λ ∈ R K + | K k=1 λ k = 1}.
In this paper, we sample from the Dirichlet distribution: λ (i) ∼ Dirichlet(1, . . . , 1 K ), i = 1, . . . , N, which ensures uniform coverage over ∆ K with an unbiased mean E[λ k ] = 1 K . Then, we estimate the ideal point of each objective by computing the best achievable task performance across all fine-tuned LLMs: z * k = max (Θ Θ Θ,α)∈Ω f k (Θ Θ Θ, α). Finally, for each preference vector λ λ λ (i) , the upper-level subproblem using Tchebycheff scalarization is defined as:
Θ Θ Θ (i) = arg min Θ Θ Θ∈P max k=1,...,K λ (i) k • F para k (Θ Θ Θ) -z * k ,(3)
where F para k (Θ Θ Θ) denotes the objective function value of the merged model on the k-th task, obtained after solving the corresponding lower-level inference path problem with fixed Θ Θ Θ. Once Θ Θ Θ (i) obtained, the corresponding lower-level subproblem is defined as:
α (i) = arg min α∈Ω(Θ Θ Θ (i) ) max k=1,...,K λ (i) k • f k (Θ Θ Θ (i) , α) -z * k , (4
)
where Ω(Θ Θ Θ (i) ) denotes the feasible set of inference paths that satisfy C1 and C2.
Through this transformation, (1) is reduced to solving N scalarized bilevel subproblems, each corresponding to a distinct preference vector λ (i) . The set of solutions to all subproblems forms an approximate Pareto-optimal set of merged models.
this section cite: []

Section: HMAlgorithm
Parameter-Level Optimization After generating preference vectors {λ (i) = {λ (i) 1 , . . . , λ (i) K }} N i=1 , we proceed to search for the optimal merged parameters in the parameter space for each preference vector. Thanks to recent advancements, parameter-level merging methods have become relatively mature and efficient. Our framework is designed to be compatible with these existing techniques, such as DARE-Ties merging method. Concretely, for λ (i) , we begin by computing the residual vector: δ k = Θ Θ Θ k -Θ Θ Θ base . We then apply the Drop-and-Rescale operation to obtain δ DR k = δ k /(1 -p). Next, we perform the Ties Merging [72] procedure for λ (i) : removing redundant parameters from each δ DR k , generating a sign-consistent aggregation mask across tasks, and merging disjoint residual fragments with consistent signs to form δ ′ k . Finally, the optimal parameter for the i-th subproblem is:
Θ Θ Θ (i) * = Θ Θ Θ base + K k=1 λ (i) k • M (i) k ⊙ δ k ,(5)
where
M (i)
k is a binary mask that controls which elements of δ k are preserved and rescaled.
Architecture-Level Optimization As discussed in Section 2, architecture-level optimization is formulated as searching for an optimal inference path across the merged model and its multiple task-specific fine-tuned models. For each λ λ λ (i) , we have already obtained the corresponding optimal merged model at the parameter level, denoted as Θ Θ Θ (i) * . We assign its model index as m = K + 1.
To further expand the search space and leverage external knowledge, we construct a model-layer candidate pool that consists of: (1) the optimal merged model Θ Θ Θ (i) * from the parameter level, and (2) all layers from the K task-specific fine-tuned models used in construction of Θ Θ Θ (i) * . The corresponding architecture-level search space is then updated as:
M (i) = (m (i) , l (i) ; Θ Θ Θ (i) ) | m ∈ {1, . . . , K + 1}, l ∈ {1, . . . , L} .(6)
Although prior research has demonstrated the potential of using search algorithms to optimize layer sequences and enhance model merging performance [1], the scalability of EAs suffers significantly as the number of models and layers increases [20,18]. Moreover, EA-based approaches require training from scratch for each preference vector and incur considerable computational cost due to population-based evaluations in every generation. These inefficiencies motivate us to revisit the nature of dynamic layer selection across multiple models [69,67].
This process entails selecting the optimal model-layer pair at each step, considering the long-term impact of current decisions on future layer compositions and final task performance, thereby exhibiting the characteristics of a sequential decision-making problem. Furthermore, the combinatorial nature of the layer-path space, along with its discrete, structured constraints and well-defined state transitions, naturally suggests formulating the inference path search as a trajectory-aware Markov decision process (MDP). The overall process of architecture-merging is illustrated in Fig. 1. Then, we formally define its state, action, transition, and reward components and design an RL strategy to efficiently explore optimal architecture trajectories.
this section cite: ['b71', 'b0', 'b19', 'b17', 'b68', 'b66']

Section: 1) State Space
Since every decision in the inference path dynamically alters the feasibility of subsequent layer transitions and affects the accumulated representation distribution, we define the state to retain full trajectory history for optimal distinguishability. Formally, the state at the tth step is represented as a trajectory:
S t = (m j , l j , θ mj ,lj ) m j ∈ {1, . . . , K + 1}, l j ∈ {1, . . . , L}, j = 1, . . . , t ∈ S,(7)
where m j denotes the model index, l j denotes the layer index, and θ θ θ mj ,lj ∈ R d θ is the corresponding parameter of the selected layer. To enable policy gradient-based learning, we use a set of learnable encoders ψ m , ψ l , and φ to encode model identity, layer index, and layer parameters, respectively. The full trajectory is then embedded into a fixed-dimensional vector using a GRU encoder: h t = GRU ψ m (m j ); ψ l (l j ); φ(θ mj ,lj ) t j=1 ∈ R d h . The process on trajectory space S satisfies Markov property.
2) Action Space At the tth step, the action A t is defined as selecting the next model-layer pair to transition to:
A t = (m t+1 , l t+1 ) ∈ A, m ∈ {1, . . . , K + 1}, l ∈ {1, . . . , L}.(8)
3) Reward Function. The reward encourages the construction of efficient inference paths that yield high-quality multi-task performance with minimal complexity:
R = K k=1 λ (i) k f k (Θ, h) -β 1 T(9)
where the second term penalizes path length to encourage shorter and more efficient inference paths. The reward is computed only after the entire inference path is generated, and the MLP-based alignment is performed. This reward is uniformly assigned to all time steps in the inference path as R t = R/T, ∀t ∈ {1, . . . , T }. This uniform assignment is implemented to facilitate efficient storage of transitions and subsequent updates of the policy and value networks in the actor-critic framework.
Actor-Critic Method To solve the MDP, HM3 employs an actor-critic-based RL strategy. In this framework, a policy network parameterized by µ outputs a probability distribution over the large discrete action space, while a value network parameterized by ϕ serves as a baseline to reduce the variance of gradients under sparse reward conditions. This design facilitates stable convergence of layer sequence search under limited sampling [79,78].
The policy function π µ (A t | S t ) defines a stochastic policy conditioned on the current state S t , representing the probability distribution over candidate actions. The distribution is modeled using a Gaussian parameterization with mean µ and variance ξ 2 , from which actions are sampled to maximize the expected cumulative reward:
max µ E πµ T t=0 γ t R t = max µ E πµ T t=0 γ t K k=1 λ (i) k f k (Θ Θ Θ, h t ) -β 1 t ,(10)
where γ ∈ (0, 1] is the discount factor, and R t denotes the reward at the tth step as defined earlier.
The policy network generates a continuous proto-action Ā following a Gaussian-distributed stochastic policy: Ā = f π(A|S;Θ Θ Θ) (S) ∼ N µ π (S t ), diag σ 2 π (S t ) , where f π(A|S;Θ Θ Θ) is the state-to-action mapping under policy π.
Since the decision variables of (2) lie in a discrete action space W, the proto-action Ā must be mapped to a discrete action A ∈ W. Existing discretization approaches fall into two categories [55,83]: The simple projection method, which directly selects the nearest discrete action: A * = arg min A∈W A -Ā . However, this can result in suboptimal exploration and slow convergence. The greedy method, which selects the action with the highest Q-value: A * = arg max A∈W Q(S, A), but this is often computationally expensive and prone to local optima.
To balance exploration and exploitation, we introduce Wolpertinger policy mapping, which improves efficiency by limiting evaluation to a local neighborhood: The value function is modeled by a state-value network V ϕ (S t ), which estimates the expected cu-
A t = arg max A∈W * (At) Q ϕ (S t , A), with probability 1 -ϵ, U(W * (A t )), with probability ϵ,(11)
mulative reward of S t : V ϕ (S t ) = E πµ ∞ j=0 γ j R t+j S t ,
where ∞ j=0 γ j R t+j is the discounted return starting from the t-th step.
Network Updates To stabilize policy optimization, HM3 constrains the update step size and adopts a policy gradient approach for training. The policy network is updated using the clipped surrogate objective by proximal policy optimization [49]:
L CLIP (µ) = E t min ρ t (µ) Ât , clip(ρ t (µ), 1 -ϵ, 1 + ϵ) Ât ,(12)
where ρ t (µ) = πµ(At|St) πµ old (At|St) is the importance sampling ratio between the new and old policies, and Ât is the estimated advantage. We compute Ât using the generalized advantage estimation method:
Ât = ∞ i=0 (γβ A ) i ζ t+i , ζ t = R t + γV (S t+1 ; ϕ iter ) -V (S t ; ϕ iter ),
where ζ t is the temporal difference residual at the t-th step.
The value network is updated by minimizing the value loss:
L VF (ϕ) = E t V ϕ (S t ) -V ϕ (S t ) -Ât 2 . (13
)
The overall training objective is:
L(µ, ϕ) = L CLIP (µ) + c 1 L VF (ϕ) -c 2 H(π µ ),(14)
where H(π µ ) denotes the entropy of the policy, and c 1 , c 2 are weighting coefficients.
this section cite: ['b78', 'b77', 'b54', 'b82', 'b48']

Section: Lemma 2 (Advantage of Wolpertinger discretization).
Let Q(S, A) = r(S, A) + γV ϕ (S ′ ) de- note the one-step proxy score derived from the value network V ϕ . Consider a candidate set W * = {A 1 , . . . , A ℓ , . . . , A M }. Assume there exists a constant ξ > 0 such that: Q(S, A ℓ ) ∼ U Q(S, A * s ) -ξ, Q(S, A * s ) + ξ , ∀ℓ ̸ = ℓ ′
, and that the proxy error is bounded as:
Q(S, A) -Q(S, A) ≤ δ, ∀A ∈ W * . (15
)
When M > 1 and δ < ξ 1 -
2(2M -1) M •2 M
, we can confirm that Wolpertinger is expected to outperform simple nearest-neighbor projection in terms of the true Q-value. Moreover, by reducing the candidate space from |W| to |W * |, Wolpertinger achieves greater efficiency than full greedy search over all actions. The proof of Lemma 2 is provided in the appendix B.2.
this section cite: []

Section: Dimension Alignment via Statistical Matching
To accommodate distributional shifts across layers from different models, we introduce a feed-forward MLP network that generates a scaling matrix W m,l . The input to the MLP consists of the layer index pair (m, l) and the current time step t, and its output is defined as:
W m,l = MLP µ (m, l, t),(16)
where MLP µ is parameterized by µ and optimized via actor-critic method. This design is motivated by the theory of moment matching [53]. Further theoretical details are provided in the appendix B.4. It is worth that the proposed HM3 is in its early exploratory stage, and we discuss the existing limitations and possible future directions in the appendix D.
this section cite: ['b52']

Section: Experiment Setup
Baselines We evaluate the proposed parameter-and-architecture hierarchical merging framework HM3foot_0 against three types of baselines on both language and vision tasks: fine-tuned models, three classical parameter-level merging methods, including Task Arithmetic [27], Ties-Merging [72], and DARE-Ties Merging [80], two SOTA parameter-level merging methods, including PCB Merging [19] and Consensus Merging [65], and an architecture-level merging method named EA [1].
Benchmarks and Metrics For language tasks, we used LLAMA-2-7B [60], Qwen-2.5-1.5B [75], and LLAMA-2-13B [60] as backbones across four subtasks: generative task, text translation, math reasoning, and code generation. For generative tasks, we used GLUE benchmark [63] to evaluate the general capability of large pretrained models. For translation, we used WMT14, WMT16 [50], and IWSLT2017 [7] (WMT&ISWT), evaluated by the chrf metric as well as Xnli [15] evaluated by the accuracy metric. For math reasoning, we used GSM8K [12] with the flexible match metric, and used MathQA [3] with the accuracy metric. For code generation, HumanEval [9] and MBPP [5] was used with the pass@1 and pass@100 metric. Additionally, Qwen-2.5-1.5B was evaluated on four 3090 GPUs (24GB each), while LLaMA-2-7B and LLaMA-2-13B were evaluated on four A6000 GPUs (48GB each). All models can also be deployed on a single GPU. For vision tasks, we adopted ViT-B/32 and ViT-L/14 from CLIP [46] as backbones, and evaluated on eight datasets: DTD [11], GTSRB [52], RESISC45 [10], SUN397 [70], SVHN [45], MNIST [32], Cars [30], and EuroSAT [23], using classification accuracy. Other settings and details are summarized in the appendix C.1.
this section cite: ['b26', 'b71', 'b79', 'b18', 'b64', 'b0', 'b59', 'b74', 'b59', 'b62', 'b49', 'b6', 'b14', 'b11', 'b2', 'b8', 'b4', 'b45', 'b10', 'b51', 'b9', 'b69', 'b44', 'b31', 'b29', 'b22']

Section: Performance of Multi-Task Scenario
Merging LLAMA-2-7B LLMs Table 1 summarizes the performance of various merging methods across three language tasks on LLAMA-7B series LLMs. Among the fine-tuned models, WizardMath-7B [41] excelled at math due to task-specific training, while CodeLlama-7B [47] dominated code generation. Llama-2-7B-Chat [60] showed relatively balanced performance, particularly in translation. Across merging methods, Task Arithmetic provided moderate gains across tasks, whereas Ties Merging and DARE-Ties Merging achieved better trade-offs, especially in translation and code. However, EA underperformed, likely due to its unguided architecture search, which struggles to find optimal layer combinations with limited evaluations. Our proposed HM3 significantly outperformed all baselines, achieving top scores in all tasks. These results highlight the effectiveness of jointly optimizing both parameter fusion and architectural composition.
Merging Qwen-1.5B LLMs To assess the robustness of HM3, we conducted merging experiments using the Qwen-2.5-1.5B series LLMs. As shown in Table 2, each fine-tuned model performed best on its own task but showed clear limitations on others, reflecting the trade-offs of single-task fine-tuning. In contrast, HM3 consistently outperformed all baselines, achieving top results in math and code, and competitive performance in translation. EA performed the worst across all tasks due to its unguided structure search. An interesting observation from Table 1 and Table 2 is that the models by HM3 sometimes outperform fine-tuned models, which are typically considered performance upper bounds for their respective tasks. We discuss this in the appendix C.2. Additionally, we conducted the experiment on LLAMA-13B, and the results and analysis are provided in the appendix C.2.  Merging ViT-L/14 Table 4 shows that HM3 consistently achieves the best results across most datasets, with 90.48% on SVHN and 83.43% on GTSRB. The overall average accuracy reaches 80.30%, significantly exceeding all other methods. The detailed analysis is in the appendix C.2.
this section cite: ['b40', 'b46', 'b59']

Section: Performance of Multi-Objective Model Merging
HM3 generates a diverse set of approximately Pareto-optimal merged models, enabling flexible adaptation to different user preferences. Unlike existing methods that output a single solution, HM3 provides multiple high-quality candidates. To evaluate solution quality, we compute Pareto dominance relations by pooling all solutions. A solution x a is dominated by x b if x b is no worse in all objectives and strictly better in at least one. Figure 2 shows that every baseline is dominated by at least one HM3 solution (S1-S15), demonstrating HM3's superiority in objective space. We also compare HM3 with a multi-objective evolutionary algorithm (MOEA) baseline using the hypervolume (HV)    metric [25,56], which reflects both convergence and diversity. HM3 achieves an HV of 1.8120, significantly higher than MOEA's 1.5111, highlighting the limitations of unguided evolutionary search in complex multi-objective scenarios. Detailed analysis is provided in the appendix C.3. The effectiveness of HM3 on different numbers of objectives is provided in the appendix C.4.
this section cite: ['b24', 'b55']

Section: Ablation Study
To evaluate the effectiveness of jointly optimizing parameter and architecture spaces, we conduct ablation studies on three variants: (i) HM3, (ii) HM3 w.o. arch (no architecture optimization), and (iii) HM3 w.o. para (no parameter optimization), with results shown in Table 6 in the appendix C.5.
In the single-objective setting, HM3 outperforms both ablated versions on all tasks, especially in code generation, highlighting the synergy between parameter and architecture optimization. In the multi-objective setting, HM3 achieves the highest HV score, followed by HM3 w.o. arch, while HM3 w.o. para performs the worst. This demonstrates that parameter optimization is critical for overall performance, and architecture optimization further enhances solution quality. Detailed analysis is provided in the appendix C.5. We also analyze the computational cost of HM3 compared to the conventional pretraining and fine-tuning paradigm in the appendix C.6. Additionally, convergence analysis of RL is provided in the appendix C.7.
this section cite: []

Section: Conclusion
In this paper, we propose HM3, a hierarchical model merging framework that jointly optimizes parameter and architecture spaces. By leveraging an actor-critic strategy and preference-guided multi-objective optimization, HM3 efficiently generates customized, high-performing merged models. Extensive experiments on translation, math reasoning, and code generation tasks demonstrate HM3's superiority over existing methods. The framework learns Pareto-optimal solutions tailored to diverse user preferences, offering a flexible and scalable approach to model merging. Future work will explore applying HM3 to larger-scale pretrained models for broader generalization and adaptability.
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We specify them in our code. Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: References
Ref_id:b0 Title: Evolutionary optimization of model merging recipes Year: (2025)
Ref_id:b1 Title: Infinite dimensional analysis: a hitchhiker's guide Year: (2006)
Ref_id:b2 Title: Towards interpretable math word problem solving with operation-based formalisms Year: (2019)
Ref_id:b3 Title: A short state of the art on multi-leader-follower games. Bilevel optimization: Advances and next challenges Year: (2020)
Ref_id:b4 Title: Program synthesis with large language models Year: (2021)
Ref_id:b5 Title: Topological Spaces: Including a Treatment of Multi-valued Functions, Vector Spaces, and Convexity Year: (1997)
Ref_id:b6 Title: Overview of the iwslt 2017 evaluation campaign Year: (2017)
Ref_id:b7 Title: A survey on evaluation of large language models Year: (2024)
Ref_id:b8 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b9 Title: Remote sensing image scene classification: Benchmark and state of the art Year: (2017)
Ref_id:b10 Title: Describing textures in the wild Year: (2014)
Ref_id:b11 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b12 Title: An overview of bilevel optimization Year: (2007)
Ref_id:b13 Title: Have you merged my model? on the robustness of large language model ip protection methods against model merging Year: (2024)
Ref_id:b14 Title: Xnli: Evaluating cross-lingual sentence representations Year: (2018)
Ref_id:b15 Title: Zero-shot crossdomain dialogue state tracking via context-aware auto-prompting and instruction-following contrastive decoding Year: (2024)
Ref_id:b16 Title: Nats-bench: Benchmarking nas algorithms for architecture topology and size Year: (2021)
Ref_id:b17 Title: Neural parameter search for slimmer fine-tuned models and better transfer Year: (2025)
Ref_id:b18 Title: Parameter competition balancing for model merging Year: (2024)
Ref_id:b19 Title: Knowledge fusion by evolving weights of language models Year: (2024)
Ref_id:b20 Title: A framework for few-shot language model evaluation Year: ()
Ref_id:b21 Title: Arcee's mergekit: A toolkit for merging large language models Year: (2024)
Ref_id:b22 Title: Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification Year: (2019)
Ref_id:b23 Title: Emr-merging: Tuning-free high-performance model merging Year: (2024)
Ref_id:b24 Title: An evolution strategy with probabilistic mutation for multi-objective optimisation Year: (2003)
Ref_id:b25 Title: Qwen2. 5-coder technical report Year: (2024)
Ref_id:b26 Title: Editing models with task arithmetic Year: (2023)
Ref_id:b27 Title: Model stock: All we need is just a few fine-tuned models Year: (2025)
Ref_id:b28 Title: Solar 10.7 b: Scaling large language models with simple yet effective depth up-scaling Year: (2024)
Ref_id:b29 Title: 3d object representations for finegrained categorization Year: (2013)
Ref_id:b30 Title: An existence result for hierarchical stackelberg v/s stackelberg games Year: (2015)
Ref_id:b31 Title: The mnist database of handwritten digits Year: (1998)
Ref_id:b32 Title: It's morphing time: Unleashing the potential of multiple llms via multi-objective optimization Year: (2025)
Ref_id:b33 Title: Map: Low-compute model merging with amortized pareto fronts via quadratic approximation Year: (2024)
Ref_id:b34 Title: Deep model fusion: A survey Year: (2023)
Ref_id:b35 Title: Multi-objective large language model alignment with hierarchical experts Year: (2025)
Ref_id:b36 Title: Merge, ensemble, and cooperate! a survey on collaborative strategies in the era of large language models Year: (2024)
Ref_id:b37 Title: Fine-tuning large language models for domain adaptation: Exploration of training strategies, scaling, model merging and synergistic capabilities Year: (2025)
Ref_id:b38 Title: Twinmerging: Dynamic integration of modular expertise in model merging Year: (2024)
Ref_id:b39 Title: Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct Year: (2025)
Ref_id:b40 Title: Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct Year: (2023)
Ref_id:b41 Title: Wizardcoder: Empowering code large language models with evol-instruct Year: (2024)
Ref_id:b42 Title: Neural architecture search without training Year: (2021)
Ref_id:b43 Title: Open-llm-leaderboard: From multi-choice to open-style questions for llms evaluation, benchmark, and arena Year: (2024)
Ref_id:b44 Title: Reading digits in natural images with unsupervised feature learning Year: (2011)
Ref_id:b45 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b46 Title: Code llama: Open foundation models for code Year: (2023)
Ref_id:b47 Title: From task-specific models to unified systems: A review of model merging approaches Year: (2025)
Ref_id:b48 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b49 Title: Edinburgh neural machine translation systems for wmt 16 Year: (2016)
Ref_id:b50 Title: A review on bilevel optimization: From classical to evolutionary approaches and applications Year: (2017)
Ref_id:b51 Title: The german traffic sign recognition benchmark: a multi-class classification competition Year: (2011)
Ref_id:b52 Title: Deep coral: Correlation alignment for deep domain adaptation Year: (2016)
Ref_id:b53 Title: Cat merging: A training-free approach for resolving conflicts in model merging Year: (2025)
Ref_id:b54 Title: Reinforcement learning: An introduction Year: (2018)
Ref_id:b55 Title: Multiobjective evolutionary algorithms and applications Year: (2005)
Ref_id:b56 Title: Fusionbench: A comprehensive benchmark of deep model fusion Year: (2024)
Ref_id:b57 Title: A unified view of delta parameter editing in post-trained large-scale models Year: (2024)
Ref_id:b58 Title: Qwen2.5: A party of foundation models Year: (2024-09)
Ref_id:b59 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b60 Title: Knowledge fusion of large language models Year: ()
Ref_id:b61 Title: Fusechat: Knowledge fusion of chat models Year: (2024)
Ref_id:b62 Title: Glue: A multi-task benchmark and analysis platform for natural language understanding Year: (2019)
Ref_id:b63 Title: When large language models meet evolutionary algorithms: Potential enhancements and challenges Year: (2025)
Ref_id:b64 Title: Localizing task information for improved model merging and compression Year: (2024)
Ref_id:b65 Title: Sampling generative networks Year: (2016)
Ref_id:b66 Title: Towards robustness and explainability of automatic algorithm selection Year: ()
Ref_id:b67 Title: Evolutionary computation in the era of large language model: Survey and roadmap Year: (2025)
Ref_id:b68 Title: Large language modelenhanced algorithm selection: towards comprehensive algorithm representation Year: (2024)
Ref_id:b69 Title: Sun database: Exploring a large collection of scene categories Year: (2016)
Ref_id:b70 Title: Wizardlm: Empowering large pre-trained language models to follow complex instructions Year: (2024)
Ref_id:b71 Title: Ties-merging: resolving interference when merging models Year: (2023)
Ref_id:b72 Title: What matters for model merging at scale? arXiv preprint Year: (2024)
Ref_id:b73 Title: Infifusion: A unified framework for enhanced cross-model reasoning via llm fusion Year: (2025)
Ref_id:b74 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b75 Title: Qwen2.5-math technical report: Toward mathematical expert model via self-improvement Year: (2024)
Ref_id:b76 Title: Model merging in llms, mllms, and beyond: Methods, theories, applications and opportunities Year: (2024)
Ref_id:b77 Title: Diversity-aware policy optimization for large language model reasoning Year: (2025)
Ref_id:b78 Title: Policy space diversity for non-transitive games Year: (2023)
Ref_id:b79 Title: Language models are super mario: Absorbing abilities from homologous models as a free lunch Year: (2024)
Ref_id:b80 Title: Nature-inspired population-based evolution of large language models Year: (2025)
Ref_id:b81 Title: A survey on evolutionary construction of deep neural networks Year: (2021)
Ref_id:b82 Title: Decomposition and meta-drl based multi-objective optimization for asynchronous federated learning in 6g-satellite systems Year: (2024)
Ref_id:b83 Title: Causalbench: A comprehensive benchmark for causal learning capability of large language models Year: (2024)
Ref_id:b84 Title: Learning transferable architectures for scalable image recognition Year: (2018)
