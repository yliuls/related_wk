Title: Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems
Abstract: Existing GFlowNet-based methods for vehicle routing problems (VRPs) typically employ Trajectory Balance (TB) to achieve global optimization but often neglect important aspects of local optimization. While Detailed Balance (DB) addresses local optimization more effectively, it alone falls short in solving VRPs, which inherently require holistic trajectory optimization. To address these limitations, we introduce the Hybrid-Balance GFlowNet (HBG) framework, which uniquely integrates TB and DB in a principled and adaptive manner by aligning their intrinsically complementary strengths. Additionally, we propose a specialized inference strategy for depot-centric scenarios like the Capacitated Vehicle Routing Problem (CVRP), leveraging the depot node's greater flexibility in selecting successors. Despite this specialization, HBG maintains broad applicability, extending effectively to problems without explicit depots, such as the Traveling Salesman Problem (TSP). We evaluate HBG by integrating it into two established GFlowNet-based solvers, i.e., AGFN and GFACS, and demonstrate consistent and significant improvements across both CVRP and TSP, underscoring the enhanced solution quality and generalization afforded by our approach.

Section: Introduction
Vehicle Routing Problems (VRPs) are fundamental to real-world operations, including e-commerce logistics [54,12,39], urban delivery [56,7,21], supply chain management [11,15,8], and ridesharing systems [18,30,44]. Efficient VRP solutions directly affect cost reduction, service quality, and overall performance in transportation and supply chain networks. Over the past decades, numerous heuristic and meta-heuristic algorithms, such as the Lin-Kernighan-Helsgaun algorithm [19], ant colony optimization (ACO) [3], hybrid genetic search [46], tabu search [2], and simulated annealing [36], have been developed to address the combinatorial complexity of VRPs. However, these approaches often depend on handcrafted rules and problem-specific heuristics, which limit their adaptability and scalability across diverse VRP instances. More recently, reinforcement learning and deep learning methods have emerged as promising alternatives [38,5,52,20]. Models such as POMO [25], NeuOpt [31], and DEITSP [47] show potential in reducing dependence on handcrafted components. Yet, these methods still struggle to consistently achieve desirable performance, often becoming trapped in local optima due to the limited exploration capacity.
To improve exploration, recent work has explored the use of Generative Flow Network (GFlownet) [4], which generate diverse and high-quality solutions through a probabilistic, generative process. Unlike traditional learning-based approaches that focus on optimizing a single or a few trajectories, GFlowNet aims to learn a distribution over the solution space, making them well-suited for combinatorial problems like VRPs. However, current GFlowNet-based methods for VRPs such as GFACS [22] and AGFN [55], rely exclusively on global optimization during training. Particularly, they both adopt the
this section cite: ['b53', 'b11', 'b38', 'b55', 'b6', 'b20', 'b10', 'b14', 'b7', 'b17', 'b29', 'b43', 'b18', 'b2', 'b45', 'b1', 'b35', 'b37', 'b4', 'b51', 'b19', 'b24', 'b30', 'b46', 'b3', 'b21', 'b54']

Section: GFlowNet for Combinatorial Optimization Problems
GFlowNet has been applied across a wide range of structured generation and decision-making tasks. In molecular and drug discovery [57,34,24,41,16,40], they are used to sample diverse, high-reward molecules from complex solution spaces. In causal structure learning [28,9], GFlowNet facilitates exploration over multiple plausible directed acyclic graphs (DAGs), while in Bayesian inference [10,42,35], they serve as alternative samplers for discrete posteriors. Additional applications include symbolic reasoning [45,27], robotics planning [29,33], and solving maximum independent set (MIS) [53], where modeling solution diversity is essential. Recently, GFlowNet has also been applied to VRPs [55,22], including TSP and CVRP. In this context, learning a distribution over feasible routes offers a flexible and effective alternative to deterministic solvers. Two representative models are AGFN and GFACS. AGFN incorporates adversarial training to improve trajectory construction in an end-to-end fashion, making it the first to apply GFlowNet to VRPs directly. In contrast, GFACS integrates GFlowNet with ant colony optimization, marking the first attempt to augment heuristic search with GFlowNet-based learning. In this paper, we further enhance both AGFN and GFACS for solving VRPs using our proposed Hybrid-Balance GFlowNet framework.
this section cite: ['b56', 'b33', 'b23', 'b40', 'b15', 'b39', 'b27', 'b8', 'b9', 'b41', 'b34', 'b44', 'b26', 'b28', 'b32', 'b52', 'b54', 'b21']

Section: Hybrid-Balance GFlowNet

this section cite: []

Section: HBG Loss

this section cite: []

Section: Current State S t

this section cite: []

Section: Trajectory generation

this section cite: []

Section: Next
State S t+1 Trajectory Local Search GFACS Energy Reshaping GFACS TB Loss Discriminator AGFN TB Loss GFACS AGFN TB Training Node Embedding v 1 v 2 … v k Linear Layer Z(S t+1 ) R(S t+1 ) Pf(S t+1 ) Pb(S t+1 ) DB Loss
this section cite: []

Section: DB Training
Pheromone Map
this section cite: []

Section: GFACS Ant Colony Search

this section cite: []

Section: Edge Probability Distribution

this section cite: []

Section: AGFN Generator
Figure 1: The Overall Framework of Our Hybrid-Balance GFlowNet for Solving VRPs.
As illustrated in Fig. 1, the Hybrid-Balance GFlowNet (HBG) begins with trajectory generation using either AGFN or GFACS, during which state transition information is recorded at each step. The generated next state is then treated as the current state in the following step, and this process repeats until a complete trajectory is constructed. The blue region in Fig. 1 corresponds to the original components from AGFN and GFACS, responsible for the processing of the complete trajectory and the computation of the TB loss, which captures global optimization signals. However, relying solely on the TB loss can cause the model to overlook important relationships between individual states. To bridge this gap, our proposed HBG introduces additional components, highlighted in purple, where the DB loss is computed for each individual state transition to enhance local optimization. The final training objective is a combination of two loss, which together guide the optimization of the model.
To better illustrate the motivation for introducing DB, consider a long vehicle routing trajectory that is incrementally constructed as
A → B → C → D → E → ... → U → V → W → X → Y → Z.
Assume this complete route yields a high total cost (bad performance), primarily due to suboptimal decisions made in the early stages, such as traversing a high-cost edge from node C to node D. In contrast, the latter portion of the tour (e.g., from node U to Z) may follow a more cost-effective and well-structured pattern. Under Trajectory Balance (TB), the final reward is determined by the overall trajectory cost, and is proportionally assigned to all transitions. Consequently, even high-quality local transitions, such as W → X → Y, may receive weak or misleading training signals simply because they are embedded in a globally suboptimal trajectory. This would hinder the model's ability to learn and reinforce desirable local patterns. On the other hand, Detailed Balance (DB) operates at a step-wise granularity, evaluating the expected outcomes of individual transitions. For instance, at node W, DB can assess whether transitioning to X leads to better outcomes compared to alternative choices like Z, regardless of earlier suboptimal steps. This localized and reward-sensitive feedback enables the model to more accurately learn local quality from global performance, and promotes stronger learning signals for valuable decisions even within imperfect trajectories. This example illustrates a core limitation of Trajectory Balance (TB) in long-horizon combinatorial tasks like VRP: when the overall trajectory is suboptimal, TB lacks the ability to identify and preserve well-structured local segments within it. As a result, valuable local patterns may be overlooked or penalized. By incorporating Detailed Balance (DB) into the training objective, we address this limitation by providing fine-grained, step-level singal that helps isolate and reinforce high-quality local decisions, even when the global trajectory does not show good performance.
this section cite: []

Section: Modeling Basics
Problem Definition. For a CVRP instance, G denotes the input graph, which includes the coordinates and demands of customers, as well as the depot location. Formally, the instance is represented as a complete graph G = (V, U), where V = {v 0 , v 1 , . . . , v n } denotes the set of nodes, with v 0 as the depot and the remaining nodes representing customers, and U is the set of edges. Each customer node v i (i ≥ 1) is associated with a demand d i and a location in Euclidean space. Each edge (v i , v j ) ∈ U has an associated cost c ij , typically defined as the Euclidean distance between v i and v j . The goal of CVRP is to determine a set of vehicle routes that start and end at the depot, such that each customer is visited exactly once, the total demand on each route does not exceed the vehicle's capacity, and the total routing cost is minimized.
this section cite: []

Section: State s: In a trajectory set
T = {τ 1 , τ 2 , . . . , τ h }, the state s i denotes the sequence of nodes visited in trajectory τ i . At decision step t in τ i , the state is defined as s i t = {x i 0 , x i 1 , x i 2 , . . . , x i t }, where x i t is the most recently visited node, and x i 0 represents the depot which serves as both the starting and ending point of the route. Action a: An action a i t transitions the system from state s i t to s i t+1 . Given s i t = {x i 0 , x i 1 , . . . , x i t }, the action selects the next node x i
t+1 from the set of unvisited nodes, adhering to feasibility constraints such as vehicle capacity. Once all customers are visited, the route terminates in a final state, forming a complete trajectory τ i = {x i 0 , x i 1 , . . . , x i m }. Reward R: The reward R(τ i ) is determined by the quality of the generated trajectory τ i . We define two types of rewards: R(τ i ) and R(s i t ). The former, R(τ i ), evaluates the entire trajectory, while the latter, R(s i t ), reflects local reward signals at individual state transitions. These are defined as: R(τ
i ) = m-1 k=0 d(x i k , x i k+1 ), R(s i t ) = d(x i t-1 , x i t ), where d(x i k , x i k+1 )
denotes the Euclidean distance between consecutive nodes.
this section cite: []

Section: Graph Neural Network (GNN).
We integrate a GNN module [48] into the GFlowNet framework to more effectively capture the complex relational structures inherent in VRP instances. The detailed architecture and formulation are provided in Appendix A.1. Following the designs of AGFN and GFACS, we sparsify the fully connected graph into a k-nearest-neighbor graph G to improve scalability and reduce computational cost. The graph G is embedded into a high-dimensional feature space, encoding node coordinates and edge distances as node and edge features, respectively. The GNN, parameterized by θ, processes these features through multiple layers to produce rich representations. The resulting edge embeddings are passed through a multi-layer perceptron (MLP) to generate edge probability distribution η(G * , θ) for decision making by GFlowNet, while the node embeddings Q = {q 1 , q 2 , . . . , q b } are retained for computing state flows.
this section cite: ['b47']

Section: Hybrid-Balance GFlowNet

this section cite: []

Section: Global Optimization via TB
In VRPs, the objective is to determine the shortest route while satisfying various operational constraints, which necessitates evaluating solutions from a global perspective. Both AGFN and GFACS adopt the Trajectory Balance (TB) objective to address this requirement, as it enables the GFlowNet to be trained over entire trajectories, naturally aligning with global optimization goals.
As illustrated in Fig. 1, AGFN generates an edge probability distribution η(G * , θ generator ) using GFlowNet, which is then used to sample the next node in the route. A discriminator, trained with false labels from GFlowNet-generated trajectories and true labels from near-optimal trajectories, evaluates the quality of sampled trajectory set T = {τ 1 , τ 2 , . . . , τ h }. It assigns a quality score to each trajectory, which is then combined with the raw trajectory length R(τ ) to compute the final AGFN reward R(τ ). These rewards R(τ ), along with the source flow Z(θ generator ), forward probability P F (τ ; θ generator ), and backward probability P B (τ ) obtained from the GFlowNet, are used to compute the AGFN TB loss ℓ AG TB , defined as:
ℓ AG TB (T ; θ generator ) = 1 h h k=1 log Z(θ generator ) * P F (τ k ; θ generator ) R(τ k ) * P B (τ k ) 2 .(1)
For GFACS, the GFlowNet is used to generate a heuristic matrix η(G * , θ), which is subsequently transformed into a pheromone map to guide the ant colony optimization (ACO) in trajectory construction. Once the trajectories are generated, a local search is applied for refinement, followed by an energy reshaping step. The TB loss for GFACS, denoted ℓ GF TB , is then computed in a similar form to AGFN, as both approaches adopt the TB loss formulation to optimize their models.
this section cite: []

Section: Local-Global Optimization through Hybrid-Balance
While global optimization is essential for solving VRPs, local optimization is also important as it helps the model to capture fine-grained patterns, such as transitions between neighboring nodes. However, local information alone is insufficient for modeling global objective and constraints like total cost and capacity. To address this, we propose to unify both global and local objectives within a Hybrid-Balance GFlowNet framework. Specifically, we integrate the DB mechanism into the original TB framework of the GFACS and AGFN models to further enhance the modeling of local transitions, particularly the relationship between the current state s i t and the next state s i t+1 . As shown in Fig. 1, the model records relevant information at each step, including the current state's reward R(s i t ) and flow F (s i t ; θ), the next state's reward R(s i t+1 ) and flow F (s i t+1 ; θ), as well as the forward and backward transition probability P f (s i t+1 |s i t ; θ) and P b (s i t |s i t+1 ). Once a trajectory is completed, we apply a forward-looking technique [37] to compute the DB loss ℓ DB between two successive states as: ℓ DB (s i t , s i t+1 ; θ) = log P f (s i t+1 |s i t ; θ) • F (s i t ; θ) • exp( Ẽ(s i t+1 )) P b (s i t |s i t+1 ) • F (s i t+1 ; θ) • exp( Ẽ(s i t )) 2 . (2) Here, P f (s i t+1 |s i t ; θ) denotes the forward transition probability derived from the edge probability distribution η(G * , θ) in AGFN or the pheromone map in GFACS. The relationship between the trajectory-level forward probability P F (τ i ; θ) used in TB loss and the step-wise forward probability P f (s i t+1 |s i t ; θ) used in DB loss is given by:
P F (τ i ; θ) = m t=1 P f (s i t |s i t-1 ; θ).(3)
To ensure consistency with the trajectory-level backward probability P B (τ i ) used in TB loss, we design the step-wise backward probability P b (s i t |s i t+1 ) to reflect the structure of sub-trajectories within τ i . Specifically, we assume that each complete trajectory τ i consists of a multi-node subtrajectories and j single-node sub-trajectories, and parameter P b is accordingly determined by the varied transition structures.
this section cite: []

Section: Definition 1 (Trajectory Composition and Ordering Count).
We define A a as the set of a multinode trajectories, and J j as the set of j single-node trajectories. Together, these sequences are combined to form a complete trajectory τ i . Let B(A a , J j ) denote the number of distinct orderings of sub-trajectories in A a and J j that result in the same complete trajectory τ i .
We next present the following statement, which describes the recurrence relation for B(A a , J j ).
this section cite: []

Section: Statement 1 (Trajectory Orders' Count Recurrence).
The number of distinct trajectories composed of a multi-node trajectories and j single-node trajectories arranged in different orders, denoted by B(A a , J j ), satisfies the following recurrence relation:
B(A a , J j ) = 2a • B(A a-1 , J j ) + j • B(A a , J j-1 ).(4)
This recurrence arises from the backward destruction of CVRP trajectories, where we consider how B(A a , J j ) reached its predecessors. Suppose the current state corresponds to B(A a , J j ), where there are a remaining multi-node trajectories and j remaining single-node trajectories to be disconnected. There are two possible types of backward transitions from this state to reach its predecessor B(A a-1 , J j ) or B(A a , J j-1 ):
(1) Multi-node trajectory: If a multi-node trajectory is selected for backward destruction from the depot, either of its two nodes can serve as the immediate predecessor to the depot. Therefore, each of the a multi-node trajectories contributes two valid backward transitions, resulting in a total contribution of 2a • B(A a-1 , J j ), where the recursion proceeds with a -1 remaining multi-node trajectories and j unchanged single-node trajectories.
(2) Single-node trajectory: If a single-node trajectory is chosen, it contains only one node, which uniquely determines the depot's predecessor. Thus, each of the j single-node trajectories contributes one backward transition, resulting in j•B(A a , J j-1 ), where the recursion continues with a multi-node trajectories and j -1 single-node trajectories.
We combine both types of transitions to derive the recurrence relation as presented in Eq. 4. Subsequently, we deduce the closed-form expression of B(A a , J j ) from Eq. 4. The proof is provided in Appendix Sec. A.2, and the corresponding formulation is presented below:
B(A a , J j ) = (a + j)! • 2 a , for a, j ≥ 0.(5)
Physically, the term (a + j)! accounts for all possible orderings of the a + j sub-trajectories, where each of them is treated as an atomic step in the destruction process. Each multi-node trajectory has 2 possible directions for destruction, contributing an additional 2 a multiplicative factor. In contrast, single-node trajectories allow only one valid direction. Therefore, the total number of reward-equivalent permutations is the product of these two factors.
this section cite: []

Section: Statement 2 (TB Backward Probability).
We denote P B (τ i ) as the backward policy probability of a complete trajectory τ i in the GFlowNet framework under TB, formulated as:
P B (τ i ) = 1 (a + j)! • 2 a ,(6)
where the denominator reflects the total number of distinguishable trajectory permutations given a multi-node and j single-node trajectories to achieve complete trajectory τ i .
this section cite: []

Section: Statement 3 (DB Backward Probability).
We denote P b (s i t | s i t+1 ) as the probability of a single backward transition from state s i t+1 to its predecessor s i t , and under the DB formulation, the backward probability is defined conditionally:
P b (s i t |s i t+1 ) = 1 2a+j
if the current node is the depot, 1 otherwise.
This probability formulation originates from Eq. 4, which defines the total number of sub-trajectory backward destruction orderings. Physically, each multi-node trajectory offers two possible predecessor nodes for backward disconnection from the depot, thereby contributing the 2a term, while each single-node trajectory provides one such option, contributing the j term. The resulting probability 1 2a+j reflects a uniform selection over all valid backward transitions at the current decision step. In contrast, for all other nodes in the trajectory, only a single predecessor is feasible, and thus the backward transition becomes fully deterministic with probability 1.
Meanwhile, F (s i t ; θ) in Eq. 2 represents the flow of current state, and is derived from the node embedding q at state s i t , which is calculated as follow:
F (s i t ; θ) = 1 t x k ∈s i t (W 2 • ReLU(W 1 • q k + b 1 ) + b 2 ),(8)
where W 1 , W 2 , b 1 and b 2 are learnable parameters and ReLU [6] is the activation function. To handle the local objective associated with state transitions, we define the reward of the predecessor state s i t as zero. Consequently, the energy term Ẽ(s i t ) in Eq. 2 is also set to zero. The successor state s i t+1 , in contrast, receives a non-zero transition reward. Accordingly, the energy term Ẽ(s i t+1 ) represents the local reward signal, and its negative is defined as follow:
Ẽ(s i t ) = R(s i t ) - 1 h h k=1 R(s k t ).(9)
As training progresses, the quality of each trajectory steadily improves, resulting in smaller values of R(s i t ) as the generated routes become shorter. In Eq. 9, we compute the energy Ẽ(s i t ) for state s i t by subtracting the average reward of other trajectories at the same decision step. This formulation effectively captures the relative advantage of a given state compared to its peers, encouraging the model to assign higher energy to better-performing states. As the variance across rewards decreases during training, the energy values naturally increase.
Then, the DB loss of the completed trajectory τ i = {x i 0 , x i 1 , x i 2 , . . . , x i m } can be calculated as:
ℓ DB (τ i ; θ) = m-1 t=0 ℓ DB (s i t , s i t+1 ; θ),(10)
where ℓ DB (s i t , s i t+1 ; θ) is derived from Eq. 2. The overall loss for the Hybrid-Balance GFlowNet, denoted by ℓ HB (T ; θ), is computed by aggregating both the TB loss ℓ TB (τ ; θ) and the DB loss ℓ DB (τ ; θ) over all trajectories:
ℓ HB (T ; θ) = h i=1 ℓ HB (τ i ; θ) = h i=1 (ℓ TB (τ i ; θ) + ℓ DB (τ i ; θ)).(11)
This unified objective enables the model to simultaneously capture global trajectory-level structure and fine-grained local transitions, leading to more effective and robust optimization in VRPs. The design of Hybrid-Balance GFlowNet's backward policy reveals a key insight: as illustrated in Fig. 2, only the depot node retains flexibility in choosing among multiple predecessor candidates during trajectory destruction. This flexibility stems from the construction of sub-trajectories, each of which begins and ends at the depot. In contrast, for all customer nodes, the backward transition path is uniquely defined by the trajectory structure, i.e., once a customer node is reached, its predecessor is deterministically identified. This determinism also holds during forward trajectory construction.
this section cite: ['b5']

Section: Depot-Guided Inference
To leverage this structural characteristic, we propose a depot-guided inference mechanism defined as: the edge probability distribution η(G * , θ) in AGFN or the pheromone map in GFACS. Under this strategy, exploration through sampling is applied only at the depot, while customer nodes follow a deterministic, greedy policy.
x t+1 = x if the current node x t is depot, x * if the current node x t is customer,(12)
It is important to note that depot-guided inference is specifically designed for problems featuring a designated depot node, such as the CVRP. For problems lacking a depot or node-role differentiation, such as the TSP, we retain their original inference procedures, including hybrid decoding strategy [55] for AGFN and the ant clony search [22] for GFACS.
this section cite: ['b54', 'b21']

Section: Experiment
We conduct experiments to validate the effectiveness of the Hybrid-Balance GFlowNet (HBG) in enhancing two representative GFlowNet-based solvers, i.e., AGFN and GFACS, on CVRP. We first present comparison results, followed by ablation studies to analyze the contribution of individual components. Lastly, we extend the evaluation to other vehicle routing problem.
this section cite: []

Section: Dataset:
We adopt synthetic CVRP datasets following standard settings used in prior work [22,25,55,49]. Each instance features a single depot and multiple customers served by a vehicle with fixed capacity C. The depot and customer coordinates are sampled uniformly from the unit square
this section cite: ['b21', 'b24', 'b54', 'b48']

Section: Performance on Synthetic CVRP Instances
We compare HBG-enhanced models, i.e., HBG-AGFN and HBG-GFACS, with their original TBbased counterparts, AGFN [55] and GFACS [22]. AGFN constructs routes in an end-to-end manner, while GFACS searches for solutions by combining GFlowNet with ant colony optimization. We also include classical heuristics (LKH [19], ACO [3]) and learning-based baselines (POMO [25], GANCO [50], NeuOpt [31]) for comparison. All methods are trained on 100-node instances and evaluated on CVRP200, CVRP500, and CVRP1000 datasets, following AGFN and GFACS evaluation protocols. Additional experiments on the public benchmark CVRPLib are reported in Appendix B.1.
Table 1 shows that HBG consistently improves performance across all problem sizes for AGFN, GFACS, and GFACS with local search. The performance gains are significant, with gap reductions of up to 16.23%, 55.46%, and 8.33%, respectively. Improvements become more pronounced as instance size increases, indicating strong scalability. Inference incurs only minor overhead (0.01-0.04 seconds) due to temporary loading of flow parameters, which does not impact overall runtime or scalability. Compared to other heuristic and learning-based methods, HBG-AGFN and HBG-GFACS achieve competitive or superior solution quality across all scales. On CVRP200, both methods outperform ACO and NeuOpt. On CVRP500 and CVRP1000, they continue to generalize effectively, outperforming ACO, POMO, GANCO, and NeuOpt. These results highlight the robustness, efficiency, and strong generalization capabilities of the proposed HBG framework.
this section cite: ['b54', 'b21', 'b18', 'b2', 'b24', 'b49', 'b30']

Section: Ablation Study
Comparison of Component Contributions. We evaluate the contribution of each component in the HBG framework for both AGFN and GFACS. First, we incorporate the Hybrid-Balance (HB) module into the original models. Then, we add the depot-guided inference mechanism on top of the HB-enhanced variants. As shown in Table 2, each component contributes significantly to performance. Incorporating the HB module alone reduces the optimality gap by up to 15.07% in AGFN and 16.58% in GFACS. Adding depot-guided inference provides further gains, especially for larger instances. These results confirm that the HB module offers consistent improvements and depot-guided inference delivers additional benefits in depot-centric tasks.
this section cite: []

Section: Comparison of Balance Strategies.
To further validate the effectiveness of Hybrid Balance (HB), we conduct a comparison against Trajectory Balance (TB) and Detailed Balance (DB) under identical training settings on 100-node instances, evaluated on CVRP200, CVRP500, and CVRP1000. As shown in Table 3, the HB module consistently outperforms both TB and DB across all instance sizes for both AGFN and GFACS. Notably, HB achieves up to a 15.07% improvement over TB in AGFN and up to 16.58% in GFACS. These results highlight the superior effectiveness of Hybrid Balance as a unifying optimization strategy.
Depot-Guided Inference Variants. We assess four variants of the depot-guided inference strategy by applying either sampling or greedy decoding at the depot and customer nodes. Tests are conducted using both AGFN and GFACS on CVRP200, CVRP500, and CVRP1000. As shown in Table 4, the combination of sampling at the depot and greedy decoding at customers yields the best performance. This setting consistently outperforms all other variants, including depot greedy + customer sampling, depot greedy + customer greedy, and depot sampling + customer sampling. These results validate the effectness of our depot-guided inference mechanism.
this section cite: []

Section: Generalization to Other Vehicle Routing Problem
We further evaluate our framework on the Traveling Salesman Problem (TSP), a key VRP variant.
Baselines include GFlowNet-based solvers (AGFN [55], GFACS [22]), classical heuristics (LKH [19], ACO [3]), and learning-based models (POMO [25], GANCO [50], NeuOpt [31]). All models are trained on 100-node instances and evaluated on 200-, 500-, and 1,000-node settings. Since TSP lacks a depot node, depot-guided inference is not used. Table 5 shows that HBG-AGFN consistently outperforms AGFN, reducing the gap by up to 17.64%. HBG-GFACS also achieves
this section cite: ['b54', 'b21', 'b18', 'b2', 'b24', 'b49', 'b30']

Section: Conclusion
In this paper, we introduced the Hybrid-Balance GFlowNet (HBG) framework to enhance the performance of GFlowNet-based solvers for vehicle routing problems. HBG unifies Trajectory Balance and Detailed Balance in a principled and adaptive manner to jointly optimize local and global objectives. We also proposed a depot-guided inference strategy aligned with the Hybrid-Balance principle, specifically tailored for depot-centric problems. Extensive experiments on both CVRP and TSP benchmarks demonstrate that HBG significantly improves the performance of two representative GFlowNet-based solvers, i.e., AGFN and GFACS, showcasing improved solution quality, scalability, and generalization. A current limitation of HBG is its reliance on existing GFlowNet-based models, as its performance depends in part on the underlying solver, which might be inferior to others. In future work, we plan to integrate it with alternative stronger generative policies and solvers.
this section cite: []

Section: References
Ref_id:b0 Title: Efficiently solving very large-scale routing problems Year: (2019)
Ref_id:b1 Title: A tabu search algorithm for the vehicle routing problem Year: (1999)
Ref_id:b2 Title: Ant colony optimization techniques for the vehicle routing problem Year: (2004)
Ref_id:b3 Title: Flow network based generative models for non-iterative diverse candidate generation Year: (2021)
Ref_id:b4 Title: Learning generalizable models for vehicle routing problems via knowledge distillation Year: (2022)
Ref_id:b5 Title: Dynamic relu Year: (2020)
Ref_id:b6 Title: A multi-trip vehicle routing problem for small unmanned aircraft systems-based urban delivery Year: (2019)
Ref_id:b7 Title: Integrating distributed disassembly line balancing and vehicle routing problem in supply chain: Integer programming, constraint programming, and heuristic algorithms Year: (2023)
Ref_id:b8 Title: Human-in-the-loop causal discovery under latent confounding using ancestral gflownets Year: (2023)
Ref_id:b9 Title: Bayesian structure learning with generative flow networks Year: (2022)
Ref_id:b10 Title: The multi-echelon vehicle routing problem with cross docking in supply chain management Year: (2011)
Ref_id:b11 Title: Challenges and perspectives for the use of electric vehicles for last mile logistics of grocery e-commerce-findings from case studies in germany Year: (2021)
Ref_id:b12 Title: Invit: a generalizable routing problem solver with invariant nested view transformer Year: (2024)
Ref_id:b13 Title: Towards generalizable neural solvers for vehicle routing problems via ensemble with transferrable local policy Year: (2024)
Ref_id:b14 Title: Fuzzy green vehicle routing problem for designing a three echelons supply chain Year: (2020)
Ref_id:b15 Title: Generative flow networks for lead optimization in drug design (student abstract Year: (2025)
Ref_id:b16 Title: Concrete mathematics: a foundation for computer science Year: (1994)
Ref_id:b17 Title: The vehicle routing problem of intercity ride-sharing between two cities Year: (2022)
Ref_id:b18 Title: An effective implementation of the lin-kernighan traveling salesman heuristic Year: (2000)
Ref_id:b19 Title: Rethinking light decoder-based solvers for vehicle routing problems Year: (2025)
Ref_id:b20 Title: City vehicle routing problem (city vrp): A review Year: (2015)
Ref_id:b21 Title: Ant colony sampling with gflownets for combinatorial optimization Year: ()
Ref_id:b22 Title: Attention, learn to solve routing problems Year: ()
Ref_id:b23 Title: Rgfn: Synthesizable molecular generation using gflownets Year: (2024)
Ref_id:b24 Title: Pomo: Policy optimization with multiple optima for reinforcement learning Year: (2020)
Ref_id:b25 Title: Matrix encoding networks for neural combinatorial optimization Year: (2021)
Ref_id:b26 Title: Gfn-sr: Symbolic regression with generative flow networks Year: ()
Ref_id:b27 Title: Gflowcausal: Generative flow networks for causal discovery Year: (2022)
Ref_id:b28 Title: Cflownets: Continuous control with generative flow networks Year: ()
Ref_id:b29 Title: Research on optimization of vehicle routing problem for ride-sharing taxi Year: (2012)
Ref_id:b30 Title: Learning to search feasible and infeasible regions of routing problems with flexible neural k-opt Year: (2024)
Ref_id:b31 Title: Trajectory balance: Improved credit assignment in gflownets Year: (2022)
Ref_id:b32 Title: Robonet: A sample-efficient robot co-design generator Year: ()
Ref_id:b33 Title: Evaluating generalization in gflownets for molecule design Year: (2022)
Ref_id:b34 Title: Bayesian learning of causal structure and mechanisms with gflownets and variational bayes Year: (2022)
Ref_id:b35 Title: Metastrategy simulated annealing and tabu search algorithms for the vehicle routing problem Year: (1993)
Ref_id:b36 Title: Better training of gflownets with local credit and incomplete trajectories Year: (2023)
Ref_id:b37 Title: H-tsp: Hierarchically solving the large-scale traveling salesman problem Year: (2023)
Ref_id:b38 Title: Solution approaches for combining first-mile pickup and last-mile delivery in an e-commerce logistic network: A systematic literature review Year: (2021)
Ref_id:b39 Title: Generative flows on synthetic pathway for drug design Year: ()
Ref_id:b40 Title: Tacogfn: Target conditioned gflownet for structure-based drug design Year: (2023)
Ref_id:b41 Title: Streaming bayes gflownets Year: (2024)
Ref_id:b42 Title: Difusco: Graph-based diffusion solvers for combinatorial optimization Year: (2023)
Ref_id:b43 Title: Frontiers in service science: Ride matching for peer-to-peer ride sharing: A review and future directions Year: (2020)
Ref_id:b44 Title: Neuro-symbolic causal reasoning meets signaling game for emergent semantic communications Year: (2023)
Ref_id:b45 Title: Hybrid genetic search for the cvrp: Open-source implementation and swap* neighborhood Year: (2022)
Ref_id:b46 Title: An efficient diffusion-based non-autoregressive solver for traveling salesman problem Year: (2025)
Ref_id:b47 Title: A comprehensive survey on graph neural networks Year: (2020)
Ref_id:b48 Title: Neurolkh: Combining deep learning model with lin-kernighan-helsgaun heuristic for solving the traveling salesman problem Year: (2021)
Ref_id:b49 Title: Generative adversarial training for neural combinatorial optimization models Year: (2022)
Ref_id:b50 Title: Deepaco: Neural-enhanced ant systems for combinatorial optimization Year: (2023)
Ref_id:b51 Title: Glop: Learning global partition and local construction for solving large-scale routing problems in real-time Year: (2024)
Ref_id:b52 Title: Let the flows tell: Solving graph combinatorial problems with gflownets Year: (2023)
Ref_id:b53 Title: Forward and reverse logistics vehicle routing problems with time horizons in b2c e-commerce logistics Year: (2021)
Ref_id:b54 Title: Adversarial generative flow network for solving vehicle routing problems Year: (2025)
Ref_id:b55 Title: Urban logistics delivery route planning based on a single metro line Year: (2021)
Ref_id:b56 Title: Sampleefficient multi-objective molecular optimization with gflownets Year: (2023)
