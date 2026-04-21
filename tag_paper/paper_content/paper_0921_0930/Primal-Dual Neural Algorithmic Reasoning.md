Title: Primal-Dual Neural Algorithmic Reasoning
Abstract: Neural Algorithmic Reasoning (NAR) trains neural networks to simulate classical algorithms, enabling structured and interpretable reasoning over complex data. While prior research has predominantly focused on learning exact algorithms for polynomial-time-solvable problems, extending NAR to harder problems remains an open challenge. In this work, we introduce a general NAR framework grounded in the primal-dual paradigm, a classical method for designing efficient approximation algorithms. By leveraging a bipartite representation between primal and dual variables, we establish an alignment between primal-dual algorithms and Graph Neural Networks. Furthermore, we incorporate optimal solutions from small instances to greatly enhance the model's reasoning capabilities. Our empirical results demonstrate that our model not only simulates but also outperforms approximation algorithms for multiple tasks, exhibiting robust generalization to larger and out-of-distribution graphs. Moreover, we highlight the framework's practical utility by integrating it with commercial solvers and applying it to real-world datasets.

Section: Introduction
Understanding the algorithmic reasoning ability of neural networks is crucial for quantifying their expressivity and practical deployment (Łukasz Kaiser & Sutskever, 2016;Zhou et al., 2022;Sanford et al., 2024;de Luca & Fountoulakis, 2024). However, end-to-end supervised learning often struggles with generalization for algorithmic tasks. Neural Algorithmic Reasoning (NAR) (Veličković & Blundell, 2021) addresses this challenge by training neural networks to mimic operations of classical algorithms, such as Bellman-Ford for shortest-path problems (Veličković et al., 2020). By aligning a model's architecture with an algorithm's step-wise operations, NAR enhances generalization and sample efficiency (Xu et al., 2020;2021).
Moreover, NAR addresses a fundamental bottleneck of classical algorithms: while they guarantee correctness and interoperability, they require extensive feature engineering to compress real-world data into scalar values. By embedding algorithmic knowledge into neural models, NAR enables direct handling of structured real-world data (Deac et al., 2021;He et al., 2022;Beurer-Kellner et al., 2022;Numeroso et al., 2023). For instance, a model pre-trained with Bellman-Ford knowledge can tackle real-world transportation problems, while integrating domain-specific features such as weather conditions and traffic patterns.
Despite the success of NAR in simulating polynomial-time algorithms (Ibarz et al., 2022;Rodionov & Prokhorenkova, 2024), in particular the 30 algorithms (e.g., sorting, search, graph) from the CLRS-30 benchmark (Veličković et al., 2022) and other benchmarks (Minder et al., 2023;Markeeva et al., 2024), its extension to NP-hard problems remains an open challenge due to the difficulty in reliably generating ground-truth samplers (Veličković et al., 2022). This limitation creates a significant gap when applying NAR to real-world problems, many of which are inherently NP-hard. Addressing this gap is crucial, as the motivation behind NAR is to enable the transfer of algorithmic knowledge to tackle complex, real-world datasets effectively.
We build upon the line of NAR research that simulates classical algorithms with Graph Neural Networks (GNNs) (Xu et al., 2020;Veličković et al., 2020;Bevilacqua et al., 2023;Rodionov & Prokhorenkova, 2023;2024;Georgiev et al., 2024). Specifically, we focus on advancing NAR into the underexplored NP-hard domain (Cappart et al., 2022;Georgiev et al., 2023c) by training GNNs to replicate approximation algorithms. Additionally, we leverage findings that NAR benefits from multi-task learning (Xhonneux et al., 2021;Ibarz et al., 2022) when trained on multiple algorithms. In particular, we harness the concept of duality, which frames problems through complementary primal and dual perspectives, a principle that has been shown to enhance NAR (Numeroso et al., 2023) in the context of Ford-Fulkerson via the max-flow min-cut theorem.
Our key contributions are as follows:
• We propose Primal-Dual Neural Algorithmic Reasoning (PDNAR) -a general NAR framework based on the primal-dual paradigm to learn algorithms for both polynomial-time-solvable and NP-hard tasks.
• We establish a novel alignment between primal-dual algorithms and GNNs using a bipartite representation between primal and dual variables.
• We provide theoretical proofs showing that PDNAR can exactly replicate the classical primal-dual algorithm and inherit performance guarantees.
• We go beyond algorithm replication by incorporating optimal supervision signals from small problem instances, enabling it to outperform the algorithm it is trained on 1 .
To the best of our knowledge, this is the first NAR method designed to achieve this.
• We empirically validate PDNAR on synthetic algorithmic datasets, demonstrating strong generalization to larger problem instances and OOD distributions.
• We showcase PDNAR's practical utility by applying it to real-world datasets and commercial solvers.
Our code can be found at https://github.com/  dransyhe/pdnar.
this section cite: ['b77', 'b59', 'b11', 'b63', 'b65', 'b74', 'b13', 'b34', 'b4', 'b54', 'b37', 'b57', 'b62', 'b52', 'b51', 'b62', 'b74', 'b65', 'b5', 'b56', 'b23', 'b8', 'b69', 'b37', 'b54']

Section: Related works
The most closely related work extending NAR to NP-hard domain is by Georgiev et al. (2023c), which pretrains a GNN on algorithms for polynomial-time-solvable problems (e.g., Prim's algorithm for MST) and using transfer learning to solve NP-hard problems (e.g., TSP). However, this approach lacks generality since each NP-hard problem requires carefully selecting a related polynomial-time-solvable problem and algorithm. However, our method is inherently general, directly learning approximation algorithms for NPhard tasks and incorporating optimal solutions from small instances to enhance the model's reasoning ability.
Only one prior work has explored the role of duality in NAR. Numeroso et al. (2023) trained a GNN to replicate the Ford-Fulkerson algorithm for the max-flow (primal) and min-cut (dual) problems. By leveraging supervision signals from both problems, their model benefits from multi-task learning to achieve better performance. However, their architecture is highly specialized for Ford-Fulkerson, and both problems are polynomial-time solvable and thus benefit from strong duality. In contrast, our framework is general and applicable 1 Many classical primal-dual algorithms achieve tight worstcase approximation bounds under the Unique Games Conjecture (e.g., Khot & Regev, 2008). Although worst-case limits are unlikely to be exceeded, our empirical results demonstrate improved performance in a setting beyond the worst-case.
to a range of exact and approximation algorithms.
Another line of work, Neural Combinatorial Optimization (NCO), develops neural approaches for solving NP-hard problems. However, NCO and NAR differ fundamentally. NCO focuses on learning task-specific heuristics or end-toend optimization methods for finding (near-)optimal solutions (Dai et al., 2018;Li et al., 2018;Joshi et al., 2019;Karalias & Loukas, 2021;Wang & Li, 2023;Wenkel et al., 2024;Yau et al., 2024), whereas NAR designs neural architectures to simulate and generalize algorithmic behavior across problems, allowing them to embed algorithmic knowledge in real-world settings. While our primary focus is not NCO, our framework can serve as an algorithmically informed GNN to enhance data efficiency and generalization in supervised learning. See Appendix H for details.
this section cite: ['b54', 'b44', 'b10', 'b50', 'b40', 'b43', 'b66', 'b67', 'b76']

Section: Background

this section cite: []

Section: Problem statement: algorithmic reasoning
The task of algorithmic reasoning is to learn a model M that approximates the behavior of a target algorithm A : X → Y, where inputs x ∈ X map to outputs y = A(x) ∈ Y. Unlike standard function approximation, the goal is to model the sequence of intermediate states {S (t) } T t=0 generated by A during execution with trajectory M (t) (x) = S (t) .
In particular, we focus on simulating the primal-dual algorithm, which provides a general framework to design algorithms for both polynomial-time-solvable and NP-hard problems. We prioritize the latter due to the limited NAR research in the NP-hard domain.
this section cite: []

Section: Algorithmic tasks
We study the following algorithmic problems.
Definition 3.1 (Minimum Vertex Cover). Let G = (V, E)
be a graph where V are vertices and E are edges, and each vertex v ∈ V has a non-negative weight w v ∈ R + . A vertex cover for G is a subset C ⊆ V of the vertices such that for each edge (v, u) ∈ E, either v ∈ C, u ∈ C, or both. The objective is to minimize the total vertex weight v∈C w v .
Definition 3.2 (Minimum Set Cover). Given a ground set U and a family of sets C ⊆ 2 U with non-negative weights w S ∈ R + for all sets S ∈ C, a set cover is a subfamily C ′ ⊆ C such that ∪ S∈C ′ S = ∪ S∈C S. The objective is to minimize the total weight S∈C ′ w S .
Definition 3.3 (Minimum Hitting Set). Given a ground set E of elements e with non-negative weights w e ∈ R + and a collection T of subsets T ⊆ E, a hitting set is a subset A ⊆ E such that A ∩ T ̸ = ∅ for every T ∈ T . The objective is to minimize the total weight e∈A w e .
Figure 1. Let xe ∈ {0, 1} for each element e ∈ E be the variables, where xe = 1 represents that element e is included in the hitting set A, the IP formulation of MHS is shown in (a). Let xe ∈ R + be the primal variables, the LP relaxation of MHS is shown in (b). Let yT ∈ R + for each set T ∈ T be the dual variables, the dual problem of the LP relaxation of MHS is shown in (c).
this section cite: []

Section: Algorithm 1 General primal-dual approximation algorithm
Input: (T , E, w): a ground set E with weights w, a family of subsets T ⊆
2 E A ← ∅; for all e ∈ E, r e ← w e while ∃T : A ∩ T = ∅ do V ← {T : A ∩ T = ∅} repeat for T ∈ V do δ T ← min e∈T re |{T ′ :e∈T ′ }| for e ∈ E \ A do r e ← r e -T :e∈T δ T until ∃e / ∈ A : r e = 0 A ← A ∪ {e : r e = 0} end while Output: A Uniform increase (optional): (6.1): ∆ ← min T ∈V δ T (6.2): for e ∈ E \ A do r e ← r e -|{T : e ∈ T }|∆
The minimum vertex cover problem is a foundational NPhard problem with wide-reaching applications, where its dual problem is the well-studied maximum edge-packing problem. This primal-dual pair inspired a famous 2approximation algorithm proposed by Hochbaum (1982) and many follow-up works. The minimum set cover problem is a generalization of vertex cover to hypergraphs, providing a more complex structural setting to evaluate algorithmic reasoning. Lastly, the hitting set is equivalent to the set cover problem, but its formulation more naturally extends to a wide range of problems, including vertex cover, Steiner tree, feedback vertex set, and many more (Goemans & Williamson, 1996).
this section cite: ['b35', 'b28']

Section: A general primal-dual approximation algorithm
We now illustrate a general primal-dual approximation algorithm using the Minimum Hitting Set (MHS) problem as a concrete example. MHS can be formulated as an integer program (IP) as shown in Figure 1(a), where variables are restricted to integer values. The linear programming (LP) relaxation relaxes the integer constraints and allows variables to take continuous values, making the problem more tractable. This is illustrated in Figure 1(b).
Every LP formulation has a dual version. For MHS, Figure 1(b) is the primal and Figure 1(c) is the dual. More gener-ally, the dual of an LP min x≥0 {c ⊤ x : Ax ≥ b} is defined as max y≥0 {y ⊤ b : A T y ≤ c}. The weak duality principal states that any feasible solution to the primal problem has a larger objective value than any feasible solution to the dual problem. Based on this principle, the primal-dual framework iteratively updates both the primal and dual solutions, closing their gap and ensuring they improve in tandem.
Based on the primal-dual framework, an α-approximation algorithm (Bar-Yehuda & Even, 1981;Hochbaum, 1982;Goemans & Williamson, 1996;Khuller et al., 1994) for the general hitting set problem was developed, where α is the maximal cardinality of the subsets. The pseudocode of the algorithm is shown in Algorithm 1. Given a hitting set problem (T , E, w), the algorithm progresses over a series of rounds. At each round, the algorithm increases some of the dual variables y T until a constraint T :e∈T y T ≤ w e becomes equality, at which point the element e is added to the hitting set A. Although the algorithm does not explicitly define the dual variables y T , it can be interpreted as gradually increasing the dual variables by an amount δ T in each round, as shown in Line 1. This is implemented by defining a residual weight r e = w e -T :e∈T y T , which is defined in terms of the step sizes δ T , as shown in Line 1. Once r e = 0 for some e ̸ ∈ A (i.e. the constraint becomes tight), e is added to the hitting set A (Lines 1 and 1). This process is repeated until A is a valid hitting set (Line 1).
A general framework This algorithm can be reformulated to recover many classical (exact or approximation) algorithms (Goemans & Williamson, 1996). For example, vertex cover can be seen as a hitting set problem, where element e ∈ E corresponds to vertex v ∈ V , and subset T ∈ T corresponds to an edge that connects two vertices. This allows a direct adaptation of Algorithm 1 to solve the vertex cover problem. Moreover, Khuller et al. (1994) propose a sublinear-time vertex cover approximation algorithm which is a simple generalization of Algorithm 1 (see Appendix A). They relax the dual constraint using a parameter ϵ > 0, such that a vertex e is included in the cover if r e ≤ ϵw e , instead of r e = 0. This leads to a 2/(1 -ϵ)-approximation algorithm with a runtime of O(ln 2 |T | ln 1 ϵ ). Since set cover extends vertex cover to hypergraphs, this algorithm can be adapted into an r/(1 -ϵ)-approximation algorithm for set cover, where r is the maximal cardinality of the sets. We will later empirically show how well our framework simulates these algorithms. Moreover, the generality of the primal-dual framework is not limited by approximation algorithms for NP-hard tasks: it also recovers exact algorithms for some polynomial-time-solvable problems, such as Kruskal's minimum spanning tree algorithm (Kruskal, 1956).
Uniform increase of dual variables Some problems benefit from simultaneously increasing all dual variables δ T at the same rate (Agrawal et al., 1995;Goemans & Williamson, 1995). An example is how Kruskal's algorithm greedily selects the minimum-cost edge that connects two distinct components. This corresponds to increasing the dual variables for all connected components simultaneously until there is an edge whose dual constraint becomes tight. The incorporation of this uniform increase rule is illustrated from Line 6.1 and Line 6.2 in Algorithm 1. This rule provides a more balanced approach that attends to all dual variables, allowing the framework to adapt to a broader range of algorithms.
this section cite: ['b2', 'b35', 'b28', 'b45', 'b28', 'b45', 'b47', 'b0', 'b27']

Section: Primal-Dual Neural Algorithmic Reasoning (PDNAR)
We now present our framework of using a GNN to simulate the general primal-dual approximation algorithm, representing the primal-dual variables as two sides in a bipartite graph (Section 4.1). We also show how the uniform increase rule can be incorporated with a virtual node that connects to all dual nodes in the bipartite graph (Section 4.2). Furthermore, we explain how we use optimal solutions from integer programming solvers as additional training signals (Section 4.4), and later show how it allows the PDNAR to surpass the performance of the approximation algorithm.
this section cite: []

Section: Architecture
We adopt the encoder-processor-decoder framework (Hamrick et al., 2018) from the neural algorithmic reasoning blueprint (Veličković & Blundell, 2021). In this framework, the processor is typically a message-passing GNN (Gilmer et al., 2017), operating in a latent space to simulate a single step of algorithmic execution. The encoder transforms the input value (e.g., element weight) into this latent space, while the decoder reconstructs the final prediction from it (e.g., whether to include the element in the solution). We continue using Algorithm 1 for MHS as an example.
this section cite: ['b32', 'b63', 'b26']

Section: Bipartite graph construction
Given a hitting set problem (T , E, w), we represent it as a bipartite graph with elements e ∈ E (primal) on the left-hand side (LHS) and sets T ∈ T (dual) on the right-hand side (RHS), as illustrated in Figure 2. An edge connects an element e and a set T if e ∈ T . Let N (e) denote the set of neighbors of node e. As outlined in Algorithm 1, the algorithm incrementally adds elements e to the hitting set A. When an element is added, we remove node e by masking it out, along with its neighboring sets T ∈ N (e), which are now hit. Consequently, the violation set V = {T | A ∩ T = ∅} consists of the remaining sets T still in the graph. Once A becomes a valid hitting set, the violation set V is empty. Next, at each timestep t, we let r (t) e denote the residual weight of element e and d
e denote its current node degree. Therefore, the initial residual weight r (0) e is defined as its cost w e , and the initial degree d e , respectively, for each element e ∈ E. These encoders transform all features into a high-dimensional latent space for the processor:
h (t) e = f r (r (t-1) e ), h(t)
de = f d (d (t-1) e ).
Processor The processor is a message-passing GNN applied to the bipartite graph. A general message-passing framework (Gilmer et al., 2017) comprises a message function ψ θ and an update function ϕ θ . The node feature h
(t) v of node v is transformed via h (t) v = ϕ θ h (t) v , u∈N (v) ψ θ (h (t) u ) ,
where ψ θ and ϕ θ are usually shallow MLPs and is a permutation invariant function, such as sum or max. We now demonstrate how this message-passing framework is applied to the bipartite graph to simulate the primal-dual approximation algorithm.
Step (1): This step corresponds to Line 1 of Algorithm 1, where increment δ (t)
T for each set T ∈ V is computed. Let h (t)
T be the hidden representation of δ (t) T . We aggregate messages from its connected elements e ∈ N (T ) using a message function g e with a min aggregation operation:
h (t) T = min e∈N (T ) g e (h (t) e , h (t)
de ).
this section cite: ['b26']

Section: Step (2): This step corresponds to Line 1 of Algorithm 1, where residual weight r
(t) e for each element e ∈ E \ A is computed. Therefore, the dual variable update h (t)
T is passed back to its connected elements e using a sum aggregation and an update function g u :
h (t) e = g u h (t) e , T ∈N (e) h (t) T .
Decoder At each timestep t, Algorithm 1 computes three types of intermediate quantities: (1) whether to include an element e in the hitting set, represented by x
e ∈ {0, 1}, (2) the residual weights of an element r (t) e , and (3) the increment to the dual variable δ (t) T . We utilize separate MLP decoders, q x , q r , and q δ , to compute each of these quantities:
x(t) e = q x (h (t) e ), r(t) e = q r (h (t) e ), δ(t) T = q δ (h (t) T ).
Training Given the recurrent nature of our architecture, we apply noisy teacher forcing (Veličković et al., 2022) with a probability of 0.5 to determine whether to use hints-ground-truth values for intermediate quantities above-as inputs for the next timestep. Otherwise, the model's prediction from the previous timestep is passed on. This approach allows the model to follow its recurrent flow while reducing the risk of error propagation. The recurrent model is repeated for a maximum of |E| timesteps or terminates early when the solution becomes a valid hitting set. The loss function is defined as
L (t) algo = L BCE (x (t) e , x (t) e ) + L MSE (r (t) e , r (t) e ) + L MSE ( δ(t) T , δ(t)
T ) and averaged across timesteps. During test time, if the model output has not produced a valid hitting set, we greedily add the element e with the highest r e /d e value to the solution. Empirically, we find our model rarely requires it.
this section cite: ['b62']

Section: Uniform increase of dual variables
The uniform increase rule requires global communication among all dual variables. To achieve this, we introduce a virtual node z that connects to every set T ∈ T , as shown in Figure 2. Below, we describe how Step (2) in the processor is adjusted to accommodate this modification.
• Step (2.1): The virtual node aggregates all messages from the dual variables h
(t)
T via a min aggregation, corresponding to Line 6.1 via h
(t) z = min T ∈T h (t)
T .
• Step (2.2): The global information is passed back to dual variables with temporary h ′(t) T = h (t) z , and then to the primal variables h (t)  e with an update function g u and a sum aggregation. This corresponds to Line 6.2 via h
(t) e = g u (h (t) e , T ∈N (e) h ′(t) T ).
The intermediate quantity ∆ (t) is also given by Algorithm 1.
We use an additional decoder
q ∆ to predict ∆(t) = q ∆ (h (t) z ) and add L MSE ( ∆(t) , ∆ (t) ) to the total loss L (t) algo .
this section cite: []

Section: Theoretical justification
Algorithmic alignment is critical for generalization in NAR (Xu et al., 2020;2021). We show that PDNAR can exactly replicate the behavior of Algorithm 1.
Theorem 4.1. Given a hitting set problem (T , E, w), let A(T , E, w) be the solution produced by Algorithm 1, which terminates after K timesteps. There exists a parameter configuration Θ for a PDNAR model M Θ such that, at timestep K, the model output satisfies M
(K) Θ (T , E, w) = A(T , E, w). Furthermore, let (x (t) , r (t) , δ (t) , ∆ (t)
) be the intermediate quantities computed by Algorithm 1 at each timestep t. Then, the PDNAR model satisfies M
(t) Θ (T , E, w) = (x (t) , r (t) , δ (t) , ∆ (t)
), where ∆ (t) is omitted if the uniform increase rule is not applied.
In our proof of Theorem 4.1 (Appendix B), we show that this can be achieved with a PDNAR model using 8 layers. Therefore, PDNAR inherits the approximation ratio and convergence guarantees of the primal-dual approximation algorithm it learns, as described in Corollary 4.2.
Corollary 4.2. There exists a parameterization of the PDNAR model M Θ with 8 layers such that, after K iterations, M This guarantee also extends to the 2/(1 -ϵ)-approximation algorithm (Khuller et al., 1994) for MVC (and thus MSC).
Corollary 4.3. Given ϵ ∈ (0, 1), there exists a parameterization of the PDNAR model M Θ with 8 layers such that, after K iterations, M (K) Θ (V, E, w) yields an 2/(1 -ϵ)approximation to the optimal solution of the MVC problem, where K = O(log m log(1/ϵ)) and m = |E|.
this section cite: ['b74', 'b45']

Section: Use of optimal solutions from solvers
We can compute optimal solutions using IP solvers for small problem instances. We use the default IP solver in scipy based on HiGHS (Schwendinger & Schumacher, 2023;Huangfu & Hall, 2018). These optimal solutions are used as additional training signals to guide the model toward better outcomes. However, unlike the primal-dual algorithm, which provides intermediate steps, IP solvers only produce the final optimal solution. Therefore, the corresponding loss is defined as
L optm = L BCE (x K e , x optm e
), where K is the final timestep. The overall loss is then the sum of the intermediate losses from the primal-dual algorithm and the optimal solution loss, given by L = 1 K K t=1 L (t) algo + L optm . The motivation stems from the fact that IP solvers are computationally expensive, especially for larger problem instances. By training PDNAR using optimal solutions from IP solvers on smaller problem instances-allowing it to exceed the performance of the approximation algorithm-we can leverage its generalization ability to create a cost-efficient, highperformance model for much larger problems.
this section cite: ['b60', 'b36']

Section: Experiments
Dataset distributions and hyperparameter details of all experiments are provided in Appendices C, D, and E.
this section cite: []

Section: Synthetic algorithmic datasets
Dataset We evaluate PDNAR's ability to simulate the approximation algorithms for the three NP-hard algorithmic problems described in Section 3.2. The training dataset includes 1000 random graphs of size 16 for each task. We use Barabási-Albert graphs for vertex cover. For set cover and hitting set, we generate Barabási-Albert bipartite graphs with a preferential attachment parameter b = 5. Each test set consists of 100 graphs and is repeated for 10 seeds.
Baselines We compare with three types of baselines. (i) GNNs for end-to-end node classification trained with optimal labels: GIN (Xu et al., 2019) and GAT (Veličković et al., 2018). (ii) NAR models to simulate the approximation algorithm: NAR (MPNN) (Veličković et al., 2020) and the more expressive TripletMPNN (Ibarz et al., 2022) at the cost of training efficiency. Both models do not use the bipartite representation, and therefore are trained only on intermediate states of primal variables. (iii) Variations of PDNAR: No algo (trained without intermediate supervision from the algorithm) and no optm (trained without optimal solutions). Additionally, PDNAR's aggregation strategy is specifically tailored to align with the algorithm's structure. We test alternative mean and max aggregation methods.
Table 1 shows PDNAR achieves the best performance. By combining losses from intermediate steps of the approximation algorithm and optimal solutions from IP solvers, PDNAR yields the lowest ratios across all test cases. Moreover, PDNAR's performance remains stable across different graph sizes, indicating strong generalization to larger problems. Comparisons with other baselines (such as "No optm") show that incorporating supervision from optimal solutions improves the quality of final predictions, allowing PDNAR to outperform the primal-dual algorithm it was designed to simulate. In contrast, training without intermediate steps (such as "No algo") leads to a significant drop in generalization, highlighting that the primal-dual algorithm provides critical reasoning capabilities beyond merely learning optimal solution patterns.
Our model is a general NAR framework for a wide range of algorithms that uses the primal-dual paradigm. In Table 1, we see PDNAR is effective across all three tasks. For vertex cover, we demonstrate its applicability to graph-based algorithms. In set cover, we extend the framework to hypergraphs using a bipartite structure, highlighting its flexibility beyond traditional graph settings. Additionally, the uniform increase rule proves effective for hitting set, which can be instantiated to a range of exact and approximation algorithms. Notably, these results suggest that our approach may achieve even better performance when trained on more challenging instances where the optimal solutions significantly outperform those of approximation algorithms.
this section cite: ['b73', 'b64', 'b65', 'b37']

Section: Size and OOD generalization
Dataset We evaluate the model's generalization performance on larger and OOD graph families, using the same training set as before. For the vertex cover problem, we generate three OOD test sets comprising Erdős-Rényi (E-R), Star, Lobster, and 3-connected planar (3-Con) graphs. These graph types pose unique challenges for the vertex cover problem due to their distinct structural properties. For set cover and hitting set, we vary the preferential attachment parameter b to 3 and 8 to generate OOD bipartite graphs.
Table 1 demonstrates that PDNAR, trained on 16-node graphs, scales robustly to larger graphs of up to 1024 nodes for MVC. We highlight that prior NAR research typically evaluates up to 128 nodes. Furthermore, Table 2 shows PDNAR's ability to generalize to graphs from OOD families. Notably, these graph types can exhibit significantly different optimal sizes: Erdős-Rényi graphs require an average of 80% of nodes, while Star graphs need only 15% (see Table 5), highlighting the strong generalization ability of PDNAR. Obtaining optimal solutions for small instances is fast and efficient, even for hard problems, particularly with integer programming solvers like Gurobi (Gurobi Optimization, LLC, 2024). However, the computational complexity increases exponentially as the problem size grows. This underscores the key strength of PDNAR -we can train it efficiently using small problem instances and apply it to much larger, unseen problems that are computationally expensive to solve.
this section cite: []

Section: Real-world datasets
We now present a practical use case of our model. One of the key strengths of NAR is its ability to embed algorithmic knowledge in neural networks to tackle real-world challenges. Previous works applied NAR to reinforcement learning (Deac et al., 2021), brain vessels (Numeroso et al., 2023), and computer network configuration (Beurer-Kellner et al., 2022). Traditional algorithms designed for specific problems face a significant limitation: they cannot be directly applied to real-world data without substantial preprocessing. Indeed, real-world graphs generally consist of highdimensional features rather than simple scalar weights-the standard input for most traditional algorithms. As a result, applying traditional algorithms would necessitate extensive feature engineering to derive scalar weights, which may lead to the loss of crucial information embedded in the high-dimensional features.
In contrast, PDNAR overcomes this limitation by integrating a feature encoder that learns to estimate vertex weights directly from raw data, eliminating the need for manual preprocessing. We showcase such an application using the Airports datasets (Brazil, Europe, USA) (Ribeiro et al., 2017). In these graphs, the nodes represent airports, and the edges represent commercial flight routes. The goal is to predict the activity level of each airport, where vertex cover solutions can be highly valuable in predicting node influence.
this section cite: ['b13', 'b54', 'b4', 'b55']

Section: Architecture and baselines
We follow similar evaluation settings as Numeroso et al. (2023). We use three base models to perform node classification: GCN (Kipf & Welling, 2017), GAT (Veličković et al., 2018), and GraphSAGE (Hamilton et al., 2017). Then, we use PDNAR to produce embeddings to be concatenated with the base model's outputs before a final classification layer. We use a pretrained PDNAR on B-A graphs of size 16, keeping only the processor and degree encoder. We train a new encoder that learns to map new node features into the shared latent space of the processor, enabling it to replicate the vertex cover problem-solving behavior on airport data. The pretrained components are frozen, and a single message-passing step is performed. We compare PDNAR's embeddings against several baselines: Node2Vec (Grover & Leskovec, 2016) and a degree encoder. The latter helps to show that PDNAR captures more complex information beyond node degrees.
We also compare with two positional encodings: LapPE (Dwivedi et al., 2022a) and RWPE (Dwivedi et al., 2022b).
Table 3 shows that PDNAR achieves significant improvements in all datasets. Note that Node2Vec has an additional advantage by directly training on the graphs to produce the embeddings. PDNAR's superior performance over the degree encoder also indicates that it captures more complex information by integrating both learned node weights from features and degree information to replicate vertex cover. This showcases the important practical value of NAR in embedding algorithmic knowledge in real-world applications, overcoming the bottleneck of traditional algorithms.
this section cite: ['b54', 'b46', 'b64', 'b31', 'b29']

Section: Commercial optimization solvers
Another practical use case for PDNAR is to warm start large-scale commercial solvers, such as Gurobi (Gurobi Optimization, LLC, 2024), by initializing variables with its predictions. The motivation is that providing a starting point closer to the optimal solution can lead to faster solving times and improved efficiency.
Dataset We evaluate the vertex cover problem by comparing Gurobi's default initialization to warm starts using solutions from the primal-dual algorithm and our model.
Trained on 1000 B-A graphs of size 16, the model generates solutions for random larger B-A graphs, with 100 graphs per size. Gurobi's default parameters are used, with a thread count of 1 and a 1-hour time limit. Metrics We report the average solving times of all cases when optimal solutions are found (Solve time). We also measure the average computation time to generate the warmstart solutions using the primal-dual algorithm and the model per graph (Compute time). Results are recorded in seconds and averaged across 5 seeds. Table 4 shows that PDNAR outperforms both the default initialization and the approximation algorithm in all cases, achieving the fastest mean solving time. The improvement from using the model over the algorithm is greater than that of the algorithm over no warm start. Additionally, the model's inference time is nearly 10 times faster than the approximation algorithm's computation time. This demonstrates a practical use case: by simulating an approximation algorithm and leveraging optimal solutions on small instances, the model generates high-quality solutions for larger problems, improving efficiency for large-scale commercial solvers, such as Gurobi.
this section cite: []

Section: Conclusions
We propose a general NAR framework using the primaldual paradigm. Our approach can simulate approximation algorithms for NP-hard problems, greatly extending NAR's capability beyond the polynomial-time-solvable domain. Furthermore, we leverage both the intermediate states generated by the primal-dual algorithm and optimal solutions obtained from integer programming solvers on small problem instances, which can be obtained efficiently. While intermediate algorithmic steps provides a foundation for reasoning, incorporating optimal solutions enables the model to surpass the algorithm's performance. Empirical results demonstrate that our framework is effective and robust, showing strong generalization to larger problems and OOD distributions. Additionally, we present two practical applications: generating algorithmically informed embeddings for real-world datasets and warm-starting commercial solvers.
this section cite: []

Section: Limitation and future work
The primal-dual framework underlies many algorithmic problems, making our approach broadly applicable. Our architecture is currently designed for the hitting set problem, which can be reformulated as many other algorithmic problems. However, problems that do not reduce to hitting set may require architectural extensions. For instance, the uncapacitated facility location problem involves two types of dual variables that need specialized handling. Other advanced techniques (Williamson & Shmoys, 2011), such as selectively updating dual variables to improve scalability, can further extend our framework to a broader class of approximation algorithms. Future work could also explore the multi-task learning setting (Ibarz et al., 2022) in PDNAR by training on multiple approximation algorithms simultaneously.
this section cite: ['b68', 'b37']

Section: Impact Statement
This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.
this section cite: []

Section: A.3. Primal-dual pair: Set cover and element packing
The minimum set cover (MSC) problem is a generalization of MVC to hypergraphs. Similar to the edge packing, the element packing is the dual of the LP relaxation of the set cover problem.
Definition A.3 (Minimum Set Cover). Given a universe U and a family of sets C ⊆ 2 U with non-negative weights w : C → R + , a set cover is a subfamily C ′ ⊆ C such that ∪ S∈C ′ S = ∪ S∈C S. The objective is to minimize the total weight S∈C ′ w(S). Definition A.4 (Maximum Element Packing). An element-packing is an assignment p : U → R + of non-negative weights to the elements e ∈ U, such that for any set S ∈ C, the total weight e∈S p(e) is at most w(S). The objective is to maximize the total element weight e∈U p(e).
Let x S ∈ {0, 1} indicate whether each set S ∈ C is in the cover C ′ , and y e ∈ R + be the non-negative weight assigned to each element e ∈ U. The two problems can be formulated as:
Min S∈C w(S)x S sub. to S:e∈S x S ≥ 1, ∀e ∈ U x S ∈ {0, 1}, ∀S ∈ C Max e∈U y e sub. to e∈S y e ≤ w(S), ∀S ∈ C y e ≥ 0, ∀e ∈ U.
this section cite: []

Section: A.4. Pseudocode of MSC Algorithm
The above algorithm can be extended for set cover as an r/(1 -ϵ)-approximation algorithm, where r is the maximal cardinality of sets. The following pseudocode is the adapted approximation algorithm to solve vertex cover on a hypergraph.
Algorithm 3 COVER(G = (V, E), w, ϵ) 1: for v ∈ V do 2: w p (v) ← w(v); E p (v) ← E(v); d p (v) ← |E(v
)| 3: end for 4: while edges remain do 5: for each remaining edge e do 6: δ(e) ← min v∈e wp(v) dp(v) 7: end for 8:
for each remaining vertex v do 9:
w p (v) ← w p (v) -e∈Ep(v) δ(e) 10: if w p (v) ≤ ϵ • w(v)
then 11: delete v and its incident edges; update E p (•) and d p (•) 12: end if 13: end for 14: end while 15: Output: the set of deleted vertices
this section cite: []

Section: B. Proof of Theorem 4.1
Given a hitting set problem (T , E, w), we construct a bipartite graph B as explained in Section 4.1. Let B (t) denote the bipartite graph at timestep t, then B (0) = B. If we remove a node e and its incident edges from the graph when an element e is included in the hitting set at timestep t, an action denoted by x (t) e = 1, then the bipartite graph B (t) changes accordingly. Therefore, the degrees of primal nodes d (t) can be computed via
d (t) = f (T , E, max t ′ ∈[0,...,t] (x (t ′ )
)) for some function f , where the max function is applied element-wise. WLOG, let n denote the hidden dimension. We can now prove the theorem using mathematical induction.
1. Base case (t = 0): This is true because the inputs M (0) Θ (T , E, w) = (0, {w e : e ∈ E}, 0, 0) = (x (0) , r (0) , δ (0) , ∆ (0) ). 2. Induction step (t > 0): To formulate the strong induction hypothesis, let (x (t ′ ) , r (t ′ ) , δ (t ′ ) , ∆ (t ′ ) ) be the intermediate quantites computed by Algorithm 1 for each timestep t ′ ∈ [0, ..., t -1], assume M (t ′ ) Θ (T , E, w) = (x (t ′ ) , r (t ′ ) , δ (t ′ ) , ∆ (t ′ ) ). We now prove that M (t) Θ (T , E, w) = (x (t) , r (t) , δ (t) , ∆ (t) ).
The inputs for the t th step of our recurrent model are the outputs from the (t -1) th step. By the induction hypothesis, the model output r(t-1) = r (t-1) . Furthermore, since d (t) = f (T , E, max t ′ ∈[0,...,t] (x (t) ′ )) for some function f , and the model outputs satisfy x(t) ′ = x (t) ′ for all t ′ ∈ [0, ..., t -1] via the strong induction hypothesis, we have d(t-1) = d (t-1) . We take the natural logarithmic form of r (t-1) and d (t-1) as inputs to the encoders. Both encoders f r and f d are MLPs. We define the weight of f r as W fr = [1, 0, ..., 0] ∈ R n×1 . We define the weight of
f d as W f d = [1, 0, ..., 0] ∈ R n×1 . Therefore, h (t) e = f r (r (t-1) e ) = W fr ln r (t-1) e = [ln r (t-1) e , 0, ..., 0] ∈ R n×1 h (t) de = f d (d (t-1) e ) = W f d ln d (t-1) e = [ln d (t-1) e , 0, ..., 0] ∈ R n×1
The first step performs message-passing from the primal node representations h (t)  e to the dual nodes h
T via h
T = min e∈N (T ) g e (h (t) e , h(t)
de ). Let g e be an MLP with ELU activation, then we set
W ge = [1, 0, ..., 0, -1, 0, ..., 0] ⊤ ∈ R 1×2n , b ge = [1] ∈ R 1 , and W ′ ge = [1, 0, ..., 0] ∈ R n×1 thus: g e (h ′(t) e ) = W ′ ge ELU W ge [h (t) e ∥h (t) de ] + b ge = W ′ ge ELU [1, 0, ..., 0, -1, 0, ..., 0] ⊤ [ln r (t-1) e , 0, ..., 0, ln d (t-1) e , 0, ..., 0] + 1 = W ′ ge ELU ln r (t-1) e -ln d (t-1) e + 1 = W ′ ge ELU ln r (t-1) e d (t-1) e + 1 Since ELU(x) = e x -1 if x ≤ 0, and ln r (t-1) e d (t-1) e ≤ 0, g e (h ′(t) e ) = W ′ ge exp ln r (t-1) e d (t-1) e -1 + 1 = [1, 0, ..., 0] r (t-1) e d (t-1) e = [ r (t-1) e d (t-1) e , 0, ..., 0] ∈ R n×1
Therefore, the hidden representation h
T for each set T ∈ T , is:
h (t) T = min e∈N (T ) g e (h ′(t) e ) = min e∈N (T ) [ r (t-1) e d (t-1) e , 0, ..., 0]
Algorithm 4 General primal-dual approximation algorithm
= [δ (t) T , 0, ..., 0] ∈ R n×1 (From Line 1 of Algorithm 4)
The next step performs message-passing from the dual node representations h (t)
T and then updates the primal representations h
(t) e via h (t) e = g u (h (t) e , T ∈N (e) h(t)
T ). We use ELU activation function on the previously computed h (t) e and a bias b r = [1, 0, ..., 0] ∈ R n×1 . Since ln r (t-1) e ≤ 0, we have:
h (t) e = ELU h (t) e + b r = ELU [ln r (t-1) e , 0, ..., 0] + [1, 0, ..., 0] = [exp (ln r (t-1) e ) -1 + 1, 0, ..., 0] = [r (t-1) e , 0, ..., 0] ∈ R n×1
The update function g u is also an MLP. We define its weights to be W gu = [1, 0, ..., 0, -1, 0, ..., 0] ⊤ ∈ R 1×2n . Then, the hidden dimension h (t) e for each element e ∈ E becomes:
h (t) e = g u   h (t) e , T ∈N (e) h (t) T   = W gu   h (t) e ∥ T ∈N (e) h (t) T   = [1, 0, ..., 0, -1, 0, ..., 0] ⊤ [r (t-1) e , 0, ..., 0, T ∈N (e) δ (t) T , 0, ..., 0] = [r (t-1) e - T ∈N (e) δ (t) T , 0, ..., 0] = [r (t) e , 0, ..., 0] ∈ R n×1 (From Line 4 of Algorithm 4)
Alternatively, if the uniform increase rule is incorporated, we have an additional virtual node z that connects to all dual variables. Its hidden representation h (t) z is computed as: t) , 0, ..., 0] ∈ R n×1 (From Line 6.1 of Algorithm 4)
h (t) z = min T ∈T h (t) T = min T ∈T [δ (t) T , 0, ..., 0] = [min T ∈T δ (t) T , 0, ..., 0] = [∆ (
Then let h ′(t)
T = h (t) z , the primal variable updates becomes:
h (t) e = g u   h (t) e , T ∈N (e) h ′(t) T   = W gu   h (t) e ∥ T ∈N (e) h ′(t) T   = [1, 0, ..., 0, -1, 0, ..., 0] ⊤ [r (t-1) e , 0, ..., 0, T ∈N (e) ∆ (t) , 0, ..., 0] = [r (t-1) e -d (t-1) e ∆ (t) , 0, ..., 0] = [r (t) e , 0, ..., 0] ∈ R n×1 (From Line 6.2 of Algorithm 4)
For the decoders q x , q r , q δ (and q ∆ if uniform increase is used), they map hidden representations h (t) e , h
T (and h (t) z if uniform increase is used) to predictions for the intermediate quantities computed by the algorithm. We define
W qx = W qr = W q δ = W q∆ = [1, 0, ..., 0] ⊤ ∈ R 1×n . For x (t)
e , it is a binary classification task, where
x (t) e = 1 if r (t)
e = 0 to add the element e to the hitting set. Define o : R → {0, 1}, where
o(x) = 1, if x ≤ 0 0, else
We note that although we use a sigmoid function in our architecture, the sigmoid function can approximate o(x) to arbitrary precision by adjusting its temperature. Therefore, we have
x(t) e = o(q x (h (t) e )) = o W qx (h (t) e = o [1, 0, ..., 0] ⊤ [r (t) e , 0, ..., 0] = o r (t) e = x (t) e (Definition of x (t) e )
For the other three intermediate quantities:
r(t) e = q r (h (t) e ) = W qr (h (t) e ) = [1, 0, ..., 0] ⊤ [r (t) e , 0, ..., 0] = r (t) e δ(t) T = q δ (h (t) T ) = W q δ (h (t) T ) = [1, 0, ..., 0] ⊤ [δ (t) T , 0, ..., 0] = δ (t) T ∆(t) = q δ (h (t) z ) = W q∆ (h (t) z ) = [1, 0, ..., 0] ⊤ [∆ (t) , 0, ..., 0] = ∆ (t) Therefore, M (t) Θ (T , E, w) = (x (t) , r (t) , δ (t) , ∆ (t)
) and the induction step is completed.
this section cite: []

Section: C. Datasets

this section cite: []

Section: C.1. Synthetic datasets
We provide the details of the graph distributions used to generate random graphs for both training and testing.
this section cite: []

Section: Barabási-Albert (B-A) graph
Barabási-Albert graphs for both training and testing are randomly generated using networkx.barabasi_albert_graph. The number of edges to attach from a new node to existing nodes is randomly chosen from [1,10]. Node weights for primal variables are uniformly sampled from [0, 1].
this section cite: []

Section: Erdős-Rényi (E-R) graph
We use networkx.erdos_renyi_graph to generate random Erdős-Rényi graphs with edge probability uniformly sampled from [0.2, 0.8].
Star graph Star graphs are generated by randomly partitioning the nodes into 1 to 5 sets. Within each node set, a star graph is generated with a center node connected to all other nodes. Random edges between the star graphs are then added to ensure the graph is connected.
Lobster graph To generate a lobster graph, the number of nodes on the "backbone" m is randomly sampled from [1, n-1], where n is the total number of nodes in the graph. Then, another k nodes are added to the backbone nodes to start "branches", where 1 ≤ k ≤ (n -m). Finally, the remaining nodes (if any left from n -m -k) are randomly attached to the branches.
this section cite: []

Section: 3-Connected
(3-Con) Planar graph A 3-regular graph is randomly generated with networkx.random_regular_graph, and then checked to see if it is 3-connected and planar. The process is repeated until a valid 3-connected planar graph is found, or if it reaches the limit of 100 attempts.
this section cite: []

Section: E. Additional architectural details
In practice, due to the potentially high degree variance, we apply log transformation on the node degree d
e before encoding it with f d for better generalization, i.e. h
(t) de = f d (ln(d (t-1) e + 1)). Then, for decoding x(t) e = q x (h (t)
e ), which represents whether to include element e to the solution, we apply a sigmoid activation function to convert logits to probabilities. For minimum vertex cover and minimum set cover, since multiple elements can be included into the solution at each timestep, we set a threshold of 0.5 to decide whether to include the element. For minimum hitting set, since the uniform increase rule is used, only one element is included into the solution at each timestep, we choose the element with the highest probability to include in the hitting set. Lastly, we add dropouts in between processor layers with probability of 0.2 for Gurobi and real-world experiments. This helps the model to generalize to much larger graphs at a slight cost of approximation ratios.
this section cite: []

Section: F. Limitation and future work
While our framework is a general one that can learn any algorithm designed with the primal-dual paradigm, the architecture described in the paper can be directly adapted to solve any problem represented by the hitting set formulation. The hitting set provides a flexible structure for modeling various optimization problems by selecting a subset of elements that "hits" or "covers" all required constraints, represented as sets. This formulation is versatile because it can capture diverse constraints (e.g., nodes, edges, paths, cycles), making it applicable to numerous problems. As discussed in Section 3.3, Algorithm 1 can be reformulated to recover many classical (exact or approximation) algorithms for problems that are special cases of the hitting set, covering both polynomial-time solvable and NP-hard problems. Some of these special cases are illustrated in Goemans & Williamson (1996) and Williamson & Shmoys (2011), including shortest s-t path, minimum spanning tree, vertex cover, set cover, minimum-cost arborescence, feedback vertex set, generalized Steiner tree, minimum knapsack, and facility location problems.
We note that not all problems can be directly represented by the hitting set formulation. As a minimization problem, the hitting set does not naturally align with maximization objectives, making the primal-dual approximation algorithm less straightforward to apply. However, the primal-dual framework can still be extended to maximization problems by carefully reformulating the primal-dual pair, where the dual is a minimization problem, and adapting Algorithm 1. Then, similarly, the algorithm starts with a feasible dual solution and iteratively updates both primal and dual variables to reduce the gap between them. Therefore, PDNAR, with its bipartite graph structure, can still be extended to handle maximization problems with appropriate adjustments to Algorithm 1. Furthermore, while Algorithm 1 provides a general framework for designing primal-dual approximation algorithms, it can be further strengthened by incorporating techniques to enhance the algorithmic performance. One such technique is the uniform increase rule, which we have shown how it can be integrated into our framework. Future work can incorporate other advanced techniques, such as those outlined in (Williamson & Shmoys, 2011), to further extend our framework's ability to accommodate a broader range of primal-dual approximation algorithms with improved worst-case guarantees.
Lastly, our method may not explicitly preserve the worst-case approximation guarantees of the primal-dual algorithm in practice. Instead, our model focuses on learning solutions that perform better for the training distribution and generalize well to new instances. While this does not mean that the worst-case guarantees are secured, it is important to note that such cases are often less common in real-world scenarios. One of the core strengths of NAR lies in leveraging a pretrained GNN with embedded algorithmic knowledge to tackle real-world datasets. Therefore, we believe it is valuable to train models to produce high-quality solutions for common cases, even if they do not preserve worst-case guarantees. This aligns with the overarching goals of NAR.
this section cite: ['b28', 'b68', 'b68']

Section: G. Related works on neural algorithmic reasoning
We provide a more comprehensive review of existing works on Neural Algorithmic Reasoning (NAR) and highlight our contributions in this context.
this section cite: []

Section: Neural algorithmic reasoning
The algorithmic alignment framework proposed by Xu et al. (2020) suggests that GNNs are particularly well-suited for learning dynamic programming algorithms due to their shared aggregate-update mechanism. Additionally, Veličković et al. (2020) demonstrates the effectiveness of GNNs in learning graph algorithms such as BFS and Bellman-Ford. These foundational works have contributed to the development of neural algorithmic reasoning (Veličković & Blundell, 2021), which investigates the potential of neural networks, particularly GNNs, to simulate traditional algorithmic processes. This research direction has since inspired several follow-up studies, including efforts to instantiate the framework for specific algorithms (Deac et al., 2020;Georgiev & Liò, 2020;Zhu et al., 2021), applications to real-world use cases (Deac et al., 2021;He et al., 2022;Beurer-Kellner et al., 2022;Georgiev et al., 2023a;Numeroso et al., 2023;Estermann et al., 2024), architectural improvements (Georgiev et al., 2022;Bevilacqua et al., 2023;Rodionov & Prokhorenkova, 2023;Jain et al., 2023;Engelmayer et al., 2023;Mirjanic et al., 2023;Georgiev et al., 2023b;Dudzik et al., 2024;Jürß et al., 2024;Xhonneux et al., 2024;Georgiev et al., 2024;Rodionov & Prokhorenkova, 2024;Xu & Veličković, 2024;Kujawa et al., 2024), and integration with large language models (Bounsi et al., 2024). Our work advances NAR by introducing a general framework designed to tackle combinatorial optimization problems, particularly NP-hard ones, with the objective of simulating and outperforming primal-dual approximation algorithms.
this section cite: ['b74', 'b65', 'b63', 'b12', 'b20', 'b78', 'b13', 'b34', 'b4', 'b54', 'b17', 'b21', 'b5', 'b56', 'b38', 'b16', 'b53', 'b7', 'b42', 'b23', 'b57', 'b71', 'b48', 'b7']

Section: Combinatorial optimization with GNNs
The CLRS benchmark (Veličković et al., 2022) and its extensions (Minder et al., 2023;Markeeva et al., 2024) are widely recognized for evaluating GNNs on 30 algorithms from the CLRS textbook (Cormen et al., 2001) and more. However, these algorithms are limited to polynomial-time problems, leaving the more challenging NP-hard problems largely unexplored in neural algorithmic reasoning. A comprehensive review by Cappart et al. (2022) summarizes the current progress of using GNNs for combinatorial optimization. The most relevant work (Georgiev et al., 2023c) trains GNNs on algorithms for polynomial-time-solvable problems and test them on NP-hard problems, demonstrating the value of algorithmic knowledge over non-algorithmically informed models. In contrast, our approach bridges this gap by extending GNNs to tackle NP-hard problems through the use of primal-dual approximation algorithms. Furthermore, we integrate optimal solutions from integer programming, which guides the model toward better outcomes during training. To the best of our knowledge, our method is the first of its kind to surpass the performance of the algorithms it was originally trained on.
Multi-task learning for NAR Early work by Veličković et al. (2020) demonstrated that BFS and Bellman-Ford are best learned jointly, and subsequent studies have highlighted broader benefits of multi-task learning when GNNs are trained on multiple algorithms simultaneously (Xhonneux et al., 2021;Ibarz et al., 2022). Building on this, Numeroso et al. (2023) leveraged the primal-dual principle from linear programming to successfully learn the Ford-Fulkerson algorithm using the max-flow min-cut theorem. However, their approach was tailored specifically for Ford-Fulkerson and did not generalize to other primal-dual scenarios or address NP-hard problems. To overcome these limitations, our work introduces a general framework that employs the primal-dual principle to enable GNNs to benefit from multi-task learning across a broad range of optimization problems, particularly those expressible as instances of the general hitting set problem.
this section cite: ['b62', 'b52', 'b51', 'b9', 'b8', 'b65', 'b69', 'b37', 'b54']

Section: A. Additional details of vertex cover and set cover A.1. Primal-dual pair: vertex cover and edge packing
Given a graph G = (V, E), where V are vertices and E are edges, each vertex v ∈ V has a non-negative weight w : V → R + .
Definition A.1 (Minimum vertex cover). A vertex-cover for G is a subset C ⊆ V of the vertices such that for each edge (v, u) ∈ E, either v ∈ C, u ∈ C, or both. The objective is to minimize the total vertex weight v∈C w(v).
Definition A.2 (Maximum edge packing). An edge-packing is an assignment p : E → R + of non-negative weights to the edges e ∈ E, such that for any vertex v ∈ V , the total weight e:v∈e p(e) assigned to the edges e that are incident to v is at most w(v). The objective is to maximize the total edge weight e∈E p(e).
The edge packing problem is the dual of the LP relaxation of the vertex cover problem, which also has many practical implications, such as resource allocation. Because of this relationship, the primal-dual pair becomes key problems for studying approximation algorithms and the primal-dual framework. Let x v ∈ {0, 1} indicate whether each vertex v ∈ V is in the cover, and y e ∈ R + be the non-negative weight assigned to each edge e ∈ E. The two problems can be formulated as:
Min v∈V w(v)x v sub. to x u + x v ≥ 1, ∀e = (u, v) ∈ E x v ∈ {0, 1}, ∀v ∈ V Max e∈E y e sub. to e:v∈e y e ≤ w(v), ∀v ∈ V y e ≥ 0, ∀e ∈ E.
this section cite: []

Section: A.2. Pseudocode of MVC algorithm
A 2/(1 -ϵ)-approximation algorithm for the minimum vertex cover (MVC) problem was proposed by Khuller et al. (1994).
It can be interpreted as an instantiation of the general primal-dual approximation algorithm without uniform increase (Algorithm 1). In the following, we give the original algorithm as illustrated in the original paper (Khuller et al., 1994).
Intuitively, the algorithm maintains a packing p and partial cover C p = {v ∈ V : p(E(v)) ≥ (1 -ϵ)w(v)}, and gradually increases the edge weights p(e) as much as possible. When the constraint on the residual vertex weight is met, a vertex v is removed and added to the cover C p . The process iterates until p is ϵ-maximal and C p is a cover. Let E p (v) be the set of remaining edges incident to vertex v, d p (v) = |E p (v)| be the degree, and w p (v) = w(v) -p(E(v)) be the residual weight.
The following is a pseudocode of the algorithm as described in Khuller et al. (1994).
Algorithm 2 COVER(G = (V, E), w, ϵ) 1: for v ∈ V do 2: w p (v) ← w(v); E p (v) ← E(v); d p (v) ← |E(v)| 3: end for 4: while edges remain do 5: for each remaining edge (u, v) do 6: δ((u, v)) ← min wp(u) dp(u) , wp(v) dp(v) 7: end for 8:
for each remaining vertex v do 9:
w p (v) ← w p (v) -e∈Ep(v) δ(e) 10: if w p (v) ≤ ϵ • w(v)
then
11: delete v and its incident edges; update E p (•) and d p (•) 12: end if 13: end for 14: end while 15: Output: the set of deleted vertices
The parameter b can be varied to generate OOD graphs. For training, we choose b = 5. For testing OOD generalization, we use b = 3 and b = 8 to generate test graphs with sparser and denser preferential attachment.
For vertex cover and set cover, we use Algorithm 2 and Algorithm 3 (Khuller et al., 1994) to generate intermediate supervisions, which are instantiations of Algorithm 1 with improved efficiency, as explained in Section 3.3. For hitting set, we use Algorithm 1 with the uniform increase rule. Furthermore, the optimal solutions are generated with the default IP solver in scipy, which is based on HiGHS (Schwendinger & Schumacher, 2023;Huangfu & Hall, 2018).
this section cite: ['b45', 'b45', 'b45', 'b45', 'b60', 'b36']

Section: C.2. Gurobi datasets
Random B-A graphs are generated following the same distribution described above. For testing, we generate B-A graphs with 500, 600, and 750 nodes. We use the trained model to perform inference on the testing set and retrieve vertex cover solutions. We also use Algorithm 2 (Khuller et al., 1994) to compute solutions with ϵ = 0.1. The comparison of the solutions from the model and the algorithm is shown in Table 6. The two sets of solutions are then used to initialize variables to warm-start the Gurobi solver. We also compare them with Gurobi's default initialization (i.e., no warm start). We use the default parameter settings of Gurobi, setting thread count to 1 (model.setParam('Threads', 1)), the time limit to 3600s (model.setParam('TimeLimit', 3600)), and random seed (model.setParam('Seed', seed)). Each experiment is repeated with 5 seeds. Airports The Airports datasets (Ribeiro et al., 2017) consist of three airport networks from Brazil (131 nodes, 1038 edges), Europe (399 nodes, 5995 edges), and the USA (1190 nodes, 13599 edges). The nodes represent airports, and the edges represent commercial flight routes. The node features are one-hot encoded node identifiers, as described in (Jin et al., 2019).
The task is to predict the activity level of each airport, measured by the total number of landings plus takeoffs, or the total number of people arriving plus departing. It is a classification task with 4 labels, with label 1 assigned to the 25% least active airports, and so on, according to the quartiles of the activity distribution. We create 10 random train/val/test splits for the transductive task with a ratio of 60%/20%/20%.
this section cite: ['b45', 'b55', 'b39']

Section: D. Hyperparameters
All GPU experiments were performed on Nvidia Quadro RTX 8000 with 48GB memory. The Gurobi experiments were conducted on Intel Xeon E7-8890x with 144 cores and 12TB memory.
this section cite: []

Section: Synthetic and Gurobi experiments
For training, we use the Adam optimizer with an initial learning rate of 1e-3 and weight decay of 1e-4, coupled with the ReduceLROnPlateau scheduler with default settings. Additionally, we use a batch size of 32, a hidden dimension of 32, and a maximum of 100 epochs. For testing, we use the trained model with the lowest validation loss.
this section cite: []

Section: Real-world dataset experiments
We use the same optimizer and scheduler settings as in the synthetic experiments. Additionally, we also apply early stopping with a patience of 10 epochs based on validation loss and set the scheduler with a patience of 20 epochs. All embeddings have a fixed dimension of 32. For testing, we use the model with the lowest validation loss. The base models are used with max jumping knowledge (Xu et al., 2018) and L2 normalization after each layer (Rossi et al., 2024). We conduct hyperparameter search for each base model on Airports datasets (Brazil, Europe, USA), then use the setting for all embedding methods. Due to the high computational costs of WikipediaNetwork (Chameleon, Squirrel) and PPI datasets, hyperparameter search is only done with GCN. We use the default TPE hyperparameter search algorithm from optuna (Akiba et al., 2019) with a median pruner. The searchable parameters are lr=[0.01, 0.001, 0.0005], hid_dim=[32,
this section cite: ['b72', 'b58', 'b1']

Section: H. Contribution to Neural Combinatorial Optimization (NCO)
Most GNN-based supervised learning methods for NCO learn task-specific heuristics or optimize solutions in an end-to-end manner (Joshi et al., 2019;Li et al., 2018;Gasse et al., 2019;Fu et al., 2021). These end-to-end approaches rely exclusively on supervision signals derived from optimal solutions, which are computationally expensive to obtain for hard instances. Furthermore, dependence on such labels can limit generalization (Joshi et al., 2022). In contrast, our method trains on synthetic data obtained efficiently from a polynomial-time approximation algorithm. The embedding of algorithmic knowledge also demonstrates strong generalization. The addition of optimal solutions are derived from small problem instances, enabling our model to outperform the approximation algorithm and generalize effectively to larger problem sizes.
Our approach represents a previously underexplored area of NCO research with GNNs, offering a general framework to build an algorithmically informed GNN to tackle combinatorial optimization problems. Unlike end-to-end methods, we leverage intermediate supervision signals from polynomial-time approximation algorithms, which can be generated efficiently, to address key bottlenecks in data efficiency and generalization. Additionally, we fill the gap in autoregressive methods for NCO using GNNs by aligning our architecture with the primal-dual method, enabling the GNN to simulate a single algorithmic step in an efficient and structured manner.
We provide additional empirical evidences on the RB benchmark graphs, which are well-known hard instances for the Minimum Vertex Cover (MVC) problem. We compared our method with several NCO baselines. Given the supervised nature of our approach, obtaining optimal solutions for RB200/500 graphs is computationally challenging. Therefore, we tested the generalization of our model, trained on Barabási-Albert graphs of size 16 (using intermediate supervision from the primal-dual algorithm and optimal solutions), on the larger RB200/500 graphs. We compare with EGN (Karalias & Loukas, 2021) and Meta-EGN (Wang & Li, 2023), two powerful NCO baselines for these benchmarks, as well as two algorithms (the primal-dual approximation algorithm and the greedy algorithm) and Gurobi. The results are summarized in Table 8. We note that both EGN and Meta-EGN are unsupervised methods, whereas our method adopts a supervised approach. A fairer comparison would involve supervised baselines for NCO. However, current focus of NCO has primarily shifted towards unsupervised methods, leaving existing supervised baselines with outdated performances. Obtaining labels for large graphs, such as RB200/500, is also computationally prohibitive. Additionally, supervised methods usually require a combination of external solvers or search algorithms, which is a different setting than ours. In summary, our supervised approach addresses this underexplored direction of NCO by addressing its key challenges -PDNAR as an algorithmically informed GNN that leverages efficiently obtainable labels while demonstrating strong generalization capabilities.
this section cite: ['b40', 'b50', 'b19', 'b18', 'b41', 'b43', 'b66']

Section: I. Duality in linear programming
Duality in linear programming has been utilized in training neural networks to tackle optimization problems. Li et al. (2024) introduces a Learning-to-Optimize method to mimic Primal-Dual Hybrid Gradient method for solving large-scale LPs. While they focus on developing efficient solvers for LPs, we aim to simulate the primal-dual approximation algorithm for NP-hard problems using GNNs. Although both approaches reference the primal-dual framework, this similarity is superficial.
The primal-dual terminology is widely used in optimization, but our work applies it to study algorithmic reasoning. For example, the primal-dual approximation algorithm can be instantiated to many traditional algorithms, such as Kruskal's algorithm for MST. Furthermore, unlike Li et al. (2024), our method relies on intermediate supervision from the primal-dual algorithm to guide reasoning, ensuring that the model learns to mimic algorithmic steps. We also incorporate optimal solutions into the training process to improve solution quality, allowing our model to outperform the primal-dual algorithm it is trained on. Moreover, the architectures differ significantly: our method employs a recurrent application of a GNN to iteratively solve problems, while Li et al. (2024) does not use GNNs or recurrent modeling. These distinctions highlight that our focus is not on solving LPs but on leveraging NAR to generalize algorithmic reasoning for NP-hard problems.
this section cite: ['b49', 'b49', 'b49']

Section: References
Ref_id:b0 Title: When trees collide: An approximation algorithm for the generalized steiner problem on networks Year: (1995)
Ref_id:b1 Title: A next-generation hyperparameter optimization framework Year: (2019)
Ref_id:b2 Title: A linear-time approximation algorithm for the weighted vertex cover problem Year: (1981)
Ref_id:b3 Title: URL Year: ()
Ref_id:b4 Title: Learning to configure computer networks with neural algorithmic reasoning Year: (2022)
Ref_id:b5 Title: Neural algorithmic reasoning with causal regularisation Year: (2023-07)
Ref_id:b6 Title: An experimental study of algorithms for online bipartite matching Year: (2018)
Ref_id:b7 Title: Transformers meet neural algorithmic reasoners Year: (2024)
Ref_id:b8 Title: Combinatorial optimization and reasoning with graph neural networks Year: (2022)
Ref_id:b9 Title: Introduction to Algorithms Year: (2001)
Ref_id:b10 Title: Learning Combinatorial Optimization Algorithms over Graphs Year: (2018-02)
Ref_id:b11 Title: Simulation of graph algorithms with looped transformers Year: (2024)
Ref_id:b12 Title: Graph neural induction of value iteration Year: (2020)
Ref_id:b13 Title: Neural algorithmic reasoners are implicit planners Year: (2021)
Ref_id:b14 Title: Benchmarking graph neural networks Year: (2022)
Ref_id:b15 Title: Graph neural networks with learnable structural and positional representations Year: ()
Ref_id:b16 Title: Parallel algorithms align with neural execution Year: (2023)
Ref_id:b17 Title: Puzzles: A benchmark for neural algorithmic reasoning Year: (2024)
Ref_id:b18 Title: Generalize a small pre-trained model to arbitrarily large tsp instances Year: (2021-05)
Ref_id:b19 Title: Exact Combinatorial Optimization with Graph Convolutional Neural Networks Year: (2019-10)
Ref_id:b20 Title: Neural bipartite matching Year: (2020)
Ref_id:b21 Title: Algorithmic concept-based explainable reasoning Year: (2022-06)
Ref_id:b22 Title: NARTI: Neural Algorithmic Reasoning for Trajectory Inference Year: (2023)
Ref_id:b23 Title: The deep equilibrium algorithmic reasoner Year: (2024)
Ref_id:b24 Title: Beyond erdos-renyi: Generalization in algorithmic reasoning on graphs Year: ()
Ref_id:b25 Title: Neural algorithmic reasoning for combinatorial optimisation Year: ()
Ref_id:b26 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b27 Title: A general approximation technique for constrained forest problems Year: (1995)
Ref_id:b28 Title: The primal-dual method for approximation algorithms and its application to network design problems Year: (1996)
Ref_id:b29 Title: node2vec: Scalable feature learning for networks Year: (2016)
Ref_id:b30 Title: LLC. Gurobi Optimizer Reference Manual Year: (2024)
Ref_id:b31 Title: Inductive representation learning on large graphs. Advances in neural information processing systems Year: (2017)
Ref_id:b32 Title: Relational inductive bias for physical construction in humans and machines Year: (2018)
Ref_id:b33 Title: Matching algorithms via GNNs for online valueto-go approximation Year: (2024)
Ref_id:b34 Title: Continuous neural algorithmic planners Year: (2022-12)
Ref_id:b35 Title: Approximation algorithms for set covering and vertex cover problems Year: (1982)
Ref_id:b36 Title: Parallelizing the dual revised simplex method Year: (2018)
Ref_id:b37 Title: A generalist neural algorithmic learner Year: (2022)
Ref_id:b38 Title: Neural priority queues for graph neural networks Year: (2023)
Ref_id:b39 Title: Gralsp: Graph neural networks with local structural patterns Year: (2019)
Ref_id:b40 Title: An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem Year: (2019-10)
Ref_id:b41 Title: Learning the Travelling Salesperson Problem Requires Rethinking Generalization Year: (2022-05)
Ref_id:b42 Title: Recursive algorithmic reasoning Year: (2024-11-30)
Ref_id:b43 Title: Erdos Goes Neural: an Unsupervised Learning Framework for Combinatorial Optimization on Graphs Year: (2021-03)
Ref_id:b44 Title: Vertex cover might be hard to approximate to within 2-ε Year: (2008)
Ref_id:b45 Title: A primal-dual parallel approximation technique applied to weighted set and vertex covers Year: (1994-09)
Ref_id:b46 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b47 Title: On the shortest spanning subtree of a graph and the traveling salesman problem Year: (1956)
Ref_id:b48 Title: Neural algorithmic reasoning with multiple correct solutions Year: (2024)
Ref_id:b49 Title: PDHG-Unrolled Learning-to-Optimize Method for Large-Scale Linear Programming Year: (2024-06)
Ref_id:b50 Title: Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search Year: (2018-10)
Ref_id:b51 Title: The clrs-text algorithmic reasoning language benchmark Year: (2024)
Ref_id:b52 Title: SALSA-CLRS: A sparse and scalable benchmark for algorithmic reasoning Year: (2023)
Ref_id:b53 Title: Latent space representations of neural algorithmic reasoners Year: (2023)
Ref_id:b54 Title: Dual algorithmic reasoning Year: (2023)
Ref_id:b55 Title: struc2vec: Learning node representations from structural identity Year: (2017-08)
Ref_id:b56 Title: Neural algorithmic reasoning without intermediate supervision Year: (2023)
Ref_id:b57 Title: Discrete neural algorithmic reasoning Year: (2024)
Ref_id:b58 Title: Edge directionality improves learning on heterophilic graphs Year: (2024)
Ref_id:b59 Title: Understanding transformer reasoning capabilities via graph algorithms Year: (2024)
Ref_id:b60 Title:  Year: (2023)
Ref_id:b61 Title: R-project.org/package=highs. R package version 0 Year: ()
Ref_id:b62 Title: The clrs algorithmic reasoning benchmark Year: (2022)
Ref_id:b63 Title:  Year: (2021-07)
Ref_id:b64 Title: Graph attention networks Year: (2018)
Ref_id:b65 Title: Neural execution of graph algorithms Year: (2020)
Ref_id:b66 Title: Unsupervised Learning for Combinatorial Optimization Needs Meta-Learning Year: (2023-01)
Ref_id:b67 Title: Towards a general recipe for combinatorial optimization with multi-filter GNNs Year: (2024)
Ref_id:b68 Title: The primal-dual method Year: (2011)
Ref_id:b69 Title: How to transfer algorithmic reasoning knowledge to learn new algorithms? Year: (2021)
Ref_id:b70 Title: Deep equilibrium models for algorithmic reasoning Year: ()
Ref_id:b71 Title: Recurrent aggregators in neural algorithmic reasoning Year: (2024)
Ref_id:b72 Title: Representation learning on graphs with jumping knowledge networks Year: (2018)
Ref_id:b73 Title: How powerful are graph neural networks? Year: (2019)
Ref_id:b74 Title: What can neural networks reason about Year: (2020)
Ref_id:b75 Title: How neural networks extrapolate: From feedforward to graph neural networks Year: (2021)
Ref_id:b76 Title: Are graph neural networks optimal approximation algorithms? Year: (2024)
Ref_id:b77 Title: Teaching algorithmic reasoning via in-context learning Year: (2022)
Ref_id:b78 Title: Neural bellman-ford networks: A general graph neural network framework for link prediction Year: (2016)
