Title: Graph-Based Attention for Differentiable MaxSAT Solving
Abstract: The use of deep learning to solve fundamental AI problems such as Boolean Satisfiability (SAT) has been explored recently to develop robust and scalable reasoning systems. This work advances such neural-based reasoning approaches by developing a new Graph Neural Network (GNN) to differentiably solve (weighted) Maximum Satisfiability (MaxSAT). To this end, we propose SAT-based Graph Attention Networks (SGATs) as novel GNNs that are built on t-norm based attention and message passing mechanisms, and structurally designed to approximate greedy distributed local search. To demonstrate the effectiveness of our model, we develop a local search solver that uses SGATs to continuously solve any given MaxSAT problem. Experiments on (weighted) MaxSAT benchmark datasets demonstrate that SGATs significantly outperform existing neural-based architectures, and achieve state-of-the-art performance among continuous approaches, highlighting the strength of the proposed model. 1

Section: Introduction
Neuro-symbolic AI aims to develop robust and scalable reasoning systems by combining the strengths of both symbolic logic and neural networks [20,6]. Boolean Satisfiability (SAT), a fundamental reasoning problem in AI and Computer Science, has long been examined as an important topic in neuro-symbolic research. There have been many studies on developing SAT solving methods using deep learning, which can either solve SAT problems in an end-to-end manner [22,1], or can be combined with existing discrete solvers to enhance their performance [21,11].
Maximum Satisfiability (MaxSAT), an optimization generalization of SAT, has been viewed as a promising approach towards accomplishing various neuro-symbolic tasks [9,28]. SATNet is one representative approach, capable of learning to solve structured reasoning tasks such as visual sudoku by employing a differentiable MaxSAT solving layer [27]. This layer is built upon a continuous optimization algorithm, specifically based on the use of Semidefinite Programming (SDP) [26]. While this algorithm has proved effective on certain problem types, other promising approaches have also been proposed, such as the use of Fourier analysis [12] and Graph Neural Networks [17]. The latter, however, remains largely underexplored.
Graph Neural Networks (GNNs) have been widely used in relational and symbolic domains, as well as in multiple neuro-symbolic systems [13]. There has recently been a surge of interest in using GNNs as a key building block for combinatorial optimization problems [3]. In particular, several works have applied GNNs to MaxSAT solving, either in an end-to-end manner [17], or by using their predictions as heuristics for existing solvers [16]. In contrast, we propose GNN architectures capable of differentiably solving weighted MaxSAT problems, and are effective against practical problem instances.
In this paper, we present SAT-based Graph Attention Networks (SGATs) as novel GNNs crafted for solving MaxSAT, which are the first GNNs to be able to handle weighted MaxSAT problems. SGATs are composed of GNN layers with novel t-norm based attention, where attention mechanisms operate on values computed using t-norms. When a clause is unsatisfied by a candidate assignment, the clause node sends messages to all its connected variable nodes, requesting them to update their values in parallel towards satisfying the clause. The attention mechanism then computes priorities to decide which variable nodes should change their values to maximally satisfy those requests from clauses, allowing the model to learn which clauses to focus on. Intuitively, this can be regarded as learning a general heuristic that can be applied to a variety of MaxSAT problems. To show the effectiveness of SGATs, we further build a local search algorithm that finds solutions with the continuous optimization of GNNs through iterative training, which can be used with any GNN architecture as the backbone.
To evaluate the effectiveness of our model, we first compare the performance against existing neuralbased architectures using benchmark instances from the MaxSAT Evaluations in 2018. We then conduct ablation studies to analyze the effectiveness of each component of SGATs, including the t-norm based attention and message passing mechanisms. Subsequently, we use benchmark instances used in MaxSAT evaluations from 2020 to 2024, and compare the qualities of the predictions with solvers based on state-of-the-art (SOTA) continuous approaches. The experimental results show that SGATs demonstrate excellent stability and performance compared to existing neural architectures and mechanisms during training, and is able to achieve the SOTA performance for continuous solvers on all benchmark sets. We discuss the limitations and broader impacts of our work in Appendix F and Appendix G, respectively.
Our key contributions are as follows:
• We present SAT-based Graph Attention Networks (SGATs) as the first GNN architecture specifically designed for MaxSAT solving, and is able to handle weighted MaxSAT problems.
• We introduce novel t-norm based attention and message passing mechanisms that are specifically designed to approximate greedy distributed local search.
• We demonstrate that SGATs outperform existing neural architectures, and achieve SOTA performance for continuous solvers on all benchmark sets.
2 Related Work
this section cite: ['b21', 'b5', 'b23', 'b0', 'b22', 'b11', 'b9', 'b29', 'b28', 'b27', 'b12', 'b18', 'b14', 'b2', 'b18', 'b17']

Section: Differentiable Solvers
In SAT, there have been multiple works attempting to construct differentiable solvers, with deep learning being the main method to accomplish this. Particularly, most works have focused on using GNNs [1,11] as well as Recurrent Neural Networks (RNNs) [22]. Others have attempted to use these models as heuristics for state-of-the-art (SOTA) solvers, in an effort to further bridge the gap between the two fields [21,31], with reinforcement learning being one prominent approach [8,11].
To support further research in this domain, [15] has built a codebase that deploys a wide range of neural architectures, as well as benchmarks known to date. Constraint Satisfaction Problems (CSPs) are another related field, with works focusing on learning search heuristics or end-to-end solvers via neural architectures, with the use of GNNs and transformer variants [23,29].
In the context of MaxSAT solving, not many differentiable solvers have been proposed to this day, and works that do mainly focus on solving synthesized problems with the use of GNNs [17].
However, there are no works that utilize Graph Attention Networks (GATs), which had been shown to be effective for related problems such as SAT, CSP, and Minimal Unsatisfiable Subset extraction [4,29,18]. In contrast, our work focuses on developing a much more robust GNN architecture that employs attention and message passing mechanisms that utilize t-norm computation, a direction not explored in prior works to the best of our knowledge.
this section cite: ['b0', 'b11', 'b23', 'b22', 'b32', 'b7', 'b11', 'b16', 'b24', 'b30', 'b18', 'b3', 'b30', 'b19']

Section: Continuous Optimization Based MaxSAT Solvers
Several approaches have been proposed for MaxSAT solving with the use of continuous optimization. The Mixing method [26] is one such approach, specifically using low-rank coordinate descent for Semidefinite Programming (SDP). Together with multiple techniques such as branch-and-bound, it has achieved SOTA performance in solving Max2SAT problems, a special case of MaxSAT problems where each clause has strictly 2 literals [25]. FourierSAT is another continuous optimization based approach, utilizing the Fourier analysis of Boolean functions to handle various types of Boolean constraints [12]. Although this is the only line of work to address weighted MaxSAT, their current implementation does not fully support this, indicating the practical difficulties of handling them.
this section cite: ['b27', 'b26', 'b12']

Section: Background

this section cite: []

Section: SAT, MaxSAT and Weighted MaxSAT
In Boolean Satisfiability (SAT), a propositional logic formula consisting of variables, negations (¬), conjunctions (∧), and disjunctions (∨) is encoded into Conjunctive Normal Form (CNF). A CNF formula is composed by conjunctions of multiple sub-formulas called clauses, with each clause being composed by disjunctions of variables or their negations, called literals. Each variable can be assigned a logical value of false (0) or true (1), and the formula is satisfied if and only if there exists an assignment where at least one literal in each clause is mapped to true (each clause is satisfied).
In Maximum Satisfiability (MaxSAT), the objective is to find the assignment of variables that maximizes the number of satisfied clauses in the formula. In weighted MaxSAT, the objective shifts to finding the assignment that maximizes the total weight of satisfied clauses. In the following sections, we will use n and m to denote the number of variables and clauses in a problem, and w i to denote the weight of clause C i . Furthermore, we refer to cost as the total sum of weights of unsatisfied clauses.
In MaxSAT, there are two major categories of algorithms: complete and incomplete. Complete algorithms guarantee that given solutions are optimal, while incomplete algorithms aim to find good-quality assignments within a reasonable time frame. In our work, we focus on the latter, mainly as guaranteeing optimalities of solutions are difficult with continuous approaches. The current SOTA incomplete solvers are based on Stochastic Local Search, with multiple of them being proposed in recent years [5,33].
this section cite: ['b4', 'b34']

Section: Graph Attention Network
Recent research has shown the usage of GNNs with combinatorial optimization problems (including MaxSAT) to be promising [3,17]. As such, we take inspiration from Graph Attention Networks (GATs), one of the SOTA architectures for graph based learning [24]. GATs are equipped with attention mechanisms that compute specific weights (attention coefficients) for each neighboring node in order to prioritize important nodes, enabling a leap in model capacity.
The attention mechanism of GATs proceeds as follows. First, the attention coefficients ϵ ij and normalized attention coefficients α ij for neighbor node j of node i are calculated:
ϵ ij = LeakyReLU a T [Wh i ∥Wh j ∥W e e ij ] α ij = exp (ϵ ij ) k∈Ni exp (ϵ ik )(1)
Here, a ∈ R 2F ′ is a weight vector, ∥ is a concatenation operation, and W ∈ R F ′ ×F , W e ∈ R F ′ ×Fe are weight matrices (F and F ′ represent the number of features in a node at the input and output, and nd F e the number of features in an edge), with the LeakyReLU nonlinearity being applied (ReLU with a slight negative slope). Furthermore, h i ∈ R F is the feature of node i, e ij ∈ R Fe is the edge feature between node i and j, and N i is the set of neighbors of node i. The above notation is slightly modified to allow the use of multi-dimensional edge features [7]. Subsequently, these normalized attention coefficients are used to compute the final output feature h ′ i ∈ R F ′ for node i:
h ′ i = j∈Ni α ij Wh j(2)
this section cite: ['b2', 'b18', 'b25', 'b6']

Section: SAT-based Graph Attention Network
In this section, we present key building blocks for constructing SGATs as shown in Figure 1. SGATs are composed of three main components: (i) T-norm layers that compute clause valuations v(C) from the valuations of connected variables v(x), (ii) SGAT layers that update the valuations of variables based on the valuations of connected clauses, and (iii) a normalization layer that ensures the valuations of all variables are sufficiently spread out. A single block of SGATs approximates a distributed local search step, with the attention mechanism learning which clauses to focus on, essentially serving as a heuristic. We denote variables by x i and clauses by C j , with valuations v(x i ), v(C j ) ∈ [0, 1]. For clarity, v(x i ) and v(C j ) correspond to features at nodes h i and h j+n , respectively. Before applying GNNs to MaxSAT problems, we have to transform them into graphs. To accomplish this, we employ a factor graph representation of given formulas, as done in most prior works involving SAT and MaxSAT Solving [22,30,17]. Specifically, we opt for a representation similar to [30], and obtain a bipartite graph with two types of nodes for both the clauses and variables as shown in Figure 2. The positive and negative polarities of the variables are then embedded into the edge features.
this section cite: ['b23', 'b31', 'b18', 'b31']

Section: Graph Representation of MaxSAT Problems
While existing graph representations used edge features to mainly differentiate between the polarities of variables, we propose to embed crucial information as edge features, to allow the model to further differentiate between similar connections. We define the edge feature e i,j+n between variable x i and clause C j as follows:
e i,j+n =        1 |N + j | , 1 |N - j | , 0, 0 • w norm j if i ∈ N + j 0, 0, 1 |N + j | , 1 |N - j | • w norm j if i ∈ N - j , where w norm j = w j max k w k
Here, N + j and N - j correspond to the set of variables with positive and negative polarities included in clause C j , and w norm j ∈ (0, 1] represents the normalized weight of clause C j . Note that when the denominator is 0, we set the corresponding value to 0. These features align with common strategies  prioritizing shorter clauses over long ones [10], while also giving higher scores to clauses with larger weights; this is especially relevant as edge features are directly used for attention computation.
this section cite: ['b10']

Section: T-norm Layer: Variable to Clause
Here, we propose the use of t-norm based aggregations when computing clause valuations from connected variable valuations, as shown in the bottom left of Figure 1. By using t-norms, we compute clause valuations continuously while keeping all valuations in [0, 1]. Given fuzzy truth degrees a 1 , . . . , a k ∈ [0, 1] the fuzzy conjunctions of these values can be computed using t-norms T : [0, 1] k → [0, 1]. We specifically use the following three well-known t-norms:
1. Gödel t-norm: T G (a 1 , . . . , a k ) = min(a 1 , . . . , a k ) 2. Product t-norm: T P (a 1 , . . . , a k ) = k i=1 a i 3. Łukasiewicz t-norm: T L (a 1 , . . . , a k ) = max 0, k i=1 a i -(k -1)
As each clause is a disjunction of literals, we compute valuations by negating the conjunction of each negated literal. Using strong negation defined as 1 -a for a ∈ [0, 1], we set ṽir = 1 -v(x ir ) if l ir = x ir and ṽir = v(x ir ) if l ir = ¬x ir for a clause C i = l i1 ∨ • • • ∨ l iℓi , and define
v ⋆ (C i ) = 1 -T ⋆ ṽi1 , . . . , ṽiℓi ,
where ⋆ ∈ {G, P, L}.
For simplicity, we refer to the clause valuation computed by the chosen t-norm as v(C i ).
Unsupervised Loss Function. By applying a t-norm aggregation, we obtain clause valuations v(C k ). We then define a loss function that computes the total cost of the current assignment as:
L = m k=1 w k (1 -v(C k )) 2 m k=1 w k (3
)
where m is the total number of clauses. The equation is a weighted mean squared error of how unsatisfied each clause is, with the weights corresponding to the given weights for each clause. This loss function forces the model to train to minimize the total weight of unsatisfied clauses, without requiring any sort of ground truth labels, with the assumption that a cost of 0 (all clauses satisfied) is the optimal solution.
this section cite: []

Section: SGAT Layer: Clause to Variable
To efficiently learn which variables to update, we employ an attention mechanism and a messagepassing function that update the valuations of variables based on the valuations of the clauses to which they are connected (Figure 3b). Specifically, we define the update message m U ij and attention message m A ij between variable x i and clause C j-n as follows:
m U ij = v(x i ) + 1 -v(C j-n ) if i ∈ N + j-n v(x i ) -1 -v(C j-n ) otherwise , m A ij = m U ij if i ∈ N + j-n 1 -m U ij otherwise
The update messages m U ij represent the valuation the clause requests the variable to move toward, as shown in Figure 3a. The attention messages m A ij represent the strength of this request by flipping the valuation for negatively appearing literals, aligning all messages in a common direction regardless of polarity. Using these messages, Equations ( 1) and ( 2) are redefined as:
ϵ ij = LeakyReLU a T Wm A ij ∥W e e ij v ′ (x i ) = j∈Ni α ij m U ij
From the equation, as long as the input valuations v(x i ) ∈ [0, 1], α ij ∈ [0, 1], and j∈Ni α ij = 1, the outputs v ′ (x i ) ∈ [0, 1] are guaranteed to hold. This eliminates the need for additional activation functions after each layer, simplifying the computation of both variable assignments and the loss function. Additionally, we clamp the values of update messages to range [0, 1] whenever they leave the range (possible with certain t-norms). Extensive information on how SGATs handle input-output dimensions is provided in Appendix A.
SGATs are approximations of greedy distributed local search; they perform local search in a fully parallel and greedy manner. After computing clause valuations v(C), they send messages (m U ij ) to all connected variable nodes (greedy) in parallel, with the strength proportional to how unsatisfied the clause valuation remains (m A ij ). The variables then use the attention mechanism explained above to decide which clause messages to prioritize, to maximally satisfy those requests from clauses. SGATs learn these distributed local search heuristics through attention, which lead to the best approximations.
Multi-head attention. Velickovic et al. [24] further employed multi-head attention to stabilize the learning process of self-attention for GATs. We also employ the same strategy by computing K independent attention coefficients and averaging the outputs of all heads. As the outputs for each head are guaranteed to be in range [0,1], the final outputs here are also guaranteed to be in range [0,1].
this section cite: ['b25']

Section: Normalization Layer
To prevent feature values from being overly concentrated or dispersed, we apply a normalization layer based on the sigmoid function, as shown in the bottom right of Figure 1. The normalized feature for node h ′ i ∈ [0, 1] is computed as:
h ′ i = σ γ (2h i -1 -k)
where σ(•) denotes the sigmoid function, γ a parameter for controlling the spread of the values, and k ∈ [-1, 1] a parameter for controlling the center of the values. The input feature h i ∈ [0, 1] is first scaled to the [-1, 1] range, after which the sigmoid modulates how close the output is to 0 or 1.
this section cite: []

Section: Approximation Ratio of SGATs
In previous works [17,25], the capabilities of differentiable MaxSAT solvers were evaluated theoretically using the approximation ratio, which is the lower bound of the ratio of the value computed with the algorithm to the optimal value. In [17], a 1/2-approximation ratio for unweighted MaxSAT is achieved by GNN architectures with hidden dimension sizes dependent on the number of clauses. We can prove that SGATs achieve a 1/2-approximation for any unweighted Max-EkSAT problem (problems with exactly k literals per clause), and that this guarantee holds with fixed architectural settings independent of the number of clauses. This result establishes a deterministic baseline that helps us understand the theoretical capabilities of SGATs. The details are provided in Appendix E.
this section cite: ['b18', 'b26', 'b18']

Section: Local Search with GNNs
To solve MaxSAT problems with SGATs, we introduce LS-GNN (Local Search with GNN, Algorithm 1), a novel local search solver based on the continuous optimization of SGATs. LS-GNN takes as input a weighted MaxSAT instance F and returns the best valuation v best and cost C best found for a given problem within a time limit. Starting from a random valuation v, SGAT maps the current valuations of variables and clauses to valuations that aim to maximize clause satisfaction. We iteratively optimize SGATs to minimize the loss in Equation (3), moving the valuations toward better solutions.
this section cite: []

Section: Algorithm 1: LS-GNN
In : Weighted MaxSAT instance F, timeout T . Out : Best valuation v best and its cost C best . 1 v best = ∅, C best = +∞, v = random assignment 2 while elapsed time < T do 3 k = 0, k local = 0, v local = ∅, C local = +∞ 4 while early stopping ≥ k -k local do 5 optimize SGAT with respect to v 6 C = cost F, SGAT(F, v) 7 if C < C local then 8 v local = SGAT(F, v), k local = k, C local = C 9 end 10
k = k + 1 11 end 12 if C local < C best then 13 v best = v local , C best = C local 14 end 15 v = (1 -β) • v + β • ∆ 16 end
To reduce the risk of the algorithm being stuck in a local optimum, we periodically randomize a subset of valuations (L15). This is done by sampling a binary mask β via thresholding a uniform random distribution, and injecting new random valuations ∆ for the masked variables. The randomization procedure is triggered whenever the best cost stops improving for early stopping epochs (L4). The parameters early stopping, elapsed time, β, and ∆ control the termination and randomization behavior of LS-GNN; their specific values are provided in Appendix A.
While Algorithm 1 used SGATs as the GNN model, it can be used with any arbitrary model that is able to output assignment predictions. For simplicity, we refer to the LS-GNN with SGATs and GATs as LS-SGAT and LS-GAT, respectively.
this section cite: []

Section: Experiments
In this section, we conduct multiple experiments to answer the following questions regarding SGATs: Q1) Are SGATs better than existing GNN architectures? Q2) What component makes our model efficient? and Q3) How can LS-GNN be compared with other continuous solvers? For evaluation, we use instances used in non-partial unweighted and weighted benchmark instances provided in MaxSAT evaluationsfoot_2 , which we denote as MS and WMS (and collectively referred to as WMS+). We also prepare a subset of these datasets with instances that are below specific file sizes such as 2MB for purposes such as training, and denote as WMS+(2MB). The training and testing splits are shown in Appendix D.
this section cite: []

Section: Model Architecture
To evaluate the effectiveness of our model in learning to solve practical instances (MaxSAT Evaluation benchmark instances), we first conduct experiments using different model architectures. Provided that there is only one work regarding end-to-end MaxSAT solving with GNNs [17], we also compare with models that were built for SAT. Specifically, we use NeuroSAT [22] and GGNN [14] with unsupervised loss functions presented in [19], as they were shown to empirically work well [15]. Specific details regarding models are given in Appendix B.
We compare the model performance on the MS2018(2MB) dataset, with training done on two different datasets: (i) MS2018(2MB), and (ii) SR(U(40,200)), a randomly generated dataset with 40   to 200 variables [22]. We use the satisfied weight ratio as a performance metric, which is defined as:
Satisfied Weight Ratio = m k=1 w k • round v G (C k ) m k=1 w k , where v G (C k ) ∈ [0, 1]
is the valuation of clause C k computed using the Gödel t-norm, w k is the weight of clause C k , and m is the total number of clauses. This score represents the ratio of the total weight of satisfied clauses to the total weight of all clauses, with a higher score indicating a strictly better solution.
We used SGATs with 6 SGAT blocks composed with Gödel T-norm layers and SGAT layers with 2 attention heads and 4 channels. For training, we used the Adam optimizer with a learning rate of 2 × 10 -3 , and a batch size of 4. For existing models, the default provided settings were used. The number of blocks are generally chosen to strike the best balance between performance and computational efficiency, with the specific empirical results shown in Appendix D.
this section cite: ['b18', 'b23', 'b15', 'b20', 'b16', 'b23']

Section: Results
As shown in Figure 4, our model is able to learn to solve from practical instances, while others completely fail to do so. Given that the training dataset consists of problems from various domains, we can infer that SGATs are capable of learning to solve a wide range of problems. Figure 4b further supports this conclusion; even with synthesized datasets, our model is able to learn a distributed local search heuristic that has high performance on practical (non-synthesized) datasets. Overall, this shows our models' dominant strength in solving MaxSAT problems. Additionally, the scalability difference between existing models and SGATs is shown to be huge, due to the large difference in parameter numbers. For the training with MS2018(2MB), existing architectures could only handle a batch size of 1. In contrast, SGATs could support batch sizes of up to 4, contributing to the high stability of our model. Extensive results are shown and discussed in Appendix D.
this section cite: []

Section: Ablation Study
To highlight which components of our model yields the highest performance increase, we performed ablation studies focusing on two points: (a) SGAT and T-norm layer, and (b) types of t-norm. For (a), we consider two variants of our model, one with SGATs swapped to GAT with Sigmoid, and another swapping out the T-norm layers with GATs. As SGATs are dependent on T-norm layers being used, we do not experiment with T-norm layers swapped with SGATs. For (b), we compare using three different fundamental t-norms that have been frequently used for machine learning: Gödel, Product, and Łukasiewicz. The experiments were all performed with WMS+2018(2MB) as the training and testing set, with the same model parameters as the previous experiment.
this section cite: []

Section: Results
Figure 5 shows the results for each ablation study. From the first ablation study, we can see that our current model architecture achieves the best performance, with significantly high stability. Furthermore, we can see that having the T-norm layer for clause value updates in place of a GAT layer highly increases the stability, even though the total number of trainable layers essentially halve.
From the second ablation study, we can observe that Gödel and Product t-norms work substantially better than Łukasiewicz. This is thought to be because Łukasiewicz outputs 0 when the input values are too low, making the gradients zero in certain regions, hindering the training process. On the other hand, the other two t-norms do not have this issue, resulting in more stable and effective optimization. Additionally, we can observe that Gödel converges slightly faster than Product. This is thought to be due to the compatibility between SGAT layers and Gödel t-norm; as it is guaranteed that no variable will have higher values than the clause it is connected to, the update message never goes out of the range [0,1], removing the need of any clamping procedures.
this section cite: []

Section: Local Search with SGATs
In our second experiment, we tested the performance of our solvers based on continuous optimization at approximating solutions on both unweighted and weighted benchmark instances. We used LS-GNN with GATs and SGATs pretrained in the previous experiment (as LS-GAT and LS-SGAT) as backbone models. We compared our solvers to three existing solvers based on continuous optimization: the Mixing method [25], the MIXSAT algorithm [26], and FourierSAT [12]. The default settings were used for all solvers. As the only continuous solver that addresses weighted MaxSAT is FourierSAT, we conduct experiments comparing our model only with FourierSAT on WMS instances. We specifically make implementation changes to support weights, as the original does not do so at default.
To evaluate the approximations computed by each solver, we calculated the incomplete scores, a metric that has been used to evaluate incomplete MaxSAT solvers in recent years of MaxSAT evaluations. The incomplete scores are defined as score(s, i) = (1 + cost of best known cost for i) (1 + cost of solution for i found by s) ,
where i is the instance and s is the solver used. For reference, we retrieve the best known cost for each instance from the official MaxSAT Evaluation results. We combine this with the best found cost for each dataset and timeout setting, keeping it comparable with existing methods. However, we note that due to our solvers finding better solutions than the competition results, the best known cost is not completely identical to the official results.
this section cite: ['b26', 'b27', 'b12']

Section: Results
Table 1 shows the average incomplete score for each solver depending on the year of evaluation. From the results, we can see that LS-SGAT clearly outperforms every other existing solver based on continuous optimization, with an average improvement of 0.055 for MS, and 0.09 for WMS versus the second best solver. However, we can also observe that the differences are not constant for every year. This is analyzed to be due to the types of problem each year contains; some solvers perform better on specific instances than others. Nevertheless, we can confirm that the overall, SGATs perform significantly better on a wide range of problems compared to existing continuous methods.
Another important factor that we observed was that the size of problems the algorithm was able to handle was much different. While SGATs were able to handle all but one instance, others went over the timelimit on tens to up to hundreds of instances, showing that our model and algorithm scales much better than existing approaches. This is mainly due to the efficiency of GNN architectures; GNNs tend to have much fewer parameters, which are independent of the problem size. We further analyze this with more experimental results in Appendix D.
this section cite: []

Section: SGATs as Initialization Heuristics.
While our primary concern is to develop a differentiable method for MaxSAT solving, it is worth investigating how SGATs can predict good assignments for a given Weighted MaxSAT instance and whether such a predicted assignment can be used as an initial assignment for a SOTA solver. In this context, we have conducted an additional experiment where the predictions of SGATs are used as initialization heuristics for state-of-the-art incomplete solvers. The results were positive, with incomplete solvers being able to achieve substantially higher incomplete scores when combined with SGATs, supporting our claim that SGATs can work equally well in practical settings. The extensive results are shown in Appendix C.
this section cite: []

Section: Conclusion and Future Work
We presented SGATs as novel GNNs that utilize attention and message passing mechanisms that operate on t-norms. SGAT layers approximate greedy distributed local search, with their heuristics being the main learnable component. To demonstrate the effectiveness of our model, we further developed a continuous local search algorithm that is built on top of SGATs to solve given MaxSAT instances in a continuous manner. Experimental results showed that SGATs train in a highly stable manner, with approximations clearly outperforming those produced by existing neural-based architectures. Our model also outperforms state-of-the-art continuous solving approaches, demonstrating the strength of our model to output good approximations, even against theoretically sound approaches. Future works will be focused on expanding the current framework to support partial MaxSAT problems, which require the satisfaction of hard clauses, and is known to be difficult for Neural Networks. Another interesting direction would be to incorporate SGATs into machine learning systems for tasks such as recognition and prediction. Especially in neuro-symbolic systems where MaxSAT solvers are used [9], SGATs can easily replace the solvers to allow for end-to-end learning.
this section cite: ['b9']

Section: References
Ref_id:b0 Title: Learning To Solve Circuit-SAT: An Unsupervised Differentiable Approach Year: (2019)
Ref_id:b1 Title: Old techniques in new ways: Clause weighting, unit propagation and hybridization for maximum satisfiability Year: (2020)
Ref_id:b2 Title: Combinatorial Optimization and Reasoning with Graph Neural Networks Year: (2023)
Ref_id:b3 Title: Predicting Propositional Satisfiability Based on Graph Attention Networks Year: ()
Ref_id:b4 Title: NuWLS: Improving Local Search for (Weighted) Partial MaxSAT by New Weighting Techniques Year: ()
Ref_id:b5 Title: Neurosymbolic AI: the 3rd wave Year: (2023)
Ref_id:b6 Title: Fast Graph Representation Learning with PyTorch Geometric Year: (2019)
Ref_id:b7 Title: A Deep Reinforcement Learning Heuristic for SAT based on Antagonist Graph Neural Networks Year: ()
Ref_id:b8 Title:  Year: (2022)
Ref_id:b9 Title: ROAD-R: the autonomous driving dataset with logical requirements Year: (2023)
Ref_id:b10 Title: Revisiting the Learned Clauses Database Reduction Strategies Year: (2018)
Ref_id:b11 Title: Can Q-Learning with Graph Networks Learn a Generalizable Branching Heuristic for a SAT Solver? Year: (2020)
Ref_id:b12 Title: FourierSAT: A Fourier Expansion-Based Algebraic Framework for Solving Hybrid Boolean Constraints Year: ()
Ref_id:b13 Title:  Year: (2020)
Ref_id:b14 Title: Graph Neural Networks Meet Neural-Symbolic Computing: A Survey and Perspective Year: (2020)
Ref_id:b15 Title: Gated Graph Sequence Neural Networks Year: (2016)
Ref_id:b16 Title: G4SATBench: Benchmarking and Advancing SAT Solving with Graph Neural Networks Year: (2024)
Ref_id:b17 Title: Optimizing local search-based partial MaxSAT solving via initial assignment prediction Year: (2024)
Ref_id:b18 Title: Can Graph Neural Networks Learn to Solve MaxSAT Problem? Year: (2021)
Ref_id:b19 Title: GNN Based Extraction of Minimal Unsatisfiable Subsets Year: ()
Ref_id:b20 Title: Goal-Aware Neural SAT Solver Year: (2022)
Ref_id:b21 Title: Neuro-symbolic artificial intelligence Year: (2021)
Ref_id:b22 Title: Guiding High-Performance SAT Solvers with Unsat-Core Predictions Year: (2019)
Ref_id:b23 Title: Learning a SAT Solver from Single-Bit Supervision Year: (2019)
Ref_id:b24 Title: One Model, Any CSP: Graph Neural Networks as Fast Global Search Heuristics for Constraint Satisfaction Year: ()
Ref_id:b25 Title: Graph Attention Networks Year: (2018)
Ref_id:b26 Title: Low-Rank Semidefinite Programming for the MAX2SAT Problem Year: (2019)
Ref_id:b27 Title: The Mixing method: coordinate descent for low-rank semidefinite programming Year: (2017)
Ref_id:b28 Title: SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver Year: (2019)
Ref_id:b29 Title: Deep Weighted MaxSAT for Aspectbased Opinion Extraction Year: (2020)
Ref_id:b30 Title: Learning to Solve Constraint Satisfaction Problems with Recurrent Transformer Year: ()
Ref_id:b31 Title: Learning Local Search Heuristics for Boolean Satisfiability Year: (2019)
Ref_id:b32 Title: NLocalSAT: Boosting Local Search with Solution Prediction Year: ()
Ref_id:b33 Title: BandMaxSAT: A Local Search MaxSAT Solver with Multi-armed Bandit Year: (2022-07)
Ref_id:b34 Title: Rethinking the Soft Conflict Pseudo Boolean Constraint on MaxSAT Local Search Solvers Year: (2024)
