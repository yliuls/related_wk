Title: Generalizable Reasoning through Compositional Energy Minimization
Abstract: Generalization is a key challenge in machine learning, specifically in reasoning tasks, where models are expected to solve problems more complex than those encountered during training. Existing approaches typically train reasoning models in an end-to-end fashion, directly mapping input instances to solutions. While this allows models to learn useful heuristics from data, it often results in limited generalization beyond the training distribution. In this work, we propose a novel approach to reasoning generalization by learning energy landscapes over the solution spaces of smaller, more tractable subproblems. At test time, we construct a global energy landscape for a given problem by combining the energy functions of multiple subproblems. This compositional approach enables the incorporation of additional constraints during inference, allowing the construction of energy landscapes for problems of increasing difficulty. To improve the sample quality from this newly constructed energy landscape, we introduce Parallel Energy Minimization (PEM). We evaluate our approach on a wide set of reasoning problems.Our method outperforms existing state-of-the-art methods, demonstrating its ability to generalize to larger and more complex problems.

Section: Introduction
Being able to solve complex reasoning problems, such as logical reasoning, combinatorial puzzles and symbolic manipulation, is one of the key challenges in machine learning. This is particularly interesting because it requires models to go beyond pattern recognition. For a model to successfully perform reasoning tasks, it is expected to be able to generalize to unseen distributions during test time. That is, they are expected to solve not only problems similar to those encountered during training, but also to be able to generalize to novel conditions and distributions [9].
The standard paradigm in machine learning for solving reasoning tasks is to train models end-to-end to map inputs to outputs. During training, models are exposed to a large number of solutions and learn statistical heuristics that allow them to solve similar problems. This contrasts with human reasoning, where we first learn the rules and constraints governing a problem, and then apply them in a compositional manner to arrive at a solution. Notably, humans are able to solve such problems without having seen an exact solution before. Moreover, when prompted with harder tasks, we can invest more time, effectively searching for a solution, rather than relying on heuristics [1,37].
In this work, we present an approach to reasoning where the overall process is cast as an optimization problem [22]. Specifically, we learn an energy function E θ (x, y) across all possible solutions y of the problem, where x are the given conditions of the problem. This energy function is learned such that valid solutions are assigned lower energy, while invalid solutions receive higher energy.
Train Evaluation Z0Z0Z0Z0 ZqZ0Z0Z0 0Z0Z0Z0Z Z0Z0Z0Z0 0Z0Z0Z0Z Z0Z0Z0Z0 0Z0Z0Z0Z Z0Z0Z0Z0 0Z0Z0Z0Z Z0Z0Z0Z0 0Z0Z0l0Z Z0ZqZ0Z0 0Z0Z0ZqZ l0Z0Z0Z0 0Z0Z0Z0l ZqZ0Z0Z0 0Z0ZqZ0Z Z0l0Z0Z0
Figure 1: Compositional Generalizable Reasoning. We formulate reasoning as optimization problem with inputs x and solutions y. By combining multiple optimization objectives, we can generalize to larger problem instances (bottom) than those seen during training (top). This enables us to solve a complex instance of N-queens (left) or a more complex instance of graph coloring (right).
Reasoning corresponds to minimizing the energy function to find low-energy solutions. Harder problems can be solved by spending more time reasoning and minimizing the optimization objective.
To solve more complex problems than those seen during training (see Figure 1), we can jointly reason and minimize the sum of several optimization objectives at the same time. For instance, in logical reasoning, we can learn the energy landscape of individual clauses, and then optimize over multiple clauses simultaneously. This composition of energy objectives enables the model to find assignments that satisfy all clauses simultaneously, enabling it to solve larger, more complex problems. However, combining multiple objectives makes the landscape increasingly complex and introduces local minima, making it hard to optimize. To address this, we propose a parallel strategy where we use a system of particles for optimization. In this setup, we leverage the energy function as a resampling mechanism to improve the quality of the samples and avoid local minima that attract particles. This approach improves exploration and ultimately makes optimization more effective.
We illustrate the applicability of our approach across a set of difficult reasoning problems, including the N-Queens, 3-SAT and the Graph Coloring. We compare against domain-specific state of the art combinatorial optimization models, and show that our approach outperforms them in terms of solution quality and generalization to larger and more complex problems. We further show that, by adjusting the computational budget, we can adapt the model to solve more complex problems. Finally, ablation studies show that our training strategy leads to improved results and that our sampling strategy is able to produce better quality solutions than existing samplers.
Overall, the contributions of this work are threefold. First, we propose a compositional approach to reasoning generalization, where we combine energy landscapes during inference to solve more complex problems. Second, we introduce a new sampling strategy, Parallel Energy Minimization (PEM), a particle-based optimization strategy that enables us to effectively optimize composed energy functions to solve hard reasoning tasks. Finally, we illustrate the efficacy of our approach empirically across a wide set of reasoning tasks, outperforming many existing combinatorial optimization approaches in generalization.
this section cite: ['b8', 'b0', 'b36', 'b21']

Section: Related work
Reasoning as Optimization. Reasoning includes multiple cognitive processes, such as logical inference, decision-making, planning, and scheduling. Many of these can be formulated as optimization problems, where the goal is to find variable assignments that minimize an objective function under certain constraints. Prior works have integrated logical reasoning into neural networks through differentiable relaxations of SAT [26] and MAXSAT solvers [64], differentiable theorem proving [53,47], probabilistic logic [45], differentiable logic rules [59], and gradient-based methods for discrete distributions [49]. Other approaches incorporate continuous optimization directly into models via differentiable convex [2], quadratic [5] or integer solvers [61]. These methods, however, often target specific domains or rely on strong assumptions.
Another line formulates reasoning using general-purpose optimization frameworks. For instance, [54,15] simulate physical dynamics using energy minimization. Latent space optimization has been used in variational methods for molecule generation [28] and the Traveling Salesman Problem (TSP) [35]. More related to our work, [22] learns energy functions backpropagating through optimization steps or with diffusion-based losses [23]. Nonetheless, most of these approaches adopt end-to-end methods to learn reasoning tasks, limiting their generalization ability. We instead propose a compositional strategy, combining energy landscapes learned on subproblems to tackle larger tasks.
Finally, recent approaches focus on learning-based methods for Combinatorial Optimization (CO), which aim to reduce the computational cost by generating near-optimal solutions. Graph Neural Networks (GNNs) are the standard in this domain due to their ability to represent variable-constraint relations. Recent works have employed GNNs to directly predict solutions [10,36,55], learning discrete diffusion over graphs [60,42], using reinforcement learning [8,3] or learning Markov processes [72]. However, it is well known that GNNs struggle out of distribution [67,27] and require large, diverse datasets. Our approach leverages the compositional nature of reasoning problems by producing more generalizable energy landscapes combining multiple energy objectives.
Reasoning as Iterative Computation. Some strategies use neural networks to iteratively refine solutions to reasoning problems. This motivation is drawn from optimization solvers, which operate with iterative updates. Within this category, we can identify three main directions: (1) methods that incorporate explicit program representations [30,52,12,70,48], (2) works based on recurrent neural networks [29,38,13,69,18,56,70], and (3) techniques that approximate solutions via iterative refinement [58,7,43,42,64,44]. In our work, we cast reasoning problems as optimization problems, hence we use optimization algorithms as refinement steps for solution search.
this section cite: ['b25', 'b63', 'b52', 'b46', 'b44', 'b58', 'b48', 'b1', 'b4', 'b60', 'b53', 'b14', 'b27', 'b34', 'b21', 'b22', 'b9', 'b35', 'b54', 'b59', 'b41', 'b7', 'b2', 'b71', 'b66', 'b26', 'b29', 'b51', 'b11', 'b69', 'b47', 'b28', 'b37', 'b12', 'b68', 'b17', 'b55', 'b69', 'b57', 'b6', 'b42', 'b41', 'b63', 'b43']

Section: Energy-Based Models and Diffusion Models.
Our work is closely related to Energy-Based Models (EBMs) [32,40,24]. Most of the work in this field has focused on learning probabilistic models over data [24,50,21,6,66,17]. In contrast, we train an EBM for solving reasoning tasks by performing optimization over the learned energy landscape.
this section cite: ['b31', 'b39', 'b23', 'b23', 'b49', 'b20', 'b5', 'b65', 'b16']

Section: Method

this section cite: []

Section: Reasoning as Energy Minimization
Let D = {X, Y } be a dataset of reasoning problems with inputs x ∈ R O and solutions y ∈ R M . We wish to find an operator f (•) that can generalize to test problems f (x ′ ) where x ′ ∈ R O ′ , is potentially larger and more complex than x. Let E θ (x, y) : R O × R M → R, be an EBM defined across all possible solutions y given x, such that ground-truth solutions y are assigned lower energy. Finding a solution to the reasoning problem corresponds to finding an assignment ŷ such that:
ŷ = arg min y E θ (x, y)(1)
To find the solution ŷ, one can use gradient descent:
y t = y t-1 -λ∇ y E θ (x, y t-1 )(2)
where λ is the step size, and yfoot_0 is the initial solution drawn from a fixed noise distribution (e.g. Gaussian). The resulting solution y T is found after T iterations of the above update.
this section cite: []

Section: Diffusion Energy-Based Models.
The effective training of EBMs is a challenging task, and currently, many approaches exist in the literature for this purpose [24,11,20]. Previous works trained EBMs by backpropagating the gradient through T generative steps [22]. However, this could lead to instabilities in the training and high computational cost of backpropagation.
In this work, we propose instead to use the denoising diffusion training objective introduced in [20]. Specifically, we train the gradient of the EBM to match the noise distribution at each timestep t. Formally, given a truth label y from the dataset, and a gaussian corrupted label y * , where y * = √ 1 -σ t y + σ t ϵ and ϵ ∼ N (0, I) we can define the diffusion objective as:
L MSE (θ) = E y,ϵ∼N (0,I) ∥ϵ + σ t ∇ y E θ (y * , t)∥ 2(3)
with E θ (y * , t) being a explicitly defined scalar function 0 , and σ t a sequence of fixed noise schedules.
This formulation allows us to supervise the gradient of the energy function at each optimization step t, avoiding the need to backpropagate through a sequence of T steps. As a result, we learn an energy gradient that transforms a noisy input into the target distribution, through a series of optimization steps. To generate outputs, we can then use, for example, the update rule given in (2).
this section cite: ['b23', 'b10', 'b19', 'b21', 'b19']

Section: Shaping the Energy landscape.
The training objective presented in Eq. ( 7) does not guarantee that the target label y is assigned the energy minima of the energy landscape. In this work, to enforce that the energy minima align to the ground-truth label, and to enhance regions of the landscape not covered by the diffusion-based training, we follow the approach of [23], and introduce an additional contrastive loss function to shape the energy landscape.
This contrastive loss guides the energy function by comparing noise-corrupted labels of given pairs of positive and negative samples. Formally, the objective at step t is formulated as:
L CL (θ) = -log e E + e E + + e E -(4)
where E + = E θ (ỹ + , t) and E -= E θ (ỹ -, t), with ỹ+ and ỹbeing positive and negative noise corrupted samples respectively, this is, ỹ+ = √ 1 -σ t y + + σ t ϵ and ỹ-= √ 1 -σ t y -+ σ t ϵ.
this section cite: ['b22']

Section: Compositional Reasoning
We wish to construct a reasoning framework that can generalize to complex problems that are much harder than those seen at training time, consisting of a significantly greater number of constraints.
To construct an effective energy function to tackle such problems, we propose to decompose the energy function into smaller ones that are defined over tractable subproblems that have been seen before. These subproblems are then simpler to handle and represent with energy functions compared to trying to solve the original problem. In particular, we propose decomposing a full reasoning problem x into simpler subproblems x = {x 1 , . . . , x N }, such that finding a solution y i to each subproblem x i solves the original problem x.
Given this decomposition, let E k θ (x, y) be an EBM of the k-th subproblem, where y k is assigned the lowest energy when it is a solution to subproblem x k . A complete solution ŷ to the original problem x is obtained by solving all subproblems simultaneously, formally optimizing the composition of each energy function:
ŷ = arg min y N k=1 E k θ (x k , y k ) (5
)
where ŷ can be found as in (2). We illustrate how to effectively optimize these objectives next.
this section cite: ['b1']

Section: Improving Sampling with Parallel Energy Minimization
Optimization over EBMs can be done through approximate methods such as Markov Chain Monte Carlo (MCMC). MCMC simulates a Markov chain, starting from an initial state y 0 , drawn from a noise distribution, with subsequent samples generated from a transition distribution. A common approach to MCMC sampling in EBMs is Unadjusted Langevin Dynamics (ULA) [24,50], which is defined as:
y t = y t-1 -λ∇ y E θ (x, y t-1 ) + √ 2λξ, ξ ∼ N (0, 1)(6)
where λ is the step size of the optimization method. This essentially corresponds to performing gradient descent on the energy function with some added noise.
However, such a noisy optimization procedure will often become stuck in local minima, which are especially prevalent in composed energy landscapes such as Equation 5. In MCMC, a common technique to more effectively sample from such difficult probability distributions is Sequential Monte Carlo (SMC) [19], where a set of parallel particles is maintained and evolved over iterations of sampling to help prevent premature convergence to local minima.
Inspired by this insight, we propose Parallel Energy Minimization (PEM), a parallel optimization procedure for optimizing composed energy landscapes. We initialize and optimize a parallel set of P particles across the T steps of optimization as presented in Algorithm 1 and illustrated in Figure 2. At each optimization step, PEM resamples particles based on their energy values, allowing local minima to be discarded. To further help particles cover the entire landscape of solutions, we add a predefined amount of noise to each particle each time they are resampled.
Algorithm 1 Parallel Energy Minimization (PEM) Input: T optimization steps, P particles Given a set of particles {y (T ) i } P i=1 , y (T ) i ∼N (0, 1) for timestep t in T, . . . , 1 do ▷ Importance evaluation w (t) ← softmax(-E θ (y (t) , t)) ▷ Selection Resample y (t) based on weights w (t) ▷ Resampling ỹ(t) ← y (t) + σtξ ξ ∼ N (0, 1) ▷ Optimize solutions with gradient y (t-1) ← ỹ(t) + σt∇E θ (ỹ (t) , t) end for return x (0) E θ (t) E θ (t-1) y (t) Particles w (t) ỹ(t)
y (t-1)
Figure 2: PEM Sampling. At timestep t, particles y (t) are first resampled using weights w (t) derived from E θ (y (t) , t). Next, scheduled Gaussian noise is added to obtain a new set of resampled particles. Finally, the particles of the next timestep t-1 are generated optimizing the gradient of the energy function at time t.
this section cite: ['b23', 'b49', 'b18']

Section: Refinement of the Energy Landscape
Composing multiple energy objectives together can effectively generate complex energy landscapes suitable for solving larger problems. However, as the number of objectives increases, the energy landscape becomes increasingly complex, which can lead to inaccuracies in the overall energy function. In particular, minima might appear in the function that incorrectly assigns lower energy to invalid solutions.
To mitigate this issue, we propose a refinement strategy for the composed landscape. Given a set of N energy functions {E k θ (x k , y k )} N k=1 , each trained on a subproblem x k , we refine the composed energy function using ground-truth solutions y. The resulting training objective is defined as:
L MSE (θ) = E y,N (ϵ,0,I) ∥ϵ + σ t ∇ y N k=1 E k θ (y * , t)∥ 2(7)
having y * = √ 1 -σ t y + σ t ϵ and ϵ ∼ N (0, I). This refinement helps align energy minima with valid solutions, correcting inaccuracies and improving robustness.
4 Experiments 4.1 N-Queens Problem 0Z0Z0l0Z Z0ZqZ0Z0 0Z0Z0ZqZ l0Z0Z0Z0 0Z0Z0Z0l ZqZ0Z0Z0 0Z0ZqZ0Z Z0l0Z0Z0 E c1 θ E r1 θ E d8 θ + E total θ Figure 3: N-Queens Problem Composition.
To compose a row model to solve the N-queens problem, we add the energy of each row i (E ri θ ), each column j (E cj θ ), each diagonal k (E dk θ ) of the chessboard. We then sample from the resulting energy function E total θ to generate valid solutions.
this section cite: []

Section: Setup.
The N-queens problem involves placing N queens on an N ×N chessboard such that no two queens threaten each other, meaning no two queens can be placed in the same row, column or diagonal. We evaluate how well different methods can generate valid solutions to the problem. During training we use only one single instance of the N-queens problem for a given value N . In our approach, we use the N rows of a single instance to train, and then compose this model row-wise, column-wise, and diagonal-wise to form a 2D chessboard. That is, we train a model to generate a valid row and then reuse it simultaneously for rows, columns and diagonals (see Figure 3).
Baselines. We compare against existing baselines for neural combinatorial optimization solvers, for which the N-queens problem is represented as a graph, and the solution corresponds to a Maximum Independent Set (MIS). As baselines we include: a reinforcement learning approach, where the model learns to defer harder nodes when solving the problem (LWD, [3]), an unsupervised method, where the model learns a Markov decision process over graphs (GFlowNets, [72]), and a supervised categorical diffusion solver (DIFUSCO, [60]). Furthermore, we also compare against previous state
Model Type Correct Instances ↑ Size ↑ LWD RL + S 22 7.1000 ± 0.5744 GFlowNets UL + S 14 6.9293 ± 0.5904 DIFUSCO (T =50) SL + S 17 6.9400 ± 0.6452 Fast T2T (T S =1, T G =1) SL + S 21 6.8200 ± 0.8761 Fast T2T (T S =1, T G =1) SL + GS 12 6.7000 ± 0.7141 Fast T2T (T S =5, T G =5) SL + S 20 7.0600 ± 0.5800 Fast T2T (T S =5, T G =5) SL + GS 41 7.3800 ± 0.5436 EBM (P =1024) (Ours) SL + PEM 97 7.9699 ± 0.1714 Table 1: 8-Queens Problem Evaluation. We compare the performance against state-of-the-art combinatorial optimization models on the 8-queens solution generation task. All the models were trained with 1 single instance of the 8-queens problem. We sampled 100 8-queens solutions. IR: Iterative Refinement, BP: Belief Propagation, TS: Tree Search, S: Sampling, GS: Guided Sampling, 0Z0Z0l0Z Z0ZqZ0Z0 0Z0Z0ZqZ l0Z0Z0Z0 0Z0Z0Z0l ZqZ0Z0Z0 0Z0ZqZ0Z Z0l0Z0Z0 0Z0Z0l0Z Z0ZqZ0Z0 0Z0Z0ZqZ l0Z0l0Z0 0Z0Z0Z0l ZqZ0Z0Z0 0Z0ZqZ0Z Z0l0Z0Z0 Reverse diffusion of the art combinatorial optimization models (Fast T2T [42]), with different inference steps T s and gradient search steps T g . For the latter, we also compared against the guided sampling version, where a penalty function is added for the MIS problem to guide denoising.
T =100 T =75 T =50 T =25 T =10 T =4 T =3 T =2 T =1 Decoded Solution PEM (P=2) PEM (P=8)
For all the methods evaluated, we report the number of correct instances of the problem found, and the average number of queens placed in the chessboard. We follow previous works approach to decode solutions, where, given a model heatmap of the chessboard, we perform greedy decoding by sequentially placing queens in the board until a conflict is found.
Quantitative Results. We report the comparison of our approach with the previous baselines in Table 1. For all the methods reported, we sampled 100 different solutions. In this table, we can see that our approach is able to generate nearly all perfect solutions to the problem. Furthermore, our method significantly outperforms the previous state-of-the-art solvers. Out of 100 generated samples, 97 are valid 8-queens solutions, while state-of-the-art methods are able to find 41 correct instances at most.
this section cite: ['b2', 'b71', 'b59', 'b41']

Section: Qualitative Results.
In Figure 5 we visualize the sampling process of our approach using reverse diffusion and PEM. We can observe that PEM is able to generate much better quality samples than reverse diffusion. Additionally, more particles leads to the generation of valid solutions. We include an example of parallel sampling with 8 particles in Appendix B. In Figure 4 we can see that a higher energy is assigned to rows and columns where the constraints are violated.
Performance with Increased Computation. In Table 2 we report the performance of our approach on the 8-queens problem with increasing number of particles during sampling. We show that increasing the number of particles significantly improves the quality of the generated samples and, as a consequence, a larger number of correct instances are found. In Figure 6, we visualize the performance of our model with different number of particles on different complexity levels of the problem. We can see that by adjusting the number of particles, we can adapt the ability of our model to solve more difficult problems.   Table 4: Loss Ablation. Ablations proposed for the loss function on the performance on the 8-queens problem.
We sampled 100 solutions from the 8-queens problem. A combination of both a diffusion and contrastive loss to shape the landscape produces the best results on the task. In all cases we sampled using PEM (P =1024).
this section cite: []

Section: Ablation Study.
In Table 3, we compare the performance with different methods proposed for EBM sampling, including Unadjusted Langevin Dynamics (ULA), Metropolis Adjusted Langevin Dynamics (MALA), Unadjusted Hamiltonian Monte Carlo (UHMC) and Hamiltonian Monte Carlo (HMC). We show that our approach substantially outperforms existing samplers in the 4-queens task. In Table 4, we compare three models trained with different loss functions. We show that the combination of the diffusion and contrastive losses produces improved results on the task.
this section cite: []

Section: SAT Problem
Setup. In this section we evaluate the performance of our approach on the Boolean satisfiability problem (SAT), well known to be an NP-complete problem. The 3-SAT problem is a binary decision problem where a Boolean formula is given in Conjunctive Normal Form (CNF), with each clause having exactly 3 literals. The task is to find a truth assignment to the variables (true or false) such that the formula evaluates to true. For training, we generated random 3-SAT instances with number of variables within [10,20]. The number of clauses was set to be in phase transition, that is, it was set to be 4.258 × n, where n is the number of variables [57]. For our approach, we train a model to generate a satisfiable assignment to only one individual clause of the 3-SAT problem. We then compose the model to generate a solution to the entire problem. This enables the generalization to an arbitrary number of clauses. We evaluate using the SATLIB benchmark [34]. For a distribution similar to the training one, we used 100 instances with 20 variables and 91 clauses. For a larger distribution, we used 100 instances with 50 variables and 218 clauses.
Baselines. We compare against existing baselines for neural SAT solvers, including: the seminal neural SAT solver, where the solution is iteratively refined with increasing number of steps (NeuroSAT [58]), and the state-of-the-art neural solver based on belief-propagation (NSNet [43]). In both cases, we use different number of steps T for solution refinement. When feasible, we also compare with combinatorial optimization models by encoding the 3-SAT problem as a graph.
Quantitative Results. In Table 5 we find that our method significantly outperforms the previous state-of-the-art neural SAT solvers, and is able to find a larger number of correct instances of the problem. In the similar distribution 91 instances are solved compared to 58 instances solved by NSNet. In the larger distribution our method still outperforms the previous other methods with 43 correct instances, with NSNet solving 37 correct instances. Table 7: Fine-tuning Ablation. Ablations proposed for the finetuning of the model on the performance on the 3-SAT problem. We show that finetuning the composed model with complete instances leads to better performance. In all cases we sampled using PEM (P=1024).
Qualitative Results. In Appendix B, we present additional qualitative results for 3-SAT, where we show that unsatisfied clauses are assigned higher energy, while satisfied clauses are assigned lower.
Performance with Increased Computation. We assess in Table 6, the impact of the number of particles on the performance on the 3-SAT problem. We show that increasing the number of particles improves the results on both the similar and larger distributions. By formulating the problem as an energy minimization problem, we can adjust the number of particles to adapt to the difficulty of the task.
this section cite: ['b9', 'b19', 'b56', 'b33', 'b57', 'b42']

Section: Ablation Study.
In Appendix C we include ablation for the sampling procedure, showing that PEM outperforms existing methods, as well as ablations on the training loss. Moreover, in Table 7 we report that finetuning the composed model improves the overall performance.
this section cite: []

Section: Graph Coloring
Setup. In this section, we evaluate our approach on the graphical problem of graph coloring. Given a graph instance, the task is to assign a color to each node in the graph such that no two adjacent nodes share the same color using at most k colors. This problem is known to be NP-complete. The chromatic number χ of a graph is the minimum number of colors needed to color the graph. To train baselines, we followed the same approach as in [41], and generated random graphs with number of nodes within [20,40], density within [0.01, 0.5], and chromatic number χ within [3,8]. For our approach, we train a model to generate a valid coloring of an individual edge given a set of colors.
We then compose the model for all the edges of the graph to generate a valid coloring solution. To train our model we generate random pairs of different colors. For evaluation, we use graphs from the well-known COLOR benchmark 1 . Additionally, we also evaluate on random graph instances generated following different graph distributions, namely: Erdos-Renyi [25], Holme-Kim [33], and random regular expander graphs [4]. For each distribution, we generate smaller graphs with nodes within [20,40] and larger graphs with nodes within [80, 100]. Moreover, we also evaluate on densely connected regular graphs such as Paley graphs [51] and complete graphs. For all the methods, we We present an incorrect solution with two conflicting edges in (d) and the corresponding energy for each edge in (e). As expected, a higher energy is assigned to conflicting edges.
Distribution V E d χ GCN GAT XLVIN GNN-GCP EBM (Ours) (P=128) Erdos Renyi [20, 39] [29,76] 0.12 [3, 4] 46.80 ± 20.47 34.00 ± 11.55 25.00 ± 7.81 15.20 ± 4.32 8.60 ± 4.82 [81, 99] [193, 225] 0.05 [3, 4] 151.60 ± 12.09 130.20 ± 11.47 93.80 ± 31.12 53.80 ± 8.34 29.20 ± 8.05 Holme Kim [22, 34] [56, 92] 0.26 [4, 4] 74.00 ± 14.74 51.20 ± 10.03 29.00 ± 7.75 13.20 ± 7.46 10.60 ± 2.70 [86, 100] [398, 469] 0.10 [5, 6] 408.00 ± 26.40 253.20 ± 50.71 182.60 ± 24.73 55.20 ± 12.63 59.00 ± 3.74 Regular Expander [21, 40] [63, 120] 0.22 [4, 4] 87.60 ± 22.58 58.60 ± 12.44 29.00 ± 7.75 15.40 ± 6.65 11.00 ± 4.89 [86, 100] [184, 200] 0.23 [3, 3] 144.80 ± 6.90 118.80 ± 12.59 112.60 ± 10.97 141.60 ± 69.47 37.20 ± 4.71 Paley [19, 37] [171, 465] 0.80 [6, 10] 285.00 ± 117.70 239.20 ± 159.43 151.80 ± 92.13 91.20 ± 63.14 34.80 ± 20.27 Complete [8, 12] [36, 66] 1.00 [8, 12] 46.00 ± 15.04 46.00 ± 15.04 34.80 ± 16.42 30.00 ± 2.54 3.40 ± 1.14
Table 8: Graph Coloring Evaluation. We compare the performance against canonical GNNs and GNN-based methods for graph coloring on different random graph distributions and the COLOR benchmark. Performance is measured as the number of conflicting edges, with lower indicating better. For each distribution, we report the average over five instances. Our approach outperforms existing methods on most instances and generalizes better to larger and denser graphs. Here V= Nodes, E= Edges, d= Average density, χ= Chromatic number.
generate a coloring of the graph with k colors, where k is the chromatic number of the given graph, and report the number of conflicting edges in the generated solution.
Baselines. We compare against existing baselines for neural solvers for graph coloring that are trained to generalize to novel graph instances (GNN-GCP [41]). We also include a comparison with canonical Graph Neural Networks (GCN [39] and GAT [62]), and RL guided by Neural Algorithmic Reasoners (XLVIN [16]).
this section cite: ['b40', 'b19', 'b39', 'b2', 'b7', 'b24', 'b32', 'b3', 'b19', 'b39', 'b50', 'b40', 'b38', 'b61', 'b15']

Section: Quantitative Results.
We compare our approach with GNN-based methods for graph coloring. In Table 8, we report the performance on random graphs generated following different distributions and the average performance on the COLOR benchmark. We show that our approach is able to generalize to larger graphs and different distributions better than existing methods. The detailed performance on the COLOR benchmark can be found in Appendix B, where our approach significantly outperforms existing methods. Notably, while GNN-based methods show increasingly worse performance on larger graphs, our method maintains a good performance across scales.
Qualitative Results. We present in Figure 7 a graph instance with a valid coloring solution. We visualize the energy map and show that low energy is assigned to all the edges that compose the whole graph. We also show an incorrect solution with two conflicting edges. The energy of the two conflicting edges is higher than the non-conflicting ones.
Performance with Increased Computation. In Appendix B we report that increasing the number of particles from 8 to 1024 decreases the average number of conflicting edges from 15.0 to 8.0.
Ablation Study. We include in Appendix C ablations on the sampling procedure, which yields 8.0 conflicting edges on average with PEM compared to 12.3 edges with UHMC. Additionally, we ablate the training losses, where we obtain an average of 15.0 conflicting edges with diffusion loss only and 9.0 when using contrastive loss only.
this section cite: []

Section: Crosswords
Setup. In this section, we report the results on crosswords puzzle solving. Crosswords are word puzzles where letters are arranged in a grid, with words intersecting both horizontally and vertically. Each word is associated with a clue that provides a definition, context or hint for the answer. The goal is to fill the grid so that all words satisfy both the clues and the grid constraints. For our approach, we
P 1 P 2 P 3 P 4 P 5 P 6 P 7 P 8 y (50)
y (20)   y (10)   y (1) train a model to generate a valid word given precomputed embeddings of the corresponding hint. To solve a complete crossword, we compose horizontally and vertically the model to form the given grid. We evaluate on the Crosswords Mini Benchmark introduced in [71]. Baselines. We compare against different inference algorithms for Large Language Models, including: Standard Input-Output (IO), Chain of Thought (CoT) [65] and Tree of Thought (ToT) [71].
Quantitative Results. In Table 9 we compare our approach with various LLM inference methods. Our approach significantly outperforms both the IO and CoT baselines. Furthermore, it achieves performance competitive with ToT. While ToT attains a slightly higher average word success rate (60.0% vs. 50.5%), our method achieves a higher overall grid completion rate (80.4% vs. 78.0%).
Qualitative Results. Figure 8 shows the particles generated for a crossword across timesteps. It can be seen that, during the optimization process, different particles explore different solutions to the puzzle. In the end, the optimization algorithm successfully finds a valid solution to the crossword.
this section cite: ['b70', 'b64', 'b70']

Section: Limitations and Conclusion
Limitations. A limitation of our method is that it assumes a starting Gaussian distribution and models optimization as a sequence of Gaussian increments. Future work could explore non-Gaussian objectives and initializations that enable recurrent improvement of initial solutions. Another limitation is that, while our method excels on N-Queens and 3-SAT, there is still room for improvement in achieving optimal solutions for graph coloring. Further research could investigate alternative training strategies for EBMs to produce more accurate energy landscapes
this section cite: []

Section: References
Ref_id:b0 Title: Meta-reasoning: Monitoring and control of thinking and reasoning Year: (2017)
Ref_id:b1 Title: Differentiable convex optimization layers Year: (2019)
Ref_id:b2 Title: Learning what to defer for maximum independent sets Year: (2020)
Ref_id:b3 Title: Random cayley graphs and expanders Year: (1994)
Ref_id:b4 Title: Optnet: Differentiable optimization as a layer in neural networks Year: (2017)
Ref_id:b5 Title: Generalized energy based models Year: (2020)
Ref_id:b6 Title: End-to-end algorithm synthesis with recurrent networks: Extrapolation without overthinking Year: (2003)
Ref_id:b7 Title: Neural combinatorial optimization with reinforcement learning Year: (2016)
Ref_id:b8 Title: From machine learning to machine reasoning: An essay Year: (2014)
Ref_id:b9 Title: Combinatorial optimization and reasoning with graph neural networks Year: (2023)
Ref_id:b10 Title: Efficient training of energy-based models using jarzynski equality Year: (2023)
Ref_id:b11 Title: Compositional generalization via neural-symbolic stack machines Year: (2020)
Ref_id:b12 Title: Hierarchical multiscale recurrent neural networks Year: (2016)
Ref_id:b13 Title: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities Year: (2025)
Ref_id:b14 Title: Inferring relational potentials in interacting systems Year: (2023)
Ref_id:b15 Title: Xlvin: executed latent value iteration nets Year: (2020)
Ref_id:b16 Title: On energybased models with overparametrized shallow neural networks Year: ()
Ref_id:b17 Title: Neural logic machines Year: (2019)
Ref_id:b18 Title: An introduction to sequential monte carlo methods Year: (2001)
Ref_id:b19 Title: Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc Year: ()
Ref_id:b20 Title: Improved contrastive divergence training of energy based models Year: (2020)
Ref_id:b21 Title: Learning iterative reasoning through energy minimization Year: (2022)
Ref_id:b22 Title: Learning iterative reasoning through energy diffusion Year: (2024)
Ref_id:b23 Title: Implicit generation and modeling with energy based models Year: (2004)
Ref_id:b24 Title: On the strength of connectedness of a random graph Year: (1961)
Ref_id:b25 Title: Learning explanatory rules from noisy data Year: (2018)
Ref_id:b26 Title: Generalizing graph neural networks on out-of-distribution graphs Year: (2023)
Ref_id:b27 Title: Automatic chemical design using a data-driven continuous representation of molecules Year: (2018)
Ref_id:b28 Title: Adaptive computation time for recurrent neural networks Year: (2016)
Ref_id:b29 Title: Neural turing machines Year: (2014)
Ref_id:b30 Title: Deepseek-r1 incentivizes reasoning in llms through reinforcement learning Year: (2025)
Ref_id:b31 Title: Training products of experts by minimizing contrastive divergence Year: (2002)
Ref_id:b32 Title: Growing scale-free networks with tunable clustering Year: (2002)
Ref_id:b33 Title: Satlib: An online resource for research on sat. Sat Year: (2000)
Ref_id:b34 Title: Learning a latent search space for routing problems using variational autoencoders Year: (2021)
Ref_id:b35 Title: An efficient graph convolutional network technique for the travelling salesman problem Year: (2019)
Ref_id:b36 Title: Thinking, fast and slow. macmillan Year: (2011)
Ref_id:b37 Title: Neural gpus learn algorithms Year: (2015)
Ref_id:b38 Title: Semi-supervised classification with graph convolutional networks Year: (2016)
Ref_id:b39 Title: A tutorial on energy-based learning Year: (2006)
Ref_id:b40 Title: Graph colouring meets deep learning: Effective graph neural network models for combinatorial problems Year: (2019)
Ref_id:b41 Title: Fast t2t: Optimization consistency speeds up diffusion-based training-to-testing solving for combinatorial optimization Year: (2024)
Ref_id:b42 Title: Nsnet: A general neural probabilistic framework for satisfiability problems Year: (2022)
Ref_id:b43 Title: Self-supervised learning of iterative solvers for constrained optimization Year: (2024)
Ref_id:b44 Title: Deepproblog: Neural probabilistic logic programming. Advances in neural information processing systems Year: (2018)
Ref_id:b45 Title: [re] end-to-end algorithm synthesis with recurrent networks: Logical extrapolation without overthinking Year: (2022)
Ref_id:b46 Title: Learning reasoning strategies in end-to-end differentiable proving Year: (2020)
Ref_id:b47 Title: Neural programmer: Inducing latent programs with gradient descent Year: (2015)
Ref_id:b48 Title: Implicit mle: backpropagating through discrete exponential family distributions Year: (2021)
Ref_id:b49 Title: On the anatomy of mcmc-based maximum likelihood learning of energy-based models Year: (2020)
Ref_id:b50 Title: On orthogonal matrices Year: (1933)
Ref_id:b51 Title:  Year: (2015)
Ref_id:b52 Title: Advances in neural information processing systems Year: (2017)
Ref_id:b53 Title: Constraint-based graph network simulator Year: (2021)
Ref_id:b54 Title: Combinatorial optimization with physics-inspired graph neural networks Year: (2022)
Ref_id:b55 Title: Can you learn an algorithm? generalizing from easy to hard problems with recurrent networks Year: (2021)
Ref_id:b56 Title: Generating hard satisfiability problems Year: (1996)
Ref_id:b57 Title: Learning a sat solver from single-bit supervision Year: (2018)
Ref_id:b58 Title: Differentiable neuro-symbolic reasoning on large-scale knowledge graphs Year: (2023)
Ref_id:b59 Title: Difusco: Graph-based diffusion solvers for combinatorial optimization Year: (2023)
Ref_id:b60 Title: A differentiable integer linear programming solver for explanation-based natural language inference Year: (2024)
Ref_id:b61 Title: Graph attention networks Year: (2017)
Ref_id:b62 Title: Automated crossword solving Year: (2022)
Ref_id:b63 Title: Satnet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver Year: (2019)
Ref_id:b64 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b65 Title: Vaebm: A symbiosis between variational autoencoders and energy-based models Year: (2020)
Ref_id:b66 Title: How neural networks extrapolate: From feedforward to graph neural networks Year: (2020)
Ref_id:b67 Title: Qwen3 technical report Year: (2025)
Ref_id:b68 Title: Differentiable learning of logical rules for knowledge base reasoning Year: (2017)
Ref_id:b69 Title: Learning to solve constraint satisfaction problems with recurrent transformer Year: (2023)
Ref_id:b70 Title: Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems Year: (2023)
Ref_id:b71 Title: Let the flows tell: Solving graph combinatorial problems with gflownets Year: (2023)
