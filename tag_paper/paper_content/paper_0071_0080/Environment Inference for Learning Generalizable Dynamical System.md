Title: Environment Inference for Learning Generalizable Dynamical System
Abstract: Data-driven methods offer efficient and robust solutions for analyzing complex dynamical systems but rely on the assumption of I.I.D. data, driving the development of generalization techniques for handling environmental differences. These techniques, however, are limited by their dependence on environment labels, which are often unavailable during training due to data acquisition challenges, privacy concerns, and environmental variability, particularly in large public datasets and privacy-sensitive domains. In response, we propose DynaInfer, a novel method that infers environment specifications by analyzing prediction errors from fixed neural networks within each training round, enabling environment assignments directly from data. We prove our algorithm effectively solves the alternating optimization problem in unlabeled scenarios and validate it through extensive experiments across diverse dynamical systems. Results show that DynaInfer outperforms existing environment assignment techniques, converges rapidly to true labels, and even achieves superior performance when environment labels are available.

Section: Introduction
Data-driven approaches, especially neural networks, offer a powerful alternative or complement to traditional physics-based methods for understanding complex dynamical systems [4]. Neural network-based emulators are particularly valuable for their ability to provide fast, cost-effective approximations of complex simulations [9,22], making them especially useful in scenarios where the underlying physics are poorly understood or misinterpreted, or where external disturbances are difficult to model [43,34]. These emulators are adept at handling large sets of variables and solving problems that are challenging for conventional solvers. Recent advancements in deep learning, along with innovative methods for modeling temporal and spatio-temporal systems, have led to a significant increase in applications across various fields, ranging from simple Hamiltonian dynamics to more complex areas like fluid dynamics and climatology [32,7].
While recent advancements have shown promising results, they often rely on the assumption that abundant, static data are available to satisfy the independent and identically distributed (IID) hypothesis. However, this assumption is frequently violated in practice due to challenges in data collection, associated costs, and environmental changes driven by exogenous factors [24,26]. Recent work in dynamical systems addresses this by introducing a multi-environment setting, where trajectories follow distinct dynamics across environments. These studies developed generalization methods that learn a shared global component while accounting for environment-specific variations, avoiding the limitations of underperforming averaged models [43,17].
Env. Label A Station A Trajectories From A Env. Label B Station B ... Parameter Difference Required by OOD Methods to infer Similarities and Differences Unavailable ... ... ... Nevertheless, a key limitation of many generalization techniques is their reliance on partitioning datasets across distinct domains or environments, which are assumed to capture underlying variations. These environment labels enable algorithms to identify and exploit both similarities and differences across environments. However, obtaining such environment labels during training is often challenging due to data acquisition difficulties or privacy constraints. For example, in scientific research, data may be collected over time under uncontrolled or unknown conditions [42]. In ecological studies, critical environmental parameters such as temperature or rainfall may vary unpredictably or remain unrecorded. [2]. Similarly, when aggregating data from multiple sources, environment labels are frequently lost or omitted, a common issue in large public datasets [35]. Furthermore, in privacysensitive domains like healthcare, finance, or social networking, access to environment-specific information is often restricted [18]. These limitations highlight the need for generalization methods that do not depend on explicit environment labels.
To address the challenge of unknown environment labels, we propose a novel approach that infers environment specifications by leveraging the key insight that trajectories within the same environment share consistent dynamics and exhibit similar prediction losses under the same neural network. This inherent consistency enables us to automatically derive meaningful environment assignments directly from the training data. We introduce an environment inference objective designed for dynamical systems, which minimizes environment-specific prediction losses. Using fixed neural networks, we first infer environments and then iteratively refine these networks with the inferred environments, ultimately learning a generalizable dynamical system.
Our model identifies environment labels directly from mixed trajectories of dynamical systems, facilitating the training of off-the-shelf generalization algorithms in scenarios where such labels are absent. Importantly, our findings demonstrate that inferring environments from mixed sequence data can improve the performance of generalization strategies, even compared to cases where environments are manually assigned.
Our main contributions are as follows:
• We present the first investigation into the challenge of unlabeled environment conditions in the context of learning generalizable dynamical systems, and propose a general framework named DynaInfer that utilizes the prediction loss to accurately infer latent environment labels from mixed sequence datafoot_0 .
• We theoretically establish that our algorithm effectively solves the alternating optimization problem without requiring environment labels, demonstrating its capacity to discern heterogeneous environments and infer generalizable mechanisms.
• We examine the efficacy of DynaInfer through experiments in both in-domain settings and adaptation scenarios using three representative dynamical systems. Results confirm that the environment labels assigned by DynaInfer converge rapidly to the true labels.
The remainder of this paper is structured as follows. Section 2 clarifies the problem definition. Section 3 introduces our framework and provides the theoretical underpinnings. Section 4 details the experimental setup and discusses the results. Related work is reviewed in Section 5, and Section 6 concludes the paper.
this section cite: ['b3', 'b8', 'b21', 'b42', 'b33', 'b31', 'b6', 'b23', 'b25', 'b42', 'b16', 'b41', 'b1', 'b34', 'b17']

Section: Problem Definition

this section cite: []

Section: Dynamical Systems
We examine dynamical systems determined by unidentified differential equations evolving over time, expressed as,
dx t dt = f (x t )(1)
where t ∈ R is the time index within a time interval I = [0, T ], and x t is a time-variant state within a bounded set A. The evolution function f : A → T A maps x t to its temporal derivative in the tangent space T A and belongs to a class of vector fields F.
In this paper, we consider both ordinary differential equation (ODE) and partial differential equation (PDE). For ODEs, A ⊂ R d ; for PDEs, A represents a d ′ -dimensional vector field within a bounded spatial domain (such as 2D or 3D Euclidean space) denoted as S ⊂ R d ′ . The function f characterizes the data distribution of trajectories T . Trajectories initiated from x 0 ∼ p(X 0 ) are computed by integrating the derivatives: x t = x 0 + t 0 f (x u )du, ∀t ∈ I.
this section cite: []

Section: Multi-Environment Dynamical Systems Learning
In contrast to the standard expected risk minimization (ERM) framework, which assumes i.i.d. trajectories, the multi-environment learning problem involves learning trajectories from M different environments. In each environment e ∈ [M ] = {1, 2, . . . , M }, the trajectories are governed by unique differential equations described by function f e . Specifically, consider N trajectories {x 1 , x 2 , . . . , x N }, where each trajectory x i is associated with an environment e i ∈ [M ]. The dynamics of each trajectory x i are thus modeled by the differential equation dx i t /dt = f ei (x i t ). The set of environments for all trajectories is denoted by e = {e 1 , e 2 , . . . , e N } ∈ [M ] N .
In multi-environment learning, the goal is to enhance traditional ERM methods by exploiting both the commonalities and disparities across diverse environments. To this end, the dynamics is decomposed into two components: a global component shared across all environments, parameterized by θ, and an environment-specific component, parameterized by ϕ e for each environment e. The set of environment-specific parameters is denoted by ϕ = {ϕ e } e∈[M ] . Consequently, the dynamics of each trajectory x i are parameterized by both the universal and environment-specific parameters,
dx i t dt = h x i t ; θ, ϕ e .
This parametrization entails a decomposition that can be implemented either functionally or parametrically. The functional decomposition, expressed as h x i t ; θ, ϕ e = f θ (x i t ) + g ϕe (x i t ), distinguishes between a shared function f θ and an environment-specific function g ϕe [43]. Alternatively, the parametric decomposition integrates the environment-specific parameters directly, formulated as h x i t ; θ, ϕ e = f θ+ϕe (x i t ) [17]. Intuitively, the key ingredient for multi-environment learning is that θ should encapsulate the maximal shared dynamics, whereas ϕ e should exclusively reflect the unique characteristics of each environment e not described by θ. However, directly optimizing both parameters poses an ill-posed problem, often resulting in trivial solutions where the global component learns nothing meaningful. To counteract this, the regularization term Ω(ϕ e ) is introduced to effectively penalize ϕ e , thereby facilitating learning in the global component. Consequently, with the information about the environments e = {e 1 , e 2 , . . . , e N }, the loss function is given by,
R e (θ, ϕ) = N i=1 t∈I dx i t dt -h x i t ; θ, ϕ ei 2 2 dt + λ M e=1 Ω(ϕ e ).(2)
The first term evaluates the regression precision of the parameterized function h(•; θ, ϕ e ). The ground truth vector field (VF) is not explicitly known and derived from trajectory data. Using the learned VF, a simulated trajectory is generated and used to calculate the regression loss by referring to real trajectories during training. The term Ω(ϕ e ) serves as a regularization term for ϕ e , with λ controlling the intensity of the regularization.
this section cite: ['b42', 'b16']

Section: Environment Inference for Multi-Environment Learning
In many real-world scenarios, the environment label for a trajectory sample is unknown. We aim to infer an environment assignment for each sample that maximizes the model's generalization ability across different environments. To achieve this goal, we reformulate the learning objective into an optimization problem contingent on a specific environment assignment e. Specifically, our aim is to learn the environment assignment ê = {ê 1 , ê2 , . . . , êN } ∈ [M ] N for each trajectory to effectively optimize Equation (2). The overall objective is defined as follows:
ê * , θ * , ϕ * = arg min ê,θ,ϕ R ê(θ, ϕ).(3)
In this paper, we explore a particularly challenging scenario where the total number of training environments M is also unknown. We investigate the development of a practical model that maintains favourable performance even when the exact number of true environments is unknown.
this section cite: ['b1']

Section: The DynaInfer Framework
In this section, we introduce our framework that operates on field functions without prior domain knowledge, proving especially effective in dynamical systems where exogenous factors are unobserved and in situations where relevant environmental information is unclear or absent. While some clustering methods infer labels for CV data, they operate on finite-dimensional vectors in Euclidean space, which drastically differs from field functions, making them inapplicable.
The optimization challenge in Equation ( 3) is primarily due to the inherently discrete nature of the environment assignments ê, which take values in the set [M ]. This discrete categorization impedes the direct application of traditional gradient descent methods, which are typically designed for continuous parameter spaces. To effectively address this challenge, we develop a dual iterative strategy that concurrently updates the environment assignments ê and the model parameters θ, ϕ. The first step in our approach centers on inferring environment labels by analyzing the prediction errors of the trajectories output by the neural network during the current training round. This analysis serves as a diagnostic tool to uncover critical discrepancies that signify distinct dynamical environments. Following this, the second step entails refining the neural network parameters based on the newly inferred environment assignments in an unbiased manner, enabling the neural network to precisely adapt to the unique characteristics of each identified environment. Through this adaptive refinement, our model progressively enhances its accuracy and generalization capability across different dynamic settings. The complete method is detailed in Algorithm 1 and is visually depicted in Figure 2. Update θ (r) , ϕ (r) based on Equation (5) 6: end for 7: Output: θ, ϕ.
this section cite: []

Section: Environment Assignment

this section cite: []

Section: 𝑒 𝑟

this section cite: []

Section: Mixed Trajectories

this section cite: []

Section: Bias-aware Environment Assignment
The environment inference step receives a single dataset as input and generates a partition of the data into multiple environments. Intuitively, trajectories originating from the same environment adhere to consistent dynamics. Employing the same neural network to model these trajectories should yield similar estimation error across them, reflecting a coherence in their dynamic parameters.
Upon examination, we observe that the optimization framework defined in Equation (2) shares conceptual similarities with classical centroid-based clustering methodologies, although the latter generally operate in Euclidean space. In K-means clustering, the primary goal is to minimize the within-cluster sum of squares, often referred to as cluster inertia [37]. This minimization effort concentrates on reducing the distances between the points within each cluster and their corresponding centroid, which typically converges to a local optimum. This characteristic enables K-means to efficiently delineate distinct and compact clusters, capturing the core essence of data distribution with respect to spatial proximity.
This insight prompts us to explore a conceptual analogy wherein the neural network that minimizes the loss most effectively operates analogously to a "centroid" for a cluster of trajectories within the same dynamic environment. We characterise the distance between a trajectory (data point) and the network (centroid) by the regression loss of the trajectory using the network. Initially, with a randomly initialized network-analogous to a randomly initialized centroid in K-means clustering-we assign each trajectory a label based on the minimal prediction loss calculated from all available networks. Subsequently, we refine this "centroid" by optimizing it to minimize the loss as specified in Equation (2). Through this iterative optimization process, we can achieve the objective stated in Equation (3).
More specifically, at round r, given the fixed network parameters from the previous iteration θ (r-1) , ϕ (r-1) , the environment assignment ê(r) i is updated through the following process,
ê(r) i = arg min e∈[M ] t∈I dx i t dt -h x i t ; θ (r-1) , ϕ (r-1) e 2 2 dt.(4)
If multiple solutions exist for Equation (4) and ê(r-1) i minimizes it, we retain this assignment for the next round, i.e., ê(r
) i = ê(r-1) i
. This approach ensures the validity of a constant loss reduction (to be stated in Proposition 3.1).
this section cite: ['b36', 'b1']

Section: Assignment-driven Optimization
After assigning trajectories to specific clusters, we proceed to update the conceptual centroid by optimizing network parameters. In the K-means algorithm, centroids are recalculated by averaging the positions of all points within each cluster. Similarly, our method updates network parameters by considering the mean estimation error over trajectories within a cluster, ensuring unbiased contributions from each trajectory. This approach not only improves the representational accuracy of each cluster but also enables the network to dynamically adapt to the underlying structure of the trajectories, thereby enhancing the efficacy and reliability of our learning process in unlabeled scenarios. Therefore, the parameters θ (r) and ϕ (r) are given by:
θ (r) , ϕ (r) = arg min θ,ϕ R ê(r) (θ, ϕ).(5)
this section cite: []

Section: Theoretical Property
We begin by demonstrating that Algorithm 1 effectively optimizes Equation (3). Proposition 3.1. For all rounds 1 ≤ r < T r , we must have
R ê(r+1) θ (r+1) , ϕ (r+1) ≤ R ê(r) θ (r) , ϕ (r) .
Furthermore, suppose the space of arg min θ,ϕ R ê(θ, ϕ) is finite for all ê ∈ [M ] N . Then there exists a constant C > 0 such that if r > 1 and R ê(r+1
) (θ (r+1) , ϕ (r+1) ) < R ê(r) (θ (r) , ϕ (r) ), we must have R ê(r+1) θ (r+1) , ϕ (r+1) ≤ R ê(r) θ (r) , ϕ (r) -C.
Remark 3.1. Given the assumptions made in prior works [43,17] that h(•; θ, ϕ e ) is linear with respect to θ and ϕ e , and that Ω(ϕ e ) is strictly convex with respect to ϕ e , it follows logically that the space of arg min θ,ϕ R ê(θ, ϕ) is finite for all ê ∈ [M ] N , as is evident from Equation ( 2). The proof is provided in Appendix A.
This proposition demonstrates that, as long as the loss in consecutive rounds of Algorithm 1 decreases, the loss must decrease by a constant C > 0.
this section cite: ['b42', 'b16']

Section: Experiments
Our experiments investigate three dynamical systems governed by specific differential equations: an ODE for biological modeling, PDEs for reaction-diffusion in chemistry, and the Navier-Stokes equations for incompressible fluid dynamics. These complex, nonlinear systems test our method's ability to classify spatio-temporal patterns and physical laws across diverse environments.
this section cite: []

Section: Environment Specification
We provide a basic introduction to the datasets here, with detailed descriptions in Appendix E. Lotka-Volterra (LV) [23] The system models the dynamics between a prey-predator pair in an ecosystem, captured by the following ODE:
dm/dt = αm -βmn, dn/dt = δmn -γn
where m and n represent the population densities of the prey and predator, respectively, and α, β, δ, and γ are the interaction parameters between the two species.
Gray-Scott (GS) [28] The model uses simple reaction-diffusion equations to effectively study complex pattern formation in chemical and biological systems, following underlying PDE dynamics:
∂m/∂t = D m ∆m -mn 2 + F (1 -m), ∂n/∂t = D n ∆n -mn 2 -(F + k)n.
where m and n represent the concentrations of two chemical components in the spatial domain S with periodic boundary conditions; D m and D n are their constant diffusion coefficients; and F and k are the reaction parameters that govern the spatio-temporal dynamic patterns.
Navier-Stokes (NS) [22] The Navier-Stokes PDE describes the motion of viscous fluid substances:
∂m/∂t = -n∇m + ν∆m + ξ, ∇v = 0 where n is the velocity field, m = ∇ × n is the vorticity, both n and m lie in a spatial domain S with periodic boundary conditions, ν is the viscosity (fixed at 1e -3 ), and ξ is the constant forcing term in the domain S.
this section cite: ['b22', 'b27', 'b21']

Section: Experimental Setting and Baselines
Settings. We evaluate DynaInfer in two distinct settings: in-domain generalization on E o and adaptation to new environments in E u , with E o and E u hosting disjoint environments. For in-domain experiments, both training and testing occur on E o . At test time, environment labels are also not provided and are instead inferred from the prediction bias over an initial segment of the trajectory (less than 2∆t for practical reasons). For adaptation experiments, we follow standard domain adaptation practice: initial training on the source domain E o is followed by fine-tuning and testing on the target domain E u , where environment labels are provided.
this section cite: []

Section: Dataset Preparation.
For in-domain experiments, we generate four LV trajectories in each of nine environments, ten GS trajectories in each of three environments, and eight NS trajectories in each of four environments. For adaptation experiments, we simulate the same number of trajectories per environment, conducting finetuning in two additional environments e ∈ E u . All dynamic environment parameters are detailed in Appendix E. For evaluation, we sample 32 trajectories per environment, initialized according to the underlying distribution p(x 0 ). The LV and GS data are generated using the DOPRI5 solver [8,12], while the NS data is simulated with the pseudo-spectral method as in [22].
Baselines. We explore three potential strategies for assigning environment labels in the absence of environmental information, compared to our method (DynaInfer): grouping all samples into a single environment (All in One), assigning a distinct environment label to each sample (One per Env), and random assignment (Random). Additionally, we consider an "Oracle" assignment method where labels are fully known during training, bringing the total to five labeling strategies. Furthermore, we consider three base models for dynamical system generalization: LEADS [43], CoDA-l 1 , and CoDA-l 2 [17]. We utilize the neural network architectures and parameter configurations as described in their papers for each type of dynamic system. By combining these assignment methods with base models, we generate fifteen distinct methods for evaluation. In adaptation experiments, during fine-tuning, LEADS and CoDA adhere to the protocol described in their papers, by fixing the shared components or parameters and rendering only the E u -specific components trainable. All neural network architectures, optimizers, and parameters for the base models are configured as described in their respective papers.
this section cite: ['b7', 'b11', 'b21', 'b42', 'b16']

Section: Metrics.
To rigorously evaluate predictive accuracy in dynamical system learning, we adopt two complementary metrics: Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE), averaged over 5 independent runs.
this section cite: []

Section: Experimental Results

this section cite: []

Section: In-domain Generalization Results
The in-domain generalization results detailed in Table 1 illustrate the performance implications of various assignment strategies. We observe that the "All in One" and "Random" assignment strategies consistently underperform across multiple datasets and baseline models. While the "One per Env" strategy yields only mediocre results, it provides a viable initial approach in scenarios where no labels are available. Across all datasets, DynaInfer significantly outperforms other assignment strategies. Furthermore, DynaInfer consistently shows effectiveness across all tested base models and datasets, underscoring its robustness against diverse methods and datasets. Notably, DynaInfer either matches or exceeds Oracle performance, particularly in complex PDE environments like GS and NS, suggesting that its bias-aware approach effectively compensates for not having access to the true labels available to Oracle.
In Figure 3, DynaInfer's predicted states qualitatively align closely with the ground truth and Oracle, occasionally outperforming Oracle (e.g., in GS dataset with LEADS base model, where Oracle shows some jitters). 11
this section cite: []

Section: Domain Adaptation Results
The results in Table 2 demonstrate the performance of different assignment strategies under the domain adaptation setting. The "One per Env" strategy consistently outperforms the "All in One" approach.
While the "Random" assignment benefits the base model LEADS, it slightly diminishes CoDA's performance across all datasets. DynaInfer shows strong adaptation capabilities across various datasets and base generalization methods, consistently outperforming other non-Oracle techniques. This indicates that DynaInfer effectively captures commonalities across environments, enabling smoother adaptation to new conditions. Furthermore, the performance gap between DynaInfer and the Oracle is significantly narrower in adaptation tasks compared to in-domain generalization.
Oracle DynaInfer Truth L E A D S C o D A L E A D S C o D A Oracle DynaInfer Truth L E A D S C o D A L E A D S C o D A LEADS CoDA
this section cite: []

Section: Assignments Convergence
We illustrate the probability of environment assignments with DynaInfer over training time in Figure 4. Initially, our model may default to random assignments due to unoptimized neural networks. However, the assignments quickly converge to the true labels. Notably, systems with simpler dynamics, like LV compared to NS, enable quicker learning of base generalization methods, resulting in faster convergence of environment assignments.
this section cite: []

Section: Performance across Varying Number of Assumed Environments
As the true number of environments (|E o |) might be unknown, we examine DynaInfer's performance with varying assumed environments M in Figure 5. Our findings show that prior knowledge of the true M is beneficial: performance peaks when the assumed M aligns with the true count. Additionally, our model demonstrates robustness to over-estimations of M . Due to its ability to account for bias, our model effectively identifies trajectories from the same environment and remains robust to an excess of inaccurately trained neural networks, even when the assumed M is too large. The above observations suggest that incrementally increasing the number of environments until peak performance is achieved can be a straightforward way to identify the true M . Lastly, DynaInfer consistently outperforms other non-oracle approaches when M is underestimated, except for the One-per-Env baseline (which requires M equal to the number of trajectories and is computationally infeasible). 1 5 9 1 3 1 7 2 1 2 5 2 9 3 3 Epoch 200 123456789 1 5 9 1 3 1 7 2 1 2 5 2 9 3 3 Epoch 400 123456789 1 5 9 1 3 1 7 2 1 2 5 2 9 3 3 Epoch 600 123456789 1 5 9 1 3 1 7 2 1 2 5 2 9 3 3 Epoch 800 123456789 1 5 9 1 3 1 7 2 1 2 5 2 9 3 3 Epoch 1000 123456789 1 5 9 1 3 1 7 2 1 2 5 2 9 3 3 Epoch 1200 123456789 1 5 9 1 3 1 7 2 1 2 5 2 9 3 3 Epoch 1400 123456789 1 5 9 1 3 1 7 2 1 2 5 2 9 3 3 Epoch 1600 123456789 1 5 9 1 3 1 7 2 1 2 5 2 9 3 3 Epoch 1800 123456789 1 5 9 1 3 1 7 2 1 2 5 2 9 3 3 Epoch 2000 0.0 0.2 0.4 0.6 0.8 1.0 1 2 3 4 1 5 9 1 3 1 7 2 1 2 5 2 9 Epoch 400 1 2 3 4 1 5 9 1 3 1 7 2 1 2 5 2 9 Epoch 800 1 2 3 4 1 5 9 1 3 1 7 2 1 2 5 2 9 Epoch 1200 1 2 3 4 1 5 9 1 3 1 7 2 1 2 5 2 9 Epoch 1600 1 2 3 4 1 5 9 1 3 1 7 2 1 2 5 2 9 Epoch 2000 1 2 3 4 1 5 9 1 3 1 7 2 1 2 5 2 9 Epoch 2400 1 2 3 4 1 5 9 1 3 1 7 2 1 2 5 2 9 Epoch 2800 1 2 3 4 1 5 9 1 3 1 7 2 1 2 5 2 9 Epoch 3200 1 2 3 4 1 5 9 1 3 1 7 2 1 2 5 2 9 Epoch 3600 1 2 3 4 1 5 9 1 3 1 7 2 1 2 5 2 9 Epoch 4000 0.0 0.2 0.4 0.6 0.8 1.0 Figure 4: Environment assignment probability over time, averaged over 5 runs, with LEADS as base model (on LV (top) and NS (bottom); see Appendix G for GS). The assignment converges to the true label faster than the designated training steps. A similar trend is observed with the CoDA model. En v= 3 En v= 4 En v= 5 En v= 6 En v= 7 En v= 8 En v= 9 En v= 10 En v= 11 LEADS 2.47E-2 1.04E-2 7.02E-3 4.24E-3 2.59E-3 1.21E-3 7.93E-5 2.41E-4 3.71E-4 LV En v= 3 En v= 4 En v= 5 En v= 6 En v= 7 En v= 8 En v= 9 En v= 10 En v= 11 CoDA-l1 En v= 3 En v= 4 En v= 5 1.87E-3 7.25E-5 1.20E-4 7.98E-5 En v= 2 En v= 3 En v= 4 En v= 5 En v= 6 4.11E-2 2.12E-2 7.05E-3 1.13E-2 1.18E-2 NS En v= 2 En v= 3 En v= 4 En v= 5 En v= 6 4.28E-2 2.17E-2 1.62E-2 1.76E-2 1.69E-2 En v= 2 En v= 3 En v= 4 En v= 5 En v= 6 4.04E-2 1.88E-2 1.19E-2 9.72E-3 1.23E-2 The peak performance aligns with the true number of environments (bold on x-axis) with high probability, and remains stable thereafter.
5 Related Work
this section cite: []

Section: Domain Generalization and Adaptation
Domain generalization (DG) seeks to train a model on one or multiple distinct but related source domains so that it generalizes effectively to any out-of-distribution (OOD) target domain. DG methods assume data heterogeneity and use additional environment labels to develop models that remain robust across unseen and shifted test data. Many DG strategies focus on domain alignment, aiming to minimize divergence among source domains to achieve domain-invariant representations [21,14,29,31,40]. Other approaches enhance the diversity of training data by augmenting source domains [5,33,39]. Additionally, some methods leverage meta-learning and invariant risk minimization for regularization, further enhancing generalization [20,1].
Domain adaptation (DA) methods enable model generalization to target domains with shifted data distributions and are primarily classified into three categories. Instance-based methods reweight or adjust training samples to reflect the test distribution [15,6]. Feature-based approaches align feature distributions across training and test domains [38,36]. Model-based strategies focus on developing models that are either robust to domain shifts or specifically tailored for the target domain [25,10].
this section cite: ['b20', 'b13', 'b28', 'b30', 'b39', 'b4', 'b32', 'b38', 'b19', 'b0', 'b14', 'b5', 'b37', 'b35', 'b24', 'b9']

Section: Generalization for Dynamical Systems
Generalization in dynamical systems remains underexplored in literature. Among the limited studies, LEADS emerges as a novel multi-task learning framework that effectively generalizes across the functional space of dynamical systems [43]. Alternatively, CoDA optimizes within the parameter space, enhancing model adaptability and efficiency while accommodating increased environmental variability without requiring multiple distinct network trainings for each setting [17]. In contrast, DyAd is a context-aware meta-learning approach that adjusts the dynamics model by decoding a timeinvariant context from observed states [41]. Despite its novelty, DyAd relies on potentially impractical weak supervision based on physics-derived quantities and uses Adaptive Instance Normalization, which may degrade performance.
Currently, three notable weaknesses prevail in generalization works for dynamical systems. First, there is an assumption that prior knowledge about the target domain exists, and without it, most generalization methods would fail [11]. Second, the predominant use of the mean squared error as a loss function is inadequate for evaluating the reconstruction accuracy of chaotic systems. Lastly, the influence of unlabelled trajectory data on the process of learning generalizable dynamical systems remains both unexplored and unresolved -a gap this paper examines for the first time. While switched systems learning methods [19] infer modes by classifying individual data points (analogous to environment inference), our work operates on trajectories governed by ODEs or PDEs.
this section cite: ['b42', 'b16', 'b40', 'b10', 'b18']

Section: Conclusion
We propose an environment inference method that improves the understanding and generalization of complex dynamical systems across various environments without using manually labeled data. DynaInfer infers environment labels directly from training data, overcoming the challenges associated with explicit annotation. Theoretical analysis ensures convergence of DynaInfer, and experiments show DynaInfer often surpasses non-oracle methods and matches or exceeds oracle performance.
Future research could unfold along the following promising directions: First, to improve generalization in chaotic systems, MSE-based methods could be replaced with more suitable metrics such as the sliced Wasserstein-1 distance, which would require developing a tailored inference model. Similarly, effectively inferring environments for dynamics with complex boundaries remains a significant open problem, as current learning methods often oversimplify boundary conditions. Furthermore, for systems requiring interpretable parameters, our method could be extended to jointly optimize environment labels and physical coefficients. To improve convergence, coordinate optimization techniques may help escape local optima for objectives that are convex in individual variable blocks [13,30].
Finally, techniques such as adaptive early stopping, dynamic batching, and membership functions [3] could further enhance training efficiency.
this section cite: ['b12', 'b29', 'b2']

Section: References
Ref_id:b0 Title:  Year: (2019)
Ref_id:b1 Title: Robust optimization Year: (2009)
Ref_id:b2 Title: A constrained clustering approach to bounded-error identification of switched and piecewise affine systems Year: (2022)
Ref_id:b3 Title: Discovering governing equations from data by sparse identification of nonlinear dynamical systems Year: (2016)
Ref_id:b4 Title: Domain generalization by solving jigsaw puzzles Year: (2019)
Ref_id:b5 Title: Boosting for transfer learning Year: (2007)
Ref_id:b6 Title: Deep learning for physical processes: Incorporating prior scientific knowledge Year: (2019)
Ref_id:b7 Title: A family of embedded runge-kutta formulae Year: (1980)
Ref_id:b8 Title: Turbulence modeling in the age of data Year: (2019)
Ref_id:b9 Title: Domain-adversarial training of neural networks Year: (2016)
Ref_id:b10 Title: Outof-domain generalization in dynamical systems reconstruction Year: (2024)
Ref_id:b11 Title: Array programming with numpy Year: (2020)
Ref_id:b12 Title: Coordinate optimization for bi-convex matrix inequalities Year: (1997)
Ref_id:b13 Title: Domain generalization via multidomain discriminant analysis Year: (2020)
Ref_id:b14 Title: Instance weighting for domain adaptation in NLP Year: (2007)
Ref_id:b15 Title: Learning stable nonlinear dynamical systems with gaussian mixture models Year: (2011)
Ref_id:b16 Title: Generalizing to new physical systems via context-informed dynamics model Year: (2022)
Ref_id:b17 Title: Fairness without demographics through adversarially reweighted learning Year: (2020)
Ref_id:b18 Title: Estimating the probability of success of a simple algorithm for switched linear regression Year: (2013)
Ref_id:b19 Title: Episodic training for domain generalization Year: (2019)
Ref_id:b20 Title: Domain generalization with adversarial feature learning Year: (2018)
Ref_id:b21 Title: Fourier neural operator for parametric partial differential equations Year: ()
Ref_id:b22 Title: Elements of physical biology Year: (1925)
Ref_id:b23 Title:  Year: (2017)
Ref_id:b24 Title: Unified deep supervised domain adaptation and generalization Year: (2017)
Ref_id:b25 Title: Efficient computation of electrograms and ecgs in human whole heart simulations using a reaction-eikonal model Year: (2017)
Ref_id:b26 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b27 Title: Complex patterns in a simple system Year: (1993)
Ref_id:b28 Title: Efficient domain generalization via common-specific low-rank decomposition Year: (2020)
Ref_id:b29 Title: Domain-robust mixture-of-experts for MILP solution prediction across domains Year: (2025)
Ref_id:b30 Title: Learning to optimize domain specific normalization for domain generalization Year: (2020)
Ref_id:b31 Title: Data-driven approaches for predicting spread of infectious diseases through dinns: Disease informed neural networks Year: (2022)
Ref_id:b32 Title: Generalizing across domains via cross-gradient training Year: (2018)
Ref_id:b33 Title: Dgm: A deep learning algorithm for solving partial differential equations Year: (2018)
Ref_id:b34 Title: Robustness to spurious correlations via human annotations Year: (2020)
Ref_id:b35 Title: Deep CORAL: correlation alignment for deep domain adaptation Year: (2016)
Ref_id:b36 Title: Regularized k-means clustering of high-dimensional data and its asymptotic consistency Year: (2012)
Ref_id:b37 Title: Deep domain confusion: Maximizing for domain invariance Year: (2014)
Ref_id:b38 Title: Generalizing to unseen domains via adversarial data augmentation Year: (2018)
Ref_id:b39 Title: Out-of-distribution generalization with causal feature separation Year: (2023)
Ref_id:b40 Title: Meta-learning dynamics forecasting using task inference Year: (2022)
Ref_id:b41 Title: Integrating physics-based modeling with machine learning: A survey Year: (2020)
Ref_id:b42 Title: Leads: Learning dynamical systems that generalize across environments Year: (2021)
Ref_id:b43 Title: Learning efficient and robust ordinary differential equations via invertible neural networks Year: (2022)
