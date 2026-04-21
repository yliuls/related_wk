Title: Differentiable Cyclic Causal Discovery Under Unmeasured Confounders
Abstract: Understanding causal relationships between variables is fundamental across scientific disciplines. Most causal discovery algorithms rely on two key assumptions: (i) all variables are observed, and (ii) the underlying causal graph is acyclic. While these assumptions simplify theoretical analysis, they are often violated in real-world systems, such as biological networks. Existing methods that account for confounders either assume linearity or struggle with scalability. To address these limitations, we propose DCCD-CONF, a novel framework for differentiable learning of nonlinear cyclic causal graphs in the presence of unmeasured confounders using interventional data. Our approach alternates between optimizing the graph structure and estimating the confounder distribution by maximizing the loglikelihood of the data. Through experiments on synthetic data and real-world gene perturbation datasets, we show that DCCD-CONF outperforms state-of-the-art methods in both causal graph recovery and confounder identification. Additionally, we provide consistency guarantees for our framework, reinforcing its theoretical soundness.Contributions. In this work, we tackle three key challenges in causal discovery: directed cycles, nonlinearity, and unmeasured confounders. Our main contributions are:• We introduce DCCD-CONF, a novel differentiable causal discovery framework for learning nonlinear cyclic relationships under Gaussian exogenous noise, with confounders modeled as correlations in the noise term.• We show that exact maximization of the proposed score function results in identification of the interventional equivalence class of the ground truth graph.• We conduct extensive evaluations, comparing DCCD-CONF with state-of-the-art causal discovery methods on both synthetic and real-world datasets.

Section: Introduction
Modeling cause-effect relationships between variables is a fundamental problem in science [1,2,3], as it enables the prediction of a system's behavior under previously unseen perturbations. These relationships are typically represented using directed graphs (DGs), where nodes correspond to variables, and directed edges capture causal dependencies. Consequently, causal discovery reduces to learning the structure of these graphs.
Existing causal discovery algorithms can be broadly classified into three categories: (i) constraintbased methods, (ii) score-based methods, and (iii) hybrid methods. Constraint-based methods, such as the PC algorithm [4,5,6], search for causal graphs that best satisfy the independence constraints observed in the data. However, since the number of conditional independence tests grows exponentially with the number of nodes, these methods often struggle with scalability. Score-based methods, such as the GES algorithm [7,8], learn graph structures by maximizing a penalized score function, such as the Bayesian Information Criterion (BIC), over the space of graphs. Given the vast search space, these methods often employ greedy strategies to reduce computational complexity. A significant breakthrough came with Zheng et al. [9], who introduced a continuous constraint formulation to restrict the search space to acyclic graphs, inspiring several extensions [10,11,12,13,14,15] that frame causal discovery as a continuous optimization problem under various model assumptions. Hybrid methods [16,17,18] integrate aspects of both constraint-based and score-based approaches, leveraging independence constraints while optimizing a score function.
Most causal structure learning methods assume (i) a directed acyclic graph (DAG) with no directed cycles and (ii) complete observability, meaning no unmeasured confounders.
39th Conference on Neural Information Processing Systems (NeurIPS 2025). While these assumptions simplify the search space, they are often unrealistic, as real-world systems-especially in biology-frequently exhibit feedback loops and hidden confounders [19]. Enforcing these constraints can also increase computational complexity, particularly in ensuring acyclicity, which often requires solving challenging combinatorial or constrained optimization problems. These limitations hinder the practical applicability of existing methods in settings where such violations are unavoidable.
Several approaches have been developed to address the challenge of feedback loops within causal graphs. Early work by Richardson [20] extended constraint-based approaches for Directed Acyclic Graphs (DAGs) to accommodate directed cycles. Another key contribution came from Lacerda et al. [21], who generalized Independent Component Analysis (ICA)-based causal discovery to handle linear non-Gaussian cyclic graphs. More recently, a growing body of research has focused on score-based methods for learning cyclic causal graphs [22,23,24,25]. Additionally, some approaches leverage interventional data to improve structure recovery in cyclic systems. For instance, Hyttinen et al. [26] and Huetter and Rigollet [22] introduced frameworks that explicitly incorporate interventions to refine cyclic graph estimation. Sethuraman et al. [27] further advanced this line of research by introducing a differentiable framework for learning nonlinear cyclic graphs. Unlike differentiable DAG learners that enforce acyclicity through augmented Lagrangian-based solvers, their approach sidesteps these constraints by directly modeling the data likelihood, enabling more efficient and flexible learning of cyclic causal structures. However, their method assumes the absence of unmeasured confounders, which limits its applicability in real-world settings where hidden confounders are often present.
Causal discovery in the presence of latent confounders has seen limited development, with most existing approaches grounded in constraint-based methodologies. Extensions of the PC algorithm, such as the Fast Causal Inference (FCI) algorithm [28], construct a Partial Ancestral Graph (PAG) to represent the equivalence class of DAGs in the presence of unmeasured confounders, and can accommodate nonlinear dependencies depending on the chosen conditional independence tests. However, standard FCI does not incorporate interventional data, prompting extensions such as JCI-FCI [29] and related approaches [30] that combine observational and interventional settings. Jaber et al. [31] further advanced this line of work by allowing for unknown interventional targets. Additionally, Forré and Mooij [32] introduced σ-separation, a generalization of d-separation, enabling constraint-based causal discovery in the presence of both cycles and latent confounders. Suzuki and Yang [33] introduce LiNGAM-MMI, a generalization of the ICA-based LiNGAM [34] that quantifies and mitigates confounding via a KL divergence minimization. Integer-programming-based formulations have also been proposed for causal discovery under latent confounding, such as [35,36]. A few recent approaches, such as Bhattacharya et al. [37], have explored continuous optimization frameworks using differentiable constraints, though these methods are currently limited to linear settings. In parallel, several works exist on causal inference under latent confounding-most notably Abadie et al. [38], Chernozhukov et al. [39]-propose doubly robust estimators that integrate outcome modeling, weighting, and cross-fitting for reliable effect estimation. Overall, a unified framework capable of handling nonlinearity, cycles, latent confounders, and interventions remains largely absent.
Organization. The paper is structured as follows: Section 2 introduces the problem setup. In Section 3, we present DCCD-CONF, our differentiable framework for nonlinear cyclic causal discovery with unmeasured confounders. We then evaluate its effectiveness on synthetic and realworld datasets in Section 4. Finally, Section 5 concludes the paper.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b21', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38']

Section: Problem Setup

this section cite: []

Section: Structural Equations for Cyclic Causal Graphs
Let G = (V, E, B) represent a possibly cyclic directed mixed graph (DMG) that encodes the causal dependencies between the variables in the vertex set V = [d], where [d] = {1, . . . , d}. E denotes the set of directional edges of the form i → j in G, and B denotes the set of bidirectional edges of the form i ↔ j in G. Each node i is associated with a random variable X i with the directed edge i → j ∈ E representing a causal relation between X i and X j , and the bidirectional edge i ↔ j ∈ B indicates the presence of a hidden confounder between X i and X j . Following the framework proposed by Bollen [40] and Pearl [41], we use structural equations model (SEM) to algebraically describe the system:
X i = F i (X pa G (i) , Z i ), i = 1, . . . , d,(1)
where pa G (i) := {j ∈ [d] : j → i ∈ E} represents the parent set of X i in G, and X pa G (i) denotes the components of X = (X 1 , . . . , X d ) indexed by the parent set pa G (i). We exclude self-loops (edges of the form X i → X i ) from G, as their presence can lead to identifiability challenges [42]. The function F i , referred to as causal mechanism, encodes the functional relationship between X i and its parents X pa G (i) , and the exogenous noise variable Z i .
The collection of exogenous noise variables Z = (Z 1 , . . . , Z d ) account for the stochastic nature as well as the confounding observed in the system. We make the assumption that the exogenous noise vector follows a Gaussian distributions: Z ∼ N (0, Σ Z ). Notably, if (Σ Z ) ij ̸ = 0, then variables X i and X j are confounded, i.e., i ↔ j ∈ B. In other words, confounding is modeled through correlations in the exogenous noise variables. Intuitively, if X i and X j share a hidden cause, their unexplained variation (the part not accounted for by their observed parents) will tend to move together. By allowing the noise terms Z i and Z j to be correlated, this shared influence can be effectively captured. This formulation generalizes prior work by allowing cycles, extending both nonlinear cyclic models that assume independent noise terms [27], and acyclic models without confounders [43].
By collecting all the causal mechanisms into the joint function F = (F 1 , . . . , F d ), we can then combine (1) over i = 1, . . . , d to obtain the equation
X = F(X, Z).(2)
We will use (2) to represent the causal system due to its simplicity for subsequent discussion. The observed data represents a snapshot of a dynamical process where the recursive equations in (2) define the system's state at its equilibrium. Thus, in our experiments we assume that the system has reached the equilibrium state. For a given random draw of Z, the value of X is defined as the solution to ( 2). To that end, we assume that (2) admits a unique fixed point for any given Z. We refer to the map f x : X → Z as the forward map, and f z : Z → X as the reverse map. In Section 3.1, we show that the chosen parametric family of functions indeed guarantees the existence of a unique fixed point. Under these restrictions, the probability density of X is well defined and is given by
p G (X) = p Z f x (X) det J fx (X) ,(3)
where J fx (X) denotes the Jacobian matrix of the function f x at X.
this section cite: ['b39', 'b40', 'b41', 'b26', 'b42']

Section: Interventions
In our work, we consider surgical interventions [41], also known as hard interventions, where all the incoming edges to the intervened nodes are removed from G. Given a set of intervened upon nodes (also known as interventional targets), denoted as I ⊆ V, the structural equations in (1) are modified as follows
X i = C i , if X i ∈ I, F i (X pa G (i) , Z i ), if X i / ∈ I,(4)
where C i is a random variable sampled from a known distribution, i.e., C i ∼ p I (C i ). We denote do(I)(G) to be the mutilated graph under the intervention I (see Figure 1). Note that X i is no longer confounded if it is intervened on.
We consider a family of K interventional experiments I = {I k } k∈[K] , where I k represents the interventional targets for the k-th experiment. Let U k ∈ {0, 1} d×d denote a diagonal matrix with (U k ) ii = 1 if i / ∈ I k , and (U k ) ii = 0 if i ∈ I k . Similar to the observational setting, (4) can be vectorized to obtain the following form
X = U k F(X, Z) + C,(5)
where C = (C 1 , . . . , C d ) is a vector with C i ∼ p I (C i ) if i ∈ I k , and C i = 0 otherwise. For the interventional targets I k ∈ I, let f
(I k ) x
denote the forward map. Similar to the observational setting, we make the following assumption on the set of interventions. Assumption 1 (Interventional stability). Let I = {I k } k∈[K] be a family of interventional targets. For each I k ∈ I, the structural equations in (5) admits a unique fixed point given the exogenous noise vector Z.
Thus, the probability distribution of X for the interventional targets I k is given by
p do(I k )(G) (X) = p I (C)p Z f (I k ) x (X) U k det J f (I k ) x (X) ,(6)
where U k = {i : i ∈ V \ I} denotes the index of purely observed nodes, and p Z f
(I k ) x
(X) U k is the marginal distribution of the combined vector Z, restricted to the components indexed by U k .
Given a family of interventions I, our goal is to learn the structure of the DMG by maximizing the log-likelihood of the data, in addition to identifying the variables that are being confounded by the unmeasured confounders Z. The next section presents our approach to addressing this problem.
this section cite: ['b40']

Section: DCCD-CONF: Differentiable Cyclic Causal Discovery with Confounders
In this section, we present our framework for differentiable learning of cyclic causal structures in the presence of unmeasured confounders. We start by modeling the causal mechanisms, then define the score function used for learning, followed by a theorem that validates its correctness. Finally, we outline the algorithm for estimating the model parameters.
this section cite: []

Section: Modeling Causal Mechanism
We model the structural equations in (2) using implicit flows [44], which define an invertible mapping between x and z by solving the root of a function G(x, z) = 0, where G : R 2d → R d . Specifically, we take G(x, z) = x -F(x, z). General implicit mappings, however, do not guarantee invertibility or permit efficient computation of the log-determinant required for evaluating (6). To balance expressiveness with tractability, we adopt the structured form proposed by Lu et al. [44] for the causal mechanism:
F(x, z) = -g x (x) + g z (z) + z,(7)
where g x and g z are restricted to be contractive functions. A function g : R d → R d is contractive if there exists a constant L < 1 such that ∥g(x) -g(y)∥ ≤ L∥x -y∥ for all x, y ∈ R d . This contractiveness ensures that the associated implicit map is uniquely solvable and invertible (see Theorem 1 in [44]). In other words, contractivity ensures that the process defined by the SEM converges to an equilibrium state.
Under this formulation, the forward map takes the form f x (x) = (id + g z ) -1 • (id + g x )(x), where id denotes the identity map. Given x (or z), the corresponding value of z (or x) can be computed via a root-finding procedure, i.e., z = RootFind(x -F(x, •)), specifically, we employ a quasi-Newton method (i.e., Broyden's method [45]) to find the root. To capture more complex nonlinear interactions between the observed variables X and latent confounders Z, multiple such implicit blocks can be stacked. This is true since f x is highly nonlinear and by suitably parameterizing g x and g z any nonlinear interaction between x and z can be modeled. For simplicity, we focus on a single implicit flow block for subsequent discussion.
We parameterize the functions g x and g z using neural networks. The adjacency matrix of the causal graph G is encoded as a binary matrix M G ∈ {0, 1} d×d , representing the presence of directed edges and serving as a mask on the inputs to g x . The diagonal entries of M G are explicitly enforced to be zero to prevent self-loops. Similarly, the identity matrix is used to mask the inputs to g z . Consequently, the causal mechanism is defined as:
[F θ (X, Z)] i = -NN(M G * ,i ⊙ X | θ x ) + NN(Z i | θ z ) + Z i i ,(8)
where NN(• | θ) denotes a fully connected neural network parameterized by θ, ⊙ denotes the Hadamard product, and M G * ,i is the i-th column of M G . The contractivity of g x and g z can be enforced by rescaling their weights using spectral normalization [46]. Moreover, the contractive nature of the causal mechanism facilitates efficient computation of the score function used for learning causal graphs, as discussed in Section 3.2.
While the contractivity assumption may seem restrictive, it ensures stability and well-posedness in the presence of directed cycles. If the causal graph is known to be acyclic, this assumption can be relaxed (see Appendix C.1).
this section cite: ['b43', 'b5', 'b43', 'b43', 'b44', 'b45']

Section: Score function
Given a family of interventions I = {I k } k∈[K] , we would like to learn the parameters of the structural equation model, i.e., causal graph structure, causal mechanism, and confounder distribution. To that end, similar to prior work [27,47,15] in this domain we employ regularized log-likelihood of the observed nodes as the score function to be maximized. That is,
S I (G) := sup θ,Σ Z K k=1 E X∼p (k) log p do(I k )(G) (X) -λ|G|(9)
where p (k) is the data generating distribution for the k-th interventional experiment I k , Σ Z is the parameter (covariance matrix) governing the confounder distribution p Z , θ = (θ x , θ z ) is the combined causal mechanism parameters, and |G| denotes a sparsity enforcing regularizer on the edges of G, and p do(I k )(G) (X) is given by (6).
We now present the main theoretical result of this paper. The following theorem establishes that, under appropriate assumptions, the graph Ĝ estimated by maximizing (9) belongs to the same general directed Markov equivalence class (introduced by [42]) as the ground truth graph G * for each interventional setting I k ∈ I, denoted as Ĝ ≡ I G * , see Appendix A.1. Due to space constraints we provide the proof sketch below, see Appendix A.3 for complete proof of Theorem 2. Theorem 2. Let I = {I k } K k=1 be a family of interventional targets, let G * denote the ground truth directed mixed graph, let p (k) denote the data generating distribution for I k , and Ĝ := arg max G S(G). Then, under the Assumptions 1, A.13, A.14, and A.15, and for a suitably chosen λ > 0, we have that
Ĝ ≡ I G * . That is, Ĝ is I-Markov equivalent to G * .
Theorem 2 rests on three key assumptions. Assumption A.13 ensures that the data-generating distribution lies within the model class, while Assumption A.14 guarantees that every statistical independence in the data corresponds to a σ-separation in the ground-truth graph. Finally, Assumption 1 prevents the score function from diverging to infinity.
this section cite: ['b26', 'b46', 'b14', 'b5', 'b41']

Section: Proof (Sketch).
Building on the characterization of general directed Markov equivalence class by Bongers et al. [42], extended to the interventional setting, we show that any graph outside this equivalence class has a strictly lower score than the ground truth graph G * . This follows from the fact that certain independencies present in the data are not captured by graphs outside the equivalence class. Combined with the expressiveness of the model class, this prevents such graphs from fitting the data properly.
If the intervention set consists of all single-node interventions, I = {I k } d k=1 with I k = {k}, Hyttinen et al. [26] showed that the ground truth DMG can be uniquely recovered in the linear setting. Moreover, in the absence of cycles and confounders, this result extends to the nonlinear case, as demonstrated by Brouillard et al. [15]. However, determining the necessary conditions on interventional targets for perfect recovery in general DMGs with cycles and confounders remains an open problem. Nonetheless, in practice, we find that observational distribution in combination with single-node interventions across all nodes lead to perfect recovery of the ground truth, even in the nonlinear case, as shown in Section 4.
this section cite: ['b41', 'b25', 'b14']

Section: Updating model parameters
In practice, we use gradient based stochastic optimization to maximize (9). For this purpose, following Sethuraman et al. [27] and Brouillard et al. [15], the entries of adjacency matrix M ij are modeled as Bernoulli random variable with parameters b ij , grouped into the matrix σ(B). We denote M ∼ σ(B) to indicate that M ij ∼ Bern(b ij ) for all i, j ∈ [d]. In this formulation, the sparsity regularizer is ∥M∥ 0 , which is computationally intractable and thus we use the ℓ 1 -norm, ∥M∥ 1 as a proxy. Consequently, the score function in (9) is replaced by the following relaxation:
ŜI (B) := sup θ,Σ Z E M∼σ(B) K k=1 N k i=1 log p do(I k )(G) (x (i,k) ) -λ∥M∥ 1 ,(10)
where we replace the expectation with respect to data distribution in (9) with sum over the finite samples, x (i,k) represents the i-th data sample in the k-the interventional setting. We note that, since p Z = N (0, Σ Z ), the covariance of the exogenous confounder vector, Σ Z , is implicitly embedded within p do(I k )(G) (x (i,k) ) in the score function.
The optimization of the score function is carried in two steps. First, we optimize Ŝ(B) with respect to the neural network parameters θ and the graph structure parameters B. Next, we optimize Ŝ(B) with respect to the parameters of the exogenous noise distribution, Σ Z . However, maximizing ŜI (B) presents two main challenges: (i) computing log p X (X) is computationally expensive due to the presence of | det(J f (I k )
x (X))|, which requires O(d 2 ) gradient calls, and (ii) updating Σ Z via stochastic gradients could lead to stability issues as Σ Z may loose its positive definiteness.
We now describe how these challenges are addressed, along with the specific procedures for updating the individual model parameters.
this section cite: ['b8', 'b26', 'b14', 'b8']

Section: Computing log determinant of the Jacobian
As discussed earlier, computing log |J f (I k ) x (X)| is a significant challenge in maximizing the score function Ŝ(B). To address this, we utilize the unbiased estimator of the log-determinant of the Jacobian introduced by Behrmann et al. [46], which is based on the power series expansion of log(1 + x). Since f
(I k ) x (x) = (id + U k g z ) -1 • (id + U k g x )(x) log det J f (I k ) x (X) = log det I + J U k gx (X) -log det I + J U k gz (Z) = ∞ m=1 (-1) m+1 m Tr J m U k gx (X) -Tr J m U k gz (Z) ,(11)
where I ∈ R d×d denotes the identity matrix, J m U k gx represents the Jacobian matrix raised to the m-th power, and Tr denotes the trace of matrix. The series in (11) is guaranteed to converge if the causal functions g x and g z are contractive [48].
In practice, the power series is truncated to a finite number of terms, which may introduce bias into the estimator. To mitigate this issue, we follow the stochastic approach of Chen et al. [49]. Specifically, we sample a random cut-off point n ∼ p N (n) for truncating the power series and weight the i-term in the finite series by the inverse probability of the series not ending at i. This yields the following unbiased estimator
log det J f (I k ) x (X) = E n∼p N (N ) n m=1 (-1) m+1 m • Tr J m U k gx (X) -Tr J m U k gz (Z) p N (ℓ ≥ m) . (12
)
The gradient calls can be reduced even further using the Hutchinson trace estimator [50], see Appendix B for more details.
this section cite: ['b45', 'b47', 'b48', 'b49']

Section: Updating neural network and graph parameters.
In the first step of the parameter update, keeping Σ Z fixed, the parameters of the neural network θ and the graph structure B are updated using the backpropagation algorithm with stochastic gradients.
The gradient of the score function ŜI (B) with respect to B is computed using the Straight-Through Gumbel estimator. This involves using Bernoulli samples in the forward pass while computing score, and using samples from Gumbel-Softmax distribution in the backward pass to compute the gradient, which can be differentiated using the reparameterization trick [51].
this section cite: ['b50']

Section: Updating the confounder-noise distribution parameters
In second parameter update step, we fix the value of θ and B and focus on the confounder-noise distribution parameter Σ Z . First, consider the case where no interventions are applied, i.e, I k = ∅. Note that the dependence of Ŝ(B) on Σ Z arises solely from p Z , which is embedded within p do(I k )(G) (X). Therefore, we can thus ignore the remaining terms in Ŝ(B) and focus exclusively on p Z . Let {x (i) } N i=1 denote the observational data. From the forward map, we have z (i) = f x (x (i) ). Given that p Z = N (0, Σ Z ), the relevant parts of Ŝ(B) with respect to Σ Z , denoted as L(I k ), are expressed as:
L(I k ) = sup Σ Z N i=1 - 1 2 log |Σ Z | - (z (i) ) ⊤ Σ -1 Z z (i) 2 .(13)
Simplifying (13) yields a more convenient form:
L(I k ) = sup Σ Z -Tr(SΣ -1 Z ) -log |Σ Z |,(14)
where
S = 1 n N i=1 z (i) (z (i) ) ⊤
is the sample covariance of Z. Maximizing (14) directly using backpropagation and stochastic gradients results in stability issues as Σ Z may lose its positive definiteness. However, Friedman et al. [52] demonstrated that the sparsity-regularized version of ( 14) is a concave optimization problem in Σ -1 Z that can be efficiently solved by optimizing the columns of Σ Z individually. This is achieved by formulating the column recovery as a lasso regression problem. We adopt this strategy while updating the Σ Z during the maximization of Ŝ(B).
Let W = Σ Z be the estimate of the covariance matrix. We reorder W such that the column and row being updated can be placed at the end, resulting in the following partition
W = W 11 w 12 w ⊤ 12 w 22 , S = S 11 s 12 s ⊤ 12 s 22 .(15)
Then, as shown by Friedman et al. [52], w 12 = W 11 β, where β is the solution to the following lasso regression problem, denoted as lasso(W 11 , s 12 , ρ):
min β 1 2 ∥W 1/2 11 β -y∥ 2 + ρ∥β∥ 1 ,(16)
where y = W -1/2 11 s 12 , and ρ is the regularization constant that promotes sparsity in Σ -1 Z . In an interventional setting I k , the dependence of Ŝ(B) on Σ Z arrises from the marginal distribution of Z restricted to components indexed by U k , i.e., purely observed nodes. Since Z follows a Gaussian distribution, Z U k also follows a Gaussian distribution with Z U k ∼ N (0, ΣI k ). From the properties of Gaussian distribution [53], we have ΣI k = (Σ Z ) U k ,U k . Consequently, for the interventional setting I k , (14) becomes
L(I k ) = sup Σ Z -Tr S I k (Σ Z ) -1 U k ,U k -log (Σ Z ) U k ,U k ,(17)
where
S I k = 1 n N i=1 z (i) U k (z (i) U k ) ⊤
is the sample covariance of Z corresponding to the purely observed nodes. In this case, we set W = (Σ Z ) U k ,U k and the rest of the update procedure remains the same. The overall parameter update procedure is summarized in Algorithm 1 in Appendix B.
this section cite: ['b51', 'b51', 'b52']

Section: Experiments
The code for DCCD-CONF is available at the repository: https://github.com/muralikgs/  dccd_conf.
We evaluated DCCD-CONF on both synthetic and real-world datasets, comparing its performance against several state-of-the-art baselines: NODAGS-Flow [27], LLC [26], DAGMA [54], and the linear ADMG recovery method proposed by Bhattacharya et al. [37] (which we refer to as ADMG). NODAGS-Flow learns nonlinear cyclic causal graphs but does not model unmeasured confounders. LLC accounts for confounders but is limited to linear cyclic SEMs. DAGMA handles nonlinearity under causal sufficiency while being limited to acyclic graphs. ADMG handles confounding but is limited to acyclic graphs and linear SEMs. Note that both DAGMA and ADMG do not natively support interventional data and hence we use these models in combination with the Joint Causal Inference (JCI) framework [29] and treat interventions as multiple contexts. We also include a comparison between DCCD-CONF and two constraint based models LiNGAM-MMI [33] and JCI-FCI [29] in the Appendix (see Appendix C).
this section cite: ['b26', 'b25', 'b53', 'b36', 'b28', 'b32', 'b28']

Section: Synthetic data
In all synthetic experiments, the cyclic graphs were generated using Erdős-Rényi (ER) random graph model with the outgoing edge density set to 2. We evaluated DCCD-CONF and the baselines on both linear as well as nonlinear SEMs described in Section 2. Our training data set consists of observational data and single-node interventional over all the nodes in the graph, i.e, I = ∅, {1}, . . . , {d} (unless stated otherwise), with N k = 500 samples per intervention. Furthermore, in all the experiments presented here, the SEM was constrained to be contractive. However, we also compare the performance of DCCD-CONF to the baselines on non-contractive SEMs in the appendix. For causal graph recovery (directed edges), we use the normalized structural Hamming distance (SHD) as the error metric. SHD counts the number of operations (addition, deletion, and reversal) needed to match the estimated causal graph to the ground truth, and normalization is done with respect to the number of nodes in the graph (lower the better). For confounder identification (bidirectional edges), we compare the non-diagonal entries of the estimated confounder-noise covariance matrix to those of the ground truth. We use F1 score as the error metric (higher the better). More details regarding the experimental setup is provided in Appendix B.
Impact of confounder count. We evaluate the performance of DCCD-CONF and the baselines using the previously defined error metrics, varying the confounder ratio (number of confounders divided by the number of nodes) from 0.2 to 0.8. In this case, the number of nodes in the graph is set to d = 10. The results, summarized in Figure 2a, show that DCCD-CONF consistently achieves lower SHD across all confounder ratios in both linear and nonlinear SEMs. Notably, in nonlinear SEMs, DCCD-CONF outperforms all baselines in causal graph recovery. Additionally, it demonstrates competitive results in confounder identification, highlighting its robustness in both tasks.
Impact of number of cycles. With d = 10 nodes and a confounder ratio of 0.3, we vary the number of cycles in the graph from 0 to 8. Figure 2b compares the performance of DCCD-CONF with the baselines under this setting. As shown, increasing the number of cycles does not lead to any noticeable degradation in performance for either directed or bidirected edge recovery.
this section cite: []

Section: Impact of degree of nonlinearity.
In this experiment, we vary the degree of nonlinearity in the SEM by adjusting β between 0 and 1, where
x = (1 -β)(W ⊤ x + z) + β tanh(W ⊤ x + z).
The SEM is fully linear when β = 0 and fully nonlinear when β = 1. Figure 2c summarizes the results. As shown, DCCD-CONF attains the highest performance as β approaches one, for both directed and bidirected edge recovery. When β is small (i.e., the system is more linear), LLC slightly outperforms DCCD-CONF, with both models performing comparably around β = 0.25.
this section cite: []

Section: Impact of number of interventions.
In this section, we evaluate graph recovery performance as the number of training interventions K varies from 0 to d, with d = 10 fixed. The case K = 0 corresponds to the observational dataset. Results for the nonlinear SEM setting are presented in Figure 2d. As illustrated, with fewer interventions all DCCD-CONF and the baselines tend to exhibit similar performance (less then 3 interventions). As the numbre of interventions increase, the performance gap widens with DCCD-CONF dominating all of the baselines. It is also worth noting that LLC cannot operate in the purely observational setting (K = 0).
this section cite: []

Section: Scaling with nodes.
We compare the performance of DCCD-CONF and the baselines as the number of nodes (d) varies from 10 to 80, with results summarized in Figure 3. The number of confounders is set to 0.3d. As the number of nodes increases, SHD rises across all methods, reflecting the increased difficulty of causal graph recovery in larger graphs. However, DCCD-CONF consistently outperforms the baselines in many cases, achieving lower SHD and higher F1 score, suggesting superior scalability with increasing graph size.  [55], which contains gene expression data from 218,331 melanoma cells across three conditions: (i) control, (ii) co-culture, and (iii) IFN-γ. Due to computational constraints, we restrict our analysis to a subset of 61 genes from the 20,000 genes in the genome, following the experimental setup of Sethuraman et al. [27] (see Appendix B for details). Each cell condition is treated as a separate dataset consisting of single-node interventions on the selected 61 genes.
this section cite: ['b54', 'b26']

Section: Real World data
Since the dataset does not provide a ground truth causal graph, SHD cannot be used for direct performance comparison. Instead, we assess DCCD-CONF and the baselines based on predictive performance over unseen interventions. To evaluate performance, we split each dataset 90-10, using the smaller portion as the test set, and measure performance using negative log-likelihood (NLL) on the test data after model training (lower the better). The results are presented in Table 1.
From Table 1, we can see that DCCD-CONF outperforms all the baselines across all the three cell conditions, showcasing the efficacy of the model and prevalence of confounders in real-world systems. Additionally, we also report the performance of DCCD-CONF and the baselines with respect to MAE on the test data error metric in Table 3 in Appendix C with two additional baselines: DCDFG [47] and Bicycle [56].
Additional experiments. Additionally, we also provide results in Appendix C for the following settings: (i) performance comparison on non-contractive SEMs when the underlying graph is restricted to DAGs, (ii) performance comparison as a function of training data size, (iii) performance comparison as a function of noise variance, (iv) performance comparison as a function outgoing edge density, and (v) performance comparison between DCCD-CONF and additional baselines: JCI-FCI and LiNGAM-MMI.
this section cite: ['b46', 'b55']

Section: Discussion
In this work, we introduced DCCD-CONF, a novel differentiable causal discovery framework that handles directed cycles and unmeasured confounders, assuming Gaussian exogenous noise. It models causal mechanisms via neural networks and learns the causal graph structure by maximizing penalized data likelihood. We provide consistency guarantees in the large-sample regime and demonstrate, through extensive synthetic and real-world experiments, that DCCD-CONF outperforms state-of-the-art methods, maintaining robustness with increasing confounders and graph size. On the Perturb-CITE-seq dataset, our model achieves superior predictive accuracy.
While the focus of this work is limited to Gaussian exogenous noise, we plan to investigate other noise distributions for future research. Other future directions include supporting missing data, and relaxing interventional assumptions by incorporating soft interventions and unknown interventional targets.
this section cite: []

Section: References
Ref_id:b0 Title: Causal protein-signaling networks derived from multiparameter single-cell data Year: (2005)
Ref_id:b1 Title: Learning module networks Year: (2005)
Ref_id:b2 Title: Integrated systems approach identifies genetic nodes and networks in late-onset Alzheimer's disease Year: (2013)
Ref_id:b3 Title: Causation, prediction, and search Year: (2000)
Ref_id:b4 Title: Constraint-based causal discovery from multiple interventions over overlapping variable sets Year: (2015)
Ref_id:b5 Title: Invariant causal prediction for nonlinear models Year: (2018)
Ref_id:b6 Title: Graphical Models: Selecting causal and statistical models Year: (1997)
Ref_id:b7 Title: Characterization and greedy learning of interventional markov equivalence classes of directed acyclic graphs Year: (2012)
Ref_id:b8 Title: DAGs with NO TEARS: Continuous optimization for structure learning Year: (2018)
Ref_id:b9 Title: DAG-GNN: DAG structure learning with graph neural networks Year: (2019)
Ref_id:b10 Title: On the role of sparsity and DAG constraints for learning linear dags Year: (2020)
Ref_id:b11 Title: Masked gradient-based causal structure learning Year: (2022)
Ref_id:b12 Title: Learning sparse nonparametric DAGs Year: (2020-08)
Ref_id:b13 Title: Scaling structural learning with NO-BEARS to infer causal transcriptome networks Year: (2019)
Ref_id:b14 Title: Differentiable causal discovery from interventional data Year: (2020)
Ref_id:b15 Title: The max-min hill-climbing bayesian network structure learning algorithm Year: (2006)
Ref_id:b16 Title: Consistency guarantees for permutation-based causal inference algorithms Year: (2017)
Ref_id:b17 Title: Permutation-based causal inference algorithms with interventions Year: (2017)
Ref_id:b18 Title: Systematic discovery and perturbation of regulatory genes in human T cells reveals the architecture of immune networks Year: (2022-07)
Ref_id:b19 Title: A discovery algorithm for directed cyclic graphs Year: (1996)
Ref_id:b20 Title: Discovering cyclic causal models by independent components analysis Year: (2008)
Ref_id:b21 Title: Estimation rates for sparse linear cyclic causal models Year: (2020-08)
Ref_id:b22 Title: Structure learning for cyclic linear causal models Year: (2020)
Ref_id:b23 Title: Cyclic causal discovery from continuous equilibrium data Year: (2013)
Ref_id:b24 Title: Computation of maximum likelihood estimates in cyclic structural equation models Year: (2019)
Ref_id:b25 Title: Learning linear cyclic causal models with latent variables Year: (2012)
Ref_id:b26 Title: Nodags-flow: Nonlinear cyclic causal structure learning Year: (2023)
Ref_id:b27 Title: An anytime algorithm for causal inference Year: (2001-01)
Ref_id:b28 Title: Joint causal inference from multiple contexts Year: (2020)
Ref_id:b29 Title: Characterization and learning of causal graphs with latent variables from soft interventions Year: (2019)
Ref_id:b30 Title: Causal discovery from soft interventions with unknown targets: Characterization and learning Year: (2020)
Ref_id:b31 Title: Constraint-based causal discovery for non-linear structural causal models with cycles and latent confounders Year: (2018)
Ref_id:b32 Title: Generalization of lingam that allows confounding Year: (2024)
Ref_id:b33 Title: A linear non-gaussian acyclic model for causal discovery Year: (2006)
Ref_id:b34 Title: Learning of maximally ancestral graphs Year: (2025)
Ref_id:b35 Title: Integer programming for causal structure learning in the presence of latent variables Year: (2021)
Ref_id:b36 Title: Differentiable causal discovery under unmeasured confounding Year: (2021)
Ref_id:b37 Title: Doubly robust inference in causal latent factor models Year: (2024)
Ref_id:b38 Title: Double/debiased machine learning for treatment and causal parameters Year: (2024)
Ref_id:b39 Title: Structural equations with latent variables Year: (1989)
Ref_id:b40 Title:  Year: (2009)
Ref_id:b41 Title: Foundations of structural causal models with cycles and latent variables Year: (2021)
Ref_id:b42 Title: Identifiability of gaussian structural equation models with equal error variances Year: (2014)
Ref_id:b43 Title: Implicit normalizing flows Year: (2021)
Ref_id:b44 Title: A class of methods for solving nonlinear simultaneous equations Year: (1965)
Ref_id:b45 Title: Invertible residual networks Year: (2019)
Ref_id:b46 Title: Large-scale differentiable causal discovery of factor graphs Year: (2022)
Ref_id:b47 Title: Lie Groups, Lie Algebras, and Representations Year: (2013)
Ref_id:b48 Title: Residual flows for invertible generative modeling. Advances in Neural Information Processing Systems Year: (2019)
Ref_id:b49 Title: A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines Year: (1989)
Ref_id:b50 Title: Categorical reparameterization with gumbel-softmax Year: (2017)
Ref_id:b51 Title: Sparse inverse covariance estimation with the graphical lasso Year: (2008)
Ref_id:b52 Title: An introduction to probabilistic graphical models Year: (2003)
Ref_id:b53 Title: Dagma: Learning dags via m-matrices and a log-determinant acyclicity characterization Year: (2022)
Ref_id:b54 Title: Multimodal pooled Perturb-CITE-seq screens in patient models define mechanisms of cancer immune evasion Year: (2021)
Ref_id:b55 Title: Bicycle: Intervention-based causal discovery with cycles Year: (2024)
Ref_id:b56 Title: Markov properties for graphical models with cycles and latent variables Year: (2017)
Ref_id:b57 Title: Directed cyclic graphical representations of feedback models Year: (2013)
Ref_id:b58 Title: A factorization criterion for acyclic directed mixed graphs Year: (2009)
Ref_id:b59 Title: Local computation with valuations from a commutative semigroup Year: (1997)
Ref_id:b60 Title: A polynomial time algorithm for determining dag equivalence in the presence of latent variables and selection bias Year: (1997-01)
Ref_id:b61 Title: Markov equivalence for ancestral graphs Year: (2009-10)
Ref_id:b62 Title: Causal reasoning with ancestral graphs Year: (2008)
Ref_id:b63 Title: Adam: A method for stochastic optimization Year: (2015)
