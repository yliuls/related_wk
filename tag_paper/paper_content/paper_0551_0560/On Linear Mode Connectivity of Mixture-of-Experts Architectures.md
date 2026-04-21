Title: On Linear Mode Connectivity of Mixture-of-Experts Architectures
Abstract: Linear Mode Connectivity (LMC) is a notable phenomenon in the loss landscapes of neural networks, wherein independently trained models have been observed to be connected-up to permutation symmetries-by linear paths in parameter space along which the loss remains consistently low. This observation challenges classical views of non-convex optimization and has implications for model ensembling, generalization, and our understanding of neural loss geometry. Inspired by recent studies on LMC in standard neural networks, we systematically investigate this phenomenon within Mixture-of-Experts (MoE) architectures-a class of models known for their scalability and computational efficiency, which combine traditional neural networks-referred to as experts-through a learnable gating mechanism. We begin by conducting a comprehensive analysis of both dense and sparse gating regimes, demonstrating that the symmetries inherent to MoE architectures are fully characterized by permutations acting on both the expert components and the gating function. Building on these foundational findings, we propose a matching algorithm that enables alignment between independently trained MoEs, thereby facilitating the discovery of LMC. Finally, we empirically validate the presence of LMC using our proposed algorithm across diverse MoE configurations-including dense, sparse, and shared-expert variants-under a wide range of model settings and datasets of varying scales and modalities. Our results confirm the existence of LMC in MoE architectures and offer fundamental insights into the functional landscape and optimization dynamics of deep learning models. The code is publicly available at https://github.com/MLResearchX/lmc-moe.

Section: Introduction
Despite the high-dimensional and non-convex nature of deep neural network (DNN) training, stochastic optimization methods-such as stochastic gradient descent (SGD) and its variants-consistently find solutions that generalize well. This empirical success contrasts with the theoretical complexity of the loss landscape, which contains many local minima. A growing body of work has uncovered a surprising phenomenon known as mode connectivity, wherein minima found by independent SGD runs can often be connected by continuous low-loss paths.
Mode Connectivity. Related work on mode connectivity includes [34,45,67,85,61,76,91,94]. Empirical studies demonstrate low-loss connections between independently trained models on MNIST and CIFAR- 10 [31, 33, 21], and have shown that nearly any two solutions can be linked by a lowerror curve [32]. The implications of mode connectivity extend beyond theoretical curiosity. It provides insight into why weight-space ensembling techniques, such as Stochastic Weight Averaging, yield improved generalization [39,65,89]. Moreover, it has proven useful in studying adversarial robustness [93], generalization theory [63,44,55], and the geometry of loss landscapes [35,87,56]. A more restrictive and analytically convenient variant of this phenomenon is linear mode connectivity (LMC), where two models are connected by a linear interpolation in parameter space along which the loss remains low [30,24].
Permutation Invariance. A central challenge in observing LMC stems from the permutation invariance of neural networks: permuting neurons within a hidden layer leaves the network function unchanged [7,23,29,9,60]. Consequently, two functionally equivalent models may appear disconnected in parameter space unless one is suitably permuted. To address this, recent work considers LMC up to permutation, wherein a low-loss linear path exists after aligning hidden units [71,2,36]. Optimal Transport (OT) [86,82,78,79,80] methods have been proposed to compute soft matchings between neurons [71,72], enabling applications such as model fusion in federated learning. Empirically, OT-based alignment achieves near-zero error barrier LMC between independently trained ResNets on CIFAR-10 [2], with connectivity improving with width and degrading with depth. Theoretically, dropout-stable networks have been shown to exhibit mode connectivity [48,70], and [24] demonstrate that LMC up to permutation can already arise at initialization, especially in the NTK regime [41]. [28] further provide formal guarantees for LMC under OT alignment. These insights support the convexity conjecture [25], which posits that the SGD solution set is approximately convex once symmetries are accounted for. This view is strengthened by [68], who propose simultaneous linear connectivity, where a single model aligns linearly with multiple others. Additional studies explore the geometry of the solution space [3,90] and identify star-shaped regions conducive to LMC [73].
Functional Equivalence. Prior work on LMC has primarily focused on feedforward and convolutional architectures, owing to the well-characterized permutation symmetries in these models [11,62,13,77,88]. In contrast, the symmetry structures of more modern architectures-such as Transformers [84,19,12] and Mixture-of-Experts [40,69,52,26]-remain relatively underexplored. The study of these symmetries falls under the broader concept of functional equivalence, which aims to characterize when different parameter configurations yield identical input-output behavior [37,15,27,50,5,6]. Recent work has extended this analysis to attention-based models, with [83,46] examining symmetries in attention mechanisms, particularly permutations of heads and group actions induced by the general linear group. A rigorous understanding of functional equivalence is essential for uncovering and formalizing LMC in such complex, structured architectures.
this section cite: ['b33', 'b44', 'b66', 'b84', 'b60', 'b75', 'b90', 'b93', 'b31', 'b38', 'b64', 'b88', 'b92', 'b62', 'b43', 'b54', 'b34', 'b86', 'b55', 'b29', 'b23', 'b6', 'b22', 'b28', 'b8', 'b59', 'b70', 'b1', 'b35', 'b85', 'b81', 'b77', 'b78', 'b79', 'b70', 'b71', 'b1', 'b47', 'b69', 'b23', 'b40', 'b27', 'b24', 'b67', 'b2', 'b89', 'b72', 'b10', 'b61', 'b12', 'b76', 'b87', 'b83', 'b18', 'b11', 'b39', 'b68', 'b51', 'b25', 'b36', 'b14', 'b26', 'b49', 'b4', 'b5', 'b82', 'b45']

Section: Permutation Alignment Methods.
These methods align parameter permutations to establish LMC. [25] proposed a simulated annealing-based algorithm. [72] employed Optimal Transport, while [4] utilized the Wasserstein Barycenter. [2] introduced three methods: activation matching (using intermediate activations), weight matching (in parameter space), and the Straight-Through Estimator (which minimizes interpolation midpoint loss via gradients); all are based on solving the Linear Assignment Problem [49,42,17]. [36] developed Sinkhorn re-basin, a differentiable method that improves alignment but struggles with residual connections due to layer-independent optimization. Contribution. Motivated by this line of work, we extend the investigation of LMC to Mixture-of-Experts (MoE) architectures, which are related to traditional neural networks in that they aggregate outputs of multiple subnetworks via a gating mechanism. The paper is organized as follows:
1. In Section 3, we introduce the concept of the weight space of MoE architectures and define a group action on this space that preserves the functional behavior of MoE models.
2. In Section 4, we present two core results concerning functional equivalence in MoE models. We demonstrate that the proposed group action characterizes all inherent symmetries of the MoE gating mechanism, with rigorous theoretical justification.
3. In Section 5, we first observe that permutation invariance alone is sufficient to induce LMC. Building on this insight, we develop a Weight Matching algorithm that enables alignment between independently trained MoEs, thereby facilitating the discovery of LMC.
4. In Section 6, we provide empirical evidence of LMC across a wide range of MoE configurationsincluding dense, sparse, and shared-expert variants-evaluated under diverse model settings and across datasets of varying scales and modalities. Additionally, we assess the effectiveness of our proposed expert-matching algorithms, and conduct ablation studies to analyze LMC at different layers within deep MoE models.
Section 2 offers background on LMC and MoE architectures. Table of notation, along with theoretical foundations, experimental details, and additional content, is provided in the Appendix.
this section cite: ['b24', 'b71', 'b3', 'b1', 'b48', 'b41', 'b16', 'b35']

Section: Preliminaries
This section provides an overview of Linear Mode Connectivity and Mixture-of-Experts architectures.
this section cite: []

Section: Linear Mode Connectivity
Let f (•; θ) be a function parameterized by θ ∈ Θ, where Θ denotes the weight space. Given a non-negative loss function L(θ), we seek to minimize L(θ) over Θ. The notion of the loss barrier was originally introduced by [30] and later formalized by [24] as follows: for θ A , θ B ∈ Θ, define
B(θ A , θ B ) = sup t∈[0,1] [L(tθ A + (1 -t)θ B ) -(tL(θ A ) + (1 -t)L(θ B ))] .(1)
θ A and θ B are said to be linearly mode connected if their loss barrier is negligible, i.e., B(θ A , θ B ) ≈ 0.
In many architectures, the function f (•; θ) is invariant under certain permutations of the weight spacenamely, for a permutation g and all θ ∈ Θ, we have f (•; θ) = f (•; gθ). Therefore, θ A and θ B are said to be linearly mode connected up to permutation if there exists a permutation g such that B(θ A , gθ B ) ≈ 0. Any such permutation g is referred to as a winning permutation [24].
this section cite: ['b29', 'b23', 'b23']

Section: Mixture-of-Experts Architectures
A Mixture-of-Experts (MoE) is a neural network architecture that combines the outputs of multiple models into a single prediction through a gating mechanism that assigns input-dependent weights.
The two most commonly used gating strategies are dense and sparse gating. While we provide a brief overview of these concepts here, a comprehensive and formal treatment is deferred to Appendix A.
Expert. Let d denote the input dimension. We define an Expert as a function E(•; θ) : R d R d , parameterized by θ ∈ R e , where e ∈ N represents the total number of trainable parameters of E. In this work, we consider each expert E(•; θ) to be a feedforward neural network with ReLU activation functions. Unless stated otherwise, all experts are assumed to share the same architecture.
this section cite: []

Section: Mixture-of-Experts with dense gating.
Given n denoting the number of experts, we define Mixtureof-Experts with dense gating as a function D : R d R d such that
D x; {W i , b i , θ i } n i=1 = n i=1 softmax i s 1 (x), . . . , s n (x) E(x; θ i ),(2)
where s i (x) = W i x + b i determines the contribution of each expert to the final output. Here, θ i are the parameters of the i th expert. The function s = (s 1 , . . . , s n ) is called the gating score, and is parameterized as s(•; {W i , b i } n i=1 ) with (W i , b i ) ∈ R d × R are the corresponding gating parameters. Mixture-of-Experts with sparse gating. Given a positive integer k ≤ n denoting the number of activated experts, define the Top-k map by Top-k(z) = {i 1 , . . . , i k } for z = (z 1 , . . . , z n ) ∈ R n , where i 1 , . . . , i k are the indices corresponding to the k largest components of x. In the event of ties, we select smaller indices first. We define a Mixture-of-Experts with sparse gating (SMoE) as a function S : R d R d as follows. For x ∈ R d , let T (x) = Top-k(s(x)), then define:
S x; {W i , b i , θ i } n i=1 = i∈T (x) softmax i s i (x) i∈T (x) • E(x; θ i )(3)
In other words, the Top-k selects the k highest-scoring experts used to compute the output.
this section cite: []

Section: Group Action on Weight Space of Mixture-of-Experts
In this section, we introduce the formal notion of the weight space associated with MoE models. Furthermore, we define a group action on this space that preserves the functionality of MoE. A complete and rigorous exposition of these definitions and their implications is provided in Appendix A.
this section cite: []

Section: Weight Space of Mixture-of-Experts
The map D is parameterized by:
ϕ = (W i , b i , θ i ) i=1,...,n ∈ Φ(n) := (R d × R × Θ) n = (R d × R × R e ) n .(4)
Here, Φ(n) is called the weight space of a Mixture-of-n-Experts. Varying the number of experts leads to a general Mixture-of-Experts weight space that spans across expert sets of different sizes, denoted by Φ = ⊔ n>0 Φ(n). Note that the weight space of E coincides with that of S, since the map Top-k does not introduce any new trainable parameters.
this section cite: []

Section: Group Action on Weight Space of Mixture-of-Experts
We define the group G(n) as the direct product G(n) = R d × R × S n of the groups R d , R with addition, and the permutation group S n . Each element g ∈ G(n) is of the form g = (c W , c b , τ ), where c W ∈ R d , c b ∈ R and τ ∈ S n . The group G(n) acts on the weight space Φ(n) as follows. For g ∈ G(n) and ϕ ∈ Φ(n) presented as in Equation ( 4), define:
gϕ := W τ (i) + c W , b τ (i) + c b , θ τ (i) i=1,...,n ∈ Φ(n).(5)
The result below establishes that this group action preserves the MoE map.
this section cite: []

Section: A proof of Proposition 3.1 is presented in Proposition A.3. An analogous invariance result holds in the case of the SMoE function S.
However, since the Top-k selection map is generally discontinuousprimarily due to tie cases in the gating scores-additional conditions are required to ensure the validity of the invariance result. To address this, we focus on a subset of R d where the Top-k scores are unambiguously defined. Specifically, for {W i , b i } n i=1 ∈ (R d × R) n , we define:
Ω {W i , b i } n i=1 := x ∈ R d : (W i x + b i ) n i=1 are pairwise distinct .(6)
The SMoE function S exhibits well-behaved properties on this domain. First, consider the case where the pairs {W i , b i } n i=1 are not pairwise distinct. In this case, the set Ω({W i , b i } n i=1 ) is clearly empty. However, when the pairs {W i , b i } n i=1 are pairwise distinct, the set Ω({W i , b i } n i=1 ) becomes an open and dense subset of R d . Moreover, the SMoE function S is continuous on Ω({W i , b i } n i=1 ). These properties are formally established in Propositions A.1 and A.2. We now show that the invariance property of the SMoE map holds under restriction to this domain. Proposition 3.2 (Weight space invariance of Sparse Mixture-of-Experts). Given the SMoE function S, as defined in Equation (3). Assume that {W i , b i } are pairwise distinct for i = 1, . . . , n. Then, the set Ω(
{W i , b i } n i=1 ) is invariant under the group action of G(n), i.e., for g = (c W , c b , τ ) ∈ G(n), we have Ω({W i , b i } n i=1 ) = Ω({W τ (i) + c W , b τ (i) + c b } n i=1 ). Moreover, the function S, restricted to Ω({W i , b i } n i=1 ), is invariance under the action of G(n) on its weight space Φ(n), i.e. S(•; ϕ) = S(•; gϕ) on Ω({W i , b i } n i=1 ). A proof of Proposition 3.2 is presented in Proposition A.4.
this section cite: []

Section: Remark 3.3.
The permutation invariance of the summation operator and the translation invariance of the softmax function are the two primary sources of invariance for the functions D and S, as stated in Propositions 3.1 and 3.2. In the case of SMoE, these invariance properties are additionally upheld by the permutation and translation invariance of the Top-k operator.
this section cite: []

Section: Symmetries and Functional Equivalence in Mixture-of-Experts
In the remainder of this section, we let ϕ ∈ Φ(n) and ϕ ′ ∈ Φ(n ′ ) denote the parameters of two Mixture-of-Experts models with n and n ′ experts, respectively:
ϕ = (W i , b i , θ i ) i=1,...,n ∈ Φ(n), and ϕ ′ = (W ′ i , b ′ i , θ ′ i ) i=1,...,n ′ ∈ Φ(n ′ ).(7)
this section cite: []

Section: Functional Equivalence in Mixture-of-Experts
We aim to characterize the conditions under which distinct parameter sets yield functionally equivalent MoE models. The central result of this section is that the symmetries of the gating mechanism are fully characterized by the group action of G(n), as defined in Equation ( 5). Given the fundamentally different structural and analytical characteristics of the two gating mechanisms, we treat each case independently in the analysis. In addition, we introduce a set of assumptions, the motivations for which are discussed in detail in the following subsection. We begin with the case of dense gating.
Theorem 4.1 (Functional equivalence in Mixture-of-Experts with Dense Gating). Suppose ϕ, ϕ ′ define the same MoE function, i.e., D(•; ϕ) = D(•; ϕ ′ ). Assume that the following conditions hold: 1. Both {E(•; θ i )} n i=1 and {E(•; θ ′ i )} n ′ i=1 consist of pairwise distinct functions; 2. Both {W i -W j } 1≤i<j≤n and
{W ′ i -W ′ j } 1≤i<j≤n ′ consist of pairwise distinct vectors in R d .
Then n = n ′ , and there exists g = (c W , c b , τ ) ∈ G(n) such that for all i = 1, . . . , n, we have
W ′ i = W τ (i) + c W , b ′ i = b τ (i) + c b , and E(•; θ ′ i ) = E(•; θ τ (i) ) on R d .
For the case of sparse gating, we require the notion of the strongly distinct property. Specifically, two functions f and g defined on R d are said to be strongly distinct if the set {x ∈ R
d : f (x) ̸ = g(x)} is dense in R d . For
instance, distinct polynomials are strongly distinct, whereas distinct ReLU networks are not strongly distinct in general. A formal definition of this property, along with illustrative examples, is provided in Definition C.1 and Example C.2. We now state a result that parallels Theorem 4.1, adapted to the sparse gating regime with k > 1, and established under a set of assumptions that are stronger than those previously required. Theorem 4.2 (Functional equivalence in Mixture-of-Experts with Sparse Gating). Suppose ϕ, ϕ ′ define the same SMoE function, i.e., S(•; ϕ) = S(•; ϕ ′ ). Assume that the following conditions hold: 1. Both {E(•; θ i )} n i=1 and {E(•; θ ′ i )} n ′ i=1 consist of pairwise strongly distinct functions; 2. Both {W i-1 -W i } n i=2 and
{W ′ i-1 -W ′ i } n ′ i=2
are linearly independent subsets of R d . Then n = n ′ , and there exists g = (c W , c b , τ ) ∈ G(n) such that for all i = 1, . . . , n, we have
W ′ i = W τ (i) + c W , b ′ i = b τ (i) + c b , and E(x; θ ′ i ) = E(x; θ τ (i) ) for all x ∈ Ω({W i , b i } n i=1 ) such that τ (i) ∈ Top-k((W j x + b j ) n j=1 ).
The proofs of Theorems 4.1 and 4.2 are provided in Appendix B and Appendix C, respectively. Both arguments build upon two essential ingredients: a result establishing the linear independence of exponential functions, as formulated in Lemma B.3, and a key observation on the piecewise affine structure of ReLU networks, discussed in Appendix B.1.
Remark 4.3. While Theorem 4.2 shares a conceptual parallel with Theorem 4.1, it is important to underscore that the sparse gating case presents significantly deeper mathematical difficulties. The core source of complexity lies in the discontinuous behavior of the Top-k operator, which induces nontrivial discontinuities in the gating function, thereby disrupting smoothness and complicating the functional analysis required to establish equivalence. As a result, standard analytical techniques used in the dense setting become insufficient, necessitating more delicate arguments to account for the combinatorial and piecewise structure inherent to sparse gating.
Theorems 4.1 and 4.2 provide a rigorous characterization of functional equivalence in both dense and sparse MoE models, with particular attention given to the role and structure of the gating mechanism. Nonetheless, it is important to note that these results do not account for all potential symmetries inherent in the MoE and SMoE mappings defined in Equations ( 2) and (3). Notably, further symmetries may exist within the expert networks, especially when they are implemented as ReLU neural networks. Given that this work is primarily concerned with the MoE architecture itself, our analysis is restricted to the behavior of the gating mechanism. Consequently, we treat the experts as black-box functions and abstract away from their internal parameterizations.
this section cite: []

Section: Necessity and Implications of Technical Assumptions
At a conceptual level, symmetry analysis aims to identify universal invariances-those that persist regardless of specific parameter values-while explicitly excluding singular symmetries, which emerge only under special, degenerate configurations of the model parameters. The assumptions introduced in our results are designed precisely to rule out such singularities, which do not represent inherent structural invariances of the architecture but instead arise from pathological or measure-zero subsets of parameter space. We provide a brief overview of the motivation behind these assumptions here; a more detailed justification, along with illustrative examples, is given in Remarks B.6 and C.8.
The case of dense gating. Assumption 1 is introduced to eliminate degenerate scenarios in which two experts compute identical functions and receive identical gating scores-situations where permuting the corresponding experts has no effect on the model's output. By excluding linear dependencies among the gating weight vectors, Assumption 2 ensures that expert activations remain distinguishablethereby preventing the emergence of non-structural, symmetry-like artifacts.
The case of sparse gating. Theorem 4.2 relies on a stronger set of assumptions than those required in Theorem 4.1. This added strength is necessary due to a key feature of sparse gating: an expert's behavior on inputs where it is not selected is unconstrained, meaning that the expert can act arbitrarily outside its region of activation. Consequently, different sets of expert functions may result in the same overall model output, provided their outputs agree where they are active.
The case of k = 1. In the particular case where k = 1, the SMoE architecture employs a Top-1 gating mechanism that activates only the expert associated with the highest gating score. Consequently, the softmax output reduces to a one-hot vector, with a single component equal to 1. Under this regime, the SMoE map exhibits additional nontrivial symmetries, specifically invariance under the multiplicative group R >0 . That is, for any positive scalar c > 0, the following identity holds:
S x; {W i , b i , θ i } n i=1 = S x; {cW i , cb i , θ i } n i=1 .(8)
This invariance holds because the argmax used for expert selection is unaffected by positive scaling, i.e., argmax i=1,...,n (W i x + b i ) = argmax i=1,...,n (cW i x + cb i ), for all x ∈ Ω({W i , b i } n i=1 ). Additionally, since only a single expert contributes to the output, the model lacks any form of explicit aggregation across experts. Due to the analytical challenges posed by these additional invariances, we do not consider k = 1 in our primary results and instead defer its investigation to future work.
this section cite: []

Section: Linear Mode Connectivity of Mixture-of-Experts Architectures
This section outlines the theoretical foundations of LMC in MoE architectures. We first show that permutation invariance suffices to explain the existence of LMC in MoEs, then propose an algorithm to identify such permutations, which we later employ to empirically assess LMC in MoE models.
this section cite: []

Section: Permutation Invariance Sufficiency in Linear Mode Connectivity of Mixture-of-Experts
Theorems 4.1 and 4.2 show that the group action as in Equation (5) suffices to uncover LMC between two MoE models. Each group element g = (c W , c b , τ ) ∈ G(n) consists of two components: a translation term (c W , c b ) ∈ R d × R applied to the gating function, and a permutation τ that reorders both the experts and their associated gating scores. Observe that the translation component does not affect the barrier loss defined in Equation (1). Indeed, for any h = (c W , c b , id n ) ∈ G(n)-with id n denoting the identity permutation in S n -the loss barrier remains unchanged:
B(ϕ A , ϕ B ) = B(ϕ A , hϕ B ).(9)
This invariance, established in Proposition D.1, indicates that only the permutation component τ influences LMC behavior. Combined with prior work showing that permutation symmetries are sufficient to induce LMC in standard neural networks [24,28], this suggests that analyzing permutations alone suffices to capture LMC in MoE models.
this section cite: ['b4', 'b0', 'b23', 'b27']

Section: Permutation Alignment Algorithm for Mixture-of-Experts
We propose the Permutation Alignment Algorithm to align MoEs. Inspired by the data-independent Weight Matching algorithm [3], our method operates in the parameter space of two MoEs, enabling efficient permutation alignment tailored to MoE architectures in large-scale settings. One Billion Word Figure 1: LMC curves for ViT (subplots 1-3) and GPT-2 (subplot 4) with a 4-expert MoE replacement at the first Transformer layer, on CIFAR-100, ImageNet21k CIFAR-100, ImageNet-1k, and One Billion Word datasets, respectively. Plots show consistent low-loss linear interpolation paths between fine-tuned models, indicating strong linear mode connectivity.
this section cite: ['b2']

Section: Proposed Algorithm.
In MoEs, equivalent functionality can arise from different expert orderings and internal neuron configurations. To address this, we propose a two-stage alignment method that first reorders experts to match their roles, then aligns the internal weights of corresponding experts. Both steps are formulated as Linear Assignment Problems (LAPs) [10]. An LAP assigns N tasks to N agents using a cost matrix C ∈ R N ×N , where each entry C i,j represents the cost of assigning task i to agent j. The objective is to find a permutation π that minimizes the total cost: min π∈S N N i=1 C i,π(i) . This is efficiently solved using the Hungarian algorithm in O(N 3 ) time [49,42,17]. Suppose we have ϕ = (W i , b i , θ i ) i=1,...,n and ϕ ′ = (W ′ i , b ′ i , θ ′ i ) i=1,...,n representing the parameters of two distinct MoEs with n experts. In practice, each expert is implemented as an MLP with one hidden layer [26,52,22]. We denote the parameters of expert i in ϕ and expert j in ϕ ′ as θ i = {A i , u i , B i , v i } and θ ′ j = {A ′ j , u ′ j , B ′ j , v ′ j }, respectively, where h+1) , and similarly for A ′ j and B ′ j , as concatenated matrices. Matching Order of Experts. In MoEs, experts and their gating can be reordered without affecting functionality. We propose two methods to align the expert order between ϕ and ϕ ′ , using LAP.
A i ∈ R h×d , u i ∈ R h , B i ∈ R d×h , v i ∈ R d , with h as the hidden dimension. Denote A i = [A i , u i ] ∈ R h×(d+1) , B i = [B i , v i ] ∈ R d×(
this section cite: ['b9', 'b48', 'b41', 'b16', 'b25', 'b51', 'b21']

Section: Method 1 -Gating Weights.
To address the translation-invariance of the softmax function in the gating, we center the weights and biases of the gating. For ϕ, these are W i = W i -1 n n m=1 W m and bi = b i -1 n n m=1 b m per expert i, and similarly for ϕ ′ . The cost matrix with respect to gating weights is defined as C = {C i,j } 1≤i≤n,1≤j≤n ∈ R n×n , where
C i,j = ∥ W i -W ′ j ∥ 2 2 + ( bi -b′ j ) 2 1 2 ,(10)
measuring dissimilarity between centered gating parameters of expert i in ϕ and expert j in ϕ ′ .
this section cite: []

Section: Method 2 -Internal Weights.
Experts exhibit permutation invariance in hidden neuron orderings. We use Gram matrices for a permutation-invariant representation (see Appendix D.2). The cost matrix with respect to expert internal weights is defined as C = {C i,j } 1≤i≤n,1≤j≤n ∈ R n×n , where
C i,j = ( A i ) ⊤ A i -( A ′ j ) ⊤ A ′ j 2 F + B i ( B i ) ⊤ -B ′ j ( B ′ j ) ⊤ 2 F 1 2 ,(11)
quantifying dissimilarity between weight representations of expert i in ϕ and expert j in ϕ ′ .
For both methods, the optimal expert ordering τ is determined by solving the LAP: τ = arg min π∈Sn n i=1 C i,π(i) in O(n 3 ) time. Aligning Expert Internal Weights. After finding the permutation τ to align experts between ϕ and ϕ ′ , we align the internal weights of each matched expert pair (i, τ (i)) using established MLP alignment methods [2,36,25,72], specifically the Weight Matching algorithm [3]. This LAP-based algorithm aligns two MLPs with one hidden layer of size h in O(h 3 ) time, ensuring efficient scaling. Applying this to each pair (i, τ (i)) guarantees consistent neuron ordering across aligned experts.
this section cite: ['b1', 'b35', 'b24', 'b71', 'b2']

Section: Weight Matching Algorithm
We propose a Weight Matching Algorithm designed to align MoEs by addressing permutation invariance in expert ordering and internal parameters, as detailed in Algorithm 1. The algorithm employs gate-based and expert-based methods to optimally permute experts, with both performing comparably well in loss barrier evaluations during model interpolation (see Section 6.3). Operating solely in parameter space, the algorithm avoids data-dependent computational overhead, with a complexity of O(n 3 + nh 3 ), ensuring scalability for large-scale MoE alignment tasks. Lnaive-Ltop1 × 10 2 across layer placements, averaged over 10 checkpoint pairs. Please refer to Tables 8 and 9 for SMoE and DeepSeekMoE variants. Dataset Layer replaced Expert Weight Matching Gate Weight Matching Dataset Layer replaced Expert Weight Matching Gate Weight Matching Rank L Rank L Rank L Rank L CIFAR-10 1 2.50 ± 1.50 2.12 ± 0.42 3.00 ± 1.00 3.42 ± 0.55 CIFAR-100 1 2.80 ± 0.40 3.17 ± 0.25 2.90 ± 0.70 2.73 ± 1.03 4 2.10 ± 0.50 1.04 ± 0.32 2.60 ± 0.92 1.54 ± 0.44 4 3.60 ± 1.20 1.15 ± 0.55 2.70 ± 1.00 2.03 ± 0.93 8 2.70 ± 0.46 0.60 ± 0.27 2.90 ± 0.30 0.74 ± 0.17 8 3.30 ± 0.78 0.67 ± 0.14 3.20 ± 0.87 1.13 ± 0.55 12 4.60 ± 2.00 0.13 ± 0.05 3.80 ± 1.66 0.09 ± 0.03 12 3.40 ± 0.92 0.07 ± 0.03 4.20 ± 0.89 0.11 ± 0.04 Model 1 Model 2 0.5 1.0 1.5 2.0 2.5 Test Loss CIFAR-10 Naive Expert Gating Others Model 1 Model 2 0.06 0.08 0.10 0.12 0.14 CIFAR-10 Model 1 Model 2 1 2 3 4 5 CIFAR-100 Model 1 Model 2 0.4 0.5 0.6 0.7 0.8 0.9
this section cite: []

Section: Algorithm 1 Weight Matching for Mixture-of-Experts
Input: MoE model weights ϕ = (W i , b i , θ i ) i=1,...,n , ϕ ′ = (W ′ i , b ′ i , θ ′ i ) i=1
CIFAR-100 path. We examine three MoE variants: dense MoE [40,43], SMoE with top-2 routing (k = 2) [26], and DeepSeekMoE with top-2 and one shared expert (k = 2, s = 1) (see Appendix D.3 for formal formulations) [54].
this section cite: ['b39', 'b42', 'b25', 'b53']

Section: Datasets and Models.
We use ViT [20] for image classification (MNIST [51], CIFAR-10/100 [47], ImageNet [18]) and GPT-2 [64] for language modeling (WikiText103 [59] and One Billion Word [14]). Hyperparameters such as batch size, optimizer, number of experts, and hidden size are fixed, while the learning rate is tuned per setting. Full details are provided in Appendix F.
this section cite: ['b19', 'b50', 'b46', 'b17', 'b63', 'b58', 'b13']

Section: Linear Mode Connectivity Verification Experimental Objective.
We analyze LMC in pretrained Transformer models where the FFN of a Transformer layer is replaced with an MoE. Our study focuses on two configurations: replacing the FFN in the first, last and all Transformer layers. For each configuration, we fine-tune three independently initialized models. We vary the number of experts (e.g., 2, 4, 6, 8, 16) and the number of Transformer layers to assess LMC under different architectural scales. LMC is evaluated by linearly interpolating between all model pairs using our proposed algorithm (Algorithm 1).
Results. Table 1 shows results for first-layer MoE replacement, with full details for three MoE variants in Table 5. The results for both the last-layer and all-layer configurations are provided in Appendix G.2 and G.3. LMC persists in all cases, but is less pronounced in the last layer due to greater stability and reduced influence on convergence. Early layers strongly shape the solution basin, making first-layer changes more revealing of connectivity. This depth-dependent effect is also demonstrated in Section 6.4 and Appendix E. Figure 1 presents representative LMC curves across our testing cases.
6.3 Expert Order Matching Method Experimental Objective. We evaluate two Expert Order Matching methods (Step 1, Algorithm 1)-Expert Weight Matching and Gate Weight Matching-using 12-layer ViT models with 4-expert MoE replacements. The methods are assessed by ranking the selected permutation among all 24 possible expert permutations after Weight Matching (Step 2), with MoE integrations at layers 1, 4, 8, or 12 on CIFAR-10 and CIFAR-100, repeated five times. We report the rank and a scaled metric
this section cite: []

Section: L =
Lmethod-Ltop1
Lnaive-Ltop1 × 10 2 , averaged over ten checkpoint pairs, where L method , L top1 , and L naive denote loss barriers of our methods, the best permutation, and the naive interpolation, respectively.
Results. Table 2 shows both methods achieve low ranks and near-zero L, indicating near-optimal matching. Full results are presented in Tables 8 (loss) and 9 (accuracy). Figure 2 visualizes the representative performance of our methods across all 24 permutations, highlighting the importance of correct expert order matching, as poor ordering can yield performance close to the naive interpolation. To further validate the importance of expert alignment, we extend our analysis to the large-scale One Billion Word dataset. As shown in Table 10, the proposed Total Weight Matching consistently yields lower loss barriers compared to the Skipping-Expert-Order Matching baseline across all MoE variants, confirming that incorrect expert ordering substantially hampers interpolation smoothness. These results demonstrate that accurate Expert-Order Matching is crucial for revealing true Linear Mode Connectivity between MoE models, as even minor permutation mismatches can disrupt the underlying shared representation space.
this section cite: []

Section: Ablation Study on Number of Layers
This study investigates the impact of the number of Transformer layers on LMC in MoE models on CIFAR-10. We evaluate models with 2, 4, and 6 layers, incorporating MoE at all possible layer positions. Employing Algorithm 1, we compute ratios for four metrics-loss barrier, loss Area Under the Curve (AUC, relative to the straight line connecting the metrics of the two endpoint models), accuracy barrier, and accuracy AUC, relative to linear interpolation-across five repeated experiments, evaluating ten model pairs per configuration. Table 3 reports results for MoE, showing significant reduction ratios across all configurations. Corresponding results for SMoE and DeepSeekMoE are provided in Table 11, with detailed loss and accuracy metrics presented in Tables 12 and 13.
this section cite: []

Section: Conclusion
This paper presents a systematic analysis of LMC in MoE models. We show that functional equivalence via permutation symmetries is sufficient to construct low-loss linear paths-when they existbetween independently trained models. To enable this, we introduce an algorithm for identifying such permutations. Empirical evaluations confirm the presence of LMC across a wide range of MoE configurations-including dense, sparse, and shared-expert variants-under diverse hyperparameter settings and across datasets of varying scales and modalities. While our method does not provide theoretical bounds on the loss barrier, this remains a common limitation across prior LMC studies. Overall, our work offers a principled foundation for extending LMC analysis to other architectures, such as Transformers and State Space Models.
Table of Notation R d d-dimensional Euclidean space S d-1 (d -1)-dimensional hypersphere ∥ • ∥ 2 Euclidean norm ∥ • ∥ F Frobenius norm E Expert function D Mixture-of-Experts with Dense gating S Mixture-of-Experts with Sparse gating Top-k(•) The Top-k map θ, θ ′ parameter of Expert functions Θ weight space of Expert functions ϕ, ϕ ′ parameter of Mixture-of-Experts Φ(n) weight space of Mixture-of-n-Experts S n permutation group of order n τ element of permutation group c W , c b elements of additive group of Euclidean space G, G(n) group act on weight space of Mixture-of-n-Experts Ω, Ω 1 , Ω 2 sets with specific purposes B barrier loss C, C ij cost matrix and its entries A, B, u, v weight and bias of Expert functions P permutation matrix ∂ boundary of a set in a topological space σ, ReLU rectifier activation function H space of holomorphic functions F space of meromorphic functions C[x], C[x 1 , . . . , x n ] polynomial ring in complex variables C(x), C(x 1 , . . . , x n ) field of rational functions p i , r i polynomial Appendix of "On Linear Mode Connectivity of Mixture-of-Experts Architectures" Table of Contents A On the Weight Spaces of Mixture-of-Experts Architecture A.1 Weight Space of Mixture-of-Experts . . . . . . . . . . . . . . . . . . . . . . . . . A.2 Group Action on Weight Spaces of Mixture-of-Experts . . . . . . . . . . . . . . . B Results on Mixture-of-Experts with Dense Gating B.1 A Structural Property of Neural Networks with ReLU Activation . . . . . . . . . . B.2 A Technical Lemma on Holomorphic Functions in C n . . . . . . . . . . . . . . . . B.3 Functional Equivalence in Mixture-of-Experts with Dense Gating . . . . . . . . . C Results on Mixture-of-Experts with Sparse Gating C.1 Strongly Distinctness Property . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.2 Functional Equivalence in Mixture-of-Experts with Sparse Gating . . . . . . . . . D Technical Details for Sections 5 and 6 D.1 Proof for the Sufficiency of Permutation Invariance in LMC of MoE . . . . . . . . D.2 Proof of the Permutation-Invariant Property for Equation (11) . . . . . . . . . . . . D.3 Formal formulation of DeepSeekMoE . . . . . . . . . . . . . . . . . . . . . . . . E Impact of Feedforward Reinitialization on Pretrained Transformer Performance F Experimental Details and Hyperparameters G Experimental Results G.1 Verification of Linear Mode Connectivity across diverse configurations . . . . . . . G.1.1 Dense Mixture-of-Experts . . . . . . . . . . . . . . . . . . . . . . . . . . G.1.2 Sparse Mixture-of-Experts . . . . . . . . . . . . . . . . . . . . . . . . . . G.1.3 DeepSeek Mixture-of-Experts . . . . . . . . . . . . . . . . . . . . . . . . G.2 Linear Mode Connectivity Analysis: Last Layer . . . . . . . . . . . . . . . . . . . G.3 Linear Mode Connectivity Analysis: All Layer . . . . . . . . . . . . . . . . . . . G.4 Expert Matching Method . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . G.5 Ablation Study on Number of Layers . . . . . . . . . . . . . . . . . . . . . . . . .
this section cite: []

Section: References
Ref_id:b0 Title: Complex Analysis Year: (1979)
Ref_id:b1 Title: Git re-basin: Merging models modulo permutation symmetries Year: (2022)
Ref_id:b2 Title: Git Re-Basin: Merging Models modulo Permutation Symmetries Year: (2022-12)
Ref_id:b3 Title: Wasserstein barycenter-based model fusion and linear mode connectivity of neural networks Year: (2022)
Ref_id:b4 Title: For neural networks, function determines form Year: (1993)
Ref_id:b5 Title: Identifiability of discrete-time neural networks Year: (1993)
Ref_id:b6 Title: A convergence theory for deep learning via over-parameterization Year: (2019)
Ref_id:b7 Title: Dbpedia: A nucleus for a web of open data Year: (2007)
Ref_id:b8 Title: Reconciling modern machinelearning practice and the classical bias-variance trade-off Year: (2019)
Ref_id:b9 Title: Network optimization: continuous and discrete models Year: (1998)
Ref_id:b10 Title: Weight-space symmetry in deep networks gives rise to permutation saddles, connected by equal-loss valleys across the loss landscape Year: (2019)
Ref_id:b11 Title: Language models are few-shot learners Year: (2020)
Ref_id:b12 Title: Functional vs. parametric equivalence of relu networks Year: (2020)
Ref_id:b13 Title: One billion word benchmark for measuring progress in statistical language modeling Year: (2013)
Ref_id:b14 Title: On the geometry of feedforward neural network error surfaces Year: (1993)
Ref_id:b15 Title: Functions of One Complex Variable Year: (1978)
Ref_id:b16 Title: On implementing 2d rectangular assignment algorithms Year: (2016)
Ref_id:b17 Title: Imagenet: A largescale hierarchical image database Year: (2009)
Ref_id:b18 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b19 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b20 Title: Essentially no barriers in neural network energy landscape Year: (2018-07)
Ref_id:b21 Title: Glam: Efficient scaling of language models with mixture-of-experts Year: (2022)
Ref_id:b22 Title: Gradient descent finds global minima of deep neural networks Year: (2019)
Ref_id:b23 Title: The role of permutation invariance in linear mode connectivity of neural networks Year: (2021)
Ref_id:b24 Title: The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks Year: (2022-07)
Ref_id:b25 Title: Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity Year: (2022)
Ref_id:b26 Title: Recovering a feed-forward net from its output Year: (1993)
Ref_id:b27 Title: Proving linear mode connectivity of neural networks via optimal transport Year: (2024)
Ref_id:b28 Title: The lottery ticket hypothesis: Finding sparse, trainable neural networks Year: (2018)
Ref_id:b29 Title: Linear mode connectivity and the lottery ticket hypothesis Year: (2020)
Ref_id:b30 Title: Topology and geometry of half-rectified network optimization Year: (2016)
Ref_id:b31 Title: Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs Year: (2018-10)
Ref_id:b32 Title: Loss surfaces, mode connectivity, and fast ensembling of dnns Year: (2018)
Ref_id:b33 Title: Qualitatively characterizing neural network optimization problems Year: (2014)
Ref_id:b34 Title: Using mode connectivity for loss landscape analysis Year: (2018)
Ref_id:b35 Title: Re-basin via implicit Sinkhorn differentiation Year: (2023-06)
Ref_id:b36 Title: On the algebraic structure of feedforward network weight spaces Year: (1990)
Ref_id:b37 Title: The human knowledge compression contest Year: (2012)
Ref_id:b38 Title: Averaging weights leads to wider optima and better generalization Year: (2018)
Ref_id:b39 Title: Adaptive mixtures of local experts Year: (1991)
Ref_id:b40 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b41 Title: A shortest augmenting path algorithm for dense and sparse linear assignment problems Year: (1988)
Ref_id:b42 Title: Hierarchical mixtures of experts and the em algorithm Year: (1994)
Ref_id:b43 Title: Linear connectivity reveals generalization strategies Year: (2022)
Ref_id:b44 Title: On large-batch training for deep learning: Generalization gap and sharp minima Year: (2016)
Ref_id:b45 Title: Accelerating training with neuron interaction and nowcasting networks Year: (2024)
Ref_id:b46 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b47 Title: Explaining landscape connectivity of low-cost solutions for multilayer nets Year: (2019)
Ref_id:b48 Title: The hungarian method for the assignment problem Year: (1955)
Ref_id:b49 Title: Functionally equivalent feedforward neural networks Year: (1994)
Ref_id:b50 Title: Gradient-based learning applied to document recognition Year: (1998)
Ref_id:b51 Title: Gshard: Scaling giant models with conditional computation and automatic sharding Year: (2020)
Ref_id:b52 Title: Visualizing the loss landscape of neural nets Year: (2018)
Ref_id:b53 Title: Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model Year: (2024)
Ref_id:b54 Title: Mechanistic mode connectivity Year: (2023)
Ref_id:b55 Title: Analyzing monotonic linear interpolation in neural network loss landscapes Year: (2021)
Ref_id:b56 Title: Learning word vectors for sentiment analysis Year: (2011)
Ref_id:b57 Title: Building a large annotated corpus of english: The penn treebank Year: (1993)
Ref_id:b58 Title: Pointer sentinel mixture models Year: (2016)
Ref_id:b59 Title: Towards understanding the role of over-parametrization in generalization of neural networks Year: (2018)
Ref_id:b60 Title: What is being transferred in transfer learning? Advances in neural information processing systems Year: (2020)
Ref_id:b61 Title: Sensitivity and generalization in neural networks: an empirical study Year: (2018)
Ref_id:b62 Title: Deep networks on toroids: removing symmetries reveals the structure of flat regions in the landscape geometry Year: (2022)
Ref_id:b63 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b64 Title: Diverse weight averaging for out-of-distribution generalization Year: (2022)
Ref_id:b65 Title: Real and Complex Analysis Year: (1987)
Ref_id:b66 Title: Empirical analysis of the hessian of over-parametrized neural networks Year: (2017)
Ref_id:b67 Title: Simultaneous linear connectivity of neural networks modulo permutation Year: (2024)
Ref_id:b68 Title: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer Year: (2017)
Ref_id:b69 Title: Landscape connectivity and dropout stability of sgd solutions for over-parameterized neural networks Year: (2020)
Ref_id:b70 Title: Model fusion via optimal transport Year: (2020)
Ref_id:b71 Title: Model Fusion via Optimal Transport Year: (2021-02)
Ref_id:b72 Title: Do deep neural network solutions form a star domain Year: (2024)
Ref_id:b73 Title: Complex Analysis, volume 2 of Princeton Lectures in Analysis Year: (2003)
Ref_id:b74 Title: How to train your vit? data, augmentation, and regularization in vision transformers Year: (2021)
Ref_id:b75 Title: Optimizing mode connectivity via neuron alignment Year: (2020)
Ref_id:b76 Title: An Nguyen The, and Tan Nguyen. Monomial matrix group equivariant neural functional networks Year: (2024)
Ref_id:b77 Title: Spherical tree-sliced Wasserstein distance Year: (2025)
Ref_id:b78 Title: Distance-based tree-sliced Wasserstein distance Year: (2025)
Ref_id:b79 Title: Tree-sliced Wasserstein distance: A geometric perspective Year: (2025)
Ref_id:b80 Title: A clifford algebraic approach to e (n)-equivariant high-order graph neural networks Year: (2024)
Ref_id:b81 Title: Tree-sliced Wasserstein distance with nonlinear projection Year: (2025)
Ref_id:b82 Title: Equivariant neural functional networks for transformers Year: (2024)
Ref_id:b83 Title: Attention is all you need Year: (2017)
Ref_id:b84 Title: Spurious valleys in one-hidden-layer neural network optimization landscapes Year: (2019)
Ref_id:b85 Title: Optimal Transport: Old and New Year: (2008)
Ref_id:b86 Title: What can linear interpolation of neural network loss landscapes tell us Year: (2022)
Ref_id:b87 Title: Equivariant polynomial functional networks Year: (2024)
Ref_id:b88 Title: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time Year: (2022)
Ref_id:b89 Title: A compact representation for bayesian neural networks by removing permutation symmetry Year: (2023)
Ref_id:b90 Title: On convexity and linear mode connectivity in neural networks Year: ()
Ref_id:b91 Title: Character-level convolutional networks for text classification Year: (2015)
Ref_id:b92 Title: Bridging mode connectivity in loss landscapes and adversarial robustness Year: (2020)
Ref_id:b93 Title: Going beyond linear mode connectivity: The layerwise linear feature connectivity Year: (2023)
