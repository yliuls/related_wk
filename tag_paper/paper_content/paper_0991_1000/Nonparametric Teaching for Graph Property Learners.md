Title: Nonparametric Teaching for Graph Property Learners
Abstract: Inferring properties of graph-structured data, e.g., the solubility of molecules, essentially involves learning the implicit mapping from graphs to their properties. This learning process is often costly for graph property learners like Graph Convolutional Networks (GCNs). To address this, we propose a paradigm called Graph Neural Teaching (GraNT) that reinterprets the learning process through a novel nonparametric teaching perspective. Specifically, the latter offers a theoretical framework for teaching implicitly defined (i.e., nonparametric) mappings via example selection. Such an implicit mapping is realized by a dense set of graph-property pairs, with the GraNT teacher selecting a subset of them to promote faster convergence in GCN training. By analytically examining the impact of graph structure on parameter-based gradient descent during training, and recasting the evolution of GCNs-shaped by parameter updates-through functional gradient descent in nonparametric teaching, we show for the first time that teaching graph property learners (i.e., GCNs) is consistent with teaching structureaware nonparametric learners. These new findings readily commit GraNT to enhancing learning efficiency of the graph property learner, showing significant reductions in training time for graphlevel regression (-36.62%), graph-level classification (-38.19%), node-level regression (-30.97%) and node-level classification (-47.30%), all while maintaining its generalization performance.

Section: Introduction
Graph-structured data, commonly referred to as graphs, are typically represented by vertices and edges (Hamilton et al., Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). 2017; Chami et al., 2022). The vertices, or nodes, contain individual features, while the edges link these nodes and capture the structural information, collectively forming a complete graph. Graph properties can be categorized as either node-level or graph-levelfoot_0 . For example, the node category is a node-level property in social network graphs (Fan et al., 2019), while the solubility of molecules is a graphlevel property in molecular graphs (Ramakrishnan et al., 2014). Inferring these graph properties essentially involves learning the implicit mapping from graphs to these properties (Hamilton et al., 2017). An intuitive illustration of this mapping is provided in Figure 1. As a representative graph property learner, the Graph Convolutional Network (GCN) (Defferrard et al., 2016;Kipf & Welling, 2017) has shown strong generalizability, delivering impressive performance across various fields such as social networks (Min et al., 2021;Li et al., 2023), quantum chemistry (Gilmer et al., 2017;Mansimov et al., 2019), and biology (Stark et al., 2006;Burkhart et al., 2023).
However, the learning process of the implicit mapping-i.e., the training-can be quite expensive for GCNs, particularly when dealing with large-scale graphs (Liu et al., 2022a). For example, learning node-level properties in realworld e-commerce relational networks involves millions of nodes (Robinson et al., 2024). In the case of graph-level property learning tasks, the scale can become prohibitively large (Hu et al., 2021). As a result, there is a pressing need to reduce training costs and improve the learning efficiency.
Recent studies on nonparametric teaching (Zhang et al., 2023b;a;2024a) offer a promising solution to the above problem. Specifically, nonparametric teaching provides a theoretical framework for efficiently selecting examples when the target mapping (i.e., either a function or a model) being taught is nonparametric, i.e., implicitly defined. It builds on the idea of machine teaching (Zhu, 2015;Zhu et al., 2018)-involving designing a training set (dubbed the teaching set) to help the learner rapidly converge to the target functions-but relaxes the assumption of target functions being parametric (Liu et al., 2017;2018), allowing for the teaching of nonparametric (viz. non-closed-form) target functions, with a focus on function space. Unfortunately, these studies focus solely on regular feature data and overlook the structural aspects of the inputs, resulting in difficulty when the inputs are irregular graphs-universal data structures that include both features and structure (Chami et al., 2022). Moreover, the update of a GCN is generally carried out through gradient descent in parameter space, leading to a gap compared to the functional gradient descent used in nonparametric teaching within function space (Zhang et al., 2023b;a;2024a). These call for more examination prior to the adoption of nonparametric teaching theory in the context of graph property learning.
To this end, we systematically explore the impact of graph structure on GCN gradient-based training in both parameter and function spaces. Specifically, we analytically examine the impact of the adjacency matrix, which encodes the graph structure, on parameter-based gradient descent within parameter space, and explicitly show that the parameter gradient maintains the same form when the graph size is scaled. The structure-aware update in parameter space drives the evolution of GCN, which can be expressed using the dynamic graph neural tangent kernel (GNTK) (Du et al., 2019;Krishnagopal & Ruiz, 2023), and is then cast into function space. We prove that this dynamic GNTK converges to the structure-aware canonical kernel utilized in functional gradient descent, suggesting that the evolution of GCN under parameter gradient descent is consistent with that under functional gradient descent. Therefore, it is natural to interpret the learning process of graph properties via the theoretical lens of nonparametric teaching: the target mapping is realized by a dense set of graph-property pairs, and the teacher selects a subset of these pairs to provide to the GCN, ensuring rapid convergence of this graph property learner. Consequently, to enhance the learning efficiency of GCN, we introduce a novel paradigm called GraNT, where the teacher adopts a counterpart of the greedy teaching algorithm from nonparametric teaching for graph property learning, specifically by selecting graphs with the largest discrepancy between their property true values and the GCN outputs. Lastly, we conduct extensive experiments to validate the effectiveness of GraNT in a range of scenarios, covering both graph-level and node-level tasks. Our key contributions are listed as follows:
• We propose GraNT, a novel paradigm that interprets graph property learning within the theoretical context of nonparametric teaching. This enables the use of greedy algorithms from the latter to effectively enhance the learning efficiency of the graph property learner, GCN.
• We analytically examine the impact of graph structure on parameter-based gradient descent within parameter space, and reveal the consistency between the evolution of GCN driven by parameter updates and that under functional gradient descent in nonparametric teaching. We further show that the dynamic GNTK, stemming from gradient descent on the parameters, converges to the structure-aware canonical kernel of functional gradient descent. These connect nonparametric teaching theory to graph property learning, thus expanding the applicability of nonparametric teaching in the context of graph property learning.
• We demonstrate the effectiveness of GraNT through extensive experiments in graph property learning, covering regression and classification at both graph and node levels. Specifically, GraNT saves training time for graph-level regression (-36.62%), graph-level classification (-38.19%), node-level regression (-30.97%) and node-level classification (-47.30%), while upkeeping its generalization performance.
this section cite: ['b7', 'b14', 'b50', 'b22', 'b10', 'b29', 'b46', 'b34', 'b18', 'b45', 'b56', 'b51', 'b26', 'b74', 'b75', 'b79', 'b80', 'b37', 'b7', 'b74', 'b75', 'b12', 'b30']

Section: Related Works
Graph property learning. Due to the versatility of graphs in modeling diverse data types (Chami et al., 2022), there has been a recent surge of research interest on graphs (Xia et al., 2021), especially attempts of learning implicit mapping from graph data to interested properties (Guo et al., 2021;Zhuang et al., 2023;Cao et al., 2023) for diverse downstream tasks, such as those related to proteins (Fout et al., 2017;Gligorijević et al., 2021) and molecular fingerprints (Duvenaud et al., 2015). There have been various efforts to the learner design for better mapping learning, such as the GCN learner (Defferrard et al., 2016;Kipf & Welling, 2017), which borrows the idea of convolutional neural networks used in image tasks (LeCun et al., 2015), and the graph attention network, which applies the attention operation (Veličković et al., 2018), and to the learning efficiency (Chen et al., 2018;Liu et al., 2022b;Zhang et al., 2023c), such as normalization (Cai et al., 2021), graph decomposition (Xue et al., 2023) and lazy update (Narayanan et al., 2022). Differently, we approach graph property learning from a fresh perspective of nonparametric teaching (Zhang et al., 2023b;a) and adopt a corresponding version of the greedy algorithm to enhance the training efficiency of GCN.
Nonparametric teaching. Machine teaching (Zhu, 2015;Zhu et al., 2018) focuses on designing a teaching set that enables the learner to quickly converge to a target model function. It can be viewed as the reverse of machine learning: while machine learning seeks to learn a mapping from a given training set, machine teaching aims to construct the set based on a desired mapping. Its effectiveness has been demonstrated across various domains, including crowdsourcing (Singla et al., 2014;Zhou et al., 2018), robustness (Alfeld et al., 2017;Ma et al., 2019;Rakhsha et al., 2020), and computer vision (Wang et al., 2021;Wang & Vasconcelos, 2021). Nonparametric teaching (Zhang et al., 2023b;a) advances iterative machine teaching (Liu et al., 2017;2018) by broadening the parameterized family of target mappings to include a more general nonparametric framework. In addition, the practical effectiveness of this theoretical framework has been confirmed in improving the efficiency of multilayer perceptrons (MLPs) when learning implicit mappings from signal coordinates to their corresponding values (Sitzmann et al., 2020;Tancik et al., 2020;Luo et al., 2023;2024;2025;Zhang et al., 2024a). Nevertheless, the limited focus on the structural aspects of the input in these studies makes it difficult to directly apply their findings to general tasks involving graph-structured data (Hamilton et al., 2017;Chami et al., 2022). This work systematically examines the impact of graph structure and highlights the alignment between the evolution of GCN driven by parameter updates and that guided by functional gradient descent in nonparametric teaching. These insights, for the first time, broaden the scope of nonparametric teaching theory in graph property learning and position our GraNT as a means to improve the learning efficiency of GCN.
this section cite: ['b7', 'b68', 'b21', 'b81', 'b6', 'b15', 'b19', 'b13', 'b10', 'b29', 'b33', 'b60', 'b8', 'b77', 'b5', 'b72', 'b48', 'b74', 'b79', 'b80', 'b54', 'b78', 'b0', 'b44', 'b49', 'b74', 'b37', 'b55', 'b58', 'b41', 'b43', 'b75', 'b22', 'b7']

Section: Background
Notation. 2 Let G (n) = (V, E) be a graph, where V denotes the set of n vertices (nodes) and E denotes the set of edges. The d-dimensional feature vector is denoted as
[x i ] d = (x 1 , • • • , x d ) ⊤ ∈ R d , where the entries x i are in- dexed by i ∈ N d (N d := {1, • • • , d}).
For simplicity, this feature vector may be denoted as x. The collection of feature vectors for all nodes is represented by an n × d feature matrix X n×d (abbreviated as X). The i-th row and j-th column of this matrix, corresponding to the i-th node and j-th feature, are denoted by X (i,:) and X (:,j) , respectively. Equivalently, these can be expressed as e ⊤ i X and Xe j , where e i is a basis vector with its i-th entry equal to 1 and all other entries equal to 0. The structure of the graph G (n
) 2 A notation table is provided in Appendix A.1.
is captured by its adjacency matrix A ∈ R n×n , allowing the graph to be concisely represented as G = (X, A) ∈ G. The property of the graph is denoted by y ∈ Y, where y is a scalar for graph-level properties (i.e., Y ⊆ R) and a vector for node-level properties (i.e., Y ⊆ R n ). A set containing m elements is written as {a i } m . If {a i } m ⊆ {a i } n holds, then {a i } m represents a subset of {a i } n of size m, with indices i ∈ N n . A diagonal matrix with elements a 1 , • • • , a m is denoted by diag(a 1 , • • • , a m ), and if all m entries are identical, it is simplified as diag(a; m).
Consider K(G, G ′ ) : G × G → R
as a symmetric and positive definite graph kernel (Vishwanathan et al., 2010). It can also be written as (Liu & Wang, 2016;Zhang et al., 2023b;a). Rather than assuming the ideal case with a closed-form solution f * , we consider the more practical scenario where the realization of f * is given (Zhang et al., 2023b;a;2024a). For simplicity, we assume the function is scalar-valued, aligning with the focus on the graph level in this discussion 3 . Given the target mapping
K(G, G ′ ) = K G (G ′ ) = K G ′ (G), and for simplicity, K G (•) can be abbreviated to K G . The reproducing kernel Hilbert space (RKHS) H as- sociated with K(G, G ′ ) is defined as the closure of the linear span {f : f (•) = r i=1 a i K(G i , •), a i ∈ R, r ∈ N, G i ∈ G}, equipped with the inner product ⟨f, g⟩ H = ij a i b j K(G i , G j ), where g = j b j K Gj
f * : G → Y, it uniquely returns y † for the corresponding graph G † such that y † = f * (G † ).
With the Riesz-Fréchet representation theorem (Lax, 2002;Schölkopf et al., 2002;Zhang et al., 2023b), the evaluation functional is defined as follows:
Definition 1. Let H be a reproducing kernel Hilbert space with a positive definite graph kernel K G ∈ H, where G ∈ G. The evaluation functional E G (•) : H → R is defined with the reproducing property as follows:
E G (f ) = ⟨f, K G (•)⟩ H = f (G), f ∈ H.(1)
Additionally, for a functional F : H → R, the Fréchet derivative (Coleman, 2012;Liu, 2017;Zhang et al., 2023b) of F is given as follows:
Definition 2. (Fréchet derivative in RKHS) The Fréchet derivative of a functional F : H → R at f ∈ H, denoted by ∇ f F (f ), is implicitly defined by F (f + ϵg) = F (f ) + ⟨∇ f F (f ), ϵg⟩ H + o(ϵ) for any g ∈ H and ϵ ∈ R. This derivative is also a function in H.
Graph convolutional network (GCN) is proposed to learn the implicit mapping between graphs and their properties (Kipf & Welling, 2017;Xu et al., 2018). Specifically, a L-layer GCN f θ (G) ≡ X (L) resembles a L-layer MLP, with the key difference being the feature aggregation at the start of each layer, which is based on the adjacency matrix A (Wu et al., 2019;Krishnagopal & Ruiz, 2023).
X (ℓ) = σ (A + I) κ ℓ X (ℓ-1) W (ℓ) , ℓ ∈ N L-1 X (L) = ρ (A + I) κ L X (L-1) • W (L) ,(2)
where W (ℓ) is the weight matrix at layer ℓ, with dimensions h ℓ-1 × h ℓ , h ℓ denotes the width of layer ℓ with h 0 = d, and κ ℓ denotes the convolutional order at the ℓ-th layer. X (0) is the input feature matrix, σ represents activation function (e.g., ReLU), and ρ refers to the pooling operation (e.g., ρ(a) = 1 ⊤ a for summation pooling).
Nonparametric teaching (Zhang et al., 2023b) is defined as a functional minimization over a teaching sequence D = {(x 1 , y 1 ), . . . (x T , y T )}, where the input x ∈ R d represents regular feature data without considering structure, with the set of all possible teaching sequences denoted as D:
D * = arg min D∈D M( f , f * ) + λ • card(D) s.t. f = A(D).(3)
The formulation above involves three key components: M which quantifies the disagreement between f and f * (e.g.,
L 2 distance in RKHS M( f * , f * ) = ∥ f * -f * ∥ H ), card(•),
which represents the cardinality of the teaching sequence D, regularized by a constant λ, and A(D), which refers to the learning algorithm used by the learners, typically employing empirical risk minimization:
f = arg min f ∈H E x∼P(x) (L(f (x), f * (x)))(4)
with a convex (w.r.t. f ) loss L, which is optimized using functional gradient descent:
f t+1 ← f t -ηG(L, f * ; f t , x t ),(5)
where t = 0, 1, . . . , T is the time step, η > 0 represents the learning rate, and G denotes the functional gradient computed at time t.
To derive the functional gradient, given by
G(L, f * ; f † , x) = E x ∂L(f * , f ) ∂f f † • K x ,(6)
Zhang et al., 2023b;a introduce the chain rule for functional gradients (Gelfand et al., 2000) (see Lemma 3) and use the Fréchet derivative to compute the derivative of the evaluation functional in RKHS (Coleman, 2012) (cf. Lemma 4).
Lemma 3. (Chain rule for functional gradients) For differentiable functions G(F ) : R → R that depend on functionals F (f ) : H → R, the expression
∇ f G(F (f )) = ∂G(F (f )) ∂F (f ) • ∇ f F (f )(7)
is typically referred to as the chain rule.
Lemma 4. The gradient of the evaluation functional at the feature x, defined as E x (f ) = f (x) : H → R, is given by ∇ f E x (f ) = K(x, •), where K(x, x ′ ) : R d × R d → R represents a feature-based kernel.
this section cite: ['b61', 'b36', 'b74', 'b74', 'b75', 'b32', 'b53', 'b74', 'b9', 'b35', 'b74', 'b29', 'b71', 'b66', 'b30', 'b74', 'b17', 'b9']

Section: GraNT
We begin by analyzing the effect of the adjacency matrix on parameter-based gradient descent. Then, by translating the evolution of GCN-driven by structure-aware updates in parameter space-into function space, we show that the evolution of GCN under parameter gradient descent aligns with that under functional gradient descent. Lastly, we present the greedy GraNT algorithm, which efficiently selects graphs with steeper gradients to improve the learning efficiency of GCN.
this section cite: []

Section: Structure-aware update in the parameter space
In GCNs, the structural information of graphs is captured through feature aggregation, expressed as (A + I) κ X, as shown in Equation 2. The use of (A + I) κ limits the flexibility in learning aggregated hidden features, σ((A + I) κ XW ), because it applies the same weights to features aggregated from different convolutional orders within a single layer. This paper considers more flexible GCNs, where the weights for features aggregated from different convolutional orders within a single layer are handled independently (Krishnagopal & Ruiz, 2023). Before presenting the detailed formulation, we introduce the concatenation operation and define A [κ] :
= κ-1 i=0 A i = [I A • • • A κ-1
], an n × κn matrix. By unfolding the aggregated features at different orders and assigning them distinct weights (Krishnagopal & Ruiz, 2023), the flexible GCN can be expressed as
X (ℓ) = σ A [κ ℓ ] diag(X (ℓ-1) ; κ ℓ ) • W (ℓ) , ℓ ∈ N L-1 X (L) = ρ A [κ L ] diag(X (L-1) ; κ L ) • W (L) .(8)
Here, the notations are consistent with those in Equation 2, with the exception that W (ℓ) is the weight matrix of size κ ℓ h ℓ-1 × h ℓ . Figure 2 presents an example that illustrates the workflow of this flexible GCN.
Let the column vector θ ∈ R m represent the weights of all layers in a flattened form, where m denotes the total number of parameters in the GCN. Given a training set of size N , {(G i , y i )|G i ∈ G, y i ∈ Y} N , the parameter update using gradient descent (Ruder, 2016) is expressed as follows:
θ t+1 ← θ t - η N N i=1 ∇ θ L(f θ t (G i ), y i ). (9
)
Due to the sufficiently small learning rate η, the updates are tiny over several iterations, which allows them to be treated as a time derivative and then converted into a differential equation (Du et al., 2019):
∂θ t ∂t = - η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • ∂f θ t (G i ) ∂θ t N . (10
)
The term ∂f θ (G) ∂θ (with the indexes omitted for simplicity), which indicates the direction for updating the parameters, can be written more specifically as
∂f θ (G) ∂θ = ∂X (L) ∂W (L) , ∂X (L) ∂W (L-1) (:,1) , • • • , ∂X (L) ∂W (L-1) (:,h L-1 ) w.r.t. the (L -1)-th layer , • • • , ∂X (L) ∂W (1) (:,1) , • • • , ∂X (L) ∂W (1) (:,h1)   w.r.t. the first layer . (11
)
Here, each term represents the derivative of output X (L) w.r.t. weight column vectors. In contrast to derivatives with regular features as inputs, where the derivatives are independent across features, the adjacency matrix A dictates feature aggregation in these derivatives in a batch manner, where each feature of a single node is treated individually (Du et al., 2019). To clearly demonstrate, in an analytical and explicit manner, how A directs structure-aware updates in the parameter space, we present an example involving the derivative of a two-layer GCN with summation pooling:
∂f θ (G) ∂θ =   ∂X (2) ∂W (2) , ∂X (2) ∂W (1) (:,1) , • • • , ∂X (2) ∂W (1) (:,h1)   , (12
)
where the term ∂X (2) ∂W (2) is given by
1 ⊤ n A [κ2] size: n×κ2n size: κ2n×κ2h1 diag(σ(A [κ1] diag(X (0) ; κ 1 )W (1) ); κ 2 ) size: 1×κ2h1 ,(13)
and for i ∈ N h1 , the term with σ =
∂X (2) ∂W (1) (:,i) is 1 ⊤ A [κ2] size: n×κ2n size: κ2n×κ1h0    σ • A [κ1] diag(X (0) ; κ 1 ) • W (2) (i-h1+h1) • • • σ • A [κ1] diag(X (0) ; κ 1 ) • W (2) (i-h1+κ2h1)    size: 1×κ1h0 , (14)
∂σ(A [κ 1 ] diag(X (0) ;κ1)W (1) (:,i) ) ∂A [κ 1 ] diag(X (0) ;κ1)W (1) (:,i)
. Orange indicates the first layer, green denotes the second layer, and 1 ⊤ n in purple corresponds to the summation poolingfoot_2 . When using ReLU as the activation function, σ • A [κ1] diag(X (0) ; κ 1 ) = σ A [κ1] diag(X (0) ; κ 1 ) . The derivation can be found in Appendix A.2.
When the convolutional order κ is reduced to 1 for all layers, meaning structural information is excluded, the GCN gradient computed for a single input graph exactly matches that of the MLP when applied to a batch composed of the node features. This suggests that the structure-aware, parameterbased gradient is more general than the one used in MLPs, indicating that this work can be seen as a generalization of Zhang et al., 2024a. Furthermore, from the explicit expressions in Equations 13 and 14, it can be observed that the gradient of GCN does not depend on the size of the input graph (i.e., the number of nodes). Instead, it depends on the feature dimension and convolutional order. In other words, the parameter gradient retains the same form even when the input graph size n is scaled.
this section cite: ['b30', 'b30', 'b52', 'b12', 'b12', 'b75']

Section: The functional evolution of GCN
The structure-aware update in the parameter space drives the functional evolution of f θ ∈ H. This variation of f θ , which captures how f θ changes in response to updates in θ, can be derived using Taylor's theorem as follows:
f (θ t+1 ) -f (θ t ) = ⟨∇ θ f (θ t ), θ t+1 -θ t ⟩ + o(θ t+1 -θ t ), (15)
where f (θ † ) ≡ f θ † . In a manner similar to the transformation of parameter updates, it can be rewritten in a differential form (Zhang et al., 2024a):
∂f θ t ∂t = ∂f (θ t ) ∂θ t , ∂θ t ∂t ( * ) +o ∂θ t ∂t .(16)
By plugging in the specific parameter updates, i.e., Equation 10, into the first-order approximation term ( * ) of this variation, we get
∂f θ t ∂t = -η N ∂L(f θ t (Gi),yi) ∂f θ t (Gi) ⊤ N • [K θ t (G i , •)] N + o ∂θ t ∂t , (17
)
where the symmetric and positive definite
K θ t (G i , •) := ∂f θ t (Gi) ∂θ t , ∂f θ t (•) ∂θ t
(see the detailed derivation in Appendix A.3). Due to the inclusion of nonlinear activation functions in f (θ), the nonlinearity of f (θ) with respect to θ causes the remainder o(θ t+1 -θ t ) to be nonzero. In a subtle difference, Jacot et al., 2018;Du et al., 2019;Krishnagopal & Ruiz, 2023 apply the chain rule directly, giving less attention to the convexity of L with respect to θ. This leads to the first-order approximation being derived as the variation, with K θ being referred to as the graph neural tangent kernel (GNTK). It has been shown that the GNTK stays constant during training when the GCN width is assumed to be infinite (Du et al., 2019;Krishnagopal & Ruiz, 2023). However, in practical applications, there is no need for the GCN width to be infinitely large, which leads us to investigate the dynamic GNTK (Figure 5 in Appendix A.3 provides an example of how GNTK is computed).
Consider describing the variation of f θ ∈ H from a highlevel, functional perspective (Zhang et al., 2024a). Using functional gradient descent, it can be expressed as:
∂f θ t ∂t = -ηG(L, f * ; f θ t , {G i } N ),(18)
where the functional gradient is given by:
G(L, f * ; f θ t , {G i } N ) = 1 N ∂L(f θ t (Gi),yi) ∂f θ t (Gi) ⊤ N • [K(G i , •)] N . (19
)
The asymptotic relationship between GNTK and the structure-aware canonical kernel (Vishwanathan et al., 2010;Zhang et al., 2024a) in the context of functional gradient is given in Theorem 5 below, with the proof in Appendix B.
Theorem 5. For a convex loss L and a given training set {(G i , y i )|G i ∈ G, y i ∈ Y} N , the dynamic GNTK, derived from gradient descent on the parameters of a GCN, converges pointwise to the structure-aware canonical kernel in the dual functional gradient with respect to the input graphs. Specifically, the following holds:
lim t→∞ K θ t (G i , •) = K(G i , •), ∀i ∈ N N .(20)
This suggests that GNTK, which incorporates structural information, serves as a dynamic alternative to the structureaware canonical kernel in functional gradient descent with graph inputs, making the GCN's evolution through parameter gradient descent align with that in functional gradient descent (Kuk, 1995;Du et al., 2019;Geifman et al., 2020). This functional insight bridges the teaching of the graph property learner, GCN, with that of structure-aware nonparametric learners, while also simplifying further analysis (e.g., a convex functional L preserves its convexity with respect to f θ from a functional perspective, but is typically nonconvex when considering θ). By leveraging the functional insight and employing the canonical kernel (Dou & Liang, 2021) instead of GNTK (which should be considered alongside the remainder), it aids in deriving sufficient reduction concerning L in Proposition 6, with the proof deferred to Appendix B.
Proposition 6. (Sufficient Loss Reduction) Suppose the convex loss L is Lipschitz smooth with a constant τ > 0, and the structure-aware canonical kernel is bounded above by a constant γ > 0. If the learning rate η satisfies η ≤ 1/(2τ γ), then it follows that a sufficient reduction in L is guaranteed, as shown by
∂L ∂t ≤ - ηγ 2 1 N N i=1 ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) 2 .(21)
This demonstrates that the variation of L over time is capped by a negative value, meaning it decreases by at least the magnitude of this upper bound as time progresses, guaranteeing convergence.
this section cite: ['b75', 'b27', 'b12', 'b30', 'b12', 'b30', 'b75', 'b61', 'b75', 'b31', 'b12', 'b16', 'b11']

Section: GraNT algorithm
Building on the insights regarding the effect of the adjacency matrix, which captures the graph structure, on parameterbased gradient descent, as well as the consistency between teaching a GCN and a nonparametric learner, we introduce the GraNT algorithm. This algorithm seeks to increase the steepness of gradients to improve the learning efficiency of GCN. By interpreting the gradient as a sum of the projec-
tions of ∂L(f θ ,f * ) ∂f θ onto the basis {K(G i , •)} N , increasing the gradient can be achieved by simply maximizing the projection coefficient ∂L(f θ (Gi),yi) ∂f θ (Gi)
, without the need to calculate the norm of the basis ∥K(G i , •)∥ H (Wright, 2015;Zhang et al., 2024a). This suggests that selecting graphs that either maximize
∂L(f θ (Gi),yi) ∂f θ (Gi) or correspond to larger components of ∂L(f θ ,f * ) ∂f θ
can effectively increase the gradient, which implies
{G i } m * = arg max {Gi}m⊆{Gi} N ∂L(f θ (G i ), y i ) ∂f θ (G i ) m 2 . (22
)
Algorithm 1 GraNT Algorithm Input: Target mapping f * realized by a dense set of graphproperty pairs, initial GCN f θ 0 , the size of selected training set m ≤ N , small constant ϵ > 0 and maximal iteration number T .
Set f θ t ← f θ 0 , t = 0.
while t ≤ T and ∥[
f θ t (G i ) -f * (G i )] N ∥ 2 ≥ ϵ do
The teacher selects m teaching graphs:
/ * (Graph-level) Graphs corresponding to the m largest |f θ t (Gi) -f * (Gi)|. * / {G i } m * = arg max {Gi}m⊆{Gi} N ∥[f θ t (G i ) -f * (G i )] m ∥ 2 .
/ * (Node-level) Graphs associated with the m largest
∥f θ t (G i )-f * (G i )∥ 2 n i . * / {G i } m * = arg max {Gi}m⊆{Gi} N f θ t (Gi)-f * (Gi) ni m F , with Frobenius norm ∥ • ∥ F .
Provide {G i } m * to the GCN learner.
The learner updates f θ t based on received {G i } m * :
// Parameter-based gradient descent. θ t ← θ t -η m Gi∈{Gi}m * ∇ θ L(f θ t (G i ), f * (G i )). Set t ← t + 1. end
From a functional standpoint, when handling a convex loss functional L, the norm of the partial derivative of L with respect to f θ , represented as ∥ ∂L(f θ )
∂f θ ∥ H , is positively correlated with ∥f θ -f * ∥ H . As f θ progressively converges to f * , ∥ ∂L(f θ )
∂f θ ∥ H diminishes (Boyd et al., 2004;Coleman, 2012). This relationship becomes especially noteworthy when L is strongly convex with a larger convexity constant (Kakade & Tewari, 2008;Arjevani et al., 2016). Leveraging these insights, the GraNT algorithm selects graphs by
{G i } m * = arg max {Gi}m⊆{Gi} N ∥[f θ (G i ) -f * (G i )] m ∥ 2 . (23
)
The pseudo code, including the node-level version, is provided in Algorithm 1.
this section cite: ['b65', 'b75', 'b3', 'b9', 'b28', 'b1']

Section: Experiments and Results
We start by evaluating GraNT on graph-level regression and classification tasks, then proceed to validate it on nodelevel tasks.
The overall results on the test set are shown in Table 1, which clearly highlights the effectiveness of GraNT in graph property learning: it reduces training time by 36.62% for graph-level regression, 38.19% for graphlevel classification, 30.97% for node-level regression, and 47.30% for node-level classification, all while maintaining 0 5000 10000 15000 20000 25000 30000 Wallclock Time (s) 3.0 3.5 4.0 4.5 5.0 5.5 Validation Loss without GraNT with GraNT (B) with GraNT (S) (a) ZINC Loss. 0 5000 10000 15000 20000 25000 30000 Wallclock Time (s) 0.0050 0.0055 0.0060 0.0065 0.0070 0.0075 Validation MAE without GraNT with GraNT (B) with GraNT (S) (b) ZINC MAE. 0 500 1000 1500 2000 Wallclock Time (s) 0.1 0.2 0.3 0.4 0.5 0.6 Validation Loss without GraNT with GraNT (B) with GraNT (S) (c) ogbg-molhiv Loss. 0 500 1000 1500 2000 Wallclock Time (s) 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 Validation ROC-AUC without GraNT with GraNT (B) with GraNT (S)
(d) ogbg-molhiv ROC-AUC. comparable testing performance. Detailed settings are given in Appendix C.
Given the common practice of training GCN in batches, i.e., graphs are fed in batches, it is both natural and intuitive to implement GraNT at the batch level. This involves selecting batches that exhibit the largest average discrepancy between the actual properties and the corresponding GCN outputs, referred to as GraNT (B). Meanwhile, another variant, called GraNT (S), selects single graph with the largest discrepancies within each batch in proportion, then reorganizes the selected graphs into new batches.
Graph-level tasks. We evaluate GraNT using several widely recognized benchmark datasets as follows:
• QM9 (Wu et al., 2018): 130k organic molecules graphs with quantum chemical properties (regression task);
• ZINC (Gómez-Bombarelli et al., 2018): 250k molecular graphs with bioactivity and solubility chemical properties (regression task);
• ogbg-molhiv (Hu et al., 2020): 41k molecular graphs with HIV inhibitory activity properties (binary classification task);
• ogbg-molpcba (Hu et al., 2020): 438k molecular graphs with bioactivity properties (multi-task binary classification task).
To clearly illustrate the practical efficiency of GraNT, we plot the wallclock time versus loss/metric curves. This is done by conducting a validation after each training epoch, i.e., performing an evaluation on the validation dataset after each training process. Specifically, we display the validation set loss and the typical Mean Absolute Error (MAE) for ZINC in Figure 3 (a) and (b), respectively. In both plots, GraNT Dataset Time (s) Loss ↓ MAE ↓ ROC-AUC ↑ AP ↑ ✗ QM9 9654.81 2.0444 0.0051±0.0009 --ZINC 33033.82 3.1160 0.0048±0.0004 -ogbg-molhiv 2163.50 0.1266 -0.7572±0.0005 ogbg-molpcba 130191.26 0.0577 --0.3270±0.0000 gen-reg 3344.78 0.0086 0.0007±0.0001 -gen-cls 11662.25 0.1314 -0.9150±0.0024 -✓ B S QM9 6392.26 (-33.79%) 2.0436 0.0051±0.0009 --ZINC 20935.24 (-36.62%) 3.1165 0.0048±0.0004 -ogbg-molhiv 1457.39 (-32.64%) 0.1238 -0.7676±0.0036 ogbg-molpcba 80465.06 (-38.19%) 0.0577 --0.3358±0.0001 gen-reg 2308.97 (-30.97%) 0.0086 0.0007±0.0001 -gen-cls 6145.72 (-47.30%) 0.1314 -0.9157±0.0013 -QM9 7076.37 (-26.71%) 2.0443 0.0051±0.0009 --ZINC 22265.83 (-32.60%) 3.1170 0.0048±0.0004 -ogbg-molhiv 1597.69 (-26.15%) 0.1421 -0.7705±0.0027 ogbg-molpcba 89858.65 (-30.98%) 0.0575 --0.3351±0.0025 gen-reg 2337.46 (-30.12%) 0.0086 0.0007±0.0001 -gen-cls 8171.21 (-29.93%) 0.1313 -0.9157±0.0014 -Table 1: Training time and testing results across different benchmarks. GraNT (B) and GraNT (S) demonstrate similar testing performance while significantly reducing training time compared to the "without GraNT", across graph-level (QM9, ZINC, ogbg-molhiv, ogbg-molpcba) and node-level (gen-reg, gen-cls) datasets, for both regression and classification tasks. Time (s) MAE ↓ AL-3DGraph ‡ (Subedi et al., 2024) 9200.27 0.7991 AL-3DGraph ♯ (Subedi et al., 2024) 9364.74 0.4719 AL-3DGraph § (Subedi et al., 2024) 12601.77 0.1682 GraNT (B) 6392.26 0.0051 GraNT (S) 7076.37 0.0051 ‡ : lr=5e-5, batch_size=256, which matches GraNT settings. ♯ : lr=5e-4, batch_size=256.
§ : lr=5e-4, batch_size=32, which corresponds to the default settings used in the provided code for that paper. Table 3: Comparison of GraNT with recent efficient methods on the ogbg-molhiv dataset.
once the wallclock time reaches approximately 500s. However, the curves appear relatively jagged, which can be attributed to the label imbalance in this benchmark dataset. This imbalance also explains why, even when the validation loss decreases significantly, the ROC-AUC curve does not rise to a higher range. The detailed numerical results for training time and testing performance are provided in Table 1. The comparisons between GraNT and recent SOTA methods are shown in Table 2 for QM9 and Table 3 for ogbg-molhiv.
Node-level tasks. We also assess GraNT for node-level property learning using synthetic data. Specifically, we utilize the graphon, a typical limit object of a convergent sequence of graphs (Xu et al., 2021;Xia et al., 2023), to generate two synthetic datasets: gen-reg (containing 50k
0 500 1000 1500 2000 2500 3000 3500 Wallclock Time (s) 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 Validation Loss without GraNT with GraNT (B) with GraNT (S) (a) gen-reg Loss. 0 500 1000 1500 2000 2500 3000 3500 Wallclock Time (s) 0.0010 0.0015 0.0020 0.0025 0.0030 Validation MAE without GraNT with GraNT (B) with GraNT (S) (b) gen-reg MAE. 0 2000 4000 6000 8000 10000 12000 Wallclock Time (s) 0.5 1.0 1.5 2.0 Validation Loss without GraNT with GraNT (B) with GraNT (S) (c) gen-cls Loss. 0 2000 4000 6000 8000 10000 12000 Wallclock Time (s) 0.725 0.750 0.775 0.800 0.825 0.850 0.875 0.900 Validation ROC-AUC without GraNT with GraNT (B) with GraNT (S) (d) gen-cls ROC-AUC. graphs) for regression and gen-cls (containing 50k graphs) for classification.
Figures 4 (a) and (b) illustrate the validation loss and MAE curves for gen-reg, respectively. From both plots, it is clear that GraNT reached convergence more quickly in terms of wallclock time compared to the "without GraNT", highlighting its efficiency.
Figure 4 (c) and (d) show the validation loss and ROC-AUC for gen-cls, respectively. Both plots demonstrate that GraNT requires less wallclock time to converge compared to the "without GraNT". Furthermore, although this dataset is generated with imbalanced labels, similar to ogbg-molhiv, there is a notable difference: when the validation loss is low, the corresponding ROC-AUC exceeds 0.9, which is a relatively high value. This underscores the effectiveness of GraNT. The detailed numerical results for training time and testing performance are shown in Table 1.
All experimental results demonstrate that GraNT (B) and GraNT (S) offer significant time-saving benefits while maintaining comparable generalization performance, and in some cases, even outperforming the "without GraNT". Further experimental results, including result plots for QM9 and ogbgmolpcba, training curves for the aforementioned datasets, and additional validations on the AMD device, can be found in Appendix C.
this section cite: ['b67', 'b20', 'b25', 'b25', 'b70', 'b69']

Section: Concluding Remarks and Future Work
This paper proposes GraNT, a novel paradigm that enhances the learning efficiency of graph property learner (GCN) through nonparametric teaching theory. Specifically, GraNT reduces the wallclock time needed to learn the implicit mapping from graphs to properties of interest by over 30%, while maintaining comparable test performance, as shown through extensive experiments. Furthermore, GraNT establishes a theoretical connection between the evolution of a GCN using parameter-based gradient descent and that of a function using functional gradient descent in nonparametric teaching. This connection between nonparametric teaching theory and GCN training broadens the potential applications of nonparametric teaching in graph property learning.
In future work, it would be interesting to investigate other variations of GraNT for different graph property learners, such as graph attention networks (Veličković et al., 2018). Moreover, exploring the practical applications of GraNT to improve the efficiency of data-driven methods (Henaff, 2020;Touvron et al., 2021;Müller et al., 2022) within the field of graph property learning offers promising opportunities for future progress, particularly in fields like molecular biology and protein research.
this section cite: ['b60', 'b23', 'b59', 'b47']

Section: Impact Statement
Recent interest in learning implicit mappings from graph data to specific properties has grown significantly, especially in science-related fields, driven by the ability of graphs to model diverse types of data. This work focuses on improving the learning efficiency of implicit graph property mappings through a novel nonparametric teaching perspective, which has the potential to positively impact graph-related fields and society. Furthermore, it connects nonparametric teaching theory with GCN training, expanding its application in graph property learning. As a result, it also holds promise for valuable contributions to the nonparametric teaching community.
Table 4: Summary of Key Notations.
A.2. The derivation of structure-aware updates in the parameter space.
Let's focus on the derivative of a two-layer GCN with summation pooling:
∂f θ (G) ∂θ =       ∂X (2) ∂W (2) the second layer , ∂X (2) ∂W (1) (:,1) , • • • , ∂X (2) ∂W (1) (:,h1) the first layer       ⊤ . (24
)
Using the chain rule, we can calculate the derivative of f θ (G) w.r.t. the second-layer weights W (2) , which is a vector of size κ 2 h 1 , as follows:
∂X (2) ∂W (2) = ∂1 ⊤ n A [κ2] diag(X (1) ; κ 2 ) • W (2) ∂W (2) = 1 ⊤ n A [κ2] diag(X (1) ; κ 2 ) = 1 ⊤ n A [κ2] size: n×κ2n size: κ2n×κ2h1 diag(σ(A [κ1] diag(X (0) ; κ 1 )W (1) ); κ 2 )) size: 1×κ2h1 . (25
)
The derivative of f θ (G) w.r.t. the first-layer weights is more complex. For i ∈ N h1
∂X (2) ∂W (1) (:,i) = ∂1 ⊤ A [κ2] diag(X (1) e i e ⊤ i ; κ 2 ) • W (2) ∂W (1) (:,i) = ∂1 ⊤ A [κ2] size: κ2n×κ2 diag(X (1) e i ; κ 2 ) • size: κ2×κ2h1 diag(e ⊤ i ; κ 2 ) W (2) ∂W (1) (:,i) = ∂1 ⊤ A [κ2] size: κ2n×κ2 diag(X (1) e i ; κ 2 ) • size: κ2×1    W (2) (i-h1+h1) • • • W (2) (i-h1+κ2h1)    ∂W (1) (:,i) = ∂1 ⊤ A [κ2] size: κ2n×1    X (1) e i W (2) (i-h1+h1) • • • X (1) e i W (2) (i-h1+κ2h1)    ∂W (1) (:,i) = 1 ⊤ A [κ2]     ∂X (1) ei ∂W (1) (:,i) W (2) (i-h1+h1) • • • ∂X (1) ei ∂W (1) (:,i) W (2) (i-h1+κ2h1)     = 1 ⊤ A [κ2] size: n×κ2n size: κ2n×κ1h0    σ • A [κ1] diag(X (0) ; κ 1 ) • W (2) (i-h1+h1) • • • σ • A [κ1] diag(X (0) ; κ 1 ) • W (2) (i-h1+κ2h1)    size: 1×κ1h0 , (26
) with σ = ∂σ(A [κ 1 ] diag(X (0) ;κ1)W (1) (:,i) ) ∂A [κ 1 ] diag(X (0) ;κ1)W (1) (:,i)
. Orange marks the first-layer elements, green colors the second-layer elements, and
1 ⊤
n in purple refers to the summation pooling. If we use ReLU as the activation function, σ • A [κ1] diag(X (0) ; κ 1 ) = σ A [κ1] diag(X (0) ; κ 1 ) .
For the GCN shown in Figure 2, the derivative, i.e., Equation 25 is specified with κ 1 , κ 2 = 3, 2 as
∂X (2) ∂W (2) = 1 ⊤ A [2]
size: n×2n size: 2n×2h1 diag(σ(A [3] diag(X (0) ; 3)W ( 1) ); 2)
size: 1×2h1 . (27
)
∂X (2) ∂W (1) (:,i) = 1 ⊤ A [2] size: n×2n size: 2n×3h0 σ • A [3] diag(X (0) ; 3) • W (2) (i) σ • A [3] diag(X (0) ; 3) • W (2) (i+h1
) size: 1×3h0 ,(28)
When a graph is input into a GCN, the adjacency matrix A governs the operations between nodes, ensuring the update is structure-aware by performing row-wise operations on the feature matrix. Meanwhile, the weight matrix W controls how the features are processed, by performing column-wise operations on the feature matrix.
K θ (G (3) , G ′ (4) ) = ∂f θ (G) ∂θ , ∂f θ (G ′ ) ∂θ = ∂f θ (G) ∂W (1) (1,1) ∂f θ (G ′ ) ∂W (1) (1,1) + • • • + ∂f θ (G) ∂W (1) (κ 1 d,h 1 ) ∂f θ (G ′ ) ∂W (1) (κ 1 d,h 1 ) + ∂f θ (G) ∂W (2) (1) ∂f θ (G ′ ) ∂W (2) (1) + • • • + ∂f θ (G) ∂W (2) (κ 2 h 1 ) ∂f θ (G ′ ) ∂W (2) (κ 2 h 1 ) . A.3. Graph Neural Tangent Kernel (GNTK)
By substituting the parameter evolution (Equation 10)
∂θ t ∂t = - η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • ∂f θ t (G i ) ∂θ t N .(29)
into the first-order approximation term ( * ) of Equation 16, it obtains
( * ) = ∂f θ t (•) ∂θ t , - η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • ∂f θ t (G i ) ∂θ t N = - η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • ∂f θ t (•) ∂θ t , ∂f θ t (G i ) ∂θ t N = - η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • ∂f θ t (•) ∂θ t , ∂f θ t (G i ) ∂θ t N = - η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • [K θ t (G i , •)] N ,(30)
which derives Equation 17as
∂f θ t ∂t = - η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • [K θ t (G i , •)] N + o ∂θ t ∂t ,(31)
where the symmetric and positive definite
K θ t (G i , •) := ∂f θ t (Gi) ∂θ t , ∂f θ t (•) ∂θ t
is referred to as graph neural tangent kernel (GNTK) (Du et al., 2019;Krishnagopal & Ruiz, 2023). Figure 5 illustrates the GNTK calculation process. In simple terms, examining a model's behavior by focusing on the model itself, rather than its parameters, often involves the use of kernel functions.
It can be observed that the quantity
∂f θ t (•) ∂θ t
representing the partial derivative of the GCN with respect to its parameters,
present in K θ t (G i , •) = ∂f θ t (Gi) ∂θ t , ∂f θ t (•) ∂θ t
, is determined by both the structure and the specific parameters θ t , but does not depend on the input graphs. The other term
∂f θ t (Gi) ∂θ t
relies not only on the GCN structure and specific θ t , but also on the input graph. If the input for
∂f θ t (Gi) ∂θ t
is not specified, the GNTK simplifies to a general form K θ t (•, •). When a specific graph G j is defined as the input for
∂f θ t (•) ∂θ t , GNTK becomes a scalar as K θ t (G i , G j ) = ⟨ ∂f θ t (Gi) ∂θ t , ∂f θ t (Gj ) ∂θ t
⟩. These are in line with the kernel used in functional gradient descent. By providing the input graph G i , one coordinate of K θ t is fixed, causing the GCN to update along K θ t (G i , •), based on the magnitude of
∂f θ t (Gi) ∂θ t
. This process aligns with the core principle of functional gradient descent. In summary, both the GNTK and the canonical kernel are consistent in their mathematical formulation and show alignment in how they affect the evolution of the corresponding GCN. Furthermore, Theorem 5 highlights the asymptotic connection between the GNTK and the canonical kernel used in functional gradient descent.
this section cite: ['b12', 'b30']

Section: B. Detailed Proofs
Before providing the detailed proofs, we first introduce the gradient of an evaluation functional E G (f ).
this section cite: []

Section: Lemma 7. The gradient of an evaluation functional E
G (f ) = f (G) : H → R is ∇ f E G (f ) = K G .
Proof of Lemma 7 Let us define a function ϕ by adding a small perturbation ϵg (ϵ ∈ R, g ∈ H) to f ∈ H, so that ϕ = f + ϵg. ϕ ∈ H since RKHS is closed under addition and scalar multiplication. Therefore, for an evaluation functional
E G [f ] = f (G) : H → R, we can evaluate ϕ at G as E G [ϕ] = E G [f + ϵg] = E G [f ] + ϵE G [g] + 0 = E G [f ] + ϵ⟨K(G, •), g⟩ H + 0 (32
) Recall implicit definition of Fréchet derivative in RKHS (see Definition 2) E G [f + ϵg] = E G [f ] + ϵ⟨∇ f E G [f ], g⟩ H + o(ϵ),
it follows from Equation 32that we have the gradient of a evaluation functional
∇ f E G [f ] = K G .
this section cite: []

Section: ■
Proof of Theorem 5 By describing the evolution of a GCN in terms of parameter variations and from a high-level perspective within the function space, we obtain
- η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • [K(G i , •)] N = - η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • ∂f θ t (G i ) ∂θ t , ∂f θ t (•) ∂θ t N + o ∂θ t ∂t .(33)
After the reorganization, we get
- η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • [K(G i , •) -K θ t (G i , •)] N = o ∂θ t ∂t .(34)
By inserting the evolution of the parameters
∂θ t ∂t = -η ∂L ∂θ t = - η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • ∂f θ t (G i ) ∂θ t N (35
)
into the remainder, we have
- η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • [K(G i , •) -K θ t (G i , •)] N = o - η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N • ∂f θ t (G i ) ∂θ t N .(36)
When training a GCN with a convex loss L, which is convex concerning f θ but often not with regard to θ, we have the limit of the vector lim t→∞ ∂L(f θ t (Gi),yi) ∂f θ t (Gi) N = 0. Since the right side of the equation is a higher order infinitesimal than the left, to preserve this equality, it leads us to the conclusion that
lim t→∞ [K(G i , •) -K θ t (G i , •)] N = 0. (37
)
This means that for each G ∈ {G i } N , GNTK converges pointwise to the canonical kernel.
this section cite: []

Section: ■ Proof of Proposition 6
By recalling the definition of the Fréchet derivative in Definition 2, the convexity of L implies that
∂L ∂t ≤ ∂L ∂f θ t+1 , f θ t ∂t H Ξ .(38)
By identifying the Fréchet derivative of ∂L ∂f θ t+1 and the evolution of f θ t , the right-hand side term Ξ can be represented as
Ξ = G t+1 , -ηG t H = - η N 2 ∂L(f θ t+1 (G i ), y i ) ∂f θ t+1 (G i ) ⊤ N • [K Gi ] N , [K Gi ] ⊤ N • ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) N H = - η N 2 ∂L(f θ t+1 (G i ), y i ) ∂f θ t+1 (G i ) ⊤ N • [K Gi ] N , [K Gi ] ⊤ N H • ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) N = - η N ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) ⊤ N K ∂L(f θ t+1 (G i ), y i ) ∂f θ t+1 (G i ) N ,(39)
where K = K/N , and K is an N × N symmetric, positive definite matrix with elements K(G i , G j ) positioned at the i-th row and j-th column. For convenience, we adopt the simplified notation that
∂L(f θ □ (Gi),yi) ∂f θ □ (Gi) := ∂ f θ □ L(f θ □ ; G i ).
The final term in Equation 39 can be rewritten as
- η N ∂ f θ t L(f θ t ; G i ) ⊤ N K ∂ f θ t+1 L(f θ t+1 ; G i ) N = - η N ∂ f θ t L(f θ t ; G i ) ⊤ N K ∂ f θ t+1 L(f θ t+1 ; G i ) N + ∂ f θ t L(f θ t ; G i ) N -∂ f θ t L(f θ t ; G i ) N = - η N ∂ f θ t L(f θ t ; G i ) ⊤ N K ∂ f θ t L(f θ t ; G i ) N - η N ∂ f θ t L(f θ t ; G i ) ⊤ N K ∂ f θ t+1 L(f θ t+1 ; G i ) N -∂ f θ t L(f θ t ; G i ) N = - η N ∂ f θ t L(f θ t ; G i ) ⊤ N K ∂ f θ t L(f θ t ; G i ) N + η N ∂ f θ t+1 L(f θ t+1 ; G i ) ⊤ N -∂ f θ t L(f θ t ; G i ) ⊤ N -∂ f θ t+1 L(f θ t+1 ; G i ) ⊤ N • K • ∂ f θ t+1 L(f θ t+1 ; G i ) N -∂ f θ t L(f θ t ; G i ) N .(40)
The last term in Equation 40above can be further detailed as
η N ∂ f θ t+1 L(f θ t+1 ; G i ) ⊤ N -∂ f θ t L(f θ t ; G i ) ⊤ N -∂ f θ t+1 L(f θ t+1 ; G i ) ⊤ N • K ∂ f θ t+1 L(f θ t+1 ; G i ) N -∂ f θ t L(f θ t ; G i ) N = η N ∂ f θ t+1 L(f θ t+1 ; G i ) N -∂ f θ t L(f θ t ; G i ) N ⊤ K ∂ f θ t+1 L(f θ t+1 ; G i ) N -∂ f θ t L(f θ t ; G i ) N - η N ∂ f θ t+1 L(f θ t+1 ; G i ) ⊤ N K ∂ f θ t+1 L(f θ t+1 ; G i ) N -∂ f θ t L(f θ t ; G i ) N = η N ∂ f θ t+1 L(f θ t+1 ; G i ) -∂ f θ t L(f θ t ; G i ) ⊤ N K ∂ f θ t+1 L(f θ t+1 ; G i ) -∂ f θ t L(f θ t ; G i ) N - η N ∂ f θ t+1 L(f θ t+1 ; G i ) N - 1 2 ∂ f θ t L(f θ t ; G i ) N ⊤ K ∂ f θ t+1 L(f θ t+1 ; G i ) N - 1 2 ∂ f θ t L(f θ t ; G i ) N + η 4N ∂ f θ t L(f θ t ; G i ) ⊤ N K ∂ f θ t L(f θ t ; G i ) N . (41
)
Since K is positive definite, it is apparent that
η N ∂ f θ t+1 L(f θ t+1 ; G i ) N - 1 2 ∂ f θ t L(f θ t ; G i ) N ⊤ K ∂ f θ t+1 L(f θ t+1 ; G i ) N - 1 2 ∂ f θ t L(f θ t ; G i ) N
is a non-negative term. Hence, by combining Equations 39, 40, and 41, we obtain
Ξ ≤ - 3η 4N ∂ f θ t L(f θ t ; G i ) ⊤ N K ∂ f θ t L(f θ t ; G i ) N 1 ⃝ + η N ∂ f θ t+1 L(f θ t+1 ; G i ) -∂ f θ t L(f θ t ; G i ) ⊤ N K ∂ f θ t+1 L(f θ t+1 ; G i ) -∂ f θ t L(f θ t ; G i ) N 2 ⃝ . (42
)
Based on the definition of the evaluation functional and the assumption that L is Lipschitz smooth with a constant τ > 0, the term 2 ⃝ in the final part of Equation 42 is bounded above as
2 ⃝ = ∂ f θ t+1 L(f θ t+1 ; G i ) -∂ f θ t L(f θ t ; G i ) ⊤ N K ∂ f θ t+1 L(f θ t+1 ; G i ) -∂ f θ t L(f θ t ; G i ) N = E Gi ∂L(f θ t+1 ) ∂f θ t+1 - ∂L(f θ t ) ∂f θ t ⊤ N K E Gi ∂L(f θ t+1 ) ∂f θ t+1 - ∂L(f θ t ) ∂f θ t N ≤ τ 2 [E Gi (f θ t+1 -f θ t )] ⊤ N K [E Gi (f θ t+1 -f θ t )] N = τ 2 (f θ t+1 -f θ t ) , [K Gi ] ⊤ N H • K • ⟨[K Gi ] N , (f θ t+1 -f θ t )⟩ H = η 2 τ 2 • ∂ f θ t L(f θ t ; G i ) ⊤ N [K Gi ] N , [K Gi ] ⊤ N H N • K • [K Gi ] N , [K Gi ] ⊤ N H N • ∂ f θ t L(f θ t ; G i ) N . (43
)
Under the assumption that the canonical kernel is bounded above by a constant γ > 0, we have
[K Gi ] N , [K Gi ] ⊤ N H ≤ γ [1] N , [1] ⊤ N , and K ≤ γ N [1] N , [1] ⊤ N .
As a result, 1 ⃝ is bounded above by
1 ⃝ ≤ γ N N i=1 ∂ f θ t L(f θ t ; G i ) ⊤ N , [1] N [1] ⊤ N , N i=1 ∂ f θ t L(f θ t ; G i ) N = γ N N i=1 ∂ f θ t L(f θ t ; G i ) 2 .(44)
Meanwhile, the final term in Equation 43 is also bounded above:
η 2 τ 2 • ∂ f θ t L(f θ t ; G i ) ⊤ N [K Gi ] N , [K Gi ] ⊤ N H N • K • [K Gi ] N , [K Gi ] ⊤ N H N • ∂ f θ t L(f θ t ; G i ) N ≤ η 2 τ 2 γ N N i=1 ∂ f θ t L(f θ t ; G i ) ⊤ • K • γ N N i=1 ∂ f θ t L(f θ t ; G i ) N ≤ η 2 τ 2 γ 3 N 1 N N i=1 ∂ f θ t L(f θ t ; G i ) ⊤ N , [1] N [1] ⊤ N , 1 N N i=1 ∂ f θ t L(f θ t ; G i ) N = η 2 τ 2 γ 3 N N i=1 ∂ f θ t L(f θ t ; G i ) 2 .(45)
Hence, by combining Equations 42, 43, 44, and 45, we derive
Ξ ≤ -ηγ 3 4 -η 2 τ 2 γ 2 1 N N i=1 ∂ f θ t L(f θ t ; G i ) 2 , (46
) which means ∂L ∂t ≤ Ξ ≤ -ηγ 3 4 -η 2 τ 2 γ 2 1 N N i=1 ∂ f θ t L(f θ t ; G i ) 2 . (47) Therefore, if η ≤ 1 2τ γ , it follows that ∂L ∂t ≤ - ηγ 2 1 N N i=1 ∂ f θ t L(f θ t ; G i ) 2 = - ηγ 2 1 N N i=1 ∂L(f θ t (G i ), y i ) ∂f θ t (G i ) 2 . (48
) ■ C. Experiment Details
This section outlines the experiment details, covering the experimental setup, supplementary results, and a brief analysis of graph-level and node-level tasks on benchmark datasets.
this section cite: []

Section: C.1. Experimental Setup
Device Setup. We mainly conduct experiments using NVIDIA Geforce RTX 3090 (24G).
this section cite: []

Section: Dataset Splitting.
The train / val / test split configurations for the benchmark datasets are provided in
Table 5. Dataset train validation test QM9 110000 10000 10831 ZINC 220011 24445 5000 ogbg-molhiv 32901 4113 4113 ogbg-molpcba 350343 43793 43793 gen-reg 30000 10000 10000 gen-cls 30000 10000 10000 Table 7: Performance comparison w.r.t. "start-ratio" for different datasets.
this section cite: []

Section: C.2. Graph-level Tasks
We train the GCN using GraNT (B), GraNT (S), and the "without GraNT", all under a common experimental setup for graph-level tasks. For GraNT (B) and GraNT (S), we adopt a curriculum learning strategy (Bengio et al., 2009;Zhang et al., 2024a). Intuitively, at the start of training, the model is undertrained and undergoes significant changes, which calls for more frequent selection by the teacher but with small subsets to help the learner better digest the provided graphs; in contrast, by the end of training, the model stabilizes and is able to digest large subsets. Specifically, the selection interval progressively widens over 50 stages, beginning from the first epoch and gradually extending to the maximum interval. The initial selection ratios are predefined (i.e., the "start-ratio" hyperparameter) for each benchmark dataset. Additionally, for ogbg-molhiv and ogbg-molpcba, we use the ReduceLROnPlateau (Hinton et al., 2012) as the learning rate scheduler, with the 'lr' values in Table 6 representing the initial learning rate. Besides, we activate the learning rate restarting scheme whenever a new selection action is initiated. More experimental results and a brief analysis are provided below.
0 2000 4000 6000 8000 10000 Wallclock Time (s) 2.0 2.5 3.0 3.5 4.0 4.5 5.0 Validation Loss without GraNT with GraNT (B) with GraNT (S) (a) Validation Loss. 0 2000 4000 6000 8000 10000 Wallclock Time (s) 0.0040 0.0045 0.0050 0.0055 0.0060 0.0065 0.0070 Validation MAE without GraNT with GraNT (B) with GraNT (S) (b) Validation MAE. 0 2000 4000 6000 8000 10000 Wallclock Time (s) 3 4 5 6 7 Training Loss without GraNT with GraNT (B) with GraNT (S) (a) Training Loss. 0 2000 4000 6000 8000 10000 Wallclock Time (s) 0.005 0.006 0.007 0.008 Training MAE without GraNT with GraNT (B) with GraNT (S) (b) Training MAE. First, we show the validation and training performance for the regression task on QM9 in Figure 6 and Figure 7, along with the training performance for the regression task on ZINC in Figure 8. For the classification tasks, the training performance on ogbg-molhiv is displayed in Figure 9. Moreover, the validation / training performance on ogbg-molpcba are provided in Figure 10 and Figure 11, respectively.
For QM9, as shown in Figure 6, GraNT (B) and GraNT (S) require less time than "without GraNT" and show a quicker decline in validation loss and MAE curves. In the early stages, the curves follow a nearly linear trend. This occurs because the learning rate is small, allowing the model to learn simple features and adjust its parameters gradually at the beginning. As the optimizer moves through the parameter space with steady steps, the model begins to capture more complex features, resulting in a shift to a nonlinear phase in the validation loss and MAE. In Figure 7, the training loss and MAE curves show significant fluctuations early on, which can be attributed to the frequent selection process. This requires the model to rapidly learn and adapt to new teaching graphs. Over time, the curves take on a step-like shape, suggesting that the selection process is stabilizing and the model has become adequately adaptive. Similarly, the training loss and MAE curves for ZINC in Figure 8 display similar trends.However, it is worth mentioning that GraNT (B) and GraNT (S) cut down the training time by around 3.5 hours for ZINC compared to "without GraNT", all while maintaining similar performance.
For ogbg-molhiv, as shown in Figure 9, the training ROC-AUC curve for GraNT (B) and GraNT (S) fluctuates due to the underlying data imbalance. However, the overall trend and final results align with expectations, even surpassing the "without GraNT" curve. To showcase the generalizability and scalability of GraNT, we further evaluate it on the large-scale multi-classification benchmark dataset ogbg-molpcba, which is 10 times larger than the ogbg-molhiv dataset. Figure 10 shows that the validation AP (Average Precision) curves for GraNT (B) and GraNT (S) consistently outperform the "without GraNT" curve in the mid-to-late stages. Additionally, it is notable that GraNT (B) and GraNT (S) reduce the training time by approximately 13.8 hours compared to the "without GraNT" setup. These results further emphasize GraNT's ability to
0 20000 40000 60000 80000 100000 120000 Wallclock Time (s) 0.2 0.4 0.6 0.8 1.0 Validation Loss without GraNT with GraNT (B) with GraNT (S) (a) Validation Loss. 0 20000 40000 60000 80000 100000 120000 Wallclock Time (s) 0.10 0.15 0.20 0.25 0.30 0.35 Validation AP without GraNT with GraNT (B) with GraNT (S) (b) Validation AP. 0 20000 40000 60000 80000 100000 120000 Wallclock Time (s) 0.1 0.2 0.3 0.4 0.5 0.6 0.7 Training Loss without GraNT with GraNT (B) with GraNT (S) (a) Training Loss. 0 20000 40000 60000 80000 100000 120000 Wallclock Time (s) 0.10 0.15 0.20 0.25 0.30 0.35 Training AP without GraNT with GraNT (B) with GraNT (S) (b) Training AP. epochs, even though it requires more computational time. Over time, GraNT selects more representative teaching graphs (those with larger gradients), gradually accumulating enough information for the implicit mapping. In Figure 15, GraNT (B) takes the least time and shows a faster decline in the training loss curve compared to GraNT (S), while also reaching the final ROC-AUC results more quickly. Additionally, the training ROC-AUC curve exhibits significant oscillations early on, which can be attributed to the frequent selection of a diverse set of teaching graphs and the label imbalance.
In addition, we conduct experiments on node-level tasks with the gen-cls dataset using the AMD Instinct MI210 (64GB) device, further showcasing cross-device generalizability of GraNT. Specifically, GraNT (B) and GraNT (S) save about one-third of the time compared to without GraNT, while achieving even better performance. The validation and training performance results are shown in Figure 16 and Figure 17, respectively. As observed, when the loss curves flatten at their lowest points, the final validation and training ROC-AUC curves for GraNT (B) and GraNT (S) surpass those of the "without GraNT" baseline, highlighting GraNT's effectiveness in improving performance while reducing time consumption.
0 5000 10000 15000 20000 25000 30000 Wallclock Time (s) 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 Validation Loss without GraNT with GraNT (B) with GraNT (S) (a) Validation Loss. 0 5000 10000 15000 20000 25000 30000 Wallclock Time (s) 0.008 0.009 0.010 0.011 0.012 0.013 0.014 Validation MAE without GraNT with GraNT (B) with GraNT (S) (b) Validation MAE. 0 5000 10000 15000 20000 25000 30000 Wallclock Time (s) 3 4 5 6 7 Training Loss without GraNT with GraNT (B) with GraNT (S) (a) Training Loss. 0 5000 10000 15000 20000 25000 30000 Wallclock Time (s) 0.010 0.012 0.014 0.016 0.018 Training MAE without GraNT with GraNT (B) with GraNT (S) (b) Training MAE. 0 500 1000 1500 2000 2500 3000 3500 Wallclock Time (s) 0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 Training Loss without GraNT with GraNT (B) with GraNT (S) (a) Training Loss. 0 500 1000 1500 2000 2500 3000 3500 Wallclock Time (s) 0.0010 0.0015 0.0020 0.0025 0.0030 0.0035 Training MAE without GraNT with GraNT (B) with GraNT (S) (b) Training MAE.
this section cite: ['b2', 'b75', 'b24']

Section: 
Appendix A. Additional Discussions A.1. Notation Overview Notation Description G (n) = (V, E) Graph with n vertices and edge set E V Set of n vertices (nodes) in the graph E Set of edges in the graph [x i ] d d-dimensional feature vector with entries x i x Simplified notation for [x i ] d X n×d
Feature matrix of all nodes (n × d) X (i,:) i-th row of X (feature vector of node i) X (:,j) j-th column of X (feature j across nodes) e i i-th basis vector (1 at i-th position, 0 elsewhere)
A Adjacency matrix of graph G (n) G = (X, A)
Representation of graph with feature matrix and adjacency matrix G
this section cite: []

Section: Collection of all graphs y
Property of the graph (scalar or vector)
Y Space of graph properties (R or R n ) {a i } m
Set of m elements diag(a 1 , . . . , a m ) Diagonal matrix with elements a 1 , . . . , a m diag(a; m) Diagonal matrix with m repeated entries a
N d := {1, • • • , d} Set of natural numbers from 1 to d K(G, G ′ ) Positive definite graph kernel function H Reproducing kernel Hilbert space (RKHS) defined by K f * Target mapping from G to Y y † Property f * (G † ) of graph G †
0 5000 10000 15000 20000 25000 30000 Wallclock Time (s) 3.0 3.5 4.0 4.5 5.0 5.5 6.0 Training Loss without GraNT with GraNT (B) with GraNT (S) (a) Training Loss. 0 5000 10000 15000 20000 25000 30000 Wallclock Time (s) 0.0050 0.0055 0.0060 0.0065 0.0070 0.0075 Training MAE without GraNT with GraNT (B) with GraNT (S) (b) Training MAE. 0 500 1000 1500 2000 Wallclock Time (s) 0.2 0.4 0.6 0.8 1.0 1.2 Training Loss without GraNT with GraNT (B) with GraNT (S) (a) Training Loss. 0 500 1000 1500 2000 Wallclock Time (s) 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 Training ROC-AUC without GraNT with GraNT (B) with GraNT (S) (b) Training ROC-AUC. enhance training time efficiency on large-scale and complex datasets, particularly in the fields of chemistry and biomedical sciences. Additionally, the training loss and AP curves are shown in Figure 11.
Furthermore, on the AMD Instinct MI210 (64GB) device, we also validate the effectiveness of GraNT for graph-level tasks on the QM9 dataset, highlighting its cross-device effectiveness. As illustrated in Figure 12, GraNT (B) and GraNT (S) converge more quickly than "without GraNT," showing a faster decline in validation loss and MAE. Roughly speaking, GraNT (B) and GraNT (S) save more than 40% of the time compared to "without GraNT" while achieving comparable results.The training curves are displayed in Figure 13.
this section cite: []

Section: C.3. Node-level Tasks
We train the GCN with GraNT (B), GraNT (S), and "without GraNT" using a standard experimental setup for node-level tasks, while applying the same curriculum learning strategy used in graph-level tasks. To evaluate GraNT on synthetic data, we utilize graphon (Xu et al., 2021;Xia et al., 2023) to create the gen-reg and gen-cls datasets. Specifically, we set the resolution of the synthetic graphon to 1000×1000, which produces graphs that can be easily aligned by arranging the node degrees in strictly increasing order. Additionally, each graph contains approximately 100 nodes, with a total of 50k graphs across both datasets. Lastly, we use a 2-layer GCN with a specific initialization to assign regression properties or classification labels to each node, while the dimension of node features in all graphs is set to 40.
The training performance results for gen-reg and gen-cls are shown in Figure 14 and Figure 15. In particular, Figure 14 clearly demonstrates that GraNT converges more quickly in terms of wallclock time compared to "without GraNT," which initially drops faster. This faster initial drop may occur because, for gen-reg, training on all the graphs provides sufficient information about the implicit mapping from graphs to their properties, resulting in better validation performance in the early
0 2000 4000 6000 8000 10000 12000 Wallclock Time (s) 0.5 1.0 1.5 2.0 Training Loss without GraNT with GraNT (B) with GraNT (S) (a) Training Loss. 0 2000 4000 6000 8000 10000 12000 Wallclock Time (s) 0.70 0.75 0.80 0.85 0.90 Training ROC-AUC without GraNT with GraNT (B) with GraNT (S) (b) Training ROC-AUC. 0 500 1000 1500 2000 2500 3000 Wallclock Time (s) 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Validation Loss without GraNT with GraNT (B) with GraNT (S) (a) Validation Loss. 0 500 1000 1500 2000 2500 3000 Wallclock Time (s) 0.55 0.60 0.65 0.70 0.75 0.80 0.85 Validation ROC-AUC without GraNT with GraNT (B) with GraNT (S) (b) Validation ROC-AUC. Figure 16: Validation set performance of node-level tasks for gen-cls (classification) on AMD device. 0 500 1000 1500 2000 2500 3000 Wallclock Time (s) 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Training Loss without GraNT with GraNT (B) with GraNT (S) (a) Training Loss. 0 500 1000 1500 2000 2500 3000 Wallclock Time (s) 0.55 0.60 0.65 0.70 0.75 0.80 0.85 Training ROC-AUC without GraNT with GraNT (B) with GraNT (S) (b) Training ROC-AUC.
this section cite: ['b70', 'b69']

Section: References
Ref_id:b0 Title: Explicit defense actions against test-set attacks Year: (2017)
Ref_id:b1 Title: On lower and upper bounds in smooth and strongly convex optimization Year: (2016)
Ref_id:b2 Title: Curriculum learning Year: (2009)
Ref_id:b3 Title: Convex optimization Year: (2004)
Ref_id:b4 Title: Biology-inspired graph neural network encodes reactome and reveals biochemical reactions of disease Year: ()
Ref_id:b5 Title: Graphnorm: A principled approach to accelerating graph neural network training Year: (2021)
Ref_id:b6 Title: Learning large graph property prediction via graph segment training Year: (2023)
Ref_id:b7 Title: Machine learning on graphs: A model and comprehensive taxonomy Year: (2022)
Ref_id:b8 Title: Fastgcn: Fast learning with graph convolutional networks via importance sampling Year: (2018)
Ref_id:b9 Title: Calculus on normed vector spaces Year: (2007)
Ref_id:b10 Title: Convolutional neural networks on graphs with fast localized spectral filtering Year: (2016)
Ref_id:b11 Title: Training neural networks as learning data-adaptive kernels: Provable representation and approximation benefits Year: (2021)
Ref_id:b12 Title: Graph neural tangent kernel: Fusing graph neural networks with graph kernels Year: (2019)
Ref_id:b13 Title: Convolutional networks on graphs for learning molecular fingerprints Year: (2015)
Ref_id:b14 Title: Graph neural networks for social recommendation Year: (2019)
Ref_id:b15 Title: Protein interface prediction using graph convolutional networks Year: (2017)
Ref_id:b16 Title: On the similarity between the laplace and neural tangent kernels Year: (2020)
Ref_id:b17 Title: Calculus of variations Year: (2000)
Ref_id:b18 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b19 Title: Structure-based protein function prediction using graph convolutional networks Year: (2021)
Ref_id:b20 Title: Automatic chemical design using a data-driven continuous representation of molecules Year: (2018)
Ref_id:b21 Title: Few-shot graph learning for molecular property prediction Year: (2021)
Ref_id:b22 Title: Representation learning on graphs: Methods and applications Year: (2017)
Ref_id:b23 Title: Data-efficient image recognition with contrastive predictive coding Year: (2020)
Ref_id:b24 Title: Improving neural networks by preventing co-adaptation of feature detectors Year: (1920)
Ref_id:b25 Title: Open graph benchmark: Datasets for machine learning on graphs. booktitle Year: (2020)
Ref_id:b26 Title: Ogb-lsc: A large-scale challenge for machine learning on graphs Year: (2021)
Ref_id:b27 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b28 Title: On the generalization ability of online strongly convex programming algorithms Year: (2008)
Ref_id:b29 Title: Semi-supervised classification with graph convolutional networks Year: (2008)
Ref_id:b30 Title: Graph neural tangent kernel: Convergence on large graphs Year: (2023)
Ref_id:b31 Title: Asymptotically unbiased estimation in generalized linear models with random effects Year: (1995)
Ref_id:b32 Title: Functional analysis Year: (2002)
Ref_id:b33 Title: Deep learning Year: (2015)
Ref_id:b34 Title: A survey of graph neural network based recommendation in social networks Year: (2023)
Ref_id:b35 Title: Stein variational gradient descent as gradient flow Year: (2017)
Ref_id:b36 Title: Stein variational gradient descent: A general purpose bayesian inference algorithm Year: (2016)
Ref_id:b37 Title: Iterative machine teaching Year: (2017)
Ref_id:b38 Title: Towards black-box iterative machine teaching Year: (2018)
Ref_id:b39 Title: Survey on graph neural network acceleration: An algorithmic perspective Year: (2022)
Ref_id:b40 Title: Survey on graph neural network acceleration: An algorithmic perspective Year: ()
Ref_id:b41 Title: Protecting the copyright of neural radiance fields Year: (2023)
Ref_id:b42 Title: Imaging interiors: An implicit solution to electromagnetic inverse scattering problems Year: (2024)
Ref_id:b43 Title: The nerf signature: Codebook-aided watermarking for neural radiance fields Year: (2025)
Ref_id:b44 Title: Policy poisoning in batch reinforcement learning and control Year: (2019)
Ref_id:b45 Title: Molecular geometry prediction using a deep generative graph neural network Year: (2019)
Ref_id:b46 Title: Stgsn-a spatial-temporal graph neural network framework for time-evolving social networks Year: (2021)
Ref_id:b47 Title: Instant neural graphics primitives with a multiresolution hash encoding Year: (2022)
Ref_id:b48 Title: Efficient gcn training via lazy updates Year: (2022)
Ref_id:b49 Title: Policy teaching via environment poisoning: Training-time adversarial attacks against reinforcement learning Year: (2020)
Ref_id:b50 Title: Quantum chemistry structures and properties of 134 kilo molecules Year: (2014)
Ref_id:b51 Title: Relbench: A benchmark for deep learning on relational databases Year: (2024)
Ref_id:b52 Title: An overview of gradient descent optimization algorithms Year: (2016)
Ref_id:b53 Title: Learning with kernels: support vector machines, regularization, optimization, and beyond Year: (2002)
Ref_id:b54 Title: Near-optimally teaching the crowd to classify Year: (2014)
Ref_id:b55 Title: Implicit neural representations with periodic activation functions Year: (2020)
Ref_id:b56 Title: Biogrid: a general repository for interaction datasets Year: (2006)
Ref_id:b57 Title: Empowering active learning for 3d molecular graphs with geometric graph isomorphism Year: (2024)
Ref_id:b58 Title: Fourier features let networks learn high frequency functions in low dimensional domains Year: (2020)
Ref_id:b59 Title: Training data-efficient image transformers & distillation through attention Year: (2021)
Ref_id:b60 Title: Graph attention networks Year: (2018)
Ref_id:b61 Title: Graph kernels Year: (2010)
Ref_id:b62 Title: Graph mixture of experts: Learning on large-scale graphs with explicit diversity modeling Year: (2023)
Ref_id:b63 Title: A machine teaching framework for scalable recognition Year: (2021)
Ref_id:b64 Title: Gradientbased algorithms for machine teaching Year: (2021)
Ref_id:b65 Title: Coordinate descent algorithms Year: (2015)
Ref_id:b66 Title: Simplifying graph convolutional networks Year: (2019)
Ref_id:b67 Title: Moleculenet: a benchmark for molecular machine learning Year: (2018)
Ref_id:b68 Title: Graph learning: A survey Year: (2021)
Ref_id:b69 Title: Implicit graphon neural representation Year: (2023)
Ref_id:b70 Title: Learning graphons via structured gromov-wasserstein barycenters Year: (2021)
Ref_id:b71 Title: How powerful are graph neural networks? Year: (2018)
Ref_id:b72 Title: Sugar: Efficient subgraph-level training via resource-aware graph partitioning Year: (2023)
Ref_id:b73 Title: Nonparametric teaching for multiple learners Year: ()
Ref_id:b74 Title: Nonparametric iterative machine teaching Year: (2004)
Ref_id:b75 Title: Nonparametric teaching of implicit neural representations Year: (1920)
Ref_id:b76 Title: Gder: Safeguarding efficiency, balancing, and robustness via prototypical graph pruning Year: (2024-08)
Ref_id:b77 Title: A survey on graph neural network acceleration: Algorithms, systems, and customized hardware Year: (2023-02)
Ref_id:b78 Title: Unlearn what you have learned: Adaptive crowd teaching with exponentially decayed memory learners Year: (2018)
Ref_id:b79 Title: Machine teaching: An inverse problem to machine learning and an approach toward optimal education Year: (2015)
Ref_id:b80 Title: An overview of machine teaching Year: (2018)
Ref_id:b81 Title: Graph sampling-based meta-learning for molecular property prediction Year: (2023)
