Title: Covered Forest: Fine-grained generalization analysis of graph neural networks
Abstract: The expressive power of message-passing graph neural networks (MPNNs) is reasonably well understood, primarily through combinatorial techniques from graph isomorphism testing. However, MPNNs' generalization abilities-making meaningful predictions beyond the training setremain less explored. Current generalization analyses often overlook graph structure, limit the focus to specific aggregation functions, and assume the impractical, hard-to-optimize 0-1 loss function. Here, we extend recent advances in graph similarity theory to assess the influence of graph structure, aggregation, and loss functions on MPNNs' generalization abilities. Our empirical study supports our theoretical insights, improving our understanding of MPNNs' generalization properties.

Section: Introduction
Graphs represent interactions across life, natural, and formal sciences, such as atomistic systems (Duval et al., 2023;Zhang et al., 2023) or social networks (Easley & Kleinberg, 2010;Lovász, 2012), highlighting the need for machine learning methods for graph data. Message-passing graph neural networks (MPNNs) (Gilmer et al., 2017;Scarselli et al., 2009) have recently gained attention, achieving success in areas like drug design (Wong et al., 2023), weather forecasting (Lam et al., 2023), and combinatorial optimization (Cappart et al., 2023;Scavuzzo et al., 2024;Qian et al., 2024).
Despite their success, MPNNs' theoretical properties are underexplored (Morris et al., 2024), with most studies focusing on expressivity, i.e., their ability to separate graphs or approximate permutation-invariant graph functions. Ex-pressivity is analyzed via algorithmic alignment with the 1-dimensional Weisfeiler-Leman algorithm (1-WL) (Weisfeiler & Leman, 1968;Weisfeiler, 1976;Morris et al., 2023b), a heuristic for graph isomorphism, or universal approximation theorems (Azizian & Lelarge, 2021;Böker et al., 2023;Chen et al., 2019;Geerts & Reutter, 2022;Maehara & NT, 2019;Rauchwerger et al., 2024). For instance, Morris et al. (2019); Xu et al. (2019) showed that the 1-WL bounds MPNNs' expressivity in distinguishing non-isomorphic graphs, while recent work (Böker et al., 2023;Chen et al., 2022;2023;Rauchwerger et al., 2024) refines expressivity using advances in graph similarity, such as the Tree distance (Böker, 2021).
Understanding when MPNNs generalize to unseen graphs is equally crucial but understudied (Morris et al., 2024). Recent works (Franks et al., 2023;Liao et al., 2021;Morris et al., 2023a;Scarselli et al., 2018) often use Vapnik-Chervonenkis dimension (VC dimension) (Vapnik, 1995) or related formalisms to derive generalization bounds based on simplistic graph parameters, e.g., maximum degree. For example, Morris et al. (2023a, Proposition 1 and 2) linked 1-WL expressivity with VC dimension, assuming the 0-1 loss function and specific aggregation functions. Their reliance on 1-WL implies a simplistic notion of graph similarity, where graphs are either equivalent or entirely dissimilar, leading to vacuous bounds irrelevant to practice. Moreover, in general, most analyses overlook architectural variations like aggregation functions. Present work Here, we extend modern generalization frameworks based on (data-dependent) covering numbers (Xu & Mannor, 2012;Kawaguchi et al., 2022) to address the above shortcomings. Specifically, we investigate how refined notions of graph similarity, or pseudo-metrics on graphs, enable smaller graph coverings and tighter analyses of MPNNs' generalization error. See Figure 1 for an illustration of how pseudo-metrics induce distinct geometries, leading to improved generalization analysis. Our analysis also incorporates various architectural choices and loss functions.
Concretely, (1) we demonstrate that the choice of pseudometric on n-order graphs significantly affects generalization analysis. For instance, the Tree distance (Böker, 2021) yields a strictly tighter generalization error bound than Morris et al. (2023a). We provide a general proof technique for this improvement, covering loss functions such as cross-entropy and mean absolute error, unlike Morris et al. (2023a). (2) We link our pseudo-metrics to the Tree Mover's distance (TMD) (Chuang & Jegelka, 2022), deepening the understanding of TMD, originally defined via a recursive transportation formula. (3) We define the 1-MWL, a heuristic for graph isomorphism that characterizes the expressivity of MPNNs with mean aggregation and satisfies the Lipschitz property required for our generalization analysis. We also derive a corresponding pseudo-metric on graphs equivalent to 1-MWL in distinguishing non-isomorphic graphs. (4) Empirically, our theoretical findings translate into practice, offering nuanced insights into MPNN generalization.
In summary, our results highlight how refined graph similarity notions improve the understanding of MPNNs' generalization, considering graph structure, architectural choices, and loss functions.
See Appendix A for a detailed discussion of related work.
this section cite: ['b25', 'b83', 'b28', 'b41', 'b60', 'b79', 'b15', 'b62', 'b57', 'b53', 'b78', 'b77', 'b52', 'b4', 'b13', 'b18', 'b40', 'b58', 'b48', 'b81', 'b13', 'b16', 'b58', 'b10', 'b53', 'b35', 'b51', 'b61', 'b72', 'b80', 'b10', 'b51', 'b51', 'b19']

Section: Background
In the following, we provide the necessary background on (pseudo-)metric spaces, covering numbers, and MPNNs. We use standard notation for graphs and norms; see Appendix B.
this section cite: []

Section: Metric spaces and coverings
In the remainder of the paper, "distances" between graphs play an essential role, which we make precise by defining a pseudo-metric (on the set of graphs). Let X be a set equipped with a pseudometric d : X × X → R + , i.e., d is a function satisfying d(x, x) = 0 and d(x, y) = d(y, x) for x, y ∈ X , and d(x, y) ≤ d(x, z) + d(z, y), for x, y, z ∈ X . The latter property is called the triangle inequality. The pair (X , d) is called a pseudo-metric space. For (X , d) to be a metric space, d additionally needs to satisfy d(x, y) = 0 ⇒ x = y, for x, y ∈ X .foot_4 See Appendix B for the definition of (Lipschitz/uniform) continuity of a function between pseudometric spaces. Let (X , d) be a pseudo-metric space. Given an ε > 0, an ε-cover of X is a subset C ⊆ X such that for all elements x ∈ X there is an element y ∈ C such that d(x, y) ≤ ε. Given ε > 0 and a pseudo-metric d on the set X , we define the covering number of X , N (X , d, ε) := min{m | there is an ε-cover of X of cardinality m}, i.e., the smallest number m such that there exists a ε-cover of cardinality m of the set X with regard to the pseudometric d.
The covering number provides a direct way of constructing a partition of X . Let K := N (X , d, ε) so that, by definition of the covering number, there is a subset {x 1 , . . . , x K } ⊂ X representing an ε-cover of X . We define a partition {C 1 , . . . , C K } where C i := {x ∈ X | d(x, x i ) = min j∈[K] d(x, x j )}, for i ∈ [K]. To break ties, we take the smallest i in the above. Note that X = i∈[K] C i . The diameter of a set is the maximal distance between any two elements in the set. Implied by the definition of an ε-cover and the triangle inequality, each C i has a diameter of at most 2ε.
Supervised learning on graphs We define supervised learning on graphs as follows. Let G be the set of all graphs and Y a set. Unless specified otherwise, G includes graphs of varying orders with vertex features in arbitrary domains. Later, we restrict G by bounding graph order or vertex feature domains. Similarly, Y varies depending on the task, e.g., regression or binary classification. We consider classes H ⊆ X G of graph embeddings, i.e., mappings from G to X for some set X . A graph learning algorithm learns such embeddings from data samples. More formally, let Z := G × Y with a probability distribution µ. A data sample S is a collection of elements from Z, drawn i.i.d. according to µ. We denote (G, y) ∼ µ for an element drawn from Z under µ, and S ∼ µ k for a sample of size k ∈ N. A graph learning algorithm for H maps each sample S to a graph embedding h S ∈ H. The "goodness" of h S is assessed using a loss function ℓ : X × Y → R + , assumed bounded by some M > 0. That is, |ℓ(h(G), y)| ≤ M for all h ∈ H, G ∈ G, and y ∈ Y. We define the expected and empirical error as: ℓ exp (h S ) := E (G,y)∼µ ℓ h S (G), y and ℓ emp (h S ) := 1 |S| (G,y)∈S ℓ h S (G), y , where S ∼ µ k for k ∈ N. The generalization error, |ℓ exp (h S )-ℓ emp (h S )|, is what we aim to bound in this work.
Message-passing graph neural networks One particular, well-known class of graph embeddings is MPNNs. Following Gilmer et al. (2017), let G be an attributed graph with initial vertex-feature h
(0) v ∈ R d0 , d 0 ∈ N, for v ∈ V (G).
An MPNN architecture consists of a stack of L neural network layers for some L > 0. In each layer, t ∈ N, we compute a vertex feature h (t)
v := UPD (t) h (t-1) v , AGG (t) { {h (t-1) u | u ∈ N (v)} } ∈ R dt , d t ∈ N, for v ∈ V (G),
where UPD (t) and AGG (t) are parameterized functions, e.g., neural networks. In the case of graph-level tasks, e.g., graph classification, one uses a readout,
h G := READOUT { {h (L) v | v ∈ V (G)} } ∈ R d ,
to compute a single vectorial representation based on learned vertex features after iteration L. Again, READOUT may be a parameterized function.
this section cite: ['b41']

Section: G1 G2 G3 G4
(a) A set of padded binary trees.
γ(ε) γ(ε) ε ε ε ε MPNN(G1 ) MPN N(G 2 ) MPN N (G3 ) MP NN (G4 ) γ(ε) γ(ε) G1 γ(ε) G2 G3 G4
(b) Geometry under the trivial 1-WL discrete pseudometric. All graphs are equally far apart, not taking into account their similarity, mapping each graph to a unique ε-ball.
γ(ε) ε M PNN (G 1 ) M PN N(G 2 ) MPN N (G 3 ) M PN N(G 4 ) G1 G2 G3 G4
(c) Geometry under the Tree distance δ ∥•∥ . The similarity of graphs is preserved under MPNNs, leading to a smaller covering and a tighter generalization analysis.
Figure 1. Illustrating how the choice of pseudo-metrics influences the geometry and the size of coverings, leading to a tighter generalization analysis.
this section cite: []

Section: Special cases of MPNN layers
In the following, we discuss special cases of MPNNs, which we will subsequently use in our analyses of MPNNs' generalization abilities. First, for an unlabeled n-order graph G, we assume all initial vertex-features are identical, i.e., h
v = h(0)
w , for v, w ∈ V (G). In this case, we define an MPNN layer using order-normalized sum aggregation and order-normalized readout, where
h (t) v := φ t 1 /|V (G)| u∈N (v) h (t-1) u ,(1)
h G := ψ 1 /|V (G)| u∈V (G) h (L) u ,(2)
for v ∈ V (G), where φ t : R dt-1 → R dt is a L φt -Lipschitz continuous function, with respect to the 2-norm-induced metric, for d t ∈ N and t ∈ [L], and ψ : R d L → R d is a L ψ -Lipschitz continuous function. 2 Secondly, for an attributed n-order graph (G, a), i.e., we set h (0) v = a(v), for v ∈ V (G), we define an MPNN layer using sum aggregation and order-normalized readout, where
h (t) v := φ t h (t-1) v W (1) t + u∈N (v) h (t-1) u W (2) t ,(3)
h G := ψ 1 /|V (G)| u∈V (G) h (L) u ,(4)
where φ t and ψ are defined as in Equation ( 1) and W
t , W
(2) t are d t-1 × d t matrices over R. Under the additional assumption of positive homogeneity of φ t , i.e., φ t (λx) = λφ t (x), for λ > 0, we define an MPNN layer using mean aggregation and order-normalized readout, where
h (t) v := φ t h (t-1) v W (1) t + 1 /|N(v)| u∈N (v) h (t-1) u W (2) t ,(5)
h G := ψ 1 /|V (G)| u∈V (G) h (L) u .(6)
2 Implied by Morris et al. (2019), choosing the function φt appropriately, leads to a 1-WL-equivalent MPNN-layer.
this section cite: ['b48']

Section: MPNN classes
Based on the above three types of MPNN layers, we consider the following classes of MPNNs. Let G be a class of graphs, we denote the class of all L-layer MPNNs following Equation (1) where ψ is represented by a feed-forward neural network (FNN) (See Appendix B.1 for a formal definition of FNNs) with Lipschitz constant L FNN and bounded by M ′ ∈ R by MPNN ord L,M ′ ,L FNN (G), i.e.,
MPNN ord L,M ′ ,L FNN (G) := h : G → R h(G) := FNN θ • h G where θ ∈ Θ ,
where Θ is the parameter set of FNNs. Analogously, for Equation (3) and Equation ( 5), we define the classes MPNN sum L,M ′ ,L FNN (G) and MPNN mean L,M ′ ,L FNN (G), respectively. Finally, we denote the class of all MPNN architectures consisting of L layers with a readout layer represented by an FNN at the (L + 1)th-layer by MPNN L (G).
In Appendix C, we provide the formal definition of the 1-Weisfeiler-Leman algorithm (1-WL), characterizing the distinguishing power of sum aggregation MPNNs (Morris et al., 2019). We also introduce 1-MWL, a variant of 1-WL, characterizing the distinguishing power of mean aggregation MPNNs.
this section cite: ['b48']

Section: Pseudo-metrics on the set of graphs
We define and study (pseudo-)metrics on the set of graphs, which we later use for a fine-grained generalization analysis of MPNNs; see Appendix D for an extended discussion. These pseudo-metrics are designed to align with MPNNs, reflecting their computational properties. Since the expressiveness of MPNNs is bounded by the 1-WL, we aim to define pseudo-metrics such that two graphs have a distance of 0 if, and only if, they are 1-WL indistinguishable.
Building on Böker (2021), we define the Tree distance, as follows. A matrix S ∈ [0,1] n×n is a doubly-stochastic matrix if ∥S i,• ∥ 1 = 1, for i ∈ [n], and ∥S •,j ∥ 1 = 1, for j ∈ [n]. Let D n denote the set of n × n doubly-stochastic matrices. Given a matrix norm ∥•∥, for two graphs G and H with adjacency matrices A(G) and A(H), respectively, we define the Tree distance
δ T ∥•∥ (G, H) := min S∈Dn ∥A(G)S -SA(H)∥.(7)
Similarly, we extend the above definition to labeled graphs in G B n,d , 3 resulting in the labeled Tree distance,
δ T ∥•∥ (G, H) := min S∈Dn ∥A(G)S -SA(H)∥ + Tr(S ⊺ L(G, H)), (8
)
where L(G, H) = [dist(ℓ G (i), ℓ H (j))] i∈V (G),j∈V (H) for some distance function dist between node features (see Appendix D). The following result shows that Equations ( 7) and ( 8) are valid pseudo-metrics for many norms. Proposition 1. For every entry-wise matrix norm or the cut
norm ∥•∥, δ T ∥•∥ is a pseudo-metric on G B n,d . Additionally, for two vertex-labeled graphs G and H in G B n,d , δ T ∥•∥ (G, H) = 0 if,
and only, if G and H are 1-WL indistinguishable.
Although the Tree distance and its extension to labeled graphs match the 1-WL algorithm in expressivity, they do not consider the number of iterations needed to distinguish two graphs. This limits our ability to provide tighter generalization error bounds for MPNNs with a fixed number of layers. To address this, we define the Forest distance on G R n,d , which matches the expressivity of 1-WL up to L iterations for fixed L ∈ N.
Forest distance To define the Forest distance, formally, let (G, a G ) and (H, a H ) be attributed graphs of order n, where 4 Given a graph G, a node u ∈ V (G), and a depth L ∈ N, we define the unrolling tree of u at depth L as a rooted tree at u with depth L, where each node has as children its neighbors in the graph G. This tree is also called the computation tree (see Appendix C for a formal definition). Now, for a fixed L ∈ N, consider the following multiset of unrolling trees:
a G : V (G) → R d \ {0 R d } and a H : V (H) → R d \ {0 R d }, for d ∈ N.
T L G := { {τ (unr(G, u, L)) | u ∈ V (G)} }.
To account for unrolling trees of different orders, we perform a padding process on all trees in T L G . That is, for each vertex of each unrolling tree in the multiset T L G , we add children with the label 0 until each vertex has exactly n -1 children. Hence, after padding, all trees in T L G will have the exact same structure, though the vertices will have different labels, and we denote their disjoint union, i.e., a 3 Note that this pseudometric can also be defined on the space G R n,d , which includes graphs with continuous vertex features. However, we restrict our proof to Boolean features to establish the equivalence with the 1-WL algorithm. 4 Without loss of generality, we use the zero vector 0 for padding purposes.
forest, by F G,L . We perform the same procedure for the graph (H, ℓ H ), resulting in the forest F H,L . Note that the forests F G,L and F H,L have the same structure but are possibly non-isomorphic due to their vertex labels not matching. However, we can always find an edge-preserving bijection between V (F G,L ) and V (F H,L ). Finally, given a vertex u ∈ V (F G,L ), we denote by l(u) ∈ [L] the level of vertex u in the forest F G,L . Now given a weight function ω : N → R + , we define the Forest distance of depth L and weights ω, between G and H as
FD L,ω (G, H) := min φ u∈V (F G,L ) ω(l(u)) • ∥a G (u) -a H (φ(u))∥ 2 , (9
)
where the minimum is taken over all edge-preserving bijections φ between V (F G,L ) and V (F H,L ). 5 When ω(l) = 1, for l ∈ N, we denote the Forest distance by FD L . See Figure 2 for an illustration of the distance. The following result shows that FD
L is a valid pseudo-metric. Lemma 2. For every ω : N → R + , L ∈ N, the Forest distance FD L,ω is a well-defined pseudo-metric on G R n,d . In addition, for two graphs, G, H, FD L,ω (G, H) = 0 if and only if G, H are 1-WL indistinguishable after L iterations.
Now the following result shows that the Forest distance is a simplified version of the TMD defined by Chuang & Jegelka (2022), see Appendix F, providing a streamlined, easy-tounderstand definition of the TMD distance while preserving all the essential properties required for our analysis.
Lemma 3. The Forest distance is equivalent to the Tree Mover's distance. That is, for all n, d, L ∈ N, ω : N → R + , and for all graphs G, H ∈ G R n,d ,
TMD (ω) L (G, H) = FD L, ω (G, H), where ω(n) = n i=1 ω(i), for i ∈ N.
While mathematically equivalent, the Forest distance and TMD have distinct advantages. TMD allows more efficient computation, making it practical for applications, whereas the Forest distance intuitively captures structural differences between graphs.
this section cite: ['b10', 'b19']

Section: Mean Forest distance
The Forest distance motivates the definition of the mean-Forest distance, based on mean unrolling (computation) trees (see Appendix G). This is the first graph pseudo-metric that precisely captures the distinguishing power of mean aggregation MPNNs (see Lemma 29). These MPNNs also satisfy a Lipschitz property regarding the mean-Forest distance (see
G 1 G 2 φ Figure 2.
An illustration of the computation of the Forest distance for depth L = 2 for two labeled graphs G1 and G2. Grey vertices indicate the padded vertices in the unrollings, and φ represents an edge-preserving bijection between the two forests.
Lemma 48). We remark that Chuang & Jegelka (2022) (Appendix B.1) introduced the unnormalized tree mover's distance (TMD * L ω ) and proved its Lipschitz property for mean aggregation MPNNs. However, TMD * L ω is more powerful than the 1-MWL, as it relies on optimal transport between unrolling trees, capturing 1-WL's distinguishing power instead of 1-MWL's.
this section cite: []

Section: Robustness framework
In the following, we introduce the generalization framework for studying the generalization abilities of MPNNs. We refer to Section 1.1 for the formal setup and notation. Slightly modifying the notation from Xu & Mannor (2012), we say that a (graph) learning algorithm for the hypothesis class H, e.g., a class of graph embeddings, on G := X × Y, is (K, ε)robust, for K and ε, mappings from the set of all possible samples to N and R + , respectively, if for all samples S, Z can be partitioned into K(S) > 0 sets,
{C i } K(S) i=1 , such that the following holds. If (G, y) ∈ C i , for some i ∈ [K], then for all (G ′ , y ′ ) ∈ C i , ℓ h S (G), y -ℓ h S (G ′ ), y ′ < ε(S),
where h S is the graph embedding the learning algorithm returns regarding the data sample S. Intuitively, the above definition requires that the difference of the losses of two data points in the same part of the partition is small. Xu & Mannor (2012) showed that a (K, ϵ)-robust learning algorithm implies a bound on the generalization error. Theorem 4 (Theorem 3 in Xu & Mannor (2012)). For any (K, ϵ)-robust (graph) learning algorithm for H on Z, we have that for all δ ∈ (0,1), with probability at least 1 -δ,
|ℓ exp (h S )-ℓ emp (h S )| ≤ ϵ(S)+M 2K(S) log(2) + 2 log( 1 /δ) |S| ,
where h S , as before, denotes a graph embedding from H returned by the learning algorithm given the data sample S of Z. We recall that M is the bound on the loss function ℓ.
See Appendix I for the refined results of Kawaguchi et al. (2022) and our extension allowing for different radii within each partition set.
Connecting robustness, continuity, and expressivity A natural approach to ensure the robustness of the learning algorithm is to impose continuity assumptions on both the loss functions and the graph embeddings. To formalize this, let us consider two pseudo-metric spaces (X , d X ) and (Y, d Y ), and define the pseudo-metric
d ∞ : (X ×Y)×(X × Y) → R + as d ∞ ((x, y), (x ′ , y ′ )) := max{d X (x, x ′ ), d Y (y, y ′ )}.
Let ℓ : X × Y → R + be a c ℓ -Lipschitz continuous loss function concerning the metric d ∞ and (G, d G ) be a pseudometric space over a set of graphs. If the covering numbers for the spaces (G, d G ) and (Y, d Y ) can be bounded above, then uniform continuity implies robustness, as shown next.
Proposition 5. Let (G, d G ), (X , d X ), and (Y, d Y ) be pseudo-metric spaces, and let H denote the class of uniformly continuous graph embeddings from
(G, d G ) to (X , d X ). If ℓ : X × Y → R + is a c ℓ -Lipschitz continuous loss function, regarding d ∞ , then for any ε > 0, 6 a graph learning algorithm for H on G × Y is N G, d G , γ(ε, •)/2 • N (Y, d Y , ε/2), c ℓ ε -robust.
Here, γ = γ(ε, h S ) depends on both ε and S is the positive function used in the definition of the uniform continuity of h S ∈ H.
this section cite: ['b80', 'b80', 'b80']

Section: Lipschitzness and equicontinuity of the hypothesis class
By Proposition 5 and Theorem 4, we observe that the generalization bound depends on the function γ, which, in turn, depends on the specific choice of the MPNN h S . Ideally, we aim to eliminate this dependency on S by deriving a uniform bound for all uniformly continuous MPNNs. This can be achieved by showing the existence of a "uniform" choice of γ(ε) that satisfies the uniform continuity definition for all possible MPNNs within a hypothesis class. This property is also known as equicontinuity of the hypothesis class.
For instance, this holds for all the hypothesis classes of all c-Lipschitz MPNNs, sharing the same Lipschitz constant c, in which case we can set γ(ε) = ε /c.
Corollary 6. Let (G, d G ), (X , d X )
, and (Y, d Y ) be pseudometric spaces, and let H denote the class of c H -Lipschitz continuous graph embeddings from
(G, d G ) to (X , d X ).
Assume that the loss function is c ℓ -Lipschitz, regarding d ∞ . Then, graph learning algorithms for H are
N G, d G , ε/(2c H ) • N (Y, d Y , ε/2), c ℓ ε -robust.
As will become clear later, we will use Corollary 6 to derive our generalization bounds in the regression setting, assuming a Lipschitz-continuous loss function. However, for the classification setting, we will specifically use the cross-entropy loss, which is Lipschitz-continuous, concerning only one argument. In this case, we will need a slightly modified version of Corollary 6 to establish our generalization bounds, resulting in the following proposition, which follows from Xu & Mannor (2012, Theorem 14)
Proposition 7. Let (G, d G ), (X , d X ), and (Y, d Y ) be pseudo-metric spaces, let H be a class of graph embed- dings from (G, d G ) to (X , d X )
, and let ℓ : X × Y → R + be a loss function. For a graph learning algorithm for H on G × Y, assume that there exists a positive γ > 0 such that for all samples S and
G 1 , G 2 ∈ G, y 1 , y 2 ∈ Y, max{d G (G 1 , G 2 ), d Y (y 1 , y 2 )} ≤ γ =⇒ |ℓ(h S (G 1 ), y 1 ) -ℓ(h S (G 2 ), y 2 )| < ϵ(S), then, H is (N (Y, d Y , γ/2) • N (G, d G , γ/2), ϵ)-robust.
See Appendix J.1 for more details on the relationship between robustness and continuity.
this section cite: []

Section: Fine-grained generalization analysis of MPNNs
In this section, we derive the main generalization results of this work using the robustness framework described in Section 3. First, in Appendix L, we recover the VC-dimension bounds from Morris et al. (2023a) within the robustness framework, showing that these bounds can be obtained using a trivial discrete pseudo-metric between graphs, defined as 1 when the graphs are 1-WL, distinguishable and 0 otherwise, leading to vacuous generalization bounds. This motivates us to analyze the impact of alternative pseudo-metrics that yield more fine-grained generalization bounds within the robustness framework compared to the trivial 1-WL-based pseudo-metric. Our analysis considers different aggregation functions, vertex labeling schemes, and loss functions.
We begin by exploring the generalization abilities of MPNN architectures using sum aggregation. Specifically, we consider unlabeled graphs and analyze MPNN layers defined in Equation (1), i.e., order-normalized sum-aggregation MPNNs, using the pseudo-metrics defined in Equation ( 7), i.e., the Tree distance. This leads to Proposition 9, Theorem 10, and Proposition 12 (for a class of binary trees). Next, we consider labeled graphs and analyze MPNN layers defined in Equation (3), i.e., sum-aggregation MPNNs, using the pseudo-metrics defined in Equation ( 9), i.e., the Forest distance. This yields Proposition 15 and Proposition 16. These results can be extended to mean-aggregation MPNNs using the mean-Forest distance, as described in Appendix M.
We first consider unlabeled n-order graphs, i.e., every vertex has the same label. We analyze the binary classification setting. Following Section 1.1 and Equation ( 1), here we investigate the class MPNN ord L,M ′ ,L FNN (G n ). We further consider the binary cross-entropy loss ℓ : R×{0,1} → R, where ℓ(x, y) := y log(σ(x)) + (1 -y) log(1 -σ(x)), where σ is the sigmoid function. It is known that ℓ is Lipschitz on the first argument (since it has a bounded first-order derivative) for some Lipschitz constant L ℓ .
In Proposition 1 in Section 1.1, we observed that the Tree distance of Equation ( 7) induces various pseudo-metrics that are equivalent to the 1-WL in terms of expressivity. Utilizing the analysis in Böker et al. (2023), we show that the Tree distance using the cut norm satisfies the uniform continuity property using the same γ(ε) for all MPNNs in
MPNN ord L,M ′ ,L FNN (G n ), i.e.
, it is equicontinuous. Formally, we consider the following variant of the Tree distance on G n ,
δ T □ (G, H) := min S∈Dn ∥A(G)S -SA(H)∥ □ ,
where ∥•∥ □ denotes the cut norm, see Equation (11) in Appendix B for details.
The following result shows the equicontinuity property of MPNNs.
Theorem 8. For all ε > 0, L ∈ N there exists γ(ε) > 0 such that for n ∈ N,
M ′ ∈ R, G, H ∈ G n and all MPNN architectures in MPNN ord L,M ′ ,L FNN (G n ), we have δ T □ (G, H) < n 2 • γ(ε) =⇒ ∥h G -h H ∥ 2 ≤ ε.
The proof is a direct implication of Theorems 29 and 31 in Böker et al. (2023) using the uniform continuity of MPNNs concerning the Prokhorov metric (Theorem 29) and then the ε-δ equivalence between the Prokhorov metric and the Tree distance δ T □ (Theorem 31). It is important to note that, in the above result, we proved that all functions in MPNN ord L,M ′ ,L FNN (G n ), are uniformly continuous sharing the same γ(ε), or equivalently the hypothesis class MPNN ord L,M ′ ,L FNN (G n ) satisfies the equicontinuity property. This property is our primary motivation for choosing the tree distance as the pseudo-metric.
Based on the above result, we define the following nondecreasing function s γ : R + ∪ {+∞} → R + ∪ {+∞} where γ(+∞) := +∞, for all G, H ∈ Note that s γ ← is also non-decreasing. We further make the convention that inf{∅} = +∞. We now establish the following family of generalization bounds for the class MPNN ord L,M ′ ,L FNN (G n ) based on the results from Proposition 7.
G n , for all h ∈ MPNN ord L,M ′ ,L FNN (G n ), s γ(ϵ) := sup{δ > 0 | δ T □ (G, H) ≤ n 2 δ ⇒ ∥h G -h H ∥ ≤
Proposition 9. For n, L ∈ N, M ′ ∈ R, and ε > 1 2 d * , where d * is the minimum non-zero δ T □ distance between two graphs in G n , any graph learning algorithm for the class
MPNN ord L,M ′ ,L FNN (G n ) is 2N (G n , δ T □ , ε), L ℓ • L FNN • s γ ← 2ε n 2 -robust.
Hence, for any sample S and δ ∈ (0,1), with probability at least 1 -δ,
|ℓ exp (h S ) -ℓ emp (h S )| ≤ L ℓ • L FNN • s γ ← 2ε n 2 + M 4N (G n , δ T □ , ε) log(2) + 2 log( 1 /δ) |S| ,
where M is an upper bound for the loss function ℓ and L ℓ is the Lipschitz constant of ℓ(•, y), and y ∈ {0,1}.
As can be observed by Proposition 9, the tree distance-a more fine-grained metric compared to the trivial 1-WL metric (which is 1 for 1-WL-distinguishable graphs and 0 otherwise)-enables deriving a wider range of generalization bounds by allowing any real number as the radius for the covering number. Choosing the radius ε is critical for tight generalization bounds: larger ε reduces the covering number but increases the first term of the bound, while smaller ε has the opposite effect. Balancing this trade-off is important, though computing the covering number as a function of ε is an extremely hard problem. For a given n, the covering number can instead be bounded in terms of ε and m n = | Gn /∼ WL |, yielding covering number bounds dependent on ε. For n-order graphs G n and ε = 4k, a cover of size mn /(k + 1) can be constructed for all k ∈ N, leading to a family of bounds parametrized by k.
Theorem 10. For n, L ∈ N, and M ′ ∈ R, for any graph learning algorithm for the class MPNN ord L,M ′ ,L FNN (G n ) and for any sample S and δ ∈ (0,1), with probability at least 1 -δ, we have
|ℓ exp (h S ) -ℓ emp (h S )| ≤ L ℓ • L FNN • s γ ← 8k n 2 + M 4 mn k+1 log(2) + 2 log 1 δ |S| .(10)
for k ∈ N, where M is an upper bound on the loss function ℓ, and
m n = | Gn /∼ WL |
In Appendix I in Proposition 36, we combine the extended robustness definition along with Theorem 35 to extend the generalization bounds from Theorem 10 to the class of unlabeled graphs with at most n vertices, denoted as G ≤n . The proof of Proposition 36 is omitted, as it directly follows from the tree construction described in the proof of Theorem 10.
this section cite: ['b51', 'b13', 'b13']

Section: Tighter bounds on the covering number
The proof of Theorem 10 establishes an upper bound on the covering number, decreasing linearly with the radius. We can use this bound to determine the optimal radius minimizing the generalization bound in Equation ( 10). However, it remains unclear if this upper bound on the covering number is tight or if the linear decay can be improved. Moreover, the bound mn /(k + 1) relies on the number of equivalence classes m n , which is hard to compute.
To address this, we analyze specific graph classes that yield tighter bounds and simplify the computation of 1-WL-distinguishable graphs. First, consider the class of unordered, unlabeled, full binary trees on n vertices, denoted T
(2) n , where n = 2j + 1 and j ∈ N. This class includes rooted trees where each non-root vertex has 0 or 2 children. The root need not be explicitly specified here since it is the only vertex with degree 2; all others have degree 1 or 3. The graphs in T
(2) n are also called Otter trees, named after Richard Otter's work on their enumeration (Otter, 1948). The total number of Otter trees on n vertices, for n = 2j +1, is given by the Wedderburn-Etherington number w j , with j ∈ N. Although no closed-form formula exists for w j , recursive methods enable efficient computation. Additionally, the following asymptotic result holds. Lemma 11 ( (Finch, 2005, p. 295)). The asymptotic growth of w n is given by w n ∼ A•n -3/2 •b n , for a positive constant A whose precise value is not relevant for our purposes and b ≈ 2.4832.
Based on this lemma, we show that for sufficiently large n, we can bound the covering number of the set of Otter trees with a function that decreases exponentially regarding the radius of the cover leading. This results in the following (tighter) generalization bounds for MPNNs on the space of Otter trees. Proposition 12. For L ∈ N, M ′ ∈ R and sufficiently large n ∈ N, for any graph learning algorithm on
MPNN ord L,M ′ ,L FNN (T(2)
2n+1 ) and for any sample S, with δ ∈ (0,1), with probability at least 1 -δ, we have
|ℓ exp (h S ) -ℓ emp (h S )| ≤ L ℓ • L FNN • s γ ← 16k (2n + 1) 2 + M 4 wn /b 2k log(2) + 2 log( 1 /δ) |S| ,
where k ∈ N, M is an upper bound on the loss function ℓ, b ≈ 2.4832, and
w n = |T (2) 2n+1 |.
Note that using the above bound with a radius mildly dependent on n, say log b (n), we can derive an upper bound on the covering number that decreases quadratically with n.
Additionally, in Appendix N, we derive a graph class (denoted as F n ) that leads to non-constant improvement for a constant ε and lift the above results to the regression setting.
Vertex-labeled graphs Finally, in this section, we aim to extend the previous results to the space of discretelylabeled n-order graphs, where vertex features are drawn from a finite collection of d elements, i.e., G B n,d . For ease of notation, we consider one-hot encoding of the labels. Thus, all vertices are initially labeled with a d-dimensional vector, having a 1 in one position and 0 elsewhere. Furthermore, we consider the set of graphs with bounded-degree q, for q ∈ N. We denote this set as G n,d,q , and the number of equivalence classes induced by the 1-WL after L iterations on this set by m n,d,q,L , or simply m n,d,q when L = n -1.
To establish our generalization bounds based on the robustness framework, we use the Forest distance FD L as defined in Section 2. We begin with the following simple lemma, bounding the Forest distance between two graphs differing by either one edge or one label. Moreover, the following lemma explains the assumption of bounded degree in our graph class. Without this assumption, the bound below would grow with n, making it impossible to establish uniform generalization bounds.
Lemma 13. For d, q ∈ N, there exists a constant b(d, q, L) such that for n ∈ N and G, H ∈ G n,d,q , if G can be derived by either deleting an edge or changing a single vertex feature of H, then FD L (G, H) ≤ b(d, q, L), where b(d, q, L) = 2q L 1-q L 1-q .
In the following, we analyze the generalization power of MPNN sum L,M ′ ,L FNN (G n,d,q ). Similar to the case of unlabeled graphs, we consider the crossentropy function ℓ : R × {0,1} → R, defined by ℓ(x, y) = y log(σ(x)) + (1 -y) log(1 -σ(x)), where σ(x) is the sigmoid function. As with unlabeled graphs, our generalization analysis requires MPNNs to be Lipschitz continuous with respect to the chosen pseudo-metric. The following result demonstrates the Lipschitz continuity prop-erty of MPNNs in MPNN sum L,M ′ ,L FNN (G n,d,q ) concerning the Forest distance. While our analysis is restricted to G n,d,q , we state and prove the following result in the more general setting of graphs in G R n,d , i.e., with real-valued vertex features. Lemma 14. For L, n, d ∈ N, M ′ ∈ R and for all MPNNs
in MPNN sum L,M ′ ,L FNN (G R n,d ), we have ∥h G -h H ∥ 2 ≤ 1 n C(L)L ψ L i=1 L φi FD L (G, H),
for G, H ∈ G R n,d , where C(L) is a constant that depends on L and the Lipschitz constants of the MPNN layers.
We are now ready to present the generalization bound extension for labeled graphs.
Proposition 15. For ε > 0 and n, L, d, q ∈ N, M ′ ∈ R, any graph learning algorithm for
MPNN sum L,M ′ ,L FNN (G n,d,q ) is (2N (G n,d,q , FD L , ε), L ℓ • L FNN • C FD L • 2ε)-robust.
Hence, we have the following generalization bounds. For any sample S and δ ∈ (0,1), with probability at least 1 -δ,
|ℓ exp (h S ) -ℓ emp (h S )| ≤ Cε + M N (G n,d,q , FD L , ε)4 log(2) + 2 log( 1 /δ) |S| , for ε > 0, where C = 2 /nL ℓ L FNN C FD L , C FD L = C(L)L ψ L i=1
L φi is the Lipschitz constant in Lemma 14 and M is an upper bound of the loss function ℓ.
Using a similar construction as Theorem 10, below, we upper bound the covering number in the generalization bound by a function of the corresponding radius and m n,d,q,L , leading to the following generalization bound. Proposition 16. For n, q, d, L ∈ N, and M ′ ∈ R, for any graph learning algorithm for the class MPNN sum L,M ′ ,L FNN (G n,d,q ) and for any sample S and δ ∈ (0,1), with probability at least 1 -δ, we have
|ℓ exp (h S ) -ℓ emp (h S )| ≤ 2 Cb(d, q, L)k + M m n,d,q,L k+1 4 log(2) + 2 log( 1 /δ) |S| , for k ∈ N, where C = 2 /nL ℓ L FNN C FD L , C FD L = C(L)L ψ L i=1
L φi is the Lipschitz constant in Lemma 14 and M is an upper bound of the loss function ℓ.
In Appendix M, we lift the results to MPNNs using mean aggregation. Additionally, in Section 5, we discuss our analysis's limitations and future work.
this section cite: ['b55']

Section: Limitations, possible road maps, and future work
While our results are the first to successfully incorporate non-trivial graph similarities
, architectural choices, and loss functions into the generalization analysis, many open questions remain regarding MPNN generalization properties. First, our techniques are tailored towards discretelylabeled graphs, e.g., not accounting for real-valued labels. While Rauchwerger et al. (2024) did a first step in this direction, their generalization bounds are implicit or existential, i.e., they do not derive concrete upper bounds on the size of the covering, only showing the existence of a finite covering via the compactness of the investigated pseudo-metric spaces. Secondly, while our analysis can be extended to other aggregation functions, e.g., weighted mean, it does not account for other commonly used architectural choices, such as normalization layers or skip connections. Thirdly, although the experimental results in Section 6 indicate that our generalization analysis holds in practice to some extent, it does not explain why gradient descent-based algorithms converge to generalizing solutions. Hence, future work should extend our results to graphs with real-valued features and study whether gradient descent-based algorithms can converge to parameter assignments inducing the desired coverings.
this section cite: []

Section: Experimental study
In the following, we investigate to what extent our theoretical results translate into practice. Specifically, we answer the following questions.
Q1 To what extent do the empirical covering numbers for different graph families match the theoretical upper bounds derived in Section 4?
Q3 How are the Forest distance and MPNN outputs correlated?
Q3 How does the covering number influence the generalization performance of MPNNs?
See https://github.com/benfinkelshtein/  CoveredForests for source code and instructions to reproduce all results.
See Appendix P for an overview of employed data sets, neural architectures, experimental protocols, and model configurations.
this section cite: []

Section: Results and discussion
In the following, we answer Q1-Q3.
Q1 See Figures 8 and 9 in the appendix. Across the graph classes G n , T
n , F n , and all real-world datasets, we ob- serve that the covering number increases with larger graph orders or smaller radii, aligning with the expected behavior.
Figure 10 in the appendix demonstrates that the covering number bound presented in Theorem 10 is tighter than the number of 1-WL indistinguishable graphs, m n , when compared to the optimal cover, N (•, δ T 1 , ε), highlighting the usefulness of our upper bound. This observation holds even though the optimal cover is based on the 1-norm while our bound is derived using the cut norm, further solidifying the improved bound introduced in Theorem 10.
Q2 See Figure 11 in the appendix. We observe a strong correlation between the Forest distance and the MPNN output variations, indicated by a high Pearson correlation coefficient across varying datasets and MPNN layers. This finding supports the validity of defining the Forest distance's Lipschitz constant in Lemma 14, showing that it captures the computation of MPNNs.
Q3 See Table 1. The results demonstrate that the covering number bound presented in Proposition 15 is tight in real-world settings, in the sense that the empirical generalization gap and the computed bound are of the same order of magnitude, closely reflecting the real-world generalization behavior. Furthermore, incorporating the upper bound presented in Proposition 16 yields an even tighter bound over simply equating the covering number to the total number of 1-WL indistinguishable graphs.
this section cite: []

Section: Conclusion
Here, we focused on understanding how choosing different pseudo-metrics on the set of graphs and capturing the computation of different MPNN architectures allow for a tighter analysis of the generalization error of MPNNs. Unlike previous works, our refined analysis allows us to account for non-trivial graph similarities, the impact of different aggregation functions, and various practically relevant loss functions. Our empirical study confirmed the validity of our theoretical findings. Overall, our theoretical framework constitutes an essential initial step in unraveling how graph structure and architectural choices influence the MPNNs' generalization properties.
Hamilton, W. L., Ying, Z., and Leskovec, J. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, 2017. 15 Hammer, B. Generalization ability of folding networks. IEEE Transactions on Knowledge and Data Engineering, 13(2):196-206, 2001. 15 Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., and Leskovec, J. Open graph benchmark: Datasets for machine learning on graphs. In Advances in Neural Information Processing Systems, 2020. 48 Ju, H., Li, D., Sharma, A., and Zhang, H. R. Generalization in graph neural networks: Improved PAC-Bayesian bounds on graph diffusion. arXiv preprint, 2023. 15 Karczewski, R., Souza, A., and Garg, V. On the generalization of equivariant graph neural networks. In International Conference on Machine Learning, 2024. 15 Karpinski, M. and Macintyre, A. Polynomial bounds for VC dimension of sigmoidal and general Pfaffian neural networks. Journal of Computer and System Sciences, 54 (1):169-176, 1997. 15 Kawaguchi, K., Deng, Z., Luh, K., and Huang, J. Robustness implies generalization via data-dependent generalization bounds. In International Conference on Machine Learning, 2022. 1, 5, 15, 32 Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. 48 Kipf, T. N. and Welling, M. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017. 15 Kireev, D. B. Chemnet: A novel neural network based method for graph/property mapping. Journal of Chemical Information and Computer Sciences, 35(2):175-180, 1995. 15 Kriege, N. M., Morris, C., Rey, A., and Sohler, C. A property testing framework for the theoretical expressivity of graph kernels. In International Joint Conference on Artificial Intelligence, 2018. 16 Kriege, N. M., Johansson, F. D., and Morris, C. A survey on graph kernels. Applied Network Science, 5(1):6, 2020. 16 Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P., Fortunato, M., Alet, F., Ravuri, S., Ewalds, T., Eaton-Rosen, Z., Hu, W., Merose, A., Hoyer, S., Holland, G., Vinyals, O., Stott, J., Pritzel, A., Mohamed, S., and Battaglia, P. Learning skillful medium-range global weather forecasting. Science, 382(6677):1416-1421, 2023. 1 Lee, J., Hwang, M., and Whang, J. J. PAC-Bayesian generalization bounds for knowledge graph representation learning. In International Conference on Machine Learning, 2024. 15 Levie, R. A graphon-signal analysis of graph neural networks. In Advances in Neural Information Processing Systems, 2023. 15, 16 Levie, R., Monti, F., Bresson, X., and Bronstein, M. M. Cayleynets: Graph convolutional neural networks with complex rational spectral filters. IEEE Transactions on Signal Processing, 67(1):97-109, 2019. 15 Li, S., Geerts, F., Kim, D., and Wang, Q. Towards bridging generalization and expressivity of graph neural networks. arXiv preprint, 2024. 16 Liao, R., Urtasun, R., and Zemel, R. S. A PAC-Bayesian approach to generalization bounds for graph neural networks. In International Conference on Learning Representations, 2021. 1, 15 Lovász, L. Large Networks and Graph Limits. American Mathematical Society, 2012. 1, 34 Maehara, T. and NT, H. A simple proof of the universality of invariant/equivariant graph neural networks. arXiv preprint, 2019. 1 Mao, A., Mohri, M., and Zhong, Y. Cross-entropy loss functions: Theoretical analysis and applications. In International Conference on Machine Learning, 2023. 48 Maskey, S., Lee, Y., Levie, R., and Kutyniok, G. Generalization analysis of message passing neural networks on large random graphs. In Advances in Neural Information Processing Systems, 2022. 15 Maskey, S., Kutyniok, G., and Levie, R. Generalization bounds for message passing networks on mixture of graphons. arXiv preprint, 2024. 15 Merkwirth, C. and Lengauer, T. Automatic generation of complementary descriptors with molecular graph networks. Journal of Chemical Information and Modeling, 45(5):1159-1168, 2005. 15 Micheli, A. Neural network for graphs: A contextual constructive approach. IEEE Transactions on Neural Networks, 20(3):498-511, 2009. 15 Micheli, A. and Sestito, A. S. A new neural network model for contextual processing of graphs. In Italian Workshop on Neural Nets Neural Nets and International Workshop on Natural and Artificial Immune Systems, 2005. 15
this section cite: []

Section: References
Ref_id:b0 Title: Approximating the cut-norm via grothendieck's inequality Year: (2004)
Ref_id:b1 Title: Random sampling and approximation of max-csp problems Year: (2002)
Ref_id:b2 Title:  Year: (2004)
Ref_id:b3 Title: On the power of color refinement Year: (2015)
Ref_id:b4 Title: Characterizing the expressive power of invariant and equivariant graph neural networks Year: (2021)
Ref_id:b5 Title: Canonical labelling of graphs in linear average time Year: (1979)
Ref_id:b6 Title: Graph convolution for semi-supervised classification: Improved linear separability and out-of-distribution generalization Year: (2021)
Ref_id:b7 Title: Spectrallynormalized margin bounds for neural networks Year: (2017)
Ref_id:b8 Title: A neural device for searching direct correlations between structures and properties of chemical compounds Year: (1997)
Ref_id:b9 Title: A family of tractable graph metrics Year: (2019)
Ref_id:b10 Title: Graph similarity and homomorphism densities Year: (2021)
Ref_id:b11 Title: Graph kernels: State-of-the-art and future challenges Year: (2020)
Ref_id:b12 Title: Spectral networks and deep locally connected networks on graphs Year: (2014)
Ref_id:b13 Title: Fine-grained expressivity of graph neural networks Year: (2023)
Ref_id:b14 Title: An optimal lower bound on the number of variables for graph identifications Year: (1992)
Ref_id:b15 Title: Combinatorial optimization and reasoning with graph neural networks Year: (2023)
Ref_id:b16 Title: International Conference on Machine Learning Year: (2022)
Ref_id:b17 Title: The Weisfeiler-Lehman distance: Reinterpretation and connection with GNNs Year: (2023)
Ref_id:b18 Title: On the equivalence between graph isomorphism testing and function approximation with GNNs Year: (2019)
Ref_id:b19 Title: Tree mover's distance: Bridging graph metrics and stability of graph neural networks Year: (2022)
Ref_id:b20 Title: Measuring generalization with optimal transport Year: (2021)
Ref_id:b21 Title: A note on the VC dimension of 1-dimensional GNNs Year: (2024)
Ref_id:b22 Title: Convolutional neural networks on graphs with fast localized spectral filtering Year: (2016)
Ref_id:b23 Title: Lovász meets Weisfeiler and Leman Year: (2018)
Ref_id:b24 Title: Vc dimension of graph neural networks with pfaffian activation functions Year: (2024)
Ref_id:b25 Title: A hitchhiker's guide to geometric GNNs for 3d atomic systems Year: (2023)
Ref_id:b26 Title: Convolutional networks on graphs for learning molecular fingerprints Year: (2015)
Ref_id:b27 Title: On recognizing graphs by numbers of homomorphisms Year: (2010)
Ref_id:b28 Title: Crowds, and Markets: Reasoning About a Highly Connected World Year: (2010)
Ref_id:b29 Title: Operator Theoretic Aspects of Ergodic Theory Year: (2015)
Ref_id:b30 Title: Transductive rademacher complexity and its applications Year: (2007)
Ref_id:b31 Title: Learning theory can (sometimes) explain generalisation in graph neural networks Year: (2021)
Ref_id:b32 Title: How powerful are k-hop message passing graph neural networks Year: (2022)
Ref_id:b33 Title: Fast graph representation learning with PyTorch Geometric Year: (2019)
Ref_id:b34 Title: Mathematical constants Year: (2005)
Ref_id:b35 Title: A systematic approach to universal random features in graph neural networks Year: (2023)
Ref_id:b36 Title: Weisfeiler-leman at the margin: When more expressivity matters Year: (2024)
Ref_id:b37 Title: Quick approximation to matrices and applications Year: (1999)
Ref_id:b38 Title: Convolutional neural network architectures for signals supported on graphs Year: (2019)
Ref_id:b39 Title: Generalization and representational limits of graph neural networks Year: (2020)
Ref_id:b40 Title: Expressiveness and approximation properties of graph neural networks Year: (2022)
Ref_id:b41 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b42 Title: Introduction to testing graph properties Year: (2010)
Ref_id:b43 Title: Learning task-dependent distributed representations by backpropagation through structure Year: (1996)
Ref_id:b44 Title: Descriptive Complexity, Canonisation, and Definable Graph Structure Theory Year: (2017)
Ref_id:b45 Title: The logic of graph neural networks Year: (2021)
Ref_id:b46 Title: Foundations of Machine Learning Year: (2018)
Ref_id:b47 Title: Geometric deep learning on graphs and manifolds using mixture model cnns Year: (2017)
Ref_id:b48 Title: Weisfeiler and Leman go neural: Higher-order graph neural networks Year: (2019)
Ref_id:b49 Title: TUDataset: A collection of benchmark datasets for learning with graphs Year: (2020)
Ref_id:b50 Title: Weisfeiler and Leman go sparse: Towards higher-order graph embeddings Year: (2018)
Ref_id:b51 Title: WL meet VC Year: (2023)
Ref_id:b52 Title: Weisfeiler and Leman go machine learning: The story so far Year: (2023-01)
Ref_id:b53 Title: Future directions in the theory of graph machine learning Year: (2024)
Ref_id:b54 Title: Interior-Point Polynomial Algorithms in Convex Programming. Interior-Point Polynomial Algorithms in Convex Programming Year: ()
Ref_id:b55 Title: The number of trees Year: (1948)
Ref_id:b56 Title: On the expressivity and sample complexity of nodeindividualized graph neural networks Year: (2024)
Ref_id:b57 Title: Probabilistically rewired message-passing neural networks Year: (2024)
Ref_id:b58 Title: Generalization, expressivity, and universality of graph neural networks on attributed graphs Year: (2009)
Ref_id:b59 Title: E(n) equivariant graph neural networks Year: (2023)
Ref_id:b60 Title: The graph neural network model Year: (2009)
Ref_id:b61 Title: The Vapnik-Chervonenkis dimension of graph and recursive neural networks Year: (2018)
Ref_id:b62 Title: Machine learning augmented branch and bound for mixed integer linear programming Year: (2024)
Ref_id:b63 Title: Generalization error of invariant classifiers Year: (2017)
Ref_id:b64 Title: Supervised neural networks for the classification of structures Year: (1997)
Ref_id:b65 Title: A bi-lipschitz wl-equivalent graph neural network Year: (2024)
Ref_id:b66 Title: Towards understanding generalization of graph neural networks Year: (2023)
Ref_id:b67 Title: Graph isomorphism and theorems of Birkhoff type. Computing Year: (1986)
Ref_id:b68 Title: A note on compact graphs Year: (1991)
Ref_id:b69 Title: Minimax lower bounds for realizable transductive classification Year: (2016)
Ref_id:b70 Title: Weak Convergence and Empirical Processes: With Applications to Statistics Year: (1996)
Ref_id:b71 Title: Statistical learning theory Year: (1998)
Ref_id:b72 Title: The Nature of Statistical Learning Theory Year: (1995)
Ref_id:b73 Title: A note on one class of perceptrons Year: (1964)
Ref_id:b74 Title: On the uniform convergence of relative frequencies of events to their probabilities Year: (1971)
Ref_id:b75 Title: Graph attention networks Year: (2018)
Ref_id:b76 Title: Stability and generalization of graph convolutional neural networks Year: (2019)
Ref_id:b77 Title: On Construction and Identification of Graphs Year: (1976)
Ref_id:b78 Title: The reduction of a graph to canonical form and the algebra which appears therein Year: (1968)
Ref_id:b79 Title: Discovery of a structural class of antibiotics with explainable deep learning Year: (2023)
Ref_id:b80 Title: Robustness and generalization Year: (2012)
Ref_id:b81 Title: How powerful are graph neural networks? Year: (2019)
Ref_id:b82 Title: From local structures to size generalization in graph neural networks Year: (2021)
Ref_id:b83 Title: Artificial intelligence for science in quantum, atomistic, and continuum systems Year: (2023)
