Title: DISTRIBUTIONAL EQUIVALENCE IN LINEAR NON-GAUSSIAN LATENT-VARIABLE CYCLIC CAUSAL MOD-ELS: CHARACTERIZATION AND LEARNING
Abstract: Causal discovery with latent variables is a fundamental task. Yet most existing methods rely on strong structural assumptions, such as enforcing specific indicator patterns for latents or restricting how they can interact with others. We argue that a core obstacle to a general, structural-assumption-free approach is the lack of an equivalence characterization: without knowing what can be identified, one generally cannot design methods for how to identify it. In this work, we aim to close this gap for linear non-Gaussian models. We establish the graphical criterion for when two graphs with arbitrary latent structure and cycles are distributionally equivalent, that is, they induce the same observed distribution set. Key to our approach is a new tool, edge rank constraints, which fills a missing piece in the toolbox for latent-variable causal discovery in even broader settings. We further provide a procedure to traverse the whole equivalence class and develop an algorithm to recover models from data up to such equivalence. To our knowledge, this is the first equivalence characterization with latent variables in any parametric setting without structural assumptions, and hence the first structural-assumption-free discovery method. Code and an interactive demo are available at https://equiv.cc.

Section: INTRODUCTION
At the core of scientific inquiry lies causal discovery, the task of learning causal relations from observational data (Spirtes et al., 2000;Pearl, 2009). In many real-world scenarios, the variables of interest can be unobserved. For instance, in psychology, personality traits are hidden behind survey responses, and in biology, crucial regulators may be unobserved due to technical inaccessibility. Discovering the causal structure with these latent variables, referred to as latent-variable causal discovery, is essential for understanding and reasoning, yet remains a challenging task.
Latent-variable causal discovery has seen significant development over the past three decades. A milestone was the Fast Causal Inference (FCI) algorithm (Spirtes, 1992), which exploits conditional independence (CI) constraints under hidden confounding. However, FCI is typically not regarded as a method of latent-variable causal discovery, as it focuses solely on causal relations among observed variables, with no intension or capability to identify those among latent variables. In fact, though FCI is already maximally informative under nonparametric CI constraints (Richardson & Spirtes, 2002;Zhang, 2008a), it is still not informative enough for recovering latent structure. This limitation has motivated the development of many recent approaches that go beyond CI constraints, typically by introducing parametric assumptions, such as linearity (Silva et al., 2003;Dong et al., 2024), non-Gaussianity (Hoyer et al., 2008;Jin et al., 2024), mixture models (Kivva et al., 2021), and distribution shifts (Zhang et al., 2024). Within each setting, a rich array of techniques has emerged. For example, in the linear non-Gaussian setting alone, methods have been developed based on overcomplete independent component analysis (OICA) (Salehkaleybar et al., 2020), regression (Tashiro et al., 2014), Bayesian estimation (Shimizu & Bollen, 2014), independence testing (Xie et al., 2020), cumulants (Robeva & Seby, 2021), independent subspace analysis (Dai et al., 2024), and many more.
However, despite this prosperity, most methods share a clear limitation: they rely on structural assumptions, often about how latent variables are indicated and how they can interact with others. Common examples include measurement models where observed variables have to be pure measure-ments of latents (Silva & Scheines, 2004;Zhang et al., 2018); hierarchical models that prohibit effects from observed variables (Choi et al., 2011;Huang et al., 2020); sufficient number of pure children per latent (Squires et al., 2022;Jin et al., 2024); and assumptions like triangle-or bow-freeness (Dong et al., 2024;Wang & Drton, 2023). In addition, most methods also assume acyclicity, even though feedback loops are common in real systems. These assumptions, often overly strong and untestable, not only limit applicability but also complicate method selection for practitioners.
A pressing question naturally arises: after decades of progress, is it possible now to have a general structural-assumption-free approach for latent-variable causal discovery that, like FCI, allows arbitrary relations among latent and observed variables, yet goes beyond FCI's limited informativeness?
One core obstacle, we argue, is the lack of a general equivalence characterization with latent variables. Equivalence is a notion fundamental to causal discovery: when different causal models induce the same observed distribution set (known as distributional equivalence), no method can, or should, distinguish among them, without extra information like interventions or sparsity constraints. The expected discovery output is thus the entire equivalence class, the best one can hope to identify from data. In practice, equivalence can also be defined more coarsely, depending on the specific constraints used. One example is Markov equivalence, capturing when models entail the same CI constraints. A well-known and nice result is that in causally sufficient, acyclic, and nonparametric models, Markov equivalence coincides with distributional equivalence (Spirtes et al., 2000); the resulting equivalence class is represented by a completed partially directed acyclic graph (CPDAG).
In the presence of cycles or latent variables, however, equivalence characterization becomes more complex. For example, the nice coincidence between Markov and distributional equivalences breaks down, even with only cycles (Spirtes, 1994;Mooij & Claassen, 2020), or only latent variables (Verma & Pearl, 1991;Richardson et al., 2023), let alone both. The resulting equivalence classes, be it Markov (Richardson & Spirtes, 2002;Claassen & Mooij, 2023) or distributional (Nowzohour et al., 2017;Evans, 2018), also become far more complex. Such complications carry over to parametric settings: for cycles alone, distributional equivalence has been characterized in linear non-Gaussian and Gaussian models (Lacerda et al., 2008;Ghassami et al., 2020); yet for latent variables, no characterization of any kind, whether distributional or constraint specific, is currently known to us. The closest result (Adams et al., 2021) gives conditions for when a linear non-Gaussian acyclic model can be uniquely identified, but leaves open describing the equivalence when such identifiability fails.
All such complications from latent variables and cycles have so far prevented a general equivalence characterization, which is exactly what obstruct progress towards a structural-assumption-free method. The need for such a characterization is yet clear: without knowing what can be identified, one generally cannot design methods for how to identify it. This is echoed in history: PC algorithm followed CPDAGs; FCI's guarantee followed maximal ancestral graphs (MAGs) (Richardson & Spirtes, 2002).
Our goal in this work is hence to overcome these challenges and establish a general equivalence notion with latent variables and cycles. We focus on linear non-Gaussian models, a parametric setting that has received much attention. Under this setting, we address three questions: 1) When are two graphs with arbitrary latent variables and cycles equivalent? 2) How can one traverse the entire equivalence class? 3) How can one recover latent-variable models up to equivalence from data? Centered around these three questions, our contributions are summarized as follows:
1. We present a general equivalence notion that allows arbitrary latent structure and cycles in linear non-Gaussian models. This is the first such result known to us in any parametric setting ( §2). 2. We introduce a new tool, edge rank constraints. It contributes a missing piece to the broader toolbox for latent-variable causal discovery, with potential use across many settings ( §3). 3. We characterize equivalence graphically and provide procedures to traverse the entire class. Results are cleaner than expected. We provide an interactive demo at https://equiv.cc ( §4). 4. We develop an efficient algorithm to recover the equivalence class from data, which is, to our knowledge, the first structural-assumption-free method for latent-variable causal discovery ( §5).
this section cite: ['b110', 'b83', 'b105', 'b88', 'b103', 'b22', 'b46', 'b55', 'b57', 'b94', 'b118', 'b99', 'b134', 'b91', 'b19', 'b101', 'b149', 'b14', 'b50', 'b113', 'b55', 'b22', 'b130', 'b110', 'b106', 'b127', 'b90', 'b88', 'b16', 'b80', 'b35', 'b62', 'b41', 'b0', 'b88']

Section: PROBLEM SETUP
In this section, we lay the groundwork for our study. In §2.1, we define the notion of distributional equivalence in linear non-Gaussian latent-variable causal models. Then in §2.2, we introduce the idea of irreducibility to rule out trivial cases, clearing the way for the main results to come.
this section cite: []

Section: PRELIMINARIES FOR LINEAR NON-GAUSSIAN MODELS
Notations on matrices. For a matrix M , we let M i,j be its (i, j)-th entry. For two index sets A, B, we let M A,B = (M a,b ) a∈A,b∈B be the submatrix of M with rows indexed by A and columns indexed by B. We let M A,: be the rows in M indexed by A, and similarly M :,B for the columns. For a finite set A, we denote its cardinality by |A|. We denote by Scale(d) the set of all d × d diagonal matrices with nonzero diagonal entries, and by Perm(d) the set of all d × d permutation matrices.
For a permutation π : V → V on a ground set V , we denote π(F ) := {π(i) : i ∈ F } for any set F ⊆ V , and extend this notation to families of sets by π(F) := {π(F ) : F ∈ F} for F ⊂ 2 V .
Notations on graphs. Throughout, by a digraph we refer to a directed graph that may contain cycles but no self-loops (edges from a vertex to itself). In a digraph G, let V (G) be its vertex set. For vertices a, b, we say a is a parent of b and b is a child of a, denoted by a ∈ pa G (b) and b ∈ ch G (a), when a → b is an edge in G, written a → b ∈ G; a is an ancestor of b and b is a descendant of a, denoted by a ∈ an G (b) and b ∈ de G (a), when a = b or there is a directed path a → • • • → b in G.
These notations extend to sets: e.g., for a vertex set A, an G (A) := a∈A an G (a).
Linear non-Gaussian (LiNG) causal models. We consider a linear non-Gaussian model associated with a digraph G, in which random variables
V = (V 1 , • • • , V |V | ) ⊤
, corresponding to the vertices of G, are generated according to the structural equation:
V = BV + E,(1)
where E = (E 1 , • • • , E |V | ) ⊤ consists of mutually independent, non-constant, non-Gaussian exogenous noise terms. The matrix B ∈ B(G) is a weighted adjacency matrix (whose entries represent direct causal effects) that follows G, where B(G), all adjacency matrices that follow G, is defined as:
B(G) := {B ∈ R |V |×|V | : B Vj ,Vi ̸ = 0 =⇒ V i → V j ∈ G}.
(2) Assuming I -B is invertible, solving for Equation (1) gives an equivalent mixing form:
V = (I -B) -1 E =: AE,(3)
where A is called the weighted mixing matrix. The entry A Vj ,Vi represents the total causal effect from V i to V j . All mixing matrices that follow G, denoted by A(G), is defined as:
A(G) := {(I -B) -1 : B ∈ B(G), I -B invertible}.
(4)
Latent-variable LiNG models. Let the vertices V of a digraph G be partitioned as V = L ∪ X, where L denotes latent (unobserved) variables and X denotes observed variables. A latent-variable model is specified by the tuple (G, X), with latent variables L omitted when clear from context.
Given a full mixing matrix A ∈ A(G), the submatrix A X,: ∈ R |X|×|V | maps exogenous noise terms to the observed variables. The collection of such wide rectangular mixing matrices is defined as: A(G, X) := {A X,: : A ∈ A(G)}.
(5) Accordingly, the induced observed distribution set of G on X, that is, the set of all distributions over X that can arise from a LiNG model over (G, X), denoted P(G, X), is given by:
P(G, X) := {p(X) : X = AE, A ∈ A(G, X), E ∈ NG(|V |)},(6)
where p(X) denotes the probability distribution of the random vector X, and NG(d) denotes the set of all d-dim random vectors with mutually independent, non-constant, and non-Gaussian components.
We are now ready to formalize the central notion of this work: distributional equivalence. Definition 1 (Distributional equivalence). Let G and H be two digraphs with possibly different vertices, and X ⊆ V (G)∩V (H) be the shared observed variables. We say G and H are distributionally equivalent (or for short, equivalent) on X, denoted by G X ∼ H, when P(G, X) = P(H, X).
The equivalence (Definition 1) captures when two models yield identical observed distribution set, i.e., observationally indistinguishable. With this notion in place, next we clean up some trivialities.
this section cite: []

Section: IRREDUCIBILITY: TO FIRST RULE OUT TRIVIAL CASES OF EQUIVALENCE
To study identifiability, let us first see what is inherently non-identifiable. For instance, one can freely add latent vertices that are not ancestors of any observed variables X to a digraph G without affecting P(G, X), yielding trivially equivalent models. Identifying those latents is both impossible and meaningless. To rule out such trivialities, we introduce the notion of irreducibility.
L 1 L 2 X 1 X 2 L 1 X 1 X 2 reduce L 1 L 2 L 3 X 1 X 2 X 3 L 1 L 2 X 1 X 2 X 3 reduce L 1 L 2 X 1 X 2 X 1 X 2 reduce
Figure 1: Examples of reducing models to their irreducible forms via the procedure in Proposition 2. Throughout, white circles denote observed variables and grey squares denote latent variables.
Definition 2 (Irreducibility). We say a latent-variable model (G, X) is irreducible, when there exists
no digraph H with |V (H)| < |V (G)| such that G X ∼ H.
Irreducibility captures when an observed distribution set cannot arise from any other model with fewer latent variables. We now present a simple graphical condition for this property.
Proposition 1 (Graphical condition for irreducibility). A model (G, X) is irreducible, if and only if for each non-empty set l ⊆ L, | ch G (l) \ l| ≥ 2, i.e.
, it has more than one child outside.
Note that when G is acyclic, it suffices to check each single L i ∈ L, consistent with the condition previously derived by Salehkaleybar et al. (2020). The proof of Proposition 1, along with others, is provided in Appendix B. The key idea here is that any violation of the condition leads to proportional columns in mixing matrices A(G, X), so that the observed distributions can be equivalently generated by a smaller graph with these columns merged to one. Conversely, identifiability results of OICA (Eriksson & Koivunen, 2004) suggest that as long as in the absence of such proportional columns, the mixing matrix is identifiable up to column scaling and permutation, so the number of latents is identifiable.
We next provide an explicit procedure for reducing an arbitrary model to its irreducible form.
Proposition 2 (Procedure of reduction to the irreducible form). Given any latent-variable model (G, X), the following procedure outputs a digraph H such that H Step 4. For each l ∈ mrl, let c be the exact child in ch H (l)\l; for each parent p ∈ pa H (l)\l\{c}, add an edge p → c into H if not already present; finally, remove l vertices from H.
Illustrative examples of this reduction are shown in Figure 1. This reduction lets us, without loss of generality, restrict attention to irreducible models for the remainder, as arbitrary models are equivalent if and only if their irreducible forms are equivalent. Note that irreducibility is not a structural assumption as discussed in §1, but rather a canonicalization to eliminate trivialities. As a side note, applying the reduction in Proposition 2 does not increase the number of edges or cycles.
this section cite: ['b94', 'b33']

Section: DEVELOPING GRAPHICAL TOOLS FOR CHARACTERIZING EQUIVALENCE
In the previous section, we defined distributional equivalence and irreducibility to rule out trivial unidentifiable cases, so we can focus solely on irreducible models in what follows. Then, when are two irreducible models equivalent? In this section, we tackle this question step by step.
Specifically, in §3.1 we first show that distributional equivalence reduces to an algebraic condition on mixing matrices, and further to a graphical condition involving a concept familiar to the community: path ranks, given by max-flow-min-cuts in digraphs. Although familiar, path ranks are difficult to work with due to their global, non-local nature, as we illustrate in §3.2. To overcome this, we introduce a new tool: edge ranks, a local, edge-level constraint that complements path ranks and is easier to manipulate. This new tool, developed in §3.3, not only enables our final result to come in the next section, but also enriches the broader rank-based picture beyond our specific setting.
this section cite: []

Section: EQUIVALENCE VIA PATH RANKS
We start by examining the algebra behind distributional equivalence. By Definition 2, all equivalent irreducible models must have the same number of latents. This follows from OICA, which guarantees exact recovery of the number of (nontrivial) latent variables. Hence, in what follows, when considering the equivalence of two irreducible models (G, X) and (H, X), we can, without loss of generality, denote their latent variables by a same set of labels, so that V (G) = V (H) = X ∪ L.
We then observe that distributional equivalence can be rephrased in terms of the mixing matrices: two models are equivalent if and only if for every mixing matrix one model can generate, the other can also generate a version of it up to column scaling and permutation, and vice versa, due to the scaling and permutation closedness of exogenous noise terms. Formally, Lemma 1 (Equivalence via mixing matrices closure). Two irreducible models are equivalent, written G X ∼ H, if and only if A(G, X) = A(H, X), where for a set of matrices A ⊆ R m×d , we let: A := {AP D : A ∈ A, P ∈ Perm(d), D ∈ Scale(d)}, (8) that is, the closure of A up to column scaling and permutation.
Then, what are exactly these mixing matrices, namely, A(G, X)? As defined in Equations ( 2) to ( 5), it arises from a mapping over the free parameters in adjacency matrices. Concretely, each entry of the mixing matrix is a rational function: the numerator polynomial reflects "total causal effects" between variables, and the denominator polynomial accounts for "global cycle discounts", which is simply 1 when the digraph is acyclic. In cyclic cases, there is a small pathological locus where denominators vanish, that is, where I -B becomes singular and cycles "cancel themselves." But as we will show in the proof, this does not affect our results. So for now, let us progress with the Zariski closure of A(G, X), an algebraic variety that can be defined by finitely many equality constraints.
We now study these constraints. One fundamental class of them is the so-called rank constraints, which admits a nice graphical interpretation in terms of max-flow-min-cut in digraphs, defined below:
Definition 3 (Path ranks). In a digraph G, for two sets of vertices Z, Y ⊆ V (G), the path rank ρ G (Z, Y ) is defined as the maximum number of vertex-disjoint directed paths from Y to Z in G.
By (Menger, 1927), this max-flow quantity can also be defined by its min-cut version:
ρ G (Z, Y ) := min c⊆V (G) {|c| : c's removal from G ensures no directed path from Y \c to Z\c}. (9
)
These purely graphical quantities can be read off from the mixing matrices by examining the matrix ranks of corresponding submatrices, which is the well-known (path) rank constraint: Lemma 2 (Path rank constraints in mixing matrices). In a digraph G, for any two sets of vertices Z, Y ⊆ V (G) that need not be disjoint, the following equality holds for generic choice of A ∈ A(G):
rank(A Z,Y ) = ρ G (Z, Y ).(10)
Here, rank denotes the usual matrix rank, and "generic" means the equality holds almost everywhere except for a Lebesgue measure zero set where coincidental lower matrix ranks occur.
Rank constraints bridge algebra in matrices with geometry in digraphs. They were initially proved for acyclic graphs only (Lindström, 1973;Gessel & Viennot, 1985), and later generalized by (Talaska, 2012). They are powerful: as we will show in the proof, rank constraints alone, together with a column permutation, suffice to determine equivalence. We directly state the result below: Lemma 3 (Equivalence via path ranks). Two irreducible models are distributionally equivalent, written G X ∼ H, if and only if there exists a permutation π over the vertices V (G), such that
ρ G (Z, Y ) = ρ H (Z, π(Y )) for all Z ⊆ X and Y ⊆ V (G).(11)
From Lemma 1 to Lemma 3, so far we have arrived at a first purely graphical view of equivalence.
this section cite: ['b76', 'b63', 'b39', 'b117']

Section: THE COMPLEXITY OF MANIPULATING PATH RANKS
In §3.1 we have arrived at Lemma 3, a purely graphical characterization of equivalence, which, perhaps surprisingly, is expressed in terms of a familiar concept: path ranks. However, this is only a start and far from operational: verifying it requires searching over all vertex permutations and all (Z, Y ) pairs, which quickly becomes intractable due to their factorial and exponential growth, let alone the costly graph traversal required for each single path rank computation. As an analogy to the acyclic, causally sufficient case, Lemma 3 is like saying "having all the same d-separations," whereas what we seek is something simpler and more local, like "same adjacencies and v-structures."
Y 1 Y 2 • • • Y m C 1 C 2 Z 1 Z 2 • • • Z n Y 1 Y 2 • • • Y m C 1 C 2 C 1 C 2 Z 1 Z 2 • • • Z n × × × 0 × • • • × × × 0 × × • • • × 0 0 × × 0 • • • 0 0 0 × × 0 • • • 0 . . . . . . . . . . . . . . . . . . . . . 0 0 × × 0 • • • 0                 Y1 Y2 C1 C2 Y3 • • • Ym C1 C2 Z1 Z2
. . .
this section cite: []

Section: Zn
Figure 2: An illustration of path ranks, edge ranks, and their duality. Left: a digraph G with vertices V partitioned to Y , C, and Z, shown by different colors. The path rank ρ G (Z, Y ) = 2, with C being a min-cut. Right: the dual edge rank r G (V \Y, V \Z) = 4, given by the maximum bipartite matching from V \Z to V \Y , i.e., from Y ∪ C to Z ∪ C, with four matched edges highlighted in red. Four corresponding nonzero entries placed on diagonal, also in red, confirm mrank(Q
(G) V \Y, V \Z ) = 4.
One may examine the duality in Theorem 1: w.l.o.g. let m ≤ n, there is m -2 = m + n + 2n -4.
Then, does a simpler local condition naturally follow from Lemma 3? Unfortunately, not quite. Path ranks are hard to work with due to their global nature: they summarize the size of "bottlenecks", but say nothing about which paths are involved or how they interact. Each single edge may lie on multiple bottlenecks, so even a small local alteration to a digraph may trigger unpredictable global changes in path ranks. Conversely, with latent variables, seemingly very different digraphs can still share the same path ranks. We illustrate such complexity with the following example. Example 1 (Complexity of viewing equivalence via path ranks). Consider the digraph G on the left of Figure 2, with vertices partitioned into Y , C, and Z. Obviously, the path rank ρ G (Z, Y ) = 2. Now, suppose vertices {C 1 , C 2 } become latent and all others remain observed. What models are equivalent? This is not obvious anymore. It usually takes some thought to realize that adding edges or cycles within C, or removing one or two edges from C to Z, still preserves path ranks as in Lemma 3. What about the Y to C structure then? This is more subtle: when n > 2, it must remain fixed; but when n = 2, C is no longer a unique bottleneck, and suddenly, Y can point freely to both C and Z.
Things become even less intuitive when other variables are latent. For example, with m = n = 4, if {C 1 , C 2 } are latent, there are 17 digraphs in the equivalence class (view them online). When {Y 1 , Y 2 } or {Y 1 , C 1 } are latent, this number comes to 872 (view) and 1, 024 (view), respectively. Note that all this comes from a well structured digraph; arbitrary structures only lead to greater complexity. △ Example 1 illustrates the complexity of path ranks in inferring graph structures. In fact, this complexity is well recognized in literature: despite various techniques developed to estimate path ranks from data (Dai et al., 2022;Sturma et al., 2024), and well-studied counterparts in the linear Gaussian (Sullivant et al., 2010) and discrete settings (Chen et al., 2024b) or even with selection bias (Dai et al., 2025b), when it comes to structure learning from ranks, usually restrictive structural assumptions are required to ensure clean interpretation to where and how these paths can be.
All observations above motivate a question: is there a more local, graph-manipulable alternative to path ranks, not just for building equivalence in this work but also as a piece in the broader toolbox?
Interestingly, the answer is yes, and we develop such a tool next: edge ranks.
this section cite: ['b18', 'b115', 'b116']

Section: EDGE RANKS: A NEW TOOL IN THE RANK-BASED PICTURE
We now introduce a new tool: edge ranks. As the name suggests, edge ranks directly operate on edges in digraphs, which is more local and accessible in contrast to the paths used in path ranks. For intuition, one may refer to Figure 2, which illustrates all the concepts and results below.
Let us first define edge ranks, similar to how we define path ranks previously in Definition 3: Definition 4 (Edge ranks). In a digraph G, for two sets of vertices Z, Y ⊆ V (G), the edge rank r G (Z, Y ) is defined as the size of the maximum bipartite matching from Y to Z via edges in G, where self-matches (a to a for a ∈ Y ∩ Z) are allowed. Edge ranks also admit a min-cut version: r G (Z, Y ) := min z⊆Z, y⊆Y, z∪y⊇Z∩Y {|z| + |y| : there is no edge from Y \y to Z\z in G}. (12)
In parallel to how path ranks correspond to matrix ranks of mixing submatrices (cf. Lemma 2), the pure graphical quantities of edge ranks also have their algebraic counterpart. This time, it is not the mixing matrices at play, but directly the adjacencies.
For clarity, let us introduce a new matrix notation, Q, in addition to the already familiar notations of B and A, and a new notion of matching ranks, in addition to the already familiar matrix ranks.
Definition 5 (Support matrix). For a digraph G, its binary support matrix in shape G) , is given by:
|V (G)| × |V (G)|, denoted Q (
Q (G) Vj ,Vi = ' × ' if V i = V j or V i → V j ∈ G, and 0 otherwise. (13
)
Definition 6 (Matching rank of a matrix). The matching rank of a matrix M ∈ K m×n is given by: mrank(M ) := max
P ∈Perm(n) i=1,••• ,min(m,n) 1((MP ) i,i ̸ = 0).(14)
In simple terms, the matching rank of a matrix, denoted mrank, is the maximum number of nonzero entries that can be positioned on the diagonal by permuting its columns (or rows).
We can now give the edge rank constraints, as a counterpart to path rank constraints (cf. Lemma 2). Unlike the algebraic efforts required there, this result follows immediately from definition: Lemma 4 (Edge rank constraints in support matrices). In a digraph G, for any two sets of vertices Z, Y ⊆ V (G) that need not be disjoint, the following equality holds:
mrank(Q (G) Z,Y ) = r G (Z, Y ).(15)
So far, we have defined both path ranks and edge ranks, which at first glance appear so different: graphically, one is global, focusing on paths, while the other is local, operating on edges; algebraically, one is tied to weighted mixing matrices, the other to binary support matrices. However, despite these apparent differences, a surprising and elegant duality exists between them: Theorem 1 (Duality between path ranks and edge ranks). In a digraph G with vertices V , for any two sets of vertices Z, Y ⊆ V that need not be disjoint, the following equality holds:
min(|Z|, |Y |) -ρ G (Z, Y ) = |V | -max(|Z|, |Y |) -r G (V \Y, V \Z). (16
)
This duality is powerful: it suggests that every statement phrased in terms of path ranks and its variants, including the familiar d-separation and t-separation, can be equivalently rephrased in terms of edge ranks. It reveals that, despite the very different graphical objects involved in the two ranks, they offer complementary perspectives on a same notion in the digraph, namely, bottleneck, which captures how dependencies arise in observed data, and is thus central to causal discovery.
In fact, this duality has long been studied in the matroid community (Kőnig, 1931;Perfect, 1968;Ingleton & Piff, 1973), while only the path rank side has been well known in causal discovery. We thus introduce edge ranks here, filling the other side to the rank-based toolbox. It is not that edge ranks are always better, but having both perspectives is beneficial. Within this work, edge ranks indeed lead to simpler derivations. For instance, let us rephrase Lemma 3 using edge ranks below: Lemma 5 (Equivalence via edge ranks). Two irreducible models are distributionally equivalent, written G X ∼ H, if and only if there exists a permutation π over the vertices
V (G), such that r G (Z, Y ) = r H (π(Z), Y ) for all Z, Y ⊆ V (G) with L ⊆ Y.(17)
As we will see in the next section, this formulation paves the way to our final criterion for equivalence.
To conclude this section, we provide a side-by-side comparison of two ranks (Table 1; Appendix C.1).
this section cite: ['b60', 'b85', 'b52']

Section: THE GRAPHICAL CHARACTERIZATION OF DISTRIBUTIONAL EQUIVALENCE
In previous sections, through a step-by-step breakdown of equivalence, we have arrived at a key result, Lemma 5, which, notably, is framed by a new tool we introduced: edge ranks. Building on this foundation, in this section, we provide our final graphical criterion for distributional equivalence, and present a transformational characterization that enables traversal of all digraphs in the equivalence class.
We first study the task of deciding whether two given models are equivalent. For this purpose, although Lemma 5 offers a more local condition for each rank check, it still requires a large number of total checks: one must go through all sets Y ⊇ L, which amounts to all subsets x ⊆ X. As noted in our earlier analogy ( §3.2), this remains akin to "same d-separations," instead of a practical criterion like "same adjacencies and v-structures." Then, does Lemma 5 yield such a practical criterion?
Fortunately, this time, the answer is yes. Unlike the complexities encountered with path ranks in §3.2, edge ranks allow Lemma 5 to admit a nice local decomposition: instead of checking all subsets x ⊆ X, it suffices to check each singleton X i ∈ X independently. This yields our final graphical criterion:
Theorem 2 (Graphical criterion for distributional equivalence). In a digraph G, we define the "children bases" of a vertex set Y ⊆ V (G) as vertex sets that admit perfect edge matchings from Y :
bases G (Y ) := {Z ⊆ ch G (Y ) ∪ Y : r G (Z, Y ) = |Z| = |Y |}.(18)
Then, two irreducible models (G, X) and (H, X) are distributionally equivalent, if and only if there exists a permutation π over the vertices V (G), such that the following conditions hold:
bases G (L) = π(bases H (L)), and bases G (L ∪ {X i }) = π(bases H (L ∪ {X i })) for each X i ∈ X.(19)
To interpret this criterion, let us consider the causally sufficient case where L = ∅. In this case, each bases G ({X i }) is just X i with its children. Then, Theorem 2 immediately reduces to the classical result of exact digraph identification up to permutation (Lacerda et al., 2008). Interestingly, that result has recently been revisited also from a bipartite matching view used here (Sharifian et al., 2025).
Having established Theorem 2 as an efficient criterion for determining equivalence, we now turn to another task of traversing all digraphs in an equivalence class. For this purpose, however, a determining criterion alone offers little guidance. Again, we recall the analogy with Markov equivalence. Note that except for the criterion of "same adjacencies and v-structures," there is an alternative characterization: "two acyclic digraphs are equivalent if and only if one can reach the other via a sequence of covered edge reversals," known as "Meek conjecture" (Meek, 1997). Such a transformational characterization offers a natural way for equivalence class traversal. In light of it, we next develop such a transformational characterization, analogous to "Meek conjecture" for our setting.
We start with the permutation part in Theorem 2, which corresponds to row permutations to the support matrix Q (G) . Such permutations must result in valid support matrices, i.e., ones with nonzero diagonals. By cycle decomposition of permutations, this leads to an observation: disjoint cycles in the digraph can be freely reversed without affecting equivalence. Formally:
Lemma 6 (Admissible cycle reversals). For a digraph G, let C be any collection of vertex-disjoint simple cycles in G. Define a new digraph H where for each edge V i → V j ∈ G:
1. If V i → V j is on a cycle in C, then include V j → V i in H; 2. Otherwise, if V j is on a cycle in C with the predecessor V k → V j , then include V i → V k in H; 3. Otherwise, simply include V i → V j in H.
Then, with this new H, the equivalence
G X ∼ H still holds, for every X ⊆ V (G).
This result was also shown by (Lacerda et al., 2008). It highlights that in the linear non-Gaussian setting, cycles do not introduce substantial complexity. One may illustrate it using examples in Figure 3.
We then examine a more subtle part in Theorem 2, concerning edge rank equivalence, that is, when all the involved perfect bipartite matchings via edges are unchanged. Intuitively, it is about how edges are structurally "crucial" for maintaining matchings. This leads to the following criterion about edge additions or deletions, corresponding to flipping entries in the support matrix:
Lemma 7 (Admissible edge additions/deletions). Let (G, X) be an irreducible model. For any edge
V i → V j not currently in G, adding it to G preserves equivalence on X if and only if: r G (V i 's nonchildren\{V j }, L\{V i }) < r G (V i 's nonchildren, L\{V i }),(20)
where
V i 's nonchildren denotes V (G)\ ch G (V i )\{V i }, i.e., zero entries in support column Q (G)
:,Vi . Conversely, an edge can be deleted if and only if it can be re-added by this criterion.
In layman's term, Lemma 7 says that an edge V i → V j can be added, only when in the bipartite graph from latents to all vertices currently not V i 's children (including V j ), V j stands as a "pillar" across the maximum matchings; in matroid terms, it is a coloop. Then, since V j is already a "pillar", adding this edge will not be noticed by any Y containing latent variables. Note that both V i and V j may be in X or L: edges can be added within each or in either direction. Let us examine an example. Example 2 (Illustrating edge additions via Lemma 7). We consider the digraph G 1 in Figure 3, and check why the edge
L 1 L 2 X 1 X 2 X 3 G 1 L 1 L 2 X 1 X 2 X 3 G 2 L 1 L 2 X 1 X 2 X 3 G 3 L 1 L 2 X 1 X 2 X 3 G 4 L 1 L 2 X 1 X 2 X 3 G 5 L 1 L 2 X 1 X 2 X 3 G 6 Edge +/- Edge +/- Edge +/- Edge +/- Cycle reversal Cycle reversal
X 2 → L 2 can be added. From L\{X 2 } = {L 1 , L 2 } to X 2 's nonchildren {L 1 , L 2 , X 1 }, there is a full matching of size 2, with (L 1 , L 2 ) matched to either (L 1 , L 2 ) or (X 1 , L 2 ). Since L 2 appears in both as a "pillar", adding X 2 → L 2 preserves edge ranks. In contrast, X 2 → L 1 cannot be added, which, for instance, will change r G1 ({L 1 , L 2 , X 1 }, {L 1 , L 2 , X 2 }) from 2 to 3. △
We have introduced two graphical operations that preserve equivalence, namely, cycle reversals and edge additions/deletions. Remarkably, these two operations are not only sufficient but also necessary: together, they fully characterize equivalence. This brings us to our transformational characterization: Theorem 3 (Transformational characterization of the equivalence class). Two irreducible models (G, X) and (H, X) are equivalent if and only if G can be transformed into H, up to L-relabeling, via a sequence of admissible cycle reversals and edge additions/deletions, as defined in Lemmas 6 and 7.
Here, "up to L-relabeling" means there exists a relabeling of L in H yielding a digraph H ′ such that G reaches H ′ via the sequence. Moreover, at most one cycle reversal is needed in this sequence.
Thanks to this transformational characterization, Theorem 3 offers a natural way to traverse an equivalence class by e.g., running BFS or DFS over the space of digraphs connected via admissible operations. Such equivalence class structures are illustrated by Figure 3, Figure 5 (Appendix C.2), and more in our online demo. Note that this traversal can be further accelerated in implementation, by traversing each vertex's children independently in parallel (Lemmas 9 and 12; Appendix B).
Finally, let us return once more to the analogy with Markov equivalence. We have now established counterparts of both "same adjacencies and v-structures" and "Meek conjecture". A natural question is then whether a counterpart of the CPDAG, an informative presentation of the equivalence class, can also be developed. The answer is again yes. We show that within each cycle-reversal configuration, there exists a unique maximal equivalent digraph of which all others are subgraphs. We further provide efficient criteria to construct this maximal digraph, and to determine edges invariant across the equivalence class (similar to arrows in a CPDAG). Due to space limit, this result is presented in Theorem 4 (Appendix C.3). To conclude this section, we provide a side-by-side overview that places our results with their analogues across various classical settings (Table 2; Appendix C.5).
this section cite: ['b62', 'b97', 'b74', 'b62']

Section: ALGORITHM AND EVALUATION
In this section, we develop a structural-assumption-free algorithm to recover the underlying causal models from observed data up to distributional equivalence. We name this algorithm as general latentvariable Linear Non-Gaussian causal discovery (glvLiNG). Evaluation results are also provided.
Algorithm. The glvLiNG pipeline consists of three main steps: it first runs OICA on data to estimate a mixing matrix Ã, then constructs a digraph G to realize rank patterns in Ã, and finally, starting from G, traverses the equivalence class using the procedure introduced in Theorem 3. Under the assumptions of access to an oracle OICA and faithfulness (no coincidental low ranks in the mixing matrix beyond those structurally entailed; formally stated in Assumption 1 at Appendix A), glvLiNG is guaranteed to recover the entire class of irreducible models equivalent to the ground-truth model.
Proofs and detailed formulations of the glvLiNG algorithm are deferred to Appendix A for page limit. Here, we briefly highlight the core second step: constructing a digraph to realize the observed ranks.
The main challenge lies in this second step, a rank realization task. While the satisfiability nature of this task may suggest brute-force solutions like integer programming, glvLiNG instead offers a more efficient constraint-based approach. Specifically, it proceeds in two phases. Phase 1 recovers edges from latent variables L to all variables V , which reduces to a bipartite realization problem known in matroid theory. Phase 2 is more delicate: recover edges from observed variables X to V . This may seem combinatorially complex at first glance, since all ranks induced by all subsets of X must be jointly satisfied (Lemma 3). Fortunately, as we have shown in Theorem 2, these global constraints admit a local decomposition, allowing each single X i 's outgoing edges to be recovered independently. To recover these edges, we give an explicit construction (Lemma 10 in Appendix A) based directly on querying ranks in the OICA mixing matrix, with no need for solving complex constraint systems.
Evaluation. We evaluate our approach from five aspects: 1) quantifying the sizes of equivalence classes, 2) assessing glvLiNG's runtime, 3) benchmarking existing methods under oracle inputs, 4) evaluating glvLiNG's performance in simulations, and 5) applying glvLiNG to a real-world dataset.
For 1), we quantify the sizes of equivalence classes, in order to provide an illustrative sense of the uncertainty in latent-variable models. We exhaustively partition digraphs with up to 6 vertices under various latent configurations. For example, there are 1, 027, 080 weakly connected digraphs with 5 vertices, of which 26, 430 are acyclic. When the first 2 vertices are latent, 480, 640 of these digraphs yield irreducible models, which finally form 783 equivalence classes. Full statistics are shown in Table 3.
For 2), we assess the efficiency gain enabled by glvLiNG's constraint-based design. We compare the execution time against a linear programming baseline for constructing digraphs to satisfy ranks of oracle OICA mixing matrices. Results confirm substantial speedup: glvLiNG solves cases with n = 10 vertices in under 5s, while the baseline takes hours beyond n = 5. Full results in Table 4.
For 3), we examine how existing methods behave under structural misspecification by applying them to arbitrary latent-variable models possibly beyond their assumptions. We evaluate LaHiCaSl (Xie et al., 2024) and PO-LiNGAM (Jin et al., 2024), given oracle access to their required tests. Both methods tend to produce overly sparse graphs and misidentify over half of the edges. Full results in Table 5.
For 4), we evaluate glvLiNG with existing methods under finite samples. We simulate data from random irreducible models, varying numbers of observed and latent variables, graph density, and sample size. We observe that glvLiNG performs particularly better than baselines on denser graphs and stays more robust to latent dimensionality, likely due to avoiding model misspecification, while baselines perform better on sparser graphs. Full setup and results are provided in Appendix D.4.
For 5), we apply glvLiNG to a real-world dataset of daily stock returns (Jan 2000-Jun 2005) from 14 major Hong Kong companies spanning banking, real estate, utilities, and commerce. glvLiNG recovers meaningful patterns, such as major banks acting as central causal sources. The two latent variables recovered seem also to admit plausible interpretations. Full results are in Appendix D.5.
this section cite: ['b136', 'b55']

Section: Final remarks.
We conclude with a reflection on the use of OICA in glvLiNG. While one may be concerned about OICA's known inefficiency in practice, we would like to note that the main focus of this work is to characterize distributional equivalence. The glvLiNG algorithm serves more as a proof of concept, showing that such equivalence is indeed recoverable without any structural assumption.
That said, we do see two promising directions for future improvement. 1) For estimation, several existing methods allow partial access to rank information in the mixing matrix without explicitly running OICA. They could be integrated into glvLiNG. 2) For algorithmic efficiency, while glvLiNG already scales well, further pruning is possible. For instance, Theorem 3 implies that ancestral relations among observed variables are identifiable, which may help reduce the search space.
this section cite: []

Section: CONCLUSION AND LIMITATIONS
In this work, we provide a graphical characterization of distributional equivalence for linear non-Gaussian latent-variable models. Based on it, we develop a constraint-based algorithm, glvLiNG, that recovers the underlying model up to equivalence from data without any structural assumptions. Central to our approach is the introduction of edge rank constraints, a new tool in the rank-based picture. One limitation is the use of OICA in glvLiNG, as discussed above. Future directions include developing OICA-free algorithms, and extending new tools to broader settings like linear Gaussian systems.
Independent/dependent sets. Ind(Q) := {Z ⊆ E : r(Z) = |Z|}, Dep(Q) := 2 E \ Ind(Q). (A.2)
Note. Z is independent if and only if the rows Z admit a matching into the columns; dependent sets are those with a matching deficiency.
Bases. bases(Q) := {B ⊆ E : B ∈ Ind(Q), |B| = r(E)}.
(A.3) Note. Bases are the maximal independent sets: all its subsets are independent, and all its proper supersets are dependent. Bases are the maximum-cardinality independent sets (all have size r(E)). Bases uniquely determine the matroid.
this section cite: []

Section: References
Ref_id:b0 Title: Identification of partially observed linear causal models: Graphical conditions for the Non-Gaussian and heterogeneous cases Year: (2021)
Ref_id:b1 Title: Towards characterizing Markov equivalence classes for directed acyclic graphs with latent variables Year: (2005)
Ref_id:b2 Title: Alexandros Grosdos, Roser Homs, and Elina Robeva. Third-order moment varieties of linear Non-Gaussian graphical models Year: (2023)
Ref_id:b3 Title: Structural identifiability of graphical continuous lyapunov models Year: (2025)
Ref_id:b4 Title: Learning linear Bayesian networks with latent variables Year: (2013)
Ref_id:b5 Title: A characterization of Markov equivalence classes for acyclic digraphs Year: (1997)
Ref_id:b6 Title: On the completeness of causal discovery in the presence of latent confounding with tiered background knowledge Year: (2020)
Ref_id:b7 Title: An Affine Representation for Transversal Geometries Year: (1975)
Ref_id:b8 Title: Enumerating equivalence classes of Bayesian networks using ec graphs Year: (2016)
Ref_id:b9 Title: Identification of causal structure with latent variables based on higher order cumulants Year: (2024)
Ref_id:b10 Title: Learning discrete latent variable structures with tensor rank conditions Year: (2024)
Ref_id:b11 Title: Identification of latent confounders via investigating the tensor ranks of the nonlinear observations Year: (2025)
Ref_id:b12 Title: A transformational characterization of equivalent Bayesian network structures Year: (1995)
Ref_id:b13 Title: Optimal structure identification with greedy search Year: (2002-11)
Ref_id:b14 Title: Learning latent tree graphical models Year: (2011)
Ref_id:b15 Title: Greedy equivalence search in the presence of latent confounders Year: (2022)
Ref_id:b16 Title: Establishing Markov equivalence in cyclic directed graphs Year: (2023)
Ref_id:b17 Title: Learning the causal structure of copula models with latent variables Year: (2018)
Ref_id:b18 Title: Independence testing-based approach to causal discovery under measurement error and linear Non-Gaussian models Year: (2022)
Ref_id:b19 Title: Local causal discovery with linear Non-Gaussian cyclic models Year: (2024)
Ref_id:b20 Title: When selection meets intervention: Additional complexities in causal discovery Year: (2025)
Ref_id:b21 Title: Latent variable causal discovery under selection bias Year: ()
Ref_id:b22 Title: A versatile causal discovery framework to allow causally-related hidden variables Year: (2024)
Ref_id:b23 Title: Permutation-based rank test in the presence of discretization and application in causal discovery with mixed data Year: (2025)
Ref_id:b24 Title: Score-based greedy search for structure identification of partially observed linear causal models Year: (2026)
Ref_id:b25 Title: Algebraic problems in structural equation modeling Year: (2018)
Ref_id:b26 Title: Algebraic factor analysis: tetrads, pentads and beyond Year: (2007)
Ref_id:b27 Title: Causal discovery for linear Non-Gaussian models with disjoint cycles Year: (2025)
Ref_id:b28 Title: Identifiability of homoscedastic linear structural equation models using algebraic matroids Year: (2025)
Ref_id:b29 Title: Normalizing flows for conditional independence testing Year: (2024)
Ref_id:b30 Title: Almost optimal intervention sets for causal discovery Year: (2008)
Ref_id:b31 Title: Interventions and causal inference Year: (2007)
Ref_id:b32 Title: On the number of experiments sufficient and in the worst case necessary to identify all causal relations among n variables Year: (2005)
Ref_id:b33 Title: Identifiability, separability, and uniqueness of linear ica models Year: (2004)
Ref_id:b34 Title: Graphs for margins of Bayesian networks Year: (2016)
Ref_id:b35 Title: Margins of discrete Bayesian networks Year: (2018)
Ref_id:b36 Title: Markov properties for graphical models with cycles and latent variables Year: (2017)
Ref_id:b37 Title: The chain graph Markov property Year: (1990)
Ref_id:b38 Title: Graphical models and exponential families Year: (1996)
Ref_id:b39 Title: Binomial determinants, paths, and hook length formulae Year: (1985)
Ref_id:b40 Title: Learning causal structures using regression invariance Year: (2017)
Ref_id:b41 Title: Characterizing distribution equivalence and structure learning for cyclic and acyclic directed graphs Year: (2020)
Ref_id:b42 Title: Identifiability of hierarchical latent attribute models Year: (2023)
Ref_id:b43 Title: Characterization and greedy learning of interventional Markov equivalence classes of directed acyclic graphs Year: (2012)
Ref_id:b44 Title: Jointly interventional and observational data: estimation of interventional Markov equivalence classes of directed acyclic graphs Year: (2015)
Ref_id:b45 Title: Active learning of causal networks with intervention experiments and optimal designs Year: (2008-11)
Ref_id:b46 Title: Estimation of causal effects using linear Non-Gaussian causal models with hidden variables Year: (2008)
Ref_id:b47 Title: Identification and estimation of nonlinear models with misclassification error using instrumental variables: A general solution Year: (2008)
Ref_id:b48 Title: The econometrics of unobservables: Applications of measurement error models in empirical industrial organization and labor economics Year: (2017)
Ref_id:b49 Title: Faster algorithms for Markov equivalence Year: (2020)
Ref_id:b50 Title: Causal discovery from heterogeneous/nonstationary data Year: (2020)
Ref_id:b51 Title: Latent hierarchical causal structure discovery with rank constraints Year: (2022)
Ref_id:b52 Title: Gammoids and transversal matroids Year: (1973)
Ref_id:b53 Title: Discovery of causal models that contain latent variables through Bayesian scoring of independence constraints Year: (2017)
Ref_id:b54 Title: Learning nonparametric latent causal graphs with unknown interventions Year: (2023)
Ref_id:b55 Title: Structural estimation of partially observed linear Non-Gaussian acyclic model: A practical approach with identifiability Year: (2024)
Ref_id:b56 Title: Characteristic imsets for cyclic linear causal models and the chickering ideal Year: (2025)
Ref_id:b57 Title: Learning latent causal graphs via mixture oracles Year: (2021)
Ref_id:b58 Title: A cross-moment approach for causal effect estimation Year: (2023)
Ref_id:b59 Title: Characterization and learning of causal graphs with latent variables from soft interventions Year: (2019)
Ref_id:b60 Title: Graphs and matrices Year: (1931)
Ref_id:b61 Title: Causal clustering for 1-factor measurement models Year: (2016)
Ref_id:b62 Title: Discovering cyclic causal models by independent components analysis Year: (2008)
Ref_id:b63 Title: On the vector representations of induced matroids Year: (1973)
Ref_id:b64 Title: Learning linear Non-Gaussian graphical models with multidirected edges Year: (2021)
Ref_id:b65 Title: Gene regulatory network inference in the presence of selection bias and latent confounders Year: (2025)
Ref_id:b66 Title: Characterization and learning of causal graphs with latent confounders and post-treatment selection from interventional data Year: (2026)
Ref_id:b67 Title: Identifiability and estimation in high-dimensional nonparametric latent structure models Year: (2025-07-04)
Ref_id:b68 Title: RCD: Repetitive causal discovery of linear non-Gaussian acyclic models with latent confounders Year: (2020)
Ref_id:b69 Title: Ancestral causal inference Year: (2016)
Ref_id:b70 Title: Measurement dependence inducing latent causal models Year: (2020)
Ref_id:b71 Title: A transformational characterization of unconditionally equivalent Bayesian networks Year: (2022)
Ref_id:b72 Title: On a class of matroids arising from paths in graphs Year: (1972)
Ref_id:b73 Title: Causal inference and causal explanation with background knowledge Year: (1995)
Ref_id:b74 Title: Graphical Models: Selecting causal and statistical models Year: (1997)
Ref_id:b75 Title: Methods for causal inference from gene perturbation experiments and validation Year: (2016)
Ref_id:b76 Title: Zur allgemeinen kurventheorie Year: (1927)
Ref_id:b77 Title: Constraint-based causal discovery using partial ancestral graphs in the presence of cycles Year: (2020)
Ref_id:b78 Title: Joint causal inference from multiple contexts Year: (2020)
Ref_id:b79 Title: Score-based causal discovery of latent variable causal models Year: (2024)
Ref_id:b80 Title: Distributional equivalence and structure learning for bow-free acyclic path diagrams Year: (2017)
Ref_id:b81 Title: A hybrid causal search algorithm for latent variable models Year: (2016)
Ref_id:b82 Title: Matroid theory Year: (2006)
Ref_id:b83 Title: Causality: models, reasoning and inference Year: (2009)
Ref_id:b84 Title: Identifying independencies in causal graphs with feedback Year: (1996)
Ref_id:b85 Title: Applications of menger's graph theorem Year: (1968)
Ref_id:b86 Title: Interpreting and using CPDAGs with background knowledge Year: (2017)
Ref_id:b87 Title: Overcomplete independent component analysis via sdp Year: (2019)
Ref_id:b88 Title: Ancestral graph Markov models Year: (2002)
Ref_id:b89 Title: Discovering cyclic causal structure Year: (1996)
Ref_id:b90 Title: Nested Markov properties for acyclic directed mixed graphs Year: (2023)
Ref_id:b91 Title: Multi-trek separation in linear structural equation models Year: (2021)
Ref_id:b92 Title: Learning causal cyclic graphs from unknown shift interventions Year: (2015)
Ref_id:b93 Title: A graphical representation of equivalence classes of amp chain graphs Year: (2006)
Ref_id:b94 Title: Learning linear Non-Gaussian causal models in the presence of latent variables Year: (2020)
Ref_id:b95 Title: Goodness-of-fit tests for linear Non-Gaussian structural equation models Year: (2025)
Ref_id:b96 Title: Causal discovery of linear Non-Gaussian causal models with unobserved confounding Year: (2024)
Ref_id:b97 Title: Near-optimal experiment design in linear Non-Gaussian cyclic models Year: (2025)
Ref_id:b98 Title: Statistical causal discovery: LiNGAM approach Year: (2022)
Ref_id:b99 Title: Bayesian estimation of causal direction in acyclic structural equation models with individual-specific confounder variables and Non-Gaussian distributions Year: (2014)
Ref_id:b100 Title: A linear Non-Gaussian acyclic model for causal discovery Year: (2006)
Ref_id:b101 Title: Generalized measurement models Year: (2004)
Ref_id:b102 Title: Learning instrumental variables with structural and Non-Gaussianity assumptions Year: (2017)
Ref_id:b103 Title: Learning measurement models for unobserved variables Year: (2003)
Ref_id:b104 Title: Learning the structure of linear latent variable models Year: (2006)
Ref_id:b105 Title: Building causal graphs from statistical data in the presence of latent variables Year: (1992)
Ref_id:b106 Title: Conditional independence in directed cyclic graphical models representing feedback or mixtures Year: (1994)
Ref_id:b107 Title: An algorithm for fast recovery of sparse causal graphs Year: (1991)
Ref_id:b108 Title: A polynomial time algorithm for determining DAG equivalence in the presence of latent variables and selection bias Year: (1996)
Ref_id:b109 Title: Equivalence of causal models with latent variables Year: (1992)
Ref_id:b110 Title: Causation, prediction, and search Year: (2000)
Ref_id:b111 Title: Calculation of entailed rank constraints in partially non-linear and cyclic models Year: (2013)
Ref_id:b112 Title: Permutation-based causal structure learning with unknown intervention targets Year: (2020)
Ref_id:b113 Title: Causal structure discovery between clusters of nodes induced by latent factors Year: (2022)
Ref_id:b114 Title: Enumeration of labelled chain graphs and labelled essential directed acyclic graphs Year: (2003)
Ref_id:b115 Title: Unpaired multi-domain causal representation learning Year: (2024)
Ref_id:b116 Title: Trek separation for Gaussian graphical models Year: (2010)
Ref_id:b117 Title: Determinants of weighted path matrices. arXiv: Combinatorics Year: (2012)
Ref_id:b118 Title: Parcelingam: A causal ordering method robust against latent confounders Year: (2014)
Ref_id:b119 Title: An introduction to proximal causal inference Year: (2024)
Ref_id:b120 Title: Identifiability of mixtures of product measures Year: (1967)
Ref_id:b121 Title: Generating Markov equivalent maximal ancestral graphs by single edge replacement Year: (2005)
Ref_id:b122 Title:  Year: (2001)
Ref_id:b123 Title: Learning linear Non-Gaussian polytree models Year: (2022)
Ref_id:b124 Title: Parameter identification in linear Non-Gaussian causal models under general confounding Year: (2024)
Ref_id:b125 Title: Causal effect identification in lvLiNGAM from higher-order cumulants Year: (2025)
Ref_id:b126 Title: Towards complete causal explanation with expert knowledge Year: (2024)
Ref_id:b127 Title: Equivalence and synthesis of causal models Year: (1991)
Ref_id:b128 Title: An efficient maximal ancestral graph listing algorithm Year: (2024)
Ref_id:b129 Title: Polynomial-delay mag listing with novel locally complete orientation rules Year: (2025)
Ref_id:b130 Title: Causal discovery with unobserved confounding and Non-Gaussian data Year: (2023)
Ref_id:b131 Title: Permutation-based causal inference algorithms with interventions Year: (2017)
Ref_id:b132 Title: Efficient enumeration of Markov equivalent DAGs Year: (2023)
Ref_id:b133 Title: Conditional independent component analysis for estimating causal structure with latent variables Year: (2026)
Ref_id:b134 Title: Generalized independent noise condition for estimating latent variable causal graphs Year: (2020)
Ref_id:b135 Title: Testability of instrumental variables in linear Non-Gaussian acyclic causal models Year: (2022)
Ref_id:b136 Title: Generalized independent noise condition for estimating causal structure with latent variables Year: (2024)
Ref_id:b137 Title: Double robust conditional independence test for novel biomarkers given established risk factors with survival data Year: (2025)
Ref_id:b138 Title: Characterizing and learning equivalence classes of causal DAGs under interventions Year: (2018)
Ref_id:b139 Title: Causal discovery in linear latent variable models subject to measurement error Year: (2022)
Ref_id:b140 Title: Causal discovery in linear models with unobserved variables and measurement error Year: (2024)
Ref_id:b141 Title: Sigma-maximal ancestral graphs Year: (2025)
Ref_id:b142 Title: Unifying causal representation learning with the invariance principle Year: (2025)
Ref_id:b143 Title: Covariate selection in causal learning under Non-Gaussianity Year: (2024)
Ref_id:b144 Title: Identifiability guarantees for causal disentanglement from soft interventions Year: (2023)
Ref_id:b145 Title: On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias Year: (2008)
Ref_id:b146 Title: Causal reasoning with ancestral graphs Year: (2008)
Ref_id:b147 Title: A transformational characterization of Markov equivalence for directed acyclic graphs with latent variables Year: (2005)
Ref_id:b148 Title: Discovery and visualization of nonstationary causal models Year: (2015)
Ref_id:b149 Title: Causal discovery with linear Non-Gaussian models under measurement error: Structural identifiability results Year: (2018)
Ref_id:b150 Title: Causal representation learning from multiple distributions: A general setting Year: (2024)
