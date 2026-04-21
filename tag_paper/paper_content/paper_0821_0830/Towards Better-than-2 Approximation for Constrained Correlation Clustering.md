Title: Towards Better-than-2 Approximation for Constrained Correlation Clustering
Abstract: In the Correlation Clustering problem, we are given an undirected graph and are tasked with computing a clustering (partition of the nodes) that minimizes the sum of the number of edges across different clusters and the number of nonedges within clusters. In the constrained version of this problem, the goal is to compute a clustering that satisfies additional hard constraints mandating certain pairs to be in the same cluster and certain pairs to be in different clusters. Constrained Correlation Clustering is APX-Hard, and the best known approximation factor is 3 (van Zuylen et al. [SODA '07]). In this work, we show that in order to obtain a better-than-2 approximation, solving the (exponentially large) Constrained Cluster LP would be sufficient 1 .

Section: Introduction
Clustering is a fundamental unsupervised learning task and has many applications in machine learning and data mining. The high-level goal is to partition a set of elements into clusters such that similar elements are in the same cluster and dissimilar elements are in different clusters. Over the years, several definitions of clustering have been considered, each with a different objective.
Correlation Clustering, proposed by Bansal, Blum, and Chawla (Bansal et al., 2004), is one such clustering formulation, which was widely studied since its introduction. The appeal of Correlation Clustering stems from its simple and natural definition and from the absence of requirement to specify the number of clusters as part of the input. The problem has found success in many applications including, but not limited to, automated labeling (Agrawal Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). et al., 2009;Chakrabarti et al., 2008), clustering ensembles (Bonchi et al., 2013), community detection (Chen et al., 2012;Veldt et al., 2018), disambiguation tasks (Kalashnikov et al., 2008), duplicate detection (Arasu et al., 2009) and image segmentation (Kim et al., 2011;Yarkony et al., 2012).
Given a complete unweighted undirected graph (V, E + ⊎ E -), whose edges are labeled either "plus" or "minus", the Correlation Clustering problem is to output a partition (clustering) C = {C 1 , . . . , C k } of the vertex set V . The sets C i of C are called clusters. The objective is to minimize the sum of the number of plus edges across different clusters and the number of minus edges within clusters.
One can view the edges in the input graph as modeling pairwise relationships between entities corresponding to the nodes. Specifically, a plus edge denotes the preference of its endpoints to be clustered together and a minus edge denotes their preference to be separated. The cost of a clustering is then the number of preferences violated by the clustering and the goal of Correlation Clustering translates to finding a clustering that satisfies the preferences of the maximum number of pairs. The real-world applications that motivate clustering as above often necessitate incorporating additional hard constraints that result from prior knowledge. In such a semi-supervised setting, it is imperative to study the more general problem of constrained clustering. There are several variants of constrained clustering including, but not limited to constrained k-means (Wagstaff et al., 2001), spectral clustering with constraints (Wang & Davidson, 2010), constrained ranking and clustering (van Zuylen & Williamson, 2009), and constrained fuzzy clustering (Pham, 2002). Constrained clustering is particularly valuable as it improves the quality of clusters, aligns outcomes with domain-specific knowledge, and reduces sensitivity to noisy data.
In this work, we study Constrained Correlation Clustering, a variant of Correlation Clustering capturing the idea of critical pairs of nodes, which was first studied by van Zuylen and Williamson (van Zuylen & Williamson, 2009). Con-strained Correlation Clustering introduces hard constraints in addition to the pairwise preferences, where one could have must-link (requiring nodes to be in the same cluster) or cannot-link (requiring nodes to be separate) constraints. A clustering is valid if it satisfies all hard constraints, and the goal is to find a valid clustering of minimum cost.
Constrained Correlation Clustering with must-link and cannot-link constraints is well-motivated in both theory and practice. For instance, it has been used to cluster news articles about the same event across different languages (Gael & Zhu, 2007). The hard constraints here ensure that news articles about different events from the same language do not end up in the same cluster.
this section cite: ['b5', 'b17', 'b10', 'b20', 'b35', 'b30', 'b3', 'b31', 'b38', 'b36', 'b37', 'b34', 'b33', 'b34', 'b29']

Section: Previous Results
Correlation Clustering was defined by Bansal, Blum, and Chawla (Bansal et al., 2004), who, in addition to proving its NP-Hardness, provided a deterministic constantfactor approximation, the constant being larger than 15,000. Subsequent results improved the approximation guarantee by relying on LP-rounding techniques: Charikar, Guruswami and Wirth showed that the problem is APX-Hard and also gave a deterministic 4-approximation (Charikar et al., 2005), Ailon, Charikar and Newman gave a randomized 2.5-approximation (Ailon et al., 2008), and Chawla, Makarychev, Schramm and Yaroslavtsev gave a deterministic 2.06-approximation (Chawla et al., 2015). The last result is nearly optimal among algorithms that round the natural LP formulation, since its integrality gap is at least 2. In a breakthrough result by Cohen-Addad, Lee and Newman, a (1.994 + ϵ)-approximation using the Sherali-Adams relaxation broke the 2 barrier (Cohen-Addad et al., 2022). The approximation guarantee was later improved to 1.73 + ϵ (Cohen-Addad et al., 2023) by Cohen-Addad, Lee, Li and Newman, and even to 1.437 by (Cao et al., 2024a). There is also a combinatorial 1.847-approximation by (Cohen-Addad et al., 2024b) for the problem. The current-state of the art is an adaptation of the 1.437 approximation that runs in linear (Cao et al., 2025b), and even sublinear (Cao et al., 2025a) time.
Correlation Clustering has also been studied in different settings such as dynamic algorithms (Cohen-Addad et al., 2024a), parameterized algorithms (Fomin et al., 2014), sublinear and streaming algorithms (Assadi & Wang, 2022;Cambus et al., 2024;Behnezhad et al., 2022;2023;Cambus et al., 2024;Makarychev & Chakrabarty, 2023), massively parallel computation (MPC) algorithms (Cohen-Addad et al., 2021;Cao et al., 2024b), and differentially private algorithms (Bun et al., 2021).
Regarding Constrained Correlation Clustering, the state-ofthe-art approximation is given in the work of van Zuylen and Williamson (van Zuylen & Williamson, 2009) who de-signed a deterministic 3-approximation. We note here that algorithms with better-than-3 approximation for Correlation Clustering existed for a long time after the 3-approximation by (van Zuylen & Williamson, 2009), but they do not seem to extend to the case of Constrained Correlation Clustering. Due to the significance of the problem, a follow-up work by Fischer, Klausen, Kipouridis and Thorup (Fischer et al., 2024) focused on faster algorithms for Constrained Correlation Clustering, even at the expense of worse approximation.
It is worth noting that it is not possible, in general, to handle the must-link constraints by simply merging the nodes in connected components induced by these constraints into supernodes, as such an aggregation results in a weighted instance outside the problem.
this section cite: ['b5', 'b18', 'b2', 'b19', 'b22', 'b28', 'b4', 'b12', 'b7', 'b12', 'b32', 'b21', 'b11', 'b34', 'b34']

Section: Related Work
Closely related to the problems considered by us are hierarchical clustering problems under similar constrained settings. In particular, for each pair {u, v} we are given an upper and a lower bound regarding the distance of u and v in the output (Farach, Kannan, and Warnow (Farach et al., 1995)). Ailon and Charikar, in (Ailon & Charikar, 2011), make progress on constrained hierarchical clustering, but optimal algorithms are still elusive. We note that Constrained Correlation Clustering is a special case of constrained hierarchical clustering (where the hierarchy is trivial).
this section cite: ['b26', 'b1']

Section: Our Contribution and Techniques
Our main result is a proof that, to achieve a better-than-2 approximation for Constrained Correlation Clustering, it suffices to design an efficient solution to the Constrained Cluster LP (Figure 1). Theorem 1.1 (Informal). If there exists a polynomial-time algorithm to solve the Constrained Cluster LP, then there exists a polynomial time algorithm for Constrained Correlation Clustering whose approximation factor is 1.92.
Once again, we note that we do not claim a solution to the Constrained Cluster LP. We consider it plausible because the corresponding unconstrained Cluster LP has been solved in polynomial (Cao et al., 2024a), linear (Cao et al., 2025b), and even sublinear time (Cao et al., 2025a).
There are two main techniques for (unconstrained) Correlation Clustering to achieve a better-than-2 approximation. One (Cao et al., 2024a) is based on the Cluster LP, an LP stronger than the natural onefoot_1 , while the other (Cohen-Addad et al., 2024b) is based on a local search technique.
The problem in extending the LP based approach is that, even given a valid solution to the Constrained Cluster LP, its rounding involves creating clusters based on independent sampling of nodes (based on some probability distributions dictated by the LP solution). However, when hard constraints are enforced, we can no longer use independent sampling since, for example, placing a node u in a cluster implies that all nodes with a cannot-link constraint with u cannot be in the same cluster.
The local-search approach, on the other hand, is based on a simple principle: if we have a clustering whose cost is more than 2 times the optimal cost, there exists a cluster in the optimal solution that we can "force" into our clusteringfoot_2 and get a clustering with smaller cost. The proof is based on a simple counting argument, and the technique would directly lend itself to a 2 approximation, if we could spend exponential time to find such a cluster. The novelty of (Cohen-Addad et al., 2024b) is that they manage to (nearly) simulate this in polynomial time, and their approximation is 2 + ε. As one would expect, achieving this is the most technically challenging part of their paper. It is possible that these techniques could be extended to the Constrained Correlation Clustering setting, but even if true, the resulting algorithm and analysis would be highly non-trivial.
Instead, in this work, we propose a simple approach for Constrained Correlation Clustering that combines the strengths of both aforementioned techniques.
Perhaps the most critical observation in our paper is that even though we do not have access to the optimal clustering, an optimal fractional clustering (that is, a solution to the LP) would suffice to guide the local search, and thus bypass the need for the heavy machinery introduced in (Cohen-Addad et al., 2024b). In particular, one could embed the hard constraints in the LP, and guarantee that the output of the local search is a valid clustering.
Despite the success of using the optimal fractional clustering to get a 2-approximation, it does not come without its own shortcomings. In particular, the problem occurs when trying to break the 2 barrier. In (Cohen-Addad et al., 2024b), they show that normally the output of the local search is already a better-than-2 approximation, except for some very specific cases. Then, they run a second local search, which is incentivised to give a very different solution than the first (thus, intuitively, avoiding the specific bad cases). To argue that one of the two solutions achieves a better-than-2 approximation, they show that if both approximations were at least 2, then it would be possible to "mix" 3 different clusterings (the two solutions and the optimal one) in order to obtain a clustering that is better than optimal (a contradiction).
In our case however, it is not even clear what mixing two clusterings and a fractional clustering would mean. A straightforward approach would be to first round the fractional clustering, obtain a clustering C that is close to optimal, and then mix the two solutions and C. The problem with this approach is that it would normally only show that we can get a better clustering than C, not better than the optimal, which is no longer a contradiction.
Still, we show that there exists one particular way to round the fractional clustering, so that the mixing is guaranteed to be below the optimal, not just C. In fact, this particular rounding simply samples clusters with probabilities proportional to the values of the LP solution. Its properties were analysed already in (Cao et al., 2024a), as one part of their overall algorithm.
Finally, we note that in our case we can algorithmically compute and use C, given access to the fractional optimal (unlike in (Cohen-Addad et al., 2024b) where they do not have access to the optimal clustering, and therefore C is only part of the analysis).
this section cite: []

Section: Organization
The rest of the paper has the following structure. In Section 2, we formally define the problem, and give some basic notation. In Section 3, we describe the LP that models Constrained Correlation Clustering. Finally, in Section 4 we give a full analysis for our main algorithm and how it achieves its approximation factor of (1.92 + ε).
this section cite: []

Section: Preliminaries
The graphs (V, E + ⊎ E -) we consider in this paper are complete, unweighted, undirected, and every edge is labeled plus or minus (it belongs to one of E + or E -, respectively). We typically set n = |V |. Let A 2 denote the set of all size-2 subsets of a set A. We often abbreviate the (unordered) set { u, v } by uv. Before introducing our problem, we first define the cost of a clustering: Definition 2.1. Let E + (C) and E -(C) be the sets of plus and minus edges that are not satisfied in a clustering C.
That is E + (C) = { uv ∈ E + | ∄C ∈ C with {u, v} ⊆ C } and E -(C) = { uv ∈ E -| ∃C ∈ C with {u, v} ⊆ C }. The cost of C is defined as cost(C) = |E + (C)| + |E -(C)|.
We now formally define Constrained Correlation Clustering.
Definition 2.2 (Constrained Correlation Clustering). Given an instance (V, E + ⊎ E -, F, H), where (V, E + ⊎ E -) is an edge-labeled graph, F ⊆ V
2 is a set of friendly pairs, and H ⊆ V 2 is a set of hostile pairs, compute a minimum cost clustering C = { C 1 , . . . , C k } of V such that no pair uv ∈ F has u, v in different clusters and no pair uv ∈ H has u, v in the same cluster.
this section cite: []

Section: LP with constraints
In this section, we describe the linear program used to model the problem. We will use its solution to guide the local searches for the algorithm later.
Given an instance (V, E + ⊎ E -, F, H) of Constrained Correlation Clustering, we can define the following linear program, which we call the Constrained Cluster LP. It has two types of variables: x uv for every uv ∈ V 2 and z C for every C ⊆ V . The variable x uv describes the desirability of nodes u and v to be in different clusters (one can view it as a distance between the two nodes) and the variable z C captures the probability of cluster C to be in the final clustering.
min uv∈E + x uv + uv∈E - (1 -x uv ) s.t. S∋u z S = 1 ∀u ∈ V, S⊇{u,v} z S = 1 -x uv ∀uv ∈ V 2 , z S ≥ 0 ∀S ⊆ V, S ̸ = ∅, x uv = 0 ∀uv ∈ F, x uv = 1 ∀uv ∈ H
The S∋u z S = 1 constraints ensure that all vertices belong to some cluster in the final clustering. The S⊇{u,v} z S = 1 -x uv constraints state that the closeness of two nodes is proportional to the total weight of clusters containing both of them. Finally, the x uv = 0 and x uv = 1 constraints model the hard constraints of the instance. We note that without the last two types of constraints, the above LP is identical to the Cluster LP for (unconstrained) Correlation Clustering, solved in (Cao et al., 2024a;2025b;a). Assumption 3.1. Let OPT be the value of the optimal solution for a Constrained Correlation Clustering instance (V, E + ⊎ E -, F, H). We assume that for any constants c > 0, ε > 0 there exists an algorithm that runs in time n poly (1/ε) and with probability 1 -n -c returns a solution to the Constrained Cluster LP such that:
• its value is at most (1 + ε)OPT,
• it has at most poly(n, 1 ε ) subsets S with z S > 0, • for every uv ∈ F there is no subset S with z S > 0 and |S ∩ { u, v }| = 1,
• for every uv ∈ H there is no subset S with z S > 0 and S ⊇ { u, v }.
From this point on, we always assume that we have access to a particular solution to the LP, with the properties described in Assumption 3.1.
this section cite: []

Section: Local Search Algorithm
In this section, we describe our algorithm and give a full proof for its approximation factor.
At a high level, the algorithm solves the Constrained Cluster LP (Assumption 3.1) whose solution we view as a (fractional) optimal clusteringfoot_3 ; we then perform a simple local search procedure (guided by this optimal) that gives a 2-approximation. In fact, this solution already gives a betterthan-2 approximation, unless some really specific conditions occur. It turns out that it suffices to run local search once again, but this time we discourage the new clustering from looking similar to the previous one (by adding a penalty in the objective function, when the two clusterings are similar).
In (Cohen-Addad et al., 2024b), they prove that one of the two clusterings gives a better-than-2 approximation as follows: assuming this is not true, they show how to obtain a new clustering that has approximation better than the optimal (a contradiction). To obtain this clustering, they "mix" the optimal with the two local search solutions. We perform the same steps, but in our case we can actually perform these steps as part of the algorithm, not just in the analysis, exactly because in our case we assume we have obtained a (fractional) optimal solution.
this section cite: []

Section: Local Search Algorithm
We first define an operation (which we call local move) that transforms a clustering C. Intuitively, the operation takes a new cluster C and swaps it in the clustering; the resulting clustering is defined as C C = {S \ C : S ∈ C} ∪ {C}. We call a local move legal, if the cluster C that we introduce to the clustering C does not violate any constraints in F or H. In other words, C does not contain both endpoints of a hostile edge nor does it contain only one endpoint of a friendly edge.
Lemma 4.1. Starting from a feasible clustering C, applying any number of legal local moves results in a clustering that does not violate any constraints in F or H.
Proof. We consider a single legal local move, as by induction on the number of legal local moves we directly get the lemma.
For a friendly pair to be violated, there must exist a cluster containing exactly one of its endpoints. By the definition of a legal move, this cannot happen in the introduced cluster C (which either contains both or none of the endpoints). Additionally, this cannot happen in other clusters as well; if it did, then the other endpoint would be in C, which we already proved cannot happen. Similarly, for a hostile pair to be violated, the two endpoints must end up in the same cluster. The only cluster that can contain nodes that were in different clusters in C is cluster C, which by definition does not contain both endpoints of a hostile pairs.
Observation 4.2. We note that it is direct to algorithmically obtain a feasible clustering C, by creating a cluster for each connected component of (V, F ). Informally, this gives minimal clusters satisfying F . If any hard constraint in H was not satisfied, then the instance itself would be infeasible (two nodes are forced to be together, by F , and to be separated, by H).
We now define the local search procedure. Let M be the collection of clusters C with z C > 0 in the LP solution.
By Assumption 3.1, there are poly(n, 1 ε ) clusters in M. Furthermore, they do not violate the hard constraints, and can thus be used for legal local moves.
this section cite: []

Section: Analysis of Local Search: 2-approximation
Recall that we view the LP solution as a fractional clustering.
Here, we introduce notation to separately capture the cost incurred by plus edges that go across clusters and the cost incurred by minus edges within clusters.
Definition 4.3. Let LP + and LP -denote the costs of the plus and minus edges incurred by the fractional clustering given by the LP solution. That is LP + = uv∈E + x uv and
LP -= uv∈E -(1 -x uv ).
We slightly abuse notation and use LP to refer both to the linear program, and the cost of its acquired solution (LP = LP + + LP -).
We say a clustering C is α-approximate if cost(C) ≤ αLP.
We first show that L is 2-approximate. In fact, we prove something even stronger, that we later use to get an algorithm that is better than 2-approximate.
Lemma 4.4. Let L be as in Algorithm 1. Then,
cost(L) ≤ 2LP -LP -- uv∈E + (L)∪E -(L)
x uv ,
where LP denotes the value of the fractional LP solution.
Proof. Let C be a cluster such that z C > 0 in the LP solution returned (C ∈ M). Recall that, L C denotes the clustering obtained by applying a local move with cluster C on a clustering L. This local move can affect an edge (i.e. make it satisfied or unsatisfied) only if the edge has at least one endpoint in cluster C (if not, then the endpoints of the edge do not change cluster, which means that they remain as they are -satisfied or unsatisfied). We say that these edges are covered by C. By the local optimality of L, we have cost(L) ≤ cost(L C ). Informally, this means that after swapping cluster C in, more edges covered by C are unsatisfied than before. Let C + denote the set of plus-edges that have exactly one endpoint in C. Similarly, let C -denote the set of minus edges that have both endpoints in C. The set C + ∪ C -is the set of edges unsatisfied by the cluster C. Next, let E + C (L) ⊆ E + (L) be the set of plus edges uv ∈ E + (L) that are covered by C, i.e.,
|{u, v} ∩ C| ≥ 1. Let E - C (L) ⊆ E -(L) be the set of minus edges uv ∈ E -(L) such that |{u, v} ∩ C| ≥ 1.
From the above discussion, for each cluster C with z C > 0, we have,
|C + | + |C -| ≥ |E + C (L)| + |E - C (L)|. This further implies that C z C • (|C + | + |C -|) ≥ C z C • (|E + C (L)| + |E - C (L)|).
We simplify the left-hand side of the above inequality by separately accounting for the contributions of plus and minus edges to the summation. The contribution of a plusedge uv to the left-hand side of the above sum is simply C:|{u,v}∩C|=1 z C . Recall, from the LP in Figure 1, that C:u∈C z C = C:v∈C z C = 1 and C:{u,v}⊆C z C = 1 -x uv . From this, it follows that C:|{u,v}∩C|=1 z C = 2x uv . Similarly, one can see that the contribution of a minusedge uv to the left-hand side of the sum is 1-x uv . Summing over the plus and minus edges separately,
C z C • (|C + | + |C -|) = 2LP + + LP -= 2LP -LP -.
We similarly transform the right-hand side
C z C • (|E + C (L)| + |E - C (L)|) of the above inequality. Each plus edge uv that contributes to C z C • (|E + C (L)| + |E - C (L)|) belongs to E + (L). A plus edge in E + (L)
contributes an amount of z C to the sum if either u or v is contained in C. Thus, its total contribution is C:|C∩{u,v}|≥1 z C , which is equal to 1 + x uv . Using a similar argument, the contribution of a minus edge uv in E -(L) to the sum can also be seen to be 1 + x uv . Thus, the summation simplifies to x uv .
Substituting these rewritten terms in the inequality, we get the desired statement.
The last lemma directly gives an approximation ratio of 2, which we reduce further on.
Corollary 4.5. Let L be as in Algorithm 1. Then,
cost(L) ≤ 2LP -LP -- uv∈E + (L) x uv cost(L) ≤ 2LP -|E -(L)| - uv∈E + (L) x uv
Proof. The first inequality is just Lemma 4.4 with the term -uv∈E -(L) x uv dropped. The second inequality starts from Lemma 4.4 and uses the fact that:
LP -+ uv∈E -(L) x uv ≥ uv∈E -(L) (1 -x uv + x uv )
where the right-hand side is equal to |E -(L)|.
this section cite: []

Section: Local Search with Penalties
Let δ = 2 25 . For the rest of the proof, it is instructive to think about δ as a very small constant.
Directly from Corollary 4.5, we observe that either L is already better than 2-approximate, or some very particular conditions happen:
• Regarding minus-edges, the cost of both L and the LP is negligible.
• Regarding plus-edges paid by L, the corresponding cost of the LP is negligible.
We formalize this intuition in the following corollary.
Corollary 4.6. If L is not (2 -δ)-approximate, then:
LP -+ uv∈E + (L)
x uv < δLP,
|E -(L)| + uv∈E + (L)
x uv < δLP Proof. Immediately from Corollary 4.5.
Following from the above, if L is not (2 -δ)-approximate, then its cost is dominated by the plus edges. The idea here is that we try to run a local search that produces a clustering that looks very different from L (and thus ideally does not satisfy the aforementioned conditions). To do that, we simply penalize the plus edges paid by L in the cost function (Algorithm 2). We note that this part is exactly as in (Cohen-Addad et al., 2024b).
Definition 4.7. For a clustering C the new cost function is:
cost L (C) = cost(C) + |E + (C) ∩ E + (L)|
Algorithm 2: Local Search with penalties
1 cost L (L ′ ) 2 return cost(L ′ ) + |E + (L ′ ) ∩ E + (L)| 3 LOCALSEARCH-WITH-PENALTY() 4 L ← LOCALSEARCH() 5 L ′ ← arbitrary feasible clustering 6 while ∃C ∈ M such that cost L (L ′ C ) < cost L (L ′ ) do 7 L ′ ← L ′ C 8 return L ′
Working exactly as for Corollary 4.6, but using cost L instead of cost, we get similar conditions for when L ′ is not better than 2-approximate: Lemma 4.8. If L ′ is not (2 -δ)-approximate, then:
|E -(L ′ )|+ uv∈E + (L ′ ) x uv + |E + (L ′ ) ∩ E + (L)| < δLP + 2 uv∈E + (L) x uv Proof. Proved in Appendix A.
Directly by Corollary 4.6 and Lemma 4.8 we note that if neither L nor L ′ are (2-δ)-approximate, then the following conditions occur:
• Regarding minus edges, the cost of LP, the cost of L, and the cost of L ′ are all negligible.
• Regarding plus edges, not too many of them can contribute to both L and L ′ , and, moreover, the ones that contribute to L or L ′ cannot contribute much to the cost of the LP.
The following corollary formalizes the above.
Corollary 4.9. If neither L and L ′ are (2 -δ)-approximate, then:
LP -+ |E -(L)| + |E -(L ′ )| + uv∈E + (L) x uv + uv∈E + (L ′ ) x uv + |E + (L) ∩ E + (L ′ )| < 4δLP.
Proof. Follows from the inequalities of Corollary 4.6 and Lemma 4.8.
this section cite: []

Section: Final Algorithm and Analysis
The structure of the rest of the proof in (Cohen-Addad et al., 2024b) is as follows: assuming that none among L, L ′ are (2 -δ)-approximate, the authors construct a clustering that is better than 1-approximate (a contradiction). To construct this clustering, they propose a way to "mix" (in the analysis) L, L ′ , and the optimal clustering. For this to work in our case, we are required to use the fractional optimal clustering. This is so because the guarantees for our two clusterings L, L ′ are with respect to the fractional optimal instead of the actual optimal, as was done in the paper that introduced the local search approach for Correlation Clustering (Cohen-Addad et al., 2024b). However, the mixing procedure cannot be straightforwardly extended to fractional clusterings. Instead, we first round the fractional clustering C * to an integral one, whose properties are crucially very close to that of C * .
This rounding (see (Cao et al., 2024a)) gives a 2approximate clustering that samples clusters from M. Lemma 4.10 (Generalization of Lemma 6 of (Cao et al., 2024a)). For every uv ∈ V 2 the probability that u, v are separated in I returned by Algorithm 3 is 2xuv 1+xuv .
Proof. Directly from the algorithm, the probability that u, v are separated is equal to the probability that we select a cluster containing only one of them, conditioned on the fact that we select a cluster containing at least one of them. This
is |S∩{ u,v }|=1 z S |S∩{ u,v }|≥1 z S = S∋u,S̸ ∋v z S + S∋v,S̸ ∋u z S S∋v z S + S∋u,S̸ ∋v z S . By the LP in Figure 1 we have S∋v z S = 1 and S∋u,S̸ ∋v z S = S∋u z S -S⊇{u,v} z S = 1 -(1 -x uv ) = x uv , which proves the claim. Corollary 4.11. It holds that E |E -(I)|+|E + (I) ∩ E + (L)| + |E + (I) ∩ E + (L ′ )| ≤ LP -+2 • uv∈E + (L) x uv + 2 • uv∈E + (L ′ ) x uv
where the expectation is over the randomness of Algorithm 3.
Proof. Notice that for a minus edge uv, LP pays 1 -x uv , while the expected cost of I is 1
-2xuv 1+xuv = 1 1+xuv (1 - x uv ) ≤ 1 -x uv .
Similarly, for the other two inequalities: for a plus edge uv the cost of LP solution is x uv while the expected cost of I is 2xuv 1+xuv ≤ 2x uv . Corollary 4.12. For any constants c > 0, ε > 0, we can run Algorithm 3 for O(log n) times, to get a clustering I such that
|E -(I)|+|E + (I) ∩ E + (L)| + |E + (I) ∩ E + (L ′ )| ≤ (1 + ε)(LP -+2 • uv∈E + (L) x uv + 2 • uv∈E + (L ′ ) x uv )
with probability at least 1 -n -c . Proof. By Markov Inequality and Corollary 4.11 we have probability at most 1/(1
+ ε) = 1 -ε 1+ε < e -ε 1+ε
to get a clustering I that does not satisfy the claim on a particular execution of Algorithm 3. Therefore the probability of failure in
1+ε ε c ln n repetitions is ≤ e -c ln n = n -c .
Finally, we show that I is a clustering satisfying the hard constraints.
Lemma 4.13. Clustering I returned by Algorithm 3 satisfies all hard constraints.
Proof. By Assumption 3.1, if an endpoint of a friendly pair uv ∈ F is contained in a cluster C with z C > 0, then the other endpoint will also be contained in that cluster. Thus, a sampled cluster will always have both endpoints of a friendly pair. This means that endpoints of friendly pairs are removed from S simultaneously. Therefore, when introducing C ∩ S, it either contains both u and v or neither of them. Analogously, by Assumption 3.1 endpoints of hostile pairs are never in the same cluster C with z C > 0. Hence, they cannot end up in the same cluster in clustering I.
We are now ready to show that if Algorithms 1 and 2 return clusterings that are not (2 -δ)-approximate, then we can create a new clustering that is 24δ-approximate, which is a better-than-2 approximate clustering for our setting of δ.
To do that, we first observe that if neither L nor L ′ are (2 -δ)-approximate, then the following conditions occur:
• Regarding minus edges, the cost of I, the cost of L, and the cost of L ′ are all negligible.
• Regarding plus edges, most of them only contribute to the cost of at most one among L, L ′ , I.
This is similar to Corollary 4.9, and uses the fact that by Corollary 4.12 clustering I behaves very similar to the LP.
Corollary 4.14. For any constants c > 0, ε > 0, if neither of L and L ′ are (2 -δ)-approximate, and we obtain I by using Corollary 4.12 then with probability at least 1 -n -c :
|E -(I)| + |E -(L)| + |E -(L ′ )|+ |E + (I) ∩ E + (L)| + |E + (I) ∩ E + (L ′ )| + |E + (L) ∩ E + (L ′ )| < (8 + ε)δLP
Proof. Beginning from the left hand side and using Corollary 4.12 we get that with probability at least 1 -n -c :
|E -(I)| + |E -(L)| + |E -(L ′ )|+ |E + (I) ∩ E + (L)| + |E + (I) ∩ E + (L ′ )| + |E + (L) ∩ E + (L ′ )| ≤ |E -(L)| + |E -(L ′ )| + |E + (L) ∩ E + (L ′ )|+ (1 + ε)(LP -+ 2 • uv∈E + (L) x uv + 2 • uv∈E + (L ′ ) x uv )
Now, we further manipulate the right hand side, by factoring out the constants:
|E -(L)| + |E -(L ′ )| + |E + (L) ∩ E + (L ′ )|+ (1 + ε)(LP -+ 2 • uv∈E + (L) x uv + 2 • uv∈E + (L ′ ) x uv ) ≤ 2(1 + ε)(LP -+ uv∈E + (L) x uv + uv∈E + (L ′ ) x uv + |E -(L)| + |E -(L ′ )| + |E + (L) ∩ E + (L ′ )|)
By Corollary 4.9 this is less than (8 + 8ε)δLP, which proves the claim by scaling ε.
We will now use clusterings L, L ′ and I to create a new clustering P such that:
• If P pays for a minus edge, then at least one of L, L ′ and I pays for this edge.
• If P pays for a plus edge, then at least two of L, L ′ and I pays for this edge.
Therefore, by Corollary 4.14, the total cost of P is small.
We now describe how to construct P (Algorithm 4). First, we assign to each node u a label (X, Y, Z), such that u belongs to clusters X ∈ L, Y ∈ L ′ , and Z ∈ I.
Definition 4.15. An equivalence class Q XY Z is the set of all nodes that have label equal to (X, Y, Z).
The construction of the new clustering P is shown in Algorithm 4:
Algorithm 4: Pivoting Procedure 1 PIVOT(L, L ′ ) 2 Obtain I by Corollary 4.12 3 Assign labels (X, Y, Z) to nodes u ∈ V 4 V ′ ← V , P ← ∅ 5 while V ′ ̸ = ∅ do 6
S ← an arbitrary equivalence class Q XY Z with all its nodes in V ′ and maximum cardinality 7 C ← S ∪ {u ∈ V ′ : the label of u differs in exactly one label variable from (X, Y, Z)} Proof. Notice that directly from the algorithm, if two nodes are in the same equivalence class, then they end up in the same cluster of P; therefore P does not pay for plus edges with endpoints in the same equivalence class (that is, endpoints that do not differ in any label variable). Furthermore, if u, v are in the same cluster in P, it is because they share at least two label variables with some equivalence class Q XY Z . Therefore, by pigeonhole principle, u, v share at least one label variable, meaning that P does not pay for minus edges with their endpoints differing in all label variables.
We conclude that there are 3 categories of edges that we contribute towards the cost of P:
1. minus edges whose endpoints differ in at most two label variables.
2. plus edges whose endpoints differ in exactly one label variable.
3. plus edges whose endpoints differ in at least two label variables.
In Case 1, minus edges are bounded by Corollary 4.14, as at least one of the clusterings L, L ′ and I pay for them (at least one variable in the labels of their endpoints is the same, meaning that there is a cluster in one of L, L ′ or I that contains both). Similarly, in Case 3, plus edges are also bounded by Corollary 4.14, as at least two of the clusterings L, L ′ and I pay for them (at least two variables in the labels of their endpoints are different, therefore they are in different clusters in at least two of the clusterings).
For every edge uv in Case 2, we show that it can be charged to some edge of the previous cases, and each edge from the previous cases is charged by at most two edges of Case 2. This results in cost 3 • (8 + ε)δLP , by Corollary 4.14.
Consider an iteration of Algorithm 4 that creates a cluster C in P by selecting an equivalence class S = Q XY Z . For a plus edge uv in Case 2 that we pay for, it must be that one of its endpoints differs in one label variable from (X, Y, Z) (therefore it is in C), and the other differs in two (so that it is not in C, because we pay for uv, but still only differs in one label variable from u). W.l.o.g. assume v ∈ Q XY ′ Z ′ and u ∈ Q XY ′ Z or u ∈ Q XY Z ′ , meaning that v can be associated with at most |Q XY ′ Z |+|Q XY Z ′ | edges in Case 2 that we pay for.
By maximality of S we have Q XY ′ Z ≤ S, and therefore we can find an injective function f : Q XY ′ Z → S. If { v, f (u) } is a plus edge, then it is a Case 3 edge and if { v, f (u) } is a minus edge, then it is a Case 1 edge. Similarly for u ∈ Q XY Z ′ , which proves our claim.
Lemma 4.17. P does not violate any hard constraints.
Proof. A friendly pair uv will be in the same cluster in all 3 clusterings L, L ′ , I (as they are feasible clusterings), thus u and v will be in the same equivalence class Q XY Z . By the construction of P, all nodes of an equivalence class end up in the same cluster. Similarly, a hostile pair uv will have its endpoints u and v in different clusters in all 3 clusterings L, L ′ , I, which means that their labels will differ in all three variables. Again, by construction, no two nodes that differ in all the label variables end up in the same group in P.
Our final algorithm returns the best out of all clusterings.
this section cite: []

Section: Concluding Remarks
In this paper, we present a better-than-2 approximation algorithm for Constrained Correlation Clustering, conditioned on an efficient solution to the Constrained Cluster LP. We obtain our algorithm by combining the assumed Cluster LP solution with a local search technique. Our algorithm is conceptually simpler to analyze and gives a better-than-2 approximation for the special case of Correlation Clustering.
The main open direction is to determine whether there exists an efficient solution for the Constrained Cluster LP. It would also be interesting to explore whether our approach of combining the LP and local search has broader applications. Another open direction is with respect to the inapproximability of Constrained Correlation Clustering. Finally, it is intriguing to know whether inapproximability results stronger than the ones directly implied by those for Correlation Clustering exist for the constrained setting.
this section cite: []

Section: A. Analysis of Local Search with Penalty
In this section, we present the analysis of Algorithm 2, which very closely follows the analysis of Algorithm 1.
Lemma 4.8. If L ′ is not (2 -δ)-approximate, then:
|E -(L ′ )|+ uv∈E + (L ′ ) x uv + |E + (L ′ ) ∩ E + (L)| < δLP + 2 uv∈E + (L) x uv
Proof. Let C be a cluster such that z C > 0 in the LP solution returned (C ∈ M). Recall that, L ′ C denotes the clustering obtained by applying a local move with cluster C on a clustering L ′ . This local move can affect an edge (i.e. make it satisfied or unsatisfied) only if the edge has at least one endpoint in cluster C (if not, then the endpoints of the edge do not change cluster, which means that they remain as they are -satisfied or unsatisfied). We say that these edges are covered by C. By the local optimality of L ′ , we have cost ′ (L ′ ) ≤ cost ′ (L ′ C ). This means that after swapping cluster C in, more edges covered by C are unsatisfied than before. Let C + denote the set of plus-edges that have exactly one endpoint in C. Similarly, let C -denote the set of minus edges that have both endpoints in C. The set C + ∪ C -is the set of edges unsatisfied by the cluster C. Next, let E + C (L ′ ) ⊆ E + (L ′ ) be the set of plus edges uv ∈ E + (L ′ ) that are covered by C, i.e., |{u, v} ∩ C| ≥ 1.
Let E - C (L ′ ) ⊆ E -(L ′ ) be the set of minus edges uv ∈ E -(L ′ ) such that |{u, v} ∩ C| ≥ 1.
From the above discussion, for each cluster C with z C > 0, we have,
|C + | + |C + ∩ E + (L)| + |C -| ≥ |E + C (L ′ )| + |E + C (L ′ ) ∩ E + (L)| + |E - C (L ′ )|. This further implies that C z C • (|C + | + |C -|) + C z C • |C + ∩ E + (L)| ≥ C z C • (|E + C (L ′ )| + |E - C (L ′ )|) + C z C • |E + C (L ′ ) ∩ E + (L)|
We simplify the first term of every side of the inequality in the same manner as in Lemma 4.4, which gives us:
2LP -LP -+ C z C • |C + ∩ E + (L)| ≥ cost(L ′ ) + uv∈E + (L ′ )∪E -(L ′ ) x uv + C z C • |E + C (L ′ ) ∩ E + (L)|
We simplify the term C z C • |C + ∩ E + (L)| of the left-hand side by accounting the contributions of plus edges to the summation.
A plus edge in E + (L) contributes an amount of z C to the sum if either u or v is contained in C. Thus, its total contribution is C:|C∩{u,v}|≥1 z C , which is equal to 1 + x uv .
For an edge to contribute to the term it must be in E + (L) and only have one of its endpoints in cluster C. This means that every edge in E + (L) contributes exactly C:|{u,v}∩C|=1 z C = 2x uv . Thus, we get:
C z C • |C + ∩ E + (L)| = 2 uv∈E + (L) x uv
We similarly transform the term C z C • |E + C (L ′ ) ∩ E + (L)| of the right-hand side. For an edge to contribute to the term it must be in both E + (L) and E + (L ′ ) and at least one of its endpoints in cluster C. This means that every edge in E + (L) ∩ E + (L ′ ) contributes exactly C:|{u,v}∩C|≥1 z C = 1 + x uv . Thus, we get:
C z C • |E + C (L ′ ) ∩ E + (L)| = uv∈E + (L)∩E + (L ′ ) (1 + x uv ) = |E + (L) ∩ E + (L ′ )| + uv∈E + (L)∩E + (L ′ ) x uv
By substituting these rewritten terms in the inequality, we get the following:
cost(L ′ ) ≤ 2LP -LP -- uv∈E + (L ′ )∪E -(L ′ ) x uv -|E + (L) ∩ E + (L ′ )| + 2 uv∈E + (L) x uv - uv∈E + (L)∩E + (L ′ ) x uv ≤ 2LP -LP -- uv∈E + (L ′ )∪E -(L ′ ) x uv -|E + (L) ∩ E + (L ′ )| + 2 uv∈E + (L) x uv ≤ 2LP -|E -(L ′ )| - uv∈E + (L ′ ) x uv -|E + (L) ∩ E + (L ′ )| + 2 uv∈E + (L) x uv
For the second inequality we used the fact uv∈E + (L)∩E + (L ′ ) x uv ≥ 0 and for the third inequality the fact
LP -+ uv∈E -(L ′ ) x uv ≥ uv∈E -(L ′ ) (1 -x uv + x uv ).
Finally, if L ′ is not (2 -δ)-approximate, then:
|E -(L ′ )| + uv∈E + (L ′ ) x uv + |E + (L ′ ) ∩ E + (L)| < δLP + 2 uv∈E + (L)
x uv which is the desired statement.
this section cite: []

Section: References
Ref_id:b0 Title: Generating labels from clicks Year: (2009-02-09)
Ref_id:b1 Title: Fitting tree metrics: Hierarchical clustering and phylogeny Year: (2011)
Ref_id:b2 Title: Aggregating inconsistent information: Ranking and clustering Year: (2008)
Ref_id:b3 Title: Large-scale deduplication with constraints using dedupalog Year: (2009-03-29)
Ref_id:b4 Title: Sublinear time and space algorithms for correlation clustering via sparse-dense decompositions Year: (2022-02-03)
Ref_id:b5 Title: Correlation clustering Year: (2004)
Ref_id:b6 Title: URL Year: ()
Ref_id:b7 Title: Almost 3-approximate correlation clustering in constant rounds Year: (2022)
Ref_id:b8 Title: Singlepass streaming algorithms for correlation clustering Year: (2023)
Ref_id:b9 Title:  Year: ()
Ref_id:b10 Title: Overlapping correlation clustering Year: (2013)
Ref_id:b11 Title: Differentially private correlation clustering Year: (2021-07-24)
Ref_id:b12 Title: A (3 + ε)-approximate correlation clustering algorithm in dynamic streams Year: (2024)
Ref_id:b13 Title: Understanding the cluster linear program for correlation clustering Year: (2024)
Ref_id:b14 Title: Breaking 3-factor approximation for correlation clustering in polylogarithmic rounds Year: (2024)
Ref_id:b15 Title: Solving the correlation cluster LP in sublinear time Year: ()
Ref_id:b16 Title: Solving the correlation cluster lp in nearly linear time Year: (2025)
Ref_id:b17 Title: A graphtheoretic approach to webpage segmentation Year: (2008)
Ref_id:b18 Title: Clustering with qualitative information Year: (2003)
Ref_id:b19 Title: Near optimal LP rounding algorithm for correlation clustering on complete and complete k-partite graphs Year: (2015)
Ref_id:b20 Title: Clustering sparse graphs Year: (2012)
Ref_id:b21 Title: Correlation clustering in constant many parallel rounds Year: (2021-07-24)
Ref_id:b22 Title: Correlation clustering with Sherali-Adams Year: (2022-11-03)
Ref_id:b23 Title: Handling correlated rounding error via preclustering: A 1.73-approximation for correlation clustering Year: (2023-09)
Ref_id:b24 Title: Dynamic correlation clustering in sublinear update time Year: (2024)
Ref_id:b25 Title: Combinatorial correlation clustering Year: (2024)
Ref_id:b26 Title: A robust model for finding optimal evolutionary trees Year: (1993)
Ref_id:b27 Title: A faster algorithm for constrained correlation clustering Year: ()
Ref_id:b28 Title: Tight bounds for parameterized complexity of cluster editing with a small number of clusters Year: (2014)
Ref_id:b29 Title: Correlation clustering for crosslingual link detection Year: (2007)
Ref_id:b30 Title: Web people search via connection analysis Year: (2008)
Ref_id:b31 Title: Higherorder correlation clustering for image segmentation Year: (2011-12-14)
Ref_id:b32 Title: Single-pass pivot algorithm for correlation clustering. keep it simple Year: (2023)
Ref_id:b33 Title: Fuzzy clustering with spatial constraints Year: (2002)
Ref_id:b34 Title: Deterministic pivoting algorithms for constrained ranking and clustering problems Year: (2007)
Ref_id:b35 Title: A correlation clustering framework for community detection Year: (2018)
Ref_id:b36 Title: Constrained k-means clustering with background knowledge Year: (2001)
Ref_id:b37 Title: Flexible constrained spectral clustering Year: (2010)
Ref_id:b38 Title: Fast planar correlation clustering for image segmentation Year: (2012)
