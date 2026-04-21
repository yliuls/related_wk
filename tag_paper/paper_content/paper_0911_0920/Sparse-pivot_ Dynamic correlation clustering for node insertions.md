Title: SPARSE-PIVOT: Dynamic correlation clustering for node insertions
Abstract: We present a new Correlation Clustering algorithm for a dynamic setting where nodes are added one at a time. In this model, proposed by Cohen-Addad, Lattanzi, Maggiori, and Parotsidis (ICML 2024), the algorithm uses database queries to access the input graph and updates the clustering as each new node is added. Our algorithm has the amortized update time of O ε (log O(1) (n)). Its approximation factor is 20 + ε, which is a substantial improvement over the approximation factor of the algorithm by Cohen-Addad et al. We complement our theoretical findings by empirically evaluating the approximation guarantee of our algorithm. The results show that it outperforms the algorithm by Cohen-Addad et al. in practice.

Section: Introduction
In this paper, we present a new dynamic algorithm with node updates for the Correlation Clustering problem with complete information. 1 Correlation Clustering is a well-studied problem that seeks to partition a set of objects into clusters based on their similarity. The problem is defined on a set of items represented as nodes in a graph, with similarity information provided through a set of edges. We assume that all pairs of nodes are classified as either "similar" or "dissimilar" by a noisy classifier. For every pair of similar nodes u and v, there is an edge (u, v) between them (sometimes referred to as a positive edge). For every pair of dissimilar nodes u and v, no edge exists between them (such pairs are sometimes called negative edges). The objective is to find a clustering that minimizes the number of disagreements with the given edge set. An edge (u, v) disagrees with the clustering if u and v are placed in different clusters, while a non-edge (u, v) disagrees if u and v are placed in the same cluster.
The problem was introduced by Bansal, Blum, and Chawla (2004). It can be easily solved if a disagreement-free clustering exists, as each cluster then corresponds to a connected component of the similarity graph G. However, when the classifier makes mistakes, and a disagreement-free solution does not exist, the problem becomes NP-hard. In their original paper, Bansal, Blum, and Chawla proposed a constantfactor approximation, which was subsequently improved in a series of works Charikar et al. (2005); Demaine et al. (2006); Chawla et al. (2015); Cohen-Addad et al. (2022;2023); Cao et al. (2024). In 2005, Ailon, Charikar, and Newman introduced combinatorial and LP-based algorithms with approximation factors of 3 and 2.5, respectively. The best-known approximation factor for Correlation Clustering with Complete Information is currently 1.437 (Cao, Cohen-Addad, Lee, Li, Newman, and Vogl, 2024). Nevertheless, the combinatorial algorithm by Ailon et al. (2008), known as PIVOT, remains one of the preferred choices in practice due to its simplicity and good empirical performance.
Researchers have proposed various variants of PIVOT that operate in parallel and streaming settings (Bonchi et al., 2014;Chierichetti et al., 2014;Pan et al., 2015;Cohen-Addad et al., 2021;Cambus et al., 2022;Behnezhad et al., 2022;2023;Cambus et al., 2022;Chakrabarty & Makarychev, 2023). Recently, there has been growing interest in algorithms that support dynamic updates. Consider a scenario where similarity information is received over time, requiring the clustering to be updated dynamically. Dalirrooyfard, Makarychev, and Mitrovic (2024) showed how to maintain a (3 + ε)-approximation clustering with constant update time per edge insertion or deletion (see also papers by Behnezhad et al. (2019) and Chechik & Zhang (2019)).
Cohen-Addad, Lattanzi, Maggiori, and Parotsidis proposed an algorithm designed for a setting where nodes and edges are stored in a database. Over time, new nodes are added to the database along with their incident edges. After nodes are added, the dynamic algorithm updates the existing clustering. In this model, each edge is included in the database immediately after both its endpoints are inserted; no ad-ditional edges can be inserted or deleted afterward. The sequence of inserted nodes is non-adaptive, meaning it does not depend on the decisions of the clustering algorithm. Formally, we assume that the graph and the order of node arrivals are fixed in advance. Cohen-Addad et al. showed how to achieve a constant (albeit very large) approximation with an update time of log O(1) n, measured in terms of database operations defined as follows: (1) retrieving the degree of a node v; (2) selecting a random neighbor of v; and (3) checking whether two nodes u and v are connected by an edge. The database model was introduced by Assadi and Wang (2022). The algorithm by Cohen-Addad et al. was the first dynamic algorithm for node insertions with sublinear update time. Although it provides a constant approximation, the proof suggests that the constant is very large (the paper does not estimate it). In this work, we propose a (20 + ε)approximation algorithm for Correlation Clustering with an update time of O ε (log O(1) n) database operations per node insertion.
In addition to node insertion, the algorithm by Cohen-Addad et al. supports deleting random nodes. Our algorithm supports a slightly weaker type of deletion: soft deletions of random vertices. When a node is soft-deleted from the graph, it initially remains in the database but is marked as soft-deleted. Moreover, the classifier continues creating edges between newly arriving and soft-deleted nodes. Only when the algorithm requests their deletion are they purged from the database.
To motivate the model, consider the following example: An online store adds new items to its stock daily and aims to cluster all items based on similarity. Whenever a new item is added, the store runs a classifier to identify items similar to the new one. A record for the new item is then created and inserted into the database, along with edges connecting it to similar items. Instead of reclustering the entire dataset, the dynamic algorithm efficiently updates the clustering, requiring only O(log O(1) n) operations per item insertion.
this section cite: ['b2', 'b10', 'b19', 'b11', 'b16', 'b8', 'b8', 'b0', 'b6', 'b13', 'b22', 'b15', 'b7', 'b4', 'b7', 'b9', 'b18', 'b3', 'b12', 'b1']

Section: Algorithm
Our algorithm is based on the 5-approximation variant of PIVOT, developed by Behnezhad, Charikar, Ma, and Tan (2023) for the semi-streaming model. Their algorithm in the static setting works as follows: First, it selects a random ordering π of all nodes. For each node u, it picks the neighbor of u with the smallest rank and stores this neighbor in the pivot array p: p(u) = arg min w∈N (u) π(w). Next, the algorithm identifies all nodes u such that p(u) = u, which we refer to as pivots. We call p(v) the pivot for node v. Then, the algorithm creates a new cluster for every pivot u and assigns to it nodes v with p(v) = u. All remaining unassigned nodes are placed in singleton clusters. Note that for each node u in a singleton cluster, p(u) remains equal to the neighbor of u with the smallest rank. To reiterate, in this variant of PIVOT, each vertex u belongs to the cluster of p(u) if p(u) = p(p(u)), and to a singleton cluster if p(u) ̸ = p(p(u)). Behnezhad, Charikar, Ma, and Tan (2023) provide a dynamic edge-insertion version of this algorithm with constant update time. A dynamic implementation of this algorithm for node insertions can be easily inferred and is presented in Algorithm 1. In this version, when the node u being inserted is a pivot, the algorithm runs the EXPLORE process. This process updates the pivot of each neighbor w of u and reassigns w and its neighbors to the cluster of u or singleton clusters, if necessary. We refer to the clustering produced by this algorithm as REFERENCE CLUSTERING, as our algorithm aims to approximate this clustering.
Algorithm 1 Insertion for REFERENCE CLUSTERING 1: input node u, graph G, ordering π, pivot array p. 2: Compute p(u) = arg min w∈N [u] π(w). 3: if p(u) = u: 4:
Mark u as a pivot.
5:
Create a new cluster with u in it.
6:
Run EXPLORE(u, G, π, p) 7: else if p(u) is a pivot: 8:
Put u in p(u)'s cluster. 9: else: 10: Make u a singleton.
this section cite: ['b5', 'b5']

Section: Making the REFERENCE CLUSTERING algorithm faster
The challenge with REFERENCE CLUSTERING is that, after each insertion, it scans the entire neighborhood of the inserted node u, which can be as large as Θ(n), where n is the size of the current graph. This makes this REFERENCE CLUSTERING prohibitively expensive. To improve the efficiency, we perform this exhaustive search only for pivots. For non-pivot nodes u, we attempt to recover the unknown pivot v by sampling O(log n) neighbors of u, examining the set of pivots of these sampled neighbors, and setting p(u) to the neighbor with the smallest ranked among the neighbors scanned in this process (which could be the samples or their Algorithm 2 EXPLORE 1: input pivot u, graph G, ordering π, pivot array p.
2: for all w ∈ N [u]:
3: if π(p(w)) > π(u): 4:
if w is a pivot, i.e., p(w) = w:
5:
for all z where p(z) = w: 6:
if z ∈ N (u), then p(z) ← u, 7:
else make z a singleton 8:
p(w) ← u.
pivots). We show that this approach succeeds when the cluster C in the reference clustering is sufficiently dense and u does not have too many neighbors outside of C.
Instead of selecting a random ordering of vertices π, our algorithm assigns each value π(u) uniformly at random from the interval [0, 1]. This simplifies maintaining a random ordering in dynamic settings, where the exact number of nodes is not known in advance and also makes the logic of our algorithm a bit simpler.
We present our algorithm for node insertion in Algorithm 3. We use three main ideas:
(1) For vertices u with π(u) ≤ L d(u) , where L = O(log n), we run EXPLORE on u if u is a pivot, similarly to REFER-ENCE CLUSTERING. If u is not a pivot, we run EXPLORE on its pivot to update its cluster.
(2) For vertices u with π(u) > L d(u) , we find a pivot by examining the set of pivots of Θ(log n) random neighbors of u and selecting the neighbor of u with smallest ranked in that set as a pivot for u.
(3) In the obtained tentative clustering, we identify certain nodes, remove them from their clusters, and place them into singleton clusters. Specifically, we remove nodes whose in-cluster degree is below a threshold t. The optimal value of t is determined by trying out O(log n) possible choices. In particular, when a new cluster is made upon arrival of a pivot node, the subroutine BREAK-CLUSTER removes some nodes from the cluster and makes them singletons. When the node inserted is not a pivot and is assigned to pivot v, the subroutine UPDATE-CLUSTER updates the set of nodes assigned to v that are put in singletons upon insertion of the new node (See Appendix A for details).
We now provide some motivation for the algorithm. First, we observe that the expected time required for finding pivots in items (1) and ( 2) is log O(1) n. In item (1), we spend d(u) log n time with probability log n/d(u). In item (2), we always spend O(log n) time by examining O(log n) random neighbors and their pivots.
We run the step described in item (3) after (roughly) every ϵs insertions, where s is the size of the cluster. In this step, we test O(log n) different thresholds t and, for each choice of t, estimate the cost of the split defined by t by sampling random edges within the cluster (see Appendix A for details). Hence, the total running time of one such step is s log O(1) n, and the amortized cost per insertion is log O(1) n.
We now discuss the approximation factor. Item (1) ensures that all pivots in the reference clustering are marked as pivots by our algorithm with high probability. Specifically, for every pivot u in the reference clustering, we have π(u) ≤ L/d(u) with probability at least 1poly(1/n), ensuring that EXPLORE is called on every pivot u. This follows from the observation that if π(u) > L/d(u), then π(v) > L/d(u) for all neighbors v ∈ N [u]. The probability of this event is at most (1
-L/d(u)) d(u) ≤ poly(1/n).
We define good nodes as follows: loosely speaking, a node is good if it is connected to most nodes within its cluster and to relatively few outside of it. We show that good nodes in our clustering are assigned the same pivots as in the reference clustering (when our algorithm and REFERENCE CLUSTERING use the same ordering π). For now, let us assume that all nodes are good. If a good node u arrives after its pivot w, then u will join w's cluster when w scans all its neighbors, including u. If u arrives after w while more than 25% nodes in w's cluster have yet to arrive, then for at least one node v arriving after u, we will call EXPLORE, which will assign u the correct pivot w. Here, we rely on the assumption that all nodes are good. Finally, if u arrives among the last 25% of nodes in w's cluster, then by the time u arrives, the algorithm will have assigned the correct pivot w to most of its neighbors in the reference cluster. Consequently, a high fraction of u's neighbors will have w as their pivot, and some of these neighbors will be included in the random sample of neighbors (see item (2)). Hence, u will also be assigned the correct pivot. Our full analysis can be found in Section 3.
Let us now consider bad (not good) nodes. These nodes incur a very substantial cost in the reference clustering. If we could remove them from their reference clusters and place them into singleton clusters, the overall clustering cost would not increase significantly-in fact, it might even decrease. Unfortunately, our algorithm cannot identify these bad nodes; as a result, they may join clusters other than their own reference clusters. This can substantially increase the cost of those clusters. To address this issue, we partition each tentative cluster into two parts (see item (3)): the first part remains a cluster, while the second part is broken into singleton clusters.
Theorem 2.1 provides the approximation guarantees of our algorithm which we call SPARSE-PIVOT. Theorem 2.1. For any ϵ < 1/1000, the expected cost of SPARSE-PIVOT with parameter ϵ is at most 4(1 + O(ϵ)) times the expected cost of the REFERENCE CLUSTERING.
Our algorithm for soft deletions is very simple: We ignore them! In fact, we recompute the whole clustering again after Θ(ε)N many updates, where N is the number of nodes when we last recomputed the clustering. The recomputation is only necessary for deletions. Our running time guarantees are provided in Theorem 2.2. Theorem 2.2. Let T be the total number of updates. With high probability, Algorithm 3 runs in amortized poly(log T, 1 ε ) time.
Algorithm 3 INSERT-NODE-SPARSE-PIVOT 1: input node u to be inserted, current graph G of size n, ordering π, pivot vector p.
2:
Let π(u) ∈ [0, 1] be chosen uniformly at random.
3: Let p(•) indicate the lowest rank neighbor of each node. 4: Let B v indicate the set of nodes with pivot v, and let C v ⊆ B v be the nodes of B v that are in v's cluster. 5: if π(u) ≤ L d(u) : 6:
Find v = arg min w∈N [u] π(w).
7:
if v = u: 8:
Make u a pivot: p(u) ← u, and make a new cluster with u in it:
B u = {u}, t v = 0 9: EXPLORE(u, G, π, p) 10: C u ← BREAK-CLUSTER(B u ). 11: else if v ̸ = u and v is a pivot: 12: p(u) ← v, B v ← B v ∪ {u}. 13: C v ←UPDATE-CLUSTER(u, B v ) 14: if d(v) ≤ L π(u) : 15: EXPLORE(v, G, π, p) 16:
else if v ̸ = u and v is not a pivot: 17: make u a singleton.
18:
else:
19: Let S be a O(log n)-sized sample of N [u]. 20: Let s * = arg min s∈S,{p(s),u}∈E(G) π(p(s)). Let v := p(s * ) 21: if π(v) < π(u): 22: p(u) ← v, B v ← B v ∪ {u}. 23: C v ←UPDATE-CLUSTER(u, B v ) 24:
if u did not get clustered: 25: make u a singleton, u does not have a pivot.
this section cite: []

Section: Analysis Preliminaries
We use subscript ref to refer to REFERENCE CLUSTERING. We fix time, and we compare the cost of our algorithm to the cost of REFERENCE CLUSTERING at this time. We refer to the current graph as G. For a node v, ordering π and clustering algorithm A, let C π A (v) be the cluster of v in A with respect to the ordering π. We drop the superscript π and subscript A when it is clear from the context. The clustering algorithms that we consider throughout our analysis all define pivots for all non-singleton clusters.
Given a ordering π and clustering algorithm A, let p π A (u) denote the pivot of u chosen by algorithm A. For a set of nodes S, let d S (v) be the degree of v in S, and let d(v) be the degree of v in the graph G. We will classify nodes depending on how their neighborhoods intersect the cluster to which they are assigned.
Definition 2.3. Let α < 1 and β > 1. Let A be a clustering algorithm. For a fixed ordering π, we call vertex u
• A-light if d C (u) ≤ |C|
3 , where C = C π A (u).
• (A, α)-poor if d(u) ≤ αd(p A (u)) and u is not light.
• A-heavy if d(u) ≥ β|C|, where C = C π A (u).
• A-bad if u is (A, 3αβ)-poor, A-heavy or A-light, and A-good otherwise.
• A-lost if the number of A-bad neighbors of u in C π A (u) is at least β times the number of A-good neighbors of u in C π A (u).
We drop Aif the clustering algorithm is clear from the context.
Definition 2.4 (Poor clusters). For a fixed ordering π and clustering algorithm A, let C be a cluster with pivot v. We call C an (A, α)-poor cluster if C has at least one (A, α)poor node.
Definition 2.5 (Good and bad clusters). Let γ < 1 be a constant. For a fixed ordering π and clustering algorithm A, we call a cluster C π A good if it is not (A, α)-poor, and it has at least γ|C π A | good nodes. Otherwise, we call it bad. Remark 2.6. Note that instead of fixing the clustering algorithm A and ordering π, we can still have the above definitions if we fix the clusters and pivots.
Definition 2.7 (Cost of a cluster). Given a clustering, the cost of a cluster C is the number of non-edges inside C, and half of the number of edges with exactly one endpoint in C.
The cost of a clustering is the sum of the cost of its clusters.
this section cite: []

Section: Analysis

this section cite: []

Section: Analysis outline
First, we explain the intuition behind classifying nodes in Definition 2.3. Let the cost of a node be half the number of non-neighbors it has in its cluster, plus half the number of neighbors it has outside the cluster. Note that the sum of the costs of nodes in a cluster equals the cost of the cluster. Consider A to be the reference clustering in Definition 2.3. A light node in cluster C has a lot of non-edges attached to it in C, and a heavy node has a lot of edges attached to it that leave the cluster C. So both have a high cost. In fact, we show that if we make them singletons, the cost of the clustering does not change much. So, in a sense, in our algorithm, we do not care how ref -heavy or ref -light nodes are being clustered as long as their cost is somewhat comparable to their cost as singletons.
Consider poor nodes. A poor node has a much lower degree than its pivot. We show that if a cluster has at least one poor node, then any node in this cluster that is not heavy or light must be poor (Lemma 3.1). Then we show that, in fact, if we make the whole cluster singleton, the cost of the clustering does not change much on average. This is because the pivot of this cluster has a very high cost, and any node becomes the pivot of a poor cluster with low probability.
In summary, we have shown that if we make the bad nodes (heavy, poor, or light) in the reference clustering singleton, the cost of the clustering does not change much (Lemma 3.2). We further show that we can make all the nodes in a bad cluster singleton as well since the cost of this cluster is already too high. We call this clustering ref ′ .
We then show that in SPARSE-PIVOT, not only do we detect pivots correctly with high probability (Lemma 3.4), but also all the good nodes that are not lost, i.e., they do not have many bad neighbors, are assigned the correct pivot. We show this in two parts: Consider a pivot v, and suppose C is the set of all the ref -good nodes with pivot v that are not ref -lost. First, we show that if a good node u ∈ C arrives rather early compared to other nodes in C, at some point its pivot runs EXPLORE and it detects if u is clustered incorrectly (Lemma 3.5). If u arrives rather late, then the sampling procedure will hit one of the neighbors of u in C that is correctly clustered and so correctly assigns v as the pivot of u (Lemma 3.6). Now since we cluster the ref -good nodes correctly with high probability, if we could detect ref -bad nodes in SPARSE-PIVOT, we could make them singletons, and thus get ref ′ . However, instead, if B v is the set of nodes that have v as their pivot, we make a dense subset C v of B v one cluster, and make the rest of B v singleton. This step is crucial since there might be many (bad or lost) nodes incorrectly assigned to v that increase the number of the non-edges in B v significantly. We show that the cost of making B v \ C v singleton is at most 4 times the cost of making the ref -bad nodes in B v singleton (Lemma 3.7). Now we begin our formal analysis. Let
β ≥ 4+ϵ ϵ , α < min( ϵ 24β , 1 39β ), and γ ≤ ϵ 2 . Let L ≥ 4cβ ϵγα log n for some arbitrary large constant c. Let the number of samples when the inserted u satisfies π(u) > L/d(u) be at least 100 log( 1 1-x ) = O(log n), where x = ( 1 β+1 -ϵ)/β.
this section cite: []

Section: Making all the bad and lost nodes singleton in Reference clustering
We show that if a cluster has one α-poor node, then all the nodes in that cluster are 3αβ-poor.
Lemma 3.1. Let π be an ordering and A be a clustering algorithm. If C π A is an (A, α)-poor cluster, then any u ∈ C π A which is not light or heavy is (A, 3αβ)-poor.
Proof. Let v be the pivot of C = C π A . First, since C is an (A, α)-poor cluster there is a node w ∈ C that is (A, α)poor. Since, by definition of poor nodes, w is not light, we have that |C|/3 ≤ d(u) ≤ αd(v). So 3αd(v) ≥ |C|. Now for any u that is not heavy we have d(u) ≤ β|C| ≤ 3αβd(v). Thus if u is not light, then u is (A, 3αβ)-poor.
The following lemma shows that if, in REFERENCE CLUS-TERING, we make all the bad nodes and lost nodes singleton, the cost only increases by a factor of (1 + O(ϵ)).
this section cite: []

Section: Lemma 3.2 (Cost of making bad and lost nodes singletons).
Consider a clustering algorithm A where the probability of any node v being a pivot is at most 1/d(v). Let B be the algorithm that first runs A, and then makes A-heavy, A-light, (A, 3αβ)-poor and A-lost nodes singletons. Then
E π [cost B ] ≤ (1 + 7ϵ)E π [cost A ]
Proof. Let α ′ = 3αβ, and let A ′ be the algorithm that runs A and then makes the A-heavy and A-light nodes singleton. Let A ′′ be the algorithm that runs A ′ , and makes all the (A, α ′ )-poor nodes singleton. For the clarity and brevity of notation, we use δ
def = 4α ′ 1+3/2•α ′ 1-5/2•α ′ .
For any fixed ordering π, by Lemma C.2, cost π A ′ ≤ β+1 β-1 cost π A , and
cost A ′ ≤ β + 1 β -1 cost A .
Now note that the probability of any node v being a pivot in A ′ is at most the probability of any node being a pivot in A, which is at most 1/d(v). The cost of making (A, α ′ )poor nodes singletons is, at most, the sum of their degrees. By Lemma C.1, that expected sum of (A, α ′ )-poor nodes degrees is upper bounded by δ • E π [cost A ]. Hence, we have
E π [cost A ′′ ] ≤ (1 + δ) E π [cost A ′ ] .
Finally, note that any A-lost node is a A ′′ -light node or A ′′heavy node with parameter β/3. To see this, fix a ordering π. Consider a A-lost node u, and let
C = C π A (u). Let C ′ ⊆ C be the set of A-bad nodes in C. Let C ′′ := C π A ′′ (u) = C \C ′ . By the definition of A-lost nodes, we have d C ′ (u) ≥ βd C ′′ (u). If u is not A ′′ -light, then d C ′′ (u) ≥ |C ′′ |/3. So d(u) ≥ d C ′ (u) ≥ βd C ′′ (u) ≥ β 3 |C ′′ |. So u is A ′′ - heavy node with parameter β/3. By Lemma C.2, cost π B ≤ β+3 β-3 cost π A ′′ . So cost B ≤ β+3 β-3 cost A ′′ . Putting all the steps together yields E π [cost B ] ≤ β + 1 β -1 • (1 + δ) • β + 3 β -3 • E π [cost A ] . Now since β ≥ 4+ϵ ϵ , we have β+1 β-1 < β+3 β-3 ≤ (1 + ϵ), and since α ′ < 1/13, we have 1+3/2•α ′ 1-5/2•α ′ < 2, and α ′ < ϵ/8 gives us 4α ′ 1+3/2•α ′ 1-5/2•α ′ < ϵ. So E π [cost B ] ≤ (1 + ϵ) 3 E π [cost A ] ≤ (1 + 7ϵ)E π [cost A ].
Lemma 3.3. Consider a fixed ordering π and a clustering algorithm A. If B is a clustering algorithm that runs A and makes all A-bad and A-lost nodes, as well as all the nodes in bad clusters in A singleton, then cost(B) ≤ (1 + 8ϵ)cost(A).
Proof. Let A ′ be the algorithm that runs A and makes the A-bad nodes and A-lost nodes singleton. Then we can see B as running A ′ and making all the good nodes remaining in A-bad clusters singleton. Note that by Lemma 3.2 cost(A ′ ) ≤ (1 + 7ϵ)cost(A). Note that in A ′ all poor clusters are singletons, since a poor cluster has at least one (A, α) poor node, and by Lemma 3.1 all the non-pivot nodes in this poor cluster that are not heavy or light are (A, 3αβ)poor. By definition of bad nodes, all the nodes in a poor cluster are bad. Consider a bad cluster C in A. The cost of C in A is at least 2 3 (1 -γ)|C| 2 by Lemma C.3. Making the good nodes in C singleton in A ′ adds at most γ 2 |C| 2 to the cost of A ′ since there are at most γ|C| good nodes in C, and the only cost added by making these nodes singleton is through the edges between them. So the total cost added to the cost of A ′ by making these good nodes singleton is at most
3γ 2 2(1-γ) cost(A). So cost(B) ≤ cost(A ′ ) + 3γ 2 2(1-γ) cost(A). Since γ ≤ ϵ/2, we have 3γ 2 2(1-γ) ≤ 4γ 2 (1-γ) 2 ≤ ϵ 2 < ϵ. So cost(B) ≤ (1 + 8ϵ)cost(A).
this section cite: []

Section: SPARSE-PIVOT comparison to REFERENCE CLUSTERING
Let P π alg be the pivot set of our algorithm and P π ref be the pivot set of the reference clustering.
Lemma 3.4. Let L ′ = L/2β. If v ∈ P π ref , then π(v) ≤ L ′ /d(v) holds with probability at least 1 -1/n c-1 .
Note that not only does Lemma 3.4 say that each pivot in REFERENCE CLUSTERING is also a pivot in SPARSE-PIVOT, but it also provides a stronger guarantee on π(v).
For the next Lemma, note that the definitions of light, poor, etc, are well-defined if the clustering is fixed (and not necessarily the ordering π). We show that with high probability, one of the good nodes in C ref (v) triggers EXPLORE function for v so that v can correct its cluster.
Lemma 3.5. Fix a clustering and its pivots that REFER-
ENCE CLUSTERING algorithm can produce. Consider a pivot v whose cluster C ref (v) is good. Let C ref (v)[good]
be the good nodes in C ref (v). Then, with high probability, our algorithm assigns v as the pivot of u, for any [good]. The "first" here is taken with respect to the dynamic ordering and the probability taken over rankings that produce the fixed clustering.
(1 -ϵ)|C ref (v)[good]| first nodes u of C ref (v)
Lemma 3.6 shows that SPARSE-PIVOT identifies the pivot of all the good nodes that are not lost correctly. Lemma 3.6. Fix a clustering and its pivots that the reference algorithm can produce. Let v be a pivot in that reference clustering and u be in the v's cluster. With high probability, any u that is ref-good and not ref-lost is assigned to v by Algorithm 3.
Proof. Let C = C ref (v). First, if u is among the first (1 -ε) fraction of the nodes of C[good] with respect to the dynamic ordering, then by Lemma 3.5, with high probability Algorithm 3 assigns u to v's cluster.
So, second, consider the case when u is in the last ε fraction of C [good]. Let D refer to the ref -good neighbors of u that are among the first 1 -ε fraction of the nodes of C[good]. By Lemma 3.5, with high probability Algorithm 3 assigns the nodes in D to v's cluster. We now lower-bound |D|. We will use that lower bound to show that the sampling process in Algorithm 3 (when π(u) < L/d(u)) will sample at least one node from D, and hence assign v to u's cluster.
Since u is not ref -lost, by Definition 2.3, u has at most β times more ref -bad than ref -good neighbors in C. Hence, at least 1/(1 + β) • |C| neighbors of u in C are ref -good. Also, observe that 1/(1 + β) • |C| -ε • |C[good]| ≥ (1/(1 + β) -ε) • |C| of those neighbors are among the first 1 -ε fraction of C[good]. Therefore, |D| ≥ (1/(1 + β) -ε) • |C|. On the other hand, since u is a ref -good node, we have that d(u) < β • |C|.
Finally, we conclude that
|D| d insert (u) ≥ |D| d(u) ≥ 1/(1 + β) -ε β .
where d insert (u) is the degree of u at the time of insertion.
Let n be the number of nodes when inserting u. If x := 1/(1+β)-ε β , the probability of not sampling any node in D is at most (1 -x) |S| , where S is the sample set. Recall that |S| ≥ 100 log( 11-x ) • log n, so this probability is at most 1/n 100 . Note that if n is the current number of nodes, n ≥ n/(1 + ϵ) since we recompute everything when the number of updates is at most ε times the number of nodes at last RECOMPUTE. So the probability that u is clustered correctly is at most 1 -(1 + ϵ)/n 100 .
Next, Lemma 3.7 aids us to compare REFERENCE CLUS-TERING with SPARSE-PIVOT clustering. Given pivot v and set B v , let C t be the set of nodes in B v with degree at least t. Let cost(B v |C t ) be the cost of the clustering on B v where all the nodes in C t are clustered as one cluster and the nodes in B v \ C t as singletons. In particular, this cost equals half of the number of edges from B v to outside of B v , plus the number of edges with at least one endpoint in B v \ C t , plus the number of non-edges in C t .
Lemma 3.7. Consider a pivot v, and let C * be the set of good nodes that are not lost in C ref (v). There is a threshold t ∈ {1, (1 + ϵ), . .
. , (1 + ϵ) ⌈log n⌉+1 }, such that cost(B v |C t ) ≤ 4 1-2ε cost(B v |C * ).
this section cite: []

Section: Putting it all together
Proof of Theorem 2.
1. We refer to REFERENCE CLUSTER-ING as ref and the clustering of SPARSE-PIVOT by B. Let A be the clustering algorithm that runs REFERENCE CLUS-TERING to obtain ref , and then makes all ref -bad, ref -lost nodes as well as all the nodes in ref -bad clusters singleton. By Lemma 3.3, E(cost(A)) ≤ (1 + 8ϵ)E(cost(ref )). Note that all the nodes that are not singletons in A are good nodes that are not lost and are not in bad clusters in ref . Next we show that E(cost(B)) ≤ 4(1 + O(ϵ))E(cost(A)). Consider a pivot v in REFERENCE CLUSTERING. By Lemma 3.4 v is also a pivot in SPARSE-PIVOT with high probability. We will compare clustering costs by dividing up the clusters into groups: We consider all the nodes that are in B v for a pivot v together and compare the cost of clustering these nodes in A and in B. We have two cases: Case 1: C ref (v) is a good cluster. Consider B v , the set of nodes that are assigned to v as their pivot, and C v ⊆ B v , the set of nodes in B v that are clustered with v and are not singletons. Let C t = {u ∈ B v |d(u) ≥ t}, and let cost(B v |C t ) be the cost of clustering all nodes in C t as one cluster and the nodes in B v \ C t as singletons. Let t * be the threshold in {1, (1 + ϵ), . . . , (1 + ϵ) ⌈log n⌉+1 } where cost(B v |C t * ) is minimized. By Theorem A.5, cost(B
v |C v ) ≤ (1 + 220ε)cost(B v |C t * ).
If C * v is the set of good nodes that are not lost in C ref (v), then by Lemma 3.6 all the nodes in C * v are correctly assigned to v, and so they are in B v . Note that A clusters the node in B v as follows: put all the nodes in C * v in one cluster and make all the nodes in B v \ C v singleton. By Lemma 3.7, we have that
cost(B v |C t * ) ≤ 4 1-2ε cost(B v |C * v ), so cost(B v |C v ) ≤ 4(1+220ε) (1-2ε) cost(B v |C * v ) ≤ 4(1 + 230ε)cost(B v |C * v ) since ε ≤ 1/1000. Case 2: C ref (v) is a bad or poor cluster. We know that all the nodes in C ref (v) are singletons in A. Let t * be the threshold in {1, (1 + ϵ), . . . , (1 + ϵ) ⌈log n⌉+1 } where cost(B v |C t * ) is minimized. So cost(B t * |C v ) is at most the cost of making B v single- ton, i.e. cost(B v |C (1+ϵ) ⌈log n⌉+1 ) = cost(B v |∅). By The- orem A.5 cost(B v |C v ) ≤ (1 + 220ε)cost(B v |C t * ). So cost(B v |C v ) ≤ (1 + 220ε)cost(B v |∅).
Finally, note that a node not in any B v is a singleton in both B and A. Putting the above two cases together, So we have
E(cost(B)) ≤ 4(1 + 50ϵ)E(cost(A)) ≤ 4(1 + 230ϵ)(1 + 8ϵ)E(cost(ref )) ≤ 4(1 + 1000ϵ)E(cost(ref )
), where the last inequality uses the fact that ϵ < 1/1000.
this section cite: []

Section: Random deletions
Let n 0 be the number of nodes in the graph just after the last recomputation. Our algorithm for deletions is quite simple:
1. Ignore deletions.
2. After εn 0 /6 updates, counting both insertions and deletions, recompute the clustering from scratch by treating all non-deleted nodes as if they had been inserted one by one again. These insertions are processed by Algorithm 3. Our recomputing procedure is described in Appendix B.
Recall that on a deletion update, a node to be deleted is chosen uniformly at random among the existing nodes. Observe that the choice of deletions is independent of the randomness used by our algorithm. At time t of the algorithm, let D t be the nodes deleted since the last clustering recomputation. We think of D t as the nodes waiting to be deleted.
By construction, we have |D t | ≤ εn 0 /6. Since after the recomputation there are n 0 nodes in the graph, D t is a subset of (at least) n 0 nodes, and hence Pr (u ∈ D t ) ≤ |D t |/n 0 ≤ ε/6. There is inequality instead of equality, as the εn 0 /6 updates might contain insertions, resulting in a reduced probability of a node appearing in D t .
Let C NO-DEL be the clustering obtained by 5-approximate REFERENCE CLUSTERING at time t where deletions D t are ignored. Let C DEL be the clustering obtained by REFERENCE CLUSTERING at time t in which deletions D t are considered. Let P t ⊆ V × V be the node pairs that C NO-DEL pays for. We aim to lower-bound the expected number of pairs in P t that C DEL also pays for. Consider a pair e = {u, v} ∈ P t . We will lower-bound the probability that the clustering of u and v is the same in C DEL as in C NO-DEL Consider a node u; the exact same analysis applies to v.
First, assume that u is a singleton in C NO-DEL . This implies that the neighbor of u with smallest rank, w, is not a pivot. Node w is not a pivot because it has a neighbor w ′ with a rank smaller than w. If none of w, w ′ , or u is in D t , then u is a singleton in C DEL . Since we have Pr (w ∈ D t or w ′ ∈ D t or u ∈ D t ) ≤ Pr (w ∈ D t ) + Pr (w ′ ∈ D t ) + Pr (u ∈ D t ) ≤ ε/2, then in this case, the clustering of u in C DEL is the same as in C NO-DEL with probability at least 1 -ε/2.
Second, assume that u is not a singleton in C NO-DEL . This implies that the highest-rank neighbor w of u is a pivot. By deleting nodes, and unless w is deleted, w remains a pivot. So, unless u or w are in D t , the clustering of u is the same in C DEL and C NO-DEL . Since we have Pr (w ∈ D t or u ∈ D t ) ≤ ε/3, in this case, the clustering of u in C DEL is the same as in C NO-DEL with probability at least 1 -ε/3. This analysis implies that the clustering of a pair {u, v} is the same in C DEL as in C NO-DEL with probability at least 1 -5ε/6. Hence, by the linearity of expectation,
E [cost(C DEL )] ≥ (1 -5ε/6)|P t | ≥ (1 -ε) cost(C NO-DEL ).
this section cite: []

Section: Experiments
In this section, we empirically demonstrate that our approximation guarantee is better than that of REFERENCE CLUSTERING and the algorithm of (Cohen-Addad et al.). In the rest, we use DYNAMIC AGREEMENT to refer to the approach in (Cohen-Addad et al.).
Algorithm Parameters In SPARSE-PIVOT, for the BREAK-CLUSTER and UPDATE-CLUSTER subroutines we do the following: in BREAK-CLUSTER we consider O(log n) many candidates for C v , estimate their costs and pick one with the lowest cost. In UPDATE-CLUSTER we update our O(log n) estimates by adding the new node, and again pick the one with lowest cost. To simplify the code, we heuristically alter the BREAK-CLUSTER and UPDATE-CLUSTER subroutines as follows. In BREAK-CLUSTER, for each node u ∈ B v , we sample O(log n) nodes in B v . If u is attached to half of them, we add u in C v . In UPDATE-CLUSTER, we add u to C v , even though u might not be attached to many nodes in C v . After at least ε|B v | nodes are added to B v , we rerun BREAK-CLUSTER. Note that the reason we get a (20 + O(ε)) approximation instead of a (5 + O(ε)) approximation is the BREAK-CLUSTER subroutine, so depending on the application, one can replace this subroutine with a version that one sees fit. Furthermore, we run RECOMPUTE every time the number of deletions reaches εN , instead of the total number of updates. We observe that this does not degrade the approximation guarantee and slightly improves the running time.
We set the experiment parameters and the parameters for DYNAMIC-AGREEMENT to be the same as in (Cohen-Addad et al.). We choose a random ordering for the arrival of the nodes, and at each step, with probability 0.8, we insert the next node, and with probability 0.2, we delete a random node. If all the nodes have been inserted once, we delete them until no node is left. We set the parameter ε for SPARSE-PIVOT to be 0.1.
this section cite: []

Section: Datasets
We use the same datasets as in (Cohen-Addad et al.) for a complete comparison. We evaluate the algorithms on two types of graphs.
(1) Sparse real-world graphs from SNAP (Jure, 2014): a social network (musae-facebook), an email network (email-Enron), a collaboration network (ca-AstroPh), and a paper citation network (cit-HepTh)
(2) The drift dataset (Vergara et al., 2012;Rodriguez-Lujan et al., 2014) from ICO Machine Learning Repository (Dua et al., 2017), which includes 13,910 points embedded in a space of 129 dimensions. A graph is constructed by placing an edge between two nodes if their Euclidean distance is less than a certain threshold. This setup is used to easily change the density of the graph and test how it affects the algorithms. The thresholds we choose are the same as in (Cohen-Addad et al.), and they are the mean of the distances between all nodes divided by c ∈ {10, 15, 20, 25, 30}. The lower the threshold, the sparser the graph. The density of a graph is the ratio of the number of edges and the number of nodes.
Baselines We use three baselines: making all nodes singletons, which we call SINGLETONS, DYNAMIC-AGREEMENT, and REFERENCE CLUSTERING. We divide the cost of each algorithm by the cost of SINGLETONS. Since REFERENCE CLUSTERING handles only node insertions, we process deletions in a way similar to (Cohen-Addad et al.). Note that our results slightly differ from that of (Cohen-Addad et al.) since they depend on the randomness of node arrivals. Moreover, the running time depends on the machine in which the algorithm is being run. Nevertheless, the scale of results we obtain does reproduce that of (Cohen-Addad et al.).
Results: Approximation Guarantee For all the datasets, our approximation guarantee is better than DYNAMIC-AGREEMENT and SINGLETONS. For SNAP graphs, we plot the correlation clustering objective every 50 steps. Figure 1 shows this objective for one of these graphs, and the rest can be found in Appendix F. The average clustering objective for the drift dataset graphs is shown in Table 1.
Results: Running time Our experiments focus on the solution quality of SPARSE-PIVOT. Nevertheless, we compare the running times for completeness and illustrate that SPARSE-PIVOT is faster than DYNAMIC-AGREEMENT in practice, see Appendix F.
Density DA RC SP 235.36 0.69 0.59 0.6 114.87 0.59 0.64 0.49 69.74 0.5 0.5 0.41 52.17 0.39 0.42 0.32 42.25 0.35 0.35 0.29 Table 1. Clustering Objective of DYNAMIC-AGREEMENT (DA), REFERENCE CLUSTERING (RF) and SPARSE-PIVOT (SP). The smaller the number, the better.
this section cite: ['b21', 'b24', 'b23', 'b20']

Section: A. Implementing cost estimates
Our clustering procedures, e.g., Algorithm 3, for each pivot v maintain two sets of nodes: B v and C v . The set B v is a set of nodes whose pivot is v. However, having B v as one cluster might sometimes be very far from an optimal clustering of the nodes within B v . So, our algorithm computes a cluster C v ⊆ B v for which we can guarantee a relatively low cost; details of this analysis are provided in our proof of Theorem 2.1. To compute a cluster C v , our algorithm estimates the costs of several clusters and chooses C v as the cluster with the lowest estimated cost. In this section, we describe how to estimate the cost of a cluster efficiently, that is, in only poly(log n, 1/ε) time. We need to handle two cases: how to estimate the cost of a given cluster C from scratch, i.e., in a static manner, and how to maintain the cost estimate of a cluster C under node insertions.
We need the former case for our recomputation or when we create an entirely new B v because v is just becoming a pivot. It might be tempting to create new B v by "pretending" that the nodes of B v have been inserted one by one. However, this approach has a small subtlety. Namely, when a node v is inserted, only at that point are the edges incident to v included in our graph, and no edge between v and a node inserted in the future is known. On the other hand, if we "pretend" that an already existing sequence of nodes is just now inserted, then a currently processed node also has edges to its neighbors that have yet to be processed/inserted. This scenario slightly affects how we count edges and non-edges within B v or C v .
this section cite: []

Section: A.1. Static version

this section cite: []

Section: A.1.1. WITHIN-CLUSTER COST ESTIMATE
We first design a procedure to estimate the cost within a cluster C, i.e., the number of non-edges within C, by spending only O(log n) time per edge. It is provided as Algorithm 4. As shown by Lemma A.1, this estimate is tightly concentrated as
Algorithm 4 IN-CLUSTER-COST-ESTIMATE 1: Input set C ⊆ V 2: τ C def = 5 • |C| • log(n)/ε 3 3: for i = 1 . . . τ C : 4:
Uniformly at random, sample two distinct nodes v and w from C 5:
if {w, v} is a non-edge, then S ← S + 1 6: return S • |C| 2 /τ C long as the number of non-edges is in Ω(|C|). If the number of non-edges is lower, then their actual number is irrelevant to our algorithm. Updating this cost dynamically is more involved, and we elaborate on details in Appendix A.2.
Lemma A.1 (In-cluster cost estimate). Let C ⊆ V be a set of nodes. Then, for ε < 1/2, Algorithm 4 (IN-CLUSTER-COST-ESTIMATE) uses O(|C| log(n)/ε 2 ) running time and outputs Y for which with high probability the following holds:
• If the number of non-edges within C if at least 2ε|C|, then Y is a 1 ± ε multiplicative approximation of that number of non-edges.
• Otherwise, Y < 3ε|C|.
Proof. Let t be the number of non-edges in C. Let X i be a random 0/1 variable equal 1 iff the i-th {v, w} pair sampled by Algorithm 4 is a non-edge. Then,
E [X i ] = Pr (X i = 1) = t |C| 2 .
Let S ′ be the value of S at the end of Algorithm 4. Since
S ′ = τ C i=1 X i , we have E [S ′ ] = τ C • t |C| 2 . (1
) Let Y def = S ′ • |C| 2 τ C
be the output of Algorithm 4. Observe that E [Y ] = t. Therefore, the expected value of Y is the desired one. In the rest, we analyze the concentration bounds of this estimator.
Consider two cases based on the value of t.
Case t ≥ 2ε|C|. Recall that τ C = 5|C| log(n)/ε 3 . Replacing the bounds on t and τ C in Equation ( 1) yields
E [S ′ ] ≥ 10|C| 2 • log n ε 2 |C| 2 ≥ 20 log n ε 2 . (2)
Since S ′ is a sum of independent 0/1 random variables, by the Chernoff bound, it holds thatfoot_0
Pr (|S ′ -E [S ′ ] | < εE [S ′ ]) ≤ n -6 .
This now implies that for t ≥ 2ε|C|, with probability at least 1 -n -6 , Y is a (1 ± ε) multiplicative approximation of t.
Case t < 2ε|C|. In this case, we would like to claim that very likely it holds that Y < 3ε|C|. This can be argued by applying the Chernoff bound as follows.
Observe that for this value of t we have
E [S ′ ] < 20 log n ε 2 . Hence, Pr S ′ > (1 + ε) 20 log n ε 2 ≤ n -6 .
Therefore, with probability at least 1 -n -6 , it holds that
Y ≤ (1 + ε) 20 log n ε 2 • |C| 2 τ C < (1 + ε)2|C| < 3ε|C|, for ε < 1/2.
this section cite: []

Section: A.1.2. SINGLE-CLUSTER + SINGLETONS COST ESTIMATE
We now discuss how to estimate the cost of B for a given C, where C is taken as a single cluster, while all the nodes in B -C are singletons. By cost of B we mean the number of edges with at least one endpint in B -C and the other endpoint in B, plus the number of non-edges in C. Note that the true correlation clustering cost of B is the above cost plus half of the edges from B to outside of B, but since we need these costs to compare different choices of B and the number of edges going outside of B is indipendent of this choice, it does not influence our comparison.
In this section, given two node subsets X and Y , we use e(X, Y ) to denote the number of edges with one endpoint in X and the other in
Y . In particular, e(X, X) is the number of edges in G[X].
We abbreviate e(X, X) to e(X).
First, the entire cost of C, denoted by cost(C), equals the sum of e(C, V -C) and the number of non-edges within C.
Observe that e(C) = |C| 2 -[the number of non-edges within C].
So, we have Second, it remains to account for making the nodes in B -C singletons. The edges E(C, B -C) are already accounted for by w∈C d(w). To account for the cost of making B -C singletons, the following procedure can be used:
• For each node w ∈ B -C, iterate over all the edges in its adjacency list and
• to an edge from E(w, C) assign weight 0; to an edge from E(w, B -C) assign weight 1/2; and to an edge from E(w, V -B) assign weight 1.
Observe that an edge {x, y} with x, y ∈ B -C is counted twice: once in the adjacency list of x and once in the adjacency list of y. Hence, the sum of these edge weights and cost(C) equals the cost of clustering B.
However, using the above procedure directly can result in a running time that is too long. Instead, we would like a procedure with the running time of O(|B| • poly(log n, 1/ε)). Nevertheless, estimating the sum of the edge weights in the desired time is simple. We outline one such approach in Algorithm 5.
Algorithm 5 COST-ESTIMATE 1: Input node sets B and C ⊆ B.
2: cost = w∈C d(w) -2 |C| 2 + 3 • IN-CLUSTER-COST-ESTIMATE(C) 3: Let η def = 10 • log(n)/ε 3 4: for w ∈ B -C: 5:
Sample η edges incident to w, each edge sampled independently and uniformly at random 6: For S ∈ {C, B -C, V -B}, let Z S (w) be d(w)/η multiplied by number of sampled edges incident to S 7:
cost = cost + Z B-C (w) 2 + Z V -B (w) 8: return ( cost + 9ε|C|)/(1 -37ε)
Let Z S (w) be as defined in Algorithm 5. Observe that E [Z S (w)] = e(w, S). A straightforward analysis, and identical to that presented in the proof of Lemma A.1, shows that for e(w, S) ≥ 2εd(w) the value of Z S (w) computed in Algorithm 5 is with high probability a (1 ± ε) factor approximation of e(w, S). For e(w, S) < 2εd(w), the same analysis yields that with high probability, it holds that Z S (w) < 3εd(w). Hence, when e(w, S) < 2εd(w), the estimate Z S (w) is not necessarily within 1 ± ε factor of its expected value, and thus the error has to be accounted for differently. Next, we explain how to account for it.
Trivially, at least one among e(w, C), e(w, B -C), and e(w, V -B) is at least d(w)/3 > 2εd(w), for ε < 1/6. If e(w, C) ≥ d(w)/3, we charge each Z S (w) < 3εd(w) to the cost of E(w, C) paid by w∈C d(w). This incurrs an extra cost of at most (2 • 3εd(w))/(d(w)/3) = 18ε per an edge in E(w, C). The analogous analysis applies to the case e(w, V -B) ≥ d(w)/3 and e(w, B -C) ≥ d(w)/3. The only difference is that Z B-C (w) is divided by 2 in Algorithm 5, so for that case, the analysis yields a 36ε increased cost per edge.
Overall, this analysis implies that, with high probability, cost in COST-ESTIMATE is a (1 ± 37ε) multiplicative and 9ε|C| additive approximation of the cost of clustering B. The additive approximation comes from Lemma A.1 and the fact that 3 • IN-CLUSTER-COST-ESTIMATE(C) figures in the output of COST-ESTIMATE.
Lemma A.2 (Cost estimate of single-cluster + singletons). Let ε < 1/111. Given two node sets B and C ⊆ B, let cost * (B|C) be the cost of clustering B in which C is a single cluster and B -C are singletons, which is defined to be the number non-edges in C plus the number of edges in B with at least one endpoint in B -C.
Then, if COST-ESTIMATE(B, C) (Algorithm 5) outputs X, we have cost * (B|C) ≤ X ≤ (1 + 111ε)cost * (B|C) + 27ε|C|. Moreover, the algorithms run in O(|B| • log(n)/ε 3 ) time. Note that since (1 -37ε)cost * (B|C) -9ε|C| ≤ cost ≤ (1 + 37ε)cost * (B|C) + 9ε|C| and X = ( cost + 9ε|C|)/(1 -37ε), and ε < 1/111 we have that cost * (B|C) ≤ X ≤ (1 + 111ε)cost * (B|C) + 27ε|C|.
this section cite: []

Section: A.1.3. COST COMPARISON
Let the estimate that Algorithm 5 makes for cost * (B|C) be cost(B|C). We show that cost(B|C) is a good enough measure for choosing a C with low cost( * B|C).  Proof. First suppose that C ′ ⊂ C. Take a node u ∈ C \ C ′ , we know that cost(B|C
Lemma A.3. If cost * (B|C) ≥ |C|/4 then cost * (B|C) ≤ cost(B|C) ≤ (1 + 219ε)cost * (B|C).
) ≥ |C| -1 -d C (u) ≥ |C|/2 -d C (u). So |C|/2 ≤ d C (u). Now we have cost * (B|C ′ ) ≥ d C (u) ≥ |C|/2 > cost * (B|C). Similarly, suppose C ⊂ C ′ . Take a node u ∈ C ′ \ C. We know that cost * (B|C) ≥ d C (u), so d C (u) ≤ |C|/4. Moreover, cost * (B|C ′ ) ≥ |C| -1 -d C (u) > |C|/2. So in both cases cost * (B|C ′ ) > |C|/2 > cost * (B|C). Furthermore, by Lemma A.2 cost(B|C) ≤ (1 + 111ε)|C|/4 + 27ε|C| and by Lemma A.3 we have cost(B|C ′ ) ≥ (1 + 219ε)cost * (B|C) ≥ (1 + 219ε)|C|/2. So we have cost(B|C ′ ) < cost(B|C).
We use Algorithm 6 to develop Algorithm 7 that finds a dense cluster C v inside B v , where B v is the set of all the nodes that are assigned to pivot v. Recall that C t is the set of nodes in B v with degree at least t.
Algorithm 7 BREAK-CLUSTER 1: Input Set B v . 2: For any t > 0, let C t = {u ∈ B v , d(u) ≥ t}. 3: initialize t v = 0, C v = B v . 4: for t ∈ {1, (1 + ϵ), (1 + ϵ) 2 , . . . , (1 + ϵ) ⌈log n⌉ }: 5: C v ←COST-COMPARISON(B v , C t , C v ). 6: return C v .
For any C ⊆ B, let cost(B|C) be the cost of making C a cluster, and B -C singletons. This cost is equal to half the number of edges with exactly one endpoint in B, plus the number of edges with one endpoint in B -C and the other endpoint in B, plus the number of non-edges in C. In fact, cost(B|C) is cost * (B|C) plus half the number of edges that leave B. Theorem A.5. Let t * be the threshold among We prove the Theorem by induction: Suppose that for any j, t j is the threshold among 1, (1 + ε), . . . , (1 + ε) j such that cost * (B v |C tj ) is minimized, and the output of the for loop in Algorithm 7 for t ∈ {1, (1+ε), . . . , (1+ε
1, (1 + ε), . . . , (1 + ε) ⌈log n⌉ where cost(B v |C t * ) is minimized. If Algorithm 7 returns C t, then cost(B v |C t) ≤ (1 + 219ε)cost(B v |C t * ). Proof. First note that since cost(B v |C t ) is cost * (B v |C t ) plus half the number of edges that leave B v , for any t, t ′ we have cost(B v |C t ) -cost(B v |C t ′ ) = cost * (B v |C t ) -cost * (B v |C t ′ ),
) j } is C tj . Fix some i. Suppose that cost * (B v |C ti ) ≤ (1 + 219ε)cost * (B v |C ti ). Note that C ti+1 = COST-COMPARISON(B v , C (1+ε) i+1 , C ti ). We show that cost * (B v |C ti+1 ) ≤ (1 + 219ε)cost * (B v |C ti+1 ).
For ease of notation let C 1 = C ti , C 2 = C ti+1 , and
C ′ = C (1+ε) i+1 . So we have C 2 = arg min C∈{C1,C ′ } cost * (B v |C). Let C1 = C ti and C2 = C ti+1 . Assuming that cost * (B v | C1 ) ≤ (1 + 219ε)cost * (B v |C 1 ), we need to show that cost * (B v | C2 ) ≤ (1 + 219ε)cost * (B v |C 2 ), where C2 = COST-COMPARISON(B v , C ′ , C1 ). Note that C ′ ⊆ C1 . This is because C1 = C ti and C ′ = C (1+ε) i+1 where ti ≤ (1 + ε) i < (1 + ε) i+1
. We prove the rest of the theorem in the following cases.  If cost(B|C ′ ) < cost(B| C1 ), then we have C2 = C ′ . If C 2 = C ′ , then we are done. So assume that
(B|C ′ ) < cost * (B| C1 ) < (1 + 219ε)cost * (B|C 1 ). Since cost * (B|C ′ ) < (1 + 219ε)cost * (B|C ′ ) and C 2 ∈ {C ′ , C 1 }, we have cost * (B| C2 ) = cost * (B|C ′ ) ≤ (1 + 219ε)cost * (B|C 2 ). Case 2: cost * (B| C1 ) < | C1 |/4: Similar to case 1, in this case by Lemma A.4 we know that cost * (B| C1 ) < cost * (B|C ′ ) and cost(B| C1 ) < cost(B|C ′ ). So C2 = C1 . By induction hypothesis, we have cost * (B| C1 ) < (1 + 219ε)cost * (B|C 1 ). Since cost * (B| C1 ) < (1 + 219ε)cost * (B|C ′ ) and C 2 ∈ {C ′ , C 1 }, we have cost * (B| C2 ) = cost * (B| C1 ) ≤ (1 + 219ε)cost * (B|C 2 ).
C 2 = C 1 , which means that cost * (B|C 1 ) < cost * (B|C ′ ). So we have cost * (B|C ′ ) ≤ cost(B|C ′ ) ≤ cost(B| C1 ) ≤ (1 + 219ε)cost * (B|C 1 )
where the last inequality comes from the induction hypothesis. So we have
cost * (B| C2 ) ≤ (1 + 219ε)cost * (B|C 2 )
Similarly if cost(B|C ′ ) > cost(B| C1 ), then we have C2 = C1 . If C 2 = C 1 , then we are done by induction hypothsis. So assume that
C 2 = C ′ , which means that cost * (B|C 1 ) > cost * (B|C ′ ). So we have cost * (B| C1 ) ≤ cost(B| C1 ) < cost(B|C ′ ) ≤ (1 + 219ε)cost * (B|C ′ ). So cost * (B| C2 ) ≤ (1 + 219ε)cost * (B|C 2 ). Theorem A.6. Algorithm 7 runs in |B v | poly(log n, 1/ε) time.
Proof. Note that Algorithm 7 calls Algorithm 6 O(log n) times. Each run of Algorithm 6 takes |B v | poly(log(n), 1/ε) time: this is because estimating cost(B|C t ) by Algorithm 5 for a set C takes O(|C| log(n)/ε 3 ) time.
this section cite: []

Section: A.2. Dynamic version: UPDATE-CLUSTER(B v , u)
The previous section describes how to estimate the cost of clustering B in which C ⊆ B is a cluster, while B -C are singletons. When a node is inserted, we cannot afford to estimate these costs from scratch. Rather, we want to update the estimate based on the inserted node.
this section cite: []

Section: A.2.1. INSERTING A NODE INTO B -C
Assume that we already have an estimate of the cost of clustering B as guaranteed by Lemma A.2; let cost be that estimate. Assume that a node z is inserted in B -C. Hence, z will be a singleton. When cost was computed no edge incident to z was in the graph. Thus, updating the cost estimate is easy in this case: the cost estimate of clustering C together and B -C + z as singletons equals cost + d(z).
this section cite: []

Section: A.2.2. INSERTING A NODE INTO C
Assume that we already have an estimate of the cost of clustering B as guaranteed by Lemma A.2; let cost be that estimate. Assume that a node z is inserted in C. Let cost Let S C be the value of S at the end of IN-CLUSTER-COST-ESTIMATE(C) invocation. Let S C+z be a value of S corresponding to IN-CLUSTER-COST-ESTIMATE(C + z) that we aim to obtain. We initialize S C+z = S C , and then update S C+z as follow.
First, given definition of τ C on Line 2 of Algorithm 4, to compute S C+z we sample 5 • log(n)/ε 3 pairs (v, w) ∈ (C + z) × (C + z), and for each non-edge {v, w} we increment S C+z -the same as Algorithm 4 does.
• Purge from the database soft-deleted nodes.
• Assign to each node u a rank π(u) chosen uniformly at random from [0, 1].
• Initialize p(u) = u for all nodes u.
• Sort the nodes in the increasing order with respect to π.
• Insert the nodes, one by one, in this sorted order. The insertions are handled by Algorithm 3, except that a new π(u) value is not obtained within Algorithm 3, but is used the one computed in the first step of RECOMPUTE.
The nodes are processed in the ordering based on their π values for the following reasons. Given a node u, Algorithm 3 performs updates or exploration only when ranks are smaller than π(u). In particular, the node v defined in that algorithm is used only if π(v) ≤ π(u).
The running time of RECOMPUTE is a constant factor of the running time used to process the insertions. To see that, charge a recomputation running time to the εn 0 /6 most recent updates. Observe that this kind of charge is applied to each update during only one RECOMPUTE. Hence, at most n 0 + εn 0 /6 < 2n 0 insertions are charged to εn 0 /6 updates. So, each update is charged 12/ε insertions. Since an insertion takes poly(log n, 1/ε) amortized time, this additional charge also takes poly(log n, 1/ε) time per update.
this section cite: []

Section: C. Auxilary Lemmas
Lemma C.1. Let α < 1 be a constant. Let A be a clustering algorithm, where the probability of a node v being a pivot over all orderings is at most 1/d(v). The expected sum of the degrees of (A, α)-poor nodes is at most 4α 1+3/2•α 1-5/2•α times the total expected cost of A.
Proof. Let N low (v) be the neighbors of v with degree at most αd(v). Observe that N low (v) is independent of π and A.
The sum of poor-node degrees in a cluster. First, we show that for any π and v such that v is a pivot in A wrt to π, the sum of the degrees of the (A, α)-poor nodes in v's cluster is at most αd
(v) • min(3αd(v), |N low (v)|).
Fix a ordering π of nodes where v is a pivot in A. Note that if N low (v) = ∅, then there are no (A, α)-poor nodes in v's cluster, and the sum of degrees of all its (A, α)-poor nodes is zero. Hence, assume N low (v) ̸ = ∅. The size of v's cluster is at most 3αd(v) since the cluster size is at most three times the degree of an (A, α)-poor node. The latter is the case as an (A, α)-poor node u in a cluster C is not light by definition. Hence, d C (u) > |C|/3, which further implies |C| < 3d C (u) ≤ 3d(u) ≤ 3αd(v), where d(u) ≤ αd(v) be definition of (A, α)-poor nodes. So, the sum of degrees of all
(A, α)-poor nodes in v's cluster is at most αd(v) • min(3αd(v), |N low (v)|).
The expected sum of poor-node degrees in A. Let Q π be the set of (A, α)-poor nodes wrt the ordering π. Let P ivot v be the event that v is a pivot in A. We have
E π   u∈Q π d(u)   = π 1 n! v∈P π ref u∈Q π ∩C π v d(u) = v π : v∈P π ref 1 n! u∈Q π ∩C π v d(u) ≤ v π : v∈P π ref 1 n! • αd(v) • min(3αd(v), |N low (v)|) = v   αd(v) • min(3αd(v), |N low (v)|) • π : v∈P π ref 1 n!   = v Pr (P ivot v ) • αd(v) • min(3αd(v), |N low (v)|). Note that Pr (P ivot v ) ≤ 1 d(v) in A. Hence, the expected sum of degrees of (A, α)-poor nodes in A is at most v α • min(|N low (v)|, 3αd(v)) = v,N low (v)̸ =∅ α • min(|N low (v)|, 3αd(v)).(3)
Lower bounding the cost of A. Now, we compare the above value to the cost of A. (The analysis we provide applies to the cost of any clustering, even the one incurred by the optimal solution.) Fix a ordering π. All the costs below are defined wrt π, and we avoid the superscript π.
Recall that for a vertex w, C A (w) denotes the cluster of w in A. We define a cost function cost A to redistribute the cost cost A as follows
cost A (w) def = 1 2 cost A (w) + u∈C A (w) cost A (u) |C A (w)| .
In Now we compute E cost A (v) for v with N poor (v) ̸ = ∅. Note that we are not conditioning on v being a pivot here. Let t = |N poor (v)|, and let t ′ = min(t, 3αd(v)). We consider three cases based on |C A (v)|:
• Case |C A (v)| ≥ d(v) + t ′ 2 : then, there are at least t ′ 2 non-neighbors of v in C A (v) and hence cost A (w) ≥ t ′ 2 . Also, 2 • cost A (w) ≥ cost A (w) ≥ t ′ 2 .
•
Case |C A (v)| ≤ d(v) -t ′ 2 : then, at least t ′ 2 neighbors of v are outside of C A (v) and hence cost A (w) ≥ t ′ 2 . Again, 2 • cost A (w) ≥ cost A (w) ≥ t ′ 2 .
• Case
d(v) -t ′ 2 ≤ |C A (v)| ≤ d(v) + t ′ 2 : Let t ′′ = |N low (v) ∩ C A (v)|. The number of non-neighbors of a node u ∈ N poor (v) ∩ C A (v) is least |C A (v)| -d(u) ≥ d(v) -t ′ /2 -αd(v) ≥ (1 -5/2 • α)d(v), where we used the fact that t ′ /2 ≤ 3/2 • αd(v). So u∈C A (v) cost A (u) ≥ u∈N low (v)∩C A (v) cost A (u) ≥ t ′′ (1 -5/2 • α)d(v). Moreover, for each node u ∈ N low (v) \ C A (v), v incurs one unit of cost. Also, by definition, t ≥ |N low (v)|. Therefore, cost A (v) ≥ t -t ′′ . Recall that this case assumes |C A (v)| ≤ d(v) + t ′ /2 ≤ d(v) + 3/2 • αd(v) = (1 + 3/2 • α)d(v). This yields 2 • cost A (v) ≥ t -t ′′ + t ′′ (1 -5/2 • α)d(v) |C A (v)| ≥ t -t ′′ + t ′′ 1 -5/2 • α 1 + 3/2 • α .(4)
Observe that for x ∈ [0, 1] and t ≥ t ′′ we have (1 -x)t ≥ (1 -x)t ′′ , and hence t -t ′′ + t ′′ x ≥ tx. Thus, Equation ( 4) implies
2 • cost A (v) ≥ t 1 -5/2 • α 1 + 3/2 • α ≥ t ′ 1 -5/2 • α 1 + 3/2 • α .
Therefore, in any case,
cost A (v) ≥ 1 4 t ′ 1-5/2•α 1+3/2•α . So, the total cost of clustering A is at least v,Npoor(v)̸ =∅ 1 4 min(|N poor (v)|, 3αd(v)) 1 -5/2 • α 1 + 3/2 • α .
Note that the above lower bound is deterministic; in particular, it holds for any randomness used by A. Comparing this lower bound to Equation (3), we conclude that the expected sum of degrees of poor nodes is at most 4α 1+3/2•α 1-5/2•α times the (expected) total cost of A, as advertised by the claim.
Lemma C.2 (Cost of making light and heavy singletons). Consider a fixed ordering π and a clustering algorithm A. Let A ′ be the algorithm that first runs A and, after, makes a subset of A-light and A-heavy nodes singletons. The cost of A ′ is at most β+1 β-1 the cost of A.
Proof. For any algorithm B, let d B in (u), d B out (u) and dB in (u) be the number of neighbors of u inside its cluster, number of neighbors of u outside its cluster, and the number of non-neighbors of u inside its cluster, respectively. Note that since π is fixed, algorithm B determines the clusters. Let cost B denote the cost of algorithm B (with respect to π). We have 2 • cost B = u [d B out (u) + din (u)]. Let L be the set of A-light nodes that A ′ makes singletons, and H be the set of A-heavy nodes that A ′ makes singletons.
We can write the cost of algorithm A ′ as follows:
2cost A ′ = u∈L∪H d(u) + u / ∈L∪H [d A ′ out (u) + dA ′ in (u)] ≤ u∈L∪H [d(u) + d A in (u)] + u / ∈L∪H [d A out (u) + dA in (u)]
where the inequality comes from the fact that going from A ′ to A, any new cost on a node u / ∈ L ∪ H is due to an edge inside u's cluster in A that it attached to a node in L ∪ H. We prove that u∈L∪H [d(u)
+ d A in (u)] ≤ β+1 β-1 u∈L∪H [d A out (u) + dA in (u)]. Given this, we will have that cost A ′ ≤ β+1 β-1 cost A . If u is A-heavy, then d(u) ≥ β|C A (u)| ≥ βd A in (u). So d A out (u) ≥ (β -1)d A in (u), and hence d(u) + d A in (u) = d A out (u) + 2d A in (u) ≤ β+1 β-1 d A out (u).
If u is a light node, then we have that
d A in (u) ≤ |C A (u)|/3, so d A in (u) ≤ 1 2 dA in (u), and hence d(u) + d A in (u) = d A out (u) + 2d A in (u) ≤ d A out (u) + dA in (u).
Lemma C.3. Consider a fixed ordering π and a clustering algorithm A, and consider a A-bad cluster C π A . Then, its cost is at least
2 3 (1 -γ)|C π A | 2 .
Proof. Recall that a bad cluster does not have any (ref, α)-poor nodes. So, all the bad nodes in a bad cluster are either light or heavy. Let C = C π ref .
For a light node u ∈ C, the cost of u is at least 2|C|/3, since d C (u) ≤ |C|/3. For a heavy node u ∈ C, the cost of u is at least (β -1)|C|/2, since u has at least (β -1)|C| neighbors outside C. So the cost of any bad node in C is at least min {(β -1)/2, 2/3} • |C|. There are at least (1 -γ)|C| many bad nodes in C. Thus, the total cost of C is at least min {
(β -1)/2, 2/3} • (1 -γ) • |C| 2 . Since β ≥ 4, min {(β -1)/2, 2/3} = 2/3. D. Ommited proofs D.1. Proof of Lemma 3.4 It holds that Pr π(v) ≤ L ′ /d(v) v ∈ P π ref = 1 -Pr π(v) > L ′ /d(v) v ∈ P π
ref , and we upper-bound the latter probability:
Pr π(v) > L ′ /d(v) v ∈ P π ref = Pr π(v) > L ′ /d(v) and v ∈ P π ref Pr v ∈ P π ref = d(v) • Pr (π(v) > L ′ /d(v) and ∀u ∈ N (v) : π(u) > π(v)) ≤ d(v) • 1 - L ′ d(v) d(v)+1 ≤ d(v) • e -L ′ .
In the derivation, we used the fact that the entries of π are chosen independently and uniformly from range [0, 1]. Now note
that d(v)e -L ′ ≤ ne c log n < n -c+1 , so Hence, Pr π(v) ≤ L ′ /d(v) v ∈ P π ref ≥ 1 -n -(c-
1) , as desired. D.2. Proof of Lemma 3.5 Let L ′ = L/2β. Let A be the set of first (1 -ϵ)C ref (v)[good] good nodes in C ref (v) and let B be the set of last ϵC ref (v)[good] good nodes in C ref (v). So C ref (v)[good] = A ∪ B.
First we introduce some notation: For two nodes u and w where w is inserted before u, the degree of a vertex w at the time where u is inserted is denoted by d (u) (w). Recall that the current degree of w is denoted by d(w), and d(w) ≥ d (u) (w).
Next, note that if the pivot v comes after all the nodes in A, then since v is a pivot in REFERENCE CLUSTERING by Lemma 3.4 we have that whp π(v) ≤ L ′ /d(v) ≤ L/d(v) and so it scans its neighborhood and invokes EXPLORE(v) which assigns all these nodes to v as their pivot. So suppose that v comes before the nodes in B.
Now consider a good node u ∈ C ref (v) [good] and suppose that u comes after v in the dynamic ordering. Since u is not heavy, we have d(u
) ≤ β|C ref (v)| ≤ βd(v).
We compute the probability that u is assigned to v's cluster and also that u invokes EXPLORE(v). In particular, we want to lower-bound the probability that π(v) < π(u)
≤ L d (u) (u) and d (u) (v) ≤ L π(u) , conditioned on π(v) < π(u), i.e., in REFERENCE CLUSTERING u is in C v .
First note that if d(v) ≤ L and d(u) ≤ L, then both these conditions are satisfied. So we assume that max(d(u), d(v)) > L > L ′ β.
Next, recall that since v is a pivot in the REFERENCE CLUSTERING, by Lemma 3.4, we have that π(v) ≤ L ′ /d(v) holds with probability 1 -1/n c-1 for a large constant c by Lemma 3.4. Moreover, since
d (u) (v) ≤ d(v), the probability that d (u) (v) ≤ L π(u) is at least the probability that d(v) ≤ L π(u) . Similarly, the probability that π(u) ≤ L d (u) (u) is at least the probability that π(u) ≤ L d(u) as d (u) (u) ≤ d(u). So: Pr (u invokes EXPLORE(v) | u ∈ C ref (v)) = Pr π(v) < π(u) ≤ L d (u) (u)
and
d (u) (v) ≤ L π(u) | π(v) < π(u) ≤ 1 ≥ Pr π(v) < π(u) ≤ L d(u)
and
d(v) ≤ L π(u) | π(v) < π(u) ≤ 1 = Pr π(v) < π(u) ≤ L max(d(u), d(v)) | π(v) < π(u) ≤ 1 = Pr π(v) < π(u) ≤ L max(d(u), d(v)) | π(v) < π(u) ≤ 1, π(v) ≤ L ′ /d(v) • Pr (π(v) ≤ L ′ /d(v)) + Pr π(v) < π(u) ≤ L max(d(u), d(v)) | π(v) < π(u) ≤ 1, π(v) > L ′ /d(v) • Pr (π(v) > L ′ /d(v)) ≥ Pr L ′ d(v) < π(u) ≤ L max(d(u), d(v)) | L ′ d(v) < π(u) ≤ 1 • 1 -n -c+1 If d(v) ≤ d(u), then since d(u) ≤ β|C ref (v)| ≤ βd(v) and L ′ β = L/2, we have Pr (u invokes EXPLORE(v) | u ∈ C ref (v)) ≥ Pr L ′ β d(u) < π(u) ≤ L d(u) | L ′ β d(u) < π(u) ≤ 1 • 1 -n -c+1 = L -L ′ β d(u) -L ′ β • 1 -n -c+1 ≥ L -L ′ β d(u) • 1 -n -c+1 ≥ L/2 d(u) • 1 -n -c+1 ≥ αL/2 d(u) • 1 -n -c+1
Nnote that d(u) -L ′ β > 0 since we assume that max(d(u), d(v)) > L > L ′ β.
If d(v) ≥ d(u), we use the fact that since u is not (α, A)-poor, we have d(u) ≥ αd(v), and so
Pr (u invokes EXPLORE(v) | u ∈ C ref (v)) ≥ Pr L ′ d(v) < π(u) ≤ L d(v) | L ′ d(v) < π(u) ≤ 1 • 1 -n -c+1 ≥ L -L ′ d(v) -L ′ • 1 -n -c+1 ≥ L/2 d(v) • 1 -n -c+1 ≥ αL/2 d(u) • 1 -n -c+1
The above inequality holds for each u ∈ B. We show that with high probability, a node in B invokes EXPLORE(v). First recall that C ref (v) is a good cluster, so |C ref (v)[good]| ≥ γ|C| ≥ γd(u)/β. Thus we have |B| ≥ γϵd(u)/β. Let t = ϵγα 4β . Since (1 -n -c+1 ) > 1/2, we have: Pr (u invokes EXPLORE(v)) ≥ tL |B| So the probability that none of u ∈ B invokes EXPLORE(v) is at most (1 -tL |B| ) |B| ≤ e -tL ≤ n -c , where we use L ≥ 4cβ
ϵγα log n = c t log n. So with probability 1 -n -c , EXPLORE(v) is invoked by a node in B, and so all the nodes in A are going to be assigned to v as their pivot.
this section cite: []

Section: D.3. Proof of Lemma 3.7
Let k = C * and let t be the smallest power of (1 + ϵ) no smaller than 2k 3 . So 2k 3 ≤ t ≤ 2k(1+ϵ) 3 . Let S = C * \ C t , and let T = C t \ C * . We refer to the clustering where C t is one cluster and all B v \ C t is singleton as C 1 and the clustering where C * is one cluster and all B v \ C * is singleton as C * .
The cost of C 1 and C * differ in edges and non-edges with one endpoint in S or T . They share any other cost associated to an edge or non-edge that is in disagreement with the clustering. So we only need to compare the excess cost that is not part of this common cost.
By the cost of a node u ∈ S ∪ T , we mean the number of v / ∈ S ∪ T where uv is in disagreement with the clustering plus half the number of v ∈ S ∪ T where uv is in disagreement with the clustering. We let cost(u) C denote the cost of u in a clustering C ∈ {C 1 , C * }. Let cost(C). Note that the total cost of C is the sum of the individual costs of u ∈ S ∪ T , plus the common cost between C 1 and C * .
First, assume that |T | ≤ k. We show that cost(C 1 ) ≤ 4 1-2ϵ cost(C * ). Consider a node u ∈ S. In C 1 , the cost of u is at most t since each node in S has degree at most t, and u is a singleton in C 1 . In C 2 , the cost of u is at least k-t 2 since there are at least k -t non-edges attached to u in C * . Note that for t ≤ 2k(1+ϵ) 3 , we have cost(u) C1 ≤ t ≤ 4 1-2ϵ • (k -t)/2 ≤ 4 1-2ϵ cost(u) C2 . To compute the cost of u ∈ T , for a set C recall that d C (u) is the degree of u inside C. So d(u) -d Ct (u) is the number of edges attached to u outside C t . In C 1 the cost of u is the number of nodes attached to u outside of C t plus the number of nodes v not attached to u inside C t \ T , plus half of the number of nodes v ∈ T that are not attached to u. So
cost(u) C1 = d(u) -d Ct (u) + (|C t | -|T | -d Ct\T ) + (|T | -1 -d T (u))/2 ≤ d(u) -d Ct (u) + (|C t | -|T | -d Ct\T ) + (|T | -1 -d T (u)) = |C t | + d(u) -1 -2d Ct (u) ≤ |T | + |C * ∩ C t | + d(u) -2d T (u) ≤ 2k + t -2d T (u)
Meanwhile we have that there are at least d(u) -d T (u) nodes attached to u outside of T , so we have
cost(u) C2 = d(u) -d T (u) + d T (u)/2 ≥ t -d T (u) 2
Note that when for t ≥ 2k/3, we have 2k + t -2d T (u) ≤ 4( t-d T (u) 2 ), so cost(u) C1 ≤ 4cost(u) C2 . And so cost(C 1 ) ≤ 4 1-2ϵ cost(C * ).  Correlation clustering cost for cit-hepth graph Density DA SP 253.36 36.75 31.91 114.87 43.08 29.69 69.74 50.77 26.36 52.17 49.27 23.36 42.25 41.23 25.14 Table 3. Running time comparison on SNAP datasets
this section cite: []

Section: 
However, the (v, w) pairs sampled by Algorithm 4 to compute S C are simple from C × C, while for S C+z we would like each pair to be sampled from (C + z) × (C + z). So, second, to account for that, we resample some of the pairs used in computing S C . This also implies that while our algorithm estimates the cost of C by Algorithm 4, it also stores in an array all the {v, w} pairs it sampled within the for-loop. Let that array of samples be called A C .
When computing S C , a pair {v, w} is sampled with probability p C = 1/ |C| 2 . However, when computing S C+z , that same pair is sampled with probability p C+1 = 1/ |C|+1 2 . Let q = p C /p C+1 So, in A C , we resample a pair {v, w} with probability 1-q, and otherwise, with probability q, {v, w} is not resampled. With this process, we have that {v, w} ∈ C ×C remains unchanged with probability 1/ |C|+1 2 , as desired.
If a pair is resampled, then the new pair is {z, u}, where u is a node from C chosen uniformly at random. So, we have
Pr ({z, u} is sampled) = 1 |C| • 1 - |C| 2 |C|+1 2 = 1 |C| • 2 |C| + 1 = 1 |C|+1 2
, as we aim to achieve. But how many pairs are resampled? How does one choose which pairs to resample in poly(log n, 1/ε) time?
The expected number of resampled pairs is
E [|A C | • (1 -q)] = τ C • 2 |C| + 1 = O(log(n)/ε 3 ).
Hence, by a direct application of the Chernoff bound, with high probability, the number of resampled pairs from A C is poly(log n, 1/ε).
The final piece elaborates on efficiently finding pairs to resample. Consider a process that iterates over the elements in A C and resamples each with probability 1 -q. A downside is that this approach takes Θ(|C|) time, which is too slow for our goal. Instead, we observe that the index of the first element resampled in A C is drawn from the geometric distribution with parameter 1 -q. This observation leads to the following efficient procedure for resampling elements from A C :
1. Initialize i = 0.
2. Repeat while i ≤ |A C |:
• sample an index j from the Geometric distribution with parameter 1 -q;
• i = i + j; • if i ≤ |A C |, resample the i-th {v, w} pair in A C .
This latter approach enables us to spend time proportional to the number of resampled pairs -as opposed to |A C | -which we know is poly(log n, 1/ε) with high probability.
this section cite: []

Section: Lemma A.7 (Dynamic cost estimate of single-cluster + singletons).
There exists an algorithm that, on a node insertion, updates COST-ESTIMATE(B, C) in poly(log n, 1/ε) time with high probability. The approximation guarantees are the same as those stated in Lemma A.2.
this section cite: []

Section: A.2.3. UPDATE-CLUSTER(B v , u)
Subroutine UPDATE-CLUSTER(B v , u) is essentially BREAK-CLUSTER with the costs of C and B -C updated dynamically as explained in Appendices A.2.1 and A.2.2.
More precisely, when BREAK-CLUSTER is invoked, we store all the estimates cost(B v |C t ), the sets B v and C t , for t = 1, (1 + ϵ), . . . , (1 + ϵ) log n . Then, we use Lemma A.7 to update these costs when u is inserted into B v in poly(log n, ε) time. Note that if d(u) ≥ t, then u joins C t , and otherwise u joins B v \C t . So each run of Algorithm 6 takes poly(log n, 1/ε), and so UPDATE-COST(B v , u) works in poly(log n, 1/ε) time.
this section cite: []

Section: B. Recompute
Let n 0 be the number of nodes in the graph just after the last recompute. After εn 0 /6 updates, we perform a recomputation. Our RECOMPUTE procedure is as follows:
Now suppose that |T | > k. Let cluster C 2 be the clustering where all the nodes in B v are singleton. In fact, C 2 is the clustering for when t = n. In this case, the cost of C 2 is the cost of C * plus the number of edges between the nodes in C * . We have cost(C 2 ) ≤ cost(C * ) + k 2 /2. Now to bound cost(C * ), the cost of cost(C * ) is at least the number of edges with at least one endpoint in T . This value is at least t|T |/2. So cost(C * ) ≥ t|T |/2. Now since |T | ≥ k and t ≥ 2k/3, we have
3 • t|T |/2 ≥ k 2 /2, so cost(C 2 ) ≤ cost(C * ) + k 2 /2 ≤ cost(C * ) + 3 • t|T |/2 ≤ 4cost(C * ).
this section cite: []

Section: E. Running Time Analysis
In Appendix B, we show that our RECOMPUTE subroutine takes poly(log n, 1/ε) amortized time per update. Next, we analyze the running time of Algorithm 3 and prove Theorem 2.2.
Proof. We first introduce some notation: Let the degree of a node w at the time of the insertion of node u be d (u) (w). Recall that d(w) is the current degree of w and d(w
) ≥ d (u) (w).
First note that if π(u) > L/d (u) (u), then the running time is O(log n) plus the running time of UPDATE-CLUSTER, which is in poly(log n, 1 ε ) ≤ poly(log T, 1 ε ) time; see Appendix A.2.3 for details. Next, consider the case where π(u) ≤ L/d (u) (u). For each such u, we set aside a budget of L/π(u) • poly(log n, 1/ε). In this case, the algorithm scans all the neighbors of u, which takes
d (u) (u) < L/π(u) time.
Next, we analyze the running time of EXPLORE. For this, consider a pivot node v. Note that when we run EXPLORE on a node v, we scan all its neighbors, and if a neighbor w is also a pivot, we scan all of the neighbors of w as well and remove w from being a pivot. Observe that once a node is removed from being a pivot, it can never be a pivot until the next recompute, when nodes get new ranks. We pay for scanning the neighbors of w from the budget of w, not v. This way, when a node v runs EXPLORE, it only needs to pay at most d(v) from its budget. Now note that a node v might run EXPLORE multiple times. In particular, v runs EXPLORE either when we are processing v, or when a neighbor of v, say u is being processed, and d (u) (v) < L/π(u). In the first case, we pay the cost of EXPLORE from v's budget. In the second case, we pay the cost from u's budget. Note that for a pivot node v, not only we have
d (v) (v) < L/π(v), but also we have d(v) < L/π(v) (see Lemma 3.4) whp.
So, in total, a pivot node v only spends a budget of at most d(v) • poly(log n, 1/ε) < L/π(v) • poly(log n, 1/ε): it spends d (v) (v) ≤ d(v) for scanning all its neighbors when it is being inserted, d (v) (v) ≤ d(v) when it runs EXPLORE, at most d(v) (possibly) when another pivot w runs EXPLORE and removes v from being a pivot, and d (v) (v) poly(log n, 1/ε) ≤ d(v) poly(log n, 1/ε) for running BREAK-CLUSTER (Theorem A.6).
A non-pivot node u spends at most 2L/π(u): once for scanning all its neighbors and once for paying for EXPLORE for its pivot v.
Since L = O(log n), each node u spends poly(log n, 1/ε) + O(log n/π(u)). Given that π(u) is chosen uniformly at random from the range [0, 1], in expectation, a node spends poly(log n, 1/ε) ≤ poly(log T, 1/ε) running time.
this section cite: []

Section: F. Experiments-continued F.1. Approximation guarantee results
Below we include the approximation guarantee on the three remaining SNAP graphs.
this section cite: []

Section: F.2. Running Time
Even though theoretically, our running time (and DYNAMIC-AGREEMENT running time) is faster than REFERENCE CLUSTERING, this advantage only appears in very big graphs. This is because both algorithms have a lot of bookkeeping, which means that the constant behind the O(log n) running time is rather big. Nevertheless, we show in Table 2 and Table 3 that SPARSE-PIVOT is faster than DYNAMIC-AGREEMENT.
this section cite: []

Section: References
Ref_id:b0 Title: Aggregating inconsistent information: ranking and clustering Year: (2008)
Ref_id:b1 Title: Sublinear time and space algorithms for correlation clustering via sparse-dense decompositions Year: (2022)
Ref_id:b2 Title: Correlation clustering Year: (2004)
Ref_id:b3 Title: Fully dynamic maximal independent set with polylogarithmic update time Year: (2019)
Ref_id:b4 Title: Almost 3-approximate correlation clustering in constant rounds Year: (2022)
Ref_id:b5 Title: Singlepass streaming algorithms for correlation clustering Year: (2023)
Ref_id:b6 Title: Correlation clustering: from theory to practice Year: (2014)
Ref_id:b7 Title: A parallel algorithm for (3 + ε)-approximate correlation clustering Year: (2022)
Ref_id:b8 Title: Understanding the cluster lp for correlation clustering Year: (2024)
Ref_id:b9 Title: Single-Pass Pivot Algorithm for Correlation Clustering. Keep it simple! In NeurIPS Year: (2023)
Ref_id:b10 Title: Clustering with qualitative information Year: (2005)
Ref_id:b11 Title: Near optimal lp rounding algorithm for correlation clustering on complete and complete k-partite graphs Year: (2015)
Ref_id:b12 Title: Fully dynamic maximal independent set in expected poly-log update time Year: (2019)
Ref_id:b13 Title: Correlation clustering in mapreduce Year: (2014)
Ref_id:b14 Title: Dynamic correlation clustering in sublinear update time Year: ()
Ref_id:b15 Title: Correlation clustering in constant many parallel rounds Year: (2021)
Ref_id:b16 Title: Correlation clustering with sherali-adams Year: (2022)
Ref_id:b17 Title: Handling correlated rounding error via preclustering: A 1.73approximation for correlation clustering Year: (2023)
Ref_id:b18 Title: Pruned pivot: Correlation clustering algorithm for dynamic, parallel, and local computation models Year: (2024)
Ref_id:b19 Title: Correlation clustering in general weighted graphs Year: (2006)
Ref_id:b20 Title: Uci machine learning repository Year: (2017)
Ref_id:b21 Title: Snap datasets: Stanford large network dataset collection Year: (2014)
Ref_id:b22 Title: Parallel correlation clustering on big graphs Year: (2015)
Ref_id:b23 Title: On the calibration of sensor arrays for pattern recognition using the minimal number of experiments Year: (2014)
Ref_id:b24 Title: Chemical gas sensor drift compensation using classifier ensembles Year: (2012)
