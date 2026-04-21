Title: An Improved Clique-Picking Algorithm for Counting Markov Equivalent DAGs via Super Cliques Transfer
Abstract: Efficiently counting Markov equivalent directed acyclic graphs (DAGs) is crucial in graphical causal analysis. Wienöbst et al. (2023) introduced a polynomial-time algorithm, known as the Clique-Picking algorithm, to count the number of Markov equivalent DAGs for a given completed partially directed acyclic graph (CPDAG). This algorithm iteratively selects a root clique, determines fixed orientations with outgoing edges from the clique, and generates the unresolved undirected connected components (UCCGs). In this work, we propose a more efficient approach to UCCG generation by utilizing previously computed results for different root cliques. Our method introduces the concept of super cliques within rooted clique trees, enabling their efficient transfer between trees with different root cliques. The proposed algorithm effectively reduces the computational complexity of the Clique-Picking method, particularly when the number of cliques is substantially smaller than the number of vertices and edges.

Section: Introduction
Directed acyclic graphs (DAGs) are widely used to represent multivariate causal structures across diverse fields, including epidemiology, biology, and economics (Pearl, 1988;Pingault et al., 2018). In a DAG, nodes represent variables, and directed edges denote causal relationships (Koller & Friedman, 2009;Spirtes et al., 2001). Under the Markov condition and faithfulness assumption, the causal structure can be inferred from statistical data to identify a DAG. The d-separation properties of the identified DAG correspond to the conditional independencies observed in the data (Pearl, 2009;Verma & Pearl, 1992;Spirtes et al., 2001). However, observational data alone is often insufficient to uniquely determine the true DAG. Instead, it can identify a set of DAGs that encode the same conditional independencies, collectively known as a Markov equivalence class (MEC). This limitation has driven extensive research into learning MECs from both observational and interventional data (Perlman, 2001;Geiger & Heckerman, 2002;Chickering, 2002;Castelo & Perlman, 2004;Maathuis et al., 2009).
A MEC can be uniquely represented by an essential graph (Andersson et al., 1997), also known as the completed partially directed acyclic graph (CPDAG). They use both directed and undirected edges to represent causal relationships that are consistent across all DAGs in the equivalence class, with directed edges indicating fixed causal directions and undirected edges reflecting ambiguous dependencies unresolved by conditional independence constraints. The size of a MEC, defined as the number of DAGs within the class, plays a critical role in the design of causal intervention experiments (He & Geng, 2008) and average causal effect estimation (Maathuis et al., 2009).
Exhaustive search for all Markov equivalent DAGs is only computationally feasible for small graphs (Madigan et al., 1996;Gillispie & Perlman, 2002). Generally, the size of a MEC grows superexponentially in the number of its vertices. He et al. (2015) addressed the counting challenge by introducing five special MECs with explicit size formulas, and exploiting recursive partitioning into the respective subclasses for efficient counting. A modified approach by Ghassami et al. (2019) leverages the clique tree representation to decompose the essential graph into smaller components. More recently, dynamic programming enhancements (Ganian et al., 2022) and iterative methods over possible interventional essential graphs (AhmadiTeshnizi et al., 2020) have been proposed.
Notably, Wienöbst et al. (2023) introduces the Clique-Picking (CP) algorithm, which is a polynomial-time algorithm for determining the size of a MEC. This method partitions the MEC into subclasses by fixing a clique as a root and avoids overcounting using minimal separators derived from the clique tree representation. However, for a given chordal graph G = (V, E), the algorithm needs to recursively select a clique as root, introduce outgoing edges from the root, and determine smaller undirected connected components (UCCG) from the resulting graph. Suppose there are m maximal cliques in G, this intensive process has a cost of O(m(|V | + |E|)). Moreover, this process has to be repeated during the recursive function calls, until the reduced UCCG only contains a single maximal clique.
Fortunately, an improvement is feasible because there are considerable structure overlaps when different cliques are selected as the roots. We propose a novel approach for this purpose. Our contributions are summarized as follows.
1. We introduce higher level structures called super clique and super residual for a clique tree in Section 5. For a chordal graph G with some clique selected as the root, we show the UCCGs can be easily identified from the super residuals.
2. When two different cliques (K i , K j ) are selected the root, we show structure changes can be easily identified for the corresponding super cliques and super residuals. Hence, when the UCCGs are known for K i as the root, we can efficiently identify UCCGs for the case when the other K j is the root. This leads to the super clique transfer operation in Section 6.
3. The above techniques lead to the Super Cliques Transfer Algorithm in Section 4. Overall, our procedure of UCCG identification for all root cliques has a reduced cost of O(m 2 ).
4. To provide a solid theoretical foundation for our algorithm, we characterize super cliques and super residuals from two distinct perspectives: the clique-rooted tree perspective in the main text and the clique sequence perspective in the Appendix. The former offers an intuitive understanding, while the latter provides a more fundamental framework that facilitates theoretical proofs.
The rest of the paper is as follows. Section 2 reviews the concepts for Markov equivalent DAGs and clique rooted trees. Section 3 reviews the Clique-Picking algorithm of Wienöbst et al. (2023) at a high level. Our proposed algorithm and its detailed operations will be presented in Sections 4-6. Section 7 presents the experimental results.
this section cite: ['b17', 'b20', 'b13', 'b21', 'b18', 'b23', 'b21', 'b19', 'b7', 'b4', 'b3', 'b15', 'b1', 'b11', 'b15', 'b16', 'b9', 'b12', 'b8', 'b6', 'b0', 'b24', 'b24']

Section: Preliminaries

this section cite: []

Section: Markov equivalent DAGs
A graph G = (V, E) is a tuple consisting of a vertex set V = {v 1 , • • • , v n } and an edge set E. An edge v i -v j is undirected if (v i , v j ), (v j , v i ) ∈ E and directed v i → v j if (v i , v j ) ∈ E and (v j , v i ) / ∈ E. We denote the induced subgraph of G on a set C ⊆ V by G[C]
, which only keeps the vertices in C and the egdes connecting them. A directed acyclic graph (DAG) is a directed graph without any directed cycle. A topological ordering of a DAG is a linear ordering of its vertices such that for every directed edge v i → v j , vertex v i appears before vertex v j in the ordering.
The skeleton of a graph G is the undirected graph formed by ignoring the edge directions in G, while retaining its vertices and edges. An induced subgraph of the form v 1 → v 2 ← v 3 is a v-structure.
A Markov equivalence class (MEC) is the set of all DAGs that encode the same conditional independence relations among the variables. Verma & Pearl (1990) state that two DAGs are markov equivalent if and only if they share the same skeleton and v-structures. Furthermore, Andersson et al. (1997) show that a MEC can be uniquely represented by a completed partially directed acyclic graph (CPDAG), denoted as G * , which is the union of all DAGs in the equivalence class. An undirected graph is chordal if every cycle of length greater than three has a chord, i.e., an edge connecting two nonconsecutive vertices in the cycle. Each undirected connected component of a CPDAG is a connected chordal graph, referred to as a UCCG. In particular, each UCCG is itself a CPDAG representing a MEC.
Let Size(G * ) denote the size of the Markov equivalence class represented by a CPDAG G * . The value of Size(G * ) equals the product of the number of Markov equivalent DAGs for each UCCG of G * (Andersson et al., 1997):
Size(G * ) = UCCG G in G * Size(G).
However, the above equation is not directly applicable in general to compute Size(G * ). This is because the value of Size(G) can grow superexponentially with respect to its vertex number |V |. It is essential to develop an efficient approach for the computation of Size(G).
this section cite: ['b22', 'b1', 'b1']

Section: Clique Rooted Trees
In a graph, a clique is a set of pairwise adjacent vertices. For a UCCG G, we denote the set of all its maximal cliques as K G = {K 1 , . . . , K m }. For example, the chordal graph G in Figure 1(a) contains seven cliques: K 1 = {a, b, c}, K 2 = {b, c, d}, K 3 = {b, e}, K 4 = {e, f }, K 5 = {b, g, j}, K 6 = {b, g, i} and K 7 = {b, h, j}. The maximal cliques in a chordal graph G can be ordered to satisfy the running intersection property (RIP, Blair & Peyton, 1993).
Definition 2.1. (Running intersection property) A clique sequence, K 1 , K 2 , . . . , K m , has the running intersection property (RIP) if for each clique K p (with p = 2, . . . , m), there exists a clique K t for some t ∈ {1, . . . , p -1}, such that  From any sequence of cliques satisfying RIP, we construct a rooted tree on K G by making each clique K p adjacent to a "parent" clique K t in (1). This tree has the first clique K 1 in the sequence as its root, and is denoted as T K1 . For the example in Figure 1(a), K 1 , K 2 , K 3 , K 4 , K 5 , K 6 , K 7 is an RIP sequence. The corresponding rooted tree T K1 is shown in Figure 1(c).
K p ∩ (K 1 ∪ K 2 ∪ • • • ∪ K p-1 ) ⊂ K t .(1)
Given a UCCG G, a clique tree T K1 with some root clique K 1 can be generated via the MCS algorithm (Blair & Peyton, 1993). For each clique K p , its separator S p is as
S p = K p ∩ (K 1 ∪ • • • ∪ K p-1 ) = K p ∩ K t ,
and its residual R p is defined as R p = K p \S p . For the root clique, we simply set S 1 = ∅ and R 1 = K 1 . For the rooted clique tree T K1 , the collection of all separators is denoted as Sep(K 1 ), and all residuals as Res(K 1 ).
this section cite: ['b2']

Section: The Clique-Picking Algorithm
We now review the Clique-Picking algorithm proposed by Wienöbst et al. (2023) and highlight, at the end of this section, the specific part where our super clique approach can provide improvements. Wienöbst et al. (2023) exploited the fact that each DAG within MEC can be represented by topological vertex orderings, and a maximal clique can be selected as the prefix of an ordering. In this way, the Markov equivalent DAGs can be divided into small groups for more efficient computation. Wienöbst et al. (2023) introduced several concepts to formalize the idea. Suppose K be a clique in G and selected as the root. Let π(K) be a permuted ordering of the vertices in K, and consider all topological orderings of G that start with π(K). The π(K)-orientation of G, denoted G π(K) , is the union of all DAGs within the MEC represented by G that have topological orderings beginning with π(K). Then,
Algorithm 1 Function CP-Count(•) Input: A UCCG G Output: Size(G) 1: Generate a rooted clique tree of G; 2: Generate C G (K p ) for each K p ∈ K G
, which is selected as the root for clique tree T Kp ; 3: Evaluate Size(J) for all UCCG J inside C G (K p ) by recursively calling CP-Count(J); 4: Compute Size(G) in (2).
denote by C G (π(K)) the undirected connected components of G π(K) [V \ K]. Furthermore, let G K denote the union of π(K)-orientations of G over all permutation π. That is, G K = π G π(K) . We also denote C G (K) as the undirected connected components of G K [V \ K].
For the graph in Figure 1(a), suppose K 1 = {a, b, c} is picked as the prefix of the ordering, then the corresponding graphs G K1 is shown in Figure 1(b). We can see, by picking K 1 as the root, we introduce outgoing edges from K 1 in G K1 , compared with the original undirected G in Figure 1(a). For G K1 , we have the undirected connected components C Wienöbst et al. (2023) show that the size of the Markov equivalence class represented by G K can be calculated by:
G (K 1 ) = {G[e], G[d], G[f ], G[g, h, i, j]}.
Size(G K ) = |K|! • J∈C G (K) Size(J ). It is tempting to select each K in K G , compute Size(G K )
and sum all these values to get Size(G) for a UCCG G. However, this will count some DAGs multiple times, as a DAG can be represented by multiple topological orderings with different cliques as the root. To resolve this issue, Algorithm 2 Super Cliques Transfer Algorithm Input: A UCCG G and a rooted clique tree T K 1 of KG Output: CG(K1), CG(K2), . . . , CG(Km).
1: L
(1) , CG(K1), Sep(K1), Res(K1) ← SC-Create-Op (G, T K 1 ) via Algorithm 3; 2: for i = 2 to m do 3: Kt ← The parent clique of Ki in T K 1 ; 4: Initialize Sep(Ki) ← Sep(Kt) and Res(Ki) ← Res(Kt); 5: Update Si ← ∅ and St ← Ki ∩ Kt in Sep(Ki); 6: Update Ri ← Ki and Rt ← Kt \ (Ki ∩ Kt) in Res(Ki); 7: Run Algorithm 4 to get CG(Ki), L (i) ← SC-Trans-Op(CG(Kt), L (t) , T K t , Sep(Ki), Res(Ki)); 8: Get T K i by reversing the edge "Kt → Ki" in T K t to "Ki → Kt"; 9: end for Wienöbst et al. (2023) further introduced the correct iterative formula
(G) = m p=1 ϕ(Kp, FP(Kp, T K 1 )) • J∈C G (Kp) Size(J), (2)
where ϕ(•) is a corrected multiplicative factor to avoid overcounting. The formal definition of the above ϕ(•) is provided in Section 4.3 of Wienöbst et al. (2023), where the authors discuss it in detail. It is important to note Step 2 of Algorithm 1 can be further improved to achieve greater efficiency. This is because there are considerable structure overlap for C G (K) with different K ∈ K G selected as the root. For example, we can easily see that, when K 3 is selected as the root, we have
C G (K 3 ) = {G[a, c, d], G[f ], G[g, h, i, j]}. It is clear that the undirected connected components G[f ] and G[g, h, i, j] appear in both C G (K 1 ) and C G (K 3 ).
To address this redundancy, we will introduce our super clique transfer algorithm in the next section. With the proposed algorithm, once we get C G (K 1 ), we can more efficiently compute and derive all the other C G (K p ) with p = 2, . . . , m. The reduced cost is O(m 2 ) for Step 2 of Algorithm 1.
this section cite: ['b24', 'b24', 'b24', 'b24', 'b24']

Section: The Super Cliques Transfer Algorithm
Our main contribution is a novel algorithm that reduces
Algorithm 3 SC-Create-Op Input: A UCCG G, a rooted clique tree T K 1 of KG. Output: L (1) , CG(K1), Sep(K1),
and Res(K1). 1: Initialize CG(K1) ← {}; 2: Sep(K1) ← the set of separators S1, . . . , Sm; 3: Res(K1) ← the set of residuals R1, . . . , Rm; 4: Based on Sep(K1), get the set of super cliques of T K 1 and denote it as L (1) ; 5: for SK (1) p + in L (1) do 6: Obtain SR (1) p + for SK (1) p + based on Res(K1); 7: CG(K1) ← CG(K1) ∪ {G[SR
p + ]}. 8: end for the computation cost of Step 2 of Algorithm 1. The main idea is to group the cliques in a rooted clique tree into higher level structure, called super cliques. We connect the super cliques with the UCCG, and develop an efficient super clique transfer algorithm to obtain the UCCGs when different cliques are selected as the root.
The proposed approach, referred to as the Super Cliques Transfer (SC-Trans) Algorithm, is outlined in Algorithm 2. It takes as input a UCCG G and a corresponding rooted clique tree T K1 , and outputs all sets
C G (K 1 ), . . . , C G (K m ).
Step 1 identifies all separators Sep(K 1 ), all residuals Res(K 1 ), all super cliques L (1) and C G (K 1 ) for T K1 . It utilizes the super clique create operation (SC-Create-Op) in Algorithm 3, which will be introduced in details in Section 5. Steps 2-9 sequentially generate the other C G (K 2 ), . . . , C G (K m ). These steps depend on the technical details to be presented in Section 6. In each iteration of i ∈ {2, . . . , m}, the parent clique K t of K i in T K1 is found. Steps 4-7 then efficiently identify structure changes from T Kt to T Ki . In particular, Algorithm 4 in Step 7 is the super clique transfer operation (SC-Trans-Op) in Section 6. Step 8 then updates T Kt to become a rooted tree for K i .
We have the following results for Algorithm 2, the proof of which can be found in Appendix E. Proof of Theorem 4.1 and Proof of Theorem 4.2. Theorem 4.1. Let G be a UCCG, and T K1 be a rooted clique tree with cliques ordered as K 1 , . . . , K m according to the MCS algorithm. Algorithm 2 will correctly return
C G (K 1 ), C G (K 2 ), . . . , C G (K m ).
Theorem 4.2. Algorithm 2 runs in time O(m 2 ), where m is the number of cliques of UCCG G.
this section cite: []

Section: Super Cliques and Undirected Connected Components
In this section, we discuss the details of Algorithm 3. It is a novel approach to compute C G (K 1 ) based on the concept of super cliques for T K1 . The new concepts are built upon the basic rooted clique tree structures introduced in Section 2.2.
Algorithm 4 SC-Trans-Op Input: CG(Kt), L (t) , T K t , Sep(Ki), Res(Ki). Output: CG(Ki) and L (i) . 1: Initialize CG(Ki) ← {}, L (i) ← {}, SK (i) t + ← {Kt}, SR (i) t + (Ki) ← {Rt} 2: for SK (t) p + in L (t) do 3:
if p = i then 4:
for all child clique Kq of Ki in T K t do 5:
Induce
SK (i) q+and
SR (i) q + from SK (t) i + and SR (t) i + ; 6: CG(Ki) ← CG(Ki) ∪ G[SR (i) q+ ] ; 7: L (i) ← L (i) ∪ SK (i) q + ; 8:
end for 9:
else if Kp is a child clique of Kt in T K t , and St is a proper subset of Sp then 10:
SK (i) t + ← SK (i) t + ∪ SK (t) p+ , SR (i) t + ← SR (i) t + ∪ SR (t)
p+ ; 11: else 12:
SR (i) p+ ← SR (t) p+ , SK (i) p + ← SK (t) p + ; 13: CG(Ki) ← CG(Ki) ∪ G[SR (i) p+ ] ; 14: L (i) ← L (i) ∪ SK (i) p + ; 15:
end if 16: end for 17:
L (i) ← L (i) ∪ SK (i) t + ; 18: CG(Ki) ← CG(Ki) ∪ G[SR (i) t + ] .
Definition 5.1. (Clique header, clique tail) Let T K1 be a rooted clique tree with RIP clique order K 1 , K 2 , . . . , K m . i. For any p = 2, . . . , m, K p is a clique header within T K1 if for any ancestral clique K q of K p with q ̸ = 1, the corresponding S q is not a proper subset of S p .
ii. For p = 2, . . . , m, suppose K p is a clique header within T K1 , a descendant clique K q of K p is a clique tail that follows K p if S p ⊊ S q .
Note the root K 1 is neither a clique header nor a clique tail, as we require p > 1 in the above definitions. For the example in Figure 1(c), K 2 , K 3 , K 4 and K 5 are clique headers within T K1 , K 6 and K 7 are the clique tails that follows K 5 . Using the concepts of clique header and clique tail, we can define the super clique and super residual within T K1 . Definition 5.2. (Super clique, super residual) Within a clique tree T K1 , suppose K p is a clique header and K p1 , . . . , K pr are all its clique tails. i. The clique set SK (1) p + = SK (1) p|p1,...,pr
:= {K p , K p1 , . . . , K pr } is called a super clique.
ii. The set of the residuals corresponding to the cliques within SK
(1) p + is called a super residual, and denoted as
SR (1
)
p + = SR(1)
p|p1,...,pr := {R p , R p1 , . . . , R pr }. A clique header K p will form a super clique itself if it does not have any clique tail. For the clique tree in the left panel of Figure 2, there are four super cliques:
SK (1) 2| = {K 2 }, SK (1) 3| = {K 3 }, SK(1)
4| = {K 4 } and SK (1) 5|6,7 = {K 5 , K 6 , K 7 } within T K1 . Note the superscript "(1)" in these notations emphasizes that they are super cliques (or super residuals) within the clique tree T K1 rooted at K 1 .
Regarding the super cliques in T K1 , we can observe a few properties. Firstly, for each SK (1)
p + , the subgraph T K1 [SK (1)
p + ] is connected and constitutes a subtree of T K1 . This is because the clique tree T K1 generated from a UCCG G satisfies the so called induced-subtree property (Blair & Peyton, 1993). The property states that, for every vertex v ∈ V of G, the set of all cliques containing v induces a connected subtree of T . Consequently, the subgraph
T K1 [SK (1)
p + ] is connected because all the cliques within SK p + share the common node set S p .
Secondly, we observe that C G (K 1 ) can be easily obtained from the set of super residuals. Consider again the clique tree in the left panel of Figure 2 in Figure 1
(c), we can see G[SR (1) 2| ] = G[d], G[SR (1) 3| ] = G[e], G[SR (1) 4| ] = G[f ], and G[SR (1) 5|6,7 ] = G[g, h, i, j] are the undirected connected components in C G (K 1 ). In fact, this observation holds in general. For any super residual SR (1) p + , the induced subgraph G SR (1) p + is exactly an element of the set C G (K 1 ). More- over, C G (K 1
) is just the collection of all such subgraphs induced by every super residual.
Theorem 5.3. Let T K1 be a rooted clique tree of a chordal graph G with MCS clique order K 1 , K 2 , . . . , K m . Then
CG(K1) = G[SR (1) p + ] : SR (1) p + is a super residual within T K 1 .
For a given T K1 , Algorithm 3 is designed to return the set L (1) of all super cliques, all separators and residuals, and C G (K 1 ). Algorithm 3 is valid due to Theorem 5.3. It seems natural that, for i = 2, . . . , m, we can apply the same procedure to each T Ki for getting the corresponding C G (K i ). However, such procedure is unnecessary. Recall we have discussed that, for any pair of cliques K t and K i with k ̸ = i, there are many shared undirected connected components between C G (K i ) and C G (K t ). We can reuse the computation results for one rooted clique tree to speed up the computation for the other. In the next section, we will present an efficient strategy serving this purpose.
this section cite: ['b2']

Section: The Super Cliques Transfer Operation
We now present our efficient super clique transfer operation to generate all the other
C G (K 2 ), C G (K 3 ), . . . , C G (K m ), given C G (K 1 ). The overall iterative strategy is described in Algorithm 2, which generates C G (K i ) based on C G (K t ), where K t is a parent clique of K i in T K1 .
To efficiently obtain C G (K i ) from C G (K t ), we need to construct an appropriate clique tree T Ki with minimal structure changes from T Kt . We exploit the computed results from T Kt and our super clique transfer operation to reduce the computation cost. Now, without loss of generality, we discuss in details the particular situation where we transit from T K1 to T Ki , where K i is a child clique of K 1 in T K1 . The procedure for the other cases is similar.
Recall T K1 corresponds to a clique sequence K 1 , . . . , K m that satisfies the RIP. As stated in Lemma 6.1 below, there always exists a permuted sequence K σ(1) , . . . , K σ(m) , which starts with K i and also satisfies the RIP. Lemma 6.1. (Proposition 2.4 of Leimer (1993)) Let K 1 , . . . , K m be an RIP sequence of the clique set. For any i = 2, . . . , m, there exists a permutation σ satisfying that σ(1) = i and σ(2) = 1, and meanwhile K σ(1) , . . . , K σ(m) is still an RIP sequence.
In fact, the permuted sequence has a simple structure change. The proof of Leimer (1993) actually states the permuted indices as: 1) σ(1) = i and σ(2) = 1; 2) for p = 2, . . . , i -1, we have σ(p + 1) = p; and 3) for p = i + 1, . . . , m, we have σ(p) = p . The permuted RIP sequence has minimal change of the clique order. Based on the permuted sequence, the new clique tree T Ki rooted at K i can be obtained. We continue to examine the structure changes from T Kt to T Ki in more details.
this section cite: ['b14', 'b14']

Section: Basic Structure Changes in the Clique Trees
To understand the structure changes for T Ki , we first state a property regarding the permuted sequence. Proposition 6.2. Assume K i is a clique such that
K i ∩(K 1 ∪ • • • ∪ K i-1 ) ⊂ K 1 . Let K σ(1) , . . . , K σ(m)
be the permuted clique obtained by applying Lemma 6.1 with σ(1) = i. For any p ∈ [m] \ {1, i} and any q ∈
[m], if K p ∩ (K 1 ∪ • • • ∪ K p-1 ) ⊂ K q , then for p ′ and q ′ with p = σ(p ′ ) and q = σ(q ′ ), it holds that K σ(p ′ ) ∩ (K σ(1) ∪ • • • ∪ K σ(p ′ -1) ) ⊂ K σ(q ′ ) .
Note in the above p / ∈ {1, i}. Proposition 6.2 has the following implication for any K q and its child clique
K p in T K1 . Suppose the p-th clique K p in T K1 corresponds to p ′ -th clique K σ(p ′ ) in T Ki , and suppose K q corresponds to K σ(q ′ ) . We have K σ(p ′ ) is a child clique of K σ(q ′ ) in T Ki .
The above discussion implies that, from T K1 to this T Ki , only one edge changes. That is,
K 1 → K i in T K1 becomes K i → K 1 in T Ki .
The other edges in T Ki remain unchanged. Due to this edge direction change, we can see that their separators and residuals also change. The changes are summarized in Table 1.
Additionally, the separators and residuals for the other
Table 1. The separators and residuals for the cliques K1 and Ki within the two rooted clique trees T K 1 and T
K i . K 1 K i separator residual separator residual T K1 ∅ K 1 K 1 ∩ K i K i \(K 1 ∩ K i ) T Ki K 1 ∩ K i K 1 \(K 1 ∩ K i ) ∅ K i cliques remain unchanged, which is stated in the follow- ing proposition. Proposition 6.3. Assume K i is a clique such that K i ∩ (K 1 ∪ • • • ∪ K i-1 ) ⊂ K 1 . Let K σ(1) , . . . , K σ(m)
be the permutation obtained by applying Lemma 6.1 with σ(1) = i. Then for any p in [m] \ {1, i} and p ′ satisfying p = σ(p ′ ), we have S p = S σ(p ′ ) and R p = R σ(p ′ ) .
The proof of above two propositions are claimed in Appendix E. Proof of Propositions 6.2 & 6.3. The edge
K 1 → K i in T K1 will be redirected as K i → K 1 in T Ki .
This implies K 1 becomes a child clique of K i in T Ki . As the root K i is the only ancestral clique of K 1 in T Ki , K 1 must be a clique header within T Ki by Definition 5.1. The child cliques of K i in T K1 will all become clique headers in T Ki . Additionally, some cliques that were headers in T K1 will become clique tails of
K 1 in T Ki . Specifically, if K p (p ̸ = i) is a child clique of K 1 in T K1 , then K p is a clique header within T K1 , but it can possibly become a clique tail within T Ki . We need to check whether K 1 ∩ K i is a proper subset of S p . If this is true, K p will become a clique tail of K 1 in T Ki ; otherwise, K p remain a clique header in T Ki .
For Figure 2, let us consider the structure changes from T K1 in the left panel to T K5 in the right panel. We can see that K 6 and K 7 both become clique headers within T K5 . Furthermore, K 1 also becomes a clique header in T K5 . Since K 1 ∩ K 5 is not a proper subset of S 3 , K 3 remains a clique header in T K5 . On the other hand, K 1 ∩ K 5 is a proper subset of S 2 , so K 2 becomes a clique tail that follows K 1 within T K5 . The clique K 4 , which is not adjacent to K 1 , remains as a clique header within T K5 .
this section cite: []

Section: High-level Structure Changes in the Clique Trees
We can further characterize higher level structure changes from T K1 to T Ki , in terms of super cliques and super residuals. In fact, all super cliques in T Ki can be identified from those of T K1 . There are three cases to consider:
1. Consider the super clique SK
(1) i + with clique header K i in T K1 . Suppose K i has h child clique(s) in T K1 : K p1 , . . . , K p h . Then, the super clique SK (1)
i + of T K1 get split into h super clique(s) in T Ki : SK (i) p + 1 , . . . , SK (i) p + h
. These super cliques have K p1 , . . . , K p h as their clique headers, respectively. In the special case that K i has no child clique in T K1 , we can simply ignore SK (1) i + when generating super cliques for T Ki . 2. Consider all child cliques of K 1 in T K1 but with K i excluded. Among these child cliques, select those that become clique tails of K 1 in T Ki , and denote these selected cliques as K p1 , . . . , K p h . Then, their corresponding super cliques SK
p + 1 , . . . , SK (1) p + h in T K1 , together with K 1 , will form a new super clique in T Ki :
SK (i) 1 + = {K 1 } ∪ SK (1) p + 1 ∪ • • • ∪ SK (1) p + h .(3)
Note, if there does not exits any clique tail of K 1 in T Ki , then (3) simply becomes SK (i) 1 + = {K 1 }. For the child cliques of of K 1 in T K1 that are not selected for (3), their corresponding super cliques remain unchanged and continue to constitute super cliques in T Ki . 3. Aside from the Case 1 and Case 2 discussed above, all other super cliques in T K1 remain unchanged and continue to form super cliques in T Ki . All super residuals in T Ki can be identified from those in T K1 in the same spirit as the three cases above. Recall by Proposition 6.3 the only difference between Res(K i ) and Res(K 1 ) lies in the pair (R 1 , R i ), and the changes are summarized in Table 1. Corresponding to Case 1 above, the super residual SR (1) i + of T K1 get split into h super residual(s) in T Ki : SR (i) p + 1 , . . . , SR (i) p + h . As for Case 2 in the above, the super residual SR
(i) 1 + corresponding to K 1 in T Ki will be SR (i) 1 + = {K 1 \ (K 1 ∩ K i )} ∪ SR (1) p + 1 ∪ • • • ∪ SR (1) p + h . (4) Except for SR (1)
i + and the super residual in (4), all the other super residuals of T K1 remain exactly the same in T Ki .
Once the super residuals in T Ki are identified, the undirected components in C G (K i ) can be immediately determined based on Theorem 5.3. We now illustrate the structure changes from T K1 to T K5 for the example in Figure 2. Corresponding to Cases 1-3, we have the following:
1. K 5 has two child cliques in T K1 : K 6 and K 7 . Then the super clique SK
5|6,7 in T K1 get split in two super cliques in T K5 : SK (5) 6| , and SK (5) 7| . Correspondingly, we can generate the undirected connected components G[i] and G[h] in C G (K 5 ).
2. Additionally, consider the child cliques of K 1 in T K1 .
In T K5 , K 2 becomes a child tail of K 1 . Then SK
this section cite: []

Section: The Iterative Algorithm
In the above, we focus on the case where K i is one child clique of K 1 in T K1 , and we identify all super cliques and super residuals of T Ki from those of T K1 . The results can be easily generalized. For any clique K i in K G with 1 < i ≤ m, suppose its parent clique of K i in T K1 as K t , we can efficiently obtain super cliques and super residuals of T Ki from those of T Kt . This leads to the super clique transfer operation (SC-Trans-Op) in Algorithm 4, which derives the set of super cliques of T Ki from that of T Kt , and generates C G (K i ) from C G (K t ). In Algorithm 4, Lines 3-8, Lines 9-10 & 17, and Lines 11-15 correspond to Cases 1-3 in Section 6.2, respectively.
Now, let us return to Algorithm 2. In Algorithm 2, its Lines 4-6 and Line 8 are due to the basic structure changes in Section 6.1. When transitioning from
T Kt to T Ki , the set of separators (Sep(K i )) and the set of residuals (Res(K i )) can be readily determined, as we discussed for Table 1 and Proposition 6.3. Line 8 is due to Proposition 6.2 and the subsequent discussion there. Line 7 employs Algorithm 4.
In the following theorem, we prove the correctness of Algorithm 4.
Theorem 6.4. For any i = 2, . . . , m, let K t denote the parent clique of K i in T K1 , L (t) be the set of super cliques of T Kt . Then given C G (K t ), L (t) , T Kt , Sep(K i ), and Res(K i ), Algorithm 4 will return C G (K i ) and the set L (i) of super cliques of T Ki .  We use the minimal triangulation method to generate chordal graphs (Dethlefsen & Højsgaard, 2005).  3 presents both the difference and the ratio between these average running times. Figure 3(d) shows the average runtime difference of Step 2 of CP and ICP. The results highlight substantial improvements achieved by our method in terms of efficiency.
this section cite: ['b5']

Section: Conclusion
In this work, we propose an enhancement to the Clique-Picking algorithm (Wienöbst et al., 2023) by avoiding the intensive and repeated generation of C G (K j ) for each clique K j of a chordal graph G. Our improvement introduces a higher-level structure, termed a super clique, within the clique tree. We demonstrate that an efficient transfer of super cliques is possible between two clique trees with different choices of K j as the root. The proposed algorithm significantly reduces the computational cost of Step 2 in Algorithm 1.
In this appendix, Section A contains a list of the main symbols and their meaning in the paper. There are several definitions and theorems necessary for the main theoretical derivations. These preliminary results are presented in Sections B-D. In Section E, we present the detailed proofs for the theorems and propositions in the main paper. Finally,Section F includes a detailed example illustrating how the proposed algorithm proceed step by step. Rooted clique tree Clique tree with rooted K π(K)
this section cite: ['b24']

Section: References
Ref_id:b0 Title: Lazyiter: a fast algorithm for counting markov equivalent dags and designing experiments Year: (2020)
Ref_id:b1 Title: A characterization of markov equivalence classes for acyclic digraphs Year: (1997)
Ref_id:b2 Title: An introduction to chordal graphs and clique trees Year: (1993)
Ref_id:b3 Title: Learning essential graph markov models from data Year: (2004)
Ref_id:b4 Title: Learning equivalence classes of bayesiannetwork structures Year: (2002)
Ref_id:b5 Title: A common platform for graphical models in R: The gRbase package Year: (2005)
Ref_id:b6 Title: An efficient algorithm for counting markov equivalent dags Year: (2022)
Ref_id:b7 Title: Parameter priors for directed acyclic graphical models and the characterization of several probability distributions Year: (2002)
Ref_id:b8 Title: Counting and sampling from markov equivalent dags using clique trees Year: (2019)
Ref_id:b9 Title: The size distribution for markov equivalence classes of acyclic digraph models Year: (2002)
Ref_id:b10 Title: Graph Decomposition: Theory, Algorithms and Applications Year: (2010)
Ref_id:b11 Title: Active learning of causal networks with intervention experiments and optimal designs Year: (2008)
Ref_id:b12 Title: Counting and exploring sizes of markov equivalence classes of directed acyclic graphs Year: (2015)
Ref_id:b13 Title: Probabilistic Graphical Models: Principles and Techniques -Adaptive Computation and Machine Learning Year: (2009)
Ref_id:b14 Title: Optimal decomposition by clique separators Year: (1993)
Ref_id:b15 Title: Estimating high-dimensional intervention effects from observational data Year: (2009)
Ref_id:b16 Title: Bayesian model averaging and model selection for markov equivalence classes of acyclic digraphs Year: (1996)
Ref_id:b17 Title: Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference Year: (1988)
Ref_id:b18 Title: Causality: Models, Reasoning and Inference Year: (2009)
Ref_id:b19 Title: Graphical model search via essential graphs Year: (2001)
Ref_id:b20 Title: Using genetic data to strengthen causal inference in observational research Year: (2018)
Ref_id:b21 Title: Causation, Prediction, and Search Year: (2001)
Ref_id:b22 Title: Equivalence and synthesis of causal models Year: (1990)
Ref_id:b23 Title: An algorithm for deciding if a set of observed independencies has a causal explanation Year: (1992)
Ref_id:b24 Title: Polynomial-time algorithms for counting and sampling markov equivalent dags with applications Year: (2023)
