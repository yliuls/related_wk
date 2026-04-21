Title: Differential Privacy on Fully Dynamic Streams
Abstract: A fundamental problem in differential privacy is to release privatized answers to a class of linear queries with small error. This problem has been well studied in the static case. In this paper, we consider the fully dynamic setting where items may be inserted into or deleted from the dataset over time, and we need to continually release query answers at every time instance. We present efficient black-box constructions of such dynamic differentially private mechanisms from static ones with only a polylogarithmic degradation in the utility.

Section: Introduction
In the data streaming model, a stream consists of a pair of possibly infinitely long vectors (⃗ s, ⃗
x). It defines a dynamically changing dataset D t , which is a multiset of elements from a universe X . Initially, D 0 = ∅. At time t ∈ N + , (s t , x t ) arrives, where s t ∈ {+ + +, ---, ⊥ ⊥ ⊥} indicates the operation and x t ∈ X is the relevant element. Then D t is defined inductively as follows:
• If s t = + + +, then D t := D t-1 ⊎ {x t }, i.e., one copy of x t is inserted into the dataset.
• If s t = ---, then D t := D t-1 -{x t }, i.e., one copy of x t is deleted from the dataset (it is required that at least one copy of x t exists in D t-1 ).
• If s t = ⊥ ⊥ ⊥, then D t := D t-1 .
Two variants of the streaming model have been considered in the literature [25,5]: The model above is referred to as the fully dynamic setting (a.k.a. the turnstile model), while the case where s t can only be + + + or ⊥ ⊥ ⊥ is called the insertion-only setting (a.k.a. the cash register model).
A fundamental problem in differential privacy (DP) is how to answer a class of linear queries on a given dataset D privately and accurately. In particular, the private multiplicative weights mechanism (PMW) [15,17] can answer a set of arbitrary linear queries with maximum errorfoot_0 Õ( |D|). Better error bounds can be achieved if the class of queries exhibit good structures. For instance, d-dimensional half-space queries can be answered with error Õ(|D| 1 2 -1 2d ) [26]; please see Appendix A for a review of results on various query classes. Note that, since there can be many queries in the class, these algorithms do not return the query answers explicitly; instead, a privatized data structure (e.g., a histogram in the case of PMW) is often returned, and a data analyst can subsequently extract the answer of any query from the data structure.
Combining the query-answering problem and the data streaming model naturally gives rise to the problem of differential privacy under continual observation, first studied in the foundational work of Dwork et al. [9]. Here, the goal is to continually release privatized query answers under the following requirements:
• Online: For any t, we must release a data structure M (t) (D t ) before (s t+1 , x t+1 ) arrives.
• Private: All released data structures (M (1) (D 1 ), M (2) (D 2 ), . . . ) jointly satisfy DP.
• Accurate: For every t, all queries on D t can be answered from (M (1) (D 1 ), . . . , M (t) (D t ))
with small error, ideally matching the error in the static case, i.e., as if all queries were answered on D t one-shot.
• Efficient: The mechanism should run efficiently.
The pioneer work [9,4] has only studied the basic counting problem, i.e., the query just asks for |D t |, under the insertion-only streaming model. However, we observe that their algorithms are actually black-boxed, and can be instantiated with any static mechanism for some other query class, as long as the queries are union-preserving, with a polylogarithmic-factor increase in the error bound. Formally, a query f is called union-preserving if f (D (1) ⊎ D (2) ) = f (D (1) ) + f (D (2) ) for any datasets D (1) , D (2) , where ⊎ denotes the multiset union. For example, by plugging in PMW, any class of linear queries on D t can be answered with error Õ( |D t |) for every t. Perhaps not realizing this, [6] presented a white-box version of PMW for the insertion-only streaming model, but the error is Õ(|D t | 3/4 ). They also showed a black-box solution, but the error is Õ(|D t | 5/6 ) when instantiated with PMW.
A standard approach to dealing with the fully-dynamic case is to divide the stream into two insertiononly steams: one only containing insertions and one only containing deletions but treating these deletions as insertions. Let D + + + t := ⊎ i≤t:si=+ + + {x i } be all items inserted up to time t, and D --- t := ⊎ i≤t:si=---{x i } all items deleted up to time t, then D t = D + + + t -D --- t . For any union-preserving query f , we have f (D t ) = f (D + + + t ) -f (D --- t ), so we can run two separate instances of the insertion-only mechanism, and use their difference to answer f on D t . However, the worst-case error of this solution is very bad. Suppose we instantiate the insertion-only algorithm of [9,4] with PMW. Then f (D + + + t ) and f (D --- t ) can be answered with error Õ |D + + + t | and Õ |D --- t | , respectively, which means that the error for f (D t ) is Õ |D + + + t | + |D --- t | . This can be arbitrarily worse than the optimal error of Õ( |D t |) in the static case: D + t and D - t may both be very large but D t = ∅. Our Contributions.
In this paper, we present a black-box algorithm for fully-dynamic streams that can answer any class of union-preserving queries with error matching that in the static case, up to polylogarithmic factors. For instance, when instantiated with PMW, our algorithm achieves an error of Õ( |D t |) for every t. Furthermore, its total running time up to time t is also just a polylogarithmic factor higher than that of the static mechanism run on all the stream elements up to time t.
this section cite: ['b24', 'b4', 'b14', 'b16', 'b25', 'b8', 'b8', 'b3', 'b5', 'b8', 'b3']

Section: Preliminaries

this section cite: []

Section: Differential Privacy
Let X be the universe of items. A static dataset is a multiset of items D ∈ N X . Two static datasets D, D ′ ∈ N X are neighbors, denoted D ∼ D ′ , if there exists an item x ∈ X , such that D = D ′ ⊎ {x} or vise versa. Differential privacy (DP) [8] is defined as follows.
this section cite: ['b7']

Section: Definition 1 (Differential Privacy [8]).
A randomized mechanism M : N X → Y satisfies (ε, δ)-DP if for any neighboring datasets D ∼ D ′ and any subset of outputs Y ⊆ Y,
Pr[M(D) ∈ Y ] ≤ e ε • Pr[M(D ′ ) ∈ Y ] + δ .(1)
In the streaming setting, two streams are neighbors if one has one more update, which can be either an insertion or a deletion, than the other [9]. Formally, two streams (⃗ s, ⃗ x) and (⃗ s ′ , ⃗ x ′ ) are neighbors, if ∃i ∈ N + such that (s j , x j ) = (s ′ j , x ′ j ) for any j ̸ = i, and either s i or s ′ i is ⊥ ⊥ ⊥. Plugging this neighboring relationship into Definition 1 yields the DP definition for the streaming model. More precisely, M(D) and M(D ′ ) in (1) are replaced by (M (1) (D 1 ), M (2) (D 2 ), . . . ) and (M (1)
(D ′
1 ), M (2) (D ′ 2 ), . . . ) respectively, where D i and D ′ i are the datasets induced by any two neighboring streams at time i.
An important property used in designing DP mechanisms is composition, which comes in two settings: Theorem 2 (Sequential Composition [8]). Let M i : N U → Y i each be an (ε i , δ i )-DP mechanism. Then the composed mechanism M(D) = (M 1 (D), . . . ,
M k (D)) is ( k i=1 ε i , k i=1 δ i )-DP.
Note that there are many "advanced" versions of sequential composition [10,21,2] with better dependencies on k. Nevertheless, as k is logarithmic in all of our algorithms, those advanced versions do not offer better bounds for the problem studied in this paper. Theorem 3 (Parallel Composition [24]). Let U = U 1 ∪ • • • ∪ U k be a partitioning of the universefoot_2 U, and M i : N Ui → Y i each be an (ε i , δ i )-DP mechanism. Then the composed mechanism
M(D) = (M 1 (D ∩ U 1 ), . . . , M k (D ∩ U k )) is (max k i=1 ε i , max k i=1 δ i )-DP.
this section cite: ['b8', 'b7', 'b9', 'b20', 'b1', 'b23']

Section: Linear Queries
A linear query is specified by a function f : X → [0, 1]. The result of evaluating f on D is defined as f (D) := x∈D f (x). A fundamental problem in differential privacy is the following: Given a set of linear queries F = {f 1 , . . . , f |F | }, design a DP mechanism M that, on any given D, outputs a data structure M(D), from which an approximate fM (D) can be extracted for any f ∈ F. We say that M has error α with probability
1 -β, if Pr max f ∈F fM (D) -f (D) > α ≤ β
for any D, where the probability is taken over the internal randomness of M. Clearly, α is a function of the privacy parameters (ε, δ) and the failure probability β. For most mechanisms, it also depends on the data size |D|, number of queries |F|, and domain size |X |. To simplify notation, we often omit some of these parameters from the full list α(ε, δ, β, |D|, |F|, |X |) if they are clear from the context. Similarly, we denote the running time of the mechanism by time(|D|, •), which depends on the data size |D| and possibly other parameters.
There is extensive work [30,15,17,29,14,1,22,23,26] on the best achievable α for various families of linear queries. This paper takes a black-box approach, i.e., we present dynamic algorithms that can work with any M that has been designed for queries F on a static dataset D. The error for the dynamic algorithm will be stated in terms of the α function of the mechanism M that is plugged into the black box. Nevertheless, we often derive the explicit bounds for the following two most interesting and extreme cases:
Basic counting. If F consists of a single query f (•) ≡ 1, which simply returns f (D) = |D|, then the "data structure" M(D) consists of just one number, which is a noise-masked f (D). The most popular choice of the noise is a random variable drawn from the Laplace distribution Lap( 1 ε ). Its error function α is given by α Lap (ε, β) = O 1 ε log 1 β . Alternatively, one can add a Gaussian noise, which is (ε, δ)-DP for δ > 0 and yields α Gauss (ε, δ, β) = O 1 ε log 1  δ log 1 β . The two error bounds are generally incomparable, but the former is usually better since δ ≤ β in common parameter regimes. Both the Laplace and the Gaussian mechanism have time(|D|) = O(|D|).
Arbitrary queries. If F is a set of arbitrary linear queries, then the private multiplicative weights (PMW) [15,17] mechanism achieves α PMW = Õ( |D|) for δ > 0 and Õ(|D| 2/3 ) for δ = 0. These error bounds are optimal for |F| sufficiently large. The running time of PMW is
Õ(|D| + |X | • |F | • |D| 2 /α 2 ).
There are many possibilities between the two extreme cases, and the achievable error bound α intricately depends on the discrepancy of the query set F [16,26]. We include a brief review in Appendix A, which is not necessary for the understanding of this paper. We make a reasonable assumption that α does not depend on any of its parameters exponentially, which allows us to ignore the constant coefficients in the parameters, e.g., α(O(ε)) is asymptotically the same as α(ε). This assumption holds for most existing mechanisms for linear queries.
this section cite: ['b29', 'b14', 'b16', 'b28', 'b13', 'b0', 'b21', 'b22', 'b25', 'b14', 'b16', 'b15', 'b25']

Section: Differential Privacy for Insertion-only Streams
The insertion-only algorithms, as well as our fully dynamic algorithm, work by decomposing D = D (1) ⊎ • • • ⊎ D (k) for some k, so f (D) = f (D (1) ) + • • • + f (D (k) ) by the union-preserving property. Thus, if the error of each f (D (i) ) is α(•, β), the error of f (D) is at most k • α(•, β/k) by a union bound. Further, if the estimates of f (D (i) ) are unbiased with good concentration properties, this bound can be tightened. For example, for the basic counting query f (D) = |D|, if we use the Laplace mechanism to estimate each f (D (i) ), then the error for f (D) is O 1
ε k log 1 β + log 1 β , which is better than k •α Lap (•, β/k) = O k ε log k β .
We derive this result, as well as the error bounds of some other mechanisms under such a disjoint union, in Appendix B. In the main text, for generality we will use α (k) to denote this error bound. Note that α (k) is also a function of |D|, ε, δ, β, |F|, |X |, but we may omit these parameters when the context is clear. Using this notation, the result of existing insertion-only algorithms can be restated as follows.
this section cite: []

Section: Lemma 4 ([4]).
For a set of union-preserving queries F, let M : N X → Y be an (ε, δ)-DP static mechanism whose error is α(ε, δ, β, |D|, |F|, |X |) on any D, with running time time(|D|). Then there is an (ε, δ)-DP mechanism M F ,ins for insertion-only streams, so that at any time t, it answers queries F on D t with error
γ(t; ε, δ, β, |D t |, |F|, |X |) = α (log t) ε log t , δ log t , β, |D t |, |F|, |X | .
The total running timefoot_3 of M F ,ins up to time t is O(log t) • time(|D t |).
Our algorithm will also use a special case of Lemma 4 when F is the basic counting query. Plugging in our bound on α
Lap in Appendix B leads to the following explicit error bound, which is slightly better than that in [4].
this section cite: ['b3']

Section: Corollary 1.
There is an ε-DP mechanism M cnt,ins that at any time t, answers the counting query f (D t ) = |D t | on an insertion-only stream with error
γ cnt (t; ε, β) = α (log t) Lap ε log t , β = O log 1.5 t ε log 1 β + log t ε log 1 β .
this section cite: []

Section: Differential Privacy on Fully Dynamic Streams
In this section, we describe our algorithm for answering a set of union-preserving queries on a possibly infinite fully dynamic stream under differential privacy. We use n t := |D t | to denote the size of the dataset at time t, and N t to denote the total number of updates up to time t. Note that for an insertion-only stream we have n t = N t , but on a fully dynamic stream n t may be much smaller than N t . Our goal is to achieve an error bound that depends on n t .
We will treat a fully dynamic stream as a set of labeled time intervals. An interval [i, j) labeled with item x ∈ X represents that a copy of x is inserted at time i and deleted at time j. It is possible that j = ∞, if the item is never deleted. Note that this interval representation of a stream is not unique, e.g., when many copies of the same item are inserted and then deleted. Any representation can be used; in fact, our algorithm does not use this interval representation, only the analysis does.
Using the interval representation, D t consists of all items whose intervals are stabbed by t, i.e., all intervals [i, j) ∋ t. We will make use of the interval tree [7] to organize all the intervals. In an interval tree, each D t is decomposed into a logarithmic number of subsets, each of which consists of one-sided intervals, which will allow us to use the insertion-only mechanism. However, there are two technical difficulties in implementing such a plan. First, the intervals are given in an online fashion, i.e., at time t, we only see the endpoints of the intervals prior to t. When we see the left endpoint of an interval, we do not know where in the interval tree to put this interval, yet, we need to immediately release privatized information about this interval. Second, the interval tree on an infinite stream is also infinitely large, so we have to build it incrementally, while allocating the privacy budget appropriately.
We describe how to overcome the difficulties above in Section 3.1. In Section 3.2, we introduce a DP mechanism running at each node of the new tree structure to support querying at any time with respect to intervals stored in the tree. The output of the whole mechanism is obtained by combining the individual mechanisms at the tree nodes, which is summarized in Section 3.3.
this section cite: ['b6']

Section: Online Interval Tree
We first build a binary tree T over the timestamps, where each timestamp corresponds to a tree node. Figure 1 shows an (offline) interval tree built on the first 8 timestamps. It is clear that the tree on timestamps {1, . . . , t} has t nodes and O(log t) height. In the online setting, T will grow from left to right. We order the nodes using an in-order traversal of T : v 1 , v 2 , . . . , and we will release the information about v i right after time i.
Ignoring differential privacy for now, let us first focus on how to answer a stabbing query using an interval tree, i.e., report all items (intervals) in the dataset D q at query time q. In a standard interval tree, an interval is stored at the highest node v t such that its timestamp t stabs the interval. Denote by D(v) the set of labeled intervals stored at v. For example, D(v 4 ) = {a, b, c} in Figure 1. Given a query at time q, we follow the root-to-node path to v q in T . For each left node v l on the path where l ≤ q, we find all intervals [i, j) in D(v l ) such that j > q; for each right node v r on the path where r > q, we find all intervals [i, j) in D(v r ) such that i ≤ q. Standard analysis on the interval tree shows that these subsets form a disjoint union of all intervals stabbed by q.
this section cite: []

Section: Example 1 (Interval Tree Query).
Given the interval tree in Figure 1, assume a query is issued at time q = 6. We follow the path (v 6 , v 4 , v 8 , . . . ). Along the path, v 4 is on the left, where a, c ∈ D(v 4 ) are deleted after q = 6; v 8 is on the right, where d ∈ D(v 8 ) begins at (or before) q = 6. Thus we report D 6 = {a, c, d}, which are exactly the elements present in the dataset at the query time. 1 2 3 4 5 6 7 8 9 v 1 {a} v 2 {a, b} v 3 {c} v 4 {a, b, c} v 5 v 6 {d} v 7 v 8 {d} a c b d Figure 2: An online interval tree.
In an online setting, however, we do not know which node is the highest to put an interval in, since we do not know the deletion time when an item is inserted. Consider timestamp 3 in Figure 1, where item c is inserted into D 2 = {a, b}. If c were to be deleted at timestamp 4, then v 3 becomes the highest node that stabs it, which by definition should store c. However, if c is deleted at some timestamp among {5, 6, 7}, then v 4 is the highest node. Other possible candidates are v 8 , v 16 , . . . . One idea is to put a copy of the interval into every node where the interval might be placed into. But there are infinitely many such nodes, therefore we do so lazily. More formally, we design a novel online interval tree that is capable of handling an infinite stream.
this section cite: []

Section: Definition 5 (Online Interval Tree).
In an online interval tree T , a tree node v t stores an interval [i, j) if and only if both A) i ≤ t < j; and B) v i is in the subtree rooted at v t .
Compared with the standard offline interval tree, an online interval tree may store an interval [i, j) at multiple nodes. Nevertheless, condition B) implies that all these nodes lie on the root-to-node path to v i , so at each level, there is at most one node that stores [i, j), which is the key property we will need. We use Figure 2 to illustrate. Interval a is stored at v 1 , v 2 , v 4 , which are ancestors of v 1 . It is not stored in v 8 since condition A) is violated: intuitively, by timestamp 8, a is already deleted, so there is no need to store a at v 8 . On the other hand, a needs to be stored in all v 1 , v 2 , v 4 (in the standard interval tree, it is only stored at v 4 ), because by timestamp 1 or 2, we still do not know the deletion time of a. Similarly, interval d is stored in both v 6 and v 8 .
this section cite: []

Section: Building the Online Interval Tree
This online interval tree can be incrementally constructed easily. After observing the update at timestamp t, we first compute the dataset D t at time t. These are exactly the elements that satisfy condition A). Then we construct the dataset D(v t ) for the node v t out of elements in D t , keeping only those that also satisfy condition B). These will be the intervals whose insertion-time node v i = v t or lies in the left-subtree of v t . For any node on the left-most path of T (where t is a power of 2), D(v t ) = D t simply contains all elements in the current dataset. Otherwise, D(v t ) ⊆ D t will only contain items inserted after its closest left-ancestor.
this section cite: []

Section: Example 2 (Online Interval Tree Construction).
In Figure 2, we construct D(v 4 ) = D 4 = {a, b, c} at t = 4. To construct D(v 6 ), we first compute D 6 = {a, c, d}. But we only consider items whose insertion-time is in the left-subtree (namely {v 5 , v 6 }, therefore D(v 6 ) = {d} will only include d from D 6 . The intuition is that a and c have already been covered by v 4 .
Note that when D(v t ) is first constructed, we do not have the deletion times of the items in D(v t ), which will be added when these items are actually deleted later. For example, in Figure 2, D(v 2 ) = {a, b} is constructed at timestamp 2 but neither item is associated with a deletion time. After timestamp 5, we add the deletion time of b, augmenting D(v 2 ) to {a, (b, 5)}; after timestamp 8, it becomes {(a, 8), (b, 5)}. Note that there is no need to associate the left endpoints (i.e., insertion times) to the items as in the standard interval tree, and we will see why below.
this section cite: []

Section: Querying the Online Interval Tree
Now we show how to answer a stabbing query using the online interval tree. Since the online interval tree includes multiple copies of an item, the standard interval tree query algorithm will not work, as it may report duplicates. For the stabbing problem itself, duplicates are not an issue as they can be easily removed if they have been reported already. However, for answering linear queries, we actually need to cover all stabbed intervals by a disjoint union of subsets. To achieve it, we modify the stabbing query process as follows. Given a query at time q. We follow the root-to-node path to v q in T , and only consider left nodes v l on the path where l ≤ q. For each v l , we report all intervals [i, j) in D(v l ) where j > q. Example 3 (Online Interval Tree Query). Again consider a query at q = 6 in Figure 2, on the root-to-node path, nodes v 4 and v 6 satisfy l ≤ q. We visit them and report {a, c} and {d} respectively, which jointly form the dataset D 6 .
Unlike in the standard interval tree, we do not query those right ancestors (e.g. v 8 ). It turns out that the items stored in the right ancestors are exactly compensated by the extra copies of items stored in the left ancestors of v q . The following lemma formalizes this guarantee. Lemma 6. The query procedure described above reports each stabbed interval exactly once. Proof. Given a query at time q, consider any item inserted at time i and deleted at time j, represented by interval [i, j). We first prove that this item will not be reported if it is not in the dataset D q , i.e. interval [i, j) is not stabbed by q. This happens when: (1) the item has been deleted at query time (j ≤ q). As we only report an item whose deletion time j > q, the interval is filtered out; (2) the item has not arrived by query time (i > q). As we only visit left nodes v l where l ≤ q, it follows that l ≤ q < i. By definition, v l can store an item only if l ≥ i, so this item is not stored by v l .
i l q v i v l v q Nodes v ′ i storing [i, j) Nodes v ′ q queried at timestamp q
The final case is when i ≤ q < j, and the interval is supposed to be reported by exactly one node. This is shown in Figure 3. For the trivial case that q = i, the newly constructed node v i is the only node reporting this interval. Otherwise, consider the minimum subtree containing both v i and v q , and assume it is rooted at v l . We must have v i in its left subtree and v q in its right subtree by the minimum property, with the only exception that one of them can be v l itself, i.e., i ≤ l ≤ q. We can argue that v l is the only node that reports the interval: any node v ′ i ̸ = v l that stores [i, j) is either in the left subtree of v l , or an ancestor of v l that is on the right side of v l ; any node v ′ q ̸ = v l queried at q is either in the right subtree of v l , or an ancestor of v l that is on the left side of v l . Thus the only node that can possibly report this interval is v l . Since v l is queried and j > q, this stabbing interval is reported exactly once by v l .
this section cite: []

Section: Deletion-only Mechanism at Each Node
We have shown that the online interval tree can be incrementally constructed, such that at any time t, we can obtain the current dataset D t by a disjoint union of O(log t) subsets, each from v t or a left ancestor of v t in the interval tree. Consider each queried node v i (i ≤ t) where D t (v i ) denotes the set of items that node v i stores at time t. This implies that a linear query f (D t ) can be answered by computing the sum i f (D t (v i )) over queried nodes v i , as specified in Section 3.1.2. Answering queries F on items stored by v i at time t is a deletion-only problem: when D(v i ) is first constructed at timestamp i, it has size n(v i ) ≤ n i and no element is associated with deletion time. Then, items in D(v i ) get deleted as time goes by. Now we focus on the deletion-only problem at node v i . To distinguish, we use D(v i ) to denote the initial state of node v i when it first gets constructed at timestamp i, and use D t (v i ) to denote the remaining items in node v i at time t. Conceptually, we also consider a dynamic dataset D --- t (v i ), which consists of all the deleted elements from node v i . Their sizes will be denoted as n(v i ), n t (v i ) and n --- t (v i ) respectively. For example, D(v 2 ) = {a, b} when constructed at time 2, whereas D 5 (v 2 ) = {a} and D --- 5 (v 2 ) = {b}. Clearly at any time t, D t (v i ) ⊎ D --- t (v i ) = D(v i ) and therefore n t (v i ) + n --- t (v i ) = n(v i ). We then have n t (v i ) ≤ n t and n t (v i ) ≤ n(v i ) ≤ n i , but there is no relationship between n i and n t .
A simple solution for this deletion-only problem is to first release a privatized f (D(v i )) when initialized, and then run an insertion-only mechanism over the conceptual dataset D --- t (v i ) so that f (D --- t (v i )) can be obtained at any time. Using their difference, f (D t (v i )) can be answered. The problem here is that given a static mechanism M with error function α(|D|, •), the initial f (D(v i )) is answered with error α(n(v i ), •). Although n(v i ) is bounded by n i , it has no relationship with the current data size n t at query time. In particular, the initial data size n(v i ) can be arbitrarily larger than its current size n t (v i ), which fails to achieve our target error bound α(n t , •).
To fix the problem, we ensure that no more than n(v i )/2 items should be deleted so that at any time we can guarantee n(v i ) = O(n t ). When half the items have been deleted from D(v i ), we restart the process on a new D(v i ) that consists of the remaining items with fresh privacy budgets.
There are still a few privacy-related issues with the above idea. First, we cannot restart when exactly half the items have been deleted, which would violate DP. Instead, we run a basic counting mechanism M cnt,ins over the conceptual dataset for deletions to approximately keep track of the number of deletions; we show that such an approximation will only contribute an additive polylogarithmic error. Second, since we restart the process above multiple times, we need to allocate the privacy budget carefully using sequential composition. But the privacy degradation is only polylogarithmic since we have only O(log n(v i )) = O(log t) restarts with high probability. Finally, as the decision of restarting the process depends on random noises, the total number of restarts can not be fixed in advance. Since we must guarantee differential privacy for any possible instantiation, we need to allocate the privacy budget through a convergent sequence. Similar to [4], we allocate the privacy budgets proportional to r -(1+η) in round r for a small constant η > 0. The total privacy is then bounded regardless of the number of actual rounds, as
∞ r=1 r -(1+η) < 1 η + 1.
This incurs another logarithmic-factor degradation. Algorithm 1 details the steps we run at each node v in the online interval tree. We present in Lemma 7 its accuracy guarantee, assuming each node is allocated with (ε, δ)-DP.
Algorithm 1: (ε, δ)-DP Algorithm at node v = v i Input: Fully-dynamic stream (. . . , (s t , x t ), . . . ), constant η, privacy budget (ε, δ), probability β, static (M F ) and insertion-only (M F ,ins ) mechanisms for queries F, continual counting mechanism M cnt,ins
Output: F(D t (v i )) at any time t ≥ i   Proof. We use some extra notation in the proof for simplicity. As we focus on the deletion-only problem at node v i , we denote
1 θ ← η 1+η , r ← 1, (ε 1 , δ 1 ) ← θ 4 ε, θ 2 δ , β 1 ← θ 6 β; // Initialize 2 D(v i ) ← All
γ ← Error bound γ cnt (t; ε r , β r ) in Corollary 1; 15 if n---> n/2 + 2γ then // Restart 16 r ← r + 1, (ε r , δ r ) ← θ 4r 1+η ε, θ 2r 1+η δ , β r ← θ 6r 1+η β; 17 D ← D t (v i );
D = D t0 (v i ), D ---= D-D t (v i ), n = n t0 (v i ) and n ---= n-n t (v i ),
where t is the query time and t 0 is the beginning time of the current round. γ is the public error bound of basic counting on infinite streams given in Corollary 1, which only depends on parameters in the round r and the time t.
Privacy. Algorithm 1 uses four black-box mechanisms. In each round, n is released using the Laplace mechanism and F(D) is released using static mechanism M F . In addition, two insertion-only mechanisms M F ,ins and M cnt,ins are used to track F and the basic counting query over the deletions D ---. In any round r, the composition of these four mechanisms is (4ε r , 2δ r ) = θ r 1+η ε, θ r 1+η δ -DP. As we restart these mechanisms, they are sequentially composed, which guarantees the whole mechanism at node v i is (ε, δ)-DP, since
∞ r=1 θ r 1+η < η 1+η ( 1 η + 1) = 1.
Note that the privacy guarantee holds regardless of the number of restarts.
Accuracy. Let random variable R denote the number of restarts before the mechanism terminates, we first bound R as follows. When a restart happens, we have n---> n/2 + 2γ, where γ = Õ(1). With probability 1 -2β r , both n--and n/2 have error at most γ. Conditioned on this happening, we get n ---> n/2: at least half of the remaining items have been deleted from D. As v i was initialized with n(v i ) ≤ n i ≤ N i ≤ N t items, with high probability this can happen at most O(log N t ) times before there are only n t (v i ) ≤ γ items left. Still conditioned on the noise being small, line 19 becomes true and the algorithm halts by answering 0, which has error at most O(γ) for any linear query. To conclude, conditioned on the events that in each round, the noises in counting are smaller than the bound γ, which happens with probability 1 -∞ r=1 2β r > 1 -β/3, there can only be O(log N t ) rounds at time t, where N t is the number of updates up till time t.
We next bound the error of F(D t (v i )) in any round r ≤ R, as a function of r and with probability 1 -2β/3. If the algorithm decides to restart at time t, an up-to-date dataset D t (v i ) is computed and a fresh F(D t (v i )) is obtained from the static mechanism M F with privacy budget (ε r , δ r ), whose error is α ε r , δ r , 2β  3 , n t (v i ), • = α(ε r , δ r , β, n t , •) with probability 1 -2β/3. Otherwise (line 23), with probability 1 -2β r we have the actual number of deletions n ---≤ n/2 + 4γ. Namely the current data size is at least n t (v i ) = nn ---≥ n/2 -4γ. Conditioned on this, we calculate F(D t (v i )) from the difference of F(D) and F(D ---). Note that with probability 1 -β/6, the error for F(D)
from M F is α ε r , δ r , β
6 , n, • by definition; also with probability 1 -β/6, the error for F(D ---) from M F ,ins is α (log t) εr log t , δr log t , β 6 , n ---, • by Lemma 4. But now both terms can be bounded by α (log t) εr log t , δr log t , β, n t + γ, • with high probability, as n ---≤ n ≤ 2n t (v i ) + 8γ = O(n t + γ). This means their difference, F(D t (v i )) has error α (log t) εr log t , δr log t , β, n t + γ, • with probability at least 1 -2β r -2β/6 ≥ 1 -2β/3.
Finally we condition on the event that there are only r = O(log N t ) rounds to get rid of the dependency on r, which happens with all but β/3 probability. By Corollary 1, the counting error in round r at time t is γ(t; ε r , β r ) = O log
1.5 t εr log 1 βr + log t εr • log 1 βr = O log 1.5 t εr • log 1 βr . Therefore the error for F(D t (v i )) is (with probability 1 -β) α (log t) ε (log 1+η N t ) log t , δ (log 1+η N t ) log t , β, n t + (log 1+η N t ) log 1.5 t ε log log N t β , |F|, |X | .
this section cite: ['b3']

Section: Full Algorithm
Lemma 7 assumes each node is under (ε, δ)-DP, which we cannot afford since we have an online interval tree of depth log t. Since the tree size grows with t and can be infinite, we allocate (ε(v), δ(v)) proportional to 1/ℓ 1+η to a node v at level ℓ in the online interval tree, so that the composition of mechanisms at all the nodes still satisfies (ε, δ)-DP at any time t. Since each item will only affect one single node in each level, nodes at the same level enjoy parallel composition. The error of the final sum is then decomposed into log t online interval tree nodes, where the error of each node is given by Lemma 7 but under (ε(v), δ(v))-DP.
To provide an error guarantee to the final results, note that it is the disjoint union of (log 2 t) static mechanisms. This is because each of the (log t) queried online interval tree nodes runs an insertiononly mechanism to support querying the deleted elements, the error of each one is from the error of (log t) static mechanisms. We are left with analyzing the error of each building block: the static mechanisms. Their error depends on the privacy budgets (ε ′ , δ ′ ) allocated to it. In our construction, it has a polylogarithmic degradation compared to the (ε, δ) for the whole mechanism. In particular, there are 3 factors that accounts for the allocation of privacy budgets.
1. We have an online interval tree of height (log t). While nodes on the same level enjoy parallel composition, it is possible that the change of one timestamp affects multiple tree nodes on the root-to-node path corresponding to this timestamp. Therefore, we must allocate privacy budget with sequential composition. Moreover, the tree can grow infinitely. So instead of an even allocation, we apply a convergent sequence to divide the budgets, which causes a log 1+η t overhead to the worst building block at time t. 2. At each online interval tree node, we restart the 4 mechanisms several times. Since this may repeatedly reveal information of the same entry, we also need to divide the privacy budget accordingly. Again, we allocate the budgets in each round using a convergent sequence again to make the DP guarantee independent of the number of restarts. With high probability, no queried node will restart more than log N t times, so the minimum privacy budget in any round is a 1/(log 1+η N t ) fraction of the budget allocated to the node. 3. Finally, in each round of each node, the insertion-only mechanisms M cnt,ins and M F ,ins are built from static mechanisms, each receiving a 1/(log t) fraction of the budget.
Putting things together, we arrive at the main result of this paper. For the running time, observe that the mechanism run at each node of T is dominated by the insertion-only mechanism from Lemma 4 that tracks the deletions. Since there can be O(log t) restarts, and the total running time over all nodes in the same level is O(log t) • time(N t ). Summing over all O(log t) levels, we obtain the running time stated. Theorem 8. If there is a static DP mechanism with error function α (k) (ε, δ, β, |D|, |F|, |X |) for a set of union-preserving queries F, then there is an (ε, δ)-DP mechanism M dyn for fully-dynamic streams, so that at any time t, it answers queries F on D t with error
ζ(t; ε, δ, β, n t , |F|, |X |) = α (log 2 t) ε log 3+2η t , δ log 3+2η t , β, n t + log 3.5+2η t ε log log t β , |F|, |X | , for any constant η > 0. The running time of M dyn up to time t is O(log 2 t) • time(N t ).
Plugging in PMW as the static DP mechanism, we obtain a fully dynamic algorithm for answering a set of arbitrary linear queries with the following error bounds: Corollary 2. Given a set of linear queries F, there is an (ε, δ)-DP mechanism M F ,dyn that at any time t, answers any query f ∈ F on a fully-dynamic stream with error
ζ F (t; ε, δ, β, n t , |F|, |X |) = α ( Õ(1)) PMW ε Õ(1) , δ Õ(1) , β, n t + Õ(1), |F|, |X | = Õ( √ n t ), δ > 0 , Õ(n 2/3 t ), δ = 0 .
Our mechanism only has a polynomial dependency on the data size n t at time t, matching results in the static setting, whereas a straightforward extension of insertion-only mechanisms [9,4] will have a polynomial dependency on the number of updates N t ≫ n t .
this section cite: ['b8', 'b3']

Section: Limitations
While polylogarithmic factors are often neglected in theoretical studies, they still present significant overhead in practice, which limits the practicality of the algorithms proposed in this paper. How to reduce this overhead remains an interesting problem for further investigation. A possible future direction is to consider white-box mechanisms that improve the accuracy (though they can only reduce the polylogarithmic factors) while being practical. It is also interesting to study how to answer non-union-preserving queries (e.g. distinct count) accurately under the streaming DP setting.
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: This is a theoretical paper that does not include experiments.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
this section cite: []

Section: A DP Mechanisms for Linear Queries: A Review
In this section, we present some important DP mechanisms for linear queries and their error bounds. The analysis is similar to that in in [30], but we clarify the dependency on β, which will be needed for analyzing the simultaneous error in the streaming setting. Note that a linear query has sensitivity 1 by definition.
Laplace Mechanism. When F = {f } is a single query, the Laplace mechanism that outputs M Lap (D) = f (D) + Lap( 1 ε ) has error α Lap (ε, β) = 1 ε ln 1 β . When F contains multiple queries, we may add Lap( |F | ε ) noise to each query result and apply basic composition to guarantee ε-DP of the whole mechanism. To translate it into an error bound, we bound the failure probability of each noise by β |F | , so that a union bound will bring the total failure probability to β. A similar argument can be made using advanced composition. To conclude, answering a set of queries F using Laplace mechanism achieves error (for δ ≥ 0)
α Lap (ε, δ, β, |F|) =              O |F| ε log |F| β , δ ≤ e -Ω(|F |) ; O   |F| log 1 δ ε log |F| β   , δ ≥ e -O(|F |) .
this section cite: ['b29']

Section: Gaussian Mechanism.
Similar to the Laplace mechanism, the Gaussian mechanism protects (ε, δ)-DP of query f by outputting
M Gauss (D) = f (D) + N 0, 2 ε 2 ln 1.25 δ . Its error is α Gauss (ε, δ, β) = 2 ε ln 1.25 δ ln 2 β .
When composing multiple Gaussian mechanisms that each answers a query from F, zCDP composition [2] can be applied, which shows adding N 0, O |F | ε 2 log 1 δ noise to each query suffices to protect (ε, δ)-DP of the whole mechanism. Therefore answering a set of queries F using Gaussian mechanism achieves error (for δ > 0)
α Gauss (ε, δ, β, |F|) = O   |F| log 1 δ ε log |F| β   .
this section cite: ['b1']

Section: Private Multiplicative Weights.
When there are many queries |F| = Ω(|D|), composing individual mechanisms has error polynomial in |F|, thus also in |D|. The Private Multiplicative Weights mechanism [15,17] performs better in this case. The following error bound is presented in [17,8].
α PMW =                O |D| 2 log |X | log(|F|/β) ε 1 3 , δ = 0 ; O   |D| log |X | log(1/δ) log(|F|/β) ε 1 2   , δ > 0 .
Apart from mechanisms mentioned above, there are other private mechanisms for linear queries. For example, the optimal composition [21] can be used in place of basic or advanced composition to provide a better allocation of privacy budget, yet computing it is costly. The log |F| factor is removable for Laplace mechanism [29] and almost removable for Gaussian mechanism [14]. Under pure-DP, SmallDB [1] has asymptotically the same error as PMW, but its running time is prohibitive. The Matrix mechanism [22,23] exploits structural properties within the query set F and works well in practice. But it does not have a closed-form error bound for general queries, and finding an optimal querying strategy is time-consuming.
In general, the best mechanism is related to the hereditary discrepancy [16,26] of the set of queries. For example, for d-dimensional halfspace counting queries, [26] has error Õ(|D|
1 2 -1 2d
). In this paper we use α as a function of ε, δ, β, and possibly |D|, |F|, |X | to denote the error of any mechanism answering linear queries on static datasets, without detailing the best mechanism under a specific setting and choice of the parameters. Since our paper takes a black-box approach, all these algorithms can be plugged into our framework so as to support dynamic data, while incurring a polylogarithmicfactor degradation.
this section cite: ['b14', 'b16', 'b16', 'b7', 'b20', 'b28', 'b13', 'b0', 'b21', 'b22', 'b15', 'b25', 'b25']

Section: B Error Bounds under Disjoint Union
Note that α (k) denotes the error of the sum of k mechanisms, each of which having error α under the same parameter settings. It always holds that α (k) (•, β) ≤ k • α(•, β k ) by union bound. In this section, we show cases where α (k) can be tightened for specific mechanisms. As our running example, consider α Lap (ε, β) = 1 ε ln 1 β . The union bound reduction above gives
α (k) Lap (ε, β) ≤ k ε ln k β
to bound the error of summing k Laplace noises Lap( 1 ε ). Next we show how this can be tightened.
this section cite: []

Section: B.1 Unbiasedness
If a mechanism M is unbiased with error α, naturally the error only scales with √ k. To see why this is true, we can argue that with all but β 2 probability, individual mechanisms have their errors bounded by α(•, β 2k ) simultaneously. Conditioned on this happening, apply Hoeffding's inequality with the remaining β 2 probability, we get
α (k) Unbiased (•, β) ≤ 2k ln 4 β • α •, β 2k .
This can be applied to the Laplace mechanism to get
α (k) Lap (ε, β) = O √ k ε log 1 β log k β .
this section cite: []

Section: B.2 Concentration Bounds
For specific distributions like Laplace (sub-exponential) and Gaussian (sub-gaussian), the concentration bounds are usually tighter than using unbiasedness only. It helps save the log k factor from union bound. Consider the Laplace mechanism, note that the Lap( 1 ε ) random variable is sub-exponential with norm ∥Lap( 1 ε )∥ Ψ1 = 2 ε . We can then apply Bernstein's inequality [31]. Lemma 9 (Bernstein's inequality). Let X 1 , . . . , X k be i.i.d. zero-mean sub-exponential random variables with norm Ψ 1 . There is an absolute constant c so that for any t ≥ 0,
Pr k i=1 X i > t ≤ 2 exp -c min t 2 kΨ 2 1 , t Ψ 1 .
We therefore conclude the Laplace mechanism has error function
α (k) Lap (ε, β) = O 1 ε k + log 1 β • log 1 β .(2)
As another example, for the Gaussian mechanism, the sum of k Gaussian noises is still a Gaussian noise with its variance scaled up by k, thus the disjoint union of k Gaussian mechanisms has error
α (k) Gauss (ε, δ, β) = O 1 ε k log 1 δ log 1 β .
For the PMW mechanism, the union bound remains the best we know, that is
α (k) PMW =                O k • |D| 2 log |X | log(k|F|/β) ε 1 3 , δ = 0 ; O   k • |D| log |X | log(1/δ) log(k|F|/β) ε 1 2   , δ > 0 . =      Õ k • |D|2 3
, δ = 0 ;
Õ k • |D|1 2
, δ > 0 .
this section cite: ['b30']

Section: C Related Work for DP under Continual Observation
In the Binary Tree mechanism [9], each timestamp is treated as a leaf node in the binary tree, and the mechanism privately releases the count of each tree node. If the stream has bounded length T , then each node receives a 1 log T fraction of the privacy budget, and the final count consists of counts from O(log t) tree nodes. Using our notation, the mechanism has error α (log t) Lap ( ε log T , β) for basic counting. Chan et al. [4] first showed that allocating the privacy budget through a convergent sequence will allow the Binary Tree mechanism to also work for infinite streams, where the error is worse by a logarithmic factor as α (log t) Lap ( ε log 1+η t , β) for constant η > 0. They further proposed a hybrid mechanism to show that α (log t) Lap ( ε log t , β) can be achieved by separately releasing the counts for [1, 2), [2, 4), [4,8), . . . . Using the α (k) Lap function in Equation (2) above, we obtain the error
α (log t) Lap ε log t , β = O log 1.5 t ε log 1 β + log t ε • log 1 β .
for continual counting, which was presented in Corollary 1 as our building block. Note that this is a per-query bound: with constant probability, a query at any time t can be answered with error O( 1 ε log 1.5 t). Alternatively, if we replace β with β/T where T is an upper bound on t, we obtain a bound that holds simultaneously for all queries 1 ≤ t ≤ T : with constant probability, any query can be answered with error O( 1 ε log 2 T ). The simultaneous bound given in [9] was O( 1 ε log 2.5 T ) using unbiasedness but not concentration, which was corrected in [11]. In While the per-query bound is the same, for the simultaneous bound, this only gives O( 1 ε log 2.5 T ), whereas our tighter bound gives O( 1 ε log 2 T ). The same simultaneous bound was presented in [13, 18] with improved constants using the Matrix Mechanism [22,23] as building block. For sparse finite streams, [11] achieves O( log 1.5 Nt ε log 1  β + 1 ε log T β ) per-query error for basic counting, which has an asymptotic improvement when N t = t o (1) .
We briefly mention some other work under similar settings but are less related. [3] studies the DP histogram problem in the streaming setting, which is a special case of linear queries. Their main contribution is when the universe X or the sensitivity of the queries are unbounded. Otherwise, the algorithm is equivalent to Corollary 1. [19] studies the DP distinct counting problem under the turnstile (fully-dynamic) model. Distinct counting queries are known to be non-linear and not union-preserving, which is separate from the interest of this paper. There are also existing work that study graph queries under the continual observation model of differential privacy, e.g. [28,12,20,27].
this section cite: ['b8', 'b3', 'b3', 'b7', 'b8', 'b10', 'b21', 'b22', 'b10', 'b2', 'b18', 'b27', 'b11', 'b19', 'b26']

Section: NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
• You should answer [Yes] , [No] , or [NA] .
• [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• Please provide a short (1-2 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
this section cite: []

Section: IMPORTANT, please:
• Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist", • Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: In both the abstract and the introduction, we stated that this paper presents black-box algorithms for DP queries on fully-dynamic streams with a polylogarithmic degradation in the utility compared to static ones. Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We have discussed the limitations in Section 4.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: We stated the assumptions in Section 2, and presented our results with proofs in Section 3.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [NA] Justification: This is a theoretical paper that does not include experiments.
Guidelines:
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [NA]
Justification: This is a theoretical paper that does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [NA]
Justification: This is a theoretical paper that does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [NA]
Justification: This is a theoretical paper that does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: This paper confirms with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This paper studies DP query answering from a theoretical perspective, which should not have a societal impact.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: This paper poses no such risks since it does not involve specific data or model.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA]
Justification: This paper does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] Justification: This paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in the research does not involve LLMs. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: A learning theory approach to noninteractive database privacy Year: (2013)
Ref_id:b1 Title: Concentrated differential privacy: Simplifications, extensions, and lower bounds Year: (2016)
Ref_id:b2 Title: Differentially private histograms under continual observation: Streaming selection into the unknown Year: (2022)
Ref_id:b3 Title: Private and continual release of statistics Year: (2010)
Ref_id:b4 Title: Small summaries for big data Year: (2020)
Ref_id:b5 Title: Differential privacy for growing databases Year: (2018)
Ref_id:b6 Title: Computational geometry: algorithms and applications Year: (2008)
Ref_id:b7 Title: The algorithmic foundations of differential privacy Year: (2014)
Ref_id:b8 Title: Differential privacy under continual observation Year: (2010)
Ref_id:b9 Title: Boosting and differential privacy Year: (2010)
Ref_id:b10 Title: Pure differential privacy for rectangle queries via private partitions Year: (2015)
Ref_id:b11 Title: Differentially private algorithms for graphs under continual observation Year: (2021)
Ref_id:b12 Title: Constant matters: Fine-grained complexity of differentially private continual observation using completely bounded norms Year: (2023)
Ref_id:b13 Title: Privately answering counting queries with generalized gaussian mechanisms Year: (2021)
Ref_id:b14 Title: A multiplicative weights mechanism for privacy-preserving data analysis Year: (2010)
Ref_id:b15 Title: On the geometry of differential privacy Year: (2010)
Ref_id:b16 Title: A simple and practical algorithm for differentially private data release Year: (2012)
Ref_id:b17 Title: Almost tight error bounds on differentially private continual counting Year: (2023)
Ref_id:b18 Title: Counting distinct elements in the turnstile model with differential privacy under continual observation Year: (2023)
Ref_id:b19 Title: Time-aware projections: Truly node-private graph statistics under continual observation Year: (2024)
Ref_id:b20 Title: The composition theorem for differential privacy Year: (2017)
Ref_id:b21 Title: Optimizing linear counting queries under differential privacy Year: (2010)
Ref_id:b22 Title: The matrix mechanism: optimizing linear counting queries under differential privacy Year: (2015)
Ref_id:b23 Title: Privacy integrated queries: an extensible platform for privacy-preserving data analysis Year: (2009)
Ref_id:b24 Title: Data streams: Algorithms and applications Year: (2005)
Ref_id:b25 Title: Optimal private halfspace counting via discrepancy Year: (2012)
Ref_id:b26 Title: Fully dynamic algorithms for graph databases with edge differential privacy Year: ()
Ref_id:b27 Title: Differentially private continual release of graph statistics Year: (2018)
Ref_id:b28 Title: Between pure and approximate differential privacy Year: (2016)
Ref_id:b29 Title: The complexity of differential privacy Year: (2017)
Ref_id:b30 Title: High-Dimensional Probability: An Introduction with Applications in Data Science Year: (2018)
