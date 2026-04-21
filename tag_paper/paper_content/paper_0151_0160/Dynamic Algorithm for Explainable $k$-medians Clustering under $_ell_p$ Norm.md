Title: Dynamic Algorithm for Explainable k-medians Clustering under ℓ p Norm
Abstract: We study the problem of explainable k-medians clustering introduced by Dasgupta, Frost, Moshkovitz, and Rashtchian (2020). In this problem, the goal is to construct a threshold decision tree that partitions data into k clusters while minimizing the k-medians objective. These trees are interpretable because each internal node makes a simple decision by thresholding a single feature, allowing users to trace and understand how each point is assigned to a cluster. We present the first algorithm for explainable k-medians under ℓ p norm for every finite p ≥ 1. Our algorithm achieves an Õ p(log k) 1+1/p-1/p 2 approximation to the optimal k-medians cost for any p ≥ 1. Previously, algorithms were known only for p = 1 and p = 2. For p = 2, our algorithm improves upon the existing bound of Õ(log 3/2 k), and for p = 1, it matches the tight bound of log k + O(1) up to a multiplicative O(log log k) factor. We show how to implement our algorithm in a dynamic setting. The dynamic algorithm maintains an explainable clustering under a sequence of insertions and deletions, with amortized update time O(d log 3 k) and O(log k) recourse, making it suitable for large-scale and evolving datasets.

Section: Introduction
Artificial intelligence systems play an increasingly important role in everyday life, influencing decisions that affect individuals, businesses, and society as a whole. As their impact grows, so does the need for transparency and human oversight. In response, there is a growing emphasis on making AI decisions understandable to people. This has led to the development of models that aim to present their decision-making processes in a clear and interpretable manner.
In this paper, we study algorithms for explainable clustering. The notion of explainable k-means and k-medians clustering was introduced by Dasgupta, Frost, Moshkovitz, and Rashtchian (2020) as a way to make clustering decisions more accessible to humans. Both k-means and k-medians are classical clustering objectives widely used in practice. Here, we focus on k-medians clustering under the ℓ p norm. A k-medians clustering of a dataset X ⊂ R d is defined by a collection of k centers c 1 , c 2 , . . . , c k . Each point x ∈ X is assigned to the closest center in the ℓ p norm, that is, the center minimizing ∥x -c i ∥ p . Consequently, every clustering corresponds to a Voronoi partition under the ℓ p norm. The cost of the clustering is defined as
cost p (X; c 1 , . . . , c k ) = k i=1 x∈Pi ∥x -c i ∥ p ,
where P i denotes the set of points assigned to center c i . We refer to this as unconstrained k-medians clustering.
While this objective is simple to define and machines can easily compute the nearest centers, the resulting cluster assignments are often difficult for humans to interpret. To make clustering more comprehensible to humans, Dasgupta et al. (2020) proposed using threshold decision trees to represent clusterings. They referred to this approach as explainable k-means and k-medians. For k-medians, they considered the ℓ 1 norm. In a threshold decision tree, each internal node compares a single coordinate of the input to a threshold and directs the point to the left or right subtree accordingly. Each leaf of the tree represents a cluster. We denote the center assigned to x by the decision tree as T (x). The cost of the clustering is then defined similarly to the unconstrained case:
cost p (X, T ) = x∈X ∥x -T (x)∥ p .
Assigning a data point to a cluster using a threshold decision tree avoids complex distance computations and instead follows a simple, transparent process: each decision is based on a sequence of threshold comparisons. This makes it clear how a particular assignment was made and which features influenced it.
The central question is how much clustering quality is lost in exchange for interpretability. This trade-off is captured by the the cost of explainability or competitive ratio, defined as the worst-case ratio between the cost of the explainable clustering and that of the optimal unconstrained k-medians clustering:
max X cost p (X, T ) OPT k,p (X)
,
where OPT k,p (X) = min c1,...,c k cost p (X; c 1 , . . . , c k ) denotes the cost of the optimal (unconstrained) k-medians clustering of X. Dasgupta et al. (2020) showed-perhaps surprisingly-that the competitive ratio for explainable k-medians under the ℓ 1 norm does not depend on the number of points in the dataset and can be bounded solely as a function of k; specifically, it is at most O(k). They also established a lower bound of Ω(log k). This result sparked significant interest and led to extensive study of explainable k-medians under the ℓ 1 norm. Makarychev and Shan (2021) and Esfandiari, Mirrokni, and Narayanan (2022) improved the upper bound to Õ(log k); see also Laber and Murtinho (2021) and Gamlath, Jia, Polak, and Svensson (2021) for related results. The approximation factor was later improved to O(log k) by Gupta, Pittu, Svensson, and Yuan (2023) and Makarychev and Shan (2023). Finally, Gupta et al. (2023) established a tight upper bound of (1 + H k-1 ) for the ℓ 1 norm, where H k-1 denotes the (k -1)st harmonic number. Bandyapadhyay, Fomin, Golovach, Lochet, Purohit, and Simonov (2022) developed fixed-parameter tractable algorithms that compute the optimal explainable k-medians clustering under the ℓ 1 norm in time (nd) k+O(1) and n 2d (nd) O(1) . They also proved that the problem is NP-complete and cannot be solved in f (k)n o(k) time for any computable function f (•) unless the Exponential Time Hypothesis (ETH) fails. Gupta et al. (2023) showed that this problem is hard to approximate better than (1/2 -o(1)) ln k unless P=NP.
Beyond the ℓ 1 case, much less was known. For p > 1, the only prior result was due to Makarychev and Shan (2021), who provided a Õ(log 3/2 k)-competitive algorithm and a lower bound of Ω(log k) for the ℓ 2 norm. In this paper, we extend the study of explainable k-medians clustering to general ℓ p norms with finite p ≥ 1. Specifically, we design an algorithm that constructs a threshold decision tree with k leaves, such that the cost of the resulting clustering satisfies
E[cost p (X, T )] ≤ O(p • log 1+1/p-1/p 2 k • log log k) • OPT k,p (X).
This improves upon the best known bound for p = 2, and for p = 1 it matches the optimal guarantee up to an O(log log k) factor. Note that the exponent of the logarithm, 1 + 1/p -1/p 2 , always lies in the interval [1, 1.25].
We now discuss the second contribution of the paper. In recent years, researchers have turned their attention to dynamic clustering algorithms, which maintain a high-quality clustering as the dataset evolves and is continuously updated. Recent work in this area includes papers by Lattanzi and Vassilvitskii (2017); Chan, Guerqin, and Sozio (2018); Cohen-Addad, Hjuler, Parotsidis, Saulpic, and Schwiegelshohn (2019); Deng, Li, and Rabani (2022); Bhattacharya, Costa, Lattanzi, and Parotsidis (2023); Bhattacharya, Costa, Garg, Lattanzi, and Parotsidis (2024); Bhattacharya, Costa, and Farokhnejad (2025).
Dynamic algorithms are typically evaluated based on two key metrics: the update time for insertions and deletions, and the recourse-the number of changes made to the solution (in this case, centers inserted or deleted) in response to each update. Bhattacharya et al. (2025) presented an approximation algorithm with O(1)-approximation ratio, O(log 2 ∆) recourse and Õ(k) update time (where ∆ is an aspect ratio of the metric space).
In this paper, we initiate the study of dynamic algorithms for explainable k-medians clustering. Specifically, we ask whether our explainable algorithm can be combined with state-of-the-art dynamic k-medians clustering algorithms-and we answer this question affirmatively.
Most known algorithms for explainable k-medians clustering first compute a clustering using an existing off-the-shelf method, which we refer to as the reference clustering, and then use it to construct a decision tree. Importantly, this second step is oblivious to the dataset-that is, it relies only on the reference clustering and not on the actual data points. Our algorithm is no exception: it takes as input a set of reference centers and outputs a threshold decision tree whose cost is upper bounded by Õ(p • log 1+1/p-1/p 2 k) times the cost of the reference clustering. However, existing algorithms for explainable clustering are not designed to operate in a dynamic setting.
We present a dynamic implementation of our algorithm, in which the set of reference centers evolves over time through insertions and deletions. Our algorithm supports updates in O(d log 3 k) time and modifies only O(log k) nodes in the tree per update (i.e., it has O(log k) recourse), while maintaining the same Õ(p • log 1+1/p-1/p 2 k) competitive ratio.
Our algorithm can be integrated with the dynamic algorithms for unconstrained k-medians mentioned above. We begin by updating the set of centers using one of these low-recourse algorithms, and then apply our dynamic algorithm to update the decision tree for explainable clustering. Our algorithm can also be used to construct explainable clusterings for multiple values of k -for example, when selecting a suitable k within a given range using the elbow method. In such cases, we can run an algorithm (such as k-means++) that outputs centers incrementally, and feed these centers into our dynamic algorithm, which updates the decision tree on the fly.
this section cite: ['b6', 'b6', 'b6', 'b14', 'b8', 'b12', 'b9', 'b10', 'b16', 'b10', 'b0', 'b10', 'b14', 'b13', 'b5', 'b7', 'b1', 'b2', 'b3', 'b3']

Section: Techniques
Our static algorithm for explainable k-medians under the ℓ p norm builds on and refines a prior algorithm by Makarychev and Shan (2021) developed for the ℓ 2 norm. In this work, we generalize the approach to all ℓ p norms with finite p ≥ 1 and provide a tighter analysis. In particular, for the ℓ 2 norm, we improve the competitive ratio from the previous bound of Õ(log 1.5 k) to Õ(log 1.25 k).
As we noted earlier, our algorithm takes as input a set of reference centers produced by an off-the-shelf clustering algorithm and does not access the dataset points directly.
This algorithm relies on the PARTITION_LEAF procedure. Each call to PARTITION_LEAF takes a cell of the space containing some subset of centers C u and constructs a partial threshold decision tree that partitions the cell into several subcells, each containing at most a γ fraction of the input centers, where γ < 1. We apply PARTITION_LEAF recursively, starting with the cell containing all centers c 1 , . . . , c k , to construct the full decision tree.
PARTITION_LEAF first selects an anchor point within the cell. This anchor, denoted m u , is the median or an approximate median of the centers in C u and remains fixed throughout the execution of PARTITION_LEAF. The procedure partitions the space using random cuts drawn from a specially crafted distribution. Each time a cut is sampled and applied (some cuts may be discarded), the algorithm removes the centers that are separated from the anchor and places them into one of the output parts. Each cut is defined by a coordinate i and a threshold θ, and has the form Lef t = {x : x i < θ} and Right = {x : x i ≥ θ}. If a sampled cut does not separate any centers, it is discarded.
Random cuts in the algorithm are drawn as follows: PARTITION_LEAF selects a random coordinate i ∈ {1, . . . , d}, a random threshold θ ′ ∈ [0, R t ], and a random sign σ ∈ {±1} (where R t is the radius of the cell; see Section 2 for details). It lets θ = m u i + σθ ′ . The cumulative density function for θ ′ is given by x p /R p t . The algorithm terminates when fewer than γn centers remain unseparated from the anchor.
We note that using a uniform distribution for θ (i.e., selecting a random coordinate i and then choosing a threshold θ uniformly at random from [-R t , R t ]) would result in a poor competitive ratio, as illustrated in the following example. Consider a k-medians clustering with the ℓ p norm, defined by k + 1 centers located at the positions e 1 , . . . , e k , and 0, where e i denotes the i-th standard basis vector. We focus on a single data point x with coordinates (ε, . . . , ε). Suppose we pick cuts by selecting a random coordinate i ∈ {1, . . . , d} and a threshold θ ∈ [0, 1] uniformly at random. In this case, a constant fraction of the centers will be separated from the anchor m u in Θ(k) steps. The probability that one of the cuts made during these steps separates x from its closest center (the center located at the origin) is Θ(εk), assuming ε is sufficiently small. If x is separated from 0, it will be assigned to a different center, i.e., one of the vectors e i . In that case, the ℓ p distance from x to the new center is approximately 1. Therefore, the expected cost of the clustering produced by this variant of the algorithm for point x is Θ(εk), while the optimal (unconstrained) cost is εk 1/p . Hence, the competitive ratio of such an algorithm is at least Θ(k 1-1/p ).
In this paper, we prove -through a careful analysis of the algorithm -that the aforementioned choice of random distribution yields an O(p log 1+1/p-1/p 2 log log k) upper bound on the algorithm's competitive ratio.
We then show how to implement our static clustering algorithm in the dynamic setting. Our approach builds on the idea of assigning each decision node a timestamp drawn from an exponential distribution -a technique previously introduced in Gupta et al. (2023); Makarychev and Shan (2023) solely for the purpose of analyzing an explainable clustering algorithm under the ℓ 1 norm. We extend this idea by integrating the exponential clock directly into the algorithm's design. Specifically, we assume that random cuts are selected with arrival rates governed by a Poisson process. Each cut is assigned a timestamp corresponding to its selection time.
The high-level idea behind the dynamic algorithm is as follows. When a new center is inserted, we identify the earliest cut -based on its timestamp -that separates the new center from the anchor. To efficiently find such a cut, we employ data structures that enable this operation in O(d log k) time. We prove that this earliest cut corresponds to the one that would have been used by the static algorithm to separate the center c from the anchor m u . There are two possible cases: either the decision tree already contains a node corresponding to this cut, or it does not. In the latter case, the algorithm creates a new decision node to incorporate the cut.
Implementing this idea presents several challenges. The dynamic PARTITION_LEAF algorithm is not permitted to modify the anchor; consequently, it may need to rebuild the entire decision tree for a cell and its descendants once the number of updates in that cell exceeds a certain threshold. Moreover, the dynamic algorithm must terminate at a fixed time-one that cannot be adjusted as centers are added or removed. As a result, unlike the static version, it cannot stop based on the number of remaining centers falling below a given threshold. In this paper, we address these challenges and present a complete dynamic algorithm for the problem.
this section cite: ['b14', 'b10', 'b16']

Section: Algorithm
In this section, we present our algorithm for constructing an explainable clustering tree for the k-medians problem in ℓ p space. The algorithm takes a set of k centers C as input and produces a binary threshold tree T with k leaves, each leaf containing a distinct center in C. The construction begins by initializing the root node r of the tree with all centers C, and recursively partitioning the centers using the procedure PARTITION_LEAF (as shown in Figure 1). We initiate the construction by calling PARTITION_LEAF(r).
While this algorithm is static, we show an efficient dynamic algorithm that achieves the same behavior as this algorithm in Section 5. To couple the dynamic algorithm with the static algorithm, we present our algorithm based on two oracles: STOPPING_ORACLE and GET_ANCHOR. The STOPPING_ORACLE takes a cut ω and the current subtree T u rooted at u as input and outputs a Boolean value; if it is True, then it stops partitioning centers; otherwise, the algorithm continues to partition centers. This oracle guarantees that when partitioning stops, every leaf in T u contains at most a γ fraction of centers in C u , where γ < 1. The oracle GET_ANCHOR takes a subset of centers C u and returns an anchor point m u ∈ R d such that for each coordinate i ∈ 4. Call PARTITION_LEAF(v) for each leaf v containing more than one center in the subtree rooted at u. 5. Return the tree T u rooted at node u.
this section cite: []

Section: Figure 1: Algorithm PARTITION_LEAF for explainable k-medians in ℓ p
We now describe the procedure PARTITION_LEAF(u). The procedure PARTITION_LEAF(u) operates on a node u that contains a set of centers C u . It first queries the oracle GET_ANCHOR to get an anchor point m u . We always refer to the leaf that contains m u as the main part, and denote it by u 0 . Initially, we set u 0 = u.
PARTITION_LEAF iteratively splits the subset C u using randomized threshold cuts until the STOPPING_ORACLE returns True. In each iteration t, it computes the maximum ℓ p distance from m u to any center in the current main part C u0 , denoted by R t = max c∈Cu 0 ∥c -m u ∥ p . Then, it samples a random threshold cut ω t as follows. A coordinate i t ∈ {1, 2, • • • , d} and a sign σ t ∈ {-1, 1} are chosen uniformly at random. Next, it draws a random variable Z t uniformly from the interval [0, (R t ) p ] and set θ t = (Z t ) 1/p . The resulting threshold cut is ω t = (i t , ϑ t ), where ϑ t = m u i + σ t • θ t . If this threshold cut separates at least two centers in C u0 , the algorithm partitions the current main part u 0 into two disjoint cells. It adds two children u L , u R to the node u 0 and assigns centers C u L = {c ∈ C u0 : c it < ϑ t } to node u L and centers C u R = {c ∈ C u0 : c it ≥ ϑ t } to node u R . The child node, either u L or u R , that contains anchor m u becomes the updated main part u 0 . This process continues until the STOPPING_ORACLE returns True. Finally, it recursively calls the PARTITION_LEAF(v) on each leaf v that contains more than one center in the subtree rooted at u.
this section cite: []

Section: Analysis of approximation factor
In this section, we provide the approximation guarantees for our algorithm. Theorem 3.1. Given a set of points X and a set of k centers C, for any p ≥ 1, Algorithm finds a threshold tree T with k leaves that has k-medians cost E[cost p (X, T )] ≤ O p • (log k)
1+ 1 p -1 p 2 log log k cost p (X, C).
We analyze the approximation guarantee by bounding the expected cost incurred by each point x ∈ X. Fix an arbitrary point x ∈ X and let c ∈ C be its closest center. We show that the expected cost of assigning x in the constructed threshold tree T is bounded by E[cost p (x, T )] ≤ O p • (log k)
1+ 1 p -1 p 2 log log k ∥x -c∥ p .(1)
If x equals its closest center c, then x is always assigned to c by any tree T , and thus incurs zero cost, cost p (x, T ) = 0. In this case, the above bound holds trivially. Therefore, we may assume from now on that x ̸ = c.
Consider the path from the root to the leaf in the tree that contains this point x. We index the node on this path by t = 1, 2, • • • , T , where u 1 is the root of the tree and u T is the leaf that contains x. Let T t be the partially built tree when the node u t is generated in the algorithm. Given any tree T t , let T t (x) be the closest center in the same leaf as x in tree T t . We define the following upper bound on the approximation factor. Definition 3.2. Let A k be the smallest number such that the following inequality holds for every partially built tree
T t , E [cost p (x, T ) | T t ] ≤ A k • ∥x -T t (x)∥ p .
Since all centers are contained in the root u 1 , we have T 1 (x) = c. Thus, we have A k is an upper bound on the approximation factor. We then prove the following lemma, which provides a recurrence relation for bounding A k .
Lemma 3.3. For some absolute constant β > 0, we have for any step
t * E cost p (x, T ) ∥x -T t * (x)∥ p | T t * ≤ 3 + 2A k k + β • p(log k) 1+ 1 p -1 p 2 • log(A k log 2 k).
We first show how to use Lemma 3.3 to get the desired bound on A k , which also provides the approximation factor for the algorithm.
Proof of Theorem 3.1. By Lemma 3.3 and the definition of A k , we get the following recurrence relation on A k , A k ≤ 3 + 2A k k + β • p(log k)
1+ 1 p -1 p 2 • log(A k log 2 k).
Then, we have that A k is bounded by A k ≤ O p(log k)
1+ 1 p -1 p 2 log log k .
By the definition of A k , we bound the expected cost of any point x ∈ X given by tree T as shown in Equation (1). By taking the sum over all points in X, we get the approximation factor for the algorithm.
this section cite: []

Section: Radius and diameter bounds
Before proving the main recurrence lemma, we establish several key results that describe how the radius and diameter of clusters evolve during the recursive partitioning process. These results serve as essential tools in our main proof. We defer the proofs to Appendix A.1.
We first show that the radius R u decreases exponentially in one partition leaf call. Consider any partition leaf call on a node u. Let R t be the radius of the main part before the iteration t of this partition leaf call. Then, we have R 1 = R u . We use T t to denote the partial tree given by the algorithm before the iteration t of this partition leaf call. Lemma 3.4. Consider any partition leaf call on node u. Let L = ⌈2 p+3 d ln k⌉. Then for every t ≥ 1, we have Pr{R t+L > R t /2 | T t } ≤ 1 k 3 .
We define the diameter of a node u to be D u := max c,c ′ ∈Cu ∥c -c ′ ∥ p . We use the following relation between R u and D u for a node u at the beginning of a partition leaf call, which generalizes Lemma 6.1 in Makarychev and Shan (2022) to ℓ p norm. Lemma 3.5 (Lemma 6.1 in Makarychev and Shan (2022)). For every node u on which the algorithm calls partition leaf, we have R u /4 1/p ≤ D u ≤ 2R u .
We define D u for every node u as follows. If the algorithm calls partition leaf on node u, then D u = D u . Now consider any node v in the partition leaf call of a node u, on which the algorithm does not call the partition leaf. Let d(u, v) be the distance from v to u in the tree. We set
D v = max D v , D u • Rv Ru .
By the definition, D u is an upper bound of the diameter D u for every node u. We now show that D u is non-increasing along any path from the root to a leaf in the tree. Since R v is non-increasing in one partition leaf call, D v is also non-increasing in one partition leaf call. Moreover, since D v ≥ D v for every node v and D u = D u on node u where the algorithm calls partition leaf, we have D v is also non-increasing across partition leaf calls.
Lemma 3.6. For every node u, we have R u /4 1/p ≤ D u ≤ 2R u .
We then show that D u decreases exponentially along any path from the root to a leaf in the tree. Lemma 3.7. Let L ′ = ⌈2 2p+6 d ln k⌉. For every node u, let node v be any descendant of u at depth L ′ in the tree T . Then, we have
Pr{ D v ≥ D u /2 | T u } ≤ 4 k 3 .
this section cite: ['b15', 'b15']

Section: Recurrence lemma
In this section, we provide a proof overview of Lemma 3.3, which establishes the recurrence relation of A k . The details of the proof are deferred to Appendix A.2.
We fix an arbitrary point x ∈ X. Without loss of generality, we consider the step t * = 1 and then T t * (x) = c is the closest center to x in C. We then focus on the nodes in T that contain this point x, which form a path from the root to the leaf containing x. We index the node along this path by step t = 1, 2, • • • , T , where u 1 is the root of the tree and u T is the leaf that contains x. Let T t be the partially built tree when the node u t is generated in the algorithm.
We now bound the cost of this point x given by the tree T . We begin by assuming that the radius R t and the diameter substitute D t decrease by a factor of 2 after every L and L ′ steps, respectively. By Lemma 3.4 and 3.7, and applying the union bound over all iterations, this good event holds with probability at least 1 -1/k. If this good event fails to hold, then we simply upper bound the expected cost of x by A k ∥x -c∥ p , which contributes the A k /k factor.
Consider a node u t such that both x and c are contained in u t , and let ω t be the cut sampled at this node. Let C t be the set of centers contained in u t and D t be the diameter of u t . If x and c are separated by this cut ω t , then x is eventually assigned to a different center in C t by T . By the triangle inequality, we have the cost of x in T is at most ∥x -c∥ p + D t . Alternatively, we can use a more refined bound based on the notion of the fallback center, following the approach in Makarychev andShan (2021, 2022). If x is separated from c by this cut ω t , then we define the fallback center of x to be the closest center c ′ ∈ C t+1 to x that is not separated from x by this cut ω t . This fallback center depends on the tree T ′ and the cut ω t . Let M t (ω t ) denote the distance fro m x to the fallback center. Then, by the definition of A k , the expected cost of x can also be upper bounded by A k M t (ω t ).
We now partition the steps {1, 2, • • • , T } into three disjoint cases based on the radius R t and the fallback distance M t (ω) as follows. We introduce the following definitions.
Definition 3.8. For a fixed parameter α > 0, we say that step t is a light step if the radius satisfies
R t ≤ 6 log α k • max ∥x -m t ∥ p , ∥c -m t ∥ p .
Otherwise, step t is called a heavy step.
If x and c are separated by a cut ω t , then we refer to this cut as a light cut if step t is a light step, and a heavy cut if step t is a heavy step.
Definition 3.9. For each step t, we say a cut ω t separating x and c a safe cut if A k M t (ω t ) ≤ Rt 6 p log 2 k . Otherwise, this cut ω t is called an unsafe cut.
Therefore, if x and c are separated by the tree T , then exactly one of the following three events must occur: (1) they are separated by a safe cut; (2) they are separated by a light cut; (3) they are separated by a heavy and unsafe cut. We then show how to bound the contribution of each case to the expected cost separately.
Safe cut: Suppose x and c are contained in node u t . The probability that x and c are separated by the cut ω t is at most
Pr{x & c separated by ω t | T t } ≤ 1 2d • p∥x -c∥ p (∥x -m t ∥ p-1 p + ∥c -m t ∥ p-1 p ) R p t .
In this case, we use A k M t as the upper bound of the expected cost since it is much smaller than the radius R t . We show that 3R t ≥ ∥x -m t ∥ p + ∥c -m t ∥ p . Thus, the expected cost of a safe cut at step t is at most
p 2d • A k Mt Rt • 3 p-1 • ∥x -c∥ p .
In each partition leaf call, we know that M t is non-decreasing as t increases and R t decreases by a factor of 2 after every L steps. Hence, A k M t /R t forms an increasing geometric series in every L steps. Since A k M t /R t ≤ 1/(6 p log 2 k) for safe cuts, the expected cost due to safe cuts in one partition leaf call is at most
L • p 2d • 2 6 p log 2 k • 3 p-1 • ∥x -c∥ p ≤ O 1 log k ∥x -c∥ p .
Combining over all O(log k) partition leaf calls, this case is bounded by O(1) • ∥x -c∥ p .
Light cut: Consider the node u t contains x and c. The probability that x or c is separated from the anchor m t by ω t is at least
Pr{x or c separated from m t by ω t | T t } ≥ 1 2d • max{∥x -m t ∥ p p , ∥c -m t ∥ p p } R p t .
Thus, in each partition leaf call, the probability that x and c are separated by a light cut at the end of the partition leaf call is most
p∥x -c∥ p (∥x -m t ∥ p-1 p + ∥c -m t ∥ p-1 p ) max{∥x -m t ∥ p p , ∥c -m t ∥ p p } .
We upper bound the expected penalty by D t ≤ 2R t ≤ 12 log α k • max {∥x -m t ∥ p , ∥c -m t ∥ p } by the definition of a light cut. Since the number of partition leaf calls is at most O(log k), the expected cost due to a light cut is at most
O(log k) • D t • p∥x -c∥ p (∥x -m t ∥ p-1 p + ∥c -m t ∥ p-1 p ) max{∥x -m t ∥ p p , ∥c -m t ∥ p p } ≤ O(p log 1+α k)∥x -c∥ p .
Heavy and unsafe cut: Consider a heavy step t when x and c are contained in node u t . For each coordinate i, we define U i (t) = {ϑ : (i, ϑ) is unsafe} to be all thresholds ϑ such that the cut ω t = (i, ϑ) is unsafe at step t. Let δ i (t) be the Lebesgue measure of the unsafe threshold U i (t). Then, the probability that x and c are separated by an unsafe cut at the heavy step t is at most
p 2d • d i=1 max{|x i -m t i |, |c i -m t i |} p-1 R p t • δ i (t).
Note that all steps in P s in one partition leaf call s uses the same anchor point m s . Let P ′ s ⊆ P s be all heavy steps in the partition leaf call. We define a vector ∆(s) ∈ R d whose i-th coordinate is ∆ i (s) = t∈P ′ s δ i (t). By summing the above separation probability over all steps in P ′ s and applying Hölder's inequality, the probability that x and c are separated by a heavy and unsafe cut in partition leaf call s is at most
p 2d • ∥∆(s)∥ p • ∥x -m s ∥ p-1 p + ∥c -m s ∥ p-1 p R p t .
In this case, we upper bound the penalty of separation by D t ≤ 2R t . Since R t ≥ 6 log α k • max {∥x -m t ∥ p , ∥c -m t ∥ p } for heavy steps, we have the expected penalty due to heavy and unsafe cuts is at most
p d • 2 (6 log α k) p-1 • S s=1 ∥∆(s)∥ p .
We then bound S s=1 ∥∆(s)∥ p . Since the number of partition leaf calls is S = O(log k), we show that
S s=1 ∥∆(s)∥ p ≤ log 1-1 p k S s=1 ∆(s) p .
Consider any fixed cut ω = (i, ϑ) that separates x and c. This cut is unsafe at step t if and only if M t (ω) ≥ R t /(6 p log 2 k • A k ). Moreover, it always holds M t (ω) ≤ D t . By Lemma 3.6, we have R t ≥ D t and D t ≥ D t . Since by Lemma 3.7, D t decreases by a factor of 2 after every
L ′ = ⌈2 2p+6 d ln k⌉ steps, this cut ω is unsafe in at most L ′ • log(2 • 6 p log 2 k • A k ) steps. Thus, we have S s=1 ∆(s) p ≤ O(4 p • d log k • p log(log 2 k • A k ))∥x -c∥ p .
Therefore, the expected cost due to heavy and unsafe cuts is at most
O (log k) 2-1 p -α(p-1) log(log 2 k • A k ) ∥x -c∥ p .
Finally, combining all three cases and taking α = 1 /p -1 /p 2 , we get the conclusion.
this section cite: []

Section: Lower bounds
In this section, we present two lower bound results for explainable k-medians under ℓ p norms. First, we provide an Ω(log k) lower bound on the competitive ratio of explainable k-medians under ℓ p norm, for any fixed p ≥ 1. Second, we show that no explainable clustering algorithm can, without knowing p in advance, achieve a good competitive ratio simultaneously for all p ≥ 1. In particular, there exists an instance on which any such algorithm incurs a competitive ratio of Ω(d 1/4 ) for some p ≥ 1.
We extend the lower bound instance for explainable k-medians in ℓ 2 by Makarychev and Shan (2021) to all ℓ p norms with p ≥ 1. The proof is provided in Appendix D.
Theorem 4.1. For every p ≥ 1, there exists an instance X ⊆ R d , such that for every threshold tree T , its clustering cost is at least cost p (X, T ) = Ω(log k)OPT k,p (X), where OPT k,p (X) is the ℓ p cost of the optimal (unconstrained) k-medians clustering of X.
The competitive ratio of our algorithm is upper bounded by Õ(p(log k) 1+1/p-1/p 2 ). Thus, for every p > 1, there remains an Õ((log k) 1/p-1/p 2 ) gap, which is maximized at p = 2 as Õ(log 1/4 k).
We then investigate whether it is possible to design an explainable clustering algorithm that, without knowing p in advance, produces a single threshold tree (or a distribution over threshold trees) with a good competitive ratio for all p ≥ 1 simultaneously. The following theorem shows that this is not possible. The proof is in the Appendix C. Theorem 4.2. There exists an instance X ⊆ R d , such that for any distribution over threshold trees, the expected competitive ratio is at least Ω(d 1/4 ) for some p ≥ 1.
this section cite: ['b14']

Section: Dynamic algorithm
In this section, we present a dynamic algorithm for the setting where the input set of points X and centers C change over time. We show that after each update, our algorithm maintains a threshold tree with low k-medians cost and analyze its update time and recourse.
Let X 1 , X 2 , . . . , X t , . . . denote a changing data set after each update t and let C 1 , C 2 , . . . , C t , . . . be the corresponding sequence of center sets. Our goal is to output after each update t a threshold tree T t with |C t | leaves that approximates the clustering of X t with centers C t . Similarly to the static setting, our dynamic algorithm only depends on the center sets to construct the trees T t . Thus, we focus on the setting where the center sets change through a sequence of insertion or deletion requests, i.e. C t = C t-1 ∪ {c}, if t is an insertion request of a new center c, or C t = C t-1 \ {c}, if t is a deletion request of an existing center c ∈ C t-1 . We show the following theorem, with the proof in Appendix B. Theorem 5.1. Given a sequence of requests, where each request is either an insertion or a deletion of a single center in R d , there is a dynamic algorithm that for each center set C t , outputs a threshold tree T t such that for any data set
X ⊆ R d , E[cost p (X, T t )] ≤ O(p • (log k t ) 1+1/p-1/p 2 log log k t ) cost p (X, C t ),
where k t = |C t |. The amortized update time of the algorithm is O(d log 3 k) and the amortized recourse (number of tree nodes updated) is O(log k), where k = max t i=1 |C i |.
Note that naively classifying a data point x using a threshold tree T t takes O(k) time in the worst case, if T t has height O(k). In contrast, our dynamic algorithm efficiently updates the current threshold tree in only O(d log 3 k) time, by modifying on average O(log k) nodes after each request.
Moreover, our dynamic algorithm extends naturally to the fully-dynamic explainable clustering setting, where the input is a stream of insertion or deletion requests of data points instead of centers. Specifically, we invoke a fully-dynamic clustering algorithm by Bhattacharya et al. (2025) to maintain a sequence of center sets C t that provide a constant-factor approximation on X t . Since the algorithm of Bhattacharya et al. (2025) guarantees that only Õ(1) centers change on average after each update, our dynamic algorithm applies directly by treating each center change as a center update request and invoking Theorem 5.1. See Corollary B.6 for the formal statement.
To implement our dynamic algorithm, we reinterpret the PARTITION_LEAF procedure (Figure 1) in an equivalent but more convenient way using the exponential clock. This version generates all random cuts in advance. Without loss of generality, we assume that all centers lie within [-1, 1] d ; otherwise, we rescale the instance accordingly. The procedure generates an infinite sequence of candidate cuts ω 1 , ω 2 , . . . , where each cut ω t = (i t , ϑ t ) is constructed as follows: a coordinate i t , a sign σ t ∈ {-1, 1}, and a parameter Z t ∈ [0, 2 p ] are sampled uniformly at random. The threshold is then set to ϑ t = m it + σ t • (Z t ) 1/p , where m denotes the anchor point. Additionally, each cut ω t is assigned an arrival time ρ t , such that ρ 1 ≤ ρ 2 ≤ . . . follows the arrival times of a Poisson Process with rate λ = 1.
The algorithm attempts the next cut (ω t , ρ t ) in the sequence until the STOPPING_ORACLE returns True. If ω t separates at least two centers from the main part, the cut is made; otherwise, it is ignored. Since the arrival times ρ t are independent of cut choices ω t , this version yields the same distribution of threshold trees as the original PARTITION_LEAF procedure. These arrival times ρ t are crucial for the design of our dynamic algorithm. In the following discussion, we assume there is a data structure that stores this sequence of cuts with their arrival times. It also provides a function GET_EARLIEST_CUT that takes a center c and returns the earliest cut ω from the sequence that separates c and the anchor m.
We provide a dynamic implementation of the PARTITION_LEAF procedure, which we apply recursively to obtain a fully dynamic version of the entire clustering algorithm. The dynamic variant of PARTITION_LEAF supports three operations: (1) REBUILD, (2) INSERT CENTER, and (3) DELETE CENTER. We now briefly describe each of these operations.
Rebuild: Reconstruct the subtree rooted at node u, partitioning all centers in C u into distinct leaves via recursive calls to the PARTITION_LEAF procedure. In particular, GET_ANCHOR(C u ) returns the true coordinate-wise median of the centers C Insert: Suppose a new center c is inserted. The algorithm calls GET_EARLIEST_CUT to find the earliest cut ω in the pre-generated sequence with its arrival time ρ that separates c from the anchor
m u . Let (ω ′ 1 , ρ ′ 1 ), • • • , (ω ′ r , ρ ′ r )
be the cuts currently used in this partition leaf call. Let ρ u be the stopping time assigned to this partition leaf call during its most recent rebuild. We consider three cases as follows: (1
) ρ = ρ ′ j for some j ∈ [r]; (2) ρ > ρ u ; (3) ρ ≤ ρ u and ρ ̸ = ρ ′ j for any j ∈ [r]
. Case (1): Assign this new center c to the node v generated by cut ω ′ j and recursively maintain the partition leaf call rooted at v. Case (2): This new center c remains in the main part u 0 until this partition leaf call ends. We then recursively maintain the partition leaf call on the main part u 0 .
Case (3): It finds the smallest index j ∈ [r] such that ρ < ρ ′ j or sets j = r + 1 if no such index exists. Then we insert this new cut ω at position j and add a new leaf node containing c to the tree.
Delete: Now suppose a center c ∈ C u is deleted. We locate the leaf node containing c in this partition leaf call. If this leaf contains only one center c, we remove both the leaf and the cut that created it. Otherwise, we delete c from the leaf and maintain the next partition call recursively.
this section cite: ['b3', 'b3']

Section: NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
• You should answer [Yes] , [No] , or [NA] .
• [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• Please provide a short (1-2 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
IMPORTANT, please:
• Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist",
• Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: We provide complete proofs of all the claims we make in the abstract, introduction, and other sections of the paper.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [NA]
Justification: The performance of the algorithms is analyzed without making any assumptions about the data.
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
Answer: [Yes] Justification: We provide complete proofs of all our results.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [NA]
Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
7. Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [NA]
Justification: The paper does not include experiments.
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
Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The authors have reviewed the NeurIPS Code of Ethics and confirm that this research was conducted in accordance with it.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10. Broader impacts Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This work focuses on the development and analysis of approximation algorithms for explainable k-medians clustering under general ℓ p norms. It does not develop any technology that has harmful or malicious applications. As such, we believe there are no direct negative societal impacts resulting from the work in its current form.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [NA] Justification: The paper does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release any assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According
to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: A Proofs in Section 3
A.1 Proofs in Section 3.1 Lemma 3.4. Consider any partition leaf call on node u. Let L = ⌈2 p+3 d ln k⌉. Then for every t ≥ 1, we have Pr{R t+L > R t /2 | T t } ≤ 1 k 3 .
Proof of Lemma 3.4. Let C t be the centers contained in the main part before the iteration t of the partition leaf call. Then, we have C 1 = C u be the set of centers contained in node u. Let m u be the median of centers in C u . Consider any center c ∈ C t with ∥c -m u ∥ p > R t /2. Suppose the algorithm chooses the coordinate i at iteration t. Then, this center c is separating from m u at iteration t if and only if σ t = sgn(c i -m u i ) and θ t ∈ (0,
|c i -m u i |]. Thus, we have Pr{c, m u are separated at t | T t , i t = i} = 1 2 |c i -m u i | p R p t .
Combining all coordinates, the probability that c is separated from m u at iteration t is at least
Pr{c, m u are separated at t | T t } = d i=1 1 d • Pr{c, m u are separated at t | T t , i t = i} = d i=1 1 2d • |c i -m u i | p R p t = 1 2d ∥c -m u ∥ p p R p t ≥ 1 2d • 2 p = 1 2 p+1 d .
Since in one partition leaf call, the radius R t is non-increasing as t increases, for any iteration t ′ ≥ t, we have ∥c -m u ∥ p > R t ′ /2. Hence, conditioned on T t , if c is not separated from m u before iteration t ′ ≥ t, then c is separated from m u at iteration t ′ with probability at least 1 /2 p+1 d. Therefore, the probability that c is not separated from m u after L = ⌈2 p+3 d ln k⌉ iterations is at most
1 - 1 2 p+1 d L ≤ e -L 2 p+1 d = 1 k 4 .
Since there are at most k centers with distance to m u greater than R t /2, by the union bound over all such centers, we have
Pr{R t+L > R t /2 | T t } ≤ 1 k 3 .
We show the following relation between the radius R u and the diameter D u for each node u on which the algorithm calls the partition leaf.
Lemma 3.5 (Lemma 6.1 in Makarychev and Shan ( 2022)). For every node u on which the algorithm calls partition leaf, we have R u /4 1/p ≤ D u ≤ 2R u .
Proof of Lemma 3.5. It is easy to get the second bound from the triangle inequality of the ℓ p norm. Let m u be the median of centers in C u . We have for any two centers c, c
′ ∈ C u 1 , ∥c -c ′ ∥ p ≤ ∥c -m u ∥ p + ∥m u -c ′ ∥ p ≤ 2R u .
We then show the first bound. For any function f : C u → R, let avg c∈Cu f (c) = 1 |Cu| c∈Cu f (c) be the average of f (c) over all centers in C u . Let c ′ = arg max c∈Cu ∥c -m u ∥ p be the center that is farthest from the median m u in ℓ p norm. For any pair of centers c, ĉ ∈ C u , the distance between c and ĉ is at most the diameter of u, ∥c -ĉ∥ p ≤ D u . Thus, we have
D p u ≥ avg c∈Cu ∥c ′ -c∥ p p = avg c∈Cu d i=1 |c ′ i -c i | p = d i=1 avg c∈Cu |c ′ i -c i | p .
Since m u is the output of GET_ANCHOR which always returns an approximate median of the centers in C u , at least 1 4 of the centers c ∈ C u lie on the opposite side of the hyperplane {x : x i = m u i } from the center c ′ . Thus, for these centers c ∈ C u , we have
|c ′ i -c i | ≥ |c ′ i -m u i |. As a result D p u ≥ d i=1 avg c∈Cu |c ′ i -c i | p ≥ d i=1 1 4 • |c ′ i -m u i | p = 1 4 • ∥c ′ -m u ∥ p p = 1 4 R p u , which implies R u /4 1/p ≤ D u .
Lemma 3.6. For every node u, we have R u /4 1/p ≤ D u ≤ 2R u .
Proof of Lemma 3.6. For any node u on which the algorithm calls the partition leaf, we have D u = D u . By Lemma 3.5, we have R u /4 1/p ≤ D u ≤ 2R u .
We then consider any node v which is not a partition leaf call node. Let u be the node of partition leaf call that generates the node v. Since D u ≤ 2R u , we have
D u R v /R u ≤ 2R v . Note that D v ≤ 2R v . Thus, we have D v ≤ 2R v . Since D u ≥ R u /4 1/p , we have D v ≥ D u • R v /R u ≥ R v /4 1/p .
We then show that D u decreases exponentially along any path from the root to a leaf in the tree. First, we show that any pair of centers that are far apart in the node are separated with high probability. Let T u be the partial tree when node u is generated in the algorithm.
Lemma A.1. For every two centers c ′ and c ′′ in C u at distance at least D u /2,
Pr{c ′ , c ′′ are separated at u | T u } ≥ 1 d • 2 2p+2 .
Proof. Suppose the algorithm picks coordinate i at node u. For every two centers c ′ , c ′′ ∈ C u , we consider the following two cases: (1) c ′ and c ′′ are on the same side of the median m u in coordinate i; (2) c ′ and c ′′ are on the opposite side of the median m u on coordinate i.
For the first case, without loss of generality, we assume that c ′′ i ≥ c ′ i ≥ m u i . Then, two centers c ′ and c ′′ are separated by the cut at node u if and only if the algorithm picks σ u = 1 and θ u ∈ (c
′ i -m u i , c ′′ i -m u i ].
Let T u be the partial tree when node u is generated. Then, we have
Pr{c ′ , c ′′ are separated at u | i u = i, T u } = 1 2 • (c ′′ i -m u i ) p -(c ′ i -m u i ) p R p u ≥ (c ′′ i -c ′ i ) p 2R p u ,
where the inequality is because x p is convex and increasing on [0, ∞).
For the second case, c ′ i and c ′′ i are on the opposite side of m u i . Assume that c ′ i ≥ m u i ≥ c ′′ i . Thus, centers c ′ and c ′′ are separated by the cut at node u if and only if
σ = +1, θ ∈ (0, c ′ i -m u i ] or σ = -1, θ ∈ (0, c ′′ i -m u i ].
Thus, we have
Pr{c ′ , c ′′ are separated at u | i u = i, T u } = 1 2 • |c ′′ i -m u i | p + |c ′ i -m u i | p R p u ≥ |c ′′ i -c ′ i | p /2 p-1 2R p u ,
where the inequality is from (a
p + b p )/2 ≥ ((a + b)/2) p for a, b ≥ 0 since x p is convex on [0, ∞).
Combining all coordinates, we have the probability that c ′ and c ′′ are separated at node u is at least
Pr{c ′ , c ′′ are separated at u | T u } ≥ d i=1 1 d • |c ′′ i -c ′ i | p (2R u ) p ≥ ∥c ′′ -c ′ ∥ p p d(2R u ) p . Since D u ≥ R u /4 1/p , we have for every two centers c ′ , c ′′ ∈ C u with ∥c ′′ -c ′ ∥ p ≥ D u /2, Pr{c ′ , c ′′ are separated at u | T u } ≥ R p u 4 • 2 p • 1 d(2R u ) p = 1 2 2p+2 d .
Lemma 3.7. Let L ′ = ⌈2 2p+6 d ln k⌉. For every node u, let node v be any descendant of u at depth L ′ in the tree T . Then, we have Pr{ D
v ≥ D u /2 | T u } ≤ 4 k 3 .
Proof of Lemma 3.7. Let u ′ be the node at which the algorithm calls the partition leaf that generates the node v. Then, we consider two cases: (1
) d(u ′ , v) ≥ 4 • L; (2) d(u ′ , v) < 4 • L, where L = ⌈2 p+3 d ln k⌉ used in Lemma 3.4.
In the first case, by Lemma 3.4, we have with probability at least (1 -1/k 3 ) 4 ≥ 1 -4/k 3 (where we used Bernoulli's inequality),
D v ≤ 2R v ≤ 2 • R u ′ 2 4 ≤ 2 • 4 1/p D u ′ 2 4 ≤ D ′ u 2 .
In the second case, we have
d(u, u ′ ) ≥ d(u, v) -d(v, u ′ ) ≥ 2 2p+5 d ln k.
Thus, by Lemma A.1, we have every two centers in node u at distance of at least D u /2 are not separated at node u ′ with probability at most
1 - 1 d • 2 2p+2 2 2p+5 d ln k ≤ 1 k 5 .
By the union bound over all pairs of centers and all nodes, we have with probability at least 1 -1/k 3 , all such pairs are separated at node u ′ . Thus, we have with probability at least 1 -4/k 3
D v ≤ D u ′ = D u ′ ≤ D u 2 .
A.
2 Proof of Lemma 3.3 Lemma 3.3. For some absolute constant β > 0, we have for any step t * E cost p (x, T ) ∥x -T t * (x)∥ p | T t * ≤ 3 + 2A k k + β • p(log k)
1+ 1 p -1 p 2 • log(A k log 2 k).
Proof of Lemma 3.3. Fix an arbitrary point x ∈ X. Without loss of generality, suppose the step t * = 1, in which case T t * (x) = c is the closest center to x in C. Otherwise, if t * > 1, then conditioned on T t * , we consider the subinstance consisting of centers that lie in the same leaf of T t * as x.
We consider all steps in which the algorithm samples a cut to split the node containing x in the partial tree. With a slight abuse of notation, we index these steps by t = 1, 2, . . . . Note that some of these sampled cuts may be rejected by the algorithm if they fail to separate any centers within the node. Let T t be the partially built tree before the cut at step t and let u t be the node containing x in T t .
The sequence of nodes u 1 , u 2 , . . . thus form a path from the root to the leaf in the final tree T that contains x. 2 We divide the iterations into consecutive parts P 1 , • • • , P S , each corresponding to one of the S partition leaf calls. Within each part P s , all steps t ∈ P s for t ∈ P s occur in the same partition leaf call and share the same anchor point m s . Since the STOPPING_ORACLE ensures that for each PARTITION_LEAF call, when partitioning stops, each leaf contains at most a γ fraction of the centers in its root for some constant γ < 1, the number of partition leaf calls is bounded by O(log k).
Suppose that at step t, the point x and the center c are contained in the same node u t before the cut is applied. Let ω t = (i, ϑ) be the cut selected by the algorithm at this step. We define the penalty ϕ t (ω t ), or equivalently ϕ t (i, ϑ), for the cut (i, ϑ) at step t as follows. If x and c are not separated by cut (i, ϑ), then we set ϕ t (i, ϑ) = 0. Otherwise, the penalty is given by
ϕ t (i, ϑ) = E[cost p (x, T ) | T t , ω t = (i, ϑ)] -∥x -c∥ p .
We now show two upper bounds on this penalty term. Conditioned on the partial tree T t , we know that in the final tree T , the point x must eventually be assigned to a center in C ut , the set of centers contained in node u t . By the triangle inequality, the final cost for x is at most ∥x -c∥ p + D t , where D t is the diameter of node u t . Thus, the penalty is at most D t . If x and c are separated by cut (i, ϑ) at iteration t, then we call the center c ′ closest to x in u t+1 as the fallback center. Define M t (i, ϑ) = ∥x -c ′ ∥ p as the distance from x to its fallback center. By the definition of A k , we have the penalty in this case is at most A k • M t (i, ϑ). Combining both bounds, we obtain ϕ t (i, ϑ) ≤ min{D t , A k M t (i, ϑ)}.
Let L = ⌈2 p+3 d ln k⌉ and L ′ = ⌈2 2p+6 d ln k⌉. We define the stopping time τ to be the first step t such that one of the following events happens: (1) R t < ∥x -c∥ p ;
(2) x and c are separated by the cut chosen at step t;
(3) D t ≥ D t-L ′ /2 for t > L ′ ; (4) R t ≥ R t-L /2 for t > L. We define four disjoint events as follows,
•
E 1 = {R τ < ∥x -c∥ p },
• E 2 = {x and c are separated by the cut chosen at step τ } \ E 1 ,
• E 3 = { D τ ≥ D τ -L ′ /2, τ > L ′ } \ (E 1 ∪ E 2 ), • E 4 = {R τ ≥ R τ -L /2, τ > L} \ (E 1 ∪ E 2 ∪ E 3 ).
We call E 1 , E 2 good events and E 3 , E 4 bad events. By Lemma 3.4 and 3.7, we have that the events E 3 and E 4 happen with probability at most Pr{E 3 } ≤ 1/k and Pr{E 4 } ≤ 1/k. If either E 3 or E 4 occurs, we upper bound the expected cost of x in T by A k • ∥x -c∥ p since x and c remain unseparated at step τ . Therefore, the expected cost of point x given by the tree T is at most
E[cost p (x, T )] = E[cost p (x, T ) 1{E 1 ∪ E 2 }] + E[cost p (x, T ) | E 3 ∪ E 4 ] Pr{E 3 ∪ E 4 } ≤ E[cost p (x, T ) 1{E 1 ∪ E 2 }] + A k ∥x -c∥ p • 2 k .
We then bound the expected cost of point x under the good events,
E[cost p (x, T ) 1{E 1 ∪ E 2 }].
When the event E 1 happens, we have x and c are not separated before step τ . Since the diameters of nodes containing x are non-increasing, the final cost for x in this case can be bounded by
∥x -c∥ p + D τ ≤ ∥x -c∥ p + 2R τ < 3∥x -c∥ p .
Thus, we have
E[cost p (x, T ) 1{E 1 }] ≤ 3∥x -c∥ p • Pr{E 1 } ≤ 3∥x -c∥ p .
We now turn to analyzing the event E 2 . We further partition this event based on the step at which x and c are first separated. For each step t ≥ 1, we define E 2,t = {x and c are separated by the cut chosen at step τ & τ = t} \ E 1 .
These events E 2,t are disjoint and we have E 2 = t≥1 E 2,t . Therefore, the expected cost of x under E 2 can be expressed as
E[cost p (x, T ) 1{E 2 }] = ∞ t=1 E[cost p (x, T ) 1{E 2,t }].
We upper bound the expected cost of x under event E 2 by Lemma A.2.
By combining all events E 1 , E 2 , E 3 , E 4 , we have that the expected cost of x is at most
E[cost p (x, T )] = E[cost p (x, T ) 1{E 1 }] + E[cost p (x, T ) 1{E 2 }] + E[cost p (x, T ) 1{E 3 ∪ E 4 }] ≤ 3 + 2A k k ∥x -c∥ p + ∞ t=1 E[ϕ t (ω t ) 1{E 2,t }] ≤ 3 + 2A k k + β • p(log k) 1+ 1 p -1 p 2 • log(A k log 2 k) ∥x -c∥ p ,
where β is an absolute constant. We now proceed to prove Lemma A.2.
Lemma A.2. For some absolute constant β > 0, we have
∞ t=1 E[ϕ t (ω t ) 1{E 2,t }] ≤ β • p(log k) 1+ 1 p -1 p 2 • log(A k log 2 k)∥x -c∥ p .
Proof. Under the event E 2 , the point x and the center c are separated by a cut. We classify the cut that separates x and c into three cases as follows. We first recall the definitions of light and heavy steps, as well as safe and unsafe cuts, given in Definitions 3.8 and 3.9.
Fix a parameter α > 0 which is specified later. We say that the step t is a light step if R t ≤ 6 log α k max{∥x -m t ∥ p , ∥c -m t ∥ p }, where m t is the anchor of the node u t . Otherwise, we call it a heavy step. Furthermore, if the cut separates x and c at a light step, then we call it a light cut; otherwise, it is a heavy cut. Additionally, at step t, we say that a cut ω t = (i, ϑ) that separates x and c is safe, if
A k M t (i, ϑ) < R t 6 p log 2 k .
Otherwise, we call this cut unsafe.
Then, we split the analysis into three cases: (1) safe cuts; (2) light and unsafe cuts;
(3) heavy and unsafe cuts.
Case 1 (Safe cuts): Suppose the event E 2,t happens and x and c are separated by a safe cut ω t = (i, ϑ). By definition, a safe cut satisfies that the distance from x to the fallback center c ′ after separation is significantly smaller than the current radius, specifically A k M t (i, ϑ) < R t /(6 p log 2 k). In this case, we use A k M t (i, ϑ) as an upper bound on the penalty incurred by separating x and c.
For each step t, coordinate i ∈ {1, 2, • • • , d}, and direction σ ∈ {-1, 1}, we define the safe cut set
G t,i,σ = θ : A k M t (i, m t i + σθ) < R t 6 p log 2 k
& (i, m t i + σθ) separates x and c , which contains all parameters θ ∈ R such that the corresponding cut ω t = (i, m t i + σθ) is safe. Then, the expected penalty due to safe cuts is at most
∞ t=1 E [ϕ t (ω t ) 1{ω t is safe} 1{E 2,t }] ≤ ∞ t=1 E A k M t (i t , m t i + σ t θ t ) 1{θ t ∈ G t,it,σt } 1{E 2,t } ≤ ∞ t=1 d i=1 σ∈{-1,1} 1 2d Gt,i,σ A k M t (i, m t i + σθ) • p • θ p-1 R p t • 1{E 2,t } • dθ = ∞ t=1 d i=1 σ∈{-1,1} 1 2d Gt,i,σ A k M t (i, m t i + σθ) R t • p • θ p-1 R p-1 t • 1{E 2,t } • dθ.
Here, the second inequality uses the fact that the coordinate i is chosen uniformly from {1, 2, • • • , d} and the direction σ is chosen uniformly from {-1, 1} and that θ is drawn from a distribution with density pθ p-1 /R p t . The safe cuts are those with θ ∈ G t,i,σ . Now we derive an upper bound for θ/R t to control the integral. Since center c lies in node u t , we have ∥c -m t ∥ p ≤ R t . Additionally, since the event E 1 does not occur, we have R t ≥ ∥x -c∥ p . Using the triangle inequality, we have
∥x -m t ∥ p ≤ ∥x -c∥ p + ∥c -m t ∥ p ≤ 2R t .
Therefore, we have 3R t ≥ ∥x -m t ∥ p + ∥c -m t ∥ p .
Furthermore, for any θ ∈ G t,i,σ , the cut (i, m t i + σθ) separates x and c, which implies θ ≤ max{|x i -m t i |, |c i -m t i |}. Therefore, conditioned on the event 1{E 2,t } = 1, we have for any θ ∈ G t,i,σ ,
θ R t ≤ 3 max{|x i -m t i |, |c i -m t i |} ∥x -m t ∥ p + ∥c -m t ∥ p .
We now analyze each partition leaf call separately. Fix a partition leaf call P s . Throughout this partition leaf call, the anchor m s stays the same. Thus, the expected penalty due to safe cuts within this call is at most
t∈Ps E [ϕ t (ω t ) 1{ω t is safe} 1{E 2,t }] ≤ p 2d d i=1 3 p-1 max{|x i -m s i |, |c i -m s i |} p-1 (∥x -m s ∥ p + ∥c -m s ∥ p ) p-1 t∈Ps σ∈{-1,1} Gt,i,σ A k M t (i, m s i + σθ) R t
• dθ.
By Hölder's inequality, the expected penalty above is at most
p 2d • d i=1 3 p max{|x i -m s i |, |c i -m s i |} p (∥x -m s ∥ p + ∥c -m s ∥ p ) p p-1 p •   d i=1   t∈Ps σ∈{-1,1} Gt,i,σ A k M t (i, m s i + σθ) R t • dθ   p   1 p .
Then, we bound the two terms in the above formula separately. First, we have
d i=1 max{|x i -m s i |, |c i -m s i |} p ≤ ∥x -m s ∥ p p + ∥c -m s ∥ p p .
Thus, we have the first term is
d i=1 3 p max{|x i -m s i |, |c i -m s i |} p (∥x -m s ∥ p + ∥c -m s ∥ p ) p p-1 p ≤ 3 p-1 .
We now bound the second term. Note that for any fixed cut ω = (i, ϑ), the fallback distance M t (i, ϑ) is non-decreasing with respect to the step t. Meanwhile, within each partition leaf call P s , the radius R t is non-increasing and decreases by a factor of 2 after every L steps under event E 2 . Therefore, for each coordinate i ∈ {1, 2, • • • , d}, we have t∈Ps σ∈{-1,1} Gt,i,σ
A k M t (i, m s i + σθ) R t • dθ ≤ t∈Ps σ∈{-1,1} A k M t ((i, m s i + σθ)) R t 1{θ ∈ G t,i,σ } • dθ ≤4L • 1 6 p log 2 k • |x i -c i |,
where the last inequality follows from the definition of safe cuts, which ensures that A k M t (i, ϑ) < Rt 6 p log 2 k whenever θ ∈ G t,i,σ , and A k Mt(i,ϑ) Rt forms a geometric sequence increases by a factor of 2 every L steps. Therefore, we have the second term is at most
  d i=1   t∈Ps σ∈{-1,1} Gt,i,σ A k M t (i, m s i + σθ) R t • dθ   p   1 p ≤ 4L • 1 6 p log 2 k • ∥x -c∥ p .
Since there are at most O(log k) partition leaf calls and L = ⌈2 p+3 d ln k⌉, the expected penalty due to safe cuts is at most
O(log k) • p 2d • 3 p-1 • 4L • 1 6 p log 2 k ∥x -c∥ p ≤ O(p) • ∥x -c∥ p .
Case 2 (Light and unsafe cuts): In this case, we have that the radius R t is relatively small compared to ∥x -m t ∥ p and ∥c -m t ∥ p , specifically, R t ≤ 6 log α k max{∥x -m t ∥ p , ∥c -m t ∥ p }. Therefore, in this case, we use D t ≤ 2R t as an upper bound on the penalty. Then, the expected penalty due to a light and unsafe cut is
∞ t=1 E [ϕ t (ω t ) 1{t is light} 1{ω t is unsafe} 1{E 2,t }] ≤ ∞ t=1 E [ϕ t (ω t ) 1{t is light} 1{E 2,t }] ≤ ∞ t=1 E [2R t 1{t is light} 1{E 2,t }] .
For each step t, suppose both x and c are contained in the node u t . We define the new event E ′ t as the event that either x or c is first separated from the anchor m t by the cut chosen at step t. To bound the expected penalty above, we show that
∞ t=1 E [R t 1{t is light} 1{E 2,t }] ≤ 24p log α k • ∥x -c∥ p • ∞ t=1 E [1{t is light} 1{E ′ t }] .
To show this, we define the stochastic process {Y t } t≥0 as follows. Let Y 0 = 0 and for any t ≥ 1,
Y t = t t ′ =1 (R t ′ 1{E 2,t ′ } -∥x -c∥ p • 24p log α k 1{E ′ t ′ }) 1{t ′ is light}.
We now show that this stochastic process {Y t } t≥0 forms a supermartingale. Note that for each step t ≥ 1, we have
Y t = Y t-1 + (R t 1{E 2,t } -∥x -c∥ p • 24p log α k 1{E ′ t }) 1{t is light}. If step t is heavy, then Y t = Y t-1 .
In the following analysis, we focus on the case where t is a light step and both x and c are contained in the node u t . In this case, we first analyze the probability that the chosen cut separates x and c, and the probability that separates either x or c from the anchor m t .
Claim A.3. Suppose both x and c are contained in the node u t at this step t. Then, the probability that x and c are separated by the chosen cut is at most
Pr{x and c are separated at step t | T t } ≤ p d ∥x -c∥ p ∥x -m t ∥ p-1 p + ∥c -m t ∥ p-1 p R p t .
The probability that either x or c is first separated from m t by the cut chosen at step t is at least
Pr{E ′ t | T t } ≥ 1 2d • max{∥x -m t ∥ p p , ∥c -m t ∥ p p } R p t .
Thus, we have for a light step t,
E[Y t | T t ] -Y t-1 = R t Pr{E 2,t | T t } -24p log α k • ∥x -c∥ p Pr{E ′ t | T t } ≤ p d ∥x -c∥ p ∥x -m t ∥ p-1 p + ∥c -m t ∥ p-1 p R p-1 t -∥x -c∥ p 12p log α k d max{∥x -m t ∥ p p , ∥c -m t ∥ p p } R p t ≤ p d ∥x -c∥ p 2 max{∥x -m t ∥ p-1 p , ∥c -m t ∥ p-1 p } R p-1 t 1 -6 log α k max{∥x -m t ∥, ∥c -m t ∥} R t ≤ 0,
where the last inequality follows from the definition of a light step. Therefore,
{Y t } t≥0 is a su- permartingale. Hence, E[Y T ] ≤ E[Y 0 ] for every fixed T . Since E[Y 0 ] = 0, we have E[Y T ] ≤ 0 and T t=1 E [R t 1{t is light} 1{E 2,t }] ≤ 24p log α k • ∥x -c∥ p • T t=1 E [1{t is light} 1{E ′ t }] .
Letting T → ∞, we obtain
∞ t=1 E [R t 1{t is light} 1{E 2,t }] ≤ 24p log α k • ∥x -c∥ p • ∞ t=1 E [1{t is light} 1{E ′ t }] .
To bound the right-hand side, it suffices to control the expected number of times the event E ′ t occurs. Recall that E ′ t denotes the event that either x or c is first separated from the anchor m t at step t. We begin by noting that the number of partition-leaf calls is at most O(log k). Within each partitionleaf call, the anchor point m t remains fixed, and once x is separated from m t , it will no longer be involved in further cuts associated with that anchor. Therefore, x can be separated from m t in at most one step per partition-leaf call, contributing at most O(log k) occurrences of E ′ t . Additionally, observe that the center c can be separated from the anchor m t without x being separated at most once. After such a separation, c will no longer lie in the same node as x and will not contribute to future events E ′ t . Combining these observations, we conclude that the expected number of steps where
E ′ t occurs is at most O(log k), which yields E ∞ t=1 ϕ t (ω t ) 1{t is light} 1{ω t is unsafe} 1{E 2,t } ≤ O p log 1+α k ∥x -c∥ p .
Case 3 (Heavy and unsafe cuts): Suppose the event E 2,t occurs and that x and c are separated by an unsafe cut ω t = (i, ϑ). For each step t, coordinate i ∈ {1, 2, • • • , d}, and direction σ ∈ {-1, 1}, we define the the corresponding unsafe cut set as and c , that is, the set of threshold θ for which the cut (i, m t i + σθ) is both unsafe and separates x from c. Let δ i,σ (t) = µ(U t,i,σ ) denote the Lebesgue measure of the set U t,i,σ and define δ i (t) = δ i,-1 (t)+δ i,1 (t) as the total measure across both directions for coordinate i.
U t,i,σ = θ : A k M t (i, m t i + σθ) ≥ R t 6 p log 2 k & (i, m t i + σθ) separates x
Thus, the probability that ω t is an unsafe cut is at most
Pr{ω t is unsafe} = 1 2d Ut,i,σ p • θ p-1 R p t • dθ ≤ p 2d d i=1 σ∈{-1,1} max{|x i -m t i |, |c i -m t i |} p-1 R p t • δ i,σ (t) = p 2d d i=1 max{|x i -m t i |, |c i -m t i |} p-1 R p t • δ i (t).
In this case, we use the radius 2R t as the upper bound on the penalty for separating x and c. Therefore, the expected penalty incurred from heavy and unsafe cuts is bounded by
∞ t=1 E [ϕ t (ω t ) 1{ω t is unsafe} 1{t is heavy} 1{E 2,t }] ≤ ∞ t=1 E [2R t Pr{ω t is unsafe} 1{t is heavy} 1{E 2,t }] ≤ ∞ t=1 E R t d i=1 p d δ i (t) max{|x i -m t i |, |c i -m t i |} p-1 R p t 1{t is heavy} 1{E 2,t } . Since step t is heavy, we have R t ≥ 6 log α k • max{∥x -m t ∥ p , ∥c -m t ∥ p }, which implies 1 R p-1 t 1{t is heavy} ≤ 1 (6 log α k) p-1 max{∥x -m t ∥ p , ∥c -m t ∥ p } p-1 .
Substituting this into the previous bound, we obtain that the expected penalty in this case is at most
∞ t=1 E [ϕ t (ω t ) 1{ω t is unsafe} 1{t is heavy} 1{E 2,t }] ≤ ∞ t=1 E d i=1 p d δ i (t) max{|x i -m t i |, |c i -m t i |} p-1 (6 log α k) p-1 max{∥x -m t ∥ p , ∥c -m t ∥ p } p-1 1{E 2,t } .
Note that all steps within the same partition leaf call P s share the same anchor point. Let ms denote the anchor point used in the partition leaf call P s , and define ∆ i (s) = t∈Ps δ i (t). Then, the expected penalty above is at most
E S s=1 t∈Ps d i=1 p d δ i (t) max{|x i -ms i |, |c i -ms i |} p-1 (6 log α k) p-1 max{∥x -ms ∥ p , ∥c -ms ∥ p } p-1 1{E 2,t } ≤ E S s=1 p d d i=1 ∆ i (s) max{|x i -ms i |, |c i -ms i |} p-1 (6 log α k) p-1 max{∥x -ms ∥ p , ∥c -ms ∥ p } p-1 1{E 2,t } . Let ∆(s) denote the d-dimensional vector with coordinates ∆ i (s) for i ∈ {1, 2, • • • , d}. Applying Hölder's inequality, we get ∞ t=1 E [ϕ t (ω t ) 1{ω t is unsafe} 1{t is heavy} 1{E 2,t }] ≤ E     S s=1 p d ∥∆(s)∥ p d i=1 |x i -ms i | p p-1 p + d i=1 |c i -ms i | p p-1 p (6 log α k) p-1 max{∥x -ms ∥ p , ∥c -ms ∥ p } p-1 1{E 2 }     ≤ E S s=1 p d ∥∆(s)∥ p ∥x -ms ∥ p-1 p + ∥c -ms ∥ p-1 p (6 log α k) p-1 max{∥x -ms ∥ p , ∥c -ms ∥ p } p-1 1{E 2 } ≤ p (6 log α k) p-1 d E S s=1 ∥∆(s)∥ p • 1{E 2 } .
Finally, we use the following claim to bound the expected penalty.
Claim A.4. We have
E S s=1 ∥∆(s)∥ p • 1{E 2 } = O 4 p • d • log 2-1 p k • log(6 p • A k • log 2 k) ∥x -c∥ p .
By Claim A.4, we have that the expected penalty in this case is at most
p (6 log α k) p-1 d E S s=1 ∥∆(s)∥ p • 1{E 2,t } ≤ p (6 log α k) p-1 d • O 4 p • d • log 2-1 p k • log(6 p • A k • log 2 k) ∥x -c∥ p ≤O (log k) 2-1 p -α(p-1) • log(A k • log 2 k) ∥x -c∥ p .
Combining all three cases and setting α = 1 /p -1 /p 2 we get the conclusion.
To complete the proof, we prove Claim A.3 and A.4 below. Proof of Claim A.3. We first analyze the probability that x and c are separated by the cut chosen at step t. To bound the separation probability, we fix a coordinate i ∈ {1, 2, • • • , d} and consider the probability that the cut on coordinate i separates x and c. Suppose x and c are on the same side of anchor m t in coordinate i. Then, the threshold cut ω t = (i, m t i + σθ) separates x and c if and only if σ has the same sign as x i -m t i and θ is between |x i -m t t | and |c i -m t i |. Thus, the separation probability on this coordinate is at most 1
2 • ||c i -m t i | p -|x i -m t i | p | R p t ≤ p • max{|x i -m t i | p-1 , |c i -m t i | p-1 } R p t • |x i -c i |,
where the inequality is from the mean value theorem. Suppose x and c are on the opposite side of anchor m t in coordinate i. Then, the separation probability on this coordinate is at most
1 2 • |c i -m t i | p + |x i -m t i | p R p t ≤ p • max{|x i -m t i | p-1 , |c i -m t i | p-1 } R p t • |x i -c i |.
Combining all coordinates and applying Hölder's inequality, we obtain
1 d d i=1 p • max{|x i -m t i | p-1 , |c i -m t i | p-1 } R p t • |x i -c i | ≤ p d • R p t ∥x -c∥ p •   d i=1 |x i -m t i | p p-1 p + d i=1 |c i -m t i | p p-1 p   ≤ p d ∥x -c∥ p • ∥x -m t ∥ p-1 p + ∥c -m t ∥ p-1 p R p t .
For point x, the probability that it is separated from m t at step t is given by
1 2d d i=1 |x i -m t i | p R p t = 1 2d • ∥x -m t ∥ p p R p t .
An identical argument applies to the center c, yielding the same expression with ∥c-m t ∥ p p . Therefore, the probability that either x or c is separated from m t by the threshold cut at step t is at least
1 2d • max{∥x -m t ∥ p p , ∥c -m t ∥ p p } R p t , as claimed.
To prove Claim A.4, we first show the following lemma.
Lemma A.5. For k vectors v 1 , • • • , v k ∈ R d that are entrywise non-negative, we have k i=1 ∥v i ∥ p ≤ k 1-1 p • k i=1 v i p .
Proof. We first upper bound the left-hand side. By Hölder's inequality, we have
k i=1 ∥v i ∥ p = k i=1 1 • ∥v i ∥ p ≤ k 1 q k i=1 ∥v i ∥ p p 1 p = k 1-1 p k i=1 ∥v i ∥ p p 1 p .
We then lower bound the right-hand side. Since vectors v 1 , • • • , v k are nonnegative in every coordinate, we have for any coordinate j,
k i=1 v i j p ≥ k i=1 (v i j ) p .
Combining all coordinates, we have
k i=1 v i p p = d j=1 k i=1 v i j p ≥ d j=1 k i=1 (v i j ) p = k i=1 ∥v i ∥ p p .
Combining the two parts, we get the conclusion.
Proof of Claim A.4. By Lemma A.5 and the number of partition leaf calls is at most O(log k), we have
E S s=1 ∥∆(s)∥ p • 1{E 2 } ≤ O log 1-1 p k E   S s=1 ∆(s) p • 1{E 2 }   .
For any fixed coordinate i, we have
S s=1 ∆ i (s) = S s=1 t∈Ps δ i (t) = ∞ t=1 δ i (t) = ∞ t=1
1{θ ∈ U t,i,1 }dθ + 1{θ ∈ U t,i,-1 }dθ.
We now show that each cut ω = (i, ϑ) that separates x and c is unsafe in at most
L ′′ = L ′ • log(6 p • A k • log 2 k) steps.
Consider any cut ω = (i, ϑ) that separates x and c. This cut ω is unsafe at step t if and only if R t ≤ 6 p log 2 k • A k M t (i, ϑ). For every step t, by the triangle inequality, the penalty to the fallback center is at most M t (i, ϑ) ≤ D t ≤ D t . We know that M t (i, ϑ) is non-decreasing as t increases. Let t ω be the first step when ω is unsafe. Let t ′ ω be the last step when ω is unsafe. Then, by the definition of unsafe cut, we have R tω ≤ 6 p log 2 k • A k M tω (i, ϑ). Then, we have
D t ′ ω ≥ M t ′ ω (ω) ≥ M tω (ω) ≥ R tω 6 p • log 2 k • A k . Since R t /4 1/p ≤ D t ≤ 2R t , we have D t ′ ω ≥ R tω 6 p • log 2 k • A k ≥ D tω 2 • 6 p • log 2 k • A k .
By Lemma 3.7, we have D t decreases by a factor of 2 after L ′ = ⌈2 2p+6 d ln k⌉ steps. Thus, we have that the number of unsafe steps is at most
t ′ ω -t ω ≤ L ′ • log 2 (2 • 6 p • log 2 k • A k ) ≤ O(4 p • d • log k • p log(A k • log 2 k)).
Therefore, we have that when the event E 2 happens,
S s=1 ∆ i (s) ≤ O(4 p • d • log k • log(6 p • A k • log 2 k))|x i -c i |.
Hence, combining all coordinates, we have
E   S s=1 ∆ u (s) p 1{E 2 }   = O(4 p • d • log k • log(6 p • A k • log 2 k))∥x -c∥ p ,
which completes the proof.
this section cite: []

Section: B Dynamic algorithm implementation and analysis
In this section, we provide the full description of the dynamic algorithm, along with an analysis of its approximation guarantee, update time, and recourse.
this section cite: []

Section: B.1 Dynamic algorithm and approximation guarantee
We begin by presenting the detailed dynamic algorithm and proving that, after each update, the distribution of its output is equivalent to that of a corresponding static algorithm. Lemma B.1. Given a sequence of k requests, where each request is either an insertion or a deletion of a single center, let T t be the threshold tree maintained by the dynamic algorithm for the center set C t . Let T ′ t be the tree constructed by the static algorithm PARTITION_LEAF (Figure 1) with specific oracles on centers C t . Then, the two trees are identically distributed T t d = T ′ t . The following corollary is immediate from Lemma B.1 and Theorem 3.1. Corollary B.2. Given a sequence of requests, where each request is either an insertion or a deletion of a single center, the dynamic algorithm provides a threshold tree T t for each center set C t such that for any set of points X, E[cost p (X, T t )] = O p(log k t )
1+ 1 p -1 p 2 log log k t cost p (X, C t ), where k t = |C t |.
We provide a dynamic implementation of the PARTITION_LEAF procedure in Figure 2, which is applied recursively to obtain a fully dynamic version of the entire clustering algorithm. The dynamic variant of PARTITION_LEAF supports three operations: (1) REBUILD, (2) INSERT CENTER, and (3) DELETE CENTER.
We begin with the REBUILD operation, which reconstructs the subtree from scratch using the PARTITION_LEAF procedure as follows.
REBUILD: Reconstruct the subtree rooted at node u, partitioning all centers in C u into distinct leaves via recursive calls to the PARTITION_LEAF procedure. During each such PARTITION_LEAF call on node v in this operation, the following oracle outputs are used and remain fixed throughout subsequent updates until the next rebuild:
• GET_ANCHOR sets the anchor m v as the coordinate-wise median of centers in C v .
• STOPPING_ORACLE determines whether to stop accepting further cuts based on a stopping time ρ v . It returns True if and only if the the timestamp ρ of the input cut ω satisfies ρ > ρ v . The stopping time ρ v is defined during the rebuild as the timestamp of the last accepted cut such that the main part v 0 contains at most half of centers in C v , i.e.
|C v0 | ≤ |C v |/2.
We now describe the condition under which the rebuild operation is triggered in the dynamic algorithm.
Let u be the node on which this operation is applied. Suppose a center c is inserted into or deleted from the set of centers assigned to u. For each partition leaf call, we maintain a counter that tracks the number of such updates since the last rebuild. Let k ′ be the number of centers in node u at the time of the last rebuild. When the update count exceeds k ′ /4, we rebuild the partial tree rooted at node u.
We now proceed to handle the update.
INSERT CENTER: Suppose a new center c is inserted in the subtree rooted at a node u. The algorithm calls GET_EARLIEST_CUT to find the earliest cut ω in the pre-generated sequence with its arrival time ρ that separates c from the anchor
m u . Let (ω ′ 1 , ρ ′ 1 ), • • • , (ω ′ r , ρ ′ r )
be the cuts currently used in this partition leaf call. Let ρ u be the stopping time assigned to this partition leaf call during its most recent rebuild. We consider three cases as follows: (1) ρ = ρ ′ j for some j ∈ [r];
(2) ρ > ρ u ; (3) ρ ≤ ρ u and ρ ̸ = ρ ′ j for any j ∈ [r]. Case (1): Assign this new center c to the node v generated by cut ω ′ j and recursively maintain the partition leaf call rooted at v. Case (2): This new center c remains in the main part u 0 until this partition leaf call ends. We then recursively maintain the partition leaf call on the main part u 0 .
Case (3): It finds the smallest index j ∈ [r] such that ρ < ρ ′ j or sets j = r + 1 if no such index exists. Then we insert this new cut ω at position j and add a new leaf node containing c to the tree. DELETE CENTER: Now suppose a center c ∈ C u is deleted. We locate the leaf node containing c in this partition leaf call. If this leaf contains only one center c, we remove both the leaf and the cut that created it. Otherwise, we delete c from the leaf and maintain the next partition call recursively.
Algorithm DYNAMIC_PARTITION_LEAF Input: A sequence of updates Q = (q 1 , q 2 , . . . ), where each q t is either: INSERT(c): insert a new center c ∈ R d ; or DELETE(c): delete a center c. Output: For each update q t in Q, maintain a threshold tree T t over the current center set C t . Main(Q):
1. Initialize the root r to be empty. 2. For each update q t at time t:
• If q t is INSERT(c): call INSERT_CENTER(c, r) where r is the root.
• If q t is DELETE(c): call DELETE_CENTER(c, r) where r is the root.
• Output the updated tree T t .
Procedure REBUILD(u):
1. Let C u be the current center set at u, set anchor m u be the coordinate-wise median of centers C u . Initialize the main part u 0 = u. 2. Initialize an update counter at u to be Cnt u = 0 and set k u = |C u |. 3. Compute a sequence of candidate cuts {(ω t , ρ t )} using an exponential clock: For each t, sample i t ∈ [d], σ t ∈ {-1, 1}, Z t ∼ Unif[0, 2 p ]. Define the cut ω t = (i t , ϑ t ), where ϑ t = m it + σ t (Z t ) 1/p . Assign the timestamps ρ t as the arrival times of a Poisson process. 4. Iterate over the cuts ω t in increasing order of their timestamps ρ t . Accept it iff ω t separates two centers in the main part u 0 . After each accepted cut, update the main part u 0 to the side containing the anchor. Stop when the main part contains fewer than |C u |/2 centers. Then, set the stopping time ρ u to be the timestamp of the last accepted cut, ρ u = max{ρ t : cut ω t is accepted}.
5. Call REBUILD(v) for each leaf v containing more than one center in the subtree rooted at u.
Procedure INSERT_CENTER(c, u):
1. Increment update counter at u; if updates exceed k u /4, call REBUILD(u). 2. Get the earliest cut (ω, ρ) = GET_EARLIEST_CUT(c) that separates c and m u . 3. Let (ω ′ 1 , ρ ′ 1 ), . . . , (ω ′ r , ρ ′ r ) be cuts used by u and ρ u be the stopping time. 4. If ρ = ρ ′ j for some j: Assign c to node v separated by the cut ω ′ j , and call INSERT_CENTER(c, v). 5. If ρ > ρ u : Assign c to the main part u 0 , and call INSERT_CENTER(c, u 0 ). 6. If ρ ≤ ρ u and ρ ̸ = ρ ′ j for every j: Insert new cut ω into the sequence of cuts used by u, maintaining increasing order by ρ. Create a new leaf node containing c, and attach it to the tree at the cut point.
Procedure DELETE_CENTER(c, u):
1. Increment update counter at u; if updates exceed k u /4, call REBUILD(u). 2. Locate the leaf node v containing c.
this section cite: []

Section: 3.
If the leaf contains only c: Remove both the leaf and its parent cut. 4. Else: Delete c from the leaf v and call DELETE_CENTER (c, v). Proof of Lemma B.1. We describe an implementation of the static algorithm on the set of centers C t , using specific oracles GET_ANCHOR and STOPPING_ORACLE.
To couple with the dynamic algorithm, we mirror each partition leaf call currently maintained in the dynamic algorithm solution. We begin with the partition leaf call at the root node. Let t ′ ≤ t denote the time of the most recent rebuild of this root partition leaf as of time t, and let k t ′ = |C t ′ | be the number centers present at that rebuild time. Assume both the dynamic and static algorithms use the same infinite sequence of candidate cuts with associated timestamps for the root PARTITION_LEAF call.
For any fixed sequence of cuts with timestamps, let m r be the anchor and ρ r be the stopping time used by the dynamic algorithm for this root partition leaf. In the static algorithm, we adopt the same oracles as the dynamic one: the oracle GET_ANCHOR returns m r and STOPPING_ORACLE returns True if and only if the timestamp of the input cut exceeds ρ r . As a result, the static algorithm accepts exactly the same sequence of cuts as the dynamic algorithm. Therefore, the partial tree rooted at r produced by this PARTITION_LEAF call in the static algorithm is identical to that maintained by the dynamic algorithm. We will show that these two oracles are valid for the static algorithm, which means they satisfy the required properties in Section 2.
We first show that GET_ANCHOR returns an approximate median of centers C t . Because this is the most recent rebuild of the root node r, there have been fewer than k t ′ /4 updates since then. Note that the anchor m r is chosen as the coordinate-wise median of all centers in C t ′ at time t ′ . For each coordinate i, at most half of the centers in C t ′ lie on either side of m r . Hence, even after k t ′ /4 updates, there remain at most 3k t /4 centers in C t on either side of m r along every coordinate. 3Therefore, the anchor m r remains an approximate median for the current set of centers C t .
We next show that the STOPPING_ORACLE guarantees that when partitioning stops, every leaf contains at most a 3/4 fraction of centers in C t . Consider any leaf that is separated from the main part during the partitioning. Each such leaf contains only centers that lie on one side of the anchor m r along the coordinate used by the cut that separates it. Since the anchor m r is an approximate median of centers in C t , at most 3k/4 centers lie on either side of m r along every coordinate. Therefore, each separated leaf contains at most a 3/4 fraction of centers in C t . As for the main part, recall that at the stopping time ρ r during the last rebuild, it contains at most k t ′ /2 centers in C t ′ . After at most k t ′ /4 updates, the main part contains at most a 3/4 fraction of centers in C t .
At each recursive step, we use the same sequence of cuts and adopt the corresponding anchor and stopping time used by the dynamic algorithm. This guarantees that the static algorithm mirrors the behavior of the dynamic one at every level of the recursion. Therefore, the static algorithm constructs exactly the same threshold tree as the dynamic algorithm. This completes the coupling argument and establishes that the output of the dynamic algorithm is identically distributed to that of the static algorithm on input C t .
nodes. The recourse in this case is R(i) = (2k ′ -1) + (2k ′ -3) = 4k ′ -4. In either case, we have the bound R(i) ≤ 4k ′ .
We now analyze the total recourse for S 2 . Each node u on which the algorithm calls a REBUILD stores an update counter Cnt u . This update counter is initialized to zero when the node is rebuilt and is incremented by one each time an update (insertion or deletion) involves node u. This node u also stores the number of centers k u in this node when it is rebuilt. Since the dynamic algorithm rebuilds this node u after k u /4 updates, we have k ′ ≤ k u + k u /4. Therefore, we have Cnt ui = k ui /4 ≥ k ′ /5. Hence, we have
i∈S2 R(i) ≤ i∈S2 20 • Cnt(u i ).
(
The right-hand side of ( 3) is bounded by the total number of times any node's counter is incremented.
According to the analysis in Lemma B.1, the dynamic algorithm guarantees that after the partition leaf call of a node u, each leaf has at most a 3/4 fraction of the centers contained in u. Let k = max t i=1 |C i | be the maximum number of centers during the first t requests. Therefore, each update request is involved in at most O(log k) calls of INSERT_CENTER or DELETE_CENTER. Thus, the total number of times any node's counter is incremented is bounded by O(t log k). Combining this with (2) and (3), we conclude that t i=1 R(i) = O(t log k) and thus the amortized recourse is O(log k).
Update Time: As in the amortized recourse analysis, let S 1 ⊆ [t] be the set of time steps where REBUILD is called on some node u i , and let S 2 = [t] \ S 1 . We now split the analysis into two cases, depending on whether or not a rebuild is triggered.
Case 1 (i ∈ S 1 ): Suppose the request i is an insertion of center c i . Let u 1 , u 2 , . . . , u l be the nodes for which INSERT_CENTER(c i , u j ) is called. Each such call on node u takes O(d log k) time, where
k = max t i=1 |C i |. It takes • O(d log k)
time to update the d self-balancing binary search trees stored in u;
• O(d log k) time to compute the earliest cut through GET_EARLIEST_CUT(c i );
• O(log k) time to locate this earliest cut and insert the center by searching the self-balancing binary search tree that maintains all cuts (ω ′ 1 , ρ ′ 1 ), (ω ′ 2 , ρ ′ 2 ), . . . , (ω ′ r , ρ ′ r ) currently used in the partition leaf call of u.
Since the center c i is involved in at most O(log k) INSERT_CENTER calls, the update time for an insertion request i ∈ S 1 is Time(i) = O(d log 2 k). The same asymptotic bound holds for deletions, as finding the leaf that contains the deletion center c i takes O(d log 2 k) time, and the removal takes constant time. Thus, we have i∈S1 Time
(i) = O(|S 1 | • d log 2 k).(4)
Case 2 (i ∈ S 2 ): Let u i be the node that is rebuilt at request i. As in Case 1, the time to process the request before the rebuild is O(d log 2 k). If u i contains k ′ centers at this request i, then REBUILD(u i ) takes O(k ′ d log 2 k) time.
Since when REBUILD(u i ) is triggered, we have the update counter Cnt ui ≥ k ′ /5. Thus, we charge the rebuild time to the update counter. That is the update time Time(i) ≤ O(Cnt ui • d log 2 k).
Therefore, we have
i∈S2 Time
(i) ≤ O(d log 2 k) • i∈S2 Cnt ui .(5)
By the analysis in recourse, we have i∈S2 Cnt ut ≤ O(t log k). Combining (4) and ( 5), we obtain that the total update time is at most t i=1 Time(i) = O(td log 3 k) and so the amortized update time is O(d log 3 k).
We now prove the main theorem of the dynamic algorithm.
Proof of Theorem 5.1. By Corollary B.2 and Lemma B.3, we get the approximation guarantee, amortized recourse, and the amortized update time of the dynamic algorithm.
this section cite: []

Section: B.3 Fully Dynamic Explainable Clustering Algorithm
In this section, we provide a fully dynamic explainable clustering algorithm for the setting in which the clustering data set evolves over time through insertions or deletions of data points. This algorithm maintains an explainable k-clustering that is competitive against the optimal (unconstrained) kclustering. This setting contrasts with Sections B.1 and B.2, where the cluster centers change over time.
Formally, the input is a stream of updates on the data set, where each update is an insertion or deletion of a data point. This generates a sequence of datasets X 1 , X 2 , . . . . If t is an insertion request of a new data point x t , then X t = X t-1 ∪ {x t }, whereas if t is a deletion request of an existing data point x t ∈ X t-1 , then X t = X t-1 \ {x t }. We obtain our fully dynamic explainable clustering algorithm by combining our dynamic algorithm from Section 5 with the fully dynamic k-medians algorithm of Bhattacharya et al. (2025). This fully dynamic k-medians algorithm maintains a constant-factor approximation while changing only Õ(1) centers per update. Corollary B.6. Given a positive integer k and a stream of updates that are insertion or deletion requests of data points in R d , for every p ≥ 1 there exists a fully-dynamic explainable clustering algorithm that outputs a threshold tree T t for every t ≥ 1 satisfying
1. E[cost p (X t , T t )] ≤ O p(log k)
1+ 1 p -1 p 2 log log k OPT k,p (X t ), 2. the expected amortized update time is Õ(kd + (log ∆) 2 d log 3 k), 3. the expected amortized recourse is O((log ∆) 2 log k)
where ∆ is the aspect ratiofoot_2 of all data points in X = t i=1 X i , OPT k,p (X t ) is the ℓ p cost of an optimal (unconstrained) k-medians clustering of X t and Õ hides polylogarithmic factors in ∆, k and n = |X|.
To prove Corollary B.6, we first show how to combine any fully-dynamic (unconstrained) k-medians clustering algorithm under the ℓ p norm with our dynamic algorithm from Section 5 to get a fullydynamic explainable clustering algorithm. Definition B.7. An algorithm A is an (α, u, r) dynamic k-medians clustering algorithm under the ℓ p norm, if for every stream of updates that are insertion or deletion requests of data points, the algorithm outputs k centers C t after each update t, such that E[cost p (X t , T t )] ≤ α OPT k,p (X t ), the expected amortized update time is u and the expected amortized recourse is r.
Fix an iteration t of an (α, u, r) dynamic k-medians clustering algorithm under the ℓ p norm for p ≥ 1. After processing the t-th update request, the algorithm updates the current set of centers from C t-1 to C t . To apply Theorem 5.1, we treat each c ∈ C t-1 \ C t as a deletion from the current center set C t-1 and each c ∈ C t \ C t-1 as an insertion into it. Algorithm 3 formalizes this procedure, and its performance guarantees are proved in Proposition B.8. Proposition B.8. Given a positive integer k, a stream of updates that are insertion or deletion requests of data points in R d , and an (α, u, r) dynamic k-medians clustering algorithm A under the ℓ p norm for some p ≥ 1, Algorithm 3 outputs a threshold tree T t for every time t ≥ 1 satisfying 1. E[cost p (X t , T t )] ≤ O α • p(log k)
1+ 1 p -1 p 2 log log k OPT k,p (X t )
2. the expected amortized update time is O(u + r • d log 3 k)
3. the expected amortized recourse is O(r • log k).
Algorithm FULLY_DYNAMIC_PARTITION_LEAF Input: an integer k, a number p ≥ 1, a stream of update requests of data points q 1 , q 2 , . . . and an (α, u, r) dynamic k-medians clustering algorithm A under the ℓ p norm. Output: threshold trees T 1 , T 2 , . . . 1. Initialize the root root to be empty. 2. Initialize C 0 to be an empty set of centers. 3. For every t ≥ 1:
• Run algorithm A to process request q t and get a new set of centers C t .
• For every center c ∈ C t-1 \ C t : Call DELETE_CENTER(c, root) in Figure 2.
• For every center c ∈ C t \ C t-1 :
Call INSERT_CENTER(c, root) in Figure 2.
• Output the threshold tree T t rooted at root. Before we prove Proposition B.8, we show how it yields Corollary B.6 by choosing the fully dynamic k-medians algorithm A by Bhattacharya et al. (2025). Proof of Corollary B.6. The dynamic algorithm for k-medians from Bhattacharya et al. (2025) achieves an O(1) approximation. It has O(log 2 ∆) expected amortized recourse and Õ(kd) expected amortized update time.foot_3 As a result, by Proposition B.8, we get the conclusion. We proceed to prove Proposition B.8. Proof of Proposition B.8. Fix any t ≥ 1. For every i ∈ {1, 2, . . . , t}, let C i denote the set of centers produced by A after processing the i-th request. Let r i = |C i △C i-1 | denote the recourse at time i. During iteration i, Algorithm 3 produces r i intermediate center sets C ′ i,1 , C ′ i,2 , . . . , C ′ i,ri = C i corresponding to the individual center update requests applied to C i-1 . Since deletions are processed before insertions, each intermediate set has size at most k. Let T ′
i,1 , T ′ i,2 , . . . , T ′ i,ri denote the intermediate threshold trees produced by Algorithm 3 after each center update during iteration i. For the rest of the proof, we condition on a fixed sequence of center sets C ′ 1,1 , C ′ 1,2 , . . . , C ′ t,rt = C t . Approximation: Applying Theorem 5.1, for every i ∈ {1, 2, . . . t} and j ∈ {1, 2, . . . , r i } the following inequality holds:
E[cost p (X i , T ′ i,j ) | C ′ 1,1 , . . . , C ′ t,rt ] ≤ O p(log k) 1+ 1 p -1 p 2 log log k cost p (X i , C ′ i,j
). Therefore, choosing j = r t we obtain E[cost p (X t , T t ) | C ′ 1,1 , . . . , C ′ t,rt ] ≤ O p(log k)
1+ 1 p -1 p 2 log log k cost p (X t , C t ).
Taking the expectation at both sides of the inequality and using the fact that A is an α-approximation algorithm, the approximation guarantee follows.
Recourse: By Theorem 5.1, the amortized recourse of DYNAMIC_PARTITION_LEAF is O(log k) with probability 1. Hence, after processing t requests, the total number of tree nodes modified is O(R log k), where R = t i=1 r i denotes the total recourse of algorithm A, i.e., the total number of center update requests. Therefore, the expected total number of tree nodes modified up to the t-th request is O(E[R] log k) = O(rt log k), which corresponds to the expected total recourse. Dividing by t, we obtain the expected amortized recourse of O(r log k).
Update Time: The total update time of Algorithm 3 equals the sum of the running time of A for processing all requests and the time taken by DYNAMIC_PARTITION_LEAF to handle all R = t i=1 r i center update requests. By Theorem 5.1, the amortized update time of DYNAMIC_PARTITION_LEAF is O(d log 3 k) with probability 1. Thus, the total update time is O(U + Rd log 3 k), where U = t i=1 u t is the total running time of A. Since the expected amortized update time and recourse of A are u and r respectively, the total expected update time of Algorithm 3 is O(ut + rt • d log 3 k) and the expected amortized update time guarantee follows.
this section cite: ['b3']

Section: C Lower bound for universal algorithms
In this section, we provide a lower bound on the competitive ratio for any universal explainable clustering algorithm. A universal algorithm is required to output a distribution over threshold trees that perform well for all p ≥ 1 without the prior knowledge of p.
Our algorithm for explainable k-medians clustering under ℓ p norm samples threshold cuts from a carefully designed distribution that depends crucially on p. A natural question is whether there exists an explainable clustering algorithm that is independent of p while achieving a good approximation to the optimal ℓ p cost for all p ≥ 1 simultaneously. We answer this question in the negative by showing an Ω(d 1/4 ) lower bound on the worst-case competitive ratio of any universal explainable clustering algorithm.
Theorem 4.2. There exists an instance X ⊆ R d , such that for any distribution over threshold trees, the expected competitive ratio is at least Ω(d 1/4 ) for some p ≥ 1.
Proof. The instance has two centers, one at the origin c 1 = (0, 0, . . . , 0), and the other at c 2 = (1 + d 3/4 , 1, . . . , 1), along with many data points co-located at each center and one special point x = (1, 1, . . . , 1). We show that any distribution D over threshold trees (a single threshold cut in this case) yields an explainable clustering such that either the ℓ 1 or the ℓ 2 cost is in expectation Ω(d 1/4 ) times the corresponding unconstrained clustering cost.
Case 1: If distribution D assigns x to c 1 with probability at least 1/2, then the expected ℓ 1 cost of the explainable clustering is at least d/2, while the optimal ℓ 1 clustering cost is d 3/4 (by assigning x to c 2 ).
Case 2: If distribution D assigns x to c 2 with probability at least 1/2, the expected ℓ 2 cost of the explainable clustering is at least d 3/4 /2, while the optimal ℓ 2 clustering cost is √ d (by assigning x to c 1 ).
this section cite: []

Section: D Lower bound for explainable k-medians under ℓ p norm
In this section, we present a lower bound on the competitive ratio for the explainable k-medians problem under ℓ p norm for all p ≥ 1. In particular, we extend the lower bound instance for explainable k-medians clustering under ℓ 2 norm in Makarychev and Shan (2021) to ℓ p norm for all p ≥ 1.
Theorem 4.1. For every p ≥ 1, there exists an instance X ⊆ R d , such that for every threshold tree T , its clustering cost is at least cost p (X, T ) = Ω(log k)OPT k,p (X), where OPT k,p (X) is the ℓ p cost of the optimal (unconstrained) k-medians clustering of X.
We construct the lower bound instance X as follows. Consider the grid G = {0, ϵ, 2ϵ, . . . , 1} d that is obtained by discretizing the hypercube, where d = ⌈64p 4 ln k⌉ and ϵ = 1/ ln k. We choose k centers C uniformly at random from the grid G and for each c ∈ C, we place two data points x c1 = c + (ϵ, ϵ, . . . , ϵ) and x c2 = c -(ϵ, ϵ, . . . , ϵ). Moreover, for every c ∈ C, we place n data points x cj , j = 3, 4, . . . , n + 2 that coincide with c (i.e. x cj = c). We will show that the clustering instance X = c∈C {x cj , j ∈ [n + 2]} satisfies with positive probability two properties captured by Lemma D.1 and Lemma D.2 and then show that these properties suffice to prove Theorem 4.1.
The first property we show is that the with high probability all centers in the random set C are well separated.
Lemma D.1. With probability at least 1 -1 k 2 , for any two distinct centers c, c ′ ∈ C, it holds that ∥c -c ′ ∥ p ≥ d 1 p 12 .
Proof of Lemma D.1. An equivalent way to choose a center from the grid {0, ϵ, 2ϵ, . . . , 1} d uniformly at random, is to first choose c ∈ [-ϵ 2 , 1 + ϵ 2 ] d uniformly at random and then choose c to be the closest center of c in the grid. Consider c, c ′ ∈ C be two distinct centers of the instance and let c and c′ be their corresponding uniform random variables in [-
ϵ 2 , 1 + ϵ 2 ] d . We have E[∥c -c′ ∥ p p ] = d i=1 E[|c i -c′ i | p ] = 2d(1 + ϵ) p (p + 1)(p + 2) , where we used that for each coordinate i, ci and c′ i are independent uniform random variables in [-ϵ 2 , 1 + ϵ 2 ]. Moreover, the variables |c i -c′ i | p are independent for different i and are bounded in [0, (1 + ϵ) p ]. By Hoeffding's inequality, we have Pr ∥c -c′ ∥ p p ≤ 2d(1 + ϵ) p (p + 1)(p + 2) -(1 + ϵ) p √ 2d ln k ≤ 1 k 4 . Because d ≥ 64 p 4 ln k, we get that (1 + ϵ) p √ 2d ln k ≤ d(1+ϵ) p (p+1)(p+2) , thus Pr ∥c -c′ ∥ p p ≤ d(1 + ϵ) p (p + 1)(p + 2) ≤ 1 k 4 . (6) This means that with probability at least 1 -1/k 4 , ∥c -c′ ∥ p ≥ (1 + ϵ)d 1 p (p + 1) 1 p (p + 2) 1 p . Because c is the closest point in the grid G to c, then ∥c -c∥ p ≤ ϵ 2 d 1 p (the same holds for c ′ and c′ ). Thus, by the triangle inequality ∥c -c ′ ∥ p ≥ (1 + ϵ)d 1 p (p + 1) 1 p (p + 2)
1 p -ϵd 1 p ≥ d 1 p12
.
The second inequality holds for sufficiently large k, since ϵ = 1/ ln k can be made arbitrarily small by increasing k, and because the function ((p + 1)(p + 2)) 1/p is decreasing for p ≥ 1 and thus attains its maximum value 6 at p = 1. By applying the union bound over all pairs of centers in C, the claim follows.
To describe the second property, we introduce some notation. Consider a threshold tree T and a node u of this tree. Let F u ⊆ C be the set of undamaged centers contained in u, i.e. the set of centers c in the node such that all the points in the optimal cluster of c are contained in the node u. We also define a path sequence as any sequence of tuples (i 1 , θ 1 , σ 1 ), (i 2 , θ 2 , σ 2 ), . . . (i t , θ t , σ t ), such that t ≥ 1 is an integer, i j ∈ [d], θ j ∈ R and σ j ∈ {±1}. Note that any node u is fully specified by the path from the root of T to u and thus by a path sequence π(u), where (i j , θ j ) is the j-th threshold cut in the path and σ j indicates the direction of the next node in the path. Inversely, for a given path sequence π we denote u(π) as the node that π specifies, i.e.
u(π) = (i,θ,σ)∈π {x ∈ R : σ(x i -θ) ≥ 0}.
Lemma D.2. With probability at least 1 -1 k , for every t ≤ log 2 k 4 , for every path sequence π = (i 1 , θ 1 , σ 1 ), . . . (i t , θ t , σ t ) with i j ∈ [d], θ j ∈ [0, 1], σ j ∈ {±1}, one of the following holds:
1. the number of undamaged centers in u(π) is at most |F u(π) | ≤ √ k; or 2. any cut that separates two centers in u(π) damages at least ϵ|F u(π) |/2 centers in F u(π) .
this section cite: ['b14']

Section: B.2 Efficient implementation and analysis
In this section, we present a practical implementation of dynamic algorithm as shown in Figure 2. We evaluate the efficiency of the algorithm from two perspectives: update time and recourse.
First, the update time at request q t refers to the time required to modify the threshold tree T t-1 in response to the t-th request q t (either an insertion or deletion of a center), resulting in a new tree T t . Second, the recourse at request q t is defined to be the number of nodes that differ between T t-1 and T t , i.e., the size of their symmetric difference between the two trees.
We focus on bounding these quantities in the amortized sense, i.e., the total update time and total recourse over all requests, averaged across the requests. The following lemma summarizes the performance guarantees of the dynamic algorithm. Lemma B.3. Given a sequence of requests, where each request is either an insertion or a deletion of a single center, the dynamic algorithm satisfies with probability 1 the following guarantees for every t ≥ 1 1. the amortized recourse is O(log k), 2. the amortized update time is O(d log 3 k), where k = max t i=1 |C i |.
We first describe an efficient implementation of the dynamic algorithm. For each node u where REBUILD is called, we maintain a self-balancing binary search tree that stores all cuts with timestamps (ω ′ 1 , ρ ′ 1 ), (ω ′ 2 , ρ ′ 2 ), . . . (ω ′ r , ρ ′ r ) used in the partial tree rooted at u. This data structure enables efficient updates. When a new request arrives to insert or delete a center c, we call GET_EARLIEST_CUT(c) to compute the earliest cut that separates c from the anchor m u , and then search the binary search tree to locate where this separation occurs in the partition leaf path of u.
We now describe an efficient implementation of the function GET_EARLIEST_CUT. Without loss of generality, we assume that all centers are in [-1, 1] d . The function GET_EARLIEST_CUT takes a center c as input and outputs the earliest cut (ω, ρ) that separates c from the anchor m u among a sequence of candidate cuts (ω 1 , ρ 1 ), (ω 2 , ρ 2 ), . . . . Each cut ω t = (i t , ϑ t ) is generated by sampling a coordinate i, a sign σ ∈ {-1, 1} uniformly at random, and a parameter θ ∈ [0, 2] drawn from the distribution with density f (x) = px p-1 /2 p . The threshold is then set as ϑ t = m u i + σθ. The associated timestamps ρ t follow the arrival times of a Poisson Process with rate λ = 1.
To facilitate efficient implementation, we first observe that the problem naturally decomposes across coordinates. Specifically, for each coordinate i ∈ {1, 2, • • • , d}, we can independently maintain and query the earliest cut that separates c from m u along coordinate i. We then return the cut with the minimum timestamp across all coordinates.
To achieve this, we maintain an independent stream of candidate cuts for each pair of coordinate i and direction σ ∈ {-1, 1}. Each such stream consists of cuts ω = (i, ϑ) where ϑ = m u i + σθ and the timestamps given by the arrival times of a Poisson process with rate 1 /2d. This decomposition is formally justified by the Coloring Theorem (see, e.g. Kingman (1992), page 53 or Mitzenmacher and Upfal (2017), page 223), which states: Theorem B.4 (Coloring Theorem). Let Π t be a Poisson process on the real line with rate λ. Assign to each event of the process a color from a finite set {1, • • • , M }, where each event is independently colored with probability p i of receiving color i. Then the counts of events of each color, Π 1 , • • • , Π M , form independent Poisson processes, with rates λp 1 , • • • , λp M , respectively.
The original sequence of candidate cuts has timestamps given by the arrival times of a Poisson process with rate 1. Each cut is independently assigned a pair (i, σ) with uniform probability 1 /2d over all 2d possible combinations. By the Coloring Theorem, the subset of cuts corresponding to any fixed pair (i, σ) forms an independent Poisson process with rate 1 /2d and these 2d streams are independent. Therefore, the union of all these subsequences of cuts has the same distribution as the original sequence of candidate cuts.
We then formulate the earliest cut along each coordinate as the following general problem. We are given a fixed anchor value m ∈ [-1, 1], and a sequence of random cuts specified by thresholds ϑ t drawn from [m, m + 2] according to a probability density function f (x), with associated timestamps ρ t , corresponding to the arrival times of a Poisson process with rate λ 0 . For a query point y ∈ [m, 1], we aim to find the earliest cut that separates y and m, i.e., the cut with the smallest timestamp such that its threshold ϑ t lies in (m, y]. This formulation arises naturally in our setting, where λ 0 = 1 /2d, the density function f (x) = p(x -m) p-1 /2 p , m represents the i-th coordinate of the anchor, and y corresponds to the i-th coordinate of some center c. A simple approach for solving this problem is to simulate the sequence of cuts with timestamps and return the first one that lies in (m, y]. We refer to this as the static algorithm.
We now describe a data structure that efficiently retrieves the earliest cut along a given coordinate. This data structure maintains a self-balancing binary search tree. Given an anchor m and a set of k values m ≤ y 1 < y 2 < • • • < y k ≤ 1, this binary search tree maintains these values in increasing order. Each node in the binary search tree stores a value y along with the earliest cut that separates y from the anchor m, including the timestamp of that cut. If the queried value y is present in the tree, the associated earliest separating cut can be retrieved in O(log k) time. Now suppose we need to insert a new value y ∈ [m, 1] into this data structure. Assume the binary search tree currently stores k values m ≤ y 1 < y 2 < • • • < y k ≤ 1. We first locate the position of y in the tree in O(log k) time, either identifying the smallest index j such that y < y j , or determining that y > y k . Let y 0 = m. If there exists some 1 ≤ j ≤ k such that y j-1 < y < y j , then we first retrieve the earliest cut (ϑ, ρ) that separates y j from m. We consider two different cases:
1. y j-1 < y < y j for some 1 ≤ j ≤ k and (ϑ, ρ) also separates y from m, (i.e. ϑ ≤ y); 2. either y j-1 < y < y j and (ϑ, ρ) does not separates y from m (i.e. ϑ > y) or y k < y ≤ 1.
For the first case, we store this cut (ϑ, ρ) at the node y as the earliest cut that separates y from m.
For the second case, we first sample a new cut as follows. If y ≥ y k , then let y j-1 = y k . Sample a new threshold ϑ ′ ∈ (y j-1 , y] using the weighted density function
f (x) = f (x) Pr{ϑ ∈ (y j-1 , y]} = f (x) y yj-1 f (t)dt , x ∈ (y j-1 , y].
We then sample a timestamp for this cut as ρ ′ = ρ + z, if y ≤ y k , otherwise if y > y k , ρ ′ = z, where z ∼ exp(λ) with rate
λ = λ 0 • Pr{ϑ ∈ (y j-1 , y]} = λ 0 • y yj-1 f (t)dt,
where λ 0 is a parameter of the data structure. Let (ϑ ′′ , ρ ′′ ) be the earliest cut that separates y j-1 from m. We then compare the two cuts and store at node y the one with the smaller timestamp. If ρ ′ < ρ ′′ , then we store the new cut (ϑ ′ , ρ ′ ) at node y as the earliest cut; otherwise, we store the cut (ϑ ′′ , ρ ′′ ). Lemma B.5. Given a sequence of query points y 1 , y 2 , • • • , the earliest cuts maintained by the data structure are distributed identically to those returned by the static algorithm.
Proof. We prove this lemma by induction. For the first query point, the data structure and the static algorithm samples the earliest cut that separates this point from the same distribution. We now assume that for the first k query points y 1 , • • • , y k , the earliest cuts returned by the data structure are distributed identically to those returned by the static algorithm. By coupling these two algorithms, we further assume that the data structure and the static algorithm return exactly the same earliest cuts for these query points.
We now consider a new query point y k+1 and argue that the earliest cuts returned by two algorithms are distributed identically. Let y (1) , y (2) , • • • , y (k) be the first k query points sorted in increasing order. Let y (0) = m. Suppose this new query point is in the first case, which means there exists 1 ≤ j ≤ k such that y (j-1) < y k+1 < y (j) and the earliest cut (ϑ t , ρ t ) that separates y (j) maintained by the data structure also separates y k+1 . Since y k+1 < y (j) and this cut (ϑ t , ρ t ) is the earliest cut that separates y (j) in the static algorithm, this cut is also the earliest cut for y k+1 returned by the static algorithm.
We now consider this new query point is in the second case, either y (j-1) < y k+1 < y (j) and the earliest cut (ϑ t , ρ t ) that separates y (j) does not separates y k+1 or y (k) < y k+1 ≤ 1. If y k+1 > y (k) , we set y (j-1) = y (k) . We decompose the sequence of cuts used in the static algorithm into three disjoint subsequences. These three subsequences contain all cuts in three disjoint intervals (m, y (j-1) ], (y (j-1) , y k+1 ], and (y k+1 , m + 2] respectively. By the Coloring Theorem, the timestamps of these subsequences follow the arrival times of three independent Poisson processes. Since the cut is sampled from (y (j-1) , y k+1 ] with probability p = y k+1 y (j-1) f (t)dt, the timestamps of all cuts in (y (j-1) , y k+1 ] follows the arrival times of a Poisson process with rate
λ = λ 0 • Pr{ϑ ∈ (y (j-1) , y k+1 ]} = λ 0 • y k+1 y (j-1) f (t)dt.
Suppose there exists 1 ≤ j ≤ k such that y (j-1) < y k+1 < y (j) . Since the earliest cut (ϑ t , ρ t ) that separates y (j) does not separate y k+1 in the static algorithm, the first cut in the interval (y (j-1) , y k+1 ] must arrive after ρ t . The time of the first arrival of this subsequence follows an exponential distribution with rate λ. Due to the memoryless property of the exponential distribution, the first arrival of cuts in (y (j-1) , y k+1 ] follows ρ t + z, where z ∼ exp(λ). Suppose y k+1 > y (k) . Then, the time of the first arrival in this subsequence is z ∼ exp(λ). Therefore, in the static algorithm, the first cut in (y (j-1) , y k+1 ] has the exact same distribution as the new cut sampled in the data structure. If y (j-1) ̸ = m, then the first cut in (m, y (j-1) ] is the same in the data structure and the static algorithm. Combining two parts, the earliest cut that separates y k+1 returned by the data structure has the same distribution as that returned by the static algorithm.
this section cite: ['b11']

Section: Remark.
The assumption that all centers lie in [-1, 1] d is made for the ease of exposition. The algorithm can be implemented without this assumption. Under the ℓ p norm, the threshold θ is drawn from a distribution with density f (x) = px p-1 /R p where R > y is the bounding radius. Conditioned on x ∈ (y j-1 , y], the probability density function becomes
f (x) = f (x) y yj-1 f (t)dt = p(x -m) p-1 (y -m) p -(y j-1 -m) p .
To sample a threshold ϑ ′ following this distribution, we draw a uniform random variable U ∈ [(y j-1 -m) p , (y -m) p ] and set ϑ ′ = U 1/p . Moreover, multiplying all timestamps by the same positive number does not affect the analysis of B.5. Thus, we can equivalently sample z ∼ exp(λ) with λ = (y -m) p -(y j-1 -m) p , without altering the analysis. With these minor modifications, the algorithm no longer depends on the boundedness assumption that the centers lie in [-1, 1] d .
We now analyze the recourse and the update time of the dynamic algorithm with the above implementation.
Proof of Lemma B.3. Fix t ≥ 1 and condition on the randomness of the algorithm until time t. Since the subsequent argument holds for any fixed randomness, the guarantees hold with probability 1.
Recourse: Let R(i) be the recourse incurred by request i. We partition the requests into two sets: Let S 1 ⊆ [t] be the set of requests for which the REBUILD operation is not called during the update due to this request. Let S 2 = [t] \ S 1 be the remaining requests where the REBUILD operation is called. We analyze each case separately.
Case 1 (i ∈ S 1 ): In this case, the request does not trigger a REBUILD operation, and the recourse is at most R(i) ≤ 2. This is because if the request is an insertion, at most two nodes are added to T i-1 ; if it is a deletion, at most two nodes are removed, i.e., the leaf that contains the center c and its parent in both cases. As a result, the total recourse over all such requests is bounded by
i∈S1 R(i) ≤ 2|S 1 | ≤ 2t.(2)
Case 2 (i ∈ S 2 ): The REBUILD will only be called on one node u i for each request i. Let C ui be the set of centers contained in the node u i of T i-1 , and let
k ′ = |C ui |. Since REBUILD(u i ) is called, all 2k ′ -1 nodes in the subtree rooted at u i are removed from T i-1 .
If the request i is an insertion of a center c, a new threshold tree is constructed at u i using the updated center set C ui ∪ {c}, which has size k ′ + 1. This results in inserting 2(k ′ + 1) -1 = 2k ′ + 1 nodes back into the tree. Therefore, the recourse is R(i) = 2k ′ -1 + 2k ′ + 1 = 4k ′ . If the request i is a deletion of a center c, the updated center set is C ui \ c of size k ′ -1, and the rebuilt threshold tree contains 2(k ′ -1) -1 = 2k ′ -3
Proof of Lemma D.2. It suffices to prove the lemma for path sequences such that θ j ∈ { ϵ 2 , 3ϵ 2 , . . . , 1ϵ 2 }. This restriction is without loss of generality, since for every coordinate i ∈ [d] and for every r ∈ {0, 1, . . . , 1 ϵ }, all the cuts in the interval (rϵ, (r + 1)ϵ] are equivalent, in the sense that they induce the same partition of the grid points and thus of the instance X.
Fix any path sequence π of size t ≤ log 2 k 4 and denote u = u(π) for simplicity. Assume that the total number of undamaged centers in u is
|F u | = k ′ > √ k.
Given a threshold cut ω = (i, θ), we define Z ω to be the number of undamaged centers c ∈ F u that are damaged by ω. Conditioned on |F u | = k ′ , the undamaged centers contained in u are distributed as k ′ points drawn independently and uniformly from the grid points G inside u, excluding the leftmost and rightmost grid points in each coordinate. Consider each undamaged center c ∈ F (u). The new cut ω damages this center c if and only if c i ∈ {θ -ϵ/2, θ + ϵ/2}. Since there are at most 1/ϵ possible grid positions for c i , this undamaged center c is damaged by the cut ω with probability at least ϵ. Therefore, we have
E[Z ω | |F u | = k ′ ] ≥ ϵk ′
, where the expectation is taken over the randomness of centers in F (u). Thus, by the Chernoff bound
Pr Z ω ≤ ϵ 2 k ′ |F u | = k ′ ≤ e -ϵk ′ 8 ≤ e -ϵ √ k 8
.
By taking the union bound over all possible cuts in u (at most d/ϵ = O(p 4 ln 2 k) in total), we obtain some cut damages less than ϵk ′ /2 undamaged centers in F (u) with probability at most e -ϵ √ k 16 for sufficiently large k. Thus, the probability that both (1) and (2) do not hold is at most e -ϵ √ k 16 . Moreover, the number of different path sequences at a fixed size t is at most 2d ϵ t = O(p 4t ln 2t k).
Thus, by taking the union bound over all possible path sequences for every t ≤ log 2 k 4 , the probability that both (a) and (b) do not hold is at most
log 2 k 4 2d ϵ log 2 k 4 e -ϵ √ k 16 = e O(log(p 2 log k) log k) e -ϵ √ k 16 ≤ 1 k ,
where the inequality holds for any fixed p when k is sufficiently large.
By Lemma D.1 and D.2 there exists an instance X with k centers and d = ⌈64p 4 ln k⌉ such that both properties of these lemmas hold. Moreover, the optimal clustering has ℓ p cost OPT k,p ≤ 2kϵd 1 p , as we can assign each data point x cj to center c. Consider any threshold tree T with k leaves. We will show that cost p (X, T ) = Ω(log k)OPT k,p .
First, we consider the case where T does not separate all centers in C, that is, there exists a leaf of the tree that contains two centers c and c ′ . Note that there are n data points located at each of the centers c and c ′ . Hence, the cost of this leaf is at least n∥c -c ′ ∥ p /2 ≥ nd 1 p /24 by Lemma D.1. This cost can be arbitrarily large since n can be arbitrarily large.
Next, consider the threshold tree T in which each leaf contains exactly one center from C. We divide it into the following two cases. In the first case, suppose there exists a level 1 ≤ t ≤ log 2 k 4 that contains at least k 2 damaged centers. For each damaged center, there is a data point that was assigned to it in the optimal solution but is reassigned to another center by T . Each such reassignment incurs a cost of Ω(d 1/p ). Thus, the total cost of T is at least Ω( k 2 d 1/p ) = Ω(log k)OPT k,p since ϵ = 1/ ln k. In the second case, assume that for every 1 ≤ t ≤ log 2 k 4 , the number of undamaged centers at level t of T is at most k 2 . We call a node u small if it contains at most √ k undamaged centers, and large otherwise. Fix any t in {1, 2, . . . log 2 k 4 }. Since the total number of nodes at level t is at most k 1 4 , the small nodes together contain at most k 3 4 undamaged centers. Hence, the large nodes contain at least k 2 -k 3 4 ≥ k 4 undamaged centers for sufficiently large k. Because T contains exactly one center from C, all thresholds of cuts lie within [0, 1]. By Lemma D.2, the number of undamaged centers that become damaged at level t of T is at least ϵk 4 . Since each damaged center incurs a reassignment cost of Ω(d 1/p ) by Lemma D.1, the total cost at level t is Ω(ϵkd 1/p ). By summing over all levels 1 ≤ t ≤ log 2 k 4 , the total cost is Ω log k • ϵkd 1 p = Ω(log k)OPT k,p .
this section cite: []

Section: References
Ref_id:b0 Title: How to find a good explanation for clustering? Year: (2022)
Ref_id:b1 Title: Fully dynamic k-clustering in o(k) update time Year: (2023)
Ref_id:b2 Title: Fully dynamic k-clustering with fast update time and small recourse Year: (2024)
Ref_id:b3 Title: Fully dynamic k-median with nearoptimal update time and recourse Year: (2025)
Ref_id:b4 Title: Fully dynamic k-center clustering Year: (2018)
Ref_id:b5 Title: Fully dynamic consistent facility location Year: (2019)
Ref_id:b6 Title: Explainable k-means and k-medians clustering Year: (2020)
Ref_id:b7 Title: Approximation algorithms for clustering with dynamic points Year: (2022)
Ref_id:b8 Title: Almost tight approximation algorithms for explainable clustering Year: (2022)
Ref_id:b9 Title: Nearly-tight and oblivious algorithms for explainable clustering Year: (2021)
Ref_id:b10 Title: The price of explainability for clustering Year: (2023)
Ref_id:b11 Title:  Year: (1992)
Ref_id:b12 Title: On the price of explainability for some clustering problems Year: (2021)
Ref_id:b13 Title: Consistent k-clustering Year: (2017)
Ref_id:b14 Title: Near-optimal algorithms for explainable k-medians and k-means Year: (2021)
Ref_id:b15 Title: Explainable k-means: don't be greedy, plant bigger trees Year: (2022)
Ref_id:b16 Title: Random cuts are optimal for explainable k-medians Year: (2023)
Ref_id:b17 Title: Probability and computing: Randomization and probabilistic techniques in algorithms and data analysis Year: (2017)
