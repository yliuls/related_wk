Title: On Differential Privacy for Adaptively Solving Search Problems via Sketching
Abstract: Recently differential privacy has been used for a number of streaming, data structure, and dynamic graph problems as a means of hiding the internal randomness of the data structure, so that multiple possibly adaptive queries can be made without sacrificing the correctness of the responses. Although these works use differential privacy to show that for some problems it is possible to tolerate T queries using O( √ T ) copies of a data structure, such results only apply to numerical estimation problems, and only return the cost of an optimization problem rather than the solution itself. In this paper, we investigate the use of differential privacy for adaptive queries to search problems, which are significantly more challenging since the responses to queries can reveal much more about the internal randomness than a single numerical query. We focus on two classical search problems: nearest neighbor queries and regression with arbitrary turnstile updates. We identify key parameters to these problems, such as the number of c-approximate near neighbors and the matrix condition number, and use different differential privacy techniques to design algorithms returning the solution vector with memory and time depending on these parameters. We give algorithms for each of these problems that achieve similar tradeoffs.

Section: Introduction
In modern algorithmic design, data structures are playing a more and more important role, in particular in the recent breakthroughs in optimization, machine learning and graph problems. These highly efficient data structures are usually randomized, with one crucial distinction from traditional data structures: one needs these data structures to be robust against an adaptive adversary, in which the adversary could design the input to the data structure based on the prior outputs of the data structure. This is often because these data structures are incorporated in an iterative process, where the data structure takes an input formed by the process, returns a result, and the process utilizes the output to form a new input. Hence, the input to a data structure may be highly correlated with the internal randomness of the data structure. Most traditional randomized data structures, unfortunately, are not robust enough to handle this type of task, as they only have probabilistic guarantees against oblivious adversariesfoot_0 , in which the input sequence to the data structure is predetermined, and independent of its internal randomness.
To address this challenge, various reductions have been proposed to build an adaptive data structure from oblivious data structures (Ben-Eliezer et al., 2020;Cherapanamjeri & Nelson, 2020;2022;Beimel et al., 2022;Hassidim et al., 2022;Brand et al., 2022;Song et al., 2023b;Bateni et al., 2024). These reductions usually proceed as follows: (1) create k independent copies of the oblivious data structure, (2) during the query phase sample and query a subset of the data structures, and (3) aggregate the subsampled answers.
We focus on the following search data structure problem: we are given n points in R d represented by a matrix U ∈ R n×d that is allowed to be preprocessed. Given a sequence of adaptive queries or updates {v 1 , . . . , v T }, we need to design a data structure that answers queries efficiently and succeeds with high probability. Throughout the paper we will consider two key examples of this: (1) the first type of problem is the approximate near-neighbor problem, where a query is a c-approximate near-neighbor of v t in U , and (2) the second type of problem is the regression problem, where U is in addition augmented by a response vector b ∈ R n , it could be the solution to the regression problem arg min x∈R d ∥U x -b∥ 2 2 where either U or b can also be updated by v t . Note that one can interpret regression as an unconstrained search problem over R d . the simplest reduction is to prepare T copies of an oblivious data structure and for each query, use a fresh new data structure to respond. Since a data structure is never used more than once, an adaptive adversary cannot design adversarial inputs to this data structure, and hence we obtain an adaptive data structure. In fact, this simple approach already has led to many improvements in convex optimization (Lee et al., 2019;Song & Yu, 2021;Jiang et al., 2021;Qin et al., 2023). However, in many applications, the overhead T in both preprocessing time and space usage is prohibitive, and more time-and space-efficient alternatives are needed. If in addition the adaptive queries are vectors in R d and each data structure succeeds with constant probability, then one can adapt an ε-net argument and show that O(d) 2 oblivious data structures suffice; at query time, one can then sample a logarithmic number of these data structures and output the the best answer. This approach has been widely adapted for problems such as distance estimation (Cherapanamjeri & Nelson, 2020;2022) and approximate nearneighbor search (Song et al., 2022;Bateni et al., 2024), but is insufficient if d ≈ T or even d ≫ T , which is the case for convex optimization. Moreover, for many machine learning applications, T is often a hyperparameter that could be chosen and is much smaller than d, making the approach based on ε-net arguments inefficient for these applications.
Is it possible to improve upon the solutions that use min{d, T } data structures? In pioneering work of (Hassidim et al., 2022) and further extended by (Beimel et al., 2022), it was shown that differential privacy could be utilized to reduce such overhead. In particular, for numerical estimation problems, where one only needs to output a numerical value, these works show that O( √ T ) data structures are sufficient. The idea is to view the internal randomness of the oblivious data structures as the private database one would like to protect. Concretely, one prepares O( √ T ) oblivious data structures, and for each adaptive query, samples O(1) of the data structures, and outputs the differentially private median of the answers (Beimel et al., 2022). By the advanced composition theorem (Dwork et al., 2010), O( √ T ) data structures suffice to provide a correct answer to the numerical estimation problem. This leads to drastically improved algorithms when √ T ≪ d, and in (Beimel et al., 2022) they apply it to a large number of graph algorithms. This technique was later extended in (Cherapanamjeri et al., 2023) for estimating the cost of a regression problem and the value of kernel density estimation. Can these techniques be extended beyond numerical estimation, to search problems? This is a natural question, as instead of the regression cost, one could naturally be more interested in estimating the regression solution and using it 2 We use O(•) to suppress polylogarithmic factors in n, d and 1/δ where δ is the failure probability of the data structure.
to, e.g., label future examples. On the other hand, simply extending the private median framework to high dimensions seems insufficient, as the regression solution could potentially reveal much more about the internal randomness of a data structure than what only the cost would reveal. This problem becomes more evident for approximate near neighbor search: for this problem, if there exists a row u * of U for which ∥u * -v∥ ≤ r for an adaptively chosen query v ∈ R d and a distance parameter r, then the data structure needs to output a row u of U with ∥ u -v∥ ≤ cr for c > 1. It is not clear how to extend the differential privacy framework to such a scenario, as we will have to return a row of U , while the private median mechanism would not be able to do this. Motivated by these problems and barriers, we ask Is it possible to design robust search data structures with fewer than min{d, T } independent copies?
We provide an affirmative answer to this question under mild assumptions on these problems. We start by introducing the definition of the (c, r)-Approximate Near Neighbor ((c, r)-ANN) problem and its corresponding assumption.
Definition 1.1 ((c, r)-Approximate Near Neighbor). Let U ⊆ R d be a dataset, v ∈ R d be a query point, c > 1 be the approximation parameter, and r > 0 be the distance parameter. The (c, r)-approximate near neighbor problem asks, if there exists a point u * ∈ U with ∥u * -v∥ ≤ r, then return a point u ∈ U with ∥ u -v∥ ≤ cr. Otherwise, the data structure can either return nothing or any point u ∈ U with ∥ u -v∥ ≤ cr. 3We now state the assumption imposed on the ANN problem, in its most general form. Given a query v and a data matrix U , we will use f v (U ) to denote the set of candidate solution of querying v, For example, the predicate function for (c, r)-ANN can be defined as f v (U ) = U ∩ B(v, cr) where B(v, cr) is the ball centered at v with radius cr under some norm. The assumption is: Assumption 1.2. Let U ⊆ R d be an n-point dataset, and let {v 1 , v 2 , . . . , v T } ⊆ R d be a sequence of (possibly adaptive) queries. We assume for all t ∈ [T ], |f vt (U )| ≤ s.
The assumption states that the ball around v t of radius cr cannot intersect with more than s points in U , which is equivalent to that the query v t cannot have too many approximate near neighbors in U , when v t has an r-near neighbor.
To facilitate a comparison with other popular assumptions for this task, such as constant expansion and bounded doubling dimension, we state a version of the assumption for ANN that solely depends on the matrix U :
Assumption 1.3. Let U ⊆ R d , and let {B u } u∈U be the collection of norm balls with B u = B(u, cr). We assume for each u ∈ U , that B u intersects at most s other distinct balls in the collection.
Let us first see how this assumption implies the condition on f v (U ). By the triangle inequality, any two points in B(v, cr) have distance at most 2cr, that is, the norm balls with these two points being the respective centers with radius cr must intersect. If v has more than s approximate nearest neighbors, then there must exist some u ∈ B(v, cr) whose norm ball intersects with more than s distinct balls, a contradiction. Intuitively, Assumption 1.3 states that the dataset U is not too dense, and in particular each point in U does not have too many close neighbors. This structural assumption can also be achieved by preprocessing U : one could run a clustering algorithm on U to group points that are close to each other into a cluster, and replace a cluster using its center. Then, the downstream ANN algorithm is performed on these centers; once a center is returned, one could return a point in the corresponding cluster. This approach has been implemented in various approximate nearest neighbor search libraries in practice, and has been a driving force for the Google ScaNN framework (Sun, 2020;Guo et al., 2020).
For efficiency reasons, we will restrict s ≤ n ρ .
We also compare Assumption 1.3 to two popular assumptions for nearest neighbor search: constant expansion (Karger & Ruhl, 2002;Beygelzimer et al., 2006) and doubling dimension (Gupta et al., 2003;Krauthgamer & Lee, 2004a;b). The constant (local) expansion states that if
|B(v, r) ∩ U | ≥ log n, then |B(v, 2r) ∩ U | ≤ c exp • |B(v, r) ∩ U |
for a constant c exp . The query time of nearest neighbor search data structures under the constant expansion assumption usually has a polylogarithmic dependence on n but a large polynomial dependence on c exp . On the other hand, we only need |B(
v, cr) ∩ U | ≤ n ρ . If we let |B(v, r) ∩ U | = log n, then one can see that |B(v, n ρ r) ∩ U | ≤ n ρ .
Thus, for any c < n ρ (which is a very large approximation factor), our assumption is strictly weaker. Moreover, we note that constant expansion restricts the growth across different distances, while our assumption only requires a relatively sparse neighborhood at the final level. A more robust version of constant expansion is the notion of doubling dimension, which asks how many balls of radius r are needed to cover a ball of radius 2r. It has been shown in (Krauthgamer & Lee, 2004b) that bounded doubling dimension is a strictly stronger assumption than constant expansion, and it provides a bound of the form X) , where ∆ ≥ 2 is the aspect ratio of U and X is the metric space. If X = R d with any norm, then dim(X) = Θ(d), and therefore for this bound to be meaningful, one must have d = o(log n). Moreover, data structures with a doubling dimension assumption often have their preprocessing time and space exponential in dim(X).
|B(v, cr) ∩ U | ≤ ∆ dim(
As a final note, we would like to point out that when working with U ⊆ {0, 1} d and when the norm is ∥ • ∥ 1 , i.e., the Hamming ANN problem, Assumption 1.3 automatically holds for d = n α and α ≤ ρ cr . In particular for Hamming LSH, ρ = O(1/c) (Indyk & Motwani, 1998) and thus we only need d ≤ n Θ(1/(c 2 r)) . Our result will provide a Hamming LSH that is robust against adaptive adversaries, and in particular against the attack of (Kapralov et al., 2024) in low dimensions.
Next, we consider the problem of adaptively updating the regression problem, and outputting the regression solution vector whenever queried. Assumption 1.4. Let U ∈ R n×d be the design matrix and b ∈ R n be the response vector. Let {v 1 , . . . , v T } be a sequence of (possibly) adaptive updates to the problem in one of two forms: (1
) Update U : v t ∈ R n×d and U is updated via U ← U + v t ; (2) Update b: v t ∈ R n and b is updated via b ← b + v t .
We will use (U t , b t ) to denote the pair after being updated by v t . The goal is to design a data structure such that for any t ∈
[T ], it outputs a (1 + α)- approximate solution x t ∈ R d for which ∥U t x t -b t ∥ 2 2 ≤ (1 + α) • min x∈R d ∥U t x -b t ∥ 2
2 holds with high probability. For all t ∈ [T ], we assume the condition number of U t defined as κ(U t ) := σmax(Ut) σmin(Ut) , is upper bounded by κ.
The problem and assumption can be succinctly described as follows: an adversary could adaptively perturb entries of U and b, and our goal is to design a data structure that outputs the regression solution in the presence of these perturbations.
In addition, we assume the perturbations are bounded, in the sense that they do not change the conditioning of the design matrix by too much. For this problem, note that one could naïvely store the matrix U and solve the regression problem exactly. In this case, as the algorithm is deterministic, it is automatically adaptive. However, this approach would require Ω(nd) space and O(nd ω-1 ) time, which is both spaceand time-inefficient for large n. Alternatively, one could prepare T independent sketching matrices S 1 , . . . , S T , one for each query, and store S t U, S t b for all t ∈ [T ]. When receiving an update to U or b, one can simply update the corresponding entries of S t U and S t b, as sketching matrices are linear. This approach requires Ω(T • poly(d)) space, a preprocessing time of O(T • (nd + poly(d))), and a query time of poly(d). While the query time is much more efficient for n ≫ d, the space and preprocessing time are prohibitive given that T is large. Hence, our goal is to design algorithms that use o(T • poly(d)) space, have a sublinear in T dependence in the preprocessing time, and have a query time of poly(d).
this section cite: ['b10', 'b19', 'b92', 'b9', 'b13', 'b8', 'b66', 'b54', 'b77', 'b19', 'b92', 'b80', 'b8', 'b9', 'b9', 'b37', 'b9', 'b22', 'b86', 'b45', 'b58', 'b11', 'b46', 'b52', 'b57']

Section: Main Results
We state the main results for ANN and adaptive regression in this section.
this section cite: []

Section: ADAPTIVE ANN FOR SPARSE NEIGHBORHOODS
Our most general result for ANN is as follows: Theorem 1.5. Let U ⊆ R d be an n-point dataset and let
f v : (R d ) n → (R d
) s be the predicate function. Let A be an oblivious algorithm, i.e., for a fixed query v ∈ R d , with probability at least 1 -δ, A returns a point (or a subset) of f v (U ) if f v (U ) is non-empty. Moreover, A has preprocessing time T prep , space usage S space and query time T query . Then there exists an adaptive algorithm A such that given a sequence of adaptive queries {v 1 , . . . , v T } satisfying Assumption 1.2: (1) Preprocesses U in time O( √ T • s) • T prep ; (2) Uses space O( √ T • s) • S space ; (3) For all t ∈ [T ], given query v t , it returns a point (or a subset) of f vt (U ) in time O(s)•T query with probability at least 1-δ. In particular, the amortized cost per query is O(s/ √ T )•T prep + O(s)•T query .
Let us interpret Theorem 1.5. It states that as long as s = o( √ T ), then we can turn an oblivious data structure into an adaptive data structure using fewer than T independent copies of the oblivious data structure. Of course, this comes at a cost of a slightly worse query time by a factor of O(s). However, if we consider the amortized cost per query for using T data structures, the cost is dominated by T prep , as the algorithm needs to prepare a fresh data structure for each query. In contrast, Theorem 1.5 has an amortized cost of O(s/ √ T )•T prep + O(s)•T query per query. Typically, T query is much smaller than T prep ; for example, for LSH, T prep = n 1+ρ d and T query = n ρ d, so it is much more important to obtain a reduction on the number of data structures. Applying Theorem 1.5, we immediately obtain adaptive algorithms for LSH under different norms, using fewer than T data structures. Theorem 1.6. Let U ⊆ R d be an n-point dataset satisfying Assumption 1.3. There exists an adaptive algorithm A such that given a sequence of adaptive queries {v 1 , . . . , v T }, it solves the (c, r)-ANN problem (Definition 1.1) and ( 1
) Preprocesses U in time O( √ T • s • n 1+ρ d); (2) Uses space O( √ T • s • n 1+ρ + nd); (3) For all t ∈ [T ], given query v t , it returns a point in B(v, cr) ∩ U if B(v, r) ∩ U ̸ = ∅ in time O(s • n ρ d), with probability at least 1 -δ.
A generic LSH data structure template restricts the number of points it looks at per query to n ρ , so if s ≤ n ρ , we obtain the following results: Corollary 1.7. Let U ⊆ R d satisfy Assumption 1.3 with s ≤ n ρ . Then there exists an adaptive algorithm A such that given a sequence of adaptive queries {v 1 , . . . , v T }, the data structure
(1) Preprocesses U in time O( √ T • n 1+O(ρ) d); (2) Uses space O( √ T •n 1+O(ρ) +nd); (3) For all t ∈ [T ], given query v t , it returns a point in B(v, cr)∩U if B(v, r)∩U ̸ = ∅ in time O(n O(ρ) d), with probability at least 1 -δ.
Utilizing these adaptive data structures, we obtain improved runtime for problems such as online weighted matching with adversarial arrival, and terminal embeddings. We refer the reader to Section 4 for more details. In addition, when these data structures need to be updated (insert or delete points from the data structures), we provide procedures based on fast rectangular matrix multiplication that beat the alternative of simply updating all data structures (Section C). In Table 1, we compare our result with prior works that use d or T copies of the LSH's.
this section cite: []

Section: ADAPTIVE REGRESSION FOR WELL-CONDITIONED INSTANCES
For adaptive regression, we can use fewer than T copies of an oblivious data structures if the condition number upper bound κ is small: Theorem 1.8. Let U ∈ R n×d , b ∈ R n and {v 1 , . . . , v T } be a sequence of adaptive updates to (U, b), satisfying Assumption 1.4. Then, there exists an adaptive algorithm that (1)
Preprocesses (U, b) in time O( √ T d • (nnz(U ) + nnz(b) + d 3 + d 2 κ 2 /α 2 )); (2) Uses space O( √ T • d 2.5 κ 2 /α 2 ); (3) For all t ∈ [T ], it updates (U t-1 , b t-1 ) 4 to (U t , b t ) in time O( √ T d • (nnz(v t ) + d 3 + d 2 κ 2 /α 2 ));
(4) For all t ∈ [T ], given update v t , it returns a solution x t that is a (1 + α)approximation to the regression problem using (U t , b t ) in time O(d ω+1 κ 2 /α 2 ), with probability at least 1 -δ.
Theorem 1.8 offers an algorithm with extremely efficient query time. When the condition number bound κ is small, it provides a space bound of √ T • poly(d), both sublinear in T and with no polynomial dependence on n. The preprocessing time outperforms the simple algorithm which generates T sketches when d ≪ T and κ is small. While for ANN, one could prepare O(d) data structures during preprocessing to prepare for all possible queries, and at query time just sample a small number of these, this is not possible for regression. Indeed, for regression one would need to prepare O(nd) such data structures, given the number of possible design matrices, which would be prohibitive. We again compare our result with prior approaches that use nd or T copies of sketches, in Table 2.
We next study the model in (Cherapanamjeri et al., 2023), where only the response vector b is allowed to be updated in at most s positions. In (Cherapanamjeri et al., 2023), it is shown that one can corporate the techniques of (Beimel et al., 2022) to output the cost of the regression problems, but a key open question was whether one could output the actual solution vector to the regression problems. We resolve this question by utilizing the tools we developed in Theorem 1.8 in conjunction with a preconditioner for U to remove the dependence on κ. Below, we will use nnz(U ) Method Space Amortized Prep Time Query Time Update Time T copies
T n 1+ρ + nd n 1+ρ d n ρ d T n ρ d d copies n 1+ρ d d T n 1+ρ d n ρ d n ρ d 2 This paper √ T sn 1+ρ + nd s √ T n 1+ρ d sn ρ d √ T sn ρ d
Table 1: Specifications of data structures given a sequence of T adaptive queries and updates for ANN problem. We ignore O(•) notation for clarity. We use s to denote the parameter for Assumption 1.3.
this section cite: ['b22', 'b22', 'b9']

Section: Method Space Amortized Prep Time
Query Time Update Time T copies
T d 2 /α 2 nnz(U, b) + d 3 + d 2 /α 2 d ω+1 /α 2 T (nnz(v t ) + d 3 + d 2 /α 2 ) nd copies nd 3 /α 2 nd T (nnz(U, b) + d 3 + d 2 /α 2 ) d ω+1 /α 2 nd(nnz(v t ) + d 3 + d 2 /α 2 ) This paper √ T d 2.5 κ 2 /α 2 d T (nnz(U, b) + d 3 + d 2 κ 2 /α 2 ) d ω+1 κ 2 /α 2 √ T d(nnz(v t ) + d 3 + d 2 κ 2 /α 2 )
Table 2: Specifications of data structures given a sequence of T adaptive queries and updates for regression. We ignore O(•) notation for clarity. We use nnz(U, b) as a shorthand for nnz(U ) + nnz(b), and let κ be the parameter for Assumption 1.4.
to denote the number of nonzero entries in U .
Theorem 1.9. Let U ∈ R n×d and b ∈ R n . Let {v 1 , v 2 , . . . , } ⊂ R n be a sequence of adaptive updates to b, such that for all t, ∥v t ∥ 0 ≤ s. Let T be a batch size parameter. There exists an adaptive algorithm that (1) It has amortized update time
O( d/T • (nnz(U ) + nnz(b) + d 3 + d ω /α 2 ) + √ T d • (s + d 3 + d 2 /α 2 )); (2) It outputs a (1 + α)-approximate solution x t in time O(d 2 ).
The main advantage of Theorem 1.9 over Theorem 1.8 is that the quadratic dependence on the condition number can be removed via choosing a proper preconditioner in this setting, since U is not changing. We also have extremely fast query time, as the solution vector can be quickly updated via a matrix-vector product instead of solving the regression problem from scratch.
When the condition number κ is as large as poly(n) and U is dynamically changing, Theorem 1.8 no longer gives efficient space and preprocessing time. To address this problem, we develop an algorithm with only logarithmic dependence on the condition number κ, based on the bounded computation path technique (Ben-Eliezer et al., 2020). Theorem 1.10. Let U ∈ R n×d , b ∈ R n and {v 1 , . . . , v T } be a sequence of adaptive updates to (U, b), satisfying Assumption 1.4. Let P denote set of possible output sequences the algorithm can provide to the adversary; note that we always have |P| ≤ (nκ) Θ(dT ) . There exists an adaptive algorithm that (1
) Preprocesses (U, b) in time O(nd ω-2 (d + log |P| + log 1 δ )/α 2 ); (2) Uses space O(d(d+log |P|+log 1 δ )/α 2 ); (3) For all t ∈ [T ], it updates (U t-1 , b t-1 ) to (U t , b t ) in time O((d+log |P|+log 1 δ )/α 2 ); (4
) For all t ∈ [T ], it returns a (1 + α)-approximate solution x t to the regression problem using (U t , b t ) in time O(d ω-1 (d + log |P| + log 1 δ )/α 2 ), with probability at least 1 -δ.
Compared to Theorem 1.8, Theorem 1.10 offers a better dependence on the condition number κ at the expense of a linear dependence on the length of update sequence T . While one might be tempted to simply generate an independent sketch for each update, it is worth noting that |P| could be much smaller than (nκ) Θ(dT ) whenever the updates and the solutions are known to be relatively stable. As a concrete example, when only one entry changes in between queries, then the number of computation paths is only (nd) T , which may be much less than (nκ) Θ(dT ) . In such scenario, it requires smaller number of sketches with improved space, preprocessing and update time. This is similar to past work where the complexity depends on a stability parameter (Hassidim et al., 2022).
this section cite: ['b10']

Section: Related Work Differential Privacy.
Differential privacy is a central concept in data privacy, introduced in (Dwork et al., 2006). The main idea of differential privacy is that when the inputs to the algorithm are close to each other, it would be almost impossible to differentiate the outputs. Since its introduction, differential privacy has seen rich applications in general machine learning (Chaudhuri & Monteleoni, 2008;Williams & McSherry, 2010;Jayaraman & Evans, 2019;Triastcyn & Faltings, 2020), deep neural network (Abadi et al., 2016;Bagdasaryan et al., 2019), computer vision (Zhu et al., 2020;Luo et al., 2021;Tan et al., 2019), natural language processing (Yue et al., 2021;Weggenmann & Kerschbaum, 2018), federated learning (Sun et al., 2023;Song et al., 2023a) and large language model (Yu et al., 2022;Gao et al., 2023;Liang et al., 2024;Gu et al., 2025;Nagesh* et al., 2025). Designing data structures with differential privacy guarantees is crucial, as it automatically ensures the privacy of any downstream tasks (Cohen-Addad et al., 2022;Dhulipala et al., 2023;Li et al., 2023;Chen et al., 2023;Andoni et al., 2023;Li et al., 2024). It is of particular interest to design differentially private data structure in the function release model, where the quality of the output won't degrade as the data structure processes more queries (Hall et al., 2013;Huang & Roth, 2014;Wang et al., 2016;Aldà & Rubinstein, 2017;Coleman & Shrivastava, 2021;Wagner et al., 2023;Backurs et al., 2024;Liu et al., 2024;Hu et al., 2024;Ke et al., 2025;Li et al., 2025).
Adaptive Data Structure. In recent years, data structures have been integrated into iterative processes to speed up the algorithm. This has been the foundation for various recent breakthroughs in fast convex optimization (Cohen et al., 2019;Lee et al., 2019;Brand et al., 2020;2022;Jiang et al., 2021;Qin et al., 2023). These data structures possess the ability to answer adaptive queries, that could depend on previous outputs from the data structure, with high success probability. For streaming problems, the adversary could also be adaptive, in the sense that it would feed the streaming algorithm with updates, after observing the prior decisions of the algorithm (Ben-Eliezer et al., 2020;Woodruff & Zhou, 2021;Ajtai et al., 2022;Feng & Woodruff, 2023;Woodruff & Zhou, 2024;Feng et al., 2024;Gribelyuk et al., 2024;Gu et al., 2025). In this work, we focus in particular on the adaptive data structures based on differential privacy (Hassidim et al., 2022;Beimel et al., 2022;Song et al., 2023b;Cherapanamjeri et al., 2023).
this section cite: ['b36', 'b17', 'b93', 'b53', 'b88', 'b0', 'b7', 'b101', 'b73', 'b87', 'b100', 'b91', 'b85', 'b99', 'b41', 'b70', 'b44', 'b74', 'b26', 'b32', 'b67', 'b18', 'b5', 'b68', 'b47', 'b50', 'b90', 'b3', 'b27', 'b89', 'b6', 'b71', 'b49', 'b60', 'b69', 'b25', 'b66', 'b12', 'b92', 'b54', 'b77', 'b10', 'b97', 'b2', 'b39', 'b40', 'b43', 'b44', 'b9', 'b22']

Section: Technical Overview
We divide the technical overview into two parts. For adaptive ANN, we demonstrate a novel framework based on differentially private selection over a sparse vector. For adaptive regression, we show how to upgrade the private median framework of (Beimel et al., 2022) to output the solution vector, and how to obtain utility guarantees through a novel use of ℓ ∞ guarantees provided by the sketch-andsolve framework (Price et al., 2017).
this section cite: ['b9', 'b76']

Section: Adaptive ANN via Differentially Private Selection
Existing Results. Before providing an overview of our techniques, we start by examining existing results for adaptive approximate nearest neighbor search data structures. The first candidates are of course deterministic data structures. Deterministic approximate nearest neighbor data structures have been a central topic of study since the 1970s (de Berg et al., 2008;Cormen et al., 2022). However, without any structural assumptions on the query or dataset, these data structures suffer from the curse of dimensionality, i.e., their preprocessing time and space usage scale exponentially in d, making them infeasible for slightly large dimensions. If one is willing to make strong structural assumptions on the dataset, e.g., given any query point v, the number of points in B(v, 2r) only grows by a constant factor compared to the number of points in B(v, r) (Clarkson, 1997;Karger & Ruhl, 2002;Beygelzimer et al., 2006;Krauthgamer & Lee, 2004b), then it is possible to design a data structure with polynomial preprocessing time and space (note that the dependence on the growth constant or doubling dimension is large, but still polynomial), and logarithmic query time. However, these assumptions are strong, as they greatly limit the potential geometric structure of the dataset.
The celebrated work of Indyk and Motwani (Indyk & Motwani, 1998) shows that if one relaxes the problem to allow for answering c-approximate near neighbor queries instead, then one can achieve (slightly) super-linear preprocessing time and space, and sublinear query time. These data structures are inherently random, as they rely on partitioning the space using random directions. As this randomness is determined during the preprocessing phase, these data structures are not robust against an adaptive adversary. In fact, for Hamming approximate near neighbor search, (Kapralov et al., 2024) provides an efficient attack that can always force the data structure to output a false negative if there exists at least one point that is isolated from other points.
Since the issue of false negatives is most prevalent for Monte Carlo data structures, one might consider to use a class of Las Vegas ANN data structures whose running times are random variables, but are guaranteed to output a correct answer with no false negatives (Kushilevitz et al., 1998;Pagh, 2016;Ahle, 2017;Wei, 2022). Unfortunately, the space and runtime analysis of these data structures are performed based on the assumption that the query sequence is oblivious. An adaptive adversary could design a sequence of queries with a much higher average response time than an oblivious one.
this section cite: ['b31', 'b28', 'b23', 'b58', 'b11', 'b52', 'b57', 'b64', 'b75', 'b1', 'b92']

Section: Reduction To Differentially Private Selection.
We start by introducing the differentially private selection problem. Given n categories, a collection of s binary vectors over {0, 1} n denoted by b (1) , . . . , b (s) , the goal is to find the category j * = arg max j∈[n] s i=1 b (i),j , i.e., the most common category among all vectors. This can be achieved via the following mechanism: 1). Compute the overall count vector B = s i=1 b (i) ; 2). Add independent Laplace noise Lap(1/ε) to each of the counts; 3). Report the index with the largest noisy count. The privacy parameter ε does not scale with either the number n of categories or the number s of vectors, as it only outputs a single index.
We now show how to frame approximate near neighbor search as a differentially private selection problem. We create a category for each point u ∈ U , and for each data structure, and we ask it to output all the near neighbors it finds instead of a single one of them. This creates an indicator vector for the points: if the data structure finds u i , then the corresponding vector has its i-th entry equal to 1. We can then apply the differentially private selection procedure for these indicator vectors, and output the point with maximum noisy count. To see this mechanism indeed protects the privacy of the internal randomness of the data structures, note that fixing the dataset U and the adaptive query point v, the indicator vector is determined by the random strings of these data structures. Thus, the mechanism is (ε, 0)-DP, and we can apply the advanced composition theorem of differential privacy to reduce the number of data structures from T to O( √ T ).
This is, however, not sufficient to provide a utility guarantee, as the Laplace noise could have large magnitude, thus making all counts too noisy. While it is fortunate that we can set ε = O(1), and with high probability, the magnitude of the noise is Θ(log n), this might still be too large to be useful.
To see how this issue can be resolved, let us consider the case that the query point v only has one approximate near neighbor, i.e., there exists only one u for which ∥v -u∥ ≤ r and ∥v -u∥ ≤ cr. In this case, the data structure can only output u (conditioned on the data structure succeeding). If we were to sample ω(log n) data structures and compute the count vector, we can guarantee that the entry corresponding to the near-neighbor has larger magnitude than the noise. Thus, as long as a constant fraction of the data structures succeed, we can ensure we output the correct point with high probability.
Note that our above argument assumes that a constant fraction of data structures succeed, which is not guaranteed when facing an adaptive adversary! Fortunately, we can circumvent this problem by proving the algorithm is differentially private first; subsequently, we can apply the generalization property of DP to ensure that indeed, a constant fraction of data structures succeed. Note the two-stage nature of this argument: we can only argue about the utility if the privacy is preserved. There are two main issues left: (1) the assumption that the query has only one near-neighbor is too restrictive, and (2) the differentially private selection procedure takes O(n) time to respond to each query, making the data structure very inefficient. For the first issue, we note that we can extend this to the setting when v has only s approximate near-neighbors. In this setting, we can sample ω(s • log n) data structures for each query. By the pigeonhole principle, there must exist at least one point whose count has magnitude larger than the noise, and we obtain the desired utility guarantee. Since the LSH data structures only output n ρ points for each query, we could pick s = O(n ρ ), i.e., allow the query to have O(n ρ ) nearneighbors. We can alternatively translate this assumption into a structural property on the dataset: as long as the ball centered at each point with radius cr intersects at most s other balls, then we are guaranteed that the query has at most s near-neighbors. We note that this assumption is automatically satisfied for Hamming nearest-neighbor search and for the setting of (Kapralov et al., 2024).
The second issue is algorithmic: if we naïvely implement the differentially private selection mechanism for each query, we will have to generate noise for each count which inevitably takes Ω(n) time. On the other hand, the count vector B is s-sparse, and these non-zero entries have magnitude larger than the noise. This problem has been studied before in the context of publishing a private database for sparse data (Cormode et al., 2012); however, existing solutions either require modifying the problem definition so that the privacy becomes challenging to prove, or converting a Monte Carlo algorithm into a Las Vegas one, with only expected runtime guarantees. We develop a novel algorithm to resolve these issues termed as the sparse argmax mechanism: given an s-sparse vector, (1) Adding s exponential noises Exp(1/ε) to the support of the sparse vector;
(2) Generating the X from the n-th order statistics distribution of Exp(1/ε); (3) Flip a biased coin with head probability s n , if head, generate s -1 i.i.d. exponential noises until none of them exceed X, add them including X to the support, and output the maximum entry index; (4) If tail, generate s i.i.d. exponential noises until none of them exceed X, add these noises to the support, assign X a random index in the n -s non-support, output the maximum index associated with the noisy entries and X. We prove that with high probability, generating these noises can be done O(s log n) time, and the output distribution with sparse noises is the same as the generating n i.i.d. exponential noises.
this section cite: ['b57', 'b29']

Section: Adaptive Regression via Differentially Private
Median and ℓ ∞ Guarantee Existing Results. We first recall the standard setup of solving the over-constrained ℓ 2 regression problem. Given a design matrix U ∈ R n×d with n ≫ d and a response vector b, the goal is to compute x * := arg min x∈R d ∥U x -b∥ 2 2 . In the static setting, x * can be computed via the normal equations: x * = (U ⊤ U ) † U b, but this is usually too expensive to be directly solved. The sketch-and-solve paradigm (see, e.g., (Woodruff, 2014) for a survey) provides a wide array of algorithmic tools to speed up this process. In particular, one can pick a random matrix S ∈ R r×n from a certain structured family of random matrices, so that r is a small polynomial in d, and S can be quickly applied to U . One can then solve the sketched ℓ 2 regression problem min x∈R d ∥SU x -Sb∥ 2 2 , for which the optimal solution is a good approximation to x * . The ℓ 2 regression problem has also been extensively in the streaming (Clarkson & Woodruff, 2009;Braverman et al., 2021) and dynamic (Jiang et al., 2023;Cherapanamjeri et al., 2023) models, where either the design matrix U or the response vector b can be updated, and one has to produce a high quality approximate solution after each update. These works are either not robust to adaptive adversaries (Clarkson & Woodruff, 2009), only support inserting or removing entire rows at once (Braverman et al., 2021;Jiang et al., 2023), or can only return the cost instead of the solution vector (Cherapanamjeri et al., 2023). In the realistic settings, the design matrix needs to tolerate perturbations to its entries due to the presence of noise or updated data. We consider the most general model where U can be updated by perturbation v t ∈ R n×d that can be adaptively chosen.
Coordinate-wise Private Median and the ℓ ∞ Guarantee. One natural idea is to extend the private median framework of (Beimel et al., 2022) to outputting an approximation to the solution vector. Our algorithm follows the generic template: prepare k copies of sketching matrices S 1 , . . . , S k and preprocess (U, b) as (S i U, S i b) for all i ∈ [k]. During an adaptive update, we simply update the corresponding sketches of the design matrix and of the response vector. At query time, we sample s = O(1) of our sketches and solve these sketched regression problems. Let x (1) , . . . , x (s) be the returned solution vectors. We next need to perform a private aggregation on these vectors to craft our output.
The (Beimel et al., 2022) approach for private aggregation is to compute the private median of these numbers. We first consider a natural extension: for each i ∈
[d], we compute g i = PMEDIAN((x (1) ) i , . . . , (x (s) ) i ) and output g = (g 1 , . . . , g d ).
The privacy of this approach is readily established: since each entry of g is private, we can conclude the final privacy guarantee by using the advanced composition theorem. This comes at a price of requiring
√ T d sketches instead of the √
T data structures of (Beimel et al., 2022), but it still offers an improvement as long as d ≪ T . The main challenge now lies in proving the utility of our approach, where we need to show that g is in fact a good enough solution to the regression problem.
To quantify ∥U g -b∥ 2 , note that ∥U g -b∥ 2 ≤ ∥U x * -b∥ 2 + ∥U (x * -g)∥ 2 , where the second term can be bounded by ∥U (
x * -g)∥ 2 ≤ σ max (U ) • ∥x * -g∥ 2 ≤ σ max (U ) √ d • ∥x * -g∥ ∞ .
In other words, if we can get a good handle on ∥x * -g∥ ∞ , then we can hopefully obtain a useful bound on the error. Since g is the coordinate-wise private median of x (1) , . . . , x (s) , which are solutions of sketched ℓ 2 regression problems, the standard sketching error guarantee only ensures that the forward error is small, i.e., ∥U x (i) -b∥ 2 ≤ (1 + α)∥U x * -b∥ 2 for any i ∈ [s]. For a backward error type guarantee, i.e., a bound on ∥x (i) -x * ∥ 2 , one could convert directly from the forward error guarantee. If U is reasonably well-conditioned, then closeness in forward error implies closeness in backward error.
However, for the private median guarantee, it is important that each entry of x (i) -x * is small, rather than just the bound that ∥x (i) -x * ∥ 2 is small. While the ideal scenario would be ∥x
(i) -x * ∥ ∞ ≈ 1 √ d ∥x (i) -x * ∥ 2 ,
this is generally not true for most sketching matrices. Fortunately, for sketching matrices such as the Subsampled Randomized Hadamard Transform (SRHT), it has been shown that the sketched solution in fact has a much stronger ℓ ∞ guarantee (Price et al., 2017;Song et al., 2023c):
∥x (i) -x * ∥ ∞ ≤ α √ d • ∥U x * -b∥ 2 • 1 σmin(U )
. By properly choosing the parameters for the the private median estimator, we can show that with high probability this also holds for g. Hence, we have ∥U (
x * -g)∥ 2 ≤ σ max (U ) √ d • ∥x * -g∥ ∞ ≤ ακ(U ) • ∥U x * -b∥ 2 .
To offset the blowup in condition number, we scale down α by a factor of κ, and thus establish the utility of the coordinate-wise private median mechanism. We find the private median to be a surprising application of sketching with the ℓ ∞ guarantee, which is a less studied guarantee in the sketching literature.
To further speed up the preprocessing and update time, we compose the SRHT sketch with Count Sketch matrices (Charikar et al., 2002) so that both operations can be realized in input sparsity time. In addition, these sketches and the sketched design matrices can be stored in O(d 2 κ 2 /α 2 ) words of space. Since we use O(
√ T d) in- dependent sketches, the space usage is O( √ T d 2.5 κ 2 /α 2 ).
this section cite: ['b96', 'b24', 'b14', 'b55', 'b22', 'b24', 'b14', 'b55', 'b22', 'b9', 'b9', 'b9', 'b76', 'b16']

Section: Applications
In this section, we utilize our adaptive ANN data structures to speed up a range of optimization processes and other data structures. Let T mat (a, b, c) to be the time complexity of multiplying an a × b matrix with a b × c matrix.
this section cite: []

Section: Online Weighted Matching
Consider the following online weighted matching problem on bipartite graphs: we are given a set of left vertices U ⊆ R d and we will receive a sequence of online right vertices V := {v 1 , . . . , v n } ⊆ R d . The edge weight between a left vertex u ∈ U and a right vertex v ∈ V is defined to be 1 ∥u-v∥ for some norm ∥ • ∥. Our goal is to design an algorithm that given a vertex v t , make an immediate decision to match with a point u t ∈ U with edge weight 1 ∥ut-vt∥ . The goal is to compute an online matching that maximizes the total weight T t=1 1 ∥ut-vt∥ . The online weighted matching problem has been widely studied in the edge-arrival model (Fahrbach et al., 2022), but in many practical machine learning applications, one usually encounters the vertex-arrival model (Karp et al., 1990). For example, the left vertices of the graph represent Uber/Lyft drivers, and the right vertices represent customers. The goal is to match each incoming customer with a driver that is closest in geographical distance. Another common application is movie recommendations on streaming platforms such as Netflix. Here the left vertices are movies, and the right vertices are viewers, and the goal is to find each viewer his/her favorite movie, defined by the distance between the feature embeddings of movies and viewers (Koren et al., 2009;Su & Khoshgoftaar, 2009;Shi et al., 2014). 5We consider the vertex-arrival online weighted matching problem when the right vertices are chosen by an adaptive adversary, i.e., the next vertex v t+1 could depend on previous matching results {(u i , v i )} t i=1 . This setting is particularly common for practical applications such as Ride Apps, since if a certain app raises its price due to high demand, the customers might choose to use another app, or move to a new location with smaller demand. A good approximation scheme for this problem is the greedy matching approach, i.e., each time we observe v t , we simply choose u t to be the point which maximizes 1 ∥ut-vt∥ (Karp et al., 1990). One could reduce this problem to a nearest neighbor search problem against an adaptive adversary. There is a caveat though: if we restrict it to be a bipartite matching, then once a vertex u is matched, it should not be matched again. This means our adaptive LSH should delete the point that has been outputted by the data structure after each query. We show that it is possible to further augment our framework for the data structure to be decremental, and the deletion can be performed efficiently using fast rectangular matrix multiplication (Duan et al., 2023). Before proceeding, we let θ(a, b) for a, b > 0 be the value for which T mat (a, b, θ(a, b)) = (ab) 1+o(1) .
Theorem 4.1 (Online Weighted Matching). Let U ⊆ R d be an n-point dataset satisfying Assumption 1.3 with parameter s. Given a possibly adaptive sequence {v 1 , . . . , v n } ⊆ R d , we can design an adaptive data structure which (1)
Preprocesses U in time O(T mat ( √ T •s, d, n)+ √ T •s•n 1+ρ ); (2) Uses space S space = O( √ T • s • n 1+ρ + nd); (3) Given a point v t , it returns a point in U \ {u 1 , . . . , u t-1 } that is a 1.1c-approximate near neighbor of v t , in time O(s•(d+n ρ )).
This step succeeds with probability at least 1-δ; (4) Deletes a point u ∈ U from the data structure in amortized time O((
√ T • s • d) 1+o(1) /θ( √ T • s, d) + √ T • s • n ρ ).
To get a better perspective on the deletion time, suppose s = poly log n. Also note that if √ T ≥ d, then we could simply use d independent copies via a net argument, so we assume
√ T ≤ d. Hence, θ( √ T , d) ≥ θ( √ T , √ T ) ≥ T α/2
, where α ≈ 0.32 is the dual matrix multiplication exponent (Williams et al., 2024;Le Gall, 2024). The runtime can be further simplified to (T 1/2-α/2 • d) 1+o(1) + T 1/2 • n ρ . We compare this result to updating all O( √ T ) data structures, which takes a total of T 1/2 • d + T 1/2 • n ρ time. By utilizing fast rectangular matrix multiplication and batch updates, we improve the exponent on √ T for the first term.
this section cite: ['b38', 'b59', 'b61', 'b84', 'b78', 'b59', 'b34', 'b95', 'b65']

Section: Terminal Embedding
A terminal embedding concerns the following problem: given a metric space (X, d X ) and a set of points u 1 , . . . , u n ∈ X, the goal is to design an embedding to another metric space (Y, d Y ) such that for any q ∈ X,
C • d X (u i , q) ≤ d Y (u i , q) ≤ Cρ • d X (u i , q) for all i ∈ [n],
where C > 0 is a constant and ρ ≥ 1 is the distortion factor. In contrast to metric embeddings such as the Johnson-Lindenstrauss lemma, a terminal embedding requires the distance to be preserved between a fixed set of terminals and all points in the metric space. When both X and Y are the Euclidean metric, it has been shown that an embedding dimension of O(ε -2 log n) is possible for 1 + ε distortion. (Cherapanamjeri & Nelson, 2021) shows that it is possible to implement a data structure that answers queries in time O(n 1-Θ(ε 2 ) + d) and space O(dn 1+o( 1) ). At the core of their algorithm is an adaptive ANN data structure. In particular, they create O(d) independent copies to deal with issues of adaptivity. If you know in advance that you only need to answer T adaptive queries for T ≤ d, then the O(d) copies via a net argument is sub-optimal. For a fair comparison, in the following we will assume (Cherapanamjeri & Nelson, 2021) uses T copies of data structures instead of d. We apply the adaptive LSH data structure we have designed to obtain an improvement in the space complexity of (Cherapanamjeri & Nelson, 2021)'s data structure when √ T • s ≤ d and the query points q satisfy Assumption 1.2. Before proceeding, we need to introduce a few concepts.
Definition 4.2. Given U = {u 1 , . . . , u n } ⊆ R d and ε > 0, we say a matrix S ∈ R k×d is a convex hull distortion for U if for any z ∈ conv(T ) where T = { u-v ∥u-v∥ : u, v ∈ U }, we have |∥Sz∥ -∥z∥| ≤ ε.
Theorem 4.3 (Terminal Embedding). Let U ⊆ R d be an n-point dataset and V = {v 1 , . . . , v T } ⊆ R d be a sequence of T adaptive queries that satisfy Assumption 1.2 with parameter s. Let ρ 1 , ρ 2 , ρ 3 , ρ 4 and ρ rep > 0 be parameters. There exists a randomized algorithm that computes a data structure D and a linear map S ∈ R k×d , such that (1) S is a convex hull distortion for U ; (2) Given any v t ∈ V , D produces a vector z vt ∈ R k+1 such that with probability at least
1 -1/ poly(n), (1 -ε) • ∥v t -u∥ ≤ z vt - Su 0 ≤ (1 + ε) • ∥v t -u∥ for all u ∈ U . (3) For any v t ∈ V , the runtime of computing z vt is O(s•(d+n ρ2 +n ρ4 +n ρ4+(1+ρ3-ρ4-ρrep)ρ2 )); (4) The space complexity of D is O( √ T • s • (n ρrep+(1+ρ1) + n ρ3+(1+ρ1) )).
To interpret our result, we note that the query time is increased by a factor of s, as in all previous results. The space complexity is reduced from T to √ T • s. If s = poly log n, then the query time is only increased by a polylogarithmic factor, and the space complexity is improved by a factor of √ T .
Theorem B.2 (Advanced Composition, see (Dwork et al., 2010)). Given three parameters ε, δ 0 , δ ≥ 0. If A 1 , • • • , A k are each (ε, δ)-DP algorithms, then the k-fold adaptive composition
A k • • • • • A 1 is (ε 0 , δ 0 + kδ)-DP where ε 0 = 2k ln(1/δ 0 ) • ε + 2kε 2
One can boost the privacy budget by subsampling the dataset first, then running the differentially private algorithm on the subsampled set. We state a version where δ = 0. Note that it can be easily extended to account for non-zero δ.
Theorem B.3 (Amplification via Sampling (Lemma 4.12 of (Bun et al., 2015))). Let ε ∈ (0, 1] be parameters. Let A denote an (ε, 0)-DP algorithm. Let S denote a dataset.
Suppose that A ′ is an algorithm that,
• constructs a database T ⊂ S by subsampling with repetition k ≤ n/2 rows from S,
• returns A(T ).
Then, we have
A ′ is 6k n ε, 0 -DP.
this section cite: ['b37', 'b15']

Section: Theorem B.4 (Generalization of Differential Privacy (DP)).
Given two accuracy parameters ε ∈ (0, 1/3) and δ ∈ (0, ε/4), suppose that the parameter t satisfies that t ≥ ε -2 log(2ε/δ).
We use D to represent a distribution over a domain X . Suppose S ∼ D t is a database containing t elements sampled independently from D. Let A be an algorithm that, given any database S of size t, outputs a predicate h : X → {0, 1}.
If A is (ε, δ)-DP, then the empirical average of h on sample S, i.e.,
h(S) = 1 |S| x∈S h(x),
and h's expectation over underlying distribution D, i.e.,
h(D) = E x∼D [h(x)]
are within 10ε with probability at least 1 -δ/ε:
Pr S∼D t ,h←A(S) h(S) -h(D) ≥ 10ε ≤ δ/ε.
We next consider the task of differentially private selection: we have a discrete space of outcomes, and the goal is to output the largest histogram cell:
Definition B.5 (Differentially Private Selection). Given n categories, let the database S be such that, each of its rows is a binary vector of length n, with each entry corresponding to whether the row belongs to the i-th category or not. We say an algorithm is an (ε, 0)-differentially private selection if it releases the index of the approximately most popular category over the database S, and is (ε, 0)-DP.
For any differentially private mechanism, one cares about privacy but also utility, i.e., after adding noise, how good are the estimates compared to the algorithm without adding noise? Here, we focus on privacy, and we will later use the privacy to provide utility guarantees. For count queries, the most common approach is the Laplace mechanism, where properly chosen Laplace noises are added to each count. Here, we use a similar approach that instead adds exponential noises. This approach is more widely adopted for Exponential Mechanism, yet it still offers the same privacy budget as that of the standard Laplace Mechanism for reporting the noisy max index. See e.g., (Dwork & Roth, 2014;Ding et al., 2021) for more details.
Theorem B.6 (REPORTONESIDEDNOISYARGMAX, Theorem 3.13 of (Dwork & Roth, 2014)). Given a database S with each row being a binary vector of length n, consider the following algorithm:
• Compute the count for each category;
• Add an i.i.d. exponential noise Exp(1/ε) to each count;
• Output the category with the maximum noisy count.
The algorithm gives (ε, 0)-differentially private selection.
this section cite: ['b35', 'b33', 'b35']

Section: B.2. Reducing Monte Carlo Search to Selection: A Slow Algorithm
In this section, we show how to utilize the REPORTNOISYMAX procedure for adaptive data structure design. Recall the framework introduced in (Hassidim et al., 2022) for streaming and later extended for improving the preprocessing and update time for dynamic data structures (Beimel et al., 2022;Song et al., 2023b;Cherapanamjeri et al., 2023): the idea is to set up the database as the random strings used by oblivious data structures. Suppose we have A 1 , . . . , A k oblivious data structures, each with their associated random strings r 1 , . . . , r k ∈ {0, 1} * . The database is R, where the i-th row is the random string r i . Notions such as sensitivity are in turn defined with respect to these random strings. We will now illustrate how to tie these Monte Carlo search data structures to the selection problem. We will use (c, r)-ANN as an example, but our reduction does not exploit any additional structure on (c, r)-ANN, so it can be well-extended to more general search data structures.
Given a dataset U ∈ R d and a possibly adaptive query v ∈ R d , recall that a (c, r)-ANN data structure would have the following behavior:
• If there exists some u ∈ U with ∥u -v∥ ≤ r, then it outputs some point u ′ ∈ U with ∥u ′ -v∥ ≤ cr. This succeeds with probability at least 9 10 (note we do not need probability boosting for our base data structure, at this stage). We modify the data structure to output all such points it has found.
• If there is no such point, report NULL.
Note that ANN data structures can only produce false negatives but no false positives. We can associate the output of the data structure with a binary vector of length n as follows: the i-th entry of the vector corresponds to the decision the data structure has made on the point u i , i.e., if it outputs u i , then we set the i-th entry to be 1, and 0 otherwise. If the data structure outputs nothing, we set the vector to be 0 n . Fixing dataset U and query v, these binary vectors are completely determined by r 1 , . . . , r k . Denote these binary vectors as b 1 , . . . , b k . We then execute the REPORTONESIDEDNOISYARGMAX mechanism on these vectors, and output the corresponding index. We now describe the details of our algorithm.
We will use T to denote the number of queries, and k to denote the total number of data structures. We assume each data structure succeeds with probability at least 1 -δ. We let ε DP denote the DP parameter for REPORTONESIDED-NOISYARGMAX. We will use δ fail to denote the overall failure probability of our algorithm, and β = δ fail /T to denote the failure probability of a single query step. The algorithm is as follows.
Initialization.
• Prepare k independent copies of oblivious data structures, denoted by A 1 , . . . , A k , over the dataset U .
this section cite: ['b9', 'b22']

Section: Query.
• Receive query vector v ∈ R d .
• Sample l = O(s) indices independent and uniformly from [k] with replacement. Denote the corresponding data structures by A (1) , . . . , A (l) .
• Feed v into A (1) , . . . , A (l) , receive binary vectors b (1) , . . . , b (l) .
• Compute REPORTONESIDEDNOISYARGMAX(b (1) , . . . , b (l) ) with parameter ε DP . Let i be the corresponding index.
• Output u i if ∥u i -v∥ 2 ≤ cr, otherwise output NULL.
Update.
• Receive update vector u ∈ R d .
• Update A 1 , . . . , A k with u.
We will specify the parameters to show that the algorithm is indeed differentially private with respect to the internal randomness of the data structures.
Setting Parameters. We start by setting the parameter ε DP which affects the privacy guarantee of REPORTONESIDED-NOISYARGMAX. We will set ε DP = 1/2 (in fact, for our analysis, ε DP can be any constant smaller than 1). Set the number of samples c to be
l = O(s log 2 (n/β))
and the total number of data structures k to be
k = 200 • 6l • ε DP • 2T ln(100/β) = O( √ T • s).
Privacy Guarantees. Our analysis differs from standard DP analysis, in which one could reason over the privacy and utility simultaneously. We will start by showing that the algorithm is indeed differentially private, then use this fact to argue the output of the algorithm provides good utility.
To understand the privacy of the algorithm, we adapt the framework from (Beimel et al., 2022): let r 1 , . . . , r k ∈ {0, 1} * denote the random strings used by data structures A 1 , . . . , A k and let R = {r 1 , . . . , r k } be the database whose i-th row is r i . We will prove the transcript of the interaction between the adaptive adversary and algorithm is differentially private with respect to R. Fix a time step t. We let out t (R) denote the index output by the algorithm at time t. Even fixing R, the output out t (R) is still a random variable as it depends on i.i.d. exponential noise that is oblivious to the algorithm. Similar to (Beimel et al., 2022), we define the transcript at time t as T t (R) = (v t , out t (R)). Define the overall transcript as
T (R) = T 1 (R), . . . , T T (R).
We view T t and T as algorithms that given a database R, output the transcripts. This enables us to reason about differential privacy with respect to R. Lemma B.7. For any time step t, T t is ( 6l k • ε DP , 0)-DP with respect to R.
Proof. Note that T t (R) = (v t , out t (R)). For any fixed single step t, v t does not provide extra information about R, so we only need to consider out t (R).
We let out t (R) := REPORTONESIDEDNOISYARGMAX(b (1) , . . . , b (l) ) with parameter ε DP . As we have argued in the preceding discussion, the binary vectors are determined by {r (1) , . . . , r (l) } ⊂ R which are subsampled from R. As out t (R) is (ε DP , 0)-DP with respect to the subset (Theorem B.6), by Theorem B.3, it is ( 6l k • ε DP , 0)-DP with respect to R. Finally, observe that out t (R) is a post-processing of out t (R) as it's a deterministic mapping that only depends on the output of out t (R). Hence, we conclude that out t (R) is ( 6l k • ε DP , 0)-DP with respect to R, as desired.
Given that a single step is DP with respect to R, we can then perform advanced composition to prove the privacy of T . Lemma B.8. T is ( 1 100 , β 100 )-DP with respect to R.
Proof. By Lemma B.7, we know that for any fixed t ∈ [T ], T t (R) is ( 6l k • ε DP , 0)-DP with respect to R. Further, observe that T is an adaptive composition of T T • . . . • T 1 , so we can apply Theorem B.2 with ε = 6l k • ε DP and δ = 0, δ 0 = β/100. This yields a mechanism that is (ε 0 , δ 0 )-DP with respect to R, where
ε 0 = 2T ln(100/β) • ε + 2T ε 2 ≤ 1 200 + 1 200 = 1 100
.
Thus, we conclude T is ( 1 100 , β 100 )-DP with respect to R.
Accuracy: Differentiating Signals from Noise. Given the privacy guarantees of T , we are ready to prove accuracy against an adaptive adversary. Similar to (Beimel et al., 2022), define v t = (v 1 , v 2 , . . . , v t ) to be the query sequence up to time t and consider feeding a random string r (for initialization) and v t to an oblivious data structure A. Let A(r, v t ) denote the output. Finally, define acc vt (r) to be the indicator of whether A(r, v t ) succeeds, i.e., if v t is a "yes" instance, then A(r, v t ) indeed outputs a subset of points in U that are at most cr away from v t . We will prove that with high probability, a large constant fraction of all data structures succeed.
Lemma B.9. For any fixed time step t ∈ [T ], we have k j=1 acc vt (r j ) ≥ 4 5 k with probability at least 1 -β.
Proof. The proof will rely on the generalization property of DP. First note that acc vt is determined by the transcript T , as v t is a substring of T and v t determines acc vt . To invoke Theorem B.4, we first note that each row of R is drawn uniformly, and due to Lemma B.8, we know that T is ( 1 100 , β 100 )-DP with respect to R. Thus, we have that the empirical average of acc vt over R which is 1 k k j=1 acc vt (r j ) and the expectation of acc vt over the uniform distribution E[acc vt (r)] are close. Setting ε = 1/100, δ = β/100 and k ≫ 1 ε 2 log(2ε/δ), we conclude
Pr   1 k k j=1 acc vt (r j ) -E[acc vt (r)] ≥ 1 10   ≤ β.
It remains to get a good handle on the expectation. If we let U denote the uniform distribution we sample r from, then we can write the expectation E r∼U [acc vt (r)] = Pr r∼U [acc vt (r) = 1] as acc vt (r) is an indicator. The crucial point here is that the probability is taken over the randomness of r when the sequence of queries v t is fixed. Thus, we can utilize the property of our oblivious data structure, i.e., Pr r∼U [acc vt (r) = 1] ≥ 9 10 . To put it together, we have
1 k k j=1 acc vt (r j ) ≥ 9 10 - 1 10 = 4 5 .
Thus, we complete the proof.
To improve the efficiency of querying, recall that we resort to sampling. We prove that sampling does not hurt the accuracy.
Lemma B.10. For all t ∈ [T ], let l be the sampled indices of time t, we have l j=1 acc vt (r (j) ) ≥ 3 4 l with probability at least 1 -δ fail .
Proof. Fix any t ∈ [T ], we condition on the event that Lemma B.9. Therefore k j=1 acc vt (r j ) ≥ 4 5 k. This means that any i.i.d. sample (with replacement) from these k indices succeeds with probability at least 4/5. Consequently
E[ l j=1 acc vt (r (j) )] ≥ 4
5 l. By Hoeffding's bound, we have
Pr   l j=1 acc vt (r (j) ) -E[ l j=1 acc vt (r (j) )] ≥ α   ≤ 2 exp(- 2α 2 l ),
and setting α = 1 5 l, we conclude that l j=1 acc vt (r (j) ) ≥ 3 5 l with probability at least 1 -exp(-Θ(l)).
We present an abstract way to reason over the data structure task. Fix the preprocessed dataset U ∈ R d . Consider any (possibly adaptive) query v ∈ R d , and let b ∈ {0, 1} n denote the characteristic vector of v with respect to U , in terms of (c, r)-ANN: if there exists some u i ∈ U with ∥u i -v∥ ≤ r, then: for any j ∈ [n] with ∥u j -v∥ ≤ cr, we set b j = 1. Note that if no point in U is r-close to v, then b is either the all-0s vector or it has some nonzero entries that are cr-close to v. Similarly, for each data structure A 1 , . . . , A k and their corresponding random strings r 1 , . . . , r k , we use b (i) to denote the characteristic vector under the random string r i , i.e., b
j = 1 if and only if the LSH A i discovers that ∥u j -v∥ ≤ cr. Note that it is completely possible that the vector b has large support size, but some b (i) = 0 n . Now, imagine we have an O(n) time budget. Then our algorithm is essentially identical to reporting the noisy max: we first sum over all characteristic vectors: b sum = k i=1 b (i) . Note that the i-th entry of b sum counts the total number of successful data structures that report point u i . Then, we sample n independent Laplacian noise variables of scale 1/ε DP , and then add these noise variables to each entry of b sum . Then we simply report the maximum index of the noisy counts with a simple post-processing by directly computing the distance between the corresponding point and the query. The algorithm is naturally (ε DP , 0)-DP with respect to the random strings r 1 , . . . , r k ! To prove the utility of the algorithm, i.e., that we can actually differentiate the signal from the noise, we require the following tail bound for exponential random variables:
Lemma B.11. Let Y ∼ Exp(b), then Pr[Y ≥ t • b] ≤ exp(-t)
Let us choose t = C log n for some absolute constant C > 0. This means that even if we choose ε DP to be 1/2, the magnitude of the noise will be roughly Θ(log n). Consider a simple case, where the query v only has one point u ∈ U with ∥u -v∥ 2 ≤ r and ∥u -v∥ 2 ≤ cr. Then, for a successful data structure A i , it will have exactly one non-zero entry. Let us assume that we sample l = O(log 2 n) independent data structures to perform the mechanism, using ε DP = 1/2. Then by amplification via subsampling, the algorithm is 6l k • ε DP -DP. Performing an advanced composition over all T queries, we conclude the algorithm is ( 1 100 , 1 100•poly(n) )-DP. We can then use the generalization property of DP to conclude that, with probability at least 1 -1/ poly(n), at least a 3/4-fraction of the sampled data structures succeed.
Conditioning on these events, we argue that the noisy max is indeed the correct answer: we know that a large constant fraction of data structures succeed, therefore the corresponding point in b sum has count Ω(log 2 n). On the other hand, the exponential noise itself has magnitude at most O(log n). This means that adding the exponential noise, we can still differentiate the point. We formalize this argument with the following lemma.
Lemma B.12. For all t ∈ [T ], the algorithm outputs a point u t that is a c-ANN for the adaptive query v t if v t satisfies Assumption 1.2, with probability at least 1 -δ fail -1/ poly(n).
Proof. By Lemma B.10, among the l samples, at least a 3 4 -fraction of the data structures succeed, i.e., if there exists an r-near neighbor of v t , then at least 3 4 l data structures have their characteristic vectors with at least 1 non-zero entry. By Assumption 1.2, the support size of each characteristic vector can be at most s; since we set l = O(s log 2 (n/β)), by the pigeonhole principle, there exists at least one entry in b sum with magnitude Ω(log 2 (n/β)). Now, conditioning on the event that all n Laplacian noise variables have magnitude O(log n); this holds with probability 1 -1/ poly(n). Thus, we know that there exists at least one entry of b sum which has magnitude Ω(log 2 n) and we can differentiate this entry from all the noise added. Now, consider the case that there is no point u ∈ U with ∥u -v t ∥ 2 ≤ r. Then the data structure is allowed to either return points that are cr-near neighbors of v t , or return nothing. Since the base data structure either outputs a false negative (a cr-near neighbor) or outputs nothing, we analyze two cases:
• Case 1: ∥b sum ∥ ∞ ≥ C log 2 n for some fixed constant C. In this case, we know that the max index before adding the noise must correspond to a cr-near neighbor. Hence, adding the noise and outputting the noisy max will not hide this large entry, and so we can return it and satisfy the specification of an c-ANN data structure.
• Case 2: ∥b sum ∥ ∞ < C log 2 n. In this case, it must be the case there is no r-near neighbor. Regarding the REPORTONESIDEDNOISYARGMAX mechanism, since the magnitude of the signal in this case is O(log n) and with high probability, the exponential noise has magnitude O(log n), we have that REPORTONESIDEDNOISYARGMAX outputs a noisy index (meaning that the corresponding index is not a cr-near neighbor). In the post-processing phase, we ensure the algorithm outputs only a cr-near neighbor by checking the distance.
This concludes the proof of correctness.
To compute the failure probability, note that a constant fraction of the data structures succeed with probability at least 1 -δ fail , and all exponential random variables have magnitude Θ(log n) with probability at least 1 -1/ poly(n). A union bound concludes the bound on the failure probability.
Of course, this algorithm is by no means efficient -it takes O(n) time to generate all n noise variables per query, making the sublinear query time linear. In the next section, we will show it is sufficient to generate s exponential variables for the support of b sum , and take the noisy max over these entries.
this section cite: ['b9', 'b9', 'b9']

Section: B.3. Sparse Noise and Order Statistics: A Fast Algorithm
We make the following observation for the REPORTONESIDEDNOISYARGMAX algorithm: if b sum = 0 n , then the entry with the max noise must be the one contains X (n) , the largest order statistics of the exponential distribution. If b sum has s nonzero entries, then there are two cases: if X (n) is among the nonzero entries, since those entries are all positive, the max entry must be within the s nonzero entries; if X (n) is not among the nonzero entries, then the max entry is among the entry contains X (n) , and the nonzero entry with the largest magnitude after adding noises. Note that the first event happens with probability s n , and for this case, we generate s -1 i.i.d. noises until they do not exceed X (n) so that they obey the order statistics distribution. For the second case, we generate s i.i.d. noises until they do not exceed X (n) , randomly assign a zero entry to X (n) , and output the max between X (n) and the largest nonzero noisy entry. Note that this algorithm is extremely efficient -according to Fact A.3, X (n) can be generated in O(1) time, and the above procedure runs in O(s log n) time with high probability, proportional to the support size.
Fact B.13. Let Z be a geometric random variable with success probability p ∈ (0, 1), then for any positive integer k, we have
Pr[Y > k] ≤ exp(-k log(1/(1 -p)))
Proof. Note that by definition,
Pr[Y > k] = (1 -p) k = exp(k log(1 -p)) = exp(-k log(1/(1 -p))).
Lemma B.14. Let A 1 , A 2 be two algorithms, with the following behavior. Let s > 0:
• A 1 , at time t, it receives an s-sparse vector u ∈ N n and let U t ⊆ [n] denote the support of u. We add a dense exponential noise vector with parameter 2, denoted by η ∈ R n and add this to u ∈ R n , and output index i * such that i * = arg max j∈[n] (u + η) j ;
• A 2 , at time t, it receives an s-sparse vector u ∈ N n and let U t ⊆ [n] denote the support of u. We flip a biased coin with probability of being head s n . If head, we generate a max noise X from the distribution X (n) of parameter 2, and then repeatedly generating s -1 i.i.d. exponential noises of parameter 2 until none of them exceed X, and let η 1 , . . . , η s-1 denote these noises, form vector η ∈ R n with the support being U t , and the values being randomly assigned from η 1 , . . . , η s-1 , X. Then output i * = arg max j∈[n] (η + µ) j . If the coin flips tail, then generate X from X (n) , and generate s i.i.d. exponential noises until none of them exceed X. Randomly assign X to one of the non-support entry denoted by i 1 , and let η denote the noises for the support, output i 1 if X > max j∈[n] (µ + η) j and output arg max j∈[n] (µ + η) j otherwise.
Then, A 1 and A 2 have the same output distribution. Moreover, A 2 runs in O(s log n) time.
Proof. We will prove that A 2 has exactly the same output distribution as A 1 . We do so by analyzing an hypothetical algorithm A ′ 2 that generates X from X (n) first and assign it to a uniformly chosen entry i * , then generate the noises for all other entries conditioning they are smaller than X. We will prove the distribution of the noise generated by A ′ 2 is identical to A 1 , then it's easy to see that the output distribution of A 2 is the same as A 1 .
Suppose the noise η i is generated from distribution with PDF f and CDF F , then the CDF of X (n) is G(x) = F (x) n , and PDF g(x) = nF (x) n-1 f (x). Let the joint CDF of noise generated by A 1 be F
1 (x 1 , . . . , x n ) = F (x 1 ) • F (x 2 ) • . . . • F (x n ),
and the PDF noise generated by A ′ 2 be F 2 (x 1 , . . . , x n ). Let the sorted array of x 1 , . . . , x n be x * 1 , . . . , x * n , and x * 0 = -∞ we have :
F 2 (x 1 , . . . , x n ) = 1 n n i=1 xi 0 g(x) xj <x F (x j ) F (x) dx = n i=1 i j=1 x * j x * j-1 f (x)F (x) n-1 j-1 k=1 F (x * k ) F (x) dx = n j=1 (n -j + 1) x * j x * j-1 f (x)F (x) n-j j-1 k=1 F (x * k ) dx = n j=1 j-1 k=1 F (x * k ) • F (x * j ) n-j+1 -F (x * j-1 ) n-j+1 = F (x * 1 ) • • • • • F (x * n ) = F 1 (x 1 , . . . , x n )
Hence, the noise distributions of these two algorithms are identical.
this section cite: []

Section: To see A 2 and A ′
2 have the same noise distribution, consider two cases. If the max noise X is in the support (happens with probability s n ), then we know that the max index must be within the support. For the noise distribution, it's easy to see that as long as all η i 's are smaller than X, then the noise distributions are identical. For the tail case, the reasoning is similar, except that our maximum must be over X and the noisy entries in the support.
Regarding the running time, we only need to examine the probability that the noises η 1 , . . . , η s-1 or η 1 , . . . , η s do not exceed X. We without loss of generality prove for the case where we generate s independent exponential noises. Let λ be the parameter, note that this is equivalent to the probability that Y (s) ≤ X (n) , where Y (s) is the max order statistics for another independent sequence of exponential noises. The CDF of Y (s) is F Y (s) (y) = (1 -e -λy ) s , and the PDF of
X (n) is f X (n) (x) = nλe -λx (1 -e -λx ) n-1 , then we have Pr[Y (s) ≤ X (n) ] = ∞ 0 Pr[Y (s) ≤ x]f X (n) (x) dx = ∞ 0 (1 -e -λx ) s nλe -λx (1 -e -λx ) n-1 dx = ∞ 0 nλe -λx (1 -e -λx ) n+s-1 dx
we do a change of variable u = 1 -e -λx , so du = λe -λx dx, and for x = 0, u = 0 and for x = ∞, u = 1, therefore
∞ 0 nλe -λx (1 -e -λx ) n+s-1 dx = n 1 0 u s+n-1 du = n n + s ,
similarly, for s -1 noises, the probability is n n+s-1 . In both cases, we have that the success probability is at least 1 2 , and note that this is a geometric random variable Z, by Fact B.13, we have that Pr[Z > k] ≤ exp(-k), set k = c log n for some large enough constant c so that k is an integer, we have this happens with probability at most 1/ poly(n), hence with high probability, we only need to generate the noises for k = O(log n) times. Moreover, X can be efficiently generated via inverse CDF sampling, so the overall runtime for generating these noises is O(s log n), as desired.
this section cite: []

Section: C. Speeding Up Updates via Batching
In this section, we focus on developing a fast update procedure for decremental ℓ 2 LSH against an adaptive adversary. This is crucial for applications such as online weighted matching (see Section 4.1).
this section cite: []

Section: C.1. Adaptive Low-Dimensional ℓ 2 LSH
We start by reviewing the algorithm for low-dimensional ℓ 2 LSH, against an oblivious adversary. Throughout, we let ε, δ ∈ (0, 1) denote the precision and failure probability.
Initialization.
• Prepare one Johnson-Lindenstrauss (Johnson & Lindenstrauss, 1984) transform S : R d → R m where m = O(ε -2 log(n/δ)).
• Compute SU .
• Initialize an (c, r)-ℓ 2 LSH data structure on SU .
Query.
• Given query point v ∈ R d , first compute Sv.
• Query the LSH with Sv, output the corresponding point.
Deletion.
• Given a point u ∈ U to-be-deleted, first compute Su.
• Locate Su in the hash buckets of the LSH, remove Su.
The deletion procedure first computes the embedded point in time kd = O(ε -2 d log(n/δ)), then locates the hash bucket in O(n ρ ) time. In order to make such data structure adaptive, we need to create √ T • s independent copies, and naïvely update all √ T • s data structures takes time (up to polylogarithmic factors and assume ε = O(1)):
√ T • s • (d + n ρ )
While the term √ T • s • n ρ seems to be unavoidable as we will have to update all LSH data structures, the first step of applying Johnson-Lindenstrauss could be further acclerated via batching and fast rectangular matrix multiplication.
this section cite: ['b56']

Section: C.2. Faster Update via Batching
Recall that we define θ(a, b) for a, b > 0 to be the value such that T mat (a, b, θ(a, b)) = (ab) 1+o(1) . Note that θ is monotone in its argument, i.e., fixing one argument, θ grows or decreases with the other argument. Our idea is the following: instead of eagerly updating all data structures whenever a point is deleted, we can delay the deletion until the data structure is queried, which is when the update must be performed. We will partition the length-T query sequence into blocks of size θ( √ T • s, d) and for simplicity, we will denote it as θ in the following.
this section cite: []

Section: Initialization.
• Prepare k = √ T • s JL transforms (Johnson & Lindenstrauss, 1984) S 1 , . . . , S k : R d → R m where m = O(ε -2 log(n/δ)).
•
Set S =      S 1 S 2 . . . S k      ∈ R mk×d .
• Compute SU .
• Initialize k (c, r)-ℓ 2 LSH's on S 1 U, S 2 U, . . . , S k U .
• Associate a deletion list to each data structure. The initial deletion lists are empty. Also initialize a global deletion list that keeps track of all points deleted so far.
Query.
• Receive query point v t ∈ R d .
• Sample l = O(s log 2 n) data structures, denote the sampled JLs as S (1) , . . . , S (l) .
• Batch compute
   S (1) . . . S (l)    v t ∈ R lm .
• Update the sampled data structures that are not up-to-date, update their corresponding deletion lists.
• Query corresponding LSH's with S (1) v t , . . . , S (l) v t .
• Compute the characteristic vectors of these LSH's outputs, apply sparse REPORTONESIDEDNOISYARGMAX to the count vector and compute the order statistics.
• Output the noisy max index.
Deletion.
• If the algorithm is at the end of a block:
-Update all k data structures with θ points in the block. In particular, let u (1) , . . . , u (θ) denote the points to-bedeleted.
-Compute S   | . . . | u (1) . . . u (θ) | . . . |   . -Delete S i u (1) , . . . , S i u (θ) from the i-th LSH.
-Update the deletion list of all k data structures and the global deletion list.
• Otherwise, update the global deletion list.
We first note that the correctness of the data structure follows directly from the design: for each query, we will update the data structures first. So we focus on analyzing the runtime.
Theorem C.1. Let U ⊆ R d satisfy Assumption 1.3 and {v 1 , . . . , v T } ⊆ R d be a sequence of adaptive queries. There exists a randomized data structure with the following guarantees:
• It preprocesses U in time O(T mat ( √ T • s, d, n) + √ T • s • n 1+ρ ); • It uses space O( √ T • s • n 1+ρ + nd);
• Given a query v t for t ∈ [T ], it returns a point u ∈ U with ∥v t -u∥ 2 ≤ cr if B(v, r) ∩ U ̸ = ∅ and NULL otherwise, with probability at least 1 -1/ poly(n). This procedure takes time O(s
• (d + n ρ )); • If u t is the output of v t , it deletes u t in amortized time O(( √ T • s • d) 1+o(1) /θ( √ T • s, d) + √ T • s • n ρ ).
Proof. We note that preprocessing time, space and query time, together with their guarantees are straightforward, so we will focus on bounding the deletion time. Our analysis will be over a block of size θ, and the final update time will be amortized over θ steps. There are two cases to consider.
Case 1. Update during the query. Each time we receive a query, we need to sample l = O(s log 2 n) data structures and update these data structures accordingly. Suppose we are at the i-th step of the block. Then in the worst case, all of these data structures need to be updated for all prior i -1 points deleted. We can compute the update time over the block as follows:
θ i=1 T mat (l, d, i) time to apply JL + iln ρ time to update LSH ≤ θ i=1 O(is • (d + n ρ )) = O(θ 2 s • (d + n ρ )).
Thus, the amortized cost per update-during-the-query is O(θs
• (d + n ρ )).
Case 2. Update at the end of the block. In this case, we need to update all k = O( √ T • s) data structures with θ points. Applying JL takes time
T mat ( √ T • s, d, θ( √ T • s, d)) = O(( √ T • s • d) 1+o(1) )
by the definition of θ, and updating k LSH's takes time O(θ
• √ T • s • n ρ ). Thus, the amortized cost per step is O(( √ T • s • d) 1+o(1) /θ + √ T • s • n ρ ).
As the second cost dominates, we obtain the desired update time.
Remark C.2. As our data structure gains an advantage over using d copies via a net argument when
√ T • s ≤ d, we know that θ( √ T • s, d) ≥ θ( √ T • s, √ T • s) = ( √ T • s) α
as θ is monotone, and α ≈ 0.32 is the dual matrix multiplication exponent (Williams et al., 2024;Le Gall, 2024) where T mat (n, n, n α ) = n 2+o(1) . This implies an amortized cost per update of
( √ T • s) 1-α+o(1) • d 1+o(1) + √ T • s • n ρ ≈ ( √ T • s) 0.68+o(1) • d 1+o(1) + √ T • s • n ρ(1)
Compared to the naïve update in which the first term is √ T • s • d, this improvement is significant for relatively large √ T • s.
this section cite: ['b56', 'b95', 'b65']

Section: C.3. Generalization to ℓ p LSH
We note that the above algorithm does not exploit any particular structure of ℓ 2 ; we use the Johnson-Lindenstrauss transform (Johnson & Lindenstrauss, 1984) in its most general formulation, as the speedup we obtain comes from batch matrix multiplication. Thus, we could generalize the algorithm to any ℓ p norm for p ∈ (0, 2], using p-stable sketches (Indyk, 2006;Datar et al., 2004).
Corollary C.3. Let U ⊆ R d satisfying Assumption 1.3 and {v 1 , . . . , v T } ⊆ R d be a sequence of adaptive queries. Let p ∈ (0, 2].
There exists a randomized data structure with the following guarantees:
• It preprocesses U in time O(T mat ( √ T • s, d, n) + √ T • s • n 1+ρ ); • It uses space O( √ T • s • n 1+ρ + nd);
• Given a query v t for t ∈ [T ], it returns a point u ∈ U with ∥v t -u∥ p ≤ cr if B(v, r) ∩ U ̸ = ∅ and NULL otherwise, with probability at least 1 -1/ poly(n). This procedure takes time O(s
• (d + n ρ )); • If u t is the output of v t , it deletes u t in amortized time O(( √ T • s • d) 1+o(1) /θ( √ T • s, d) + √ T • s • n ρ ).
this section cite: ['b56', 'b51', 'b30']

Section: D. Adaptive Regression with Private Median and ℓ ∞ Guarantee
In this section, we provide the necessary background for adaptive regression problem under turnstile updates, with a variety of algorithms with diverse guarantees.
this section cite: []

Section: D.1. Preliminaries on Adpative Regression and Sketching
The adaptive regression we will be studying is defined in Assumption 1.4, we will state here again for completeness.
Let U ∈ R n×d be a design matrix and b ∈ R n be a response vector with n ≫ d, the goal is to solve the ℓ 2 regression problem:
x * := arg min
x∈R d ∥U x -b∥ 2 2 ,
with the optimal solution given by the normal equation:
x * = (U ⊤ U ) † U ⊤ b
with M † being the Moore-Penrose inverse of the matrix M . As solving the normal equation exactly is time-and spaceconsuming, so one is usually interested in finding an approximate solution x ∈ R d such that
∥U x -b∥ 2 ≤ (1 + α)∥U x * -b∥ 2 for some α > 0.
In the adaptive turnstile update model, we assume an adversary could curate a sequence of T updates {v 1 , . . . , v T } adaptively, where each v t for t ∈ [T ] could take one of the two forms:
• v t ∈ R n×d , i.e., v t is an update to the design matrix;
• v t ∈ R n , i.e., v t is an update to the response vector.
We need to design an algorithm that processes these adaptive updates to either U or b, for simplicity we will use (U t , b t ) to denote the design matrix and response vector after the t-th update. Our algorithm needs to respond with an approximate solution x t to the t-th ℓ 2 regression:
∥U t x t -b t ∥ 2 ≤ (1 + α) min x∈R d ∥U t x -b t ∥ 2 .
The main objective is to design an algorithm that • Uses space that is sublinear in n, sublinear in T and small polynomial in d;
• Has efficient update that depends on the size of v t and small polynomial in d.
In the following, for the simplicity of presentation, we will make common assumptions that the length of the stream is T = poly(n), and that in any round t ≤ T , entries of U t , b t are always in the range [-n γ , n γ ] for some constant γ > 0. These assumptions can be removed at the expense of introducing additional factors to our results.
Sketching will be a key algorithmic tool to speed up the procedure of solving ℓ 2 regression, we introduce them in the following.
Definition D.1 (Oblivious Subspace Embedding). Fix dimension parameters n, d, approximation factor α, and failure probability β. Suppose D is a distribution on r × n matrices S, where r is a function of n, d, α, and β. Suppose that with probability at least 1 -β, for any fixed n × d matrix U , a matrix S drawn from D satisfies: for all vectors x ∈ R d ,
∥SU x∥ 2 2 = (1 ± α)∥U x∥ 2 2 ,
then we call D an (α, β)-Oblivious Subspace Embedding (OSE).
We will mainly need the OSE property with α = O(1) and β = O(1). We refer to this as O(1)-OSE. We list a collection of sketching matrices that will be used throughout this section.
Definition D.2 (Count Sketch (Charikar et al., 2002)). Let S ∈ R r×n be constructed via the following procedure: Randomly draw h : [n] → [r] from a pairwise independent hash family, and draw σ : [n] → {-1, 1} from a 4-wise independent hash family. For each of the n columns S i∈[n] , we choose a row h(i) ∈ [r] and an element σ(i) from {-1, 1}. We set S h(i),i = σ(i) for all i ∈ [n], and set all other entries of S to be 0. We call such S a Count Sketch matrix.
Definition D.3 (Gaussian Sketching Matrix). We say S ∈ R r×n is a Gaussian sketching matrix if all entries are independently sampled from the distribution N (0, 1/r).
Definition D.4 (Subspace Randomized Hardamard Transform (SRHT) (Lu et al., 2013)). The Subspace Randomized Hardamard Transform (SRHT) matrix S ∈ R r×n is defined as the scaled matrix product S := 1 √ r P HD, where each row of matrix P ∈ {0, 1} r×n contains exactly one 1 at a random position, H is the n × n Hadamard matrix, and D is a n × n diagonal matrix with each diagonal entry being a value in {-1, 1} with equal probability.
We will use the following differentially private median procedure:
Theorem D.5. There exists an (ε, 0)-differentially private algorithm that given a database S ∈ X * , outputs an element x ∈ X such that with probability at least 1 -β, there are at least |S|/2 -Γ elements in S that are bigger or equal to x, and there are at least |S|/2 -Γ elements in S that are smaller or equal to x, where
Γ = O( 1 ε log |X| β ). Moreover, private median runs in time O(ε -1 |S| log 3 (|X|/β) • poly log |S|).
We will use PMEDIAN(x 1 , . . . , x s ) to denote the invocation of private median on x 1 , . . . , x s .
this section cite: ['b16', 'b72']

Section: D.2. Generic Algorithm with Private Median
A critical property we will be leveraging in designing the algorithm is the ℓ ∞ guarantee of the sketched solution (Price et al., 2017;Song et al., 2023c): this guarantee states that the sketched solution x and the optimal solution x * are close not only in terms of their costs, i.e., ∥U x -b∥ 2 = (1 ± α)∥U x * -b∥ 2 , but are close in the sense that ∥ x -x * ∥ ∞ is small. Note that a naïve bound of ∥ x -x * ∥ ∞ is just ∥ x -x * ∥ 2 , as one could have the scenario that all the discrepancy concentrate in a few coordinates with most coordinates are the same. When S is an SRHT matrix, (Price et al., 2017) shows that ∥ x -x * ∥ 2 is too pessimistic and a stronger bound exists, in particular, they prove
∥ x -x * ∥ ∞ ≤ α √ d ∥U x * -b∥ 2 • 1 σ min (U )(2)
holds with probability 1 -β. The number of rows required is further improved in (Song et al., 2023c). To improve the space usage and runtime efficiency, we further compose the SRHT matrix with a Count Sketch matrix.
Lemma D.6. Let D be a distribution of matrix product S SRHT • S CS , where S SRHT ∈ R r×m is an SRHT matrix and
S CS ∈ R m×n is a Count Sketch matrix. For some r = O( d log 3 (n/β)α 2
) and m = O(d 2 + d α 2 β ), D satisfies the ℓ ∞ guarantee in Equation 2. We will refer this property as (α, β)-accuracy.
Proof. As shown in (Price et al., 2017), all we need is that both S SRHT and S CS are drawn from distributions that individually satisfy Equation (2) (up to a constant factor loss in failure probability and approximation factor). The main result of (Song et al., 2023c) proves that SRHT with r rows satisfies Equation (2).
For S CS , following the core lemma in (Song et al., 2023c), it suffices to show that it is an Oblivious Subspace Embedding (OSE) with an O(1) approximation factor, and it satisfies the following Matrix Multiplication guarantee: for any fixed vectors g, h ∈ R n ,
Pr[|g ⊤ S ⊤ CS S CS h -g ⊤ h| ≥ α √ m ∥g∥ 2 ∥h∥ 2 ] ≤ β.
It is shown, e.g. in (Woodruff, 2014) that 1. Count Sketch with m = O(d 2 / poly log d) is an O(1)-OSE, and 2. Count Sketch with m = O(d/α 2 β) satisfies the Approximate Matrix Multiplication property.
This concludes that D satisfies the ℓ ∞ guarantee in Equation ( 2).
We are now ready to describe our algorithm. Throughout, we will let α ∈ (0, 1) be the approximation factor, β ∈ (0, 1) be the failure probability and we require an extra parameter κ which is an upper bound on the condition number of U t throughout the update sequence. Since our algorithm would utilize private median, we let ε DP to denote the privacy parameter which will be specified later. We will refer this algorithm to ADAPTIVEREGDP.
Initialization.
• (Parameters for sketches): Let α ′ = α κ and
β ′ = 0.01, set r = O( d log 3 (n/β ′ ) α ′2
) and m = O(d 2 + d α ′2 β ′ );
• (Parameters for privacy): Let Γ = O( 1 εDP log T d|X| β ) where X = {-n γ , . . . , -n -γ , 0, n -γ , . . . , n γ }, set k = O(ε DP • Γ • T d log(1/β));
• Prepare k independent copies S 1 , . . . , S k ∈ R r×n according to Lemma D.6;
• Let sk i U = S i U and sk i b = S i b for all i ∈ [k].
Update.
• Receive update v t ;
• If v t is an update to U , then update
sk i U ← sk i U + S i v t for all i ∈ [k], otherwise update sk i b ← sk i b + S i v t .
Query.
• Sample with replacement s = O(1) indices from [k], let j 1 , . . . , j s denote the sampled indices;
• Compute x ji = arg min x∈R d ∥sk i U x -sk i b ∥ 2 for all i ∈ [s]; • Compute g l = PMEDIAN((x j1 ) l , . . . , (x js ) l )
with privacy budget ε DP (Theorem D.5) for all l ∈ [d];
• Output g = (g 1 , g 2 , . . . , g d ).
We now prove the privacy and utility of our algorithm.
Lemma D.7. The algorithm ADAPTIVEREGDP satisfies (ε, δ)-differential privacy w.r.t. the collection of random strings R for ε := 1/100 and δ := β/100.
Proof. By Theorem D.5, each instance of primed is (ε DP , 0)-differential private. But since we sample s = O(1) copies of the oblivious algorithm to use, this amplifies the privacy of each PMEDIAN call to ( 6s K ε DP , 0)-DP by Theorem B.3. In total, we have at most d • T instances of PMEDIAN.
By the advanced composition theorem, the entire algorithm is (ε, δ)-differential private for
ε = 2T d ln(100/β) • ( 6s k ε DP ) + 2T d • ( 6s k ε DP ) 2 ≤ 1/100
and δ = 100/β, since we set K = 200 • 6sε DP • 2T d ln(100/β) .
In the following discussion, we follow previous work (Hassidim et al., 2022) and assume the returned solution vectors to the regressions have their coordinates in the range
[-n γ , -1/n γ ] ∪ {0} ∪ [1/n γ , n γ ].
Lemma D.8. With probability at least 1 -β, in all rounds t ∈ [T ] during the update sequence, The algorithm ADAP-TIVEREGDP outputs g that satisfies
∥U t g -b t ∥ 2 ≤ (1 + α) min x∈R d ∥U t x -b t ∥ 2 ,
for underlying design matrix U t and vector b t .
Proof. Consider any round t ∈ [T ] during the stream, let (U t , b t ) be the underlying vectors defined by the stream up to round t. And let σ min denote the minimum singular value of U t and σ max denote the maximum. Let f Ut,bt (r) be the indicator for the following event:
∥x * -x t ∥ ∞ ≤ α ′ √ d ∥U t x * -b t ∥ 2 • 1 σ min
(i.e., Equation ( 2)) :
x * := arg min
x∈R d ∥U t x -b t ∥ 2 x t := arg min x∈R d ∥S(U t x -b t )∥ 2
where S is generated as Lemma D.6, using random string r.
For ε = 1/100 and δ = β/100, observe that s ≫ 1 ε 2 log( 2ε δ ). Thus, we can apply Theorem B.4 with n = s to show that
| E r [f Ut,bt (r)] - 1 s s i=1 f Ut,bt (r ji )| ≤ 10ε = 1/10.
with probability at least 1 -δ/ε = 1 -β. In the following, we condition on the event that this holds.
We have E r [f Ut,bt (r)] ≥ 9/10 by the (α ′ , β ′ )-accuracy of each copy of sketch. Therefore, at least 4s/5 of the samples
{x ji : i ∈ [s]} satisfies ∥x * -x ji ∥ ∞ ≤ α ′ √ d ∥U t x * -b t ∥ 2 • 1 σmin .
We call such x ji a "good approximation". The algorithm runs PMEDIAN on each coordinate l ∈ [d] across all approximated vectors x j1 , x j2 , • • • , x js . For each l ∈ [d], Theorem D.5 combined with our choice of s = 100Γ guarantees that with probability at least 1 -β/(T d), we have |{i ∈ [s] : (x ji ) l ≥ (g) l }| ≥ 4s/10 and |{i ∈ [s] : (x ji ) l ≤ (g) l }| ≥ 4s/10.
Since there are at least 4s/5 good approximations satisfying Equation (2), this means there exist good approximations
x jp , x jq such that (g) l ∈ [(x jp ) l , (x jq ) l ], thus |(g) l -(x * ) l | ≤ α ′ √ d ∥U t x * -b t ∥ 2 • 1
σmin . This holds simultaneously for all l ∈ [d] and all k independent copies with probability at least 1 -β. Condition on the above event, we have
∥U t g -b t ∥ 2 ≤ ∥U t x * -b t ∥ 2 + ∥U t ( g -x * )∥ 2 ≤ ∥U t x * -b t ∥ 2 + σ max ∥ g -x * ∥ 2 ≤ ∥U t x * -b t ∥ 2 + σ max ( √ d • ∥ g -x * ∥ ∞ ) ≤ ∥U t x * -b t ∥ 2 + σ max σ min ( √ d α ′ √ d ∥U t x * -b t ∥ 2 ) = (1 + σ max σ min α ′ )∥U t x * -b t ∥ 2 .
By setting α ′ = α κ , this gives a (1 + α)-approximation. Overall, with probability at least 1 -β, all the approximation vectors are accurate to within a multiplicative error of (1 + α).
It remains to show that ADAPTIVEREGDP is both time-and space-efficient.
Theorem D.9. Given a sequence of T adaptive updates {v 1 , . . . , v T }, algorithm ADAPTIVEREGDP has the following specifications:
• It preprocesses (U, b) in time O( √ T d • (nnz(U ) + nnz(b) + d 3 + d 2 κ 2 /α 2 ));
• It uses space O( √ T • d 2.5 κ 2 /α 2 );
• Given an update v t for t ∈ [T ], it takes time O(
√ T d • (nnz(v t ) + d 3 + d 2 κ 2 /α 2 )) to update; • It outputs a (1 + α)-approximate solution x t in time O(d ω+1 κ 2 /α 2 ).
Proof. We prove the theorem item by item.
Preprocessing time. During preprocessing, we prepare k = O( √ T d) independent copies of sketching matrices due to Lemma D.6 and compute S i U, S i b for all i ∈ [k]. Note that S i is a composition of a Count Sketch matrix and an SRHT matrix, so S i U takes time O(nnz(U )) for the Count Sketch, and this results in a matrix of size m × d, applying SRHT to this matrix takes time O(md) = O(d 3 + d 2 κ 2 /α 2 ). Hence, the overall time for preprocessing is
O( √ T d • (nnz(U ) + nnz(b) + d 3 + d 2 κ 2 /α 2 )).
Space usage. The algorithm stores R and sk i U , sk i b for all i ∈ [k]. We start by considering the random strings to generate k pairs of SRHT and Count Sketch matrices. To store the pairwise and 4-wise independent hash functions for Count Sketch, we only need O(1) bits of space. To store an r × m SRHT matrix S SRHT = 1 √ r P HD, since H is deterministic, we only consider P ∈ R r×m and D ∈ R m×m . P contains exactly one 1 in each row (and contains 0 everywhere else), thus can be stored in O(r log m) bits of space. D is a diagonal matrix containing {-1, 1}, which can be stored in O(m) bits of space. Therefore, R in total takes
O( d log 3 n α ′2 log m + d 2 + d α ′2 ) = O( dκ 2 poly log n α 2 + d 2 )
bits of space.
For sk k U , sk k b for i ∈ [k], the total bits of space is
O(k • r • d log n) = O(poly log n • log( T β ) • T d log( 1 β ) • dκ 2 log 3 n α 2 • d log n) = O( d 2.5 κ 2 √ T α 2
• poly log(n, L, 1/β))
assuming a word size of O(log n) bits. This is also asymptotically the total space usage.
Update time. For update, we simply apply the sketch S i to the update v t for all i ∈ [k], hence the total update time is
O( √ T d • (nnz(v t ) + d 3 + d 2 κ 2 /α 2 )).
this section cite: ['b76', 'b76', 'b76', 'b96']

Section: Query time.
To answer the query, we need to solve a sketched regression for s instances, where each regression could be solved in time O(rd ω-1 ) = O(d ω+1 κ 2 /α 2 ). The private median is then performed over s numbers, and each private median runs in time O(s). Repeating this procedure for all d coordinates, the overall time for forming g is then O(d). Hence, the query time is
O(d ω+1 κ 2 /α 2 ).
this section cite: []

Section: D.3. Improved Algorithm with Bounded Computation Path Technique
When the condition number κ is as large as poly n, a quadratic dependence on κ is prohibitive. We provide an algorithm for such case based on the bounded computation path technique (Hassidim et al., 2022). Let P denote the set of all possible computation paths given the adaptive update sequence {v 1 , . . . , v T }. The intuition is that the total number of possible computation paths |P| is at most n Θ(dT ) , hence we could prepare a large Gaussian sketching matrix with size roughly log |P|, that is guaranteed to succeed against all such paths. In the case where the updates are stable, e.g., when the updates are always concentrated in a few coordinates, then |P| could be much smaller than n Θ(T ) . This leads to a more efficient alternative and a better dependence on the condition number κ. We will refer to this algorithm as ADAPTIVEREGPATH.
• (Parameters for rounding): Let P be the set of all possible output sequences of the algorithm and let β 0 = β/(C • |P|) for large constant C;
• (Parameters for sketches): Let r = O((d + log(1/β 0 ))/α 2 ), generate the Gaussian sketching matrix S ∈ R r×n by a length-O(r) bits random seed r S ;
• Let sk U = SU and sk b = Sb.
Update.
• Receive update v t ;
• Compute Sv t using r G and Lemma D.13, if v t is an update to U then sk U ← sk U + Sv t , otherwise sk b ← sk b + Sv t .
Query.
• Compute x t = arg min x∈R d ∥sk Ut x -sk bt ∥ 2 ;
• Round x t to the nearest point in any sequence in P, denote it by g;
• Output g.
We start by showing the utility guarantee of ADAPTIVEREGPATH when the updates are oblivious.
Claim D.10. If the update sequence is oblivious, at each round t ∈ [T ], ADAPTIVEREGPATH outputs a vector g that satisfies
∥U t g -b t ∥ 2 ≤ (1 + α) min x∈R d ∥U t x -b t ∥ 2
for underlying input matrix U t and vector b t , with probability at least 1 -β 0 .
This follows from our setting of r = O((d + log(1/β 0 ))/α 2 ), which is sufficient for the distribution of Gaussian matrices to be an (α, β 0 )-OSE (Woodruff, 2014).
Lemma D.11. The Gaussian sketching matrix S in ADPATIVEREGPATH can be pseudo-randomly generated using a random seed of length O(d + log(1/β 0 )/α 2 ) bits, while still satisfying Claim D.10.
Proof Sketch. This can be seen by opening up the proof e.g., in Section 2.1 of (Woodruff, 2014) that the Gaussian distribution satisfies OSE. The i.i.d. property is used for showing that the random Gaussian matrix is a JL transform (Lemma 2.12 in (Williams, 2012)). However, there one needs to consider at most an O(log(9 d /β 0 )/α 2 )-tuple of Gaussian variables, where 9 d is the size of the 1/2-net for a unit sphere in the subspace that we hope to embed in. Therefore the proof goes through as long as the entries of S are generated using at least r = O(d + log(1/β 0 )/α 2 )-wise independence. It is well known that such an r-wise independent hash function can be compressed as a length O(r)-bit seed, which concludes the proof.
Lemma D.12. With probability at least 1 -β, in all rounds t ∈ [T ], ADAPTIVEREGPATH given an adaptive update v t outputs a vector g that satisfies:
∥U t g -b t ∥ 2 ≤ (1 + α) min x∈R d ∥U t x -b t ∥ 2 ,
for underlying input matrix U t and vector b t .
Proof. As argued in (Ben-Eliezer et al., 2020), we may assume the adversary to be deterministic. This means, in particular, that the output sequence we provide to the adversary fully determines its stream of updates v 1 , v 2 , • • • , v T . Note that the collection of all distinct output sequence after the rounding step is at most |P|. Each output sequence as above uniquely determines a corresponding stream of updates for the deterministic adversary. For every stream, with probability at least 1 -β 0 , ADAPTIVEREGPATH will be correct in every round. Applying the union bound over these streams, we can conclude that with probability at least 1 -O(|P|) • β 0 = 1 -β, it outputs a (1 + α)-approximation in every round against any adversary.
Lemma D.13. The time to generate S via r G is O(rn).
Proof Sketch. Note that to generate one column S i of the sketching matrix, we need r evaluations of the r-wise independent hash function. Using fast multipoint evaluation of polynomials (Gathen & Gerhard, 2013), these evaluations can be done in O(r) time. Hence, generating S takes O(rn) time, as desired.
Theorem D.14. Given a sequence of T adaptive updates {v 1 , . . . , v T }, algorithm ADAPTIVEREGPATH has the following specifications:
• It preprocesses (U, b) in time O(nd ω-2 (d + log |P| + log 1 β )/α 2 ); • It uses space O(d(d + log |P| + log 1 β )/α 2 );
• Given an update v t for t ∈ [T ], it takes time O((d
+ log |P| + log 1 β )/α 2 ) to update; • It outputs a (1 + α)-approximate solution x t in time O(d ω-1 (d + log |P| + log 1 β )/α 2 ).
Here β denotes the failure probability.
Proof. We prove the theorem item by item. The parameter r is expanded as d + log |P| + log 1 β to obtain the theorem.
Preprocessing time. During preprocessing, we generate the sketching matrix S using our random seed, which by Lemma takes O(rn) time. Afterward, computing the sketches is dominated by computing the matrix product of r-by-n matrix S and n-by-d matrix U , which is O(rnd ω-2 ).
Space usage. The sketches sk U and sk b together take O(dr) bits of space, and the random seed takes O(r) bits of space.
Update time. We first generate one column of the sketching matrix, which takes O(r) time. Then we compute the inner product between this column and the update, which takes at most O(r) time.
this section cite: ['b96', 'b96', 'b94', 'b42']

Section: Query time.
To answer the query, we solve a sketched regression instance in time O(rd ω-1 ). After that, we round it to the nearest point in P, which can be done in O(log |P|) time using binary search.
this section cite: []

Section: D.4. Handling Sparse Label Shifts with Preconditioner
In machine learning, it is common that the design or data matrix remains unchanged, as it might be generated via embedding raw data into high dimensional vectors, and these embeddings are expensive to compute. The response or label vector, in contrary, might constantly drift. Hence (Cherapanamjeri et al., 2023) studies a model where only b receives a sparse update v t with ∥v t ∥ 0 ≤ s. For example, b could be the count of a large class of images, and only a few popular classes get their counts incremented frequently. In (Cherapanamjeri et al., 2023), they are only able to output regression cost against sparse, adaptive updates to b. Here, we develop an algorithm based on ADAPTIVEREGDP to output the solution vector. We will use ADAPTIVEREGPRECONDITIONER to denote this algorithm.
Initialization.
• (Parameters for sketches): Let α ′ = α and
β ′ = 0.01, set r = O( d log 3 (n/β ′ )α ′2
) and m = O(d 2 + d α ′2 β ′ );
• (Parameters for privacy): Let Γ = O( 1 εDP log T d|X| β ) where X = {-n γ , . . . , -n -γ , 0, n -γ , . . . , n γ }, set k = O(ε DP • Γ • T d log(1/β));
• (Parameters for batch size): Let T be a parameter denote the batch size depending on nnz(U ), d and α. Let counter = 1 be the batch counter parameter.
• Prepare k independent copies S 1 , . . . , S k ∈ R r×n according to Lemma D.6;
• Compute a preconditioner P ∈ R d×d such as κ(AP ) = O(1);
• Prepare M i = (S i U P ) † S i and sk i = M i b for all i ∈ [k].
Update.
• If counter = T , regenerate sketching matrices S 1 , . .
. , S k ∈ R r×n , set M i = (S i U P ) † S i and sk i = M i b for all i ∈ [k]. Reset the counter = 1; • Receive update v t ; • Update b ← b + v t and sk i ← sk i + M i v t for all i ∈ [k].
Query.
• Sample with replacement s = O(1) indices from [k], let j 1 , . . . , j s denote the sampled indices;
• Compute g l = PMEDIAN((sk j1 ) l , . . . , ((sk js ) l ) with privacy budget ε DP (Theorem D.5) for all l ∈ [d];
• Let g = (g 1 , g 2 , . . . , g d ), output g = P g.
We note that the privacy and utility guarantee remains the same as ADAPTIVEREGDP, so we focus on the complexity. Instead of fixing the length of update sequence in advance, we allow for arbitrary length of update sequence, and we will restart the data structure every T updates. Theorem D.15. Given a sequence of adaptive updates {v 1 , v 2 , . . . , }, algorithm ADAPTIVEREGPRECONDITIONER has the following specifications:
• It has amortized update time O( d/T • (nnz(U ) + nnz(b) + d 3 + d ω /α 2 ) + √ T d • (s + d 3 + d 2 /α 2 ));
• It outputs a (1 + α)-approximate solution x t in time O(d 2 ).
Proof. We analyze over batches.
At the start of each batch, we generate k sketching matrices and S i U for all i ∈ [k], this step takes O( √ T d • (nnz(U ) + d 3 + d 2 /α 2 )) time. Right multiplying by P then computing the pseudoinverse takes O( √ T d • d ω /α 2 ) time. Finally, note that we don't need to right multiply by S i , rather we could first compute S i b in nnz(b) + d 2 + d/α 2 time, then multiplying the matrix with the vector in O( √ T d • rd) = O( √ T d • d 2 /α 2 ) time. Hence, the amortized time for this process over T steps is O((
√ d • (nnz(U ) + nnz(b)) + d 3.5 + d ω+0.5 /α 2 )/ √ T ).
When receiving v t , it takes s time to update the response vector, and s
+ d 3 + d 2 /α 2 time to update sk i . Hence, the overall time is O( √ T • (sd 0.5 + d 3.5 + d 2.5 /α 2 )).
Now, to output a query, we simply compute private median over s sampled solution vectors in d coordinates, and output P g in d 2 time.
In the next few paragraphs, we will explain how to make choice for T .
For the case when nnz(U ) terms are small, and the ideal ω ≈ 2, the right choice of T = Θ(1).
For the case when nnz(U ) terms are small, and we use the current omega, the right choice is T = Θ(d ω-2 ).
For the case when nnz(U ) terms are large, then we just need to balance √ d nnz(U )/ √ T with (d 3.5 + d 2.5 /α 2 ) √ T which means we need to chose T = Θ( nnz(U ) d 3 +d 2 /α 2 ).
To obtain the optimal choice for the batch size T , we case on nnz(U ). Let C be some constant. Claim D.16. If nnz(U ) ≤ C • (d 3 + d ω /α 2 ) , then let T be O((d 3 + d ω /α 2 )/s), the update time of the algorithm is O( √ s • (d 2 + d ω/2+0.5 /α)).
Claim D.17. If nnz(U ) > C • (d 3 + d ω /α 2 ), then let T be O(nnz(U )/s), the update time of the algorithm is O( d • nnz(U )/s).
this section cite: ['b22', 'b22']

Section: E. Improved Algorithm Against Adversarial Attack for Hamming Space LSH
In a recent work, (Kapralov et al., 2024) shows that for Hamming space, there exists a query efficient algorithm that can quickly compute a point v adaptively, such that there exists u ∈ U such that ∥u -v∥ ≤ r (here ∥ • ∥ is the Hamming distance) but the data structure returns nothing. We state their result here.
Definition E.1. Let U ⊆ R d be an n-point dataset, we say a point z ∈ U is an isolated point if for all points u ∈ U with u ̸ = z, ∥u -z∥ ≥ 2cr. Here ∥ • ∥ is the Hamming distance.
Theorem E.2 ( (Kapralov et al., 2024)). Let n denote the size of dataset, d be the dimension, (c, r) be the parameters for ANN, λ be a complexity parameter. Suppose they satisfy the following relations:
• cr ≤ d;
• ln 3 n ≤ r ≤ d/ ln n; • λ ≤ min{ r ln n , n 1/8 }; • c ≥ 1 + ln λ ln n .
Then with probability at least 1/4 -1/n, there exists an algorithm that finds a point q such that an isolated point z in the dataset is at most r away from q, but the algorithm returns nothing. The algorithm makes O(log(cr) • λ) queries to the LSH.
In their original theorem statement, they also require c ≤ ln n. This is unnecessary, as it is derived using the upper bound r ≤ d/ ln n and cr ≤ d. If r is much smaller, then c can be larger. Their algorithm starts from the isolated point z, then gradually moves away from z while ensuring no other points could collide with it. In the end, it moves at most r-away from z, but with high probability, no point in U has been hashed to the same bucket as it.
To combat such an adversarial attack, one could either repeat the data structures d times or T times, which are both relatively inefficient. We will show that whenever the dimension d is relatively small, then we could use Theorem 1.6 with √ T • s data structures, against the attack of (Kapralov et al., 2024). Lemma E.3. Let n, d be positive integers and c ≥ 1 and r > 0 be parameters. Let ρ ∈ (0, 1) be a parameter. If 3d = s α and α ≤ 1 2cr , then Assumption 1.3 holds for any n-point set U ⊆ {0, 1} d . In particular, if 3d ≤ n ρ 2cr then Assumption 1.3 holds for all s ≤ n ρ .
Proof. We note that Assumption 1.3 could be rephrased as for each u ∈ U , it can have at most 2cr-near neighbors. In the Hamming space, it means that they could differ by at most 2cr bits. Fix a vector u. The total number of points that differ by at most 2cr bits from u is at most
2cr i=1 d i ≤ 2cr d 2cr ≤ (d • e) 2cr • (2cr) 1-2cr ≤ (3d) 2cr
as long as 2cr log s (3d) ≤ 1, we are guaranteed the total number of possible near neighbors is at most s. By a simple calculation, this also gives us the bound when s ≤ n ρ .
For Hamming space LSH, it is known that ρ = O(1/c), and thus the bound on d becomes d ≤ n Θ(1/(c 2 r)) . In order for it to be useful against Theorem E.2, we could let r = ln 3 n and c = 1 + o(1).
Thus the bound on d becomes d ≤ exp(-Θ(log 2 n)) = n o(1) . We obtain a corollary against Theorem E.2 in this setting. Corollary E.4. If d = n o(1) and with the parameters in Theorem E.2, there exists an adaptive LSH data structure for Hamming space with • Preprocessing time O( √ T • n 1+O(ρ) d); • Space usage O( √ T • n 1+O(ρ) + nd);
• For all t ∈ [T ], with high probability, if B(v t , r) ∩ U ̸ = ∅ then it returns a point in B(v t , cr) ∩ U in time O(n O(ρ) d).
Here B(•, •) is the Hamming ball.
In particular, to be robust against the attack of Theorem E.2, it suffices to pick T = O(log(cr) • λ).
The above corollary essentially asserts that, to be robust against the attack of (Kapralov et al., 2024), it is sufficient to blow up the number of data structures by a log(cr) • λ factor, even in the presence of an isolated point.
this section cite: ['b57', 'b57', 'b57', 'b57']

Section: F. Locating the Thin Level via Approximate Counting
If one considers Assumption 1.2 for (c, r)-ANN, it states that for an adaptive query v, if B(v, r) ∩ U ̸ = ∅ then |B(v, cr) ∩ U | ≤ s. In an iterative process, one usually does not care for a particular r but only wants to find an approximate nearest neighbor, and this is often achieved via binary search over the choice of r. We also note that since our assumption is strictly weaker than constant expansion and doubling dimension, it is completely possible that for some small r, Assumption 1.2 holds for query v and it no longer holds for 2r. In fact, even if |B(v, cr) ∩ U | is small, |B(v, 2cr) ∩ U | could be the entirety of U if the diameter of U is between cr and 2cr. Thus, in order to deploy our data structure, it is crucial that we have the ability to identify the thin level, i.e., given v, the largest possible r for which the assumption holds. Recall that due to efficiency concerns and the nature of LSH data structures, we set s ≤ n ρ .
To do so, we will utilize an approximate near neighbor counting procedure for ℓ 2 norm, developed in (Andoni et al., 2023).
In essence, they develop an algorithm based on the LSH scheme of (Andoni et al., 2017) that, given an oblivious query v, it can with high probability return an approximate count ans such that 1) ).
(1 -o(1)) • |B(v, r) ∩ U | ≤ ans ≤ |B(v, cr) ∩ U | + O(n ρ+o(
We first note this provides a tool for us to do binary search on r, to locate the thin level: we could start with small r, and run the approximate counting procedure of (Andoni et al., 2023). Conditioned on these counts being accurate, whenever we are still below or at the thin level, ans must be below O(n ρ+o(1) ). If we are at a level where |B(v, r) ∩ U | ≥ ω(n ρ+o(1) ), then the counting data structure could successfully detect it, so the only troublesome case is when 1) ). We can indeed detect this case by searching over the range (cr, c 2 r), and the data structure should indeed report that the count is already too large. We could then refine our search by looking at the range (r/c, r), in which the correct thin level is guaranteed to be in this interval.
|B(v, r) ∩ U | is small but |B(v, cr) ∩ U | ≥ ω(n ρ+o(
One last issue remains: the (Andoni et al., 2023) data structure only works for oblivious queries; for adaptive queries, they run a net argument with d independent copies. To improve upon that, we note that the data structure only outputs a real number, and thus it falls into the category of estimation data structures, which can be augmented via the framework of (Beimel et al., 2022).
Theorem F.1 (Adaptive Approximate Near Neighbor Counting). Let U ⊆ R d be an n-point dataset and {v 1 , . . . , v T } ⊆ R d be a sequence of adaptive queries. Let c ≥ 1, r ≥ 0 be parameters and ρ ∈ (0, 1) be a parameter that depends on c. There exists a randomized data structure with
• Preprocessing time O( √ T • n 1+o(1) d);
• Space usage O( √ T • n 1+o(1) d);
• Given v t , it outputs a number ans such that
(1 -o(1)) • |B ∥•∥2 (v t , r) ∩ U | ≤ ans ≤ 1.01|B ∥•∥2 (v t , cr) ∩ U | + O(n ρ+o(1) )
holds with probability at least 1 -1/ poly(n). The time to output ans is O(n ρ+o(1) d).
Proof. The proof is a combination of the data structure of (Andoni et al., 2023) and augmentation of (Beimel et al., 2022).
We start from the oblivious data structure. In the proof of Lemma 3.3 from (Andoni et al., 2023), we note that their argument is to first provide a bound on the noiseless count, and then bound the error incurred by Laplacian noise. The noiseless count ans indeed has a bound
(1 -o(1)) • |B ∥•∥2 (v t , r) ∩ U | ≤ ans ≤ |B ∥•∥2 (v t , cr) ∩ U | + O(n ρ+o(1) )
with high probability, albeit for oblivious v t . To augment it for an adaptive adversary, we apply Theorem 3.1 of (Beimel et al., 2022), which states one could use √ T copies of the data structure when the output is a real number. Regarding the output quality, only the upper bound is increased by a factor of 1 + α, if we set α = 0.01, we obtain the desired result.
Given such an adaptive data structure for approximate near neighbor counting, we could then instantiate a meta algorithm for ℓ 2 LSH under Assumption 1.2. Without loss of generality, assume U ⊆ S d-1 and all queries lie on the unit sphere. We first pick a small discretization value τ and apply the transformation
x →    ⌊x 1 /τ ⌋ • τ . . . ⌊x d /τ ⌋ • τ  
 to all points in U and all queries. We use U and v to denote transformed points. Note that these points have their entries being a multiple of τ , and the difference of the norm ∥x -x∥ 2 ≤ √ nτ , if we choose τ = n -C for a large enough constant C, then the difference is 1/ poly(n), which is negligible. The advantage of this discretization framework is that the difference of norms for points over the unit sphere could also be discretized into O(1/τ ) levels. Hence we could perform binary search over these discrete O(1/τ ) levels in O(log(1/τ )) = O(log n) steps. We will prepare O(log(1/τ )) data structures by initializing an LSH for r = (c/2) i τ where i ∈ {0, 1, . . . , log(1/τ )} for c ≥ 2. This ensures that for any r in this form, there exists a node in between r/c and c, as desired.
this section cite: ['b5', 'b4', 'b5', 'b5', 'b9', 'b5', 'b9', 'b5', 'b9']

Section: References
Ref_id:b0 Title: Deep learning with differential privacy Year: (2016)
Ref_id:b1 Title: Optimal las vegas locality sensitive data structures Year: (2017)
Ref_id:b2 Title: The white-box adversarial data stream model Year: (2022)
Ref_id:b3 Title: The bernstein mechanism: function release under differential privacy Year: (2017)
Ref_id:b4 Title: Optimal hashing-based time-space trade-offs for approximate near neighbors Year: (2017)
Ref_id:b5 Title: Differentially private approximate near neighbor counting in high dimensions Year: (2023)
Ref_id:b6 Title: Efficiently computing similarities to private datasets Year: (2024)
Ref_id:b7 Title: Differential privacy has disparate impact on model accuracy Year: (2019)
Ref_id:b8 Title: Efficient centroid-linkage clustering Year: (2024)
Ref_id:b9 Title: Dynamic algorithms against an adaptive adversary: Generic constructions and lower bounds Year: (2022)
Ref_id:b10 Title: A framework for adversarially robust streaming algorithms Year: (2020)
Ref_id:b11 Title: Cover trees for nearest neighbor Year: (2006)
Ref_id:b12 Title: Solving tall dense linear programs in nearly linear time Year: (2020)
Ref_id:b13 Title: Faster maxflow via improved dynamic spectral vertex sparsifiers Year: (2022)
Ref_id:b14 Title: Adversarial robustness of streaming algorithms through importance sampling Year: (2021)
Ref_id:b15 Title: Differentially private release and learning of threshold functions Year: (2015)
Ref_id:b16 Title: Finding frequent items in data streams Year: (2002)
Ref_id:b17 Title: Privacy-preserving logistic regression Year: (2008)
Ref_id:b18 Title: Differentially private all-pairs shortest path distances: Improved algorithms and lower bounds Year: (2023)
Ref_id:b19 Title: On adaptive distance estimation Year: (2020)
Ref_id:b20 Title: Terminal embeddings in sublinear time Year: ()
Ref_id:b21 Title: Uniform approximations for randomized hadamard transforms with applications Year: ()
Ref_id:b22 Title: Robust algorithms on adaptive inputs from bounded adversaries Year: (2023)
Ref_id:b23 Title: Nearest neighbor queries in metric spaces Year: (1997)
Ref_id:b24 Title: Numerical linear algebra in the streaming model Year: (2009)
Ref_id:b25 Title: Solving linear programs in the current matrix multiplication time Year: (2019)
Ref_id:b26 Title: Near-optimal private and scalable kclustering Year: (2022)
Ref_id:b27 Title: A one-pass distributed and private sketch for kernel sums with applications to machine learning at scale Year: (2021)
Ref_id:b28 Title: Introduction to algorithms Year: (2022)
Ref_id:b29 Title: Differentially private summaries for sparse data Year: (2012)
Ref_id:b30 Title: Locality-sensitive hashing scheme based on p-stable distributions Year: (2004)
Ref_id:b31 Title:  Year: (2008)
Ref_id:b32 Title: Near-optimal differentially private k-core decomposition Year: (2023)
Ref_id:b33 Title: The permute-and-flip mechanism is identical to report-noisy-max with exponential noise Year: (2021)
Ref_id:b34 Title: Faster matrix multiplication via asymmetric hashing Year: (2023)
Ref_id:b35 Title: The algorithmic foundations of differential privacy Year: (2014)
Ref_id:b36 Title: Calibrating noise to sensitivity in private data analysis Year: (2006)
Ref_id:b37 Title: Boosting and differential privacy Year: (2010)
Ref_id:b38 Title: Edge-weighted online bipartite matching Year: (2022)
Ref_id:b39 Title: Improved algorithms for white-box adversarial streams Year: (2023)
Ref_id:b40 Title: Fast white-box adversarial streaming without a random oracle Year: (2024)
Ref_id:b41 Title: Differentially private attention computation Year: (2023)
Ref_id:b42 Title: Fast polynomial evaluation and interpolation Year: (2013)
Ref_id:b43 Title: A strong separation for adversarially robust ℓ 0 estimation for linear sketches Year: (2024)
Ref_id:b44 Title: Differential privacy mechanisms in neural tangent kernel regression Year: (2025)
Ref_id:b45 Title: Accelerating large-scale inference with anisotropic vector quantization Year: (2020)
Ref_id:b46 Title: Bounded geometries, fractals, and low-distortion embeddings Year: (2003)
Ref_id:b47 Title: Differential privacy for functions and functional data Year: (2013)
Ref_id:b48 Title: Adversarially robust streaming algorithms via differential privacy Year: ()
Ref_id:b49 Title: On differentially private string distances Year: (2024)
Ref_id:b50 Title: Exploiting metric structure for efficient private query release Year: (2014)
Ref_id:b51 Title: Stable distributions, pseudorandom generators, embeddings, and data stream computation Year: (2006)
Ref_id:b52 Title: Approximate nearest neighbors: towards removing the curse of dimensionality Year: (1998)
Ref_id:b53 Title: Evaluating differentially private machine learning in practice Year: (2019)
Ref_id:b54 Title: A faster algorithm for solving general lps Year: (2021)
Ref_id:b55 Title: The complexity of dynamic least-squares regression Year: (2023)
Ref_id:b56 Title: Extensions of lipschitz mappings into a hilbert space Year: (1984)
Ref_id:b57 Title: On the adversarial robustness of locality-sensitive hashing in hamming space Year: (2024)
Ref_id:b58 Title: Finding nearest neighbors in growth-restricted metrics Year: (2002)
Ref_id:b59 Title: An optimal algorithm for on-line bipartite matching Year: (1990)
Ref_id:b60 Title: Securing bloom filters with differential privacy Year: (2025)
Ref_id:b61 Title: Matrix factorization techniques for recommender systems Year: (2009)
Ref_id:b62 Title: The black-box complexity of nearest neighbor search Year: ()
Ref_id:b63 Title: Navigating nets: Simple algorithms for proximity search Year: (2004)
Ref_id:b64 Title: Efficient search for approximate nearest neighbor in high dimensional spaces Year: (1998)
Ref_id:b65 Title: Faster rectangular matrix multiplication by combination loss analysis Year: (2024)
Ref_id:b66 Title: Solving empirical risk minimization in the current matrix multiplication time Year: (2019)
Ref_id:b67 Title: Differentially private partial set cover with applications to facility location Year: (2023)
Ref_id:b68 Title: Computing epidemic metrics with edge differential privacy Year: (2024)
Ref_id:b69 Title: Fast john ellipsoid computation with differential privacy optimization Year: (2025)
Ref_id:b70 Title: Differential privacy of cross-attention with provable guarantee Year: (2024)
Ref_id:b71 Title: Differentially private kernel density estimation Year: (2024)
Ref_id:b72 Title: Faster ridge regression via the subsampled randomized hadamard transform Year: (2013)
Ref_id:b73 Title: Scalable differential privacy with sparse network finetuning Year: (2021)
Ref_id:b74 Title: Private text generation by seeding large language model prompts Year: (2025)
Ref_id:b75 Title: Locality-sensitive hashing without false negatives Year: (2016)
Ref_id:b76 Title: Fast regression with an ℓ ∞ guarantee Year: (2017)
Ref_id:b77 Title: An online and unified algorithm for projection matrix vector multiplication with application to empirical risk minimization Year: (2023)
Ref_id:b78 Title: Collaborative filtering beyond the user-item matrix: A survey of the state of the art and future challenges Year: (2014)
Ref_id:b79 Title: Oblivious sketching-based central path method for solving linear programming problems Year: ()
Ref_id:b80 Title: Speeding up sparsification using inner product search data structures Year: (2022)
Ref_id:b81 Title: Sketching for first order method: efficient algorithm for low-bandwidth channel and vulnerability Year: (2023)
Ref_id:b82 Title: Sketching meets differential privacy: fast algorithm for dynamic kronecker projection maintenance Year: (2023)
Ref_id:b83 Title: A nearly-optimal bound for fast regression with ℓ ∞ guarantee Year: (2023)
Ref_id:b84 Title: A survey of collaborative filtering techniques Year: (2009)
Ref_id:b85 Title: Dpauc: differentially private auc computation in federated learning Year: (2023)
Ref_id:b86 Title: Announcing scann: Efficient vector similarity search Year: (2020)
Ref_id:b87 Title: On efficient retrieval of top similarity vectors Year: (2019)
Ref_id:b88 Title: Bayesian differential privacy for machine learning Year: (2020)
Ref_id:b89 Title: Fast private kernel density estimation via locality sensitive quantization Year: (2023)
Ref_id:b90 Title: Differentially private data releasing for smooth queries Year: (2016)
Ref_id:b91 Title: Syntf: Synthetic and differentially private term frequency vectors for privacypreserving text mining Year: (2018)
Ref_id:b92 Title: Optimal las vegas approximate near neighbors in ℓ p Year: (2022)
Ref_id:b93 Title: Probabilistic inference and differential privacy Year: (2010)
Ref_id:b94 Title: Multiplying matrices faster than coppersmith-winograd Year: (2012)
Ref_id:b95 Title: New bounds for matrix multiplication: from alpha to omega Year: (2024)
Ref_id:b96 Title: Sketching as a tool for numerical linear algebra Year: (2014)
Ref_id:b97 Title: Tight bounds for adversarially robust streams and sliding windows via difference estimators Year: (2021)
Ref_id:b98 Title: Adversarially robust densesparse tradeoffs via heavy-hitters Year: (2024)
Ref_id:b99 Title: Differentially private finetuning of language models Year: (2022)
Ref_id:b100 Title: Differential privacy for text analytics via natural text sanitization Year: (2021)
Ref_id:b101 Title: Privateknn: Practical differential privacy for computer vision Year: (2020)
