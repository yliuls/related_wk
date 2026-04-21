Title: Discrepancy Minimization in Input-Sparsity Time
Abstract: A recent work by [Larsen, SODA 2023]  introduced a faster combinatorial alternative to Bansal's SDP algorithm for finding a coloring x ∈ {-1, 1} n that approximately minimizes the discrepancy disc(A, x) := ∥Ax∥ ∞ of a realvalued m × n matrix A. Larsen's algorithm runs in O(mn 2 ) time compared to Bansal's O(mn 4.5 )time algorithm, with a slightly weaker logarithmic approximation ratio in terms of the hereditary discrepancy of A [Bansal, FOCS 2010]. We present a combinatorial O(nnz(A) + n 3 )-time algorithm with the same approximation guarantee as Larsen's, optimal for tall matrices where m = poly(n). Using a more intricate analysis and fast matrix multiplication, we further achieve a runtime of O(nnz(A) + n 2.53 ), breaking the cubic barrier for square matrices and surpassing the limitations of linear-programming approaches [Eldan and Singh, RS&A 2018]. Our algorithm relies on two key ideas: (i) a new sketching technique for finding a projection matrix with a short ℓ 2 -basis using implicit leverage-score sampling, and (ii) a data structure for efficiently implementing the iterative Edge-Walk partial-coloring algorithm [Lovett and Meka, SICOMP 2015], and using an alternative analysis to enable "lazy" batch updates with low-rank corrections. Our results nearly close the computational gap between real-valued and binary matrices, for which inputsparsity time coloring was recently obtained by [Jain, Sah and Sawhney, SODA 2023].

Section: Introduction
Discrepancy theory is a fundamental subject in combinatorics and theoretical computer science, studying how to color elements of a finite set-system S 1 , . . . , S m ⊆ {1, . . . , n} with two colors (e.g., red and blue) to minimize the maximum imbalance in color distribution across all sets. It finds diverse applications in fields such as computational geometry (Matousek, 1999b;De Berg, 2000), probabilistic algorithms (Spencer, 1985;Chazelle, 2000), machine learning (Vapnik & Chervonenkis, 1971;Talagrand, 1995;Karnin & Liberty, 2019;Bechavod et al., 2022;Han et al., 2025), differential privacy (Muthukrishnan & Nikolov, 2012;Nikolov et al., 2013), and optimization (Beck & Fiala, 1981;Bansal, 2010;2012;Nikolov, 2015).
Definition 1.1 (Discrepancy). The discrepancy of a real matrix A ∈ R m×n with respect to a "coloring" vector x ∈ {±1} n is defined as
disc(A, x) := ∥Ax∥ ∞ = max j∈[m] |(Ax) j |.
The discrepancy of a real matrix A ∈ R m×n is defined as disc(A) := min x∈{±1} n disc (A, x).
This is a natural generalization of the classic combinatorial notion of discrepancy of set systems, corresponding to binary matrices A ∈ {0, 1} m×n where rows represent (the indicator vector of) m sets over a ground set of [n] elements, and the goal is to find a coloring x ∈ {±1} n which is as "balanced" as possible simultaneously on all sets.
Most of the history of this problem was focused on the existential question of understanding the minimum possible discrepancy achievable for various classes of matrices. For general set systems of size n (arbitrary n × n binary matrices A), the classic result of Spencer (1985) states that disc(A) ≤ 6 √ n, which is asymptotically better than random coloring (Θ( √ n log n). More recent works have focused on restricted matrix families, such as sparse matrices (Banaszczyk, 1998;Beck & Fiala, 1981), showing that it is possible to achieve the o( √ n) discrepancy in these restricted cases. For example, the minimum-discrepancy coloring of k-sparse binary matrices (sparse set systems) turns out to have discrepancy at most O( √ k log n) (Banaszczyk, 1998).
All of these results, however, do not provide polynomial time algorithms since they are non-constructive-they only argue about the existence of low-discrepancy colorings, which prevents their use in algorithmic applications that rely on low-discrepancy (partial) coloring, such as binpacking problems (Rothvoß, 2013;Hoberg & Rothvoss, 2017). 1 Indeed, the question of efficiently finding a lowdiscrepancy coloring, i.e., computing disc(A), was less understood until recently and is more nuanced: Charikar et al. (2011) showed that it is NP-hard to distinguish whether a matrix has disc(A) = 0 or disc(A) = Ω( √ n), suggesting that (ω(1)) approximation is inevitable to achieve fast runtimes. The celebrated work of Bansal (2010) gave the first polynomial-time algorithm which achieves an additive O( √ n)-approximation to the optimal coloring of general n × n matrices, matching Spencer's non-constructive result. Bansal's algorithm has approximation guarantees in terms of the hereditary discrepancy (Lovász et al., 1986).
Definition 1.2 (Hereditary discrepancy). Given a matrix A ∈ R m×n , its hereditary discrepancy is defined as
herdisc(A) := max B∈A disc(B),
where A is the set of all matrices obtained from A by deleting some columns from A. Bansal (2010) gave an SDP-based algorithm that finds a coloring x satisfying disc(A, x) = O(log(mn) • herdisc(A)) for any real matrix A, in time O(mn 4.5 ) assuming state-ofthe-art SDP solvers (Jiang et al., 2020b;Huang et al., 2022b).
In other words, if all submatrices of A have low-discrepancy colorings, then it is in fact possible (yet still quite expensive) to find an almost-matching overall coloring.
Building on Bansal's work, Lovett & Meka (2015) designed a simpler algorithm for set-systems (binary matrices A), running in O((n + m) 3 ) time. The main new idea of their algorithm, which is also central to our work, was a subroutine for repeatedly finding a partial-coloring via random-walks (a.k.a EDGE-WALK), which in every iteration "rounds" a constant-fraction of the coordinates of a fractional-coloring to an integral one in {-1, 1} (more on this in the next section). Followup works by Rothvoss (2017) and Eldan & Singh (2018) extended these ideas to the real-valued case, and developed faster convex optimization frameworks for obtaining low-discrepancy coloring, the latter requiring O(log n) linear programs instead of a semidefinite program (Bansal, 2010), assuming a value oracle to herdisc(A). 2 In this model, Eldan & Singh (2018) yields an O * (max{mn + n 3 , (m + n) w }) time approximate discrepancy algorithm via state-of-art (tall) LP solvers (Brand et al., 2020). This line of work, however, has a fundamental setback for achieving input-sparsity time, which is a major open problem for (high-accuracy) LP solvers (Bubeck et al., 2018). Sparse matrices are often the realistic case for discrepancy problems, and have been widely studied in this context as discussed earlier in the introduction. Another drawback of convex-optimization based algorithms is that they are far from being practical due to the complicated nature of fast LP solvers.
Interestingly, in the binary (set-system) case, these limitations have been very recently overcome in the breakthrough work of Jain et al. (2023), who gave an O(n+nnz(A))-time coloring algorithm for binary matrices A ∈ {-1, 1} m×n , with near optimal (O( n log(m/n + 2))) discrepancy.
While their approach, too, was based on convex optimization, their main observation was that an approximate LP solver, using first-order methods, in fact suffices for a logarithmic approximation. Unfortunately, the input-sparsity running time of the algorithm in Jain et al. (2023) does not extend to real-valued matrices, as their LP is based on a "heavy-light" decomposition of the rows of binary matrices based on their support size. More precisely, generalizing the argument of Jain et al. (2023) to matrices with entries in range, say, [-R, R], guarantees that a uniformly random vector would only have discrepancy O(poly(R) • √ n) on the "heavy" rows, and this term would also govern the approximation ratio achieved by their algorithm.
By contrast, a concurrent result of Larsen (2023) gave a purely combinatorial (randomized) algorithm, which is not as fast, but handles general real matrices and makes a step toward practical coloring algorithms. Larsen's algorithm improves Bansal's SDP from O(mn 4.5 ) to O(mn 2 + n 3 ) time, at the price of a slightly weaker approximation guarantee: For any A ∈ R m×n , Larsen (2023) finds a coloring x ∈ {-1, +1} n such that disc(A, x) = O(herdisc(A) • log n • log 1.5 m). Recent work by Jambulapati et al. (2024) introduced a new framework for efficient computing of the discrepancy, with which they successfully recovered Spencer's result in time O(nnz(A) • log 5 (n)) for matrix A with entries restricted within 1.
The recent exciting developments naturally raise the following question:
Is it possible to achieve input-sparsity time for discrepancy minimization with general (real-valued) matrices?
In fact, this was one of the main open questions raised in 2018 workshop on Discrepancy Theory and Integer Programming (Dadush, 2018).
Table 1. Progress on approximate discrepancy-minimization algorithms of real-valued m × n matrices (m ≥ n). For simplicity, we ignore n o(1) and poly(log n) factors in the table. " (Eldan & Singh, 2018)*" refers to a (black-box) combination of our Theorem 1.5 with (Eldan & Singh, 2018) and using state-of-art LP solvers for square and tall matrices (Lee & Sidford, 2014;Brand et al., 2020;Jiang et al., 2021).
this section cite: ['b41', 'b112', 'b31', 'b119', 'b114', 'b16', 'b53', 'b89', 'b93', 'b17', 'b9', 'b10', 'b92', 'b112', 'b8', 'b17', 'b8', 'b55', 'b30', 'b9', 'b83', 'b9', 'b99', 'b44', 'b9', 'b44', 'b21', 'b25', 'b61', 'b61', 'b61', 'b74', 'b62', 'b38', 'b44', 'b44', 'b77', 'b21']

Section: References Methods
Running Time Bansal (2010) SDP (Jiang et al., 2020c;Huang et al., 2022b) mn 4.5 Eldan & Singh (2018)* (Step 1: Theorem 1.5) + (Step 2: LP (Jiang et al., 2021)) m ω + m 2+1/18 Eldan & Singh (2018)* (Step 1: Theorem 1.5) + (Step 2: LP (Brand et al., 2020)) mn + n 3 Eldan & Singh (2018)* (Step 1: Theorem 1.5) + (Step 2: LP (Lee & Sidford, 2014)) nnz(A) √ n + n 2.5 Larsen (2023) Combinatorial mn 2 + n 3 Ours (Theorem 1.3) Combinatorial nnz(A) + n 3 Ours (Theorem 1.3) (Step 1: Theorem 1.5) + (Step 2: Lemma E.12) nnz(A) + n 2.53
this section cite: ['b9', 'b130']

Section: Our Results
We answer this question in the affirmative. To this end, we develop an algorithm that achieves a near-optimal runtime for tall matrices with m = poly(n) and a subcubic runtime for square matrices, while maintaining the same approximation guarantees as Larsen's algorithm, up to constant factors. We state our main result in the following theorem.
Theorem 1.3 (Main result, informal version of Theorem D.1 and Theorem D.2). For any parameter a ∈ [0, 1], there is a randomized algorithm that, given a real matrix A ∈ R m×n , finds a coloring x ∈ {-1, +1} n such that
disc(A, x) = O(herdisc(A) • log n • log 1.5 m). Moreover, it runs in time O(nnz(A) + n ω + n 2+a + n 1+ω(1,1,a)-a ).
Here, ω(a, b, c) denotes the time for multiplying an n a × n b matrix with an n b × n c matrix, and ω := ω(1, 1, 1) denotes the exponent of fast matrix multiplication (FMM).
Remark 1.4. With the current value of ω ≈ 2.371, according to Table 1 in Alman et al. (2025), we choose the parameter a ≈ 0.53 to balance the terms n 2+a and n 1+ω(1,1,a)-a . Consequently, the running time in Theorem 1.3 simplifies to O(nnz(A) + n 2.53 ). Without FMM, our algorithm runs in O(nnz(A) + n 3 ) time and is purely combinatorial.
Let α denote the dual exponent of matrix multiplication, i.e., 2 = ω(1, 1, α), and let a ∈ [0, 1] be the tunable parameter of Theorem 1.3. Currently, α ≈ 0.32 (Williams et al., 2024). Note that the running time of our algorithm (when using FMM) has a tradeoff between the additive terms n 2+a and n 1+ω(1,1,a)-a , so it is never better than n 2.5 , as it is never beneficial to set a < 1/2. This tradeoff also means that our runtime is barely sensitive to future improvements in the value of the dual exponent (even α ≈ 1 would improve the exponent of the additive term by merely 0.03). Curiously, a similar phenomenon occurs in recent FMM-based LP solvers (Cohen et al., 2019;Jiang et al., 2021), dynamic attention (Brand et al., 2024) and weight pruning (Li et al., 2024) in large language models.
A central technical component of Theorem 1.3 is the following theorem, which allows us to quickly find a "hereditary projection" matrix (i.e., a subspace such that the projection of a constant fraction of the rows of A to its orthogonal complement has small ℓ 2 -norm), and is of independent interest in randomized linear algebra. We state it as follows.
Theorem 1.5 (Fast hereditary projection, informal version of Theorem C.2 and Theorem C.3). Let A ∈ R m×n with m ≥ n, and let d = n/4. There is a randomized algorithm that outputs a d × n matrix V such that with probability 1δ, the followings hold simultaneously:
• For all l ∈ [d], we have ∥V l, * ∥ 2 = 1,
• It holds that max j∈[m] ∥(A(I -V ⊤ V )) j, * ∥ 2 = O(herdisc(A) log(m/n)),
where V l, * denotes the l-th row of V for any l ∈ [d].
Moreover, it runs in time O(nnz(A) + n ω ), where the Onotation hides a log(m/δ) factor.
Theorem 1.5 directly improves Theorem 3 in Larsen (2023) which runs in O(mn 2 ) time without using FMM, and in O(mn ω-1 ) time using FMM. In either case, Larsen's algorithm pays at least mn ω-1 time, even when A is sparse (see more discussion in Section 2.1). Theorem 1.5 is in some sense best possible: Reading the input matrix A requires O(nnz(A)) time, and explicitly computing the projection matrix P = V ⊤ V in Theorem 1.5 requires n ω time. We note that, in the case of binary matrices, the projection-free algorithm of Jain et al. (2023) avoids this bottleneck (and hence the n ω term), but for real-valued matrices, all known discrepancy algorithms involve projections (Bansal, 2010;Lovett & Meka, 2015;Larsen, 2023;Rothvoss, 2017).
1.2. Related Work Algorithmic discrepancy theory Constructive discrepancy theory has become a pivotal area of research, focusing on efficiently finding low-discrepancy solutions. Bansal's seminal work introduced a semidefinite programming (SDP)-based algorithm that achieves additive O( √ n)
discrepancy (Bansal, 2010), matching the nonconstructive bounds but incurring a significant computational cost of O(mn 4.5 ). Subsequent contributions, such as Lovett & Meka (2015) and Eldan & Singh (2018), developed faster algorithms leveraging random walks and convex optimization techniques, respectively. Recent advances include Larsen's combinatorial approach (Larsen, 2023), which attains a runtime of O(mn 2 +nfoot_6 ), and the near-input-sparsity algorithms for binary matrices proposed by Jain et al. (2023). Furthermore, Jambulapati et al. (2024) introduced an efficient framework for computing discrepancy, recovering Spencer's result in O(nnz(A) • log 5 (n)) time for matrices A with entries bounded by 1. These innovations have substantially narrowed the computational gap between theoretical results and practical applications, advancing the field's capabilities within polynomial time. For further details, we refer readers to related works (Cohen, 2016b;Bansal et al., 2018;2019;Dadush et al., 2018;Alweiss et al., 2021;Bansal et al., 2020;2022;Pesenti & Vladu, 2023;Jambulapati et al., 2024).
Most recently, Han et al. (2025) uses the discrepancy theory to approximate the attention computation in the streaming model.
Sketching and leverage-score sampling Sketching is a versatile technique employed across numerous fundamental problems, including linear programming (Jiang et al., 2021;Brand et al., 2020;Song & Yu, 2021;Liu et al., 2023), empirical risk minimization (Lee et al., 2019;Qin et al., 2023;Gu et al., 2025), and semidefinite programming (Jiang et al., 2020a;Huang et al., 2022a;Song et al., 2023b). It is particularly prominent in randomized linear algebra, where it has been applied to a wide range of tasks (Clarkson & Woodruff, 2013;Nelson & Nguyên, 2013;Razenshteyn et al., 2016;Boutsidis et al., 2016;Song et al., 2017;Xiao et al., 2018;Song et al., 2019b;Lee et al., 2019;Jiang et al., 2021;Song & Yu, 2021;Brand et al., 2021;Song et al., 2022a;Hu et al., 2022;Gu & Song, 2022). Sketching frequently serves as an effective tool for oblivious dimension reduction (Clarkson & Woodruff, 2013;Nelson & Nguyên, 2013). The use of sampling matrices to enhance computational efficiency is a well-established approach in numerical linear algebra (see Clarkson & Woodruff (2013
this section cite: ['b3', 'b121', 'b37', 'b23', 'b80', 'b61', 'b9', 'b84', 'b74', 'b99', 'b9', 'b84', 'b44', 'b74', 'b61', 'b62', 'b11', 'b39', 'b5', 'b13', 'b94', 'b62', 'b53', 'b21', 'b103', 'b82', 'b95', 'b52', 'b34', 'b90', 'b96', 'b19', 'b104', 'b123', 'b103', 'b22', 'b56', 'b49', 'b34', 'b90']

Section: Technical Overview
In this section, we give the overview of the techniques used to prove the main results, and the formal proofs are in Appendix. In Section 2.1, we first give an overview and discuss the barriers of Larsen's algorithm. In Section 2.2, we introduce our techniques used to improve the Larsen's algorithm and overcome the barriers.
this section cite: []

Section: Overview and Barriers of Larsen's Algorithm
Larsen's algorithm (Larsen, 2023) is a clever reimplementation of the iterated partial-coloring subroutine of Lovett & Meka (2015): In each iteration, with constant probability, this subroutine "rounds" at least half of the coordinates of a fractional coloring x ∈ R n to {-1, 1}.
As in Lovett & Meka (2015), this is done by performing a random-walk in the orthogonal complement subspace V ⊥ spanned by a set of rows from A (a.k.a "Edge-Walk" (Lovett & Meka, 2015)). The key idea in Larsen (2023) lies in a clever choice of V : Using a connection between the eigenvalues of A ⊤ A and herdisc(A) (Larsen, 2017), Larsen shows that there is a subspace V spanned by ≤ n/4 rows of A, so that the projection of A onto the complement V ⊥ has small ℓ 2 norm, i.e., every row of
A(I -V ⊤ V ) has norm less than O(herdisc(A) log(m/n)).
Larsen shows that computing this projection operator (henceforth B ⊤ B) can be done in O(mn 2 ) time combinatorially, or in T mat (m, n, n) = O(mn ω-1 ) time 3 using FMM.
The second part (PARTIALCOLORING) of Larsen's algorithm is to repeatedly apply the above subroutine (PROJECTTOSMALLROWS) to implement the Edge-Walk of (Lovett & Meka, 2015): Starting from a partial coloring x ∈ R n , the algorithm first generates a subspace V 1 by calling the afformentioned PROJECTTOSMALLROWS subroutine. Then it samples a fresh random Gaussian vector g t in each iteration t ∈ [N ] = O(n), and projects it to obtain g t = (I -V ⊤ t V t )g t . The algorithm then decides whether or not to update V ; More precisely, it maintains a vector u t , and gradually updates it to account for large entries of x + u t+1 that have reached ≈ 1 in absolute value (as in Lovett & Meka (2015)). When a coordinate i ∈ [n] reaches this threshold at some iteration, the corresponding unit vector e i is added to V t and V t is updated to V t+1 . This update is necessary to ensure that in future iterations, no amount will be added to the i-th entry of u t . At the end of the loop, the algorithm outputs a vector x new = x + u N +1 with property that for each row a i of A, the difference between ⟨a i , x⟩ and ⟨a i , x new ⟩ is less than O(herdisc(A) • log 1.foot_8 (m)). Since V t is constantly changing thoughout iterations, each iteration requires an online Matrix-Vector multiplication (I -V ⊤ t V t )g t . Hence, since there are N = O(n) iterations, O(mn 2 + n 3 ) time is required to implement this part, even if fast-matrix multiplication is allowed -Indeed, the Online Matrix-Vector conjecture (Henzinger et al., 2015) postulates that mn 2 is essentially best possible for such online problem. Beating this barrier for the PARTIALCOLORING subroutine therefore requires to somehow avoid this online problem, as we discuss in the next subsection.
Below we summarize the computational bottlenecks in Larsen's algorithm, and then explain the new ideas required to overcome them and achieve the claimed overall runtime of Theorem 1.3.
Implementing the first part of Larsen's algorithm (PROJECTTOSMALLROWS) incurs the following computational bottlenecks, which we overcome in Theorem 1.5:
• Barrier 1. Computing the projection matrix B t = A(I -V ⊤ t V t ) ∈ R m×n explicitly (exactly) already takes T mat (m, n, n) time. 4 • Barrier 2. Computing the jth-row's norm ∥e ⊤ j B t ∥ 2 requires T mat (m, n, n) time.
• Barrier 3. Computing B ⊤ t B t ∈ R n×n requires multiplying an n × m with a m × n matrix, which also takes T mat (n, m, n) time.
Implementing the second part of Larsen's algorithm (PARTIALCOLORING) incurs the following two (main) computational bottlenecks:
• Barrier 4. The coloring algorithm requires computing η = max j∈[m] ∥e ⊤ j A(I -V ⊤ V )∥ 2 , which takes T mat (m, n, n) time.
• Barrier 5. In each of the N = O(n) iterations, the algorithm first chooses a Gaussian vector g ∼ N (0, I d ) and projects it to the orthogonal span of V t , i.e., g Vt = (I -V ⊤ t V t )g. Next, it finds a rescaling factor g by solving a single variable maximization problem. 5  Finally, the algorithm checks whether |⟨a j , v+g⟩| ≥ τ , |⟨a j , v⟩| < τ for each j ∈ [m]. The overall runtime is therefore mn 2 .
As mentioned above, the last step is conceptually challenging, as it must be done adaptively (V t is being updated throughout iterations), and cannot be batched via FMM (assuming the OMv Conjecture (Henzinger et al., 2015)) 6 . We circumvent this step by slightly modifying the algorithm and analysis of Larsen, and using the fact that, while the subspace V t is changing throughout iterations, the projected random Gaussian vectors g t themselves are independent -We show this enables to batch the projections and then perform low-rank corrections as needed in the last step. We now turn to explain our technical approach and the main ideas for overcoming these computational bottlenecks.
this section cite: ['b74', 'b84', 'b84', 'b84', 'b74', 'b73', 'b84', 'b84', 'b54', 'b54']

Section: Our Techniques
A natural approach to accelerate the first part of the above algorithm is to use linear sketching techniques (Clarkson & Woodruff, 2013) as they enable working with much smaller matrices in the aforementioned steps. However, sketching techniques naturally introduce (spectral) error to the algorithm, which is exacerbated in iterative algorithms. Indeed, a nontrivial challenge is showing that Larsen's algorithm can be made robust to noise, which requires to modify his analysis in several parts of the algorithm. The main technical obstacle (not present in vanilla "sketch-and-solve" problems such as linear-regression, low-rank, tensor, inverse problems (Clarkson & Woodruff, 2013;Nelson & Nguyên, 2013;Razenshteyn et al., 2016;Song et al., 2017;2019a;Lee et al., 2019;Jiang et al., 2021)) is that we can never afford to explicitly store the projection matrix B t , as even writing it would already require O(mn) time. This constraint makes it more challenging to apply non-oblivious sketching tools, in particular approximate leverage-score sampling (LSS, (Spielman & Srivastava, 2011)), which are key to our algorithm.
Breaking the n 3 runtime of Larsen's partial-coloring algorithm (which is important for near-square matrices m ≈ n) requires a different idea, as this bottleneck stems from the Online Matrix-Vector Conjecture (Henzinger et al., 2015). To circumvent this bottleneck, we modify the analysis and implementation of the Edge-Walk subroutine (dropping certain verification steps via concentration arguments), and then design a "guess-and-correct" data structure (inspired by "lookahead" algorithms (Brand et al., 2019)) that batches the prescribed gaussian projections, with low-rank amortizes corrections. We now turn to formalize these three main ideas.
this section cite: ['b34', 'b34', 'b90', 'b96', 'b104', 'b113', 'b54', 'b20']

Section: ROBUST ANALYSIS OF LARSEN'S ALGORITHM
Approximate norm-estimation suffices. The original algorithm (Larsen, 2023) explicitly calculates the exact norm of each row of B t = A(I -V ⊤ t V t ). Since the JL lemma guarantees ∥e ⊤ j B t ∥ 2 ∈ (1 ± ϵ 0 )∥e ⊤ j B t ∥ 2 , ∀j ∈ [m] except with polynomially-small probability δ 0 (as the sketch dimension is logarithmic in 1/δ 0 ), it is not hard to show that, even though our approximation could potentially miss some "heavy" rows, this has a minor effect on the correctness of the algorithm: the rows our algorithm selects will be larger than (1ϵ 0 ) times a pre-specified threshold, and the ones that are not chosen will be smaller than (1 + ϵ 0 ) times the threshold. This allows to set ϵ 0 = Ω(1).
this section cite: ['b74']

Section: Approximate SVD suffices.
Recall that in the PROJECT-TOSMALLROWS algorithm, expanding the subspace V iteratively, requires to compute SVD(B ⊤ t B t ). Since exact SVD is too costly for us, we wish to maintain an approximate SVD instead. Even though this may result in substantially different eigen-spectrum, we observe that a spectralapproximation of SVD(B ⊤ t B t ) suffices for this subroutine, since we only need eigenvectors to be: (i) orthogonal to the row space of A(I -V ⊤ t V t ); (ii) orthonormal to each other. Our correctness analysis shows that an ϵ B = Θ(1) spectral approximation will preserve (up to constant factor) the row-norm guarantee of PROJECTTOSMALLROWS.
this section cite: []

Section: OVERCOMING THE BARRIERS
Speeding-up the "hereditary-projection" step In the PROJECTTOSMALLROWS subroutine in Larsen (2023), the matrix B t is used for (i) detecting rows with largest norms; (ii) extracting the largest rows to generate a matrix B; and (iii) computing the eigenvectors of B ⊤ B. To optimize this process and reduce computational overhead, we avoid the explicit representation of matrix B t by substituting it with a product of appropriately-chosen sketching matrices. Given the aforementioned robustness-guarantees, Barriers in the first step can be bypassed straight-forwardly by using a JL sketch. Specifically, we utilize a random matrix R ∈ R n×ϵ -2 0 log(1/δ0) to obtain the compressed matrix B t := A(I -V ⊤ t V t )R. Similarly, detecting rows with large ℓ 2 norm can be done in the sketched subspace since we can preserve norms up to constant by choosing ϵ 0 = Θ(1) and δ 0 = δ/ poly(m, n), which reduces the time for querying row-norms from O(mn 2 ) to O(nnz(A) + n ω ).
this section cite: ['b74']

Section: Implicit leverage-score sampling
To address the hardness of computation of B ⊤ t B t (overcoming Barrier 3), we use robust analysis to ensure that approximating the Top-k SVD of B ⊤ t B t (where k ≈ n/ log(m/n)) using leveragescore sampling (Drineas et al., 2012;Clarkson & Woodruff, 2013;Nelson & Nguyên, 2013) preserves algorithm correctness. However, we lack explicit access to the input matrix B t , so we must perform implicit leverage-score sampling w.r.t B t in ∼ nnz(A) time. We propose IMPLICITLEVER-AGESCORE (Algorithm 9), which takes A and an orthonormal basis V and generates a sparse embedding matrix S 1
B A I V ⊤ V R ← × ( -) × m O(ϵ -2 0 ) m n n n O(ϵ -2 0 )
The original matrix B by Larsen
B ∈ R m× O(ϵ -2 0 ) is our sketched matrix. A ∈ R m×n is the original data matrix. I ∈ R n×n is an n × n unit matrix. V ⊤ V ∈ R n×n is the projection matrix onto the row span of V . And R ∈ R n× O(ϵ -2 0 )
is our JL sketching matrix. After sketched, we are able to fast query row norms of B, which are close to row norm of B with an accuracy ϵ0. We select the rows from B using this approximated norms. For the details of the selection operation, see Figure 2 and Figure 3.
to produce a compressed matrix M , whose QR factorization gives R. Using another sparse embedding matrix S 2 , we calculate the compressed matrix N , which is used to compute the approximate leverage scores. This allows us to carry out the LSS lemma without computing B t . With IMPLICITLEVERAGESCORE, we can generate a diagonal sampling matrix D in O(nnz(A) + n ω ) time. Using this subroutine to calculate B t = D t B t approximates the SVD of B ⊤ t B t . Finally, we prove that adding the eigenvectors of B ⊤ t B t instead of B ⊤ t B t to V will still satisfy the required "oversampling" prerequisite for each row in B.
this section cite: ['b43', 'b34', 'b90']

Section: BEATING THE CUBIC BARRIER
Where does the n 3 barrier arise from? Recall that, in order to simulate the Edge-Walk process, the PARTIALCOL-ORING algorithm generates, in each iteration, a Gaussian vector g and projects it to Span(V ). This requires a generic Mat-Vec product
(I -V ⊤ t V t )g, which takes n 2 time. Since V t is dynamically changing throughout iterates (depend- ing on whether |x i + v i + g V,i | = 1 and |x i + v i | < 1 is satisfied or not for each i),
and there are N = O(n) iterations, the OMv Conjecture (Henzinger et al., 2015) generally implies an Ω(n 3 ) runtime for implementing this iterative loop (Note that each e i can be added to V at most once, O(n) iterations indeed suffice). Nonetheless, we show how to re-implement PARTIALCOLORING using a "lookahead" (guess-and-correct) data structure, which combines FMM with low-rank corrections. We now describe the main ingredients of this data structure.
B t D t A I V ⊤ t V t ← × × ( -) m n m m n n n n ∥D t ∥ 0 = m t (a) Larsen's selection matrix B t D t D t A I V ⊤ t V t ← × × × ( -) m n m m m n n n n ∥D t ∥ 0 = m t ∥ D t ∥ 0 = O(ϵ -2 B n) (b) Our subsample matrix
this section cite: ['b54']

Section: Precomputing Gaussian projections.
To overcome the ∼ n 2 running time for computing the Gaussian projections for g, we add a preprocessing phase (INIT subroutine) to the PARTIALCOLORING iterative algorithm, which generatesin advance-a Gaussian matrix G ∈ R n×N and stores the projection of every column of G to the row space of V , denoted {g Vt } g∈G . We also design a QUERY procedure, which outputs any desired output vector g = (I -V ⊤ t V t )g on-demand. Since we do not know apriori if the subspace V t will change (and which coordinates e i will be added to V t ), we use a "Guess-and-Correct" approach: We guess the batch Gaussian projections, and then in iteration t, we perform a low-rank corrections via our update and query procedure in data structure. This idea is elaborated in more detail below.
Lazy updates for the past rank-1 sequence. Updating all the projections g stored in the data structure would result in prohibitively expensive runtime. Instead, we use the idea of lazy updates: We divide the columns of G into different batches, each batch having the size K. We also use a counter τ u to denote which batch is being used currently, and initialize two counters k q and k u to record the times QUERY and UPDATE were called, respectively. Every time we call QUERY or UPDATE, we increment the counter by 1. When either k q or k u reaches the threshold K, we RESTART the process, and "accumulate" the current updates as well as some clean-up operations for future iterations, adding up the vectors which are not present yet, and computes a new batch of g's for future use). This procedure runs in time O(T mat (n, K, n)), and contributes T mat (n, K, n) time to the RESTART procedure. Now, Recall that at the beginning of the PARTIALCOLOR-ING algorithm, when we initialize the data structure, we compute the first K projections g 1 , . . . , g k . Then we enter the iteration. Recall in INIT, we generate a matrix P which is defined to be V ⊤ V for the input matrix V . And we precompute a batch of projections onto the row space of V . ( G = P • G * ,S , where S = [K] at the beginning.) Besides, if there is some rows are added to V during the running of the algorithm, we first find its factor that is vertical to the row space of V . Then we rescale this vector to unit and name it w. We maintain at most K w's. Then through the running of the algorithm, when we call the QUERY, we just simply select the corresponding row in G, denoted as g, and output the vector
g -g - ku i=1 w i w ⊤ i g.
Recall that the number of g we pre-computed and the number of w we maintained are both limited to be less than K, when one of them reach the limit, we call the RESTART procedure. When this condition happens, QUERY and UPDATE will take T mat (K, n, n). Thus by applying this lazy update idea, we finally reach the subcubic running time. The final runtime of PARTIALCOLORING using our data structure is therefore a tradeoff based on the choice of batch size K. Denoting . . . t-1 Vt-1)R, where R is an JL matrix. Then we select the rows based on the approximated norms. Thus we significantly reduce the time cost of norm computation. We show that, under our setting, the norm of the rows not selected will have another guarantee, that is, (1 + ϵ0) • C0 • T • herdisc(A). This constant loss will still make our algorithm correct. (Since our row norm computation is approximated, there is a constant loss effecting the selecting operation. To demonstrate this, we use the darker red to demonstrate the real largest rows without being selected.) K = n a , we get a total runtime of
m t-1 largest rows . . . m t largest rows Compute B t = A(I -V ⊤ t-1 V t-1 ) Compute B t-1 = A(I -V ⊤ t-2 V t-2 ) Compute B t+1 = A(I -V ⊤ t V t ) B t-1 B t • • • • • • B t-1 B t (a) Larsen's matrix row selection . . . m t-1 approximated largest rows . . . m t approximated largest rows Compute B t = A(I -V ⊤ t-2 V t-2 )R Compute B t-1 = A(I -V ⊤ t-2 V t-2 )R Compute B t-1 = A(I -V ⊤ t-2 V t-2 )R B t-1 B t • • • • • • B t-1 = A(I -V ⊤ t-2 V t-2 ) B t = A(I -V ⊤ t-1 V t-1 ) (b) Our matrix row selection
O(nnz(A) + n ω + n 2+a + n 1+ω(1,1,a)-a ).
For the current upper bound on the fast rectangular matrix multiplication function ω(•, •, •) (Alman et al., 2025), setting a = 0.53 yields an optimal overall runtime time of O(nnz(A) + n 2.53 ). We note that even an ideal value of ω would not improve this result by much, as it would merely enable setting a = 0.5 which in turn would translate into an n 2.5 time algorithm, and this is the limit of our approach.
Faster Iterative Coloring Recall that with the approximate small-projection matrix in hand, we still need to overcome Barriers 4 and 5 above. Computing the heaviest row η := max j∈[m] ∥a ⊤ j (I -V ⊤ V )∥ 2 (Barrier 1) can again be done (approximately) via the JL-sketch (as in the first step), by computing
η := max j∈[m] ∥a ⊤ j (I -V ⊤ V )
R∥ 2 using the JL matrix R. Since R has only O(ϵ -2 1 log(1/δ 1 )) columns (where choosing ϵ 1 = Θ(1) and δ 1 = δ/ poly(mn) is sufficient ) this step reduces from ss ss Iteration start Generating random Gaussian vector g Projecting g onto Span(V ) Condition check 1 Add e i to V Condition check 2 Add a i to V Condition check 3 Return fail Return yes no yes yes no no (a) Larsen's Partial Coloring Algorithm ss ss Generating a set of random Gaussian vectors g and store a batch of them projected onto V Iteration Start Query a projected g directly Condition check 1 Update e i to V and update the projection automatically Condition check 3 Condition check 2 Return Return fail yes yes no no no yes (b) Our Fast Partial Coloring Algorithm is the new-generated Gaussian vector and g is the projected vector. That is, g := (I -
V ⊤ V )g.
T mat (m, n, n) time to O(n 2 ).
As discussed earlier, Barrier 5 is a different ballgame and major obstruction to speeding up Larsen's algorithm, since the verification step
E τ := m j=1 (|⟨a j , v + g⟩| ≥ τ ∧ |⟨a j , v⟩| < τ ) (1)
needs to be performed adaptively in each of the n iterations, which appears impossible to perform in ≪ mn 2 time given the OMv Conjecture (Henzinger et al., 2015). Fortunately, it turns out we can avoid this verification step using a slight change in the analysis of Larsen (2023): Larsen's analysis shows the event E τ in Eq. ( 1) happens with constant probability. By slightly increasing the threshold to τ ′ = τ (δ), we can ensure the event E τ ′ happens with probability 1δ. Setting δ to a small enough constant so as not to affect the other parts of the algorithm, we can avoid this verification step altogether. At this point, the entire algorithm can be boosted to ensure high-probability of success. combining these ideas yields a (combinatorial) coloring algorithm that runs in O(nnz(A) + n 3 ). Next, we turn to explain the new idea required to overcome the cubic term n 3 , which is important for the near-square case (m ≈ n).
this section cite: ['b3', 'b54']

Section: Conclusion
In this work, we address the longstanding challenge of discrepancy minimization for real-valued matrices with a focus on achieving input-sparsity time algorithms. By introducing novel algorithmic components such as implicit leveragescore sampling and lazy update, our combinatorial algorithm achieves a runtime of O(nnz(A) + n 3 ) and a FMM-based variant breaks the cubic barrier to reach O(nnz(A) + n 2.53 ).
We significantly improve the computational efficiency while matching the approximation guarantees of existing methods. Our approach not only introduces novel algorithmic components but also demonstrates the potential of combining sketching and fast matrix multiplication in advancing computational geometry and optimization. We believe these techniques are of independent interest. Future work could extend these ideas to reduce the runtime of other problems in combinatorial optimization, or adapt techniques to tackle the problems in streaming and distributed models.
this section cite: []

Section: References
Ref_id:b0 Title: Follow the compressed leader: faster online learning of eigenvectors and faster mmwu Year: (2017)
Ref_id:b1 Title: Using optimization to obtain a width-independent, parallel, simpler, and faster positive SDP solver Year: (2016)
Ref_id:b2 Title: A refined laser method and faster matrix multiplication Year: (2021)
Ref_id:b3 Title: More asymmetry yields faster matrix multiplication Year: (2025)
Ref_id:b4 Title: The space complexity of approximating the frequency moments Year: (1996)
Ref_id:b5 Title: Discrepancy minimization via a self-balancing walk Year: (2021)
Ref_id:b6 Title: The volumetric barrier for semidefinite programming Year: (2000)
Ref_id:b7 Title: A combinatorial, primal-dual approach to semidefinite programs Year: (2007)
Ref_id:b8 Title: Balancing vectors and gaussian measures of n-dimensional convex bodies Year: (1998)
Ref_id:b9 Title: Constructive algorithms for discrepancy minimization Year: (2010)
Ref_id:b10 Title: Semidefinite optimization in discrepancy theory Year: (2012)
Ref_id:b11 Title: The gramschmidt walk: a cure for the banaszczyk blues Year: (2018)
Ref_id:b12 Title: An algorithm for komlós conjecture matching banaszczyk's bound Year: (2019)
Ref_id:b13 Title: Online vector balancing and geometric discrepancy Year: (2020)
Ref_id:b14 Title: A unified approach to discrepancy minimization Year: (2022)
Ref_id:b15 Title: Twiceramanujan sparsifiers Year: (2012)
Ref_id:b16 Title: Information discrepancy in strategic learning Year: (2022)
Ref_id:b17 Title: integer-making" theorems Year: (1981)
Ref_id:b18 Title: Fast matrix multiplication. Theory of Computing Year: (2013)
Ref_id:b19 Title: Optimal principal component analysis in distributed and streaming models Year: (2016)
Ref_id:b20 Title: Dynamic matrix inverse: Improved algorithms and matching conditional lower bounds Year: (2019)
Ref_id:b21 Title: Solving tall dense linear programs in nearly linear time Year: (2020)
Ref_id:b22 Title: Training (overparametrized) neural networks in near-linear time Year: (2021)
Ref_id:b23 Title: Algorithm and hardness for dynamic attention maintenance in large language models Year: (2024)
Ref_id:b24 Title: Subquadratic space representation of flows Year: (2025)
Ref_id:b25 Title: An homotopy method for lp regression provably beyond selfconcordance and in input-sparsity time Year: (2018)
Ref_id:b26 Title: Algebraic complexity theory Year: (2013)
Ref_id:b27 Title: A rank-1 sketch for matrix multiplicative weights Year: (2019)
Ref_id:b28 Title: Tight cell probe bounds for succinct boolean matrix-vector multiplication Year: (2018)
Ref_id:b29 Title: Finding frequent items in data streams Year: (2002)
Ref_id:b30 Title: Tight hardness results for minimizing discrepancy Year: (2011)
Ref_id:b31 Title: The discrepancy method: randomness and complexity Year: (2000)
Ref_id:b32 Title: Towards multi-pass streaming lower bounds for optimal approximation of max-cut Year: (2023)
Ref_id:b33 Title: Deep hashing via discrepancy minimization Year: (2018)
Ref_id:b34 Title: Low rank approximation and regression in input sparsity time Year: (2013)
Ref_id:b35 Title: Nearly tight oblivious subspace embeddings by trace inequalities Year: (2016)
Ref_id:b36 Title: Ramanujan graphs in polynomial time Year: (2016)
Ref_id:b37 Title: Solving linear programs in the current matrix multiplication time Year: (2019)
Ref_id:b38 Title: Discrepancy open problems Year: (2018)
Ref_id:b39 Title: Balancing vectors in any norm Year: (2018)
Ref_id:b40 Title: Maximization of a linear function of variables subject to linear inequalities. Activity analysis of production and allocation Year: (1947)
Ref_id:b41 Title: Computational geometry: algorithms and applications Year: (2000)
Ref_id:b42 Title: Attention scheme inspired softmax regression Year: (2023)
Ref_id:b43 Title: Fast approximation of matrix coherence and statistical leverage Year: (2012)
Ref_id:b44 Title: Efficient algorithms for discrepancy minimization in convex sets Year: (2018)
Ref_id:b45 Title: Improved rectangular matrix multiplication using powers of the coppersmith-winograd tensor Year: (2018)
Ref_id:b46 Title: An iterative algorithm for rescaled hyperbolic functions regression Year: (2023)
Ref_id:b47 Title: Sublinear time algorithms for approximate semidefinite programming Year: (2016)
Ref_id:b48 Title: Learning-augmented algorithms for online linear and semidefinite programming Year: (2022)
Ref_id:b49 Title: A faster small treewidth sdp solver Year: (2022)
Ref_id:b50 Title: Log-concave sampling over a convex body with a barrier: a robust and unified dikin walk Year: ()
Ref_id:b51 Title: Low rank matrix completion via robust alternating minimization in nearly linear time Year: (2024)
Ref_id:b52 Title: A nearly-linear time algorithm for structured support vector machines Year: (2025)
Ref_id:b53 Title: Kv cache compression through discrepancy theory Year: (2025)
Ref_id:b54 Title: Unifying and strengthening hardness for dynamic problems via the online matrix-vector multiplication conjecture Year: (2015)
Ref_id:b55 Title: A logarithmic additive integrality gap for bin packing Year: (2017)
Ref_id:b56 Title: Training overparametrized neural networks in sublinear time Year: (2022)
Ref_id:b57 Title: Solving sdp faster: A robust ipm framework and efficient implementation Year: (2022)
Ref_id:b58 Title: Solving sdp faster: A robust ipm framework and efficient implementation Year: (2022)
Ref_id:b59 Title: Solving sdp faster: A robust ipm framework and efficient implementation Year: (2022)
Ref_id:b60 Title: A parallel approximation algorithm for positive semidefinite programming Year: (2011)
Ref_id:b61 Title: Spencer's theorem in nearly input-sparsity time Year: (2023)
Ref_id:b62 Title: Linear-sized sparsifiers via near-linear time discrepancy theory Year: (2024)
Ref_id:b63 Title: A faster interior point method for semidefinite programming Year: ()
Ref_id:b64 Title:  Year: (2020)
Ref_id:b65 Title: A faster interior point method for semidefinite programming Year: (2020)
Ref_id:b66 Title: An improved cutting plane method for convex optimization, convex-concave games, and its applications Year: (2020)
Ref_id:b67 Title: Convex minimization with integer minima in õ (n 4) time Year: (2024)
Ref_id:b68 Title: Faster dynamic matrix inverse for faster lps Year: ()
Ref_id:b69 Title: Extensions of lipschitz mappings into a hilbert space Year: (1984)
Ref_id:b70 Title: A new polynomial-time algorithm for linear programming Year: (1984)
Ref_id:b71 Title: Discrepancy, coresets, and sketches in machine learning Year: (2019)
Ref_id:b72 Title: Polynomial algorithms in linear programming Year: (1980)
Ref_id:b73 Title: Constructive discrepancy minimization with hereditary l2 guarantees Year: (2017)
Ref_id:b74 Title: Fast discrepancy minimization with hereditary guarantees Year: (2017)
Ref_id:b75 Title: Faster online matrix-vector multiplication Year: (2017)
Ref_id:b76 Title: An O(m/ϵ 3.5 )-cost algorithm for semidefinite programs with diagonal constraints Year: (2020)
Ref_id:b77 Title: Path finding methods for linear programming: Solving linear programs in O ( √ rank) iterations and faster algorithms for maximum flow Year: (2014)
Ref_id:b78 Title: A faster cutting plane method and its implications for combinatorial and convex optimization Year: (2015)
Ref_id:b79 Title: Solving empirical risk minimization in the current matrix multiplication time Year: (2019)
Ref_id:b80 Title: A tighter complexity analysis of sparsegpt Year: (2024)
Ref_id:b81 Title: Solving regularized exp, cosh and sinh regression problems Year: (2023)
Ref_id:b82 Title: Space-efficient interior point method, with applications to linear programming and maximum weight bipartite matching Year: (2023)
Ref_id:b83 Title: Discrepancy of set-systems and matrices Year: (1986)
Ref_id:b84 Title: Constructive discrepancy minimization by walking on the edges Year: (2015)
Ref_id:b85 Title: Fast rank-1 lattice targeted sampling for black-box optimization Year: (2023)
Ref_id:b86 Title: Subgroup-based rank-1 lattice quasi-monte carlo Year: (2020)
Ref_id:b87 Title: Geometric discrepancy: An illustrated guide Year: (1999)
Ref_id:b88 Title: Geometric discrepancy: An illustrated guide Year: (1999)
Ref_id:b89 Title: Optimal private halfspace counting via discrepancy Year: (2012)
Ref_id:b90 Title: OSNAP: Faster numerical linear algebra algorithms via sparser subspace embeddings Year: (2013)
Ref_id:b91 Title: Conic formulation of a convex programming problem and duality Year: (1992)
Ref_id:b92 Title: Randomized rounding for the largest simplex problem Year: (2015)
Ref_id:b93 Title: The geometry of differential privacy: the sparse and approximate cases Year: (2013)
Ref_id:b94 Title: Discrepancy minimization via regularization Year: (2023)
Ref_id:b95 Title: An online and unified algorithm for projection matrix vector multiplication with application to empirical risk minimization Year: (2023)
Ref_id:b96 Title: Weighted low rank approximations with provable guarantees Year: (2016)
Ref_id:b97 Title: A polynomial-time algorithm, based on newton's method, for linear programming Year: (1988)
Ref_id:b98 Title: Approximating bin packing within O(log(OPT) loglog(OPT)) bins Year: ()
Ref_id:b99 Title: Constructive discrepancy minimization for convex sets Year: (2017)
Ref_id:b100 Title: Improved approximation algorithms for large matrices via random projections Year: (2006)
Ref_id:b101 Title: A variant of azuma's inequality for martingales with subgaussian tails Year: (2011)
Ref_id:b102 Title: Matrix Theory: Optimization, Concentration and Algorithms Year: (2019)
Ref_id:b103 Title: Oblivious sketching-based central path method for linear programming Year: (2021)
Ref_id:b104 Title: Low rank approximation with entrywise l1-norm error Year: (2017)
Ref_id:b105 Title: Relative error tensor low rank approximation Year: (2019)
Ref_id:b106 Title: Relative error tensor low rank approximation Year: (2019)
Ref_id:b107 Title: Accelerating frankwolfe algorithm using low-dimensional and adaptive data structures Year: (2022)
Ref_id:b108 Title: Speeding up sparsification using inner product search data structures Year: (2022)
Ref_id:b109 Title: Sketching for first order method: efficient algorithm for low-bandwidth channel and vulnerability Year: (2023)
Ref_id:b110 Title: Sketching meets differential privacy: Fast algorithm for dynamic kronecker projection maintenance Year: (2023)
Ref_id:b111 Title: Streaming semidefinite programs: o( {n}) passes, small space and fast runtime Year: (2023)
Ref_id:b112 Title: Six standard deviations suffice Year: (1985)
Ref_id:b113 Title: Graph sparsification by effective resistances Year: (2011)
Ref_id:b114 Title: Concentration of measure and isoperimetric inequalities in product spaces Year: (1995)
Ref_id:b115 Title: Improved analysis of the subsampled randomized hadamard transform Year: (2011)
Ref_id:b116 Title: An introduction to matrix concentration inequalities Year: (2015)
Ref_id:b117 Title: An algorithm for linear programming which requires O(((m+n)n 2 +(m+n) 1.5 n)L) arithmetic operations Year: (1987)
Ref_id:b118 Title: A new algorithm for minimizing convex functions over convex sets Year: (1989)
Ref_id:b119 Title: On the uniform convergence of relative frequencies of events to their probabilities Year: (1971)
Ref_id:b120 Title: Unsupervised learning of graph matching with mixture of modes via discrepancy minimization Year: (2012)
Ref_id:b121 Title: New bounds for matrix multiplication: from alpha to omega Year: (2024)
Ref_id:b122 Title: Sketching as a tool for numerical linear algebra Year: (2014)
Ref_id:b123 Title: Generative networks with metric embeddings Year: (2018)
Ref_id:b124 Title: Scalable semidefinite programming Year: (2019)
Ref_id:b125 Title: Speeding up optimizations via data structures: Faster search, sample and maintenance. Master's thesis Year: (2022)
Ref_id:b126 Title: Table 2. Results of experiments on uniform matrices. Here Larsen's algorithm is Larsen (2023). The runtime is measured in second Year: ()
Ref_id:b127 Title: Results of experiments on 2D corner matrices. Here Larsen's algorithm is Larsen (2023). The runtime is measured in second Year: ()
Ref_id:b128 Title: Results of experiments on 2D halfspace matrices. Here Larsen's algorithm is Larsen (2023). The runtime is measured in second Year: ()
Ref_id:b129 Title: Karmarkar's interiorpoint method (Karmarkar, 1984) was a major advance, offering polynomial complexity together with strong empirical performance. When the number of constraints d satisfies d = Ω(n) (where n is the number of variables), Karmarkar's algorithm runs in O Year: (1989)
Ref_id:b130 Title: c) shows that when m = Ω(n 2 ), an SDP can be solved in O(m ω + m 2+1/4 ) time. First-order algorithms avoid second-order information Year: (1992)
Ref_id:b131 Title:  Year: (2011)
Ref_id:b132 Title:  Year: (2016)
Ref_id:b133 Title:  Year: (2016)
Ref_id:b134 Title:  Year: (2017)
Ref_id:b135 Title:  Year: (2019)
Ref_id:b136 Title:  Year: (2019)
Ref_id:b137 Title:  Year: (2022)
Ref_id:b138 Title:  Year: (2023)
Ref_id:b139 Title: Applications of discrepancy theory in machine learning Matousek Year: (1999)
Ref_id:b140 Title: leverage discrepancy minimization for unsupervised graph matching by aligning predictions from classical solvers and neural models. Han et al. (2025) proposed an algorithm for compressing the KV cache recursively using a geometric correlated sampling process based on discrepancy theory. Nikolov et al. (2013) investigated the relationship between discrepancy minimization and differential privacy in the context of linear queries over histograms. Quasi-Monte Carlo methods Year: (2018)
