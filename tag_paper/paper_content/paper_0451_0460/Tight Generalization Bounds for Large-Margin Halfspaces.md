Title: Tight Generalization Bounds for Large-Margin Halfspaces
Abstract: We prove the first generalization bound for large-margin halfspaces that is asymptotically tight in the tradeoff between the margin, the fraction of training points with the given margin, the failure probability and the number of training points.

Section: Introduction
Halfspaces are arguably among the simplest and most fundamental classic learning models. Given a normal vector w ∈ R d and a bias b ∈ R defining a hyperplane, the corresponding halfspace classifier predicts the label of a data point x ∈ R d by returning sign(⟨w, x⟩ + b), corresponding to a +1 label on points inside the halfspace above the hyperplane, and -1 on points below.
Classic examples of learning algorithms for obtaining a halfspace classifier from a training set of points S = {(x i , y i )} n i=1 with (x i , y i ) ∈ R d × {-1, 1}, includes the Perceptron Learning Algorithm (PLA) (Mcculloch and Pitts [1943]) and Support Vector Machines (SVM) (Cortes and Vapnik [1995]). A key intuition underlying SVM, is the empirical observation that halfspaces with a large margin to the training data tend to generalize well. Ignoring the bias variable b (which we later handle by adding a special feature) and assuming w ∈ S d-1 (i.e. w has unit length), the margin of the halfspace with normal vector w on a labeled point (x, y) is y⟨w, x⟩. Observe that ⟨w, x⟩ gives the signed distance of x from the hyperplane, and the margin is positive when sign(⟨w, x⟩) correctly predicts the label y. With this definition, hard-margin SVM computes the normal vector w of the hyperplane with the largest minimum margin. There are also margin variants of the Perceptron (Freund and Schapire [1999]) that computes a halfspace with minimum margin approaching the optimal, as in hard-margin SVM.
To handle data that is not linearly separable, and to add robustness to outliers, the soft-margin SVM relaxes the optimization problem to the following min w,ξ
∥w∥ 2 2 + λ i ξ i , s.t. y i ⟨w, x i ⟩ ≥ 1 -ξ i , ξ i ≥ 0.
Here λ > 0 is a regularization parameter. The soft-margin SVM thus allows for smaller margins on some training points at the cost of a penalty λξ i . For fast implementations of SVM, see Gu et al. [2025] andS. Shalev-Shwartz andCotter [2011].
To theoretically justify and explain the empirical success of focusing on large margins, Bartlett and Shawe-Taylor [1999] proved the first generalization bounds upper bounding the probability L D (w) := P (x,y)∼D [sign(⟨w, x⟩) ̸ = y] of misclassifying the label of a new data point. Concretely, Bartlett and Shawe-Taylor first studied the hard-margin case and proved that for any distribution D over B d 2 × {-1, 1} and any 0 < δ < 1, it holds with probability at least 1 -δ over a training set S ∼ D n that for every w ∈ S d-1 and every margin 0 < γ < 1, if y⟨w, x⟩ ≥ γ for all (x, y) ∈ S then
L D (w) ≤ c • ln 2 (n) γ 2 n + ln(e/δ) n ,(1)
for a constant c > 0. Here B d 2 is the d-dimensional unit ball and S d-1 is the d-dimensional unit sphere, both with respect to the l 2 -norm. The restriction to x ∈ B d 2 can be relaxed by multiplying the first term by R 2 for x ∈ R • B d 2 . A dependency on the scaling of input points is inevitable as margins scale with ∥x∥ 2 . Throughout the paper, we state bounds for R = 1 and remark that all bounds generalize to arbitrary R by replacing γ by γ/R.
Defining L γ S (w) as the fraction of data points in a training set S where w has margin at most γ, Bartlett and Shawe-Taylor [1999] also prove a more general result, saying that with probability 1 -δ over S ∼ D n , it holds for every w ∈ S d-1 that
L D (w) ≤ L γ S (w) + c • ln 2 (n) γ 2 n + ln(e/δ) n .(2)
This was later improved by Bartlett and Mendelson [2002] using Rademacher complexity arguments, replacing the ln 2 (n) term in (2) by 1. Here, and throughout the paper, we refer to L γ S (w) as the (empirical) margin loss.
First-Order Bounds. The first work to interpolate between the hard-margin and soft-margin bounds was due to McAllester [2003], who gave a general tradeoff of
L D (w) ≤ L γ S (w) + c • L γ S (w) • ln n γ 2 n + ln n γ 2 n + ln n + ln(e/δ) n .(3)
Notice how the L γ S (w) term is multiplied onto ln n/(γ 2 n) inside the first square-root. Since the hard-margin case corresponds to this term being 0, this gives a way of interpolating between the cases. Such bounds are often referred to as first-order bounds. Unfortunately, (3) still has the seemingly superfluous (ln n + ln(e/δ))/n term even when L γ S (w) = 0 and thus falls short of even matching (1) in the hard-margin case.
The current state-of-the-art generalization bound is due to Grønlund et al. [2020a] and states that with probability 1 -δ over S ∼ D n , it holds for every w ∈ S d-1 that
L D (w) ≤ L γ S (w) + c • L γ S (w) • ln n γ 2 n + ln(e/δ) n + ln n γ 2 n + ln(e/δ) n .(4)
This improves previous hard-margin bounds by a logarithmic factor and gives a cleaner interpolation between the hard-and soft-margin cases. Furthermore, the bound is close to optimal. Concretely, the dependency on δ is optimal by tweaking standard results for agnostic PAC learning, see e.g. Devroye et al. [1996] [Chapter 11]. Moreover, Grønlund et al. [2020a] complemented their upper bound by the following lower bound
Theorem 1 (Grønlund et al. [2020a]). There is a constant c > 0 such that for any cn -1/2 < γ < c -1 , any parameter 0 ≤ τ ≤ 1, and any n ≥ c, there is a distribution D such that it holds with constant probability over S ∼ D n that there is a w ∈ S d-1 such that L γ S (w) ≤ τ and
L D (w) ≥ L γ S (w) + c • τ • ln(e/τ ) γ 2 n + ln(γ 2 n) γ 2 n ≥ L γ S (w) + c •   L γ S (w) • ln(e/L γ S (w)) γ 2 n + ln(γ 2 n) γ 2 n   .
Notice how the parameter τ allows for showing that the upper bound (4) is nearly tight across the range of L γ S (w). Let us also remark that Grønlund et al. [2020a] states their lower bound with a ln n rather than ln(eγ 2 n), but require that γ > n -0.499 . A careful examination of their proof however reveals the more general lower bound stated here.
Unfortunately, there still remains a discrepancy between the lower bound and (4). Concretely, there is a gap of ln n/ ln(e/L γ S (w)). Moreover, for constant L γ S (w), the Rademacher complexity based bound in (2) improves over both of the first-order bounds (3) and (4), and matches the lower bound in Theorem 1. This seems to suggest that a better upper bound might be possible.
this section cite: ['b17', 'b3', 'b7', 'b11', 'b18', 'b1', 'b1', 'b2', 'b16']

Section: Our Contribution.
In this work, we settle the generalization performance of large-margin halfspaces by proving a new upper bound matching the lower bound in Theorem 1 across the entire tradeoff between γ, L γ S (w) and n (and is also tight in terms of δ). Our result is stated in the following theorem Theorem 2. There is a constant c > 0 such that for any distribution D over B d 2 × {-1, 1}, it holds with probability at least 1 -δ over S ∼ D n that for every w ∈ S d-1 and every margin n -1/2 ≤ γ ≤ 1, we have
L D (w) ≤ L γ S (w) + c L γ S (w) • ln(e/L γ S (w)) γ 2 n + ln(e/δ) n + ln(eγ 2 n) γ 2 n + ln(e/δ) n .
Using a simple reduction, our results generalize to all non-homogeneous halfspaces (i.e. including a bias term) and data in a ball of radius R, yielding Theorem 2 with γ/R in place of γ. See Appendix A.
While one might argue that our improvement is small in magnitude, this finally pins down the exact generalization performance of a classic learning model. Furthermore, our proof of Theorem 2 brings several novel ideas that we hope may find further applications in generalization bounds.
We next proceed to give an overview of our proof and new ideas in Section 2, before giving the full details of the proof in Section 3.
this section cite: []

Section: Proof Overview
In this section, we present the main ideas in our proof of Theorem 2. As our proof builds on, and greatly extends, the work of Grønlund et al. [2020a] establishing the previous state-of-the-art in (4), we first present their overall proof strategy and the barriers we need to overcome to obtain our tight generalization bound. Throughout this proof overview, we use the notation x ≲ y to denote that there is an absolute constant c > 0 so that x ≤ cy.
this section cite: []

Section: Previous Proof
The proof of Grønlund et al. [2020a] follows a framework proposed by Schapire et al. [1998] for proving generalization of large-margin voting classifiers (i.e. boosting). The main idea is to randomly discretize the infinite hypothesis set S d-1 to obtain a finite set G ⊆ R d → {-1, 1}. If G is small enough, then a standard union bound over all h ∈ G suffices to bound the difference between the empirical error and the true error L D (h) for every h ∈ G. The key trick is to exploit large margins to allow for a discretization to a smaller G.
To elaborate on the above, let us first generalize our notation L D (w) and L γ S (w) a bit. For a distribution D over
B d 2 × {-1, 1}, let L γ D (w) := P (x,y)∼D [y⟨w, x⟩ ≤ γ], that is, L γ D (w)
is the probability over a fresh sample (x, y) from D, of w having margin no more than γ on (x, y). For a training set S, we slightly abuse notation and write (x, y) ∼ S to denote a uniform random sample from S. We thus have
L γ S (w) := P (x,y)∼S [y⟨w, x⟩ ≤ γ] = |{(x, y) ∈ S : y⟨w, x⟩ ≤ γ}| |S| .
When writing L D (w) we implicitly mean L 0 D (w) and note that this coincides with our previous definition of L D (w) = P (x,y)∼D [sign(⟨w, x⟩) ̸ = y] (defining sign(0) = 0).
this section cite: ['b19']

Section: Random Discretization.
With this notation, the main idea in the proof of Grønlund et al. [2020a], is to apply a Johnson-Lindenstrauss transform (Johnson and Lindenstrauss [1984]), followed by a random snapping to a grid, in order to map each w ∈ S d-1 to a point on a grid G of size exp(ck) in R k , with c > 0 a sufficiently large constant. In more detail, let A be a k × d matrix with i.i.d. N (0, 1/k) normal distributed entries. Such a matrix is a classic implementation of the Johnson-Lindenstrauss transform and has the property that |⟨Aw, Ax⟩ -⟨w, x⟩| is greater than ε with probability at most exp(-ε 2 k/c) when ∥w∥ 2 , ∥x∥ 2 ≤ 1 (Dasgupta and Gupta [2003]). Note that this also preserves the norm of a vector w by considering x = w and noting ⟨w, w⟩ = ∥w∥ 2 2 . Secondly, following an idea of Alon and Klartag [2017] in a lower bound proof for the Johnson-Lindenstrauss transform, Grønlund et al. [2020a] randomly round Aw to a point h A,t (w) with coordinates integer multiples of k -1/2 while guaranteeing that |⟨h A,t (w), Ax⟩ -⟨Aw, Ax⟩| is less than ε, except with probability exp(-ε 2 k/c). Here we use t to denote the randomness involved in the rounding. Now choosing ε = γ/4 gives, by the triangle inequality, that |⟨h A,t (w), Ax⟩ -⟨w, x⟩| ≤ γ/2, except with probability 2 exp(-γ 2 k/(16c)). Furthermore, by plugging in x = w and setting ε = 1, we can also deduce that ∥h A,t (w)∥ 2 ≤ 2 except with probability exp(-k/c). Simple counting arguments show that there are only exp(ck) many vectors of norm at most 2 with all coordinates integer multiples of k -1/2 . That is, except with probability exp(-k/c), h A,t (w) belongs to a finite set G of exp(ck) many points.
Framework. With the above random discretization, the proof of Grønlund et al. [2020a] now follows the framework of Schapire et al. [1998] by relating
L D (w) to L γ/2 AD (h A,t (w)) and L γ S (w) to L γ/2 AS (h A,t (w)).
Here AD is the distribution obtained by sampling (x, y) ∼ D and returning (Ax, y). Similarly, AS is the training set obtained by replacing each (x, y) ∈ S by (Ax, y). The intuition is that the random discretization changes margins by no more than γ/2 for most data points and hence points with margin at most 0 under D often have margin at most γ/2 under AD and similarly for S and AS. Let us make this more formal. We have for any A, t in the support of A, t that
L D (w) ≤ L γ/2 AD (h A,t (w)) + P (x,y)∼D [y⟨w, x⟩ ≤ 0 ∧ y⟨h A,t (w), Ax⟩ > γ/2].
(5) Similarly, we have
L γ S (w) ≥ L γ/2 AS (h A,t (w)) -P (x,y)∼S [y⟨w, x⟩ > γ ∧ y⟨h A,t (w), Ax⟩ ≤ γ/2].(6)
Taking expectation we see that
L D (w) -L γ S (w) = E A,t [L D (w) -L γ S (w)] ≤ E A,t [L γ/2 AD (h A,t (w)) -L γ/2 AS (h A,t (w))] (7) + E A,t [P (x,y)∼D [y⟨w, x⟩ ≤ 0 ∧ y⟨h A,t (w), Ax⟩ > γ/2]] (8) + E A,t [P (x,y)∼S [y⟨w, x⟩ > γ ∧ y⟨h A,t (w), Ax⟩ ≤ γ/2]].(9)
To bound (7), we exploit that h A,t (w) belongs to the grid G, except with probability exp(-k/c). Using Bernstein's inequality (and a careful partitioning of hypotheses w depending on L γ D (w)), it is possible to union bound over the entire grid and conclude
E A,t [L γ/2 AD (h A,t (w)) -L γ/2 AS (h A,t (w))] ≤ E A,t [sup h∈G L γ/2 AD (h) -L γ/2 AS (h)] + P A,t [h A,t (w) / ∈ G] ≲ L γ S (w) • ln(|G|/δ) n + ln(|G|/δ) n + exp(-k/c) ≲ L γ S (w) • k + ln(e/δ) n + k + ln(e/δ) n + exp(-k/c).(10)
To bound (9), we use the guarantees of the random discretization to conclude that
E A,t [P (x,y)∼S [y⟨w, x⟩ > γ ∧ y⟨h A,t (w), Ax⟩ ≤ γ/2]] = E (x,y)∼S [P A,t [y⟨w, x⟩ > γ ∧ y⟨h A,t (w), Ax⟩ ≤ γ/2]] ≤ E (x,y)∼S [P A,t [y⟨h A,t (w), Ax⟩ ≤ γ/2 | y⟨w, x⟩ > γ]] ≤ 2 exp(-γ 2 k/(16c
)). We can bound (8) in a similar fashion (even with slightly better guarantees scaled by L D (w), but this does not help for ( 9)). The final generalization error thus becomes
L D (w) ≤ L γ S (w) + c ′ • L γ S (w) • k + ln(e/δ) n + k + ln(e/δ) n + exp(-γ 2 k/c ′ ) ,(11)
where c ′ > 0 is a sufficiently large constant. Comparing this expression with the desired bound from Theorem 2, we see that we have to choose k large enough that c ′ exp(-
γ 2 k/c ′ ) is no larger than L γ S (w) • ln(e/L γ S (w)) γ 2 n + ln(e/δ) n + ln(eγ 2 n) γ 2 n + ln(e/δ) n .
This basically solves to
k ≳ γ -2 ln γ 2 n L γ S (w) ln(e/L γ S (w)) ≥ γ -2 ln γ 2 n .
Inserting this k in (11) recovers the bound by Grønlund et al. [2020a] stated in (4).
this section cite: ['b12', 'b5', 'b0', 'b19']

Section: Barriers.
In light of the above discussion, we identify some key barriers for the previous proof technique. Concretely, if we examine (11), the term L γ S (w)k/n requires us to choose k no larger than cγ -2 ln(e/L γ S (w)) to match the optimal bound we get in Theorem 2. Unfortunately, the additive exp(-γ 2 k/c ′ ) term originating from handling (8) and ( 9) then becomes poly(L γ S (w)), which is too expensive. In fact, even the additive exp(-k/c) term from handling (7) is too expensive for e.g. constant γ. Nonetheless, we will in fact choose such k and identify a tighter strategy for analysing L D (w) -L γ S (w).
this section cite: []

Section: Our Key Improvements
Our first main observation is that the two upper bounds in ( 5) and ( 6) are not completely tight, i.e. they are inequalities, not equalities. In (5) we for instance ignore points (x, y) that had a margin greater than 0 for w, but where the margin of (Ax, y) is less than γ/2 for h A,t (w). Taking these into accounts, we get the tighter bounds
L D (w) = L γ/2 AD (h A,t (w)) + P (x,y)∼D [y⟨w, x⟩ ≤ 0 ∧ y⟨h A,t (w), Ax⟩ > γ/2] -P (x,y)∼D [y⟨w, x⟩ > 0 ∧ y⟨h A,t (w), Ax⟩ ≤ γ/2],and
L γ S (w) = L γ/2 AS (h A,t (w)) -P (x,y)∼S [y⟨w, x⟩ > γ ∧ y⟨h A,t (w), Ax⟩ ≤ γ/2] + P (x,y)∼S [y⟨w, x⟩ ≤ γ ∧ y⟨h A,t (w), Ax⟩ > γ/2].
With these refined bounds, we can now split L D (w) -L γ S (w) into a sum of three terms:
L γ/2 AD (h A,t (w)) -L γ/2 AS (h A,t (w)) + P D [y⟨w, x⟩ ≤ 0 ∧ y⟨h A,t (w), Ax⟩ > γ/2] -P S [y⟨w, x⟩ ≤ γ ∧ y⟨h A,t (w), Ax⟩ > γ/2] (12
) + P S [y⟨w, x⟩ > γ ∧ y⟨h A,t (w), Ax⟩ ≤ γ/2] -P D [y⟨w, x⟩ > 0 ∧ y⟨h A,t (w), Ax⟩ ≤ γ/2].(13)
The first line is the same as ( 7) from before, but ( 12) and ( 13) improves over ( 8) and ( 9) by subtracting off a term. Intuitively, our more refined bounds allow us to argue that if the randomized rounding creates a big difference between L D (w) and L γ/2 D (h A,t (w)), then it creates a comparably large difference between L γ S (w) and L γ/2 S (h A,t (w)), thereby canceling out. We will carefully exploit this in the following. Let us focus on (12) and remark that (13) is handled symmetrically. For (12), we see that
P (x,y)∼S [y⟨w, x⟩ ≤ γ ∧ y⟨h A,t (w), Ax⟩ > γ/2] ≥ P (x,y)∼S [y⟨w, x⟩ ≤ 0 ∧ y⟨h A,t (w), Ax⟩ > γ/2],
and thus (12) is at most
P (x,y)∼D [y⟨w, x⟩ ≤ 0 ∧ y⟨h A,t (w), Ax⟩ > γ/2] -P (x,y)∼S [y⟨w, x⟩ ≤ 0 ∧ y⟨h A,t (w), Ax⟩ > γ/2].
Now introducing the expectation over the randomized rounding A and t as in the previous proof, and using linearity of expectation, we want to bound the following expression with probability 1 -δ over
S ∼ D n sup w∈S d-1 E A,t [P (x,y)∼D [y⟨w, x⟩ ≤ 0 ∧ y⟨h A,t (w), Ax⟩ > γ/2]]- E A,t [P (x,y)∼S [y⟨w, x⟩ ≤ 0 ∧ y⟨h A,t (w), Ax⟩ > γ/2]] = sup w∈S d-1 E (x,y)∼D [P A,t [y⟨w, x⟩ ≤ 0 ∧ y⟨h A,t (w), Ax⟩ > γ/2]]- E (x,y)∼S [P A,t [y⟨w, x⟩ ≤ 0 ∧ y⟨h A,t (w), Ax⟩ > γ/2]] .(14)
This now has a form that looks familiar. Concretely, we have a function
ψ w (x, y) = 1{y⟨w, x⟩ ≤ 0} • P A,t [y⟨h A,t (w), Ax⟩ > γ/2].(15)
for each w ∈ S d-1 , and wish to bound sup w E (x,y)∼D [ψ w (x, y)] -E (x,y)∼S [ψ w (x, y)] with high probability over S ∼ D n . Rademacher complexity (see e.g. Shalev-Shwartz and Ben-David [2014]) is one key tool for bounding such differences. In particular, the contraction principle from Ledoux and Talagrand [1991] allows us to bound such a supremum when the functions ψ w are composite functions ψ w = f • g w with f : R → R having bounded Lipschitz constant. In (Bartlett and Mendelson [2002]) this method is used, with f being the ramp loss, resulting in a bound on ( 14) of 1/(γ 2 n). We wish to take a similar approach for our ψ w in (15).
To argue that ψ w = f • g w with g w (x, y) = y⟨w, x⟩, we need the probability in (15) to only depend on the original margin y⟨w, x⟩. This is precisely the statement of Claim 3, which is proven in Appendix B.1. We thus proceed to bound the Lipschitz constant of the function f in (15). To avoid discontinuities, we have to alter ψ w (x, y) somewhat to not include the discontinuous indicator function, and we eventually bound the Lipschitz constant L by roughly
L ≲ γ -1 P A,t [y⟨h A,t (w), Ax⟩ > γ/2 | y⟨w, x⟩ = 0].
With a slight abuse of notation, we write P A,t [y⟨h A,t (w), Ax⟩ > γ/2 | y⟨w, x⟩ = 0] to denote the probability P A,t [y⟨h A,t (w), Ax⟩ > γ/2] for an arbitrary x, w ∈ S d-1 and y ∈ {-1, 1} with y⟨w, x⟩ = 0 as y⟨w, x⟩ completely determines this probability as argued above.
Since our randomized rounding preserves inner products to within γ/2 except with probability exp(-γ 2 k/c), we get L ≲ γ -1 exp(-γ 2 k/c). This finally bounds (14) by
c • exp(-γ 2 k/c) γ 2 n .
This should be compared to proof by Grønlund et al. [2020a] that got a bound of c exp(-γ 2 k/c) and the 1/(γ 2 n) bound mentioned above. This improvement is precisely enough to derive our tight Theorem 2. Indeed, as mentioned in (10), we can bound
L γ/2 AD (h A,t (w)) -L γ/2 AS (h A,t (w)) by L γ S (w) • k + ln(e/δ) n + k + ln(e/δ) n + exp(-k/c).
If we ignore the exp(-k/c) term and set k = c ′ γ -2 ln(e/L γ S (w)), this gives the tight bound in Theorem 2.
Unfortunately, we cannot afford to ignore the exp(-k/c) term and we need additional ideas for dealing with it. Recall that in the previous proof by Grønlund et al. [2020a], it originates from bounding
E A,t [L γ/2 AD (h A,t (w)) -L γ/2 AS (h A,t (w))] ≤ E A,t [sup h∈G L γ/2 AD (h) -L γ/2 AS (h)] + P A,t [h A,t (w) / ∈ G],
and upper bounding P A,t [h A,t (w) / ∈ G] by exp(-k/c). Here we instead consider an infinite sequence of discretizations/grids G 0 , G 1 , . . . , and argue that the random rounding A, t and training set S is simultaneously good (for some appropriate definition) for all grids with high probability.
Here the grids G i correspond to increasingly large norms of h A,t (w), i.e. G i contains all vectors of norm at most 2 i+1 B d 2 and all coordinates integer multiples of k -1/2 . Multiple careful applications of Cauchy-Schwartz, Jensen's inequality and upper bounds on the probability that h A,t (w) / ∈ G i allows us to finally get rid of the exp(-k/c) factor.
this section cite: ['b15', 'b2']

Section: Main Proof
We now set out to prove Theorem 2 following the proof outline sketched in Section 2. We start by a series of reductions that allow us to focus on a simpler task of establishing Theorem 2 only for a small range of γ and L γ S (w). We describe these reductions in Section 3.1 and then proceed to the main arguments in Section 3.2.
this section cite: []

Section: Setup
When eventually bounding the Lipschitz constant, as discussed in Section 2, the task turns out to be simpler if ∥x∥ 2 = 1 (and not just ∥x∥ 2 ≤ 1) for all x in the support of D and if |⟨w, x⟩| < c γ for a constant c γ sufficiently smaller than 1 for all hypotheses w and data points (x, y) in the support of D. We reduce to this case in Appendix A. The reduction maps every w ∈ S d-1 to a vector in H := S d-1 × {0}, and every x in the support of D to a vector in X , where X is the set of all vectors x ′ in S d where the norm of x ′ without its (d + 1)'st coordinate is at most c γ .
From hereon, we let D be an arbitrary distribution over X × {-1, 1}, and set out to prove that there is a constant c > 1, such that with probability at least 1 -δ over S ∼ D n , it holds for all margins γ ∈ (n -1/2 , c γ ] and all w ∈ H that
L D (w) ≤ L γ S (w) + c L γ S (w) • ln(e/L γ S (w)) γ 2 n + ln(e/δ) n + ln(eγ 2 n) γ 2 n + ln(e/δ) n .(16)
Theorem 2 follows as a corollary.
Smaller Tasks. We now break the task of establishing (16) into smaller tasks, where we consider margins γ in a small range (γ i , γ i+1 ] and only vectors w ∈ H with L
(3/4)γi D (w) in a small range (ℓ j , ℓ j+1 ]. The purpose here is, that for one sub-task, we can treat margins and margin losses as the same within constant factors. A union bound over all the sub-tasks then suffices to establish (16).
For a given distribution D, partition the range of values of the margin γ ∈ (n -1/2 , c γ ] into intervals
Γ i = (2 i-1 n -1/2 , 2 i n -1/2 ] for i = 1, . . . , lg 2 (c γ n 1/2 ). Similarly, partition the possible values of L γ D (w) ∈ [0, 1] into intervals L 0 = [0, n -1 ] and L i = (2 i-1 n -1 , 2 i n -1 ] with i = 1, . . . lg 2 n. For a pair (Γ i , L j ) with Γ i = (γ i , γ i+1 ], define H(Γ i , L j ) = {w ∈ H : L (3/4)γi D (w) ∈ L j }.
For each pair (Γ i , L j ) we now prove an equivalent of ( 16), but tailored to the sub-task. The result is stated in the following lemma Lemma 3. There is a constant c > 1, such that for any 0 < δ < 1 and any pair (Γ i , L j ) = ((γ i , γ i+1 ], (ℓ j , ℓ j+1 ]), it holds with probability at least 1 -δ over a random sample S ∼ D n that
sup w∈H(Γi,Lj ) γ∈Γi |L D (w) -L γ S (w)| ≤ c ℓ j+1 ln(e/ℓ j+1 ) γ 2 i+1 n + ln(e/δ) n + ln(e/ℓ j+1 ) γ 2 i+1 n + ln(e/δ) n .(17)
Observe that while (16) depends on γ and (17) depends on γ i+1 , this is fine since γ ≤ γ i+1 for all γ ∈ Γ i . However, recall that H(Γ i , L j ) refers to w ∈ H with L
(3/4)γi D (w) ∈ L j = (ℓ j , ℓ j+1 ]. But the ℓ j+1 terms in (17) need to be replaced by L γ S (w) to obtain (16). Thus we relate the two via the following lemma Lemma 4. There is a constant c > 1, such that for any 0 < δ < 1 and any Γ i = (γ i , γ i+1 ], it holds with probability at least 1 -δ over a random sample S ∼ D n that ∀w ∈ H :
L γi S (w) ≥ L (3/4)γi D (w) 4 -c ln(eγ 2 i+1 n) γ 2 i+1 n + ln(e/δ) n .(18)
We combine the sub-tasks and conclude Claim 1. For any 0 < δ < 1, it holds with probability 1 -δ over S ∼ D n that equations (17) and (18) simultaneously hold for all (Γ i , L j ) and Γ i , with slightly different constants c.
Since Claim 1 follows by a simple union bound, exploiting that for different values of ℓ j+1 and γ i+1 , we can afford to use different δ i,j ≈ δ exp(-γ 2 i+1 ln(e/ℓ j+1 )) and δ i ≈ δ exp(-γ -2 i+1 ln(eγ 2 i+1 n)), we have deferred the proof to Appendix E.
A simple combination of ( 17) and (18) now gives Claim 2. For any 0 < δ < 1 and training set S, if equations (17) and (18) hold simultaneously for all (Γ i , L j ) and Γ i , then equation (16) holds for all γ ∈ (n -1/2 , c γ ] and all w ∈ H for a large enough constant c > 1 in (16).
Claim 2 follows by using that γ ≤ γ i+1 for γ ∈ Γ i , and by using Lemma 4 to relate all occurrences of ℓ j+1 in (17) to L γ S (w). As this is rather straight forward calculations, we have deferred the proof to Appendix E.
What remains is thus to establish Lemma 3 and Lemma 4, where we may now focus on a small range of γ and L
(3/4)γi D (w). While both require substantial work and non-trivial arguments, the proof of Lemma 4 follows mostly the previous work by Grønlund et al. [2020a] and has thus been deferred to Appendix D.
this section cite: []

Section: Random Discretization
We now set out to prove Lemma 3. So let 0 < δ < 1, and fix a pair (Γ i , L j ). Following the proof outline in Section 2, we now consider the following random discretization of hypotheses in H(Γ i , L j ): Let k = k(i, j) be an integer parameter to be determined. Sample a random k × d matrix A with each entry N (0, 1/k) distributed as well as k random offsets t = (t 1 , . . . , t k ) all independent and uniformly distributed in [0, 1].
Let G be the set of all vectors in R k with coordinates in {(1/2)(10
√ k) -1 + z(10 √ k) -1 | z ∈ Z}.
For w ∈ H and an outcome (A, t) of (A, t), define h A,t (w) ∈ G as the vector obtained as follows: Consider each coordinate (Aw) i and let z i denote the integer such that (1/2)(10
√ k) -1 + z i (10 √ k) -1 ≤ (Aw) i < (1/2)(10 √ k) -1 + (z i + 1)(10 √ k) -1 . Let (h A,t (w)) i equal (1/2)(10 √ k) -1 + z i (10 √ k) -1 if t i ≤ p((Aw) i ) ((
Aw) i rounded down) and otherwise let it equal (1/2)(10 √ k) -1 + (z i + 1)(10 √ k) -1 . By standard arguments, which we have deferred to Appendix E, we can choose p((Aw) i ) ∈ [0, 1] such that the expected value of the coordinates satisfy E t [(h A,t (w)) i ] = (Aw) i . The random discretization has the desirable property that it approximately preserves margins/inner products as stated
this section cite: []

Section: in the following
Lemma 5. There is a constant c > 0, such that for any integer k ≥ 1, w ∈ H, x ∈ X and any γ ∈ (0, 1], it holds that P A,t [|⟨h A,t (w), Ax⟩ -⟨w, x⟩| > γ] < c exp(-γ 2 k/c).
The proof of Lemma 5 follows the work by Alon and Klartag [2017] in their work on lower bounds for the Johnson-Lindenstrauss transform, and has thus been deferred to Appendix E. We now observe that
L D (w) = L γi/2 AD (h A,t (w)) + P (x,y)∼D [y⟨h A,t (w), Ax⟩ > γ i /2 ∧ y⟨w, x⟩ ≤ 0] -P (x,y)∼D [y⟨h A,t (w), Ax⟩ ≤ γ i /2 ∧ y⟨w, x⟩ > 0].
Similarly, we have for γ ∈ Γ i and any training set S that
L γ S (w) = L γi/2 AS (h A,t (w)) + P (x,y)∼S [y⟨h A,t (w), Ax⟩ > γ i /2 ∧ y⟨w, x⟩ ≤ γ] -P (x,y)∼S [y⟨h A,t (w), Ax⟩ ≤ γ i /2 ∧ y⟨w, x⟩ > γ].
We now have for any γ ∈ Γ i that
sup w∈H(Γi,Lj ) L D (w) -L γ S (w) = sup w∈H(Γi,Lj ) E A,t [L γi/2 AD (h A,t (w)) -L γi/2 AS (h A,t (w))]+ E A,t [P D [y⟨h A,t (w), Ax⟩ > γ i /2 ∧ y⟨w, x⟩ ≤ 0] -P S [y⟨h A,t (w), Ax⟩ > γ i /2 ∧ y⟨w, x⟩ ≤ γ]]+ E A,t [P S [y⟨h A,t (w), Ax⟩ ≤ γ i /2 ∧ y⟨w, x⟩ > γ] -P D [y⟨h A,t (w), Ax⟩ ≤ γ i /2 ∧ y⟨w, x⟩ > 0]] .(19)
A critical observation is that the distribution of y⟨h A,t (w), Ax⟩ depends only on y⟨w, x⟩. Claim 3. For any (x, y) ∈ X × {-1, 1} and any w ∈ H, the distribution of y⟨h A,t (w), Ax⟩ is completely determined from y⟨w, x⟩.
We prove Claim 3 in Appendix B.1 by exploiting that the entries of A are i.i.d. N (0, 1/k) distributed and using the rotational invariance of the Gaussian distribution.
As outlined in the proof overview in Section 2, we can now use Claim 3 together with the contraction inequality of Rademacher complexity to bound several of the terms in ( 19). Similarly to the introduction of the ramp loss in classic proofs of generalization for large-margin halfspaces, we need to introduce a continuous function upper bounding the probabilities above. With this in mind, we now define the following functions ϕ and ρ:
ϕ(α) =      P A,t [y⟨h A,t (w), Ax⟩ > γ i /2 | y⟨w, x⟩ = α] if -c γ ≤ α ≤ 0 (γi-α) γi P A,t [y⟨h A,t (w), Ax⟩ > γ i /2 | y⟨w, x⟩ = 0] if 0 < α ≤ γ i 0 if γ i < α ≤ c γ ρ(α) =      P A,t [y⟨h A,t (w), Ax⟩ ≤ γ i /2 | y⟨w, x⟩ = α] if γ i < α ≤ c γ α γi P A,t [y⟨h A,t (w), Ax⟩ ≤ γ i /2 | y⟨w, x⟩ = γ i ] if 0 < α ≤ γ i 0 if -c γ ≤ α ≤ 0 .
Here we slightly abuse notation and write P A,t [y⟨h A,t (w), Ax⟩ > γ i /2 | y⟨w, x⟩ = α] to denote the probability P A,t [y⟨h A,t (w), Ax⟩ > γ i /2] for an arbitrary w ∈ H, (x, y) ∈ X × {-1, 1} with y⟨w, x⟩ = α and remark that this probability is the same for all such w, x, y by Claim 3.
We now observe that ϕ and ρ upper and lower bounds the terms in (19) Remark 6. For any training set S and distribution D over X × {-1, 1}, we have
E A,t [P (x,y)∼D [y⟨h A,t (w), Ax⟩ > γ i /2 ∧ y⟨w, x⟩ ≤ 0]] ≤ E (x,y)∼D [ϕ(y⟨w, x⟩)] E A,t [P (x,y)∼S [y⟨h A,t (w), Ax⟩ > γ i /2 ∧ y⟨w, x⟩ ≤ γ]] ≥ E (x,y)∼S [ϕ(y⟨w, x⟩)] E A,t [P (x,y)∼S [y⟨h A,t (w), Ax⟩ ≤ γ i /2 ∧ y⟨w, x⟩ > γ]] ≤ E (x,y)∼S [ρ(y⟨w, x⟩)] E A,t [P (x,y)∼D [y⟨h A,t (w), Ax⟩ ≤ γ i /2 ∧ y⟨w, x⟩ > 0]] ≥ E (x,y)∼D [ρ(y⟨w, x⟩)].
The proof of Remark 6 follows from the definition of ϕ and ρ, along with monotonicity of
P A,t [y⟨h A,t (w), Ax⟩ > γ i | y⟨w, x⟩ = α]
as a function of α. The proofs have been deferred
to Appendix E. Continuing from ( 19) using Remark 6, linearity of expectation and the triangle inequality, we have for any γ ∈ Γ i that
sup w∈H(Γi,Lj ) L D (w) -L γ S (w) ≤ sup w∈H(Γi,Lj ) E A,t [L γi/2 D (h A,t (w)) -L γi/2 S (h A,t (w))] (20
)
+ sup w∈H(Γi,Lj ) E (x,y)∼D [ϕ(y⟨w, x⟩)] -E (x,y)∼S [ϕ(y⟨w, x⟩)](21)
+ sup w∈H(Γi,Lj ) E (x,y)∼D [ρ(y⟨w, x⟩)] -E (x,y)∼S [ρ(y⟨w, x⟩)] . (22
)
In Appendix C, we carefully use Bernstein's plus a (highly non-trivial) union bound over infinitely many grids of increasing size to bound (20) as follows Lemma 7. There is a constant c > 0 such that with probability at least 1 -δ over S ∼ D n we have
(20) ≤ c   (ℓ j+1 + exp(-γ 2 i+1 k/c))(k + ln(e/δ)) n + (k + ln(e/δ)) n   .
In Appendix B, we then use Rademacher complexity and a bound on the Lipschitz constants of ϕ and ρ to bound ( 21) and ( 22) as follows Lemma 8. There are constants c, c ′ > 0 such that when k ≥ c ′ γ -2 i+1 , it holds with probability at least
1 -δ over S ∼ D n that max{(21), (22)} ≤ c exp(-γ 2 i+1 k/c) • (k + γ -2 i+1 + ln(e/δ))/n.
To balance the expressions in Lemma 7 and Lemma 8, we now set
k = cγ -2 i+1 ln(e/ℓ j+1 ) for a sufficiently large constant c > 0 so that exp(-γ 2 i+1 k/c) ≤ ℓ j+1 /e and k ≥ c ′ γ -2 i+1 .
Combining Lemma 7 and Lemma 8 via a union bound with δ ′ = δ/2 and inserting into (20), ( 21) and ( 22) gives
sup w∈H(Γi,Lj ) L D (w) -L γ S (w) ≤ c ℓ j+1 (γ -2 i+1 ln(e/ℓ j+1 ) + ln(e/δ))/n + c ℓ j+1 (γ -2 i+1 ln(e/ℓ j+1 ) + ln(e/δ)) n + γ -2 i+1 ln(e/ℓ j+1 ) + ln(e/δ) n ,
for a constant c > 0. This completes the proof of Lemma 3, which together with Lemma 4 completes the proof of our main result, Theorem 2.
this section cite: ['b0']

Section: Conclusion
We have established the first asymptotically tight generalization bound for large-margin halfspaces, resolving a long-standing gap between upper and lower bounds in this fundamental setting. Our main theorem precisely characterizes the interplay between the margin, empirical margin loss, sample size, and confidence parameter. The proof introduces several new analytical techniques, including a refined initial analysis and a Rademacher-based treatment of randomized rounding.
Beyond settling the generalization theory of large margin halfspaces, our framework provides novel tools that may prove useful for deriving tight bounds in related models. Concretely, our techniques are in essence a refinement of the techniques introduced by Schapire et al. [1998] for proving generalization of large margin voting classifiers. For voting classifiers, the best known generalization upper (Gao and Zhou [2013]) and lower bounds (Grønlund et al. [2020b]) for finite hypothesis sets have a similar logarithmic gap as our techniques managed to remove for halfspaces. Another related topic is kernel methods, in particular in the context of support vector machines. Here there are also prior works deriving generalization bounds based on margins, see e.g. Cortes et al. [2010], however these works are not "first-order bounds", i.e. the √ • terms in the generalization bounds do not decrease with L γ S (w). We are hopeful that our techniques may also yield improvements in these areas of interest. In particular, it is worth mentioning that the original Johnson-Lindenstrauss transform Johnson and Lindenstrauss [1984] also provides dimensionality reduction from infinite-dimensional Hilbert spaces (e.g. kernel space), thus suggesting that a similar randomized discretization might be possible.
this section cite: ['b19', 'b8', 'b4', 'b12']

Section: References
Ref_id:b0 Title: Optimal compression of approximate inner products and dimension reduction Year: (2017)
Ref_id:b1 Title: Generalization performance of support vector machines and other pattern classifiers Year: (1999)
Ref_id:b2 Title: Rademacher and gaussian complexities: Risk bounds and structural results Year: (2002)
Ref_id:b3 Title: Support-vector networks Year: (1995)
Ref_id:b4 Title: Generalization bounds for learning kernels Year: (2010-06)
Ref_id:b5 Title: An elementary proof of a theorem of Johnson and Lindenstrauss Year: (2003)
Ref_id:b6 Title: A Probabilistic Theory of Pattern Recognition Year: (1996)
Ref_id:b7 Title: Large margin classification using the perceptron algorithm Year: (1999-12)
Ref_id:b8 Title: On the doubt about margin explanation of boosting Year: (2013)
Ref_id:b9 Title: Near-tight margin-based generalization bounds for support vector machines Year: (2020)
Ref_id:b10 Title: Margins are insufficient for explaining gradient boosting Year: (2020)
Ref_id:b11 Title: Faster algorithms for structured linear and kernel support vector machines Year: (2025)
Ref_id:b12 Title: Extensions of Lipschitz mappings into a Hilbert space Year: (1982)
Ref_id:b13 Title: Probability Theory: A Comprehensive Course Year: (2020)
Ref_id:b14 Title: Adaptive estimation of a quadratic functional by model selection Year: (2000)
Ref_id:b15 Title: Probability in Banach Spaces: Isoperimetry and Processes. A Series of Modern Surveys in Mathematics Series Year: (1991)
Ref_id:b16 Title: Simplified pac-bayesian margin bounds Year: (2003-08-24)
Ref_id:b17 Title: A logical calculus of ideas immanent in nervous activity Year: (1943)
Ref_id:b18 Title: Pegasos: primal estimated sub-gradient solver for svm Year: (2011)
Ref_id:b19 Title: Boosting the margin: A new explanation for the effectiveness of voting methods Year: (1998)
Ref_id:b20 Title: Understanding Machine Learning -From Theory to Algorithms Year: (2014)
Ref_id:b21 Title: High-Dimensional Statistics: A Non-Asymptotic Viewpoint Year: (2019)
