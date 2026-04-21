Title: High-Dimensional Calibration from Swap Regret
Abstract: We study the online calibration of multi-dimensional forecasts over an arbitrary convex set P ⊂ R d relative to an arbitrary norm • . We connect this with the problem of external regret minimization for online linear optimization, showing that if it is possible to guarantee O( √ ρT ) worst-case regret after T rounds when actions are drawn from P and losses are drawn from the dual • * unit norm ball, then it is also possible to obtain -calibrated forecasts after T = exp(O(ρ/ 2 )) rounds. When P is the d-dimensional simplex and • is the ℓ 1 -norm, the existence of O( √ T log d)-regret algorithms for learning with experts implies that it is possible to obtain -calibrated forecasts after T = exp(O(log d/ 2 )) = d O(1/ 2 ) rounds, recovering a recent result of [Pen25]. Interestingly, our algorithm obtains this guarantee without requiring access to any online linear optimization subroutine or knowledge of the optimal rate ρ -in fact, our algorithm is identical for every setting of P and • . Instead, we show that the optimal regularizer for the above OLO problem can be used to upper bound the above calibration error by a swap regret, which we then minimize by running the recent TreeSwap algorithm ([DDFG24, PR24]) with Follow-The-Leader as a subroutine. The resulting algorithm is highly efficient and plays a distribution over simple averages of past observations in each round. Finally, we prove that any online calibration algorithm that guarantees T ℓ 1calibration error over the d-dimensional simplex requires T ≥ exp(poly(1/ )) (assuming d ≥ poly(1/ )). This strengthens the corresponding d Ω(log 1/ ) lower bound of [Pen25], and shows that an exponential dependence on 1/ is necessary.

Section: Introduction
Consider the problem faced by a forecaster who must report probabilistic predictions for a sequence of events (e.g. whether it will rain or not tomorrow). One of the most common methods to evaluate the quality of such a forecaster is to verify whether they are calibrated: for example, does it indeed rain with probability 40% on days where the forecaster makes this prediction? In addition to calibration being a natural property to expect from predictions, several applications across machine learning, fairness, and game theory require the ability to produce online calibrated predictions [ZME20,GPSW17,HJKRR18,FV97].
When events have binary outcomes, calibration can be quantified by the notion of expected calibration error, which measures the expected distance between a prediction made by a forecaster and the actual empirical probability of the outcome on the days where they made that prediction. In a seminal result by Foster and Vohra [FV98], it was proved that it is possible for an online forecaster to efficiently guarantee a sublinear calibration error of O(T 2/3 ) against any adversarial sequence of T binary events. Equivalently, this can be interpreted as requiring at most O( -3 ) rounds of forecasting to guarantee an per-round calibration error on average. However, many applications require forecasting sequences of multi-dimensional outcomes. The previous definition of calibration error easily extends to the multi-dimensional setting where predictions and outcomes belong to a d-dimensional convex set P ⊂ R d . Specifically, if a forecaster makes a sequence of predictions p 1 , p 2 , . . . , p T ∈ P for the outcomes y 1 , y 2 , . . . , y T ∈ P, their
• -calibration error (for any norm • over R d ) is given by
Cal • T = T t=1 p t -ν pt
where ν pt is the average of the outcomes y t on rounds where the learner predicted p t .
The algorithm of Foster and Vohra extends to the multidimensional calibration setting, but at the cost of producing bounds that decay exponentially in the dimension d. In particular, their algorithm only guarantees that the forecaster achieves an average calibration error of after (1/ ) Ω(d) rounds. Until recently, no known algorithm achieved a sub-exponential dependence on d in any non-trivial instance of multi-dimensional calibration.
In 2025, [Pen25] presented a new algorithm for high-dimensional calibration, demonstrating that it is possible to obtain ℓ 1 -calibration rates of T in d O(1/ 2 ) rounds for predictions over the d-dimensional simplex (i.e., multi-class calibration). In particular, this is the first known algorithm achieving polynomial calibration rates in d for fixed constant . [Pen25] complements this with a lower bound, showing that in the worst case d Ω(log 1/ ) rounds are necessary to obtain this rate (implying that a fully polynomial bound poly(d, 1/ ) is impossible).
this section cite: ['b36', 'b15', 'b18', 'b12', 'b13', 'b28', 'b28']

Section: Our results
Although the algorithm of [Pen25] is simple to describe, its analysis is fairly nuanced and tailored to ℓ 1 -calibration over the simplex (e.g., by analyzing the KL divergence between predictions and distributions of historical outcomes). We present a very similar algorithm (TreeCal) for multidimensional calibration over an arbitrary convex set P ⊂ R d , but with a simple, unified analysis that provides simultaneous guarantees for calibration with respect to any norm • . In particular, we prove the following theorem. Theorem 1.1 (Informal restatement of Corollary C.5). Fix a convex set P and a norm • . Assume there exists a function R : P → R that is 1-strongly-convex with respect to • and has range (max x∈P R(x) -min p∈P R(x)) at most ρ. Then TreeCal guarantees that the calibration error of its predictions is bounded by Cal
• T ≤ T for T ≥ (diam • (P)/ ) O(ρ/ 2 ) .
Interestingly, the function R(p) and parameter ρ appearing in the statement of Theorem 1.1 have an independent learning-theoretic interpretation: if we consider the online linear optimization problem where a learner plays actions in P and the adversary plays linear losses that are unit bounded in the dual norm • * , then it is possible for the learner to guarantee a regret bound of at most O( √ ρT ) by playing Follow-The-Regularized-Leader (FTRL) with R(p) as a regularizer. In fact, since universality results for mirror descent guarantee that some instantiation of FTRL achieves near-optimal rates for online linear optimization (as long as the action and loss sets are centrally convex) [SST11,GSJ24], this allows us to relate the performance of Theorem 3.1 directly to what rates are possible in online linear optimization. Corollary 1.2 (Informal restatement of Corollary C.6). Let P ⊆ R d be a centrally symmetric convex set, and let L = {y ∈ R d | y * ≤ 1} for some norm • . Then if there exists an algorithm for online linear optimization with action set P and loss set L that incurs regret at most O( √ ρT ), TreeCal guarantees that the calibration error of its predictions is bounded by Cal • T ≤ T for T ≥ (diam • (P)/ ) O(ρ/ 2 ) . Theorem 1.1 and its corollary allow us to immediately recover several existing and novel bounds on calibration error in a variety of settings:
• When P is the d-simplex ∆ d and • is the ℓ 1 -norm, the existence of the negative entropy regularizer R(x) = d i=1 x i log x i (which is 1-strongly convex w.r.t. the ℓ 1 norm with range ρ = log d) implies that the ℓ 1 calibration error of TreeCal is at most (1/ ) O(log d/ 2 ) = d Õ(1/ 2 ) . This recovers the result of [Pen25].
• When P is the ℓ 2 ball and • is the ℓ 2 norm, the Euclidean regularizer (R(x) = x 2 ) implies a calibration bound of (1/ ) O(1/ 2 ) (notably, this bound is independent of d).
It should be emphasized here that running TreeCal does not require any online linear optimization subroutine, nor any knowledge of these regularizers R(x) or optimal rates ρ. TreeCal has no functional dependence on any specific • . It achieves • -calibration at the above rate (Theorem 1.1) for all • simultaneously. The TreeCal algorithm is nearly identical 5 to the algorithm of [Pen25] both algorithms initialize a tree of sub-forecasters and at each round play a uniform combination of some subset of them (see Figure 1).
The novelty in our analysis stems from the observation that TreeCal is simply a specific instantiation of the TreeSwap swap regret minimization algorithm [DDFG24,PR24] and can be analyzed directly in this way. In particular, our analysis consists of the following steps:
1. First, minimizing calibration error can be reduced to minimizing swap regret, generalizing an idea of [LSS25, FKO + 25]. That is, it is possible to assign the learner loss functions ℓ t : P → R at each round such that their calibration error is upper bounded by the gap between the total loss they received, and the minimal loss they could have received after applying an arbitrary "swap function" π : P → P to their predictions. In fact, any strongly convex function R (w.r.t. the norm • ) gives rise to one such reduction, by setting the loss function ℓ t (p) to equal the Bregman divergence D R (y t |p).
2. Second, the TreeSwap algorithm of [DDFG24,PR24] provides a general recipe for converting external regret minimization algorithms into swap regret minimization algorithms. We obtain TreeCal by plugging in the Follow-The-Leader algorithm (the learning algorithm which simply always best responds to the current history) into TreeSwap.
3. Instead of analyzing the swap regret bound of TreeSwap with Follow-The-Leader (which may not have a good enough external regret bound, as discussed in Section 3.3), we instead analyze the swap regret of TreeSwap with Be-The-Leader (the fictitious algorithm that best responds to the current history, including the current round). Though it is not possible to actually implement Be-The-Leader due to its clairvoyance, we use it as a tool for analysis. We then relate the calibration error of TreeSwap with Be-The-Leader to that of TreeSwap with Follow-The-Leader using the fact that Be-The-Leader and Follow-The-Leader make similar predictions.
In the above step 1, we will choose R to be • -norm 1-strongly convex, which guarantees that D R (y|p) ≥ y -p 2 . Going through the analysis, this actually leads to the stronger guarantee that TreeCal minimizes squared-norm calibration error. Theorem 1.3 (Informal restatement of Theorem 3.1). Fix a convex set P and a norm • . Assume there exists a function R : P → R that is 1-strongly-convex with respect to • and has range (max x∈P R(x) -min p∈P R(x)) at most ρ. Then TreeCal guarantees that the calibration error of its predictions is bounded by Cal
• 2 T ≤ T for T ≥ (diam • (P)/ √ ) O(ρ/ ) .
Note here we have only singly-exponential dependence on 1/ . We arrive at Theorem 1.1 as a corollary of this result by simply applying Cauchy-Schwarz. Finally, we strengthen the lower bound of [Pen25] by showing an exponential dependence on 1/ is necessary.
this section cite: ['b28', 'b35', 'b16', 'b28', 'b28', 'b6', 'b29', 'b6', 'b29', 'b28']

Section: Theorem 1.4 (Informal restatement of Theorem 4.3).
There is a sufficiently small constant c > 0 so that the following holds. Fix any > 0, d ∈ N. Then for any T ≤ exp(c • min{d 1/14 , -1/6 }), there is an oblivious adversary producing a sequence of outcomes so that any learning algorithm must incur ℓ 1 -calibration error Cal
• 1 T ≥ • T.
5 One minor difference is that the algorithm of [Pen25] regularizes each sub-forecaster by slightly mixing their prediction with the uniform distribution, which TreeCal does not require.
Unlike the lower bound of [Pen25], this lower bound requires no specialized construction. Instead, it follows from the original observation of [FV98] that any algorithm for online calibration can be used to construct an algorithm for swap regret minimization by simply best responding to a sequence of calibrated predictions of the adversary's losses. The existing lower bound for swap regret in [DFG + 24] then immediately precludes the existence of sufficiently strong calibration bounds (e.g., of the form d O(log 1/ ) , which was still allowed by the work of [Pen25]).
Using a similar technique, in Theorem D.2, we show a similar lower bound for ℓ 2 calibration, namely that exp(Ω(min{d 1/14 , -1/7 })) time steps are needed to achieve ℓ 2 calibration error at most • T . For d ≥ -2 , this bound is tight up a polynomial in the exponent.
We discuss additional related work in the appendix.
this section cite: ['b28', 'b28', 'b13', 'b28']

Section: Setup
For a positive integer n, we let [0 : n -1] denote the sequence 0, 1, . . . , n -1, and [n] denote the sequence 1, 2, . . . , n. We say a convex set S ⊆ R d is centrally symmetric if s ∈ S ⇔ -s ∈ S for all s ∈ R d . A norm • is a function corresponding to a convex, bounded, centrally-symmetric set S of the form s = inf {c ∈ R ≥0 |s ∈ cS}. The corresponding dual norm is defined v * = sup { s, v | s ≤ 1}.
this section cite: []

Section: Calibration
We consider the following setting of multi-dimensional calibration. Positive integers d ∈ N representing the number of dimensions and T ∈ N representing the number of rounds are given. We let P ⊂ R d denote a bounded convex subset of R d . An adversary and a learning algorithm interact for a total of T timesteps; at each time step t ∈ [T ]:
• The learning algorithm chooses a distribution 6 x t ∈ ∆(P) with finite support.
• The adversary observes x t and chooses an outcome y t ∈ P.
In order for the learner to be calibrated, we would like the average outcome conditional on the learner making a specific prediction p to be "close" to p. We formalize this as follows. For a point p ∈ P, we define ν p to be the average outcome conditioned on the learner predicting p, that is:
ν p := T t=1 x t (p) • y t T t=1 x t (p) .(1)
Fix a distance measure D : P × P → R ≥0 , namely an arbitrary non-negative valued function on P × P. Given a distance measure D, we define the D-calibration error as follows:
Cal D T (x 1:T , y 1:T ) := p∈P T t=1 x t (p) • D(ν p , p).
In the event that D(p, q) = p -q , we will write Cal
• T (x 1:T , y 1:T ) = Cal D T (x 1:T , y 1:T ), and we define Cal • 2 T (x 1:T , y 1:T ) analogously.
this section cite: []

Section: Regret minimization
For a sequence of actions p 1 , • • • , p T ∈ P and loss functions ℓ 1 , • • • , ℓ T : P → R, we define ExtReg T (p 1:T , ℓ 1:T ) := sup
p * ∈P T t=1 p∈P ℓ t (p t ) -ℓ t (p * )
6 Some authors refer to this setting as "pseudo-calibration" or "distributional calibration", and reserve the term "calibration" for the setting where the learner is required to randomly select a pure forecast pt ∈ P each round instead of a distribution. In Appendix E we describe how to extend our results to this pure-strategy setting of calibration.
For a sequence of distributions x 1 , • • • , x T ∈ ∆(P) and loss functions ℓ 1 , • • • , ℓ T : P → R, we define FullSwapReg T (x 1:T , ℓ 1:T ) := sup π:P→P T t=1 p∈P
x t (p) • (ℓ t (p) -ℓ t (π(p))).
(
Here, we adopt the convention of [FKO + 25], referring to the latter quantity as Full Swap Regret to emphasize that we consider all swap transformations π : P → P (instead of e.g. just linear transformations π).
Throughout, we consider the performance of regret minimizing algorithms. These algorithms sequentially map loss functions ℓ 1 , • • • , ℓ T to actions p 1 , • • • , p T or action distributions x 1 , • • • , x T with the goal of minimizing the above quantities. We consider the performance of these algorithms on adversarially selected loss functions from a set L. Abusing notation slightly, for an external regret minimizing algorithm Alg : L T → P T , we define ExtReg T (Alg) := sup
ℓ 1:T ∈L T ExtReg T (Alg(ℓ 1:T ), ℓ 1:T )(3)
and for a full swap regret minimizing algorithm Alg : L T → ∆(P) T , we define FullSwapReg T (Alg) := sup ℓ 1:T ∈L T FullSwapReg T (Alg(ℓ 1:T ), ℓ 1:T ) .
We will denote the tth action played by Alg on a sequence of losses ℓ 1:T by Alg t (ℓ 1:T ). One important subclass of external regret minimization problems is the setting of online linear optimization (OLO), where all loss functions in ℓ are linear. Here we slightly abuse notation and identify L with a subset of R d (with the understanding that an element ℓ ∈ L refers to the linear loss function ℓ(p) = p, ℓ ). Although we will never actually employ any OLO algorithms themselves, the calibration bounds we obtain will be closely related to optimal regret bounds for instances of OLO (we discuss this further in Section 2.4).
this section cite: []

Section: From swap regret to calibration
As noted in [LSS25, FKO + 25], calibration with a distance measure D that corresponds to a Bregman divergence can be written as a full swap regret with loss functions given by the associated proper scoring rule. Given a convex function R : P → R, the Bregman divergence associated to R, D R : P × P → R ≥0 , is defined asfoot_0 D R (y|p) := R(y) -R(p) -∇R(p), y -p Geometrically, this divergence is defined by taking the hyperplane tangent to R at p and computing the difference in height between R and the hyperplane at y (see Figure 2).
When viewed as a loss function in p, the Bregman divergence D R (y|p) also has the property that it is a proper scoring rule. This refers to the fact that if y is drawn from some distribution y ∈ ∆(P), the optimal response p (to minimize the expected loss D R (y|p)) is simply the expectation ȳ = E y∼y [y].
In particular, we have the following lemma. Lemma 2.1. For any y ∈ ∆(P) and convex function R : P → R, let ȳ = E y∼y [y]. and R(y) = E y∼y [R(y)]. For all p ∈ P, E y∼y [D R (y|p)] = D R (ȳ|p) + R(y) -R(ȳ). In particular, ℓ(p) = E y∼y [D R (y|p)] is minimized at p = ȳ at a value of R(y) -R(ȳ) (Figure 3).
This implies the following connection between full swap regret and calibration. Lemma 2.2. Fix any convex function R : P → R. For any sequence of distributions x 1 , x 2 , . . . , x T ∈ ∆(P) and outcomes y 1 , y 2 , . . . , y T ∈ P, define the sequence of loss functions ℓ 1 , ℓ 2 , . . . , ℓ T via ℓ t (p) = D R (y t |p). Then, FullSwapReg T (x 1:T , ℓ 1:T ) = Cal D R T (x 1:T , y 1:T ).
The proofs of Lemmas 2.1 and 2.2 may be found in Appendix B.
this section cite: []

Section: Rates and regularization
In order to reduce our general calibration problem to a swap regret minimization problem (via Lemma 2.2), we will need to construct a convex function R whose Bregman divergence upper bounds our distance measure. It turns out that the optimal choice of such a function is closely related to the design of optimal regularizers for online linear optimization. In this section, we describe this functional optimization problem and detail this connection.
We say that a convex function R : P → R is α-strongly convex with respect to a given norm • if for any points y, p ∈ P it is the case that R(y) ≥ R(p) + ∇R(p), y -p + α y -p 2 . Equivalently, the Bregman divergence must satisfy D R (y|p) ≥ α y -p 2 . Thus, • 2 -calibration error is bounded by D R -calibration error if R is • -norm 1-strongly convex.
Our later analysis will need not only R to be strongly convex with respect to our norm, but for the Bregman divergence to have a small maximal value. Motivated by this, we will say that a convex function R : P → R has rate ρ with respect to a given norm • if: (1) R is 1-strongly convex with respect to • , and (2) the range of the Bregman divergence is at most ρ, i.e., max y,p∈P D R (y|p) ≤ ρ. We define Rate(P, • ) to be the infimum of the rates of all 1-strongly convex functions R : P → R.
As mentioned earlier, we call this quantity a "rate" due to its connection with the optimal regret rates for online linear optimization. For a learning algorithm Alg : L T → P T , we defined (in (3)) ExtReg T (Alg) to be the worst-case regret against any sequence ℓ 1:T of T losses. It is known that for any fixed action set and loss set, the optimal worst-case regret bound is of the form Rate OLO (P, L) • T + o( √ T ), for some constant Rate OLO (P, L). Formally, we define Rate OLO (P, L) = lim sup T →∞ inf Alg 1 T • ExtReg T (Alg) 2 . One important class of learning algorithms for online linear optimization is the class of Follow-The-Regularized-Leader (FTRL) algorithms. Each algorithm in this class is specified by a convex "regularizer" function R : P → R, and at round t selects the action p t = argmin p∈P t-1 s=1 p, ℓ t + R(p). The work of [SST11] and [GSJ24] shows that there always exists some instantiation of FTRL which achieves (up to a universal constant factor) the optimal regret rate of Rate OLO (P, L) • T + o( √ T ) defined above. Moreover, the optimal regularizer for this instance can be constructed by solving a similar functional optimization problem over strongly convex regularizers R, as described in the following theorem. Theorem 2.3. Let P and L be centrally symmetric convex sets. Then, if the function R : P → R is 1strongly-convex with respect to the norm • L * and has range ρ (i.e., max p∈P R(p)-min p∈P R(p) = ρ), then Rate OLO (P, L) ≤ ρ. Conversely, there exists a function R : P → R that is 1-strongly-convex with respect to • L * and has range O(Rate OLO (P, L)).
Proof. The first result (that Rate OLO (P, L) ≤ ρ) follows from the standard analysis of FTRL -see e.g. Theorem 5.2 in [H + 16]. The converse result follows from Theorem 2 of [GSJ24].
Theorem 2.3 allows us to relate the quantity Rate(P, • ) to the quantity Rate OLO (P, L) (where L is chosen to be the unit dual norm ball). Note that there is a slight difference in the two functional optimization problems defined above -the one for Rate(P, • ) asks us to bound the range of the Bregman divergence of R, while the one for Rate OLO (P, L) asks us to bound the range of R itself. While these two quantities do not directly bound each other (the negative entropy function R(p) = p i log p i has bounded range over the simplex but unbounded Bregman divergence), we can nonetheless show that optimal solutions to one problem can be used to construct optimal solutions to the other problem of similar quality. Lemma 2.4. If the action set P is centrally symmetric and L = {y ∈ R d | y * ≤ 1} (i.e., the unit ball in the dual norm to • ), then Rate OLO (P, L) = Θ(Rate(P, • )).
this section cite: ['b35', 'b16', 'b16']

Section: Main result
We now describe our main algorithm for calibration, TreeCal (Algorithm 1). As we will see, it is equivalent to the TreeSwap algorithm for Full Swap Regret minimization ([DDFG24, PR24]; Algorithm 2), where the loss functions are given by appropriate Bregman divergences as determined by Lemma 2.2. Moreover, TreeCal is effectively the same as the main algorithm of [Pen25]. However, the perspective that TreeCal can be viewed as a particular instance of TreeSwap (Lemma 3.2) is novel to this work, and it enables us to tackle a much more general set of calibration problems (Theorem 3.1). We first describe the TreeCal and TreeSwap algorithms, then state Theorem 3.1 which establishes our main upper bound for TreeCal, and finally discuss the proof of Theorem 3.1, which uses the TreeSwap algorithm as a tool in the analysis.
this section cite: ['b28']

Section: Algorithm description
Given some number of rounds T ∈ N, TreeCal and TreeSwap sequentially produce distributions x 1 , • • • , x T ∈ ∆(P). TreeCal receives from the adversary an outcome sequence y 1 , • • • , y T ∈ P whereas TreeSwap receives loss functions ℓ 1 , • • • , ℓ T : P → R.
To describe how the algorithms use the adversary's actions to produce the distributions x t , we need some additional ntation. The algorithms take as input parameters H, L ∈ N satisfying H ≥ 2 and H L-1 ≤ T ≤ H L . We index time steps t ∈ [T ] via base-H L-tuples: in particular, for t ∈ [T ], we let t 1 , . . . , t L ∈ [0 : H -1] be the base-H representation of t -1; we will write t -
1 = (t 1 t 2 • • • t L ). For all 0 ≤ l ≤ L, for all k ∈ [0 : H -1] l , let Γ (l) k ⊂ [T ] represent the interval of times t with prefix k. That is, t ∈ Γ (l) k iff t i = k i for all i ∈ [1 : l].
These intervals may be arranged to form an H-ary depth-L tree, where the children of
Γ (l) k are Γ (l+1) k0 , Γ (l+1) k1 , • • • , Γ (l+1) k,H-1 . 8
Both TreeCal and TreeSwap operate by assigning an action p (l) k to each node Γ (l) k of the tree, except the root. At time t, both algorithms return the uniform distribution over the actions on the root-to-leaf-t path, namely x t := Unif p (1)
t1 , p (2) t1t2 , • • • , p (L) t1t2•••t L
(see Figure 1).
The algorithms differ in how the actions p (l) k are chosen: p (1) 0 p (1) 1 p (2) 00 p (2) 01 p (2) 02 p (2) 10 p (2) 11 p (3) 000 p (3) 001 p (3) 002 p (3) 010 p (3) 011 p (3) 012 p (3) 020 p (3) 021 p (3) 022 p (3) 100 p (3) 101 p (3) 102 p (3) 110 p (3) 111 . . . For H = 3, we depict the intervals Γ of the first three non-root levels of the tree (l = 1, 2, 3). Each rectangular node represents an interval, with sibling nodes separated by red lines. We represent the specific time step t via the vertical dashed green line. The yellow intervals it intersects at each level correspond to the nodes on the root-to-leaf-t path. Accordingly, xt will be the uniform distribution over the labels p of these yellow intervals. We see that the algorithm has committed to the labels of all intervals that started at or before time t, and has yet to label the future intervals.
• TreeCal (Algorithm 1) assigns actions to nodes as follows. For all 1 ≤ l ≤ L, k ∈ [0 :
H -1] l-1 , h ∈ [0 : H -1], at the start of Γ (l) kh , TreeCal sets p (l)
kh to be the average over all y t that have been observed thus far in the parent interval
Γ (l-1) k . That is, p (l) kh = 1 hH L-l h-1 i=0 t∈Γ (l) ki y t (4
)
• The more general TreeSwap algorithm (Algorithm 2) also takes as a parameter an external regret-minimizing algorithm Alg, which operates with horizon of length H: we denote the resulting algorithm by TreeSwap.Alg. TreeSwap.Alg associates each internal node of the tree, Γ (l-1) k (with 1 ≤ l ≤ L), with an instance Alg, denoted Alg (l-1) k
. The subroutine Alg (l-1) k is responsible for choosing the actions p
(l) k0 , p (l) k1 , • • • , p (l)
k(H-1) . It does so by responding to the average losses over each of its child intervals. In particular: at the end of each child interval Γ (l) kh , we pass Alg (l-1) k the average loss over that interval. Alg (l-1) k then outputs the action p (l) k(h+1) assigned to the next child interval.
this section cite: []

Section: Main result
Theorem 3.1 upper bounds the calibration error of TreeCal with respect to the squared norm • 2 .
Theorem 3.1 (Main theorem). Let P ⊂ R d be a bounded convex set and • be an arbitrary norm. Then, TreeCal (Algorithm 1) guarantees that for an arbitrary sequence of outcomes y 1 , . . . , y T ∈ P, the • 2 calibration error of its predictions x 1 , . . . , x T ∈ ∆(P) is bounded as follows:
Cal • 2 T (x 1:T , y 1:T ) ≤ T for T ≥ (diam(P)/ √ ) O(Rate(P, • )/ )
It is straightforward to derive from Theorem 3.1 via an application of Jensen's inequality an upper bound on the calibration error of TreeCal with respect to the (non-squared) norm • , as stated in Theorem 1.1; see Corollary C.5. In Appendix E, we additionally consider a variant of TreeCal which plays pure actions in P (i.e., not distributions) by sampling from the distributions x t for each t ∈ [T ]. We show that the pure calibration error of this variant can be bounded by a similar quantity to that in Theorem 3.1.
this section cite: []

Section: Outline of the proof of Theorem 3.1
Step 1: Reduction from calibration error to swap regret. Let us choose a convex function R : P → R given P, • as described in Section 2.4. The first step in the proof of Theorem 3.1 is to reduce the problem of minimizing (squared-norm) calibration error to that of minimizing full swap regret for an appropriate sequence of loss functions. In particular, for any sequence x 1 , . . . , x T ∈ ∆(P) and y 1 , . . . , y T ∈ P, we have
Cal • 2 T (x 1:T , y 1:T ) ≤ Cal D R T (x 1:T , y 1:T ) = FullSwapReg R (x 1:T , ℓ 1:T ),(5)
where ℓ t : P → R is the loss function given by ℓ t (p) := D R (y t |p): the inequality uses strong convexity of R, and the subsequent equality uses Lemma 2.2.
this section cite: []

Section: Step 2: Equivalence with
TreeSwap. Thus, it suffices to find an algorithm which minimizies the full swap regret quantity on the right-hand side of (5). Fortunately, the TreeSwap algorithm is known to do exactly this! (See Theorem C.1, from [DDFG24], for a formal statement for the swap regret bound of TreeSwap.) In order to apply the swap regret bound of Theorem C.1, we need to ensure that the TreeCal algorithm is an instantiation of TreeSwap.Alg for an appropriate choice of (a) the loss functions fed as input to TreeSwap and (b) the Alg subroutine. The loss functions have already been defined: given a sequence y 1 , . . . , y T , recall that we chose ℓ t (p) := D R (y t |p). Moreover, we let the Alg subroutine be given by Follow-the-Leader (FTL), which simply chooses an action at each step minimizing the sum of losses up to the previous time step. The following lemma shows that TreeSwap with the losses ℓ t and the FTL subroutine produces the same action distributions as TreeCal: Lemma 3.2. Let P ⊂ R d be a bounded convex set and let R : P → R be a convex function. For a sequence of loss functions
ℓ 1 , • • • , ℓ H : P → R, define FTL h (ℓ 1:H ) = arg min p∈P h-1 s=1 ℓ s (p)
. For all sequences of outcomes y 1:T ∈ P T , the action distributions x t produced by TreeCal on y 1:T equal those produced by TreeSwap.FTL on loss functions ℓ t (p) = D R (y t |p) for all t.
The proof of Lemma 3.2 (given in full in the appendix) is a straightforward consequence of the fact that the Bregman divergence is a proper scoring rule: the action p ∈ P minimizing an average of Bregman divergences D R (y|p) is simply the average of the constituent points y (Lemma 2.1).
Step 3: Applying the swap regret bound of TreeSwap to BTL. Finally, we want to apply the main result of [DDFG24] (restated as Theorem C.1) to bound the full swap regret for the iterates x 1:T produced by TreeSwap.Alg, for an appropriate choice of Alg. The most natural way to do so would be to try to directly apply this result in the case when Alg = FTL (which corresponds to how we actually implement TreeSwap). However, applying this theorem requires an external regret bound on FTL for an arbitrary sequence of losses. While FTL is known to possess strong external regret bounds in some situations (e.g., when all the loss functions are strongly convex), the loss functions p → D R (y|p) are not necessarily even convex in p and so it is not a priori clear how to establish such bounds.
Instead, the main idea is to consider the "Be-The-Leader" algorithm BTL, which is the same as FTL but where actions are shifted ahead in time by 1 time step: in particular, the action chosen by BTL at time step h given a sequence ℓ 1 , ℓ 2 , . . . , ℓ H : P → R is BTL h (ℓ 1:H ) = FTL h+1 (ℓ 1:H ) = argmin p∈P h s=1 ℓ s (p). BTL is not implementable since its action at time step h depends on the (unobserved) loss ℓ h at that time step. However, since its regret is always non-positive (i.e., ExtReg H (BTL) ≤ 0 for any H), if we apply Theorem C.1 to the algorithm TreeSwap.BTL, we get that FullSwapReg T (TreeSwap.BTL) ≤ • T as long as T ≥ H O(ρ/ ) for any choice of H (the arity parameter H used in TreeSwap). Using (5), this implies that the calibration error of the iterates produced by TreeSwap.BTL can also be bounded above by • T .
Of course, this result on its own is uninteresting (since BTL is unimplementable, as mentioned above). However, the key insight is that we can show that the actions chosen by TreeSwap.BTL are close to (as measured by the norm • ) those chosen by TreeSwap.FTL, which in turn is equivalent to TreeCal (Lemma 3.2). This closeness is an immediate consequence of the fact that the actions chosen by FTL for our loss functions D R (y 1 |•), D R (y 2 |•), . . . are simply the empirical average of all actions y 1 , y 2 , . . . ∈ P of the adversary up to the previous time step. 9 In turn, we can use this closeness to show that the calibration error of TreeSwap.FTL is close to that of TreeSwap.BTL. This latter part of the argument becomes slightly tricky due to the possibility that different nodes of the tree might output the same action p ∈ P; accordingly, we need to work with a labeled variant of the action set and bound the swap regret over this labeled variant; see Appendix C for further details.
this section cite: ['b6', 'b6']

Section: Lower bound
To prove our calibration lower bound, we make use of the following swap regret lower bound. Theorem 4.1 (Theorem 4.1 of [DFG + 24]). There is a sufficiently small constant c 4.1 > 0 so that the following holds. Fix any > 0. For any d ∈ N, there is a subset X ⊂ [-1, 1] d so that the following holds for any T ≤ exp c 4.1 min{d 1/14 , -1/6 } . There is an oblivious adversary producing a sequence v 1 , . . . , v T with v t 1 ≤ 1 and v t ∞ ≤ max{d -13/14 , 13/6 } for all t, which satisfies the following property. For linear loss functions ℓ(x, v) = v, x for vectors v ∈ R d and x ∈ R d , any learning algorithm producing x 1 , . . . , x T ∈ ∆(X ),
FullSwapReg T (x 1:T , ℓ(•, v 1:T )) = sup π:X →X T t=1 p∈X x t (p) • ( v t , p -v t , π(p) ) ≥ • T.
We leverage the classic reduction from swap-regret minimization to calibration [FV98]: by producing calibrated predictions of the upcoming loss and best-responding to it, we can effectively minimize swap regret. This is formalized in the following lemma, proved in Appendix D. Lemma 4.2. Fix a set P ⊂ R d , a norm • , and write D(p, p ) := p -p . Suppose that, for some > 0, T ∈ N, there is an algorithm which chooses x 1 , . . . , x T ∈ ∆(P) and which ensures that for every oblivious adversary choosing y 1 , . . . , y T ∈ P, we have Cal D T (x 1:T , y 1:T ) ≤ • T . Then for every set P ⊂ R d , there is an algorithm which chooses x 1 , . . . , x T ∈ ∆(P ) and which ensures that for every oblivious adversary choosing y 1 , . . . , y T ∈ P, we have
FullSwapReg T (x 1:T , ℓ(•, y 1:T )) ≤ • T • diam • (P ).
Combining these two ideas, we demonstrate that an algorithm -calibrated predictions of outcomes on the simplex in T ≤ exp(poly(1/ )) rounds could be used in Lemma 4.2 to achieve a swap regret algorithm contradicting Theorem 4.1. This gives the following (proved in Appendix D). 9 An observant reader might note that this same argument also lets us provide bounds on the regret of FTL for these losses. One subtlety in the analysis is that we obtain better calibration bounds by bounding the distance between the predictions of FTL and BTL in the • norm rather than in the losses DR(yt|•), and so it is important that we directly analyze TreeSwap.BTL instead of TreeSwap.FTL (the latter causes us to pick up an extra factor related to the smoothness of R).
this section cite: ['b13']

Section: Theorem 4.3.
There is a sufficiently small constant c > 0 so that the following holds. Write D(p, p ) = p -p 1 , and fix any > 0, d ∈ N. Then for any T ≤ exp(c • min{d 1/14 , -1/6 }), there is an oblivious adversary producing a sequence y 1 , . . . , y T ∈ ∆ d so that for any learning algorithm producing x 1 , . . . , x T ∈ ∆(∆ d ), Cal D T (x 1:T , y 1:T ) ≥ • T.
In Theorem D.2 (see Appendix D.2), we show a similar lower bound for ℓ 2 calibration over the unit ℓ 2 ball.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: We prove all stated claims.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: We discuss limitations.
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
Answer: [Yes] Justification: We prove all theorems and lemmas.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: The paper does not include experiments requiring code.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: The paper does not include experiments.
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
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: 9.

this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
this section cite: []

Section: Answer: [NA]
Justification: There is no societal impact of the work performed.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA] Justification: The paper does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: A Additional Related Work
There is a large range of other existing work on online (sequential) calibration [Daw82, FV97, FV98, QV21, DDF + 24, Har22, Fos99, FL99, KF08, MSA07, MS10, AM11, HK12, FH18, LSS24, NRRX23, KLST23, GJRR24, QZ24, ACRS25]. We briefly survey some of these areas below.
this section cite: []

Section: Binary outcomes.
For binary outcomes (i.e., one-dimensional calibration), classical results of [FV97, Fos99, BM07, AM11] demonstrate that it is possible to efficiently guarantee O(T 2/3 ) ℓ 1calibration. The optimal possible rates for ℓ 1 -calibration remain a major unsolved problem in online learning. Recently [QV21] improved over the naive lower bound of Ω( √ T ) by demonstrating a lower bound of Ω(T 0.528 ); this was further improved to Ω(T 0.543 ) by [DDF + 24], who also improved on the upper bound, demonstrating the existence of an algorithm with O(T 2/3-) calibration for some constant > 0.
Calibration and swap regret. The connection between calibration and swap regret has been acknowledged since the earliest works on swap regret. For example, the earliest algorithms for minimizing swap regret worked by best responding to online calibrated predictions [FV97] (later algorithms for swap regret minimization, such as [BM07] and [DDF + 24] obtain better swap regret bounds by side-stepping the need to generate calibrated predictions). In the other direction, several works minimize calibration via relating it to a swap regret that can then be minimized [FKO + 25, LSS25, AM11, Fos99].
Other forms of calibration. Due to the difficulty of minimizing (high-dimensional) calibration, there has been a line of work on designing forecasting algorithms that minimize weaker forms of calibration that recover some of the important guarantees of calibration (e.g., trustworthy-ness by a decision-maker). These include distance from calibration [BGHN23, QZ24, ACRS25], omniprediction error / U-calibration [KLST23, LSS24, GJRR24], calibration conditioned on downstream outcomes [NRRX23], and prediction for downstream swap regret [RS24,HW24]. Other work focuses on minimizing notions of calibration designed to lead to specific classes of equilibria, e.g. weak calibration [HK12], deterministic calibration [KF08], and smooth calibration [FH18]. Proof of Lemma 2.1.
this section cite: ['b30', 'b12', 'b3', 'b27', 'b32', 'b20', 'b19', 'b21', 'b8']

Section: B Proofs of preliminary results
E y∼y [D R (y|p)] = E y∼y [R(y) -R(p) -∇R(p), y -p ] = R(y) -R(p) -∇R(p), ȳ -p = D R (ȳ|p) + R(y) -R(ȳ)
See Figure 3 for a visual proof. Proof of Lemma 2.2. Fix any p ∈ P, and consider the quantity
p, R(p) y1, R(y1) y2, R(y2) ȳ, R(y) ȳ, R(ȳ) ȳ, ∇R(p), ȳ-p + R(p) R(y) -R(ȳ) DR(ȳ | p)
max p * ∈P t x t (p)(D R (y t |p) - D R (y t |p *
)). By considering the distribution y that has weight x t (p)/ t x t (p) on y t , Lemma 2.1 implies that this quantity is maximized when p * = ν p = ( t x t (p)y t )/( t x t (p)). At this optimal value of p * , this quantity can be rewritten as:
t x t (p)(D R (y t |p) -D R (y t |ν p )) = t x t (p) [(R(y t ) -R(p) -∇R(p), y t -p ) -(R(y t ) -R(ν p ) -∇R(ν p ), y t -ν p )] = t x t (p) [(R(ν p ) -R(p) -∇R(p), ν p -p ) + ∇R(ν p ) -∇R(p), y t -ν p ] = t x t (p)D R (ν p |p) + ∇R(ν p ) -∇R(p), t x t (p)(y t -ν p ) = t x t (p)D R (ν p |p).
(Here the last term vanishes since t x t (p)y t = t x t (p)ν p ). We therefore have that:
FullSwapReg T (x 1:T , ℓ 1:T ) = sup π:P→P T t=1 p∈P x t (p) • (ℓ t (p) -ℓ t (π(p))) = p∈P max p * ∈P T t=1 x t (p) • (ℓ t (p) -ℓ t (p * )) = p∈P max p * ∈P T t=1 x t (p) • (D R (y t |p) -D R (y t |ν p )) = p∈P t x t (p)D R (ν p |p) = Cal D R T (x 1:T , y 1:T ).
Proof of Lemma 2.4. Note that if we define L = {y ∈ R d | y * ≤ 1} to be the unit dual norm ball for some norm • , then by duality the norm • L * corresponding to L * is simply the original norm • . It therefore suffices to show that given a 1-strongly convex function R with bounded range ρ, it is possible to construct a 1-strongly convex function R with bounded Bregman divergence O(ρ) (and vice versa). Assume R(p) is 1-strongly convex and satisfies
max p∈P R(p) -min p∈P R(p) = ρ. Define R (p) = 4R p
2 (since P is centrally symmetric, p/2 is guaranteed to belong to P). If R is 1-strongly convex, then R(p/2) is 1/4-strongly convex, and so R (p) is also 1-strongly convex. We claim the maximum Bregman divergence of R is at most O(ρ). To show this, we first argue that for any z 1 , z 2 ∈ P, ∇R( z12 ), z 2 ≤ 2ρ. To see this, note that since R(p) is convex and has range bounded by ρ, we have that
ρ ≥ R(p) -R( z1 2 ) ≥ ∇R( z1 2 ), x -z1 2 . If we set p = z1+z2 2 , it then follows that ∇R( z1 2 ), z 2 ≤ 2ρ. Now, note that max y,p∈P D R (y|p) = R (y) -R (p) -∇R (p), y -p = R y 2 -R p 2 - 1 2 ∇R p 2 , y -p ≤ R y 2 -R p 2 + ∇R p 2 , y -p 2 ≤ 3ρ. Conversely, if R(p) is 1-strongly convex and satisfies max y,p∈P D R (y|p) ≤ ρ, define R (p) = R(p) -∇R(0), p -R(0) (i.e.
, subtracting a linear function to make zero a minimizer of R (p)). Since R and R differ by a linear function, R is also 1-strongly convex. But also, note that
D R (y|0) = R(y) -R(0) -∇R(0), y = R (y); since D R is bounded in range by ρ, it follows that so is R .
this section cite: []

Section: C Proof of Theorem 3.1
In this section, we prove Theorem 3.1. First, in Appendix C.1, we introduce a slightly stronger notion of calibration error and swap regret to deal with a technicality in the proof. We then give the proof of Theorem 3.1.
this section cite: []

Section: C.1 Labeled calibration and swap regret
Intuition. Recall that the TreeCal algorithm labels each interval Γ
k of the tree with some action, p Write the base-H representation of t -
1 as t = (h 1 • • • h L ), for h 1 , . . . , h L ∈ [0 : H -1]. 3: for 1 ≤ l ≤ L do 4: Write k := (h 1 • • • h l-1 ) ∈ [0 : H -1] l-1 . 5: if h l+1 = • • • = h L = 0 or l = L then 6: If h l > 0, define ν (l) k,h l -1 := 1 H L-l • s∈Γ (l) k,h l -1 y s . 7: Define p (l) k,h l := 1 h l h l -1 i=0 ν (l) k,i if h l > 0, otherwise choose arbitrary p(l)
k,h l ∈ P.
this section cite: []

Section: 8:
end if 9:
end for 10:
Output the uniform mixture x t := Unif({p
(1) h1 , . . . , p (L) h1•••h L })
, and observe y t . 11: end for with Γ (l) k t. When evaluating the calibration error, suppose that the actions p (l) k are all distinct, for l ∈ [L], k ∈ [0 : H -1] l-1 (as we discuss below, this case is in some sense the "worst case"). In this event, each action p (l) k is compared to the average outcome over the interval
Γ (l) k : ȳ(l) k = 1 Γ (l) k t∈Γ (l) k y t .
Formally, this would give
Cal D T (x 1:T , y 1:T ) = L l=1 H L-l L k∈[H] l D ȳ(l) k , p(l) k . (6)
as each level l action is selected with 1 L mass for H L-l rounds.
If it happened that two distinct intervals Γ
k1 , Γ(l1)
k2 were assigned the same action p = p
k1 = p(l1)
k1 , then the calibration error would be at most the quantity on the right-hand side of (6) (by Jensen's inequality). In particular, rather than having to compare p to two potentially distinct quantities D(ȳ
(l1) k1 , p), D(ȳ (l2)
k2 , p), the mass placed on p would be categorized under the same forecast and we would only compare p to an appropriately-weighted average of ȳ(l1) k1 and ȳ(l2) k2 . For technical reasons, it will turn out to be necessary to upper bound the "worst case quantity" on the right-hand side of (6) (and an analogous version for swap regret), even in the even that the actions p (l) k are not all distinct. To streamline our notation, we introduce a generalization of these quantities which apply for arbitrary algorithms, which we call labeled calibration error and labeled swap regret.
this section cite: []

Section: Formal definitions.
Given a convex set P ⊂ R d , we define its labeled extension to be P := P × {0, 1} , i.e., elements of P are tuples (p, σ), where σ ∈ {0, 1} is a string that is said to label p. For a loss function ℓ : P → R, we extend its domain to P in the natural way, i.e., ℓ((p, σ)) := ℓ(p) for (p, σ) ∈ P. Given a sequence of distributions over the labeled extension, x 1 , . . . , x T ∈ ∆( P), and loss functions ℓ 1 , . . . , ℓ T : P → R, we define FullSwapReg T (x 1:T , ℓ 1:T ) := sup π: P→ P T t=1 p∈
P x t (p) • (ℓ t (p) -ℓ t (π(p))).
In words, the full swap regret of x 1:T with respect to ℓ 1:T is defined identically as in (2) except that the swap function π can now depend on the label σ. In particular, the labeled extension allows us to consider a more refined notion of swap regret where identical actions played in different rounds can be swapped (via π) to different alternatives as long as they have different labels.
In a similar manner we define the calibration error for a sequence of labeled distributions: given x 1 , . . . , x T ∈ ∆( P) and y 1 , . . . , y T ∈ P, we define Cal D T := (p,σ)∈ P T t=1
x t ((p, σ)) • D(ν (p,σ) , p), ν (p,σ) := T t=1 x t ((p, σ)) • y t T t=1 x t ((p, σ))
.
The main result of [DDFG24] shows that the swap regret of TreeSwap is bounded, even when one labels the action produced at each node of the tree by the node of the tree. This labeled variant of TreeSwap is given in Algorithm 2. It functions exactly as discussed in Section 3.1, except that the distribution x t output at time step t is in ∆( P) instead of ∆(P). In particular, each p (l) k ∈ P in the support of x t is labeled by the tuple k ∈ [0 : H -1] l . 10Theorem C.1 (TreeSwap; Theorem 3.1 of [DDFG24]). Suppose that H, L ∈ N satisfy H ≥ 2 and H L-1 ≤ T ≤ H L . For bounded convex action set P ⊂ R d and loss function set L ⊂ {ℓ : P → [0, b]}, let Alg H : L H → P H be any algorithm. Then, the labeled TreeSwap algorithm (Algorithm 2) parametrized by T, H, L, P, L, Alg H outputs labeled distributions x 1 , . . . , x T ∈ ∆( P) satisfying the following: for any sequence ℓ 1 , . . . , ℓ T ∈ L,
FullSwapReg T (x 1:T , ℓ 1:T ) ≤ T • ExtReg H (Alg H ) H + 3b L .
Algorithm 2 TreeSwap.Alg(P, L, T, H, L), labeled variant (see Appendix C.1)
Require: Action set P ⊂ R d , convex loss class L ⊂ (P → R), no-external regret algorithm Alg, time horizon T , parameters H, L with T ≤ H L . 1: For each sequence h 1 • • • h l-1 ∈ L l=1 [0 : H -1] l-1 , initialize an instance of Alg with time horizon H, denoted Alg h 1:l-1 . 2: for 1 ≤ t ≤ T do 3: Write the base-H representation of t -
1 as t -1 = (h 1 • • • h L ), for h 1 , . . . , h L ∈ [0 : H -1]. 4: for 1 ≤ l ≤ L do 5: Write k := (h 1 • • • h l-1 ) ∈ [0 : H -1] l-1 . 6: if h l+1 = • • • = h L = 0 or l = L then 7: If h l > 0, define ℓ (l) k,h l -1 := 1 H L-l • s∈Γ (l) k,h l -1 ℓ s ∈ L. 8: Define p (l) k,h l = Alg k,h l +1 (ℓ (l)
k,0:h l -1 ) ∈ P. The h l th action of Alg k given the loss sequence ℓ (l) k,1:h l -1 .
this section cite: ['b6', 'b6']

Section: 9:
end if 10: end for 11:
Output the uniform mixture x t := Unif({(p (1) h1 , h 1 ), . . . , (p (L) h1•••h L , h 1:L )}) ∈ ∆( P), and observe ℓ t .
this section cite: []

Section: Each action p (l)
k is labeled by the sequence k (see Appendix C.1). 12: end for C.2 Proof of the main theorem First, we recall some definitions from Section 3. For all l ∈ [0 : L], for all k ∈ [H] l , let Γ (l) k represent the interval of times t with prefix k. That is, t ∈ Γ (l) k iff t i = k i for all i ∈ [1 : l]. These intervals form an H-ary depth-L tree, where the children of Γ
(l) k are Γ (l+1) k0 , Γ (l+1) k1 , • • • , Γ (l+1)
k(H-1) . In the calibration setting where the learner receives outcomes y 1:T , let ν (l)
k = 1 Γ (l) k t∈Γ (l) k
y t (as defined on Line 6 of Algorithm 1). In the swap regret setting where the learner receives loss functions ℓ 1:T , let ℓ (l)
k = 1 Γ (l) k t∈Γ (l) k
ℓ t (as defined in Line 7 of Algorithm 2).
Finally, recall that for an online learning algorithm Alg with time horizon H, we define its action at time step h ∈ [H] given losses ℓ 1 , . . . , ℓ H : P → R by Alg h (ℓ 1 , . . . , ℓ H ). If Alg h only depends on the first g losses, then we will write Alg h (ℓ 1 , . . . , ℓ g ). In the proof of Theorem 3.1 we will consider two algorithms in particular; the first, Follow-The-Leader (FTL) is defined as follows: for ℓ 1 , . . . , ℓ h-1 : P → R, we have
FTL h (ℓ 1 , . . . , ℓ h-1 ) = argmin p∈P h-1 i=1 ℓ i (p).
The second algorithm we consider is the Be-The-Leader algorithm (BTL), which is defined as follows: for ℓ 1 , . . . , ℓ h : P → R, we have
BTL h (ℓ 1 , . . . , ℓ h ) = argmin p∈P h i=1 ℓ i (p).
Note that since BTL h (ℓ 1:h ) depends on the unobserved loss ℓ h at time step h, it is unimplementable. Nevertheless, it will be useful in our analysis.
Next we prove Lemma 3.2, establishing the equivalence of TreeCal and TreeSwap.FTL. In fact, we establish the stronger claim, which immediately implies Lemma 3.2. Lemma C.2. Fix distributions q 0 , . . . , q h ∈ ∆(P), and define ℓ h (p
) := E y∼q h [D R (y|p)]. Then for each h > 0, FTL h (ℓ 0 , . . . , ℓ h-1 ) = 1 h h-1 i=0 E y∼qi [y].
Proof. The lemma is an immediate consequence of Lemma 2.1, noting that ). It remains to demonstrate that both algorithms assign actions p (l) k to intervals Γ (l) k identically. Fixing a choice of l ∈ [L] and k ∈ [0 : H -1] l-1 , this is an immediate consequence of Lemma C.2 with q h = Unif({y t : t ∈ Γ (l) k,h }) and the fact that:
FTL h (ℓ 0 , . . . , ℓ h-1 ) = argmin p∈P h-1 i=0 ℓ i (p) = argmin p∈P E i∼[0:h-1],y∼qi [D R (y i |p)] = 1 h h-1 i=0 E y∼qi [y].
• In TreeCal, p (l) k,h = 1 h h-1 i=0 ν (l) k,i with ν (l) k,i = E t∼Unif(Γ (l) k,i ) [y t ];
• Whereas in TreeSwap.FTL, p
(l) k,h = FTL h+1 (ℓ (l) k,0 , . . . , ℓ (l) k,h-1 ) with ℓ (l) k,i = E t∼Unif(Γ (l) k,i ) [D R (y t |•)].
We are now ready to prove Theorem 3.1.
Proof of Theorem 3.1. Fix any convex set P and a norm • , and let R : P → R be chosen to be 1-strongly convex which has range ρ > 0. Lemma 3.2 gives that the actions x 1 , . . . , x T ∈ ∆(P) are identical to the actions played by TreeSwap.FTL with losses ℓ t (p) = D R (y t |p) (Algorithm 2; we are ignoring the labels here). Thus, from here on, it suffices to bound the calibration error of the corresponding distributions x 1 , . . . , x T of TreeSwap.FTL. The actions p
(l) k,h (for l ∈ [L], k ∈ [0 : H -1] l-1 , h ∈ [0 : H -1]) of TreeSwap.FTL satisfy p (l) k,h = FTL h+1 (ℓ (l) k,0 , . . . , ℓ (l) k,h-1 ).
Next, let p(l) k,h denote the corresponding actions played by TreeSwap.BTL, i.e., p(l) k,h = BTL h+1 (ℓ (l) k,0 , . . . , ℓ (l) k,h ). We let x t ∈ ∆( P) denote the (labeled) distribution chosen by TreeSwap.FTL (Line 11 of Algorithm 2), and let xt ∈ ∆( P) denote the corresponding distribution for TreeSwap.BTL. To be concrete, if t -1 = (h 1 • • • h L ), then
x t = Unif({(p (1) h1 , h 1 ), . . . , (p (L) h1•••h L , h 1:L )}), xt = Unif({(p(1)
h1 , h 1 ), . . . , (p
(L) h1•••h L , h 1:L )}), .(8)
We state the below claim, whose proof is deferred to the end of the section. (We remark that the primary purpose of introducing labeling is so that it is possible to establish Claim C.3.)
this section cite: []

Section: Claim C.3. It holds that

this section cite: []

Section: Cal
• 2 T (x 1:T , y 1:T ) -2Cal
• 2 T (x 1:T , y 1:T ) ≤ 2 • diam(P) 2 H 2 • T. (9
)
The fact that BTL enjoys non-positive external regret (e.g., [SS11, Lemma 2.1] gives that for an arbitrary sequence of loss functions ℓ t : P → R, the external regret of BTL H satisfies ExtReg H (BTL H ) ≤ 0. Thus, by Theorem C.1, the swap regret of (the labeled version of) TreeSwap T applied with Alg H = BTL H may be bounded as follows: for any sequence of losses ℓ 1 , . . . , ℓ T : P → [0, ρ],
FullSwapReg T (x 1:T , ℓ 1:T ) ≤ T • 3ρ L .
Using Lemma 2.2foot_3 and (9), we get that for an arbitrary sequence y 1 , . . . , y T ∈ P,
Cal • 2 T (x 1:T , y 1:T ) ≤2 • Cal • 2 T (x 1:T , y 1:T ) + 2 • diam(P) 2 H 2 • T ≤2 • Cal D R T (x 1:T , y 1:T ) + 2 • diam(P) 2 H 2 • T =2 • FullSwapReg T (x 1:T , D R (y 1:T |•)) + 2 • diam(P) 2 H 2 • T ≤ 6ρ • T L + 2 • diam(P) 2 • T H 2 .
Given any desired accuracy > 0, choosing L = 12ρ/ and H = diam(P)/ √ gives that we can guarantee Cal x t ((p (l)
h1•••h l , h 1 • • • h l )) • ν h1•••h l -p (l) h1•••h l 2 (11) -2 T t=1 xt ((p (l) h1•••h l , h 1 • • • h l )) • ν h1•••h l - p(l) h1•••h l 2 = l∈[L],h 1:l ∈[0:H-1] l H L-l L • ν h1•••h l -p (l) h1•••h l 2 -2 ν h1•••h l - p(l) h1•••h l 2 ≤2 l∈[L],h 1:l ∈[0:H-1] l H L-l L • p (l) h1•••h l - p(l) h1•••h l 2 ≤ 2 L L l=1 h 1:l-1 ∈[0:H-1] l-1 diam(P) 2 • H L-l ≤ 2 L L l=1 diam(P) 2 • H L-1 = 2T diam(P) 2 H ,
where the second-to-last inequality uses that
H-1 h l =0 p (l) h1•••h l - p(l) h1•••h l 2
≤ diam(P) 2 for all choices of h 1 • • • h l-1 (a consequence of Lemma C.4 and Lemma C.2).
Lemma C.4. Fix any convex set P ⊂ R d and a convex function R : P → R. Fix a sequence y 1 , . . . , y H ∈ P, and set
p h = 1 h -1 h-1 i=1 y i ∀h ∈ [H], h > 1, ph = 1 h h i=1 y i ∀h ∈ [H],
as well as p 1 ∈ P arbitrarily. Then
H h=1 p h -ph 2 ≤ 2 • diam(P) 2 . Proof. Note that ph -p h = y h h - 1 h(h -1) h-1 i=1 y i , which implies that ph -p h 2 ≤ π 2 6 • diam(P) 2 < 2diam(P) 2 .
Applying Cauchy-Schwarz, we get the following corollary, Corollary C.5. Let P ⊂ R d be a bounded convex set and • be an arbitrary norm. Then, TreeCal (Algorithm 1) guarantees that for an arbitrary sequence of outcomes y 1 , . . . , y T ∈ P,
the • calibration error of its predictions x 1 , . . . , x T ∈ ∆(P) is bounded Cal • T (x 1:T , y 1:T ) ≤ T for T ≥ (diam • (P)/ ) O(Rate(P, • )/ 2 ) Proof. Using the fact that p∈P T t=1 x t (p) = 1 together with Jensen's inequality, we have 1 T Cal • T (x 1:T , y 1:T ) = 1 T p∈P T t=1 x t (p) • ν p -p ≤ 1 T p∈P T t=1 x t (p) • ν p -p 2 = 1 T Cal • 2 T (x 1:T , y 1:T ) ≤ for T ≥ (diam(P)/ ) O(Rate(P, • )/ 2 ) by Theorem 3.1. Thus, Cal • T (x 1:T , y 1:T ) ≤ T for T ≥ (diam(P)/ ) O(Rate(P, • )/ 2 )
, incurring an additional factor of 2 in the exponent constant, as desired.
Finally, for the setting of centrally symmetric P, we can apply Lemma 2.4 to directly relate this regret bound to the optimal possible rate of an online linear optimization problem.
Corollary C.6. Let P ⊂ R d be a bounded centrally symmetric convex set and • be an arbitrary norm. Then, TreeCal (Algorithm 1) guarantees that for an arbitrary sequence of outcomes y 1 , . . . , y T ∈ P, the • calibration error of its predictions x 1 , . . . , x T ∈ ∆(P) is bounded Cal
• T (x 1:T , y 1:T ) ≤ T for T ≥ (diam • (P)/ ) O(Rate OLO (P, • )/ 2 )
Proof. Follows immediately by applying Lemma 2.4 to Corollary C.5.
this section cite: []

Section: D Proofs for Section 4
In this section, we prove lower bounds on high-dimensional calibration that tell us that in order to achieve calibration error at most • T , we need to take T exp(poly(1/ )). First, in Appendix D.1, we prove a lower bound for ℓ 1 calibration over the d-dimensional simplex, and then, in Appendix D.2, we prove a lower bound for ℓ 2 calibration over the unit d-dimensional Euclidean ball.
this section cite: []

Section: D.1 Lower bound on ℓ 1 calibration
First, we prove Theorem 4.3 which gives a lower bound on ℓ 1 calibration over the simplex P = ∆ d .
Proof of Lemma 4.2. Fix an algorithm Alg which ensures that Cal D T (x 1:T , y 1:T ) ≤ • T as in the statement of the lemma. We construct the following algorithm Alg : it simulates Alg, but whenever Alg outputs the distribution x t ∈ ∆(P), Alg chooses instead x t ∈ ∆(P ), defined by x t (p ) := p∈P: p =argmin q∈P q,p x t (p).
To simplify notation, we define BR(p) := argmin q∈P q, p . It follows that, for any oblivious adversary choosing a (random) sequence y 1 , . . . , y T ∈ P, FullSwapReg T (x 1:T , ℓ(•, y 1:T )) = sup π:P →P p ∈P t∈[T ]
x t (p ) • ( y t , p -π(p ) ) = sup π:P →P p∈P t∈[T ]
x t (p) • ( y t , BR(p) -π(BR(p)) ) = sup
π:P →P p∈P   t∈[T ] x t (p)   • ( ν p , BR(p) -π(BR(p)) ) = sup π:P →P p∈P   t∈[T ] x t (p)   • ( ν p -p, BR(p) -π(BR(p)) + p, BR(p) -π(BR(p)) ) ≤ sup π:P →P p∈P   t∈[T ] x t (p)   • ( ν p -p • BR(p) -π(BR(p)) ) ≤diam • (P ) • Cal D
T (x 1:T , y 1:T ), where in the final inequality we have used the fact that BR(p) -π(BR(p)) ≤ diam • (P ).
For p > 0, d ∈ N, write B d p := {x ∈ R d | x p ≤ 1} to denote the unit ℓ p norm ball.
To map the lower bound Theorem 4.1 from the • 1 -norm unit ball B d 1 to the simplex and arrive at the desired contradiction using the above lemma, we use the following. Lemma D.1. Fix d ∈ N, and write D(x, y) := x -y 1 for x, y ∈ R d . Suppose that there is an algorithm Alg for calibration over the domain P = ∆ 2d+1 which produces x 1:T given the choices of an adversary y 1:T achieving calibration error Cal D T (x 1:T , y 1:T ) ≤ R(T ), for T ∈ N. Then there is an algorithm Alg for calibration over the domain B d 1 which produces x 1:T given y 1:T achieving calibration error Cal D T (x 1:T , y 1:T ) ≤ R(T ).
Proof of Lemma D.1. We define a mapping φ :
B d 1 → ∆ 2d+1 as follows: for y ∈ B d 1 ⊂ R d , we define φ(y) i =    [y j ] + : i = 2j -1, j ∈ [d] [y j ] - : i = 2j, j ∈ [d] 1 -y : i = 2d + 1.
It is straightforward to see that φ has a left inverse ψ, defined as follows: for z ∈ ∆ 2d+1 ,
ψ(z) i = z 2i-1 -z 2i , i ∈ [d],
so that ψ • φ(y) = y for all y ∈ R d .
We define the algorithm Alg as follows: given y t ∈ B d 1 ⊂ R d , it defines y t ∈ ∆ 2d+1 by y t = φ(y t ). Alg then feeds y t into Alg, and if we denote the distribution output by Alg at time step t by x t , Alg then plays the push-forward measure x t := ψ • x t ∈ ∆(B d 1 ).
Our bound on the calibration error of Alg gives Cal D T (x 1:T , y 1:T ) = p∈∆ 2d+1 T t=1 x t (p) • ν p -p 1 ≤ R(T ), where ν p = T t=1 xt(p)•yt T t=1 xt(p) ∈ ∆ 2d+1 . For p ∈ B d 1 , let us denote ν p := T t=1 x t (p )•y t T t=1 x t (p ) = ψ T t=1 x t (p )•yt T t=1 x t (p )
, using linearity of ψ.
this section cite: []

Section: E Pure calibration and pure full swap regret E.1 Pure calibration
In certain settings of calibration, the learner is required to randomly select a pure forecast p t ∈ P rather than a distribution x t ∈ ∆(P). In these settings, the above definition of calibration is instead referred to as "pseudo-calibration". Here, we stick to calling the above calibration, as we believe it to be the more natural definition, and instead refer to this alternative setting as "pure-calibration". The learning task changes as follows.
At each time step t ∈ [T ]:
• The learning algorithm chooses a distribution x t ∈ ∆(P).
• The adversary observes x t and chooses an outcome y t ∈ P.
• The learner samples p t ∼ x t .
We adjust the definitions of the "pure average outcome" and "pure-calibration" accordingly:
νp := Fix any 1 ≤ i ≤ T /S and 1 ≤ j ≤ S, and let F S(i-1)+j denote the σ-algebra generated by y 1 , . . . , y S(i-1)+j+1 and p 1 , . . . , p S(i-1)+j ; since TreeCal is deterministic, it follows that x 1 , . . . , x i ∈ ∆(P) are F i -measurable. For any 1 ≤ j ≤ S, we have that, for any p ∈ supp(x i ), E (p -y S(i-1)+j ) • 1[p S(i-1)+j = p] | F S(i-1)+j-1 = (p -y S(i-1)+j ) • x i (p).
as long as S ≥ 8•Rate(L, • )•diam • (P) 2 •L 2 2 . The guarantee of Corollary C.5 gives that, as long as T /S ≥ (diam • (P)/ ) O(Rate(P, • )/ 2 ) , then Cal • T (x 1:T /S , ȳ1:T/S ) = p∈P T /S i=1 x i (p) • (p -ȳi ) = p∈P T /S i=1 1 S S j=1 x i (p) • (p -y S(i-1)+j ) ≤ T S . (15) By combining Equations (14) and (15), it follows that for an arbitrary adaptive adversary who chooses a sequence y 1 , . . . , y T ∈ P, E PureCal • T (p 1:T , y 1:T ) =E   p∈P T t=1 (p -y t ) • 1[p t = p]   ≤E   p∈P T /S i=1 S j=1 (p -y S(i-1)+j ) • x i (p) + T /S i=1 S j=1 (p -y S(i-1)+j ) • (1[p S(i-1)+j = p] -x i (p))   ≤2 T.
The result follows by rescaling and our choice of L = O(Rate(P, • )/ 2 ).
As example applications of Theorem E.1:
• When • is the ℓ 1 norm and P is the simplex, we have diam • (P) = 1, L = {f ∈ R d | f ∞ ≤ 1} satisfies Rate(L, • ) ≤ d (as we can take the function R(x) = x 2 2 ), which gives that for T ≥ d O(1/ 2 ) , we can have E[PureCal • 1 T ] ≤ T . This result recovers the main upper bound of [Pen25] (Theorem 1.1 therein).
• When • is the ℓ 2 norm and P is the unit ℓ 2 ball, we have diam • (P) = 1, L = {f ∈ R d | f 2 ≤ 1} satisfies Rate(L, • ) ≤ 1 (as we can take the function R(x) = x 2 2 ), which gives that for T ≥ exp(O(1/ 2 )), we can have E[PureCal
• 1 T ] ≤ T .
this section cite: ['b28']

Section: E.2 Sequential law of large numbers
Fix a convex set P ⊂ R d and a norm • on R d . We define R n (P, • ) := sup
p E 1 n n i=1 i p i ( ) ,
where the supremum is over all sequences of mappings p 1 , . . . , p n , where p i : {-1, 1} i-1 → P, and the expectation is over an i.i.d. sequence of Rademacher random variables = ( 1 , . . . , n ), i ∼ Unif({±1}). The below lemma (essentially contained in [RST15]) establishes a martingale law of large numbers for P-valued martingales, in terms of geometric properties of P and • .
this section cite: ['b33']

Section: Lemma E.2 ([RST15]
). Consider a convex set P ⊂ R d a norm • on R d , and let M 1 , . . . , M n denote a sequence of random variables adapted to a filtration (F i ) i∈[n] . Let L = {f | f ≤ 1} be the unit ball of the dual norm • . Then
E n i=1 M i -E[M i | F i-1 ] ≤ diam • (P) • 8n • Rate(L, • ).
Proof. By applying an appropriate translation to P, we can assume that P contains the origin. We apply Theorem 2 of [RST15] with the domain Z equal to P and the function class F equal to the class of mappings {z → z, f : f ≤ 1} indexed by unit-dual norm linear functions on Z. The theorem implies that
E 1 n n i=1 M i -E[M i | F i-1 ] ≤2 • sup p E sup f ≤1 1 n n i=1 i p i ( ), f =2 • R n (P, • )).
Write L = {f ∈ R d : f ≤ 1} denote the unit ball for the dual norm • . Proposition 16 of [RST15] gives that, if there is a function R : L → R which is 1-strongly convex with respect to
• and which has range ρ, then R n (P,
• ) ≤ 2ρ n • diam • (P). In particular, R n (P, • ) ≤ diam • (P) • 2Rate(L, • ) n .
this section cite: ['b33', 'b33']

Section: 
We may now bound the calibration error of Alg by Cal D T (x 1:T , y 1:T ) = p ∈B d 1 T t=1
x t (p ) • ν p -p 1 ≤ p∈∆ 2d+1 T t=1
x t (p) • ψ(ν p ) -ψ(p) 1 ≤Cal D T (x 1:T , y 1:T ).
Proof of Theorem 4.3. Suppose to the contrary that there was an algorithm Alg which bounded calibration error by T for T ≤ exp(c • min{d 1/14 , -1/6 }). Then by Lemma D.1, for d = (d -1)/2 there is an algorithm Alg for calibration on the domain B d 1 ⊂ R d produces x 1:T given y 1:T satisfying Cal D T (x 1:T , y 1:T ) ≤ • T for any T ≤ exp(c • min{d 1/14 , -1/6 }). We now apply Lemma 4.2 for P = B d 1 ⊂ R d , the norm given by the ℓ 1 norm • 1 , and P := [-1, 1] d . Note that diam • ∞ (P ) = 1. Then Lemma 4.2 ensures that there is an algorithm Alg which chooses x 1 , . . . , x T ∈ ∆(P ) which ensures that for every oblivious adversary choosing y 1 , . . . , y T ∈ B d 1 , we have FullSwapReg T (x 1:T , ℓ(•, y 1:T )) ≤ • T . But if T ≤ exp(c 4.1 •min{(d ) 1/14 , -1/6 }), we have a contradiction to Theorem 4.1, thus completing the proof of the theorem.
this section cite: []

Section: D.2 Lower bound for ℓ 2 calibration
Next, we prove a lower bound for ℓ 2 calibration. Theorem D.2. There is a sufficiently small constant c > 0 so that the following holds. Write D(p, p ) = p -p 2 and fix any > 0, d ∈ N. Then for any T ≤ exp(c • min{d 1/14 , -1/7 }), there is an oblivious adversary producing a sequence y 1 , . . . , y T ∈ B d 2 so that for any learning algorithm producing x 1 , . . . , x T ∈ ∆(B d 2 ), Cal D T (x 1:T , y 1:T ) ≥ • T .
Proof. Fix > 0, d ∈ N, and write ˜ = 6/7 . We may assume without loss of generality that d ≤ ˜ -14/6 , so that min{d 1/14 , ˜ -1/6 } = min{d 1/14 , -1/7 } = d 1/14 : if this were not the case, we simply use the adversary resulting from ˜ -14/6 dimensions and project the forecaster's predictions down into this lower-dimensional subspace, which can only decrease calibration error. Now suppose to the contrary that there was an algorithm Alg which bounded calibration error by T for T ≤ exp(c•min{d 1/14 , -1/7 }) = exp(c•d 1/14 ). Then by Lemma 4.2 with P = B d 2 and norm • = • 2 , for any subset P ⊂ B d 2 we get that there is an algorithm which chooses x 1 , . . . , x T ∈ ∆(P ) and which ensures that for every oblivious adversary choosing y 1 , . . . , y T ∈ B d 2 , we have
FullSwapReg T (x 1:T , ( •, y 1 , . . . , •, y T )) ≤ • T.(12)
On the other hand, the oblivious adversary of Theorem 4.1 guarantees a subset X ⊂ [-1, 1] d ⊂ and an oblivious adversary producing a sequence v 1 , . . . , v T ∈ R d with v t ∞ ≤ d -13/14 for all t ∈ [T ], so that
FullSwapReg T (x 1:T , ( •, v 1 , . . . , •, v T )) ≥ ˜ • T (13
) as long as T ≤ exp(c 4.1 • d 1/14 ). We have v t 2 ≤ d 1/2-13/14 = d -3/7 for all t, and scaling X down by a factor of 1/ √ d (i.e., letting P = X / √ d) and all vectors v t up by a factor of d 3/7 (i.e., letting v t = √ d • v t ensures that any algorithm producing x 1 , . . . , x T ∈ P must still have full swap regret FullSwapReg T (x 1:T , ( •, v 1 , . . . , •, v T )) > ˜ • T • d -1/14 ≥ ˜ 7/6 • T = • T, but now ensures that P ⊂ B d 2 and that v t ∈ B d 2 for all t. By taking c = c 4.1 , this contradicts (12).
this section cite: []

Section: References
Ref_id:b0 Title: An elementary predictor obtaining distance to calibration Year: (2025)
Ref_id:b1 Title: Does an efficient calibrated forecasting strategy exist? Year: (2011)
Ref_id:b2 Title: A unifying theory of distance from calibration Year: (2023)
Ref_id:b3 Title: From external to internal regret Year: (2007)
Ref_id:b4 Title: The well-calibrated bayesian Year: (1982)
Ref_id:b5 Title: Breaking the t 2/3 barrier for sequential calibration Year: (2024)
Ref_id:b6 Title: From external to swap regret 2.0: An efficient reduction for large action spaces Year: (2024)
Ref_id:b7 Title: A lower bound on swap regret in extensive-form games Year: (2024)
Ref_id:b8 Title: Smooth calibration, leaky forecasts, finite recall, and nash dynamics Year: (2018)
Ref_id:b9 Title: Full swap regret and discretized calibration Year: (2025)
Ref_id:b10 Title: An easier way to calibrate Year: (1999)
Ref_id:b11 Title: A proof of calibration via blackwell's approachability theorem Year: (1999)
Ref_id:b12 Title:  Year: (1997)
Ref_id:b13 Title: Asymptotic calibration Year: (1998)
Ref_id:b14 Title: Oracle efficient online multicalibration and omniprediction Year: (2024)
Ref_id:b15 Title: On calibration of modern neural networks Year: (2017-08)
Ref_id:b16 Title: H + 16] Elad Hazan et al. Introduction to online convex optimization Year: (2016)
Ref_id:b17 Title: Calibrated forecasts: The minimax proof Year: (2022)
Ref_id:b18 Title: Multicalibration: Calibration for the (Computationally-identifiable) masses Year: (2018-07)
Ref_id:b19 Title: weak) calibration is computationally hard Year: (2012)
Ref_id:b20 Title: Predict to minimize swap regret for all payoff-bounded tasks Year: (2024)
Ref_id:b21 Title: Deterministic calibration and nash equilibrium Year: (2008)
Ref_id:b22 Title: U-calibration: Forecasting for an unknown agent Year: (2023)
Ref_id:b23 Title: Optimal multiclass u-calibration error and beyond Year: (2024)
Ref_id:b24 Title: Simultaneous swap regret minimization via kl-calibration Year: (2025)
Ref_id:b25 Title: A geometric proof of calibration Year: (2010)
Ref_id:b26 Title: Online calibrated forecasts: Memory efficiency versus universality for learning in games Year: (2007)
Ref_id:b27 Title: High-dimensional prediction for sequential decision making Year: (2023)
Ref_id:b28 Title: High dimensional online calibration in polynomial time Year: (2025)
Ref_id:b29 Title: Fast swap regret minimization and applications to approximate correlated equilibria Year: (2024)
Ref_id:b30 Title: Stronger calibration lower bounds via sidestepping Year: (2021)
Ref_id:b31 Title: On the distance from calibration in sequential prediction Year: (2024)
Ref_id:b32 Title: Forecasting for swap regret for all downstream agents Year: (2024)
Ref_id:b33 Title: Sequential complexities and uniform martingale laws of large numbers Year: (2015)
Ref_id:b34 Title: Online learning and online convex optimization Year: (2011)
Ref_id:b35 Title: On the universality of online mirror descent Year: (2011)
Ref_id:b36 Title: Individual calibration with randomized forecasting Year: (2020-07)
