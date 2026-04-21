Title: Efficient Fairness-Performance Pareto Front Computation
Abstract: There is a well known intrinsic trade-off between the fairness of a representation and the performance of classifiers derived from the representation. In this paper we propose a new method to compute the optimal Pareto front of this trade off. In contrast to the existing methods, this approach does not require the training of complex fair representation models. Our approach is derived through three main steps: We analyze fair representations theoretically, and derive several structural properties of optimal representations. We then show that these properties enable a reduction of the computation of the Pareto Front to a compact discrete problem. Finally, we show that these compact approximating problems can be efficiently solved via off-the shelf concave-convex programming methods. In addition to representations, we show that the new methods may also be used to directly compute the Pareto front of fair classification problems. Moreover, the proposed methods may be used with any concave performance measure. This is in contrast to the existing reduction approaches, developed recently in fair classification, which rely explicitly on the structure of the non-differentiable accuracy measure, and are thus unlikely to be extendable. The approach was evaluated on several real world benchmark datasets and compares favorably to a number of recent state of the art fair representation and classification methods.

Section: Introduction
Fair representations are a central topic in the field of Fair Machine Learning, Mehrabi et al. (2021), Pessach and Shmueli (2022), Chouldechova and Roth (2018). Since their introduction in Zemel et al. (2013), Fair representations have been extensively studied, giving rise to a variety of approaches based on a wide range of modern machine learning methods, such GANs, variational auto encoders, numerous variants of Optimal Transport methods, and direct variational formulations. See the papers Feldman et al. (2015), Madras et al. (2018), Gordaliza et al. (2019); Zehlike et al. (2020), Song et al. (2019), Du et al. (2020), Zhao and Gordon (2022), Jovanović et al. (2023), Dehdashtian et al. (2024), for a sample of existing methods.
For a given representation learning problem and a target classification problem, since the fairness constraints reduce the space of feasible classifiers, the best possible classification performance will usually be lower as the fairness constraint becomes stronger. This phenomenon is known as the Fairness-Performance trade-off. Assume that we have fixed a way to measure fairness. Then for a given representation learning method, one is often interested in the fairness-performance curve (γ, E(γ)). Here, γ is the fairness level, and E(γ) is the classification performance of the method at that level. The curve (γ, E(γ)) where E(γ) is the best possible performance over all representations and classifiers under the constraint is known as the Fairness-Performance Pareto Front.
As indicated by the above discussion, representation learning methods typically involve models with high dimensional parameter spaces, and complex, possibly constrained non-convex optimisation algorithms. As such, these methods may be prone to local minima and sensitivity to a variety of hyper-parameters, such as architecture details, learning rates, and even initializations. While the representations produced by such methods may often be useful, it nevertheless may be difficult to decide whether their associated Fairness-Performance curve is close to the true Pareto Front.
In this paper we propose a new method to compute the optimal Pareto front, which does not require the training of complex fair representation models. In other words, we show that, perhaps somewhat surprisingly, the computation of the Pareto Front can be decoupled from that of the representation, and only relies on learning of much simpler, unconstrained classifiers on the data. To achieve this, we first show that the optimal fair representations satisfy a number of structural properties. While these properties may be of independent interest, here we use them to express the points on the Pareto Front as the solutions of small discrete optimisation problems. These problems, known as concave minimisation problems Benson (1995), have been extensively studied and can be efficiently solved using modern dedicated optimisation frameworks, Shen et al. (2016).
We now describe the results in more detail. Let X ∈ X and A ∈ A denote the data features and the sensitive attribute, respectively. We assume that A is binary, while X may take values in an arbitrary space X , typically with X = R d . In addition, we have a target variable Y , taking values in a finite set Y. We are then interested in representations that maximise the performance of prediction of Y , under the fairness constraint. The representation is denoted by Z, and is typically expressed by constructing the conditional distributions P (Z|X = x, A = a), for a ∈ {0, 1}. The problem setting is illustrated in Figure 1.
Our approach consists of three main steps: We first observe that one can map the data features (x, a) ∈ X × A to a much smaller space ∆ Y of distributions on the set of label values Y, without loosing any information necessary for the computation of the Pareto front. The mapping is done by means of the optimal Bayes classifier. This result is referred to as Factorization Lemma, Section 3.2, where the mapping is done via the optimal Bayes classifier. Similar arguments were recently implicitly used in the study of fair classification tradeoffs, (Xian et al., 2023;Wang et al., 2023), but were restricted to classification and to accuracy loss (Section 2).
The advantage of working with a small space such as ∆ Y is that it can be easily discretized. For instance, if Y = 0, 1, then we essentially have ∆ Y = [0, 1], which is descretised trivially. We note that other, data dependent discretisation schemes, such as clustering, maybe possible for problems involving highly multi label targets. Alternatively, one can also consider the dataset itself as a grid, a view that is typically taken by transportation based approaches, e.x. Gordaliza et al. (2019), Xian et al. (2023).
Next, assuming the data is discretized and finite, we ask how large the represntation space Z should be, in order to support both optimal performance and fairness? For instance, we believe the answer to the following question is not apriori obvious: Can representations on infinite spaces be approximated, in terms of performance and fairness, by representations on finite and bounded spaces Z? These questions are addressed by the Invertibility Theorem, Section 3.3, which asserts that all optimal representations may be taken in a certain canonical form, which we term invertible. This result, in conjunction with an additional approximation lemma, is used in Section 4.1 to construct representations with any desired degree of approximation.
Finally, based on these results in Section 4.1 we also introduce the MIFPO (Model Independent Fairness-Performance Optimization), a discrete optimisation problem that is essentially equivalent to a computation of the fairness-performance tradeoff on a discrete set. We show that in this situation MIFPO is a concave minimisation problem with linear constraints, and we solve it using the disciplined convex-concave programming framework, DCCP, Shen et al. (2016).
We evaluate our approach on standard fairness benchmark datasets and compare its fairnessperformance curve to multiple state-of-the-art fair representation methods. We also compare MIFPO to the fairness-performance Pareto front of fair classifiers 1 . As expected, MIFPO effectively serves as an upper bound on almost all other algorithms in both cases.
it can not be extended to general concave performance measures, such as the standard (minus) log loss, for instance. Roughly speaking, in the appropriate sense, accuracy largely ignores classification probabilities. This allows the simple description of classifiers as small confusion matrices in Wang et al. (2023) (extending the approach of Kim et al. (2020)), and the restriction of the distributions to the vertices of the simplex in Xian et al. (2023). The special structure of accuracy is highlighted also in our Lemma 3.1, where we show that classifiers with accuracy may be effectively described by representations using only 2 points. We conclude that even when restricted to classification, our approach analyses a fundamentally more complex situation compared to previous work. On the other hand, Xian et al. (2023) and Wang et al. (2023) support non binary sensitive attributes and group labels, while such an extension for our methods is out of scope for this paper. A comparison of computational complexities for these algorithms may be found in Supplementary J.
this section cite: ['b27', 'b31', 'b10', 'b41', 'b16', 'b26', 'b17', 'b40', 'b36', 'b14', 'b42', 'b21', 'b13', 'b6', 'b34', 'b39', 'b37', 'b17', 'b39', 'b34', 'b37', 'b23', 'b39', 'b39', 'b37']

Section: Structure of Fair Representations
In this Section we describe several theoretical properties of fair representations. In Section 3.1 we introduce the problem setup and the necessary notation. In Section 3.2 we discuss relations to classification with accuracy loss and the factorization result, which allows to reduce the size of the representation space. The Invertibility Theorem is introduced in Section 3.3.
this section cite: []

Section: Problem Setting
Let A be a binary sensitive variable, and let X be an additional feature random variable, with values in a set X , typically with X = R d . Assume also that there is a target variable Y with finitely many values in a set Y, jointly distributed with X, A.
A representation Z of (X, A) is defined as a random variable taking values in some space Z, with (i) distribution given through P θ (Z|X, A), where θ are the parameters of the representation, and (ii) such that conditioned on (X, A), Z is independent of the rest of the variables of the problem. In particular, we have
Z ⊥ ⊥ Y |(X, A),(1)
where ⊥ ⊥ denotes statistical independence.
Fairness in this paper will be measured by the Total Variation distance. For two distributions, µ, ν on R d , with densities f µ , f ν , respectively, this distance is defined as Cover and Thomas (2012).
∥µ -ν∥ T V = 1 2 sup g s.t. ∥g∥ ∞ ≤1 g(x) • [f µ (x) -f ν (x)] dx = 1 2 |f µ (x) -f ν (x)| dx. (2) Note that |f µ (x) -f ν (x)| dx is in fact the L 1 distance, and the equivalence ∥•∥ T V = 1 2 ∥•∥ L1 is well known, see
For a ∈ {0, 1}, let µ a be the distribution of Z given A = a, i.e. µ a (•) := P (Z = •|A = a). We denote the distance induced by the representation as D T V (Z) = ∥µ 0 -µ 1 ∥ T V , and for γ ≥ 0, we say that the representation Z is γ-fair iff
D T V (Z) = ∥µ 0 -µ 1 ∥ T V ≤ γ (Fairness Condition).(3)
Note that (3) is a quantitative relaxation of the "perfect fairness" condition in the sense of statistical parity, which requires Z ⊥ ⊥ A. Specifically, observe that by definition, Z ⊥ ⊥ A iff (3) holds with γ = 0 (i.e. µ 0 = µ 1 ). In addition, as shown in Madras et al. (2018), (3) implies several other common fairness criteria, in particular, bounds on demographic parity and equalized odds metrics for any downstream classifier built on top of Z.
Next, we describe the measurement of information loss in Y due to the representation. Let h : ∆ Y → R be a continuous and concave function on the set of probability distributions on Y, ∆ Y . The quantity h(P (Y |X = x)) will measure the best possible prediction accuracy of Y conditioned on X = x, for varying x. As an example, consider the case of binary Y , Y = {0, 1}. Every point in ∆ Y can be written as (p, 1 -p) for p ∈ [0, 1], and we may choose h to be the optimal binary classification error,
h((1 -p, p)) = min(p, 1 -p).(4)
Another possibility it to use the entropy, h((1 -p, p)) = p log p + (1 -p)log(1 -p). The average uncertainty of Y is given by E x∼X h(P (Y |X = x)). Note that this notion does not depend on a particular classifier, but reflects the performance the best classifier can possibly achieve (under appropriate cost).
The goal of fair representation learning is then to find representations Z that for a given γ ≥ 0 satisfy the constraint (3), and under that constraint minimize the objective E = E θ given by
E θ = E z∼Z h(P (Y |Z = z)).(5)
That is, the representation should minimise the optimal Y prediction error (using Z) under the fairness constraint.
The curve that associates to every 0 ≤ γ ≤ 1 the minimum of ( 5) over all representations Z which satisfy (3) with γ is referred to as the Pareto Front of the Fairness-Performance trade-off.
In supplementary material Section A we show that for any representation,
E z∼Z h(P (Y |Z = z)) ≥ E x∼X h(P (Y |X = x)
), i.e. representations generally decrease or maintain the performance.
this section cite: ['b11', 'b26']

Section: Classification and Factorization
In this section we show that the Pareto front of binary classifiers, with accuracy performance and statistical parity fairness measure, can be computed from the Pareto front of representations with total variation fairness measure. In fact, Lemma 3.1 below states that both Pareto fronts amount to the same curve. As discussed in Section 1, this equivalence implies that MIFPO can be used to evaluate fair classifiers, in addition to fair representations.
For a binary classifier Ŷ of Y , with (X, A) as features. The prediction error is defined as usual by ϵ( Ŷ ) := P Ŷ ̸ = Y . The statistical parity of Ŷ is defined as
D SP ( Ŷ ) := P Ŷ = 1|A = 1 -P Ŷ = 1|A = 0 .(6)
Lemma 3.1. Let Ŷ be a classifier of Y , let the representation uncertainty measure be given by (4).
Then there is a representation given by a random variable Z on a set Z with |Z| = 2, such that
E z∼Z h(P (Y |Z = z)) ≤ ϵ( Ŷ ) and ∥µ 0 -µ 1 ∥ T V ≤ D SP ( Ŷ ).(7)
Conversely, for any given representation Z, there is a classifier Ŷ of Y as a function of Z (and thus of (X, A)), such that
ϵ( Ŷ ) ≤ E z∼Z h(P (Y |Z = z)) and D SP ( Ŷ ) ≤ ∥µ 0 -µ 1 ∥ T V .(8)
The Proof of Lemma 3.1 is presented in Supplementary Material Section I.
We now describe the Factorization result. Let f * : X × A → ∆ Y be the Bayes optimal classifier of Y given X, A. That is, for every x ∈ X , a ∈ A, f * (x, a) is the conditional distribution of Y given x, a, i.e.
f * (x, a) = P (Y = •|X = x, A = a). Denote by (X ′ , A) a new pair of random variables, taking values in ∆ Y × A, given by (X ′ , A) = (f * (X, A), A). Lemma 3.2 (Factorization). For any representation Z of (X, A), there is a representation Z ′ of (X ′ , A), such that E z ′ ∼Z ′ h(P (Y |Z ′ = z ′ )) ≤ E z∼Z h(P (Y |Z = z)) and D T V (Z ′ ) ≤ D T V (Z).(9)
In words, for every representation Z, we can find a representation Z ′ that only accesses (x, a) through the value f * (x, a), and is at least as good in terms of both fairness and performance. Equivalently, this means that any two points (x 1 , a) and (x 2 , a) with coinciding conditional Y distribution may be treated as identical for the purposes of constructing optimal representations. As a result, to find optimal tradeoffs, we can only consider the representations Z ′ on the small space ∆ Y × A, rather than Z on the much bigger space X × A.
Observations related to Lemma 3.2 were made in the context of classification in Kim et al. (2020), Xian et al. (2023), and Wang et al. (2023), which also start from the Bayes optimal classifier. Lemma 3.2 generalizes these observations to representations and to general losses. The proof may be found in Supplementary K.
this section cite: ['b23', 'b39', 'b37']

Section: The Invertibility Theorem
In this section we define the notion of invertibility for representations, and show that considering invertible representations is sufficient for computing the Pareto front.
A representation Z on a set Z is invertible if for every z ∈ Z and every a ∈ {0, 1}, there is at most one x ∈ X such that P (Z = z|X = x, A = a) > 0. In words, a representation is invertible, if any given z can be produced by at most two original features (x, a), and at most one for each value a.
For z ∈ Z, we say that an (x, a) is a parent of z if P (Z = z|X = x, A = a) > 0.
Theorem 3.1. Let Z be any representation of (X, A) on a set Z. Then there exists an invertible representation Z ′ of (X, A), on some set Z ′ , such that
E z ′ ∼Z ′ h(P (Y |Z ′ = z ′ )) ≤ E z∼Z h(P (Y |Z = z)) and D T V (Z ′ ) = D T V (Z).(10)
In words, for every representation, we can find an invertible representation of the same data which satisfies at least as good a fairness constraint, and has at least as good performance as the original. In particular, this implies that when one searches for optimal performance representations, it suffices to only search among the invertible ones.
The proof proceeds by observing that if an atom z ∈ Z has more than one parent for a fixed a, then one can split this atom into two, with each having less parents. However, the details of this construction are somewhat intricate and the full argument can be found in Section C.
Although in this paper we concentrate on the case of binary sensitive variable, we note that Theorem 3.1 may be extended to multi valued attributes, with a similar argument. In that case, invertibility would mean that every z ∈ Z would still have at most two parents, u, v, corresponding to different values a, a ′ of A.
this section cite: []

Section: The Model Independent Optimization Problem
In this Section we motivate and introduce the MIFPO optimisation problem, and then discuss the full Pareto front computation procedure starting from the raw data.
this section cite: []

Section: MIFPO Definition
For the purposes of this Section, we assume that the x feature space X is finite. In the next section, Section 4.2, we describe how we obtain such finite spaces by using the factorization result and discretizing ∆ Y . Note, however, that the full original, possibly high dimensional feature space X , is never discretized.
Write S 0 = {(x, 0) | x ∈ X } = X × {0}, and similarly S 1 = X × {1}, for the two halves of the full feature space, X × A = S 0 ∪ S 1 .
Parameters: The MIFPO parameters model the data distribution and are as follows: (a) the probability distributions β 0 ∈ ∆ S0 and β 1 ∈ ∆ S1 , on S 0 and S 1 respectively, modeling P ((X, A)|A = 0) and P ((X, A)|A = 1) respectively, i.e. the distribution of the data features on each sensitive subgroup. (b) The subgroup proportions α a = P (A = a), and (c) the conditional Y distributions,
ρ u , ρ v ∈ ∆ Y , modeling ρ u = P (Y = •|(X, A) = u) when a = 0 or ρ v = P (Y = •|(X, A) = v) when a = 1.
Representation Space: Perhaps the first question one can ask when constructing a representation of the data as above is: How large the representation space should be? We now answer this question using the theory of Section 3.
Fix an integer k ≥ 2. The representation space Z will be a finite set which can be written as
Z = S 0 × S 1 × [k],(11)
where [k] := {1, 2, . . . , k}. That is, every point z ∈ Z corresponds to some triplet (u, v, j), with u ∈ S 0 , v ∈ S 1 , j ∈ [k]. To explain this choice, recall that by the Invertibility result, we know that we may consider only invertible representations. In such representations, every point z ∈ Z is indexed by a pair of parents (u, v) ∈ S 0 × S 1 , suggesting that we may index the points by S 0 × S 1 to begin with. Next, for a given such pair (u, v), we may ask how many points z should have the same pair (u, v) as their parents? In Supplementary Section D, we show that using k points for every pair, we can obtain uniform approximation over all representations. That is, given a degree of approximation ε, Lemma D.1 provides a bound on k which is sufficient to obtain such approximation. While such a bound would clearly depend on ε, we not that it does note depend on the sizes |S 0 | , |S 1 |. These considerations explain the choice of (11) as the representation space. We have used k = 5 in all experiments.
Variables: The variables of the problem model the representation itself. They will be denoted by r a u,v,j for (u, v, j) ∈ Z and a ∈ A, and model the probabilities r a u,v,j = P (Z = (u, v, j)|(X, A) = s), where either a = 0 and s = u ∈ S 0 , or a = 1 and s = v ∈ S 1 for some v ∈ X . That is, for a = 0, points u transition to (u, v, j) for some v ∈ S 1 , j ∈ [k], and similarly for a = 1, points v transition to (u, v, j) for some u ∈ X , j ∈ [k]. This notation preserves our convention that (u, v, j) ∈ Z has u and v as its only parents. The situation is illustrated in Figure 2(a).
Note that the variables represent probabilities, and thus satisfy the following constraints: r a u,v,j ≥ 0, ∀(u, v, j) ∈ Z, ∀a ∈ A
v∈S1,j∈[k] r 0 u,v,j = 1 ∀u ∈ S 0 and
u∈S0,j∈[k] r 1 u,v,j = 1 ∀v ∈ S 1 (13
)
Performance Objective and Fairness Constraints: With these preparations, we are ready to write the performance cost (5) in the new notation:
E r = z=(u,v,j) α 0 β 0 (u)r 0 u,v,j + α 1 β 1 (v)r 1 u,v,j • h ρ u α 0 β 0 (u)r 0 u,v,j + ρ v α 1 β 1 (v)r 1 u,v,j α 0 β 0 (u)r 0 u,v,j + α 1 β 1 (v)r 1 u,v,j .(14)
Indeed, observe that due to the structure of our representations, every z has two parents, and we have
P (Z = z) = α 0 β 0 (u)r 0 u,v,j + α 1 β 1 (v)r 1 u,v,j . Similarly, P (Y |Z = z) is computed via P (Y |z) = x,a P (Y |x, a, z) P (a, x|z) = x,a P (Y |x, a) P (z|a, x) P (a, x) /P (z) ,
and substituted inside h to obtain (14). As we show in Supplementary E, the cost (14) is a concave function of the variables r.
We now proceed to discuss the fairness constraint. Recall that we define µ a (z) = P (Z = z|A = a), for a ∈ {0, 1}. For z = (u, v, j) we have then µ a ((u, v, j)) = β a (u)r a u,v,j , for a ∈ {0, 1}, and we can write
D T V (Z) = ∥µ 0 -µ 1 ∥ T V = 1 2 z |µ 0 (z) -µ 1 (z)| = 1 2 (u,v,j) β 0 (u)r 0 u,v,j -β 1 (v)r 1 u,v,j(15)
and the Fairness constraint, for a given γ ∈ [0, 1], is thus simply 1 2
(u,v,j) β 0 (u)r 0 u,v,j -β 1 (v)r 1 u,v,j ≤ γ.(16)
We now summarise the full MIFPO problem.
Definition 4.1 (MIFPO). For a fixed finite ground set X × A = S 0 ∪ S 1 , the problem parameters are the weight α 0 , the distributions β 0 , β 1 , on S 0 and S 1 respectively, and the distributions ρ x ∈ ∆ Y for every x ∈ S 0 ∪ S 1 . The problem variables are r 0 u,v,j , r 1 u,v,j (u,v,j)∈Z as defined above. We are interested in minimizing the concave function (14), subject to the constraints (12), (13), and (16).
The relationship between MIFPO and the Optimal Transport problem is detailed in Supplementary B.
Finally, observe that the MIFPO constraints above linear, with (16) being equivalent to two linear inequality constraints. We note that these constraints may be replace by equivalent linear equality constraints, via appropriate slack variables, which is more convenient in practice. See Supplementary F for details.
this section cite: []

Section: The Full Algorithm
In this Section we summarize the full Pareto front computation algorithm, including the estimation of the MIFPO parameters α,β and ρ as discussed above.
Let D = {((x i , a i ), y i )} i≤N be the dataset, and write
D a = {((x i , a i ), y i ) ∈ D | a i = a}, so that D = D 0 ∪ D 1 .
The algorithm proceeds in the following steps: Step 1: we learn the probability estimators c 0 , c 1 : R d → ∆ Y , separately on D 0 and D 1 . These estimators should approximate the optimal Bayes classifier (Section 3.2). Note that such estimation of probabilities is well studied, and is known as calibration, see Niculescu-Mizil and Caruana (2005), Kumar et al. (2019), (Berta et al., 2024).
Step 2: (Discretization and Parameter estimation) For a given integer L > 0, the space ∆ Y is discretized into L bins. This corresponds to taking the ground sets S 0 , S 1 in MIFPO to be of size L. The data D a is then mapped into the L bins using c a . The distribution β a (w) is then simply measures the proportion of points {c a (x)} (x,a)∈Da that fall into bin w ≤ L. Finally, for every bin w, we choose an arbitrary point inside that bin as the representative distribution, ρ w . The parameters α a are estimated simply by α a = |D a | / |D|. Note that for binary Y , ∆ Y is simply the interval [0, 1], which is trivial to discretize. See Figure 2(b) for an example of such histograms on real data. We note that one could easily consider more complex discretization schemes, such as clustering, which could be applied efficiently to multi label problems. See Supplementary J for a discussion.
Step 3: For a given a fairness threshold γ > 0, we can now construct the MIFPO instance, Definition 4.1, with |S 0 | = |S 1 | = L, the additional approximation parameter k, and α, β, ρ as discussed above. As discussed in Section 4.1, we found it sufficient to use k = 5 throughout the paper. The MIFPO is then solved using the existing methods, as detailed in Section 5. The full algorithm is schematically show as Algorithm 1, Supplementary H.1.
this section cite: ['b29', 'b24', 'b7']

Section: Experiments
Our approach requires two main computational components: building calibrated classifiers to evaluate c a , and solving the discrete optimization problem described in Section 4.2. For the calibrated classifier, we have used XGBoost (Chen et al., 2015), with Isotonic Regression calibration, as implemented in sklearn, Pedregosa et al. (2011). Next, as discussed in Sections 1, 4.1, MIFPO is a concave minimisation problem, under linear constraints. To solve its, we have used the DCCP framework and the associated solver (Shen et al., 2016(Shen et al., , 2024)), which are based on the combination of convexconcave programming (CCP) Lipp and Boyd (2016) and disciplined convex programming, Grant et al. (2006). We note that although local minima are theoretically possible, the above framework is well-established, and the concave structure can be exploited to allow finding optimal solutions in most practical cases, (Shen et al., 2016). In particular, our results do not indicate local minima issues. However, it may also be worth noting that MIFPO could in principle be also solved with the classical branch-and-bound methods, Benson (1995), which may be slower but do guarantee the global optimum solution.
Throughout the experiments, we use the missclassification error loss h given by (4). Additional implementation details may be found in Supplementary Section H.
Our experimental validation of MIFPO encompasses three standard fairness benchmarks: the Health dataset alongside two variants of ACSIncome-one restricted to California (ACSIncome-CA) and another spanning the entire United States (ACSIncome-US). In Figure 3 we evaluate MIFPO against five state-of-the-art fair representation techniques: CVIB (Moyer et al., 2019), FCRL (Gupta et al., 2021), FNF (Balunović et al., 2022b), sIPM (Kim et al., 2023), and Fare (Jovanović et al., 2023). Each competitive approach was tuned across diverse hyperparameter settings to generate a spectrum of representations balancing fairness and accuracy. Moreover, we evaluated MIFPO against several fair-classification methods on multiple datasets as presented in the Supplementary Section H.
The empirical results presented in Figure 3 demonstrate MIFPO's effectiveness relative to prior approaches. MIFPO consistently achieves performance equal to or superior than the baseline methods across almost all operating points. Furthermore, MIFPO provides a significant methodological advantage through its ability to characterize the complete Pareto frontier. In the figure, MIFPO's performance is visualized as a solid line with points that trace the entire Pareto front, while competing algorithms are represented as individual points corresponding to different hyperparameter configurations.
this section cite: ['b9', 'b30', 'b34', 'b35', 'b25', 'b18', 'b34', 'b6', 'b28', 'b19', 'b4', 'b22', 'b21']

Section: Implementation
All evaluations can be found at https://github.com/bp6725/  Efficient-Fair-Pareto-Paper. The MIFPO algorithm's source code is available in the https://github.com/bp6725/FairPareto repository. The algorithm is also implemented as the "FairPareto" Python package on PyPI, which provides a scikit-learn compatible API for computing optimal fairness-performance Pareto fronts. The package supports two usage modes: a tabular mode with automatic classifier training and calibration given a sensitive attribute column, and a second mode for any data type (images, text) where users provide pre-trained classifiers for each sensitive group. This enables researchers to benchmark their fair classification methods against theoretical optimality with minimal code and make informed decisions about fairness-performance trade-offs.
The package is open-source, available on PyPI, and includes comprehensive documentation with examples for both tabular and image data.
this section cite: []

Section: Conclusions, Limitations, And Future Work
In this paper we have introduced new fundamental properties of optimal fair representations. In particular, these are the first theoretical results that allow approximation of the Pareto front for arbitrary concave performance measures. We have used these results to develop a model independent procedure for the computation of Fairness-Performance Pareto front from data, demonstrated the procedure on real datasets, and have shown that it may be used as a benchmark for other representation learning algorithms.
We now discuss limitations and a few possible directions for future work. This work primarily concentrated on binary sensitive attribute A and binary Y , with the aim to develop the underlying new principles in the simplest case first. As discussed earlier (Sections 1, J), the multi-label case may be treated by more elaborate discretizations. We also noted that the Invertibility Theorem holds for multi valued sensitive attributes as well, which allows to extend the approximation analysis to that case too. Both of these steps, however, would increase the MIFPO problem size. On the other hand, it is also worth noting that this size does not depend directly neither on the feature dimension d, nor on the sample size N and thus the problem scales well in that sense.
In view of these observations, we believe it would be of interest to study the following question on the true complexity of the tradeoff evaluation: Suppose we are given access to the Bayes optimal classifier of the data, f * . This encapsulates, in a sense, most of the "continuous" information of the problem. Then, how scalable can Pareto estimation methods be made theoretically, in terms of |Y| , |A|, while still maintaining controllable approximation bounds?
NeurIPS Paper Checklist 1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: All claims are supported in the paper.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: A discussion may be found in Section 6, as well as in Section 2 in context of comparison to other methods.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: All results are formally stated and have proofs in the Supplementary.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: All information on reproducibility is included in Section 5 and corresponding Supplementary sections. The full code will be published with the final version of the paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [No] Justification: While we describe all reproducibility details, the full code will be published with the final version of the paper. The datasets used are publicly available.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental setting/details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We supply all the necessary details for the reproduction of the results. Furthermore, the full details will be included as part of the published code.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: The methods discussed in this paper produce identical results over multiple runs (std <= 1e-5). We note that for some benchmark methods we have used existing evaluation results; however, the authors of those evaluations claim to have conducted multiple runs to validate stability, and the results presented on the graphs are from different hyperparameters as an integral part of the analysis.
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
Answer: [Yes] Justification: Yes, relevant information is provided in the Supplementary. We note that all experiments of this paper together run in under day on a standard desktop.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: We have examined the Code of Ethics and verified that the research conforms with the code.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] Justification: This work is in Fairness, with obvious societal impact implications. However, while our approach is algorithmically new and can provide practically and empirically better evaluation of fairness-performance tradeoffs, the general societal implications of such tradeoffs are well understood in the field and we believe require no special new consideration in this particular work.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: We do not see reasonable scenarios in which release of our code may be risky.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: All assets are precisely referenced.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: We do not release new assets this time. As mentioned earlier, the implementation of the methods in this paper will be released with the final version of the paper.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing or reesearch with humans. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16.
Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: This research does not involve LLMs as any important, original, or nonstandard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described. Supplementary Material A Monotonicity of Loss Under Representations 20 B MIFPO and Optimal Transport 21 C Proof Of Theorem 3.1 21 D Uniform Approximation and Two Point Representations 25 E Concavity Of E r 27 F MIFPO Equality Constraint 28 G Additional Proofs 28 H Experiments 29 H.1 Implementations and computational details . . . . . . . . . . . . . . . . . . . . . 29 H.2 Additional Evaluations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30 H.3 Minimization of Concave Functions under Convex Constraints and the Entropy Loss 31 I Fair Classifiers As Fair Representations 33 J Computational Complexities 34 K Factorization 34
this section cite: []

Section: A Monotonicity of Loss Under Representations
As discussed in Section 3.1, we observe that representations can not increase the performance of the classifier (i.e decrease the loss). Lemma A.1. For every (Y, X, A), every representation Z as above, and concave h,
E z∼Z h(P (Y |Z = z)) ≥ E (x,a)∼(X,A) h(P (Y |X = x, A = a)).(17)
Note that the right hand-side above can be considered a "trivial" representation, Z = (X, A).
In what follows, to simplify the notation we use expressions of the form P (x, a|z) to denote the formal expressions P (X = x, A = a|Z = z), whenever the precise interpretation is clear from context.
Proof. For every value y ∈ Y, we have
P (Y = y|Z = z) = a dx P (Y = y|x, a, z) ∂P (x|a, z) ∂x P (a|z) (18
) = a dx P (Y = y|x, a) ∂P (x|a, z) ∂x P (a|z) (19
) = E (x,a)∼(X,A)|Z=z P (Y = y|x, a) .(20)
Here, on line (18), ∂P(x|a,z)   ∂x is the density of P (x|a, z) with respect to dx. Crucially, the transition from ( 18) to ( 19) is using the property (1). The transition ( 19) to ( 20) is a change of notation. Using (20) and the concavity of h, we obtain
E z∼Z h(P (Y |Z = z)) ≥ E z∼Z E (x,a)∼(X,A)|Z=z h (P (Y |x, a)) (21) = E (x,a)∼(X,A) h(P (Y |X = x, A = a)).(22)
this section cite: []

Section: B MIFPO and Optimal Transport
In this Section we discuss the relation between the MIFPO minimisation problem, Definition 4.1, and the problem of Optimal Transport (OT). General background on OT may be found in Peyré et al. (2019). We discuss the similarity between OT and the minimisation of ( 14) under the constraint ( 16) with γ = 0, i.e. the perfectly fair case. In this case, ( 16) is equivalent to the condition β 0 (u)r u,v,j = β 1 (v)r v,u,j , for all (u, v, j) ∈ Z. Next, note that thus in is case the expression for
P (Y |Z = (u, v, j)) is ρ u α 0 β 0 (u)r u,v,j + ρ v α 1 β 1 (v)r v,u,j α 0 β 0 (u)r u,v,j + α 1 β 1 (v)r v,u,j = ρ u α 0 + ρ v α 1 α 0 + α 1 , (23
)
and this is independent of the variables r! Therefore we can write the cost ( 14) as
u,v (α 0 + α 1 )h ρ u α 0 + ρ v α 1 α 0 + α 1 1 2   j≤k β 0 (u)r u,v,j + β 1 (v)r v,u,j   .(24)
Note further that for fixed u, v, the different j's in this expression play similar roles and could be effectively merged as a single point.
The cost (24) has several similarities with OT. First, in both problems we have two sides, S 0 and S 1 , and we have a certain fixed loss associated with "matching" u and v. In case of ( 24), this loss is
(α 0 + α 1 )h ρuα0+ρvα1 α0+α1
, which describes the information loss incurred by colliding u and v in the representation. And second, similarly to OT, (24) it is linear in the variables r. Linear programs are conceptually considerably simpler than minimisation of the concave objective (14).
this section cite: ['b32']

Section: C Proof Of Theorem 3.1
In this Section we prove Theorem 3.1.
To keep the notation and the main argument concise, we prove the result under the assumption that (X, A) is finitely supported. Since no assumptions are made on the cardinalities of the supports, the general measurable case follows by standard approximation arguments.
We now introduce the additional notation necessary for the proof. Let S 0 , S 1 be finite disjoint sets, where S a represents the values of (X, A) when A = a, for a ∈ {0, 1}. Denote S = S 0 ∪ S 1 . We are assuming that there is a probability distribution ζ on S, and A is the random variable A = 1 {s∈S1} .
X is defined as taking the values s ∈ S, with P (X = s) = ζ(s). Further, the variable Y is defined to take values in a finite set Y, and for every s ∈ S, its conditional distribution is given by ρ s ∈ ∆ Y . That is, P (Y = y|X = s, A = a) = ρ s (y) = ρ s,a (y).foot_1 This completes the description of the data model.
For a ∈ {0, 1} we denote α a = P (A = a) = ζ(S a ), and β a (s) = P (X = s|A = a). Observe that β a (s) = 0 if s / ∈ S a , and
β a (s) = ζ(s)/ζ(S a ) if s ∈ S a .
We now describe the representation. The representation will take values in a finite set Z. For every s ∈ S and z ∈ Z, let T a (z, s) = P (Z = z|X = s, A = a) be the conditional probability of representing s as z. T a are sometimes referred as the transition kernels of the representation. For fixed (X, A), the T a 's fully define the distribution of the representation Z and we shall refer to the representation as T or as Z in interchangeably. Finally, for a ∈ {0, 1} denote µ a (z) = T a β a = P (Z = z|A = a) = s∈Sa β a (s)T a (z, s).
With the new notation, a representation T is invertible if for every z ∈ Z and every a ∈ {0, 1}, there is at most one s ∈ S a such that T a (z, s) > 0. In words, a representation is invertible, if any given z can be produced by at most two original features s, and at most one in each of S 0 and S 1 .
Given a representation T and z ∈ Z, we say that an s ∈ S is a parent of z if T a (z, s) > 0 for the appropriate a.
Proof Of Theorem 3.1. Assume T is not invertible. Then there is a z ∈ Z which has at least two parents in either S 0 or S 1 . Assume without loss of generality that z has two parents in S 0 . Let
U = {s ∈ S 0 | T 0 (z, s) > 0} , V = {s ∈ S 1 | T 1 (z, s) > 0}(26)
be the sets of parents of z in S 0 and S 1 respectively. Chose a point x ∈ U , and denote by U r = U \{x} the remainder of U . By assumption we have |U r | ≥ 1. We also assume that |V | > 0. The easier case |V | = 0 will be discussed later.
Now, we construct a new representation, T ′ . The range of T ′ will be Z ′ = Z \ z ∪ {z ′ , z ′′ }. That is, we remove z and add two new points. Denote
κ = P (x|z, a = 0) = β 0 (x)T 0 (z, x) s∈U β 0 (s)T 0 (z, s) .(27)
Then T ′ is defined as follows:
             T ′ a (h, s) = T a (h, s)
for all a ∈ {0, 1}, all s ∈ S and all h ∈ Z \ {z}
T ′ 0 (z ′ , x) = T 0 (z, x) T ′ 0 (z ′′ , u) = T 0 (z, u) for all u ∈ U r T ′ 1 (z ′ , v) = κT 1 (z, v) for all v ∈ V T ′ 1 (z ′′ , v) = (1 -κ)T 1 (z, v) for all v ∈ V . (28
)
All values of T ′ that were not explicitly defined in (28) are set to 0. In words, on the side of S 0 , we move all the parents of z except x to be the parents of z ′′ , while z ′ will have a single parent, x. On the S 1 side, both z ′ and z ′′ will have the same parents as z, with transitions multiplied by κ and 1 -κ respectively. The multiplication by κ is crucial for showing both inequlaities in (10).
Note that T ′ can be though of as splitting z into z ′ and z ′′ , such that z ′ has one parent on the S 0 side, and z ′′ has strictly less parents than z had. Once we show that T ′ satisfies (10), it is clear that by induction we can continue splitting T ′ until we arrive at an invertible representation which can no longer be split, thus proving the Lemma.
In order to show (10) for T ′ , we will sequentially show the following claims:
P (z) = P (z ′ ) + P (z ′′ ) and P (z ′ ) = κP (z) (29
)
P (a|z) = P (a|z ′ ) = P (a|z ′′ ) for a ∈ {0, 1}(30)
   for a = 1 P (s|z, a) = P (s|z ′ , a) = P (s|z ′′ , a) ∀s ∈ S for a = 0, s ∈ U r P (x|z ′ , a) = 1 P (x|z ′′ , a) = 0 P (s|z ′ , a) = 0 P (s|z ′′ , a) = (1 -κ) -1 P (s|z, a)(31)
P (Y |z) = κP (Y |z ′ ) + (1 -κ)P (Y |z ′′ ) (32
) P (z) h(P (Y |z)) ≥ P (z ′ ) h(P (Y |z ′ )) + P (z ′′ ) h(P (Y |z ′′ )) (33
) |µ 0 (z) -µ 1 (z)| = |µ ′ 0 (z ′ ) -µ ′ 1 (z ′ )| + |µ ′ 0 (z ′′ ) -µ ′ 1 (z ′′ )| .(34)
Here the probabilities involving z ′ , z ′′ refer to the representation T ′ . Observe that the left hand side of ( 33) is the contribution of z to the performance cost E t∼Z h(P (Y |Z = t)) of T , while the right hand side of ( 33) is the contribution of z ′ , z ′′ to the performance cost of T ′ . Since all other elements t ∈ Z have identical contributions, this shows the first inequality in (10). Similarly, recall that
∥µ 0 -µ 1 ∥ T V = 1 2 t∈Z |µ 0 (t) -µ 1 (t)| ,(35)
and thus the left hand side of ( 34) is the contribution of z to ∥µ 0 -µ 1 ∥ T V , with the right hand side being the contribution of
z ′ , z ′′ to ∥µ ′ 0 -µ ′ 1 ∥ T V , therefore yielding the claim ∥µ ′ 0 -µ ′ 1 ∥ T V = ∥µ 0 -µ 1 ∥ T V .
Claim (29): By definition,
P (z ′ ) = α 0 β 0 (x)T ′ 0 (z ′ , x) + α 1 s∈V β 1 (s)T ′ 1 (z ′ , s)(36)
= α 0 β 0 (x)T 0 (z, x) + κα 1 s∈V β 1 (s)T 1 (z, s)(37)
= κα 0 s∈U β 0 (s)T 0 (z, s) + κα 1 s∈V β 1 (s)T 1 (z, s)(38)
= κP (z) .
Similarly, by definition we have
P (z ′′ ) = α 0 s∈U r β 0 (s)T 0 (z, s) + (1 -κ)κα 1 s∈V β 1 (s)T 1 (z, s),(40)
and summing this with (37), we obtain P (z) = P (z ′ ) + P (z ′′ ).
Claim (30): Note that it is sufficient to prove the claim for a = 0 since the probabilities sum to 1. Write
P (a = 0|z) = P (a = 0, z) P (z) (41
) = α 0 s∈U β 0 (s)T 0 (z, s) P (z)(42)
= κ • α 0 s∈U β 0 (s)T 0 (z, s) κ • P (z)(43)
= α 0 β 0 (x)T 0 (z, x) P (z ′ )(44)
= P (a = 0|z ′ ) .(45)
Similarly,
P (a = 0|z) = P (a = 0, z) P (z) (46
) = α 0 s∈U β 0 (s)T 0 (z, s) P (z)(47)
= (1 -κ) • α 0 s∈U β 0 (s)T 0 (z, s) (1 -κ) • P (z)(48)
= α 0 s∈U r β 0 (s)T 0 (z, s) P (z ′′ ) (49
) = P (a = 0|z ′′ ) .(50)
Claim (34): By definition, for every representation, µ a (z) = P (Z = z|a) = P(z|a)P(z) P(a)
. Thus, using ( 29),(30) we have for a ∈ {0, 1},
µ a (z ′ ) = κµ a (z) and µ a (z ′′ ) = (1 -κ)µ a (z),(66)
which in turn yields (34).
It remains only to recall that we have derived (33),( 34) under the assumption that |V | > 0. That is, we assumed that the point z which fails invertability on S 0 has some parents in S 1 . The case when |V | = 0, i.e. there are no parents in S 1 can be treated using a similar argument, but is much simpler. Indeed, in this case one can simply split z into z ′ and z ′′ and splitting the S 0 weight between them as before, without the need to carefully balance the interaction of probabilities with S 1 via κ.
this section cite: []

Section: D Uniform Approximation and Two Point Representations
As discussed in Sections 1,4.1, we are interested in showing that all optimal invertible representations, no matter which, and no matter on which set Z ′ , can be approximated using a representation with the following property: For every u ∈ S 0 , v ∈ S 1 , there are at most k points z ∈ Z that have (u, v) as parents, see Figure 2(a). Here k would depend only on the desired approximation degree, but not on Z ′ , or on the exact representation we are approximating. We therefore refer to this result as the Uniform Approximation result. Its implications for practical use were discussed in Section 4.1.
The notation used in this Section was introduced in the beginning of Section C.
To proceed with the analysis, in what follows we introduce the notion of two-point representation.
The main result is given as Lemma D.1 below.
This Section uses the notation of Section 3.3. Let T be an invertible representation, let u ∈ S 0 , v ∈ S 1 be some points, and denote by Z uv = z j k 1 the set of all points z ∈ Z which have u and v as parents. Denote by
w u = k j=1 β 0 (u)T 0 (z j , u) and w v = k j=1 β 1 (v)T 1 (z j , v)(67)
the total weights of β 0 and β 1 transferred by the representation from u and v respectively to Z uv .
Recall that ρ u , ρ v denote the distributions of Y conditioned on u, v. We call the situation above, i.e. the collection of numbers β 0 (u)T 0 (z j , u) j≤k , β 1 (v)T 1 (Z j , v) j≤k , a two point representation, since it describes how the weight from the points u, v is distributed in the representation, independently of the rest of the representation. The contribution of Z uv to the global performance cost is
E uv,T := j≤k P z j h(P Y |z j )(68)
= j≤k α 0 β 0 (u)T 0 (z j , u) + α 1 β 1 (v)T 1 (z j , v) h α 0 β 0 (u)T 0 (z j , u)ρ u + α 1 β 1 (v)T 1 (z j , v)ρ v α 0 β 0 (u)T 0 (z j , u) + α 1 β 1 (v)T 1 (z j , v) ,(69)
while its contribution to the fairness condition is
F uv,T = 1 2 j≤k β 0 (u)T 1 (z j , u) -β 1 (v)T 1 (z j , v) .(70)
Let us now consider two extreme cases of two-point representations. Assume that the total amounts of weight to be represented, w u , w v are fixed. The first case is when k = 1, and this is the maximum fairness case, since in this case the weights w u , w v overlap as much as possible. Indeed, the contributions to the fairness penalty and performance cost in this case are
|w u -w v | and (α 0 w u + α 1 w v ) h α 0 w u ρ u + α 1 w v ρ v α 0 w u + α 1 w v(71)
respectively. The other extreme case is when w u and w v do not overlap at all. This case be realised with k = 2, by sending all w u to z 1 and all w v to z 2 . The fairness and performance contributions would be
w u + w v and α 0 w u • h (ρ u ) + α 1 w v • h (ρ v ) ,(72)
respectively. Note that the fairness penalty is the maximum possible, while the performance cost is the minimum possible (indeed, this is the cost before the representation, and any representation can only increase it, by Lemma A.1). We thus observed that each two points u, v, with fixed total weight w u , w v , can have their own Pareto front of performance-fairness. One could, in principle, fix a threshold γ uv , |w u -w v | ≤ γ uv ≤ w u + w v for the fairness penalty (70), and obtain a performance cost between that in (71) and ( 72). However, it is not clear how large the number of points k should be in order to realise such intermediate representations. In the following Lemma we show that one can uniformly approximate all the points on the two-point Pareto front using a fixed number of points, that depends only on the function h. This means that in practice one can choose a certain number n of z points, and have guaranteed bounds on the possible amount of loss incurred with respect to all representations of all other sizes. Lemma D.1. For every ε > 0, there a number n = n ε depending only on the function h, with the following property: For every two-point representation
β 0 (u)T 0 (z j , u) j≤k , β 1 (v)T 1 (Z j , v) j≤k , with total weights w u , w v , there is a two point representation T ′ on a set Z ′ u,v
, with the same total weights, such that |Z ′ uv | ≤ n, and such that F uv,T ′ ≤ F uv,T and
E uv,T ′ ≤ E uv,T + 2(w u + w v )ε.(73)
Proof. To aid with brevity of notation, define for j ≤ k
c j 0 = α 0 β 0 (u)T 0 (z j , u), c j 1 = α 1 β 1 (v)T 1 (z j , v).(74)
Then we can write
E uv,T = j≤k (c j 0 + c j 1 ) • h c j 0 ρ u + c j 1 ρ v c j 0 + c j 1 , F uv,T = 1 2 j≤k Λc j ,(75)
where Λ is the vector Λ = (α -1 0 , -α -1 1 ), c j = (c j 0 , c j 1 ), and Λc j is the inner product of the two.
Observe that the cost E uv,T depends on c j mainly through the fractions
c j 0 c j 0 +c j 1 .
Our strategy thus would be to approximate all k of such fractions by a δ-net of a size independent of k. To this end, set
p j = c j 0 c j 0 + c j 1 (76
)
and define h uv :
[0, 1] → R by h uv (p) = h(pρ u + (1 -p)ρ v ). (77
)
Since h is continuous (by assumption), and defined on a compact set, it is uniformly continuous, and so is h uv . By definition, this means there is a δ > 0 such that for all p, p ′ with |p -
p ′ | ≤ δ, it holds that |h uv (p) -h uv (p ′ )| ≤ ε. Let us now choose {x i } n i=1 to be a δ net on [0, 1]. For every i ≤ n set Γ i = j | p j -x i ≤ δ, and i is minimal with this property . (78
)
That is, Γ i is the set of indices j such that p j is approximated by x i . Using x i we construct the representation T ′ as follows: For a ∈ {0, 1} set v). Note that the total weights are preserved, i≤n c ′i 0 = w u and i≤n c ′i 1 = w v . Next, for every j ∈ Γ i we have c j 0
c ′i a = j∈Γi c j a . (79
) For n new points, z i ∈ Z ′ uv , set T ′ 0 (z ′i , u) = c ′i 0 /β 0 (u), T ′ 1 (z ′i , v) = c ′i 1 /β 1 (
c j 0 + c j 1 -x i ≤ δ. (80
) Thus c ′i 0 c ′i 0 + c ′i 1 -x i = j∈Γj c j 0 -(c j 0 + c j 1 )x i c ′i 0 + c ′i 1 ≤ j∈Γj δ(c j 0 + c j 1 ) c ′i 0 + c ′i 1 = δ.(81)
Next, observe that by the construction of x i ,
E uv,T - i (c ′i 0 + c ′i 1 )h uv (x i ) = i j∈Γi (c j 0 + c j 1 )h uv (p j ) - i (c ′i 0 + c ′i 1 )h uv (x i ) (82
) ≤ i j∈Γi (c j 0 + c j 1 )ε (83
) = (w u + w v )ε.(84)
In addition,
E uv,T ′ - i (c ′i 0 + c ′i 1 )h uv (x i ) = i (c ′i 0 + c ′i 1 )h uv ( c ′i 0 c ′i 0 + c ′i1
) - i (c ′i 0 + c ′i 1 )h uv (x i ) (85) ≤ (w u + w v )ε,(86)
where we have used (81) in the last transition.
Combining the two inequalities yields the second part of (73),
|E uv,T ′ -E uv,T | ≤ 2(w u + w v )ε.(87)
Finally, note that
i Λc ′i = i j∈Γj Λc j (88
) ≤ i j∈Γj Λc j (89
) = j Λc j , (90
)
yielding the first part of, and thus completing the proof of, statement (73).
It remains to observe that above we have used a δ net for h uv , which depends on ρ u , ρ v . However, we can directly build an appropriate δ-net in full range of h, the simplex ∆ Y , which would produce bounds valid for all u, v. Indeed, let δ ′ be such that |h
(ν) -h(ν)| ≤ ε for all µ, ν ∈ ∆ Y with ∥u -v∥ 1 ≤ δ ′ . Observe that the map p → pρ v + (1 -p)ρ u is 2-Lipschitz from R to ∆ Y equipped with the ∥•∥ 1 norm, for any u, v ∈ ∆ Y . Thus, choosing δ = 1 2 δ ′ , we have |h uv (p) -h uv (p ′ )| ≤ ε if |p -p ′ | ≤ δ.
This completes the proof of the Lemma.
this section cite: []

Section: E Concavity Of E r
Note that the variables r appear in (14) both as coefficients multiplying h and inside the arguments of h, in a fairly involved manner. Nevertheless, the cost turns out to still retain an interesting structure, as it is concave, if h is. We record this in the following Lemma.
Lemma E.1. If h : ∆ Y → R is concave, then of every ρ 1 , ρ 2 ∈ ∆ Y the function g : R 2 → R, given by g((c 1 , c 2 )) = (c 1 + c 2 )h( c1ρ1+c2ρ2 c1+c2 ) is concave.
Proof. It is sufficient to show that for every c, c ′ ∈ R 2 , we have g((c + c ′ )/2) ≥ 1 2 (g(c) + g(c ′ ). To this end, define the map F : R 2 → ∆ Y by
F (c) = c 1 ρ 1 + c 2 ρ 2 c 1 + c 2(91)
and note that
F ((c + c ′ )/2) = F (c) c 1 + c 2 c 1 + c 2 + c ′ 1 + c ′ 2 + F (c ′ ) c ′ 1 + c ′ 2 c 1 + c 2 + c ′ 1 + c ′ 2 . (92
)
It then follows that
g((c + c ′ )/2) = 1 2 (c 1 + c 2 + c ′ 1 + c ′ 2 )h(F ((c + c ′ )/2)) (93
) = 1 2 (c 1 + c 2 + c ′ 1 + c ′ 2 )h F (c) c 1 + c 2 c 1 + c 2 + c ′ 1 + c ′ 2 + F (c ′ ) c ′ 1 + c ′ 2 c 1 + c 2 + c ′ 1 + c ′ 2 (94) ≥ 1 2 (c 1 + c 2 + c ′ 1 + c ′ 2 ) c 1 + c 2 c 1 + c 2 + c ′ 1 + c ′ 2 h(F (c)) + c ′ 1 + c ′ 2 c 1 + c 2 + c ′ 1 + c ′ 2 h(F (c ′ )) (95
) = 1 2 (g(c) + g(c ′ )) .(96)
this section cite: []

Section: F MIFPO Equality Constraint
As noted in the main text, although the inequality constraint ( 16) is convex in the variables r, and can be incorporated directly into most optimisation frameworks, it may be significantly more convenient to work with equality constraints. Using the following Lemma, we can find equivalent equality constraints in a particularly simple form. Lemma F.1. Let µ 0 , µ 1 ∈ ∆ Z be two probability distributions over Z and fix some γ ≥ 0.
If ∥µ 0 -µ 1 ∥ T V = γ then there exist ϕ 0 , ϕ 1 ∈ ∆ Z such that µ 0 + γϕ 0 = µ 1 + γϕ 1 . In the other direction, if there exist ϕ 0 , ϕ 1 ∈ ∆ Z such that µ 0 + γϕ 0 = µ 1 + γϕ 1 , then ∥µ 0 -µ 1 ∥ T V ≤ γ.
The proof may be found in Section G.
As consequence of this result, if we find distributions ϕ 0 , ϕ 1 ∈ ∆ Z such that µ 0 + γϕ 0 = µ 1 + γϕ 1 holds, then we know that (16) also holds, and conversely, if (16) holds, then distributions as above exist.
Using this observation, we introduce new variables, ϕ 0 u,v,j and ϕ 1 u,v,j , for every (u, v, j) ∈ Z, which correspond to ϕ 0 ((u, v, j)) and ϕ 1 ((u, v, j)) respectively. These variables will be required to satisfy the following constraints:
ϕ 0 u,v,j ≥ 0, ϕ 1 u,v,j ≥ 0 ∀(u, v, j) ∈ Z (97
) u,v,j ϕ 0 u,v,j = 1 and u,v,j ϕ 1 u,v,j = 1 (98
) β 0 (u)r 0 u,v,j + γϕ 0 u,v,j = β 1 (v)r 0 v,u,j + γϕ 1 u,v,j ∀(u, v, j) ∈ Z.(99)
Here the first two lines encode the fact that ϕ 0 , ϕ 1 are probabilities, while the third line encodes the fairness constraint, as discussed above.
this section cite: []

Section: G Additional Proofs
Proof Of Lemma F.1
Proof. For this proof it is more convenient to work with the ℓ 1 norm ∥•∥ 1 directly. Recall that
∥µ 0 -µ 1 ∥ T V = 1 2 ∥µ 0 -µ 1 ∥ 1 and that ∥µ 0 -µ 1 ∥ 1 = z |µ 0 (z) -µ 1 (z)| . (100
) Assume that ∥µ 0 -µ 1 ∥ 1 = 2γ. Define the functions φ0 (z) = 1 {µ1≥µ0} (z) • (µ 1 (z) -µ 0 (z)) and φ1 (z) = 1 {µ0≥µ1} (z) • (µ 0 (z) -µ 1 (z)). Note that we then have z φ0 (z) = z φ1 (z) = γ.(101)
Indeed, define
η(z) = 1 {µ0≥µ1} (z) • µ 1 (z) + 1 {µ1≥µ0} (z) • µ 0 (z).(102)
Clearly, η + φ0 = µ 1 and thus z η(z) + φ0 (z) = 1. Similarly, z η(z) + φ1 (z) = 1. Therefore we have
z φ0 (z) = z φ1 (z).(103)
Note also that we can write
2γ = ∥µ 0 -µ 1 ∥ 1 = z |µ 0 (z) -µ 1 (z)| = z φ0 (z) + φ1 (z) ,(104)
which combined with (103) yields (101).
Next, we can also directly verify that
µ 0 + φ0 = µ 1 + φ1 ,(105)
and thus setting ϕ 0 = γ -1 φ0 , ϕ 1 = γ -1 φ1 completes the proof of this direction.
In the other direction, given
ϕ 0 , ϕ 1 ∈ ∆ Z such that µ 0 + γϕ 0 = µ 1 + γϕ 1 , we have z |µ 0 (z) -µ 1 (z)| = γ z |ϕ 1 (z) -ϕ 0 (z)| ≤ 2γ,(106)
thus completing the proof.
this section cite: []

Section: H Experiments
This section describes additional evaluation details and experiments with fair classifiers. In SectionH.1 we provide the main algorithm figure, and discuss technical implementation details. Section H.2 contains the comparison to a number of fair classifiers, and Section H.3 discusses implementation of the entropy cost h within the DCCP framework.
this section cite: []

Section: H.1 Implementations and computational details
Algorithm 1 MIFPO Implementation  For a calibrated classifier, we are using standard model calibration. Model calibration is a wellstudied problem where we fit a monotonic function to the probabilities of some base model so that the probabilities will reflect real probabilities, that is, P (Y |X). Here, we used Isotonic regression (Berta et al., 2024) for model calibration with XGBoost (Chen et al., 2015) as the base model. For training the XGBoost model, a GridSearchCV approach is employed to find the best hyperparameters from a specified parameter grid, using 3-fold cross-validation.
Input: Data {(x i , a i , y i )} i≤N , integers L, k. For a ∈ {0, 1} denote X a = {x i | a i = a}. 1. Learn calibrated classifiers c 0 , c 1 : R d → [0, 1], such that c a (x) ∼ P (Y = 1|X = x, A = a) 2. Construct the histograms {β a (l)} L l=1 , a ∈ {0, 1}, for the sets H a = {c a (x) | (x, a) ∈ D a } ⊂ [0, 1]. Choose bin representatives {ρ l } L l=1
The experiments were conducted on a system with an Intel Core i9-12900KS CPU (16 cores, 24 threads), 64 GB of RAM, and an NVIDIA GeForce RTX 3090 GPU.
this section cite: ['b7', 'b9']

Section: H.2 Additional Evaluations
Following the equivalence between the Pareto fronts of fair binary classification and representations for the accuracy cost (Section 3.1), we evaluated the accuracy-fairness trade-off for some common fair classification algorithms. For our evaluation, we selected the most widely used algorithms based on GitHub repository stars and citation counts in the literature, demonstrating the importance of our proposed method in comparison to common approaches. We also evaluate FairFront, a more recent approach introduced in Wang et al. (2023).
Fair classifiers generally fall into pre-processing, in-processing, and post-processing categories. Pre-and post-processing types often utilize standard classifiers as part of their fair classification pipeline. We specifically evaluated two widely adopted algorithms, representing different categories: FairGBM Cruz et al. (2023): An in-processing method where a boosting trees algorithm (LightGBM) is subjected to pre-defined fairness constraints. Balanced-Group-Threshold Jang et al. (2021): A post-processing method which adjusts the threshold per group to obtain a fairness criterion. For FairGBM, we used the original implementation provided by the authors. For Balanced-Group-Threshold post-processing, we utilized implementations available via Aequitas Saleiro et al. (2018), a popular bias and fairness audit toolkit.
We conducted our evaluation using three of the most common datasets in this field, which are known to have relevance to real-world decision-making processes: the Adult dataset (income prediction), COMPAS (recidivism prediction), and LSAC (law school admission).
It is important to note that, as a rule, common fairness classification methods are not designed to control the fairness-accuracy trade-off explicitly. Instead, in most cases, these methods rely on rerunning the algorithm for a wide range of hyperparameter settings, in the hope that different hyperparameters would result in different fairness-accuracy trade-off points. However, there typically is no direct known and controlled relation between hyperpatameters and the obtained fairness-accuracy trade-off. For FairGBM, we utilized the hyperparameter ranges specified in the original paper, Cruz et al. (2023). In the case of the balancing post-processing method, we conducted a grid search over the full range of all possible hyperparameters to ensure a comprehensive analysis.
Figure 4 shows the MIFPO computed Pareto front, and all hyperparameter runs of the two algorithms above, with accuracy evaluated on the test set. These experiments demonstrate the following two points: (a) The standard classifiers achieve a considerably lower accuracy than what is theoretically possible at a given fairness level. (b) the existing methods are also unable to present solutions for the full range of the statistical parity values. The values from the FGBM and the post-processing algorithms all have statistical parity ≤ 0.2. Similarly to to the case of fair representations, these results emphasize the limitations of current fair classifiers in achieving optimal trade-offs between accuracy and fairness across the full range of fairness values.
Additionally, we add a comparison to FairFront (Wang et al., 2024), which, similarly to our work, depends on estimates of P (Y |X). See the discussion in Section 2. We note, however, that the utility of this comparison is limited due to the very strict setup defined in that paper, which cannot be applied to natural datasets. Specifically: 1) The implementation in Wang et al. ( 2023) requires creating a finite discrete space by binning over the full R d feature spacefoot_2 , which allows for perfect modeling of the distribution P (Y |X) (by simple counting). This is not normally possible on real datasets in practice.
2) The binning itself was performed manually for each dataset separately, and we were unable to discern the logic behind the selected parameters. 3) To allow reasonable binnig, the true dimensions of the data are manually reduced, again by picking features manually for each dataset. Due to these issues, it was practically impossible to apply the method in Wang et al. (2023) to other standard datasets, or even to the datasets used in Wang et al. (2023) but with standard features.
Nonetheless, for the sake of comparison, we compared MIFPO and FairFront on the same restricted, preprocessed version of the datasets that FairFront used, with the same bin-based probability estimator.
The results are shown in Figure 5.
this section cite: ['b37', 'b12', 'b33', 'b12', 'b38', 'b37', 'b37']

Section: H.3 Minimization of Concave Functions under Convex Constraints and the Entropy Loss
As described in figure 5, we used the disciplined convex concave programming (DCCP) framework and the associated solver, (Shen et al., 2016(Shen et al., , 2024) ) for solving the concave minimization with convex constraints problem.
Minimizing concave functions under convex constraints is a common problem in optimization theory. Unlike convex optimization where global minima can be readily found, in concave minimization problems we only know that the local minimas lie on the boundaries of the feasible region defined by the convex constraints. While techniques such as branch-and-bound algorithms, cutting plane methods, and heuristic approaches are often employed, here we used the framework of DCCP which gain a lot of popularity in recent years.
The DCCP framework extends disciplined convex programming (DCP) to handle nonconvex problems with objective and constraint functions composed of convex and concave terms. The idea behind a "disciplined" methodology for convex optimization is to impose a set of conventions inspired by basic principles of convex analysis and the practices of convex optimization experts. These "disciplined" conventions, while simple and teachable, allow much of the manipulation and transformation required for analyzing and solving convex programs to be automated. DCCP builds upon this idea, providing an organized heuristic approach for solving a broader class of nonconvex problems by combining DCP principles with convex-concave programming (CCP) methods, and is implemented as an extension to the CVXPY package in Python.
While convenient, the use of the disciplined framework bears some limitations. Mainly, generic operations like element-wise multiplication are not under the allowed set of operations (and for obvious reasons), which limits the usability. Notice, that for the prediction accuracy measure h(p) = min(p, 1 -p) this is not a problem, but for the entropy classification error h((1 -p, p)) = -p log p -(1 -p) log(1 -p), this is more challenging. Nevertheless, here we show that the standard DCCP framework allows for entropy classification error.
Lemma H.1. Let a = (1 -α)V(v)r v,z and b = α • U(u)r u,z , with p v = p a and p u = p b .
We can write the cost function under the entropy accuracy error as:
(a + b) • entropy a • p a + b • p b a + b = entropy(a • (1 -p a ) + b • (1 -p b )) + entropy(a • p a + b • p b ) -entropy(a + b) Proof. entropy(x) = entropy(x, 1 -x) = x [log(x) -log(1 -x)] + log(1 -x) Thus, entropy a • p a + b • p b a + b = a • p a + b • p b a + b log a • p a + b • p b a + b -log 1 - a • p a + b • p b a + b + log 1 - a • p a + b • p b a + b = a • p a + b • p b a + b [log(a • p a + b • p b ) -log(a + b -a • p a -b • p b )] + log(a + b -a • p a -b • p b ) -log(a + b)
Hence : Given the element-wise entropy function x • (1 -x) is with known characteristics and under the dccp framework, we can use the entropy error for our cost using :
(
=entropy((1 -α)V(v)r v,z • (1 -p v ) + α • U(u)r u,z • (1 -p u )) -entropy((1 -α)V(v)r v,z • p v + α • U(u)r u,z • p u ) -entropy((1 -α)V(v)r v,z + α • U(u)r u,z )
this section cite: ['b34', 'b35']

Section: I Fair Classifiers As Fair Representations
As discussed in Section 3.1, Pareto front of binary classifiers with statistical parity can be computed from the Pareto front of representations with total variation fairness distance. In this Section we provide the proof of this result, Lemma 3.1. The Lemma and the related notation are restated here for convenience.
For a binary classifier Ŷ of Y , its prediction error is defined as usual by ϵ( Ŷ ) := P Ŷ ̸ = Y . The statistical parity distance of Ŷ is defined as
∆ SP ( Ŷ ) := P Ŷ = 1|A = 1 -P Ŷ = 1|A = 0 .(107)
Let the uncertainy measure h be defined by ( 4). Note that the first part of the Lemma does use the special properties of this h and does not necessarily hold for other costs h. Lemma I.1. Let Ŷ be a classifier of Y . Then there is a representation given by a random variable Z on a set Z with |Z| = 2, such that
E z∼Z h(P (Y |Z = z)) ≤ ϵ( Ŷ ) and ∥µ 0 -µ 1 ∥ T V ≤ ∆ SP ( Ŷ ).(108)
Conversely, for any given representation Z, there is a classifier Ŷ of Y as a function of Z (and thus of (X, A)), such that
ϵ( Ŷ ) ≤ E z∼Z h(P (Y |Z = z)) and ∆ SP ( Ŷ ) ≤ ∥µ 0 -µ 1 ∥ T V .(109)
Proof. Let us begin with the second part of the Lemma, inequalities (109). Given a representation
Z, ϵ( Ŷ ) ≤ E z∼Z h(P (Y |Z = z)) follows since E z∼Z h(P (Y |Z = z))
is the error of the optimal classifier of Y as a function of Z. We choose Ŷ to be such an optimal classifier and thus satisfy the above inequality, with equality. Next, the second inequality in (109) holds for for any classifier Ŷ derived from Z. The argument below is a slight generalisation of the argument in Madras et al. (2018). Define f (z) = P Ŷ = 1|Z = z . Note that for a ∈ {0, 1}, P Ŷ = 1|A = a = f (z)dµ a (z). Thus
P Ŷ = 1|A = 0 -P Ŷ = 1|A = 1 = f (z)dµ 0 (z) -f (z)dµ 1 (z)(110)
≤ sup g||g|≤1 g(z)dµ 0 (z) -g(z)dµ 1 (z)(111)
= ∥µ 0 -µ 1 ∥ T V ,(112)
where we have used |f | ≤ 1 in the second line. Repeating the argument also for P Ŷ = 1|A = 1 -P Ŷ = 1|A = 0 , we obtain the second inequality in (109).
We now turn to the first statement, (108). Let Ŷ be a classifier of Y as a function of (X, A).
Observe that thus by definition P Ŷ |X, A induces a distribution on the set {0, 1}, and thus may be considered as a representation Z := Ŷ of (X, A) on that set. We now relate the properties of this Z as a representation to the quantities ϵ( Ŷ ) and ∆ SP ( Ŷ ). Similarly to the argument above, the first part of (108) follows since E z∼Z h(P (Y |Z = z)) is the best possible error over all classifiers. For the second part, note that since Ŷ is binary, we have P Ŷ = 1|A = 0 -P Ŷ = 1|A = 1 = -P Ŷ = 0|A = 0 + P Ŷ = 0|A = 1 . (113) It follows that
∥µ 0 -µ 1 ∥ T V = 1 2 v∈{0,1} P Ŷ = v|A = 0 -P Ŷ = v|A = 1 (114
) = P Ŷ = 1|A = 0 -P Ŷ = 1|A = 1 (115
) = ∆ SP ( Ŷ ),(116)
where we have used (113) for the second to third line transition. This completes the proof of the second part of (108). for all z ∈ Z, y ∈ Y. Write
P (Y = y, Z = z)(129)
= a σ σ(y)P (X ′ = σ, A = a) P (Z ′ = z|X ′ = σ, A = a)(136)
= a σ σ(y)P (Z ′ = z, X ′ = σ, A = a)(137)
= a σ P (Y = y|X ′ = σ, A = a) P (Z ′ = z, X ′ = σ, A = a) ((138)
) = P (Y = y, Z ′ = z) .139
Here, the transition (132)-( 133) is due to the independence condition (1). On line (134) we split the sum over x into sum over subsets {x | f * (x, a) = σ} and an outer some over all σ. The transition (136)-( 137) is due to the equality (122)-(121). Finally, for the transition (137)-( 138), we have used the fact that σ(y
) = P (Y = y|X ′ = σ, A = a) ,(141)
which holds by definition of X ′ . Similarly to the earlier discussion on merging of x with similar value of f * , the above argument proceeded by regrouping the summation over x by the value of f * (x, a). The computation thus showed that this process yields the definition of Z ′ . In particular, this regrouping process and equation ( 141) explain why the space ∆ Y is special and all representations may be factored through it. This completes the proof of the Lemma.
In the above argument we have used the summation over σ, i.e. σ . . .. This is formally possible when (X, A) has a discreet distribution. The full general case may be obtained simply by replacing the summation by integration and conditioning on σ by the general conditional expectation operator.
this section cite: ['b26']

Section: 
Claim (31): For a = 1, let us show P (s|z, a) = P (s|z ′ , a).
P (s|z, a = 1) = α 1 β 1 (s)T 1 (z, s) α 1 s ′ ∈V β 1 (s ′ )T 1 (z, s ′ ) (51
) = κα 1 β 1 (s)T 1 (z, s) κα 1 s ′ ∈V β 1 (s ′ )T 1 (z, s ′ ) (52
) = α 1 β 1 (s)T ′ 1 (z, s) α 1 s ′ ∈V β 1 (s ′ )T ′ 1 (z, s ′ )(53)
= P (s|z ′ , a = 1) .
The statement P (s|z, a) = P (s|z ′′ , a) is shown similarly. Next, for a = 0, we have P (x|z ′ , a) = 1 and P (x|z ′′ , a) = 0 by the definition of the coupling T ′ . Moreover, for s ∈ U r , P (s|z ′ , a) = 0 also follows by the definition of T ′ . Finally, write
P (s|z ′′ , a = 0) = α 0 β 0 (s)T ′ 0 (z ′′ , s) s∈U r α 0 β 0 (s ′ )T ′ 0 (z ′′ , s ′ ) (55
) = α 0 β 0 (s)T 0 (z, s) s∈U r α 0 β 0 (s ′ )T 0 (z, s ′ ) (56
) = α 0 β 0 (s)T 0 (z, s) (1 -κ) s∈U α 0 β 0 (s ′ )T 0 (z, s ′ ) (57
) = (1 -κ) -1 P (s|z, a = 0) .(58)
Claim (32): We first observe that for any representation (and any z),
P (Y = y|Z = z) = s,a P (Y = y, s, a, z) P (z)(59)
where we have used the property (1) for the transition (59)-( 60). Now, using (31), for a = 1 we have
s∈S1 P (Y = y|s, a = 1) P (s|a = 1, z) = s∈S1 P (Y = y|s, a = 1) P (s|a = 1, z ′ ) = s∈S1 P (Y = y|s, a = 1) P (s|a = 1, z ′′ ) .(62)
For a = 0, we have for z ′ using (31):
s∈S0 P (Y = y|s, a = 1) P (s|a = 1, z ′ ) = P (Y = y|x, a = 1) .(63)
For a = 0 and z ′′ we have
s∈S0 P (Y = y|s, a = 1) P (s|a = 1, z ′′ ) = s∈U r P (Y = y|s, a = 1) P (s|a = 1, z ′′ ) (64) = (1 -κ) -1 s∈U r P (Y = y|s, a = 1) P (s|a = 1, z)(65)
where we have used (31) again on the last line.
Combining (62),( 63),(65), and using (30) and the general expression (61), we obtain the claim (32).
Claim (33): This follows immediately from (32) by using ( 29) and the concavity of h.
this section cite: []

Section: J Computational Complexities
In this Section we discuss alternative discretization schemes for the MIFPO algorithm. We also discuss various complexity aspects of the classification algorithms Xian et al. (2023) and Wang et al. (2023) and relate them to the complexity of MIFPO.
In Section 4.2 we described a data independent discretization of ∆ Y by binning. While effective for small label sets Y, larger sets would require a different approach. One alternative is to cluster the data instead of binning ∆ Y itself. Indeed, by choosing cluster centers {η i } M 1 ⊂ ∆ Y , such that each data point {f * (x, a)} (x,a)∈D is well approximated by one the centers (or just most points are approximated), we can guarantee arbitrarily good approximation of the true Pareto front. The cardinality M of such clustering would depend on the intrinsic dimension of the data, which we typically expect to be lower than the full dimension of ∆ Y , due to the Manifold Hypothesis, Fefferman et al. (2016).
Another possibility is to not use any explicit discretization, and instead to use each data point as a separate bin (equivalently, we use the points themselves as the cluster centers {η i }). In this case, the complexity scales with the size of the dataset, but not with the dimensions of ∆ Y . The MIFPO construction in Section 4.1 implies that MIFPO in this case would have O(N 2 ) variables for a dataset of size N . While not applicable for large N , this is similar to the complexity of a variety of often used algorithms. Classical example of such complexity is the Spectral Clustering. We also observe that Xian et al. (2023), a recent state of the art fairness classification algorithm mentioned above, also has such complexity. Indeed, the approach in Xian et al. (2023) involves the computation of a certain transportation plan between data points, and the encoding of such plans also requires O(N 2 ) variables. Thus problem sizes occuring in MIFPO would be smaller or equal to those in Xian et al. (2023), despite the fact that MIFPO is solving a considerably more general problem (see Section 2).
Finally, the algorithm in Wang et al. (2023) involves optimization in the space of confusion matrices, with dimensions of size |A| • |Y| × |Y|. As discussed in Section 2, the reduction of the problem to confusion matrices of possible due to special properties of the classification problem and the accuracy loss.
The algorithm in Wang et al. (2023), FairFront, is an iterative algorithm, where each iteration involves solving a certain difference-of-convex (DC) program which is constructed from a full dataset. The class of DC programs is equivalent to that convex-concave programs considered in this paper (see Shen et al. (2016)). In fact, similarly to MIFPO, the algorithm in Wang et al. (2023) also uses DCCP, Shen et al. (2016), although applied to a different problem.
In each iteration, the solution of DC program is then used to add new constraints to a certain main convex program. While it is proved that asymptotically this process converges to the optimal front, there are no bounds on the number of iterations. This may lead to the convex solver crashing due to too many constraints, and in fact we have observed such crashes in our evaluation.
To summarize 4 , MIFPO involves solving one convex-concave problem, with size which may be independent of the data size. In contrast, FairFront involves iteratively solving convex-concave problems and a main convex program, where the number of terms in the objective of each convexconcave problem scales with the size of the data, and the number of constraints grows in the convex problem grows with iterations, thus making the iterations progressively harder.
this section cite: ['b39', 'b37', 'b15', 'b39', 'b39', 'b39', 'b37', 'b37', 'b34', 'b37', 'b34']

Section: K Factorization
In this Section we the factorization result, Lemma 3.2. We restate the result for convenience.
Recall that f * (x, a) denotes the Bayes optimal classifier of Y , i.e. f * : X × A → ∆ Y is given by
f * (x, a) := P (Y = •|X = x, A = a) .(117)
We define a new variable X ′ , with values in ∆ Y , by X ′ := f * (X, A).
Lemma K.1 (Factorization). For any representation Z of (X, A), there is a representation Z ′ of (X ′ , A), such that
E z ′ ∼Z ′ h(P (Y |Z ′ = z ′ )) = E z∼Z h(P (Y |Z = z)) and D T V (Z ′ ) ≤ D T V (Z).(118)
Proof. The representation Z ′ of X ′ , A will be defined as follows: For σ ∈ ∆ Y , a ∈ A, z ∈ Z set:
P (Z ′ = z|X ′ = σ, A = a) := P (Z = z|X ′ = σ, A = a) (119
) = P (Z = z|f * (X, a) = σ, A = a) (120
)
where the second line is the definition and is added for clarity. To see intuition behind this definition note that we have
P (Z ′ = z|X ′ = σ, A = a) = P (Z = z|f * (X, a) = σ, A = a) (121
) = x∈X P (Z = z|X = x, f * (x, a) = σ, A = a) P (X = x|f * (x, a) = σ, A = a) = x∈X P (Z = z|X = x, A = a) P (X = x|f * (x, a) = σ, A = a)(122)
In words, for a fixed a, to compute P (Z ′ = z|X ′ = σ, A = a) we effectively collect all x such that f * (x, a) = σ and average all of their representations. Equivalently, all points x with the same σ are merged into one point, and their representations summed up according to their relative weight.
We will now show that neither the performance nor the fairness condition change under this operation.
Since D T V (Z) is defined solely in terms of the distributions P (Z = •|A = a), to show that
D T V (Z) = D T V (Z ′ ) it is enough to show that P (Z = •|A = a) = P (Z ′ = •|A = a) ∀a ∈ A.(123)
To this end, we have
P (Z ′ = z|A = a) = σ P (Z ′ = z|X ′ = σ, A = a) P (X ′ = σ|A = a)(124)
= σ P (Z = z|X ′ = σ, A = a) P (X ′ = σ|A = a)(125)
= P (Z = z|A = a) .(126)
where the transition (124) to ( 125) is by the definition of Z ′ . Thus we have shown that D T V (Z) = D T V (Z ′ ).
Next, note that the above argument implies also that P (Z = z) = P (Z ′ = z). Thus, in order to show the performance equality,
E z ′ ∼Z ′ h(P (Y |Z ′ = z ′ )) = E z∼Z h(P (Y |Z = z)),(127)
it is enough to show that P (Y |Z ′ = z) = P (Y |Z = z) for every z ∈ Z. Further, again since P (Z = •) = P (Z ′ = •), we can show that
P (Y = y, Z ′ = z) = P (Y = y, Z = z)(128)
this section cite: []

Section: References
Ref_id:b0 Title: A reductions approach to fair classification Year: (2018)
Ref_id:b1 Title:  Year: ()
Ref_id:b2 Title: Beyond adult and compas: Fair multi-class prediction via information projection Year: (2022)
Ref_id:b3 Title: Fair normalizing flows Year: (2022)
Ref_id:b4 Title: Github for fair normalizing flows Year: (2022)
Ref_id:b5 Title: Fairness and machine learning: Limitations and opportunities Year: (2023)
Ref_id:b6 Title: Concave minimization: theory, applications and algorithms Year: (1995)
Ref_id:b7 Title: Classifier calibration with roc-regularized isotonic regression Year: (2024)
Ref_id:b8 Title:  Year: ()
Ref_id:b9 Title: Xgboost: extreme gradient boosting Year: (2015)
Ref_id:b10 Title: The frontiers of fairness in machine learning Year: (2018)
Ref_id:b11 Title: Elements of Information Theory Year: (2012)
Ref_id:b12 Title: Fairgbm: Gradient boosting with fairness constraints Year: (2023)
Ref_id:b13 Title: Utility-fairness trade-offs and how to find them Year: (2024)
Ref_id:b14 Title: Fairness in deep learning: A computational perspective Year: (2020)
Ref_id:b15 Title: Testing the manifold hypothesis Year: (2016)
Ref_id:b16 Title: Certifying and removing disparate impact Year: (2015)
Ref_id:b17 Title: Obtaining fairness using optimal transport theory Year: (2019)
Ref_id:b18 Title: Disciplined convex programming Year: (2006)
Ref_id:b19 Title: Controllable guarantees for fair outcomes via contrastive information estimation Year: (2021)
Ref_id:b20 Title: Group-aware threshold adaptation for fair classification Year: (2021)
Ref_id:b21 Title: Fare: Provably fair representation learning with practical certificates Year: (2023)
Ref_id:b22 Title: Learning fair representation with a parametric integral probability metric Year: (2023)
Ref_id:b23 Title: Fact: A diagnostic for group fairness trade-offs Year: (2020)
Ref_id:b24 Title: Verified uncertainty calibration. Advances in Neural Information Processing Systems Year: (2019)
Ref_id:b25 Title: Variations and extension of the convex-concave procedure Year: (2016)
Ref_id:b26 Title: Learning adversarially fair and transferable representations Year: (2018)
Ref_id:b27 Title: A survey on bias and fairness in machine learning Year: (2021)
Ref_id:b28 Title:  Year: (2019)
Ref_id:b29 Title: Predicting good probabilities with supervised learning Year: (2005)
Ref_id:b30 Title: Scikit-learn: Machine learning in Python Year: (2011)
Ref_id:b31 Title: A review on fairness in machine learning Year: (2022)
Ref_id:b32 Title: Computational optimal transport: With applications to data science Year: (2019)
Ref_id:b33 Title: Aequitas: A bias and fairness audit toolkit Year: (2018)
Ref_id:b34 Title: Disciplined convex-concave programming Year: (2016)
Ref_id:b35 Title: Github for disciplined convex-concave programming Year: (2024)
Ref_id:b36 Title: Learning controllable fair representations Year: (2019)
Ref_id:b37 Title: Aleatoric and epistemic discrimination: Fundamental limits of fairness interventions Year: (2023)
Ref_id:b38 Title: Aleatoric and epistemic discrimination: fundamental limits of fairness interventions Year: (2024)
Ref_id:b39 Title: Fair and optimal classification via post-processing Year: (2023)
Ref_id:b40 Title: Matching code and law: achieving algorithmic fairness with optimal transport Year: (2020)
Ref_id:b41 Title: Learning fair representations Year: (2013)
Ref_id:b42 Title: Inherent tradeoffs in learning fair representations Year: (2022)
