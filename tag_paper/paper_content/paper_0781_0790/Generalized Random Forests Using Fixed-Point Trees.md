Title: Generalized Random Forests using Fixed-Point Trees
Abstract: We propose a computationally efficient alternative to generalized random forests (GRFs) for estimating heterogeneous effects in large dimensions. While GRFs rely on a gradient-based splitting criterion, which in large dimensions is computationally expensive and unstable, our method introduces a fixed-point approximation that eliminates the need for Jacobian estimation. This gradientfree approach preserves GRF's theoretical guarantees of consistency and asymptotic normality while significantly improving computational efficiency. We demonstrate that our method achieves a speedup of multiple times over standard GRFs without compromising statistical accuracy. Experiments on both simulated and real-world data validate our approach. Our findings suggest that the proposed method is a scalable alternative for localized effect estimation in machine learning and causal inference applications.

Section: Introduction
In many real-world machine learning (ML) applications, practitioners seek to estimate how quantities of interest vary across different feature subgroups rather than assuming uniform effects. For example, medical interventions and policy treatments often have heterogeneous impacts across subpopulations, making localized estimation crucial for improving outcomes (Imai & Ratkovic, 2013;Knaus et al., 2021;Murdoch et al., 2019;Lee et al., 2020). Similarly, individualized recommendation systems adapt to user-specific features to enhance performance (Kohavi et al., 2013).
A key example of localized estimation arises in causal inference, where modern applications prioritize individualized treatment effects over average treatment effects (Neyman, 1923;Rubin, 1974). The double machine learning framework (Chernozhukov et al., 2018) unifies various ML-based causal estimation methods, including lasso (Belloni et al., 2017), random forests (Athey et al., 2019;Cevid et al., 2022), boosting (Powers et al., 2018), deep learning (Johansson et al., 2016;Shalit et al., 2017), and general-purpose meta-algorithms (Nie & Wager, 2021;Künzel et al., 2019), all of which focus on capturing variation over feature space.
Generalized random forests (GRFs) (Athey et al., 2019;Wager & Athey, 2018) have emerged as a powerful tool for such tasks, leveraging adaptive partitioning with problemspecific moment conditions instead of standard loss-based splits. GRFs apply broadly to a wide range of important statistical models -local linear regression (Friedberg et al., 2020), survival analysis and missing data problems (Cui et al., 2023), nonparametric quantile regression, heterogeneous treatment effect estimation, and nonlinear instrumental variables regression (Athey & Imbens, 2016;Athey et al., 2019). Unlike local linear models (Fan et al., 1995;Fan & Gijbels, 1996;Friedberg et al., 2020) or kernel-based models (Staniswalis, 1989;Severini & Staniswalis, 1994;Lewbel, 2007;Speckman, 1988;Robinson, 1988) which suffer from the curse of dimensionality (Robins & Ritov, 1997), the tree-based approach of GRF offers a more scalable solution.
However, GRFs' gradient-based approach (Athey et al., 2019) becomes computationally expensive and unstable in large dimensions due to the reliance on Jacobian estimators for tree splitting. To address this, we propose a gradientfree approach based on fixed-point iteration, eliminating the need for Jacobian estimation while retaining GRF's theoretical guarantees of consistency and asymptotic normality. Our method significantly improves computational efficiency while maintaining statistical accuracy, achieving significant speedups in experiments on simulated and real-world datasets.
this section cite: ['b23', 'b26', 'b32', 'b29', 'b27', 'b34', 'b40', 'b13', 'b6', 'b3', 'b12', 'b37', 'b24', 'b45', 'b35', 'b28', 'b3', 'b49', 'b20', 'b14', 'b2', 'b3', 'b19', 'b18', 'b20', 'b47', 'b44', 'b30', 'b46', 'b39', 'b38', 'b3']

Section: Background and Related Work
Given data (X i , O i ) ∈ X × O, GRF estimates a target function θ * (x), defined as the solution to an estimating equation of the form
0 = E O|X ψ θ * (x),ν * (x) (O) | X = x ,(1)
for all x ∈ X , where ψ is a score function that identifies the true (θ * (x), ν * (x)) as the root of (1), and ν * (x) is an optional nuisance function. GRF can be understood from a nearest-neighbor perspective as approximating θ * (x) through a locally parametric θ * within small neighborhoods of test point x. Suppose L(x) ⊂ {X i } n i=1 is a subset of training observations of the covariates found in a region around x ∈ X over which θ * (x) can be well-approximated by a local parameter. Observations X i ∈ L(x) serve as local representatives for x in estimating θ * (x) such that, given sufficiently many training samples in a small enough neighborhood of x, an empirical version of (1) over X i ∈ L(x) defines an estimator θL(x) that approaches θ * (x), ( θL(x) , νL(x) ) ∈ arg min
θ,ν n i=1 1(X i ∈ L(x)) |L(x)| • ψ θ,ν (O i ) .(2)
In GRF, the set of local representatives L(x) is determined by tree-based partitions which divide the input space into disjoint regions, or leaves. The training samples X i that fall in the same leaf as x form the subset L(x). However, single trees are known to have high variance with respect to small changes in the training data (Amit & Geman, 1997;Breiman, 1996;2001;Dietterich, 2000), leading to estimates (2) that do not generalize well to values of x that are not part of the training set. GRF improves its estimates by leveraging an estimating function that averages many estimating functions of the form (2). Specifically, let L b (x) denote the set of training covariates that fall in the same leaf as x, identified by a tree trained on an independent subsample of the data, indexed by b = 1, . . . , B. The GRF estimator is obtained by aggregating the individual estimating functions (2) across a forest of B independently trained trees, i.e. the solution to the following forest-averaged estimating equation:
( θ(x), ν(x)) ∈ arg min θ,ν
1 B B b=1 n i=1 α bi (x)ψ θ,ν (O i ) .
(3) where α bi (
x) := 1(Xi∈L b (x)) |L b (x)|
. Define observational weights α i (x) that measure the relative frequency with which training sample X i falls in the same leaf as x, averaged over B trees:
α i (x) := 1 B B b=1 α bi (x),(4)
for i = 1, . . . , n. Then, the solution ( θ(x), ν(x)) to the forest-averaged model (3) is equivalent to solving the following locally weighted estimating equation ( θ(x), ν(x)) ∈ arg min θ,ν n i=1 α i (x)ψ θ,ν (O i ) . (5) Athey et al. (2019) present (5) as the definition of the GRF estimator, motivated in part by the mature analyses of local kernel methods (Newey, 1994) alongside more recent work on tree-based partitioning and estimating equations (Athey & Imbens, 2016;Zeileis & Hornik, 2007;Zeileis et al., 2008). The GRF algorithm for estimating θ * (x) can be summarized as a two-stage procedure. Stage I: Use trees to calculate weight functions α i (x) for any test observation x ∈ X , measuring the relative importance of the i-th training sample to estimating θ * (•) near x. Stage II: Given a test observation x ∈ X , compute estimate θ(x) of θ * (x) by solving the locally weighted empirical estimating equation (5).
Our contribution improves the computational cost of Stage I by introducing a more efficient procedure to train the trees.
Training the forest is the most resource-intensive step of GRF, and the cost of each split in the existing approach scales quadratically with the dimension of θ * (x). We adopt a gradient-free splitting mechanism and significantly reduce both the time and memory demands of Stage I. Crucially, solving Stage II with weights α i (x) following our streamlined Stage I produces an estimator θ(x) that preserves the finite-sample performance and asymptotic guarantees of GRF.
this section cite: ['b0', 'b9', 'b17', 'b3', 'b33', 'b2', 'b53', 'b54']

Section: Our Method
In this section we describe the details of our accelerated algorithm for GRF. We closely follow the approach of Athey et al. (2019), and define θ(x) as the solution to a locally weighted problem (5) with weighting functions α i (x) of the form (4). The weight functions are induced by a collection of local subsets {L b (x)} B b=1 , such that each subset L b (x) is determined by the partition rules of a tree trained on a subsample. The construction of each tree, in turn, is determined by recursive splits of the subsample based on a splitting criterion designed to identify regions of X that are homogeneous with respect to θ * (x). Therefore, to fully specify the weight functions α i (x), we must describe a feasible criterion for producing a split of X .
this section cite: ['b3']

Section: The target tree-splitting criterion for Stage I
In GRF, the goal of Stage I is to use recursive tree-based splits of the training data to induce a partition over the input space. Each split starts with a parent node P ⊂ X and results in child nodes C 1 , C 2 ⊂ X , defined by a binary, axisaligned splitting rule of the form C 1 = {X i : X i,ℓ ≤ t} and C 2 = {X i : X i,ℓ > t}, where ℓ denotes a candidate splitting feature/axis and t ∈ R the splitting threshold. For a parent P and any child nodes C 1 , C 2 of P , let ( θP , νP ) and ( θCj , νCj ) denote local solutions analogous to (2) defined over the samples in P and C j , respectively:
( θP , νP ) ∈ arg min θ,ν {i:Xi∈P } ψ θ,ν (O i ) ,
( θCj , νCj ) ∈ arg min
θ,ν {i:Xi∈Cj } ψ θ,ν (O i ) ,(7)
for j = 1, 2. A strategy to split P into two subsets of greater homogeneity with respect to θ * (•) is as follows: Find child nodes C 1 and C 2 such that the total deviation between the local solutions θCj and the target θ * (X) is minimized, conditional on X ∈ C j , j = 1, 2. A natural measure of deviation is the squared-error loss, err(C 1 , C 2 ) := j=1,2
P (X ∈ C j | X ∈ P ) × E θ * (X) -θCj 2 X ∈ C j ,
such that the resulting split (C 1 , C 2 ) corresponds to least-squares optimal solutions θC1 and θC2 . However, err(C 1 , C 2 ) is intractable since θ * (•) is unknown. GRF considers a criterion that measures heterogeneity across a pair of local solutions over a candidate split
∆(C 1 , C 2 ) := n C1 n C2 n 2 P θC1 -θC2 2 ,(8)
where n C1 , n C2 , and n P denote the number of observations in C 1 , C 2 , and P , respectively. In particular, rather than minimizing err(C 1 , C 2 ), one can seek a split of P such that the cross-split heterogeneity between θC1 and θC2 is maximized. Athey et al. (2019) observe that err(C 1 , C 2 ) and ∆(C 1 , C 2 ) are coupled according to err(C 1 , C 2 ) = K(P ) -E [∆(C 1 , C 2 )] + o(r 2 ), where r > 0 is a small radius term tied to the sampling variance, and K(P ) does not depend on the split of P . That is, splits that maximize ∆(C 1 , C 2 ) -which emphasize the heterogeneity of θCj across a split -will asymptotically minimize err(C 1 , C 2 ), which aims to improve the homogeneity of θCj within a split.
Although the criterion ∆(C 1 , C 2 ) is computable, evaluating it is very computationally expensive since it requires solving (7) to obtain θC1 , θC2 for all possible splits of P , and closedform solutions for θCj are generally not available except in special cases of ψ. Instead, GRF approximates the target ∆-criterion based on a criterion of the form
∆ grad (C 1 , C 2 ) := n C1 n C2 n 2 P θgrad C1 -θgrad C2 2 , (9
)
where θgrad Cj denotes a gradient-based approximation of θCj .
Specifically, θgrad Cj is a first-order approximation interpreted as the result of taking a gradient step away from the parent estimate in the direction towards the true child solution θCj :
θgrad Cj := θP - 1 n Cj {i:Xi∈Cj } ξ ⊤ A -1 P ψ θP ,ν P (O i ), (10
)
where ( θP , νP ) is the local solution over the parent, A P is any consistent estimator of the local Jacobian matrix
∇ (θ,ν) E[ψ θP ,ν P (O i ) | X i ∈ P ]
, and ξ ⊤ can be thought of as a term that selects a θ-subvector from a (θ, ν)-vector, e.g. if θ ∈ R K and ν ∈ R, then ξ ⊤ such that θ = ξ ⊤ (θ, ν) ⊤ is the rectangular diagonal matrix ξ ⊤ = [I K 0]. When the scoring function ψ is continuously differentiable in (θ, ν), the Jacobian estimator A P can be computed as
A P = ∇ (θ,ν) 1 n P {i:Xi∈P } ψ θP ,ν P (O i ) = 1 n P {i:Xi∈P } ∇ (θ,ν) ψ θP ,ν P (O i ).(11)
this section cite: ['b3']

Section: Limitations of gradient-based approximation
The use of the Jacobian estimator A P in (10) introduces considerable computational challenges. First, each parent node P in every tree of the forest requires a distinct A P matrix, which imposes a significant computational burden when explicitly calculating A -1 P ψ θP ,ν P (O i ) to determine θgrad Cj . Second, if the local Jacobian ∇ (θ,ν) E[ψ θP ,ν P (O i ) | X i ∈ P ] is ill-conditioned, then the resulting A P estimator may be nearly singular. This instability can lead to highly variable gradient-based approximations θgrad Cj and highly variable splits of P . For example, consider the following varying-coefficient model for an outcome Y i given regres-
sors W i = (W i,1 , . . . , W i,K ) ⊤ in the presence of mediating auxiliary covariates X i : E[Y i | X i = x] = ν * (x) + W ⊤ i θ * (x),(12)
where ν * (•) is a nuisance intercept function and
θ * (x) = (θ * 1 (x), . . . , θ * K (x)) ⊤
are the target coefficients. Models of the form (12) encompass time-or spatially-varying coefficient frameworks, where (X i , Y i , W i ) represent the i-th sample associated with spatiotemporal values X i . Such models are particularly relevant in applications like heterogeneous treatment effects; see Section 5 for a more in-depth discussion. The local estimating function ψ θ,ν (Y i , W i ), identifying (θ * (x), ν * (x)) through moment conditions as in (1), is given by:
ψ θ,ν (Y i , W i ) := (Y i -W ⊤ i θ -ν) • W i Y i -W ⊤ i θ -
ν . 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98 0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00 Regressor Correlation Split Value Method grad FPT Boxplots: Split values over regressor correlations 1e-4 1e-3 1e-2 1e-1 0.80 0.85 0.90 0.95 Regressor Correlation Split Variance Method grad FPT Median split variance over 250 replications Consequently, the corresponding local Jacobian estimator is
A P = 1 n P {i:Xi∈P } ∇ (θ,ν) ψ θ,ν (Y i , W i ) = - 1 n P {i:Xi∈P } W i W ⊤ i W ⊤ i W i 1 .(13)
When the regressors are highly correlated, the summation over the W i W ⊤ i block of the A P matrix leads to nearly singular values of A P , resulting in an unstable matrix inverse A -1 P , and therefore unstable values of θgrad Cj and unstable splits. This issue becomes more pronounced as the number of parent samples n P decreases, as is the case at deeper levels of the tree. These challenges highlight the limitations of relying on A P as part of an approximation for the child solutions θCj .
As an illustration, consider a simple varying coefficient model with primary regressors W i,1 , W i,2 ∼ N (0, 1), auxiliary covariates X i ∼ Unif(0, 1), and outcomes Y i generated as
Y i = 1(X i > 0.5)W i,1 + W i,2 + ϵ i ,(14)
where ϵ i ∼ N (0, 1). Figure 1 illustrates the distribution of 2000 ∆ grad -optimal binary splits (gradient-based tree stumps) fit over 1000 samples of the varying coefficient model ( 14), repeated over different regressor correlation levels Corr(W i,1 , W i,2 ) ∈ {0.80, 0.81, . . . , 0.98, 0.99}. It is clear that splits based on the ∆ grad -criterion exhibit high variability when the correlation between the regressors is large. In contrast, our proposed method, discussed in the next section, does not suffer from the same problem.
this section cite: []

Section: Fixed-point approximation
To address the limitations of gradient-based approximations, we propose a gradient-free approach based on the form of a single fixed-point iteration. Let Ψ Cj (θ, ν) := 1 n C j {i:Xi∈Cj } ψ θ,ν (O i ) denote the empirical estimating function for the child solution ( θCj , νCj ) such that (7) is equivalently written as:
( θCj , νCj ) ∈ arg min θ,ν Ψ Cj (θ, ν) , j = 1, 2. (15)
Under mild regularity conditions, ( θCj , νCj ) is a Zestimator that solves the estimating equation Ψ Cj (θ, ν) = 0. Reformulating this equation as a fixed-point problem, we write:
(θ, ν) = (θ, ν) -ηΨ Cj (θ, ν) =:f (θ,ν) , η > 0.(16)
A necessary and sufficient condition for ( θCj , νCj ) to be a solution of ( 15) is characterized by the fixed-point problem ( θCj , νCj ) = f ( θCj , νCj ), where f is as defined in ( 16). Iterative fixed-point methods (Picard, 1890;Lindelöf, 1894;Banach, 1922;Ryu & Boyd, 2016;Yang et al., 2021) solve such problems by considering an update rule of the form
(θ + , ν + ) ← f (θ, ν).(17)
The form of (17) inspires us to approximate the true child solution θCj using a single fixed-point update taken from the parent solution θP :
θFPT Cj := θP -ηξ ⊤ Ψ Cj ( θP , νP ) = θP - η n Cj ξ ⊤ {i:Xi∈Cj } ψ θP ,ν P (O i ),(18)
where the product with ξ ⊤ is interpreted similarly to its role in the gradient-based approximation (10) and to express the update (17) solely in terms of the target θ-quantity. We interpret θFPT Cj as an approximation of θCj obtained by taking a step from θP in a direction that reduces the magnitude of the local estimating function Ψ Cj . Notably, the approximation θFPT Cj does not involve the A P matrix, relying only on the scores ψ θP ,ν P (O i ) evaluated at the parent solutions. In general, removing the inverse A -1 P provides computational cost savings of O(K 3 ). The corresponding splitting criterion, which uses the fixed-point approximations θFPT Cj as substitutes for θCj is given by
∆ FPT (C 1 , C 2 ) := n C1 n C2 n 2 P θFPT C1 -θFPT C2 2 . (19
)
Revisiting the varying coefficient example from Section 3.2, we see that splits based on fixed-point approximations θFPT Cj are significantly more stable than those based on θgrad Cj . Specifically, Figure 1 illustrates that splits that maximize ∆ FPT (C 1 , C 2 ) are more robust to ill-conditioning in the underlying local Jacobian ∇ (θ,ν) E[ψ θP ,ν P (O i ) | X i ∈ P ], as is the case for highly correlated regressors in the varying coefficient model ( 14), and leading to highly stable splits.
this section cite: ['b36', 'b31', 'b5', 'b41', 'b52']

Section: Pseudo-outcomes
Approximations θCj of the form (10) and ( 18) offer an additional benefit: they enable the ∆-criteria of the form (9) and ( 19) to be efficiently optimized through a single multivariate CART split. A CART split performed with respect to vectorvalued responses ρ i ∈ R K over a parent node P produces a split (C 1 , C 2 ) that minimizes the following least-squares criterion:
{i : Xi∈C1} ∥ρ i -ρC1 ∥ 2 + {i : Xi∈C2} ∥ρ i -ρC2 ∥ 2 , (20
)
where ρCj :=foot_2 n C j {i:Xi∈Cj } ρ i . 1 Equivalently, a CART split that minimizes (20) will maximize:
n C1 ∥ρ C1 ∥ 2 + n C2 ∥ρ C2 ∥ 2 . (21
)
The equivalence between the split that minimizes the leastsquares CART criterion (20) and the split that maximizes (21) is shown in Appendix B.1.1. GRF performs its splits by adopting gradient-based pseudo-outcomes, defined as
ρ grad i := -ξ ⊤ A -1 P ψ θP ,ν P (O i )(22)
such that the gradient-based approximation θgrad Cj in (10) is equivalently written:
θgrad Cj = θP + 1 n Cj {i:Xi∈Cj } ρ grad i = θP + ρ grad Cj .
In the case of fixed-point approximation, we define fixedpoint pseudo-outcomes:
ρ FPT i := -ηξ ⊤ ψ θP ,ν P (O i ), η ̸ = 0,(23)
such that the fixed-point approximation θFPT Cj in (18) is equivalently written as
θFPT Cj = θP + 1 n Cj {i:Xi∈Cj } ρ FPT i = θP + ρ FPT Cj . (24
)
Substitute the above form of θFPT Cj into the ∆ FPT -criterion (19) to equivalently express the criterion in terms of the FPT pseudo-outcomes:
∆ FPT (C 1 , C 2 ) = n C1 n C2 n 2 P ρ FPT C1 -ρ FPT C2 2 ,(25)
where an analogous equivalence holds for ∆ grad in terms of the gradient-based pseudo-outcomes. We demonstrate in Lemma B.1 (in Appendix B.1.2) that maximizing the fixedpoint criterion ∆ FPT (C 1 , C 2 ) is equivalent to maximizing the CART criterion (21), and extend this property to any ∆-style criterion induced by pseudo-outcomes that can be expressed as a split-independent linear transformation of the parent scores ψ θP ,ν P (O i ).
Note that our method does not rely on iterative fixed-point procedures at all. Instead, it uses only a single step of fixed-point approximation to simplify the pseudo-outcomes. These simplified pseudo-outcomes are then passed directly to a standard CART algorithm for splitting.
The numerical convergence of our method therefore relies solely on CART's established and well-known stability, not on fixedpoint iteration. CART splits on pseudo-outcomes are computationally efficient. Given a parent node P , the value ρ i = -Bψ θP ,ν P (O i ) does not depend on a candidate split (C 1 , C 2 ) for any matrix B that is fixed with respect to the parent. This allows much of the computation required to maximize ∆ FPT (C 1 , C 2 ) to be done at the parent level, and in particular avoids re-calculating the approximations θFPT C1 and θFPT C2 across the sequence of candidate splits. Once P is fixed and ρ FPT i are computed, the value of ∆ FPT (C 1 , C 2 ) for the first candidate split requires O(n P ) time, and the value for all other candidate splits of P are queried in O(1) time. While gradient-based pseudo-outcomes share this property, the use of fixed-point pseudo-outcomes eliminates the computational overhead and instability associated with estimating A P , as discussed in Section 3.2. We show in Lemma B.2 (Appendix B.1.3) that choosing different values of η does not change the outcome of the fixed-point splitting mechanism. Specifically, the optimal split identified by CART on pseudo-outcomes ρ FPT i of the form (23) does not depend on η. This can be heuristically understood by studying how the criterion changes as a function of the candidate splits. To illustrate, we consider a VCM model of the form (12) for bivariate regressors W i , univariate X i ∈ [0, 1], and scalar outcomes Y i . A detailed summary of the settings is found in Appendix D.1. The sequence of valid candidate child nodes obtained by a split over univariate X i can be parameterized through scalar t as C 1 (t) := {X i : X i ≤ t} and C 2 (t) := {X i : X i > t}. Let ∆(t) := ∆(C 1 (t), C 2 (t)) denote the parameterized target criterion (8), and consider the behavior of ∆(t), ∆ grad (t), and two fixed-point criteria ∆ FPT 1 (t) and ∆ FPT 2 (t) of the form (25) based on pseudooutcomes with scale factors η = 1 and η = 1/ √ 2, respectively. Figure 2 illustrates the different splitting criteria values plotted against the sequence of candidate splits. The visualization clearly shows that the criteria curves for ∆(t), ∆ grad (t), and ∆ FPT 1 (t) with η = 1 are all very close to one 0.0 0.2 0.4 0.00 0.25 0.50 0.75 1.00 Threshold t Criterion value ∆ ∆ ∆ ~grad ∆ ~1 FPT η = 1 ∆ ~2 FPT η = 1 2 C 1 = {X i : X i ≤ t } and C 2 = {X i : X i > t } Criterion values ∆(C 1 ,C 2 ) over candidate splits {C 1 ,C 2 } Figure 2: Criterion values across candidate splits (C1(t), C2(t)) over threshold t ∈ [0, 1]. The location of the optimal split under each criterion is given by the corresponding vertical line.
another. Critically, the fixed-point criterion with η = 1/ √ 2, i.e. ∆ FPT 2 (t), although scaled differently, still identifies the same maximizing split as ∆ FPT 1 (t). This is because CART chooses a split based on a rank ordering of the criterion over all candidate splits. The absolute scale of the CART criterion does not matter, and it is only criterion rankings over the candidates that determines the optimal split. Therefore, choosing a different scalar η does not change the outcome of the splitting process.
Based on the scale-invariance of our splitting criterion, we now detail the recursive procedure for growing our fixedpoint trees pseudo-outcomes with η = 1.
The fixed-point tree algorithm. The entire fixed-point tree-growing procedure recursively applies the following two steps on a given parent node P :
(i) Labeling: Solve (6) over P to obtain the parent estimate ( θP , νP ). Compute the pseudo-outcomes:
ρ FPT i := -ξ ⊤ ψ θP ,ν P (O i ),(26)
for all i such that X i ∈ P .
(ii) Regression: Maximize ∆ FPT (C 1 , C 2 ) by performing a CART split on the pseudo-outcomes ρ FPT i over P .
this section cite: []

Section: Estimates of θ(x) for Stage II
The fixed-point tree algorithm generates a single tree-based partition of X . Repeating this process over subsamples of the training data yields a forest of trees, each specifying local leaf functions L b (x). These leaf functions define the local weight functions α i (x) via (4), completing Stage I of GRF. The full fixed-point tree training algorithm is described in Algorithm 1, while Algorithm 2 provides the pseudocode for the forest-wide Stage I procedure.
To compute the final GRF estimates θ(x) for the target θ * (x), we follow the standard GRF mechanism for Stage II. After the fixed-point trees are trained in Stage I, a test observation x 0 ∈ X is assigned to local leaves L b (x 0 ), indexed by trees b ∈ {1, . . . , B}. Each leaf L b (x 0 ) contains the training observations that fall into the same leaf as x 0 in tree b. Using these local leaves, the forest computes training weights α i (x 0 ) as in (4). The final estimate θ(x 0 ) is obtained by solving the locally weighted estimating equation (5).
Importantly, as discussed in Section 2, solving for θ(x 0 ) in Stage II is independent of the specific mechanism used in Stage I. The only requirement is that Stage I produces valid weights. This ensures that Stage II remains a standard weighted estimating equation, enabling the fixed-point tree algorithm to integrate seamlessly into GRF's two-stage framework. We refer to the complete algorithm for estimating θ * (x) using fixed-point trees as GRF-FPT. By preserving Stage II of GRF, the GRF-FPT estimator θ(x) retains GRF's theoretical guarantees of consistency and asymptotic normality while offering a computationally efficient tree-building method. Pseudocode for Stage II of the GRF-FPT algorithm is provided in Algorithm 3, located in Appendix C.3.
this section cite: []

Section: Theoretical Analysis
In this section, we provide a theoretical foundation for the GRF-FPT estimator θ(x). For Stage I, Proposition 4.1 establishes an asymptotic equivalence between the FPT criterion and a weighted oracle criterion ∆ V (C 1 , C 2 ) in (27), while Lemma 4.2 demonstrates that the Specifications A.2 are met by a forest based on the ∆ V -criterion whenever they are met by a forest based on the ∆-criterion. Assumptions A.1 and Specifications A.2 are the sufficient conditions for the consistency and asymptotic normality of θ(x) in (5), and thus are used to formally justify the FPT algorithm as a mechanism for specifying an estimator of θ * (x).
Proposition 4.1. Suppose Assumptions A.1 hold, and assume moreover Neyman orthogonal moment conditions (defined in Appendix A.4). Denote by r := sup {i:Xi∈P } ∥X i -x P ∥ the radius of the parent P , where
x P denotes the center of mass over X i ∈ P . Let V θθ (x P ) denote the θ-block of V (x P ) in (37). Denote by ∥•∥ V the weighted Euclidean norm ∥z∥ V := ∥V θθ (x P )z∥ 2 = z ⊤ V ⊤ θθ (x P )V θθ (x P )z. Define the weighted oracle crite- rion ∆ V (C 1 , C 2 ): ∆ V (C 1 , C 2 ) := n C1 n C2 n 2 P θC1 -θC2 2 V .(27)
Then, treating the split as fixed with r -2 ≪ n C1 , n C2 and sufficiently small r > 0,
∆ FPT (C 1 , C 2 ) = ∆ V (C 1 , C 2 ) + o P r 2 , 1 n C1 , 1 n C2 .
Lemma 4.2. Let T (∆) denote a tree whose splitting mechanism seeks splits that maximize ∆(C 1 , C 2 ) defined in (8), and let T (∆ V ) denote a tree whose splitting mechanism seeks splits that maximize ∆ V (C 1 , C 2 ) defined in (27). Suppose Assumptions A.1 hold and assume moreover that T (∆) is a tree that satisfies Specifications A.2. Then, T (∆ V ) satisfies Specifications A.2.
this section cite: []

Section: For Stage II, Theorem 4.3 establishes the consistency of the GRF-FPT estimator θ(x):
Theorem 4.3. Suppose that Assumptions A.1 hold, and let ( θ(x), ν(x)) be estimates that solve (5) based on weights induced by a forest of trees grown under the fixed-point tree algorithm satisfying Specifications A.2. Then,
( θ(x), ν(x)) converges in probability to (θ * (x), ν * (x)).
The proof of Theorem 4.3 follows directly from Theorem 3 of Athey et al. (2019), which, under Assumptions A.1, establishes consistency for estimates ( θ(x), ν(x)) that solve (5) with weights from a forest that satisfies Specifications 1-5. Thanks to Lemma 4.2, these forest specifications must also apply to a forest grown under the FPT mechanism. Specifications 1-3 collectively impose mild boundary conditions on the splitting procedure. Meanwhile, Specification 4 requires that trees are trained on subsamples drawn without replacement (Biau et al., 2008;Scornet et al., 2015;Wager et al., 2014;Wager & Athey, 2018), and Specification 5 requires that trees must be grown using an additional subsample splitting mechanism known as honesty (Athey & Imbens, 2016;Biau, 2012;Denil et al., 2014). Appendix C.1 provides a detailed explanation of the subsampling and honest sample splitting procedure.
Finally, Theorem 4.4 establishes the asymptotic normality of the GRF-FPT estimator θ(x):
Theorem 4.4. Under the conditions of Theorem 4.3, suppose moreover that Regularity Condition 1 holds, and that a forest is grown on subsamples of size s scaling as s = n β , where β satisfies Regularity Condition 2. Then, there exists a sequence σ n (x) such that ( θn (x) -θ * (x))/σ n (x) ⇝ N (0, 1) and σ 2 n (x) = polylog(n/s) -1 s/n, where polylog(n/s) is a function that is bounded away from 0 and increases at most polynomially with the log of the inverse sampling ratio log(n/s).
The proof of Theorem 4.4 is an immediate consequence of Theorem 5 of Athey et al. (2019). Theorems 4.3 and 4.4 demonstrate that the GRF-FPT estimator is able to meet key statistical guarantees.
this section cite: ['b3', 'b8', 'b42', 'b51', 'b49', 'b2', 'b7', 'b16', 'b3']

Section: Applications
In this section, we explore applications of GRF-FPT for two related models: varying coefficient models and heterogeneous treatment effects. We consider an outcome model of the form introduced in Section 3.2. For each observation, let Y i denote the observed outcome, W i = (W i,1 , . . . , W i,K ) ⊤ a K-dimensional regressor, and X i a set of mediating auxiliary variables, such that
Y i = ν * (X i ) + W ⊤ i θ * (X i ) + ϵ i ,(28)
where
ν * (•) is a nuisance intercept function, θ * (x) = (θ * 1 (x), . . . , θ * K (x)) ⊤ are the target effect functions local to X i = x, under the assumptions E[ϵ i | X i = x] = 0 and E[ϵ i W i | X i = x] = 0.
Varying coefficient models (VCM). Given regressors W i ∈ R K , models of the form (28) can be characterized as varying coefficient models (Hastie & Tibshirani, 1993). As discussed in Section 3.2, we must also assume that the regressors W i are conditionally exogenous given X i = x.
this section cite: ['b22']

Section: Heterogeneous treatment effects (HTE).
A special case of ( 28) arises within the Neyman-Rubin potential outcome framework, which models the causal effect of treatment on an outcome (Neyman, 1923;Rubin, 1974). Here,
θ * (x) = (θ * 1 (x), . . . , θ * K (x)) ⊤
represents heterogeneous treatment effects associated with K discrete treatment levels. Let T i ∈ {1, . . . , K} denote the observed treatment level for the i-th observation, and Y i (k) the potential outcome that would have been observed if treatment level k had been applied. The regressors W i ∈ {0, 1} K in (28) are interpreted as a vector of dummy variables indicating the observed treatment level, W i,k := 1(T i = k). The auxiliary variables X i account for potential confounding effects. The conditional average treatment effect of treatment level k ∈ {2, . . . , K} relative to the baseline level k = 1 is then defined as:
θ * k (x) := E [Y i (k) -Y i (1) | X i = x] ,
where the baseline contrast is set to θ * 1 (x) := 0. Under exogeneity of the regressors, the target effects θ * (x) in models (28) are identified by moment conditions (1) for scoring function (Angrist & Pischke, 2009;Athey et al., 2019)
ψ θ,ν (Y i , W i ) := (Y i -W ⊤ i θ -ν) • W i Y i -W ⊤ i θ -ν .
The gradient-based pseudo-outcomes (22) are computed as
ρ grad i = -A -1 P (W i -W P ) Y i -Y P -(W i -W P ) ⊤ θP ,(29)
A P = - 1 n P {i:Xi∈P } (W i -W P )(W i -W P ) ⊤ . (30
)
Computing ρ grad i in ( 29) involves the OLS coefficients θP from regressing Y i -Y P on W i -W P , over the observations in P :
θP := -A -1 P 1 n P {i:Xi∈P } (W i -W P )(Y i -Y P ). (31
)
In comparison, ρ FPT i in ( 26) are computed as:
ρ FPT i := -ξ ⊤ ψ θP ,ν P (Y i , W i ), = -(W i -W P ) Y i -Y P -(W i -W P ) ⊤ θP ,(32)
The relationship ρ grad
i = A -1 P ρ FPT i
reveals a significant benefit of FPT pseudo-outcomes. The form of ρ FPT i eliminates the computational cost associated with the multiplication of A -1 P , leading to O(K 3 ) computational savings. Furthermore, the computation of θP in (32) no longer requires solving for A -1 P . Therefore, we can further enhance computational efficiency by using an accelerated form of pseudo-outcome ϕ FPT i instead of ρ FPT i :
ϕ FPT i := -(W i -W P ) Y i -Y P -(W i -W P ) ⊤ θP ,(33)
where θP is replaced by θP in (32), which is defined as a one-step gradient descent approximation of θP taken from the origin:
θP := γ 1 n P {i:Xi∈P } (W i -W P )(Y i -Y P ).(34)
Here, γ denotes the exact line search step size for the regression of
Y i -Y P on W i -W P over P : γ := (W -W P ) ⊤ (Y -Y P ) 2 2 (W -W P )(W -W P ) ⊤ (Y -Y P ) 2 2 ,(35)
where
W = [W 1 • • • W n P ] ⊤ and Y = [Y 1 • • • Y n P ] ⊤ with
the notation W -W P and Y -Y P understood as row-wise centering.
The computational cost associated with θP is comparatively small because many of the products that appear in (34) and ( 35) are already computed as part of ρ FPT i in (32). Meanwhile, we show in Appendix B.3 that the approximation for the FPT child estimator:
θFPT Cj := θP + 1 n Cj {i:Xi∈Cj } ϕ FPT i ,
is consistent for the original FPT child estimator θFPT Cj as ∥ θFPT
Cj -θFPT Cj ∥ = o P (1), meaning that this approximation does not alter the asymptotic behavior of our estimator. These accelerations are particularly compelling when the dimension of θ * (x) is large and computational efficiency is critical, as in large-scale A/B testing with multiple concurrent treatment arms or observational studies with numerous treatment levels (Kohavi et al., 2013;Bakshy et al., 2014).
this section cite: ['b34', 'b40', 'b1', 'b3', 'b27', 'b4']

Section: Simulations
In this section, we perform empirical evaluations of the computational efficiency and estimation accuracy of the GRF-FPT method. We let GRF-FPT1 denote the FPT algorithm using the exact form of the FPT VCM/HTE pseudooutcomes (32) and we let GRF-FPT2 denote the accelerated FPT algorithm based on the form of the FPT pseudooutcome approximation (33) in Section 5. We compare both implementations relative to GRF-grad under VCM and HTE designs. Implementation details and links to the reproducible code are found in Appendix C.4.
this section cite: []

Section: Settings.
We follow the structural model in (28). The auxiliary variables X i are drawn from the Gaussian copula with latent covariance matrix Σ, where
[Σ] j,k = (0.3) |j-k| . Supporting experiments for multicollinearity in X i can be found in Appendix D.2. The outcomes Y i follow (28) with Gaussian noise ϵ i ∼ N (0, 1). For VCM experiments, re- gressors W i ∈ R K are sampled from N K (0, I). For HTE experiments, W i ∈ {0, 1} K follows a multinomial distribu- tion, W i | X i = x ∼ Multinomial(1, (π 1 (x), . . . , π K (x))),
where π k (x) is the probability of treatment level k ∈ {1, . . . , K}, characterizing a variety of different locationspecific dependence structures through the setting of π k (•). We set ν * (x) := 0 and vary the target effect functions θ * k (x) and treatment probabilities π k (x) across different settings, fully detailed in Appendix C.4. Throughout our experiments we use subsampling ratio s/n = 0.5. Supporting experiments under different subsample ratios are found in Appendix D.2.
Results. The relative computational advantage of forests trained under GRF-FPT is displayed in Figure 3, while Figure 5 (in Appendix D.3) summarizes the absolute fit times across the three methods. These data show that the FPT mechanism is able to consistently offer a relative advantage, observing speedups of up to 3.5× faster than the gradientbased approach at the largest dimension K = 256. Figure 3 also shows increasing gains with increasing K and provides an empirical measurement of the theoretical scaling benefits discussed in Section 5. Moreover, the absolute fit times in Figure 5 (in Appendix D.3) illustrate that our method consistently remains faster than GRF-grad, with no clear computational or algorithmic bottleneck as a function of either n or K. Supporting experiments exploring the ef- fects of sample sizes up to n = 500, 000 are presented in Appendix D.2, while Figures 7 and 8 (in Appendix D.3) show that even when n is small, GRF-FPT still observes a noticeable gain relative to GRF-grad. Additional timing benchmarks for VCM experiments and all HTE experiments are discussed in Appendix D.3.
dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) = 5 dim(X) =
To assess estimation accuracy, we evaluate the mean squared error (MSE) of θ(x) across 50 replications of the model and testing on a separate set of 5, 000 observations. Figure 6 in Appendix D.3 confirms that GRF-FPT matches the accuracy of GRF-grad, while significantly reducing computation time. Further comparisons for both VCM and HTE settings are provided in Appendix D.3.
this section cite: []

Section: Real Data Application
Data. In this section we apply GRF-FPT to the analysis of geographically-varying effects θ * (x) on housing prices. The data, first appearing in Kelley Pace & Barry (1997), contains 20,640 observations of housing prices taken from the 1990 California census. Each observation corresponds to measurements aggregated over a small geographical census block, and contains measurements of 9 variables: median housing value, longitude, latitude, median housing age, total rooms, total bedrooms, population, households, and median income. We employ a VCM design of the form (28) where Y i denotes the housing value, X i denote the spatial coordinates, and W i = (W i,1 , . . . , W i,6 ) ⊤ are the remaining six regressors. Details of the model and data transformations used for the California housing analysis is found in Appendix F.
Results. Table 7 summarizes the computational benefit of GRF-FPT applied to the California housing data. Figure 4 illustrates the six geographically-varying effect estimates
San Francisco Los Angeles Sacramento San Diego San Francisco Los Angeles Sacramento San Diego San Francisco Los Angeles Sacramento San Diego San Francisco Los Angeles Sacramento San Diego San Francisco Los Angeles Sacramento San Diego San Francisco Los Angeles Sacramento San Diego log Population log Census Block Bedrooms log Census Block Rooms log Households Median Housing Age log Median Income Effect on log value -3 -2 -1 0 1 2 3 Spatially-varying effects on log median house values California housing data: GRF-FPT2 under GRF-FPT2, with qualitatively similar results shown in Figure 16 for GRF-FPT1 and GRF-grad in Appendix F.
Figure 4 shows clearly the geographically-dependent relationship between different housing features and housing prices. In major urban centers such as LA, San Francisco, and Sacramento, housing prices tend to decrease with an increasing number of households, and may reflect overcrowding in densely populated areas. In contrast, rural regions show the opposite trend: prices rise slightly when rural areas have a larger number of housing units. This suggests that, in sparsely populated rural areas, a modest increase in households makes these places more attractive and livable. Median income, however, consistently shows a positive effect on prices across nearly all of California, while population size tends to show a negative effect, highlighting broader state-wide pressures on housing affordability.
this section cite: []

Section: Conclusion
Our results demonstrate that the FPT algorithm offers a substantial computational advantage over GRF-grad with comparable statistical accuracy, and highlights GRF-FPT as a powerful method for multi-dimensional estimation, particularly when estimates of the target function must be learned from the data rather than observed directly. Future work may explore extensions to larger-scale problems and alternative estimation tasks, as in unsupervised learning and structured prediction. Our findings position GRF-FPT as a scalable and robust alternative for practitioners seeking efficient localized estimation.
this section cite: []

Section: References
Ref_id:b0 Title: Shape quantization and recognition with randomized trees Year: (1997)
Ref_id:b1 Title: Mostly harmless econometrics: An empiricist's companion Year: (2009)
Ref_id:b2 Title: Recursive partitioning for heterogeneous causal effects Year: (2016)
Ref_id:b3 Title: Generalized random forests Year: (2019)
Ref_id:b4 Title: Designing and deploying online field experiments Year: (2014)
Ref_id:b5 Title: Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales Year: (1922)
Ref_id:b6 Title: Program evaluation and causal inference with high-dimensional data Year: (2017)
Ref_id:b7 Title: Analysis of a random forests model Year: (2012)
Ref_id:b8 Title: Consistency of random forests and other averaging classifiers Year: (2008)
Ref_id:b9 Title: Bagging predictors Year: (1996)
Ref_id:b10 Title: Random forests Year: (2001)
Ref_id:b11 Title: Classification and Regression Trees Year: (1984)
Ref_id:b12 Title: Distributional random forests: Heterogeneity adjustment and multivariate distributional regression Year: (2022)
Ref_id:b13 Title: Double/debiased machine learning for treatment and structural parameters Year: (2018)
Ref_id:b14 Title: Estimating heterogeneous treatment effects with rightcensored data via causal survival forests Year: (2023-02)
Ref_id:b15 Title: Multivariate regression trees: a new technique for modeling species-environment relationships Year: (2002)
Ref_id:b16 Title: Narrowing the gap: Random forests in theory and in practice Year: (2014-06)
Ref_id:b17 Title: An experimental comparison of three methods for constructing ensembles of decision trees: Bagging, boosting, and randomization Year: (2000)
Ref_id:b18 Title: Local Polynomial Modelling and Its Applications Year: (1996)
Ref_id:b19 Title: Local polynomial kernel regression for generalized linear models and quasilikelihood functions Year: (1995)
Ref_id:b20 Title: Local linear forests Year: (2020)
Ref_id:b21 Title: Greedy function approximation: a gradient boosting machine Year: (2001)
Ref_id:b22 Title: Varying-coefficient models Year: (1993)
Ref_id:b23 Title: Estimating treatment effect heterogeneity in randomized program evaluation Year: (2013)
Ref_id:b24 Title: Learning representations for counterfactual inference Year: (2016)
Ref_id:b25 Title: Sparse spatial autoregressions Year: (1997)
Ref_id:b26 Title: Machine learning estimation of heterogeneous causal effects: Empirical monte carlo evidence Year: (2021)
Ref_id:b27 Title: Online controlled experiments at large scale Year: (2013)
Ref_id:b28 Title: Metalearners for estimating heterogeneous treatment effects using machine learning Year: (2019)
Ref_id:b29 Title: Deep learning in personalization of cardiovascular stents Year: (2020)
Ref_id:b30 Title: A local generalized method of moments estimator Year: (2007)
Ref_id:b31 Title: Sur l'application de la méthode des approximations successives aux équations différentielles ordinaires du premier ordre Year: (1894)
Ref_id:b32 Title: Definitions, methods, and applications in interpretable machine learning Year: (2019)
Ref_id:b33 Title: Kernel estimation of partial means and a general variance estimator Year: (1994)
Ref_id:b34 Title: Sur les applications de la théorie des probabilités aux experiences agricoles: Essai des principes Year: (1923)
Ref_id:b35 Title: Quasi-oracle estimation of heterogeneous treatment effects Year: (2021)
Ref_id:b36 Title: Mémoire sur la théorie des équations aux dérivées partielles et la méthode des approximations successives Year: (1890)
Ref_id:b37 Title: Some methods for heterogeneous treatment effect estimation in high dimensions Year: (2018)
Ref_id:b38 Title: Toward a curse of dimensionality appropriate (coda) asymptotic theory for semiparametric models Year: (1997)
Ref_id:b39 Title: Root-n-consistent semiparametric regression Year: (1988)
Ref_id:b40 Title: Estimating causal effects of treatments in randomized and nonrandomized studies Year: (1974)
Ref_id:b41 Title: A primer on monotone operator methods (survey) Year: (2016)
Ref_id:b42 Title: Consistency of random forests Year: (2015)
Ref_id:b43 Title: Tree-structured methods for longitudinal data Year: (1992)
Ref_id:b44 Title: Quasi-likelihood estimation in semiparametric models Year: (1994)
Ref_id:b45 Title: Estimating individual treatment effect: generalization bounds and algorithms Year: (2017-08)
Ref_id:b46 Title: Kernel smoothing in partial linear models Year: (1988)
Ref_id:b47 Title: The kernel estimate of a regression function in likelihood-based models Year: (1989)
Ref_id:b48 Title: Generalized Random Forests Year: (2024)
Ref_id:b49 Title: Estimation and inference of heterogeneous treatment effects using random forests Year: (2018)
Ref_id:b50 Title: Adaptive concentration of regression trees, with application to random forests Year: (2015)
Ref_id:b51 Title: Confidence intervals for random forests: The jackknife and the infinitesimal jackknife Year: (2014)
Ref_id:b52 Title: Flexible regularized estimating equations: Some new perspectives Year: (2021)
Ref_id:b53 Title: Generalized m-fluctuation tests for parameter instability Year: (2007)
Ref_id:b54 Title: Model-based recursive partitioning Year: (2008)
