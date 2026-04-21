Title: Ridge Boosting is Both Robust and Efficient
Abstract: Estimators in statistics and machine learning must typically trade off between efficiency, having low variance for a fixed target, and distributional robustness, such as multiaccuracy, or having low bias over a range of possible targets. In this paper, we consider a simple estimator, ridge boosting: starting with any initial predictor, perform a single boosting step with (kernel) ridge regression. Surprisingly, we show that ridge boosting simultaneously achieves both efficiency and distributional robustness: for target distribution shifts that lie within an RKHS unit ball, this estimator maintains low bias across all such shifts and has variance at the semiparametric efficiency bound for each target. In addition to bridging otherwise distinct research areas, this result has immediate practical value. Since ridge boosting uses only data from the source distribution, researchers can train a single model to obtain both robust and efficient estimates for multiple target estimands at the same time, eliminating the need to fit separate semiparametric efficient estimators for each target. We assess this approach through simulations and an application estimating the age profile of retirement income.

Section: Introduction
Estimators in statistics and machine learning must typically trade off between efficiency and robustness. Efficient estimators, largely developed in semiparametric statistics and econometrics, focus on having the smallest asymptotic variance (the "efficient variance") among unbiased estimates for a single target estimand. Importantly, such estimators provide valid asymptotically Normal confidence intervals-critical in many empirical applications-and these intervals have the smallest possible width. By contrast, robust estimators, the focus of an active literature in Distributionally Robust Optimization (DRO) and other subfields, instead aim to have good performance for many, possibly unspecified targets. For example, Kim et al. [2022] show that a class of "multi-accurate" estimators-based on boosting an initial predictor-constrains worst-case bias for predicting the unknown mean in a large class of covariate shift problems. In general, we expect that controlling worst-case bias across many estimands would come at the cost of increased variance.
Surprisingly, we show that a simple version of boosting, once-boosting with ridge regression, is simultaneously robust over a large set of possible distribution shifts, while also achieving the efficient variance and smallest possible confidence interval for each estimation target separately. Constructing this ridge boosting predictor is simple: we start with any initial predictor, and then perform one step of boosting using ridge regression in a Reproducing Kernel Hilbert Space (RKHS). For all target populations whose density ratio with respect to the source population is well-approximated by the RKHS, the resulting estimator is approximately unbiased and achieves the semiparametric efficiency bound. This is a very general (but not completely general) class of distribution shifts: it includes any shift whose density ratio can be expressed as linear in a fixed transformation of the covariates, even infinite-dimensional transformations. For example, this includes distribution shifts whose density ratio can be approximated as linear in the last-layer embedding of a pre-trained large language model, but not the more general class that would involve fine-tuning the neural network.
We similarly establish this result for a broad class of linear estimands, generalizing the results from Kim et al. [2022] beyond the missing mean to more complex targets like average derivatives and impulse responses. In this more general setting, we replace the density ratio with the more flexible Riesz representer corresponding to the estimand of interest [Chernozhukov et al., 2021]. Our key technical insight is that kernel ridge regression implicitly estimates the Riesz representer needed for semiparametric estimation: the ridge boosting estimator we analyze is in fact numerically equivalent to the "Automatic Debiased Machine Learning" estimator of Chernozhukov et al. [2021] and inherits its optimality properties. As a result, we can train a single predictor using only source distribution data. Deploying this predictor to estimate any target estimand (whose Riesz representer is in the RKHS) will then have both low bias under distribution shift and asymptotically optimal confidence intervals-without ever explicitly computing target-specific bias correction terms.
Our results have immediate practical implications. In settings where practitioners must estimate many related quantities under different covariate shifts-such as estimating health outcomes across multiple hospitals, or computing age profiles of economic variables-our approach eliminates the need to fit separate semiparametric efficient estimators for each target. As we show in simulations, this approach also yields valid confidence intervals for scalar estimands, an important requirement for many applications.
Paper organization. The paper proceeds as follows. Section 2 formalizes the problem setup, defining our estimation targets and contrasting robustness and efficiency. Section 3 introduces ridge boosting and proves it is both multiaccurate and semiparametrically efficient. Section 4 demonstrates performance through simulations and an empirical application. Section 5 concludes with limitations and future directions.
this section cite: ['b29', 'b29', 'b6', 'b6']

Section: Related literature
Multiaccuracy and Multicalibration: Multi-calibration, introduced by Hébert-Johnson et al. [2018], is a refinement of group calibration that requires a predictor to be simultaneously calibrated across a rich collection of (potentially overlapping) subpopulations. For a prediction task, calibration requires that among the individuals which receive prediction f (x) = v, the true expectation is v. Variants of the original definition have been studied by a number of works [Kim et al., 2019, 2022, Deng et al., 2023, Jung et al., 2021, Gopalan et al., 2022]). Multiaccuracy [Kim et al., 2019] is a weaker version of multi-calibration: it weakens multi-calibration by removing conditioning on the predicted values. Both concepts strengthen classical group fairness by ensuring fine-grained predictive accuracy without sacrificing overall performance. The multiaccuracy criterion is a special case of DRO [Hastings et al., 2024]. The link between multicalibration and boosting is discussed extensively in Globus-Harris et al. [2023]. Long et al. [2025] consider boosting over an RKHS to achieve multiaccuracy, but for classification. It would be interesting to see whether or not we could extend our result to their setting.
Semiparametric efficiency and doubly robustness: Semiparametric efficiency theory provides a rigorous foundation for the efficient estimation of target parameters in models that incorporate both parametric and nonparametric components [Bickel et al., 1993, Newey, 1994]. In the context of causal inference, doubly robust estimators [Robins et al., 1994, Kennedy, 2024] form a central class of methods that can attain semiparametric efficiency under correct specification of both. One motivation for these estimators comes from orthogonal (or Neyman-orthogonal) estimating equations [Chernozhukov et al., 2018, 2021, Foster and Syrgkanis, 2023], which reduce sensitivity to errors in nuisance function estimation. A complementary line of work [Zubizarreta, 2015, Ben-Michael et al., 2021, Athey et al., 2018, Hirshberg and Wager, 2021, Bruns-Smith et al., 2025a] focuses on balancing weights, which aim to reweight samples so that covariate distributions are matched across treatment groups. When balancing weights are combined with outcome regression, the resulting augmented estimators inherit both double robustness and semiparametric efficiency. In parallel, targeted maximum likelihood estimation (TMLE) [Van Der Laan andRubin, 2006, Van der Laan et al., 2011] shows that, by incorporating a targeting step based on the efficient influence function of the parameter of interest and grounded in likelihood theory, TMLE achieves semiparametric efficiency. Cho et al. [2024] consider the TMLE update in an RKHS, and find a closely related universal adaptability property. In future work, it may be possible to unify their results with our boosting and multicalibration setting.
this section cite: ['b19', 'b28', 'b10', 'b23', 'b15', 'b28', 'b17', 'b31', 'b2', 'b32', 'b34', 'b26', 'b5', 'b13', 'b43', 'b1', 'b0', 'b20', 'b9']

Section: Connection between multicalibration and causal inference:
There are several recent papers discussing the connection between multicalibration and causal inference. Wu et al. [2024] show the connection between invariant risk minimization and multicalibration in the context of concept shift. Ye and Li [2024] explores multicalibration and universal adaptability in survival analysis. Kern et al. [2024] shows that the multi-accurate conditional average treatment effect estimate is robust to unknown covariate shifts. Van Der Laan et al. [2023] also calibrate a baseline model to achieve semiparametric efficiency, albeit without using multicalibration.
2 Problem setup: Robustness vs efficiency
this section cite: ['b41', 'b42', 'b27', 'b37']

Section: Notation
Let X ∈ X denote covariates and Y ∈ Y ⊆ R an outcome of interest. We consider a source distribution P over (X, Y ). We assume that we have n p independent and identically distributed observations from P , denoted by {(X i , Y i )} np i=1 ∼ i.i.d. P . We let X p ∈ R np×d denote the matrix of observed covariates and Y p ∈ R np the corresponding vector of outcomes. Define γ 0 (x) := E P [Y |X = x], the optimal mean-squared error predictor of Y given X in P .
this section cite: []

Section: Defining our estimation target
We consider the goal of estimating a scalar summary of the optimal predictor γ 0 [see Chernozhukov et al., 2018]. Examples include estimating a missing mean under covariate shift, estimating an average treatment effect, and estimating an average derivative. This setup generalizes the estimands considered in Kim et al. [2022].
Definition 1 (Target Estimand). For any function f : X → Y, define:
θ target (f ) := E P [m(f, X)],
where m is some real-valued function of f and X such that θ target is linear in f . Our target estimand is:
θ 0 := θ target (γ 0 ).
this section cite: ['b5', 'b29']

Section: Assumption 1 (Continuity).
We assume that θ target is a continuous linear functional. That is, there exists a constant C > 0 such that:
θ(f ) 2 ≤ CE P [f (X) 2 ], for all f with E[f (X) 2 ] < ∞.
We now make this concrete with some examples.
Example 1 (Missing Mean Under Covariate Shift): We begin with an example that will be familiar to machine learning practitioners. Let Q be another distribution on (X, Y ). We assume that we observe samples of X drawn from Q, but that Y is unobserved.
Our goal is to estimate the missing mean E Q [Y ]. For example, if we collect health outcomes (Y ) in New York City (P ), our goal might be to use that data to infer average health outcomes in another city like Raleigh (Q). The issue is that New York and Raleigh are very different cities. But under the covariate shift assumption that E P [Y |X] = E Q [Y |X], the missing mean can be written as:
E Q [Y ] = E Q [γ 0 (X)] = E P dQ dP (X)γ 0 (X) =: θ target (γ 0 ),
exactly as in Definition 1. In this example, Assumption 3 holds if and only if Q is absolutely continuous with respect to P and
E P dQ dP (X) 2 < ∞.
The analogous example in causal inference is estimating the counterfactual potential outcome for treated units when targeting the Average Treatment Effect on the Treated: P are the control units, Q are the treated units, and Y (0) replaces Y . See Johansson et al. [2022] for discussion.
Example 2 (Average Derivative): We now consider an example common in applied economics. Let X 1 denote the first component of X. Then define the average derivative as:
θ target (γ 0 ) = E P ∂γ 0 (X) ∂X 1 .
This θ target is also a linear functional. For example, if Y is household spending, X 1 is household income, and the remainder of X contains other household characteristics, then θ target (γ 0 ) measures the average spending response to a change in income, known as the "Marginal Propensity to Consume."
A central object in what follows will be the Riesz representer corresponding to the estimand θ target : Definition 2 (Riesz representer). Every continuous linear functional θ has a corresponding Riesz representer, a unique function α θ (x) such that:
θ(f ) = E P [α θ (X)f (X)],
for all f such that E P [f (X) 2 ] < ∞. We will write α target (x) to denote the Riesz representer of θ target .
Example (Density Ratio): When θ target (f ) = E Q [f (X)], then the Riesz representer is the density ratio, α target (x) = dQ/dP (x). This has a known analytic form: α target (x) = e(x)/(1 -e(x)) where e(x) is the propensity score or domain classifier for Q vs. P .
Note that our setup focuses on scalar summaries of the optimal predictor, and does not, for example, consider finding a predictor that achieves small mean squared error uniformly over a target distribution Q. Recent work in Kern et al. [2024] suggests that we could extend our results to hold uniformly over X . We leave such an extension to future work.
this section cite: ['b22', 'b27']

Section: Plug-in estimation and regularization bias
Before turning to robustness and efficiency, we introduce a natural starting place, the plug-in estimator. This first fits γ(X) by predicting Y from X using samples from population P and then computes:
θtarget (γ) := 1 n p np i=1 m(γ, X i ).(1)
In the special case of the missing mean (Example 1), we fit our predictor under P , but apply it to covariates drawn from Q. That is, say that we observe n q iid samples of X from Q. The plug-in estimate is then:foot_0 θtarget (γ) := 1 nq nq j=1 γ(X j ). The core difficulty with the plug-in estimator is regularization bias. When fitting γ via machine learning, standard methods regularize the predictor to generalize better out-of-sample. Unfortunately, γ might regularize away parts of the sample-space that are important for θ target . Say there is a particular combination of X that is very common in Q, but relatively rare in P . Then a cross-validated predictor trained under P might regularize away the predictions on those values of X to reduce variance. While optimal for prediction under P , this would lead to meaningful bias for E Q [Y ], which in turn could lead to a very poor estimate of the target estimand. Furthermore, bias means that the estimate will not be asymptotically normal, meaning that standard confidence intervals will not be valid -often an important desideratum in applied work.
this section cite: []

Section: Robustness: Constraining worst-case bias across many unknown targets
A large literature in robust optimization and algorithmic fairness focuses on constructing estimators that modify γ above such that the plug-in estimate θ(γ) has small bias for a range of target quantities [Kim et al., 2022]. For example, say we observe data from a single source hospital P , but we want to estimate E Q [Y ] across many different target hospitals Q 1 , Q 2 , . . . Q K , where collecting unlabeled data and estimating density ratios α θ for each target site would be costly or possibly infeasible due to privacy concerns. Can we still estimate a γ from P that is robust to many unknown distribution shifts? Kim et al. [2022] show that the answer is yes-although as we discuss below, we might be concerned about this procedure inflating the variance.
Consider a target estimand θ target satisfying Definition 1 and Assumption 3, but now assume that we do not have access to θ target ahead of time. Our goal is to use the observations in P to construct a predictor γ such that the resulting plug-in estimator for θ target is approximately unbiased. In other words, we want to control the worst-case bias over a large set of possible estimands. In the fair machine learning literature, this condition is known as multiaccuracy: Definition 3 (Multiaccuracy). Given an "auditing" function class C and source population P , a predictor f is (C, a)-multiaccurate if for every function c(X) ∈ C, sup Kim et al. [2019] show that an initial predictor γinit can be modified to be multiaccurate by running a simple boosting procedure, including a version of our main proposal of boosting with ridge regression.
c∈C |E P [(Y -f (X)) • c(X)]| ≤ a.
We now generalize the result in Kim et al. [2022], which considered the specific case of estimating a missing mean under covariate shift, to the more general class defined in Definition 1, where we replace the density ratio with the Riesz representer. Even if we do not know θ target in advance, if we can construct a set Θ such that we believe θ target ∈ Θ, then we can still obtain an unbiased estimator of θ target if we can construct a multiaccurate predictor γma . Proposition 1. Let Θ be some set of functionals θ such that Definition 1 and Assumption 3 hold. Let A be the corresponding set of Riesz representers:
A := {α : ∃θ ∈ Θ s.t. θ(f ) = E[f (X)α(X)], ∀f with E[f (X) 2 ] < ∞}. Then if γma is (A, a)-multiaccurate: |θ(γ ma ) -θ(γ 0 )| ≤ a, ∀θ ∈ Θ. Remark 1 (Distributionally-Robust Optimization). Let Θ contain θ(f ) = E Q k [f (X)]
for many distributions Q k , where A contains the corresponding density ratios, dQ k /dP . In this case, Definition 3 is a special case of the more general literature on Distributionally-Robust Optimization [Hastings et al., 2024]. For a target estimand,
θ target (f ) = E Qtarget [f (X)], if dQ target /dP ∈ A, then θ target (γ ma ) is approximately unbiased for θ target (γ 0 ).
However, we might be concerned about the cost of robustness in terms of additional variance. Since we enforce small bias over a potentially large class of target estimands θ ∈ Θ, we would therefore expect larger variance for our specific target estimand θ target .
this section cite: ['b29', 'b29', 'b28', 'b29', 'b17']

Section: Efficiency: Unbiased estimate with the smallest variance for a single, known target
In many applications, we know our target functional θ target in advance, such as if we observe samples of X from the distribution Q at training time. In this case, one popular strategy is to "bias correct" the initial estimate θtarget (γ), a problem studied extensively in the semiparametric statistics literature [Chernozhukov et al., 2024]. Importantly, the resulting estimator has the smallest possible variance among all unbiased estimators [Chernozhukov et al., 2018].
Following the general setup in Chernozhukov et al. [2021], we focus on bias correction using the Riesz representer of θ target . Under minimal conditions, if γ is a consistent estimator of E P [Y |X], and α is a consistent estimator of α target (X), then the estimator,
θefficient := θtarget (γ) + 1 n p np i=1 α(X i )(Y i -γ(X i ))
bias correction term , has the following three properties, asymptotically: (1) it is unbiased, i.e., E P [ θefficient ] = θ target (γ 0 );
(2) it is normally distributed; and (3) it has the smallest variance of all regular asymptotically-linear estimators. This third property is called semiparametric efficiency, and the corresponding variance is called the semiparametric efficiency bound, denoted V * θ for θ(γ 0 ). Formally, we have:
√ n θefficient -θ target (γ 0 ) → N (0, V * θtarget ), and V → p V * θtarget , where V is the sample variance, V := 1 n p np i=1 m(γ, X i ) + α(X i )(Y i -γ(X i )) -θefficient 2 .
See e.g. Chernozhukov et al. [2023] for a set of minimal conditions under which this result holds.
this section cite: ['b8', 'b5', 'b6', 'b7']

Section: Ridge boosting simultaneously achieves robustness and efficiency
Thus far, we have explored two classes of estimators: robust estimators that have low bias over many estimands, and bias-corrected estimators that are unbiased and efficient for a specific target estimand.
In this section, we demonstrate that it is possible to construct an estimator that is both robust and efficient. Specifically, when the set of target Riesz representers A is the norm ball in an RKHS, we can construct a multiaccurate predictor that has small worst-case bias over the corresponding Θ while simultaneously achieving the semiparametric efficient variance for every target θ ∈ Θ. We refer to the resulting procedure as once-boosting with ridge regression or, more simply, ridge boosting.
While our most general theoretical results only hold when A is a norm-ball in an RKHS, in the Appendix we sketch out a version of our result for boosting with Random Forests. Whether there exists a more general result is a exciting topic for future work.
this section cite: []

Section: Ridge boosting
In this section, we introduce the main estimator we analyze, once-boosting with ridge regression.
Setup. An RKHS H is a set of functions h : X → R defined by an inner product. In the most general case, for all x ∈ X there exists ϕ(x) ∈ H such that for any h ∈ H, h(x) = ⟨h, ϕ(x)⟩. One special case is the finite dimensional Hilbert space where ϕ(x) is some feature map from X → R d and H = {h(x) = β ⊤ ϕ(x) : β ∈ R d }; our results, however also hold for infinite-dimensional RKHS's. For any h ∈ H, we define the norm ∥h∥ 2 H = ⟨h, h⟩. For some RKHS H, we consider the following function class:
A = {h ∈ H : ∥h∥ H ≤ B},
for some B > 0. We set B = 1 (which will be without loss of generality), so that A forms a unit ball.
Recall that in our robustness setup, A corresponds to the set of Riesz representers for all θ ∈ Θ. In the covariate shift setting, A is the set of density ratios. Restricting our attention to Riesz representers that belong to such an A is very general, but not fully general. Such a set can include highly non-linear functions of x, but only functions that can be written in terms of the fixed basis ϕ(x). For example, A could be a set of functions that are linear in the last-layer embedding of a pre-trained large language model (LLM). But A could not include all functions achievable by fine-tuning that pre-trained LLM.
Estimator. We now introduce once-boosting with ridge regression. For notational simplicity, we present the algorithm in the case where H is a finite-dimensional RKHS with ϕ(x) ∈ R d -the arguments are identical in the infinite-dimensional case. We will write Φ p ∈ R np×d for the matrix with rows ϕ(x i ) for each observation i. Assume that we have an initial estimator of E P [Y |X], γinit (X), which could have been fit with some arbitrary machine learning algorithm. We then perform a ridge boosting step on the residuals Y p -γinit (X p ): min
β∈R d ∥Y p -γinit (X p ) -Φ p β∥ 2 2 + λ∥β∥ 2 2 .
Call the solution βboost and define γboost (x) := ϕ(x) ⊤ βboost . Then define:
γma (x) = γinit (x) + γboost (x).
(2)
this section cite: []

Section: Ridge boosting is multiaccurate
Next we will show that γma is indeed multiaccurate. In other words, θ(γ ma ) is an approxiamtely unbiased estimate of θ(γ 0 ), for all θ ∈ Θ. We first define the notion of the multiaccuracy error.
Definition 4 (Multi-accuracy error). For a given auditing function class C and a source population P , the multiaccuracy error of an estimator f (X) and its sample analog are defined as:
MAE C ( f ) = sup c∈C |E P [c(X) • (Y -f (X))]|, MAE C ( f ) = sup c∈C |c(X p ) ⊤ (Y p -f (X p ))|.
Then we have the following guarantees:
Theorem 1. For γma defined in (2), we have:
MAE A (γ ma ) ≤ max 1≤j≤d λ λ + σ 2 j MAE A (γ init ),
where σ 2 j are the eigenvalues of Φ ⊤ p Φ p . Under standard regularity conditions, with probability 1 -η,
MAE A (γ ma ) ≤ O δ n + 1/η n p ,
for δ n such that δ n → 0 as n → ∞.
We provide a proof and additional discussion in the Appendix. The first result shows that one step of ridge boosting is guaranteed to decrease the sample multiaccuracy error. The second result shows we can generalize out of sample. The rate of convergence of δ n depends on the dimensionality and smoothness of H. When H is finite-dimensional with dimension d, δ n ≤ d/n. Remark 2. We emphasize that Theorem 1 is not a fundamentally new result. The multiaccuracy literature already proves generalization bounds on the multiaccuracy error for boosting estimators using ridge regression; see Kim et al. [2019]. However, our boosting procedure differs slightly in its specifics (e.g. linear boosting instead of exponential weighting) and so we provide Theorem 1 for completeness. Our proof uses standard techniques from the analysis of kernel ridge regression.
this section cite: ['b28']

Section: Ridge boosting is semiparametrically efficient
We showed above that boosting with ridge regression produces a predictor that is multiaccurate with respect to A. We now show that this multiaccurate estimator is also semiparametrically efficient for all θ ∈ Θ. Specifically, we will show that ridge boosting implicitly estimates the Riesz representer and performs semiparametric bias correction. In fact, the resulting estimator θ(γ ma ) is numerically equivalent to a special case of Automatic Debiased Machine Learning using kernel Riesz regression Chernozhukov et al. [2021], Singh [2024].
this section cite: ['b6', 'b36']

Section: Ridge regression implicitly estimates Riesz representers
The key fact that will lead to our main result is that ridge regression implicitly estimates Riesz representers. To see this, notice that for any continuous linear functional θ, the Riesz representer α θ (X) is the unique solution to the following loss minimization problem
α θ = argmin α:E[α(X) 2 ]<∞ {E P [α(X) 2 ] -2θ(α)}.(3)
See Chernozhukov et al. [2021] for more discussion. One way to approximate α θ (X) is to minimize (3) over an RKHS H. The sample version of this optimization problem is:
min η∈R d 1 n p η ⊤ Φ ⊤ p Φ p η -2θ(Φ p ) ⊤ η + λ∥η∥ 2 2 ,(4)
with minimizer ηλ and corresponding Riesz representer estimate αλ θ (x) := ϕ(x) ⊤ ηλ . See Singh [2024] for an analysis of this estimator.
Ridge regression when used to estimate θ can always be rewritten as a weighting estimator with weights αλ θ (X p ), as we show in the following proposition. Proposition 2. Let θ be any continuous linear functional, and define the Riesz representer estimate αλ θ from (4). Let Z p be any function of X p and Y p . Consider the ridge regression in H that predicts Z p given X p : min
β∈R d ∥Z p -Φ p β∥ 2 2 + λ∥β∥ 2 2 .
Call the solution βridge and the corresponding predictor γridge (x) = ϕ(x) ⊤ βridge . Then:
θ(γ ridge ) = 1 n p αλ θ (X p ) ⊤ Z p .
This is a well-known result; see Kallus [2020], and see Bruns-Smith et al. [2025a] for an extensive discussion of the implications for semiparametric estimation. Remark 3. Ridge regression can be written as a linear smoother. The result above says that when we compute θ(γ ridge ), the smoother weights estimate the Riesz representer. Random forests can also be written as a linear smoother, and Lin and Han [2022] show that the weights converge to the Riesz representer in a similar sense. We use this connection to show that the same robustness/efficiency properties apply to boosting with Random Forests in Appendix B.
this section cite: ['b6', 'b36', 'b24', 'b30']

Section: Main result: Semiparametric efficiency for all θ ∈ Θ
We now apply Section 3.3.1 to our estimator γma to establish our main result: θ(γ ma ) does not just have small worst-case bias over all θ ∈ Θ, it is semiparametrically efficient for each individual θ ∈ Θ.
For any θ ∈ Θ, we have the following:
θ(γ ma ) = θ(γ init ) + θ(γ boost ) = θ(γ init ) + 1 n p αλ θ (X p ) ⊤ (Y p -γinit (X p )),(5)
where the second equality follows from applying Proposition 2 for Z p = Y p -γinit (X p ).
Note that (6) has exactly the form of θefficient from Section 2.5. In fact, this estimation strategy -in which we fit an arbitrary machine learning estimator γinit and use a αθ (X) that minimizes (3) for bias correction -is a well-studied estimator from the semiparametric statistics literature [Athey et al., 2018, Hirshberg and Wager, 2021, Chernozhukov et al., 2021, Bruns-Smith et al., 2025a].
The particular form of αλ θ (X) used here, which is obtained by minimizing (4) in an RKHS, is specifically considered in Hirshberg et al. [2019], Kallus [2020], Hazlett [2020], Singh [2024]. Thus, once-boosting with ridge provides a multiaccurate predictor, but the resulting point estimate θ(γ ma ) is numerically-equivalent to well-studied semiparametrically efficient estimators. We leverage this connection to establish our main theoretical result. Theorem 2 (Informal). Given standard regularity assumptions and some conditions on the quality of γinit , then for all θ ∈ Θ, the ridge boosting plug-in estimator is asymptotically Normal and its variance achieves the asymptotic variance lower bound
V * θ : √ n θ(γ ma ) -θ(γ 0 ) → N (0, V * θ ), and V → p V * θ .
where V is the sample variance,
V := 1 n p np i=1 m(γ ma , X i ) -θ(γ ma ) 2 .
See the Appendix for a formal Theorem statement and proof. This result establishes that for any estimand θ whose Riesz representer belongs to the RKHS ball A, the ridge boosting estimator θ(γ ma ) is not just robust, it is semiparametrically efficient.
This estimator is also computationally convenient for practitioners interested in efficient inference. We simply take the initial predictor and run one step of boosting with kernel ridge regression, which has many readily-available implementations. And since a plug-in estimator using this new predictor is semiparametrically efficient for any estimand in Θ, we do not have to (explicitly) fit individual Riesz representer estimates αθ for each θ, which can be expensive when there are many θs of interest. We also do not require specialized code for minimizing the Riesz loss (3), which can be a barrier for practitioners with less familiarity with Riesz representers.
this section cite: ['b0', 'b20', 'b6', 'b21', 'b24', 'b18', 'b36']

Section: Experiments

this section cite: []

Section: Simulation study
We now demonstrate in simulation that we simultaneously achieve robust and efficient inference by deploying our multiaccurate predictor on different estimands in different environments. To demonstrate the generality of the framework, we consider estimating an average derivative with correlated covariates -on its own, already a difficult task -under distribution shift. We will fit both ridge and once-boosted ridge models in a training sample, and then compute the usual 95% asymptotic Normal confidence interval for the average derivative across three test distributions, assessing empirical coverage over simulation draws.
Simulation Setup: We consider three-dimensional correlated covariates: X 1 , X 2 ∼ N (µ, 1), and
X 3 = 4 • σ(X 1 -X 2 ) + ϵ -2,
where σ(•) is the sigmoid, and ϵ ∼ N (0, 2 2 ). For the training distribution, P , µ = 0; for the test distributions, we vary µ. The outcomes are generated as:
Y = Y ∼ f (X) + η, f (X) = X 1 • (0.2 + sin(X 1 ) + σ(X 2 ) -0.2 • X 3 ), η ∼ N (0, 2 2 ).
The estimand is the average derivative of f (X) with respect to
X 1 under Q, E Q [∂f (X)/∂X 1 ].
We consider three different distributions for Q, with the same setup but with µ ∈ {-1, 0, 1}. The dependence of X 3 on both X 1 and X 2 makes the average derivative more challenging to estimate.
this section cite: []

Section: Methods:
We compare two estimators. (1) Naive Kernel Ridge: a standard kernel ridge outcome regression trained on the source data and plugged in for each target estimand. (2) Boosted Kernel Ridge: a one-step kernel ridge boosting procedure applied to the residuals of the initial kernel ridge regression. Both models are trained solely on the training (source) data and evaluated on each of the three test (target) distributions.
Monte Carlo Simulation: For each simulation, we draw a training sample of X and Y from the source distribution, and fit the kernel ridge and boosted kernel ridge model. We then draw one sample of X from each of the three test distributions, and estimate the average derivative with respect to X 1 on that test distribution by symmetric differencing, along with the usual 95% asymptotic normal confidence interval. The whole process is repeated for sample sizes ranging from 50 to 500 and with 1,000 Monte Carlo replications. We report the empirical coverage of the confidence intervals for both methods across sample sizes and replications. The full simulation study is run on a four-core laptop.
Results: The results are shown in Figure 1. Across all three test distributions, the naive confidence intervals using the base ridge model under cover. By contrast, the confidence intervals from onceboosted ridge regression achieve good empirical coverage even with a moderate sample size (n = 300). The same pattern holds across all three randomly generated covariate shifts, showing the boosted ridge regression can achieve robustness toward covariate shifts and statistical efficiency at the same time. This reveals a new practical benefit of the multiaccurate estimator in this setting previously unexplored in the multicalibration literature: we achieve valid uncertainty quantification under distribution shift.
this section cite: []

Section: Empirical application to retirement income
We now consider an empirical economics application: estimating the age profile of income throughout retirement.
For an individual i, let Y i be total income (including retirement income), let A i be age, and let X i be other covariates like education, race, and marital status. Define γ 0 (A, X) := E[Y |A, X]. Our estimand is the age profile of income (for ages 65, . . . , 89), which is defined as the following counterfactual means:
θ 65 := E[γ 0 (65, X)], θ 66 := E[γ 0 (66, X)], ... , θ 89 := E[γ 0 (89, X)].
That is, for each θ a , we want the average value of γ 0 (A, X), but where we have counterfactually replaced the age of every individual with the fixed value a. This is a key input into structural models of the macroeconomy -see, for example, Gourinchas and Parker [2002], Kaplan and Violante [2014]. Because the distribution of the covariates X varies with A, this can induce very significant distribution shift, especially late in retirement. We emphasize that this is a highly simplified example inspired by Bruns-Smith et al. [2025b], although applying our methodology to their setting is a promising direction for future work.
State of the art modeling here would construct separate semiparametric efficient estimators for each point in the age profile, necessarily requiring a separate debiasing term at every age. In this application, we instead use a single multiaccurate predictor to estimate the 25 different target estimands-one for each year of retirement age-demonstrating the utility of our procedure in practice. We estimate the age profile of retirement using data from 2018 American Community Survey as processed by the FolkTables package [Ding et al., 2021]. As in Section 4.1 we fit a single kernel ridge and boosted kernel ridge model, and then compute plug-in estimates of θ a using these two models. The results are displayed in Figure 2. Whereas the naive estimate (without boosting) features a steep decline of $11k from ages 65-89, our boosted estimates are substantially flatter -better matching the theoretical model for pension and social security income from Kaplan and Violante [2014]. Furthermore, while the naive confidence intervals shrink for the highest ages, our boosted confidence intervals actually grow slightly, suggesting that the naive model may undercover for the oldest part of the age profile.
this section cite: ['b16', 'b25', 'b11', 'b25']

Section: Conclusion
In this manuscript, we investigate the connection between multiaccuracy and semiparametric efficiency. Specifically, we show that boosting an initial predictor with kernel ridge produces an estimator that is not only multiaccurate over estimands in an RKHS norm ball but is also semiparametrically efficient for each target separately. This result can be understood through the lens of Riesz regression: ridge boosting implicitly performs Riesz regression, thereby yielding an augmented balancing weight estimator that attains the semiparametric efficiency bound. We demonstrate one practical benefit: valid confidence intervals across distributions under covariate-shifts not seen at training time. However, our results are limited to shifts described by an RKHS. And making this proposal fully practical requires additional investigation of appropriate hyperparameter tuning. We hope this initial work leads to further exchange between the robustness and semiparametrics literatures.
this section cite: []

Section: References
Ref_id:b0 Title: Approximate residual balancing: debiased inference of average treatment effects in high dimensions Year: (2018)
Ref_id:b1 Title: The balancing act in causal inference Year: (2021)
Ref_id:b2 Title: Efficient and adaptive estimation for semiparametric models Year: (1993)
Ref_id:b3 Title: Augmented balancing weights as linear regression Year: ()
Ref_id:b4 Title: Disentangling age, time, and cohort effects in income inequality: A proxy machine learning approach Year: (2025-10)
Ref_id:b5 Title: Double/debiased machine learning for treatment and structural parameters Year: (2018)
Ref_id:b6 Title: Automatic debiased machine learning via riesz regression Year: (2021)
Ref_id:b7 Title: A simple and general debiased machine learning theorem with finite-sample guarantees Year: (2023)
Ref_id:b8 Title: Applied causal inference powered by ml and ai Year: (2024)
Ref_id:b9 Title: Kernel debiased plug-in estimation: Simultaneous, automated debiasing without influence functions for many target parameters Year: (2024)
Ref_id:b10 Title: Happymap: A generalized multi-calibration method Year: (2023)
Ref_id:b11 Title: Retiring adult: New datasets for fair machine learning Year: (2021)
Ref_id:b12 Title: Sobolev norm learning rates for regularized least-squares algorithms Year: (2020)
Ref_id:b13 Title: Orthogonal statistical learning Year: (2023)
Ref_id:b14 Title: Multicalibration as boosting for regression Year: (2023)
Ref_id:b15 Title: Low-degree multicalibration Year: (2022)
Ref_id:b16 Title: Consumption over the life cycle Year: (2002)
Ref_id:b17 Title: Taking a moment for distributional robustness Year: (2024)
Ref_id:b18 Title: Kernel balancing Year: (2020)
Ref_id:b19 Title: Multicalibration: Calibration for the (computationally-identifiable) masses Year: (2018)
Ref_id:b20 Title: Augmented minimax linear estimation Year: (2021)
Ref_id:b21 Title: Minimax linear estimation of the retargeted mean Year: (2019)
Ref_id:b22 Title: Generalization bounds and representation learning for estimation of potential outcomes and causal effects Year: (2022)
Ref_id:b23 Title: Moment multicalibration for uncertainty estimation Year: (2021)
Ref_id:b24 Title: Generalized optimal matching methods for causal inference Year: (2020)
Ref_id:b25 Title: A model of the consumption response to fiscal stimulus payments Year: (2014)
Ref_id:b26 Title: Semiparametric doubly robust targeted double machine learning: a review. Handbook of Statistical Methods for Precision Medicine Year: (2024)
Ref_id:b27 Title: Multi-cate: Multi-accurate conditional average treatment effect estimation robust to unknown covariate shifts Year: (2024)
Ref_id:b28 Title: Multiaccuracy: Black-box post-processing for fairness in classification Year: (2019)
Ref_id:b29 Title: Universal adaptability: Targetindependent inference that competes with propensity scoring Year: (2022)
Ref_id:b30 Title: On regression-adjusted imputation estimators of the average treatment effect Year: (2022)
Ref_id:b31 Title: Kernel multiaccuracy Year: (2025)
Ref_id:b32 Title: The asymptotic variance of semiparametric estimators Year: (1994)
Ref_id:b33 Title: Score-preserving targeted maximum likelihood estimation Year: (2025)
Ref_id:b34 Title: Estimation of regression coefficients when some regressors are not always observed Year: (1994)
Ref_id:b35 Title: Highly adaptive ridge Year: (2024)
Ref_id:b36 Title: Kernel ridge riesz representers: Generalization, mis-specification, and the counterfactual effective dimension Year: (2024)
Ref_id:b37 Title: Causal isotonic calibration for heterogeneous treatment effects Year: (2023)
Ref_id:b38 Title: Targeted maximum likelihood learning Year: (2006)
Ref_id:b39 Title: Targeted learning: causal inference for observational and experimental data Year: (2011)
Ref_id:b40 Title: High-dimensional statistics: A non-asymptotic viewpoint Year: (2019)
Ref_id:b41 Title: Bridging multicalibration and out-of-distribution generalization beyond covariate shift Year: (2024)
Ref_id:b42 Title: Multicalibration for censored survival data: Towards universal adaptability in predictive modeling Year: (2024)
Ref_id:b43 Title: Stable weights that balance covariates for estimation with incomplete outcome data Year: (2015)
