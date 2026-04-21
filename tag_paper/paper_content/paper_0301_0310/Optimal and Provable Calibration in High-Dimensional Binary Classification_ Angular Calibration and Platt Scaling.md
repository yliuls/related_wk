Title: Optimal and Provable Calibration in High-Dimensional Binary Classification: Angular Calibration and Platt Scaling
Abstract: We study the fundamental problem of calibrating a linear binary classifier of the form σ( ŵ⊤ x), where the feature vector x is Gaussian, σ is a link function, and ŵ is an estimator of the true linear weight w ⋆ . By interpolating with a noninformative chance classifier, we construct a well-calibrated predictor whose interpolation weight depends on the angle ∠( ŵ, w ⋆ ) between the estimator ŵ and the true linear weight w ⋆ . We establish that this angular calibration approach is provably well-calibrated in a high-dimensional regime where the number of samples and features both diverge, at a comparable rate. The angle ∠( ŵ, w ⋆ ) can be consistently estimated. Furthermore, the resulting predictor is uniquely Bregman-optimal, minimizing the Bregman divergence to the true label distribution within a suitable class of calibrated predictors. Our work is the first to provide a calibration strategy that satisfies both calibration and optimality properties provably in high dimensions. Additionally, we identify conditions under which a classical Platt-scaling predictor converges to our Bregman-optimal calibrated solution. Thus, Platt-scaling also inherits these desirable properties provably in high dimensions.

Section: Introduction
Calibration of predictive models is a fundamental problem in statistics and machine learning, especially in applications that require reliable uncertainty quantification. A well-calibrated model ensures that its predicted probabilities align closely with true event probabilities-a property essential in fields such as medical decision-making [7,41], meteorological forecasting [14,28,23,65,64], self-driving systems [60], and natural language processing [66,29].
Numerous algorithms have been proposed for calibrating the outputs of a trained model, including classical methods such as Platt scaling [70,13,69,31], histogram binning [89,77], isotonic regression [90,40,12,36,44], and more recent approaches such as temperature scaling [29,47], ensemble-based methods [50,58,86,81], and Bayesian strategies [45,21], among others.
While extensive prior work has studied calibration [48,77,30,74,43], this literature primarily focuses on traditional asymptotic theories or finite-sample learning theoretic arguments. These approaches often overlook the impact of problem dimensionality, which is particularly relevant for high-dimensional settings where the number of features may be substantial. Alternatively, a separate line of research has explored calibration within a high-dimensional proportional asymptotic regime, where the sample size n and the feature dimension d both diverge, at a comparable rate. This proportional scaling regime has gained significant traction in modern statistics and machine learning. In statistics, its popularity stems from the fact that theories derived under this regime capture high-dimensional phenomena observed in moderate to large sized datasets unusually well [42,6,25,80,91,5,78,39,63,84,52,53]. Consequently, this has spurred the creation of innovative methods displaying remarkable practical performance [61,27,9,54,76,57]. In machine learning, this regime has proven exceptionally valuable and effective in analyzing the behavior of modern neural networks and other interpolation learners under overparametrization [55,34,56,59,2,75,67]. For binary classification in this proportional regime, a substantial line of work [79,78,92] establishes that classical logistic regression yields seriously biased estimates; building upon these, [3] shows that logistic regression tends to be inherently overconfident, while [22] discusses the impact of regularization under the same model. Finally, [21] introduces expectation consistency and derives a limiting calibration error formula as a function of the signal prior and other problem parameters.
Despite these advancements, an approach that is provably calibrated in high dimensions, without knowledge of the true signal prior, is missing. Moreover, there is a lack of principled understanding regarding optimal calibration strategies from among the available options. Additionally, rigorous guarantees on the performance of classical calibration methods, such as Platt scaling, in modern high-dimensional scenarios is notably absent from the literature. In this paper, we address these gaps. We consider the challenge of calibrating a binary linear predictor in a frequentist setting under a Gaussian design. Our contributions are three-fold: (i) we introduce a data-driven predictor that can provably calibrate in a broad class of high-dimensional binary classification problems;
(ii) we show that our calibrated predictor is Bregman-optimal, meaning it uniquely minimizes any Bregman divergence relative to the true label-generation probability; (iii) we establish conditions under which a classical Platt-scaled predictor converges to this Bregman-optimal calibrated solution, thereby formally showing that Platt scaling is both well-calibrated and Bregman optimal in our high-dimensional setting. Although we derive our theoretical results assuming Gaussian features, extensive recent universality results suggest that these should continue to hold for sufficiently light tailed distributions (see Section 8 for a discussion). We provide experiments that demonstrate this robustness to the Gaussian assumption (Section H.2 and 4).
We construct our calibrated predictor by interpolating with an uninformative ("chance") predictor, where the interpolation weight is determined by the angle ∠( ŵ, w ⋆ ) between the estimated linear weight ŵ and the true weight w ⋆ . Our construction crucially leverages recent developments from the literature on observable estimation of unknown parameters in high dimensions. For instance, leveraging advances in [38,8,9,10,18,54], we can show that the angle ∠( ŵ, w ⋆ ) is consistently estimable when n and d grow proportionally. To our knowledge, this is the first provable calibration method in a high-dimensional setting, and it uncovers a conceptual link between optimal calibration and ∠( ŵ, w ⋆ ): the poorer the alignment of w with w ⋆ , the greater the noise needed to be injected to prediction logits to ensure calibration.
this section cite: ['b6', 'b40', 'b13', 'b27', 'b22', 'b64', 'b63', 'b59', 'b65', 'b28', 'b69', 'b12', 'b68', 'b30', 'b88', 'b76', 'b89', 'b39', 'b11', 'b35', 'b43', 'b28', 'b46', 'b49', 'b57', 'b85', 'b80', 'b44', 'b20', 'b47', 'b76', 'b29', 'b73', 'b42', 'b41', 'b5', 'b24', 'b79', 'b90', 'b4', 'b77', 'b38', 'b62', 'b83', 'b51', 'b52', 'b60', 'b26', 'b8', 'b53', 'b75', 'b56', 'b54', 'b33', 'b55', 'b58', 'b1', 'b74', 'b66', 'b78', 'b77', 'b91', 'b2', 'b21', 'b20', 'b37', 'b7', 'b8', 'b9', 'b17', 'b53']

Section: Setting
Suppose we observe i.i.d. data (y i , x i ) satisfying
y i iid ∼ Bern σ w ⊤ ⋆ x i , i = 1, . . . , n,(1)
where σ : R → [0, 1] denotes the link function and the covariates x i ∈ R d are drawn independently as x i iid ∼ N (0, Σ), with Σ assumed to be known (say from a separate unlabeled dataset as in [17,18]). The true linear weight w ⋆ ∈ R d is an arbitrary deterministic vector, and we assume without loss of generality that
w ⊤ ⋆ Σw ⋆ = ∥w ⋆ ∥ 2 Σ = 1. The training dataset is denoted as X = [x 1 , . . . , x n ] ⊤ ∈ R n×d and y = [y 1 , . . . , y n ] ⊤ ∈ R n .
To quantify the degree of miscalibration, we define the calibration error at level p for any predictor f as
∆ cal p ( f ) = p -E xnew σ w ⊤ ⋆ x new | f (x new ) = p ,
where E xnew denotes the expectation over x new ∼ N (0, Σ). A predictor is said to be well-calibrated if ∆ cal p ( f ) = 0 for all p in the range of f . Intuitively, this means that when the predictor assigns a probability p to label 1, the true probability of label 1 is indeed p.
We consider the regularized M-estimator
w = arg min w 1 n n i=1 ℓ yi w ⊤ x i + g(w),
where g(•) is a convex penalty and ℓ(•) is a convex loss function. In this setting, we consider a sequence of problem instances {y(d), X(d), w ⋆ (d)} d≥1 such that X(d) ∈ R n(d)×d and y(d) ∈ R n(d) generated from (1). It is well-known that in the special case where ℓ(•) equals the logistic loss and g(•) is zero, the corresponding predictor σ( ŵ⊤ x new ) is grossly mis-calibrated in the high-dimensional regime n d → (0, +∞) [79,78,3,22], even where it is well-defined and unique [16]. In what follows, we present, for the first time, a predictor that is provably well-calibrated in this regime (for general convex losses and penalties beyond the special case mentioned above). We achieve this through an angular calibration idea, and furthermore, establish that this is optimal in the sense that it minimizes any Bregman divergence to the true label distribution. We conclude showing an interesting connection-Platt scaling converges to our angular predictor-and therefore is both provably well-calibrated and optimal in the aforementioned sense.
this section cite: ['b16', 'b17', 'b0', 'b78', 'b77', 'b2', 'b21', 'b15']

Section: Introducing angular calibration
Most calibration strategies adjust a pre-trained predictor by learning a mapping F : u → F (u) of the logits ŵ⊤ x new . Platt scaling, for example, stipulates the parametric form F (u) = σ(Au + B), where A, B ∈ R are fit on a holdout dataset. This raises a natural question: Among all well-calibrated predictors of the form F ( ŵ⊤ x new ), which one is "the best"?
In this section, we introduce a predictor with such an optimality property by interpolating between the prediction logits ŵ⊤ x and an uninformative chance predictor. Specifically, we show that if the interpolation weight is determined by the angle between the estimator w and the true weight w ⋆ , given by
θ * = arccos ⟨w ⋆ , w⟩ Σ ∥ w∥ Σ ∥w ⋆ ∥ Σ ,(2)
then the resulting interpolated predictor minimizes any Bregman divergence to the true label distribution among all predictors of the form F ( ŵ⊤ x new ). To the best of our knowledge, a predictor that is both provably calibrated and optimal (in the aforementioned sense) has not been previously introduced for high-dimensional problems.
Notably, our predictor uses the angle defined in (2), thus to define a data-driven predictor, we require a consistent estimate of this angle. Fortunately, recent advances in the high-dimensional literature (c.f., [38,8,9,10,18,54]) allow us to estimate the inner product ⟨ w, w ⋆ ⟩ Σ , and therefore θ * , when n and d grow proportionally. We discuss the details of this estimation scheme later in Section 6. For now, we present our angular calibration idea assuming that we have access to a consistent estimator θ for θ * . Definition 3.1.
(Angular Predictor) Let fang w ⊤ x new ; θ = E Z σ cos θ • w ⊤ x new ∥ w∥ Σ + sin θ • Z(3)
where θ is a consistent estimator of θ * defined as in (13) and E Z denotes expectation with respect to the Gaussian noise Z ∼ N (0, 1). We will later refer to fang as the angular predictor for simplicity.
Theorem 3.2 below shows that the angular predictor is well-calibrated. We defer the proof to Section A Theorem 3.2. Assume the link function σ is continuous. Then, the predictor fang defined in (3) is well-calibrated as d, n → ∞, n/d → (0, ∞). That is, for any p contained in the range of σ, we have that
∆ cal p fang •; θ = p -E xnew σ w ⊤ ⋆ x new | fang w ⊤ x new ; θ = p → 0,
in probability where θ is a consistent estimator for θ ⋆ (Cf. Proposition 6.2).
The above utilizes the result that when θ = θ * exactly, fang (•; θ * ) is exactly well-calibrated (we state and prove this formally in Theorem A.1) and that θ is consistent for θ ⋆ . We will later show that fang •; θ ≈ fang (•; θ * ) is in fact optimal in the sense that it minimizes any Bregman divergence
to the true label distribution. The construction (3) admits an intuitive interpretation. By the basic trigonometric identity cos 2 (θ * ) + sin 2 (θ * ) = 1, we can see that the logits, that is, the argument of σ(•) in (3), is an interpolation between the informative component w ⊤ x new and the noninformative Gaussian noise Z. Notice that when w is well aligned with w ⋆ (i.e., cos 2 (θ * ) = 1), the angular predictor fang lies closer to the informative predictor σ( w ⊤ x new ). Conversely, when w ⋆ and w are orthogonal (i.e., sin 2 (θ * ) = 1), fang defaults to the non-informative chance predictor E[σ(Z)] = E[σ(w ⊤ ⋆ x new )]. In other words,
The poorer the alignment between w ⋆ and w, the greater the magnitude of noise Z required to maintain calibration.
We will show in the next section that this angular interpolation idea leads to a uniquely Bregman optimal calibrated predictor. This provides the first calibration procedure that is calibrated and optimal in high dimensions, provably. Peusdocode for angular calibration, using angle estimator from Section 6, is included in Section G.
this section cite: ['b37', 'b7', 'b8', 'b9', 'b17', 'b53', 'b12']

Section: Main Result I: Calibrating Optimally using Angular Calibration
Before formally stating our results on optimality of angular calibration, we first define the following random probability vectors for label distribution,
q ⋆ := σ(w ⊤ ⋆ x new ) 1 -σ(w ⊤ ⋆ x new ) , qF := F ( w ⊤ x new ) 1 -F ( w ⊤ x new ) , qang ( θ) := fang ( w ⊤ x new ; θ) 1 -fang ( w ⊤ x new ; θ)(4)
where F : R → [0, 1] is any measurable function. Here, q ⋆ corresponds to the ground-truth probability distribution of the new label, qF the prediction probability distribution of an F -calibrated predictor, and q ang the prediction probability distribution of our angular predictor.
Next, we define the Bregman loss function. Definition 4.1 (Bregman Loss Functions). Let ϕ : R 2 → R be a strictly convex differentiable function. Then, the Bregman loss function D ϕ : R 2 × R 2 → R is defined as
D ϕ (x, y) = ϕ(x) -ϕ(y) -⟨x -y, ∇ϕ(y)⟩.
The Bregman loss function class covers common losses such as the squared loss D ϕ (x, y) := ∥x -y∥ 2 2 and Kullback-Liebler (KL) divergence D ϕ (x, y) = 2 j=1 x j log (x j /y j ) between two probability vectors x, y. Theorem 4.2 states that the prediction probability qang generated by the angular predictor uniquely minimizes any Bregman loss against the ground-truth probability vector q⋆ within the class of q F for any F . We defer the proof of the Theorem below to Section B. Theorem 4.2 (Optimality of angular predictor). Let ϕ : R 2 → R be any strictly convex differentiable function, and let D ϕ be the corresponding Bregman loss function. Let E xnew [ϕ(q ⋆ )] be finite. Then, the expected Bregman loss E xnew [D ϕ (q ⋆ , qF )] admits a unique minimizer (up to a.s. equivalence) among all q F , ∀F ∈ F := {f : R → [0, 1]}. Let this minimizer be
F ⋆ = arg min F ∈F E xnew [D ϕ (q ⋆ , qF )].
Further suppose that the link function σ is continuous. We then have that as n, d → ∞, we have
qang ( θ) -qF⋆ ( w ⊤ x new ) 2 2 → 0 in probability.
That is, the label prediction probability vector from angular calibration converges to the optimal label prediction probability vector given by F ⋆ .
We note that as θ = θ * , qang ( θ) precisely attains the optimal solution F * ( w ⊤ x new ). We defer the technical statement to Theorem B.2 in Section B. 5 Main Result II: Platt scaling is provably calibrated and Bregman-optimal
Platt scaling is arguably the most widely used calibration method in modern machine learning, yet its theoretical properties in high-dimensional settings remain unexplored. In this section, we identify conditions under which Platt scaling converges to our angular predictor, and is therefore well-calibrated and Bregman-optimal in high dimensions.
Platt scaling finds a mapping F of the prediction logits w ⊤ x new by minimizing the log-likelihood on a holdout dataset. In this section, we specifically consider the setting where we have a holdout dataset (x ho,i , y ho,i ) n ho i=1 and the negative log-likelihood
ln ho (F ) := n ho i=1 -y ho,i log F ( w ⊤ x ho,i ) -(1 -y ho,i ) log 1 -F ( w ⊤ x ho,i ) .(5)
The Platt calibration procedure then searches for a mapping F within some hypothesis class F platt that minimizes the negative log-likelihood. Elementary asymptotic theory then shows that as n ho → ∞ (i.e. the holdout set is sufficiently large), L(θ) in (5) converges to the population loss
ℓ ⋆ (F ) = E xnew D KL σ(w ⊤ ⋆ x new ) 1 -σ(w ⊤ ⋆ x new ) F ( w ⊤ x new ) 1 -F ( w ⊤ x new )(6)
almost surely up to a constant. This is exactly the argument of the Bregman loss E xnew [D ϕ (q ⋆ , qF )] from Theorem 4.2 for ϕ specialized as the negative Shannon entropy. That is, such calibration procedures are essentially trying to optimizing the Bregman loss but within the restricted hypothesis class.
This naturally raises the question of whether calibration procedures such as Platt scaling can achieve the optimal Bregman loss, say in the limit of a sufficiently large holdout set. Theorem 5.1 shows that, if σ is a probit link function (or is closely approximated by one up to an affine transformation-for instance, sigmoid(x) ≈ Φ( π/8 x)), and if the negative log-likelihood (5) is minimized over the hypothesis class of the form σ(Au + B), A, B ∈ R then the resulting Platt-scaled predictor converges to our angular predictor as n ho → ∞. Combining this connection with our results for the predictor fang (u; θ * ) which is our angular predictor fang (u; θ) with θ = θ * exactly. As mentioned previously (see also Theorem A.1 and Theorem B.2 in Appendix), fang (u; θ * ) is exactly calibrated and Bregmanoptimal, which shows that Platt scaling is both provably calibrated and Bregman optimal. This offers the first formal high-dimensional guarantees of this kind for the widely well-known Platt scaling procedure. We defer the proof of the Theorem below to Section C.
Theorem 5.1. Consider the predictor f n ho platt (u) calibrated by the Platt scaling procedure, that is,
f n ho platt ( w ⊤ x new ) = σ( Ân ho • w ⊤ x new + Bn ho ), with Ân ho , Bn ho = argmin (A,B)∈H ln ho (u → σ(Au + B))(7)
for ln ho (•) defined in (5). If the link function σ satisfies σ(x) = Φ(a•x+b) for some a ∈ R\{0}, b ∈ R and the point
(A * , B * ) defined in (9) is contained in a compact subset H ⊂ R 2 , the angular predictor defined in (3) satisfies fang (u; θ * ) = σ(A * • u + B * ) ∈ F platt ,(8) where
A * = cos(θ * ) ∥ ŵ∥ Σ 1 + a 2 sin 2 (θ * ) , B * = b a   1 1 + a 2 sin 2 (θ * ) -1   (9
)
and F platt = {u → σ(Au + B) : A, B ∈ R}. Moreover, as n ho → ∞, we have that Ân ho → A * , Bn ho → B * in probability and
sup u∈R f n ho platt (u) -fang (u; θ * ) → 0 (10
) in probability.
Here, in-probability convergence is with respect to the randomness of {(x ho,i , y ho,i ) n ho i=1 }. To be clear, the above theorem considers the asymptotics in the holdout set n ho for a fixed sample size and dimension n, d of the training dataset. We illustrate Theorem 5.1 in Figure 1 (see Section 7 for detailed settings) where the solid red line plots our angular predictor u → fang (u) defined in (3) and the dashed lines plot the predictor u → f n ho platt (u) calibrated by Platt scaling on increasingly large holdout sets n ho . We observe that the Platt scaling predictors indeed converge to our angular predictor as the holdout set sizes n ho increase.
this section cite: ['b4']

Section: Consistent angle estimation
Observe that the angular predictor fang depends on the unobserved quantity ⟨w ⋆ , w⟩ Σ . Using recent advancements from Equation (11) we are able to provide a consistent estimator for this quantity. For simplicity, we outline the estimation procedure here for a twice-differentiable loss function ℓ and strongly convex, twice-differentiable penalty g; analogous results for unregularized M-estimation and other losses/penalties found in [8]. We note that this result is part of a long line of development in observable estimation of unknown quantities in high dimensions [38,9,10,18,54].
A data driven estimator for ⟨w ⋆ , w⟩ 2 Σ proposed in [8] is:
â2 * = v n ∥X w -γ ψ∥ 2 + 1 n ψ⊤ X w -γ r2 2 1 n 2 Σ -1 2 X ⊤ ψ 2 + 2v n ψ⊤ X w + v2 n ∥X w -γ ψ∥ 2 -d n r2(11)
where ψ ∈ R n is the vector with components ψi = -
ℓ ′ i x ⊤ i w , v = 1 n Tr D -DX ĤX T D , γ = Tr X ĤX ⊤ D for D = diag ℓ ′′ y (X w) and Ĥ = X ⊤ DX + n∇ 2 g( w) -1 and r = ( ∥ ψ∥ 2 n ) 1/2 . It can be shown that â2 * -⟨w ⋆ , w⟩ 2 Σ → 0 in suitable high-dimensional sense.
We refer the technical statement to Theorem D.1 in Section D.
To estimate ⟨w ⋆ , w⟩ Σ , we also need to estimate its sign. We require reserving a constant fraction n ho = α • n of the n training data for the sign estimator, It can be shown that the probability of wrong sign identification using sgn decreases exponentially with n ho . We defer the proof to Section E. Proposition 6.1. Suppose that σ ′ (x) is well-defined and non-negative almost everywhere and σ ′ (x) > 0 on a set with non-zero Lebesgue measure. We then have for some absolute constant c > 0,
sgn := sign n ho i=1 w ⊤ x ho i • y ho i . (12
P ho ( sgn = sign (⟨w ⋆ , w⟩ Σ )) ≥ 1 -2 exp -cn ho (cos (θ * ) • Eσ ′ (Z)) 2
where Z ∼ N (0, 1) and P ho is with respect to the randomness in
y ho i , x ho i n ho i=1 .
Plugging (11) and ( 12) into (2), we have the following estimator θ for θ * in (2) θ := arccos ∥ w∥ -1
Σ sgn • â2 * .(13)
Theorem 6.2 below shows that θ is consistent. Corollary 6.2. Under the assumption of Proposition 6.1 and Theorem D.1, as n, d → ∞, we have that | θ -θ * | → 0 in probability where n ho = α • n for a fixed constant α > 0.
7 Numerical experiments
this section cite: ['b10', 'b7', 'b37', 'b8', 'b9', 'b17', 'b53', 'b7', 'b10']

Section: Simulations
This section presents a simple simulation to demonstrate results in Section 6. We generate i.i.d. samples x i iid ∼ N (0, Σ), i = 1, ..., n where Σ = 1 d Σ and Σkl = 0.5 |k-l| , ∀k, l ∈ {1, ..., d}; we also generate labels from (1) with σ(u) = sigmoid(3u + 1) = 1/(1 + exp(-(3u + 1))) and w ⋆ ∼ N (0, I d ) (normalized to ∥w ⋆ ∥ Σ = 1). We consider the case of ridge logistic regression with ℓ yi (w
⊤ x) = -y i log(p w (x i )) -(1 -y i ) log(1 -pw (x i )) with pw (x i ) = 1 1+exp(-xiw)
and g(w) = λ 2d ∥w∥ 2 2 with λ = 0.5. We assume that we are in a data deficient setting where n = 1000, p = 2000.
The realizability plots in Figure 2 are generated from a test set of size n test = 20000. To produce these plots, we bin the predicted probabilities for label 1 (on the x-axis) and then compute the average of the observed label within each bin (on the y-axis). Perfect calibration would align the binned Table 1: ECE on pretrained feature extractors. "Uncal." = uncalibrated; "Angular" = our method; "Platt/Iso" use n ho ∈ {100, 500} labeled hold-out points.
this section cite: []

Section: Model-Dataset Uncal. Angular Platt 100 Iso 100 Platt 500 Iso 500
ResNet-34-CIFAR-10 0.1236 0.0199 0.0561 0.0484 0.0259 0.0298 MiniLM-20 Newsgroups 0.1392 0.0249 0.0931 0.1107 0.0679 0.0813 ChemBERTa-Tox21 0.1389 0.0132 0.0236 0.0497 0.0175 0.0293 points with the 45°line. In the left and right panels, Platt scaling is derived using holdout sets of n ho = 100 and n ho = 20000, respectively, whereas both the uncalibrated predictor and the angular predictor remain unchanged across the two panels.
From the reliability plots, we see that the uncalibrated predictor (blue) is poorly calibrated, while, as expected, the angular predictor (green) shows good calibration. Here, we have used the angle estimator (11) and the sign estimator (12) to estimate the value of ⟨w ⋆ , w⟩ Σ . The estimated value for ⟨w ⋆ , w⟩ Σ is 0.4356 while the true value is 0.4526. We also ran 5000 Monte Carlo trials where we found the probability of incorrect sign estimation to be 0.89% with a holdout set of size n ho = 100.
In contrast, the left panel of Figure 2 shows that Platt scaling (orange) with a holdout set size of n ho = 100 fails to properly calibrate. However, when the holdout set size is increased to n ho = 20000, Platt scaling also calibrates correctly. When n ho = 20000, the predictor calibrated from Platt scaling is found to be (using scikit-learn package's CalibratedClassifierCV routine [68])
f n ho platt ( w ⊤ x new ) = σ( Ân ho • w ⊤ x new + Bn ho ),
this section cite: ['b10', 'b11', 'b67']

Section: Semi-real experiments
We assess angular calibration on semi-real tasks that keep real-data covariates but simulate labels from the known generative model. We maintain settings in Section 7.1 but replace data covariates with: (i) final-layer logits of pretrained deep networks; and (ii) classic UCI benchmarks.
We found that (11) plug-in estimator for ⟨w ⋆ , ŵ⟩ is unstable on these real datasets. This is a known issue for estimators like (11) that are based on Wigner-type random-matrix-theoretic assumptions. Modifying these estimator are an active research area [54,57]. To isolate calibration effects, we simulate w ⋆ and labels, using the true angle.
Each dataset is split into training set n, a large unlabeled pool n cov ≫ n train for covariance estimation, and n test . We report Expected Calibration Error (ECE; lower is better) [29]. Post-hoc baselines use a labeled hold-out of size n ho ∈ {100, 500} ("Platt 100/500" and "Iso 100/500").
this section cite: ['b53', 'b56', 'b28']

Section: Pretrained representations.
We fit a linear head on frozen embeddings and calibrate the resulting logits: ResNet-34 (ImageNet-1K pretrain) on CIFAR-10 [35,24,46], MiniLM sentence embeddings on 20 Newsgroups [85,72,51], and ChemBERTa on Tox21 from MoleculeNet [20,87]. We have n × d = (300 × 512)/(800 × 384)/(800 × 768), n cov = 30,000/3,000/3,000 and n test = 10,000/1,000/500 for CIFAR-10 / 20NG / Tox21. The results are reported in Table 1; the reliability plots are deferred to Section H.2.
UCI benchmarks. On Communities & Crime, Splice-junction Gene Sequences, and Madelon [71,1,32], we train linear predictors on raw covariates with n×d = (200×100)/(100×180)/(300× 500), n cov = 700/2000/900, and n test = 593/900/900 for Communities & Crime / Splice-junction / Madelon. The results are reported in Table 2; the reliability plots are deferred to Section H.2.
this section cite: ['b34', 'b23', 'b45', 'b84', 'b71', 'b50', 'b19', 'b86', 'b70', 'b0', 'b31']

Section: Extensions and future directions
We derived our theoretical results assuming that the covariates are Gaussian-although at first pass this might appear stylistic, recent universality results [37,56,33,26,49,62] demonstrate that these results should continue to hold as long as the covariates have sufficiently light tails. We demonstrate this with further experiments. In Section H.2 and Figure 4, we reproduce Figure 1 and 2 with non-Gaussian design matrices (iid Rademacher and uniform entries respectively) where we observe that our results continue to be accurate. Establishing such universality formally should be an interesting avenue for future work-we include an informal discussion here. Consider a general setting where x i d = Σ 1/2 z i , where z i has iid entries with zero mean, unit variance and finite moments, and Σ = p -1 Σ where Σ has bounded condition number. Denote w = Σ1/2 w, w ⋆ = Σ1/2 w ⋆ and x new , z new to be observations at test time with the same distribution as x i , z i respectively. If we could apply the multivariate CLT [11], we would obtain
p -1/2 p i=1 w ⋆,i z new,i , p -1/2 p i=1 w i z new,i ⇒ (Z 1 , Z 2 ) ,(14)
where (Z 1 , Z 2 ) ∼ N (0, L) for some positive definite covariance matrix L ∈ R 2×2 . To apply the multivariate CLT, we require to check the following moment condition (c.f. [11])
p -3/2 p i=1 E z 3 2 new ,i ( w ⋆,i , w i ) ∥ w∥ 2 Σ ⟨ w, w ⋆ ⟩ Σ ⟨ w, w ⋆ ⟩ Σ ∥w ⋆ ∥ 2 Σ -1 w ⋆,i w i 3 2 ≤ p -1/2 E z 3 2 new ,i σ -1 min (L) 1 p p i=1 | w ⋆,i | 3 + 1 p p i=1 | w i | 3 = o 1 √ p .
We claim that the above holds almost surely for sufficiently large p, if we have constants W 1 , W 2 > 0 for which the following holds
∥ w∥ 2 Σ ⟨ w, w ⋆ ⟩ Σ ⟨ w, w ⋆ ⟩ Σ ∥w ⋆ ∥ 2 Σ → L, 1 p p i=1 | w ⋆,i | 3 , 1 p p i=1 | w i | 3 → (W 1 , W 2 ).
Recent universality results for either approximate message passing algorithms [19] or convex gaussian minmax theorems (CGMT) [33] allow one to prove this beyond Gaussian designs. Numerous works have already applied such arguments in the context of other high-dimensional problems [37,62,56,49]. Using (14), the conditional distribution (15) in the proof of Theorem 3.2 can be extended to non-Gaussian, Wigner-type features preconditioned by some known Σ 1/2 , thus leading to a proof of Theorem 2 beyond Gaussian designs. In the interest of space, we defer formalizing this to future work.
Finally, we consider binary classification in this work; it would be interesting to extend our results to multi-index models, which includes multi-class classification, additive and interaction models, and two-layer neural networks [82,88,15]; see details in Section F. Multi-index model can be defined as follows: for K ≥ 2 and unobserved indices W ⋆ = [w ⋆1 , . . . , w ⋆K ] ∈ R d×K , the true logits and model outputs are
G := W ⊤ ⋆ x new ∈ R K , π (x new ) = g(G)
where g is a generalized link (vector-or scalar-valued). We show in Section F that, given an estimator W = [ w 1 , . . . , w K ] of W ⋆ , angular predictor in Definition 3.1 may be extended for the multi-index model as follows,
f ang W ⊤ x new := E Z [g (M ⋆ S + L ⋆ Z)]
where the matrix quantity M ⋆ and L ⋆ depends on cross-index angles ⟨w ⋆k , w ℓ ⟩ Σ , ℓ, k ∈ [K]. Though no estimators are given for these cross-index angles in literature as far as we know, recent theory for multi-index models [82] suggests that analogues of the single-index angle estimators [8] are feasible. We leave this to future works.
this section cite: ['b36', 'b55', 'b32', 'b25', 'b48', 'b61', 'b10', 'b10', 'b18', 'b32', 'b36', 'b61', 'b55', 'b48', 'b14', 'b81', 'b87', 'b14', 'b81', 'b7']

Section: A Proof of Theorem 3.2
Before proving Theorem 3.2, we first show that fang (•; θ * ) is exactly calibrated.
Theorem A.1. The predictor fang defined in (3) is well-calibrated at all p ∈ [0, 1] when . That is, for any p ∈ [0, 1] and any d, n ∈ N +
∆ cal p fang (•; θ * ) = p -E xnew σ w ⊤ ⋆ x new | fang w ⊤ x new ; θ * = p = 0.
Proof of Theorem A.1. Let us define the following event
A := fang w ⊤ ⋆ x new ; θ * = p .
We have
E xnew σ w ⊤ ⋆ x new | A (i) = E xnew E xnew σ w ⊤ ⋆ x new | x ⊤ new w | A (ii) = E xnew E Z σ 1 ∥ w∥ Σ • cos (θ * ) • x ⊤ new w + sin (θ * ) • Z | A = E xnew fang w ⊤ ⋆ x new ; θ * | A = p.
where (i) follows from tower property of expectation and the fact that fang (x new ) depends on x new only through x ⊤ new w, (ii) follows from conditional expectation of multivariate Gaussian distribution
w ⊤ ⋆ x new | x ⊤ new w L = 1 ∥ w∥ Σ • cos (θ * ) • x ⊤ new w + sin (θ * ) • Z(15)
for some Z ∼ N (0, 1).
Proof of Theorem 3.2. Using result from Theorem A.1, it suffices to show that as | θ -θ * | → 0 in probability, we have that
∆ cal p fang (•; θ * ) -∆ cal p fang •; θ → 0.(16)
Let us introduce the following notation for the ease of presentation:
X := σ(w ⊤ ⋆ x new ), Ŷ = fang w ⊤ x new ; θ , Y * = fang w ⊤ x new ; θ * .
Then, we can write LHS of ( 16) as
E[X | Ŷ = p] -E [X | Y ⋆ = p] = 1 f Y (p) 1 0 xf X, Ŷ (x, p)dx - 1 f Y⋆ (p) 1 0 xf X,Y⋆ (x, p)dx
where f Ŷ , f Y * , f X,Y⋆ , f X, Ŷ are the distribution density functions of Ŷ , Y * and joint density functions of (X, Y ⋆ ) and (X, Ŷ ). We now show that the RHS of the above converges to 0. Firstly,
| 1 f Y⋆ (p) - 1 f Ŷ (p) | → 0 because | Ŷ -Y ⋆ | → 0 in probability (
and thus in distribution) by continuous mapping theorem. Secondly,
1 0 xf X, Ŷ (x, p)dx - 1 0 xf X,Y⋆ (x, p)dx → 0
by bounded convergence theorem and the fact that (X, Ŷ ) converges to (X, Y * ) jointly. We conclude the proof.
this section cite: []

Section: B Proof of Theorem 4.2
We first state a result from [4] for general random variables. Proposition B.1 (Theorem 1, [4]). Let ϕ : R d → R be a strictly convex differentiable function, and let D ϕ be the corresponding Bregman loss function. Let X be an arbitrary random variable taking values in R d for which both E[X] and E[ϕ(X)] are finite. Then, among all functions of Z, the conditional expectation is the unique minimizer (up to a.s. equivalence) of the expected Bregman loss, i.e.,
arg min Y ∈σ(Z) E [D ϕ (X, Y )] = E[X | Z].
Using the above results, we show that angular calibration with θ = θ * minimizes Bregman divergence to true label distribution among predictors of the form F ( w ⊤ x new ). Theorem B.2 (Optimality of angular predictor). Let ϕ : R 2 → R be any strictly convex differentiable function, and let D ϕ be the corresponding Bregman loss function. Let E xnew [ϕ(q ⋆ )] be finite. Then, the expected Bregman loss E xnew [D ϕ (q ⋆ , qF )] admits a unique minimizer (up to a.s. equivalence) among all q F , ∀F ∈ F := {f : R → [0, 1]}. Let this minimizer be
F ⋆ = arg min F ∈F E xnew [D ϕ (q ⋆ , qF )].
We then have that almost surely
qang (θ * ) = F * ( w ⊤ x new )
where qang (θ * ) is the label prediction probability vector by angular calibration given in (4) with θ replaced by θ * .
Proof of Theorem B.2. Firstly, we set X, Y, Z in Theorem B.1 as
X ← σ(w ⊤ ⋆ x new ) 1 -σ(w ⊤ ⋆ x new ) , Y ← F ( w ⊤ x new ) 1 -F ( w ⊤ x new ) , Z ← w ⊤ x new .
The result then follows from Theorem B.1 and the following
E[X | Z] = E σ(w ⊤ ⋆ x new ) 1 -σ(w ⊤ ⋆ x new ) | x ⊤ new w = fang (x new ) 1 -fang (x new )
where we used (15) and (3) for the last equality.
Now we are ready to state proof of Theorem 4.2.
Proof. Using result from Theorem B.2, it suffices to show that qang ( θ) -qang (θ * ) 2 → 0 in probability. This is an immediate consequence of the continuous mapping theorem under the assumption σ is continuous.
this section cite: ['b3', 'b3']

Section: C Proof of Theorem 5.1
Before proving Theorem 5.1, we first state two classic analysis results that we will later use. Proposition C.1 (Theorem 5.7, [83]). Let ℓ n be random functions on H, ℓ ⋆ be a fixed function on H, and θ ⋆ ∈ H such that (i) uniform convergence of n -1 ℓ n to ℓ ⋆ holds:
sup θ∈H 1 n ℓ n (θ) -ℓ ⋆ (θ)
inprob.
----→ n→∞ 0, (ii) the mode of ℓ ⋆ is well-separated, i.e for all ε > 0,
sup θ∈H:d(θ,θ ⋆ )≥ε ℓ ⋆ (θ) < ℓ ⋆ (θ ⋆ )
Then any sequence θn maximizing ℓ n converges in probability to θ ⋆ .
Proposition C.2 (Theorem 10.8, [73]). Let C be a relatively open convex set, and let f 1 , f 2 , . . ., be a sequence of finite convex functions on C. Suppose that the sequence converges pointwise on a dense subset of C, i.e. that there exists a subset C ′ of C such that its closure satisfies clC ′ ⊃ C and, for each x ∈ C ′ , the limit of f 1 (x), f 2 (x), . . ., exists and is finite. The limit then exists for every x ∈ C, and the function f , where
f (x) = lim i→∞ f i (x)
is finite and convex on C. Moreover the sequence f 1 , f 2 , . . ., converges to f uniformly on each closed bounded subset of C.
this section cite: ['b82', 'b72']

Section: Now we are ready to prove Theorem 5.1.
Proof of Theorem 5.1. (8) is obtained from applying the well-known identity below for probit function Φ(•)
EΦ(µ + σ • Z) = Φ µ √ 1 + σ 2 , Z ∼ N (0, 1)
to ( 3).
To prove that Ân ho → A * , Bn ho → B * as n ho → ∞, we would like to apply (C.1) by setting n ← n ho , θ ← {A, B}, θ ⋆ ← {A * , B * },
ℓ n (θ) ← ℓ n ho (A, B) := n ho i=1 -y ho,i log F A,B ( w ⊤ x ho,i ) -(1 -y ho,i ) log 1 -F A,B ( w ⊤ x ho,i ) ,(17)
and
ℓ ⋆ (θ) ← ℓ ⋆ (A, B) :=E xnew -σ w ⊤ ⋆ x new log F A,B w ⊤ x new -1 -σ w ⊤ ⋆ x new log 1 -F A,B w ⊤ x new .(18)
where we used the notation Here, RHS of ( 18) is up to an affine transform of the KL divergence (6) and, therefore, it follows from ( 8) and (??) that its minimizer is indeed A * , B * .
To verify condition (i) of Theorem C.1, we first note that (17) converges to ( 18 where H := w ⊤ ⋆ x new ∼ N (0, 1). Then, we have that
∇ 2 f (A, B) = σ(H) • f ′′ 1 (aA • H + aB + b) • +(1 -σ(H))f ′′ 2 (aA • H + aB + b) • a 2 H 2 a 2 H a 2 H a 2
which is positive-definite almost surely when a ̸ = 0. Hence, we have that almost surely
f (t(A 1 , B 1 ) + (1 -t)(A 1 , B 1 )) < tf ((A 1 , B 1 )) + (1 -t)f ((A 1 , B 1 ))
which implies that
E xnew f (t(A 1 , B 1 ) + (1 -t)(A 1 , B 1 )) < tE xnew f ((A 1 , B 1 )) + (1 -t)E xnew f ((A 1 , B 1 )) .
The claim that ℓ ⋆ (A, B) is strictly convex follows.
It then follows from Theorem C.1 that Ân ho → A * , Bn ho → B * in probability as n ho → ∞. The uniform convergence f n ho platt (u) → fang (u) follows immediately.
this section cite: []

Section: D Inner product estimation
We restate the following results from Theorem 4.4, [8]. We note that the quantity r4 v2t2 in the error bound is observable and is typically of constant order in the proportional regime. See [8] for details. Theorem D.1. Suppose ℓ is continuously differentiable and g is strongly convex and twice differential penalty function. Assume also that 1 2δ ≤ d n ≤ 1 δ for some δ > 0, for arbitrarily large probability 1 -δ, the following holds
E â2 * -⟨w ⋆ , w⟩ 2 Σ ≤ C r4 v2 t2 • n -1/2
where C is a constant depending only on g and δ.
this section cite: ['b7', 'b7']

Section: E Sign estimation
Proof of Theorem 6.1. With respect to randomness in validation dataset (that is, w is treated as deterministic), we have that w ⊤ x ho i • y ho i are iid across i and satisfies that
w ⊤ x ho i • y ho i L = w ⊤ x ho i • Bern i σ ⟨w ⋆ , w⟩ Σ w ⊤ Σ w w ⊤ x ho i + sin (θ * ) • Z i L = ∥ w∥ Σ U i • Bern i (σ (cos (θ * ) U i + sin (θ * ) • Z i )) = H i
where Z i iid ∼ N (0, 1) and we have used w ⊤ x ho i L = ∥ w∥ Σ • U i for U i iid ∼ N (0, 1). It follows from Gaussian integration by parts that
EH i = ⟨w ⋆ , w⟩ Σ • Eσ ′ (cos (θ * ) U i + sin (θ * ) • Z i ) = ⟨w ⋆ , w⟩ Σ Eσ ′ (Z).
Meanwhile, H i is subGaussian with subGaussian norm
∥H i ∥ 2 ψ2 ≤ ∥ w∥ 2 Σ ∥U i ∥ 2 ψ2 ≤ 3∥ w∥ 2 Σ .
By theorem assumption, Eσ ′ (Z) > 0 and
sign (⟨w ⋆ , w⟩ Σ Eσ ′ (Z)) = sign (⟨w ⋆ , w⟩ Σ )
So the sign identification of sgn is correct if the following event holds
1 n ho n ho i=1 w ⊤ x i • y i -⟨w ⋆ , w⟩ Σ Eσ ′ (Z) < |⟨w ⋆ , w⟩ Σ Eσ ′ (Z)| .
this section cite: []

Section: By Hoeffding's inequality,
P ho 1 n ho n ho i=1 w ⊤ x i • y i -⟨w ⋆ , w⟩ Σ Eσ ′ (Z) > |⟨w ⋆ , w⟩ Σ Eσ ′ (Z)| ≤ 2 exp - cn ho (Eσ ′ (Z)) 2 ⟨w ⋆ , w⟩ 2 Σ ∥ w∥ 2 Σ = 2 exp -cn ho (cos (θ * ) • Eσ ′ (Z)) 2 .
The theorem statement follows.
this section cite: []

Section: F Angular calibration for multi-index models
Let
x new ∼ N (0, Σ) ⊂ R d . Fix K ≥ 2 and let W ⋆ = [w ⋆1 , . . . , w ⋆K ] ∈ R d×K . Define G := W ⊤ ⋆ x new ∈ R K , π(x new ) = g(G)
, where g is a generalized link (vector-or scalar-valued). This setup covers:
• Two-layer nets (frozen outer layer):
g(u) = k a k σ(u k ) • Multi-class softmax: g(u) = softmax(u) • Additive index model: g(u) = k f k (u k ) • Interaction index model: g(u) = k f k (u k ) + k<ℓ h kℓ (u k , u ℓ )
The following extends angular calibration to mutli-index models. Note that to apply the angular predictor (19), we must estimate cross-index angles ⟨w ⋆,k , w ℓ ⟩ Σ , similarly to the single-index case. Though no estimator is given in literature as far as we know, recent theory for multi-index models [82] suggests that analogues of the single-index angle estimators [8] are feasible. We leave derivation of these estimators to future works. Theorem F.1 (Angular calibration for multi-index models). Let W = [ w 1 , . . . , w K ] ∈ R d×K be any estimator. Set
D := diag(∥ w 1 ∥ Σ , . . . , ∥ w K ∥ Σ ), S := D -1 W ⊤ x new ∈ R K . Then we have K × K covariance blocks Cov(G) = W ⊤ ⋆ ΣW ⋆ , R := Cov(S) = D -1 W ⊤ Σ W D -1 , C := Cov(G, S) = W ⊤ ⋆ Σ W D -1 where R kℓ = ⟨ w k , w ℓ ⟩ Σ ∥ w ℓ ∥ Σ ∥ w k ∥ Σ , C kℓ = ⟨w ⋆k , w ℓ ⟩ Σ ∥ w ℓ ∥ Σ ∥w ⋆k ∥ Σ
Then, assuming that R is invertible, we may define
M ⋆ := CR -1 , Σ ⋆ := Cov(G) -CR -1 C ⊤
and we have that G | S ∼ N (M ⋆ S, Σ ⋆ ). For any factor L ⋆ with L ⋆ L ⊤ ⋆ = Σ ⋆ and any Z ∼ N (0, I K ) independent of everything, define the multi-index angular predictor
f ang W ⊤ x new := E Z g M ⋆ S + L ⋆ Z .(19)
Then for any p ∈ ∆ K-1 and any d, n ∈ N + ,
p -E π(x new ) f ang ( W ⊤ x new ) = p = 0.
Proof. Let x new ∼ N (0, Σ) and define
G := W ⊤ ⋆ x new ∈ R K , S := D -1 W ⊤ x new ∈ R K , with D = diag(∥ w 1 ∥ Σ , . . . , ∥ w K ∥ Σ ) and ∥u∥ Σ := √ u ⊤ Σu, ⟨u, v⟩ Σ := u ⊤ Σv. Set the 2K × d linear map T := W ⊤ ⋆ D -1 W ⊤ , Y := G S = T x new .
Since x new is Gaussian and Y is a linear transform, Y is jointly Gaussian with mean 0 and covariance
Cov(Y ) = T ΣT ⊤ = W ⊤ ⋆ ΣW ⋆ W ⊤ ⋆ Σ W D -1 D -1 W ⊤ ΣW ⋆ D -1 W ⊤ Σ W D -1 . Thus, Cov(G) = W ⊤ ⋆ ΣW ⋆ , R := Cov(S) = D -1 W ⊤ Σ W D -1 , C := Cov(G, S) = W ⊤ ⋆ Σ W D -1 . In particular, for k, ℓ ∈ [K], R kℓ = ⟨ w k , w ℓ ⟩ Σ ∥ w k ∥ Σ ∥ w ℓ ∥ Σ , C kℓ = ⟨w ⋆k , w ℓ ⟩ Σ ∥w ⋆k ∥ Σ ∥ w ℓ ∥ Σ .
Define the K × K matrices
M ⋆ := CR -1 , Σ ⋆ := Cov(G) -CR -1 C ⊤ .
this section cite: ['b81', 'b7']

Section: Consider the linear residual
U := G -M ⋆ S = G -CR -1 S. Because Y is Gaussian, U is Gaussian; further, Cov(U, S) = Cov(G, S) -CR -1 Cov(S, S) = C -CR -1 R = 0,
so U and S are independent (uncorrelated jointly Gaussian vectors are independent). Moreover,
Cov(U ) = Cov(G) -CR -1 C ⊤ = Σ ⋆ .
Hence we have the orthogonal decomposition
G = M ⋆ S + U, U ⊥ ⊥ S, U ∼ N (0, Σ ⋆ ).
Equivalently, the conditional distribution is
G S ∼ N M ⋆ S, Σ ⋆ ,
which is the standard multivariate normal conditioning formula via the Schur complement.
Finally, let L ⋆ be any matrix satisfying L ⋆ L ⊤ ⋆ = Σ ⋆ and let Z ∼ N (0, I K ) independent of (G, S). Then U d = L ⋆ Z and G S d = M ⋆ S + L ⋆ Z. Therefore, for any measurable g (vector-or scalar-valued) with the requisite integrability,
E[g(G) | S] = E Z g M ⋆ S + L ⋆ Z ,
which yields the stated multi-index angular predictor
f ang W ⊤ x new := E Z g M ⋆ S + L ⋆ Z .
To conclude, set X := π(x new ) = g(G) ∈ ∆ K-1 and
Y := f ang W ⊤ x new = E[g(G) | S] . Then Y is σ(S)-measurable and Y = E[X | S] almost surely. We claim that E[X | Y ] = Y a.s.
Indeed, for any bounded measurable φ :
∆ K-1 → R, E φ(Y ) (X -Y ) = E E φ(Y ) (X -Y ) | S = E φ(Y ) E[X | S] -Y = 0, so E[X | Y ] = Y (coordinate-
this section cite: []

Section: wise) by the defining property of conditional expectation.
By the existence of regular conditional expectations, this implies that for P Y -almost every p ∈ ∆ K-1 ,
E π(x new ) f ang ( W ⊤ x new ) = p = E[X | Y = p] = p, i.e., p -E π(x new ) f ang ( W ⊤ x new ) = p = 0.
This establishes exact calibration of the multi-index angular predictor.
this section cite: []

Section: G Pseudocode
Algorithm 1: Angular Calibration Input :Training data {(x i , y i )} n i=1 ; link σ; convex loss ℓ and penalty g; covariance Σ (known or estimated from unlabeled data); holdout set {(x ho i , y ho i )} n ho i=1 for sign estimation; Output :Calibrated predictor fang (x) that returns p ∈ [0, 1] for any new x.
(A) Fit base linear model w ← arg min w
1 n n i=1 ℓ yi (w ⊤ x i ) + g(w) ∥ w∥ Σ ← ( w ⊤ Σ w) 1/2 (B) Observable magnitude of ⟨w ⋆ , w⟩ Σ ψi ← -ℓ ′ yi (x ⊤ i w); ψ ← ( ψ1 , . . . , ψn ) ⊤ D ← diag ℓ ′′ y (X w) ; Ĥ ← X ⊤ DX + n ∇ 2 g( w) -1 v ← 1 n Tr(D -DX ĤX ⊤ D); γ ← Tr(X ĤX ⊤ D); r2 ← ∥ ψ∥ 2 /n â2 * ← v n ∥X w -γ ψ∥ 2 + 1 n ψ⊤ X w -γ r2 2 1 n 2 Σ -1 2 X ⊤ ψ 2 + 2v n ψ⊤ X w + v2 n ∥X w -γ ψ∥ 2 -d n r2 //
Est. of ⟨w ⋆ , w⟩ 2 Σ (C) Sign via holdout correlation sgn ← sign n ho i=1 ( w ⊤ x ho i ) y ho i (D) Angle & interpolation weights c ← sgn â2 * ∥ w∥ Σ ; c ← min{1, max{-1, c}} // numerical clip θ ← arccos(c); α ← cos θ = c; β ← sin θ = √ 1 -α 2 (E) Define calibrated predictor fang For any new x: u ← w ⊤ x; s ← u/∥ w∥ Σ draw z 1 , . . . , z M iid ∼ N (0, 1) and set fang (x) ≈ 1 M M j=1 σ(αs + βz j ) return fang (x)
this section cite: []

Section: I Simulation code and reproduction
The simulation code Angular_calibration.ipynb, Semi_experiments.ipynb included as supplementary material. The experiment can be run on modern personal computers without needing special computing hardware. Detailed instructions is provided in the simulation code.
this section cite: []

Section: 
H Additional plots H.1 Universality 0.10 0.30 0.50 0.70 0.90 Predicted Probability 0.10 0.30 0.50 0.70 0.90 Empirical Probability 45° line Uncalibrated Angular Platt scaling (nho = 100) 0.10 0.30 0.50 0.70 0.90 Predicted Probability 0.10 0.30 0.50 0.70 0.90 Empirical Probability 45° line Uncalibrated Angular Platt scaling (nho = 20000) 6 4 2 0 2 4 6 u 0.0 0.2 0.4 0.6 0.8 1.0 prediction prob. 0.10 0.30 0.50 0.70 0.90 Predicted Probability 0.10 0.30 0.50 0.70 0.90 Empirical Probability 45° line Uncalibrated Angular Platt scaling (nho = 100) 0.10 0.30 0.50 0.70 0.90 Predicted Probability 0.10 0.30 0.50 0.70 0.90 Empirical Probability 45° line Uncalibrated Angular Platt scaling (nho = 20000) 6 4 2 0 2 4 6 u 0.0 0.2 0.4 0.6 0.8 1.0 prediction prob.  1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We have clearly state the claims made, including the contributions made in the paper and important assumptions and limitations, in abstract and introduction Guidelines: • The answer NA means that the abstract and introduction do not include the claims made in the paper. • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Please see Limitations and future directions section. Guidelines: • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: All assumptions are included either in section 2 or in each theorem statement. Full proofs are included in appendices.
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
Answer: [Yes] Justification: We have fully disclose all the information needed to reproduce the main experimental results.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
Answer: [Yes] Justification: We have provided source code for our simulations in supplementary file, along with detailed instructions.
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
Answer: [Yes] Justification: All simulation details are disclosed in Section 7 Simulations.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: See Section 7.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: See Appendix F Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The paper conform, in every respect, with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: Our work is purely theoretical, aimed at advancing the mathematical understanding of calibration in an idealistic setting. It does not involve any personal, sensitive, or protected information. As such, there is no plausible way for our methods to be used for discrimination, profiling, surveillance, or any other harmful purpose.
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
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
this section cite: []

Section: References
Ref_id:b0 Title: Molecular biology (splice-junction gene sequences) [dataset]. UCI Machine Learning Repository Year: (1991)
Ref_id:b1 Title: The neural tangent kernel in high dimensions: Triple descent and a multi-scale theory of generalization Year: (2020)
Ref_id:b2 Title: Don't just blame over-parametrization for over-confidence: Theoretical analysis of calibration in binary classification Year: (2021)
Ref_id:b3 Title: On the optimality of conditional expectation as a bregman predictor Year: (2005)
Ref_id:b4 Title: Optimal errors and phase transitions in high-dimensional generalized linear models Year: (2019)
Ref_id:b5 Title: Optimal m-estimation in high-dimensional regression Year: (2013)
Ref_id:b6 Title: The need for uncertainty quantification in machine-assisted medical decision making Year: (2019)
Ref_id:b7 Title: Observable adjustments in single-index models for regularized m-estimators Year: (2022)
Ref_id:b8 Title: De-biasing the lasso with degrees-of-freedom adjustment Year: (2022)
Ref_id:b9 Title: Debiasing convex regularized estimators and interval estimation in linear models Year: (2023)
Ref_id:b10 Title: Vidmantas Bentkus. A lyapunov-type bound in rd. Theory of Probability & Its Applications Year: (2005)
Ref_id:b11 Title: Classifier calibration with roc-regularized isotonic regression Year: (2024)
Ref_id:b12 Title: On the appropriateness of platt scaling in classifier calibration Year: (2021)
Ref_id:b13 Title: Reliability, sufficiency, and the decomposition of proper scores Year: (2009)
Ref_id:b14 Title: Survey on algorithms for multi-index models Year: (2025)
Ref_id:b15 Title: The phase transition for the existence of the maximum likelihood estimate in high-dimensional logistic regression Year: (2020)
Ref_id:b16 Title: Correlation adjusted debiased lasso: debiasing the lasso with inaccurate covariate model Year: (2024)
Ref_id:b17 Title: The lasso with general gaussian designs with applications to hypothesis testing Year: (2023)
Ref_id:b18 Title: Universality of approximate message passing algorithms Year: (2021)
Ref_id:b19 Title: ChemBERTa: Large-scale self-supervised pretraining for molecular property prediction Year: (2020)
Ref_id:b20 Title: Expectation consistency for calibration of neural networks Year: (2023)
Ref_id:b21 Title: Theoretical characterization of uncertainty in high-dimensional linear classification Year: (2023)
Ref_id:b22 Title: The comparison and evaluation of forecasters Year: (1983)
Ref_id:b23 Title: ImageNet: A large-scale hierarchical image database Year: (2009)
Ref_id:b24 Title: Message-passing algorithms for compressed sensing Year: (2009)
Ref_id:b25 Title: Spectral universality in regularized linear regression with nearly deterministic sensing matrices Year: (2024)
Ref_id:b26 Title: A unifying tutorial on approximate message passing Year: (2022)
Ref_id:b27 Title: Weather forecasting with ensemble methods Year: (2005)
Ref_id:b28 Title: On calibration of modern neural networks Year: (2017)
Ref_id:b29 Title: Distribution-free binary classification: prediction sets, confidence intervals and calibration Year: (2020)
Ref_id:b30 Title: Online platt scaling with calibeating Year: (2023)
Ref_id:b31 Title:  Year: (2003)
Ref_id:b32 Title: Universality of regularized regression estimators in high dimensions Year: (2023)
Ref_id:b33 Title: Surprises in highdimensional ridgeless least squares interpolation Year: (2022)
Ref_id:b34 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b35 Title: Isotonic distributional regression Year: (2021)
Ref_id:b36 Title: Universality laws for high-dimensional learning with random features Year: (2022)
Ref_id:b37 Title: Debiasing the lasso: Optimal sample size for Gaussian designs Year: (2018)
Ref_id:b38 Title: A new central limit theorem for the augmented ipw estimator: Variance inflation, cross-fit covariance and beyond Year: (2022)
Ref_id:b39 Title: Smooth isotonic regression: a new method to calibrate predictive models Year: (2011)
Ref_id:b40 Title: Calibrating predictive model estimates to support personalized medicine Year: (2012)
Ref_id:b41 Title: Statistical challenges of high-dimensional data Year: (2009)
Ref_id:b42 Title: Moment multicalibration for uncertainty estimation Year: (2021)
Ref_id:b43 Title: The isotron algorithm: High-dimensional isotonic regression Year: (2009)
Ref_id:b44 Title: Being bayesian, even just a bit, fixes overconfidence in relu networks Year: (2020)
Ref_id:b45 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b46 Title: Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration Year: (2019)
Ref_id:b47 Title: Verified uncertainty calibration Year: (2019)
Ref_id:b48 Title: Universality in block dependent linear models with applications to nonlinear regression Year: (2024)
Ref_id:b49 Title: Simple and scalable predictive uncertainty estimation using deep ensembles Year: (2017)
Ref_id:b50 Title: Learning to filter netnews Year: (1995)
Ref_id:b51 Title: Random linear estimation with rotationally-invariant designs: Asymptotics at high temperature Year: (2023)
Ref_id:b52 Title: Understanding optimal feature transfer via a fine-grained bias-variance analysis Year: (2024)
Ref_id:b53 Title: Spectrum-aware adjustment: A new debiasing framework with applications to principal components regression Year: (2023)
Ref_id:b54 Title: Just interpolate: Kernel "ridgeless" regression can generalize Year: (2020)
Ref_id:b55 Title: A precise high-dimensional asymptotic theory for boosting and minimum-l1-norm interpolated classifiers Year: (2022)
Ref_id:b56 Title: Roti-gcv: Generalized cross-validation for rightrotationally invariant data Year: (2024)
Ref_id:b57 Title:  Year: (2019)
Ref_id:b58 Title: The generalization error of random features regression: Precise asymptotics and the double descent curve Year: (2022)
Ref_id:b59 Title: Evaluating uncertainty quantification in end-to-end autonomous driving control Year: (2018)
Ref_id:b60 Title: Approximate message passing with spectral initialization for generalized linear models Year: (2021)
Ref_id:b61 Title: Universality of empirical risk minimization Year: (2022)
Ref_id:b62 Title: A friendly tutorial on mean-field spin glass techniques for non-physicists Year: (2024)
Ref_id:b63 Title: A new vector partition of the probability score Year: (1973)
Ref_id:b64 Title: Reliability of subjective probability forecasts of precipitation and temperature Year: (1977)
Ref_id:b65 Title: Posterior calibration and exploratory analysis for natural language processing models Year: (2015)
Ref_id:b66 Title: Optimal ridge regularization for out-ofdistribution prediction Year: (2024)
Ref_id:b67 Title: Scikit-learn: Machine learning in python Year: (2011)
Ref_id:b68 Title: Using platt's scaling for calibration after undersampling-limitations and how to address them Year: (2024)
Ref_id:b69 Title: Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods Year: (1999)
Ref_id:b70 Title: Communities and crime [dataset] Year: (2002)
Ref_id:b71 Title: Sentence-BERT: Sentence embeddings using siamese BERT-networks Year: (2019)
Ref_id:b72 Title: Convex analysis Year: (2015)
Ref_id:b73 Title: Sample complexity of uniform convergence for multicalibration Year: (2020)
Ref_id:b74 Title: Generalization error of min-norm interpolators in transfer learning Year: (2024)
Ref_id:b75 Title: Hede: Heritability estimation in high dimensions by ensembling debiased estimators Year: (2024)
Ref_id:b76 Title: Minimum-risk recalibration of classifiers Year: (2024)
Ref_id:b77 Title: A modern maximum-likelihood theory for highdimensional logistic regression Year: (2019)
Ref_id:b78 Title: The likelihood ratio test in high-dimensional logistic regression is asymptotically a rescaled chi-square. Probability theory and related fields Year: (2019)
Ref_id:b79 Title: The gaussian min-max theorem in the presence of convexity Year: (2014)
Ref_id:b80 Title: Hydra: Preserving ensemble diversity for model distillation Year: (2020)
Ref_id:b81 Title: Fundamental computational limits of weak learnability in high-dimensional multi-index models Year: (2024)
Ref_id:b82 Title:  Year: (2000)
Ref_id:b83 Title: Which bridge estimator is the best for variable selection? Year: (2020)
Ref_id:b84 Title: Deep self-attention distillation for task-agnostic compression of pre-trained transformers Year: (2020)
Ref_id:b85 Title: Batchensemble: an alternative approach to efficient ensemble and lifelong learning Year: (2020)
Ref_id:b86 Title: MoleculeNet: A benchmark for molecular machine learning Year: (2018)
Ref_id:b87 Title: Estimating high-dimensional non-gaussian multiple index models via stein's lemma Year: (2017)
Ref_id:b88 Title: Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers Year: (2001)
Ref_id:b89 Title: Transforming classifier scores into accurate multiclass probability estimates Year: (2002)
Ref_id:b90 Title: Statistical physics of inference: Thresholds and algorithms Year: (2016)
Ref_id:b91 Title: Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: LLM is only used for minor grammar editing. Guidelines: • The answer NA means that the core method development Year: (2022)
Ref_id:b92 Title: LLM) for what should or should not be described Year: ()
