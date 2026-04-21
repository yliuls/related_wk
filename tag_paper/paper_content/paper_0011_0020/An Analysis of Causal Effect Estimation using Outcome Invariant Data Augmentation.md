Title: An Analysis of Causal Effect Estimation using Outcome Invariant Data Augmentation
Abstract: The technique of data augmentation (DA) is often used in machine learning for regularization purposes to better generalize under i.i.d. settings. In this work, we present a unifying framework with topics in causal inference to make a case for the use of DA beyond just the i.i.d. setting, but for generalization across interventions as well. Specifically, we argue that when the outcome generating mechanism is invariant to our choice of DA, then such augmentations can effectively be thought of as interventions on the treatment generating mechanism itself. This can potentially help to reduce bias in causal effect estimation arising from hidden confounders. In the presence of such unobserved confounding we typically make use of instrumental variables (IVs)-sources of treatment randomization that are conditionally independent of the outcome. However, IVs may not be as readily available as DA for many applications, which is the main motivation behind this work. By appropriately regularizing IV based estimators, we introduce the concept of IV-like (IVL) regression for mitigating confounding bias and improving predictive performance across interventions even when certain IV properties are relaxed. Finally, we cast parameterized DA as an IVL regression problem and show that when used in composition can simulate a worst-case application of such DA, further improving performance on causal estimation and generalization tasks beyond what simple DA may offer. This is shown both theoretically for the population case and via simulation experiments for the finite sample case using a simple linear example. We also present real data experiments to support our case.

Section: Introduction
A classical problem in machine learning is that of regression-using i.i.d. samples from some fixed, unknown distribution P X,Y , we predict outcome Y values for unlabeled treatment X values. The use of regularization techniques is crucial for this task to achieve good generalization from training to test data [1]. Data augmentation (DA) [2,3] is one such method, where each sample is randomly perturbed multiple times to grow the dataset size. However, these regression models cannot generally be interpreted causally as the statistical relationship between X and Y may arise from shared common causes, known as confounders, rather than from X influencing Y . Removing such confounders requires independently assigning values of X during data generation, known as an intervention [4,5].
Unfortunately, we seldom have access to the data generation process to be able to intervene on variables. A common workaround is to use auxiliary variables to correct for unobserved confounders [6][7][8]. One such approach is that of instrumental variables (IVs) that represent certain conditional independences in the system which can be used to identify the causal effect of X on Y [9][10][11]. Alas, IVs too are generally hard to find in many popular applications such as computer vision and natural language processing, motivating the need for more accessible ways to mitigate unobserved confounding.
Recent work therefore seeks to leverage more commonly available auxiliary variables to reduce confounding-induced bias even when the causal effect itself cannot be identified [12][13][14][15]. Collectively referred to as causal regularization, these methods aim to learn predictors that generalize out-ofdistribution (OOD) by discouraging reliance on spurious (i.e., non-causal,) correlations. Since distribution shifts often correspond to interventions on parts of the data-generating process [16,4], models that fail under such shifts typically do so because they exploit confounded relationships [17]. Tackling this root cause directly, causal regularization offers a principled approach for more robust prediction.
In the same vein, more ambitious works have also explored the use of common regularization techniques, such as ℓ 1 , ℓ 2 [18] and the min-norm interpolator [19], for the same purpose of causal regularization. This is in contrast to the canonical use of such regularizers for estimation variance reduction and i.i.d. prediction generalization [1]. Other popular regularization methods, however, remain understudied in a similar context of un-identifiable causal effect estimation, motivating our work.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b3', 'b16', 'b17', 'b18', 'b0']

Section: Our contributions.
To this end, we provide a first analysis of DA for estimating un-identifiable causal effects using only observational data for (X, Y ). Our contributions, summarized in Tab. 1, include: (i) DA as a soft intervention (Sec. 4.1): We show that DA can synthesize treatment interventions when the outcome function is invariant to DA, lowering bias in causal effect estimates when the intervention acts along spurious features. (ii) Introducing IV-like regression (Sec. 3): Relaxing the properties of IVs, we introduce the concept of IV-like (IVL) variables. This generalization renders IV regression ineffective at identifying causal effects, but when regularized appropriately via our proposed IVL regression, may still reduce confounding bias and improve prediction generalization across treatment interventions. (iii) DA parameters as IVL (Sec. 4.2): By casting parameterized DA as IVL, we show that its composition DA+IVL with IVL regression further reduces confounding bias beyond just simple DA by essentially simulating a worst-case or adversarial application of the DA.
We validate our approach with theoretical results in a linear setting for the infinite-sample case, and simulation and real-data experiments in the finite-sample case.
this section cite: []

Section: Preliminaries
Consider treatment X and outcome Y taking values in X ⊆ R m and Y ⊆ R l respectively. Given the set of functions H := {h : X → Y}, the canonical setting described in the literature [4,15,20] deals with estimating the function f ∈ H in the structural equation model (SEM) M of the following form 1X = τ (Y, Z, C, N X ), Y = f (X) + ϵ(C) + N Y , (1) where Z, C, N X , N Y are exogenous (and therefore mutually independent) random variables and the residual ξ := Yf (X) = ϵ(C) + N Y is assumed to be zero mean, i.e. E M [ξ] = 0. Since M is potentially cyclic, a priori it may entail several or no distributions at all. However, here we make the assumption that for all (x 0 , y 0 ) ∈ X × Y the unique limits
x := lim t→∞ x t = lim t→∞ τ (y t-1 , z, c, n X ), y := lim t→∞ y t = lim t→∞ f (x t-1 ) + ϵ(c) + n Y exist for any (z, c, n X , n Y ) ∼ P M Z,C,N X ,N Y , meaning that the unique distribution entailed by M is in this equilibrium state. Of course, if M is acyclic, these limits always exist. Note that assuming the existence of such an equilibrium does not violate the classic independent causal mechanism (ICM) principle [4]; we defer interested readers to Appendix B for further details on cyclic SEMs and the ICM.
Given a proper convex loss ℓ : R l × R l → R + , empirical risk minimization (ERM) uses a dataset D := {(x i , y i )} n i=0 of n samples from M to minimize an empirical version of the statistical risk R M ERM (h) := E M [ℓ(Y, h(X))],
over h ∈ H. However, since the residual ξ in Eq. ( 1) is generally correlated with X, i.e., E M [ξ | X] ̸ = 0, the ERM minimizer ĥM ERM typically yields a biased estimate of f [5,4]. This bias arises due to the exclusion of the (unobserved) common parent C of X and Y , i.e. a confounder, in the ERM objective (hence fittingly called the omitted-variable bias [21]) and/or the model is cyclic (simultaneity bias [20,22], or reverse causality [5] in the degenerate case). For simplicity we shall refer to either case by saying that X and Y are confounded and the resulting bias as the confounding bias [5]. 2   IV regression simulates such an intervention using only observational data.
this section cite: ['b3', 'b14', 'b19', 'b3', 'b4', 'b3', 'b20', 'b19', 'b21', 'b4', 'b4']

Section: Intervention for causal effect estimation
We can make X and the residual ξ uncorrelated via an interventionfoot_2 do(X := X ′ ), where we explicitly set X to some independently sampled X ′ in Eq. ( 1) irrespective of its parents, resulting now in the new SEM M; do(X := X ′ ) or M; do(X) as a shorthand for when X ′ ∼ P M X . The distribution induced by this modified SEM is called an interventional distribution (with respect to M) under which the ERM objective from Eq. ( 2) now defines the following causal risk (CR) [12,19,24] as
R M CR (h) := R M;do(X) ERM (h) = R M;do(X:=X ′ ) ERM (h), s.t. X ′ ∼ P M X .(3)
Minimizing Eq. (3) is meaningful in two important cases where ERM fails: (i) Causal effect estimation: The minimizer ĥM CR of Eq. (3) gives us an unbiased estimate of the average treatment effect (ATE) [6] E M;do(X:=x) [Y | X = x] = f (x) that measures the causal influence of X on Y . (ii) Robust prediction: ATE based prediction of Y values for unlabeled X values is robust in the sense that it can generalize across arbitrary OOD treatment interventions or shifts in the treatment distribution [25]. Consequently, the causal risk minimizer ĥM CR is also a robust predictor over the support of P M X . Specifically, ĥM CR minimizes the worst-case ERM objective over the set P of all possible intervention distributions P X ′ over the support of P M X [25], i.e. for P :=
P X ′ supp(P X ′ ) ⊆ supp P M X , ĥM CR ∈ argmin h∈H max P X ′ ∈P R M;do(X:=X ′ ) ERM (h).
To better isolate the estimation error due to confounding, we define the causal excess risk (CER) [19] CER M (h) := R M CR (h) -R M CR (f ). This removes the irreducible noise from Eq.  (3), we usually rely on observational data/ distribution and additional variables to approximate them, as outlined in the next section.
this section cite: ['b11', 'b18', 'b23', 'b24', 'b24', 'b18']

Section: Instrumental variable regression
One way to get an unbiased estimate of f from the observational distribution of M is to use socalled instrumental variables Z with the properties [5,4,10,9,26] of: (i) Treatment Relevance: Z ̸⊥ ⊥ X. (ii) Exclusion Restriction: Z enters Y only through X, i.e. Z ⊥ ⊥ Y M;do(X:=x) . 4 (iii) Unconfoundedness: Z ⊥ ⊥ ξ. (iv) Outcome Relevance: Z carries information about Y , i.e. Y ̸⊥ ⊥ Z. (1) on Z and using E[ξ | Z] = E[ξ] = 0 from the unconfoundedness property gives
this section cite: ['b4', 'b3', 'b9', 'b8', 'b25']

Section: Conditioning Eq
E M [Y | Z] = E M [f (X) | Z].
(4) IV regression therefore entails solving Eq. ( 4) for f , which can be done by minimizing the risk [26] R
M IV (h) := E M ℓ Y, E M [h(X) | Z] .(5)
For linear f (•) := f ⊤ (•), h(•) := h ⊤ (•) with f , h ∈ R m and squared loss ℓ(y, y ′ ) := ∥yy ′ ∥ 2 , this gives the two-stage-least-squares (2SLS) [27] solution where the first stage regresses X from Z, and the second stage regresses Y from predictions E[X | Z] of the first stage to get the estimate ĥM IV .
this section cite: ['b25', 'b26']

Section: Data augmentation
In this work we restrict ourselves to data augmentation with respect to which f is invariant [3,28].
The action of a group G is a mapping δ : X × G → X which is compatible with the group operation.
For convenience we shall write gx := δ(x, g). We say that
f is invariant under G (or G-invariant) if f (gx) = f (x), ∀ (g, x) ∈ G × X .
Less formally, we say that the map gx, henceforth assumed to be continuous in x, is a valid outcomeinvariant DA transformation parameterized by the vector g ∈ G. Let G have a (unique) normalized Haar measure and P G be the corresponding distribution defined over it. For some G ∼ P G , the canonical application of DA seeks to minimize an empirical version of the following risk.
R M DA G +ERM (h) := E M [ℓ(Y, h(GX))].(6
) Note that it is sufficient to have some prior information about the symmetries of f in order to be able to construct such a DA. For example, when classifying images of cats and dogs we already know that whatever the true labeling function may be, it would certainly be invariant to rotations on the images. G would then represent the random rotation angle, whereas Gx would be the rotated image x.
We wish to contrast the use of DA in this work with the canonical setting-to mitigate overfitting, DA is used to grow the sample size by generating multiple augmentations (Gx, y) for each data sample (x, y) ∼ P M X,Y [3,28,29]. Such regularization, overfitting mitigation, estimation variance reduction, or i.i.d. prediction generalization is not the focus of this work and we intentionally provide Eq. ( 6) along with theoretical results that follow in the population case to emphasize that DA is not being used as a conventional regularizer. Instead, our goal is to improve causal effect estimation and robust prediction by re-purposing DA to mitigate hidden confounding bias in the data.
this section cite: ['b2', 'b27', 'b2', 'b27', 'b28']

Section: Faithfulness and Outcome Relevance in IVs
The distribution P M X,Y,Z,C is faithful to the graph of M if it only exhibits independences implied by the graph [4,30]. 5 This standard assumption in IV settings renders outcome-relevance implicit and therefore rarely mentioned. In this section we discuss the case where only the first three IV properties are satisfied, i.e. outcome-relevance may not hold. Since such a Z may not be a valid IV, therefore identifiability of ATE is not possible in general as the problem in Eq. ( 4) can now be misspecified, having multiple, potentially infinitely many solutions when Y ⊥ ⊥ Z. Nevertheless, we shall refer to such a Z as IV-like (IVL) to emphasize that while Z may not be an IV, it may still be 'instrumental' for reducing confounding bias when estimating the ATE compared to the standard ERM baseline.
this section cite: ['b3', 'b29']

Section: ERM regularized IV regression.
Despite problem misspecification for a IVL Z, the target function f remains a minimizer for the IV risk in Eq. ( 5). Albeit, potentially not unique-for example, a linear h with squared loss leads to an under-determined problem in Eq. (5). We therefore propose the following regularized version of the IV risk for such an IVL setting,
R M IVL α (h) := R M IV (h) + αR M ERM (h),(7)
where α > 0 is the regularization parameter. The ERM risk as a penalty allows our estimations to have good predictive performance while the IV risk encourages solution search within the subspace where we know f to be present. We refer to minimizing the risk in Eq. ( 7) as IVL regression.
Note that the motivation behind IVL regression is not the identifiability of f , but rather potentially better estimations of f with lower confounding bias. The next section provides a concrete example. The observational distribution of (GX, Y, G, C) and (X, Y, G, C) for graphs (a) and (b) respectively are the same. The former applies DA on X, whereas the later applies a (soft) intervention on X. Furthermore, for the graph in (b), G is IVL. Example 1 (a linear Gaussian IVL example). For scalar σ > 0, non-zero matrices Γ, T ∈ R * ×m and vectors
τ ⊤ , f , ϵ ∈ R m such that f ⊤ τ ⊤ ̸ = 1 so that the following SEM M is solvable in (X, Y ) 6 X = τ ⊤ Y + Γ ⊤ Z + T ⊤ C + σN X , Y = f ⊤ X + ϵ ⊤ C + σN Y ,
where Z, C, N X , N Y are conformable, centered Gaussian random vectors and Z is IVL w.r.t. (X, Y ). 7Now, the task is to improve our estimation of f compared to standard ERM. We evaluate an estimate ĥD using the CER, which for a squared loss and covariance Σ M X in Example 1 simply comes out to be
CER M ĥD = ĥD -f 2 Σ M X .(8)
Prior works use this form to quantify the error in ATE estimation [19,12] or measure some notion of strength of confounding [18,31,24]. Similarly, we use it to measure confounding bias of population estimates ĥM (Appendix A) and estimation error in finite sample experiments. The next results follow.
this section cite: ['b4', 'b18', 'b11', 'b17', 'b30', 'b23']

Section: Theorem 1 (robust prediction with IVL regression).
For SEM M in Example 1, the following holds:
ĥM IVL α ∈ argmin h max ζ∈Pα R M;do(Γ ⊤ (•):=ζ) ERM (h), s.t. P α := ζ ζζ ⊤ ≼ 1 α + 1 Γ ⊤ Σ M Z Γ .
Proof. See Appendix F.3 for the proof. Theorem 2 (causal estimation with IVL regression). In SEM M of Example 1, for α < ∞, we have
CER M ĥM IVL α ≤ CER M ĥM ERM , equality iff E M [X | Z] ⊥ a.s. E M [X | ξ].
Proof. See Appendix F.4 for the proof.
Theorem 1 shows that IVL regression achieves optimal predictive performance across treatment interventions within the perturbation set P α defined by α. Theorem 2 further states that this strictly reduces confounding bias in ATE estimates iff the perturbations align with spurious features of X, as indicated by the equality condition (also necessary for identifiability in linear IV settings [32,25]).
this section cite: ['b31', 'b24']

Section: Causal Effect Estimation using Data Augmentation
We dedicate this section to the main topic and point of this work-discussing the potential of data augmentation for improving predictive performance across interventions and reducing confounding bias in ATE estimates. To that effect, for the rest of this work we shall consider the following SEM A X = τ (Y, C, N X ), Y = f (X) + ϵ(C) + N Y , (9) which is assumed to have a unique stationary distribution with exogenous C, N X , N Y and the residual ξ := Yf (X) is zero-mean, i.e. E[ξ] = 0. We also have access to DA transformations GX of X parameterized by G ∼ P A G such as described in Sec. 2.3. Figure 2a shows the graph of A post DA. Given samples for only (X, Y ) and some valid DA parameterized by G, the task is to improve predictive performance across interventions and reduce confounding bias in ATE estimates. We now make two observations in the following sections and state the respective results that follow thereof.
this section cite: []

Section: Data augmentation as a soft intervention
Consider a (soft) intervention on A where we substitute the mechanism τ of X with Gτ . With some abuse of notation, we shall represent this SEM by A; do(τ := Gτ ) the graph of which is shown in Fig. 2b. Note that this SEM also has a unique stationary distribution (proof in Appendix F.2). Comparing the DA mechanism in A (Fig. 2a) and the intervention A; do(τ := Gτ ) (Fig. 2b), we see:
this section cite: []

Section: Observation 1 (soft intervention with DA). Distributions P A
GX,Y,G,C and P A;do(τ :=Gτ ) X,Y,G,C are identical.
We can hence treat samples generated from A via DA as if they were instead generated from A; do(τ := Gτ ) by intervening on X. This allows us to re-write the DA+ERM risk from Eq. ( 6) as,
R A DA G +ERM (h) = R
A;do(τ :=Gτ ) ERM (h), to emphasize that DA is equivalent to a (soft) intervention and as such can be used to reduce confounding bias when estimating f , as we will show with the following example. Example 2 (a linear Gaussian DA example). For scalars κ, σ > 0, non-zero matrices Γ, T ∈ R * ×m and vectors
τ ⊤ , f , ϵ ∈ R m such that f ⊤ τ ⊤ ̸ = κ -1 so that the following SEM A is solvable in (X, Y ) X = κ • τ ⊤ Y + T ⊤ C + σN X , Y = f ⊤ X + κ • ϵ ⊤ C + σN Y , GX := X + γ • Γ ⊤ G,
where G, C, N X , N Y are conformable, centered Gaussian random vectors, κ determines how much (X, Y ) are confounded and range Γ ⊤ ⊆ null f ⊤ so that GX is a valid outcome invariant DA transformation of X parameterized by G with strength γ > 0. This transformation can be viewed as translating X along its level-set as shown in Fig. 3 and represents our prior knowledge about the symmetries of f for the purposes of this example. Theorem 3 (causal estimation with DA+ERM). For SEM A in Example 2, the following holds:
CER A ĥA DA G +ERM ≤ CER A ĥA ERM , equality iff E A [GX | G] ⊥ a.s. E A [X | ξ].
Proof. See Appendix F.5 for the proof.
That is, DA strictly reduces confounding bias in ATE estimate iff the induced intervention perturbs X along spurious features. Importantly, Theorem 3 suggests that lower confounding bias is not a 'free lunch' with outcome invariance of DA and practitioners may need domain knowledge to construct DA that targets spurious features. Fortunately however, Theorem 3 also suggests that with outcome invariance, DA should not perform worse than ERM. We say that DA+ERM dominates ERM on causal estimation [33, p. 48]. Practitioners may therefore be advised to generously use such DA, as it achieves regularization in the worst case, and mitigates confounding bias as a 'bonus' in the best case.
this section cite: []

Section: Worst-case data augmentation with IVL regression
We once again point our attention to the graph of A; do(τ := Gτ ) from Fig. 2b to observe that: Observation 2 (IV-like DA parameters). In SEM A; do(τ := Gτ ), the DA parameters G are IVL.
In light of this we can now re-write the IV and IVL risks for A; do(τ := Gτ ) to respectively read
R A DA G +IV (h) = R A;do(τ :=Gτ ) IV (h), R A DA G +IVL α (h) = R A;do(τ :=Gτ ) IVL α (h).
this section cite: []

Section: Corollary 1 (worst-case DA with DA+IVL regression).
For SEM A in Example 2, it holds that
ĥA DA G +IVL α ∈ argmin h max g∈Gα R A DAg+ERM (h), s.t. G α := g Γ ⊤ gg ⊤ Γ ≼ 1 α + 1 Γ ⊤ Σ A G Γ .
Proof. The result follows from Observation 1, Observation 2 and Theorem 1. Corollary 2 (causal estimation with DA+IVL regression). For α, γ < ∞ in SEM A from Example 2,
CER A ĥA DA G +IVL α ≤ CER A ĥA DA G +ERM , equality iff E A [GX | G] ⊥ a.s. E A [X | ξ]. Proof.
The result follows directly from Theorem 2 and Observation 2.
Using DA parameters as IVL therefore simulates a worst-case, or adversarial application of DA within a set of transforms G α . Of course Corollary 1 can also be viewed as a predictor that generalizes to treatment interventions encoded by G α . As is intuitive, such a worst-case intervention improves our ATE estimation so long as the features of X intervened along include some that are spurious (Corollary 2). DA and IVL regression may therefore be used in composition if the application can benefit from regularization and/ or better prediction generalization across DA-induced interventions, with a 'bonus' of lower confounding bias if the DA also augments any spurious features of X.
0.0 0.2 0.4 0.6 0.8 1.0 κ 0.0 0.1 0.2 0.3 0.4 0.5 nCER (a) γ = 1, κ ∈ [0, 1]
this section cite: []

Section: Related Work
Causal regularization is perhaps the most appropriate classification for this work. These methods aim for more robust prediction by mitigating the upstream problem of confounding bias in a more accessible way than is required for full identification. This is done, for example, by relaxing properties of auxiliary variables [12][13][14][15], as we have done via our IVL approach. Most relevant, however, are methods that re-purpose common regularizers, canonically used for estimation variance reduction and i.i.d. prediction generalization, for confounding bias mitigation. Of note is [18], where a certain linear modeling assumption allows the estimation of ∥f ∥ 2 from observational (X, Y ) data, which is then used to develop a cross-validation scheme for ℓ 1 , ℓ 2 regularization. [19] conducted a similar theoretical analysis for the min-norm interpolator. To the best of our knowledge, we are the first to study the same for DA-re-purposing yet another ubiquitous regularizer to mitigate confounding bias. [34] methods aim for prediction generalization to unseen test domains via robust optimization (RO) [35] over a perturbation set P of possible test domains ρ ∈ P as
this section cite: ['b11', 'b12', 'b13', 'b14', 'b17', 'b18', 'b33', 'b34']

Section: Domain generalization (DG)
R P RO (h) := max ρ∈P R ρ ERM (h),
Since generalizing to arbitrary test domains is impossible, the choice of perturbation set encodes one's assumptions about which test domains might be encountered. Instead of making such assumptions a priori, it is often assumed to have access to data from multiple training domains which can inform one's choice of perturbation set. This setting is explored in group distributionally robust optimization (DRO) [36]. Variations have been used to mitigate confounding bias and subsequently generalize to treatment interventions when used with interventional data [16,37], confounder information (i.e. entire graph) [38][39][40] or some proxy thereof in the form of environments [41][42][43]38]. We, however, do not assume access to any of these and instead synthesize interventions via DA.
Counterfactual DA strategies have been the primary lens for causal analyses of DA [44][45][46][47][48][49][50]. These aim for prediction robustness to treatment interventions via DA simulated counterfactuals. 8 As with counterfactual reasoning more broadly, this requires strong assumptions-such as access to the full SEM [45,46], auxiliary variables [44,46,49,50], or causal graphs [47,48]. By contrast, we show that outcome invariance of DA suffices for treatment intervention robustness without invoking counterfactuals. Moreover, prior work has largely overlooked causal effect estimation, often assuming reverse-causal settings where the ATE becomes trivial [44,46,45]. Ours is the first framework to study ATE estimation via DA with minimal structural assumptions.
Invariant prediction based methods aim to make predictions based on statistical relationships that remain stable across all domains in P. A common assumption, for instance, is that P Y |X is invariant across P, with only the marginal P X being allowed to vary. Invariance is also closely linked to causal discovery-following the classic ICM principle [4], causal mechanisms remain stable under interventions on inputs [25,17]. This connection has inspired approaches that enforce invariance conditions to recover causal structures [16,37]. IV regression can also be viewed as one such method, where the goal is to learn predictors whose residuals are invariant to the instruments [10,9,26,51,7]. More broadly, the principle of invariance, whether motivated by causality or otherwise, has proven useful for improving prediction generalization across heterogeneous settings [15,41,52,14,[53][54][55][56]34].
this section cite: ['b35', 'b15', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b37', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b44', 'b45', 'b43', 'b45', 'b48', 'b49', 'b46', 'b47', 'b43', 'b45', 'b44', 'b3', 'b24', 'b16', 'b15', 'b36', 'b9', 'b8', 'b25', 'b50', 'b6', 'b14', 'b40', 'b51', 'b13', 'b52', 'b53', 'b54', 'b55', 'b33']

Section: Experiments
We began by presenting results in the infinite-sample setting to emphasize that mitigating confounding bias is fundamentally not a sample size issue, i.e., not solvable through traditional regularization alone. In this section, we turn to the finite-sample regime and empirically evaluate the effectiveness of DA in reducing hidden confounding bias. Importantly, we do not use DA for its conventional purpose of augmenting data to improve i.i.d. generalization or reduce estimation variance. Throughout all experiments, we therefore fix the number of samples in the augmented dataset to match that of the original dataset since our focus lies squarely on robust prediction via confounding bias mitigation.
Finding baselines for evaluating our results is however a challenge-the problem of mitigating confounding bias given only observational (X, Y ) data and symmetry knowledge via DA is quite underexplored. Nevertheless, for the sake of completeness we make an effort to re-purpose existing methods from domain generalization, invariance learning and causal inference literature to be used as baselines. These methods often require access to additional variables (e.g. IVs, confounders, domains/environments, etc.), and to maintain fairness we will replace these with DA parameters G. Such a comparison is conceptually valid since by virtue of being DG methods, they are essentially solving a robust loss of a similar form as in Corollary 1, giving us meaningful baselines for DA+IVL.
In addition to standard ERM, DA and IV regression, our baselines include DRO [36], invariant risk minimization (IRM) [41], invariant causal prediction (ICP) [16], regularization with invariance on causal essential set (RICE) [56], variance risk extrapolation (V-REx) and minimax risk extrapolation (MM-REx) [38]. We also include the causal regularization method by Kania and Wit [12] and the ℓ 1 , ℓ 2 approaches by Janzing [18]. We discretize G if the method accepts only discrete variables. For IVL regression, we select the regularization parameter α in a variety of ways, including vanilla cross validation (CV), level-based CV (LCV) and confounder correction (CC) as described in Appendix D. Other implementation details are provided in Appendix E, and the code to reproduce our results is publicly released at https://github.com/uzairakbar/causal-data-augmentation.
To make CER based evaluation more interpretable for our experiments, we propose the normalization
nCER M (h) := CER M (h) CER M (h) + CER M (h 0 ) ∈ [0, 1], h 0 (•) := E M;do(X) [Y ],
where h 0 represents the null treatment effect, i.e. when X has no causal influence on Y , then
E M;do(X) [Y | X] = E M;do(X) [Y ].
The normalized CER (nCER) can be considered a generalization of the metrics used by [18,24,31] in linear settings and similarly has the interesting property that it is 0 for the ground-truth causal solution h = f ̸ = h 0 but 1 if there is pure confounding for h ̸ = f = h 0 . Janzing argues in [24,31] that using an Euclidean norm instead of the weighted norm in Eq. ( 8) is more relevant for causal settings, which also motivates our choice when evaluating results of the simulation and optical-device experiments described below. Conceptually, this is equivalent to evaluation based on the causal risk in Eq. ( 3) under the interventional distribution X ′ ∼ N N (0 m , I m ).
this section cite: ['b35', 'b40', 'b15', 'b55', 'b37', 'b11', 'b17', 'b17', 'b23', 'b30', 'b23', 'b30']

Section: Simulation experiment
For the finite sample results of the linear SEM A from Example 2, by taking m = 32, k = 31 (dimension of G), σ = 0.1 and fixing τ ⊤ = 0 m , 9 we sample a new f , ϵ and T ∈ R m×m from a standard normal distribution for each of the 32 experiments for every combination of κ and γ. Each time we construct a Γ := V 0 with k rows as orthonormal basis of null(f ), such that the SVD of f is
f = [u U 0 ] λ 0 1×(m-1) 0 (m-1)×1 0 (m-1)×(m-1) v ⊤ V ⊤ 0 .
Although this construction of Γ relies on direct knowledge of f , which is of course unavailable in practice, we include it here purely for illustrative purposes. We treat access to Γ as having prior knowledge about the structural symmetries of f , noting that this information alone is insufficient to recover f .
0.1 0.2 0.3 0.4 0.5 0.6 0.7 nCER ERM DA+ERM DA+IVL CV α DA+IVL LCV α DA+IVL CC α DA+IV IRM ICP DRO RICE V-REx MM-REx ℓ1 Janzing '19 ℓ2 Janzing '19 Kania, Wit '23 Simulation Data 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 nCER ERM DA+ERM DA+IVL CV α DA+IVL LCV α DA+IVL CC α DA+IV IRM ICP DRO RICE V-REx MM-REx ℓ1 Janzing '19 ℓ2 Janzing '19 Kania, Wit '23 Optical Device Data 0.30 0.35 0.40 0.45 0.50 nCER ERM DA+ERM DA+IVL CV α DA+IVL LCV α DA+IVL CC α DA+IV IRM ICP DRO RICE V-REx MM-REx ℓ1 Janzing '19 ℓ2 Janzing '19 Kania, Wit '23 Colored MNIST Data We then generate n = 2048 samples of (X, Y ) for each experiment. For ERM we use a closed form linear OLS solution. For DA+IV, we make use of linear 2SLS. Finally, DA+IVL α was implemented using a closed form linear OLS solution between empirical versions (see Proposition 1) of
X ′ := √ αX + √ 1 + α - √ α E[X | Z], Y ′ := √ αY + √ 1 + α - √ α E[Y | Z].
Our first experimental result in Fig. 4a compares the different estimation methods across varying levels of confounding κ ∈ [0, 1]. As expected, ERM performance degrades with increasing confounding. Applying DA alone already brings us closer to the causal solution, while DA+IVL achieves even better performance. DA+IV regression is unstable and generally performs poorly as it is under-determined.
Next, we fix the confounding and DA strengths at κ = γ = 1, and sweep over the regularization parameter α ∈ [10 -5 , 10 5 ] for DA+IVL α . Figure 4b shows that optimal performance is achieved for intermediate values of α, confirming that arbitrarily small values of α, while beneficial in the theoretical population setting (as suggested by Eq. ( 27) in the proof of Theorem 2), are suboptimal for finite samples. 10 We also find that both CV and CC strategies effectively select reasonable values of α.
Lastly, Fig. 4c examines sensitivity to the DA strength γ ∈ [10 -2.5 , 10], for fixed confounding strength κ = 1. As expected, stronger DA results in stronger interventions on X, which improves causal effect estimation. However, we also observe diminishing returns; when the variation induced by DA is either too small or too large, DA+IVL α does not yield significant improvements over the DA+ERM baseline.
For completeness, we also benchmark our approach against other baseline methods on 16 distinct simulation SEMs with 2048 samples each. Aggregated results are presented in Fig. 5 (left most).
this section cite: []

Section: Real data experiments
Optical device dataset. The dataset from [24] consists of 3 × 3 pixel images X displayed on a laptop screen that cause voltage readings Y across a photo-diode. A hidden confounder C controls two LEDs; one affects the webcam capturing X, the other affects the photo-diode measuring Y . The ground-truth predictor f is computed by first regressing Y on (ϕ(X), C), where ϕ(X) are polynomial features of X with degree d ∈ {1, • • • , 5} that best explains the data (degree 2 in most cases). The component corresponding to C is then removed to recover f . We add Gaussian noise G ∼ N N (0, Σ X /10) for DA and fit the methods from Sec. 6.1 on features ϕ(GX) for n = 1000 samples across 12 datasets. Note that using the same ground-truth polynomial degree for ϕ during evaluation is important here so as to avoid introducing statistical bias from model-misspecification as our analysis squarely focuses on confounding bias. Figure 5 (middle) shows the results, where DA+ERM improves over ERM, and DA+IVL performs even better, outperforming other baselines.
Colored MNIST. We evaluate on the colored MNIST dataset [41], where labels are spuriously correlated with image color during training, but this correlation is flipped at test time. We use the same neural architecture and parameters as [41] across all baselines, training with the IV-based objective described in the Appendix C. DA is implemented via small perturbations to hue, brightness, contrast, saturation, and translation, each parameterized by G ∼ β β(2, 2). Although these do not directly manipulate color, the actual spurious feature, they still help reduce confounding. Results in Fig. 5 (rightmost) show that ERM underperforms, DA+ERM provides substantial gains, and DA+IVL α performs competitively with the best DG baselines, with DA+IVL CV α achieving the best overall performance. Interested readers may also visit Appendix E.3, where we clarify the connection of the colored MNIST model with the cyclic SEM from Eq. ( 9).
this section cite: ['b23', 'b40', 'b40']

Section: Limitations
Necessity and practicality of prior knowledge. As discussed in Sec. 4, outcome invariance alone does not suffice to lower confounding bias and practitioners may need domain knowledge to construct DA that targets spurious features as well. Alternatively, one can also take a 'carpet bombing' approach by exhausting all available outcome invariant DA in hope that some may align with spurious features. Nevertheless, under outcome invariance, our methods should perform no worse than standard ERM. Fundamentally, causal estimation from purely observational data is impossible without untestable assumptions. For instance, the IV (or IVL) assumptions of un-confoundedness and exclusion restriction are inherently untestable and must be justified through domain knowledge. Moreover, the requirement of alignment with spurious features in Theorem 2 is not an artifact of our IVL relaxation-it is a rephrasing of the exclusion principle that underlies identifiability in IV regression. If an IV does not influence Y through the spurious features of X, the corresponding causal components of f cannot be identified [25]. IVLs, being relaxations of IVs, inherit these same untestable premises.
Viewed through the lens of IVs/IVLs (Observation 2), our assumptions on DA are arguably more modest than they may initially seem, especially since a symmetry-based DA model has well-established precedent in the literature [3,28,53,[57][58][59][60][61][62][63]. This correspondence can be summarized as follows: Finally, we recognize the hesitation in committing to strict notions of outcome invariance in practice and leave a more thorough exploration of approximate or even violated invariance to future work.
this section cite: ['b24', 'b2', 'b27', 'b52', 'b56', 'b57', 'b58', 'b59', 'b60', 'b61', 'b62']

Section: Choice of α.
Selecting the IVL regularization parameter α in finite-sample settings is not straightforward. As outlined in Appendix D, we propose several strategies that work well empirically, though some may appear less principled since α is tuned via cross-validation within the same distribution, even though the task concerns OOD generalization. This challenge is not unique to IVL, but rather a broader limitation common to DG methods [64].
this section cite: ['b63']

Section: Conclusion
We conclude that our proposed causal framework for data augmentation (DA) enables re-purposing the widely used i.i.d. generalization tool for OOD generalization across treatment interventions. By interpreting outcome-invariant DA as interventions and IV-like variables, our approach reduces confounding bias and consequently improves both causal effect estimation and robust prediction.
this section cite: []

Section: References
Ref_id:b0 Title: Vladimir Naumovich Vapnik. Statistical learning theory Year: (1998)
Ref_id:b1 Title: A survey on image data augmentation for deep learning Year: (2019)
Ref_id:b2 Title: On the benefits of invariance in neural networks Year: (2020)
Ref_id:b3 Title: Elements of causal inference: Foundations and learning algorithms Year: (2017)
Ref_id:b4 Title: Causality Year: (2009)
Ref_id:b5 Title: A neural mean embedding approach for back-door and front-door adjustment Year: (2022)
Ref_id:b6 Title: Spectral representation for causal estimation with hidden confounders Year: (2025)
Ref_id:b7 Title: Proximal causal learning with kernels: Two-stage estimation and moment restriction Year: (2021)
Ref_id:b8 Title: Kernel instrumental variable regression Year: (2019)
Ref_id:b9 Title: Instrumental variable regression via kernel maximum moment loss Year: ()
Ref_id:b10 Title: A class of algorithms for general instrumental variable models Year: (2020)
Ref_id:b11 Title: Causal regularization: On the trade-off between in-sample risk and out-ofsample risk guarantees Year: (2023)
Ref_id:b12 Title: Deconfounding and causal regularisation for stability and external validity Year: (2020)
Ref_id:b13 Title: Regularizing towards causal invariance: Linear models with proxies Year: (2021)
Ref_id:b14 Title: Anchor regression: Heterogeneous data meet causality Year: (2021)
Ref_id:b15 Title: Causal inference by using invariant prediction: Identification and confidence intervals Year: (2016)
Ref_id:b16 Title: When shift happens -confounding is to blame Year: (2025)
Ref_id:b17 Title: Causal regularization Year: (2019)
Ref_id:b18 Title: Ulrike von Luxburg, and Debarghya Ghoshdastidar. Interpolation and regularization for causal learning Year: (2022)
Ref_id:b19 Title: Econometric analysis Year: (2003)
Ref_id:b20 Title: The Phantom Menace: Omitted variable bias in econometric research Year: (2005)
Ref_id:b21 Title: Simultaneous equation models and two-stage least squares Year: (1979)
Ref_id:b22 Title: Endogeneity in empirical corporate finance Year: (2013)
Ref_id:b23 Title: Detecting confounding in multivariate linear models via spectral analysis Year: (2018)
Ref_id:b24 Title: A causal framework for distribution generalization Year: (2022)
Ref_id:b25 Title: Dual instrumental variable regression Year: (2020)
Ref_id:b26 Title: Two-or three-stage least squares? Year: (1988)
Ref_id:b27 Title: A group-theoretic framework for data augmentation Year: (2020)
Ref_id:b28 Title: Adversarial appearance learning in augmented Cityscapes for pedestrian recognition in autonomous driving Year: (2020)
Ref_id:b29 Title: Probabilistic graphical models: principles and techniques Year: (2009)
Ref_id:b30 Title: Detecting non-causal artifacts in multivariate linear regression models Year: (2018)
Ref_id:b31 Title: Econometric Analysis of Cross Section and Panel Data Year: (2010)
Ref_id:b32 Title: Theory of Point Estimation Year: (1998)
Ref_id:b33 Title: Domain generalization via invariant feature representation Year: (2013)
Ref_id:b34 Title:  Year: (2009)
Ref_id:b35 Title: Distributionally robust neural networks Year: (2020)
Ref_id:b36 Title: Invariant causal prediction for nonlinear models Year: ()
Ref_id:b37 Title: Out-of-distribution generalization via risk extrapolation (REx) Year: (2021)
Ref_id:b38 Title: Invariant causal representation learning for out-of-distribution generalization Year: (2022)
Ref_id:b39 Title: Counterfactual cocycles: A framework for robust and coherent counterfactual transports Year: (2025)
Ref_id:b40 Title:  Year: (2019)
Ref_id:b41 Title: Characterizing vocal hyperfunction using ecological momentary assessment of relative fundamental frequency Year: (2024)
Ref_id:b42 Title: Reinforcement learning applied to the optimization of power delivery networks with multiple voltage domains Year: ()
Ref_id:b43 Title: Selecting data augmentation for simulating interventions Year: (2021)
Ref_id:b44 Title: Not Just Pretty Pictures: Toward interventional data augmentation using text-to-image generators Year: (2024)
Ref_id:b45 Title: Data augmentations for improved (large) language model generalization Year: (2023)
Ref_id:b46 Title: MoCoDA: Model-based counterfactual data augmentation Year: (2022)
Ref_id:b47 Title: Causal action influence aware counterfactual data augmentation Year: (2024)
Ref_id:b48 Title: Domain generalization using causal matching Year: (2021)
Ref_id:b49 Title: Cat Phuoc Le, and Vahid Tarokh. CATE estimation with potential outcome imputation from local regression Year: (2025)
Ref_id:b50 Title: Learning deep features in instrumental variable regression Year: (2021)
Ref_id:b51 Title: Learning from conditional distributions via dual embeddings Year: (2017)
Ref_id:b52 Title: Transformation-invariant learning and theoretical guarantees for OOD generalization Year: (2024)
Ref_id:b53 Title: Invariance-inducing regularization using worstcase transformations suffices to boost accuracy and spatial robustness Year: (2019)
Ref_id:b54 Title: DROID: Learning from offline heterogeneous demonstrations via reward-policy distillation Year: (2023)
Ref_id:b55 Title: Out-of-distribution generalization with causal invariant transformations Year: (2022)
Ref_id:b56 Title: A theory of PAC learnability under transformation invariances Year: (2022)
Ref_id:b57 Title: Manitest: Are classifiers really invariant? Year: (2015)
Ref_id:b58 Title: Lossy compression for lossless prediction Year: (2021)
Ref_id:b59 Title: Approximation-generalization trade-offs under (approximate) group equivariance Year: (2023)
Ref_id:b60 Title: Learning partial equivariances from data Year: (2022)
Ref_id:b61 Title: Understanding the generalization benefit of model invariance from a data perspective Year: (2021)
Ref_id:b62 Title: Understanding data augmentation for classification: When to warp? Year: (2016)
Ref_id:b63 Title: In search of lost domain generalization Year: (2021)
Ref_id:b64 Title: Bias-variance decompositions: The exclusive privilege of Bregman divergences Year: (2025)
Ref_id:b65 Title: Chain graph models and their causal interpretations Year: (2002)
Ref_id:b66 Title: Discovering cyclic causal models by independent components analysis Year: (2008)
Ref_id:b67 Title: Learning linear cyclic causal models with latent variables Year: (2012)
Ref_id:b68 Title: On causal discovery with cyclic additive noise models Year: (2011)
Ref_id:b69 Title: Foundations of structural causal models with cycles and latent variables Year: ()
Ref_id:b70 Title: The Cowles Commission's contributions to econometrics at Chicago, 1939-1955 Year: (1994)
Ref_id:b71 Title: The Cobweb theorem Year: ()
Ref_id:b72 Title: Rational expectations and the theory of price movements Year: (1961)
Ref_id:b73 Title: Three-stage least squares: Simultaneous estimation of simultaneous equations Year: (1962)
Ref_id:b74 Title: Generalized method of moments Year: (2003)
Ref_id:b75 Title: Deep generalized method of moments for instrumental variable analysis Year: (2019)
Ref_id:b76 Title: Adversarial generalized method of moments Year: (2018)
Ref_id:b77 Title:  Year: (1971)
Ref_id:b78 Title: Matrix Analysis Year: (1985)
Ref_id:b79 Title: Matrix Mathematics: Theory, Facts, and Formulas Year: (2009)
