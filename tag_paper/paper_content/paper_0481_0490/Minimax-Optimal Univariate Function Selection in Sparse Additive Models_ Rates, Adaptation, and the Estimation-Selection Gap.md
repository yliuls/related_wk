Title: Minimax-Optimal Univariate Function Selection in Sparse Additive Models: Rates, Adaptation, and the Estimation-Selection Gap
Abstract: The sparse additive model (SpAM) offers a trade-off between interpretability and flexibility, and hence is a powerful model for high-dimensional research. This paper focuses on the variable selection, i.e., the univariate function selection problem in SpAM. We establish the minimax separation rates from both the perspectives of sparse multiple testing (FDR + FNR control) and support recovery (wrong recovery probability control). We further study how adaptation to unknown smoothness affects the minimax separation rate, and propose an adaptive selection procedure. Finally, we discuss the difference between estimation and selection in SpAM: Procedures achieving optimal function estimation may fail to achieve optimal univariate function selection.

Section: Introduction
The Sparse Additive Model (SpAM) is a pivotal topic of recent statistical research [Ravikumar et al., 2009, Meier et al., 2009, Koltchinskii and Yuan, 2010, Raskutti et al., 2012, Dalalyan et al., 2014, Yuan and Zhou, 2016, Tyagi et al., 2016, Tan and Zhang, 2019, Haris et al., 2022]. It extends the generalized additive model [Hastie and Tibshirani, 1987], balancing interpretability and flexibility while avoiding the curse of dimensionality and adapting to high-dimensional settings.
In this paper, we focus on the variable selection, i.e., univariate function selection problem of the SpAM, which is a fundamental problem with broad implications in multi-channel detection [Ingster and Lepski, 2003], multi-task learning [Wang et al., 2020], sparse neural network [Xu et al., 2023], and so on. We consider a Gaussian white noise (GWN) model with p covariates x = (x 1 , • • • , x p ) ∈ X p , which takes the form as
dY x = f (x)dx + σdB x = p j=1 f j (x j )dx j + σdB x ,
where X is the domain of each covariate x j , B x is a standard Wiener process on X p , and σ > 0 measures the intensity of the white noise. We assume f j is the univariate function corresponding to variable x j . Under the setting of sparsity, the response Y x is influenced by no more than s covariates, and hence f can be expressed as f (x) = j∈S f f j (x j ), where S f ⊆ {1, • • • , p} is the index set of these support covariates. In this continuous-time SpAM framework, our main goal is to recover the index set S f , i.e., to select which f j ̸ = 0. This paper studies the univariate function selection in SpAMs from two perspectives-namely, as a sparse multiple testing problem and as a support recovery problem. We employ the truncated procedures and establish the non-asymptotic minimax separation rates, delivering, to our knowledge, the first optimal finite-sample guarantees for univariate function selection in SpAMs.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12']

Section: Related work
Background of variable selection The variable selection problem has attracted significant interest recently [Butucea et al., 2018, Rabinovich et al., 2020, Belitser and Nurushev, 2022, Song and Cheng, 2023, Butucea et al., 2023a, Abraham et al., 2024]. The general assumption is that the response depends on only a few covariates, and the main aim is to find them. This problem can be framed as either a sparse multiple testing problem (controlling False Discovery Rate (FDR), False Negative Rate (FNR), etc.) [Rabinovich et al., 2020, Song and Cheng, 2023, Abraham et al., 2024], or a support recovery problem (controlling Hamming loss) [Wainwright, 2007, Butucea et al., 2018, Gao and Stoev, 2020, Butucea et al., 2023a], based on different setting of the loss function. Much of the existing studies concentrated on the sparse sequence model X i = β i +ϵ i , i = 1, • • • , p independently, with assuming p i=1 1(β i ̸ = 0) ≤ s and each ϵ i drawn from distributions like Gaussian [Butucea et al., 2018, Song andCheng, 2023] or generalized Gaussian [Gao and Stoev, 2020, Rabinovich et al., 2020, Abraham et al., 2024]. Though these studies demonstrated interesting phase-transition phenomena and established the asymptotically sharp minimax separation rate, they cannot be directly applied to the univariate function selection in SpAM.
Background of univariate function selection in SpAM Univariate function selection in SpAM stands as a pivotal problem in statistical learning [Lin and Zhang, 2006, Ravikumar et al., 2009, Huang et al., 2010, Chouldechova and Hastie, 2015, Xu et al., 2016, Wood et al., 2015, Butucea and Stepanova, 2017, Dai et al., 2023]. Most existing studies firstly provided minimax-optimal estimators for the function f via M-estimation with group-lasso-type penalties on each f j . Then, by utilizing the estimation results, the selection performances were often established as by-products [Ravikumar et al., 2009, Huang et al., 2010, Dai et al., 2023]. Although these methods ensured asymptotic variable selection consistency [Ravikumar et al., 2009, Huang et al., 2010] or FDR control [Dai et al., 2023], they did not guarantee minimax optimality for the univariate function selection problem. This implies that their minimum signal conditions, typically quantified by "min j∈S f ∥f j ∥ 2 2 ≥ some rate", are sufficient but not necessary: Their signal strength assumptions may be overly restrictive.
Existing optimal univariate function selection in SpAM From the viewpoint of support recovery with Hamming loss, Ingster and Stepanova [2014] and Butucea and Stepanova [2017] provided the minimax optimal (i.e., necessary and sufficient) signal condition for exact support recovery and almost-full support recovery, respectively. Comminges and Dalalyan [2012] analyzed support recovery in a p-dimensional nonparametric regression with an intrinsic s-variate underlying function. In an additive model allowing k-dimensional interaction effects, Stepanova and Turcicova [2025a,b] provided the optimal signal condition for exact support recovery. These studies offered asymptotically minimax optimal results in some specific function classes, but may not be persuasive in the general function space with a finite sample size. For instance, they rely on certain additional assumptions, like log p = o(σ -2/(2α+1) ) and σ → 0, in the Sobolev space with smoothness parameter α.
Inspiration from cutting-edge work Building on the monotone likelihood ratio property, Butucea et al. [2023a] recently established rate-optimal signal conditions for support recovery under group sparsity, improving upon conclusions from Lounici et al. [2011]. Kotekal and Gao [2024] extended the hard-thresholding estimator of Collier et al. [2017] to develop a minimax optimal goodness-of-fit test for SpAM (i.e., testing whether f = j∈S f f j = 0). These advances motivate the development of a non-asymptotic minimax optimal univariate function selector within a generalized SpAM framework, covering Sobolev-smooth, analytic, and other function classes.
this section cite: ['b13', 'b14', 'b15', 'b16', 'b18', 'b14', 'b16', 'b18', 'b19', 'b13', 'b20', 'b16', 'b20', 'b14', 'b18', 'b21', 'b0', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b0', 'b22', 'b27', 'b0', 'b22', 'b27', 'b28', 'b26', 'b29', 'b32', 'b33', 'b34']

Section: Main contributions and organization
This paper answers the following questions:
In a generalized SpAM framework, can we achieve non-asymptotic and minimax optimal univariate function selection? What is the difference between function estimation and univariate function selection?
The main contributions are threefold:
1. Minimax separation rates From both the viewpoints of sparse multiple testing (FDR+FNR control) and support recovery (wrong recovery probability control), we establish the non-asymptotic minimax separation rates for univariate function selection in a generalized SpAM framework. This result is, to our knowledge, the first optimal finite-sample guarantees. We also develop truncated-type selectors to achieve the minimax rate-optimality, respectively. 2. Minimax adaptation We provide a rate-optimal selection procedure that adapts to the smoothness parameter of the Sobolev spaces. We show that an additional log log(σ -2 ) term in the signal condition is required for this adaptation. 3. Difference between estimation and selection Within the class of truncated-type estimators, we demonstrate that the optimal function estimations can not yield optimal univariate function selection in some cases. This gap underscores the necessity to proceed differently in selection versus estimation, a finding with deep statistical implications.
The rest of the paper is organized as follows: Section 1 establishes the notation used throughout the paper. Section 2 introduces the model setup and the background of our problem. Section 3 establishes the minimax separation rates for univariate function selection from two viewpoints. Section 4 provides a rate-optimal selector adaptive to the smoothness parameter in the Sobolev space. Section 5 offers an in-depth discussion about the difference between estimation and selection in SpAMs. The limitations, future directions, numerical experiments, and all technical proofs are provided in the appendices. For a square intergral function f with support X , denote by ∥f ∥ 2 = X f 2 (x)dxfoot_0/2 its L 2 norm.
Let C, C 0 , C 1 , • • • denote absolute positive constants whose values may change from one occurrence to the next.
this section cite: []

Section: Preliminary and problem setup
Let us recall that we observe Y x and x ∈ χ p such that
dY x = j∈S f f j (x j )dx + σdB x .(1)
To ensure the identifiability of univariate functions, we assume X f j (x j )dx j = 0 for each j ∈ [p].
In theoretical research, the GWN model and nonparametric regression model are asymptotically equivalent, as shown by Brown and Low [1996], Reiß [2008] 1 . Moreover, the GWN model simplifies the analysis by avoiding unnecessary technical complexities while keeping the focus on the statistical essence [Kotekal and Gao, 2024]. Consequently, many foundational nonparametric statistics theories are developed based on the GWN model [Fan, 1991, Donoho and Johnstone, 1998, Baraud, 2002, Tsybakov, 2009, Comminges and Dalalyan, 2012, Johnstone, 2017, Han et al., 2020]. Therefore, to maintain this theory-driven tradition, we conduct our analysis based on the GWN model (1).
this section cite: ['b35', 'b36', 'b33', 'b38', 'b39', 'b40', 'b37', 'b29', 'b41', 'b42']

Section: Function settings
We propose a general smoothness assumption based on the series expansion of univariate functions. For each j ∈ [p], assume that f j : X → R can be decomposed from an orthonormal basis
{ψ i } i∈N + , as f j (x j ) = ∞ i=1 θ ij ψ i (x j ), where θ ij = θ ij (f j ) := X ψ i (x j )f j (x j )dx j is the coefficient of ψ i for each i ∈ N + . Define θ •j = θ •j (f j ) := {θ ij } i∈N + .
We assume that each f j is sufficiently smooth and belongs to the ellipsoid class
E := f j = ∞ i=1 θ ij ψ i : ∞ i=1 θ 2 ij µ i ≤ 1 ,(2)
where {µ i } ∞ i=1 is a non-increasing sequence of positive numbers, i.e., µ 1 ≥ µ 2 ≥ • • • , and we assume µ 1 ≍ 1 to ensure f j has finite L 2 norm. This ellipsoid setting is a broad smoothness assumption that renders our theoretical results applicable to Reproducing Kernel Hilbert Space (RKHS) [Raskutti et al., 2012, Yuan and Zhou, 2016, Kotekal and Gao, 2024], Fourier basis [Comminges and Dalalyan, 2012, Ingster and Stepanova, 2014, Butucea and Stepanova, 2017], etc. The function space of SpAM is defined as
F s :=    f (x) = p j=1 f j (x j ) : p j=1 1(f j ̸ = 0) ≤ s, f j ∈ E for all j ∈ [p]    .
(3) Each f ∈ F s corresponds uniquely to a Θ = Θ(f ) := (θ
•1 (f 1 ), • • • , θ •p (f p )) ∈ R N + ×p .
Therefore, f ∈ F s and Θ ∈ F s will be used interchangeably in the subsequent text. For every f ∈ F s and every i ∈ N + , j ∈ [p], based on the continuous process Y x in model (1), we have access to the following random variables
X ij := X p ψ i (x j )dY x = θ ij + X p ψ i (x j )σdB x ∼ N (θ ij , σ 2 ).
By orthogonality, the set X = {X ij } i∈N + ,j∈[p] is a collection of independent random observations.
this section cite: ['b3', 'b5', 'b33', 'b29', 'b28', 'b26']

Section: Problem setup
Within the SpAM space F s , our primary task is to establish a minimax optimal (i.e., necessary and sufficient) signal condition of each support f j , for the univariate function selection. Before delving into our analysis, we revisit the function estimation problem in SpAM, where Raskutti et al. [2012] established the minimax rate as:
inf f sup f ∈Fs E f f (X) -f 2 2 ≍ s × σ 2 log(ep/s) High-dimensional selection error +s × max k∈N + (σ 2 k) ∧ µ k inf f j sup f j ∈E E f j ∥ fj (X•j )-fj ∥ 2 2 , (4
) which is composed of s times the "high-dimensional selection error" and s times the "minimax estimation rate of a single univariate function", with no interplay between these two parts. This result shows that the first term σ 2 s log(ep/s) is independent of the univariate function space E, and the estimation term (the second term) is dimension-free (p-free) [Kotekal and Gao, 2024]. Therefore, it is natural to speculate that the univariate function selection shares a similar property, with its optimal signal condition, quantified by the squared L 2 norm, of the rate:
σ 2 log(ep/s) High-dimensional selection error + max k∈N + (σ 2 √ k) ∧ µ k ,(5)
where the second term is the minimax separation rate for the goodness-of-fit test of a single univariate function in E [Baraud, 2002].
However, in Section 3 we prove that this is not the case. In univariate function selection, there is an interplay between the high-dimensional sparse structure (selection error) and the ellipsoid space E, complicating the form of its minimax rate.
this section cite: ['b3', 'b33', 'b40']

Section: Main result: optimal univariate function selection
In this section, we demonstrate that the truncated-type selectors lead to minimax optimal results. Define the decoder η j = η j (f j ) := 1(f j ̸ = 0), and the corresponding vector η = η(f ) := (η 1 (f 1 ), • • • , η p (f p )) ∈ {0, 1} p . We also define the selector, i.e., the estimation of η, as η = η(X) = (η 1 (X), • • • , ηp (X)) ∈ {0, 1} p , and Ŝ = {j ∈ [p] : ηj = 1} as the estimated support set corresponding to η. Define the SpAM space with the signal strength condition as
F s (r 2 ) :=    f = j∈[p] f j ∈ F s : ∥f j ∥ 2 2 ≥ r 2 for all f j ̸ = 0    ,(6)
indicating that each support f j has a signal separated from 0. Here r 2 is a positive value and we
additionally assume r 2 ≤ µ 1 to ensure F s (r 2 ) ̸ = ∅ (since ∥f j ∥ 2 2 ≤ µ 1 i θ 2 ij µi ≤ µ 1 based on f j ∈ E).
In the next two subsections, we derive the minimax separation rates from two viewpoints, sparse multiple testing and support recovery, respectively.
this section cite: []

Section: From sparse multiple testing: FDR + FNR control
Preliminary setup From the viewpoint of testing, the selection can be realized as a multiple-testing problem H 0j : f j = 0, H 1j : f j ̸ = 0, for all j ∈ [p], under the exactly s-sparse function space
f ∈ F =s (r 2 ) :=    f = j∈[p] f j ∈ F s (r 2 ) : j∈[p] 1(f j ̸ = 0) = s    .
We consider the multiple testing risk combined with the false discovery rate (FDR) plus the false negative rate (FNR), which is of the form
R(f, η) = E f j / ∈S f ηj 1 ∨ j∈[p] ηj + j∈S f (1 -ηj ) s .
This combined risk balances the proportion of type I and type II errors, and is frequently used in the sparse testing [Arias-Castro and Chen, 2017, Rabinovich et al., 2020, Abraham et al., 2024].
Definition 1 (Minimax separation rate of sparse multiple testing) We say ϵ 2 test is the nonasymptotic minimax separation rate of the sparse multiple testing problem for (1) if:
(1) For all δ ∈ (0, 1), there exists c δ > 0 depending only on δ such that for all 0 < c < c δ ,
inf η sup f ∈F=s(cϵ 2 test ) R(f, η) ≥ 1 -δ.
(2) For all δ ∈ (0, 1), there exists C δ > 0 depending only on δ such that for all C > C δ ,
inf η sup f ∈F=s(Cϵ 2 test ) R(f, η) ≤ δ,
where inf η denotes the infimum over all selector η(X) : R N + ×p → {0, 1} p .
this section cite: ['b43', 'b14', 'b18']

Section: K-truncated selector
For each sequence X •j , we truncate by the first K entries and construct the corresponding selector
ηtest j (X •j ) = 1 K i=1 X 2 ij ≥ σ 2 K + λ 2 (K) , j ∈ [p],(7)
where the truncation K := min k ∈ N + : µ k ≤ σ 2 k log(p/s) , and the parameter λ 2 (K) will be determined in Theorem 1. Denote by ηtest = (η test 1 , • • • , ηtest p ) ∈ {0, 1} p the corresponding selector vector. The following theorem employs an analysis to control the combined risk at a low level.
Theorem 1 (Upper bound for sparse multiple testing) Let δ be an arbitrary number in (0, 1), and assume that σ -2 > C δ,1 log(p/s) µ1 , p/s ≥ C δ,2 , and s ≥ C δ,3 . Then, assuming
r 2 ≥ 6 √ δ √ 10 + 2 + √ 2 max k∈N + σ 2 k log(p/s) ∧ µ k + 36 δ σ 2 log(p/s)(8)
and taking
λ 2 (K) = 2σ 2 5K log p sδ + 5 log p sδ , we have sup f ∈F=s(r 2 ) R(f, ηtest ) ≤ δ,
where C δ,1 , C δ,2 , C δ,3 are positive constants only determined by δ.
The next theorem shows that the rate in (8) is also necessary for controlling the testing risk.
Theorem 2 (Lower bound for sparse multiple testing) Let δ be an arbitrary number in (0, 1), and assume that σ -2 > C δ,1 log(p/s) µ1 , p/s ≥ C δ,2 , and s ≥ C δ,3 . Then, for all r 2 satisfies
0 < r 2 ≤ c δ,4 σ 2 log(p/s) + max k∈N + σ 2 k log(p/s) ∧ µ k , we have inf η sup f ∈F=s(r 2 ) R(f, η) ≥ 1 -δ,
where C δ,1 , C δ,2 , C δ,3 and c δ,4 are four positive constants only determined by δ.
Therefore, combining Theorem 1 and 2, we establish the minimax separation rate for the sparse multiple testing in the SpAM (1) as
ϵ 2 test ≍ σ 2 log(p/s) + max k∈N + σ 2 k log(p/s) ∧ µ k .(9)
We also illustrate that a truncated-type selector possesses such minimax optimality.
Remark 1 (Truncation) So far, the equation (9) reveals that our initial speculation (5), in the end of Section 2.2, is inaccurate: The high-dimensional sparsity structure influences both terms in the minimax separation rate. This is because the selection problem is related to the chi-squared distribution, whose heavy tail leads to the selection error of the rate σ 2 log(p/s) + K log(p/s) . Therefore, we have to choose an appropriate truncation level K to balance the residual signal strength µ K with this composite error bound, i.e., µ K ≍ σ 2 log(p/s) + K log(p/s) . Consequently, the high-dimensional structure affects the choice of truncation, revealing an interplay that is not only sufficient but also necessary.
Remark 2 (SpAM and GSM) The Gaussian sequence model (GSM, mentioned in Section 1.1) can be seen as a simplified SpAM, where θ 1j = 1 and θ ij = 0 for each i ≥ 2 and j ∈ S f . Therefore, in GSM, we can just choose truncation K ≡ 1, and analyze the selection error caused by the Gaussian distribution [Butucea et al., 2018, Song andCheng, 2023]. In contrast, to get an optimal truncation K in general SpAM space, our selector (7) requires trading off the truncation bias against sub-exponential error. Both the analysis and outcome demonstrate that univariate function selection in SpAM is more challenging than variable selection in GSM.
Additionally, our theoretical results can be extended to the following specific cases.
this section cite: ['b16']

Section: Corollary 1
Assume that all assumptions in Theorem 2 and Theorem 1 hold. Then we have:
• Sobolev Take µ i ≍ i -2α with smoothness parameter α, the minimax separation rate for multiple testing is
ϵ 2 test ≍ σ 2 log(p/s) + σ 4 log(p/s) 2α 1+4α
.
• Finite dimension Take µ 1 = • • • = µ m > µ m+1 = µ m+2 = • • • = 0 for some positive integer m, the minimax separation rate for multiple testing is
ϵ 2 test ≍ σ 2 log(p/s) + σ 2 m log(p/s) ∧ µ 1 .
• Exponential decay Take µ i ≍ exp(-c 1 i γ ), where c 1 is a positive constant and γ > 0, the minimax separation rate for multiple testing is
ϵ 2 test ≍ σ 2 log(p/s) + σ 2 log(p/s) • log 1 2γ σ -4 log(p/s)
.
this section cite: []

Section: Remark 3 (Finite dimension case)
We now give a further discussion of the finite dimension case. Under the assumption σ -2 ≳ log(p/s) µ1 , the minimax separation rate exhibits two regimes:
1. If µ 1 ≲ σ 2 m log(p/s), then we derive that
√ m log(p/s) µ1 ≳ σ -2 ≳ log(p/s) µ1 , leading m ≳ σ -2 µ 1 . In this case ϵ 2 test ≍ σ 2 log(p/s) + µ 1 ≍ µ 1 .
2. If µ 1 ≻ σ 2 m log(p/s), then we get ϵ 2 test ≍ σ 2 log(p/s) + σ 2 m log(p/s), which aligns with the minimax separation rate in the group sparsity setting [Butucea et al., 2023a].
Combining these cases gives a more intuitive separation rate
ϵ 2 test ≍ min σ 2 log(p/s) + m log(p/s) , µ 1 .
Here the minimum reflects our ellipsoid space constraint: by definition of E and F s (r 2 ) in Section 2.1, every active univariate function
f j obeys ∥f j ∥ 2 2 ≤ µ 1 i∈N + θ 2 ij µi ≤ µ 1 , leading its L 2 norm upper bounded by µ 1 .
In the case µ 1 ≲ σ 2 m log(p/s), we have ϵ 2 test ≍ µ 1 . In other words, to control FDR+FNR, we would need each f j to satisfy ∥f _j∥_2 ≥ Cµ 1 for some large constant C > 0. However, each support f j obeys ∥f _j∥ 2 ≤ µ 1 . Hence, there are basically no f j that can attain a detectable norm, and the support recovery problem is essentially trivial in this case.
this section cite: []

Section: From support recovery: wrong recovery probability control
Preliminary setup The univariate function selection can also be viewed as a support recovery problem. We measure the selection error between the estimated support Ŝ and the true support S by using the Hamming loss 1(η(X) ̸ = η(f )), where the probability of wrong recovery
P f Ŝ(X) ̸ = S(f ) = P f (η(X) ̸ = η(f )) = E 1 η(X) ̸ = η(f )
serves as the risk function, which characterizes how we can exactly recover the support set [Wainwright, 2007, Butucea et al., 2023a].
Definition 2 (Minimax separation rate of support recovery) We say ϵ 2 rec is the non-asymptotic minimax separation rate of support recovery for (1) if:
(1) For all δ ∈ (0, 1), there exists c δ > 0 depending only on δ such that for all 0 < c < c δ ,
inf η sup f ∈Fs(cϵ 2 rec ) P f (η(X) ̸ = η(f )) ≥ 1 -δ.
(2) For all δ ∈ (0, 1), there exists C δ > 0 depending only on δ such that for all C > C δ ,
inf η sup f ∈Fs(Cϵ 2 rec ) P f (η(X) ̸ = η(f )) ≤ δ,
where inf η denotes the infimum over all selector η(X) : R N + ×p → {0, 1} p .
Similar to Section 3.1, we next establish the minimax separation rate for the support recovery problem in SpAM (1).
Theorem 3 (Minimax separation rate of support recovery) Let δ be an arbitrary number in (0, 1), and assume that σ -2 > C δ,1 log p µ1
, p ≥ C δ,2 , and s ≥ C δ,3 . Then the minimax separation rate for the support recovery problem with respect to the wrong recovery probability
P f (η(X) ̸ = η(f )) is ϵ 2 rec ≍ σ 2 log p + max k∈N + σ 2 k log p ∧ µ k .(10)
The minimax separation rate for support recovery in (10) is a little greater than that for sparse multiple testing in (9) (log p versus log(p/s)), showing that controlling the wrong recovery probability is more demanding than controlling the combined risk (FDR + FNR). Indeed, sparse testing requires | Ŝ∆S| = o(s), while exact recovery requires | Ŝ∆S| = o(1), necessitating a slightly stronger signal condition. In addition, this discrepancy leads to a higher thresholding level for optimal selection in support recovery, as detailed in the following.
Remark 4 (Rate-optimal selector) Under assumptions in Theorem 3 and the signal condition
r 2 ≥ C δ σ 2 log p + max k∈N + σ 2 k log p ∧ µ k , the selector ηrec j (X) = 1    K ′ i=1 X 2 ij ≥ σ 2 K ′ + 2σ 2 K ′ log(2p/δ) + log(2p/δ)    , j ∈ [p](11)
controls the wrong recovery probability effectively:
sup f ∈Fs(r 2 ) P f (η rec (X) ̸ = η(f )) ≤ δ,
where K ′ := min k ∈ N + : µ k ≤ σ 2 √ k log p and C δ > 0 is a constant only determined by δ.
Remark 5 (Relation to existing work) For the Sobolev space with smoothness parameter α, we rewrite the minimax separation rate (10) for exact support recovery as:
ϵ 2 rec ≍ σ 8α 4α+1 (log p) 2α 4α+1 if log p ≲ σ -2 2α+1 , σ 2 log p if σ -2 ≳ log p ≳ σ -2 2α+1 .(12)
Therefore, in the case log p = o σ -2 2α+1
, we match the rate derived from Ingster and Stepanova [2014], Butucea and Stepanova [2017]. Additionally, our findings establish the non-asymptotic minimax separation rate for the case σ -2 ≳ log p > σ -2 2α+1 , which was not provided in previous studies. In this case, the selection error exhibits sub-Gaussian behavior, resulting in the rate aligning with that in the Gaussian sequence model [Butucea et al., 2018, Song andCheng, 2023].
this section cite: ['b19', 'b28', 'b26', 'b16']

Section: Adaptation to the smoothness
Thus far, our analysis has assumed full knowledge of the smoothness sequence {µ i } i∈N + , which is often unrealistic. This section investigates how adaptation to unknown smoothness affects the minimax separation rate. For simplicity, we consider the Sobolev space with µ i = i -2α , α > 0, and rewrite the original space F s (r 2 ) as F s (r 2 , α). The wrong recovery probability P f (η(X) ̸ = η(f )) is used as the risk function.
A selector adaptive to the unknown α Define the truncation set
K rec := 2, 4, • • • , 2 log 2 σ -4 log p .
For every δ ∈ (0, 1) and k ∈ K rec , we denote η(k) (X) := η(k) 1 (X), • • • , η(k) p (X) ∈ {0, 1} p as the selector vector with respect to k, where
η(k) j := 1 σ -2 k i=1 X 2 ij ≥ k + 2 k log 8p log(σ -2 ) δ + 2 log 8p log(σ -2 ) δ .
Now, we define the adaptive selector
ηad (X) := max k∈Krec η(k) 1 , • • • , max k∈Krec η(k) p ∈ {0, 1} p .(13)
For each f j , our selector (13) firstly constructs individual tests for each k ∈ K rec , and then aggregates them by taking the maximum over K rec . Equivalently, f j is declared supported as soon as it is identified as nonzero under any candidate k ∈ K rec ; conversely, f j is declared non-supported only if it is identified as zero for all k ∈ K rec . We next establish the sufficient signal condition for the wrong recovery probability control.
Theorem 4 (Upper bound for adaptation) Let δ be an arbitrary number in (0, 1), and assume that σ -2 > C δ,1 log p µ1
. Then, for all r 2 satisfies
r 2 ≥ 12 √ 2 + 1 σ 8α 1+4α log 2α 1+4α 8p log(σ -2 ) δ + 18σ 2 log 8p log(σ -2 ) δ , we have sup α>0 sup f ∈Fs(r 2 ,α) P f ηad (X) ̸ = η ≤ δ.
Compared to ( 12), an additional log log(σ -2 ) term in the signal strength condition is required. The following theorem shows that log log(σ -2 ) is also necessary for the adaptation to the smoothness.
Theorem 5 (Lower bound for adaptation) Let δ be an arbitrary number in (0, 1), and assume that σ -2 > C δ,1 log p µ1
and p ≥ C δ,2 . Then, for all r 2 satisfies
0 < r 2 ≤ c δ,3 σ 8α 1+4α log 2α 1+4α p log(σ -2 ) + σ 2 log p log(σ -2 ) , we have inf η sup α>0 sup f ∈Fs(r 2 ,α) P f (η(X) ̸ = η) ≥ 1 -δ.
Theorem 4 and 5 establish the adaptive minimax separation rate as
σ 8α 1+4α log 2α 1+4α p log(σ -2 ) + σ 2 log p log(σ -2 ) .(14)
In the high-dimensional case log p ≳ log log(σ -2 ) , the log log(σ -2 ) term becomes negligible and ( 14) achieves the same rate as (12), indicating that adaptation incurs no additional cost on the rate. However, when p is a large constant that is much smaller than σ -2 , ( 14) indicates that, with the smoothness unknown, achieving support recovery requires a stronger signal strength compared to ( 12).
this section cite: []

Section: Discussion: difference between optimal estimation and selection
We finally end this paper by discussing the difference between estimation and selection. For simplicity, we assume s ≤ p 1-β , where β ∈ (0, 1) is a constant, therefore log(ep/s) ≍ log p. Next, we establish a minimax-optimal estimator for f ∈ F s through a truncated hard-thresholding procedure:
θij = X ij • 1(i ≤ K e ) Truncation • 1 σ -2 Ke i ′ =1 X 2 ij ≥ K e + CK e log p + C log p Hard thresholding ,(15)
where we define K e := min{k ∈ N + : µ k ≤ σ 2 k}, and C > 0 is a fixed constant.
Theorem 6 (Optimal truncation for function estimation
) Assume σ -2 ≥ C1 log p µ1
and the canstant C in (15) satisfies C ≥ 4. Then the estimator (15) is rate-optimal:
sup f ∈Fs E f ∥ f ( Θ) -f ∥ 2 2 = sup f ∈Fs E f ∥ Θ -Θ(f )∥ 2 2 ≲ σ 2 s log p + s × max k∈N+ (σ 2 k) ∧ µ k .
Combined with (4), Theorem 6 implies that by only using the first K e entries in each observation sequence (i.e., only using X [Ke]×[p] := {X ij } 1≤i≤Ke,1≤j≤p ), one can achieve a minimax optimal function estimation. However, it may fail to guarantee optimal univariate function selection by only using these truncated observations, as shown below.
Theorem 7 (Suboptimal selection) Assume σ -2 ≥ C1 log p µ1
and p ≥ C 2 . Then, for all r 2 satisfies
0 < r 2 ≤ c 3 σ 2 log p + max k∈[Ke] σ 2 k log p ∧ µ k + µ Ke+1 ,(16)
we have a lower bound as
inf η(X [Ke ]×[p] )∈{0,1} p sup f ∈Fs(r 2 ) P f η(X [Ke]×[p] ) ̸ = η ≥ 1 2 ,
where the infimum inf η(X[K e ]×[p]) ∈{0,1} p takes over all restricted selectors that only use observations X [Ke]×[p] , and C 1 , C 2 , c 3 > 0 are absolute constants.
This theorem demonstrates that in the family of truncation estimators, optimal estimation sometimes leads to a suboptimal univariate function selection. For example, consider the Sobolev space with
µ i = i -2α in the case log p = o σ -2 1+2α .
To exactly recover the support set, the necessary signal strength (16) (by only using X
[Ke]×[p] ) is of the rate σ 4α 1+2α , which exceeds the minimax separation rate σ 8α 4α+1 (log p) 2α 4α+1
, as illustrated in ( 12). This gap directly shows that optimal univariate function selection cannot be treated as a byproduct of optimal SpAM function estimation. See Figure 1 for a clearer difference.
µ k = k -2α σ 2 k σ 2 √ k log p k Signal strength r 2 Truncation for optimal estimation Ke≍σ - 2 1+2α σ 4α 1+2α
Truncation for optimal selection
K≍σ - 4 1+4α (log p) - 1 1+4α σ 8α 4α+1 (log p) 2α 4α+1
Figure 1: The difference between optimal estimation and selection in the case log p = o σ -2 1+2α .
Appendix A discusses some future directions for this paper, and Appendix B provides the numerical experiment to confirm our theoretical findings.
this section cite: []

Section: Chao Gao, Fang Han, and Cun-Hui Zhang. On estimation of isotonic piecewise constant signals.
The Annals of Statistics, 48(2):629 -654, 2020. doi:10.1214/18-AOS1792. URL https:  //doi.org/10.1214/18-AOS1792. Anru R. Zhang and Yuchen Zhou. On the non-asymptotic and sharp lower tail bounds of random variables. Stat, 9(1):e314, 2020. doi:https://doi.org/10.1002/sta4.314. URL https:  //onlinelibrary.wiley.com/doi/abs/10.1002/sta4.314. e314 sta4.314. Malay Ghosh. Exponential tail bounds for chisquared random variables. Journal of Statistical Theory and Practice, 15(2):35, 2021. Lucien Birgé. An alternative point of view on lepski's method. Lecture Notes-Monograph Series, pages 113-133, 2001.
NeurIPS Paper Checklist
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: [Yes] The abstract of this paper precisely outlines our contributions in terms of minimax separation rates for univariate function selection in sparse additive models.
We also discuss the adaptation to the smoothness and the difference between the optimal function estimation and univariate function selection. Section 1 and 2 also present the main contributions and assumptions in this paper.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: [Yes] This paper sets certain limitations on the model. Our analysis is based on the Gaussian white noise model instead of the empirical nonparametric regression model, which means our results cannot be directly used in a real application. However, as we point up in Section 2, the Gaussian white noise model simplifies the analysis by avoiding unnecessary technical complexities while keeping the focus on the statistical essence, and our results are asymptotically applied to the nonparametric regression model. We also provide an in-depth discussion of our limitations and future directions in Appendix A.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: [Yes] Each theorem in this paper comes with detailed assumptions. All formal proofs of all theorems are provided in Appendix C-I, and the formal proofs of auxiliary lemmas are provided in Appendix K.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: [Yes] This paper provides rate-optimal selectors and an adaptive selection procedure. We thoroughly outline the experimental parameter settings and simulation procedures in Appendix B. Additionally, we upload all the R code required for the experiments in the supplementary material.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
5. Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
Answer: [Yes] Justification: [Yes] The supplementary material includes all code used in our experiments, covering data generation, preprocessing, truncation selection, effectiveness analysis, and so on. The code provides specific parameter settings and random seeds to ensure the complete reproducibility of all results shown in Appendix B.
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
Answer: [Yes] Justification: [Yes] In Appendix B, we present comprehensive introductions to our procedure and data, covering data generation, preprocessing, truncation selection, performance metrics, and so on. Furthermore, the supplementary material includes all the code used in our experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: [Yes] The simulation results in this paper report 1-sigma error bars based on standard errors from 300 Monte Carlo simulations (see figures in Appendix B). The variability of error bars arises from the randomness of error terms in simulations.
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
Answer: [Yes] Justification: [Yes] We provide the information on the computer resources in Appendix B. All simulations are conducted using R and executed on a personal laptop equipped with an AMD Ryzen 7 5800H processor operating at 3.20 GHz and 16.00GB of RAM.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: [Yes] This paper consists solely of theoretical analysis and simulation experiments, with all data being randomly generated. It does not engage human subjects or participants, nor does it raise data security concerns such as personal privacy. Additionally, the supplemental material contains all the code for our experiments, ensuring the reproducibility of our results. Therefore, the research presented in this paper adheres to the NeurIPS Code of Ethics in all respects.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] Justification: [NA] In this paper, we purely discuss the theoretical minimax separation rates of univariate function selection in sparse additive models. It belongs to the domain of statistical theory research, and therefore does not involve societal impacts.
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
Answer: [NA]
Justification: [NA] In this paper, we purely discuss the theoretical minimax separation rates of univariate function selection in sparse additive models. Therefore, it belongs to the domain of statistical theory research and poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: [Yes] We cite all the papers that inspired this work, and also provide citations for the techniques used in the proofs.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes] Justification: [Yes] We upload all the code in the supplementary materials as a zipped file, covering data generation, pre-processing and so on. The code provides specific hyperparameter settings and random seeds to ensure the complete reproducibility of all results.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: [NA] This paper purely discusses the theoretical minimax separation rates of univariate function selection in sparse additive models. Therefore, it does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: [NA] This paper purely discusses the theoretical minimax separation rates of univariate function selection in sparse additive models. Therefore, it does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: [NA] This paper purely discusses the theoretical minimax separation rates of univariate function selection in sparse additive models. Therefore, it does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/  LLM) for what should or should not be described.
These appendices provide the future directions, numerical experiments and the technical proofs of the main manuscript. For notational convenience, throughout all appendices we define n := σ -2 to represent the noise intensity.
this section cite: []

Section: A Limitations and future directions
Besov ball or L q ball One key direction for future work is to extend our univariate function selection results from the L 2 -ellipsoid (2) to richer nonparametric classes such as Besov balls B σ r,q or, more generally, L q -ellipsoids. These spaces naturally align with wavelet bases and are foundational to practical methods in signal processing and denoising. However, under the high-dimensional setting, the techniques in Baraud [2002] may not be useful anymore. Perhaps a more viable approach is to construct selectors based on the nonquadratic estimation procedure in Cai andLow [2005, 2006], which may also lead to a minimax adaptation result simultaneously.
this section cite: ['b40']

Section: Univariate function selection under local differential privacy
Integrating the differential privacy (DP) mechanism into univariate function selection for SpAM represents a direction for future research. DP ensures rigorous protection of individual data while allowing valid statistical inference; therefore is welcomed by the computer science, machine learning, and statistics communities recently. In the local DP setting, Butucea et al. [2023b] established phase transitions for support recovery in the sparse mean model, deriving minimax separation rates for exact recovery and for almost-full recovery. Butucea et al. [2020Butucea et al. [ , 2023c] ] studied the function estimation and the quadratic functional estimation in the nonparametric univariate function, respectively, where the latter plays an important role in goodness-of-fit testing. All these works demonstrated that DP leads to some markedly different minimax rates compared to non-private benchmarks.
Consequently, when extending univariate function selection in SpAM to local DP constraints, one should expect that the minimax separation rates will differ from the results in this paper: the optimal truncation should be recalibrated to account for the additional privacy-induced noise. Designing and analyzing such privacy-preserving selectors for SpAMs remains an important and challenging problem.
A general conclusion about estimation and selection Another significant extension lies in generalizing the minimax lower bound in Theorem 7, which currently restricts the infimum to selectors relying solely on truncated observations X [Ke]× [p] . To this end, we define the minimax optimal estimation class
E opt := f : R N + ×[p] → R N + ×[p] sup f ∈Fs E f f (X) -f 2 2 ≲ σ 2 s log p + s × max k∈N + (σ 2 k) ∧ µ k .
The general version of Theorem 7 should focus on the necessary signal condition for selectors induced by estimation class E opt :
inf f ∈Eopt inf η=η( f ) sup f ∈Fs(r 2 ) P f η( f ) ̸ = η(f ) ≥ c.
Ideally, this lower bound could quantify how the minimax optimal estimations perform in the support recovery problem. It could also lead to a more comprehensive realization of the difference between estimation and selection.
Establishing such a result will likely require some new analytic tools, and we think the techniques in Song and Cheng [2023] may give some help. We leave this interesting problem for future research.
this section cite: ['b47', 'b16']

Section: B Numerical experiment
We conduct three simulation studies to evaluate the performance of our truncated-type selectors in sparse additive models. For ease of display, we define n = σ -2 . 1. Compare the performance of our proposed method across varying dimension p and signal strength r 2 . 2. Compare the performance of different selection methods across varying variance 1/n.
this section cite: []

Section: Compare the performance of different selection methods across varying smoothness parameters.
In all experiments, we take X = [0, 1], s = 5, and let the support covariates be j = 1, . . . , 5, with centered functions
f 1 (x) = x 2 2 x-1 -(x -0.5) 2 e x -0.5424, f 2 (x) = 12(x -0.5) 2 -12, f 3 (x) = 3x 2 2 x-1 cos(15x) -0.1002, f 4 (x) = 2x -1, f 5 (x) = 8(x -0.7) 3 + 0.4640,
which all belong to the Sobolev space with α = 1/2.
Performance is measured by the Hamming loss
1 η(X) ̸ = η(f ) ,
and the combined FDR plus FNR loss
j / ∈S f ηj 1 ∨ j∈[p] ηj + j∈S f (1 -ηj ) s .
For each simulation, we execute 300 repetitions, with 1-sigma error bars provided in the figures. All simulations are conducted using R and executed on a personal laptop equipped with an AMD Ryzen 7 5800H processor operating at 3.20 GHz and 16.00GB of RAM.
this section cite: []

Section: B.1 Simulation 1: dimension and signal strength
We fix n = 300, and vary p ∈ {10, 100, 1000, 10000}. We take a • f j as the support function, for j = 1, • • • , 5, where a > 0 quantifies the effect of the signal strength.
Figure 2 shows that, as the signal strength a increases, the selection errors (both Hamming loss and FDR plus FNR loss) for each p decay toward a relatively low level, but larger p demands higher a to reach the same error level. Moreover, controlling FDR + FNR requires weaker signal strengths: at p = 10000, a = 0.5 suffices to keep FDR + FNR = 0.5, whereas the Hamming loss drops below 0.5 until a = 0.7. This behavior reflects the fundamental difference between sparse multiple testing and exact support recovery, as we discussed after Theorem 3.
this section cite: []

Section: B.2 Simulation 2: noise variance 1/n
We fix p = 500 and vary n from 20 to 300. Four types of selectors are considered in this simulation:
1. Optimal The rate-optimal selector (11).
this section cite: []

Section: Adaptation
The adaptive selector (13).
3. Univariate The selector that takes truncation at K u = min k ∈ N + : µ k ≤ √ k n .
4. Suboptimal The selector that takes truncation at K e = min k ∈ N + : µ k ≤ k n .
Figure 3 illustrates that, as n grows, all methods see error decay, but the Optimal and Adaptation methods maintain the lowest selection errors across most regimes. Additionally, as we discussed in Remark 5, for n ≲ (log p) 1+2α , the minimax separation rate is log p/n, under which the K etruncation remains rate-optimal, giving the Suboptimal selector a temporary advantage (for n < 100).
Once n ≳ (log p) 1+2α , the minimax separation rate becomes n -4α 4α+1 (log p) 2α 4α+1 , and truncation at K e cannot be optimal anymore.
this section cite: []

Section: B.3 Simulation 3: smoothness parameter β
We fix p = 500 and n = 300, and assess the effect of smoothness on univariate function selection. First, for j = 1, . . . , 5, we compute the original basis coefficients of each f j , denoted by {θ ij } i∈N + . We next reweight these coefficients and get the new functions
f (β) j = i∈N + i 1 2 -β θ ij ψ i , j = 1, • • • , 5, so that each f (β) j
lies in the Sobolev ball with smoothness parameter β. We vary β ∈ [0.2, 1] and compare the performance of the four methods.
As shown in Figure 4, only the Adaptation method consistently achieves low error across all β, demonstrating its optimality and robustness to unknown smoothness and verifying our theoretical guarantees in Section 4.
this section cite: []

Section: C Proof of Theorem 3
We first introduce the proof of the lower bound and upper bound in Theorem 3. These proofs are instructive and lead to clearer proofs of Theorem 1 and Theorem 2.
0.0 0.1 0.2 0.3 0.4 0.25 0.50 0.75 1.00 Smoothness beta Hamming Loss Support Recovery 0.00 0.05 0.10 0.15 0.20 0.25 0.50 0.75 1.00 Smoothness beta FDR + FNR Loss Sparse Multiple Testing Method Adaptation Optimal Suboptimal Univariate We first define the Hamming distance
H(η(X), η(f )) := p j=1 |η j -η j | = p j=1 1(η j ̸ = η j ),
which can upper-bound the wrong classification probability
E f H(η, η) = p w=1 wP f H(η, η) = w ≥ P f η ̸ = η .(17)
For notational convenience, throughout all proofs we set n = σ -2 to represent the noise intensity.
this section cite: []

Section: C.1 The lower bound
To better clarify the truncation construction, we define
E (k) (r 2 ) := θ ∈ R N + : θ i = r √ k for all 1 ≤ i ≤ k, θ i = 0 for all i > k ⊂ E.
The SPAM function set induced by E (k) (r 2 ) is:
F (k) s (r 2 ) :=    f = p j=1 f j ∈ F s (r 2 ) : f j ∈ E (k) (r 2 ) for all f j ̸ = 0    .
Now we consider:
inf η:R N + ×p →{0,1} p sup f ∈Fs(r 2 ) P f η(X) ̸ = η(f ) ≥ inf η:R N + ×p →{0,1} p sup f ∈F (k) s (r 2 ) P f η(X) ̸ = η(f ) = inf η:R k×p →{0,1} p sup f ∈F (k) s (r 2 ) P f η(X k×p ) ̸ = η(f ) ,(18)
which means that in F (k) s (r 2 ), we only need to consider those selectors η based on the first k observations in each univariate function f j , j ∈ [p]. Now, for some fixed k ∈ N + , we set a least favorable subset of F (k) s (r 2 ), and then derive its lower bound of the minimax separation rate.
this section cite: []

Section: C.1.1 The least favorable subset
For every fixed δ ∈ (0, 1), k ∈ N + which satisfy µ k ≥ c1δ
25 ∧ 1 max log(p-s) n , √ k log(p-s) n (where c 1 > 0 is a constant defined in Lemma 2), consider the subset:
F(k) s (r 2 ) :=    Θ ∈ F (k) s (r 2 ) : p j=1 1(θ •j ̸ = 0) = s    .(19)
Therefore, for each f ∈ F(k) s (r 2 ), if its j-th univariate function f j ≡ 0, the random variable n k i=1 X 2 ij follows from a central χ 2 -distribution with k degrees of freedom (note that
X ij ∼ N (θ ij , 1/n)). If f j ̸ = 0, n k i=1 X 2
ij follows from a non-central χ 2 -distribution with k degrees of freedom and with non-centrality parameter nr 2 . Let f 0 and f 1 be the densities of these two distributions with respect to the Lebesgue measure:
f 0 (z) = z k/2-1 e -z/2 2 k/2 Γ(k/2) , z > 0, f 1 (z) = 1 2 k/2 e -nr 2 /2 ∞ i=0 nr 2 4 i z k/2+i-1 e -z/2 i! Γ(k/2 + i) , z > 0.(20)
Once the positive integer k is fixed, by Lemma 1 we only need to consider the selector based on the norm ∥X 1:k,j ∥ 2 , which we call them the norm selectors. Then we conclude
inf η:R k×p →{0,1} p sup f ∈F (k) s (r 2 ) P f η(X k×p ) ̸ = η(f ) ≥ inf η:R k×p →{0,1} p sup f ∈ F (k) s (r 2 ) P f η(X k×p ) ̸ = η(f ) (i) ≥ inf η: norm selector sup f ∈ F (k) s (r 2 ) P f η ∥X 1:k,1 ∥ 2 , • • • , ∥X 1:k,p ∥ 2 ̸ = η(f ) (ii) ≥ P e(s) min j=1,••• ,s f 1 f 0 n∥X 1:k,j ∥ 2 2 ≤ max j=s+1,••• ,p f 1 f 0 n∥X 1:k,j ∥ 2 2 (iii) = P e(s) min j=1,••• ,s n∥X 1:k,j ∥ 2 2 ≤ max j=s+1,••• ,p n∥X 1:k,j ∥ 2 2 , (21
)
where inequality (i) follows from Lemma 1, inequality (ii) follows from Theorem 6 in Butucea et al.
[2023a], where we denote by P e(s) a probability measure in which only the first s univariate functions are non-zero, i.e., f j = 0 ⇔ j / ∈ [s]. Equality (iii) follows from the monotonic increasing property of the likelihood ratio f1 f0 (z) on z ∈ R + .
this section cite: []

Section: C.1.2 The tail probabilities
With the fixed δ ∈ (0, 1), k ∈ N + which satisfy µ k ≥ c1δ 25 ∧ 1 max log
(p-s) n , √ k log(p-s) n , we aim to prove that the last probability in (21) is greater than 1-δ if log(p-s) ≥ 16 c 2 1 + log log (2/δ) c2 ∨ 2 log log (2/δ) c2 ∨ 8 c1 and r 2 ≤ c1δ 25 ∧ 1 max log(p-s) n , √ K log(p-s) n , where c 1 > 0 and c 2 ∈ (0, 1) are two positive constants defined in Lemma 2. Firstly, by taking x = log c2(p-s) log(2/δ) > 0, we conclude P e(s) max j=s+1,••• ,p
n∥X 1:k,j ∥ 2 2 ≥ k + c 1 x + c 1 √ kx =1 -1 -P χ 2 k (0) ≥ k + c 1 x + c 1 √ kx p-s (i) ≥1 -1 -c 2 e -x p-s =1 -1 - log(2/δ) p -s p-s (ii) ≥ 1 - δ 2 ,(22)
where inequality (i) follows from (59) in Lemma 2, inequality (ii) follows from the assumption
p -s ≥ log(2/δ) c2 2 > log(2/δ) c2 > 1. Besides, by taking r 2 ≤ c1δ 25 max log(p-s) n , √ k log(p-s) n and log(p -s) ≥ 2 log log(2/δ) c2 , we conclude x ≥ 1 2 log(p -s) > δ 2 log(p -s)
and
nr 2 + 2 √ 2nr 2 < 2c 1 25 x + √ kx + 2 2 • 2c 1 25 x + √ kx < c 1 2 x + √ kx ,(23)
where the last inequality follows from c 1 x + √ kx ≥ 4 led by log(p -s) ≥ 8/c 1 . Then, by assuming s ≥ log(2/δ) and log(p -s) ≥ 16 c 2 1 + log log(2/δ) c2 , we conclude x ≥ 16/c 2 1 , therefore
nr 2 + 2 (k + 2nr 2 ) log(2/δ) s + 2 log(2/δ) s ≤nr 2 + 2 k + 2nr 2 + 2 ≤nr 2 + 2 √ 2nr 2 + 2 √ k + 1 (i) < c 1 2 x + √ kx + c 1 2 √ x √ k + 1 ≤c 1 x + √ kx ,
where inequality (i) follows from ( 23). Therefore, we conclude
P e(s) min j=1,••• ,s n∥X 1:k,j ∥ 2 2 ≤ k + c 1 x + c 1 √ kx =1 -P χ 2 k (nr 2 ) ≥ k + c 1 x + c 1 √ kx s ≥1 -P χ 2 k (nr 2 ) ≥ k + nr 2 + 2 (k + 2nr 2 ) log(2/δ) s + 2 log(2/δ) s s (i) ≥1 -exp - log(2/δ) s s = 1 - δ 2 ,(24)
where inequality (i) follows from ( 61) in Lemma 2 with non-centrality parameter B = nr 2 . Combining (21), ( 22) and ( 24), we conclude that
inf η sup f ∈Fs(r 2 ) P f η(X) ̸ = η(f ) ≥P e(s) min j=1,••• ,s ∥X 1:k,j ∥ 2 2 ≤ max j=s+1,••• ,p ∥X 1:k,j ∥ 2 2 ≥P e(s) min j=1,••• ,s n∥X 1:k,j ∥ 2 2 ≤ k + c 1 x + c 1 √ kx × P e(s) max j=s+1,••• ,p n∥X 1:k,j ∥ 2 2 ≥ k + c 1 x + c 1 √ kx ≥ 1 - δ 2 2 > 1 -δ.
this section cite: []

Section: C.1.3 The optimal truncation K
By the definition of F(k) s (r 2 ), F
s (r 2 ) and U (k) (r 2 ), the minimax separation rate is lower bounded by the constrained maximum:
max : c δ max log(p -s) n , k log(p -s) n , subject to : c δ max log(p -s) n , k log(p -s) n ≤ µ k , k ∈ N + ,(25)
where c δ = c1δ 25 ∧ 1 ∈ (0, 1]. For ease of display, we define
K (c δ ) := min k ∈ N + : µ k ≤ c δ k log(p -s) n L (c δ ) := max k ∈ N + : µ k ≥ c δ k log(p -s) n , (26
) By assuming n > c δ log(p-s) µ1 , we derive that 1 ≤ L (c δ ) ≤ K (c δ ) ≤ L (c δ ) + 1.
Then we analyze the maximum into two cases:
• Case A: When µ ⌈log(p-s)⌉ ≥ c δ √ ⌈log(p-s)⌉ log(p-s) n . We derive that L (c δ ) ≥ ⌈log(p -s)⌉ hence c δ √ L (c δ ) log(p-s) n ≥ c δ log(p-s) n .
Then the maximum of ( 25) is
c δ √ L (c δ ) log(p-s) n .
• Case B: When µ ⌈log(p-s)⌉ <
c δ √ ⌈log(p-s)⌉ log(p-s) n . We derive that 1 ≤ L (c δ ) ≤ ⌊log(p -s)⌋ hence c δ √ L (c δ ) log(p-s) n ≤ c δ log(p-s) n .
Then the maximum of ( 25) is
c δ log(p-s) n .
Therefore, we establish the lower bound of the minimax separation rate as:
c δ • max log(p -s) n , L (c δ ) log(p -s) n (i) ≍ max log(p -s) n , K (c δ ) log(p -s) n (ii) ≍ max log(p -s) n , max k∈N + µ k ∧ k log(p -s) n ≍ log(p -s) n + max k∈N + µ k ∧ k log(p -s) n ,(27)
where equality (i) follows from 1 ≤ L (c δ ) ≤ K (c δ ) ≤ L (c δ ) + 1 and equality (ii) follows from Lemma 4. By (27), we derive the lower bound of the minimax separation rate.
this section cite: []

Section: C.2
The upper bound By (17), we only need to prove that sup f ∈Fs(r 2 ) E f H(η(X), η(f )) ≤ δ. For ease of display, we denote λ 2 (K ′ ) = 2 n K ′ log(2p/δ) + log(2p/δ) .
this section cite: []

Section: C.2.1 Preliminary
For a fixed SPAM f ∈ F s (r 2 ), we use S f ⊂ [p] as the index set of the support univariate functions f j ̸ = 0. Then we have
E f H(η(X), η(f )) =E f p j=1 1 ηj (X) ̸ = η j (f j ) = p j=1 P f ηj (X) ̸ = η j (f j ) = j∈S f P fj   K ′ i=1 X 2 ij < K ′ n + λ 2 (K ′ )   + j / ∈S f P fj   K ′ i=1 X 2 ij ≥ K ′ n + λ 2 (K ′ )   .
(28) Therefore, we will discuss the Hamming loss on the support and non-support separately.
this section cite: []

Section: C.2.2 Support
With signal condition
∥f j ∥ 2 2 ≥ 24 1 δ + √ 2 max k∈N + √ k log p n ∧ µ k + 36 log p δn
holds for all j ∈ S f , we have
∥θ 1:K ′ ,j ∥ 2 2 = K ′ i=1 θ 2 ij ≥∥f j ∥ 2 2 -µ K ′ ∞ i=K+1 θ 2 ij µ i (i) ≥ 12 2 δ + 1 K ′ log(p) n + 36 log p δn -µ K ′ (ii) ≥ 12 2 δ K ′ log(p) n + 36 log p δn (iii) ≥ 12 K ′ log(2p/δ) n + 18 log(2p/δ) n ,(29)
where inequality (i) follows from the signal condition and a proof strategy similar to (63) in Lemma 4, inequality (ii) follows from the definition of K ′ and inequality (iii) follows from (2/δ) log p ≥ log(2p/δ) when p ≥ 2.
We decompose X ij = θ ij + ξ ij with each ξ ij ∼ N (0, 1/n) independently. Then we get
P fj   K ′ i=1 X 2 ij < K n + λ 2   =P fj ∥ξ 1:K ′ ,j ∥ 2 2 + ∥θ 1:K ′ ,j ∥ 2 2 + 2 ⟨ξ 1:K ′ ,j , θ 1:K ′ ,j ⟩ < K ′ n + λ 2 (K ′ ) (i) ≤P fj ∥ξ 1:K,j ∥ 2 2 + ∥θ 1:K,j ∥ 2 2 + 2 ⟨ξ 1:K,j , θ 1:K,j ⟩ < K n + λ 2 (K ′ ) ∩ A j + P fj A c j ≤P fj ∥ξ 1:K ′ ,j ∥ 2 2 + ∥θ 1:K ′ ,j ∥ 2 2 -∥θ 1:K ′ ,j ∥ 2 8 log(2p/δ) n < K ′ n + λ 2 + exp (-log(2p/δ)) (ii) ≤ P fj ∥ξ 1:K,j ∥ 2 2 + 1 3 ∥θ 1:K,j ∥ 2 2 < K n + λ 2 (K ′ ) + δ 2p (iii) ≤ P fj n∥ξ 1:K ′ ,j ∥ 2 2 < K ′ -2 K ′ log(2p/δ) + δ 2p ≤ δ p ,(30)
where in inequality (i) we define event
A j = ⟨ξ 1:K ′ ,j , θ 1:K ′ ,j ⟩ > - 2∥θ 1:K ′ ,j ∥ 2 2 log(2p/δ) n
, where
⟨ξ 1:K ′ ,j , θ 1:K ′ ,j ⟩ ∼ N 0, ∥θ 1:K ′ ,j ∥ 2 2 n
. Inequality (ii) and (iii) follow from ( 29), and the last inequality follows from (60) in Lemma 2.
this section cite: []

Section: C.2.3 Non-support
We now focus on the Hamming loss on the non-support. For every j / ∈ S, we have n
K i=1 X 2 ij ∼ χ 2 K (0), therefore P fj K i=1 X 2 ij ≥ K n + λ 2 ≤ δ 2p ,(31)
where the last inequality follows from (61) in Lemma 2.
Combining ( 28), (30), and (31), we conclude
E f H(η(X), η(f )) = j∈S f P fj   K ′ i=1 X 2 ij < K n + λ 2 (K ′ )   + j / ∈S f P fj   K ′ i=1 X 2 ij ≥ K n + λ 2 (K ′ )   (i) ≤ |S f |δ p + (p -|S f |)δ 2p = (p + |S f |)δ 2p ≤ δ,
where in inequality (i), we use
|S f | to denote the cardinal number of the support index set S f , hence 1 ≤ |S f | ≤ s ≤ p.
Therefore, we complete the proof of the upper bound and also Theorem 3.
this section cite: []

Section: D Proof of Theorem 1
In the proof of the upper bound, we first recall our signal condition
∥f j ∥ 2 2 ≥ 6 √ δ √ 10 + 2 + √ 2 max k∈N + k log(p/s) n ∧ µ k + 36 δn log(p/s)
and the selector
ηj = 1 n K i=1 X 2 ij ≥ K + 2 5K log(p/(sδ)) + 10 log(p/(sδ)) ,
where
K = min k ∈ N + : µ k ≤ √ k log(p/s) n
and n = σ -2 . We assume p/s ≥ √ 12 ∨ 6/δ, s ≥ 16 ∨ C δ , and n ≥ C δ log(p/s) µ1 , where C δ is a positive constant solely determined by δ ∈ (0, 1).
this section cite: []

Section: D.1 FNR control
Similar to ( 29) in the proof of Theorem 3, we get
∥θ 1:K,j ∥ 2 2 ≥ 6 √ 5 + √ 2 K log(p/(sδ)) n + 36 n log(p/(sδ)).
Then for j ∈ S f , similar to ( 30) we get
E f j∈S f (1 -ηj ) s = E f (1 -ηj ) ≤P χ 2 K (0) + n 3 ∥θ 1:K,j ∥ 2 2 < K + 2 5K log(p/(sδ)) + 10 log(p/(sδ)) + P 2n ⟨ξ 1:K,j , θ 1:K,j ⟩ ≤ -4 n∥θ 1:K,j ∥ 2 2 log(p/(sδ)) ≤P fj χ 2 K (0) < K -8K log(p/(sδ)) + s p 2 δ ≤2 s p 2 δ,(32)
which also leads to
E f   j∈S f ηj   ≥ s -2s s p 2 δ.
Besides, by Hoeffding's inequality, we get
P   j∈S f ηj -E j∈S f ηj ≤ -s 3/4   ≤ exp -2 √ s , yielding P   j∈S f ηj ≤ s -2s s p 2 δ -s 3/4   ≤ exp -2 √ s .(33)
D.2 FDR control By Markov's inequality, we conclude
P   j / ∈S f ηj > s (p/s) 2   ≤ (p/s) 2 s j / ∈S f P n K i=1 X 2 ij ≥ K + 2 5K log(p/(sδ)) + 10 log(p/(sδ)) ≤ s p 2 δ,(34)
which yields E    j / ∈S f ηj 1 ∨ j∈[p] ηj • 1   j∈S f ηj > s -2s s p 2 δ -s 3/4   • 1   j / ∈S f ηj ≤ s (p/s) 2      ≤ s (p/s) 2 s (p/s) 2 + s -2s (s/p) 2 δ -s 3/4 (i) ≤ s (p/s) 2 s (p/s) 2 + s 3 ≤ 2δ 3 ,(35)
where inequality (i) follows from p/s ≥ √ 12 and s ≥ 16, and the last inequality follows from p/s ≥ 9/(2δ). Besides, by a similar technique, we have
E    j / ∈S f ηj 1 ∨ j∈[p] ηj • 1   j / ∈S f ηj > s (p/s) 2      ≤ P    j / ∈S f ηj > s (p/s) 2    ≤ s p 2 δ,(36)
where the last inequality follows from ( 34), and the first inequality holds because
j / ∈S ηj 1∨ j∈[p] ηj ≤ 1. We also get E    j / ∈S f ηj 1 ∨ j∈[p] ηj • 1   j∈S f ηj ≤ s -2s s p 2 δ -s 3/4   • 1   j / ∈S f ηj ≤ s (p/s) 2      ≤ s (p/s) 2 P    j∈S f ηj ≤ s -2s s p 2 δ -s 3/4    (i) ≤ se -2 √ s (p/s) 2 ≤ s p 2 δ,(37)
where inequality (i) follows from ( 33), and the last inequality is based on that the function g(x) = xe -2 √
x is monotonically decreasing and tends to 0 on (1, ∞), and hence for every δ ∈ (0, 1), there exists a corresponding C δ such that s > C δ yields se -2 √ s ≤ δ.
this section cite: []

Section: D.3 Conclusion
Combining (32), (35) (36), and (37), we get
E j / ∈S f ηj 1 ∨ j∈[p] ηj + j∈S f (1 -ηj ) s =E    j / ∈S f ηj 1 ∨ j∈[p] ηj • 1   j∈S f ηj > s -2s s p 2 δ -s 3/4   • 1   j / ∈S f ηj ≤ s (p/s) 2      + E    j / ∈S f ηj 1 ∨ j∈[p] ηj • 1   j∈S f ηj ≤ s -2s s p 2 δ -s 3/4   • 1   j / ∈S f ηj ≤ s (p/s) 2      + E    j / ∈S f ηj 1 ∨ j∈[p] ηj • 1   j / ∈S f ηj > s (p/s) 2      + E j∈S f (1 -ηj ) s ≤ 2δ 3 + s p 2 δ + s p 2 δ + 2 s p 2 δ ≤ δ,
where the last inequality follows from p/s ≥ √ 12. Therefore we get an upper bound of the combined risk in sparse multiple testing with a rate-optimal signal condition, which completes the proof of Theorem 1.
this section cite: []

Section: References
Ref_id:b0 Title: Sparse Additive Models Year: (2009)
Ref_id:b1 Title: High-dimensional additive modeling Year: (2009)
Ref_id:b2 Title: Sparsity in multiple kernel learning Year: (2010)
Ref_id:b3 Title: Minimax-optimal rates for sparse additive models over kernel classes via convex programming Year: (2012)
Ref_id:b4 Title: Statistical inference in compound functional models Year: (2014)
Ref_id:b5 Title: Minimax optimal rates of estimation in high dimensional additive models Year: (2016)
Ref_id:b6 Title: Learning sparse additive models with interactions in high dimensions Year: (2016-05)
Ref_id:b7 Title: Doubly penalized estimation in additive regression with highdimensional data Year: (2019)
Ref_id:b8 Title: Generalized sparse additive models Year: (2022)
Ref_id:b9 Title: Generalized additive models: Some applications Year: (1987)
Ref_id:b10 Title: Multichannel nonparametric signal detection Year: (2003)
Ref_id:b11 Title: Multi-task additive models for robust estimation and automatic structure discovery Year: (2020)
Ref_id:b12 Title: Sparse neural additive model: Interpretable deep learning with feature selection via group sparsity Year: (2023)
Ref_id:b13 Title: Variable selection with hamming loss Year: (2018)
Ref_id:b14 Title: Optimal rates and trade-offs in multiple testing Year: (2020)
Ref_id:b15 Title: Uncertainty quantification for robust variable selection and multiple testing Year: (2022)
Ref_id:b16 Title: Optimal false discovery control of minimax estimators Year: (2023)
Ref_id:b17 Title: Variable selection, monotone likelihood ratio and group sparsity Year: (2023)
Ref_id:b18 Title: Sharp multiple testing boundary for sparse sequences Year: (2024)
Ref_id:b19 Title: Information-theoretic bounds on sparsity recovery in the high-dimensional and noisy setting Year: (2007)
Ref_id:b20 Title: Fundamental limits of exact support recovery in high dimensions Year: (2020)
Ref_id:b21 Title: Component selection and smoothing in multivariate nonparametric regression Year: (2006)
Ref_id:b22 Title: Variable selection in nonparametric additive models Year: (2010)
Ref_id:b23 Title: Generalized additive model selection Year: (2015)
Ref_id:b24 Title: Faithful variable screening for high-dimensional convex regression Year: (2016)
Ref_id:b25 Title: Generalized additive models for large data sets Year: (2015)
Ref_id:b26 Title: Adaptive variable selection in nonparametric sparse additive models Year: (2017)
Ref_id:b27 Title: Kernel knockoffs selection for nonparametric additive models Year: (2023)
Ref_id:b28 Title: Adaptive variable selection in nonparametric sparse regression Year: (2014)
Ref_id:b29 Title: Tight conditions for consistency of variable selection in the context of high dimensionality Year: (2012)
Ref_id:b30 Title: Exact variable selection in sparse nonparametric models Year: (2001)
Ref_id:b31 Title: Adaptive exact recovery in sparse nonparametric models. Statistical Inference for Stochastic Processes Year: (2025)
Ref_id:b32 Title: Oracle inequalities and optimal inference under group sparsity Year: (2011)
Ref_id:b33 Title: Minimax signal detection in sparse additive models Year: (2024)
Ref_id:b34 Title: Minimax estimation of linear and quadratic functionals on sparsity classes Year: (2017)
Ref_id:b35 Title: Asymptotic equivalence of nonparametric regression and white noise Year: (1996)
Ref_id:b36 Title: Asymptotic equivalence for nonparametric regression with multivariate and random design Year: (1957)
Ref_id:b37 Title: Introduction to nonparametric estimation Year: (2009)
Ref_id:b38 Title: On the Estimation of Quadratic Functionals Year: (1991)
Ref_id:b39 Title: Minimax estimation via wavelet shrinkage Year: (1998)
Ref_id:b40 Title: Non-asymptotic minimax rates of testing in signal detection Year: (2002)
Ref_id:b41 Title: Gaussian estimation: Sequence and wavelet models Year: (2017)
Ref_id:b42 Title: On estimation of l r -norms in gaussian white noise models Year: (2020)
Ref_id:b43 Title: Distribution-free multiple testing Year: (1983)
Ref_id:b44 Title: Nonquadratic estimators of a quadratic functional Year: (2005)
Ref_id:b45 Title: Optimal adaptive estimation of a quadratic functional Year: (2006)
Ref_id:b46 Title: Phase transitions for support recovery under local differential privacy Year: (2023)
Ref_id:b47 Title: Local differential privacy: Elbow effect in optimal density estimation and adaptation over Besov ellipsoids Year: (2020)
Ref_id:b48 Title: Interactive versus noninteractive locally differentially private estimation: Two elbows for the quadratic functional Year: (2023)
