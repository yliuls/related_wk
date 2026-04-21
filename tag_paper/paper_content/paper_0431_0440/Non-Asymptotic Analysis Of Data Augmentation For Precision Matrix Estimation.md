Title: Non-Asymptotic Analysis Of Data Augmentation For Precision Matrix Estimation
Abstract: This paper addresses the problem of inverse covariance (also known as precision matrix) estimation in high-dimensional settings. Specifically, we focus on two classes of estimators: linear shrinkage estimators with a target proportional to the identity matrix, and estimators derived from data augmentation (DA). Here, DA refers to the common practice of enriching a dataset with artificial samples-typically generated via a generative model or through random transformations of the original data-prior to model fitting. For both classes of estimators, we derive estimators and provide concentration bounds for their quadratic error. This allows for both method comparison and hyperparameter tuning, such as selecting the optimal proportion of artificial samples. On the technical side, our analysis relies on tools from random matrix theory. We introduce a novel deterministic equivalent for generalized resolvent matrices, accommodating dependent samples with specific structure. We support our theoretical results with numerical experiments.

Section: Introduction
In this work, we consider the problem of estimating the inverse covariance matrix, also known as the precision matrix, of a random vector from i.i.d. zero-mean samples [X 1 , • • • , X n ] ∈ R d×n with true covariance Σ X = E X 1 X 1 . Here, n denotes the number of samples, and d is the dimensionality of the data. This problem has important applications in statistics and signal processing (see, e.g., [FLL16,Car88]).
We are particularly interested in the high-dimensional regimes, where the data dimension d and the number of samples n are of the same order. In this setting, the sample covariance matrix C X = n -1 XX may be non-invertible or poorly-conditionned. As a result, using its inverse as an estimator can lead to numerical instability and high estimation error. To address this issue, shrinkage estimators for the covariance matrix have been proposed as a regularization method [BGP16, LW04, SS05, LSW03], which involve adding a target matrix to C X . The simplest and most common choice for the target is a multiple of the identity matrix, which effectively shifts the eigenvalues above a threshold λ > 0, improving stability. In addition to linear shrinkage and even more importantly, this paper also explores the use of data augmentation (DA) as an alternative strategy.
Data augmentation (DA) involves increasing the size of a dataset by incorporating additional artificial samples. The underlying intuition is that, in many cases, it is possible to artificially replicate the data distribution, thereby reducing the variance of the model while maintaining relatively low bias. Due to its effectiveness in low-data regimes and its ability to mitigate overfitting, DA has become increasingly popular and is now widely used in machine learning and data science [SK19, GSK18, CKNH20, GSA + 20]. It finds applications across a variety of fields, including computer vision [SK19], natural language processing [FGW + 21], and neuroscience [LLM20].
Two main types of data augmentation (DA) can be distinguished. The first is Transformative Data Augmentation (TDA), where original samples are transformed through a random mapping-for example, by adding Gaussian noise or applying a random mask. The second is Generative Data Augmentation (GDA), in which artificial samples are generated using a generative model and added to the training dataset. In the case of GDA, we assume that the generative model has been pre-trained on the original samples X.
In both cases, the artificial samples are dependent on the original data. Although TDA and GDA differ conceptually, our framework and results encompass both approaches. More precisely, we consider the inverse of the empirical covariance matrix associated with the augmented dataset X = [X 1 , . . . , X n , G 1 , . . . , G m ] as an estimator of Σ -1 X , where each G i ∈ R d is an artificial sample. That is, we study the estimator which consists of the inverse of (n + m) -1 X X + λ I d , for some regularization parameter λ > 0.
Although there is extensive empirical evidence that DA improves the performance of machine learning models, the theoretical literature on the subject remains relatively limited. In this work, our goal is to establish performance guarantees that enable meaningful comparisons between different DA strategies and, as a by-product, allow for the optimization of certain associated hyperparameters. To this end, we leverage tools from random matrix theory to construct estimators of the quadratic error for DA-based estimators, which, under mild assumptions, satisfy exponential concentration inequalities. Our analysis can also be adapted-and in fact simplifies-to cover linear shrinkage estimators with a target proportional to the identity matrix. To summarize, our main contributions are as follows:
• In Section 2, we focus on the estimator of Σ -1 X given by the inverse of a linear shrinkage estimator, where the shrinkage target is a scalar multiple of the identity matrix. Specifically, we derive estimators for the quadratic error and show that they satisfy non-asymptotic exponential concentration bounds. Our results hold under standard assumptions from random matrix theory-namely, the Lipschitz concentration property for X.
• In Section 3, we extend our analysis to data-augmented estimators under appropriate conditions on the DA procedure, which we show hold true for common DA.
• Finally, for both scenarios, we show how our estimators for the quadratic error can be used to compare and tune their corresponding methods with respect to key hyper-parameters such as the nomber of additional samples m for DA. These conclusions are illustrated numerically on real data in Section 4.
this section cite: ['b19', 'b6', 'b49', 'b33']

Section: Notation and convention.
Motivated by high-dimensional statistics, we will consider that n, d and m are variables, yet for notation simplicity, we will most often not reflect dependancies in n, m or d in our notations. Additionally, we write, x y (resp x y) whenever x ≤ Cy (resp x ≥ Cy) for a universal constant C that neither depends on the model's parameters, nor on the parameters n, m, d. For any matrix H ∈ R d×k , we denoted by C H = k -1 HH the corresponding covariance matrix. For any symmetric matrix Σ ∈ R d×d , we denote by λ d (Σ) ≤ . . . ≤ λ 1 (Σ) its eigenvalues. The Frobenius and operator norms are denoted • F and • op respectively. Random variables will be referred to by capital letters X, G, Z, and we will denote by Ber(p), N(m, Σ) and Unif(E) the Bernoulli distribution of parameter p ∈ [0, 1], the Gaussian distribution of mean m ∈ R d and covariance Σ ∈ R d×d , and the uniform distribution over a discrete set E. Furthermore, we introduce the p-Wasserstein metric, W p p (ν 1 , ν 2 ) = inf γ∈Γ(ν1,ν2)
x -y p F dγ(x, y) for any distributions ν 1 and ν 2 on R d×k , and where γ ∈ Γ(ν 1 , ν 2 ) if and only if for any E ∈ R d×k , γ(R d×n , E) = ν 1 (E), and γ(E, R d×n ) = ν 2 (E).
this section cite: []

Section: Related work

this section cite: []

Section: Data Augmentation.
Numerous empirical studies have demonstrated the benefits of using DA when training machine learning models [MM22b,MMN22,vDM01]. Among popular DA schemes, we can mention AutoAugment [CZM + 19, LKK + 19, ZWZZ20], which aims to learn an optimal augmentation policy from data by combining a set of sub-policies. In addition, significant works have also explored the incorporation of knowledge about the distribution invariances directly into the training procedures [CDL20].
However, DA does not always lead to a systematic improvement in test error [KIB + 23, HGK20, CL25], and very little theoretical understanding backs-up the improvement observed empirically. An early analysis by [Bis95] showed that adding Gaussian noise to data points is equivalent to applying Tikhonov regularization. Building on this seminal work-and given the practical importance of DA for machine learning practitioners-a few studies have sought to develop a theoretical understanding of its effects.
In the context of kernel methods, [DGR + 19] showed that DA can be approximated by a combination of first-order feature averaging and second-order variance regularization. Similarly, in the context of linear and logistic regressions, [LKDM22] revealed that DA induces implicit spectral regularization in two ways: first, by adjusting the relative proportions of the eigenvalues of the data covariance matrix in a training-dependent way; and second, by uniformly shifting the entire spectrum via ridge regression.
Taking a different perspective, [WZVR20] consider a family of linear transformations and study their effects on the ridge estimator in an over-parametrized linear regression setting. First, they show that transformations that preserve the labels of the data can improve estimation by increasing the span of the training data. Second, they show that transformations that mix data can improve estimation by playing a regularization effect. They proposed an augmentation scheme that searches over the linear span of a set of transformations, aiming to maximize model uncertainty on the transformed data.
Recently, studies have shown that even small amounts of artificial data can lead to model collapse [SSZ + 24, DFK24, DFSK25], a phenomenon where the performance of generative models deteriorates when recursively trained on synthetic data.
this section cite: ['b42', 'b43', 'b53', 'b7', 'b3', 'b32', 'b59']

Section: Precision Matrix Estimation.
In a high-dimensional settings-where the number of covariates is comparable to or exceeds the number of observations-traditional covariance estimation methods often suffer from poor conditioning, making the estimation of the precision matrix particularly challenging. To address this, [LW04] introduced linear shrinkage methods, which involve forming a convex combination C X + (1 -)T of the sample covariance matrix C X with a shrinkage target T, where ∈ [0, 1]. [LW03,LW04] derived the optimal value of that minimizes the mean squared error between the shrinkage estimator and the true covariance matrix Σ X .
Extending this work, [LW12,LW22,BGBP23] proposed and analyzed non-linear shrinkage estimators of the form U f (D)U , where C X = U DU is the eigenvalue decomposition of C X and f is a suitably chosen function applied to the eigenvalues.
Fewer works have addressed shrinkage methods specifically designed for precision matrix estimation. Among them, [BGP16] studied estimators of the form (1 -)C -1 X + Π, where Π is a deterministic shrinkage target, and derived the optimal shrinkage intensity . In addition, [WPTZ15] considered estimators of the form Ω X ( ) = ((1 -)C X + I d ) -1 and derived the optimal to minimize the objective function Ω X ( )Σ X -I d F in the high-dimensional regime where d/n → γ > 0.
An independent line of research, motivated by Gaussian graphical models [YL07], has focused on estimating sparse precision matrices. In particular, [MH12,CLL11] introduced the Graphical Lasso method, which has become widely used for this purpose.
this section cite: ['b37', 'b36', 'b37', 'b38', 'b39', 'b1', 'b2', 'b58', 'b60', 'b40', 'b11']

Section: Random Matrix Theory.
Since the pioneering work of [Wis28], numerous studies have investigated the behavior of the eigenvalues and eigenvectors of the sample covariance matrix C X in the highdimensional regime where d/n → γ > 0; see, for example, [MP67,Sil89]. More recently, [AEK + 14] first demonstrated that, in the isotropic setting, the resolvent (C X + λ I d ) -1 converges weakly to a scalar multiple of the identity matrix as d/n → γ > 0. This was later extended to the anisotropic case by [KY17], who established so-called deterministic equivalent results for (C X + λ I d ) -1 . These results have been further generalized to settings with more complex dependency structures. In particular, [Cho22,LC23] showed that deterministic equivalents continue to hold under weaker assumptions.
Building on these foundational results, several studies have established connections between classical random matrix theory-particularly the results of [MP67,Sil89]-and the behavior of modern machine learning models. In particular, random matrix theory has proven instrumental in explaining the double-descent phenomenon, initially observed in linear models [HMRT22, DLM20, MVSS20, BLLT20, DKT21], and later extended to certain classes of shallow models, such as random feature models [MM22a, LCM21, GLK + 21, DRBK20]. Complementing these works, [SDCL24,SCDL24] provided a sharp asymptotic characterization of the test error in deep random feature models-representing a significant step toward understanding generalization in deeper architectures. Finally, [ITL + 24] leveraged random matrix theory to develop precise performance estimates for multi-task learning across a variety of statistical models.
this section cite: ['b57', 'b44', 'b48', 'b28', 'b8', 'b30', 'b44', 'b48', 'b47', 'b46']

Section: Inverse covariance estimation using shrinkage method
We consider here the following estimator R X (λ) of the precision matrix, and its squared error:
R X (λ) = (C X + λ I d ) -1 , E X (λ) := 1 d R X (λ) -Σ -1 X 2 F .(1)
Here, λ > 0 is a hyperparameter that controls the strength of the regularization. Note that this estimator is not per se a shrinkage estimator, but it is the inverse of a shrinkage estimator of the covariance matrix. In addition, R X (λ) is also referred to as the diagonal loading estimator in the signal processing community; see e.g., [LSW03]. Furthermore, note that our result can be readily applied to an estimators of the form ((1 -α)C X + ασ I d ) -1 = (1 -α) -1 (C X + ασ I d /(1 -α)) -1 , for any σ ∈ R + and α ∈ (0, 1).
This estimator does not rely on any data augmentation procedure. However, as we will see, applying data augmentation leads to results that are closely related to this regularization approach. This is not surprising, as it is relatively well known that data augmentation induces an implicit regularization effect [Bis95,LKDM22]. For this reason, we present this simple case in detail as a preliminary step, which will allow us to transition more smoothly to the data-augmented case in Section 3. As already emphasized in the introduction, our main goal is to derive a data-centric estimate for the error E X (λ).
To this end, we introduce two assumptions.
H1 (Concentration of X). The random matrix X ∈ R d×n writes Σ 1/2 Z, where Z ∈ R d×n has independant sub-Gaussian entries with parameter σ X .
H1 is standard in the random matrix theory literature, yet one can employ a more general framework, as in [Cho22, ITL + 24, LC18] by introducing the notion of Lipschitz concentration (H6), we detail this generalization throughout the appendix and stick to this simpler framework in the main body for the sake of simplicity. In addition, we expect our results to remain valid under a finite-moment assumption on the entries of X, albeit with weaker-typically polynomial-concentration bounds, as done in [LP11]. We leave a detailed investigation of this extension for future work.
We next suppose that with high probability the leave one-out covariance matrix C - X is wellconditionned. This matrix is defined for any X ∈ R d×n as the covariance matrix
C - X = C X -, X -= [0, X 2 , . . . , X n ] .
More formally, for any η > 0 define
A η = {X ∈ R d×n : λ d (C - X ) ≥ η} .(2)
this section cite: ['b35', 'b3', 'b32', 'b34']

Section: H2 (Model conditionning).
There exist η > 0 and c X > 0 such that P (X / ∈ A η ) e -c X n .
We highlight in Appendix A that H2 holds provided H1 holds and n ≥ K X (d + η + 1) for some constant K X depending only on σ X .
We are now ready to present our estimator for E X (λ) and to state its concentration properties. The estimator is given by:
ÊX (λ) := 1 d tr R X (λ) 2 - 2(1 -d/n) tr (R X (0)) λ 1 Aη (X) + 2tr (R X (λ)) λb(λ) + tr Σ -2 X (3) b(λ) := 1 1 -d/n + (λ/n) tr (R X (λ))
.
Note that for a fixed η > 0, ÊX (λ) is computable from the data X only, up to an additive constant.
Theorem 1. Assume H1 and H2. Then, it holds for all t ≥ 0 and λ > 0,
P E X (λ) -ÊX (λ) ≥ t + ∆ X (λ) exp -cλ d (Σ X ) 2 σ 2 X ndη 3 t 2
for a universal constant c > 0 and where
∆ X (λ) := C 1 σ 2 X √ d Σ X 3 op nλ d (Σ X )η 6 + C 2 e -c X n + 1 λ 3 nd .
Here
C 1 , C 2 > 0 are explicit polynomial functions of Σ X -1 op , λ d (Σ X ), (η + λ) and c -1 X , see (B).
From a practical standpoint, the previous result can be used to optimize the hyperparameter λ by minimizing the function λ → E X (λ), using ÊX as a proxy. Moreover, it is worth noting that the derivative of λ → ÊX depends only on the data matrix X, and not on the true covariance Σ X . As a result, ÊX can be minimized using a gradient descent scheme, provided that the parameter η > 0 satisfies H2. We illustrate these applications on real data in Section 4.
Although the full proof of Theorem 1 is deferred to Appendix B, we provide here a sketch of its derivation. We highlight the main ideas and technical challenges, and note that the proof of our result on estimation using data augmentation, Theorem 2, shares several of these steps.
First, expanding the Frobenius norm in (1), we have,
E X (λ) = (1/d) tr R X (λ) 2 -(2/d) tr Σ -1 X R X (λ) + (1/d) tr Σ -2 X .
The first term in the previous expansion is directly computable from the data matrix X, while the last term is constant with respect to λ and can be ignored when the goal is to optimize λ. Therefore, it suffices to establish a deterministic equivalent for R X (λ) and thus of tr Σ -1 X R X (λ) . To this end, we rely on the following result, whose proof is postponed to appendix B. Proposition 1. Assume H1 and H2. Let B ∈ R d×d be a deterministic matrix, then we have for all λ ≥ 0,
P 1 d tr B R X (λ)1 Aη (X) -E R X (λ)1 Aη (X) ≥ t exp -c(η + λ) 3 σ 2 X ndt 2 . Furthermore, defining f λ (b) = 1 + n -1 tr Σ X (Σ X /b + λ I d ) -1 and b * := b * (λ) as the unique fixed point of f λ on [1, ∞), we have E R X (λ) - Rb * X (λ) 1 Aη (X) F C 1 σ 2 X √ d Σ X 3 op nλ 1 (Σ X )(η + λ) 6 + C 2 e -c X n , Rb * X (λ) = Σ X b * + λ I d -1 ,
where C 1 , C 2 are defined as in Theorem 1.
Proposition 1 shows that R X (λ) concentrates around Rb * X (λ), which we refer to as a deterministic equivalent of R X (λ). Our result extends that of [Cho22] by covering the case of vanishing regularization (λ → 0). This extension constitutes the main technical innovation required for the proof of Theorem 1.
We can now leverage the deterministic equivalent of Proposition 1 to rewrite (1/d) tr Σ -1 X R X (λ) . Informally, it holds with high probability that
1 d tr Σ -1 X R X (λ) ≈ 1 d tr Σ -1 X Σ X b * (λ) + λ I d -1 1 Aη (X) = 1 λdb * (0) tr Rb * (0) X (0) 1 Aη (X) - 1 λdb * (λ) tr Rb * (λ) X (λ) 1 Aη (X)
where the last equality follows from the identity A -1 -B -1 = A -1 {B -A}B -1 . Finally, using Proposition 1 again, and that b * (0) = (1
-d/n) -1 is the fixed point of f 0 : b → 1 + bd/n, we get 1 d tr Σ -1 X R X (λ) ≈ 1 -d/n λd tr (R X (0)) 1 Aη (X) - 1 λdb * (λ) tr (R X (λ)) .(4)
Finally, by the definition of b * (λ) and through straightforward algebraic manipulations, we obtain
b * (λ) = 1 + 1 n tr Σ X Rb * (λ) X (λ) = 1 + b * (λ) d n - λ n tr Rb * (λ) X (λ) .
Therefore, applying Proposition 1 again yields
b * (λ) = 1 1 -(d/n) + (λ/n) tr Rb * (λ) X (λ) ≈ 1 1 -(d/n) + (λ/n) tr (R X (λ)) = b(λ) ,
Plugging this estimate in (4), we identify ÊX (λ) (3) and it completes the proof of Theorem 1. The formal proof is postponed to Appendix B in the supplement.
this section cite: ['b8']

Section: Precision matrix estimation using generic data augmentation
In this section, we investigate a data augmentation strategy to improve the estimation of the inverse covariance matrix of X. Specifically, we consider an additional set of artificial samples
G = [G 1 , • • • , G m ]
, which may depend on X and are typically generated using either TDA or GDA techniques; see Table 1. More precisely, given X, we assume that G is drawn from a known regular conditional distribution (X, A) → ν X (A), meaning that for any measurable set E ⊂ R d , we have
P (G i ∈ E | X) = ν X (E).
Then, we consider the following new estimator and define its quadratic error as:
R Aug (λ) := (n + m) -1 {XX + GG } + λ I d -1 , E Aug (λ) := (1/d) R Aug (λ) -Σ -1 X 2 F .
In the following, in addition to H1 and H2, which pertain to X, we introduce further assumptions on G. These are organized into three categories: a concentration assumption on G, a smoothness assumption on ν X , and a stability assumption on ν X . H3 (Concentration of G). The random matrix G ∈ R d×m has i.i.d centered columns conditionally on X, i.e., E[G j |X] = 0 for any j ∈ {1, . . . , m}. In addition, (i) The columns of G are sub-Gaussian, with parameter σ G (ii) There exist 0 ≤ β ≤ 1, and
Λ G : R d×n → R d×d such that almost surely E [C G | X] = βC X + Λ G (X) , and Λ G (X) is a positive definite matrix satisfying for some κ > 0, κ -1 ≤ λ d (Λ G (X)) ≤ λ 1 (Λ G (X)) ≤ κ almost surely on A η defined in (2).
Part (i) of H3 is a concentration assumption on G conditional on X, similar to H1.
Regarding the second part (ii), it can be interpreted as a structural assumption. In most cases, the parameters β and Λ G can be directly derived from the definition of the augmentation process. Table 1 provides values of β and Λ G for a range of common DA schemes. As an example, consider the case where G is drawn from a TDA procedure of the form G j = g(X Ij , Z j ), where {Z j } m j=1 are i.i.d. and 1-Lipschitz concentrated (Definition 1), and (I j ) m j=1 are i.i.d, with
I 1 ∼ Unif({1, • • • , n}), we also Augmentation Name Description ΛG β GDA Fixed Gaussian GDA Gj ∼ N(0, Λ) Λ 0 Gaussian mixture GDA Gj ∼ k i=1 wiN(µi, Λi) k i=1 wi{Λi + µiµ i } 0 TDA Fixed Gaussian TDA XI j + Zj , Zj ∼ N(0, Λ) Λ 1 Random mask TDA XI j Zj , Zj ∼ Ber(ρ) ⊗d ρ(1 -ρ) diag(CX ) ρ Salt & Pepper TDA XI j Zj + (1 -Zj) N(0, σ 2 ) ρ(1 -ρ) diag(CX ) + (1 -ρ)σ 2 I d ρ
Table 1: Various augmentation procedures and corresponding β and Λ G . We used the notation
I j ∼ Unif({1, • • • , n}).
For more details, we refer to Appendix A.
assume that for any e) x for some β (e) ≥ 0. Then, it is straightforward to verify that H3-(ii) is satisfied with β ← β (e) and Λ(X) ← Λ (e) (X) where
x ∈ R d , E Z [g(Z 1 , x)] = β (
Λ (e) (X) = 1 n n i=1 E {g(Z 1 , X i ) -β (e) X i }{g(Z 1 , X i ) -β (e) X i } X i .(5)
Table 1 below, shows the value of β and Λ G (X) for a variety of common data-augmentation scheme.
Our second assumption on G suppose that X → ν X and X → Λ G (X) are Lipschitz. More precisely: H4 (Smoothness of the artificial distribution). There exist L G ≥ 0 and L Λ ≥ 0 such that for any X, Y ∈ R d×n , and m ∈ N,
W 1 (ν ⊗m X , ν ⊗m Y ) ≤ √ mL G X -Y F , Λ G (X) -Λ G (Y) F ≤ L Λ X -Y F .
Note that the DA examples Table 1 all satisfy this assumption provided X has compact support.
Otherwise, we believe that our results are robust enough to hold only when Λ G and X → ν X are locally Lipschitz, albeit with slightly weaker convergence guarantees.
this section cite: []

Section: H5 (Stability of the artificial distribution).
(i) The map X → ν X is invariant under permutation of the columns of X, i.e., for any permutation ς : {1, . . . , n} → {1, . . . , n}, ν X = ν Xς where
X ς = [X ς(1) , . . . , X ς(n) ].
(ii) Furthermore, we assume that there exists K ≥ 0, such that for any m ∈ N,
W 1 (ν ⊗m X , ν ⊗m X -) ≤ √ mK , a.s.
Typically, K should remain bounded with respect to both n and d. H5 can be interpreted as a condition ensuring that the data augmentation procedure used to generate the {G j } m j=1 does not depend on any specific individual sample. It is met by various data augmentation procedures found in the literature. We provide in our next result a condition on ν X and ν X -only, which implies H5-(ii). It proof is postponed to Appendix A. Proposition 2. Suppose that W 2 (ν X , ν X -) ≤ K. Then, H5-(ii) holds.
Remark 1. As a non-trivial example of a DA scheme that satisfies H4 and H5, let us consider the Random mask TDA, described in Table 1. We further illustrate our assumptions on other DA strategies in Appendix A. Let X, Y ∈ R d×n , and consider the coupling of ν X and ν Y , defined as for j ∈ {1, . . . , m}, G j = Z j X Ij , G j = Z j Y Ij , where ⊗d , and is the elementwise multiplication. Then we have by the Cauchy-Schwarz inequality,
I j ∼ Unif({1, • • • , n}), Z j ∼ Ber(ρ)
W 1 (ν ⊗m X , ν ⊗m Y ) ≤ √ mW 2 (ν X , ν Y ) ≤ √ m E [ (X I1 -Y I1 ) Z 1 2 2 ] ≤ √ mρ X -Y F .
Furthermore, from Table 1, we know that Λ G (X) = 1-ρ ρ diag(C X ), therefore it is locally-Lipschitz only. However, assuming that X is bounded, we can always find another function ΛG satisfying H 3-(ii) and which is Lipschitz.
We show through similar computations and using Proposition 2 that H5 is satisfied
W 2 (ν X , ν X -) ≤ E (X I1 -X - I1 ) Z 1 2 2 ≤ ρ n -1 E [ X 1 2 2 ] = ρ n -1 tr (Σ X ) .
We are now ready to introduce our estimate of E Aug (λ). To this end, for any a ≥ 1,
Ra G|X (λ) := (1 -α) C X + α E [C G | X] a + λ I d -1 .(6)
where α = m/(n + m). In addition, we also consider the quantities
a x (X) = 1 + 1 -(1 -β/a g (X))α n X 1 E [R X -G (λ) | X] X 1 , a g (X) = 1 + α m tr (E [C G | X] E [R X G (λ) | X]) ,(7)
and the two functions
Φ 1 (X) = (1 -d/n) d tr R X (0) αΛ G (X) a g (X) + λ I d -1 1 Aη (X) , Φ 2 (X) = 1 -(1 -β/a g (X))α da x (X) tr Rag(X) G|X (λ) αΛ G (X) a g (X) + λ I d -1 ,(8)
Finally, we set
ÊAug (λ) := 1 d tr R Aug (λ) 2 -2(Φ 1 (X) -Φ 2 (X)) + 1 d tr Σ -2 X ,(9)
and we emphasize that ÊAug (λ) is computable from X alone, provided we can sample from the distribution of G conditionally on X. Our main result below states conditions under which ÊAug (λ) concentrates around E Aug (λ), for λ arbitrarily small.
Theorem 2. Assume H1 to H5. Let ÊAug (λ) be defined in (9). Denoting ε = min{η, λ}, for two scalars τ 1 and τ 2 , (also independant of n, d and m, and depending polynomially on ε) defined in (78), it holds
P ÊAug (λ) -E Aug (λ) ≥ t + ∆ Aug n exp -k(n + m) min{ε 9 t 2 /τ 2 , ε 7 t/τ 1 } ,where
∆ Aug := C1 (σ 2 X + σ 2 G )(1 + c -1 X )( Σ X 4 op κ + Σ X op κ 4 ) (1 -α)nλ d (Σ X ) 2 ε 7 + C2 E [ Λ G (X) -E [Λ G (X)] F ] ε 3 √ d + Σ X 2 op √ dε 2 Σ X E [Λ G (X)] -E [Λ G (X)] Σ X F .
and the constants C1 and C2 depend polynomially on λ d (Σ X ), Σ X -1 op , κ -1 , n/m, K, L G and ε.
In the statement above, the three contributions to ∆ Aug are small under natural conditions. The first term decays like n -1 provided the covariance matrices Σ X and Λ G (X) remain well-conditioned and the fraction of artificial samples stays bounded away from one. The second term vanishes if the fluctuations of Λ G (X) are adequately controlled. Finally, the third term is negligible only when E [Λ G (X)] approximately commutes with Σ X , for instance, when the eigenvectors of Σ X are known, when the augmentation is isotropic on average (so that E [Λ G (X)] is a scalar matrix), or more generally when E [Λ G (X)] splits into a low-rank component plus a multiple of the identity (as in Gaussian mixture augmentations with few components relative to d, c.f. table 1).
this section cite: []

Section: Numerical experiments
In this section, we illustrate Theorem 1 and Theorem 2 on real datasets. We use MNIST and CIFAR10, consisting of 70,000 labeled 28 × 28 images and 60,000 labeled 32 × 32 images, respectively, with the following preprocessing:
MNIST. We discard the labels, normalize pixel values to [0, 1], and add pixel-level Gaussian noise with standard deviation σ = 0.1 to ensure that the covariance matrix Σ X is well-conditioned.
this section cite: []

Section: CIFAR10.
We discard the labels and convert images to grayscale.
For both datasets, we denote by X = [X 1 , . . . , X n ] ∈ R d×n the matrix formed by the first n samples, for varying n > 0. To approximate E X (λ) and E Aug (λ), we use the sample covariance matrix ΣX computed from all available samples (70,000 for MNIST, 60,000 for CIFAR10), and consider the proxies
E D X (λ) := 1 d R X (λ) -Σ-1 X 2 F and E D Aug (λ) := 1 d R Aug (λ) -Σ-1 X 2 F ,(10)
which are expected to closely approximate E X (λ) and E Aug (λ) since the sample size greatly exceeds the data dimension.
Figure 1 summarizes our results for MNIST. In particular, figure 1a reports λ → ÊX (λ) for various γ = 784/n over λ ∈ [10 -3 , 1], and compares it with the proxy above. Figure 1b and Figure 1c present ÊAug (0) as a function of α = m/(n + m) under two data-augmentation schemes. The first is a k-centroid Gaussian-mixture GDA,
G j = m Ij (X) + σN(0, I d ),
where the centroids {m i } k i=1 are estimated via EM on X and I j ∼ Unif({1, . . . , k}). The second is a Gaussian-noise TDA, G j = X Ij + σN(0, I d ), with I j ∼ Unif({1, . . . , n}). In both cases, the minimizers of λ → ÊX (λ) and λ → ÊAug (λ) are consistently close to those of the proxies E D X (λ) and E D Aug (λ), which should very closely approximate the true errors.
Symmetrically, for CIFAR10 (after grayscale conversion, so d = 1024), figure 2a reports λ → ÊX (λ) for various γ = 1024/n over λ ∈ [10 -3 , 1] and compares it with the proxy in (10). Figures 2b and 2c present ÊAug (0) as a function of α = m/(n + m) under the same k-centroid Gaussian-mixture GDA and Gaussian-noise TDA schemes as above. In all cases, the minimizers of λ → ÊX (λ) and λ → ÊAug (λ) closely match those of the proxies E D X (λ) and E D Aug (λ).
this section cite: []

Section: Conclusion
In this paper, we established new results based on random matrix theory that allow one to quantify from data only the impact of the regularization effect induced by data augmentation on a common class of precision matrix estimates. In the meantime, we presented a formula that allows one to compute from data only the error of a non-augmented "Ridgelike" precision matrix estimator. From a practical point of view, our results might allow one to optimally tune the hyperparameters of a data augmentation scheme for estimating the bottom eigenvalues and eigenvectors of the covariance matrix of the data, provided the data augmentation scheme satisfies a strict commutativity condition. Furthermore, it is well understood that the precision matrix is a fundamental object in many statistical models; hence, a natural extension of this work would be to study the generalization error of various machine learning models, such as linear regression, kernel regression, or some class of shallow networks.
this section cite: []

Section: References
Ref_id:b0 Title: Isotropic local laws for sample covariance and generalized wigner matrices Year: (2014)
Ref_id:b1 Title: Optimal cleaning for singular values of cross-covariance matrices Year: (2023-04)
Ref_id:b2 Title: Direct shrinkage estimation of large dimensional precision matrix Year: (2016)
Ref_id:b3 Title: Training with noise is equivalent to tikhonov regularization Year: (1995)
Ref_id:b4 Title: Benign overfitting in linear regression Year: (2020)
Ref_id:b5 Title: The sub-gaussian norm of a binary random variable Year: (2013-08-20)
Ref_id:b6 Title: Covariance matrix estimation errors and diagonal loading in adaptive arrays Year: (1988)
Ref_id:b7 Title: A group-theoretic framework for data augmentation Year: (2020)
Ref_id:b8 Title: Quantitative deterministic equivalent of sample covariance matrices with a general dependence structure Year: (2022)
Ref_id:b9 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b10 Title: Synthetic data for portfolios: A throw of the dice will never abolish chance Year: (2025)
Ref_id:b11 Title: A constrained ℓ 1 1 minimization approach to sparse precision matrix estimation Year: (2011)
Ref_id:b12 Title: Learning augmentation strategies from data Year: (2019)
Ref_id:b13 Title: Model collapse demystified: The case of regression Year: (2024)
Ref_id:b14 Title: A kernel theory of modern data augmentation Year: (2019)
Ref_id:b15 Title: A model of double descent for high-dimensional binary linear classification. Information and Inference: A Year: ()
Ref_id:b16 Title: Exact expressions for double descent and implicit regularization via surrogate random design Year: (2020)
Ref_id:b17 Title: Double trouble in double descent: Bias and variance(s) in the lazy regime Year: (2020-07)
Ref_id:b18 Title: A survey of data augmentation approaches for NLP Year: (2021)
Ref_id:b19 Title: An overview of the estimation of large covariance and precision matrices Year: (2016)
Ref_id:b20 Title: Generalisation error in learning with random features and the hidden manifold model Year: (2021)
Ref_id:b21 Title: Bootstrap your own latent-a new approach to self-supervised learning Year: (2020)
Ref_id:b22 Title: Unsupervised representation learning by predicting image rotations Year: (2018)
Ref_id:b23 Title: Data augmentation instead of explicit regularization Year: (2020)
Ref_id:b24 Title: Surprises in high-dimensional ridgeless least squares interpolation Year: (2022)
Ref_id:b25 Title: Analysing multi-task regression via random matrix theory with application to time series forecasting Year: (2024)
Ref_id:b26 Title: Understanding the detrimental class-level effects of data augmentation Year: (2023)
Ref_id:b27 Title: Über die zusammenziehende und lipschitzsche transformationen Year: (1934)
Ref_id:b28 Title: Anisotropic local laws for random matrices Year: (2017)
Ref_id:b29 Title: Concentration of measure and large random matrices with an application to sample covariance matrices Year: (2018)
Ref_id:b30 Title: Spectral properties of sample covariance matrices arising from random matrices with independent non identically distributed columns Year: (2023)
Ref_id:b31 Title: A random matrix analysis of random fourier features: beyond the gaussian kernel, a precise phase transition, and the corresponding double descent Year: (2021)
Ref_id:b32 Title: The good, the bad and the ugly sides of data augmentation: An implicit spectral regularization perspective Year: (2022)
Ref_id:b33 Title: Data augmentation for deep-learning-based electroencephalography Year: (2020)
Ref_id:b34 Title: Eigenvectors of some large sample covariance matrix ensembles Year: (2011)
Ref_id:b35 Title: On robust capon beamforming and diagonal loading Year: (2003)
Ref_id:b36 Title: Improved estimation of the covariance matrix of stock returns with an application to portfolio selection Year: (2003)
Ref_id:b37 Title: A well-conditioned estimator for large-dimensional covariance matrices Year: (2004)
Ref_id:b38 Title: Nonlinear shrinkage estimation of large-dimensional covariance matrices Year: (2012)
Ref_id:b39 Title: The power of (non-) linear shrinking: A review and guide to covariance matrix estimation Year: (2022)
Ref_id:b40 Title: The graphical lasso: New insights and alternatives Year: (2012)
Ref_id:b41 Title: The generalization error of random features regression: Precise asymptotics and the double descent curve Year: (2021)
Ref_id:b42 Title: Data augmentation: A comprehensive survey of modern approaches Year: (2022)
Ref_id:b43 Title: A review: Data preprocessing and data augmentation techniques Year: (2022)
Ref_id:b44 Title: Distribution of eigenvalues for some sets of random matrices Year: (1967)
Ref_id:b45 Title: Harmless interpolation of noisy data in regression Year: (2020)
Ref_id:b46 Title: Deterministic equivalent and error universality of deep random features learning* Year: (2024-10)
Ref_id:b47 Title: Asymptotics of learning with deep structured (random) features Year: (2024-07)
Ref_id:b48 Title: On the eigenvectors of large dimensional sample covariance matrices Year: (1989)
Ref_id:b49 Title: A survey on image data augmentation for deep learning Year: (2019)
Ref_id:b50 Title: Adjustment of an inverse matrix corresponding to a change in one element of a given matrix Year: (1950)
Ref_id:b51 Title: A shrinkage approach to large-scale covariance matrix estimation and implications for functional genomics Year: (2005)
Ref_id:b52 Title: Ai models collapse when trained on recursively generated data Year: (2024)
Ref_id:b53 Title: The art of data augmentation Year: (2001)
Ref_id:b54 Title: High-dimensional probability Year: (2009)
Ref_id:b55 Title: Introduction to the non-asymptotic analysis of random matrices Year: (2011)
Ref_id:b56 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b57 Title: The generalised product moment distribution in samples from a normal multivariate population Year: (1928)
Ref_id:b58 Title: Shrinkage estimation of large dimensional precision matrix using random matrix theory Year: (2015)
Ref_id:b59 Title: On the generalization effects of linear transformations in data augmentation Year: (2020-07)
Ref_id:b60 Title: Model selection and estimation in the gaussian graphical model Year: (2007)
Ref_id:b61 Title: Adversarial autoaugment Year: (2020)
