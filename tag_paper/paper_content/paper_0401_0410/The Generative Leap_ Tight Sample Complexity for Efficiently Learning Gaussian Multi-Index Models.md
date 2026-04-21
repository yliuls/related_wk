Title: The Generative Leap: Tight Sample Complexity for Efficiently Learning Gaussian Multi-Index Models
Abstract: In this work we consider generic Gaussian Multi-index models, in which the labels only depend on the (Gaussian) d-dimensional inputs through their projection onto a low-dimensional r = O d (1) subspace, and we study efficient agnostic estimation procedures for this hidden subspace. We introduce the generative leap exponent, a natural extension of the generative exponent from Damian et al. [2024] to the multiindex setting. We show that a sample complexity of n = Θ(d 1∨k ⋆ /2 ) is necessary in the class of algorithms captured by the Low-Degree-Polynomial framework; and also sufficient, by giving a sequential estimation procedure based on a spectral U-statistic over appropriate Hermite tensors.

Section: Introduction
We consider learning Gaussian multi-index models: Definition 1. We say that (X, Y ) follows a Gaussian multi-index model with index r if X ∼ N (0, I d ) := γ d and there exists a subspace U ⋆ ∈ G(r, d) such that the conditional law P[Y |X] only depends on the orthogonal projection P U ⋆ X.
A Gaussian multi-index model can be thus specified by choosing a basis W ⋆ of U ⋆ (ie, an element of the Stiefel manifold S(r, d)), and the law P of (Z, Y ) ∈ P(R r × R)foot_0 , where Z = (W ⋆ ) ⊤ X. The subspace U ⋆ is referred as the index space.
Given a joint distribution P of (Z, Y ), a natural statistical task associated with such a model is to plant a subspace W ⋆ , uniformly drawn from the Haar measure of S(r, d), and draw n iid samples from the multi-index distribution P W ⋆ ,P parametrized by W ⋆ and P. Our task will be then to recover U ⋆ = span[W ⋆ ] given these samples. We note that this task is only well-posed when the 'intrinsic' dimension of the model is r, namely that P does not admit a factorization P = γ r ′ ⊗ P S , where P S (z S , y) is the marginal of P over a subspace of R r × R of dimension < r + 1 that includes the last coordinate. We will assume this property from now on.
We place ourselves in the setting where r = O d (1), and consider the high-dimensional regime. Since the dimensionality of S(r, d) is of order rd, one expects that a brute-force estimation procedure that fits P P,Wj over a suitable ϵ-net of {W j } j ⊂ S(r, d) requires O(dϵ -2 ) samples to estimate the index space up to accuracy ϵ. Our main motivation is to understand this question from the lens of computational-statistical gaps: how many samples are needed, as a function of d, P, to produce an estimate of the planted subspace using polynomial-time algorithms, as opposed to using brute-force? This question enjoys a large literature, spanning high-dimensional statistics and learning theory, starting from the inverse regression methods from Li [1991] and beyond Xia [2008], Xia et al. [2002], Hristache et al. [2001], Cook and Li [2002], Cook [2000], Vempala [2010], Klivans et al. [2008], Mossel et al. [2003], Daniely et al. [2025] (see also Bruna and Hsu [2025] for a recent survey), where efficient algorithms have been developed for specific instances. Multi-index models are an appealing semiparametric model, and provide arguably the simplest instance of linear feature learning, in the sense that the index space provides an adapted low-dimensional representation to perform high-dimensional learning. Some notorious examples include • (noisy) Gaussian parity: Y |Z d = ξ sign[Z 1 • Z 2 . . . Z r ], with P[ξ = -1] = η, P[ξ = 1] = 1 -η independent of Z and η < 1/2.
• Gaussian staircase functions: Y |Z d = ϕ 1 (Z 1 ) + ϕ 2 (Z 1 , Z 2 ) + . . . + ϕ r (Z 1 , . . . , Z r ).
• Intersection of r half-spaces:
Y |Z d = 2 r j=1 1(v ⊤ j Z > α j ) -1.
• Low-rank shallow neural network: Y |Z = a ⊤ ρ(V ⊤ Z) + ξ for some a ∈ R M , V ∈ R r×M , ρ : R → R, additive noise ξ independent of Z.
• Polynomials: Y |Z = q(Z) where q is a polynomial.
Focusing on the Gaussian setting, several works, starting from Dudeja and Hsu [2018], Ben Arous et al. [2021] and followed by Abbe et al. [2021Abbe et al. [ , 2023]], Bietti et al. [2022], Damian et al. [2022], Ba et al. [2022], Dandi et al. [2024a] have built a harmonic analysis framework to analyze a large class of algorithms, including stochastic gradient descent over NN architectures, leading to sample complexities of the form n = Θ(d k ), where k is an explicit exponent associated with a certain harmonic expansion of P. In particular, Damian et al. [2024], focusing on Single-Index models (where r = 1), identified the generative exponent k ⋆ = k ⋆ (P) (see Section 2) as the fundamental quantity driving the sample complexity, in the sense that n = Θ(d 1∨k ⋆ /2 ) is both necessary and sufficient in the class of algorithms implemented by SQ (Statistical Queries) and Low-Degree Polynomials. In essence, the generative exponent arises from an expansion of the inverse regression of Z given Y , as put forward in the original Li [1991]. Lee et al. [2024], Arnaboldi et al. [2024], Dandi et al. [2024b] showed that SGD with reused samples can learn single index models dependent on the generative exponent, instead of the information exponent.
In this work, we extend this notion of generative exponent to the general multi-index setting. As already pointed out in the literature Abbe et al. [2023], Bietti et al. [2025], Troiani et al. [2024], Diakonikolas et al. [2025b], the general r > 1 setting gives rise to important new phenomena not present in the single-index case. In particular, gradient-based learning exhibits a sequential behavior in the form of saddle-to-saddle dynamics, where the index space is revealed incrementally along specific subspaces, with different timescales associated with each step. Moreover, such incremental alignment requires solving a semi-parametric problem, where both the subspace and the link function need to be estimated jointly. We overcome these additional challenges by identifying a suitable generalization of the generative exponent, the leap generative exponent k ⋆ (see Definition 3), arising from a 'canonical' orthogonal decomposition of the index space, the leap decomposition (see Section 2).
We first show that this exponent provides a computational lower bound of n = O(d k ⋆ /2 ) under the Low-degree polynomial (LDP) framework, by extending the previously established lower bound in the single-index setting Damian et al. [2024] to an appropriate detection task that is dominated by the index estimation task (Theorem 1). Next, and more importantly, we provide an algorithm that sequentially estimates the index space along the leap decomposition from the spectrum of a novel kernel U -statistic (see Eq ( 5)). This algorithm recovers the index space as soon as n ≳ d k ⋆ /2 , thus matching the LDP lower bound, and, crucially, it does not require prior knowledge of the multi-index model P (Theorem 3). We complement these general results by several case studies that give novel guarantees on specific multi-index models, such as general ReLU networks or Gaussian Parities; see Section 5. Taken together, our results therefore provide the correct, sharp dimension dependence for any (Gaussian) multi-index model. In particular, as soon as k ⋆ > 2, they provide evidence of a computational-to-statistical gap at the polynomial scale.
Related Works Chen and Meka [2020] show that any polynomial multi-index model can be learned with n ≳ d samples via an iterated filtered PCA algorithm. Chen et al. [2022] extended this to the case of multi-index models with ReLU activation with a similar algorithm. As we will show in Section 5, the generative exponent satisfies k ⋆ ≤ 2, so our algorithm also requires only n ≳ d.
Gradient descent on two-layer networks has been extensively studied [Bietti et al., 2025, Ren and Lee, 2024, Ren et al., 2025, Damian et al., 2022, Abbe et al., 2023], these papers typically require at least n = Θ(d 1∨l ⋆ -1 ), where l ⋆ is the information exponent [Arous et al., 2021], an upper bound of the generative exponent. Abbe et al. [2023] provide a similar definition of leap exponent but tailored to the information exponent, and thus larger than the generative leap exponent. In the setting of sparse juntas, Joshi et al. [2024] showed that by changing the loss function from square loss to another loss, gradient queries learn with complexity governed by the SQ-exponent, which is analagous to the generative exponent but restricted to juntas. Troiani et al. [2024] characterize the generative leap exponent for leaps ≤ 2. Defilippis et al. [2025], Kovačević et al. [2025] give spectral estimators for the special case when the subspace is fully identified in the first leap of generative exponent ≤ 2. See Section 2 for further discussion.
Tensor PCA. In the context of Tensor PCA, Montanari and Richard [2014] proposed the Tensor PCA model and presented several algorithms including tensor unfolding. Zheng and Tomioka [2015] proposed a rectangular unfolding algorithm closely related to a single step of our algorithm, and showed it attain the conjectured optimal sample complexity of n = Θ(d 1∨k ⋆ /2 ). Dudeja and Hsu [2021] provided statistical query lower bounds for the symmetric and asymmetric Tensor PCA model, and Hopkins et al. [2015Hopkins et al. [ , 2017] ] gave the corresponding lower bound in the low-degree / SOS models. Dudeja and Hsu [2024] provided a comprehensive study of communication lower bounds and efficient algorithms for Tensor PCA and the related problem of Non-Gaussian Component Analysis. Arous et al. [2024] initiated the study of stochastic gradient descent over the Stiefel manifold for the multi-spike Tensor PCA model, showing a time complexity of d 1∨k ⋆ -1 where k ⋆ is the order of the tensor (analogous to the generative exponent). Chen et al. [2020] show that deep neural networks can simulate unfolding-like algorithms and learn multi-index functions with n = Θ(d ⌈k/2⌉ ) where k is the degree of the polynomial approximation to the groundtruth function. This method requires that the groundtruth is close to a polynomial.
Vempala [2010], Klivans et al. [2024] provides the current best known result for learning intersection of k-halfspaces in d-dimensions with n = Θ(d). Vempala and Xiao [2012] provide a moment-based algorithm for learning multi-index models when the first leap learns all relevant variables.
While this work was being finalized, we became aware of Diakonikolas et al. [2025b,a], which introduces a similar estimation procedure based on subspace conditioning. They define the class of m-well-behaved multi-index models. For the special case of single index models with generative exponent k ⋆ , Diakonikolas et al. [2025b][Appendix D.2] and Diakonikolas et al. [2025a][Appendix C.3.2] show m = k ⋆ ; we believe a similar equivalence holds also for multi-index models. However the proposed algorithm requires sample complexity n = d O(k ⋆ ) even in the realizable setting, whereas the algorithm of Damian et al. [2024] and this work, require only n = Θ(d k ⋆ /2 ) and apply when y is either continuous or discrete. On the other hand, Diakonikolas et al. [2025b,a] algorithms aim for agnostic PAC learning, not just recovery of the subspace, and thus are able to explicitly characterize the dependence in the hidden constant C(P) in n ≥ C(P)d 1∨k ⋆ /2 . By building an explicit piecewise constant discretization in the subspace, they explicitly characterize the dependence on r, ϵ, and Lipschitz parameters. We expect that our subspace recovery algorithm can be combined with a discretization algorithm to attain similar guarantees, but with improved dependence on d. We also study several examples of the leap generative exponents in Section 5 including piecewise linear functions (deep ReLU Networks with bias) and general deep neural networks with r-dimensional first hidden layer, improving upon previous results specific to multi-index polynomials and homogeneous piecewise linear functions Chen and Meka [2020], Chen et al. [2022].
Notation h k denotes the normalized k-th Hermite tensor, defined as h k (u) := (-1) k √ k!
∇ ⊗k γ d (u) γ d (u)
for u ∈ R d . S(r, d) is the Stiefiel manifold of r × d orthogonal matrices, and G(r, d) is the Grassman manifold, obtained by quotienting S(r, d) by r-dimensional basis transformations. For two subspaces T ⊆ T ′ , we write T ′ \ T as the orthogonal complement of T in T ′ . For two subspaces T, T ′ we define their distance d(T, T ′ ) to be ∥Π T -Π T ′ ∥ op where Π T is the orthogonal projection onto T .
Paper outline. Section 2 recalls the generative exponent for single-index models and shows how to generalize the definition to multi-index models via the leap generative exponent, and discuss the relation to the leap information exponent. Section 3 gives the main computational lower bound result which shows that in the low-degree polynomial framework the multi-index model with leap generative exponent k ⋆ requires n ≥ Ω(d k ⋆ /foot_2 ). Section 4 gives our main algorithm, an iterative spectral method based on a Hermite Kernel U-statistic, that recovers the subspace with the optimal sample complexity. Finally in Section 5, we study the leap generative exponents of several function classes including piecewise linear functions and general neural networks with r-dimensional first hidden layer.
this section cite: ['b46', 'b56', 'b57', 'b39', 'b20', 'b19', 'b54', 'b42', 'b49', 'b14', 'b33', 'b9', 'b0', 'b2', 'b10', 'b21', 'b6', 'b23', 'b46', 'b45', 'b3', 'b2', 'b11', 'b53', 'b23', 'b16', 'b17', 'b11', 'b50', 'b21', 'b2', 'b4', 'b2', 'b40', 'b53', 'b28', 'b43', 'b48', 'b58', 'b37', 'b38', 'b35', 'b5', 'b41', 'b55', 'b23', 'b16', 'b17']

Section: The Leap Decomposition
Preliminaries: The Generative Exponent for Single-Index Models We start by recalling the generative exponent for single-index models Damian et al. [2024]. Given (Z, Y ) drawn from a joint distribution P ∈ P(R × R) with first marginal equal to a Gaussian, and such that P ̸ = γ 1 ⊗ P y , we define for each integer k ≥ 1,
ζ k := E[h k (Z)|Y ] ∈ L 2 (P y ), and k ⋆ = inf{k; ∥ζ k ∥ L 2 (Py) > 0}.
Equivalently, k ⋆ is the smallest integer k such that there exists a measurable function T : R → R and a mean-zero k-th degree polynomial q such that E[T (Y )q(Z)] ̸ = 0. The main takeaway from Damian et al. [2024] is that n = Θ(d k ⋆ /2∨1 ) is both necessary (under the SQ and the LDP frameworks) and sufficient for recovery of the planted direction 2 .
Subspace Filtration and Leap exponents: We begin by generalizing the coefficients {ζ k } from [Damian et al., 2024]. The key novel ingredient is the notion of subspace filtration, capturing the sequential nature of the multi-index estimation, and which appears in several existing multi-index estimation procedures Abbe et al. [2022Abbe et al. [ , 2023]], Bietti et al. [2025], Diakonikolas et al. [2025b]. In essence, we now need to extend the expectations ζ k , which were conditional on the label y, to conditional expectations on an 'augmented label' that includes all the previously estimated directions of the index space. More formally, let S ∈ G(r ′ , r) be a subspace, and for z ∈ R r let z S ∈ S denote the orthogonal projection of z onto S. We write ȳS := (z S , y) ∈ R 1+r ′ and zS := z S ⊥ ∈ R r-r ′ .
For any S ∈ G(r ′ , r), we then define:
ζ k,S := E[h k ( ZS )| ȲS ] ∈ L 2 (R 1+r ′ , (S ⊥ ) ⊗k , P ȲS ) , Λ k (S) := E ȲS [ζ k,S ( ȲS ) ⊗ ζ k,S ( ȲS )] ∈ (S ⊥ ) ⊗2k , λ 2 k (S) := E ȲS [ ζ k,S ( ȲS ) 2 F ]
. Intuitively, these tensors capture whether there is any information "of order k" that can be captured, given knowledge of the subspace S. When S = ∅ and r = 1, these definitions reduce to those in Damian et al. [2024]. Finally, we note that these definitions only depend on the joint distribution P of (Z, Y ) and are independent of the choice of W ⋆ .
Given a subspace S, we define the associated null distribution P S by:
dP S [Z, Y ] = dP[ ZS ]dP[Z S , Y ] .
Under P S , (Y, Z S ) and ZS have the same marginals as under P, but are independent. The label transformations ζ k appear as the Hermite coefficients of the density ratio dP dP S : Lemma 1 (Density Ratio expansion). We have the following formal expansion in L 2 (P S ):
dP dP S [Z, Y ] = k≥0 h k ( ZS ), ζ k (Y ; Z S ) .
This implies the following decomposition of χ 2 (P||P S ), whenever this divergence exists:
Lemma 2 (Mutual Information Expansion). If χ 2 (P||P S ) < ∞, χ 2 (P||P S ) = k≥1 λ 2 k (S).
Notice that while χ 2 (P||P S ) may be infinite in some cases, e.g. in deterministic models where Y = σ(Z), the quantities λ k (S) are well-defined for all k, since ζ k,S ∈ L 2 (P ȲS )foot_3 . Given this expansion, we can immediately define the leap k(S) of a subset S: Definition 2 (Generative Leap relative to S). k(S) is the smallest k ≥ 1 such that λ 2 k (S) > 0.
Note that k(S) < ∞ so long as P ̸ = P S .
The Leap Decomposition: We will define the flag
F = {∅ = S 0 ⊊ S 1 ⊊ • • • ⊊ S L = R r } inductively as follows.
Given a subspace S i , i ≥ 0, we define k i+1 := k(S i ) and S i+1 by:
S i+1 := S i ⊕ span[Λ ki+1 (S i )].(1)
Here, we have defined the span of a symmetric tensor T ∈ (R r ) ⊗k as span(T ) = span Mat r,r k-1 [T ] where Mat r,r k-1 [T ] denotes T reshaped as an r × r k-1 matrix.
Definition 3 (Generative Leap Exponent). Let k i , i = 1, . . . , L be defined as above. The generative leap exponent is defined as k ⋆ := max i k i .
We now verify that the Leap decomposition is well-defined, and give a variational representation.
Definition 4. Given two subspaces S ⊊ T , we define the relative leap k(S, T ) of a subspace S towards T as
k(S, T ) := inf{k; T \ S ⊆ ⊕ k ′ ≤k span(Λ k ′ (S))} .(2)
In words, the relative leap measures the order of the Hermite tensor needed to 'reach' the subspace T from conditional expectations over y and z S . Observe that we can relate the leaps k(S) and k(S, T ) as k(S) = inf T ;S⊊T k(S, T ). Proposition 1 (Variational Characterization of Leap Generative Exponent). The leap decomposition terminates in a finite number of steps L ≤ r. Moreover, we have
k ⋆ = inf F ={∅=R0⊂•••⊂R r } max j k(R j , R j+1 ) .(3)
Finally, k ⋆ is invariant to rotation: if P = (U ⊗ Id) # P where U ∈ O r is any rotation of the model, we have k ⋆ ( P) = k ⋆ (P).
Relationship with Information Leap Exponent Finally, we relate the generative leap exponent to the information leap exponent, first introduced in Abbe et al. [2023] (referred to as IsoLeap in the setting of Gaussian input data); see also Bietti et al. [2025] and Dandi et al. [2024a]. Let us first recall its definition in our context. For any S ∈ G(r ′ , r), we define:
ζk,S := E[Y h k ( ZS )|Z S ] ∈ L 2 (R 1+r ′ , (S ⊥ ) ⊗k , P Z S ) , Λk (S) := E Z S [ ζk,S (Z S ) ⊗ ζk,S (Z S )] ∈ (S ⊥ ) ⊗2k , λ2 k (S) := E Z S ζk,S (Z S ) 2 F .
By analogy with Definition 2, we define l(S) to be the smallest k such that λ2 k (S) > 0. Equipped with this object, the information leap exponent is recovered as follows.
Definition 5 (Information Leap Exponent, Abbe et al. [2023], Bietti et al. [2025]). The information leap exponent of the multi-index model P is given by l ⋆ := max i l i , where l i+1 = l( Si ) and ( Si ) i is defined recursively by S0 = ∅ and Si+1 = Si ⊕ span[ Λli+1 ( Si )].
Let us now relate the Information Leap exponent to the generative leap. We start with a direct generalization of [Damian et al., 2024, Prop 2.6]: Proposition 2 (Generative and Information Exponents relative to subspaces). For any subspace S, k(S)[P] = inf
T ∈L 2 (Pȳ S ) l(S)[(Id z ⊗ T y ) # P] .(4)
In particular we have k(S) ≤ l(S) for any subspace S.
In words, the generative exponent relative to a subspace S is the largest k such that E P [T (y, z S )q(z S ⊥ )] = 0 for any measurable function T and any polynomial q of degree < k. This provides a useful characterization, as illustrated in the examples of Section 5.
As expected, the generative leap is upper bounded by the information leap: Proposition 3 (Relationship with Leap Information Exponent). We have k ⋆ ≤ l ⋆ .
Proofs of these results are deferred to Appendix B.
this section cite: ['b23', 'b23', 'b23', 'b1', 'b2', 'b11', 'b23', 'b2', 'b11', 'b2', 'b11', 'b23']

Section: Computational Lower Bounds in the Low-Degree Polynomial Class
Let us first establish a computational lower bound for the estimation of a multi-index model. Following Damian et al. [2024], and relying on the fact that detecting planted structure is a necessary byproduct of estimating the index space, we instantiate a hypothesis testing adapted to the leap decomposition.
Given P and its associated leap decomposition (1), we consider S the subspace of dimension r 0 associated with the generative leap, ie k ⋆ = k( S). Let ȳ = ( Sz, y) be the effective label, with ȳ ∈ R r0+1 , W = S⊤ W ⋆ the planted subspace associated with S, and x = W ⊥ x ∈ R d-r0 the effective input. Viewing P as the joint distribution of ( S⊥ z, ȳ), we define P ȳ as the marginal over ȳ.
Note that r 0 < r by definition. We consider the following detection problem, conditional on W :
• H 1 : there is a planted model of dimension r > r 0 using P as link function and ȳ as label. Specifically, (x, ȳ) ∼ E W P W , where P W (ȳ|x) = P(ȳ| W ⊤ x).
• H 0 : there is only planted structure up to dimension r 0 ; i.e., (x, ȳ) ∼ P 0 := γ d-r0 ⊗ P ȳ .
By considering the likelihood ratio R = dH1 dH0 and its orthogonal projection R ≤D in L 2 (H 0 ) onto polynomials of degree at most D, one can assess the ability of low-degree polynomials to solve this hypothesis testing problem Bandeira et al. [2022], Hopkins [2018]. [Bandeira et al., 2022, Proposition 6.2].
Specifically, if ∥R ≤D ∥ L 2 (H0) = 1 + o d (1), then no degree-D polynomial f in the input samples can weakly separate H 0 from H 1 , ie satisfy max{Var 0 [f ], Var 1 [f ]} = O(| E 0 [f ] -E 1 [f ]| 2 ) as d → ∞
Theorem 1 (Weak separation lower bound). Consider d ≫ max(r, k ⋆ ), D = O(log(d)
2 ), and
n = O(d k ⋆ /2-γ ) for any γ > 0. Then ∥R ≤D ∥ L 2 (H0) = 1 + o d (1).
In other words, any degree-D polynomial test needs n ≥ Ω(d k ⋆ /2 ) samples to weakly detect H 1 from H 0 . Polynomial tests of degree ω(log d) are considered a powerful step towards ruling out all noise-tolerant polynomial-time algorithms Bandeira et al. [2022], Kunisky et al. [2019]. The proof can be found in Appendix C. This low-degree lower bound extends the previous LDP lower-bound from the single-index setting Damian et al. [2024]. In that single-index setting, this LDP lower bound agrees with a SQ lower bound of n = Θ(d k ⋆ /2 ) samples. While it is possible to translate our LDP lower bounds to SQ lower bounds, eg via Brennan et al. [2021], we note that there is a fundamental distinction arising in the multi-index setting, stemming from the inherent inability to perform certain spectral tasks in SQ. Dudeja and Hsu [2021] illustrated this mismatch in the setting of Tensor PCA, where asymmetric structures (such as the ones faced by multi-index model estimation) incur in additional dimension-factors. That said, some SQ lower bounds are known for the multi-index setting. Joshi et al. [2024] establishes SQ lower bounds for the number of queries of order Θ(d k ⋆ ), and Diakonikolas et al. [2025b,a] obtains sample complexity lower bounds of order Θ(d k ⋆ /2 ) (where the generative leap is replaced by the equivalent m in their notation), thus matching our LDP lower bounds.
this section cite: ['b23', 'b7', 'b36', 'b7', 'b44', 'b23', 'b13', 'b34', 'b40']

Section: Upper Bound via Hermite Kernel U-Statistic
We begin by describing a spectral estimator that works for a single leap. To motivate it, recall that the spectral estimator for single index models in Damian et al. [2024] began by estimating the tensor:
T = E X,Y [T (Y )h k (x)].
For a suitable label transformation T , the true expectation is proportional to (w ⋆ ) ⊗k , so estimating w ⋆ is similar to a single-spike tensor PCA problem. For this problem, the partial trace estimator is an effective way to estimate w ⋆ . This estimator consists in repeatedly contracting indices T ← T [I] until you are left with a vector whose expectation is w ⋆ or a matrix whose expectation is w ⋆ (w ⋆ ) ⊤ . However, this trick does not work in the multi-index setting. For example, consider Gaussian k-parity: y = sign(z 1 • • • z k ). For this problem, we can compute the population mean of an order k estimator:foot_4
T = E[Y h k (X)] = 2 π k/2 √ k! Sym(w ⋆ 1 ⊗ • • • ⊗ w ⋆ r
). Thus, this behaves like a symmetric multi-spike tensor PCA problem. For this problem, note that because the {w ⋆ i } are mutually orthogonal, T [I] = 0 so taking any partial traces of this tensor will fail to produce a consistent estimator. For standard tensor PCA, this can be solved by tensor unfolding Montanari and Richard [2014]. For example, Zheng and Tomioka [2015] showed it was sufficient to unfold T into a d × d k-1 matrix and compute the left singular vectors. Explicitly if A = Mat (d,d k-1 ) [T ] denotes T reshaped as a d × d k-1 matrix, then you can perform a spectral decomposition of AA ⊤ ∈ R d×d and the top eigenvectors will recover the hidden directions.
Returning to the multi-index setting, this would motivate the following estimator. Given n samples {(x i , y i )} n i=1 , we define the embedding ϕ, the flattened tensor Φ and the matrix estimator M n by:
ϕ(x) := Mat (d,d k-1 ) [h k (x)], Φ = 1 n n i=1 T (y i )ϕ(x i ) ∈ R d×d k-1 , M n := ΦΦ ⊤ ∈ R d×d .
We can then perform a spectral decomposition of M n . Note that this is exactly equivalent to estimating the tensor
1 n n i=1 T (y i )h k (x i ),
unfolding it into a d × d k-1 matrix, and computing its left singular vectors. However, this strategy cannot achieve the optimal threshold of n ≳ d k 2 because the "diagonal" terms dominate the matrix and destroy the concentration. More specifically, we can expand M n as:
M n = 1 n 2 i,j T (y i )T (y j )ϕ(x i )ϕ(x j ) ⊤ = 1 n 2 i T (y i ) 2 ϕ(x i )ϕ(x i ) ⊤ (I) + 1 n 2 i̸ =j T (y i )T (y j )ϕ(x i )ϕ(x j ) ⊤ (II)
.
For this estimator, one can show that the spikes in E M n get lost in the bulk of the eigenvalues corresponding to (I) unless n ≳ d 1∨ 2k-1 3 , which falls short of the optimal threshold d 1∨ k 2 . To improve this estimator, we therefore isolate the second term (II):
U n = 1 n(n -1) i̸ =j T (y i )T (y j )ϕ(x i )ϕ(x j ) ⊤ .
This is an order 2 matrix U -statistic which only sums over the disjoint pairs i ̸ = j. As a result the expectation is preserved: E U n = E Φ E Φ ⊤ and we prove that U n does concentrate to its expectation in operator norm with n ≳ d k/2 samples (Theorem 2). However, it is not true in general that a single label transformation T is enough for E U n to span the entire space when there are multiple leaps i.e. it may be necessary to use a label transformation T 1 to estimate the first direction w ⋆ 1 and T 2 to estimate w ⋆ 2 . Rather than computing the top eigenvector of this matrix U -statistic for each label transformation T i , we could simply add them together into T (Y ) = [T 1 (Y ), . . . , T m (Y )] ∈ R m and form an aggregate matrix:
U n = 1 n(n -1) i̸ =j ϕ(x i )ϕ(x j ) ⊤ ⟨T (y i ), T (y j )⟩ .
Because T only enters the U -statistic through inner products, we can use the kernel trick and replace it with a general PSD kernel K:
U n = 1 n(n -1) i̸ =j ϕ(x i )ϕ(x j ) ⊤ K(y i , y j ) ,(5)
Algorithm 1: A Single Leap Input: dataset D = {(xi, yi)} n i=1 , moment k, recovery dimension s, PSD Kernel K ϕi ← Mat d×d k-1 [h k (xi)] for i = 1, . . . , n Un ← 1 n(n-1) i̸ =j ϕiϕ ⊤ j K(yi, yj) [S, V ] ← eig(Un) Output: span[v1, . . . , vs]
which reduces to the above setting by taking K(y i , y j ) = ⟨T (y i ), T (y j )⟩. However, by allowing more general kernels K which correspond to "infinite" embedding vectors T , this allows to automatically average over an "infinite number" of label transformations. We will show that this allows us to learn the subspace corresponding to the next leap with the optimal sample complexity of n ≳ d k 2 without any knowledge of the multi-index model P. To begin, we prove the following lemma which controls the expectation of this matrix U -statistic: Lemma 3. If K is integrally strictly positive definite,foot_5 there exist c(P, K), C(P, K) > 0 independent of d such that if S := (U ⋆ ) ⊤ span[Λ k ] denotes the subspace corresponding to the next leap then
c(P, K)Π S ⪯ E U n ⪯ C(P, K)Π S .
We note that commonly used kernels like the RBF kernel automatically satisfy the assumption in Lemma 3. This implies that if we could estimate the span of E U n , we could recover the next leap. To estimate the span, we use the following theorem which bounds
U n -E U n in operator norm: Theorem 2 (Concentration of U-Statistic). Let K be a PSD kernel with K(y, y) ≤ 1 for all y. Then if n ≳ k d k/2 /ϵ + dr k /ϵ 2 , we have that ∥U n -E U n ∥ op ≤ ϵ with probability at least 1 -exp(-d c ) for an absolute constant c > 0.
As a corollary, by Davis-Kahan we can recover the subspace up to error ϵ with n ≳ d k/2 /ϵ + d/ϵ 2 samples where the hidden constant is independent of d and depends only on the multi-index model P: Corollary 1 (Subspace Recovery). For any multi-index model P, there exists a constant C(P, K)
independent of d such that if n ≥ C(P, K) d k/2 ϵ + d ϵ 2 then the output S ⊂ R d of Algorithm 1 satisfies d(S, (U ⋆ ) T span[Λ k ]
) ≤ ϵ with probability at least 1 -exp(-d c ) for an absolute constant c > 0.
this section cite: ['b23', 'b48', 'b58']

Section: Iterating over Leaps
Once we have recovered an partial subspace S, which we hope is approximately contained in span[(U ⋆ ) ⊤ ], we need to continue this process to take the next leap. We can consider the augmented label ȲS = (Y, Π S x). Then X, Ȳ again form a multi-index model with hidden dimension at most r so we can repeat our matrix U-statistic estimator from the previous section. Note that the kernel K now maps R |S|+1 × R |S|+1 → R. We will denote the resulting kernel by U (S) n :
U (S) n := 1 n(n -1) n i=1 ϕ i ϕ ⊤ j K([y i , Π S x i ], [y j , Π S x j ]).
We can directly apply Corollary 1 to show that for any subspace S, we can recover the span of (U ⋆ ) T Λ k (S) up to error ϵ with n ≳ d k/2 /ϵ + d/ϵ 2 samples. We will now control the accumulation of errors to show that we can recover the full multi-index model with n ≳ C(P, K)d k ⋆ /2 /ϵ samples: Lemma 4. If the kernel K is L-Lipschitz, then there exists a constant C(P, K) such that the map
S → E U (S) n is C(P, K)L-Lipschitz in operator norm.
A common example of a Lipschitz kernel is the RBF kernel which is 1/σ-Lipschitz. Therefore if we run this estimator starting with the wrong subspace Ŝ with d(S, Ŝ) ≤ ϵ, then the span of our estimator
Algorithm 2: Iterating over Leaps
Input: dataset D = {(xi, yi)} n i=1 , moments {ki} m i=1 , subspace dimensions {si} m i=1 , Kernels {Ki} m i=1 S ← ∅ for i = 1, ..., m do Draw ⌊n/m⌋ fresh samples Di from D y ← [y, ΠSx] ∈ R |S|+1 for (x, y) ∈ Di S ← S⊕ Algorithm 1(Di,ki,si,Ki) end Output: S
can only change by C(P, K)Lϵ. By iterating this argument, Theorem 2 implies that Algorithm 2 will succeed in recovering span[U ⋆⊤ ] up to error ϵ given n ≳ d k ⋆ /2 /ϵ + d/ϵ 2 samples: Theorem 3 (Main Result). For any multi-index model P, there exists a constant C(P, K) independent
of d such that if n ≥ C(P, K) d k ⋆ /2 ϵ + d ϵ 2 then the output S ⊂ R d of Algorithm 2 satisfies d(S, span (U ⋆ ) ⊤ ) ≤ ϵ with probability at least 1 -exp(-d c ) for some c = c(k ⋆ ) > 0.
Remark 4. Our main upper bound, Algorithm 2, requires knowledge of the sizes of each leap {k i } and the dimension of each leap {s i }. However, these restrictions can be easily lifted, in the spirit of Dudeja and Hsu [2018]. Using the guarantee in Theorem 2, we could start with k = 1 for each leap and increase k until we detect outlier eigenvalues outside of the d k/2 /n-bulk. However, for simplicity we have written the algorithm assuming knowledge of both {k i } and {s i }.
Our Algorithm 2 is thus a streamlined version of a subspace conditioned spectral method. While it shares similarities with recent methods in the literature Chen and Meka [2020], Chen et al. [2022], Diakonikolas et al. [2025b,a], Troiani et al. [2024], it crucially relies on a U-statistic in order to reach the optimal sample complexity of d k ⋆ /2 . An additional feature of our algorithm -that to our knowledge is novel in the literature -is the use of a generic kernel over the already discovered labels, which eliminates the need to perform successive non-parametric regressions during the subspace recovery. At the technical level, the concentration of the U-statistic is a priori challenging due to the heavy tails of the associated Hermite tensors; this is addressed using Gaussian universality results from Brailovskaya and van Handel Brailovskaya and van Handel [2024], with a dedicated analysis in the setting where k ⋆ ≤ 2 to avoid spurious log-factors.
this section cite: ['b33', 'b16', 'b17', 'b53', 'b12']

Section: Case Studies
We conclude this article by computing the generative leap exponent of representative multi-index models. For some of these models our upper and lower bounds recover known results in the literature, but some are new. For simplicity, we focus here on noiseless models where Y |Z = σ(Z) for a given link function σ : R r → R. Proofs for this section can be found in Appendix E.
this section cite: []

Section: Polynomial and Threshold Functions
We start by computing the generative leap for 'classic' multi-index classes given by parities, intersection of half-spaces and polynomials. Proposition 4 (Generative Leaps for representative models). We have:
(i) r-Gaussian Parity has k ⋆ = l ⋆ = r. (ii) Staircase Parity functions have k ⋆ ≤ l ⋆ = 1, (iii) Intersection of halfspaces have k ⋆ ≤ 2, (iv) Polynomials have k ⋆ ≤ 2.
For r-Gaussian parity, we thus obtain an efficient learning algorithm that requires n = Θ(d r/2 ) samples (which is optimal within the LDP class), and is to the best of our knowledge the first result 6 that succeeds with Θ(d r/2 ) samples. The sample complexity of learning intersection of half-spaces is thus linear in dimension: this was known since Vempala [2010], Diakonikolas et al. [2017], Klivans et al. [2024], and for polynomials the same conclusion was established in Chen and Meka [2020]. We emphasize that while our results do capture the correct dependency in d, they are not fine-grained enough to provide the correct dependencies in r.
this section cite: ['b54', 'b30', 'b41', 'b16']

Section: Piecewise Linear Functions
Piecewise linear continuous functions, in part motivated by ReLU architectures, have been extensively studied in the context of Gaussian Multi-index models Chen et al. [2022Chen et al. [ , 2023]], Diakonikolas and Kane [2024]. When σ is 1-homogeneous, as in bias-free ReLU networks, it is not hard to see that k ⋆ ≤ 2, by considering diverging level sets {z; |σ(z)| ≥ λ} with λ → ∞. Here we extend this result to the general piece-wise linear setting, including arbitrary ReLU networks with non-zero biases. Proposition 5 (Generative Leap for Piecewise Linear Functions). If σ is continuous and piece-wise linear then k ⋆ (σ) ≤ 2.
The proof exploits the analytic properties of Hermite functions, i.e. functions of the form f (z) = p(z)γ(z). As an immediate corollary, our Algorithm from Section 4 learns arbitrary ReLU networks low-rank arbitrary ReLU networks in the proportional regime n = Θ(d): This improves the result of Chen et al. [2022] by allowing biases. Once the subspace is recovered, one could 'upgrade' to PAC learning the model using a standard non-parametric method, by regressing over the covariates z = S ⊤ x. This would incur in an additional sample complexity with potentially exponential dependencies in r and 1 ϵ , but, importantly, independent of d.
this section cite: ['b17', 'b18', 'b29', 'b17']

Section: Generative Leap under Linear Transformations
An important feature of the generative leap exponent is that the statement "k ⋆ (P) ≤ k" is an 'open' property, meaning that one should expect the leap exponent to be preserved or reduced by slightly perturbing the distribution P. We formalize this intuition in the following result which shows that for almost all weight matrices, the generative leap is ≤ 2. Proposition 6 (Generative Leap under linear transformations). Let σ(z) : R r → R ∈ L 2 (γ r ), σ ̸ = C, and let M r denote the set of r × r real matrices.
(i) For Θ ∈ M r , define y Θ = σ(Θ ⊤ z). Then (z, y Θ ) ∼ P Θ satisfies k ⋆ (P Θ ) ≤ 2 for every Θ, except possibly for a set of r 2 -dimensional Lebesgue measure zero, (ii) Assume that (z, σ(z)) ∼ P has a single leap with generative exponent k ⋆ . Let Γ : D ⊆ R s → M r be any analytic map such that I r ∈ Im(Γ) and Γ(θ) is invertible for all θ ∈ D. For θ ∈ D, define y θ = σ(Γ(θ) ⊤ z). Then (z, y θ ) ∼ P θ satisfies k ⋆ (P θ ) ≤ k ⋆ for every θ, except possibly for a set of s-dimensional Lebesgue measure zero.
this section cite: []

Section: Shallow Neural Networks
Finally, we study two layer neural networks of the form σ(z) = j ρ j (z • θ j ). When the θ j are orthogonal, estimating the index space requires estimating each neuron, and
k ⋆ (σ) = k ⋆ (ρ): Proposition 7 (Generative Leap for Orthogonal Weights). Let y = r j=1 a j ρ(z j ). Then k ⋆ ≥ k ⋆ (ρ). Moreover, if all moments of ρ(Z) exist, then k ⋆ = k ⋆ (ρ).
We can extend this result to almost all networks with unit norm, linearly independent columns: Corollary 2 (Non-orthogonal, invertible weights).
Let y V = r j=1 a j ρ(v ⊤ j z) with ∥v j ∥ = 1. Then k ⋆ V ≤ k ⋆ (ρ) for all V , except possibly for a set of r(r -1)-dimensional measure 0.
An interesting question left for future work is whether this uniform control of the generative exponent by k ⋆ (ρ) for any V could be extended to general link functions; in other words whether the exclusion of these zero-measure sets is necessary in Proposition 6.
the r-Gaussian parity. Thus Chen et al. [2020] can be used to get error better than random guessing, but not vanishing error.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: We define the generative leap in Definition 3. We prove a sample complexity lower bound in Theorem 1 and we prove a matching upper bound in Theorem 3.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: See Appendix A.
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
Answer: [Yes] Justification: The proof of Theorem 1 can be found in Appendix C. The proof of Theorem 3 can be found in Appendix D.
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
Answer: [NA] Justification: This paper does not contain experiments.
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
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: This paper does not provide experiments.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental setting/details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [NA] Justification: This paper does not provide experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: This paper does not provide experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [NA] Justification: This paper does not provide experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: This research conforms to the NeurIPS code of ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10. Broader impacts Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This work is purely theoretical and is focused on understanding the fundamental limitations of learning with synthetic Gaussian data.
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
Answer: [NA] Justification: We are not releasing data or models.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA] Justification: We do not use any original assets.
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
Answer: [NA] Justification: We do not introduce new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: We did not conduct any crowdsourcing experiments or research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According
to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: This paper did not use LLMs as any important, original, or non-standard component. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: A Conclusions
In this work, we have extended the generative exponent k ⋆ to the general class of Gaussian multi-index models, and established a tight sample complexity n = Θ(d k ⋆ /2∧1 ) for learning their associated index space under no prior knowledge of the link function. We provide a lower bound based on the low-degree polynomial framework, and a matching upper bound obtained with a novel spectral method that incrementally reveals directions of the index space from a kernel U-statistic. The resulting upper bound recovers and extends several dedicated estimation procedures for specific families of multi-index models, such as ReLU networks or intersection of half-spaces.
There are several avenues for future work. First, this paper focuses on the simple setting of isotropic Gaussian data. Extending both the information leap and generative leap to more complicated data distributions is left to future work. Next, we focus on deriving estimators that work with minimal information about the multi-index model P, and which succeed with the optimal sample complexity in the ambient dimension d. As a result, our sample complexity guarantees scale with constants C(P) which could potentially be exponentially large in the hidden dimension r. Finally, we focus primarily on subspace estimation, as it is a requirement for full end-to-end learning.
this section cite: []

Section: References
Ref_id:b0 Title: The staircase property: How hierarchical structure can guide deep learning Year: (2021)
Ref_id:b1 Title: The merged-staircase property: a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer neural networks Year: (2022)
Ref_id:b2 Title: Sgd learning on neural networks: leap complexity and saddle-to-saddle dynamics Year: (2023)
Ref_id:b3 Title: Repetita iuvant: Data repetition allows sgd to learn high-dimensional multi-index functions Year: (2024)
Ref_id:b4 Title: Online stochastic gradient descent on non-convex losses from high-dimensional inference Year: (2021)
Ref_id:b5 Title: Stochastic gradient descent in high dimensions for multi-spiked tensor pca Year: (2024)
Ref_id:b6 Title: High-dimensional asymptotics of feature learning: How one gradient step improves the representation Year: (2022)
Ref_id:b7 Title: The franzparisi criterion and computational trade-offs in high dimensional statistics Year: (2022)
Ref_id:b8 Title: Matrix concentration inequalities and free probability Year: (2023-06)
Ref_id:b9 Title: Online stochastic gradient descent on non-convex losses from high-dimensional inference Year: (2021)
Ref_id:b10 Title: Learning single-index models with shallow neural networks Year: (2022)
Ref_id:b11 Title: On learning gaussian multi-index models with gradient flow Year: (2025)
Ref_id:b12 Title: Universality and sharp matrix concentration inequalities Year: (2024)
Ref_id:b13 Title: Statistical query algorithms and low-degree tests are almost equivalent Year: (2021)
Ref_id:b14 Title: Survey on algorithms for multi-index models Year: (2025)
Ref_id:b15 Title: Towards understanding hierarchical learning: Benefits of neural representations Year: (2020)
Ref_id:b16 Title: Learning polynomials in few relevant dimensions Year: (2020)
Ref_id:b17 Title: Learning deep relu networks is fixed-parameter tractable Year: (2022)
Ref_id:b18 Title: Learning narrow one-hidden-layer relu networks Year: (2023)
Ref_id:b19 Title: SAVE: a method for dimension reduction and graphics in regression Year: (2000)
Ref_id:b20 Title: Dimension reduction for conditional mean in regression Year: (2002)
Ref_id:b21 Title: Neural networks can learn representations with gradient descent Year: (2022)
Ref_id:b22 Title: Smoothing the landscape boosts the signal for sgd: Optimal sample complexity for learning single index models Year: (2023)
Ref_id:b23 Title: Computational-statistical gaps in gaussian singleindex models Year: (2024)
Ref_id:b24 Title: How two-layer neural networks learn, one (giant) step at a time Year: (2024)
Ref_id:b25 Title: The benefits of reusing batches for gradient descent in two-layer networks: breaking the curse of information and leap exponents Year: (2024)
Ref_id:b26 Title: Online learning of neural networks Year: (2025)
Ref_id:b27 Title: Decoupling of U-Statistics and U-Processes Year: (1999)
Ref_id:b28 Title: Optimal spectral transitions in high-dimensional multi-index models Year: (2025)
Ref_id:b29 Title: Efficiently learning one-hidden-layer relu networks via schur polynomials Year: (2024)
Ref_id:b30 Title: Learning geometric concepts with nasty noise Year: (2017)
Ref_id:b31 Title: Algorithms and sq lower bounds for robustly learning real-valued multi-index models Year: (2025)
Ref_id:b32 Title: Robust learning of multi-index models via iterative subspace approximation Year: (2025)
Ref_id:b33 Title: Learning single-index models in gaussian space Year: (2018)
Ref_id:b34 Title: Statistical query lower bounds for tensor pca Year: (2021)
Ref_id:b35 Title: Statistical-computational trade-offs in tensor PCA and related problems via communication complexity Year: (2024)
Ref_id:b36 Title: Statistical inference and the sum of squares method Year: (2018)
Ref_id:b37 Title: Tensor principal component analysis via sum-of-square proofs Year: (2015)
Ref_id:b38 Title: The power of sum-of-squares for detecting hidden structures Year: (2017)
Ref_id:b39 Title: Structure adaptive approach for dimension reduction Year: (2001)
Ref_id:b40 Title: On the complexity of learning sparse functions with statistical and gradient queries Year: (2024)
Ref_id:b41 Title: Learning intersections of halfspaces with distribution shift: Improved algorithms and sq lower bounds Year: (2024)
Ref_id:b42 Title: Learning geometric concepts via gaussian surface area Year: (2008)
Ref_id:b43 Title: Spectral estimators for multi-index models: Precise asymptotics and optimal weak recovery Year: (2025)
Ref_id:b44 Title: Notes on computational hardness of hypothesis testing: Predictions using the low-degree likelihood ratio Year: (2019)
Ref_id:b45 Title: Neural network learns low-dimensional polynomials with sgd near the information-theoretic limit Year: (2024)
Ref_id:b46 Title: Sliced inverse regression for dimension reduction Year: (1991)
Ref_id:b47 Title: The zero set of a real analytic function Year: (2015)
Ref_id:b48 Title: A statistical model for tensor pca Year: (2014)
Ref_id:b49 Title: Learning juntas Year: (2003)
Ref_id:b50 Title: Learning orthogonal multi-index models: A fine-grained information exponent analysis Year: (2024)
Ref_id:b51 Title: Emergence and scaling laws in sgd learning of shallow neural networks Year: (2025)
Ref_id:b52 Title: Hilbert Space Embeddings and Metrics on Probability Measures Year: (2010)
Ref_id:b53 Title: Fundamental limits of weak learnability in high-dimensional multi-index models Year: (2024)
Ref_id:b54 Title: Learning convex concepts from gaussian distributions with pca Year: (2010)
Ref_id:b55 Title: Structure from local optima: Learning subspace juntas via higher order pca Year: (2012)
Ref_id:b56 Title: A multiple-index model and dimension reduction Year: (2008)
Ref_id:b57 Title: An adaptive estimation of dimension reduction space Year: (2002)
Ref_id:b58 Title: Interpolating convex and non-convex tensor decompositions via the subspace norm Year: (2015)
