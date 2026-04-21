Title: Score Matching with Missing Data
Abstract: Score matching is a vital tool for learning the distribution of data with applications across many areas including diffusion processes, energy based modelling, and graphical model estimation. Despite all these applications, little work explores its use when data is incomplete. We address this by adapting score matching (and its major extensions) to work with missing data in a flexible setting where data can be partially missing over any subset of the coordinates. We provide two separate score matching variations for general use, an importance weighting (IW) approach, and a variational approach. We provide finite sample bounds for our IW approach in finite domain settings and show it to have especially strong performance in small sample lower dimensional cases. Complementing this, we show our variational approach to be strongest in more complex highdimensional settings which we demonstrate on graphical model estimation tasks on both real and simulated data.

Section: Introduction
Over the last decade, score matching has established itself as a powerful tool with downstream use in many areas of machine learning. Examples include: energy based modelling (Swersky et al., 2011;Bao et al., 2020;Li et al., 2019b), mode-seeking clustering (Sasaki et al., 2014), and perhaps most prominently of all Diffusion processes (Song & Ermon, 2019;Song et al., 2021b;Tashiro et al., 2021;Song et al., 2021a;Huang et al., 2021). Score matching aims to learn the score of a distribution which is the gradient of the log of the probability density function (PDF) (s(x) = ∇ x log p(x)). In contrast to modelling the density directly, the score does not need to integrate to one meaning there is no need to calculate a normalising constant. This allows it to be much more more flexibly modelled than the Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). density itself. Furthermore, the validity of the score matching objective itself requires only very mild assumptions of the family of proposed scores further ensuring this flexibility. Alongside the classical method (Hyvärinen, 2005), various adaptations of score matching have arisen to improve performance, decrease computational cost, and extend the approach to a wider range of settings (Hyvärinen, 2007;Vincent, 2011;Song et al., 2020;Liu et al., 2022). In this work, we extend the score matching framework to handle missing data at training time. Specifically, we learn the full score function from partially missing multidimensional input data, a paradigm we term missing score matching. Crucially, our approach is compatible with any parameterised score model, enabling its application to both explicit score formulations and more general approaches such as neural networks (NNs). We propose two methods to adapt the original score matching method as well as its popular adaptations, truncated, sliced, and denoising score matching (Hyvärinen, 2007;Vincent, 2011;Liu & Wang, 2017;Song et al., 2020). These two distinct but closely related methods complement each other allowing for a wide range of problems to be tackled. The first method is a simpler importance weighting (IW) approach which we refer to as marginal IW score matching. For this method we obtain finite sample bounds in the bounded domain setting under certain conditions. We also provide experimental results demonstrating its efficacy in lower dimensional settings and where less data is available. Our second approach is a more computationally sophisticated variational approach which we refer to as marginal variational score matching. We demonstrate the efficacy of this approach in more complex, high dimensional settings by applying it to the problem of graphical model estimation with both real and synthetic datasets.
In section 2 we discuss relevant works for score matching and related fields. In section 3 we will introduce our problem more formally including score matching and any notation used. Section 4 will be used to introduce our methods. Section 5 will present results on some real and simulated datasets. In Section 6 we give our conclusion.
this section cite: ['b24', 'b0', 'b19', 'b20', 'b25', 'b7', 'b8', 'b9', 'b27', 'b21', 'b15', 'b9', 'b27', 'b14', 'b21']

Section: Related Works
While there has been some work which utilises score matching with missing data, these approaches mostly do so exclusively through the lens of diffusion models. Specifically works such as MissDiff (Ouyang et al., 2023) and Ambient Diffusion (Daras et al., 2023) require the score function itself to take the form of a neural network (NN) which learns the scores of the fully-observed and corrupted scores simultaneously. This prohibits their use in situations where our model for the score is some explicit parameterisation whose parameters we want to learn as is the case in settings such as energy based modelling Li et al. (2023); Bao et al. (2020); Salimans & Ho (2021) and Gaussian graphical models (Lin et al., 2016;Yu et al., 2018). Ambient Diffusion also requires the data to be further artificially corrupted in order to create a pseudo-supervised learning paradigm making both Ambient Diffusion and MissDiff subject to various levels of out of sample learning without specific adjustments for this phenomenon.
Looking more generally at distribution estimation with missing data, multiple works in the field of generative modelling have looked to tackle the problem of providing a generative model for a distribution given corrupted samples from it. Prominent among these are MisGAN (Li et al., 2019a), which presents a marginalised GAN framework and MCFlow (Richardson et al., 2020) , which presents a EM like normalising flow framework. Neither of these approaches allow for flexible specification of a parametric density estimate however with MCFlow requiring the density to be a normalising flow and MisGAN having no model for the density whatsoever.
To our knowledge, the only approach which seems to adapt score matching to missing data in a parameter preserving manner is presented in (Uehara et al., 2020) using an iterative EM-like procedure. However they themselves admit that there is little intuitive understanding of when this approach will converge. Additionally, due to the nature of the score matching objective, the expectation step cannot be directly approximated using Monte Carlo estimation and instead requires fractional importance weighting, a method which employs nested Monte Carlo estimates introducing bias into the training objective.
Parallel to this, some papers have looked to extend score matching to the latent variable setting, an area with much commonality to missing data (Vértes & Sahani, 2016;Bao et al., 2020;2021). Latent variable modelling differs in two crucial aspects from missing score matching. Firstly the components which are unobserved (the latent variables) remain constant between samples, and secondly there is not necessarily a notion of a ground truth for the unobserved components in when data is corrupted. Additionally each of these works has limitations; Vértes & Sahani (2016) only applies to exponential families, Bao et al. (2020) requires a gradient unrolling step in its optimisation which is computationally expensive and can lead to errors in the op-timisation procedure (as acknowledged in their follow on work), and Bao et al. (2021) is only given for denoising score matching, not for classical or sliced score matching.
this section cite: ['b16', 'b5', 'b12', 'b0', 'b18', 'b13', 'b31', 'b17', 'b26', 'b28', 'b0', 'b28', 'b0', 'b1']

Section: Setting

this section cite: []

Section: Notation
For n ∈ N let [n] := {1, . . . , n}. For a random variable Z we use supp(Z) for the support of Z. For f : R d → R we write ∂ j f (x) := ∂f ∂xj where x = (x 1 , . . . , x d ) ⊤ and ∇ x f (x) := (∂ j f (x), . . . , ∂ d f (x)) ⊤ , the gradient of f . For f : R d → R d take f (x) j as the j th component of f (x) and write ∇ x • f (x) := ∂ 1 f (x) 1 + • • • + ∂ d f (x) d . Finally for a, b ∈ R d , take a • b to be the Hadamard product.
We now introduce some indexing notation which we will be using for RVs and functions throughout. This will prove useful when identifying the missing non-missing components of our data. Let Z be a random variable taking values in R d . We use Z j to refer to the j th component Z and for λ ⊆ [d] take Z λ = {Z j } j∈λ . We use negation in indexing to mean the complementing coordinates. More precisely we let -j denote [d] \ {j} and let -λ denote [d] \ λ. We typically use Z (i) to denote an independent copy of Z. For a function f : X → Y and
x λ ∈ X λ , x ′ -λ ∈ X -λ , we take f (x λ , x ′ -λ ) to be f (z)
where
z j := x j if j ∈ λ x ′ j if j ∈ -λ .
We will take X to be a RV taking values in X ⊆ R d representing our original dataset and X ′ to be a RV representing some generative/variational/importance weighting distribution. i.e., the "artificial distributions" we will utilise in our method. Similarly, we take E, E ′ to be expectations with respect to (w.r.t.) X, X ′ respectively.
Throughout we take p to be the pdf of the RV, X, and p θ to be a model therein. We let q represent an unnormalised density (i.e. N -1 • q = p for some normalising constant N > 0.) We will write marginalisations/conditionings for both true and model densities implicitly with p(x λ ) := X p(x)dx -λ and p(x λ |x -λ ) being the conditional density of X λ |X -λ = x -λ for example. Now that we have introduced our notation we can move onto the key area of focus for our work, score matching.
this section cite: []

Section: Score Matching
First proposed by (Hyvärinen, 2005), score matching aims to learn the gradient of the log-density (score). The advantage of this framework over full density approaches such as maximum likelihood estimation (MLE) is that we are not restricted to parametric models which integrate to 1. This allows us to be much more flexible in how we param-eterise in turn making high dimensional distribution modelling more feasible. We now introduce the approach.
Let X be a RV over R d with PDF p. We say that q is the unnormalised density of X if N -1 • q(x) = p(x) where p is the PDF of X and N is the normalising constant of q. Define the score, of X to be
s(x) := ∇ x log p(x) = ∇ x log q(x).
The aim of score matching is to learn s from a collection of IID copies of X which we denote D := {X (i) } n i=1 . Following Hyvärinen (2005), we introduce a generic parameterised proposal score s θ for θ ∈ Θ ⊆ R p and aim to minimise the Fisher Divergence between the true distribution and our proposal distribution which is given by
F (θ) := E[∥s(X) -s θ (X)∥ 2 ].
The key result from Hyvärinen (2005) which enables us to practically implement score matching is that under certain (fairly minimal) regularity conditions, which we provide in Appendix D.1, we have
L(θ) := E 2∇ X • s θ (X) + ∥s θ (X)∥ 2 = F (θ) -C(1)
where here and throughout, we take C to represent any constant which does not depend upon θ. Crucially, L(θ) is now an expectation of observable random variables. Hence we can now approximate this with our data and take θ as
θ := argmin θ 1 n n i=1 2∇ X (i) • s θ (X (i) ) + ∥s θ (X (i) )∥ 2 .
this section cite: ['b8', 'b8', 'b8']

Section: TRUNCATED SCORE MATCHING
A limitation of standard score matching is that it requires lim xi→∞ p(x) = 0 for all x i ∈ R. Thus it cannot be used for many distributions with compact support if the density does not converge to zero at the (topological) boundary. Initial work to adapt score matching to truncated distributions was presented in (Hyvärinen, 2007) for distributions on [0, ∞) then further expanded in (Liu et al., 2022;Yu et al., 2022) to general compact spaces X . For our compact space X ⊆ R d we use ∂X to denote the (topological) boundary. We now minimise some weighted version of the Fisher divergence whose weights go to zero at the boundary. Specifically let g : X → R be a function satisfying
lim x→x ′ g(x) j = 0 for any x ′ ∈ ∂X , j ∈ [d]. Our objec- tive is then F T (θ) := E g 1 2 (X) • (s θ (X) -s(X)) 2 .
Just as in classical score matching we obtain an equivalence (though this time via Green's theorem rather than simple integration by parts) giving us that under certain regularity conditions on g, s, and X ,
L T (θ) :=E   j∈d g(X) j 2∂ j s θ (X) j + s θ (X) 2 j   + E   j∈d ∂ j g(X) j s θ (X) j   = F T (θ) -C.
This can again be approximated via data using standard Monte Carlo approximation. Full details on the conditions required for this approach alongside the proof can be found in (Liu et al., 2022). Two other key extensions of score matching are sliced score matching (Song et al., 2020) and denoising score matching (Vincent, 2011). We introduce these extensions in Appendix D with our corresponding adaptations to missing data given in Appendix A.1. Now, we give our missing data scenario.
this section cite: ['b9', 'b15', 'b32', 'b15', 'b21', 'b27']

Section: Missing Data Scenario
Instead of observing samples from X we assume that we observe samples from the corrupted version of the RV given by X. To define X we introduce a mask RV M over {0, 1} d and then define X by
Xj = X j if M j = 1 ∅ if M j = 0
where Xj = ∅ represents that coordinate being missing. We will be focussing on the missing completely at random scenario where M ⊥ X. However, we do provide an extension to missing not at random data in Appendix A.1.4. We introduce the RV Λ on P([d]) defined by Λ := {i ∈ [d]|M i = 1} so that Λ gives the non-corrupted coordinates of X and take λ to be a sample of Λ. Crucially given samples from X, we also have samples from X Λ .
Our aim is to adapt the score matching objective to estimate the full score s by a parameterised score s θ using samples from the corrupted data D := { X(i)
} n i=1 ≡ {X (i) Λi } n i=1 .
this section cite: []

Section: Marginal Score Matching
To motivate our approach we look at how we might use MLE in the case where the normalising constant and conditional normalising constants were calculable. For p θ our parametric model of the density, we would choose θ to be
θ := argmax θ n i=1 log pθ ( X(i) )
where pθ is the associated corrupted data density when X ∼ p θ . As our data is missing completely at random this is actually equivalent to maximising
n i=1 log p θ;Λi (X (i) Λi ), where p θ;λ (x λ ) := X -λ p θ (x)dx -λ .
For notational simplicity we will thus reframe our problem as working with marginal samples
{X (i) Λi } n i=1 .
this section cite: []

Section: Marginal Score Matching
Our approach is to directly alter the score matching objective similarly. Just as densities have associated marginal densities so do scores have associated marginal scores.
Definition 4.1 (Marginal Score function). Let s be a score function with s(x) = ∇ x log q(x) for q an unnormalised PDF. Then the associated marginal score function is
s λ (x λ ) := ∇ x λ log R d-|Λ| q(x)dx -λ .(2)
This definition of marginal scores restricts s to a genuine score function. For this reason we will also want s θ to always be a genuine score function or at least to have an anti-derivative. The simplest way to achieve this is to work with q θ : X → (0, ∞) as our baseline and define s θ (x) := ∇ x log q θ (x). We will also take p θ (x) := X q θ (x)dx -1 q θ (x) which we assume to be unknown.
With this notion of a marginal score we can define our marginal Fisher divergence to be
F M (θ) := E[∥s Λ (X Λ ) -s Λ;θ (X Λ )∥ 2 ] (3
)
where s λ;θ is defined analogously to s λ . As with normal score matching can relate this objective to one involving no terms of s λ . We first need the following assumptions.
Assumption 4.2. For any θ > 0, λ ∈ supp(Λ):
(a) p θ is well defined, i.e. X q θ (x)dx < ∞; (b) E[∥s λ (X λ )∥ 2 ], E[∥s λ;θ (X λ )∥ 2 ] < ∞; (c) p λ (x) is differentiable and q λ;θ is twice differentiable; (d) p λ (x λ )s λ;θ (x λ )-→0 as ∥x∥-→∞; (e) p λ;θ (X λ ) = p λ (X λ ) almost surely (a.s.) for all λ ∈ supp(Λ), implies that p θ (X) = p(X)
a.s.. Assumption (a) ensures that our proposal unnormalised density is always a genuine unnormalised density. Assumptions (b)-(d) are similar to the standard assumptions given for standard score matching. Assumption (e) is an identifiability assumption which is required to be feasibly able to learn the true data distribution from our corrupted data. Proposition 4.3. Given Assumptions 4.2(a)-(d) hold
L M (θ) :=E[2∇ XΛ • s Λ;θ (X Λ ) + ∥s Λ;θ (X Λ )∥ 2 ] (4) =F M (θ) -C.
If (e) also holds and there exists some θ * such that s θ * (X) = s(X) a.s.. Then if θ is a minimiser of L M (θ) we have that q θ (X) = N p(X) a.s. for some constant N , i.e. the minimiser is the true unnormalised density.
Through this result we have shown, much like with standard score matching, that under certain regularity conditions our objective is uniquely minimised by the true unnormalised density. We then approximate this objective by
LM;n (θ) := 1 n n i=1 ∇ X (i) Λ i • s Λi,θ (X (i) Λi ) + ∥s Λi;θ (X (i) Λi )∥ 2
and choose θ = argmin θ LM;n (θ).
Unfortunately this approach in its current state is practically infeasible as the integrals involved in deriving the marginal scores for any non-trivial problem will be intractable. Hence, we must devise a way to estimate the marginal scores without having to compute the integrals. We tackle this issue in Section 4.2, but first we provide a similar result for the case of truncated score matching.
this section cite: []

Section: TRUNCATED SCORE MATCHING
Truncated score matching can be adapted similarly to standard score matching by simply having marginal weighting functions g λ : X λ → [0, ∞) for each subset λ ∈ supp(Λ) and taking the marginal truncated Fisher divergence to be
F TM (θ):= E g Λ (X Λ ) 1 2 • (s Λ (X Λ ) -s Λ;θ (X Λ )) 2 .
using integration by parts gives the following equivalence
L TM (θ):=E   j∈Λ g Λ (X Λ ) j s Λ;θ (X Λ ) 2 j +2∂ j s Λ;θ (X Λ ) j   + E   j∈Λ 2∂ j g Λ (X Λ ) j s Λ;θ (X Λ ) j   (5
) =F TM (θ) -C.
Proof given in Appendix C.1.1. We then take LTM;n as the Monte-Carlo estimate of L TM . We also construct similar objectives from sliced and denoising score matching as well as a similar result for missing not at random data in Appendix A.1. We now move to the task of estimating the marginal scores in these objectives.
this section cite: []

Section: Importance Weighting
Our first proposal is an importance weighting approach. Let p ′ be a density over R d-|λ| which we can both evaluate and sample from then
R d-|λ| q θ (x)dx -λ = E X ′ λ ∼p ′ q θ (x λ , X ′ -λ ) p ′ (X ′ -λ ) . (6) Algorithm 1 Marginal IW Score Matching Input: {X (i) Λi } i∈[n] , q θ , p ′ , θ 0 , r ∈ N. Set θ = θ 0 . repeat for i=1 to n do Sample {X ′(i,k) } k∈r from p ′ (.|X (i) Λi ). Use X (i) Λi , {X ′(i,k) -Λi } k∈[r] to get Monte-Carlo esti- mates, ŝΛi,r;θ (X (i) Λi
), of the marginal scores by (7). end for Use ŝΛi,r;θ (X (i) Λi ) to obtain LM/TM;n,r (θ) by (4). Compute ∇ θ LM/TM;n,r (θ) and update the value of θ. until Maximum iteration reached. This allows us to define our marginal score estimate.
Definition 4.4 (Marginal Score Estimate). For a given λ ∈ supp(Λ) , x λ ∈ X λ , score model, s θ , and r ∈ N we take our estimate of s θ;λ,r (x λ ) to be
ŝλ,r;θ := ∇ x λ log 1 r r k=1 q θ (x λ , X ′(k) -λ ) p ′ (X ′ (k) -λ ) (7) where X ′(1) -λ , . . . , X ′(r) -λ are IID copies of X ′ -λ ∼ p ′ .
this section cite: []

Section: IW SAMPLE OBJECTIVE
We can now plug these marginal score estimates into our sample objective for either normal or truncated score matching. We use M/TM to denote analogous definitions and results for both marginal and truncated marginal score matching. Let {X (i) Λi } n i=1 be our samples from X Λ . We then take our IW sample objective to be as LM/TM;n (θ) but with ŝΛi,r;θ (X
(i) Λi ) replacing s Λi;θ (X (i) Λi
). The full objective is given in Appendix E.1.1 We refer to this sample objective as LM/TM;n,r (θ) and take our estimate to be θ := argmin θ LM/TM;n,r (θ).
Algorithm 1 gives our high level estimation algorithm.
Remark 4.5. Algorithm 1 can directly be applied to both sliced and denoised score matching by replacing equation (4) by equations (13) and (15) respectively.
this section cite: []

Section: FINITE SAMPLE BOUNDS
A benefit of truncated score matching is that it allows us to work on distributions with densities bounded below which enables us to give finite sample bounds for the error of our estimated score w.r.t. our marginal objective. We briefly present these now with more detail given in Appendix A.2.
Theorem 4.6. Suppose assumption 4.2 alongside assumptions A.1, A.11, A.13 from the Appendix hold and let θ n,r ∈ Θ be the minimiser of LTM;n,r (θ). If Θ ⊆ R p with diam(Θ) = A then for sufficiently large n, r
P F TM (θ n,r ) ≥β 1 p log(dnrA/δ) min{r, n} < δ.
Note that r is the number of importance weighting samples for each data sample and therefore is something we can choose ourself. This means that with this approach we can achieve approximately √ n convergence rates. A downside however is that to achieve this we need r to be of order at least n which would lead to an O(n 2 ) computational cost. In practice we find relatively strong performance choosing r small. Setting it at r = 10 in our experiments.
Remark 4.7. The error presented is measured with respect to our Marginal Fisher Divergence, rather than the full Fisher Divergence (which would be the preferred accuracy metric). Relating these two quantities requires connecting the fully observed distribution to its marginals, a task that depends on the specific form of the distribution. Investigating the assumptions and conditions under which this connection can be made offers an interesting and valuable direction for future research.
this section cite: []

Section: Gradient First Approach
A key limitation with an IW approach is that it will struggle in higher dimensional scenarios. Additionally the importance weighting is embedded inside other functions which leads to the same nested expectation issue as the EM approach of Uehara et al. (2020), causing bias in our estimator. As an alternative to this we build upon a variational approach initially discussed in the context of latent variable models in Vértes & Sahani (2016); Bao et al. (2020;2021).
The core idea is to start with L M as before and then take gradients w.r.t. our parameters before then writing our objective in terms of expectations over X -λ|λ;θ . As we don't then need to take gradients of these expectations w.r.t. θ, we can estimate them with any black-box method we desire, opening the door for variational approximation to be used. This approach has been explored for exponential family distributions (Vértes & Sahani, 2016) and for denoising score matching (Bao et al., 2021) however we provide the most general version of this result which can be applied to any of the score matching methods and any model class. We first introduce the following key Lemma.
Lemma 4.8. Fix λ ⊆ [d], x λ ∈ X λ . We have that for any function h θ : X → R.
s θ;λ (x λ ) =E ′ [s θ (x λ , X ′ -λ ) λ ] (8) ∇E ′ [h θ (x λ , X ′ -λ )] =E ′ [∇h θ (x λ , X ′ -λ )] (9) +Cov ′ (s θ (x λ , X ′ -λ ), h θ (x λ , X ′ -λ ))
where ∇ represents the gradient w.r.t. either x λ or θ and here E ′ , Cov ′ are w.r.t. X ′ -λ |X λ = x λ ∼ p θ (.|x λ ). This results allows us to obtain our alternative objective. Corollary 4.9. Let L M be defined as in (4). We have that
∇ θ L M (θ) =E 2 j∈Λ Ψ Λ (s θ (.) 2 j + ∂ j s θ (.) j ) (10
) -E ′ [s θ (X Λ , X ′ -Λ ) j ]Ψ Λ (s θ (.) j )
where for any function h
θ : R d → R, λ ⊆ [d], Ψ Λ (h θ ) =E ′ [∇ θ h θ (X Λ , X ′ -Λ )] + Cov ′ ∇ θ log q θ (X Λ , X ′ -Λ ), h θ (X Λ , X ′ -Λ ) and E ′ , Cov ′ are w.r.t. X ′ -Λ |X Λ ∼ p θ (.|X Λ ) with E being w.r.t. X Λ ∼ p.
Proofs for both results are given in Appendix C.3. Crucially E ′ , Cov ′ can be estimated freely. This allows us to use variational inference to approximate p θ (x -λ |x λ ) and in turn the expectations and covariances in (10).
Remark 4.10. We provide additional implementation details for computing this gradient estimate in Appendix A.5. We also discuss equivalences between this objective and our marginal IW objective in A.3.
We explore estimation of E ′ , Cov ′ in Section 4.3.2 but first we provide a similar result for truncated score matching.
this section cite: ['b26', 'b28', 'b0', 'b28', 'b1']

Section: TRUNCATED SCORE MATCHING
We define a similar objective for truncated score matching.
Corollary 4.11. With L TM defined as in (5) we have that
∇ θ L TM (θ) =E 2 j∈Λ g Λ (X Λ ) j Ψ Λ (s θ (.) 2 j + ∂ j s θ (.) j ) -E ′ [s θ (X Λ , X ′ -Λ ) j ]Ψ Λ (s θ (.) j ) + ∂ j g Λ (X Λ ) j Ψ Λ (s θ (.) j )(11)
with Ψ Λ and E ′ defined as in Corollary 4.9.
Proof given in Appendix C.1.1. Similar results for sliced and denoising score matching are given in Appendix A.1.
this section cite: []

Section: VARIATIONAL APPROXIMATION
We can now use variational approximation to estimate the expectations and covariances in Corollaries 4.9 & 4.11. Specifically, let p ′ ϕ (x -λ |x λ ) be some generative conditional distribution dependent upon parameter ϕ. We want to train p ′ ϕ to approximate p θ . We may write ϕ(θ) to highlight the dependence on our current parameter estimate however we will omit this for brevities sake. The following proposition from Bao et al. (2020) shows us how to train ϕ.
Proposition 4.12 (Bao et al. (2020)). For distributions p ′ , p let F (p ′ |p) and KL(p ′ |p) be the Fisher and KL divergences between p ′ and p. We have that for any λ ⊆
[d], x λ ∈ X λ KL(p ′ ϕ (.|x λ )|p θ (.|x λ )) = E ′ log p ′ ϕ (X ′ -λ |x λ ) q θ (x λ , X ′ -λ ) +B F (p ′ ϕ (.|x λ )|p θ (.|x λ )) = E ′ ∇ X ′ -λ log p ′ ϕ (X ′ -λ |x λ ) -s θ (x λ , X ′ -λ ) -λ 2 where expectations are w.r.t. X ′ -Λ ∼ p ′ ϕ (.|x λ )
and B is a constant not depending upon ϕ (but will depend on θ.) In other words we can fit to the conditional density p θ (.|x λ ) given only the unconditional unnormalised density q θ (x λ , .) or full score s θ (x λ , .). This allows us to train p ′ ϕ (.|x λ ) to approximate the conditional density, p θ (.|x λ ). In our case we won't be learning this variational model for a fixed x λ or even fixed observed coordinates λ. Hence we take our objective to be one of
J KL (ϕ, θ) := E log p ′ ϕ (X ′ -Λ |X Λ ) q θ (X Λ , X ′ -Λ ) J F (ϕ, θ) := E ∇ X ′ -Λ log p ′ ϕ (X ′ -Λ |X Λ ) q θ (X Λ , X ′ -Λ ) 2 with (X Λ , X ′ -Λ ) ∼ p ′ ϕ (X ′ -Λ |X Λ )p(X Λ ).
We then take and ĴF , ĴF to be the Monte-Carlo approximations with samples (X Λ , X ′ -Λ ) from the same distribution.
Remark 4.13. J F has the advantage of not needing to know the normalising constant of q ′ ϕ = N ϕ • p ′ ϕ either.
Remark 4.14. As ϕ depends upon θ, we need to update it each time we update θ. In practice we find taking 10 gradient steps of ϕ for each gradient step of θ to work well.
With this, we define ∇ θ L M/TM (θ) to be the Monte-Carlo estimate of (10)/(11) with samples {(X
(i) Λi , X ′(i,k) -Λi )} (i,k)∈[n]×[r]
where
X (i) Λ
are our original corrupted data samples from p and X
′(i,k) -Λ are our variational samples from p ′ ϕ (.|X (i) Λi ).
We can now state our full variational approach which is given in Algorithm 2. Remark 4.15. Algorithm 2 can directly be applied to both sliced and denoised score matching by replacing equation (10) by equations ( 14) and ( 16) respectively.
this section cite: ['b0', 'b0']

Section: Algorithm 2 Marginal Variational Score Matching
Input: 10)/(11). Use this gradient estimate to update θ. until Maximum iterations reached.
{X (i) Λi } i∈[n] , q θ , p ′ ϕ , θ 0 , ϕ 0 , L ∈ N, r ∈ N. Set θ = θ 0 , ϕ = ϕ 0 . repeat for l = 1 to L do For i ∈ [n] sample X ′(i) from p ′ ϕ (.|X (i) Λi ). Use {(X (i) Λi , X ′(i) -Λi )} i∈[n] to get Monte-Carlo ap- proximates of J KL/F (ϕ, θ) given by ĴKL/F (ϕ, θ). Compute ∇ ϕ ĴKL/F (ϕ, θ) and update ϕ. end for For i ∈ [n] sample {X ′(i,k) } k∈r from p ′ ϕ (.|X (i) Λi ). Use {(X (i) Λi , X ′(i,k) -Λi )} (i,k)∈[n]×[r] to get our Monte- Carlo estimate, ∇ θ L M/TM (θ) using equation (
this section cite: []

Section: Results
Here we go through simulated results comparing our IW approach (Marg-IW) in Algorithm 1 and our variational approach (Marg-Var) in Algorithm 2 to the EM approach of Uehara et al. (2020). We also compare to a naive marginalisation approach involving zeroing out the missing dimensions and only taking the observed output dimensions of the score, which we call Zeroed Score Matching. This approach is the natural adaptation of MissDiff from Ouyang et al. (2023) away from NN to explicitly parameterised models. We describe Zeroed Score Matching and its relation to MissDiff in Appendix D.2. In our experiments, we highlight a unique strength of our methods by applying them to explicitly parameterised score models. We could however, equally apply them to more complex, noninterpretable models such as NNs. More implementation details can be found in Appendix E.3. not converge, a phenomenon we discuss more in Appendix D.2. In Appendix B.1.1 we present the average mean and precision estimation error for this experiment. In Appendix B.1.2 we present the untruncated results and illustrate how the naive marginalisation poorly models strong relationship between dimensions 1 and 10.
this section cite: ['b26', 'b16']

Section: NON-GAUSSIAN MODEL
For this experiment we tested our parameter estimation for a an ICA inspired unnormalisable model of the form
p(x) ∝ exp i,j θ * i,j x 2 i x 2 j .
Here we parameterise our model identically with the aim of estimating θ * . We vary the dimension of X and plot the estimation error with a sample size of 1,000 and each coordinate missing independently with probability 0.5. The results are presented in Figure 2. Our variational method (Marg-Var) consistently yields the lowest error. Moreover, as the dimensionality increases, the performance gap between Marg-Var and the other methods widens. This supports the notion that our approach is more accurately able to capture complex marginalisations than the competing approaches which fail as the dimension grows. We note that all other methods perform comparably with the performance of EM and Marg-IW being indistinguishable, a pattern we observe throughout our experiments. This similarity is unsurprising both approaches use self normalised importance weighting to approximate conditional expectations with respect to our current score estimate while being broadly motivated by fitting to the marginal scores. Nevertheless, the precise mechanism for this similarity remains unclear and warrants further exploration. Additional experiments exploring the effect of sample size and missingness probability on estimation accuracy are given in appendix B.1.3.
this section cite: []

Section: Gaussian Graphical Model Estimation
Gaussian graphical models (GGM) are a popular way of modelling dependence between dimensions of data. Let us assume that the underlying data follows a Gaussian distribution with mean µ ∈ R d and precision P ∈ R d×d . In this setting, a Bayesian network (BN) can represent the dependencies between the dimensions of X with the (undirected) edges of the BN exactly being the non-zero off-diagonal entries of the precision, P . Hence estimating the precision matrix P gives the BN. Score matching has been shown to be an effective way of achieving this with L1-regularisation on the off-diagonal of P to push terms to 0 (Lin et al., 2016;Yu et al., 2018). Decreasing the level of L1-regularisation then gives a range of classifiers with increasing True and False positive rates (TPR/FPR) as the level of regularisation decreases. Score matching can also be applied to truncated GGMs where we aim to learn the original BN but only observe the samples inside some truncated region.
We apply our methods to learn GGMs and truncated GGMs with missing data as well. We use varying levels of L1 regularisation on our objective via proximal stochastic gradient descent in our optimisation (Beck, 2017).
5.2.1. STAR SHAPED TRUNCATED GRAPHICAL MODEL Here we create a star shaped GGM in which one node has a high probability of being connected with each other node independently and all other connections have probability 0. We truncate the data along a random hyperplane such that 20% of the distribution lies outside of the truncation boundary. Each coordinate is then MCAR independently with the same probability. We run multiple experiments with this probability ranging from 0.2 to 0.9 and present the results in figure 3. As we can see here Marg-Var performs best with all other approaches performing comparably. For illustrative purposes, we plot individual ROC curves from this experiment in Appendix B.2.3. 0.2 0.4 0.6 0.8 Missingness Probability 0.6 0.7 0.8 0.9 1.0 AUC Marg-IW (Ours) Marg-Var (Ours) Zeroed EM Figure 3: Mean AUC of star graph edge detection with varying missingness alongside 95% C.I.s. Higher is better. 5.2.2. UNSTRUCTURED DENSE GRAPHICAL MODEL Here we create a GGM by making each edge occur independently with probability 0.5. The rest of the experiment was constructed as before. Results are given in Figure 4. Again we can see that our variational approach performs 0.2 0.4 0.6 0.8 Missingness Probability 0.60 0.65 0.70 0.75 0.80 0.85 AUC Marg-IW (Ours) Marg-Var (Ours) Zeroed EM Figure 4: Mean AUC of dense graph edge detection with varying missingness alongside 95% C.I.s. Higher is better.
best though not as clearly as in the previous example. We believe this to be because for more unstructured problems, naive marginalisation performs moderately well.
this section cite: ['b13', 'b31', 'b2']

Section: INCREASING NUMBER OF STARS
To explore this further, we construct and experiment where we vary the number of star centres (high degree nodes) while keeping the edge density constant. We present the results in Figure 5. As we increase the number of star centres, Marg-Var no longer noticeably outperforms the other approaches. This is because as the number of stars increases, (i.e. the structure of the graph decreases) naive marginalisation is a better approximation. This is illustrated on the marginal precisions themselves in Appendix B.2.1.
this section cite: []

Section: S&P 100
Here we took closing price data over 5 years for the 100 stocks in the S&P 100 with each stock being a dimension and each day being a sample. Gaussian graphical models with various levels of connectivity were then constructed using standard score matching on the fully observed data. The data was then artificially corrupted and each missing score matching approach applied. The AUC was then calculated for each method taking the GGM from fully observed score matching as the ground truth. More details given in appendix E.  As we can see Marg-Var clearly out performs all the other approaches which appear to perform equivalently.
5.2.5. YEAST DATA Here data first introduced in Brem & Kruglyak (2005) is used consisting of readings of expression for 7086 genes/ORFs across 262 yeast segregants. Each gene represents a dimension with each segregant representing a sample. We subset the data to take the 106 genes present in at least 95% of the samples with the aim of learning the relationship between them. The same approach as the previous section is applied with the results shown in figure 7. Again Marg-Var clearly outperforms the other approaches which all perform comparably.
this section cite: ['b3']

Section: Conclusion
To conclude, score matching is a versatile method whose applications at the heart of modern machine learning problems. In this work we have tackled the problem of adapting score matching to partially missing data. We have presented two separate but related approaches to this method, one using importance weighting and another using variational approximation. We have also provided extensions of these methods to truncated score matching, sliced and denoising score matching. For truncated score matching with our IW approach we have provided finite sample bounds on the accuracy of the estimated score in terms of the marginal truncated Fisher divergence.
We have provided several simulated and real world experiments demonstrating our methods' efficacy for both parameter estimation and downstream GGM edge detection. We have shown the benefits and drawbacks of each approach with IW performing best in lower dimensional settings with less data and the variational approach performing best in more complicated higher dimensional settings.
There is, however still much work to be done in this area. From a theoretical perspective, while we have finite sample bound on the error of our loss, marginal nature of the loss makes it unclear exactly how this translates to parameter or general score model accuracy, leaving room for further theoretical exploration. From an implementation perspective, variational inference in the presence of missing data requires accounting for the randomness of "latent" and "observed" variables. The standard variational inference technique can be further refined to accommodate this setting. Finally, since our method is compatible with denoised score matching, it can naturally be extended to diffusionbased model. This paves the way for future work on applying our approach to generative modelling with diffusion processes in the presence of missing data.
this section cite: []

Section: References
Ref_id:b0 Title: Bi-level score matching for learning energy-based latent variable models Year: (2020)
Ref_id:b1 Title: Variational (gradient) estimate of the score function in energy-based latent variable models Year: (2021-07)
Ref_id:b2 Title: The proximal gradient method Year: (2017)
Ref_id:b3 Title: The landscape of genetic complexity across 5,700 gene expression traits in yeast Year: (2005-02)
Ref_id:b4 Title: Importance weighted autoencoders Year: (2016-05-02)
Ref_id:b5 Title: Ambient diffusion: Learning clean distributions from corrupted data Year: (2023)
Ref_id:b6 Title: Graphical lasso and thresholding: Equivalence and closed-form solutions Year: (2019)
Ref_id:b7 Title: A variational perspective on diffusion-based generative models and score matching Year: (2021)
Ref_id:b8 Title: Estimation of non-normalized statistical models by score matching Year: (2005)
Ref_id:b9 Title: Some extensions of score matching Year: (2007)
Ref_id:b10 Title: Learning from Incomplete Data with Generative Adversarial Networks Year: (2019)
Ref_id:b11 Title: Learning energy-based models in high-dimensional spaces with multi-scale denoising score matching Year: (2019)
Ref_id:b12 Title: Learning energybased models in high-dimensional spaces with multiscale denoising-score matching Year: (2023)
Ref_id:b13 Title: Estimation of highdimensional graphical models using regularized score matching Year: (2016)
Ref_id:b14 Title: Learning Deep Energy Models: Contrastive Divergence vs Year: (2017-07)
Ref_id:b15 Title: Estimating density models with truncation boundaries using score matching Year: (2022)
Ref_id:b16 Title: Training diffusion models on tabular data with missing values Year: (2023)
Ref_id:b17 Title: Monte Carlo Flow Models for Data Imputation Year: (2020-06)
Ref_id:b18 Title: Should EBMs model the energy or the score? Year: (2021)
Ref_id:b19 Title: Clustering via mode seeking by direct estimation of the gradient of a log-density Year: (2014)
Ref_id:b20 Title: Generative Modeling by Estimating Gradients of the Data Distribution Year: (2019)
Ref_id:b21 Title: Sliced score matching: A scalable approach to density and score estimation Year: (2020-07)
Ref_id:b22 Title: Maximum likelihood training of score-based diffusion models Year: (2021)
Ref_id:b23 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b24 Title: On autoencoders and score matching for energy based models Year: (2011)
Ref_id:b25 Title: CSDI: Conditional score-based diffusion models for probabilistic time series imputation Year: (2021)
Ref_id:b26 Title: Imputation estimators for unnormalized models with missing data Year: (2020-08)
Ref_id:b27 Title: A connection between score matching and denoising autoencoders Year: (2011)
Ref_id:b28 Title: Learning doubly intractable latent variable models via score matching Year: (2016)
Ref_id:b29 Title: Robust gaussian graphical modeling with the trimmed graphical lasso Year: (2015)
Ref_id:b30 Title: Gain: Missing data imputation using generative adversarial nets Year: (2018)
Ref_id:b31 Title: Graphical models for non-negative data using generalized score matching Year: (2018-04)
Ref_id:b32 Title: Generalized score matching for general domains. Information and Inference: A Year: (2022)
