Title: Hide & Seek: Transformer Symmetries Obscure Sharpness & Riemannian Geometry Finds It
Abstract: The concept of sharpness has been successfully applied to traditional architectures like MLPs and CNNs to predict their generalization. For transformers, however, recent work reported weak correlation between flatness and generalization. We argue that existing sharpness measures fail for transformers, because they have much richer symmetries in their attention mechanism that induce directions in parameter space along which the network or its loss remain identical. We posit that sharpness must account fully for these symmetries, and thus we redefine it on a quotient manifold that results from quotienting out the transformer symmetries, thereby removing their ambiguities. Leveraging tools from Riemannian geometry, we propose a fully general notion of sharpness in terms of a geodesic ball on the symmetrycorrected quotient manifold. In practice, we need to approximate the geodesics. Doing so up to first order yields existing adaptive sharpness measures, and we demonstrate that including higher-order terms is crucial to recover correlation with generalization. We present results on diagonal nets with synthetic data and show that our geodesic sharpness reveals strong correlation with generalization for real-world transformers on both text and image classification tasks.

Section: Introduction
Predicting generalization of neural nets (NNs)-the discrepancy between training and test set performance-remains an open challenge. Generalization-predictive metrics are valuable though: they enable explicit regularization of training to enhance generalization (Foret et al., 2021), and provide broader theoretical insights into generalization itself.
There is a long history of hypotheses linking sharpness to generalization, but evidence has been conflicting (Hochreiter & Schmidhuber, 1994;Andriushchenko et al., 2023). Generalization has been speculated as correlating with flatness, but recent evidence has indicated that, in the case of transformers, it has little to no correlation whatsoever. Measures of sharpness have varied widely, ranging from trace of the Hessian to worst-case loss within a local neighborhood, with adaptive and relative variations proposed to address specific challenges (Kwon et al., 2021;Petzka et al., 2021).
We suspect that some of the confusion stems from the specificity of the problem these measures have attempted to address: the issue of parameter rescaling. In contrast, we argue that rescaling (Dinh et al., 2017) is merely a special case of a broader, more fundamental obstacle to measuring sharpness accurately: the presence of full and continuous parameter symmetries. Addressing this challenge is crucial to ensure that we are studying the right quantity when investigating the relationship between sharpness and generalization.
Beyond discrete permutation symmetries, neural nets naturally exhibit continuous symmetries in their parameter space. These symmetries are intrinsic, data-independent properties that emerge from standard architectural components. For example: normalization layers (Ioffe & Szegedy, 2015;Ba et al., 2016;Wu & He, 2018) induce scale invariance on the pre-normalization weights (Salimans & Kingma, 2016); homogeneous activation functions like ReLU introduce re-scaling symmetries between pre-and post-activation weights (Dinh et al., 2017); some normalization layers and softmax impose translation symmetries in the preceding layer's biases (Kunin et al., 2021). As a result, arguably almost any NN, along with its corresponding loss, exhibit symmetries and can therefore represent the same function using different parameter values (Figure 1a).
Adaptive flatness (Kwon et al., 2021) accounts for some symmetries, both element-and filter-wise re-scaling, but fails to capture the attention mechanism's full symmetry, represented by GL(h) (re-scaling by invertible h × h matrices, where h is the hidden dimension), as we will discuss later. Aiming to break the cycle between discovery of a specific symmetry and techniques to deal with it, we ask:
Can we provide a one-size-fits-many recipe for developing symmetry-invariant quantities for a wider range of continuous symmetries?
Here, we positively answer this question by proposing a principled approach to eliminate ambiguities stemming from symmetry. Essentially, this boils down to using the geometry that correctly captures symmetry-imposed parameter equivalences. We apply concepts from Riemannian geometry to work on the Riemannian quotient manifold implied by a symmetry group (Boumal, 2023, §9). We thus identify objects on the quotient manifold-like the Riemannian metric and gradient-and show how to translate them back to the Euclidean space. Our contributions are the following:
(a) We introduce the application of Riemannian geometry (Boumal, 2023) to the study of NN parameter space symmetries by using geometry from the quotient manifold induced by a symmetry as a general recipe to remove symmetry-induced ambiguities in parameter space. We do so by translating concepts like gradients from the quotient manifold back to the original space through horizontal lifts.
(b) Based on (a), we propose and analyze geodesic sharpness, a novel adaptive sharpness measure: By Taylorexpanding our refined geometry, we show that (i) symmetries introduce curvature into the parameter space, which (ii) results in previous adaptive sharpness measures when ignored. Geodesic sharpness differs from traditional sharpness measures in two key aspects: (i) the norm of the perturbation parameter is redefined to reflect the underlying geometry; (ii) perturbations follow geodesic paths in the quotient manifold rather than straight lines in the ambient space.
(c) For diagonal nets, we analytically solve geodesic sharpness and find a strong correlation with generalization. Then, we apply our approach to the unstudied and higher-dimensional GL(h) symmetry in the attention mechanism. On both large vision transformers and language models, we empirically find stronger correlation than any previously seen (that we are aware of) between our geodesic sharpness and generalization.
2. Related Work
Symmetry versus reparameterization: Kristiadi et al. (2023) pointed out how to fix ambiguities stemming from reparameterization, i.e. a change of variables to a new parameter space: Invariance under reparameterization follows by correctly transforming the (often implicitly treated) Riemannian metric into the new coordinates. Our work focuses on invariance of the parameter space M under a symmetry group G with action
ψ : G × M → M, (g, θ) → ψ(g, θ)
that operates on a single parameter space.
this section cite: ['b10', 'b2', 'b6', 'b12', 'b32', 'b26', 'b6']

Section: Symmetry teleportation:
Another ways to use symmetryimplied ambiguity is to view it as a degree of freedom and develop adaptation heuristics to improve algorithms which are not symmetry-agnostic (Zhao et al., 2022a).
this section cite: []

Section: Geometric constraints & NN dynamics: Previous stud-
ies analyze how parameter space symmetries impose geometric constraints on derivatives and introduce conserved quantities during training (Kunin et al., 2021). Our approach differs by systematically removing symmetry-induced ambiguity through quotienting out the the symmetry group.
We generalize earlier post-hoc solutions for simpler symmetries (e.g., GL(1)) to more complex, higher-dimensional symmetries such as GL(h), common in neural network attention mechanisms. Unlike Kunin et al. (2021), who consider geometry in augmented spaces for simpler symmetries, we directly use the quotient space geometry. Objects are then 'lifted' back into the original space, yielding symmetrycorrected quantities. This method provides a principled framework capable of handling high-dimensional symmetries, leading to a more effective dimensionality reduction.
Quotient manifolds in deep neural networks: Rangamani et al. (2019) introduce a quotient manifold construction for re-scaling symmetries and then use the Riemannian spectral norm as a measure of worst-case flatness. This differs from our approach in several ways: (a) Our approach is more general and contains both the GL(h) symmetry of transformers, and the original rescaling/scaling symmetry of CNNs/MLPs, rendering it applicable to a wider range of modern architectures. (b) Our experimental setup is more challenging: we test on large-scale models (large transformers vs CNNs) and large-scale datasets (ImageNet vs CIFAR-10). Sharpness measures that account for re-scaling/scaling symmetries (e.g. adaptive sharpness) work quite well on CIFAR-10 with CNNs, and tends to break down on datasets like ImageNet with transformers. (c) Conceptually, Rangamani et al. (2019) defines worstcase sharpness on the usual norm-ball, appropriately generalized to the Riemannian setting. We propose instead that the ball should be the one traced out by geodesics, to better respect the underlying geometry.
(d) Performance-wise, our approach is cheaper as it does not use the Hessian and only uses symmetry-corrected gradients (see Dagréou et al. (2024) for an in-depth cost comparison of computing Hessians vs gradients).
Relative sharpness: Another promising approach to sharpness was proposed by Petzka et al. (2021), where the generalization gap is shown to admit a decomposition into a representativeness term and a feature robustness term. Focusing on the feature robustness term, they introduce relative sharpness, which is invariant to a layer-and neuron-wise re-scaling, and performs better than traditional sharpness measures (Adilova et al., 2023;Walter et al., 2025).
this section cite: ['b5', 'b1', 'b28']

Section: Preliminary Definitions, Notation & Math
Generalization measures: We consider a neural net f θ with parameters θ ∈ R d that is trained on a data set D train using a loss function ℓ by minimizing the empirical risk
L Dtrain (θ) := 1 |D train | (x,y)∈Dtrain ℓ(f θ (x), y) .
Our goal is to compute a quantity on the training data that is predictive of the network's generalization, i.e. performance on a held-out data set.
Sharpness: A popular way to predict generalization is via sharpness-i.e., how much the loss changes when perturbing the weights in a small neighbourhood-like average (S avg ) or worst-case sharpness (S max ) (Keskar et al., 2017)
S avg = E S [L S (θ + δ) -L S (θ)] , δ ∼ N (0, ρ 2 I) , S max = E S max ∥δ∥2≤ρ (L S (θ + δ) -L S (θ)) ,
with batches S ∼ D train of size |S| = m, neighbourhood size ρ, and perturbation δ. Near critical points, they closely relate to the Hessian H (and thus parameter space curvature): S avg ∝ Tr(H), and S max ∝ λ max (H).
Adaptive sharpness: Hessian-based sharpness measures can be made to assume arbitrary values by rescaling parameters, even though the NN function stays the same. To fix this inconsistency, Kwon et al. (2021) proposed adaptive sharpness (invariant under special symmetries), and Andriushchenko et al. (2023) use adaptive notions of sharpness that are invariant to element-wise scaling,
S ad max (w, c) = E S max ∥δ⊘c∥2≤ρ L S (θ + δ) -L S (θ) , (1
)
with scaling vector c (usually set to |θ|, Kwon et al., 2021).
The problem: Adaptive sharpness only considers the symmetry induced by element-wise re-scaling. But symmetries of transformers go beyond the invariance that adaptive sharpness captures. Maybe unsurprisingly, Andriushchenko et al. (2023) find inconsistent trends for adaptive sharpness in transformers, with sharpness failing to correlate with generalisation, versus other architectures. We hypothesize this is related to adaptive sharpness not accounting for the full symmetry in transformers. In this paper, we address this. The central question is: If adaptive sharpness is the fix for a special symmetry, can we provide a more general solution for the symmetries of transformers, to fix the above inconsistency?
this section cite: ['b14', 'b2']

Section: Symmetries in Neural Networks
Here, we give a brief overview and make the notion of NN symmetries more concrete, focusing on those studied by Kunin et al. (2021). Those symmetries lead to rather small effective dimensionality reduction as they are often of GL(1) or GL + (1), but they can still impact the network behaviour. Let θ denote the parameters of a neural net, 1 A a binary mask, and 1 ¬A its complement such that their sum is a vector of ones, 1 A + 1 ¬A = 1. Let θ A := θ ⊙ 1 A with ⊙ the element-wise product. Further, let A 1,2 be two disjoint subsets, A 1 ∩ A 2 = ∅ with masks 1 A1 , 1 A2 . Then we have the following common symmetries, characterized by their symmetry group G, such that for any g ∈ G the parameters ψ(g, θ) and θ represent the same function:
• Translation: ψ(α, θ) = 1 A ⊙ α + θ with α ∈ R h • Scaling: ψ(α, θ) = αθ A + θ ¬A with α ∈ R >0 • Re-scaling: ψ(α, θ) = αθ A1 + 1 /αθ A2 + θ ¬(A1∨A2) with α ∈ R >0
Their associated groups are G = R h , GL + (1), GL + (1). In practice, there may be multiple symmetries acting onto disjoint parameter sub-spaces. Note that re-scaling is essentially the symmetry that adaptive sharpness corrects for.
this section cite: []

Section: Rescale Symmetry of Transformers
Transformers exhibit a higher-dimensional symmetry than the previous examples; we formalize the treatment of this symmetry in the following canonical form.
Definition 3.1 (Functional GL-symmetric building block).
Consider a function f (G, H) on R m×h × R n×h that con- sumes two matrices G ∈ R n×h , H ∈ R m×h but only uses the product GH ⊤ , i.e. f (G, H) = g(GH ⊤ ) for some g over R m×n . f is symmetric under the general linear group GL(h) := A ∈ R h×h | A invertible
with dim(GL(h)) = h 2 and action
ψ(A, (G, H)) = (GA -1 , HA ⊤ ).(3)
In other words, we can insert and then absorb the identity A -1 A into G, H to obtain equivalent parameters GA -1 , HA ⊤ that represent the same function.
Example A.2 illustrates GL symmetry for a shallow linear net. Indeed, many popular NN building blocks feature this form, most prominently the attention mechanism in transformers (Vaswani et al., 2017). We give the attention symmetry in Example A.1, and we provide the symmetry for low-rank adapters (Hu et al., 2022) in Example A.3. These examples are NN building blocks that introduce GL symmetries into a loss function and can all be treated through the canonical form in Definition 3.1. In contrast to the symmetries from Section 3.1, they lead to more drastic dimensionality reduction. Consider for example a single self-attention layer where h = h v = h k . The number of trainable parameters is 4h 2 and the two GL(d) symmetries reduce the effective dimension to 4h 2 -2 dim(GL(h)) = 2h 2 , i.e. they render half the parameter space redundant. We hypothesize that the impact of a low-dimensional symmetry on objects like the Euclidean Hessian's trace (Dinh et al., 2017) may be amplified for such higher-dimensional symmetries.
this section cite: ['b27', 'b36', 'b6']

Section: Mathematical Concepts for Riemannian Geometry
We now outline required properties of manifolds for the full development of our approach. We list essential concepts here, and provide definitions and a brief review Appendix B. For further information, see for instance Lee (2003). Figure 2 illustrates the main concepts we will require.
Ambient embedding space: We assume that the manifold of possible parameters is embedded in a linear Euclidean space E ≃ R d with d the number of parameters. We can think of E as the ambient space. For instance, for a loss function ℓ : E → R, θ → ℓ(θ) , we can use ML libraries to evaluate its value, as well as its Euclidean gradient
grad θ ℓ = ∂ℓ(θ) ∂θ i i=1,...,d ∈ R d .
Because the geometry of E is flat, i.e. uses the standard metric ⟨θ 1 , θ 2 ⟩ := θ ⊤ 1 θ 2 , this object consists of partial derivatives. However, the Riemannian generalization will add correction terms. In what follows we consider only the restriction of objects like ℓ to the parameter manifold.
Definition 3.2. We take M to be the manifold of network parameters, and consider it a sub-manifold embedded into E, the computational space of matrices on which all our numerical calculations are done. We call M the total space. On the total space we have a loss function ℓ : M → R.
Our goal is to calculate derivatives/geometric quantities after removing the NN's symmetries. The symmetry relation induces natural equivalence classes, which we write [θ], and explain in Appendix B.1. We let M = M/ ∼ represent the quotient of the original parameter space manifold by the equivalence relation ∼ associated with the symmetry (Appendix B.2). We also require tangent vectors; these are straightforward on the total space M, but the tangent space of the quotient manifold, M, requires more machinery: vertical and horizontal spaces, and corresponding lifts. These concepts are all defined in Appendix B.3.
Once we endow M with a smooth inner product over its tangent vectors, we obtain a Riemannian manifold (defined in Appendix B.4). This construction lets us analyze differ- x and a vertical component ξV x . The vertical component points along the direction where the quotient space x = [x] remains unaffected. The horizontal component points along the direction that changes the equivalence class. We can use ξH
x x = [x] = x ′ ȳ y x′ ξx ξV x ξH x ξx M E M = M/G E Ambient embedding space M Total space M Quotient space G Symmetry group x, ȳ
x as a representation of the tangent vector ξ x ∈ T x M on the quotient space. The component ξH
x represents the horizontal lift of ξ x .
ential objects that live on quotient manifolds, in the ambient space in a natural way. Furthermore, this allows us to define the horizontal space as the orthogonal complement of the vertical space (Appendix B.4), and to define a Riemannian gradient (Appendix B.5). Most properties from the Euclidean case still hold for the Riemannian gradient, but of particular interest to us is the fact that the direction gradf (x) is still the steepest-ascent direction of f at a point x.
We additionally make use of geodesic curves. Intuitively, geodesic curves can either be seen as curves of minimal distance between two points on a manifold M, or equivalently, as curves through a given point with some initial velocity, and whose acceleration is zero-a generalization of Euclidean straight lines. See Appendix B.6 for details.
Putting it all together, this gives us a recipe for computing quantities invariant to a given symmetry relation: (i) find a Riemannian metric compatible with this symmetry; (ii) determine the vertical space for the symmetry relation; (iii) use the metric to find the orthogonal complement of this vertical space, i.e. the projector into the horizontal space;
(iv) find the horizontal geodesics. Combined, these steps allow us to do calculations in the quotient manifold along the proper paths (given by geodesics).
this section cite: ['b19']

Section: Geodesic Sharpness
We posit that adaptive sharpness measures should take into account the geometry of the quotient parameter manifold that arises after removing symmetries from the parameter space. We base our sharpness measure on the notion of a geodesic ball: the set of points that can be reached by geodesics, starting at a point p and whose initial velocity has a norm smaller than ρ, after one time unit. In R d this is just the usual definition of a ball, since the geodesics are straight lines. If ξ ∈ H xM is a horizontal vector, and γ(t) is a geodesic starting at θ and with initial velocity ξ:
S ρ max (w) = E S max ∥ ξ∥ γ(0) ≤ρ L S (γ ξ(1)) -L S (γ ξ(0)) . (4)
If the initial velocity, ξ, is a horizontal vector, then the velocity of the geodesic, γ ξ, will stay horizontal. The choice of t = 1 in γξ(1) is not as arbitrary as it first seems (do Carmo, 1992): since for a positive a, γ ξ(at) = γa ξ(t), positions reached with arbitrary t can be reached by instead fixing t = 1 and manipulating the initial velocity's norm via ρ.
When we do not have an analytical solution for the geodesic, we can use the approximation:
γi ξ(t) = γi ξ(0) + ξi t - 1 2 Γ i kl ξk ξl t 2 + O( ξ3 ) , (5
)
where ξ = ( ξi ) is the initial (horizontal) velocity, and Γ i kl are the Christoffel symbols. We show that geodesic sharpness reduces to adaptive sharpness measures in Appendix F, under appropriate metric choices and by taking a first-order approximation to the geodesics, that is, ignoring the terms corresponding to the curvature, Γ i kl .
this section cite: []

Section: Geodesic Sharpness in Practice
We now apply geodesic sharpness to concrete examples. A fully worked out scalar toy model is in Appendix D.
Following previous works by Dziugaite et al. (2020); Kwon et al. (2021); Andriushchenko et al. (2023), we use the Kendall rank correlation coefficient (Kendall, 1938) to assess the correlation between generalization and sharpness in the empirical validations of our approach:
τ (t, s) = 2 M (M -1) i<j sign(t i -t j ) sign(s i -s j ) ,
where t and s are the vectors of observed variables between which we are measuring correlation.
Although the criterion of symmetry compatibility restricts the class of suitable metrics, these are not necessarily unique. As long as it is symmetry-compatible, we have no reason to prefer one metric over another, except for practical aspects like numerical cost and stability. We will present results on two symmetry-compatible metrics that are simple, yet nontrivial, and often used in the related literature on Riemannian optimization on fixed-rank matrix spaces (Luo et al., 2023).
this section cite: ['b8', 'b2', 'b13', 'b20']

Section: Diagonal Networks
We start by studying diagonal linear nets, one of the simplest non-trivial neural networks (Pesme et al. (2021), Woodworth et al. ( 2020)). These have two parameters, u, v, and predict a label, y, given an input, x, via y = x ⊤ (u ⊙ v). We consider linear regression with labels y ∈ R n , a data matrix X ∈ R n×d , and take as our loss
L(u, v) = ∥X(u ⊙ v) -y∥ 2 2 . Our parameter manifold M is R d × R d . The nets are symmetric under element-wise rescaling: (u, v) → (αu, α -1 v), leaves β = u ⊙ v and L invariant.
Metric: At a point (u, v) ∈ M, for two tangent vectors η = (η u , η v ), ν = (ν u , ν v ) ∈ T (u,v) M, we use the following two symmetry-compatible metrics:
⟨η, ν⟩ inv := d i=1 η i u ν i u (u i ) 2 + η i v ν i v (v i ) 2 , ,(6)
⟨η, ν⟩ mix := d i=1 η i u ν i u (v i ) 2 + η i v ν i v (u i ) 2 .(7)
Horizontal space: Both have the same horizontal space
H (u,v) M = (η u , η v ) ∈ T (u,v) M | η i u u i = η i v v i ∀i . Geodesics: With b i := η i u u i = η i v v i , the geodesics are γ inv (t) i = u i 0 exp(b i t), v i 0 exp(b i t) , γ mix (t) i = u i 0 1 + 2b i t, v i 0 1 + 2b i t ,
with starting points u i 0 and v i 0 , i.e. the trained parameters. The minimum norm least squares predictor is
β * := (X ⊤ X) -1 X ⊤ y = X ⊤ y.
Using Equation ( 4) (details in Appendix E), we get (to first and to second order)
S ρ max; inv (u, v) =4ρ∥β 0 ⊙ (β 0 -β * )∥ 2 + 4ρ 2 max (β i 0 ) 2 ,(8)
which depends on ρ and the difference between the learned, and the optimal minimum norm, predictor. Eq. 8 is the square of adaptive sharpness (when the residual ∥β 0 ⊙(β 0β * )∥ 2 is small) if very carefully chosen hyperparameters were used (by contrast, this result naturally appears using our geodesic approach). For the second metric, we have
S ρ max; mix (u, v) = ρ∥β 0 -β * ∥ 2 .
this section cite: ['b22']

Section: EMPIRICAL VALIDATION
Experimental setup: We follow Andriushchenko et al. (2023), generate a randomly distributed data matrix X, a random ground-truth vector β * that is 90% sparse, and train 50 diagonal networks to 10 -5 training loss on a regression task.
We focus on the more practically relevant case of overparametrization (d > n). One downside of this is that the theoretical expressions derived in the previous section, while a useful sanity check, no longer hold (since overparameterization breaks the assumption X ⊤ X = I d=200 ). To obtain our geometric sharpness, we directly solve Equation ( 4).
Results: All three notions of sharpness are able to predict generalization (Figure 3). Geodesic sharpness, although closely related for diagonal nets to adaptive worst-case sharpness, does slightly better. This applies to both metrics studied, and they perform roughly the same. See Section 7 for comments about the sign of the correlation.
this section cite: ['b2']

Section: Attention Layers
Next, we look at the symmetric functional block from Definition 3.1. Our computation space is E := R n×h ×R m×h ≃ R (n+m)h and we restrict weights to have full column rank: Assumption 5.1. The rank of G, H corresponds to their number of columns, rank(G) = rank(H) = h.
This implies h ≤ n, m, which is usually satisfied in (multihead) attention layers (Example A.1) for the default choices of d v , d k . While the weights of multi-head attention tend to have high column rank (Yu & Wu, 2023), they are not guaranteed to be full column rank. To account for this, we introduce a small relaxation parameter, ϵ, to the Gram matrices s.t.
G ⊤ G → G ⊤ G + ϵI h .
Empirically, we observe that as long as ϵ is sufficiently small, it does not affect our results (Appendix H.2). Therefore, we restrict both G, H to the set of fixed-rank matrices,
M ← R n×h h × R m×h h where R n×h k := B ∈ R n×h | rank(B) = k . We can represent a point x ∈ M by a matrix tuple (G, H) ∈ R n×h h × R m×h h . Its tangent space T xM is T xM = η = ( ηG , ηH ) ∈ R n×h × R m×h , Metric: We endow M with the two metrics ⟨•, •⟩ inv,mix x : T xM × T xM → R (proof they are valid in Appendix I.1): ⟨ η, ζ⟩ inv x := Tr (G ⊤ G) -1 η⊤ G ζG + (H ⊤ H) -1 η⊤ H ζH ,(9)
⟨ η, ζ⟩ mix x := Tr (H ⊤ H) η⊤ G ζG + (G ⊤ G) η⊤ H ζH .(10)
They differ from the Euclidean metric that simply flattens and concatenates the matrix tuples into vectors and takes their dot product, ⟨η,
ζ⟩ = Tr η ⊤ G ζ G + η ⊤ H ζ H .
Importantly, they are invariant under symmetries of the attention mechanism, and thus define valid metrics on the quotient manifold (Absil et al., 2008).
Horizontal space: For ⟨•, •⟩ inv, mix x and ξG,H ∈ R m×r we have (for a proof, see for example Luo et al. (2023))
H inv x M = {( ξG , ξH ) | ξ⊤ G GH ⊤ H = G ⊤ GH ⊤ ξ ⊤ H } , H mix x M = {( ξG , ξH ) | G ⊤ ξG H ⊤ H = G T Gξ ⊤ H H} .
Projection onto horizontal space: Given ξ ∈ T x M in the total tangent space, the horizontal space is
H inv, mix x M = ( ξG + GΛ inv, mix , ξH -H(Λ inv, mix ) ⊤ )
where Λ inv solves the Sylvester equation
AΛ + ΛA ⊤ = B, with A = G ⊤ GH ⊤ H, B = G ⊤ GH ⊤ ξH - ξ⊤ G GH ⊤ H, whereas Λ mix has an explicit form: Λ mix = 1 /2 ξ⊤ H H(H ⊤ H) -1 -(G ⊤ G) -1 G ⊤ ξG .
Geodesics: We are unaware of analytical solutions for the geodesics of either (Eq. 9 and Eq. 10), so we approximate them with Eq. 5. For horizontal tangent vectors ( ξG , ξH ), we have for ⟨
•, •⟩ inv x (Γ i kl ) inv ξk G ξl G = -ξG (G ⊤ G) -1 ξ⊤ G G + G ⊤ ξG + G(G ⊤ G) -1 ξ⊤ G ξG(11)
(similar for the H components). For ⟨•, •⟩ mix x , the geodesic equations are coupled and the G components are
(Γ i kl ) mix ξk ξl G = ξG ξ⊤ H H + H ⊤ ξH (H ⊤ H) -1 -G( ξT H ξH )(H ⊤ H) -1(12)
(the H components are similar, proof in Appendix I.2).
this section cite: ['b34', 'b0', 'b20']

Section: Transformers
Transformers have a mix of attention layers and layers with more restricted symmetries for which adaptive sharpness is more appropriate. We present in Appendix C.1 how we treat each layer of transformers. We introduce relaxations In Appendix C.2 we present Algorithm 1, which we use to solve for geodesic sharpness.
this section cite: []

Section: EMPIRICAL VALIDATION: VISION TRANSFORMERS
Experimental setup: We follow Andriushchenko et al. (2023), and look at models obtained from fine-tuning CLIP on ImageNet-1k (Radford et al., 2021). Specifically, we use the trained classifiers after fine-tuning a CLIP ViT-B/32 on ImageNet with randomly selected hyperparameters from (Wortsman et al., 2022). We compute adaptive worst-case, and our geodesic, sharpness on the same 2048 data points from the ImageNet training set, divided into batches of 256, by calculating sharpness on each batch separately, then averaging the results. The generalization gap is the difference between test and training error.
Results: Figure 4 shows our results. We find a strong correlation between geodesic sharpness and the generalization gap on ImageNet. This correlation is stronger than
0.020 0.022 0.024 Adaptive 0.102 0.104 0.106 0.108 Generalization Gap τ =0.06 6.5 7.0 7.5 Geodesic (inv) τ =0.28 5.00 5.25 5.50 Geodesic (mix) τ =0.38  2020, and show the generalization gap on the MNLI dev matched set (Williams et al., 2018). Geodesic sharpness shows the largest correlation.
that observed with adaptive sharpness and is consistently negative, implying that the geodesically sharpest models studied on ImageNet are those that generalize best-contrary to what might have been expected, but consistent with the correlation from the diagonal networks.
this section cite: ['b2', 'b24', 'b31', 'b29']

Section: EMPIRICAL VALIDATION: LANGUAGE MODELS

this section cite: []

Section: Experimental Setup:
We also consider BERT models that were fine-tuned on MNLI (Williams et al., 2018) by Mc-Coy et al. (2020) . We compute adaptive worst-case, and our geodesic, sharpness on the same 1024 data points from the MNLI training set, with batches of 128 points, by calculating then averaging sharpness on each batch.
Results: Figure 5 shows our results. We find a consistent correlation between geodesic sharpness and the generalization gap on MNLI for both metrics, while adaptive sharpness (τ = 0.06) cannot find any correlation. The correlation is positive, i.e. geodesically flatter models generalize better.
this section cite: ['b29']

Section: Additional Experiments

this section cite: []

Section: Comparison With Relative Sharpness
Relative sharpness (Petzka et al., 2021) is a promising sharpness measure that has proven useful in regularizing transformer training, outperforming other approaches (Adilova et al., 2023). We compare it with our geodesic sharpness in the language model setting from Section 5.3.2; see Figure 6.
this section cite: ['b1']

Section: Verification of Reparametrization Invariance
Mathematically, geodesic sharpness is invariant to symmetry transformations of the form of Equation (3). Here, we verify empirically that our practical version that can be computed efficiently numerically is close to invariant.
this section cite: []

Section: Experimental setup:
We take a single batch and language model from Section 5.3.2, and compute geodesic sharpness
0.020 0.022 0.024 Adaptive 0.102 0.104 0.106 0.108 Generalization Gap τ =0.06 5.00 5.25 5.50 Geodesic (mix) τ =0.38 4 6
Relative τ =-0.09
Figure 6: Extension of Figure 5 to relative sharpness. We find that relative flatness (Petzka et al., 2021) fails to find a significant correlation, compared to our geodesic sharpness.
for various points on an orbit that represent the same function. Specifically, we reparametrize using A = aG, where G is a random standard Gaussian matrix (which is almost always invertible and sampled once in each run), and control the scale a. We sample one G for each attention head. We compare this with adaptive sharpness.
Results: Figure 7 visualizes the sharpness ratio before and after reparameterization. The colors represent different values of the scale factor, which goes from 10 -2 to 10 2 . Our numerically computed geodesic sharpness remains constant.
10 -2 10 -1 10 0 10 1 10 2
Scale factor (a)
1.0 1.2 1.4 1.6
this section cite: []

Section: Sharpness ratio
Adapative Geodesic
Figure 7: Variation of adaptive vs. geodesic sharpness within an orbit where the neural net function remains unchanged. We show the ratios between the original sharpness and the sharpness obtained after applying a symmetry transformation. Geodesic sharpness stays constant, whereas adaptive sharpness assumes several different values.
this section cite: []

Section: Remarks, Limitations & Future Work
Discovering correlation: Adaptive sharpness, as discussed thoroughly by Andriushchenko et al. (2023), is unable to reveal a correlation between sharpness and generalization for transformers. Our geodesic sharpness consistently recovers strong correlation on transformers, and strengthens the correlation in the case of diagonal networks.
this section cite: ['b2']

Section: Metric choice:
Our results are robust w.r.t. the choice of metric, as long as it captures the parameter symmetry. The mixed metric yields slightly better results on BERT, perhaps owing to its more stable numerics (e.g. possible inversion of nearly singular matrices is side-stepped). Additionally, the mixed metric avoids calling expensive Sylvester equation solvers and has a simple horizontal space projection.
Sign of the correlation: One of our surprising results is that the sign of the correlation between geodesic sharpness and generalization varies depending on the setting and is at times negative, somewhat at odds with the common view that sharpness always positively correlates with generalization (i.e., flatter models generalize better). This artifact is not inherent to our proposed metrics. E.g., adaptive sharpness anti-correlates with generalization in our diagonal network setting, but was previously found to positively correlate with generalization on other tasks (Kwon et al., 2021).
Our geodesic sharpness improves over adaptive sharpness in the following sense: Where adaptive sharpness finds no correlation, our metrics do find a signed correlation, and where adaptive sharpness finds signed correlation, our metrics find a stronger similarly-signed correlation. That is, we for the first time observe consistent correlations withintask for transformers, opening questions for further study.
Limitations: While our geodesic sharpness is more general than previous measures, there remain symmetries for which taking the quotient may be computationally expensive or intractable. Still, we think that accounting for some symmetry is better than none, and even under computational constraints it could be useful as a diagnostic "probe".
Our new measures detect previously undetected correlation with generalization. In the process, however, we also discovered that the sign of the correlation, while consistent across metrics and models, can vary across tasks. Until this new variability is understood, this limits the utility of geodesic sharpness, e.g. for regularizing transformer training.
Future work: Our work is concerned with accounting for parameter space symmetries that are data-independent. This opens up the question: what is the role of data and how can it be integrated into our framework? A more complete understanding of the interplay between data and parameter symmetries might help explain when geodesic sharpness correlates or anti-correlates with generalization.
this section cite: []

Section: Conclusion
In this paper, we revisited the limitations of traditional sharpness measures attempting to predict generalization for transformers, highlighting how traditional sharpness measures fail to properly account for the rich GL(h) symmetries present in transformers. Addressing this, we introduced geodesic sharpness, a measure defined on the Riemannian quotient manifold obtained by quotienting out transformer symmetries. This framework provides a principled, symmetry-aware measure of sharpness and contains prior adaptive sharpness metrics as first-order approximations.
Through experiments on diagonal networks, vision trans-formers (ImageNet), and language models (MNLI), we demonstrated that properly accounting for the transformer symmetries restores the correlation between sharpness and generalization. Interestingly, our findings indicate that the sign of the correlation between sharpness and generalization can vary across tasks, suggesting deeper underlying relationships involving data distribution and model structure. This work lays the groundwork for further exploration of these interactions and motivates future research into geometryinformed optimization strategies tailored to transformers.
this section cite: []

Section: References
Ref_id:b0 Title: Optimization algorithms on matrix manifolds Year: (2008)
Ref_id:b1 Title: Relative flatness aware minimization Year: (2023)
Ref_id:b2 Title: A modern look at the relationship between sharpness and generalization Year: (2023)
Ref_id:b3 Title: Layer normalization. 2016. Boumal, N. An introduction to optimization on smooth manifolds Year: (2023)
Ref_id:b4 Title: Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks Year: (2020)
Ref_id:b5 Title: How to compute hessian-vector products? Year: (2024)
Ref_id:b6 Title: Sharp minima can generalize for deep nets Year: (2017)
Ref_id:b7 Title:  Year: (1992)
Ref_id:b8 Title: In search of robust measures of generalization Year: (2020)
Ref_id:b9 Title: Sharpnessaware minimization for efficiently improving generalization Year: ()
Ref_id:b10 Title: Simplifying neural nets by discovering flat minima Year: (1994)
Ref_id:b11 Title: Low-rank adaptation of large language models Year: ()
Ref_id:b12 Title: Batch normalization: Accelerating deep network training by reducing internal covariate shift Year: (2015)
Ref_id:b13 Title: A new measure of rank correlation Year: (1938)
Ref_id:b14 Title: On large-batch training for deep learning Year: (2017)
Ref_id:b15 Title: Fast algorithms for the sylvester equation Year: (2001)
Ref_id:b16 Title: The geometry of neural nets' parameter spaces under reparametrization Year: ()
Ref_id:b17 Title: Symmetry, conservation laws, and learning dynamics in neural networks Year: ()
Ref_id:b18 Title: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks Year: ()
Ref_id:b19 Title: Introduction to Smooth Manifolds Year: (2003)
Ref_id:b20 Title: On geometric connections of embedded and quotient geometries in riemannian fixed-rank matrix optimization Year: (2023)
Ref_id:b21 Title: Berts of a feather do not generalize together: Large variability in generalization across models with similar test set performance Year: (2020)
Ref_id:b22 Title: Implicit bias of sgd for diagonal linear networks: a provable benefit of stochasticity Year: (2021)
Ref_id:b23 Title: Relative flatness and generalization Year: ()
Ref_id:b24 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b25 Title: A Scale Invariant Flatness Measure for Deep Network Minima Year: (2019-02)
Ref_id:b26 Title: Weight normalization: A simple reparameterization to accelerate training of deep neural networks Year: (2016)
Ref_id:b27 Title: Attention is all you need Year: (2017)
Ref_id:b28 Title: The uncanny valley: Exploring adversarial robustness from a flatness perspective Year: (2025)
Ref_id:b29 Title: A broad-coverage challenge corpus for sentence understanding through inference Year: (2018)
Ref_id:b30 Title: Kernel and rich regimes in overparametrized models Year: (2020)
Ref_id:b31 Title: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time Year: (2022)
Ref_id:b32 Title: Group normalization Year: (2018)
Ref_id:b33 Title: Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization Year: (2024)
Ref_id:b34 Title: Compressing transformers: Features are lowrank, but weights are not! Proceedings of the AAAI Conference on Artificial Intelligence Year: (2023-06)
Ref_id:b35 Title: Symmetry teleportation for accelerated optimization Year: ()
Ref_id:b36 Title: Penalizing gradient norm for efficiently improving generalization in deep learning Year: (2022)
