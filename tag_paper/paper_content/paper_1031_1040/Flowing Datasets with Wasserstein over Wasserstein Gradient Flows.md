Title: Flowing Datasets with Wasserstein over Wasserstein Gradient Flows
Abstract: Many applications in machine learning involve data represented as probability distributions. The emergence of such data requires radically novel techniques to design tractable gradient flows on probability distributions over this type of (infinitedimensional) objects. For instance, being able to flow labeled datasets is a core task for applications ranging from domain adaptation to transfer learning or dataset distillation. In this setting, we propose to represent each class by the associated conditional distribution of features, and to model the dataset as a mixture distribution supported on these classes (which are themselves probability distributions), meaning that labeled datasets can be seen as probability distributions over probability distributions. We endow this space with a metric structure from optimal transport, namely the Wasserstein over Wasserstein (WoW) distance, derive a differential structure on this space, and define WoW gradient flows. The latter enables to design dynamics over this space that decrease a given objective functional. We apply our framework to transfer learning and dataset distillation tasks, leveraging our gradient flow construction as well as novel tractable functionals that take the form of Maximum Mean Discrepancies with Sliced-Wasserstein based kernels between probability distributions.

Section: Introduction
Probability measures provide a powerful way to represent many data types. For instance, they allow to naturally represent documents (Kusner et al., 2015), genes (Bellazzi et al., 2021), point clouds (Qi et al., 2017;Geuter et al., 2025), images (Sodini et al., 2025), or single-cell data (Persad et al., Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). 2023;Haviv et al., 2024b). Remarkably, it has been shown that one can embed any finite dataset with little or no distortion (Andoni et al., 2018;Kratsios et al., 2023) in the Wasserstein space, i.e., the space of probability distributions (e.g., over a Euclidean space) equipped with the Wasserstein-2 distance from Optimal Transport (OT). This has motivated the use of this space to embed many types of data ranging from words (Vilnis & McCallum, 2015) to knowledge graphs (He et al., 2015;Wang et al., 2022), graphs (Bojchevski & Günnemann, 2018;Petric Maretic et al., 2019), or neuroscience data (Bonet et al., 2023). Therefore, it is essential to develop tools to work on the space of probability measures over probability measures, also known as random measures. In particular, they provide a natural way to represent labeled datasets as mixtures (Alvarez-Melis & Fusi, 2020).
A natural distance on this space is the Wasserstein over Wasserstein distance (WoW) (Nguyen, 2016;Catalano & Lavenant, 2024), also known as the Hierarchical OT distance, which lifts the Wasserstein distance between probability distributions as a ground cost, to define a Wasserstein distance between random measures. The latter has been used for generative modeling applications (Dukler et al., 2019), domain adaptation tasks (El Hamri et al., 2022), comparing documents (Yurochkin et al., 2019) or multilevel clustering (Ho et al., 2017). It has also been used to compare Gaussian mixtures (Chen et al., 2018;Delon & Desolneux, 2020;Wilson et al., 2024) or generic mixtures (Dusson et al., 2023;Chen & Zhang, 2024). However, its poor sample complexity has motivated the development of alternative distance measures, such as those based on Integral Probability Metrics (Catalano & Lavenant, 2024). Nonetheless, this space possesses a rich Riemannian structure, enabling the definition of concepts like geodesics. This has been leveraged recently by Haviv et al. (2024a) to perform generative modeling over the space of probability distributions with Flow Matchings.
While this space naturally supports a range of machine learning tasks, optimization methods tailored to it have received limited attention. Yet, this is important for multiple applications, including variational inference with a Gaussian mixture family (Lambert et al., 2022;Huix et al., 2024), computing barycenters (Delon & Desolneux, 2020), or flowing datasets (Alvarez-Melis & Fusi, 2021), e.g., for domain adaptation, transfer learning (Alvarez-Melis & Fusi, 2021;Hua et al., 2023) or dataset distillation (Wang et al., 2018).
In this paper, we propose to leverage the Riemannian structure of random measures equipped with the WoW distance, by defining and simulating gradient flows, i.e., paths of random measures that follow the steepest descent of a given objective functional.
Related works. An elegant and popular way to perform optimization over probability distributions (over a manifold) is to leverage the Riemannian structure of the Wasserstein space (Otto, 2001), and to use Wasserstein gradient flows (Ambrosio et al., 2008;Santambrogio, 2017). Several time discretizations of these flows have been studied (Jordan et al., 1998;Salim et al., 2020;Bonet et al., 2024), and they have been applied to simulate the flow dynamics of multiple objectives such as the Kullback-Leibler divergence (Wibisono, 2018;Salim et al., 2020;Diao et al., 2023), the Maximum Mean Discrepancy (MMD) (Arbel et al., 2019;Altekrüger et al., 2023;Hertrich et al., 2024a;b) and variants thereof (Glaser et al., 2021;Chen et al., 2024;Neumayer et al., 2024;Chazal et al., 2024) or the Sliced-Wasserstein distance (Liutkus et al., 2019;Du et al., 2023;Bonet et al., 2025). Yet, all these works focus on the case where the probability distributions are defined over a finite-dimensional manifold, e.g. R d . In practice, simulating these flows often boils down to simulating a particle system in R d . Hence, these works do not address probability distributions defined on infinite-dimensional spaces, such as the space of probability measures, which is the focus of this work.
The closest works to ours are the ones of Alvarez-Melis & Fusi (2021) and Hua et al. (2023). These papers cast labeled datasets as measures over a product space of the features and the conditional distributions (i.e., the distributions of the features of a given class). However, they circumvent the issue of designing gradient flows on this space by modeling the conditional probabilities as Gaussian distributions, hence parametrized by a mean and covariance, which are finitedimensional objects. While this enables them to leverage standard Wasserstein gradient flows, this Gaussian modeling of mixture components is a strong assumption that may not capture the true shape of many labeled datasets in practice.
Contributions. In this work, we introduce a principled framework for optimizing functionals over the space of probability measures on probability measures, leveraging the Riemannian structure of this space to develop Wasserstein over Wasserstein (WoW) gradient flows. We provide a theoretical construction of the flows, and then a practical implementation through time discretization using a forward Euler scheme. We also propose a novel functional objective, that writes as an MMD with kernel between distributions based on the Sliced-Wasserstein distance, and whose gradient flow simulation is tractable. We then apply this scheme to flow datasets viewed as random measures; specifically, as mixtures of probability distributions corresponding to the class-conditional distributions. We focus on image datasets, and show that the flow enables structured transitions of classes toward other classes, with applications to transfer learning and dataset distillation.
Notations. For a Riemannian manifold M, d : M × M → R + is its geodesic distance. For x ∈ M, we denote by T x M the tangent space at x, and by ∥ • ∥ x the Riemannian metric. We define by T M = {(x, v), x ∈ M and v ∈ T x M} the tangent bundle. We define for (x, v) ∈ T M the projections π M (x, v) = x and π v (x, v) = v. exp : T M → M is the exponential map. For x ∈ M, if exp x : T x M → M is invertible, we note log x its inverse. ∇ and div refer to the Riemannian gradient and divergence on M. For a metric space (X, d), P 2 (X) denotes the space of probability distributions on X with second finite moments, i.e., P 2 (X) = {µ ∈ P(X), d(x, o) 2 dµ(x) < ∞} with o ∈ X some arbitrary origin. For any µ ∈ P 2 (M), L 2 (µ, T M) is the set of functions v : M → T M such that ∥v(x)∥ 2
x dµ(x) < ∞. For a measurable map T : M → M, we note by T # µ the pushforward measure. Id denotes the identity map on M. P 2,ac (M) ⊂ P 2 (M) is the space of measures absolutely continuous w.r.t. the volume measure on M. For µ, ν ∈ P(X), we denote µ ≪ ν if µ is absolutely continuous w.r.t. ν. Π(µ, ν) = {γ ∈ P(X × X), π 1 # γ = µ, π 2 # γ = ν} with π i : (x 1 , x 2 ) → x i , is the set of couplings, and Π o (µ, ν) the set of optimal couplings.
this section cite: ['b45', 'b9', 'b69', 'b76', 'b18', 'b5', 'b43', 'b80', 'b82', 'b10', 'b68', 'b11', 'b2', 'b64', 'b21', 'b36', 'b39', 'b87', 'b24', 'b33', 'b85', 'b37', 'b23', 'b21', 'b46', 'b33', 'b3', 'b3', 'b83', 'b65', 'b4', 'b75', 'b73', 'b12', 'b84', 'b73', 'b34', 'b6', 'b0', 'b16', 'b61', 'b22', 'b52', 'b35', 'b13', 'b3']

Section: Background
We begin by introducing some background on Optimal Transport (OT) and on Wasserstein Gradient Flows. For theoretical purposes, we provide background on the geometry of (P 2 (M), W 2 ) with M a Riemannian manifold, as in the next section, we will rely on results which hold on compact Riemannian manifolds (without boundary). Nonetheless, the applications will be done for M = R d . The reader may refer to Appendix A for more details.
Optimal Transport. The Wasserstein distance between µ, ν ∈ P 2 (M) is defined as
W 2 2 (µ, ν) = inf γ∈Π(µ,ν) d(x, y) 2 dγ(x, y).(1)
The metric space (P 2 (M), W 2 ) has a Riemannian structure (Otto, 2001;Erbar, 2010). In particular, if the log map is well defined µ-almost everywhere (a.e.), (constant-speed) geodesics between µ, ν are defined as µ t = exp π 1 •(t log π 1 •π 2 ) # γ with γ ∈ Π o (µ, ν) an optimal coupling. If µ ∈ P 2,ac (M), there is a map T, namely the OT map, such that T # µ = ν and γ = (Id, T) # µ by McCann's theorem for a wide range of manifolds (McCann, 2001;Figalli, 2007). In particular, T = exp Id •(-∇φ µ,ν )
with φ µ,ν a Kantorovich potential between µ and ν, and geodesics become µ t = exp Id •(-t∇φ µ,ν ) # µ. At any µ ∈ P 2,ac (M), we can define a tangent space
T µ P 2 (M) = {∇φ, φ ∈ C ∞ c (M)} L 2 (µ,T M)
with C ∞ c (M) the space of smooth compactly supported functions on M (Erbar, 2010;Gigli, 2011). This is a Hilbert space endowed with the L 2 (µ, T M) inner product. The exponential map on P 2 (M) is then defined as exp µ (v) = (exp Id •v) # µ for µ ∈ P 2 (M), v ∈ T µ P 2 (M). For instance, when M = R d with d(x, y) 2 = ∥x-y∥ 2  2 , then exp x (y) = x + y and log x (y) = y -x for all x, y ∈ R d .
Let P 2 (T M) := {γ ∈ P(T M), (d(x, o) 2 + ∥v∥ 2
x )dγ(x, v) < ∞} where o ∈ M is any reference point. Following (Gigli, 2011), we define for every µ, ν ∈ P 2 (M), exp -1 µ (ν) := γ ∈ P 2 (T M), π M # γ = µ, exp # γ = ν,
∥v∥ 2 x dγ(x, v) = W 2 2 (µ, ν)(2)
the set of plans γ ∈ P 2 (T M) such that (π M , exp) # γ is an OT plan between µ and ν. This allows one to avoid using the logarithm map, which might not be well defined everywhere, e.g. being multivalued. This space carries more information than the set of optimal couplings as it precises which geodesic was chosen to move the mass, as µ t = exp π M •(tπ v ) # γ are constant speed geodesics between µ and ν (Gigli, 2011, Theorem 1.11). On P 2 (R d ), this translates as exp -1 µ (ν) = {(π 1 , π 2 -π 1 ) # γ, γ ∈ Π o (µ, ν)} (Gigli, 2004;Hertrich et al., 2024a). We show in the next proposition, whose proof can be found in Appendix C.1, that we can build a surjective map from exp -1 µ (ν) to Π o (µ, ν). Proposition 2.1. Let µ, ν ∈ P 2 (M). A surjective map from exp -1 µ (ν) to Π o (µ, ν) is given by γ → (π M , exp) # γ. In particular, exp -1 µ (ν) is not empty, and if γ ∈ exp -1 µ (ν), then d x, exp x (v) = ∥v∥ x for γ-a.e. (x, v) ∈ T M.
Additionally, if M is compact and connected, and µ ∈ P 2,ac (M), then there exists a unique γ ∈ exp -1 µ (ν), of the form γ = (Id, -∇φ µ,ν ) # µ.
Wasserstein Gradient Flows. Let F : P 2 (M) → R be a lower semi-continuous functional. We briefly introduce the differential structure on (P 2 (M), W 2 ), i.e., probability measures on manifolds, inspired by (Erbar, 2010) and (Lanzetti et al., 2025).
Let µ ∈ P 2 (M). We say that ∇ W2 F(µ) ∈ L 2 (µ, T M) is a Wasserstein gradient of F at µ if for any ν ∈ P 2 (M) and any γ ∈ exp -1 µ (ν), we have the Taylor expansion
F(ν) = F(µ) + ⟨∇ W2 F(µ)(x), v⟩ x dγ(x, v) + o W 2 (µ, ν) . (3)
If such a gradient exists, then we say that F is Wasserstein differentiable at µ. There is a unique gradient belonging to T µ P 2 (M) and we restrict to this gradient. Informally, the Wasserstein gradient of F can be computed as ∇ W2 F(µ) = ∇ δF δµ (µ), with δF δµ (µ) the first variation defined, when it exists, as the unique function (up to an additive constant) such that, for χ satisfying dχ = 0, d dt F(µ + tχ) t=0 = δF δµ (µ) dχ (Ambrosio et al., 2008, Lemma 10.4.1). Examples of differentiable functionals include potential energies V(µ) = V dµ and interaction energies W(µ) = W (x, y)dµ(x)dµ(y) for V : M → R and W : M × M → R twice differentiable with bounded Hessian, for which
∇ W2 V(µ) = ∇V and ∇ W2 W(µ)(x) = ∇ 1 W (x, y)+∇ 2 W (y, x) dµ(y).
Moreover, if the functional F : P 2 (R d ) → R has a closed-form over discrete measures, i.e., there exists F :
(R d ) n → R such that F 1 n n i=1 δ xi = F (x 1 , . . . , x n ),
then we can use backpropagation on F and find the Wasserstein gradient of F using the relation
∇ W2 F( 1 n n i=1 δ xi )(x i ) = n∇ i F (x 1 , . . . , x n ) (see Proposition A.9).
A Wasserstein gradient flow of a differentiable functional F is a curve t → µ t which is a (weak) solution of the continuity equation ∂ t µ t = div µ t ∇ W2 F(µ t ) . A possible discretization is the Riemannian Wasserstein gradient descent (Bonnabel, 2013;Bonet et al., 2025), defined as
µ k+1 = exp Id -τ ∇ W2 F(µ k ) # µ k . For discrete distri- butions µ k = 1 n n i=1 δ x k i , it translates as, for all k ≥ 0, i ∈ {1, . . . , n}, x k+1 i = exp x k i -τ ∇ W2 F(µ k )(x k i ) . For M = R d , this is simply x k+1 i = x k i -τ ∇ W2 F(µ k )(x k i )
, which corresponds to Wasserstein gradient descent.
this section cite: ['b65', 'b57', 'b47', 'b14', 'b13']

Section: Wasserstein over Wasserstein Space
We introduce in this section the Wasserstein over Wasserstein space (P 2 P 2 (M) , W W2 ), i.e., the space of probability distributions over probability distributions P 2 P 2 (M) , endowed with the OT distance with the squared Wasserstein distance on P 2 (M) as groundcost. We first state some properties of this distance, and then introduce a differential structure on this space which will be used in the next sections to develop suitable optimization methods. In the following, M is a compact and connected manifold. The proofs can be found in Appendix C.
this section cite: []

Section: OT Distance and Riemannian Structure
The WoW distance is defined as the OT problem with the squared Wasserstein distance on P 2 (M) as groundcost, i.e., for P, Q ∈ P 2 P 2 (M) ,
W W2 (P, Q) 2 = inf Γ∈Π(P,Q) W 2 2 (µ, ν) dΓ(µ, ν). (4)
This defines a distance (Nguyen, 2016). Analogously to P 2 (M) and Brenier-McCann's theorem, it has been shown that there exists an OT map from P to Q under absolute continuity of P with respect to a suitable reference measure P 0 ∈ P 2 P 2 (M) (Emami & Pass, 2025), which has no atom and satisfies an integration by part formula (Dello Schiavo, 2020). We refer to Appendix B for more details.
Now, let us denote for any γ ∈ P 2 (T M), the projections ϕ M (γ) = π M # γ, ϕ exp (γ) = exp # γ and ϕ v (γ) = π v # γ. For any P, Q ∈ P 2 P 2 (M) , let us also define
exp -1 P (Q) := Γ ∈ P 2 P 2 (T M) , ϕ M # Γ = P, ϕ exp # Γ = Q, ∥v∥ 2 x dγ(x, v)dΓ(γ) = W 2 W2 (P, Q) . (5
)
Relying on Proposition 2.1, we can define for any P, Q ∈ P 2 P 2 (M) a surjective map from exp -1
P (Q) to Π o (P, Q). Proposition 3.1. Let P, Q ∈ P 2 P 2 (M) . Then, Γ → (ϕ M , ϕ exp ) # Γ is a surjective map from exp -1 P (Q) to Π o (P, Q). In particular, exp -1 P (Q) is not empty and if Γ ∈ exp -1 P (Q), γ ∈ exp -1 π M # γ (exp # γ)
for Γ-a.e. γ. Additionally, if P ≪ P 0 , there exists a unique Γ ∈ exp -1 P (Q), of the form (µ → (Id, -∇φ µ,T(µ) ) # µ) # P with T the unique transport map from P to Q and φ µ,T(µ) a Kantorovich potential between µ, T(µ) ∈ P 2 (M).
The proof of Proposition 3.1 can be found in Appendix C.2. The previous construction enables us to formalize the Riemannian structure of (P 2 P 2 (M) , W W2 ), without having to define a notion of logarithm map on P 2 (M), which might be ill-defined when the OT plan is not unique. Between P, Q ∈ P 2 P 2 (M) , we can define for Γ ∈ exp -1 P (Q) a geodesic t → P t = exp ϕ M •(tϕ v ) # Γ, which satisfies for all s, t ∈ [0, 1], W W2 (P s , P t ) = |t -s|W W2 (P, Q), see Appendix B. For P ≪ P 0 , using Proposition 3.1, the curve simplifies as P t = exp Id •(-t∇φ Id,T ) # P. Moreover, for M = R d , this reads as P t = µ → (Id -t∇φ µ,T(µ) ) # µ # P.
this section cite: ['b64', 'b40']

Section: Differential Structure
We now provide a differential structure to (P 2 P 2 (M) , W W2 ), following the one of (Ambrosio et al., 2008;Erbar, 2010;Lanzetti et al., 2025) for (P 2 (M), W 2 ). In this section, let F : P 2 P 2 (M) → R be a lower semi-continuous functional. We define formally the Hilbert space L 2 (P, T P 2 (M)) of functions from P 2 (M) to T P 2 (M) in Appendix B.1. First, we define the notions of (extended) sub-and super-differential. Definition 3.2. ξ ∈ L 2 (P, T P 2 (M)) belongs to the subdifferential ∂ -F(P) of F at P if for all Q ∈ P 2 P 2 (M) ,
F(Q) ≥ F(P)+sup Γ ⟨ξ(π M # γ)(x), v⟩ x dγ(x, v)dΓ(γ) + o W W2 (P, Q) , (6
)
where the Γ in the sup are selected in exp -1 P (Q). Similarly, ξ ∈ L 2 (P, T P 2 (M)) belongs to the super-differential ∂ + F(P) of F at P if -ξ ∈ ∂ -(-F)(P).
If the functional admits a sub-and super-differential, which coincide, we can define a gradient. Definition 3.3. F is Wasserstein differentiable at P ∈ P 2 P 2 (M) if ∂ + F(P) ∩ ∂ -F(P) ̸ = ∅. In this case, we say that ξ ∈ ∂ -F(P) ∩ ∂ + F(P) is a WoW gradient of F at P, and it satisfies for any Q ∈ P 2 P 2 (M) , Γ ∈ exp -1 P (Q),
F(Q) = F(P) + ⟨ξ(π M # γ)(x), v⟩ x dγ(x, v)dΓ(γ) + o W W2 (P, Q) . (7)
In the following, we note ∇ WW 2 F(P) such a gradient.
We can also define a notion of strong sub-and superdifferential, as well as gradient, by allowing the coupling Γ to be non-optimal, in contrast with the previous definitions.
Definition 3.4. ξ ∈ L 2 (P, T P 2 (M)) is a strong subdifferential of F at P if for all Q ∈ P 2 (P 2 (M)), for all
Γ ∈ P 2 P 2 (T M) s.t. ϕ M # Γ = P, ϕ exp # Γ = Q, F(Q) ≥ F(P) + ⟨ξ(π M # γ)(x), v⟩ x dγ(x, v)dΓ(γ) + o ∥v∥ 2 x dγ(x, v)dΓ(γ) . (8)
Strong superdifferentials and gradients are defined similarly.
The latter definition is particularly useful when perturbing a measure along a non-optimal direction, as in the case of the forward Euler schemes we will compute in the next section.
We now turn to examples of functional on P 2 P 2 (M) that take the form of free energies. Given F : P 2 (M) → R, we define a potential energy V :
P 2 P 2 (M) → R as V(P) = F(µ)dP(µ). Analogously to classical Wasserstein gradients, its WoW gradient is obtained as ∇ WW 2 V(P)(µ) = ∇ W2 F(µ).
Given a kernel W : P 2 (R d ) × P 2 (R d ) → R, we define interaction energies as W(P) = W(µ, ν) dP(µ)dP(ν), and their WoW gradients are obtained as ν). We refer to Appendix B.4 for more details.
∇ WW 2 W(P)(µ) = ∇ W2,1 W(µ, ν) + ∇ W2,2 W(ν, µ) dP(
Let us now define cylinder functions, which provide a class of Wasserstein differentiable functionals (von Renesse & Sturm, 2009;Dello Schiavo, 2020;Fornasier et al., 2023). Definition 3.5. A functional F : P 2 (M) → R is a cylinder if there exists k ≥ 0, F ∈ C ∞ c (R k ) and V 1 , . . . , V k ∈ C ∞ c (M) such that, for all µ ∈ P 2 (M),
F(µ) = F V 1 dµ, . . . , V k dµ .(9)
In this case, we note F ∈ Cyl P 2 (M) . Similarly, for I an interval, we note F ∈ Cyl I × P 2 (M) , if F(t, µ) = F t, V 1 dµ, . . . , V k dµ for every t ∈ I and µ ∈ P 2 (M), this time for some F ∈ C ∞ c (I × R k ).
Using the chain rule, any F ∈ Cyl P 2 (M) is Wasserstein differentiable and for all µ ∈ P 2 (M),
∇ W2 F(µ) = k i=1 ∂ ∂x i F V 1 dµ, • • • , V k dµ ∇V i .(10)
This provides the main building block for defining a tangent space, in which we will show that WoW gradients reside.
Definition 3.6. The tangent space at P ∈ P 2 P 2 (M) is
T P P 2 P 2 (M) = {∇ W2 φ, φ ∈ Cyl P 2 (M) } (11)
where the closure is taken in the space L 2 (P, T P 2 (M)).
We now justify the definition of this tangent space. We show the existence of velocity fields (v t ) t belonging to the latter, associated to any absolutely continuous curves (P t ) t , such that the pair (v t , P t ) t satisfy a continuity equation. We recall that a curve (P t ) t∈[0,1] is absolutely continuous if there exists g ∈ L 1 ([0, 1]) such that W W2 (P s , P t ) ≤ t s g(u)du, and its metric derivative is |P ′ |(t) = lim h→0 1 h W W2 (P t+h , P t ), which exists a.e. (Ambrosio et al., 2008, Th. 1.1.2).
Proposition 3.7. Let (P t ) t∈I be an absolutely continuous curve on P 2 P 2 (M) . Then, for a.e. t ∈ I, there exists v t ∈ T Pt P 2 P 2 (M) such that ∥v t ∥ L 2 (Pt,T P2(M)) ≤ |P ′ |(t) and for all φ ∈ Cyl(I × P 2 (M)),
∂ t φ t (µ) + ⟨∇ W2 φ t (µ), v t (µ)⟩ L 2 (µ) dP t (µ)dt = 0. (12
)
The proof of Proposition 3.7 is deferred to Appendix C.3. We leave the investigation of the converse implication to future work, i.e. that satisfying the (weak) continuity equation ( 12) implies absolute continuity of the curve (P t ) t . We then have the following properties, which show that elements of the tangent space are strong gradients and are unique. Their proofs are deferred to Appendix C.4 and Appendix C.5.
Proposition 3.8. Let ξ ∈ ∂ -F(P) ∩ T P P 2 P 2 (M) . Then ξ is a strong subdifferential of F at P.
this section cite: ['b4', 'b47', 'b81', 'b32', 'b76']

Section: Proposition 3.9.
There is at most one element in ∂ -F(P) ∩ ∂ + F(P) ∩ T P P 2 P 2 (M) .
As L 2 (P, T P 2 (M)) and the tangent space are Hilbert spaces, one can always decompose a WoW gradient with a part in T P P 2 P 2 (M) and another part orthogonal to it. We show in Appendix C.5 that under technical assumptions, this orthogonal part has a null contribution in the Taylor expansion given in (7). Thus, in this case, we can restrict ourselves to the unique WoW gradient belonging to the tangent space, in particular to write optimization schemes.
this section cite: []

Section: WoW Gradient Flows
In this section, we aim at minimizing F : P 2 P 2 (R d ) → R some functional. We first show the existence of the WoW gradient flow of this functional as the limit of the JKO scheme (Jordan et al., 1998) for F convex along generalized geodesics (Ambrosio et al., 2008). Then, building on the differentiable structure of the space introduced earlier, we propose a forward (explicit) scheme that is computationally more efficient in practice than the implicit JKO scheme, and tractable for relevant functionals.
this section cite: ['b4']

Section: Optimization Schemes on P
2 P 2 (R d )
this section cite: []

Section: JKO Scheme.
Let P 0 ∈ P 2 P 2,ac (R d ) . The JKO sheme of F is defined, for all k ≥ 0 and τ > 0, as
P k+1 ∈ argmin P∈P2(Pac(R d )) 1 2τ W W2 (P, P k ) 2 + F(P). (13
)
Its Wasserstein gradient flow is defined as the limit when τ → 0. Leveraging (Ambrosio et al., 2008, Theorem 4.0.4), we show in the next Proposition the existence of the flow for functionals F that are λ-convex along generalized geodesics P t = (1 -t)T π 2 π 1 + tT π 3 π 1 # π 1 # Γ between Q, O ∈ P 2 P 2 (R d ) , where Γ ∈ Π(P, Q, O) satisfies π 1,2 # Γ ∈ Π o (P, Q), π 1,3 # Γ ∈ Π o (P, O) with π 1,2 : (x, y, z) → (x, y), π 1,3 : (x, y, z) → (x, z) and P ∈ P 2 P 2,ac (R d ) . Since P ∈ P 2 P 2,ac (R d ) , there is always an OT map starting from µ ∼ P towards any ν ∈ P 2 (R d ), which we write T ν µ . Proposition 4.1. Let λ ≥ 0. Let F : P 2 P 2 (R d ) → R be proper, coercive, lower-semi continuous and λ-convex along generalized geodesics, i.e., satisfying for all t ∈ [0, 1],
F(P t ) ≤ (1-t)F(P 0 )+tF(P 1 )- λt(1 -t) 2 W 2 W2 (P 0 , P 1 ), (14
) for P t = (1 -t)T π 2 π 1 + tT π 3 π 1 # π 1 # Γ, Γ ∈ Π(P, Q, O) that satisfies π 1,2
# Γ ∈ Π o (P, Q), π 1,3 # Γ ∈ Π o (P, O) and P ∈ P 2 P 2,ac (R d ) . Then, the gradient flow of F exists and is unique.
The proof of Proposition 4.1 can be found in Appendix C.6. Examples of λ-convex F on P 2 (P 2 (R d )) include potential energies for any F λ-convex along generalized geodesics on P 2 (R d ), and interaction energies for λ = 0 and W jointly convex along generalized geodesics, see Appendix B.5.
this section cite: ['b4']

Section: Forward Scheme.
Given the existence of the WoW gradient of F, as established in the previous section, we propose an alternative to the implicit JKO scheme: a forward scheme, commonly referred to as Wasserstein gradient descent, defined as follows ∀k ≥ 0, P k+1 = exp P k -τ ∇ WW 2 F(P k ) .
(15) At the "distribution particle" level in P 2 (M), this means that for each distribution µ k ∼ P k , we update it as
µ k+1 = exp µ k -τ ∇ WW 2 F(P k )(µ k ) .(16)
In practice, we will mostly focus on distributions of the form P =foot_0 C C c=1 δ µ c with µ c = 1 n n i=1 δ x c i , which notably include labeled datasets (assuming for simplicity now that all classes c = 1, . . . , C contain n examples). Thus, we apply to each particle in M = R d the update
x c i,k+1 = exp x c i,k -τ ∇ WW 2 F(P k )(µ c k )(x c i,k ) = x c i,k -τ ∇ WW 2 F(P k )(µ c k )(x c i,k ).(17)
We see that there are two levels of interactions for each particle in M: one "intra-class" through the dependence in the distribution µ c and one "inter-class" between the distributions µ c ∼ P through the dependence in P k in the gradient. Thus, we expect to observe an interaction between particles of each distribution µ c , but also between each distribution µ c .
this section cite: []

Section: Examples of Discrepancies
Classical functionals in the study of Wasserstein gradient flows are obtained as linear combinations of potential energies, interaction energies and internal energies (Santambrogio, 2015). We focus here on potential energies and interaction energies. We leave the study of internal energies on this space for future works. We refer to e.g. (von Renesse & Sturm, 2009;Sturm, 2024) for discussions of entropy functionals on this space.
A classical discrepancy to compare probability distributions, which can be written as a sum of a potential energy and an interaction energy, is the Maximum Mean Discrepancy (MMD) (Gretton et al., 2012). Given a positive definite kernel K : P 2 (R d ) × P 2 (R d ) → R, P, Q ∈ P 2 P 2 (R d ) , let F(P) = 1 2 MMD 2 (P, Q) be defined as
F(P) = 1 2 K(µ, ν) d(P -Q)(µ)d(P -Q)(ν) = V(P) + W(P) + cst,(18)
where
V(P) = V(µ)dP(µ), V(µ) = -K(µ, ν)dQ(ν), W(P) = 1 2 K(µ, ν) dP(µ)dP(ν)
and the constant only depends on Q that is fixed. For K, we will use kernels based on the Sliced-Wasserstein (SW) (Rabin et al., 2012;Bonneel et al., 2015), defined between µ, ν ∈ P 2 (R d ) as
SW 2 2 (µ, ν) = S d-1 W 2 2 (P θ # µ, P θ # ν) dσ(θ),(19)
with S d-1 = {θ ∈ R d , ∥θ∥ 2 = 1} the sphere, P θ (x) = ⟨x, θ⟩ the coordinate of the projection of x ∈ R d on the line θR for θ ∈ S d-1 , and σ the uniform measure on S d-1 . For instance, positive definite kernels include the Gaussian SW kernel K(µ, ν) = e -SW 2 2 (µ,ν)/h (Kolouri et al., 2016;Carriere et al., 2017;Meunier et al., 2022). We also experiment with the Riesz SW kernel K(µ, ν) = -SW 2 (µ, ν) in analogy with the Riesz kernel (sometimes referred to as negative distance kernel), k(x, y) = -∥x -y∥ 2 on R d × R d , which is not positive definite, but which has demonstrated very good results in practice (Hertrich et al., 2024b) and does not require tuning a bandwidth h.
WoW gradient of the MMD. Given ν ∈ P 2 (R d ), if K ν : µ → K(µ, ν) is a Wasserstein differentiable functional, then F is differentiable, and its WoW gradient at P ∈ P 2 P 2 (R d ) is of the form, for all µ ∈ P 2 (R d ),
∇ WW 2 F(P)(µ) = ∇ W2 K ν (µ) d(P -Q)(ν). (20
)
For the Gaussian SW kernel K(µ, ν) = e -1 2h SW 2 2 (µ,ν) , denoting F(µ) = 1 2 SW 2 2 (µ, ν), its gradient can be obtained by the chain rule as
∇ W2 K ν (µ) = - 1 h e -1 2h SW 2 2 (µ,ν) ∇ W2 F(µ),(21)
where ∇ W2 F(µ) = S d-1 ψ ′ θ (⟨x, θ⟩)θ dσ(θ) with ψ θ the Kantorovich potential between P θ # µ and P θ # ν (Bonnotte, 2013, Proposition 5.1.7). In practice, the Sliced-Wasserstein distance, involving an integral over the sphere, is approximated through Monte Carlo. Moreover, for discrete measures P = 1 C C c=1 δ µ c,n and Q = 1 C C c=1 δ ν c,n with µ c,n = 1 n n i=1 δ x c i and ν c,n = 1 n n i=1 δ y c i , we use autodifferentiation over x := (x c i ) i,c of F (x) = F(P), and rescale the Euclidean gradient of F by n × C to obtain the WoW gradient
∇ WW 2 F(P)(µ c,n )(x c i ) = nC∇ i,c F (x)
. This is analogous to the Wasserstein gradient case, and coincides with the WoW gradient for functionals with a closed-form over discrete measures (see Proposition B.7).
this section cite: ['b74', 'b81', 'b77', 'b70', 'b15', 'b42', 'b20', 'b58']

Section: Applications
In this section, we minimize the MMD on P 2 P 2 (R d ) to solve various tasks 1 . We represent labeled datasets with C classes as distributions P = 1
this section cite: []

Section: Gaussian Iter 0
Iter 10 Iter 25 Iter 100 Riesz Figure 1: Minimization of F(P) = 1 2 MMD 2 (P, Q) with Q a mixture of 3 rings, and with kernels either the Gaussian SW kernel with bandwidth h = 0.05 or Riesz SW kernel, for a learning rate of τ = 0.1. We observe that they first form a ring for each distribution, and then each ring converges to a target ring.
µ c,n = 1 n n i=1 δ x c
i is the distribution of samples belonging to class c. We emphasize that we are the first to represent labeled datasets this way. We first verify on synthetic data and datasets of images that minimizing such distance allows to transport classes between the source and target. Then, we leverage this property on a dataset distillation and a transfer learning task. We focus here on learning target distributions of the form Q = 1 C C k=1 δ ν c,n where ν c,n = 1 n n i=1 δ y c i are empirical distributions, each ν c,n has the same number of particles n and the number of class C is supposed to be known. Similarly as (Hertrich et al., 2024b), we add a momentum to accelerate the scheme for image-based datasets. We refer to Appendix D for more details about the experiments, as well as additional experiments using other kernels and an ablation study for the number of projections to approximate SW. Related works (Alvarez-Melis & Fusi, 2021; Hua et al., 2023) are described in detail in Appendix E.
Synthetic Data. We illustrate on Figure 1 the evolution of particles when minimizing the MMD with kernels K(µ, ν) = e -SW 2 2 (µ,ν)/(2h) and K(µ, ν) = -SW 2 (µ, ν), for a target being the three-ring dataset. Each ring represents a distribution ν c,n with n = 80, and the target is thus a mixture of three Dirac, i.e., Q = 1 3 3 c=1 δ ν c,n . We learn a distribution P of the same form, with the same number of particles for each distribution. We observe for both kernels that the particles of each distribution µ c,n (i.e., the different point clouds) form a ring early in the gradient flow dynamics, and then move in a structured manner towards the target. This illustrates the two level of interactions at the intra and inter distributions levels. In Appendix D.2, we add comparisons with other hyperparameters and other kernels. Overall, the kernel K(µ, ν) = -SW 2 (µ, ν) is the simplest to use, as it does not require tuning a bandwidth, and converges well in general. Thus, in the following experiments, we restrict ourselves to this kernel, and name the resulting loss MMDSW.
this section cite: []

Section: Domain Adaptation.
We now focus on the case where both the source P 0 and the target Q are distributions of im- ages with C classes. Thus, we have P 0 = 1 C C c=1 δ µ c,n and Q = 1 C C c=1 δ ν c,n , and ν c,n , µ c,n represent the empirical distribution of images belonging to the class c. We consider the *NIST datasets, i.e., MNIST (LeCun & Cortes, 2010), Fashion-MNIST (FMNIST) (Wang et al., 2018), KM-NIST (Clanuwat et al., 2018) and USPS (Hull, 1994). These datasets all have C = 10 classes and are of size 28 × 28 (except for USPS which is upscaled to 28 × 28). We also consider CIFAR10 (Krizhevsky et al., 2009) and SVHN (Netzer et al., 2011) which are of size 32 × 32 × 3. We first show in Figure 2 examples of trajectories starting from MNIST to the other *NIST datasets (with step size τ = 0.05, momentum m = 0.9 and n = 200). We see that samples from MNIST are sent to samples of the target dataset, i.e. that the flow converges well. We also observe that images from each class are mapped one-to-one to images within the same class (see Figure 11 in the Appendix), without overlap or collapse across classes.
To verify this quantitatively, we perform a domain adaptation task as in (Alvarez-Melis & Fusi, 2021, Section 7.3). Here, we first train a classifier on 5000 samples of MNIST with n = 500 images by class. Then, we flow the other  datasets to MNIST (with τ = 0.1 and momentum m = 0.9), and measure the accuracy of the pretrained classifier on the flowed dataset. Note that while we use the class labels of the flowed dataset to perform the gradient flow dynamics, we do not know a priori which class in the flowed dataset corresponds to which class in MNIST, yet it is needed for the evaluation of domain adaptation. To perform this alignment, we solve an OT problem with W 2 2 as groundcost between P and Q (i.e., the WoW OT problem) with P the distributions obtained at the end of the flow dynamic and Q the ones of the target dataset. Since these distributions have a finite support of the same size (C), solving this OT problem provides such an alignment: we can associate a prediction of the pretrained model to an image and a "true class" of the flowed dataset. We also perform this experiment with a pretrained neural network on CIFAR10, flowing SVHN toward CIFAR10, with n = 100 samples by class, step size τ = 0.1 and momentum m = 0.9.
On Figure 3, we report the accuracy of the pretrained classifier on the data flowed starting from FMNIST towards MNIST and from SVHN towards CIFAR10, over the iterations (averaged over 3 flows started at different splits of the source data). We also report the value from (Alvarez-Melis & Fusi, 2021) using OTDD on the MNIST dataset. We observe that the classifier converges to 100% accuracy for a sufficient number of iterations. This demonstrates that the flow is able to perfectly match one class from the source dataset with a class of the target dataset, on which the classifier has been trained.
We note that in a realistic setting of unsupervised domain adaptation, we would not have access to the labels of the source dataset (Courty et al., 2016). Thus, to flow the data as we did just earlier, we would need first to find pseudo-labels on the source datasets, e.g. with clustering (Alvarez-Melis & Fusi, 2021;El Hamri et al., 2022). However, this is not the goal of the paper.
this section cite: ['b83', 'b27', 'b44', 'b60', 'b28', 'b3', 'b39']

Section: Dataset Distillation.
Dataset distillation or condensation (Wang et al., 2018) seeks to produce a compact synthetic dataset derived from a large training set, such that training a neural network on the synthetic data yields performance close to that obtained with the full dataset. Zhao & Bilen (2023) proposed to learn the synthetic dataset by performing Distribution Matching, i.e., denoting ν c the distribution of each class c of the target dataset, they minimize
F (µ c ) c = E θ,ω C c=1 MMD 2 k ψ θ # A ω # µ c , ψ θ # A ω # ν c ) ,(22)
with k the linear kernel k(x, y) = ⟨x, y⟩, A ω : R d → R d a random data augmentation (e.g. rotation, cropping, see (Zhao & Bilen, 2021)) and ψ θ : R d → R d ′ with d ′ ≪ d a randomly initialized neural network used to embed the data.
Let Q = 1 C C c=1 δ ν c be the target dataset, ϕ θ,ω (µ) = ψ θ # A ω # µ and P = 1 C C c=1 δ µ c . Note that ϕ θ,ω # P = 1 C C c=1 δ ψ θ # A ω # µ c . We propose to minimize F(P) = E θ,ω MMD 2 K (ϕ θ,ω # P, ϕ θ,ω # Q) ,(23)
with Riesz SW kernel K between distributions. We compare on Table 1 the accuracy of a classifier on a test set of MNIST and FMNIST, trained on the synthetic dataset with p ∈ {1, 10, 50} samples by class, either generated with MMD with Riesz SW kernel (MMDSW) or with Distribution Matching (DM), in 4 scenarios: in the ambient space with (ψ θ = Id) and without augmentation (ψ θ = A ω = Id), and with an embedding with (A ω + ψ θ ) and without an augmentation (A ω = Id). We solve it with a stochastic gradient descent, sampling one augmentation and embedding at each step, for 20K iterations and initializing the samples on true data. The results are averaged over 3 synthetic datasets obtained initializing the flow at different samples, and 5
Table 2: Accuracy of classifier on augmented datasets for k ∈ {1, 10, 10, 100}. M refers to MNIST, F to Fashion MNIST, K to KMNIST and U to USPS.
Dataset k Train on Q MMDSW OTDD (Hua et al., 2023) M to F 1 26.0 ±5.3 40.5 ±4.7 30.5 ±4.2 36.4 ±3.3 5 38.5 ±6.7 61.5 ±4.6 59.7 ±1.8 62.7 ±1.1 10 53.9 ±7.9 65.4 ±1.5 64.0 ±1.4 66.2 ±1.0 100 71.1 ±1.5 74.7 ±0.8 -73.5 ±0.7 M to K 1 18.4 ±3.1 20.9 ±2.0 18.8 ±2.1 19.4 ±1.9 5 25.9 ±4.0 37.4 ±2.2 31.3 ±1.4 39.0 ±1.0 10 30.9 ±4.6 44.7 ±1.8 34.1 ±0.9 44.1 ±1.2 100 60.1 ±1.1 66.8 ±0.8 66.3 ±0.9 62.4 ±1.2 M to U 1 32.4 ±7.9 37.4 ±6.1 39.5 ±7.9 35.0 ±5.6 5 51.4 ±9.8 73.0 ±1.0 73.3 ±1.4 69.6 ±1.3 10 60.3 ±10.1 77.2 ±1.2 72.7 ±2.7 75.6 ±1.2 100 87.5 ±0.7 89.7 ±0.4 -88.1 ±0.6
training of the classifier. On a Nvidia v100 GPU, the flow implemented in Jax (Bradbury et al., 2018) runs in around 10 minutes with the embedding and in 30 seconds without it. The baseline "random" refers to the classifier trained on data sampled randomly from the original training set, and "full data" to the classifier trained on the full training set. We observe on Table 1 that MMDSW consistently outperforms DM when flowing in the ambient space, and is competitive when adding an embedding. This indicates that adding interactions between classes appears to improve the results, possibly by distributing the samples more effectively and mitigating the presence of ambiguous samples near class borders.
Transfer Learning. We now focus on the task of k-shot learning. In this setting, we are interested in training a classifier for datasets which have k samples by class, where k is typically small. Following (Alvarez-Melis & Fusi, 2021;Hua et al., 2023), we propose to augment the dataset by generating new synthetic samples for each class. To do this, we will flow a larger source dataset, with possibly different classes, towards the small target dataset, and then concatenate the synthetic and true samples to train the classifier on it. More precisely, let Q = 1 C C c=1 δ ν c,k the target dataset, with ν c,k = 1 k k i=1 δ y c i an empirical distribution with k samples, representing the distribution of the class c.
Let P 0 = 1 C C c=1 δ µ c,n be a source dataset, with µ c,n = 1 n n i=1 δ x c
i and n = 200. Then, the goal is to flow P 0 towards Q by minimizing F(P) = 1 2 MMD 2 (P, Q) with kernel K(µ, ν) = -SW 2 (µ, ν). We expect to augment each class c of Q with n samples. Then, we train a classifier with a LeNet5 architecture on the dataset obtained as Q = 1 C C c=1 δ η c,n+k with η c,n+k = 1 n+k n+k i=1 δ z c i where z c i = x c i for i ≤ n and z c i = y c i-n for i > n. We report the results for MNIST as P 0 and FMNIST, KMNIST and USPS as Q on Table 2 for k ∈ {1, 5, 10, 100}, compared with the baseline where we train directly on Q, and the baselines where we trained on the synthetic data obtained by minimizing OTDD (Alvarez-Melis & Fusi, 2021) or the MMD with product kernel as in (Hua et al., 2023). The results are averaged over 5 training of the networks, and 3 outputs of the flows. Both methods have been reimplemented and we add more details in Appendix D.6. We observe that all three methods improve upon the baseline, with a slight advantage for MMDSW.
Complexity. Given P = 1 C C c=1 δ µ c,n and Q = 1 C C c=1 δ ν c,n with µ c,n and ν c,n discrete distributions with n samples each, the MMD with a Sliced-Wasserstein based kernel requires to compute C 2 Sliced-Wasserstein distances, which has a total complexity of O C 2 Ln(log n + d) using L projections to approximate SW. We report on Table 3 the runtimes for the transfer learning experiment, averaged over 3 outputs of the flows and trained for 5K epochs for each method. MMDSW is much faster than both OTDD and the MMD with product kernel, at least with our implementations in jax detailed in Appendix D.6. Both OTDD and the method of Hua et al. ( 2023) are implemented using a dimension reduction technique in 2D and a Gaussian approximation to embed the conditional distributions.
this section cite: ['b83', 'b88', 'b19', 'b3']

Section: Conclusion
This work provides the first theoretical framework and practical implementation of gradient flows of a suitable MMD objective over the space of random measures, endowed with the Wasserstein over Wasserstein distance. On the theoretical side, we provided a rigorous differential structure on that space and showed that these flows are well-posed. On the numerical side, our results demonstrate that this novel approach provides meaningful dynamics for interpolating between random measures. There are many possible extensions of our study. For instance, it would be interesting to investigate the minimization of alternative functionals over the space of random measures, e.g., MMD with other kernels (Bachoc et al., 2023;Kachaiev & Recanatesi, 2024), integral probability metrics (Müller, 1997;Catalano & Lavenant, 2024) or f-divergences (Csiszár, 1967). Future work could also address the theoretical treatment of non-compact manifolds or derive a continuity equation for Wasserstein over Wasserstein (WoW) gradient flows. Then, another topic of future research would be to provide quantitative guarantees on the convergence of these schemes.
this section cite: ['b7', 'b59', 'b21', 'b29']

Section: References
Ref_id:b0 Title: Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels Year: (2023)
Ref_id:b1 Title: Averaging on the Bures-Wasserstein manifold: dimensionfree convergence of gradient descent Year: (2021)
Ref_id:b2 Title: Geometric Dataset Distances via Optimal Transport Year: (2020)
Ref_id:b3 Title: Dataset Dynamics via Gradient Flows in Probability Space Year: (2021)
Ref_id:b4 Title: Gradient Flows: in Metric Spaces and in the Space of Probability Measures Year: (2008)
Ref_id:b5 Title: Snowflake universality of Wasserstein spaces Year: (2018)
Ref_id:b6 Title: Maximum Mean Discrepancy Gradient Flow Year: (2019)
Ref_id:b7 Title: Gaussian Processes on Distributions based on Regularized Optimal Transport Year: (2023)
Ref_id:b8 Title: Fast Online Optimization on Transport Polytopes Year: (2023)
Ref_id:b9 Title: The Gene Mover's Distance: Singlecell similarity via Optimal Transport Year: (2021)
Ref_id:b10 Title: Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking Year: (2018)
Ref_id:b11 Title: Sliced-Wasserstein on Symmetric Positive Definite Matrices for M/EEG Signals Year: (2023)
Ref_id:b12 Title: Mirror and Preconditioned Gradient Descent in Wasserstein Space Year: (2024)
Ref_id:b13 Title: Sliced-Wasserstein Distances and Flows on Cartan-Hadamard Manifolds Year: (2025)
Ref_id:b14 Title: Stochastic Gradient Descent on Riemannian Manifolds Year: (2013)
Ref_id:b15 Title: Sliced and Radon Wasserstein Barycenters of Measures Year: (2015)
Ref_id:b16 Title: A Pontryagin Maximum Principle in Wasserstein Spaces for Constrained Optimal Control Problems Year: (2019)
Ref_id:b17 Title: Unidimensional and Evolution Methods for Optimal Transportation Year: (2013)
Ref_id:b18 Title: An Introduction to Optimization on Smooth Manifolds Year: (2023)
Ref_id:b19 Title: JAX: composable transformations of Python+NumPy programs Year: (2018)
Ref_id:b20 Title: Sliced Wasserstein Kernel for Persistence Diagrams Year: (2017)
Ref_id:b21 Title: Hierarchical Integral Probability Metrics: A distance on random probability measures with low sample complexity Year: (2024)
Ref_id:b22 Title: Statistical and Geometrical properties of the Kernel Kullback-Leibler divergence Year: (2024)
Ref_id:b23 Title: Optimal Transport for Mixtures of Radial Functions Year: (2024)
Ref_id:b24 Title: Optimal Transport for Gaussian Mixture Models Year: (2018)
Ref_id:b25 Title:  Year: (2024)
Ref_id:b26 Title: Scalable Wasserstein Gradient Flow for Generative Modeling through Unbalanced Optimal Transport Year: (2024)
Ref_id:b27 Title: Deep Learning for Classical Japanese Literature Year: (2018)
Ref_id:b28 Title: Optimal Transport for Domain Adaptation Year: (2016)
Ref_id:b29 Title: Information-Type Measures of Difference of Probability Distributions and Indirect Observations Year: (1967)
Ref_id:b30 Title: Sinkhorn Distances: Lightspeed Computation of Optimal Transport Year: (2013)
Ref_id:b31 Title: Optimal Transport Tools (OTT): A Jax Toolbox for all things Wasserstein Year: (2022)
Ref_id:b32 Title: Rademacher-type theorem on L2-Wasserstein spaces over closed Riemannian manifolds Year: (2020)
Ref_id:b33 Title: A Wasserstein-type distance in the space of Gaussian Mixture Models Year: (2020)
Ref_id:b34 Title: Forward-Backward Gaussian Variational Inference via JKO in the Bures-Wasserstein Space Year: (2023)
Ref_id:b35 Title: Nonparametric Generative Modeling with Conditional Sliced-Wasserstein Flows Year: (2023)
Ref_id:b36 Title: Wasserstein of Wasserstein Loss for Learning Generative Models Year: (2019)
Ref_id:b37 Title: A Wassersteintype metric for generic mixture models, including location-scatter and group invariant measures Year: (2023)
Ref_id:b38 Title: Computational Optimal Transport: Complexity by Accelerated Gradient Descent is Better than by Sinkhorn's Algorithm Year: (2018)
Ref_id:b39 Title: Hierarchical Optimal Transport for Unsupervised Domain Adaptation Year: (2022)
Ref_id:b40 Title: Optimal transport with optimal transport cost: the Monge-Kantorovich problem on Wasserstein spaces. Calculus of Variations and Partial Differential Equations Year: (2025)
Ref_id:b41 Title: Equinox: neural networks in JAX via callable PyTrees and filtered transformations. Differentiable Programming workshop at Neural Information Processing Systems 2021 Year: (2021)
Ref_id:b42 Title: Sliced Wasserstein Kernels for Probability Distributions Year: (2016)
Ref_id:b43 Title: Small Transformers Compute Universal Metric Embeddings Year: (2023)
Ref_id:b44 Title: Learning Multiple Layers of Features from Tiny Images Year: (2009)
Ref_id:b45 Title: From Word Embeddings to Document Distances Year: (2015)
Ref_id:b46 Title: Variational inference via Wasserstein gradient flows Year: (2022)
Ref_id:b47 Title: First-Order Conditions for Optimization in the Wasserstein Space Year: (2025)
Ref_id:b48 Title: MNIST handwritten digit database Year: (2010)
Ref_id:b49 Title: Riemannian Manifolds: An Introduction to Curvature Year: (2006)
Ref_id:b50 Title: Particle Semi-Implicit Variational Inference Year: (2024)
Ref_id:b51 Title: Wasserstein Task Embedding for Measuring Task Similarities Year: (2025)
Ref_id:b52 Title: Sliced-Wasserstein Flows: Nonparametric Generative Modeling via Optimal Transport and Diffusions Year: (2019)
Ref_id:b53 Title: Decoupled Weight Decay Regularization Year: (2019)
Ref_id:b54 Title: Accelerating Langevin Sampling with Birth-Death Year: (2019)
Ref_id:b55 Title: Birth-death dynamics for sampling: global convergence, approximations and their asymptotics Year: (2023)
Ref_id:b56 Title: MMD-Regularized Unbalanced Optimal Transport Year: (2024)
Ref_id:b57 Title: Polar factorization of maps on Riemannian manifolds Year: (2001)
Ref_id:b58 Title: Distribution Regression with Sliced Wasserstein Kernels Year: (2022)
Ref_id:b59 Title: Integral Probability Metrics and Their Generating Classes of Functions Year: (1997)
Ref_id:b60 Title: Reading Digits in Natural Images with Unsupervised Feature Learning Year: (2011)
Ref_id:b61 Title: Wasserstein Gradient Flows for Moreau Envelopes of f-Divergences in Year: (2024)
Ref_id:b62 Title: Hierarchical Hybrid Sliced Wasserstein: A Scalable Metric for Heterogeneous Joint Distributions Year: (2024)
Ref_id:b63 Title: Lightspeed Geometric Dataset Distance via Sliced Optimal Transport Year: (2025)
Ref_id:b64 Title: Borrowing strengh in hierarchical Bayes: Posterior concentration of the Dirichlet base measure Year: (2016)
Ref_id:b65 Title: The Geometry of Dissipative Evolution Equations: The Porous Medium Equation Year: (2001)
Ref_id:b66 Title: Scikit-learn: Machine Learning in Python Year: (2011)
Ref_id:b67 Title: SEACells infers transcriptional and epigenomic cellular states from single-cell genomics data Year: (2023)
Ref_id:b68 Title: GOT: an Optimal Transport framework for Graph comparison Year: (2019)
Ref_id:b69 Title: PointNet: Deep learning on Point Sets for 3D Classification and Segmentation Year: (2017)
Ref_id:b70 Title: Wasserstein Barycenter and its Application to Texture Mixing Year: (2011-06-02)
Ref_id:b71 Title: ELBOing Stein: Variational Bayes with Stein Mixture Inference Year: (2025)
Ref_id:b72 Title: McGraw-Hill Science/Engineering/Math Year: (1986)
Ref_id:b73 Title: The Wasserstein Proximal Gradient Algorithm Year: (2020)
Ref_id:b74 Title: Optimal Transport for Applied Mathematicians Year: (2015)
Ref_id:b75 Title: Gradient Flows: an overview Year: (2017)
Ref_id:b76 Title: Unbalanced Optimal Transport, from Theory to Numerics. Handbook of Numerical Analysis Year: (2023)
Ref_id:b77 Title: Diffusion on Multidimensional Spaces Year: (2024)
Ref_id:b78 Title: Topics in Optimal Transportation Year: (2003)
Ref_id:b79 Title: Optimal Transport: Old and New Year: (2009)
Ref_id:b80 Title: Word Representations via Gaussian Embedding Year: (2015)
Ref_id:b81 Title: Entropic Measure and Wasserstein Diffusion Year: (2009)
Ref_id:b82 Title: Knowledge Graph Embedding with Dirichlet Distribution Year: (2022)
Ref_id:b83 Title:  Year: (2018)
Ref_id:b84 Title: Sampling as optimization in the space of measures: The Langevin dynamics as a composite optimization problem Year: (2018)
Ref_id:b85 Title: A Wasserstein-Type Distance for Gaussian Mixtures on Vector Bundles with Applications to Shape Analysis Year: (2024)
Ref_id:b86 Title: Learning Gaussian Mixtures Using the Wasserstein-Fisher-Rao Gradient Flow Year: (2024)
Ref_id:b87 Title: Hierarchical Optimal Transport for Document Representation Year: (2019)
Ref_id:b88 Title: Dataset Condensation with Differentiable Siamese Augmentation Year: (2021)
