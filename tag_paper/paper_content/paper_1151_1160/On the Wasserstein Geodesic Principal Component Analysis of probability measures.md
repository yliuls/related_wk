Title: ON THE WASSERSTEIN GEODESIC PRINCIPAL COMPO-NENT ANALYSIS OF PROBABILITY MEASURES
Abstract: This paper focuses on Geodesic Principal Component Analysis (GPCA) on a collection of probability distributions using the Otto-Wasserstein geometry. The goal is to identify geodesic curves in the space of probability measures that best capture the modes of variation of the underlying dataset. We first address the case of a collection of Gaussian distributions, and show how to lift the computations to the space of invertible linear maps. For the more general setting of absolutely continuous probability measures, we leverage a novel approach to parameterizing geodesics in Wasserstein space with neural networks. Finally, we compare to classical tangent PCA through various examples and provide illustrations on realworld datasets.

Section: INTRODUCTION
In this paper, we are interested in computing the main modes of variation of a dataset of absolutely continuous (a.c.) probability measures supported in R d . For data points living in an arbitrary Hilbert space, the classical approach defined by Principal Component Analysis (PCA) consists in finding a sequence of nested affine subspaces on which the projected data retain a maximal part of the variance of the original dataset, or equivalently, yield best lower-dimensional approximations. When dealing with a set of a.c. probability distributions, a natural choice is to identify the probability measures with their probability density functions and to perform PCA on these using the L 2 Hilbert metric. Unfortunately, as highlighted in Cazelles et al. (2018), the components computed in this manner fail to capture the intrinsic structure of the dataset: the projections onto the components most likely result in non-positive and un-normalized functions. Using the Wasserstein metric W 2 instead has proven to overcome these limitations by taking into account the geometry of the space of distributions.
The Wasserstein metric endows the space of probability distributions with a Riemannian-like structure, framing the problem as PCA on a (positively) curved Riemannian manifold. A first approach to solve this task, known as Tangent PCA (TPCA), consists in embedding the data into the tangent space at a reference point, and applying classical PCA in this flat space, as in Fletcher et al. (2003). TPCA is computationally advantageous but can generically induce distortion in the embedded data, depending on the curvature of the manifold at the reference point and the dispersion of the data. A more geometrically coherent approach is Geodesic PCA (GPCA) proposed for Riemannian manifolds in Huckemann et al. (2010); Huckemann & Ziezold (2006), where principal modes of variations are geodesics that minimize the variance of the projection residuals. Following this approach, the first geodesic component of a set of probability measures ν 1 , . . . , ν n in the Wasserstein space solves
inf t →µ(t) geodesic n i=1 inf ti W 2 2 (µ(t i ), ν i ).(1)
The Riemannian geometry of the Bures-Wasserstein metric in equation 3 can be described by considering S ++ d as the quotient of the manifold GL d of invertible matrices by the right action of the orthogonal group O d . In this geometry, GL d is decomposed into equivalence classes called fibers. The fiber over Σ ∈ S ++ d is defined to be the pre-image of Σ under the projection
π : A ∈ GL d → AA ⊤ ∈ S ++ d ,(4)
and can be obtained as the result of the action of O d on a representative, e.g. Σ 1/2 the only SPD square root of Σ: π -1 (Σ) = {A ∈ GL d , AA ⊤ = Σ} = Σ 1/2 O d .
Tangent vectors to GL d are said to be horizontal if they are orthogonal to the fibers with respect to the Frobenius metric, i.e. if they belong to the space
Hor A : = {X ∈ R d×d , X ⊤ A -A ⊤ X = 0}, (5
)
for a given point A ∈ GL d . Then the projection π in equation 4 defines an isometry between the horizontal subspace Hor A equipped with the Frobenius inner product ⟨X, Y ⟩ : = tr(XY ⊤ ), and S ++ d equipped with a Riemannian metric that induces the Bures-Wasserstein distance (equation 3) as the geodesic distance. In particular, this means that moving horizontally along straight lines in the top space GL d is equivalent to moving along geodesics in the bottom space S ++ d (see Figure 1), as recalled in the following proposition.
S + + d Id A 1 Σ 2 Id π G L d X ∈ Hor A 1 Σ 1 A 2 π -1 (I d ) π -1 (Σ 1) π -1 (Σ 2)
ho riz on ta l ge od es ic W as ser ste in ge od esi c
Figure 1: The Bures-Wasserstein geometry of centered Gaussian distributions, inspired by Khesin et al. (2021).
Proposition 1 (Takatsu (2011); Malagò et al. (2018); Bhatia et al. (2019)). Any geodesic t → Σ(t) in S ++ d for the Bures-Wasserstein metric in equation 3 is the π-projection of a horizontal line segment in GL d , that is
Σ(t) = π(A + tX) = (A + tX)(A + tX) ⊤ , A ∈ GL d , X ∈ Hor A ,(6)
where t is defined in a certain time interval (t min , t max ). Also, the Bures-Wasserstein distance between two covariance matrices Σ 1 , Σ 2 ∈ S ++ d is given by the minimal Euclidean distance between their fibers
BW 2 (Σ 1 , Σ 2 ) = inf Q1,Q2∈O d ∥Σ 1/2 1 Q 1 -Σ 1/2 2 Q 2 ∥ = inf Q∈SO d ∥Σ 1/2 1 -Σ 1/2 2 Q∥,(7)
where ∥ • ∥ is the Frobenius norm and SO d is the special orthogonal group.
It is essential to note that the geodesic equation 6 cannot be extended for all time t ∈ R (the only geodesic lines are those obtained by translation (Kloeckner, 2010, Proposition 3.6)). Therefore, equation 6 is only defined on a time interval (t min , t max ) that depends on the eigenvalues of XA -1 (see Appendix B.3). More details on the Bures-Wasserstein geometry can be found in Appendix B.2.
Otto-Wasserstein geometry of a.c. probability measures The Riemannian structure described for Gaussian distributions is a special case of Otto (2001)'s more general construction : the bottom space becomes the space Prob(Ω) of a.c. distributions supported on a compact set Ω ⊂ R d while the top space is the space of diffeomorphisms Diff(Ω) endowed with the L 2 metric with respect to a fixed reference measure ρ (see Figure 19 in Appendix B). The fibers of Diff(Ω) are then defined to be the pre-images under the projection π : φ ∈ Diff(Ω) → π(φ) = φ # ρ ∈ Prob(Ω).
In this setting, horizontal displacements in Diff(Ω) are along vector fields that are gradients of functions. The projection π defines an isometry between the horizontal subspace equipped with the L 2 (ρ)-inner product and Prob(Ω) equipped with a Riemannian metric that induces the Wasserstein distance as the geodesic distance. In particular, we have the following result. Proposition 2 (Otto (2001)). Any geodesic t → µ(t) for the Wasserstein metric given in equation 2 is the π-projection of a line segment in Diff(Ω) going through a diffeomorphism φ at horizontal speed ∇f • φ for some smooth function f ∈ C(R d ). That is, for t defined in a certain interval (t min , t max ),
µ(t) = π(φ + t∇f • φ) = (id +t∇f ) # (φ # ρ).(9)
Another geodesic μ(t) = π(φ + t∇ f • φ) is orthogonal to µ(t) at t = 0 for the Riemannian metric inducing the Wasserstein distance if and only if ⟨∇f • φ, ∇ f • φ⟩ L 2 (ρ) = 0.
We emphasize that f need not be convex in equation 9, unlike in the more classical parametrization of geodesics due to McCann (1997) between two distributions µ 0 and µ 1 = ∇u # µ 0 : µ(t) = (id +t(∇u -id)) # µ 0 , with t ∈ [0, 1] and u a convex function.
(10) Note that equation 9 parametrizes geodesics provided that id +t∇f is a diffeomorphism, and thus it is defined on a time interval that depends on the eigenvalues of the Hessian of f . On the other hand, the convexity condition on the function u in the parametrization of equation 10 ensures that time t is defined on [0, 1]. Both are completely equivalent (see Appendix B.3 for details).
this section cite: ['b9', 'b13', 'b19', 'b18', 'b20', 'b40', 'b26', 'b3', 'b21', 'b31', 'b31', 'b27']

Section: GEODESIC PCA ON CENTERED GAUSSIAN DISTRIBUTIONS
In this section, we consider the exact GPCA problem for the Bures-Wasserstein metric in equation 3. The data are n centered Gaussian distributions identified with their covariance matrices Σ 1 , . . . , Σ n ∈ S ++ d . Following Huckemann et al. (2010), we define the first component as the geodesic t → Σ(t) ∈ S ++ d that minimizes the sum of squared residuals of the BW 2 -projections of the data:
inf t →Σ(t) geodesic n i=1 inf ti BW 2 2 (Σ(t i ), Σ i ).(11)
The second principal component is defined to be the geodesic that minimizes the same cost function, with the constraint of intersecting the previous component orthogonally. The subsequent principal components have the additional constraint of going through the intersection of the first two principal geodesics. This definition does not impose that the geodesic components go through the Wasserstein barycenter (see Agueh & Carlier (2011)), and in Section 5 we show an example where this is indeed not verified. This gives an observation of the phenomenon already described in Huckemann & Ziezold (2006) for spherical geometry. The proofs of this section are deferred to Appendix D.
this section cite: ['b19', 'b0', 'b18']

Section: Learning the geodesic components
Following Propositon 1, we lift the GPCA problem in equation 11 to the total space GL d of Otto's fiber bundle. This has several advantages: the Bures-Wasserstein distance in the cost function of equation 11 is replaced by the Frobenius norm ∥ • ∥, the geodesic is replaced by a horizontal line segment, and the projection times t i become explicit. The price to pay is an optimization over variables (Q i ) n i=1 in SO d , needed to represent the covariance matrices Σ i by invertible matrices Σ
1/2 i Q i in their respective fibers. Proposition 3. Let π : GL d → S ++ d , A → AA ⊤ and (A 1 , X 1 , (Q i ) n i=1 ) be a solution of inf F (A 1 , X 1 , (Q i ) n i=1 ) : = n i=1 ∥A 1 + p A1,X1 (t i )X 1 -Σ 1/2 i Q i ∥ 2 , subject to A 1 ∈ GL d , X 1 ∈ Hor A1 , ∥X 1 ∥ 2 = 1, Q 1 , . . . , Q n ∈ SO d .(12)
Then there exist t min , t max ∈ R such that the geodesic Σ : t ∈ [t min , t max ] → π(A 1 + tX 1 ) in S ++ d minimizes equation 11.
Here the t i are projection times given by
t i = ⟨Σ 1/2 i Q i -A 1 , X 1 ⟩, and p A,X
is a projection operator that clips any t ∈ R onto a closed interval [t min , t max ] depending on A and X, such that A+p A,X (t)X is invertible for any t in this interval (see Appendix B.3). Clipping the time parameter of the line segment is necessary to ensure it remains within GL d and projects onto a geodesic in S ++ d . The second component is thus defined by Σ 2 (t) = π(A 2 + tX 2 ), found by solving:
inf F (A 2 , X 2 , (Q i ) n i=1 ) subject to A 2 = (A 1 + t * X 1 )R * , R * ∈ SO d , t * ∈ [t min , t max ] X 2 ∈ Hor A2 , ∥X 2 ∥ 2 = 1, ⟨X 2 , X 1 R * ⟩ = 0, Q 1 , . . . , Q n ∈ SO d . (13)
Note that this step requires to find new rotation matrices (Q i ) n i=1 . The first two components fix the intersection point π(A 2 ) through which all other geodesic components will pass, see Figure 2. For every higher order component, we search for a velocity vector X k that is horizontal at some point in the fiber over π(A 2 ) and orthogonal to the lifts of the velocity vectors of the previous components. Details on the implementation of these components are given in Appendix D.2.
The second component is a geodesic of S ++ d that orthogonally intersects the first component. Lifting again the problem to GL d , this boils down to searching for a horizontal line t → A 2 + tX 2 where A 2 = (A 1 + t * X 1 )R * for a rotation matrix R * , a time t * ∈ [t min , t max ] and a horizontal vector X 2 ∈ Hor A2 such that ⟨X 2 , X 1 R * ⟩ = 0. The equation for A 2 ensures that the π-projections of the first two horizontal lines intersect, while the condition on X 2 ensures that they intersect orthogonally (since X 1 R * is horizontal at A 2 as can easily be checked). See Figure 2.
GL d S ++ d π A 1 X 1 A 1 + t * X 1 X 2 X 1 R * A 2 π(A 1 ) π(A 2 ) dπ A 2 (X 2 ) dπ A 2 (X 1 R * )
Figure 2: First (red) and second (blue) geodesic components of Gaussian GPCA, where dπ A denotes the differential of the projection π : A → AA ⊤ at A ∈ GL d .
Quantifying the difference between TPCA and GPCA In the following, we quantify the distortion induced by linearization in the case of covariances matrices with same eigenvalues. Proposition 4. Let Σ ∈ S ++ 2 with eigenvalues a 2 , b 2 and Σ ′ = P θ ΣP ⊤ θ where P θ is the rotation matrix of angle θ. Then, denoting Σ = ((a + b)/2)
2 I, we have
BW 2 2 (Σ, Σ ′ ) BW 2 2, Σ(Σ, Σ ′ ) = 1 - a -b a + b 2 cos 2 θ + O((a -b) 4 ),(14)
where BW 2, Σ is the linearized Bures-Wasserstein distance at Σ recalled in equation 29.
For a given θ, equation 14 shows that the distorsion is most important for |a-b| |a+b| close to 1, which corresponds to matrices that are close to the border of the cone, as illustrated in Section 5.1.
On the restriction to the space of Gaussian distributions Geodesic PCA can also be defined in the more general space of a.c. probability distributions, as presented in Section 4. A natural question that arises is whether performing GPCA in the whole space of probability distributions gives the same result as restricting to the space of Gaussian distributions, which is totally geodesic. The answer is yes in dimension one, as shown in Appendix D. Proposition 5. Let ν i = N (m i , σ 2 i ) for i = 1, . . . , n, be n univariate Gaussian distributions. The first principal geodesic component t ∈ [0, 1] → µ(t) solving equation 1 remains in the space of Gaussian distributions for all t ∈ [0, 1].
Up to our knowledge, this remains an open question in higher dimension.
this section cite: []

Section: GEODESIC PCA ON A.C. PROBABILITY MEASURES: GPCAGEN
We now tackle the task of performing GPCA on a set of a.c. probability measures ν 1 , . . . , ν n using the Otto-Wasserstein geometry. We propose a parameterization of the geodesic principal components based on Otto's formulation, leveraging neural networks. Additionally, we introduce a dedicated cost function to optimize the different geodesic components.
Parameterizing geodesics Following Proposition 2 and equation 9, any geodesic t → µ(t) in the Wasserstein space (Prob(Ω), W 2 ) can be expressed as µ(t) = (φ + t∇f • φ) # ρ, for t in some interval [t min , t max ], φ : R d → R d a diffeomorphism, f : R d → R a smooth function, and ρ a fixed reference measure, taken to be the standard Gaussian distribution in this work. Using multilayer perceptrons (MLPs) to parametrize the functions φ and f , denoted φ θ and f ψ , respectively, the curve
t → µ θ,ψ (t) = (id + t∇f ψ ) # (φ θ # ρ)
is a geodesic for t ∈ [t min , t max ], provided that id + t∇f ψ ∈ Diff(Ω) for all t in this interval. Equivalently, this condition holds if the Hessian matrix I d + tH f ψ (x) is positive definite for all x ∈ R d and t ∈ [t min , t max ], where H f ψ (x) denotes the Hessian of f ψ at x. In practice, we enforce this constraint by monitoring the eigenvalues of I d +tH f ψ (x) (see Appendix B.3) and either clipping t or adjusting the interval [t min , t max ] to ensure that all eigenvalues remain positive. This representation enables to sample from the distributions along the geodesic. Specifically, given the learned vector field φ θ and function f ψ , one can sample from µ θ,ψ (t) by first drawing x ∼ ρ and then applying the transformations φ θ and id + t∇f ψ sequentially as φ θ (x) + t∇f ψ (φ θ (x)) ∼ µ θ,ψ (t).
Learning the geodesic components The first principal component in GPCA minimizes the objective in equation 1. The scalar variables t i specify the projection time of each distribution ν i onto the geodesic t → µ(t). Leveraging the explicit form of Otto's geodesic, equation 1 can be reformulated as:
inf f ∈C(R d ),φ∈Diff(Ω) t1,...,tn∈[tmin,tmax] L(f, φ, t 1 , . . . , t n ) : = n i=1 W 2 2 ((id +t i ∇f ) # (φ # ρ), ν i ).(15)
We jointly learn the parameters t i together with the neural networks φ θ and f ψ to minimize the objective in equation 15. In practice, we use the Sinkhorn divergence S ε that has been proven to be a differentiable and computationally efficient approximation of the squared Wasserstein distance W 2 2 , see Frogner et al. (2015); Genevay et al. (2018); Chizat et al. (2020), and represent the distributions ρ and ν i using batches of m samples x k ∼ ρ and y j ∼ ν i . The optimization proceeds by updating the parameters based on a single distribution ν i sampled at each iteration, as detailed in Algorithm 1. To compute t min and t max on line 5 of Algorithm 1, we approximate the extremal eigenvalues of H f ψ by evaluating the largest and smallest eigenvalues over the finite set {H f ψ (x k )} m k=1 , and substitute these estimates into the theoretical bounds from Appendix B.
3. Algorithm 1 Geodesic PCA algorithm for a.c. measures: GPCAGEN 1: Initialize φ θ , f ψ and the t i for 1 ≤ i ≤ n 2: while not converged do 3: for i = 1 to n do 4: Draw m i.i.d samples y (i) j ∼ ν i and draw m i.i.d samples x k ∼ ρ 1 ≤ j, k ≤ m 5: Estimate t min , t max with {H f ψ (x k )} m k=1 and set t ′ i = min(max(t i , t min ), t max ) 6:
z (i) k ← (id +t ′ i ∇f ψ ) • (φ θ )(x k ) for 1 ≤ k ≤ m 7: L θ,ψ,ti ← S ε 1 m m k=1 δ z (i) k , 1 m m j=1 δ y (i) j 8:
Update φ θ , f ψ and the t i with ∇L θ,ψ,ti 9:
end for 10: end while
The second principal component minimizes the objective in equation 1 subject to the constraint that it intersects the first component orthogonally. Similar to the first component, we use two MLPs, f ψ2 and φ θ2 , to parameterize the geodesic t → µ θ2,ψ2 (t), along with n scalar variables t 2 i , to optimize the objective in equation 15. We also introduce two additional scalar variables, t 1 inter and t 2 inter , which define the intersection times of the two geodesics, along with the regularization terms:
I(ξ 1 , ξ 2 , t 1 inter , t 2 inter ) = ∥ξ 1 (t 1 inter ) -ξ 2 (t 2 inter )∥ 2 2 and O(g, h) = ⟨g, h⟩ 2 L 2 (ρ) ∥g∥ 2 L 2 (ρ) ∥h∥ 2 L 2 (ρ)
, where I enforces the two geodesics in Diff(Ω), ξ 1 (t) = (id +t∇f ψ ) • φ θ and ξ 2 (t) = (id +t∇f ψ2 ) • φ θ2 , to intersect at the respective times t 1 inter and t 2 inter while O(g, h) ensures orthogonality between the corresponding horizontal vector fields g = ∇f ψ (φ θ ) and h = ∇f ψ2 (φ θ2 ) in L 2 (ρ). The total objective used to optimize the second principal component incorporates these regularization terms and is given by:
L(f ψ2 , φ θ2 , t 2 1 , . . . , t 2 n ) + λ I I(ξ θ,ψ , ξ θ2,ψ2 , t 1 inter , t 2 inter ) + λ O O(∇f ψ (φ θ ), ∇f ψ2 (φ θ2
)) with ξ θ,ψ (t) = (id +t∇f ψ ) • φ θ and ξ θ2,ψ2 (t) = (id +t∇f ψ2 ) • φ θ2 and where λ I and λ O are the regularization parameters controlling the trade-off between the intersection and orthogonality regularization terms, respectively. Note that in virtue of Proposition 2, the L 2 (ρ) inner product in the regularization term O truly enforces orthogonality of the geodesic components with respect to the Riemannian metric associated to the Wasserstein distance. Note also that I enforces the geodesics to intersect in Diff(Ω) which means that, at the intersection time, the geodesics µ θ1,ψ1 and µ θ2,ψ2 in Prob(Ω) intersect and share the same representative. An alternative implementation would be to enforce the intersection of the geodesics µ θ1,ψ1 and µ θ2,ψ2 in Prob(Ω) and to impose the orthogonality of ∇f ψ (φ θ ) • R * and ∇f ψ2 (φ θ2 ) in L 2 (ρ), where R * = ξ θ2,ψ2 (t 2 inter ) • ξ θ1,ψ1 (t 1 inter ) -1 . This approach is the one used in the Gaussian case. However, computing R * is computationally expensive, and we therefore preferred to impose ξ θ1,ψ1 (t 1 inter ) = ξ θ2,ψ2 (t 2 inter ) which directly yields R * = id.
The training algorithm used to optimize the second principal component follows the same structure as Algorithm 1, except for the seventh line, where the regularization terms, estimated using the minibatch x k ∼ ρ, are added to the loss function. Higher-order components can be computed similarly.
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTS ON CENTERED GAUSSIAN DISTRIBUTIONS
In this section, we consider toy examples in S ++ 2 and compare GPCA to its widely used linearized approximation, TPCA (see Appendix C). We use two coordinate systems for matrices in S ++ 2 : the first comes from the spectral decomposition, and the second maps any SPD matrix to a point in the interior of the cone C = {(x, y, z) ∈ R 3 , z > 0, z 2 > x 2 + y 2 }:
Σ = P θ a 2 0 0 b 2 P ⊤ θ = z + y x x z -y , (a, b, θ) ∈ R * + × R * + × R, (x, y, z) ∈ C, (16
)
where P θ is the rotation matrix of angle θ. Generically, GPCA and TPCA yield very similar results: for sets of n = 50 covariance matrices randomly generated using a uniform distribution on the parameters (a, b, θ), GPCA reduces the objective in equation 11 of less than 1% w.r.t. TPCA, on average for 100 trials. This suggests that TPCA is generally a very good approximation of GPCA. Two extreme cases are described below: (i) GPCA and TPCA are equivalent and (ii) GPCA and TPCA drastically differ.
Matrices with same orientation If we consider a set of covariance matrices that live in the subspace θ = constant in notations of equation 16, then both GPCA and TPCA yield exactly the same results, namely that of linear PCA in the (a, b)-coordinates. This is because any such subspace has zero curvature for the Wasserstein metric, and geodesics are straight lines in the (a, b)-coordinates (Appendix D.1).
Figure 3 shows the geodesic components obtained for a set of matrices in the subspace θ = 0 that form a regular rectangular grid in the (a, b)
coordinates, i.e. Σ ij = diag(a 2 i , b 2 j )
where the a i 's and b j 's are equally spaced. They are indeed straight lines that capture the variations in a and b respectively.
1 0 1 1 0 1 0 1 2 Figure 3: GPCA on a set of diagonal covariance matrices Σ ij with varying eigenvalues 1 ≤ a 2 i ≤ 3, 1 ≤ b 2 j ≤ 2.
The matrices form a planar grid inside the cone C of SPD matrices in equation 16 (left), and correspond to ellipses of varying width and height (right). The first component (red) captures the variation in a, while the second component (blue) captures the variation in b.
Matrices with same eigenvalues Now we consider covariance matrices that all have the same eigenvalues but different orientations. Specifically, we choose Σ i = P θi diag(a 2 , b 2 )P ⊤ θi , for positive reals a > b, θ i = iπ/n for i = 0, . . . , n -1 and an even number n. In the (x, y, z) coordinates (equation 16), the covariance matrices are displayed on a circle of equation z = cst (constant trace) and x 2 + y 2 = cst (constant determinant), as shown in Figure not the pairwise distances between the Σ i 's. Proposition 4 evaluates the level of this distorsion. Note that since (a -b) 2 /(a + b) 2 = (x 2 + y 2 )/z 2 , the distorsion is most important when covariance matrices are close to the border of the cone, see Figure 4 (left). Indeed, in that case, the results of GPCA can be very different from those of TPCA and the first component may not even go through the Wasserstein barycenter Σ, see Figure 4 (middle) and Figure 8 in Appendix A.
In that case GPCA may be seen as worse-behaved as TPCA, as some of the Gaussian distributions will project onto the first geodesic component boundaries, yielding a poor separation. Figure 4 (right) shows the percentage of improvement of the cost in equation 11 (in terms of minimization) of GPCA with respect to TPCA, in the setting previously described for different values of the ratio |a -b|/|a + b|. on average for 10 runs per value of the ratio. The blue strip indicates standard deviation.
this section cite: []

Section: Weather dataset
In this paragraph, we use the Weather CORGIS Dataset to illustrate GPCA based on empirical covariance matrices. The dataset provides weekly measures of precipitation and wind speed recorded from March to January 2016 across the 50 U.S. states and the territory of Puerto Rico. From these measures, we construct two histograms for each state: one for precipitation and one for wind speed. We then compute the 50 empirical covariances from these histograms. We show in Figure 14 the projection of each state onto the two first GPCA components computed from the empirical covariance matrices. We can clearly identify clusters of different weather behavior among the states.
this section cite: []

Section: EXPERIMENTS ON ABSOLUTELY CONTINUOUS DISTRIBUTIONS
We conduct a preliminary experiment on a synthetic dataset with known geodesics to verify that our algorithm, GPCAGEN (Section 4), accurately recovers the first two principal components. We then apply GPCAGEN to 3D point clouds from the ModelNet40 dataset (Wu et al. (2015)) and to color distributions of images from the Landscape Pictures dataset (Rougetet (2020)). An additional experiment in AppendixA.3 demonstrates how GPCA can be used for outlier detection. For these experiments, f ψ and φ θ are MLPs with four hidden layers of size 128 and an output layer of size 1
Figure 5: Densities of probability distributions uniformly sampled along the first and second principal geodesics components. GPCAGEN successfully recovers the two orthogonally intersecting geodesics constructed from MNIST data. The first component (left) captures variation in color space, while the second component (right) recovers the interpolation from the digit "1" to the digit "2". and d respectively. We found that setting the regularization coefficients λ I and λ O to 1.0 ensures the algorithm works as expected in all experiments. A discussion of the regularization coefficients, along with details on the architecture and hyperparameters, is provided in Appendix E.
this section cite: ['b44', 'b35']

Section: MNIST geodesics.
We represent each image from the MNIST dataset (LeCun et al. ( 2010)) as a probability measure over R 4 . The grayscale pixel intensities define a normalized density over spatial coordinates (x, y) ∈ R 2 , and we further assign each pixel two additional values corresponding to red and blue color channels. We construct two orthogonal geodesics: the first one interpolates between a digit "1" and a digit "2", both assigned a fixed purple by setting the color channels to 0.5. The second one is defined from the midpoint of the first, by linearly interpolating the color from red to blue. As shown in Figures 5 and 9, GPCAGEN successfully recovers the two geodesics intersecting orthogonally. A second experiment on the MNIST dataset is displayed in Appendix A.
this section cite: []

Section: 3D point cloud.
We use the ModelNet40 3D point cloud dataset (Wu et al. (2015)) and apply GPCA to a subset of 100 randomly selected lamp point clouds. Figure 6 (middle row) and Figure 7 (left) demonstrate that the first principal component captures the distinction between hanging lamps (chandeliers) and standing lamps (floor lamps), while the second component reflects variations in the thickness of the lamp structure. We conduct a similar experiment on 100 point clouds from ModelNet40 representing different chairs. As shown in Figure 6 (top row) and Figure 10, the first principal component distinguishes between chairs and armchairs, while the second component captures the height of the seat.
Landscape images. We took 39 images from the Landscape Pictures dataset (Rougetet (2020)) and use GPCAGEN on the corresponding point clouds, where each point cloud represents color distribution in the image. Figure 6 (bottom row) and Figure 7 (right) show that the first component captures variations in overall brightness, ranging from dark to bright images, while the second component separates mostly blue images from mostly green ones.
Baselines An obvious baseline for GPCAGEN is TPCA. Unlike GPCAGEN, which learns continuous geodesics from empirical distributions of absolutely continuous measures, TPCA acts on discrete measures. A direct numerical comparison between the two methods is therefore not meaningful. However, we include in Appendix A.2 the two principal components returned by TPCA on the 3D point cloud experiments. We observe in Figure 16 that the discrete nature of TPCA produces artifacts, including holes in certain regions and excessive mass concentration in others.
Another natural baseline consists in embedding point clouds into a latent space of dimension d then performing standard PCA on the resulting latent vectors. This approach, in addition to being computationally expensive, does not produce meaningful modes of variation, as shown in Section A.2 of the appendices.
this section cite: ['b44']

Section: A ADDITIONAL EXPERIMENTS AND FIGURES
A.1 GEODESIC PCA Here we present additional figures to further explain the experiments described in the paper. Figure 8 concerns the experiment on Gaussian distributions with diagonal covariances described in Section 5.1 corresponding to Figure 4. It shows all three principal components found by tangent PCA (left) and geodesic PCA, in two equally optimal solutions (middle, right). Figure 9 displays on the plane the two first geodesic components of the MNIST experiment of Section 5.2, while Figure 10 shows the planar representation of the 3D point cloud of chairs experiment given by the projection onto the first two geodesic components found by GPCAGEN algorithm and depicted in Figure 6 (top row).
Figure 9: Each point cloud, corresponding to a distribution along one of the artificially constructed geodesics, is embedded in the plane according to its projection times onto the first and second geodesics returned by the GPCAGEN algorithm. We observe that GPCAGEN successfully recovers the two orthogonally intersecting geodesics designed from MNIST-based interpolations of digit shape and color. Finally, we present an additional experiment on the MNIST dataset. We use the same color construction as in the experiment presented in Section 5.2, we then apply GPCAGEN to a dataset of 20 red digits "1", 20 blue digits "1", 20 red digits "2", and 20 blue digits "2" (see Figure 12). As shown in Figures 11 and 12, GPCAGEN again identifies two orthogonal geodesics: the first primarily captures variation in color, while the second captures variation in shape-from digit "2" to digit "1".
Figure 11: Densities of probability distributions uniformly sampled along the geodesics corresponding to the first and second principal components. The first component (left) returned by GPCAGEN captures variation in color space, while the second component (right) recovers the interpolation between digit "2" and digit "1".
Figure 12: Each MNIST digit is embedded in the plane (the arrows indicate the exact position of each digit) according to its projection times onto the first and second geodesics returned by the GPCAGEN algorithm. We observe that the first principal component recovered by GPCAGEN captures variation in color, while the second component reflects the transformation from digit "2" to digit "1".
this section cite: []

Section: A.2 COMPARISON OF GPCA TO RELATED METHODS

this section cite: []

Section: Other notions of PCA on Gaussian distributions
There exist a wide variety of metrics on the space of symmetric positive definite matrices, such as e.g. the log-Euclidean, Euclidean-Cholesky or affine-invariant metrics (see Thanwerdas (2022) for a comprehensive overview). Each of these metrics could be used to perform PCA on centered Gaussian distributions. However, there is no obvious quantitative way to compare the results. Each method optimizes its own criterion, and any metric that one could think of to compare the methods would rely on a choice of underlying metric on the space of SPD matrices. Comparison of PCA methods with two different metrics thus boils down to comparing the metrics themselves. We illustrate in Figure 13 the behavior of covariances matrices along geodesics for different metrics.   For the lamps dataset, the first component is similar and captures the distinction between hanging and standing lamps. The second component focuses on the object thickness, like the second GPCAGEN component, but also on whether mass is concentrated at the extremities or the middle of the lamp structure.
For the chairs dataset, both geodesics obtained by TPCA resemble those returned by GPCA. However, the second TPCA component also appears to account for whether the mass is concentrated or not.
Finally, due to the discrete nature of the TPCA algorithm, we observe discretization artifacts in the TPCA components: holes in some parts of the space, mass concentration in others.  PCA computed in the latent space of PointNet. For the 3D point-cloud datasets, we evaluated the natural baseline that consists in embedding point clouds into a latent space of dimension d and then performing standard PCA on the resulting latent vectors. We used a pretrained PointNet autoencoder (Qi et al., 2017) from the public repository https://github.com/vinits5/  pc_autoencoder, trained on ModelNet40, to encode each point cloud (chairs and lamps) into a d-dimensional latent representation, on which PCA was applied. Figure 17 shows the resulting 2D projections. We observe some clustering of similar objects; for example, large lamps tend to group together in the lamp dataset, and chairs versus armchairs form distinguishable clusters. The second principal component for chairs appears to correlate with the height of the seat. Beyond these observations, however, PCA provides limited separability (especially for lamps), and the recovered components are difficult to interpret. More generally, this approach presents several important limitations:
• Training a point-cloud autoencoder requires a large collection of distributions. In our case (100 distributions), we need to rely on a pretrained autoencoder trained on related dataset.
• PCA on autoencoder embeddings relies heavily on the geometry learned by the encoder.
The learned geometry is not guaranteed to align with the Wasserstein structure and the recovered principal components may not reflect meaningful modes of variation (as observed in the experiments above). Moreover, for a given autoencoder that we wish to train, different random seeds at initialization can lead to different learned geometries and thus different PCA components, which is not suitable. A.3 APPLICATION OF GPCA TO OUTLIER DETECTION
In this section, we demonstrate how GPCA can be used for outlier detection. The underlying intuition is that GPCA components capture the structure of the dataset on which they are trained, and samples from a different dataset are expected to lie far from the learned components in Wasserstein distance.
In this experiment, we use the ModelNet40 3D point cloud dataset (Wu et al., 2015) and apply GPCA to a subset of 100 randomly selected chair point clouds to compute the first two components.
For a new point cloud X, we define its score as the sum of the Wasserstein distances between X and its projections onto the first two learned GPCA components. To compute the Wasserstein distance between X and a component, we use ot.emd from the POT library. Specifically, for each component, we perform a grid search over 20 equally spaced values of t between t min and t max , computing the Wasserstein distance between X and 2048 samples drawn from the component at each t, and select the t that minimizes this distance. We repeat the same procedure for the second component and sum the two minimal distances to obtain the final score.
We evaluate this approach on 120 point clouds: 60 new chairs (not used for training) and 60 point clouds of cars. The left histogram in Figure 18 shows the resulting scores. We observe that the scores
this section cite: ['b41', 'b33', 'b44']

Section: B THE OTTO-WASSERSTEIN GEOMETRY
In this section, we briefly describe the fiber bundle structure over the Wasserstein space due to Otto (2001), that is behind the Riemannian interpretation of the Wasserstein distance. We then present its restriction to the space of centered non-degenerate Gaussian distributions, which coincides with the Bures-Wasserstein Riemannian geometry on SPD matrices. Finally, we relate Otto's parametrization of geodesics to McCann's interpolation.
We present these well-known results without proofs and refer the interested reader to Otto ( 2001 Consider the space Prob(Ω) of absolutely continuous probability measures with smooth densities with respect to the Lebesgue measure, and support included in a compact set Ω ⊂ R d , as well as the space Diff(Ω) of diffeomorphisms on Ω. These spaces can be equipped with an infinite-dimensional manifold structure, see e.g. Ebin & Marsden (1970), that we will not describe here. The tangent space of Diff(Ω) at φ ∈ Diff(Ω) is given by
T φ Diff(Ω) = {v • φ, v : Ω → R d vector field}.
We fix a reference measure ρ ∈ Prob(Ω) and equip Diff(Ω) with the L 2 -metric with respect to ρ, defined for any tangent vectors u • φ, v • φ ∈ T φ Diff(Ω) as
⟨u • φ, v • φ⟩ L 2 (ρ) : = (u • φ) • (v • φ) dρ = u • v dµ,
where µ = φ # ρ. Then the space of diffeomorphisms can be decomposed into fibers, defined to be equivalence classes under the projection π : Diff(Ω) → Prob(Ω), φ → φ # ρ.
Specifically, the fiber over µ ∈ Prob(Ω) is given by π -1 (µ) = {φ ∈ Diff(Ω), φ # ρ = µ}, see Figure 19 (right). The tangent space to the fiber π -1 (µ) at φ ∈ Diff(Ω) and its orthogonal with respect to the L 2 (ρ)-metric are refered to as the vertical and horizontal spaces respectively :
Ver φ : = ker dπ φ , Hor φ := (Ver φ ) ⊥ ,
where dπ φ : T φ Diff(Ω) → T π(φ) Prob(Ω) denotes the differential of π at φ. Moving along vertical vectors in Diff(Ω) means staying in the same fiber, i.e. projecting always to the same measure µ in the bottom space. On the contrary, moving along horizontal vectors means moving orthogonally to the fibers, i.e., in the direction that gets fastest away from the fiber. The following proposition gives the form of vertical and horizontal vectors. Proposition 6. Let φ ∈ Diff(Ω). Then
Ver φ = {w • φ, ∇ • (wµ) = 0}, Hor φ = {∇f • φ, f ∈ C ∞ (Ω)}.
The following results state that line segments and L 2 (ρ)-distances in Diff(Ω) can be used to compute Wasserstein geodesics and distances in the space of probability measures Prob(Ω), provided we restrict to horizontal displacements. Proposition 7. The projection π : Diff(Ω) → Prob(Ω) is a Riemannian submersion, i.e. dπ φ : Hor φ → T π(φ) Prob(Ω) is an isometry for any φ ∈ Diff(Ω).
This implies the following. Proposition 8 (Proposition 2 in main). Any geodesic t → µ(t) for the Wasserstein metric in equation 2 is the π-projection of a line segment in Diff(Ω) going through a diffeomorphism φ at horizontal speed ∇f • φ for some smooth function f ∈ C(R d ). That is, for t defined in a certain interval (t min , t max ),
µ(t) = π(φ + t∇f • φ) = (id +t∇f ) # (φ # ρ).(17)
Another geodesic μ(t) = π(φ + t∇ f • φ) is orthogonal to µ(t) at t = 0 for the Riemannian metric inducing the Wasserstein distance if and only if ⟨∇f
• φ, ∇ f • φ⟩ L 2 (ρ) = 0.
We comment on the link between this parametrization and McCann's interpolation in Section B.3.
this section cite: ['b31', 'b12']

Section: B.2 THE OTTO-WASSERSTEIN GEOMETRY OF GAUSSIAN DISTRIBUTIONS
The Bures-Wasserstein distance in equation 3 on the space S ++ d of symmetric positive definite (SPD) matrices is the geodesic distance induced by a Riemannian metric g BW , which can be written in different ways. Here we use the expression from (Thanwerdas, 2022, Table 4.7), defined for Σ = P DP ⊤ ∈ S ++ d and U = P U ′ P ⊤ ∈ S d , by
g BW Σ (U, U ) = 1 2 1≤i,j≤d 1 d i + d j U ′ ij 2 ,(18)
where the d i 's are the diagonal elements of D. The associated Riemannian geometry can be described by Otto's fiber bundle restricted to the space of centered Gaussian distributions, in the following way.
In this setting, diffeomorphisms are restricted to invertible linear maps φ : u → Au for some invertible matrix A, i.e. the space of diffeomorphisms is replaced by the Lie group of invertible matrices GL d . Tangent vectors are then given by linear maps u → Xu for any matrix X ∈ R d×d . Fixing the standard normal distribution ρ = N (0, Id) as reference measure, the L 2 -metric with respect to ρ between u → Xu and u → Y u is then written, for any X, Y ∈ R d×d :
R d φ(u) ⊤ ψ(u)dρ(u) = R d tr(φ(u)ψ(u) ⊤ )dρ(u) = tr R d Xuu ⊤ Y ⊤ dρ(u) = tr(XY ⊤ ),
yielding the standard Frobenius inner product on (the tangent space of) GL d . We obtain a fibration of the top space GL d over the bottom space S ++ d by considering the following projection
π : GL d → S ++ d , A → AA ⊤ ,(19)
see Figure 19 (left). The fiber over Σ ∈ S
++ d is π -1 (Σ) = {A ∈ GL d , AA ⊤ = Σ} = Σ 1/2 O d ,(20)
where O d denotes the space of orthogonal matrices and Σ 1/2 denotes the only SPD square root of the SPD matrix Σ. The differential of the projection π(A) = AA ⊤ is given by
dπ A (X) = XA ⊤ + AX ⊤ .(21)
Therefore, vertical vectors, which are those tangent to the fibers, or equivalently, those belonging to the kernel of dπ A (X), are given by
Ver A : = {X ∈ R d×d , XA ⊤ + AX ⊤ = 0} = {X ∈ R d×d , XA ⊤ is antisymmetric} = {X = K(A ⊤ ) -1 , K ∈ S ⊥ d } = S ⊥ d (A ⊤ ) -1 .
where S ⊥ d denotes the space of antisymmetric matrices of size d. Once again, moving along vertical vectors in GL d means staying in the same fiber, i.e. projecting always to the same SPD matrix in the bottom space S ++ d . Horizontal vectors are those that are orthogonal to all vertical vectors (for the Frobenius metric), i.e. matrices X such that for any antisymmetric matrix K:
0 = ⟨X, K(A ⊤ ) -1 ⟩ = tr(XA -1 K ⊤ )
which is equivalent to XA -1 symmetric (this can be seen by taking for K the basis elements of S ⊥ d in the above equation), yielding
Hor A : = {X ∈ R d×d , (A ⊤ ) -1 X ⊤ = XA -1 } = {X ∈ R d×d , X ⊤ A -A ⊤ X = 0} = {X = KA, K ∈ S d } = S d A
where S d denotes the space of symmetric matrices. Proposition 9. The projection π : GL d → S ++ d , A → AA ⊤ is a Riemannian submersion, i.e. dπ A is an isometry from Hor A equipped with the Frobenius inner product to T π(A) S ++ d equipped with the inner product g BW π(A) , for any A ∈ GL d .
Just like in the general case, this yields a way to lift the computation of geodesics and distances. Proposition 10 (Propositon 1 in main). Any geodesic t → Σ(t) in S ++ d for the Bures-Wasserstein metric in equation 3 is the π-projection of a horizontal line segment in GL d , that is
Σ(t) = π(A + tX) = (A + tX)(A + tX) ⊤ , A ∈ GL d , X ∈ Hor A , (22
)
where t is defined in a certain time interval (t min , t max ). Also, the Bures-Wasserstein distance between two covariance matrices Σ 1 , Σ 2 ∈ S ++ d is given by the minimal distance between their fibers
BW 2 (Σ 1 , Σ 2 ) = inf Q1,Q2∈O d ∥Σ 1/2 1 Q 1 -Σ 1/2 2 Q 2 ∥ = inf Q∈SO d ∥Σ 1/2 1 -Σ 1/2 2 Q∥,(23)
where ∥ • ∥ is the Frobenius norm and SO d is the special orthogonal group.
Formula in equation 22 and the first equality of equation 23 are direct consequences of the fact that π is a Riemannian submersion. To obtain the second equality of equation 23, we first notice that optimizing on Q 1 , Q 2 ∈ O d is equivalent to optimizing on a single Q ∈ O d thanks to the invariance of the Frobenius metric w.r.t. the right action of O d . And second, that the infimum is attained at (see (Bhatia et al., 2019, Equations 3 and 35))
Q * = Σ -1/2 2 T Σ 1/2 1 , where T = Σ -1/2 1 (Σ 1/2 1 Σ 2 Σ 1/2 1 ) 1/2 Σ -1/2 1
is the Monge map from Σ 1 to Σ 2 (see (Malagò et al., 2018, equation 8)), and so Q * has positive determinant and belongs to SO d .
Thus the closest element of the fiber
π -1 (Σ 2 ) to Σ 1/2 1 is given by Σ 1/2 2 Q * = T Σ 1/2 1 , i.e. by left multiplying Σ 1/2 1
by the Monge map T . This is more generally true for any representative of Σ 1 :
Proposition 11. Let Σ 1 , Σ 2 ∈ S ++ d , T the Monge map from Σ 1 to Σ 2 , A 1 ∈ π -1 (Σ 1 ).
Then A 2 := T A 1 is said to be aligned with respect to A 1 , that is, it is the closest point in π -1 (Σ 2 ) to A 1 . More precisely, we have
1. A 2 -A 1 = (T -I)A 1 ∈ Hor A1 2. Log Σ1 (Σ 2 ) : = dπ A1 ((T -I)A 1 ) = (T -I)Σ 1 + Σ 1 (T -I) S + + d Id A 1 Σ 2 Id π G L d X ∈ Hor A 1 Σ 1 A 2 π -1 (I d ) π -1 (Σ 1) π -1 (Σ 2)
ho riz on ta l ge od es ic W as se rs te in ge od es ic
P r o b ( Ω ) id φ 1 µ 2 ρ π D iff (Ω ) ∇f ∈ Hor φ 1 µ 1 φ 2 π -1 (ρ ) π -1 (µ 1) π -1 (µ 2)
ho riz on ta l ge od es ic W as se rs te in ge od es ic
this section cite: []

Section: BW
2 (Σ 1 , Σ 2 ) = ∥Log Σ1 Σ 2 ∥ BW Σ1 = ∥(T -I)A 1 ∥ where Log is the Riemannian logarithm map, ∥ • ∥ BW Σ = g BW Σ (•, •) and ∥ • ∥ is the Frobenius norm.
This means that to compute the Bures-Wasserstein distance between two covariance matrices Σ 1 and Σ 2 , one can consider any representative A 1 in the fiber over Σ 1 , compute the representative A 2 of Σ 2 aligned to A 1 (using the Monge map) and finally compute the Frobenius norm of A 2 -A 1 .
this section cite: []

Section: B.3 GEODESIC PARAMETRIZATION
There are two classical parameterizations for Wasserstein geodesics in the space of a.c. probability measures.
this section cite: []

Section: McCann's interpolation
The first one, due to McCann (1997), is given between two probability distributions µ 0 and µ 1 , and depends on the optimal transport map in equation 2, obtained as the gradient of a convex function u, that is T µ1 µ0 = ∇u and
µ t = ((1 -t) id +t∇u) # µ 0 = (id +t(∇u -id)) # µ 0 , t ∈ [0, 1].(24)
Otto's geodesic The second one, exploiting Otto's fiber bundle geometry in Otto (2001), consists in writing a geodesic in the Wasserstein space as the projection of a horizontal geodesic in the total space of diffeomorphisms. Such a horizontal geodesic is a line segment going through a diffeomorphism φ with a horizontal speed ∇f • φ, where f is any smooth function (not necessarily convex). Therefore we get
µ s = (φ + s∇f • φ) # ρ = (id +s∇f ) # (φ # ρ), s ∈ (s 0 , s 1 ).
(25) In this second expression, the bounds on the time s depends on the function f . Indeed, for µ s to be a geodesic, id +s∇f needs to remain is the space of diffeomorphisms for a given s, which means that id +sHess f needs to be positive definite. Therefore, we get the following conditions depending on the minimum λ min and maximum λ max eigenvalues of Hess f :
   s ∈ (-∞, -1/λ min ) if λ max < 0, s ∈ (-1/λ max , +∞) if λ min > 0, s ∈ (-1/λ max , -1/λ min ) if λ min < 0 < λ max . (26)
It is clear that equation 24 is a particular case of equation 25, where we choose φ # ρ = µ 0 and ∇f = ∇u -id. Conversely, one can write equation 25 under the form of equation 24. For a given diffeomorphism φ and function f , consider the geodesic given by equation 25, and set µ 0 = φ # ρ. Assume that we are in the case where all eigenvalues of Hess f are negative, then s must be in ] -∞, -1/λ min [. Consider s * ∈]0, -1/λ min [, and define µ 1 : = µ s * = (id +s * ∇f ) # µ 0 . Setting t = s/s * we have that the geodesic between µ 0 and µ 1 is written
µ t = (id +ts * ∇f ) # µ 0 = (id +t(∇u -id)) # µ 0 , t ∈ [0, 1].
for u(x) = s * f + ∥x∥ 2 /2. Now for any eigenvalue λ i of H f the Hessian of f , we have λ i > λ min > -1/s * i.e. s * λ i + 1 > 0. by the interval of definition of s * . This means that the Hessian H u = s * H f + id is positive definite, which means that u is necessarily convex. The other cases work similarly.
The Gaussian case Transposing Otto's formulation in equation 25 to the case of a geodesic between Gaussian distributions means that for A ∈ GL d and X ∈ Hor A such that ∥X∥ = 1, the interval of definition of a geodesic depends on the invertibility of A + sX. In turn, the maximal interval of definition of s ∈ (s 0 , s 1 ) is defined from the eigenvalues of XA -1 , through the same formula in equation 26.
this section cite: ['b27', 'b31']

Section: C LINEARIZED OPTIMAL TRANSPORT AND TANGENT PCA
In this section, we provide the definition of linearized Wasserstein distance and details on how to perform tangent PCA for both Gaussian distributions and general a.c. distributions. Tangent PCA is a widely used approach to compute PCA on the Wasserstein space, that consists in embedding probability distributions into the tangent space at some reference measure ρ, and performing PCA in the tangent space with respect to the linearized Wasserstein distance.
this section cite: []

Section: C.1 THE CASE OF CENTERED GAUSSIAN DISTRIBUTIONS
We consider n covariance matrices Σ 1 , . . . , Σ n and their Bures-Wasserstein barycenter (or Fréchet mean) Σ, that is, the SPD matrix verifying (see Agueh & Carlier (2011)):
Σ = arg min Σ∈S ++ d n i=1 BW 2 2 (Σ, Σ i ).(27)
The idea behind tangent PCA is to represent each data point by the corresponding tangent vector, given by the Riemannian logarithm map, in the tangent space at the reference point Σ, i.e.
{LogΣΣ i } n i=1 ⊂ TΣS ++ d . (28
)
Now, one can lift the computations from the tangent space at Σ to the horizontal space at a point in the fiber over Σ, say A : = Σ1/2 , by aligning all representatives to A, see Proposition 11. The key point is that the tangent space at Σ equipped with the Bures-Wasserstein Riemannian metric is isometric to Hor A : = S d A equipped with the Frobenius inner product -where we recall that S d is the space of symmetric matrices. This means that instead of performing PCA for the Bures-Wasserstein inner product on the tangent vectors in equation 28, we can instead perform linear PCA on their pre-images by dπ A , see Proposition 11:
{(T i -I)A} n i=1 ⊂ Hor A1 , where T i = Σ -1/2 i (Σ 1/2 i ΣΣ 1/2 i ) 1/2 Σ -1/2 i . T i is the optimal transport map from Σ to Σ i , see Section B.2. Now, noticing that ⟨K 1 A, K 2 A⟩ = Tr(K 1 AA ⊤ K ⊤ 2 ) = Tr(K 1 ΣK ⊤ 2 ), ∀K 1 , K 2 ∈ S d
, we see that the space Hor A equipped with the Frobenius inner product is itself isometric to S d equipped with the Frobenius inner product weighted by Σ. Therefore, tangent PCA is performed through Euclidean PCA on the (centered) vectors {T i -I} n i=1 , in the vector space S d , with respect to the Frobenius metric weighted by Σ. Another way to see this is by noticing that the linearized Bures-Wasserstein distance BW 2, Σ with respect to Σ is given by
BW 2, Σ(Σ 1 , Σ 2 ) : = ∥LogΣΣ 1 -Log ΣΣ 2 ∥ BW Σ = ∥dπΣ1/2((T 1 -I) Σ1/2 ) -dπΣ1/2 ((T 2 -I) Σ1/2 ∥ BW Σ = ∥(T 1 -I) Σ1/2 -(T 2 -I) Σ1/2 ∥ = ∥(T 1 -T 2 ) Σ1/2 ∥
where ∥•∥ BW denotes the norm associated to the Bures Wasserstein Riemannian metric in equation 18, π is Otto's projection in equation 19, and we have used Propositions 9 and 11. Finally,
BW 2, Σ(Σ 1 , Σ 2 ) : = ∥LogΣΣ 1 -Log ΣΣ 2 ∥ BW Σ = ∥T 1 -T 2 ∥Σ,(29)
where ∥ • ∥Σ denotes the Frobenius norm weighted by Σ.
this section cite: ['b0']

Section: C.2 THE CASE OF A.C. DISTRIBUTIONS
Similarly, one can embed a.c. probability distributions ν 1 , . . . , ν n into the L 2 (ρ) space at some a.c. reference measure ρ through the optimal maps ν i → T νi ρ in the Monge problem in equation 2. Then, the Wasserstein distance can be approximated by the linearized Wasserstein distance in Wang et al. (2013) given by
W 2,ρ (ν 1 , ν 2 ) = ∥T ν1 ρ -T ν2 ρ ∥ L 2 (ρ) .(30)
Note that as previously mentioned, this metric induces distortions : while the radial distances from ρ to any µ i are preserved, that is ∥id - Letrouit & Mérigot (2024) proved however, that under some assumptions, W 2,ρ is bi-Hölder equivalent to W 2 , which indicates that the distortion effect can be controlled.
T νi ρ ∥ L 2 (ρ) = W 2 (ρ, ν i ), other distances are not ∥T ν1 ρ -T ν2 ρ ∥ L 2 (ρ) ̸ = W 2 (ν 1 , ν 2 ). A recent paper by
Then, denoting νn the Wasserstein barycenter as in Agueh & Carlier (2011
) of ν 1 , . . . , ν n , that is the solution of νn ∈ arg min ν n i=1 W 2 2 (ν, ν i ),(31)
tangent PCA consists in performing classical PCA, see e.g. Ramsay & Silverman (2002), of
(T νi νn -id) n i=1 in the Hilbert space L 2 (ν n ).
this section cite: ['b43', 'b23', 'b0', 'b34']

Section: D GEODESIC PCA FOR GAUSSIAN DISTRIBUTIONS
In this section, we present the proofs related to geodesic PCA for Gaussian distributions and the implementation of our algorithm in this case.
this section cite: []

Section: D.1 PROOFS RELATED TO GPCA FOR GAUSSIAN DISTRIBUTIONS
We first prove the existence of mimimizers for the GPCA problems lifted to Otto's fiber bundle.
Lemma 1. The GPCA problem in equation 12 for the first component admits a global minimum.
Proof. First, let us define the set of normalized matrices B : = {X ∈ R d×d , ∥X∥ = 1}. By denoting λ min (resp. λ max ) the smallest (resp. largest) eigenvalue of XA -1 , extending the geodesic t → A + tX as far as possible (see Section B.3) means that the closed interval [t min , t max ] is defined for some fixed ε > 0 by
   (-∞, -1/λ min -ε] if λ max < 0, [-1/λ max + ε, +∞) if λ min > 0, [-1/λ max + ε, -1/λ min -ε] if λ min < 0 < λ max .(32)
Let us now consider the function F :
GL d × B × (R d×d ) n -→ R (A, X, (Q i ) n i=1 ) -→ n i=1 ∥A + p (A,X) (t i )X -Σ 1/2 i Q i ∥ 2 =: n i=1 g i (A, X, Q i ),
where
t i = ⟨Σ 1/2 i Q i -A, X⟩ and p (A,X) : R → R is the projection operator that clips a point t into [t min , t max ], which depends on A and X. Then the function F is continuous on GL d × B × (R d×d ) n
as composition of linear and continuous functions. Note that the function (A, X) → p (A,X) (t i ) is continuous by eigenvalue continuity, see Li & Zhang (2019). Additionally, the function F is coercive (see e.g. Zalinescu (2002)
) on GL d × B × (R d×d ) n . Indeed, on a diagonal {A = Σ 1/2 i Q i , for (A, Q i ) ∈ GL d × R d×d } for some i ∈ {1, . . . , n}, we have t i = 0, and therefore we have either g i (A, X, Q i ) = 0 if p (A,X) (0) = 0, or g i (A, X, Q i ) = ε∥X∥ 2 = ε otherwise. This would imply that g i (A, X, Q i ) doesn't go to infinity when the norm ∥(A, X, Q i )∥ → ∞. However, in this case, we have g j (A, X, Q j ) → ∞ when ∥(A, X, Q j )∥ → ∞ for any j ̸ = i. Moreover, as p (A,X) (t i
) is a clipping, it won't play a role in the coercivity. We conclude by the fact that the function
(A, X) → X ⊤ A -A ⊤ X is continuous, implying that the set of constraint {(A, X) ∈ GL d × R d×d : X ⊤ A -A ⊤ X = 0}
is closed and B and SO d are compact. The optimization problem in equation 12 thus admits a global minimum.
Note that this result also applies for the second component in equation 13 and the higher order components. Proposition 12 (Proposition 3 in main). Let π :
GL d → S ++ d , A → AA ⊤ and (A 1 , X 1 , (Q i ) n i=1 ) be a solution of inf F (A 1 , X 1 , (Q i ) n i=1 ) : = n i=1 ∥A 1 + p A1,X1 (t i )X 1 -Σ 1/2 i Q i ∥ 2 , subject to A 1 ∈ GL d , X 1 ∈ Hor A1 , ∥X 1 ∥ 2 = 1, Q 1 , . . . , Q n ∈ SO d .
Then there exist t min , t max ∈ R such that the geodesic Σ :
t ∈ [t min , t max ] → π(A 1 + tX 1 ) in S ++ d minimizes equation 11. Proof. A horizontal geodesic in GL d is a straight line going through a base point A ∈ GL d in the direction of a horizontal vector X ∈
Hor A (that we consider normalized, ie. ∥X∥ 2 = 1), i.e. t → A + tX ∈ GL d . Denoting [t min , t max ] the interval constructed in equation 32 which depends on the eigenvalues of XA -1 , we have that (π(A + tX)) t∈[tmin,tmax] is a geodesic in the Bures-Wasserstein sense, see Proposition 1, and min t∈[tmin,tmax]
BW 2 2 (π(A + tX), Σ i ) = min t∈[tmin,tmax] inf Qi∈SO d ∥A + tX -Σ 1/2 i Q i ∥ 2 = inf Qi∈SO d ∥A + p (A,X) (t i )X -Σ 1/2 i Q i ∥ 2 ,
where [tmin,tmax] , solution of problem in equation 11.
t i = ⟨Σ 1/2 i Q i -A, X⟩ is the (orthogonal) projection time of Σ 1/2 i Q i onto the line t → A + tX. We therefore deduce that a set of solution (A, X, (Q i ) n i=1 ) of equation 12 defines a proper geodesic (π(A + tX)) t∈
Proposition 13 (Proposition 5 in main). Let ν i = N (m i , σ 2 i ) for i = 1, . . . n be n univariate Gaussian distributions. The first principal geodesic component t ∈ [0, 1] → µ(t) solving equation 1 remains in the geodesic space of Gaussian distributions for all t ∈ [0, 1].
Proof. Let Prob 2 (R) be the set of a.c. probability measures on R that have finite second moment, and Q the set of corresponding quantile functions :
Q = {F -1 ν ; ν ∈ Prob 2 (R)} Q is the set of increasing, left-continuous functions q : (0, 1) → R, and a convex cone in L 2 ([0, 1]), the set of square-integrable functions on [0, 1]. The mapping Φ : ν → F -1 ν (33)
defines an isometry between Prob 2 (R) equipped with the Wasserstein metric, and Q equipped with the L 2 metric (see e.g. Bigot et al. (2017)), that is, for any µ, ν ∈ Prob 2 (R),
W 2 (µ, ν) = ∥F -1 µ -F -1 ν ∥ L 2 ([0,1])
. The map Φ in equation 33 also defines an isometry from the set of (univariate) Gaussian distributions to the set of all Gaussian quantile functions G. This space G is the upper-half of the plane F spanned by the constant function 1 and the quantile function F -1 0 of the standard normal distribution:
G = R • 1 + R * + • F -1 0 ⊂ F : = span(1, F -10
). Now, consider n normal distributions ν 1 , . . . , ν n , and (µ(t)) t∈[0,1] the first principal geodesic component found by minimizing equation 1, the sum of squared residuals in Prob 2 (R). Since µ is a Wasserstein geodesic in Prob 2 (R) and Φ is an isometry, the curve t → Φ(µ
)(t) = F -1 µ(t) is an L 2 ([0, 1])-geodesic in Q, i.e. a line segment t ∈ [0, 1] → F -1 µ(t) = (1 -t)F -1 µ(0) + tF -1 µ(1) .
Since {1, F -1 0 } forms an orthonormal basis of F, the orthogonal projection of this line segment on F is given by t ∈ [0, 1] → ⟨F -1 µ(t) , 1⟩1 + ⟨F -1 µ(t) , F -1 0 ⟩F -1 0 , which lies in G. To see this, we need to show that the following value is positive:
⟨F -1 µ(t) , F -1 0 ⟩ = 1 0 F -1 µ(t) (y)F -1 0 (y)dy = R xF -1 0 • F µ(t) (x)dµ(t)(x) = E(XT (X)),
where X ∼ µ(t) and T = F -1 0 •F µ(t) is the Monge map from µ(t) to the standard normal distribution. Since T is increasing, we indeed have E(XT (X)) > 0 (see e.g. the proof of Theorem 2. 2 in Schmidt  (2014)).
Finally, since Φ(µ) orthogonally projects from Q to G w.r.t the L 2 metric and Φ defines an isometry, we get that the geodesic µ orthogonally projects to a geodesic π(µ) in the space of Gaussian distributions, w.r.t. the Wasserstein metric. By the distance minimizing property of orthogonal projections, we know that the cost function in equation 1 evaluated at π(µ) is no larger than its value at µ. Since µ is optimal, we get that µ = π(µ) and µ belongs to the space of Gaussian distributions.
Proposition 14. Let Σ 1 , Σ 2 two SPD matrices that are diagonalizable in the same orthonormal basis, i.e.
Σ 1 = P a 2 1 0 0 b 2 1 P ⊤ and Σ 2 = P a 2 2 0 0 b 2 2 P ⊤ , where P is orthogonal. Then BW 2 2 (Σ 1 , Σ 2 ) = (a 1 -a 2 ) 2 +(b 1 -b 2 ) 2
, and thus the Bures-Wasserstein geodesic between Σ 1 and Σ 2 is given by
Σ(t) = P ((1 -t)a 1 + tb 1 ) 2 0 0 ((1 -t)a 2 + tb 2 ) 2 P ⊤ , 0 ≤ t ≤ 1.
Proof. This is a straightforward computation using equation 3.
Proposition 15. Let us consider n = 2p covariance matrices Σ i = Σ(a, b, θ i ) as defined in equation 16, where θ i = iπ/n for i = 0, . . . , n -1. Then, the Bures-Wasserstein barycenter in equation 27 of these covariance matrices is given by Σ = (a + b) 2 /4 I.
Proof. Each pair of covariance matrices
Σ i = P θi a 2 0 0 b 2 P ⊤ θi , and Σ i+p = P θi+π/2 DP ⊤ θi+π/2 = P θi b 2 0 0 a 2 P ⊤ θi
are diagonalizable in the same basis, and so by Proposition 14, the geodesic from
Σ i to Σ i+p is Σ(t) = P θi ((1 -t)a + tb) 2 0 0 ((1 -t)b + ta) 2 P ⊤ θi , 0 ≤ t ≤ 1.
In particular, the Fréchet mean is given by Σ = Σ(1/2) = ((a + b)/2) 2 I. Since each pair of covariance matrices has the same Fréchet mean, the Fréchet mean of the whole set Σ 1 , . . . , Σ n is also given by Σ.
Proposition 16 (Proposition 4 in main). Let Σ ∈ S ++ 2 with eigenvalues a 2 , b 2 and Σ ′ = P θ ΣP ⊤ θ where P θ is the rotation matrix of angle θ. Then, denoting Σ = ((a + b)/2) 2 I we have
BW 2 2 (Σ, Σ ′ ) BW 2 2, Σ(Σ, Σ ′ ) = 1 - a -b a + b 2 cos 2 θ + O((a -b) 4 ).(34)
Proof. Recall that the linearized Bures-Wasserstein distance at Σ between Σ and Σ ′ is given by the distance between their images by the Riemannian logarithm map U : = LogΣΣ and U ′ : = LogΣΣ ′ in the tangent space at Σ, i.e.
BW 2, Σ(Σ, Σ ′ ) = ∥U -U ′ ∥ BW Σ ,
where ∥•∥ BW denotes the norm associated to the Bures-Wasserstein Riemannian metric in equation 18.
As in any Riemannian manifold, the true geodesic distance can be approximated by this linearized distance in the tangent space, corrected by the curvature (see e.g. Lemma 1 in Harms et al. (2019)) :
BW 2 2 (Σ, Σ ′ ) = ∥U -U ′ ∥ BW Σ 2 - 1 3 RΣ(U, U ′ , U, U ′ ) + O(∥U ∥ BW Σ + ∥U ′ ∥ BW Σ ) 6 , (35
)
where RΣ is the curvature tensor.
Recall from equation 18 that the Bures-Wasserstein norm of a vector U is expressed in an eigenvector basis of the base point, here Σ. Since any basis is an eigenvector basis of Σ, it is convenient to choose that of Σ, which we can assume without loss of generality to be the canonical basis. Thus we write Σ = D where D = diag(a 2 , b 2 ) and Σ ′ = P θ DP ⊤ θ , and the norm associated to the Bures-Wasserstein Riemannian metric is given by
∥U ∥ BW Σ = 1 2 1≤i,j≤2 1 d i + d j U 2 ij
where the d i 's are the eigenvalues of Σ, given here by d 1 = d 2 = ((a + b)/2) 2 . From Proposition 11 we have
U : = Log ΣΣ = (T -I) Σ + Σ(T -I), U ′ : = LogΣΣ ′ = (T ′ -I) Σ + Σ(T ′ -I),
where
T : = Σ-1/2 ( Σ1/2 Σ Σ1/2 ) 1/2 Σ-1/2 = 2 a + b D 1/2 , T ′ : = Σ-1/2 ( Σ1/2 Σ ′ Σ1/2 ) 1/2 Σ-1/2 = 2 a + b P θ D 1/2 P ⊤ θ ,
and easily get
U = a 2 -b 2 2 J, U ′ = a 2 -b 2 2 P θ JP ⊤ θ , where P θ JP ⊤ θ =
cos 2θ sin 2θ sin 2θ -cos 2θ and J = diag(1, -1). Thus after some computations we obtain
∥U ∥ BW Σ = ∥U ′ ∥ BW Σ = |a -b|/ √ 2, BW 2, Σ(Σ, Σ ′ ) = ∥U -U ′ ∥ BW Σ = √ 2|(a -b) sin θ|. (36
)
To compute the curvature tensor, we use the following formula from (Thanwerdas, 2022, Table 4.7) RΣ(U, U ′ , U, U ′ ) = 3 2 i,j
d i d j d i + d j [U 0 , U ′ 0 ] 2 ij
where [A, B] = AB -BA is the Lie bracket of matrices, U 0 and U ′ 0 are the only symmetric matrices verifying the Sylvester equations U = U 0 Σ + ΣU 0 and U ′ = U ′ 0 Σ + ΣU ′ 0 respectively. Since Σ is a multiple of the identity, we easily get
U 0 = a -b a + b J, U ′ 0 = a -b a + b P θ JP ⊤ θ and straightforward computations yield RΣ(U, U ′ , U, U ′ ) = 3 2 (a -b) 4 (a + b) 2 sin 2 2θ.(37)
Finally, putting together equation 35, equation 36 and equation 37 and we obtain
BW 2 2 (Σ, Σ ′ ) = BW 2 2, Σ(Σ, Σ ′ ) -2 (a -b) 4 (a + b) 2 sin 2 θ cos 2 θ + O((a -b) 6 ),
and dividing by the squared linearized optimal transport distance yields the desired result.
Published as a conference paper at ICLR 2026
this section cite: ['b24', 'b45', 'b16', 'b41']

Section: D.2 IMPLEMENTATION OF GPCA FOR GAUSSIAN DISTRIBUTIONS
As described in Section 3, the first and second components of geodesic PCA are respectively found by solving the minimization problems in equation 12 and equation 13. The geodesic components are given by
Σ i (t) = (A i + tX i )(A i + tX i ) ⊤ , for i = 1, 2,
where A 1 ∈ GL d and X 1 ∈ Hor A1 are minimizers of equation 12, and A 2 ∈ GL d and X 2 ∈ Hor A2 minimizers of equation 13. The matrix π(A 2 ) is the crossing point through which all geodesic components intersect, see Figure 2. The higher order components are found in a analogous way: for the k-th component, we search for a horizontal segment t → A k + tX k where A k belongs to the fiber over the intersection point (we parametrize it w.r.t. the previous position in the fiber, i.e. A k = A k-1 R k-1 for a certain R k-1 ∈ SO d ) and the horizontal velocity vector X k is orthogonal to the lifts of the velocity vectors of the previous component. Thus, the k-th component, k ≥ 3, solves: In dimension d = 2, any rotation matrix Q can be parametrized by a scalar angle θ and both steps are solved using the Sequential Least Squares Programming (SLSQP) algorithm (see e.g. Ma et al. ( 2024)) available on the scipy python library and given by Virtanen et al. (2020). In higher dimension, each minimization with respect to a rotation matrix is performed using Riemannian gradient descent on SO d , relying on the Riemannian geometry of SO d induced by the standard Frobenius metric of the ambient space R d×d . In particular we use the exponential map implemented in the Python library geomstats developed by Miolane et al. (2020). More details on the Riemannian geometry of SO d and the Riemannian gradient descent procedure can be found e.g. in (Boumal, 2023, Sections 7.4 and 4.3).
inf F (A k , X k , (Q i ) n i=1 ) subject to A k = A k-1 R k-1 , R k-1 ∈ SO d , X k ∈ Hor A k , ∥X k ∥ 2 = 1, ⟨X k , X k-ℓ R k-ℓ . . . R k-1 ⟩ = 0, 1 ≤ ℓ ≤ k -1, Q 1 , . . . , Q n ∈ SO d .(38)
Unfortunately, we cannot ensure the convergence of the iterates of the proposed block alternating algorithm, as classical arguments require uniqueness of the minimizer at each iterations as proven in Powell (1973). This is unachievable in our problem: Scalability of the algorithm Surely, the computational time of our algorithm for Gaussian distributions will increase with the dimension. However, the algorithm can be made less sensitive to the number of input covariance matrices by parallelizing (Step 2) of our algorithm, which consists in updating the orthogonal matrices (Q i ) n i=1 . This would significantly reduce the overall computational cost of the algorithm. Also, we currently use the scipy toolbox to solve (Step 1), which could also be accelerated using a more powerful optimization toolbox. All experiments were conducted on a single V100 GPU with 32GB of memory, using a shared set of hyperparameters detailed in Table 1. The same hyperparameters are used for computing both the first and second geodesic components, except for the number of gradient steps (see Table 1), which is increased for the second component. This is likely due to the additional complexity introduced by the intersection and orthogonality constraints enforced through regularization. Both f ψ and φ θ are implemented as standard multilayer perceptrons (MLPs) with four hidden layers of width 128. We use ELU activation functions in f ψ because its gradient is used to parameterize a transport map in our formulation, and ELUs are commonly employed in such settings. The Sinkhorn divergence S ε is used in the loss function as a surrogate for the squared Wasserstein distance to compute the geodesic components. The regularization parameter ε must be adapted to the scale of the data; we set it as ε = 0.01 E x,x ′ ∼νi ∥x -x ′ ∥ 2 , where the expectation is approximated via Monte Carlo using the current minibatch samples. Note that setting ε this way is the default configuration in the OTT-JAX library. For computing the second geodesic component, we fix the regularization coefficients λ O and λ I to 1.0, which we found to be robust across all experiments. While increasing them (e.g., to 10.0) typically yields similar results, excessively large values may degrade performance. Conversely, if these regularization terms are too small, the algorithm tends to recover the first component as the second, due to its lower cost. In practice, we monitor the regularization terms during optimization to ensure they decrease sufficiently relative to their initial values, confirming that the optimization effectively optimize the intersection and orthogonality constraints. To determine the hyperparameters in Table 1, we performed a grid search over the optimizer learning rate for the t i in 5e -4 , 1e -3 , 5e -3 , 1e -2 , and over the regularization coefficients λ O and λ I in 0.1, 1.0, 10.0, 100.0. We found that setting both regularization terms to 1.0 consistently yielded good performance across all experiments, see Section E.2.
Note on φ parameterization. Note that although φ is theoretically required to be a diffeomorphism in Otto's parameterization of geodesics (equation 9), we parameterize it using a simple MLP. Initially, we experimented with normalizing flows to ensure invertibility, but observed that a standard MLP yielded similar results. In Otto's geodesic framework, φ serves to modify the reference measure ρ and define the measure at t = 0 along the geodesic. If φ is not a diffeomorphism and the pushforward φ # ρ is not absolutely continuous, the resulting geodesic becomes degenerate, which may hinder optimization of the loss equation in equation 1. In practice, however, we found that the MLP φ θ reliably produces absolutely continuous measures, which is sufficient for our method.
this section cite: ['b42', 'b28', 'b32']

Section: E.2 IMPACT OF THE REGULARIZATIONS ON GPCA
For the estimation of the second GPCA component, we introduce two regularization terms, I(ξ θ,ψ , ξ θ2,ψ2 , t 1 inter , t 2 inter ) and O(∇f ψ (φ θ ), ∇f ψ2 (φ θ2 )), with their associated regularization coefficients λ I and λ O . The first term enforces that the two components intersect, while the second ensures that the components remain orthogonal. Experimentally, we observe that setting both coefficients to λ I = λ O = 1.0 robustly enforces these constraints across all experiments while still producing meaningful principal components. Conversely, if these regularization terms are too small, the algorithm tends to recover the first component as the second, at it gives the lowest cost. In practice, we monitor the regularization terms during optimization to ensure they decrease sufficiently relative to their initial values. This permits to confirm that the optimization effectively optimize the intersection and orthogonality constraints. This section aims at quantifying the impact of the two regularizing coefficients λ I and λ O on the computed geodesics. We focus on the 3D point-cloud experiments with lamps.
this section cite: []

Section: E.2.1 ORTHOGONALITY REGULARIZATION
In this part, we set the regularization term λ I to 1.0 and compute GPCA for different values of λ O . The resulting second component is shown in Figure 20. The GPCA cost of this component, as defined in equation 15, together with the quantity measuring the orthogonality between components, O(∇f ψ (φ θ ), ∇f ψ2 (φ θ2 )), are reported in Table 2. The quantities reported in Table 2 are estimated on batches of size 2048. The variance is computed over 100 runs for the orthogonality measure and 5 runs for the GPCA cost. Note that each run of the orthogonality estimation already involves computing 100 Wasserstein distances, since we have 100 point clouds.
Note that the GPCA cost of the second component should be compared with that of the first component, which is around 3.0. Table 2 shows that for low values of λ O (i.e., 0.001 and 0.01), the orthogonality quantity is large, and the recovered "second" component is in fact identical to the first component, as illustrated in Figure 20. This is also reflected in the GPCA cost (see Table 2), which matches the one of the first component. For higher values of λ O (0.1, 1.0, 10.0, 100.0), the algorithm successfully recovers a distinct second component. For the highest value (i.e., λ O = 100.0), a loss of performance is observed.
λ O
Orthogonality: O(∇f ψ (φ θ ), ∇f ψ2 (φ θ2 )) GPCA cost (second component) 0.001 0.909 ± 0.005 2.96 ± 0.01 0.01 0.811 ± 0.008 3.00 ± 0.01 0.1 2.1 × 10 -3 ± 3 × 10 -4 5.75 ± 0.02 1.0 3.1 × 10 -4 ± 6.5 × 10 -5 5.76 ± 0.02 10.0 2.2 × 10 -4 ± 5 × 10 -5 5.89 ± 0.02 100.0 1.1 × 10 -5 ± 2 × 10 -6 5.99 ± 0.02
Table 2: Orthogonality regularization value and second-component loss for different values of λ O .
this section cite: []

Section: E.2.2 REGULARIZATION ON THE INTERSECTION OF THE GEODESICS
In this part, we set the regularization term λ O to 1.0 and compute GPCA for different values of λ I .
The second component is displayed in Figure 21; the GPCA cost of this component, as well as the quantity measuring the intersection of the components, I(ξ θ,ψ , ξ θ2,ψ2 , t 1 inter , t 2 inter ), are reported in Table 3. The quantities reported in Table 3 are estimated on batches of size 2048. The variance is computed over 100 runs for the intersection measure and 5 runs for the GPCA cost.
We observe from the recovered geodesics in Figure 21 that this regularization term plays a less significant role than the orthogonality term. Moreover, Table 3 shows that increasing λ I does not affect negatively the GPCA cost of the recovered component. Published as a conference paper at ICLR 2026 λ I Intersection: I(ξ θ,ψ , ξ θ2,ψ2 , t 1 inter , t 2 inter ) GPCA cost (second component) 0.001 6.5 × 10 -2 ± 1 × 10 -3 5.76 ± 0.02 0.01 2.3 × 10 -2 ± 4 × 10 -4 5.77 ± 0.02 0.1 2.7 × 10 -3 ± 1 × 10 -4 5.77 ± 0.01 1.0 1.0 × 10 -3 ± 6 × 10 -5 5.76 ± 0.02 10.0 7.2 × 10 -5 ± 3 × 10 -6 5.74 ± 0.02 100.0 3.1 × 10 -5 ± 1 × 10 -6 5.91 ± 0.02
Table 3: Squared Euclidean distance between ξ 1 (t 1 inter ) and ξ 2 (t 2 inter ) and second-component loss for different values of λ I . E.2.3 SCALABILITY OF OUR GPCAGEN ALGORITHM For general distributions, there are two types of "scaling" that can affect the algorithm: 1. Number of probability measures (n): The number of measures ν i directly determines the iterations of the inner loop in Algorithm 1 (line 3). Consequently, the training time scales linearly with n.
2. Dimension of the space (d): As the dimension of the space in which the ν i lies increases, the main challenge consists in accurately estimating the maximum and minimum eigenvalues that the Hessian of f can take. As discussed with reviewer oUMT, in high dimensions, it becomes necessary to use algorithms that avoid computing the full Hessian and instead rely on matrix-vector products, such as the LOBPCG algorithm Duersch et al. (2018). Furthermore, rather than relying solely on the samples in the training batch, an adversarial approach would be needed to track the eigenvectors corresponding to the worst-case eigenvalues.
this section cite: ['b11']

Section: F USE OF LARGE LANGUAGE MODELS (LLMS)
LLMs were used only to assist with polishing the writing; all research ideas, experiments, and analyses were conducted independently by the authors.
this section cite: []

Section: References
Ref_id:b0 Title: Barycenters in the Wasserstein space Year: (2011)
Ref_id:b1 Title: Gradient flows: in metric spaces and in the space of probability measures Year: (2008)
Ref_id:b2 Title: A user's guide to optimal transport. Modelling and Optimisation of Flows on Networks: Cetraro Year: (2009)
Ref_id:b3 Title: On the Bures-Wasserstein distance between positive definite matrices Year: (2019)
Ref_id:b4 Title: Geodesic PCA in the Wasserstein space by convex PCA Year: (2017)
Ref_id:b5 Title: Distribution's template estimate with Wasserstein metrics Year: (2015)
Ref_id:b6 Title: An introduction to optimization on smooth manifolds Year: (2023)
Ref_id:b7 Title: Polar factorization and monotone rearrangement of vector-valued functions Year: (1991)
Ref_id:b8 Title: Populations of unlabelled networks: Graph space geometry and generalized geodesic principal components Year: (2024)
Ref_id:b9 Title: Geodesic PCA versus log-PCA of histograms in the Wasserstein space Year: (2018)
Ref_id:b10 Title: Franc ¸ois-Xavier Vialard, and Gabriel Peyré. Faster wasserstein distance estimation with the sinkhorn divergence Year: (2020)
Ref_id:b11 Title: A robust and efficient implementation of LOBPCG Year: (2018)
Ref_id:b12 Title: Groups of diffeomorphisms and the motion of an incompressible fluid Year: (1970)
Ref_id:b13 Title: Statistics of shape via principal geodesic analysis on Lie groups Year: (2003)
Ref_id:b14 Title: Learning with a wasserstein loss Year: (2015)
Ref_id:b15 Title: Learning generative models with sinkhorn divergences Year: (2018)
Ref_id:b16 Title: Approximation of Riemannian distances and applications to distance-based learning on manifolds Year: (1904)
Ref_id:b17 Title: Riemannian proximal gradient methods Year: (2022)
Ref_id:b18 Title: Principal component analysis for Riemannian manifolds, with an application to triangular shape spaces Year: (2006)
Ref_id:b19 Title: Intrinsic shape analysis: Geodesic PCA for Riemannian manifolds modulo isometric Lie group actions Year: (2010)
Ref_id:b20 Title: Geometric hydrodynamics and infinite-dimensional Newton's equations Year: (2021)
Ref_id:b21 Title: A geometric study of Wasserstein spaces: Euclidean spaces Year: (2010)
Ref_id:b22 Title: MNIST handwritten digit database Year: (2010)
Ref_id:b23 Title: Gluing methods for quantitative stability of optimal transport maps Year: (2024)
Ref_id:b24 Title: Eigenvalue continuity and Gersgorin's theorem Year: (2019)
Ref_id:b25 Title: Improved SQP and SLSQP algorithms for feasible path-based process optimisation Year: (2024)
Ref_id:b26 Title: Wasserstein Riemannian geometry of Gaussian densities Year: (2018)
Ref_id:b27 Title: A convexity principle for interacting gases Year: (1997)
Ref_id:b28 Title: Geomstats: A Python package for Riemannian geometry in machine learning Year: (2020)
Ref_id:b29 Title: Geometry of matrix decompositions seen through optimal transport and information geometry Year: (2017)
Ref_id:b30 Title: Mémoire sur la théorie des déblais et des remblais Year: (1781)
Ref_id:b31 Title: The geometry of dissipative evolution equations: the porous medium equation Year: (2001)
Ref_id:b32 Title: On search directions for minimization algorithms Year: (1973)
Ref_id:b33 Title: Pointnet: Deep learning on point sets for 3d classification and segmentation Year: (2017)
Ref_id:b34 Title: Applied functional data analysis: methods and case studies Year: (2002)
Ref_id:b35 Title: Landscape pictures dataset Year: (2020)
Ref_id:b36 Title: On inequalities for moments and the covariance of monotone functions Year: (2014)
Ref_id:b37 Title: Principal geodesic analysis for probability measures under the optimal transport metric Year: (2015)
Ref_id:b38 Title: The differential of the exponential map, Jacobi fields and exact principal geodesic analysis Year: (2010)
Ref_id:b39 Title: Optimization over geodesics for exact principal geodesic analysis Year: (2014)
Ref_id:b40 Title: Wasserstein geometry of Gaussian measures Year: (2011)
Ref_id:b41 Title: Riemannian and stratified geometries on covariance and correlation matrices Year: (2022)
Ref_id:b42 Title: Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python Year: (2020)
Ref_id:b43 Title: A linear optimal transportation framework for quantifying and visualizing variations in sets of images Year: (2013)
Ref_id:b44 Title: 3d shapenets: A deep representation for volumetric shapes Year: (2015)
Ref_id:b45 Title: Convex analysis in general vector spaces Year: (2002)
Ref_id:b46 Title: Inexact Riemannian gradient descent method for nonconvex optimization Year: (2024)
