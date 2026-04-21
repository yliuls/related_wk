Title: Projective Equivariant Networks via Second-order Fundamental Differential Invariants
Abstract: Equivariant networks enhance model efficiency and generalization by embedding symmetry priors into their architectures. However, most existing methods, primarily based on group convolutions and steerable convolutions, face significant limitations when dealing with complex transformation groups, particularly the projective group, which plays a crucial role in vision. In this work, we tackle the challenge by constructing projective equivariant networks based on differential invariants. Using the moving frame method with a carefully selected cross section tailored for multi-dimensional functions, we derive a complete and concise set of second-order fundamental differential invariants of the projective group. We provide a rigorous analysis of the properties and transformation relationships of their underlying components, yielding a further simplified and unified set of fundamental differential invariants, which facilitates both theoretical analysis and practical applications. Building on this foundation, we develop PDINet, the first framework for deep projective equivariant networks, achieving full projective equivariance without discretizing or sampling the group. Empirical results on the projectively transformed STL-10 and Imagenette datasets show that PDINet achieves improvements of 11.39% and 5.66% in accuracy over the respective standard baselines under out-of-distribution settings, demonstrating its strong generalization to complex geometric transformations.

Section: Introduction
Incorporating symmetry as an inductive bias into neural networks has emerged as a powerful approach to enhance model efficiency and generalization. Convolutional neural networks (CNNs) [Krizhevsky et al., 2012, Simonyan and Zisserman, 2015, He et al., 2016, Chen et al., 2017], which are among the most widely used architectures in deep learning, owe much of their success to the inherent translational equivariance. Building on this idea, Cohen and Welling [2016a] proposed Group Equivariant CNNs (G-CNNs), which generalize equivariance to broader transformations like rotations and reflections. Equivariant networks achieve symmetry incorporation by constructing network layers whose outputs transform in a predictable pattern under group actions applied to the inputs.
The development of equivariant networks began with G-CNNs, whose feature map can be seen as a function on a group. Although G-CNNs have proven effective in various tasks [Worrall and Brostow, 2018, Esteves et al., 2019, Lafarge et al., 2021, Shamsolmoali et al., 2021], they are less suited 💡 Figure 1: An illustration of a projective transformation: when a planar object is projected onto another plane, a projective transformation occurs.
to continuous groups, as handling such groups typically requires group sampling or discretization, which introduces approximation errors and computational complexity. Then, steerable CNNs [Cohen andWelling, 2016b, Weiler andCesa, 2019] were proposed to overcome these limitations by viewing features as fields that transform according to specified group representations. In this framework, G-CNNs can be interpreted as a special case where the group representation is chosen to be the regular representation. Steerable CNNs are capable of handling continuous groups such as SO(2) and SO(3) directly, thereby significantly broadening the scope of equivariant networks to include real-world symmetries beyond discrete groups [Weiler et al., 2018, Wang et al., 2020, Wang and Walters, 2022]. However, for more complex non-compact Lie groups such as the projective group, deriving closedform steerable basis filters becomes intractable, inherently limiting the applicability of steerable CNNs. To address this, MacDonald et al. [2022] enabled group convolutions over finite-dimensional Lie groups by computing the integral on the Lie algebra, thus introducing a projective equivariant model, homConv. However, this method still relies on group sampling, which results in exponential memory growth with increasing network depth, thereby hindering scalability to deeper architectures. Mironenco and Forré [2024] improved sampling efficiency via group decompositions, but focused solely on affine subgroups like R n ⋊ GL + (n, R) and R n ⋊ SL(n, R), without addressing more complex groups such as the projective group. Recently, Li et al. [2024Li et al. [ , 2025] ] proposed InvarLayer and steerable EquivarLayer that construct affine equivariant networks based on invariants, enabling closed-form and sampling-free affine equivariance. However, these works remain specialized to the affine group and do not yet generalize to more complex non-compact groups like the projective group.
Actually, projective transformations play a fundamental role in computer vision [Mohr and Triggs, 1996, Birchfield, 1998, Hartley and Zisserman, 2003], as they capture the relationships between objects and their images under perspective projections (see Figure 1). Achieving equivariance on projective transformations is especially critical in practical applications such as mobile robot navigation, 3D scene analysis, and camera pose estimation, where accurately handling perspective effects and viewpoint changes can significantly enhance model robustness and accuracy [Lee et al., 2000, Hartley and Zisserman, 2003, Mur-Artal et al., 2015, Schönberger et al., 2016]. Early on, Suk and Flusser [2004] proposed a projective invariant feature extraction method based on projective moment invariants. Nevertheless, the invariants are formulated as infinite series of moment products, leading to significant computational overhead and intractable error analysis in practical implementations. To overcome this limitation, Li et al. [2018] proposed an alternative framework that constructs projective invariants using finite combinations of weighted moments, where the weights are derived from relative projective differential invariants. Note that moment-based projective invariants are essentially global image descriptors, which makes them inappropriate for constructing equivariant operators that act locally on feature fields for capturing fine-grained spatial patterns. Instead, differential invariants inherently hold the property of acting locally at each spatial position, which makes them a natural foundation for building equivariant operators [Sangalli et al., 2022, 2023, Li et al., 2024, 2025]. While Olver [2023] has proposed a systematic framework for the computation of projective differential invariants via the moving frame method, it requires at least third-order derivatives because the projective action is not free at second order for scalar functions. Besides, they only consider single-channel cases, and the extension to multi-dimensional cases needs more complex expressions involving high-order derivatives, which limits their practicality in computation and applications on color images.
In this work, we construct projective equivariant networks based on differential invariants, achieving full projective equivariance without relying on group discretization or sampling. This overcomes the depth limitations of homConv [MacDonald et al., 2022] and enables effective scaling to deeper architectures. A core challenge lies in deriving concise and practical projective differential invariants. To support color images and multi-channel intermediate features in modern neural networks, we focus on the differential invariants for multi-dimensional functions. In this case, the projective group acts freely on the second-order jet space, allowing us to derive a complete set of second-order fundamental differential invariants using the moving frame method [Olver, 2015], which can express any secondorder invariant of the projective group. However, the choice of cross section used to define the moving frame significantly affects the form of the resulting invariants. While a direct extension of the cross section in [Olver, 2023] to the multi-dimensional case is theoretically valid, it leads to prohibitively long expressions with hundreds of terms, rendering them impractical. Instead, we propose a new cross section tailored to the multi-dimensional structure, which involves up to second-order derivatives, yielding a much more concise set of fundamental differential invariants. Further analysis reveals that these fundamental invariants are composed of a set of simpler components. By exploring the algebraic properties and transformation relationships of these components, we further simplify invariants into a unified set of fundamental invariants, facilitating both practical use and theoretical analysis. Based upon these simplified invariants, we design learnable equivariant operators by combining them with parameterized multi-layer perceptrons (MLPs), and embed the operators into standard neural network backbones to build PDINet, the first framework for deep projective equivariant networks free from group sampling. Empirical evaluations under challenging out-of-distribution settings demonstrate the strong generalization ability of our model to complex geometric transformations.
We summarize our main contributions as follows:
• We employ the moving frame method to derive a complete set of second-order fundamental differential invariants of the projective group for multi-dimensional functions, enabling support for color images and multi-channel features.
• We conduct an in-depth analysis of the algebraic structure and transformation properties of these invariants, resulting in a further simplified and unified set of fundamental invariants that facilitate both theoretical understanding and practical computation.
• We develop PDINet based on second-order projective differential invariants. It is the first time that deep networks achieve full projective equivariance without relying on group discretization or sampling, thus allowing effective scaling to deeper architectures.
• Numerical experiments on projectively deformed STL-10 and Imagenettefoot_0 under out-ofdistribution settings demonstrate the effectiveness of our model, with improvements of 11.39% and 5.66% over the standard baseline results, showcasing its strong generalization capability under complex geometric transformations.foot_1 2 Method
this section cite: ['b23', 'b53', 'b13', 'b2', 'b60', 'b8', 'b24', 'b49', 'b58', 'b59', 'b57', 'b56', 'b35', 'b28', 'b29', 'b37', 'b0', 'b12', 'b26', 'b12', 'b39', 'b48', 'b55', 'b27', 'b46', 'b28', 'b34', 'b42', 'b43']

Section: Basic concepts and notations
To begin with, we introduce some basic concepts and notations necessary for our formulation. An image can be viewed as a continuous function u(x, y) defined on a 2D plane. For example, an RGB image corresponds to a three-dimensional function. Likewise, intermediate features in neural networks can also be interpreted as functions, and each layer can be seen as an operator that maps one function to another.
A central concept in this work is equivariance. If the output of an operator undergoes a corresponding transformation when the input is transformed, it is referred to as equivariance. The formal definition is as follows:
Definition 1 An operator ψ : F 1 → F 2 is said to be equivariant with respect to a group G if
g • ψ(u) = ψ(g • u), ∀g ∈ G, u ∈ F 1 ,(1)
where F 1 and F 2 are the input and output function spaces, respectively.
Let X denote the domain, U = R n be the range of a function, and
U (d) = U × U 1 × • • • × U d
be the derivative space up to order d. A group action g • x on the domain naturally induces an action on functions, defined as (g • u)(x) = u(g -1 • x), which models how geometric transformations deform images. This action further extends to derivatives through prolongation to the jet space X × U (d) . For example, the first-order prolongation of the action on X × U (1) can be expressed as: (x, u(x), ∇u(x)) → (x, ũ(x), ∇ũ(x)), where x ≜ g • x and ũ ≜ g • u, with ũ(x) = u(x).
A differential invariant is a quantity that remains unchanged under the prolonged group action. The definition is given below.
Definition 2 Given a group G acting on X, a d-th order differential invariant is a function I :
X × U (d) → R such that I(g • (x, u (d) )) = I(x, u (d) ), ∀g ∈ G, (x, u (d) ) ∈ X × U (d) ,(2)
where g • (x, u (d) ) denotes the prolonged group action on the jet space X × U (d) .
The definition can be extended to the multi-dimensional case. Specifically, we call I = (I 1 , . . . , I k ) ⊤ an k-dimensional differential invariant. In addition, we define relative differential invariants, which may transform with a weight function under the group action:
R(g • (x, u (d) )) = w(g, x) • R(x, u (d) ),(3)
where w(g, x) is a scalar weight depending on the group element g and the point x. Notably, differential invariants are closely tied to equivariance, as a (multi-dimensional) differential invariant
I yields an equivariant operator Î(u)(x) ≜ I(x, u (d) ) satisfying Î(g • u) = g • Î(u).
In this work, we focus on constructing such differential invariants and using them to build projective equivariant operators for neural networks.
this section cite: []

Section: Method of moving frames
The method of moving frames is a powerful technique for deriving differential invariants [Olver, 2003[Olver, , 2015]]. We begin with the definition of a moving frame.
Definition 3 [Olver, 2015] Let G be a Lie group acting on a manifold M. A moving frame is a map
η : M → G such that η(g • z) = η(z) • g -1 , g ∈ G, z ∈ M.(4)
Given a moving frame, the invariantization of a function F : M → R is defined as
ι(F )(z) ≜ F (η(z) • z),(5)
which converts an arbitrary function F into a group-invariant function satisfying ι(F
)(g•z) = ι(F )(z).
More generally, we can define an invariant as I(g • z) = I(z), g ∈ G, z ∈ M. In our context, the manifold of interest is the jet space M = X × U (n) and we focus on differential invariants.
A necessary and sufficient condition for the existence of a moving frame is that the group G acts freely and regularly on the manifold M. Under this condition, a moving frame can be constructed via a cross section, as described below:
Theorem 4 [Olver, 2015] Let G be a r-dimensional Lie group acting freely and regularly on a m-dimensional manifold M. Given local coordinates z = (z 1 , . . . , z m ) on M, let K be a cross section of the form K = {z 1 = c 1 , z 2 = c 2 , . . . , z r = c r } ⊂ M, where c i are constants. Then for z ∈ M, there exists a unique g ∈ G such that g • z ∈ K. Defining η(z) = g, namely η(z) • z ∈ K, yields a map η : M → G, which is a moving frame.
Here, the group action is said to be free if for any g ∈ G, g • z = z implies g = e, where e is the identity element of the group. Usually, the group action can be made free by increasing the order of the jet space. The action is regular if the orbits form a regular foliation, which is typically satisfied in common groups.
With a moving frame obtained from Theorem 4, we can construct a complete set of fundamental invariants, meaning any invariant can be expressed as a combination of these fundamental invariants.
Theorem 5 [Olver, 2015] Let η : M → G be a moving frame from Theorem 4 and define w(g, z) ≜ g • z. Then
w(η(z), z) = (c 1 , c 2 , . . . , c r , w r+1 (η(z), z), . . . , w m (η(z), z)),(6)
where I 1 (z) ≜ w r+1 (η(z), z), . . ., I m-r (z) ≜ w m (η(z), z) constitute a complete system of functionally independent invariants, called fundamental invariants.
This theorem provides a method to construct fundamental invariants via the moving frame and indicates that the number of fundamental invariants is m -r. In the following sections, we will leverage these results to derive projective differential invariants.
this section cite: ['b41', 'b42', 'b42', 'b42', 'b42']

Section: Projective transformation
Projective transformations are ubiquitous in the visual world as two different views of the same planar object can be related by a 2D projective transformation. A standard projective group action is described by the projective special linear group PSL(3, R) acting on the 2D projective plane RP 2 , which can be interpreted as the set of equivalence classes of points (x, y, p) ∼ (cx, cy, cp) for any c ̸ = 0. Points in RP 2 with p ̸ = 0 can be represented in inhomogeneous coordinates as (x, y), corresponding to the homogeneous coordinate (x, y, 1). Thus, the action of a projective transformation on 2D coordinates can be written as
x = αx + βy + γ ρx + σy + τ , ỹ = λx + µy + ν ρx + σy + τ ,(7)
where the transformation is parameterized by the coefficient matrix
P = α β γ λ µ ν ρ σ τ .(8)
Since the transformation is defined up to a nonzero scaling factor, we can normalize by requiring the determinant of P to be 1, i.e., ∆ = det(P) = 1. Thus, there are 8 independent degrees of freedom. The transformation reduces to an affine transformation when ρ = σ = 0, while a pure projective transformation, characterized by ρ 2 + σ 2 ̸ = 0, exhibits nonlinear behavior. Thus, projective transformations represent a more general and complex class of geometric transformations.
For an n-dimensional function u(x, y), the projective transformation of coordinates induces a natural action on the function, ũ(x, ỹ) = u(x, y), which can be further prolonged to its derivatives. We denote the derivatives of the i-th component function u [i] as
u [i] jk ≜ D j x D k y u [i] ,(9)
where D x and D y are the differentiation operators with respect to x and y, respectively. Under a projective transformation, these derivatives transform as
u [i] jk → ũ[i] jk = D j xD k ỹ ũ[i] ,(10)
with the transformed differential operators given by
D x = ρx + σy + τ ∆ (((µρ -λσ)x + µτ -νσ)D x + ((µρ -λσ)y -λτ + νρ)D y ) ,(11)
D ỹ = ρx + σy + τ ∆ (((ασ -βρ)x -βτ + γσ)D x + ((ασ -βρ)y + ατ -γρ)D y ) .(12)
this section cite: []

Section: Projective differential invariants of multi-dimensional functions
The projective group action is not free on the second-order jet space for scalar functions, requiring prolongation to the third-order jet space to achieve freeness. This leads to complex formulations [Olver, 2023], which may limit the practicality of the resulting invariants due to their complexity and computational cost. Moreover, in practice, third-order derivatives are harder to estimate reliably from data than lower-order ones. In this work, we focus on multi-dimensional functions, which naturally align with applications such as color image processing. In this setting, the group action is free on the second-order jet space, allowing the existence of second-order differential invariants. Using the method of moving frames, we can derive these invariants, where the choice of cross section significantly influences the simplicity of the resulting expressions. Although the cross section proposed by Olver [2023] can be extended to the multi-dimensional case, the resulting invariants tend to be lengthy, typically involving hundreds of terms, which makes them less practical. Instead, we propose an alternative cross section that leverages multiple dimensions while relying only on derivatives up to second order, yielding invariants with significantly more concise and tractable forms.
Specifically, we choose the following cross section:
K = {x = y = 0, u [1] x = 1, u [1] y = 0, u [2] x = 0, u [2] y = 1, u [1] xx = u [1] xy = 0}.(13)
Hereafter, we use the standard shorthand notation for partial derivatives, e.g., u x , u y , u xx , u xy , u yy . The above cross section defines 8 normalization equations, which, together with the constraint ∆ = 1, determine all group parameters, thereby establishing the moving frame η. The detailed derivation of the moving frame is provided in the Appendix.
With the moving frame η constructed, we can then apply the invariantization process according to Theorem 5 to obtain a complete set of fundamental differential invariants. For the coordinates involved in the cross section, we have ι(x) = 0, ι(y) = 0, ι(u [1]  x ) = 1, ι(u [1]  y ) = 0, ι(u [2]  x ) = 0, ι(u [2]  y ) = 1, ι(u [1]  xx ) = 0, ι(u [1]  xy ) = 0. The remaining coordinates of the second-order jet space yield the following differential invariants: ι(u [1]  yy ) =
T 111 J 2 12 , (14
) ι(u [2] xx ) = T 222 J 2 12 , (15
) ι(u [2] xy ) = - T 212 + 2T 122 2J 2 12 ,(16)
ι(u [2] yy ) = T 121 + 2T 112 J 2 12 ,(17)
ι(u [i]  x ) = -
J 2i J 12 , 3 ≤ i ≤ n,(18)
ι(u [i] y ) = J 1i J 12 , 3 ≤ i ≤ n,(19)
ι(u [i] xx ) = J 12 T 2i2 + J 2i T 212 J 3 12 , 3 ≤ i ≤ n,(20)
ι(u [i] xy ) = - 2J 12 T 1i2 + 2J 12 T 21i + 3J 2i T 112 J 3 12 , 3 ≤ i ≤ n,(21)
ι(u [i] yy ) = J 12 T 1i1 + 2J 1i T 112 J 3 12 , 3 ≤ i ≤ n,(22)
where J ij and T ijk are two key quantities defined as:
J ij ≜ u [i] x u [j] y -u [j] x u [i] y ,(23)
T ijk ≜ u [i]  xx u [j]  y u [k]  y + u [i]  yy u [j]  x u [k]  x -u [i]  xy (u [j]  x u [k]  y + u [k]  x u [j] y ),
satisfying J ii = 0, J ij = -J ji , and T ijk = T kji .
The invariants ( 14)-( 22), together with the obvious zeroth-order invariants
S 0 = {u [i] | 1 ≤ i ≤ n},
form a complete set of second-order fundamental differential invariants of the projective group. Compared to projective invariants for scalar functions [Olver, 2023], our results involve up to second-order derivatives and are expressed in a more concise form.
this section cite: ['b43', 'b43', 'b43']

Section: Fundamental components of projective differential invariants
In the previous subsection, we have derived a complete set of second-order fundamental differential invariants. While relatively concise, their expressions are asymmetric and depend on the specific choice of the first two dimensions used in the cross section. To obtain a simpler, more unified, and elegant formulation, we conduct a deeper analysis of the fundamental components of these invariants. This enables us to further simplify their structure while preserving completeness.
Note that the numerators and denominators in ( 14)-( 22) are all relative invariants. Thus, we focus on the properties of these relative invariants, as absolute invariants can be obtained by taking the ratio of two relative invariants with the same weight. Moreover, since the expressions are built from the basic quantities J ij and T ijk , we will delve into their transformation properties and algebraic relationships.
We first present three classes of simplified relative differential invariants of the projective group.
Theorem 6 Let W = (ρx+σy+τ ) 3 ∆ . Then the following quantities are relative differential invariants of the projective group:
• For i ̸ = j, J ij is a relative differential invariant of weight W .
• For 1 ≤ i ≤ n, T iii is a relative differential invariant of weight W 2 .
• For
1 ≤ i, j ≤ n, T iji + 2T iij is a relative differential invariant of weight W 2 .
These relative invariants are not functionally independent; rather, they can be transformed into one another. Given that there are 6n -6 second-order fundamental differential invariants according to Section 2.4, we expect a complete and independent set of relative invariants to contain 6n -5 elements. To this end, we investigate the transformation rules among the relative invariants and aim to identify a minimal generating set sufficient to express all fundamental differential invariants. We start with the transformation properties of J ij .
Theorem 7 For any indices 1 ≤ i 1 , i 2 , i 3 , i 4 ≤ n, the following equation holds:
J i1i2 • J i3i4 + J i1i3 • J i4i2 + J i1i4 • J i2i3 = 0.(25)
This implies that for any four distinct indices i 1 , i 2 , i 3 , i 4 , the six pairwise combinations of J ij are dependent such that once any five are known, the remaining one can be determined. Based on Theorem 7, we can construct a subset of {J ij | i ̸ = j} that is sufficient to express all J ij .
Theorem 8 Define the following sets of relative invariants:
S 1 ≜ {J 12 , J 23 , . . . , J n-1,n },(26)
S 2 ≜ {J 13 , J 24 , . . . , J n-2,n }.(27)
Then S 1 ∪ S 2 is a generating set for the collection {J ij | i ̸ = j}, meaning that any J ij can be expressed as a functional combination of these elements.
According to Theorem 7, J i,i+3 can be written in terms of J i,i+1 , J i+1,i+2 , J i+2,i+3 , J i,i+2 , and J i+1,i+3 , all of which belong to S 1 ∪ S 2 . By induction, any J ij can thus be recovered from the generating set. A complete and rigorous proof is provided in the Appendix.
Before establishing the transformation relationships for T ijk , we provide a more compact representation of J ij and T ijk to clarify their structural relationships: where g i ≜ u
J ij = g ⊤ i Qg j ,(28)
T ijk = g ⊤ i Q ⊤ H j Qg k ,(29)
[i]
x , u
[i] y ⊤ is the gradient of the i-th component function, H j is the Hessian matrix of the j-th component function, and Q is a fixed orthogonal matrix defined as:
Q ≜ 0 1 -1 0 .
Notably, {J ij } can be used to establish the transformation relationships between the gradients.
this section cite: []

Section: Lemma 9
For any distinct indices i, j, k, the gradient g k can be expressed in terms of g i and g j as:
g k = J kj J ij g i + J ki J ji g j .(30)
Using Lemma 9 and the form of T ijk in (29), we derive the transformation rule for T ijk given {J ij }.
Theorem 10 Let i ̸ = k, and let i ′ , j, k ′ be arbitrary indices. Then T i ′ jk ′ can be expressed as:
T i ′ jk ′ = J i ′ k J k ′ k J 2 ik T iji - J i ′ k J k ′ i + J i ′ i J k ′ k J 2 ik T ijk + J i ′ i J k ′ i J 2 ik T kjk .(31)
This result implies that, for a fixed j, the triplet {T iji , T ijk , T kjk } serves as a generating set for
{T i ′ jk ′ | 1 ≤ i ′ , k ′ ≤ n}, provided {J ij } is known.
With the transformation relationships for J ij and T ijk established, we can now construct a minimal set of relative invariants that suffices to express all the relative invariants in Theorem 6.
Theorem 11 Define the following sets of relative invariants:
S 3 ≜ {T 111 , T 222 , . . . , T nnn },(32)
S 4 ≜ {T 121 + 2T 112 , T 131 + 2T 113 , . . . , T n-1,n,n-1 + 2T n-1,n-1,n },(33)
S 5 ≜ {T 212 + 2T 221 , T 323 + 2T 332 , . . . , T n,n-1,n + 2T n,n,n-1 }.(34)
Then S 1 ∪ S 2 ∪ S 3 ∪ S 4 ∪ S 5 can express all the relative invariants in Theorem 6.
It can be shown (see the Appendix) that the fundamental differential invariants ( 14)-( 22) can be expressed via the relative invariants in Theorem 6. Therefore, we arrive at the following conclusion:
Theorem 12 The union S ≜ S 0 ∪ S 1 ∪ S 2 ∪ S 3 ∪ S 4 ∪ S 5
forms a complete set of relative differential invariants to express all second-order differential invariants of the projective group.
This set contains exactly 6n -5 elements, matching the expected minimal number needed to express all second-order fundamental differential invariants. Compared to the invariants derived in Section 2.4, the current formulation is further simplified and independent of the specific choice of the first two dimensions in the cross section, exhibiting a more unified structure. Moreover, while the original invariants involve polynomials of degree up to five, the present set only contains at most cubic expressions, resulting in lower computational complexity.
this section cite: []

Section: Projective equivariant networks
As discussed before, with a set of relative invariants obtained, we can convert them into absolute invariants by dividing each by another relative invariant with the same weight. We apply this procedure to relative invariants in S to construct a complete set of fundamental differential invariants.
Specifically, we select R 0 = 1 n (J 12 + J 23 + . . . + J n1 ) as the denominator, which is a relative invariant of weight W . We keep the elements in S 0 unchanged, divide the elements in S 1 ∪ S 2 by R 0 , and divide those in S 3 ∪ S 4 ∪ S 5 by R 2 0 . This yields a set of differential invariants sufficient to express all second-order fundamental differential invariants. In fact, this set with 6n -5 invariants may contain one redundant element, but completeness is of greater concern. To avoid division by zero, we add a positive constant ϵ to the denominator during division, enhancing numerical stability.
Let I = (I 1 , . . . , I N ) ⊤ denote the set of differential invariants we obtained, which naturally induces an equivariant operator Î. Theoretically, the invariants I 1 , . . . , I N are sufficient to express all second-order differential invariants. In practice, we leverage the expressive power of neural networks to combine I 1 , . . . , I N using a two-layer MLP to produce the output [Li et al., 2024[Li et al., , 2025]]. This leads to a learnable equivariant operator:
u out = h θ • Î(u in ),(35)
where h θ is an MLP parameterized by θ. By integrating this operator into standard network architectures, we can build projective equivariant models. We refer to the resulting model as the Projective Differential Invariant Network (PDINet), as illustrated in Figure 2.
this section cite: ['b28', 'b29']

Section: Experiments
For empirical evaluation, we conduct image classification tasks under out-of-distribution settings, where models are trained on the original dataset and tested on images deformed by projective transformations. We adopt ResNet-18 [He et al., 2016] as the backbone and replace its convolutional layers with our equivariant operators defined in (35) to construct a projective equivariant network, PDINet. As the main counterpart, we consider homConv, a projective equivariant model proposed by MacDonald et al. [2022]. To ensure a fair comparison, we attempted to implement homConv using the same backbone. However, homConv relies on group sampling, which leads to exponential memory growth with network depth, resulting in out-of-memory (OOM) issues. Therefore, we follow the original network configuration of MacDonald et al. [2022] and reduce the number of samples to avoid OOM. In addition, we also include ResNet-18 trained with projective data augmentation (DA) as a reference baseline.
this section cite: ['b13', 'b34']

Section: Proj-STL-10
STL-10 [Coates et al., 2011] is a dataset containing 5000 training images and 8000 test images. Each image has a resolution of 96 × 96 with RGB channels. We apply random projective transformations to the test set to generate the Proj-STL-10 dataset. Models are trained on the original STL-10 dataset (or with projective data augmentation for the DA baseline) and evaluated on Proj-STL-10, forming a challenging out-of-distribution setting that assesses the model's ability to generalize beyond the training distribution.
this section cite: ['b3']

Section: Proj-Imagenette
Imagenette is a ten-class subset of the ImageNet dataset [Deng et al., 2009], consisting of 9469 training images and 3925 test images. All images are adapted to a uniform resolution of 256 × 256 for model input. We apply random projective transformations to the test set to generate the Proj-Imagenette dataset, while keeping the training set unchanged. This setup simulates an out-of-distribution scenario and evaluates the model's ability to generalize to geometric transformations.
this section cite: ['b6']

Section: Conclusion
In this work, we propose PDINet, a framework for projective equivariant networks, based on second-order differential invariants of the projective group. Our method overcomes the exponential memory growth encountered by homConv [MacDonald et al., 2022], enabling effective scaling to deeper networks. Leveraging the moving frame method and a carefully chosen cross section tailored to multi-dimensional functions, we derive a complete and concise set of second-order projective fundamental differential invariants. Further analysis reveals transformation relationships among projective invariants, allowing us to obtain a unified and simplified formulation that enhances both theoretical clarity and computational efficiency. Building upon these invariants, we design a learnable projective equivariant operator that can be seamlessly integrated into various network architectures. It is the first time to achieve full projective equivariance in deep networks without group sampling or discretization. Experiments under out-of-distribution settings demonstrate the strong generalization ability of our model. With the prevalence and significance of projective transformations in vision, PDINet holds promising potential for broader applications in computer vision.
One limitation of our approach is that the second-order invariants we derive vanish in the onedimensional case, preventing the direct application of PDINet to grayscale images. In addition, this work focuses on group actions on scalar fields and does not yet cover more general cases involving arbitrary group representations, which we consider a valuable direction for future research.
NeurIPS Paper Checklist 1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The claims in the abstract and introduction strictly follow the paper's contributions and scope.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: ['b34']

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss the limitations of the work in Section 4.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: All theorems in Section 2 are presented with the full set of assumptions. We also provide the complete proofs in the Appendix.
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
Answer: [Yes] Justification: We provide all the information for experimental reproduction in Section 3 and the Appendix.
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
Answer: [No] Justification: We will make the data and code publicly available upon acceptance.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so ?No? is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: We provide the experimental setup and implementation details in Section 3 and the Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: Error bars are reported for all results in Section 3.
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
Answer: [Yes] Justification: We provide sufficient information on the computer resources in the Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: Our work conforms with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] Justification: Our paper focuses on the research of equivariant deep learning algorithms, and we do not foresee immediate positive or negative societal outcomes.
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
Answer: [NA] Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: For the code and dataset used in the paper, we respect the license and terms of use, and cite the relevant papers properly.
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
Answer: [NA]
Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16.
Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: A Related work A.1 Differential invariants
Differential invariants are quantities involving derivatives that remain unchanged under the action of a given transformation group [Olver, 1993]. They have proven useful in various image analysis applications [Mundy and Zisserman, 1992, Olver et al., 1999, Hartley and Zisserman, 2003, Li et al., 2018]. Leveraging the connection between differential invariants and symmetric partial differential equations (PDEs) [Olver, 1993], Liu et al. [2010Liu et al. [ , 2013] ] constructed learnable PDEs as linear combinations of differential invariants, achieving shift and rotation equivariance. This connection has also been explored for symmetry-informed discovery of governing PDEs [Hu et al., 2025d]. The method of moving frames offers a systematic framework for deriving differential invariants given a transformation group [Fels and Olver, 1999, Olver, 2003, 2015]. Based on this approach, Olver [2023] provided a characterization of differential invariants of the projective group. However, since the projective group does not act freely on the second-order jet space of scalar functions, Olver [2023] prolonged the group action to the third-order jet space, resulting in complex expressions that are computationally expensive and difficult to estimate from data due to the involvement of higher-order derivatives. Differently, our work focuses on multi-dimensional functions, where the projective group acts freely on the second-order jet space. This allows us to derive a complete and concise set of second-order fundamental differential invariants, which we further simplify through structural analysis.
this section cite: ['b40', 'b38', 'b12', 'b27', 'b40', 'b31', 'b32', 'b41']

Section: A.2 Equivariant networks
Early developments in equivariant networks primarily focused on designing architectures that are equivariant to discrete groups, such as cyclic or dihedral groups. A representative approach is the G-CNN [Cohen and Welling, 2016a] framework, which models feature maps as functions defined on a group and implements convolution-like operations via discrete permutations over the input domain. Building upon this viewpoint, subsequent works [Shen et al., 2020, Romero and Cordonnier, 2021, He et al., 2021] developed a broader range of group-equivariant operators beyond standard group convolutions. To achieve equivariance with respect to continuous groups, steerable CNNs [Cohen andWelling, 2016b, Weiler andCesa, 2019] were proposed, which treat feature maps as fields transforming according to specified group representations. This framework enables principled handling of Euclidean groups in 2D and 3D settings [Weiler and Cesa, 2019, Fuchs et al., 2020, Wang et al., 2020, Zhao et al., 2022, Shen et al., 2022, Liao and Smidt, 2022, Hu et al., 2024, 2025a,b], by explicitly encoding the group action on the feature spaces.
Symmetries are ubiquitous and can be leveraged in model design, either from prior knowledge or discovered from data [Yang et al., 2023, 2024, Hu et al., 2025c,e]. However, generalizing equivariant architectures to high-dimensional Lie groups, such as the projective group, poses significant challenges. For such groups, deriving explicit steerable basis filters is often intractable, limiting the applicability of steerable CNNs, while G-CNN based methods require group discretization or sampling, introducing scalability issues. Bökman et al. [2023] explored equivariance in a projective sense and achieved equivariant models with respect to projective representations of certain simple groups, rather than the full projective group. Another related effort is Möbius Convolution (MC) [Mitchel et al., 2022], which achieves equivariance to spherical Möbius transformations for geometry and spherical image processing tasks. While both MC and our method aim to build equivariant models under certain projective group actions, MC operates on the Riemann sphere under the complex-valued SL(2, C), whereas our work considers planar projective transformations governed by SL(3, R), corresponding to the commonly studied projective transformations in vision. Additionally, EMLP [Finzi et al., 2021] and EKAN [Hu et al., 2025f] incorporate arbitrary matrix group equivariance into MLPs and KANs [Liu et al., 2025], respectively, but they target linear transformations on vectors or tensors and are not directly suitable for image inputs. Shen et al. [2024] combined rotation-equivariant networks with data augmentation to obtain nearly affine invariant features, without achieving full equivariance. LieConv [Finzi et al., 2020] attempts to handle Lie group equivariance by sampling from the Haar measure, but struggles to extend to complex groups due to inaccessibility to the Haar measure. To mitigate this problem, MacDonald et al. [2022] performed group convolutions by computing the integral on the Lie algebra, resulting in a projective equivariant model, homConv. Nevertheless, it suffers from exponential memory growth as network depth increases, making it impractical for deep networks. Mironenco and Forré [2024] improved sampling efficiency via group decompositions, but their method remains constrained to affine subgroups. Besides these two mainstream approaches, G-CNNs and steerable CNNs, Li et al. [2024Li et al. [ , 2025] ] resorted to another route by constructing equivariant networks based on differential invariants, achieving affine equivariance without requiring group sampling or discretization. However, their method is confined to affine groups and does not generalize to the projective group. In particular, projective differential invariants for multi-channel inputs have not yet been developed, and the SupNorm normalization technique used to construct affine invariants cannot be directly extended to the projective group. In our work, we target projective equivariance by deriving a complete set of second-order fundamental differential invariants for multi-channel inputs using a tailored moving frame construction. Through algebraic analysis, we obtain simplified projective invariants for practical use, which enable the construction of deep projective equivariant networks without relying on sampling or discretizing the group.
this section cite: ['b50', 'b45', 'b14', 'b58', 'b58', 'b11', 'b57', 'b63', 'b30', 'b16', 'b61', 'b1', 'b36', 'b10', 'b33', 'b52', 'b9', 'b35', 'b28', 'b29']

Section: B Detailed proofs
In this section, we provide the detailed derivation of the moving frame and complete proofs of Theorems 8, 11, and 12.
For completeness, we briefly comment on the results that can be verified primarily through direct computation, while omitting the detailed algebraic steps:
• Theorem 6 follows directly from the definitions of J ij and T ijk , together with the transformation rules for first-and second-order derivatives under the group action derived from ( 11)-( 12), followed by straightforward simplification.
• Theorem 7 and Lemma 9 can be verified by substituting the definition of J ij and expanding the expressions algebraically.
• Theorem 10 can be proved by rewriting 29), then applying Lemma 9 to express the gradient vectors g i ′ and g k ′ in terms of g i and g k , and simplifying the resulting expression.
T i ′ jk ′ in the compact form g ⊤ i ′ Q ⊤ H j Qg k ′ as shown in (
this section cite: []

Section: B.1 Derivation of the moving frame
We begin by setting x = ỹ = 0, yielding
γ = -αx -βy,(36)
ν = -λx -µy. (37
) Next, imposing ũ[1] x = 1, ũ[1] ỹ = 0 gives α = u [1] x (ρx + σy + τ ),(38)
β = u [1] y (ρx + σy + τ ).(39)
Similarly, enforcing ũ
] x = 0, ũ[2] ỹ = 1 leads to λ = u [2] x (ρx + σy + τ ),[2
µ = u [2] y (ρx + σy + τ ).(40)
We then apply the conditions ũ[1] xx = ũ[1] xỹ = 0, which yield the equations for σ and τ .
σ = ρ J 12 (u [1] yy u [2] x -u [1] xy u [2] y ) + u [2] y T 112 -u [1] x T 212 + 2u [2] x T 112 , (42
) τ = ρ x(u [1]
x T 212 -2u
[2]
x T 112 ) + y(u
[1] y T 212 -2u [1] y T 112 ) -u [1]
x T 212 + 2u
[2]
x T 112 .
(43)
Finally, setting ∆ = 1 determines the parameter ρ as:
ρ = u [1]
x T 212 -2u
[2]
x T 112 2J 7/3 12 .
(44)
With all the parameters now fully determined, we obtain the moving frame η.
this section cite: []

Section: II

this section cite: []

Section: B.2 Proof of Theorem 8
Proof. Define J k ≜ {J i,i+k | 1 ≤ i ≤ n -k}, k = 1, 2, . . . , n -1. To prove the theorem, it suffices to show that each J k for 1 ≤ k ≤ n -1 can be expressed in terms of elements from S 1 ∪ S 2 .
We proceed by mathematical induction on k.
Base case (k = 1, 2): This holds by definition, as J 1 = S 1 and J 2 = S 2 .
Inductive step: Assume that for all k ≤ l -1 with l ≥ 3, each set J k can be expressed in terms of elements from S 1 ∪ S 2 . We aim to show that J l can also be written using elements from S 1 ∪ S 2 .
By the induction hypothesis, the following quantities can all be expressed by elements in S 1 ∪ S 2 :
J i,i+1 , J i,i+l-1 , J i+1,i+l-1 , J i+1,i+l , J i+l-1,i+l .
For any J i,i+l ∈ J l , note that when l ≥ 3, the indices i, i + 1, i + l -1, i + l are distinct. By Theorem 7, it follows that J i,i+l can be expressed as a function of these five relative invariants. Hence, J i,i+l can be expressed by elements in S 1 ∪ S 2 .
By the principle of mathematical induction, all J k for any 1 ≤ k ≤ n -1 can be generated from S 1 ∪ S 2 , completing the proof. □
this section cite: []

Section: B.3 Proof of Theorem 11
Proof. Recall that J ij has been established in Theorem 8, and T iii is already included in S 3 . So we focus on proving that T iji + 2T iij can be expressed in terms of elements from S 1 ∪ S 2 ∪ S 3 ∪ S 4 ∪ S 5 . Define
T + k ≜ {T i,i+k,i + 2T i,i,i+k | 1 ≤ i ≤ n -k} ,(45)
T - k ≜ {T i,i-k,i + 2T i,i,i-k | k + 1 ≤ i ≤ n} . (46
)
Our goal is to show that each element in T + k and T - k for 1 ≤ k ≤ n -1 can be expressed in terms of elements from S 1 ∪ S 2 ∪ S 3 ∪ S 4 ∪ S 5 . We present the proof for T + k ; the case of T - k follows analogously due to index symmetry.
We proceed by mathematical induction on k.
Base case (k = 1): This holds by definition, as T + 1 = S 4 . Inductive step: Assume that for all k ≤ l-1 with l ≥ 2, each element in T + k can be expressed in terms of elements from S 1 ∪ S 2 ∪ S 3 ∪ S 4 ∪ S 5 . We aim to show that every element T i,i+l,i + 2T i,i,i+l ∈ T + l can also be expressed using this union. From the transformation rule in Theorem 10, we have:
T i,i+l,i + 2T i,i,i+l(47)
= (C 1 T i+l-1,i+l,i+l-1 -C 2 T i+l-1,i+l,i+l + C 3 T i+l,i+l,i+l ) + 2 (C 4 T iii -C 5 T i,i,i+l-1 ) (48) =C 1 (T i+l-1,i+l,i+l-1 + 2T i+l-1,i+l-1,i+l ) - 1 2 C 2 (2T i+l-1,i+l,i+l + T i+l,i+l-1,i+l ) + C 3 T i+l,i+l,i+l + 2C 4 T iii -C 5 (2T i,i,i+l-1 + T i,i+l-1,i ) -2C 1 T i+l-1,i+l-1,i+l + 1 2 C 2 T i+l,i+l-1,i+l + C 5 T i,i+l-1,i ,(49)
where the coefficients are defined as:
C 1 = J 2 i,i+l J 2 i+l-1,i+l , C 2 = 2J i,i+l J i,i+l-1 J 2 i+l-1,i+l , C 3 = J 2 i,i+l-1 J 2 i+l-1,i+l , C 4 = J i+l,i+l-1 J i,i+l-1 , C 5 = J i+l,i J i,i+l-1 .(50)
By the induction hypothesis, 2T i,i,i+l-1 + T i,i+l-1,i ∈ T + l-1 can be expressed using elements from S 1 ∪S 2 ∪S 3 ∪S 4 ∪S 5 . In addition, it holds that T iii ∈ S 3 , T i+l-1,i+l,i+l-1 +2T i+l-1,i+l-1,i+l ∈ S 4 , III and 2T i+l-1,i+l,i+l + T i+l,i+l-1,i+l ∈ S 5 . The only remaining terms are:
-2C 1 T i+l-1,i+l-1,i+l + 1 2 C 2 T i+l,i+l-1,i+l + C 5 T i,i+l-1,i
= -2C 1 T i+l-1,i+l-1,i+l + 1 2 C 2 T i+l,i+l-1,i+l + C 5 (C 1 T i+l-1,i+l-1,i+l-1 -C 2 T i+l-1,i+l-1,i+l + C 3 T i+l,i+l-1,i+l ) (52) =C 5 C 1 T i+l-1,i+l-1,i+l-1 ,(51)
which can be expressed in terms of S 1 ∪ S 2 ∪ S 3 ∪ S 4 ∪ S 5 as well. Hence, the full expression for
T i,i+l,i + 2T i,i,i+l is a combination of elements in S 1 ∪ S 2 ∪ S 3 ∪ S 4 ∪ S 5 .
Therefore, by the principle of mathematical induction, all elements in T + k for any 1 ≤ k ≤ n -1 can be generated from the given sets. □
this section cite: []

Section: B.4 Proof of Theorem 12
Proof. To prove Theorem 12, it suffices to show that all second-order fundamental differential invariants of the projective group, namely the quantities in ( 14)-( 22), along with the set S 0 , can be expressed in terms of the elements in S. Since S 0 ⊂ S, we may disregard S 0 in the following discussion.
Moreover, Theorem 11 has established that all relative invariants given in Theorem 6 can be generated by elements in the union S 1 ∪ S 2 ∪ S 3 ∪ S 4 ∪ S 5 ⊂ S. Therefore, it remains to prove that the expressions ( 14)-( 22) can be written in terms of the relative invariants from Theorem 6.
Among these fundamental invariants, ( 14)-( 19) can be directly expressed using the relative invariants from Theorem 6. Thus, it remains to focus on the final three expressions ( 20)-( 22). Since the denominators are already included in the set S, it suffices to consider only the numerators:
J 12 T 2i2 + J 2i T 212 , 3 ≤ i ≤ n, (54
) 2J 12 T 1i2 + J 12 T 21i + 3J 2i T 112 , 3 ≤ i ≤ n, (55
) J 12 T 1i1 + 2J 1i T 112 , 3 ≤ i ≤ n. (56
)
We now show that each of these numerators can indeed be expressed using the relative invariants in Theorem 6.
For (54), we have
J 12 T 2i2 + J 2i T 212 (57) =J 12 (T 2i2 + 2T 22i ) + J 2i (T 212 + 2T 221 ) -2J 12 T 22i -2J 2i T 221 (58) =J 12 (T 2i2 + 2T 22i ) + J 2i (T 212 + 2T 221 ) -2J 12 (C 6 T 221 + C 7 T 222 ) -2J 2i T 221 (59) =J 12 (T 2i2 + 2T 22i ) + J 2i (T 212 + 2T 221 ) -2J 12 C 7 T 222 ,(60)
where the coefficients are defined as:
C 6 = J i2 J 12 , C 7 = J i1 J 21 .(61)
For (55), we first rewrite T 1i2 as:
T 1i2 =C 8 T 1i1 + C 9 T 1ii (62
) =C 8 (T 1i1 + 2T 11i ) + 1 2 C 9 (2T 1ii + T i1i ) -2C 8 T 11i - 1 2 C 9 T i1i (63
) =C 8 (T 1i1 + 2T 11i ) + 1 2 C 9 (2T 1ii + T i1i ) -2C 8 (C 6 T 111 + C 7 T 112 ) - 1 2 C 9 (C 2 6 T 111 + 2C 6 C 7 T 112 + C 2 7 T 212 ),(64)
where the coefficients are defined as:
C 8 = J 2i J 1i C 9 = J 21 J i1 .(65)
IV Similarly, we rewrite T 21i as:
T 21i = C 6 T 211 + C 7 T 212 .(66)
Substituting these into (55), we have:
2J 12 T 1i2 + J 12 T 21i + 3J 2i T 112 (67
) =2J 12 C 8 (T 1i1 + 2T 11i ) + 1 2 C 9 (2T 1ii + T i1i ) + 2J 12 -2C 8 (C 6 T 111 + C 7 T 112 ) -1 2 C 9 (C 2 6 T 111 + 2C 6 C 7 T 112 + C 2 7 T 212 ) + J 12 (C 6 T 211 + C 7 T 212 ) + 3J 2i T 112 (68)
=2J 12 C 8 (T 1i1 + 2T 11i ) + 1 2 C 9 (2T 1ii + T i1i ) -4J 12 C 6 C 8 + J 12 C 9 C 2 6 T 111 .(69)
For (56), we have:
J 12 T 1i1 + 2J 1i T 112 (70) =J 12 (T 1i1 + 2T 11i ) + 2J 1i T 112 -2J 12 T 11i (71) =J 12 (T 1i1 + 2T 11i ) + 2J 1i T 112 -2J 12 (C 6 T 111 + C 7 T 112 ) (72) =J 12 (T 1i1 + 2T 11i ) -2J 12 C 6 T 111(73)
This completes the proof that all second-order fundamental differential invariants in ( 14)-( 22) are expressible in terms of the relative invariants in Theorem 6, and thus can be generated by S. □
this section cite: []

Section: C Implementation details of PDINet
While our theoretical foundation is developed in the continuous setting, practical applications involve discrete image data defined on grid points, where derivatives must be approximated numerically. To this end, we estimate spatial derivatives using Gaussian derivatives [Li et al., 2018, He et al., 2022, Li et al., 2024]. For example, the partial derivative with respect to x is computed via convolution as ∂f ∂x ≈ f * ∂Gσ ∂x , where G σ is a Gaussian kernel with zero mean and standard deviation σ. In our implementation, we set σ = 0.99 and use a kernel size of 9.
As mentioned in Subsection 2.6, we construct projective invariants by dividing each relative invariant by a designated relative invariant R 0 or R 2 0 , which is computed from the input image. To prevent division by zero, we add a positive constant ϵ to the denominator. We set ϵ = 1. After obtaining the invariants, we apply SupNorm normalization [Li et al., 2024[Li et al., , 2025]], which preserves equivariance and is beneficial for training stability. Then we combine the invariants using a two-layer MLP, which is implemented as a sequence of two 1 × 1 convolutions with a ReLU activation function in between.
Given a standard convolutional network architecture, we construct a projective equivariant network by replacing each convolutional layer with our equivariant operator. If a convolutional layer has stride greater than 1, we insert an average pooling layer with a kernel size equal to the stride before the second 1 × 1 convolutional layer in the equivariant operator. Additionally, we also apply the same pooling when computing R 0 to ensure resolution consistency between the numerator and denominator. Figure 3 shows the training loss curve on the STL-10 dataset, demonstrating stable optimization behavior of PDINet during training.
this section cite: ['b27', 'b15', 'b28', 'b28', 'b29']

Section: D Experimental details
All experiments are conducted on a single NVIDIA RTX 3090 GPU. Each experiment is repeated five times with independently generated test sets using random projective transformations, and we report the mean accuracy and standard deviation.
Experiments on Proj-STL-10. Models are trained on the 5000 samples of the STL-10 training set and tested on the 8000 samples of the Proj-STL-10 test set, which is generated by applying projective transformations to each sample in the STL-10 test set. Specifically, we decompose a projective transformation into an affine transformation followed by horizontal and vertical pure projections, defined as (x, ỹ) ⊤ = ( x 1+c1x , y 1+c1x ) ⊤ and (x, ỹ) ⊤ = ( x 1+c2y , y 1+c2y ) ⊤ , respectively. The affine transformation consists of random rotation between -90 • and 90 • , scaling in the range [0.9, 1.1], shear within ±4 • , and translation within [-0.1, 0.1], while the projection parameters c 1 and c 2 are uniformly sampled from [-0.0001, 0.0001].
V
All images are normalized by channel-wise mean subtraction and standard deviation division. Following [Sosnovik et al., 2019], data augmentation during training includes 12-pixel zero-padding followed by random cropping to 96 × 96, random horizontal flipping, and Cutout [DeVries and Taylor, 2017] with a single 32 × 32 hole. We train the models for 1000 epochs using SGD optimizer with Nesterov momentum of 0.9 and a batch size of 64. The initial learning rate is set to 0.1 and decayed by a factor of 0.2 at epochs 300, 400, 600, and 800. For the DA baseline, since the projective group is a complex non-compact group and the specific range or distribution of transformation parameters in test scenarios is typically unknown, we adopt a considerably wide range of geometric transformations. Specifically, we apply random rotation between -180 Experiments on Proj-Imagenette. Models are trained on the 9469 samples of the Imagenette training set and evaluated on the 3925 samples of the Proj-Imagenette test set. The Proj-Imagenette dataset is generated by applying projective transformations to each test image in the original Imagenette dataset, following the same procedure as Proj-STL-10. All images are normalized by subtracting the perchannel mean and dividing by the per-channel standard deviation. During training, data augmentation includes random resized cropping to 224 × 224 and random horizontal flipping. We train the models for 100 epochs using AdamW optimizer with a batch size of 64. The initial learning rate is set to 0.002 and decayed via a cosine annealing scheduler. The same strategy as in experiments on Proj-STL-10 is used for the DA baseline, while homConv is also trained with Adam as in [MacDonald et al., 2022]. Theoretically, the projective equivariance of our operators is rigorously guaranteed by the fundamental properties of differential invariants. In implementation, however, derivatives are estimated on discrete grids, which inevitably introduces minor equivariance errors. To quantitatively evaluate this effect, VI we follow the protocol in [MacDonald et al., 2022] and define the equivariance error as
this section cite: ['b54', 'b34', 'b34']

Section: E Additional experiments E.1 Equivariance error
Error = ∥g • ψ(u) -ψ(g • u)∥ 2 ∥g • ψ(u)∥ 2 ,
where ψ denotes the equivariant layer and g is a random projective transformation. Since the projective equivariance of PDINet is intrinsic and does not rely on training, we measure this error using a randomly initialized equivariant layer. We compute the error on the Imagenette test set, resizing images to multiple resolutions. As shown in Table 3, the equivariance error remains consistently small across all resolutions and decreases monotonically with increasing image size, which is expected due to more accurate derivative approximation at higher resolutions.
To further visualize equivariance, Figure 4 compares the features g • ψ(u) and ψ(g • u), showing that they are nearly identical and thus confirming the commutativity between the equivariant layer and projective transformations.
this section cite: []

Section: E.2 Computational complexity
Our projective equivariant layer exhibits linear growth in both time and space complexity with respect to the input size. Specifically, the computation includes estimating derivatives, computing differential invariants, and combining differential invariants, each with linear complexity. To quantify the overhead, we compare PDINet and ResNet-18 in terms of memory usage and FLOPs using torchstat, and report the results in Table 4. As shown, both models exhibit linear scaling with input resolution, highlighting the scalability of our model to higher-resolution inputs. While PDINet consumes slightly more memory than ResNet-18, it requires fewer FLOPs.
To provide an explicit runtime comparison, we further evaluate both models on the same hardware by measuring the total time required to process 1000 RGB images (224 × 224) with a batch size of 100. Each experiment is repeated five times, and the mean and standard deviation are reported in Table 5. Despite its lower theoretical FLOPs, PDINet runs slower in practice, primarily due to the highly optimized low-level implementations of ResNet-18. We expect this gap can be narrowed through further engineering and implementation optimization.
this section cite: []

Section: E.3 Application on keypoint detection
To further demonstrate the practical benefit of PDINet, we conduct an additional experiment on keypoint detection, a task that inherently involves projective distortions and thus provides a natural
VII Table 4: Memory usage and FLOPs of ResNet-18 and PDINet at different input resolutions. Memory (MB) FLOPs Input Size ResNet-18 PDINet ResNet-18 PDINet 32 × 32 0.53 0.93 3.72 × 10 7 3.65 × 10 7 64 × 64 2.10 3.73 1.49 × 10 8 1.46 × 10 8 128 × 128 8.38 14.91 5.94 × 10 8 5.84 × 10 8 256 × 256 33.50 59.63 2.38 × 10 9 2.33 × 10 9 512 × 512 134.00 238.50 9.51 × 10 9 9.34 × 10 9 setting to evaluate the effectiveness of built-in projective equivariance. We integrate PDINet as the backbone into the REKD [Lee et al., 2022] framework for keypoint detection. To reduce confounding factors, we simplify the pipeline by removing the orientation estimation branch (which depends on steerable filters) and retain only the keypoint detection component. A three-layer PDINet is used and compared against a CNN baseline with the same architecture. The number of channels is adjusted to keep the parameter counts on the same order of magnitude, with PDINet using fewer parameters overall.
We evaluate both models on the viewpoint split of the HPatches dataset, which includes 59 scenes. Each scene contains a reference image and five target images captured from different viewpoints, resulting in projective distortions between image pairs. We follow the setup of Lee et al. [2022] for data construction and loss formulation, and train each model for 20 epochs using the AdamW optimizer with a cosine learning rate schedule (initial learning rate 0.01). A downsampling pyramid with scaling factor 1.2 is applied during training, and a symmetric pyramid with scaling factor √ 2 (plus an identity branch) is used at inference, with two levels of down-sampling and up-sampling.
We report the Repeatability metric, which measures the consistency of keypoint detection under viewpoint changes. A higher value indicates better robustness. As shown in Table 6, PDINet achieves higher repeatability than the baseline while using fewer parameters, suggesting that built-in projective equivariance enhances geometric consistency in keypoint detection. This supplementary experiment complements our main results and demonstrates the broader applicability of PDINet to real-world tasks.
this section cite: ['b25', 'b25']

Section: F Discussion
While PDINet consistently outperforms the baselines in our main experiments, some misclassifications still occur. We believe these errors are largely attributable to the nature of synthetically generated data. Synthetic projective transformations involve interpolation and padding, which can introduce aliasing, distortions, and unnatural edges, especially under strong shearing, scaling or non-orthogonal rotations. Such artifacts can interfere with the model's ability to maintain equivariance and thus degrade performance. In addition, since PDINet relies on differential invariants that depend on discrete approximations of derivatives, image resolution may also affect performance. For example, PDINet performs better on Proj-Imagenette than on Proj-STL-10, may partially attributed to the higher resolution enabling more accurate derivative estimation.
this section cite: []

Section: VIII
We expect that applying PDINet to real-world data at higher resolutions with naturally occurring projective distortions would help mitigate these artifacts and better demonstrate its full potential. In practical scenarios, such as multiview settings involving planar objects, transformations between different viewpoints are well modeled by projective mappings, making PDINet naturally suited for these cases. While exact equivariance may not strictly hold for non-planar or 3D objects, the projective inductive bias still contributes to improved robustness by approximately preserving geometric structure under near projective transformations. Exploring such extensions, including more general 3D settings, is a promising direction for future work.
Another avenue for exploration is to extend our method to other transformation groups. According to the general theory of differential invariants [Olver, 1993], such invariants exist for any regular Lie group action on a smooth manifold, for example, SL(2, C) acting on the sphere, which corresponds to Möbius transformations. In principle, our framework can be adapted to these settings by deriving the appropriate differential invariants and designing corresponding equivariant architectures. While identifying explicit, concise, low-order, and numerically stable invariant forms for different groups (e.g., the Möbius group) is highly non-trivial, our experience suggests that recognizing relative invariants as atomic components can greatly facilitate the construction and simplification of a complete and practical invariant basis.
this section cite: ['b40']

Section: References
Ref_id:b0 Title: An introduction to projective geometry (for computer vision) Year: (1998)
Ref_id:b1 Title: In search of projectively equivariant networks Year: (2023)
Ref_id:b2 Title: DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs Year: (2017)
Ref_id:b3 Title: An analysis of single-layer networks in unsupervised feature learning Year: (2011)
Ref_id:b4 Title: Group equivariant convolutional networks Year: (2016)
Ref_id:b5 Title: Steerable CNNs Year: (2016)
Ref_id:b6 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b7 Title: Improved regularization of convolutional neural networks with cutout Year: (2017)
Ref_id:b8 Title: Moving coframes: II. regularization and theoretical foundations Year: (1999)
Ref_id:b9 Title: Generalizing convolutional neural networks for equivariance to Lie groups on arbitrary continuous data Year: (2020)
Ref_id:b10 Title: A practical method for constructing equivariant multilayer perceptrons for arbitrary matrix groups Year: (2021)
Ref_id:b11 Title: SE(3)-transformers: 3D rototranslation equivariant attention networks Year: (2020)
Ref_id:b12 Title: Multiple view geometry in computer vision Year: (2003)
Ref_id:b13 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b14 Title: Efficient equivariant network Year: (2021)
Ref_id:b15 Title: Neural ePDOs: Spatially adaptive equivariant partial differential operator based networks Year: (2022)
Ref_id:b16 Title: Orbitgrasp: SE(3)-equivariant grasp learning Year: (2024)
Ref_id:b17 Title: Push-grasp policy learning using equivariant models and grasp score optimization Year: (2025)
Ref_id:b18 Title: Robert Platt, and Robin Walters. 3D equivariant visuomotor policy learning via spherical projection Year: (2025)
Ref_id:b19 Title: Explicit discovery of nonlinear symmetries from dynamic data Year: (2025)
Ref_id:b20 Title: Governing equation discovery from data based on differential invariants Year: (2025)
Ref_id:b21 Title: Symmetry discovery for different data types Year: (2025)
Ref_id:b22 Title: Incorporating arbitrary matrix group equivariance into KANs Year: (2025)
Ref_id:b23 Title: Imagenet classification with deep convolutional neural networks Year: (2012)
Ref_id:b24 Title: Roto-translation equivariant convolutional networks: Application to histopathology image analysis Year: (2021)
Ref_id:b25 Title: Self-supervised equivariant learning for oriented keypoint detection Year: (2022)
Ref_id:b26 Title: Self-localization of a mobile robot without camera calibration using projective invariants Year: (2000)
Ref_id:b27 Title: Image projective invariants Year: (2018)
Ref_id:b28 Title: Affine equivariant networks based on differential invariants Year: (2024)
Ref_id:b29 Title: Affine steerable equivariant layer for canonicalization of neural networks Year: (2025)
Ref_id:b30 Title: Equiformer: Equivariant graph attention transformer for 3D atomistic graphs Year: (2022)
Ref_id:b31 Title: Learning PDEs for image restoration via optimal control Year: (2010)
Ref_id:b32 Title: Toward designing intelligent PDEs for computer vision: an optimal control approach Year: (2013)
Ref_id:b33 Title: KAN: Kolmogorov-Arnold networks Year: (2025)
Ref_id:b34 Title: Enabling equivariance for arbitrary Lie groups Year: (2022)
Ref_id:b35 Title: Lie group decompositions for equivariant neural networks Year: (2024)
Ref_id:b36 Title: Möbius convolutions for spherical CNNs Year: (2022)
Ref_id:b37 Title: Projective geometry for image analysis Year: (1996)
Ref_id:b38 Title: Geometric invariance in computer vision Year: (1992)
Ref_id:b39 Title: Orb-slam: A versatile and accurate monocular slam system Year: (2015)
Ref_id:b40 Title: Applications of Lie groups to differential equations Year: (1993)
Ref_id:b41 Title: Moving frames Year: (2003)
Ref_id:b42 Title: Modern developments in the theory and applications of moving frames Year: (2015)
Ref_id:b43 Title: Projective invariants of images Year: (2023)
Ref_id:b44 Title: Affine invariant detection: edge maps, anisotropic diffusion, and active contours Year: (1999)
Ref_id:b45 Title: Group equivariant stand-alone self-attention for vision Year: (2021)
Ref_id:b46 Title: Differential invariants for SE(2)-equivariant networks Year: (2022)
Ref_id:b47 Title: Moving frame net: SE(3)-equivariant network for volumes Year: (2023)
Ref_id:b48 Title: Pixelwise view selection for unstructured multi-view stereo Year: (2016)
Ref_id:b49 Title: Rotation equivariant feature image pyramid network for object detection in optical remote sensing imagery Year: (2021)
Ref_id:b50 Title: PDO-eConvs: Partial differential operator based equivariant convolutions Year: (2020)
Ref_id:b51 Title: PDO-s3DCNNs: Partial differential operator based steerable 3D CNNs Year: (2022)
Ref_id:b52 Title: Efficient learning of scale-adaptive nearly affine invariant networks Year: (2024)
Ref_id:b53 Title: Very deep convolutional networks for large-scale image recognition Year: (2015)
Ref_id:b54 Title: Scale-equivariant steerable networks Year: (2019)
Ref_id:b55 Title: Projective moment invariants Year: (2004)
Ref_id:b56 Title: SO(2) equivariant reinforcement learning Year: (2022)
Ref_id:b57 Title: Incorporating symmetry into deep dynamics models for improved generalization Year: (2020)
Ref_id:b58 Title: General E(2)-equivariant steerable CNNs Year: (2019)
Ref_id:b59 Title: steerable CNNs: Learning rotationally equivariant features in volumetric data Year: (2018)
Ref_id:b60 Title: Cubenet: Equivariance to 3D rotation and translation Year: (2018)
Ref_id:b61 Title: Generative adversarial symmetry discovery Year: (2023)
Ref_id:b62 Title: Latent space symmetry discovery Year: (2024)
Ref_id:b63 Title: Integrating symmetry into differentiable planning with steerable convolutions Year: (2022)
