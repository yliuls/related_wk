Title: Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization
Abstract: Solving bilevel optimization (BLO) problems to global optimality is generally intractable. A common surrogate is to compute a hyper-stationary point-a stationary point of the hyper-objective function obtained by minimizing or maximizing the upper-level objective over the lower-level solution set. Existing methods, however, either provide weak notions of stationarity or require restrictive assumptions to guarantee the smoothness of hyper-objective functions. In this paper, we eliminate these impractical assumptions and show that strong (Clarke) hyper-stationarity remains computable even when the hyper-objective is nonsmooth. Our key ingredient is a new structural property, called set smoothness, which captures the variational dependence of the lower-level solution set on the upper-level variable. We prove that this property holds for a broad class of BLO problems and ensures weak convexity (resp. concavity) of pessimistic (resp. optimistic) hyper-objective functions. Building on this foundation, we show that a zeroth-order algorithm that computes approximate Clarke hyper-stationary points with non-asymptotic convergence guarantees. To the best of our knowledge, this is the first computational guarantee for Clarke-type stationarity in nonsmooth BLO. Beyond this specific application, the set smoothness property emerges as a structural concept of independent interest, with potential to inform the analysis of broader classes of optimization and variational problems.

Section: Introduction
Bilevel optimization (BLO) models hierarchical decision-making with two agents acting sequentially [13,14]. The follower responds to the leader's decision by solving a lower-level optimization problem, while the leader seeks an optimal strategy to minimize its upper-level objective subject to this reaction. The follower's attitude plays a central role: If the follower is favorable (resp. adverse) to the leader, the resulting BLO is termed optimistic (resp. pessimistic) [13,51,36]. Formally, the optimistic and pessimistic BLO take the following forms: These formulations appear in diverse domains such as Stackelberg games [14,4,50], hyperparameter optimization [17,3,6], reinforcement learning [29,55,21], and interdiction games [36,5], among others. A standard approach to tackle such nested problems is to reformulate them into single-level problems via hyper-objective functions. Let S(x) := arg min y ′ ∈R n f (x, y ′ ) denote the follower's optimal response set. The optimistic and pessimistic hyper-objectives are then defined as φ o (x) := min y∈S(x) F (x, y), φ p (x) := max y∈S(x) F (x, y).
Solving an optimistic (resp. pessimistic) BLO is therefore equivalent to minimizing the corresponding hyper-objective φ o (resp. φ p ).
Despite the single-level reformulation, the resulting hyper-objective functions are highly nonconvex [8,31], which makes global optimization intractable. In practice, researchers therefore focus on finding stationary points rather than global minimizers, using algorithms such as implicit gradient descent [17,18,3] and fully first-order methods [32,21]. These approaches assume that the lowerlevel problem is strongly convex, ensuring a unique solution, i.e., S(x) := {y ⋆ (x)}. Under this assumption, the hyper-objective reduces to a smooth function φ(x) := F (x, y ⋆ (x)) [19]. One can then seek an ϵ-approximate hyper-stationary point satisfying ∥∇φ(x)∥ ≤ ϵ. Convergence is well understood under smoothness and uniqueness assumptions [19], but these conditions rarely hold in practice. With multiple lower-level solutions, the existing methods break down.
To move beyond the singleton lower-level solution set, Kwon et al. [32] introduced a penaltybased framework that allows multiple follower solutions. Building on this idea, Chen et al. [8] obtained a refined scheme with near-optimal convergence. However, ensuring smoothness of the induced hyper-objective still demands strong regularity: The penalized model function h σ (x, y) := σF (x, y) + f (x, y) must satisfy, uniformly in σ ∈ [0, σ], an error bound or a Polyak-Łojasiewicz (PŁ) condition in y. Such requirements are often unrealistic in practice, as F and f typically have mismatched structures. More fundamentally, the Kurdyka-Łojasiewicz (KŁ) exponent is not preserved under summation [23], so smoothness of hyper-objective functions cannot be guaranteed.
Without relying on these stringent conditions, Chen et al. [7] and Khanduri et al. [28] proposed algorithms for nonsmooth hyper-objectives; however, by their zero-respecting nature (cf. [8, Thm. 3.2]), they cannot in general approximate hyper-stationary points and thus only guarantee convergence to (approximate) Goldstein stationary points [20]-a relatively weak notion. By contrast, a separate line of work studies alternative stationarity concepts via reformulations [35,52,38,1,39]; yet these notions (e.g., KKT stationarity [37, Sec. 2.1] and penalization stationarity [54, Sec. 4.2]) are posed jointly in (x, y) and do not ensure that, for a stationary pair ( x, ȳ), the lower-level solution ȳ actually minimizes or maximizes F ( x, y) over S( x).
Given the above discussion, existing algorithms either fail to approximate a meaningful hyperstationary point or rely on stringent assumptions to do so. This naturally leads to a fundamental question:
Can strong hyper-stationarity be computed in general settings where multiple lower-level solutions exist?
Addressing this question is challenging for a simple reason chain. When the lower level admits multiple solutions, the induced hyper-objective is typically nonsmooth and, under standard assumptions, no better than Lipschitz continuous [7,Corollary 6.1]. At precisely this level of regularity, computing (stronger) approximate Clarke stationary points is, in general, computationally intractable [30,46]. Thus Lipschitz regularity alone is too weak for algorithmic purposes, motivating new, verifiable structural conditions that make meaningful hyper-stationarity attainable.
Our Contributions. In this paper, we address the above challenges and show that (strong) Clarke stationarity of hyper-objective functions is computable for a broad class of BLO problems. As our key contribution, we identify a hidden weak convexity/concavity structure of the hyper-objective in nonconvex-PŁ BLO, 1 which places the analysis within the well-studied weakly convex/concave framework. Within this setting, approximate hyper-stationarity admits a natural Clarke-subdifferential characterization that we leverage to obtain computable guarantees.
The foundation of our analysis is a new concept, set smoothness (Definition 3), which extends classical smoothness to set-valued mappings and encompasses several variational regularity notions [40,15,7,27]. Building on this notion, we prove two complementary statements. First, if the lower-level solution mapping is set smooth, then the optimistic (resp. pessimistic) hyper-objective is weakly concave (resp. weakly convex). Second, a broad and verifiable condition guarantees set smoothness: When the lower-level function satisfies an error bound condition-equivalently, the PŁ condition-the solution mapping is set smooth. Together, these statements provide checkable criteria under which the hyper-objective inherits a weak convexity/concavity structure.
Once the hidden weak convexity/concavity of the hyper-objective is in place, approximate Clarke hyper-stationary points can be computed by a simple inexact zeroth-order scheme. In the weakly convex case, results based on the Moreau envelope [12,56,41] provide convergence and complexity guarantees. For the weakly concave case, however, no existing algorithmic guarantee is known, and the absence of a Moreau-type smoothing technique makes the analysis significantly more challenging. We overcome this by developing a novel convergence proof based on a Brøndsted-Rockafellartype approximation result [43,Theorem 2], and establish, to the best of our knowledge, the first general computational guarantee for finding approximate Clarke stationary points of nonsmooth hyper-objective functions.
Overall, these developments, particularly set smoothness, provide a principled foundation for the computability of hyper-stationarity in BLO and open new avenues for other structured nonsmooth optimization problems.
Organization. This paper is organized as follows. Sec. 2 collects assumptions and preliminaries. Sec. 3 introduces set smoothness and uses it to reveal a weak convexity/concavity structure of the hyper-objective. Sec. 4 presents an inexact zeroth-order scheme and establishes convergence guarantees for computing approximate Clarke hyper-stationary points. Sec. 5 concludes with final remarks.
Notation. The notation used in this paper is mostly standard. We use ∥x∥ to denote the Euclidean norm of a vector x and ∥A∥ to denote the l 2 norm of a matrix A. We use B(z, r) to denote the ball centering at z with radius r, i.e., {x : ∥x -z∥ ≤ r}. For a scalar α ∈ R and a set S ⊆ R n , we use α • S to denote their product {αx : x ∈ S}. We define the distance from a vector x ∈ R n to S by dist(x, S) := min z∈S ∥x -z∥ and the projection of x onto S by Π S (x) := arg min z∈S ∥x -z∥.
We use Conv(S) to denote the convex hull of S. For two sets S 1 , S 2 ⊆ R n , define their Minkowski sum by S 1 + S 2 := {x 1 + x 2 : x 1 ∈ S 1 , x 2 ∈ S 2 }, and define their Hausdorff distance (with respect to ∥ • ∥) by
d H (S 1 , S 2 ) := max sup x1∈S1 dist(x 1 , S 2 ), sup x2∈S2 dist(x 2 , S 1 ) .
For a differentiable function g : R m × R n → R, we use ∇g to denote its gradient w.r.t. the joint variables (x, y) and ∇ x g (resp. ∇ y g) to denote its gradient w.r.t. x (resp. y).
this section cite: ['b12', 'b13', 'b12', 'b50', 'b35', 'b13', 'b3', 'b49', 'b16', 'b2', 'b5', 'b28', 'b54', 'b20', 'b35', 'b4', 'b7', 'b30', 'b16', 'b17', 'b2', 'b31', 'b20', 'b18', 'b18', 'b31', 'b7', 'b22', 'b6', 'b27', 'b19', 'b34', 'b51', 'b37', 'b0', 'b38', 'b6', 'b29', 'b45', 'b39', 'b14', 'b6', 'b26', 'b11', 'b55', 'b40', 'b42']

Section: Preliminaries
In this paper, we focus on nonconvex-PŁ BLO problems and make the following assumptions:
Assumption 1 (Lower-level Functions).
this section cite: []

Section: (A1).
The function f is L f -smooth and twice differentiable. Moreover, ∇∇ y f is H f -Lipschitz continuous, i.e., for all x 1 , x 2 ∈ R m and y 1 , y 2 ∈ R n ,
∥∇∇ y f (x 1 , y 1 ) -∇∇ y f (x 2 , y 2 )∥ ≤ H f (∥x 1 -x 2 ∥ + ∥y 1 -y 2 ∥) .
this section cite: []

Section: (A2).
The solution set S(x) = arg min y∈R n f (x, y) is nonempty closed convex for all x ∈ R m .
this section cite: []

Section: (A3).
There exists a scalar τ > 0 such that for all x ∈ R m and y ∈ R n , dist(y, S(x)) ≤ τ ∥∇ y f (x, y)∥.
this section cite: []

Section: Assumption 2 (Upper-level and Hyper-objective Functions).
(B1). The function F is M F -Lipschitz continuous and L F -smooth.
this section cite: []

Section: (B2).
There exists x ⋆ ∈ R m such that φ o (x ⋆ ) > -∞ (resp. φ p (x ⋆ ) < +∞) for the optimistic (resp. pessimistic) setting.
Assumptions (A1), (A2), and (B1) are standard in BLO settings; see, e.g., [21,8,19,52,1] and the references therein. Assumption (B2) guarantees that the hyper-objective functions are well-defined and is imposed without loss of generality. Assumption (A3) imposes an error bound in the lower-level variable that holds uniformly over the upper-level parameter. This requirement is strictly weaker than the strong convexity-in-y conditions commonly used in prior work [18,22,21], as it allows the solution set arg min y f (x, y) to be multi-valued. Under L f -smoothnes f in y, (A3) implies the PŁ inequality
f (x, y) -min y∈R n f (x, y) ≤ τ L 2 f 2 ∥∇ y f (x, y)∥ 2 for all x ∈ R m , y ∈ R n ,
vice versus [33, Theorem 3.1]. Hence, our setting aligns with the widely adopted nonconvex-PŁ framework for BLO [45,52,35].
Under the standing assumptions, we begin with the solution mapping S, which under (A3) admits the following equivalent characterization: S(x) = {y ∈ R n : ∇ y f (x, y) = 0} .
(2) Furthermore, under Assumption 1, the solution mapping S as well as the hyper-objective functions φ o and φ p are the Lipschitz continuous. The Lipschitz continuity of the hyper-objective functions ensures that the Clarke and Goldstein subdifferentials are well defined.
Lemma 1 (Lipschitz Continuity of S(x)). (cf. [7, Proposition 6.1]) Under Assumption 1, the lower- level solution set function is M S -Lipschitz continuous with M S = L f τ , i.e., for any x 1 , x 2 ∈ R m , d H (S(x 1 ), S(x 2 )) ≤ M S ∥x 1 -x 2 ∥.
Definition 1 (Clarke Subdifferential). (cf. [9, Definition 1.1]) For a Lipschitz continuous function g : R m → R, the Clarke subdifferential of g at a point x ∈ R m is defined by ∂g(x) := Conv ({s ∈ R m : ∃ x ′ → x, ∇g(x ′ ) exists, ∇g(x ′ ) → s}) . We say that x is an (ϵ, δ)-approximate Clarke stationary point of g if
dist   0, z∈B(x,δ) ∂g(z)   ≤ ϵ.
Remark 1. The Clarke subdifferential ∂g reduces to the gradient ∇g when g is smooth. Moreover, if g is convex, then ∂g(x) coincides with the vanilla subgradients defined by {s :
g(z) ≥ g(x) + s T (z -x) ∀ z ∈ R m }.
Then, the Goldstein δ-subdifferential at x can be constructed by the convex hull of the Clarke subdifferentials taken over a δ-neighborhood of x. Here is the formal definition of Goldstein δ-subdifferential.
Definition 2 (Goldstein δ-Subdifferential). (cf. [20, Definition 2.2]) For a Lipschitz continuous function g : R m → R and a scalar δ ≥ 0, the Goldstein δ-subdifferential of g at a point x ∈ R m is defined by
∂ δ g(x) := Conv      z∈B(x,δ) ∂g(z)      .
We say that x is an (ϵ, δ)-approximate Goldstein stationary point of g if dist (0, ∂ δ g(x)) ≤ ϵ.
Leveraging the Lipschitz continuity of the hyper-objective, recent work has established the computability of (ϵ, δ)-Goldstein hyper-stationary points [7]. However, Goldstein stationarity is strictly weaker and does not, in general, imply Clarke stationarity. Indeed, there exists a convex, 2-Lipschitz function g : R 2 → R and a point x such that x is (0, δ)-Goldstein stationary while min z∈B(x,δ) dist(0, ∂g(z)) ≥ 2 √ 5 ; see [47,Proposition 2.7]. To obtain stronger algorithmic guarantees for hyper-objective minimization, we therefore focus on computing Clarke stationary points (and their (ϵ, δ)-approximate variants).
Despite the well-definedness, approximate Clarke stationarity is not achievable in finite time for general Lipschitz functions [30,46]. A sufficient condition for its computability is the weak convexity of g [12]. To elaborate on this, we review some basic properties of weakly convex functions. Given a function g : R m → R and a scalar r > 0, we say that g is r-weakly convex if the function x → g(x) + r 2 ∥x∥ 2 is convex. The following equivalent characterizations are useful for our analysis. (i) g is r-weakly convex.
(ii) For any θ ∈ [0, 1] and x 1 , x 2 ∈ R m , we have
g (θx 1 + (1 -θ)x 2 ) ≤ θg(x 1 ) + (1 -θ)g(x 2 ) + r 2 θ(1 -θ)∥x 1 -x 2 ∥ 2 .
(iii) For any x 1 , x 2 ∈ R m with ∂g(x 1 ) ̸ = ∅, and all subgradients v ∈ ∂g(x 1 ), we have
v T (x 2 -x 1 ) ≤ g(x 2 ) -g(x 1 ) + r 2 ∥x 2 -x 1 ∥ 2 .
For an r-weakly convex function g : R m → R with γ ∈ (0, 1 r ), we define its Moreau envelope and the proximal mapping by g γ (x) := inf z∈R n g(z) + 1 2γ ∥x -z∥ 2 , prox γ,g (x) := arg min
z∈R n g(z) + 1 2γ ∥x -z∥ 2 .
Clearly, prox γ,g (x) is single-valued and well-defined, when g is r-weakly convex and γ < 1 r . Next, we provide the standard result, which establishes a stationarity measure based on the gradient of the Moreau envelope.
Lemma 4 (Properties of Moreau Envelope). (cf. [56, Proposition 2.1]) Suppose that g : R n → R is a r-weakly convex function and γ < 1 r . The following hold:
(i) g γ (x) ≤ g(x) -1-γr 2γ x -prox γ,g (x) 2 . (ii) γ dist (0, ∂g( x)) ≤ x -prox γ,g (x) ≤ γ 1-γr dist (0, ∂g(x)). (iii) x = prox γ,g (x) if and only if 0 ∈ ∂g(x). (iv) ∇g γ (x) = 1 γ x -prox γ,g (x) .
Lemma 4 (ii) and (iv) show that ∥∇g γ (x)∥ equals zero if and only if x = prox γ,g (x) and 0 ∈ ∂g(x). Thus ∥∇g γ (x)∥ is a valid Clarke stationarity measure. Moreover, by Lemma 4 (iv), if ∥∇g γ (x)∥ ≤ ϵ, then ∥prox γ,g (x) -x∥ = γ∥∇g γ (x)∥ ≤ γϵ, hence dist 0, ∂g(prox γ,g (x)) ≤ ϵ. Equivalently,
∥∇g γ (x)∥ ≤ ϵ =⇒ dist   0, z∈B(x,γϵ) ∂g(z)   ≤ ϵ.(3)
Since Davis and Drusvyatskiy [12] establish non-asymptotic rates for finding x with ∥∇g γ (x)∥ ≤ ϵ when g is weakly convex, (3) implies that (ϵ, γϵ)-approximate Clarke stationarity is computable in this regime. This observation motivates us to establish weak-convexity-type structure for hyperobjectives; see Sec. 3.
Before leaving this section, we record weak concavity, a notion closely related to weak convexity. For a function g : R n → R, we say that g is r-weakly convex if -g is r-weakly convex. We have the following facts.
this section cite: ['b20', 'b7', 'b18', 'b51', 'b0', 'b17', 'b21', 'b20', 'b44', 'b51', 'b34', 'b6', 'b46', 'b29', 'b45', 'b11', 'b11']

Section: Fact 1.
If a function g : R m → R is r-weakly concave, then for any x 1 , x 2 ∈ R m , and v ∈ ∂g(x 1 ), we have g(x 2 ) ≤ g(x 1 ) + v T (x 2 -x 1 ) + r 2 ∥x 2 -x 1 ∥ 2 . Fact 2. If a function g : R m → R is r-smooth, then g is r-weakly convex and r-weakly concave, and the following inequality holds:
|θg(x 1 ) + (1 -θ)g(x 2 ) -g (θx 1 + (1 -θ)x 2 )| ≤ r 2 θ(1 -θ)∥x 1 -x 2 ∥ 2 .
this section cite: []

Section: Unveiling Hidden Structural Properties
This section is devoted to unveiling the hidden structural properties of the hyper-objective functions, which is the key contribution of this paper. Recall that the hyper-objective functions in (1) are defined by minimizing/maximizing the upper-level function w.r.t. y over the parameterized set S(x). We are motivated to investigate the property of the set-valued function S. Inspired by the smoothness of real-valued functions, we propose a novel concept of smoothness for set-valued functions, formalized in Definition 3. As we will show, the lower-level solution set function S satisfies this smoothness property, which in turn ensures the weak concavity (resp. convexity) of φ o (resp. φ p ).
Definition 3 (Set Smoothness). For a set-valued function Y : R m ⇒ R n with a convex domain dom(Y) ⊆ R m , we say that it is L-smooth if for any x 1 , x 2 ∈ dom(Y), θ ∈ [0, 1], and all y ∈ Y(θx 1 + (1 -θ)x 2 ), there exist y 1 ∈ Y(x 1 ) and y 2 ∈ Y(x 2 ) such that
∥θy 1 + (1 -θ)y 2 -y∥ ≤ L 2 θ(1 -θ)∥x 1 -x 2 ∥ 2 ;(4)
∥y 1 -y 2 ∥ 2 ≤ L∥x 1 -x 2 ∥ 2 . (5
)
The condition (4) can be viewed as a natural extension of the gradient-Lipschitz smoothness condition for real-valued functions to the setting of set-valued mappings. It guarantees that a convex combination of y 1 ∈ Y(x 1 ) and y 2 ∈ Y(x 2 ) provides a close approximation to a point in Y(θx 1 + (1 -θ)x 2 ), with an error that decays quadratically in ∥x 1 -x 2 ∥. This yields the following set inclusion:
Y (θx 1 + (1 -θ)x 2 ) ⊆ θY(x 1 ) + (1 -θ)Y(x 2 ) + L 2 θ(1 -θ)∥x 1 -x 2 ∥ 2 • B(0, 1).(6)
Intuitively, (5) enforces a consistent branch selection between Y(x 1 ) and Y(x 2 ): The chosen representatives y 1 and y 2 must remain aligned (Lipschitz-close) as the input varies, thereby excluding cross-branch pairings that could make the interpolation in (4) hold trivially while the underlying geometry is severely mismatched. Example 1 (Why the condition (5) is needed: A trivialization for the condition (4)). Define the setvalued map Y : R ⇒ R 2 by Y(x) = {(z, x) : z ∈ R}. Pick x 1 = a > 0, x 2 = -a, and θ = 1 2 ; then θx 1 + (1 -θ)x 2 = 0 and Y(0) = {(z, 0) : z ∈ R}. Choose y = 0 ∈ Y(0), y 1 = (K, a) ∈ Y(a), and y 2 = (-K, -a) ∈ Y(-a). We have 1 2 y 1 + 1 2 y 2 = y, so the condition (4) holds with zero error even though ∥y 1 -y 2 ∥ = 2 √ a 2 + K 2 can be made arbitrarily large as K → ∞. Hence the condition (4) alone does not preclude severely mismatched pairings on a convex domain. In contrast, (5) enforces ∥y 1 -y 2 ∥ 2 ≤ L∥x 1 -x 2 ∥ 2 = 4La 2 , which forces K 2 ≤ (L -1)a 2 and thereby rules out such cross-branch selections unless the Lipschitz modulus is correspondingly large.
this section cite: []

Section: □
With the notion of set smoothness in place, we now present our first main theoretical result. It shows that set smoothness serves as the key vehicle for establishing weak convexity/concavity of parametric optimization problems with coupled constraints: Under mild Lipschitz-type assumptions, the induced value function inherits weak convexity (or weak concavity). This is formalized in Theorem 1 below.
this section cite: []

Section: Theorem 1 (Implication of Set Smoothness).
Consider a real-valued function g : R m × R n → R and a set-valued function Y : R m ⇒ R n . Let ϕ(x) := max y∈Y(x) g(x, y) and D := {x : ϕ(x) > -∞}.
Assume that D is a nonempty closed convex set,
Y is L Y -smooth on D, and g is M g -Lipschitz continuous w.r.t. y, L g -smooth on D × Conv x∈D Y(x) . Then, the function ϕ is ρ-weakly convex with ρ = M g L Y + L g (1 + L Y ).
We now instantiate the framework in the bilevel setting. Our first step is to certify set smoothness for the lower-level solution map. Under the error-bound (EB) condition in Assumption 1 (A3), the mapping S : x → arg min y∈R n f (x, y) is L S -smooth (Theorem 2). Combining this with Theorem 1 shows that the pessimistic hyper-objective φ p inherits weak convexity (resp. the optimistic φ o inherits weak concavity).
this section cite: []

Section: Theorem 2 (EB Implies Set Smoothness).
If a function f : R m × R n → R satisfies Assumption 1, then its associated solution set function S :
x → arg min y∈R n f (x, y) is L S -smooth with L S = max{2H f τ (1 + 9L 2 f τ 2 ), 4L 2 f τ 2 }.
Proof idea (why residual backfilling is essential) Fix x 1 , x 2 and θ ∈ (0, 1), and set
x θ := θx 1 + (1 -θ)x 2 .
Given any y ∈ S(x θ ), our goal is to select y 1 ∈ S(x 1 ) and y 2 ∈ S(x 2 ) so that (4) and ( 5) hold. A natural choice is to project y onto the endpoint fibers, yielding ȳi := Π S(xi) (y) for i = 1, 2. Using Lemma 1, it is easy to see that this naive selection satisfies (5).
However, even when each fiber S(x) is convex, the midpoint ȳθ := θ ȳ1 +(1-θ) ȳ2 may correspond, at x θ , to a different local selection of the set-valued map S(•) than the given y ∈ S(x θ ). Consequently, in general multi-solution settings the naive midpoint error can be first-order,
∥ ȳθ -y∥ = Θ ∥x 1 -x 2 ∥ ,
which motivates an additional correction to synchronize the selections.
We therefore align the midpoint and backfill the residual: First project the naive midpoint to the middle fiber, ŷ := Π S(x θ ) ( ȳθ ), and then use the residual yŷ to refine the endpoint representatives:
y i := Π S(xi) ȳi + (y -ŷ) , i = 1, 2.
This construction cancels the first-order branch mismatch in the convex combination and leaves only a quadratic remainder. Consequently, (4) holds while (5) remains valid.
this section cite: []

Section: Remark 2.
Think of S(x) as a family of convex "fibers". The direct projections ȳ1 , ȳ2 may live on selections that are not synchronized with the selection containing y, so their convex combination carries a first-order drift. Projecting ȳθ to S(x θ ) identifies the correct selection at the midpoint; adding the same residual yŷ to both endpoints moves them to the same selection as y, making the first-order terms cancel in the average and exposing the desired O(∥x 1 -x 2 ∥ 2 ) behavior.
Theorem 2 is not limited to the lower-level problem of BLO but applies to general parametric optimization problems. The established set smoothness property offers new insights into the structure of the solution mapping, which goes beyond the variational conditions considered in the literature [40,27,15,7,53].
this section cite: ['b39', 'b26', 'b14', 'b6', 'b52']

Section: Remark 3 (Local Conditions are Sufficient).
Suppose the solution mapping S is defined on a bounded convex domain D ⊆ R m . To ensure Theorem 2, it suffices that Assumption 1 holds on the set D × Y, where
Y = Conv x∈D S(x) + 1 2 M S diam(D) B(0, 1).
The following simple example shows that the set smoothness of S does not, in general, require Assumption 1. This suggests that alternative sufficient conditions may guarantee set smoothness; identifying such conditions is an interesting direction for future work. Example 2. Consider f : R 2 → R defined by f (x, y) = g(sin x + y), where g : R → R has a nonempty set of minimizers V = arg min z∈R g(z). Then the solution set admits a closed form:
S(x) = arg min y∈R f (x, y) = V -sin x := { v -sin x : v ∈ V }.
In particular, S is 1-smooth in the sense of Definition 3 (since it is a translation of the fixed set V by the scalar -sin x), even though f need not satisfy Assumption 1. □
With Theorems 1 and 2 in place, we now state our main result on the weak convexity/concavity of the hyper-objective φ o (resp. φ p ).
this section cite: []

Section: Theorem 3 (Weak convexity/concavity of the hyper-objectives).
Assume Assumptions 1 and 2. Let L S be the set-smoothness modulus of S from Theorem 2, and define ρ := M F L S + L F (1 + L S ). Then the following hold:
(i) The optimistic hyper-objective φ o is ρ-weakly concave.
(ii) The pessimistic hyper-objective φ p is ρ-weakly convex.
Proof of Theorem 3. Theorem 2 guarantees that the set-valued function S is L S -smooth. Then, the result (ii) directly follows from Theorem 1. Hence, we only need to prove (i). Note that
-φ o (x) = -min y∈S(x) F (x, y) = max y∈S(x)
-F (x, y).
We see that -φ o is ρ-weakly convex by Theorem 1. It follows that φ o is ρ-weakly concave.
Theorem 3 establishes the weak concavity/convexity of the hyper-objectives in nonconvex-PŁ bilevel optimization (BLO). This stands in contrast to classical results (e.g., [19, Lemma 2.2]), which impose strong convexity of the lower level to obtain smooth hyper-objectives. Our result is significant because it places the minimization of these generally nonsmooth hyper-objectives within the framework of weakly concave/convex optimization. As a consequence, computing approximate Clarke hyperstationary points becomes tractable-an avenue we pursue in the next section. Crucially, all of these developments hinge on the set smoothness property (Definition 3), highlighting the utility of this notion.
this section cite: []

Section: Remark 4 (Lower-level Constraints Matter).
Under Assumptions 1 and 2, imposing an upperlevel constraint x ∈ X ⊆ R m with X nonempty, closed, and convex preserves the conclusions of Theorem 3: The functions φ o (x) + ι X (x) and φ p (x) + ι X (x) remain weakly concave and weakly convex, respectively, where ι X denotes the indicator of X . In contrast, adding a lower-level constraint y ∈ Y can destroy the weak concavity/convexity of the hyper-objectives, because the set smoothness of S may fail in this case; see Example 3. Developing structural conditions that recover such properties for lower-level constrained BLO is an interesting direction for future work.
Example 3. Let Y = [0, 1] × [0, 1]. Consider the pessimistic bilevel problem with a lower-level constraint:
min x∈R max y∈R 2 -1 ⊤ y s.t. y ∈ arg min y ′ ∈Y ∥y ′ -(x, 2)∥ 2 .(7)
Assumptions 1 and 2 are directly satisfied for (7), except for the unconstrained lower level; the only deviation here is the added constraint y ∈ Y.
The lower-level solution set is the projection of (x, 2) onto the box Y, hence
S(x) =    {(0, 1)}, x ≤ 0, {(x, 1)}, 0 ≤ x ≤ 1, {(1, 1)}, x ≥ 1.
Therefore the pessimistic hyper-objective is
φ p (x) =    -1, x ≤ 0, -x -1, 0 ≤ x ≤ 1, -2, x ≥ 1.
This function is not weakly convex. Indeed, for any ρ ≥ 0 consider h ρ (x) := φ p (x) + ρ 2 x 2 . Then h ρ has left and right derivatives at x = 0 given by h ′ ρ (0 -) = 0 and h ′ ρ (0 + ) = -1 (the quadratic term has zero slope at 0), which violates the monotonicity of one-sided derivatives required by convexity. Hence no ρ makes h ρ convex, i.e., φ p is not weakly convex. □
this section cite: []

Section: Computing Approximate Clarke Hyper-stationarity
Equipped with the weak convexity/concavity of the hyper-objectives, our next goal is to establish the computability of Clarke stationary points. First-order methods are impractical here because subgradients of the hyper-objectives are typically unavailable. In contrast, under mild conditions-e.g., F (x, •) is concave (resp. convex) so that the inner maximization (resp. minimization) is tractable, the function values of the hyper-objectives can be (approximately) evaluated at a given x [16,44]. This motivates the use of zeroth-order methods for minimizing hyper-objectives [7,34]. In particular, we adopt the inexact zeroth-order method (IZOM) in Algorithm 1, which employs a deterministic subroutine A to approximately evaluate φ β (x) (with additive accuracy w) by solving the inner problem in (1); see [7,[24][25][26] for practical implementations of A.
Algorithm 1 Inexact Zeroth-order Method (cf. [7, Algorithm 2]) Input: Radius ε > 0, iteration number T ∈ N, stepsize η, initial point x 0 ∈ R m , inexact error w > 0, and mode parameter β ∈ {1, 0} for t = 0, 1, . . . , T -1 do Sample u t from the the uniform distribution on the unit sphere in R m Compute A β w (x t + εu t ) and
A β w (x t -εu t ) by subroutine A Set G(x t ) = m 2ε (A β w (x t + εu t ) -A β w (x t -εu t ))u t x t+1 = x t -η G(x t ) end for Output: x uniformly chosen from {x t } T -1 t=0 Algorithm 2 Deterministic Subroutine A Input: Accuracy w > 0, iterate point x ∈ R m , and mode β ∈ {1, 0} if β = 1 then Compute a value φ(x) satisfying | φ(x) -φ o (x)| ≤ w else Compute a value φ(x) satisfying | φ(x) -φ p (x)| ≤ w end if Output: A β w (x) = φ(x)
Let φ p,γ denote the Moreau envelope of φ p with parameter γ. We quantify hyper-stationarity as follows: In the optimistic case we use the approximate Clarke stationarity measure, i.e., Definition 1, while in the pessimistic case we use the gradient norm of the envelope, i.e., ∥∇φ p,γ (x)∥. These two criteria can be unified in principle via (3); in either form they are strictly stronger than the Goldstein stationarity measure; see Sec. 2.
We then present the main theorem of this section.
Theorem 4. Suppose that Assumptions 1 and 2 hold. Given an iteration number T ∈ N, set η = Θ(m -1 2 T -1 2 ), ε = O(T -1 2 ), w = O(m -3 4 T -3 4 ) for Algorithm 1. Then, the following hold:
(i) Let ∆ o := φ o (x 0 ) -min x φ o (x) + 2M φ ε with M φ given in Lemma 2. For optimistic BLO,
we have
E dist 0, z∈B( x,δ) ∂φ o (z) 2 = O √ m(∆ o + 1) √ T with δ = O T -1 4 .
(ii) Let γ ∈ (0, 1 ρ+1 ) with ρ > 0 given in Theorem 3, and ∆ p := φ p,γ (x 0 ) -min x φ p,γ (x). For pessimistic BLO, we have
E[∥∇φ p,γ ( x)∥ 2 ] = O √ m(∆ p + 1) √ T .
Theorem 4 demonstrates, for the first time, that approximate Clarke hyper-stationarity is computable for nonconvex-PŁ BLO in both optimistic and pessimistic settings. This result significantly improves the existing computational guarantees for nonsmooth hyper-objective functions, which are mainly based on the Goldstein stationarity [7,28]. The proof of the optimistic case relies on a Brøndsted-Rockafellar-like relation, details of which can be found in Appendix D.1.
this section cite: ['b15', 'b43', 'b6', 'b33', 'b0', 'b6', 'b23', 'b24', 'b25', 'b6', 'b27']

Section: Conclusion and Discussion
In this paper, we established the first theoretical guarantee for computing approximate Clarke hyperstationarity in nonconvex-PŁ BLO. The key step is unveiling the hidden structural properties of hyper-objective functions via the newly introduced smoothness concept for set-valued functions. Specifically, we proved that (i) the smoothness of the set-valued function Y ensures the weak convexity of the function x → max y∈Y(x) ϕ(x, y); and (ii) the lower-level solution set function of BLO satisfies set smoothness. Consequently, we obtained the weak convexity/concavity of hyperobjective functions. With these properties in hand, we showed that an inexact zeroth-order method can compute approximate Clarke stationary points of hyper-objective functions.
We believe that our developments contribute to a deeper understanding of the computability properties of BLO and open up several directions for future research. First, with the established structural properties, our work calls for designing faster algorithms for computing Clarke hyper-stationarity. Second, it would be valuable to generalize our methodology to establish adapted properties for BLO in other settings (e.g., structured lower-level constrained BLO [28]). Furthermore, our set smoothness property, along with Theorem 1, may find applications in other fields such as coupled minmax optimization [48] and set-valued optimization [27], where set-valued functions play a central role.
NeurIPS Paper Checklist
this section cite: ['b27', 'b47', 'b26']

Section: References
Ref_id:b0 Title: Perturbed gradient descent via convex quadratic approximation for nonconvex bilevel optimization Year: (2025)
Ref_id:b1 Title: A unified analysis of descent sequences in weakly convex optimization, including convergence rates for bundle methods Year: (2023)
Ref_id:b2 Title: Implicit differentiation of lasso-type models for hyperparameter optimization Year: (2020)
Ref_id:b3 Title: Stackelberg games for adversarial prediction problems Year: (2011)
Ref_id:b4 Title: Bilevel knapsack with interdiction constraints Year: (2016)
Ref_id:b5 Title: Lower-level duality based reformulation and majorization minimization algorithm for hyperparameter optimization Year: (2024)
Ref_id:b6 Title: On bilevel optimization without lower-level strong convexity Year: (2023)
Ref_id:b7 Title: On finding small hyper-gradients in bilevel optimization: Hardness results and improved analysis Year: (2024)
Ref_id:b8 Title: Generalized gradients and applications Year: (1975)
Ref_id:b9 Title: Optimization and Nonsmooth Analysis Year: (1990)
Ref_id:b10 Title: Filling the gap between lower-C1 and lower-C2 functions Year: (2005)
Ref_id:b11 Title: Stochastic model-based minimization of weakly convex functions Year: (2019)
Ref_id:b12 Title: Foundations of Bilevel Programming Year: (2002)
Ref_id:b13 Title: Bilevel optimization Year: (2020)
Ref_id:b14 Title: Implicit Functions and Solution Mappings Year: (2009)
Ref_id:b15 Title: Algorithms for simple bilevel programming. Bilevel Optimization: Advances and Next Challenges Year: (2020)
Ref_id:b16 Title: Forward and reverse gradient-based hyperparameter optimization Year: (2017)
Ref_id:b17 Title: Bilevel programming for hyperparameter optimization and meta-learning Year: (2018)
Ref_id:b18 Title: Approximation methods for bilevel programming Year: (2018)
Ref_id:b19 Title: Optimization of Lipschitz continuous functions Year: (1977)
Ref_id:b20 Title: A two-timescale stochastic algorithm framework for bilevel optimization: Complexity analysis and application to actor-critic Year: (2023)
Ref_id:b21 Title: Bilevel optimization: Convergence analysis and enhanced design Year: (2021)
Ref_id:b22 Title: Hölderian error bounds and Kurdyka-Łojasiewicz inequality for the trust region subproblem Year: (2022)
Ref_id:b23 Title: Generalized Frank-Wolfe algorithm for bilevel optimization Year: (2022)
Ref_id:b24 Title: A near-optimal algorithm for convex simple bilevel optimization under weak assumptions Year: (2024)
Ref_id:b25 Title: A method with convergence rates for optimization problems with variational inequality constraints Year: (2021)
Ref_id:b26 Title: Set-valued Optimization Year: (2016)
Ref_id:b27 Title: A doubly stochastically perturbed algorithm for linearly constrained bilevel optimization Year: (2025)
Ref_id:b28 Title: On actor-critic algorithms Year: (2003)
Ref_id:b29 Title: Oracle complexity in nonsmooth nonconvex optimization Year: (2021)
Ref_id:b30 Title: On penalty methods for nonconvex bilevel optimization and first-order stochastic approximation Year: (2023)
Ref_id:b31 Title: A fully first-order method for stochastic bilevel optimization Year: (2023)
Ref_id:b32 Title: Error bounds, PŁ condition, and quadratic growth for weakly convex functions, and linear convergences of proximal point methods Year: (2024)
Ref_id:b33 Title: Gradient-free methods for deterministic and stochastic nonsmooth nonconvex optimization Year: (2022)
Ref_id:b34 Title: Bome! Bilevel optimization made easy: A simple first-order approach Year: (2022)
Ref_id:b35 Title: Pessimistic bilevel optimization: A survey Year: (2018)
Ref_id:b36 Title: Averaged method of multipliers for bi-level optimization without lower-level strong convexity Year: (2023)
Ref_id:b37 Title: Slm: A smoothed first-order lagrangian method for structured constrained nonconvex optimization Year: (2023)
Ref_id:b38 Title: First-order penalty methods for bilevel optimization Year: (2024)
Ref_id:b39 Title: Second-order Variational Analysis in Optimization, Variational Stability, and Control: Theory, Algorithms, Applications Year: (2024)
Ref_id:b40 Title: Adaptive first-and zeroth-order methods for weakly convex stochastic optimization problems Year: (2020)
Ref_id:b41 Title: Cubic regularization of newton method and its global performance Year: (2006)
Ref_id:b42 Title: Linear convergence of epsilon-subgradient descent methods for a class of convex functions Year: (1999)
Ref_id:b43 Title: An inertial extrapolation method for convex simple bilevel optimization Year: (2021)
Ref_id:b44 Title: On penalty-based bilevel gradient descent method Year: (2023)
Ref_id:b45 Title: On the hardness of computing near-approximate stationary points of clarke regular nonsmooth nonconvex problems and certain DC programs Year: (2021)
Ref_id:b46 Title: On the finite-time complexity and practical computation of approximate stationarity concepts of Lipschitz functions Year: (2022)
Ref_id:b47 Title: Minimax problems with coupled linear constraints: Computational complexity and duality Year: (2023)
Ref_id:b48 Title: Strong and weak convexity of sets and functions Year: (1983)
Ref_id:b49 Title: Fast algorithms for stackelberg prediction game with least squares loss Year: (2021)
Ref_id:b50 Title: Pessimistic bilevel optimization Year: (2013)
Ref_id:b51 Title: An generalized alternating method for bilevel optimization under the Polyak-Łojasiewicz condition Year: (2023)
Ref_id:b52 Title: Relative Lipschitz-like property of parametric systems via projectional coderivatives Year: (2023)
Ref_id:b53 Title: Overcoming lower-level constraints in bilevel optimization: A novel approach with regularized gap functions Year: (2024)
Ref_id:b54 Title: A two-time-scale stochastic optimization framework with applications in control and reinforcement learning Year: (2024)
Ref_id:b55 Title: Randomized coordinate subgradient method for nonsmooth optimization Year: (2022)
