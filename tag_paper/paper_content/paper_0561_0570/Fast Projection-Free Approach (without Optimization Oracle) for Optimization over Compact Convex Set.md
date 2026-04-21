Title: Fast Projection-Free Approach (without Optimization Oracle) for Optimization over Compact Convex Set
Abstract: Projection-free first-order methods, e.g., the celebrated Frank-Wolfe (FW) algorithms, have emerged as powerful tools for optimization over simple convex sets such as polyhedra, because of their scalability, fast convergence, and iteration-wise feasibility without costly projections. However, extending these methods effectively to general compact convex sets remains challenging and largely open, as FW methods rely on expensive linear optimization oracles (LOO), while penalty-based methods often struggle with poor feasibility. We tackle this open challenge by presenting Hom-PGD, a novel projection-free method without expensive (optimization) oracles. Our method constructs a homeomorphism between the convex constraint set and a unit ball, transforming the original problem into an equivalent ball-constrained formulation, thus enabling efficient gradient-based optimization while preserving the original problem structure. We prove that Hom-PGD attains optimal convergence rates matching gradient descent with constant step-size to find an ϵ-approximate (stationary) solution: O(log(1/ϵ)) for strongly convex objectives, O(ϵ -1 ) for convex objectives, and O(ϵ -2 ) for non-convex objectives. Meanwhile, Hom-PGD enjoys a low per-iteration complexity of O(n 2 ), without expensive oracles like LOO or projection, where n is the input size. Our framework further extends to certain non-convex sets, broadening its applicability in practical optimization scenarios with complex constraints. Extensive numerical experiments demonstrate that Hom-PGD achieves comparable convergence rates to state-of-theart projection-free methods, while significantly reducing per-iteration runtime (up to 5 orders of magnitude faster) and thus the total problem-solving time.

Section: Introduction
We consider constrained optimization where the objective is smooth, possibly non-convex, and the constrained set is compact convex. Although popular second-order methods, such as interior-point methods [PW00,Wri97] and cutting plane methods [B + 15], achieve linear convergence rates, their per-iteration computational complexity scales super-linearly with the problem size, typically on the order of O(n 3 ) due to solving a linear system. Consequently, these methods become impractical for large-scale problems. Alternative approaches, such as projection-based gradient descent (PGD) first-order methods (see, e.g., [Bec17,ZPL22,ZL22]), provide a computational benefit for simple convex sets where orthogonal projections can be performed efficiently, such as Euclidean balls and boxes. Despite their slower convergence rate of O(1/ϵ) in the convex smooth setting and O(1/ϵ 2 ) in the non-convex smooth setting, these methods are favorable in practice due to their relatively low per-iteration cost. However, the projection operation, i.e., solving a convex problem with a quadratic objective over constraints, is computationally expensive except for simple constraint sets. [LMY23] NC Simplex -Hadamard Parameterization + Pertubed PGD Constant O(n) O(ϵ -foot_0 ) [LG23] C C SC (Obj + Ctr Set) Sub-GD Alg. for the RD [4]   Implicit and Diminishing QOO O(ϵ -1 ) S (Obj + Ctr Set) O(ϵ -0.5 ) SC & S (Obj + Ctr Set) O(log ϵ -1 )
[
Gri24b] C C Upper Radial Obj Accelerate Sub-GD Alg. for the RD [4] Implicit or Vanishing MO O(ϵ -0.5 ) [MHSY25] DC C -Frank-Wolfe Diminishing LOO O(ϵ -2 ) Theorem 1 Theorem 2 Theorem 3 C C ND Minimizer Hom-PGD (Sec. 3) Constant MO O(ϵ -1 ) SC C -O(log ϵ -1 ) NC C -O(ϵ -2 )
1 Abbreviations: C = "convex", NC = "non-convex", DC = "difference of convex", SC = "strongly convex", S = "Smooth", Obj = "objective", Ctr = "constraint", GD = "gradient descent", ND = "non-degenerate", RD= "radial dual", LOO = "linear optimization oracle", QOO = "quadratic optimization oracle", MO = "membership oracle". 2 Step-size: (i) vanishing step-size: depends on ϵ, (ii) diminishing step-size: decreases as poly(1/K) with the number of iterations K, (iii) constant step-size: is independent of both ϵ and K, and (iv) implicit step-size: has implicit parameters such as smoothness and optimal objective. 3 Convergence Rate: number of iterations for finding an ϵ-approximate stationary point for non-convex optimizations or an ϵapproximate optimum for convex optimizations. 4 The radial dual (RD) of a convex constrained problem is an unconstrained min-max problem [Gri24a,Gri24b].
To circumvent these issues, projection-free methods based on the Frank-Wolfe (FW) algorithm [FW + 56] have been widely studied (e.g., [MZWG16,THZK21,Mha22,MHSY25]). These methods employ a linear optimization oracle (LOO) at each iteration instead of projections, with the former often being performed efficiently [CP21]. However, LOO can still be computationally expensive over complex constrained sets; therefore, FW methods are confined to scenarios where the LOO is efficient. Moreover, they exhibit oscillatory behavior near the solution, resulting in slow convergence [BRZ24,FG16]. Beyond FW methods, penalty-based approaches [Ber76, SCB + 97, LMX22] struggle with ill-conditioning as penalty parameters increase and perform poorly, particularly when dealing with complex constraints. Recent advances explore projection-free strategies leveraging techniques such as reparameterization [LMY23,TT24,CV25] and radial dual reformulation [Gri24a,Gri24b]. These methods highlight recent efforts to address the limitations of classical projection-free methods, but remain restricted to structured constraint sets and may suffer from practical drawbacks, such as impractical step-size choices (see Table 1). Please refer to Appendix A for a detailed discussion of related work to reduce the per-iteration cost and accelerate convergence for solving convexconstrained optimization. Despite the success of existing projection-free methods, the research gap still remains:
Can we design a projection-free approach for optimization over a general compact convex set with desirable properties, including fast convergence and cheap per-iteration cost?
In this paper, we propose a novel projection-free framework that positively answers this question. Concretely, we make the following contributions: ▷ In Sec. 3, we design the novel projection-free method, termed as Hom-PGD, that re-parameterizes the optimization over convex compact sets to equivalent ball-constrained optimization. By solving the equivalent problem with gradient descent with closed-form projection and mapping the converged solution back, we obtain the solution to the original constrained problem.
▷ In Sec. 4, we establish convergence and complexity analysis for Hom-PGD: O(1/ϵ) in the convex setting, O(log 1/ϵ) in the strongly convex setting, and O(1/ϵ 2 ) in the non-convex setting, where established convergence rates are optimal 2 under unaccelerated settings. Moreover, the per-iteration complexity of Hom-PGD is cheap as O(n 2 ) without linear/quadratic optimization oracles. We also extend our framework to optimization over certain non-convex sets in Sec. 5.
▷ In Sec. 6, through extensive numerical experiments over convex and non-convex problems, including applications to max-cut SDP problems, we demonstrate Hom-PGD outperforms existing first-order approaches in computational efficiency, which achieve similar convergence rate but significantly lower per-iteration cost (up to 3-5 orders of magnitude).
To the best of our knowledge, the proposed Hom-PGD is the first projection-free, first-order framework capable of solving optimization over general convex compact sets while achieving the optimal convergence rate under unaccelerated settings without expensive optimization oracles.
this section cite: ['b96', 'b112', 'b6', 'b117', 'b116', 'b71', 'b46', 'b47', 'b85', 'b103', 'b81', 'b83', 'b27', 'b17', 'b37', 'b71', 'b105', 'b28', 'b46', 'b47']

Section: Problem Statement
We consider the following continuous convex constrained optimization problem:
min x f (x), s.t. x ∈ K,(P)
where x ∈ R n is the decision variable, f (•) is the objective function, and the constraint set K ⊂ R n is compact and convex. For ease of analysis and without loss of generality, we assume the constraint set K is defined by inequalitiesfoot_1 as K = {x ∈ R n | g(x) ≤ 0} with g = (g 1 , • • • , g m ), where g i : R n → R are functions.
this section cite: []

Section: Open Issues:
Although convex optimization has been extensively studied, existing methods face significant limitations when dealing with complex constraints. As discussed in Sec. 1, projectionbased approaches incur high computational costs beyond simple sets, while projection-free methods such as FW and primal-dual approaches are largely restricted to structured convex sets. For example, when solving semidefinite programming (SDP), the LOO required by FW involves solving an SDP itself, defeating its purpose as a low-cost alternative. These challenges underscore the need for projection-free algorithms that preserve fast convergence and maintain computational efficiency across broader convex programs.
this section cite: []

Section: Homeomorphic Optimization Approach
Motivated by recent advances in low-complexity schemes and reparameterization techniques for solving constrained optimization problems [LCL23, LMY23, PP23, LCL24, RSW24, LC25], we propose a novel approach that transforms the original constrained problem via a homeomorphic mapping between the convex constraint set K and the unit ball B. This transformation preserves the essential problem structure while replacing the potentially complex constraint set with the geometrically simple unit ball, thereby enabling efficient gradient-based optimization without expensive projection operations.
Definition 3.1 (Homeomorphic constrained optimization). Given a homeomorphism ψ : B → K, we define the transformed optimization problem with objective function h(z) = (f • ψ)(z) and constraint set B = ψ -1 (K) as: min z h(z), s.t. z ∈ B (H) Homeomorphism (or homeomorphic mapping) is a bi-continuous bijection from two topological spaces, guaranteeing the topological equivalence. It is a classic result that any compact convex set is homeomorphic to a unit ball [Ges12,Bre13], i.e., there exists a homeomorphism ψ such that B = ψ -1 (K) and K = ψ(B). Thus, we can transform any optimization problem P over a compact convex set into a ball-constrained program H. In practice, this transformation relies on an explicit homeomorphic mapping, which we will discuss how to obtain in Sec. 3.2.
this section cite: ['b40', 'b15']

Section: Remark:
The transformed problem H has a non-convex objective function h(•) due to the non-linear mapping ψ even though the original objective is convex, but it features a simple constraint set as a unit ball, leading to a closed-form projection. Moreover, under the homeomorphic transformation, the original problem and its homeomorphic counterpart are equivalent, i.e., there exists a bijective correspondence between their optimal solution sets P * and H * , where P * = {x | x ∈ arg min{P}} and similarly for H * . Specifically, for any x ∈ P * , there exists a unique z ∈ H * such that x = ψ(z), and vice versa. Thus, we can solve the transformed problem H without expensive projection to obtain the corresponding optimal solution of the original problem P.
this section cite: []

Section: Algorithm Overview
Algorithm 1 Hom-PGD Input: initial point z 0 , problem H with ψ and maximum iteration number K for k = 0 to K do Compute stepsize α k Update:
z k+1 = Π B (z k -α k ∇h(z k )) end for Output: x K = ψ(z K )
As illustrated in Fig. 1, PGD in the original space suffers from expensive projection operations during iteration over the constraint boundary. To solve problem P without expensive projection, we transform problem P into a ball-constrained optimization H by a homeomorphic mapping ψ. Then we apply regular PGD to efficiently solve the homeomorphic optimization H with a closed-form projection, thereby termed projection-freefoot_2 methods. Finally, we map the obtained solution back to the original space to recover the corresponding solution for the original problem. We call the combination of homeomorphic transformation and regular PGD as Hom-PGD, shown in Alg. 1. Next, we discuss how to construct an explicit ψ for a general compact convex set K.
this section cite: []

Section: Construction of Homeomorphism
We introduce an explicit-form homeomorphic mapping between convex set K and unit ball B as follows, termed Gauge mapping [TZ22, LLC25]: Definition 3.2 (Gauge mapping). Let γ K (x, x • ) = inf{λ ≥ 0 | x ∈ λ(K -x • )} be the Gauge/Minkowski function [BM08] given an interior point x • ∈ int(K). The gauge mapping ψ : B → K is defined between a unit ball and a compact convex set:
ψ(z) = ∥z∥ γ K (z, x • ) z + x • , ∀z ∈ B; ψ -1 (x) = γ K (x -x • , x • ) ∥x -x • ∥ (x -x • ), ∀x ∈ K. (1
)
Figure 2: Gauge mapping illustration.
First, gauge mapping ψ establishes a homeomorphism between any compact convex set and the unit ball, ensuring that K = ψ(B) and B = ψ -1 (K). Intuitively, this mapping transforms the unit ball by first translating it to align with an interior point of the convex set, then scaling points radially outward from this interior point until the ball's boundary conforms to the convex set's boundary (illustrated in Fig. 2). Moreover, the gauge mapping has closed-form expressions for common convex sets (linear, quadratic, second-order cone, and linear matrix inequality constraints) and can be efficiently computed via bisection methods for general convex constraints. For a comprehensive property and computation of gauge mapping, we refer readers to Appendix B.
Leveraging this explicit homeomorphic gauge mapping ψ, we transform problem P to problem H, and apply Hom-PGD (Alg. 1) to solve it without expensive projection. However, the gauge mapping ψ depends on the choice of interior point x • (shown in Def. 3.2). Different choices of x • yield distinct gauge mappings that alter the landscape of the transformed problem H and affect the convergence behavior of Hom-PGD. As we will establish rigorously in Sec. 4, gauge mappings with smaller Lipschitz constants favor faster convergence of Hom-PGD. Thus, we proceed to analyze the Lipschitz properties of the gauge mapping as: Proposition 3.3 (Bi-Lipschitz constants of the gauge mapping). Let K ⊂ R n be a compact convex set and let x • ∈ int(K) be an interior point. Define the inner and outer radii with respect to x • as
r i := sup{r ≥ 0 : B(x • , r) ⊆ K}, r o := inf{r ≥ 0 : K ⊆ B(x • , r)}, such that B(x • , r i ) ⊆ K ⊆ B(x • , r o ).
Then the Lipschitz constant (denoted as L(•)) of gauge mapping ψ associated with K satisfies the following bounds:
Forward Lipschitz: κ 2 := L(ψ) ≤ 2 r o + r 2 o /r i , Inverse Lipschitz: 1 κ 1 := L(ψ -1 ) ≤ 2/r i .
Therefore, to reduce the Lipschitz constant of the gauge mapping and boost the convergence of Hom-PGD, we can select a "central" interior point with large inner radius r i or small outer radius r o .
In practice, we may solve a convex problem by minimizing the constraint residual to find a "central" interior point approximately following [THH23] (refer to Appendix B for details).
this section cite: ['b10', 'b102']

Section: Performance Analysis
In this section, we present a comprehensive performance analysis for Hom-PGD, including the landscape analysis, convergence rate, and run-time complexity.
General Assumptions (with details in Appendix C.2): The objective f and constrained functions g i in P are continuously differentiable and smooth. The homeomorphic mapping ψ is invertible with a non-singular Jacobian matrix and is (κ 1 , κ 2 )-bi-Lipschitz continuous. Additionally, the Jacobian matrix of ψ (denoted as J ψ ) exists and is Lipschitz continuous.
We remark that our theoretical results hold for any homeomorphism satisfying these assumptions, and we construct a specific homeomorphism, the gauge mapping, in practice. Moreover, gauge mapping meets these assumptions (with details in Appendix B.3).
this section cite: []

Section: Landscape Analysis
First, under general assumptions, the composite function h = f • ψ in problem H inherits favorable properties as follows (more properties of h are included Lemma D.1). Next, recall that a point x * is said to be a stationary point of problem min x∈K f (x) with convex set K, if ∇f (x * ) ⊤ (x -x * ) ≥ 0 for any x ∈ K. It is well-known that any stationary point is a global optimum for a convex constrained optimization problem. A natural question arises: does this property also hold for the non-convex optimization problem H under convex problem P? For unconstrained cases, this property does hold, as the function h is invex with the property that every stationary point of an invex function is a global optimum [Mar85]. For the constrained case, we provide the following formal statement where the proof is deferred to Appendix D.6. Proposition 4.2 (Global Optimality of H). Suppose problem P is a convex optimization. If z * is a stationary point of problem H and LICQ (Def. D.3) holds at z * , then x * = ψ(z * ) is a stationary point of problem P (thus a global optimum). Hence, z * is a global optimum.
We remark that the LICQ assumption is mild in our setting. For problem (H), the only constraint is ∥z∥ 2 ≤ 1, so LICQ holds at any boundary point where the constraint is active. Moreover, since there are no equality constraints, LICQ trivially holds at any interior point. Moreover, we derive that there is a one-to-one correspondence for KKT points and non-degenerate stationary points between P and H. The relevant definitions, formal statements, and proofs are provided in Appendix D.5.
this section cite: ['b79']

Section: Convergence Analysis
In this section, we provide a theoretical convergence analysis of Hom-PGD (Alg. 1) for solving problem H. Our main result demonstrates that: Hom-PGD achieves the same convergence rate as the standard PGD for the (non-)convex problem P under mild regularity conditions, despite operating on the non-convex formulation H.
Before moving on, we recall some basic definitions. An ϵ-stationary point x * for problem P is defined by ∥G(x * )∥ ≤ ϵ where the gradient mapping G(x
) := G 1/α (x) = 1 α [x -Π K (x -α∇f (x))].
Note that in the definition, we omit the dependence of the function G on K and α.
this section cite: []

Section: (Strongly) Convex Objective f
The following theorem provides the convergence analysis of Hom-PGD for convex optimization P. Theorem 1. Suppose the strict complementary slackness condition (Def. D.4) holds for both problem P and H, and the problem P is convex with a non-degeneratefoot_3 minimizer x * . Let {z k } be the sequence generated by Hom-PGD with step-size α ∈ (0, 2
L h ]. For sufficient small ϵ > 0, {z k } K k=1 with K = O(L h /ϵ) contains z ′ such that h(z ′ ) -h ⋆ ≤ ϵ.
Proof Intuition. Under the invertible mapping, z * = ψ(x * ) is also a non-degenerate point from Lemma D.10. Consequently, it satisfies local strong convexity or local PL condition, meaning that h(z) -h * ≤ O(∥G(z)∥ 2 ) holds within a sufficiently small ball centered at z * . As a result, it follows from Theorem 3 that PGD only requires O(1/ϵ) complexity to find a O( √ ϵ)-stationary point z ′ such that h(z ′ ) -h(z * ) ≤ O(ϵ). The idea is motivated by [LMY23], which applies specific Hadamard parameterization to transform a simplex-constrained optimization to a sphere-constrained optimization. We extend their results to a general homeomorphic mapping ψ, and proof details can be found in Appendix E.1.
If the objective in P is strongly convex, we will have a faster (i.e., linear) convergence rate as: Theorem 2. Suppose problem P is convex with µ f -strongly convex objective f . Let {z k } k≥0 be generated by Hom-PGD with proper constant step-size α ∈ (0, 2 L h ]. With K = O(κ log 1/ϵ) where κ = L h /(µ f κ 1 ), we have h(z K ) -h(z * ) ≤ ϵ, and ∥z K -z * ∥ ≤ ϵ.
this section cite: ['b71']

Section: Proof Intuition.
If f is a strongly convex function over a convex set, it satisfies a generalized PL condition. The homeomorphic mapping preserves this generalized PL condition such that the linear convergence can be established for H. Details can be found in Appendix E.2.
Remark. (i) The convergence rates derived in Theorem 1 and 2 match the lower bounds in the unacclerated convex and strongly convex settings (referring to e.g., [AP23]). (ii) Theorem 1 and Theorem 2 demonstrate that the Hom-PGD algorithm not only maintains projection-free properties over the unit ball, reducing per-iteration computational complexity; but also achieves the same convergence rates as standard PGD when applied to the original convex optimization problem P, which is non-trivial since Hom-PGD operates on the non-convex objective in problem H.
this section cite: ['b1']

Section: Non-convex Objective f
For a non-convex objective f , a classical result (e.g., Theorem 9.15 [Bec14]) can be leveraged to show that Hom-PGD algorithms can converge to an ϵ-stationary point with O(1/ϵ 2 ) iterations. Theorem 3. Consider a problem min z∈Z h(z) with a convex set Z. Suppose h is non-convex and differentiable with L h -Lipschitz continuous gradient. Then the sequence {z k } K k=0 with K = O(L h /ϵ 2 ) generated by Hom-PGD algorithm with a constant step-size α ∈ (0, 2
L h ] contains an ϵ-stationary point z ′ , i.e, ∥G(z ′ )∥ ≤ ϵ for some z ′ ∈ {z k } K k=0 .
Remark. (i) The convergence rate in Theorem 3 matches the optimal rate for smooth non-convex optimization problems [CDHS20]. (ii) While Theorem 3 also applies to (strongly) convex objectives, it yields slower convergence than standard PGD in these cases, as it fails to exploit the convex structure of problem P and the hidden convexity of problem H [FHH23]. This highlights the significance of our results in Theorems 1 and 2, where Hom-PGD achieves the same convergence rates of standard PGD while avoiding expensive projection in the transformed domain. (iii) For non-convex objectives satisfying regularity conditions such as the KL property or error bound conditions, linear convergence rates as in Theorem 2 can also be achieved. See Remark E.5 for further discussion.
In these sections, we establish convergence results for Hom-PGD across different problem classes (Theorems 1, 2, and 3).
The convergence rates depend on the Lipschitz constants of the constructed gauge mapping, specifically: (i) the forward Lipschitz constant κ 2 , which relates to the parameter L h established in Lemma 4.1 and appears in Theorems 1 and 3; and (ii) the inverse Lipschitz constant 1/κ 1 , which appears in Theorem 2. As demonstrated in Prop. 3.3, the choice of interior point directly influences the Lipschitz constants of the gauge mapping ψ. Consequently, different interior points modify the Hom-PGD convergence rate by constant factors while preserving the fundamental convergence order.
Additionally, our theoretical analysis assumes access to exact gradients and gauge mappings, which is consistent with standard practice in the optimization literature (e.g., [LBGH23]). However, for general convex sets, the gauge mapping is numerically approximated using the bisection algorithm (Alg. 2) to compute both its value and gradient within a specified error tolerance δ at each iteration. This numerical approximation introduces an additional O(δ) term in the optimality gap of convergence results [DGN14]. Since δ can be chosen arbitrarily small, this additional error term remains negligible and does not affect the fundamental convergence guarantees of our algorithm.
this section cite: ['b5', 'b21', 'b38', 'b61', 'b30']

Section: Complexity Analysis
In this subsection, we analyze the total run-time complexity of Hom-PGD, including initialization complexity, per-iteration complexity, and last-step complexity.
Oracles. We list specific oracles in Hom-PGD besides general ones (e.g., zeroth-order oracle for explicit function evaluation).
• Membership oracle: Given x ∈ R n , this oracle M K (x) := I(x ∈ K) : R n → {0, 1} returns 1 if and only if x ∈ K. This oracle performs only feasibility checking without requiring the solution of optimization subproblems. For common convex sets including polyhedra and second-order cones, the membership oracle can be implemented with computational complexity not exceeding O(n 2 ) [Mha22], with significantly lower computational burden than LOO or projection in practice.
• Interior point oracle: This oracle returns an interior point of K by solving a convex feasibility problem. This requirement, common in projection-free frameworks [Mha22,Gri24a,Gri24b], can be addressed using first-order methods with Õ(n 2 )foot_4 complexity or interior-point methods with Õ(n 3.5 ) complexity. Notably, this oracle is only called once for the entire optimization algorithm.
Basic operations in Hom-PGD (with details in Appendix B.4). Hom-PGD requires computing the gradient of h = f • ψ per iteration and transforming the final solution via the gauge mapping.
• Computing gauge mapping ψ: Õ(n). To compute the gauge mapping in the general case, one may evaluate the gauge function γ K (•, x • ) to an accuracy ϵ with O(log 1/ϵ) membership oracle calls, plus O(n) operations for the scalar-vector product in Def. 3.2.
• Computing gradient of h: Õ(n 2 ). Numerical differentiation techniques (finite/automatic differentiation [BF97, BPRS18, LBGH23]) can be applied, e.g., computing each component ∇ i h (i = 1, 2, • • • , n) requires O(1) zeroth-order oracle calls of f and Õ(n) cost for evaluating ψ, yielding a total complexity of Õ(n 2 ).
Total run-time complexity of Hom-PGD.
• Initialization complexity (IC): One interior point oracle call to obtain an interior point of K.
• Per-iteration complexity (PiC): Each iteration cost Õ(n 2 ), comprising gradient computation ∇h(z) at Õ(n 2 ) and unit ball projection at O(n).
• Last-step complexity (LsC): It costs Õ(n) for computing ψ to map the converged solution z * back to the original space via x * = ψ(z * ).
• Number of iterations (#I): Convergence rate varies from different settings, referring to Sec. 4.2.
In conclusion, the total complexity of Hom-PGD equals to IC + #I • Pic + LsC = O(n 2 • #I). This complexity is significantly lower than that of second-order methods, which typically incur O(n 3 ) per-iteration cost, thereby highlighting the scalability of Hom-PGD to high-dimensional problems. Moreover, our method achieves an optimal convergence rate under the first-order setting, ensuring both efficiency and theoretical soundness.
this section cite: ['b81', 'b81', 'b46', 'b47']

Section: Discussions
5.1 Beyond Vanilla Gradient Methods Mirror gradient methods: Mirror descent methods use a mirror map to transform points into a dual space, where optimization steps are performed. Essentially, these methods can be viewed as generalized projection methods [BT03]. For example, with a quadratic mirror map, it becomes standard PGD; with a negative entropy mirror map, it recovers exponential gradient descent for simplex-constrained problems. However, for general convex constraints, mirror descent often lacks explicit mirror maps and suffers from high projection complexity, despite having convergence rates comparable to ours [Bec17].
this section cite: ['b18', 'b6']

Section: Advanced gradient methods.
One may note that we can replace the unaccelerated gradient methods with any advanced optimizers, such as momentum-based or Nesterov-accelerated methods, or the Adam optimizer popular in deep learning [KB14]. Despite their potential for accelerating optimization of the non-convex landscape in problem H, analyzing the convergence behavior by utilizing the hidden convexity structure remains challenging, which opens directions for future research. We also conduct illustrative experiments comparing vanilla gradient descent and Adam in Section 6.4.
Second-order methods. We may also consider second-order methods with projection to optimize problem H. However, significant challenges arise from both computational and theoretical perspectives. From a computational standpoint, evaluating the Hessian of the composite objective h = f • ψ requires computing a third-order tensor of the gauge mapping, which is both computationally expensive and memory intensive. From a convergence perspective, analyzing convergence over the non-convex landscape to a global optimum becomes substantially more difficult.
this section cite: ['b55']

Section: Extension to Non-Convex Constraints
Star-shaped set. Our framework can naturally extend to optimization over non-convex sets where an explicit homeomorphism ψ exists such that ψ(B) = K. For star-shaped sets, all points are visible from a star center x • [Lee10], allowing the construction of a gauge mapping that bijectively maps a standard ball to the set. Such star-shaped constrained problems arise in machine learning tasks such as ℓ p -constrained adversarial attacks in neural networks [EBMA21]. Applying Hom-PGD to such problems maintains the O(1/ϵ 2 ) convergence rate for finding an ϵ-stationary point per Theorem 3.
Ball-homeomorphic set. Our framework may also extend to non-convex sets that are homeomorphic to a unit ball. However, determining if a non-convex set is ball-homeomorphic requires examining its topological properties, which is challenging. While a homeomorphism exists for such sets, there is generally no explicit form for the homeomorphism. Learning-based methods for approximating such homeomorphisms [LCL23,LCL24] present promising directions for future research.
this section cite: ['b34', 'b63', 'b64']

Section: General non-convex set.
For more general non-convex sets that may not be ball-homeomorphic (e.g., disconnected sets), our framework remains applicable but without optimality guarantees. Given an interior point x • in K, we can define X as the largest contained star-shaped set with center x • . Similar to constraint restriction methods [LNDT19], the optimality gap depends on the Hausdorff distance between K and X .
this section cite: ['b72']

Section: Empirical Study
In this section, we conduct simulations to demonstrate the effectiveness of Hom-PGD on both convex and non-convex constrained optimization problems. The detailed problem formulations, experimental settings, algorithm hyperparameters, and supplementary experiment results are in Appendix F.
Baselines: (i) PGD: Regular projected gradient descent applied to problem P, where the projection operation is called at each iteration. (ii) FW: Frank-Wolfe methods, which solve an update direction with a linearized objective and update the decision variables. (iii) ALM: Augmented Lagrangian methods for problem P that alternately update primal and dual coefficients for the unconstrained formulation to problem P. (iv) RD: Radial-Dual framework, which applies radial-dual to formulate the constrained problem into unconstrained min-max optimization. (v) Hom-PGD: Projected gradient descent applied to the transformed problem H shown in Sec. 3.
this section cite: []

Section: Illustrative Examples: Optimization over Polyhedron and Star-shaped Set
We examine a two-dimensional illustrative optimization problem involving quadratic optimization over both a (convex) polyhedron and a (non-convex) star-shaped set to demonstrate our method's efficiency. As shown in Fig. 3, Hom-PGD outperforms other first-order algorithms in both settings. The iteration trajectories in the transformed space reveal the mechanism behind this efficiency: Hom-PGD avoids complex projections while effectively performing gradient descent in a structured landscape of problem H to the optimum, even though it is non-convex.
this section cite: []

Section: Solving Second-order Cone Programming (SOCP)
We next evaluate the performance of algorithms on SOCP, which encompasses fundamental convex programs (LP, QP, convex QCQP) and has widespread applications in portfolio optimization [BBV04] and optimal power flow problems [Low14a]. Problem instances are randomly generated following the CVXPY documents. As shown in Fig. 4, our method not only converges rapidly to the target error tolerance but also demonstrates significantly lower per-iteration costs (up to 3-5 orders of magnitude) compared to projection-based or Frank-Wolfe methods, since Hom-PGD does not need complex optimization oracles such as projection or LOO during iterations. Further, we also use a commercial solver, MOSEK, which typically applies primal-dual interior point methods to solve convex programs. Notably, the solver costs 5424 seconds to solve the 1000-dim instance, while Hom-PGD takes less than 600 seconds to reach a 10 -3 objective optimality gap.
this section cite: ['b4', 'b73']

Section: Solving Max-Cut Semi-Definite Programming (SDP)
We further evaluate our method on the more challenging max-cut SDP problem. While max-cut is an NP-hard combinatorial problem, SDP relaxation with randomized rounding achieves an expected approximation ratio of 0.878 [GW95]. We generate random Erdős-Rényi graphs as test instances [HSS08]. Since the optimum of the max-cut SDP is typically low-rank, the gauge mapping encounters non-differentiability at these solutions. To address this practical issue, we apply the smoothing techniques described in Appendix B.3.2. As shown in Figure 5, our approach demonstrates efficient optimization even in high-dimensional decision spaces (50 2 variables) with positive semi-definite cone constraints. Notably, Hom-PGD exhibits a slower convergence rate on SDP with linear objectives  compared to SOCP with quadratic objectives. This behavior aligns with our theoretical analysis in Theorems 1 and 2, where strongly convex (quadratic) objectives yield faster convergence rates than convex (linear) objectives. The ALM methods solve the Burer-Monteiro SDP formulation [BM03] with log-rank (log N ) and Barvinok-Pataki (bp)-rank ( √ 2N ) [Bar95, Pat98, BVB20]
. Despite the scalability of this low-rank formulation, it incurs violation on additional equality constraint and high iteration cost for each inner minimization. However, the per-iteration complexity of our method still outperforms other approaches, resulting in comparable convergence in terms of total running time.
this section cite: ['b48', 'b53']

Section: Scalability Tests and Ablation Study
We first evaluate the scalability of gauge mapping computation across various constraints and dimensions in Fig. 6, demonstrating efficiency (less than 0.01 seconds) up to 3000-dimensional constraints. Our ablation study further examines critical framework components: (i) interior point selection (Fig. 20), confirming that central points (smaller Lipschitz) accelerate convergence as predicted by our theory analysis in Sec. 4.2; and (ii) gradient method variants (Fig. 21), revealing that advanced optimization techniques (e.g., Adam [KB14]) further enhance performance for solving non-convex problem H, suggesting promising directions for future research.
this section cite: ['b55']

Section: Conclusion and Limitations
In this work, we propose Hom-PGD, a projection-free method that transforms constrained optimization over general convex (and certain non-convex) sets into a ball-constrained problem via a homeomorphism. Hom-PGD achieves optimal convergence rates with O(n 2 ) per-iteration complexity without expensive projections or oracles. Numerical results show competitive convergence with significantly lower iteration costs. Despite its efficiency, there are several limitations to be addressed in future work: (i) Extending Hom-PGD to more general non-convex sets is non-trivial, as discussed in Sec. 5. (ii) From the convergence theory perspective, while Hom-PGD achieves optimal convergence rates under unaccelerated settings, it remains an open question whether acceleration techniques (e.g., Nesterov-style methods) can be incorporated to attain optimal accelerated rates. The challenge stems from the non-convexity of the transformed problem H. (iii) The gauge mapping used in this work serves as a simple and explicit homeomorphism but may not be the optimal choice in terms of conditioning or convergence behavior. Exploring alternative homeomorphisms tailored to specific problem structures could further improve performance.
this section cite: []

Section: References
Ref_id:b0 Title: Inexact gradient projection method with relative error tolerance Year: (2023)
Ref_id:b1 Title: Acceleration by stepsize hedging: Multi-step descent and the silver stepsize schedule Year: (2023)
Ref_id:b2 Title: Convex optimization: Algorithms and complexity Year: (2015)
Ref_id:b3 Title: Problems of distance geometry and convex properties of quadratic maps Year: (1995)
Ref_id:b4 Title: Convex optimization Year: (2004)
Ref_id:b5 Title: Introduction to nonlinear optimization: Theory, algorithms, and applications with MATLAB Year: (2014)
Ref_id:b6 Title: First-order methods in optimization Year: (2017)
Ref_id:b7 Title: On penalty and multiplier methods for constrained minimization Year: (1976)
Ref_id:b8 Title:  Year: (1997)
Ref_id:b9 Title: A nonlinear programming algorithm for solving semidefinite programs via low-rank factorization Year: (2003)
Ref_id:b10 Title: Set-theoretic methods in control Year: (2008)
Ref_id:b11 Title: Inexact spectral projected gradient methods on convex sets Year: (2003)
Ref_id:b12 Title: An introduction to optimization on smooth manifolds Year: (2023)
Ref_id:b13 Title: Automatic differentiation in machine learning: a survey Year: (2018)
Ref_id:b14 Title: Blended conditonal gradients Year: (2019)
Ref_id:b15 Title: Topology and geometry Year: (2013)
Ref_id:b16 Title: Learning deep linear neural networks: Riemannian gradient flows and convergence to global minimizers Year: (2022)
Ref_id:b17 Title: Frank-wolfe and friends: a journey into projection-free first-order optimization methods Year: (2024)
Ref_id:b18 Title: Mirror descent and nonlinear projected subgradient methods for convex optimization Year: (2003)
Ref_id:b19 Title: Deterministic guarantees for burer-monteiro factorizations of smooth semidefinite programs Year: (2020)
Ref_id:b20 Title: Deep frank-wolfe for neural network optimization Year: (2018)
Ref_id:b21 Title: Lower bounds for finding stationary points i. Mathematical Programming Year: (2020)
Ref_id:b22 Title: Efficient algorithms for minimizing compositions of convex functions and random functions and its applications in network revenue management Year: (2022)
Ref_id:b23 Title: On the burer-monteiro method for general semidefinite programs Year: (2021)
Ref_id:b24 Title: Acceleration with a ball optimization oracle Year: (2020)
Ref_id:b25 Title: A first-order primal-dual algorithm for convex problems with applications to imaging Year: (2011)
Ref_id:b26 Title: Boosting frank-wolfe by chasing gradients Year: (2020)
Ref_id:b27 Title: Complexity of linear minimization and projection on some sets Year: (2021)
Ref_id:b28 Title: Optimization over a probability simplex Year: (2025)
Ref_id:b29 Title: A primal-dual algorithm for linear programs Year: (1956)
Ref_id:b30 Title: First-order methods of smooth convex optimization with inexact oracle Year: (2014)
Ref_id:b31 Title: Efficiency of minimizing compositions of convex functions and smooth maps Year: (2019)
Ref_id:b32 Title:  Year: (1970)
Ref_id:b33 Title: Rates of convergence for conditional gradient algorithms near singular and nonsingular extremals Year: (1979)
Ref_id:b34 Title: Adversarial robustness with non-uniform perturbations Year: (2021)
Ref_id:b35 Title: Measure theory and fine properties of functions Year: (2018)
Ref_id:b36 Title: Regularity properties of a semismooth reformulation of variational inequalities Year: (1998)
Ref_id:b37 Title: New analysis and results for the frank-wolfe method Year: (2016)
Ref_id:b38 Title: Stochastic optimization under hidden convexity Year: (2023)
Ref_id:b39 Title: On the inexact scaled gradient projection method Year: (1956)
Ref_id:b40 Title: Convex open subsets of rn are homeomorphic to n-dimensional open balls Year: ()
Ref_id:b41 Title: Faster rates for the frank-wolfe method over stronglyconvex sets Year: (2015)
Ref_id:b42 Title: A linearly convergent variant of the conditional gradient algorithm under strong convexity, with applications to online and stochastic optimization Year: (2016)
Ref_id:b43 Title: The ball-proximal (=" broximal") point method: a new algorithm, convergence theory, and applications Year: (2025)
Ref_id:b44 Title: The ellipsoid method and its consequences in combinatorial optimization Year: (1981)
Ref_id:b45 Title: Some comments on wolfe's 'away step Year: (1986)
Ref_id:b46 Title:  Year: (2024)
Ref_id:b47 Title: Radial duality part ii: applications and algorithms Year: (2024)
Ref_id:b48 Title: Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming Year: (1995)
Ref_id:b49 Title: Prox-pda: The proximal primal-dual algorithm for fast distributed nonconvex optimization and learning over networks Year: (2017)
Ref_id:b50 Title: On the linear convergence of the alternating direction method of multipliers Year: (2017)
Ref_id:b51 Title: An equivalence between critical points for rank constraints versus low-rank factorizations Year: (2020)
Ref_id:b52 Title: Finite-dimensional variational inequality and nonlinear complementarity problems: a survey of theory, algorithms and applications Year: (1990)
Ref_id:b53 Title: Exploring network structure, dynamics, and function using networkx Year: (2008)
Ref_id:b54 Title: Efficient image and video co-localization with frank-wolfe algorithm Year: (2014)
Ref_id:b55 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b56 Title: Affine invariant analysis of frank-wolfe on strongly convex sets Year: (2021)
Ref_id:b57 Title: Linear convergence of gradient and proximal-gradient methods under the polyak-łojasiewicz condition Year: (2016)
Ref_id:b58 Title: A primal-dual interior point algo-rithm for linear programming Year: (1989)
Ref_id:b59 Title: Hidden convexity in qcqp with toeplitzhermitian quadratics Year: (2015)
Ref_id:b60 Title: The complexity of large-scale convex programming under a linear optimization oracle Year: (2013)
Ref_id:b61 Title: Projection-free adaptive regret with membership oracles Year: (2023)
Ref_id:b62 Title: Efficient bisection projection to ensure neuralnetwork solution feasibility for optimization over general set Year: (2025)
Ref_id:b63 Title: Low complexity homeomorphic projection to ensure neural-network solution feasibility for optimization over (non-)convex set Year: (2023)
Ref_id:b64 Title: Homeomorphic projection to ensure neural-network solution feasibility for constrained optimization Year: (2024)
Ref_id:b65 Title: Introduction to topological manifolds Year: (2010)
Ref_id:b66 Title: Gauges and accelerated optimization over smooth and/or strongly convex sets Year: (2023)
Ref_id:b67 Title: The effect of smooth parametrizations on nonconvex optimization landscapes Year: (2024)
Ref_id:b68 Title:  Year: (2012)
Ref_id:b69 Title: Gauge flow matching for efficient constrained generative modeling over general convex set Year: (2025)
Ref_id:b70 Title: Complexity of an inexact proximalpoint penalty method for constrained smooth non-convex optimization Year: (2022)
Ref_id:b71 Title: From the simplex to the sphere: faster constrained optimization using the hadamard parametrization. Information and Inference: A Year: (2023)
Ref_id:b72 Title: Convex restriction of power flow feasibility sets Year: (2019)
Ref_id:b73 Title: Convex relaxation of optimal power flow-part i: Formulations and equivalence Year: (2014)
Ref_id:b74 Title: Convex relaxation of optimal power flow-part ii: Exactness Year: (2014)
Ref_id:b75 Title: Constrained minimization methods. USSR Computational mathematics and mathematical physics Year: (1966)
Ref_id:b76 Title: On the convergence of the coordinate descent method for convex differentiable minimization Year: (1992)
Ref_id:b77 Title: Error bounds and convergence analysis of feasible descent methods: a general approach Year: (1993)
Ref_id:b78 Title: On the convergence rate of dual ascent methods for linearly constrained convex minimization Year: (1993)
Ref_id:b79 Title: The essence of invexity Year: (1985)
Ref_id:b80 Title: On the implementation of a primal-dual interior point method Year: (1992)
Ref_id:b81 Title: Efficient projection-free online convex optimization with membership oracle Year: (2022)
Ref_id:b82 Title: Stochastic conditional gradient methods: From convex minimization to submodular maximization Year: (2020)
Ref_id:b83 Title: Revisiting frank-wolfe for structured nonconvex optimization Year: (2025)
Ref_id:b84 Title: Fixed-rank matrix factorizations and riemannian low-rank optimization Year: (2014)
Ref_id:b85 Title: Scalable robust matrix recovery: Frank-wolfe meets proximal methods Year: (2016)
Ref_id:b86 Title: Lectures on convex optimization Year: (2018)
Ref_id:b87 Title: Smooth minimization of non-smooth functions. Mathematical programming Year: (2005)
Ref_id:b88 Title: A novel frank-wolfe algorithm. analysis and applications to large-scale svm training Year: (2014)
Ref_id:b89 Title: Numerical optimization Year: (1999)
Ref_id:b90 Title: Error bounds in mathematical programming Year: (1997)
Ref_id:b91 Title: On the rank of extreme matrices in semidefinite programs and the multiplicity of optimal eigenvalues Year: (1998)
Ref_id:b92 Title: Affine invariant convergence rates of the conditional gradient method Year: (2023)
Ref_id:b93 Title: Computational complexity of inexact proximal point algorithm for convex optimization under holderian growth Year: (2021)
Ref_id:b94 Title: On the convergence of inexact projection primal first-order methods for convex minimization Year: (2018)
Ref_id:b95 Title: Smooth over-parameterized solvers for non-smooth structured optimization Year: (2023)
Ref_id:b96 Title: Interior-point methods Year: (2000)
Ref_id:b97 Title: Hidden convexity, optimization, and algorithms on rotation matrices Year: (2024)
Ref_id:b98 Title:  Year: (2009)
Ref_id:b99 Title: Penalty functions. Handbook of evolutionary computation Year: (1997)
Ref_id:b100 Title: Scalable projection-free optimization methods via multiradial duality theory Year: (2024)
Ref_id:b101 Title: Convergence rates of inexact proximalgradient methods for convex optimization Year: (2011)
Ref_id:b102 Title: Rayen: Imposition of hard convex constraints on neural networks Year: (2023)
Ref_id:b103 Title: Frank-wolfe algorithm for learning svm-type multi-category classifiers Year: (2021)
Ref_id:b104 Title: Sensitivity analysis for variational inequalities Year: (1986)
Ref_id:b105 Title: Optimization over convex polyhedra via hadamard parametrizations Year: (2024)
Ref_id:b106 Title: Computationally efficient safe reinforcement learning for power systems Year: (2022)
Ref_id:b107 Title: Acceleration of frank-wolfe algorithms with open-loop step-sizes Year: (2023)
Ref_id:b108 Title: Convergence properties of inexact projected gradient methods Year: (2006)
Ref_id:b109 Title: Convergence theory in nonlinear programming. Integer and nonlinear programming Year: (1970)
Ref_id:b110 Title: Fast convergence of frank-wolfe algorithms on polytopes Year: (2024)
Ref_id:b111 Title: Accelerated affine-invariant convergence rates of the frank-wolfe algorithm with open-loop step-sizes Year: (2025)
Ref_id:b112 Title: Primal-dual interior-point methods Year: (1997)
Ref_id:b113 Title: Combining binary search and newton s method to compute real roots for a class of real functions Year: (1994)
Ref_id:b114 Title: Quadratic programming over an ellipsoid Year: (2001)
Ref_id:b115 Title: A proximal alternating direction method of multiplier for linearly constrained nonconvex minimization Year: (2020)
Ref_id:b116 Title: A global dual error bound and its application to the analysis of linearly constrained nonconvex optimization Year: (2022)
Ref_id:b117 Title: On the iteration complexity of smoothed proximal alm for nonconvex optimization problem with convex constraints Year: (2022)
Ref_id:b118 Title: Inexact primal-dual gradient projection methods for nonlinear optimization on convex set Year: (2020)
