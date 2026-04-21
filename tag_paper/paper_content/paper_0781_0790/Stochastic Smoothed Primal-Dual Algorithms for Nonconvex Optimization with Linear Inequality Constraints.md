Title: Stochastic Smoothed Primal-Dual Algorithms for Nonconvex Optimization with Linear Inequality Constraints
Abstract: We propose smoothed primal-dual algorithms for solving stochastic nonconvex optimization problems with linear inequality constraints. Our algorithms are single-loop and only require a single (or two) samples of stochastic gradients at each iteration. A defining feature of our algorithm is that it is based on an inexact gradient descent framework for the Moreau envelope, where the gradient of the Moreau envelope is estimated using one step of a stochastic primal-dual (linearized) augmented Lagrangian algorithm. To handle inequality constraints and stochasticity, we combine the recently established global error bounds in constrained optimization with a Moreau envelope-based analysis of stochastic proximal algorithms. We establish the optimal (in their respective cases) O(ε -4 ) and O(ε -3 ) sample complexity guarantees for our algorithms and provide extensions to stochastic linear constraints. Unlike existing methods, iterations of our algorithms are free of subproblems, large batch sizes or increasing penalty parameters in their iterations and they use dual variable updates to ensure feasibility.

Section: Introduction
We focus on the problem template
min x∈X f (x) subject to Ax = b,(1)
where f : R n → R is L f -smooth, the set X ⊆ R n is polyhedral, and easy to project. In particular, let X be given as X = {x : Hx ≤ h} for some matrix H and vector h.
Taking H = I, for example, gives this template the ability to model linear inequality constraints.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
In particular, when we have the problem
min x∈R n f (x) subject to Ax ≤ b,(2)
we introduce a slack variable t = Ax-b so that Ax-t = b and our optimization variable becomes x t . Then, we can equivalently write the problem in the template (1) by using the constraint t ≤ 0, where the set X = { x t : x ∈ R n , t ≤ 0} is easy to project. As such, we focus on (1) and our results directly apply to solving (2) by using this standard slack variable reformulation.
The assumption of X being easy-to-project is without loss of generality. Indeed, when X is not easy to project, we can add a slack variable for Hx ≤ h similar to the above paragraph, to have a linear equality constrained problem with projectable constraints (cf. (1)). We refer to (Li et al., 2021, Remark 6), for the classical conversion of an ε-stationarity point of the problem with the slack variable to the original inequality constrained problem. Throughout, we assume that we have access to an unbiased oracle F (x) such that E[F (x)] = ∇f (x), and E∥F (x) -∇f (x)∥ 2 ≤ σ 2 . (3) A common setting is when f (x) = E ξ∼Ξ [f (x, ξ)] where Ξ is an unknown distribution that we can draw i.i.d. samples from. In this case, it is common to set F (x) = ∇f (x, ξ) where E[∇f (x, ξ)] = ∇f (x). This will be our main focus.
Inclusion of the set X in (1) increases the modeling power of (1) significantly, while causing difficulties in the analysis. Many problems fit this template, including constrained and distributed optimization, nonnegative matrix factorization, sparse subspace estimation and collaborative learning, see for example 1 (Zhang et al., 2022;Hong, 2016). Moreover, reformulations of nonconvex problems are also common by using linear inequality constraints (Zhang et al., 2022).
Algorithm development for (1) and related templates with global complexity guarantees, have been active in the last couple of years (Alacaoglu & Wright, 2024;Zhang & Luo, 2020;Zhang et al., 2020;Lu et al., 2024;Li et al., 2021;Lin et al., 2022;Yan & Xu, 2022;Li et al., 2024;Boob et al., 2023;Hong, 2016), mainly due to the new applications of functionally constrained nonconvex optimization problems in the context of neural network training (Katz-Samuels et al., 2022;Dener et al., 2020). In these applications with problems involving nonconvex functional constraints, stochastic augmented Lagrangian methods (ALM) have found widespread use, whereas their behavior for even linearly constrained nonconvex optimization of the form (1) remain poorly understood. Our focus is to improve our understanding of stochastic ALM in the context of nonconvex optimization, by focusing on the fundamental template (1).
Compared to the setting of convex f , where the global complexity analysis is mature for ALM and its stochastic version (Yan & Xu, 2022), nonconvexity of f poses significant difficulties in the analysis of ALM. Many works in the literature focus on penalty based algorithms (which will be formally introduced later in this section) that do not perform dual updates (or perform negligible dual updates that we clarify later) (Lu et al., 2024;Li et al., 2021;Lin et al., 2022), rather than primal-dual algorithms such as ALM. However, in practice, dual updates are known to be essential for accelerating convergence. Penalty methods are known to be unstable since increasing penalty parameter causes Lipschitz constant of the subproblems to increase and can lead to numerical issues. These differences in behavior between penalty and augmented Lagrangian methods are well-known, see, for example, the classical books (Bertsekas, 2014, Sec. 2.2.5) (Nocedal & Wright, 1999, Sec. 17.5).
For problem (1) with access to full gradients of f and the full matrix A, optimal complexity with primal-dual methods are obtained in the work of Zhang & Luo (2022). When one has access to stochastic gradients of f and the matrix A, a recent work by Alacaoglu & Wright (2024) showed optimal complexity guarantees under expected smoothness (see Assumption 5.2), for the special case of (1) when X = R n . However, this latter restriction significantly reduces the generality of the template. For example, modeling standard quadratic programming requires X to be a half-space, which was not supported in the analysis of Alacaoglu & Wright (2024). Our goal is to go beyond these results by handling both the case when X ̸ = R n as well as the case when we do not have access to the matrix A but only to an unbiased estimate of A, by keeping optimal complexity guarantees. A more detailed comparison of complexity guarantees will be made in Section 6 and a summary is provided in Table 1.
Lagrangian, penalty and augmented Lagrangian. The standard approach to tackle (1) is to design algorithms operating on the Lagrangian, augmented Lagrangian or penalty functions. In particular, the Lagrangian function is given as L(x, y) = f (x) + ⟨Ax -b, y⟩, with the dual variables y, whereas the penalty function (or more precisely the quadratic penalty (QP)) has the form of Pen ρ (x) = f (x) + ρ 2 ∥Ax -b∥ 2 .
It is common for algorithms based on the penalty function to require ρ → ∞ for convergence (Bertsekas, 2014, Sec. 2.2.5). One major disadvantage of this strategy is that ρ getting larger makes the subproblem of minimizing the penalty function more and more ill-conditioned (cf. ( 4)).
An influential idea was the introduction of the augmented Lagrangian (AL) function which combined the idea of the Lagrangian and penalty formulations (Hestenes, 1969;Powell, 1969). In particular, the AL function is defined as
L ρ (x, y) = f (x) + ⟨Ax -b, y⟩ + ρ 2 ∥Ax -b∥ 2 .
Augmented Lagrangian methods in the classical literature were favoured because they did not require ρ to grow arbitrarily large. In fact, many instances of ALM converge to the optimal solution with fixed ρ since the incorporation of the dual variable updates aids in satisfying feasibility (Bertsekas, 2014, Prop. 2.4, Prop. 2.6).
Primal vs primal-dual algorithms. The algorithms based on the penalty function are generally referred to as penalty algorithms and are easier to analyze in different settings since they are primal-only algorithms, meaning that they only perform updates on primal variable x where approximate feasibility is ensured by ρ → ∞. In particular, a classical penalty method iterates for k = 1, 2, . . . as
x k+1 ≈ arg min x∈X f (x) + ρ k 2 ∥Ax -b∥ 2 , Select ρ k+1 > ρ k .(4)
The algorithms based on the AL function are generally more difficult to analyze due to the additional dynamics coming from the dual updates which are critical to ensure that the approximate feasibility is attained with constant ρ. An ALM iteration proceeds for k = 1, 2, . . . by updating
x k+1 ≈ arg min x∈X f (x) + ⟨y k , Ax -b⟩ + ρ 2 ∥Ax -b∥ 2 , y k+1 = y k + σ(Ax k+1 -b).
For penalty methods and ALM, different strategies exist to generate x k+1 that approximately minimize the penalty or augmented Lagrangian functions by either iterating multiple steps of gradient descent (GD), known as inexact algorithms, or applying one step of GD, known as linearized algorithms (Ouyang et al., 2015).
In view of the earlier discussion, when f is nonconvex, most of the literature focuses on either analyzing penalty methods, or analyzing ALM with negligible dual updates and increasing penalty parameters ρ, due to the inherent difficulty in analyzing the dual variable and its effect in convergence. In particular, as also highlighted in (Alacaoglu & Wright, 2024), many of the recent analysis of ALM is of the form of a perturbed penalty analysis, meaning that the feasibility is driven by increasing penalty parameters, and the dual updates are designed so that they do not deteriorate the estimates too much. Because of this, the dual step sizes are selected to be small to ensure boundedness of the dual variable (or controlling the growth of the dual variable).
We refer to such updates as negligible dual updates since the analyses do not harness the benefit of such updates in ensuring feasibility. Feasibility is driven by large penalty parameters. Some representative examples are (Lu et al., 2024), (Li et al., 2021), (Lin et al., 2022), (Li et al., 2024).
This is the case even in the deterministic setting and the only method that we are aware that can handle true ALM with fixed penalty parameters and non-negligible dual updates are due to (Zhang & Luo, 2022) that uses a linearized proximal AL function with a dynamic adjustment on the proximal center, which will be clarified in Section 2 since it will form the basis of our algorithmic development.
this section cite: ['b12', 'b0', 'b22', 'b19', 'b21', 'b30', 'b20', 'b4', 'b12', 'b14', 'b8', 'b30', 'b22', 'b19', 'b21', 'b0', 'b0', 'b10', 'b27', 'b25', 'b0', 'b22', 'b19', 'b21', 'b20']

Section: Contributions
In this paper, we propose a stochastic smoothed linearized ALM for solving (1) that only uses a single sample of stochastic gradient at every iteration. This algorithm also works with a constant penalty parameter and incorporates non-negligible dual updates for feasibility where the dual step sizes have the same order as the primal step sizes. We show that this method has its iteration complexity and sample complexity guarantees in the order of O(ε -4 ). Such a sample complexity result is optimal even in the unconstrained nonconvex case under our assumptions (see Assumption 1.1) (Arjevani et al., 2023). In contrast, the prior results with optimal complexity required large penalty parameters, no dual updates and further assumptions (Lu et al., 2024). We then prove that this complexity can be improved to O(ε -3 ) with variance reduction when an additional expected smoothness assumption is made (see Assumption 5.2). Under this stronger assumption, this is the optimal complexity even without constraints (Arjevani et al., 2023).
We consider extensions of this framework when we have linear constraints that hold in expectation, that is, when the constraints are given as E ξ [A ξ xb ξ ] = 0, with the same complexity guarantees. To our knowledge, this is the first algorithm achieving the optimal O(ε -4 ) benchmark sample complexity for nonconvex optimization with stochastic constraints using one sample per iteration, going beyond the best-known O(ε -5 ) complexity that is achieved for a more general problem that does not capture the structure of linear constraints (Li et al., 2024;Alacaoglu & Wright, 2024).
A more detailed comparison with the related works is given in Section 6. A summary is given in Table 1.
this section cite: ['b1', 'b22', 'b1', 'b20', 'b0']

Section: Preliminaries
We denote the indicator function of a convex closed set
X as I X (z) = 0 if x ∈ X and I X (x) = ∞ if x ̸ ∈ X.
The notation ∂f for a convex, closed function denotes the subdifferential set and ∂I X (x) is the normal cone of X at x, by definition. For matrix A, ∥A∥ denotes its operator norm.
Given closed and convex X, projection onto X is given as
proj X (x) = arg min v∈X ∥x -v∥ 2 .
Similarly, we define the proximal operator of f as
prox f (x) = arg min v f (v) + 1 2 ∥v -x∥ 2 .
We say that f is L-smooth when its gradient is L-Lipschitz:
∥∇f (x) -∇f (y)∥ ≤ L∥x -y∥.
We say that f is ρ-weakly convex when f + ρ 2 ∥•∥ 2 is convex. An L-smooth function is automatically L-weakly convex. Moreau envelope of the weakly convex f is defined as
φ λ (z) = min v f (v) + 1 2λ ∥v -z∥ 2 ,
which can be interpreted as a notion of smoothing. Moreau envelope has many useful properties such as being smooth when f is nonsmooth and weakly convex, when λ is selected accordingly. Moreover, stationary points of f and the Moreau envelope coincide (Drusvyatskiy & Paquette, 2019, Lemma 4.3). The gradient of the Moreau envelope can be computed as
λ -1 (x -prox λφ (x)).
this section cite: []

Section: Stationary points.
A succinct way of characterizing a stationary point of (1) is the following: x ⋆ is a stationary point if there exists y ⋆ such that the following hold:
0 ∈ ∇f (x ⋆ ) + A ⊤ y ⋆ + ∂I X (x ⋆ ) and 0 = Ax ⋆ -b.
One may, for example, refer to (Rockafellar, 2000). Accordingly, we say that (x, y) is ε-stationary if ∥Ax -b∥ ≤ ε and
∥v∥ ≤ ε where v ∈ ∇f (x) + A ⊤ y + ∂I X (x)(5)
which is a common notion used in related works, for example (Zhang & Luo, 2022).
We also use the following related notion of near-stationarity, as used in (Davis & Drusvyatskiy, 2019). We say that x is ε-near stationary if it satisfies
∥∇Ψ(x)∥ ≤ ε,(6)
where Ψ(x) is the Moreau envelope of the objective function f (x) + I X (x) + I {v:Av=b} (x) in (1), see also (7). We refer to (Davis & Drusvyatskiy, 2019) for the precise notion of near stationarity.
this section cite: ['b29', 'b7', 'b7']

Section: Assumptions
We next state the assumptions that will be used throughout. These assumptions are standard and to our knowledge, the weakest, in the literature for both deterministic and stochastic nonconvex problems with linear constraints (Zhang & Luo, 2022;Alacaoglu & Wright, 2024). A more detailed comparison of assumptions will be made in Section 6. Assumption 1.1. For the problem (1), the following holds:
1. The function f is L f -smooth and lower bounded over the feasible set: f (x) ≥ f > -∞ for any x ∈ X and Ax = b.
2. The set X admits an efficient projection and is polyhedral. That is, it has the form X = {x : Hx ≤ h} for some H, h.
3. We have access to stochastic gradients satisfying (3).
this section cite: ['b0']

Section: Algorithm
We introduce Algorithm 1 in this section. To gain a deeper understanding of the algorithm, we will go over two different ways of interpreting it.
Interpretation 1: Linearized proximal ALM. Algorithm 1 incorporates a single-step SGD approximation of the proximal AL function. This strategy is also known as the linearized proximal ALM. In particular, the first step of the algorithm approximates the proximal AL function 2 , that is,
x t+1 ≈ arg min x∈X L ρ (x, y t+1 ) + µ 2 ∥x -z t ∥ 2 ,
by a single step of projected SGD, followed by a dual variable update and updating the proximal center z t , which 2 Note that this is also a classical function (Rockafellar, 1976).
takes average of z t and x t , resulting in the terminology smoothed that we use for the algorithm.
Interpretation 2: Inexact GD on the Moreau envelope. 3Algorithm 1 can also be interpreted as an inexact gradient descent step on the Moreau envelope of the function in (1). In particular, this Moreau envelope is given as
Ψ(z t ) = min x∈X,Ax=b f (x) + µ 2 ∥x -z t ∥ 2 .(7)
By observing that minimizing the Moreau envelope helps on obtaining a near-stationary point in view of (6) (cf. (Davis & Drusvyatskiy, 2019)), inexact gradient update on this function requires the computation of
arg min x∈X,Ax=b f (x) + µ 2 ∥x -z t ∥ 2 ,
which is a nontrivial optimization subproblem. However, it is easier than (1) because the regularization provides us a strongly convex objective in the subproblem (given that λ is larger than L f ). As a result, we can approximate the solution of this problem by applying one iteration of ALM since this problem is a strongly convex optimization problem over linear constraints. We show that just one step of stochastic ALM is sufficient at every iteration by using a stochastic gradient computed with a single sample and one dual update, followed by the update of the proximal center z t .
On the surface, this algorithm strongly resembles that of Zhang & Luo (2022), from which we draw many ideas. However, in addition to using stochastic gradients, there is another subtle change, on the update of z t+1 . Unlike (Zhang & Luo, 2022), we update z t+1 by using x t to be able to continue the analysis with the bounded variance assumption on G (cf. Algorithm 1) instead of boundedness assumption on G, since the latter would require bounded domains. Thanks to this small change in this section, we handle the case with unbounded primal and dual domains.
this section cite: ['b28', 'b7']

Section: Convergence Analysis
In this section, we first provide the main complexity results, then introduce the main analysis tools and a proof sketch.
this section cite: []

Section: Main Theorem
In view of the two stationarity notions given in Section 1.2, we start with the result showing that Algorithm 1 outputs a point at which the norm of the gradient of Moreau envelope is small, in expectation.
For the result, we state the algorithmic parameters. To avoid clutter, we write the orders of the parameters by highlighting their dependences on the problem parameters. The explicit forms of the parameters are given in (25), in App. A.
τ ≍ 1 √ T , η ≍ 1 √ T , β ≍ 1 √ T , µ ≍ L f , λ ≍ L f + µ(∥A∥ 2 + 1).(8)
We are now ready to state the first main result.
Theorem 3.1. Let Assumption 1.1 hold and run Alg. 1 with parameters from (8). We have that E∥∇Ψ(z t * )∥ ≤ ε where t * is selected uniformly at random from {0, . . . , T -1} with T = Ω(ε -4 ). The stochastic oracle complexity is O(ε -4 ).
In particular, the above result gives us an ε-near stationary point in view of (Davis & Drusvyatskiy, 2019). To get an ε-stationary point, we perform a post-processing procedure to obtain the following output from the result of Alg. 1:
x = proj X (x t * -τ Ĝ(x t * , y t * +1 , z t * )),(9)
with
τ ≤ 1 L K where L K is the Lipschitz constant of L ρ (•, y, z) + λ 2 ∥ • -x∥ 2 (cf. (25)) and Ĝ(x t * , y t * +1 , z t * ) = 1 B B i=1 G(x t * , y t * +1 , z t * , ξ i )
for ξ i i.i.d. and B = Θ(ε -2 ). This is the only place where we use a large batch size and Algorithm 1 only runs with a single sample at every iteration. This post processing step is only done once and does not affect the overall complexity.
The details are given in Appendix A.3.
Corollary 3.2. Let Assumption 1.1 hold. From the output of Algorithm 1, we can obtain x which is an ε-stationary point. The complexity of the whole procedure is O(ε -4 ).
this section cite: ['b7']

Section: Analysis Tools
In our analysis, Moreau envelope of two functions is critical. The first was the Moreau envelope of the composite objective in (1), defined in (7). We next define the Moreau envelope on the proximal AL which is the main function to analyze projected SGD, cf. ( Davis & Drusvyatskiy, 2019)
φ 1/λ (x, y, z) = min u∈X L ρ (u, y) + µ 2 ∥u -z∥ 2 + λ 2 ∥u -x∥ 2 . (10
)
Another important quantity that has a significant role in the analysis is the proximal point
u * (x, y, z) = arg min u∈X L ρ (u, y) + µ 2 ∥u -z∥ 2 + λ 2 ∥u -x∥ 2 .(11)
With this, we trivially have
φ 1/λ (x, y, z) = L ρ (u * (x, y, z), y) + µ 2 ∥u * (x, y, z) -z∥ 2 + λ 2 ∥u * (x, y, z) -x∥ 2 .
This is the main point of departure from (Zhang & Luo, 2022) where the proximal AL function is used in the analysis, in the potential function. This is because (Zhang & Luo, 2022) used a projected full GD step on the proximal AL function for which, a descent inequality follows directly. In our case, because we apply a projected SGD step, to be able to handle updates with single-sample stochastic gradients, we need to use the Moreau envelope of the proximal AL function in our potential. This analysis of projected SGD was pioneered in (Davis & Drusvyatskiy, 2019).
The first result is a descent result on the Moreau envelope. Lemma 3.3 (cf. Lemma A.5). Under Assumption 1.1, for the x t+1 update given in Algorithm 1, we have
16E φ 1/λ (x t+1 , y t+1 , z t+1 ) ≤ 16E φ 1/λ (x t , y t+1 , z t+1 ) -τ λ 2 E∥u * (x t , y t+1 , z t ) -x t ∥ 2 + 8λτ 2 σ 2 + 2 4λτ µ + 16λτ 2 µ 2 + τ λ 2 µ 2 /γ 2 s E∥z t -z t+1 ∥ 2 , where γ s = 2µ + ρ∥A∥.
This follows mostly from (Davis & Drusvyatskiy, 2019) and handles the transition from x t to x t+1 in our analysis. One additional error term we have here is ∥z t+1 -z t ∥ 2 , due to the change in the proximal center z t , a term that was not involved in the analysis of (Davis & Drusvyatskiy, 2019).
Next, we incorporate the dynamics of the updates on the dual variable y t and the proximal center z t . These results use some ideas from (Zhang & Luo, 2022) with additional insights. This is because Zhang & Luo (2022) use Algorithm 1 Stochastic smoothed and linearized ALM Initialize: x 0 = z 0 ∈ X, y 0 ∈ R m and ρ ≥ 0. for t = 0 to T -1 do y t+1 = y t + η(Ax t -b) Sample ξ t ∈ Ξ i.i.d. and let G(x t , y t+1 , z t , ξ t ) = ∇f (x t , ξ t
) + A ⊤ y t+1 + ρA ⊤ (Ax t -b) + µ(x t -z t ). x t+1 = proj X (x t -τ G(x t , y t+1 , z t , ξ t )) z t+1 = z t + β(x t -z t )
L ρ (x, y) + λ 2 ∥x -z∥ 2 in their potential, so their analysis only characterizes the change in y and z in this function. Our analysis however, needs to characterize this change in the Moreau envelope of this function. This requires further estimations using the properties of the Moreau envelope, and the proximal point u * (x, y, z) (see e.g. Lem. A.6). Lemma 3.4. (cf. Lemma A.6) Under Assumption 1.1, for the iterates of Alg. 1, we have
2E φ 1/λ (x t , y t+1 , z t+1 ) ≤ 2E φ 1/λ (x t , y t , z t ) -2E⟨y t+1 -y t , Au * (x t , y t , z t ) -b⟩ -µE⟨z t -z t+1 , 2u * (x t , y t+1 , z t ) -z t+1 -z t ⟩.
It is easy to notice that combining the last two lemmas will give us a bound on the change of φ 1/λ from t to t+1. On the other hand, the inner products appearing on the right-hand side of the last bound will require an intricate analysis after combining with the terms coming from other components in the potential function, introduced next. One aim, is to make sure we get enough slack to be able to cancel error terms coming from ∥z t+1 -z t ∥ 2 in the previous lemma and further errors that will arise as we handle the inner products.
this section cite: ['b7', 'b7', 'b7']

Section: Proof Sketch

this section cite: []

Section: ONE ITERATION INEQUALITY ON THE POTENTIAL
As alluded to earlier, we introduce the potential function we work with, which incorporates the Moreau envelopes defined in ( 10) and ( 7):
V t = φ 1/λ (x t , y t , z t ) -2d(y t , z t ) + 2Ψ(z t ),
where we used the new notation
d(y, z) = min x∈X L ρ (x, y) + µ 2 ∥x -z∥ 2 . (12
)
There are two main changes compared to the analysis of (Zhang & Luo, 2022). The first is that the primal descent portion of our analysis investigates the behavior of the Moreau envelope of the proximal AL function (given in ( 10)) whereas the analysis of (Zhang & Luo, 2022) analyzes the proximal AL function (given in ( 19)) directly.
The reason for this departure is the well-known difficulty while analyzing SGD for constrained problems with single sample of stochastic gradients. Hence, it is not clear if it is possible to show a useful inequality with the proximal AL function in the constrained case. In particular, until the work of (Davis & Drusvyatskiy, 2019), convergence analyses of projected SGD required large batches.
In addition to combining the bounds from the previous section on the change of φ 1/λ , we have to characterize the change in d(y, z) and Ψ(z), for which we can use the following estimations, which only use the definition of y t+1 and hence have the same proof as the previous work. Lemma 3.5. (Zhang & Luo, 2020, Lemma 3.2, Lemma 3.3) For d(y, z) and Ψ(z) defined in (7) and (12), we have
2d(y t+1 , z t+1 ) -2d(y t , z t ) ≥ 2η⟨Ax t -b, Ax * (y t+1 , z t ) -b⟩ + µ⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩,and
Ψ(z t+1 ) -Ψ(z t ) ≤ µ⟨z t+1 -z t , z t -x * (z t )⟩ + µ 2σ 4 ∥z t -z t+1 ∥ 2 , where σ 4 = µ-L f µand
x * (y, z) = arg min x∈X L ρ (x, y) + µ 2 ∥x -z∥ 2 , (13
) x * (z) = arg min x∈X,Ax=b f (x) + µ 2 ∥x -z∥ 2 .(14)
We continue with the main inequality on the potential function with one iteration of Alg. 1. The proof of this lemma is rather intricate and requires a careful combination of the inner products coming from the previous lemmas, and uses the particular update of the proximal center z t+1 as well as parameter selections. Recall that u * (x, y, z) and x * (y, z) appearing in the lemma are defined in (11) and ( 13). Lemma 3.6 (cf. Lemma A.9). With Assumption 1.1 and parameters in (8) (see (25)), we have for Alg. 1 that
EV t -EV t+1 ≥ c β E∥z t+1 -z t ∥ 2 -λτ 2 σ 2 /2 + c τ E∥u * (x t , y t+1 , z t ) -x t ∥ 2 + c η E∥Ax * (y t+1 , z t ) -b∥ 2 ,(15)
where
c τ = Θ(1/ √ T ), c η = Θ(1/ √ T ), c β = Θ(1/ √ T
) with their precise definitions given in Lemma A.9.
One novelty in our analysis is to show that this potential function is still lower bounded and decreases, in expectation, up to an error term depends on τ 2 and the variance. To integrate this change into the framework of (Zhang & Luo, 2022) under reasonable assumptions on the stochastic oracle as mentioned earlier in Section 2, we also slightly changed the definition of z t+1 in the algorithm, due to technical reasons. In particular, in our case, we lose the control over ∥x t+1 -x t ∥ 2 (since we do not assume bounded domains in this section), whereas the deterministic analysis of (Zhang & Luo, 2022) have a natural control over such terms.
The other change is the error coming from the variance of stochastic gradients. This causes the complexity to deteriorate compared to the deterministic case, which is an effect common with algorithms based on SGD. In particular, with a correctly selected step size, we obtain a sample complexity with the same-order as SGD, which is optimal even for unconstrained nonconvex problems (Arjevani et al., 2023).
this section cite: ['b7', 'b1']

Section: COMPLEXITY ANALYSIS After Lemma 3.6, it is straightforward to obtain
E∥z t+1 -z t ∥ 2 ≤ ε 2 , E∥Ax * (y t+1 , z t ) -b∥ 2 ≤ ε 2 , E∥u * (x t , y t+1 , z t ) -x t ∥ 2 ≤ ε 2 , when T = Θ(ε -4 ).
Then, by tedious but straightforward calculations, we can directly get the bound on the norm of the gradient of the Moreau envelope, ∇Ψ(z t ), obtaining near-stationarity. The details appear in Appendix A.2.
A couple more steps let us go from this result to an εstationary point. The idea is simple: since we know that small ∥∇Ψ(z t )∥ means that we are near a stationary point, we can perform just one more iteration of SGD with batch size ≈ ε -2 to get an ε-stationary point, without changing the worst-case complexity. The details are in App. A.3.
this section cite: []

Section: Extension to Random Linear Constraints
We turn to the case when constraints are sampled, that is, we do not have access to the full matrix A, or vector b but only unbiased samples of them. This is a suitable setting, when, for example, we have a large matrix A. In particular, we have
A = E ζ∼P [A ζ ], b = E ζ∼P [b ζ ] and use A ζ , b ζ in the algorithm. We rewrite the template for convenience, as min x∈X f (x) subject to E ζ∼P [A ζ x -b ζ ] = 0.(16)
In this case, to get an unbiased stochastic gradient for proximal AL, we need to sample two i.i.d. samples of ζ:
G(x, y, z, ξ) = ∇f (x, ξ) + A ⊤ ζ 1 y + ρA ⊤ ζ 1 (A ζ 2 x -b ζ 2 ) + µ(x -z).(17)
An immediate issue here is that the variance of stochastic gradients of the proximal AL function scales linearly with x and y. Hence, assuming bounded variance would require assuming bounded dual variables, which is a strong assumption that is not satisfied in practice. To go around this difficulty, we have two adjustments, (i) we assume a constraint qualification (CQ) and compactness of X and (ii) we include a safeguarding procedure in the algorithm to monitor when the dual variable gets too large. Under these two modifications, we obtain the same complexity guarantees as our previous setting with deterministic constraints. Assumption 4.1. For problem (16), the following holds:
1. The feasible set {x : x ∈ X, Ax = b} is bounded.
2. The origin is in the relative interior of the set {Axb : x ∈ X}.
this section cite: []

Section: 3.
A has full row-rank.
In addition to the assumptions in the earlier setting, we require a Slater's condition as well as compact domains to ensure boundedness of the dual variable. Slater's condition is a classical CQ, see e.g., (Bertsekas et al., 2003, Sec.
this section cite: []

Section: 5.3.1).
Remark 4.2. The choice of M y is given next, which admittedly can be difficult in practice.
Let M V = max x,z∈X {K(x, 0, z) -2d(0, z) + 2Ψ(z)}, M = max x,z∈X {|f (x)|+ µ 2 ∥x-z∥ 2 + ρ 2 ∥Ax-b∥ 2 },
where K is defined in (19) and M Ψ is a uniform lower bound of Ψ(z t ), e.g., f . According to Assumption 4.1, there exists r > 0 such that for any direction d ∈ Range(A), we can find x ∈ X satisfying ∥Ax-b∥ = r and Ax-b has the same direction as d. Then, we choose M y as M y > M V -MΨ+2M r .
In this setting, we only state our theorem for nearstationarity. The ε-stationarity would follow in the same way as the previous section by a post-processing step. Theorem 4.3. Let Assumptions 1.1 and 4.1 hold and run Alg. 2 with parameters from (8). We have that E∥∇Ψ(z t * )∥ ≤ ε where t * is randomly selected from {0, . . . , T -1} with T = Ω(ε -4 ). The stochastic oracle complexity is O(ε -4 ).
As mentioned earlier, the optimal sample complexity for nonconvex optimization with Lipschitz ∇f is O(ε -4 ) (Arjevani et al., 2023). Our result matches this complexity while handling linear constraints with random sampling.
this section cite: ['b1']

Section: Extension with Variance Reduction
We now integrate the STORM variance reduction technique from (Cutkosky & Orabona, 2019) into our framework to solve (1) (See arXiv:2504.07607 for extension to stochastic constraints). We obtain Alg. 3, which improves the iteration and oracle complexity from O(ε -4 ) to O(ε -3 ) under a stronger assumption on the oracle, compared to Sec. 3. This not only leads to an improved rate, but also to a simpler analysis that does not rely on the Moreau envelope φ 1/λ .
this section cite: ['b6']

Section: Algorithm 2 Stochastic smoothed and linearized ALM for stochastic constraints with dual safeguarding
Input and Initialization:
M y > M V -MΨ+2M r (check Remark 4.2), x 0 = z 0 ∈ X, y 0 ∈ R m , ρ ≥ 0. for t = 0 to T -1 do y t+1 = y t + η(A ζt x t -b ζt ) where ζ t ∼ P is generated i.i.d. if ∥y t+1 ∥ ≥ M y then y t+1 = 0 Sample ξ t ∈ Ξ i.i.d. and generate E ξt [G(x t , y t+1 , z t , ξ t )] = ∇ x L ρ (x t , y t+1 ) + µ(x t -z t ) as in (17) x t+1 = proj X (x t -τ G(x t , y t+1 , z t , ξ t )) z t+1 = z t + β(x t -z t )
Algorithm 3 Stochastic smoothed and linearized ALM with STORM Initialize:
x 0 = z 0 ∈ X, y 0 ∈ R m , ∇f 0 = 1 N N i=1 ∇f (x 0 , ζ i ), N = T 1/3 and ρ ≥ 0 for t = 0 to T -1 do y t+1 = y t + η(Ax t -b) G(x t , y t+1 , z t ) = ∇f t + A ⊤ y t+1 + ρA ⊤ (Ax t -b) + µ(x t -z t ) x t+1 = proj X (x t -τ G(x t , y t+1 , z t )) z t+1 = z t + β(x t -z t ) Sample ξ t+1 ∼ Ξ i.i.d. and set ∇f t+1 = ∇f (x t+1 , ξ t+1 ) + (1 -α)( ∇f t -∇f (x t , ξ t+1 ))
Alg. 3 and Alg. 1 mainly differ in the update of stochastic gradient estimate ∇f t . If α = 0, Alg. 3 trivially reduces to Alg. 1. We next see that a particular choice of α gives better complexity under Assumption 5.2 (which is stronger than the oracle access and smoothness in Assumption 1.1). Remark 5.1. We only use a minibatch in the initialization, which does not affect the overall complexity. The minibatch size is N = T 1/3 , which is small compared to the total number of iterations T . Iterations of our algorithm only require 2 stochastic gradients, ∇f (x t , ξ t+1 ) and ∇f (x t+1 , ξ t+1 ).
For the analysis of Alg. 3, we introduce Assumption 5.2, used, e.g., in (Arjevani et al., 2023). In particular, Arjevani et al. (2023) showed that the oracle complexity O(ε -3 ) is tight under Assumption 5.2 even with no constraints. Assumption 5.2. We have access to a stochastic gradient of f satisfying (3). For a given ξ ∼ Ξ, we can query ∇f (x, ξ) and ∇f (y, ξ) for different points x, y. Moreover, we have
E ξ∼Ξ ∥∇f (x, ξ) -∇f (y, ξ)∥ 2 ≤ L 2 0 ∥x -y∥ 2 .
We introduce the potential Vt differing from Sec. 3 and 4. This is similar to (Zhang & Luo, 2022), except the last term which controls the error from the variance. Define
Vt = K(x t , y t , z t ) -2d(y t , z t ) + 2Ψ(z t ) + 1 48(L 2 0 + L 2 f )τ ∥ ∇f t -∇f (x t )∥ 2 , (18
)
where
K(x, y, z) = L ρ (x, y) + µ 2 ∥x -z∥ 2 . (19
)
One-step evolution of Vt that we analyze next is a key step in the analysis. Compared to (Zhang & Luo, 2022), we have the extra error due to using ∇f t instead of the full gradient.
Lemma 5.3 (cf. Lemma C.4). Under Assumptions 1.1 and 5.2, with parameters
µ = max{2, 4L f }, τ = T -3/2 , η = Θ(τ ), β = Θ(τ ), α = Θ(τ 2 ),(20)
(for detailed parameters, see (82)) we have
E Vt -E Vt+1 ≥ 2µ β E∥z t -z t+1 ∥ 2 + 1 2τ E∥x t -x t+1 ∥ 2 + 2ηE∥Ax * (y t+1 , z t ) -b∥ 2 + τ E∥ ∇f t -∇f (x t )∥ 2 -O(σ 2 τ 3 ). (21
)
Note that, on a high level, the main difference between Lemma 5.3 and Lemma 3.6 is that the order of τ in the error term is different. In Lemma 5.3, the order of τ is O(τ 3 ), while in Lemma 3.6, the order of τ is O(τ 2 ), which contribute to a faster convergence rate in for Alg. 3.
Theorem 5.4. Let Assumptions 1.1 and 5.2 hold. We have that E∥∇Ψ(z t * )∥ ≤ ε, where t * is selected uniformly at random from {0, . . . , T -1} with T = Ω(ε -3 ). The complexity of the whole procedure is O(ε -3 ).
this section cite: ['b1', 'b1']

Section: Related Works
We now compare the complexity results for obtaining an ε-stationary point, in view of Section 1.2.
this section cite: []

Section: Deterministic objective and deterministic constraints.
The setting when objective f in (1) is deterministic is the most well-studied with many results in the classical literature (Bertsekas, 2014). Recent work characterized the global oracle complexity of Lagrangian-based methods or ALM.
With nonlinear and nonconvex constraints, many of the existing works analyzing AL-based algorithms rely on strong CQs and boundedness assumptions and use large penalty parameters to ensure feasibility (Li et al., 2021;Lin et al., 2022;Kong et al., 2019;Kong & Monteiro, 2023;Kong et al., 2023). The existing frameworks so far fail to capture the importance of dual variable updates, which are, in fact, the main reason behind the ability to use constant penalty parameters while ensuring convergence, see e.g., (Bertsekas, 2014, Sec. 2.2.5). Recent works mentioned above obtained the complexity bound O(ε -3 ) for general nonlinear constraints with no specialization for linear constraints. When specialized to convex functional constraints, the best-known complexity for these methods is O(ε -2.5 ) (Lin et al., 2022).
When the constraints are linear, such as (1) with X = R n , Hong (2016) analyzed ALM with constant penalty parameters and non-negligible dual updates to get optimal complexity O(ε -2 ). The case of X ̸ = R n turned out to be significantly more challenging with many works focusing on variants of ALM with large penalty parameters (depending on the inverse of the final accuracy) to ensure near-feasibility and negligible dual updates that do not help with feasibility (Kong & Monteiro, 2023;Kong et al., 2023) and obtained the suboptimal complexity O(ε -2.5 ). The exceptions are the works (Zhang & Luo, 2020; 2022) that showed, for the case X polyhedral, near-optimal complexity O(ε -2 ) with a constant penalty parameter and dual steps with constant step sizes, with no constraint qualification. The key step was the global error bound that our work also relied on. Stochastic objective and deterministic constraints. One important step in generalizing the template to tasks arising in ML was to consider stochastic objectives where we access unbiased estimates. With general nonlinear constraints and Lipschitzness of ∇f , the optimal sample complexity is O(ε -4 ), obtained with double loop algorithms (Curtis et al., 2024;Boob et al., 2023;Ma et al., 2020). These works require strong assumptions on the boundedness of the primal domain as well as constraint qualifications, which are often not necessary with linear constraints.
Another set of results concerns stochastic optimization with deterministic nonlinear constraints with penalty-based algorithms. These works require large penalty parameters to ensure near-feasibility rather than dual updates (Lu et al., 2024;Alacaoglu & Wright, 2024). They assume expected Lipschitzness as Assumption 5.2, which is stronger than Lipschitzness of ∇f . Since these works focus on nonlinear functional constraints, the analysis requires boundedness assumptions as well as constraint qualifications, unlike our results in Section 3 for deterministic linear constraints.
Alacaoglu & Wright (2024) considered ALM with a constant penalty parameter and non-negligible dual updates and obtained the complexity O(ε -3 ) for linear equality con-straints under Assumption 5.2. This work only covered the case X = R n and left open the question of handling the case of general X, see (Alacaoglu & Wright, 2024, Sec. 5).
We resolve a special case of this question when X is polyhedral (covering many applications), allowing our analysis to cover linear inequality constraints. Alacaoglu & Wright (2024) used variance reduction for ∇f , which meant that they required Assumption 5.2, stronger than Assumption 1.1. In Sec. 5, we get the same complexity as this paper while allowing a polyhedral X to cover linear inequality constraints, which cannot be handled by Alacaoglu & Wright (2024).
Moreover, we also get the complexity O(ε -4 ) under Assumption 1.1. This is optimal under Assumption 1.1 and we refer to (Arjevani et al., 2023) for further details on the lower bounds. In contrast, the work in (Alacaoglu & Wright, 2024) does not have guarantees without Assumption 5.2.
In addition, though (Lu et al., 2024) considers the more general problem with nonconvex functional constraints, they make strong assumptions which are not easy to verify. It is not clear if their assumptions would hold with a general polyhedral constraint we have (see e.g., their Assumption 1(iv) and Eq. ( 7)). When the constraints are deterministic, we do not have any bounded domain assumption (our Sec.
3) whereas the assumptions of (Lu et al., 2024) are rather difficult to be satisfied without a bounded primal domain. Lu et al. (2024) analyzes a QP-based method, whereas we analyze an ALM-variant. ALM is known to be more stable and desirable in practice, but significantly more difficult to analyze, which is because the penalty parameter is fixed in ALM and it increases to infinity for QP. Our ALM algorithm could be extended to stochastic constraints, while (Lu et al., 2024) only handles deterministic constraints. Alacaoglu & Wright (2024) highlights the importance of analyzing ALM compared to QP methods in their Sections 1 and 6. Stochastic objective and stochastic constraints. This is the most general class, where the existing results come with many assumptions that are not always easy to interpret, similar to the case of stochastic objective and deterministic constraints described above (Li et al., 2024;Alacaoglu & Wright, 2024). The best-known complexity O(ε -5 ) is obtained by using Assumption 5.2, with an inexact, doubleloop, ALM in (Li et al., 2024) and by a single-loop QP algorithm in (Alacaoglu & Wright, 2024). These results concerning ALM need to use large penalty parameters, which renders them essentially as QP-methods since the dual updates do not contribute to the analysis for ensuring the feasibility. Other approaches for solving this sub-case also require double-loop algorithms and stronger assumptions since they focus on a generic nonconvex constraint (Boob et al., 2023;Ma et al., 2020), obtaining O(ε -6 ) without expected Lipschitzness. Hence, in this sub-case, none of these results harness the structure of linear constraints.
this section cite: ['b2', 'b19', 'b21', 'b16', 'b2', 'b21', 'b5', 'b4', 'b23', 'b22', 'b0', 'b0', 'b1', 'b0', 'b22', 'b22', 'b22', 'b22', 'b0', 'b20', 'b0', 'b20', 'b0', 'b4', 'b23']

Section: Stochastic Smoothed Primal-Dual Algorithms for Nonconvex Optimization with Linear Inequality Constraints
Notation.
Let us note that we define by E t the expectation conditioned on all the randomness up to and including x t .
this section cite: []

Section: A. Proofs for Section 3
In the proofs, let us recall
K(x, y, z) = L ρ (x, y) + µ 2 ∥x -z∥ 2 = f (x) + ⟨Ax -b, y⟩ + ρ 2 ∥Ax -b∥ 2 + µ 2 ∥x -z∥ 2 . (22
)
With this notation, we have the following, equivalent to ( 11):
u * (x, y, z) = arg min u∈X K(u, y, z) + λ 2 ∥u -x∥ 2 = arg min u∈X L ρ (u, y, z) + µ 2 ∥u -z∥ 2 + λ 2 ∥u -x∥ 2 . (23
)
We also recall (10).
φ 1/λ (x, y, z) = min u∈X L ρ (u, y) + µ 2 ∥u -z∥ 2 + λ 2 ∥u -x∥ 2 = min u∈X K(u, y, z) + λ 2 ∥u -x∥ 2 .(24)
We also introduce here some parameters that are used throughout, for convenience.
µ = max{2, 4L f }, L K = L f + ρ∥A∥ + µ, λ = 2L K , σ 4 = µ -L f µ , τ = 1 6λ 2 √ T , η = min 2µ + ρ∥A∥ 4∥A∥ 4 , τ 200∥A∥ 2 , τ (2µ + ρ∥A∥ 2 ) 20∥A∥ 2 , β = min τ 100 , 1 50λ , η 36µσ 2 , γ s = 2µ + ρ∥A∥, γ = (µ -L f )λ µ -L f + λ , γ K = µ -L f .(25)
We also mention the following basic facts that are used in the sequel.
Fact A.1. For x ∈ X, we have that x → K(x, y, z) is strongly convex with modulus γ K := µ-L f , and
x → ∇ x K(x, y, z) is L K := (L f + ρ∥A∥ 2 + µ)-Lipschitz continuous. For u ∈ X, u → K(u, y, z) + λ 2 ∥x -u∥ 2 is strongly convex with modulus γ s = µ -L f + λ, and u * (x, y, z) = arg min u∈X K(u, y, z) + λ 2 ∥x -u∥ 2 . Lemma A.2. (Planiden & Wang, 2016, Lemma 2.19) Let r > 0. The function f is r-strongly convex if and only if f 1 (x) = min u f (u) + 1 2 ∥x -u∥ 2 is r r+1 -strongly convex. Lemma A.3. The function x → φ 1/λ (x, y, z) is γ = (µ-L f )λ µ-L f +λ -strongly convex.
Proof. By definition, we have
φ 1/λ (x, y, z) = min u K(u, y, z) + I X (u) + λ 2 ∥x -u∥ 2 = λ min u K(u, y, z) + I X (u) λ + 1 2 ∥x -u∥ 2 .
Recall that γ K = µ -L f . Then, since K(x, y, z)/λ is γ K λ -strongly convex, we have min u
K(u,y,z)+I X (u) λ + 1 2 ∥x -u∥ 2 is γ K /λ γ K /λ+1 -strongly convex, by Lemma A.2. Hence, φ 1/λ (x, y, z) is strongly convex with modulus γ K γ K /λ+1 = λγ K λ+γ K = (µ-L f )λ µ-L f +λ . ■
this section cite: []

Section: A.1. Proofs for Lemma 3.6
In the next lemma, the first part is using the idea of Davis & Drusvyatskiy (2019) to analyze the algorithm under the bounded variance assumption instead of the restrictive bounded stochastic gradient assumption. The second part of the lemma also follows a similar idea as this work, with the exception of the dependence on the changing center point z t . This introduces additional issues, since the stochastic gradient in the update of x t+1 depends on z t whereas the proximal point u * (x t , y t+1 , z t+1 ) (that characterizes the iteration below) depends on z t+1 . Our analysis below estimates this additional error and shows it to be in the order of ∥z t+1 -z t ∥ 2 , which will be handled later.
Lemma A.4. Suppose that Assumption 1.1 holds, for the proximal point u * (x t , y t+1 , z t+1 ), defined as (11) we have the characterization
u * (x t , y t+1 , z t+1 ) = proj X (τ λx t + (1 -τ λ)u * (x t , y t+1 , z t+1 ) -τ ∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 )). (26
)
Moreover, for the sequence x t+1 calculated as Algorithm 1, with λ = 2L K and τ ≤ 1 6λ , where
L K = L f + ρ∥A∥ 2 + µ, we have E∥u * (x t , y t+1 , z t+1 ) -x t+1 ∥ 2 ≤ 1 - τ λ 4 E∥u * (x t , y t+1 , z t+1 ) -x t ∥ 2 + (τ µ + 2τ 2 µ 2 )E∥z t -z t+1 ∥ 2 + τ 2 σ 2 .
Proof. From the definition of u * (x t , y t+1 , z t+1 ) in (11) (see also ( 23)), we have
λ(x t -u * (x t , y t+1 , z t+1 )) ∈ ∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 ) + ∂I X (u * (x t , y t+1 , z t+1 )).
Multiplying both sides by the step size τ , adding u * (x t , y t+1 , z t+1 ) to both sides, and rearranging give
τ λx t -τ ∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 ) + (1 -τ λ)u * (x t , y t+1 , z t+1 ) ∈ u * (x t , y t+1 , z t+1 ) + τ ∂I X (u * (x t , y t+1 , z t+1 )).
Since (I + τ ∂I X ) -1 = prox I X = proj X due to ∂I X being a cone and proximal operator of a normal cone being the projection to the set, we have the first assertion.
We next establish the second assertion. Using the just established identity (26), the update rule of x t+1 in Algorithm 1, and nonexpansiveness of the projection, we derive
∥u * (x t , y t+1 , z t+1 ) -x t+1 ∥ 2 ≤ ∥τ λx t + (1 -τ λ)u * (x t , y t+1 , z t+1 ) -τ ∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 ) -[x t -τ G(x t , y t+1 , z t , ξ t )]∥ 2 .
We add and subtract ∇ x K(x t , y t+1 , z t ) inside the squared norm on the right-hand side, expand and take conditional expectation to obtain
E t ∥u * (x t , y t+1 , z t+1 ) -x t+1 ∥ 2 = ∥(1 -τ λ)(u * (x t , y t+1 , z t+1 ) -x t ) -τ ∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 ) + τ ∇ x K(x t , y t+1 , z t )∥ 2 + τ 2 E t ∥G(x t , y t+1 , z t , ξ t ) -∇ x K(x t , y t+1 , z t )∥ 2 , (27
)
where the cross term disappeared because
E t [G(x t , y t+1 , z t , ξ t )] = ∇ x K(x t , y t+1 , z t )
and x t , y t+1 , z t+1 , u * (x t , y t+1 , z t+1 ) are deterministic under the conditioning since z t+1 defined in Algorithm 1 only depends on x t (that is, z t+1 is independent of ξ t ).
The second term on the right-hand side of ( 27) is trivially bounded by the oracle assumptions, that is,
E t ∥G(x t , y t+1 , z t , ξ t ) -∇ x K(x t , y t+1 , z t )∥ 2 ≤ σ 2 .(28)
For the first term on the right-hand side of ( 27), we further estimate as
∥(1 -τ λ)(u * (x t , y t+1 , z t+1 ) -x t ) -τ ∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 ) + τ ∇ x K(x t , y t+1 , z t )∥ 2 = (1 -τ λ) 2 ∥u * (x t , y t+1 , z t+1 ) -x t ∥ 2 + 2τ (1 -τ λ)⟨u * (x t , y t+1 , z t+1 ) -x t , ∇ x K(x t , y t+1 , z t ) -∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 )⟩ + τ 2 ∥∇ x K(x t , y t+1 , z t ) -∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 )∥ 2 .(29)
Next, we turn to estimating
∥∇ x K(x t , y t+1 , z t ) -∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 )∥ ≤ ∥∇ x K(x t , y t+1 , z t ) -∇ x K(x t , y t+1 , z t+1 )∥ + ∥∇ x K(x t , y t+1 , z t+1 ) -∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 )∥.(30)
Note that, by definition, we have
∇ x K(x t , y t+1 , z t ) -∇ x K(x t , y t+1 , z t+1 ) = µ(z t+1 -z t ).
Using this and the L K -Lipschitzness of ∇ x K(•, y t+1 , z t+1 ) as per Fact A.1, in (30), we obtain
∥∇ x K(x t , y t+1 , z t ) -∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 )∥ ≤ µ∥z t+1 -z t ∥ + L K ∥u * (x t , y t+1 , z t+1 ) -x t ∥.
We plug this bound into the second term on the right-hand side of (29) after using Cauchy-Schwarz inequality, and then, we use Young's inequality to get
2τ (1 -τ λ)⟨u * (x t , y t+1 , z t+1 ) -x t , ∇ x K(x t , y t+1 , z t ) -∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 )⟩ ≤ 2τ (1 -τ λ)∥u * (x t , y t+1 , z t+1 ) -x t ∥(µ∥z t+1 -z t ∥ + L K ∥u * (x t , y t+1 , z t+1 ) -x t ∥) ≤ τ (1 -τ λ)(2L K + µ)∥u * (x t , y t+1 , z t+1 ) -x t ∥ 2 + τ (1 -τ λ)µ∥z t+1 -z t ∥ 2 .
Using the last two inequalities in (29), along with Young's inequality, we obtain
∥(1 -τ λ)(u * (x t , y t+1 , z t+1 ) -x t ) -τ ∇ x K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 ) + τ ∇ x K(x t , y t+1 , z t )∥ 2 ≤ [(1 -τ λ) 2 + τ (1 -τ λ)(2L K + µ) + 2τ 2 L 2 K ]∥u * (x t , y t+1 , z t+1 ) -x t ∥ 2 + (τ (1 -τ λ)µ + 2τ 2 µ 2 )∥z t+1 -z t ∥ 2 . (31
)
We estimate the coefficient of the first term. First, note that 1 -τ λ ≤ 1. As a result, we have
(1 -τ λ) 2 + τ (1 -τ λ)(2L K + µ) + 2τ 2 L 2 K ≤ 1 -2τ λ + τ 2 λ 2 + τ (2L K + µ) + 2τ 2 L 2 K ≤ 1 -2τ λ + 1 6 τ λ + τ λ + 1 2 τ λ + 1 12 τ λ = 1 - τ λ 4 ,
where in second inequality, we use τ λ
≤ 1 6 ,L K = 1 2 λ and τ µ ≤ τ L K = 1 2 τ λ. Finally, since τ (1 -τ λ)µ + 2τ 2 µ 2 ≤ τ µ + 2τ 2 µ 2
, the proof is completed after taking full expectation of ( 27) and plugging in (28) and ( 31).
this section cite: ['b7']

Section: ■
Lemma A.5 (cf. Lemma 3.3). Let Assumption 1.1 hold. Then, if λ = 2L K and τ ≤ 1 6λ , we have for the iterates of Algorithm 1 that
Eφ 1/λ (x t+1 , y t+1 , z t+1 ) ≤ Eφ 1/λ (x t , y t+1 , z t+1 ) - τ λ 2 16 E∥u * (x t , y t+1 , z t ) -x t ∥ 2 + λτ µ 2 + λτ 2 µ 2 + τ λ 2 µ 2 8γ 2 s E∥z t -z t+1 ∥ 2 + λτ 2 σ 2 2 ,(32)
where γ s = 2µ + ρ∥A∥.
Proof. By the definition of φ 1/λ from ( 24) and u * (x, y t+1 , z t+1 ) from ( 23), we have
Eφ 1/λ (x t+1 , y t+1 , z t+1 ) ≤ EK(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 ) + λ 2 E∥u * (x t , y t+1 , z t+1 ) -x t+1 ∥ 2 ≤ EK(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 ) + λ 2 - τ λ 2 8 E∥u * (x t , y t+1 , z t+1 ) -x t ∥ 2 + λτ µ 2 + λτ 2 µ 2 E∥z t -z t+1 ∥ 2 + λτ 2 σ 2 2 = Eφ 1/λ (x t , y t+1 , z t+1 ) - τ λ 2 8 E∥u * (x t , y t+1 , z t+1 ) -x t ∥ 2 + λτ µ 2 + λτ 2 µ 2 E∥z t -z t+1 ∥ 2 + λτ 2 σ 2 2 . (33
)
We next bound the second term on the right-hand side by using Young's inequality as
∥u * (x t , y t+1 , z t+1 ) -x t ∥ 2 ≥ 1 2 ∥u * (x t , y t+1 , z t ) -x t ∥ 2 -∥u * (x t , y t+1 , z t+1 ) -u * (x t , y t+1 , z t ∥ 2 ≥ 1 2 ∥u * (x t , y t+1 , z t ) -x t ∥ 2 - µ 2 γ 2 s ∥z t -z t+1 ∥ 2 ,(34)
where the last line used ( 61).
We substitute the last inequality into (33) to conclude. ■
Since the previous result only allowed us to connect φ 1/λ (x t+1 , y t+1 , z t+1 ) to φ 1/λ (x t , y t+1 , z t+1 ), we now need to analyze the effect of changing y t+1 and z t+1 in φ 1/λ . The main idea of this lemma is similar to (Zhang & Luo, 2022), where the difference lies in the fact that our potential involves the Moreau envelope of K(x, y, z) whereas the potential of (Zhang & Luo, 2022) involves K(x, y, z). Hence this work considers the change of the arguments in the function K instead of φ 1/λ . Therefore, our proof uses the properties of the Moreau envelope which was not needed in (Zhang & Luo, 2022).
Lemma A.6. (cf. Lemma 3.4) Suppose that Assumption 1.1 holds, for φ 1/λ defined in (10), we have for the iterates of Algorithm 1 that
φ 1/λ (x t , y t , z t ) -φ 1/λ (x t , y t+1 , z t ) ≥ ⟨y t -y t+1 , Au * (x t , y t , z t ) -b⟩ + γ s 2 ∥u * (x t , y t , z t ) -u * (x t , y t+1 , z t )∥ 2 , φ 1/λ (x t , y t+1 , z t ) -φ 1/λ (x t , y t+1 , z t+1 ) ≥ µ 2 ⟨z t+1 -z t , 2u * (x t , y t+1 , z t ) -z t+1 -z t ⟩ + γ s 2 ∥u * (x t , y t+1 , z t+1 ) -u * (x t , y t+1 , z t )∥ 2 ,
where γ s = 2µ + ρ∥A∥.
Proof. We first consider the change in y argument of φ 1/λ . By using the definition of φ 1/λ in (24), we have
φ 1/λ (x t , y t , z t ) -φ 1/λ (x t , y t+1 , z t ) = K(u * (x t , y t , z t ), y t , z t ) + λ 2 ∥x t -u * (x t , y t , z t )∥ 2 -K(u * (x t , y t+1 , z t ), y t+1 , z t ) - λ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 = K(u * (x t , y t , z t ), y t , z t ) -K(u * (x t , y t , z t ), y t+1 , z t ) + K(u * (x t , y t , z t ), y t+1 , z t ) + λ 2 ∥x t -u * (x t , y t , z t )∥ 2 -K(u * (x t , y t+1 , z t ), y t+1 , z t ) - λ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 ,(35)
where the second equality adds and subtracts K(u * (x t , y t , z t ), y t+1 , z t ).
From the definition of K in ( 22), it trivially follows that K(u * (x t , y t , z t ), y t , z t ) -K(u * (x t , y t , z t ), y t+1 , z t ) = ⟨y t -y t+1 , Au * (x t , y t , z t ) -b⟩.
Next, we use the property that K(•, y t+1 , z t ) + λ 2 ∥ • -x t ∥ 2 is γ s -strongly convex with minimizer u * (x t , y t+1 , z t ) (see Fact A.1 and ( 23)) to obtain
K(u * (x t , y t , z t ), y t+1 , z t ) + λ 2 ∥x t -u * (x t , y t , z t )∥ 2 -K(u * (x t , y t+1 , z t ), y t+1 , z t ) - λ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 ≥ γ s 2 ∥u * (x t , y t , z t ) -u * (x t , y t+1 , z t )∥ 2 .
Combining the last two estimates in (35) gives the first assertion.
Next, we analyze the effect of changing the z component in φ 1/λ . Similar to the proof of the first assertion, we start with the definition of φ 1/λ and then add and subtract K(u * (x t , y t+1 , z t+1 ) to obtain
φ 1/λ (x t , y t+1 , z t ) -φ 1/λ (x t , y t+1 , z t+1 ) = K(u * (x t , y t+1 , z t ), y t+1 , z t ) + λ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 -K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 ) - λ 2 ∥x t -u * (x t , y t+1 , z t+1 )∥ 2 = K(u * (x t , y t+1 , z t ), y t+1 , z t ) -K(u * (x t , y t+1 , z t ), y t+1 , z t+1 ) + K(u * (x t , y t+1 , z t ), y t+1 , z t+1 ) + λ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 -K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 ) - λ 2 ∥x t -u * (x t , y t+1 , z t+1 )∥ 2 .(36)
First, by definition, of K, it trivially follows that
K(u * (x t , y t+1 , z t ), y t+1 , z t ) -K(u * (x t , y t+1 , z t ), y t+1 , z t+1 ) = µ 2 ∥u * (x t , y t+1 , z t ) -z t ∥ 2 - µ 2 ∥u * (x t , y t+1 , z t ) -z t+1 ∥ 2 .
For the remaining terms on the right-hand side, we again use that K(
•, y t+1 , z t+1 ) + λ 2 ∥ • -x t ∥ 2 is γ s -strongly convex with minimizer u * (x t , y t+1 , z t+1 ) to deduce K(u * (x t , y t+1 , z t ), y t+1 , z t+1 ) + λ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 -K(u * (x t , y t+1 , z t+1 ), y t+1 , z t+1 ) - λ 2 ∥x t -u * (x t , y t+1 , z t+1 )∥ 2 ≥ γ s 2 ∥u * (x t , y t+1 , z t+1 ) -u * (x t , y t+1 , z t )∥ 2 .
Plugging in the last two estimates in (36) gives the second assertion. ■ Corollary A.7. Suppose that Assumption 1.1 holds, for φ 1/λ defined in (10), if λ = 2L K and τ ≤ 1 6λ , we have that
Eφ 1/λ (x t , y t , z t ) -Eφ 1/λ (x t+1 , y t+1 , z t+1 ) ≥ τ λ 2 16 E∥u * (x t , y t+1 , z t ) -x t ∥ 2 - λτ µ 2 + λτ 2 µ 2 + τ λ 2 µ 2 8γ 2 s E∥z t -z t+1 ∥ 2 - λτ 2 σ 2 2 -ηE⟨Ax t -b, Au * (x t , y t , z t ) -b⟩ + µ 2 E⟨z t+1 -z t , 2u * (x t , y t+1 , z t ) -z t+1 -z t ⟩,
where γ s = 2µ + ρ∥A∥.
where the last inequality is due to β ≤ 1.
Next, for the remaining inner products in (38), we have
µ⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩ -2µ⟨z t+1 -z t , z t -x * (z t )⟩ = µ∥z t+1 -z t ∥ 2 + 2µ⟨z t+1 -z t , x * (z t ) -x * (y t+1 , z t+1 )⟩.(39)
We can use Cauchy-Schwarz, triangle and Young's inequalities on the second term here to get
⟨z t+1 -z t , x * (z t ) -x * (y t+1 , z t+1 )⟩ ≥ -∥z t+1 -z t ∥(∥x * (z t ) -x * (y t+1 , z t )∥ + ∥x * (y t+1 , z t ) -x * (y t+1 , z t+1 )∥) ≥ - 1 2ζ + 1 σ 4 ∥z t+1 -z t ∥ 2 - ζ 2 ∥x * (z t ) -x * (y t+1 , z t )∥ 2 ,
where the last step also used (63). Consequently, plugging in this estimate to ( 39), we obtain
µ⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩ -2µ⟨z t+1 -z t , z t -x * (z t )⟩ ≥ µ - µ ζ - 2µ σ 4 ∥z t+1 -z t ∥ 2 -µζ∥x * (z t ) -x * (y t+1 , z t )∥ 2 .
After combining these estimates in (38), we get
E[V t ] -E[V t+1 ] ≥ τ λ 2 16 E∥u * (x t , y t+1 , z t ) -x t ∥ 2 - 1 2 λτ µ + λτ 2 µ 2 + τ λ 2 µ 2 8γ 2 s + µ ζ + 3µ σ 4 -µ - µ 2β E∥z t -z t+1 ∥ 2 - 1 2 λτ 2 σ 2 -ηE⟨Ax t -b, Au * (x t , y t , z t ) -Ax t ⟩ -ηE∥Ax t -Ax * (y t+1 , z t )∥ 2 + ηE∥Ax * (y t+1 , z t ) -b∥ 2 -µζE∥x * (z t ) -x * (y t+1 , z t )∥ 2 + µE⟨z t+1 -z t , u * (x t , y t+1 , z t ) -x t ⟩.(40)
We will now operate on some of terms from the right-hand side of (40), by using Lemma A.11 and A.12. First, we have by Cauchy-Schwarz and Young's inequalities that
-η⟨Ax t -b, Au * (x t , y t , z t ) -Ax t ⟩ ≥ - η 4 ∥Ax t -b∥ 2 -η∥Au * (x t , y t , z t ) -Ax t ∥ 2 ≥ - η 4 ∥Ax t -b∥ 2 -2η∥Au * (x t , y t , z t ) -Au * (x t , y t+1 , z t )∥ 2 -2η∥Au * (x t , y t+1 , z t ) -Ax t ∥ 2 .
Next, by using the Lipschitzness of u * (x t , •, z t ) from ( 60), we have
∥Au * (x t , y t , z t ) -Au * (x t , y t+1 , z t )∥ 2 ≤ ∥A∥ 2 ∥u * (x t , y t , z t ) -u * (x t , y t+1 , z t )∥ 2 ≤ ∥A∥ 4 γ 2 s ∥y t -y t+1 ∥ 2 = ∥A∥ 4 η 2 γ 2 s ∥Ax t -b∥ 2 ,
where the last step also used the definition of y t+1 . Using this estimation along with (66) gives
-η⟨Ax t -b, Au * (x t , y t , z t ) -Ax t ⟩ ≥ - η 4 + 2∥A∥ 4 η 3 γ 2 s ∥Ax t -b∥ 2 -2η∥A∥ 2 ∥u * (x t , y t+1 , z t ) -x t ∥ 2 ≥ - η∥A∥ 2 λ 2 2γ 2 + 4∥A∥ 6 η 3 λ 2 γ 2 γ 2 s + 2η∥A∥ 2 ∥u * (x t , y t+1 , z t ) -x t ∥ 2 - η 2 + 4∥A∥ 4 η 3 γ 2 s ∥Ax * (y t+1 , z t ) -b∥ 2 .
We next have by Young's inequality that for any θ > 0:
µ⟨z t+1 -z t , u * (x t , y t+1 , z t ) -x t ⟩ ≥ - µ 4θ ∥z t+1 -z t ∥ 2 -θµ∥u * (x t , y t+1 , z t ) -x t ∥ 2 .
The inequality derived in (65) directly implies
-η∥Ax t -Ax * (y t+1 , z t )∥ 2 ≥ - η∥A∥ 2 λ 2 γ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 .
The key global error bound given in Lemma A.12 originally proved in (Zhang & Luo, 2022) results in
-6µβ∥x * (y t+1 , z t ) -x * (z t )∥ 2 ≥ -6µβ σ2 ∥Ax * (y t+1 , z t ) -b∥ 2 .
Combining these estimates in (40) leads to
E[V t ] -E[V t+1 ] ≥ - 1 2 λτ µ + λτ 2 µ 2 + τ λ 2 µ 2 8γ 2 s + µ ζ + 3µ σ 4 -µ - µ 2β + µ 4θ E∥z t -z t+1 ∥ 2 - 1 2 λτ 2 σ 2 + τ λ 2 16 - 3∥A∥ 2 λ 2 η 2γ 2 - 4∥A∥ 6 η 3 λ 2 γ 2 s γ 2 -2η∥A∥ 2 -µθ E∥u * (x t , y t+1 , z t ) -x t ∥ 2 + η 2 - 4∥A∥ 4 η 3 γ 2 s -6µβ σ2 E∥Ax * (y t+1 , z t ) -b∥ 2 . (41
)
We now estimate the coefficients inside the parantheses, with straightforward but tedious calculations which follow from the parameter settings.
First, we estimate the coefficient of
E∥z t -z t+1 ∥ 2 in (41): Let µ ≥ 4L f , we have σ 4 ≥ 1 2 because σ 4 = µ-L f µ . Then letting ζ = 6β, β < 1 30 , we have µ - 3µ σ 4 ≥ -5µ ≥ - µ 6β , µ ζ = µ 6β .
Therefore, we have that
µ 2β + µ - 3µ σ 4 - µ ζ ≥ 1 2 - 1 6 - 1 6 µ β ≥ µ 6β . (42
)
Hence, we estimate:
coefficient of E∥z t -z t+1 ∥ ≥ - 1 2 λτ µ -λτ 2 µ 2 - τ λ 2 µ 2 8γ 2 s + µ 6β - µ 8β . Let η = η ′ 2∥A∥ 2 , θ = 2β, η ′ ≤ 1 40 , and µ = max{2, 4L f }, λ = 2L K = 2(L f + ρ∥A∥ + µ), τ ≤ 1 10λ 2 , and γ s = µ -L f + γ from Fact A.1. We have -λτ µ ≥ -µ 10 and -2λτ 2 µ 2 ≥ -µ 100 , then coefficient of E∥z t -z t+1 ∥ ≥ µ 24β - µ 20 - µ 100 -τ λ 2 µ 2 (µ -L f + λ) 2 . By β ≤ 1/30, we have 1 24β -1 20 -1 100 ≥ 1 30β . In addition, using τ λ 2 µ 2 (µ-L f +λ) 2 ≤ τ λ 2 ≤ 1 10 , we fanally obtain: coefficient of E∥z t -z t+1 ∥ ≥ µ 30β - 1 10 µ≥2 ≥ µ 50β . (43
)
Then we estimate the coefficient of E∥u * (x t , y t+1 , z t ) -x t ∥ 2 in (41).
From above assumptions, we can easily get γ =
(µ-L f )λ µ-L f +λ ≥ 1 2 because λ ≥ µ ≥ 2. Moreover, we assume η ′ ≤ τ 40 , η ′ µ-L f +λ ≤ τ 10 , β ≤ τ 40 First, by our new notations, we have coefficient of E∥u * (x t , y t+1 , z t ) -x t ∥ 2 = τ λ 2 16 - 3η ′ λ 2 4γ 2 - η ′3 λ 2 2γ 2 γ 2 s -η ′ -2µβ
By γ ≥ 1 2 and the definition of γ s , we have
-3η ′ λ 2 4γ 2 ≥ -3η ′ λ 2 , -η ′3 λ 2 2γ 2 γ 2 s ≥ -η ′3 λ 2 (µ-L f +λ) 2 , Then coefficient of E∥u * (x t , y t+1 , z t ) -x t ∥ 2 ≥ τ λ 2 16 -3η ′ λ 2 - 2η ′3 λ 2 (µ -L f + λ) 2 -η ′ -2µβ. With 2 ≤ µ ≤ λ, η ′ ≤ τ 100 , η ′ µ-L f +λ ≤ τ 10 , β ≤ τ 200 , we can obtain -3η ′ λ 2 ≥ -3τ λ 2 400 , -2η ′3 λ 2 (µ-L f +λ) 2 ≥ -λ 2 τ 2 400 ≥ -λ 2 τ 400 , -η ′ ≥ -τ 100 ≥ -τ λ 2 100 , -2µβ ≥ -τ µ 50 µ≤λ ≥ -τ λ 50 ≥ -τ λ 2 100 . Hence, coefficient of E∥u * (x t , y t+1 , z t ) -x t ∥ 2 ≥ τ λ 2 16 - 3τ λ 2 100 - τ λ 2 400 - τ λ 2 400 - τ λ 2 100 = 7τ λ 2 400 .(44)
Last, we estimate the coefficient of E∥Ax * (y t+1 , z t ) -b∥ 2 in (41). By 6µβ σ2 ≤ η 6 and the definition η ′ , γ s , we have
-4∥A∥ 2 η 3 γ 2 s = -η ′2 η (µ-L f +λ) 2 η ′ µ-L f +λ ≤ τ 10 ≥ -ητ 2 100 ≥ -η 100 and -6µβ σ2 ≥ -η 6 . Hence, we have coefficient of E∥Ax * (y t+1 , z t ) -b∥ 2 ≥ η 2 - η 100 - η 6 ≥ η 4 .(45)
Plugging ( 43), ( 44) and ( 45) to ( 41), we finish the proof. ■
this section cite: []

Section: A.2. Proof of Theorem 3.1
Proof of Theorem 3.1. We start from the result in Lemma A.9. First, it follows from the definition of z t+1 that
∥z t -z t+1 ∥ = β∥x t -z t ∥.
So, we rewrite (37), as:
EV t -EV t+1 ≥ β 2 c β E∥x t -z t ∥ 2 + c τ E∥u * (x t , y t+1 , z t ) -x t )∥ 2 + c η E∥Ax * (y t+1 , z t ) -b∥ 2 - 1 2 λτ 2 σ 2 . (46
)
For t > 0, we have V t ≥ f , which is proven in Lemma A.13. It then follows that
T -1 t=0 (EV t -EV t+1 ) = V 0 -EV T ≤ V 0 -f .(47)
Then, summing up (46), using ( 47), and the fact that c τ = Θ(τ ), c η = Θ(τ ), β 2 c β = Θ(τ ) from ( 25), we have
V 0 -f + 1 2 T λτ 2 σ 2 ≥ T t=1 C 0 τ E∥x t -z t ∥ 2 + E∥u * (x t , y t+1 , z t ) -x t ∥ 2 + E∥Ax * (y t+1 , z t ) -b∥ 2 ,
for some explicit constant C 0 .
Dividing both sides by T , rearranging and using the definition τ = 1 6λ 2 √ T gives
1 T T -1 t=0 E∥x t -z t ∥ 2 + E∥u * (x t , y t+1 , z t ) -x t ∥ 2 + E∥Ax * (y t+1 , z t ) -b∥ 2 ≤ 1 C 0 √ T 6λ(V 0 -f ) + σ 2 12 . (48
)
Since we have ∇Ψ(z t ) = µ(z t -x * (z t )), by Danskin's theorem, we deduce for any t
1 µ 2 ∥∇Ψ(z t )∥ = ∥z t -x * (z t )∥ ≤ ∥z t -x * (y t+1 , z t )∥ + ∥x * (y t+1 , z t ) -x * (z t )∥ ≤ ∥z t -x * (y t+1 , z t )∥ + σ∥Ax * (y t+1 , z s ) -b∥ ≤ ∥z t -x t ∥ + ∥x t -x * (y t+1 , z t )∥ + σ∥Ax * (y t+1 , z t ) -b∥ ≤ ∥z t -x t ∥ + λ γ ∥x t -u * (x t , y t+1 , z t )∥ + σ∥Ax * (y t+1 , z t ) -b∥,
where the first inequality is by triangle inequality, the second by (A.12), the third by triangle inequality and the fourth by ( 58).
Next, we take square of both sides, take expectation, use Young's inequality, sum for all t = 0, 1, . . . , T -1, divide by T and use (48) to derive
1 µ 2 1 T T -1 t=0 E∥∇Ψ(z t )∥ 2 ≤ 1 T T -1 t=0 E 3∥z t -x t ∥ 2 + 3λ 2 γ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 + 3σ 2 ∥Ax * (y t+1 , z t ) -b∥ 2 = O 1 √ T .
The result then follows since t * is selected uniformly at random from {0, 1, 2, . . . , T -1}. ■
this section cite: []

Section: A.3. Proof of Corollary 3.2
Proof of Corollary 3.2. From the definition of x, we have
0 ∈ Ĝ(x t , y t+1 , z t ) + 2 τ (x -x t ) + ∂I X (x). Let us set v = ∇ x K(x, y t+1 , z t ) -Ĝ(x t , y t+1 , z t ) - 2 τ (x -x t ) -ρA T (Ax -b) -µ(x -z t ).(49)
Combining with the optimality condition, we have
v ∈ ∇ x K(x, y t+1 , z t ) -ρA T (Ax -b) -µ(x -z t ) + ∂I X (x) = ∇f (x) + A T y t+1 + ∂I X (x).
Hence, we need to estimate E∥Ax -b∥ and E∥v∥.
For the mini-batch gradient in the post-processing step, we have
E∥ Ĝ(x, y, z) -∇K(x, y, z)∥ 2 ≤ σ 2 B .(50)
which is a standard calculation, see for example, (Lan, 2020, Section 5.2.3). Since B = Θ(ε -2 ), this gives us
E∥ Ĝ(x, y, z) -∇K(x, y, z)∥ 2 ≤ ε 2 .(51)
First, let us note that the purpose of x is to estimate u * (x t , y t+1 , z t ), where
u * (x t , y t+1 , z t ) = arg min u∈X {l(u) := K(u, y t+1 , z t ) + λ 2 ∥x t -u∥ 2 }.
Note that the gradient of this objective is
∇l(u) = ∇ x K(x, y t+1 , z t ) + λ(x -x t ).
As a result, we have ∇l(x t ) = ∇ x K(x t , y t+1 , z t ).
Let us also denote
x * t = proj X (x t -τ ∇l(x t )).
That is, x * t is the output of doing a full-gradient step on x t . Of course, this is not tractable in our setting, but we only use this as a theoretical tool.
Since this is a GD step on the objective l which is L K -smooth and convex with optimizer u * (x t , y t+1 , z t ), the standard analysis for GD gives
∥x * t -u * (x t , y t+1 , z t )∥ 2 ≤ ∥x t -u * (x t , y t+1 , z t )∥ 2 ,(52)
as long as τ ≤ 1 L K . Next, by the definitions of x * t and x, along with nonexpansiveness of the projection, we have
E∥x * t -x∥ 2 ≤ Eτ 2 ∥ Ĝ(x t , y t+1 , z t+1 ) -∇ x K(x t , y t+1 , z t )∥ 2 ≤ τ 2 ε 2 ,(53)
where the second inequality used (51).
In view of (49), we estimate ∥v∥ as
∥v∥ ≤ ∥∇ x K(x t , y t+1 , z t ) -Ĝ(x t , y t+1 , z t )∥ + L K ∥x t -x∥ + 2 τ ∥x -x t ∥ + ρ∥A∥∥Ax -b∥ + µ∥x -z t ∥.
On this, multiple applications of triangle inequality give
∥x -x t ∥ ≤ ∥x -x * t ∥ + ∥x * t -u * (x t , y t+1 , z t )∥ + ∥u * (x t , y t+1 , z t ) -x t ∥ ≤ ∥x -x * t ∥ + 2∥u * (x t , y t+1 , x t ) -x t ∥,(54)
where the second line is due to ( 52).
For the feasibility, we have by triangle inequality that
∥x -z t ∥ ≤ ∥x -x t ∥ + ∥x t -z t ∥.(55)
As a result, we have that
∥v∥ = O ∥x -x * t ∥ + ∥x t -u * (x t , y t+1 , z t )∥ + ∥Ax -b∥ + ∥x t -z t ∥ + ∥∇ x K(x t , y t+1 , z t ) -Ĝ(x t , y t+1 , z t )∥ .(56)
For the feasibility, we have
∥Ax -b∥ ≤ ∥Ax -Ax t ∥ + ∥Ax t -b∥ ≤ ∥A∥∥x -x t ∥ + ∥Ax t -b∥.
Now, by invoking the above inequality for t = t * , taking expectation, using Young's inequality, (54), ( 53) and ( 48) along with (66), we get that
E∥Ax -b∥ 2 ≤ ε 2 ,(57)
since T = Ω(ε -4 ).
Finally, using t = t * , taking square and then expectation of (56), using Young's inequality and then combining (57), ( 53), ( 51) and ( 48) gives the result since T = Ω(ε -4 ). ■
this section cite: []

Section: A.4. Auxiliary Results
Lemma A.10. Under Assumption 1.1, for any x, z, z ′ ∈ X, we have
λ γ ∥x -u * (x, y, z)∥ ≥ ∥x -x * (y, z)∥,(58)
∥u * (x, y, z) -x∥ ≤ ∥x -x * (y, z)∥,(59)
∥u * (x, y, z) -u * (x, y ′ , z)∥ ≤ ∥A∥ γ s ∥y -y ′ ∥,(60)
∥u * (x, y, z) -u * (x, y, z ′ )∥ ≤ µ γ s ∥z -z ′ ∥,(61)
∥z ′ -z∥ ≥ µ -L f µ ∥x * (y, z ′ ) -x * (y, z)∥,(62)
∥y ′ -y∥ ≥ γ K ∥A∥ ∥x * (y ′ , z) -x * (y, z)∥,(63)
∥x * (z) -x * (z ′ )∥ ≤ µ µ -L f ∥z -z ′ ∥,(64)
where γ = (µ-L f )λ µ-L f +λ , γ s = µ -L f + λ, γ K = µ -L f .
Proof. The proofs for (62), (63), and (64) appear in (Zhang & Luo, 2022), so we omit these proofs.
We first prove (58). Let us note that x * (y, z) minimizes φ 1/λ , see for example (Hiriart-Urruty & Lemarechal, 1993, Theorem XV4.1.7). As a result, we have
∇ x φ 1/λ (x * (y, z), y, z) = 0. From Lemma A.3, we have that φ 1/λ (•, y, z) is γ = (µ-L f )λ µ-L f +λ
-strongly convex. Then, by strong convexity, we have
⟨∇ x φ 1/λ (x * (y, z), y, z) -∇ x φ 1/λ (x, y, z), x * (y, z) -x⟩ ≥ γ∥x -x * (y, z)∥ 2 ⇐⇒ ∥∇ x φ 1/λ (x, y, z)∥ ≥ γ∥x -x * (y, z)∥,
where the inclusion used ∇ x φ 1/λ (x * (y, z), y, z) = 0 established in the previous paragraph as well as Cauchy-Schwarz inequality. Then, using ∇ x φ 1/λ (x, y, z) = λ(x -u * (x, y, z)), we obtain (58).
From definition of u * (x, y, z), we have,
K(u * (x, y, z), y, z) + λ 2 ∥x -u * (x, y, z)∥ 2 ≤ K(x * (y, z), y, z) λ 2 ∥x -x * (y, z)∥ 2 ,
where we also remark that x * (y, z) ∈ X. Combining with K(x * (y, z), y, z) ≤ K(u * (x, y, z), y, z), which follows from the definition of x * (y, z) we have (59).
The proofs of the other two assertions will use a similar idea to (Zhang & Luo, 2022), but there will be differences in the estimations since this previous work did not use the function φ 1/λ .
For (60), we proceed by using the definition of φ 1/λ and adding and subtracting K(u * (x, y ′ , z), y, z) to get
K(u * (x, y, z), y, z) + λ 2 ∥u * (x, y, z) -x∥ 2 -K(u * (x, y ′ , z), y ′ , z) - λ 2 ∥u * (x, y ′ , z) -x∥ 2 = K(u * (x, y, z), y, z) + λ 2 ∥u * (x, y, z) -x∥ 2 -K(u * (x, y ′ , z), y, z) - λ 2 ∥u * (x, y ′ , z) -x∥ 2 + K(u * (x, y ′ , z), y, z) -K(u * (x, y ′ , z), y ′ , z) ≤ -γ s 2 ∥u * (x, y, z) -u * (x, y ′ , z)∥ 2 + ⟨y -y ′ , Au * (x, y ′ , z) -b⟩,
where last step uses u → K(u, y, z) + λ 2 ∥u -x∥ 2 being γ s -strongly convex (cf. Fact A.1) with minimizer u * (x, y, z), as well as the definition of K.
We then argue similarly, this time adding and subtracting K(u * (x, y, z), y ′ , z):
K(u * (x, y, z), y, z) + λ 2 ∥u * (x, y, z) -x∥ 2 -K(u * (x, y ′ , z), y ′ , z) - λ 2 ∥u * (x, y ′ , z) -x∥ 2 = K(u * (x, y, z), y ′ , z) + λ 2 ∥u * (x, y, z) -x∥ 2 -K(u * (x, y ′ , z), y ′ , z) - λ 2 ∥u * (x, y ′ , z) -x∥ 2 -K(u * (x, y, z), y ′ , z) + K(u * (x, y, z), y, z) ≥ γ s 2 ∥u * (x, y, z) -u * (x, y ′ , z)∥ 2 + ⟨y -y ′ , Au * (x, y, z) -b⟩.
where last step uses that u → K(u, y ′ , z)
+ λ 2 ∥u -x∥ 2 is γ s -strongly convex (cf. Fact A.1
) with minimizer u * (x, y ′ , z) and the definition of K.
Combining the last two estimates give
⟨y -y ′ , Au * (x, y ′ , z) -Au * (x, y, z)⟩ ≥ γ s ∥u * (x, y, z) -u * (x, y ′ , z)∥ 2 .
this section cite: []

Section: Using Cauchy-Schwarz inequality and the definition of operator norm gives (60).
The proof of ( 61) is similar to the proof of (60), just completed. In particular, by adding and subtracting K(u * (x, y, z), y, z ′ ), we have
K(u * (x, y, z), y, z) + λ 2 ∥u * (x, y, z) -x∥ 2 -K(u * (x, y, z ′ ), y, z ′ ) + λ 2 ∥u * (x, y, z ′ ) -x∥ 2 = K(u * (x, y, z), y, z) + λ 2 ∥u * (x, y, z) -x∥ 2 -K(u * (x, y, z ′ ), y, z) - λ 2 ∥u * (x, y, z ′ ) -x∥ 2 -K(u * (x, y, z ′ ), y, z ′ ) + K(u * (x, y, z ′ ), y, z) ≤ - γ s 2 ∥u * (x, y, z) -u * (x, y, z ′ )∥ 2 + µ 2 (∥u * (x, y, z ′ ) -z∥ 2 -∥u * (x, y, z ′ ) -z ′ ∥ 2 ),
where we used that u → K(u, y, z) + λ 2 ∥u -x∥ 2 is γ s -strongly convex with minimizer u * (x, y, z) and the definition of K.
Finally, we add and subtract K(u * (x, y, z ′ ), y, z) to get
K(u * (x, y, z), y, z) + λ 2 ∥u * (x, y, z) -x∥ 2 -K(u * (x, y, z ′ ), y, z ′ ) - λ 2 ∥u * (x, y, z ′ ) -x∥ 2 = K(u * (x, y, z), y, z ′ ) + λ 2 ∥u * (x, y, z) -x∥ 2 -K(u * (x, y, z ′ ), y, z) - λ 2 ∥u * (x, y, z ′ ) -x∥ 2 + K(u * (x, y, z), y, z) -K(u * (x, y, z), y, z ′ ) ≥ γ s 2 ∥u * (x, y, z) -u * (x, y, z ′ )∥ 2 + µ 2 (∥u * (x, y, z) -z∥ 2 -∥u * (x, y, z) -z ′ ∥ 2 ),
where we used that u → K(u, y, z ′ ) + λ 2 ∥u -x∥ 2 is γ s -strongly convex with minimizer u * (x, y, z ′ ) and the definition of K.
Combining the last two inequalities give
µ⟨u * (x, y, z ′ ) -u * (x, y, z), z ′ -z⟩ ≥ γ s ∥u * (x, y, z) -u * (x, y, z ′ )∥ 2 .
this section cite: []

Section: Using Cauchy-Schwarz inequality concludes the proof. ■
Lemma A.11. Under Assumption 1.1, for x t , y t+1 , z t generated by Algorithm 1, we have
∥Ax t -Ax * (y t+1 , z t )∥ 2 ≤ ∥A∥ 2 λ 2 γ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 , (65
) ∥Ax t -b∥ 2 ≤ 2∥A∥ 2 λ 2 γ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 + 2∥Ax * (y t+1 , z t ) -b∥ 2 , (66
) ∥Au * (x t , y t , z t ) -Ax t ∥ 2 ≤ 2∥A∥ 4 γ 2 s ∥y t -y t+1 ∥ 2 + 2∥A∥ 2 ∥u * (x t , y t+1 , z t ) -x t ∥ 2 , (67
)
where γ, γ s are defined in (25).
Proof. The assertion in (65) follows directly from ( 58) since
∥Ax t -Ax * (y t+1 , z t )∥ 2 ≤ ∥A∥ 2 ∥x t -x * (y t+1 , z t )∥ 2 ≤ ∥A∥ 2 λ 2 γ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 .
Combining the first assertion with Young's inequality gives the second assertion, since
∥Ax t -b∥ 2 ≤ 2∥Ax t -Ax * (y t+1 , z t )∥ 2 + 2∥Ax * (y t+1 , z t ) -b∥ 2 ≤ 2∥A∥ 2 λ 2 γ 2 ∥x t -u * (x t , y t+1 , z t )∥ 2 + 2∥Ax * (y t+1 , z t ) -b∥ 2 .
Young's inequality and (60) gives the third assertion
∥Au * (x t , y t , z t ) -Ax t ∥ 2 ≤ 2∥Au * (x t , y t , z t ) -Au * (x t , y t+1 , z t )∥ 2 + 2∥Au * (x t , y t+1 , z t ) -Ax t ∥ 2 ≤ 2∥A∥ 4 γ 2 s ∥y t -y t+1 ∥ 2 + 2∥A∥ 2 ∥u * (x t , y t+1 , z t ) -x t ∥ 2 .
The proof is completed.
this section cite: []

Section: ■
The following important lemma is known as the global error bound in (Zhang & Luo, 2022).
This global result holds in its entirety in our case, so we only state it here and refer to where it appeared originally for the precise definition of the constant σ which depends on Hoffman constant of certain linear systems. Lemma A.12. (Zhang & Luo, 2022, Lemma 3.2) If µ > L f , then we have ∥x * (y, z) -x * (z)∥ ≤ σ∥Ax * (y, z) -b∥ for any y, z,
where σ > 0 depends only on the constants C 1 = (L f + ρ∥A∥ 2 + µ), C 2 = -L f + µ, and the matrices A, H and is always finite.
Lemma A.13. If (x, z) ∈ X × X, we have φ 1/λ (x, y, z) -2d(y, z) + 2Ψ(z) ≥ f .
Proof. Because x * (y, z) minimizes φ 1/λ (•, y, z) (see for example (Hiriart-Urruty & Lemarechal, 1993, Theorem XV4.1.7)), we have φ 1/λ (x, y, z) ≥ φ 1/λ (x * (y, z), y, z) = K(x * (y, z), y, z).
We can then deduce
φ 1/λ (x, y, z) -2d(y, z) + 2Ψ(z) ≥ K(x * (y, z), y, z) -2d(y, z) + 2Ψ(z) = d(y, z) -2d(y, z) + 2Ψ(z) = Ψ(z) + Ψ(z) -d(y, z) ≥ Ψ(z) ≥ f
Then we estimate as
E⟨η(Ax t -b), Ax * (y t , z t ) -b⟩ - 1 2 η 2 L 2 - ∥A∥ 4 2γ 2 K E∥y t+1 -y t ∥ 2 ≥ηE⟨(Ax t -b), Ax * (y t , z t ) -b⟩ - 1 2 + ∥A∥ 4 2γ 2 K η 2 L 2 =ηE[⟨(Ax t -b), Ax * (y t+1 , z t ) -b⟩ + η⟨(Ax t -b), -Ax * (y t+1 , z t ) + Ax * (y t , z t )⟩] - 1 2 + ∥A∥ 4 2γ 2 K η 2 L 2 ≥ηE[⟨(Ax t -b), Ax * (y t+1 , z t ) -b⟩ -8∥Ax * (y t+1 , z t ) -Ax * (y t , z t )∥ 2 ] - η 2 32 ∥Ax t -b∥ 2 - 1 2 + ∥A∥ 4 2γ 2 K η 2 L 2 ≥ηE⟨(Ax t -b), Ax * (y t+1 , z t ) -b⟩ - η 2 32 E∥Ax t -b∥ 2 - 1 2 η 2 L 2 - 17∥A∥ 4 2γ 2 K η 2 L 2 ,
where the first inequality comes from E[∥A ζt x t -b ζt ∥ 2 ] ≤ L and the second inequality comes from ⟨a, b⟩ ≤ 1 32 ∥a∥ 2 + 8∥b∥ 2 (∀a, b). And in last inequality we use (63) again.
The estimation of EΨ(z t+1 ) -EΨ(z t ) is the same as Lemma A.8. Because the randomness of ζ t in the stochastic dual update does not change the recursion in EΨ(z t+1 ) -EΨ(z t ), where z t , z t+1 only depend on the randomness before ζ t . Hence we omit the proof here. This completes the proof. ■ Lemma B.2. Let Assumption 1.1 and 4.1 hold. By using the parameters (25) in Algorithm 2, then in the iteration t + 1, if the dual update runs as y t+1 = y t + η(A ζt x t -b ζt ), we obtain
EV t -EV t+1 ≥ c β E∥z t+1 -z t ∥ 2 + c τ E∥u * (x t , y t+1 , z t ) -x t ∥ 2 + c η E∥Ax * (y t+1 , z t ) -b∥ 2 -λτ 2 σ 2 2 -1 + 17∥A∥ 4 γ 2 K η 2 L 2 , (69
)
where c β = µ 50β , c τ = 6τ λ 2 400 , c η = η 8 and E∥G(x t , y t , z t , ξ t ) -∇ x K(x t , y t , z t )∥ 2 ≤ σ 2 2 .
Proof. First, we show E∥G(x t , y t , z t , ξ t ) -∇ x K(x t , y t , z t )∥ 2 is bounded.
Recall that in Equation ( 17) we have
G(x, y, z, ξ) = ∇f (x, ξ) + A ⊤ ζ 1 y + ρA ⊤ ζ 1 (A ζ 2 x -b ζ 2 ) + µ(x -z).(70)
We estimate by using Young's inequalities
E∥G(x t , y t , z t , ξ t ) -∇ x K(x t , y t , z t )∥ 2 ≤ 2E∥G(x t , y t , z t , ξ t ) -G(x t , 0, z t , ξ t )∥ 2 + 2E∥G(x t , 0, z t , ξ t ) -∇ x K(x t , y t , z t )∥ 2 ≤ 2EL G ∥y t ∥ 2 + 2E∥G(x t , 0, z t , ξ t ) -∇ x K(x t , y t , z t )∥ 2 ≤ 2EL G ∥y t ∥ 2 + 4E∥G(x t , 0, z t , ξ t ) -∇ x K(x t , 0, z t )∥ 2 + 4E∥∇ x K(x t , 0, z t ) -∇ x K(x t , y t , z t )∥ 2 ≤ 2L G M 2 y + 4E∥G(x t , 0, z t , ξ t ) -∇ x K(x t , 0, z t )∥ 2 + 4∥A∥ 2 ∥y t ∥ 2 ,
where in second inequality we use L G is the Lipschitz constant of G with respect to variable y, then in third inequality we use M y as the upper bound of ∥y t ∥.
Because x t , y t , z t are all bounded, E∥G(x t , y t , z t , ξ t ) -∇ x K(x t , y t , z t )∥ 2 is bounded, we denote the upper bound as σ 2 2 . Note that Corollary A.7 still holds for x t , y t , z t , x t+1 , y t+1 , z t+1 , but the variance σ is changed to σ 2 (since this corollary and the lemmas used in its proof do not use the particular form of y t+1 ). Then combining with Lemma B.1, we have
E[V t -V t+1 ] = E φ 1/λ (x t , y t , z t ) -φ 1/λ (x t+1 , y t+1 , z t+1 ) + 2d(y t+1 , z t+1 ) -2d(y t , z t ) + 2Ψ(z t ) -2Ψ(z t+1 ) ≥ τ λ 2 16 E∥u * (x t , y t+1 , z t ) -x t ∥ 2 - λτ µ 2 + λτ 2 µ 2 + τ λ 2 µ 2 8γ 2 s E∥z t -z t+1 ∥ 2 - λτ 2 σ 2 2 2 -ηE⟨Ax t -b, Au * (x t , y t , z t ) -b⟩ + µ 2 E⟨z t+1 -z t , 2u * (x t , y t+1 , z t ) -z t -z t+1 ⟩ + 2ηE⟨Ax t -b, Ax * (y t+1 , z t ) -b⟩ + µE⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩ -2µE⟨z t+1 -z t , z t -x * (z t )⟩ - µ σ 4 E∥z t -z t+1 ∥ 2 - η 2 32 E∥Ax t -b∥ 2 - 1 2 + 17∥A∥ 4 2γ 2 K η 2 L 2 , where -η 2 32 E∥Ax t -b∥ 2 -1 2 + 17∥A∥ 4 2γ 2 K η 2 L 2
is the difference comparing to the deterministic linear constraints result in Lemma A.9. We then estimate like Lemma A.9 to have
EV t -EV t+1 ≥ c β E∥z t+1 -z t ∥ 2 + c τ E∥u * (x t , y t+1 , z t ) -x t ∥ 2 + c η E∥Ax * (y t+1 , z t ) -b∥ 2 - 1 2 λτ 2 σ 2 2 - η 2 16 E∥Ax t -b∥ 2 -1 + 17∥A∥ 4 γ 2 K η 2 L 2 ,(71)
where c β = µ 50β , c τ = 7τ λ 2 400 , c η = η 4 . We also have by Young's inequality and Lemma A.11 that
- η 2 16 E∥Ax t -b∥ 2 ≥ - ∥A∥ 2 λ 2 η 2 8γ 2 E∥x t -u * (x t , y t+1 , z t )∥ 2 - η 2 8 E∥Ax(y t+1 , z t ) -b∥ 2 .
By the parameter choices, we have
7τ λ 2 400 -∥A∥ 2 λ 2 η 2 8γ 2 ≥ 6τ λ 2 400 and η 4 -η 2 8 ≥ η 8 . Using these estimations in (71) gives the proof. ■ Proposition B.3. Under Assumption 4.1, ∥y t ∥ ≤ Ψ(zt)-d(yt,zt)+2M r
, where M = max x,z∈X {|f (x)| + µ 2 ∥x -z∥ 2 + ρ 2 ∥Ax -b∥ 2 } and r > 0 is defined as ∥Ax -b∥ = r where x is in the relative interior of the constraints. The existence of this is guaranteed by our assumption.
Proof. Given x ∈ X, we have
Ψ(z t ) -d(y t , z t ) ≥ f (x * (z t )) + µ 2 ∥x * (z t ) -z t ∥ 2 -K( x, y t , z t ) ≥ f (x * (z t )) + µ 2 ∥x * (z t ) -z t ∥ 2 -[f ( x) + ⟨y t , A x⟩ + ρ 2 ∥A x -b∥ 2 + µ 2 ∥ x -z t ∥ 2 ] = f (x * (z t )) + µ 2 ∥x * (z t ) -z t ∥ 2 -f ( x) - µ 2 ∥ x -z t ∥ 2 -⟨y t , A x -b⟩ - ρ 2 ∥A x -b∥ 2 = f (x * (z t )) + µ 2 ∥x * (z t ) -z t ∥ 2 -f ( x) - µ 2 ∥ x -z t ∥ 2 - ρ 2 ∥A x -b∥ 2 -⟨y t , A x -b⟩ ≥ -2M -⟨y t , A x -b⟩,
where the first inequality comes from the definition of Ψ(z t ) and
d(y t , z t ) = min x∈X K(x, y, z).
Finally, in the last inequality, we let
M = max (x,z)∈X×X {|f (x)| + µ 2 ∥x -z∥ 2 + ρ 2 ∥Ax -b∥ 2 }.
As a result, we have the last inequality.
According to Assumption 4.1(2), there exists a positive r > 0 such that for any direction d ∈ Range(A), we can find a x ∈ X satisfying ∥Ax -b∥ = r and Ax -b has the same direction as d. Because y t ∈ Range(A) (by assumption 4.1(3), Range(A) = R m ) we can choose x such that A xb is of the same direction as -y t and ∥A x -b∥ = r. Then we obtain Ψ(z t ) -d(y t , z t ) ≥ -2M + r∥y t ∥ =⇒ ∥y t ∥ ≤ Ψ(z t ) -d(y t , z t ) + 2M r , ∀t ∈ {0, 1, ..., T }.
This concludes the proof. ■
this section cite: []

Section: Proof of Theorem 4.3
Proof of Theorem 4.3. First, let M V = max x,z∈X {K(x, 0, z) -2d(0, z) + 2Ψ(z)} and M y > M V -MΨ+2M r where M Ψ is a uniform lower bound of Ψ(z t ), for example, f . Here, We denote the x, y, z generated by Algorithm 2 at iteration t as x t , y t , z t and the output of iteration t + 1 as x t+1 , y t+1 , z t+1 .
If ∥y t + η(A ζt x t -b ζt )∥ ≤ M y , then E [V (x t , y t , z t ) -V (x t+1 , y t+1 , z t+1 )] ≥ c β E∥z t+1 -z t ∥ 2 + c τ E∥u * (x t , y t+1 , z t ) -x t ∥ 2 + c η E∥Ax * (y t+1 , z t ) -b∥ 2 - 1 2 λτ 2 σ 2 2 -1 + 17∥A∥ 4 γ 2 K η 2 L 2 = c β E∥z t+1 -z t ∥ 2 + c τ E∥u * (x t , y t + η(A ζt x t -b ζt ), z t ) -x t ∥ 2 + c η E∥Ax * (y t + η(A ζt x t -b ζt ), z t ) -b∥ 2 - 1 2 λτ 2 σ 2 2 -1 + 17∥A∥ 4 γ 2 K η 2 L 2 ,(72)
where the first inequality use Lemma B.2 and the equality comes from the update of y t+1 when ∥y t +η(A ζt x t -b ζt )∥ ≤ M y .
If ∥y t + η(A ζt x t -b ζt )∥ > M y , we have y t+1 = 0. Let us use ŷt+1 , xt+1 , ẑt+1 denote the iteration generated with ŷt+1 = y t + η(A ζt x t -b ζt ). Then
K(x t+1 , ŷt+1 , ẑt+1 ) -2d(ŷ t+1 , ẑt+1 ) + 2Ψ(ẑ t+1 ) ≥ Ψ(ẑ t+1 ) -d(ŷ t+1 , ẑt+1 ) + Ψ(ẑ t+1 ) ≥ r∥ŷ t+1 ∥ -2M + M Ψ ≥ rM y -2M + M Ψ ≥ M V = max x,z∈X {K(x, 0, z) -2d(0, z) + 2Ψ(z)} ≥ K(x t+1 , 0, z t+1 ) -2d(0, z t+1 ) + 2Ψ(z t+1 ) = K(x t+1 , y t+1 , z t+1 ) -2d(y t+1 , z t+1 ) + 2Ψ(z t+1 ),
where the first step used d(ŷ t+1 , ẑt+1 ) ≤ K(x t+1 , ŷt+1 , ẑt+1 ) and the second line uses Prop. B.3 and Ψ(ẑ t+1 ) ≥ M Ψ .
Hence we have
EV (x t , y t , z t ) -EV (x t+1 , y t+1 , z t+1 ) = E [K(x t , y t , z t ) -2d(y t , z t ) + 2Ψ(z t )] -E [K(x t+1 , y t+1 , z t+1 ) -2d(y t+1 , z t+1 ) + 2Ψ(z t+1 )] ≥ E [K(x t , y t , z t ) -2d(y t , z t ) + 2Ψ(z t )] -E [K(x t+1 , ŷt+1 , ẑt+1 ) -2d(ŷ t+1 , ẑt+1 ) + 2Ψ(ẑ t+1 )] ≥ c β E∥ẑ t+1 -z t ∥ 2 + c τ E∥u * (x t , ŷt+1 , z t ) -x t ∥ 2 + c η E∥Ax * (ŷ t+1 , z t ) -b∥ 2 - 1 2 λτ 2 σ 2 2 -1 + 17∥A∥ 4 γ 2 K η 2 L 2 = c β E∥z t+1 -z t ∥ 2 + c τ E∥u * (x t , y t + η(A ζt x t -b ζt ), z t ) -x t ∥ 2 + c η E∥Ax * (y t + η(A ζt x t -b ζt ), z t ) -b∥ 2 - 1 2 λτ 2 σ 2 2 -1 + 17∥A∥ 4 γ 2 K η 2 L 2 ,(73)
where in last inequality, we use Lemma B.2, and in the last equality we use the fact that ẑt+1 = z t + β(x t -z t ) = z t+1 , ŷt+1 = y t + η(A ζt x t -b ζt ).
Combining ( 72) and ( 73), we have that
E [V t -V t+1 ] ≥ c β E∥z t+1 -z t ∥ 2 + c τ E∥u * (x t , y t + η(A ζt x t -b ζt ), z t ) -x t ∥ 2 + c η E∥Ax * (y t + η(A ζt x t -b ζt ), z t ) -b∥ 2 - 1 2 λτ 2 σ 2 2 -1 + 17∥A∥ 4 γ 2 K η 2 L 2 ,
holds for both ∥y t + η(A ζt x t -b ζt )∥ ≤ M y and ∥y t + η(A ζt x t -b ζt )∥ > M y , which means it holds for x t+1 , y t+1 , z t+1 generated by Algorithm 2. Then we can telescope as before and the convergence result follows.
We also now sketch the argument for the complexity. We have for the gradient of the Moreau envelope that
1 µ 2 ∥∇Ψ(z t )∥ = ∥z t -x * (z t )∥ ≤ ∥z t -x * (y t + η(A ζt x t -b ζt ), z t )∥ + ∥x * (y t + η(A ζt x t -b ζt ), z t ) -x * (z t )∥ ≤ ∥z t -x * (y t + η(A ζt x t -b ζt ), z t )∥ + σ∥Ax * (y t + η(A ζt x t -b ζt ), z s ) -b∥ ≤ ∥z t -x t ∥ + ∥x t -x * (y t + η(A ζt x t -b ζt ), z t )∥ + σ∥Ax * (y t + η(A ζt x t -b ζt ), z t ) -b∥ ≤ ∥z t -x t ∥ + λ γ ∥x t -u * (x t , y t + η(A ζt x t -b ζt ), z t )∥ + σ∥Ax * (y t + η(A ζt x t -b ζt ), z t ) -b∥,
where the second line is by triangle inequality, the second inequality is by Lemma A.12, and the fourth line is by triangle inequality and the last estimation is by ( 58).
The rest of the proof for the complexity result proceeds the same as Appendix A.2 up to simple changes in the constants, and hence is omitted. ■
this section cite: []

Section: C. Proofs for Section 5
Notation. Let us note that we define by E ξt the expectation conditioned on all the randomness before ξ t .
this section cite: []

Section: C.1. Proofs for Theorem 5.3
First, with the idea of the STORM estimator of Cutkosky & Orabona (2019), we have the following lemma to control the variance of the stochastic gradient. Lemma C.1. (from (Cutkosky & Orabona, 2019)) Let Assumption 5.2 hold. We have the estimation of the variance as:
E∥ ∇f t+1 -∇f (x t+1 )∥ 2 ≤ (1 -α) 2 E∥ ∇f t -∇f (x t )∥ 2 + 3(L 2 0 + L 2 f )E∥x t+1 -x t ∥ 2 + 3α 2 σ 2 .
Proof. By the definition of ∇f t+1 in Alg. 3, we have
∇f t+1 -∇f (x t+1 ) = ∇f (x t+1 , ξ t+1 ) + (1 -α)( ∇f t -∇f (x t , ξ t+1 )) -∇f (x t+1 ) = ∇f (x t+1 , ξ t+1 ) + (1 -α)( ∇f t -∇f (x t )) + (1 -α)(∇f (x t ) -∇f (x t , ξ t+1 )) -∇f (x t+1 ) = (1 -α)( ∇f t -∇f (x t )) + (1 -α)(∇f (x t ) -∇f (x t , ξ t+1 )) + ∇f (x t+1 , ξ t+1 ) -∇f (x t+1 ),(74)
where in the second equality, we added and subtracted (1 -α)∇f (x t ).
Then, we compute the squared norm of (74) and expand to get
∥ ∇f t+1 -∇f (x t+1 )∥ 2 = (1 -α) 2 ∥ ∇f t -∇f (x t )∥ 2 + ∥(1 -α)(∇f (x t ) -∇f (x t , ξ t+1 )) + ∇f (x t+1 , ξ t+1 ) -∇f (x t+1 )∥ 2 + 2(1 -α)⟨ ∇f t -∇f (x t ), (1 -α)(∇f (x t ) -∇f (x t , ξ t+1 )) + ∇f (x t+1 , ξ t+1 ) -∇f (x t+1 )⟩.
where in the last inequality, we use Lem. C.6 and Lem. A.12, then combine the like-terms.
Then we need to estimate the coefficients of each terms in the above inequality. Let us recall from ( 25) that σ 4 = µ-L f µ > 1 2 and let ζ = 6β.
We now estimate the coefficient of E∥z t -z t+1 ∥ 2 in (90). First, by σ 4 > 1 2 , we have µ σ 2 4 ≤ 4µ and µ σ4 ≤ 2µ. By also using ζ = 6β, we have:
The coefficient of E∥z t -z t+1 ∥ 2 ≥ µ β - 3µ 4 -4µ - µ 6β -2µ.
Using β ≤ 1/50, we obtain ( 3 4 + 4 + 2)µ ≤ µ 5β , then we estimate:
The coefficient of E∥z
t -z t+1 ∥ 2 ≥ µ β - µ 5β - µ 6β ≥ µ 2β .
We move on to estimating the coefficient of E∥x
t -x t+1 ∥ 2 in (90). With η ≤ (µ-L f ) 2 τ 8∥A∥ 2 , we have 2η∥A∥ 2 1 τ 2 (µ-L f ) 2 ≤ 1 4τ , we have: The coefficient of E∥x t -x t+1 ∥ 2 ≥ 1 4τ - L K 2 -µ.
Last, we work on the coefficient of E∥Ax * (y t+1 , z t ) -b∥ 2 in (90). Because ζ = 6β, it follows that η -µζ σ2 = η -6µβ σ2 .
With β ≤ η 36µσ 2 , we have 6µβ σ2 ≤ η 6 , then we estimate:
the coefficient of E∥Ax * (y t+1 , z t ) -b∥ 2 ≥ η - η 6 ≥ η 2 .
Next, we estimate the coefficient of E∥∇f (
x t ) -∇f t ∥ 2 . With η ≤ (µ-L f ) 2 τ 8∥A∥ 2 , we have -τ 2 -2η∥A∥ 2 (µ-L f ) 2 ≥ -3 4 τ . Finally, we have EV t -EV t+1 ≥ µ 2β E∥z t -z t+1 ∥ 2 + 1 4τ - L K 2 -µ E∥x t -x t+1 ∥ 2 + η 2 E∥Ax * (y t+1 , z t ) -b∥ 2 - 3τ 4 E∥∇f (x t ) -∇f t ∥ 2 = µ 2β E∥z t -z t+1 ∥ 2 + 1 4τ - L K 2 -µ E∥x t -x t+1 ∥ 2 + η 2 E∥Ax * (y t+1 , z t ) -b∥ 2 + τ 4 E∥∇f (x t ) -∇f t ∥ 2 -τ E∥∇f (x t ) -∇f t ∥ 2 .
Then recalling Lemma C.1 and assuming 0 < α ≤ 1, we have
E∥ ∇f t+1 -∇f (x t+1 )∥ 2 ≤ (1 -α)E∥ ∇f t -∇f (x t )∥ 2 + 3(L 2 0 + L 2 f )E∥x t+1 -x t ∥ 2 + 3α 2 σ 2 .(91)
We multiply (91) by τ α , rearrange, and plug into (91), to get
EV t -EV t+1 ≥ µ 2β E∥z t -z t+1 ∥ 2 + 1 4τ - L K 2 -µ E∥x t -x t+1 ∥ 2 + η 2 E∥Ax * (y t+1 , z t ) -b∥ 2 + τ 4 E∥∇f (x t ) -∇f t ∥ 2 + τ α E∥ ∇f t+1 -∇f (x t+1 )∥ 2 - τ α E∥ ∇f t -∇f (x t )∥ 2 - 3(L 2 0 + L 2 f )τ α E∥x t -x t+1 ∥ 2 -3ασ 2 τ.(92)
Because α = 48(L 2 0 + L 2 f )τ 2 and τ ≤ min
1 8L K +16µ , 1 √ 48(L 2 0 +L 2 f )
, we obtain
L K 2 + µ ≤ 1 16τ , 3(L 2 0 + L 2 f )τ α = 1 16τ
.
Hence, we have
EV t -EV t+1 ≥ µ 2β E∥z t -z t+1 ∥ 2 + 1 8τ E∥x t -x t+1 ∥ 2 + η 2 E∥Ax * (y t+1 , z t ) -b∥ 2 + τ 4 E∥∇f (x t ) -∇f t ∥ 2 + 1 48(L 2 0 + L 2 f )τ E∥ ∇f t+1 -∇f (x t+1 )∥ 2 -1 48(L 2 0 + L 2 f )τ E∥ ∇f t -∇f (x t )∥ 2 -144(L 2 0 + L 2 f )σ 2 τ 3 . Finally, we move 1 48(L 2 0 +L 2 f )τ E∥ ∇f t+1 -∇f (x t+1 )∥ 2 -1 48(L 2 0 +L 2 f )τ E∥ ∇f t -∇f (x t )∥ 2 to the left-hand side of the above inequality and use the definition of Vt in (18) to get the desired result. ■ C.2. Proofs for Theorem 5.4 First, we need two lemmas for the error bound that helps us analyze the sample complexity that we include for being self-contained. Lemma C.5. (Zhang & Luo, 2020, Lemma 3.10) Under Assumption 1.1, we have
∥x -proj X (x -τ ∇K(x, y, z))∥ ≥ τ (µ -L f ) 2 ∥x -x * (y, z)∥ ,
where K(x, y, z) = L ρ (x, y) + µ 2 ∥x -z∥ 2 , and x * (y, z) = arg min x∈X K(x, y, z).
Proof. First, we denote that x = x -proj X (x -τ ∇K(x, y, z)), then by the definition of x * (y, z), we have ⟨x -xx * (y, z), τ ∇K(x * (y, z), y, z)⟩ ≥ 0, where we use the fact that xx ∈ X.
Then by the definition of projection (that is, z = proj X (z) ⇐⇒ ⟨z -z, t -z⟩ ≥ 0 ∀t ∈ X), the definition of x, and x * (y, z) ∈ X, we have ⟨x * (y, z) -proj X (x -τ ∇K(x, y, z)), x -τ ∇K(x, y, z) -proj X (x -τ ∇K(x, y, z))⟩ =⟨x * (y, z) -(x -x), -τ ∇K(x, y, z) + x⟩ ≤ 0.
Combining above two inequalities and rearranging terms, we have ⟨x -x * (y, z), τ ∇K(x, y, z) -τ ∇K(x * (y, z), y, z)⟩ ≤ ⟨x, τ ∇K(x, y, z) -τ ∇K(x * (y, z), y, z) + xx * (y, z)⟩ -∥x∥ 2 ≤ ∥x∥∥τ ∇K(x, y, z) -τ ∇K(x * (y, z), y, z)
+ x -x * (y, z)∥ ≤ ∥x∥(τ L K + 1)∥x -x * (y, z)∥ ≤ 2∥x∥∥x -x * (y, z)∥,
where in the second inequality we use the Cauchy-Schwarz inequality and in the last inequality we use the Lipschitz continuity of ∇K with respect to x.
By K(x, y, z) being (µ -L f )-strongly convex with respect to x (see Fact A.1), we have
⟨x -x * (y, z), τ ∇K(x, y, z) -τ ∇K(x * (y, z), y, z)⟩ ≥ τ (µ -L f )∥x -x * (y, z)∥ 2 .
Then, the desired result follows by combining the above two inequalities and using the definition of x = x -proj X (xτ ∇K(x, y, z)). ■
With the next lemma, we proceed to prove that ∥x t -x * (y t+1 , z t )∥ is bounded by a combination of ∥x t -x t+1 ∥ and ∥ ∇f t -∇f (x t )∥.
Lemma C.6. Under Assumption 1.1, for the iterates generated by Algorithm 3 we have
∥x t -x * (y t+1 , z t )∥ ≤ 2 τ (µ -L f ) ∥x t -x t+1 ∥ + 2 (µ -L f ) ∥ ∇f t -∇f (x t )∥.
Proof. Taking x, y, z as x t , y t+1 , z t in Lemma C.5, we have
∥x t -x * (y t+1 , z t )∥ ≤ 2 τ (µ -L f ) ∥x t -proj X (x t -τ ∇K(x, y t+1 , z t ))∥ ≤ 2 τ (µ -L f ) ∥x t -proj X (x t -τ G(x t , y t+1 , z t ))∥ + 2 τ (µ -L f ) ∥ proj X (x t -τ ∇K(x t , y t+1 , z t )) -proj X (x t -τ G(x t , y t+1 , z t ))∥ ≤ 2 τ (µ -L f ) ∥x t -x t+1 ∥ + 2 (µ -L f ) ∥ ∇f t -∇f (x t )∥,
where the second inequality comes form triangle inequality and the last inequality comes from the fact that proj X is nonexpansive and ∇K(x t , y t+1 , z t ) -G(x t , y t+1 , z t ) = ∇f (x t ) -∇f t . ■
We now continue with the proof of Theorem 5.4.
Proof of Theorem 5.4. Because z t+1 -z t = β(x t -z t ), µβ 2 = Θ(τ ) and η 2 = Θ(τ ) in view of Lemma C.4, hence there exists a constant C such that we get from ( 83):
E Vt -E Vt+1 ≥ Cτ {E∥x t -z t ∥ 2 + E∥τ -1 (x t -x t+1 )∥ 2 + E∥Ax * (y t+1 , z t ) -b∥ 2 + E∥∇f (x t ) -∇f t ∥ 2 } -144 L 2 0 + L 2 f σ 2 τ 3 .(93)
Then, summing up (93) over t = 0, 1, . . . , T -1, we have
V0 -E VT ≥ T -1 t=0 Cτ {E∥x t -z t ∥ 2 + E∥τ -1 (x t -x t+1 )∥ 2 + E∥Ax * (y t+1 , z t ) -b∥ 2 + E∥∇f (x t ) -∇f t ∥ 2 } -144 L 2 0 + L 2 f σ 2 τ 3 T.(94)
From the definition, we have K(x, y, z) ≥ d(y, z) (since d(y, z) = min x∈X K(x, y, z)) and Ψ(z) ≥ d(y, z) (see also Lemma A.13), then V t = K(x t , y t , z t ) -2d(y t , z t ) + 2Ψ(z t ) ≥ Ψ(z t ) ≥ f .
Consequently, we have
Vt = K(x t , y t , z t ) -2d(y t , z t ) + 2Ψ(z t ) + 1 48(L 2 0 + L 2 f )τ E∥ ∇f t -∇f (x t )∥ 2 ≥ f .(95)
Let τ = T -1/3 and use mini-batch in the initial step where we will have E∥ ∇f 0 -∇f (x 0 )∥ 2 ≤ T -1/3 σ 2 (by the definition of ∇f 0 and a standard computation), then
V0 = K(x 0 , y 0 , z 0 ) -2d(y 0 , z 0 ) + 2Ψ(z 0 ) + 1 48(L 2 0 + L 2 f )τ E∥ ∇f 0 -∇f (x 0 )∥ 2 ≤ K(x 0 , y 0 , z 0 ) -2d(y 0 , z 0 ) + 2Ψ(z 0 ) + σ 2 48(L 2 0 + L 2 f ) ,(96)
where the right-hand is proportional to a constant independent of T , we denote it as C 0 .
Combining (94) with ( 95) and (96), we have
1 T T -1 t=0 C{E∥x t -z t ∥ 2 + E∥τ -1 (x t -x t+1 )∥ 2 + E∥Ax * (y t+1 , z t ) -b∥ 2 + E∥∇f (x t ) -∇f t ∥ 2 } ≤ T -2/3 C 0 -f + 144(L 2 0 + L 2 f )σ 2 .(97)
Then, for index s selected uniformly at random from {0, 1, ..., T -1}, we have
E∥x s -z s ∥ 2 = O(T -2/3 ), E∥τ -1 (x s -x s+1 )∥ 2 = O(T -2/3 ), E∥Ax * (y s+1 , z s ) -b∥ 2 = O(T -2/3 ), E∥∇f (x t ) -∇f t ∥ 2 = O(T -2/3 ).(98)
According to Algorithm 3, we have
x s+1 = arg min x ⟨G(x s , y s+1 , z s ), x -x s ⟩ + 1 τ ∥x -x s ∥ 2 + ∂I X (x) .
By the definition of x s+1 , we have
0 ∈ G(x s , y s+1 , z s ) + 2 τ (x s+1 -x s ) + ∂I X (x s+1 ).(99)
We now set
v = ∇ x K(x s+1 , y s+1 , z s ) -G(x s , y s+1 , z s ) - 2 τ (x s+1 -x s ) -ρA ⊤ (Ax s+1 -b) -µ(x s+1 -z s ).
Now, by using the definition of K(x, y, z) from ( 19) and (99), we obtain (cf. ( 5)
) v ∈ ∇f (x s+1 ) + A ⊤ y s+1 + ∂I X (x s+1 )
We now derive the guarantees on the feasibility and the norm of v. First, by triangle inequality, we have
∥Ax s+1 -b∥ ≤ ∥Ax * (y s+1 , z s ) -b∥ + ∥Ax s+1 -Ax s ∥ + ∥A(x s -x * (y s+1 , z s ))∥ ≤ ∥Ax * (y s+1 , z s ) -b∥ + ∥A∥∥x s+1 -x s ∥ + 2∥A∥ τ (µ -L f ) ∥x s -x s+1 ∥ + 2∥A∥ µ -L f ∥ ∇f s -∇f (x s )∥ = O(T -1/3 ),(100)
where in the second inequality, we use Lemma C.6 and the last estimate uses (98).
Then, we have by triangle inequality that
∥v∥ ≤ ∥∇ x K(x s+1 , y s+1 , z s ) -∇ x K(x s , y s+1 , z s )∥ + ∥∇ x K(x s , y s+1 , z s ) -G(x s , y s+1 , z s )∥ + 2 τ ∥x s+1 -x s ∥ + ρ∥A∥∥Ax s+1 -b∥ + µ∥x s+1 -z s ∥ ≤ L K + 2 τ ∥x s+1 -x s ∥ + ∥∇f (x s ) -∇f s ∥ + ρ∥A∥∥Ax s+1 -b∥ + µ (∥x s -z s ∥ + ∥x s+1 -x s ∥) = O(T -1/3 ),
where in first inequality, we introduce a term ∇ x K(x s , y s+1 , z s ) and then use triangle inequality. The second inequality used Lipschitzness of K, the definition of G, and the triangle inequality. The last step uses ( 98) and ( 100) and ρ = O(1) since it is chosen arbitrarily in Alg. 3. ■
this section cite: ['b6', 'b6']

Section: 
Proof. We sum up the results in Lemma A.5 and Lemma A.6, plug in the definition of y t+1 and discard two nonnegative terms on the right-hand side to get the result.
this section cite: []

Section: ■
Next, we analyze the rest of the terms appearing in the potential function. This lemma is only using the definition of d(y, z) and Ψ(z) and is equivalent to (Zhang & Luo, 2022) and hence we omit its proof. Notably, these bounds are agnostic to the algorithm used to generate the sequences. Note that the only difference is that in the result below, we do not use the definition of y t+1 whereas the proof in (Zhang & Luo, 2022) uses this definition. The rest of the estimations are precisely the same. Lemma A.8. (Zhang & Luo, 2020, Lemma 3.2, Lemma 3.3) For the functions d(y, z) and Ψ(z) defined in (12) and (7),we have
d(y t+1 , z t+1 ) -d(y t , z t ) ≥ η⟨Ax t -b, Ax * (y t+1 , z t ) -b⟩ + µ 2 ⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩, Ψ(z t+1 ) -Ψ(z t ) ≤ µ⟨z t+1 -z t , z t -x * (z t )⟩ + µ 2σ 4 ∥z t -z t+1 ∥ 2 ,
where σ 4 is defined in (25).
In the next lemma, we will join the previous lemmas and characterize the change in the potential function.
Lemma A.9 (cf. Lemma 3.6). Let Assumption 1.1 hold. By using the parameters (25) in Algorithm 1, we obtain
EV t -EV t+1 ≥ c β E∥z t+1 -z t ∥ 2 + c τ E∥u * (x t , y t+1 , z t ) -x t ∥ 2 + c η E∥Ax * (y t+1 , z t ) -b∥ 2 - 1 2 λτ 2 σ 2 ,(37)
where
c β = µ 50β , c τ = 7τ λ 2 400 , c η = η 4 .
Proof. Combining Corollary A.7 and Lemma A.8, we obtain
E[V t -V t+1 ] = E φ 1/λ (x t , y t , z t ) -φ 1/λ (x t+1 , y t+1 , z t+1 ) + 2d(y t+1 , z t+1 ) -2d(y t , z t ) + 2Ψ(z t ) -2Ψ(z t+1 ) ≥ τ λ 2 16 E∥u * (x t , y t+1 , z t ) -x t ∥ 2 - λτ µ 2 + λτ 2 µ 2 + τ λ 2 µ 2 8γ 2 s E∥z t -z t+1 ∥ 2 - λτ 2 σ 2 2 -ηE⟨Ax t -b, Au * (x t , y t , z t ) -b⟩ + µ 2 E⟨z t+1 -z t , 2u * (x t , y t+1 , z t ) -z t -z t+1 ⟩ + 2ηE⟨Ax t -b, Ax * (y t+1 , z t ) -b⟩ + µE⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩ -2µE⟨z t+1 -z t , z t -x * (z t )⟩ - µ σ 4 E∥z t -z t+1 ∥ 2 .(38)
We next manipulate the terms on the right-hand side. First, by adding and subtracting Ax t on the second argument of the first inner product on the right-hand side, we get
-η⟨Ax t -b, Au * (x t , y t , z t ) -b⟩ = -η∥Ax t -b∥ 2 -η⟨Ax t -b, Au * (x t , y t , z t ) -Ax t ⟩.
Consequently, we use this estimate and rewrite the third inner product on the right-hand side of (38) with quadratics to have
-η⟨Ax t -b, Au * (x t , y t , z t ) -b⟩ + 2η⟨Ax t -b, Ax * (y t+1 , z t ) -b⟩ = -η∥Ax t -Ax * (y t+1 , z t )∥ 2 + η∥Ax * (y t+1 , z t ) -b∥ 2 -η⟨Ax t -b, Au * (x t , y t , z t ) -Ax t ⟩.
Second, adding and subtracting 2x t in the second argument of the second inner product on the right-hand side of (38) gives
µ 2 ⟨z t+1 -z t , 2u * (x t , y t+1 , z t ) -z t -z t+1 ⟩ = µ 2 ⟨z t+1 -z t , 2u * (x t , y t+1 , z t ) -2x t ⟩ + µ 2 ⟨z t+1 -z t , 2x t -z t -z t+1 ⟩.
For the right-hand side of this term, note that z t+1 = z t + β(x t -z t ) ⇐⇒ 2x t -2z t = 2 β (z t+1 -z t ) and hence
µ 2 ⟨z t+1 -z t , 2x t -z t -z t+1 ⟩ = µ 2 ⟨z t+1 -z t , 2x t -2z t ⟩ + µ 2 ⟨z t+1 -z t , z t -z t+1 ⟩ = µ 2 2 β -1 ∥z t -z t+1 ∥ 2 ≥ µ 2β ∥z t -z t+1 ∥ 2 ,
The second inequality in the above chain comes from definition, that is, denoting
x * µ = arg min x∈X,Ax=b {f (x) + µ 2 ∥x - z∥ 2 } in view of (7), we have d(y, z) = min x∈X K(x, y, z) ≤ K(x * µ , y, z) = f (x * µ ) + µ 2 ∥x * µ -z∥ 2 = Ψ(z),
where the first inequality also uses x * µ ∈ X, which is by definition. ■
this section cite: []

Section: B. Proofs for Section 4
Notation. In this section, we have ∥∇f (x, ξ t ) -∇f (x t )∥ 2 ≤ σ 2 and E∥A ζt x t -b ζt ∥ 2 ≤ L, then we denote the boundedness of variance as E∥G(x t , y t , z t , ξ t ) -∇ x K(x t , y t , z t )∥ 2 ≤ σ 2 2 , where the boundedness is proved in B.2. We start with some helper lemmas before proving Theorem 4.3.
Lemma B.1. Let Assumption 4.1 hold. With the update rule of y t+1 = y t + η(A ζt x t -b ζt ), where E ζt [A ζt x t -b ζt ] = Ax t -b, we have Ed(y t+1 , z t+1 ) -Ed(y t .z t ) ≥ ηE⟨(Ax t -b), Ax * (y t+1 , z t ) -b⟩ - η 2 32 E∥Ax t -b∥ 2 - 1 2 + 17∥A∥ 4 2γ K η 2 L 2 + µ 2 E⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩, EΨ(z t+1 ) -EΨ(z t ) ≤ µE⟨z t+1 -z t , z t -x * (z t )⟩ + µ 2σ 4 E∥z t -z t+1 ∥ 2 ,(68)
where γ K , σ 4 are introduceed in A.10, and by Assumption 4.1, we have E∥A ζt x t -b ζt ∥ 2 ≤ L for some finite L.
Proof. It is easy to derive, for example as (Zhang & Luo, 2020, Lemma 3.2), that
d(y t+1 , z t+1 ) -d(y t , z t ) ≥ ⟨y t+1 -y t , Ax * (y t+1 , z t ) -b⟩ + µ 2 ⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩.
Hence, by using the update rule of y t+1 , we get
d(y t+1 , z t+1 ) -d(y t , z t ) ≥ ⟨y t+1 -y t , Ax * (y t , z t ) -b⟩ + ⟨y t+1 -y t , Ax * (y t+1 , z t ) -Ax * (y t , z t )⟩ + µ 2 ⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩ ≥ ⟨y t+1 -y t , Ax * (y t , z t ) -b⟩ - 1 2 ∥y t+1 -y t ∥ 2 - 1 2 ∥Ax * (y t+1 , z t ) -Ax * (y t , z t )∥ 2 + µ 2 ⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩ ≥ ⟨η(A ζt x t -b ζt ), Ax * (y t , z t ) -b⟩ - 1 2 η 2 L 2 - ∥A∥ 4 2γ 2 K ∥y t+1 -y t ∥ 2 + µ 2 ⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩,
where we use Cauchy-Schwarz inequality in the second step, and the last inequality comes from the bound of E∥A ζt x tb ζt ∥ 2 also (63).
After taking expectation and using tower property along with y t , z t being deterministic under the conditioning, we have
Ed(y t+1 , z t+1 ) -Ed(y t , z t ) ≥ E⟨η(Ax t -b), Ax * (y t , z t ) -b⟩ - 1 2 η 2 L 2 - ∥A∥ 4 2γ 2 K E∥y t+1 -y t ∥ 2 + µ 2 E⟨z t+1 -z t , z t+1 + z t -2x(y t+1 , z t+1 )⟩.
Next, we take expectation with respect to the randomness of ξ t+1 to obtain
E ξt+1 ∥ ∇f t+1 -∇f (x t+1 )∥ 2 = (1 -α) 2 E ξt+1 ∥ ∇f t -∇f (x t )∥ 2 + E ξt+1 ∥(1 -α)(∇f (x t ) -∇f (x t , ξ t+1 )) + ∇f (x t+1 , ξ t+1 ) -∇f (x t+1 )∥ 2 , (75
)
which is due to ∇f t -∇f (x t ) being independent of ξ t+1 , as well as
E ξt+1 [∇f (x t ) -∇f (x t , ξ t+1 )] = 0, E ξt+1 [∇f (x t+1 , ξ t+1 ) -∇f (x t+1 )] = 0.
Finally, we estimate the last term on the right-hand side of ( 75):
E ξt+1 ∥(1 -α)(∇f (x t ) -∇f (x t , ξ t+1 )) + ∇f (x t+1 , ξ t+1 ) -∇f (x t+1 )∥ 2 = E ξt+1 ∥∇f (x t+1 , ξ t+1 ) -∇f (x t , ξ t+1 ) + ∇f (x t ) -∇f (x t+1 ) + α(f (x t , ξ t+1 ) -∇f (x t ))∥ 2 ≤ 3E ξt+1 ∥∇f (x t+1 , ξ t+1 ) -∇f (x t , ξ t+1 )∥ 2 + ∥∇f (x t ) -∇f (x t+1 )∥ 2 + ∥α(∇f (x t , ξ t+1 ) -∇f (x t ))∥ 2 ≤ 3L 2 0 ∥x t+1 -x t ∥ 2 + 3L 2 f ∥x t -x t+1 ∥ 2 + 3α 2 σ 2 ,
where in the first equality, we rearrange the terms, and in the first inequality, we use Young's inequality. In the second inequality, we use Assumption 5.2, L f -smoothness of f (x) and E ξ ∥∇f (x, ξ) -∇f (x)∥ 2 ≤ σ 2 . We use this estimation in (75) and take total expectation to get the result. ■ Let us recall from ( 18) that
Vt = K(x t , y t , z t ) -2d(y t , z t ) + 2Ψ(z t ) + 1 48(L 2 0 + L 2 f )τ ∥ ∇f t -∇f (x t )∥ 2 ,(76)
where (as ( 22)
) K(x, y, z) = L ρ (x, y) + µ 2 ∥x -z∥ 2(77)
and
x → ∇K(x, y, z) is L K -Lipschitz with L K = L f + ρ∥A∥ + µ (see also Fact A.1).
We already have the descent-type lemma of d(y t , z t ) and Ψ(z t ) in Lemma A.8, and only need to show the descent-type lemma of K(x t , y t , z t ). We write K(x t , y t , z t ) -K(x t+1 , y t+1 , z t+1 ) as:
[K(x t , y t+1 , z t ) -K(x t+1 , y t+1 , z t )] + [K(x t , y t , z t ) -K(x t , y t+1 , z t )] + [K(x t+1 , y t+1 , z t ) -K(x t+1 , y t+1 , z t+1 )]
and lower bound each term separately in the following lemmas. Lemma C.2. Let Assumption 1.1 hold. For the iterates generated by Algorithm 3, we have
K(x t+1 , y t+1 , z t ) -K(x t , y t+1 , z t ) ≤ τ 2 ∥∇f (x t ) -∇f t ∥ 2 - 1 2τ - L K 2 ∥x t+1 -x t ∥ 2 .
Proof. We have, by smoothness of K(•, y t+1 , z t ):
K(x t+1 , y t+1 , z t ) ≤ K(x t , y t+1 , z t ) + ⟨∇ x K(x t , y t+1 , z t ), x t+1 -x t ⟩ + L K 2 ∥x t+1 -x t ∥ 2 . (78
)
We estimate the inner product here as
⟨∇ x K(x t , y t+1 , z t ), x t+1 -x t ⟩ = ⟨G(x t , y t+1 , z t ), x t+1 -x t ⟩ + ⟨∇ x K(x t , y t+1 , z t ) -G(x t , y t+1 , z t ), x t+1 -x t ⟩. (79
)
We first have, in view of Alg. 3 that
∇ x K(x t , y t+1 , z t ) -G(x t , y t+1 , z t ) = ∇f (x t ) -∇f t .
The definition of x t+1 in Alg. 3 gives
⟨x t+1 -x t + τ G(x t , y t+1 , z t ), x t -x t+1 ⟩ ≥ 0 ⇐⇒ ⟨G(x t , y t+1 , z t ), x t+1 -x t ⟩ ≤ - 1 τ ∥x t+1 -x t ∥ 2 . (80) Using ⟨∇ x K(x t , y t+1 , z t ) -G(x t , y t+1 , z t ), x t+1 -x t ⟩ ≤ τ 2 ∥∇f (x t ) -∇f t ∥ 2 + 1 2τ ∥x t+1 -x t ∥ 2
along with ( 80) in (79), we have
⟨∇ x K(x t , y t+1 , z t ), x t+1 -x t ⟩ ≤ τ 2 ∥∇f (x t ) -∇f t ∥ 2 - 1 2τ ∥x t+1 -x t ∥ 2 .
Then the result follows after substituting the last estimate in (78). ■ Lemma C.3. Let Assumption 1.1 hold. For the iterates generated by Algorithm 3, we have
K(x t , y t , z t ) -K(x t+1 , y t+1 , z t+1 ) ≥ -η∥Ax t -b∥ 2 + µ β - 3µ 4 ∥z t+1 -z t ∥ 2 - τ 2 ∥∇f (x t ) -∇f t ∥ 2 + 1 2τ - L K 2 -µ ∥x t+1 -x t ∥ 2 .(81)
Proof. First, from the definition of K in ( 22), we have
K(x t , y t , z t ) -K(x t , y t+1 , z t ) = -η∥Ax t -b∥ 2 .
Moreover, it follows that
K(x t+1 , y t+1 , z t ) -K(x t+1 , y t+1 , z t+1 ) = µ 2 (∥x t+1 -z t ∥ 2 -∥x t+1 -z t+1 ∥ 2 ) = µ 2 ⟨z t+1 -z t , 2x t+1 -z t -z t+1 ⟩ = µ 2 ⟨z t+1 -z t , 2x t+1 -2x t + 2x t -2z t + z t -z t+1 ⟩ = µ 2 ⟨z t+1 -z t , 2x t+1 -2x t ⟩ + µ 2 ⟨z t+1 -z t , 2x t -2z t ⟩ - µ 2 ∥z t+1 -z t ∥ 2 ≥ - µ 4 ∥z t+1 -z t ∥ 2 -µ∥x t+1 -x t ∥ 2 + µ β ∥z t -z t+1 ∥ 2 - µ 2 ∥z t+1 -z t ∥ 2 ,
where the first equality comes from the definition of K. In the last inequality, we use ⟨a, b⟩ ≥ -1 4 ∥a∥ 2 -∥b∥ 2 and x t -z t = zt+1-zt β by the definition of z t+1 in Algorithm 3. Fanally combining the above two results with Lemma C.2 and combining like-terms yields the claim.
this section cite: []

Section: ■
We next follow with a detailed restatement of Lemma 5.3 and its proof.
Lemma C.4 (cf. Lemma 5.3). Under Assumption 1.1 and Assumption 5.2, with the parameters chosen as:
µ = max{2, 4L f }, τ ≤ min    1 8L K + 16µ , 1 48(L 2 0 + L 2 f )    η = min (µ -L f ) 2 τ 8∥A∥ 2 , 2µ + ρ∥A∥ 4∥A∥ 4 , τ 200∥A∥ 2 , τ (2µ + ρ∥A∥ 2 ) 20∥A∥ 2 , β = min τ 100 , 1 50 , η 36µσ 2 , α = 48(L 2 0 + L 2 f )τ 2 ,(82)
where
L K = L f + ρ∥A∥ + µ, σ is defined in Lemma A.12, we have E Vt -E Vt+1 ≥ µ 2β E∥z t -z t+1 ∥ 2 + 1 8τ E∥x t -x t+1 ∥ 2 + η 2 E∥Ax * (y t+1 , z t ) -b∥ 2 + τ 4 E∥ ∇f t -∇f (x t )∥ 2 -144(L 2 0 + L 2 f )σ 2 τ 3 .(83)
Proof. We denote V t = K(x t , y t , z t ) -2d(y t , z t ) + 2Ψ(z t ).
Joining (81) with Lemma A.8 (since this lemma only uses the update rules of y t+1 , z t+1 that is common in Alg. 1 and Alg.
3), we have
EV t -EV t+1 ≥ -ηE∥Ax t -b∥ 2 + µ β - 3µ 4 E∥z t+1 -z t ∥ 2 - τ 2 E∥∇f (x t ) -∇f t ∥ 2 + 1 2τ - L K 2 -µ E∥x t+1 -x t ∥ 2 + 2ηE⟨Ax t -b, Ax * (y t+1 , z t ) -b⟩ + µE⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩ -2µE⟨z t+1 -z t , z t -x * (z t )⟩ - µ σ 4 E∥z t -z t+1 ∥ 2 .(85)
First, let us combine the first and fifth terms on the right-hand side to obtain
-η∥Ax t -b∥ 2 + 2η⟨Ax t -b, Ax * (y t+1 , z t ) -b⟩ = -η∥Ax t -Ax * (y t+1 , z t )∥ 2 + η∥Ax * (y t+1 , z t ) -b∥ 2 . (86)
Next, we combine the sixth and seventh terms on the right-hand side of (85) to get
µ⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩ -2µ⟨z t+1 -z t , z t -x * (z t )⟩ = µ⟨z t+1 -z t , z t+1 -z t -2x * (y t+1 , z t+1 ) + 2x * (z t )⟩ = µ∥z t+1 -z t ∥ 2 + 2µ⟨z t+1 -z t , -x * (y t+1 , z t+1 ) + x * (z t )⟩.(87)
We now single out the inner product in the last equality and estimate it by adding and subtracting x * (y t+1 , z t ) in the second argument of the inner product:
2µ⟨z t+1 -z t , -x * (y t+1 , z t+1 ) + x * (z t )⟩ = 2µ⟨z t+1 -z t , -x * (y t+1 , z t+1 ) + x * (y t+1 , z t )⟩ + 2µ⟨z t+1 -z t , -x * (y t+1 , z t ) + x * (z t )⟩ ≥ -µ∥z t+1 -z t ∥ 2 -µ∥x * (y t+1 , z t ) -x * (y t+1 , z t+1 )∥ 2 - µ ζ ∥z t+1 -z t ∥ 2 -µζ∥x * (z t ) -x * (y t+1 , z t )∥,(88)
for any ζ, where we used Young's inequality twice. Then, we plug this into (87) to obtain
µ⟨z t+1 -z t , z t+1 + z t -2x * (y t+1 , z t+1 )⟩ -2µ⟨z t+1 -z t , z t -x * (z t )⟩ ≥ - µ σ 2 4 ∥z t+1 -z t ∥ 2 - µ ζ ∥z t+1 -z t ∥ 2 -µζ∥x * (z t ) -x * (y t+1 , z t )∥ 2 ,(89)
where we use (62) to bound the second term on the right-hand side of (88), with σ 4 being as (25).
Then we use ( 86) and ( 89) in (85) to obtain
EV t -EV t+1 ≥ µ β - 3µ 4 E∥z t+1 -z t ∥ 2 - τ 2 E∥∇f (x t ) -∇f t ∥ 2 + 1 2τ - L K 2 -µ E∥x t+1 -x t ∥ 2 -ηE∥Ax t -Ax * (y t+1 , z t )∥ 2 + ηE∥Ax * (y t+1 , z t ) -b∥ 2 - µ σ 2 4 E∥z t+1 -z t ∥ 2 - µ ζ E∥z t+1 -z t ∥ 2 -µζ∥x * (z t ) -x * (y t+1 , z t )∥ 2 - µ σ 4 E∥z t+1 -z t ∥ 2 ≥ µ β - 3µ 4 - µ σ 2 4 - µ ζ - µ σ 4 E∥z t+1 -z t ∥ 2 - τ 2 E∥∇f (x t ) -∇f t ∥ 2 - 2η∥A∥ 2 (µ -L f ) 2 E∥∇f (x t ) -∇f t ∥ 2 + 1 2τ - L K 2 -µ -η∥A∥ 2 2 τ 2 (µ -L f ) 2 E∥x t+1 -x t ∥ 2 + η -µζ σ2 E∥Ax * (y t+1 , z t ) -b∥ 2 ,(90)
this section cite: []

Section: References
Ref_id:b0 Title: Complexity of single loop algorithms for nonlinear programming with stochastic objective and constraints Year: (2024)
Ref_id:b1 Title: Lower bounds for non-convex stochastic optimization Year: (2023)
Ref_id:b2 Title: Constrained optimization and Lagrange multiplier methods Year: (2014)
Ref_id:b3 Title: Convex analysis and optimization Year: (2003)
Ref_id:b4 Title: Stochastic first-order methods for convex and nonconvex functional constrained optimization Year: (2023)
Ref_id:b5 Title: Worstcase complexity of an sqp method for nonlinear equality constrained stochastic optimization Year: (2024)
Ref_id:b6 Title: Momentum-based variance reduction in non-convex sgd Year: (2019)
Ref_id:b7 Title: Stochastic model-based minimization of weakly convex functions Year: (2019)
Ref_id:b8 Title: Training neural networks under physical constraints using a stochastic augmented lagrangian approach Year: (2020)
Ref_id:b9 Title: Efficiency of minimizing compositions of convex functions and smooth maps Year: (2019)
Ref_id:b10 Title: Multiplier and gradient methods Year: (1969)
Ref_id:b11 Title: Convex Analysis and Minimization Algorithms II: Advanced Theory and Bundle Methods Year: (1993)
Ref_id:b12 Title: Decomposing linearly constrained nonconvex problems by a proximal primal dual approach: Algorithms, convergence, and applications Year: (2016)
Ref_id:b13 Title: Single-loop stochastic algorithms for difference of max-structured weakly convex functions Year: (2024)
Ref_id:b14 Title: Training OOD detectors in their natural habitats Year: (2022)
Ref_id:b15 Title: An accelerated inexact dampened augmented lagrangian method for linearlyconstrained nonconvex composite optimization problems Year: (2023)
Ref_id:b16 Title: Complexity of a quadratic penalty accelerated inexact proximal point method for solving linearly constrained nonconvex composite programs Year: (2019)
Ref_id:b17 Title: Iteration complexity of an inner accelerated inexact proximal augmented lagrangian method based on the classical lagrangian function Year: (2023)
Ref_id:b18 Title: First-order and stochastic optimization methods for machine learning Year: (2020)
Ref_id:b19 Title: Rate-improved inexact augmented lagrangian method for constrained nonconvex optimization Year: (2021)
Ref_id:b20 Title: Stochastic inexact augmented lagrangian method for nonconvex expectation constrained optimization Year: (2024)
Ref_id:b21 Title: Complexity of an inexact proximal-point penalty method for constrained smooth non-convex optimization Year: (2022)
Ref_id:b22 Title: Variance-reduced first-order methods for deterministically constrained stochastic nonconvex optimization with strong convergence guarantees Year: (2024)
Ref_id:b23 Title: Quadratically regularized subgradient methods for weakly convex optimization with weakly convex constraints Year: (2020)
Ref_id:b24 Title: Numerical optimization Year: (1999)
Ref_id:b25 Title: An accelerated linearized alternating direction method of multipliers Year: (2015)
Ref_id:b26 Title: Strongly convex functions, moreau envelopes, and the generic nature of convex functions with strong minimizers Year: (2016)
Ref_id:b27 Title: A method for nonlinear constraints in minimization problems. Optimization Year: (1969)
Ref_id:b28 Title: Augmented lagrangians and applications of the proximal point algorithm in convex programming Year: (1976)
Ref_id:b29 Title: Extended nonlinear programming. Nonlinear optimization and related topics Year: (2000)
Ref_id:b30 Title: Adaptive primal-dual stochastic gradient method for expectation-constrained convex stochastic programs Year: (2022)
Ref_id:b31 Title: A proximal alternating direction method of multiplier for linearly constrained nonconvex minimization Year: (2020)
Ref_id:b32 Title: A global dual error bound and its application to the analysis of linearly constrained nonconvex optimization Year: (2022)
Ref_id:b33 Title: A singleloop smoothed gradient descent-ascent algorithm for nonconvex-concave min-max problems Year: (2020)
Ref_id:b34 Title: Decentralized non-convex learning with linearly coupled constraints: Algorithm designs and application to vertical learning problem Year: (2022)
