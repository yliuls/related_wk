Title: Expected Variational Inequalities
Abstract: Variational inequalities (VIs) encompass many fundamental problems in diverse areas ranging from engineering to economics and machine learning. However, their considerable expressivity comes at the cost of computational intractability. In this paper, we introduce and analyze a natural relaxation-which we refer to as expected variational inequalities (EVIs)-where the goal is to find a distribution that satisfies the VI constraint in expectation. By adapting recent techniques from game theory, we show that, unlike VIs, EVIs can be solved in polynomial time under general (nonmonotone) operators. EVIs capture the seminal notion of correlated equilibria, but enjoy a greater reach beyond games. We also employ our framework to capture and generalize several existing disparate results, including from settings such as smooth games, and games with coupled constraints or nonconcave utilities.

Section: Introduction
Variational inequalities (VIs) provide a unifying framework for analyzing a wide range of optimization and equilibrium problems. They have a host of important applications in engineering and economics (Facchinei & Pang, 2003), including identifying stationary points in constrained optimization; computing Nash equilibria in multi-player games (Nash, 1951), such as Cournot's classical model of oligopoly (Cournot, 1838); predicting economic activitycommodity prices and consumer consumption-in a closed, competitive economy (Arrow & Debreu, 1954), which is at the heart of general equilibrium theory; traffic equilibrium Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
problems-estimating the steady-state of a congested network wherein users compete for its resources (Dafermos, 1980); frictional contact problems in mechanical engineering (Capatina, 2014); and pricing options, a foundational problem in financial economics (Black & Scholes, 1973).
Formally, in a general form, a VI can be defined as follows. 1Definition 1.1. Let X be a convex and compact subset of R d and F : X → R d a bounded map. The variational inequality (VI) problem asks for a point x ∈ X such that
⟨F (x), x ′ -x⟩ ≥ 0 ∀x ′ ∈ X .(1)
For computational purposes, it is common to consider the ϵ-approximate VI problem, wherein the right-hand side of (1) is replaced by -ϵ for some precision parameter ϵ > 0.
Definition 1.1 abstracts the description of F and X . As a concrete example, when F : x → -∇u(x) is the negative gradient of a differentiable function u : X → R, the solutions to ( 1) are points that satisfy the first-order optimality conditions for maximizing u (Boyd & Vandenberghe, 2004).
Unfortunately, the considerable expressivity of VIs comes at the expense of intractability: even when F is linear and ϵ is an absolute constant, identifying an ϵ-approximate VI solution is computationally hard; this follows readily from the intractability of Nash equilibria-under plausible complexity assumptions (Daskalakis et al., 2008;Chen et al., 2009;Rubinstein, 2016). Unconditional, query-complexity lower bounds have also been established (Hirsch et al., 1989;Babichenko, 2016); cf. Milionis et al. (2023) and Hart & Mas-Colell (2003) for other pertinent impossibility results.
This bleak realization has shifted the focus of contemporary research primarily to characterizing specific subclasses of VIs that elude those complexity barriers, with the ensuing line of work flourishing in recent years. Some notable examples include the classical Minty property (Facchinei & Pang, 2003;Mertikopoulos & Zhou, 2019), as well as certain relaxations thereof (Diakonikolas et al., 2021;Böhm, 2023;Bauschke et al., 2021;Combettes & Pennanen, 2004;Gorbunov et al., 2023;Cai et al., 2024b;Alacaoglu et al., 2023;Pethick et al., 2022;Lee & Kim, 2021;Patris & Panageas, 2024;Choudhury et al., 2024;Anagnostides et al., 2024).
Despite these important advances, the scope of such results is severely restricted. In this paper, we pursue a different, orthogonal avenue. Instead of restricting the class of problems to achieve computational tractability, we relax the underlying solution concept. Our main research question is:
Are there meaningful relaxations of the VI problem that can always be solved efficiently?
When specialized to games, this question can be seen as part of the research agenda recently outlined by Daskalakis (2022) in his address at the Nobel symposium about equilibrium computation in nonconcave games-a major, new frontier in the interface of game theory and optimization.
this section cite: ['b40', 'b78', 'b4', 'b26', 'b19', 'b13', 'b15', 'b30', 'b22', 'b88', 'b61', 'b7', 'b58', 'b40', 'b72', 'b14', 'b8', 'b24', 'b0', 'b29']

Section: Our Contribution: The Expected VI Problem
To make progress on that central question, we introduce a natural relaxation of VIs (in the context of Definition 1.1).
Definition 1.2. Given a set of deviations Φ ⊆ X X , the ϵ-approximate Φ-expected variational inequality (Φ-EVI) problem asks for a distribution µ ∈ ∆(X ) such that E x∼µ ⟨F (x), ϕ(x) -x⟩ ≥ -ϵ ∀ϕ ∈ Φ.
(The above definition does not specify how X , F, Φ, and µ should be represented for computational purposes, but we will be explicit about representation whenever it is relevant.)
In words, Definition 1.2 only imposes (approximate) nonnegativity in expectation for points x drawn from µ ∈ ∆(X ). It certainly relaxes Definition 1.1: if x satisfies (1), then the distribution µ that always outputs x is also a Φ-EVI solution. Φ-EVIs are thus no harder than VIs (assuming that solutions exist). However, as we shall see, the primary justification of Φ-EVIs is that they can be easier than VIs.
Definition 1.2 is crucially parameterized by Φ; the larger the set of deviations Φ, the tighter the set of solutions. As will become clear, Definition 1.2 is intimately connected with notions of correlated equilibrium (CE) from game theory (e.g., Aumann, 1974). The more permissive case where Φ comprises only constant functions, Φ = Φ CON = {ϕ x : x ∈ X } where ϕ x (x ′ ) = x for all x ′ ∈ X , is perhaps the most basic relaxation of Definition 1.1; we call the Φ CON -EVI problem simply the EVI problem.
Algorithms and complexity for Φ-EVIs As it turns out, imposing no constraints on Φ results in an impasse: Φ-EVIs are in general tantamount to regular VIs-thereby being PPAD-hard (Corollaries 3.7 and 3.9). On the other hand, unlike general VIs, one of our key contributions is to show that when Φ contains only linear maps, Φ LIN , Φ-EVIs can be solved in time polynomial in the dimension d and log(1/ϵ) (Theorem 4.1), establishing the promised computational property that separates EVIs from VIs. This result is based on ellipsoid against hope (EAH), the seminal algorithm of Papadimitriou & Roughgarden (2008) developed for computing correlated equilibria in multi-player games. (Section 2.1 gives a self-contained overview of EAH.) In doing so, we extend the scope of that algorithm to a much broader class of problems well beyond the realm of game theory. Notably, Theorem 4.1 applies even when X is given implicitly through a membership oracle; this extension makes use of the recent technical approach of Daskalakis et al. (2025), discussed in more detail in Section 4.
One limitation of Theorem 4.1 is that it relies on the EAH algorithm, which is slow in practice. We address this by also establishing more scalable algorithms that use convex quadratic optimization (Theorem 4.3) instead of the ellipsoid algorithm. As a byproduct, we obtain the best-known algorithm for linear-swap regret minimization over explicitly represented polytopes, improving on Daskalakis et al. (2025) by reducing the per-iteration complexity.
In addition to their more favorable computational properties, we further show that Φ-EVIs admit (approximate) solutions under more general conditions than their associated VIsnamely, without F being continuous (Theorem 3.1); Section 3 documents further interesting aspects on existence.
Connection to other solution concepts As we have alluded to, Φ-EVIs generalize (Examples 5.1 and 5.2) the seminal concept of a (coarse) correlated equilibrium à la Aumann (1974) and Moulin & Vial (1978) in finite games, and more generally Φ-equilibria (Greenwald & Jafari, 2003;Stoltz & Lugosi, 2007;Gordon et al., 2008) of concave games. What is more surprising is that Φ LIN -EVIs refine CEs even in normal-form games; we give illustrative examples, together with an interpretation, in Section 5. We also note that Φ-EVIs can be used even in games with nonconcave utilities (Daskalakis, 2022;Cai et al., 2024a) or noncontinuous gradients (as in nonsmooth optimization), as well as in (pseudo-)games with coupled constraints (cf. Bernasconi et al., 2023 and Appendix A for related work).
Further properties As further motivation, we show that for certain structured problems, such as (quasar-)concave optimization and polymatrix zero-sum games, EVIs essentially coincide with VIs (Propositions 6.1 and 6.3).
Finally, in certain applications, one might be interested in a VI solution mainly insofar as it provides guarantees in terms of an underlying objective, such as misclassification error or social welfare. Through that prism, the question is whether performance guarantees for VIs can be translated
this section cite: ['b6', 'b80', 'b6', 'b76', 'b53', 'b90', 'b51', 'b29']

Section: Result Description Reference
Existence of (ϵ-approx. to EVIs as well. In Section 7, we establish a framework for accomplishing that (Definition 7.1) by extending the celebrated smoothness framework of Roughgarden (2015), and provide interesting examples beyond game theory.
Taken together, these properties provide compelling justification for Φ-EVIs as a solution concept in place of VIs.
Table 1 gathers our main results. (Proofs are in Appendix C.)
this section cite: ['b84']

Section: Preliminaries
This section provides some basic notation and background together with an overview of the EAH algorithm. Additional preliminaries, which are not necessary for the main body, are given later in Appendix B.
Notation We use boldface, lowercase letters, such as x and y, to denote vectors in a Euclidean space. Capital, boldface letters, such as A, represent matrices. For x, x ′ ∈ R d , we use ⟨x, x ′ ⟩ to denote their inner product. ∥x∥ := ⟨x, x⟩ is the Euclidean norm of x. B r (x) is the (closed) Euclidean ball centered at x with radius r > 0. conv(•) represents the convex hull. An endomorphism on X is a function mapping X to X .
Returning to Definition 1.2, for computational purposes, we assume throughout that F has an explicit polynomial representation, so that F (x) ∈ R d can be evaluated in poly(d) time. Further, there exists B > 0 such that ∥F (x)∥ ≤ B for all x ∈ X . We will also restrict the support supp(µ) of µ to be poly(d, 1/ϵ), unless stated otherwise. With regard to X , we assume that we have oracle access. In particular, we consider the following three types of oracle access.
• Membership: given x ∈ R d , decide whether x ∈ X .
• Separation: given x ∈ R d , decide whether x ∈ X ; if not, return a hyperplane that separates x from X .
• Linear optimization: Given u ∈ R d , return a vector in argmax x∈X ⟨x, u⟩.
In addition, we will assume that X ⊆ B R (0) for some R ≤ poly(d), and X contains a ball of radius 1 in its relative interior; this is a standard regularity condition that ensures X is geometrically well-behaved, which can be met by bringing X into isotropic position (Appendix B). Under this assumption, the three oracles listed above are polynomially equivalent (Grötschel et al., 1993;1981). As a result, we may assume that X is given implicitly via a (poly(d)-time) membership oracle, which suffices for Theorem 4.1.
All our positive results with respect to the set of linear endomorphisms Φ LIN readily carry over to affine endomorphisms as well.
this section cite: ['b55', 'b54']

Section: Ellipsoid Against Hope
This ellipsoid against hope (EAH) algorithm was famously introduced by Papadimitriou & Roughgarden (2008) to compute correlated equilibria in multi-player games. We proceed with an overview of EAH, and in particular a generalized version thereof, crystallized by Farina & Pipis (2024).
Consider an arbitrary optimization problem of the form
find µ ∈ ∆(X ) s.t. E x∼µ ⟨y, G(x)⟩ ≥ 0 ∀y ∈ Y,(2)
where Y ⊆ R m , and G : X → R m is an arbitrary function. Suppose that we are given an evaluation oracle for G and a separation oracle for Y. Assume further that we are given a good-enough-response (GER) oracle, which, given any y ∈ Y, returns x ∈ X such that ⟨y, G(x)⟩ ≥ 0. The upshot is that EAH enables us to solve (2) with just the above tools. Indeed, consider the following problem, which is an ϵ-approximate version of the dual of (2).
find y ∈ Y s.t. ⟨y, G(x)⟩ ≤ -ϵ ∀x ∈ X . (3)
Since a GER oracle exists, (3) is infeasible. What is more, a certificate of infeasibility of (3) yields an ϵ-approximate solution to (2). It thus suffices to run the ellipsoid algorithm on (3) and extract a certificate of infeasibility; in a nutshell, this is what EAH does (cf. Algorithm 1 in Appendix C). Theorem 2.1 (Generalized form of EAH; Farina & Pipis, 2024). Given a GER oracle and a separation oracle (SEP) for Y, EAH runs in time poly(d, m, log(1/ϵ)) and returns an ϵ-approximate solution to (2).
One of our main results (Theorem 4.1) crucially hinges on a strengthening of Theorem 2.1 due to Daskalakis et al. (2025), discussed further in Section 4 and Appendix C.3.
this section cite: ['b80']

Section: Existence and Complexity Barriers
Perhaps the most basic question about Φ-EVIs concerns their totality-the existence of solutions. If one is willing to tolerate an arbitrarily small imprecision ϵ > 0, we show that solutions exist under very broad conditions.
Theorem 3.1. Suppose that F : X → R d is measurable and there exists L > 0 such that every ϕ ∈ Φ is L-Lipschitz continuous. Then, for any ϵ > 0, there exists an ϵ-approximate solution to the Φ-EVI problem.
In particular, our existence proof does not rest on F being continuous. Instead, we consider the continuous function F that maps x → E x∼∆(B δ (x)∩X ) F ( x) (Claim C.1), where B δ (x) is the Euclidean ball centered at x with radius δ = δ(ϵ). It then suffices to invoke Brouwer's fixed-point theorem for the gradient mapping x → Π X (x -F (x)), where Π X is the Euclidean projection with respect to X . Theorem 3.1 implies that a Φ-EVI can have approximate solutions even when the associated VI problem does not. 2   Corollary 3.2. There exists a VI problem that does not admit approximate solutions when ϵ = Θ(1), but the corresponding ϵ-approximate Φ-EVI is total for any ϵ > 0.
In the proof, we set F to be the sign function (Example C.2). By contrast, if one insists on exact solutions, EVIs do not necessarily admit solutions.
Proposition 3.3. When F is not continuous, there exists an EVI problem with no solutions. Furthermore, Theorem 3.1 raises the question of whether it is enough to instead assume that every ϕ ∈ Φ is continuous. Our next result dispels any such hopes.
Theorem 3.4. There are Φ-EVI instances that do not admit ϵ-approximate solutions even when ϵ = Θ(1), F is piecewise constant, and Φ contains only continuous functions.
Our final result on existence complements Theorems 3.1 and 3.4 by showing that when Φ is finite-dimensional, it is enough if every ϕ ∈ Φ admits a fixed point (this holds, for example, when ϕ is continuous-by Brouwer's theorem).
2 Noncontinuity of F manifests itself prominently in nonsmooth optimization (e.g., Zhang et al., 2020;Davis et al., 2022;Tian et al., 2022;Jordan et al., 2023a); recent research there focuses on Goldstein stationary points (Goldstein, 1977), which are conceptually related to EVIs.
Theorem 3.5. Suppose that 1. Φ is finite-dimensional, that is, there exists k ∈ N and a kernel map m : X → R k such that every ϕ ∈ Φ can be expressed as Km(x) for some K ∈ R d×k ; and 2. every ϕ ∈ Φ admits a fixed point, that is, a point X ∋ x = FP(ϕ) such that ϕ(x) = x.
Then, the Φ-EVI problem admits an ϵ-approximate solution with support size at most 1 + dk for every ϵ > 0.
Notably, this theorem guarantees the existence of solutions with finite support; the proof makes use of the minimax theorem (e.g., Sion, 1958) in conjunction with Carathéodory's theorem on convex hulls (Carathéodory, 1911).
Complexity Having established some basic existence properties, we now turn to the complexity of Φ-EVIs.
Let us define the VI gap function VIGap(x) := min x ′ ∈X ⟨F (x), x ′ -x⟩, which is nonnegative. If we place no restrictions on Φ, it turns out that Φ-EVIs are tantamount to regular VIs: Proposition 3.6. If Φ contains all measurable functions from X to X , then any solution µ ∈ ∆(X ) to the ϵapproximate Φ-EVI problem satisfies
E x∼µ VIGap(x) ≤ ϵ.(4)
In proof, it suffices to consider a ϕ that maps x ∈ X to an appropriate point in argmin x ′ ∈X ⟨F (x), x ′ -x⟩. When µ must be given explicitly, Proposition 3.6 immediately implies that Φ-EVIs are computationally hard, because (4) implies that VIGap(x) ≤ ϵ for some x in the support of µ, and such a point can be identified in polynomial time. 3Corollary 3.7. The ϵ-approximate Φ-EVI problem is PPADhard even when ϵ is an absolute constant and F is linear.
Coupled with Proposition 3.6, this follows from the hardness result of Rubinstein (2015) concerning Nash equilibria in (multi-player) polymatrix games (for binary-action, graphical games, Deligkas et al., 2023 recently showed that PPAD-hardness persists up to ϵ < 1 /2). Corollary 3.7 notwithstanding, it is easy to see that the set of solutions to Φ-EVIs is convex for any Φ ⊆ X X . Remark 3.8. Let X = X 1 ×• • •×X n , as in an n-player game. Whether Corollary 3.7 applies under deviations that can be decomposed as ϕ :
x → ϕ(x) = (ϕ 1 (x 1 ), . . . , ϕ n (x n ))
is a major open question in the regime where ϵ ≪ 1 (cf. Dagan et al., 2024;Peng & Rubinstein, 2024).
Viewed differently, a special case of the Φ-EVI problem arises when Φ = {ϕ} and F (x) = xϕ(x), for some fixed map ϕ : X → X . In this case, the Φ-EVI problem reduces to finding a µ ∈ ∆(X ) such that
E x∼µ ⟨F (x), ϕ(x) -x⟩ = -E x∼µ ∥ϕ(x) -x∥ 2 ≥ -ϵ. (5)
As a result, µ must contain in its support an ϵ-approximate fixed point of ϕ, a problem which is PPAD-hard already for quadratic functions (Zhang et al., 2024a).
Corollary 3.9. The ϵ-approximate Φ-EVI problem is PPADhard even when ϵ is an absolute constant, F is quadratic, and Φ = {ϕ} for a quadratic map ϕ : X → X .
It is also worth noting that, unlike Corollary 3.7, Φ in the corollary above contains only continuous functions.
It also follows from (5) that for ϵ = 0, Φ-EVIs capture exact fixed points. The complexity class FIXP characterizes such problems (Etessami & Yannakakis, 2007).
Corollary 3.10. The Φ-EVI problem is FIXP-hard, assuming that supp(µ) ≤ poly(d).
Exponential lower bounds in terms of the number of function evaluations of F also follow from Hirsch et al. (1989).
On a positive note, the next section establishes polynomialtime algorithms when Φ contains only linear endomorphisms.foot_3
this section cite: ['b49', 'b89', 'b21', 'b87', 'b38', 'b61']

Section: Efficient Computation with Linear Maps
The hardness results of the previous section highlight the need to restrict the set Φ in order to make meaningful progress. Our main result here establishes a polynomialtime algorithm when Φ contains only linear endomorphisms.
Theorem 4.1. If Φ contains only linear endomorphisms, the ϵ-approximate Φ-EVI problem can be solved in time poly(d, log(B/ϵ)) given a membership oracle for X .
The proof relies on the ellipsoid against hope (EAH), and in particular, a recent generalization by Daskalakis et al. (2025). In a nutshell, the main deficiency in the framework covered earlier in Section 2.1 is that one needs a separation oracle for Y (Theorem 2.1), where Y for us is the set of deviations Φ. Unlike some applications, in which Y has an explicit, polynomial representation (Papadimitriou & Roughgarden, 2008), that assumption needs to be relaxed to account for Φ LIN (Daskalakis et al., 2025, Theorem 3.4). Daskalakis et al. (2025) address this by considering instead the SEPorGER oracle. As the name suggests, for any y ∈ R m , it either returns a hyperplane separating y from Y, or a good-enough-response x ∈ X . They showed that Theorem 2.1 can be extended under this weaker oracle (in place of GER and SEP); the formal version is given in Theorem C.4.
In our setting, we consider the feasibility problem
find ϕ ∈ Φ LIN s.t. (6) ⟨F (x), ϕ(x) -x⟩ ≤ -ϵ ∀x ∈ X . Equivalently, find K ∈ R d×d s.t. ⟨F (x), Kx -x⟩ ≤ -ϵ ∀x ∈ X , Kx ∈ X ∀x ∈ X .
This program is infeasible since, for any ϕ ∈ Φ LIN , the fixed point x of ϕ makes the left-hand side of the constraint 0. And a certificate of infeasibility is an ϵ-approximate Φ LIN -EVI solution. Thus, it suffices to show how to run the ellipsoid algorithm on (6). By Theorem C.4, it suffices if for any K ∈ R d×d , we can compute efficiently either
• some x ∈ X such that Kx = x (GER), or • some hyperplane separating K from Φ LIN (SEP).
This is precisely the semi-separation oracle solved by Daskalakis et al. (2025, Lemma 4.1), stated below. Lemma 4.2 (Daskalakis et al., 2025). There is an algorithm that takes K ∈ R d×d , runs in poly(d) time, makes poly(d) oracle queries to X , and either returns a fixed point X ∋ x = Kx, or a hyperplane separating K from Φ LIN .
On a separate note, Theorem 4.1 only accounts for approximate solutions. We cannot hope to improve that in the sense that exact solutions might be supported only on irrational points even in concave maximization (cf. Proposition 6.3).
this section cite: ['b80']

Section: Regret Minimization for EVIs on Polytopes
One caveat of Theorem 4.1 is that it relies on the impractical EAH algorithm. To address this limitation, we will show that Φ-EVIs are also amenable to the more scalable approach of regret minimization-albeit with an inferior complexity growing as poly(1/ϵ).
Specifically, in our context, the regret minimization framework can be applied as follows. At any time t ∈ N, we think of a "learner" selecting a point x (t) ∈ X , whereupon F (x (t) ) is given as feedback from the "environment," so that the utility at time t reads -⟨x (t) , F (x (t) )⟩. Φ-regret is a measure of performance in online learning, defined as
Φ-Reg (T ) := max ϕ∈Φ T t=1 ⟨F (x (t) ), ϕ(x (t) ) -x (t) ⟩. The uniform distribution µ on {x (1) , . . . , x (T ) } is clearly a Φ-Reg (T ) /T -approximate Φ-EVI solution.
In what follows, we will assume that X is a polytope given explicitly by linear constraints, i.e., X = {x ∈ R d : Ax ≤ b}, where A ∈ Q m×d and b ∈ Q m are given as input.
To minimize Φ-regret, we will make use of the template by Gordon et al. (2008), which comprises two components. The first is a fixed-point oracle, which takes as input a function ϕ ∈ Φ LIN and returns a point x ∈ X with x = ϕ(x); given that ϕ is linear, it can be implemented efficiently via linear programming. The second component is an algorithm for minimizing (external) regret over the set Φ LIN . In Theorem D.1, we devise a polynomial representation for Φ LIN : Theorem 4.3. For an arbitrary polytope X given by explicit linear constraints, there is an explicit representation of Φ LIN as a polytope with O(d 2 + m 2 ) variables and constraints.
As a consequence, we can instantiate the regret minimizer operating over Φ LIN with projected gradient descent. Corollary 4.4. There is a deterministic algorithm that guarantees Φ LIN -Reg (T ) ≤ ϵ after poly(d, m)/ϵ 2 rounds, and requires solving a convex quadratic program with O(d 2 + m 2 ) variables and constraints in each iteration.
An additional benefit of Corollary 4.4 compared to using EAH is that the former is more suitable in a decentralized environment-for example, in multi-player games (cf. Example 5.1). There, Corollary 4.4 corresponds to each player running their own independent no-regret learning algorithm. Even in this setting, our algorithms actually yield an improvement over the best-known algorithms for minimizing Φ LIN -regret over explicitly-represented polytopes: the previous state of the art, due to Daskalakis et al. (2025), requires running the ellipsoid algorithm on each iteration, which is slower than quadratic programming (Appendix D).
this section cite: ['b51']

Section: Game Theory Applications of EVIs
A major motivation for studying Φ-EVIs lies in a strong connection to (C)CEs (Aumann, 1974;Moulin & Vial, 1978) in games. Indeed, we begin this section by pointing out that Φ-EVIs capture (C)CEs for specific choices of Φ.
We will mostly consider n-player concave games. Here, each player i ∈ [n] selects a strategy x i ∈ X i from some convex and compact set X i , and its utility is given by u
i : (x 1 , . . . , x n ) → R.
We assume that u i (x i , x -i ) is differentiable and concave in x i for any x -i , and that the gradients (Moulin & Vial, 1978) if for any player i ∈ [n],
∇ xi u i (x i , x -i ) are bounded. We let X := X 1 × • • • × X n . Example 5.1 (CCE). A distribution µ ∈ ∆(X ), is an ϵ- coarse correlated equilibrium (CCE)
δ i := max x ′ i ∈Xi E x∼µ u i (x ′ i , x -i ) -E x∼µ u i (x) ≤ ϵ. (7)
Now, consider an ϵ-approximate EVI solution µ of the problem defined by
F := (-∇ x1 u 1 (x), . . . , -∇ xn u n (x)).
Such µ satisfies, by concavity, n i=1 δ i ≤ ϵ; it is not necessarily an ϵ-approximate CCE since it is possible that for some i ∈ [n], all deviations strictly decrease i's utility (so that δ i in (7) is negative)-µ is technically an average CCE in the parlance of Nadav & Roughgarden (2010). To capture CCE via Φ-EVIs, one can instead consider a richer set of deviations of the form (x 1 , . . . ,
x n ) → (x 1 , . . . , x ′ i , . . . , x n ) for all i ∈ [n] and x ′ i ∈ X i .
A canonical example of the above formalism is a normalform game, in which each constraint set X i is the probability simplex ∆(A i ) over a finite set of actions A i , and each utility u i is a multilinear function.
Example 5.2 (LCE). A distribution µ ∈ ∆(X ) is an ϵ- linear correlated equilibrium (LCE) if for any i ∈ [n], max ϕi∈Φi E x∼µ u i (ϕ i (x i ), x -i ) -E x∼µ u i (x) ≤ ϵ,
where
Φ i contains all linear functions from X i to X i . To capture LCE via Φ-EVIs, it suffices to consider deviations of the form (x 1 , . . . , x n ) → (x 1 , . . . , ϕ i (x i ), . . . , x n ) for all i ∈ [n] and ϕ i ∈ Φ i .
For normal-form games, LCEs amount to the usual notion of CEs (Aumann, 1974). LCEs were introduced in the context of extensive-form games (Farina & Pipis, 2023;2024).
Refining correlated equilibria In fact, and more surprisingly, Φ LIN -EVI solutions can be a strict subset of LCEs. 5This separation can already be appreciated in the setting of normal-form games, and manifests itself in at least two distinct ways. First, there exist games for which a CE need not be a solution to the Φ LIN -EVI. In this sense, Φ LIN -EVIs yield a computationally tractable superset of Nash equilibria that is tighter than CEs. Second, computation suggests that the set of solutions of the Φ LIN -EVI for the game need not be a polyhedron, unlike the set of CEs. We provide a graphical depiction of this phenomenon in Figure 1. The figure depicts the set of Φ LIN -EVI solutions to a simple "Bach or Stravinsky" game, in which the players receive payoffs (3, 2) if they both pick Bach, (2, 3) if they both pick Stravinsky, and (0, 0) otherwise.
Interpretation The reason for this separation is that, for a map ϕ : X → X , each player's mapped strategy ϕ(x) i can also depend (linearly) on other players' strategies x -i . Indeed, the EVI formulation of a game does not take into account the identities of the players. For this reason, we will call the set of Φ LIN -EVI solutions in a concave game anonymous linear correlated equilibria, or ALCE for short. We give two game-theoretic interpretations of ALCEs.
First, the ALCEs of a game Γ are the symmetric LCEs of the "symmetrized" game in which the players are randomly shuffled before the game begins. That is, consider the nplayer game Γ sym defined as follows. Each player's strategy set is X . For strategy profile (x 1 , . . . , x n ) ∈ X n , the utility to player i is given by
u sym i (x 1 , . . . , x n ) = 1 n! σ∈Gn u σ(i) (x σ -1 (1) 1 , . . . , x σ -1 (n) n ),
where
G n is the set of permutations σ : [n] → [n].
The following result then follows almost by definition.
Proposition 5.3. For a given distribution µ ∈ ∆(X ), define the distribution µ n ∈ ∆(X n ) by sampling x ∼ µ and outputting (x, . . . , x) ∈ X n . Then, µ is a ALCE of Γ if and only if µ n is an LCE of Γ sym .
Second, for normal-form games, the ALCEs are the distributions µ ∈ ∆(X ) such that no player i has a profitable deviation of the following form. The correlation device first samples x ∼ µ, and samples recommendations a j ∼ x j for each player j. Then, the player selects another player j (possibly j = i) whose recommendation it wishes to see. The player then observes a sample a ′ j ∼ x j that is independently sampled from a j . 6 Finally, the player chooses an action a * i ∈ A i , and each player j gets reward u j (a * i , a -i ). Thus, players are allowed (modulo the independent sampling) to spy on each others' recommendations.
Further discussion about ALCEs and formal proofs of the claims in this section are deferred to Appendix F.
this section cite: ['b6', 'b76', 'b76', 'b77', 'b6']

Section: Coupled constraints
Continuing from Examples 5.1 and 5.2, we observe that (Φ LIN -)EVIs can be used even in "pseudo-games," in which X does not necessarily decompose into X 1 × • • • × X n ; this means that x i ∈ X i (x -i ). As we discuss in Appendix A, most prior work in such settings has focused on generalized Nash equilibria, with the exception of Bernasconi et al. (2023). (Φ LIN -)EVIs induce an interesting notion of LCE/CCE in pseudo-games, albeit not directly comparable to the one put forward by Bernasconi et al. (2023). It is worth noting that Bernasconi et al. (2023) left open whether efficient algorithms for computing their notion of (coarse) correlated equilibria exist. Definition 5.4. Given an n-player pseudo-game with concave, differentiable utilities and joint constraints
X , a distri- bution µ ∈ ∆(X ) is an ϵ-ALCE if max ϕ∈Φ LIN E x∼µ n i=1 u i (ϕ(x) i , x -i ) - n i=1 u i (x) ≤ ϵ.
By virtue of our main result (Theorem 4.1), such an equilibrium can be computed in polynomial time.
Noncontinuous gradients In fact, our results do not rest on the usual assumption that each player's gradient is a continuous function, thereby significantly expanding the scope of prior known results even in games. For example, we refer to Dasgupta & Maskin (1986); Bichler et al. (2021); Martin & Sandholm (2024) for pointers to some applications.
Nonconcave games Last but not least, Φ-EVIs give rise to a notion of local Φ-equilibrium (Definition G.1) in nonconcave games. It turns out that this captures recent results by Cai et al. (2024a) and S ¸eref Ahunbay (2025), but our framework has certain important advantages. First, we give a poly(d, log(1/ϵ))-time algorithm (Theorem 4.1), while theirs scale polynomially in 1/ϵ. Second, our results do not assume continuity of the gradients. And finally, our algorithms are polynomial even when Φ contains all linear endomorphisms (Theorem 4.1). Appendix G elaborates further on those points.
this section cite: ['b28', 'b11', 'b71']

Section: Problems Where EVIs Coincide with VIs
We saw earlier, in Proposition 3.6, that when Φ comprises all functions from X to X , the Φ-EVI problem is tantamount to the associated VI problem. However, if one restricts the functions contained in Φ, are there still structured VIs where we retain this equivalence? In this section, we consider certain structured VIs, and show their equivalence to the corresponding EVIs (that is, Φ CON -EVIs). Unlike general VIs, the ones we examine below are tractable.
this section cite: []

Section: Polymatrix Zero-Sum Games and Beyond
The first important class of VIs we consider is described by a condition given below.
Proposition 6.1. Suppose that for any x ′ ∈ X , the function g : x → ⟨F (x), x ′ -x⟩ is concave. Then, if µ ∈ ∆(X ) is an ϵ-approximate solution to the EVI, E x∼µ x is an ϵapproximate solution to the VI.
The proof follows directly from Jensen's inequality.
The precondition of Proposition 6.1 is satisfied, e.g., when: (i) ⟨F (x), x⟩ = 0 for all x ∈ X , and (ii) F is a linear map. In the context of n-player games, the first condition amounts to the zero-sum property: n i=1 u i (x) = 0 for all x. Of course, this property is not enough to enable efficient computation of Nash equilibria, for every two-player (general-sum) game can be converted into a 3-player zero-sum game. This is where the second condition comes into play: F is a linear map-that is, each player's gradient must be linear in the joint strategy. Those two conditions are satisfied in polymatrix zero-sum games (Cai et al., 2016); in such games, the conclusion of Proposition 6.1 is a well-known fact.
this section cite: ['b16']

Section: Quasar-Concave Functions
We next consider the problem of maximizing a (single) function that satisfies quasar-concavity-a natural generalization of concavity that has received significant interest (Hardt et al., 2018;Fu et al., 2023;Hinder et al., 2020;Gower et al., 2021;Guminov et al., 2023;Caramanis et al., 2024). 7   Definition 6.2 (Quasar-concavity). Let γ ∈ (0, 1] and x ⋆ ∈ X be a maximizer of a differentiable function u : X → R. We say that u is γ-quasar-concave with respect to
x ⋆ if u(x ⋆ ) ≤ u(x) + 1 γ ⟨∇u(x), x ⋆ -x⟩ ∀x ∈ X . (8)
In particular, in the special case where γ = 1, (8) is equivalent to star-concavity (Nesterov & Polyak, 2006). If in addition (8) holds for all x ⋆ ∈ X (not merely w.r.t. a global maximizer), it captures the usual notion of concavity. 7 Prior literature mostly uses the term quasar-convexity, which is equivalent to quasar-concavity for the opposite function -u.
Any reasonable solution concept for such problems should place all mass on global maxima; EVIs pass this litmus test: Proposition 6.3. Let F = -∇u for a γ-quasar-concave and differentiable function u : X → R. Then, for any solution µ ∈ ∆(X ) to the EVI problem,
E x∼µ u(x) ≥ max x∈X u(x). Thus, P x∼µ [u(x ⋆ ) = u(x)] = 1, for x ⋆ ∈ argmax x u(x). Indeed, by Definition 6.2, 0 ≤ E x∼µ ⟨∇u(x), x -x ⋆ ⟩ ≤ γ E x∼µ [u(x) -u(x ⋆ )] for any EVI solution µ ∈ ∆(X ).
Thus, under quasar-concavity, VIs basically reduce to EVIs.
this section cite: ['b57', 'b60', 'b56', 'b79']

Section: Performance Guarantees for EVIs
In many settings, a VI solution is used as a proxy to approximately maximize some underlying objective function; machine learning offers many such applications. The question is whether performance guarantees pertaining to VIs can be extended-potentially with some small degradationto EVIs as well. The purpose of this section is to provide a framework for achieving that based on the following notion.
Definition 7.1. An EVI problem is (λ, ν)-smooth, for λ > 0, ν > -1, w.r.t. W : X → R and x ⋆ ∈ argmax x W (x) if ⟨F (x), x ⋆ -x⟩ ≤ -λW (x ⋆ )+(ν +1)W (x) ∀x ∈ X .
Example 7.2. When the underlying problem corresponds to a multi-player game and W is the (utilitarian) social welfare, Definition 7.1 coincides with the celebrated notion of smoothness à la Roughgarden (2015); this is a consequence of multilinearity, which implies that W (x) = -⟨x, F (x)⟩ and ⟨x ⋆ , F (x)⟩ = -n i=1 u i (x ⋆ i , x -i ) for all x ∈ X . We also refer to the recent treatment of smoothness by S ¸eref Ahunbay (2025) in the context of nonconcave games, which builds on the primal-dual framework of Nadav & Roughgarden (2010).
Definition 7.1 is an extension of the more general notion of "local smoothness," introduced by Roughgarden & Schoppmann (2015) in the context of splittable congestion games. However, it goes beyond games. Indeed, the following definition we introduce generalizes Definition 6.2, making a new connection between smoothness and quasar-concavity.
Definition 7.3 (Extension of quasar-concavity). Let x ⋆ ∈ X be a maximizer of a differentiable function u : X → R. We say that u is (λ, ν)-smooth with respect to
x ⋆ if ⟨∇u(x), x ⋆ -x⟩ ≥ λu(x ⋆ ) -(ν + 1)u(x) ∀x ∈ X .
In particular, when λ := γ and ν := γ -1, the above definition captures γ-quasar-concavity. In Appendix E, we provide an example of a polynomial that satisfies Definition 7.3 without being quasar-concave. Now, the key property of Definition 7.1 is that any EVI solution approximates the underlying objective-by a factor of ρ := λ /1+ν. Theorem 7.4. Let µ ∈ ∆(X ) be an ϵ-approximate solution to a (λ, ν)-smooth EVI problem w.r.t. W : X → R. Then,
E x∼µ W (x) ≥ λ 1 + ν max x∈X W (x) - ϵ 1 + ν .
The proof follows directly from Definition 7.1, using that E x∼µ ⟨F (x), x ⋆ -x⟩ ≥ -ϵ and linearity of expectation.
this section cite: ['b84', 'b77', 'b85']

Section: Conclusions and Future Research
In summary, our main contribution was to introduce and examine a natural relaxation of VIs, which we refer to as expected VIs. Unlike VIs, which are marred by computational intractability, we showed that EVIs can be solved efficiently under minimal assumptions. We also uncovered many other intriguing properties of EVIs (cf. Table 1).
There are many promising avenues for future work. VIs enjoy a great reach in a wide range of applications, some of which were discussed earlier in our introduction. It would be interesting to explore in more detail how EVIs fare in such settings compared to VIs. In particular, given that EVIs relax VIs, in addition to their more favorable computational properties, it is likely that they unlock new, more desirable solutions not present under VIs. For example, it is well known (e.g., Ashlagi et al., 2005) that CEs can achieve better welfare than Nash equilibria in games. In light of the prominence of correlated equilibria in the rich setting of multi-player games, we anticipate EVIs to solidify their place also in other application areas beyond the realm of game theory.
this section cite: ['b5']

Section: References
Ref_id:b0 Title: Beyond the golden ratio for variational inequality algorithms Year: (2023)
Ref_id:b1 Title: Infinite Dimensional Analysis: a Hitchhiker's Guide Year: (2006)
Ref_id:b2 Title: Optimistic policy gradient in multi-player Markov games with a single controller: Convergence beyond the Minty property Year: ()
Ref_id:b3 Title: Generalized Nash equilibria for the service provisioning problem in cloud systems Year: (2012)
Ref_id:b4 Title: Existence of an equilibrium for a competitive economy Year: (1954)
Ref_id:b5 Title: On the value of correlation Year: (2005)
Ref_id:b6 Title: Subjectivity and correlation in randomized strategies Year: (1974)
Ref_id:b7 Title: Query complexity of approximate Nash equilibria Year: (2016)
Ref_id:b8 Title: Generalized monotone operators and their averaged resolvents Year: (2021)
Ref_id:b9 Title: Constrained Phi-equilibria Year: ()
Ref_id:b10 Title: On the role of constraints in the complexity of min-max optimization Year: (2024)
Ref_id:b11 Title: Learning equilibria in symmetric auction games using artificial neural networks Year: (2021)
Ref_id:b12 Title: Convergence of Probability Measures Year: (1999)
Ref_id:b13 Title: The pricing of options and corporate liabilities Year: (1973)
Ref_id:b14 Title: Solving nonconvex-nonconcave min-max problems exhibiting weak Minty solutions Year: (2023)
Ref_id:b15 Title: Convex Optimization Year: (2004)
Ref_id:b16 Title: Zero-sum polymatrix games: A generalization of minmax Year: (2016)
Ref_id:b17 Title: On tractable Φ-equilibria in non-concave games Year: ()
Ref_id:b18 Title: Accelerated algorithms for constrained nonconvex-nonconcave min-max optimization and comonotone inclusion Year: (2024)
Ref_id:b19 Title: Variational inequalities and frictional contact problems Year: (2014)
Ref_id:b20 Title: Optimizing solution-samplers for combinatorial problems: the landscape of policy-gradient methods Year: ()
Ref_id:b21 Title: Uber den Variabiletätsbereich der Fourier'schen Konstanten von positiven harmonischen Funktionen Year: (1911)
Ref_id:b22 Title: Settling the complexity of computing two-player Nash equilibria Year: (2009)
Ref_id:b23 Title: Singlecall stochastic extragradient methods for structured nonmonotone variational inequalities: Improved analysis under weaker conditions Year: ()
Ref_id:b24 Title: Proximal methods for cohypomonotone operators Year: (2004)
Ref_id:b25 Title: Recherches sur les principes mathématiques de la théorie des richesses (Researches into the Mathematical Principles of the Theory of Wealth) Year: ()
Ref_id:b26 Title: Traffic equilibrium and variational inequalities Year: (1980)
Ref_id:b27 Title: From external to swap regret 2.0: An efficient reduction for large action spaces Year: ()
Ref_id:b28 Title: The existence of equilibrium in discontinuous economic games 1: Theory Year: (1986)
Ref_id:b29 Title: Non-concave games: A challenge for game theory's next 100 years Year: (2022)
Ref_id:b30 Title: The complexity of computing a Nash equilibrium Year: (2008)
Ref_id:b31 Title: The complexity of constrained min-max optimization Year: ()
Ref_id:b32 Title: A lower bound on swap regret in extensiveform games Year: (2024)
Ref_id:b33 Title: Efficient learning and computation of linear correlated equilibrium in general convex games Year: ()
Ref_id:b34 Title: A gradient sampling method with complexity guarantees for Lipschitz functions in high and low dimensions Year: ()
Ref_id:b35 Title: Tight inapproximability for graphical games Year: ()
Ref_id:b36 Title: Efficient methods for structured nonconvex-nonconcave min-max optimization Year: ()
Ref_id:b37 Title: A mean-field analysis of twoplayer zero-sum games Year: (2020)
Ref_id:b38 Title: On the complexity of Nash equilibria and other fixed points Year: (2007)
Ref_id:b39 Title: Generalized Nash equilibrium problems Year: (2010)
Ref_id:b40 Title: Finite-dimensional variational inequalities and complementarity problems Year: (2003)
Ref_id:b41 Title: Generalized Nash equilibrium problems and Newton methods Year: (2009)
Ref_id:b42 Title: Polynomial-time linear-swap regret minimization in imperfect-information sequential games Year: ()
Ref_id:b43 Title: Polynomial-time computation of exact Phi-equilibria in polyhedral games Year: ()
Ref_id:b44 Title: Simple uncoupled no-regret learning dynamics for extensive-form correlated equilibrium Year: (2022)
Ref_id:b45 Title: Generalized Nash equilibrium problems-recent advances and challenges Year: (2014)
Ref_id:b46 Title: Accelerated stochastic optimization methods under quasar-convexity Year: ()
Ref_id:b47 Title: Bayes correlated equilibria and no-regret dynamics Year: (2023)
Ref_id:b48 Title: Exploitability minimization in games and beyond Year: ()
Ref_id:b49 Title: Optimization of Lipschitz continuous functions Year: (1977)
Ref_id:b50 Title: Convergence of proximal point and extragradient-based methods beyond monotonicity: the case of negative comonotonicity Year: ()
Ref_id:b51 Title: No-regret learning in convex games Year: (2008)
Ref_id:b52 Title: SGD for structured nonconvex functions: Learning rates, minibatching and interpolation Year: ()
Ref_id:b53 Title: A general class of no-regret learning algorithms and game-theoretic equilibria Year: (2003)
Ref_id:b54 Title: The ellipsoid method and its consequences in combinatorial optimization Year: (1981)
Ref_id:b55 Title: Geometric algorithms and combinatorial optimization Year: (1993)
Ref_id:b56 Title: Accelerated methods for weakly-quasi-convex optimization problems Year: (2023)
Ref_id:b57 Title: Gradient descent learns linear dynamical systems Year: (2018)
Ref_id:b58 Title: Uncoupled dynamics do not lead to Nash equilibrium Year: (2003)
Ref_id:b59 Title: Introduction to online convex optimization Year: (2016)
Ref_id:b60 Title: Near-optimal methods for minimizing star-convex functions and beyond Year: (2020)
Ref_id:b61 Title: Exponential lower bounds for finding Brouwer fix points Year: (1989)
Ref_id:b62 Title: Finding mixed Nash equilibria of generative adversarial networks Year: (2019)
Ref_id:b63 Title: Computing an extensiveform correlated equilibrium in polynomial time Year: (2008)
Ref_id:b64 Title: Deterministic nonsmooth nonconvex optimization Year: ()
Ref_id:b65 Title: First-order algorithms for nonlinear generalized Nash equilibrium problems Year: (2023)
Ref_id:b66 Title: The computational complexity of variational inequalities and applications in game theory Year: (2024)
Ref_id:b67 Title: An introduction to variational inequalities and their applications Year: (2000)
Ref_id:b68 Title: Extensive games and the problem of information Year: (1953)
Ref_id:b69 Title: Fast extra gradient methods for smooth structured nonconvex-nonconcave minimax problems Year: ()
Ref_id:b70 Title: Simulated annealing in convex bodies and an O * (n 4 ) volume algorithm Year: (2006)
Ref_id:b71 Title: Joint-perturbation simultaneous pseudo-gradient Year: (2024)
Ref_id:b72 Title: Learning in games with continuous action sets and unknown payoff functions Year: (2019)
Ref_id:b73 Title: An impossibility theorem in game dynamics Year: ()
Ref_id:b74 Title: Efficient deviation types and learning for hindsight rationality in extensive-form games Year: ()
Ref_id:b75 Title: Hindsight and sequential rationality of correlated play Year: ()
Ref_id:b76 Title: Strategically zero-sum games: The class of games whose completely mixed equilibria cannot be improved upon Year: (1978)
Ref_id:b77 Title: The limits of smoothness: A primal-dual framework for price of anarchy bounds Year: (2010)
Ref_id:b78 Title: Non-cooperative games Year: (1951)
Ref_id:b79 Title: Cubic regularization of newton method and its global performance Year: (2006)
Ref_id:b80 Title: Computing correlated equilibria in multi-player games Year: (2008)
Ref_id:b81 Title: Learning Nash equilibria in rank-1 games Year: ()
Ref_id:b82 Title: Fast swap regret minimization and applications to approximate correlated equilibria Year: ()
Ref_id:b83 Title: Escaping limit cycles: Global convergence for constrained nonconvex-nonconcave minimax problems Year: ()
Ref_id:b84 Title: Intrinsic robustness of the price of anarchy Year: (2015)
Ref_id:b85 Title: Local smoothness and the price of anarchy in splittable congestion games Year: (2015)
Ref_id:b86 Title: The price of anarchy in auctions Year: (2017)
Ref_id:b87 Title: Inapproximability of Nash equilibrium Year: (2015)
Ref_id:b88 Title: Settling the complexity of computing approximate two-player Nash equilibria Year: (2016)
Ref_id:b89 Title: On general minimax theorems Year: (1958)
Ref_id:b90 Title: Learning correlated equilibria in games with compact sets of strategies Year: (2007)
Ref_id:b91 Title: Learning generalized Nash equilibria in a class of convex games Year: (2018)
