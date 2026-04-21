Title: Generalized Gradient Norm Clipping & Non-Euclidean (L 0 , L 1 )-Smoothness
Abstract: This work introduces a hybrid non-Euclidean optimization method which generalizes gradient norm clipping by combining steepest descent and conditional gradient approaches. The method achieves the best of both worlds by establishing a descent property under a generalized notion of (L 0 ,L 1 )-smoothness. Weight decay is incorporated in a principled manner by identifying a connection to the Frank-Wolfe short step. In the stochastic case, we show an order optimal O(n -1/4 ) convergence rate by leveraging a momentum based gradient estimator. We discuss how to instantiate the algorithms for deep learning, which we dub Clipped Scion, and demonstrate their properties on image classification and language modeling. The code is available at https://github.com/LIONS-EPFL/ClippedScion. * Equal contribution. 2 By conditional gradient based methods, we mean those methods which leverage a linear minimization oracle lmo(d) = arg min x∈D ⟨d, x⟩ when updating their parameters with an open-loop stepsize. 39th Conference on Neural Information Processing Systems (NeurIPS 2025).

Section: Introduction
Recent work [Pethick et al., 2025] has shown that conditional gradient methods 2 , traditionally used for constrained optimization, can also solve unconstrained problems-offering an alternative to steepest descent. From their analysis it becomes apparent that the two methods have distinct properties: whereas steepest descent requires the stepsize γ for L-smooth objectives to be taken as γ < 2 /L, conditional gradient methods have no such requirement, thus allowing for large stepsizes, while remaining stable.
The price to pay for the stability is that conditional gradient based methods are not descent methods and thus eventually needs a diminishing stepsize to converge, even in the deterministic case. The problem becomes very apparent if the iterates are close to the solution, since the iterates always move by a fixed magnitude and are thus pushed away from the solution. Steepest descent does not suffer from the same problem since the effective stepsize automatically becomes smaller as the iterates approach a solution. This observation naturally raises the following question: Table 1: Special instantiations of Algorithm 1 according to different choices of norm. Control on the norm of the parameters is guaranteed by the constrained variant of the method (Algorithm 2).
Method Norm type Norm ball lmo(d) ∥d∥ * Reference Clipped GD Vector Euclidean ∥ • ∥2-ball -d ∥d∥2 ∥d∥2 [Mikolov et al., 2012] Clipped Sign Vector Max-norm ∥ • ∥∞-ball sign(d) ∥d∥1 This paper Clipped Spectral Matrix Spectral norm ∥ • ∥S ∞ -ball -UV ⊤1 tr(lmo(d) ⊤ d) This paper Clipped Scion (Algorithms 3 and 4) Product Max-norm ball over layers {rl lmo∥•∥ W l (dl)}l∈[D]l ⟨rl lmo(dl), dl⟩ This paper 1 The reduced SVD is given as d = U diag(σ)V ⊤ .
Can we combine the two methods and get the best of both worlds? That is, does a stable method exist which takes large steps initially but adapts the stepsize when near a solution?
In this paper we answer the above in the affirmative by considering a hybrid method that combines a conditional gradient method with steepest descent. The proposed method generalizes gradient norm clipping [Mikolov et al., 2012] beyond the Euclidean case. In practice, gradient norm clipping has been widely adopted to stabilize training of recurrent neural networks (RNNs), Transformers and diffusion models, especially in large-scale settings. Theoretically, a precise characterization of the benefits has emerged under the (L 0 , L 1 )-smoothness assumption [Zhang et al., 2019, 2020, Koloskova et al., 2023]. Expanding on this, we show that these benefits of clipping can be made compatible with non-Euclidean methods. Besides clipping, we provide a novel analysis of conditional gradient methods without clipping under these same smoothness assumptions.
Concretely, we make the following contributions:
(i) We introduce a hybrid method between a conditional gradient method and steepest descent (Algorithm 1), which in the Euclidean case recovers gradient norm clipping. The benefit of the hybrid method is made precise by showing a descent property under a generalized (L 0 , L 1 )-smoothness condition.
(ii) In the stochastic case we show an order optimal O(n -1/4 ) rate by leveraging a momentum estimator. Convergence for a clipped algorithm with stochastic feedback appears to be new even in the Euclidean case.
(iii) We establish a connection between clipping and the short step from the Frank-Wolfe literature, which similarly enjoys a descent property. The connection enables us to combine clipping with weight decay in a principled manner that maintains convergence guarantees. We propose a stochastic variant of the short step (Algorithm 2) and establish a O(n -1/4 ) rate.
(iv) We explicitly instantiate the algorithms for deep learning through a product norm over layers (Algorithms 3 and 4) and demonstrate their properties through experiments on image classification and language modeling.
this section cite: ['b30', 'b26', 'b46', 'b20']

Section: Preliminaries
Given a continuously differentiable objective function f : X → R, the classical gradient descent method (GD) with a stepsize γ > 0 can be written as
x k+1 = arg min x∈X γ⟨∇ f (x k ), x⟩ + 1 2 ∥x -x k ∥ 2 2 = x k -γ∇ f (x k ).(GD)
The normalized gradient descent method with radius ρ > 0 is, in comparison, defined as follows
x k+1 = arg min ∥x-x k ∥ 2 ≤ρ γ⟨∇ f (x k ), x⟩ = x k + ρ arg min ∥x∥ 2 ≤1 γ⟨∇ f (x k ), x⟩ = x k -γ ρ ∇ f (x k ) ∥∇ f (x k )∥ 2 . (Normalized GD)
A hybrid variant is much more popular in practice,
x k+1 = arg min ∥x-x k ∥ 2 ≤ρ γ⟨∇ f (x k ), x⟩ + 1 2 ∥x -x k ∥ 2 2 = x k -γ min{1, ρ ∥∇ f (x k )∥ 2 }∇ f (x k ), (Clipped GD)
which we notice can be rewritten by combining GD and Normalized GD. Indeed, all three of these algorithms correspond to minimizing
γ⟨∇ f (x k ), x⟩ + R(x)
for different choices of R. For GD, R(x) = 1 2 ∥xx k ∥ 2 2 while for Normalized GD, R(x) = ι ρD (xx k ), the indicator function for Euclidean ball D = {x : ∥x∥ 2 ≤ 1} scaled by the radius ρ; Clipped GD combines both by taking R(x) = 1 2 ∥xx k ∥ 2 2 + ι ρD (xx k ). This results in the iterates of Clipped GD being generated by the update in Normalized GD if ∥∇ f (x k )∥ 2 is large, but reducing to the update in GD when ∥∇ f (x k )∥ 2 is small enough.
Observation I Our first observation is that both GD and Normalized GD can be generalized to the non-Euclidean case. Define the sharp-operator [Nesterov, 2012, Kelner et al., 2014],
d ♯ ∈ arg max x∈X {⟨d, x⟩ -1 2 ∥x∥ 2 }.
Then, we can write the (possibly non-Euclidean) steepest descent method (SD) as follows
x k+1 = x k -γ[∇ f (x k )] ♯ (SD)
Observe that we recover GD when choosing the Euclidean ℓ 2 norm.
Generalizing Normalized GD to non-Euclidean norms is possible by noticing that the normalization can be written in terms of the linear minimization oracle (lmo)
lmo(d) ∈ arg min x∈D ⟨d, x⟩
where the constraint is a (now assumed to be non-Euclidean) norm-ball D := {x | ∥x∥ ≤ 1}. By choosing the ℓ 2 -norm ball, Normalized GD can be seen as an instance of the so-called unconstrained conditional gradient method (uCG) [Pethick et al., 2025],
x k+1 = x k + γρ lmo(∇ f (x k )).(uCG)
Observation II Our second central observation is that uCG can in general be considered a normalized version of steepest descent. This relationship follows from noticing that the sharp operator and lmo can be defined in terms of each other. Specifically, we have that
lmo(d) = -d ♯ ∥d∥ *
or, equivalently, d ♯ = -∥d∥ * lmo(d).
(1)
In the following section we use this observation to generalize Clipped GD to the non-Euclidean case.
this section cite: ['b28', 'b19', 'b30']

Section: Method
We propose the generalized gradient norm clipping method (GGNC)
x k+1 = x k -γτ k [d k ] ♯ with τ k := min{1, ρ ∥d k ∥ * }.(GGNC)
There is freedom in how to compute the dual norm ∥d k ∥ * due to the following equivalence property for the sharp operator, ∥s∥ 2 * = ∥s ♯ ∥ 2 = ⟨s, s ♯ ⟩. This form is useful, e.g., in the Euclidean case where the sharp-operator is readily available, since then [d k ] ♯ = d k .
For norm choices where the lmo is more naturally available we can equivalently write GGNC as
x k+1 = x k + γη k lmo(d k ) with η k := min{ρ, ∥d k ∥ * }.
We have that ∥d k ∥ * = -⟨d k , lmo(d k )⟩ due to the definition of the dual norm and the optimality of lmo(d k ). So, provided that lmo has been computed, we can obtain ∥d k ∥ * with very little overhead. From this rewriting we also see that ρ can also be interpreted as the radius of the norm-ball constraint over which we compute the lmo.
The GGNC update rule can be seen as the solution to the following optimization problem:
x k+1 ∈ arg min ∥x-x k ∥≤ρ γ ⟨d k , x -x k ⟩ + 1 2 ∥x -x k ∥ 2
The objective is the same quadratic approximation that gives rise to SD, but the iterates are further constrained to a trust-region of radius ρ in the chosen norm, as in uCG.
Algorithm 1 Generalized Gradient Norm Clipping (GGNC) Input: Horizon n, init. x 1 ∈ X, d 0 = 0, momentum α k ∈ (0, 1], stepsize γ ∈ (0, 1) 1: for k = 1, . . . , n do 2:
Sample ξ k ∼ P 3:
d k ← α k ∇ f (x k , ξ k ) + (1 -α k )d k-1 4: v k ← -lmo(d k ) 5: η k ← min{ρ, ⟨d k , v k ⟩} 6: x k+1 ← x k -γη k v k 7: Choose xn uniformly at random from {x 1 , . . . , x n } Return xn Equivalently to step 4-6: x k+1 ← x k -γτ k v k with τ k = min{1, ρ ⟨d k ,v k ⟩ 1/2 } and v k = [d k ] ♯ . Algorithm 2 Stochastic Short Step Conditional Gradient (S 3 CG) Input: Horizon n, init. x 1 ∈ βD = {x ∈ X : ∥x∥ ≤ β}, d 0 = 0, momentum α k ∈ (0, 1], stepsize γ ∈ (0, 1], ball radius β > 0 1: for k = 1, . . . , n do 2:
Sample ξ k ∼ P 3:
d k ← α k ∇ f (x k , ξ k ) + (1 -α k )d k-1 4: v k ← x k -β lmo(d k ) 5: Variant 1: η k ← min{ρ, ⟨d k ,v k ⟩ ∥v k ∥ 2 } 6: Variant 2: η k ← min{ρ, ⟨d k ,v k ⟩ 4β 2 } 7: x k+1 ← x k -γη k v k 8: Choose xn uniformly at random from {x 1 , . . . , x n } Return xn
this section cite: []

Section: Stochastic case
In the deterministic case we can simply take the direction to be d k = ∇ f (x k ). In the stochastic case, one has to proceed with more care, since lmo(d k ) can be biased even when d k is unbiased, due to its potential nonlinearity. With α k ∈ (0, 1], we define the momentum based gradient estimator
d k = (1 -α k )d k-1 + α k ∇ f (x k , ξ k ).
The final algorithm involving the momentum based gradient estimator is presented in Algorithm 1.
this section cite: []

Section: Weight decay & constrained problems
Weight decay is a very popular technique, both as a regularizer to avoid overfitting and for ensuring numerical stability. A precise characterization exists for weight decay when combined with the conditional gradient based schemes like uCG, since the resulting update reduces to the classical conditional gradient method (a.k.a. Frank-Wolfe) designed for solving constrained problems [Chen et al., 2023, D'Angelo et al., 2023, Xie and Li, 2024, Pethick et al., 2025],
x k+1 = (1 -γ k )x k + γ k β lmo(∇ f (x k )),(CG)
where β > 0 is the radius of norm-ball constraint and γ k > 0 is some stepsize to be defined. The simplicial combination ensures that the iterates remain within the constraint set βD and, as a result, ensure that ∥x k ∥ ≤ β for all k.
The CG method is not necessarily a descent method. For the classical open-loop stepsize choice γ k = 2 /k+2, it is possible to step too far in the direction given by the lmo, since the stepsize does not decrease near a critical point. Naively adopting the adaptive stepsize choice from GGNC does not seem appropriate in the constrained case, since ∥d k ∥ * might not necessarily be zero at a solution.
Instead, we will argue that the correct analog of clipping in the constrained setting corresponds to a clipped version of the Frank-Wolfe short step. Like GGNC, this stepsize ensures an analogous descent property.
The short step is almost an immediate consequence of the L-smoothness descent lemma, from which we have
f (x k+1 ) ≤ f (x k ) -γ k ⟨∇ f (x k ), x k -β lmo(∇ f (x k ))⟩ + γ 2 k L 2 ∥x k -β lmo(∇ f (x k )∥ 2 (2) ≤ f (x k ) -γ k ⟨∇ f (x k ), x k -β lmo(∇ f (x k ))⟩ + 2γ 2 k Lβ 2 .(3)
By optimizing this bound with respect to γ k , we arrive at two variants of the short step
γ k (2) = min{1, ⟨∇ f (x k ),x k -β lmo(∇ f (x k ))⟩ L∥x k -β lmo(∇ f (x k )∥ 2 } or γ k (3) = min{1, ⟨∇ f (x k ),x k -β lmo(∇ f (x k ))⟩ 4Lβ 2 }
where the second variant is useful when the norm ∥ • ∥ is expensive to compute. What is particularly noteworthy of these stepsize choices is that they lead to descent, i.e., f (x k+1 ) ≤ f (x k ), by construction. We extend these stepsize choices to the stochastic case with Algorithm 2, where we propose a slightly different parameterization given by
η k = min{ρ, ⟨d k ,x k -β lmo(d k )⟩ ∥x k -β lmo(d k )∥ 2 } or η k = min{ρ, ⟨d k ,x k -β lmo(d k )⟩ 4β 2 }.
A careful reader might have noticed the similarity between the short step in Algorithm 2 and gradient clipping in Algorithm 1. These schemes are indeed equivalent when v k is appropriately modified in Algorithm 2 to be -β lmo(d k ). This connection motivates our parameterization of the updates in Algorithm 2, which are scaled by βγη k , so that the following holds
βγη k = βγ min{ρ, -⟨d k ,β lmo(d k )⟩ ∥β lmo(d k )∥ 2 } = βγ min{ρ, β∥d k ∥ * ∥β lmo(d k )∥ 2 } = βγ min{ρ, β∥d k ∥ * β 2 } = γ min{ρ, ∥d k ∥ * }. The modified Step 7 of Algorithm 2 then becomes x k+1 = x k + γ min{ρ, ∥d k ∥ * } lmo(d k ) which is exactly what is used in GGNC.
this section cite: ['b5', 'b8', 'b42', 'b30']

Section: Norm choices
Algorithm 1 and Algorithm 2 crucially generalize beyond the Euclidean case of Clipped GD. The following section focuses on the unconstrained variant (Algorithm 1) for simplicity, but its constrained counterpart follows in a straightforward way through Algorithm 2.
Sign A simple non-Euclidean example is the ℓ ∞ vector norm for which GGNC reduces to a sign-based update
x k+1 = x k -γη k sign(d k ) (Clipped
Sign) where η k := min{ρ, ∥d k ∥ 1 }. The update is dense in the sense that each coordinate undergoes the same magnitude change. Spectral The matrix analog of the ℓ ∞ norm is the Schatten-∞ matrix norm, a.k.a. the spectral norm, which induces the following update
x k+1 = x k -γη k U k (V k ) ⊤ (Clipped Spectral)
where the reduced singular value decomposition (SVD) is given as
d k = U k diag(σ k )(V k ) ⊤ .
The dual norm can be computed given the lmo as
∥σ k ∥ 1 = ∥d k ∥ S 1 = -⟨d k , lmo(d k )⟩ = -tr(lmo(d k ) ⊤ d k ) = -flatten(lmo(d k )) ⊤ flatten(d k ),
where ∥ • ∥ S 1 is the Schatten-1 norm, a.k.a. the nuclear norm. This scheme is a clipped variant of the stochastic spectral descent method [Carlson et al., 2015b,a].
Product norm The neural networks in deep learning consist of multiple layers and it will therefore be useful to consider what we will call a product norm. Consider x = (W 1 , ..., W D ). A norm of x can be composed using norms on {W l } l∈[D] : [Flynn, 2017] and the ℓ ∞ -norm choice made by the modular norm [Large et al., 2024]. Interestingly, if ∥•∥ X is the max-norm, ∥ • ∥ X = ∥ • ∥ ∞ , then:
∥x∥ = ∥( 1 r 1 ∥W 1 ∥ W 1 , ..., 1 r D ∥W D ∥ W D )∥ X for radius parameters r l > 0. Notable choices of ∥ • ∥ X include the ℓ 1 -norm
(i) The lmos can be computed separately as lmo X (
x) = {r 1 lmo W 1 (W 1 ), ..., r D lmo W D (W D )} (ii)
The dual norm requires summing over all l elements, i.e., ∥x∥ * = D l=1 1 r l ∥W l ∥ W l, * .
As a particular example, consider the LARS optimizer [You et al., 2017], which performs normalized SGD layer-wise. The update rule can be written in terms of the lmo-based scheme uCG with the norm choice ∥x∥ = max l ∥W l ∥ F . Writing the analog sharp-operator based scheme (i.e., SD), we see that it does not correspond to simply removing the normalization as for the ℓ 2 norm. Instead, using the relationship (1), we see that the correct form for the hybrid GGNC method is
W k+1 l = W k l -γ min{ρ, i ∥d k i ∥ F )} d k l ∥d k l ∥ F ∀l ∈ [D]
where
d k = {d k 1 , ..., d k D }
and γ > 0 is the stepsize. Through this duality, we see that while the lmo only requires local information, the dual norm computation (and consequently also the sharp-operator in SD) requires global information.
In Algorithms 3 and 4 of the appendix we specialize Algorithms 1 and 2 to the particular case where ∥ • ∥ X is the max-norm. The resulting algorithms can be seen as clipped variants of the (unconstrained) Scion algorithm [Pethick et al., 2025] so we refer to them as (unconstrained) ClippedScion.
this section cite: ['b11', 'b23', 'b44', 'b30']

Section: Analysis
Why might it be useful to consider a hybrid of SD and uCG? As we will see, the convergence properties of the two methods are complementary.
One can show for SD under L-smoothness that
f (x k+1 ) ≤ f (x k ) -γ(1 -γL /2)∥∇ f (x k )∥ 2 * .
In other words, SD is a descent method in the sense that it decreases the function value f (x k ) at every iteration. The price we pay for this descent is that the stepsize needs to be taken sufficiently small, specifically as γ < 2 /L.
On the other hand, under the same L-smoothness assumption, uCG instead satisfies
f (x k+1 ) ≤ f (x k ) -γρ∥∇ f (x k )∥ * + Lγ 2 ρ 2 2 .
Notice that this is not a descent method, due to the positive contribution of Lγ 2 ρ 2 2 . However, there are no restrictions on the stepsize, and we can in fact show a fast rate of O( 1 /k) for the norm of the gradient with a constant stepsize (as opposed to O( 1 / √ k) of SD), albeit only to a neighborhood whose radius is proportional to γρ, as we formalize in the following result. Proposition 4.1. Suppose f is L-smooth with respect to ∥ • ∥ * and denote f ⋆ = inf x∈X f (x). Then, the iterates {x k } k∈N * of uCG satisfy, for all n ∈ N * ,
min 1≤k≤n ∥∇ f (x k )∥ * ≤ 1 n n k=1 ∥∇ f (x k )∥ * ≤ f (x 1 )-f ⋆ γρn + Lγρ 2 .
Recall that GGNC reduces to uCG when the gradient norm is large, so we can expect in the early phase GGNC will converge rapidly to a neighborhood of size Lγρ 2 . If the gradient norm is small in this region, then GGNC reduces to SD, which converges to an exact critical point even with constant stepsize and which can adapt to the loss landscape through the gradient norm.
We can make this intuition precise by analyzing these algorithms under the following generalization of (L 0 , L 1 )-smoothness to arbitrary norms. Assumption 4.2. The gradient ∇ f is said to be (L 0 ,L 1 )-smooth with L 0 , L 1 ∈ [0, ∞) if, for all x, y ∈ X with ∥x -y∥
≤ 1 L 1 , it holds ∥∇ f (x) -∇ f (y)∥ * ≤ (L 0 + L 1 ∥∇ f (x)∥ * )∥x -y∥.(4)
this section cite: []

Section: Deterministic case
We now proceed to generalizing Koloskova et al. [2023, Thm. 2.1] in the deterministic case. The main argument relies on establishing that GGNC (Algorithm 1) is a descent method even under the generalized (L 0 , L 1 )-smoothness assumption, which enables the scheme to converge even for a fixed, horizon-independent stepsize γ. For the remainder of the paper, we will always denote f ⋆ := inf x∈X f (x) (where it is understood this infimum is taken over βD for constrained problems) and ∆ := f (x 1 )f ⋆ .
Theorem 4.3. Suppose Assumption 4.2 holds and let n ∈ N * . Consider {x k } 1≤k≤n generated by GGNC with d k = ∇ f (x k ), and γ ≤ 1 /(L 0 +ρL 1 ). Then, the following holds
min 1≤k≤n ∥∇ f (x k )∥ * ≤ ∆ γn + 2∆ γρn .
Specifically, with ρ = L 0 L 1 and γ = 1 L 0 , we have
min 1≤k≤n ∥∇ f (x k )∥ * ≤ L 0 ∆ n + 2L 1 ∆ n .
Remark 4.4. Note that the condition ∥x kx k+1 ∥ ≤ 1 /L 1 of Assumption 4.2 required in the proof is always satisfied, since γρ ≤ 1 /L 1 holds for any ρ. We note that descent can also be established for SD with an adaptive stepsize In contrast with GGNC, uCG is not a descent method and requires a diminishing stepsize to converge as suggested by the following theorem. The uCG method trades off the descent property with being agnostic to the Lipschitz constant L 0 .
γ k = 1 /L 0 +L 1 ∥∇ f (x k )∥ * (
Theorem 4.5. Suppose Assumption 4.2 holds and let n ∈ N * . Consider {x k } 1≤k≤n generated by uCG with γρ < 1 /2L 1 . Then, the following holds
min 1≤k≤n ∥∇ f (x k )∥ * ≤ 2∆ γρn + 2L 0 γρ.
Remark 4.6. The assumption that γρ ≤ 1 /2L 1 can be relaxed to γρ < 1 /L 1 while still ensuring convergence, modulo a different constant in the convergence rate.
Let us now turn to the constrained case. The following theorem establishes a convergence rate for Algorithm 2 in the deterministic setting, i.e., with d k = ∇ f (x k ). The convergence rate is established for a quantity called the Wolfe-gap, max u∈βD ⟨∇ f (x), x -u⟩, which, when equal to 0, certifies that x is a critical point for the constrained problem. It is the equivalent of the dual norm of the gradient but for constrained problems, since the gradient might not vanish at a critical point in the constrained setting. The theorem also includes an assumption that f is L-smooth rather than (L 0 , L 1 )-smooth. Because the iterates of Algorithm 2 are guaranteed to never leave the compact set βD, L-smoothness is implied by (L 0 , L 1 )-smoothness here. Theorem 4.7. Suppose f is L-smooth and let n ∈ N * . Consider {x k } 1≤k≤n generated by Algorithm 2 with d k = ∇ f (x k ), γ ≤ 1 L , and ρ ≤ L so that γρ ≤ 1. Then, for all u ∈ βD, the following holds
min 1≤k≤n ⟨∇ f (x k ), x k -u⟩ ≤ 2β ∆ γn + 2∆ γρn .
this section cite: []

Section: Stochastic case
We consider the following standard assumption about the bias and variance of the stochastic oracle. Assumption 4.8. For the stochastic gradient estimator ∇ f (•, ξ) : X → R d the following holds.
(i) Unbiased:
E ξ ∇ f (x, ξ) = ∇ f (x) ∀x ∈ X. (ii) Bounded variance: E ξ ∥∇ f (x, ξ) -∇ f (x)∥ 2 2 ≤ σ 2 ∀x ∈ X, σ ≥ 0.
In order to establish convergence in what follows, an important quantity to introduce is the error produced by the stochastic estimator d k , which we denote by λ k := d k -∇ f (x k ).
We establish the following order optimal convergence guarantee for GGNC under (L 0 , L 1 )-smoothness using a momentum-based estimator. These convergence results for clipping with momentum appear to be new, even in the Euclidean case. Theorem 4.9. Suppose Assumptions 4.2 and 4.8 hold and let n ∈ N * . Consider the iterates {x k } 1≤k≤n generated by Algorithm 1 with a constant stepsize γ ≤ 1 /L 0 and γρ ≤ 1 /2L 1 . Then,
E[∥∇ f ( xn )∥ * ] ≤ 4 √ ∆ √ γn + 8∆ γρn + 4 √ ϵ n + 8ϵ n ρ where ∆ := f (x 1 ) -f ⋆ and ϵ n := 1 n n k=1 O( E[∥λ k ∥ 2 2 ] + E[∥λ k ∥ 2 2 ]).
Furthermore, assuming f is L-smoothfoot_0 and taking
α = 1 / √ n, γ = 1 / √ nL 0 and ρ = L 0/2n 1/4 L 1 such that γρ = 1 /2n 3/4 L 1 we have that E[∥∇ f ( xn )∥ * ] ≤ O 1 n 1/4
. Remark 4.10. For ease of exposition, the guarantee is presented with horizon-dependent parameter choices, but the result can be extended to an any time guarantee in a straightforward manner by choosing the parameters as a function of k instead of n and modifying the proofs accordingly.
In the constrained case, we have the following convergence guarantee for SCG with a clipped short step (Algorithm 2) using a momentum-based estimator. To the best of our knowledge, this is the first convergence proof using the short step in the stochastic setting. Theorem 4.11. Suppose Assumptions 4.2 and 4.8 hold and let n ∈ N * . Consider the iterates {x k } 1≤k≤n generated by Algorithm 2 (Variant 1) with a constant stepsize γ ≤ 1 /L √ n and ρ ≤ 1 /n 1/4 . Then, for all u ∈ βD,
E[⟨∇ f ( xn ), xn -u⟩] ≤ 4 √ ∆ √ γn + 8∆ γρn + 4 √ ϵ n + 8ϵ n ρ where ∆ := f (x 1 ) -f ⋆ and ϵ n := 1 n n k=1 O( E∥λ k ∥ 2 2 + E∥λ k ∥ 2 2 ). Furthermore, taking α = 1 / √ n, γ = 1 /(L √ n) and ρ = 1 /n 1/4 such that γρ = 1 /(Ln 3/4 ) we have that E[⟨∇ f ( xn ), xn -u⟩] ≤ O 1 n 1/4 .
We additionally provide an identical guarantee for Algorithm 2 (Variant 2) in the appendix.
5 Related work (L 0 , L 1 )-smoothness An (L 0 , L 1 )-smoothness condition was introduced based on the Hessian in [Zhang et al., 2019] and later generalized to the first-order notion that we extend to the non-Euclidean case [Zhang et al., 2020]. (L 0 ,L 1 )-smoothness was used to analyze signSGD under heavy-tailed noise assumptions in Kornilov et al. [2025]. A coordinate-wise (L 0 ,L 1 )-smoothness condition has also been considered for analyzing a generalized version of signSGD [Crawshaw et al., 2022].
In the Euclidean case, a descent property under (L 0 , L 1 )-smoothness was shown for both gradient clipping [Zhang et al., 2020, Koloskova et al., 2023] and gradient descent with an appropriate adaptive stepsize as studied in the two concurrent works Gorbunov et al. [2024] andVankov et al. [2024].
Parameter-agnostic In the deterministic case, gradient descent with backtracking line-search was shown to converge under (L 0 , L 1 )-smoothness without knowledge of the Lipschitz constants [Hübler et al., 2024]. For (star)-convex problems, an interesting connection was established between gradient norm clipping and the Polyak stepsize in Takezawa et al. [2024] and further analyzed in Gorbunov et al. [2024] andVankov et al. [2024]. The adaptive stepsize removes the need for knowing both L 0 and L 1 . Unfortunately, the Polyak stepsize is deeply tied to the Euclidean and (star)-convex structure and thus does not seem to be directly extendable to our more general setting.
In the stochastic case, the current best known parameter-agnostic method introduces an undesirable exponential dependency on L 1 in the complexity [Hübler et al., 2024]. However, knowledge of L 0 can be removed without such issues through either an AdaGrad type stepsize [Wang et al., 2023, Faw et al., 2023] or normalized gradient descent with momentum [Cutkosky and Mehta, 2020] as shown in Hübler et al. [2024]. This mirrors results from the online learning community where both AdaGrad and gradient normalization are known to adapt to Hölder smoothness [Orabona, 2023].
this section cite: ['b46', 'b45', 'b21', 'b6', 'b45', 'b20', 'b13', 'b39', 'b14', 'b35', 'b13', 'b39', 'b14', 'b41', 'b10', 'b7', 'b14', 'b29']

Section: Short step
In contrast with gradient descent, the Frank-Wolfe algorithm [Frank et al., 1956] does not ensure descent with an open-loop stepsize even in the deterministic setting. Descent can be ensured by an adaptive stepsize known as the short step, originally introduced by Frank & Wolfe [Frank et al., 1956] and extended by Rubinov & Dem'yanov [Dem'yanov and Rubinov, 1968]. See Pokutta [2024] for an expository treatment.
Spectral norm methods Clipped Spectral can be viewed as a hybrid method between the stochastic spectral descent [Carlson et al., 2015b] and the Muon optimizer [Jordan et al., 2024b], with some crucial differences.
Muon builds the gradient estimator d k differently. Specifically they take
d k = ∇ f (x k , ξ k ) + βd k-1 if Nesterov momentum is disabled. This is equivalent to our choice d k = α∇ f (x k , ξ k ) + (1 -α)d k-1
for LMO-based schemes, since the LMO is scale-invariant (i.e., lmo(a • s) = lmo(s) for a > 0) [Pethick et al., 2025]. However, for SD this equivalence no longer holds (in fact we have [a • s] ♯ = a[s] ♯ for a ∈ R). The appropriate choice of d k , which generalizes to SD and GGNC, turns out to be the convex combination.
Stochastic spectral descent [Carlson et al., 2015b] does not construct a gradient estimator and instead takes d k = ∇ f (x k , ξ k ). This restricts their convergence result to the case of (mild) relative noise.
In this sense, Clipped Spectral could just as well be called Clipped Muon (not to be confused with the unrelated MuonClip [Team et al., 2025]) but we prefer Clipped Spectral as the algorithm itself is not tied to momentum nor to Newton-Schulz, as the name Muon fundamentally is. Tuddenham et al. [2022] also studied an optimization algorithm focused on orthogonalization, however they orthogonalize before doing the momentum step. Pethick et al. [2025] analyzed a more general algorithm called Averaged LMO Directional Descent which admits as a special case the algorithm studied in Tuddenham et al. [2022]; their empirical and theoretical findings found this algorithm to be worse than orthogonalization after the momentum step, e.g., the Scion family of algorithms [Pethick et al., 2025].
We note that many works have recently analyzed the convergence behavior of algorithms using spectral LMOs like Muon and Scion, starting first with Pethick et al. [2025], Li and Hong [2025] and then Kovalev [2025], Sfyraki and Wang [2025], but always under L-smoothness assumptions, in contrast to this work.
Modular norm [Large et al., 2024] introduced a norm choice for neural networks and established a smoothness condition for a given neural network provided the parameter remains bounded. The dual norm computation needed in GGNC is particularly easy to implement in the accompanying Modula software package since ∥d∥ * :=flatten(lmo(d)) ⊤ flatten(d), which in Modula code reads as dual_norm=-sum(model.dualize(d)*d).
Weight decay Weight decay [Pratt, 1992] is a crucial component in deep learning and has become standard in training modern neural networks through its integration with Adam [Loshchilov and Hutter, 2017]. When combined with LMO based updates such as sign descent and the normalized gradient descent the resulting methods can be seen as instantiations of the conditional gradient method for constrained optimization problems [Chen et al., 2023, D'Angelo et al., 2023, Xie and Li, 2024, Pethick et al., 2025]. Our adaptive stepsize in Algorithm 2 effectively scales the weight decay as well as the update. This is similar to scheduled weight decay [Xie et al., 2023] which uses the adaptive stepsize in Adam to also scale the weight decay parameter.
this section cite: ['b12', 'b12', 'b9', 'b31', 'b30', 'b36', 'b38', 'b30', 'b38', 'b30', 'b30', 'b24', 'b22', 'b33', 'b23', 'b32', 'b25', 'b5', 'b8', 'b42', 'b30', 'b43']

Section: Experiments
For the norm choice of Scion and ClippedScion we use the (Sign → Spectral → Sign) and (Spectral → Spectral → Sign) configurations for language modeling and image classication respectively (see Pethick et al. [2025, Tbl. 2-4] for the associated scaling factors). To compute the spectral lmo we use the efficient implementation provided in Jordan et al. [2024b] of the Newton-Schultz iteration proposed in Bernstein and Newhouse [2024]. There have been recent efforts to move beyond the "N" (Newton-Schulz) in Muon, the most popular algorithm computing the spectral LMO, through alternative subroutines; our algorithm is compatible with these alternatives, like the optimized PolarExpress routine [Amsel et al., 2025] or power iterations [Ahn et al., 2025, Vogels et al., 2019], although we do not explore them here.
this section cite: ['b2', 'b1', 'b0', 'b40']

Section: Image classification
We test on a convolutional neural network (CNN) on the CIFAR10 dataset. Hyperparameters can be found in Table 2 in Appendix C. We consider both a fixed stepsize setting and stepsize scheduling using linear rampdown to investigate if the theoretical results are predictive of practice. We report the experimental results in Figure 1 where mean and standard deviation are computed over 5 independent runs. 0 1000 2000 3000 4000 5000 Iteration k 3.0 3.1 3.2 3.3 3.4 3.5 3.6 Validation Loss Adam Unconstrained Scion Unconstrained ClippedScion 10% speedup 1000 2000 3000 4000 5000 Iteration k 3000 4000 5000 6000 7000 8000 Gradient norm d k Unconstrained ClippedScion Clipping threshold Figure 2: For fixed stepsize comparison clipping improves over Scion by more than a 10% speedup on NanoGPT (1B). We observe similar gains on the smaller 124M parameter model size (cf. Appendix C).
We find that clipping can substantially improve the test accuracy in the fixed stepsize setting, when the gradient norm (i.e. ∥d k ∥ * = ⟨d k , v k ⟩) is decreasing. This separation is in agreement with the theoretical separation between Theorem 4.3 and Theorem 4.5 on fixed stepsizes. In the constrained case (Algorithm 2) we surprisingly find that ⟨d k , v k ⟩ is increasing (cf. Figure 6 in Appendix C) which requires further investigation. With stepsize scheduling we observe that clipping (i.e., Unconstrained ClippedScion) and normalization (i.e., Unconstrained Scion) achieve similar performance, which aligns with the matching theoretical rates of GGNC (Theorem 4.9) and uSCG (Pethick et al. [2025, Thm. 5.4]) in the stochastic case when stepsizes are taken decreasing.
We also evaluate the unconstrained case (Algorithm 1) using Vision Transformers (ViT) on the ImageNet dataset. We train a DeiT-base model using the DeiT codebase [Touvron et al., 2021] with replacing LayerNorm by RMS norm following [Pethick et al., 2025]. Table 3 in Appendix C contains the hyperparameter details. As shown in Figure 7 (Appendix C), Unconstrained ClippedScion achieves an 11% speedup over Unconstrained Scion, even though its gradient norm (∥d k ∥ * ) is increasing. This observation requires further exploration.
NanoGPT We additionally test on NanoGPT Karpathy [2023] in Figure 2 with modernizations following [Jordan et al., 2024a]: rotary embeddings are used instead of positional embeddings, RMS norm is used instead of LayerNorm, and the ReLU 2 [So et al., 2021] instead of GELU activation function. All methods are trained for 5100 iterations with a batchsize of 512 and context length of 1024 on the FineWeb dataset (see Table 4 Appendix C for further details). The empirical observations matches those for CIFAR10 experiments.
this section cite: ['b37', 'b30', 'b18', 'b34']

Section: Conclusion
We have shown that clipping can be extended to non-Euclidean settings and even constrained problems by establishing a precise connection to the Frank-Wolfe short step. A descent property was established under a generalized notion of (L 0 , L 1 )-smoothness, which opens up a range of interesting directions:
The descent property both in the unconstrained and constrained case enables integration with adaptive stepsize choices such as AdaGrad and backtracking line-search.
The non-Euclidean notion of (L 0 , L 1 )-smoothness we introduce might be a suitable condition to study for neural networks. Large et al. [2024] showed that neural networks are smooth in the modular norm provided that the parameters are constrained. However, in practice, violating the constraints seem to be unproblematic for optimization, which suggests that a looser smoothness assumption might hold such as Assumption 4.2.
this section cite: ['b23']

Section: References
Ref_id:b0 Title: Dion: Distributed orthonormalized updates Year: (2025)
Ref_id:b1 Title: The polar express: Optimal matrix sign methods and their application to the muon algorithm Year: (2020)
Ref_id:b2 Title: Old optimizer, new norm: An anthology Year: (2024)
Ref_id:b3 Title: Stochastic spectral descent for restricted boltzmann machines Year: (2015)
Ref_id:b4 Title: Stochastic spectral descent for discrete graphical models Year: (2015)
Ref_id:b5 Title: Lion secretly solves constrained optimization: As lyapunov predicts Year: (2023)
Ref_id:b6 Title: Robustness to unbounded smoothness of generalized signsgd Year: (2022)
Ref_id:b7 Title: Momentum improves normalized sgd Year: (2020)
Ref_id:b8 Title: Aditya Varre, and Nicolas Flammarion. Why do we need weight decay in modern deep learning? arXiv preprint Year: (2023)
Ref_id:b9 Title: Minimization of functionals in normed spaces Year: (1968)
Ref_id:b10 Title: Beyond uniform smoothness: A stopped analysis of adaptive sgd Year: (2023)
Ref_id:b11 Title: The duality structure gradient descent algorithm: analysis and applications to neural networks Year: (2017)
Ref_id:b12 Title: An algorithm for quadratic programming Year: (1956)
Ref_id:b13 Title: Methods for convex (l_0, l_1)-smooth optimization: Clipping, acceleration, and adaptivity Year: (2024)
Ref_id:b14 Title: Parameter-agnostic optimization under relaxed smoothness Year: (2024)
Ref_id:b15 Title: Cifar-10 airbench Year: (2024)
Ref_id:b16 Title: @fernbear.bsky.social, Boza Vlado, You Jiacheng, Franz Cesista, Braden Koszarsky, and @Grad62304977. modded-nanogpt: Speedrunning the nanogpt baseline Year: (2024)
Ref_id:b17 Title: Muon: An optimizer for hidden layers in neural networks Year: (2024)
Ref_id:b18 Title:  Year: (2023)
Ref_id:b19 Title: An almost-linear-time algorithm for approximate max flow in undirected graphs, and its multicommodity generalizations Year: (2014)
Ref_id:b20 Title: Revisiting gradient clipping: Stochastic bias and tight convergence guarantees Year: (2023)
Ref_id:b21 Title: Sign operator for coping with heavy-tailed noise in non-convex optimization: High probability bounds under (l_0, l_1)-smoothness Year: (2025)
Ref_id:b22 Title: Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization Year: (2025)
Ref_id:b23 Title: Scalable optimization in the modular norm Year: (2024)
Ref_id:b24 Title: A note on the convergence of muon and further Year: (2025)
Ref_id:b25 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b26 Title: Statistical language models based on neural networks Year: (2012)
Ref_id:b27 Title: Stochastic conditional gradient methods: From convex minimization to submodular maximization Year: (2020)
Ref_id:b28 Title: Efficiency of coordinate descent methods on huge-scale optimization problems Year: (2012)
Ref_id:b29 Title: Normalized gradients for all Year: (2023)
Ref_id:b30 Title: Training deep learning models with norm-constrained lmos Year: (2025)
Ref_id:b31 Title: The frank-wolfe algorithm: a short introduction Year: (2024)
Ref_id:b32 Title: Non-literal transfer among neural network learners Year: (1992)
Ref_id:b33 Title: Lions and muons: Optimization via stochastic frank-wolfe Year: (2025)
Ref_id:b34 Title: Searching for efficient transformers for language modeling Year: (2021)
Ref_id:b35 Title: Polyak meets parameterfree clipped gradient descent Year: (2024)
Ref_id:b36 Title: Kimi k2: Open agentic intelligence Year: (2025)
Ref_id:b37 Title: Training data-efficient image transformers & distillation through attention Year: (2021)
Ref_id:b38 Title: Orthogonalising gradients to speed up neural network optimisation Year: (2022)
Ref_id:b39 Title: Optimizing (l_0, l_1)-smooth functions by gradient methods Year: (2024)
Ref_id:b40 Title: Powersgd: Practical low-rank gradient compression for distributed optimization Year: (2019)
Ref_id:b41 Title: Convergence of adagrad for non-convex objectives: Simple proofs and relaxed assumptions Year: (2023)
Ref_id:b42 Title: Implicit bias of AdamW: ℓ ∞ norm constrained optimization Year: (2024)
Ref_id:b43 Title: On the overlooked pitfalls of weight decay and how to mitigate them: A gradient-norm perspective Year: (2023)
Ref_id:b44 Title: Large batch training of convolutional networks Year: (2017)
Ref_id:b45 Title: Improved analysis of clipping algorithms for non-convex optimization Year: (2020)
Ref_id:b46 Title: Why gradient clipping accelerates training: A theoretical justification for adaptivity Year: (2019)
