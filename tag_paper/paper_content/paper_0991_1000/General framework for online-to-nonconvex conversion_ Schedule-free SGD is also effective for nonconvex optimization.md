Title: General framework for online-to-nonconvex conversion: Schedule-free SGD is also effective for nonconvex optimization
Abstract: This work investigates the effectiveness of schedule-free methods, developed by A. Defazio et al. (NeurIPS 2024), in nonconvex optimization settings, inspired by their remarkable empirical success in training neural networks. Specifically, we show that schedule-free SGD achieves optimal iteration complexity for nonsmooth, nonconvex optimization problems. Our proof begins with the development of a general framework for onlineto-nonconvex conversion, which converts a given online learning algorithm into a nonconvex optimization algorithm. Our general framework not only recovers existing conversions but also leads to two novel conversion schemes. Notably, one of these new conversions corresponds directly to schedule-free SGD, allowing us to establish its optimality. Additionally, our analysis provides valuable insights into the parameter choice for schedule-free SGD, addressing a theoretical gap that the convex theory cannot explain.

Section: Introduction
Training large-scale neural network models, such as large language models, requires a well-designed optimization strategy to ensure stable and fast convergence. For instance, training typically requires a carefully designed optimizer, such as the Adam optimizer (Kingma and Ba, 2014), along with meticulously tuned learning rate scheduling.
Recently, Defazio et al. (2024) introduced the schedule-free method, which achieves impressive training performance without any learning rate scheduling. The schedule-free method is an add-on scheme that can be applied to any chosen base optimizer, converting it into a schedule-free variant. While this method has shown strong empirical performance in training large neural network models, its theoretical analysis has been limited to the convex setting (Defazio et al., 2024). Our aim is to extend the theoretical understanding of schedule-free methods to nonconvex optimization.
As an initial step, this work focuses on the version where the base optimizer is chosen as SGD, referred to as schedulefree SGD. For a given learning rate γ > 0 and interpolation weights c t , κ t ∈ [0, 1], the updates of schedule-free SGD maintain x t , y t , and z t , as follows:
        
x t = (1 -c t )x t-1 + c t z t , y t = (1 -κ t )z t + κ t x t , g t = a stochastic gradient at y t , z t+1 = z t -γg t .
(SF-SGD) Here, z t corresponds to the base SGD trajectory, x t is a (weighted) average of z t , and y t is an interpolation between x t and z t where the stochastic gradient is computed.
this section cite: []

Section: Our main result and approach
To understand the effectiveness of schedule-free SGD for training neural networks, we analyze the method in the nonsmooth and nonconvex setting (Zhang et al., 2020;Davis et al., 2022b;Tian et al., 2022). Specifically, we adopt the (λ, ϵ)-stationarity criterion from (Zhang and Cutkosky, 2024;Ahn and Cutkosky, 2024) (Definition 1), which seeks approximate Goldstein stationary points (Goldstein, 1977). This criterion emerges as a practical framework for analyzing optimization algorithms, particularly considering recent findings that show practical optimizers like SGD with momentum and Adam are theoretically optimal (Zhang and Cutkosky, 2024;Ahn and Cutkosky, 2024).
Our main results demonstrate that schedule-free SGD is optimal not only for convex optimization, as established by Defazio et al. (2024), but also for the most challenging case of nonsmooth and nonconvex optimization. Our main results can be summarized informally as follows:
Theorem 1.1 (Informal; see Section 5). Schedule-free SGD (SF-SGD), with an appropriate choice of parameters γ, c t , κ t , achieves optimal rates for nonsmooth and nonconvex F . Cutkosky et al. (2023). In essence, this framework takes an online learner as input and outputs an optimization algorithm. As its name suggests, this framework translates online learning guarantees into nonconvex guarantees, analogous to the well-known onlineto-batch conversion for convex settings (Cesa-Bianchi et al., 2004). As suggested by our title, we first introduce a general framework for online-to-nonconvex conversion. This framework not only encompasses the previous conversion (Cutkosky et al., 2023;Zhang and Cutkosky, 2024;Ahn and Cutkosky, 2024) as a special case but also enables new conversion schemes, as we present in Section 4.2 and Section 5.
this section cite: ['b0', 'b0', 'b0', 'b9', 'b5', 'b9', 'b0']

Section: Our proof technique leverages the online-to-nonconvex conversion framework pioneered by
With the new conversion schemes, our main observation is that one of these novel conversions, outlined in Algorithm 5, directly corresponds to schedule-free SGD. Specifically, by choosing a basic online mirror descent as the online learner in Algorithm 5, we naturally recover the schedule-free SGD algorithm. From this, our main result, Theorem 1.1, follows.
Notably, our approach provides fresh practical insight into the parameter choice of schedule-free methods, which cannot be explained by prior convex analysis in Defazio et al. (2024). Our results suggest that setting κ t close to 1 is advantageous for nonconvex optimization. This finding clarifies the curious importance of choosing κ t near 1 in (Defazio et al., 2024) for empirical performance -a phenomenon not previously explained by convex theory.
this section cite: []

Section: Related work
Our work builds on a line of research focused on convergence guarantees for nonsmooth, nonconvex optimization. Intuitively, our convergence criterion aims to find approximate Goldstein stationary points (Goldstein, 1977). The formal study of iteration complexity for finding approximate Goldstein stationary points was initiated by Zhang et al. (2020) and has since garnered significant interest (Davis et al., 2022b;Tian and So, 2022;Lin et al., 2022;Chen et al., 2023;Jordan et al., 2023;Cutkosky et al., 2023;Kornowski and Shamir, 2024). Alternative convergence notions are also widely used, including approaches based on the Moreau envelope or by imposing weak convexity conditions (Davis et al., 2018;2022a). In particular, the criterion we consider in this work, formalized in Definition 1, follows the relaxed version proposed by Zhang and Cutkosky (2024), which slightly modifies the original criterion from Zhang et al. (2020). Additional discussion of related work is provided in Section 7.
this section cite: ['b6', 'b9', 'b0']

Section: Preliminaries
We first introduce the key assumptions and the notion of convergence. Throughout this paper, unless specified otherwise, ∥•∥ denotes the L 2 norm.
this section cite: []

Section: Setting
Following Cutkosky et al. (2023), we consider optimizing a loss function F that satisfies the following conditions.
Assumption 1. Let F : R d → R be a differentiable function with the following properties:
• Let ∆ F := F (x 0 ) -inf x F (x).
• For any two points x and w, F (x) -F (w) = 1 0 ⟨∇F (w + t(x -w)), x -w⟩ dt.
• F is G-Lipshitz, i.e., for any point x, ∥∇F (x)∥ ≤ G.
Here, the second condition, called well-behavedess in (Cutkosky et al., 2023, Definition 1), is a mild regularity condition. For any locally Lipschitz function F , applying an arbitrarily small perturbation is sufficient to ensure this condition (Cutkosky et al., 2023, Proposition 2).
We optimize the loss function F by via a stochastic gradient oracle, which is formalized as follows:
Assumption 2 (Stochastic gradient oracle). We assume access to a stochastic gradient oracle at any point x. More formally, for any given point x, each call to the oracle independently returns a stochastic gradient g that satisfies the following properties:
E[g] = ∇F (x), and E ∥g -∇F (x)∥ 2 ≤ σ 2 .
We denote the stochastic gradient oracle as STOGRAD.
When the oracle returns the stochastic gradient g at point x, we write this as g ← STOGRAD(x).
this section cite: ['b9', 'b9', 'b9']

Section: Approximate Goldstein stationary point
For the notion of optimality, we follow Zhang and Cutkosky (2024); Ahn and Cutkosky (2024) and consider the following notion of stationarity for nonconvex and nonsmooth functions. This notion can be regarded as an approximate version of the notion of a Goldstein stationarity point.
Definition 1 ((λ, ε)-stationary point). Suppose F : R d → R is differentiable. We say x is a (λ, ε)-stationary point of F if ∥∇F (x)∥ [λ] ≤ ε, where
∥∇F (x)∥ [λ] := inf ∥E[∇F (w)]∥ + λE ∥w -x∥ 2 .
Here the infimum is taken over the distribution p ∈ P(R d ) such that E w∼p [w] = x.
Our algorithms will identify (λ, ε)-stationary points using O λ 1/2 ϵ -7/2 calls to a stochastic gradient oracle, which is the optimal rate (Zhang and Cutkosky, 2024).
To further motivate this definition, we remark that (λ, ε)stationary points retain the desirable properties of Goldstein stationary points. Specifically, the following result (Zhang and Cutkosky, 2024, Lemma 2.3) demonstrates that, akin to Goldstein stationary points, (λ, ε)-stationary points can be reduced to first-order stationary points with appropriate choices of λ when the objective function is smooth or second-order smooth.
Proposition 2.1. If F is L-smooth, then an (L 2 ε -1 , ε)- stationary point x of F satisfies ∥∇F (x)∥ ≤ 2ε. Moreover, if F is H-second-order-smooth, then an (H/2, ε)-stationary point x of F satisfies ∥∇F (x)∥ ≤ 2ε.
Note that another popular notion of approximate Goldstein stationarity is called (δ, ε)-stationarity due to Zhang et al. (2020). However, the algorithms that achieve optimal complexity under that notion often require clipping on the momentum term (Cutkosky et al., 2023), which introduces deviations from the practical optimization algorithms. Hence, in this work, we adopt the notion of (λ, ε)-stationarity, for which it has been demonstrated that optimal algorithms does not require clipping operations (Zhang and Cutkosky, 2024). We also remark that algorithms that identify (λ, ε)-stationary points can also identify (δ, ε)-stationary points when F is Lipschitz, as demonstrated in (Zhang and Cutkosky, 2024, Lemma 2.4).
As mentioned above, the stochastic gradient oracle complexity of finding a (λ, ε) stationary point is Θ(λ 1/2 ε -7/2 ) (Zhang and Cutkosky, 2024). By Proposition 2.1, any algorithms achieving this rate (such as the ones we will present), can also find a point x with ∥∇F (x)∥ ≤ ε in O(ϵ -4 ) oracle calls when F is smooth and O(ϵ -3.5 ) calls when F is second-order smooth. These are the optimal rates for their respective function classes (Arjevani et al., 2023;2020).
this section cite: ['b0', 'b0', 'b9', 'b3', 'b2']

Section: Online learning
In this section, we provide a brief background on online learning, which plays a key role in our development. Online learning is modeled as a sequential decision-making process over T rounds. In each round t, the online learner chooses a point δ t ∈ R d , and then a loss function ℓ t : R d → R is revealed. The learner then incurs a loss ℓ t (δ t ). The choice of δ t is based on the previous loss sequence ℓ 1:t-1 , and after selecting δ t , the learner observes the next loss ℓ t .
The performance of the online learner is measured using the regret with respect to a comparator u, formally defined as: Regret T (u) := T t=1 (ℓ t (δ t ) -ℓ t (u)). However, recent works (Cutkosky et al., 2023;Ahn et al., 2024) demonstrate that when designing nonconvex optimization algorithms, base online learners must be adaptive to time-varying comparators, also known as the dynamic regret setting. Inspired by Ahn et al. (2024); Zhang and Cutkosky (2024); Ahn et al. (2024), we design such dynamic online learners by considering the following "discounted" version of regret.
Definition 2 (Discounted regret). Consider the loss sequence ℓ 1:T . For any T ≥ 1, we define the discounted regret of an online learner with respect to a comparator u as:
Regret [β] T (u) := T t=1 β T -t (ℓ t (δ t ) -ℓ t (u)) .
For instance, Ahn et al. (2024) demonstrate that online learners with low discounted regret can achieve low dynamic regret through what they call the discounted-to-dynamic conversion. Additionally, discounted regret has been shown to be a more effective metric for designing adaptive online learners in dynamic environments across various contexts, including conformal prediction (Zhang et al., 2024) and online linear regression (Jacobsen and Cutkosky, 2024).
this section cite: ['b9', 'b1']

Section: General framework for online-to-nonconvex conversion
In this section, we introduce a general scheme (Algorithm 1) for converting online learning guarantees into nonconvex optimization guarantees. As a preview, we will demonstrate that our Algorithm 1 not only recovers existing approaches as special cases, but also enables the design of novel conversion methods. Let us begin with the pseudocode.
Algorithm 1 General scheme for online-to-nonconvex conversion 1: Input: Initial iterates x 0 = w 0 , an online learner A, and T ∈ N, regularization strength µ ≥ 0. 2: for t = 1, 2, . . . , T do 3: Receive δ t from A.
4: Choose x t arbitrarily. // The design choice 5: Update w t = x t + δ t . 6: Update y t = x t + s t δ t where 0 ≤ s t ≤ 1 is drawn uniformly i.i.d. 7: Compute g t ← STOGRAD(y t ). 8: Send loss ℓ t (•) = ⟨g t , •⟩ + µ 2 ∥•∥ 2 to A.
this section cite: []

Section: 9: end for
Overall, Algorithm 1 generates three sequences of iterates:
x t , y t , and w t . At each iteration, the stochastic gradients are fed into the online learner, which outputs δ t .
The main design feature of Algorithm 1 that gives it flexibility is the ability to choose x t arbitrarily at each iteration. However, there are technical properties we want x t to sat-isfy in order to achieve better nonconvex guarantees, which are detailed in Section 3.3.
The online learner's output δ t is then used alongside x t to compute w t according to the update rule w t = x t + δ t . Additionally, the iterates y t are sampled uniformly from the line segment connecting x t and w t , and the stochastic gradients are computed at these y t iterates.
this section cite: []

Section: Output of the general scheme
To establish nonconvex optimization guarantees, we utilize the exponential moving average (EMA) of the y t iterates, similar to ( Zhang and Cutkosky, 2024; Ahn and Cutkosky,  2024). We begin by formally defining these EMA iterates.
Definition 3. Given a discount factor β ∈ (0, 1) and a sequence of iterates {y s } t s=1 , the β-EMA of the sequence up to time t, denoted as y t , is defined as
y t = 1 -β 1 -β t t s=1 β t-s y s .
In particular, as we will see in Lemma 3.1, the final output will be the random EMA iterate y τ , where τ is a carefully selected random index, defined as follows.
Definition 4 (Random index distribution). Let τ be a random index distributed over {1, 2, . . . , T } with the following distribution:
Pr(τ = t) = 1-β t T , for t = 1, . . . , T -1, 1 1-β • 1-β T T , for t = T.
We note that previous works select the output uniformly at random from the sequence {y t } T t=1 (Zhang and Cutkosky, 2024; Ahn and Cutkosky, 2024). Our carefully designed random EMA iterate from Definition 4 leads to an improved conversion result, offering stronger nonconvex optimization guarantees, as we will discuss in Section 4.
Notice that as β approaches 1, the random index τ assigns a significantly higher probability (by a multiplicative factor of 1 1-β ) to the final index T . This aligns with common practice, where the final iterate-rather than an average of previous iterates-is often used as the output. Indeed, we will select β very close to 1 for our nonconvex guarantees (we will choose β = 1 -O(ε 2 )).
this section cite: ['b0']

Section: Conversion guarantees
Given the description of the algorithm and its output, we now present the online-to-nonconvex conversion guarantees. We begin by introducing a key definition that quantifies the stability of the x t sequence. Definition 5 (x-iterate stability). Consider the iterates generated by Algorithm 1. The iterate stability factor, denoted by C x ≥ 0, is the smallest nonnegative constant such that:
E T t=1 ∥x t -x t-1 ∥ 2 ≤ C x • E T t=1 ∥δ t ∥ 2 .
With the concept of the iterate stability factor, we can now state the conversion result. Lemma 3.1 (Generic online-to-nonconvex conversion). Consider the iterates generated according to Algorithm 1. For β ∈ (0, 1) and D > 0, define the comparators for online learner as follows:
∀t ∈ [T ], u t := -D t s=1 β t-s ∇F (y s ) t s=1 β t-s ∇F (y s ) .
Then, as long as the regularization strength satisfies µ ≥ 8λD 1 + C x (1 -β) -2 , the following holds:
E τ ∥∇F (y τ )∥ [λ] ≤ β DT E Regret [β] T (u T ) + 1 -β DT T t=1 E Regret [β] t (u t ) + 1 DT E T t=1 (F (x t ) -F (w t )) + µD 2 + σ T √ 1 -β + σ 1 -β.
Proof. See Appendix A.
Lemma 3.1 provides an upper bound on the main quantity of interest, E τ ∥∇F (y τ )∥ [λ] . One of the main terms in this upper bound is the sum of the discounted regret terms, which indicates that improved performance by the online learner leads to better guarantees in nonconvex optimization. Thus, Lemma 3.1 effectively translates the online learning guarantee into a nonconvex optimization guarantee, as suggested by its name, online-to-nonconvex conversion.
However, in its current form, the presence of several additional terms in the upper bound makes the result less interpretable. Before demonstrating the strength of this general conversion framework, we first reformulate Lemma 3.1 into a more interpretable and user-friendly form.
this section cite: []

Section: User-friendly nonconvex optimization guarantees
In this section, we apply a concrete discounted regret bound to Lemma 3.1 to derive a more user-friendly nonconvex guarantee, which will be used throughout the remainder of the paper. In particular, a discounted version of composite objective online mirror descent (OMD) (Beck and Teboulle, 2003;Duchi et al., 2010;Zhang and Cutkosky, 2024) achieves the following discounted regret bound.
Lemma 3.2. Let β ∈ (0, 1), µ ≥ 0, η > 0, and a sequence of vectors {g t } T t=1 . Suppose that E ∥g t ∥ 2 ≤ G 2 +σ 2 for all t ∈ [T ]. Consider an online learner initialized with δ 1 = 0 and updated as follows:
δ t+1 = β 1 + ηµ (δ t -ηg t ) . (β-OMD)
Then, this online learner with η = 2 G+σ ∥u∥ √ 1 -β achieves the following discounted regret bound:
E Regret [β] T (u) ≤ 2 ∥u∥ (G + σ) β √ 1 -β + µ 2 ∥u∥ 2 . (1)
Proof. See Section D.1.
We expect that alternative online learning frameworks, such as follow-the-regularized-leader, can achieve the discounted regret bound similar to ( 1). However, a comprehensive exploration of discounted online learners lies outside the scope of this work.
Using the bound (1), we can derive an user-friendly nonconvex guarantee as follows. In the following two sections, only the regret bound (1) is important; the specifics of the discounted OMD algorithm are irrelevant. Any algorithm achieving a similar (or better) regret bound would provide the same (or improved) results.
Theorem 3.1 (Generic nonconvex guarantees). Consider the iterates generated according to Algorithm 1, where the online learner A achieves the discounted regret given by (1). Let ε > 0 be such that ε ≤ 7 2 (G + σ) and let T ≥ 49(G + σ) 2 ε -2 . Then, there exists a choice of parameters β = β ⋆ , D = D ⋆ , µ = µ ⋆ such that the following holds:
E τ ∥∇F (y τ )∥ [λ] ≤ 3ε + 4λ 1/2 ε -1/2 Γ x T .
Here the algorithm-dependent quantity Γ x is defined as
Γ x := 1 + 49(G + σ) 2 √ C x ε 2 E T t=1 (F (x t ) -F (w t ))
and the iterate stability factor C x is defined in Definition 5.
Proof. See Appendix B.
The key takeaway is that the nonconvex guarantees are determined by the magnitude of Γ x . Thus, it is essential to select the iterates x t in a way that minimizes Γ x . As a warm-up, we examine two simple special cases that lead to optimal complexity.
this section cite: ['b4']

Section: Warm-up: two simple ways to achieve optimal nonconvex guarantees
In this section, as a warm-up, we present two simple ways to achieve the optimal nonconvex guarantee using our general conversion scheme, Algorithm 1. Since Algorithm 1 maintains two sequences of iterates, x t and w t , perhaps the two simplest options for x t are as follows:
I. Set x t = w t-1 for all t ∈ [T ].
II. Set x t = x t-1 for all t ∈ [T ].
As we will demonstrate shortly, the first option recovers the previous conversion, while the second option leads to a novel conversion scheme, showcasing the versatility of our general framework. We will build on these warm-up cases to analyze schedule-free SGD in Section 5. Throughout this section, A can be any online learner that achieves the discounted regret bound (1).
this section cite: []

Section: Option I leads to previous conversion
We begin with the first option, as described in Algorithm 2.
Algorithm 2 Option I
1: In Algorithm 1, set x t = w t-1 for all t ∈ [T ].
Algorithm 2 recovers previous approaches (Cutkosky et al., 2023;Zhang and Cutkosky, 2024;Ahn and Cutkosky, 2024).
In particular, since Algorithm 2 leads to the update w tw t-1 = δ t , it provides a nice interpretation of selecting the increments w t -w t-1 based on the online learner's output, as highlighted by (Ahn et al., 2024).
The main advantage of Algorithm 2 is that the cumulative sum of loss decrements can be kept small due to a telescoping sum (recall that
∆ F := F (x 0 ) -inf x F (x)): T t=1 (F (x t ) -F (w t )) = T t=1 (F (w t-1 ) -F (w t )) = F (w 0 ) -F (w T ) ≤ ∆ F .(2)
Moreover, from Algorithm 2, we have ∥x t -x t-1 ∥ = ∥δ t-1 ∥ for t ≥ 2, and for t = 1, ∥x t -x t-1 ∥ = ∥w 0 -x 0 ∥ = 0. Therefore, we have
E T t=1 ∥x t -x t-1 ∥ 2 ≤ E T t=1 ∥δ t ∥ 2 ,(3)
which shows that the iterate stability factor C x is at most 1.
Combining these two calculations, Theorem 3.1 leads to the following nonconvex optimization guarantee.
Corollary 4.1. Consider the iterates of Algorithm 2. Under the setting of Theorem 3.1, it holds that E τ ∥∇F (y τ )∥ [λ] ≤ 4ε, provided that
T ≥ 49(G + σ) 2 ε -2 • max 8∆ F λ 1/2 ε -3/2 , 1 . (4)
Proof. With Algorithm 2, the above inequalities, (2) and (3), show that T t=1 (F (x t ) -F (w t )) ≤ ∆ F and C x ≤ 1. Therefore, it follows that
Γ x ≤ 1 + 49(G + σ) 2 ε 2 ∆ F ≤ 2 • 49(G + σ) 2 ε 2 ∆ F , since ε ≤ 7 2 (G + σ).
Thus, Theorem 3.1 yields:
E τ ∥∇F (y τ )∥ [λ] ≤ 3ε + 8λ 1/2 ε -1/2 49(G+σ) 2 ε 2 ∆ F T , provided that T ≥ 49(G + σ) 2 ε -2 . The second term in the upper bound is at most ε if T ≥ 49(G+σ) 2 •8∆ F λ 1/2 ε -7/2 .
Hence taking the maximum of the two requirments for T , the iteration complexity bound (4) follows.
We note that the complexity bound in Corollary 4.1 is optimal, in light of the lower bound results of (Zhang and Cutkosky, 2024). In fact, our use of the random index τ leads to a slightly better guarantee; the upper bound in (Zhang and Cutkosky, 2024) has the second argument of the maximum as O (G + σ)ε -1 , whereas ours is O (1).
We emphasize here that Algorithm 2 recovers commonly used momentum-based optimizers as special cases. Zhang and Cutkosky (2024) show that setting A as the OMD in Lemma 3.2 corresponds to SGD with momentum, while Ahn and Cutkosky (2024) demonstrate that choosing A as a discounted version of FTRL results in the Adam optimizer, up to minor modifications.
this section cite: ['b9', 'b0', 'b0', 'b0']

Section: Option II leads to a novel optimal conversion
We now consider the second option. Since x t = x t-1 for all t, it follows that x t ≡ x 0 for all t ∈ [T ].
Algorithm 3 Option II
1: In Algorithm 1, set x t = x 0 for all t ∈ [T ].
The main advantage of Algorithm 3 is that C x = 0 because x t is fixed for all iterates. This implies that
Γ x = E T t=1 (F (x 0 ) -F (w t )) .(5)
Hence, we only need to control the term (5), which can be done via the following anchoring scheme.
this section cite: []

Section: Algorithm 4
Anchoring scheme 1: Input: Initial iterate x 0 , and integers N, T ∈ N.
2: Set the initial anchor point a 1 := x 0 . 3: for n = 1, 2, . . . , N do 4:
Starting from a n , run Algorithm 3 for T iterations to generate the iterates {x
(n) t , w (n) t , y(n)
t } T t=1 . (By the choice in Algorithm 3, we have x (n) t ≡ a n for all t ∈ [T ].) 5: Sample the next anchor point a n+1 uniformly at random from {w (n) t } T t=1 . 6: end for Corollary 4.2. Consider the iterates of Algorithm 4. Under the setting of Theorem 3.1, it holds that
E n∼Unif([N ]) E τ ∇F (y (n) τ ) [λ] ≤ 4ε,(6)
provided that N ≥ 4∆ F λ 1/2 ε -3/2 and T ≥ 49(G + σ) 2 ε -2 . Hence, the total iteration complexity is:
T • N ≥ 49(G + σ) 2 ε -2 • max 4∆ F λ 1/2 ε -3/2 , 1 .
Proof. By Theorem 3.1 together with (5), the following holds for each epoch n:
E τ ∇F (y (n) τ ) [λ] ≤ 3ε + 4λ 1/2 ε -1/2 T E T t=1 F (a n ) -F (w (n) t ) = 3ε + 4λ 1/2 ε -1/2 E (F (a n ) -F (a n+1 )) ,
where the last equality follows because a n+1 is chosen uniformly at random from {w
(n) t } T t=1 .
Summing over all epochs, since
N n=1 (F (a n ) - F (a n+1 )) = F (a 1 ) -F (a N +1 ), we have: E n∼Unif([N ]) E τ ∇F (y (n) τ ) [λ] ≤ 3ε + 4λ 1/2 ε -1/2 N E (F (a 1 ) -F (a N +1 )) ≤ 3ε + 4λ 1/2 ε -1/2 ∆ F N .
Thus, setting N ≥ 4∆ F λ 1/2 ε -3/2 ensures that the righthand side is at most 4ε.
This result demonstrates that Algorithm 4 provides an alternative approach for achieving the optimal nonconvex guarantee. Interestingly, this method bears a conceptual resemblance to the classic online-to-convex conversion (Cesa-Bianchi et al., 2004). In particular, the traditional conversion runs an online learner for T iterations and then selects an iterate uniformly at random (or averages iterates, applying Jensen's inequality). This approach closely parallels the procedure of a single epoch of Algorithm 3. This is analogous to non-convex optimization approaches based upon repeatedly solving convex subproblems created by appropriate regularization, as discussed by Chen and Hazan (2024).
Thus far, we explored two simple methods for selecting x t , both of which can lead to the optimal nonconvex guarantee.
In the next section, we will consider yet another conversion approach that leads to the optimal guarantee.
this section cite: ['b5', 'b7']

Section: Schedule-free SGD is effective for nonconvex optimization
In this section, we build on Section 4 and consider another special case of Algorithm 1 that achieves the optimal nonconvex guarantee. Specifically, we fix the online learner A to be β-OMD from Lemma 3.2. This is in contrast to the previous sections in which the specifics of the online learner were not important. Recall the update rule of β-OMD:
δ t+1 = ζ (δ t -ηg t ) . (β-OMD)
where we let ζ := β 1+ηµ to simplify.
this section cite: []

Section: Yet another optimal conversion
Consider the following special case of Algorithm 1, specifically designed for β-OMD.
this section cite: []

Section: Algorithm 5
Option III 1: In Algorithm 1 with A chosen as β-OMD, choose x t = x t-1 + 1 ζ δ t for all t ∈ [T ].
We first demonstrate that this conversion scheme achieves the optimal nonconvex guarantee. The key idea is that, with Algorithm 5, the iterates x t remain sufficiently close to w t-1 , allowing us to take advantage of the telescoping sum from Algorithm 2. More specifically, with Algorithm 5, one can easily check that the following holds:
x t -w t-1 = 1 ζ δ t -δ t-1 = -ηg t-1 .(7)
In other words, ∥x t -w t-1 ∥ is significantly smaller than the size of the update made by A. With this observation, we can now show that Algorithm 5 achieves the optimal nonconvex optimization guarantee. Corollary 5.1. Consider the iterates of Algorithm 5 with the parameter choices as in Theorem 3.1, i.e., β = β ⋆ , D = D ⋆ , µ = µ ⋆ , and the parameters of β-OMD chosen as:
η = η ⋆ := 2 G + σ D ⋆ 1 -β ⋆ and ζ = ζ ⋆ := β ⋆ 1 + η ⋆ µ ⋆ . -η 1-ζ g t 1 ζ δ t+1 δ t+1 1 1-ζ δ t+1 x t x t+1 w t z t z t+1 w t+1
Figure 1. Illustration of how Algorithm 5 can be interpreted as schedule-free SGD. By defining the z-iterates according to ( 8), it becomes clear that the z-iterates follow the base SGD trajectory of schedule-free SGD (SF-SGD).
Then, it holds that E τ ∥∇F (y τ )∥ [λ] ≤ 5ε, provided that
T ≥ 49(G + σ) 2 ε -2 • max 20∆ F λ 1/2 ε -3/2 , 1 .
Proof. The proof is very similar to that of Corollary 4.1; see Appendix C.
Corollary 5.1 shows that Algorithm 5 achieves the same optimal complexity as the conversion methods discussed in Section 4, differing only by multiplicative constant factors.
Next, we will examine the update rule more closely.
this section cite: []

Section: Option III is equivalent to schedule-free SGD
A striking outcome of our general conversion framework (Algorithm 1) is that one of its special cases, Algorithm 5, turns out to be equivalent to the schedule-free SGD method.
To demonstrate this equivalence, we introduce a set of extrapolated iterates, z t , defined as follows:
z t := x t + 1 1 -ζ δ t .(8)
These z t iterates follow an SGD trajectory, just like the base method used in schedule-free SGD, with an effective step size of γ = η 1-ζ . See Section D.2 for details. Proposition 5.1. Under Algorithm 5 and using the extrapolated iterates defined in (8), the following holds:
z t+1 -z t = - η 1 -ζ g t .
Next, we demonstrate that Algorithm 5 mirrors the update rule of schedule-free SGD. We begin by presenting the explicit form of Algorithm 5, where we substitute the choice of x t according to Algorithm 5, i.e., x t = x t-1 + 1 ζ δ t , along with the choice of A as β-OMD. The resulting explicit update form is shown in Algorithm 6.
Algorithm 6 Explicit form of Algorithm 5 1: for t = 1, 2, . . . , T do 2:
Receive δ t from A =β-OMD, i.e., δ t = ζ(δ t-1 -ηg t-1 ).
3: Update x t = x t-1 + 1 ζ δ t . 4: Update w t = x t + δ t . 5: Set y t = x t + s t δ t , where s t is drawn uniformly from [0, 1] i.i.d. 6: Compute g t ← STOGRAD(y t ). 7: Send loss ℓ t (•) = ⟨g t , •⟩ + µ 2 ∥•∥ 2 to A. 8: end for Algorithm 7 Rewriting of Algorithm 5 using the extrapolated z-iterates (8) 1: Input: Initial iterates x 0 = z 0 . 2: for t = 1, 2, . . . , T do 3: Update x t = ζx t-1 + (1 -ζ)z t . 4: Set y t = κ t x t + (1 -κ t )z t , where κ t is drawn uniformly from [ζ, 1], i.i.d. 5:
Compute g t ← STOGRAD(y t ).
6:
Update z t+1 = z t -γg t , where the step size is chosen as γ = η 1-ζ . 7: end for We can observe that Algorithm 6 can be reformulated in terms of the iterates x t , y t , and z t , eliminating the dependence on w t . This follows from ( 8), along with the choice x t = x t-1 + 1 ζ δ t , which implies the following relationship:
x t = ζx t-1 + (1 -ζ)z t .
This result is also visually illustrated in Figure 1. Additionally, since y t = x t + s t δ t , we have:
y t = x t + s t (1 -ζ)(z t -x t ) = 1 -s t (1 -ζ) x t + s t (1 -ζ)z t .
By combining these steps, we obtain Algorithm 7, a reformulation of Algorithm 6. Notably, this algorithm is equivalent to the schedule-free SGD method. Specifically, Algorithm 7 selects κ t uniformly from the interval [ζ, 1] at each iteration, employs a step size γ = η 1-ζ , and consistently sets c t ≡ 1 -ζ for all t.
this section cite: []

Section: Practical insights from our results
Our results highlight an important property of schedule-free SGD: it not only achieves the optimal convex guarantee established by Defazio et al. (2024) but also attains the optimal nonconvex guarantee. This versatility helps explain the empirical success of schedule-free methods across a broad spectrum of optimization problems. It is important to note that our current analysis requires distinct parameter settings for γ and c t , depending on whether F is convex or nonconvex.
We now discuss how our results offer new insight into parameter selection for schedule-free SGD. To begin, we note the following fact (see Section D.3 for further details).
Proposition 5.2. With the parameter choices given in Corollary 5.1, we have
ζ = ζ ⋆ = 1 -Θ ε 2 (G+σ) 2 . Specifically, the parameter κ t in schedule-free SGD is chosen uniformly from [ζ ⋆ , 1], implying that 1 -Θ ε 2 (G+σ) 2 ≤ κ t ≤ 1 for all t.
This selection ensures that κ t remains close to 1.
We interpret these parameter choice in light of empirical findings by Defazio et al. (2024) that lacked theoretical explanation. Their convex guarantee (Defazio et al., 2024, Theorem 2) permits κ t to be chosen arbitrarily within the interval [0, 1], yet experimental results indicated that selecting κ t near 1 (e.g., 0.98) was crucial for strong empirical performance. Our results provides theoretical support for this choice, as noted in the first bullet point of Proposition 5.2.
this section cite: []

Section: Conclusion and future directions
Motivated by the impressive empirical performance of schedule-free methods, this work investigates their effectiveness for nonconvex optimization. As a first step, we demonstrate that schedule-free SGD achieves optimal iteration complexity for nonsmooth, nonconvex optimization. This is accomplished through a general conversion framework that not only recovers existing conversions but also introduces two novel conversion schemes. Notably, one of these novel conversions directly corresponds to schedulefree SGD, which serves as the basis for our analysis.
While this paper lays important groundwork, it merely scratches the surface of our understanding of schedule-free methods and opens up several avenues for future research. Below, we outline a few of these potential directions for the reader's interest.
Other special cases of our general conversion. In this work, we explore three specific instances of the general conversion framework. However, it is unlikely that these are the only viable conversions. Since these special cases yield highly practical optimizers, it would be worthwhile to investigate additional special cases and assess the practical implications of those conversions.
Adaptive schedule-free methods. Considering the impressive practical performance of the schedule-free version of Adam, as highlighted in (Defazio et al., 2024), it would be intriguing to explore whether this method can be under-stood as a special case of the general online-to-nonconvex conversion framework. Our current analysis provides a correspondence only for schedule-free SGD.
Advanced weighting schemes. The convex analysis of (Defazio et al., 2024) allows for an arbitrary sequence of "weights" for each example which inform the choice for c t in (SF-SGD). While uniform weighting (corresponding to c t = 1/t) is worst-case optimal, it is not instance optimal, and in fact certain empirical heuristics such as learning rate warmup can be recovered by instance-dependent weighting (Defazio et al., 2023). Our analysis makes use of exponentially increasing weights, corresponding to a consant c t = γ as detailed in Algorithm 7. While our weights achieve the optimal worst-case convergence guarantees, we conjecture that improvements are possible by incorporating instancedependent weighting.
Truly universal methods. It is noteworthy that the schedule-free method achieves optimal rates for nonsmooth losses regardless of their convexity. However, as discussed in Section 5.3, the current analysis requires distinct parameter settings for γ and c t , depending on whether F is convex or nonconvex. This indicates that we have not yet developed a fully unified algorithm that seamlessly addresses both cases. Nevertheless, these findings suggest the potential for a unified algorithmic framework. It would be valuable to explore whether a single parameter choice can be effective for both scenarios and to determine if such choices align with those observed in practice.
Why is schedule-free schedule-free? Lastly, we acknowledge that the main limitation of our analysis is its inability to fully explain why schedule-free methods can effectively alleviate the need for learning rate decay schedules, which is their most notable empirical advantage. Addressing this gap likely requires the development of a theoretical framework for understanding learning rate scheduling in nonconvex optimization.
this section cite: []

Section: References
Ref_id:b0 Title: Adam with model exponential moving average is effective for nonconvex optimization Year: (2024)
Ref_id:b1 Title: Understanding adam optimizer via online learning of updates: Adam is ftrl in disguise Year: (2024)
Ref_id:b2 Title: Second-order information in nonconvex stochastic optimization: Power and limitations Year: (2020)
Ref_id:b3 Title: Lower bounds for non-convex stochastic optimization Year: (2023)
Ref_id:b4 Title: Mirror descent and nonlinear projected subgradient methods for convex optimization Year: (2003)
Ref_id:b5 Title: On the generalization ability of on-line learning algorithms. Information Theory Year: (2004)
Ref_id:b6 Title: Faster gradient-free algorithms for nonsmooth nonconvex stochastic optimization Year: (2023)
Ref_id:b7 Title: Open problem: Black-box reductions and adaptive gradient methods for nonconvex optimization Year: (2024)
Ref_id:b8 Title: Anytime online-to-batch, optimism and acceleration Year: (2019)
Ref_id:b9 Title: Optimal stochastic non-smooth non-convex optimization through onlineto-non-convex conversion Year: (2023)
