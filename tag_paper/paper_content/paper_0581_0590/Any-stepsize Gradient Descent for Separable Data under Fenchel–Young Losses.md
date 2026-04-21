Title: Any-stepsize Gradient Descent for Separable Data under Fenchel-Young Losses
Abstract: The gradient descent (GD) has been one of the most common optimizer in machine learning. In particular, the loss landscape of a neural network is typically sharpened during the initial phase of training, making the training dynamics hover on the edge of stability. This is beyond our standard understanding of GD convergence in the stable regime where stepsize is chosen sufficiently smaller. Recently, Wu et al. [63]  have shown that GD converges with much larger stepsize under linearly separable logistic regression. Although their analysis hinges on the self-bounding property of the logistic loss, which seems to be a cornerstone to establish a modified descent lemma, our pilot study shows that other loss functions without the selfbounding property can make GD attain arbitrarily small loss with large stepsize. To further understand what property of a loss function matters in GD, we aim to show large-stepsize GD convergence for a general loss function based on the framework of Fenchel-Young losses. We essentially leverage the classical perceptron argument to derive the iteration complexity for achieving ε-optimal loss, which is possible for a majority of Fenchel-Young losses. This convergence result highlights that the self-bounding property may not be necessary for GD to attain arbitrarily small loss. Moreover, when a loss function entails separation margin, a notion relevant to the margin in support vector machines, GD often yields faster convergence than typical GD rate T = Ω(ε -1 ) for convex smooth objectives. Specifically, GD with the Tsallis entropy attains ε-optimal loss with the rate T = Ω(ε -1/2 ), and the Rényi entropy achieves the far better rate T = Ω(ε -1/3 ).

Section: Introduction
Gradient-based optimizers are prevalent in the modern machine learning community with deep learning thanks to its scalability and plasticity. Among many variants, GD remains to be a standard choice. GD with constant stepsize is written as follows: w t+1 := w t -η∇L(w t ), for t = 0, 1, . . . , T -1,
where w ∈ R d is the optimization variables, L(•) is the loss function, and η > 0 is stepsize fixed across all steps. The descent lemma [42, Section 1.2.3] is a key to GD convergence: for β-smooth objective L, the stepsize choice η < 2/β ensures that L(w t ) monotonically decreases. Nonetheless, little optimization theory has been known beyond the threshold η > 2/β; though modern neural networks exhibit much smaller smoothness values than practically used stepsize values [66,57].
2 1 2 3 2 5 2 7 2 9 t 2 20 2 17 2 14 2 11 2 8 2 5 2 2 2 1 2 4 L(w t ) tsallis (q = 0.5) = 2 0 = 2 1 = 2 2 = 2 3 = 2 4 2 1 2 3 2 5 2 7 2 9 t 2 20 2 17 2 14 2 11 2 8 2 5 2 2 2 1 2 4 L(w t ) logistic = 2 0 = 2 1 = 2 2 = 2 3 = 2 4 2 1 2 3 2 5 2 7 2 9 t 2 20 2 17 2 14 2 11 2 8 2 5 2 2 2 1 2 4 L(w t ) tsallis (q = 1.5) = 2 0 = 2 1 = 2 2 = 2 3 = 2 4 2 1 2 3 2 5 2 7 2 9 t 2 20 2 17   2 14   2 11   2 8   2 5   2 2   2 1   2 4L(w t ) tsallis (q = 2.0) = 2 0 = 2 1 = 2 2 = 2 3 = 2 4   Figure 1: Pilot studies of GD with the same toy dataset as [63]. The dataset consists of four points, x1 = [1, 0.2] ⊤ , y1 = 1, x2 = [-2, 0.2] ⊤ , y2 = 1, x3 = [-1, -0.2] ⊤ , y3 = -1, x4 = [2, -0.2] ⊤ , and y4 = -1.
GD is run with initialization w0 = [0, 0] ⊤ . Note that the logistic loss corresponds to the Tsallis 1-loss. The Tsallis 2and q-loss are also known as the modified Huber loss [67] and q-entmax loss [45], respectively.
Moreover, recent studies have reported that GD trajectories of neural networks tend to inflate the sharpness of the loss landscape and hover on the edge of stability (EoS) before convergence [34,18,2].
Among several recent developments in the theory of large-stepsize GD (which we will review in Section 1.1), Wu et al. [63] investigated the large-stepsize behavior of GD by using the binary logistic regression with a linearly separable data, a minimal synthetic setting. They showed that GD initially oscillates with non-monotonic loss values (the EoS phase), which terminates in finite time (phase transition), and then the loss value decreases monotonically (the stable phase). Beyond the logistic loss, these results have been extended to loss functions with the self-bounding property: for a differentiable loss function ℓ : R → R and its absolute derivative g(•) := |ℓ ′ (•)|, ℓ satisfies ℓ(z) ≤ ℓ(x) + ℓ ′ (x)(z -x) + C β g(x)(z -x) 2 ∀z, x with |z -x| < 1, for some C β > 0. (1)
The self-bounding property generalizes the polynomially-tailed loss [31,30], and refines the standard smoothness property by allowing the smoothness modulus locally adaptive to the derivative, such that C β g(x). Thus, large η can be cancelled out with the vanishingly small loss gradient after the phase transition [63,Lemma 29], and GD follows the descent direction.
In this paper, we study GD with large stepsize under a wide range of loss functions to identify a key factor to induce the convergent behavior. This is motivated by our pilot study shown in Figure 1, where we found that GD with large stepsize such as η = 2 4 remains to converge under the Tsallis q-loss (detailed in Section 4), even if the stepsize has gone beyond the classical stable regime. It is noteworthy therein that the Tsallis q-loss with q > 1 does not enjoy the self-bounding property. How much does the self-bounding property play a vital role in large-stepsize GD convergence?
We specifically consider Fenchel-Young losses [11], a class of convex loss functions generated by a potential function ϕ, as a template of loss functions. Fenchel-Young losses have been used in applications such as structured prediction [43], differentiable programming [10], and model selection [7], while being used as a theoretical tool for online learning [51,52]. We identify that Fenchel-Young losses with separation margin (formally introduced in Section 2), a relevant notion to the margin in support vector machines, can often benefit from better GD convergence rates. We say a loss function has separation margin if the loss value vanishes with a sufficiently large positive prediction margin. Specifically, our main result is informally stated as follows.
this section cite: ['b65', 'b56', 'b62', 'b66', 'b44', 'b33', 'b17', 'b1', 'b62', 'b30', 'b29', 'b62', 'b10', 'b42', 'b9', 'b6', 'b50', 'b51']

Section: Theorem 1 (Informal version of Theorem 5).
Consider a binary classification dataset that is linearly separable. We run (GD) with arbitrary constant stepsize η > 0 and initialization w 0 = 0 under a Fenchel-Young loss generated by twice continuously differentiable and convex potential ϕ with separation margin. For ε > 0, after at most T steps of (GD), where T = Ω(ε -α ) and α = lim sup µ↓0 ϕ ′ (µ) µϕ ′′ (µ) 1 -ϕ(µ) µϕ ′ (µ) , we have L(w T ) ≤ ε. 1As defined in Section 2, a loss function with separation margin vanishes for a sufficiently large prediction margin, which is a natural indicator of correct classification used in support vector machines. The order of the convergence rate T = Ω(ε -α ) differs across various potential ϕ. With a specific choice, the rate can be T = Ω(ε -1/2 ) (with ϕ being the Tsallis 2-entropy) and T = Ω(ε -1/3 ) (with ϕ being the Rényi 2-entropy, also known as the collision entropy [14]). Remarkably, these convergence rates are better than the classical GD convergence rate T = Ω(ε -1 ) under the stable regime, and even better than the convergence rate of the logistic loss after undergoing the EoS and phase transition [63]. Both the Tsallis and Rényi entropies above lack the self-bounding property but have separation margin. Therefore, we advocate the importance of separation margin for better GD convergence rates. We compare different Fenchel-Young losses in Section 4 and contrast our convergence result with the EoS and implicit bias in Section 5.
We present Theorem 1 formally in Section 3. Our proof leverages the classical perceptron argument [44] without relying on the descent lemma at all. Intuitively speaking, we track the growth of the parameter alignment ⟨w t , w * ⟩ with the optimal separator w * . When a loss entails separation margin, ⟨w t , w * ⟩ cannot grow arbitrarily large (as we simulate in Figure 2 later) while each step of (GD) improves a lower bound on ⟨w t , w * ⟩, leading to the convergence. Section 3.1 describes this proof overview in detail. This is different from the proof of Wu et al. [63], whose core is the modified decent lemma (recapped in Lemma 21 in the appendix) based on the self-bounding property.
Although the perceptron argument is partially used therein [63], the average loss is finally controlled by the modified descent lemma, and thus the proof is only applicable to the self-bounding losses.
this section cite: ['b13', 'b62', 'b43', 'b62', 'b62']

Section: Related work
Gradient descent with large stepsize has attracted attention recently. Specifically, non-monotonic behaviors of loss functions [65] and the sharpness adaptivity to loss landscapes [34,18] have been observed empirically. It was argued that the sharpness tends to initially increases until the classical stable regime breaks down, and hovers on this boundary, termed as the edge of stability [18]. This observation mainly sparks two questions: why the loss landscape hovers on the EoS, and why converging. Answering either question must go beyond the classical optimization theory under the stable regime.
On why hovering on the EoS, let us make a brief review, though it is not a central focus of this paper: Ahn et al. [2] is a seminal work to empirically investigate the homogeneity of loss functions contributes to maintain the EoS. Later, it was showed that normalized GD (represented by scaleinvariant losses) adaptively leads their intrinsic stepsize toward sharpness reduction [36]. The sharpness fluctuation is often attributed to the non-negligible third-order Taylor remainder of the loss landscape [37,20].
We rather focus on why GD attains arbitrarily small loss with much larger stepsize. In this line, previous studies show convergence based on specific models such as multi-scale loss function [32], quadratic functions [5], matrix factorization [59,17], a scalar multiplicative model [68,33], a sparse coding model [3], and linear logistic regression [62]. Among them, we advocate the logistic regression setup proposed by Wu et al. [62] because it is relevant to implicit bias of GD [54,29,46], and moreover, Wu et al. [63] corroborates the benefit of large stepsize in GD convergence rate. Our work is provoked by Wu et al. [63], questioning what structure in a loss function leads GD to arbitrarily small loss. Indeed, we observe in Figure 1 that loss functions without the self-bounding property (1) can make GD attain arbitrarily small loss, though the self-bounding property seems essential to calm the EoS down to the stable phase [63] as well as to establish the max-margin directional convergence [29,46]. A similar question to ours is raised by Tyurin [58], who argues that the stable convergence of large-stepsize logistic regression might be an artifact due to the functional form of the logistic loss-eventually Tyurin [58] argued that large-stepsize logistic regression behaves like the classical perceptron. To this end, we show in Theorem 5 that arbitrary-stepsize GD can converge under a wide range of losses even without the self-bounding property (1), and moreover, occasionally yielding a better rate than the classical stable convergence rate. We discuss it more in Section 5. Note that one work attempts to extend the separable logistic regression setup to the non-separable one [41]; yet, we still do not have satisfactory results beyond the one-dimensional case. Due to its intricateness, we follow the separable case.
Lastly, our work benefits the study of regret bounds of surrogate losses [9,1,23,6,38,8]. A surrogate regret bound connect a surrogate loss to a downstream task loss, while the optimization error of the surrogate loss is usually ignored. Our GD convergence analysis can be integrated to surrogate regret bounds when discussing a downstream task performance.
this section cite: ['b64', 'b33', 'b17', 'b17', 'b1', 'b35', 'b36', 'b19', 'b31', 'b4', 'b58', 'b16', 'b67', 'b32', 'b2', 'b61', 'b61', 'b53', 'b28', 'b45', 'b62', 'b62', 'b62', 'b28', 'b45', 'b57', 'b57', 'b40', 'b8', 'b0', 'b22', 'b5', 'b37', 'b7']

Section: Notation
Let R ≥0 be the set of nonnegative reals. Let [n] := {1, . . . , n} for n ∈ N. Let 1 be the all-ones vector and e i ∈ R d be the i-th standard basis vector, i.e., all zeros except for the i-th entry being one. For S ⊆ R d , int(S) denotes its (relative) interior, and I S : R → {0, ∞} its indicator function, which takes zero if µ ∈ S and ∞ otherwise. For Ω : R d → R ∪ {∞}, dom(Ω) := µ ∈ R d Ω(µ) < ∞ denotes its effective domain and Ω * (θ) := sup ⟨θ, µ⟩ -Ω(µ) µ ∈ R d its convex conjugate. Let △ d := µ ∈ R d ≥0 ⟨1, µ⟩ = 1 be the probability simplex. We introduce C k (I) as the set of k-th continuously differentiable functions on the interval I ⊆ R. [49,Section 26]).
Let Ψ : R d → R ∪ {∞} be a strictly convex function differentiable throughout int(dom Ψ) ̸ = ∅. We say Ψ is of Legendre-type if lim i→∞ ∥∇Ψ(x i )∥ 2 = ∞ whenever x 1 , x 2 , . . . is a sequence in int(dom Ψ) converging to a boundary point of int(dom Ψ) (see
this section cite: ['b48']

Section: Preliminary on Fenchel-Young losses
Fenchel-Young losses have been introduced by Blondel et al. [11] as a general class of surrogate losses for structured prediction, which are classification-calibrated [60]. This can be seen as a Bregman divergence comparing primal and dual points [25,4]. Despite that the logistic and hinge losses are widely prevailing in practice, we can improve the performance of some prediction tasks by changing specific Fenchel-Young losses, as reported by Roulet et al. [50]. We choose Fenchel-Young losses because a vast majority of convex, Lipschitz, and classification-calibrated losses are included in this class-otherwise, GD convergence is hardly obtained beyond the edge of stability. Moreover, the separation margin property, one of the key features of Fenchel-Young losses, controls GD behaviors significantly. Definition 2. Let Ω : R K → R ∪ {∞} be a potential function. The Fenchel-Young loss ℓ Ω : dom(Ω * ) × dom(Ω) → R ≥0 generated by Ω is defined as ℓ Ω (θ; µ) := Ω * (θ) + Ω(µ) -⟨θ, µ⟩ .
In multiclass classification, ℓ Ω (θ; µ) measures the proximity between a score θ and a target label µ = e i (for a class i ∈ [K]). By definition, ℓ Ω (•, µ) is convex for any µ ∈ dom(Ω). Moreover, ℓ Ω (θ; µ) = 0 holds if and only if µ ∈ ∂Ω * (θ) due to the equality condition of the Fenchel-Young inequality.
We follow [11,Section 4.4] to consider binary (K = 2) loss functions. The following set of assumptions is imposed on a potential function Ω. The asymmetric generalization is possible, but we choose to keep the analysis simpler so that we can focus more on the essence of GD convergence. Assumption 1. For a potential function Ω, assume dom(Ω) ⊆ △ K and that Ω satisfies the zeroentropy condition Ω(µ) = 0 for µ ∈ {e i } i∈[K] ; convexity Ω((1 -α)µ + αµ ′ ) ≤ (1 -α)Ω(µ) + αΩ(µ ′ ) for µ ̸ = µ ′ and α ∈ (0, 1); symmetry Ω(µ) = Ω(Pµ) for any K × K permutation P.
Let us restrict ourselves to K = 2 (binary classification) and write ϕ(µ) := Ω([µ, 1 -µ] ⊤ ). If we choose θ = [s, -s] ⊤ ∈ R 2 as a score vector, the Fenchel-Young loss can be written as
ℓ Ω (θ; e i ) = ϕ * (-s) if i = 1, ϕ * (s) if i = 2,
and dom(ϕ * ) = R. Hence, the Fenchel-Young loss is simplified as ϕ * (-ys) if we relabel two classes i = 1 and i = 2 with y = 1 and y = -1, respectively. Thus, we suppose the form of a symmetric margin-based loss function ℓ(z) := ϕ * (-z). Therein, a Fenchel-Young loss ℓ extends a proper canonical composite loss [47] over the entire prediction space z ∈ R, as discussed in [7].
Separation margin. For specific potential functions, Fenchel-Young losses entail separation margin [11,Section 5], which is a generalized notion of classical margin in support vector machines.
Definition 3. For a loss ℓ : R → R ≥0 , we say ℓ has the separation margin property if there exists m > 0 such that any prediction z ≥ m incurs ℓ(z) = 0. The smallest m is the separation margin of ℓ.
Hence, ℓ(z) = 0 indicates the prediction z ∈ R not only correctly classifies a given point but also has safe margin m away from the classification boundary z = 0. It is shown that the existence of the separation margin property can be tested through the subgradient ∂ϕ [11,Proposition 6].
this section cite: ['b10', 'b59', 'b24', 'b3', 'b49', 'b10', 'b46', 'b6', 'b10', 'b10']

Section: Proposition 4 ([11]).
A Fenchel-Young loss ℓ(z) = ϕ * (-z) satisfying Assumption 1 has separation margin if and only if ∂ϕ(µ) ̸ = ∅ for any µ ∈ [0, 1]. When ϕ ∈ C 1 ((0, 1)) has separation margin m,
m = -lim µ↓0 ϕ ′ (µ).
For a differentiable ϕ, the nonempty-subgradient condition requires that the derivative ϕ ′ (µ) does not explode at the boundary points of the domain µ ∈ dom(ϕ) = {0, 1}. In this case, ϕ is not of Legendre-type [49]. As we will see later, the convergence behavior of GD hinges on the separation margin property of a loss function. More detailed analysis of the separation margin property for binary classification can be found in [7].
In Section E, we show that a loss satisfying the self-bounding inequality (1) does not have separation margin (but not the other way around).
this section cite: ['b48', 'b6']

Section: Examples.
With the Shannon negentropy ϕ(µ) = µ ln µ + (1 -µ) ln(1 -µ), we recover the logistic loss ϕ * (-z) = ln(1 + exp(-z)). With the negative of the Gini index ϕ(µ) = µfoot_1 -µ, we can generate the modified Huber loss ϕ * (-z) = max {0, 1 -z} 2 /4 if z ≥ -1 and ϕ * (-z) = -z otherwise [67], which is the binarized sparsemax loss [39]. If we choose ϕ(µ) = max {µ, 1 -µ}, we recover the hinge loss ϕ * (-z) = max {0, 1 -z}. We discuss more examples in Section 4.
this section cite: ['b66', 'b38']

Section: Convergence of large stepsize GD under Fenchel-Young losses
We consistently assume that the dataset is bounded and linearly separable. Assumption 2. Assume the training data (x i , y i ) i∈[n] satisfies
• for every i ∈ [n], ∥x i ∥ ≤ 1 and y i ∈ {±1};
• there is γ > 0 and a unit vector w * such that ⟨w * , z i ⟩ ≥ γ for every i ∈ [n], where z i := y i x i .
Instead of logistic regression, we choose a Fenchel-Young loss ℓ(z) = ϕ * (-z) associated with a binary potential function ϕ, and minimize the following risk by (GD) with fixed stepsize η > 0 to learn a linear classifier w:
2 L(w) := 1 n i∈[n] ℓ(⟨w, y i x i ⟩) = 1 n i∈[n] ℓ(⟨w, z i ⟩).(2)
We impose the following assumptions on our loss function. Assumption 3. Consider a loss ℓ : R → R ≥0 .
A. Fenchel-Young loss. Assume that ℓ(z) is a Fenchel-Young loss ϕ * (-z) generated by a potential ϕ : R → R ∪ {∞} such that ϕ ∈ C 2 ((0, 1)) satisfies Assumption 1, ϕ is strictly convex, and ϕ ′′ > 0 on the interval (0, 1).
this section cite: []

Section: B.
Regularity. Assume that ρ(λ) := min z∈R λℓ(z) + z 2 (for λ ≥ 1) is well-defined.
C. Lipschitz continuity. For g(•) := |ℓ ′ (•)|, assume g(•) ≤ C g for some C g > 0.
We will later see that ρ characterizes the growth rate of the parameter norm ∥w t ∥ during GD in (5). This notion is inherited from [63]. Now, we are ready to state our main result, the GD rate to attain arbitrarily small loss for linearly separable data under Fenchel-Young losses. Remarkably, we show convergence without the self-bounding property of a loss function, unlike [63]. Theorem 5 (Main result). Suppose Assumption 2 and consider (GD) with stepsize η > 0 and w 0 = 0 under a Fenchel-Young loss ℓ satisfying Assumption 3. For any ε ∈ (0, 1), let α := sup µ∈(0,ε]
ϕ ′ (µ) µϕ ′′ (µ) 1 - ϕ(µ) µϕ ′ (µ) and C ϕ := μ [μϕ ′ (μ) -ϕ(μ)] α ,(3)
where C ϕ > 0 depends on ϕ and ε solely and μ := min g(ℓ -1 (ε)), 1 . If α, C ϕ ∈ (0, ∞) and
for ε ∈ (0, ε), T > n C ϕ γ 2 4 ρ(γ 2 ηT ) η + C g ε -α
holds, then we have min t∈[T ] L(w t ) ≤ ε.
This convergence guarantee even applies to non-smooth Fenchel-Young losses as long as Assumption 3 is satisfied-note that ϕ must be strongly convex to ensure the smoothness of the associated Fenchel-Young loss [11, Proposition 2.4]. As seen later in Section 4, α and C ϕ neither diverge nor degenerate for arbitrarily small ε under many examples of ϕ. When ℓ has separation margin, Lemma 9 in Section A provides a finite upper bound on ρ, yielding the following simpler form. Corollary 6. Under the same setup with Theorem 5, we additionally assume that ℓ has separation margin m > 0. For any ε ∈ (0, 1), if (α, C ϕ ) defined in (3) satisfies α, C ϕ ∈ (0, ∞) and
for ε ∈ (0, ε), T > n C ϕ γ 2 4m η + C g ε -α
holds, then we have min t∈[T ] L(w t ) ≤ ε.
A loss function without separation margin does not have finite ρ, which typically yields slower convergence as we see in Section 4. Therefore, (GD) operated on many common Fenchel-Young losses converges under the separability, regardless of the choice of η. Note that the classical GD convergence analysis under convex smooth functions provides T = Ω(ε -1 ). As we see later in Section 4, some loss functions entail better rates with α < 1, summarized in Table 1.
this section cite: ['b62', 'b62']

Section: Proof outline
The proof of Theorem 5 essentially relies on the perceptron convergence analysis [44] and the asymptotical order evaluation of rate functions [6]. We sketch the proof in this section to highlight the structure of the GD convergence in our setup and complete the proof in Section B.
When we show the convergence of perceptron, we leverage an inequality of the following type:
C L t ≤ ⟨w t , w * ⟩ (♣) ≤ ∥w t ∥ ≤ C U (t) (♢) for t ≥ 1,(4)
where C L > 0 is a non-degenerate constant independent of t. The inequality (♣) holds only while perceptron misclassifies some examples. Thus, perceptron correctly classifies all examples after at most T iterations such that C L T > C U (T ). Such T exists as long as C U (t) is sublinear in T .
When it comes to our setup, an inequality (♣) is obtained by recursively expanding the update (GD)foot_2
⟨w t , w * ⟩ ≥ ⟨w t-1 , w * ⟩ + γη n g( w t-1 , z it-1 ) ≥ • • • ≥ γη n t-1 k=0 g(⟨w k , z i k ⟩) =: γη G(w k ),
where z i k is a misclassified example by w k . Perceptron enjoys an inequality of (♣)-type immediately because it optimizes the loss function ℓ per (z) = max {-z, 0}, which yields g(z) = 1 if z < 0 (i.e., if misclassified). When considering a Fenchel-Young loss satisfying Assumption 3, we do not have a non-degenerate lower bound for g(z) because we can make g(z) arbitrarily close to zero. Instead, we lower-bound g(z) by a (non-degenerate) error tolerance ε 1 > 0, g(z) ≥ ε 1 , before we attain the ε-optimal loss. Lemma 11 and (a part of) Lemma 13 in Section B are relevant to (♣). Note that the perceptron argument is used in [63] but in a different way: they control L(w k ) through the upper bound on G(w k ) (see Lemma 25). This is applicable only to self-bounding losses.
To obtain an inequality of (♢)-type, by following the standard perceptron analysis, we directly expand the update (GD) ∥w t ∥ 2 = ∥w t-1 -η∇L(w t-1 )∥ 2 recursively, and upper-bound it by noting that ℓ per has separation margin, leading to C U (t) = O( √ t). Though this is possible for a Fenchel-Young loss with separation margin, we can improve this bound by borrowing the split optimization technique, introduced by [63]. Eventually, we can upper-bound ∥w t ∥ as follows:
∥w t ∥ ≤ 4 ρ(γ 2 ηt) + ηC g γ .(5)
In particular, we have ρ(λ) = O(1) when a loss has separation margin (see Lemma 9), and therein C U (t) = O(1). This is where separation margin plays a crucial role. We recap the split optimization technique in Lemma 10, based on which Lemma 19 in Section D shows this inequality of (♢)-type.
The remaining piece is to assess the order of the convergence rate. After solving the inequality (♣, ♢) with t = T being the stopping time, we have T as a function of the error tolerance ε, T = f (ε), where f is a nondecreasing rate function depending on ϕ. To characterize the asymptotic order at vanishing ε, we attempt to evaluate in the form f (ε) ≃ ε α0 for an order parameter α 0 > 0, which can be estimated by εf ′ (ε) f (ε) ε↓0 -→ α 0 , if the limit exists.
Thus, the order parameter α 0 is solely determined by the functional form of potential function ϕ. This technique has been initially developed in functional analysis to estimate moduli of Banach and Orlicz spaces [53,27,13], and recently introduced in convex analysis to approximate a convex function by power functions [28] and estimate moduli of convexity [6,8]. The general statement of the order evaluation is given in Lemma 12 and instantiated for GD convergence in Lemma 13 in Section B.
this section cite: ['b43', 'b5', 'b62', 'b62', 'b52', 'b26', 'b12', 'b27', 'b5', 'b7']

Section: Examples of loss functions
Now, we instantiate Theorem 5 for several examples of Fenchel-Young losses to discuss the convergence rate. Instead of specifying a loss function ℓ(z) = ϕ * (-z), we directly specify its potential function ϕ subsequently. For each ϕ, we compute (α, C ϕ ) in (3) to investigate the convergence rate given by Theorem 5, by taking ε (and thus μ) vanishingly small. In addition, we can compute separation margin m by Proposition 4 if exists; otherwise, we need to compute ρ for a loss (see Lemma 27). Table 1 summarizes different loss functions and their GD convergence rates. All the detailed calculations are deferred to Section F, where we have an additional example of ϕ (pseudo-spherical entropy) with non-converging α.
this section cite: []

Section: Shannon entropy. Consider the binary Shannon (neg)entropy
ϕ(µ) = µ ln µ + (1 -µ) ln(1 -µ).
The generated Fenchel-Young loss is the logistic loss ℓ(z) = ln(1 + exp(-z)), which enjoys the selfbounding property and hence does not have separation margin (see Section E). The loss parameters are α = 1 and C ϕ = 1. Moreover, we know C g = 1 and ρ(λ) ≤ 1 + ln 2 (λ) [63]. Plugging this back to Theorem 5, we have the ε-optimal risk at most after
T ≳ 4 √ 2(log 2 (γ 2 η) + 1) η + 1 ln 2 nε -1 γ 2 iterations,
where logarithmic factors in ε -1 are ignored. This indicates the rate T = Ω(ε -1 ), recovering the standard GD convergence rate under the stable regime but with arbitrary stepsize η. In Section 5, we compare this rate with [63] in more detail.
this section cite: ['b62', 'b62']

Section: Table 1: Comparison of Fenchel-Young losses generated by different potential function ϕ.
Here, m = ∞ and β = ∞ indicate the lack of separation margin and smoothness, respectively. Since we do not have closed-form β for the Rényi entropy with q ∈ (1, 2), we merely show its lower bound. The convergence rates ignore the dependency on {m, n, γ, η}, and hold for arbitrary stepsize η regardless of η < 2/β.
Potential ϕ Parameter q Sep. mgn. m Smoothness β Order α Conv. rate for
T Shannon - ∞ 1/4 1 Ω(ε -1 ) Semi-circle - ∞ 1/4 2 Ω(ε -4 ) Tsallis (0, 1) ∞ 2 q-3 q 1 q Ω(ε -2/q ) (1, 2] q q -1 Ω(ε -1/q ) (2, ∞) ∞ 1/2 Ω(ε -1/2 ) Rényi (0, 1) ∞ 1/4q 1 q Ω(ε -2/q ) (1, 2) q q -1 (≥ 1/4q) Ω(ε -1/q ) 2 ∞ 1/3 Ω(ε -1/3 ) Semi-circle entropy. Consider ϕ(µ) = -2 µ(1 -µ).
The generated Fenchel-Young loss (we call the semi-circle loss) ℓ(z) = (-z + √ z 2 + 4)/2 enjoys the self-bounding property and does not have separation margin since ϕ ′ (µ) → -∞ as µ ↓ 0 (see Section E). The semi-circle loss is relevant to the exponential/boosting loss ℓ exp (z) = exp(-z), which has the semi-circle entropy as the Bayes risk [15,1]. The loss parameters are α = 2 and C ϕ = 1. Moreover, we have C g = 1 and ρ(λ) ≤ 5λ/(2 ln λ). Plugging this back to Theorem 5, we have the ε-optimal risk at most after
T > 40n 6 γ 2 η ln(2γ 2 η) ε -4
extra price for lacking separation margin
+ 2n γ 2 ε -2 iterations,
where the first term Ω(ε -4 ) is an extra price due to the lack of separation margin of the semi-circle loss. For arbitrary stepsize η, the convergence rate is T = Ω(ε -4 ), and stepsize η as large as η = Ω(ε -2 ) improves the rate to be T = Ω(ε -2 ) by cancelling the extra term out.
This convergence rate of the semi-circle loss is even worse than the GD convergence rate for general convex smooth functions, T = Ω(ε -1 ). This is because the perceptron argument is merely sufficient for GD convergence. Nonetheless, the perceptron argument more informatively states that we have ⟨w t , w * ⟩ /∥w t ∥ ≳ ε α after minimizing the risk at the ε-optimal level-by combining the inequalities (♣, ♢) (in Eq. ( 4)). This indicates that the loss function with larger α yields slower parameter alignment toward w * .
this section cite: ['b14', 'b0']

Section: Tsallis entropy.
For q > 0 with q ̸ = 1, consider the Tsallis q-(neg)entropy
ϕ(µ) = µ q + (1 -µ) q -1 q -1
generalizing the Shannon entropy for non-extensive systems [55]. It recovers the Shannon entropy at the limit q → 1. The generated Fenchel-Young loss is known as the q-entmax loss [45]. We divide the case depending on parameter q:
• When 0 < q < 1: (α, C ϕ ) = (1/q, 1), and ϕ * does not have separation margin.
• When 1 < q ≤ 2: (α, C ϕ ) = (1/q, 1), and ϕ * has separation margin m = q/(q -1).
• When 2 < q: (α, C ϕ ) = (1/2, 2/q), and ϕ * has separation margin m = q/(q -1).
For all cases, α and C ϕ stay in (0, ∞). The convergence rate is T = Ω(ε -2/q ) for q ∈ (0, 1) (by Corollary 28); T = Ω(ε -1/q ) for q ∈ (1, 2); T = Ω(ε -1/2 ) for 2 ≤ q. This suggests that we have a better convergence rate over the Shannon case when q > 1 and the best rate is Ω(ε -1/2 ).
this section cite: ['b54', 'b44']

Section: Rényi entropy.
For q ∈ (0, 2] \ {1}, consider the Rényi q-(neg)entropy
ϕ(µ) = 1 q -1 ln [µ q + (1 -µ) q ]
generalizing the Shannon entropy (with the limit q → 1) while preserving additivity for independent events [48]. The Rényi entropy extended beyond q > 2 becomes nonconvex, which we do not consider. The Rényi 2-entropy is referred to as the collision entropy [14].
We divide the case depending on parameter q:
• When 0 < q < 1: (α, C ϕ ) = (1/q, 1), and ϕ * does not have separation margin.
• When 1 < q < 2: (α, C ϕ ) = (1/q, 1), and ϕ * has separation margin m = q/(q -1).
• When q = 2: (α, C ϕ ) = (1/3, 3 3/8), and ϕ * has separation margin m = 2.
For all cases, α and C ϕ stay in (0, ∞). The convergence rate is T = Ω(ε -2/q ) for q ∈ (0, 1) (by Corollary 28); T = Ω(ε -1/q ) for q ∈ (1, 2); T = Ω(ε -1/3 ) for q = 2. Surprisingly, we have a "leap" of the order from ε -1/q to ε -1/3 as q ↑ 2, and the convergence rate Ω(ε -1/3 ) is far better than the Shannon and Tsallis cases. When q = 2, Corollary 6 implies that we have the ε-optimal risk at most after
T > 3 8/3 n γ 2 8 η + 1 ε -1/3 iterations.
this section cite: ['b47', 'b13']

Section: Discussion and open problems
Comparison with Wu et al. [63]. The large-stepsize logistic regression has been shown to exhibit the following phase transition [63]: the GD sequence initially stays in the EoS phase such that the risk L(w t ) fluctuates initially with its average t -1 k L(w k ) controlled. Once we experience L(w t ) ≲ min {1/η, ℓ(0)/n}, which is possible within O(η) steps at most, GD leaves the EoS and the loss converges in the rate L(w t ) = O(1/(ηt)). The stepsize η trades off the phase transition time for the stable convergence rate. If we know the maximum number of steps T in advance, the choice η = Θ(T ) balances them, achieving the acceleration to L(w t ) = O(1/T 2 ). We detail them in Section D. This is arguably interesting to demonstrate how GD benefits from large stepsize.
Nevertheless, we would like to highlight two caveats. First, we must undergo L(w t ) ≤ ℓ(0)/n before exiting the EoS phase. This means that our linear model has already classified all points correctly during the EoS phase since any single point z i incurs loss at most ℓ(⟨w s , z i ⟩) ≤ ℓ(0) =⇒ ⟨w s , z i ⟩ ≥ 0 (cf. Lemma 22 in Section D). 4 GD keeps improving the logistic loss after the stable phase just because the logistic loss does not enjoy separation margin and never touches strict zero. We refer interested readers to the relevant discussion in Tyurin [58], who argues that the faster convergence in the stable phase is attributed to the choice of the logistic loss.
Second, the EoS termination condition L(w t ) ≲ 1/η suggests that the risk must be once O(1/T )optimal (with the optimally balancing choice η = Θ(T )) before benefitting from the super-fast rate O(1/T 2 ). Yet, GD under some loss functions including the Tsallis q-loss (q > 1) and the Rényi q-loss (q > 1) achieves better risk with the same GD steps. If our goal is simply to classify all training points, these alternative losses might do better jobs in terms of optimization solely.
Self-bounding property and implicit bias. Having said that, the phase transition may play an important role in implicit bias. It was shown under the linearly separable case that logistic regression optimized with GD enlarges the norm ∥w t ∥ toward the max-margin direction in rate Ω(ln(t)) [54,62,16]. Thus, we may argue that w t gradually comes to classify all data points correctly during the EoS phase and evolves toward the max-margin direction in the stable phase.
We reported how ∥w t ∥ evolves under the pilot setup in Figure 2 with different loss functions. As seen, the logistic and Tsallis 0.5 losses inflate ∥w t ∥ endlessly, which do not have separation margin. In
2 1 2 3 2 5 2 7 2 9 t 0 10 20 30 40 50 60 wt tsallis (q = 0.5) = 2 0 = 2 1 = 2 2 = 2 3 = 2 4 2 1 2 3 2 5 2 7 2 9 t 0 5 10 15 20 25 30 wt logistic = 2 0 = 2 1 = 2 2 = 2 3 = 2 4 2 1 2 3 2 5 2 7 2 9 t 0 5 10 15 20 wt tsallis (q = 2.0) = 2 0 = 2 1 = 2 2 = 2 3 = 2 4 stark contrast, the Tsallis 2-loss prevents ∥w t ∥ from growing endlessly just because of its separation margin-recall the norm upper bounds of the norm ∥w t ∥ in (11) and the growth rate ρ in Lemma 9. This raises two open questions: (1) Do we have similar implicit bias aligning toward the max-margin direction under a loss function with separation margin? (2) What are benefits and caveats of endless growing of ∥w t ∥? The latter is particularly relevant to the overconfidence issue due to excessively large ∥w t ∥ [61] and worse generalization due to prohibitively large within-class variance [26]. Wu et al. [64] argues that excessively large ∥w t ∥ leads to an inconsistent estimator.
The study on implicit bias for loss functions with the self-bounding property has been very scarce. To our knowledge, [35] crafted the complete hinge loss, which behaves like the hinge loss before GD converges to the zero risk yet incurs an extra penalty to artificially align the parameter toward the max-margin direction. Together with the benefits and caveats of the max-margin implicit bias, we believe this is an interesting open topic.
Dependency on n. Our main result (Theorem 5) provides the rate depending on the factor n. This extra factor with respect to n arises due to the worst-case analysis such that we have at least one "bad" direction z i before the convergence, corresponding to the inequality (9) in the proof of Theorem 5 (see Section B). This worst-case scenario supposes that all data points are nearly equidistant, which is unlikely since most data points tend to cluster in similar directions. We conjecture that this n-dependency is not essential with additional mild data assumptions.
The stochastic case. Our result can be extended for the stochastic gradient descent (SGD). Consider the scenario where we sample one fresh data point at each t and update the linear parameter with the loss function computed on this sample. Under the similar setup to Theorem 5, the population risk is ε-optimal with high probability after T = Ω(ε -(α+2) ). The formal statement and proof are shown in Section C. While this rate apparently looks significantly slower than the GD rate T = Ω(ε -α ), the extra iterations ε -2 is necessary for collecting sufficient samples to estimate the population risk. By noting that the GD/SGD updates consume n/one samples, the GD/SGD rates are comparable in terms of the number of consumed samples.
Finite-time convergence. Last but not least, we may potentially have another benefit of loss functions with separation margin. Take a look at Figure 1 again. Loss functions without separation margin, such as the Tsallis 1.5and 2-losses, converge to exact zero within finite time when sufficiently large stepsize is used. Such finite-time convergence under the linearly separable case can be shown without significant challenges if we use perceptron, or even the hinge loss, while becoming highly non-trivial in the case of twice-differentiable loss functions. This is because the perceptron argument requires a non-degenerate lower bound on ⟨w t , w * ⟩ (see (4)), which is not straightforward therein as the loss gradient can be arbitrarily small positive (due to the twice differentiability of the loss). We conjecture that an additional data assumption is necessary because the loss gradient could be adversarially vanishing against GD convergence, and leave this as future work.
this section cite: ['b62', 'b62', 'b57', 'b53', 'b61', 'b15', 'b10', 'b60', 'b25', 'b63', 'b34']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: In Section 1, we summarized our main result informally in Theorem 1.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The implicit bias structure for large-stepsize GD is nuanced. We carefully discussed this in Section 5.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: Our main set of assumptions are stated in Assumption 3. The complete proof corresponding to our main result is in Section B, and the proof sketch is in Section 3.1.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: All experimental details are described in the caption of Figure 1, despite that it is a synthetic simulation. Since the problem is convex by construction, the reproduction is straightforward.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [No] Justification: Our synthetic experiments reported in Figure 1 are not challenging to reproduce because the dataset and model are extremely small and the problem is convex.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: See the caption of Figure 1.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [No] Justification: We do not report the statistical significance for the synthetic simulation in Figure 1 because there is no randomness. All of the initialization, algorithm, and datasets are deterministic.
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
Answer: [No] Justification: Since the synthetic experiments in Figure 1 are extremely small-scale, we do not need a huge amount of computational resources to reproduce them. The experiments can be finished within a minute with a consumer laptop.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: This paper studies theoretical aspects of optimization, which hardly face such a challenge.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] Justification: This paper studies theoretical aspects of optimization, which hardly face such a challenge.
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
Answer: [NA]
Justification: This paper studies theoretical aspects of optimization, which hardly face such a challenge.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA] Justification: All datasets used in the simulation in Figure 1 are synthetic.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
this section cite: []

Section: New assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] Justification: All datasets used in the simulation in Figure 1 are synthetic.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: 14.
Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: All datasets used in the simulation in Figure 1 are synthetic. Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: This paper does not involve any human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: We did not use LLM in any aspects when conducting this research. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: A Technical lemmas
We introduce the gradient potential for a loss function in consideration as follows:
G(w) := 1 n n i=1 g(⟨w, z i ⟩).(6)
Lemma 7. Consider a loss ℓ satisfying Assumption 3B. Then, we have ℓ ρ(λ) ≤ ρ(λ) λ .
Proof. See [63,Lemma 20].
Lemma 8. Consider a Fenchel-Young loss ℓ(z) = ϕ * (-z) satisfying Assumption 3A. Then, ℓ and g are nonincreasing. Moreover, ℓ is strictly decreasing on (-∞, m)(⊆ R) if ℓ has separation margin m > 0; otherwise, ℓ is strictly decreasing on R.
Proof. We have by Danskin's theorem [21] (ϕ * ) ′ = (ϕ ′ ) -1 , and then
ℓ ′ (z) = -(ϕ * ) ′ (-z) = -(ϕ ′ ) -1 (-z) ∈dom(ϕ)⊆[0,1]
≤ 0, which implies that ℓ is nonincreasing. Since ℓ is convex and nonincreasing we have that g(•
) = |ℓ ′ (•)| = -ℓ ′ (•) is nonincreasing.
For the latter part, if nonincreasing ℓ has separation margin, ℓ(z) = 0 if and only if z ≥ m. Then, we have ℓ ≡ 0 on the interval [m, ∞) ⊆ R and ℓ > 0 on the interval (-∞, m) ⊆ R. On the latter interval, ℓ must be strictly decreasing because of its convexity. We can prove similarly for ℓ lacking separation margin.
this section cite: ['b62', 'b20']

Section: Lemma 9.
Consider a loss ℓ satisfying Assumption 3B. Suppose that ℓ has separation margin m > 0.
Then, we have ρ(λ) ≤ m 2 for any λ ≥ 1.
Proof. When ℓ has separation margin m (see Definition 3), we have
λℓ(z) + z 2 = z 2 for z ≥ m.
By the definition of ρ, we have
ρ(λ) = min z∈R λℓ(z) + z 2 ≤ min z≥m z 2 = m 2 .
Lemma 10 (Split optimization [63]). Suppose Assumption 2 and consider a convex and nonincreasing loss ℓ satisfying Assumption 3C and let u := u 1 + u 2 such that
u 1 = θw * , u 2 = ηC g 2γ w * .
For every t ≥ 1, we have
∥w t -u∥ 2 2ηt + 1 t t-1 k=0 L(w k ) ≤ ℓ(γθ) + 1 2ηt θ + ηC g 2γ 2 .
Proof. For k < t, we have
∥w k-1 -u∥ 2 = ∥w k -u∥ 2 + 2η ⟨∇L(w k ), u -w k ⟩ + η 2 ∥∇L(w k )∥ 2 = ∥w k -u∥ 2 + 2η ⟨∇L(w k ), u 1 -w k ⟩ + η(2 ⟨∇L(w k ), u 2 ⟩ + η∥∇L(w k )∥ 2 ).
For the second term, we have
⟨∇L(w k ), u 1 -w k ⟩ = 1 n n i=1 ℓ ′ (⟨w k , z i ⟩) ⟨z i , u 1 -w k ⟩ = 1 n n i=1 ℓ ′ (⟨w k , z i ⟩)(⟨u 1 , z i ⟩ -⟨w k , z i ⟩) ≤ 1 n n i=1 [ℓ(⟨u 1 , z i ⟩) -ℓ(⟨w k , z i ⟩)] (ℓ: convex) ≤ ℓ(γθ) -L(w k ). (ℓ: nonincreasing) (7
)
For the third term, we have
2 ⟨∇L(w k ), u 2 ⟩ + η∥∇L(w k )∥ 2 = 2 n n i=1 ℓ ′ (⟨w k , z i ⟩) ⟨z i , u 2 ⟩ + η 1 n n i=1 ℓ ′ (⟨w k , z i ⟩)z i 2 ≤ 2 n n i=1 ℓ ′ (⟨w k , z i ⟩) ⟨z i , u 2 ⟩ + η 1 n n i=1 ℓ ′ (⟨w k , z i ⟩) 2 (∥z i ∥ ≤ 1) ≤ 2∥u 2 ∥ n n i=1 ℓ ′ (⟨w k , z i ⟩) ⟨z i , w * ⟩ + ηC g • G(w k ) (Assumption 3C and G(w k ) ≥ 0) ≤ -2γ∥u 2 ∥G(w k ) + ηC g • G(w k ) (Assumption 2 and G(w k ) ≥ 0) = 0,
where the last equality is by the choice of u 2 and G(w) is defined in (6).
By combining them altogether, we have for k < t,
∥w k+1 -u∥ 2 ≤ ∥w k -u∥ 2 + 2η [ℓ(γθ) -L(w k )] .
Telescoping the sum from 0 to t -1 and rearranging, we get
∥w t -u∥ 2 2ηt + 1 t t-1 k=0 L(w k ) ≤ ℓ(γθ) + ∥w 0 -u∥ 2 2ηt ,
which completes the proof.
this section cite: ['b62', 'b5']

Section: B Proof of Theorem 5
Lemma 11. Suppose Assumption 2 and consider (GD) with any stepsize η > 0 under a Fenchel-Young loss ℓ that satisfies Assumption 3A. For t ≥ 1, assume G(w k ) ≥ G min > 0 for all k ∈ [0, t-1], where G(w) is defined in (6). Then, we have
γηG min t ≤ ⟨w t , w * ⟩ -⟨w 0 , w * ⟩ .
Proof. By the perceptron argument [44], we have
⟨w k+1 , w * ⟩ = ⟨w k , w * ⟩ -η ⟨∇L(w k ), w * ⟩ = ⟨w k , w * ⟩ - η n n i=1 ℓ ′ (⟨w k , z i ⟩)z i , w * = ⟨w k , w * ⟩ + η n n i=1 g(⟨w k , z i ⟩) ⟨w * , z i ⟩ (use g(•) = -ℓ ′ (•) by Lemma 8) ≥ ⟨w k , w * ⟩ + γηG(w k ) (note g(•) ≥ 0 and Assumption 2) ≥ ⟨w k , w * ⟩ + γηG min .
Telescoping the sum, we have the desired inequality.
this section cite: ['b43']

Section: Lemma 12 (Order evaluation).
Let I ⊆ R ≥0 be an open interval containing zero as the left end. For f : R ≥0 → R ≥0 ∪ {∞} that is nondecreasing and differentiable on I and satisfies f (0) = 0, let α := sup
x∈(0,x0] xf ′ (x) f (x)
for some x 0 ∈ I.
Then, for any x ∈ (0, x 0 ), we have f (x) ≥ Cx α , where C := f (x 0 )
x α 0 .
Proof. By the definition of α, we have
α ≥ xf ′ (x) f (x) for all x ∈ (0, x 0 ].
Then, for any x ∈ (0, x 0 ), we have
α ln x 0 x = α x0 x ds s ≥ x0 x f ′ (s) f (s) ds = ln f (x 0 ) f (x) .
By reorganizing this inequality, we can prove the original argument.
Lemma 13. Consider a Fenchel-Young loss ℓ(z) = ϕ * (-z) satisfying Assumption 3A. Then, for arbitrary 0 < ε < 1 and α defined in (3), we have
g(ℓ -1 (ε)) ≥ C ϕ ε α for ε ∈ (0, ε), where C ϕ := g(ℓ -1 (ε)) εα .
Proof. Choose any ε 0 > 0. For ε ∈ (0, ε 0 ), we can invert to get z ≡ ℓ -1 (ε) because ℓ is strictly decreasing on ℓ -1 ((0, 1)) (by Lemma 8) and hence invertible. Note that z ∈ (ℓ -1 (ε 0 ), m) if ℓ has separation margin m > 0, z ∈ (ℓ -1 (ε 0 ), ∞) otherwise, because ℓ -1 (•) is nonincreasing. We write this range as I, then z = ℓ -1 (ε) ∈ I for ε ∈ (0, ε 0 ).
Let us verify g(z) = -ℓ ′ (z) is differentiable at z ∈ I first. By the definition of Fenchel-Young losses, we have
ℓ(z) = ϕ * (-z) = sup x∈(0,1) [x • (-z) -ϕ(x)] = -z • (ϕ ′ ) -1 (-z) -ϕ (ϕ ′ ) -1 (-z) ,
for z ∈ I, where we used the first-order optimality -z = ϕ ′ (x) of the convex conjugate at the last identity. Since ϕ is twice continuously differentiable and ϕ ′′ > 0 on the interval (0, 1) by Assumption 3A, we can apply the inverse function theorem to have
ℓ ′ (z) = -(ϕ ′ ) -1 (-z) - z ϕ ′′ ((ϕ ′ ) -1 (-z)) - ϕ ′ (ϕ ′ ) -1 (-z) ϕ ′′ ((ϕ ′ ) -1 (-z)) = -(ϕ ′ ) -1 (-z),
for z ∈ I. Since ϕ is twice continuously differentiable, we can apply the inverse function theorem once again to get ℓ ′′ (z) for z ∈ I, and hence g is differentiable on I.
In addition, ℓ is continuously differentiable with non-degenerate derivative at z ∈ I because ℓ is strictly decreasing on I. From these observations, we can see that g(ℓ -1 (•)) =: f (•) is nondecreasing on I and differentiable on I (because it is the composition of two nondecreasing and differentiable functions g and ℓ -1 ). Now we can apply Lemma 12 to this f . Let us compute the exponent α ε defined in (8). By the differentiability and non-degenerate derivative of ℓ on I, we can apply the inverse function theorem on ℓ to have
εf ′ (ε) f (ε) = ε g(ℓ -1 (ε)) • g ′ (ℓ -1 (ε)) • 1 ℓ ′ (ℓ -1 (ε)) (inverse function theorem) = ℓ(z)g ′ (z) g(z)ℓ ′ (z) (ε ≡ ℓ(z)) = ℓ(z)ℓ ′′ (z) [ℓ ′ (z)] 2 (g(•) = -ℓ ′ (•)) = ϕ * (z) • (ϕ * ) ′′ (z) [(ϕ * ) ′ (z)] 2 (z := -z) (A) = [µϕ ′ (µ) -ϕ(µ)] • 1 ϕ ′′ (µ) µ 2 (µ ≡ (ϕ * ) ′ (z)) = ϕ ′ (µ) µϕ ′′ (µ) 1 - ϕ(µ) µϕ ′ (µ) ,
where at (A) we introduce µ as the dual of z by the mirror map ϕ ′ such that
z = ϕ ′ (µ) and µ = (ϕ * ) ′ (z), which implies ϕ * (z) = µϕ ′ (µ) -ϕ(µ)
together with the definition of the convex conjugate, and
[ϕ ′′ (µ)] • [(ϕ * ) ′′ (z)] = 1
with Danskin's theorem [21] and the inverse function theorem. Note that this identity is often referred to as Crouzeix's identity [19]. Here, we have
z ∈ -m, -ℓ -1 (ε 0 ) and µ ∈ 0, g(ℓ -1 (ε 0 )) if ℓ has separation margin m > 0, z ∈ -∞, -ℓ -1 (ε 0 ) and µ ∈ 0, g(ℓ -1 (ε 0 )) otherwise, by noting that g(z) = -ℓ ′ (z) = (ϕ * ) ′ (-z) is nonincreasing. With this primal-dual relationship, we have ϕ * (z) = µz -ϕ(µ) and [ϕ ′′ (µ)] • [(ϕ *
) ′′ (z)] = 1 by the definition of the convex conjugate and Crouzeix's identity. Now we are ready to apply Lemma 12, which yields
g(ℓ -1 (ε)) ≥ C ϕ ε α for ε ∈ (0, ε),where
α := sup µ∈(0,ε] ϕ ′ (µ) µϕ ′′ (µ) 1 - ϕ(µ) µϕ ′ (µ) , C ϕ := g(ℓ -1 (ε)) εα ,and
ε := g(ℓ -1 (ε 0 )).
Since the choice of ε 0 > 0 was arbitrary and Im(g) = Im((ϕ * ) ′ ) = dom(ϕ ′ ) ⊆ [0, 1], we can choose such ε ∈ (0, 1).
Proof of Theorem 5. For a fixed k ∈ [T -1], if we have L(w k ) > ε, there exists i ∈ [n] such that ℓ(⟨w k , z i ⟩) > ε. Then, we have for this specific i ∈ [n],
⟨w k , z i ⟩ < ℓ -1 (ε) (ℓ is strictly decreasing when ℓ > 0 by Lemma 8) =⇒ g(⟨w k , z i ⟩) ≥ g(ℓ -1 (ε)). (g is nonincreasing by Lemma 8) This implies that G(w k ) = 1 n j∈[n] g(⟨w k , z j ⟩) ≥ 1 n g(⟨w k , z i ⟩) ≥ 1 n g ℓ -1 (ε)(9)
holds while L(w k ) is ε-suboptimal, that is, L(w k ) > ε.
Next, fix w 0 = 0 and consider the case where L(w k ) > ε holds for all k ∈ [T -1]. By Lemma 8, we can use Lemma 19. By Lemmas 11 and 19, we can take G min = g(ℓ -1 (ε))/n (noting (9)) and have
γηg ℓ -1 (ε) T n ≤ ⟨w T , w * ⟩ -⟨w 0 , w * ⟩ (Lemma 11) = ⟨w T , w * ⟩ (with the choice of w 0 = 0) ≤ ∥w T ∥ (the Cauchy-Schwarz inequality with ∥w * ∥ = 1) ≤ 4 ρ(γ 2 ηT ) + ηC g γ . (Lemma 19)
By reorganizing and applying Lemma 13, we leverage the primal-dual relationship to have
T ≤ n γ 2 4 ρ(γ 2 ηT ) η + C g • 1 g (ℓ -1 (ε)) ≤ n γ 2 4 ρ(γ 2 ηT ) η + C g • 1 C ϕ ε α , where C ϕ is defined in Lemma 13. Therefore, L(w k ) is ε-suboptimal after at most n C ϕ γ 2 4 ρ(γ 2 ηt) η + C g ε -α (=: T (ε))
iterations. That is, if T > T (ε), the gradient lower bound (9) must be violated at some t ∈ [T ], and for this t, we achieve L(w t ) ≤ ε.
Finally, we verify that C ϕ defined in Lemma 13 matches (3). By introducing z as the dual of μ such that z = ϕ ′ (μ) and μ = (ϕ * ) ′ (z),
we have ε = ℓ(g -1 (μ)) (g is invertible when 0 < g(•) < 1) = ϕ * (z) (μ = (ϕ * ) ′ (z) = g(-z) implies g -1 (μ) = -z) = μϕ ′ (μ) -ϕ(μ),
where the invertibility of g can be verified through the differentiability of g as in Lemma 13 (by relying upon ϕ ′′ > 0 in Assumption 3A), and the last identity follows by the definition of the convex conjugate. Plugging this into C ϕ defined in Lemma 13, we see that it matches C ϕ in (3). Thus, we have proven all statements.
this section cite: ['b7', 'b20', 'b18']

Section: C Extension to the stochastic gradient descent
We discuss the extension of Theorem 5 to the stochastic setup. We consider the constant-stepsize online stochastic gradient descent (SGD) as follows: w t+1 := w t -η∇L t (w t ), where L t (w) := ℓ(⟨w, y t x t ⟩), t ≥ 0, (SGD) for a loss function ℓ : R → R ≥0 . Here, (x t , y t ) t≥0 are independently and identically distributed according to the following assumption. Assumption 4. Assume the training data (x t , y t ) t≥0 are independent copies of (x, y) following a distribution such that 1. ∥x∥ ≤ 1, and y i ∈ {±1}, almost surely; 2. there is γ > 0 and a unit vector w * such that ⟨w * , z t ⟩ ≥ γ for z t := y t x t , almost surely. Proposition 14. Suppose Assumption 4 and consider (SGD) with stepsize η > 0 and w 0 = 0 under a Fenchel-Young loss ℓ satisfying Assumption 3, and additionally having separation margin m > 0.
For arbitrary δ, ε ∈ (0, 1), define (α, C ϕ ) as in Eq. ( 3), for which we assume α, C ϕ ∈ (0, ∞). In addition, for arbitrary ε ∈ (0, ε), define
M η,γ := ℓ - 4m + ηC g γ and t • := max 32M 2 η,γ ln(1/δ) ε 2 , 8M η,γ ln(1/δ) ε .
Then, if we run (SGD) with T iterations such that
T > N • t • , where N := 2 α C ϕ γ 2 4m η + C g ε -α , then we have min t∈[T ] E[L t (w t )] ≤ ε with probability at least (1 -δ) N .
Before proving Proposition 14, several auxiliary lemmas are presented. Since the proof of Proposition 14 closely follows Theorem 5, the following Lemmas 15 and 16 are almost identical to the deterministic versions (Lemmas 11 and 19, respectively), and hence we omit the proofs.
Lemma 15. Suppose Assumption 4 and consider (SGD) with any stepsize η > 0 under a Fenchel-Young loss that satisfies Assumption 3A. Moreover, assume that there exists k ∈ [0, t -1] such that we have a non-trivial lower bound g(⟨w k , z k ⟩) ≥ g min > 0. Then, we have
γηg min ≤ ⟨w t , w * ⟩ -⟨w 0 , w * ⟩ .
Lemma 16. Suppose Assumption 4 and consider (SGD) with any stepsize η > 0 under a convex and nondecreasing loss ℓ satisfying Assumptions 3B and 3C. For every t ≥ 1, we have
∥w t ∥ ≤ 4 ρ(γ 2 ηt) + ηC g γ .
The following concentration result is an additional argument that we need in the stochastic case. Lemma 17. Consider (SGD) with any stepsize η > 0 under a Fenchel-Young loss that satisfies Assumptions 3A and 3C. Let Z 0 , . . . , Z t-1 be the independent copies of data z = yx following Assumption 4. We introduce the filtration {F t } t≥0 , where F k is a σ-algebra defined on Z 1 , . . . , Z k , and let H k := ℓ(⟨w k , Z k ⟩) be a random variable, where w k is an F k-1 -measurable random variable. We write S t := t-1 k=0 H k for a random variable standing for the accumulated loss, and let S t := S t /t. Moreover, we introduce the following assumptions:
• (Bounded mean) µ k := E[H k |F k-1 ] ≤ M for k = 0, . . . , t -1. • (Bounded variance) Var(H k |F k-1 ) ≤ σ 2 for k = 0, . . . , t -1.
Then, for any ε > 0 and δ ∈ (0, 1), if we have
E[S t ] t > ε and t ≥ max 32σ 2 ln(1/δ) ε 2 , 8M ln(1/δ) ε ,(10)
then we have S t > ε/2 with probability at least 1 -δ.
Proof. First, we apply a type of the martingale inequality, Freedman's inequality [22,Theorem 1.6].
Since E[H k -µ k |F k-1 ]
is the mean-zero martingale difference with the bounded variance σ 2 , Freedman's inequality is applicable. Then, we have
Pr {S t ≤ E[S t ] -u} ≤ exp - u 2 2(M u + tσ 2 )
.
Equivalently, we have the following inequality with probability at least 1 -δ:
S t > E[S t ] -M 2 ln 2 (1/δ) + 2tσ 2 ln(1/δ) -M ln(1/δ) > E[S t ] -2tσ 2 ln(1/δ) -2M ln(1/δ).
Dividing by t, we have
S t > E[S t ] t - 2σ 2 ln(1/δ) t - 2M ln(1/δ) t > ε - ε 4 - ε 4 = ε 2 ,
where we used the conditions (10) at the second inequality. Thus, the desired inequality is shown.
Now we are ready to prove Proposition 14. Overall, the proof consists of the perceptron argument and the concentration property.
Proof of Proposition 14. The first step is to establish the concentration property. To apply Lemma 17, we confirm the bounded moment conditions. For the mean
E[H k |F k-1 ] = E[ℓ(⟨w k , Z k ⟩)|F k-1 ], where F k-1 is the σ-algebra defined on {Z l } k-1 l=1 , we have E[ℓ(⟨w k , Z k ⟩)|F k-1 ] ≤ E[ℓ(-∥w k ∥)|F k-1 ] (ℓ is nonincreasing and ∥Z k ∥ ≤ 1) ≤ ℓ -
4m + ηC g γ (Lemmas 9 and 15) =: M η,γ .
For the variance Var(H k |F k-1 ), we similarly have
Var(H k |F k-1 ) = E[H 2 k |F k-1 ] -E[H k |F k-1 ] 2 ≤ E[H 2 k |F k-1 ] ≤ ℓ - 4m + ηC g γ 2 = M 2 η,γ .
Note that these moment bounds hold uniformly for any k. By plugging M = M η,γ and σ 2 = M 2 η,γ into Lemma 17, if we have
1 t • t0+t • -1 k=t0 E [L k (w k )] > ε and t • ≥ max 32M 2 η,γ ln(1/δ) ε 2 , 8M η,γ ln(1/δ) ε ,(11)
then we have 1 t
• t0+t • -1 k=t0 L k (w k ) > ε 2 with probability at least 1 -δ.
Let us use this concentration argument. Split the interval [0, T ] into length-t • sub-intervals (for t • satisfying (11)) such that
[0, T ] = [0, t • -1] =:I1 ⊔ [t • , 2t • -1] =:I2 ⊔ [2t • , 3t • -1] =:I3 • • • ⊔ [(N -1)t • , N t • -1] =:I N ⊔[N t • , T ]. (12
)
Consider the scenario where E[L k (w k )] > ε holds for all k ∈ [0, T ], and focus on an arbitrary sub-interval I l . Since we have 1
t • k∈I l E[L k (w k )] > ε, the concentration argument implies that 1 t • k∈I l L k (w k ) > ε/2
with probability at least 1 -δ. This further indicates the high-probability existence of k l ∈ I l such that L k l (w k l ) > ε/2. In this case, we have g(⟨w k l , z k l ⟩) ≥ g(ℓ -1 (ε/2)) for this specific k l ∈ I l , which can be seen in the same manner as the proof of Theorem 5. Since this concentration argument does not depend on the sub-interval choice I l , there exists a set of indices {k l } l∈[N ] such that each of g(⟨w k l , z k l ⟩) ≥ g(ℓ -1 (ε/2)) holds with probability at least 1 -δ.
Next, we invoke the perceptron argument. By combining Lemmas 15 and 16 with the choice g min = g(ℓ -1 (ε/2)), we have
γηg ℓ -1 ε 2 • N ≤ ⟨w T , w * ⟩ ≤ ∥w T ∥ ≤ 4 ρ(γ 2 ηT ) + ηC g γ ≤ 4m + ηC g γ ,
with probability at least (1 -δ) N , where we additionally used Lemma 9 at the last inequality. By applying Lemma 13 at the left-most side, we have
γηC ϕ • ε 2 α • N ≤ 4m + ηC g γ , which implies N ≤ 2 α C ϕ γ 2 4m η + C g ε -α (=: N (ε)).
Therefore, if N > N (ε) (or T > N (ε) • t • ), with probability at least (1 -δ) N , we have E[L k (w k )] ≤ ε for some k ∈ [0, T ].
Finally, we need verify that C ϕ defined in Lemma 13 matches (3), but we skip it because it can be confirmed in the same manner as in the proof of Theorem 5.
this section cite: ['b21', 'b10']

Section: C.1 Comparison between GD and SGD
Whereas the iteration complexity for GD given by Corollary 6 is T = Ω(ε -α ), the iteration complexity for SGD given by Proposition 14 is T = Ω(ε -(α+2) ). Hence, the SGD rate is significantly slower than the GD case. This deterioration is because we can observe a non-trivial lower bound g(⟨w k , z k ⟩)
after every t • = Ω(ε -2 ) steps. As in the GD case, we need N = Ω(ε -α ) such non-trivial lower bounds, and hence the total iteration number amounts to N t • = Ω(ε -(α+2) ). While this apparently looks a big bottleneck, note that the GD rate is T ≳ nε -α if we explicitly write the n-dependency. Since (SGD) consumes only one fresh sample at every update (while (GD) consumes n samples), the extra complexity N = Ω(ε -2 ) appearing in the SGD case compensates for this gap of sample sizes. The GD/SGD rates are comparable in this sense.
this section cite: []

Section: D Phase transition of large-stepsize GD
We recap Wu et al. [63], who shows the existence of the phase transition from the EoS to stable phases. Assumption 5. Consider a loss ℓ ∈ C 1 (R) that is convex, nonincreasing, and ℓ(+∞) = 0.
A. Self-bounding property. For some C β > 0, g(•) ≤ C β ℓ(•) and 2 for z and x such that |z -x| < 1.
ℓ(z) ≤ ℓ(x) + ℓ ′ (x)(z -x) + C β g(x)(z -x)
this section cite: ['b62']

Section: B. Exponential tail.
There is a constant C e > 0 such that ℓ(z) ≤ C e g(z) for z ≥ 0.
this section cite: []

Section: Theorem 18 ([63]
). Consider (GD) with stepsize η > 0 and initialization w 0 = 0 under a loss ℓ satisfying Assumptions 3B and 3C, and 5A. Let T be the maximum number of steps. Then, we have the following:
• The EoS phase. For every t > 0, we have
1 t t-1 k=0 L(w k ) ≤ [6 ρ(γ 2 ηt) + ηC g ] 2 8γ 2 ηt .
• The stable phase. If s < T is such that
L(w s ) ≤ min 1 4C 2 β η , ℓ(0) n ,(13)
then (GD) is in the stable phase, that is, (L(w t )) T t=s decreases monotonically, and moreover, L(w t ) ≤ 5 ρ(γ 2 η(t -s)) γ 2 η(t -s) , t ∈ (s, T ].
• Phase transition time. There exists a constant C 1 > 0 that only depends on C g , C β , and ℓ(0) such that the following holds. Let
τ := 1 γ 2 max ψ -1 (C 1 (η + n)) η , C 1 (η + n)η , where ψ(λ) := λ ρ(λ)
.
If τ ≤ T , (13) holds for some s ≤ τ . Moreover, if ℓ additionally satisfies Assumption 5B and η ≥ 1, there exists C 2 > 0 that depends on C e , C g , C β , ℓ(0), and n such that τ is improved as follows:
τ := C 2 γ 2 max {η, n} .
The proof consists of Lemma 19 (the EoS phase), Lemma 23 (the stable phase), and Lemmas 24 and 25 (phase transition time), respectively. Most of the results in this section have already been provided in Wu et al. [63]. We restate the statements and proofs here to make the paper self-contained, and moreover, simplify the statements from the NTK setup to the linear-model case to highlight the essential structures.
this section cite: ['b62']

Section: Lemma 19 (EoS phase).
Suppose Assumption 2 and consider a convex and nonincreasing loss ℓ satisfying Assumptions 3B and 3C. For every t ≥ 1, we have
1 t t-1 k=0 L(w k ) ≤ [6 ρ(γ 2 ηt) + ηC g ] 2 8γ 2 ηt ,and
∥w t ∥ ≤ 4 ρ(γ 2 ηt) + ηC g γ .
Proof. By invoking Lemma 10 with the choice of θ θ = ρ(γ 2 ηt) γ , we have
∥w t -u∥ 2 2ηt + 1 t t-1 k=0 L(w k ) ≤ ℓ(γθ) + 1 2ηt θ + ηC g 2γ 2 ≤ ρ(γ 2 ηt) γ 2 ηt + 1 2ηt θ + ηC g 2γ 2 , (Lemma 7) which implies that ∥w t ∥ ≤ ∥w t -u∥ + ∥u∥ ≤ 2ρ(γ 2 ηt) γ 2 + θ + ηC g 2γ 2 + θ + ηC g 2γ ≤ 2ρ(γ 2 ηt) γ + 2 θ + ηC g 2γ ( √ a + b ≤ √ a + √ b) ≤ 4 ρ(γ 2 ηt) + ηC g γ ,and
1 t t-1 k=0 L(w k ) ≤ ρ(γ 2 ηt) γ 2 ηt + 1 2ηt θ + ηC g 2γ 2 = ρ(γ 2 ηt) γ 2 ηt + (2 ρ(γ 2 ηt) + ηC g ) 2 8γ 2 ηt ≤ [2(1 + √ 2) ρ(γ 2 ηt) + ηC g ] 2 8γ 2 ηt (a 2 + b 2 ≤ (a + b) 2 for a, b ≥ 0) ≤ [6 ρ(γ 2 ηt) + ηC g ] 2 8γ 2 ηt .
Thus, the proof is completed.
Lemma 20. Suppose Assumption 2 and consider a convex and nonincreasing loss ℓ satisfying Assumptions 3B and 3C. Then, we have
1 t t-1 k=0 G(w k ) ≤ 4 ρ(γ 2 ηt) + ηC g γ 2 ηt , t ≤ T,
where G(w) is defined in (6).
Proof. By the perceptron argument [44], we have
⟨w t+1 , w * ⟩ = ⟨w t , w * ⟩ -η ⟨∇L(w t ), w * ⟩ = ⟨w t , w * ⟩ - η n n i=1 ℓ ′ (⟨w t , z i ⟩) ⟨w * , z i ⟩ ≥ ⟨w t , w * ⟩ - γη n n i=1 ℓ ′ (⟨w t , z i ⟩) (Assumption 2 and note -ℓ ′ (•) ≥ 0) = ⟨w t , w * ⟩ -γηG(w t ).
Telescoping the sum, we have
1 t t-1 k=0 G(w k ) ≤ ⟨w t , w * ⟩ -⟨w 0 , w * ⟩ γηt ≤ ∥w t ∥ γηt ≤ 4 ρ(γ 2 ηt) + ηC g γ 2 ηt ,
where the last inequality is due to the parameter bound in Lemma 19.
Lemma 21 (Modified descent lemma). Consider a loss satisfying 5A. Suppose there exists s < T such that L(w s ) ≤ 1 4C 2 β η , then for every t ∈ [s, T ] we have 1. L(w t ) ≤ 1/(4C 2 β η) and G(w t ) ≤ 1/(4C β η), 2. L(w t+1 ) ≤ L(w t ) -3η 4 ∥∇L(w t )∥ 2 ≤ L(w t ), where G(w) is defined in (6).
Proof. We first show that Claim 1 implies Claim 2. By Assumption 5A, we have
ℓ(⟨w t+1 , z i ⟩) ≤ ℓ(⟨w t , z i ⟩) + ℓ ′ (⟨w t , z i ⟩) ⟨w t+1 -w t , z i ⟩ + C β g(⟨w t , z i ⟩) ⟨w t+1 -w t , z i ⟩ 2 ≤ ℓ(⟨w t , z i ⟩) + ℓ ′ (⟨w t , z i ⟩) ⟨w t+1 -w t , z i ⟩ + C β g(⟨w t , z i ⟩)∥w t+1 -w t ∥ 2 .
Taking average over i ∈ [n], we get
L(w t+1 ) ≤ L(w t ) + ⟨∇L(w t ), w t+1 -w t ⟩ + C β G(w t )∥w t+1 -w t ∥ 2 = L(w t ) -η∥∇L(w t )∥ 2 + C β η 2 G(w t )∥∇L(w t )∥ 2 ≤ L(w t ) -η∥∇L(w t )∥ 2 + η 4 ∥∇L(w t )∥ 2 (Claim 1) = L(w t ) - 3η 4 ∥∇L(w t )∥ 2 ,
which verifies Claim 2.
Next, we prove Claim 1 by induction. The base case t = s holds by Assumption 5A as follows:
G(w s ) = 1 n n i=1 g(⟨w s , z i ⟩) ≤ 1 n n i=1 C β ℓ(⟨w s , z i ⟩) = C β L(w s ) ≤ 1 4C β η .(14)
To prove the step case, we suppose L(w k ) ≤ 1/(4C 2 β η) and G(w k ) ≤ 1/(4C β η) for k = s, s + 1, . . . , t and prove them for k = t + 1. Since Claim 1 implies Claim 2, we have
L(w t+1 ) ≤ L(w t ) ≤ • • • ≤ L(w s ) ≤ 1 4C 2 β η .
Since G(w t+1 ) ≤ C β L(w t+1 ) holds as in ( 14), we have
G(w t+1 ) ≤ C β L(w t+1 ) ≤ 1 4C β η .
Thus, the step case is shown, and all claims are proven.
this section cite: ['b5', 'b43']

Section: Lemma 22.
Consider a nonincreasing and nonnegative loss ℓ. For every w such that L(w) ≤ ℓ(0) n , we have y i ⟨w,
x i ⟩ ≥ 0 for i ∈ [n].
Proof. See [63,Lemma 31].
this section cite: ['b62']

Section: Lemma 23 (Stable phase).
Consider a nonincreasing loss ℓ satisfying Assumptions 5A and 3B. Suppose there exists s < T such that
L(w s ) ≤ min 1 4C 2 β η , ℓ(0) n .
Then, for every t ∈ [0, T -s], we have L(w s+t ) ≤ 5 ρ(γ 2 ηt) γ 2 ηt .
Proof. By the lemma assumption, we can apply Lemma 21 for s onwards. Therefore, we have for k ≥ 0,
η∥∇L(w s+k )∥ 2 ≤ 4 3 [L(w s+k ) -L(w s+k+1 )] ≤ 4 3 L(w s+k ).(15)
Choose a comparator centered at w s , u := w s + θw * , θ := ρ(η 2 γt) γ .
For k ≤ t -1, we have
∥w s+k+1 -u∥ 2 = ∥w s+k -u∥ 2 + 2η ⟨∇L(w s+k ), u -w s+k ⟩ + η 2 ∥∇L(w s+k )∥ 2 ≤ ∥w s+k -u∥ 2 + 2η ⟨∇L(w s+k ), u -w s+k ⟩ + 4 3 ηL(w s+k ).
(by ( 15))
Following the same derivation of (7), we can bound the second term as follows:
⟨∇L(w s+k ), u -w s+k ⟩ ≤ 1 n n i=1 ℓ(θγ + ⟨w s , z i ⟩) -L(w s+k ).
The assumption L(w s ) ≤ ℓ(0)/n allows us to apply Lemma 22, so ⟨w s , z i ⟩ ≥ 0 and thus
ℓ(θγ + ⟨w s , z i ⟩) ≤ ℓ(θγ) = ℓ( ρ(γ 2 ηt)),
where ℓ is nonincreasing due to the lemma assumption. Consequently, we can control the second term by ⟨∇L(w s+k ), uw s+k ⟩ ≤ ℓ( ρ(γ 2 ηt)) -L(w s+k ).
Plugging this back, we get
∥w s+k+1 -u∥ 2 ≤ ∥w s+k -u∥ 2 + 2η[ℓ( ρ(γ 2 ηt)) -L(w s+k )] + 4 3 ηL(w s+k ) ≤ ∥w s+k -u∥ 2 + 2ηℓ( ρ(γ 2 ηt)) - 2 3 ηL(w s+k ).
Telescoping the sum from 0 to t -1 and rearranging, we get
3∥w s+t -u∥ 2 2ηt + 1 t t-1 k=0 L(w s+k ) ≤ 3ℓ( ρ(γ 2 ηt)) + 3∥w s -u∥ 2 2ηt ≤ 3 ρ(γ 2 ηt) γ 2 ηt + 3∥w s -u∥ 2 2ηt . (Lemma 7)
Finally, we can show the claims as follows:
1 t t-1 k=0 L(w s+k ) ≤ 3 ρ(γ 2 ηt) γ 2 ηt + 3∥w s -u∥ 2 2ηt = 9 2 ρ(γ 2 ηt) γ 2 ηt ≤ 5 ρ(γ 2 ηt) γ 2 ηt .
By Lemma 21, L(w t ) is nonincreasing for t ≥ s, and thus we have
L(w s+t ) ≤ 1 t t-1 k=0 L(w s+k ) ≤ 5 ρ(γ 2 ηt) γ 2 ηt .
Lemma 24 (Phase transition). Suppose Assumption 2 and consider a convex and nonincreasing loss ℓ that satisfies Assumptions 3B and 3C. Define
ψ(λ) = λ ρ(λ)
, λ > 0.
Then, there is C > 0 depending on C g , C β , and ℓ(0) such that the following holds. Let
τ := 1 γ 2 max ψ -1 (C(η + n)) η , C(η + n)η .
If τ < T , then there exists s ∈ [0, τ ] such that
L(w s ) ≤ min 1 4C 2 β η , ℓ(0) n .
Proof. Applying Lemma 19 with t = τ , we have
1 τ τ -1 k=0 L(w k ) ≤ [6 ρ(γ 2 ητ ) + ηC g ] 2 8γ 2 ητ = 3 √ 2 ρ(γ 2 ητ ) γ 2 ητ + √ 2 4 ηC g γ 2 ητ 2 .
Choose τ such that
γ 2 ητ ≥ max ψ -1 18[4C 2 β η + n/ℓ(0)] , 1 2 (ηC g ) 2 (4C 2 β η + n/ℓ(0)) . It is clear that 1 ψ(λ) = ρ(λ) λ = min z ℓ(z) + z 2 λ
is a decreasing function. Then, we have
3 √ 2 ρ(γ 2 ητ ) γ 2 ητ = 3 √ 2 1 ψ(γ 2 ητ ) ≤ 3 √ 2 1 18[4C 2 β η + n/ℓ(0)] = 1 2 1 4C 2 β η + n/ℓ(0) and √ 2 4 ηC g γ 2 ητ ≤ 1 2 1 4C 2 β η + n/ℓ(0) .
These two inequalities together imply that
1 τ τ -1 k=0 L(w k ) ≤ 1 4C 2 β η + n/ℓ(0) ≤ min 1 4C 2 β η , ℓ(0) n ,
which implies that there exists s ≤ τ for L(w s ) satisfies the right-hand side bound.
Lemma 25 (Phase transition time under exponential tail). Suppose Assumption 2 and consider a nonincreasing loss ℓ satisfying Assumptions 3B and 3C, and 5B. Furthermore, assume η ≥ 1. Then, there exists C > 0 depending on C e , C g , C β , ℓ(0), and n such that the following holds. Let τ := C γ 2 max {η, n ln n} .
If τ ≤ T , then there exists s ∈ [0, τ ] such that
L(w s ) ≤ min 1 4C 2 β η , ℓ(0) n .
Proof. Under Assumption 5B, we have ℓ(z) ≤ C e g(z) = -C e ℓ ′ (z), for z ≥ 0, which implies ℓ ′ (z) ℓ(z) ≤ -C -1 e , for z ≥ 0.
Integrating both sides, we get
ln ℓ(z) ≤ ln ℓ(0) + z 0 ℓ ′ (ζ) ℓ(ζ) dζ ≤ ln ℓ(0) -C -1 e z, for z ≥ 0,
which implies ℓ(z) ≤ ℓ(0) exp(-C -1 e z), for z ≥ 0. Using the exponential tail property, we have
ρ(λ) = min z∈R λℓ(z) + z 2 ≤ λℓ(C e ln(λ)) + C 2 e ln 2 (λ) ≤ ℓ(0) + C 2 e ln 2 (λ).
Applying Lemma 20, we have
1 τ τ -1 k=0 G(w k ) ≤ 4 ρ(γ 2 ητ ) + ηC g γ 2 ητ ≤ 4 ℓ(0) + C 2 e ln 2 (γ 2 ητ ) + ηC g γ 2 ητ ≤ 4 ℓ(0) + 4C e ln(γ 2 ητ ) + ηC g γ 2 ητ ( √ a + b ≤ √ a + √ b) ≤ 4C e η ln(γ 2 τ ) γ 2 τ + C g + 4C e γ 2 τ + 4 ℓ(0) η 1 γ 2 τ .
Here, we take C > 0 depending on C e , C g , C β , ℓ(0), and additionally n such that γ 2 τ ≥ C max {η, n} and
ln C C ≤ min 1 4CeC 2 β , ℓ(0) Ce 4C e (1 + ln n) + C g + 4C e + 4 ℓ(0) .
This choice is possible with sufficiently large C ≥ e because (ln C)/C is strictly decreasing in C ≥ e toward zero. Such C enables us to have
1 τ τ -1 k=0 G(w k ) ≤ 1 C max {η, n} 4C e η (ln C + ln max {η, n}) + C g + 4C e + 4 ℓ(0) η ≤ 4C e (ln C + ln n) + C g + 4C e + 4 ℓ(0) C max {η, n} (η ≥ 1) ≤ ln C C 4C e (1 + ln n) + C g + 4C e + 4 ℓ(0) max {η, n} (ln C ≥ 1) ≤ min 1 4CeC 2 β , ℓ(0) Ce max {η, n} ≤ min 1 4C e C 2 β η , ℓ(0) C e n .
From this we have some s ≤ τ such that G(w s ) ≤ min 1 4C e C 2 β η , ℓ(0) C e n .
This ensures that for every i ∈ [n],
1 n g(⟨w s , z i ⟩) ≤ G(w s ) = 1 n n i=1 g(⟨w s , z i ⟩) ≤ ℓ(0) C e n ≤ g(0) n ,
where the last inequality is due to Assumption 5B. The above implies ⟨w s , z i ⟩ ≥ 0 since g(•) is nonincreasing. Thus, we can apply Assumption 5B for ⟨w s , z i ⟩ and get
ℓ(⟨w s , z i ⟩) ≤ C e g(⟨w s , z i ⟩).
Taking an average over i ∈ We complete the proof by plugging in the upper bound on G(w s ).
this section cite: []

Section: E Separation margin and self-bounding property
In this section, we discuss the relationship between separation margin and the self-bounding property. First, we show that a loss function does not have separation margin if it satisfies the self-bounding property. Proposition 26. Consider a loss ℓ : R → R that is continuously differentiable and nonincreasing, and satisfies ℓ(z 0 ) > 0 for some z 0 ∈ R. If ℓ satisfies Assumption 5A, then ℓ does not have separation margin.
Proof. Choose any ε ∈ (0, 1/C β ). The convexity of ℓ implies that g(z) = -ℓ ′ (z) ≥ ℓ(z) -ℓ(z + ε) ε for any z ∈ R.
By the self-bounding property (Assumption 5A), we further have
ℓ(z) -ℓ(z + ε) ε ≤ g(z) ≤ C β ℓ(z).
Solving this, we have ℓ(z + ε) ≥ (1 -C β ε)ℓ(z). Thus, if ℓ(z) > 0 holds, we additionally have ℓ(z + ε) > 0 for ε ∈ (0, 1/C β ), and we conclude that ℓ cannot have separation margin because ℓ > 0 holds on the entire R.
Next, we argue that the converse of Proposition 26 does not hold, that is, even if a loss ℓ does not have separation margin, it does not always imply that ℓ satisfies the self-bounding property. A counterexample is a Fenchel-Young loss generated by the following potential function:
ϕ(µ) = µ 0 Φ -1 (p)dp, where Φ is the standard normal CDF Φ(x) := 1 2 1 + erf x √ 2
and erf is the error function. The generated Fenchel-Young loss is relevant to the probit model [40] because ϕ ′ is nothing else but the probit link function prevailing in generalized linear models. Hence, we call the generated Fenchel-Young loss the probit Fenchel-Young loss for convenience. We can have a concise form of the probit Fenchel-Young loss:
ℓ(z) = ϕ * (-z) = -z -∞ (ϕ ′ ) -1 (ζ)dζ = -z -∞ Φ(ζ)dζ = min µ∈[0,1]
λ[µϕ ′ (µ) -ϕ(µ)] + [ϕ ′ (µ)] 2 , where we use the definition of the convex conjugate ϕ * (z) = µz -ϕ(µ) at the last identity. Now, we write the objective as R(µ): R(µ) := λ[µϕ ′ (µ) -ϕ(µ)] + [ϕ ′ (µ)] 2 .
Differentiating R, we have
R ′ (µ ⋆ ) = [λµ ⋆ + 2ϕ ′ (µ ⋆ )]ϕ ′′ (µ ⋆ ) = 0 ϕ ′′ >0 =⇒ ϕ ′ (µ ⋆ ) = - λ 2 µ ⋆
at the minimizer µ ⋆ of R. Plugging this back to R, we have
ρ(λ) = R(µ ⋆ ) = λ µ ⋆ - λ 2 µ ⋆ -ϕ(µ ⋆ ) + - λ 2 µ ⋆ 2 = -λϕ(µ ⋆ ) ≤ -ϕ 1 2 λ,
where the last inequality owes to that a convex potential satisfying Assumption 1 is minimized at the uniform distribution µ ⋆ = 1/2 [11,Proposition 4].
By using Lemma 27, we can simplify the convergence rate of (GD) given by Theorem 5 for a loss that does not have separation margin. Note that the following convergence rate is not sufficiently tight due to overestimation of ρ by Lemma 27; nevertheless, the provided convergence rate is convenient when we do not have an access to ρ analytically.
this section cite: ['b39', 'b10']

Section: Corollary 28.
Under the same setup with Theorem 5, we additionally assume that ℓ does not have separation margin. If (α, C ϕ ) with (3) satisfies α, C ϕ ∈ (0, ∞) and
T > 2C g n C ϕ γ 2 ε -α + 16[-ϕ(1/2)]n 2 C 2 ϕ γ 2 η ε -2α for ε ∈ (0, ε),
then we have L(w T ) ≤ ε.
Proof. Combining Theorem 5 and Lemma 27, we have the following convergence rate:
T > 4n -ϕ(1/2)ε -α C ϕ γ √ η √ T + C g nε -α C ϕ γ 2 .
Defining a := 4n -ϕ(1/2)ε -α C ϕ γ √ η and b := C g nε -α C ϕ γ 2 , we have the following inequality in T :
T 2 -(a 2 + 2b)T + b 2 > 0.
This can be solved for T ≥ 1:
T > a 2 + 2b 2 1 + 1 - 2b a 2 + 2b 2 ≤1
, for which T > a 2 + 2b is sufficient. Thus, we have shown the statement.
Throughout this section, we repeatedly use L'Hôpital's rule. When it is used, we notate by ( ‡).
this section cite: []

Section: F.1 Shannon entropy
For the Shannon entropy ϕ(µ) = µ ln µ + (1 -µ) ln(1 -µ), we have ϕ ′ (µ) = ln µ -ln(1 -µ) and ϕ ′′ (µ) = 1 µ + 1 1 -µ , which imply
lim µ↓0 ϕ ′ (µ) µϕ ′′ (µ) 1 - ϕ(µ) µϕ ′ (µ) = lim µ↓0 ln µ 1-µ 1 -µ 1-µ 1 - µ ln µ + (1 -µ) ln(1 -µ) µ ln µ -µ ln(1 -µ) = lim µ↓0 ln µ 1 -µ • µ ln µ -µ ln(1 -µ) -µ ln µ -(1 -µ) ln(1 -µ) µ ln µ 1-µ = lim µ↓0 -ln(1 -µ) µ ( ‡) = lim µ↓0 1 1 -µ = 1,and
lim µ↓0 µ µϕ ′ (µ) -ϕ(µ) = lim µ↓0 µ µ ln µ -µ ln(1 -µ) -µ ln µ -(1 -µ) ln(1 -µ) = lim µ↓0 µ -ln(1 -µ) • 1 2µ -1 = lim µ↓0 µ ln(1 -µ) ( ‡) = lim µ↓0 (1 -µ) = 1.
Finally, we derive the convergence rate for the logistic loss. Plugging α = 1, C ϕ = 1, C g = 1, and ρ(λ) ≤ 1 + ln 2 (λ) ≤ 2 ln 2 (λ) to Theorem 5, we have
T > n γ 2 4 √ 2 ln(γ 2 ηT ) η + 1 ε -1 = 4 √ 2 ln(γ 2 η) η + 1 + 4 √ 2 η ln T nε -1 γ 2 . Dividing both ends by ln T , we have T ln T > 4 √ 2 ln(γ 2 η) η + 1 1 ln T + 4 √ 2 η nε -1 γ 2 , for which the following is sufficient when T ≥ 2: T ln T > 4 √ 2 ln(γ 2 η) η + 1 1 ln 2 + 4 √ 2 η nε -1 γ 2 = 4 √ 2 log 2 (γ 2 η) η + 1 ln 2 + 4 √ 2 η nε -1 γ 2 . By ignoring the logarithmic factor, we have T ≳ 4 √ 2 log 2 (γ 2 η) η + 1 ln 2 + 4 √ 2 η nε -1 γ 2 .
this section cite: []

Section: F.2 Semi-circle entropy
For the semi-circle entropy ϕ(µ) = -2 µ(1 -µ), we first derive the analytical form of the corresponding Fenchel-Young loss. We have ϕ ′ (µ) = 2µ -1 µ(1 -µ) and ϕ ′′ (µ) = 1 2[µ(1 -µ)] 3/2 .
The dual transform (ϕ * ) ′ is given by
(ϕ * ) ′ (z) = (ϕ ′ ) -1 (z) = 1 2   z 2 z 2 2 + 1 + 1   ,
thanks to the Danskin's theorem [21]. Then, we can derive the Fenchel-Young loss by the definition of the convex conjugate:
ℓ(z) = ϕ * (-z) = -z(ϕ * ) ′ (z) -ϕ ((ϕ * ) ′ (-z)) = -z + √ z 2 + 4 2 .
Next, we compute the loss parameters α and C ϕ respectively as follows: Finally, we derive the convergence rate for the semi-circle loss. Plugging α = 2, C ϕ = 1, C g = 1, and ρ(λ) ≤ 5λ/(2 ln λ) to Theorem 5, we have
lim µ↓0 ϕ ′ (µ) µϕ ′′ (µ) 1 - ϕ(µ) µϕ ′ (µ) = lim µ↓0 2µ-1 √ µ(1-µ) µ 2[µ(1-µ)] 3/2   1 + 2 µ(1 -µ) µ(2µ-1) √ µ(1-µ)    = lim µ↓0 2(2µ -1)(1 -µ) 1 + 2(1 -µ) 2µ -1 = 2,
T > n γ 2   4 5 2 γ 2 ηT ln(γ 2 ηT ) η + 1   ε -2 = 2 √ 10 η γ 2 ηT ln(γ 2 ηT ) + 1 nε -2 γ 2 ,
for which the following is sufficient when T ≥ 2:
T > 2 √ 10 η γ 2 ηT ln(2γ 2 η) + 1 nε -2 γ 2 .
Subsequently, we follow the same flow as in the proof of Corollary 28. Defining
a := 2 √ 10nε -2 γ 2 η γ 2 η ln(2γ 2 η)
and b := nε -2 γ 2 , we have the following inequality in T :
T 2 -(a 2 + 2b)T + b 2 > 0.
This can be solved for T ≥ 1:
T > a 2 + 2b 2 1 + 1 - 2b a 2 + 2b 2 ≤1 , for which T > a 2 + 2b is sufficient, namely, T > 40n 2 γ 2 η ln(2γ 2 η) ε -4 + 2n γ 2 ε -2
is sufficient. Thus, the convergence rate is T = Ω(ε -4 ).
this section cite: ['b20']

Section: F.3 Tsallis entropy
For the Tsallis entropy
ϕ(µ) = µ q + (1 -µ) q -1 q -1 , define ϕ 0 (µ) = µ q + (1 -µ) q -1, ϕ 1 (µ) = µ q-1 -(1 -µ) q-1 , ϕ 2 (µ) = µ q-2 + (1 -µ) q-2 .
When 0 < q < 2 (q ̸ = 1),
lim µ↓0 ϕ ′ (µ) µϕ ′′ (µ) 1 - ϕ(µ) µϕ ′ (µ) = 1 q(q -1) lim µ↓0 1 ϕ 2 (µ) • qµϕ 1 (µ) -ϕ 0 (µ) µ 2 = 1 q(q -1) lim µ↓0 1 1 + µ 1-µ 2-q • qµϕ 1 (µ) -ϕ 0 (µ) µ q = 1 q(q -1) lim µ↓0 qµϕ 1 (µ) -ϕ 0 (µ) µ q ( ‡) = 1 q(q -1) lim µ↓0 qϕ 1 (µ) + q(q -1)µϕ 2 (µ) -qϕ 1 (µ) qµ q-1 = 1 q lim µ↓0 µ q-1 + (1 -µ) q-2 µ µ q-1 ( ‡) = 1 q lim µ↓0 (q -1)µ q-2 + (1 -µ) q-2 -(q -2)(1 -µ) q-3 µ (q -1)µ q-2 = 1 q lim µ↓0 1 + 1 q -1 µ 1 -µ 2-q -(q -2) µ 1 -µ 3-q = 1 q .
this section cite: []

Section: F.4 Rényi entropy
For the Rényi entropy
ϕ(µ) = 1 q -1 ln [µ q + (1 -µ) q ] ,define
ϕ 0 (µ) = µ q + (1 -µ) q , ϕ 1 (µ) = µ q-1 -(1 -µ) q-1 , ϕ 2 (µ) = µ q-2 + (1 -µ) q-2 , ϕ 3 (µ) = µ q-3 -(1 -µ) q-3 .
When 0 < q < 2 with q ̸ = 1,
lim µ↓0 ϕ ′ (µ) µϕ ′′ (µ) 1 - ϕ(µ) µϕ ′ (µ) = lim µ↓0 ϕ1(µ) ϕ0(µ) (q -1)µ ϕ2(µ) ϕ0(µ) -qµ ϕ1(µ) 2 ϕ0(µ) 2   1 - 1 q-1 ln ϕ 0 (µ) q q-1 µ ϕ1(µ) ϕ0(µ)   = lim µ↓0 1 (q -1) µϕ2(µ) ϕ1(µ) -q µϕ1(µ) ϕ0(µ) • 1 - ϕ 0 (µ) ln ϕ 0 (µ) qµϕ 1 (µ) = 1 (q -1) lim µ↓0 µϕ2(µ) ϕ1(µ) -q lim µ↓0 µϕ1(µ) ϕ0(µ) • 1 - lim µ↓0 ϕ 0 (µ) q • lim µ↓0 ln ϕ 0 (µ) µϕ 1 (µ) = 1 (q -1) • 1 -q • 0 • 1 - 1 q • 1 = 1 q ,
where we use
ϕ 0 (µ) → 1, µϕ 2 (µ) ϕ 1 (µ) = 1 + µ 1-µ 2-q 1 -µ 1-µ 2-q → 1, µϕ 1 (µ) = µ q - µ (1 -µ) 1-q → 0,and
ln ϕ 0 (µ) µϕ 1 (µ) ( ‡) → 1 ϕ 0 (µ) • ϕ ′ 0 (µ) µϕ ′ 1 (µ) + ϕ 1 (µ) → ϕ ′ 0 (µ) µϕ ′ 1 (µ) + ϕ 1 (µ) = qϕ 1 (µ) (q -1)µϕ 2 (µ) + ϕ 1 (µ) = q (q -1) µϕ2(µ) ϕ1(µ) + 1 → q (q -1) • 1 + 1 = 1.
In addition, we have
lim µ↓0 µ [µϕ ′ (µ) -ϕ(µ)] 1/q = lim µ↓0 µ q µϕ ′ (µ) -ϕ(µ) 1/q
this section cite: []

Section: 
= [ζΦ(ζ) + Φ ′ (ζ)] -z -∞ = -zΦ(-z) + Φ ′ (-z).
The probit Fenchel-Young loss does not have separation margin because ϕ ′ (µ) = Φ -1 (µ) → -∞ as µ ↓ 0 (see Proposition 4). However, it does not satisfy the self-bounding property. To see this, we have g(z) ℓ(z) = -ℓ ′ (z) ℓ(z) = --Φ(-z) -zΦ(-z) + Φ ′ (-z) = Φ(z) zΦ(z) + Φ ′ (z) , (z ≡ -z) which implies
lim z→∞ g(z) ℓ(z) = lim z→-∞ Φ ′ (z) Φ(z) + zΦ ′ (z) + Φ ′′ (z) (L'Hôpital's rule) = lim z→-∞ Φ ′ (z) Φ(z) = lim z→-∞ Φ ′′ (z) Φ ′ (z) (L'Hôpital's rule) = lim z→-∞ -zΦ ′ (z) Φ ′ (z) = ∞.
Hence, g(z) cannot always be bounded from above by ℓ(z), that is, the self-bounding property is not satisfied.
this section cite: []

Section: F Omitted calculation for examples
Here, we compute for each ϕ,
lim µ↓0 ϕ ′ (µ) µϕ ′′ (µ)
1 -ϕ(µ) µϕ ′ (µ) to estimate the power α of the convergence rate provided in Theorem 5, by making the error parameter ε > 0 in (3) arbitrarily small. Correspondingly, we compute
lim µ↓0 µ [µϕ ′ (µ) -ϕ(µ)] α
to estimate the constant C ϕ in the convergence rate, verifying that C ϕ neither degenerates nor diverges for arbitrarily small error parameter ε > 0. Before proceeding with each example, we provide a rough estimate of ρ for loss functions without separation margin. Lemma 27. Consider a loss ℓ satisfying Assumptions 3A and 3B that does not have separation margin. Then,
ρ(λ) ≤ -ϕ 1 2 λ.
Proof. First, we rewrite ρ as a dual form. By introducing the dual variable µ of z by z = ϕ ′ (µ) and µ = (ϕ * ) ′ (z), we have
ρ(λ) = min z∈R λℓ(z) + z 2 = min z∈R λϕ * (z) + z 2
In addition, we have
lim µ↓0 µ [µϕ ′ (µ) -ϕ(µ)] 1/q = lim µ↓0 (q -1) 1/q µ [qµϕ 1 (µ) -ϕ 0 (µ)] 1/q = (q -1) 1/q lim µ↓0 qµϕ 1 (µ) -ϕ 0 (µ) µ q -1/q = (q -1) 1/q q -1 -lim µ↓0 qµ(1 -µ) q-1 + (1 -µ) q -1 µ q -1/q ( ‡) = (q -1) 1/q q -1 -lim µ↓0 q(1 -µ) q-1 -q(q -1)µ(1 -µ) q-2 -q(1 -µ) q-1 qµ q-1 -1/q = (q -1) 1/q q -1 -(q -1) lim µ↓0 µ 1 -µ 2-q -1/q = (q -1) 1/q • (q -1 + 0) -1/q = 1. When q ≥ 2, lim µ↓0 ϕ ′ (µ) µϕ ′′ (µ) 1 - ϕ(µ) µϕ ′ (µ) = 1 q(q -1) lim µ↓0 1 ϕ 2 (µ) • qµϕ 1 (µ) -ϕ 0 (µ) µ 2 = 1 q(q -1) lim µ↓0 qµϕ 1 (µ) -ϕ 0 (µ) µ 2 = 1 q(q -1) lim µ↓0 q[µ q -(1 -µ) q-1 µ] -µ q -(1 -µ) q µ 2 ( ‡) = 1 q(q -1) lim µ↓0 q[qµ q-1 -(1 -µ) q-1 + (q -1)(1 -µ) q-2 µ] -qµ q-1 + q(1 -µ) q-1 2µ = lim µ↓0 µ q-2 + (1 -µ) q-2 2 = 1 2 .
In addition, we have
lim µ↓0 µ [µϕ ′ (µ) -ϕ(µ)] 1/2 = (q -1) 1/2 lim µ↓0 qµϕ 1 (µ) -ϕ 0 (µ) µ 2 -1/2 ( ‡) = (q -1) 1/2 lim µ↓0 qϕ 1 (µ) + q(q -1)µϕ 2 (µ) -qϕ 1 (µ) 2µ -1/2 = (q -1) 1/2 q(q -1) 2 lim µ↓0 [µ q-2 + (1 -µ) q-2 ] -1/2 = (q -1) 1/2 • q(q -1) 2 -1/2 = 2 q . = lim µ↓0 (q -1)µ q ϕ 0 (µ) qµϕ 1 (µ) -ϕ 0 (µ) ln ϕ 0 (µ) 1/q = lim µ↓0 (q -1)µ q qµϕ 1 (µ) -ϕ 0 (µ) ln ϕ 0 (µ) 1/q (ϕ 0 (µ) → 1) ( ‡) = (q -1) lim µ↓0 qµ q-1 qϕ 1 (µ) + q(q -1)µϕ 2 (µ) -qϕ 1 (µ) ln ϕ 0 (µ) -qϕ 1 (µ) 1/q = (q -1) lim µ↓0 µ q-1 (q -1)µϕ 2 (µ) -ϕ 1 (µ) ln ϕ 0 (µ) 1/q ( ‡) =    (q -1) lim µ↓0 (q -1)µ q-2 (q -1)ϕ 2 (µ) + (q -1)(q -2)µϕ 3 (µ) -qϕ1(µ) 2 ϕ0(µ) -(q -1)ϕ 2 (µ) ln ϕ 0 (µ)    1/q = lim µ↓0 1 + µ 1-µ 2-q +(q -2) 1 -µ 1-µ 3-q - qϕ1(µ) 2 (q-1)µ q-2 ϕ0(µ) -1 + µ 1-µ 2-q ln ϕ 0 (µ) q -1 -1 q (A) = 1 + (q -2) • 1 -0 -1 • 0 q -1 -1/q = 1,
where at (A) we used
ϕ 1 (µ) 2 µ q-2 ϕ 0 (µ) → µ 2-q ϕ 1 (µ) 2 = µ q -2(1 -µ) q-1 µ + (1 -µ) 2q-2 µ 2-q → -2(1 -µ) q-1 µ + (1 -µ) 2q-2 µ 2-q = (1 -µ) 2q-2 -2(1 -µ) q-1 µ q-1 µ q-2 ( ‡) → (2q -2)(1 -µ) 2q-3 + 2(q -1)(1 -µ) q-2 µ q-1 -2(q -1)(1 -µ) q-1 µ q-2 (q -2)µ q-3 = 2(q -1) q -2 • 1 (1 -µ) 2-q • (1 -µ) q-1 + µ q-1 -(1 -µ)µ q-2 µ q-3 → 2(q -1) q -2 • 1 • (1 -µ) q-1 + µ q-1 -(1 -µ)µ q-2 µ q-3 = 2(q -1) q -2 • µ 1 -µ 1-q µ 2 + µ 2 -(1 -µ)µ → 0.
When q = 2, we leverage
ϕ 0 (µ) = 2µ 2 -2µ + 1, ϕ 1 (µ) = 2µ -1, ϕ 2 (µ) = 2, ϕ ′ 0 (µ) = 2ϕ 1 (µ), ϕ ′ 1 (µ) = 2 to have lim µ↓0 ϕ ′ (µ) µϕ ′′ (µ) 1 - ϕ(µ) µϕ ′ (µ) = lim µ↓0 ϕ 0 (µ) • 2µϕ 1 (µ) -ϕ 0 (µ) ln ϕ 0 (µ) 4µ 2 [ϕ 0 (µ) -ϕ 1 (µ) 2 ] = lim µ↓0 2µϕ 1 (µ) -ϕ 0 (µ) ln ϕ 0 (µ) 4µ 2 [ϕ 0 (µ) -ϕ 1 (µ) 2 ] (ϕ 0 (µ) → 1) ( ‡) = lim µ↓0 2[ϕ 1 (µ) + 2µ] -2ϕ 1 (µ) ln ϕ 0 (µ) -2ϕ 1 (µ) 4{2µ[ϕ 0 (µ) -ϕ 1 (µ) 2 ] + µ 2 [2ϕ 1 (µ) -4ϕ 1 (µ)]} = lim µ↓0 2µ -ϕ 1 (µ) ln ϕ 0 (µ) 4µ[ϕ 0 (µ) -ϕ 1 (µ) 2 -µϕ 1 (µ)] ( ‡) = lim µ↓0 2 -2 ln ϕ 0 (µ) -2ϕ1(µ) 2 ϕ0(µ) 4[ϕ 0 (µ) -ϕ 1 (µ) 2 -µϕ 1 (µ)] + 4µ[2ϕ 1 (µ) -4ϕ 1 (µ) -ϕ 1 (µ) -2µ] = lim µ↓0 1 -ln ϕ 0 (µ) -ϕ1(µ) 2 ϕ0(µ) 2[ϕ 0 (µ) -ϕ 1 (µ) 2 -4µϕ 1 (µ) -2µ 2 ] ( ‡) = lim µ↓0 -2ϕ1(µ) ϕ0(µ) -4ϕ0(µ)ϕ1(µ)-2ϕ1(µ) 3 ϕ0(µ) 2 2[2ϕ 1 (µ) -4ϕ 1 (µ) -4ϕ 1 (µ) -8µ -4µ] = lim µ↓0 ϕ 1 (µ) ϕ 0 (µ) 2 • 3ϕ 0 (µ) -ϕ 1 (µ) 2 6[ϕ 1 (µ) + 2µ] = 1 3 .
In addition, defining
ζ := 2µϕ 1 (µ) -ϕ 0 (µ) ln ϕ 0 (µ) ϕ 0 (µ) , we have (ξ :=) lim µ↓0 µ [µϕ ′ (µ) -ϕ(µ)] 1/3 = lim µ↓0 µ 2µϕ1(µ)-ϕ0(µ) ln ϕ0(µ) ϕ0(µ)1/3
implies ξ = lim µ↓0 µζ -1/3 ; we will use this below at ($)
( ‡) = lim µ↓0 3ζ 2/3 2ϕ1(µ) ϕ0(µ) + 2µ 2ϕ0(µ)-2ϕ1(µ) 2 ϕ0(µ) 2 -2ϕ1(µ) ϕ0(µ) = lim µ↓0 3ϕ 0 (µ) 2 4 ζ 2/3 µ[ϕ 0 (µ) -ϕ 1 (µ) 2 ] ( ‡) = lim µ↓0 3 4 2 3 • 2µ 2ϕ0(µ)-2ϕ1(µ) 2 ϕ0(µ) 2 {ϕ 0 (µ) -ϕ 1 (µ) 2 + µ[2ϕ 1 (µ) -4ϕ 1 (µ)]} ζ 1/3 = lim µ↓0 2µ[ϕ 0 (µ) -ϕ 1 (µ) 2 ] [ϕ 0 (µ) -ϕ 1 (µ) 2 -2µϕ 1 (µ)]ζ 1/3 ( ‡) = lim µ↓0 3ζ 2/3 2 1 µϕ 0 (µ)[ϕ 0 (µ) -ϕ 1 (µ) 2 ] -3 ϕ0(µ) 3 [ϕ1(µ)+µ]ζ ϕ0(µ)-ϕ1(µ) 2 -2µϕ1(µ) = lim µ↓0 3ζ 2/3 2 1 µ[ϕ 0 (µ) -ϕ 1 (µ) 2 ] + 3ζ ϕ0(µ)-ϕ1(µ) 2 -2µϕ1(µ) = lim µ↓0 2 3 ϕ 0 (µ) -ϕ 1 (µ) 2 µ • (µζ -1/3 ) 2 + lim µ↓0 2µ ϕ 0 (µ) -ϕ 1 (µ) 2 -2µϕ 1 (µ) 1 µζ -1/3 -1 ($) = 2 3 ξ 2 lim µ↓0 (2 -2µ) + 1 ξ lim µ↓0 1 2 -3µ -1 = 4 3 ξ 2 + 1 2ξ -1 , which implies ξ = 1 4 3 ξ 2 + 1 2ξ .
By solving this, we have
this section cite: []

Section: F.5 Pseudo-spherical entropy
Consider the q-pseudo-spherical entropy ϕ(µ) = [µ q + (1 -µ) q ] 1/q -1 for q > 1 [24]. It is also known as the q-norm (neg)entropy [12]. When q = 2, it recovers the spherical entropy associated with the spherical loss [1]. When q ↑ ∞, it approaches ϕ ∞ (µ) = max {µ, 1 -µ} -1, which is the Bayes risk of the hinge/0-1 losses [15]. As seen in Figure 3, the limit α (in (3)) does not exist, which indicates that we cannot guarantee the ε-optimal risk for vanishingly small ε.
this section cite: ['b23', 'b11', 'b0', 'b14']

Section: References
Ref_id:b0 Title: Surrogate regret bounds for bipartite ranking via strongly proper losses Year: (2014)
Ref_id:b1 Title: Understanding the unstable convergence of gradient descent Year: (2022)
Ref_id:b2 Title: Learning threshold neurons via edge of stability Year: (2023)
Ref_id:b3 Title: Information Geometry and Its Applications Year: (2016)
Ref_id:b4 Title: Understanding gradient descent on the edge of stability in deep learning Year: (2022)
Ref_id:b5 Title: Proper losses, moduli of convexity, and surrogate regret bounds Year: (2023)
Ref_id:b6 Title: Fenchel-Young losses with skewed entropies for classposterior probability estimation Year: (2021)
Ref_id:b7 Title: Proper losses regret at least 1/2-order Year: (2024)
Ref_id:b8 Title: Convexity, classification, and risk bounds Year: (2006)
Ref_id:b9 Title: Learning with differentiable pertubed optimizers Year: (2020)
Ref_id:b10 Title: Learning with Fenchel-Young losses Year: (2020)
Ref_id:b11 Title:  Year: (1980)
Ref_id:b12 Title: Uniformly convex functions on Banach spaces Year: (2009)
Ref_id:b13 Title: Collision entropy and optimal uncertainty Year: (2012)
Ref_id:b14 Title: Loss functions for binary class probability estimation and classification: Structure and applications Year: (2005)
Ref_id:b15 Title: Large stepsize gradient descent for non-homogeneous two-layer networks: Margin improvement and fast optimization Year: (2024)
Ref_id:b16 Title: Beyond the edge of stability via two-step gradient updates Year: (2023)
Ref_id:b17 Title: Gradient descent on neural networks typically occurs at the edge of stability Year: (2021)
Ref_id:b18 Title: A relationship between the second derivatives of a convex function and of its conjugate Year: (1977)
Ref_id:b19 Title: Self-stabilization: The implicit bias of gradient descent at the edge of stability Year: (2023)
Ref_id:b20 Title: The theory of max-min, with applications Year: (1966)
Ref_id:b21 Title: On tail probabilities for martingales Year: (1975)
Ref_id:b22 Title: Surrogate regret bounds for polyhedral losses Year: (2021)
Ref_id:b23 Title: Strictly proper scoring rules, prediction, and estimation Year: (2007)
Ref_id:b24 Title: Regret bounds for prediction problems Year: (1999)
Ref_id:b25 Title: A closer look at prototype classifier for few-shot image classification Year: (2022)
Ref_id:b26 Title: Lower and upper estimations of the modulus of convexity in some Orlicz spaces Year: (1991)
Ref_id:b27 Title: Hierarchy of deformations in concavity Year: (2022)
Ref_id:b28 Title: The implicit bias of gradient descent on nonseparable data Year: (2019)
Ref_id:b29 Title: Characterizing the implicit bias via a primal-dual analysis Year: (2021)
Ref_id:b30 Title: Gradient descent follows the regularization path for general losses Year: (2020)
Ref_id:b31 Title: Stochasticity of deterministic gradient descent: Large learning rate for multiscale objective function Year: (2020)
Ref_id:b32 Title: Gradient descent monotonically decreases the sharpness of gradient flow solutions in scalar networks and beyond Year: (2023)
Ref_id:b33 Title: The large learning rate phase of deep learning: The catapult mechanism Year: (2020)
Ref_id:b34 Title: Completion of hinge loss has an implicit bias Year: (2020)
Ref_id:b35 Title: Understanding the generalization benefit of normalization layers: Sharpness reduction Year: (2022)
Ref_id:b36 Title: Beyond the quadratic approximation: The multiscale structure of neural network loss landscapes Year: (2022)
Ref_id:b37 Title: Cross-entropy loss functions: Theoretical analysis and applications Year: (2023)
Ref_id:b38 Title: From softmax to sparsemax: A sparse model of attention and multi-label classification Year: (2016)
Ref_id:b39 Title: Generalized Linear Models Year: (1989)
Ref_id:b40 Title: Gradient descent on logistic regression with non-separable data and large step sizes Year: (2024)
Ref_id:b41 Title: Lectures on Convex Optimization Year: (2018)
Ref_id:b42 Title: SparseMAP: Differentiable sparse structured inference Year: (2018)
Ref_id:b43 Title: On convergence proofs for perceptrons Year: (1962)
Ref_id:b44 Title: Sparse sequence-to-sequence models Year: (2019)
Ref_id:b45 Title: The implicit bias of gradient descent on separable multiclass data Year: (2024)
Ref_id:b46 Title: Composite binary losses Year: (2010)
Ref_id:b47 Title: On measures of information and entropy Year: (1961)
Ref_id:b48 Title: Convex Analysis Year: (1970)
Ref_id:b49 Title: Loss functions and operators generated by f -divergences Year: (2025)
Ref_id:b50 Title: Online structured prediction with Fenchel-Young losses and improved surrogate regret for online multiclass classification with logistic loss Year: (2024)
Ref_id:b51 Title: Revisiting online learning approach to inverse linear optimization: A Fenchel-Young loss perspective and gap-dependent regret analysis Year: (2025)
Ref_id:b52 Title: Interpolation and extrapolation of linear operators in Orlicz spaces Year: (1964)
Ref_id:b53 Title: The implicit bias of gradient descent on separable data Year: (2018)
Ref_id:b54 Title: Possible generalization of Boltzmann-Gibbs statistics Year: (1988)
Ref_id:b55 Title: Flavors of margin: Implicit bias of steepest descent in homogeneous neural networks Year: (2025)
Ref_id:b56 Title: Normalized flat minima: Exploring scale invariant definition of flat minima for neural networks using PAC-Bayesian analysis Year: (2020)
Ref_id:b57 Title: From logistic regression to the perceptron algorithm: Exploring gradient descent with large step sizes Year: (2025)
Ref_id:b58 Title: Large learning rate tames homogeneity: Convergence and balancing effect Year: (2022)
Ref_id:b59 Title: Unified binary and multiclass margin-based classification Year: (2024)
Ref_id:b60 Title: Mitigating neural network overconfidence with logit normalization Year: (2022)
Ref_id:b61 Title: Implicit bias of gradient descent for logistic regression at the edge of stability Year: (2023)
Ref_id:b62 Title: Large stepsize gradient descent for logistic loss: Non-monotonicity of the loss improves optimization efficiency Year: (2024)
Ref_id:b63 Title: Benefits of early stopping in gradient descent for overparameterized logistic regression Year: (2025)
Ref_id:b64 Title: A walk with SGD Year: (2018)
Ref_id:b65 Title: Hessian-based analysis of large batch training and robustness to adversaries Year: (2018)
Ref_id:b66 Title: Solving large scale linear prediction problems using stochastic gradient descent algorithms Year: (2004)
Ref_id:b67 Title: Understanding edge-ofstability training dynamics with a minimalist example Year: (2023)
