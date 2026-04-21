Title: LoRA Training Provably Converges to a Low-Rank Global Minimum or It Fails Loudly (But it Probably Won't Fail)
Abstract: Low-rank adaptation (LoRA) has become a standard approach for fine-tuning large foundation models. However, our theoretical understanding of LoRA remains limited as prior analyses of LoRA's training dynamics either rely on linearization arguments or consider highly simplified setups. In this work, we analyze the LoRA loss landscape without such restrictive assumptions. We define two regimes: a "special regime", which includes idealized setups where linearization arguments hold, and a "generic regime" representing more realistic setups where linearization arguments do not hold. In the generic regime, we show that LoRA training converges to a global minimizer with low rank and small magnitude, or a qualitatively distinct solution with high rank and large magnitude. Finally, we argue that the zeroinitialization and weight decay in LoRA training induce an implicit bias toward the low-rank, smallmagnitude region of the parameter space-where global minima lie-thus shedding light on why LoRA training usually succeeds in finding global minima.

Section: Introduction
With the recent explosive trend of scale, fine-tuning a pretrained foundational model to target downstream tasks has become a dominant approach to deep learning. Low-rank adaptation (LoRA) (Hu et al., 2022) is a parameter-efficient fine-tuning method freezing the pre-trained weight matrix W 0 ∈ R m×n , and training a low-rank update X = AB ⊺ to it using
W = W 0 + X = W 0 + AB ⊺ , 0 rank ≤ r⋆ r⋆ < rank < r rank = r X⋆ Xspurious Xspurious Xspurious
Figure 1: In LoRA fine-tuning, under the assumption that the global minimum X ⋆ has low rank and small magnitude, we show that spurious local minima X spurious may exist, but they have high rank and large magnitude.
where r ≪ min(m, n), A ∈ R m×r and B ∈ R n×r . The low-rank factor matrices A and B are respectively initialized as a random Gaussian matrix and a zero matrix, leading to X = 0 at initialization. By training fewer parameters, LoRA fine-tuning significantly reduces memory usage, making fine-tuning feasible on GPUs with limited GPU memory.
The broad use of LoRA has spurred theoretical works aimed at understanding its effectiveness. One line of work focuses on analyzing LoRA's training dynamics, exploring why optimizers like SGD or Adam successfully find effective low-rank updates despite the significant non-convexity introduced by the factorization X = AB ⊺ , as well as the inherent non-convexity of neural networks, by utilizing some degree of linearization. Specifically, Malladi et al. (2023) studies LoRA under a complete linearization, effectively holding A fixed during fine-tuning and viewing the training as a convex optimization problem. A subsequent work (Jang et al., 2024) presents a more refined analysis linearizing with respect to the product X = AB ⊺ , retaining the non-convexity arising from the interaction between A and B.
Beyond linearization, Dayi & Chen (2024) analyzes a twolayer teacher-student setup for rank-1 LoRA. In this work, we carry out a theoretical analysis without any linearizations and any restriction on layers or LoRA rank.
this section cite: ['b15', 'b25', 'b17', 'b6']

Section: Contribution.
We analyze the loss landscape of LoRA fine-tuning and show that in the "generic regime", a more practical setup where linearization arguments do not hold, a local minimizer is either (i) a global minimizer with small rank and small magnitude or (ii) a spurious local minimizer with high rank and large magnitude. We further argue that the zero-initialization and weight decay in LoRA training induce an implicit bias toward the low-rank, small-magnitude region of the parameter space, where global minima lie. Altogether, we shed light on why practical LoRA training effectively converges to global minima.
Our key assumptions, formally defined and justified in Section 2, are the existence of a low-rank global minimizer for full fine-tuning, restricted strong convexity, and restricted smoothness. Notably, our analysis does not rely on any linearization arguments, making it more applicable to practical fine-tuning setups compared to prior work.
this section cite: []

Section: Prior works
PEFT methods and LoRA Parameter-Efficient Finetuning (PEFT) methods have emerged as effective approaches for fine-tuning large language models on downstream tasks while reducing computational and storage requirements. Among numerous proposed methods (Ben Zaken et al., 2022;Li & Liang, 2021;Lester et al., 2021), Low-Rank Adaptation (LoRA) (Hu et al., 2022) has become a predominant approach by decomposing weight updates into low-rank matrices. Several variants such as LoRA+ (Hayou et al., 2024), rsLoRA (Kalajdzievski, 2023), PiSSA (Meng et al., 2024), and MiLoRA (Wang et al., 2025) have been built upon the LoRA framework, addressing the discrepancy with full fine-tuning in optimization and performance.
Theoretical foundation of LoRA. Existing theoretical works on LoRA focus on the expressive power and the training dynamics of LoRA. Zeng & Lee (2024) demonstrates that a certain LoRA rank suffices to express a given fine-tuning function. Jang et al. (2024) proves that under the NTK regime, LoRA with rank Ω( √ N ) can express the global minimizer of the original model. Malladi et al. (2023) argues that the LoRA fine-tuning dynamics are nearly equivalent to the kernel regression. Under this framework, Jang et al. (2024) proves LoRA fine tuning loss has no spurious local minima when the rank is O( √ N ). Beyond the kernel regime, Dayi & Chen (2024) analyzes a two-layer teacherstudent setup for LoRA and explains why SGD leads to convergence to a global minimum in this context. Zhang et al. (2025) also identifies the training dynamics in a 2-layer setup, proving LoRA will align to a singular subspace of one-step gradient of full fine-tuning.
Low-rank optimization. The low-rank optimization problem min X∈R m×n , rank(X)≤r f (X) has been extensively studied in the optimization literature, including matrix sensing (Recht et al., 2010) and matrix completion (Candès & Recht, 2012). Rather than directly optimizing over the space of low-rank matrices, it is often preferred to employ the Burer-Monteiro factorization (Burer & Monteiro, 2003), which formulates the problem by parameterizing X as X = U V ⊺ , U ∈ R m×r , V ∈ R n×r .
As the Burer-Monteiro factorization introduces nonconvexity, a large body of work has identified conditions under which this approach avoids spurious local minima (Bhojanapalli et al., 2016;Ge et al., 2017;Park et al., 2017;Zhang, 2021). Further studies extend these results to general settings (Ha et al., 2020;Zhang, 2024). In our work, we utilize the framework established in these studies with novel techniques to extend its boundary to optimization guarantees in LoRA training. Neural network. Let f (• ; •) : P × X → R K be a neural network where P is the parameter space, X is the data space, and R K is the output space. Assume the model is pre-trained to Θ 0 ∈ P, i.e., the pre-trained model is f (Θ 0 ; •).
this section cite: ['b1', 'b24', 'b22', 'b15', 'b14', 'b18', 'b26', 'b34', 'b38', 'b17', 'b25', 'b17', 'b6', 'b41', 'b30', 'b5', 'b3', 'b2', 'b12', 'b29', 'b39', 'b13', 'b40']

Section: Notation and preliminaries
Matrix notation. For X ∈ R m×n , denote its singular values as σ 1 (X) ≥ σ 2 (X) ≥ • • • ≥ σ r (X) ≥ 0. For matri- ces A and B, let ∥A∥ 2 = σ 1 (A) denote the spectral norm, ∥A∥ * = σ i (A) the nuclear norm, ∥A∥ F = σ i (A) 2 the
Fine-tuning loss. Let W 0 = (W (1) 0 , . . . , W (L) 0 ) ⊂ Θ 0 be the pre-trained value of the weights W that we choose to fine-tune. We wish to fine-tune the pre-trained model f (Θ 0 ; •) on a downstream task with data distribution (x, y) ∼ D. With slight abuse of notation, write f (W ; •) to denote f (Θ ; •), where all parameters of Θ excluding W are fixed to their corresponding values in Θ 0 . Let
X = (X (1) , . . . , X (L) )
be the change of W during the (full) fine-tuning. The true objective one hopes to minimize is
L full (X) = E (x,y)∼D [ℓ(f (W 0 + X; x), y)]
with some loss function ℓ(•, •). We assume ℓ(x, y) is nonnegative and twice-differentiable with respect to x for any y.
In practice, we have access to a finite dataset {(x i , y i )} N i=1 , so we minimize the empirical risk
L full (X) = 1 N N i=1 ℓ(f (W 0 + X; x i ), y i ).
LoRA. Low-rank adaptation (LoRA) uses a rank-r parameterization for each update matrix
X (l) = A (l) (B (l) ) ⊺ ∈ R m l ×n l with A (l) ∈ R m l ×r and B (l) ∈ R n l ×r for l = 1, . . . , L. Denote A = (A (1) , . . . , A (L) ), B = (B (1) , . . . , B (L) )
and
AB ⊺ = A (1) (B (1) ) ⊺ , . . . , A (L) (B (L) ) ⊺ .
Under this parametrization, we define the empirical LoRA risk as
L lora (A, B) ≜ L full (AB ⊺ )
We adopt the standard initialization (Hu et al., 2022), respectively initializing each A and B as a random gaussian and zero, leading to AB ⊺ = 0 at initialization.
Second-order stationary points. Let L : R n → R be twice-continuously differentiable. We say X ∈ R n is a (first-order) stationary point if
∇L(X) = 0. We say X ∈ R n is a second-order stationary point (SOSP) if ∇L(X) = 0, ∇ 2 L(X)[U, U ] ≥ 0,
for any U ∈ R n . Lastly, we say X ∈ R n is a local minimum if there exists an open ball B that contains X and
L(X) ≤ L(X ′ )
for any X ′ ∈ B. It follows that a local minimum is an SOSP. If a local minimum is not a global minimum, we say it is a spurious local minimum.
Prior works have established that stochastic gradient descent applied to twice-continuously differentiable functions (regardless of convexity) roughly converges to SOSPs.
Theorem (Theorem 4.1 of Lee et al. (2016)). Gradient descent on twice-differentiable functions with random initialization, almost surely, does not converge to strict saddle points. I.e.,if gradient descent converges, it converges to an SOSP, almost surely.
Theorem (Informal, Theorem 1 of Ge et al. (2015)).
Stochastic gradient descent with noise on twicedifferentiable strict saddle functions (i.e., every stationary point is either a local minimum or a strict saddle) does not converge to strict saddle points with high probability. I.e., if stochastic gradient descent with noise converges, it converges to an SOSP with high probability.
In the context of our work, the implication is that LoRA training converges to SOSPs. The question we address is whether such SOSPs are global minima or whether it is possible to converge to a bad local minimum.
this section cite: ['b15', 'b21', 'b11']

Section: Weight decay and nuclear norm regularization
Let λ ≥ 0 and
L lora λ (A, B) ≜ L lora (A, B) + λ 2 ∥A∥ 2 F + ∥B∥ 2 F .
Practical LoRA training typically employs weight decay (Hu et al., 2022;Dettmers et al., 2023) and applying SGD with weight decay on L lora is equivalent to minimizing L lora λ without weight decay. In other words, the effect of weight decay is equivalent to adding ℓ 2 -regaularization. Let
L full λ (X) ≜ L full (X) + λ ∥X∥ ⋆ .
From the prior literature on low-rank matrix sensing and Burer-Monterio factorizations (Recht et al., 2010, Lemma 5.1), it is known that minimizing this ℓ 2 -regularized problem in A and B is mathematically equivalent to minimizing the nuclear-norm regularized loss in the product X = AB ⊺ subject to a rank constraint. In other words minimize
A, B L lora λ (A, B) ⇔ minimize X L full λ (X) subject to rank(X) ≤ r
When using LoRA, we hope to match the performance of full fine-tuning. We expect this to be feasible if the full finetuning problem (with nuclear norm regularization) admits a global minimizer whose rank is at most r, since the LoRA update AB ⊺ cannot represent updates of rank larger than r. Therefore, as we discuss further in Section 2.1, we conduct our analysis under the assumption that L full λ has a low-rank global minimizer.
Nuclear norm regularization. Nuclear norm regularization is a popular technique that promotes low-rank solutions in matrix optimization. As the convex envelope of the rank function on the unit ball (Fazel et al., 2001), the nuclear norm penalty provides a tractable alternative to directly minimizing rank. Its effectiveness in yielding low-rank solutions has been demonstrated both theoretically and empirically across various fields, including matrix sensing (Recht et al., 2010), computer vision (Cabral et al., 2013) optimization (Hu et al., 2021), deep learning (Kobayashi et al., 2024), and LoRA (Jang et al., 2024). Collectively, these prior results make the assumption that L full λ admits a low-rank minimizer more natural.
this section cite: ['b15', 'b7', 'b9', 'b30', 'b4', 'b16', 'b19', 'b17']

Section: Main assumptions
In this section, we define and quickly justify the main assumptions used in our analyses of Section 3.
this section cite: []

Section: Existence of a low-rank minimizer
Throughout our analysis, we assume that there exists a rank r ⋆ global minimizer of full fine-tuning loss L full λ and that our LoRA module uses rank r ≥ r ⋆ .
We argue that there is sufficient conceptual and experimental justification supporting the assumption. Initially, LoRA (Hu et al., 2022) was proposed based on the insight that learned over-parameterized models lie in a low intrinsic dimension (Li et al., 2018;Aghajanyan et al., 2021), making them amenable to low-rank updates during fine-tuning. Moreover, as discussed in Section 1.3, training LoRA with weight decay is equivalent to nuclear norm regularization in full fine-tuning, thereby strongly biasing the solution toward low rank. As shown in Table 1 and further discussed in Section 4, we experimentally verify the low-rank assumption in a few setups. Finally, the extensive empirical literature demonstrating the success of LoRA with small rank r further justifies this assumption.
Nevertheless, it may sometimes be more realistic to assume that the global minimizer of full fine-tuning is only approximately low rank. We address this issue in Section 3.3, where we generalize the analysis to the case where the global minimizer of full fine-tuning is not exactly low rank.
this section cite: ['b15', 'b23', 'b0']

Section: Restricted strong convexity and smoothness
Our analyses also rely on the assumptions of restricted smoothness and restricted strong convexity, which are weaker assumptions compared to the smoothness and strong convexity assumptions commonly used in optimization.
We say a twice-differentiable function
f : R m×n → R is (α, r, D)-restricted strongly convex about X ⋆ if ⟨∇f (X) -∇f (X ⋆ ), X -X ⋆ ⟩ ≥ α∥X -X ⋆ ∥ 2 F .
for any X ∈ R m×n such that ∥X -X ⋆ ∥ F ≤ D and rank(X) ≤ r. We denote the largest α such that f is (α, r, D)-restricted strongly convex about X ⋆ as the (r, D)-RSC constant of f about X ⋆ .
We say a twice-differentiable function
f : R m×n → R is (β, r, D)-restricted smooth about X ⋆ if ∇ 2 f (X)[U X + XV, U X + XV ] ≤ β∥U X + XV ∥ 2 F for any [X ∈ R m×n such that ∥X -X ⋆ ∥ F ≤ D and rank(X) ≤ r], [U ∈ R m×m such that ∥U ∥ F = ∥V ∥ F = 1]
and rank(U ) = 1], and [V ∈ R n×n such that ∥V ∥ F = 1 and rank(U ) = rank(V ) = 1]. We denote the smallest β such that f is (β, r, D)-restricted smooth about X ⋆ (or β = ∞ if there is no such finite value) as the (r, D)-RSM constant of f about X ⋆ .
In this work, we consider the case where α > 0 and β < ∞. Although deep learning objectives are typically neither strongly convex nor have small smoothness constants, the restricted notions of strong convexity and smoothness are valid in many practical fine-tuning scenarios as we empirically demonstrate in Section 4. Finally, this current definition treats f as a function of a single matrix X. In Section 3.2, we generalize the definitions to X = (X (1) , X (2) , . . . , X (L) ) with multiple matrices.
this section cite: []

Section: Spurious Local minima of LoRA
In this section, we analyze the loss landscape of LoRA finetuning and show that in the "generic regime", a second-order stationary point (SOSP) is either (i) a global minimizer with small rank and small magnitude or (ii) a spurious solution with high rank and large magnitude.
Section 3.1 starts by presenting the result in the simpler setup of fine-tuning a single matrix when a low-rank global minimizer exists. Section 3.2 extends the result to the setup of fine-tuning multiple matrices. Section 3.3 extends the theory to work when an approximately low-rank global minimizer exists. The extensions of Sections 3.2 and 3.3 slightly complicate the notation, but the qualitative conclusion is maintained. In Section 3.4, we discuss why first-order optimizers with zero-initialization and weight decay, are unlikely to converge to the spurious local minimizers.
this section cite: []

Section: LoRA converges to a global minimizer or fails loudly
We now state our main result. 1. If 2α > β (special regime), X □ is a global minimum.
2. If 2α ≤ β (generic regime), one of the following holds.
(i) X □ is a global minimum. (ii) X □ is not a global minimum, rank(X □ ) = r with σ r (X □ ) ≥ 2α β σ r⋆ (X □ ),and
∥X □ -X ⋆ ∥ 2 F ≥ ∥X □ -Π rank≤r⋆ (X □ )∥ 2 F 1 - 2ασr ⋆ βσr , where Π rank≤r⋆ (X □ ) is the projection of X □ onto the set of matrices of rank r ⋆ or less.
To clarify, when we say X □ is or is not a global minimum, it is with respect to L full λ .
We denote 2α > β as the special regime, as the loss objective should be very well-conditioned to fall in this regime. Most practical setups would fall into the generic regime with β ≥ 2α, thereby being the regime of primary interest.
The global minimizer X ⋆ of the full fine-tuning loss L full λ is assumed to be low rank, and we intuitively understand that X ⋆ should have small magnitude since we are fine-tuning. Theorem 1 states that in the generic regime, there may be additional spurious local minima, but those will have high rank and will be far away from the global minimizer X ⋆ .
The following corollary restates Theorem 1 in an alternate form that clarifies its main conclusions.
Corollary 1. Consider the setup in Theorem 1. Further assume the strict inequality r > r ⋆ . Let (A, B) be a SOSP of L lora λ with X □ = AB ⊺ and ∥X □ -X ⋆ ∥ F ≤ D. Then, (i) If σ r (X □ ) ≤ 2α β σ r⋆ (X □ ), then X □ is a global mini- mizer. (ii) If σ r (X □ ) > 2α β σ r⋆ (X □ ),
then X □ is a spurious solution, and further X □ has large magnitude with
∥X □ ∥ F ≥ r s=r⋆+1 σ 2 s (X □ ) 1 - 2ασr ⋆ βσr -∥X ⋆ ∥ F .
LoRA training converges to a global minimizer or fails loudly. As discussed in Section 1.2, Lee et al. (2016) and Ge et al. (2015) imply that LoRA fine-tuning with SGD converges to a SOSP. In Section 3.4, we argue why it is likely that the SOSP we converge to is a global minimizer.
However, if LoRA fine-tuning does converge to a spurious solution, its high rank and large magnitude would be noticeable, and, as the experiments in Section 4 show, generalization will be poor. In this sense, we describe this mode of failure to be "failing loudly."
Relation to prior work. Interestingly, Theorem 1 completely includes the prior loss landscape analysis of (Jang et al., 2024), which considers a linearized loss in the NTK regime with an ε-perturbation. This perturbation ensures 2α = 2ε > β = ε, placing the loss objective in the special regime. Then, with Theorem 1, we conclude that any SOSP is a global minimum.
this section cite: ['b21', 'b11', 'b17']

Section: PROOF OUTLINE OF THEOREM 1
Our proof technique takes inspiration from the low-rank optimization literature. In fact, the analysis in the special regime 2α ≥ β naturally extends results from matrix sensing (Zhu et al., 2018;Ha et al., 2020). On the other hand, the analysis on the generic regime 2α < β is a novel result of ours. In the matrix sensing setting, showing that local minimizers near the solution are global minimizers has limited meaning since there is not a good estimate of the global minimizer, so such results were not pursued. On the other hand, in the LoRA fine-tuning setup, 0, the pre-trained baseline, is a good estimate of the global minimizer.
We defer the full proof of Theorem 1 to Appendix A, providing a brief outline here. For notational simplicity, write f (X) = L full (X) and g(A, B) = L lora (A, B), and X = X □ . (So X is assumed to be an SOSP.
) Denote the compact SVD of X as L X Σ X R ⊺
X , and σ i , u i , v i as the i-th (largest) singular value of X and the corresponding singular vectors. From the first and second-order optimality of g(A, B), we acquire the following properties:
1. 0 = ∇ A g(A, B) = ∇f (X) • B + λA 2. 0 = ∇ B g(A, B) = ∇f (X) ⊺ • A + λB 3. ∇ 2 g(A, B)[(U, V ), (U, V )] = 2⟨∇f (X), U V ⊺ ⟩ + ∇ 2 f (X)[AV ⊺ + U B ⊺ , AV ⊺ + U B ⊺ ] + λ(∥U ∥ 2 F + ∥V ∥ 2 F ) ≥ 0 for any (U, V ).
Properties 1 and 2 imply ∇f (X) can be represented as
∇f (X) = -λL X R ⊺ X + S, L ⊺ X S = SR X = 0 (1) for some matrix S. Furthermore, plugging in (U, V ) = (-u ⋆ u ⊺ r A, v ⋆ v ⊺ r B
) into property 3 and using the βrestricted smoothness of f , where (u ⋆ , v ⋆ ) are the top singular vectors of S, we find
∥S∥ 2 ≤ λ + βσ r .
(2) From ( 1), (2) we can induce there exists a subgradient g ∈ ∂(λ∥X∥ * ) such that ∥g + ∇f (X)∥ 2 ≤ βσ r .
Denoting Z = g + ∇f (X) and κ = σr ⋆ βσr , we see ∥κZ∥ ≤ σ r⋆ and therefore the top r ⋆ singular vectors of X -κZ coincide with those of X. Thus, by the Eckart-Young-Mirsky Theorem, we have
Π rank≤r⋆ (X) ∈ argmin rank(Y )≤r⋆ ∥Y -(X -κZ)∥ 2 F . Since rank(X ⋆ ) = r ⋆ , ∥X r⋆ -X + κZ∥ 2 F ≤ κX∥X ⋆ -X + κZ∥ 2 F which is again equivalent to ∥X r⋆ -X∥ 2 F ≤ ∥X ⋆ -X∥ 2 F + 2κ⟨X ⋆ -X, Z⟩ Now α-restricted convexity at X ⋆ implies ⟨X -X ⋆ , ∇f (X) -∇f (X ⋆ )⟩ ≥ α ∥X -X ⋆ ∥ 2 F
By the global optimality of X ⋆ , from Mordukhovich & Shao (1995, Theorem 3.1) we have -∇f (X ⋆ ) ∈ ∂(λ∥X ⋆ ∥ * ) and thus the subgradient property implies
⟨X ⋆ , g -(-∇f (X ⋆ ))⟩ ≥ 0
Summing up the three inequalities, we have
(2κα -1)∥X ⋆ -X∥ 2 F + ∥X r⋆ -X∥ 2 F ≤ 0
Therefore when 2κα > 1, X ⋆ = X, and when 2κα < 1 the inequality of the theorem holds.
this section cite: ['b42', 'b13']

Section: Extension to fine-tuning multiple matrices
For the sake of notational convenience, Theorem 1 was stated for the case of fine-tuning a single weight matrix.
In this section, we generalize the result to the case of finetuning multiple matrices.
First, we extend the definition of restricted smoothness and strong convexity to the multiple matrix case.
Let f : R m1×n1 × • • • × R m L ×n L → R be twice dif- ferentiable. Let X = (X (1) , X (2) , . . . , X (L) ), α = (α (1) , . . . , α (L)
), and β = (β (1) , . . . , β (L) ).
We say f is (α, r, D)-restricted strongly convex about X ⋆ if for each 1 ≤ l ≤ L, ⟨∇ l f (X ⋆ )-∇ l f (X), X (l) -X (l) ⋆ ⟩ ≥ α (l) ∥X (l) -X ⋆ (l) ∥ 2 F .
for any X such that ∥X -X ⋆ ∥ F ≤ D. We denote the tuple α of the largest α (l) s such that f is (α, r, D)-restricted strongly convex about X ⋆ as the (r, D)-RSC constant of f about X ⋆ .
We say a twice-differentiable function
f : R m×n → R is (β, r, D)-restricted smooth about X ⋆ if for each 1 ≤ l ≤ L, ∇ 2 l,l f (X)[U X (l) + X (l) V, U X (l) + X (l) V ] ≤ β (l) ∥U X (l) + X (l) V ∥ 2 F for any X such that ∥X -X ⋆ ∥ F ≤ D, U ∈ R m l ×m l such that rank(U ) = 1 and ∥U ∥ F = 1 , V ∈ R n l ×n l such that rank(V ) = 1
and ∥V ∥ F = 1. We denote the tuple β of the largest β (l) such that f is (β, r, D)-restricted strongly convex about X ⋆ as the (r, D)-RSM constant of f about X ⋆ . Here ∇ l , ∇ 2 l,l refers to the gradient and Hessian respect to the lth matrix X (l) .
Next, under this extended notion of restricted smoothness and convexity, we present the natural extension of Theorem 1 below. The proof follows the same reasoning as in Theorem 1 and is detailed in Appendix A Theorem 2. Let λ ≥ 0.
Assume the full finetuning loss
L full λ has a rank-r ⋆ global minimizer X ⋆ = (X (1) ⋆ , . . . , X (L) ⋆ ).
Respectively denote the (r, D)-RSC and (r, D)-RSM constants of L full about X ⋆ as α = (α (1) , . . . , α (L) ) and β = (β (1) , . . . , β (L) ). Assume α (1) , . . . , α (L) > 0 and β (1) , . . . , β (L) < ∞. Assume we use LoRA modules all with rank r ≥ r ⋆ . Then, every SOSP (A, B) of L λ with X □ = AB ⊺ and ∥X □ -X ⋆ ∥ F ≤ D satisfies the following. l) for some l = 1, . . . , L (generic regime), one of the following holds.
1. If 2α (l) ≥ β (l) for all l = 1, . . . , L (special regime), X □ is a global minimum 2. If 2α (l) < β (
(i) X □ is a global minimum. (ii) X □ is not a global minimum, X (l) □ is exactly rank r with σ r (X (l) □ ) > 2α (l) β (l) σ r⋆ (X (l) □ )
and
X (l) □ -X (l) ⋆ 2 F ≥ X (l) □ -Π rank≤r⋆ (X (l) □ ) 2 F 1 - 2α (l) σr ⋆ β (l) σr for some l = 1, . . . , L, where Π rank≤r⋆ (X (l) □ ) is the projection of X (l) □ onto the set of matrices of rank r ⋆ or less.
To clarify, when we say X □ is or is not a global minimum, it is with respect to L full λ .
this section cite: []

Section: Extension to approximately low-rank minimizers
In Sections 3.1 and 3.2, we assumed the nuclear-norm regularized full fine-tuning loss L full λ has a low-rank minimizer, but this assumption may be unrealistic especially when the weight-decay parameter λ is too small. In this section, we relax this assumption and consider the case where L full λ has an approximately low-rank minimizer. As in Section 3.1, we present here the result for the single matrix case. In Appendix A, we provide a 'Master Theorem' that combines the generalizations of Theorems 2 and 3. We say X (δ) ⋆ is a δ-global minimizer of full fine-tuning if
∥X (δ) ⋆ -X ⋆ ∥ F ≤ δ
for some X ⋆ that exactly minimizes L full .
Theorem 3. Let ε > 0 and λ ≥ 0. Assume the full finetuning loss L full λ has a rank-r ⋆ δ-global minimizer X
⋆ with δ = o(ε 3 ). Respectively denote the (r, D)-RSC and (r, D)-RSM constants of L full about X ⋆ as α and β. Assume 0 < α and β < ∞. Assume we use a LoRA module with rank r ≥ r ⋆ . Then, every SOSP (A, B) of L lora λ with X □ = AB ⊺ and ∥X □ -X ⋆ ∥ F ≤ D satisfies the following.
1. If 2α ≥ β(1 + ε) (special regime), X □ is an ε-global minimizer.
2. If 2α < β(1 + ε) (generic regime), one of the following holds.
(
i) X □ is an ε-global minimizer. (ii) X □ is not an ε-global minimizer, X □ is exactly rank r with σ r (X □ ) ≥ max{ 2α β(1+ε) σ r⋆ (X □ ), α 2β √ r • ε}, and either σ r (X □ ) ≤ 2α β σ r⋆ (X □ ) or ∥X □ -X ⋆ ∥ F ≥ X □ -Π rank≤r⋆ (X □ ) 2 F -ε 3 1 - 2ασr ⋆ βσr -ε 2
where Π rank≤r⋆ (X □ ) is the projection of X □ onto the set of matrices of rank r ⋆ or less.
To clarify, when we say X □ is or is not an ε-global minimizer, it is with respect to L full λ .
this section cite: []

Section: LoRA training probably won't fail; it probably won't converge to spurious local minima
In the analysis of Section 3.1 and its subsequent generalizations, we showed that in the generic regime, spurious local minima may exist, but if the training converges to them, this will be very noticeable (failing loudly), as the spurious solutions have high rank and large magnitude. In this section, we argue that the standard LoRA fine-tuning procedure induces implicit biases that make it unlikely for the LoRA training to converge to these spurious local minima.
Zero-initialization biases the optimization towards minima with smaller magnitude. LoRA fine-tuning is initialized with B = 0, leading to X = AB ⊺ = 0 at initialization. This choice comes from the intuition that fine-tuning should not change the model too much, i.e., that X ⋆ should be small, so the initialization should be at 0.
When weight decay is used, we can make this argument further quantitative. The global minimizer X ⋆ satisfies
L(X ⋆ ) + λ∥X ⋆ ∥ * ≤ L(0) + λ∥0∥ * , thus ∥X∥ * < L(0) λ .
Here, L(0) is the loss corresponding to directly applying the pre-trained model to the fine-tuning task, so L(0) should not be inordinately large when the fine-tuning task is not too different from tasks seen during pre-training.
On the other hand, spurious local minima exist only outside a neighborhood of zero, as argued in Corollary 1. Because the SGD or Adam optimizers used for LoRA training are initialized at 0, the optimization is biased towards smallermagnitude solutions near the starting point, which are the global minima. In Section 4, we experimentally test this theory by fine-tuning LoRA with a non-zero initialization; indeed, we find there is an instance in this scenario, where the fine-tuning gets trapped in spurious local minima.
Weight decay implicitly biases the optimization towards low-rank matrices. Practical LoRA training typically employs weight decay (Hu et al., 2022;Dettmers et al., 2023), and it is shown in prior theoretical work that weight decay induces an implicit bias toward low-rank matrices. This makes it more likely for the LoRA training to converge to the low-rank global minimizer, rather than to a spurious local minima with high rank being σ r (X) > 2α β σ r⋆ (X). For deep linear networks, this implicit bias is characterized somewhat precisely. Theorem (Informal, Theorem 3.2 of Wang & Jacot (2024)). When training a deep linear network with positive weight decay, a sufficiently small learning rate, and a ground-truth teacher model with low effective rank, there is a positive probability of jumping from a high-rank critical point to a lower-rank one, but the probability of jumping back is zero.
While the theory of Wang & Jacot (2024) does not immediately apply to general deep (non-linear) neural networks, it does provide meaningful insight into the implicit bias towards low rank. In the more general setup, Galanti et al. (2024) argues for a similar implicit bias. Adapting their arguments to LoRA training, we get the following statement. Lemma 1. Consider LoRA training with SGD with batch size b, learning rate µ, and weight decay λ > 0. For any lowrank update X = AB ⊺ of a weight matrix in the network, if the sequence of X-values throughout training converges to a matrix X, then X is approximately low rank in the sense that for any ε > 0, there exists some W with
X ∥ X∥ -W < ε, rank(W ) ≤ b log(ε/4) log(1 -µλ)
We provide the proof of Lemma 1 in Appendix A.4.
this section cite: ['b15', 'b7', 'b35', 'b35', 'b10']

Section: Experiments
In this section, we validate our theory through real-world experiments. First, we verify our assumptions outlined in Section 2. Then, we present both the success and failure modes of LoRA fine-tuning, where training either converges to a low-rank, small-magnitude global minimizer or stuck on a high-rank, large-magnitude local minimizer.
this section cite: []

Section: Experimental setup.
We conduct experiments on two tasks in NLP and vision. For the NLP task, we fine-tune a RoBERTA-base model (Zhuang et al., 2021) on a sentiment analysis task, using the SST-2 dataset (Socher et al., 2013) from the GLUE benchmark (Wang et al., 2018). For the vision task, we fine-tune a vision transformer (Dosovitskiy et al., 2021) on the CIFAR100 dataset (Krizhevsky, 2009). Both models have 12 attention layers, and we tune the query and value weights of each layer, following the prescription of Hu et al. (2022). We describe further details in Appendix C.
Results: Verifying low-rank global minima exist. First, we validate our assumption of a low-rank global minimum. We perform full fine-tuning on the nuclear norm regularized loss objective L f ull λ with varying values of weight decay λ. The results of Table 1 exhibit a clear decreasing trend on the rank of the global minimum as a function of λ. Notably, when λ is set to values at least 0.001, the resulting rank is lower than typical LoRA ranks (4, 8, or 16).
Results: Verifying RSC and RSM. Next, we verify our assumption of restricted strong convexity and smoothness. As it is infeasible to exactly compute α and β values, we estimate them by Monte-Carlo sampling with 1000 samples within rank bound r = 8, 16, 32, 64, distance bound D = 5, and λ = 0.01. Table 2 presents the α and β values for the largest β/α value across weight matrices. We see as r increases, α decreases and β increases. In fact, when r is as large as 64, the requirement α > 0 breaks, and our theory no longer applies. These results demonstrate that our assumption of α > 0 and β < ∞ is plausible using a low LoRA rank r. This also suggests that reduced memory footprint is not the only benefit of using small r; the α, βvalues that determine the loss landscape also become more favorable with small r.
Results: Validating main theorem. Finally, we verify our main result through an illustrative example for the SST2 task with λ = 0.01 and r = 8. To clarify, our results prove that spurious local minima may not exist, but when they do, they exhibit high rank and large norm, being readily distinguishable from the global minimum and thereby avoidable through zero initialization. We present in Figure 2 that such spurious local minimum found by large random initialization indeed fails loudly in the sense that it has high rank, large magnitude, and poor generalization performance. We further demonstrate in Appendix C that spurious local minima isn't found in any smaller initializations.
this section cite: ['b43', 'b31', 'b33', 'b8', 'b20', 'b15']

Section: Conclusion
In this work, we theoretically analyze LoRA fine-tuning and obtain a new type of result: that a second-order station-ary point is either a global minimizer with low rank and small magnitude or is a spurious solution with high rank and large magnitude. Unlike previous analyses based on linearization, our approach relies on a general condition of restricted strong convexity and smoothness, which are conditions the experiments of Section 4 confirm to be practical. We further argue that zero-initialization and weight decay in LoRA training induce an implicit bias toward this low-rank small-magnitude region, explaining why LoRA typically converges to global minima in practice.
While the primary focus of this work is on establishing the theoretical convergence of LoRA, our framework possesses broader practical relevance. The properties of spurious local minima that we characterize may be used to diagnose and monitor the fine-tuning process. Furthermore, as our framework relies solely on the low-rank decomposition structure of LoRA and a few minimal assumptions, our theory applies to many LoRA variants, including LoRA+ (Hayou et al., 2024), rsLoRA (Kalajdzievski, 2023), PiSSA (Meng et al., 2024), and MiLoRA (Wang et al., 2025) as well.
Our results open several avenues for future work. One is to perform a more rigorous analysis of the implicit bias induced by weight decay and zero initialization. Another intriguing insight is that the restricted strong convexity and smoothness constants α and β improve as the LoRA rank decreases, suggesting that smaller-rank parameterizations enjoy more favorable optimization landscapes. This observation contrasts with the modern wisdom of deep learning theory that overparameterization helps training and aligns with recent results indicating that overparameterization can slow down training (Xu & Du, 2023;Xiong et al., 2024).
Exploring this phenomenon further is another promising direction.
this section cite: ['b14', 'b18', 'b26', 'b34', 'b37', 'b36']

Section: References
Ref_id:b0 Title: Intrinsic dimensionality explains the effectiveness of language model fine-tuning Year: (2021)
Ref_id:b1 Title: Simple parameter-efficient fine-tuning for transformer-based masked language-models Year: (2022)
Ref_id:b2 Title: Global optimality of local search for low rank matrix recovery Year: (2016)
Ref_id:b3 Title: A nonlinear programming algorithm for solving semidefinite programs via low-rank factorization Year: (2003)
Ref_id:b4 Title: Unifying nuclear norm and bilinear factorization approaches for low-rank matrix decomposition Year: (2013)
Ref_id:b5 Title: Exact matrix completion via convex optimization Year: (2012)
Ref_id:b6 Title: Gradient dynamics for low-rank fine-tuning beyond kernels Year: (2024)
Ref_id:b7 Title: Efficient finetuning of quantized LLMs. Neural Information Processing Systems Year: (2023)
Ref_id:b8 Title: An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations Year: (2021)
Ref_id:b9 Title: A rank minimization heuristic with application to minimum order system approximation Year: (2001)
Ref_id:b10 Title: SGD and weight decay secretly minimize the rank of your neural network Year: (2024)
Ref_id:b11 Title: Escaping from saddle points -online stochastic gradient for tensor decomposition Year: (2015)
Ref_id:b12 Title: No spurious local minima in nonconvex low rank problems: A unified geometric analysis Year: (2017)
Ref_id:b13 Title: An equivalence between critical points for rank constraints versus low-rank factorizations Year: (2020)
Ref_id:b14 Title: LoRA+: efficient low rank adaptation of large models Year: (2024)
Ref_id:b15 Title: Low-rank adaptation of large language models Year: (2022)
Ref_id:b16 Title: Low Rank Regularization: A review Year: (2021)
Ref_id:b17 Title: LoRA training in the NTK regime has no spurious local minima Year: (2024)
Ref_id:b18 Title: A rank stabilization scaling factor for finetuning with LoRA Year: (2023)
Ref_id:b19 Title: Weight decay induces low-rank attention layers Year: (2024)
Ref_id:b20 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b21 Title: Gradient descent only converges to minimizers Year: (2016)
Ref_id:b22 Title: The power of scale for parameter-efficient prompt tuning Year: (2021)
Ref_id:b23 Title: Measuring the intrinsic dimension of objective landscapes Year: (2018)
Ref_id:b24 Title: Prefix-tuning: Optimizing continuous prompts for generation Year: (2021)
Ref_id:b25 Title: A kernel-based view of language model fine-tuning Year: (2023)
Ref_id:b26 Title: Principal singular values and singular vectors adaptation of large language models Year: (2024)
Ref_id:b27 Title: On nonconvex subdifferential calculus in banach spaces Year: (1995)
Ref_id:b28 Title: Proximal algorithms Year: (2014)
Ref_id:b29 Title: Nonsquare matrix sensing without spurious local minima via the Burer-Monteiro approach Year: (2017)
Ref_id:b30 Title: Guaranteed minimum-rank solutions of linear matrix equations via nuclear norm minimization Year: (2010)
Ref_id:b31 Title: Recursive deep models for semantic compositionality over a sentiment treebank Year: (2013)
Ref_id:b32 Title: Understanding linear probing then fine-tuning language models from NTK perspective Year: (2024)
Ref_id:b33 Title: GLUE: A multi-task benchmark and analysis platform for natural language understanding Year: (2018)
Ref_id:b34 Title: Harnessing minor singular components for parameterefficient LLM finetuning Year: (2025)
Ref_id:b35 Title: Implicit bias of SGD in L 2regularized linear DNNs: One-way jumps from high to low rank Year: (2024)
Ref_id:b36 Title: How over-parameterization slows down gradient descent in matrix sensing: The curses of symmetry and initialization Year: (2024)
Ref_id:b37 Title: Over-parameterization exponentially slows down gradient descent for learning a single neuron Year: (2023)
Ref_id:b38 Title: The expressive power of low-rank adaptation Year: (2024)
Ref_id:b39 Title: Sharp global guarantees for nonconvex lowrank matrix recovery in the overparameterized regime Year: (2021)
Ref_id:b40 Title: Improved global guarantees for the nonconvex Burer-Monteiro factorization via rank overparameterization Year: (2024)
Ref_id:b41 Title: LoRA-One: One-step full gradient suffices for low-rank fine-tuning, provably and efficiently Year: (2025)
Ref_id:b42 Title: Global optimality in low-rank matrix optimization Year: (2018)
Ref_id:b43 Title: A robustly optimized BERT pre-training approach with post-training Year: (2021)
