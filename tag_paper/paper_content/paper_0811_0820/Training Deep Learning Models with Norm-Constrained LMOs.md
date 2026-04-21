Title: Training Deep Learning Models with Norm-Constrained LMOs
Abstract: In this work, we study optimization methods that leverage the linear minimization oracle (lmo) over a norm-ball. We propose a new stochastic family of algorithms that uses the lmo to adapt to the geometry of the problem and, perhaps surprisingly, show that they can be applied to unconstrained problems. The resulting update rule unifies several existing optimization methods under a single framework. Furthermore, we propose an explicit choice of norm for deep architectures, which, as a side benefit, leads to the transferability of hyperparameters across model sizes. Experimentally, we demonstrate significant speedups on nanoGPT training using our algorithm, Scion, without any reliance on Adam. The proposed method is memory-efficient, requiring only one set of model weights and one set of gradients, which can be stored in halfprecision.

Section: Introduction
Deep learning has greatly benefited from adaptive optimization methods such as RMSProp (Hinton et al., 2012), AdaGrad (Duchi et al., 2011;McMahan & Streeter, 2010), and Adam (Kingma, 2014), which dynamically change the geometry of the problem based on gradients encountered on-the-fly during training. While these methods have demonstrated remarkable success, they fundamentally treat neural networks (NNs) as optimization problems where we lack any prior knowledge about their particular setting.
However, NNs are far from being black boxes-their structure is not only known but they are deliberately designed. This simple observation raises directly the question:
Is it more beneficial to adapt the optimizer a priori, instead of exploring their respective geometries on-the-fly?
Adaptation on-the-fly has been the defacto standard in this setting, with adaptive algorithms, such as Adam (Kingma, 2014), dominating the deep learning model training.
One possible way for adaptation a priori, which we focus on in this work, is to modify the underlying norm used to measure distances in the parameter space. There is precedence to our proposal, as the early work by Carlson et al. (2015a;b;c) introduced the stochastic spectral descent method (SSD), which performs steepest descent in the spectral norm, and demonstrated that the method can substantially accelerate deep learning training.
The significance of the SSD approach has been very recently brought back to attention by Bernstein & Newhouse (2024b), who showed that the Shampoo optimizer (Gupta et al., 2017)-winner of the external tuning track at the 2024 AlgoPerf: Training Algorithms competition (Dahl et al., 2023)-can be viewed as SSD when a certain accumulation is disabled. Moreover, Bernstein & Newhouse (2024b) introduced an efficient Newton-Schultz iteration to replace the approximate SVD calculations previously required. Jordan et al. (2024b) incorporated the Newton-Schultz iteration with additional momentum into SSD under the name Muon to achieve impressive results on the nanoGPT architecture by applying it to the hidden layers.
Contributions This work focuses on developing an algorithmic framework that can exploit an appropriate choice of norm for the entire neural network with particular emphasis on hyperparameter transfer across model sizes (Yang & Hu, 2021), convergence and practical performance.
To adapt to the geometry a priori, we will build on a classical (but unexpected) family of algorithms in contrast to the steepest descent methods, namely the ones involving the linear minimization oracle (lmo) over a norm-ball constraint known as the Conditional Gradient (CG) methods.
While classically being used for constrained problems, we take the slightly unusual approach by showing that the lmos can be used even for unconstrained problems. The algorithm, dubbed as the unconstrained Stochastic Conditional Gradient method (uSCG), shows improvements both theoretically and practically when the norm-ball constraint matches the natural geometry of the problem.  (Hazan et al., 2015) Momentum Normalized SGD [0, 1] Unconstrained Euclidean ∥ • ∥ 2 -ball -ρ d ∥d∥2 (Cutkosky & Mehta, 2020) SignSGD 1 Unconstrained Max-norm ∥ • ∥ ∞ -ball -ρ sign(d) (Bernstein et al., 2018, Thm. 1) 2 Signum [0, 1] Unconstrained Max-norm ∥ • ∥ ∞ -ball -ρ sign(d) (Bernstein et al., 2018, Thm. 3 Jordan et al., 2024b) 1 With non-Nesterov based momentum. 2 The theoretical guarantee relies on increasing batch size.
) 2 Muon 1 [0, 1] Unconstrained Spectral ∥ • ∥ S∞ -ball -ρU V ⊤(
In particular, we build on the Stochastic Conditional Gradient (SCG) method of Mokhtari et al. (2020) from the constrained setting, which provides explicit control on the norm of NN weight matrices. This is particularly relevant for robust image classification (Cisse et al., 2017), generalization bounds (Bartlett et al., 2017), Lipschitz control of generative adversarial networks (Arjovsky et al., 2017;Miyato et al., 2018), diffusion models (Karras et al., 2024, Sec. 2.3), and for ensuring Lipschitz continuity of NNs (Large et al., 2024).
Concretely, we make the following contributions:
Theoretical rates: We introduce a new, stochastic lmo based family of algorithms uSCG, which can exploit the specific geometry of the problem. In doing so we achieve the O(n -1/4 ) order optimal convergence rate under general nonconvexity and stochasticity for uSCG (Arjevani et al., 2022). Moreover, we provide a new analogous guarantee for the constrained case for SCG. A major benefit of both methods is that their stepsize is agnostic to the Lipschitz constant, in contrast to steepest descent which requires the stepsize to be taken small enough.
Unification: Our lmo-based approach provides a unifying framework for various popular algorithms, based on the norm choice (see Table 1); as a byproduct we establish the first provable rate for the Muon optimizer with and without weight decay. More importantly, this generality allows us to design a new method for deep learning based on operator norms called SCION (Algorithm 3), which enjoys zero-shot hyperparameter transferability (Yang et al., 2022), and can be implemented storing only one set of parameters and one gradient (stored in half-precision), economizing on memory in large-scale training.
Numerical validation: We carry out exhaustive numerical evaluation of SCION ranging from small scale experiments on MLPs and CNNs to ViT on ImageNet and NanoGPT models with up to 3B parameters. We consistently observe the transferability properties across all settings for SCION. The scheme is more tolerant to large batch sizes and exhibits superior performance due to the a priori adaptation.
An additional lmo-based algorithm ALMOND can be found in Appendix D.3, generalizing the Normalized SGD based method of Zhao et al. (2020), for training with largebatches. Key differences of ALMOND with uSCG and SCG are discussed to further motivate our algorithms.
this section cite: ['b29', 'b20', 'b43', 'b37', 'b37', 'b25', 'b17', 'b58', 'b27', 'b15', 'b7', 'b45', 'b12', 'b4', 'b2', 'b44', 'b39', 'b1', 'b59', 'b66']

Section: Preliminaries
We are interested in solving the following general (possibly nonconvex) optimization problem
min x∈X f (x) ,(1)
where f is smooth in some not necessarily Euclidean norm and the problem is either unconstrained (e.g., X = R d ) or constrained to X = D where D is the norm-ball defined as D := {x | ∥x∥ ≤ ρ}.
The central primitive in the algorithms considered in this work is the linear minimization oracle (lmo) defined as lmo(s) ∈ arg min x∈D ⟨s, x⟩ ,
where we are particularly interested in the special case where the constraint set is a norm constraint ∥x∥ ≤ ρ, for some ρ > 0 and some norm ∥ • ∥, which does not have to be the Euclidean norm. Examples of norm-constrained lmos are provided in Table 1 and Table 2 regarding operator norms. An important property of the lmo is that the operator is scale invariant, i.e., lmo(a • s) = lmo(s) for a > 0, and in fact we have by construction under the norm constraints that ∥ lmo(s)∥ ≤ ρ. Thus, it is only the direction of the input s that matters and not the magnitude.
A classical method for solving the constrained variant of problem 1, when the lmo is available, is the Conditional Gradient method (CG) (Frank et al., 1956;Clarkson, 2010;Jaggi, 2013), which proceeds as follows with γ k ∈ (0, 1)
x k+1 = (1 -γ k )x k + γ k lmo(∇f (x k )),(CG)
ensuring the feasibility of x k via simplicial combination.
Algorithm 1 Unconstrained SCG (uSCG) Input: Horizon n, initialization x 1 ∈ X , d 0 = 0, momentum α k ∈ (0, 1], and stepsize γ k ∈ (0, 1) 1: for k = 1, . . . , n do 2:
Sample ξ k ∼ P 3:
d k ← α k ∇f (x k , ξ k ) + (1 -α k )d k-1
4:
x k+1 ← x k + γ k lmo(d k ) 5: Choose xn uniformly at random from {x 1 , . . . , x n } Return xn
Usually, the CG is attractive when the constraint set is an atomic set (e.g., the ℓ 1 -norm ball) in which case each update may be efficiently stored. Our focus lies in the more unconventional cases of the vector ℓ ∞ -norm ball and spectral norm ball for which the updates are in contrast dense. Furthermore, we are interested in the unconstrained case in addition to the constrained problem which CG solves.
In the stochastic regime, the analyzing lmo-based algorithms is involved. Even when the stochastic oracle ∇f (x, ξ) is unbiased, the direction of the updates, as defined by lmo(∇f (x, ξ)), is not unbiased. To help overcome this difficulty, we will employ a commonly used trick of averaging past gradients with α k ∈ (0, 1] (aka momentum),
d k = (1 -α k )d k-1 + α k ∇f (x k , ξ k ),(3)
which will rigorously help with algorithmic convergence.
this section cite: ['b23', 'b13', 'b30']

Section: Our Methods
For the unconstrained case we introduce a new method, dubbed the unconstrained SCG method (uSCG):
x k+1 = x k + γ k lmo(d k )(uSCG)
with stepsizes γ k ∈ (0, 1). Instead of the convex combination in CG, the update rule simply sums the lmos. In contrast with e.g., gradient descent, the update always has the same magnitude regardless of the size of the gradient average d k . The final algorithm is presented in Algorithm 1.
For the constrained case, we revisit the SCG method of Mokhtari et al. (2020) and adopt it for the non-convex objectives typically encountered in deep learning model training. This algorithm (Algorithm 2) proceeds as follows
x k+1 = (1 -γ k )x k + γ k lmo(d k ) (SCG)
with stepsizes γ k ∈ (0, 1).
Connection to weight decay For uSCG, weight decay has a very precise interpretation, since the method reduces to SCG. Consider the following variant of uSCG with weight decay
x k+1 = x k + γ k lmo(d k ) -γ k µx k .
this section cite: ['b45']

Section: Algorithm 2 Stochastic Conditional Gradient (SCG)
Input: Horizon n, initialization x 1 ∈ D, d 0 = 0, momentum α k ∈ (0, 1], and stepsize γ k ∈ (0, 1) 1: for k = 1, . . . , n do 2:
Sample ξ k ∼ P 3:
d k ← α k ∇f (x k , ξ k ) + (1 -α k )d k-1
4:
x k+1 ← (1 -γ k )x k + γ k lmo(d k ) 5: Choose xn uniformly at random from {x 1 , . . . , x n } Return xn
The weight decay parameter µ ∈ [0, 1] interpolates between uSCG and SCG. If the weight decay is in (0, 1) then the algorithm is still an instance of SCG and thus solve a constrained problem, but one with a larger radius of ρ ′ = ρ µ with a stepsize chosen as γ ′ k = γ k µ. Therefore, all schemes in Table 1 guarantees a norm bound of ρ µ on the parameters when combined with weight decay. The connection between weight decay and constrained optimization, in the special case where lmo = sign (when the norm-constraint in (2) is the vector ℓ ∞ -norm) has also been observed in Xie & Li (2024); D' Angelo et al. (2023). Due to the fixed magnitude of the lmo both methods provides a guarantee on the maximum norm of the parameters.
Insight 3.1. Both uSCG and SCG provide explicit control on the norm of the parameters:
(i) SCG guarantees ∥x∥ ≤ ρ. (ii) uSCG guarantees ∥x∥ ≤ ρ n k=1 γ k .
Norm control is particularly useful for long runs (cf. Figure 9) and to avoid overfitting in multi-epoch training (cf. Figures 10 and 11 regarding CIFAR10 experiments).
this section cite: ['b57']

Section: Choice of Norm Constraint
To choose an appropriate norm for deep learning, we build on the operator norm perspective of Large et al. (2024); Bernstein & Newhouse (2024a). To simplify the presentation we will consider a linear MLP as a running example, but in Section 3.2, we point to our theoretical guarantees with activation functions.
Let us a consider a linear MLP with the initial hidden layer defined as h 1 (z) = W 1 z + b 1 and the remaining layers
h ℓ (z) = W ℓ h ℓ-1 (z) + b ℓ , ∀ℓ ∈ 2, .., L;
with b L = 0. We denote the global loss as L(h L (z), y) where L is the loss function and y is a 1-hot encoded target vector. We use the overloaded notation W ℓ ∈ R dout×din , where d out and d in implicitly have dependency on ℓ and can thus be distinct across different layers.
Table 2. Example operator norms and the associated lmos of a matrix A ∈ R d out ×d in . The reduced SVD is given as A = U diag(σ)V ⊤ , sign acts elementwise, colj(A) := A•,j and rowi(A) := Ai,•. Note that this table is not exhaustive.
1 → RMS (ColNorm) 1 → ∞ (Sign) RMS → RMS (Spectral) RMS → ∞ (RowNorm) Norm max j 1 √ dout ∥ col j (A)∥ 2 max i,j |A i,j | din /dout∥A∥ S∞ max i √ d in ∥ row i (A)∥ 2 LMO col j (A) → - √ d out colj (A) ∥ colj (A)∥2 A → -sign(A) A → -dout /dinU V ⊤ row i (A) → -1 √ din rowi(A) ∥ rowi(A)∥2
Table 3. The choice of lmo can be different between layers and can depend on the assumptions on the input. For simplicity we overload notation and write the reduced SVD as
W ℓ = U diag(σ)V ⊤ ∈ R d out ×d in for all ℓ ∈ [L]. Parameter W 1 (image domain) {W ℓ } ℓ∈[2,...,L-1] W L b ℓ Norm RMS → RMS RMS → RMS RMS → RMS RMS → ∞ 1 → ∞ RMS LMO -max(1, dout /din)U V ⊤ -dout /dinU V ⊤ -dout /dinU V ⊤ row i (W L ) → -1 √ din rowi(WL) ∥ rowi(WL)∥2 -1 din sign(W L ) -bℓ ∥bℓ∥RMS Init. Semi-orthogonal Semi-orthogonal Semi-orthogonal Row-wise normalized Gaussian Random sign 0 Table 4. Example lmo choices for 1-hot encoded inputs. Parameter W1 (1-hot encoded input) Norm 2 → RMS 1 → RMS 1 → ∞ LMO - √ doutU V ⊤ colj(W1) → - √ dout colj (W1) ∥ colj (W1)∥2 -sign(W1) Init.
this section cite: ['b39']

Section: Semi-orthogonal Column-wise normalized Gaussian Random sign
We need that none of the intermediary hidden states h ℓ (z) blows up by requiring one of the following norm bounds:
(i) 1 dout ∥h ℓ (z)∥ 1 ≤ 1 (the average entry is bounded) (ii) ∥h ℓ (z)∥ RMS ≤ 1 (the typical entry is bounded) (iii) ∥h ℓ (z)∥ ∞ ≤ 1 (the maximum entry is bounded)
where ∥z∥ RMS := 1 √ d ∥z∥ 2 for z ∈ R d . Assuming the input to any given layer is bounded in some norm ∥ • ∥ α , this requirement corresponds to placing an operator norm constraint on the weight matrices {W ℓ } ℓ∈[L] and a norm constraint on the biases {b ℓ } ℓ∈ [L-1] .
The operator norm is in turn defined as follows
∥A∥ α→β := max z∈R d ,z̸ =0 ∥Az∥ β ∥z∥ α = sup ∥z∥α=1 ∥Az∥ β .(4)
Directly from the definition, we have that if the input z is bounded through ∥z∥ α ≤ 1, then the output ∥Az∥ β will be bounded when ∥A∥ α→β is bounded.
A collection of operator norms and their resulting lmos is provided in Table 2. It will be convenient to convert between bounds on these different operator norm which the following fact makes precise.
Fact 3.1. The operator norm satisfies for some ρ > 0
(i) ∥z∥ β ≤ ρ∥z∥ c , ∀z ∈ R d ⇒ ∥A∥ α→β ≤ ρ∥A∥ α→c . (ii) ∥z∥ α ≥ 1 ρ ∥z∥ c , ∀z ∈ R d ⇒ ∥A∥ α→β ≤ ρ∥A∥ c→β .
Fact 3.1 tells us that we can bound operator norms using bounds on the vector norms, i.e.,
∥z∥ ∞ ≤ ∥z∥ 2 ≤ ∥z∥ 1 ≤ √ d∥z∥ 2 ≤ d∥z∥ ∞ , ∀z ∈ R d .(5)
We start by focusing on controlling the RMS norm, but will later consider other norms. There are three example operator norms to consider for the MLP in consideration:
(i) Initial layer h 1 (z): ∥W 1 ∥ α1→RMS ≤ 1. (ii) Intermediary layers h ℓ (z): ∥W ℓ ∥ RMS→RMS ≤ 1 ∀ℓ ∈ {2, .., L -1}. (iii) Last layer h L (z): ∥W L ∥ RMS→β L ≤ 1.
Note that the operator norm ∥ • ∥ RMS→RMS is a scaled spectral norm, i.e., ∥A∥ RMS→RMS = din /dout∥A∥ 2→2 = din /dout∥A∥ S∞ for A ∈ R dout×din .
To concisely write the layerwise norm constraints in terms of a norm constraint on the joint parameter x = {W ℓ , b ℓ } ℓ∈[L] , we can define the norm in the lmo (2) as ∥x∥ := max
ℓ∈[L] 1 ρ ℓ max{∥W ℓ ∥ α ℓ →β ℓ , ∥b ℓ ∥ β ℓ } ≤ 1(6)
where ρ ℓ is a layerwise scaling factor of the constraint radius. What is particularly convenient algorithmically for the ℓ ∞ -norm is that the layerwise lmos can be computed separately (cf. Algorithm 3). A norm choice across layers was made through the ℓ 1 -norm in Flynn (2017) and through the ℓ ∞ -norm with the modular norm (Large et al., 2024) and block-normalization (Balles et al., 2020;Yu et al., 2017;Ginsburg et al., 2019).
A choice needs to be made for the input norm α 1 and output norm β L , which depends on the application:
Input layer For image domains, usually the input is rescaled pixel-wise to e.g., ensure that z ∈ [-1, 1] in which case ∥z∥ RMS ≤ 1 and the appropriate operator norm for the first layer becomes ∥W 1 ∥ RMS→RMS = din /dout∥W 1 ∥ S∞ .
In order to deal with the case where d in > d out , we choose the radius to be max(1, dout /din) (cf., Appendix B.1).
For language tasks, the input z is usually a 1-hot encoded vector in which case ∥z∥ ∞ = ∥z∥ 2 = ∥z∥ 1 = 1. In turn,
∥W 1 ∥ ∞→RMS = ∥W 1 ∥ 2→RMS = ∥W 1 ∥ 1→RMS
holds on this restricted domain, where we can freely pick the operator norm that leads to the simplest update rule (Table 2).
The simplest form for the lmo is arguably induced by ∥ • ∥ 1→RMS since the lmo can be computed exactly, while from ∥ • ∥ 2→RMS we can observe a more aggressive scaling factor in the lmo, - Large et al. (2024) for 1-hot encoded input. Through the above reasoning we see how the norm is equivalent to an appropriately scaled spectral norm.
√ d out U V ⊤ , than the -dout /dinU V ⊤ used in intermediate layers. The norm choice ∥ • ∥ 1→RMS was first proposed in
Output layer For the final layer, we are not restricted to bounding the output in ℓ RMS and can alternatively choose bounding the maximal entry through ℓ ∞ . Additionally, we can bound ∥A∥ RMS→∞ ≤ 1 din ∥A∥ 1→∞ , by using (5) through Fact 3.1, which leads to a dimension scaled sign update rule for the last layer.
We summarize the different norm choices and their resulting lmos in Tables 3 and 4. Table 3 provides an overview of norm choices of output layers, while Table 4 provides choices for input layers under 1-hot encoded input.
Provided that the input is bounded as described, each of the hidden states h ℓ (z) and logits h L (z) will be bounded in the RMS norm. In order to ensure feasibility of the initialization in the constrained case when employing SCG, we propose to initialize on the boundary similar to Large et al.  (2024)
this section cite: ['b22', 'b39', 'b3', 'b63', 'b24', 'b39']

Section: (see Appendix B for details).
Insight 3.2.
(i) For 1-hot encoded input, ColNorm and Spectral are equivalent for the first layer, in which case ColNorm is favored since the lmo can be computed exactly.
(ii) Sign can be used both for the first and last layer which is crucial for weight sharing.
(iii) To transfer learning rate from proxy models when the width is smaller than the input dimension it is important to rescale the lmo as max(1, dout /din).
These observations leads to the recommendations below.
Recommendation 3.1. We refer to the instantiation of uSCG and SCG using operator norms as UNCONSTRAINED SCION and SCION respectively (cf. Algorithm 3), which stands for Stochastic Conditional Gradient with Operator Norms. We recommend the following configurations of the layer norms (First layer → Intermediary layers → Last layer): (i) image domains: Spectral → Spectral → Sign (ii) 1-hot input: ColNorm → Spectral → Sign (iii) weight sharing: Sign → Spectral → Sign
The lmo names are defined in Table 2 and weight sharing refers to parameter sharing between the first and last layer. Each layer should be scaled appropriately according to Tables 3 and 4.
Other norm choices So far our argument has been based on the invariance provided by ∥ • ∥ RMS→RMS for intermediary layers: i.e., the RMS norm of the output of layer ℓ is bounded, so the input of next layer ℓ + 1 is also bounded in the RMS norm. Since the lmo of ∥ • ∥ RMS→RMS can be computed efficiently we can directly use this norm choice for our update rule. However, it is possible to choose another norm such as ∥ • ∥ 1→RMS , as long as the RMS norm guarantee on the output of W ℓ is converted into a guarantee on the ℓ 1 -norm of the input of layer ℓ + 1. Specifically, we have that ∥ • ∥ RMS→RMS ≤ d in ∥ • ∥ 1→RMS through Fact 3.1. Alternatively, we can rely on the invariance provided by ∥ • ∥ ∞→∞ , for which Fact 3.1 tells us that ∥•∥ ∞→∞ ≤ √ d in ∥•∥ RMS→∞ and ∥•∥ ∞→∞ ≤ d in ∥•∥ 1→∞ . We obtain methods exclusively relying on RowNorm, Col-Norm and Sign as summarized in Table 6 of Appendix B.
this section cite: []

Section: Hyperparameter Transfer
The intuition behind why (UNCONSTRAINED) SCION may enjoy hyperparameter transfer is suggested by the spectral scaling rule of Yang et al. (2023), which states that feature learning may be ensured by requiring that, for MLPs with weight matrices W ℓ ∈ R d ℓ-1 ×d ℓ , the following holds:
∥W ℓ ∥ S∞ = Θ d ℓ d ℓ-1 and ∥∆W ℓ ∥ S∞ = Θ d ℓ d ℓ-1
where ∥ • ∥ S∞ denotes the spectral norm and ∆W ℓ is the update change. For uSCG, the update change is given by x k+1 -x k = γ lmo(d k ), so the requirement is automatically satisfied by the spectral norm choice from Table 3.
We formalize this intuition in Lemma C.6 of Appendix C.2 following the proof technique of Yang et al. (2023), which holds for losses including logistic regression and MSE, and activation functions including ReLU, GELU and Tanh. Specifically, we show that the so-called maximal update learning rate γ * (i.e., the learning rate that enables the hidden layer preactivations to undergo the largest possible change in a single update step) is independent of width. Our theoretical analysis is conducted under a simplified setting with momentum α k = 1 in Algorithm 1, which we adopt to facilitate a clean derivation of the width-invariant property. Thus, a learning rate tuned on a smaller model can be directly applied to a wider model without compromising the maximal update property.
this section cite: ['b60', 'b60']

Section: Related Works
Hyperparameter transfer Yang & Hu (2021); Yang et al. (2022) showed that there exists a parameterization (i.e., a choice of initialization and layerwise stepsize scaling) for which the features in every single layer evolve in a width-independent manner. The so-called Maximal Update Parametrization (µP) allows transferring optimal hyperparameter from a small proxy model to a large model.
A relationship with the spectral norm was established in Yang et al. (2023). An operator norm perspective was taken in the modular norm framework of Large et al. (2024); Bernstein & Newhouse (2024a), which was used to show Lipschitz continuity with constants independent of width. We build on this perspective and propose the 1 → ∞ operator norm and RMS → ∞, which leads to a sign update rule and row normalization respectively.
Steepest descent in a normed space Steepest descent in a possibly non-Euclidean space can be written in terms of the lmo (cf., Appendix A.1) provided a stream of stochastic gradients (g k ) k∈N and an initialization x 0 ∈ X ,
x k+1 = x k -γ[g k ] ♯ = x k + γ ρ ∥g k ∥ * lmo(g k ),(7)
where [•] ♯ := arg max x∈X ⟨•, x⟩ -1 2 ∥x∥ 2 is the sharpoperator (Nesterov, 2012;Kelner et al., 2014).
The deterministic case is analyzed in Nesterov (2012); Kelner et al. (2014), and was extended to the stochastic case in Carlson et al. (2015b), with a particular empirically focus on the spectral norm, named as (preconditioned) stochastic spectral descent (SSD) (Carlson et al., 2015a;b;c). Their SSD algorithm is an instance of the Majorization-Minimization (MM) algorithms (Lange, 2016), which iteratively minimizes a locally tight upper bound. The dualization in Bernstein & Newhouse (2024a) is also motivated by the sharp-operator.
In contrast to ( 7), uSCG and SCG are invariant to the magnitude of the gradients and do not need to compute the dual norm ∥ • ∥ * , which cannot be computed independently across layers. Intuitively, the scale invariance of the lmo allows convergence to be established without knowledge of the Lipschitz constant L (cf. Theorems 5.4 to 5.7).
Unlike the lmo-based schemes, extending sharp-operatorbased algorithms to handle constrained problems is nontrivial even in the vector case (El Halabi, 2018). Additionally, a practical concern of using specifically spectral norm projections in deep learning is that the model weights themselves can be dense (so the required SVD would be expensive), while gradients used in the lmo are usually low-rank (allowing efficient SVD approximations).
this section cite: ['b58', 'b59', 'b60', 'b39', 'b46', 'b36', 'b46', 'b36', 'b38', 'b21']

Section: Muon
The Muon optimizer (Jordan et al., 2024b) is introduced as a steepest descent method. The implementation interestingly ignores the scaling ∥ • ∥ * appearing in the update (cf., ( 7)), so Muon is effectively using the lmo over the spectral norm instead of the sharp operator. Provided a stream of stochastic gradients (g k ) k∈N and an initialization x 0 ∈ X the Muon optimizer can then be written as follows
G k = g k + βG k-1 (Muon) x k+1 = x k + γ lmo(g k + βG k ) if Nesterov x k + γ lmo(G k ) otherwise
where the lmo corresponds implicitly to the spectral norm.
The accumulation G k can be written in terms of the averaged gradient d k in Algorithm 2. We have that d k = αG k by picking α = (1 -β). Since the lmo is scale invariant, d k and G k can be used interchangeably without changing the update. Thus, we can alternatively write Muon (with non-Nesterov based momentum) exactly as uSCG.
In practice Muon is only applied to hidden layers, thus excluding the first layer and the last layer for which Adam(W) or SGD is used. In contrast, we apply uSCG and SCG to all layers and demonstrate transferability of the stepsize.
this section cite: []

Section: Moonlight
Since the first version of this paper appeared on arXiv on the 11th of February, Liu et al. (2025) also proposed integrating Muon with weight decay to control the norm of the parameters. They use the same scaling factor of the lmo, max{d in , d out }, as the Muon baseline we compare against (cf. Appendix E.4).
this section cite: ['b41']

Section: MARS
The MARS-Shampoo optimizer (Yuan et al., 2024, Alg. 4) can be seen as an instance of SCG with spectral norm constraints, but using the STORM gradient estimator (Cutkosky & Orabona, 2019) instead of (3).
Sign SignSGD and the momentum variant Signum were brought to prominence and further analyzed in Bernstein et al. (2018) motivated by efficient communication for distributed optimization, while they are originally introduced with the dual norm scaling and used only for weight bias updates in Carlson et al. (2015a;b;c). These schemes are typically studied under the framework of steepest descent, which results in the ∥g k ∥ 1 stepsize scaling in (7) usually not present in practice as remarked in Balles et al. (2020).
Normalization The LARS optimizer (You et al., 2017) uses normalized gradient and was shown to be particularly useful for large batch settings. The method can be viewed as performing normalized SGD with momentum (Cutkosky & Mehta, 2020) layerwise with a particular adaptive parameter-dependent stepsize.
The layerwise normalization can be captured by uSCG with the norm choice max ℓ ∥W ℓ ∥ F . The LAMB optimizer (You et al., 2019) incorporates the update into an Adam-like structure. Zhao et al. (2020) considers averaging the normalized gradients rather than the gradients prior to normalization. The update can be written in terms of an lmo, with the (flattened) norm choice ∥x∥ 2 , which we generalize with a new algorithm in Appendix D.3 to arbitrary norms.
Continuous greedy With zero initialization, x 1 = 0, and stepsize γ k = γ = 1/n, uSCG recovers the stochastic continuous greedy method (Mokhtari et al., 2020;Vondrák, 2008), which can be used to solve DR-submodular maximization problems under Matroid polytope constraints.
LMO for deep learning SCG for training neural networks has been suggested in Pokutta et al. (2020) and Lu et al. (2022), where optimization was specifically constrained to the K-sparse polytope with increasing batchsize for handling stochasticity. Beyond these works, we provide convergence guarantees for SCG with constant batch-sizes and, introduce a principled framework for selecting effective constraints based on the input and output space geometries of the layers of the network.
The perturbation in the sharpness-aware minimization (SAM) has been interpreted as an lmo and generalized to arbitrary norms (Pethick et al., 2025), focusing on the maxnorm over nuclear norms, max ℓ ∥W ℓ ∥ S1 .
Scion can also be seen as an instantiation of the Lion-K (Chen et al., 2023) algorithm with ∂K chosen to be the lmo.
In this case, however, K is not smooth and the continuoustime analysis presented in Chen et al. (2023) and related works do not apply.
this section cite: ['b16', 'b7', 'b3', 'b61', 'b15', 'b62', 'b66', 'b45', 'b55', 'b49', 'b42', 'b48', 'b11', 'b11']

Section: Trust-region
The SCG method can be seen as a trustregion method with a linear surrogate. Usually, the surrogate is taken to be quadratic (cf. Wright (2006, Ch. 4)). We refer to Conn et al. (2000) for an extensive overview of trust-region methods.
Preconditioned SGD We also recognize the spectral lmo used in SCION as being related to the Preconditioned SGD (PSGD) family of algorithms (Li, 2017;Pooladzandi & Li, 2024) in the sense of whitening the update. In contrast to those methods, we do not keep track of an explicit preconditioner; we compute the lmo at each iteration instead.
this section cite: ['b14', 'b40', 'b50']

Section: Natural gradient
Early work on non-Euclidean methods considered measuring the "distance" between models through the Kullback-Leibler (KL) divergence between the output distributions of the models (Amari, 1998). Approximating the KL divergence through a Taylor expansion leads to a steepest descent method, which preconditions with the Fisher information matrix, known as the natural gradient method. Trust-region variants of the natural gradient method were considered with TRPO (Schulman et al., 2015) similarly to how uSCG can be seen as a trust-region variant of steepest descent.
this section cite: ['b0', 'b52']

Section: Analysis
We begin by presenting the two main assumptions we will make to analyze Algorithms 1 and 2. The first is an assumption on the Lipschitz-continuity of ∇f with respect to the norm ∥ • ∥ * restricted to X . We do not assume this norm to be Euclidean which means our results apply to the geometries relevant to training neural networks.
Assumption 5.1. The gradient ∇f is L-Lipschitz with L ∈ (0, ∞), i.e.,
∥∇f (x) -∇f (x)∥ * ≤ L∥x -y∥ ∀x, y ∈ X .(8)
Furthermore, f is bounded below by f ⋆ .
Remark 5.2. Strictly speaking, uSCG only needs Lipschitz continuity to hold locally within a radius γρ, since the assumption is only invoked between two consecutive iterates.
Our second assumption is that the stochastic gradient oracle we have access to is unbiased and has a bounded variance, a typical assumption in stochastic optimization.
Assumption 5.3. The stochastic gradient oracle ∇f (•, ξ) :
X → R d satisfies. (i) Unbiased: E ξ [∇f (x, ξ)] = ∇f (x) ∀x ∈ X . (ii) Bounded variance: E ξ ∥∇f (x, ξ) -∇f (x)∥ 2 2 ≤ σ 2 ∀x ∈ X , σ ≥ 0.
With these assumptions, we can state our worst-case convergence rates, first for Algorithm 1 and then for Algorithm 2.
To bridge the gap between theory and practice, we investigate these algorithms when run with a constant stepsize γ, which depends on the specified horizon n ∈ N * , and momentum which is either constant α ∈ (0, 1) (except for the first iteration where we take α = 1 by convention) or vanishing α k ↘ 0. All results can be extended to any time guarantees in a straightforward manner by choosing γ k as a function of the iteration counter k instead of horizon n and modifying the proofs accordingly.
The exact constants for the rates can be found in the proofs in Appendix D; we try to highlight the dependence on the parameters L and ρ, which correspond to the natural geometry of f and D, explicitly here. Our rates are nonasymptotic and use big O notation for brevity.
Theorem 5.4 (Convergence rate for uSCG with constant α). Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 1 with constant stepsize γ = 1 √ n and constant momentum α ∈ (0, 1). Then, it holds that
E[∥∇f (x n )∥ * ] ≤ O Lρ √ n + σ .
Theorem 5.5 (Convergence rate for uSCG with vanishing α k ). Suppose that Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 1 with a constant stepsize γ satisfying 1 2n 3/4 < γ < 1 n 3/4 and vanishing momentum
α k = 1 √ k . Then, it holds that E[∥∇f (x n )∥ * ] = O 1 n 1/4 + Lρ n 3/4 .
Insight 5.1. The uSCG algorithm remarkably does not require knowledge of the Lipschitz constant L, which is intuitively explained by viewing the method as a normalized version of steepest descent through the relationship lmo(•) = - [•] ♯ ∥•∥ * . Normalizing gradients with the dual norm has also been used in the online learning community to adapt to (local) Hölder smoothness as a simple alternative to AdaGrad-Norm (Orabona, 2023).
These results show that, in the worst-case, running Algorithm 1 with constant momentum α guarantees faster convergence but to a noise-dominated region with radius proportional to σ. In contrast, running Algorithm 1 with vanishing momentum α k is guaranteed to make the expected dual norm of the gradient small but at a slower rate. Algorithm 2 exhibits the analogous behavior, as we show next.
Before stating the results for Algorithm 2, we emphasize that they are with constant stepsize γ, which is atypical for conditional gradient methods. However, like most conditional gradient methods, we provide a convergence rate on the so-called Frank-Wolfe gap which measures criticality for the constrained optimization problem over D.
Finally, we remind the reader that the iterates of Algorithm 2 are always feasible for the set D by the design of the update and convexity of the norm ball D.
Theorem 5.6 (Convergence rate for SCG with constant α). Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 2 with constant stepsize γ = 1 √ n and constant momentum α ∈ (0, 1). Then, for all u ∈ D, it holds that
E[⟨∇f (x n ), xn -u⟩] = O Lρ 2 √ n + σ .
Theorem 5.7 (Convergence rate for SCG with vanishing α k ). Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 2 with a constant stepsize γ satisfying 1 2n 3/4 < γ < 1 n 3/4 and vanishing momentum α k = 1 √ k . Then, for all u ∈ D, it holds that
E[⟨∇f (x n ), xn -u⟩] = O 1 n 1/4 + Lρ 2 n 3/4 .
Insight 5.2. For both algorithms, our worst-case analyses for constant momentum suggest that tuning α requires balancing two effects. Making α smaller helps eliminate a constant term that is proportional to the noise level σ. However, if α becomes too small, it amplifies an O(1/ √ n) term and an O(σ/n) term. The stepsize γ must also align with the choice of momentum α; for vanishing α k the theory suggests a smaller constant stepsize like γ = 3 4(n 3/4 ) to ensure convergence.
this section cite: ['b47']

Section: Experiments
For computing the lmo of layers using a spectral norm constraint, we use the efficient implementation provided in Jordan et al. (2024b) of the Newton-Schultz iteration proposed in Bernstein & Newhouse (2024b). In this section, Muon (Jordan et al., 2024b) refers to the version used in practice, which uses AdamW for the first layer and last layer and Nesterov type momentum.
GPT We build on the excellent modded-nanogpt codebase (Jordan et al., 2024a), which makes the following modernizations to Karpathy (2023): rotary embeddings is used instead of positional embeddings, RMS norm is used instead of LayerNorm, and linear decay schedule instead of a cosine stepsize, and the ReLU 2 instead of GELU activation function (scaled according to Appendix E.3). SCION and UNCONSTRAINED SCION use the (Sign → Spectral → Sign) configuration with scaling factors in accordance with Tables 3 and 4. We train for 5100 iterations with a batchsize of 512 on the FineWeb dataset (see Table 7 regarding hyperparameters). In comparison with Adam, both Muon and (UNCONSTRAINED) SCION do not require learning rate warmup. We sweep over stepsizes and model width in Figure 1. 2 16 2 14 2 12 2 10 2 8 2 6 Learning Rate 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 Validation Loss Adam Width (Model Size) 512 (64M) 768 (124M) 1280 (300M) 2560 (1B) 2 16 2 14 2 12 2 10 2 8 Learning Rate 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 Muon Width (Model Size) 512 (64M) 768 (124M) 1280 (300M) 2560 (1B) 2 16 2 14 2 12 2 10 2 8 Learning Rate 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 Scion Width (Model Size) 512 (64M) 768 (124M) 1280 (300M) 2560 (1B) 2 16 2 14 2 12 2 10 2 8 Learning Rate 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 Unconstrained Scion Width (Model Size) 512 (64M) 768 (124M) 1280 (300M) 2560 (1B) Figure 1. Performance on NanoGPT with between 64M and 1B parameters. The optimal learning rate of SCION is invariant to width. 1000 2000 3000 4000 5000 6000 Batch Size 3.4 3.6 3.8 4.0 4.2 Minimum Validation Loss Method Adam Muon Scion Unconstrained Scion 0 50 100 150 200 250 300 Epoch 40 45 50 55 60 65 70 75 80 85 Accuracy AdamW Scion Accuracy=81.8% From Figure 1, we observe that the optimal stepsize of SCION and UNCONSTRAINED SCION transfer across model width as oppose to Adam and Muon. The 124M model size configuration in Figure 1 corresponds to one of the official speedrun entries of Jordan et al. (2024a), for which SCION has slightly lower validation loss (across 3 runs) than Muon. Even when Muon is tuned on the largest model size it achieves a validation loss of 2.988 in comparison with 2.984 of UNCONSTRAINED SCION.
Our methods completely remove the need for using Adam otherwise present in the Muon implementation, which permits an implementation that only requires storing one set of weights and one set of gradient (stored in half-precision) across all layers (see Appendix E.2). The experiments additionally demonstrates that our method works for weight sharing.
3B model Using the optimal configuration of the 124M parameter proxy model, we perform a large model experiment on a 3B parameter model, which also increases the depth. Specifically, we take the embedding dimension to be 2560 and the depth to be 36. We observe in Table 5 that UNCONSTRAINED SCION outperforms all other methods. The loss curve is provided in Figure 8 of Appendix E. Large batches To test the effect of large batches we fix the total number of tokens for the 124M parameter model and sweep over the batch sizes while rescaling the total number of steps accordingly. The stepsize γ is optimized over {2 -17 , 2 -16 , ..., 2 -5 } for each combination of batch size and optimizer. We observe that (UNCONSTRAINED) SCION is better at maintaining a low validation loss with increasing batch size than the baselines (cf., Figure 2).
For large batches, SCION achieves a significantly better validation loss than Muon. To assess the implication for training time, we perform an additional experiment for the large batch size of 6144, where we reduce the number of iterations until SCION matches the larger validation loss of Muon. We find that SCION can achieve the same validation loss as Muon with a 25% smaller wallclock time.
We provide two possible explanations for why SCION favors large batches: The treatment of the noise is tied to the Euclidean geometry, in order to exploit the unbiasedness of the stochastic oracle, as apparent from the analysis. Additionally, in the extreme case of a single sample, the gradients of the linear layers are rank-1 and all Schatten norms become equivalent (e.g., Frobenius and Spectral norm).
this section cite: ['b34']

Section: Image classification
We additionally test on vision transformers (ViT) on ImageNet and convolutional neural networks (CNN) on the CIFAR10 dataset using the configuration (Spectral → Spectral → Sign). The explicit control on the norm provided by SCION circumvents the need for the Frobenius norm normalization of the weights present in the CIFAR10 implementation of Muon (Jordan, 2024).
The results regarding ImageNet are shown in Figure 3 (cf. Appendix E.4 for details and experiments on CIFAR10).
this section cite: ['b31']

Section: Impact Statement
This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.
this section cite: []

Section: Training Deep Learning Models with Norm-Constrained LMOs A. Preliminaries

this section cite: []

Section: A.1. Relationship between steepest descent and uSCG
There are two prominent families of norm-based non-Euclidean method, namely the ones based on the lmo and the ones based on the sharp operator, both of which can be expressed in the terms of the Fenchel conjugate. The Fenchel conjugate of a proper, convex, and lower semicontinuous function h : X → R ∪ {∞} is defined as:
h * (s) = sup x∈X {⟨s, x⟩ -h(x)} ,
where s ∈ X . The subdifferential ∂h * is equivalent to the argmax of the conjugate operation, i.e.,
∂h * (s) = argmax x∈X {⟨s, x⟩ -h(x)} .
This follows from the Fenchel-Young inequality (see e.g. Bauschke & Lucet ( 2012)).
LMO The lmo is a special case when h is an indicator function of a convex set, i.e.,
lmo(s) = ∂h * (-s) with h(x) = ι D (x) := 0 x ∈ D +∞ otherwise
The lmo is commonly used for constrained minimization in e.g., CG since the operator ensure feasibility on the constrained set D. When D := {x | ∥x∥ ≤ ρ}, the lmo satisfies ⟨s, lmo(s)⟩ = -ρ∥s∥ * , which is central to the convergence proof of uSCG (see Lemma D.1).
this section cite: []

Section: Sharp operator
Another important example is the sharp operator (Nesterov, 2012;Kelner et al., 2014) defined as
s ♯ ∈ arg max x∈X {⟨s, x⟩ -1 2 ∥x∥ 2 }
for some norm ∥ • ∥, which can equivalently be written as
s ♯ ∈ ∂h * (s) with h(x) = 1 2 ∥x∥ 2 .
The sharp operator satisfies ⟨s, s ♯ ⟩ = ∥s ♯ ∥ 2 = ∥s∥ 2 * (Kelner et al., 2014, App. A.1). The sharp operator and lmo can be defined in terms of each other when D := {x | ∥x∥ ≤ ρ}, specifically
s ♯ = -1 ρ ∥s∥ * lmo(s)(9)
From ( 9) we see a clear distinction between the lmo and the sharp operator, namely that, while the lmo is scale invariant (i.e. lmo(a • s) = lmo(s) for a > 0) the sharp operator is not (since [a • s] ♯ = a[s] ♯ for a ∈ R).
Steepest descent Steepest descent in a normed space can be written in terms of the sharp operator as follows
x k+1 = x k -γ[∇f (x k )] ♯
with a stepsize γ > 0. From ( 9) it becomes apparent that uSCG can be seen as a normalized variant of steepest descent with momentum.
this section cite: ['b46', 'b36']

Section: B. Method

this section cite: []

Section: B.1. Input radius scaling
Based on the spectral norm perspective (Yang et al., 2023), which requires that ∥W ℓ ∥ S∞ = Θ( dout /din), one might be inclined to pick the initialization such that ∥W ℓ ∥ S∞ = dout /din is ensured exactly. This argument is indeed valid
Algorithm 3 (Unconstrained) Scion Input: Horizon n, init.
x 1 = (W 1 1 , ..., W 1 L ), d 0 = 0, momentum α k ∈ (0, 1], stepsize γ ∈ (0, 1), radii ρ i ∈ R + . 1: for k = 1, . . . , n -1 do 2: Sample ξ k ∼ P 3: d k ← α k ∇f (x k , ξ k ) + (1 -α k )d k-1 4: x k+1 ℓ ← x k ℓ + γρ ℓ lmo ∥•∥ α ℓ →β ℓ (d k ℓ ) if unconstrained (1 -γ)x k ℓ + γρ ℓ lmo ∥•∥ α ℓ →β ℓ (d k ℓ ) else ∀ℓ ∈ [L] Return x n
SCG and uSCG with the layerwise norm choice from ( 6). For simplicity we ignore biases. asymptotically, since input dimension is kept fixed. However, when the input dimension is larger than the output dimension, this does not lead to constant preactivations as demonstrated through a coordinate check (Yang & Hu, 2021) carried out in Figure 4.
Kaiming initialization (He et al., 2015) fortunately circumvents this problem. From random matrix theory we have that (Vershynin, 2018). So the Kaiming initialization, [W ℓ ] ij ∼ N (0, 1 /din), leads to ∥W ℓ ∥ S∞ ≈ 1 + dout /din, which prevents the preactivation from going to zero as d out → 0. Alternatively, one can simply choose ∥W ℓ ∥ S∞ = max(1, dout /din).
∥A∥ S∞ ≈ σ( √ d in + √ d out ) for A ij ∼ N (0, σ 2 )
Ensuring a correct norm scaling is particularly important for SCG and uSCG, since the scaling not only affects initialization but also the update rule itself. Specifically, if the methods were run with the norm bound choice ∥W ℓ ∥ S∞ ≤ dout /din for the input layer, then the issue in Figure 4 persists, due the lmo always lying on the boundary of the norm ball. The choice ∥W ℓ ∥ S∞ = max(1, dout /din) resolves this issue.
this section cite: ['b60', 'b58', 'b28', 'b54']

Section: B.2. Alternative norm choices
The primary argument in Section 3.1 for the norm choice is based on the invariance provided by the RMS → RMS operator norm: i.e., the RMS norm of the output of layer ℓ is bounded, so the input of next layer ℓ + 1 is also bounded in the RMS norm. Since the lmo of ∥ • ∥ RMS→RMS can be computed efficiently, we can directly use this norm choice for our update rule.
However, it is possible to choose another norm such as ∥ • ∥ 1→RMS , as long as the RMS norm guarantee on the output of W ℓ is converted into a guarantee on the ℓ 1 -norm of the input of layer ℓ + 1. Specifically, we have that ∥ • ∥ RMS→RMS ≤ d in ∥ • ∥ 1→RMS through Fact 3.1. Alternatively, we can rely on the invariance provided by ∥ • ∥ ∞→∞ , for which Fact 3.1 tells us that ∥
• ∥ ∞→∞ ≤ √ d in ∥ • ∥ RMS→∞ and ∥ • ∥ ∞→∞ ≤ d in ∥ • ∥ 1→∞ .
The resulting lmo choices for the three norm choices across all layers are summarized in Table 6.
Table 6. It is possible to use the same norm throughout the network if scaled appropriately. By not treating the network as a flattened vector, hyperparameter can transfer across model sizes (cf. Figure 6). We have made use of Fact 3.1 and (5) to derive the correct layerwise scaling. We assume that the image input dimension is smaller than the dout of the first layer (otherwise see Appendix B.1).
this section cite: []

Section: Weight norm (bias norm)
W
1 (1-hot encoded) W 1 (image domain) (W ℓ ) ℓ∈[2,...,L-1] W L b ℓ Spectral RMS → RMS (RMS) lmo - √ d out U V ⊤ -dout /dinU V ⊤ -bℓ ∥bℓ∥RMS ColNorm 1 → RMS (RMS) lmo col j (W 1 ) → - √ d out colj (W1) ∥ colj (W1)∥2 col j (W ℓ ) → - √ dout din colj (Wℓ) ∥ colj (Wℓ)∥2 -bℓ ∥bℓ∥RMS RowNorm RMS → ∞ (RMS) lmo row i (W 1 ) → -rowi(W1) ∥ rowi(W1)∥2 row i (W ℓ ) → - rowi(Wℓ) √ din∥ rowi(Wℓ)∥2 -bℓ ∥bℓ∥RMS Sign 1 → ∞ (∞) lmo -sign(W 1 ) -1 din sign(W ℓ ) -sign(b ℓ )
this section cite: []

Section: B.3. Boundary initialization
Semi-orthogonal Following Saxe et al. (2013), perform QR decomposition of a random matrix
G ij ∼ N (0, 1), ∀i, j G = QR Use Q ′ = Q sign(diag(R))
as the semi-orthogonal matrix as the initialization.
Column-wise normalized Gaussian As proposed in Large et al. (2024), initialize each column as follows
W ij ∼ N (0, 1), ∀i, j col j (W ) = col j (W ) ∥ col j (W )∥ 2 , ∀i
Row-wise normalized Gaussian Initialize each row as follows
W ij ∼ N (0, 1), ∀i, j row i (W ) = row i (W ) ∥ row i (W )∥ 2 , ∀j
Random sign W ij = +1 with probability 0.5 -1 with probability 0.5 ∀i, j
Each initialization should be scaled by the corresponding scaling of the lmo elementwise.
C. Proofs for Section 3 (Our Methods) C.1. Notation and detailed problem setting First, we introduce some notation and detailed problem settings. Consider an L-layer neural network with input dimension d 0 , hidden layer widths {d 1 , d 2 , . . . , d L-1 }, and output dimension d L . For any input data z ∈ R d0 , the forward pass of the network is defined as
f (1) (z) = W (1) z, h (1) (z) = σ f (1) (z) , f (ℓ) (z) = W (ℓ) h (ℓ-1) (z), h (ℓ) (z) = σ f (ℓ) (z) , ∀ℓ = 2, . . . , L -1, f (L) (z) = W (L) h (ℓ-1) (z).(10)
Here, W (ℓ) ∈ R d ℓ ×d ℓ-1 are the weight matrices of the network, and σ(•) is an element-wise activation function. We denote f (ℓ) (z) ∈ R d ℓ as the preactivation of the l-th layer and h (ℓ) (z) ∈ R d ℓ as the corresponding postactivation. The network output f (L) (z) is calculated as a linear transformation of the final hidden layer representation.
Below, we provide a brief overview of the specific assumptions for the input data and initialization. Then we explicitly define the spectral norm choice used in our analysis:
Assumption C.1 (Input data). Training samples (z, y) are drawn from a distribution P, where z ∈ R d0 and y ∈ R d L . We assume z has bounded second moments, i.e., ∥z∥ 2 2 < ∞. Assumption C.2 (Initialization schemes). The initialization satisfy the following condition for the stability of the maximal update learning rate (Yang et al., 2023):
∥W (ℓ) ∥ S∞ = Θ d ℓ d ℓ-1 .
this section cite: ['b51', 'b39', 'b60']

Section: Both semi-orthogonal initialization and (with high probability) Gaussian initialization satisfy this condition.
Assumption C.3 (Spectral norm choice for lmo). We adopt the layer-wise linear maximization oracles (lmos) from Tables 3 and 4. Specifically, for the first layer W (1) , we use 1) . For each intermediate layer W (ℓ) , ℓ ∈ {2, . . . , L}, we similarly set
lmo ∇ W (1) L = max 1, dout din U (1) V (1)⊤ , where W (1) = U (1) Λ (1) V (1)⊤ is the reduced SVD of W (
lmo ∇ W (ℓ) L = dout din U (ℓ) V (ℓ)⊤ , Remark C.4.
In all cases, these choices are consistent with the (Spectral → Spectral → Spectral) configuration from Tables 3 and 4. Our results will simultaneously hold for the (ColNorm → Spectral → Spectral) configuration, due to the equivalence under 1-hot encoded (cf. Insight 3.2).
To investigate how these neuron preactivations change after one step of Algorithm 1, we consider a batch size = 1 setting. We analyze the update dynamics of the network parameters under general loss functions L, including but not limited to mean squared error (MSE) and logistic loss.
We follow Yang & Hu (2021) which states that a good learning rate enables hidden layer preactivations to undergo the largest possible change in a single update step, while still avoiding divergence when the network width is large.
Let ∆f (ℓ) i (z) be the change in the preactivation of the i-th neuron in the l-th hidden layer after one step of Algorithm 1 on (z, y). The so-called "maximal update" heuristic requires that:
The maximal update learning rate γ * := the learning rate for which E ∆f
(ℓ) i (z) 2 ≃ 1,
with the expectation again taken over the initialization distribution.
this section cite: ['b58']

Section: C.2. Hyperparameter transfer
To begin with, we prove the following lemma that derives the lmo of the gradient in an L-layer neural network using the spectral norm choice from Table 3.
Lemma C.5 (Spectral lmo for the gradient with respect to W (ℓ) ). Consider an L-layer neural network with input dimension d 0 , hidden layer widths {d 1 , d 2 , . . . , d L-1 }, and output dimension d L . Training samples (z, y) are drawn from some distribution P, where z ∈ R d0 and y ∈ R d L . The network follows the forward pass (10). For convenience, we set f (0) = h (0) = z, making the notation consistent for all layers. The linear maximization oracle (lmo) over the scaled spectral norm ball,
D = W | ∥W∥ S∞ ≤ d ℓ d ℓ-1 , for ∇ W (ℓ) L(z, y), denoted as lmo(∇ W (ℓ) L(z, y)), is given by lmo(∇ W (ℓ) L(z, y)) = d ℓ d ℓ-1 dL(z,y) df (ℓ) h (ℓ-1) ⊤ dL(z,y) df (ℓ) 2 ∥h (ℓ-1) ∥ 2 .
Proof. First, we express the gradient with respect to W (ℓ) . By applying the chain rule we have
∇ W (ℓ) L(z, y) = dL(z, y) df (ℓ) h (ℓ-1) T .
We immediately have that the lmo is given as
lmo(∇ W (ℓ) L(z, y)) = d ℓ d ℓ-1 dL(z,y) df (ℓ) h (ℓ-1) ⊤ dL(z,y) df (ℓ) 2 ∥h (ℓ-1) ∥ 2 .
Now, we are equipped to state and prove Lemma C.6. We follow the proof technique of Yang et al. (2023) in our derivations. Lemma C.6 (Width-invariance of the maximal update learning rate). Consider an L-layer MLP with widths d 0 , d 1 , . . . , d L (where d 1 ≥ d 0 ), and assume d 0 is a fixed constant. Let its activation function σ have Lipschitz constant L σ and satisfy σ(0) = 0 (e.g., ReLU, Tanh, or GELU). Suppose:
(i) The input data (z, y) meets the requirements in Assumption C.1, (ii) The network is initialized according to Assumption C.2, and
(iii) Parameter updates use Algorithm 1 with the spectral-norm-based lmo described in Assumption C.3.
For various loss functions L (e.g., MSE, logistic), define the maximal update condition by
E ∆f (ℓ) i (z) 2 ≃ 1 for all ℓ ≤ L,where
∆f (ℓ) i (z)
is the change in the preactivation of the i-th neuron in the ℓ-th hidden layer after one update step. Unless all activations are simultaneously zero during training (which is highly unlikely in practice), the optimal learning rate γ * (under the setting α k = 1 in Algorithm 1) satisfying this condition is independent of the hidden-layer widths.
Proof. For Algorithm 1 with the setting α k = 1, we have through the spectral norm choice Assumption C.3 and Lemma C.5 that
∆f (ℓ) i (z) = d ℓ-1 j=1 ∆W (ℓ) i,j h (ℓ-1) j (z) + d ℓ-1 j=1 W (ℓ) i,j ∆h (ℓ-1) j (z) = γ d ℓ-1 j=1 lmo(∇ W (ℓ) L(z, y)) i,j h (ℓ-1) j (z) + (W (ℓ) ∆h (ℓ-1) ) i = γ d ℓ-1 j=1 d ℓ d ℓ-1 dL(z,y) df (ℓ) i dL(z,y) df (ℓ) 2 h (ℓ-1) j (z) ∥h (ℓ-1) (z)∥ 2 h (ℓ-1) j (z) + (W (ℓ) ∆h (ℓ-1) ) i = γ d ℓ d ℓ-1 dL(z,y) df (ℓ) i dL(z,y) df (ℓ) 2 ∥h (ℓ-1) (z)∥ 2 + (W (ℓ) ∆h (ℓ-1) ) i .
Unless these terms perfect cancel each other out, we have
E ∥∆f (ℓ) (z)∥ 2 2 = E d ℓ i=1 ∆f (ℓ) i (z) 2 ≃ E d ℓ i=1 γ 2 d ℓ d ℓ-1 dL(z,y) df (ℓ) 2 i dL(z,y) df (ℓ) 2 2 ∥h (ℓ-1) (z)∥ 2 2 + E d ℓ i=1 (W (ℓ) ∆h (ℓ-1) ) 2 i = γ 2 d ℓ d ℓ-1 E∥h (ℓ-1) (z)∥ 2 2 + E W (ℓ) ∆h (ℓ-1) 2 2 .
For the second term, we have
E∥∆h (ℓ-1) (z)∥ 2 2 ≤ L 2 σ E ∆f (ℓ-1) (z) 2 2 So E W (ℓ) ∆h (ℓ-1) 2 2 ≤ d ℓ d ℓ-1 L 2 σ E ∆f (ℓ-1) (z) 2 2
Recall that, under Assumption C.1, the input z has bounded second moments, i.e., ∥z∥ 2 2 < ∞. Consequently, we can treat ∥z∥ 2 2 , d 0 , and L σ as constants. Under these conditions, we have:
E ∥∆f (ℓ) (z)∥ 2 2 ≃ γ 2 d ℓ d ℓ-1 E∥h (ℓ-1) (z)∥ 2 2 + E W (ℓ) ∆h (ℓ-1) 2 2 , E W (ℓ) ∆h (ℓ-1) 2 2 = d ℓ d ℓ-1 O E ∆f (ℓ-1) (z) 2 2 , E∥∆h (ℓ-1) (z)∥ 2 2 = O E ∆f (ℓ-1) (z) 2 2 ,
Recall that, by Assumption C.1, the input z has bounded second moments, i.e., E[∥z∥ 2 2 ] < ∞. Focusing on the case ℓ = 1, we obtain
E ∥∆ f (1) (z)∥ 2 2 = γ 2 d 1 ∥z∥ 2 2 d 0 ≃ γ 2 d 1 .
Moreover, by the Assumption C.2 and leveraging the proof from Yang et al. (2023) (specifically, by Eq. ( 8) at initialization), we have:
E∥h (ℓ) (z)∥ 2 2 = Θ(d ℓ ) ∀ℓ ∈ [L -1] .
By induction, unless all activations are simultaneously zero during training (which is unlikely in practice), we have:
E ∥∆ f (ℓ) (z)∥ 2 2 ≃ γ 2 d ℓ .
By symmetry, we obtain:
E ∥∆ f (ℓ) i (z)∥ 2 2 = 1 d ℓ E ∥∆ f (ℓ) (z)∥ 2 2 ≃ γ 2 ∀i ∈ [d ℓ ] ,
where independent with the width for every hidden layer.
Remark C.7. The maximal update condition ensures that the network operates in a stable but maximally adaptive regime, balancing efficient learning and numerical stability. Remark C.8. As the hidden layer widths d 1 , . . . , d L-1 increase, the learning rate required to maintain the maximal update property remains unchanged, demonstrating width invariance in deep networks. Consequently, a learning rate tuned on a smaller model can be directly applied to a wider model without sacrificing training dynamics.
this section cite: ['b60', 'b60']

Section: D. Proofs for Section 5 (Analysis)
In this section we present the proofs of the main convergence results of the paper as well as some intermediary lemmas that we will make use of along the way. Throughout this section, we adopt the notation:
(stochastic gradient estimator error) λ k := d k -∇f (x k ) (diameter of D in ℓ 2 norm) D 2 := max x,y∈D ∥x -y∥ 2 (radius of D in ℓ 2 norm) ρ 2 := max x∈D ∥x∥ 2 (norm equivalence constant) ζ := max x∈X ∥x∥ * ∥x∥ 2 (Lipschitz constant of ∇f with respect to ∥•∥ 2 ) L 2 := inf{M > 0 : ∀x, y ∈ X , ∥∇f (x) -∇f (y)∥ 2 ≤ M ∥x -y∥ 2 }
We analyze each algorithm separately, although the analysis is effectively unified between the two, modulo constants. This is done in Appendices D.1 and D.2, respectively. Our convergence analysis proceeds in three steps: we begin by establishing a template descent inequality for each algorithm via the descent lemma. Next, we analyze the behavior of the second moment of the error E[ λ k 2 2 ] under different choices for α. Then, we combine these results to derive a convergence rate. Finally, we note that when analyzing algorithms with constant momentum, we will still always take α = 1 on the first iteration k = 1.
this section cite: []

Section: D.1. Convergence analysis of uSCG
We begin with the analysis of Algorithm 1 by establishing a generic template inequality for the dual norm of the gradient at iteration k. This inequality holds regardless of whether the momentum α k is constant or vanishing, as long as it remains in (0, 1].
Lemma D.1 (uSCG template inequality). Suppose Assumption 5.1 holds. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 1 with a constant stepsize γ > 0. Then we have
E[∥∇f (x n )∥ * ] ≤ E[f (x 1 ) -f ⋆ ] ργn + Lργ 2 + 1 n ρ 2 ρ + ζ n k=1 E[∥λ k ∥ 2 2 ].(11)
Proof. Under Assumption 5.1, we can use the descent lemma for the function f at the points x k and x k+1 to get, for all k ∈ {1, . . . , n},
f (x k+1 ) ≤ f (x k ) + ⟨∇f (x k ), x k+1 -x k ⟩ + L 2 x k+1 -x k 2 = f (x k ) + ⟨∇f (x k ) -d k , x k+1 -x k ⟩ + ⟨d k , x k+1 -x k ⟩ + L 2 x k+1 -x k 2 = f (x k ) + γ⟨∇f (x k ) -d k , lmo(d k )⟩ + γ⟨d k , lmo(d k )⟩ + Lγ 2 2 lmo(d k ) 2 ≤ f (x k ) + γρ 2 λ k 2 + γ⟨d k , lmo(d k )⟩ + Lγ 2 2 ρ 2 , (12
)
the final step employing Cauchy-Schwarz, the definition of λ k , and the definition of ρ 2 as the radius of D in the ∥•∥ 2 norm. By definition of the dual norm we have, for all u ∈ X ,
∥u∥ * = max v : ∥v∥≤1 ⟨u, v⟩ = max v∈D ⟨u, 1 ρ v⟩ = -⟨u, 1 ρ lmo(u)⟩
which means that, for all k ∈ {1, . . . , n},
γ⟨d k , lmo(d k )⟩ = γρ⟨d k , 1 ρ lmo(d k )⟩ = -γρ∥d k ∥ * .
Plugging this expression for γ⟨d k , lmo(d k )⟩ into (12) gives, for all k ∈ {1, . . . , n},
f (x k+1 ) ≤ f (x k ) + γρ 2 λ k 2 -γρ∥d k ∥ * + Lγ 2 2 ρ 2 = f (x k ) + γρ 2 λ k 2 -γρ∥d k -∇f (x k ) + ∇f (x k )∥ * + Lγ 2 2 ρ 2 (a) ≤ f (x k ) + γρ 2 λ k 2 + γρ∥λ k ∥ * -γρ∥∇f (x k )∥ * + Lγ 2 2 ρ 2 (b) ≤ f (x k ) + γ(ρ 2 + ζρ) λ k 2 -γρ∥∇f (x k )∥ * + Lγ 2 2 ρ 2 ,
applying the reverse triangle inequality in (a) while (b) stems from the definition of ζ. By rearranging terms and taking expectations, we get
γρE[ ∇f (x k ) * ] ≤ E[f (x k ) -f (x k+1 )] + γ (ρ 2 + ζρ) E[ λ k 2 ] + Lρ 2 γ 2 2 .
Summing this from k = 1 to n and dividing by γρn we get
E[∥∇f (x n )∥ * ] = 1 n n k=1 E[ ∇f (x k ) * ] ≤ E[f (x 1 ) -f (x n+1 )] ργn + Lργ 2 + 1 n ρ 2 ρ + ζ n k=1 E[ λ k 2 ] (a) ≤ E[f (x 1 ) -f ⋆ ] ργn + Lργ 2 + 1 n ρ 2 ρ + ζ n k=1 E[ λ k 2 ] (b) ≤ E[f (x 1 ) -f ⋆ ] ργn + Lργ 2 + 1 n ρ 2 ρ + ζ n k=1 E[∥λ k ∥ 2 2 ],
using the definition of f ⋆ for (a) and Jensen's inequality for (b).
At this point, we need to determine the growth of the induced error captured by the quantity λ k 2 2 . To estimate this, we first use a recursion relating E[ λ k 2 2 ] and E[ λ k-1 2 2 ] adapted from the proof in Mokhtari et al. (2020, Lem. 6) and then we prove a bound on the decay of λ k 2 2 for Algorithm 1. Lemma D.2 (Linear recursive inequality for E λ k 2 2 ). Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 1 with a constant stepsize γ > 0. Then, for all k ∈ {1, . . . , n},
E[ λ k 2 2 ] ≤ 1 - α k 2 E[ λ k-1 2 2 ] + 2L 2 2 ρ 2 2 γ 2 α k + α 2 k σ 2 . Proof.
The proof is a straightforward adaptation of the arguments laid out in Mokhtari et al. (2020, Lem. 6), which in fact do not depend on convexity nor on the choice of stepsize. Let n ∈ N * and k ∈ {1, . . . , n}, then
λ k 2 2 = ∇f (x k ) -d k 2 2 = ∇f (x k ) -α k ∇f (x k , ξ k ) -(1 -α k )d k-1 2 2 = α k ∇f (x k ) -∇f (x k , ξ k ) + (1 -α k ) ∇f (x k ) -∇f (x k-1 ) -(1 -α k ) d k-1 -∇f (x k-1 ) 2 2 = α 2 k ∇f (x k ) -∇f (x k , ξ k ) 2 2 + (1 -α k ) 2 ∇f (x k ) -∇f (x k-1 ) 2 2 + (1 -α k ) 2 ∇f (x k-1 ) -d k-1 2 2 + 2α k (1 -α k )⟨∇f (x k-1 ) -∇f (x k-1 , ξ k-1 ), ∇f (x k ) -∇f (x k-1 )⟩ + 2α k (1 -α k )⟨∇f (x k ) -∇f (x k , ξ k ), ∇f (x k-1 ) -d k-1 ⟩ + 2(1 -α k ) 2 ⟨∇f (x k ) -∇f (x k-1 ), ∇f (x k-1 ) -d k-1 ⟩.
Taking the expectation conditioned on the filtration F k generated by the iterates until k, i.e., the sigma algebra generated by {x 1 , . . . , x k }, which we denote using E k [•], and using the unbiased property in Assumption 5.3, we get,
E k [ λ k 2 2 ] = α 2 k E k [ ∇f (x k ) -∇f (x k , ξ k ) 2 2 ] + (1 -α k ) 2 ∇f (x k ) -∇f (x k-1 ) 2 2 + (1 -α k ) 2 λ k-1 2 2 + 2(1 -α k ) 2 ⟨∇f (x k ) -∇f (x k-1 ), λ k-1 ⟩. From this expression we can estimate, E k [ λ k 2 2 ] (a) ≤ α 2 k σ 2 + (1 -α k ) 2 ∇f (x k ) -∇f (x k-1 ) 2 2 + (1 -α k ) 2 λ k-1 2 2 + 2(1 -α k ) 2 ⟨∇f (x k ) -∇f (x k-1 ), λ k-1 ⟩ (b) ≤ α 2 k σ 2 + (1 -α k ) 2 ∇f (x k ) -∇f (x k-1 ) 2 2 + (1 -α k ) 2 λ k-1 2 2 + (1 -α k ) 2 α k 2 ∇f (x k ) -∇f (x k-1 ) 2 2 + 2 α k λ k-1 2 2 (c) ≤ α 2 k σ 2 + (1 -α k ) 2 L 2 2 x k -x k-1 2 2 + (1 -α k ) 2 λ k-1 2 2 + (1 -α k ) 2 ( α k 2 )L 2 2 x k -x k-1 2 2 + 2 α k λ k-1 2 2 (d) ≤ α 2 k σ 2 + (1 -α k ) 2 L 2 2 ρ 2 2 γ 2 + (1 -α k ) 2 λ k-1 2 2 + (1 -α k ) 2 ( α k 2 )L 2 2 ρ 2 2 γ 2 + 2 α k λ k-1 2 2 (e) ≤ α 2 k σ 2 + (1 + α k 2 )(1 -α k )L 2 2 ρ 2 2 γ 2 + (1 + 2 α k )(1 -α k ) λ k-1 2 2
, using the bounded variance property from Assumption 5.3 for (a), Young's inequality with parameter α k /2 > 0 for (b), the Lipschitz property of f under norm ∥ • ∥ 2 for (c), the update definition from Algorithm 1 for (d), and the fact that 1 -α k < 1 for (e). To complete the proof, we note that
(1 + 2 α k )(1 -α k ) ≤ (1 -α k 2 ) and (1 -α k )(1 + α k 2 ) ≤ 2 α k
which, applied to the previous inequality and taking total expectations, yields
E[ λ k 2 2 ] ≤ 1 - α k 2 E[ λ k-1 2 2 ] + α 2 k σ 2 + 2L 2 2 ρ 2 2 γ 2 α k .
this section cite: []

Section: D.1.1. CONSTANT α
Lemma D.3. Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 1 with constant stepsize γ > 0 and constant momentum α ∈ (0, 1) with the exception of the first iteration, where we take α = 1. Then, we have for all k ∈ {1, . . . , n}
E[∥λ k ∥ 2 2 ] ≤ √ 2L 2 ρ 2 γ α + √ α + 1 - α 2 k σ.
Proof. Let n ∈ N * , k ∈ {1, . . . , n}, and invoke Lemma D.2 to get
E[ λ k 2 2 ] ≤ 1 - α 2 E[ λ k-1 2 2 ] + 2L 2 2 ρ 2 2 γ 2 α + α 2 σ 2 .
Applying Lemma D.9 with β = α 2 and η =
2L 2 2 ρ 2 2 γ 2 α + α 2 σ 2 gives directly E[ λ k 2 2 ] ≤ 2L 2 2 ρ 2 2 γ 2 α 2 + ασ 2 + 1 - α 2 k E[ λ 1 2 2 ] ≤ 2L 2 2 ρ 2 2 γ 2 α 2 + α + 1 - α 2 k σ 2
after using Assumption 5.3 in the final inequality. Taking square roots and upper bounding then yields
E[∥λ k ∥ 2 2 ] ≤ √ 2L 2 ρ 2 γ α + √ α + 1 - α 2 k σ.
Theorem 5.4 (Convergence rate for uSCG with constant α). Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 1 with constant stepsize γ = 1 √ n and constant momentum α ∈ (0, 1). Then, it holds that
E[∥∇f (x n )∥ * ] ≤ O Lρ √ n + σ .
Proof. Let n ∈ N * ; we will first invoke Lemma D.1 and then we will estimate the error terms inside using Lemma D.2 under Assumptions 5.1 and 5.3. As shown in Lemma D.1,
E[∥∇f (x n )∥ * ] ≤ E[f (x 1 ) -f ⋆ ] ργn + Lργ 2n + 1 n ρ 2 ρ + ζ n k=1 E[∥λ k ∥ 2 2 ].(13)
By Lemma D.2 with Lemma D.9, we get
E[∥λ k ∥ 2 2 ] ≤ √ 2L 2 ρ 2 γ α + √ α + 1 - α 2 k σ which, if we sum from k = 1 to n, gives us n k=1 E[∥λ k ∥ 2 2 ] ≤ n √ 2L 2 ρ 2 γ α + n √ α + 1 -α 2 1 -1 -α 2 σ.
Plugging this estimate into Equation ( 13) gives
E[∥∇f (x n )∥ * ] ≤ E[f (x 1 ) -f ⋆ ] ργn + Lργ 2 + 1 n ρ 2 ρ + ζ n k=1 E[ λ k 2 ] ≤ E[f (x 1 ) -f ⋆ ] ργn + Lργ 2 + 1 n ρ 2 ρ + ζ n √ 2L 2 ρ 2 γ α + n √ α + 1 -α 2 1 -1 -α 2 σ = E[f (x 1 ) -f ⋆ ] ργn + Lργ 2 + ρ 2 ρ + ζ √ 2L 2 ρ 2 γ α + √ α + 1 -α 2 n(1 -1 -α 2 ) σ .(14)
Finally, by substituting γ = 1 √ n and noting f (x n+1 ) ≥ f ⋆ we arrive at
E[∥∇f (x n )∥ * ] ≤ E[f (x 1 ) -f ⋆ ] √ nρ + Lρ 2 √ n + ρ 2 ρ + ζ √ 2L 2 ρ 2 α √ n + √ α + 1 -α 2 n(1 -1 -α 2 ) σ = O 1 √ n + σ . D.1.2. VANISHING α k
Lemma D.4 (Bound on the gradient error with vanishing α). Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 1 with a constant stepsize γ satisfying
1 2n 3/4 < γ < 1 n 3/4 .(15)
Moreover, consider momentum which vanishes α k = 1 √ k . Then, for all k ∈ {1, . . . , n} the following holds
E[ λ k 2 2 ] ≤ 4σ 2 + 8L 2 2 ρ 2 2 √ k .(16)
Proof. Let k ∈ {1, . . . , n}, then by invoking the recursive inequality obtained in Lemma D.2 for E[ λ k 2 2 ] we have,
E[ λ k 2 2 ] ≤ 1 - α k 2 E[ λ k-1 2 2 ] + α 2 k σ 2 + 2L 2 2 ρ 2 2 γ 2 α k .(17)
Using the particular choice of γ given in the statement of the lemma,
1 2n 3/4 < γ < 1 n 3/4 ,(18)
as well as the choice of α k and the fact that n ≥ k, we get
E[ λ k 2 2 ] ≤ 1 - α k 2 E[ λ k-1 2 2 ] + α 2 k σ 2 + 2L 2 2 ρ 2 2 α k n 3/2 ≤ 1 - α k 2 E[ λ k-1 2 2 ] + α 2 k σ 2 + 2L 2 2 ρ 2 2 α k k 3/2 = 1 - 1 2 √ k E[ λ k-1 2 2 ] + σ 2 k + 2L 2 2 ρ 2 2 k = 1 - 1 2 √ k E[ λ k-1 2 2 ] + σ 2 + 2L 2 2 ρ 2 2 k .
Then, by applying Lemma D.10 with
u k = E[ λ k 2 2 ] and c = σ 2 + 2L 2 2 ρ 2 2 we readily obtain E[ λ k 2 2 ] ≤ 4σ 2 + 8L 2 2 ρ 2 2 √ k (19
) since Q as defined in Lemma D.10 is given by Q = max{E[ λ 1 2 2 ], 4σ 2 + 8L 2 2 ρ 2 2 } ≤ 4σ 2 + 8L 2 2 ρ 2 2 , which concludes our result.
Combining these results yields our accuracy guarantees for Algorithm 1 with vanishing α k , presented in the next lemma.
Theorem 5.5 (Convergence rate for uSCG with vanishing α k ). Suppose that Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 1 with a constant stepsize γ satisfying 1 2n 3/4 < γ < 1 n 3/4 and vanishing momentum
α k = 1 √ k . Then, it holds that E[∥∇f (x n )∥ * ] = O 1 n 1/4 + Lρ n 3/4 .
Proof. Let n ∈ N * , k ∈ {1, . . . , n}; by combining Lemma D.1 and Lemma D.4 we have
E[∥∇f (x n )∥ * ] (D.1) ≤ 2E[f (x 1 ) -f ⋆ ] ρn 1/4 + 2(ρ 2 + ζρ) n k=1 E[∥λ k ∥ 2 2 ] ρn + Lρ n 3/4 (D.4) ≤ 2E[f (x 1 ) -f ⋆ ] ρn 1/4 + 2(ρ 2 + ζρ) 4σ 2 + 8L 2 2 ρ 2 2 n k=1 1 k 1/4 ρn + Lρ n 3/4 ≤ 2E[f (x 1 ) -f ⋆ ] ρn 1/4 + 2(ρ 2 + ζρ) 4σ 2 + 8L 2 2 ρ 2 2 n k=1 1 k 1/4 ρn + Lρ n 3/4 .(20)
Using the integral test and noting that x → 1 x 1/4 is decreasing on R + , we can upper bound the sum in the right hand side as
n k=1 1 k 1/4 ≤ 1 + n 1 1 x 3/4 dx = 1 + 4 3 [x 3/4 ] n 1 = 1 + 4 3 (n 3/4 -1) = 4 3 n 3/4 - 1 3 ≤ 4 3 n 3/4 .
Inserting the above estimation into (20) we arrive at
E[∥∇f (x n )∥ * ] ≤ 2E[f (x 1 ) -f ⋆ ] ρn 1/4 + 8n 3/4 (ρ 2 + ζρ) 4σ 2 + 8L 2 2 ρ 2 2 3ρn + Lρ n 3/4 = 2E[f (x 1 ) -f ⋆ ] + 8 3 (ρ 2 + ζρ) 4σ 2 + 8L 2 2 ρ 2 2 ρn 1/4 + Lρ n 3/4 = O 1 n 1/4 + Lρ n 3/4
which is the claimed result.
this section cite: []

Section: D.2. Convergence analysis of SCG
In this section we will analyze the worst-case convergence rate of Algorithm 2. To do this, we will prove bounds on the expectation of the so-called Frank-Wolfe gap, max u∈D ⟨∇f (x), x -u⟩, which ensures criticality for the constrained optimization problem over D, i.e., for
x ⋆ ∈ D 0 = ∇f (x ⋆ ) + N D (x ⋆ ) ⇐⇒ max u∈D ⟨∇f (x ⋆ ), x ⋆ -u⟩ ≤ 0
where N D is the normal cone to the set convex D.
This next lemma characterizes the descent of Algorithm 2 for any stepsize γ and momentum α k in (0, 1]. Lemma D.5 (Nonconvex analog Mokhtari et al. (2020, Lem. 2)). Suppose Assumption 5.1 holds. Let n ∈ N * and consider the iterates {x k } k=1 n generated by Algorithm 2 with constant stepsize γ ∈ (0, 1]. Then, for all k ∈ {1, . . . , n}, for all u ∈ D, it holds
γE[⟨∇f (x k ), x k -u⟩] ≤ E[f (x k ) -f (x k+1 )] + D 2 γ E[∥λ k ∥ 2 2 ] + 2Lρ 2 γ 2 .(21)
Proof. Let n ∈ N * , then by Assumption 5.1 we can apply the descent lemma for the function f at the points x k and x k+1 to get, for all k ∈ {1, . . . , n},
f (x k+1 ) ≤ f (x k ) + ⟨∇f (x k ), x k+1 -x k ⟩ + L 2 ∥x k+1 -x k ∥ 2 = f (x k ) + ⟨d k , x k+1 -x k ⟩ + ⟨λ k , x k+1 -x k ⟩ + L 2 ∥x k+1 -x k ∥ 2 = f (x k ) + γ⟨d k , lmo(d k ) -x k ⟩ + γ⟨λ k , lmo(d k ) -x k ⟩ + L 2 γ 2 ∥ lmo(d k ) -x k ∥ 2 (a) ≤ f (x k ) + γ⟨d k , u -x k ⟩ + γ⟨λ k , lmo(d k ) -x k ⟩ + L 2 γ 2 ∥ lmo(d k ) -x k ∥ 2 = f (x k ) + γ⟨-λ k , u -x k ⟩ + γ⟨∇f (x k ), u -x k ⟩ + γ⟨λ k , lmo(d k ) -x k ⟩ + L 2 γ 2 ∥ lmo(d k ) -x k ∥ 2 = f (x k ) + γ⟨∇f (x k ), u -x k ⟩ + γ⟨λ k , lmo(d k ) -u⟩ + L 2 γ 2 ∥ lmo(d k ) -x k ∥ 2 (b) ≤ f (x k ) + γ⟨∇f (x k ), u -x k ⟩ + γ⟨λ k , lmo(d k ) -u⟩ + 2Lρ 2 γ 2 ,
using the optimality of lmo(d k ) for the linear minimization subproblem for (a) and the 2ρ upper bound on ∥ lmo(d k ) -x k ∥ for (b). Rearranging and estimating we find, for all k ∈ {1, . . . , n}, for all u ∈ D,
γ⟨∇f (x k ), x k -u⟩ (a) ≤ f (x k ) -f (x k+1 ) + γ∥λ k ∥ 2 ∥ lmo(d k ) -u∥ 2 + L 2 γ 2 ∥ lmo(d k ) -x k ∥ 2 (b) ≤ f (x k ) -f (x k+1 ) + D 2 γ∥λ k ∥ 2 + 2Lρ 2 γ 2
where we have used the Cauchy-Schwarz inequality in (a) and and bounded ∥ lmo(d k ) -x k ∥ 2 using the diameter of the set D with respect to the Euclidean norm, denoted D 2 , in (b). Taking the expectation of both sides and applying Jensen's inequality we finally arrive, for all k ∈ {1, . . . , n}, for all u ∈ D,
γE[⟨∇f (x k ), x k -u⟩] ≤ E[f (x k ) -f (x k+1 )] + D 2 γE[∥λ k ∥ 2 ] + 2Lρ 2 γ 2 ≤ E[f (x k ) -f (x k+1 )] + D 2 γ E[∥λ k ∥ 2 2 ] + 2Lρ 2 γ 2 .
this section cite: []

Section: D.2.1. SCG WITH CONSTANT α
Lemma D.6. Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 2 with constant stepsize γ = 1 √ n and constant momentum α ∈ (0, 1) with the exception of the first iteration, where we take α = 1. Then we have
E[ λ k 2 2 ] ≤ 4L 2 2 D 2 2 γ 2 α 2 + 2α + 1 - α 2 k σ 2 .
Proof. Under Assumptions 5.1 and 5.3, Lemma 1 in Mokhtari et al. (2020) yields, after taking expectations, for all k ∈ {1, . . . , n}
E[∥λ k+1 ∥ 2 2 ] ≤ (1 - α k+1 2 )E[∥λ k ∥ 2 2 ] + σ 2 α 2 k+1 + 2L 2 2 D 2 2 γ 2 α k+1 .
Taking γ and α to be constant we get
E[∥λ k+1 ∥ 2 2 ] ≤ (1 - α 2 )E[∥λ k ∥ 2 2 ] + σ 2 α 2 + 2L 2 2 D 2 2 γ 2 α .
Applying Lemma D.9 to the above with
u k = E[∥λ k+1 ∥ 2 2 ], β = α 2 , and η = σ 2 α 2 + 2L 2 2 D 2 2 γ 2 α we obtain E[ λ k 2 2 ] ≤ 2ασ 2 + 4L 2 2 D 2 2 γ 2 α 2 + 1 - α 2 k E[ λ 1 2 2 ] ≤ 4L 2 2 D 2 2 γ 2 α 2 + 2α + 1 - α 2 k σ 2
with the final inequality following by the variance bound in Assumption 5.3.
Theorem 5.6 (Convergence rate for SCG with constant α). Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 2 with constant stepsize γ = 1 √ n and constant momentum α ∈ (0, 1). Then, for all u ∈ D, it holds that
E[⟨∇f (x n ), xn -u⟩] = O Lρ 2 √ n + σ .
Proof. Let n ∈ N * and let k ∈ {1, . . . , n}. By Assumption 5.1, we can invoke Lemma D.5 to get, for all k ∈ {1, . . . , n}, for all u ∈ D,
γE[⟨∇f (x k ), x k -u⟩] ≤ E[f (x k ) -f (x k+1 )] + D 2 γ E[∥λ k ∥ 2
2 ] + 2Lρ 2 γ 2 . Since Assumption 5.3 holds, we can then invoke Lemma D.6 and apply this to the above. This gives, for all u ∈ D
γE[⟨∇f (x k ), x k -u⟩] ≤ E[f (x k ) -f (x k+1 )] + 2Lρ 2 γ 2 + D 2 γ 4L 2 2 D 2 2 γ 2 α 2 + 2α + 1 - α 2 k σ 2 ≤ E[f (x k ) -f (x k+1 )] + 2Lρ 2 γ 2 + 2L 2 D 2 2 γ 2 α + D 2 γ √ 2α + 1 - α 2 k σ.
Summing from k = 1 to n then dividing by nγ we find, for all u ∈ D,
E[⟨∇f (x n ), xn -u⟩] = 1 n n k=1 E[⟨∇f (x k ), x k -u⟩] (a) ≤ E[f (x 1 ) -f (x n+1 )] γn + 2Lρ 2 γ + 2L 2 D 2 2 γ α + D 2 √ 2α + 1 n n k=1 1 - α 2 k σ (b) ≤ E[f (x 1 ) -f (x n+1 )] γn + 2Lρ 2 γ + 2L 2 D 2 2 γ α + D 2 √ 2α + 1 -α 2 n 1 -1 -α 2 σ (c) ≤ E[f (x 1 ) -f ⋆ ] γn + 2Lρ 2 γ + 2L 2 D 2 2 γ α + D 2 √ 2α + 1 -α 2 n 1 -1 -α 2 σ,(22)
applying the subadditivity of the square root for (a), geometric series due to 1 -α 2 ∈ (0, 1) for (b), and the definition of f ⋆ for (c). Taking γ = 1 √ n then gives the final result, for all u ∈ D,
E[⟨∇f (x n ), xn -u⟩] ≤ E[f (x 1 ) -f ⋆ ] √ n + 2Lρ 2 √ n + 2L 2 D 2 2 α √ n + D 2 √ 2α + 1 -α 2 n 1 -1 -α 2 σ = O Lρ 2 √ n + σ .
this section cite: ['b45']

Section: D.2.2. SCG WITH VANISHING α
We now proceed to analyze the convergence of Algorithm 2 with vanishing α k . The next lemma provides an estimation on the decay of the second moment of the noise λ k .
Lemma D.7 (Bound on the gradient error with vanishing α Algorithm 2). Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 2 with a constant stepsize γ satisfying
1 2n 3/4 < γ < 1 n 3/4 .(23)
Moreover, consider vanishing momentum α k = 1 √ k . Then, for all k ∈ {1, . . . , n} the following holds
E[ λ k 2 2 ] ≤ 4σ 2 + 8L 2 2 D 2 2 √ k .(24)
Proof. Under Assumptions 5.1 and 5.3, we have the following recursion from Lemma 1 in Mokhtari et al. (2020) after taking expectations, for all k ∈ N * ,
E[∥λ k+1 ∥ 2 2 ] ≤ (1 - α k+1 2 )E[∥λ k ∥ 2 2 ] + σ 2 α 2 k+1 + 2L 2 2 D 2 2 γ 2 α k+1 .
Comparing with the bound in Lemma D.4, we see the only difference is the change of the constant D 2 2 by ρ 2 2 . Repeating the argument in Lemma D.4, the desired claim is directly obtained with D 2 2 in place of ρ 2 2 , with the constant
Q = max{E[ λ 1 2 2 ], 4σ 2 + 8L 2 2 D 2 2 } ≤ 4σ 2 + 8L 2 2 D 2 2 since E[ λ 1 2 2 ] ≤ σ 2 by Assumption 5.3.
Theorem 5.7 (Convergence rate for SCG with vanishing α k ). Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 2 with a constant stepsize γ satisfying 1 2n 3/4 < γ < 1 n 3/4 and vanishing momentum α k = 1 √ k . Then, for all u ∈ D, it holds that
E[⟨∇f (x n ), xn -u⟩] = O 1 n 1/4 + Lρ 2 n 3/4 .
Proof. Let n ∈ N * and k ∈ {1, . . . , n}. By Assumption 5.1, we can invoke Lemma D.5 to get,
γE[⟨∇f (x k ), x k -u⟩] ≤ E[f (x k ) -f (x k+1 )] + D 2 γ E[∥λ k ∥ 2 2 ] + 2Lρ 2 γ 2 .
Applying the estimate given in Lemma D.7 to the above we get
γE[⟨∇f (x k ), x k -u⟩] ≤ E[f (x k ) -f (x k+1 )] + D 2 γ 4σ 2 + 8L 2 2 D 2 2 √ k + 2Lρ 2 γ 2 = E[f (x k ) -f (x k+1 )] + D 2 4σ 2 + 8L 2 2 D 2 2 γ 1 k 1/4 + 2Lρ 2 γ 2 .
Summing from k = 1 to n and then dividing by nγ we find, for all u ∈ D,
E[⟨∇f (x n ), xn -u⟩] = 1 n n k=1 E[⟨∇f (x k ), x k -u⟩] (a) ≤ E[f (x 1 ) -f (x n+1 )] nγ + D 2 4σ 2 + 8L 2 2 D 2 2 n n k=1 1 k 1/4 + 2Lρ 2 γ (b) ≤ E[f (x 1 ) -f (x n+1 )] nγ + 4D 2 4σ 2 + 8L 2 2 D 2 2 n 3/4 3n + 2Lρ 2 γ = E[f (x 1 ) -f (x n+1 )] nγ + 4D 2 4σ 2 + 8L 2 2 D 2 2 3n 1/4 + 2Lρ 2 γ,
using division by γn for (a) and the integral test with decreasing function x →foot_2 x 1/4 for (b). Using the definition of f ⋆ and estimating nγ > n 1/4 2 and γ < 1 n 3/4 gives
E[⟨∇f (x n ), xn -u⟩] ≤ 2E[f (x 1 ) -f ⋆ ] n 1/4 + 4D 2 4σ 2 + 8L 2 2 D 2 2 3n 1/4 + 2Lρ 2 n 3/4 = O 1 n 1/4 + Lρ 2 n 3/4 .
this section cite: ['b45']

Section: D.3. Averaged LMO Directional Descent (ALMOND)
In this section we present a variation on Algorithm 1 that computes the lmo directly on the stochastic gradient oracle and then does averaging. This is in contrast to how we have presented Algorithm 1 which first does averaging (aka momentum) with the stochastic gradient oracle and then computes the lmo. A special case of this algorithm is the Normalized SGD based algorithm of Zhao et al. (2020) when the set D is with respect to the Euclidean norm. In contrast with Algorithm 1, the method relies on large batches, since the noise is not controlled by the momentum parameter α due to the bias introduced by the lmo.
this section cite: ['b66']

Section: Algorithm 4 Averaged LMO directioNal Descent (ALMOND)
Input: Horizon n, initialization x 1 ∈ X , d 0 = 0, momentum α ∈ (0, 1), stepsize γ ∈ (0, 1)
1: for k = 1, . . . , n do 2: Sample ξ k ∼ P 3: d k ← α lmo(∇f (x k , ξ k )) + (1 -α)d k-1 4: x k+1 ← x k + γd k 5: Choose xn uniformly at random from {x 1 , . . . , x n } Return xn
Lemma D.8. Suppose Assumptions 5.1 and 5.3 hold. Let n ∈ N * and consider the iterates {x k } n k=1 generated by Algorithm 4 with stepsize γ = 1 √ n . Then, it holds
E[∥∇f (x n )∥ * ] ≤ E[f (x 1 ) -f ⋆ ] ρ √ n + L(1 -α)ρ α √ n + Lρ 2 √ n + 2µσ = O 1 √ n + 2µσ where 1 µ = max x∈X ∥x∥ * ∥x∥ 2 .
Proof. Let n ∈ N * and denote z k = 1 α x k -1-α α x k-1 with the convention that x 0 = x 1 so that z 1 = x 1 and, for all k ∈ {1, . . . , n},
z k+1 -z k = 1 α x k+1 - 1 -α α x k - 1 α x k + 1 -α α x k-1 = 1 α γd k -γ(1 -α)d k-1 = γ lmo(g k ).
Applying the descent lemma for f at the points z k+1 and z k gives
f (z k+1 ) ≤ f (z k ) + ⟨∇f (z k ), z k+1 -z k ⟩ + L 2 z k+1 -z k 2 = f (z k ) + γ⟨∇f (z k ), lmo(g k )⟩ + Lγ 2 2 lmo(g k ) 2 = f (z k ) + γ ⟨∇f (z k ) -∇f (x k ), lmo(g k )⟩ + ⟨∇f (x k ) -g k , lmo(g k )⟩ + ⟨g k , lmo(g k )⟩ + Lγ 2 2 lmo(g k ) 2 = f (z k ) + γ ⟨∇f (z k ) -∇f (x k ), lmo(g k )⟩ + ⟨∇f (x k ) -g k , lmo(g k )⟩ -ρ g k * + Lγ 2 2 lmo(g k ) 2 (a) ≤ f (z k ) + γ ∇f (z k ) -∇f (x k ) * + ∇f (x k ) -g k * lmo(g k ) -ρ g k * + Lγ 2 2 lmo(g k ) 2 (b) ≤ f (z k ) + γ ρ ∇f (z k ) -∇f (x k ) * + ∇f (x k ) -g k * -ρ g k * + Lρ 2 γ 2 2 (c) ≤ f (z k ) + γ ρ L z k -x k + ∇f (x k ) -g k * -ρ g k * + Lρ 2 γ 2 2 ,(25)
applying Hölder's inequality with norm ∥•∥ * for (a), the radius ρ of D for (b), and Assumption 5.1 for (c). We note that
x k+1 -x k = γd k = γ (1 -α)d k-1 + α lmo(g k ) = αγ lmo(g k )+(1-α)γ x k -x k-1 γ = αγ lmo(g k )+(1-α)(x k -x k-1 )
which we can use to bound
x k -x k-1 ≤ (1 -α) x k -x k-1 + αγ lmo(g k ) ≤ (1 -α) x k -x k-1 + αργ ≤ αργ (1 -α)
.
We then have
z k -x k = (1 -α) α x k -x k-1 ≤ (1 -α)ργ α
by using the definition of the update and the lmo, which can be plugged into (25) to get
ργ g k * ≤ f (z k ) -f (z k+1 ) + γρ L z k -x k + ∇f (x k ) -g k * + Lρ 2 γ 2 2 =⇒ g k * (a) ≤ f (z k ) -f (z k+1 ) ργ + L z k -x k + ∇f (x k ) -g k * + Lργ 2 (b) ≤ f (z k ) -f (z k+1 ) ργ + L(1 -α)ργ α + ∇f (x k ) -g k * + Lργ 2 =⇒ ∇f (x k ) * (c) ≤ (f (z k ) -f (z k+1 ) ργ + L(1 -α)ργ α + 2 ∇f (x k ) -g k * + Lργ 2(26)
where (a) is the result of dividing both sides by ργ, (b) is the result of bounding z k -x k , and (c) follows by the reverse triangle inequality after adding and subtracting ∇f (x k ) in the norm on the left hand side. Taking expectations, using Assumption 5.3 and the constant µ = max
x∈X ∥x∥ * ∥x∥ 2 , it holds E[ ∇f (x k ) -g k * ] ≤ µE[ ∇f (x k ) -g k 2 ] ≤ µ E[∥∇f (x k ) -g k ∥ 2 2 ] ≤ µσ which we can sum from k = 1 to n to obtain n k=1 E[ ∇f (x k ) * ] ≤ E[f (z 0 ) -f (z n+1 )] ργ + nL(1 -α)ργ α + 2nµσ + nLργ 2 .
Diving both sides by n and then plugging in γ = 1 √ n yields the desired final result.
this section cite: []

Section: D.4. Linear recursive inequalities
We now present two elementary lemmas that establish bounds for linear recursive inequalities. These results are essential for analyzing the convergence behavior of our stochastic gradient estimator, particularly when examining the error term E[ λ k 2 2 ]. Lemma D.9 (Linear recursive inequality with constant coefficients). Let n > 1 and consider {u k } n k=1 ∈ R n + a sequence of nonnegative real numbers satisfying, for all k ∈ {2, . . . , n},
u k ≤ (1 -β)u k-1 + η
with η > 0 and β ∈ (0, 1). Then, for all k ∈ {2, . . . , n}, it holds
u k ≤ η β + (1 -β) k u 1 .
Proof. We prove the claim by induction on k. For the base case k = 2 we find
u 2 ≤ (1 -β)u 1 + η ≤ η β + (1 -β)u 1
since β < 1. Assume now for some k ∈ {2, . . . , n} that the claim holds. Then, by the assumed recursive inequality on {u i } n i=1 , we have
u k+1 ≤ (1 -β)u k + η ≤ (1 -β) η β + (1 -β) k u 1 + η = (1 -β) k+1 u 1 + 1 -β β + 1 η = (1 -β) k+1 u 1 + η β
and thus the desired claim holds by induction.
The first lemma establishes a geometric decay bound for sequences with constant momentum. The following lemma extends this analysis to the case of variable coefficients, which we will use when we analyze Algorithm 1 and Algorithm 2 with vanishing momentum α k .
Lemma D.10 (Linear recursive inequality with vanishing coefficients). Let {u k } k∈N * be a sequence of nonnegative real numbers satisfying, for all k ∈ N * , the following recursive inequality
u k ≤ 1 - 1 2 √ k u k-1 + c k
where c > 0 is constant. Then, the sequence {u k } k∈N * satisfies, for all k ∈ N * ,
u k ≤ Q √ k with Q = max{u 1 , 4c}.
Proof. We prove the claim by induction. For k = 1 the inequality holds by the definition of Q, since
u 1 ≤ Q = Q √ 1 . Let k > 1 and assume that u k-1 ≤ Q √ k -1 .
Then, by the assumed recursive inequality for u k , we have
u k ≤ 1 - 1 2 √ k u k-1 + c k ≤ 1 - 1 2 √ k Q √ k -1 + c k .(27)
Since k > 1, we can estimate
1 √ k -1 = √ k k(k -1) = 1 √ k k k -1 = 1 √ k 1 + 1 k -1 ≤ 1 √ k 1 + 1 2(k -1)
which, when applied to ( 27), gives
u k ≤ 1 - 1 2 √ k 1 + 1 2(k -1) Q √ k + c k .(28)
Furthermore, as k > 1, we also have
1 - 1 2 √ k 1 + 1 2(k -1) ≤ 1 - 1 4 √ k .
Applying the above to ( 28)gives
u k ≤ 1 - 1 4 √ k Q √ k + c k = Q √ k + c -Q/4 k ≤ Q
√ k with the last inequality following since Q ≥ 4c. The desired claim is therefore obtained by induction.
this section cite: []

Section: E. Experiments

this section cite: []

Section: E.1. Additional experiments MLP
We consider a 3-layer MLP with ReLU activations to demonstrate the various output layers in Table 3. We consider the configuration (Spectral → Spectral → X) where X is the output layer. Hyperparameters are provided in Table 9. We observe in Figure 5 that the optimal learning rate transfers across model width for all output layer configurations.
Shallow GPT We consider a 3-layer GPT model (Karpathy, 2023) with the same modernizations as for the deep GPT in Section 6. We additionally remove the weight sharing between the first and last layer so that the various input layers from Table 4 can be investigated. We consider SCION and UNCONSTRAINED SCION with the configuration (X → Spectral → Sign) where X sweeps over the possible input layer lmos. We additionally consider the variant of UNCONSTRAINED SCION using the configuration (Sign → Sign → Sign), which is useful for distributed settings. The hyperparameters can be found in Table 8. We observe in Figure 7 that all configurations exhibit transferability of the optimal stepsize across layer width.
this section cite: ['b34']

Section: E.2. Implementation details
It is possible to implement SCG and uSCG, while only storing on set of parameter and one set of gradients (possibly stored in half-precision). For concreteness, we focus on SCG, but the reasoning applies to uSCG as well. Due to the scale invariance of the lmo, uSCG can be equivalently written as
G k = (1 -α)G k-1 + ∇f (x k , ξ k ) x k+1 = x k + γ k lmo(G k )
By rearranging the update, it suffice to maintain only two states:
G ← G + ∇f (x, ξ) (backpropagation) x ← x + γ lmo(G) G ← (1 -α)G
Implementation wise this approach relies on storing the averaged gradient at the memory location where backpropagation is accumulating the gradient. Thus, it is important not to zero out the gradient at any point during training. We provide a reference implementation in PyTorch referred to as ScionLight.
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 Momentum -9 -8 -7 -6 -5 -4 -3 -2 -1 0 log2 Learning Rate 40.7 40.2 40.2 39.2 38.7 38.1 37.8 36.6 36.3 36.1 53.8 54.0 52.8 52.2 51.0 50.8 49.7 49.2 48.3 47.0 69.6 69.2 68.1 66.8 65.5 64.6 63.5 63.0 61.5 60.8 82.3 81.1 79.8 78.7 77.7 76.7 75.5 74.4 73.5 72.1 88.2 87.4 86.6 85.8 84.9 84.0 83.1 82.5 81.6 80.3 90.9 90.6 90.2 89.8 89.1 88.7 88.0 87.4 86.7 85.7 92.1 92.2 92.2 92.1 91.7 91.4 91.1 90.8 90.3 89.5 91.3 92.5 92.8 92.7 92.7 92.6 92.6 92.3 92.2 91.8 75.4 86.7 89.5 91.2 91.8 91.9 92.1 92.2 92.3 91.9 10.0 10.0 10.0 10.0 10.0 10.0 10.0 40.0 48.1 80.7 Unconstrained Scion (epochs=8) 10 20 30 40 50 60 70 80 90 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 Momentum -9 -8 -7 -6 -5 -4 -3
-2 -1 0 log2 Learning Rate 52.6 53.1 51.7 50.8 50.3 49.8 48.5 47.8 47.4 45.9 70.7 69.3 67.9 66.9 65.6 64.7 63.5 62.5 61.3 60.2 83.3 81.5 80.3 79.1 78.0 76.7 75.6 74.6 73.5 72.3 88.9 87.9 86.9 86.0 85.0 84.2 83.5 82.6 81.6 81.0 91.5 90.9 90.2 89.8 89.3 88.9 88.2 87.5 87.1 86.3 92.7 92.5 92.2 91.8 91.7 91.4 91.0 90.7 90.3 89.9  0 25 50 75 100 125 150 175 200 Epoch 40 45 50 55 60 65 70 75 80 85 Accuracy Scion Unconstrained Scion Figure 12. Test accuracy curve on the DeiT-base model. The more stringent norm control of SCION is beneficial, similar to what is observed for the CIFAR10 experiments.
this section cite: []

Section: References
Ref_id:b0 Title: Natural gradient works efficiently in learning Year: (1998)
Ref_id:b1 Title: Lower bounds for nonconvex stochastic optimization Year: (2022)
Ref_id:b2 Title: Wasserstein generative adversarial networks Year: (2017)
Ref_id:b3 Title: The geometry of sign gradient descent Year: (2020)
Ref_id:b4 Title: Spectrally-normalized margin bounds for neural networks Year: (2012)
Ref_id:b5 Title: Modular duality in deep learning Year: (2024)
Ref_id:b6 Title: new norm: An anthology Year: (2024)
Ref_id:b7 Title: Compressed optimisation for non-convex problems Year: (2018)
Ref_id:b8 Title: Stochastic spectral descent for restricted Boltzmann machines Year: (2015)
Ref_id:b9 Title: Stochastic spectral descent for discrete graphical models Year: (2015)
Ref_id:b10 Title: Preconditioned spectral descent for deep learning Year: (2015)
Ref_id:b11 Title: Lion secretly solves constrained optimization: As lyapunov predicts Year: (2023)
Ref_id:b12 Title: Parseval networks: Improving robustness to adversarial examples Year: (2017)
Ref_id:b13 Title: sparse greedy approximation, and the frank-wolfe algorithm Year: (2010-09)
Ref_id:b14 Title: Trust region methods Year: (2000)
Ref_id:b15 Title: Momentum improves normalized SGD Year: (2020)
Ref_id:b16 Title: Momentum-based variance reduction in non-convex sgd Year: (2019)
Ref_id:b17 Title: Benchmarking neural network training algorithms Year: (2023)
Ref_id:b18 Title: Why do we need weight decay in modern deep learning? arXiv preprint Year: (2023)
Ref_id:b19 Title: The road less scheduled Year: (2024)
Ref_id:b20 Title: Adaptive subgradient methods for online learning and stochastic optimization Year: (2011)
Ref_id:b21 Title: Learning with structured sparsity: From discrete to convex and back Year: (2018)
Ref_id:b22 Title: The duality structure gradient descent algorithm: analysis and applications to neural networks Year: (2017)
Ref_id:b23 Title: An algorithm for quadratic programming Year: (1956)
Ref_id:b24 Title: Stochastic gradient methods with layer-wise adaptive moments for training of deep networks Year: (2019)
Ref_id:b25 Title: A unified approach to adaptive regularization in online and stochastic optimization Year: (2017)
Ref_id:b26 Title: Projection-free online learning Year: (2012)
Ref_id:b27 Title: Beyond convexity: Stochastic quasi-convex optimization Year: (2015)
Ref_id:b28 Title: Delving deep into rectifiers: Surpassing human-level performance on imagenet classification Year: (2015)
Ref_id:b29 Title: Neural networks for machine learning lecture 6a overview of minibatch gradient descent Year: (2012)
Ref_id:b30 Title: Revisiting Frank-Wolfe: Projection-free sparse convex optimization Year: (2013)
Ref_id:b31 Title: Cifar-10 airbench Year: (2024)
Ref_id:b32 Title: and @Grad62304977. moddednanogpt: Speedrunning the nanogpt baseline, 2024a Year: ()
Ref_id:b33 Title: An optimizer for hidden layers in neural networks Year: (2024)
Ref_id:b34 Title:  Year: (2023)
Ref_id:b35 Title: Analyzing and improving the training dynamics of diffusion models Year: (2024)
Ref_id:b36 Title: An almost-linear-time algorithm for approximate max flow in undirected graphs, and its multicommodity generalizations Year: (2014)
Ref_id:b37 Title: A method for stochastic optimization Year: (2014)
Ref_id:b38 Title: MM optimization algorithms Year: (2016)
Ref_id:b39 Title: Scalable optimization in the modular norm Year: (2024)
Ref_id:b40 Title: Preconditioned stochastic gradient descent Year: (2017)
Ref_id:b41 Title: Muon is scalable for llm training Year: (2025)
Ref_id:b42 Title: Learning pruning-friendly networks via Frank-Wolfe: One-shot, any-sparsity, and no retraining Year: (2022)
Ref_id:b43 Title: Adaptive bound optimization for online convex optimization Year: (2010)
Ref_id:b44 Title: Spectral normalization for generative adversarial networks Year: (2018)
Ref_id:b45 Title: Stochastic conditional gradient methods: From convex minimization to submodular maximization Year: (2020)
Ref_id:b46 Title: Efficiency of coordinate descent methods on huge-scale optimization problems Year: (2012)
Ref_id:b47 Title: Normalized gradients for all Year: (2023)
Ref_id:b48 Title: νsam: Memory-efficient sharpnessaware minimization via nuclear norm constraints Year: (2025)
Ref_id:b49 Title: Deep neural network training with Frank-Wolfe Year: (2020)
Ref_id:b50 Title: Curvature-informed sgd via general purpose lie-group preconditioners Year: (2024)
Ref_id:b51 Title: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks Year: (2013)
Ref_id:b52 Title: Trust region policy optimization Year: (2015)
Ref_id:b53 Title: Training data-efficient image transformers & distillation through attention Year: (2021)
Ref_id:b54 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b55 Title: Optimal approximation for the submodular welfare problem in the value oracle model Year: (2008)
Ref_id:b56 Title: Numerical optimization Year: (2006)
Ref_id:b57 Title: Implicit bias of AdamW: ℓ ∞ norm constrained optimization Year: (2024)
Ref_id:b58 Title: Tensor programs iv: Feature learning in infinite-width neural networks Year: (2021)
Ref_id:b59 Title: Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer Year: (2022)
Ref_id:b60 Title: A spectral condition for feature learning Year: (2023)
Ref_id:b61 Title: Large batch training of convolutional networks Year: (2017)
Ref_id:b62 Title: Large batch optimization for deep learning Year: (2019)
Ref_id:b63 Title: Block-normalized gradient method: An empirical study for training deep neural network Year: (2017)
Ref_id:b64 Title: Mars: Unleashing the power of variance reduction for training large models Year: (2024)
Ref_id:b65 Title: Exact convergence rate of the last iterate in subgradient methods Year: (2023)
Ref_id:b66 Title: Stochastic normalized gradient descent with momentum for large batch training Year: (2020)
