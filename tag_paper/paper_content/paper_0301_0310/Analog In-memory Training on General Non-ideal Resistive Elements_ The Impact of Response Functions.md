Title: Analog In-memory Training on General Non-ideal Resistive Elements: The Impact of Response Functions
Abstract: As the economic and environmental costs of training and deploying large vision or language models increase dramatically, analog in-memory computing (AIMC) emerges as a promising energy-efficient solution. However, the training perspective, especially its training dynamics, is underexplored. In AIMC hardware, the trainable weights are represented by the conductance of resistive elements and updated using consecutive electrical pulses. While the conductance changes by a constant in response to each pulse, in reality, the change is scaled by asymmetric and non-linear response functions, leading to a non-ideal training dynamics. This paper provides a theoretical foundation for gradient-based training on AIMC hardware with nonideal response functions. We demonstrate that asymmetric response functions negatively impact Analog SGD by imposing an implicit penalty on the objective. To address the issue, we propose residual learning algorithm, which provably converges exactly to a critical point by solving a bilevel optimization problem. We demonstrate that the proposed method can be extended to address other hardware imperfections, such as limited response granularity. As we know, it is the first paper to investigate the impact of a class of generic non-ideal response functions. The conclusion is supported by simulations validating our theoretical insights.

Section: Introduction
The remarkable success of large vision and language models is underpinned by advances in modern hardware accelerators, such as GPU, TPU [1], NPU [2], and NorthPole chip [3]. However, the computational demands of training these models are staggering. For instance, training LLaMA [4] cost $2.4 million, while training GPT-3 [5] required $4.6 million, highlighting the urgent need for more efficient computing hardware. Current mainstream hardware relies on the Von Neumann architecture, in which the physical separation of memory and processing units creates a bottleneck due to frequent, costly data movement between them.
In this context, the industry has turned its attention to analog in-memory computing (AIMC) accelerators based on resistive crossbar arrays [6][7][8][9][10], which excel at accelerating ubiquitous, computationally intensive matrix-vector multiplications (MVMs) operations. In AIMC hardware, the weights (matrices) are represented by the conductance states of the resistive elements in analog crossbar arrays [11,12], while the input and output of MVM are analog signals like voltage and current. Leveraging Kirchhoff's and Ohm's laws, AIMC hardware achieves 10×-10,000× energy efficiency than GPU [13][14][15] in model inference.
Despite its high efficiency, analog training is considerably more challenging than inference since it involves frequent weight updates. Unlike digital hardware, where the weight increment can be applied to the original weight in the memory cell, the weights in AIMC hardware are changed by the so-called pulse update.
Pulse update. When receiving electrical pulses from its peripheral circuits, the resistive elements change their conductance in response to the pulse polarity [16]. Receiving a pulse at each pulse cycle, the conductance is updated by ∆w min • q + (w) or ∆w min • q -(w), depending on the pulse polarity, where ∆w min is response granularity, and q + (w) and q -(w) are response functions. Geometrically, q + (w) and q -(w) are the slopes of response curves; see Figure 1. All ∆w min , q + (w), and q -(w) are element-specific parameters or functions that are set before training and hence remain fixed during training. Typically, ∆w min is known while q + (w) and q -(w) are not.
Gradient-based training implemented by analog update. Supported by pulse update, the gradientbased training algorithms are used to optimize the weights. Consider a standard training problem with objective f ( • ) : R D → R and a model parameterized by W ∈ R D W * := arg min
W ∈R D f (W ) := E ξ [f (W ; ξ)] (1
)
where ξ is a random data sample. Similar to stochastic gradient descent (SGD) in digital training (Digital SGD), the gradient-based training algorithm on AIMC hardware, Analog SGD, updates the weights by stochastic gradients ∇f (W k ; ξ k ). Digital SGD updates the weight by W k+1 = W k -α∇f (W k ; ξ k ) with learning rate α. Given a desired update ∆W = -α∇f (W k ; ξ k ), AIMC hardware implements Analog SGD by sending |[∆W ] d |/∆w min pulses to the d-th element. Ideally, q + (w) = q -(w) = 1 for every conductance states. If so, with each pulse updating [W k ] d by ∆w min , [W k ] d is ultimately updated by about [∆W ] d .
Challenges of analog training. Despite its ultra-efficiency, gradient-based training on AIMC hardware is challenging. First, the generic response functions are asymmetric (i.e. q + (w) ̸ ≡ q -(w)), and non-linear [17][18][19]. Due to the variation of response functions and conductance states, gradients are scaled by different magnitudes across different coordinates, leading to biased gradients. Furthermore, the response granularity ∆w min is a constant. When the gradients or the learning rate decay below ∆w min , pulse update no longer provides sufficient precision to perform gradient descent [20]. Other imperfections include, but are not limited to, noisy input/output (IO) of MVM operations and analog-digital conversion error [18]. This paper aims to investigate the impact of non-ideal response functions and develop a method to mitigate their negative effects. We also discuss extending the proposed method to deal with other hardware imperfections.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b17']

Section: Main results
Complementing existing empirical studies in analog in-memory computing, this paper aims to build a rigorous theoretical foundation of analog training. By introducing bias to the gradient, the asymmetric response function plays a central role in differentiating digital and analog training. In contrast, the other non-idealities hinder the training process by causing precision-related issues. Therefore, we approach the problem progressively, beginning with a simplified case that involves only the asymmetric response functions, and extending the proposed methods to more general scenarios.
As a warm-up, building upon the pulse update mechanism, we propose the following discrete-time mathematical model to characterize the trajectory of Analog SGD
Analog SGD W k+1 = W k -α∇f (W k ; ξ k ) ⊙ F (W k ) -α|∇f (W k ; ξ k )| ⊙ G(W k ) (2
)
where α > 0 is the learning rate and ξ k is the data sample of iteration k; | • | and ⊙ represent the element-wise absolute value and multiplication, respectively; and F ( • ) and G( • ) are hardwarespecific matrix which are defined by q + ( • ) and q -( • ). In Section 2, we will explain the underlying rationale of (2). Compared with the standard Digital SGD, the gradients in (2) are scaled by F ( • ) and an extra bias term is introduced. Typically, hardware imperfections lead to non-ideal response functions, i.e., F ( • ) ̸ ≡ 1 and G( • ) ̸ ≡ 0. Thus, we ask a natural question that Q1) What is the impact of non-ideal response functions and how to alleviate it?
Recently, [21] partially answers the question by showing that Analog SGD suffers from a convergence issue due to the asymmetric update, and a heuristic algorithm, Tiki-Taka [22][23][24], converges exactly by reducing the weight drift. However, their work is limited to a special case of linear response functions, which are in the form of q + (w) = 1 -w/τ, q -(w) = 1 + w/τ with hardware-specific parameter τ > 0. Given more general q + (w) and q -(w), the convergence of Tiki-Taka does not trivially hold, even though the response functions are still linear.
this section cite: ['b20', 'b21', 'b22', 'b23']

Section: Gap between theory for special linear and generic response functions.
Consider a more generic linear response q + (w) = (1 + c Lin )(1w/τ ), q -(w) = (1 -c Lin )(1 + w/τ ) with a parameter c Lin , which reduces to the setting in [21] when c Lin = 0. Figure 2 shows the damage from a non-zero c Lin to Tiki-Taka. Consistent with the conclusion in [21], Tiki-Taka significantly outperforms Analog SGD when c Lin = 0. However, when c Lin is perturbed from 0.1 to 0.3, Tiki-Taka degrades dramatically and even becomes worse than Analog SGD does. The modification is slight, but the convergence guarantee in [21] fails, and the convergence of Tiki-Taka is harmed significantly. This counter-example indicates a gap between the theory for special linear and generic response functions, and necessitates the study of the analog training with generic response functions and the exploration of exact convergence conditions.
Ignoring other imperfections temporarily, this paper first analyzes the impact of response functions. We show that Analog SGD suffers from asymptotic error due to the mismatch between the algorithmic stationary point and physical symmetric point. Inspired by that, we propose a novel algorithm framework that aligns two points, overcoming the asymmetric issues. Building on that, we endeavor to extend the proposed algorithm to more practical scenarios that involve other imperfections like limited granularity and noisy readings, prompting a second critical question:
Q2) How to extend the framework to address the limited response granularity and noisy IO issues?
To answer this question, we propose two mechanisms to further overcome these two issues.
Our contributions. This paper makes the following contributions:
C1) Building on the pulse update equation, we propose an approximate discrete-time dynamics for analog update. Enabled by this, we study the impact of response functions directly, without being limited to specific element candidates. C2) Based on that, we show that instead of optimizing f ( • ), Analog SGD optimizes another penalized objective implicitly. An implicit penalty is introduced by the asymmetric response functions, which attract the weights towards symmetric points. Consequently, Analog SGD can only converge to the optimal point inexactly.
this section cite: ['b20', 'b20', 'b20']

Section: C3)
We propose a novel Residual Learning theoretical framework to alleviate the asymmetric update and implicit penalty issues. Residual Learning explicitly introduces another residual array, which has a stationary point 0. This framework leads to Tiki-Taka heuristically proposed in [22] while it offers an understanding of how Tiki-Taka deals with the challenge from generic response functions. By properly zero-shifting so that the stationary and symmetric points overlap, Residual Learning provably converges to a critical point. C4) Building on C3), we propose a variant, Residual Learning v2, tailored for more practical training scenarios. We propose introducing a digital buffer to filter out reading errors caused by IO noise. Furthermore, we propose a threshold-based transfer rule to alleviate instability caused by limited granularity.
this section cite: ['b21']

Section: Prior art AIMC training.
Analog training has shown promising early successes with tremendous energy advantage [25,26]. Among them, on-chip training, which performs forward, backward, and update directly on analog chips [22-24, 27, 28] is considered to be the most efficient paradigm, but it is more sensitive to hardware imperfections. Sacrificing energy efficiency for robustness, hybrid digital-analog off-chip training is proposed [29][30][31][32], which offloads some computation burden to digital components. This paper focuses on the more challenging on-chip training setting.
Energy-based model and equilibrium propagation. AIMC training leverages back-propagation to compute the gradient signals. Recently, a class of energy-based models has been studied, which performs equilibrium propagation to compute gradient signals [33][34][35][36][37]. Focusing on the training dynamics instead of concrete gradient computing, our work is orthogonal to them and is expected to provide insight for algorithm designs of energy-based model training.
this section cite: ['b24', 'b25', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36']

Section: Analog Training with Generic Response Functions
This section examines the discrete-time dynamics of analog training and introduces the challenges posed by generic response functions. After that, we introduce a family of response functions that reflect crucial physical properties that interest us.
Compact formulations of analog update. We first investigate the dynamics of one element w in W ∈ R D . This paper adopts w to represent the element of the weight W k without specifying its index. As we discuss in Section 1, the response granularity ∆w min is scaled by the response functions q + (w) or q -(w). Since a desired update ∆w requires a series of pulses with each scaled by approximately q + (w) or q -(w), it is sensible that the ∆w is approximately scaled by q + (w) or q -(w) as well. Accordingly, we propose that an approximate dynamics of analog update is given by w ′ ≈ U q (w, ∆w), where U q (w, ∆w) is defined by
U q (w, ∆w) := w + ∆w • q + (w), ∆w ≥ 0, w + ∆w • q -(w), ∆w < 0.(3)
The update (3) holds at each resistive element. At the k-th iteration, We stack all the weights w k and expected increment ∆w k together into vectors W k , ∆W k ∈ R D . Similarly, the response functions q + ( • ) and q -( • ) are stacked into Q + ( • ) and Q -( • ), respectively. Let the notation U Q (W k , ∆W ) on matrices W k and ∆W denote the element-wise operation on W k and ∆W , i.e.
[U Q (W k , ∆W )] d := U [Q] d ([W k ] d , [∆w] d ), ∀d ∈ [D] with
which leads to a compact form of the Analog Update
Analog Update W k+1 = W k + ∆W k ⊙ F (W k ) -|∆W k | ⊙ G(W k ).(5)
Gradient-based training algorithms on AIMC hardware. In (5), the desired update ∆W k varies based on different algorithms. Replacing ∆W k with the stochastic gradient ∇f (W k ; ξ k ), we obtain
this section cite: []

Section: Response function
Figure 3: Examples of response functions from Definition 1; w ⋄ is the symmetric point.
the dynamics of Analog SGD shown in (2). This update is reduced to the mathematical form for linear response functions in [21]  plays a critical role in the convergence behaviors.
In addition to the asymmetric update, the function class should possess other properties. First, the conductance increases upon receipt of a positive pulse, and vice versa, resulting in positive response functions. In addition, we assume that the response functions are differentiable (and hence continuous) for mathematical tractability. Taking all factors into account, we define the following class of response functions. Definition 1 covers a wide range of response functions, including but not limited to PCM, ReRAM, ECRAM, and others mentioned in Section A. Figure 3 showcases three examples from the response functions class, including linear, non-linear but monotonic, and even non-monotonic functions.
this section cite: ['b20']

Section: Implicit Penalty and Inexact Convergence of Analog SGD
This section introduces a critical impact of the response functions, implicit penalized objective.
Affected by this, Analog SGD can only converge inexactly with a non-diminishing asymptotic error.
this section cite: []

Section: Implicit penalty
We first give an intuition through a situation where W k is already a critical point, i.e.,
E ξ [∇f (W k ; ξ)] = 0. Recall that stochastic gradient descent on digital hardware (Digital SGD) is stable in expectation, i.e. E ξ k [W k+1 ] = W k -E ξ k [α∇f (W k ; ξ k )] = W k . However, this does not work for Analog SGD E ξ k [W k+1 ] = W k -E ξ k [α∇f (W k ; ξ k ) ⊙ F (W k ) -α|∇f (W k ; ξ k )| ⊙ G(W k )] (6
) = W k -αE ξ k [|∇f (W k ; ξ k )| ⊙ G(W k )] ̸ = W k .
Consider a simplified version that the weight is a scalar (D = 1) and the function G(W ) is strictly monotonically decreasingfoot_0 to help us gain intuition on the impact of the drift in (6).
Recall G(W ⋄ ) = 0 at the symmetric point W ⋄ . G(W ) > 0 when W > W ⋄ and G(W ) < 0 otherwise. Consequently, (6) indicates that E ξ k [W k+1 ] < W k when W k > W ⋄ and E ξ k [W k+1 ] > W k otherwise.
It implies that W k suffers from a drift tendency towards W ⋄ . In addition, the penalty coefficient proportional to the noise level since the drift is proportional to
E ξ k [|∇f (W k ; ξ k )|], which is the first moment of noise E ξ k [|∇f (W k ; ξ k ) -E ξ [∇f (W k ; ξ)]|] in essence.
The following theorem formalizes the implicit penalty effect. Before that, we define an accumulated asymmetric function R c ( •
) : R D → R D , whose derivative is R(W ) := G(W ) F (W ) , i.e. d[Rc(W )] d d[W ] d = [R(W )] d = [G(W )] d [F (W )] d . If R(W ) is strictly monotonic, R c (W )
reaches its minimum at the symmetric point W ⋄ where R(W ⋄ ) = 0, so that it penalizes the weight away from the symmetric point.
this section cite: ['b5']

Section: Theorem 1 (Implicit penalty, short version).
Suppose W * is the unique minimizer of problem (1).
Let Σ := E ξ [|∇f (W * ; ξ)|] ∈ R D . Analog SGD implicitly optimizes the following penalized objective min W f Σ (W ) := f (W ) + ⟨Σ, R c (W )⟩ .(7)
The full version of Theorem 1 and its proof are deferred to Appendix G. In Theorem 1, R c (W ) plays the role of a penalty to force the weight towards a symmetric point. As shown in Appendix G, R c (W ) has a simple expression on linear response functions when c Lin = 0, leading (7) to min W f Σ (W ) := f (W ) + Σ 2τ ∥W ∥ 2 which is an ℓ 2 regularized objective. In addition, the implicit penalty has a coefficient proportional to the noise level Σ and inversely proportional to the dynamic range τ . It implies that the implicit penalty becomes active only when gradients are noisy, and the noise amplifies the effect.
With noisy gradients, an implicit penalty attracts Analog SGD towards symmetric points.
this section cite: ['b0']

Section: Inexact Convergence of Analog SGD under generic devices
Due to the implicit penalty, Analog SGD only converges to a critical point inexactly. Before showing that, We introduce a series of assumptions on the objective, as well as noise. Assumption 1 (Objective). The objective f (W ) is L-smooth and is lower bounded by f * . Assumption 2 (Unbiasness and bounded variance). The stochastic gradient is unbiased and has bounded variance σ 2 . i.e., E ξ [∇f (W ; ξ)] = ∇f (W ) and E ξ [∥∇f (W ; ξ) -∇f (W )∥ 2 ] ≤ σ 2 . Assumption 1-2 are standard in non-convex optimization [38]. This paper considers the average squared norm of the gradient as the convergence metric, given by E ASGD
K := 1 K K-1 k=0 ∥∇f (W k )∥
2 . Now, we establish the convergence of Analog SGD. Theorem 2 (Inexact convergence of Analog SGD). Under Assumption 1-2, if the learning rate is set as α = O(1/ √ K), it holds that
E ASGD K ≤ O σ 2 /K + σ 2 S ASGD K(8)
where S ASGD K denotes the amplification factor given by S ASGD
K := 1 K K-1 k=0 G(W k ) √ F (W k ) 2 ∞ .
The proof of Theorem 2 is deferred to Appendix H. Theorem 2 suggests that the convergence metric E ASGD K is upper bounded by two terms: the first term vanishes at a rate of O( σ 2 /K), which matches the Digital SGD's convergence rate [38] up to a constant; the second term contributes to the asymptotic error of Analog SGD, which does not vanish with the number of iterations K.
this section cite: ['b37', 'b37']

Section: Impact of saturation/asymmetric update.
The exact expression of S ASGD K depends on the specific noise distribution and thus is difficult to reach. However, S ASGD K reflects the saturation degree near the critical point W * when W k converges to a neighborhood of W * . If W * is far from the symmetric point W ⋄ , S ASGD K becomes large, leading to a large E ASGD K and a large asymptotic error. In contrast, if W * remains close to the symmetric point W ⋄ , the asymptotic error is small.
this section cite: []

Section: Mitigating Implicit Penalty by Residual Learning
The asymptotic error in Analog SGD is a fundamental issue that arises from the mismatch between the symmetric point and the critical point. An idealistic remedy for the inexact convergence is carefully shifting the weights to ensure the stationary point is close to a symmetric point. However, determining the appropriate shifting is challenging, as the critical point is unknown before training. Therefore, an ideal solution to address this issue is to jointly construct a sequence with a proper stationary point and a proper shift of the symmetric point.
Residual learning. Our solution overlaps the algorithmic stationary point and physical symmetric point on the special point 0. Besides the main analog array, W k , we maintain another array, P k , whose stationary point should be 0. A natural choice is the residual of the weight, P * (W ), defined by the P that minimizes the objective f (W + γP ) with a non-zero γ. Notice that P * (W k ) → 0 as W k → W * . Additionally, the goal of the main array is to minimize the residual so that the model W k approaches optimality. This process can be formulated as the following bilevel problem, whose optimal points can be proved to be those of f (W )
Residual Learning min W ∈R D ∥P * (W )∥ 2 , s.t. P * (W ) ∈ arg min P ∈R D f (W + γP ). (9
)
Now we propose a gradient-based method to solve (9). The stochastic gradient of f (W + γP ) with respect to P , given by ∇ P f (W + γP ; ξ) = γ∇f (W + γP ; ξ), is accessible with fair expense, enabling us to introduce a sequence P k to track the residual of W k by optimizing f (W k + γP )
P k+1 = P k -α∇f ( Wk ; ξ k ) ⊙ F (P k ) -α|∇f ( Wk ; ξ k )| ⊙ G(P k ).(10)
where Wk := W k + γP k is the mixed weight. We then derive the hyper-gradient of the upper-level objective. Notice ∇∥P * (W )∥ 2 = 2∇P * (W )P * (W ). Assuming W * is the unique minimum of f ( • ), we know P * (W ) satisfies γP * (W ) + W = W * . Taking gradient with respective to W on both sides, we have ∇P * (W ) = -1 γ I and hence ∇∥P * (W )∥ 2 = -2 γ P * (W ). Approximating P * (W ) by P k and absorbing 2  γ into the learning rate β, we reach the update of the main array
W k+1 = W k + βP k+1 ⊙ F (W k ) -β|P k+1 | ⊙ G(W k ).(11)
Featuring moving the residual P k to W k , (11) is referred to as transfer process. The updates (10) and (11) are performed alternatively until convergence. Tiki-Taka mentioned in [21] is the special case with linear response functions and γ = 0.
On the response functions side, it is naturally required to let zero be a symmetric point, i.e., G(0) = 0, which can be implemented by the zero-shifting technique [39] by subtracting a reference array.
Convergence properties of Residual Learning. We begin by analyzing the convergence of Residual Learningwithout considering the zero-shift first, which enables us to understand how zero-shifted response functions affect convergence.
If the optimal point W * exists and is unique, the solution of the lower-level objective has a closed form P * (W ) := W * -W γ . At that time, the upper-level objective equals ∥W * -W ∥ 2 . However, the solutions of f ( • ) are generally non-unique, especially for non-convex objectives with multiple local minima. To ensure the existence and uniqueness of W * , we assume the objective is strongly convex. Assumption 3 (µ-strong convexity). The objective f (W ) is µ-strongly convex.
Under the strongly convex assumption, the optimal point W * is unique and hence the optimal solution of the lower-level problem in (9) is unique. Since the requirement of strong convexity is non-essential in the development of bilevel optimization [40][41][42][43], we believe the proof can be extended to more general cases and will extend it for future work.
E RL K := 1 K K-1 k=0 E ∥∇f ( Wk )∥ 2 + O(∥P k -P * (W k )∥ 2 ) + O(∥W k -W * ∥ 2 ) .(12)
For simplicity, the constants in front of some terms in E RL K are hidden. Now, we provide the convergence of Residual Learning with generic responses. Theorem 3 (Convergence of Residual Learning). Under Assumptions 1-3, with the learning rate α = O 1/σ 2 K , β = O(αγ 3/2 ), it holds for Residual Learning that
E RL K ≤ O σ 2 /K + σ 2 S RL K (13
)
where S RL K denotes the amplification factor of P k given by
S RL K := 1 K K k=0 G(P k ) √ F (P k ) 2 ∞ .
The proof of Theorem 3 is deferred to Appendix I. Theorem 3 claims that Residual Learning converges at the rate O σ 2 /K to a neighbor of critical point with radius O(σ 2 S RL K ), which share almost the same expression with the convergence of Analog SGD. The difference lies in the amplification factor S RL K and S ASGD K , where the former depends on P k while the latter depends on W k .
this section cite: ['b8', 'b9', 'b10', 'b20', 'b38', 'b39', 'b40', 'b41', 'b42']

Section: Impact of response functions.
Response function affects the Analog SGD and Residual Learning similarly. However, attributed to the residual array, constructing response functions to enable exact convergence of Residual Learning is viable.
As we have discussed, P k tends to P * (W k ) which tends to 0 given W k tends to W * . Therefore, response functions with G(P ) = 0 when P = 0 are required for the exact convergence.
this section cite: []

Section: Extension of Residual Learning: limited granularity and noisy IO
This section extends Residual Learning to practical scenarios with additional hardware imperfections. To be specific, we consider the noisy IO and limited granularity as examples. We highlight that we are not trying to diminish the importance of imperfection, but rather focus on two of the primary ones known to be crucial.
IO of resistive crossbar arrays introduces noise during the reading of P k+1 in the transfer process (11),
given by W k+1 = W k +β(P k+1 +ε k )⊙F (W k )-β|P k+1 +ε k |⊙G(W k )
with a noise ε k . It incurs the implicit penalty issues again, leading to a penalized upper-level objective ∥P * (W )∥ 2 + ⟨Σ ε , R c (W )⟩, as claimed by Theorem 1, where
Σ ε = E[|ε k |]
is assumed to be a constant. To filter out the noise, we propose to use a digital buffer H k to take a moving average of noisy P k+1 signals by Intuitively, with a fixed P k+1 , H k will converge to a neighborhood of P k+1 with radius O(β). Therefore, a sufficiently small β renders H k a fair approximation of noiseless P k , enabling optimizing the upper-level objective with clearer signals. After that, H k+1 is transferred to W k as follows
H k+1 = (1 -β)H k + β(P k+1 + ε k+1 ).(14)
W k+1 = W k + βH k+1 ⊙ F (W k ) -β|H k+1 | ⊙ G(W k ).(15)
Furthermore, the transfer process suffers from a constant error of O(∆w min ) due to the discrete pulse firing, each of which changes the weight by O(∆w min ). To overcome these issues, we propose introducing a threshold mechanism that does not transfer the entire H k+1 to W k at each iteration, as in (15). Instead, we compute an intermediate value by
H k+ 1 2 = (1 -β)H k + β(P k+1 + ε k+1 ) first. At each coordinate d, if the value |[H k+ 1 2 ] d | ≥ ∆w min ,
one pulse will be fired to [W k ] d and update the digital buffer by [H k+1 ] d = [H k+ 1 2 ] d -∆w min or [H k+1 ] d = [H k+ 1 2 ] d + ∆w min , where the sign of increment is determined by the sign of [H k+ 1 2 ] d . Otherwise, no transfer is triggered if the intermediate value falls below the threshold, i.e., [H k+1 ] d = [H k+ 1 2 ] d . The proposed algorithms are referred to as Residual Learning v2.
this section cite: ['b10', 'b14']

Section: Numerical Simulations
In this section, we verify the main theoretical results by simulations on both synthetic datasets and real datasets. We use the open source toolkit IBM Analog Hardware Acceleration Kit (AIHWKIT) [44] to simulate the behaviors of Analog SGD, Residual Learning (which reduces to Tiki-Taka). Each simulation is repeated three times, and the mean and standard deviation are reported. We consider two types of response functions in our simulations: power and exponential response functions with dynamic ranges [-τ, τ ] and the symmetric point being 0, as required by Corollary 1. More details, simulations, and ablation studies can be found in Appendix K. The code of our simulations is available at github.com/Zhaoxian-Wu/analog-training.
this section cite: ['b43']

Section: FCN/CNN @ MNIST.
We train a fully-connected network (FCN) and a convolutional neural network (CNN) on the MNIST dataset and compare the performance of Analog SGD and Tiki-Taka under various dynamic range τ on power responses; see the results in Figure 4. By tracking residual, Residual Learning outperforms Analog SGD and reaches comparable accuracy with Digital SGD. For both architectures, the accuracy of Residual Learning drops by < 1%. In contrast, Analog SGD takes a few epochs to achieve a noticeable increase in accuracy in FCN training, rendering a slower convergence rate than Residual Learning. In CNN training, Analog SGD's accuracy increases more slowly than Residual Learning, eventually settling at about 80%. It is consistent with the theoretical claims.
ResNet @ CIFAR10/CIFAR100. We fine-tune three ResNet models with different scales on CIFAR10/CIFAR100 datasets. The power response functions are used, whose results are shown in Table 1. The results show that the Tiki-Taka outperforms Analog SGD by about 1.0% in most of the cases in ResNet34/50, and the gap even reaches about 7.0% for ResNet18 training on the CIFAR100 dataset. On top of that, we also compare the proposed Residual Learning v2 and Tiki-Taka v2. Both of them outperform Residual Learning since they introduce a digital buffer to filter out the reading noise. However, Residual Learning v2 outperforms Tiki-Taka v2 on the CIFAR100 dataset, demonstrating the benefit from the bilevel formulation.
Ablation study on γ. We conduct simulations to study the impact of mixing coefficient γ in (10) on the CIFAR10 or CIFAR100 dataset in the ResNet training tasks. The results are presented in Figure 5, which shows that Residual Learning achieves a great accuracy gain from increasing γ from 0 to 0.1, while the gain saturates from 0.1 to 0.4. Therefore, we conclude that Residual Learning benefits from a non-zero γ, and the performance is robust to the γ selection.
this section cite: []

Section: Conclusions and Limitations
This paper studies the impact of a generic class of asymmetric and non-linear response functions on gradient-based training in analog in-memory computing hardware. We first formulate the dynamics of Analog Update based on the pulse update rule. Based on it, we show that Analog SGD implicitly optimizes a penalized objective and hence can only converge inexactly. To overcome this issue, we propose a Residual Learning framework which solves a bilevel optimization problem. Explicitly aligning the algorithmic stationary point and physical symmetric point, Residual Learning provably converges to the optimal point exactly. Furthermore, we demonstrate how to extend Residual Learning to overcome the noisy reading and limited update granularity issues. The efficiency of the proposed method is verified through simulations. One limitation of this work is that the current analysis considers only the three hardware imperfections. While they are known to be crucial for analog training, it is also important to extend our convergence analysis and methods to more practical scenarios involving more imperfections in future work.
this section cite: []

Section: References
Ref_id:b0 Title: TPU v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings Year: (2023)
Ref_id:b1 Title: Neural acceleration for general-purpose approximate programs Year: (2012)
Ref_id:b2 Title: Neural inference at the frontier of energy, space, and time Year: (2023)
Ref_id:b3 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b4 Title: Language models are few-shot learners Year: (2020)
Ref_id:b5 Title: A comprehensive crossbar array model with solutions for line resistance and nonlinear device characteristics Year: (2013)
Ref_id:b6 Title: The next generation of deep learning hardware: Analog computing Year: (2019)
Ref_id:b7 Title: Efficient processing of deep neural networks: A tutorial and survey Year: (2017)
Ref_id:b8 Title: Memory devices and applications for in-memory computing Year: (2020)
Ref_id:b9 Title: A 64-core mixed-signal in-memory compute chip based on phase-change memory for deep neural network inference Year: (2023)
Ref_id:b10 Title: Neuromorphic computing using non-volatile memory Year: (2017)
Ref_id:b11 Title: Memristive devices for computing Year: (2013)
Ref_id:b12 Title: Neural network accelerator design with resistive crossbars: Opportunities and challenges Year: (2019)
Ref_id:b13 Title: Towards 10000TOPS/W DNN inference with analog in-memory computing -a circuit blueprint, device options and requirements Year: (2019)
Ref_id:b14 Title: TOP/s/mm 2 in-memory analog matrix-vector-multiplier for DNN acceleration Year: (1540)
Ref_id:b15 Title: Acceleration of deep neural network training with resistive cross-point devices: Design considerations Year: (2016)
Ref_id:b16 Title: Experimental demonstration and tolerancing of a large-scale neural network (165 000 synapses) using phase-change memory as the synaptic weight element Year: (2015)
Ref_id:b17 Title: Resistive memory device requirements for a neural algorithm accelerator Year: (2016)
Ref_id:b18 Title: Mitigating effects of non-ideal synaptic device characteristics for on-chip learning Year: (2015)
Ref_id:b19 Title: Accurate deep neural network inference using computational phase-change memory Year: (2020)
Ref_id:b20 Title: Towards exact gradient-based training on analog in-memory computing Year: (2024)
Ref_id:b21 Title: Algorithm for training neural networks on resistive device arrays Year: (2020)
Ref_id:b22 Title: Enabling training of neural networks on noisy hardware Year: (2021)
Ref_id:b23 Title: Fast and robust analog in-memory deep neural network training Year: (2024)
Ref_id:b24 Title: Face classification using electronic synapses Year: (2017)
Ref_id:b25 Title: In situ training of feed-forward and recurrent convolutional memristor networks Year: (2019)
Ref_id:b26 Title: SSM: a high-performance scheme for in situ training of imprecise memristor neural networks Year: (2020)
Ref_id:b27 Title: Overcoming challenges for achieving high in-situ training accuracy with emerging memories Year: (2020)
Ref_id:b28 Title: A compute-in-memory chip based on resistive random-access memory Year: (2022)
Ref_id:b29 Title: Fully hardware-implemented memristor convolutional neural network Year: (2020)
Ref_id:b30 Title: Mixed-precision architecture based on computational memory for training deep neural networks Year: (2018)
Ref_id:b31 Title: Mixedprecision deep learning based on computational memory Year: (2020)
Ref_id:b32 Title: Equilibrium propagation: Bridging the gap between energy-based models and backpropagation Year: (2017)
Ref_id:b33 Title: Energy-based analog neural network framework Year: (2023)
Ref_id:b34 Title: Energy-based learning algorithms for analog computing: a comparative study Year: (2024)
Ref_id:b35 Title: Training end-to-end analog neural networks with equilibrium propagation Year: (2020)
Ref_id:b36 Title: Equilibrium propagation with continual weight updates Year: (2020)
Ref_id:b37 Title: Optimization methods for large-scale machine learning Year: (2018)
Ref_id:b38 Title: Zero-shifting technique for deep neural network training on resistive cross-point arrays Year: (2019)
Ref_id:b39 Title: A generalized alternating method for bilevel learning under the polyak-łojasiewicz condition Year: (2023)
Ref_id:b40 Title: Non-convex bilevel games with critical point selection maps Year: (2022)
Ref_id:b41 Title: On penalty-based bilevel gradient descent method Year: (2023)
Ref_id:b42 Title: On penalty methods for nonconvex bilevel optimization and first-order stochastic approximation Year: (2024)
Ref_id:b43 Title: A flexible and fast PyTorch toolkit for simulating training and inference on analog crossbar arrays Year: (2021)
Ref_id:b44 Title: Recent Progress in Phase-Change Memory Technology Year: (2016)
Ref_id:b45 Title: An overview of phase-change memory device physics Year: (2020)
Ref_id:b46 Title: ReRAM-based synaptic device for neuromorphic computing Year: (2014)
Ref_id:b47 Title: Optimization of conductance change in Pr 1-x Ca x MnO 3 -based synaptic devices for neuromorphic systems Year: (2015)
Ref_id:b48 Title: Analog resistive switching devices for training deep neural networks with the novel Tiki-Taka algorithm Year: (2024)
Ref_id:b49 Title: Improved synaptic behavior of CBRAM using internal voltage divider for neuromorphic systems Year: (2018)
Ref_id:b50 Title: Parallel programming of an ionic floating-gate memory array for scalable neuromorphic computing Year: (2019)
Ref_id:b51 Title: ECRAM as scalable synaptic cell for high-speed, low-power neuromorphic computing Year: (2018)
Ref_id:b52 Title: Nanosecond protonic programmable resistors for analog deep learning Year: (2022)
Ref_id:b53 Title: A crossbar array of magnetoresistive memory devices for in-memory computing Year: (2022)
Ref_id:b54 Title: Adapting magnetoresistive memory devices for accurate and on-chip-training-free in-memory computing Year: (2024)
Ref_id:b55 Title: Ferroic tunnel junctions and their application in neuromorphic networks Year: (2020)
Ref_id:b56 Title: Threedimensional NAND flash for vector-matrix multiplication Year: (2018)
Ref_id:b57 Title: Efficient and robust spike-driven deep convolutional neural networks based on NOR flash computing array Year: (2020)
Ref_id:b58 Title: High-performance mixed-signal neurocomputing with nanoscale floatinggate memory cell arrays Year: (2017)
Ref_id:b59 Title: Statistical computing framework and demonstration for in-memory computing systems Year: (2022)
Ref_id:b60 Title: A maximally row-parallel MRAM in-memory-computing macro addressing readout circuit sensitivity and area Year: (2021)
Ref_id:b61 Title: Exploring cycle-to-cycle and device-to-device variation tolerance in mlc storage-based neural network training Year: (2019)
Ref_id:b62 Title: In-memory computation of a machine-learning classifier in a standard 6t SRAM array Year: (2017)
Ref_id:b63 Title: The marriage of training and inference for scaled deep learning analog hardware Year: (2019)
Ref_id:b64 Title: Improving the accuracy of analog-based in-memory computing accelerators post-training Year: (2024)
Ref_id:b65 Title: PIM-QAT: Neural network quantization for processing-in-memory Year: (2022)
Ref_id:b66 Title: Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators Year: (2023)
Ref_id:b67 Title: Reshape and adapt for output quantization (RAOQ): Quantization-aware training for in-memory computing systems Year: (2024)
Ref_id:b68 Title: Vortex: Variation-aware training for memristor x-bar Year: (2015)
Ref_id:b69 Title: NEAT: Non-linearity aware training for accurate and energy-efficient implementation of neural networks on 1t-1r memristive crossbars Year: (2020)
Ref_id:b70 Title: Training deep convolutional neural networks with resistive cross-point devices Year: (2017)
Ref_id:b71 Title: Fully memristive neural networks for pattern classification with unsupervised learning Year: (2018)
Ref_id:b72 Title: Deep learning acceleration in 14nm CMOS compatible ReRAM array: device, material and algorithm co-optimization Year: (2022)
Ref_id:b73 Title: Pipeline gradient-based model training on analog in-memory accelerators Year: (2024)
Ref_id:b74 Title: Deep physical neural networks trained with backpropagation Year: (2022)
Ref_id:b75 Title: Training of physical neural networks Year: (2024)
Ref_id:b76 Title: Holography in artificial neural networks Year: (1990)
Ref_id:b77 Title: Wave physics as an analog recurrent neural network Year: (2019)
Ref_id:b78 Title: Neuromorphic photonic networks using silicon photonic weight banks Year: (2017)
Ref_id:b79 Title: Signal and noise extraction from analog memory elements for neuromorphic computing Year: (2018)
Ref_id:b80 Title: Thousands of conductance levels in memristors integrated on CMOS Year: (2023)
Ref_id:b81 Title: Linear symmetric self-selecting 14-bit kinetic molecular memristors Year: (2024)
Ref_id:b82 Title: Programming memristor arrays with arbitrarily high precision for analog computing Year: (2024)
Ref_id:b83 Title: A heterogeneous and programmable compute-in-memory accelerator architecture for analog-ai using dense 2-d mesh Year: (2022)
Ref_id:b84 Title: Introductory Lectures on Convex Optimization: A Basic Course Year: (2013)
