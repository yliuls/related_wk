Title: Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules
Abstract: Scaling laws have emerged as a unifying lens for understanding and guiding the training of large language models (LLMs). However, existing studies predominantly focus on the final-step loss, leaving open whether the entire loss dynamics obey similar laws and, crucially, how the learning rate schedule (LRS) shapes them. We address these gaps in a controlled theoretical setting by analyzing stochastic gradient descent (SGD) on a power-law kernel regression model. The key insight is a novel intrinsic-time viewpoint, which captures the training progress more faithfully than iteration count. We then establish a Functional Scaling Law (FSL) that captures the full loss trajectory under arbitrary LRSs, with the schedule's influence entering through a simple convolutional functional. We further instantiate the theory for three representative LRSs-constant, exponential decay, and warmup-stable-decay (WSD)-and derive explicit scaling relations in both data-and compute-limited regimes. These comparisons explain key empirical phenomena: (i) higher-capacity models are more data-and compute-efficient; (ii) learning-rate decay improves training efficiency; and (iii) WSD-type schedules outperform pure decay. Finally, experiments on LLMs ranging from 0.1B to 1B parameters demonstrate the practical relevance of FSL as a surrogate model for fitting and predicting loss trajectories in large-scale pre-training.

Section: Introduction
It is well established that the training of large-scale deep learning models mysteriously follows scaling laws, which describe how model performance scales predictably with available resources such as compute or data [19]. In particular, the landmark study by Kaplan et al. [25] demonstrated that, in LLM pre-training, the loss L decreases with model size M and dataset size D according to a power-law relation:
L(M, D) = L 0 + C M M -α M + C D D -α D ,(1)
where α M and α D are the scaling exponents, L 0 denotes the irreducible loss, and C M , C D are some constants. Such empirical relations have proven remarkably robust across scales, architec- In both subplots, solid lines denote the results of SGD, while dashed lines represent the corresponding FSL predictions. (a) FSL accurately tracks the loss dynamics of SGD, averaged over 1000 runs, for three learning rate schedules: cosine, WSD-like, and a non-standard cyclic schedule. (b) FSL predictions (dashed) are computed using the analytical forms from Section 5, and compared with the mean of 200 SGD runs (solid).
tures, and training setups [20,60,36] and have become foundational principles for guiding LLM development [18,24,1,5,55,27]. In practice, they are now routinely used to design optimal resource-allocation strategies [20] and to tune key hyperparameters such as learning rates and batch sizes [36,29].
Despite their empirical success, the theoretical understanding of scaling laws remains limited. Recent studies have begun to illuminate the underlying mechanisms [54,22,39,63,23,42,43,2,14,3,7,35,50,8,70], yet two important gaps persist:
• Determinants of scaling efficiency. Existing studies lack a systematic characterization of how key factors-such as model capacity, task difficulty, and hyperparameter choices-govern scaling efficiency, as reflected by the exponents α M and α D . In particular, learning rate schedules (LRSs) are known to be critical in practice [40,4,17], but their precise role in shaping scaling behavior remains unclear.
• Beyond the final-step loss. The scaling law (1) focuses only on the end-of-training loss [25,20], thus leaving open whether the full trajectory follows similar laws. Empirical studies [59,38] suggest this possibility, but the fits there are still crude and lack theoretical grounding.
this section cite: ['b18', 'b24', 'b19', 'b59', 'b35', 'b17', 'b23', 'b0', 'b4', 'b54', 'b26', 'b19', 'b35', 'b28', 'b53', 'b21', 'b38', 'b62', 'b22', 'b41', 'b42', 'b1', 'b13', 'b2', 'b6', 'b34', 'b49', 'b7', 'b69', 'b39', 'b3', 'b16', 'b24', 'b19', 'b58', 'b37']

Section: Our Contribution
In this paper, we take a step toward addressing these gaps in a controlled yet representative theoretical setting. We study stochastic gradient descent (SGD) training of the power-law kernel (PLK) regression-a widely adopted surrogate for scaling-law analysis [7,3,50,35,8]. The PLK regression is characterized by four parameters: the task difficulty s, the capacity exponent β, the model size M , and the label-noise level σ. To capture the influence of learning-rate schedules (LRSs), we model SGD via an intrinsic-time SDE, in which the concept of intrinsic time emerges as a key quantity enabling a unified characterization of how different LRSs shape the loss dynamics. Building on this formulation, we establish the Functional Scaling Law (FSL), which provides a unified description of the entire loss dynamics-beyond the traditional final-loss prediction. Concretely, for a general intrinsic-time LRS γ : [0, ∞) → [0, ∞), and under some conditions, the dynamics of the expected loss E[R(ν t )] (where t denotes the intrinsic time) satisfies:
E[R(ν t )] - σ 2 2 irreducible error ≂ 1 M sβ approx. error + e(t)
signal learning
+ t 0 K(t -z) [e(z) + σ 2 ] γ(z) dz noise accumulation ,(2)
where e(t) = (1 + t) -s and K(t) = (1 + t) -(2-1/β) . Each term in FSL admits clear interpretation:
σ 2
2 denotes the irreducible error caused by label noise, M -sβ represents the approximation error, e(t) characterizes the signal-learning dynamics under noiseless (full-batch) gradient descent, and the final term captures the injection and dissipation of gradient noise, with the LRS γ entering through Table 1: Learning-rate schedule (LRS) strongly influences scaling efficiency in power-law kernel regression.
Efficiency is determined by two key factors: relative task difficulty s ∈ (0, ∞) and model capacity β > 1. We distinguish between an easy-learning regime (s ⩾ 1 -1/β) and a hard-learning regime (s < 1 -1/β).
this section cite: ['b6', 'b2', 'b49', 'b34', 'b7']

Section: Learning
Rate Schedule (LRS) Data-Optimal Scaling Laws Compute-Optimal Scaling Laws Easy Hard Easy Hard Constant D -s s+1 C -sβ 1+sβ+β Exponential-decay D -sβ 1+sβ (log D) sβ 1+sβ D -s (log D) s C -sβ 2+sβ (log C) sβ 2+sβ C -sβ 1+β (log C) sβ 1+β Warmup-stable-decay (WSD) D -sβ 1+sβ (log D) sβ-s 1+sβ D -s C -sβ 2+sβ (log C) sβ-s 2+sβ C -sβ 1+β
a tractable convolutional functional. The function K, a memory kernel, which we also refer to as the forgetting kernel, quantifies how fast the injected noise dissipates during training.
Building on FSL, we derive concrete scaling laws for the final-step loss under three representative LRSs-constant, exponential decay [15], and warmup-stable-decay (WSD) [69,21]-in both datalimited and compute-limited regimes. The results, summarized in Table 1, recover and extend prior analyses [7,8,50,35], and reveal several unifying insights.
• Scaling efficiency of different schedules. WSD achieves the best scaling efficiency, followed by exponential decay and then constant schedules. This efficiency hierarchy provides theoretical justification for learning-rate decay and explains empirical success of WSD [69,21,58,36].
• Role of model capacity. Higher-capacity models are consistently more efficient in both compute and data, highlighting the necessity of scaling model capacity [25].
• Data-model trade-off. Compute-optimal training requires scaling data more than model size, consistent with established heuristics in LLM pre-training [20].
• Scaling law for peak learning rate. Optimal scaling requires the peak learning rate (LR) to scale appropriately with the training budget (data or compute), revealing the importance of careful peak LR tuning [6,29].
Beyond PLK regression, we further apply the FSL ansatz to fit and predict loss trajectories from LLM pre-training experiments with model sizes ranging from 0.1B to 1B parameters, covering both dense and MoE architectures. These results highlight the potential of FSL as a practical surrogate for understanding and guiding LLM pre-training. To better situate our contribution, we provide a detailed comparison with related work in Appendix B.
this section cite: ['b14', 'b68', 'b20', 'b6', 'b7', 'b49', 'b34', 'b68', 'b20', 'b57', 'b35', 'b24', 'b19', 'b5', 'b28']

Section: Notation.
For any n ∈ N, let [n] := {1, 2, . . . , n}. For a positive semi-definite (PSD) matrix S, denote by µ j (S) its j-th largest eigenvalue, and define the S-induced norm ∥u∥ S := √ u ⊤ Su for any vector u. We write A ⪯ B (resp. A ⪰ B) if B -A (resp. A -B) is PSD. Throughout the paper, we use ≂ to denote equivalence up to a constant factor, and ≲ (resp. ≳) to denote an inequality up to a constant factor. For two nonnegative functions f, g : R ⩾0 → R ⩾0 , we write f (t) ≂ g(t) if there exist constants C 1 , C 2 > 0 (independent of t) such that C 1 f (t) ⩽ g(t) ⩽ C 2 f (t), ∀ t ⩾ 0.
• Top-M features: w j = e j for j ∈ [M ], i.e., selecting the top-M features {ϕ j } M j=1 ; • Random-M features: w j ∼ N (0, I N ) independently for j ∈ [M ].
The top-M setting is a particularly simple yet analytically representative case, widely adopted in prior scaling-law studies [43,13]. For random features [3,7,50,35,8], we will show that, in certain regimes, their scaling behavior closely parallels that of the top-M case. As clarified in Appendix A.3, our setup is equivalent to learning with the kernel K ϕ (x, x ′ ) := ϕ(x) ⊤ ϕ(x ′ ).
We now formalize the key notions of model capacity and task difficulty. Let ϕ j := ϕ j /λ 1/2 j for j ∈ [N ]. So { ϕ j } N j=1 forms an orthonormal basis of L 2 (D). Assumption 2.3 (Model capacity). The spectrum of the feature map satisfies λ j ≂ j -β , β > 1.
The condition β > 1 ensures tr(H) = N j=1 λ j ⩽ C for some constant C independent of N , making our analysis dimension-free and applicable to the infinite-dimensional setting (N = ∞).
For the top-M features, the model takes the form
f (•; v) = M j=1 v j ϕ j = M j=1 v j λ 1/2 j
ϕ j ≂ M j=1 v j j -β/2 ϕ j reveals that higher-index (less significant) features are increasingly down-weighted by the factor j -β/2 . As β increases, the spectrum decays more rapidly, and the model effectively relies on fewer features. Hence, the model's expressive power is governed by two complementary factors: (i) the model size M , which controls how many features are retained, and (ii) the capacity exponent β, which controls how quickly these features decay in importance.
Assumption 2.4 (Task difficulty). Suppose |θ * j | 2 ≂ j -1 λ s-1 j
for some s > 0.
this section cite: ['b42', 'b12', 'b2', 'b6', 'b49', 'b34', 'b7']

Section: Under Assumptions 2.3 and 2.4, the target function admits the expansion
f * = N j=1 θ * j ϕ j ≂ N j=1 j -1/2 λ s/2 j ϕ j ≂ N j=1 j -(sβ+1)/2 ϕ j .
Since { φj } are orthonormal, this assumption implies that the spectral energy of f * decays as a power law. The exponent α := sβ therefore quantifies the task's intrinsic difficulty, which depends only on the target function itself and is independent of the model spectrum. In contrast, s measures the relative difficulty with respect to a model of capacity β: for a fixed f * (fixed α), adopting a higher-capacity model (smaller β) increases s = α/β, making the task relatively easier. In other words, the same task appears easier to a higher-capacity model.
We remark that similar assumptions have been widely used in the analysis of kernel methods [12,11,57,9,39]. Our work builds upon and extends this line of research.
this section cite: ['b11', 'b10', 'b56', 'b8', 'b38']

Section: One-Pass SGD and Intrinsic-Time SDE
Given a data point z = (x, y) ∈ X ×R and a model f (•; v), define the loss ℓ(z, v) = 1 2 f (x; v)-y 2 .
Then, the population risk is R(v) = E z [ℓ(z, v)] = 1 2 ∥W ⊤ v -θ * ∥ 2 H + σ 2 2 =: E(v) + σ 2 2
, where E(v) denotes the excess risk. We minimize R(v) via one-pass SGD, given by
v k+1 = v k - η k B k z∈S k ∇ v ℓ(z, v k ),(3)
where S k := {(x k,j , y k,j )} B k j=1 is a mini-batch of i.i.d. samples, η k and B k are the learning rate and batch size, respectively. The initialization is set to v 0 = 0.
Throughout, we refer to η := (η 0 , η 1 , . . . , η K-1 ) as the learning rate schedule (LRS). Common choices in practice include the cosine [37,60], WSD [21], and multi-step [5]
this section cite: ['b36', 'b59', 'b20', 'b4']

Section: schedules (see Appendix A.2 for details).
To analyze the effect of LRS, we rewrite (3) as
v k+1 = v k -η k ∇R(v k ) + ξ k ,(4)
where the gradient noise
ξ k = 1 B k z∈S k ∇ℓ(z, v k ) -∇R(v k ) satisfies E[ξ k ] = 0, E[ξ k ξ ⊤ k ] = 1 B k Σ(v k ),
with Σ(•) denoting the noise covariance for batch size 1.
Continuous-time limit. Following prior work [30,31,32,33,51,34], we analyze the continuoustime limit of SGD rather than the discrete update (3) or (4). This perspective makes the analysis more tractable and clarifies the emergence of scaling laws. Fix a discretization step size h > 0 and let φ k := η k /h for k ∈ N. Then, (4
) becomes v k+1 = v k -φ k ∇R(v k )h -φ k hξ k .
For sufficiently small h, this iteration is well approximated by the Itô-type SDE [31,45]:
dv τ = -φ(τ )∇R(v τ ) dτ + φ(τ ) h b(τ ) Σ(v τ ) dB τ ,(5)
where B τ ∈ R M is an M -dimensional Brownian motion, and φ(•) is the continuous-time LRS satisfying φ(kh) = η k /h for all k ∈ N; b(•) is the continuous-time batch-size schedule satisfying b(kh) = B k for all k ∈ N. In (5), the learning rate affects both the drift and diffusion terms, thereby coupling the deterministic and stochastic effects.
this section cite: ['b29', 'b30', 'b31', 'b32', 'b50', 'b33', 'b30', 'b44']

Section: Intrinsic-time reparametrization.
In SDE (5), the physical time τ serves as the continuous analogue of the discrete step index k. However, when the learning rate varies over time, the actual training progress is determined not by the number of updates k but by the accumulated step size k j=1 η j , which more faithfully reflects the total optimization effort. Motivated by this observation, we introduce an intrinsic time variable that rescales the physical time τ according to the LRS:
t = T (τ ) := τ 0 φ(r) dr,(6)
which measures the LRS-adjusted training duration. Let ν t = vT -1 (t) . Applying Øksendal's time change formula [44] to the SDE (5) yields the intrinsic-time SDE:
dν t = -∇R(ν t ) dt + γ(t) Σ(ν t ) dB t with γ(t) = hφ(T -1 (t)) b(T -1 (t)) .(7)
Here γ(t) quantifies the joint effect of learning-rate and batch-size scheduling. Compared with (5), the LRS dependence is absorbed from the drift and retained only in the diffusion term, thereby decoupling the deterministic and stochastic effects. This structural simplification greatly facilitates the subsequent scaling analysis.
For a clearer explanation of the connection between the discrete SGD (4) and the SDE formulations ( 5) and ( 7), we refer the reader to Appendix A.4.
this section cite: ['b43']

Section: Intrinsic-Time Functional Scaling Laws
In this section, we present our main results on the Functional Scaling Law (FSL). All proofs are deferred to Appendix D. We begin with assumptions on the learning-rate schedule and model size. Assumption 4.1. Suppose Assumptions 2.1, 2.3 and 2.4 hold. Assume both M and N -M are sufficiently large, and the LRS satisfies sup t⩾0 γ(t) ⩽ C 3 for a sufficiently small constant C 3 > 0.
Theorem 4.2 (Intrinsic-Time FSL, top-M features, hard-regime). Under Assumption 4.1, let ν t denote the solution to the intrinsic-time SDE (7) with top-M features. Then, for f * with difficulty s ∈ (0, 1 -1/β] and any σ ⩾ 0, it holds for all t ⩾ 0 that
E[R(ν t )] -1 2 σ 2 ≂ M -sβ + e(t) + t 0 K(t -z)[e(z) + σ 2 ]γ(z) dz,(8)
where e(t) := (1 + t) -s , K(t) := (1 + t) -(2-1/β) .
This theorem establishes that, for hard tasks with s ⩽ 1 -1/β, the loss dynamics are fully characterized by the FSL (8). We explain the emergence of power laws in FSL from a multi-task learning perspective in Appendix A.5. Moreover, each term in the FSL (8) admits a clear interpretation:
• Irreducible error: 1 2 σ 2 . This term is due to label noise.
• Approximation error: M -sβ . This term corresponds to the error due to finite model size, with the scaling efficiency is determined by the task's intrinsic difficulty sβ.
• Signal learning: e(t). This term corresponds to learning under full-batch gradient descent, capturing the rate at which SGD extracts the signal f * . Moreover, the rate depends on the task's relative difficulty s. For a fixed target f * (fixed α = sβ), increasing model capacity (smaller β) accelerates its convergence since s = α/β becomes larger.
• Noise accumulation: t 0 K(t -z)[e(z) + σ 2 ]γ(z) dz. This term characterizes how the learningrate and batch-size schedules shape the accumulation and dissipation of stochastic noise. The integrand [e(z) + σ 2 ]γ(z) represents the instantaneous noise magnitude, where e(z) captures mini-batch noise and σ 2 captures label noise. The forgetting kernel K(•) quantifies how noise injected at time z still affects the loss at time t. Due to K(t) ≂ t -(2-1/β) , a higher-capacity model (smaller β) tends to forget noise more slowly.
Notably, the last two terms together constitute the optimization error and two key factors govern the trade-off between the them: (i) Model capacity: Increasing model capacity (β ↓) accelerates signal learning but simultaneously slows noise forgetting. (ii) Learning-rate and batch-size schedules: Smaller learning rates or larger batch sizes suppress noise injection but also shorten the intrinsic training time. However, sufficient intrinsic time is important: the signal-learning term requires it to effectively reduce the risk, while the noise-forgetting term relies on it to forget noise memorized in early training. Hence, effective schedules must balance these competing objectives-suppressing injected noise while maintaining enough intrinsic time for both learning and forgetting.
this section cite: ['b6', 'b7']

Section: General Results
The FSL (8) is established for the hard-learning regime where s ⩽ 1 -1/β. We now show that an analogous FSL also holds in the general case. To state the result, we define e
M (t) = M j=1 λ j |θ * j | 2 e -2λj t , K M (t) = M j=1 λ 2 j e -2λj t .
One can verify that both functions exhibit powerlaw decay for 1 ≲ t ≲ M β :
e M (t) ≂ t -s , K M (t) ≂ t -(2-1/β) , 1 ≲ t ≲ M β .(9)
Consequently, e ∞ (t) ≂ e(t) and K ∞ (t) ≂ K(t) for t ⩾ 0.
The following theorem provides a characterization of the loss dynamics for general case:
Theorem 4.3 (Intrinsic-Time FSL, top-M features, general label noise). Suppose Assumption 4.1 holds. Let ν t denote the solution to the intrinsic-time SDE (7) with the top-M features. Define
F M (t; γ) = e M (t) + t 0 K M (t -z)[e M (z) + σ 2 ]γ(z) dz.
There exists a c > 0 such that for 0 ⩽ t ⩽ cM β , it holds that
E[R(ν t )] -1 2 σ 2 ≂ M -sβ + F ∞ (t; γ). (10
)
For all cM β ⩽ t < ∞, it holds that
M -sβ + F M (t; γ) ≲ E[R(ν t )] -1 2 σ 2 ≲ M -sβ + F ∞ (t; γ).(11)
Notably, the constants implicit in ≂, ≲ are independent of the noise level σ.
A proof sketch is provided in Appendix C.3. The above characterization is uniform with respect to the label-noise level σ, and holds for all s > 0 and β > 1. It asserts that the exact FSL relation (10) (i.e., the FSL (8)) remains valid up to the intrinsic time t ⩽ cM β =: t M . For later times t > t M , although the FSL may no longer hold exactly, the loss dynamics remain controlled from both sides as in (11).
At the critical time t M , we have e M (t M ) ≂ M -sβ , indicating that signal learning has reached the approximation-error limit. Beyond this point, further training no longer improves the learned signal; instead, the dynamics become dominated by noise effects. Depending on the interaction between the stochastic gradient noise and the decaying learning rate, additional training may either inject more noise or dissipate it. Thus, it is a priori unclear whether the total error will significantly increase or decrease after t M . The upper bound in (11) ensures that the overall loss remains well-controlled, analogous to the behavior of the infinite-width limit (M = ∞).
Nevertheless, an FSL may still hold for all t ⩾ 0, under suitably stronger conditions. In Theorem 4.2, we considered the setting with tasks satisfying s ⩽ 1 -1/β. The following result shows that a similar characterization extends to general task difficulty with constant label noise.
Theorem 4.4 (Intrinsic-Time FSL, top-M features, constant label noise). Under Assumption 4.1, suppose σ ≳ 1. Let ν t denote the solution to the intrinsic-time SDE (7) with the top-M features. Then, for any s > 0 and all t ⩾ 0, E[R(ν t )] -1 2 σ 2 ≂ M -sβ + F M (t; γ).
Theorem 4.2 implies that the finite-M functions e M and K M can be replaced by their infinite-width counterparts e ∞ and K ∞ in the hard-learning regime. The next result demonstrates that the same FSL characterization naturally extends to the noiseless case σ = 0. Theorem 4.5 (Intrinsic-Time FSL, top-M features, zero label noise). Suppose Assumption 4.1 holds and σ = 0. Let ν t denote the solution to the intrinsic-time SDE (7) with the top-M features. If s ∈ [0, 2 -1/β], then for all t ⩾ 0, E[R(ν t )] ≂ M -sβ + e M (t) + t 0 K M (tz) e M (z) γ(z) dz.
Random-M features. For the random-features case, the modified feature covariance matrix is H = WHW ⊤ , whose eigenvalues we denote by
λ 1 ⩾ λ 2 ⩾ • • • ⩾ λ M . We similarly define e M (t) = M j=1 λ j |θ * j | 2 e -2 λj t , K M (t) = M j=1 λ 2 j e -2 λj t .
The next theorem establishes that the same FSL also holds when the top-M features are replaced by randomly selected features. Theorem 4.6 (Intrinsic-Time FSL, random-M features). Suppose Assumption 4.1 holds and s ∈ (0, 1]. Let ν t denote the solution to the intrinsic-time SDE (7) with the random-M features. Then, with probability at least 1 -exp(-Ω(M )) over the randomness of the projection matrix W, the results of Theorems 4.3, 4.4, and 4.5 continue to hold, after replacing e M (•) and K M (•) with their random-feature counterparts e M (•) and K M (•), respectively. Lemma 4.7. With probability at least 1 -exp(-Ω(M )) over the randomness of the projection matrix W, it holds that λ j ≂ λ j ≂ j -β for any j ∈ [M ].
Theorem 4.6 and Lemma 4.7 together imply that when the task difficulty satisfies s ⩽ 1, training with random-M features is similar to using the top-M features, up to exponentially small probability. We emphasize, however, that for easier tasks with s > 1, the behaviors of random and top feature may diverge and we leave this for future investigation.
this section cite: ['b10', 'b10']

Section: Learning Rate Schedules Impact Scaling Efficiency
Having established the general FSL, we now instantiate it under three representative LRSs-constant, exponential decay, and warmup-stable-decay (WSD)-to examine how schedule design influences scaling efficiency. All proofs can be found in Appendix E. For clarity, we make: Assumption 5.1. Assume constant label noise σ 2 ≳ 1 and batch size b(τ ) = B for all τ ⩾ 0.
Under this assumption, given a physical-time LRS function φ(•), Theorem 4.4 implies that the FSL for t ≳ 1 simplifies to E[R(ν t )] 1  2 σ 2 denote the expected excess risk after K training steps. For each LRS, we derive concrete scaling laws describing how E K scales with the model size M , the total step count K, as well as the LRS's hyperparameters. We then reinterpret these results from a resource-allocation perspective by optimizing under two canonical constraints: (i) the data-limited regime, where the total data size D := BK is fixed; and (ii) the compute-limited regime [20], where the total compute C := M D is fixed. For each regime, we further examine how the optimally tuned hyperparameters (e.g., the peak learning rate) should scale with increasing available resources. Finally, for clarity, we distinguish between two task regimes: an easy-learning regime, where s ⩾ 1 -1/β, and a hard-learning regime, where s < 1 -1/β.
-1 2 σ 2 ≂ M -sβ + e M (t) + σ 2 B t 0 K M (t -r)φ(T -1 (r)) dr. Let E K = E[R(ν Kh )] -
this section cite: ['b19']

Section: Constant LRS
Theorem 5.2 (Scaling law for constant LRS). Under Assumption 5.1, we have E K ≂ M -sβ + (ηK) -s + η B σ 2 . Let γ := η/B be the effective learning rate. Then, the scaling law can be rewritten as E K ≂ M -sβ +(γD) -s +γσ 2 =: h(γ, M, D), where the excess risk depends the learning rate via γ = η/B. This suggests that we should scale the learning rate linearly with respect to batch size (a.k.a. linear scaling rule) [26,16,40].
Data-optimal scaling. Clearly, this involves minimizing h(•) while keeping D fixed. A straightforward calculation yields:
γ opt ≂ D -s s+1 , M opt ≳ D 1 (1+s)β , E opt ≂ D -s s+1
. Notably, both the best achievable excess risk E opt and optimal learning rate γ opt depend exclusively on the task's relative difficulty s. For a fixed target (fixed α), a higher-capacity model (smaller β) gives a larger s = α/β and is therefore more data-efficient.
Compute-optimal scaling. This involves minimizing h(•) while keeping C := DM fixed. The solution is summarized as follows, with the derivation deferred to Appendix E.1:
γ opt ≂ C -sβ 1+(s+1)β , M opt ≂ C 1 1+(s+1)β , D opt ≂ C (s+1)β 1+(s+1)β , E opt ≂ C -sβ 1+sβ+β .
This shows that the performance of the compute-optimal model improves with the total compute budget C in a power law. For a fixed task (α = sβ fixed), we have the following observations:
• Increasing model capacity ( β ↓) enhances compute efficiency-the extra β in the scaling exponent sβ 1+sβ+β quantifies this gain. This explains a well-known empirical observation in LLM pre-training: Large models are more compute-efficient than small models [25,20].
• The optimal learning rate γ opt decreases as C grows, and the compute-optimal allocation favors investing more in data than in model size-again consistent with current LLM pre-training practice [5,55,20].
Note that [8] also investigated compute-optimal scaling for constant LRS but assumed a fixed learning rate and no label noise. In contrast, we consider a more realistic scenario where the learning rate is optimally tuned and the irreducible risk is present, leading to a compute-optimal scaling law that matches empirical observations.
this section cite: ['b25', 'b15', 'b39', 'b24', 'b19', 'b4', 'b54', 'b19', 'b7']

Section: Exponential Decay LRS
For a given number of training steps K [15,65], an exponential decay (exp-decay) LRS is given by φ(τ ) = a exp(-λτ ), φ(Kh) = b, where λ is chosen such that φ(Kh) = b. For brevity, we assume h = 1. Note that the hyperparameters a and b specify the peak and final learning rates, respectively.
this section cite: ['b14', 'b64']

Section: Theorem 5.3 (Scaling law for exp-decay LRS).

this section cite: []

Section: Under Assumption 5.1, we have E
K ≂ M -sβ + T -s + σ 2 b B + (a -b) min{M,T 1/β }
BT , where T = (ab)K/ log(a/b) is the total intrinsic time.
Let b = a/K. Then the intrinsic time becomes T = a(K -1)/ log K, whereas a constant LRS with step size η = a yields T = aK. Thus, exp-decay LRS drives the learning rate down to as small as a/K, yet sacrifices only a logarithmic factor of intrinsic time compared to the constant schedule.
Data-optimal scaling. Let γ = a/B be the effective peak learning rate. By minimizing the right hand side of the exponential decay scaling law with respect to a, b, K, B, M under the constraint KB = D (see Appendix E.2), We obtain M opt = ∞ and
• For s ⩾ 1 -1 β , γ opt ≂ (D/ log D) -1+sβ-β 1+sβ
and E opt ≂ (D/ log D) -sβ sβ+1 .
• For s < 1 -1 β , γ opt ≂ 1 and E opt ≂ (D/ log D) -s .
Compared with the constant LRS, exp-decay LRS achieves a strictly faster decay of the excess risk, justifying the importance of learning-rate decay in stochastic optimization.
this section cite: []

Section: Compute-optimal scaling.
A straightforward calculation (see Appendix E.2) yields:
• For s ⩾ 1 -1 β , γ opt ≂ ( C log C ) -1+sβ-β 2+sβ , M opt ≂ ( C log C ) 1 2+sβ , D opt ≂ C 1+sβ 2+sβ (log C) 1 2+sβ
, and
E opt ≂ ( C log C ) -sβ 2+sβ .
• For s < 1 -
1 β , γ opt ≂ 1, M opt ≂ ( C log C ) 1 1+β , D opt ≂ C β 1+β (log C) 1 1+β , E opt ≂ ( C log C ) -sβ 1+β .
In the easy-learning regime, the excess-risk rate is determined solely by the intrinsic difficulty α = sβ; hence, increasing model capacity alone does not lead to asymptotic gains. The compute-optimal allocation consistently favors data over model and moreover, the optimal compute split depends solely on the task's intrinsic difficulty, with ratio D opt /M opt ≂ C α/(2+α) decreasing as the task becomes harder. This implies that, for harder tasks, one should allocate more compute to increasing model size. The optimal γ opt decreases with the compute budget C, and for fixed α, higher-capacity models (β ↓) require smaller γ opt .
In the hard-learning regime, data still dominates compute allocation, but now the optimal split depends only on model capacity, independent of the task difficulty. Moreover, the optimal maximal learning rate remains constant (γ opt ≂ 1). These results imply that a single, universal choice of compute split and learning rate suffices to attain optimal scaling across all tasks satisfying s < 1 -1/β, greatly simplifying hyperparameter tuning. Finally, in this regime, higher-capacity models (β ↓) become strictly more compute-efficient, as evidenced by the excess-risk scaling exponent -sβ/(1 + β).
this section cite: []

Section: WSD-like LRS
We lastly turn to consider a WSD-like LRS [69,21], which comprises a K 1 -step stable phase followed by a K 2 -step decay phase, for a total K = K 1 + K 2 steps, given by
φ(τ ) = a , if τ ⩽ K 1 h; a exp(-λ(τ -K 1 h)) , if τ > K 1 h. (12
)
where λ is chosen such that φ(Kh) = b. For brevity, we assume h = 1 and let r = K 2 /K. This schedule is thus characterized by three hyperparameters: the peak learning rate a, the final learning rate b, and the decay proportion r, which controls the duration of decay-phase. Theorem 5.4 (Scaling law for WSD-like LRS). Under Assumption 5.1, we have for the LRS (12):
E K ≂ M -sβ + (T 1 + T 2 ) -s + σ 2 b B + (a -b) min{M,T 1/β 2 } BT2
, where T 1 = aK 1 and T 2 = (ab)K 2 / log(a/b) denote the intrinsic training times of the stable and decay phases, respectively.
We see that WSD-like LRS can leverage the initial stable phase to boost the intrinsic training time. For a decay proportion r < 1, we have T = T 1 + T 2 ⩾ (1 -r)Ka, which far exceeds the the intrinsic time T ≂ aK/ log K achieved by the pure exp-decay LRS. Consequently, WSD removes logarithmic factors in the full-batch GD term, without altering the noise term's order as long as r > 0. Building on this insights, we show that WSD can indeed improve the scaling efficiency, as detailed below.
Data-optimal scaling. Assuming b = a/K, we have M opt = ∞ and
• For s ⩾ 1 -1 β , γ opt ≂ D -1+sβ-β 1+sβ (log D) β-1 1+sβ , r opt ∈ (0, 1), E opt ≂ D -sβ sβ+1 (log D) sβ-s 1+sβ .
• For s < 1 -
1 β , γ opt ≂ 1, r opt ≳ D sβ+1-β β-1 log D, E opt ≂ D -s .
Compared with the exp-decay LRS, both regimes enjoy a logarithmic improvement in excess-risk decay. In particular, for the hard-learning regime, the logarithmic factor disappears. This improvement requires the decay-phase duration only needs to scale sublinearly with D, as indicated by r opt → 0 as D → ∞. This matches the WSD practice in LLM pre-training, where the decay phase typically occupies only 10%-20% of the total training duration. Moreover, our theory suggests that for harder tasks, the decay fraction can be reduced further to enhance compute efficiency.
Compute-optimal scaling. Analogous improvements hold in the compute-limited regime. Assuming b = a/K and imposing the compute constraint M D = C, the compute-optimal satisfies:
• For s ⩾ 1 -1 β , γ opt ≂ ( C log C ) -1+sβ-β 2+sβ , r opt ∈ (0, 1), M opt ≂ ( C log C ) 1 2+sβ , D opt ≂ C 1+sβ 2+sβ (log C) 1 2+sβ
, and
E opt ≂ C -sβ 2+sβ (log C) sβ-s 2+sβ .
• For s < 1-
1 β , γ opt ≂ 1, r opt ≳ D -β-1-sβ β-1 log D, M opt ≂ C 1 1+β , D opt ≂ C β 1+β , E opt ≂ C -sβ 1+β .
this section cite: ['b68', 'b20']

Section: Experiments 6.1 Power-Law Kernel Regression
While the FSL is derived in the continuous-time limit, we now verify that it also accurately captures the loss dynamics and scaling behavior of the discrete-time SGD (3). Specifically, we consider the PLK regression with difficulty s = 0.5 and capacity β = 4, corresponding to a hard-learning regime and the results are shown in Figure 1.
FSL accurately captures the loss dynamics of SGD. Figure 1(left) compares the loss dynamics of SGD with the predictions of the FSL under three representative LRSs: cosine, WSD, and an unconventional cyclical schedule [56]. Across all cases, the FSL provides a remarkably accurate description of the SGD's loss evolution. Comparing the WSD and cosine schedules, we observe that the loss under WSD exhibits a slower decay during the stable phase but undergoes a much sharper drop once the decay phase begins, ultimately yielding a lower final loss. This seemingly counterintuitive two-phase dynamical behavior of WSD aligns well with empirical observations in practical LLM pre-training [21,64,58].
FSL predicts the scaling behavior of SGD. Figure 1(right) further validates the scaling laws derived in Section 5 for the three canonical LRSs-constant, exponential, and WSD-like (12). The results show that the final-step loss of SGD closely follows the theoretical predictions of FSL. Among these schedules, WSD yields the best scaling performance, followed by exponential decay, while the constant schedule performs the worst. More experiment details and additional results experiments with varying (s, β) and other LRSs are provided in Appendix C.1, and exhibit consistent behaviors.
this section cite: ['b55', 'b20', 'b63', 'b57', 'b11']

Section: LLM Pre-training
We now evaluate the practical utility of FSL as a surrogate model for capturing the loss dynamics of LLM pre-training. Specifically, three popular LRSs: cosine, WSD, and the 8-1-1 [5] are considered; see Figure 2b(left) for a visualization. In the 8-1-1 LRS, the learning rate is reduced by a factor √ 10 at 80% and 90% of the total token budget, yielding a final value that is 0.1 times the peak learning rate. For more experiment details, we refer to Appendix C.2.
FSL accurately fit and predict loss curves. We first quantify the descriptive and predictive power of FSL. Following the protocol of [59] and [38], we restrict attention to the post-warmup portion of the loss trajectory. Two Llama [60] models (400 M and 1 B) are trained on 20 B tokens under the three LRSs. For each model we (i) fit the FSL parameters on the loss curve obtained using the 8-1-1 LRS and (ii) deploy the fitted FSL to predict the loss curves of the cosine and WSD schedules. Figure 2a demonstrates that FSL not only fits the 8-1-1 trajectory accurately but also generalizes reliably to the unseen WSD and cosine schedules for both model sizes.
The FSL-optimal LRS is WSD-like. We next leverage the fitted FSL to design improved LRSs. Specifically, we numerically minimize the final-step loss over the space of LRSs using the fitted FSL. This experiment employs a 1B-parameter QwenMoE model [68], trained on 20B tokens using the same three LRSs. We fit the FSL using the trajectory from the 8-1-1 LRS and numerically solve for the FSL-optimal LRS. The model is then trained under this FSL-optimal LRS, using the same compute budget, and compared against the baseline LRSs. Figure 2b(left) shows that surprisingly, the FSL-optimal LRS is WSD-like and the decay phase drives the learning rate far below the conventional 0.1η max threshold. This echos recent empirical recommendations by [4,17]. Furthermore, Figure 2b(right) demonstrates that the FSL-optimal schedule yields a strictly lower final loss than all baselines, substantiating its practical relevance. Taken together, these results suggest that FSL is a faithful surrogate for studying LLM training dynamics and a principled tool for interpreting and designing LRSs in large-scale pre-training.
this section cite: ['b4', 'b58', 'b37', 'b59', 'b67', 'b3', 'b16']

Section: Conclusion
In this paper, we present a systematic study of how LRS shapes the loss dynamics in kernel regression. Specifically, we establish a novel functional-level scaling law, which precisely characterizes the loss dynamics of SGD for general learning LRSs. The utility of our FSL is demonstrated through detailed analyses of three widely used LRSs, providing theoretical justification for several prevailing practices in LLM pre-training-most notably, offering an explanation for the effectiveness of the empirically popular but previously less-understood WSD schedules.
Subsequently, [38] proposed the Multi-Power Law (MPL), which replaces the S 2 in the Momentum Law with additional power laws to better capture the progressive loss reduction induced by learningrate decay. Specifically, the MPL takes the following form:
L k (η) = L 0 + AS -κ 1 -LD(k),(13)
where
LD(k) := C k i=2 (η i-1 -η i )G(η -κ ′ i S i ), S i := i j=1 η j , G(x) := 1 -(C ′ x + 1) -κ ′′ .
Here L 0 , A, C, C ′ , κ, κ ′ , κ ′′ are constants.
this section cite: ['b37']

Section: References
Ref_id:b0 Title: Scaling laws for generative mixed-modal language models Year: (2023)
Ref_id:b1 Title: Scaling and renormalization in high-dimensional regression Year: (2024)
Ref_id:b2 Title: Explaining neural scaling laws Year: (2024)
Ref_id:b3 Title: Straight to zero: Why linearly decaying the learning rate to zero works best for llms Year: (2025)
Ref_id:b4 Title: DeepSeek LLM: Scaling open-source language models with longtermism Year: (2024)
Ref_id:b5 Title: Scaling optimal lr across token horizons Year: ()
Ref_id:b6 Title: A dynamical model of neural scaling laws Year: (2024)
Ref_id:b7 Title: How feature learning can improve neural scaling laws Year: (2024)
Ref_id:b8 Title: Spectrum dependent learning curves in kernel regression and wide neural networks Year: (2020)
Ref_id:b9 Title: Convex optimization: Algorithms and complexity Year: (2015)
Ref_id:b10 Title: Optimal rates for the regularized least-squares algorithm Year: (2007)
Ref_id:b11 Title: Fast rates for regularized least-squares algorithm Year: (2005)
Ref_id:b12 Title: Scaling law for stochastic gradient descent in quadratically parameterized linear regression Year: (2025)
Ref_id:b13 Title: A tale of tails: Model collapse as a change of scaling laws Year: (2024)
Ref_id:b14 Title: The step decay schedule: A near optimal, geometrically decaying learning rate procedure for least squares Year: (2019)
Ref_id:b15 Title: Accurate, large minibatch SGD: training imagenet in 1 hour Year: (2017)
Ref_id:b16 Title: Scaling laws and compute-optimal training beyond fixed training durations Year: (2024)
Ref_id:b17 Title: Scaling laws for autoregressive generative modeling Year: (2020)
Ref_id:b18 Title: Deep learning scaling is predictable, empirically Year: (2017)
Ref_id:b19 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b20 Title: Unveiling the potential of small language models with scalable training strategies Year: (2024)
Ref_id:b21 Title: Learning curve theory Year: (2021)
Ref_id:b22 Title: Scaling laws for learning with real and surrogate data Year: (2024)
Ref_id:b23 Title: Power laws for hyperparameter optimization Year: (2023)
Ref_id:b24 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b25 Title: One weird trick for parallelizing convolutional neural networks Year: (2014)
Ref_id:b26 Title: Scaling laws for precision Year: (2024)
Ref_id:b27 Title: Trajectory of mini-batch momentum: batch size saturation and convergence in high dimensions Year: (2022)
Ref_id:b28 Title: Predictable scale: Part I -optimal hyperparameter scaling law in large language model pretraining Year: (2025)
Ref_id:b29 Title: Stochastic modified equations and adaptive stochastic gradient algorithms Year: (2017)
Ref_id:b30 Title: Stochastic modified equations and dynamics of stochastic gradient algorithms I: Mathematical foundations Year: (2019)
Ref_id:b31 Title: Reconciling modern deep learning with traditional optimization analyses: The intrinsic learning rate Year: (2020)
Ref_id:b32 Title: On the validity of modeling SGD with stochastic differential equations (SDEs) Year: (2021)
Ref_id:b33 Title: What happens after SGD reaches zero loss? -a mathematical framework Year: (2022)
Ref_id:b34 Title: Scaling laws in linear regression: Compute, parameters, and data Year: (2024)
Ref_id:b35 Title: DeepSeek-V3 technical report Year: (2024)
Ref_id:b36 Title: SGDR: Stochastic gradient descent with warm restarts Year: (2016)
Ref_id:b37 Title: A multi-power law for loss curve prediction across learning rate schedules Year: (2024)
Ref_id:b38 Title: A solvable model of neural scaling laws Year: (2022)
Ref_id:b39 Title: An empirical model of large-batch training Year: (2018)
Ref_id:b40 Title: Generalization error of random feature and kernel methods: hypercontractivity and kernel matrix concentration Year: (2022)
Ref_id:b41 Title: The quantization model of neural scaling Year: (2024)
Ref_id:b42 Title: An exactly solvable model for emergence and scaling laws Year: (2024)
Ref_id:b43 Title: Stochastic differential equations Year: (2003)
Ref_id:b44 Title: Continuous-time models for stochastic optimization algorithms Year: (2019)
Ref_id:b45 Title: Understanding LLM behaviors via compression: Data generation, knowledge acquisition and scaling laws Year: (2025)
Ref_id:b46 Title: Sgd in the large: Average-case analysis, asymptotics, and stepsize criticality Year: (2021)
Ref_id:b47 Title: Dynamics of stochastic momentum methods on largescale, quadratic models Year: (2021)
Ref_id:b48 Title: Homogenization of sgd in high-dimensions: Exact dynamics and generalization properties Year: (2024)
Ref_id:b49 Title: 4+3 phases of compute-optimal neural scaling laws Year: (2024)
Ref_id:b50 Title: Implicit bias of SGD for diagonal linear networks: a provable benefit of stochasticity Year: (2021)
Ref_id:b51 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b52 Title: The surprising agreement between convex optimization theory and learning-rate scheduling for large model training Year: (2025)
Ref_id:b53 Title: A neural scaling law from the dimension of the data manifold Year: (2020)
Ref_id:b54 Title: Scaling law for language models training considering batch size Year: (2024)
Ref_id:b55 Title: Cyclical learning rates for training neural networks Year: (2017)
Ref_id:b56 Title: Asymptotic learning curves of kernel methods: empirical data versus teacher-student paradigm Year: (2020)
Ref_id:b57 Title: Open agentic intelligence Year: (2025)
Ref_id:b58 Title: Scaling law with learning rate annealing Year: (2024)
Ref_id:b59 Title: LLaMA 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b60 Title: High-dimensional statistics: A non-asymptotic viewpoint Year: (2019)
Ref_id:b61 Title: A theoretical analysis of noise geometry in stochastic gradient descent Year: (2023)
Ref_id:b62 Title: More than a toy: Random matrix models predict how real-world neural representations generalize Year: (2022)
Ref_id:b63 Title: Understanding warmup-stable-decay learning rates: A river valley loss landscape perspective Year: (2024)
Ref_id:b64 Title: Last iterate risk bounds of SGD with decaying stepsize for overparameterized linear regression Year: (2022)
Ref_id:b65 Title: The implicit regularization of dynamical stability in stochastic gradient descent Year: (2023)
Ref_id:b66 Title: The alignment property of SGD noise and how it helps select flat minima: A stability analysis Year: (2022)
Ref_id:b67 Title: Qwen2.5 technical report Year: (2024)
Ref_id:b68 Title: Scaling vision transformers Year: (2022)
Ref_id:b69 Title: How does critical batch size scale in pre Year: (2024)
