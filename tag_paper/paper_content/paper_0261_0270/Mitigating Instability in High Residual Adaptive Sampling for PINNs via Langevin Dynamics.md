Title: Mitigating Instability in High Residual Adaptive Sampling for PINNs via Langevin Dynamics
Abstract: Recently, physics-informed neural networks (PINNs) have gained attention in the scientific community for their potential to solve partial differential equations (PDEs). However, they face challenges related to resource efficiency and slow convergence. Adaptive sampling methods, which prioritize collocation points with high residuals, improve both efficiency and accuracy. However, these methods often neglect points with medium or low residuals, which can affect stability as the complexity of the model increases. In this paper, we investigate this limitation and show that high residual-based approaches require stricter learning rate bounds to ensure stability. To address this, we propose a Langevin dynamics-based Adaptive Sampling (LAS) framework that is robust to various learning rates and model complexities. Our experiments demonstrate that the proposed method outperforms existing approaches in terms of relative L 2 error, and stability across a range of environments, including high-dimensional PDEs where Monte Carlo integration-based methods typically suffer from instability. The implementation code is publicly available at https://github.com/neurips2025-las/LAS-implementation.

Section: Introduction
Partial differential equations (PDEs) describe a wide range of physical phenomena, including heat transfer [17,8], fluid flow [37,18,33], wave propagation [34,5], optics, and epidemiology [26,36]. Accurate and efficient PDE solutions are vital across many industries. With recent advances in deep learning, physics-informed neural networks (PINNs) have emerged as a promising approach for solving PDEs. PINNs train by minimizing errors from initial conditions (IC), boundary conditions (BC), and PDE residuals at collocation points [31,49,12,41,23]. These error terms are treated as soft constraints, guiding the model to satisfy essential physical requirements. This collocation-based learning method enhances the capability of PINNs by reducing the need for extensive experimental data collection across spatio-temporal ranges, demonstrating success in various industries as a promising alternative to traditional numerical methods such as the finite difference method and the finite element method [50,2,25]. However, collocation-based PINN (hereafter referred to as PINNs) encounter challenges in efficiently setting collocation points within the constraints of a limited sampling budget and in achieving fast convergence to accurate solutions.
A key challenge arises from the presence of small regions with abrupt changes, in contrast to larger, smoother regions. This issue is particularly evident in stiff PDEs, which are often characterized by discontinuities, such as sudden transitions or jumps across the spatio-temporal domain.
Adaptive sampling in PINNs mainly follows two strategies: residual distribution-based methods, which resample points proportionally to residual magnitudes, and high residual-based methods, which prioritize large-residual points while neglecting low-residual regions. The latter often yields strong empirical results but risks distorting the residual distribution. This raises a key question: should adaptive sampling focus exclusively on high residuals? Addressing this question requires a careful analysis of the trade-offs and risks inherent in different sampling schemes.
In this work, we respond by proposing a principled alternative: Langevin dynamics-based adaptive sampling (LAS). Through a series of theoretical and empirical analyses, we demonstrate that LAS consistently achieves reliable relative L 2 errors and stable convergence across diverse PDEs, architectures, and hyperparameters, outperforming existing methods in robustness and scalability.
2 Background and Related Work Physics-informed neural networks. The basic PINN framework [35] utilizes deep neural networks as function approximators f θ to estimate the solution u of a non-linear PDE: u t + N x [u] = 0, x ∈ X ⊂ R d , t ∈ [0, T ];
(2.1)
u(x, 0) = h(x), x ∈ X ⊂ R d ; (2.2) u(x, t) = g(x, t), x ∈ ∂X ⊂ R d , t ∈ [0, T ],(2.3)
where u(x, t) denotes the hidden solution at spatial and temporal coordinates x, t, N x [•] is the nonlinear differential operator, X is the spatial domain, ∂X is the boundary, and T is the time range. The spatio-temporal domain is Ω = X × [0, T ], with collocation point x = (x, t) ∈ Ω. The PDE residuals R θ (x) and loss function on collocation points P = {x n } N pde n=1 ⊂ Ω are calculated as:
R θ (x) = ∂ ∂t f θ (x) + N x [f θ ](x), x ∈ Ω; (2.4) L pde ({x n }; θ) = E x∼U (Ω) |R θ (x)| k ≈ 1 N pde N pde n=1 |R θ (x n )| k , (2.5)
where U(Ω) is the uniform distribution over Ω and N pde represents the number of sample points of PDE loss. Then, in a similar manner, the total loss function L is defined as: L({x total n }; θ) = λ pde L pde ({x n }; θ) + λ ic L ic ({x ic n }; θ) + λ bc L bc ({x bc n }; θ). Hyperparameters λ pde , λ ic , and λ bc control the balance between the PDE, IC, and BC loss terms. Then, f θ is trained to estimate appropriate solution u for PDEs by minimizing the total loss L.
Adaptive sampling based on residual distribution. Classical PINNs generally adopt uniform collocation sampling. To improve efficiency, a residual-based adaptive sampling strategy was proposed [31], where each point x n is drawn with probability p(x n ) := |R θ (xn)| k m |R θ (xm)| k . Building on this idea, the residual-based adaptive distribution (RAD) [47] introduces a hyperparameter c: p(x)
∝ |R θ (x)| k E[|R θ (x)| k ] + c.
Here, k highlights high-residual regions, while c enforces a degree of uniformity. By tuning these parameters, RAD flexibly balances exploration and exploitation depending on the problem structure. More recently, Gaussian mixture distribution-based adaptive sampling (GAS) has been proposed [20], where high-residual validation points serve as the means of a Gaussian mixture model (GMM), and the reciprocals of residual gradients determine the diagonal covariances.
Adaptive sampling focused on high residuals. Alongside residual distribution-based approaches, another major line of work explores sampling methods that do not explicitly approximate the underlying distribution. These methods can often be viewed as special cases of RAD under extreme k, c settings, but we categorize them here based on whether they estimate a sampling distribution.
A representative approach is residual-based adaptive refinement (RAR) [27], which iteratively selects the top-M high-residual points until the mean residual falls below a tolerance. Although effective, RAR continually accumulates points and increases computational cost. To improve efficiency, the R3 method [10] retains high-residual points, uniformly resamples for diversity, and discards lowresidual points. Similarly, failure-informed PINNs (FI-PINNs) adaptively sample from regions where residuals exceed a threshold, using strategies such as self-adaptive importance sampling or subset simulation [14,13]. Beyond these sampling strategies, recent work has questioned the appropriateness of the L 2 loss for solving Hamilton-Jacobi-Bellman (HJB) equations [43]. In response, adversarial training methods targeting the L ∞ norm have been introduced. As we discuss later, this framework leverages partial gradient information and inherently emphasizes high-residual regions in sampling.
Theoretical analysis of the concentration effect. While many adaptive sampling methods have shown promising results, several theoretical aspects remain unclear. In particular, there is a lack of theoretical analysis regarding the concentration effect. Although numerous studies report success with adaptive sampling, there is limited analysis on the impact of focusing primarily on high residuals.
To bridge this analytical gap, we investigate how learning stability is influenced by the extent to which high residuals are emphasized relative to model complexity. Our analysis reveals that an excessive focus on high residuals may lead to performance degradation in PINN training.
this section cite: ['b16', 'b7', 'b36', 'b17', 'b32', 'b33', 'b4', 'b25', 'b35', 'b30', 'b48', 'b11', 'b40', 'b22', 'b49', 'b1', 'b24', 'b34', 'b30', 'b46', 'b19', 'b26', 'b9', 'b13', 'b12', 'b42']

Section: Analysis of the Learning Stability

this section cite: []

Section: The Effect of Sampling Concentration
In this work, we define sampling concentration as the spatial aggregation of collocation points within regions characterized by large residual magnitudes. Such strategies have been shown to improve accuracy, efficiency [28,24], stability [7,45], and physical consistency [21,46,39] in PINN training. In this section, we analyze the effects of sampling concentration theoretically.
Setup. Consider the partial differential equation defined over the domain Ω = X × [0, T ]. Assume that we have N collocation points forming the sample population P = {x n } N n=1 ⊂ Ω, sampled from a uniform distribution U(Ω). Assumption 3.1. For analytical simplicity, we assume that the residual error of the PDE at each collocation point x n can be expressed as a linear combination of feature-mapped vectors, given an appropriate feature map ϕ : Ω → R D . Specifically, we represent the residual error as follows:
R θ (x n ) = ∂ ∂t f θ (x n ) + N x [f θ ](x n ) (3.1) = a(θ) ⊤ ϕ(x n ; θ) (3.2) = D d=1
a d (θ)ϕ d (x n ; θ).
(3.3)
We regard the sampling methodology as a weighting of each sample point depending on the residual R θ (x n ) and set k = 2. Thus, we can represent the loss function L(P; θ) 2 . Assume that we are solving for the solution based on the gradient descent (GD) algorithm. Then,
= N n=1 w n |R θ (x n )|
θ l+1 = θ l -η∇ θ N n=1 w l n |R θ l (x n )| 2 ,(3.4)
where the weights assigned to each sample point for iteration l are determined as follows:
w l n ∝ exp |R θl (x n )| 2 β 2
, n ∈ {1, ..., N }.
(3.5) Additionally, w l n is normalized to satisfy x∈P w l n (x) = 1 and the parameter β > 0 preceding the residual controls the concentration of sampling with respect to the residuals. Note that the parameters θl = ( θl 1 , . . . , θl D ) used to calculate the importance weights do not participate in the model parameter update process. Furthermore, in contexts where the meaning is clear, we will no longer explicitly indicate that ϕ is parameterized by θ, i.e., denote ϕ(x; θ) as ϕ(x).
For iteration l, we focus on two extreme cases of interest: when β is too large (uniform sampling), most samples receive uniform weights, resulting in uniform sampling. Conversely, when β is close to 0 (high residual sampling), the effect is dominated by the sample with the highest residual. To explore this in more depth, consider the following propositions. Proposition 3.1 (Steepness of the uniform sampling). When the sampling concentration parameter β is sufficiently large, the maximum eigenvalue of the Hessian of the loss function can be approximated as 2λ max (Σ), where Σ = E x∼U (Ω) [ϕ(x)ϕ(x) ⊤ ] and λ max (Σ) is the maximum eigenvalue of Σ.
this section cite: ['b27', 'b23', 'b6', 'b44', 'b20', 'b45', 'b38']

Section: Proposition 3.2 (Steepness of the high residual sampling).
When the sampling concentration parameter β is sufficiently small, the maximum eigenvalue of the Hessian of the loss function can be approximated as 2∥ϕ(x * )∥ 2 , where
x * = arg max x∈P |R θ (x)| 2 is the maximum residual point.
Detailed proof can be found in Appendix B.1, B.2. It is well known that to ensure the convergence of GD algorithms, the learning rate η must satisfy the following relationship with the largest eigenvalue λ max of the Hessian of the loss function: η < 2 λmax [6]. Therefore, we consequently aim to examine the relationship of the largest eigenvalue in two extreme cases of β. Before presenting the main result, we introduce two additional assumptions that formalize phenomena typically observed in neural network-based models as their complexity increases. This assumption is substantiated by both empirical evidence and theoretical insights. The heavy-tailed nature of feature vectors has been documented in several studies [29,30,3], and theoretically, in high-dimensional settings with complex dependencies, classical assumptions of Gaussianity often break down, and heavy-tailed models provide a more accurate fit to the observed distributional behavior [4,16,40]. Next, we impose an assumption concerning the representative characteristics encoded in the norms of feature vectors. Assumption 3.3. As the model complexity D increases, the feature vector with the maximal norm becomes increasingly representative, which results in the following:
max x∈P ∥ϕ(x)∥ ≈ ∥ϕ(x * )∥ where x * = arg max x∈P |R θ (x)| 2 is the maximum residual point.
This assumption can be seen as a concentration of measure phenomenon in high-dimensional spaces [11,42,32,15]. We note that Assumptions 3.2 and 3.3 do not generally hold in isotropic settings. However, they become statistically valid under (i) the low-temperature limit β → 0, and (ii) the high-dimensional regime D ≫ 1 with heavy-tailed feature norms. Specifically, writing R θ (x) = a(θ) ⊤ ϕ(x), we have |R θ (x)| 2 = ϕ(x) ⊤ Q ϕ(x) where Q = a(θ)a(θ) ⊤ is rank-one. In this context, we reason as follows to determine under what conditions ϕ(x) ⊤ Qϕ(x) ≈ ϕ(x) ⊤ ϕ(x) holds.
In the limit β → 0, the samples concentrate on regions with large R θ (x), thereby biasing ϕ(x) toward alignment with a(θ) and large norm. Moreover, for D ≫ 1, feature vectors are nearly orthogonal, so large |R θ (x)| 2 occurs only when ϕ(x) is both high-norm and well-aligned. Consequently, arg max x |R θ (x)| 2 ≈ arg max x ∥ϕ(x)∥ 2 holds with high probability in this structured regime.
To support the validity of Assumptions 3.1 through 3.3, we provide a consolidated empirical verification in Appendix A, demonstrating their general applicability across a broad range of settings. Finally, under the scaling assumption that the number of samples N grows linearly with model size D as N = cD, we derive our main theoretical result as follows.
Theorem 3.1. Given the heavy-tailed nature of ||ϕ(x)|| and sufficiently large model complexity D, we have 2∥ϕ(x * )∥ 2 ≫ 2λ max (Σ). This inequality establishes a tighter upper bound on the learning rate for ensuring the convergence of the GD algorithm under the high residual sampling method.
The detailed proof can be found in the Appendix B.3. This indicates that the stability of the algorithm can vary significantly depending on the sampling strategy and model complexity. Specifically, in these two extreme cases (β ≪ 1 and β ≫ 1), uniform sampling may struggle to find an appropriate solution due to the difficulty of the stiff PDE problems, while high residual sampling may fail due to instability in the learning process.
this section cite: ['b28', 'b29', 'b2', 'b15', 'b39', 'b10', 'b41', 'b31', 'b14']

Section: Characteristics of Adaptive Sampling Algorithms
Building upon the previous discussion, we now briefly review how existing adaptive sampling methods operate. In particular, we focus on how the trajectory of the sample population P evolves under these algorithms. • RAD [47]: The modeling of residual distribution is relatively straightforward and relies on Monte Carlo integration (MCI) over the expectation p(x)
∝ |R θ (x)| k E|R θ (x)| k + c.
In general, a larger value of k corresponds to a high residual regime with smaller β, whereas a larger value of c indicates a tendency toward uniform sampling regime with higher β.
• R3 [10]: R3 employs a strategy that consistently maintains high residuals, resulting in an excessive skew in the distribution of collocation points as iterations progress. Moreover, this approach may fail to effectively handle multi-modal landscapes in the long-term, which, as demonstrated in our previous theoretical analysis, results in a scenario where the sampling concentration parameter β becomes extremely small.
• L ∞ [43]: During the adversarial training, to estimate the inner maximal value
sup x∈Ω |R θ (x)| k , L ∞ iteratively utilizes gradient information sign∇ x |R θ (x)| k
, allowing for some degree of access to local modes. However, there is no guarantee that the relative proportions between modes of different heights are preserved.
To facilitate an intuitive understanding of time evolving sampling methods (R3, L ∞ ), we have illustrated the working mechanisms in a schematic diagram shown in Figure 1. For a detailed visualization of the sampling trajectories, we refer the reader to Appendix D.
this section cite: ['b46', 'b9', 'b42']

Section: Proposed Approach: Langevin Adaptive Sampling (LAS)
Similar to other residual distribution-based methodologies, our primary objective is to estimate the residual-based sampling distribution. However, unlike previous methods that directly model the distribution using residuals, we employ Langevin dynamics to model the target distribution. An intuitive visualization of our LAS framework is depicted in Figure 2. Algorithm 1 Single LAS Sampling Iteration for Physics-Informed Neural Networks 1: Input: initial population P = P 0 with N pde collocation points. 2: Output: updated population P = P lL . 3: for l = 0 to l L -1 do 4: for x l n ∈ P l do 5:
Calculate the residual gradient:
∇ x |R θ (x l n )| 2 = ∇ x ∂ ∂t f θ (x l n ) + N x [f θ (x l n )]2 . 6:
Sample white Gaussian noise: z l n ∼ N (z l n ; 0, I).
this section cite: []

Section: 7:
Follow the Langevin dynamics:
x l+1 n ← x l n + τ 2 ∇ x |R θ (x l n )| 2 + β √ τ z l n . 8:
end for 9:
Update collocation population:
P l+1 ← {x l+1 n } N pde n=1
. 10: end for
this section cite: []

Section: Langevin Dynamics and Stationary Distriburion
The dynamics of the collocation points P l ⊂ Ω at the l-th iteration in LAS are given as follows:
x l+1 n = x l n + τ 2 ∇ x |R θ (x l n )| 2 + β √ τ z l n ,(4.1)
where τ > 0 is the Langevin step size, z l n ∼ N (z l n ; 0, I) represents the white Gaussian noise, and β is the sampling concentration coefficient. Additionally, the residual exponent k is set to 2. Unlike other methods that estimate the sampling distribution based on residuals at every iteration, LAS dynamically updates the data points without requiring the estimation of the sampling distribution. If the Langevin dynamics are allowed to run for a sufficient number of iterations l L with a sufficiently small step size τ , we can theoretically derive the following result regarding the collocation points. Theorem 4.1 (Stationary distribution). For fixed f θ and concentration parameter β > 0, sample population P l asymptotically follows
lim l→∞ p l (x) = p ∞ (x) ∝ exp |R θ (x)| 2 β 2 .
Although the proof is well known [9], to complete the formulation of the distribution under consideration, we include the full derivation in Appendix C.1. The detailed operational procedure is summarized in Algorithm 1.
Unlike R3 and RAD, which use MCI to resample collocation points independently at each step, LAS refines the sampling iteratively by reusing the current population. Similar to L ∞ , it exploits gradient information. However, instead of relying solely on raw gradients, LAS injects noise into the signal-a design choice whose implications are outlined below. Theorem 4.2 (LAS favors flat residual surfaces). Given a fixed residual landscape R θ and an initial set of randomly sampled collocation points P 0 , the LAS framework progressively refines the sampling towards flatter local maxima while avoiding less stable and sharp regions.
An intuition-based explanation is provided in Appendix C.2 as a substitute for a formal proof. This phenomenon reflects a desirable property of Langevin dynamics, which inherently favors flatter regions of the residual landscape. As a result, when two local maxima exhibit similar residual values, the collocation points are more likely to concentrate near the flatter one. Consequently, models trained around such flat residual regions tend to exhibit more stable learning behavior.
this section cite: ['b8']

Section: Key Strengths (and Advantages) of the Proposed LAS Framework
Robustness in high-dimensional PDEs. In high-dimensional PDE settings, the MCI-based expectation E|R θ (x)| k ≈ 1 N N n=1 |R θ (x n )| k becomes unstable due to the curse of dimensionality, which demands exponentially more samples for accurate estimation. In contrast, LAS leverages local gradient information, maintaining stability even under high-dimensional conditions.
Enhanced stability through noise injection. Stochastic gradient descent (SGD) is known to prefer flatter minima, which are associated with better generalization [22,19,48]. Similarly, LAS injects noise into the gradient signal, resulting in significantly improved training stability compared to methods that rely solely on raw gradients [43]. It is reasonable to expect that sampling schemes biased toward flatter regions inherently promote more stable training dynamics.
this section cite: ['b21', 'b18', 'b47', 'b42']

Section: Experiments
This section presents an experimental evaluation of high-residual sampling under varying model complexities and learning rates, with the number of collocation points fixed. We compare the performance of our proposed LAS method against other adaptive sampling approaches, including RAD, R3, and L ∞ , each evaluated under the default settings provided in their original papers, random sampling with resampling (Random-R), which essentially corresponds to uniform sampling.
Experimental setup. As the default settings, unless otherwise specified, the models utilized a multilayer perceptron (MLP) with 128 nodes per layer and 4 hidden layers, employing a hyperbolic tangent activation function in each hidden layer. The Adam optimizer was utilized with the learning rate of η = 0.001 and a decay factor of 0.9 applied every 5, 000 iterations. Training was conducted with 200, 000 iterations, and the number of collocation points was set to N pde = 1, 000. For the LAS configuration, the residual exponent k = 2, the Langevin step size τ = 0.002 for 1-2D PDEs and τ = 0.01 for 4-8D PDEs, and the concentration parameter β = 0.2. This hyperparameter setting represents the empirically obtained optimum, with further discussion presented in the following subsection 5.5.
this section cite: []

Section: Ablation Studies
First and foremost, we sought to verify how the analytical results regarding stability and model complexity, presented in Section 3, operate and apply to the functioning of each algorithm. In this context, we performed the following key ablation studies based on the Allen-Cahn equation using 5 different random seeds.
this section cite: []

Section: Steepness of the loss landscape.
In our stability analysis, we hypothesized that sampling algorithms targeting extremely high residuals-such as R3 and L ∞ -would induce sharper loss landscapes. To test this, we tracked the maximum eigenvalue of the Hessian of the loss throughout training, as shown in Figure 3-(a). The results support our hypothesis, indicating that high-residual-focused sampling leads to increased steepness in the loss surface. The proposed LAS method, like L ∞ , leverages gradient information but achieves more stable training by reducing residual surface steepness, similar to Random-R. This stability likely stems from the injected noise term. Overall, LAS outperforms Random-R and other baselines in relative L 2 error while maintaining the lowest steepness.
Different number of hidden layers. We employed MLP architectures with hidden layers ranging from 4 to 10 across all sampling methods, maintaining a learning rate of 0.001 and utilizing a step scheduler. As illustrated in Figure 4-(a), it can be observed that as the number of layers increases, the overall performance improves; however, for most sampling strategies at 10 layers, except for LAS, the performance diverges. In particular, for R3, it is evident that it fails to converge more prominently compared to other algorithms. These results indicate that the sensitivity to model complexity varies depending on the sampling method. In particular, high residual methods are more susceptible to increasing model complexity. For more clarity, regarding the loss curves during the training of PINNs based on model complexity, analyses are presented in Appendix E, not only from the perspective of depth but also from the perspective of width expansion. Varying learning rate η without decaying. We evaluated MLPs with four hidden layers across learning rates ranging from 0.001 to 0.004 without applying decay. As shown in Figure 4 -(b), the benchmark algorithms demonstrated performance degradation at η = 0.002 compared to η = 0.001, whereas LAS showed improvement. At η = 0.003, all methods exhibited reduced performance; however, LAS was able to partially mitigate this degradation. At η = 0.004, none of the methods produced correct solutions. In particular, we visualized the performance for very low learning rates in Figure 4-(c) and highlighted the range between η = 0.002 and 0.003, where all algorithms begin to exhibit instability in Figure 4-(d). From this, we observe that learning does not proceed properly at very low learning rates, and for layer 4, most algorithms become unstable at a learning rate as low as approximately 0.0022.
this section cite: []

Section: Experiments on Representative 1-Dimensional PDEs
The proposed LAS framework is further evaluated on representative 1-dimensional PDEs derived from various benchmark problems tackled by several established algorithms, including RAD, R3, L ∞ , Random-R. In these evaluations, we also employ the default experimental settings as outlined earlier. The specific configurations for the PDE parameters and the hyperparameters of the baseline algorithms are detailed in Appendix F.
this section cite: []

Section: Experimental results.
We evaluated each sampling method on five benchmark PDEs-Burgers', Convection, Allen-Cahn, Korteweg-De Vries, Schrödinger-using five random seeds. As summarized in the 1D case of Table 1, Random-R tended to outperform other adaptive sampling strategies in high-complexity model settings. Meanwhile, LAS consistently achieved either the best or second-best relative L 2 errors across all cases. In particular, LAS outperformed all methods on the Allen-Cahn and Schrödinger equations, while remaining competitive on the others.
this section cite: []

Section: Experiments on High Dimensional PDEs
Beyond the one-dimensional setting, we applied the existing approaches and the proposed algorithm to higher-dimensional PDE problems. Benchmark problems include 2D Burgers', 2D heat [1], and dimension flexible heat (DF-heat) 4 to 8D equations [49]. These PDEs have analytics solutions, so the performance of all the sampling methods could be fairly observed and evaluated in high dimensional PDE cases. Especially, we modified the DF-heat PDE to pose a more challenging problem setting by increasing frequency. The details for PDEs are also described in Appendix F. We first report the performance of all sampling methods under the default settings from the original papers (Table 1).
this section cite: ['b0', 'b48']

Section: Experimental results.
In the 2D Burgers' and heat equations, Table 1 shows that Random-R and LAS exhibited competitive performance. RAD, R3, and L ∞ , on the other hand, experienced notable performance degradation, primarily because the 2D PDEs represent smooth cases rather than highfrequency or stiff regimes. For the 4D, 6D and 8D cases, LAS significantly outperformed all other adaptive sampling strategies. In the 4D, Random-R and L ∞ produced plausible solutions, but their performance remained inferior to that of LAS. Meanwhile, MCI-based methods such as RAD and R3 suffered from instability. Notably, for the 6D and 8D cases, all the other sampling approaches failed to find appropriate solutions. Although convergence was observed under different balance terms, the resulting performance was inferior and highly sensitive, as discussed in the following subsection.
this section cite: []

Section: Further Discussion on Loss Balancing, Model Hyperparameters, and Applicability
To ensure fair comparison, we conducted an extensive search for the optimal loss balance terms (λ ic , λ bc , λ pde ) and hyperparameter configurations of the baseline methods, as detailed in Appendix G. Additional experiments demonstrate that RAD and R3 outperform Random-R when equipped with the optimal loss balance terms and hyperparameter settings identified in Appendix G. However, both RAD and R3 exhibit sensitivity to the problem dimensionality and the choice of loss weighting. In contrast, LAS consistently attains either the best or second-best relative L 2 error across all evaluated scenarios, including smooth, stiff, low-dimensional, and high-dimensional cases. In terms of practical application, we further evaluate robustness across different model architectures (Appendix H), a factor that may be of particular relevance to practitioners.
this section cite: []

Section: LAS Hyperparameter Tuning and Computational Complexity
The practical applicability of LAS depends on the number of Langevin iterations l L , which directly affects computational complexity. We therefore compare its cost with existing sampling methods in Appendix I, and the results are summarized in Figure 5. The computational time and memory usage are in the order: Random-R, R3, RAD, LAS, and L ∞ . Gradient-based methods such as LAS and L ∞ generally incur higher costs; in particular, L ∞ is much slower because it requires many iterations (20 by default) with re-initialization each epoch, whereas LAS needs only l L = 1 without re-initialization. Thus, increasing the number of iterations should be considered with caution.
Detailed tuning results in Appendix J show that LAS with l L = 1 and without re-initialization achieves competitive performance, indicating that LAS can be applied in practice with computational cost comparable to Random-R, R3, and RAD.
Np de = 1 0 0 Np de = 1 , 0 0 0 Np de = 1 0 , 0 0 0 Np de = 5 0 , 0 0 0 Np de = 1 0 0 , 0 0 0 (a) 1D 10 0 10 1 10 2 10 3 Computation cost [s] Np de = 1 0 0 Np de = 1 , 0 0 0 Np de = 1 0 , 0 0 0 Np de = 5 0 , 0 0 0 Np de = 1 0 0 , 0 0 0 (b) 4D Np de = 1 0 0 Np de = 1 , 0 0 0 Np de = 1 0 , 0 0 0 Np de = 5 0 , 0 0 0 Np de = 1 0 0 , 0 0 0 (c) 6D 10 0 10 1 10 2 10 3 Computation cost [s]
Np de = 1 0 0 Np de = 1 , 0 0 0 Np de = 1 0 , 0 0 0 Np de = 5 0 , 0 0 0 Np de = 1 0 0 , 0 0 0 (d) 8D LAS RAD R3 L Random-R
this section cite: []

Section: Conclusion and Future Research Directions
This paper investigates the stability of training physics-informed neural networks (PINNs) under adaptive sampling strategies, particularly as model complexity increases. Theoretical analysis shows that methods focusing excessively on high residuals may undermine stability, especially in deeper networks or with larger learning rates. To address this, we introduce Langevin dynamics-based Adaptive Sampling (LAS), which updates collocation points using residual-weighted Langevin dynamics. Empirical results demonstrate that LAS ensures stable convergence even in high-dimensional PDEs, where conventional methods often fail.
Our Langevin-based sampling scheme builds on a theoretically grounded MCMC formulation that is both simple and effective. While the current approach shows strong performance, future work could explore more efficient variants for high-dimensional problems and develop principled strategies for hyperparameter selection.
this section cite: []

Section: References
Ref_id:b0 Title: Recipes for when physics fails: recovering robust learning of physics informed neural networks Year: (2023)
Ref_id:b1 Title: Learning data-driven discretizations for partial differential equations Year: (2019)
Ref_id:b2 Title: Heavy tails in sgd and compressibility of overparametrized neural networks Year: (2021)
Ref_id:b3 Title: Statistics of extremes: theory and applications Year: (2006)
Ref_id:b4 Title: Pinneik: Eikonal solution using physics-informed neural networks Year: (2021)
Ref_id:b5 Title: Convex optimization Year: (2004)
Ref_id:b6 Title: BC-PINN: an adaptive physics informed neural network based on biased multiobjective coevolutionary algorithm Year: (2021)
Ref_id:b7 Title: Physics-informed neural networks for heat transfer problems Year: (2021)
Ref_id:b8 Title: Diffusion for global optimization in R n Year: (1987)
Ref_id:b9 Title: Mitigating propagation failures in physics-informed neural networks using retain-resample-release Year: (2023)
Ref_id:b10 Title: Concentration of measure for the analysis of randomized algorithms Year: (2009)
Ref_id:b11 Title: Active learning based sampling for high-dimensional nonlinear partial differential equations Year: (2023)
Ref_id:b12 Title: Failure-informed adaptive sampling for pinns, part ii: combining with re-sampling and subset simulation Year: (2024)
Ref_id:b13 Title: Failure-informed adaptive sampling for pinns Year: (1971)
Ref_id:b14 Title: High-dimensional location estimation via norm concentration for subgamma vectors Year: (2023)
Ref_id:b15 Title: Extreme value theory: an introduction Year: (2006)
Ref_id:b16 Title: A physicsinformed deep learning framework for inversion and surrogate modeling in solid mechanics Year: (2021)
Ref_id:b17 Title: Physicsinformed neural networks for inverse problems in supersonic flows Year: (2022)
Ref_id:b18 Title: Finding flatter minima with sgd Year: (2018)
Ref_id:b19 Title: A gaussian mixture distribution-based adaptive sampling method for physics-informed neural networks Year: (2024)
Ref_id:b20 Title: Physics-informed machine learning Year: (2021)
Ref_id:b21 Title: On large-batch training for deep learning: Generalization gap and sharp minima Year: (2017)
Ref_id:b22 Title: PINNACLE: PINN adaptive collocation and experimental points selection Year: (2024)
Ref_id:b23 Title: Revisiting pinns: Generative adversarial physics-informed neural networks and point-weighting method Year: (2022)
Ref_id:b24 Title: Fourier neural operator for parametric partial differential equations Year: (2020)
Ref_id:b25 Title: A two-stage physics-informed neural network method based on conserved quantities and applications in localized wave solutions Year: (2022)
Ref_id:b26 Title: DeepXDE: A deep learning library for solving differential equations Year: (2021)
Ref_id:b27 Title: Enhancing pinns for solving pdes via adaptive collocation point movement and adaptive loss weighting Year: (2021)
Ref_id:b28 Title: Traditional and heavy tailed self regularization in neural network models Year: (2019)
Ref_id:b29 Title: Heavy-tailed universality predicts trends in test accuracies for very large pre-trained deep neural networks Year: (2020)
Ref_id:b30 Title: Efficient training of physics-informed neural networks via importance sampling Year: (2021)
Ref_id:b31 Title: Fast approximation of the sliced-wasserstein distance using concentration of random projections Year: (2021)
Ref_id:b32 Title: Physics-informed neural networks for modeling water flows in a river channel Year: (2022)
Ref_id:b33 Title: A physics-informed neural network for sound propagation in the atmospheric boundary layer Year: (2020)
Ref_id:b34 Title: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations Year: (2019)
Ref_id:b35 Title: Einns: epidemiologically-informed neural networks Year: (2023)
Ref_id:b36 Title: Physics-informed deep learning for traffic state estimation: A hybrid paradigm informed by second-order traffic models Year: (2021)
Ref_id:b37 Title: Fourier features let networks learn high frequency functions in low dimensional domains Year: (2020)
Ref_id:b38 Title: DAS-PINNs: A deep adaptive sampling method for solving high-dimensional partial differential equations Year: (2023)
Ref_id:b39 Title: Topics in random matrix theory Year: (2012)
Ref_id:b40 Title: DATS: Difficulty-aware task sampler for meta-learning physics-informed neural networks Year: (2024)
Ref_id:b41 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b42 Title: Is L 2 physics informed loss always suitable for training physics informed neural network? Year: (2022)
Ref_id:b43 Title: An expert's guide to training physics-informed neural networks Year: (2023)
Ref_id:b44 Title: Learning the solution operator of parametric partial differential equations with physics-informed deeponets Year: (2021)
Ref_id:b45 Title: When and why pinns fail to train: A neural tangent kernel perspective Year: (2022)
Ref_id:b46 Title: A comprehensive study of nonadaptive and residual-based adaptive sampling for physics-informed neural networks Year: (2023)
Ref_id:b47 Title: A diffusion theory for deep learning dynamics: Stochastic gradient descent exponentially favors flat minima Year: (2020)
Ref_id:b48 Title: Adaptive deep neural networks methods for high-dimensional partial differential equations Year: (2022)
Ref_id:b49 Title: Phaedon-Stelios Koutsourelakis, and Paris Perdikaris. Physicsconstrained deep learning for high-dimensional surrogate modeling and uncertainty quantification without labeled data Year: (2019)
Ref_id:b50 Title: 3 Different Hyperparameters of MCI Approaches on the Best ic/bc/pde Balance Terms DF-heat 4D Year: ()
Ref_id:b51 Title: RAD Hyper-parameters k=0.5, c=1 k=1, c=1 k=2, c=1 k=3, c=1 k=4, c=1 k=5 Year: ()
Ref_id:b52 Title: Hyper-parameters k=0.5, c=1 k=1, c=1 k=2, c=1 k=3, c=1 k=4, c=1 k=5 Year: ()
Ref_id:b53 Title: Hyper-parameters k=0.5, c=1 k=1, c=1 k=2, c=1 k=3, c=1 k=4, c=1 k=5 Year: ()
Ref_id:b54 Title: R3 Hyper-parameters Max_i=1 Max_i=3 Max_i=5 Max_i=10 Max_i=15 Max_i=20 Rel L2 error Year: ()
Ref_id:b55 Title:  Year: ()
Ref_id:b56 Title: Hyper-parameters Max_i=1 Max_i=3 Max_i=5 Max_i=10 Max_i=15 Max_i=20 Rel L2 error Year: ()
Ref_id:b57 Title:  Year: ()
Ref_id:b58 Title: Hyper-parameters Max_i=1 Max_i=3 Max_i=5 Max_i=10 Max_i=15 Max_i=20 Rel L2 error Year: ()
