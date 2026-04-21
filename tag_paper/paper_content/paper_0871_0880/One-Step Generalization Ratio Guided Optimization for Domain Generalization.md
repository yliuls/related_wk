Title: One-Step Generalization Ratio Guided Optimization for Domain Generalization
Abstract: Domain Generalization (DG) aims to train models that generalize to unseen target domains but often overfit to domain-specific features, known as undesired correlations. Gradient-based DG methods typically guide gradients in a dominant direction but often inadvertently reinforce spurious correlations. Recent work has employed dropout to regularize overconfident parameters, but has not explicitly adjusted gradient alignment or ensured balanced parameter updates. We propose GENIE (Generalization-ENhancing Iterative Equalizer), a novel optimizer that leverages the One-Step Generalization Ratio (OSGR) to quantify each parameter's contribution to loss reduction and assess gradient alignment. By dynamically equalizing OSGR via a preconditioning factor, GENIE prevents a small subset of parameters from dominating optimization, thereby promoting domaininvariant feature learning. Theoretically, GENIE balances convergence contribution and gradient alignment among parameters, achieving higher OSGR while retaining SGD's convergence rate. Empirically, it outperforms existing optimizers and enhances performance when integrated with various DG and single-DG methods.

Section: Introduction
Deep neural networks (DNNs) achieve high accuracy when training and test data share a similar distribution. However, in real-world applications, data distributions often shift, causing performance degradation (Muandet et al., 2013). Domain Generalization (DG) addresses this issue by training models to generalize to out-of-distribution data from unseen domains. The main challenge is to prevent overfitting Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). In contrast, GENIE uniformly adjusts parameter-wise OSGR, mitigating overfitting to specific parameters and promoting a more balanced optimization across the entire parameter space.
to domain-specific features-known as spurious correlations-while learning invariant features and causal relationships that generalize across diverse domains (Shi et al., 2022;Hemati et al., 2023;Shah et al., 2020a;Ye et al., 2024).
Several DG methods have attempted to guide the gradient toward a dominant direction during training (Parascandolo et al., 2021;Shahtalebi et al., 2021;Shi et al., 2022;Rame et al., 2022). However, this dominant direction often itself driven by spurious features, inadvertently reinforcing undesired correlations. This suggests that aligning gradients toward a single dominant direction is insufficient to fully solve the problem, highlighting the need for other perspectives.
A recent approach (Michalkiewicz et al., 2023) introduced a parameter-wise dropout mechanism based on Gradient Signal-to-Noise Ratios (GSNR) to suppress overly predictive parameters and reduce their influence on optimization. While this strategy mitigates parameter updates driven by spurious correlations, it does not adjust the magnitudes of updates based on their individual contributions to general-ization. This raises the open question of how to design optimizers that explicitly balance parameter updates according to their principled contributions to generalization, thereby mitigating the influence of spurious correlations.
Motivated by this perspective, we propose Generalization-ENhancing Iterative Equalizer (GENIE), a novel optimizer for addressing parameter imbalance. Recent work (Liu et al., 2020) introduced the One-Step Generalization Ratio (OSGR) that measures how effectively a single gradient update reduces test loss compared to training loss, providing insight into a model's generalization potential. OSGR reflects the contributions of individual parameters to generalization, based on their convergence speed and degree of gradient alignment. To leverage this insight, GENIE integrates a preconditioning factor that dynamically balances parameter-wise OSGR throughout training. This prevents a small subset of parameters from dominating the optimization, thereby promoting more robust and domain-invariant feature learning.
Our theoretical analysis shows that existing optimizers typically focus on either convergence speed or gradient alignment, often resulting in suboptimal generalization. In contrast, GENIE explicitly balances both, achieving a higher OSGR while maintaining the convergence rate of SGD (Robbins & Monro, 1951) in non-convex settings. We empirically validated GENIE on five standard DG datasets (Li et al., 2017;Fang et al., 2013;Venkateswara et al., 2017;Beery et al., 2018;Peng et al., 2019) where it consistently outperformed established optimizers, even with extended iterations. Furthermore, using our optimizer in existing DG and Single-DG (SDG) algorithms enhances their performance. We summarize our contributions as follows:
• We propose GENIE, a novel optimizer that addresses the overlooked issue of parameter imbalance in DG. It suppresses over-predictive parameters while promoting balanced parameter updates.
• We incorporate OSGR, previously used as a generalization metric, into the optimizer's core principle. This provides an efficient and novel perspective on generalization for addressing DG.
• GENIE is a domain-agnostic optimizer. It is validated across multiple DG benchmarks and SDG tasks, demonstrating its broad applicability and scalability.
this section cite: ['b33', 'b45', 'b14', 'b56', 'b36', 'b44', 'b45', 'b39', 'b31', 'b25', 'b40', 'b21', 'b8', 'b48', 'b3', 'b37']

Section: Related Work

this section cite: []

Section: Domain Generalization
Existing DG methods address domain shift through two main strategies: (1) Feature Alignment, which aims to align features across domains to ensure consistent optimization, including methods such as domain-invariant feature learning (Sun & Saenko, 2016;Arjovsky et al., 2019;Krueger et al., 2021), data augmentation (Xu et al., 2020;Yan et al., 2020;Wang et al., 2020), and feature disentanglement (Nam et al., 2021;Mahajan et al., 2021). (2) Gradient Alignment, which focuses on aligning gradients across domains to ensure stable learning dynamics. Representative approaches include minimizing gradient differences (Koyama & Yamaguchi, 2020), increasing gradient inner products (Shi et al., 2022), updating weights only when gradient directions align (Parascandolo et al., 2021;Shahtalebi et al., 2021), and reducing inter-domain gradient variance (Rame et al., 2022). Recently, Sharpness Aware Minima (SAM) (Foret et al., 2021) has improved in-distribution generalization, inspiring the development of optimizers specifically designed for OOD tasks (Zhang et al., 2024;Wang et al., 2023). However, most DG studies overlook imbalanced parameter updates caused by differences in convergence speed or generalization capacity during optimization.
this section cite: ['b46', 'b1', 'b20', 'b52', 'b53', 'b50', 'b35', 'b27', 'b19', 'b45', 'b36', 'b44', 'b39', 'b9', 'b59', 'b49']

Section: Preconditioning
Preconditioning improves the efficiency of optimization algorithms by incorporating curvature information of the loss function or adjusting the magnitude and direction of parameter updates. It accelerates convergence and enhances stability during training and can be categorized into three main types (Ye, 2024;Amari et al., 2021) (1) Hessian Based Preconditioning: utilizes the inverse or approximations of the Hessian matrix to capture curvature information. (Montavon et al., 2012;Dennis & Moré, 1977) (2) Adaptive Learning Rate Based Preconditioning: dynamically adjusts learning rates based on gradient magnitudes, as seen in optimizers like AdaGrad (Duchi et al., 2011), RMSProp (Hinton et al., 2012), and Adam (Kingma, 2014). (3) Normalization-Based Preconditioning: normalizes inputs and activations, as exemplified by Batch Normalization (Ioffe & Szegedy, 2015), to improve the Hessian's condition number and enhance training stability. Previous preconditioning methods aim to optimize speed and stability. The application of preconditioning to improve model generalization remains underexplored.
this section cite: ['b55', 'b0', 'b32', 'b6', 'b7', 'b15', 'b18', 'b17']

Section: Method

this section cite: []

Section: Preliminary
To address the challenge of generalization in unseen target domains, a recent study (Liu et al., 2020) introduced the concept of OSGR R(Z, n). OSGR quantifies how well model updates contribute to generalization by measuring the ratio of loss reduction between test D ′ and training data D after a single optimization step:
R(Z, n) = E D,D ′ ∼Z n ∆L D ′ E D∼Z n ∆L D ,(1)
where ∆L D ′ and ∆L D represent the loss changes on test and training data, respectively. OSGR is influenced by two key factors: (1) the contribution of each parameter to loss reduction, characterized by the gradient magnitude, and
(2) the alignment of parameter gradients across the data distribution. Higher OSGR indicates better generalization, reflecting consistent and balanced parameter updates.
To better understand these dynamics, the following theorem links OSGR to parameter-wise statistics:
Theorem 3.1 (From Paper (Liu et al., 2020)). The relationship between gradient updates and generalization can be expressed as follows:
R(Z, n) = 1 - 1 n j∈J E D∼Z n [g 2 j ] j ′ ∈J E D∼Z n [g 2 j ′ ] • 1 r j + 1 n , (2)
where J denotes the set of parameter index, g 2 j is the squared gradient magnitude, ρ 2 j is the noise variance, and n is the number of samples. Parameters with higher Gradient Signal-to-Noise Ratios (GSNR), defined as r j = g 2 j ρ 2 j , yield higher OSGR, contributing more significantly to generalization.
A recent study (Michalkiewicz et al., 2023) leveraged GSNR to suppress overly predictive parameters during training, aiming to prioritize robust features and reduce noisy updates. However, this approach overlooks parameter-wise imbalances in OSGR, which limits overall generalization performance.
In this context, we propose a preconditioning-based approach that dynamically balances OSGR across parameters. By incorporating parameter-specific preconditioning factors, our method ensures that updates are aligned with both gradient magnitude and noise characteristics, preventing overfitting to noisy or well-learned features. This strategy not only enhances generalization but also ensures stable convergence in diverse DG settings.
this section cite: ['b25', 'b25', 'b31']

Section: Proposed Method
Based on Theorem 3.1, Michalkiewicz et al. (2023) introduced a gradient-masking approach that prioritizes updates for parameters with low GSNR, aiming to enhance their contribution to generalization. They argue that boosting updates to low-GSNR parameters can increase the overall GSNR and thus improve the optimization signal-to-gradient ratio (OSGR). Inspired by this perspective, we hypothesize the following relationship:
this section cite: ['b31']

Section: Conjecture Uniformly distributed OSGR across parameters indicate better generalization performance.
This conjecture guides the design of our method. Rather than modifying the dropout ratio across parameters, we in-troduce a preconditioning term that more accurately adjusts the OSGR. Next, we inject noise into all parameters to encourage exploration toward better optima. Finally, we apply random dropout to stabilize parameter updates and reduce overfitting.
this section cite: []

Section: PRECONDITIONING
We propose a preconditioning factor p j to ensure balanced contributions of each parameter to the OSGR, thus enhancing generalization. The key idea is to maintain equitable parameter influence on the overall generalization performance throughout the optimization process. We propose the following corollary for this purpose.
this section cite: []

Section: Corollary 3.2 (Preconditioning and OSGR).
If each parameter j applies a preconditioner p j , the OSGR can be expressed as:
R ′ (Z, n) = j∈J p j E D∼Z n [g 2 j ] j ′ ∈J p j ′ E D∼Z n [g 2 j ′ ] • 1 1 n•rj + 1 ,(3)
or equivalently:
R ′ (Z, n) = 1 - 1 n jinJ p j E D∼Z n [g 2 j ] j ′ ∈J p j ′ E D∼Z n [g 2 j ′ ] • 1 r j + 1 n .
(4) From Corollary 3.2, to maintain a balanced influence of parameter j on the overall OSGR, we propose:
p j = 1 E D∼Z n g 2 j r j + 1 n . (5
)
This leads to the OSGR:
R ′ (Z, n) = 1- 1 n j∈J 1 j ′ ∈J r j ′ + 1 n = 1- 1 nE j∈J r j + 1 n , (6)
where E j∈J r j + 1 n represents the average GSNR contribution across parameters.Without preconditioning, parameters with large g 2 j but low GSNR may receive higher weights in the OSGR expression, inflating the subtraction term. Our preconditioning alleviates this issue and improves the OSGR. This dynamic adjustment with preconditioning mitigates parameter-wise imbalances, ensuring that wellgeneralized features are not overwhelmed by noisy or overly dominant parameters.
In implementation, we ignore the 1 n term as n is sufficiently large, and clipping variance by tanh( 1 σ 2 ) for stability. More detailed analysis on influence of variance is described in Section 3.3.3. This preconditioner p j is straightforward to compute and requires only the gradient statistics m t and variance σ t , which can be estimated during training. This efficiency makes it suitable for a wide range of DG tasks.
this section cite: []

Section: NOISE INJECTION
To enhance exploration during optimization, we introduce noise injection, where a noise term scaled by the variance is added to the gradient. Specifically, the noise scale is determined by 1 -tanh( 1 σ 2 ), reducing noise for high variance parameters while increasing it for low variance parameters. Motivated by (Mansilla et al., 2021), this injection boosts updates to parameters with low preconditioning value.
this section cite: ['b28']

Section: RANDOM MASK
To further stabilize updates and mitigate overfitting, we apply a random dropout mask. This mask, sampled from a Bernoulli distribution, selectively zeroes out gradient components. By applying random masking after the preconditioning step, all parameters are equally considered to ensure robust updates.
this section cite: []

Section: Analysis
We provide a comprehensive theoretical analysis of our method from three perspectives. First, we examine generalization through the OSGR, which highlights how our effectively balances OSGR value across parameters. Second, we formalize our approach under the PAC-Bayes framework, showing that our method explicitly minimizes a tighter generalization bound. Finally, we establish that our optimizer retains the convergence rate of standard SGD while enabling more robust generalization. Proofs are provided in Appendix C.
this section cite: []

Section: GENERALIZATION ANALYSIS WITH OSGR
We obtain the following corollary regarding the OSGR of these optimizers: Corollary 3.3 (OSGR of Optimizers). The OSGR of our proposed optimizer is:
R Ours = 1 - 1 nE j∈J r j + 1 n ,(7)
Comparing the resulting OSGR across different optimizers, we have:
R Ours ≥ R SGD ≈ R Adam .(8)
This corollary demonstrates that our proposed preconditioning achieves better generalization by attaining a higher overall OSGR. The following remarks provide further context and analysis: Remark 3.4 (Conceptual Components of Optimizers). The preconditioning applied by common optimizers can be viewed as the element-wise product of two conceptual components:
• Convergence Term: controls the effective step size, Algorithm 1 Algorithm for GENIE Input: Mini-batches {B t } T t=1 , Learning Rate α, Total Steps T . Hyperparameters: β ∈ [0, 1], Dropout Probability p Initialize: Parameters θ 0 , m 0 ← 0, v 0 ← 0. for t = 1 to T do Compute Gradient:
g t = ∇L(θ t ; B t )
Update Moving Averages:
m t ← βm t-1 + (1 -β)g t , v t ← βv t-1 + (1 -β)g 2 t
Calculate GSNR and Preconditioning:
σ 2 t = v t -m 2 t , r j = tanh( 1 σ 2 t )m 2 t ĝt ← m t 1 -β t • 1 v t • r t
Noise Injection:
N oise t ← ξ t 1 -tanh( 1 σ 2 t ) , ξ t ∼ N (0, σ 2 )
Random Mask:
M j ∼ Bernoulli(p) ĝt ← (ĝ t + N oise t ) ⊙ M
Update Parameters:
θ t+1 ← θ t -αg t end for Output: Final parameters θ T +1 .
thus contributing to faster convergence. It includes terms such as
E D∼Z n [g 2 j ] or E D∼Z n [g j ].
• Alignment Term: adjusts gradients toward stable directions. It includes the GSNR term r j .
Table 1 summarizes the convergence term, alignment term and their resulting OSGR, including SGD, Adam, and our method. Remark 3.5 (Optimizer-Specific Analysis). SGD maintains a baseline OSGR value with no explicit adjustment. Adam introduces a convergence component combined with a partial alignment factor. In contrast, our method effectively integrates both aspects in a balanced manner.
Overall, this analysis highlights how each optimizer's design affects generalization through gradient alignment and
SGD - - 1 - 1 n j∈J Wj • 1 rj + 1 n Wj = ED∼Zn [g 2 j ] j ′ ED∼Zn [g 2 j ′ ] ADAM 1 ED∼Zn (gj) 1 1 n•r j + 1 j∈J Wj • 1 1 n•r j + 1 Wj = ED∼Zn [g 2 j ] j ′ ED∼Zn [g 2 j ′ ] GENIE 1 ED∼Zn (g 2 j ) rj + 1 n 1 - 1 n j∈J Wj • 1 Ej∈J rj + 1 n Wj = 1 |J|
convergence speed.Incorporating both perspectives, Our method leads to a higher OSGR and thus improves generalization performance. Furthermore, we demonstrate that the alignment term in our preconditioning achieves a higher OSGR value than those of existing preconditioning methods. Detailed justifications are provided in the Appendix C.
this section cite: []

Section: GENERALIZATION ANALYSIS WITH PAC-BAYES BOUND
While the previous analysis is based on alignment and convergence dynamics using OSGR, we now adopt a complementary perspective grounded in the PAC-Bayes framework. We formulate the generalization analysis under a one-step update setting, where the KL divergence between successive parameter distributions reveals the connection between our preconditioning and a tighter generalization bound.
this section cite: []

Section: Theorem 3.6 (PAC-Bayes Interpretation of Preconditioning). R(θ) is the population risk and L(θ) is empirical risk.
Assume that the loss function L(θ) is bounded in [0, C]. For any λ > 0, with probability at least 1-δ over the draw of D, and for any data-dependent distribution p over parameters θ, the following PAC-Bayes bound holds:
E θ∼ p[R(θ)] ≤ E θ∼ p[L(θ)] T1 + λC 2 8n + KL(p∥π) + log 1 δ λ T2 .
Assume that p = N (θ t+1 , Σ p) and π = N (θ t , Σ π ), where
Σ p = diag(q j • ρ 2 j ) and Σ π = diag(ρ 2 j ). Let q j = E[g] 2 E[g 2 j ]
be a variance adaptation factor from SVAG optimizer (Balles & Hennig, 2018) that minimizes the variance to reduce max
T 1 term. (θ t+1 = θ t -q ⊙ g)
Then, minimizing the T 2 term via gradient descent yields an update direction:
∇ θt KL(p∥π) = 1 E[g 2 j ] • E[g j ] 2 ρ 2 j GENIE •g j,t ,
which matches the preconditioning rule of our optimizer.
Remark 3.7 (Sharpness and Generalization via KL). This result shows that our method not only improves sharpness-as done in SAM-but also directly enhances generalization by minimizing both terms in the PAC-Bayes bound. Specifically, the variance adaptation factor q j reduces the variability of scaled gradients, thereby tightening the empirical loss term T 1 through more stable updates. Simultaneously, the 1 ρ 2 term minimizes the KL divergence term T 2 . This result shows our the generalization property of GENIE comes from correlation with Pac-Bayes theory.
this section cite: ['b2']

Section: CONVERGENCE ANALYSIS
This section analyzes the convergence properties of GENIE under non-convex settings. Specifically, we adopt three widely used assumptions in the optimization literature:
Assumption 3.8. (Bounded Gradient) There exists a con- stant G > 0 such that ∥∇L(θ t )∥ ≤ G for all t.(9)
Assumption 3.9. (L-smooth) The loss function L is Lsmooth, meaning there exists a constant L > 0 such that for all θ 1 , θ 2 :
∥∇L(θ 1 ) -∇L(θ 2 )∥ ≤ L∥θ 1 -θ 2 ∥.(10)
Assumption 3.10. (Lower bounded variance) The variance of the stochastic gradients have lower bound by a constant 1/S u :
E[∥g t -∇L(θ t )∥ 2 ] ≥ 1/S u , ∀t.(11)
Under these assumptions, we establish the following result regarding the convergence rate:
Theorem 3.11. Under Assumption 3.8 Assumption 3.9, and Assumption 3.10 the average gradient norm over T iterations can be expressed as:
E[∥∇L(θ)∥ 2 ] ≤ O 1 P l 1 + G • S 2 u 2 1 T .(12)
where P l is lower bound of preconditioning value.
Remark 3.12 (Convergence Rate and Intuition). Theorem 3.11 shows that the average gradient norm converges at O(T -1/2 ), the standard rate for stochastic gradient methods in non-convex optimization. This implies that GENIE retains the fundamental convergence properties of SGD.
Remark 3.13 (Influence of G • S u and S u ). The term G • S 2 u
represents a trade-off associated with the GSNR. A higher GSNR upper bound( G • S u ) indicates a stronger gradient signal, which enhances generalization performance. However, it also acts as a multiplicative factor in the gradient norm, potentially slowing down convergence and thereby creating a trade-off. Furthermore, the variance term(S u ) has a significant impact on the bound, further influencing the overall convergence behavior. To address this issue, we regulate the variance term using the tanh function, which effectively balances the interplay between generalization and convergence dynamics.
this section cite: []

Section: Experiment
Dataset. We followed the standardized protocols of Do-mainBed (Gulrajani & Lopez-Paz, 2021), which include dataset splits, hyperparameter searches, and model selection using validation sets. Our approach was evaluated on five DG benchmark datasets: PACS (Li et al., 2017), VLCS (Fang et al., 2013), OfficeHome (Venkateswara et al., 2017), TerraIncognita (Beery et al., 2018), and DomainNet (Peng et al., 2019).
Evaluation. In accordance with DomainBed protocols, models were trained for 15,000 iterations on DomainNet and 5,000 iterations on the other datasets. For all DG and SDG experiments, we employed the Training-domain Validation Set approach, partitioning the source domain into training and validation subsets. The optimal model was selected based on validation performance. We followed previous DG methods by constructing 20 train-validation splits, with each split repeated 3 times.
Implementation Details. We used ResNet-50 (He et al., 2016b) pre-trained on ImageNet (He et al., 2016a) as backbone architectures. Detailed implementation details are presented in Appendix D. The detailed results and correspond-ing confidence intervals of all experiments are provided in Appendix E.
this section cite: ['b11', 'b21', 'b8', 'b48', 'b3', 'b37']

Section: Comparison of Optimizers on DG
Experiment Setup. We examined the impact of various optimization methods on generalization performance under domain shifts using Baseline ERM (Vapnik, 1999). The evaluated methods included: Standard optimizers (SGD (Robbins & Monro, 1951)), Adaptive optimizers (Adam (Kingma, 2014), AdamW (Loshchilov & Hutter, 2019), Ad-aBelief (Zhuang et al., 2020), AdaHessian (Yao et al., 2021), YOGI (Zaheer et al., 2018)), Sharpness-aware optimizers (SAM (Foret et al., 2021), GAM (Zhang et al., 2023b), FAD (Zhang et al., 2023a)) and our proposed GENIE.
Results. As shown in Table 2, our optimizer achieved superior performance across most datasets, surpassing existing methods. GENIE outperformed Adam, the default optimizer in most DG algorithms (Zhang et al., 2023a), by 5.69%. Additionally, it achieved improvements of 6.36% over SGD and 4.37% over SAM. In particular, it achieved remarkable performance on VLCS, which is prone to early convergence and overfitting (Matsuura & Harada, 2020), and on TerraIncognita, a wildlife image dataset with significant challenges such as lighting variations, motion blur, occlusions, and severe class imbalance (Beery et al., 2018). These results suggest that GENIE effectively prevents overfitting and enhances the learning of causal relationships by balancing parameter contributions during training. Optimizers designed for generalization, such as SAM, GAM and FAD, outperform standard optimizers, underscoring the significant role of optimization in generalization. These results emphasize the need for developing optimizers specifically tailored for DG. Experiment Setup. The computational overhead of an optimizer is a critical factor in its practical applicability. To evaluate this, we trained models on the PACS and VLCS datasets for 5,000, 10,000, and 15,000 iterations, measuring average performance and training time per iteration.
Results. As reported in Table 3, GENIE consistently outperformed other optimizers, even at 5,000 iterations, while incurring lower computational overhead than SGD and Adam. Additionally, GENIE achieved an average of 1.3× faster training compared to SAM, as SAM's update rule requires two sequential (non-parallelizable) gradient computations per step, which doubles the training time. These results experimentally validate the theoretical convergence analysis in Section 3.3.3, confirming GENIE's ability in computational efficiency and convergence speed. Results. The performance evaluation results for DG are summarized in Table 5. GENIE consistently outperforms existing optimization methods, demonstrating its robustness and broad applicability. These results validate GENIE's scalability and compatibility with various DG algorithms. Unlike other DG methods, which often require multiple source domains or architecture modifications, GENIE seamlessly integrates with existing training pipelines, providing consistent performance gains without additional complexity. This establishes GENIE as an algorithm-agnostic and highly adaptable optimization framework for DG tasks.
this section cite: ['b40', 'b18', 'b26', 'b63', 'b54', 'b57', 'b9', 'b29', 'b3']

Section: Single Domain Generalization
Experiment Setup. We evaluated performance in Single Domain Generalization (SDG), which is more constrained but better reflects real-world applications. The flexibility to operate in SDG without structural modifications is an advantage of our method over certain existing methods that are limited to multi-source settings. In SDG, the model is trained and validated on a single domain and tested on the others, with results averaged across all source domains. We compared GENIE with Adam, SGD, and SAM, and applied it to existing DG methods.
Results. The SDG performance results are presented in Table 4. As in previous DG settings, our optimizer outperformed existing optimizers. When applied to DG methods, conventional optimizers reduced performance, whereas GENIE achieved the highest performance as a standalone model and also improved DG methods when used as an optimizer. These results show that our method enhances DG performance without requiring architectural modifications or multiple source domains, and performs well even as a standalone method.  additional robustness.   OSGR of Network Parameters Over Time. To assess whether our approach enhances the overall OSGR of network parameters during training, we tracked the average OSGR of all parameters throughout the training process. As shown in Figure 4, the OSGR measurements on the VLCS dataset show that GENIE achieves an OSGR closer to 1 than prior optimizers. This means superior generalization performance. These findings align with the theoretical Generalization analysis in Section 3.3.1 , confirming that GENIE ensures more stable and balanced parameter updates during training, which ultimately leads to improved generalization.  Interestingly, while SAM is designed for better generalization performance, it exhibits inferior OSGR values. This suggests that the sharpness-aware regime alone is insufficient for generalization, and that the OSGR regime should also be considered when addressing generalization in DG tasks. This observation is consistent with our PAC-Bayesian analysis in Section 3.3.2, which reveals that inducing balanced OSGR values leads to tighter generalization bounds, reinforcing the role of OSGR as a necessary complement to sharpness-aware optimization.
Loss Landscape. We analyzed the convergence paths of SGD, Adam, and GENIE in the loss landscape using the FashionMNIST dataset (Xiao et al., 2017). As shown in Figure 5, each corner represents the local minima of a specific source domain. All optimizers started at (-1,3) and were updated for 30 steps under the same conditions. SGD and Adam follow steep direction and converge quickly. However, fast convergence often causes overfitting to specific source domains in OOD scenarios. Generalizable features are learned later in training (Pérez et al., 2019;Shah et al., 2020b;Nakkiran et al., 2019), so rapid convergence can prevent the model from acquiring them sufficiently. In contrast, as demonstrated in the theoretical analysis in Section 3.3.2, GENIE leads optimization toward flatter minima by effec- tively reducing sharpness, thereby improving generalization (Foret et al., 2021).
this section cite: ['b51', 'b38', 'b34', 'b9']

Section: Conclusion
We introduce GENIE, an optimizer that leverages OSGR to guide gradients in effective directions, preventing overly predictive parameters from dominating while ensuring all parameters contribute equitably to learning. GENIE achieves a higher OSGR with improved generalization and ensures fast convergence rate comparable to SGD. Empirically, it outperforms state-of-the-art optimizers across five DG benchmarks, demonstrating robust performance under significant domain shifts and limited data. Seamlessly integrating with existing DG and SDG methods, GENIE consistently achieves performance improvements. This work highlights the potential of OSGR as a guiding principle, paving the way for its use in few-shot learning, meta-learning, and other tasks requiring solutions to source-domain overfitting.
A. Notation training/test dataset drawn from Z θ, θj model parameters, parameter j θt,j parameter of index j at optimization step t gD,j(θ) gradient of parameter j averaged over training set D gt gradient at step t g 2 j squared gradient for parameter j ρ 2 j variance of parameter j's gradient σ 2 j variance of gradient averaged over training set
rj gradient signal-to-noise ratio (GSNR), rj = g 2 j ρ 2 j pj proposed preconditioning factor for parameter j R(Z, n) one-step generalization ratio (OSGR) ξt ∼ N (0, σ 2 ) Gaussian noise for noise injection J set of parameter index G bound of gradient l2 norm 1/Su lower bound of gradient variance L Lipschitz constant P l
lower bound of preconditioning value Wj weighting factor showing up in optimizers,
E D∼Z n (g 2 D,j ) j ′ E D∼Z n (g 2 D ′ ,j
) in SGD p, π probability measure of posterior and prior Σp, Σπ covariance matrix of Gaussian distribution ϵ, ϵ ′ random error terms in gradients KL(p∥π) KL divergence between distributions
this section cite: []

Section: References
Ref_id:b0 Title: When does preconditioning help or hurt generalization Year: (2021)
Ref_id:b1 Title: Invariant risk minimization Year: (2019)
Ref_id:b2 Title: Dissecting adam: The sign, magnitude and variance of stochastic gradients Year: (2018)
Ref_id:b3 Title: Recognition in terra incognita Year: (2018)
Ref_id:b4 Title: Domain generalization by marginal transfer learning Year: (2021)
Ref_id:b5 Title: Domain generalization by mutual-information regularization with pretrained models Year: (2022)
Ref_id:b6 Title: Quasi-newton methods, motivation and theory Year: (1977)
Ref_id:b7 Title: Adaptive subgradient methods for online learning and stochastic optimization Year: (2011)
Ref_id:b8 Title: Unbiased metric learning: On the utilization of multiple datasets and web images for softening bias Year: (2013)
Ref_id:b9 Title: Sharpness-aware minimization for efficiently improving generalization Year: (2021)
Ref_id:b10 Title: Domainadversarial training of neural networks Year: (2016)
Ref_id:b11 Title: In search of lost domain generalization Year: (2021)
Ref_id:b12 Title: Deep residual learning for image recognition Year: (2016-06)
Ref_id:b13 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b14 Title: Understanding hessian alignment for domain generalization Year: (2023)
Ref_id:b15 Title: Neural networks for machine learning lecture 6a overview of mini-batch gradient descent Year: (2012)
Ref_id:b16 Title: Selfchallenging improves cross-domain generalization Year: (2020)
Ref_id:b17 Title: Batch normalization: Accelerating deep network training by reducing internal covariate shift Year: (2015)
Ref_id:b18 Title: A method for stochastic optimization Year: (2014)
Ref_id:b19 Title: When is invariance useful in an out-of-distribution generalization problem Year: (2020)
Ref_id:b20 Title: Out-of-distribution generalization via risk extrapolation (rex) Year: (2021-07)
Ref_id:b21 Title: broader and artier domain generalization Year: (2017)
Ref_id:b22 Title: Learning to generalize: Meta-learning for domain generalization Year: (2018)
Ref_id:b23 Title: Domain generalization with adversarial feature learning Year: (2018)
Ref_id:b24 Title: Domain generalization via conditional invariant representations Year: (2018)
Ref_id:b25 Title: Understanding why neural networks generalize well through gsnr of parameters Year: (2020)
Ref_id:b26 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b27 Title: Domain generalization using causal matching Year: (2021)
Ref_id:b28 Title: Domain generalization via gradient surgery Year: (2021)
Ref_id:b29 Title: Domain generalization using a mixture of multiple latent domains Year: (2020)
Ref_id:b30 Title: Uniform manifold approximation and projection for dimension reduction Year: (2018)
Ref_id:b31 Title: Domain generalization guided by gradient signal to noise ratio of parameters Year: (2023)
Ref_id:b32 Title:  Year: (2012)
Ref_id:b33 Title: Domain generalization via invariant feature representation Year: (2013)
Ref_id:b34 Title: Sgd on neural networks learns functions of increasing complexity Year: (2019)
Ref_id:b35 Title: Reducing domain gap by reducing style bias Year: (2021-06)
Ref_id:b36 Title: Learning explanations that are hard to vary Year: (2021)
Ref_id:b37 Title: Moment matching for multi-source domain adaptation Year: (2019)
Ref_id:b38 Title: Deep learning generalizes because the parameter-function map is biased towards simple functions Year: (2019)
Ref_id:b39 Title: Fishr: Invariant gradient variances for out-of-distribution generalization Year: (2022-07)
Ref_id:b40 Title: A Stochastic Approximation Method Year: (1951)
Ref_id:b41 Title: Distributionally robust neural networks Year: (2020)
Ref_id:b42 Title: The pitfalls of simplicity bias in neural networks Year: (2020)
Ref_id:b43 Title: The pitfalls of simplicity bias in neural networks Year: (2020)
Ref_id:b44 Title: Sand-mask: An enhanced gradient masking strategy for the discovery of invariances in domain generalization Year: (2021)
Ref_id:b45 Title: Gradient matching for domain generalization Year: (2022)
Ref_id:b46 Title: Deep coral: Correlation alignment for deep domain adaptation Year: (2016)
Ref_id:b47 Title: An overview of statistical learning theory Year: (1999)
Ref_id:b48 Title: Deep hashing network for unsupervised domain adaptation Year: (2017)
Ref_id:b49 Title: Sharpnessaware gradient matching for domain generalization Year: (2023)
Ref_id:b50 Title: Heterogeneous domain generalization via domain mixup Year: (2020)
Ref_id:b51 Title: Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms Year: (2017)
Ref_id:b52 Title: Adversarial domain adaptation with domain mixup Year: (2020)
Ref_id:b53 Title: Improve unsupervised domain adaptation with mixup training Year: (2020)
Ref_id:b54 Title: Adahessian: An adaptive second order optimizer for machine learning Year: (2021)
Ref_id:b55 Title: Preconditioning for accelerated gradient descent optimization and regularization Year: (2024)
Ref_id:b56 Title: Spurious correlations in machine learning: A survey Year: (2024)
Ref_id:b57 Title: Adaptive methods for nonconvex optimization Year: (2018)
Ref_id:b58 Title: Adaptive risk minimization: learning to adapt to domain shift Year: (2021)
Ref_id:b59 Title: Domaininspired sharpness-aware minimization under domain shifts Year: (2024)
Ref_id:b60 Title: Flatness-aware minimization for domain generalization Year: (2023)
Ref_id:b61 Title: Gradient norm aware minimization seeks first-order flatness and improves generalization Year: (2023)
Ref_id:b62 Title: Domain generalization with mixstyle Year: (2021)
Ref_id:b63 Title: Adabelief optimizer: Adapting stepsizes by the belief in observed gradients Year: (2020)
Ref_id:b64 Title:  Year: (1999)
Ref_id:b65 Title:  Year: (2020)
Ref_id:b66 Title:  Year: (2016)
Ref_id:b67 Title:  Year: (2021)
