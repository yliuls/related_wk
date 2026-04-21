Title: Generating Informative Samples for Risk-Averse Fine-Tuning of Downstream Tasks
Abstract: Risk-averse modeling is critical in safety-sensitive and high-stakes applications. Conditional Value-at-Risk (CVaR) quantifies such risk by measuring the expected loss in the tail of the loss distribution, and minimizing it provides a principled framework for training robust models. However, direct CVaR minimization remains challenging due to the difficulty of accurately estimating rare, high-loss events-particularly at extreme quantiles. In this work, we propose a novel training framework that synthesizes informative samples for CVaR optimization using score-based generative models. Specifically, we guide a diffusion-based generative model to sample from a reweighted distribution that emphasizes inputs likely to incur high loss under a pretrained reference model. These samples are then incorporated via a loss-weighted importance sampling scheme to reduce noise in stochastic optimization. We establish convergence guarantees and show that the synthesized, high-loss-emphasized dataset substantially contributes to the noise reduction. Empirically, we validate the effectiveness of our approach across multiple settings, including a real-world wireless channel compression task, where our method achieves significant improvements over standard risk minimization strategies.

Section: Introduction
Risk-averse learning has become increasingly relevant in high-stakes applications where robustness to rare but costly failures is critical. In those domains, models must not only achieve strong average-case performance but also avoid catastrophic errors on atypical inputs. A widely adopted risk measure for capturing such sensitivity is the Conditional Value-at-Risk (CVaR), which focuses on the expected loss in the worst-performing (1 → ω) fraction of the input space (Rockafellar and Uryasev, 2000), making it well-suited for applications requiring robustness guarantees, e.g., large language models, system scheduling, control, medical, wireless communications, and more (Chaudhary et al., 2024;Tan et al., 2017;Ahmadi et al., 2022;Chan et al., 2014;Yang et al., 2022).
Despite its appeal, minimizing CVaR remains challenging in practice. As the quantile level ω approaches one, loss contributions become dominated by rare, high-risk inputs that are unlikely to be observed through standard sampling from the data distribution. Without adequate coverage of these tail events, naive Monte Carlo (MC) methods yield high-variance estimates of CVaR and inefficient optimization, ultimately limiting the reliability of risk-averse training. communications, large language models, and more (Alexander et al., 2006;Andersson et al., 2001;Filippi et al., 2020;Yang et al., 2022;Chaudhary et al., 2024;Chow and Ghavamzadeh, 2014).
However, training with CVaR objective is notoriously challenging due to the high variance in empirical estimates, particularly when targeting extreme quantiles (Troop et al., 2021).
Importance Sampling for Risk Measures and Optimization. Importance sampling can be utilized for improving the variance of risk estimation, particularly when rare events dominate the objective. Prior works have explored its use in CVaR estimation and optimization (Bardou et al., 2009;Deo and Murthy, 2021;He et al., 2024a), including sampling-based gradient estimators based on likelihood ratios (Tamar et al., 2015). However, these approaches typically operate on a fixed dataset and focus on reweighting existing samples to reduce estimation variance.
Our method introduces a fundamentally new perspective: we utilize a generative model to directly synthesize importance-weighted samples. Under the availability of a generative model, we guide the sample generation process toward high-loss regions using pretrained model losses. This enables us to reduce the noise of CVaR optimization while expanding the effective support of the training distribution.
Generative Models for Data Augmentation and Downstream Task Learning. Recent progress in generative modeling, particularly in diffusion and score-based generative models, has enabled highfidelity sample generation and accurate distribution approximation (Chen et al., 2024;Wang et al., 2024b). These models have been successfully applied across diverse domains for data augmentation, including load forecasting (Xu and Zhu, 2024), medical imaging (He et al., 2024b), and audio synthesis (Bahmei et al., 2022). Recently, generative models are increasingly used to augment training datasets with specific purposes (Zheng et al., 2023), e.g., enhancing semantic diversity (Shivashankar and Miller, 2023;Trabucco et al., 2023), generating label-specific instances (Shao et al., 2019), and bridging distributional gaps between training and test data (Wang et al., 2024a).
this section cite: ['b43', 'b10', 'b49', 'b0', 'b8', 'b59', 'b1', 'b3', 'b21', 'b59', 'b10', 'b13', 'b51', 'b6', 'b19', 'b48', 'b11', 'b58', 'b5', 'b64', 'b45', 'b50', 'b44']

Section: Scope and Distinctiveness of the Proposed Approach.
While prior work has primarily leveraged generative models to enhance generalization by enriching the diversity of training data, our approach adopts a different objective: synthesizing samples that are explicitly informative for risk-sensitive training. Rather than uniformly augmenting the training distribution, we concentrate generation toward high-loss regions-those most relevant for CVaR minimization. This targeted generation paradigm aligns directly with risk-averse learning objectives, offering a principled and efficient path toward robust model training.
this section cite: []

Section: Preliminaries and Problem Formulation
Recent advances in generative modeling, particularly score-based generative models, have substantially improved the quality and controllability of synthetic data generation. A key strength of the score-based generative models lies in their ability to support guided sampling, where samples can be drawn from a distribution that is shifted or reweighted relative to a base distribution. In this work, we leverage this capability to generate rare, high-loss-inducing samples that are underrepresented in standard datasets but critical for risk-sensitive objectives. As we will show, synthesizing such samples provides both theoretical and practical advantages in CVaR minimization.
this section cite: []

Section: Score-based Generative Models and Training-Free Guided Sampling
Let X p ↑ R d1 be a random variable with density p(x), where x denotes a realization. A generative model seeks to approximate p(x) or its associated score function ↓ x log p(x) to enable efficient sampling from the underlying distribution. A key innovation of the recent score-based generative models is to model the score function not directly on p(x), but on noise-perturbed data distributions p t (x) indexed by a continuous-time parameter t ↑ [0, T ]. This enables the data generation process to be formulated as a stochastic differential equation (SDE), which has been shown to enhance both training stability and sample quality (Song et al., 2021). The perturbed distributions are modeled via the Itô SDE:
dX p (t) = f(X p (t) , t) dt + ε(t) dW (t) , X p (0) ↔ p(x),(1)
where f : R d1 ↗ [0, T ] ↘ R d1 is the drift term, ε(t) : [0, T ] ↘ R is the diffusion coefficient, and W (t) is a standard d 1 -dimensional Brownian motion. The marginal distribution of X p (t) is denoted p t (x), and the initial distribution p 0 (x) corresponds to the data distribution we aim to model. Given access to the time-indexed score ↓ x log p t (x), samples from p = p 0 can be obtained by solving the reverse-time SDE (Anderson, 1982):
dX p (t) = f(X p (t) , t) → ε(t) 2 ↓ x log p t (X p (t) ) dt + ε(t) d W(t)
, where W(t) denotes reverse-time Brownian motion. Sampling is typically initialized from a simple prior such as a standard Gaussian p T (x) for large T , and trajectories are integrated backward to recover X p (0) ↔ p 0 . For notational simplicity, scalar-vector multiplication denotes elementwise scaling.
Beyond sampling from the base distribution p(x), recent advancements in the score-based generative models allow sample generation from modified target distributions of the form q(x) ≃ w(x)p(x), where w(x) is a task-specific importance weight. While classical approaches such as the crossentropy method (CEM) and fine-tuning of generative models require retraining to realize such reweighted distributions, training-free guidance techniques for the score-based generative models enable approximate sampling from q without modifying the base generative model. These methods exploit the fact that, under the same SDE dynamics in (1), if a process begins from q(
x) = q 0 (x) ≃ w(x)p(x) instead of p as dX q (t) = f(X q (t) , t) dt + ε(t) dW (t) with X q (0) ↔ q = q 0 , then its marginal q t such that X q (t) ↔ q t satisfies q t (x) ≃ p t (x) E X p (0) ↑p(•|X p (t) =x) w(X p (0) ) , where p(• | X p (t) =
x) denotes the conditional distribution of the initial state given state x. Taking the log and the gradient yields the identity: ) ) . This additional term g, often referred to as the guidance, can be approximated using known quantities such as the score function of p t and the weight function w (Chung et al., 2023;Kim et al., 2025c;Yu et al., 2023), enabling sampling from approximated q via the reverse-time SDE without any further training.
↓ x log q t (x) = ↓ x log p t (x) + g(x, t), where g(x, t) := ↓ x log E X p (0) ↑p(•|X p (t) =x) w(X p (0
Motivated by these developments, we propose a new learning framework that leverages guided sample generation to construct informative training data specifically tailored for risk-averse learning. Rather than drawing training samples uniformly from p(x), we aim to generate samples that contribute to noise reduction in risk-sensitive objectives, thereby improving model robustness to rare but high-loss events.
this section cite: ['b47', 'b2', 'b14', 'b60']

Section: Risk-Averse Learning via Conditional Value-at-Risk
Our ultimate goal is training of risk-averse task models, in which the objective is not merely to optimize expected model performance, but to mitigate the impact of potentially rare but high-loss outcomes. We consider CVaR, one of the most widely adopted risk measures, which builds upon the concept of Value-at-Risk (VaR).
Let ϑ ↑ R d2 denote the parameters of the task model, and let ϖ(ϑ; x) be the loss incurred on input x. For a given quantile (confidence) level ω ↑ (0, 1), the VaR is defined as the smallest threshold ϱ such that the loss does not exceed ϱ with probability at least ω:
VaR ω (ϑ) = min{ϱ ↑ R : P[ϖ(ϑ; X p ) ⇐ ϱ] ⇒ ω}(2)
where P[ϖ(ϑ; X p ) ⇐ ϱ] = ε(ϑ;x)↓ϖ p(x) dx, and the distribution is assumed to be continuous with respect to ϱ. While VaR captures a quantile of the loss distribution, it does not reflect the magnitude of losses beyond the threshold. The Conditional Value-at-Risk addresses this by computing the expected loss in the tail beyond VaR ω (ϑ):
CVaR ω (ϑ) = E X p ↑p [ϖ(ϑ; X p ) | ϖ(ϑ; X p ) ⇒ VaR ω (ϑ)].(3)
Our objective is to find model parameters ϑ → that minimize the CVaR at a given quantile level ω:
ϑ → = argmin ϑ↔R d 2 CVaR ω (ϑ).(4)
this section cite: []

Section: Algorithm 1 Risk-Averse Model Training via Loss-Guided Importance Sample Generation
Input: Initial model ϑ 0 , generative model ↓ log p t (x), confidence level ω, function ς, dataset B Output: Risk-averse model ϑ K+1 1: Generate importance samples {x i } Bq i=1 ↔ q(x) ≃ ς(ϖ(ϑ 0 ; x)) p(x) 2: Compute Z = E X p ↑p [ς(ϖ(ϑ 0 ; X p ))] 3: Initialize ϱ 0 4: for k = 1 to K + 1 do 5: Sample data pair {x, ϖ(ϑ 0 ; x)} from q 6: MC estimation of ϱ k↗1 + E X q ↑q Z(ε(ϑ k→1 ;X q )↗ϖ k→1 ) + ϱ(ε(ϑ0;X q ))(1↗ω) 7: (ϑ k , ϱ k ) ↘ ⇑ SubGradientDescent(φF ω , ϑ k↗1 , ϱ k↗1 )
This objective is known to be equivalent to the following unconstrained optimization problem (Rockafellar and Uryasev, 2000) as
ϑ → , ϱ → = argmin ϑ↔R d 2 ,ϖ↔R F ω (ϑ, ϱ) where F ω (ϑ, ϱ) = ϱ + 1 1 → ω E X p ↑p [(ϖ(ϑ; X p ) → ϱ) + ]. (5)
Here, (x) + = max(x, 0) denotes the positive part function. The solution ϱ → corresponds to VaR ω , and ϑ → minimizes CVaR ω .
this section cite: ['b43']

Section: Risk-Averse Model Training via Loss-Guided Importance Samples
While the CVaR objective in (5) provides a principled framework for addressing tail-risk scenarios in downstream tasks, its empirical estimation is particularly challenging, especially at high confidence levels (ω ↘ 1). In this regime, the corresponding VaR threshold ϱ becomes large, and the expectation term E X p ↑p [(ϖ(ϑ; X p ) → ϱ) + ] becomes increasingly difficult to estimate due to the rarity of high-loss instances, for which (ϖ(ϑ; X p ) → ϱ) + is nonzero. For most samples from p(x), this term evaluates to zero, leading to high variance and poor gradient signals during training. Consequently, naive sampling from the base distribution p(x), even with a generative model, becomes inefficient and often requires an infeasibly large number of samples to stably estimate the CVaR objective for training.
this section cite: []

Section: Key Idea.
To address these, we propose leveraging the generative model to sample from an importance-weighted distribution q(x) tailored to highlight high-loss regions. Based on the availability of a pretrained model ϑ 0 and a score-based generative model capable of sampling from p(x), our approach consists of two components: (i) Sample inputs from a weighted distribution q(x) ≃ ς(ϖ(ϑ 0 ; x)) p(x), where ς : R ≃0 ↘ R ≃0 is a nondecreasing function that prioritizes high-loss examples. (ii) Use the importance samples to perform CVaR minimization via importanceweighted MC estimation.
Intuitively, this sample generation strategy concentrates on examples that are informative for CVaR optimization, those in the tail of the loss distribution. These rare, high-loss instances expose the model to critical failure modes and enable more effective risk-averse training. We refer to our approach as Risk-Averse Model training via loss-guided Importance Samples (RAMIS).
this section cite: []

Section: Algorithm
Algorithm 1 takes as input: an initial pretrained model ϑ 0 , a score-based generative model capable of sampling from p(x), a target quantile level ω ↑ (0, 1), and a non-decreasing weighting function ς used to construct the importance sampling distribution. We assume access to a dataset B = {x i } B i=1 , where each x i is drawn i.i.d. from the base distribution p(x). This dataset may be externally provided or synthesized via the generative model.
Line 1: We generate B q samples from the importance-weighted distribution, q 0 (x) = q(x) ≃ ς(ϖ(ϑ 0 ; x)) p(x), by guiding the generative model using loss values computed under ϑ 0 . In scorebased generative models, this corresponds to solving the following reverse-time SDE:
dX q (t) = f(X q (t) , t) → ε(t) 2 ↓ x log q t (X q (t) ) dt + ε(t) d W(t) .(6)
The specific implementation of this guided importance sampling process may vary based on the chosen generative model guidance method and is detailed in Appendix B.
Line 2: The expectation of the importance weight function ς(ϖ(ϑ 0 ; x)) over the base distribution p(x) is computed as E X p ↑p [ς(ϖ(ϑ 0 ; X p ))] = Z. This normalization factor Z can be approximated as Z ⇓ 1 |B| x↔B ς(ϖ(ϑ 0 ; x)) and is utilized in subsequent optimization iterations. Lines 4-7: At each training step, we draw a sample from q(x) along with its corresponding im-
portance weight ς(ϖ(ϑ 0 ; x)). The CVaR objective ϱ k↗1 + E X q ↑q Z(ε(ϑ k→1 ;X q )↗ϖ k→1 ) + ϱ(ε(ϑ0;X q ))(1↗ω)
is estimated via MC, which corresponds to a variational form in (5). Note that the likelihood ratio p(x)/q(x) simplifies to Z/ς(ϖ(ϑ 0 ; x)). We perform subgradient-based optimization using a subroutine SubGradientDescent (see Appendix A for details), updating both ϑ and ϱ in the direction that minimizes the estimated CVaR objective.
Importance Sampling Mechanism. The proposed approach differs fundamentally from conventional importance sampling techniques, which re-evaluate model performance at each iteration and dynamically adjust sampling probabilities over a fixed dataset. Such methods introduce additional per-iteration computational overhead (El Hanchi and Stephens, 2020;Needell et al., 2014;Zhao and Zhang, 2015).
In contrast, our framework adopts a fixed importance sampling distribution constructed prior to training. We guide the generative model to directly produce samples from the target importanceweighted distribution. This eliminates the need for iterative reweighting or per-batch loss evaluations. Importantly, during training (Lines 4-7 in Algorithm 1), our method introduces no additional computational overhead beyond a lightweight scalar reweighting of the loss term (ϖ(ϑ; x) → ϱ) + .
this section cite: ['b20', 'b41', 'b63']

Section: Theoretical Analysis
In this subsection, we present a convergence analysis of the proposed RAMIS framework and justify how loss-guided importance sampling improves risk-averse training. Specifically, we show that sampling from a reweighted distribution, which is biased toward high-loss regions under a reference model, reduces the noise of stochastic gradient descent. Assumption 1 (Convexity, smoothness, and bounded loss). For all x ↑ R d1 , ϖ(ϑ; x) are convex, continuously differentiable, 0 < ϖ(ϑ; x) < M < ⇔, and ϖ(ϑ; x) and the norm of ↓ϖ(ϑ; x) are L 1 -smooth and L 2 -Lipschitz, respectively. For all k ↑ [0, K], ↖ϑ k ↖ ⇐ ↼.
Assumption 1 implies the standard convexity and smoothness of the loss function. We provide a formal definition of convexity and smoothness in Appendix A. Also, the parameterized model norm is bounded. Building on the CVaR minimization analysis of Meng and Gower (2023), which relies on the stochastic model-based framework of Davis and Drusvyatskiy (2019), we have the following convergence property. Theorem 1 (Convergence Rate). Suppose that Assumption 1 holds and over iterations k = 1, . . . , K+ 1, Algorithm 1 uses realizations x k that are i.i.d. with q. Let {↽ k } be the iterates generated by
Algorithm 1 such that ↽ k = (ϑ k , ϱ k ) ↘
, ↽ → is a minimizer of F ω , and set ⇀ k = ς ⇐ K+1 . For a given quantile ω, we have
E F ω 1 K +1 K+1 k=1 ↽ k → F ω (↽ → ) ⇐ ↖(ϑ 0 , ϱ 0 ) ↘ → ↽ → ↖ 2 2⇀ ↙ K + 1 + ⇀v(q) ↙ K + 1 ,(7)
where
v(q) = E X p ↑p w ↑ (X p ) 2 (1↗ω) 2 p(X p ) q(X p ) + 1 and w → (x) = ( 2L 1 ϖ(ϑ 0 ; x) + 2L 2 ↼)) 2 + 1 1/2 .
Remark 1 (Loss-dependent Optimization Noise). Theorem 1 establishes an O(1/ ↙ K) convergence rate with a stochastic noise term v(q) that depends on the initial loss ϖ(ϑ 0 ; x). This dependence on the loss value is well-aligned with the standard results in stochastic optimization (Zhao and Zhang, 2015;Davis and Drusvyatskiy, 2019), where the stochastic noise is governed by the gradient of the loss. To understand this relationship more precisely, consider the case where the loss function satisfies a Polyak-!ojasiewicz (PL) condition (Karimi et al., 2016). That is, for some µ > 0 and all ϑ, 2µ (ϖ(ϑ 0 ; x) → ϖ → ) ⇐ ↖↓ ϑ ϖ(ϑ 0 ; x)↖ 2 where ϖ → = min ϑ ϖ(ϑ; x). Note that the PL condition holds for several classes of neural networks (Liu et al., 2019;Zhou and Liang, 2017;Charles and Papailiopoulos, 2018;Hardt and Ma, 2016). Under L 1 -smoothness, ↖↓ ϑ ϖ(ϑ 0 ; x)↖ 2 ⇐ 2L 1 ϖ(ϑ 0 ; x) and we have
2µ (ϖ(ϑ 0 ; x) → ϖ → ) ⇐ ↖↓ ϑ ϖ(ϑ 0 ; x)↖ 2 ⇐ 2L 1 ϖ(ϑ 0 ; x).(8)
This chain of inequalities implies that the high-loss samples contribute proportionately to the norm of the gradient. Remark 2 (Noise Reduction via Importance Sampling). The term v(q) in Theorem 1 is minimized when the sampling distribution q(x) is chosen as q(x) ≃ w → (x)p(x), which depends on the quantities, potentially impractical to compute in real-world scenarios. To circumvent this, we propose using a non-decreasing surrogate weighting function ς : R ≃0 ↘ R ≃0 that approximates the behavior of desired importance weights. Specifically, sampling from the distribution q(x) ≃ ς(ϖ(ϑ 0 ; x)) p(x) reduces the term v(q) relative to naive sampling from p(x) under the following condition:
v(p) ⇒ v(q) ∝′ E[w → (X p ) 2 ] ⇒ E w → (X p ) 2 ς(ϖ(ϑ 0 ; X p )) • E[ς(ϖ(ϑ 0 ; X p ))].(9)
We observe empirically that simple choices of ς, such as the square-root or identity mapping, yield strong performance (Sec. 5.1). In summary, the analysis establishes that loss-guided importance sampling based on the pretrained model can reduce the error of CVaR optimization. Rather than merely increasing the sample size, we leverage score-based generative models to synthesize samples from the proposed reweighted distribution, enabling efficient risk-averse training.
5 Experiments Evaluation Summary. We evaluate the effectiveness of our proposed framework across both synthetic and real-world tasks. Specifically, we aim to answer the following questions: (i) Can we generate high-loss-inducing samples using score-based generative models by pretrained models? (ii) Do these samples improve downstream robustness relative to existing robust optimization methods? (iii) Does the approach generalize to high-stakes, real-world applications?
To this end, we conduct two sets of experiments: Sec. 5.1: We evaluate our method on a controlled regression task over a Gaussian mixture distribution to assess robustness under data heterogeneity and sample scarcity. Sec. 5.2: We apply our method to a real-world wireless channel state information (CSI) compression task, demonstrating its potential practical utility.
Baselines and Fairness. We compare against strong risk-sensitive and robust optimization baselines: Stochastic Subgradient Method (SSGM) for CVaR minimization (Meng and Gower, 2023), DORO (Zhai et al., 2021), ⇁ 2 -DRO (Namkoong and Duchi, 2016), and standard ERM (i.e., CVaR at ω = 0). All methods start from the same pretrained checkpoint and are trained on the same number of samples from the same generative model; RAMIS uses the identical budget but replaces standard samples with loss-guided (importance) samples. Running SSGM without importance sampling isolates the contribution of our loss-guided sampling scheme, while comparisons to DORO and ⇁ 2 -DRO test whether state-of-the-art robust objectives can mitigate tail risk absent our mechanism. ERM serves as a conventional average-risk baseline.
this section cite: ['b18', 'b63', 'b18', 'b29', 'b35', 'b65', 'b9', 'b23', 'b62', 'b40']

Section: Risk-Averse Regression over Density-Heterogeneous Gaussians
Task Overview. We consider a synthetic regression task where inputs x ↑ R 2 are drawn from a Gaussian mixture distribution with three components centered at (→0.6, 0.6), (0.6, 0.6), and (0.0, →0.6), each with standard deviation 0.06 but with unbalanced mixing weights of 0.01, 0.001, and 0.989, respectively (See leftmost subplot in Figure 2). The objective is to predict the second coordinate x [1] from the first x [0] using a quadratic regression model trained via risk minimization methods with Mean Squared Error (MSE) loss. This setup can pose a significant challenge for robust learning due to the extreme imbalance: standard training on limited number of samples from p(x) yields poor performance on rare but critical components (e.g., the 0.01 or 0.001-weight modes), which dominate the tail risk.
To evaluate our method, we adopt a two-phase training strategy. First, we obtain a pretrained (reference) model ϑ 0 on 10 2 samples from p(x). Second, we use ς(x) = x 1/2 to construct an  Results. Figure 2 visually illustrates our method. The leftmost panel shows the ground-truth distribution p(x), which includes two rare Gaussian components with low probabilities (0.01 and 0.001). As depicted in the second panel, a limited number of samples drawn from p(x) rarely cover these low-density regions, resulting in limited exposure during training. Consequently, the pretrained model trained on these samples shows high loss in the low-density regions, as reflected in the loss map shown in the third panel-in the colormap, these tail regions appear in yellowish hues, indicating higher loss.
We exploit this loss landscape by constructing an importance-weighted distribution based on ϖ(ϑ 0 ; x) and guiding the generative model to sample accordingly. The fourth panel shows samples generated from this reweighted distribution. Despite using the same sample budget, these samples provide substantially better coverage of the support set, especially in the tails. The final panel shows that training on these importance samples leads to a risk-averse model that performs reliably across both high-density and tail regions of the input space.
Table 1 reports the CVaR performance across varying quantile levels ω for our method and baseline approaches. Across all quantile levels, our framework (RAMIS) consistently achieves the lowest CVaR, demonstrating superior robustness in tail-risk regimes. The baseline methods without importance samples exhibit substantially higher risk. These results demonstrate that access to high-loss-inducing importance samples generated via pretrained model guidance provides a distinct advantage that cannot be matched by applying robust optimization techniques over uniformly sampled data.
this section cite: []

Section: Additional Analysis
To further assess the efficiency of the proposed framework, we analyze (i) the computational cost of generating importance-weighted samples and (ii) the effect of the weighting function ς, which controls the emphasis placed on high-loss regions. Detailed results and ablation studies are provided in Appendix E.
this section cite: []

Section: Risk-Averse Compression of Wireless Channel State Information
Task Overview. In wireless communication systems, Channel State Information (CSI) captures key physical-layer characteristics such as signal directionality, multipath components, and propagation strength between transmitters and receivers (Lin, 2022). Accurate CSI feedback from the transmitter to the receiver is crucial for tasks like beamforming, scheduling, and adaptive modulation. However, modern CSI matrices are typically high-dimensional, necessitating efficient compression to support  bandwidth-constrained channel feedback (Guo et al., 2022). To ensure reliable communication in practical deployments, especially under worst-case scenarios, risk-averse compression is essential.
In this experiment, we assess the performance of the proposed method in the context of risk-averse CSI compression. We assume access to a pretrained score-based generative model trained on a CSI dataset generated by the Quadriga simulator (Jaeckel et al., 2021), where a single CSI instance is a 256 ↗ 32 complex matrix, and a baseline CSI compressor trained using ERM. The CSI compressor is implemented as a vector-quantized autoencoder (van den Oord et al., 2017), comprising an encoder, a quantization bottleneck, and a decoder. Following Algorithm 1, we guide the pretrained generative model using the MSE loss values of the initial compressor to generate informative, high-loss samples. These samples are then used to fine-tune the compressor using a CVaR-based objective. Detailed specifications of the dataset, model architecture, and training parameters are provided in Appendix D.
this section cite: ['b34', 'b22', 'b28', 'b52']

Section: Results.
Table 2 reports the CVaR performance in terms of reconstruction distortion (MSE) across various quantile levels ω. Lower distortion indicates better robustness. RAMIS consistently achieves the lowest CVaR in the high-risk regime (ω ↑ {0.9, 0.95, 0.99}), outperforming all baselines, including SSGM, DORO, ⇁ 2 -DRO, and ERM. As ω decreases toward 0, where CVaR approaches the expected loss, the performance gap narrows, and RAMIS converges with SSGM and ERM. This further indicates that RAMIS with importance samples does not degrade the average-case performance, confirming its consistency. The performance gain at the high ω region supports that the conventional methods, which rely on samples drawn from the original data distribution, are insufficient for minimizing tail risk.
Figure 3 further illustrates the nature of the samples generated via pretrained-loss-guided importance sampling. The top row shows representations in the spatial-frequency domain, while the bottom row visualizes the corresponding angular-delay profiles, computed via 2D inverse FFT (IFFT) with truncation to the low-delay region for interpretability. The left three columns present typical generated samples from the base generative model, chosen as the three median distortion examples by MSE.
By contrast, the right three columns show samples obtained via RAMIS, using the generative model guided by the pretrained model loss. These samples exhibit significantly higher reconstruction distortion, with the average MSE increasing by 4.7 ↗ 10 ↗4 and a maximum distortion exceeding 6 ↗ 10 ↗3 . Notably, the corresponding angular-delay representations reveal more complex scattering patterns, indicating that the proposed framework successfully targets rare, high-loss scenarios that are otherwise underrepresented in the base distribution.
this section cite: []

Section: Discussion, Limitations of Work, and Future Directions
This paper introduces RAMIS, a novel risk-averse learning framework that integrates score-based generative modeling with pretrained model feedback to synthesize high-loss, informative samples for downstream optimization. In contrast to existing generative approaches that aim to increase sample diversity or generalization, our framework targets the utility of samples specifically for minimizing tail-risk objectives such as CVaR. By leveraging pretrained loss signals as importance guidance, we enable generative models to contribute directly to risk-sensitive training.
Our study primarily focuses on the theoretical motivation behind loss-guided generative sampling and demonstrates its effectiveness through controlled synthetic experiments and a domain-specific application in wireless communication. While these results validate the core principles of RAMIS, broader applications remain to be explored. As diffusion-based generative models continue to evolve and become increasingly accessible across domains where risk-sensitive optimization is critical, we expect RAMIS to generalize naturally to these settings.
this section cite: []

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The abstract and introduction accurately reflect the paper's contributions and scope.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Limitations are discussed in Section 6. Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes]
16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The main method development in this research does not involve LLMs. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: → Corresponding Author. Source code: https://github.com/Heasung-Kim/generating-informative-samples-for-risk-averse-fine-tuning-ofdownstream-tasks 39th Conference on Neural Information Processing Systems (NeurIPS 2025).

this section cite: []

Section: 
Justification: It provides the full set of assumptions and a complete and correct proof. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: Yes, the implementation details are provided along with source code.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: Yes, we provide a Github link.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes]
Justification: The paper specifies all the implementation details in the appendices.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: Yes. We provide the standard deviation of the performance under multiple random seeds.
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
Answer: [Yes] Justification: Yes. This work provides sufficient information regarding the computational resources for the experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: Yes. This work complies with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA]
Justification: There is no specific societal impact of the work.
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
Justification: This paper does not pose such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: Yes. All external assets are properly credited.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes] Justification: Yes. The implementation code is clearly documented and released with the paper.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: This paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
this section cite: []

Section: Institutional review board (IRB) approvals or equivalent for research with human subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
Answer: [NA] Justification: This paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
this section cite: []

Section: References
Ref_id:b0 Title: Risk-averse control via CVaR barrier functions: Application to bipedal robot locomotion Year: (2022)
Ref_id:b1 Title: Minimizing CVaR and VaR for a portfolio of derivatives Year: (2006)
Ref_id:b2 Title: Reverse-time diffusion equation models Year: (1982)
Ref_id:b3 Title: Credit risk optimization with conditional value-at-risk criterion Year: (2001)
Ref_id:b4 Title: Residual diffusion models for variable-rate joint source channel coding of MIMO CSI Year: (2025)
Ref_id:b5 Title: CNN-RNN and data augmentation using deep convolutional generative adversarial network for environmental sound classification Year: (2022)
Ref_id:b6 Title: Computing VaR and CVaR using stochastic approximation and adaptive unconstrained importance sampling Year: (2009)
Ref_id:b7 Title: Attention model for massive MIMO CSI compression feedback and recovery Year: (2019)
Ref_id:b8 Title: A robust-CVaR optimization approach with application to breast cancer therapy Year: (2014)
Ref_id:b9 Title: Stability and generalization of learning algorithms that converge to global optima Year: (2018)
Ref_id:b10 Title: Risk-averse fine-tuning of large language models Year: (2024)
Ref_id:b11 Title: A comprehensive survey for generative data augmentation Year: (2024)
Ref_id:b12 Title: ILVR: Conditioning method for denoising diffusion probabilistic models Year: (2021)
Ref_id:b13 Title: Algorithms for CVaR optimization in MDPs Year: (2014)
Ref_id:b14 Title: Diffusion posterior sampling for general noisy inverse problems Year: (2023)
Ref_id:b15 Title: G: LTE/LTE-advanced for mobile broadband Year: (2013)
Ref_id:b16 Title: Extremiles: A new perspective on asymmetric least squares Year: (2019)
Ref_id:b17 Title: Extremile regression Year: (2022)
Ref_id:b18 Title: Stochastic model-based minimization of weakly convex functions Year: (2019)
Ref_id:b19 Title: Efficient black-box importance sampling for var and CVaR estimation Year: (2021)
Ref_id:b20 Title: Adaptive importance sampling for finite-sum optimization and sampling with decreasing step-sizes Year: (2020)
Ref_id:b21 Title: Conditional value-at-risk beyond finance: a survey Year: (2020)
Ref_id:b22 Title: Overview of deep learning-based CSI feedback in massive MIMO systems Year: (2022)
Ref_id:b23 Title: Identity matters in deep learning Year: (2016)
Ref_id:b24 Title: Adaptive importance sampling for efficient stochastic root finding and quantile estimation Year: (2024)
Ref_id:b25 Title: Generative data augmentation with differential privacy for non-IID problem in decentralized clinical machine learning Year: (2024)
Ref_id:b26 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b27 Title: QuaDRiGa: A 3-D multi-cell channel model with time evolution for enabling virtual field trials Year: (2014)
Ref_id:b28 Title: QuaDRiGa-quasi deterministic radio channel generator, user manual and documentation Year: (2021)
Ref_id:b29 Title: Linear convergence of gradient and proximal-gradient methods under the polyak-"ojasiewicz condition Year: (2016)
Ref_id:b30 Title: Importance sampling via score-based generative models Year: (2025)
Ref_id:b31 Title: Generative diffusion model-based compression of MIMO CSI Year: (2025)
Ref_id:b32 Title: Test-time alignment of diffusion models without reward over-optimization Year: (2025)
Ref_id:b33 Title: Generating high dimensional user-specific wireless channels using diffusion models Year: (2025)
Ref_id:b34 Title: An overview of 5G advanced evolution in 3GPP Release 18 Year: (2022)
Ref_id:b35 Title: Stochastic AUC maximization with deep neural networks Year: (2019)
Ref_id:b36 Title: Hyperrnn: Deep learning-aided downlink CSI acquisition via partial channel reciprocity for FDD massive MIMO Year: (2021)
Ref_id:b37 Title: Multi-resolution CSI feedback with deep learning in massive MIMO system Year: (2020)
Ref_id:b38 Title: Repaint: Inpainting using denoising diffusion probabilistic models Year: (2022)
Ref_id:b39 Title: A model-based method for minimizing CVaR and beyond Year: (2023)
Ref_id:b40 Title: Stochastic gradient methods for distributionally robust optimization with f-divergences Year: (2016)
Ref_id:b41 Title: Stochastic gradient descent, weighted sampling, and the randomized Kaczmarz algorithm Year: (2014)
Ref_id:b42 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b43 Title: Optimization of Conditional Value-at-Risk Year: (2000)
Ref_id:b44 Title: Generative adversarial networks for data augmentation in machine fault diagnosis Year: (2019)
Ref_id:b45 Title: Semantic data augmentation with generative models Year: (2023)
Ref_id:b46 Title: Statistical learning with conditional value at risk Year: (2020)
Ref_id:b47 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b48 Title: Optimizing the CVaR via sampling Year: (2015)
Ref_id:b49 Title: Application of CVaR risk aversion approach in the dynamical scheduling optimization model for virtual power plant connected with windphotovoltaic-energy storage system with uncertainties and demand response Year: (2017)
Ref_id:b50 Title: Effective data augmentation with diffusion models Year: (2023)
Ref_id:b51 Title: Bias-corrected peaks-over-threshold estimation of the CVaR Year: (2021)
Ref_id:b52 Title: Neural discrete representation learning Year: (2017)
Ref_id:b53 Title: Diffusers: State-of-the-art diffusion models Year: (2022)
Ref_id:b54 Title: Deep learning-based CSI feedback approach for time-varying massive MIMO channels Year: (2018)
Ref_id:b55 Title: Domain gap embeddings for generative dataset augmentation Year: (2024)
Ref_id:b56 Title: A comprehensive survey on data augmentation Year: (2024)
Ref_id:b57 Title: Deep learning for massive MIMO CSI feedback Year: (2018)
Ref_id:b58 Title: Generative modeling and data augmentation for power system production simulation Year: (2024)
Ref_id:b59 Title: Robust wireless rechargeable sensor networks Year: (2022)
Ref_id:b60 Title: Freedom: Training-free energy-guided conditional diffusion model Year: (2023)
Ref_id:b61 Title: Ulf Gustavsson, Giuseppe Durisi, and Xiaoming Chen. 5G Physical Layer: principles, models and technology components Year: (2018)
Ref_id:b62 Title: DORO: Distributional and outlier robust optimization Year: (2021)
Ref_id:b63 Title: Stochastic optimization with importance sampling for regularized loss minimization Year: (2015)
Ref_id:b64 Title: Toward understanding generative data augmentation Year: (2023)
Ref_id:b65 Title: Characterization of gradient dominance and regularity conditions for neural networks Year: (2017)
