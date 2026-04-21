Title: Learnable Sampler Distillation for Discrete Diffusion Models
Abstract: Discrete diffusion models (DDMs) have shown powerful generation ability for discrete data modalities like text and molecules. However, their practical application is hindered by inefficient sampling, requiring a large number of sampling steps. Accelerating DDMs by using larger step sizes typically introduces significant problems in generation quality, as it amplifies the impact of both the compounding decoding error due to factorized predictions and discretization error from numerical approximations, leading to a significant decrease in sampling quality. To address these challenges, we propose learnable sampler distillation (LSD), a novel approach to train fast and high-fidelity samplers for DDMs. LSD employs a distillation approach where a student sampler with a few steps learns to align its intermediate score trajectory with that of a high-quality teacher sampler with numerous steps. This alignment is achieved by optimizing learnable sampler coefficients that adaptively adjust sampling dynamics. Additionally, we further propose LSD+, which also learns time schedules that allocate steps non-uniformly. Experiments across text generation, image generation, and synthetic tasks demonstrate that our proposed approaches outperform existing samplers for DDMs, achieving substantially higher sampling quality with significantly fewer sampling steps. Our code is available at https://github.com/feiyangfu/LSD.

Section: Introduction
Diffusion models have demonstrated remarkable success across various generative tasks, particularly excelling in the synthesis of data within continuous domains like images, audio, and videos [1,2,3,4,5]. These models frame the data generation process as a gradual denoising procedure in a continuous latent space. However, many other important data modalities, such as natural language, molecular sequences, and categorical data, inherently possess discrete structures. Applying diffusion models directly to these discrete spaces is challenging, as the standard formulation relies on continuous state transitions. Recently, discrete diffusion models (DDMs) [6,7,8,9,10,11,12] have been developed to address this issue. DDMs are specifically designed to operate on discrete data, adapting the core diffusion idea to categorical variables and enabling principled generation. Recent advances in DDMs have shown promising results, achieving competitive performance in generating high-fidelity discrete data. Despite their promising applicability, DDMs face an important challenge in sampling efficiency, and they typically require a substantial number of function evaluations (NFEs), e.g., 1024 or more, making inference computationally expensive.
Current sampling methods for DDMs are mainly divided into two categories: 1) Exact simulation methods [13,14] provide unbiased samples from the target distribution but suffer from high sampling times and expensive computational costs due to numerous model evaluations, leading to poor scaling with dimensionality. 2) Approximate methods like τ -leaping [15,16] are designed for parallelization and potentially faster sampling. However, such methods are first-order accurate and require small step sizes to ensure sampling quality.
Directly accelerating the sampling of DDMs through reducing NFEs typically produces unsatisfactory results, since this amplifies the impact of the compounding decoding error [17] and discretization error. Compounding decoding error arises since DDMs employ a factorized parameterization for computational efficiency, predicting the denoised state of each token independently, and ignoring inherent dependencies between tokens in the sequence. Consequently, the learned factorized denoising distribution differs from the true reversal process. This discrepancy is exacerbated when reducing NFEs, as the approximation quality degrades over larger intervals. Discretization error occurs since large step sizes make it inaccurate for numerical methods like Euler [18] and τ -leaping [15] to approximate the reverse dynamics. Moreover, these two errors accumulate over the sampling trajectory, severely degrading sampling quality when using small NFEs. Throughout the following, we call the combination of compounding decoding error and discretization error as accumulated error for brevity. To address the issue incurred by large accumulated error, we propose learnable sampler distillation (LSD) and its improved version for efficient sampling of DDMs.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b14']

Section: Related Work
Efficient sampling in continuous diffusion models Recent efforts to accelerate sampling in continuous diffusion models largely focus on reducing the NFEs for solving the reverse-time ordinary differential equation (ODE) or stochastic differential equation (SDE) [19,4,20,21,22,23,24].
One major direction involves designing advanced ODE solvers. Some works [20,25,26,27,4] provide efficient sampling methods by establishing high-order numerical ODE solvers for continuous diffusion models.
There are also approaches that learn or optimize various components of the sampling process. AYS [28] seeks non-uniform time step schedules specific to given models and datasets, though their optimization can be computationally intensive. DMN [29] proposes a general framework for designing an optimization problem that seeks more appropriate time steps by minimizing the distance between the ground-truth solution to the ODE and an approximate solution corresponding to the numerical solver. AMED-Solver [30] learns adaptive mean estimation directions based on the observation that trajectories often reside in low-dimensional subspaces, which typically involves training an auxiliary network with high costs.
The most relevant works to us are perhaps LD3 [31] and S4S [32], both proposing learning diffusion model solvers via distillation in the continuous domain. LD3 efficiently learns the time discretization by backpropagating through the ODE-solving procedure using the proposed surrogate loss. S4S further learns the coefficients of the student solver by minimizing the distance between the final samples generated by the student and teacher solvers using learned non-uniform time schedules. However, these approaches face challenges when applied to DDMs. We highlight several distinctions in our learnable sampler distillation (LSD) approach (and its improved version) designed to address these challenges. 1) DDM sampling involves non-differentiable categorical sampling at each step, obstructing direct gradient flow from the final discrete output back to the sampler parameters. The reliance of S4S on final sample comparison is thus infeasible. We address this issue by aligning the intermediate score trajectories between the student and teacher samplers. This provides a viable path for gradient-based optimization of the learnable coefficients within the discrete sampling methods.
2) The work for S4S uses the final sample matching error to learn the time schedules, which may ignore dynamic changes of the accumulated error in the intermediate steps. We instead learn the time steps by aligning the effective transition term at intermediate stages during the reverse process. The effective transition term in the reverse process incorporates step sizes and concrete scores, which are tailored for DDMs. 3) The work for S4S optimizes the continuous initial noise using projected stochastic gradient descent (SGD) within an L 2 ball. This is inapplicable to DDMs where the initial state is often a discrete sequence, e.g., all masked tokens, which lacks a continuous gradient. To address this issue, we adapt the approach by measuring proximity using Hamming distance, which is suitable for discrete spaces and does not perform gradient updates on itself.
this section cite: ['b18', 'b3', 'b19', 'b20', 'b21', 'b22', 'b23', 'b19', 'b24', 'b25', 'b26', 'b3', 'b27', 'b28', 'b29', 'b30', 'b31']

Section: Distillation in Diffusion Models
The distillation of continuous diffusion models is a rapidly advancing field. A prominent direction is related to the consistency model [33], which aims to learn a function that maps any point on an ODE trajectory to its origin, enabling one-step or few-step generation. This paradigm has been extended to multi-step variants [34,35] for improved performance. Other significant works focus on directly matching student and teacher distributions, such as distilling guided diffusion models [36], proposing simplified and faster matching objectives [37], recursively distilling a deterministic diffusion sampler into a new model [38], or concentrating on one-step distillation [39]. While these methods are highly effective for continuous models, they usually rely on continuous paths in the sense that the sampling process of each step is differentiable. Our work diverges by proposing a distillation framework specifically for the discrete diffusion model, which does not assume such a continuous path, and addressing a different set of challenges like the non-differentiability of the outputs. A recent work for Di[M]O [40] also involves distilling discrete diffusion models. It distills a multi-step masked diffusion model into a one-step generator. This is achieved by training a new student model from scratch, using a sophisticated proxy objective that involves creating "pseudo-intermediate states" and training an auxiliary model to match conditional output distributions. Our approaches are significantly different with Di[M]O in both goal and mechanism. Similarly to LD3 and S4S, we focus on a few-step sampler distillation. We tackle the challenge of non-differentiability of sampling from categorical distributions, and we enhance an existing sampler rather than replacing the model, which avoids the complexity of training a new generator and an auxiliary model.
this section cite: ['b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39']

Section: Discrete diffusion models
DDMs have emerged and undergone substantial development recently. SEDD [6] proposes score entropy, a novel loss that naturally extends score matching to discrete spaces and integrates seamlessly to build DDMs. RADD [9] reveals that the concrete score in absorbing diffusion can be expressed as conditional probabilities of clean data, multiplied by a time-dependent scalar in an analytic form and it unifies absorbing DDMs and any-order autoregressive models.
Various strategies have been proposed to accelerate the sampling of DDMs while maintaining quality. Among these, approximate simulation methods [18,16,15,8] are widely used due to their potential for parallelization. A prominent example is the τ -leaping algorithm [8] that is adapted for DDMs. τ -leaping simulates the process by taking an approximate Euler-like step at each data dimension simultaneously and independently. Tweedie τ -leaping [6,41] is an extension to τ -leaping and is proposed to improve accuracy by specifically considering how the rate matrix changes according to the noise schedule throughout the reverse process. While these τ -leaping variants offer the advantage of parallelization, the inherent approximation error still necessitates using many small steps to achieve high sampling quality.
The recent work for JYS [17] attempts to accelerate the sampling process of DDMs by focusing on optimizing the time steps of the sampling schedule. It minimizes a Kullback-Leibler divergence upper bound (KLUB) that implicitly captures the overall impact of the compounding decoding error and strategically allocates sampling steps. However, JYS operates by optimizing when to sample, rather than how to sample. At each chosen time, it still relies on the intrinsically biased model and employs standard large-step approximations that suffer from a significant discretization error.
this section cite: ['b5', 'b8', 'b17', 'b15', 'b14', 'b7', 'b7', 'b5', 'b40', 'b16']

Section: Contributions
To address limitations mentioned above, we move beyond fixed or hand-tuned inference strategies. We introduce a novel learnable sampler distillation (LSD) approach specifically for DDMs. We employ a teacher sampler using small step sizes to approximate a high-quality trajectory. A student sampler is then trained with larger step sizes. Instead of mimicking only the final output of the teacher sampler, which is challenging due to non-differentiability in the discrete pipeline, the student sampler learns to align its intermediate score trajectory with that of the teacher sampler. This alignment is achieved by optimizing the learnable sampler coefficients, which provide the ability to adaptively adjust the sampling dynamics at each step and potentially compensate for the accumulated error given larger step sizes. Furthermore, we propose LSD+ that also learns sampling time schedules. This is done by comparing the effective transition term in the reverse process at intermediate stages and empirically works better compared with uniform sampling schedules used by LSD. We also utilize a relaxed objective during the learning process to alleviate the difficulty of hard alignment between the teacher and student samplers.
Overall, our contribution can be summarized as follows:
• We propose the LSD approach. Inspired by the insight of aligning intermediate score trajectories, LSD trains an efficient student sampler via distillation by optimizing learnable sampler coefficients and incorporating a relaxed training objective for improved feasibility.
• We further introduce LSD+, an extension to LSD that additionally learns non-uniform time schedules. This allows for adaptive allocation of sampling steps, offering a mechanism to potentially better capture varying dynamics and further reduce accumulated errors compared to using uniform time schedules.
• Extensive experiments across text generation, image generation, and synthetic data tasks demonstrate that our proposed approaches achieve significantly higher sampling quality compared to existing baselines at reduced NFEs.
this section cite: []

Section: Preliminaries

this section cite: []

Section: Continuous time discrete diffusion models
DDMs model the generative process that can be expressed as a continuous time Markov chain (CTMC) on a finite state space X = {1, . . . , N } [8,42]. The forward process describes how data is corrupted. Specifically, the probability of transitioning from state x at time t to state y after a small time interval ∆t is denoted by p t+∆t|t (y|x). This is characterized by [9]:
p t+∆t|t (y|x) = Q t (x, y)∆t + o(∆t), y ̸ = x, 1 + Q t (x, x)∆t + o(∆t), y = x,(1)
where Q t (x, y) is the (x, y) element of the transition rate matrix Q t . The transition rate matrix Q t is usually formed as σ(t)Q [8], where σ(t) is a scalar factor, Q is a pre-defined standard matrix with special structures [8]. Let p t denote the marginal distribution of states at time t. In particular, p 0 = p data is the true distribution of the data. Additionally, for the terminal time T , p T approaches a distribution π. Depending on Q, π can mainly be modeled as two distributions, namely a uniform distribution or a distribution that converts samples into masked tokens. For the reverse process that transfers p T back to p 0 , the inverse CTMC can be characterized as follows [9]:
p t-∆t|t (y|x) = Qt (x, y)∆t + o(∆t), y ̸ = x, 1 + Qt (x, x)∆t + o(∆t), y = x, (2
)
where Qt is the reverse transition rate matrix [41], which can be parameterized by:
Q(x, y) = pt(y) pt(x) Q t (y, x), y ̸ = x, -z̸ =x Qt (x, z), y = x.
(
)3
The concrete score term pt(y) pt(x) needs to be estimated, as p t is generally unknown. Therefore, the goal of training a score network s θ : X × R → R |X | is to approximate these score values. For instance, SEDD [6] provides an effective method that learns s θ such that it satisfies s θ (x, t) ≈ pt(y) pt(x) y̸ =x .
this section cite: ['b7', 'b41', 'b8', 'b7', 'b7', 'b8', 'b40', 'b5']

Section: Methods
Achieving efficient inference in DDMs with high sampling quality necessitates accurate approximation of the reverse CTMC using significantly fewer steps than traditional high-fidelity samplers. Numerical samplers like the Euler sampler approximate this process by taking discrete steps guided by the concrete score from the model. However, increasing the step size for faster inference alters the discrete transition dynamics, leading to increased accumulated errors and degraded sampling quality.
To enable accurate sampling with large step sizes, we propose learnable sampler distillation (LSD) to make some components of the numerical sampler learnable. Specifically, LSD employs learnable coefficients to dynamically adjust the influence of the concrete score at each time step, allowing the sampler to compensate for large-step discretization errors. Furthermore, while LSD could lead to significant improvement in the generation quality, we further propose LSD+. Instead of learning the coefficients using a fixed uniform schedule, LSD+ further learns a sequence of non-uniform time steps. By training these parameters through distillation from a high-quality teacher sampler, our method learns an optimized discrete-time trajectory. Figure 1 shows the pipeline of our method, and the details of the method are described in the following subsections. ). During inference, sampling starts from the initial sample x t0 . We can see that the vanilla sampling trajectory often introduces significant discretization errors with large step sizes. In contrast, LSD+ employs learnable coefficients Φ(t k ) and learnable time schedules τ k to adaptively adjust its trajectory. This enables the LSD sampler to more accurately mimic the trajectory of the teacher sampler, effectively compensating for errors inherent in accelerated sampling.
this section cite: []

Section: LSD: Coefficients
In this subsection, we illustrate our LSD approach that learns time-dependent coefficients for accelerating DDMs.
Given a pre-trained score network s θ (•, •), transition rate matrix Q t , and an initial state x T sampled from p T at the initial time T , the reverse sampling process for DDMs generates a sample by iteratively applying an update rule. For an Euler-type sampler, the transition probability for the i-th token from its current state x t at time t to the next state x t-∆t at time t -∆t can be parameterized as: p(x i t-∆t |x i t ) = δ x i t (x i t-∆t ) + ∆t Q t (x i t , x i t-∆t ) s θ (x t , t) i,x i t-∆t . (4) Here, x i t denotes the i-th token of the current state sequence x t , δ x i t (x i t-∆t ) is the Kronecker delta function, ∆t represents the time step size, Q t (x i t , x i t-∆t ) denotes the (x i t , x i t-∆t ) element of the transition rate matrix Q t , and s θ (x t , t) i,x i t-∆t is the (i, x i t-∆t ) element of the concrete score s θ (x t , t). We apply a fixed teacher sampler that approximates the true reverse process with high fidelity using time schedules {t * j } N j=0 comprising N steps, with T = t * 0 > t * 1 > • • • > t * N = ϵ > 0. 2 The state generated by the teacher sampler at time t * j along its trajectory is denoted as x * tj . The sampling process of a teacher sampler Ψ * yields a high-quality final sample x * ϵ = Ψ * x T , {t * j } N j=0 , s θ , {Q t * j } N j=0 , which is abbreviated as x * ϵ = Ψ * (x T ) for simplicity in notation. We apply a student sampler that operates with a time schedule {t k } M k=0 comprising M steps, where M ≪ N and
T = t 0 > t 1 > • • • > t M = ϵ > 0. {t k } M
k=0 is a subsequence of {t * j } N j=0 . Our goal is to learn a set of time-dependent coefficients Φ = {Φ(t k )} M k=1 to improve the quality of the output of the student sampler. 3 The state generated by the student sampler at time t k along its trajectory is denoted as x t k . The sampling process of a student sampler Ψ yields a final sample
x ϵ = Ψ x T , {t k } M k=0 , s θ , {Q t k } M k=0 , {Φ(t k )} M k=1
, which is abbreviated as x ϵ = Ψ Φ (x T ) to highlight the dependence on the coefficients {Φ(t k )} M k=1 . The update rule for the i-th token within the student sampler incorporating Φ becomes:
p(x i t k+1 |x i t k ) = δ x i t k (x i t k+1 ) + ∆t Q t k (x i t k , x i t k+1 ) (Φ(t k )s θ (x t k , t k )) i,x i t k+1 .(5)
Similar to strategies in some learning methods for continuous ODE solvers [32], a direct objective is to minimize the distance d(x ϵ , x * ϵ ), where d(•, •) is a certain distance metric. However, it is generally infeasible for DDMs to directly minimize d(x ϵ , x * ϵ ) since the non-differentiable categorical sampling at each step obstructs gradient propagation. Instead, we propose to align intermediate score predictions. At each time step t k , the student sampler computes its score s k = s θ (x t k , t k ). The teacher sampler evolves its state to the same time step t k (i.e., for certain j such that t * j = t k ) using its more accurate sampling process and caches its score s * k = s θ (x * t k , t k ). The states x t k and x * t k differ due to the distinct sampling paths taken to reach t k . Then, our objective is to minimize the discrepancy between s * k and Φ(t k )s k for all k ∈ {1, 2, . . . , M }. This can be expressed as:
L k (Φ(t k )) = E xt 0 ∼π [d (s * k , Φ(t k )s k )] .(6)
This intermediate score trajectory alignment provides a differentiable path for optimizing {Φ(t k )} M k=1 and ensures the student sampler mimics the trajectory of the teacher sampler across the full denoising path, not just at the final output. We present details of the sampling and training processes for LSD in Algorithms 1 and 2 respectively.
this section cite: ['b31']

Section: LSD+: Coefficients with learnable time schedules
While LSD improves the sampler by learning the sequence of coefficients {Φ(t k )} M k=1 under a fixed time schedule, the reverse diffusion dynamics vary significantly across time. We additionally propose LSD+ to also learn non-uniform time schedules. The intuition is that, by learning from a high-fidelity teacher sampler, the student sampler implicitly learns to allocate its limited steps in a manner that best approximates the trajectory of the teacher. Specifically, given time steps for the student sampler {t k } M k=0 , the uniform time schedule uses a step size at ∆t = T -ϵ M . Our goal is to learn customized step sizes {κ k } M k=1 , which are initialized as ∆t. The learnable time steps are calculated by:
τ k = T - k ℓ=1 κ ℓ .(7)
At each learned time step τ k , the student sampler computes its score s θ (x τ k , τ k ). To learn the step size κ k , we utilize the so-called effective transition term in the reverse process. Specifically, for the student sampler, this is proportional to κ k s θ (x τ k , τ k ), for the teacher sampler, this is proportional to
T -ϵ N s θ (x * t k , t k ),
where T -ϵ N is the step size for the teacher sampler and s θ (x * t k , t k ) is the cached score of teacher sampler. By calculating the distance of the effective transition terms between the student sampler and teacher sampler, we effectively update κ k considering the unique characteristics of DDMs. This allows the time schedule to adaptively allocate step sizes based on the specific transition structures in DDMs. The updating process can be parameterized as follows:
Lk (κ k ) = E xt 0 ∼π d κ k s θ (x τ k , τ k ), T -ϵ N s θ (x * t k , t k ) .(8)
We present details of the training and sampling processes for LSD+ in the supplementary material.
this section cite: []

Section: Algorithm 1 Sampling process of LSD
Require: Score network s θ , time schedule {t k } M k=0 for the student sampler, learned coefficients of the student sampler
{Φ(t k )} M k=1 , transition rate matrices {Q t k } M k=0 1: Sample x t0 ∼ π 2: for k = 0 to M -1 do 3:
Sample x t k+1 based on x t k and Φ(t k ):
4: p(x i t k+1 |x i t k ) = δ x i t k (x i t k+1 ) + (t k -t k+1 ) Q t k (x i t k , x i t k+1 ) (Φ(t k )s θ (x t k , t k )) i,x i t k+1
5:
x i t k+1 ∼ p(x i t k+1 |x i t k
) for all i 6: end for 7: return x ϵ Algorithm 2 Training process of LSD Require: Score network s θ , frozen teacher sampler Ψ * with N steps, learnable student sampler Ψ Φ with M steps, learning rate η, distance metric d, time schedule {t * j } N j=0 for the teacher sampler, time schedule {t k } M k=0 for the student sampler (a subsequence of {t * j } N j=0 ), transition rate matrices {Q t * j } N j=0 1: Initialize Φ(t k ) = 1 for k = 1, . . . , M 2: while not converged do
3: Sample x t0 ∼ π, set x * t0 ← x t04:
for k = 1 to M do 5:
Calculate the state x t k generated by the student sampler at time t k and calculate the score
s k = s θ (x t k , t k ) 6:
Calculate the state x * t k generated by the teacher sampler at time t k and calculate the score
s * k = s θ (x * t k , t k ) 7:
end for 8:
for k = 1 to M do 9:
L k ← d(Φ(t k )s k , s * k ) 10: Φ(t k ) ← Φ(t k ) -η∇ Φ(t k ) L k 11:
end for 12: end while 13: return {Φ(t k )} M k=1
this section cite: []

Section: Relaxed objective
For a student sampler which typically has lower NFEs compared to a teacher sampler, it is non-trivial to force it to accurately match the output of the teacher sampler given the same initial input x t0 . Thus, we adopt a relaxed training objective for both LSD and LSD+. We take LSD as an example for further presentation. Instead of strictly requiring the score of the student sampler s θ (x t0 , t 0 ) to match the score of the teacher sampler s θ (x * t0 , t 0 ), we only require that there exists an alternative input xt0 sufficiently close to the original x t0 (within a small Hamming distance [43] in our discrete token space). Specifically, xt0 satisfies:
d H (x t0 , xt0 ) ≤ ζ,(9)
where d H (•, •) denotes the Hamming distance between two sequences, ζ represents positive integer threshold that defines the maximum allowed Hamming distance between xt0 and x t0 , where we set it as around 5% of the sequence length.
Therefore, the output of the student sampler at this perturbed score should approximately match the score of the teacher sampler at the original input such that s θ (x t0 , t 0 ) ≈ s θ (x * t0 , t 0 ). Moreover, the relaxed objective function for LSD can be expressed as:
L relaxed,k (Φ(t k )) = E xt 0 ,xt 0 d s θ (x * t k , t k ), Φ(t k )s θ (x t k , t k ) ,(10)
where xt0 and x t0 satisfies Eq. ( 9) and xt k is sampled starting from xt0 . This relaxation makes the optimization task more feasible for the capacity-constrained student sampler by alleviating the rigorous matching requirement. Notably, this input perturbation xt0 is only used during training, at inference time, the student sampler receives the original and unperturbed input x t0 . We provide further discussion on the reasonableness of the relaxed objective in the supplementary material.
this section cite: ['b42']

Section: Experiments
In this section, we empirically evaluate the performance of our proposed LSD approach and its improved version LSD+. Our goal is to validate their ability to generate high-quality samples at low NFEs. We conduct evaluations across diverse settings, including text generation, image generation, and a synthetic sequence task, comparing against various baselines. We highlight that our LSD+ provides an efficient learning process for the coefficients and time schedules, typically requiring 5 minutes on an NVIDIA RTX4090 GPU, compared to around 10 minutes of training time for JYS under the same environment. And the learned student sampler introduces no additional computational burden during sampling.
this section cite: []

Section: Text generation
For the text generation task, we employed three pre-trained DDM backbones for validation, namely SEDD-small [6], SEDD-medium [6], and RADD [9]. These are absorbing DDMs of GPT-2 level for text generation, trained on the OpenWebText dataset [44]. For the uniform DDMs, please refer to the supplementary material. We compare LSD and LSD+ against standard Euler and Tweedie samplers [6] and the JYS method [17]. For the RADD baseline, we also compare with higher-order samplers, the θ-RK-2 and θ-trapezoidal [45]. 4 We evaluate the generative perplexity of unconditionally generated text using a GPT2-large model. We generated 1024 samples, each containing 1024 tokens.
The results are presented in Tables 1, 2, and 3. LSD (LSD+)-Euler (Tweedie) denotes that we implement LSD (LSD+) based on the Euler (Tweedie) sampler [6]. The empirical results show that our methods significantly outperform the baseline methods across all three backbones and all tested NFEs. Moreover, we find that LSD+ generally outperforms LSD, which indicates that the learned non-uniform time schedules help to further reduce accumulated errors. Therefore, we only present the results for LSD+ for the experiments in Sections 4.2 and 4.3.
this section cite: ['b5', 'b5', 'b8', 'b43', 'b5', 'b16', 'b44', 'b5']

Section: Image generation
We also validate our LSD+ approach on the image generation task for the CIFAR-10 dataset [46]. We utilize CTMC [8] as the baseline, which employs a Gaussian transition matrix and denoising parameterization. Each data sample is a flattened image with a size of 3 × 32 × 32, composed of tokens with values ranging from 0 to 255. We evaluate the FID score using 50k samples with the NFEs selected from {8, 16, 32, 64}. Figure 2(a) shows the results and we can observe that our method provides better FID scores compared to the baseline method.
this section cite: ['b45', 'b7']

Section: Synthetic countdown task
We follow [47] to evaluate our LSD+ approach on a synthetic sequence task with strong dependencies. The dataset features 256-token sequences (with values in 0-31) where non-zero tokens must strictly decrease by one. We trained an absorbing SEDD [6] model and measured performance by the error rate, which is the proportion of generated samples violating this countdown rule. As shown in Figure 2(b), our method achieves lower error rates across various NFEs compared to baselines.
this section cite: ['b46', 'b5']

Section: Ablation Study
The learnable coefficients form the core of our LSD approach and have demonstrated significant performance improvements. Also, we observe that LSD+ generally outperforms LSD as seen in Tables 1, 2, and 3, which indicates the benefit of the learned non-uniform time schedules. Therefore, our ablation study aims to assess the contributions of the relaxed objective during training.
this section cite: []

Section: Benefit of the relaxed objective
We proposed a relaxed objective, allowing the student sampler to match the trajectory of the teacher sampler originating from x t0 by using a perturbed starting point xt0 that is close to x t0 during the training process. Table 4 compares the performance of LSD+ trained with and without this relaxation. The results clearly indicate that employing the relaxed objective generally yields better performance than training with the strict objective. This validates the benefit of the relaxation, confirming that it makes the trajectory alignment task more feasible and leads to better convergence.
this section cite: []

Section: Impact of Hamming distance threshold
To investigate the robustness of the algorithm to the Hamming distance threshold, we conduct the ablation on the SEDD-small backbone using the Euler sampler with 32 inference steps. We train our LSD+ method using several different values for the Hamming distance threshold, specifically 0%, 1%, 5%, 10%, 20% of the sequence length, while keeping all other hyperparameters unchanged. The performance, measured by Perplexity, is reported below in Table 5.
this section cite: []

Section: Conclusion
This paper aims to address the challenge of inefficient sampling in DDMs, a major obstacle to their practical deployment. While reducing the NFEs accelerates inference, previous accelerating methods suffer from accumulated compounding decoding error and discretization errors, significantly degrading sampling quality. We introduce LSD, a novel approach that leverages distillation from a high-fidelity teacher sampler. Instead of merely matching final outputs, LSD trains a student sampler with a few steps to align its entire intermediate score trajectory with that of the teacher sampler. This is achieved by optimizing learnable, time-dependent coefficients. And we additionally propose LSD+ that also learns non-uniform sampling schedules and this allows the sampler to adaptively compensate for errors induced by larger step sizes. Extensive experiments demonstrate that our methods significantly outperform the baseline samplers across diverse tasks, achieving high sampling fidelity at low NFEs.
A promising direction for future research is to provide theoretical guarantees regarding the distributional discrepancy between the outputs of teacher and student samplers, potentially building on existing theoretical findings related to discrete diffusion models [48,49,50,51,52,45,53,54].
this section cite: ['b47', 'b48', 'b49', 'b50', 'b51', 'b44', 'b52', 'b53']

Section: References
Ref_id:b0 Title: Analog bits: Generating discrete data using diffusion models with self-conditioning Year: (2022)
Ref_id:b1 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b2 Title: DiffWave: A versatile diffusion model for audio synthesis Year: (2021)
Ref_id:b3 Title: DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps Year: (2022)
Ref_id:b4 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b5 Title: Discrete diffusion modeling by estimating the ratios of the data distribution Year: (2024)
Ref_id:b6 Title: Structured denoising diffusion models in discrete state-spaces Year: (2021)
Ref_id:b7 Title: A continuous time framework for discrete denoising models Year: (2022)
Ref_id:b8 Title: Your absorbing discrete diffusion secretly models the conditional distributions of clean data Year: (2025)
Ref_id:b9 Title: Concrete score matching: Generalized score matching for discrete data Year: (2022)
Ref_id:b10 Title: Discrete flow matching Year: (2024)
Ref_id:b11 Title: Fast sampling via discrete non-Markov diffusion models with predetermined transition time Year: (2024)
Ref_id:b12 Title: Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling Year: (2025)
Ref_id:b13 Title: Simplified and generalized masked diffusion for discrete data Year: (2024)
Ref_id:b14 Title: Approximate accelerated stochastic simulation of chemically reacting systems Year: (2001)
Ref_id:b15 Title: Tweedie's formula and selection bias Year: (2011)
Ref_id:b16 Title: Jump your steps: Optimizing sampling schedule of discrete diffusion models Year: (2025)
Ref_id:b17 Title: Error analysis of tau-leap simulation methods Year: (2011)
Ref_id:b18 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b19 Title: DPM-Solver-v3: Improved diffusion ODE solver with empirical model statistics Year: (2023)
Ref_id:b20 Title: SA-Solver: Stochastic Adams solver for fast sampling of diffusion models Year: (2023)
Ref_id:b21 Title: Accelerating convergence of score-based diffusion models, provably Year: (2024)
Ref_id:b22 Title: The surprising effectiveness of skip-tuning in diffusion sampling Year: (2024)
Ref_id:b23 Title: Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation Year: (2024)
Ref_id:b24 Title: DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic models Year: (2022)
Ref_id:b25 Title: Fast sampling of diffusion models with exponential integrator Year: (2023)
Ref_id:b26 Title: GENIE: Higher-order denoising diffusion solvers Year: (2022)
Ref_id:b27 Title: Align your steps: Optimizing sampling schedules in diffusion models Year: (2024)
Ref_id:b28 Title: Accelerating diffusion sampling with optimized time steps Year: (2024)
Ref_id:b29 Title: Fast ODE-based sampling for diffusion models in around 5 steps Year: (2024)
Ref_id:b30 Title: Learning to discretize denoising diffusion ODEs Year: (2025)
Ref_id:b31 Title: S4S: Solving for a diffusion model solver Year: (2025)
Ref_id:b32 Title: Consistency models Year: (2023)
Ref_id:b33 Title: How to build a consistency model: Learning flow maps via self-distillation Year: (2025)
Ref_id:b34 Title: Multistep consistency models Year: (2024)
Ref_id:b35 Title: On distillation of guided diffusion models Year: (2023)
Ref_id:b36 Title: Simple and fast distillation of diffusion models Year: (2024)
Ref_id:b37 Title: Progressive distillation for fast sampling of diffusion models Year: (2022)
Ref_id:b38 Title: EM distillation for one-step diffusion models Year: (2024)
Ref_id:b39 Title: Distilling masked diffusion models into one-step generator Year: (2025)
Ref_id:b40 Title: Score-based continuous-time discrete diffusion models Year: (2023)
Ref_id:b41 Title: Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design Year: (2024)
Ref_id:b42 Title: Hamming distance metric learning Year: (2012)
Ref_id:b43 Title: OpenWebText Corpus Year: (2019)
Ref_id:b44 Title: Fast solvers for discrete diffusion models: Theory and applications of high-order algorithms Year: (2025)
Ref_id:b45 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b46 Title: Informed correctors for discrete diffusion models Year: (2024)
Ref_id:b47 Title: How discrete and continuous diffusion meet: Comprehensive analysis of discrete diffusion models via a stochastic integral framework Year: (2025)
Ref_id:b48 Title: Convergence analysis of discrete diffusion model: Exact implementation through uniformization Year: (2025)
Ref_id:b49 Title: Convergence of score-based discrete diffusion models: A discrete-time analysis Year: (2025)
Ref_id:b50 Title: Theoretical benefit and limitation of diffusion language model Year: (2025)
Ref_id:b51 Title: A convergence theory for diffusion language models: An information-theoretic perspective Year: (2025)
Ref_id:b52 Title: The cosine schedule is Fisher-Rao-optimal for masked discrete diffusion models Year: (2025)
Ref_id:b53 Title: Target concrete score matching: A holistic framework for discrete diffusion Year: (2025)
Ref_id:b54 Title: Absorb and converge: Provable convergence guarantee for absorbing discrete diffusion models Year: (2025)
Ref_id:b55 Title: Convergence of score-based discrete diffusion models: A discrete-time analysis Year: (2025)
Ref_id:b56 Title: Masked generative image transformer Year: (2022)
Ref_id:b57 Title: Halton scheduler for masked generative image transformer Year: (2025)
Ref_id:b58 Title: Beyond autoregression: Fast LLMs via self-distillation through time Year: (2025)
Ref_id:b59 Title: Remasking discrete diffusion models with inference-time scaling Year: (2025)
Ref_id:b60 Title: Simple and effective masked diffusion language models Year: (2024)
Ref_id:b61 Title: The Llama 3 herd of models Year: (2024)
Ref_id:b62 Title: Fast-dLLM: Training-free acceleration of diffusion LLM by enabling KV cache and parallel decoding Year: (2025)
Ref_id:b63 Title: Scaling diffusion language models via adaptation from autoregressive models Year: (2025)
Ref_id:b64 Title: FAIRSEQ: A fast, extensible toolkit for sequence modeling Year: (2019)
Ref_id:b65 Title: Principal components analysis Year: (1993)
Ref_id:b66 Title: The informativeness of k-means for learning mixture models Year: (2019)
