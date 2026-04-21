Title: SAFE: Finding Sparse and Flat Minima to Improve Pruning
Abstract: Sparsifying neural networks often suffers from seemingly inevitable performance degradation, and it remains challenging to restore the original performance despite much recent progress. Motivated by recent studies in robust optimization, we aim to tackle this problem by finding subnetworks that are both sparse and flat at the same time. Specifically, we formulate pruning as a sparsity-constrained optimization problem where flatness is encouraged as an objective. We solve it explicitly via an augmented Lagrange dual approach and extend it further by proposing a generalized projection operation, resulting in novel pruning methods called SAFE and its extension, SAFE + . Extensive evaluations on standard image classification and language modeling tasks reveal that SAFE consistently yields sparse networks with improved generalization performance, which compares competitively to well-established baselines. In addition, SAFE demonstrates resilience to noisy data, making it well-suited for real-world conditions.

Section: Introduction
Over the past decades, the emergence of computers and the accumulation of digital data have made machine learning a viable tool for everyday applications. However, modern machine learning systems have also grown in complexity with the rapid advances in hardware and databases, demanding significant computational and memory resources. This has led to a surge of interest in strategies to reduce these substantial computational costs.
One major approach is sparsification, which aims to find solutions with mostly zero entries. This has been studied for many years in various large-scale applications in machine learning (LeCun et al., 1989;Hassibi & Stork, 1992), signal 1 POSTECH, South Korea. Correspondence to: Dongyeop Lee <dylee23@postech.ac.kr>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
processing (Blumensath & Davies, 2009), and statistics (Tibshirani, 1996;Beck & Teboulle, 2009). Notably, the recent success of highly overparameterized deep neural networks in achieving human-level computer vision and natural language processing tasks has drastically scaled these models to proportions never seen before, which has spurred considerable research on sparsifying neural networks (Hoefler et al., 2021). Since the pioneering work of Han et al. (2015), many works have proposed to remove redundant parameters by various tactics ranging from those discussed in the survey work of Hoefler et al. (2021) to their applications to large foundation models (Kwon et al., 2022;Frantar & Alistarh, 2023;Sun et al., 2024). However, it is witnessed that excessive pruning usually results in performance decline due to the reduced capacity, potentially impairing deep learning models from handling tasks of high complexity. Is there a way to restore this performance loss?
We attend to recent studies that have demonstrated that the generalization performance of a model is closely linked to the flatness of the solution landscape (Keskar et al., 2017;Jiang et al., 2020a). This insight has led to the development of techniques such as sharpness-aware minimization (SAM) (Foret et al., 2021), which explicitly regularizes the sharpness of the solution during the optimization process. SAM has been shown to deliver exceptional performance across various domains (Chen et al., 2022;Bahri et al., 2022) and has also demonstrated robustness to label noise (Baek et al., 2024). In light of the SAM's success, there has been growing interest in applying these techniques to model pruning. Research by Na et al. (2022) has explored how sharpnessaware training affects model compressibility, while Peste et al. (2022) and Bair et al. (2023) have examined different strategies to alter this process for pruning, with the goal of enhancing performance and robustness. Nevertheless, the exploration of sharpness-aware techniques within the context of sparsification is still at an early stage. We believe there is considerable potential to integrate these approaches more effectively into the sparsification process, which could lead to significant advancements in the development of efficient and robust deep learning models.
In this work, we aim to achieve exactly that, by proposing to frame it as a sharpness-aware sparsity-constrained optimization problem to simultaneously consider both sharpness and sparsity during the training process. To tackle this, we use an augmented Lagrangian dual-based approach well established in the optimization literature with convergence guarantees, and propose a new optimization-based pruning method called SAFE. We evaluate SAFE across standard benchmark tasks in image classification and large language model post-training pruning, and compare with existing alternatives to demonstrate that (i) it induces flatness on the sparse solutions, (ii) yielding performance improvements in the resulting sparse models, and (iii) its robustness to label noise, common image noise, and adversarial noise, and its (iv) effectiveness compared to similar sharpnessminimization-inspired pruning techniques. These aspects support the effectiveness and generality of our method, and we conclude by discussing current limitations and potential ideas for future work.
this section cite: ['b36', 'b18', 'b6', 'b63', 'b4', 'b23', 'b17', 'b23', 'b35', 'b15', 'b60', 'b29', 'b12', 'b10', 'b2', 'b1', 'b46', 'b52', 'b3']

Section: Background

this section cite: []

Section: Sparsity
Throughout the years, various techniques to induce sparsity in machine learning systems have been developed to simplify models for efficiency, interpretability, and generalization. This is usually framed as solving the following optimization problem with sparsity constraint:
min ∥x∥0≤d f (x),(1)
where f is the objective we wish to minimize, x is the optimization variable, ∥x∥ 0 denotes the L 0 -norm, which counts the non-zero entries within x, and d is the number of parameters we wish to preserve. The goal is to find a solution x with desired sparsity that minimizes f . Exactly solving this optimization problem is challenging due to the combinatorial nature of the L 0 -norm, as it requires an exhaustive search over all possible configurations of zeros within x. Consequently, several approaches have been proposed, such as relaxing the L 0 norm as in LASSO (Tibshirani, 1996), or employing advanced optimization techniques, including proximal methods like FISTA (Beck & Teboulle, 2009) and iterative hard thresholding (Blumensath & Davies, 2009). Additionally, strategies like optimal brain damage and surgeon (LeCun et al., 1989;Hassibi & Stork, 1992) have been explored to sparsify multi-layer perceptrons through second-order approximations of the objective function.
The recent success of increasingly large deep neural networks has further accelerated this trend, spurring the development of various methods to sparsify neural networks at different stages of training, each offering distinct advantages depending on the scenario (Hoefler et al., 2021). For instance, pruning before training (Lee et al., 2019;Tanaka et al., 2020;Wang et al., 2020) is advantageous for improving computational efficiency during training by enabling sparse training, while post-training pruning (Frantar & Al-istarh, 2023;Sun et al., 2024;Kwon et al., 2022) is ideal for enhancing inference efficiency in pre-trained models, particularly when the training process can be too costly due to large-scale data or complex models such as large language models (LLM). A widely adopted strategy in posttraining pruning is layer-wise reconstruction error minimization, which ensures that the pruned model maintains accuracy by preserving layer-wise output approximations (Frantar & Alistarh, 2023;Sun et al., 2024;Meng et al., 2024). This approach enables efficient pruning of large models by solving smaller subproblems independently for each layer, reducing computational overhead while preserving the signal in model representations. Optimization-based techniques have been proposed to refine this idea further, improving sparsity while minimizing performance degradation. Additionally, extensions of these methods explore structured sparsity patterns, such as block-wise sparsity, to enhance hardware efficiency while maintaining competitive accuracy. Also, recent studies hint at the possibility of finding an initial random sparse network that can be trained to achieve comparable performance to dense networks, although this generally involves several rounds of expensive training (Frankle & Carbin, 2019). Among various approaches, this work primarily focuses on pruning during training (Peste et al., 2021;Zhou et al., 2021;Kusupati et al., 2020;Lin et al., 2020;Sanh et al., 2020;Evci et al., 2020), which is known for achieving best results by guiding the model toward desired sparsity during training (Hoefler et al., 2021), and can be ideal for moderately sized models with adequate computational resources for training. Despite this, preserving the original dense performance remains challenging due to the complexity of tasks handled by deep learning models, often leading to the use of heuristics to manage these difficulties.
this section cite: ['b63', 'b4', 'b6', 'b36', 'b18', 'b23', 'b37', 'b62', 'b65', 'b60', 'b35', 'b15', 'b60', 'b43', 'b13', 'b51', 'b73', 'b34', 'b40', 'b55', 'b11', 'b23']

Section: Flat Minima
In-depth empirical analyses into the optimization properties of deep neural network training, especially with regards to mini-batch training, have revealed a surprising correlation between well-generalizing solutions and their flatness (Keskar et al., 2017;Jiang et al., 2020b). This finding has prompted numerous studies aiming to understand the precise nature of this relationship (Andriushchenko et al., 2023;Neyshabur et al., 2017;Zhou et al., 2020), and its impact on neural network pruning, as explored by Lee et al. (2021), who link the challenge of training highly sparse networks to sharper loss landscapes based on an analysis of scaling properties under varying sizes of mini-batches and classical optimization theory.
This also motivated researchers to develop various techniques to explicitly induce flat minima during training (Foret et al., 2021;Izmailov et al., 2018;Orvieto et al., 2022;Chaudhari et al., 2017). Among them, sharpness-aware min-imization (SAM) by Foret et al. (2021) aims to tackle this by solving the following min-max optimization problem:
min x max ∥ϵ∥2≤ρ f (x + ϵ),(2)
where we minimize the objective function over the entire ϵneighborhood with radius ρ, i.e., seek flat minima. Solving the inner maximization problem for the first-order Taylor approximation gives the following update rule for SAM:
x t+1 = x t -η∇f x t + ρ ∇f (x t ) ∥∇f (x t )∥ 2 .
This has been shown to be effective in improving generalization performance and robustness across various domains (Chen et al., 2022;Bahri et al., 2022;Baek et al., 2024).
The success of sharpness minimization techniques has naturally led to exploring their implications for neural network pruning. Na et al. (2022) studied whether a flatter loss landscape can be more compressible by employing SAM during iterative magnitude pruning for fine-tuning BERT models. Shin et al. (2025) observed that the generalization benefits of SAM can be leveraged with sparsity for overparameterized neural networks. Inspired by SAM, Peste et al. (2022) proposed compression-aware minimization (CrAM) to minimize the loss increase induced by perturbation from compression, which aims to induce robustness to post-training one-shot pruning of any sparsity. Bair et al. (2023) suggested performing additional sharpness-aware training to pre-trained models, with larger perturbations given to coordinates of low importance score (i.e., parameters to be pruned) to incur less loss increase when pruning.
While these efforts represent initial attempts to verify the effectiveness of sharpness minimization or its loosely inspired variants in enhancing pruning, we believe that sharpness minimization can be more effectively integrated into the sparsification process. Thus, in this work, we aim to weave this sharpness-minimization objective with the sparsification process to enhance the quality of the sparsified network via principled optimization-based approaches that are well established in the literature.
this section cite: ['b29', 'b0', 'b48', 'b72', 'b38', 'b12', 'b26', 'b49', 'b9', 'b12', 'b10', 'b2', 'b1', 'b46', 'b57', 'b52', 'b3']

Section: Method
In this section, we present a detailed derivation of our flatness-inducing sparsification algorithm Sparsification via ADMM with Flatness Enforcement or SAFE.
this section cite: []

Section: Problem Formulation
We begin by proposing the following min-max optimization problem with sparsity constraint:
min ∥x∥0≤d max ∥ϵ∥2≤ρ f (x + ϵ),(3)
where f is the objective function to minimize, d is the number of parameters to preserve, and ρ is the radius of the perturbation ϵ. Thus, the goal is to find a sparse solution x ⋆ that minimizes the objective function in the whole ϵneighborhood, i.e., seek flat minima.
this section cite: []

Section: Augmented Lagrangian Based Approach
A standard approach for solving such constrained optimization problems is to employ Lagrangian duality or projected gradient descent. However, the discrete nature of L 0 -norm makes Lagrangian duality infeasible, while projected gradient descent, despite its computational efficiency for L 0 constraints, can struggle with highly non-convex objectives in neural network optimization. To leverage the smooth optimization of Lagrangian and the efficiency of projection, we leverage augmented Lagrangian as described below.
To achieve this, we first employ variable splitting, a widely used trick, usually to separately deal with objective minimization and constraint satisfaction (Boyd et al., 2011). Precisely, instead of directly imposing the sparsity constraint on variable x, we first split it into variables x and z as follows:
min x,z max ∥ϵ∥2≤ρ f (x + ϵ) + I ∥•∥0≤d (z) s.t. x = z,
where I ∥•∥0≤d (z) is an indicator function for the sparsity constraint:
I ∥•∥0≤d (z) := 0 if ∥z∥ 0 ≤ d ∞ else.
We then slightly alter the Lagrangian by adding a penalty term λ/2∥x-z∥ 2 2 with penalty parameter λ, which preserves equivalence to the original problem while also acting as a proximal term for the projection step. This alteration is a form of augmented Lagrangian, which we apply to form the Lagrangian dual problem of the following:
max u , min x,z L(x, z, u) := max ∥ϵ∥2≤ρ f (x + ϵ) + I ∥•∥0≤d (z) - λ 2 ∥u∥ 2 2 + λ 2 ∥x -z + u∥ 2 2 ,
where u is a scaled dual variable for the equality constraint scaled by 1/λ. Here, the projection can be computed efficiently via hard thresholding operation (Blumensath & Davies, 2009), which sets all entries of x but the d elements with the largest magnitudes to zero. Applying dual ascent leaves us with the following x, z-minimization and u-maximization:
x k+1 , z k+1 = argmin x,z max ∥ϵ∥2≤ρ f (x + ϵ) + I ∥•∥0≤d (z) + λ 2 ∥x -z + u k ∥ 2 2 u k+1 = argmax u λ 2 ∥x k+1 -z k+1 + u∥ 2 2 - λ 2 ∥u∥ 2 2 .
To minimize each x and z separately with iterative first-order optimization and exact projection operation respectively, we compute x and z in an alternating manner which gives the following iteration:
x k+1 = argmin x max ∥ϵ∥2≤ρ f (x + ϵ) + λ 2 ∥x -z k + u k ∥ 2 2 z k+1 = proj ∥•∥0≤d (x k+1 + u k ) u k+1 = u k + x k+1 -z k+1 ,
where proj ∥•∥0≤d is a projection operation onto the sparsity constraint (i.e., the hard thresholding operator), and u-maximization is solved through applying a single step of gradient ascent with a step size of λ on y, which is to ensure that the iterate stays within feasibility once reached.
this section cite: ['b7', 'b6']

Section: x-minimization
For the x-minimization step, we first approximately solve the ϵ-maximization via first-order approximation of f :
ϵ ⋆ (x) ≈ argmax ∥ϵ∥2≤ρ f (x) + ϵ ⊤ ∇f (x) = ρ ∇f (x) ∥∇f (x)∥ 2 ,
which we apply back to the objective
x k+1 = argmin x f (x + ϵ ⋆ (x)) + λ 2 ∥x -z k + u k ∥ 2 2 .
We solve this using gradient descent, where we remove higher-order terms in the gradient as in Foret et al. (2021),
∇ x f (x + ϵ ⋆ (x)) + λ 2 ∥x -z k + u k ∥ 2 2 = (I + ∇ϵ ⋆ (x))∇f (x) x+ϵ ⋆ (x) + λ(x -z k + u k ) = ∇f x + ρ ∇f (x) ∥∇f (x)∥ 2 + λ(x -z k + u k ),(4)
thus leading to the following x-minimization steps:
x (t+1) k = x (t) k -η (t) ∇f x (t) k + ρ ∇f (x (t) k ) ∥∇f (x (t) k )∥ 2 + λ(x (t) k -z k + u k ) ,(5)
where t, η (t) are the current step of x-minimization and its step-size, respectively.
this section cite: ['b12']

Section: Extension to Generalized Projection
While the Euclidean projection onto an L 0 constraint naturally yields magnitude-based sparsification, this often yields subpar performance in practice compared to more advanced saliency scores that account for the objective function. To naturally integrate these into the projection operation, we design a generalized distance metric that introduces a positivedefinite diagonal matrix P that provides a framework to incorporate these advanced saliencies of the form P
1/2 [i,i] |x [i] | in a principled manner: z k+1 = proj P ∥•∥0≤d (x k+1 + u k ) := arg min ∥z∥0≤d 1 2 ∥z -(x k+1 + u k )∥ 2 P = arg min ∥z∥0≤d 1 2 (z -(x k+1 + u k )) ⊤ P(z -(x k+1 + u k ))
Geometrically, this can be understood as modifying the underlying distance metric to better represent the local geometric structure (e.g., the Hessian) of the original objective function. We call this SAFE + , where we leverage various saliency scores within the projection step, which we describe in detail below.
We lay out some notable examples of advanced saliency scores and the corresponding P. The simplest case is P = I, where the projection reduces to standard hard thresholding as it corresponds to the Euclidean norm, yielding the original SAFE. Taking this further, setting it as the diagonal Hessian, i.e., P = diag(∇ 2 f (x)), corresponds to Optimal Brain Damage (LeCun et al., 1989), a second-order pruning method that aims to remove parameters with minimal impact on the loss function. Also, using P = diag(∇f (x)∇f (x) ⊤ ) aligns with the first-order pruning method SNIP (Lee et al., 2019), which removes parameters based on gradient sensitivity. Furthermore, Wanda (Sun et al., 2024), a layer-wise pruning method for language models, corresponds to taking P = diag(A ⊤ A) where A ∈ R N ×d is an activation with batch size N and feature dimension d from a particular layer to prune. This corresponds to the diagonal Hessian of the reconstruction error for a single linear layer (Sun et al., 2024).
This generalized projection allows SAFE + to integrate diverse sparsification strategies all within its constrained optimization framework, thus enhancing both effectiveness and robustness in model pruning. Our empirical evaluation in Section 4.3 demonstrates its effectiveness in large language model pruning, though the methodology is not confined to this domain and is widely applicable.
this section cite: ['b36', 'b37', 'b60', 'b60']

Section: Final Algorithm: SAFE and SAFE +
Our final algorithm is summarized in Algorithm 1. We provide an intuitive description of how our algorithm performs sparsification. Every few steps of x-minimization, SAFE observes where the closest point on the sparsity constraint from the current x is and registers it on z. While performing flatness-inducing minimization of the objective function on x, it penalizes the x iterate to slightly move closer to z, the Algorithm 1 SAFE and SAFE + algorithms Require: Target parameter count d, total train iteration T , dual-update interval K, learning rate η (t) , perturbation radius ρ, penalty parameter λ, importance matrix P.
1: Initialize x (0) 2: u = 0 3: for t in T do 4: if t mod K = 0 then 5: if SAFE then 6: z = proj ∥•∥0≤d (x (t+1) + u) 7: else if SAFE + then 8: z = proj P ∥•∥0≤d (x (t+1) + u) 9:
end if 10:
u = u + x (t+1) -z 11:
end if 12:
x (t+1/2) = x (t) -η (t) ∇f x (t) + ρ • ∇f (x (t) ) ∥∇f (x (t) )∥2
13:
x (t+1) = x (t+1/2) -η (t) λ(x (t) -z + u) 14: end for 15: return proj ∥•∥0≤d (x (T ) )
latest estimate of the sparse solution. This gradually moves the dynamics of x towards sparsity without incurring a sudden change of loss, all while performing flatness induction, yielding a sparse and flat minima.
In practice, particularly for image classification, we introduce scheduling to the penalty parameter λ from zero to the target value in a cosine curve in order to apply less restriction in the initial phases of training, which slightly improves performance. Details of the ablation study on this scheduling strategy can be found in Appendix F.3.
this section cite: []

Section: Convergence Analysis
Here we present a convergence analysis of SAFE. Precisely, we first prove that our proposed iterative sharpness minimization in the x-update converges, then build the rest of the proof upon a well-studied result of ADMM (Boyd et al., 2011;Wang et al., 2019;Huang et al., 2021).
We start with standard assumptions used in the literature: We also define the following notion of stationarity for the optimization problem (1) from Huang et al. (2021):
Definition 3.4. (δ-stationary point) We say a point x is a δ-stationary point of the optimization problem (1) if x ∈ arg min a∈A a -x -δ -1 ∇f (x) , i.e., the point x cannot be locally improved using projected gradient descent with step-size δ -1 . With this definition, we ultimately demonstrate that SAFE converges to this δstationary point, which is a necessary condition for the optimal solution to problem (1).
We first provide the central lemma on the convergence of our sharpness minimizing x iterates: Lemma 3.5. (Convergence of x-minimization) Suppose that Assumptions 3.1 and 3.2 hold and let {x (t) k } be generated by Equation (5) in Algorithm 1 with step-size η (t) and perturbation radius
ρ (t) satisfying ∞ t=1 η (t) = ∞, ∞ t=1 η (t) ρ (t) < ∞, lim sup t ρ (t) < 1/β. Let L(x) = f (x) + λ 2 ∥x -z + u∥ 2 2 and assume that inf x∈N L(x (t) ) > -∞. Then ∇ L(x (t) ) → 0.
The detailed proof is provided in Appendix A.1. This shows that running Equation ( 5) produces a sequence that converges to the stationary point of the augmented Lagrangian L with respect to x.
We use this to derive the convergence of SAFE as the following corollary: Corollary 3.6. (Convergence of SAFE) Suppose that Assumptions 3.1-3.3 hold. Assume further that δ is chosen large enough so that δ -1 β 2 -(δ -µ)/2 < 0. Let (x, z, ū) be a limit point of SAFE algorithm. Then x is a δ-stationary point of the optimization problem (1).
This demonstrates that SAFE converges to the stationary point of the sparsity-constrained optimization problem (1). We note that, while the technical contributions of our analysis might be considered modest, SAFE is built on a theoretically rigorous foundation, unlike many other pruning techniques that often rely primarily on ad-hoc intuitions.
this section cite: ['b7', 'b66', 'b24', 'b24']

Section: Experiments
In this section, we demonstrate that SAFE converges to sparse and flat solutions, leading to performance improvements over baselines in both image classification and language modeling tasks. We also show that SAFE is robust to noisy label training and corruptions during inference. The codes to reproduce the results are provided in JAX and PyTorch, with further details provided in Appendix B.5.
this section cite: []

Section: Convergence to Sparse and Flat Solutions
We first show that SAFE successfully guides training towards sparse and flat solutions compared to naive baselines on a simple neural network model. Specifically, we analyze the  weight distributions of models trained with standard dense training and SAFE to assess its sparsification capability. We also measure sharpnesses of SAFE and compare it to that of ADMM (Zhang et al., 2018) as a non-sharpness-minimizing baseline, by computing maximum Hessian eigenvalues and visualizing loss landscapes. The results are presented in Figure 1. Our findings indicate that SAFE effectively induces sparsity while simultaneously enforcing flatness, as evidenced by the concentration of weights near zero in contrast to dense training and a wider minimum with a lower Hessian eigenvalue compared to ADMM. This result demonstrates the effectiveness of SAFE in tackling the sharpness-aware sparsity-constrained optimization problem (3). Further experimental details are provided in Appendix B.2.
this section cite: ['b70']

Section: Evaluations on Image Classification
In this section, we show that SAFE can achieve outstanding generalization performance among various methods for CIFAR-10/100 image classification tasks (Krizhevsky et al., 2009). Specifically, we evaluate pruning performance on VGG-19 (Simonyan, 2014) and ResNet-20/32foot_0 (He et al., 2016) using a range of representative pruning methods, including PBW (Han et al., 2015), GMP (Kurtic & Alistarh, 2022;Zhu & Gupta, 2017), LTH (Liu et al., 2024;Frankle & Carbin, 2019), ADMM (Zhang et al., 2018), and MLPrune (Zeng & Urtasun, 2018), some of which achieves competitive to state-of-the-art performance in image classification (Hoefler et al., 2021). We mostly use standard values for common hyperparameters such as training epochs, learning rate, and weight decay (Zhou et al., 2021) and tune the hyperparameters unique to SAFE, which we report in detail in Appendix B. Notably, we do not perform additional training after pruning, and instead perform a cost-efficient statistical correction on the batch-norm layers with only a few forward passes (batch-norm tuning or BNT), which is a common practice in the literature (Hubara et al., 2021;Frantar & Alistarh, 2022;Peste et al., 2022). We refer to Appendix B.3 for full experimental details. The final validation accuracies are provided in Figure 2 and Table 7 of Appendix C.
Our findings show that across most configurations and sparsity levels, SAFE generally outperforms all baselines. Also, SAFE exhibits greater robustness under extreme sparsity compared to non-sharpness-minimized approaches (e.g., 99.5%). Crucially, SAFE achieves these results without requiring costly retraining, whereas PBW and LTH depend on multiple rounds of retraining. These results highlight the effectiveness of SAFE in preserving model accuracy during sparsification, especially under aggressive pruning scenarios.
this section cite: ['b32', 'b58', 'b19', 'b17', 'b74', 'b41', 'b13', 'b70', 'b69', 'b23', 'b73', 'b25', 'b14', 'b52']

Section: Evaluation on Large Language Model Pruning
Here we scale our evaluations to modern large-scale settings and demonstrate that SAFE also delivers competitive performance against state-of-the-art LLM post-training pruning techniques.
For this purpose, we adapt SAFE and SAFE + to sequentially optimize the reconstruction error minimization (REM) objective for each transformer block (Shin et al., 2024), similarly to other LLM pruning techniques. For SAFE + , we incorporate Wanda projection z-step, which identifies superior subnetworks compared to naive magnitude-based pruning in LLMs without compromising efficiency (Sun et al., 2024).
With this, we prune one of the most widely adopted language model family, LLaMA2-7b/13b (Touvron et al., 2023) and the more recent LLaMA3-8b (Meta, 2024), to 50% and 60% sparsities, as well as structured 4:8 and 2:4 sparsities. We compare SAFE with state-of-the-art LLM post-training pruning methods such as SparseGPT (Frantar & Alistarh, 2023), Wanda (Sun et al., 2024), ALPS (ADMM-based) (Meng et al., 2024), as well as magnitude pruning (Han et al., 2015) and evaluate the perplexity on Wikitext2 (Merity et al., 2022) and C4 validation sets. We follow the common practice of randomly selecting 128 samples from the C4 training dataset (Raffel et al., 2020). We refer to Appendix B.4 for experimental details. The results are reported in Table 1.
We find that SAFE performs competitively to state-of-the-art methods, while SAFE + surpasses them across all models and sparsity settings. Considering that these baselines are tailored specifically to pruning LLMs, this demonstrates the flexibility of SAFE in adapting to different scenarios. Moreover, our method is more efficient than ALPS, which requires ×2.54 more runtime than SAFE (see Appendix E.2).
this section cite: ['b56', 'b60', 'b64', 'b15', 'b60', 'b43', 'b17', 'b53']

Section: Robustness to Noisy Data
Noisy data pose significant challenges in real-world scenarios. To address this, we evaluate SAFE on three representative challenges: incorrect training labels (Song et al., 2022), inference-time input corruption that arises naturally (Hendrycks & Dietterich, 2019), and corruptions that are de- liberately introduced by adversaries (Szegedy et al., 2014).
Training on noisy labels To assess the robustness of SAFE against label noise, we randomly corrupt {25%, 50%, 75%} of labels in CIFAR-10 and use it to train ResNet-20 with both SAFE and ADMM. The same hyperparameters from Section 4.2 are used in all experiments. As presented in Table 2, we observe that SAFE consistently outperforms ADMM across all levels of label noise and sparsity, with accuracy improvements ranging from +10% to +30%.
Additionally, we observe that ADMM relies heavily on sparsity to mitigate label noise, exhibiting an overall trend of increasing accuracy with higher sparsity levels. This dependence is further reflected in the sparse double descent pattern reported at the 25% noise ratio (He et al., 2022), where accuracy initially declines up to 80% sparsity, rises sharply to 79% at 90% sparsity, and then drops again to 77% at 95% sparsity. This contributes to the overall decreasing performance gap between ADMM and SAFE, which may be interpreted as the benefit of sharpness-minimization diminishing with fewer parameters (Shin et al., 2025). However, more crucially, the overall under-performance of ADMM indicates that sparsity alone is a poor remedy for label noise, highlighting the effectiveness of SAFE-particularly its flatness enforcement-in reducing the impact of label noise. This aligns well with previous observations that sharpnessminimization can enhance robustness toward label noise (Baek et al., 2024). Also, the lack of double descent in SAFE suggests that its effectiveness may be attributed to sharpness minimization functioning as an effective regularizer, as supported by the claims of Nakkiran et al. (2021) that 'optimal' regularization can mitigate the double descent phenomenon.
this section cite: ['b59', 'b22', 'b61', 'b20', 'b57', 'b1', 'b47']

Section: Evaluation on corrupted image
We evaluate the sparse models trained with ADMM and SAFE, as obtained in Section 4.2, on the CIFAR-10 test set with common image corruptions and adversarial perturbations. Specifically, for
Table 3: Evaluation on corrupted data. CIFAR-10C is used for common corruptions, and l ∞ and l 2 PGD attacks are used to generate adversarial corruption on the validation set of CIFAR-10. SAFE improves robustness over naturally and adversarially corrupted images.
Common corruption (avg.) Adversarial Sparsity Method intensity=3 intensity=5 l ∞ -PGD l 2 -PGD 90% ADMM 70.06 ±0.03 52.01 ±0.38 49.81 ±1.02 49.71 ±1.06 SAFE 73.98 ±0.09 55.11 ±0.27 56.43 ±1.03 56.36 ±1.11 95% ADMM 68.87 ±0.25 50.56 ±0.07 49.84 ±1.78 49.68 ±1.79 SAFE 72.92 ±0.41 54.86 ±0.51 51.40 ±0.89 51.36 ±0.94 98% ADMM 65.46 ±0.24 48.65 ±0.04 43.33 ±1.59 43.42 ±1.60 SAFE 68.20 ±0.47 49.96 ±0.83 43.34 ±0.90 43.41 ±1.03 99% ADMM 59.21 ±0.47 43.81 ±0.44 30.29 ±0.64 30.32 ±0.58 SAFE 66.02 ±0.56 49.34 ±1.03 43.70 ±1.28 32.70 ±1.28 99.5% ADMM 55.72 ±0.44 41.55 ±0.78 23.25 ±1.92 23.25 ±1.85 SAFE 56.58 ±0.36 42.27 ±0.63 29.48 ±0.68 29.45 ±0.74 common corruptions, we use CIFAR-10C (Hendrycks & Dietterich, 2019), a benchmark consisting of CIFAR-10 test images corrupted with 19 types of real-world noise (e.g., fog, snow, etc.) and distortions (e.g., jpeg compression, contrast, etc.) at five levels of intensity. We average the performance across all corruption types for intensity levels 3 and 5. For adversarial noise, we follow Zhang et al. (2024) and use a 10-step Projected Gradient Descent (PGD) attack on each sparse model under l ∞ and l 2 norm with bound ϵ = 1/255 and 3/255 and step size α = ϵ/4, respectively. As shown in Table 3, SAFE enhances robustness to both common and adversarial image corruptions, aligning with previous work on sharpness minimization (Zhang et al., 2024;Wei et al., 2023).
this section cite: ['b71', 'b67']

Section: Comparison with Other SAM-based pruner
To strengthen the comparison with closely related baselines, we compare SAFE to two pruning baselines inspired by SAM-IMP+SAM (Na et al., 2022) and CrAM (Peste et al., 2022)-on ResNet-20/CIFAR-10 across multiple sparsity levels.
IMP+SAM (Na et al., 2022) involves applying SAM during iterative magnitude pruning (Liu et al., 2024;Frankle & Carbin, 2019). While it was initially introduced for language model finetuning, we adapt this method to image classification and use the same training epochs and sharpnessminimization hyperparameter search range as SAFE to ensure fair comparison. Pruning is performed every 10 epochs with sparsity increasing either linearly, following Na et al. (2022), or cubically (Zhu & Gupta, 2017), with the latter yielding better performance.
Compression-Aware Minimizer (CrAM) (Peste et al., 2022), on the other hand, extends upon the robust optimization principles of SAM to train compressible models by en- couraging the models to maintain strong post-compression performance under the presence of small perturbations as min x max ∥ϵ∥≤ρ f (C(x + ϵ)) given some compression operation C. This leads to the CrAM update rule x t+1 = x t -η∇f (C(x + ρ∇f (x))). Along with this, we additionally compare with CrAM + , a variant introduced by Peste et al. (2022) that simply adds the original gradient ∇f (x) in the update as
x t+1 = x t -η [∇f (C(x + ρ∇f (x))) + ∇f (x))].
We run these using the optimal hyperparameters as suggested in Peste et al. (2022). It should be noted that the additional technique of CrAM + is somewhat auxiliary to the core robust optimization mechanism that connects CrAM to SAFE, and thus, care must be taken when associating their performance gains with the central strategy that defines CrAM. For better comparison, we extend this strategy to SAFE by adding the gradient computed at projected point ∇f (C(x)) during iterative x-minimization of SAFE, which we denote as SAFE +SG .
As shown in Table 4, SAFE and SAFE +SG outperforms IMP+SAM and CrAM, where SAFE +SG demonstrates competitive performance to CrAM + on moderate sparsity and outperforms it in extreme sparsity. We suspect that the pruning operation in IMP+SAM may be overly abrupt, potentially hindering the benefits of sharpness minimization. We additionally observe similarly strong performance over IMP+SAM on language model pruning in Appendix D.2. Conversely, although CrAM + achieves competitive results, its benefits fails to extend to extreme sparsity. More crucially, while SAFE achieves reliable performance without much additional techniques, CrAM, by itself, performs poorly in all sparsities, depending heavily on auxiliary techniques to drastically improve performance. This suggests that caution is warranted when attributing these gains to the effectiveness of their robust optimization formulation inspired by sharpness-minimization, which is more likely provided through the use of both the original dense gradient and the sparse gradient computed at the projected point from CrAM + . A similar trend is observed in SAFE +SG , further supporting this interpretation. SAFE, on the other hand, delivers competitive performance without relying on these additional techniques, highlighting the intrinsic effectiveness of its smooth penalization via the augmented Lagrangian and split-variable structure of the ADMM framework to jointly balance sharpness minimization and sparsity.
We provide additional comparison with other variants of CrAM in Appendix D.1.
this section cite: ['b46', 'b52', 'b46', 'b41', 'b13', 'b46', 'b74', 'b52', 'b52', 'b52']

Section: Conclusion
In this work, we propose an effective and principled approach called SAFE to obtain sparse and flat solutions by solving a constrained optimization problem based on the augmented Lagrangian, which we further extend to SAFE + by proposing a generalization for the projection operation. We show that SAFE can be applied to neural network pruning, and as a result, it not only obtains the desired flatness as well as high sparsity in the given deep model, but also enhances its generalization performance quite significantly, far better than the compared baselines, as validated across standard benchmarks. Interestingly, SAFE preserves its robustness to various data noise during both training and inference, which stems from the original sharpness minimization strategy. Finally, we compare with more directly related SAM-inspired baselines, demonstrating the intrinsic effectiveness of SAFE without much reliance on auxiliary techniques. We believe that this principled approach for obtaining sparse and flat solutions-concepts that have often been explored rather separately in the literature-offers significant potential.
this section cite: []

Section: References
Ref_id:b0 Title: A modern look at the relationship between sharpness and generalization Year: (2023)
Ref_id:b1 Title: Why is sam robust to label noise? Year: (2024)
Ref_id:b2 Title: Sharpness-aware minimization improves language model generalization Year: (2022)
Ref_id:b3 Title: Adaptive sharpness-aware pruning for robust sparse networks Year: (2023)
Ref_id:b4 Title: A fast iterative shrinkagethresholding algorithm for linear inverse problems Year: (2009)
Ref_id:b5 Title: Fast as chita: Neural network pruning with combinatorial optimization Year: (2023)
Ref_id:b6 Title: Iterative hard thresholding for compressed sensing Year: (2009)
Ref_id:b7 Title: Distributed optimization and statistical learning via the alternating direction method of multipliers Year: (2011)
Ref_id:b8 Title:  Year: (2018)
Ref_id:b9 Title: Biasing gradient descent into wide valleys. ICLR Year: (2017)
Ref_id:b10 Title: When vision transformers outperform resnets without pre-training or strong data augmentations Year: (2022)
Ref_id:b11 Title: Rigging the lottery: Making all tickets winners Year: (2020)
Ref_id:b12 Title: Sharpness-aware minimization for efficiently improving generalization Year: (2021)
Ref_id:b13 Title: The lottery ticket hypothesis: Finding sparse, trainable neural networks Year: (2019)
Ref_id:b14 Title: Optimal brain compression: A framework for accurate post-training quantization and pruning Year: (2022)
Ref_id:b15 Title: Massive language models can be accurately pruned in one-shot Year: (2023)
Ref_id:b16 Title: Deep Learning Year: (2016)
Ref_id:b17 Title: Learning both weights and connections for efficient neural network Year: (2015)
Ref_id:b18 Title: Second order derivatives for network pruning: Optimal brain surgeon Year: (1992)
Ref_id:b19 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b20 Title: Sparse double descent: Where network pruning aggravates overfitting. ICML Year: (2022)
Ref_id:b21 Title: Flax: A neural network library and ecosystem for JAX Year: (2023)
Ref_id:b22 Title: Benchmarking neural network robustness to common corruptions and perturbations Year: (2019)
Ref_id:b23 Title: Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks Year: (2021)
Ref_id:b24 Title: Alternating direction method of multipliers for quantization Year: (2021)
Ref_id:b25 Title: Accelerated sparse neural training: A provable and efficient method to find n: m transposable masks Year: (2021)
Ref_id:b26 Title: Averaging weights leads to wider optima and better generalization Year: (2018)
Ref_id:b27 Title: Fantastic generalization measures and where to find them Year: (2020)
Ref_id:b28 Title: Fantastic generalization measures and where to find them Year: (2020)
Ref_id:b29 Title: On large-batch training for deep learning: Generalization gap and sharp minima Year: (2017)
Ref_id:b30 Title: Fundamental convergence analysis of sharpnessaware minimization Year: (2024)
Ref_id:b31 Title: A method for stochastic optimization Year: (2017)
Ref_id:b32 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b33 Title: Gmp*: Well-tuned gradual magnitude pruning can outperform most bert-pruning methods Year: (2022)
Ref_id:b34 Title: Soft threshold weight reparameterization for learnable sparsity Year: (2020)
Ref_id:b35 Title: A fast post-training pruning framework for transformers Year: (2022)
Ref_id:b36 Title:  Year: (1989)
Ref_id:b37 Title: Snip: Single-shot network pruning based on connection sensitivity Year: (2019)
Ref_id:b38 Title: Understanding the effects of data parallelism and sparsity on neural network training Year: (2021)
Ref_id:b39 Title: Visualizing the loss landscape of neural nets Year: (2018)
Ref_id:b40 Title: Dynamic model pruning with feedback Year: (2020)
Ref_id:b41 Title: A survey of lottery ticket hypothesis Year: (2024)
Ref_id:b42 Title: Rethinking the value of network pruning Year: (2019)
Ref_id:b43 Title: Alps: Improved optimization for highly sparse one-shot pruning for large language models Year: (2024)
Ref_id:b44 Title: Pointer sentinel mixture models Year: (2022)
Ref_id:b45 Title: The llama 3 herd of models. arXiv Year: (2024)
Ref_id:b46 Title: Train flat, then compress: Sharpness-aware minimization learns more compressible models Year: (2022)
Ref_id:b47 Title: Optimal regularization can mitigate double descent Year: (2021)
Ref_id:b48 Title: Exploring generalization in deep learning Year: (2017)
Ref_id:b49 Title: Anticorrelated noise injection for improved generalization Year: (2022)
Ref_id:b50 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b51 Title: Ac/dc: Alternating compressed/decompressed training of deep neural networks Year: (2021)
Ref_id:b52 Title: Cram: A compression-aware minimizer Year: (2022)
Ref_id:b53 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b54 Title: What's hidden in a randomly weighted neural network? Year: (2020)
Ref_id:b55 Title: Movement pruning: Adaptive sparsity by fine-tuning Year: (2020)
Ref_id:b56 Title: Rethinking pruning large language models: Benefits and pitfalls of reconstruction error minimization Year: (2024)
Ref_id:b57 Title: Critical influence of overparameterization on sharpness-aware minimization Year: (2025)
Ref_id:b58 Title: Very deep convolutional networks for largescale image recognition Year: (2014)
Ref_id:b59 Title: Learning from noisy labels with deep neural networks: A survey Year: (2022)
Ref_id:b60 Title: A simple and effective pruning approach for large language models Year: (2024)
Ref_id:b61 Title: Intriguing properties of neural networks Year: (2014)
Ref_id:b62 Title: Pruning neural networks without any data by iteratively conserving synaptic flow Year: (2020)
Ref_id:b63 Title: Regression shrinkage and selection via the lasso Year: (1996)
Ref_id:b64 Title: Llama 2: Open foundation and finetuned chat models Year: (2023)
Ref_id:b65 Title: Picking winning tickets before training by preserving gradient flow Year: (2020)
Ref_id:b66 Title: Global convergence of admm in nonconvex nonsmooth optimization Year: (2019)
Ref_id:b67 Title: Sharpness-aware minimization alone can improve adversarial robustness Year: (2023)
Ref_id:b68 Title:  Year: (2020)
Ref_id:b69 Title: Multi-layer pruning for automated neural network compression. arXiv Year: (2018)
Ref_id:b70 Title: A systematic dnn weight pruning framework using alternating direction method of multipliers Year: (2018)
Ref_id:b71 Title: On the duality between sharpness-aware minimization and adversarial training Year: (2024)
Ref_id:b72 Title: Towards theoretically understanding why sgd generalizes better than adam in deep learning Year: (2020)
Ref_id:b73 Title: Effective sparsification of neural networks with global sparsity constraint Year: (2021)
Ref_id:b74 Title: To prune, or not to prune: exploring the efficacy of pruning for model compression Year: (2017)
