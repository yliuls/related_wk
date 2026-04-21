Title: Efficient Source-free Unlearning via Energy-Guided Data Synthesis and Discrimination-Aware Multitask Optimization
Abstract: With growing privacy concerns and the enforcement of data protection regulations, machine unlearning has emerged as a promising approach for removing the influence of forget data while maintaining model performance on retain data. However, most existing unlearning methods require access to the original training data, which is often impractical due to privacy policies, storage constraints, and other limitations. This gives rise to the challenging task of source-free unlearning, where unlearning must be accomplished without accessing the original training data. Few existing source-free unlearning methods rely on knowledge distillation and model retraining, which impose substantial computational costs. In this work, we propose the Data Synthesis-based Discrimination-Aware (DSDA) unlearning framework, which enables efficient source-free unlearning in two stages: (1) Accelerated Energy-Guided Data Synthesis (AEGDS), which employs Langevin dynamics to model the training data distribution while integrating Runge-Kutta methods and momentum to enhance efficiency. (2) Discrimination-Aware Multitask Optimization (DAMO), which refines the feature distribution of retain data and mitigates the gradient conflicts among multiple unlearning objectives. Extensive experiments on three benchmark datasets demonstrate that DSDA outperforms existing unlearning methods, validating its effectiveness and efficiency in source-free unlearning.

Section: Introduction
Modern Machine Learning (ML) models rely on vast amounts of data for training, which may contain sensitive 1 Zhejiang University, China. Correspondence to: Chaochao Chen <zjuccc@zju.edu.cn>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
or private information, posing significant privacy risks (Cao & Yang, 2015;Nguyen et al., 2022;Liu et al., 2023a;b). To mitigate these risks, multiple regulations, e.g., the European Union's General Data Protection Regulation (GDPR) (Voigt & Von dem Bussche, 2017) and the California Consumer Privacy Act (CCPA) (Pardau, 2018), mandate companies and organizations to implement data deletion mechanisms and grant individuals the right to be forgotten. This gives rise to the field of machine unlearning, whose primary goal is to ensure that the model eliminates all influence of the data requested for deletion (i.e., forget data) while preserving model integrity and performance on the remaining data (i.e., retain data).
Existing methods typically assume access to the training data, utilizing gradient computation or weight adjustment techniques to erase specific knowledge. Some methods estimate the influence of training data on model parameters, using the Fisher Information Matrix (FIM) (Foster et al., 2024;Golatkar et al., 2020) and Hessian (Mehta et al., 2022), which are prohibitively expensive due to the high dimensionality of the parameter space. Other methods retrain (Bourtoule et al., 2021;Chundawat et al., 2023a) or fine-tune (Tarun et al., 2023) the model with the training data, intentionally degrading model performance on forgot data while preserving its performance on retain data.
However, access to the original training data cannot be guaranteed in practical scenarios, due to privacy concerns, data retention policies, or other constraints. For example, many cloud platforms delete training data immediately after use to address privacy concerns and storage limitations. Similarly, in streaming data environments, real-time processing overwrites old data with new inputs, preventing historical data retention. Under these circumstances, inaccessible training data makes most existing unlearning methods infeasible. Consequently, there is a pressing need for unlearning without the training data, relying solely on the original model and limited auxiliary information (class labels) to perform unlearning, referred to as source-free unlearning.
Source-free unlearning is an emerging yet underexplored area, with only a few existing methods attempting to tackle its challenges. Specifically, GKT (Chundawat et al., 2023b) and ISPF (Zhang et al., 2024) both adopt the Data-Free Knowledge Distillation (DFKD) technique with a filtering mechanism to selectively transfer knowledge from the original model to a randomly initialized model. However, they both require training a new model from scratch during the distillation process, resulting in significant computational costs.
Motivated by the limitations of existing methods, we propose the Data Synthesis-based Discrimination-Aware (DSDA) unlearning framework to achieve efficient and effective source-free unlearning, which consists of the following two key stages:
(i) In the first stage, we overcome the unavailability of training data by proposing the accelerated energy-guided data synthesis (AEGDS) method to generate synthetic datasets. Specifically, we derive an energy function by reinterpreting the output logits of the original model and employ Langevin dynamics to implicitly model the training data distribution. To further improve efficiency, we incorporate high-order Runge-Kutta methods and momentum-based updates into the sampling process, reducing redundant sampling steps while preserving effectiveness.
(ii) In the second stage, we propose the Discrimination-Aware Multitask Optimization (DAMO) method, which utilizes the synthetic datasets for effective unlearning. Through feature space visualization, we observe that traditional unlearning losses disrupt feature distributions, causing retain class samples to become widely dispersed, leading to performance degradation. In light of this, we introduce a novel unlearning objective, discriminative feature alignment objective, to improve intra-class compactness and inter-class separability of retain classes, thereby improving model performance. Additionally, to resolve gradient conflicts arising from optimizing the triple objectives, i.e., forget, retain, and feature alignment, we develop a multitask optimization strategy, ensuring stability and balance in unlearning.
Our main contributions are summarized as follows:
• We propose DSDA, a novel two-stage framework for efficient, source-free unlearning, addressing the limitations of methods that rely on original data or incur high computational costs.
• For the first stage, we propose AEGDS to efficiently generate synthetic datasets as substitutes for the original data. For the second stage, motivated by insights from feature space analysis, we propose DAMO, an unlearning optimization method that enhances feature distribution and resolves gradient conflicts.
• Extensive experiments on three benchmark datasets demonstrate DSDA's superiority over existing methods across multiple tasks.
this section cite: ['b3', 'b31', 'b16', 'b38', 'b32', 'b11', 'b12', 'b28', 'b2', 'b36', 'b44']

Section: Related Work

this section cite: []

Section: Deep Machine Unlearning

this section cite: []

Section: Existing deep unlearning methods can be categorized based on their dependency on training data.
Methods Requiring Full Training Data. Many existing methods depend on access to the complete training data. Exact unlearning methods (Bourtoule et al., 2021;Yan et al., 2022;Kim & Woo, 2022) remove the forget data from the original dataset and retrain the model, incurring high computational costs. Approximate unlearning methods bypass complete retraining, with some methods leveraging Hessian matrices (Sekhari et al., 2021;Mehta et al., 2022;Li et al., 2023b) or Fisher Information Matrices (FIM) (Foster et al., 2024;Golatkar et al., 2020) to estimate and reverse the influence of forget data on model parameters, while others (Chundawat et al., 2023a;Thudi et al., 2022;Chen et al., 2024;Li et al., 2023a) fine-tune the original model directly using the training data.
this section cite: ['b2', 'b41', 'b18', 'b34', 'b28', 'b11', 'b12', 'b37', 'b5']

Section: Methods Requiring Partial Training Data.
When retain data is unavailable, Cha et al. (2024); Kim et al. (2024) leverage Projected Gradient Descent (PGD) (M ądry et al., 2017) to generate adversarial samples for the forget classes. SCAR Bonato et al. (2025) substitutes external datasets for the original data, and performs unlearning via Knowledge Distillation (KD). When forget data is unavailable, UNSIR (Tarun et al., 2023) generates noise matrices for the forget classes with an error-maximization mechanism to induce class-level unlearning. However, all these methods rely on full or partial access to training data, which is often infeasible in real-world scenarios.
Source-Free Unlearning. In the strictest setting of sourcefree unlearning, where neither the forget nor retain data is accessible, there are only few existing methods. GKT (Chundawat et al., 2023b) represents the first source-free unlearning method, applying DFKD within an adversarial inversion-and-distillation framework. Building on GKT, ISFP (Zhang et al., 2024) addresses the over-filtering issue in DFKD by introducing inhibited synthesis to reduce the generation of forgetting-related information. However, both methods rely on DFKD, which necessitates costly and timeconsuming model retraining, limiting their practicality for large-scale or real-time unlearning. In this work, we propose the DSDA framework, which overcomes these limitations by leveraging energy-guided data synthesis and efficient finetuning through discrimination-aware multitask optimization.
this section cite: ['b4', 'b17', 'b36', 'b44']

Section: Model Inversion
Model inversion (MI) aims to reconstruct training data by exploiting the model's outputs, gradients, or internal representations (Mahendran & Vedaldi, 2015). Gradient-based MI methods optimize synthetic data using techniques such as Momentum SGD or Adam (Kingma & Ba, 2014), minimizing the loss between model predictions and ground truth (Struppek et al., 2022;Yuan et al., 2023). Alternatively, GMI (Zhang et al., 2020) employs Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) to guide data generation, with subsequent works (Yuan et al., 2023;Nguyen et al., 2023;Chen et al., 2021) enhancing generator performance by integrating additional information from the target model. However, these methods are computationally expensive, requiring either frequent gradient evaluations or training of an auxiliary generator. In contrast, the proposed AEGDS manipulates the model's likelihood landscape through implicit distribution modeling (Chen et al., 2021), achieving efficient model inversion for source-free unlearning.
this section cite: ['b27', 'b19', 'b35', 'b43', 'b45', 'b13', 'b43', 'b30', 'b6', 'b6']

Section: Method
In this section, we define the problem of source-free unlearning and introduce two key stages of the DSDA framework, as illustrated in Figure 1. In the first stage, we propose the AEGDS, a method that efficiently generates synthetic datasets as substitutes for the original training data. In the second stage, we propose the DAMO method to perform unlearning using the synthetic datasets. DAMO incorporates a novel discriminative feature alignment objective and resolves gradient conflicts from optimizing multiple unlearning objectives simultaneously through multitask optimization, ensuring stable and balanced unlearning.
this section cite: []

Section: Preliminaries and Notations
First, we formulate machine unlearning problem as follows. Let D train = {(x train , y train )} ∈ X × Y denote the training dataset, where X and Y = {1, 2, ..., K} are input and class label space, respectively. Let C f denote the set of classes we intend to forget in a pre-trained ML model and D f denote the subset of training data corresponding to the forget classes. Similarly, C r denote the set of classes we intend to retain, with the corresponding data subset D r , satisfying D r = D train \D f . A ML model, represented by M (•; θ), generates classification probabilities for each class, where θ is the set of model parameters. M can be decomposed into two components: M f , which represents the collection of feature extraction layers, and M c , which represents the final classification layer. Let M (•; θ o ) denote the original model trained on the complete dataset D train . In this paper, we address a challenging yet practical scenario of source-free unlearning, where the unlearned model is generated directly from θ o and class information in C f , without requiring access to D train .
this section cite: []

Section: Accelerated Energy-Guided Data Synthesis
Since the original training data is inaccessible, we first introduce an energy-guided mechanism to generate synthetic datasets that approximate the distributions of the original data. To further improve efficiency, we propose the AEGDS mechanism, which integrates high-order numerical methods with momentum-based updates to reduce computational overhead while preserving the integrity of the generated samples.
Formalizing the Energy Function Energy-based models (EBMs) define a probability distribution over data samples using an energy function E(x), where the likelihood of a sample is inversely proportional to its energy (Grathwohl et al., 2019;LeCun et al., 2006). For an input x ∈ X , the energy function E : X → R maps each data sample into an energy value, which can be interpreted as an unnormalized probability (Du & Mordatch, 2019). To approximate the original data distribution for each class, the energy-based model can be constructed from the original M (•, θ o ) based on the observation that discriminative models inherently follow an energy-based framework (Grathwohl et al., 2019). Specifically, the energy function is derived from the logits of the classifier, defined as:
E θ (x, y) = -log M (x, θ o )[y],(1)
where M (x, θ o )[y] is the predicted probability of the sample x belonging to class y.
this section cite: ['b14', 'b21', 'b10', 'b14']

Section: Energy-Guided Data Synthesis (EGDS)
By specifying an energy function E θ (x, y) for a target class y, we can then perform data synthesis following the principles of Stochastic Gradient Langevin Dynamics (SGLD) (Welling & Teh, 2011), which generates samples that align with the energybased model by iterative gradient-guided updates combined with noise perturbations.
Theorem 3.1. Given the initial data probability q(x) and the target probability p(x), one can determine the transformation process via T (x) = x + ϵ by minimizing KL-Divergence as shown:
min J Langevin = KL (q T (x)∥p(x))
where :
T (x) = x + ϵ (2
)
where ϵ denotes the searching direction and the optimal solution on ϵ is given as
ϵ = -α∇ ϵ [KL (q T (x)∥p(x))]
where α denotes the step size on gradient descend. The result can be further calculated as
T (x) = x + αE x∼q(x) [∇ x log p(x)].
Proof. By taking the differentiation on KL (q T (x)∥p( x w.r.t ϵ, we can obtain the following results:
∇ ϵ [KL (q T (x)∥p(x))] = ∇ ϵ [KL (q(x)∥p T -1 (x))] = -E x∼q(x) [∇ ϵ log p(T (x))| det ∇ x T (x)|] = -E x∼q(x) [∇ ϵ log p(T (x)) + ∇ ϵ log | det ∇ x T (x)|] .
For the first term on Langevin Dynamics, it can be further calculated ∇ ϵ log p(T (x)) = ∇ x log p(T (x))∇ ϵ T (x) and ∇ ϵ T (x) = I. Therefore, by considering the searching direction ϵ with minimizing KL (q T (x)∥p(x)), we can conclude the result as:
T (x) = x -∇ ϵ [KL (q T (x)∥p(x))] = x + αE x∼q(x) [∇ x log p(x)] .
Inspired by the valuable insights into Langevin dynamics presented in Theorem 3.1, we propose a novel energyguided data synthesis mechanism, termed the EGDS mechanism. Specifically, we randomly initialize the synthetic data xy 0 for a target class y and then update them iteratively according to the principles of SGLD, where the sampling process is carried out by constructing a Markov chain:
xy t+1 = xy t -α t ∇ x E θ (x y t , y) + √ 2α t ϵ t ,(3)
where x y t and ϵ t are the sample and Gaussian noise at iteration time step t, {α i } N t=1 is the sequence of step size. As demonstrated in (Welling & Teh, 2011), by appropriately introducing noise and gradually reducing the step size, this procedure will converge to the distribution defined by the energy function.
The pseudocode are shown in Algorithm 1 Accelerated Energy-Guided Data Synthesis (AEGDS) Traditional Langevin dynamics sampling, which uses a fixed time step sequence to update data states iteratively, can be computationally expensive due to frequent gradient evaluations and noise perturbations. While this stepwise progression ensures theoretical soundness, it often results in inefficient computations due to the computational cost of performing gradient evaluations and noise perturbations at every time step. To address this limitation, we propose the AEGDS mechanism, which skips redundant time steps while preserving sampling accuracy. We begin by randomly initializing the synthetic data xy 0 and selecting a sequence of time steps {t i } N i=0 , where t i+1 -t i > 0. Starting with the previous data state xy ti at time step t i , the exact solution of xy ti+1 at the subsequent time step t i+1 is given by:
xy ti+1 = xy ti -α ti ti+1 ti ∇ x E θ (x y t , y)dt = xy ti + α ti ti+1 ti s θ (x y t , y)dt,(4)
where s θ (x, y) = -∇ x E θ (x, y). In summary, the process is equal to progressively moving the sample from the initial time step to t N , skipping intermediate steps while still capturing the target distribution.
To numerically approximate the integral, we leverage a two-stage second-order Runge-Kutta method, which improves accuracy by incorporating intermediate gradient evaluations, reducing the errors of first-order approximations. Specifically, the synthetic data at each time step t i is updated following:
x ′ = xy ti + η∆t i d i + 2α ti ϵ ti , xy ti+1 = xy ti + α ti ∆t i 1 - 1 2η d i + 1 2η d ′ i + 2α ti ϵ ti ,(5)
where d i = s θ (x y ti , y) (black arrow in Figure 1)), d ′ i = s θ (x ′ , y) (grey arrow in Figure 1)), ∆t i = t i+1 -t i and η is a weight parameter. We set η = 1, corresponding to Heun's second-order method (Ascher & Petzold, 1998).
To further accelerate the sampling process, we incorporate Nesterov's momentum (Nesterov, 1983) which can improve convergence rates in gradient-based optimization. The momentum v i+1 at each time step is updated as:
v i+1 = γv i + α ti vi+1 ,(6)
where vi+1 = 1 2 d i + 1 2 d ′ i (blue arrow in Figure 1)), v i is the previous momentum (yellow arrow in Figure 1)) and γ is a momentum decay factor. The momentum term effectively leverages past gradients to accelerate convergence and smooth the updates. We formulate the overall algorithm for AEGDS in Algorithm 1. We leverage the AEGDS mechanism to perform class-wise sampling with parallel computing, generating synthetic data for each class in C r and C f to form Dr and Df .
this section cite: ['b40', 'b40', 'b0', 'b29']

Section: Discrimination-Aware Multitask Optimization
After Section 3.2, we have obtained synthetic retain and forget datasets Dr and Df Then in this section, we propose the DAMO method to fulfill the unlearning task . To begin with, we observe from visualization of future space that the separability and compactness of the retain classes are disrupted during unlearning, which motivates the introduction of the discriminative feature alignment objective. To further resolve gradient conflicts among multiple objectives, we develop a multitask optimization strategy for more balanced and effective unlearning.
Dual unlearning objectives Following existing studies (Golatkar et al., 2020), We decompose the unlearning process into two distinct objectives, formalized as follows: The retain objective ensures the model maintains performance on the retain classes. This is expressed by the retain loss function L R ( Dr ; θ) = L CE (M (x r ; θ), y r ), where xr represents the synthetic data samples in Dr and y r represents the corresponding retain class labels. In contrast, the forget objective aims to degrade the model's performance on 5: Update data: xy i+1 ← xy i + α i s θ (x y i , y) + √ 2α i ϵ i 6: else if adopt AEDGS then 7: Sample ϵ ti ∼ N (0, I) 8:
d i ← s θ (x y ti , y) 9: x ′ ← xy ti + ∆t i d i + √ 2α ti ∆t i ϵ ti
10: (2) Conversely, in the unlearned model, the forget class samples are more widely dispersed across multiple retained classes, which weakens both the inter-class separability and the intra-class compactness of the retain classes. As a result, the model's performance declines due to the disruption in feature distributions.
d ′ i ← s θ (x ′ , y) 11: vi+1 ← 1 2 d i + 1 2 d ′ i 12: Update momentum: v i+1 ← γv i + α ti vi+1 13: Update data: xy ti+1 ← xy ti + ∆t i v i+1 + √ 2α ti ∆t i ϵ ti14
In light of the above observations and existing studies (Wang et al., 2025), we propose the discriminative feature alignment objective to improve unlearning performance. This objective is designed to simultaneously improve intra-class  compactness by clustering samples of the same class and enhance inter-class separability by pushing apart samples of different classes in the feature space.
Specifically, given data sample xr belonging to class y, we define the inter-class similarity as s j n = c T j h/(∥c j ∥∥h∥) and the intra-class similarity as s p = c T y h/(∥c y ∥∥h∥), where j ∈ C r \{y}. Here, h = M f (x r ; θ) denotes the feature vector of the data sample, and c i denotes the feature center of class i. To achieve discriminative alignment, we introduce the following loss function:
L Disc = log   1 + |Cr|-1 j=1 exp δα j n s j n exp (-δαpsp)   , (7
)
where δ is a scale factor, α j n , α p are weight factors controlling the contribution of inter-class and intra-class similarities, respectively. To ensure stability and prevent similarity scores from deviating far from their optimal values, we define the weight factors as α j n = [s j n -O n ] + , α p = [O p -s p ] + , where [] + is the "cut-off at zero" operation that ensures non-negativity.
Multitask Optimization for Unlearning Optimizing the three unlearning objectives-forget, retain, and discriminative feature alignment-simultaneously is challenging due to inherent conflicting gradients (Yu et al., 2020). These conflicts prevent all objectives from being optimized simultaneously, leading to suboptimal performance, such as over-forgetting or under-forgetting. Let g R = ∇ θ L R , g F = ∇ θ L F , and g Disc = ∇ θ L Disc be the gradients of the retain, forget, and discriminative objectives, respectively. The combined gradient is defined as
g T = λ 1 g R + λ 2 g F + λ 3 g Disc .
A gradient conflict occurs when the combined gradient g T misaligns with any individual gradient, expressed as ⟨g i , g T ⟩ < 0, i ∈ {R, F, Disc}. To address this conflict, we propose a multitask optimization strategy for unlearning, which resolves gradient conflicts and balances three unlearning objectives effectively.
Theorem 3.2. Given the gradients {g i }, i ∈ {R, F, Disc}, one can find an update direction d that minimizes gradient conflicts while remaining close to the initial combined gradient g T as:
arg max d min ⟨g R , d⟩, ⟨g F , d⟩, ⟨g Disc , d⟩ , s.t.∥d -g T ∥ ≤ c∥g T ∥,(8)
where c ∈ [0, 1) is a hyper-parameter that controls the extent of deviation from the weighted average gradient g T .
To further reduce computational complexity, we introduce auxiliary weight variables w = (w R , w F , w Disc ) ∈ R 3 , representing the contributions of each gradient. These weights satisfy i w i = 1 and w i ≥ 0. Equation ( 8) can now be reformulated as:
max d min w ⟨g w , d⟩, s.t.∥d -g T ∥ ≤ c∥g T ∥, (9
)
where g w = w R g R + w F g F + w Disc g Disc represents the weighted gradient direction.
Theorem 3.3. Given the optimization problem in Equation (9), one can obtain the optimal d by solving the following Lagrangian:
min λ≥0, i w i =1,w i ≥0 max d g ⊤ w d - λ 2 ∥g T -d∥ 2 + λϕ 2 ,(10)
where ϕ = c 2 ∥g T ∥ 2 represents a constraint on the update magnitude. The optimal solution can be expressed as
d * = g T + g w * /λ * , where w * = arg min wi=1,w≥0 g ⊤ w g T + √ ϕ∥g w ∥, λ * = ∥g w * ∥/ϕ 1/2 .
Proof. We can derive the Lagrangian of Equation ( 9) as:
max d min λ≥0, i w i =1,w i ≥0 g ⊤ w d - λ 2 ∥g T -d∥ 2 -ϕ ,(11)
where ϕ = c 2 ∥g T ∥ 2 represents a constraint on the update magnitude. Base on Equation ( 11), we can derive Equation ( 10) due to the concavity of the objective with respect to d and the linearity of its constraints, allowing the interchange of max and min operations. Then, to solve Equation ( 10), we first fix λ and w, and optimize the inner maximization problem. The optimal solution d * is obtained by setting the derivative with respect to d to zero:
∂ ∂d g ⊤ w d + λg ⊤ T d - λ 2 ∥d∥ 2 = gw + λg T -λd = 0. (12
)
Therefore, we obtain d * = g T + g w /λ. Substituting d * back into the objective function, the outer minimization problem simplifies to:
min λ≥0, i w i =1,w i ≥0 g ⊤ w g T + ∥gw∥ 2 2λ - λ 2 ∥g T ∥ 2 + λϕ 2 .(13)
Then, we compute λ by taking the derivative and solving the resulting condition:
∂ ∂λ ∥gw∥ 2 2λ - λ 2 ∥g T ∥ 2 + λϕ 2 = - ∥gw∥ 2 2λ 2 - ∥g T ∥ 2 2 + ϕ 2 .(14)
Setting the derivative to zero, the optimal λ * is given by
λ * = ∥g w ∥/ √ ϕ. Substituting λ * back into the objective, we obtain the optimal w * = arg min wi=1,w≥0 g ⊤ w g T + √
ϕ∥g w ∥, validating the correctness of Theorem 3.3.
Finally, according to Theorem 3.3, the unlearned model can be obtained by applying the optimal update direction to θ o iteratively, which resolves gradient conflicts and maintains a balance among the unlearning objectives, thereby improving the overall unlearning performance. Additionally, to clarify the shift in feature distributions after introducing the discriminative feature alignment objective, we visualize the feature space of the unlearned model using the proposed DAMO method, as shown in Figure 2 (d). The results reveal that the retained class samples exhibit both clear inter-class separability and intra-class compactness, closely aligning with the retrained model.
this section cite: ['b12', 'b39', 'b42']

Section: Experiments
In this section, we evaluate the effectiveness of DSDA across three benchmark datasets and two model architectures. Additionally, we conduct ablation experiments to analyze the contribution of its three key components. Finally, we visualize the synthetic data to verify its integrity and privacy.
this section cite: []

Section: Experiment Settings
Datasets and Tasks We conduct experiments on CIFAR-10, CIFAR-100 (Krizhevsky et al., 2009) and PinsFaceRecognition (Hereis, 2024) datasets. Following existing studies (Cha et al., 2024;Foster et al., 2024), we adopt ResNet-18 (He et al., 2016) as the backbone for CIFAR-10 and CIFAR-100, and Vision Transformer (ViT) (Dosovitskiy et al., 2021) for PinsFaceRecognition.
Baselines We compare the proposed DSDA with the original model and the following unlearning methods: (1) Retrain refers to training a model from scratch on the retain data only. (2) SSD (Foster et al., 2024) selectively dampens model parameters according to their importance to the forget data, as determined by FIM.
(3) UNSIR (Tarun et al., 2023) uses error-maximizing noise to approximate D f and combines the noise with D f to fine-tune the original model. (4) ADV+IMP (Cha et al., 2024) fine-tunes the original model using adversarial examples. (5) LAU (Kim et al., 2024) applys Partial-PGD and KD to the classification layer. (6) SCAR (Bonato et al., 2025) modifies the feature vector projections of surrogate forget data using metric learning and KD. (7) GKT (Chundawat et al., 2023b) employs a generator to maximize the information gap between teacher and student model, and refines model weights through gated knowledge transfer. (8) ISPF (Zhang et al., 2024) improves upon GKT by introducing the Inhibited Synthetic and Post-Filter methods.
Implementation Details We implement all experiments in Python 3.9 and use the PyTorch library (Paszke et al., 2019).
All experiments are conducted on two NVIDIA RTX 3090 GPUs and repeated three times with different random seeds. Both the original and retrained models are trained from scratch using a multi-step learning rate scheduler, which begins with a learning rate of 0.01, and optimized with the Adam optimizer (Kingma & Ba, 2014). For a fair comparison, the batch sizes of all methods are set to 256 in ResNet18 and 32 in ViT. We carefully tune all comparison methods to achieve their best performance.
Evaluation Metrics Following existing studies (Tarun et al., 2023;Cha et al., 2024;Foster et al., 2024), we adopt the following four metrics to measure the overall performance of an unlearning method. ( 1
this section cite: ['b20', 'b16', 'b4', 'b11', 'b15', 'b9', 'b11', 'b33', 'b19', 'b36', 'b4', 'b11']

Section: Comparison with Baselines
To comprehensively evaluate DSDA's performance, we conduct single-class unlearning experiments across three datasets (shown in Table 1) and multi-class unlearning experiments on the CIFAR-10 dataset (shown in Table 2). Based on the objective of machine unlearning, we expect the performance of a desirable unlearning method to be close to the performance of the Retrain baseline. From the experimental results, we draw the following conclusions, considering both unlearning effectiveness and efficiency.
this section cite: []

Section: Unlearn Effectiveness.
(1) DSDA achieves complete removal of forget data information, with A f reaching 0% across all unlearning tasks, while also attaining the highest A r among all source-free unlearning methods. Additionally, DSDA demonstrates the best or second-best accuracy performance compared to all baselines. Note that this is not a fair comparison, as source-free methods function without the original training data, making their unlearning task more challenging.
(2) Several baselines, including ADV+IMP and UNSIR, exhibit significantly higher MIA values than Retrain, indicating potential leakage of forget data, despite achieving desirable accuracy. DSDA achieves the best MIA results, closely aligning with the retrained model, further demonstrating its effectiveness in mitigating privacy risks.
(3) Moreover, as the number of forgotten classes increases, LAU and GKT experience notable performance degradation. In contrast, DSDA maintains robust performance regardless of the number of forget classes, underscoring its scalability and reliability in more complex unlearning scenarios.
Unlearn Efficiency. The ET results demonstrate that DSDA significantly outperforms source-free baselines in unlearning efficiency, with an average improvement of 68.50%.
(b) PinsFaceRecognition (a) CIFAR-10 Furthermore, DSDA also surpasses most non-source-free baselines in efficiency, despite the additional step of generating synthetic data to substitute the original training data.
Moreover, we analyze the relationship between A r and ET for source-free unlearning methods, as shown in Figure 3. In the figure, the A r values of DSDA remain constant before a certain point, which is due to the model being frozen during the data synthesis stage. The results demonstrate that DSDA consistently achieves higher A r than GKT and ISPF, while reaching optimal performance more efficiently.
this section cite: []

Section: Ablation Study
We conduct ablation experiments on three key components in the DSDA, to elucidate their contributions respectively. Specifically, DSDA-w-EGDS represents DSDA without data (a) Feature distribution of synthetic data and original data (b) Visualization of synthetic data Figure 5. Visualization of feature distribution and synthetic data for CIFAR-10. Round dots represent original data, square dots represent synthetic data. synthesis acceleration. DSDA-wo-disc represents DSDA without the discrimination feature alignment objective, using only L F and L R . DSDA-wo-mt represents DSDA without multi-task optimization, where model parameters are updated directly using g T . The results of A r and ET on CIFAR-10 and CIFAR-100 datasets, as shown in Figure 4, provide several important insights: (1) DSDA-w-EGDS requires significantly longer ET than DSDA, highlighting the importance of AEGDS in improving efficiency. Additionally, DSDA outperforms DSDA-w-EGDS in A r , due to the second-order Runge-Kutta method reducing approximation errors. (2) DSDAwo-disc shows lower A r compared to DSDA, underscoring the critical role of the alignment objective in preserving intra-class compactness and inter-class separability, which in turn leads to better model performance. (3) DSDA-womt leads to lower A r and higher ET, demonstrating that multi-task optimization not only balances the unlearning objectives but also accelerates the unlearning process.
this section cite: []

Section: Additional Analysis
Feature distribution of Synthetic Data. We visualize the feature distribution of both the synthetic and original CIFAR-10 dataset using the original model, as shown in Figure 5. The results reveal that the synthetic data closely overlaps with the original data in feature space, exhibiting nearly identical distributions, which suggests that the proposed AEGDS effectively models the original data distribution.
Visualization of Synthetic Data. To further evaluate the privacy implications of the synthetic data, we visualize its appearance in Figure 5 (b). The synthetic samples are visually indistinguishable, impossible for human observers to extract any meaningful information, which ensures that the synthetic data poses no privacy risks.
this section cite: []

Section: Conclusion
We propose DSDA, a novel source-free unlearning framework, which addresses the critical challenges of inaccessible training data and computational cost. The two key components of DSDA, i.e. AEGDS and DAMO, enable the generation of synthetic data and refinement of feature distributions, thereby ensuring both the removal of forgot knowledge and the preservation of performance on retain data. Extensive experiments on multiple benchmark datasets demonstrate that DSDA outperforms existing unlearning methods in terms of both efficiency and effectiveness. While DSDA is effective in removing class-level information, it remains limited in finer-grained unlearning scenarios, such as instance-wise or attribute-wise unlearning. Exploring source-free unlearning methods tailored to such fine-grained unlearning tasks presents an important direction for future work.
this section cite: []

Section: References
Ref_id:b0 Title: Computer methods for ordinary differential equations and differential-algebraic equations Year: (1998)
Ref_id:b1 Title: Is retain set all you need in machine unlearning? restoring performance of unlearned models with out-of-distribution images Year: (2025)
Ref_id:b2 Title: Machine unlearning Year: (2021)
Ref_id:b3 Title: Towards making systems forget with machine unlearning Year: (2015)
Ref_id:b4 Title: Learning to unlearn: Instance-wise unlearning for pretrained classifiers Year: (2024)
Ref_id:b5 Title: Post-training attribute unlearning in recommender systems Year: (2024)
Ref_id:b6 Title: Knowledgeenriched distributional model inversion attacks Year: (2021)
Ref_id:b7 Title: Can bad teaching induce forgetting? unlearning in deep networks using an incompetent teacher Year: (2023)
Ref_id:b8 Title: Zero-shot machine unlearning Year: (2023)
Ref_id:b9 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b10 Title: Implicit generation and modeling with energy-based models Year: (2019)
Ref_id:b11 Title: Fast machine unlearning without retraining through selective synaptic dampening Year: (2024)
Ref_id:b12 Title: Eternal sunshine of the spotless net: Selective forgetting in deep networks Year: (2020)
Ref_id:b13 Title: Generative adversarial nets Year: (2014)
Ref_id:b14 Title: Your classifier is secretly an energy based model and you should treat it like one Year: (2019)
Ref_id:b15 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b16 Title: Pins face recognition dataset Year: (2024)
Ref_id:b17 Title: Layer attack unlearning: Fast and accurate machine unlearning via layer level attack and knowledge distillation Year: (2024)
Ref_id:b18 Title: Efficient two-stage model retraining for machine unlearning Year: (2022)
Ref_id:b19 Title: A method for stochastic optimization Year: (2014)
Ref_id:b20 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b21 Title: A tutorial on energy-based learning Year: (2006)
Ref_id:b22 Title: Enhancing receraser for recommendation unlearning via error decomposition Year: (2023)
Ref_id:b23 Title: Selective and collaborative influence function for efficient recommendation unlearning Year: (2023)
Ref_id:b24 Title: Federated probabilistic preference distribution modelling with compactness co-clustering for privacypreserving multi-domain recommendation Year: (2023)
Ref_id:b25 Title: Differentially private sparse mapping for privacy-preserving cross domain recommendation Year: (2023)
Ref_id:b26 Title: Towards deep learning models resistant to adversarial attacks Year: (2017)
Ref_id:b27 Title: Understanding deep image representations by inverting them Year: (2015)
Ref_id:b28 Title: Deep unlearning via randomized conditionally independent hessians Year: (2022)
Ref_id:b29 Title: A method for solving the convex programming problem with convergence rate o (1/k2) Year: (1983)
Ref_id:b30 Title: Label-only model inversion attacks via knowledge transfer Year: (2023)
Ref_id:b31 Title: A survey of machine unlearning Year: (2022)
Ref_id:b32 Title: The california consumer privacy act: Towards a european-style privacy regime in the united states Year: (2018)
Ref_id:b33 Title: Pytorch: an imperative style, high-performance deep learning library Year: (2019)
Ref_id:b34 Title: Remember what you want to forget: Algorithms for machine unlearning Year: (2021)
Ref_id:b35 Title: Plug & play attacks: Towards robust and flexible model inversion attacks Year: (2022)
Ref_id:b36 Title: Fast yet effective machine unlearning Year: (2023)
Ref_id:b37 Title: Unrolling sgd: Understanding factors influencing machine unlearning Year: (2022)
Ref_id:b38 Title: The eu general data protection regulation (gdpr) Year: (2017)
Ref_id:b39 Title: Interand intra-similarity preserved counterfactual incentive effect estimation for recommendation systems Year: (2025)
Ref_id:b40 Title: Bayesian learning via stochastic gradient langevin dynamics Year: (2011)
Ref_id:b41 Title: Arcane: An efficient architecture for exact machine unlearning Year: (2022)
Ref_id:b42 Title: Gradient surgery for multi-task learning Year: (2020)
Ref_id:b43 Title: Pseudo label-guided model inversion attack via conditional generative adversarial network Year: (2023)
Ref_id:b44 Title: Toward efficient data-free unlearning Year: (2024)
Ref_id:b45 Title: The secret revealer: Generative model-inversion attacks against deep neural networks Year: (2020)
