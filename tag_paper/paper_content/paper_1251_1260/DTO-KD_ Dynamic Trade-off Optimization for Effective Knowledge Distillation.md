Title: DTO-KD: DYNAMIC TRADE-OFF OPTIMIZATION FOR EFFECTIVE KNOWLEDGE DISTILLATION
Abstract: Knowledge Distillation (KD) is a widely adopted framework for compressing large models into compact student models by transferring knowledge from a highcapacity teacher. Despite its success, KD presents two persistent challenges: (1) the trade-off between optimizing for the primary task loss and mimicking the teacher's outputs, and (2) the gradient disparity arising from architectural and representational mismatches between teacher and student models. In this work, we propose Dynamic Trade-off Optimization for Knowledge Distillation (DTO-KD), a principled multi-objective optimization formulation of KD that dynamically balances task and distillation losses at the gradient level. Specifically, DTO-KD resolves two critical issues in gradient-based KD optimization: (i) gradient conflict, where task and distillation gradients are directionally misaligned, and (ii) gradient dominance, where one objective suppresses learning progress on the other. Our method adapts per-iteration trade-offs by leveraging gradient projection techniques to ensure balanced and constructive updates. We evaluate DTO-KD on large-scale benchmarks including ImageNet-1K for classification and COCO for object detection. Across both tasks, DTO-KD outperforms prior KD methods, yielding state-of-the-art accuracy and improved convergence behavior. Furthermore, student trained with DTO-KD exceed the performance of their non-distilled counterparts, demonstrating the efficacy of our multi-objective formulation.

Section: INTRODUCTION
Deep neural networks have demonstrated impressive performance across a wide range of computer vision tasks; however, their practical deployment is often hindered by substantial computational and memory demands, particularly on resource-limited platforms such as edge devices and mobile hardware. This has motivated increasing interest in model compression techniques that reduce network complexity without sacrificing accuracy. Among these, knowledge distillation (KD) (Yim, 2017;Gao et al., 2018;Qiu et al., 2023;Zhou et al., 2020) has emerged as an effective strategy, in which a compact student model is trained under the guidance of a large, pre-trained teacher model. By transferring informative representations or outputs from the teacher, the student can achieve comparable performance with significantly fewer parameters, making it more amenable to deployment in constrained environments. A standard KD framework typically combines a task-specific objective (e.g., classification or detection loss) with an auxiliary distillation loss that facilitates knowledge transfer from the teacher to the student.
Early studies on knowledge distillation (KD) (Hinton et al., 2015;Zhang et al., 2018) primarily relied on training the student model using the teacher's output predictions as supervisory signals. While effective to some extent, this strategy is inherently limited, as the teacher's final outputs provide a highly compressed representation of its knowledge, and relying solely on logits constrains the richness of transferable information. To overcome these shortcomings, subsequent KD methods (Romero et al., 2015;Chen et al., 2020;Heo et al., 2019a) moved toward leveraging intermediate feature representations from the teacher, allowing for more expressive and flexible knowledge transfer. Such feature-based distillation methods often depend on heuristic design choices and introduce additional hyperparameters that require careful, task-dependent tuning. Nevertheless, despite these improvements, recent feature-level KD approaches (Chen et al., 2022;2021; Roy Miles & Deng, 2024) continue to face challenges in effectively distilling knowledge from highly expressive teacher networks into compact student models, largely due to mismatches between the objectives imposed by ground-truth supervision and those induced by the distillation process.
Recent KD methods (Wang et al., 2024;Chen et al., 2022) propose heuristic mechanisms for balancing teacher mimicking, yet optimization inconsistency persists as a critical bottleneck to efficient knowledge transfer. The primary issue limiting the performance of these approaches is two-fold. First, Gradient Conflicts (GrC) arise when the gradients of the task-specific objective and the distillation process are misaligned. Second, Gradient Dominance (GrD) occurs when the gradient magnitude of one objective (e.g., either distillation or task-specific) dominates the learning process, causing an imbalance. Figure 1 illustrates these issues by plotting gradient conflict and dominance for our method and that of Roy Miles & Deng (2024) over 500 iterations on the detection task.
To address all of these issues, we propose a novel distillation optimization strategy. Specifically, we frame the problem as a dynamic trade-off optimization, which not only efficiently resolves gradient conflicts during training but also ensures a Pareto optimal (Lin, 1976) solution. This results in a training strategy that eliminates the need for manually tuning hyperparameters to balance the contributions of each loss function. Instead, it dynamically learns the contribution of each loss function, adapting between task-specific and distillation-specific objectives throughout the training.
To be more specific, in this paper we propose a closed-form method for determining how to weight the distillation and task-specific losses during training. Unlike the prior work of (Liu et al., 2023), our approach provides an explicit solution that can be computed efficiently at each step. In teacher-student architectures, where the distillation and task losses evolve rapidly, existing taskweighting methods (Hu et al., 2024;Zheng & Yang, 2024) can struggle to adapt, causing weights to oscillate or lag behind the changing dynamics. In contrast, our closed-form solution produces an update direction that is jointly aligned with both objectives, ensuring that neither the distillation nor the task loss dominates or interferes with the other. As a result, our method naturally mitigates gradient conflict and yields a more stable and effective multi-objective learning process.
In this paper, we introduce DTO-KD (Dynamic Trade-off Optimization for Knowledge Distillation), a novel multi-objective learning framework that formulates knowledge distillation as a gradient-level optimization problem. DTO-KD improves the efficiency and effectiveness of knowledge transfer by dynamically modulating the contribution of task-specific and distillation-specific objectives during training, removing the need for manual loss weighting or extensive hyperparameter tuning. DTO-KD is trained end to end and demonstrates faster convergence, requiring fewer epochs to reach or exceed the performance of state-of-the-art distillation methods. In summary, the contributions of this paper are as follows:
• We propose DTO-KD, a dynamic trade-off optimization framework that balances task and distillation losses at the gradient level. This principled approach eliminates the need for fixed loss weighting, enabling adaptive trade-offs during training.
• DTO-KD resolves gradient conflict (GrC) and dominance (GrD) via per-iteration gradient balancing approach, leading to aligned, balanced updates and improved convergence.
• We conduct extensive experiments on both classification and detection benchmarks, achieving state-of-the-art performance. Ablation studies confirm the robustness of DTO-KD across diverse distillation setups.
this section cite: ['b46', 'b5', 'b24', 'b52', 'b9', 'b48', 'b27', 'b1', 'b3', 'b40', 'b3', 'b14', 'b17', 'b10', 'b51']

Section: RELATED WORK
This section explores KD techniques, focusing on the use of logits, CNN features, and transformer features (or tokens). Additionally, it examines multi-objective approaches relevant to the DTO-KD.
Aligning Predictive Distributions via Logit-Level Distillation: Classical Knowledge Distillation (KD) primarily relies on the teacher's output logits. Ensemble-based collaborative learning (Zhang et al., 2018), multi-stage distillation with teacher assistants (Mirzadeh et al., 2020), and decoupled distillation across teacher branches (Zhao et al., 2022a) all fall within this paradigm. Most methods use forward KL-divergence to align teacher-student distributions, though this often yields overly smoothed predictions. Reverse KL-divergence (Wang et al., 2025a) emphasizes the teacher's highconfidence modes, while α-β divergence (Wang et al., 2025b) generalizes both by interpolating between their behaviors. However, logit-based approaches transfer only final-layer information, omitting rich intermediate representations and teacher inductive biases. Consequently, students struggle to reconcile teacher outputs with task-specific objectives, often leading to limited generalization.
Transferring Intermediate Representations via Feature-Level Distillation: Feature-based Knowledge Distillation (KD) leverages intermediate activations to convey structural knowledge unavailable at the logit level (Yang et al., 2021;Xu et al., 2020). Early work such as FitNets (Romero et al., 2015) introduced stage-wise alignment, enabling the student to mimic deeper teacher features. Subsequent methods refine how features are selected and aligned. Margin ReLU filtering (Heo et al., 2019a) suppresses redundant activations and improves feature matching, while analysis of connection pathways (Chen et al., 2021) highlights the importance of optimal teacher-student layer mappings. Diffusion-based distillation (Huang et al., 2024) reduces noise in student features before transferring knowledge, and norm/direction-aware losses (Wang et al, 2024) further enhance feature alignment. Despite progress, feature-level KD still struggles to encode long-range dependencies and global contextual knowledge that is essential in modern architectures and increasingly captured by transformer-based models.
Leveraging Transformer Semantics via Token-Level Distillation: Transformer-based Knowledge Distillation (KD) focuses on transferring information encoded in self-attention and token interactions. DeiT (Touvron et al., 2022) first established efficient token distillation for convolution-free vision transformers. For detection tasks, token-matching methods (Song et al., 2021;2022) require the student to replicate teacher tokens, though naive token alignment is often insufficient. More advanced methods incorporate multiple teachers (Ren et al., 2022), manifold-based token alignment (Hao et al., 2022), and generalized f-divergence formulations (Wen et al., 2023) that flexibly weight the teacher's dominant predictions across tokens. However, transferring dark knowledge, such as subtle interactions and contextual semantics encoded in token relations, remains challenging. Approaches such as non-target logit normalization (Yang et al., 2023) or two-stage early-layer distillation pipelines (Chen et al., 2022) offer partial solutions but remain heuristic and fragmented.
In contrast, our work proposes a unified end-to-end formulation that dynamically balances objectives through trade-off optimization.
Aligning Conflicting Objectives via Multi-objective Optimization: Multi-objective optimization (MOO) enables simultaneous optimization of conflicting objectives by seeking Pareto-optimal tradeoffs. A simple approach re-weights loss functions based on manually designed criteria (Chen et al., 2018;Kendall et al., 2018), but these methods are often heuristic, ignore dynamic gradient interactions, and lack strong theoretical foundations. Gradient manipulation methods (Sener & Koltun, 2018;Yu et al., 2020;Liu et al., 2021b;a;2023) instead combine gradients from different tasks at each step. For example, Sener & Koltun (2018) uses an upper bound for efficiency, Yu et al. (2020) projects gradients to avoid conflicts, Liu et al. (2021b;a) provide a closed-form solution minimizing average loss, and Liu et al. (2023) introduces a fast dynamic weighting method. Although MOO is explored in multi-task learning, In this paper, we proposed DTO-KD which uniquely applies it to knowledge distillation, formulating it as a dynamic trade-off optimization problem to resolve conflicts between task and distillation objectives.
this section cite: ['b48', 'b20', 'b44', 'b43', 'b27', 'b2', 'b12', 'b40', 'b36', 'b31', 'b26', 'b6', 'b42', 'b45', 'b3', 'b13', 'b29', 'b47', 'b29', 'b47', 'b17']

Section: METHOD
We introduce a Dynamic Trade-off Optimization for Knowledge Distillation (DTO-KD), with a specific focus on resolving the conflicting objectives in the KD process.
Problem formulation. We aim to transfer knowledge from a high-capacity teacher model with parameters ϕ, to a more compact model student, with parameters θ, focusing mainly on classification and detection tasks in visual recognition. We show the training data with S = {(x i , y i )} N i=1 , with x i ∈ R d being the i-th input instance and y i the corresponding target (e.g. a class label, bounding box). Our goal is to train the student model to effectively mimic the behavior of the teacher model over the dataset S. Figure 2 shows an illustration of our proposed framework.
Effectively performing knowledge distillation requires balancing two objectives: the student must learn from two supervisory signals (e.g., one from the teacher and one from the task). We represent the teacher's loss as L distill and the task's loss as L task . While we will define these more specifically for image classification and object detection in the appendix, we provide their general forms here:
L distill (θ) ≜ E (x,y)∼S ℓ distill f s (x; θ), f t (x; ϕ) (1) L task (θ) ≜ E (x,y)∼S ℓ task f s (x; θ), f t (x; ϕ)(2)
The conventional KD approaches (e.g., (Hu et al., 2024;Zheng & Yang, 2024)) train the student model by optimizing the loss as
L tot (θ) ≜ α 1 L distill (θ) + α 2 L task (θ) ,(3)
where α 1 , α 2 ∈ R + are the combination weights and hyperparameters of the model. The gradient of L tot (θ
) is g tot = ∇L tot (θ) = α 1 g dist + α 2 g task (4
)
where g dist = ∇L distill (θ) and g task = ∇L task (θ) are the gradients of the distillation and task losses, respectively. Minimizing loss in Equation (3) for joint training introduces the following challenges:
Gradient Conflict (GrC). This occurs when the gradients of the distillation loss and the task loss conflict with each other. Mathematically, GrC happens when ⟨g dist , g task ⟩ < 0. During the optimization of the total loss L tot (θ), the occurrence of GrC leads to conflicting gradient updates. Specifically, the total gradient g tot may contradict either g dist or g task , causing detrimental effects on one or both objectives. This conflict can exacerbate the learning dynamics, particularly in complex vision tasks such as object detection, by introducing unnecessary complexity into the training process.
this section cite: ['b10', 'b51']

Section: Gradient Dominance (GrD).
It arises when the gradients have significantly different magnitudes, leading one to dominate the update. When minimizing L tot (θ), this imbalance may cause one objective to be completely neglected, as the update direction is primarily determined by the larger gradient, which can be estimated as ∥gdist∥ ∥gtask∥ . Lastly, tuning the hyperparameters α 1 and α 2 might become extremely tricky as the norm of gradients varies throughout optimization.
To address the aforementioned challenges, we advocate for the use of multi-objective optimization in KD. Specifically, we formulate the training process as optimizing the objective vector L tot (θ) = (L distill (θ), L task (θ))
⊤ . The goal is to find a solution θ * on the Pareto front, i.e., a solution that is not dominated by any other parameter vector θ. Formally,
θ * is Pareto optimal if is no θ such that L distill ( θ) L task ( θ) ⪯ L distill (θ * ) L task (θ * ) (5
)
The notation a ⪯ b here means that vector a achieves a lower value for all its elements simultaneously over b. As we will discuss in the next section, formulating KD using the proposed algorithm addresses both GrC and GrD by aligning the gradients. Furthermore, the use of MOO mitigates the difficulty of hyperparameter tuning, as it eliminates the need to manually define α 1 and α 2 .
this section cite: []

Section: KD AS A DYNAMIC TRADE-OFF OPTIMIZATION
Inspired by Liu et al. (2023), we followed a two stage approach for learning the optimal trade-off between conflicting objectives during the model training.
Stage 1: In stage 1 and at time t, we update the student model via θ t+1 = θ t -ηg t , where η ∈ R + is the learning step size. We define the rate of improvement for the distillation and task losses as:
r dist (g t ) = L distill (θ t ) -L distill (θ t+1 ) L distill (θ t ) , r task (g t ) = L task (θ t ) -L task (θ t+1 ) L task (θ t ) .(6)
In essence, r dist (g t ) and r dist (g t ) measure how much each loss can be improved by moving the parameters with -ηg t . A larger value of r dist or r task implies the associated task has been improved more.
Stage 2: In stage 2, our goal is to determine an update g t that maximizes the improvement over the worst-case rate. This can be achieved using a min-max optimization as:
max gt∈R n min i∈{dist,task} 1 γ r i (g t ) - 1 2 ∥g t ∥ 2 .(7)
Here, γ ∈ R + is a weighting hyperparameter. As shown in Liu et al. (2023), the solution of Equation (7) can be obtained via solving its dual problem as (see proposition 3.1 in Liu et al. (2023)). Define π = (π 1 , π 2 ) ⊤ on the simplex ∆ (i.e., π 1 + π 2 = 1, π 1 , π 2 ≥ 0), and let J t ∈ R n×2 be
J t = ∇ log L distill (θ t ) | ∇ log L task (θ t ) ⊤ (8) Then π * t ∈ arg min π∈∆ 1 2 ∥J t π∥ 2 , (9
)
and g t = J t π * = π 1 ∇ log L distill (θ t ) + π 2 ∇ log L task (θ t ) .
Theoretical Properties. The problem formulation in Equation ( 9) admits an analytical solution, unlike the general case studied in Liu et al. (2023). In this part, we establish key theoretical properties of the obtained update direction g * . Theorem 3.1 (Closed Form Solution). Let J t = [∇ log L distill (θ t ) , ∇ log L task (θ t ) ] ∈ R n×2 . The closed-form solution to the optimization problem
π * ∈ arg min π 1 2 J t π 2 s.t. π 1 + π 2 = 1 (10
) is given by π * 1 = g 22 -g 12 g 11 + g 22 -2g 12 ,(11)
π * 2 = g 11 -g 12 g 11 + g 22 -2g 12 ,(12)
where G = J ⊤ t J t is the Gram matrix:
G = g 11 g 12 g 21 g 22 ,
with elements
g 11 = ∇ log L distill (θ t ) 2 , (13
) g 12 = g 21 = ∇ log L distill (θ t ) , ∇ log L task (θ t ) ,(14)
g 22 = ∇ log L task (θ t ) 2 . (15
)
The closed-form nature of this solution allows for efficient computation of the optimal weighting factors. One key property of the derived solution is that the update direction aligns with both objectives, ensuring that both the distillation and task losses are reduced simultaneously. This directly addresses GrC by preventing destructive interference between the two gradients.
this section cite: ['b17', 'b17', 'b17', 'b17']

Section: Corollary 3.2 (Alignment of g * ).
Define g 1 = ∇ log L distill (θ t ) and g 2 = ∇ log L task (θ t ) . Then the update direction g * = π 1 g 1 + π 2 g 2 for π * defined in 11 is aligned with both g 1 and g 2 .
Another key property of the proposed solution is that it enforces equal contribution of the update direction to both gradients, effectively addressing GrD. Corollary 3.3 (Equal Contribution of g * to Both Losses). In Corollary 3.2, we showed that
⟨g * , g 1 ⟩ = ⟨g * , g 2 ⟩ = g 11 g 22 -g 2 12 ∥g 1 -g 2 ∥ 2 .
This implies that the update direction contributes equally to the descent of both the distillation and task losses, effectively mitigating gradient dominance.
An important aspect of any gradient-based optimization method is ensuring that update magnitudes remain within a controlled range to prevent vanishing or exploding gradients. Our solution satisfies both a lower and an upper bound on ∥g * ∥, ensuring stability during training. Corollary 3.4 (Lower Bound on ∥g * ∥). The norm of the optimal update direction g * satisfies the lower bound:
∥g * ∥ ≥ 1 √ 2 min(∥g 1 ∥, ∥g 2 ∥) .(16)
This implies that the update magnitude remains controlled and does not collapse under gradient imbalance. Corollary 3.5 (Upper Bound on ∥g * ∥). The norm of the optimal update direction g * satisfies the upper bound:
∥g * ∥ ≤ ∥g 1 ∥∥g 2 ∥ ∥g 1 ∥ -∥g 2 ∥ . (17
)
As such, the magnitude of the updates does not grow excessively with different gradient scales.
Finally, we observe that the algorithm's convergence is ensured by the general theoretical framework outlined in Liu et al. (2023). As our formulation aligns with it, the proposed optimization is guaranteed to converge to a Pareto optimal front.
Practical Implementation. The detailed algorithm for the proposed DTO-KD approach is detailed in Algorithm 1. The distillation and task weights π are initialized to 0.5. The algorithm begins by initializing the teacher as a frozen model and the student as a trainable model, and then extracts latent features from both for each training batch. The DistillHead and TaskHead refer to specific heads learning distillation and the task, respectively. It computes the distillation and task
Algorithm 1 Dynamic Trade-off Optimisation for KD 1: Inputs: Dataset S = {(x i , y i ), ...}; Teacher f t 2: Initialise: Student f s with θ; Task weight π distill = π task ← 1 2 3: for t = 1 : T do (iterations) 4:
x τ , y τ = {(x b , y b )} B b=1 ∼ S (batch) 5: z t , z s ← f t (x τ ), f s (x τ ) (latent features) 6: ẑt , ẑs ← z ⊤ t P z s (projection) 7: L(θ t ) = L distill L task = ℓ distill (DistillHead(ẑ s , ẑt )) ℓ task (TaskHead(z s ), y τ ) (loss vector) 8: g t = π distill ∇ log L distill (θ t ) + π task ∇ log L task (θ t ) 9: θ t+1 = θ t -γ g t (student model learning) 10: L(θ t+1 ) ← f s (x τ ) (frozen model inference) 11: r(g t ) = r distill (g t ) r task (g t ) = Ldistill(θt)-Ldistill(θt+1) Ldistill(θt) Ltask(θt)-Ltask(θt+1) Ltask(θt) (update direction)
12:
π(t + 1) = π(t) -η π ∇ π 1 2 π distill (t) log L distill (θ t ) + π task (t) log L task (θ t ) 2 (optimize task weights)
losses, combines their gradients according to the current task weights, and updates the student model accordingly. After each update, the task weights are recalculated in closed form based on the relative improvement of each loss, ensuring a balanced optimization that aligns both the distillation and task objectives. Despite having strong theoretical properties, MTL algorithms (Liu et al., 2023), including the one we have developed above, require access to per task gradient, in our case access to J = [∇ log L distill (θ t ) , ∇ log L task (θ t ) ]. This incurs performing two backpropagation per iteration, which is not desired. Instead, one can advocate to amortizing the training. This leads to an approximation to the algorithm while ensuring that an extra backprop step is not required. In short, the parameters π = (π distill , π task ) are updated via
π(t + 1) = π(t) -η π ∇ π 1 2 π distill (t) log L distill (θ t ) + π task (t) log L task (θ t ) 2 . (18
)
The update in Equation ( 18) does not guarantee π ∈ ∆, one should renormalize it via a softmax function. We have empirically observed that the amortized algorithm comfortably outperforms stateof-the-art KD algorithms with significant improvement over training speed. Specifically, the DTO-KD reaches the top performance of Roy Miles & Deng (2024) with 300 epochs in just 240 epochs.
this section cite: ['b17', 'b17']

Section: EXPERIMENTS
We evaluate DTO-KD on two distinct vision tasks: image classification and object detection. For image classification, we adopt a CNN-based teacher model, RegNetY-160 (Radosavovic et al., 2020), and use transformer-based DeiT (Touvron et al., 2022) Small and Tiny as student models. For object detection, we employ transformer-based ViDT-Base (Song et al., 2021) as the teacher model, with ViDT-Small, ViDT-Tiny, and ViDT-Nano serving as the student models. Additionally, to assess the robustness of our method, we conduct distillation experiments using Vidt-Small as the teacher.
Implementation details: In DTO-KD, we reformulate model training as a gradient-based dynamic trade-off optimization problem. For the overall optimization across both classification and detection tasks, we use AdamW with a learning rate of 0.025 and a weight decay of 0.01. For classification, we adopt the training strategy and parameters from DeiT (Touvron et al., 2021a). Additionally, for data augmentation, we follow the method outlined in Roy Miles & Deng (2024). For learning, we employ AdamW (Loshchilov & Hutter, 2019) with a learning rate of 0.001 and a weight decay of 0.05. For object detection, we adhere to the training methodology from ViDT (Song et al., 2021). DTO-KD is trained using AdamW (Loshchilov & Hutter, 2019) with an initial learning rate of 10-4 for the body, neck, and head. We use the same hyperparameters as those in the ViDT (Song et al., 2021) transformer encoder and decoder. All experiments are conducted using PyTorch (Paszke et al., 2017) framework and executed on four NVIDIA H100 GPUs. Additionally, compared to the baseline (Touvron et al., 2021a), which was trained for 1000 epochs, DTO-KD achieves a 3.1 percentage point (pp) improvement for the tiny model and a 0.5 pp improvement for the small model with just 300 epochs. This highlights the efficiency of our approach, demonstrating its ability to deliver competitive performance in significantly less training time, making it both effective and scalable.
this section cite: ['b36', 'b31', 'b19', 'b31', 'b19', 'b31', 'b23']

Section: OBJECT CLASSIFICATION USING CIFAR-100 DATASET
Conventional knowledge distillation methods are typically evaluated on both homogeneous and heterogeneous CNN architectures using the CIFAR-100 dataset. To position DTO-KD against these approaches, we benchmarked it following the protocols of prior KD works (Chen et al, 2021;Wang et al, 2024). Table 2 shows that DTO-KD also achieves superior results and reinforces its superiority, establishing a new SOTA by outperforming previous works in both small-and large-dataset settings.
this section cite: ['b2', 'b41']

Section: OBJECT DETECTION
Table 3 demonstrates that our proposed method, DTO-KD, achieves state-of-the-art object detection performance on the MS-COCO benchmark (Lin et al., 2014), leveraging the ViDT transformer architecture (Song et al., 2022) for its strong performance and efficiency on consumer hardware. DTO-KD consistently improves upon various ViDT variants, enhancing the Swin-nano backbone by 0.7 percentage points (pp), Swin-tiny by 0.5pp, and Swin-small by 1.1pp. Notably, DTO-KD-
Homogeneous Heterogeneous Methods ResNet-56 WRN-40-2 ResNet-32×4 ResNet-50 ResNet-32×4 ResNet-32×4 ResNet-20 WRN-40-1 ResNet-8×4 MobileNet-V2 ShuffleNet-V1 ShuffleNet-V2 Teacher 72.34 75.61 79.42 79.34 79.42 79.42 Student 69.06 71.98 72.50 64.60 70.50 71.82 FitNet (Romero et al., 2015) 69.21 72.24 73.50 63.16 73.59 73.54 RKD (Park et al., 2019) 69.61 72.22 71.90 64.43 72.28 73.21 PKT (Passalis et al., 2020) 70.34 73.45 73.64 66.52 74.10 74.69 KD (Hinton et al., 2015) 70.66 73.54 73.33 67.65 74.07 74.45 OFD (Heo et al., 2019b) 70.98 74.33 74.95 69.04 75.98 76.82 CRD (Tian et al., 2019) 71.16 74.14 75.51 69.11 75.11 75.65 DIST (Huang et al., 2022) 71.78 74.42 75.79 69.17 75.23 76.08 ReviewKD (Chen et al, 2021) 71.89 75.09 75.63 69.89 77.45 77.78 DKD (Zhao et al., 2022b) 71.97 74.81 75.44 70.35 76.45 77.07 ReviewKD++ (Wang et al, 2024) 72 Table 3: Object Detection task: Comparison with other detectors on COCO, with student models distilled from a pre-trained ViDT-base. Note that DTO-KD consistently outpeforms all challenging knowledge distillation baseline approaches.
small, with just 61M parameters, outperforms Swin-base (0.1B parameters) when both are trained from scratch. Additionally, DTO-KD-tiny, with 38M parameters, achieves nearly the same performance as Swin-small (61M parameters).
this section cite: ['b15', 'b32', 'b21', 'b22', 'b9', 'b33', 'b11', 'b2', 'b40']

Section: ABLATION STUDIES

this section cite: []

Section: Impact of different components in DTO-KD:
We conduct a thorough evaluation of the impact of each primary component in DTO-KD, specifically assessing both stages of dynamic trade-off optimization, and post processing using gradient clipping. These components were introduced to enhance the knowledge distillation process, and their individual contributions are analyzed in Table 4.
The results demonstrate that each stage significantly contributes to the performance of DTO-KD, with all showing a positive effect on the overall effectiveness of the model. Dynamic Trade-off opti- Table 5: Distillation from different teachers for the Object Detection task: Comparison of ViDT on COCO2017 val set. We report AP for the student models distilled from different teacher models.  mization enables the model to handle diverse objectives, and alignment between teacher and student models to facilitate smoother knowledge transfer.
this section cite: []

Section: Dynamic Balancing Strategy and π values:
To better illustrate our approach, the figure below shows the varying weighting ratios of the distillation loss (π distill ) and the task losses (π task ) during training. As illustrated in (a), we evaluate DTO-KD on a classification task using three distinct loss terms, demonstrating its ability to dynamically balance these objectives through adaptive weighting. In (b), we extend this analysis to object detection with two loss terms, where DTO-KD's gradient-based vector optimization initially prioritizes the distillation loss and progressively shifts focus toward the task-specific loss.
this section cite: []

Section: Subtask error analysis:
We conduct a thorough analysis of both classification and localization errors (Bolya et al., 2020) in the object detection task. DTO-KD outperforms other methods, achieving fewer errors in both areas while maintaining a strong balance between them. Notably, other KD techniques (Roy Miles & Deng, 2024;Song et al., 2022) underperform in the classification subtask compared to the baseline (Song et al., 2021), highlighting the superior effectiveness of our approach. See Figure 4 for more details.
Distillation from different teachers: Table 5 demonstrates DTO-KD's strong performance, even with smaller teachers like ViDT-small. This highlights its robustness, adaptability, and efficiency in resource-constrained settings, making it a versatile and effective distillation method across different teacher model scales.
this section cite: ['b0', 'b28', 'b32', 'b31']

Section: LIMITATIONS
Like other KD methods, data availability is a bottleneck. DTO-KD is designed for distillation with available data, and extending it to data-free settings, especially for distilling from large pre-trained models, remains an open challenge. Extending DTO-KD to a data-free regime through sample synthesis may be more difficult due to its min-max optimization, which requires data for the training.
this section cite: []

Section: CONCLUSION
DTO-KD introduces a principled and effective solution to longstanding challenges in knowledge distillation, particularly for transformer-based architectures. By dynamically balancing task-specific and distillation objectives at the gradient level, DTO-KD mitigates supervision conflicts and gradient imbalances that arise from architectural mismatches between teacher and student models. This multi-objective formulation enables more stable and efficient training, resulting in student models that not only match but often exceed the performance of their non-distilled counterparts. Extensive evaluations on image classification and object detection benchmarks demonstrate that DTO-KD consistently achieves state-of-the-art results, setting a new standard for gradient-aware distillation methods. These improvements come with minimal computational overhead, making DTO-KD practical for real-world deployment.
this section cite: []

Section: References
Ref_id:b0 Title: Tide: A general toolbox for identifying object detection errors Year: (2020)
Ref_id:b1 Title: Wasserstein Contrastive Representation Distillation Year: (2020)
Ref_id:b2 Title: Distilling Knowledge via Knowledge Review Year: (2021)
Ref_id:b3 Title: Dearkd: Data-efficient early knowledge distillation for vision transformers Year: (2022)
Ref_id:b4 Title: GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks Year: (2018)
Ref_id:b5 Title:  Year: (2018)
Ref_id:b6 Title: Learning efficient vision transformers via fine-grained manifold distillation Year: (2022)
Ref_id:b7 Title: A comprehensive overhaul of feature distillation Year: (2019)
Ref_id:b8 Title: A comprehensive overhaul of feature distillation Year: (2019)
Ref_id:b9 Title: Distilling the Knowledge in a Neural Network Year: (2015)
Ref_id:b10 Title: Less or more from teacher: Exploiting trilateral geometry for knowledge distillation Year: (2024)
Ref_id:b11 Title: Knowledge distillation from a stronger teacher Year: (2022)
Ref_id:b12 Title: Knowledge diffusion for distillation Year: (2024)
Ref_id:b13 Title: Multi-task learning using uncertainty to weigh losses for scene geometry and semantics Year: (2018)
Ref_id:b14 Title: Multiple-objective problems: Pareto-optimal solutions by method of proper equality constraints Year: (1976)
Ref_id:b15 Title: Microsoft COCO: Common objects in context Year: (2014)
Ref_id:b16 Title: Conflict-averse gradient descent for multi-task learning Year: ()
Ref_id:b17 Title: Famo: Fast adaptive multitask optimization Year: (2023)
Ref_id:b18 Title: Towards impartial multi-task learning Year: ()
Ref_id:b19 Title: Understanding the role of the projector in knowledge distillation Year: (2019)
Ref_id:b20 Title: Improved Knowledge Distillation via Teacher Assistant: Bridging the Gap Between Student and Teacher Year: (2020)
Ref_id:b21 Title: Relational knowledge distillation Year: (2019)
Ref_id:b22 Title: Probabilistic knowledge transfer for lightweight deep representation learning Year: (2020)
Ref_id:b23 Title: Automatic differentiation in pytorch Year: (2017)
Ref_id:b24 Title: Better teacher better student: Dynamic prior knowledge for knowledge distillation Year: (2023)
Ref_id:b25 Title: Designing Network Design Spaces Year: ()
Ref_id:b26 Title: Co-advise: Cross Inductive Bias Distillation Year: (2022)
Ref_id:b27 Title: FitNets: Hints For Thin Deep Nets Year: (2015)
Ref_id:b28 Title: Vkd : Improving knowledge distillation using orthogonal projections Year: (2024-03)
Ref_id:b29 Title: Multi-task learning as multi-objective optimization Year: (2018)
Ref_id:b30 Title: Maskedkd: Efficient distillation of vision transformers with masked images Year: (2024)
Ref_id:b31 Title: Vidt: An efficient and effective fully transformer-based object detector Year: (2021)
Ref_id:b32 Title: Vidt: An efficient and effective fully transformer-based object detector Year: (2022)
Ref_id:b33 Title:  Year: (2019)
Ref_id:b34 Title: Training data-efficient image transformers & distillation through attention Year: (2021)
Ref_id:b35 Title: Going deeper with image transformers Year: (2021)
Ref_id:b36 Title: Deit iii: Revenge of the vit Year: (2022)
Ref_id:b37 Title: Balancing forget quality and model utility: A reverse kl-divergence knowledge distillation approach for better unlearning in llms Year: ()
Ref_id:b38 Title: Abkd: Pursuing a proper allocation of the probability mass in knowledge distillation via α-β-divergence Year: ()
Ref_id:b39 Title: Proceedings of the 42nd International Conference on Machine Learning (ICML 2025 Year: (2025-07)
Ref_id:b40 Title: Crosskd: Cross-head knowledge distillation for object detection Year: (2024)
Ref_id:b41 Title: Improving knowledge distillation via regularizing feature direction and norm Year: (2024)
Ref_id:b42 Title: f-divergence minimization for sequencelevel knowledge distillation Year: (2023)
Ref_id:b43 Title: Knowledge distillation meets selfsupervision Year: (2020)
Ref_id:b44 Title: Hierarchical self-supervised augmented knowledge distillation Year: (2021)
Ref_id:b45 Title: From knowledge distillation to self-knowledge distillation: A unified approach with normalized loss and customized soft labels Year: (2023)
Ref_id:b46 Title: A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning Year: (2017)
Ref_id:b47 Title: Gradient surgery for multi-task learning Year: (2020)
Ref_id:b48 Title: Deep Mutual Learning Year: (2018)
Ref_id:b49 Title: Decoupled knowledge distillation Year: (2022)
Ref_id:b50 Title: Decoupled knowledge distillation Year: (2022)
Ref_id:b51 Title: Knowledge distillation based on transformed teacher matching Year: (2024)
Ref_id:b52 Title: Channel Distillation: Channel-Wise Attention for Knowledge Distillation Year: (2020)
