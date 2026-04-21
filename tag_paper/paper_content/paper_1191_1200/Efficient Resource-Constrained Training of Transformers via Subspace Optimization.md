Title: EFFICIENT RESOURCE-CONSTRAINED TRAINING OF TRANSFORMERS VIA SUBSPACE OPTIMIZATION
Abstract: As AI increasingly shapes daily life, energy consumption and data privacy have become pressing concerns. On-device learning trains models directly on edge devices, cutting energy consumption and safeguarding data privacy. However, the expanding scale of modern neural networks creates a major obstacle for ondevice training. Although prior work has concentrated on compact convolutional architectures, we instead apply subspace-based training to transformer models. Motivated by the idea that a model's essential information lies in a fixed subspace, we introduce Weight-Activation Subspace Iteration (WASI), a method that mitigates the memory bottleneck of backpropagation and boosts inference efficiency in transformer models by restricting training to this subspace. Our results demonstrate that WASI maintains accuracy comparable to vanilla training while reducing memory usage by up to 62× and computational cost (FLOPs) by up to 2×. On a Raspberry Pi 5, WASI achieves roughly 1.4× faster training and inference than vanilla training. The code is available at https://github.com/Le- TrungNguyen/ICLR2026-WASI.git. RELATED WORKSIn this section, we review low-rank decomposition techniques as applied to two key components of deep learning models: model weights and activation maps. Other research directions such as compact model design, quantization, sparsification, and knowledge distillation also exist, but they fall outside the scope of this work-low-rank decomposition (Cheng et al., 2017;Deng et al., 2020). Therefore they are not discussed here (see Appendix A.5 for details).Low-rank Decomposition for Model Weights. Low-rank approximation methods for model weights have been extensively studied and can generally be categorized into two main approaches: Low-rank Adapters and Low-rank Models. LoRA (Hu et al., 2022) is the most prominent example of the first category, which introduces an additional low-rank adapter while freezing the original model architecture. This strategy can reduce the number of trainable parameters by up to four orders of magnitude, but comes with two notable drawbacks. During training, memory usage grows because both the frozen weights and the new

Section: 
On-device learning has recently emerged as a promising research direction, enabling deep learning models to be finetuned directly on resource-constrained edge devices. This approach addresses critical issues such as privacy and energy consumption, improves scalability, and places control of AI capabilities directly "in user's hands" (Dhar et al., 2021). Prior work on on-device learning has largely focused on vision tasks using convolutional neural network models, primarily because of their compact architectures (Lin et al., 2022;Nguyen et al., 2024;Yang et al., 2023b;Quélennec et al., 2024;Bragagnolo et al., 2022;Nguyen et al., 2025).
In many real-world applications, however, transformer-based models have become the de facto choice due to their unique architectural mechanisms (Vaswani et al., 2017). Specifically, these models employ efficient forward propagation through the composition of linear layers, process large-scale data in parallel, and alleviate the vanishing gradient problem thanks to self-attention -key advantages that make them well-suited for handling long-range dependencies, whether in extended text sequences or high-resolution images. Notable examples of such models include GPT (Brown et al., 2020), Gemini (Team et al., 2023), LLaMA (Touvron et al., 2023), and DeepSeek (Liu et al., 2024a). Nevertheless, these mechanisms make training and deployment of transformer models resource-intensive. This is even worse when considering the ondevice learning context, where models need to be trained on separate edge devices and are often resource-constrained.
A significant fraction of training costs arises from backpropagation, especially the memory and computations needed for storing tensors in model layers (Lin et al., 2022). Various research has emerged to address the inefficiencies of backpropagation and enable learning directly on devices. For instance, Lin et al. (2022) demonstrated the feasibility of fine-tuning a predefined subnetwork under a 256KB memory constraint device while still maintaining competitive performance. Quélennec et al. (2024) took this further by dynamically adapting the subnetwork during training rather than relying on a static one, leading to better accuracy within tight memory budgets. Beyond the scope of on-device learning, many methods aim to reduce training overhead through parameter-efficient approaches, such as LoRA (Hu et al., 2022) and its variants (Xu et al., 2023;Zhang et al., 2023;Hayou et al., 2024;Liu et al., 2024b). While these techniques successfully limit the number of parameters updated at training time, they often overlook the cost of storing intermediate calculations (activation maps). Nguyen et al. (2024) address this by compressing activation maps under a controlled information-loss constraint, but lack robust memory budget control and incur considerable compression overhead.
None of these methods enhances the neural architecture itself, and inference proceeds as usual, resulting in high deployment costs on edge devices. This issue has been further addressed by ASVD (Yuan et al., 2023) and FWSVD (Hsu et al., 2022), which employ truncated Singular Value Decomposition (SVD) to decompose the model architecture, but lack a theoretical basis for choosing which singular values to truncate. Subsequently, SVD-LLM (Wang et al., 2024) was developed to overcome this limitation and outperforms the aforementioned approaches. However, these methods are specifically designed for large language models (LLMs) and are not readily applicable to all vision transformer-based models (see Appendix. A.4). Another similar effort, ESPACE (Sakr & Khailany, 2024), requires access to a downstream dataset, which is not feasible in on-device learning scenarios.
Inspired by prior studies on the stability of parameter subspaces during fine-tuning (Radiya-Dixit & Wang, 2020;Li & Zhang, 2021), we present WASI (Fig. 1), the first method for efficient modelactivation-decomposition-aware training. WASI enables transformer models to be fine-tuned and executed entirely in a low-rank representation, substantially reducing hardware costs and making vision transformer tasks feasible on edge devices. We assess its effectiveness on vision transformer models, including the Swin Transformer (SwinT) (Liu et al., 2021), the Vision Transformer (ViT) (Dosovitskiy et al., 2020), and even TinyLlama (Zhang et al., 2024).
Our main contributions are summarized as follows.
• Based on the previous studies, we formulate that the essential information of a model parameters resides in a stable subspace throughout fine-tuning (Sec. 3.3), which is then verified in Sec. 4.
this section cite: ['b7', 'b16', 'b23', 'b26', 'b1', 'b24', 'b33', 'b2', 'b30', 'b31', 'b16', 'b16', 'b26', 'b12', 'b39', 'b45', 'b10', 'b18', 'b23', 'b43', 'b11', 'b37', 'b28', 'b27', 'b15', 'b19', 'b8', 'b46']

Section: • Leveraging this hypothesis, we propose Weight-Activation Subspace Iteration (WASI) in
Sec. 3.3 to effectively compress the model architecture under a controlled information-loss constraint.
• We showcase the effectiveness of our approach through extensive experiments on multiple tasks (Sec. 4.3 and Sec. 4.4).
adapter must co-exist in memory. At inference time, the adapter is merged back into the model, resulting in inference performance that is identical to the original model, and thus losing the computational advantages of low-rank decomposition. Low-rank Models are an alternative line of research that factorizes the weight matrices themselves and trains only the low-rank components, enabling inference to run directly on the compressed representation. Methods such as ASVD (Yuan et al., 2023) and FWSVD (Hsu et al., 2022) achieve this by applying truncated SVD to each layer. These approaches, however, lack a theoretical link between the truncation loss and model performance loss, which is latter addressed by SVD-LLM (Wang et al., 2024). It is important to note that, except for SVD-LLM, all aforementioned methods are specifically tailored for LLMs, and even SVD-LLM cannot be directly applied to all vision transformer-based models with activation maps of four or more dimensions (see Appendix A.4).
this section cite: ['b43', 'b11', 'b37']

Section: Low-rank Decomposition for Activation Maps.
In addition to model weights, activation maps are a major contributor to memory consumption during training. Gradient Filter (Yang et al., 2023b) is a pioneering work that addresses this issue in on-device learning by generating approximated versions of activation maps through pooling operations with a predefined patch size, aiming to reduce memory usage and FLOPs during fine-tuning. However, this method is limited to convolutional models, and also has the drawback of the accumulated errors as fine-tuning progresses deeper into the model (Nguyen et al., 2024). To overcome this drawback, Nguyen et al. (2024) introduced Activation Map Compression (AMC), which applies High-Order Singular Value Decomposition (HOSVD) to compress activation maps while controlling the information loss via a threshold parameter ε. While AMC achieves impressive memory savings up to 120×, it incurs significant computational overhead due to the need for full HOSVD at every iteration. Additionally, the varying ranks required to meet the error threshold lead to fluctuating memory usage, which complicates deployment on devices with fixed memory budgets. Activation Subspace Iteration (ASI) (Nguyen et al., 2025) addresses both of these issues. Instead of controlling the reconstruction error, ASI fixes the activation ranks using a perplexity-based heuristic. This approach stabilizes memory usage throughout fine-tuning and allows for replacing the expensive HOSVD with subspace iteration. As a result, ASI preserves the high compression ratio of AMC while reducing computational cost by up to 252.65×. On a Raspberry Pi 5, fine-tuning with ASI is 1.56× faster than vanilla training when being tested on a highly compact convolutional model. Beyond this scope, LBP-WHT (Yang et al., 2023b) has also been explored. However, it focuses solely on reducing computational cost during training by applying the Walsh-Hadamard Transformation to tensors in gradient computations, and does not address memory bottlenecks.
Our proposed WASI overcomes the limitations posed by prior works. Hypothesizing the stability of the essential subspace of model weights, we introduce a novel method that simultaneously compresses the model architecture and activation maps while carefully controlling information loss throughout the fine-tuning process. This capability makes it feasible to fine-tune transformer-based models in on-device learning scenarios.
this section cite: ['b23', 'b23', 'b24']

Section: METHOD
In this section, we first identify the computational bottlenecks of training and inference (Sec. 3.1).
Next, we review how activation maps can be efficiently compressed (Sec. 3.2). We then introduce a compression-aware-training strategy for both model weights and activation maps that controls information loss (Sec. 3.3). Finally, we analyze the computational complexity of our method and discuss its practical advantages (Sec. 3.4).
this section cite: []

Section: BOTTLENECKS IN TRAINING AND INFERENCE
Consider a deep transformer-based model, where i denotes the index of a linear layer. This layer is represented by a weight matrix W i ∈ R Oi×Ii , which takes as input a tensor A i ∈ R B×Ni×Ii and produces an output tensor A i+1 ∈ R B×Ni×Oi . Here, B is the batch size, N i is the sequence length (or number of tokens), I i is the input feature dimension, and O i is the output feature dimension. We denote the dimensionality of the input as D i = {B, N i , I i }.
During the forward pass (similarly in inference), the output of this layer is computed as:
A i+1 = A i W ⊤ i ,(1)
Algorithm 1 Weight Subspace Iteration -WSI at iteration t 1: Input: Weight W i,(t) at iteration t, Explained variance threshold ε ∈ [0, 1]. 2: Function: 3: if t = 0 then 4: L i,(t) , R i,(t) = SVD W i,(t) , ε (see Eq. 5, Eq. 6, and Eq. 7) 5: else 6: R T i,(t) = W T i,(t) • L i,(t-1) 7:
L i,(t) = Orthogonalize W i,(t) • R T i,(t) (Using Gram-Schmidt) 8: endif 9: return L i,(t) , R i,(t)
where ⊤ denotes the matrix transpose. Eq. 1 presents a batch matrix multiplication applied over the last two dimensions of A i ; that is, for each sample in the batch and each token, a matrix multiplication is performed between a 1 × I i vector and the transposed weight matrix of size I i × O i .
Similarly, in the backward pass the chain rule of backpropagation is computed as follows:
∂L ∂W i = ∂L ∂A i+1 ⊤ • ∂A i+1 ∂W i = ∂L ∂A i+1 ⊤ • A i ,(2)
∂L ∂A i = ∂L ∂A i+1 • ∂A i+1 ∂A i = ∂L ∂A i+1 • W i ,(3)
where L is the loss computed at the output of the model. Apparently, to compute ∂L ∂Wi and ∂L ∂Ai during the backward pass, A i and W i must be stored during the forward pass. The large size of these tensors is the primary cause of memory bottlenecks during backpropagation (Lin et al., 2022). Additionally, it also contributes to high inference costs, as multiplying between large W i and A i requires significant computational resources.
this section cite: ['b16']

Section: ACTIVATION SUBSPACE ITERATION
Here, we recap how activation maps can be decomposed by subspace iteration. Given an activation memory budget B, ASI performs brute-force optimization before fine-tuning to find an optimal rank vector r i ∈ N 3 for each layer such that the resulting memory does not exceed B. Then, for each mode m ∈ {1, 2, 3}, the activation map A i is unfolded into a matrix A i,m ∈ R ai,m×bi,m , where Vogels et al. (2019) showed that warm-started subspace iteration matches SVD performance on stable tensors at much lower cost. Exploiting the stability of activation maps during fine-tuning, ASI applies this technique to each A i,m . The resulting approximation takes the form of a Tucker decomposition (Tucker, 1966):
(a i,m , b i,m ) = D i,m , j̸ =m D i,j .
A i ≈ Si × 1 Ũ (1) i × 2 Ũ (2) i × 3 Ũ (3) i ,(4)
where Si ∈ R ri,1×ri,2×ri,3 is the core tensor, representing a compressed version of A i , and each factor matrix Ũ (m) i ∈ R ai,m×ri,m contains the principal components along the m th mode.
Consequently, instead of storing all Θ space 3
m=1 D i,m elements of A i , ASI reduces the storage requirement to Θ space 3 m=1 r i,m + 3 m=1 D i,m r i,m .
Details of the algorithm can be found in Appendix A.2.
this section cite: ['b35', 'b32']

Section: WEIGHT -ACTIVATION SUBSPACE ITERATION

this section cite: []

Section: Stability of Model Parameters Subspace.
While prior work has shown that over-parameterized models in fact reside in a low-dimensional intrinsic subspace (Aghajanyan et al., 2020
K i = r i,1 = r i,2 = r i,3 D i,m = O i = 32 | m ∈ [1, 3] D i,m = O i = 64 | m ∈ [1, 3] D i,m = O i = 128 | m ∈ [1, 3] D i,m = O i = 256 | m ∈ [1, 3]
Figure 2: For the linear layer i with a single data batch of size B, given varying dimensions of W i and A i and different values of r i,m , C training and C inference illustrate the evolution in compression rates for training and inference, respectively; while S training and S inference forecast the speedup ratios for these processes.
2018), we further observe that fine-tuning introduces only minor updates at each training step due to the use of a small learning rate. As a result, our key insight is that the intrinsic subspace remains relatively stable after each training iteration and can therefore be reused in the following one (confirmed in Sec. 4.2 -Fig. 3). This is supported by the findings of Radiya-Dixit & Wang (2020) and Li & Zhang (2021), who showed that the fine-tuned models are close in parameter space to the pre-trained counterpart.
this section cite: ['b0', 'b27', 'b15']

Section: Weight Subspace Iteration.
Besides activation maps, model parameters (weights) W i are another major source of memory bottlenecks during training. To address this, we propose a low-rank weight decomposition strategy that projects each weight tensor into a smaller subspace at every training iteration, thereby preserving the meaningful subspace. The method works as follows:
Step 1. For the weight tensor W i at layer i, its SVD form is given by:
W i = U i Σ i V T i , U i ∈ R Oi×Oi , Σ i ∈ R Oi×Ii , V i ∈ R Ii×Ii ,(5)
where Σ i is a diagonal matrix containing r i singular values s i,j∈ [1,ri] , and r i is the rank of W i .
As shown in Eq. 3, truncating U i , Σ i , and V T i inevitably introduces error into ∂L ∂Ai , which then propagates backward during training. In other words, low-rank decomposition of the weights affects model convergence due to the accumulation of truncation error. To control this effect, we constrain the truncation error by enforcing a target explained variance threshold ε, similar to the strategy used in Nguyen et al. (2024). Specifically, we measure the variance explained by the j th singular value as σ 2 i,j = s 2 i,j / k s 2 i,k . Assuming the singular values are sorted in descending order (s i,j ≥ s i,k , ∀j ≤ k), the optimal rank is defined as the smallest integer
K i ∈ [1, r i ] such that Ki j=1 σ 2 i,j ≥ ε.
We then identify the essential subspace with rank K i of W i , represented by L i and R i such that:
W i ≈ Wi = L i R i ,(6)
where
L i = U i,(Ki) Σ i,(Ki) , R i = V T i,(Ki) | U i,(Ki) ∈ R Oi×Ki , Σ i,(Ki) ∈ R Ki×Ki , V i,(Ki) ∈ R Ii×Ki . (7) Step 2.
Performing full SVDs at every iteration, however, is computationally prohibitive for ondevice training (Nguyen et al., 2025). Leveraging the stability of parameter subspaces established above, Σ i can be expected to remain relatively stable. Thus, for a fixed ε, the optimal rank K i should also remain consistent (verified in Sec. 4.2). Consequently, instead of recomputing the SVD at every iteration, we compute it once at the beginning to determine the essential subspace. Subspace iteration is applied during training to minimize computational overhead. We refer to this method as Weight Subspace Iteration (WSI), with the full procedure outlined in Algorithm 1.
this section cite: ['b23', 'b24']

Section: Weight-Activation Subspace Iteration.
While WSI reduces weight-related overhead, activation maps also dominate memory usage in backpropagation (Sec. 3.1). Previous work has shown that most of the energy in activation maps is concentrated in the first few principal components across all modes (Nguyen et al., 2024). Such a distribution makes them highly compressible while still achieving high-fidelity reconstruction (confirmed in Sec. 4.2 -Fig. 4 and Sec. 4.3). Motivated by this property, we propose a unified framework in which both weights and activations are compressed under stable low-rank subspaces. Specifically, we redesign ASI with two improvements: (i) a dynamic-programming strategy that determines r i by minimizing memory usage under a target pre-tuning perplexity, rather than relying on a fixed budget B, thereby reducing the search cost from exponential to linear (Appendix A.2); and (ii) an extension to support 3D activation tensors (Appendix A.1). Together, WSI and ASI form the proposed Weight-Activation Subspace Iteration (WASI), a novel framework for low-rank training that jointly leverages the stability of both weights and activations. Under this scheme, the forward and backward passes are computed as follows:
A i+1 = A i R T i L T i ,(8)
∂L ∂W i = f LR Ãi , ∂L ∂A i+1 , (9
)
∂L ∂A i = ∂L ∂A i+1 • L i R i ,(10)
where f LR (.) denotes a linear operator applied in the low-rank space (see Appendix A.1). With learning rate η, the weight update is then computed as:
L i R i = L i R i + η • ∂L ∂W i .(11)
this section cite: ['b23']

Section: MEMORY EFFICIENCY AND COMPUTATIONAL COMPLEXITY ANALYSIS
For simplicity, we assume that the same optimal rank is applied to both Ai and Wi. By varying this value, we can predict total memory usage and speedup for WASI compared to vanilla training (Fig. 2). As model size grows and the optimal rank decreases, WASI delivers greater memory compression (C training , C inference ) and speedup (S training , S inference ), a property especially valuable in on-device learning where models are typically over-parameterized and reside in low-dimensional subspaces (Aghajanyan et al., 2020;Li et al., 2018). Conversely, as the optimal rank increases, WASI's computational cost approaches that of vanilla training, and the speedup ratios converge to 1, reflecting the upper bound set by vanilla training.
Detailed derives of C training , C inference , S training , and S inference can be found in Appendix A.3.
this section cite: ['b0', 'b14']

Section: EXPERIMENTS
In this section, we present experiments designed to demonstrate the effectiveness of WASI. We begin by outlining the experimental setup in Sec. 4.1.
Then, in Sec. 4.2, we conduct experiments to validate the assumptions introduced in Sec. 3.3 and Sec. 3.3. Sec. 4.3 compares WASI with various state-of-the-art methods across multiple datasets. Finally, all methods are evaluated in a real-world deployment scenario (Sec. 4.4. All simulation experiments are conducted using PyTorch 1.13.1 on an NVIDIA Quadro RTX A4500 with 20 GB of VRAM, while on-device experiments are run on a Raspberry Pi 5 equipped with a Cortex-A76 CPU and 8 GB of RAM.
this section cite: []

Section: EXPERIMENTAL SETUP
Our goal is to enable on-device training of transformer models, where networks pretrained on largescale datasets are fine-tuned locally with task-specific data (Murshed et al., 2021). We evaluate WASI on image classification using ViT and SwinT, both pretrained on ImageNet-1K (Deng et al., 2009), across five downstream datasets: CIFAR-10/100 (Krizhevsky, 2009), CUB (Wah et al., 2011), Flowers (Nilsback & Zisserman, 2008), and Pets (Zhang et al., 2022).
Comparisons are made against three directly comparable baselines at the time of conducting experi ments: ASI, SVD-LLM, and vanilla training (as discussed in Secs. 1, 2, Appendix A.5). We measure memory and computation costs during training and inference, focusing on linear layers within multi-perceptron blocks for fair comparison with previous methods (extended results with attention layers in Appendix B.3). All experiments are run with the same set of hyperparameters, detailed in Appendix B.1. L a y e r In d e x 0 10 20 S in g u la r V a lu e In d e x 0 50 100 Explained Variance 0.0 0.2 0.4 0.6 Mode 1 (a) L a y e r In d e x 0 10 20 S in g u la r V a lu e In d e x 0 20 40 Explained Variance 0.0 0.2 0.4 0.6 Mode 2 (b) 0 10 20 L a y e r In d e x 0 200 400 600 800 S in g u la r V a lu e In d e x 0.0 0.2 0.4 0.6 Explained Variance Mode 3 (c) Figure 4: Explained variance of each singular value of A i across all of its modes when fine-tuning ViT on the Pets dataset.
this section cite: ['b22', 'b5', 'b13', 'b36', 'b25', 'b44']

Section: PRELIMINARY RESULTS
In these experiments, we focus on fine-tuning ViT model using Pets dataset.
Stability of Layer Ranks. We apply truncated SVD to the weight tensors of the linear layers within ViT's MLP blocks at each training iteration. We constrain the decomposition by setting ε = 0.8 and monitor the layer ranks K i throughout the course of training. As shown in Fig. 3a, we observe that the ranks exhibit remarkable stability across epochs. This observation validates our insight in Sec. 3.3, confirming the stability of layer ranks during training.
this section cite: []

Section: WSI vs SVD.
Next, we compare two strategies: (1) reapplying truncated SVD at every training iteration, and (2) WSI. We evaluate their performance across a range of ε values -specifically, 0.4, 0.5, 0.6, 0.7, 0.8, and 0.9 -with each value represented by a different marker in Fig. 3b. The results demonstrate that incorporating subspace iteration through WSI leads to a significant reduction in computational complexity compared to performing a full SVD at every iteration. Specifically, WSI requires 1.36× fewer FLOPs than SVD to achieve the same level of accuracy. Moreover, when both methods are constrained to use the same amount of FLOPs, WSI outperforms SVD by approximately 35% in terms of accuracy. This result verifies that reusing the subspace in subsequent training iterations does not degrade model convergence.
this section cite: []

Section: Explained Variance Distribution of Activation Maps.
Fig. 4 illustrate the explained variances σ i,j,m of each singular value j in mode m of the activation map A i . As anticipated in Sec. 3.3, most activation-map energy lies in the first few singular values, which capture the key information during fine-tuning.
10 2 10 3 Training Mem (MB) 80 90 Accuracy (%) 10 2 Inference Mem (MB) 80 90 10 11 10 12 Training FLOPs 80 90 10 11 Inference FLOPs 80 90 WASI ASI SVD-LLM Vanilla WASI ASI SVD-LLM Vanilla WASI ASI SVD-LLM Vanilla WASI ASI SVD-LLM Vanilla 10 2 10 3 Training Mem (MB) 60 80 100 Accuracy (%) 2 × 10 1 3 × 10 1 4 × 10 1 6 × 10 1 Inference Mem (MB) 60 80 100 10 12 6 × 10 11 2 × 10 12 Training FLOPs 60 80 100 2 × 10 11 3 × 10 11 4 × 10 11 6 × 10 11 Inference FLOPs 60 80 100 Pets Flowers CUB CIFAR-10 CIFAR-100 Pets Flowers CUB CIFAR-10 CIFAR-100 Pets Flowers CUB CIFAR-10 CIFAR-100 Pets Flowers CUB CIFAR-10 CIFAR-100 Figure 6: Resource consumption when applying WASI for fine-tuning and inference of SwinT across different datasets. Each marker along the curves represents a different compression rate, while the final marker on each curve corresponds to vanilla training.
this section cite: []

Section: MAIN RESULTS
ViT on CIFAR-10. Fig. 5 presents the results of fine-tuning a ViT pretrained on ImageNet-1K using CIFAR-10. Each curve for WASI and ASI contains six markers, corresponding to explained variance thresholds ε ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 0.9} from left to right. The red diamond indicates vanilla training, and for fairness, the same compression ratios are applied to SVD-LLM. WASI achieves up to 100× higher memory efficiency than SVD-LLM at similar accuracy, owing to its avoidance of LoRA adapters. Its accuracy also improves steadily as ε increases. In contrast, at the lowest compression rates (last two markers), SVD-LLM consumes even more memory than vanilla training because of the overhead of storing sub-layer activations.
In terms of computation, LoRA adapters allow SVD-LLM to achieve the lowest FLOPs, followed by WASI, which jointly compresses weights and activations into a low-rank subspace. Since ASI only compresses activations while keeping weights intact, its computational cost is higher, and at ε = 0.9, it even exceeds vanilla training (confirmed in Tab. 2). On the other hand, ASI maintains stable accuracy across compression rates, supporting the stability assumption discussed in Sec. 3.3. At inference, both WASI and SVD-LLM achieve similar memory/FLOPs savings, while ASI resembles vanilla since the architecture is unchanged.
SwinT on Multiple Datasets. WASI on TinyLlama. The initial goal of WASI was to enable training transformer-based models on edge devices, so we focused on ViT and SwinT. To test its generality, we extended our experiments to TinyLlama, a decoder-only transformer model. The downstream dataset used is BoolQ (Clark et al., 2019). Due to limited resources, we only fine-tune up to the last 5 layers of the model and set the WASI ε to 0.1. All other training hyperparameters followes the same configuration as in our previous experiments. For comparison, we log the resource consumption only at the layers that are fine-tuned. The results are shown in Fig. 7.
10 0 10 2 Activation Mem (MB) 64 66 Accuracy (%) 10 1 10 2 Weight Mem (MB) 64 66 10 10 10 11 Training FLOPs 64 66 10 9 10 10 10 11 Inference FLOPs 64 66 WASI Vanilla WASI Vanilla WASI Vanilla WASI Vanilla 4.4 ON-DEVICE LATENCY 0.4 0.5 0.6 0.7 0.8 0.9 vanilla ε 0 5 10 15 20 Time (s) Training time Inference time Figure 8: Training and inference time per iteration for ViT on CIFAR-10 (batch size = 128) using a Raspberry Pi 5, measured under different explained variance thresholds ε. The final marker on each curve represents vanilla training.
We evaluate the practical efficiency of WASI on resource-constrained hardware by fine-tuning ViT on CIFAR-10 using a Raspberry Pi 5. Fig. 8 reports the average time required to complete a single iteration of both training and inference across different explained variance thresholds ε ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, along with vanilla training. As expected, the runtime for both training and inference under WASI increases as ε becomes larger. This trend aligns with the intuition that higher ε values retain more information and thus result in higher-rank approximations, which require more compute and memory. However, despite this increase, WASI consistently outperforms vanilla training in terms of speed. For instance, even at ε = 0.9, which corresponds to the least aggressive compression setting in this experiment, WASI remains approximately 1.4× faster than vanilla training. Thus, WASI delivers clear benefits even when preserving much of the original information. Importantly, WASI helps to reduce runtime without causing significant accuracy degradation, as discussed in earlier sections. This ability makes it a strong candidate for the deployment of transformerbased model in real-world on-device learning scenarios, where computational resources are severely constrained. Further numerical results can be found in Appendix. B.3.
this section cite: ['b4']

Section: CONCLUSION
In this work, we introduced WASI, an efficient training method for resource-constrained finetuning of transformer models. Assuming that essential parameter information lies in a stable lowdimensional subspace, WASI applies SVD and subspace iteration to obtain low-rank approximations of both weights and activations during each training iteration. This yields significant gains in memory and computation while tightly controlling information loss. Building on prior theory and validated through extensive experiments, WASI outperforms stateof-the-art methods, reducing training memory usage by up to 62× and achieving 1.4× speedup over vanilla training on a Raspberry Pi 5. These results show the potential of WASI for enabling on-device learning with transformers, a domain traditionally dominated by CNNs. While our experiments focus on transformers, the underlying principles apply broadly to any neural network trained with backpropagation.
this section cite: []

Section: References
Ref_id:b0 Title: Intrinsic dimensionality explains the effectiveness of language model fine-tuning Year: (2020)
Ref_id:b1 Title: To update or not to update? neurons at equilibrium in deep models Year: (2022)
Ref_id:b2 Title: Language models are few-shot learners Year: (2020)
Ref_id:b3 Title: A survey of model compression and acceleration for deep neural networks Year: (2017)
Ref_id:b4 Title: Exploring the surprising difficulty of natural yes/no questions Year: (2019)
Ref_id:b5 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b6 Title: Model compression and hardware acceleration for neural networks: A comprehensive survey Year: (2020)
Ref_id:b7 Title: A survey of on-device machine learning: An algorithms and learning theory perspective Year: (2021)
Ref_id:b8 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b9 Title: Convolutional neural network compression through generalized kronecker product decomposition Year: (2022)
Ref_id:b10 Title: Lora+: Efficient low rank adaptation of large models Year: (2024)
Ref_id:b11 Title: Language model compression with weighted low-rank factorization Year: (2022)
Ref_id:b12 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b13 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b14 Title: Measuring the intrinsic dimension of objective landscapes Year: (2018)
Ref_id:b15 Title: Improved regularization and robustness for fine-tuning in neural networks Year: (2021)
Ref_id:b16 Title: On-device training under 256kb memory Year: (2022)
Ref_id:b17 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b18 Title: Weight-decomposed low-rank adaptation Year: (2024)
Ref_id:b19 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b20 Title: Structured low-rank approximation and its applications Year: ()
Ref_id:b21 Title:  Year: (2008)
Ref_id:b22 Title: Machine learning at the network edge: A survey Year: (2021)
Ref_id:b23 Title: Activation map compression through tensor decomposition for deep learning Year: (2024)
Ref_id:b24 Title: Beyond low-rank decomposition: A shortcut approach for efficient on-device learning Year: (2025)
Ref_id:b25 Title: Automated flower classification over a large number of classes Year: (2008)
Ref_id:b26 Title: Towards on-device learning on the edge: Ways to select neurons to update under a budget constraint Year: (2024)
Ref_id:b27 Title: How fine can fine-tuning be? learning efficient language models Year: (2020)
Ref_id:b28 Title: Espace: Dimensionality reduction of activations for model compression Year: (2024)
Ref_id:b29 Title: Methods of simultaneous iteration for calculating eigenvectors of matrices Year: (1975)
Ref_id:b30 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b31 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b32 Title: Some mathematical notes on three-mode factor analysis Year: (1966)
Ref_id:b33 Title: Attention is all you need. Advances in neural information processing systems Year: (2017)
Ref_id:b34 Title: Lipschitz regularity of deep neural networks: analysis and efficient estimation Year: (2018)
Ref_id:b35 Title: Powersgd: Practical low-rank gradient compression for distributed optimization Year: (2019)
Ref_id:b36 Title: The caltech-ucsd birds Year: (2011)
Ref_id:b37 Title: Svd-llm: Truncation-aware singular value decomposition for large language model compression Year: (2024)
Ref_id:b38 Title: Low rank optimization for efficient deep learning: Making a balance between compact architecture and fast training Year: (2023)
Ref_id:b39 Title: Qa-lora: Quantization-aware low-rank adaptation of large language models Year: (2023)
Ref_id:b40 Title: Restructuring of deep neural network acoustic models with singular value decomposition Year: (2013)
Ref_id:b41 Title: Efficient low-rank backpropagation for vision transformer adaptation Year: (2023)
Ref_id:b42 Title: Efficient on-device training via gradient filtering Year: (2023)
Ref_id:b43 Title: Asvd: Activation-aware singular value decomposition for compressing large language models Year: (2023)
Ref_id:b44 Title: 0/1 deep neural networks via block coordinate descent Year: (2022)
Ref_id:b45 Title: Lora-fa: Memory-efficient low-rank adaptation for large language models fine-tuning Year: (2023)
Ref_id:b46 Title: Tinyllama: An open-source small language model Year: (2024)
Ref_id:b47 Title: Semi-tensor product-based tensordecomposition for neural network compression Year: (2021)
