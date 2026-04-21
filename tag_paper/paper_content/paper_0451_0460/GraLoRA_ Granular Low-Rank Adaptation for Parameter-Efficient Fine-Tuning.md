Title: GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning
Abstract: Low-Rank Adaptation (LoRA) is a popular method for parameter-efficient finetuning (PEFT) of generative models, valued for its simplicity and effectiveness. Despite recent enhancements, LoRA still suffers from a fundamental limitation: overfitting when the bottleneck is widened. It performs best at ranks 32-64, yet its accuracy stagnates or declines at higher ranks, still falling short of full fine-tuning (FFT) performance. We identify the root cause as LoRA's structural bottleneck, which introduces gradient entanglement to the unrelated input channels and distorts gradient propagation. To address this, we introduce a novel structure, Granular Low-Rank Adaptation (GraLoRA) that partitions weight matrices into sub-blocks, each with its own low-rank adapter. With negligible computational or storage cost, GraLoRA overcomes LoRA's limitations, effectively increases the representational capacity, and more closely approximates FFT behavior. Experiments on code generation, commonsense reasoning, mathematical reasoning, general language understanding, and image generation benchmarks show that GraLoRA consistently outperforms LoRA and other baselines, achieving up to +8.5% absolute gain in Pass@1 on HumanEval+. These improvements hold across model sizes and rank settings, making GraLoRA a scalable and robust solution for PEFT.

Section: Introduction
Task-specific fine-tuning enables a wide range of applications and significantly improves the quality and effectiveness of generative models. However, the massive scale of these models poses substantial challenges for practical deployment. To address these limitations, Parameter-Efficient Fine-Tuning (PEFT) methods have emerged as a cost-effective alternative [12,32]. Among them, Low-Rank Adaptation (LoRA) [13] has gained particular attention for its simplicity and effectiveness, introducing trainable low-rank matrices while keeping the pre-trained model weights frozen. Although the imposed rank-r bottleneck may lead to slight performance degradation compared to full fine-tuning (FFT), its efficiency has led to widespread adoption in practice.
To maximize the benefits of LoRA, various studies have proposed techniques such as improved initialization [4,21,23,29] and structural refinements [10,15,16,17] to enhance fine-tuning quality. While these efforts have advanced performance, a substantial quality gap remains compared to FFT, largely due to the inherent upper bound on the rank. Although using a higher rank, within hardware limits, appears to be a natural solution, unfortunately, current implementations of LoRA and its variants do not support such flexibility. Simply increasing the rank often leads to degraded accuracy in many scenarios.
In this paper, we present a theoretical analysis identifying the root cause of the rank limitation in LoRA. Our analysis reveals a fundamental issue in LoRA's structure, channel dominance in the gradient, where a small subset of outlier channels disproportionately influences the update direction. This dominance suppresses contributions from other channels, leading to under-utilization of the available rank and degraded performance in tasks that require nuanced or distributed representations.
To overcome these expressivity bottlenecks, we propose Granular Low-Rank Adaptation (GraLoRA), a novel architectural extension of LoRA. As shown in Figure 1, GraLoRA divides the weight matrix into multiple sub-blocks and applies independent LoRA modules to each, enabling fine-grained updates. This design enhances the model's capacity to capture complex, localized, or multi-faceted patterns, effectively mitigating the channel dominance issue and improving performance-especially at higher ranks.
Extensive experiments show that GraLoRA consistently outperforms vanilla LoRA across a range of NLP benchmarks, particularly in scenarios with high input heterogeneity or task complexity. These results position GraLoRA as a principled and practical advancement in the PEFT landscape.
2 Details and Limitations of LoRA
this section cite: ['b10', 'b30', 'b11', 'b3', 'b19', 'b21', 'b27', 'b8', 'b13', 'b14', 'b15']

Section: Introduction to LoRA
LoRA is one of the most widely adopted strategies for PEFT. Given a pre-trained weight matrix W 0 ∈ R M ×N , where M and N represent the input and output channel dimension, respectively, LoRA keeps W 0 frozen and introduces a trainable low-rank update defined as:
R = sBA ⊤ , A ∈ R N ×r , B ∈ R M ×r , s = α r .(1)
Here, rank r and α are user-defined hyperparameters. Then, for a given input X ∈ R N ×T , the output of the LoRA-adapted layer is Y = W 0 X + RX ∈ R M ×T , where T denotes the batch or token dimension. This low-rank decomposition allows the model to adapt using significantly fewer trainable parameters and reduced memory overhead.
While FFT updates the entire weight matrix, LoRA only updates the decomposed low-rank matrices A and B. Note that we assume s = 1 for simplicity, the gradient of the loss with respect to R is:
∇ R L = ∂L ∂R = ∂L ∂Y X ⊤ ∈ R M ×N(2)
From this, the gradients with respect to the LoRA parameters B and A are given by:
∂L ∂B = ∂L ∂Y X ⊤ A, ∂L ∂A ⊤ = B ⊤ ∂L ∂Y X ⊤ .(3)
Figure 2: Gradient dynamics of FFT and LoRA in the presence of an outlier input channel. The red channel in input X denotes the outlier. While FFT localizes the gradient impact, LoRA's entire gradient update becomes disproportionately influenced by the single outlier.
These result in the following reconstructed update in the fused weight space:
∇R L = ∂L ∂B A ⊤ + B ∂L ∂A ⊤ = ∂L ∂Y X ⊤ AA ⊤ + BB ⊤ ∂L ∂Y X ⊤ . (4
)
This expression reveals how the structure of LoRA introduces non-trivial interactions between the gradients and the input, particularly through the rank-r matrices.
this section cite: []

Section: Why Does LoRA Suffer from a Larger Rank?
When fine-tuning with a large LoRA rank (e.g., r > 64), it is often observed that accuracy degrades compared to using a moderate rank. This counterintuitive behavior arises from the distinct gradient dynamics of LoRA, which differ significantly from those of FFT.
LoRA's structural design makes its gradients inherently sensitive to the entire input space, as illustrated in Figure 2. In particular, we observe that outlier channels, input channels with abnormally high activations, can disproportionately dominate the gradient signal.
In FFT, the effect of such outliers is typically localized, affecting only a single column of the weight matrix W that directly interacts with the outlier channel. In contrast, LoRA's low-rank constraint causes the entire gradient of the adapter matrix B, denoted ∂L/∂B, to be influenced by these outliers. This results in distorted weight updates in the fused weight space, where the gradient signal from outlier channels overwhelms the contributions from other inputs. Consequently, LoRA fails to accurately replicate the gradient dynamics of FFT, limiting its ability to match FFT-level performance.
We observe that in certain layers, most notably the down-projection matrix of Layer 1 in LLaMA3.1-8B, input activations exhibit severe channel-wise imbalance (Figure 3 (a)). As shown in Figure 4, these outlier channels disproportionately impact the adapter's gradient updates. Figure 3 further illustrates that the gap between LoRA and FFT gradient updates widens as the LoRA rank increases.
These findings reveal a fundamental misalignment between LoRA updates and the gradient landscape shaped by FFT. The entangled influence of input channels caused by the low-rank projection limits LoRA's ability to selectively learn from salient features, particularly under skewed input statistics. While the negative impact of outliers has been well recognized in the context of quantization [31] [18], their influence on LoRA's behavior has not been systematically studied until now.
this section cite: ['b29']

Section: Method

this section cite: []

Section: GraLoRA: Granular Low-Rank Adaptation
Motivated by observation in previous section, we propose GraLoRA, a fine-grained and modular extension of LoRA. As illustrated in Figure 1, GraLoRA addresses the limitations of standard LoRA by partitioning the weight matrix into a grid of k × k independent blocks, each equipped with its own local low-rank adapter. Here, k is a hyperparameter that determines the number of splits along the input and output dimensions. When k = 1, GraLoRA reduces to the vanilla LoRA formulation.
Specifically, the weight update R ∈ R M ×N is expressed as the concatenation of block-wise updates:
R GraLoRA =    B 1,1 A ⊤ 1,1 • • • B 1,k A ⊤ 1,k . . . . . . . . . B k,1 A ⊤ k,1 • • • B k,k A ⊤ k,k    , A i,j ∈ R N k × r k , B i,j ∈ R M k × r k (5
)
This block-wise reparameterization provides localized control over each spatial subregion of the parameter space. As detailed in Section 3.4, GraLoRA incurs the same parameter count and computational overhead as standard LoRA when using the same rank. However, it introduces two key advantages; (1) Enhanced Expressivity and (2) Robustness to Input Outliers. By enabling independent adaptation across k 2 subspaces, GraLoRA supports more fine-grained and specialized feature learning. In addition, Localized gradient updates ensure that only the adapters associated with the affected input regions receive large gradients, thereby reducing global gradient distortion and preserving inter-channel signal balance.
this section cite: []

Section: Expression Power Analysis
While the weight update of GraLoRA was expressed as concatenation of block-wise updates in ( 5), it can also be regularized as the form of multiplication of two matrices as in the vanilla LoRA. The sparse matrix A GraLoRA ∈ R N ×kr can be constructed as Figure 5 (a), where A i,j for i, j ∈ {n ∈ N | n ≤ k} is located in position (i + (j -1) × k, j) of A GraLoRA . Other elements are masked out, thus the total number of parameter becomes N × r.
Then, B GraLoRA ∈ R N ×kr is constructed as Figure 5 (b), where matrix B i,j for i, j ∈ {n ∈ N | n ≤ k} is located in position (i, j + (i -1) × k) of B GraLoRA , Similarly, other composition of the matrix is masked, therefore the total number of parameter becomes M × r. Then the weight update of GraLoRA can be expressed as
W = W 0 + R GraLoRA = W 0 + B GraLoRA A ⊤ GraLoRA .
Assuming that all columns of [B i,1 , • • • , B i,k ] are linearly independent, the rank of B GraLoRA becomes R (B GraLoRA ) = kr. Similarly, if all columns of [A 1,j , • • • , A k,j ] are linearly independent, the rank of A GraLoRA is R (A GraLoRA ) = kr. Applying Sylvester's rank inequality to derive the lower bound and the matrix product theorem for the upper bound, we obtain:
R(B GraLoRA ) + R(A ⊤ GraLoRA ) -kr ≤ R(B GraLoRA A ⊤ GraLoRA ) ≤ min(R(B GraLoRA ), R(A ⊤ GraLoRA )) (6
) Thus, the effective rank of R GraLoRA becomes kr, which is k times higher than that of the vanilla LoRA method-effectively enhancing the model's expressive capacity. The rank analysis of finetuned LoRA and GraLoRA, summarized in Table 5 in Appendix, demonstrates that GraLoRA linearly scales the representational power of the adaptation matrix in practical settings.
this section cite: []

Section: Gradient Dynamics Under Outlier Activation
GraLoRA effectively localizes the influence of outlier channels to a limited subset of adapter blocks. Because each block processes only a specific slice of the input, only the k adapter pairs intersecting with the outlier channel are exposed to amplified gradients. In contrast, the remaining k 2 -k adapters maintain gradient magnitudes close to baseline levels. This selective gradient propagation resembles the behavior of FFT, where only weights directly connected to active inputs are significantly updated.
GraLoRA's impact on gradient dynamics can be observed by comparing gradient distributions of the down-projection matrix in Layer 1 with standard LoRA. As illustrated in the Figure 3 (c) and Figure 6, GraLoRA reduces the gradient deviation and limits the influence of outlier channels, overcoming the limitations of standard LoRA with larger ranks.
this section cite: []

Section: Tradeoff Analysis
As discussed, GraLoRA provides several advantages over standard LoRA. However, these benefits do not come without cost. In this section, we provide deeper analysis on the overhead introduced by GraLoRA.
Computation Overhead Analysis: First, we analyze the expected computational cost of LoRA in terms of FLOPs. To take advantage of the low-rank structure, LoRA computes the projection in two sequential steps. The first computes A ⊤ X ∈ R r×T , followed by the reconstruction B(A ⊤ X) ∈ Figure 6: Comparison of gradient distributions under outlier activation. In GraLoRA, only the blocks interacting with the outlier exhibit elevated gradients, mitigating global distortion and aligning with FFT behavior. R M ×T . These steps require 2N rT and 2rM T FLOPs, respectively, resulting in a total complexity of O (r(M + N )T ) .
Similarly, GraLoRA divides the computation into two steps involving k 2 adapter blocks. In the first step, the projection computes A ⊤ i,j X j ∈ R r k ×T for each of the k 2 blocks, incurring a total cost of
2• N k • r k •T •k 2 = 2N rT. In the second step, each intermediate output is processed by its corresponding B i,j , producing B i,j (A ⊤ i,j X j ) ∈ R M k ×T . This step adds another 2 • r k • M k • T • k 2 = 2rM T.
FLOPs to the total cost. Hence, the overall computational cost of GraLoRA remains O (r(M + N )T ), maintaining efficiency comparable to vanilla LoRA while significantly enhancing expressive power. A detailed analysis of computational overhead is provided in Appendix C. Memory Overhead Analysis: As with classical LoRA, GraLoRA can be merged into the original weight matrix at inference time. Therefore, our analysis focuses on the memory overhead incurred during training. Although the number of parameters and FLOPs are identical to those of LoRA, the intermediate latent representation A ⊤ GraLoRA X becomes k times larger than the corresponding A ⊤ X in standard LoRA. This expanded latent space allows for greater information preservation, which can be beneficial. However, it also leads to increased memory consumption during training time. Fortunately, the rank r is typically much smaller than the input and output dimensions, thus the additional memory required remains marginal-even for large k, as demonstrated in Table 1. Moreover, by applying recent techniques such as gradient checkpointing, the memory overhead from the expanded latent space can be effectively hidden, making the impact negligible in practice.
this section cite: []

Section: Selection of k
While GraLoRA increases the total rank from r to kr, each individual block, represented as B i,j
A ⊤ i,j ∈ R M k × N k
, is constrained to a reduced rank of r k . As a result, increasing k beyond a certain threshold can degrade performance due to limited expressiveness within each block. This effect is especially pronounced when the overall rank r is small. Empirically, we observed that maintaining a minimum block expressiveness of approximately r/k 2 ≈ 8 yields stable performance across various configurations. Based on this observation, we adopted k = 2 for ranks 16 and 32, and k = 4 for ranks 64 and 128 in our experiments. Detailed k-sweep results can be found in Section 4.7.
this section cite: []

Section: Hybrid GraLoRA
On the other hand, for smaller ranks-typically rank 16 or below-using k = 2 may still lead to performance degradation or yield only marginal gains. To address this limitation, we introduce a hybrid approach that combines the strengths of LoRA and GraLoRA. This method retains the fine-grained input handling and increased total rank offered by GraLoRA, while preserving the expressive power of larger block units through LoRA. Since LoRA shares the same parameters across both rows and columns, it can be naturally integrated with GraLoRA in a concatenated form, which we refer to as Hybrid GraLoRA (see Figure 7). Empirically, we found that allocating up to 1  2 of the total rank to the LoRA component mitigated the limitations of GraLoRA in low-rank scenarios (γ <= 16), while fully allocating the rank to GraLoRA better performed in high-rank circumstances.
this section cite: []

Section: Experiments
In order to validate the superiority of the proposed idea, we conduct an extensive analysis on large-scale dataset with the state-of-the art LLMs. We evaluate GraLoRA across five challenging domains: code generation, commonsense reasoning, mathematical reasoning, general language understanding, and personalized image generation. Our experiments are designed to assess whether the proposed granular adaptation mechanism improves performance across varying model sizes, LoRA ranks, and tasks that require nuanced reasoning and high representational fidelity.
this section cite: []

Section: Experimental Setup
Code Generation. We fine-tuned LLaMA3.1-8B ( [9]) with 4 A100 80G GPU on the Magicoder-Evol-Instruct-110k [30] train dataset, a curated and decontaminated subset of WizardCoder [20], comprising high-quality instruction-response pairs for programming tasks. Evaluation was conducted on the Humaneval+ test dataset following He et al. [10], which samples 50 completions per problem using a temperature of 0.2. We report Pass@1, Pass@5, and Pass@10 accuracy following standard protocol via BigCode Evaluation Harness [1].
Commonsense Reasoning We fine-tuned LLaMA3.2-3B, LLaMA3.1-70B, Qwen-2.5-1.5B, and Qwen-2.5-7B ( [33]) across 8 commonsense tasks: BoolQ [6], PIQA [3], SIQA [27], HellaSwag [35], WinoGrande [26], ARC-Challenge, ARC-Easy [7], and OpenBookQA [22]. We followed the training pipeline proposed by LLM-Adapters [14]. Training was performed on 2 H100 80G GPUs for 1.5-8B models, and on 8 A100 80G GPUs for the 70B model. LLaMA3.1-70B, Qwen-2.5-1.5B, and Qwen-2.5-7B were trained with rank 64, using the optimal configurations proposed by Biderman et al. [2]. LLaMA3.2-3B was trained with rank 32, following the settings of Ponkshe et al. [25] to ensure a fair comparison with results reported in the original paper.
this section cite: ['b7', 'b28', 'b18', 'b8', 'b0', 'b31', 'b4', 'b2', 'b25', 'b33', 'b24', 'b5', 'b20', 'b12', 'b1', 'b23']

Section: Mathematical Reasoning
We fine-tuned LLaMA3.2-3B on MetaMathQA [34] train dataset using 4 H100 80G GPUs. Evaluation was done on MATH [11] dataset, following the evaluation procedure and settings from He et al. [10].
this section cite: ['b32', 'b9', 'b8']

Section: General Language Understanding
We trained and evaluated RoBERTa-base [19], an encoderonly architecture model, on the GLUE [28] benchmark composed of eight sub-tasks. Following the protocol from prior works ( [17] [8]), we excluded MNLI and QQP-two time-intensive tasks-which also meant we did not apply the MNLI-based tricks for MRPC, RTE, and STS-B (as used in the original LoRA paper). Accordingly, we retrained LoRA on these tasks without this optimization and report updated results. All trainings were done on a single H100 80G GPU. Personalized Image Generation We fine-tuned SDXL [24] following the official training setup from Huggingface diffusers repository, using the Naruto-Blip-Captions [5] dataset on a single H100 80G GPU. The dataset was split 90% for training and 10% for evaluation. The quality was measured through CLIP similarity and DINOv2 similarity scores.
this section cite: ['b17', 'b26', 'b15', 'b22']

Section: Training Details
We conducted experiments on five open-sourced LLMs-LLaMA3.1-8B, LLaMA3.1-70B, LLaMA3.2-3B, Qwen-2.5-1.5B, and Qwen-2.5-7B-covering diverse architecture and sclaes across code generation, commonsense reasoning, and mathematical reasoning tasks. Following common practice ( [16,17]), we used pre-trained models rather than instruction-tuned models.
All PEFT methods were applied to the linear modules in both the attention ( W q , W k , W v , W o )and the feed-forward networks (W up , W down , W gate ). We adopted alpaca-chat instruction template for training and evaluation. We compared GraLoRA to three representative PEFT methods: LoRA, MoRA [16] and RaSA [10]. We have also handled RoBERTa-base and SDXL, to show the robustness and scalability of our method across differnt models and tasks. Hyperparameters for GraLoRA followed those introduced in Kopiczko et al. [17], except for learning rate, which was reduced by a factor of 5-10, as VeRA uses a learning rate approximately 10 times larger than LoRA. Detailed training parameters can be found in Appendix E.
this section cite: ['b14', 'b15', 'b14', 'b8', 'b15']

Section: Results on Code Generation
As shown in Table 2, GraLoRA outperformed LoRA, MoRA, and RaSA across all tested ranks for Pass@1 accuracy. At rank 64, GraLoRA achieved an absolute improvement of +2.4% in Pass@1, +4.8% in Pass@5, and +4.1% in Pass@10 over LoRA. At rank 128, the gains were even more pronounced, with increases of +8.5% in Pass@1, +6.9% in Pass@5, and +5.1% in Pass@10. Notably, while other methods struggled to fully utilize the increasing rank capacity-often reaching performance plateaus at lower ranks-GraLoRA maintained a consistent upward trajectory, effectively overcoming the limitations of LoRA.
Even in low-rank settings (e.g., rank 16), where expressive capacity is typically constrained, the hybrid variant of GraLoRA demonstrated superior performance. These improvements highlight GraLoRA's enhanced capability to preserve diverse gradient signals and resist suppression from dominant outliers. The strong results on the HumanEval+ benchmark further underscore the benefits of fine-grained adaptation in tackling complex, high-precision code generation tasks.
this section cite: []

Section: Results on Commonsense Reasoning
As shown in Table 3, GraLoRA outperformed other methods across a wide range of models and tasks. Notably, GraLoRA demonstrated superior performance across models of varying scales, achieving a 1.1% improvement in average accuracy on both Qwen2.5-1.5B and LLaMA3.1-70B. It also yielded a 0.9% gain on the widely used mid-sized model, Qwen2.5-7B. Moreover, GraLoRA achieved a 3.3% improvement on LLaMA3.2-3B, surpassing a broad range of baselines as presented in Table 6.
Furthermore, GraLoRA achieved the best results on 26 out of 32 tasks, consistently outperforming alternatives across benchmarks. These results support our analysis in Section 3.3, showing that GraLoRA's localized updates enhance alignment with FFT and promote robust generalization in multi-aspect reasoning tasks.
this section cite: []

Section: Results on Mathematical Reasoning
In mathematical reasoning task, regarded as one of the most challenging benchmarks, GraLoRA consistently outperformed LoRA across all configurations. Notably, in the high rank setting of r = 128, GraLoRA achieved a 4.2% improvement in accuracy (Table 4), mirroring the performance trends observed in the code generation experiments. These results further highlight the robustness of GraLoRA, demonstrating its capability to fully exploit the advantages enabled by increased rank capacity, thereby overcoming the inherent expressiveness constraints of previous PEFT methods.
this section cite: []

Section: Results on General Language Understanding
GraLoRA demonstrates strong performance even in the low-rank regime, outperforming all baselines in terms of average score. The Hybrid GraLoRA variant achieves the most robust results, attaining the best performance on four out of six tasks, while both the original and hybrid versions consistently surpass all other baselines, as shown in Table 7. Compared with LoRA, the best GraLoRA configuration yields a 1.8% improvement in average accuracy, with gains observed across all sub-tasks. These findings indicate that GraLoRA maintains high effectiveness even under constrained parameter budgets and generalizes well to non-LLM architectures.
this section cite: []

Section: Results on Personalized Image Generation
In the image generation task, GraLoRA consistently outperformed LoRA in both CLIP and DINOv2 similarity metrics, achieving 0.5% and 2.1% improvements, respectively (Table 8). These results further demonstrate the generality and effectiveness of GraLoRA beyond language models, extending its applicability to vision-language and generative architectures such as diffusion models.
this section cite: []

Section: Ablation Study
GraLoRA k Sweep We evaluated the impact of varying k on code generation accuracy. As shown in Figure 8 (a), k = 2 yielded the best performance at rank 32, while k = 4 was optimal at rank 128. These results are consistent with the theoretical prediction that a smaller k is preferable for lower ranks, as reduced sub-block rank can be particularly detrimental when the overall rank is limited.
this section cite: []

Section: Hybrid GraLoRA Ratio Sweep
We assessed performance across different LoRA-to-GraLoRA rank allocation ratios for the Hybrid GraLoRA configuration (Figure 8 (b)). At rank 16, partially allocating the rank to LoRA led optimal accuracy. However, for larger ranks, allocating rank to LoRA resulted in degraded performance. This suggests that Hybrid GraLoRA is advantageous in low-rank regimes, where the sub-block rank of GraLoRA alone may be insufficient. In contrast, under higher-rank settings where GraLoRA's sub-blocks are expressive enough, introducing LoRA components may lead to gradient entanglement, thereby hindering effective learning.
this section cite: []

Section: Conclusion
In this work, we introduced GraLoRA, a novel PEFT method that extends LoRA with granular, blockwise decomposition. Motivated by a rigorous analysis of LoRA's gradient behavior, we identified that input outliers can dominate the low-rank update, suppressing meaningful contributions from other input channels and misaligning with the localized gradient propagation observed in FFT.
GraLoRA addresses this limitation by dividing the adaptation space into k 2 independently trained low-rank adapters, enabling spatially localized and context-aware updates. Our theoretical analysis shows that this design increases expressivity by a factor of k, without additional parameters or computational cost. Moreover, under outlier activations, GraLoRA effectively mitigates the global gradient distortion seen in vanilla LoRA and better preserves inter-channel balance. Empirically, GraLoRA consistently outperforms standard LoRA and strong baselines such as RaSA across diverse tasks and model scales. On the code generation benchmark HumanEval+, it achieves up to +8.5% absolute gain in Pass1. GraLoRA also delivers significant improvements across other 4 additional tasks, highlighting its robustness and scalability across heterogeneous architectures and model sizes.
Future Work. While GraLoRA improves gradient locality and expressive power, its current design assumes uniform partitioning. Future extensions may explore adaptive or learned partitioning schemes, sparsity-aware block activation, or task-driven dynamic rank allocation. Additionally, applying GraLoRA to vision transformers, multimodal architectures, or continual learning setups may further highlight its potential for robust and efficient model adaptation.
Overall, GraLoRA represents a principled and practical step forward in the design of PEFT methods, bridging the gap between global low-rank reparameterization and local, fine-grained adaptation. In the previous "Computation Overhead Analysis" section 3.4 we compared the computation of LoRA and GraLoRA with the big O notation on the two major matrix multiplication steps. In this section we further examine the exact computation requirement and compare their efficiency.
this section cite: []

Section: References
Ref_id:b0 Title: A framework for the evaluation of code generation models Year: (2022)
Ref_id:b1 Title: LoRA learns less and forgets less Year: (2024)
Ref_id:b2 Title: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b3 Title: Olora: Orthonormal low-rank adaptation of large language models Year: (2024)
Ref_id:b4 Title: BoolQ: Exploring the surprising difficulty of natural yes/no questions Year: (2019-06)
Ref_id:b5 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b6 Title: Parameter-efficient fine-tuning with discrete fourier transform Year: (2024)
Ref_id:b7 Title: The llama 3 herd of models Year: (2024)
Ref_id:b8 Title: RaSA: Rank-sharing low-rank adaptation Year: ()
Ref_id:b9 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b10 Title: Parameter-efficient transfer learning for nlp Year: (2019)
Ref_id:b11 Title: Low-rank adaptation of large language models Year: (2021)
Ref_id:b12 Title: Llm-adapters: An adapter family for parameter-efficient fine-tuning of large language models Year: (2023)
Ref_id:b13 Title: HiRA: Parameter-efficient hadamard high-rank adaptation for large language models Year: ()
Ref_id:b14 Title: High-rank updating for parameter-efficient fine-tuning Year: (2024)
Ref_id:b15 Title: Vector-based random matrix adaptation Year: ()
Ref_id:b16 Title: Owq: Outlieraware weight quantization for efficient fine-tuning and inference of large language models Year: (2024)
Ref_id:b17 Title: A robustly optimized bert pretraining approach Year: (2019)
Ref_id:b18 Title: Wizardcoder: Empowering code large language models with evol-instruct Year: (2024)
Ref_id:b19 Title: Pissa: Principal singular values and singular vectors adaptation of large language models Year: (2024)
Ref_id:b20 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018)
Ref_id:b21 Title: One initialization to rule them all: Fine-tuning via explained variance adaptation Year: (2024)
Ref_id:b22 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b23 Title: Initialization using update approximation is a silver bullet for extremely efficient low-rank fine-tuning Year: (2025)
Ref_id:b24 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b25 Title: Commonsense reasoning about social interactions Year: (2019)
Ref_id:b26 Title: Glue: A multi-task benchmark and analysis platform for natural language understanding Year: (2019)
Ref_id:b27 Title: Lora-ga: Low-rank adaptation with gradient approximation Year: (2024)
Ref_id:b28 Title: Magicoder: Empowering code generation with OSS-instruct Year: (2024-07)
Ref_id:b29 Title: Smoothquant: Accurate and efficient post-training quantization for large language models Year: (2024)
Ref_id:b30 Title: Parameter-efficient fine-tuning methods for pretrained language models: A critical review and assessment Year: (2023)
Ref_id:b31 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b32 Title: Bootstrap your own mathematical questions for large language models Year: (2023)
Ref_id:b33 Title: Hellaswag: Can a machine really finish your sentence Year: (2019)
