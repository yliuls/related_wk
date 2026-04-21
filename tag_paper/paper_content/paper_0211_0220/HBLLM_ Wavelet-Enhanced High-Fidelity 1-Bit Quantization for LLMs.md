Title: HBLLM: Wavelet-Enhanced High-Fidelity 1-Bit Quantization for LLMs
Abstract: We introduce HBLLM, a wavelet-enhanced high-fidelity 1-bit post-training quantization method for Large Language Models (LLMs). By leveraging Haar wavelet transforms to enhance expressive capacity through frequency decomposition, HBLLM significantly improves quantization fidelity while maintaining minimal overhead. This approach features two innovative structure-aware grouping strategies: (1) frequency-aware multi-parameter intra-row grouping and (2) ℓ 2 -normbased saliency-driven column selection. For non-salient weights, a shared mean is employed across quantization groups within each frequency band to optimize storage efficiency. Experiments conducted on the OPT and LLaMA models demonstrate that HBLLM achieves state-of-the-art performance in 1-bit quantization, attaining a perplexity of 6.71 on LLaMA2-13B with an average weight storage of only 1.08 bits. Code available at: https://github.com/Yeyke/HBLLM.

Section: Introduction
Figure 1: Average relative perplexity (normalized to FP16) on PTB, WikiText2, and C4 for LLaMA-1 family models, comparing LLM binarization methods and our HBLLM.
In recent years, Large Language Models (LLMs) have achieved remarkable progress in natural language processing tasks. However, their massive parameter sizes-often reaching tens or even hundreds of billions-pose significant challenges for deployment on edge devices and in low-resource environments. To reduce the computational and memory burden of these models, a variety of compression techniques have been proposed, including quantization [12,33,35], pruning [11,31], and knowledge distillation [19,30]. Among them, Post-Training Quantization (PTQ) is widely adopted for its efficiency, requiring no additional training and having low deployment cost, especially in 1-bit quantization, which is considered a key approach for achieving extreme inference efficiency [13].
Although existing 1-bit PTQ methods [15,17,34] have achieved some success on base models such as GPT-2 and OPT, they tend to suffer from significant performance degradation-or even complete failure-when applied to more complex modern architectures like LLaMA3-8B [16]. To address this, recent studies have introduced several strategies to improve quantization fidelity:
• Group quantization: divides the weight matrix into multiple groups for separate quantization. For instance, outlier-aware partitioning handles critical columns independently but can be constrained by partition design and scalability [15]; • Residual approximation: adds residual terms on top of primary quantization to partially recover errors [6], though this provides limited fidelity gains and introduces extra computation; • Low-Rank Adaptation (e.g., LoRA): inserts low-rank modules to absorb quantization errors with some flexibility, like [34], but often shows sensitivity to hyperparameters; • Global orthogonal transformations: apply global rotations in [1,2,5] before model compression to enhance representational capacity, but require expensive inverse transforms (e.g., matrix multiplications at O(d 2 ) complexity for a d-dimension linear layer), leading to increased inference latency and energy consumption, making them impractical for deployment.
To overcome the structural trade-off between expressiveness and efficiency, we propose a novel 1-bit PTQ framework-HBLLM. This method is the first to integrate localized orthogonal transformations (i.e., Haar wavelets) into a BiLLM-style quantization process. Combined with structure-aware grouping, HBLLM significantly enhances expressive power under ultra-low bit budgets while maintaining negligible inverse transform cost and excellent compatibility with hardware-efficient inference.
Our main contributions are as follows:
• A localized orthogonal transformation mechanism: we apply a single Haar wavelet transform to decompose the weight matrix into high-and low-frequency components, improving binary expressiveness while reducing transform computation; • Frequency-aware multi-parameter intra-row grouping: we introduce intra-row grouping in the frequency domain to capture structural patterns; • ℓ 2 -norm-based saliency-driven column selection: we propose an ℓ 2 norm-based ranking method to retain key columns using saliency metrics, effectively reducing quantization error; • Intra-frequency-band mean sharing: for non-salient components, we introduce a mechanism that shares the mean across groups within the same row and wavelet band, reducing storage without sacrificing fidelity.
We conduct extensive experiments on OPT [37], LLaMA family [32] of LLMs. Results show that HBLLM achieves state-of-the-art performance under 1-bit quantization: Across language modeling tasks (C4, PTB, WikiText2), the perplexity ratio between HBLLM and the original FP16 model remains within the range of 1.2-2.2, shown in Fig 1, outperforming the next-best methods by 33%-66%; On 9 zero-shot QA benchmarks, HBLLM retains 73.8%-88.8% of the original model's accuracy; On modern architectures such as LLaMA3-8B, HBLLM remains stable with no performance collapse; Even with a lower average bit rate and memory usage than BiLLM and ARB-LLM RC [18], HBLLM outperforms both in overall task accuracy.
These results demonstrate that HBLLM significantly extends the applicability of 1-bit quantization, balancing extreme compression with high fidelity, and offers a new paradigm for deploying large-scale language models efficiently.
2 Related Work
this section cite: ['b11', 'b32', 'b34', 'b10', 'b30', 'b18', 'b29', 'b12', 'b14', 'b16', 'b33', 'b15', 'b14', 'b5', 'b33', 'b0', 'b1', 'b4', 'b31', 'b17']

Section: 1-Bit Post-Training Quantization
1-bit PTQ has emerged as a critical promising solution for deploying LLMs under extremely low bit budgets. Representative methods such as BiLLM [15] adopt a salient column separation mechanism, in which salient weights are quantized independently, while non-salient weights are grouped based on magnitude and quantized row-wise. ARB-LLM X [18] further introduces column-wise grouping and alternating refined binarization, achieving notable improvements in fidelity. Unlike [10,34], BiLLM can accomplish PTQ tasks without intensive computation for knowledge distillation with multi-GPUs.
However, current methods face several key limitations: (1) They heavily rely on fixed thresholds or simple ℓ 1 -based heuristics for salient column selection, which are insufficient to capture sparse but significant activation outliers; (2) They fail to account for the structural asymmetry between row and column dimensions in weight matrices, limiting their adaptability to complex model architectures; (3) They completely neglect frequency-domain information.
this section cite: ['b14', 'b17', 'b9', 'b33']

Section: Evolution and Limitations of Grouping Strategies
To improve quantization flexibility and fidelity, some studies have proposed learnable or adaptive grouping strategies. For example, Mixture of Scales [17] introduces a Mixture-of-Experts (MoE) mechanism to assign scaling factor groups, and OneBitGPT [34] uses frequency masks to control quantization range sensitivity, and AWQ [3] identifies weights with the greatest impact on model predictions only based on activation outputs. However, these methods are generally effective only on unstructured tensors, rely on fine-grained distillation, and lack explicit frequency-domain awareness.
In addition, existing grouping strategies [15,18] often apply uniform partitioning rules across the entire weight matrix, ignoring variations across different rows. This can lead to degraded expressiveness when quantizing models with significant inter-row diversity.
this section cite: ['b16', 'b33', 'b2', 'b14', 'b17']

Section: Comparison Between Global Orthogonal Transforms and Local Wavelet Transforms
Orthogonal transforms have recently been adopted to improve LLM quantization. FrameQuant [1] and QuIP [5] utilize orthogonal transforms to enhance fidelity, but inference with such global transforms incurs high overhead, requiring O(d 2 ) matrix multiplications [1] that cannot be fused into linear layers, leading to increased latency and energy cost.
By contrast, local orthogonal transforms such as the Haar wavelet [20] offer localized spectral sensitivity and have been widely applied in image compression, denoising, and edge detection [9,14]. They can be efficiently implemented via lightweight local convolutions with negligible inference cost, making them well-suited for low-bit compression and edge deployment.
this section cite: ['b0', 'b4', 'b0', 'b19', 'b8', 'b13']

Section: HBLLM: A Quantization Framework with Wavelet Transform and Frequency-Domain Grouping

this section cite: []

Section: Motivation and Core Challenges
Current mainstream 1-bit quantization methods face three key challenges in practice: (1) limited numerical expressiveness leading to high reconstruction error; (2) insufficient accuracy in salient column selection, failing to capture critical activation columns; (3) lack of structure-aware grouping strategies that adapt to heterogeneous model structures.
To characterize expressiveness under ultra-low bit settings, we introduce a new metric: the cardinality of the Inverse Quantization Set (CIQ), which measures the size of the discrete set of dequantized values within a row. CIQ serves as a unified indicator of how the above challenges constrain model fidelity. It acts both as a theoretical tool to analyze the limits of existing methods and as empirical evidence of the advantage of our proposed method.
Under 1-bit quantization, the CIQ of BiLLM and ARB-LLM X is 8 and 10, respectively. When block size sets to 128, the CIQ upper bound of ARB-LLM X can reach 128. In contrast, our method achieves a CIQ of up to 1024 after applying the Haar wavelet transform, significantly improving theoretical expressiveness. For more information on the benefits introduced by applying Haar transform, please refer to the appendix B and C.
Based on aboved analysis, we propose: (1) Haar wavelet transform to enhance expressive capacity by frequency decomposition; (2) ℓ 2 -norm-based saliency-driven column selection to prioritize critical columns;
(3) frequency-aware multi-parameter intra-row grouping to capture structural patterns. We also introduce an intra-frequency-band mean sharing strategy and local convolution optimization to reduce storage and inference cost, thus forming a 1-bit PTQ framework HBLLM .
this section cite: []

Section: Method Overview
We define the objective of HBLLM under the binary quantization setting for LLM weights. Specifically, the quantization targets the full-precision weight matrix W FP ∈ R d×d , where a binary diagonal mask matrix M sal ∈ {0, 1} d×d indicates which columns are selected as salient. The salient and . These are then reconstructed using inverse Haar transforms H -1 1 and H -1 2 . The reconstruction objective of HBLLM is twofold. For the quantization of a matrix layer W, the objective expressed in the Frobenius norm is formulated as:
min W WX -WX 2 F , (1
)
where X is the input of the matrix layer. For quantization of a matrix block W FP of W, the object is:
min Msal, W sal B , W non-sal B W FP -M sal H -1 1 W sal B -(I -M sal )H -1 2 W non-sal B 2 F .(2)
When H 1 = H 2 are fixed Haar transforms, this formulation simplifies to a quantization problem entirely in the Haar domain. In this case, the objective is the same to that of BiLLM. Layer-level quantization is commonly tackled with the GPTQ algorithm [12].
We emphasize that our approach does not aim to solve this objective function via explicit optimization.
Instead, this formulation serves as a conceptual framework that guides our method design. The actual quantization process is based on a set of heuristics and structure-aware strategies that approximate this objective in a computationally efficient and scalable manner.
Quantization Pipeline Overview. HBLLM integrates the Haar transform into a BiLLM-style quantization pipeline (see Algorithm 1 and Figure 2), consisting of the following key steps:
1. Preparation Phase: Compute the column-wise importance scores using a Hessian-based saliency metric.
this section cite: ['b11']

Section: Salient Column Selection and Quantization(SALIENT):
• Sort columns by their ℓ 2 norm significance.
• Select top-K salient columns and determine M sal .
• W sal B = HaarQuant (M sal W).
• Choose the subset with the lowest quantization error.
this section cite: []

Section: Non-Salient Region Quantization:
• Fill the missing values in salient columns using adjacent averages (FillAvg).
• W non-sal B = HaarQuant (M sal W filled + (I -M sal ) W), where W filled is from FillAvg. 4. Adjustment and Refinement:
• W = M sal W -H -1 W non-sal B .
• W sal B = HaarQuant W . Algorithm 1 Framework of HBLLM: Details of each function are shown in Algorithm E.1 func HBLLM(W, X, β, λ) Input: W ∈ R n×m -weight matrix X ∈ R r×d -calibration data β -block size λ -hessian regularizer Output: B -haared binarized weights
this section cite: []

Section: HaarQuant: One-Bit Quantization in the Wavelet Domain
To boost expressiveness, we apply Haar wavelet transform to the weight matrix of linear layers, generating a frequency-domain coefficient matrix, followed by group-wise 1-bit quantization.
To address limited numerical expressiveness, HBLLM introduces the HaarQuant algorithm. Haar-Quant consists of three stages.
this section cite: []

Section: Haar Transform.
A row of weights W is decomposed into low-and high-frequency coefficients via 1D Haar transform H:
W = H (W) = [H low-pass (W) , H high-pass (W)] ,(3)
where W is the Haar coefficient of W, H low-pass (W) and H high-pass (W) are low-and high-frequency coefficients, respecively.
Frequency-Aware Multi-Parameter Intra-Row Grouping. For each row, boundary candidates determined by the row are enumerated, and the best grouping with minimal quantization error is selected. Furthermore, we split the rows by frequency bands. This adaptive strategy captures intra-row structural differences better than global uniform boundaries used in BiLLM.
Coefficient Quantization. Each group W FP is quantized using sign-based binarization centered on its mean:
W B = α • sign( W FP -µ),(4)
where α ∈ R d is the row-wise scaling factor and µ is the group-wise mean and W B is the result.
this section cite: []

Section: Structure-aware Grouping Strategies
To enhance the fidelity and adaptability of binary quantization under structural constraints, HBLLM introduces two structure-aware grouping strategies that operate along both column and row dimensions of the weight matrix.
Saliency-Driven Column Selection via ℓ 2 Norm. This strategy is used during salient column identification and quantization to overcome the limitations of prior heuristics based on fixed thresholds or simple magnitude criteria.
• Columns are ranked by their ℓ 2 -norm scores, which correlate with their overall contribution to activation magnitude.
• The top-K columns are selected as salient and quantized in the Haar-transformed domain using column-wise transforms.
This approach helps preserve activation-critical directions, especially those dominated by outlier weights.
Frequency-Aware Multi-Parameter Intra-Row Grouping. This strategy is used during Haar domain quantization, where conventional row grouping lacks sensitivity to structural variations in weight distributions.
• Each row is first decomposed into high-and low-frequency components based on Haar subbands.
• Within each frequency band, coefficients are adaptively split into dense and sparse groups using band-specific, data-driven thresholds.
This grouping effectively doubles the number of quantization subgroups per row, enabling finer granularity and better error control.
Together, these strategies facilitate fine-grained, structure-preserving quantization across both dimensions of the weight matrix. To further guide saliency-based partitioning, we adopt the parameter importance metric used in BiLLM, defined as:
s i = w 2 i /[H -1 ] 2
ii , where H denotes the Hessian matrix of the layer, w i is the full-precision value of the i-th parameter, and [H -1 ] ii is the i-th diagonal entry of the inverse Hessian. This metric reflects the relative sensitivity of the loss to changes in each parameter: higher values indicate greater influence on the model's output, and thus prioritize that weight for accurate reconstruction.
this section cite: []

Section: Intra-frequency-band Mean Sharing
To reduce storage overhead, HBLLM shares a single mean value among 2 groups in the same frequency band within each row:
µ shared = 1 n1+n2 n1 i=1 x i + n2 j=1 y j .
It not only reduces perparameter storage by 0.25 bits, but also maintains accuracy even slightly improving downstream task performance. This optimization achieves a trade-off between compression and accuracy, improving deployment viability.
this section cite: []

Section: Efficient Haar Implementation via Local Convolutions
Instead of costly matrix multiplication, HBLLM implements Haar transform using fixed local convolutions. There are only two predefined 1D kernels, [1/2, 1/2] and [1/2, -1/2], whose kernel size is 2. Furthermore, it can be hardcoded into the model for zero runtime initialization and no training or storage is needed. In complexity comparison, HBLLM needs O(d) operatons via convolutional sliding window, while FrameQuant needs O(d 2 ) operations. As a result, HBLLM significantly lowers inference cost and is ideal for edge deployment.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Settings
Models and Evaluation Datasets. In our study, we evaluate HBLLM on various models, including those from the OPT, LLaMA-1, LLaMA-2, and LLaMA-3, as well as the recently introduced f-R1-Distill-Llama-8B. Specifically, we utilize the OPT models with 1.3B and 2.7B parameters, the LLaMA-1 and LLaMA-2 models with 7B and 13B parameters for our evaluations, and the LLaMA-3 model with 8B parameters. We measure language modeling capabilities of these models by evaluating their perplexity on the C4 [26], WikiText2 [22] and PTB [21] datasets. Additionally, we assess zero-shot accuracy on various Common Sense Reasoning Tasks such as PIQA [4], BoolQ [7], OpenBookQA [23], WinoGrande [28], ARC-e, ARC-c [8], HellaSwag [36], which are commonly used for evaluating the performance of LLM quantization methods. To further enhance evaluation coverage, we also include COPA [27] for causal reasoning and LAMBADA [25] for long-context language modeling. All evaluations are conducted using the open-source LLM evaluation framework, LM-Evaluation-Harness [24].
this section cite: ['b25', 'b21', 'b20', 'b3', 'b6', 'b22', 'b27', 'b35', 'b26', 'b24', 'b23']

Section: Details of Experiments.
All experiments are conducted with PyTorch on NVIDIA GeForce RTX 3090 GPUs with 24GB of memory. For the calibration data, we follow the settings adopted in GPTQ and BiLLM, selecting 128 samples from the C4 dataset, with a sequence length of 2048. During quantization, we set the block size to 128 in BiLLM, PB-LLM, ARB-LLM, and HBLLM. Activations are kept in full precision (FP16).
Baselines. We compare HBLLM against several state-of-the-art LLM binarization methods, including BiLLM, ARB-LLM and PB-LLM, ensuring that all implementations adhere to the details provided in their respective papers. BiLLM, ARB-LLM and PB-LLM all utilize the PTQ approach for model calibration through OBQ based method of GPTQ. For ARB-LLM, we evaluate two of its bestperforming variants, ARB-LLM X and ARB-LLM RC . Both ARB-LLM x and ARB-LLM RC employ the salient column bitmap and group bitmap (CGB) for better performance. For PB-LLM, which allows variable ratios of salient weights to enhance accuracy, we have set the ratio of salient weights to 10% to ensure the average bit width of weight parameters remains below 2 bits. Given the significant accuracy improvements demonstrated by HBLLM over traditional binarization techniques, we also include a comparison with a leading method using orthogonal transforms: FrameQuant. For FrameQuant, quantization is performed not in the original weight space but in the structured orthogonal basis constructed through Fusion Frames. We evaluate two configurations: FrameQuant (r = 1.0) and FrameQuant (r = 1.1), where the redundancy factor r controls the amount of redundancy introduced during the transformation.
this section cite: []

Section: Perplexity and Accuracy Results of 1-2 Bit Quantized Models
The perplexity and zero-shot accuracy results of previous 1-2 bit quantization methods and the proposed HBLLM are presented in Table1. HBLLM consistently outperforms existing 1-2 bit quantization techniques across all evaluation metrics.
Specifically, HBLLM reduces the language modeling perplexity by 33%-66% compared to previous methods, while achieving substantial improvements in QA task accuracy, with relative gains ranging from -0.73% to +11.3%. In our experiments, HBLLM slightly outperforms FrameQuant, a 2.2bit quantization method, and exhibits a particularly significant advantage on the LLaMA-3-8B model. Moreover, when compared with BiLLM and ARB-LLM X , HBLLM-col, demonstrates a clear advantage in both perplexity and accuracy, despite operating at comparable or lower bit-widths. These results indicate that HBLLM effectively narrows the performance gap between quantized models and their Float16 counterparts, achieving 1.22× to 2.48× of the original perplexity and retaining 73.8%-88.8% of the original QA accuracy.
this section cite: []

Section: Ablation Study Salient Column Selection Criterion.
To evaluate the impact of selection criteria in salient column screening on quantization effectiveness, we compare two strategies: the column ℓ 1 norm and the column ℓ 2 norm as significance indicators. Experimental results in Table 2a reveal that the column ℓ 2 norm consistently achieves lower quantization error and superior performance in downstream tasks, indicating its greater effectiveness in capturing energy distribution across columns and enhancing quantization quality.
Granularity of Group Quantization. To explore the influence of grouping granularity on model performance, we compare global grouping with row-wise grouping strategies, evaluating both quantization error and perplexity, as shown in Table 2b. The results reveal that row-wise grouping significantly reduces quantization error and achieves lower perplexity compared to global grouping. This suggests that finer-grained row-wise partitioning better preserves local data fidelity, leading to improved quantized inference performance.
this section cite: []

Section: Shared Mean Strategy.
Under the standard dual-partition quantization setting, we further explore a compression strategy that shares the quantization center across two partitions within each row. By unifying the mean for both partitions, the storage overhead of quantization coefficients can be significantly reduced. Experimental results in Table 2c demonstrate that the shared mean strategy even slightly reduces quantization error without degrading perplexity, verifying its effectiveness.
this section cite: []

Section: Choice of Partitioning Number.
We investigate the impact of varying the number of partition candidates on final quantization performance under the row-wise grouping setting. Specifically, for each row, we generate partition candidates based on absolute value percentiles ranging from 10% to 90%, and evaluate the corresponding quantization error and perplexity, as shown in Table 2d. Experimental results indicate that moderately increasing the number of partition candidates can effectively reduce quantization error and further lower perplexity, while excessive partitioning yields diminishing returns and increases computational cost. Consequently, we adopt 40 partition candidates as the default setting to balance performance and efficiency.
this section cite: []

Section: Time and Memory Analysis

this section cite: []

Section: Memory Comparison.
As shown in Table 4, HBLLM-col achieves better performance while occupying a storage size comparable to ARB-LLM. By employing a grouped shared-mean strategy, HBLLM improves compression efficiency without sacrificing performance. Specifically, HBLLM-col applies Haar transforms along the column dimension, such that only one grouped quantization operation is required per row on the transformed coefficients. Compared to HBLLM-row, this leads to reduced data fidelity but provides clear advantages in storage cost. Notably, the reported memory usage is measured at runtime in our setup and may be influenced by model variants and implementation choices, leaving room for further engineering optimizations. The detailed storage calculation formulas can be found in the appendix D.
this section cite: []

Section: Inference Latency Estimation
To evaluate the inference latency of HBLLM, we conduct an experiment that combines direct measurement with estimation. Due to there is no existing inference framework that fully supports the dequantization algorithm used in HBLLM, we test GEMV on layers from the OPT-175B model instead. The tests are run on an NVIDIA P100 GPU following the GPTQ benchmark setup [1]. Our estimation results show that the inference latency of HBLLM is approximately 31.8% of the FP16 baseline inference time. For more details, please refer to the appendix G.
this section cite: ['b0']

Section: Conclusion
We introduce a 1-bit weight only quantization HBLLM, which applies Haar transform to BILLM pipeline. Besides quantifying the coefficients on frequence domain, HBLLM integrates two innovative structure-aware grouping strategies to enhance fidelity. Furthermore, HBLLM optimize storage efficiency. As a results, HBLLM outperforms SOTA QAT quantization methods of LLM at 1-bit across different LLM families and tests. The current HBLLM supports only quantized dense models.
Next, we will focus on the MoE PTQ algorithm.
this section cite: []

Section: References
Ref_id:b0 Title: FrameQuant: flexible low-bit quantization for transformers Year: ()
Ref_id:b1 Title: The Twelfth International Conference on Learning Representations Year: (2024)
Ref_id:b2 Title: AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration Year: (2024)
Ref_id:b3 Title: PIQA: Reasoning about physical commonsense in natural language Year: ()
Ref_id:b4 Title: QuIP: 2-bit quantization of large language models with guarantees Year: (2023)
Ref_id:b5 Title: DB-LLM: Accurate dual-binarization for efficient LLMs Year: ()
Ref_id:b6 Title: Exploring the surprising difficulty of natural yes/no questions Year: (2019)
Ref_id:b7 Title: Think you have solved question answering? Try ARC, the AI2 reasoning challenge Year: (2018)
Ref_id:b8 Title: SAR image segmentation based on convolutional-wavelet neural network and Markov random field Year: (2017)
Ref_id:b9 Title: OAC: Output-adaptive calibration for accurate post-training quantization Year: ()
Ref_id:b10 Title: Massive language models can be accurately pruned in one-shot Year: ()
Ref_id:b11 Title: OPTQ: Accurate post-training quantization for generative pre-trained transformers Year: (2023)
Ref_id:b12 Title: A survey of low-bit large language models: Basics, systems, and algorithms Year: (2024)
Ref_id:b13 Title: Wavelet-SRNet: A wavelet-based CNN for multi-scale face super resolution Year: (2017)
Ref_id:b14 Title: BiLLM: Pushing the limit of post-training quantization for LLMs Year: ()
Ref_id:b15 Title: An empirical study of LLaMA3 quantization: From LLMs to MLLMs Year: ()
Ref_id:b16 Title: Mixture of scales: Memory-efficient token-adaptive binarization for large language models Year: (2024)
Ref_id:b17 Title: Arb-LLM: Alternating refined binarizations for large language models Year: (2024)
Ref_id:b18 Title: Less is more: Task-aware layer-wise distillation for language model compression Year: ()
Ref_id:b19 Title: A Wavelet Tour of Signal Processing Year: (1999)
Ref_id:b20 Title: The Penn Treebank: Annotating predicate argument structure Year: (1994)
Ref_id:b21 Title: Sentinel Mixture Models. International Conference on Learning Representations Year: (2017)
Ref_id:b22 Title: Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering Year: (2018)
Ref_id:b23 Title: Few-shot learning evaluation in natural language understanding Year: (2021)
Ref_id:b24 Title: The LAMBADA dataset: Word prediction requiring a broad discourse context Year: (2016)
Ref_id:b25 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b26 Title: Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning Year: (2011)
Ref_id:b27 Title: Winogrande: An adversarial Winograd schema challenge at scale Year: (2021)
Ref_id:b28 Title: PB-LLM: Partially binarized large language models Year: (2024)
Ref_id:b29 Title: Distilling reasoning capabilities into smaller language models Year: (2023)
Ref_id:b30 Title: A simple and effective pruning approach for large language models Year: (2023)
Ref_id:b31 Title: LLaMA: Open and efficient foundation language models Year: (2023)
Ref_id:b32 Title: SmoothQuant: Accurate and efficient post-training quantization for large language models Year: ()
Ref_id:b33 Title: OneBit: Towards extremely low-bit large language models Year: ()
Ref_id:b34 Title: ZeroQuant: Efficient and affordable post-training quantization for large-scale transformers Year: (2022)
Ref_id:b35 Title: HellaSwag: Can a machine really finish your sentence? Year: (2019)
Ref_id:b36 Title: OPT: Open pre-trained transformer language models Year: (2022)
Ref_id:b37 Title: OPT-175B LICENSE AGREE-MENT Year: ()
Ref_id:b38 Title: Creative Commons Zero v1.0 Universal Year: ()
Ref_id:b39 Title: Dataset provided for research purposes only Year: ()
