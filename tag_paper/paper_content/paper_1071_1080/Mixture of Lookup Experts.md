Title: Mixture of Lookup Experts
Abstract: Mixture-of-Experts (MoE) activates only a subset of experts during inference, allowing the model to maintain low inference FLOPs and latency even as the parameter count scales up. However, since MoE dynamically selects the experts, all the experts need to be loaded into VRAM. Their large parameter size still limits deployment, and offloading, which load experts into VRAM only when needed, significantly increase inference latency. To address this, we propose Mixture of Lookup Experts (MoLE), a new MoE architecture that is efficient in both communication and VRAM usage. In MoLE, the experts are Feed-Forward Networks (FFNs) during training, taking the output of the embedding layer as input. Before inference, these experts can be reparameterized as lookup tables (LUTs) that retrieves expert outputs based on input ids, and offloaded to storage devices. Therefore, we do not need to perform expert computations during inference. Instead, we directly retrieve the expert's computation results based on input ids and load them into VRAM, and thus the resulting communication overhead is negligible. Experiments show that, with the same FLOPs and VRAM usage, MoLE achieves inference speeds comparable to dense models and significantly faster than MoE with experts offloading, while maintaining performance on par with MoE. Code: https://github.com/JieShibo/MoLE.

Section: Introduction
Scaling laws indicate that, with sufficient data for training, the performance of large language models (LLMs) improves as the model size increases (Kaplan et al., 2020). However, larger LLMs also result in slower inference speeds, which can degrade the user experience. For this reason, the architecture of LLMs has increasingly focused on Mixture of Experts (MoE) (Jiang et al., 2024;Dai et al., 2024). MoE models use several Feed-Forward Networks (FFNs) as experts and employ a router to determine which subset of experts needs to be activated, rather than activating the entire model. This allows the model to maintain a large number of parameters while keeping the computational cost low.
Although MoE reduces the computational cost, the number of parameters does not decrease. This means that the VRAM requirements during inference remain unaffordable. For example, although the Mixtral-8×7B (Jiang et al., 2024) model only has 13B parameters activated at a time, its total parameter count reaches 46B, making it impossible to load into a single 80GB GPU with FP16. Existing methods (Eliseev & Mazur, 2023;Xue et al., 2024;Shen et al., 2022) reduce VRAM usage by offloading experts to larger storage devices (e.g., CPU RAM, disk, or cloud storage), and loading the selected experts into VRAM at each inference step. However, there are two drawbacks to this approach: i) Since the selection of experts is dynamically determined by the router, we must load different experts into VRAM at each inference step. Frequent transfer of large numbers of parameters can significantly increase inference latency, as in Figure 1. ii) Since different samples select different experts within a single step, loading only a subset of experts may not meet the needs of batched generation.
To address the issues mentioned above, we propose Mixture of Lookup Experts (MoLE), a new LLM architecture. MoLE has different structures in training and inference. During training, MoLE is similar to MoE, with a router and several experts. However, unlike MoE, where experts take intermediate features as input, MoLE's experts are fed with embedding tokens (i.e., the output of the embedding layer) instead. Additionally, MoLE allows all experts to be activated simultaneously. After training, MoLE is not directly used for inference but undergoes a series of re-parameterizations. Since the output of the embedding layer is fixed for specific input ids, the inputs to the experts have only a limited number of choices, equal to the model's vocabulary size. Therefore, for each token in the embedding layer, we pre-compute the outputs corresponding to all experts, creating lookup tables (LUTs) that replaces the original experts.
During inference, MoLE demonstrates several advantages:
• Computation-free experts. The experts are reparameterized from FFNs into LUTs, eliminating the need for any computation. Each expert only requires a single lookup operation.
• Low VRAM overhead and communication latency.
Although the size of the LUT is much larger than the model itself, it can be entirely offloaded to storage devices. During inference, since the output of each expert is the same number of tokens as the input, the communication required to load the lookup results into VRAM is negligible, thereby avoiding increased inference latency.
• Batch-generation friendly.
Traditional expert offloading methods introduce additional VRAM usage and communication latency during batched generation because different samples in a batch may select different experts. MoLE only transfers the pre-computed expert outputs, making its communication overhead still negligible even during batched generation.
Through extensive experiments, we validated the effectiveness of MoLE at scales of 160M, 410M, and 1B parameters. As in Figure 1, with equivalent computational cost and VRAM usage, MoLE significantly outperforms dense models while maintaining the same inference speed. Compared to MoE with expert offloading, MoLE achieves better performance and with significantly faster inference speed.
this section cite: ['b17', 'b14', 'b6', 'b14', 'b10', 'b29', 'b27']

Section: Related Work

this section cite: []

Section: Mixture-of-Experts
The concept of MoE was initially introduced by Jacobs et al. (1991); Jordan & Jacobs (1994) and has been widely explored and developed through subsequent research (Collobert et al., 2002;Rasmussen & Ghahramani, 2001;Shahbaba & Neal, 2009;Eigen et al., 2014;Theis & Bethge, 2015;Deisenroth & Ng, 2015;Aljundi et al., 2017;Shazeer et al., 2017). MoE posits that different parts of the model, i.e., the experts, focus on distinct tasks or encapsulate different kinds of knowledge. In this paradigm, only the experts relevant to a given input are activated, which allows the model to scale its capacity while keeping computational costs manageable and making full use of specialized knowledge across many experts. As the scale of LLMs increases, reducing computational overhead has become a key focus. This leads to its application in transformer-based LLMs (Lepikhin et al., 2021), making MoE a widely used architecture.
Recently, a series of industrial-scale large language models have been released, including Mixtral (Jiang et al., 2024) and DeepSeek-MoE (Dai et al., 2024). Some works also aim to improve the efficiency of MoE (Jin et al., 2025) or increase expert capacity (Yan et al., 2025) by modifying the model architecture.
this section cite: ['b13', 'b16', 'b5', 'b23', 'b25', 'b9', 'b28', 'b7', 'b0', 'b26', 'b19', 'b14', 'b6', 'b15', 'b30']

Section: Expert Offloading
Offloading techniques typically transfer a portion of the model parameters to CPU RAM or disk when GPU memory is insufficient. However, current mainstream offloading frameworks, such as Zero-Infinity (Rajbhandari et al., 2021), are designed for dense LLMs and load model parameters layer by layer on-demand. This approach overlooks the sparse activation property of MoE models, leading to the unnecessary loading of inactive experts.
Building on this, some studies have proposed expert offloading, a form of parameter offloading specifically designed for the sparse activation characteristic of MoE models (Eliseev & Mazur, 2023;Xue et al., 2024). These methods stores non-expert weights and a portion of the expert cache in VRAM, while the remaining experts are offloaded to CPU RAM or disk and loaded on-demand.
Despite being effective, existing expert offloading techniques still suffer from high latency. Subsequent research includes optimizing prefetching techniques and cache replacement strategies to accelerate inference speed (Shen et al., 2022), designing MoE architectures that are more friendly to prefetching (Hwang et al., 2024), or employing other model compression techniques to reduce prefetching latency (Yi et al., 2023).
... Attention Embedding Router Shared FFN FFN3 FFN4 FFN2 FFN1 ... Attention Embedding Shared FFN FFN3 FFN4 FFN2 FFN1 ... Attention Embedding Shared FFN id id id ... Off-Device Lookup MoE MoLE (Ours) Training Inference (Optional) Router Router Figure 2. Illustration of MoLE. During training, MoLE differs from MoE in two key structural aspects: i) The routed experts in MoLE take embedding tokens as input. ii) All experts in MoLE are activated. During inference, the routed experts in MoLE are re-parameterized as zero-computation, offloaded LUTs. For simplicity, normalization layers and residual connections of attention layers are omitted.
this section cite: ['b22', 'b10', 'b29', 'b27', 'b12', 'b31']

Section: Mixture of Lookup Experts

this section cite: []

Section: Preliminary
First, we briefly introduce the structure of MoE and the challenges it faces during inference.
For MoE, each expert is typically a FFN module. As illustrated in Figure 2 (left), a MoE layer contains N routed experts, represented as {FFN j } N j=1 , and a linear router, denoted as {r j } N j=1 . Some models may also introduce a shared expert FFN shared that is activated in all cases. Given input token h ∈ R d , the output token h ′ ∈ R d of the MoE layer is computed as
G = ArgTopK({h • r j } N j=1 )(1)
{g j } j∈G = SoftMax({h • r j } j∈G )(2)
h ′ = j∈G g j FFN i (h) + FFN shared (h) + h (3)
where G denotes the indexes of the activated experts, and g i denote the gate value for the i-th expert.
The computational efficiency of MoE lies in the fact that, in Eq. ( 3), only k routed experts are involved in the computation. However, we cannot determine which experts need to participate until Eq. ( 1) is completed. This means that we either need to store all the experts in VRAM or temporarily load the required k experts into VRAM after Eq. ( 1) is computed.
However, both of these solutions present deployment challenges. Taking Mixtral-8×7B as an example, it has 32 MoE layers, with 8 experts per layer, but only 2 experts are activated per token. Although only 13B parameters are activated per token, the total parameter count reaches up to 46B, requiring at least 92GB of VRAM for FP16 deployment.
If temporary loading is chosen, each expert is 176M in size, and loading the necessary experts for a single decoding step would require up to 11.3B of parameter transfer. If offloading to CPU VRAM is selected, using a GPU with PCIe 4.0×16 would still incur a transfer latency of 0.7s per step. Offloading to disk, on the other hand, results in an unacceptable transfer latency of over 10s per step. More importantly, since the selection of experts is dynamically determined by the router, the experts chosen for different samples are highly likely to differ when batch size > 1. This requires loading all the selected experts (may be all experts when batch size is large) into VRAM, which not only increases VRAM usage but also further exacerbates communication latency.
The reason experts need to be loaded into VRAM is that they participate in the computation, which relies on GPU.
In other words, if the experts do not require computation, we do not need to load them, thereby avoiding significant communication overhead. To address this, we introduce MoLE, a new MoE architecture whose experts can be reparameterized as computation-free LUTs in inference.
this section cite: []

Section: Training Phase
As illustrated in Figure 2, during training, MoLE and MoE have similar structures, including N routed experts
0 0 MoE 4d(kD r + D s ) 2d(N D r + D s ) 0 0 MoE + Expert Offloading 4d(kD r + D s ) 2d(kD r + D s ) 2dN D r 2dkD r (worst case) MoLE + LUT Offloading 4dD s 2dD s dN |V| dN {FFN j } N j=1
and a linear router {r j } N j=1 . Specifically, MoLE also includes a shared expert FFN shared , which is always activated for any input and does not receive weighting from the router.
Since the experts will be transformed into LUTs after training, MoLE differs from MoE in the following ways. First, LUT is computation-free, eliminating the need for sparse activation to reduce computational cost. Therefore, MoLE activates all experts, rather than just the top-k experts. The computation for the router is as
{g j } N j=1 = SoftMax({h • r i } N i=1 )(4)
Second, since the LUT is essentially a mapping between a finite set of input-output pairs, the key to re-parameterizing the experts into LUTs is ensuring that they only have a limited number of possible inputs. To this end, MoLE uses the output of the embedding layer, i.e., the embedding tokens, as the input to the experts. After training, the embedding layer is only related to the input ids, which means that the inputs to the experts are limited to a finite set. The computation for the layer is as
h ′ = N j=1 g j FFN i (e) + FFN shared (h) + h(5)
in which e = Embedding(i) ∈ R d is the embedding token, and i deonotes the input id.
All experts are activated and receive gradients during training. Therefore, we do not need to add any auxiliary losses to prevent collapse or maintain training stability. MoLE is trained solely using the cross-entropy loss of language modeling just like a dense model.
this section cite: []

Section: Inference Phase
After training, MoLE can be directly used for inference like other LLMs. However, to further reduce VRAM overhead, we can re-parameterize the experts. For each possible input id i, we pre-compute the outputs of expert FFN j as
v i j = FFN j (Embedding(i)) ∈ R d(6)
In practice, we only need to perform a single forward pass with the embedding layer's weights as the input to FFN j , which allows us to obtain v i j for all i. The items of the LUT at the l-th layer can be represented as
LUT l = {{v i j } N j=1 } |V| i=1(7)
where |V| denotes the size of the vocabulary.
After re-parameterization, the LUT is offloaded to the storage device, and the computation of the MoLE layer can be represented as
h ′ = N j=1 g j v i j + FFN shared (h) + h (8
)
The inputs of the LUT in MoLE are the input ids, meaning that no context information is included. This is a trade-off to ensure that the experts can be re-parameterized, but it does not imply that the expert layers do not contribute to contextrelated knowledge. Firstly, the router and shared experts still take intermediate features as input, which means they can access contextual information. Secondly, the output of the expert layer is part of the input to subsequent attention layers, allowing the experts to influence the behavior of later attentions. This enables the experts to adjust how the model processes the same words in different contexts, thereby still enhancing the model's capacity.
this section cite: []

Section: Complexity Analysis
Consider an MoE layer with MLP-based FFNs as experts.
Let the hidden layer dimension of the routed experts be D r , and the hidden layer dimension of the shared experts be D s . When a single token is used as input, the FLOPs for this MoE layer can be computed as
FLOPs MoE = 4d(kD r + D s ) (9
)
in which the router and normalization is neglected.
To save VRAM, we assume that all routed experts are offloaded, and then the offloaded parameter count is
OffParam MoE = 2dN D r (10
)
In the worst case, during inference, we need to load the k experts that the current token is routed to into VRAM. The number of per-step loaded parameter is For MoLE, since the experts are transformed into computation-free LUTs, its FLOPs can be computed as
LoadParam MoE = 2dkD r (11
)
FLOPs MoLE = 4dD s (12
)
The number of parameters contained in the offload LUT is
OffParam MoLE = dN |V|(13)
In each inference step, since we only need to load all the v i j from Eq. ( 8) into VRAM, the amount of parameters loaded is only
LoadParam MoLE = dN (14
)
We summarize all these comparisons in Table 1. Since |V| is typically on the order of tens of thousands, for example, |V| = 32k for Mixtral (Jiang et al., 2024) and |V| = 50k for Pythia (Biderman et al., 2023), and D r varies from thousands to tens of thousands depending on the model size, the number of offloaded parameters in MoE and MoLE will not differ by an order of magnitude. However, the number of parameters loaded per token in MoLE will be only a fraction -often hundreds or even thousands of times smaller -compared to the number of parameters loaded per token in MoE.
this section cite: ['b14', 'b1']

Section: Experiments

this section cite: []

Section: Experimental Setup
Model Architectures. As shown in Table 2, we implement models with activation parameter counts of 160M, 410M, and 1B. For the dense model, we basically follow the Pythia (Biderman et al., 2023) setup. For the MoE model, we adopt a configuration similar to Mixtral, with no shared experts and activating the top-2 routed experts, since Muennighoff et al. (2024) suggest that shared experts lead to performance degradation. To ensure the number of activation parameter is the same as that of the dense model, the hidden dimension of the MoE FFNs is set to half the hidden dimension of the dense model's FFNs.
For the MoLE model, since routed experts have no computation during inference, we use FFNs identical to those of the dense model's FFNs as the shared experts to keep the same FLOPs as the dense model. For the routed experts, since their hidden dimension does not affect the model architecture during inference, we set it to be the same as the shared experts for simplicity. We implemente MoE with both 10 and 34 experts and MoLE with both 4 and 16 experts for comparison.
this section cite: ['b1', 'b20']

Section: Data & Tokenizer.
We train all models on a 100B-token subset of the deduped Pile dataset (Gao et al., 2021), using the GPT-NeoX tokenizer employed by Pythia, with a vocabulary size of 50k.
Hyper-Parameters. We follow the learning rate settings used by Pythia, specifically 6.0 × 10 -4 for the models with 160M activated parameter, and 3.0 × 10 -4 for the models with 410M and 1B activated parameter. For the MoE model, the coefficients for the z-loss and load balance loss are set to 0.001 and 0.01, respectively, as suggested by Muennighoff et al. (2024).
Benchmarks. We use the lm-evaluation-harness package for evaluation. The benchmarks used include ARC-C (Clark et al., 2018), ARC-E (Clark et al., 2018), BoolQ (Clark et al., 2019), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), RACE (Lai et al., 2017), SIQA (Sap et al., 2019), and LAMBADA (Paperno et al., 2016). For all these benchmarks, we report the zero-shot accuracy.
Offloading Setting. To measure the deployment efficiency of different models in VRAM-constrained environments, we apply offloading strategies to both MoE and MoLE to ensure their VRAM usage is consistent with that of a dense model with the same number of activated parameters. For the MoE model, we adopt an expert offloading strategy, where only the parameters of the activated experts and all non-expert parameters are stored in VRAM. During each inference step, if the required activated expert is not in VRAM, it is loaded from the storage device into VRAM, replacing the previously loaded expert. For MoLE, we offload the LUT, while keeping the other parameters stored in VRAM.
this section cite: ['b11', 'b20', 'b4', 'b4', 'b3', 'b32', 'b2', 'b18', 'b21']

Section: Main Results

this section cite: []

Section: As shown in
Table 3, both MoE and MoLE significantly improve performance over the dense baseline. In the com-parison of five pairs of MoLE and MoE models with the same number of training parameters, MoLE outperforms MoE in four out of the five comparisons in terms of average accuracy. Notably, for MoLE-16E with 160M, 410M, and 1B activated parameters, the number of per-token loaded parameter is only about 1/1500, 1/2000, and 1/2000 of that of MoE, respectively. This demonstrates that MoLE can maintain outstanding performance while significantly reducing communication overhead, making it feasible to offload to lower-tier storage devices.
We note that the size of the LUTs in MoLE is 2.4 to 7.4 times larger than the size of the offloaded experts in MoE. However, since these parameters are offloaded to large, scalable storage devices, we believe the storage overhead of the LUTs remains within an acceptable range. Specifically, as the model size increases, the proportion of the LUTs also decreases accordingly. On models with 1B activated parameters, the size of the LUTs in MoLE-4E becomes
this section cite: []

Section: Ablation Experiments
Training loss. Unlike MoE, MoLE is a fully differentiable model, so during training, we do not encounter issues like router collapse or instability. Therefore, we do not use any additional auxiliary losses. To illustrate this, we attempt to add the load balance loss and z-loss from MoE. As shown in Table 4, after adding these losses, the model's performance decline. This is because the additional losses caused the model's optimization objectives to become misaligned with the inference requirements, leading to negative effects.
Number and size of experts. We experiment with varying the size and number of experts. As shown in Table 5, when the hidden dimension of the experts increases from d to 4d, the model's performance improves. However, further increasing the dimension to 16d leads to performance saturation. This is because increasing the size of the experts does not affect the re-parameterized models or the size of the LUTs during inference. It indicates that the knowledge embedded in LUTs with a constant size reaches saturation as the expert size increases, meaning that there is no "free lunch" -further increases in expert size do not lead to additional performance gains.
Unlike the increase in size, the increase in the number of experts results in a continuous performance improvement, demonstrating a certain level of scalability. At the same time, the size of the LUTs and the amount of parameters transferred will also increase proportionally.
Architecture designs. To ensure that routed experts can be re-parameterized, we change the input to the routed experts from intermediate features to embedding tokens. Intuitively, this modification means that the experts only receive the raw word features and no context-related information, which is likely to lead to a decrease in model performance. But on the other hand, since the re-parameterized experts do not require computation during the inference phase, we can activate all the experts while keeping the model's inference FLOPs unchanged. This helps compensate for the performance loss mentioned earlier.
To conduct an ablation study on this, we trained the following model variants, evolving from MoE-10E to MoLE-4E of 160M activated parameters:
• Full activation. All experts of MoE-10E are activated, which causes the number of activated parameters to increase from 0.16B to 0.39B. Meanwhile, all auxiliary losses are discarded.
• Reconfiguration. Based on the above model, we modify the experts by changing from 10 routed experts to 1 shared expert and 4 routed experts. Additionally, the hidden dimension of the experts is increased to twice its original size. The total number of parameters remains unchanged.
• Embedding as inputs. Based on the above model, we change the input to the routed experts to embedding tokens.
• Re-parameterization. Based on the above model, we reparameterize the routed experts as LUTs. Then the As shown in the table, using embedding tokens as the input to the routed experts only results in a 0.7 performance drop, but it brings significant benefits of enabling the experts to be re-parameterized, allowing us to activate all experts. A fully activated MoE yields a 1.5 performance gain compared to top-2 MoE, which leads to an overall performance improvement for MoLE over MoE.
this section cite: []

Section: Efficiency
We measure the per-step decoding latency of models with 410M activated parameters on NVIDIA V100 GPU using Huggingface's transformers package. Since the specific speed of parameter loading is largely influenced by the underlying implementation, we estimate the latency of loading parameters based on the maximum PCIe bandwidth of the V100, which is 16GB/s. For the MoE model, when the batch size is 1, the experts activated in the previous decoding step are retained in VRAM. When the batch size is greater than 1, random two of the activated experts from the previous decoding step are retained in VRAM for each layer.
If the experts activated in the current step overlap with those in VRAM, they will not be reloaded. Under this setup, the average number of experts loaded per step for batch sizes of 1, 8, and 32 are 1.6, 6.7, and 8.0 for MoE-10E, or 1.9, 12.3, and 27.4 for MoE-34E, respectively. The input length is fixed to 512.
As shown in Figure 3, the latency of MoLE is comparable to that of the dense model, while MoE exhibits significantly higher latency than the dense model. As the batch size increases, the number of experts being loaded also increases, further adding to the latency of MoE, but the latency of MoLE has almost no increase.
this section cite: []

Section: Reducing the Size of LUTs
Although MoLE significantly reduces data transfer in offloading scenarios compared to MoE, it has a larger storage footprint. While storage space may not be as constrained as VRAM, reducing the size of the LUTs can still alleviate deployment burdens. To address this, we conduct a simple experiment to compress the LUTs. We apply posttraining quantization to the FP16 LUTs, quantizing them to NF4 and NF3 (Dettmers et al., 2023) data types. The token-wise block sizes for quantization are 768 and 128, respectively. As shown in Table 8, the model's performance suffers minimal loss, while the storage burden and size of transferred data are reduced to 25.3% and 19.5% of the original size, respectively. This indicates that the LUTs still contains significant redundancy and has the potential for further compression. We leave this as future work.
this section cite: ['b8']

Section: Conclusion
In this paper, we address the issues of high memory consumption and loading latency in MoE by proposing MoLE, a novel language model architecture. MoLE restricts the input to experts to a limited set (embedding tokens), allowing the experts to be re-parameterized into LUTs before inference, thus avoiding the need to load expert parameters. MoLE demonstrates competitive results on downstream tasks, while significantly outperforming MoE with expert offloading in terms of inference speed. This work provides a new direction for designing edge-friendly language models. Future research could explore more diverse discrete spaces and expert architectures.
this section cite: []

Section: References
Ref_id:b0 Title: Expert gate: Lifelong learning with a network of experts Year: (2017)
Ref_id:b1 Title: Pythia: A suite for analyzing large language models across training and scaling Year: (2023)
Ref_id:b2 Title: PIQA: reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b3 Title: Exploring the surprising difficulty of natural yes/no questions Year: (2019)
Ref_id:b4 Title: Think you have solved question answering? try arc, the AI2 reasoning challenge Year: (2018)
Ref_id:b5 Title: A parallel mixture of svms for very large scale problems Year: (2002)
Ref_id:b6 Title: Towards ultimate expert specialization in mixture-of-experts language models Year: (2024)
Ref_id:b7 Title: Distributed gaussian processes Year: (2015)
Ref_id:b8 Title: Efficient finetuning of quantized llms Year: (2023)
Ref_id:b9 Title: Learning factored representations in a deep mixture of experts Year: (2014)
Ref_id:b10 Title: Fast inference of mixture-ofexperts language models with offloading Year: (2023)
Ref_id:b11 Title: The pile: An 800gb dataset of diverse text for language modeling Year: (2021)
Ref_id:b12 Title: Pre-gated moe: An algorithm-system codesign for fast and scalable mixture-of-expert inference Year: (2024)
Ref_id:b13 Title: Adaptive mixtures of local experts Year: (1991)
Ref_id:b14 Title:  Year: (2024)
Ref_id:b15 Title: Accelerating mixture-of-experts methods with zero-computation experts Year: (2025)
Ref_id:b16 Title: Hierarchical mixtures of experts and the EM algorithm Year: (1994)
Ref_id:b17 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b18 Title: RACE: large-scale reading comprehension dataset from examinations Year: (2017)
Ref_id:b19 Title: Scaling giant models with conditional computation and automatic sharding Year: (2021)
Ref_id:b20 Title: Open mixture-of-experts language models Year: (2024)
Ref_id:b21 Title: The LAMBADA dataset: Word prediction requiring a broad discourse context Year: (2016)
Ref_id:b22 Title: Zero-infinity: breaking the GPU memory wall for extreme scale deep learning Year: (2021)
Ref_id:b23 Title: Infinite mixtures of gaussian process experts Year: (2001)
Ref_id:b24 Title: Social iqa: Commonsense reasoning about social interactions Year: (2019)
Ref_id:b25 Title: Nonlinear models using dirichlet process mixtures Year: (2009)
Ref_id:b26 Title: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer Year: (2017)
Ref_id:b27 Title: Se-moe: A scalable and efficient mixture-of-experts distributed training and inference system Year: (2022)
Ref_id:b28 Title: Generative image modeling using spatial lstms Year: (2015)
Ref_id:b29 Title: Moeinfinity: Activation-aware expert offloading for efficient moe serving Year: (2024)
Ref_id:b30 Title: Tcmoe: Augmenting mixture of experts with ternary expert choice Year: (2025)
Ref_id:b31 Title: Edgemoe: Fast on-device inference of moe-based large language models Year: (2023)
Ref_id:b32 Title: Can a machine really finish your sentence? Year: (2019)
