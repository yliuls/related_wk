Title: MIXTURE-OF-EXPERTS CAN SURPASS DENSE LLMS UNDER STRICTLY EQUAL RESOURCE
Abstract: Mixture-of-Experts (MoE) language models dramatically expand model capacity and achieve remarkable performance without increasing per-token compute. However, can MoEs surpass dense architectures under strictly equal resource constraints -that is, when the total parameter count, training compute, and data budget are identical? This question remains under-explored despite its significant practical value and potential. In this paper, we propose a novel perspective and methodological framework to study this question thoroughly. First, we comprehensively investigate the architecture of MoEs and achieve an optimal model design that maximizes the performance. Based on this, we subsequently find that an MoE model with activation rate in an optimal region is able to outperform its dense counterpart under the same total parameter, training compute and data resource. More importantly, this optimal region remains consistent across different model sizes. Although additional amount of data turns out to be a trade-off for enhanced performance, we show that this can be resolved via reusing data. We validate our findings through extensive experiments, training nearly 200 language models at 2B scale and over 50 at 7B scale, cumulatively processing 50 trillion tokens. All model checkpoints are publicly available † .

Section: INTRODUCTION
In recent years, Large Language Models (LLMs) built on the Transformer architecture (Vaswani et al., 2017) have achieved strong results on a range of NLP tasks (Radford et al., 2018;Achiam et al., 2023;Touvron et al., 2023a;b;Bai et al., 2023). Meanwhile, there has been growing interest in using Mixture-of-Experts (MoE) layers (Shazeer et al., 2017) to expand model capacity while keeping the training cost reasonable (Fedus et al., 2022;Zoph et al., 2022;Rajbhandari et al., 2022). Recent opensource initiatives have explored MoE-based LLMs (Dai et al., 2024;Jiang et al., 2024;Wei et al., 2024;Xue et al., 2024;DeepSeek-AI et al., 2024b), yet many widely adopted open-source models 2017), the integration of MoE into the Transformer framework has become a popular model architecture and achieved state-of-the-art performance. As an early attempt, Switch Transformer (Fedus et al., 2022) proposes top-1 gating to simplify MoE architecture and alleviate communication overhead. More recent Transformer-based MoE LLMs include (Xue et al., 2024;Jiang et al., 2024;Wu et al., 2024;Wei et al., 2024;DeepSeek-AI et al., 2024b). The MoE architecture is briefly reviewed in Appendix A.
this section cite: ['b48', 'b38', 'b1', 'b24', 'b2', 'b43', 'b13', 'b65', 'b39', 'b9', 'b51', 'b57', 'b13', 'b57', 'b54', 'b51']

Section: ANALYSES OF MOE SPARSITY
Several studies investigated the impact of varying the number of MoE experts and adjusting granularity, both of which are factors related to sparsity. Through ablation studies, DeepSeekMoE (Dai et al., 2024) observes finer granularity results in improvement on overall model performance, and acquires a ratio between shared and routed experts that yields slightly better Pile loss. Zoph et al. (2022) summarized the results of several MoE works and indicated that the gain of increasing sparsity quickly diminishes when the number of experts is greater than 256 -a very sparse model. From a methodological perspective, our comparisons are conducted in a sufficiently trained regime (with D /N ≥ 20 for key models) and under a fixed-N setting motivated by deployment memory constraints, which differs from scaling-law sweeps that often rely on undertrained large models at limited compute. We provide a more detailed discussion in Appendix B.
Concurrently with our work, Ludziejewski et al. (2025) found that a sufficiently large MoEs trained with more tokens outperforms a dense model with the same total parameters. We further show MoE superiority even at smaller sizes and address the additional data demand via reuse. Abnar et al. (2025) studied the scaling law for optimal MoE sparsity. However, their models (up to N = 30B) were trained with C = 1e20, a much smaller budget compared to the approximately 9× and 30× compute we used for our 2B and 7B models to ensure an adequate D /N. This likely resulted in undertrained models, potentially affecting their conclusions. Moreover, our study first optimizes the MoE architecture to isolate the effect of different activation rates on performance. Detailed differences between our work and this previous study are discussed in Appendix B.
this section cite: ['b9', 'b65', 'b33', 'b0']

Section: EXPERIMENTAL METHODOLOGY
We begin by introducing a unified parameterization framework for model architecture, establishing a solid foundation. Then, we derive key insights from this parameterization, which informs our comprehensive three-step experimental methodology. Finally, we detail the experimental setup used consistently across all subsequent experiments. Our notation is summarized in Table 1.
this section cite: []

Section: ARCHITECTURE PARAMETERIZATION
To enable a comprehensive and general comparison of dense and MoE-based LLM architectures under realistic deployment constraints, we introduce a unified parameterization framework that explicitly accounts for both model parameters and per-token compute cost.
this section cite: []

Section: Dense model parameterization.
For a dense model, we approximate the number of nonembedding parameters N and the per-token forward-pass computation cost M as follows:
N ≈ (4 + 3α)D 2 m L = (4 + 3α)ζ 2 L 3 , (1) M ≈ 2N + 4D m SL = 2N + 4ζ 2 γL 3 (2) = 2N (1 + 2γ /(4 + 3α)),(3)
where α = Dffn /Dm, γ = S /Dm, and ζ = Dm /L. Here, we omit the LayerNorm parameter count as it is negligible. Inference cost or training cost can be approximated by C ≈ M × D or C ≈ 3M × D, respectively (based on the standard empirical observation that training typically takes about three times the forward pass for backward computation).
this section cite: []

Section: MoE model parameterization.
In many real-world settings, only a subset of Transformer layers are replaced with MoE layers. The approximations for total non-vocabulary parameters N , activated parameters N a , and per-token computation cost M can be expressed as
N ≈ (4 + 3µ)D 2 m L e + (4 + 3α)D 2 m L d , (4
)
N a ≈ (4 + 3β)D 2 m L e + (4 + 3α)D 2 m L d , (5
) M ≈ N a + 4D m SL = 2r a N + 4ζ 2 γL 3 ,(6)
where µ = (Dse + EDe) /Dm and β = (Dse + KDe) /Dm. We again omit the parameters and FLOPs of the gating network (router) as they are comparatively small, and the activation rate (AR) is denoted as r a . For the simple and common case where all L layers are MoE layers (i.e.,
L d = 0), we have r a = N a /N = (4 + 3β) /(4 + 3µ), (7) M ≈ 2r a N + 4ζ 2 γL 3 (8) = 2r a N (1 + 2γ /(4 + 3β)).
(9)
this section cite: []

Section: KEY OBSERVATIONS AND METHODOLOGICAL CONSIDERATIONS
Based on the above parameterization, we highlight the following insights that guide our experiments:
High structural degrees of freedom in MoE. Compared to a dense model (whose shape is almost uniquely determined by L, D m , and α), an MoE model has many more design choices: the number of MoE layers (L e ), expert-related dimensions (e.g., K, E, D e , D se ), and so forth. Even with L d = 0, the final shape depends on µ and β in addition to ζ and r a . Exhaustively searching all combinations is prohibitively expensive. Therefore, a greedy strategy should be adopted.
Activation rate r a is the primary factor. At the same total parameter count N , the ratio of pertoken FLOPs between an MoE model and a dense model is primarily determined by the activation rate r a . More specifically, if M d is the per-token cost of the dense baseline (with ζ, α fixed), then the pure MoE model's compute, normalized by M d , roughly behaves as (the union of Equ. 3 and 9):
R c = r a ( 4 + 3α + 2γ d 4 + 3β + 2γ m ) (10
)
where γ d and γ m denote S /Dm for dense and MoE layers, respectively. As γ strongly correlates with ζ, once the shape hyperparameters (ζ, α, β) are chosen, R c grows monotonically with r a .
The trade-off among N , C, and D. Once N (the total parameters) is chosen and we fix nearoptimal shapes for dense and MoE models, the total training compute for the MoE model can be approximated by C = 3 R c M d D, where M d is the per-token cost of the dense baseline with the same total parameter count, and R c is the fraction by which the MoE model reduces compute per token (relative to the dense baseline). If we want to keep the same total compute C for both MoE and dense models, the MoE model will need R c times more training tokens.
this section cite: []

Section: THREE-STEP EXPERIMENTAL METHODOLOGY
Motivated by the aforementioned observations, we propose a three-step experimental methodology which is both comprehensive and fair, so as to achieve the new perspective motivated in § 1 that enables a more conclusive comparison of MoE and dense LLMs under equal resource constraints. The methodology is outlined as follows: 1) Greedy architecture determination. First, determine the macro-level layer composition (i.e., the MoE-to-dense layer ratio L e vs. L d and related choices such as shared experts). Second, determine the micro-level MoE design within each MoE layer (e.g., top-K routing and parameter allocation among routed/shared experts), which is largely orthogonal to the global model shape. Finally, select the near-optimal shape hyperparameters (e.g., ζ, α for dense and ζ, β for MoE) for fair comparisons at a fixed N . 2) Activation rate analysis under fixed N and C. With the optimal MoE architecture chosen in the previous step, and the optimal dense LLM shape proposed by Kaplan et al. (2020), we compare MoE models versus a dense baseline of the same size, ensuring the total training compute C is matched. Since C must be the same, the MoE model typically receives up to R c times more tokens (initially considering repeated or augmented data). 3) Data reuse strategy. To ensure a truly fair comparison at the same unique data budget D, we develop a data reuse strategy that offsets MoE's additional data requirement. This enables evaluation under strictly equal N , C, and D.
this section cite: ['b24']

Section: COMMON EXPERIMENTAL SETUP
Optimal hyperparameters. MoE training is sensitive to the learning rate (η) and batch size (B) (He et al., 2024). Even minor architectural changes, such as variations in E, can lead to different optimal hyperparameters. To address this, we train all our models using the optimal η and B based on the hyperparameter scaling laws proposed in (Li et al., 2025). Specifically, Li et al. (2025) found that the optimal η and B follow power laws and depend only on N and D. Since these scaling laws are applicable to both dense and MoE models and are robust across various pretraining data distributions, we apply them to determining η and B for each of our experiments.
this section cite: ['b15', 'b31', 'b31']

Section: Dense baseline tuning.
To ensure a strictly fair comparison, we also tuned the dense baselines by searching for near-optimal structural ratios (e.g., aspect ratio ζ and FFN expansion α, equivalently L, D m , D ffn with given N), guided by the scaling-law recommendations in Li et al. (2025); the resulting dense configurations used throughout the paper are summarized in Table 13.
Others. We use internal, high-quality training and validation datasets composed primarily of diverse web text and specific domains such as mathematics and code. The training and validation sets have different distributions, requiring the evaluated models to demonstrate strong generalization capabilities. Our models incorporate RMSNorm (Zhang & Sennrich, 2019) for pre-normalization, ALiBi (Press et al., 2022) positional encoding for multi-head attention, and the SwiGLU (Shazeer, 2020) activation function for both feed-forward networks (FFNs) and MoE experts. The training procedures used consistently across all experiments are outlined in Table 4 in Appendix E. We employ cross-entropy loss (L) as the training metric and bits-per-character (BPC) as the validation metric.
this section cite: ['b31', 'b60', 'b36', 'b42']

Section: OPTIMIZED MOE ARCHITECTURE
Building on the insights discussed above, we systematically examine the following model components in the order outlined below: 1) Distribution of MoE and dense layers. 2) Gate score normalization. 3) Parameter allocation within MoE. 4) Exploration of optimal structural hyperparameters. Each component incorporates previous conclusions into its experimental settings.
MoE and dense layers arrangement. This part examines how to arrange the distribution between MoE and dense layers. We consider three layer arrangement schemes: every layer is an MoE layer (full), one dense layer followed by MoE layers (1dense), and interleaved MoE and dense layers (interleave). We additionally include shared experts (SE) for some of our experiments.  Conversely, with a fixed r a , increasing D leads to a linear performance gain. These findings indicate an optimal activation rate, r * * a = 20%, that is consistent across various D values when N is constant. (b) With a fixed training compute C, the optimal activation rate r * * a = 20% can be clearly seen.
impact on model performance. Therefore, we continue using the 1dense+SE configuration and set
D se = KD e .
Gate score normalization. The results of normalizing gate scores of chosen experts are in Table 6 in Appendix E. Although the addition of normalization does not show an obvious difference in performance loss, it tends to reduce the average balance loss Lbalance . Since normalization requires K > 1 to avoid zero gradient, we opt not to normalize given that some of our experiments have K=1.
Top-K setting. In this part, we discuss the allocation of parameters within MoE layers, focusing on the top-K setting. Expert granularity is adjusted by varying K and D e while keeping their product constant. We conduct three groups of experiments with various r a and the results are given in Table 7 in Appendix E. Note that within each experiment group, the product K •D e is not strictly maintained due to compatibility with other hyperparameters. We observe that both overly large K and the K=1 setting are generally suboptimal across the three groups. Therefore, we avoid using large K and avoid setting K=1 in our main experiments whenever possible.
Model shape ratios. As discussed in § 3.2, the shape hyperparameters include three ratios: ζ, α, and β. We set α = 2.77 (Touvron et al., 2023b) and explore the optimal ζ and µ, from which β can be derived. As illustrated in Fig. 4 in Appendix E, although performance fluctuates wildly given a value of ζ or µ, there is an overall upward trend with increasing D m for ζ and a downward trend for µ. Following the observed trend, we set ζ ≈ 88 and µ ≈ 22 for the subsequent experiments.
this section cite: []

Section: OPTIMAL ACTIVATION RATE
In this section, we analyze how the performance of MoE LLMs varies with different activation rates (AR) using model backbones optimized based on the conclusions in § 4, and examine whether MoE models can outperform dense models. Note that a concurrent study (Abnar et al., 2025) suggests that the optimal sparsity of MoE depends on model capacity. However, our findings indicate that, with optimized backbones, the optimal AR remains consistent across models of different sizes. We first detail our experimental setup and results, followed by a further discussion on the conclusions.
Setup. We built a series of MoE models with non-vocabulary parameters N ≈ 2B and N ≈ 7B, but varying activation rates r a from 8.7% to 58%. Noteworthy, the model backbones were built upon the findings in § 4, as detailed in Table 9, 10, 11 and 12 in Appendix E. Each model was trained on a proportional subset of our dataset, ensuring D /N ≥ 20 (Hoffmann et al., 2022) for sufficient training.
10 21 2 × 10 21 3 × 10 21 Total Compute C 4.66 × 10 1 4.68 × 10 1 4.7 × 10 1 4.72 × 10 1 4.74 × 10 1 4.76 × 10 1 4.78 × 10 1 4.8 × 10 1 BPC 7B MoE series, D=130B ra: 8.9% ~ 53% 7B MoE series, D=130B ra: 8.9% ~ 53% (a) Fixed D 0.1 0.2 0.3 0.4 0.5 Activation rate r a 0.454 0.456 0.458 0.460 0.462 0.464 0.466 0.468 BPC 7B MoEs, 2.8e21 FLOPs, unique data 7B MoEs, 2.8e21 FLOPs, D = 68B 7B MoEs, 2.8e21 FLOPs, D = 0.5 * D 7B Dense, 5.4e21 FLOPs, D = 130B Data ~tokens 608B 513B 444B 389B 250B 220B 130B Data ~tokens 608B 513B 444B 389B 250B 220B 130B (b) Fixed C and reusing data
Figure 2: Performance of N ≈ 7B models trained with varying data sizes D and activation rate r a . The optimal activation rate, r * * a = 20%, align with the findings for the 2B models (Figure 1). Additionally, compared to training on the unique dataset, the strict data reuse scheme shows only a slight performance reduction, while the loose scheme often yields better performance.
this section cite: ['b0', 'b19']

Section: OPTIMAL AR POINT
Focusing on the 2B models trained on the same data size D = 114B as shown by the green solid line in Figure 1a, we observe that the performance gain depends non-linearly on the training budget C. Specifically, the gain is more significant within a relatively low range of r a . Starting from points on this curve and fixing the corresponding r a values, increasing D results in linearly diminishing BPC, as indicated by the dashed lines in Figure 1a. These results confirm the existence of an optimal AR point, r * * a , that remains consistent regardless of D when N is unchanged. When plotting the results from a fixed training compute perspective (C = C 0 = 9.1e20) in Figure 1b, we clearly observe that the optimal AR point is approximately r * * a ≈ 20%.
this section cite: []

Section: COMPARISON WITH DENSE MODELS
To compare with dense models, we trained two dense models (Table 13 in Appendix E) with N ≈ 2B parameters and training budgets C 1 = C 0 = 9.1e20 and C 2 = 1.64e21 ≈ 2C 1 . The second model (C 2 ) is included for comparison to account for the typically lower Model FLOPs Utilization (MFU) in MoE training. This reduced MFU arises from load balancing and expert parallelism mechanisms that limit large-block matrix computations. As illustrated in Figure 1b, MoE models outperform their C 1 dense counterparts when r a falls within a specific range (approximately 15% to 48% for 2B models). For instance, the MoE model with the optimal AR point r * * a = 20% achieves a BPC value that is 0.0064 lower than its C 1 dense counterpart and only 0.0049 higher than the C 2 dense model. This demonstrates the existence of an optimal activation rate region R * a , where MoE models with r a ∈ R * a can outperform their dense counterparts under the same training budget C and approach the performance of dense models with double the compute. However, the performance gains of MoE models rely on a substantial increase in data, e.g., a 4.6× larger data size at r a = r * * a = 20%. To mitigate this additional data requirement, we explore a data reuse strategy in § 6.
this section cite: []

Section: CONSISTENCY OF OPTIMAL AR
As illustrated in Fig. 2, an optimal AR point r * * a also exists for 7B models. Surprisingly, r * * a remains consistent for both 2B and 7B models at approximately 20%, suggesting that r * * a is independent of model size. This finding contradicts established studies on MoE sparsity (Abnar et al., 2025), which proposes that optimal sparsity (defined as
(E -K) /E) is directly proportional to model size. Nevertheless, our experiments were conducted with strictly controlled variables using optimized backbones, leading us to believe that our conclusions are both reliable and scalable (see Appendix B 0.15 0.20 0.25 0.30 Activation rate ra 0.46 0.47 0.48 0.49 Accuracy Average 0.15 0.20 0.25 0.30 Activation rate ra 0.570 0.575 0.580 0.585 0.590 0.595 0.600 Knowledge 0.15 0.20 0.25 0.30 Activation rate ra 0.43 0.44 0.45 0.46 Reasoning Data ~tokens 511B 443B 390B 316B 250B 221B Data ~tokens 511B 443B 390B 316B 250B 221B 0.15 0.20 0.25 0.30 Activation rate ra 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 Accuracy Average 0.15 0.20 0.25 0.30 Activation rate ra 0.22 0.24 0.26 0.28 Knowledge 0.15 0.20 0.25 0.30 Activation rate ra 0.175 0.200 0.225 0.250 0.275 0.300 0.325 Reasoning 0.15 0.20 0.25 0.30 Activation rate ra 0.36 0.38 0.40 0.42 0.44 Accuracy Comprehensive 0.15 0.20 0.25 0.30 Activation rate ra 0.08 0.10 0.12 0.14 0.16 Math 0.15 0.20 0.25 0.30 Activation rate ra 0.060 0.065 0.070 0.075 0.080 0.085 0.090 Code 7B MoEs unique data 7B MoEs strict data reuse D = 68B 7B Dense unique data D = 130B for a more detailed discussion). To further validate the possible universality of our findings, we conducted experiments on N ≈ 3B models and achieved similar results (see Fig. 5 in Appendix E).
Expert specialization is another significant potential of MoEs in addition to remarkable scalability, where each expert focuses on learning specific features or patterns within the data. However, this attribute has not yet been clearly observed even in state-of-the-art MoE LLMs (Lo et al., 2024;Zhang et al., 2024), and effective approaches to achieve it remain under-explored. Based on our observation that MoEs outperform dense models when r a ∈ R * a , we conjecture a relationship between the optimal AR region and the degree of expert specialization. Specifically: 1) When the activation rate is too low (r a < 10%), the model lacks sufficient parameters to store knowledge effectively. 2) When the activation rate is relatively high (r a > 50%), more experts are typically activated, which may lead to weaker specialization. An activation rate within the optimal region R * a likely facilitates a higher degree of expert specialization, thereby enhancing the MoE model's performance compared to its dense counterpart. We leave further analysis of this potential relationship for future work.
this section cite: ['b0', 'b32', 'b62']

Section: DATA REUSE STRATEGY
As discussed in § 5.2, MoEs outperform their dense counterparts but require additional data. To eliminate this increased data demand, we investigate data reusability by training models for multiple epochs using a fixed, smaller dataset size D. We extract a sub-dataset of size D from the original training dataset. At the beginning of each epoch after the first, the data are shuffled.
Setup. We explore two distinct schemes, termed the strict and the loose data reuse schemes.
For the strict scheme, our aim is to ensure that both MoE and dense models are trained under completely equal conditions with respect to N , D, and C. Given a fixed D, the number of training epochs (ranging from 1.7 to 8.3 in our 3B and 7B model experiments) increases as r a decreases (hence decreasing M ) to maintain the compute budget C. The experimental settings are detailed in Table 14, 15, 16, in Appendix E, which are mostly the same as those in § 5.2, except for the training data used. Specifically, we set D = 65B and 114B for the 3B models, and D = 68B for the 7B models, corresponding to the data used for training the dense models.  (Lai et al., 2023) 5.70 6.90 4.60 HumanEval (Chen et al., 2021) 22.56 21.34 21.95 LeetCode (Coignion et al., 2024) 1.49 1.67 1.49 LiveCodeBench (Jain et al., 2024) 4.21 4.63 3.37
For the loose scheme, we relax the constraint of identical D by fixing the number of training epochs to 2 for all r a , hence D = 0.5D, where the exact value of D corresponds to the specific r a . We conduct experiments on 7B models and the experimental settings are in Table 17 in Appendix E.
Results. The performance under the strict scheme is illustrated by the blue dashed lines in Fig. 2b and 5 in Appendix E. Reusing data D only marginally diminishes performance compared to training on the unique dataset D for a single epoch, and MoE models continue to outperform dense baselines. Moreover, increasing D further narrows the performance gap. The similarity in curve shapes indicates that the optimal activation rate r * * a remains unchanged. These findings address the primary question posed at the beginning of this paper: Mixture-of-Experts can surpass dense LLMs under equal total parameters, compute, and data constraints, provided that the backbones are optimized and r a ∈ R * a . We further discuss the reuse-vs-unique trade-offs below. Discussion. Prior works have explored the effectiveness of multi-epoch training for dense and MoE models. Muennighoff et al. (2023) developed a scaling law that accounts for the number of repeated tokens and found negligible loss for repeating up to 4 epochs compared to training on unique data, whereas Hernandez et al. (2022) showed degradation for dense models. Xue et al. (2023) noticed no significant gain for MoEs with repeated training when high-quality data is insufficient. We emphasize that our goal here is not to claim that multi-epoch training is generally better for MoEs; rather, we examine whether MoEs can still surpass dense models when the unique-token budget is fixed. Concretely, under the loose scheme (green dashed line), for each r a we keep the consumed-token budget D fixed and compare (i) a 2-epoch run on a subset of size D = 0.5D (thus processing D tokens with reuse) and (ii) a 1-epoch run that consumes D tokens without reuse (i.e., D unique tokens), where both are sampled from the same data recipe/distribution. We find that the 2-epoch reuse setting can match, and sometimes slightly improve over, the 1-epoch unique-token baseline at several suboptimal r a points; however, for the 7B models, at the most important optimal point (r a ≈ 20%) it does not exceed the 1-epoch unique-token baseline. In all case for the 7B models, using more than two epochs (multi-epoch) consistently degrades performance. For the 3B models under the strict reuse setting, at a fixed r a , using a larger unique-token budget (e.g., D = 114B) consistently outperforms a smaller one (e.g., D = 65B), aligning with the intuition that more unique data is better even under multi-epoch training (see Fig. 5 and Tables 1516in Appendix E). Moreover, increasing the unique-token budget from 114B (trained for 2-3 epochs) to 309B (trained for 1 epoch) yields only a marginal improvement under the same token-consumption budget (Fig. 5). Moreover, under a fixed consumed-token budget (D = 309B), increasing the unique-token budget from D = 114B (trained for ∼2-3 epochs with reuse) to 309B (trained for 1 epoch without reuse) yields only a marginal improvement (Fig. 5). Overall, these results suggest that MoE models may tolerate mild repetition (around two epochs) under a fixed token-consumption budget, but additional repetition becomes harmful.
this section cite: ['b28', 'b4', 'b8', 'b22', 'b34', 'b18', 'b56']

Section: ANALYSIS OF DOWNSTREAM PERFORMANCE
To assess whether the optimal ARs generalize to downstream tasks, we conduct SFT on our 7B pre-trained models (trained w/ and w/o strict data reuse) and evaluate both the pre-trained models and SFT-ed models on a total number of 29 benchmarks (see Fig. 3 and Tab. 2), including categories such as reasoning and knowledge. The comprehensive list of benchmarks can be found in Appendix D. For all SFT trainings, we use a fixed data size D, and thus varying C across models with different r a . The 7B dense model trained with twice the compute is included for comparison.
this section cite: []

Section: MoE vs.
Dense at r a = r * * a . For both PT and SFT models, MoEs outperform their dense equivalents across all benchmark types when r a = 20%. This result aligns with upstream findings that the optimal activation rate (r * * a ) is 20%, highlighting the very possible universality of the optimal AR point across different training phases and data domains. Furthermore, r * * a remains unchanged during SFT, even with varying C, suggesting that the SFT data size may have an upper limit for performance improvement, provided the PT model is adequately trained. Additionally, the average performance at r a ̸ = r * * a is consistently inferior to that of dense models, underscoring the critical role of the optimal AR point.
this section cite: []

Section: MoE vs.
Dense at r a < r * * a . Dense models outperform MoEs across all domains for PT models. After SFT, MoEs overtake dense models on comprehensive and knowledge tasks. However, a notable performance gap remains in math, highlighting the PT stage's importance for math ability.
this section cite: []

Section: MoE vs. Dense at r a > r * *
a . Compared to the dense models, MoEs perform better on knowledge but worse on reasoning for PT models, and usually slightly underperform after SFT.
Sparser vs. Denser. For PT models, denser MoEs (r a > r * * a ) outperform or match the performance of sparser MoEs (r a < r * * a ) across all domains, consistent with Figure 2b, regardless of data reuse. When training on unique data, sparser MoEs perform better on knowledge. Notably, denser MoE performance significantly degrades with data reuse, especially for SFT.
Impact of data reuse. For both PT and SFT models, data reuse has little impact on reasoning but causes significant degradation in knowledge performance. Surprisingly, at r a = r * * a , the SFT-ed MoEs trained with data reuse outperform both MoEs and dense models trained on unique data. This implies that a model can master reasoning skills (rather than merely memorizing information (Hu et al., 2024)) with a relatively small dataset (Muennighoff et al., 2025;Wang et al., 2025) and further enhance its capabilities through multiple training epochs.
this section cite: ['b21', 'b35', 'b49']

Section: CONCLUSION AND FUTURE WORKS
In this paper, we propose a three-step experimental methodology to investigate whether MoEs can surpass their dense counterparts under the same constraints on total parameters, compute, and data. By optimizing the architecture, identifying the optimal activation rate region, and reusing data, we arrive at a positive answer to this question. Future work will explore how optimal activation rates enhance model capabilities and whether similar conclusions hold for other training methods like upcycling (Komatsuzaki et al., 2023) and MoEfication (Zhang et al., 2022). We hope this work offers valuable insights for the architectural design of next-generation models.
this section cite: ['b25', 'b63']

Section: LIMITATIONS
The limitations of this work include: 1) Hindering by the high computational cost, we did not train models larger than 7B. 2) As described in § 4, we focus mainly on the impact of several main components of MoEs, but fix the rest to narrow the scale of experiments. 3) Exploration of other elements can provide further comprehensive guidance for the architectural design of MoEs.
follows:
s i (x) = Softmax i (W g x), (11) T (x) = TopK(s(x); K), (12
) g i (x) = { s i (x) if i ∈ T (x), 0 otherwise, (13
) y = E ∑ i=1 g i (x) • E i (x),(14)
where s i (x) denotes the gate score for the i-th expert. Unless otherwise specified, we use the above non-normalized Top-K gating in this paper (i.e., we do not renormalize the K selected scores to sum to 1). For completeness, the commonly used Top-K normalization is
gi (x) =    g i (x) ∑ j∈T (x) g j (x) if i ∈ T (x), 0 otherwise. (15
)
We also adopt the standard auxiliary load-balancing loss (used throughout our experiments) to encourage uniform expert utilization. Given a minibatch B, define
f i = 1 |B| ∑ x∈B I[i ∈ T (x)],(16)
p i = 1 |B| ∑ x∈B s i (x),(17)
and compute
L balance = E E ∑ i=1 f i p i , L total = L CE + λL balance . (18
)
this section cite: []

Section: B EXTENDED ANALYSIS ON RELATED WORK
We notice a concurrent work (Abnar et al., 2025) studied scaling law for optimal MoE sparsity. We highlight the differences between our work and theirs as follows:
• Formulation: We define "sparsity" as the activation rate r a = Na /N, which is a more general definition than that proposed by Abnar et al. (2025), namely the ratio of inactive experts to the total number of experts, (E -K) /E.
• Methodology: Given that the activation rate r a does not depend on the underlying model architecture, we can thus easily take into consideration other components such as shared expert and build all our models upon the optimized architecture proposed in § 4. This ensures the observed performance differences solely attribute to the varying activation rates.
• Sufficient training: Our main comparisons operate in a sufficiently trained regime (often beyond the compute-optimal point). For example, the 2B MoE models in Table 9 have D /N ranging from 53 to 252, and the 7B MoE models in Table 11 have D /N ranging from 20 to 93, while our dense baselines satisfy D /N ≥ 20 (Table 13), aligning with the common compute-optimal guideline (Hoffmann et al., 2022). This perspective is practically important since industrial model design often cares about the best achievable performance at a fixed-N budget (e.g., training a 7B model on trillions of tokens), and it also enables more reliable downstream SFT comparisons that would be less meaningful with undertrained checkpoints.
• Perspective under limited resources: Scaling-law studies often adopt broad sweeps that trade depth for breadth under limited compute, which can lead to undertrained large-model settings (e.g., Abnar et al. (2025) uses C = 1e20 per run for a sweep up to 30B, and Ludziejewski et al. (2025) uses at most 80B tokens). In contrast, our 7B study allocates substantially more compute per setting (e.g., multiple runs with C ≈ 2.86e21 or C ≈ 5.45e21; Table 12 and Table 13), prioritizing a high-D /N regime to more deeply investigate the fixed-N & fixed-C question motivated by deployment memory constraints. • Conclusion: We discover an optimal activation rate that appears to be independent of model sizes, whereas Abnar et al. (2025) find that the optimal sparsity increases with model size.
Our conclusion regarding a consistent optimal activation rate contradicts the findings of Abnar et al. (2025). While we believe our findings are reliable, given that our experiments are conducted with strictly controlled variables using optimized backbones and sufficient training data, we acknowledge the possibility that the optimal r a might slightly shift for model sizes significantly beyond our studied range (i.e., N >> 7B). Nevertheless, we contend that the optimal r a can be considered consistent within a certain range of model sizes, in contrast to the significant changes reported by Abnar et al. (2025).
this section cite: ['b0', 'b0', 'b19', 'b0', 'b33', 'b0', 'b0', 'b0']

Section: C PRETRAINING DATA RECIPE
For reproducibility, we provide the mixture ratios of our pretraining corpus and a comparison with the LLaMA-1 recipe (Touvron et al., 2023a). Our recipe is intentionally close to a LLaMA-1-style mixture, and the corresponding data sources have public counterparts.
this section cite: []

Section: D COMPREHENSIVE LIST OF BENCHMARKS
To assess whether the optimal ARs generalize to downstream tasks, we conduct SFT on our 7B pre-trained models (trained w/ and w/o strict data reuse) and evaluate both the pre-trained models and SFT-ed models on a total number of 29 benchmarks (Figure 3). The comprehensive list of benchmarks is provided here.
For pre-trained models, we evaluate on:
• Knowledge: BBH (Suzgun et al., 2023), PIQA (Bisk et al., 2019), SCIQ (Welbl et al., 2017), SIQA (Sap et al., 2019) • Reasoning: ARC (Clark et al., 2018), BoolQ (Clark et al., 2019), CLUE (Xu et al., 2020), DROP (Dua et al., 2019), HellaSwag (Zellers et al., 2019), NaturalQA (Kwiatkowski et al., 2019), RACE (Lai et al., 2017), WinoGrande (Sakaguchi et al., 2020), XTREME (Hu et al., 2020) For SFT-ed models, we evaluate on:
• Comprehensive: AGIEVAL (Zhong et al., 2023), BBH
• Knowledge: CMMLU (Li et al., 2024), MMLU (Hendrycks et al., 2021b), MMLU-Redux (Gema et al., 2024), MMLU-Pro (Wang et al., 2024) • Reasoning: DROP, LiveBench (White et al., 2024), MuSR (Sprague et al., 2023) • Math: GAOKAO-Math24 (Zhang et al., 2023), GSM8K (Cobbe et al., 2021) • Code: APPS (Hendrycks et al., 2021a), DS-1000 (Lai et al., 2023), HumanEval (Chen et al., 2021), LeetCode (Coignion et al., 2024), LiveCodeBench (Jain et al., 2024)       Figure 5: Performance of N ≈ 3B models trained with varying data sizes D and activation rate r a . The optimal activation rate, r * * a = 20%, aligns with the findings for the 2B models (Figure 1). Additionally, compared to training on the unique dataset, the data reuse scheme shows only a slight performance reduction. To save computational costs, only one model trained on unique data is included for reference.
this section cite: ['b45', 'b3', 'b52', 'b41', 'b6', 'b5', 'b55', 'b12', 'b59', 'b26', 'b27', 'b40', 'b20', 'b64', 'b30', 'b50', 'b53', 'b44', 'b61', 'b7', 'b28', 'b4', 'b8', 'b22']

Section: E MORE EXPERIMENTAL RESULTS

this section cite: []

Section: References
Ref_id:b0 Title: Alaaeldin Mohamed Elnouby Ali, Josh Susskind, and Vimal Thilak. Parameters vs flops: Scaling laws for optimal sparsity for mixture-of-experts language models Year: (2025)
Ref_id:b1 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b2 Title: Qwen technical report Year: (2023)
Ref_id:b3 Title: Reasoning about physical commonsense in natural language Year: (2019)
Ref_id:b4 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b5 Title: Boolq: Exploring the surprising difficulty of natural yes/no questions Year: (2019)
Ref_id:b6 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b7 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b8 Title: A performance study of llm-generated code on leetcode Year: (2024)
Ref_id:b9 Title: Deepseekmoe: Towards ultimate expert specialization in mixtureof-experts language models Year: (2024)
Ref_id:b10 Title: Deepseek llm: Scaling open-source language models with longtermism Year: (2024)
Ref_id:b11 Title: Deepseek-v3 technical report, 2024b Year: ()
Ref_id:b12 Title: Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs Year: (2019)
Ref_id:b13 Title: Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity Year: (2022)
Ref_id:b14 Title: Are we done with mmlu? arXiv preprint Year: (2024)
Ref_id:b15 Title: Upcycling large language models into mixture of experts Year: (2024)
Ref_id:b16 Title: Measuring coding challenge competence with apps Year: ()
Ref_id:b17 Title: Measuring massive multitask language understanding Year: (2021)
Ref_id:b18 Title: Scaling laws and interpretability of learning from repeated data Year: (2022)
Ref_id:b19 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b20 Title: Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation Year: (2020)
Ref_id:b21 Title: Case-based or rule-based: How do transformers do the math? Year: (2024)
Ref_id:b22 Title: Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code Year: (2024)
Ref_id:b23 Title:  Year: (2024)
Ref_id:b24 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b25 Title: Sparse upcycling: Training mixtureof-experts from dense checkpoints Year: (2023)
Ref_id:b26 Title: Natural questions: a benchmark for question answering research Year: (2019)
Ref_id:b27 Title: RACE: Large-scale ReAding comprehension dataset from examinations Year: (2017-09)
Ref_id:b28 Title: Ds-1000: A natural and reliable benchmark for data science code generation Year: (2023)
Ref_id:b29 Title: Gshard: Scaling giant models with conditional computation and automatic sharding Year: (2021)
Ref_id:b30 Title: Cmmlu: Measuring massive multitask language understanding in chinese Year: (2024)
Ref_id:b31 Title: Predictable scale: Part i, step law -optimal hyperparameter scaling law in large language model pretraining Year: (2025)
Ref_id:b32 Title: A closer look into mixture-of-experts in large language models Year: (2024)
Ref_id:b33 Title: Joint moe scaling laws: Mixture of experts can be memory efficient Year: (2025)
Ref_id:b34 Title: Scaling data-constrained language models Year: (2023)
Ref_id:b35 Title: Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling Year: (2025)
Ref_id:b36 Title: Train short, test long: Attention with linear biases enables input length extrapolation Year: (2022)
Ref_id:b37 Title:  Year: (2024)
Ref_id:b38 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b39 Title: Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale Year: (2022)
Ref_id:b40 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2020)
Ref_id:b41 Title: Socialiqa: Commonsense reasoning about social interactions Year: (2019)
Ref_id:b42 Title: Glu variants improve transformer Year: (2020)
Ref_id:b43 Title: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer Year: (2017)
Ref_id:b44 Title: Testing the limits of chain-of-thought with multistep soft reasoning Year: (2023)
Ref_id:b45 Title: Challenging big-bench tasks and whether chain-of-thought can solve them Year: (2023)
Ref_id:b46 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b47 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b48 Title: Attention is all you need Year: (2017)
Ref_id:b49 Title: Reinforcement learning for reasoning in large language models with one training example Year: (2025)
Ref_id:b50 Title: Mmlu-pro: A more robust and challenging multitask language understanding benchmark Year: (2024)
Ref_id:b51 Title: Skywork-moe: A deep dive into training techniques for mixtureof-experts language models Year: (2024)
Ref_id:b52 Title: Crowdsourcing multiple choice science questions Year: (2017)
Ref_id:b53 Title: Livebench: A challenging, contaminationlimited llm benchmark Year: (2024)
Ref_id:b54 Title: Mixture of experts with attention router Year: (2024)
Ref_id:b55 Title: Clue: A chinese language understanding evaluation benchmark Year: (2020)
Ref_id:b56 Title: To repeat or not to repeat: Insights from scaling llm under token-crisis Year: (2023)
Ref_id:b57 Title: Openmoe: An early effort on open mixture-of-experts language models Year: (2024)
Ref_id:b58 Title: Zhifang Guo, and Zhihao Fan. Qwen2 technical report Year: (2024)
Ref_id:b59 Title: Hellaswag: Can a machine really finish your sentence? Year: (2019)
Ref_id:b60 Title: Root mean square layer normalization Year: (2019)
Ref_id:b61 Title: Evaluating the performance of large language models on gaokao benchmark Year: (2023)
Ref_id:b62 Title: Diversifying the expert knowledge for task-agnostic pruning in sparse mixture-of-experts Year: (2024)
Ref_id:b63 Title: Moefication: Transformer feed-forward layers are mixtures of experts Year: (2022)
Ref_id:b64 Title: Agieval: A human-centric benchmark for evaluating foundation models Year: (2023)
Ref_id:b65 Title: St-moe: Designing stable and transferable sparse expert models Year: (2022)
