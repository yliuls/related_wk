Title: ARECHO: Autoregressive Evaluation via Chain-Based Hypothesis Optimization for Speech Multi-Metric Estimation
Abstract: Speech signal analysis poses significant challenges, particularly in tasks such as speech quality evaluation and profiling, where the goal is to predict multiple perceptual and objective metrics. For instance, metrics like PESQ (Perceptual Evaluation of Speech Quality), STOI (Short-Time Objective Intelligibility), and MOS (Mean Opinion Score) each capture different aspects of speech quality. However, these metrics often have different scales, assumptions, and dependencies, making joint estimation non-trivial. To address these issues, we introduce ARECHO (Autoregressive Evaluation via Chain-based Hypothesis Optimization), a chain-based, versatile evaluation system for speech assessment grounded in autoregressive dependency modeling. ARECHO is distinguished by three key innovations: (1) a comprehensive speech information tokenization pipeline; (2) a dynamic classifier chain that explicitly captures inter-metric dependencies; and (3) a two-step confidence-oriented decoding algorithm that enhances inference reliability. Experiments demonstrate that ARECHO significantly outperforms the baseline framework across diverse evaluation scenarios, including enhanced speech analysis, speech generation evaluation, and, noisy speech evaluation. Furthermore, its dynamic dependency modeling improves interpretability by capturing inter-metric relationships. Across tasks, ARECHO offers reference-free evaluation using its dynamic classifier chain to support subset queries (single or multiple metrics) and reduces error propagation via confidence-oriented decoding.

Section: Introduction
Speech assessment and profiling are essential components in the speech processing community, owing to the inherently complex and multidimensional nature of speech signals (Huang et al., 2022;Yi et al., 2022;Shi et al., 2024;Torcoli et al., 2021). These signals typically encompass various attributes, such as clarity, naturalness, emotional expressiveness, and acoustic quality, which are challenging to characterize comprehensively. While subjective evaluation remains the gold standard for assessing speech quality due to its capacity for nuanced and context-aware judgments, it suffers from several limitations, including inter-rater variability, limited scalability, and insufficient coverage of diverse evaluation dimensions (Huang et al., 2022;Cooper et al., 2024;Zielinski et al., 2008;Loizou, 2011;Jiménez et al., 2021;Naderi et al., 2020). Consequently, objective methods have emerged as scalable and consistent alternatives that aim to approximate subjective judgments (Cooper et al., 2024;Torcoli et al., 2021;Shi et al., 2025a).
Prior work has introduced numerous specialized metrics targeting specific speech characteristics, including speech perceived quality, naturalness in synthesized speech, and paralinguistic features human evaluation remains the gold standard, with evaluations ranging from general assessments such as overall naturalness to more fine-grained dimensions such as speaking style and expressiveness (Li et al., 2024;Yang et al., 2024;Shimizu et al., 2024;Feng & Yoshimoto, 2024). However, such human evaluations face well-known limitations, including challenges in scaling, score inconsistency, and limited coverage of diverse perceptual factors (Huang et al., 2022;Zielinski et al., 2008;Loizou, 2011;Jiménez et al., 2021;Naderi et al., 2020). To overcome these limitations, recent works have introduced objective evaluation systems that are trained to predict human perceptual scores, such as mean opinion scores (MOS), using supervised learning frameworks (Falk et al., 2008;Yoshimura et al., 2016;Lo et al., 2019;Saeki et al., 2022;Huang et al., 2024c). These models have gained popularity in speech synthesis due to their scalability and ability to generalize to unseen systems. A similar trend is observed in SE, where classical signal-based metrics (e.g., signal-to-noise ratio) are widely used with simulated data experiments (Perlmutter et al., 1977;Hansen & Pellom, 1998;Xu et al., 2013). Yet, they often fail to reflect human perception accurately, especially in real-world, noisy scenarios (Hu & Loizou, 2007b). As a result, perceptually aligned objective metrics have been proposed to bridge the gap between computational evaluation and subjective judgment (Hu & Loizou, 2007a;Reddy et al., 2021;Rao et al., 2021;Yi et al., 2022).
While the above metrics are directly tied to task-specific outcomes, there is growing interest in broader frameworks for general speech assessment or profiling (Zezario et al., 2022a(Zezario et al., , 2024;;Chen & Tsao, 2022;Close et al., 2024;Zezario et al., 2022b;Kumar et al., 2023;Tjandra et al., 2025;Shi et al., 2025a,b). Such frameworks aim to extract a rich set of meta-information that spans across multiple dimensions of the speech signal. This can include not only quality and intelligibility, but also speaker traits, emotional content, and environmental context. In this work, we align with this broader vision and propose a system that moves beyond task-specific metrics to support general-purpose, multi-faceted speech evaluation. 2Multi-Metric Evaluation and Dependency Modeling. Speech evaluation inherently involves multiple dimensions, such as naturalness, intelligibility, and emotional expression, that are often correlated through shared acoustic and prosodic cues. Effectively capturing the dependencies among these metrics is therefore essential for producing faithful and insightful assessments of speech quality.
Recent work has increasingly focused on general-purpose evaluation frameworks that unify diverse metrics and applications (Zezario et al., 2022a(Zezario et al., ,b, 2024;;Chen & Tsao, 2022;Close et al., 2024;Kumar et al., 2023;Tjandra et al., 2025;Shi et al., 2025a,b). These systems aim to provide scalable, task-agnostic profiling across various speech properties, including quality, speaker traits, emotional content, and environmental conditions. Frameworks such as UniVERSA (Shi et al., 2025b) and TorchSquim (Kumar et al., 2023) support multi-metric prediction, but typically treat each metric independently, overlooking the latent correlations that can enhance performance and interpretability. For instance, improvements in naturalness often align with gains in intelligibility or perceived emotion due to overlapping signal characteristics.
In parallel, captioning-based approaches have explored generating natural language rationales to explain metric predictions (Ghosh et al., 2024;Xie et al., 2025;Ghosh et al., 2025;Wang et al., 2025b;Deshmukh et al., 2025;Ma et al., 2025;Kuan & Lee, 2025;Wen et al., 2025;Wang et al., 2025a;Huang et al., 2024aHuang et al., , 2025;;Chen et al., 2025). While these methods offer interpretability, they often rely on pretrained LLMs and may struggle with precision or non-textual metrics. ARECHO instead focuses on structured, metric-token-based modeling, enabling exact scoring and more fine-grained control.
this section cite: ['b39', 'b95', 'b74', 'b83', 'b39', 'b16', 'b54', 'b43', 'b59', 'b16', 'b83', 'b52', 'b94', 'b77', 'b23', 'b39', 'b54', 'b43', 'b59', 'b21', 'b96', 'b53', 'b72', 'b62', 'b32', 'b93', 'b69', 'b66', 'b95', 'b41', 'b12', 'b15', 'b51', 'b82', 'b12', 'b15', 'b51', 'b82', 'b51', 'b28', 'b92', 'b29', 'b18', 'b56', 'b88', 'b38', 'b8']

Section: ARECHO

this section cite: []

Section: Base Task Formulation
We adopt the general task formulation from (Shi et al., 2025b;Kumar et al., 2023) for speech multimetric estimation, with an extension to explicitly handle categorical metrics in addition to numerical ones. Let i th paired sample from dataset D be represented as (S i , Y i ), where S i denotes a singlechannel speech signal and Y i denotes the set of associated evaluation metrics. Each Y i = {y i b } b∈B consists of multiple metric values, where b is the index of a metric. Here, B is a set of indices, which can be partitioned into indices for numerical metrics B num and indices for categorical metrics B cat , i.e., B = B num ∪ B cat . 3The core model predicts all metrics directly from the input signal:
Ŷ i = f (S i ),(1)
where f (•) denotes the base prediction model and Ŷ i = {ŷ i b } b∈B represents the predicted metrics. 4The training objective for the multi-metric estimation minimizes the prediction error across all metrics using a regression (n-norm) and cross-entropy losses for B num and B cat , respectively:
L i B = L i Bnum + L i Bcat = b∈Bnum ||y i b -ŷi b || n + b ′ ∈Bcat CE(y i b ′ , ŷi b ′ ),(2)
where n = 1 in our experiments and CE(•) is the cross-entropy loss function.
Building on the task formulation introduced above and the challenges outlined in Sec. 1, including heterogeneous metric scales, partial supervision, and the lack of inter-metric reasoning, we propose ARECHO, a flexible and robust framework for multi-metric speech evaluation. ARECHO addresses these limitations through three key algorithmic components: (1) a comprehensive tokenization framework that standardizes diverse metric types into a unified representation space;
(2) a dynamic classifier chain that captures inter-metric dependencies via flexible, data-driven sequencing; and (3) a two-step confidence-oriented decoding strategy that enhances prediction reliability under uncertainty. Each of these components is described in detail in the following subsections.
this section cite: ['b51']

Section: Tokenizing Everything
To address the above Challenge I of heterogeneity across evaluation metrics, ranging from unbounded continuous scores to discrete categorical labels, we introduce a unified tokenization framework that transforms all metric values into a shared discrete representation space. This formulation enables ARECHO to model metric prediction as a sequence generation task over tokens, allowing consistent treatment of diverse metric types and facilitating autoregressive dependency modeling.
Given a sample (S i , Y i ), where Y i = {y i b } b∈B includes both numerical and categorical metrics, we define a set of tokenization functions T = T b , where each T b maps a ground-truth value y i b to a discrete token z i b from a finite vocabulary V b :
z i b = T b (y i b ), z i b ∈ V b ,(3)
where the total vocabulary is V = b∈B (V b ).
The full tokenized label sequence for sample i becomes Z i = {z i b } b∈B , which serves as the target sequence for autoregressive prediction. For numerical metrics (i.e., b ∈ B num ), we apply quantizationbased tokenization by partitioning the value range into uniformly or adaptively spaced bins. 5 For categorical metrics (i.e., b ∈ B cat ), the tokenization is direct by mapping each class label to a unique token.
The inverse function
T -1 b : V b → R or T -1 b : V b → C b (where C b is the label set for categorical metric b ∈ B cat ) is used to reconstruct predictions: ŷi b = T -1 b (ẑ i b ), ẑi b ∈ V b .(4)
This formulation allows unified modeling across metric types, improves dependency learning, and enables flexible inference (see Sec. 3.3).
this section cite: []

Section: Dynamic Classifier Chain
To address Challenge III: Dependency Modeling with Flexible Control, we introduce a dynamic classifier chain architecture that models inter-metric dependencies while allowing flexible prediction orders. Our design is inspired by the multi-label classifier chain model (Read et al., 2011(Read et al., , 2021)), but generalizes it to a token-level formulation suitable for autoregressive sequence modeling.
this section cite: ['b67', 'b68']

Section: Motivating Example.
Consider a case where we want to predict three metrics: Gender, Emotion, and MOS. Instead of predicting these values independently, our model constructs a target sequence:
T full = [<Gender>, Male, <Emotion>, Happy, <MOS>, 3.78]
Here, tokens like <Gender> and <MOS> are metadata tokens that serve as prompts, indicating which metric the model should predict next. The following tokens, Male, Happy, 3.78, are the corresponding value tokens, representing the model's predictions for each metric. 6Formal Definition. For each metric b ∈ B, we define a metadata token m b ∈ M drawn from a finite vocabulary M. These tokens act as queries for predicting the associated values. Note that M is a separate set of metadata tokens and is disjoint from V, the value token vocabulary.
The full target sequence for training is a flat, interleaved sequence of metadata and value tokens:
T i full = [m b1 , z i b1 , m b2 , z i b2 , . . . , m b K , z i b K ],(5)
where [b 1 , . . . , b K ] is a permutation of selected metrics for a given input, and z i b k is the value token for metric b k in sample i.
Training Objective. This sequence is modeled autoregressively:
P (T i full | S i ) = 2K-1 t=0 P (x i t | T i <t , S i ),(6)
where x i t ∈ M ∪ V is either a metadata or value token, and S i is the input speech representation. Randomizing the metric order during training exposes the model to diverse conditioning patterns, helping it generalize across different evaluation needs.
The advantages of the use of metadata tokens include: Metric Identification: Metadata tokens specify which metric to predict, even when value formats overlap (e.g., many metrics return scores in similar ranges). Dependency Control: Prior metric predictions are part of the sequence history, enabling context-aware inference of subsequent metrics.
this section cite: []

Section: Flexible Inference.
During inference, the model begins with a metadata token (e.g., <Emotion>) and autoregressively generates its corresponding value token (e.g., Sad). This procedure iterates for any subset of requested metrics, allowing dynamic chain-style reasoning in arbitrary query orders. Users can either (i) specify a fixed prediction order or (ii) rely on our two-step confidence-oriented decoding in Sec. 3.4 to automatically determine an effective sequence. Such flexibility is particularly advantageous when metric availability or user interests differ across samples. Support for Partial Supervision. The autoregressive formulation also helps address Challenge II: Limited Label Availability. For samples with only a few known metrics, the training target simply omits the unavailable ones. For instance:
T i partial = [m b1 , z i b1 , m b3 , z i b3 ]
. This allows the model to learn from partially labeled data without masking or imputation.
Compared to conventional classifier chains that assume fixed label spaces and static prediction orders (Chen et al., 2018;Gerych et al., 2021), our token-level formulation generalizes the idea for more expressive and modular metric modeling. We elaborate on how this supports efficient decoding in Section 3.4.
this section cite: ['b11', 'b27']

Section: Two-step Confidence-oriented Decoding
While the dynamic classifier chain enables flexible and dependency-aware prediction of multiple metrics, it introduces challenges at inference time, particularly due to the presence of metadata tokens. Since the order of metric prompts is randomly optimized during training, the model's confidence in predicting metadata tokens can be unreliable, making it difficult to guide decoding based on their probabilities.
To mitigate the decoding instability introduced by metadata tokens in the dynamic classifier chain, we adopt a two-step confidence-oriented decoding strategy that guides inference using the confidence of value tokens instead of the less reliable metadata scores. Let B K be the set of all metrics and Tprev the current decoded prefix after k metrics, corresponding to the already-predicted subset B k ⊆ B K . With K -k metrics still to predict, the decoding for each remaining metric b ∈ B K \B k proceeds in two phases:
this section cite: []

Section: • Step 1: Preliminary Prediction.
Append the metadata token mb to the prefix, T = Tprev + mb , and obtain a provisional value token ẑb together with its softmax-based confidence Conf(ẑ b ).
• For every zb ∈ Z After all candidates for every b ∈ B K \B k are evaluated, we keep the top-B partial hypotheses (by log-likelihood) to serve as prefixes for the next metric. By revisiting low-confidence predictions through this confidence-aware beam search, the proposed strategy substantially improves the stability and accuracy of autoregressive multi-metric inference.
this section cite: []

Section: Experimental Setup

this section cite: []

Section: Datasets
To comprehensively evaluate the generalization ability, robustness, and versatility of ARECHO, we conduct experiments across a diverse set of datasets spanning multiple speech domains. These datasets are selected to represent key practical scenarios that require distinct types of evaluation metrics and pose varying challenges in terms of signal complexity, perceptual quality, and annotation availability. Specifically, we include (1) basic speech data to provide foundational coverage of speech variability across domains, (2) simulated corrupted speech datasets to test objective quality prediction under controlled noise conditions, (3) enhanced speech recordings to assess robustness to natural distortions and variability, and (4) synthesized speech datasets to evaluate naturalness and expressiveness of generated speech. We briefly describe each dataset below; additional details, including licensing and preprocessing procedures, are provided in Appendix E.
this section cite: []

Section: Basic Speech Data.
We include data sampled from the OWSM-V3 corpus (Peng et al., 2023), a large-scale aggregation of speech recognition and translation datasets. This collection provides general-purpose coverage over a wide range of speakers, styles, and domains.
this section cite: ['b61']

Section: Corrupted Speech Data (Simulation).
We used simulated SE data generated via the URGENT2024 challenge (Zhang et al., 2024) training data generation script 7 . Furthermore, we also include, for training purposes only, the wVoice Bank+DEMAND Veaux et al. (2013); Thiemann et al. (2013) benchmark dataset for speech denoising.
this section cite: ['b85', 'b81']

Section: Enhanced Speech Data.
Together with the simulated data described previously, we used the blind test set from the URGENT2024 challenge. It includes both simulated and real-world recordings with speech corrupted by one or more of the distortions in the same way as mentioned above. These recordings were enhanced by various participants' submitted systems in the URGENT2024 challenge.
this section cite: []

Section: Synthesized Speech Data.
We utilize two benchmark datasets: VoiceMOS 2022 Challenge (Huang et al., 2022) and NISQA (Mittag et al., 2021). VoiceMOS 2022 challenge was developed for the VoiceMOS Challenge 2022, comprising a diverse collection of synthetic speech generated by various TTS systems. NISQA (Mittag et al., 2021) dataset also contains synthesized speech samples with corresponding quality ratings across multiple perceptual dimensions.
Given these sources, we construct unified training, development, and test sets by carefully sampling across all domains. This curation ensures that the model is exposed to a balanced mixture of evaluation metrics and acoustic conditions, promoting generalization across tasks.
To further examine the effect of training data scale, we prepare two training configurations: a smaller Base set (308.77 hours) and a larger Scale set (2137.74 hours). Each is paired with a shared development set (18.65 hours). Evaluation is performed using four domain-specific test sets corresponding to the main application areas: simulated enhancement (4.51 hours), enhanced speech data (30.12 hours), and speech synthesis-related data (3.46 hours). Full statistical details are available in the Appendix E.5.
this section cite: ['b39', 'b58', 'b58']

Section: Model Setups

this section cite: []

Section: Metrics in ARECHO.
To ensure broad coverage and comprehensive speech profiling, ARECHO incorporates a diverse set of evaluation metrics from two main sources: (1) automatically computed metrics derived from existing models or algorithms, and (2) pre-annotated information extracted from dataset metadata or human subjective evaluations.
For the first category, we employ the VERSA toolkit (Shi et al., 2025a) to estimate 47 independent metrics, 25 dependent metrics, and 7 non-matching metrics. 8 For the second category, we include 8 ground-truth metrics derived from dataset annotations, such as language labels, emotional categories, and human-annotated MOS scores. In total, ARECHO models 87 metrics, comprising 65 numerical and 22 categorical metrics.foot_8
Baseline Setup. We adopt the UniVERSA model (Shi et al., 2025b) as our baseline. It uses a Transformer-based audio encoder built on WavLM representations (Chen et al., 2022) to extract shared speech embeddings. Each metric is then independently predicted using a metric-specific pooling layer followed by an X-vector-based prediction head (Snyder et al., 2018). Regression targets are predicted as scalars, whereas classification targets use a softmax output corresponding to the number of classes.
Tokenization. To study the effect of discrete metric modeling, we implement a variant called UniVERSA-T, where all numerical metrics are converted into classification tasks via uniform quantization tokenization, using the same architecture as the original UniVERSA.
Proposed Model Setup. For ARECHO, we retain the same audio encoder as UniVERSA but replace the prediction heads with a Transformer-based decoder that autoregressively generates the full token sequence T i full as defined in Sec. 3.3. All models are trained under both the Base and Scale training configurations. Detailed architecture specifications, training procedures, and decoding hyperparameters are provided in Appendix G. Additional ablation studies exploring the effects of model components and training configurations can be found in Appendix I.
this section cite: ['b78']

Section: Evaluation
We adopt standard evaluation metrics for regression and classification tasks. For each numerical metric, we compute the mean squared error (MSE), linear correlation coefficient (LCC), and Kendall's tau (KTAU). For each categorical metric, we report accuracy (ACC) and F1 score. Evaluation scores are averaged across all numerical and categorical metrics, respectively, to provide overall regression and classification performance. 10Table 1: Main experimental results for comparison between baseline and ARECHO. The "Domain" indicates the evaluation set used for the model assessment.
Data Domain Model Token Chain Regression Metrics Classification Metrics MSE (↓) LCC (↑) KTAU (↑) Acc (↑) F1 (↑) Base Dev. UniVERSA ✗ ✗ 160.06 0.69 0.53 0.68 0.42 UniVERSA-T ✓ ✗ 40.95 0.78 0.68 0.70 0.46 ARECHO ✓ ✓ 25.73 0.86 0.72 0.71 0.51 Enhanced UniVERSA ✗ ✗ 61.54 0.71 0.54 0.69 0.43 UniVERSA-T ✓ ✗ 27.34 0.81 0.68 0.70 0.47 ARECHO ✓ ✓ 20.58 0.84 0.69 0.72 0.51 Corrupted UniVERSA ✗ ✗ 170.65 0.61 0.48 0.70 0.46 UniVERSA-T ✓ ✗ 77.72 0.74 0.67 0.71 0.50 ARECHO ✓ ✓ 44.22 0.82 0.70 0.72 0.55 Synthesized UniVERSA ✗ ✗ 58.79 0.76 0.54 0.69 0.45 UniVERSA-T ✓ ✗ 8.10 0.84 0.68 0.72 0.50 ARECHO ✓ ✓ 4.99 0.91 0.78 0.79 0.65 Avg. Test UniVERSA ✗ ✗ 96.99 0.69 0.52 0.69 0.45 UniVERSA-T ✓ ✗ 37.72 0.79 0.68 0.71 0.49 ARECHO ✓ ✓ 23.26 0.86 0.72 0.74 0.57 Scale Dev. UniVERSA ✗ ✗ 116.01 0.89 0.74 0.73 0.49 UniVERSA-T ✓ ✗ 27.98 0.86 0.75 0.74 0.52 ARECHO ✓ ✓ 29.61 0.86 0.76 0.75 0.52 Enhanced UniVERSA ✗ ✗ 43.05 0.84 0.67 0.72 0.47 UniVERSA-T ✓ ✗ 69.94 0.80 0.71 0.74 0.50 ARECHO ✓ ✓ 32.63 0.83 0.73 0.75 0.53 Corrupted UniVERSA ✗ ✗ 151.97 0.88 0.75 0.75 0.54 UniVERSA-T ✓ ✗ 39.80 0.77 0.74 0.76 0.54 ARECHO ✓ ✓ 34.37 0.84 0.76 0.77 0.56 Synthesized UniVERSA ✗ ✗ 6.46 0.84 0.65 0.71 0.47 UniVERSA-T ✓ ✗ 8.23 0.84 0.68 0.73 0.49 ARECHO ✓ ✓ 8.63 0.85 0.72 0.75 0.54 Avg. Test UniVERSA ✗ ✗ 67.16 0.86 0.70 0.73 0.50 UniVERSA-T ✓ ✗ 39.32 0.82 0.72 0.74 0.51 ARECHO ✓ ✓ 25.21 0.85 0.74 0.76 0.54 5 Experimental Results Overall Performance. Table 1 summarizes the overall performance of the proposed ARECHO model compared to baselines. Across both the Base and Scale training configurations, ARECHO consistently and significantly outperforms the UniVERSA and UniVERSA-T baselines on the majority of evaluation metrics. 11 These results highlight the effectiveness of ARECHO's dynamic classifier chain and confidence-oriented decoding strategy in capturing inter-metric dependencies and improving prediction robustness. Notably, the improvements observed with UniVERSA-T over UniVERSA demonstrate the benefit of tokenizing numerical metrics, validating our unified representation approach. Building upon this, ARECHO achieves further gains by leveraging structured autoregressive modeling, which enables more informed and context-aware metric prediction.
Effects of Data Scaling. The Scale training set introduces greater domain imbalance, with a heavier emphasis on corrupted speech compared to other scenarios. 12 Under this condition, ARECHO delivers substantial improvements on corrupted speech evaluation, showcasing its strong modeling capacity in data-rich domains. However, the gains are comparatively smaller on synthesized and enhanced speech, suggesting that domain balance remains an important factor for achieving broad generalization.
Observations on Baseline Behavior. While the original UniVERSA model underperforms in most settings, it still shows relative strength in modeling fine-grained numerical metrics, particularly reflected in its high LCC scores under the Scale configuration. This indicates that tokenization, while beneficial overall, may introduce granularity loss for certain regression tasks. Slight performance degradation in UniVERSA-T and ARECHO on specific numerical metrics highlights a trade-off between discrete modeling and numerical precision, an open challenge we aim to address in future work.
this section cite: []

Section: Ablation and Extended Analysis.
We conduct a set of ablations and diagnostic studies to assess robustness, efficiency, and practical behavior; full results are reported in Appendix I and J. These studies examine (i) tokenization resolution, (ii) decoding strategy, (iii) MOS-style perceptual metrics, and (iv) task-oriented metric subsets. Across all settings, ARECHO remains stable: it performs well with compact tokenizations, achieves near-optimal accuracy with simple greedy decoding, and shows strong gains on perceptual quality metrics (e.g., MOS). When trained on either task-specific metric subsets or the full union of metrics, ARECHO shows no evidence of negative transfer from "irrelevant" metrics and often improves on core targets (e.g., SRMR, SDR, human MOS). Together, these results indicate that ARECHO is both effective and adaptive: it can leverage cross-metric structure when helpful while preserving efficiency and specialization. We additionally report efficiency analysis in Appendix M, showing that ARECHO attains these benefits with substantially reduced training and inference cost relative to prior multi-metric systems.
Table 2: Top-5 and Bottom-5 metrics ranked by average position (Avg. Pos.) across three test sets. Please refer to Appendix F for more details about the metrics. Test Set Rank Metric Name Avg. Pos. Enhanced Top-1 Q-SpeakerGender 16.50 Top-2 Q-SpeechImpairment 20.35 Top-3 Q-SpeechStyle 21.47 Btm-3 SNR Simulation 163.52 Btm-2 NISQA Real MOS 167.91 Btm-1 VoiceMOS Real MOS 171.58 Corrupted Top-1 RIR Room Size 1.82 Top-2 Q-SpeechImpairment 12.65 Top-3 Q-SpeechDelivery 13.15 Btm-3 CER 167.26 Btm-2 NISQA Real MOS 170.62 Btm-1 VoiceMOS Real MOS 171.38 Synthesized Top-1 Q-Background 12.09 Top-2 NISQA Coloration 27.51 Top-3 Q-Purpose 27.95 Btm-3 Cbak 154.75 Btm-2 SNR Simulation 158.64 Btm-1 CER 161.61 Further Discussion on Dependency Modeling. The proposed ARECHO framework learns to adaptively control the inference order of metrics through its dynamic classifier chain and twostep decoding strategy, as introduced in Sec. 3.3 and Sec. 3.4. This design enables the model to prioritize more informative or stable metrics early in the prediction sequence, providing contextual cues that improve downstream metric predictions.
Table 2 highlights how ARECHO internally discovers and exploits an ordering rationale. 13 Across different test sets, metrics related to structured annotations or acoustic scene characteristics (e.g., Q-SpeakerGender, Q-SpeechImpairment, RIR Room Size) consistently appear early in the prediction sequence. These metrics are arguably easier to estimate and provide strong prior information for subsequent metrics. In contrast, more subjective and unstable metrics, such as MOS scores from Voice-MOS and NISQA, tend to appear later in the sequence. This suggests that ARECHO defers harder or noisier predictions until more context is available from previously decoded metrics.
Depending on the context, some metrics are also inherently harder to predict e.g., SNR simulation for enhanced and synthesized test sets. For enhanced speech data, the noise was removed by an SE model, while for the synthesized speech, the residual noise in the text-to-speech generated speech signal is usually very low.
This emergent ordering aligns with human intuition and supports the benefit of structured inter-metric reasoning. It also allows the system to maintain robustness when certain labels are missing or unreliable, by leveraging earlier predictions to guide later stages.
As discussed in Sec. 3.3, our proposed dynamic classifier chain supports both flexible order search and inference under arbitrary query sets. In particular, it also allows for inference with a fixed, static order of metrics when such an order is used during training. To investigate how different static orderings affect the performance of ARECHO, we provide additional analyses in the Appendix L.
this section cite: []

Section: Complexity and Efficiency.
While ARECHO provides several advantages as discussed above, it also maintains modest training cost and strong computational efficiency. By effectively utilizing partially labeled data, ARECHO achieves shorter training time comparable to standard multi-task learning with masking (e.g., UniVERSA). 14 Despite its sequential decoding process, ARECHO remains practical: computing the same set of metrics through their original estimators (e.g., LLM-based evaluators) is over 100× slower than a single forward pass of ARECHO.
this section cite: []

Section: Conclusion
We introduce ARECHO (Autoregressive Evaluation via Chain-based Hypothesis Optimization), a versatile and interpretable framework for multi-metric speech assessment. By unifying diverse metric types through tokenization and modeling inter-metric dependencies via a dynamic classifier chain, ARECHO offers a flexible and scalable alternative to traditional parallel prediction models.
Backed with our proposed two-step confidence-oriented decoding, ARECHO maintains the flexibility in decoding, which can support diverse challenging evaluation settings. Extensive experiments across corrupted, enhanced, and synthesized speech signals demonstrate that ARECHO consistently outperforms strong baselines while enabling more structured and adaptable evaluation. We believe ARECHO provides a step toward general-purpose, dependency-aware modeling for speech and potentially broader machine learning evaluation tasks. Ryandhimas E Zezario, Szu-Wei Fu, Fei Chen, Chiou-Shann Fuh, Hsin-Min Wang, and Yu Tsao. Deep learningbased non-intrusive multi-objective speech assessment model with cross-domain features. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:54-70, 2022a. Ryandhimas E Zezario, Bo-Ren Brian Bai, Chiou-Shann Fuh, Hsin-Min Wang, and Yu Tsao. Multi-task pseudolabel learning for non-intrusive speech quality assessment model. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 831-835. IEEE, 2024. Ryandhimas Edo Zezario, Szu wei Fu, Fei Chen, Chiou-Shann Fuh, Hsin-Min Wang, and Yu Tsao. MTI-Net: A multi-target speech intelligibility prediction model. In Interspeech 2022, pp. 5463-5467, 2022b. doi: 10.21437/Interspeech.2022-10828. Wangyou Zhang, Robin Scheibler, Kohei Saijo, Samuele Cornell, Chenda Li, Zhaoheng Ni, Jan Pirklbauer, Marvin Sach, Shinji Watanabe, Tim Fingscheidt, and Yanmin Qian. URGENT challenge: Universality, robustness, and generalizability for speech enhancement. In Proc. Interspeech, pp. 4868-4872, 2024. Slawomir Zielinski, Francis Rumsey, and Søren Bech. On some biases encountered in modern audio quality listening tests-a review. Journal of the Audio Engineering Society, 56(6):427-451, 2008.
A Terminology: Definition and use of "Metric"
In this work, we adopt the term "metric" to broadly refer to any form of meta-information that serves as a measurable characterization of a speech signal. This includes both conventional objective evaluation scores (e.g., PESQ, STOI, SNR) and more abstract attributes (e.g., emotion category, speaker identity, language, or environment type) that contribute to a comprehensive understanding of the signal.
Although some of these attributes may not be considered "metrics" in the traditional mathematical sense, such as binary tags or categorical labels, we use the term uniformly to emphasize their role in systematic evaluation and profiling. This unified terminology supports our goal of building a versatile and extensible framework that can evaluate various aspects of speech within a consistent and interpretable structure.
By adopting this general definition, we align with recent trends in universal speech evaluation (Shi et al., 2024), where a wide range of measurements are treated under a common umbrella to facilitate joint modeling, dependency reasoning, and scalable system benchmarking.
this section cite: ['b74']

Section: References
Ref_id:b0 Title: Image method for efficiently simulating small-room acoustics Year: (1979)
Ref_id:b1 Title: Common Voice: A massively-multilingual speech corpus Year: (2020)
Ref_id:b2 Title: The T05 system for the voicemos challenge 2024: Transfer learning from deep image classifier to naturalness mos prediction of high-quality synthetic speech Year: (2024)
Ref_id:b3 Title: Evaluation measures for ordinal regression Year: (2009)
Ref_id:b4 Title: Objective measures for speech quality testing Year: (1979)
Ref_id:b5 Title: Convolutive transfer function invariant SDR training criteria for multi-channel reverberant speech separation Year: (2021)
Ref_id:b6 Title: AIshell-1: An open-source mandarin speech corpus and a speech recognition baseline Year: (2017)
Ref_id:b7 Title: Rank consistent ordinal regression for neural networks with application to age estimation Year: (2020)
Ref_id:b8 Title: Audio large language models can be descriptive speech quality evaluators Year: (2025)
Ref_id:b9 Title: Analysis of a simplified normalized covariance measure based on binary weighting functions for predicting the intelligibility of noise-suppressed speech Year: (2010)
Ref_id:b10 Title: WavLM: Large-scale self-supervised pre-training for full stack speech processing Year: (2022)
Ref_id:b11 Title: Order-free RNN with visual attention for multi-label classification Year: (2018)
Ref_id:b12 Title: InQSS: a speech intelligibility and quality assessment model using a multi-task learning network Year: (2022)
Ref_id:b13 Title: Visqol v3: An open source production ready objective speech and audio metric Year: (2020)
Ref_id:b14 Title: Qwen2-audio technical report Year: (2024)
Ref_id:b15 Title: Multi-CMGAN+/+: Leveraging multiobjective speech quality metric prediction for speech enhancement Year: (2024)
Ref_id:b16 Title: A review on subjective and objective evaluation of synthetic speech Year: (2024)
Ref_id:b17 Title: PAM: Prompting audio-language models for audio quality assessment Year: (2024)
Ref_id:b18 Title: Mellow: a small audio language model for reasoning Year: (2025)
Ref_id:b19 Title: Plcmos -a data-driven non-intrusive metric for the evaluation of packet loss concealment algorithms Year: (2023)
Ref_id:b20 Title: ICASSP 2023 deep noise suppression challenge Year: (2024)
Ref_id:b21 Title: Improving instrumental quality prediction performance for the Blizzard Challenge Year: (2008)
Ref_id:b22 Title: A non-intrusive quality and intelligibility measure of reverberant and dereverberated speech Year: (2010)
Ref_id:b23 Title: Llama-VITS: Enhancing tts synthesis with semantic awareness Year: (2024)
Ref_id:b24 Title: BSS_EVAL toolbox user guide-revision Year: (2005)
Ref_id:b25 Title: Freesound datasets: A platform for the creation of open audio datasets Year: (2017)
Ref_id:b26 Title: Audio set: An ontology and human-labeled dataset for audio events Year: (2017)
Ref_id:b27 Title: Recurrent Bayesian classifier chains for exact multi-label classification Year: (2021)
Ref_id:b28 Title: GAMA: A large audio-language model with advanced audio understanding and complex reasoning abilities Year: (2024)
Ref_id:b29 Title: Audio Flamingo 2: An audio-language model with long-audio understanding and expert reasoning abilities Year: (2025)
Ref_id:b30 Title: Odyssey 2024-speech emotion recognition challenge: Dataset, baseline framework, and results Year: (2024)
Ref_id:b31 Title: Ordinal regression methods: survey and experimental study Year: (2015)
Ref_id:b32 Title: An effective quality evaluation protocol for speech enhancement algorithms Year: (1998)
Ref_id:b33 Title: ESPnet2-TTS: Extending the edge of TTS research Year: (2021)
Ref_id:b34 Title: Rotary position embedding for vision transformer Year: (2024)
Ref_id:b35 Title: Evaluation of objective quality measures for speech enhancement Year: (2007)
Ref_id:b36 Title: Subjective comparison and evaluation of speech enhancement algorithms Year: (2007)
Ref_id:b37 Title: Dynamic-SuPERB: Towards a dynamic, collaborative, and comprehensive instruction-tuning benchmark for speech Year: (2024)
Ref_id:b38 Title: Dynamic-SUPERB phase-2: A collaboratively expanding benchmark for measuring the capabilities of spoken language models with 180 tasks Year: (2025)
Ref_id:b39 Title: The VoiceMOS challenge 2022 Year: (2022)
Ref_id:b40 Title: MOS-Bench: Benchmarking generalization abilities of subjective speech quality assessment models Year: (2024)
Ref_id:b41 Title: The VoiceMOS challenge 2024: Beyond speech quality prediction Year: (2024)
Ref_id:b42 Title:  Year: (2011)
Ref_id:b43 Title: Removing the bias in speech quality scores collected in noisy crowdsourcing environments Year: (2021)
Ref_id:b44 Title: AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks Year: (2022)
Ref_id:b45 Title: ESPnet-SPK: full pipeline speaker embedding toolkit with reproducible recipes, self-supervised front-ends, and off-the-shelf models Year: (2024)
Ref_id:b46 Title: Coherence and the speech intelligibility index Year: (2005)
Ref_id:b47 Title: Prediction of perceived phonetic distance from critical-band spectra: A first step Year: (1982)
Ref_id:b48 Title: The AMI meeting corpus Year: (2005)
Ref_id:b49 Title: Can large audio-language models truly hear? tackling hallucinations with multi-task assessment and stepwise audio reasoning Year: (2025)
Ref_id:b50 Title: Mel-cepstral distance measure for objective speech quality assessment Year: (1993)
Ref_id:b51 Title: Torchaudio-Squim: Reference-less speech quality and intelligibility measures in torchaudio Year: (2023)
Ref_id:b52 Title: PL-TTS: A generalizable prompt-based diffusion tts augmented by large language model Year: (2024)
Ref_id:b53 Title: MOSNet: Deep learning-based objective assessment for voice conversion Year: (2019)
Ref_id:b54 Title: Speech quality assessment Year: (2011)
Ref_id:b55 Title: emotion2vec: Self-supervised pre-training for speech emotion representation Year: (2024-08)
Ref_id:b56 Title: Audio-CoT: Exploring chain-of-thought reasoning in large audio language model Year: (2025)
Ref_id:b57 Title: NORESQA: A framework for speech quality assessment using non-matching references Year: (2021)
Ref_id:b58 Title: NISQA: A deep cnn-self-attention model for multidimensional speech quality prediction with crowdsourced datasets Year: (2021)
Ref_id:b59 Title: Towards speech quality assessment using a crowdsourcing approach: evaluation of standardized methods Year: (2020)
Ref_id:b60 Title: The design for the wall street journal-based CSR corpus Year: (1992)
Ref_id:b61 Title: Reproducing whisper-style training using an open-source toolkit and publicly available data Year: (2023)
Ref_id:b62 Title: Evaluation of a speech enhancement system Year: (1977)
Ref_id:b63 Title: Robust speech recognition via large-scale weak supervision Year: (2023)
Ref_id:b64 Title: NOMAD: Unsupervised learning of perceptual embeddings for speech enhancement and non-matching reference audio quality assessment Year: (2024)
Ref_id:b65 Title: SCOREQ: Speech quality assessment with contrastive regression Year: (2024)
Ref_id:b66 Title: ConferencingSpeech Challenge: Towards far-field multi-channel speech enhancement for video conferencing Year: (2021)
Ref_id:b67 Title: Classifier chains for multi-label classification Year: (2011)
Ref_id:b68 Title: Classifier chains: a review and perspectives Year: (2021)
Ref_id:b69 Title: DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors Year: (2021)
Ref_id:b70 Title: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors Year: (2022)
Ref_id:b71 Title: Perceptual evaluation of speech quality (PESQ)-a new method for speech quality assessment of telephone networks and codecs Year: (2001)
Ref_id:b72 Title: UTMOS: Utokyo-sarulab system for voicemos challenge 2022 Year: (2022)
Ref_id:b73 Title: Speech-BERTScore: Reference-aware automatic evaluation of speech generation leveraging nlp evaluation metrics Year: (2024)
Ref_id:b74 Title: ESPnet-Codec: Comprehensive training and evaluation of neural codecs for audio, music, and speech Year: (2024)
Ref_id:b75 Title: VERSA: A versatile evaluation toolkit for speech, audio, and music Year: (2025-04)
Ref_id:b76 Title: Versatile speech assessment with a unified network, 2025b Year: ()
Ref_id:b77 Title: PromptTTS++: Controlling speaker identity in prompt-based text-to-speech using natural language descriptions Year: (2024)
Ref_id:b78 Title: X-vectors: Robust DNN embeddings for speaker recognition Year: (2018)
Ref_id:b79 Title: An algorithm for intelligibility prediction of time-frequency weighted noisy speech Year: (2011)
Ref_id:b80 Title: SingMOS: An extensive open-source singing voice dataset for MOS prediction Year: (2024)
Ref_id:b81 Title: The diverse environments multi-channel acoustic noise database (demand): A database of multichannel environmental noise recordings Year: (2013)
Ref_id:b82 Title: Meta Audiobox Aesthetics: Unified automatic quality assessment for speech, music, and sound Year: (2025)
Ref_id:b83 Title: Objective measures of perceptual audio quality reviewed: An evaluation of their application domain dependence Year: (2021)
Ref_id:b84 Title: A study of complexity and quality of speech waveform coders Year: (1978)
Ref_id:b85 Title: The voice bank corpus: Design, collection and data analysis of a large regional accent speech database Year: (2013)
Ref_id:b86 Title: QualiSpeech: A speech quality assessment dataset with natural language reasoning and descriptions Year: (2025)
Ref_id:b87 Title: What are they doing? joint audiospeech co-reasoning Year: (2025)
Ref_id:b88 Title: SARI: Structured audio reasoning via curriculum-guided reinforcement learning Year: (2025)
Ref_id:b89 Title: SUPERB: Speech Processing Universal PERformance Benchmark Year: (2021)
Ref_id:b90 Title: WHAM!: Extending speech separation to noisy environments Year: (2019)
Ref_id:b91 Title: EMO-SUPERB: An in-depth look at speech emotion recognition Year: (2024)
Ref_id:b92 Title: Audio-reasoner: Improving reasoning capability in large audio language models Year: (2025)
Ref_id:b93 Title: An experimental study on speech enhancement based on deep neural networks Year: (2013)
Ref_id:b94 Title: InstructTTS: Modelling expressive tts in discrete latent space with natural language style prompt Year: (2024)
Ref_id:b95 Title: ConferencingSpeech 2022 challenge: Non-intrusive objective speech quality assessment (nisqa) challenge for online conferencing applications Year: (2022)
Ref_id:b96 Title: A hierarchical predictor of synthetic speech naturalness using neural networks Year: (2016)
Ref_id:b97 Title: LibriTTS: A corpus derived from LibriSpeech for text-to-speech Year: (2019)
