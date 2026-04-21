Title: Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency
Abstract: Time series forecasting plays a pivotal role in critical domains such as energy management and financial markets. Although deep learning-based approaches (e.g., MLP, RNN, Transformer) have achieved remarkable progress, the prevailing "longsequence information gain hypothesis" exhibits inherent limitations. Through systematic experimentation, this study reveals a counterintuitive phenomenon: appropriately truncating historical data can paradoxically enhance prediction accuracy, indicating that existing models learn substantial redundant features (e.g., noise or irrelevant fluctuations) during training, thereby compromising effective signal extraction. Building upon information bottleneck theory, we propose an innovative solution termed Adaptive Masking Loss with Representation Consistency (AMRC), which features two core components: 1) Dynamic masking loss, which adaptively identified highly discriminative temporal segments to guide gradient descent during model training; 2) Representation consistency constraint, which stabilized the mapping relationships among inputs, labels, and predictions. Experimental results demonstrate that AMRC effectively suppresses redundant feature learning while significantly improving model performance. This work not only challenges conventional assumptions in temporal modeling but also provides novel theoretical insights and methodological breakthroughs for developing efficient and robust forecasting models. We have made our code available at https://github.com/MazelTovy/AMRC.

Section: Introduction
Time series forecasting, as a pivotal technology in critical domains such as energy management and financial markets, directly influences decision-making quality and economic efficiency [11,13,19,20,23]. Recent breakthroughs in deep learning have driven revolutionary advancements in time series prediction. Contemporary frameworks including Multilayer Perceptron (MLP)-based architectures [4,7,18,29,30,33], Recurrent Neural Networks (RNNs) with their variants [9,14,22], and attention mechanism-based models exemplified by the Transformer [2,6,17,21,36,37,39], have achieved remarkable breakthroughs in modeling complex temporal patterns through the construction of elaborate hierarchical temporal dependencies.
Current mainstream forecasting models predominantly adhere to the "long-sequence information gain hypothesis," which posits that extending historical data length enhances the availability of temporal dependencies [16,34]. However, through systematic experimental analysis, this study challenges this conventional assumption. As shown in Table 1, we observed a counterintuitive phenomenon across multiple benchmark datasets and diverse model architectures: appropriately truncating early segments of input sequences can significantly improve prediction accuracy. This finding reveals a critical issue in modern predictive models: during training, models inadvertently capture a substantial number of redundant features. These features not only fail to enhance performance but also interfere with the learning process, thereby limiting the models' potential to achieve optimal results.
Through systematic analysis, we have identified two typical manifestations of redundant features and their underlying mechanisms. First, input truncation optimization experiments (as shown in Figure 2b and Table 1) demonstrate that selectively masking partial historical data can significantly improve model prediction performance. This phenomenon reveals the current model's inefficient utilization of long historical windows. Second, representation similarity analysis (as illustrated in Figure 2a) shows that both the model's prediction results and intermediate embeddings exhibit an abnormally concentrated distribution, which significantly deviates from the natural dispersion characteristics of the input and label. Collectively, these observations indicate that existing models exhibit low efficiency when processing long historical windows, often encoding substantial noise or irrelevant variables rather than truly predictive signals.
Building upon information bottleneck theory [10,24,26,27], this study proposes an innovative method called Adaptive Masking Loss with Representation Consistency (AMRC). The core methodology comprises: 1) An adaptive masking mechanism that dynamically identifies key segments with high discriminative power in sequential data and leverages these informative segments to guide the gradient optimization process (as illustrated in Figure 3) ; 2) A representation consistency constraint that establishes stable mapping relationships among the input feature space, label space, and predicted outputs, thereby effectively enhancing the model's generalization capability. Experimental results (as shown in Table 2) demonstrate that the AMRC method significantly reduces the complexity of the training solution space by suppressing the model's reliance on redundant features, fully exploits the performance potential of the model architecture, and consequently improves prediction accuracy.
The primary contributions of this study include:
• Theoretical Insight: Through rigorous experimental validation, We demonstrate that existing time series forecasting models are prone to learning redundant features, which in turn constrain their performance. Building on the theory of information bottlenecks, we construct a novel theoretical framework for time series modeling and propose an innovative optimization pathway, offering a new theoretical perspective for advancing the field of time series forecasting.
• Methodological Innovation: We propose an optimization framework Adaptive Masking Loss with
Representation Consistency. By dynamically selecting discriminative temporal segments to guide gradient descent (as illustrated in Figure 1) while enforcing input-label-prediction consistency, our method effectively suppresses redundant feature learning. Extensive experiments demonstrate consistent performance gains across diverse benchmarks and architectures.
Our work advances the understanding of temporal pattern learning mechanisms while offering a practical pathway to enhance the efficiency and reliability of time series forecasting systems.
this section cite: ['b10', 'b12', 'b18', 'b19', 'b22', 'b3', 'b6', 'b17', 'b28', 'b29', 'b32', 'b8', 'b13', 'b21', 'b1', 'b5', 'b16', 'b20', 'b35', 'b36', 'b38', 'b15', 'b33', 'b9', 'b23', 'b25', 'b26']

Section: Related Work
The Information Bottleneck (IB) method was first introduced by Tishby et al. [26] as an informationtheoretic framework that aims to compress input signals while preserving as much relevant information as possible about the target output. In the field of machine learning, IB theory has been widely adopted as a regularization technique. For instance, Alemi et al. [1] proposed the Variational Information Bottleneck (VIB), which leverages variational inference to construct a tractable lower bound on the IB objective. Building upon this, Tishby and Zaslavsky [27]further explored the applicability of information-theoretic objectives to deep neural networks. Research on IB has also extended into the domain of clustering. Slonim et al. [24] developed a distributional clustering algorithm based on mutual information maximization and demonstrated its effectiveness on the 20 Newsgroups dataset, achieving substantial compression with minimal loss of relevant information. More recently, Hu et al. [10] conducted a comprehensive survey of the IB literature, reviewing over two decades of theoretical developments, methodological advances, and practical applications.
In the context of deep learning, time series forecasting methods can be broadly categorized into MLP, RNN, and Transformer-based approaches. Among MLP-based models, DLinear [33] and TSMixer [7] are representative examples, featuring relatively simple architectures while achieving strong performance across multiple datasets. RNN-based methods, such as Segrnn [6] and LSTMlong [22], focus on structural modifications to address challenges related to parallel prediction and long-sequence modeling. Transformer-based models include Informer [37], Autoformer [21], and iTransformer [15].
Informer introduces a sparse attention mechanism to improve the scalability of traditional attention for time series modeling; Autoformer incorporates frequency-domain information to enhance attention; and iTransformer further extends attention across channels by embedding multivariate sequences for variable-aware representation.
Another key research area concerns noise robustness and representation learning. Early work, such as Informer [37], used sparse attention for information distillation in long sequences, while TS2Vec [31] adopted contrastive learning to regularize temporal representations. More recently, dedicated frameworks have been proposed. For instance, TS-CoT [35] employs a dual-encoder architecture and a cross-view prototype alignment mechanism to achieve global semantic consistency. Similarly, DECL [38] guides contrastive learning to acquire denoising capabilities by constructing positive samples from denoised data and leveraging an adaptive denoiser.
this section cite: ['b25', 'b0', 'b26', 'b23', 'b9', 'b32', 'b6', 'b5', 'b21', 'b36', 'b20', 'b14', 'b36', 'b30', 'b34', 'b37']

Section: Analysis of Redundant Feature Learning
Given a multivariate time series X ∈ R T ×D , where T is the number of timesteps and D is the number of variables, the objective of time series forecasting is to learn a mapping function f θ that transforms historical observations X t-L:t ∈ R L×D (where L denotes the input length ) into future values X t+1:t+H ∈ R H×D (where H represents the forecasting horizon).
Conventional time series forecasting models follow the long-sequence information gain hypothesis [3,5,32,37], which holds that increasing the input length L improves forecasting accuracy. However, our experiments (Table 1) on multiple standard benchmarks reveal a counterintuitive result: truncating the input-such as masking the first k timesteps-often improves forecasting performance, which is measured by Mean Squared Error (MSE). We found that models tend to learn redundant features, which degrade model performance even after convergence. This finding is supported by two key observations:
Figure 1: Illustration of the effect of AMRC method. Without regularization, the model tends to overfit redundant input features, leading to suboptimal convergence. By suppressing redundant input features, AMRC restructures the optimization landscape, promoting more efficient representation learning and facilitating better convergence.
this section cite: ['b2', 'b4', 'b31', 'b36']

Section: Input Truncation Optimization
Based on the baseline model configuration (input length L = 48, forecasting horizon H = 48), we design an input truncation comparative experiment by applying a masking operator M k (•) to the input sequence. When we have an input sequence of length L at time step t, denoted as X (L) t , the masking operator M k (•) is mathematically defined as:
M k (X (L) t ) = 0 if i ≤ k X (L) t otherwise (1
)
Here, k ∈ {1, . . . , L} denotes the masking step size.
To probe redundant features, we employ an Optimal Masking strategy: Given an input sequence of length L, we generate L masked variants {M k (X (L) t )} L k=1 (zero-padded to preserve dimensionality). For instance, k = 5 yields L ′ = 43 (first 5 positions zeroed). The optimal mask length k * is selected as the configuration minimizing MSE, thereby defining the theoretical upper bound for redundancy elimination: As demonstrated in Table 1, the experimental results confirm that masked models consistently achieve lower MSE, with more than 50% of samples exhibiting improved predictive performance (Ratio > 50%). Notably, the phenomenon of redundancy learning shows strong architecture-agnostic characteristics. On the Weather dataset, both iTransformer (a Transformer-based model) and TSMixer (an MLP-based model) demonstrate similar relative improvements: iTransformer achieves an MSE reduction from 0.209 to 0.170 (-18.7%), while TSMixer improves from 0.222 to 0.195 (-12.2%).
k * = arg min k∈{1,2,...,L} E f θ M k (X (L) t ) -Y (H) t 2 (2)
These results indicate that the effectiveness of our masking strategy is not dependent on specific model architectures.
this section cite: []

Section: Representation Similarity Paradox
To further investigate the redundant feature learning phenomenon, we apply t-SNE to project the SOFTS model's high-dimensional representations of the input, embedding, prediction, and label onto a 2D plane (Figure . 2a), after normalizing all features to the [0, 1] range.
As illustrated in Figure . 2a, Normalized input (Z in ∈ R L ) and output (Z out ∈ R H ) embeddings show a clear contrast: inputs remain dispersed, while embeddings and preds cluster tightly despite large differences in their corresponding labels. This suggests that the model encodes redundant, task-irrelevant features that misrepresent semantic relationships and distort the input-output mapping.
this section cite: []

Section: Information Bottleneck Constraints on Redundancy
In time-series forcasting models, the input sequence X is typically encoded into a latent representation Z, from which a decoder then predicts the target sequence Y . The optimization objective is to learn an optimal representation Z that maximally preserves information relevant to Y while discarding irrelevant details from X. According to the Information Bottleneck (IB) Theory [24], this process can be viewed as a bottleneck that compresses input information. The informational relationships among X, Y , and Z, which are governed by the model's learnable parameter θ, can be quantified using mutual information. The objective can be thus formally expressed as maximizing the mutual information between the representation Z and the target Y :
I(Z, Y ; θ) = dx dy p(z, y | θ) log p(z, y | θ) p(z | θ)p(y | θ) .(3)
Due to inherent limitations in the data and model capacity, the amount of information that can be extracted and transmitted during training is bounded. Consequently, the representation capacity is subject to an upper information constraint I c . Based on this, the objective of the time series prediction model can be equivalently formulated as the following constrained optimization problem: max θ I(Z, Y ; θ) s.t. I(X, Z; θ) ≤ I c .
This constrained optimization problem can be transformed into an unconstrained form using the method of Lagrange multipliers, leading to the maximization of the following objective [1]: R IB (θ) = I(Z; Y ; θ) -βI(Z; X ; θ).
There are two implementation paths under this objective: one is to maximize the mutual information I(Z; Y ) between Z and Y ; the other is to minimize the mutual information I(Z; X) between Z and X. Most current sequential prediction models focus on improving I(Z; Y ) through iterative training, but have not explicitly optimized performance by penalizing redundant features via minimizing I(Z; X). Therefore, we propose an adaptive loss function that aims to minimize the mutual information between X and Z, offering a novel optimization path for improving the performance of sequential prediction models.
this section cite: ['b23', 'b0']

Section: Proposed Method

this section cite: []

Section: Adaptive Masking Loss (AML)
As discussed in Section 3.1, applying ideal masking to input data reduces the information I(X) while improving prediction accuracy. This indicates that the representation Z k * , generated by encoder p θ from masked features X t,k * , contains less redundancy and better approximates the minimal sufficient statistics (i.e., with smaller I(X, Z k * ; θ)). Based on this insight, we propose the Adaptive Masking Loss (AML) to explicitly reduce mutual information I(X, Z; θ) by guiding the encoder's output representation Z toward Z k * , thereby suppressing redundant feature learning and unleashing model potential. The overall framework of AML is illustrated in Figure 3.
this section cite: []

Section: Implementation
The exhaustive search for optimal mask k * by enumerating all possible mask lengths k ∈ {1, ..., L} results in prohibitive O(L) time complexity for long sequences. We therefore adopt an efficient stochastic approximation strategy:
1. Random Mask Generation: Independently sample m mask indices {k s } m s=1 from uniform distribution d(k) = Uniform{1, ..., L}, each generating a masked variant:
X (L) t,s = M ks (X (L) t )
(6) 2. Loss Evaluation: Compute prediction losses for both masked and original data:
ℓ s = L(f θ ( X (L) t,s ), Y (H) t )(7)
ℓ = L(f θ (X (L) t ), Y (H) t )(8)
3. Optimal Representation Selection: If ∃ℓ s < ℓ, the corresponding representation
Z s = p θ ( X (L) t,s ) satisfies I(X (L) t , Z s ) ≤ I(X (L) t , Z), where Z = p θ (X (L)
t ) is the original representation. It signifies that a masked input variant can achieve better predictive performance than the original input. This provides a clear indication that the removed information was redundant rather than essential. The optimal mask variant is selected by:
s * = arg max s (ℓ -ℓ s )(9)
this section cite: []

Section: Loss Formulation
To promote compact and informative representations, AML minimizes the distance between the original representation Z and the optimal masked variant Z s * :
L AML = β • 1 D 1 × D 2 ∥Z -Z s * ∥ 2 (10
)
The adaptive weight β = max(0, (ℓ-ℓ s * )/ℓ) ensures that this regularization term is only active when a better-performing masked representation is found. Such a setup dynamically scales the optimization intensity, guaranteeing a more substantial influence from mask variants with greater loss reduction.
this section cite: []

Section: Embedding Similarity Penalty (ESP)
Time series forecasting models often encounter two issues: semantic inconsistency, where semantically similar inputs lead to substantially different predictions, and representation collapse, where dissimilar inputs result in nearly identical outputs. While consistency regularization methods like Temporal Ensembling [12] and Mean Teacher [25] address stability for individual samples under augmentation, they do not explicitly consider the relational structure between different samples. We therefore introduce the Embedding Similarity Penalty (ESP), a strategy that directly addresses this by comparing the geometry of the embedding space with that of the output space for pairs of samples within a mini-batch.
Pairwise distances. For a batch B = {(X i , Y i )} n i=1 we denote by Z i = f enc (X i ) ∈ R L×D the encoder output and keep the ground-truth Y i ∈ R P ×D . The (normalised) squared Frobenius distances are
∆ E ij = 1 L × D ∥Z i -Z j ∥ 2 F , ∆ O ij = 1 P × D ∥Y i -Y j ∥ 2 F , 1 ≤ i, j ≤ n.(11)
Consistency penalty. Ideally ∆ E ij and ∆ O ij should match: semantically similar inputs (∆ E ij ≈ 0) ought to produce similar outputs (∆ O ij ≈ 0), and vice versa. Deviation is quantified element-wise through
P ij = ReLU ∆ E ij -∆ O ij + ReLU ∆ O ij -∆ E ij = | ∆ E ij -∆ O ij | + ,(12)
where ReLU(x) = max(0, x) and | • | + denotes the non-negative part. The Embedding-Similarity Penalty then reads
L ESP = 1 n 2 n i=1 n j=1 P ij .(13)
Equation ( 13) back-propagates smooth, unbiased gradients that jointly reshape the encoder and the predictor so that input and output manifolds remain geometrically aligned. The detailed implementation of the Embedding Similarity Penalty (ESP) is provided as pseudocode in Appendix C Algorithm 1.
this section cite: ['b11', 'b24']

Section: Overall Training Objective
Section 4.1 introduced the Adaptive Masking Loss L AML that discourages the learning of redundant temporal prefixes, while Section 4.2 proposed the Embedding-Similarity Penalty L ESP to enforce semantic-behavioural consistency. Combined with the standard prediction loss L pred (e.g., MSE between the forecast Ŷ and the target Y ), our final objective is
L total = L pred + λ AML L AML + λ ESP L ESP ,(14)
where λ AML , λ ESP > 0 control the strength of each auxiliary term. Minimizing ( 14) jointly (i) identifies the informative prefix for every sequence, (ii) preserves the intrinsic topology of the data, and (iii) improves predictive accuracy and interpretability without adding inference-time overhead.
this section cite: []

Section: Experiment

this section cite: []

Section: Experiment Setup
Datasets. We evaluate our proposed method using seven widely recognized benchmark datasets for multivariate time series forecasting: ETTh1, ETTh2, ETTm1, ETTm2, Solar-Energy, Electricity, and Weather. These datasets encompass a variety of application scenarios with different temporal resolutions, seasonality patterns, and dynamic structures. Detailed descriptions of each dataset, including their specific characteristics and collection periods, are provided in the Appendix E.
Task formulation. In our experimental setup, the forecasting task is formulated as a sequence-tosequence regression problem, applicable to multivariate time series. Each model is trained to predict a future sequence
Y (H) t ∈ R H×D from a fixed-length historical input sequence X (48) t ∈ R 48×D ,
where H denotes the prediction length and D is the number of variables. We adopt multiple prediction horizons H ∈ {48, 72, 96, 120, 144, 168, 192}.
Baselines. Our method is compared against five diverse baseline models: SOFTS [8], iTransformer [15], PatchTST [17], TSMixer [7], and TimeMixer [28]. These baselines are implemented using their official codebases and recommended hyperparameters to ensure a fair comparison under consistent experimental conditions.
Implementation details. All models are implemented in PyTorch and trained on a single NVIDIA A100 80GB GPU. To ensure a fair comparison and allow both baseline models and those augmented with our proposed modules to fully exploit their capacity, we train each model for up to 100 epochs using the Adam optimizer with an initial learning rate of 1 × 10 -4 , a cosine annealing scheduler, and a batch size of 32. Early stopping is applied based on validation loss with a patience of 20 epochs.
The best-performing checkpoint on the validation set is selected for final evaluation on the test set.
Hyperparameter selection. For the AML, the input sequence prefix length is configured as L = 48, with the mask sampling cardinality parameterized as m = 12. We fix both λ AML and λ ESP to 1 for all experiments. These settings follow standard benchmark configurations commonly used in time series forecasting.
this section cite: ['b7', 'b14', 'b16', 'b6', 'b27']

Section: Forecasting Results
We present the forecasting performance of our method-Adaptive Masking Loss with Representation Consistency (AMRC)-in comparison with five representative baseline models across seven widely used time series benchmark datasets.
Table 2 reports the Mean Squared Error (MSE) and Mean Absolute Error (MAE) for each model, both with and without the incorporation of AMRC.
Table 2: Performance Comparison of Time Series Forecasting Models With and Without AMRC. In the experimental results, we highlighted in bold the parts where the AMRC model improved by more than 0.005 in MSE and MAE metrics compared to the baseline model. The detailed hyperparameter configurations for each model can be found in Appendix B. Full results are listed in Appendix D.1 Table 5. Furthermore, a detailed statistical analysis presenting results as mean ± standard deviation over 10 runs, along with significance tests, is provided in Appendix D.1 Table 6. To further validate the robustness of AMRC, we conducted additional experiments on the ExchangeRate dataset and the challenging, low-data Illness dataset, as detailed in Appendix 7. Model ETTh1 ETTh2 ETTm1 ETTm2 Solar-Energy Electricity Weather Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE SOFTS Original 0.408 0.414 0.326 0.359 0.484 0.434 0.210 0.285 0.293 0.314 0.169 0.255 0.205 0.234 AMRC 0.389 0.393 0.311 0.362 0.475 0.423 0.198 0.265 0.290 0.309 0.162 0.244 0.196 0.220 iTransformer Original 0.413 0.415 0.329 0.362 0.517 0.448 0.213 0.290 0.395 0.352 0.176 0.260 0.209 0.237 AMRC 0.402 0.399 0.324 0.356 0.502 0.447 0.211 0.280 0.392 0.342 0.163 0.239 0.201 0.221 TimeMixer Original 0.393 0.408 0.318 0.355 0.466 0.429 0.209 0.285 0.288 0.317 0.194 0.279 0.197 0.237 AMRC 0.388 0.401 0.316 0.339 0.447 0.405 0.204 0.269 0.284 0.317 0.188 0.277 0.186 0.228 PatchTST Original 0.424 0.424 0.327 0.358 0.461 0.422 0.211 0.287 0.374 0.382 0.211 0.283 0.215 0.280 AMRC 0.411 0.415 0.319 0.356 0.456 0.413 0.196 0.271 0.361 0.376 0.207 0.285 0.210 0.264 TSMixer Original 0.402 0.412 0.324 0.357 0.440 0.413 0.201 0.279 0.288 0.314 0.172 0.258 0.222 0.288 AMRC 0.386 0.397 0.319 0.340 0.432 0.412 0.196 0.257 0.280 0.313 0.169 0.247 0.212 0.281
Consistent Performance Gains. Across all models and datasets, our method consistently yields performance improvements. For example, the MSE of the SOFTS model decreases from 0.408 to 0.389 on the ETTh1 dataset. Similar trends are observed in iTransformer, where the MSE on Electricity drops from 0.176 to 0.163. The enhancements demonstrate that AMRC effectively mitigates redundant or noisy temporal segments, thereby improving prediction stability and accuracy.
Architecture-Agnostic Effectiveness. AMRC delivers significant performance gains not only on Transformer-based architectures such as iTransformer and PatchTST, but also on MLP-based models including TimeMixer, SOFTS, and TSMixer. For instance, on the ETTm2 dataset, the MSE of PatchTST model decreases from 0.211 to 0.196 (a reduction of approximately 7.11%), while the MSE of SOFTS model drops from 0.210 to 0.198 (approximately 5.71% reduction). These results demonstrate the strong architecture-agnostic generalization ability of AMRC, highlighting its broad applicability across a wide range of time series forecasting models.
Generalization on Low-Channel Datasets. On datasets with fewer input channels (ETTh1, ETTh2, ETTm1, ETTm2), AMRC effectively enhances model performance. For instance, on ETTm1, the MSE of iTransformer decreases from 0.517 to 0.502, and that of TSMixer drops from 0.440 to 0.432. These results demonstrate AMRC's ability to mitigate overfitting and improve prediction accuracy in low-dimensional time series forecasting tasks.
Robustness on High-Channel Datasets. For high-dimensional datasets such as Weather (21 channels) and Solar-Energy (137 channels) see in Appendix E, AMRC consistently improves robustness by reducing the impact of signal noise and inter-channel redundancy. On the Weather dataset, TimeMixer's MSE decreases from 0.197 to 0.186 and MAE from 0.237 to 0.228, while iTransformer sees an MAE drop from 0.237 to 0.221. On Solar-Energy, PatchTST's MSE drops from 0.374 to 0.361, and SOFTS sees a slight MAE reduction from 0.314 to 0.309. These enhancements highlight AMRC's effectiveness in managing complexity in multivariate time series with high channel counts.
this section cite: []

Section: Generalizable Training Framework.
The consistent performance improvements observed across all models validate the strong scalability and integrability of AMRC. As a constraint-based optimization strategy, AMRC does not rely on any specific model architecture, making it highly generalizable. It serves as a versatile training framework for enhancing both the efficiency and accuracy of time series forecasting models.
this section cite: []

Section: Ablation Study
Setup. We evaluate ablation variants on four diverse datasets: ETTh1 and ETTh2, representing hourly electricity load with varying degrees of seasonality; Solar-Energy, which exhibits weatherdriven variability and periodicity; and Weather, a multivariate meteorological dataset with complex inter-variable dependencies. We adopt a fixed input horizon following standard benchmarks.We also analyzed the sensitivity to the number of sampled masks, m, used in AML. While a larger m allows for a more extensive search, it incurs greater computational cost. Our analysis, detailed in Appendix Table 8, reveals diminishing returns as mincreases. Consequently, we set m = 12 for all experiments to effectively balance performance and computational efficiency.
Evaluation protocol. For each dataset, we apply the ablation study to five baseline models SOFTS, iTransformer, TimeMixer, PatchTST, and TSMixer under four configurations:1) baseline + AML, 2) baseline + ESP, and 3) baseline + both AML and ESP. This design allows us to assess the standalone effectiveness of each module as well as their combined synergy.
Findings. We evaluate the individual and joint effects of the AML and ESP components using five representative forecasting architectures across four datasets. As shown in Table 3, both components contribute measurable performance gains in isolation, while their combination AMRC consistently leads to the best forecasting accuracy in terms of MSE and MAE. AML provides stronger improvements across most settings, supporting its role in suppressing redundant prefixes during training. ESP, while often delivering smaller standalone gains, remains beneficial by promoting geometric alignment between embedding and output spaces. Together, these findings demonstrate that each component addresses a distinct source of generalization error.
Component impact across architectures. The benefits of AML and ESP are consistently observed across all backbone models, regardless of architectural differences. For instance, models with strong expressiveness, such as iTransformer and TimeMixer, benefit significantly from AML, achieving notable MSE reductions on datasets like Weather and ETTh2. Even architectures without attention mechanisms, such as SOFTS and TSMixer, exhibit consistent gains, highlighting the broad applicability of adaptive prefix masking. In contrast, the improvements from ESP are often more dataset-dependent, being particularly effective on high-dimensional multivariate inputs where representation alignment plays a critical role. For example, ESP yields non-trivial reductions in MAE on Weather, where multiple variables evolve under shared dynamics. Notably, we observe relatively smaller improvements on the Solar-Energy dataset for transformer-based models such as PatchTST and iTransformer, which may be attributed to their reliance on longer input sequences for stable attention computation.
this section cite: []

Section: Complementarity and synergy.
The AMRC configuration, which jointly applies AML and ESP, consistently outperforms its ablated variants across all benchmarks. The performance improvement from combining both components generally exceeds the stronger of the two individual effects, indicating synergistic interaction. This complementarity can be attributed to their distinct operational scopes: AML operates on the input level by learning to suppress non-informative temporal segments, while ESP regularizes the latent space to align representations across semantically related inputs. As a result, AMRC improves both the quality of features learned from the data and the consistency of their usage in prediction. The robust gains observed across datasets and architectures suggest that jointly addressing input redundancy and representation inconsistency is critical for improving generalization in time series forecasting.
Table 4: AMRC Effectiveness with Prefix Masking at a Fixed Input Length (L = 48). Ratio is the percentage of training samples with reduced MSE under prefix masking. Ratio* is the same metric after training with AMRC. Average results across all lengths are in Appendix D.1 Table 10. Model ETTh1 ETTh2 Solar-Energy Weather Metric Ratio Ratio* Ratio Ratio* Ratio Ratio* Ratio Ratio* SOFTS 64% 57.33% 28.72% 20.28% 41.58% 33.49% 54.93% 47.12% iTransformer 60.07% 49.95% 32.16% 23.28% 68.43% 63.21% 80.26% 70.29% TimeMixer 58.04% 46.29% 44.52% 34.17% 36.25% 27.90% 66.13% 52.28% PatchTST 65.51% 51.63% 42.46% 26.19% 51.66% 47.64% 42.43% 30.78% TSMixer 59.19% 46.62% 42.13% 27.98% 40.12% 28.36% 70.88% 58.23%
this section cite: []

Section: Effectiveness of AMRC in Reducing Redundant Features
We evaluate the model's robustness to redundant input by computing the proportion of training samples with improved MSE under prefix masking Ratio and compare it to the value after applying AMRC Ratio*. As shown in Table 4, AMRC consistently improves or maintains this ratio, indicating its effectiveness in suppressing the impact of redundant temporal information.
this section cite: []

Section: Conclusion
This study pioneers the investigation into the negative effects of redundant feature learning in time series forecasting and introduces AMRC, a plug-and-play solution that suppresses such learning without requiring architectural modifications. Unlike prior work focused on enhancing predictive features, AMRC improves accuracy by reducing reliance on redundant features while maintaining model flexibility. Its key advantages include: 1) seamless integration with existing models, 2) effective suppression of feature redundancy, and 3) strong generalization performance across benchmark tests. By addressing the long-overlooked issue of redundant learning, this research provides a novel and practical methodology for optimizing forecasting models.
this section cite: []

Section: A Limitations
Despite the demonstrated effectiveness (Table 2) of our approach, AMRC has several limitations related to its underlying assumptions, interpretability, and practical trade-offs, which highlight important directions for future work.
1. Limitations of AML AML's efficacy is bound by two key factors: the temporal characteristics of the data and the interpretability of its masking mechanism.
• The prefix-masking strategy assumes that redundant information often resides in the initial segments of a time series. In scenarios where the most critical predictive information lies exclusively in the later portions of the input sequence, AML's core mechanism becomes ineffective. Masking the prefix will not improve the prediction loss, causing the adaptive coefficient β to remain zero and deactivating the regularization.
• A significant limitation is the "black box" nature of the masking process. While AML is designed to identify and suppress redundancy, it is difficult to determine precisely what kinds of patterns are being masked-whether they represent noise, outliers, or simply outdated information. The adaptive-weight mechanism improves efficiency, but the decision process is not transparent. Clarifying this is a crucial direction for future work to enhance the method's interpretability.
this section cite: []

Section: Dependency on Data Dimensionality and the Role of ESP
• We observe that ESP's improvements are more pronounced on datasets with lower feature dimensionality (e.g., the ETTh family). On higher-dimensional datasets like Weather (21 channels) and Solar-Energy (137 channels), its standalone gains are comparatively smaller.
• This occurs because ESP aligns the geometric structure between the embedding and output spaces. As feature dimensionality increases, the optimization directions for this alignment grow exponentially, introducing greater uncertainty during training and potentially yielding diminished returns.
• This limitation is effectively mitigated within the combined AMRC framework. Highdimensional datasets often contain significant feature redundancy, which is precisely the condition where AML excels. Therefore, the two components are highly complementary: ESP is most effective in lower-dimensional settings, while AML provides the primary benefit in higher-dimensional, redundant settings, ensuring that AMRC remains robust across diverse data types.
this section cite: []

Section: Inherent Design Trade-offs
The search for an optimal mask requires evaluating m variants per batch, increasing the training cost by a factor of approximately m. This makes it less suitable for latency-sensitive applications. The optimal mask is found via stochastic sampling of m candidates, which is an approximation of an exhaustive search. This practical compromise means that some redundancy may remain, though it strikes a balance with computational feasibility.
this section cite: []

Section: B Details of the Baseline Model
All models are reproduced based on their official open-source implementations:
1. SOFTS from https://github.com/Secilia-Cxy/SOFTS.
this section cite: []

Section: 2.
TimeMixer from https://github.com/kwuking/TimeMixer.
this section cite: []

Section: 3.
iTransformer from https://github.com/thuml/iTransformer.
this section cite: []

Section: 4.
PatchTST from https://github.com/yuqinie98/PatchTST.
this section cite: []

Section: 5.
TSMixer from https://github.com/ditschuk/pytorch-tsmixer.
The hyperparameters for each model on different datasets follow the official configurations provided in their corresponding GitHub repositories. For the PatchTST model on the Solar-Energy dataset, since no official configuration was provided, we adopted the hyperparameter settings from iTransformer.
this section cite: []

Section: C Model Detail
C.1 ESP Algorithm 1: Embedding-Similarity Penalty (ESP) for Time Series Forecasting Input: Mini-batch B = {(X i , Y i )} n i=1 , encoder f enc , predictor f pred Output: Penalty loss L ESP 1 1. Forward pass to compute encoder outputs 2 for i ← 1 to n do 3   0.07927 ± 0.002 0.18782 ± 0.005 0.08329 ± 0.004 0.20196 ± 0.001 99 99 120 0.10053 ± 0.001 0.21698 ± 0.002 0.10695 ± 0.001 0.22840 ± 0.001 99 99 144 0.12417 ± 0.002 0.24484 ± 0.002 0.12935 ± 0.002 0.25241 ± 0.001 99 99 168 0.14602 ± 0.002 0.25976 ± 0.001 0.16047 ± 0.003 0.28234 ± 0.003 99 99 192 0.17457 ± 0.007 0.29181 ± 0.006 0.18376 ± 0.007 0.30397 ± 0.007 95 99
Z i ← f enc (X i ) #encoder output ∈ R L×D 4 end 5 2. Compute pairwise Frobenius distances 6 Initialize ∆ E , ∆ O ∈ R n×n 7 for i ← 1 to n do 8 for j ← i to n do 9 ∆ E ij ← 1 L×D ∥Z i -Z j ∥ 2 F #embedding similarity 10 ∆ O ij ← 1 P ×D ∥Y i -Y j ∥ 2 F #output similarity 11 ∆ E ji ← ∆ E ij , 12 ∆ O ji ← ∆ O ij #
this section cite: []

Section: E Dataset description
Here we provide detailed descriptions along with download links for each dataset:
1. ETT (Electricity Transformer Temperature) [37] foot_0 : This collection includes two hourlyresolution datasets (ETTh) and two 15-minute-resolution datasets (ETTm). Each dataset captures seven key operational metrics (including oil and load measurements) from electricity transformers, spanning from July 2016 to July 2018. 2. Electricityfoot_1 : Comprising hourly power consumption records from 321 customers, this dataset covers the period from 2012 to 2014. 3. Weather: Featuring 21 meteorological indicators (such as air temperature and humidity), this dataset provides 10-minute-interval recordings throughout 2020, sourced from weather stations in Germany. 4. Solar-Energy: Documents the solar power generation output of 137 photovoltaic plants in 2006, with measurements taken at 10-minute intervals. • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: ['b36']

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We have included a link to an anonymous demo of our experiments in the abstract.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
this section cite: []

Section: NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
• You should answer [Yes] , [No] , or [NA] .
• [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• Please provide a short (1-2 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
this section cite: []

Section: IMPORTANT, please:
• Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist", • Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The main claims are clearly written in the abstract and introduction.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discussed the limitation of our method in Appendix A.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: All the theories and hypotheses we proposed are supported by experimental and mathematical derivations.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We provide detailed descriptions of the hyperparameters in the paper and appendices, along with an anonymous link to the experimental demo in the abstract.
Guidelines:
• The answer NA means that the paper does not include experiments.
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental setting/details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We provide the experimental setup details in both the main text and appendices.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: The margin of error is reported in the appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We provide sufficient computational resource details for each experiment in both the main text and appendices.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: Our methodology and implementation fully adhere to the ethical code standards set forth by NeurIPS.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] Justification: We have discussed the broader impact of time series forecasting in both abstract and introduction.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [No] Justification: This paper does not have this risk.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes]
Justification: We included it in implementation details and appendix.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
this section cite: []

Section: New assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] Justification: N/A.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: N/A. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: N/A. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [No] Justification: We did not use any large language models (LLMs) in this work. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Deep variational information bottleneck Year: (2016)
Ref_id:b1 Title: Pathformer: Multi-scale transformers with adaptive pathways for time series forecasting Year: (2024)
Ref_id:b2 Title: Transformer-xl: Attentive language models beyond a fixed-length context Year: (2019)
Ref_id:b3 Title: Long-term forecasting with tide: Time-series dense encoder Year: (2023)
Ref_id:b4 Title: Longnet: Scaling transformers to 1,000,000,000 tokens Year: (2023)
Ref_id:b5 Title: Tactis: Transformer-attentional copulas for time series Year: (2022)
Ref_id:b6 Title: Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting Year: (2023)
Ref_id:b7 Title: Softs: Efficient multivariate time series forecasting with series-core fusion Year: (2024)
Ref_id:b8 Title: Recurrent neural networks for time series forecasting: Current status and future directions Year: (2021)
Ref_id:b9 Title: A survey on information bottleneck Year: (2024)
Ref_id:b10 Title: Time series prediction in industry 4.0: a comprehensive review and prospects for future advancements Year: (2023)
Ref_id:b11 Title: Temporal ensembling for semi-supervised learning Year: (2017)
Ref_id:b12 Title: Time-series forecasting with deep learning: a survey Year: (2021)
Ref_id:b13 Title: Segrnn: Segment recurrent neural network for long-term time series forecasting Year: (2023)
Ref_id:b14 Title: itransformer: Inverted transformers are effective for time series forecasting Year: (2024)
Ref_id:b15 Title: Long input sequence network for long time series forecasting Year: (2024)
Ref_id:b16 Title: Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers Year: (2023)
Ref_id:b17 Title: N-beats: Neural basis expansion analysis for interpretable time series forecasting Year: (2019)
Ref_id:b18 Title: Deep learning methods and applications for electrical power systems: A comprehensive review Year: (2020)
Ref_id:b19 Title: Forecasting: theory and practice Year: (2022)
Ref_id:b20 Title: Autotransformer: Automatic transformer architecture design for time series classification Year: (2022)
Ref_id:b21 Title: Demand forecasting in smart grid using long short-term memory Year: (2021)
Ref_id:b22 Title: Exploring progress in multivariate time series forecasting: Comprehensive benchmarking and heterogeneity analysis Year: (2024)
Ref_id:b23 Title: Advances in neural information processing systems Year: (1999)
Ref_id:b24 Title: Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results Year: (2017)
Ref_id:b25 Title: The information bottleneck method Year: (2000)
Ref_id:b26 Title: Deep learning and the information bottleneck principle Year: (2015)
Ref_id:b27 Title: Decomposable multiscale mixing for time series forecasting Year: (2024)
Ref_id:b28 Title: Fits: Modeling time series with 10k parameters Year: (2023)
Ref_id:b29 Title: Frequency-domain mlps are more effective learners in time series forecasting Year: (2023)
Ref_id:b30 Title: Ts2vec: Towards universal representation of time series Year: (2022)
Ref_id:b31 Title: Big bird: Transformers for longer sequences Year: (2020)
Ref_id:b32 Title: Are transformers effective for time series forecasting Year: (2023)
Ref_id:b33 Title: Are transformers effective for time series forecasting Year: (2023)
Ref_id:b34 Title: A co-training approach for noisy time series learning Year: (2023)
Ref_id:b35 Title: Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting Year: (2023)
Ref_id:b36 Title: Informer: Beyond efficient transformer for long sequence time-series forecasting Year: (2021)
Ref_id:b37 Title: Denoisingaware contrastive learning for noisy time series Year: (2024)
Ref_id:b38 Title: Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting Year: (2022)
