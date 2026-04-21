Title: Learning to Factorize Spatio-Temporal Foundation Models
Abstract: Spatio-Temporal (ST) Foundation Models (STFMs) promise cross-dataset generalization, yet joint ST pretraining is computationally costly and struggles with domain-specific spatial correlations. To address this, we propose FactoST, a factorized STFM that decouples universal temporal pretraining from ST adaptation. The first stage trains a space-agnostic backbone via multi-task learning to capture multifrequency, cross-domain temporal patterns at low cost. The second stage attaches an lightweight adapter that rapidly adapts the backbone to specific ST domains via metadata fusion, interaction pruning, domain alignment, and memory replay. Extensive forecasting experiments show that in few-shot settings, FactoST reduces MAE by up to 46.4% versus UniST, uses 46.2% fewer parameters, achieves 68% faster inference than OpenCity, and remains competitive with expert models. This factorized view offers a practical, scalable path toward truly universal STFMs.

Section: Introduction
Spatio-temporal (ST) data capture how signals evolve over time across complex spatial structures-such as traffic speeds on road networks, air-pollution levels from citywide sensors, or electricity loads at substations. Modeling ST data is fundamental to forecasting and decision support across science, engineering, and society, enabling proactive anticipation and intervention [77,13,58,73].
In deep learning practice, Spatio-Temporal Graph Neural Networks (STGNNs) are the de facto workhorse for modeling such data [58,39,47,27,28], as depicted in Figure 1(a). Given the intricate nature of jointly learning spatial and temporal dependencies, early attempts decompose the learning problem into two complementary components: (i) a recurrent [32,6,46], convolutional [68,64,30], or attention-based module [34,72,18] that extracts temporal dependencies from each location's history, and (ii) a Graph Neural Network (GNN) [29] that propagates information along edges to capture spatial correlations among locations [32,68,64,57,41]. This design yields strong inductive bias, parameter efficiency, and state-of-the-art performance on a wide range of benchmarks.
Inspired by the transformative impact of Foundation Models (FMs) in language [44] and vision [5], researchers have recently begun to explore STFMs [36,17,23,14]. The core idea is simple -Pretrain a single model on diverse ST corpora (e.g., climate, traffic, energy) and adapt it to unseen datasets in a zero-shot or few-shot fashion, as shown in Figure 1(b). Such cross-domain pretraining equips STFMs with broad cross-dataset spatio-temporal knowledge and generalization beyond single-dataset scopes, often outperforming task-specific STGNNs when labeled data is scarce [40,69,33,70]. Nevertheless, training an STFM at scale presents two pronounced obstacles. First, spatial correlations differ dramatically across domains. For instance, the adjacency structure of a power grid differs greatly from urban road topology networks, making it difficult for a monolithic model to internalize all possible patterns; neighbouring air-quality stations in Beijing exhibit short-range diffusion dynamics, whereas tele-connection effects dominate climate indices across the Pacific Ocean. Second, existing STFMs [69,33] mostly rely on the paradigm of jointly learning spatial and temporal dependencies for hundreds, thousands, or even millions of locations, which is computationally expensive; memory and time footprints grow quadratically with sequence length or graph size in these architectures.
In this paper, we address these challenges by factorizing STFM learning into two lightweight stages and introduce FactoST, a new paradigm that decouples universal temporal learning from domainspecific ST adaptation. Generally, temporal patterns (such as seasonality, trends) share a common 1-D structure across domains, learnable once; spatial correlations hinge on domain-specific graphs with sizes and semantics, often requiring tailored reasoning. FactoST exploits this asymmetry: it first distils the simpler temporal dynamics across domains, then attaches a compact adapter that injects the richer, domain-specific spatial knowledge. Conceptually, FactoST can be seen as an "STGNN" in the era of FMs, reinstating spatio-temporal factorization at scale (see Figure 1(a)→(c)).
this section cite: ['b76', 'b12', 'b57', 'b72', 'b57', 'b38', 'b46', 'b26', 'b27', 'b31', 'b5', 'b45', 'b67', 'b63', 'b29', 'b33', 'b71', 'b17', 'b28', 'b31', 'b67', 'b63', 'b56', 'b40', 'b43', 'b4', 'b35', 'b16', 'b22', 'b13', 'b39', 'b68', 'b32', 'b69', 'b68', 'b32']

Section: Preliminary

this section cite: []

Section: Formulation
Definition of Spatio-Temporal Data. We define Spatio-Temporal (ST) data D = (X , A, M) as a sequence of multivariate observations recorded regularly at a fixed set of spatial locations. Formally, let V = {v 1 , . . . , v N } denote N nodes, e.g., traffic sensors, grid cells, and weather stations. Each node provides a D-dimensional feature vector at every time step over a horizon of length L, forming a tensor X ∈ R N ×L×D . Spatial interactions are represented by an (often sparse) adjacency matrix A ∈ R N ×N , whose entries encode physical distance, functional similarity, or learned affinity. Many real-world datasets also carry node metadata M = {m i } N i=1 , such as geo-coordinates or land-use types that supply auxiliary spatial context. This formulation also subsumes several commonly used representations, including multivariate time series [79,80,12,56] and ST raster data [74,69,4,19].
this section cite: ['b78', 'b79', 'b11', 'b55', 'b73', 'b68', 'b3', 'b18']

Section: Goal of Spatio-Temporal Foundation Models (STFM).
Most existing STFMs centre on ST forecasting, as accurate future prediction is the cornerstone for a majority of ST applications, from traffic management to weather early-warning systems [36,69,17]. STFMs therefore seeks to learn a representation function Φ(•) that converts a large, cross-domain corpus of ST datasets into a general-purpose representation H = Φ(D 1 , . . . , D n d ), where n d is the number of ST datasets. The key requirements for such a representation are: i) Versatility: It should support a broad spectrum of downstream forecasting tasks, including both short-term and long-term forecasting across diverse domains. ii) Efficiency: Adapting to a new task or domain must involve only a lightweight prediction head and minimal fine-tuning, while still matching or surpassing fully retrained, task-specific models.
this section cite: ['b35', 'b68', 'b16']

Section: Related Work
Spatio-Temporal Graph Neural Networks (STGNNs). STGNNs are the de-facto backbone for learning representations from complex ST data, powering tasks that range from ST forecasting and anomaly detection to classification and imputation [52,27,28,15,2]. Early STGNNs often factorize the learning problem into two complementary components: (i) temporal modules (e.g., RNNs [32,24], TCNs [64,65]) to extract sequential patterns at individual nodes, and (ii) spatial modules (e.g., GCNs [29,78], GATs [72,51]) to propagate information across graph edges. This factorized design, implemented either in stacked form [68,76] or as tightly coupled pipelines [32], has proven effective across diverse domains such as traffic forecasting [6,26], energy industry [1,53] and environmental applications [35,11]. Building on these foundations, recent work explores selfsupervised objectives (e.g., contrastive or generative pretext tasks) to extract domain-agnostic ST features without dense labels [38,25]. Transformer-style STGNNs further extend receptive fields with a self-attention mechanism while retaining the ST factorization [66,37]. Despite these advances, most existing models are still trained from scratch for each dataset, which limits their cross-domain reuse and falls short of "training once, adapt everywhere".
this section cite: ['b51', 'b26', 'b27', 'b14', 'b1', 'b31', 'b23', 'b63', 'b64', 'b28', 'b77', 'b71', 'b50', 'b67', 'b75', 'b31', 'b5', 'b25', 'b0', 'b52', 'b34', 'b10', 'b37', 'b24', 'b65', 'b36']

Section: Spatio-Temporal Foundation Models.
Recent efforts have explored STFMs that learn universal representations through cross-domain pretraining [36,17]. Notable examples include UNIST [69] and OPENCITY [33], which apply transformer-based architectures to large-scale traffic data. As shown in Figure 1(b), UNIST tokenizes ST data into a sequence of ST tokens for Transformer-based learning and prompt-based adaptation. OPENCITY integrates Transformers with GNNs for flexible graph modeling, yet its tightly coupled architecture demands domain-specific pre-processing (e.g., road networks) and is prone to overfitting to particular spatial dependencies. Both models rely on expensive joint ST pretraining, leading to suboptimal performance and high computational cost. In contrast, time series foundation models like TIMESFM [10] and CHRONOS [3] achieve strong cross-domain generalization through purely temporal pretraining, but they lack any spatial awareness.
Our factorized framework bridges this gap: it first learns universal temporal patterns in a scalable manner, then injects lightweight spatial adapters for rapid ST adaptation, achieving both versatility and efficiency without the heavy cost of joint pertaining on both spatial and temporal dimensions.
this section cite: ['b35', 'b16', 'b68', 'b32', 'b9', 'b2']

Section: Multi-Frequency Augmentation
Region 1 Region n Spatio-Temporal Metadata Fusion (STMF) Linear 𝐻, Norm & Patchify Region 1 Region 2 Region 3 Linear concat + + + + + + + + + + + … … … … … … … … … … …
this section cite: []

Section: +

this section cite: []

Section: Adjacency Matrix
1 0 0 1 0 0 1 0 1 0 1 0 0 0 1 0 1 1 0 1 0 1 0 1 1 0 0 0 0 0 𝑆)
this section cite: []

Section: Spatial Affinity
× … 𝑆, … Temporal Affinity 𝑆: Time-Lagged Affinity … α; dynamic filter M) 𝑅 𝑁 × 𝑁 𝑅 × top-k 𝛼/ dynamic filter M/ 𝑅 𝑁 × 𝑁 𝑅 × top-k 𝛼< dynamic filter M: 𝑅 𝑁 × 𝑁 𝑅 × top-k + e! e" e# e$ e% e& e' e6 Hierarchical Domain Alignment (HDA) Projection Token level Alignment u v × Layer level Alignment 𝛼! 𝛼" 𝛼# Mean Spatio-Temporal Tokens … … … 0 * 1 * 2 * L * … 0 * 1 * 2 * L * … 𝑥0 𝑒8 Continual Memory Replay (CMR) Mem Buffer Current Stream Combined Tokens data mixing Transformer Encoder Domain Prompt Tokens Transformer Decoder Prediction Head Future Value w/ Position Embedding Multi-Task Pretraining Prediction Head Recover Head self-supervised supervised 𝐸= 𝐸5 min-of-hour hour-of-day day-of-week day-of-month coordinate Linear 𝐻), 𝐻>7)0: Spatio-Temporal Filtering (STF) … E5 × H), … E5 × H), … … … 𝐻), ,?@ × H), iFFT Random Threshold 𝑥A/B 𝑥+78 𝑥C 𝑥D 𝑥A/B Concat Decoder Encoder x N Feed Forward Network Multi-Head Attention x N Feed Forward Network Multi-Head Attention
this section cite: []

Section: Methodology
Figure 2 presents the framework of FactoST for factorized STFM, consisting of two stages: • Given cross-domain ST data d ∈ R N ×L×D with N nodes, L time steps, and D-dimension, we apply a compact general-purpose temporal backbone T : R L×D → R F ×D independently to each node's sequence d[i, :, :] to predict the future horizon F . This stage integrates multi-frequency augmentation, domain prompting, and multi-task to learn universal temporal patterns.
• A lightweight adapter S is designed to rapidly adapt T to specific ST domains. For downstream input x in ∈ R N ×L×D , we reuse the pretrained backbone T to extract node-wise temporal features z ∈ R N ×L×d , where d denotes the model's hidden dimension. S-parameterized by Φ with ∥Φ∥ ≪ ∥T ∥-then refines these features by injecting ST metadata m, pruning redundant ST interactions, aligning domain gaps, and performing strategic sample mixing for fine-tuning. The final output y out = S(z; m) ∈ R N ×F ×D yields forecasts for the F -step future horizon.
this section cite: []

Section: Stage I: Universal Temporal Pretraining (UTP)
To distill transferable temporal dynamics across heterogeneous ST domains, we pretrain a spatially agnostic temporal backbone on node-wise time series using a Transformer encoder-decoder architecture. This stage deliberately omits any spatial graph structure, enabling scalable and domain-agnostic learning of universal temporal patterns such as periodicity, trends, and multi-scale fluctuations.
this section cite: []

Section: Multi-Frequency Augmentation.
Temporal patterns often exhibit both long-term trends and shortterm fluctuations. Drawing on the ideas from [63,8,60], we adopt a frequency isolation strategy to generate diverse temporal views that emphasize distinct spectral components. Given a raw input sequence x ∈ R L×D , we first apply the Fast Fourier Transform (FFT) to obtain its spectral representation x f ∈ C F , where F = ⌊L/2⌋ + 1. We then stochastically isolate either low-or high-frequency bands by sampling K m random cutoff thresholds {τ i } Km i=1 with τ i ∼ (0, F ), and binary selectors {µ i } Km i=1 with µ i ∼ {0, 1}. For each pair (τ i , µ i ), we retain only the frequency components below τ i if µ i = 0, or above τ i if µ i = 1, effectively creating a spectrally filtered version of the signal. The filtered spectra are transformed back to the time domain via inverse FFT, yielding K m augmented views {x (i) m } Km i=1 ∈ R L×D . These views-along with the original input-are independently patched into non-overlapping segments of length L ′ and projected into d-dimensional tokens. The resulting token sequences form a multi-view temporal tensor x aug ∈ R Km×N ′ ×L ′ ×d , where N ′ = L/L ′ , serving as input to the Transformer encoder. This design encourages the model to learn representations that are consistent across complementary frequency perspectives.
this section cite: ['b62', 'b7', 'b59']

Section: Multi-Domain Prompting.
Drawing inspiration from codebooks in vision [43], we propose a soft domain prompting mechanism to encode cross-domain contextual cues. Specifically, we construct a learnable codebook P = {p 1 , . . . , p Kp } ∈ R Kp×d , where each vector represents a prototypical temporal context from a specific domain. Given an input x, we first extract a compact embedding x e ∈ R d via global pooling and a linear projection. We then compute its similarity with each prompt vector p j ∈ P using negative squared Euclidean distance. Subsequently, we apply softmax normalization to derive attention weights, and obtain the final domain-specific prompt x p ∈ R d via weighted combination, thereby fusing reusable knowledge across multiple domains. Finally, x p is expanded into N p tokens and concatenated with the patched input to obtain x ctx ∈ R (N ′ +Np)×d .
s j = -∥x e -p j ∥ 2 2 , α j = exp(s j ) Kp k=1 exp(s k ) , x p = Kp j=1 α j p j , j = 1, . . . , K p .(1)
where s j is the similarity score, α j is the attention weight, and x p is the domain context prompt.
Multi-Task Pretraining. To simultaneously capture universal temporal structures and enable effective cross-domain transfer, we jointly optimize two complementary objectives: a self-supervised task that enforces consistency across multi-frequency views of the input, and a supervised forecasting task that guides the model to learn predictive, domain-aware representations:
• Self-supervised spectral consistency: The model reconstructs the original time series from the multi-view augmented input x aug , ensuring learned representations preserve coherent information across complementary frequency bands. Specifically, x aug is encoded by a Transformer to capture deep temporal interactions, then decoded via a Transformer decoder and a linear head to regenerate the original sequence, optimized with mean squared error (MSE) loss.
L spec = x -SpecHead Decoder(Encoder(x aug )) 2 2 .(2)
• Supervised forecasting with prompt alignment: The model forecasts future values from the domainprompted input x ctx to evaluate representation quality. x ctx passes through the shared Transformer encoder-decoder, but the decoder output is detached before the prediction head-using the forecasting loss as a non-backpropagated supervisory signal. This ensures only the restoration task updates shared parameters, preventing negative transfer from task-specific biases.
L pred = y -PredHead detach Decoder(Encoder(x ctx )) 2 2 + ∥x e -x p ∥ 2 2 , (3
)
where y is the ground-truth. The first term minimizes supervised forecasting error, while the second enforces prompt consistency between the input embedding x e and its soft domain prompt x p , inspired by the codebook alignment objective in VQ-VAE [54] to stabilize training and enhance representation fidelity. The pretraining objective combines both losses: L = L spec + L pred . During subsequent ST fine-tuning, domain prompts are frozen, and only the Transformer layers and prediction head are updated using L pred , enabling efficient adaptation while preserving pretrained knowledge.
this section cite: ['b42', 'b53']

Section: Stage II: Spatio-Temporal Adaptation
To adapt the pretrained temporal backbone to ST scenarios, we introduce four lightweight modules that incur minimal parameter overhead while effectively capturing ST dependencies.
this section cite: []

Section: Spatio-Temporal Metadata Fusion (STMF).
This module injects ST context into the temporal backbone via learnable identifiers. Given ST input X ∈ R N ×L×D , we first get it temporal representations H t ∈ R N ×N ′ ×d via patch embedding layer, then we define: (1) nodespecific spatial embeddings E n ∈ R N ×De ; and (2) a calendar-aware temporal embedding bank {E c } c∈S , where each calendar type c (e.g., minute-of-hour, hour-of-day, day-of-week, day-of-month, month-of-year) has an embedding table E c ∈ R Kc×De , with K c the number of discrete bins for type c (e.g., 60/24/7/31/12). The active set S is chosen by sampling frequency (e.g., hourly: {hour-of-day, day-of-week, day-of-month, month-of-year}; minutely: add minute-of-hour).
For each node-patch pair (i, τ ), we map the start timestamp to calendar bins via ϕ c (τ ) ∈ {1, . . . , K c } for each c ∈ S, and form an ST identifier by concatenating the node embedding with the selected calendar embeddings as h st (i, τ
) = W p E i n c∈S E ϕc(τ ) c
+ b p , where ∥ denotes concatenation.
The identifiers' dimension are projected and expanded to obtain H st ∈ R N ×N ′ ×d , aligning with H t .
The encoded input representation and ST identifiers are then fused together by residual addition: H fused = H t + H st . This enables integration of ST context without retraining the temporal model.
this section cite: []

Section: Spatio-Temporal Filtering (STF).
While STMF uses static ST identifiers, STF adapts to scenariodependent cue relevance-e.g., local spatial context for incidents vs. global temporal patterns for rush hours-by dynamically reweighting spatial and temporal interactions via three learnable affinities.
From H st ∈ R N ×N ′ ×d , we extract spatial (E n ) and temporal (E t ) embeddings and compute:
• Spatial Affinity (S s ): Measures the compatibility between H st and its spatial component E n via dot-product: S s = ⟨H st , E n ⟩ ∈ R N ×N ′ , where higher values indicate stronger spatial relevance for each (node, patch) pair. This avoids rigid reliance on fixed spatial identifiers.
• Temporal Affinity (S t ): Quantifies alignment between H st and its temporal component E t :
S t = ⟨H st , E t ⟩ ∈ R N ×N ′ , capturing dominant temporal patterns while filtering redundant noise.
• Time-Lagged Affinity (S d ): Models asynchronous causal effects (e.g.,upstream nodes influencing downstream nodes with delay δ). For lags δ = 1, . . . , ∆, it aggregates historical neighbor states
H (t-δ) st
and computes:
S d = ∆ δ=1 γ (δ) • ⟨H st , Agg δ (H (t-δ) st
)⟩ ∈ R N ×N ′ , where γ (δ) are learnable lag weights. Higher values reflect stronger delayed relevance.
To improve scalability, all affinity computations can be performed in a low-rank space. Specifically, we project both operands into R r (r ≪ d) via shared or separate learnable matrices, e.g., for spatial affinity:
S s = ⟨H st W (s) q , E n W (s) k ⟩, where W (s) q , W (s) k ∈ R d×r .
Analogous projections apply to S t and S d . This reduces complexity from O(d) to O(r) per inner product while preserving semantic expressiveness. Top-K sparsification may further prune weak interactions.
The three affinities are stacked as S = [S s , S t , S d ] ∈ R N ×N ′ ×3 , projected to dimension d via W att ∈ R 3×d , and normalized with a softmax (temperature τ att ) to yield dynamic weights: Finally, the refined output is obtained by aggregating the three affinity scores into S = [S s , S t , S d ] ∈ R N ×L×3 , projecting to dimension D via learnable matrix W att ∈ R 3×d , normalizing with softmax (temperature τ att ) to get dynamic weights W, then modulating H st with W and applying LayerNorm:
W = softmax SW att τ att , H st = LayerNorm (H st ⊙ W) ∈ R N ×N ′ ×d
This design enables adaptive integration of ST context without retraining the temporal backbone, effectively balancing spatial and temporal semantics while suppressing irrelevant cues.
this section cite: []

Section: Hierarchical Domain Alignment (HDA).
To bridge the discrepancy across domains and facilitate effective transfer of domain adaptation knowledge, we propose a hierarchical alignment module using the pretrained domain prompts p ∈ R Kp×D , which operates at two levels:
1. Layer-level alignment: For an input embedding x e ∈ R D , we retrieve its k nearest prompts in p based on negative Euclidean distance and compute a soft domain prototype via averaging:
K(x e ) = Topk j∈[1,Kp] (-∥x e -p j ∥ 2 ) , pk = 1 k j∈K(xe) p j ∈ R d .(4)
2. Token-level alignment: To capture dataset-specific patterns beyond the pretrained prompt knowledge, we introduce a low-rank adaptation matrix A = uv ⊤ , where u ∈ R Np and v ∈ R D . The domain-aware adjustment is then computed as
X r = 1 Np p⊤ k ⊙ A ∈ R Np×d .
Where 1 Np is a column vector of ones and ⊙ denotes element-wise multiplication. The final representation fuses the layer-level prototype and token-level refinement for enhanced cross-domain generalization.
this section cite: []

Section: Continual Memory Replay (CMR).
To mitigate knowledge forgetting during few-shot adaptation, we implement dynamic data mixing, combining current data and historical data. First, we establish a memory buffer, given training sequences {X t } T t=1 of length T , we partition the dataset into:
M = {X t } Tm t=1 (memory buffer), C = {X t } T t=Tm+1 (current stream),(5)
with T m = ⌊memory_size • T ⌋ (default: 0.2), where the memory buffer M preserves critical temporal patterns from initial learning to ensure stability under domain shift.
this section cite: []

Section: Experiments
In our experiments, we aim to address the following research questions (RQ):
• RQ1: Can FactoST outperform prior approaches (including STGNNs, STFMs and other existing models) under few-shot and zero-shot scenaries? ⇒ Sec. 4.1 & Sec. 4.2.
• RQ2: Which model component is critical to the final performance? ⇒ Sec. 4.3.1.
• RQ3: How is the data and computation efficiency of FactoST? ⇒ Sec. 4.
this section cite: []

Section: & Sec. 4.3.3.
• RQ4: Can we provide interpretability of the domain adaptation process in FactoST? ⇒ Sec. 4.3.4.
• RQ5: Is the STA module architecture-agnostic, or limited to GNN-based backbones? ⇒ Sec. 4.3.5.
Datasets. We pretrain the temporal backbone on diverse ST datasets using Monash [16], covering six domains (energy, nature, health, transport, web, economics) with 130M observations across multiple spatial nodes and sampling frequencies from 4 seconds to daily. During pretraining, we extract and process univariate time series per node independently to prevent data leakage. For evaluation, we use eight established ST benchmarks-traffic flow (PEMS03/04/07/08), speed (PEMS-BAY, METR-LA), energy (Electricity), temperature (ETTh2), and climate (Weather)-which vary widely in spatial scale (21-883 nodes), temporal resolution (5 min-1 h), and sequence length (17k-52k steps), enabling a comprehensive assessment of cross-domain and multi-scale generalization (see A.1.1 for details).
Baselines. We compare FactoST with 12 competitive models across four categories: 1) STFMs: OpenCity [33], UniST [69]; 2) TSFMs: TimesFM [10], Moirai [62]; 3) ST expert models: BigST [20], STAEformer [37], STID [48], D2STGNN [49]; 4) Time series expert models: TimeMixer [59], PatchTST [42], DLinear [71], Informer [79]. Consistent with previous works [69], we adopted Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) as evaluation metrics. More in A.1.
this section cite: ['b15', 'b32', 'b68', 'b9', 'b61', 'b19', 'b36', 'b47', 'b48', 'b58', 'b41', 'b70', 'b78', 'b68']

Section: Few-shot Prediction
Setting. We evaluate few-shot adaptation using only 10% of labeled training data under two forecasting horizons: short-term (12 → 12) and long-term (96 → 96), following standard protocols [49,48].
Results. As shown in Tables 2 and 3, FactoST consistently outperforms all baselines across both short-and long-term horizons under few-shot adaptation. In the short-term setting (12 → 12), FactoST improves MAE over STFMs-OpenCity and UniST-by 31.4% and 47.2%, respectively, where UniST suffers from its rigid grid-based design and OpenCity incurs high graph-learning overhead. Against TSFMs (TimesFM, Moirai), we observe average gains of 18.8%, demonstrating that our lightweight ST adaptation effectively enriches universal temporal representations. Notably, FactoST remains competitive with specialized ST expert models despite avoiding domain-specific
this section cite: ['b48', 'b47']

Section: Zero-shot Prediction
Setting. We evaluate zero-shot forecasting for both short-and long-term scenarios without fine-tuning.
Results. As shown in Table 4, explicit joint ST pretraining does not improve zero-shot generalization.
In fact, models without dedicated spatial modeling-such as FactoST (UTP without STA) and TimesFM-consistently outperform specialized STFMs like OpenCity and UniST. This confirms our core insight: spatial structures are highly domain-specific and hinder transfer when baked into pretraining. UniST's unstable performance-reasonable on long-term but catastrophic on short-term tasks-and its need for retraining under topology changes further expose the rigidity of fixed spatial priors. Even in-domain, OpenCity and Moirai (a multivariate TSFM) underperform, underscoring that strong zero-shot capability stems from temporal, not spatial, modeling. Remarkably, FactoST achieves the most top-2 rankings (4 first, 5 second) among all foundation models on unseen domains, despite using only 13M pretraining points-orders of magnitude fewer than Moirai (27B) or TimesFM (100B) (Table 6). Nonetheless, errors remain high in long-term scenarios, highlighting the intrinsic challenge of zero-shot ST prediction. To address this, we apply Test-Time Computing [9], which reduces zero-shot MAE and RMSE by 7.72% and 8.79% on average; see Appendix A.4.4 for details.
this section cite: ['b8']

Section: Model Analysis Ablation Studies.
We evaluate each component's contribution via ablation on PEMS-03 short-term forecasting (Figure 3). Removing CMR causes the largest drop (↑35.15% MAE, ↑32.85% RMSE), underscoring its role in preserving knowledge during few-shot adaptation. Disabling HDA or STMF degrades MAE by 22%, confirming their importance for domain alignment and metadata fusion. Among STF variants, omitting the spatial affinity matrix yields the sharpest decline (↑29.01% MAE, ↑33.70% RMSE), highlighting its efficacy in dynamic interaction modeling; temporal and time-delay matrices also contribute, albeit more moderately. We also find that the spectral consistency loss L spec in pretraining provides nontrivial gains, more details are in Appendix A.4.3.
this section cite: []

Section: Scaling Analysis.
We evaluate FactoST on PEMS-03 across downstream fine-tuning proportions from zero-shot to full-shot for both short-and long-term forecasting (Figure 4). Remarkably, FactoST achieves rapid performance gains with minimal data: in the short-term setting, MAE drops from 25.96 (1% data) to 17.54 (10% data)-already approaching full-shot performance (16.59). In the long-term setting, MAE plummets from 123.57 (zero-shot) to 28.57 with just 10% of the training data, and further improves to 25.
this section cite: []

Section: Efficiency Analysis.
As shown in Figure 5, FactoST achieves a strong MAE of 17.86 on PEMS-03 under the 10% few-shot setting with only 1.3M parameters and 8.1s inference time-outperforming nearly all baselines in both accuracy and efficiency. In contrast, joint ST models like OpenCity (1.67M params, 25.3s) incur high computational overhead due to end-to-end graph learning, while large-scale TSFMs such as Moirai (91.4M params, 22.2s) ignore spatial structure entirely. Even though models like D2STGNN attain slightly better accuracy, they suffer from severe latency (54.5s). By decoupling universal temporal pretraining from lightweight, plug-and-play spatial adaptation, FactoST achieves an exceptional accuracy-efficiency trade-off, making it highly suitable for real-world deployment.
this section cite: []

Section: Domain Adaptation Analysis.
Figure 6 visualizes ST token embeddings before and after Hierarchical Domain Adaptation (HDA) via t-SNE: original tokens (blue), adapted tokens (orange), and learned domain prompts (circled clusters). The adapted embeddings shift clearly toward their corresponding prompt clusters, demonstrating that HDA effectively steers generic temporal representations toward domain-specific semantics. Crucially, the global structure of the embedding space is preserved-indicating that adaptation is targeted and non-destructive. This provides empirical evidence that FactoST successfully bridges universal knowledge from pretraining with task-specific spatio-temporal context, enabling effective cross-domain transfer without catastrophic forgetting.
Architecture Generality of STA. The STA module is architecture-agnostic, operating solely on feature embeddings without relying on GNN-specific inductive biases. To verify this, we integrate STA into PatchTST-a non-GNN, Transformer-based time series model-and observe consistent few-shot improvements across all datasets ( Figure 7). This confirms STA's ability to inject ST context into diverse backbones. FactoST remains superior, benefiting from large-scale pretraining; this highlights that temporal knowledge learned from diverse domains generalizes effectively, and when combined with lightweight spatial adaptation, enables strong cross-domain performance.   5 Conclusion and Future Work We introduce FactoST, a two-stage spatio-temporal foundation model (STFM) that decouples universal temporal pretraining from lightweight spatio-temporal adaptation. This factorized design avoids the computational cost and poor generalization of joint ST pretraining in existing STFMs. Empirically, FactoST outperforms current STFMs in few-shot settings-reducing MAE by up to 46.4% over UniST-while using 46.2% fewer parameters and achieving 68% faster inference than OpenCity. Notably, it matches or surpasses domain-specific models without architectural customization, demonstrating the power of factorized pretraining as a scalable path toward universal STFMs.
We identify two key directions for future work. First, spatial generalization remains a bottleneck: limited zero-shot performance across STFMs-including our temporal-only backbone-suggests that rigid spatial priors impede cross-domain transfer, calling for more adaptive, semantics-aware representations. Second, the fine-tuning protocol can be improved: uniform parameter updates underutilize pretrained knowledge; parameter-efficient strategies (e.g., prompt tuning or selective retraining) could enhance transfer efficiency and mitigate catastrophic forgetting.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] .
Justification: The paper clearly states the contributions of FactoST, a novel spatio-temporal foundation model framework that decouples universal temporal learning from task-specific spatial adaptation. These claims are backed by theoretical motivation (e.g., multi pre-training and fine-tuning strategies) and extensive experiments across diverse domains (traffic, energy, weather), demonstrating strong few-shot performance and computational efficiency.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] .
Justification: In Section 5, we discuss the limitations of this work.
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

Section: Theory Assumptions and Proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] . Justification: All the proofs in this paper can be found in the method section and are verified in the experimental part.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental Result Reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] . Justification: We will provide detailed descriptions of the methods and experimental setup in the supplementary material. Additionally, we will release our code to ensure faithful reproduction of the main results upon the paper's acceptance.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in
•
The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments Compute Resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] . Justification: We provide our detailed experimental setup and complexity analysis.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code Of Ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] . Justification: We follow the NeurIPS Code of Ethics in this paper.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader Impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [Yes] . Justification: We provide a discussion of the broad implications in Appendix B. Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] .
Justification: The datasets chosen in this paper are commonly used benchmark datasets for spatio-temporal forecasting tasks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] .
Justification: Yes, we have.
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

Section: New Assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes] .
Justification: This paper follows CC 4.0, and the code is in an anonymized URL.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and Research with Human Subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] . Justification: This paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
this section cite: []

Section: Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
Answer: [NA] .
Justification: This paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [No] Justification: Large language models were used only for minor tasks such as grammar checking and formatting improvements. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: A.1.2 Model Architecture and Hyperparameters
We implement FactoST using PyTorch, and all experiments are conducted on high-performance GPU servers. The architecture consists of three encoder layers and three decoder layers, with 16 attention heads and a latent dimension d = 128. Input sequences are processed using a patching mechanism with a patch size of 12, and the dropout rate is set to 0.2 to prevent overfitting. The feed-forward network within each Transformer layer has a hidden dimension of 512.
Pretraining. During pretraining, we use the Adam optimizer with an initial learning rate of 5 × 10 -4 , and apply StepLR to decay the learning rate by a fixed factor every few epochs, improving convergence. The model is equipped with N p = 8 domain prompt learning vectors, each of dimension 128, and in supervised prediction tasks, both the input length and target forecasting horizon are fixed at 96 (The length can be set to any value, which is the maximum supported step length, here we set 96 for downstream comparison). For spectral consistency modeling, the number of augmented patches is set to K f = 4. Pretraining is performed with a large batch size of 16,384 to ensure stable optimization.
Fine-tuning. During fine-tuning, we adopt a learning rate of 1 × 10 -3 . The lookback window is set to 12 (short-term) or 96 (long-term), with matching prediction horizons. The number of domain prompt tokens (N p = 3) and patching configuration remain unchanged from pretraining. A top-k selection (k = 3) is applied during domain prompt matching to enhance generalization. Additional configuration includes memory replacement ratio of 0.3; memory size of 0.2 relative to total capacity; spatio-temporal identifier embedding dimension of 32; maximum delay step ∆ = 3.
this section cite: []

Section: A.2 Evaluation Metrics
We use commonly used regression metrics, Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), to measure the prediction performance. Suppose Y = Y 1 , ..., Y M are ground truth for real spatio-temporal data, Ŷ = Ŷ1 , ..., ŶN are the predicted values by the model, and N is the number of total testing samples, These two metrics can be formulated as follows:
RMSE(Y, Ŷ) = 1 N N i Y i -Ŷi 2 , MAE(Y, Ŷ) = 1 N N i Y i -Ŷi ,(6)
this section cite: []

Section: A.3 Baselines and Implementation
All baseline models are evaluated within a unified framework to ensure fair comparisons. All models are assessed using standardized metrics (MAE and RMSE). Hyperparameters are either set to default values reported in the original papers or tuned via grid search on the validation set. Below, we detail the implementation strategies for each category of baselines.
this section cite: []

Section: A.4 More Results

this section cite: []

Section: A.4.1 Model Size Analysis
As shown in Figure 9, we investigate the effect of model capacity by varying the number of Transformer layers in FactoST's temporal backbone (3.0M → 4.3M parameters) on ETTh2 long-term forecasting. Zero-shot performance improves steadily from 1 to 3 layers and then plateaus, indicating diminishing returns from deeper architectures. In the 10p few-shot setting, performance peaks at 3 layers and slightly degrades with 5-7 layers, with training logs revealing signs of overfitting-likely due to increased depth without proportional increases in regularization or hidden dimensionality. These results suggest that moderate model capacity is optimal under data-limited conditions, aligning with the principle of Occam's razor in transfer learning.
this section cite: []

Section: A.4.2 Pretraining Data Scalability
Figure 8 shows the impact of pretraining corpus size on generalization. We train FactoST on 20% to 100% of Monash dataset and evaluate on ETTh2 long-term forecasting. Performance improves monotonically with more pretraining data, confirming that FactoST effectively leverages larger and more diverse temporal corpora. Notably, even at 100% data (13M points), FactoST remains far below the scale of leading foundation models (e.g., TimesFM: 100B), suggesting substantial headroom for improvement with access to richer and high quality pretraining sources. Figure 9: Model size analysis.
this section cite: []

Section: A.4.3 Pretraining Objective Ablation
Our pretraining objective combines future prediction loss and spectral consistency loss with equal weighting: L pretrain = L pred + L spec . While L pred captures temporal dynamics through supervised forecasting, L spec enables self-supervised modeling of multi-frequency patterns via spectral consistency learning across frequency-isolated views. As shown in Table 8, removing L spec degrades zero-shot MAE/RMSE by 3.54%/5.27%, but only slightly affects few-shot performance (↑0.78%/0.87%). This indicates that multi-frequency spectral consistency provides complementary, non-redundant signals that are especially crucial when no target-domain supervision is available. In zero-shot transfer, L spec endows the model with robust spectral inductive biases that generalize across domains. During few-shot fine-tuning, limited labels allow the model to partially recover domain-specific patterns, reducing reliance on L spec . The modest few-shot gain likely reflects the moderate domain shift in our benchmarks; we expect L spec to yield larger benefits under stronger distributional shifts (e.g., cross-city or cross-modality transfer).
this section cite: []

Section: A.4.4 Zero-Shot Enhancement
To address the inherent limitations of zero-shot forecasting, we integrate Test-Time Computing (TTC) [9] into FactoST-a lightweight online adaptation mechanism that refines predictions during inference without retraining. TTC maintains a FIFO memory queue of recent inputs, predictions, and (pseudo) labels, constructs a frequency-domain calibrator using FFT-based amplitude/phase offsets, and updates the calibrator using only historical predictions to avoid temporal leakage. As shown in Table 9, TTC consistently improves long-term zero-shot performance across datasets, reducing MAE and RMSE by 7.72% and 8.79% on average. Future work includes: (1) spatial meta-learning to capture universal topological priors (e.g., distance decay) [22]; (2) semantic-enhanced spatial embeddings via external knowledge (e.g., LLM-derived geolocation representations) [21]; and (3) cross-domain latent alignment to bridge spatial representation gaps [55].
this section cite: ['b8', 'b21', 'b20', 'b54']

Section: A.4.5 Temporal Feature Granularity Analysis
FactoST supports flexible multi-scale periodicity modeling through its spatio-temporal metadata fusion (STMF) module. While our main experiments use time_of_day (24) and day_of_week (7) identifiers-reflecting the dominant daily/weekly cycles in high-frequency traffic datasets like PEMS-03-the architecture readily accommodates longer-term patterns (e.g., monthly, yearly) by simply extending the temporal feature set, without any architectural changes. To validate this flexibility and assess the impact of temporal granularity, we replace daily/weekly features with month_of_year (12) in a few-shot setting on PEMS-03. As shown in Table 10, performance degrades significantly: shortterm MAE increases by 12.94% and long-term MAE by 15.20%. This confirms that temporal feature design must align with the intrinsic frequency of the data-a principle our framework inherently supports through plug-and-play metadata injection.
this section cite: []

Section: B Broader Impacts
Our work introduces a factorized framework for spatio-temporal foundation models that enhances efficiency, generalization, and cross-domain adaptability. By decoupling universal temporal pretraining from lightweight spatio-temporal adaptation, our approach significantly reduces computational cost and enables rapid few-shot deployment-making it well-suited for real-world applications with limited labeled data or constrained resources. The proposed method has the potential to benefit highimpact domains such as urban planning, traffic optimization, climate modeling, energy forecasting, and public health surveillance-areas where accurate, scalable, and transferable spatio-temporal prediction is crucial. Furthermore, the modular design promotes sustainable AI development by minimizing redundant large-scale pretraining and reducing overall energy consumption.
As this work primarily focuses on scientific research and technical innovation in spatio-temporal modeling, it does not present clear negative societal impacts. Instead, it contributes to the development of more accessible, efficient, and environmentally responsible foundation models for real world urban dynamics with both spatio and temporal characteristics.
this section cite: []

Section: B.1 Limitations
Our work focuses on separating temporal pretraining from spatial adaptation to enable efficient and generalizable spatio-temporal modeling. While this factorized design offers strong empirical performance and flexibility, several limitations point to important directions for future research:
• Spatial modeling impedes zero-shot generalization. Our results reveal a key insight: explicit spatial modeling-especially when baked into pretraining-hurts cross-domain transfer because spatial structures (e.g., graph topology, sensor layout) are highly domain-specific. In contrast, temporal-only pretraining (as in FactoST 's UTP stage or TimesFM) achieves superior zero-shot performance, confirming that universal temporal patterns, not spatial priors, drive generalization. This explains why specialized STFMs (e.g., UniST, OpenCity) underperform or even fail catastrophically on unseen domains. While our factorized design avoids this pitfall by deferring spatial adaptation to the lightweight adapter, zero-shot spatio-temporal forecasting remains inherently challenging-particularly for long horizons-highlighting the need for complementary techniques (e.g., test-time computing [9]) to further bridge the domain gap.
• Challenges with dynamic and open-world spatial structures. The current framework assumes fixed node sets and static spatial topologies (e.g., traffic sensors or weather stations). It is not designed to handle scenarios where spatial units are added, removed, or reconfigured over time (e.g., mobile sensors, evolving infrastructure, or ad-hoc networks). Extending FactoST to support zero-shot spatial generalization-such as generating embeddings for unseen nodes or adapting to changing graph structures [31,81]-remains an open challenge.
• Dependency on pretraining corpus diversity. Although our adapter-based design reduces computational overhead, the universal temporal backbone still relies on the breadth and representativeness of the pretraining data [75,75]. In domains with strong physical laws or complex spatial couplings (e.g., power grids), the model may lack necessary inductive biases, limiting transferability.
• Limited integration of exogenous variables. Our current framework does not explicitly model external factors such as weather, events, or policy interventions-critical covariates in many real-world forecasting tasks [61,67]. Developing mechanisms to incorporate and adapt to such exogenous signals in a few-shot manner is an important direction for enhancing practical utility.
• Static adapter composition. The adapter S currently applies a fixed set of modules (STMF, STF, HDA, CMR) regardless of input characteristics. A more intelligent system could enable adaptive model composition: by analyzing temporal stability, spatial heterogeneity, or domain shift, it could dynamically choose between full spatio-temporal modeling, temporal-only inference, or specialized lightweight modules-improving both robustness and efficiency [50].
• Suboptimal fine-tuning protocols. Current adaptation uses uniform gradient updates on the adapter, which may underutilize pretrained knowledge and risk catastrophic forgetting. Parameterefficient strategies-such as prompt tuning, selective layer retraining, or regularization-aware updates-could better preserve temporal priors while enabling efficient spatial adaptation [45,7].
These limitations point to several promising directions for future work: (1) rethinking spatial modeling to avoid domain-specific biases in pretraining, (2) enabling adaptation to dynamic or unseen spatial configurations, (3) enriching temporal foundations with exogenous context, (4) leveraging more efficient and stable fine-tuning strategies, and (5) developing hybrid inference mechanisms (e.g., test-time adaptation) to bridge the zero-shot performance gap. Addressing these challenges will be key to building truly robust, scalable, and practical spatio-temporal foundation models.
this section cite: ['b8', 'b30', 'b80', 'b74', 'b74', 'b60', 'b66', 'b49', 'b44', 'b6']

Section: B.2 Pseudocode of FactoST
For reproducibility, we present the detailed pseudocode of FactoST in Algorithm 1, which concisely summarizes the two-stage learning paradigm: universal temporal pretraining (UTP) followed by lightweight spatio-temporal adaptation (STA).
this section cite: []

Section: 
some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] .
Justification: We will release our code to ensure faithful reproduction of the main results upon the paper's acceptance.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental Setting/Details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] .
Justification: We have provided detailed experimental settings in Appendix A.1 to facilitate reproducibility.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment Statistical Significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [No] .
Justification: In spatio-temporal forecasting tasks, it is common not to provide error bars but instead directly calculate the average by conducting multiple experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
this section cite: []

Section: A Technical Appendices

this section cite: []

Section: A.1 Implementation Details
This section provides a comprehensive overview of the implementation setup, including datasets, evaluation metrics, hyperparameters, implementation details, and training configurations.
this section cite: []

Section: A.1.1 Datasets
Pretraining Datasets: As shown in Table 5, we use cross-domain large-scale ST datasets, including energy, nature, transportation, and web, to train our temporal backbone. From these datasets, we extract node-wise temporal sequences across different locations over time. These sequences span multiple frequencies (from seconds to daily) and exhibit various temporal patterns, ensuring that the learned representations are robust and transferable across different forecasting tasks. We also add a data volume comparison in Table 6, where FactoST is much lower than other foundation models.
Table 5: List of pretraining spatio-temporal datasets.
this section cite: []

Section: Dataset Domain Frequency # Time Points Source
Aus. Electricity Demand Energy Half Hourly 1155264 Monash [16] Wind Energy 4 Seconds 7397147 Monash [16] Wind Farms Energy Minutely 172178060 Monash [16] Solar Energy 10 Minutes 7200720 Monash [16] Solar Power Energy 4 Seconds 7397222 Monash [16] London Smart Meters Energy Half Hourly 166527216 Monash [16] Temperature Rain Nature Daily 23252200 Monash [16] Saugeen River Flow Nature Daily 23741 Monash [16] Sunspot Nature Daily 73924 Monash [16] Weather Nature Daily 43032000 Monash [16] KDD Cup 2018 Nature Daily 2942364 Monash [16] US Births Nature Daily 7305 Monash [16] Pedestrian_Counts Transport Hourly 3132346 Monash [16] Web Traffic Web Daily 116485589 Monash [16] Bitcoin Economic Daily 75364 Monash [16] Table 6: Pretraining corpora and scale of STFMs and TSFMs used in this study.
this section cite: ['b15', 'b15', 'b15', 'b15', 'b15', 'b15', 'b15', 'b15', 'b15', 'b15', 'b15', 'b15', 'b15', 'b15', 'b15']

Section: Model Pretraining Corpus Scale

this section cite: []

Section: FactoST
Monash (5 domains, 15 datasets) 13M time points OpenCity 21 heterogeneous traffic datasets (10,110 regions) 151.1M observations UniST 21 multi-source grid datasets -TimesFM Large-scale real-world and synthetic time series 100B time points Moirai LOTSA dataset 27B time points
Evaluation Datasets: For downstream ST forecasting, we select real-world benchmarks covering traffic flow, speed, electricity consumption, and meteorological data, as detailed in Table 7. These datasets exhibit substantial heterogeneity in spatial granularity (e.g., city regions vs. sensors), temporal resolution (e.g., 5-minute vs. hourly intervals), and prediction targets (e.g., speed vs. volume), posing a challenging testbed for cross-task generalization. This diversity enables a rigorous evaluation of FactoST 's adaptability under both short-term and long-term forecasting settings.
this section cite: []

Section: A.3.1 Time Series (TS) Expert Models
The following TS expert models are implemented using the BasicTS framework (https://github.  com/GestaltCogTeam/BasicTS) to ensure consistency in preprocessing, training, and evaluation:
• TimeMixer [59]: A MLP-based model with past-decomposable mixing and future multi-predictor mixing, enabling multiscale information fusion from both microscopic and macroscopic views.
• PatchTST [42]: A Transformer-based model that treats time series as a sequence of patches, enabling effective long-term forecasting by capturing local and global patterns.
• DLinear [71]: A simple yet effective linear model that decomposes time series into trend and residual components, followed by independent modeling of each component for accurate forecasting.
• Informer [79]: An efficient Transformer variant with self-attention compression and generative decoder design, tailored for long sequence time series forecasting.
this section cite: ['b58', 'b41', 'b70', 'b78']

Section: A.3.2 Spatio-Temporal (ST) Expert Models
The following ST expert models are also integrated into the BasicTS framework, Graph structures follow original designs, utilizing distance-based or learned adjacency matrices where applicable.
• BigST [20]: Proposes a linear STGNN, first extracts long sequence input into a low representation, then uses a global GCN to capture spatial features, effective for large sensor node scenarios.
• STAEformer [37]: Utilizes spatial-temporal adaptive embeddings to enhance the representation learning capability of Transformers for traffic forecasting tasks.
• STID [48]: Introduces spatial-temporal identity vectors into the Transformer architecture to capture node-specific temporal dynamics and spatial dependencies.
• D2STGNN [49]: Decouples spatial and temporal dependencies using separate graph convolution and recurrent modules for improved modeling of complex spatio-temporal interactions.
this section cite: ['b19', 'b36', 'b47', 'b48']

Section: A.3.3 Time Series Foundation Models (TSFMs)
For TSFMs, we adapt the official implementations to align with our benchmarking protocol:
• TimesFM [10]: A large-scale pretrained decoder-only time series foundation model developed by Google Research, capable of high-accuracy univariate forecasting across diverse domains and frequencies. The implementation is obtained from the official repository (https://github.  com/google-research/timesfm), and we fine-tune them using context lengths and forecast horizons consistent with our experimental setup. The checkpoint of the model we use comes from https://huggingface.co/google/timesfm-1.0-200m.
• Moirai [62]: A large-scale pretrained encoder-only time series foundation model developed by Salesforce AI Research, designed to deliver universal forecasting capabilities across diverse domains, frequencies, and variable types. The implementation is obtained from the official repository https://github.com/SalesforceAIResearch/uni2ts. The checkpoint of the model we use comes from https://huggingface.co/Salesforce/moirai-1.0-R-base.
this section cite: ['b61']

Section: A.3.4 Spatio-Temporal Foundation Models (STFMs)
We evaluate two recent STFMs:
• UniST [69]: A universal STFM empowered by prompt learning, pretrained on multiple urban scenarios to achieve strong generalization. Official codebase (https://github.com/  tsinghua-fib-lab/UniST) supports fixed horizon configurations only 6-step prediction. We retrain it on 13 datasets from the original release to support 12 and 96-step forecasting scenarios.
• OpenCity [33]: A versatile STFM that supports zero-shot and few-shot forecasting across diverse city-level applications. Integrated into our pipeline using the checkpoint Opencity-plus.pth, with adapter layers introduced to enable efficient few-shot adaptation to new datasets.
this section cite: ['b68', 'b32']

Section: Algorithm 1 FACTOST: Factorized Spatio-Temporal Foundation Model
Require: Cross-domain ST datasets D = {(X (j) , Y (j) , m (j) )} j , where X (j) ∈ R Nj ×L×D , Y (j) ∈ R Nj ×F ×D , and m (j) denotes ST metadata. 1: // Stage I: Universal Temporal Pretraining (UTP) 2: Initialize temporal backbone T and domain prompts p ∈ R Kp×d 3: for each node-wise sequence x ∈ R L×D sampled from D do 4: // Multi-frequency augmentation 5:
x f ← FFT(x) 6:
for i = 1 to K m do 7: Sample τ i ∼ Uniform(0, ⌊L/2⌋ + 1), µ i ∼ Bernoulli(p) 8: Mask x (i) f ← x f [τ i :] = 0 if µ i = 0 x f [: τ i ] = 0 if µ i = 1 9: x (i) m ← iFFT(x (i) f ) 10:
end for 11:
x aug ← [x, {x (i) m } Km i=1 ]; apply patching → R Km×N ′ ×L ′ ×d 12:
// Multi-domain prompting 13:
x e ← Linear(Patch(x)) ∈ R Np×d
this section cite: []

Section: 14:
Compute attention: s j = -∥x e -p j ∥ 2 2 , α j = softmax(s j )
15:
x p ← Kp j=1 α j p j ; form x ctx = [Patch(x), x p ]16
H st ← Proj [E i n ∥ {E ϕc(τ ) c } c∈S ] ∈ R N ×N ′ ×d30
: H fused ← z + H st 31: // STF: adaptive filtering 32: Compute low-rank affinities: S s = ⟨H st W (s) q , E n W (s) k ⟩, S t = ⟨H st W (t) q , E t W (t) k ⟩ 33: S d ← ∆ δ=1 γ (δ) • ⟨H fused , Agg δ (H (t-δ) fused )⟩ 34: W ← softmax [S s , S t , S d ]W att /τ att 35:
H refined ← LayerNorm(H fused ⊙ W)36:
// HDA: hierarchical alignment 37:
K ← Topk j (-∥x e -p j ∥ 2 ), pk ← 1 k j∈K p j 38: A ← uv ⊤ , X r ← (1 Np p⊤ k ) ⊙ A 39: H aligned ← H refined + Proj(X r )40
this section cite: []

Section: References
Ref_id:b0 Title: Short-term spatio-temporal forecasting of photovoltaic power production Year: (2017)
Ref_id:b1 Title: A survey on spatio-temporal data analytics systems Year: (2022)
Ref_id:b2 Title: Learning the language of time series Year: (2024)
Ref_id:b3 Title: Spatio-temporal data mining: A survey of problems and methods Year: (2018)
Ref_id:b4 Title: Foundation models defining a new era in vision: a survey and outlook Year: (2025)
Ref_id:b5 Title: Adaptive graph convolutional recurrent network for traffic forecasting Year: (2020)
Ref_id:b6 Title: Adapts: Adapting univariate foundation models to probabilistic multivariate time series forecasting Year: (2025)
Ref_id:b7 Title: Fraug: Frequency domain augmentation for time series forecasting Year: (2023)
Ref_id:b8 Title: Learning with calibration: Exploring test-time computing of spatio-temporal forecasting Year: (2025)
Ref_id:b9 Title: A decoder-only foundation model for time-series forecasting Year: (2024)
Ref_id:b10 Title: St-gnns for weather prediction in south africa Year: (2022)
Ref_id:b11 Title: Multivariate time series forecasting via attention-based encoder-decoder framework Year: (2020)
Ref_id:b12 Title: Spatio-temporal data mining for climate data: Advances, challenges, and opportunities. Data mining and knowledge discovery for big data: Methodologies, challenge and opportunities Year: (2014)
Ref_id:b13 Title: Unraveling spatio-temporal foundation models via the pipeline lens: A comprehensive review Year: (2025)
Ref_id:b14 Title: Generative adversarial networks for spatio-temporal data: A survey Year: (2022)
Ref_id:b15 Title: Monash time series forecasting archive Year: (2021)
Ref_id:b16 Title: Spatio-temporal foundation models: Vision, challenges, and opportunities Year: (2025)
Ref_id:b17 Title: Attention based spatialtemporal graph convolutional networks for traffic flow forecasting Year: (2019)
Ref_id:b18 Title: Deep spatialtemporal 3d convolutional neural networks for traffic data forecasting Year: (2019)
Ref_id:b19 Title: Bigst: Linear complexity spatio-temporal graph neural network for traffic forecasting on large-scale road networks Year: (2024)
Ref_id:b20 Title: Geolocation representation from large language models are generic enhancers for spatio-temporal learning Year: (2025)
Ref_id:b21 Title: Spok: tokenizing geographic space for enhanced spatial reasoning in geoai Year: (2025)
Ref_id:b22 Title: Foundation models and intelligent decision-making: Progress, challenges, and perspectives. The Innovation Year: (2025)
Ref_id:b23 Title: Mapredrnn: multi-attention predictive rnn for traffic flow prediction by dynamic spatio-temporal data fusion Year: (2023)
Ref_id:b24 Title: Spatio-temporal self-supervised learning for traffic flow prediction Year: (2023)
Ref_id:b25 Title: Graph neural network for traffic forecasting: A survey. Expert systems with applications Year: (2022)
Ref_id:b26 Title: Spatio-temporal graph neural networks for predictive learning in urban computing: A survey Year: (2023)
Ref_id:b27 Title: A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection Year: (2024)
Ref_id:b28 Title: Semi-supervised classification with graph convolutional networks Year: (2016)
Ref_id:b29 Title: Stgat: Spatialtemporal graph attention networks for traffic flow forecasting Year: (2020)
Ref_id:b30 Title: St-fit: Inductive spatial-temporal forecasting with limited training data Year: (2025)
Ref_id:b31 Title: Diffusion convolutional recurrent neural network: Data-driven traffic forecasting Year: (2017)
Ref_id:b32 Title: Opencity: Open spatio-temporal foundation models for traffic prediction Year: (2024)
Ref_id:b33 Title: Geoman: Multi-level attention networks for geo-sensory time series prediction Year: (2018)
Ref_id:b34 Title: Airformer: Predicting nationwide air quality in china with transformers Year: (2023)
Ref_id:b35 Title: Foundation models for spatio-temporal data science: A tutorial and survey Year: (2025)
Ref_id:b36 Title: Spatio-temporal adaptive embedding makes vanilla transformer sota for traffic forecasting Year: (2023)
Ref_id:b37 Title: When do contrastive learning signals help spatio-temporal graph forecasting? Year: (2022)
Ref_id:b38 Title: Do we really need graph neural networks for traffic forecasting? arXiv preprint Year: (2023)
Ref_id:b39 Title: Unitime: A language-empowered unified model for cross-domain time series forecasting Year: (2024)
Ref_id:b40 Title: Histgnn: Hierarchical spatio-temporal graph neural network for weather forecasting Year: (2023)
Ref_id:b41 Title: A time series is worth 64 words: Long-term forecasting with transformers Year: ()
Ref_id:b42 Title: Neural discrete representation learning Year: (2017)
Ref_id:b43 Title: Gpt-4v(ision) system card Year: (2023)
Ref_id:b44 Title: Multi-scale finetuning for encoder-based time series foundation models Year: (2025)
Ref_id:b45 Title: Spatio-temporal wireless traffic prediction with recurrent neural network Year: (2018)
Ref_id:b46 Title: Spatio-temporal graph neural networks: A survey Year: (2023)
Ref_id:b47 Title: Spatial-temporal identity: A simple yet effective baseline for multivariate time series forecasting Year: (2022)
Ref_id:b48 Title: Decoupled dynamic spatial-temporal graph neural network for traffic forecasting Year: (2022)
Ref_id:b49 Title: Exploring progress in multivariate time series forecasting: Comprehensive benchmarking and heterogeneity analysis Year: (2024)
Ref_id:b50 Title: St-gat: A spatio-temporal graph attention network for accurate traffic speed prediction Year: (2022)
Ref_id:b51 Title: A survey on spatio-temporal series prediction with deep learning: taxonomy, applications, and future directions Year: (2024)
Ref_id:b52 Title: Evaluation of spatio-temporal forecasting methods in various smart city applications Year: (2018)
Ref_id:b53 Title: Neural discrete representation learning Year: (2017)
Ref_id:b54 Title: Unveiling the inflexibility of adaptive embedding in traffic forecasting Year: (2024)
Ref_id:b55 Title: Deep learning for multivariate time series imputation: A survey Year: (2024)
Ref_id:b56 Title: Airradar: Inferring nationwide air quality in china with deep neural networks Year: (2025)
Ref_id:b57 Title: Deep learning for spatio-temporal data mining: A survey Year: (2020)
Ref_id:b58 Title: Decomposable multiscale mixing for time series forecasting Year: (2024)
Ref_id:b59 Title: Towards a general time series forecasting model with unified representation and adaptive transfer Year: ()
Ref_id:b60 Title: Timexer: Empowering transformers for time series forecasting with exogenous variables Year: (2024)
Ref_id:b61 Title: Unified training of universal time series forecasting transformers Year: (2024)
Ref_id:b62 Title: Timesnet: Temporal 2d-variation modeling for general time series analysis Year: (2022)
Ref_id:b63 Title: Graph wavenet for deep spatial-temporal graph modeling Year: (2019)
Ref_id:b64 Title: Connecting the dots: Multivariate time series forecasting with graph neural networks Year: (2020)
Ref_id:b65 Title: Spatial-temporal transformer networks for traffic flow forecasting Year: (2020)
Ref_id:b66 Title: Intervention-aware forecasting: Breaking historical limits from a system perspective Year: (2024)
Ref_id:b67 Title: Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting Year: (2017)
Ref_id:b68 Title: Unist: A prompt-empowered universal model for urban spatio-temporal prediction Year: (2024)
Ref_id:b69 Title: Urbandit: A foundation model for open-world urban spatio-temporal learning Year: (2024)
Ref_id:b70 Title: Are transformers effective for time series forecasting Year: (2023)
Ref_id:b71 Title: Spatial-temporal graph attention networks: A deep learning approach for traffic forecasting Year: (2019)
Ref_id:b72 Title: Predicting carpark availability in singapore with crossdomain data: A new dataset and a data-driven approach Year: (2024)
Ref_id:b73 Title: Deep spatio-temporal residual networks for citywide crowd flows prediction Year: (2017)
Ref_id:b74 Title: Does cross-domain pre-training truly help time-series foundation models? Year: ()
Ref_id:b75 Title: T-gcn: A temporal graph convolutional network for traffic prediction Year: (2019)
Ref_id:b76 Title: Urban computing: concepts, methodologies, and applications Year: (2014)
Ref_id:b77 Title: Yongjing Ye, and Shihong Xia. Spatio-temporal gating-adjacency gcn for human motion prediction Year: (2022)
Ref_id:b78 Title: Informer: Beyond efficient transformer for long sequence time-series forecasting Year: (2021)
Ref_id:b79 Title: Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting Year: (2022)
Ref_id:b80 Title: Coms2t: A complementary spatiotemporal learning system for dataadaptive model evolution Year: (2025)
