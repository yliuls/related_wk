Title: CSBrain: A Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding
Abstract: Understanding and decoding human brain activity from electroencephalography (EEG) signals is a fundamental problem in neuroscience and artificial intelligence, with applications ranging from cognition and emotion recognition to clinical diagnosis and brain-computer interfaces. While recent EEG foundation models have made progress in generalized brain decoding by leveraging unified architectures and large-scale pretraining, they inherit a scale-agnostic dense modeling paradigm from NLP and vision. This design overlooks an intrinsic property of neural activity-cross-scale spatiotemporal structure. Different EEG task patterns span a broad range of temporal and spatial scales, from brief neural activations to slow-varying rhythms, and from localized cortical activations to large-scale distributed interactions. Ignoring this diversity may lead to suboptimal representations and weakened generalization ability. To address these limitations, we propose CSBrain, a Cross-scale Spatiotemporal Brain foundation model for generalized EEG decoding. CSBrain introduces two key components: (i) Cross-scale Spatiotemporal Tokenization (CST), which aggregates multi-scale features within localized temporal windows and anatomical brain regions into compact scale-aware token representations; and (ii) Structured Sparse attention (SSA), which models cross-window and cross-region dependencies for diverse decoding tasks, further enriching scale diversities while eliminating the spurious dependencies. CST and SSA are alternately stacked to progressively integrate cross-scale spatiotemporal dependencies. Extensive experiments across 11 representative EEG tasks and 16 datasets demonstrate that CSBrain consistently outperforms both task-specific models and strong foundation baselines. These results establish cross-scale modeling as a key inductive bias for generalized EEG decoding and highlight CSBrain as a robust backbone for future brain-AI research. The code and model are available at https://github.com/yuchen2199/CSBrain.

Section: Introduction
Understanding and decoding human brain activity is a long-standing ambition of neuroscience and artificial intelligence, with implications ranging from cognitive science to medical diagnostics and brain-computer interfaces (BCI) [1][2][3][4][5][6]. Among various neural recording modalities, electroencephalography (EEG) is particularly attractive due to its non-invasive nature, high temporal resolution, and affordability [7][8][9][10]. EEG signals are high-dimensional, multichannel time series that reflect the brain's dynamic electrical activity and are widely applied to abnormal identification [11,12], motor imagery [13,14], emotion analysis [15,16], sleep stage analysis [17,18], seizure detection [19,20], and various neurological disorders [21][22][23]. Over the past decade, a variety of task-specific deep learning models-spanning CNNs [24][25][26], RNNs [27][28][29], GNNs [30? -35] , and Transformers [36][37][38] -have been proposed for EEG decoding in the aforementioned applications. Although effective in narrow settings, these models are typically tightly coupled to specific datasets and task formats, limiting their generalizability and scalability [39,40].
Motivated by the paradigm shift in AI from task-specific to foundation models, recent efforts have explored brain foundation models [40][41][42][43][44][45][46] that aim to learn universal EEG representations across diverse brain decoding tasks via unified architectures and large-scale self-supervised pretraining. These models represent a significant step toward generalizable EEG decoding, and most adopt a pipeline transplanted from natural language processing and computer vision [47][48][49][50][51][52] (as shown in Figure 1(a)): EEG signals are first segmented into fixed-scale tokens, and then dense attention is applied across tokens to model dependencies. However, such scale-agnostic and dense modeling strategies fundamentally mismatch the intrinsic cross-scale spatiotemporal nature of EEG signals, which is critical for accurately capturing task-specific neural patterns. Specifically, different EEG decoding tasks inherently exhibit distinct spatiotemporal scales. From a temporal perspective, distinct EEG decoding tasks exhibit significant differences in neural dynamics [53,54]. For instance, motor and speech imagine tasks involve transient bursts of activity over short time windows [55,56], whereas sleep staging relies on slower oscillatory cycles across longer timescales [57]. From a spatial perspective, tasks differ in the activation patterns of brain regions due to the functional specialization of human brains [55,58]. Motor imagery typically involves localized or focal neural activations [59], while emotion recognition requires coordinated activity across widely distributed brain regions [60], reflecting longer-range spatial dependencies. Such variability in spatiotemporal scales across tasks places strong demands on token representations and modeling capacity. Most existing foundation models face three key limitations [61,62]: (1) Scale-agnostic tokenization: They rely on a one-time fixed-scale tokenization strategy, resulting in token representations that fail to accommodate the diverse neural patterns across tasks. These tokens often suffer from scale mismatch and semantic dilution, which undermines their effectiveness as the most fundamental modeling units. (2) Structure-agnostic dense attention: Due to the lack of spatiotemporal scale awareness in tokenization, existing foundation models rely on dense attention across all tokens to uncover task-specific neural patterns underlying noisy signals. However, indiscriminately applying attention to suboptimal tokens, without considering the intrinsic cross-scale structures, tends to introduce spurious dependencies due to the inherently high noise of EEG signals [14,63], ultimately degrading representation quality and increasing computational overhead. (3) Limited generalization: These limitations collectively hinder the generalization and adaptability of unified EEG foundation models across diverse BCI tasks with heterogeneous spatiotemporal scales.
To address these limitations, we argue that EEG foundation models must move beyond scale-agnostic, dense modeling paradigms and embrace cross-scale, structure-aware architectures that respect the intrinsic regional organisation and temporal dynamics of neural activity. In this paper, we propose CSBrain, a Cross-scale Spatiotemporal Brain foundation Model for generalized EEG decoding, as illustrated in Figure 1(b). Guided by neurophysiological priors, CSBrain decomposes EEG signals along spatial and temporal axes into distinct brain regions and windows, and captures crossscale dependencies both within and across these structures, yielding robust and highly adaptable representations. Specifically, we propose a Cross-scale Spatiotemporal Tokenization (CST) module, which captures EEG features at multiple temporal and spatial scales within localized regions. By integrating these patterns into compact token representations, CST enables adaptable alignment with diverse spatiotemporal requirements of EEG decoding tasks. Building upon these scale-aware tokens, we propose a Structured Sparse Attention (SSA) module to further enrich the diversity of modeling scales for EEG. SSA captures long-range dependencies across temporal windows and brain regions in a structured and efficient manner. By replacing costly dense attention, SSA reduces spurious dependencies and computational overhead, yielding more discriminative representations for noisy EEG signals. Finally, CST and SSA are alternately stacked and mutually reinforcing, progressively integrating cross-scale dependencies across both temporal and spatial dimensions. This hierarchical design enables CSBrain to build robust EEG representations that reflect the complex patterns of neural activity, thereby achieving superior generalization across EEG decoding tasks with inherently diverse spatiotemporal demands. CSBrain fundamentally differs from prior EEG foundation models by explicitly addressing and leveraging the intrinsic cross-scale spatiotemporal structure of neural activity, a property often overlooked by previous scale-agnostic dense modeling paradigms. Overall, our contribution can be summarized as follows: • We propose CSBrain, a Cross-scale Spatiotemporal Brain foundation model for generalized brain decoding, offering a cross-scale, structure-aware architecture that effectively captures multi-scale spatiotemporal neural patterns across diverse tasks.
• We introduce Cross-scale Spatiotemporal Tokenization (CST) to explicitly integrate multiscale neural patterns within temporal windows and brain regions into compact, scaleaware tokens. Additionally, Structured Sparse Attention (SSA) is designed to efficiently capture long-range dependencies across windows and regions, thereby enriching modeling scale diversity. Alternately stacked, CST and SSA progressively integrate cross-scale dependencies, enabling robust EEG representations across tasks.
• Extensive experiments across 11 representative EEG decoding tasks and 16 public datasets demonstrate CSBrain's superiority over both task-specific and foundation models. Further analysis reveals diverse cross-scale patterns across tasks, establishing cross-scale modeling as a key inductive bias for generalized EEG decoding.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b35', 'b36', 'b37', 'b38', 'b39', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b55', 'b56', 'b54', 'b57', 'b58', 'b59', 'b60', 'b61', 'b13', 'b62']

Section: Model Architecture
The overall framework of CSBrain is illustrated in Figure 2. The model first applies a standardized preprocessing module to extract initial feature representations from raw EEG signals. These features are then fed into the Cross-scale Spatiotemporal Tokenization (CST) module, which constructs robust cross-scale tokens by aggregating multi-resolution information within localized temporal windows and anatomically defined brain regions. Built upon these semantically enriched tokens, the Structured Sparse Attention (SSA) module expands the modeling scope, capturing long-range dependencies across time and brain regions while avoiding redundant interactions. CST and SSA are alternately stacked for L layers, progressively integrating cross-scale spatiotemporal dependencies. Finally, a lightweight task-specific head is attached to support various objectives, including masked reconstruction during pretraining and classification or regression for downstream tasks.
this section cite: []

Section: EEG Signal Preprocessing
Initially, we formalize the input EEG signals as E ∈ R |Cx|×T , where C x ⊆ C denotes the set of electrodes used, and T denotes the number of timestamps. C corresponds to the standardized international 10-20 system. To enhance signal quality and ensure feature consistency across datasets, we follow prior works [41,40] and apply a standardized signal preprocessing pipeline.
this section cite: ['b40', 'b39']

Section: Signal Standardization.
The raw signals E are processed with a band-pass filter to remove lowfrequency drifts and high-frequency noise, followed by a notch filter to eliminate power line interference. The signals are then resampled to a uniform rate of 200 Hz and scaled to 100 µV to normalize the amplitude range to [-1, 1]. To enable representation learning, the preprocessed EEG signal is temporally segmented into non-overlapping segments of length t, resulting in E p ∈ R C×n×t , where n = T t is the number of segments per electrode. Preliminary Feature Encoding. We extract both temporal and spectral features from each segment in E p . Local temporal dynamics are captured using 1D convolutions with normalization, while spectral information is obtained by applying a Fast Fourier Transform (FFT) followed by a fully connected layer to encode frequency energy distributions. The two feature types are fused to form a unified feature embedding. Finally, we add a learnable positional encoding [46], resulting in the initial EEG representation x (0) ∈ R C×n×d , where d denotes the feature embedding dimension.
this section cite: ['b45']

Section: Cross-scale Spatiotemporal Tokenization (CST)
Due to the inherent cross-scale spatiotemporal nature of EEG signals, informative neural patterns emerge at heterogeneous temporal and spatial resolutions. Previous EEG decoding methods adopting a fixed token granularity cannot effectively capture such patterns, limiting its generalizability across diverse BCI tasks. To address this, we propose Cross-scale Spatiotemporal Tokenization (CST), which encodes multi-resolution information within localized temporal windows and anatomically defined brain regions into unified token representations. These tokens form the basic modeling units for subsequent attention computation. In the following, we detail the two components of CST: Temporal Tokenization, which captures multi-scale temporal patterns within local time windows (i.e., intra-window), and Spatial Tokenization, which extracts region-specific features within anatomical brain regions (i.e., intra-region).
this section cite: []

Section: Temporal Tokenization.
To enable tokenization at various temporal resolutions, we introduce multiscale temporal convolution kernels C T = {Conv and output embedding dimension d k . Given the preprocessed EEG feature x (l-1) ∈ R C×n×d , for each location (i, j), where i ∈ {1, . . . , n} and j ∈ {1, . . . , C} denotes the temporal and electrode channel indices, respectively, we define a localized temporal window W (k) t (i) centered at i for the k-th scale with size s (k) t . Subsequently, the multi-scale temporal convolution is applied over these windows to extract temporal patterns at varying resolutions for each location. The process can be formulated as:
x(l) i,j = Concat {Conv (k) t (W (k) t (i))} K k=1 + Proj t (x (l-1) i,j ),(1)
where Proj t is a residual projection for dimension alignment. The outputs from different temporal kernels are concatenated to form a temporally cross-scale token x(l) i,j , which captures rich temporal dynamics within each channel. The resulted token representations could enhance model's robustness to signal noise and scalability to diverse decoding tasks with varying temporal patterns.
this section cite: []

Section: Spatial Tokenization.
Similar to Temporal Tokenization, we design multi-scale spatial convolution kernels C S = {Conv (k)  s } K k=1 , with each spatial kernel Conv (k)  s has a kernel size s (k) t and output embedding dimension d k . To encode spatial context across functionally related electrodes in each brain region at multiple scales, we utilize these convolution kernels to aggregate electrode features within anatomically defined brain regions. Specifically, given EEG feature x(l) ∈ R C×n×d after Temporal Tokenization, we first divide electrodes into a set of brain regions R = {R r } R r=1 based on the 10-20 system, where each region R r comprises spatially adjacent electrodes. For each electrode j ∈ R r , we define localized spatial neighborhoods W (k) s (j) ⊆ R r with size s (k) s within brain region for the k-th scale. Finally, the spatial tokenization is performed by applying multi-scale spatial convolution kernels over their corresponding neighborhoods:
x(l) i,j = Concat {Conv (k) s (W (k) s (j))} K k=1 + Proj s (x (l) i,j ),(2)
where Proj s is a residual projection for dimension alignment. The outputs from different spatial kernels are concatenated to form the final cross-scale token representation x(l) i,j , capturing localized spatial dependencies within each brain region.
x(l) i,j encapsulates spatiotemporal characteristics across diverse scales, serving as the final output of CST module and providing robust scale-aware features for downstream attention computation.
To balance representational capacity and computational efficiency, we allocate embedding dimensions across scales in an exponentially decaying scheme: d k ∝ 1  2 k , K k=1 d k = d, i.e., assigning higher dimensions to smaller kernels to retain fine-grained features, while allocating lower dimensions to larger kernels that summarize coarse context. This strategy aligns representational capacity with information density at each scale for balanced and efficient EEG representation.
this section cite: []

Section: Structured Sparse Attention (SSA)
Prior approaches typically apply dense attention uniformly across fixed-scale tokens. While effective in some cases, such indiscriminate attention can introduce spurious dependencies, reduce representational quality, and increases computational costs. Meanwhile, our CST provides structured, cross-scale token representations within localized temporal windows and brain regions, laying a robust representational space for further modeling. Building upon this, we propose Structured Sparse Attention (SSA), which efficiently captures long-range dependencies across temporal windows and spatial regions while avoiding redundant interactions. SSA complements CST along both temporal and spatial scales, and together they enable structured cross-scale modeling of EEG representations. SSA consists of two components: Inter-window Attention, responsible for capturing long-range temporal dependencies across local windows, and Inter-region Attention, which models spatial dependencies across distinct brain regions.
this section cite: []

Section: Inter-window Attention. Given the cross-scale tokens
x(l) i,j ∈ R C×n×d , we first perform a temporal grouping operation to form cross-window groups. Specifically, for each relative index g ∈ {1, . . . , w}, we collect all tokens that occupy the same position g in each window to form a temporal group G (g) t . Self-attention is then computed within each group to model structured long-range dependencies across time:
x(l,win) This design enables efficient and structured temporal interaction while avoiding redundancy.
i,j = Attn(G (g) t ) i,j + x(l) i,j .(3)
this section cite: []

Section: Inter-region Attention.
To ensure structured sparsity while maintaining coverage across all electrodes, we perform a spatial grouping operation to construct multiple spatial groups across brain regions. Unlike temporal windows (which have fixed lengths), the number of electrodes varies across brain regions. In regions with more electrodes, CST might not fully capture all information. By introducing the regional descriptor, we ensure complete information flow between regions. For each group, we sequentially sample a token x rep r from each region R r , and combine it with the average feature of R r to form a covering partition of the regional descriptor:
xr = xrep r + ϕ (P(R r )) , r = 1, . . . , R,(4)
where P(R r ) denotes the mean-pooled feature of region R r , and ϕ(•) is a learnable linear transformation. Notably, this ensures that each token is enriched by its region's context while preserving original information before inter-region attention and does not physically rearrange token positions. These descriptors are assembled into a spatial group G (g) s = {x 1 , x2 , . . . , xR }, over which self-attention is applied to capture structured dependencies across anatomical regions:
x(l,reg) i,j = Attn(G (g) s ) i,j + x(l,win) i,j .(5)
Finally, we apply layer normalization and a feedforward network with residual connection to refine the output:
x (l) i,j = FFN LN x(l,reg) i,j + x(l,reg) i,j .(6)
Together, SSA enables CSBrain to efficiently model long-range dependencies across both temporal and spatial dimensions by leveraging the robust cross-scale token layout provided by CST. This design progressively builds global context while preserving scale-awareness and mitigating spurious correlations in noisy signals, thereby enhancing generalization across diverse EEG decoding tasks.
this section cite: []

Section: Pretraining with Masked Autoencoding
To learn generalizable EEG representations, we adopt a self-supervised pretraining strategy based on masked autoencoding [51,64], as shown in Figure 3. This process encourages the model to capture meaningful spatiotemporal dependencies from unlabeled EEG signals, laying the foundation for effective downstream transfer.
this section cite: ['b50', 'b63']

Section: Masking Strategy.
Given a preprocessed EEG signal E p ∈ R C×n×t , we randomly mask a fixed ratio r ∈ (0, 1) of segments along the temporal axis using a Bernoulli sampling scheme. This results in two subsets: visible segments E v ∈ R C×nv×t and masked segments E m ∈ R C×nm×t , where n v + n m = n and nm n = r. The masked EEG sequence is encoded by alternately stacked CST and SSA modules, which jointly model neural dependencies across multiple spatiotemporal scales. Masked segments are represented by learnable embeddings and integrated alongside visible tokens during encoding.
Reconstruction Objective. The reconstruction head consists of a lightweight fully connected layer, which takes as input the encoded visible tokens and learnable masked embeddings, and projects them into the original EEG signal space. Let Êm ∈ R C×nm×t denote the predicted EEG segments and E m the corresponding ground truth. The reconstruction loss is defined as the mean squared error (MSE) over all masked positions:
L rec = ∥ Êm -E m ∥ 2 .(7)
This masked autoencoding objective drives the model to recover meaningful spatiotemporal dependencies from context, yielding robust and transferable EEG representations.  [46,65]. We apply standard preprocessing steps: signals are band-pass filtered between 0.3-75 Hz to remove low-and highfrequency noise, and a notch filter at 60 Hz is used to eliminate power line interference. All signals are resampled to 200 Hz and segmented into 30-second EEG samples. The amplitude is normalized to 100 µV to ensure the signal range falls within [-1, 1], consistent with prior work [41,46]. After cleaning, a total of 1,109,545 EEG segments (over 9,000 hours) are used for pre-training.
Pre-training Settings. We adopt a masked autoencoding objective, with a masking ratio of 50% applied to randomly sampled EEG patches. Pre-training is conducted using Python 3.11.11 and PyTorch 2.5.1 with CUDA 12.4. The model is trained with the AdamW optimizer, a learning rate of 5e-4, weight decay of 5e-2, and cosine annealing learning rate scheduling. We use a batch size of 128 and train for 40 epochs on 4 NVIDIA A100 GPUs, with a total pre-training time of approximately 101 hours. More details can be found in the supplementary material.
this section cite: ['b45', 'b64', 'b40', 'b45']

Section: Experimental Setup of Downstream BCI Tasks
Tasks & Datasets. To comprehensively evaluate the generalizability of our model, we conduct experiments on 11 representative BCI tasks spanning 16 publicly available EEG datasets, as summarized in Table 1. These tasks include Motor Imagery Classification (BCIC-IV-2a, PhysioNet-MI, SHU-MI), Emotion Recognition (FACED, SEED-V), Seizure Detection (CHB-MIT, Siena), Sleep Staging (ISRUC, HMC), Imagined Speech Classification (BCIC2020-3), Vigilance Estimation (SEED-VIG), Mental Stress Detection (MentalArithmetic), Mental Disorder Diagnosis (Mumtaz2016), Event Type Classification (TUEV), Abnormal Detection (TUAB), and Slowing Event Classification (TUSL).
In all experiments, we strictly follow the training, validation, and test splits to ensure fair and consistent evaluation. To the best of our knowledge, this is among the most comprehensive evaluations conducted on EEG foundation models to date. See supplementary for details.
Baselines & Metrics. We extensively compare CSBrain with two major categories of baselines:
(1) representative task-specific EEG decoding models, including EEGNet [24], EEGConformer (Conformer) [36], SPaRCNet [25], ContraWR [26], CNN-Transformer (C-Trans) [38], FFCL [27], and ST-Transformer (ST-Trans) [37]; and (2) recent EEG foundation models, including BIOT [40], LaBraM [41], and CBraMod [46]. Following prior works [40,41,46], we report Balanced Accuracy, Cohen's Kappa, and Weighted F1 for multiclass classification tasks; Balanced Accuracy, AUC-PR, and AUROC for binary classification; and Pearson correlation, R2 score, and RMSE for regression tasks. See supplementary for details.
this section cite: ['b23', 'b35', 'b24', 'b25', 'b37', 'b26', 'b36', 'b39', 'b40', 'b45', 'b39', 'b40', 'b45']

Section: Experimental Results

this section cite: []

Section: Performance Comparison.
To fairly and comprehensively evaluate the effectiveness of CSBrain, we benchmark it against 10 strong baselines across 11 representative BCI tasks covering 16 public EEG datasets, as summarized in Table 2. All results are averaged over five runs with different random seeds. Due to space limitations, we report two representative metrics per dataset in the main paper: Balanced Accuracy (B-Acc) and Weighted F1 (F1-W) for multi-class classification, B-Acc and AUROC for binary classification, and Pearson correlation (Corr.) and R2 score for regression tasks. Full results, including standard deviations and detailed analyses, are provided in the supplementary material.
As shown in Table 2, we highlight two key observations: (1) CSBrain consistently achieves state-ofthe-art performance across nearly all tasks and metrics. In particular, it significantly outperforms all baselines on datasets such as BCIC-IV-2a, BCIC2020-3, Siena, and TUSL. Even in cases where CSBrain ranks second on certain metrics, the performance gap to the best-performing method remains marginal. From the macro-average in the last row of Table 2, CSBrain achieves the highest overall score across all tasks, outperforming strong foundation models such as CbraMod, LaBraM, and BIOT by 3.35%, 3.98%, and 7.73%, respectively. These results demonstrate CSBrain's strong generalization and adaptability to diverse EEG decoding scenarios with varying spatiotemporal demands. (2) Foundation models outperform task-specific models by a significant margin. Across all 16 datasets, EEG foundation models-including BIOT, LaBraM, CbraMod, and CSBrain-consistently outperform task-specific models. This also validates the advantage of unified model architectures trained with large-scale pretraining, which can extract generalizable neural representations across heterogeneous EEG domains.
this section cite: []

Section: Tokenization Comparison.
To investigate how spatiotemporal scale affects EEG representation from the perspective of token construction, we vary the number of kernels K in the CST module to control token granularity. Specifically, we compare single-scale (K = 1), dual-scale (K = 2), and full crossscale (K = 3) configurations. Additionally, to evaluate the effect of stacking, we compare our full design (with CST applied before each SSA module) against a variant where cross-scale tokenization (K = 3) is applied only once before the first SSA. For efficiency, all variants are pre-trained and fine-tuned using 30% of the training data. As shown in Fig. 4, our SSA (K = 3, stacked) consistently achieves the best performance across four representative tasks: emotion recognition, motor imagery,  event classification, and imagined speech. In contrast, the single-scale variant (K = 1) performs the worst. Notably, on the motor imagery task, SSA outperforms the single-scale counterpart by 10.8%, 20.7%, and 12.1% on three evaluation metrics, respectively. Moreover, we observe that alternately stacking SSA modules throughout the network consistently outperforms applying SSA only once. This is because a single application of SSA before attention computation may fail to fully capture the broader cross-scale dependencies. These results highlight the importance of the proposed SSA for capturing diverse EEG dynamics across tasks.
this section cite: []

Section: Attention Mechanism Comparison.
Furthermore, we compare different types of attention mechanisms to assess their impact on EEG decoding performance and computational efficiency. (1) Dense Attention performs full self-attention over all tokens with quadratic complexity O(N 2 ), where N = T × C. While theoretically expressive, it introduces spurious dependencies, weakens representation quality in noisy EEG scenarios, and incurs substantial computational overhead. (2) Criss-cross Attention, which restricts attention to shared temporal or spatial indices, yielding approximate complexity O(N 1.5 ); and (3) Our Structured Sparse Attention (SSA) limits attention to grouped tokens across temporal windows and brain regions, achieving linear complexity O(N • k) with small k ≪ √ N . As shown in Figure 5, SSA consistently achieves the best performance across Balanced Accuracy, Cohen's Kappa, and Weighted F1. For instance, on the TUEV dataset, SSA improves Balanced Accuracy by +8.1% and Weighted F1 by +5.3% compared to Criss-cross Attention, and maintains low computational cost. These results highlight SSA as an effective and scalable inductive bias for EEG modeling.
this section cite: []

Section: Topography Visualization.
To further examine the spatial dynamics captured by CSBrain, we visualize activation topographies across different EEG decoding tasks. Specifically, we apply Gradientweighted Class Activation Mapping (Grad-CAM) [80] to compute the contribution of each EEG channel to the model's predictions, as shown in Figure 6. Different tasks elicit distinct activation patterns and scales. Vigilance states primarily activate the temporal and occipital lobes, indicating continuous engagement of auditory and visual systems. Motor imagery evokes highly localized activations over the contralateral motor cortex, reflecting ERD/ERS (event-related desynchronization / synchronization) phenomenon [81]. In contrast, emotion recognition and imagined speech exhibit broad, distributed activations. Emotion tasks elicit significant activation across the frontal and occipital lobes, consistent with the findings in [32]. Speech imagery evokes significant activation in frontal and temporal lobes [82], which are critically involved in imagined speech processing [83]. These observations further validate that different EEG decoding tasks exhibit distinct activation regions and scales, reinforcing the necessity of cross-scale modeling for EEG foundation models. By explicitly modeling cross-scale structure, CSBrain effectively adapts to task-specific neural patterns across diverse brain decoding scenarios.
this section cite: ['b79', 'b80', 'b31', 'b81', 'b82']

Section: Vigilance Estimation (SEED-VIG, vigilance)

this section cite: []

Section: Emotion Recognition
(SEED-V, sad)
this section cite: []

Section: Imagined Speech Classification (BCIC2020-3, 'hello')
Motor Imagery Classification (BCI-IV-2a, left hand)
this section cite: []

Section: Conclusion
In this work, we demonstrate that modeling the cross-scale spatiotemporal structure of EEG signals is essential for building generalizable brain foundation models. Through a unified architecture composed of Cross-scale Spatiotemporal Tokenization and Structured Sparse Attention, our proposed CSBrain effectively captures neural dependencies spanning multiple spatiotemporal scales. Experimental results across 11 EEG decoding tasks and 16 datasets validate the effectiveness of CSBrain. These findings establish cross-scale spatiotemporal modeling as a critical inductive bias for robust, scalable, and physiologically-aligned EEG representation learning. We believe this work lays a solid foundation for future advancements in brain-AI integration and generalized neural decoding.
this section cite: []

Section: References
Ref_id:b0 Title: Motor imagery and direct brain-computer communication Year: (2001)
Ref_id:b1 Title: Decoding mental states from brain activity in humans Year: (2006)
Ref_id:b2 Title: Beyond the brain-computer interface: Decoding brain activity as a tool to understand neuronal mechanisms subtending cognition and behavior Year: (2022)
Ref_id:b3 Title: Explicit brain functional alignment for cross-subject visual decoding from limited fmri data Year: (2025)
Ref_id:b4 Title: Adabrain-bench: Benchmarking brain foundation models for brain Year: (2025)
Ref_id:b5 Title: Unimind: Unleashing the power of llms for unified multi-task brain decoding Year: (2025)
Ref_id:b6 Title: Handbook of clinical neurology Year: (2020)
Ref_id:b7 Title: A braincomputer interface that evokes tactile sensations improves robotic arm control Year: (2021)
Ref_id:b8 Title: Combining vr with electroencephalography as a frontier of brain-computer interfaces Year: ()
Ref_id:b9 Title: Neuro-3d: Towards 3d visual decoding from eeg signals Year: (2025)
Ref_id:b10 Title: Chrononet: A deep recurrent neural network for abnormal eeg identification Year: (2019)
Ref_id:b11 Title: Automatic signal abnormality detection using time-frequency features and machine learning: A newborn eeg seizure case study Year: (2016)
Ref_id:b12 Title: Deep learning for motor imagery eeg-based classification: A review Year: (2021)
Ref_id:b13 Title: A spatial filter temporal graph convolutional network for decoding motor imagery eeg signals Year: (2024)
Ref_id:b14 Title: Eeg-based emotion recognition: a stateof-the-art review of current trends and opportunities Year: (2020)
Ref_id:b15 Title: Contrastive learning of eeg representation of brain area for emotion recognition Year: (2025)
Ref_id:b16 Title: Self-supervised eeg representation learning for automatic sleep staging Year: (2023)
Ref_id:b17 Title: Generalizable sleep staging via multi-level domain alignment Year: (2024)
Ref_id:b18 Title: A hybrid deep learning approach for epileptic seizure detection in eeg signals Year: (2023)
Ref_id:b19 Title: Eeg datasets for seizure detection and prediction-a review Year: (2023)
Ref_id:b20 Title: Automatic and efficient framework for identifying multiple neurological disorders from eeg signals Year: (2023)
Ref_id:b21 Title: A review of graph theory-based diagnosis of neurological disorders based on eeg and mri Year: (2024)
Ref_id:b22 Title: Medformer: A multi-granularity patching transformer for medical time-series classification Year: (2024)
Ref_id:b23 Title: Eegnet: a compact convolutional neural network for eeg-based brain-computer interfaces Year: (2018)
Ref_id:b24 Title: Development of expert-level classification of seizures and rhythmic and periodic patterns during eeg interpretation Year: (2023)
Ref_id:b25 Title: Self-supervised electroencephalogram representation learning for automatic sleep staging: Model development and evaluation study Year: (2023)
Ref_id:b26 Title: Motor imagery eeg classification algorithm based on cnn-lstm feature fusion network Year: (2022)
Ref_id:b27 Title: Emotion recognition based on eeg using lstm recurrent neural network Year: (2017)
Ref_id:b28 Title: An efficient lstm network for emotion recognition from multichannel eeg signals Year: (2020)
Ref_id:b29 Title: Eeg-gnn: Graph neural networks for classification of electroencephalogram (eeg) signals Year: (2021)
Ref_id:b30 Title: Graphsleepnet: Adaptive spatial-temporal graph convolutional networks for sleep stage classification Year: (2020)
Ref_id:b31 Title: Eeg-based emotion recognition using regularized graph neural networks Year: (2020)
Ref_id:b32 Title: Graph-generative neural network for eeg-based epileptic seizure detection via discovery of dynamic brain functional connectivity Year: (2022)
Ref_id:b33 Title: Graph neural network-based eeg classification: A survey Year: (2024)
Ref_id:b34 Title: Graph neural networks in eeg-based emotion recognition: a survey Year: (2024)
Ref_id:b35 Title: Eeg conformer: Convolutional transformer for eeg decoding and visualization Year: (2022)
Ref_id:b36 Title: Transformer-based spatial-temporal feature learning for eeg decoding Year: (2021)
Ref_id:b37 Title: Transformer convolutional neural networks for automated artifact detection in scalp eeg Year: (2022)
Ref_id:b38 Title: Toward reliable signals decoding for electroencephalogram: A benchmark study to eegnex Year: (2024)
Ref_id:b39 Title: Biot: Biosignal transformer for cross-data learning in the wild Year: (2023)
Ref_id:b40 Title: Large brain model for learning generic representations with tremendous eeg data in bci Year: (2024)
Ref_id:b41 Title: Eegpt: Pretrained transformer for universal and reliable representation of eeg signals Year: (2024)
Ref_id:b42 Title: Neuro-gpt: Towards a foundation model for eeg Year: (2024)
Ref_id:b43 Title: Brainwave: A brain signal foundation model for clinical applications Year: (2024)
Ref_id:b44 Title: A simple review of eeg foundation models: Datasets, advancements and future perspectives Year: (2025)
Ref_id:b45 Title: CBramod: A criss-cross brain foundation model for EEG decoding Year: (2025)
Ref_id:b46 Title: Attention is all you need Year: (2017)
Ref_id:b47 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b48 Title: Self-supervised learning: Generative or contrastive Year: (2021)
Ref_id:b49 Title: Bendr: Using transformers and a contrastive self-supervised learning task to learn from massive amounts of eeg data Year: (2021)
Ref_id:b50 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b51 Title: Cswin transformer: A general vision transformer backbone with cross-shaped windows Year: (2022)
Ref_id:b52 Title: Principles of neural science Year: (2000)
Ref_id:b53 Title: Rhythms of the Brain Year: (2006)
Ref_id:b54 Title: A weighted and directed interareal connectivity matrix for macaque cerebral cortex Year: (2014)
Ref_id:b55 Title: 2020 international brain-computer interface competition: A review Year: (2022)
Ref_id:b56 Title: The sleep slow oscillation as a traveling wave Year: (2004)
Ref_id:b57 Title: The wu-minn human connectome project: an overview Year: (2013)
Ref_id:b58 Title: Cognitive motor processes: the role of motor imagery in the study of motor representations Year: (2009)
Ref_id:b59 Title: Eeg based emotion recognition: A tutorial and review Year: (2022)
Ref_id:b60 Title: Going deeper with convolutions Year: (2015)
Ref_id:b61 Title: Transformers for eeg-based emotion recognition: A hierarchical spatial information learning model Year: (2022)
Ref_id:b62 Title: Eeg artifact removal-state-of-the-art and guidelines Year: (2015)
Ref_id:b63 Title: Simmim: A simple framework for masked image modeling Year: (2022)
Ref_id:b64 Title: NeuroLM: A universal multi-task foundation model for bridging the gap between language and EEG signals Year: (2025)
Ref_id:b65 Title: Bci competition 2008-graz data set a. Institute for knowledge discovery (laboratory of brain-computer interfaces Year: (2008)
Ref_id:b66 Title: Bci2000: a general-purpose brain-computer interface (bci) system Year: (2004)
Ref_id:b67 Title: A large eeg dataset for studying cross-session variability in motor imagery brain-computer interface Year: (2022)
Ref_id:b68 Title: Gnn4eeg: A benchmark and toolkit for electroencephalography classification with graph neural network Year: (2024)
Ref_id:b69 Title: Comparing recognition performance and robustness of multimodal deep learning models for multimodal emotion recognition Year: (2021)
Ref_id:b70 Title: Physiobank, physiotoolkit, and physionet: components of a new research resource for complex physiologic signals Year: (2000)
Ref_id:b71 Title: Application of machine learning to epileptic seizure onset detection and treatment Year: (2009)
Ref_id:b72 Title: Eeg synchronization analysis for seizure prediction: A study on data of noninvasive recordings Year: (2020)
Ref_id:b73 Title: Isruc-sleep: A comprehensive public dataset for sleep researchers Year: (2016)
Ref_id:b74 Title: Inter-database validation of a deep learning approach for automatic sleep scoring Year: (2021)
Ref_id:b75 Title: A multimodal approach to estimating vigilance using eeg and forehead eog Year: (2017)
Ref_id:b76 Title: Electroencephalograms during mental arithmetic task performance Year: (2019)
Ref_id:b77 Title: MDD Patients and Healthy Controls EEG Data (New) Year: (2016)
Ref_id:b78 Title: The temple university hospital eeg data corpus Year: (2016)
Ref_id:b79 Title: Grad-cam: Visual explanations from deep networks via gradient-based localization Year: (2017)
Ref_id:b80 Title: Event-related cortical desynchronization detected by power measurements of scalp eeg Year: (1977)
Ref_id:b81 Title: Speech production: Wernicke, broca and beyond Year: (2002)
Ref_id:b82 Title: Revealing the spatiotemporal brain dynamics of covert speech compared with overt speech: A simultaneous eeg-fmri study Year: (2024)
Ref_id:b83 Title: Fft-based deep feature learning method for eeg classification Year: (2021)
Ref_id:b84 Title: Epileptic seizure classification of eeg time-series using rational discrete short-time fourier transform Year: (2014)
Ref_id:b85 Title: An eeg based real-time epilepsy seizure detection approach using discrete wavelet transform and machine learning methods Year: (2022)
Ref_id:b86 Title: Optimizing spatial filters for robust eeg single-trial analysis Year: (2007)
Ref_id:b87 Title: Multiclass braincomputer interface classification by riemannian geometry Year: (2011)
Ref_id:b88 Title: Differential evolution algorithm as a tool for optimal feature subset selection in motor imagery eeg Year: (2017)
Ref_id:b89 Title: Multiclass support vector machines for eeg-signals classification Year: (2007)
Ref_id:b90 Title: Comparison of signal decomposition methods in classification of eeg signals for motor-imagery bci system Year: (2017)
Ref_id:b91 Title: Logistic regression for single trial eeg classification Year: (2006)
Ref_id:b92 Title: Deep learning with convolutional neural networks for eeg decoding and visualization Year: (2017)
Ref_id:b93 Title: Lmda-net: A lightweight multi-dimensional attention network for general eeg-based brain-computer interfaces and interpretability Year: (2023)
Ref_id:b94 Title: Lstm-based eeg classification in motor imagery tasks Year: (2018)
Ref_id:b95 Title: An efficient lstm network for emotion recognition from multichannel eeg signals Year: (2020)
Ref_id:b96 Title: Sleeptransformer: Automatic sleep staging with interpretability and uncertainty quantification Year: (2022)
Ref_id:b97 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b98 Title: Language models are few-shot learners Year: (2020)
Ref_id:b99 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b100 Title: Visual instruction tuning Year: (2023)
Ref_id:b101 Title: FlexCAD: Unified and versatile controllable CAD generation with fine-tuned large language models Year: (2025)
Ref_id:b102 Title: Learning from observer gaze: Zero-shot attention prediction oriented by human-object interaction recognition Year: (2024)
Ref_id:b103 Title: Skeleton-incontext: Unified skeleton sequence modeling with in-context learning Year: (2024)
Ref_id:b104 Title: Drivinggen: Efficient safety-critical driving video generation with latent diffusion models Year: (2024)
Ref_id:b105 Title: Logic unseen: Revealing the logical blindspots of vision-language models Year: (2025)
Ref_id:b106 Title: Xiaobo Xia, and Chao Gou. Where, what, why: Towards explainable driver attention prediction Year: (2025)
Ref_id:b107 Title: Brainbert: Self-supervised representation learning for intracranial recordings Year: (2023)
Ref_id:b108 Title: Foundation model for intracranial neural signal Year: (2023)
Ref_id:b109 Title: Eeg2rep: enhancing self-supervised eeg representation through informative masked inputs Year: (2024)
Ref_id:b110 Title: Ten-twenty electrode system of the international federation Year: (1958)
Ref_id:b111 Title: Ten percent electrode system for topographic studies of spontaneous and evoked eeg activities Year: (1985)
Ref_id:b112 Title: Guideline thirteen: guidelines for standard electrode position nomenclature Year: (1994)
Ref_id:b113 Title: Three-dimensional probabilistic anatomical cranio-cerebral correlation via the international 10-20 system oriented for transcranial functional brain mapping Year: (2004)
Ref_id:b114 Title: Electroencephalography: basic principles, clinical applications, and related fields Year: (2006)
Ref_id:b115 Title: The american academy of sleep medicine manual for the scoring of sleep and associated events Year: (2007)
Ref_id:b116 Title: Automatic sleep staging of eeg signals: recent development, challenges, and future directions Year: (2022)
Ref_id:b117 Title: NeuroLM: A universal multi-task foundation model for bridging the gap between language and EEG signals Year: (2025)
Ref_id:b118 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b119 Title: Visualizing data using t-sne Year: (2008)
