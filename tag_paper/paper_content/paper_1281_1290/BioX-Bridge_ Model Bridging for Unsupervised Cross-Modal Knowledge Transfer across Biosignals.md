Title: BIOX-BRIDGE: MODEL BRIDGING FOR UNSUPERVISED CROSS-MODAL KNOWLEDGE TRANSFER ACROSS BIOSIGNALS
Abstract: Biosignals offer valuable insights into the physiological states of the human body. Although biosignal modalities differ in functionality, signal fidelity, sensor comfort, and cost, they are often intercorrelated, reflecting the holistic and interconnected nature of human physiology. This opens up the possibility of performing the same tasks using alternative biosignal modalities, thereby improving the accessibility, usability, and adaptability of health monitoring systems. However, the limited availability of large labeled datasets presents challenges for training models tailored to specific tasks and modalities of interest. Unsupervised cross-modal knowledge transfer offers a promising solution by leveraging knowledge from an existing modality to support model training for a new modality. Existing methods are typically based on knowledge distillation, which requires running a teacher model alongside student model training, resulting in high computational and memory overhead. This challenge is further exacerbated by the recent development of foundation models that demonstrate superior performance and generalization across tasks at the cost of large model sizes. To this end, we explore a new framework for unsupervised cross-modal knowledge transfer of biosignals by training a lightweight bridge network to align the intermediate representations and enable information flow between foundation models and across modalities. Specifically, we introduce an efficient strategy for selecting alignment positions where the bridge should be constructed, along with a flexible prototype network as the bridge architecture. Extensive experiments across multiple biosignal modalities, tasks, and datasets show that BioX-Bridge reduces the number of trainable parameters by 88-99% while maintaining or even improving transfer performance compared to state-of-the-art methods. Our code is available at: https://github.com/chenqi-li/BioX-Bridge.

Section: INTRODUCTION
Biosignals, such as electrocardiogram (ECG), electroencephalogram (EEG), and photoplethysmography (PPG), provide critical insights into the underlying physiological states of individuals. They are essential tools in modern healthcare and have often been considered the gold standard for diagnostics (Rosenberg & Van Hout, 2013;Stracina et al., 2022). In the past decade, the advancement of artificial intelligence (AI) has enabled remarkable capabilities in automated diagnostics and monitoring, such as stress assessment (Mentis et al., 2024), sleep stage classification (Mostafa et al., 2019), and arrhythmia detection (Parvaneh et al., 2019). However, many biosignal sensors are not suitable for use outside clinical settings due to factors such as user discomfort, high manufacturing costs, and excessive power consumption.
A promising direction is to harness the correlations between different biosignal modalities and perform the same tasks using alternative modalities, making health monitoring systems more accessible, practical, and flexible (Wang et al., 2023;Yang et al., 2023). For example, being able to perform the same tasks using single-lead PPG data from a wearable smartwatch, instead of relying on 12-lead ECGs, would greatly reduce hardware complexity and cost, while enabling continuous, user-friendly monitoring in everyday environments. Unfortunately, training such models requires large-scale labeled datasets, which are often difficult to obtain in biosignal applications due to the high cost and domain-specific expertise required for data collection and annotation. This highlights the need for effective knowledge transfer between biosignal modalities, leveraging models trained in old or well-established modalities to support the development of models for new or underrepresented biosignal modalities.
Unsupervised cross-modal knowledge transfer stands out as a practical solution to address the aforementioned needs. Existing methods can be divided into two categories: data translation and knowledge distillation. As illustrated in Figure 1 (a), data translation directly translates data from the new modality to the old modality, enabling the direct reuse of existing models from the old modality (Sarkar & Etemad, 2021). However, the exploration of data translation has been limited to a certain pair of modalities, such as PPG and ECG. Figure 1(b) illustrates knowledge distillation, which seeks to train a student model for the new modality to mimic the output of a pre-trained teacher model from the old modality (Abbaspourazad et al., 2024b;Zhang et al., 2024). The distillation process is memory intensive, as it requires forward inference with both the student and teacher models, in addition to backpropagation with the student model. The computational burden is further exacerbated by the emergence of large-scale biosignal foundation models (Coppola et al., 2024;Jiang et al., 2024;Pillai et al., 2025), which are mostly trained on specific modalities and have demonstrated exceptional performance across a wide range of tasks. Although these models offer tremendous performance gains, their use in cross-modal knowledge transfer is hindered by their size, which makes traditional knowledge distillation solutions computationally prohibitive for users without access to high-end GPUs. For example, distilling knowledge from PaPaGei, a PPG-based foundation model (Pillai et al., 2025), to the ECG-FM student model (McKeen et al., 2024) on the WESAD dataset (Schmidt et al., 2018), with a batch size of eight, requires more than 32GB of VRAM. Furthermore, because of data sharing regulations and privacy concerns, such a distillation process often needs to be performed locally where the data resides, under low-resource conditions. These constraints call for the development of an effective and efficient cross-modal transfer framework that can fully leverage the representation capability and embedded knowledge of the foundation models.
To this end, we propose BioX-Bridge, a new framework for unsupervised cross-modal knowledge transfer via model bridging, as illustrated in Figure 1(c). The core idea is to construct a bridge that projects intermediate representations from one biosignal model to another, leveraging the powerful representational capability and the rich embedded knowledge of foundation models. 1 The framework comprises two key components: bridge position selection and bridge architecture design. Specifically, we introduce an efficient two-stage strategy for selecting optimal input and output positions by evaluating the quality and similarity of intermediate representations between two biosignal models.
To enable effective projection between high-dimensional spaces, we design a prototype network composed of a learnable prototype set and a low-rank approximation module to compute aggregation weights. Notably, only the bridge network requires training to enable interoperability between models of different modalities. We evaluate the effectiveness of BioX-Bridge in three biosignal datasets involving different modalities, demonstrating superior efficiency compared to existing methods. Extensive ablation studies further confirm the robustness of the proposed framework under various conditions. Our contributions can be summarized as follows:
• We propose BioX-Bridge, a novel unsupervised model bridging framework that enables crossmodal knowledge transfer through information flow between biosignal models.
• We introduce key components to support the framework, including an efficient two-stage strategy for selecting bridge positions and a prototype network with low-rank approximation for effective high-dimensional projection.
• We demonstrate the efficiency of BioX-Bridge through experiments on three biosignal datasets, four modalities, and six transfer directions, demonstrating robustness through comprehensive ablation studies.
this section cite: ['b34', 'b39', 'b27', 'b30', 'b32', 'b43', 'b45', 'b36', 'b47', 'b8', 'b19', 'b33', 'b33', 'b26', 'b37']

Section: RELATED WORKS
Unsupervised Cross-modal Knowledge Transfer Existing methods can be divided into two categories: knowledge distillation and data translation.
Knowledge distillation was introduced as a model compression technique, where a smaller student model learns to mimic a larger and high-performing teacher model by matching its output distributions (Hinton et al., 2015). The concept has since been extended to cross-modal knowledge transfer. Early efforts focused on computer vision applications across a variety of sensor modalities, such as vision to depth images (Garcia et al., 2018;Gupta et al., 2016;Hoffman et al., 2016;Tian et al., 2020), to radio frequency heatmaps (Zhao et al., 2018), and to sound (Aytar et al., 2016;Xue et al., 2021). The core idea is to leverage unlabeled but semantically aligned data pairs to bridge the modality gap and transfer relevant knowledge to the corresponding tasks (Gou et al., 2021;Moslemi et al., 2024). Recent efforts have also investigated cross-modal knowledge distillation for biosignals. For example, Brant-X (Zhang et al., 2024) introduced a unified biosignal alignment framework that transfers knowledge from EEG to other biosignal modalities through a two-level semantic alignment strategy, such that the student model can provide complementary representations to the teacher model and improve downstream task performance. In another work (Abbaspourazad et al., 2024b), the distillation of knowledge from PPG to accelerometer signals was used to accurately predict physiological states such as heart rate. However, the aforementioned methods require training a full-size student model from scratch, which becomes increasingly impractical as model sizes grow, especially for resource-constrained settings.
Data translation aims to achieve unsupervised cross-modal knowledge transfer by directly translating raw data from one modality to another. Generative adversarial networks (GAN) (Goodfellow et al., 2020) and their variants (Mirza & Osindero, 2014;Zhu et al., 2017) have been widely adopted for modality translation tasks in the visual and signal processing domains (Duan et al., 2021;Sikka et al., 2021;Yang et al., 2020). A recent work (Wang et al., 2023) leveraged knowledge graphs to learn transformations between independently trained foundation models for proteins, drugs, and text. Nevertheless, reliance on structured knowledge graphs limits their applicability to biosignal scenarios, where such structured relationships are scarce or nonexistent. In the biosignal domain, cross-modal translation efforts have been largely limited to translation from PPG to ECG (Sarkar & Etemad, 2021;Zhu et al., 2021). Extending such translation to other modalities, such as EEG to ECG, remains largely underexplored.
this section cite: ['b17', 'b12', 'b15', 'b18', 'b41', 'b49', 'b4', 'b44', 'b14', 'b29', 'b47', 'b13', 'b28', 'b50', 'b11', 'b38', 'b46', 'b43', 'b36', 'b51']

Section: Biosignals Foundation Models
Inspired by the recent success of large-scale pre-training in natural language processing (Achiam et al., 2023) and computer vision (Dosovitskiy et al., 2020), the development of foundation models for biosignals has garnered much interest (Han et al., 2024;Lai et al., 2025). Through large-scale self-supervised training on public and private biosignal datasets, several biosignal foundation models have been developed to capture rich and transferable representations, enabling more robust and efficient downstream adaptation. These models span a variety of modalities, including EEG (Chen et al., 2025a;2024;Cui et al., 2023;Jiang et al., 2024;Wang et al., 2024), ECG (Coppola et al., 2024;Li et al., 2024;McKeen et al., 2024), PPG (Chen et al., 2025b;Pillai et al., 2025;Saha et al., 2025), accelerometer (Abbaspourazad et al., 2024b), and general-purpose biosignal models (Yang et al., 2023). In addition to unimodal models, recent work has explored multimodal foundation models for various applications such as health monitoring (Abbaspourazad et al., 2024a;Luo et al., 2024), sleep (Thapa et al., 2024), and activity recognition (Narayanswamy et al., 2024). Despite their strong performance on data from modalities seen during pre-training, these models struggle with generalization to unseen modalities due to mismatches in input dimensions and data distributions (Liu et al., 2024).
We provide further discussion on model stitching and domain adaptation in Appendix C.
this section cite: ['b2', 'b10', 'b16', 'b22', 'b9', 'b19', 'b42', 'b8', 'b23', 'b26', 'b33', 'b35', 'b45', 'b25', 'b40', 'b31', 'b24']

Section: METHODS
The core concept of our proposed BioX-Bridge framework is to build a bridging network that facilitates efficient and effective projection between intermediate representations of biosignal models. This allows the framework to harness the strong representational power of one model while integrating the task-specific knowledge contained in another. We define the problem in Section 3.1 and introduce the idea of model bridging in Section 3.2. We detail the position of the bridge, its architecture and training in Sections 3.3-3.5. An overview of BioX-Bridge is presented in Figure 2.
this section cite: []

Section: PROBLEM DEFINITION
Assume that we are given an annotated dataset from an old biosignal modality,
D (old) = {(x (old) i ′ , y (old) i ′ )} |D (old) | i ′ =1
with |D (old) | labeled samples for a specific task, and a corresponding model,
g (old) ω • f (old) θ , where f (old)
θ is a pre-trained encoder parametrized by θ followed by a task head g (old)   ω parametrized by ω. We also have an un-annotated dataset from a new modality,
D (new) = {x (new) i ′ } |D (new) | i ′ =1
, which shares the same underlying label set with D (old) . We further have a disjoint, un-annotated paired dataset D
(pair) = {(x (old) i , x (new) i )} |D (pair) | i=1
. The unsupervised cross-modal knowledge transfer problem aims to obtain a model f , such that f can make predictions on D (new) .
this section cite: []

Section: MODEL BRIDGING
Let f (old) θ be the model for the old modality, parametrized by θ of L layers, and let f (new) ϕ be the model for the new modality, parametrized by ϕ of M layers. The intermediate representations from the m-th layer of the new modality model can then be extracted as:
h (new) m = f (new) ϕ ≤m x (new) ,(1)
where x (new) denotes a biosignal time series sample from the new modality. f (new) ϕ ≤m denotes the subset of the new modality model consisting of its first m layers, subject to the constraint 1 ≤ m ≤ M . h Next, we introduce a bridge network to enable the information flow between the new and old modality models by projecting representations from the new modality into the representation space of the old modality:
h(old
) l = b ψ h (new) m ,(2)
where b ψ denotes the bridge network parametrized by ψ. h(old) l denotes the projected representation from the new modality to the old modality. Note that the projected representation is designed to mimic the intermediate representation from the l-th layer of the old modality model, defined as h (old old) , where x (old) is the paired input signal from the old modality. Thus, h(old
) l = f (old) θ ≤l x (
) l , h (old) l ∈ R N (old) l ×d (old)
l are of the same dimension.
Finally, we can obtain predictions using the old modality model starting from the (l + 1)-th layer:
ỹ = g (old) ω • f (old) θ >l h(old) l = g (old) ω • f (old) θ >l • b ψ • f (new) ϕ ≤m x (new) ,(3)
where m and l are also known as the bridge input and output positions, • denotes function composition.
this section cite: []

Section: BRIDGE POSITION SELECTION
There are L × M possible locations where the bridge can be constructed between the layers of the two models. Although a brute-force search would yield the optimal bridge position, it is computationally expensive. In particular, the choice of the bridge position is one of the most influential factors affecting transfer performance, as we will show in ablation studies. To this end, we propose a two-stage strategy for efficient bridge position selection, as illustrated in Figure 3.
this section cite: []

Section: Stage 1: Bridge Input Position (m) Selection
The bridge serves to project new modality representations to the old modality representation space, enabling the bridged model to mimic the behavior of the old modality model. As the saying "garbage in, garbage out" suggests, it is important to select discriminative new modality representations that can effectively distinguish among the predictions produced by the old modality model, also known as pseudo labels. We propose to select the input position of the bridge by linear probing, which has been widely used to evaluate the quality of intermediate representations (Alain & Bengio, 2016). The bridge input position selection can be formulated as:
arg min m∈{1,...,M } 1 |D (pair) | |D (pair) | i=1 L probe g η h (new) m,i , ŷi ,(4)
where
h (new) m,i = f (new) ϕ ≤m (x (new) i
) denotes the i-th sample's intermediate representation from the m-th layer of the new modality model, and
ŷi = g (old) ω • f (old) θ (x (old) i
) denotes the pseudo label. L probe denotes the empirical loss for the linear prober g η . Note that the pseudo labels are simply the argmax of the teacher logits used in traditional knowledge distillation methods.
Stage 2: Bridge Output Position (l) Selection Since h(old) l = b ψ h (new) m is designed to mimic h (old) l
, we can ease the transformation process by selecting h (old)   l to be as similar as possible to h (new)  m . We select linear CKA (Kornblith et al., 2019) as a well-established measure to find correspondences between the intermediate representations of neural networks. Let H
(new) m ∈ R |D (pair) |×N (new) m d (new)
m denote the matrix of new modality representations extracted from the m-th layer , where the i-th row corresponds to the flattened h (new)  m,i . Similarly, let old)   l denote the matrix of old modality representations from the l-th layer. The bridge output position selection can be formulated as:
H (old) l ∈ R |D (pair) |×N (old) l d(
arg max
l∈{1,...,L} CKA linear H (new) m , H (old) l(5)
For detailed formulation of CKA linear , please refer to the appendix.
Algorithm 1: BioX-Bridge learning procedure 4) 3 Select bridge output position, l, using Eq. ( 5)
Input: Old modality model f (old) θ ; New modality model f (new) ϕ ; Task head g (old) ω ; Paired dataset D (pair) Output: BioX-Bridge g (old) ω • f (old) θ >l • b ψ • f (new) ϕ ≤m (•) Init: Bridge network b ψ = {A, B, P } 1 ▷ Bridge Position Selection 2 Select bridge input position, m, using Eq. (
4 ▷ Bridge Training 5 for epoch ← 1 to n epoch do 6 for step ← 1 to n step do 7 Sample mini-batch {(x (old) i , x (new) i )} bs i=1 ⊂ D (pair) 8 Compute h (old) L,i = f (old) θ x (old) i 9 Compute h(old) L,i = f (old) θ >l • b ψ • f (new) ϕ ≤m x (new) i 10 Compute L align h (old) L,i , h(old) L,i
using Eq. ( 7)
11 Update ψ w.r.t. gradients using ∇ ψ L 12 ▷ Bridge Inference 13 for step ← 1 to n step do 14 Sample mini-batch {x (new) i ′ } bs i ′ =1 ⊂ D (new) 15 ỹi ′ = g (old) ω • f (old) θ >l • b ψ • f (new) ϕ ≤m x (new) i ′ (a) Bridge Input Position Selection (b) Bridge Output Position Selection Layer 1 Layer 2 Layer L Layer l Layer M Layer 1 Layer M-1 Layer m argmax(CKA) ... ... Layer (Old) CKA ... ... 1 2 l L CKA linear (•, •) Layer M Layer 1 Layer 1 Layer 2 Layer L Layer l pesudo-label Linear + Acc(•, •) Layer M-1 Layer m ... ... Layer (New) Acc argmax(Acc) ... ... 1 m M-1 M Figure 3: Bridge Position Selection Strategy. For bridge input position, we select the layer from f (new) ϕ
whose intermediate representation exhibits the strongest linear association with the pseudolabels. For bridge output position, we select the layer from f (old) θ whose representation is most similar to that of the bridge input layer.
this section cite: ['b3', 'b21']

Section: BRIDGE ARCHITECTURE
Models of different modalities operate in distinct representational spaces. The bridge network should be sufficiently parametrized to enable the projection and alignment of the two spaces. A naive bridge architecture is a full-rank linear layer, but this is prohibitively expensive because of the high-dimensional projection from the new modality to the old modality. For example, using LaBraM (Jiang et al., 2024) as f (new) ϕ and HuBERT-ECG (Coppola et al., 2024) as
f (old) θ , the projection would require N (new) m × d (new) m × N (old) l × d (old) l = 181 × 200 × 93 × 512 ≈ 1.7 billion parameters.
To address the challenge of high-dimensional projection, we propose a prototype network. The prototype network consists of two modules, a prototype set, and a low-rank approximation module. The prototype set, P ∈ R Np×d
) l = Reshape N (old) l ×Np Pool h (new) m ⊗ A ⊗ B ⊗ P ,(6)
where Pool(•) denotes a pooling operation along the N (new) m dimension, and Reshape N (old) l ×Np (•) denotes the reshape operation to the specified output dimensions.
this section cite: ['b19', 'b8']

Section: BRIDGE TRAINING As the difference between h (old)
l and h(old) l approaches zero, the bridged model yields predictions identical to those of the old modality model. Formally:
h (old) l = h(old) l ⇒ f (old) θ >l h (old) l = f (old) θ >l h(old) l ⇒ h (old) L = h(old) L ⇒ ŷ = ỹ.
Naturally, the training objective for the bridge network is to align the intermediate representations in the L-th layerfoot_1 :
arg min ψ L align h (old) L , h(old) L = arg min ψ L align f (old) θ x (old) , f (old) θ >l • b ψ • f (new) ϕ ≤m x (new) , (7
)
where L align denotes the loss function, such as cosine loss and mean absolute error loss. The learning process of BioX-Bridge is presented in Algorithm 1.
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTAL SETUPS
Datasets & Tasks & Metrics We consider the following datasets in the evaluation: (i) WESAD (Schmidt et al., 2018) is a wearable stress and affect detection dataset consisting of synchronized data from a wrist-and chest-worn device, collected from 15 subjects during a lab study. We select the ECG and PPG modality for a three-class (baseline/amusement/stress) classification task. (ii) FOG (Zhang et al., 2022) is a multimodal dataset for detecting freezing of gait in Parkinson's Disease, collected from 12 patients. We select the EMG and EEG modality for a two-class (normal/FOG) classification task. (iii) ISRUC (Khalighi et al., 2016) is a sleep-staging dataset consisting of synchronized data from a polysomnography, collected from 118 subjects in a laboratory study. We select the EEG and ECG modality for a two-class (sleep/wake) classification task. Dataset splits and preprocessing are detailed in the appendix. Given the unbalanced nature of the datasets, we report Balanced Accuracy, F1-Weighted, and F1-Macro, following (Jiang et al., 2024;Pillai et al., 2025).
Backbone Foundation Models For EEG, we adopt the base version of the LaBraM architecture with 5.8M parameters (Jiang et al., 2024). For ECG, we adopt the small version of the HuBERT-ECG architecture with 30.4M parameters (Coppola et al., 2024). For PPG, we adopt the small version of the PaPaGei architecture with 5.7M parameters (Pillai et al., 2025). For EMG, we adopt NormWear with 136.1M parameters (Luo et al., 2024). Note that LaBraM, HuBERT-ECG, and NormWear adopt a CNN-transformer architecture, while PaPaGei adopts a CNN architecture. All models are initialized with the pre-trained weights provided by the original publications. Note that biosignal foundation models are still early in their development, in comparison to foundation models for language and vision. Although current models contain a relatively small number of parameters, our method for efficient cross-modal knowledge transfer would be even more valuable as they scale up.
As foundation models scale up, BioX-Bridge is efficient in terms of the bridge architecture to help reduce the number of trainable parameters. Furthermore, the bridge position selection strategy is lightweight in comparison to model training and scales linearly with the number of layers of the foundation models.
Baselines We compare our method with the following baselines evaluated on D (new) : (i) Random denotes a model that produces predictions at random. (ii) CardioGAN uses GAN to synthesize ECG from PPG (Sarkar & Etemad, 2021), and we translate the new modality data (PPG) to the old modality (ECG) for evaluation. (iii) KD (Hinton et al., 2015) is the baseline of knowledge distillation. (iv) KD-contrast (Abbaspourazad et al., 2024b) is a variant of knowledge distillation with contrast loss (Zhang et al., 2024). (v) Oracle denotes the absolute best performance that can be achieved, which is simply the performance of the old modality model using old modality data. Please refer to the appendix for implementation details and further discussions.
Table 1: Unsupervised cross-modal knowledge transfer performance on ISRUC, FOG, and WESAD. Oracle reflects supervised performance using the old modality only, also known as the teacher in knowledge distillation, whereas baselines and BioX-Bridge report performance on the new modality.
Results are reported as mean across five seeds; standard deviations can be found in the appendix due to space constraints. The metrics are: Balanced Accuracy (BAcc), F1-M (F1-Macro), F1-W (F1-Weighted), Trainable Parameters (Params). Input indicates the data modality serving as the model's input during evaluation on the test set. The best unsupervised result is indicated in bold.
ISRUC EEG (Old) → ECG (New) ECG (Old) → EEG (New) Methods Input BAcc ↑ F1-M ↑ F1-W ↑ Params ↓ Input BAcc ↑ F1-M ↑ F1-W ↑ Params ↓ Random -
50.00 46.48 53.52 --50.00 46.48 53.52 -KD ECG 60.24 61.01 72.96 30.4M EEG 62.24 63.69 75.27 5.8M KD-Contrast ECG 60.66 56.56 63.57 30.4M EEG 65.92 62.91 70.27 5.8M BioX-Bridge ECG 60.11 61.20 74.02 1.8M EEG 62.55 64.37 76.42 0.2M Oracle (Supervised) EEG 80.13 82.06 87.19 -ECG 63.54 65.54 76.86 -FOG EEG (Old) → EMG (New) EMG (Old) → EEG (New)
Methods Input BAcc ↑ F1-M ↑ F1-W ↑ Params ↓ Input BAcc ↑ F1-M ↑ F1-W ↑ Params ↓ Random -
50.00 49.99 50.01 --50.00 49.99 50.01 -KD EMG 68.64 67.62 67.78 136.1M EEG 68.03 67.73 67.75 5.8M KD-Contrast EMG 72.21 71.95 71.95 136.1M EEG 68.51 67.95 67.90 5.8M BioX-Bridge EMG 72.24 72.12 72.16 1.2M EEG 68.04 68.22 68.24 0.7M Oracle (Supervised) EEG 72.15 72.14 72.20 -EMG 87.55 87.58 87.60 -WESAD ECG (Old) → PPG (New) PPG (Old) → ECG (New)
Methods Input BAcc ↑ F1-M ↑ F1-W ↑ Params ↓ Input BAcc ↑ F1-M ↑ F1-W ↑ Params ↓Random
-33.33 31.29 35.38 --33.33 31.29 35.38 -CardioGAN PPG 39.32 19.63 20.33 28.2M -----KD PPG 47.86 43.08 45.75 5.7M ECG 47.03 46.36 60.29 30.4M KD-Contrast PPG 45.31 42.75 47.20 5.7M ECG 50.85 49.31 63.72 30.4M BioX-Bridge PPG 49.57 42.28 47.44 0.2M ECG 52.02 52.62 65.12 0.4M Oracle (Supervised) ECG 49.47 51.05 62.48 -PPG 62.96 60.97 74.52 -4.2 UNSUPERVISED CROSS-MODAL KNOWLEDGE TRANSFER PERFORMANCE Experiment results on the ISRUC, FOG, and WESAD dataset are presented in Table 1. We observe that BioX-Bridge significantly reduces the number of trainable parameters by 87.9-99.1% and continues to achieve performance comparable to or better than that of the baseline methods. For example, for WESAD (PPG → ECG), BioX-Bridge requires merely 1.3% of trainable parameters while outperforming the baseline methods by around 1-2% across all metrics.
We also observe that the knowledge transfer performance gap compared to the oracles varies between datasets and knowledge transfer directions. For example, on the ISRUC dataset, we observe approximately 20% balanced accuracy gap between BioX-Bridge (60.11%) and the supervised EEG oracle (80.13%) for EEG → ECG, but only 1% gap between BioX-Bridge (62.55%) and the supervised ECG oracle (63.54%) for ECG → EEG. From this, we can draw a few observations: (1) Based on oracle results, the ISRUC task is more difficult for ECG (63.54%) than for EEG (80.13%), which is reasonable, as EEG has been considered the gold standard for sleep. (2) Our unsupervised training with BioX-Bridge can effectively produce an ECG model (60.11%) that performs similarly to the supervised ECG oracle (63.54%), despite the absence of labeled data. (3) There is a 20% gap between BioX-Bridge (60.11%) and the supervised EEG Oracle (80.13%) because the baselines and BioX-Bridge use the ECG as input, which is a less physiologically relevant modality for sleep than the EEG used by the supervised EEG oracle. (4) Unsupervised cross-modal knowledge transfer performance using EEG (62.55%) is also constrained by the performance of the ECG teacher (63.54%), a consequence inherent to knowledge transfer itself. Therefore, our unsupervised training produces a relatively weaker EEG model (62.55%) compared to supervised EEG training (80.13%) since we are transferring from a weaker ECG model (63.54%).
On another note, BioX-Bridge and KD-Contrast achieved higher balanced accuracy than Oracle on several occasions, respectively. This is possible because balanced accuracy is simply an average of recall across classes. Higher balanced accuracy scores and lower F1 scores reflect that knowledge transfer methods achieved better recall but worse precision than Oracle.
this section cite: ['b37', 'b20', 'b19', 'b33', 'b19', 'b8', 'b33', 'b25', 'b36', 'b17', 'b47']

Section: ABLATION STUDIES
We conduct ablation studies on the WESAD dataset and the direction of knowledge transfer (PPG → ECG). Additional results are presented in Appendix D.6.
this section cite: []

Section: Bridge Rank and Prototype Set
We study the impact of different hyperparameters for the prototype network in Figures 4a and 4b. A performance drop is observed when the approximation rank and prototype set size are too small or too large, likely due to under-/over-parameterization of the bridge network. In particular, the performance peaks at around 0.75M parameters in both cases.
this section cite: []

Section: Dataset Size
We reduce the size of the paired dataset for bridge training. We observe in Figure 4c that the transfer performance slowly decays by around 2% at 20% dataset size, showcasing the robustness of the bridge under the low-data regime.
this section cite: []

Section: Bridge Position Selection
To show that the bridge position selection strategy proposed in Section 3.3 is effective, we compare the unsupervised performance of cross-modal knowledge transfer at various positions in Table 2. For "Fixed", we train nine bridges in predefined positions, which is a combination of the first, middle, and last layers of the old and new modality models. The results are the average over all nine positions in five seeds. A breakdown of the result at each of the nine predefined positions is available in Appendix D.5.
this section cite: []

Section: Foundation Model
We further analyze the impact of using different foundation models for crossmodal knowledge transfer. In Table 3, we replace the HuBERT-ECG foundation model (Coppola et al., 2024) with ECG-FM (McKeen et al., 2024). Notably, due to the large number of trainable parameters (90M), the knowledge distillation methods with ECG-FM could only be performed with a batch size of 4 on a V100 GPU. As a result, training for more than 50 epochs requires 6.5 hours for knowledge distillation methods and 1.9 hours for BioX-Bridge. Moreover, the performance gap between knowledge distillation methods and BioX-Bridge is much pronounced at 10-17%.
Table 2: Bridge Position Ablation. Comparison of different bridge position selection strategies with respect to BioX-Bridge. "Fixed" represents the average of 9 predefined positions, combining the first, middle, and last layers for both the input and output positions. Methods BAcc ↑ F1-M ↑ F1-W ↑ Fixed 48.34 46.83 58.37 BioX-Bridge 52.02 52.62 65.12
this section cite: ['b8', 'b26']

Section: CONCLUSION
We present BioX-Bridge as an efficient framework for unsupervised cross-modal knowledge transfer across biosignals. To address the challenges of high-dimensional projection between biosignal foundation models, we design a prototype-based architecture for parameter-efficient learning of transformations between representation spaces. Our proposed two-stage bridge position selection strategy further identifies connection points that enable more effective alignment of intermediate representations. Through extensive experiments on diverse biosignal datasets and tasks, we demonstrated that BioX-Bridge achieves performance comparable to or superior to that of state-of-the-art methods while drastically reducing the number of trainable parameters. This work highlights the potential of model bridging as a powerful alternative to conventional cross-modal knowledge transfer techniques, offering a pathway to more accessible, adaptable, modality-agnostic, and resource-efficient biosignal applications in real-world settings, where computing resources and labelled data are often limited.
this section cite: []

Section: References
Ref_id:b0 Title: Large-scale training of foundation models for wearable biosignals Year: ()
Ref_id:b1 Title: Wearable accelerometer foundation models for health via knowledge distillation Year: (2024)
Ref_id:b2 Title: Shyamal Anadkat, et al. GPT-4 technical report Year: (2023)
Ref_id:b3 Title: Understanding intermediate layers using linear classifier probes Year: (2016)
Ref_id:b4 Title: SoundNet: Learning sound representations from unlabeled video Year: (2016)
Ref_id:b5 Title: Large cognition model: Towards pretrained EEG foundation model Year: (2025)
Ref_id:b6 Title: EEG-Former: Towards transferable and interpretable large-scale EEG foundation model Year: (2024)
Ref_id:b7 Title: GPT-PPG: A GPT-based foundation model for photoplethysmography signals Year: (2025)
Ref_id:b8 Title: HuBERT-ECG: A self-supervised foundation model for broad and scalable cardiac applications. medRxiv Year: (2024)
Ref_id:b9 Title: Neuro-GPT: Developing a foundation model for EEG Year: (2023)
Ref_id:b10 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b11 Title: Cascade attention guided residue learning GAN for cross-modal translation Year: (2021)
Ref_id:b12 Title: Modality distillation with multiple stream networks for action recognition Year: (2018)
Ref_id:b13 Title: Generative adversarial networks Year: (2020)
Ref_id:b14 Title: Knowledge distillation: A survey Year: (2021)
Ref_id:b15 Title: Cross modal distillation for supervision transfer Year: (2016)
Ref_id:b16 Title: Foundation models in electrocardiogram: A review Year: (2024)
Ref_id:b17 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b18 Title: Learning with side information through modality hallucination Year: (2016)
Ref_id:b19 Title: Large brain model for learning generic representations with tremendous EEG data in BCI Year: (2024)
Ref_id:b20 Title: ISRUC-Sleep: A comprehensive public dataset for sleep researchers Year: (2016)
Ref_id:b21 Title: Similarity of neural network representations revisited Year: (2019)
Ref_id:b22 Title: A simple review of eeg foundation models: Datasets, advancements and future perspectives Year: (2025)
Ref_id:b23 Title: An electrocardiogram foundation model built on over 10 million recordings with external evaluation across multiple domains Year: (2024)
Ref_id:b24 Title: Towards modality generalization: A benchmark and prospective analysis Year: (2024)
Ref_id:b25 Title: Toward foundation model for multivariate wearable sensing of physiological signals Year: (2024)
Ref_id:b26 Title: ECG-FM: An open electrocardiogram foundation model Year: (2024)
Ref_id:b27 Title: Applications of artificial intelligencemachine learning for detection of stress: A critical overview Year: (2024)
Ref_id:b28 Title: Conditional generative adversarial nets Year: (2014)
Ref_id:b29 Title: A survey on knowledge distillation: Recent advancements Year: (2024)
Ref_id:b30 Title: A systematic review of detecting sleep apnea using deep learning Year: (2019)
Ref_id:b31 Title: Scaling wearable foundation models Year: (2024)
Ref_id:b32 Title: Cardiac arrhythmia detection using deep learning: A review Year: (2019)
Ref_id:b33 Title: PaPaGei: Open foundation models for optical physiological signals Year: (2025)
Ref_id:b34 Title: The american academy of sleep medicine inter-scorer reliability program: sleep stage scoring Year: (2013)
Ref_id:b35 Title: Pulse-PPG: An open-source field-trained PPG foundation model for wearable applications across lab and field settings Year: (2025)
Ref_id:b36 Title: Attentive generative adversarial network with dual discriminators for synthesis of ECG from PPG Year: (2021)
Ref_id:b37 Title: Introducing WESAD, a multimodal dataset for wearable stress and affect detection Year: (2018)
Ref_id:b38 Title: MRI to PET cross-modality translation using globally and locally aware GAN (GLA-GAN) for multi-modal diagnosis of alzheimer's disease Year: (2021)
Ref_id:b39 Title: Golden standard or obsolete method? Review of ECG applications in clinical and experimental context Year: (2022)
Ref_id:b40 Title: SleepFM: Multi-modal representation learning for sleep across brain activity, ECG and respiratory signals Year: (2024)
Ref_id:b41 Title: Contrastive representation distillation Year: (2020)
Ref_id:b42 Title: A criss-cross brain foundation model for EEG decoding Year: (2024)
Ref_id:b43 Title: BioBridge: Bridging biomedical foundation models via knowledge graphs Year: (2023)
Ref_id:b44 Title: Multimodal knowledge expansion Year: (2021)
Ref_id:b45 Title: Biot: Biosignal transformer for cross-data learning in the wild Year: (2023)
Ref_id:b46 Title: MRI crossmodality image-to-image translation Year: (2020)
Ref_id:b47 Title: Brant-X: A unified physiological signal alignment framework Year: (2024)
Ref_id:b48 Title: Multimodal data for the detection of freezing of gait in parkinson Year: (2022)
Ref_id:b49 Title: Through-wall human pose estimation using radio signals Year: (2018)
Ref_id:b50 Title: Unpaired image-to-image translation using cycle-consistent adversarial networks Year: (2017)
Ref_id:b51 Title: Learning your heart actions from pulse: ECG waveform reconstruction from PPG Year: (2021)
Ref_id:b52 Title: Revisiting model stitching to compare neural representations Year: (2021)
Ref_id:b53 Title: Similarity and matching of neural network representations Year: (2021)
Ref_id:b54 Title: Promoting cross-modal representations to improve multimodal foundation models for physiological signals Year: (2024)
Ref_id:b55 Title: Efficient stitchable task adaptation Year: (2024)
Ref_id:b56 Title: Understanding image representations by measuring their equivariance and equivalence Year: (2015)
Ref_id:b57 Title: Heterogeneous domain adaptation: An unsupervised approach Year: (2020)
Ref_id:b58 Title: Frequency-aware masked autoencoders for multimodal pretraining on biosignals Year: (2023)
Ref_id:b59 Title: Relative representations enable zero-shot latent space communication Year: (2023)
Ref_id:b60 Title: Stitchable neural networks Year: (2023)
Ref_id:b61 Title: Exploring novel algorithms for atrial fibrillation detection by driving graduate level education in medical machine learning Year: (2022)
Ref_id:b62 Title: A survey of unsupervised deep domain adaptation Year: (2020)
Ref_id:b63 Title: Heterogeneous-modal unsupervised domain adaptation via latent space bridging Year: (2025)
Ref_id:b64 Title: Deep model reassembly Year: (2022)
Ref_id:b65 Title: Multimodal data for the detection of freezing of gait in parkinson Year: (2022)
Ref_id:b66 Title: Self-supervised contrastive pre-training for time series via time-frequency consistency Year: (2022)
