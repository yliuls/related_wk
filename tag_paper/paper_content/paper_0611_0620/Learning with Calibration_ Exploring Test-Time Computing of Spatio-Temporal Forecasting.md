Title: Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting
Abstract: Spatio-temporal forecasting is crucial in many domains, such as transportation, meteorology, and energy. However, real-world scenarios frequently present challenges such as signal anomalies, noise, and distributional shifts. Existing solutions primarily enhance robustness by modifying network architectures or training procedures. Nevertheless, these approaches are computationally intensive and resourcedemanding, especially for large-scale applications. In this paper, we explore a novel test-time computing paradigm, namely learning with calibration, ST-TTC, for spatio-temporal forecasting. Through learning with calibration, we aim to capture periodic structural biases arising from non-stationarity during the testing phase and perform real-time bias correction on predictions to improve accuracy. Specifically, we first introduce a spectral-domain calibrator with phase-amplitude modulation to mitigate periodic shift and then propose a flash updating mechanism with a streaming memory queue for efficient test-time computation. ST-TTC effectively bypasses complex training-stage techniques, offering an efficient and generalizable paradigm. Extensive experiments on real-world datasets demonstrate the effectiveness, universality, flexibility and efficiency of our proposed method. Our code repository is available at https://github.com/Onedean/ST-TTC.

Section: Introduction
Spatio-temporal forecasting (STF) aims to predict the future state of dynamic systems from historical spatio-temporal observations and underpins many real-world applications, such as traffic flow forecasting [24], air quality forecasting [51], and energy consumption forecasting [76]. Although spatio-temporal neural networks [32,33,61], which couple spatial neural operators with temporal neural operators, have achieved remarkable progress on these tasks, their deployment in practical environments remains fraught with challenges. These observations, typically collected by sensors, are frequently corrupted by noise, outliers (e.g., spikes or dropouts due to hardware failure) [85], and more commonly, non-stationary distribution shifts arising from sensor aging and seasonal patterns [74].
To enhance generalization and performance, prior work has focused primarily on out-of-distribution (OOD) learning for ST data during the training phase: designing architectures that resist perturbations [31,49,68,84], augmenting training data with noise or adversarial examples [3,46,97], and introducing specialized loss functions or regularizers [43,98] to counteract distribution drift. However, these methods share fundamental limitations: they assume that the training data sufficiently captures all future target domain invariance, a premise that is rarely valid in real-world settings. Concurrently, an emerging paradigm of continual fine-tuning [10,11,35,38,69,70,71] has become popular in spatio-temporal learning by continuously tuning the model to adapt to dynamic changes. Though promising, it still divides the target domain into multiple periods of training and testing and relies on period-specific training data to optimize model, thereby failing in data-scarse scenarios.
Table 1: Formal comparison of different spatio-temporal learning paradigms for generalization from the perspective of data and learning. s denotes the source domain, t denotes the target domain, x and y denote the samples and labels sampled from X and Y, respectively. OOD learning expects inputs sampled from any environment e * ∼ E to be valid, while others are only optimized for the current training or test environment e. In particular, continual fine-tuning divides the target domain into multiple stages and optimizes for a specific stage τ environment e τ . ✗ means not involved.
this section cite: ['b23', 'b51', 'b76', 'b31', 'b32', 'b61', 'b85', 'b74', 'b30', 'b49', 'b68', 'b84', 'b2', 'b46', 'b97', 'b43', 'b98', 'b9', 'b10', 'b34', 'b38', 'b69', 'b70', 'b71']

Section: Setting
Example Works Data Perspective Learning Perspective Source Target Train-Time Test-Time 𝑓 𝜃 ↓ → ො 𝑥 𝑡 → 𝐿𝑜𝑠𝑠( ො 𝑥 𝑡 , 𝑥 𝑡 ) 𝑥 𝑡 ↓ 𝑥 𝑡 → ො 𝑦 𝑡 = 𝑓(𝑥 𝑡 ; 𝜃(𝛿)) 𝑓 𝜃(𝛿) ↓ → ො 𝑦 𝑡 → 𝐿𝑜𝑠𝑠( ො 𝑦 𝑡 , 𝑦 𝑡 ) 𝑦 𝑡 ↓ (a) Test-Time Training (b) Online Continual Learning ො 𝑦 𝑡 = 𝑓(𝑥 𝑡 ; 𝜃) 𝑓 𝜃 ↓ → ො 𝑦 𝑡 → 𝐿𝑜𝑠𝑠( ො 𝑦 𝑐𝑎𝑙 𝑡 , 𝑦 𝑡 ) 𝑦 𝑡 ↓ (c) Test-Time Computing 𝑥 𝑡 → ො 𝑦 𝑐𝑎𝑙 𝑡 = 𝑔( ො 𝑦 𝑡 ; 𝜃) 𝑔 𝜃 ↓ → ො 𝑦 𝑐𝑎𝑙 𝑡 →
Red represents the parameters that need to be optimized during the test. Recently, leveraging test-time information has attracted widespread attention for its ability to significantly improve language model performance on complex reasoning tasks [1,64]. In computer vision, this concept has already been extensively developed: test-time training (TTT) was first introduced by [66], which defines an auxiliary self-supervised task applied to both training and test samples to better balance bias and variance [20]. A similar idea was adapted to spatio-temporal forecasting in TTT-ST [9]. Unlike language and vision settings-where obtaining ground-truth labels for test samples at inference time is nearly impossible-STF benefits from label autocorrelation [14]: each observation strongly depends on its predecessor, and training instances are constructed from sliding windows, which provide access to historical samples and their true labels. Moreover, this property makes STF also require timeliness [60], that is, the additional computing time during inference must be less than the window-stride interval. A recent STF method, DOST [72], explores online continual learning, which initially explored this direction. It uses historical test sample labels to dynamically adapt the modified model architecture. Though promising, these approaches typically involve complex self-supervised tasks or structural adaptations and still fall short of the timeliness demands of STF.
To address this gap, we propose Test-Time Computing of Spatio-Temporal Forecasting (ST-TTC), an attractive complementary paradigm. ST-TTC achieves learning with calibration by iteratively leveraging available test information during inference, enabling seamless integration with diverse models. This adapts the model to evolving spatio-temporal patterns, thereby calibrating predictions. Our principal insight is that performance degradation during test time is primarily driven by nonstationary distributional shifts stemming from progressive periodic biases. Therefore, we propose a spectral domain calibrator. This involves appending a lightweight module, operating in the frequency domain, subsequent to the backbone network. This module calibrates biases by learning minor, nodespecific amplitude and phase correction factors. Furthermore, a flash gradient updating mechanism with a streaming memory queue, ensures universal, rapid, and resource-efficient test-time computing. Table 1 provides a formal comparison of our method against existing learning paradigms, and Figure 1 offers a conceptual visualization of learning with test domain. In summary, our contributions are:
• We propose a novel test-time computing paradigm of spatio-temporal forecasting, termed ST-TTC .
• We systematically explore the goals and means of achieving this paradigm. Concretely, we introduce a spectral domain calibrator with phase-amplitude modulation to mitigate periodic shift and present a flash updating mechanism with a streaming memory queue for efficient test-time computation.
• Experimental results on real-world spatio-temporal datasets in different fields, scenarios, and learning paradigms demonstrate the effectiveness and universality of ST-TTC .
this section cite: ['b0', 'b64', 'b66', 'b19', 'b8', 'b13', 'b60', 'b72']

Section: Related Work
Spatio-Temporal Forecasting. Spatio-temporal sequences can be regarded as spatially extended multivariate time series. Although one can trivially apply multivariate forecasting methods [5,7,52,96] independently at each location, such decoupling of spatial and temporal dependencies invariably yields suboptimal results [61]. Classical spatio-temporal forecasting method instead relies on shallow models or spatio-temporal kernels, including feature-based methods [53,102], state space models [2,13,56], and Gaussian process models [18,58]. Unfortunately, the overall nonlinearity of these models is limited, and the high complexity of computation and storage further hinders the availability of massive training instances [63]. In recent years, spatio-temporal neural networks [32,33,36] have been widely adopted to learn the complex dynamics of such systems. Early work concentrated on devising neural operators to extract spatial or temporal correlation [17,47,62,83,90] and on designing fusion architectures to integrate them [12,16,23,40,54]. More recent efforts have explored domain-invariant representation learning [49,103,104] and continual model adaptation [10,11,89] to better accommodate unseen environmental shifts. However, these methods still depend exclusively on offline training data and thus cannot deliver truly timely and effective adaptation in real settings.
Test-Time Computing. Test-time computation is inspired by the human cognition [34], in which additional computational effort is allocated during inference to improve task performance. This insight has recently driven considerable interest in the nature language process community, fueled by the success of reasoning-augmented language models (e.g., o1 [29] and r1 [21]) that activate and adapt internal computations at test time via supervised fine-tuning or reinforcement learning (RL) [99]. While the generalization properties of RL-based adaptation remain debated [95], the notion of supervised learning on unlabeled test data dates back to "transductive learning" [19] in the 1990s and has demonstrated empirical benefits [6,67]. In the computer vision domain, this idea was formalized as Test-Time Training [66], which attaches an auxiliary self-supervised head to enable online adaptation to each test instance-a paradigm subsequently generalized as test-time adaptation [30,44,73,77]. However, spatio-temporal forecasting has seen limited exploration of such techniques. TTT-ST [9] applies TTT-style auxiliary objectives during training and continues to update at inference, and DOST [72] further incorporates dynamic learning mechanisms within modified model architectures for test-time updates. In addition, some methods [22,100] are conceptually close to ours, such as CompFormer [100], which proposes a test-time compensated representation learning framework, but still requires access to additional training data. Notably, we formalize the test-time computing of spatio-temporal forecasting, and propose a unified learning-with-calibration framework that is general, lightweight, efficient, and effective for STF at test-time.
For more related work, we provide a more detailed introduction in Appendix A.
this section cite: ['b4', 'b6', 'b52', 'b96', 'b61', 'b53', 'b102', 'b1', 'b12', 'b56', 'b17', 'b58', 'b63', 'b31', 'b32', 'b35', 'b16', 'b47', 'b62', 'b83', 'b90', 'b11', 'b15', 'b22', 'b40', 'b54', 'b49', 'b103', 'b104', 'b9', 'b10', 'b89', 'b33', 'b28', 'b20', 'b99', 'b95', 'b18', 'b5', 'b67', 'b66', 'b29', 'b44', 'b73', 'b77', 'b8', 'b72', 'b21', 'b100', 'b100']

Section: Preliminaries
Problem Definition. Let x ∈ R T ×C denote the multivariate time series recorded at each location sensor, capturing the dynamic observations of C measured features in T consecutive time steps. Stacking these sequences for all N locations yields the spatio-temporal tensor X ∈ R N ×T ×C . Given historical observations X h ∈ R N ×T h ×C (and an optional spatial correlation graph G representing the spatial relationships of N locations), spatio-temporal forecasting aims to learn a mapping f θ :
(X h , G) -→ X f ∈ R N ×T f ×C
, where X f is the signal for the next T f time steps. In practice, according to [40,61], the feature to be predicted is usually only the target variable.
this section cite: ['b40', 'b61']

Section: Scenario Definition.
In deep learning systems, batch-based testing is typically employed to exploit parallelism. In real-world deployment, however, predictions must be produced for each incoming time-step sample-i.e., with batch size B set to 1. At time index t, once the new sliding-window input X t ∈ R N ×T h ×C arrives, the true labels for all test samples before time index t-T h -T f +1 become available. Thus, test-time computing of spatio-temporal forecasting can leverage this accumulated historical information to enhance the accuracy of the current prediction, while ensuring that any additional computation latency remains below a threshold defined by the sliding-window stride.
this section cite: []

Section: Methodology
Our test-time computing framework of spatio-temporal forecasting (ST-TTC) integrates two synergistic components: 1) a spectral domain calibrator with phase-amplitude modulation; and 2) a flash gradient update mechanism with streaming memory queue. In this section, we introduce these two key components, respectively, from the perspective of what is computed and how it is computed.
this section cite: []

Section: What to Compute? Spectral Domain Calibrator with Phase-Amplitude Modulation
Motivation. Spatio-temporal data, such as traffic flow and air quality, often exhibit periodic patterns (e.g., daily or weekly cycles). However, in real-world deployments, these patterns are not stationary; they are dynamically influenced by various internal and external factors [75]. Such influences lead to non-stationarities manifesting as fluctuations in amplitude (e.g., increased or decreased traffic peaks due to seasonal changes) or phase shifts (e.g., peak hours are advanced or delayed due to traffic congestion). Pre-training models typically fit fixed periodic patterns during training, which makes them vulnerable to performance degradation under such persistent dynamic changes during inference [74]. Therefore, we argue that the goal of test-time computation is: how to design an effective calibrator that can efficiently capture such gradual systematic bias from the pattern to correct the prediction errors caused by non-stationarity, while avoiding overfitting to random noise?
Key Challenges. While correction in the time domain is possible [22,100], it often requires extensive parameterization, leading to increased model complexity and limited ability to capture evolving periodic structures. Moreover, the coupled structural and branching modules [9,72] are prone to overfitting the random noise in the spatio-temporal evolution. To address this, we propose calibration in the spectral domain, where periodic variations are more transparently expressed as changes in the amplitude and phase of specific frequency components. Spectral correction offers a potentially more direct and robust solution. However, this introduces two main challenges: ❶ the degree of non-stationarity varies across spatial nodes; and ❷ full-spectrum parameterization is computationally expensive. The core problem thus becomes how to design a lightweight, spatial-aware calibrator.
this section cite: ['b75', 'b74', 'b21', 'b100', 'b8', 'b72']

Section: Implementation Details.
To this end, we formally introduce the spectral domain calibrator (SD-Calibrator), which is a lightweight plug-and-play module that performs spectral domain calibration on the time domain prediction results of the pre-trained model, aiming to achieve efficient test-time computation for spatio-temporal forecasting. Specifically, it can be divided into three steps:
• Spatial-aware Decomposition. To ensure spatial awareness, we apply a real-to-complex fast Fourier transform (rFFT) along the time dimension of the backbone model's prediction ŷ ∈ R B×N ×T , separately for each spatial node. This yields the frequency spectrum: Y f = rFFT(ŷ) ∈ C B×N ×M , where M = T 2 + 1 is the number of unique frequency bins for real-valued signals. Then, we decompose Y f into its amplitude A = |Y f | ∈ R B×N ×M and phase P = ∠Y f ∈ R B×N ×M .
• Group-wise Modulation. To ensure lightweight and balanced spectrum expression, we divide the M frequency bins into G contiguous groups of size ⌊M/G⌋, and learn per-group, per-node amplitude and phase offsets λ α ∈ R G×N ×1 , λ ϕ ∈ R G×N ×1 (Note: Both λ α and λ ϕ are initialized to 0 to avoid incorrect calibration of predictions before learning). For each group g ∈ {1, . . . , G}, we apply
A ′ g = A g ⊙ (1 + λ α g , P ′ g = P g + λ ϕ g
, and reconstruct the spectrum as
Y ′ f = G g=1 A ′ g ⊙ e (j P ′ g ) .
• Inverse Transform. Finally, the calibrated time-domain signal is obtained by Inverser rFFT ŷcal = irFFT Y f ) ∈ R B×N ×T , along the frequency dimension.
For clarity, we provide a Algorithm workflow 1 and Pytorch-Style Pseudocode 2 in Appendix C.1.
this section cite: []

Section: Complexity Analysis.
The full-spectrum parameterization learns independent amplitude and phase offsets for each of the M = T /2 + 1 frequency bins and N nodes, totaling 2N M parameters. In contrast, our G-group design learns only 2N G parameters. Since G is a constant and M grows linearly with T , G ≪ M is usually the case. For large-scale long-term scenario, this significantly reduces memory footprint and gradient update cost while retaining interpretable per-band calibration.
this section cite: []

Section: Theoretical Analysis.
We also provide a theoretical approximate bound on the output perturbation induced by the SD-Calibrator, ensuring controlled deviation from the original prediction to prevent overfitting (Please refer to Theorem 1 and the proof in Appendix B).
this section cite: []

Section: How to Compute? Flash Gradient Update with Streaming Memory Queue
Motivation. The SD-Calibrator provides an effective mechanism for output correction. To accommodate the dynamic nature of spatio-temporal data, its parameters (λ α , λ ϕ ) must be continuously updated during inference. Fortunately, as we discussed above, due to the streaming nature of spatio-temporal data, unlike Visual and textual tasks, we have access to the true labels of historical samples. However, simply accumulating all historical data for updates is not feasible due to the increasing computational load and memory usage. Therefore, we argue that the key to test-time computation is: How to design an efficient data selection and learning mechanism that leverages appropriate historical information to tuning the SD-Calibrator without incurring a lot of computational overhead?
Key Challenges. Although retrieving similar sequences from historical training databases can partially compensate for prediction errors [100], this assumption is unrealistic, as only test-time information is available in our scenario. Moreover, selectively storing historical test samples via memory bank primarily serves to mitigate catastrophic forgetting in the backbone model [72], which misaligns with the learning objective of our SD-calibrator. To address this, we propose freezing the backbone and updating only the calibrator using recent test samples for efficient test-time computing. However, this strategy introduces two critical challenges: ❶ recent studies [37] have shown that real-time updates may cause information leakage; and ❷ excessive updates can lead to overfitting of the calibration parameters and increased computational burden. The core problem thus becomes how to design a efficient calibration parameter learning mechanism without information leakage.
this section cite: ['b100', 'b72', 'b36']

Section: Implementation Details.
To address these challenges, we introduce the flash gradient update strategy coupled with a streaming memory queue. The process is as follows:
• Streaming Memory Queue. We maintain a first-in, first-out (FIFO) queue, denoted as Q, with a maximum size equal to the prediction horizon T f . For each incoming test instance t, after making a prediction, we store the input-label pair (X t , Y t ) into Q (Here is for engineering convenience. In real deployment, data points can be merged at each step to form the true label). Once Q is full, for every new test sample (X n , Y n ) added, the oldest sample pair (X o , Y o ) is dequeued. This dequeued sample (X o , Y o ) is then used for the gradient update, thus avoiding the information leakage.
• Flash Gradient Update. Once we have (X o , Y o ), we first obtain the backbone model's prediction for the historical input: For clarity, we provide a Algorithm workflow 3 and Pytorch-Style Pseudocode 4 in Appendix C.2.
Ŷ b o = f θ (X o ) (note
this section cite: []

Section: Complexity Analysis.
The primary focus here is the time complexity. The Streaming Memory Queue itself has an O(1) time complexity for enqueue and dequeue operations. The Lightning Gradient Update is performed only once for each incoming test sample. Each update involves: 1). Forward propagation of the backbone and calibrator (dominated by the computational cost O(N T logT ) of rFFT and irFFT) 2). Backward propagation of the calibrator (dominated by parameter cost O(N G)).
this section cite: []

Section: Theoretical Analysis.
We also show that this single update step leads to a controlled adjustment, ensuring that the calibrator makes progress on the newest sample it's trained on, without causing erratic behavior, under standard assumptions. (Please refer to Proposition 2 in Appendix B).
this section cite: []

Section: Experiments
In this section,we conduct extensive experiments to answer the following research questions (RQs):
• RQ1: Can ST-TTC have a consistent improvement on various types of models and datasets? Can ST-TTC outperform previous learning methods that leverage test data? (Effectiveness)
• RQ2: Can ST-TTC effective in various real-world scenarios, including few-shot learning, longterm forecasting, and large-scale forecasting? (Universality)
• RQ3: Can ST-TTC further enhance the performance of existing learning paradigms that utilize training data, such as OOD Learning and continual learning? (Flexibility)
• RQ4: How does ST-TTC work? Which components or strategies are crucial? Are these components or strategies sensitive to parameters or design? (Mechanism & Robustness)
• RQ5: What is the time and parameter cost of ST-TTC during test-time computation, and how does it compare to other advanced methods? (Efficiency & Lightweight)
this section cite: []

Section: Experimental Setup
Datasets. We employ publicly available benchmark datasets widely used in the literature to cover typical spatio-temporal forecasting scenarios in the traffic domain (PEMS-03, PEMS-04, PEMS-07, PEMS-08 [65]), the meteorological domain (KnowAir [79]), and the energy domain (UrbanEV [39]).
In addition, we also leverage the traffic-speed benchmark METR-LA [40], the large-scale spatiotemporal benchmark LargeST [48], and dynamic-stream benchmarks (Energy-Stream, Air-Stream, PEMS-Stream [10]) to assess our methods across varied settings and learning paradigms. Unless otherwise specified, all datasets are chronologically split into training, validation and test sets in a 6 : 2 : 2 ratio. For more detailed description of each dataset, please see the Appendix D.1.
Baseline. For the default evaluation, we cover various widely used spatio-temporal backbones, which can be divided into three categories: ( 1) Transformer-based: STAEformer [47] and STTN [86]; (2) Graph-based: GWNet [83] and STGCN [90]; (3) MLP-based: STID [62] and ST-Norm [15]. For the baselines that leverage test information, we cover three types: (1) popular test-time adaptation methods in vision: TTT-MAE [20] and TENT [73]; (2) Online time series forecasting methods: OnlineTCN [105], FSNet [57] and OneNet [81]; (3) Comparable online spatio-temporal forecasting methods: CompFormer [100] and DOST [72]. For the baselines on large-scale benchmarks, we use the efficient PatchSTG [17] as the backbone. For the baseline of OOD learning scenarios, we use the advanced STONE [68] as the default method. For the continual learning scenario, we use EAC [10] and STKEC [70] as the default methods. We follow the default parameter settings of the models for all scenarios according to the corresponding literature. For details of each method, see Appendix D.2.
Protocol. Following prior benchmarks [61], we employ a 12-to-12 forecasting protocol-using the previous 12 time steps to predict the next 12 steps and their mean-evaluated with mean absolute error (MAE), root mean square error (RMSE), and mean absolute percentage error (MAPE). For simplicity, all experiments share the same hyperparameters of our ST-TTC : the calibration module learning rate lr is set to 1e-4, the memory-queue sample count n used for updating is 1, and the number of groups m to 4. To ensure fairness, each experiment is repeated five times, with results reported as mean ± standard deviation (denoted in gray ±). More protocol details, see Appendix D.3.
this section cite: ['b65', 'b79', 'b39', 'b40', 'b48', 'b9', 'b72', 'b16', 'b68', 'b9', 'b70', 'b61']

Section: Effectiveness Study (RQ1)
Consistent Effectiveness. Table 2 presents the results of our method for 12-step future prediction across six models on six public datasets. The ✗ column denotes the results of standard testing, while the ✓ column indicates results obtained with our proposed ST-TTC approach. The best results in the ✗ and ✓ columns are highlighted in bold blue and pink fonts, respectively. We also compute the relative improvement, denoted by the ∆ column. Based on these results, we make the following observations: ❶ The application of our test-time computation method, ST-TTC , consistently yields performance gains across various backbone architectures and dataset combinations. ❷ From a modelcentric perspective, our approach can further enhance the performance of even the top-performing methods across different metrics and datasets. ❸ From a data-centric perspective, UrbanEV shows more significant relative improvement, likely due to its more pronounced distribution shift.   3, based on their respective papers. Additionally, we implemented the popular TTT-MAE method as a surrogate for the unavailable TTT-ST method.
Our observations are as follows: ❶ For the regular setting, our method achieves competitive results with more stable standard deviations. While other methods like CompFormer demonstrate similar performance, they often utilize more training information and computational resources. ❷ In the online setting, our method significantly outperforms existing approaches without requiring more complex model architecture modifications.
this section cite: []

Section: Universality Study (RQ2)
To demonstrate the universality of ST-TTC across diverse real-world scenarios, we explore various forecasting scenarios in the literature, including few-shot [94], long-term [59], and large-scale [25].
Few-Shot Scenario. To simulate limited training data, we retrained models using only the first 10% of existing training sets to investigate a more common and challenging few-shot scenario. Figure 2 shows the relative performance gains with ST-TTC (For full results, please refer Table 6 in the Appendix). We observe: ❶ ST-TTC provides more significant improvements in the few-shot setting compared to the full-shot case in Table 2, with about half exceeding 2%. ❷ KnowAir shows the largest gain compared to other datasets, likely because its four-year long period leads to a substantial test distribution shift in the few-shot scenario, where our method adapts well. Long-Term Scenario. In real-world scenarios, long-term forecasting helps to further plan future decisions. We predicted 24 future steps from 24 past steps to explore more complex temporal changes. As shown in Figure 3, we present the relative performance improvement of the advanced STID model with our ST-TTC method, and give a test set prediction visualization case on the PEMS-08 dataset (see Figure 9 in the Appendix for more examples). Our observations include: ❶ ST-TTC consistently improves long-term forecasting, even more than short-term (Table 2), likely due to more learnable information in longer windows. ❷ As the pink and orange box shows, our method learns test-time history, capturing both the global traffic decline and local fluctuations, leading to effective calibration. Large-Scale Scenario. Beyond current regional datasets, state or national-level spatio-temporal forecasting can involve tens of thousands of stations and longer time frames. We explore large-scale scenarios using the popular LargeST benchmark (comprising SD, GBA, GLA, and CA subsets). Figure 4 illustrates the 12-step prediction performance gains of the state-of-the-art efficient spatio-temporal model PatchSTG [17] with our ST-TTC , along with a comparison of inference time complexity (For full results, see Table 7 in Appendix). We observe: ❶ Our ST-TTC consistently yields further performance improvements across all datasets, even surpassing the improvement of the second-best baseline over the PatchSTG on some datasets. ❷ The additional inference time is at most 3.82 minutes, which is a clear advantage for the achieved performance gains compared to the training time cost of up to 14 hours.
this section cite: ['b94', 'b59', 'b24', 'b16']

Section: Flexibility Study (RQ3)
To illustrate the flexibility of ST-TTC in accommodating existing learning paradigms, we explore its integration with two training data-leveraging paradigms: OOD learning and Continual Learning.
this section cite: []

Section: OOD Learning Setting.
Following prior work [68], we use the SD dataset to simulate spatio-temporal shift. For the temporal dimension, we use 1-8/2019, 9-10/2019, and 11-12/2020 for training, validation, and testing, respectively. For the spatial dimension, we randomly mask 10% of nodes in the test set and consider three proportions of new nodes (10% / 15% / 20%) relative to the training node to mimic varying degrees of shift. In Figure 5, we present the 12-step average prediction performance gains of the advanced OOD learning model STONE with our ST-TTC , evaluated on all nodes and new nodes to demonstrate generalizability and scalability (Full results in Table 8).
We observe that: ❶ The STONE model with ST-TTC consistently achieves performance benefits, significantly outperforming all previous settings, indicating that existing OOD models are still insufficient for true OOD generalization, while our method is highly effective. ❷ For both all and new nodes, our improvements become more pronounced as the shift increases, further demonstrating our effectiveness in handling both generalizability and scalability in challenging scenarios.
this section cite: ['b68']

Section: Continual Learning Setting.
Following prior work [10], we used multi-period streaming spatiotemporal data to examine our ST-TTC 's integration with continual learning method. Table 4 shows the improved 12-step forecasting of advanced continual learning models EAC and STKEC with our ST-TTC . We observed: ❶ Consistent performance gains for both models across all datasets; STKEC with ST-TTC even achieved comparable performance to best model EAC. (2) Energy-  Stream achieves significant improvement over other datasets, as the ST-TTC effectively learns and calibrates temporal changes, as shown by the frequency analysis (drastic shift changes) in Figure 6. We follow the OOD setup (challenging setting with 20% new nodes) to evaluate our ST-TTC .
this section cite: ['b9']

Section: Strategy Study.
We compare different strategies: 1) simple nonlinear time domain calibration (Time), 2) learning only phase or amplitude modulation factors (Pha. / Amp.), 3) node-share modeling (Node), and (4) random selection or retrieval of the most similar samples (Rand. / Sim.). As shown in Figure 7 left, we observe: ❶ Frequency-domain calibration significantly outperforms time-domain calibration, with amplitude modulation being the primary contributor; ❷ Sharing nodes leads to performance degradation due to spatial heterogeneity in spatio-temporal data; ❸ Random sample selection reduces performance, and retrieving similar samples offers negligible gains while incurring higher computational cost. Our proposed update strategy is already near-optimal.
this section cite: []

Section: Parameter study.
We analyze the sensitivity of two parameter groups. As shown in the middle and right of Figure 7: ❶ Higher learning rates and fewer groups generally lead to poorer performance, likely due to limited parameter capacity hindering stable learning; ❷ Increasing the number of samples or update steps has minimal impact on performance (fluctuations < 1%), but significantly increases time cost, highlighting the rationale of our flash update mechanism.
this section cite: []

Section: Efficiency & Lightweight Study (RQ5)
7RWDO,QIHUHQFH7LPHV
this section cite: []

Section: Result Analysis.
We use GWNet as the backbone and compare ST-TTC with other test-time adaptation methods on METR-LA in terms of total inference time and memory usage. As shown in Figure 8, ST-TTC achieves the best overall efficiency (excluding the GWNet baseline), being 4.64× faster and reducing memory usage by 37.12% compared to the least efficient method, which is much smaller than the sliding size (5 min.), meeting the time requirement.
this section cite: []

Section: Conclusion
In this paper, we investigate the objectives of test-time computation in spatio-temporal forecasting and explore effective approaches for its implementation. We propose ST-TTC , a novel paradigm that uses a flash gradient update with streaming memory queue to learning a spectral-domain calibrator via phase-amplitude modulation, effectively addressing non-stationary errors. Extensive experiments confirm its effectiveness, universality, and flexibility. In future work, we aim to explore how to enhance the internal computational capacity of spatio-temporal foundation models during test time.
this section cite: []

Section: References
Ref_id:b0 Title: The surprising effectiveness of test-time training for few-shot learning Year: (2024)
Ref_id:b1 Title: Fast multivariate spatio-temporal analysis via low rank tensor learning Year: (2014)
Ref_id:b2 Title: Alleviating performance disparity in adversarial spatiotemporal graph learning under zero-inflated distribution Year: (2025)
Ref_id:b3 Title: Fundamental limitations of foundational forecasting models: The need for multimodality and rigorous evaluation Year: (2024)
Ref_id:b4 Title: Modeling and generating multivariate time-series input processes using a vector autoregressive technique Year: (2003)
Ref_id:b5 Title: Local learning algorithms Year: (1992)
Ref_id:b6 Title: Distribution of residual autocorrelations in autoregressiveintegrated moving average time series models Year: (1970)
Ref_id:b7 Title: Position: There are no champions in long-term time series forecasting Year: (2025)
Ref_id:b8 Title: Test-time training for spatial-temporal forecasting Year: (2024)
Ref_id:b9 Title: Expand and compress: Exploring tuning principles for continual spatio-temporal graph forecasting Year: (2025)
Ref_id:b10 Title: Trafficstream: A streaming traffic flow forecasting framework based on graph neural networks and continual learning Year: (2021)
Ref_id:b11 Title: Taming local effects in graph-based spatiotemporal forecasting Year: (2023)
Ref_id:b12 Title: Space-time modelling with an application to regional forecasting Year: (1975)
Ref_id:b13 Title: Statistics for spatio-temporal data Year: (2011)
Ref_id:b14 Title: St-norm: Spatial and temporal normalization for multi-variate time series forecasting Year: (2021)
Ref_id:b15 Title: Disentangling structured components: Towards adaptive, interpretable and scalable time series forecasting Year: (2024)
Ref_id:b16 Title: Efficient large-scale traffic forecasting with transformers: A spatial data management perspective Year: (2025)
Ref_id:b17 Title: Machine learning in space and time Year: (2015)
Ref_id:b18 Title: Learning by transduction Year: (1998)
Ref_id:b19 Title: Test-time training with masked autoencoders Year: (2022)
Ref_id:b20 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b21 Title: Online test-time adaptation of spatial-temporal traffic flow forecasting Year: (2024)
Ref_id:b22 Title: Attention based spatial-temporal graph convolutional networks for traffic flow forecasting Year: (2019)
Ref_id:b23 Title: Learning dynamics and heterogeneity of spatial-temporal graph data for traffic forecasting Year: (2021)
Ref_id:b24 Title: Bigst: Linear complexity spatiotemporal graph neural network for traffic forecasting on large-scale road networks Year: (2024)
Ref_id:b25 Title: Test-time training on nearest neighbors for large language models Year: (2024)
Ref_id:b26 Title: Wavelet diffusion neural operator Year: (2025)
Ref_id:b27 Title: Timekan: Kan-based frequency decomposition learning architecture for long-term time series forecasting Year: (2025)
Ref_id:b28 Title: Openai o1 system card Year: (2024)
Ref_id:b29 Title: Test-time adaptation via self-training with nearest neighbor information Year: (2023)
Ref_id:b30 Title: Seeing the unseen: Learning basis confounder representations for robust traffic prediction Year: (2025)
Ref_id:b31 Title: Spatio-temporal graph neural networks for predictive learning in urban computing: A survey Year: (2023)
Ref_id:b32 Title: A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection Year: (2024)
Ref_id:b33 Title: Thinking, fast and slow Year: (2011)
Ref_id:b34 Title: Team: Topological evolution-aware framework for traffic forecasting Year: (2024)
Ref_id:b35 Title: Spatio-temporal predictive modeling techniques for different domains: a survey Year: (2024)
Ref_id:b36 Title:  Year: ()
Ref_id:b37 Title: Fast and slow streams for online time series forecasting without information leakage Year: (2025)
Ref_id:b38 Title: Continual traffic forecasting via mixture of experts Year: (2024)
Ref_id:b39 Title: Urbanev: An open benchmark dataset for urban electric vehicle charging demand prediction. Scientific Data Year: (2025)
Ref_id:b40 Title: Diffusion convolutional recurrent neural network: Data-driven traffic forecasting Year: (2018)
Ref_id:b41 Title: Opencity: Open spatio-temporal foundation models for traffic prediction Year: (2024)
Ref_id:b42 Title: Urbangpt: Spatio-temporal large language models Year: (2024)
Ref_id:b43 Title: Flashst: A simple and universal prompt-tuning framework for traffic prediction Year: (2024)
Ref_id:b44 Title: A comprehensive survey on test-time adaptation under distribution shifts Year: (2025)
Ref_id:b45 Title: Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation Year: (2020)
Ref_id:b46 Title: Practical adversarial attacks on spatiotemporal traffic forecasting models Year: (2022)
Ref_id:b47 Title: Spatio-temporal adaptive embedding makes vanilla transformer sota for traffic forecasting Year: (2023)
Ref_id:b48 Title: Largest: A benchmark dataset for large-scale traffic forecasting Year: (2023)
Ref_id:b49 Title: Stop! a out-of-distribution processor with robust spatiotemporal interaction Year: (2024)
Ref_id:b50 Title: Introductory lectures on convex optimization: A basic course Year: (2013)
Ref_id:b51 Title: Climax: A foundation model for weather and climate Year: (2023)
Ref_id:b52 Title: A time series is worth 64 words: Longterm forecasting with transformers Year: (2023)
Ref_id:b53 Title: Wind speed forecasting using spatio-temporal indicators Year: (2012)
Ref_id:b54 Title: Fc-gaga: Fully connected gated graph architecture for spatio-temporal traffic forecasting Year: (2021)
Ref_id:b55 Title: Mémoire sur les séries et sur l'intégration complète d'une équation aux différences partielles linéaires du second ordre, à coefficients constants. Mém. prés. par divers savants Year: ()
Ref_id:b56 Title: A starima model-building procedure with application to description and regional forecasting Year: (1980)
Ref_id:b57 Title: Learning fast and slow for online time series forecasting Year: (2023)
Ref_id:b58 Title: Predicting spatio-temporal propagation of seasonal influenza using variational gaussian process regression Year: (2016)
Ref_id:b59 Title: Long-term spatio-temporal forecasting via dynamic multiple-graph attention Year: (2022)
Ref_id:b60 Title: Stemo: Early spatio-temporal forecasting with multi-objective reinforcement learning Year: (2024)
Ref_id:b61 Title: Exploring progress in multivariate time series forecasting: Comprehensive benchmarking and heterogeneity analysis Year: (2024)
Ref_id:b62 Title: Spatial-temporal identity: A simple yet effective baseline for multivariate time series forecasting Year: (2022)
Ref_id:b63 Title: Machine learning for spatiotemporal sequence forecasting: A survey Year: (2018)
Ref_id:b64 Title: Scaling llm test-time compute optimally can be more effective than scaling parameters for reasoning Year: (2025)
Ref_id:b65 Title: Spatial-temporal synchronous graph convolutional networks: A new framework for spatial-temporal network data forecasting Year: (2020)
Ref_id:b66 Title: Test-time training with selfsupervision for generalization under distribution shifts Year: (2020)
Ref_id:b67 Title: The nature of statistical learning theory Year: (1999)
Ref_id:b68 Title: Stone: A spatio-temporal ood learning framework kills both spatial and temporal shifts Year: (2024)
Ref_id:b69 Title: Towards dynamic spatialtemporal graph learning: A decoupled perspective Year: (2024)
Ref_id:b70 Title: Knowledge expansion and consolidation for continual traffic prediction with expanding graphs Year: (2023)
Ref_id:b71 Title: Pattern expansion and consolidation on evolving graphs for continual traffic prediction Year: (2023)
Ref_id:b72 Title: Distribution-aware online continual learning for urban spatio-temporal forecasting Year: (2024)
Ref_id:b73 Title: Tent: Fully test-time adaptation by entropy minimization Year: (2021)
Ref_id:b74 Title: Evaluating the generalization ability of spatiotemporal model in urban scenario Year: (2024)
Ref_id:b75 Title: Robust traffic forecasting against spatial shift over years Year: (2024)
Ref_id:b76 Title: A review of deep learning for renewable energy forecasting Year: (2019)
Ref_id:b77 Title: Continual test-time domain adaptation Year: (2022)
Ref_id:b78 Title: Test-time training on video streams Year: (2025)
Ref_id:b79 Title: Pm2. 5-gnn: A domain knowledge enhanced graph neural network for pm2. 5 forecasting Year: (2020)
Ref_id:b80 Title: Accuracy law for the future of deep time series forecasting Year: (2025)
Ref_id:b81 Title: Onenet: Enhancing time series forecasting models under concept drift by online ensembling Year: (2023)
Ref_id:b82 Title: Pastnet: Introducing physical inductive biases for spatio-temporal video prediction Year: (2024)
Ref_id:b83 Title: Graph wavenet for deep spatial-temporal graph modeling Year: (2019)
Ref_id:b84 Title: Deciphering spatiotemporal graph forecasting: A causal lens and treatment Year: (2023)
Ref_id:b85 Title: Spatiotemporal ego-graph domain adaptation for traffic prediction with data missing Year: (2024)
Ref_id:b86 Title: Spatial-temporal transformer networks for traffic flow forecasting Year: (2020)
Ref_id:b87 Title: Fits: Modeling time series with 10k parameters Year: (2024)
Ref_id:b88 Title: Fouriergnn: Rethinking multivariate time series forecasting from a pure graph perspective Year: (2023)
Ref_id:b89 Title: Get rid of isolation: A continuous multi-task spatio-temporal learning framework Year: (2024)
Ref_id:b90 Title: Spatio-temporal graph convolutional networks: a deep learning framework for traffic forecasting Year: (2018)
Ref_id:b91 Title: Bigcity: A universal spatiotemporal model for unified trajectory and traffic state data analysis Year: (2024)
Ref_id:b92 Title: Unist: A prompt-empowered universal model for urban spatio-temporal prediction Year: (2024)
Ref_id:b93 Title: Urbandit: A foundation model for open-world urban spatio-temporal learning Year: (2024)
Ref_id:b94 Title: Spatio-temporal few-shot learning via diffusive neural network generation Year: (2024)
Ref_id:b95 Title: Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint Year: (2025)
Ref_id:b96 Title: Are transformers effective for time series forecasting Year: (2023)
Ref_id:b97 Title: Automated spatio-temporal graph contrastive learning Year: (2023)
Ref_id:b98 Title: Spatial-temporal graph learning with adversarial contrastive adaptation Year: (2023)
Ref_id:b99 Title: What, how, where, and how well? a survey on test-time scaling in large language models Year: (2025)
Ref_id:b100 Title: Test-time compensated representation learning for extreme traffic forecasting Year: (2023)
Ref_id:b101 Title: Probabilistic traffic forecasting with dynamic regression Year: (2025)
Ref_id:b102 Title: Forecasting fine-grained air quality based on big data Year: (2015)
Ref_id:b103 Title: Coms2t: A complementary spatiotemporal learning system for data-adaptive model evolution Year: (2024)
Ref_id:b104 Title: Maintaining the status quo: Capturing invariant relations for ood spatiotemporal learning Year: (2023)
Ref_id:b105 Title: Online convex programming and generalized infinitesimal gradient ascent Year: (2003)
