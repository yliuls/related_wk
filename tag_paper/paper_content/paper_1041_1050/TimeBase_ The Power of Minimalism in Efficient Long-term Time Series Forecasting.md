Title: TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting
Abstract: Long-term time series forecasting (LTSF) has traditionally relied on large parameters to capture extended temporal dependencies, resulting in substantial computational costs and inefficiencies in both memory usage and processing time. However, time series data, unlike high-dimensional images or text, often exhibit temporal pattern similarity and low-rank structures, especially in long-term horizons. By leveraging this structure, models can be guided to focus on more essential, concise temporal data, improving both accuracy and computational efficiency. In this paper, we introduce TimeBase, an ultra-lightweight network to harness the power of minimalism in LTSF. TimeBase 1) extracts core basis temporal components and 2) transforms traditional point-level forecasting into efficient segment-level forecasting, achieving optimal utilization of both data and parameters. Extensive experiments on diverse real-world datasets show that TimeBase achieves remarkable efficiency and secures competitive forecasting performance. Additionally, TimeBase can also serve as a very effective plugand-play complexity reducer for any patch-based forecasting models. Code is available at https:  //github.com/hqh0728/TimeBase.

Section: Introduction
Long-term time series forecasting (LTSF) has been studied with significant interest in various domains, ranging from energy management, traffic accident preservation, and extreme disaster warning. With the rapid advancement of deep learning, an increasing number of models have been proposed (Qiu et al., 2024;Zheng et al., 2023;Ang et al., 2023;Chen et al., 2023;Wu et al., 2021;Zhou et al., 2023a;Wang et al., 2023b;Lin et al., 2023a;Ding et al., 2018;Chen et al., 2018;Wang et al., 2024c;Chen et al., 2019;Wang et al., 2023a;Huang et al., 2024c;Zhou et al., 2023b;Wang et al., 2024b;Lin et al., 2024a;Miao et al., 2025;Yi et al., 2024), including MLP-based (Liu et al., 2022;Huang et al., 2024a;Miao et al., 2024), RNN-based (Lin et al., 2023b), and Transformer-based (Liu et al., 2021b;Zhang and Yan, 2023), approaches, all of which employ thousands to millions of parameters to capture long-range dependencies and forecast future outcomes.
Generally, a higher number of parameters increases the model capacity, which can lead to better predictive performance (Zhou et al., 2023c;Zhao et al., 2024). In the fields of computer vision (CV) and natural language processing (NLP), large models have achieved significant success (He et al., 2016;Liu et al., 2023). For instance, Vision Transformers (ViT) (Dosovitskiy, 2020) have demonstrated outstanding capabilities in image recognition, while large language models (LLM) have made breakthrough advances across various language tasks (Devlin, 2018;Radford et al., 2019). Recently, large models are being explored for LTSF to capture complex temporal patterns and long-range dependencies (Cheng et al., 2024;Liu et al., 2025a;2024d;2025b). For example, some LLM-based forecasting methods are proposed with tens of billions of parameters (Jin et al., 2023). However, despite their impressive performance on specific forecasting tasks, these models suffer from high computational costs and resource-intensive requirements.
In fact, images and text, as high-dimensional data, contain multiple dependencies and complex underlying physical rules (Liu et al., 2021a), which necessitate the use of more parameters to model their rich semantic structures. However, as shown in Figure 1(a), one-dimensional time series data is typically much more regular, exhibiting obvious tem-  poral patterns. Moreover, in long-term time series data, this regularity can even manifest as low-rank characteristics (Liu et al., 2012), which can be reflected through singular value decomposition as shown in Figure 1(b), indicating that there is a considerable amount of redundant information (Jones and Brelsford, 1967;Hochreiter and Schmidhuber, 1997). This raises an important question: Is it truly necessary to employ such a large number of parameters to learn these regular time series patterns (Tan et al., 2024;Zuo et al., 2024;Khayati et al., 2024)?
In this study, we design an extremely lightweight time series forecasting network, TimeBase, which is centered around basis component extraction and segment-level forecasting. As illustrated in Figure 2, TimeBase utilizes only 0.39k parameters, reducing MACs by 120× and parameter count by more than 2.6 × 10 3 times compared to DLinear. Compared to the standard linear lightweight model SparseTSF, it reduces MACs by 4.5×, parameter count by 2.5×. Despite being a minimal model, TimeBase demonstrates superior predictive performance on real-world datasets of various domains and scales. Additionally, TimeBase can also serve as a very effective plug-and-play tool for patch-based forecasting methods, enabling extreme complexity reduction, i.e., 77.74%∼93.03% for PatchTST in MACs, prompting prediction accuracy. Our contributions can be summarized as follows:
• Considering the temporal pattern similarity and lowrank characteristic, we demonstrate that basis component extraction with segment-level forecasting is an effective approach for LTSF to fully utilize both data and model. This method can significantly reduce the unavoidable ultra-high complexity and large model parameters associated with current LTSF models.
• We propose TimeBase, which is currently the lightest time series forecaster and an effective plug-and-play complexity reducer. It requires only 0.39k parameters, achieving 4.5× reduction in MAC and a 2.5× decrease in the number of parameters compared to the previously lightest model SparseTSF. Besides, it could make 77.74%∼93.03% computation reduction for PatchTST.
• TimeBase not only maintains an extremely small model size but also achieves competitive forecasting performance across various real-world datasets. Specifically, TimeBase ranks Top2 on 29 out of 34 average metrics (MSE and MAE) across 17 normal scale datasets when compared to the ten state-of-the-art baselines.
• TimeBase offers a potential strategy for designing LTSF models with more efficiency and could provide valuable insights for the development of backbone architectures in large pre-trained LTSF models.
this section cite: ['b49', 'b66', 'b1', 'b6', 'b59', 'b11', 'b5', 'b4', 'b45', 'b61', 'b36', 'b44', 'b64', 'b65', 'b15', 'b39', 'b12', 'b10', 'b50', 'b7', 'b21', 'b35', 'b22', 'b16', 'b51', 'b72', 'b23']

Section: Related Work
Long-term time series forecasting (LTSF) aims to predict future sequences of considerable length using extended historical windows (Ang et al., 2024). The advancement of deep learning has significantly enhanced the accuracy of LTSF, with various foundational models, such as Transformers (Zhou et al., 2021;Zhang and Yan, 2023), Temporal Convolutional Networks (TCNs) (Luo and Wang, 2024), and Recurrent Neural Networks (RNNs) (Lin et al., 2023b), being employed to design long-term forecasting networks. These models are designed based on the different properties of time series, such as series decomposition (Wu et al., 2021), frequency domain (Xu et al., 2024), and periodic characteristics (Wu et al., 2023). As to series decomposition, for instance, Autoformer (Wu et al., 2021) introduces a series decomposition block that utilizes moving average techniques to decompose complex temporal variations into seasonal and trend components, each undergoing separate time series modeling. Additionally, FEDformer (Zhou et al., 2022) further enhances the representation capabilities of the series decomposition block by employing multiple kernels moving average to decompose data at various granularities, thereby improving forecasting performance. Considering the frequency domain characteristics of time series, FITS (Xu et al., 2024) operates on the principle that time series can be manipulated through interpolation in the complex frequency domain, achieving performance comparable to state-of-the-art models for time series forecasting. On the other hand, periodicity is a significant factor considered by many LTSF methods. TimesNet (Wu et al., 2023)proposes the use of Fourier Transform to capture multiple periodic lengths of time series, expanding one-dimensional time series into several two-dimensional components, which are processed through two-dimensional networks to handle high-dimensional data. SparseTSF (Lin et al., 2024b) directly utilizes the prior periodicity, thereby reducing the scale of network parameters. CrossGNN (Huang et al., 2024b) employs moving average techniques based on periodicity to expand single-granularity time series data into multi-granularity data, enriching the information contained within the dataset. In this paper, we propose TimeBase, which further leverages the approximate low-rank nature of long-term time series and significantly reducing the parameter scale.
this section cite: ['b67', 'b64', 'b43', 'b59', 'b38', 'b58', 'b59', 'b68', 'b38', 'b58']

Section: Method

this section cite: []

Section: Problem Definition
In LTSF, the objective is to predict future values over an extended time horizon based on very long look-back windows. Formally, let X = [x 1 , x 2 , ..., x T ] ∈ R T denote the historical time series data, where T ≫ 1 is the length of look-back window. The goal is to forecast the future values Y = [x T +1 , x T +2 , ..., x T +L ] ∈ R L with a forecasting horizon L ≫ 1. However, the exceptionally long horizon scale T and L substantially increases model size, leading to a rapid and considerable growth in the need of computation resources, which may be unnecessary for time series data that follow simple and regular patterns. Consequently, our focus shifts to designing models that not only deliver robust and efficient performance but also remain extremely lightweight.
this section cite: []

Section: TimeBase
In practical, regular time series often exhibit prominent segment-level patterns (Lin et al., 2024b), with approximate low-rank characteristics (Jones and Brelsford, 1967). For example, traffic flow typically follows a daily period, with similar patterns recurring each day. To effectively leverage time series data and accomplish efficient forecasting, we propose TimeBase, implemented through Basis Extraction and Segment-level Forecasting by two extremely smallscale linear layers. This approach drastically reduces the model parameters to the hundred level while maintaining state-of-the-art (SOTA) forecasting performance. Most existing multivariate time series (MTS) are homogeneous, meaning that each sequence within the dataset exhibits similar patterns. Based on this property, we employ the Channel Independence (Nie et al., 2023) to simplify the forecasting of MTS into separate univariate forecasting tasks. An overview of TimeBase is shown in Figure 3.
this section cite: ['b22', 'b46']

Section: BASIS COMPONENT RECONSTRUCTION
First, we need to determine the basis length P from the time series and segment the series accordingly. The determination process can be categorized into two scenarios:
(1) The time series has a predefined prior period much smaller than look-back window, for instance, in domains like electricity or traffic, the cyclic patterns often follow a daily periodicity, allowing us to directly assign P = 24 as the segment length from hourly sampled data. (2) When the time series lacks clear periodicity or the period exceeds It aims to demonstrate that even the most minimalistic models can exhibit strong predictive power, providing a design foundation for more effective time series models. In addition, TimeBase can also serve as a plug-and-play reducer to decrease the complexity of any patch-based models.
1/6 the input horizon, frequency analysis (e.g., FFT) (Wu et al., 2023) on the training set X train can help identify dominant components, whose wavelengths may serve as the basis length. In this case, a relatively smaller P is recommended to enhance the expressiveness of basis components.
Based on the predefined basis length P , we divide the one-dimensional time series X ∈ R T into N = T P non-overlapping segments, denoted as X his = [X 1 , X 2 , . . . , X N ] ∈ R N ×P , each of length P , analogous to non-overlapping patches (Nie et al., 2023). When the length of X N is insufficient to meet P , the corresponding values from X N -1 will be used to fill in the gaps. The segment operation can be represented as:
X his = Segment [N,P ] (X)(1)
where N and P in Segment [N,P ] (•) represent the number of rows and columns of the transformed 2D matrix. The maximized rank of the matrix X his is R max = min(N, P ). Given that typical time series exhibit similar temporal patterns and low-rank characteristics, we have R ≪ min(N, P ). In this context, directly designing a model for forecasting leads to unnecessary resource consumption. Fortunately, inspired from Basis Representation (Benson, 1998) and Linear Decomposition (Dantzig and Wolfe, 1960), a series fundamental temporal pattern can be identified, referred to as the basis components X basis ∈ R R×P to capture compact information and minimize the model size. Just as any vector in a coordinate system can be represented as a linear combination of its basis vectors, the combination of basis components in specific time series can represent its any segment-level temporal pattern (Hochreiter and Schmidhuber, 1997). Conversely, we can approximate the full-rank basis components using:
X basis = BasisExtract(X his )(2)
where BasisExtract(•) is implemented by a simple linear layer. Formally, X his can be expressed as a linear combination of basis components, represented as
X ⊤ his = X ⊤
basis W E +B, where W E ∈ R N ×R is the combined weight and the bias term B ∈ R R denote the temporal noise ϵ. By rearranging, we derive X T basis = X ⊤ his W † -BW † . Therefore the objective of Eq. ( 2) is to learn a linear layer of W his = W † and B his = -BW † to approximate the basis components. Next, we leverage the well-learned basis components to realize segment-level forecasting:
X pred = SegmentForecast(X basis )(3)
Here, X pred ∈ R N ′ ×P represents N ′ future segmented time series, where
N ′ = L P .
The operation SegmentForecast(•), implemented also through a linear layer, aggregates the basis components for forecasting. Finally, X pred is unfolded to obtain prediction result Y ∈ R L :
Y = Flatten(X pred ) 1:L(4)
this section cite: ['b58', 'b46', 'b2', 'b16']

Section: BASIS ORTHOGONAL RESTRICTION
To make that the learned X basis effectively captures the essential and diverse temporal patterns, an orthogonal constraint can be applied. From the perspective of the data space, the orthogonality of the basis vectors enhances its representation power, providing them ability to express as any vector in the data space through linear combination (Dantzig and Wolfe, 1960). Therefore, the temporal basis component should also be diverse and distinct, preventing the extraction of very single time-series patterns. Based on this, we apply the Basis Orthogonal Restriction.
Specifically, we penalize the deviation of X basis from an orthogonal set by adding a regularization loss L orth :
G = X ⊤ basis X basis (5
)
L orth = ∥G -diag(G)∥ 2 F (6
)
where G is the gram matrix of X basis and ∥ • ∥ F denotes the Frobenius norm. This term encourages X basis to approach an orthogonal configuration, ensuring that each basis component captures unique and uncorrelated temporal patterns.
The overall training objective is then updated to:
L = L prediction + λ orth L orth (7
)
Here, L prediction represents the original prediction loss, i.e., mean squared error (MSE) for regression, and λ orth is a hyperparameter that controls the weight of the orthogonal regularization term.
this section cite: ['b8']

Section: Parameter Scale of TimeBase
Theorem 3.1 (Parameter Scale of TimeBase). Let T denote the length of look-back window, L is the length of the forecast, P represents the length of the segment, and R gives the number of basis components. The parameter scale of TimeBase can be expressed as:
Number = R P a ×T + R + 1 P b ×L + R(8)
Theorem 3.1 shows that the parameter scale of TimeBase grows linearly with both the look-back window length T and the forecast horizon L. In a typical long-term forecasting setup where T = L = 720, the number of parameters in TimeBase, with a complexity of O(aT +bL), is significantly smaller than that of DLinear (Zeng et al., 2023), which requires 2T L parameters, and SparseTSF (Lin et al., 2024b), which uses T L P 2 + P , both with O(T L) complexity.
this section cite: ['b62']

Section: Experiment
In this section, we demonstrate the advantages of Time-Base in competitive forecasting performance, extremely light efficiency and very effective plug-and-play function.
More experiment details and additional experiment results are available at Appendix C.
this section cite: []

Section: Experiment Setup Datasets
We conduct experiments on 21 widely-used and publicly available real-world datasets, including 17 normal-scale benchmarks: ETTh1, ETTh2, ETTm1, ETTm2foot_0 , Weatherfoot_1 , Electricityfoot_2 , Trafficfoot_3 ,Solar Energy (Lai et al., 2018), Wind (Li et al., 2022), , METR-LA (Li et al., 2017), Exchange Rate (Lai et al., 2018), ZafNoo (Poyatos et al., 2020) and CzeLan (Poyatos et al., 2020), AQShunyi (Zhang et al., 2017), AQWan (Zhang et al., 2017), and 4 very large datasets: CA (4.52B), GLA (2.02B), GBA (1.24B),SD (0.38B) (Liu et al., 2024c). Adhering to the established protocol in (Wu et al., 2021;Qiu et al., 2024;Liu et al., 2024c), we partition the datasets into training, validation, and test sets with a ratio of 6:2:2 for four ETT datasets, CA, GLA, GBA, SD, and 7:1:2 for the remaining datasets. The statics of dataset is summarized in Table 1. Baselines We compare TimeBase with 10 baselines, which comprise the SOTA long-term forecasting models: TimeMixer (Wang et al., 2024d), iTransformer (Liu et al., 2024a), PatchTST (Nie et al., 2023), DLinear (Zeng et al., 2023), TimesNet (Wu et al., 2023), FEDformer (Zhou et al., 2022), Autoformer (Wu et al., 2021), and Informer (Zhou et al., 2021), relatively efficient models: FITS (Xu et al., 2024) and SparseTSF (Lin et al., 2024b).
this section cite: ['b25', 'b26', 'b27', 'b25', 'b48', 'b63', 'b63', 'b59', 'b49', 'b46', 'b62', 'b58', 'b68', 'b59', 'b67', 'b38']

Section: Forecsating Performance of TimeBase
As shown in Table 2, across 17 normal-scale benchmarks, TimeBase achieves Top-2 performance on 16 datasets, with the only exception being ETTh2, where its accuracy still remains highly competitive. These results further demonstrate the robustness and effectiveness of TimeBase across a wide range of forecasting tasks. Compared to the standard linear model DLinear, TimeBase reduces MSE and MAE by 8.82% and 7.64%, respectively. When compared with the currently most lightweight models, FITS and SparseTSF, TimeBase achieves an average reduction of 6.15∼6.34% in MSE and 4.85∼5.44% in MAE. Furthermore, against some of the most powerful models to date, such as TimeMixer, iTransformer, and PatchTST, TimeBase achieves the best results on 23 out of 34 average forecasting metrics and ranks second on 6, when carefully tuned. Despite its extremely compact size, TimeBase delivers competitive forecasting accuracy, thanks to its compact yet effective modeling design.
this section cite: []

Section: Efficiency Analysis
Main Efficiency Comparision In addition to its impressive predictive performance, TimeBase offers another major advantage: its exceptionally lightweight design. Here, we provide a more comprehensive comparison, examining both static and runtime metrics, which include (1) Parameters:
The total number of trainable parameters, reflecting the model's size. The look-back window for each model are set as 720, and the max memory is recorded with a constant batch size of 12. The efficiency analysis presented in Table 3 highlights the remarkable advantages of TimeBase over other state-of-the-art models in terms of both static and runtime metrics. TimeBase achieves a substantial reduction in the number of parameters and computational complexity compared to many parameter-heavy models. Specifically, compared to lightest model SparseTSF, TimeBase reduces the parameter count by up to 61% and the MACs by over 78%, while also using significantly less memory (29% reduction) and training faster (34% reduction in epoch time). These results further demonstrate that TimeBase not only maintains strong predictive performance but also offers superior efficiency, making it well-suited for resource-constrained environments.
Efficiency in Ultra-long Look-back Window Additionally, we evaluate the efficiency of TimeBase under ultralong look-back windows, comparing it with DLinear. The comparison focuses on three key metrics: running time per iteration, GPU memory usage, and parameter count, as shown in Figure 4. As the look-back window increases from 720 to 6480, with a fixed batch size of 12 and prediction length of 720, TimeBase consistently demonstrates its lightweight nature. Even with a ninefold increase in input sequence length, TimeBase's running time only increases by 0.05 seconds, GPU memory usage expands by a factor of 3.8, and the number of parameters grows by only 3.1 times. These results highlight the model's extreme efficiency and scalability in handling ultra-long sequences.
this section cite: []

Section: Performance on Large-Scale Datasets
To further validate the robustness of TimeBase under extreme conditions, we conduct experiments on four largescale datasets. As shown in Table 5, TimeBase consistently outperforms or matches strong baselines across all forecasting horizons, underscoring its effectiveness in handling large-scale forecasting tasks. Notably, TimeBase ranks within the top two across all horizons, highlighting its advantages in both prediction quality and computational efficiency. Compared to state-of-the-art models such as iTransformer, DLinear, and PatchTST, TimeBase achieves lower forecasting errors while significantly reducing computational cost, as reflected by its much lower Multiply-Accumulate Operations (MACs). Even on the largest dataset, CA with 8600 variables, TimeBase requires only 42.11M∼74.30M MACs, showcasing its exceptional capability in scenarios with limited computational resources or extremely large input scales. To ensure a fair comparison in terms of forecasting accuracy, computational complexity, and parameter count, we set the input sequence length to 720 for both PatchTST and PATCHTST (W/ TIMEBASE).
this section cite: []

Section: Quantitative Analysis of TimeBase Integration As shown in
Table 4, PATCHTST (W/ TIMEBASE) achieves comparable or even slightly improved forecasting accuracy, measured by MSE and MAE, while significantly reducing both parameter count and computational cost. Specifically, across 56 forecasting metrics on 7 datasets, PATCHTST (W/ TIMEBASE) outperforms the original PatchTST on 43 metrics, while the remaining 13 show only a marginal average degradation of 1.54%, which can be considered negligible. In terms of overall performance, PATCHTST (W/ TIME-BASE) reduces MSE and MAE by 1.27% and 1.13%, respectively, indicating that the basis extraction does not harm predictive power, and may even offer slight improvements. On the efficiency front, TimeBase enables dramatic reductions: MACs are reduced by 77.74% to 93.03%, and the
this section cite: []

Section: Hyperparameter Analysis
This section explores the impact of two key hyperparameters on the performance of TimeBase: the number of basis functions R and the orthogonal loss weight λ orth . Fig-   ure 5 shows the MSE results as the orthogonal loss weight λ orth is varied across [0.00, 0.04, 0.08, 0.12, 0.16, 0.20]. For datasets such as ETTh1, and ETTh2, prediction performance fluctuates with different values of λ orth . However, for Traffic and Electricity, the performance remains relatively stable. Overall, an appropriate value of λ orth can enhance forecasting performance, which are varied across different datasets. In Figure 6, the number of basis number R is varied from [2,4,6,12,18,24,30], and the corresponding MSE results for Traffic, Electricity datasets are reported. The results show that using too few basis components (e.g., R = 2) leads to a noticeable performance drop. However, once R exceeds a modest threshold (typically R ≥ 4), the model performance stabilizes. These findings indicate that TimeBase can achieve strong forecasting performance with a small number of basis components. In practice, we typically set R = 6 for most datasets. For datasets with more complex temporal patterns, a larger R is recommended to enhance the expressive capacity of the basis components.
this section cite: []

Section: Conclusion
Considering the temporal pattern similarity and low-rank characteristics, we design TimeBase, the lightest known model for long-term forecasting, with only 0.39K parameters, 2.77M MACs computation, 88.89M memory usage, and a CPU inference speed of 0.98ms. This demonstrates that even the most minimalist models can achieve strong predictive performance, providing a design foundation for more efficient time series models. Furthermore, TimeBase can function as a plug-and-play tool to reduce the complexity of any patch-based models. This approach (1) enables long-term time series forecasting models to be deployed on resource-constrained edge devices, and (2) offers new insights for lightweight model designs, encouraging time series researchers to fully leverage sequential data and inspiring the development of backbone networks for pre-trained large LTSF models.
this section cite: []

Section: References
Ref_id:b0 Title: M ↓88.90% 0.83M ↓80 Year: (2024-11)
Ref_id:b1 Title: TSGBench: Time Series Generation Benchmark Year: (2023-11)
Ref_id:b2 Title: Representations and cohomology: Volume 1, basic representation theory of finite groups and associative algebras Year: (1998)
Ref_id:b3 Title: The analysis of time series: theory and practice Year: (2013)
Ref_id:b4 Title: Real-time distributed co-movement pattern detection on streaming trajectories Year: (2019)
Ref_id:b5 Title: Price-and-time-aware dynamic ridesharing Year: (2018)
Ref_id:b6 Title: ImDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection Year: (2023-11)
Ref_id:b7 Title: Weakly Guided Adaptation for Robust Time Series Forecasting Year: (2024-03)
Ref_id:b8 Title: Decomposition principle for linear programs Year: (1960)
Ref_id:b9 Title: Long-term Forecasting with TiDE: Time-series Dense Encoder. Transactions on Machine Learning Research Year: (2024)
Ref_id:b10 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b11 Title: UlTraMan: A unified platform for big trajectory data management and analytics Year: (2018)
Ref_id:b12 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b13 Title: Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting Year: (2023)
Ref_id:b14 Title: Matrix computations Year: (2013)
Ref_id:b15 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b16 Title: Long short-term memory Year: (1997)
Ref_id:b17 Title: Matrix analysis Year: (2012)
Ref_id:b18 Title: 2024a. HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate Time Series Forecasting Year: ()
Ref_id:b19 Title: 2024b. Crossgnn: Confronting noisy multivariate time series via cross interaction refinement Year: (2024)
Ref_id:b20 Title: Leret: Languageempowered retentive network for time series forecasting Year: (2024)
Ref_id:b21 Title: Time-llm: Time series forecasting by reprogramming large language models Year: (2023)
Ref_id:b22 Title: Time series with periodic structure Year: (1967)
Ref_id:b23 Title: ImputeVIS: An Interactive Evaluator to Benchmark Imputation Techniques for Time Series Data Year: (2024-11)
Ref_id:b24 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b25 Title: Modeling long-and short-term temporal patterns with deep neural networks Year: (2018)
Ref_id:b26 Title: Generative time series forecasting with diffusion, denoise, and disentanglement Year: (2022)
Ref_id:b27 Title: Diffusion convolutional recurrent neural network: Data-driven traffic forecasting Year: (2017)
Ref_id:b28 Title: Mts-mixers: Multivariate time series forecasting via factorized temporal and channel mixing Year: (2023)
Ref_id:b29 Title: 2024a. Cyclenet: enhancing time series forecasting through modeling periodic patterns Year: (2024)
Ref_id:b30 Title: 2024b. SparseTSF: Modeling Long-term Time Series Forecasting with* 1k* Parameters Year: ()
Ref_id:b31 Title: Ruichao Mo, and Haotong Zhang. 2023b. Segrnn: Segment recurrent neural network for long-term time series forecasting Year: (2023)
Ref_id:b32 Title: 2025a. Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation Year: ()
Ref_id:b33 Title: TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment Year: (2025)
Ref_id:b34 Title: Spatial-Temporal Large Language Model for Traffic Prediction Year: (2024)
Ref_id:b35 Title: Robust recovery of subspace structures by low-rank representation Year: (2012)
Ref_id:b36 Title: Scinet: Time series modeling and forecasting with sample convolution and interaction Year: (2022)
Ref_id:b37 Title: Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting Year: (2021)
Ref_id:b38 Title: Largest: A benchmark dataset for large-scale traffic forecasting Year: (2024)
Ref_id:b39 Title: Summary of chatgpt-related research and perspective towards the future of large language models Year: (2023)
Ref_id:b40 Title: 2024a. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting Year: ()
Ref_id:b41 Title: 2024b. Koopa: Learning non-stationary time series dynamics with koopman predictors Year: (2024)
Ref_id:b42 Title: 2021a. Swin transformer: Hierarchical vision transformer using shifted windows Year: ()
Ref_id:b43 Title: ModernTCN: A modern pure convolution structure for general time series analysis Year: (2024)
Ref_id:b44 Title: Less is more: Efficient time series dataset condensation via two-fold modal matching Year: (2024)
Ref_id:b45 Title: A Parameter-Efficient Federated Framework for Streaming Time Series Anomaly Detection via Lightweight Adaptation Year: (2025)
Ref_id:b46 Title: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers Year: (2023)
Ref_id:b47 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b48 Title: Global transpiration data from sap flow measurements: the SAPFLUXNET database Year: (2020)
Ref_id:b49 Title: TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods. Proc. VLDB Endow Year: (2024-08)
Ref_id:b50 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b51 Title: Are Language Models Actually Useful for Time Series Forecasting? arXiv preprint Year: (2024)
Ref_id:b52 Title: 2024c. Towards Dynamic Spatial-Temporal Graph Learning: A Decoupled Perspective Year: ()
Ref_id:b53 Title: 2024b. Conditionguided urban traffic co-prediction with multiple sparse surveillance data Year: (2024)
Ref_id:b54 Title: 2023a. Knowledge Expansion and Consolidation for Continual Traffic Prediction With Expanding Graphs Year: (2023)
Ref_id:b55 Title: 2023b. Pattern Expansion and Consolidation on Evolving Graphs for Continual Traffic Prediction Year: ()
Ref_id:b56 Title: General Time Series Pattern Machine for Universal Predictive Analysis Year: (2024)
Ref_id:b57 Title: TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting Year: (2024)
Ref_id:b58 Title: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis Year: (2023)
Ref_id:b59 Title: Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting Year: (2021)
Ref_id:b60 Title: FITS: Modeling Time Series with 10k Parameters Year: (2024)
Ref_id:b61 Title: Get Rid of Isolation: A Continuous Multi-task Spatio-Temporal Learning Framework Year: (2024)
Ref_id:b62 Title: Are transformers effective for time series forecasting Year: (2023)
Ref_id:b63 Title: Cautionary tales on air-quality improvement in Beijing Year: (2017)
Ref_id:b64 Title: Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting Year: (2023)
Ref_id:b65 Title: Multiple Time Series Forecasting with Dynamic Graph Modeling Year: (2024-03)
Ref_id:b66 Title: DecLog: Decentralized Logging in Non-Volatile Memory for Time Series Database Systems Year: (2023-09)
Ref_id:b67 Title: Informer: Beyond efficient transformer for long sequence time-series forecasting Year: (2021)
Ref_id:b68 Title: FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting Year: (2022)
Ref_id:b69 Title: 2023c. One Fits All: Power General Time Series Analysis by Pretrained LM Year: (2023)
Ref_id:b70 Title: 2023a. GReTo: Remedying dynamic graph topology-task discordance via target homophily Year: ()
Ref_id:b71 Title: Maintaining the status quo: Capturing invariant relations for ood spatiotemporal learning Year: (2023)
Ref_id:b72 Title: DARKER: Efficient Transformer with Data-Driven Attention Mechanism for Time Series Year: (2024-08)
