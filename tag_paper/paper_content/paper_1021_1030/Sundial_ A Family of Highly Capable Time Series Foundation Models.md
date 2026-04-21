Title: Sundial: A Family of Highly Capable Time Series Foundation Models
Abstract: We introduce Sundial, a family of native, flexible, and scalable time series foundation models. To predict the next-patch's distribution, we propose a TimeFlow Loss based on flow-matching, which facilitates native pre-training of Transformers on continuous-valued time series without discrete tokenization. Conditioned on arbitrary-length time series, our models are pre-trained without specifying any prior distribution and can generate multiple probable predictions, achieving more flexibility in representation learning than using parametric densities. Towards time series foundation models, we leverage minimal but crucial adaptations of Transformers and curate TimeBench with one trillion time points, comprising mostly real-world datasets and synthetic data. By mitigating mode collapse via TimeFlow Loss, we pre-train a family of Sundial models on TimeBench, which achieve unprecedented model capacity and generalization performance. In addition to excellent scalability, Sundial achieves state-of-the-art results on both point and probabilistic forecasting benchmarks with a just-in-time inference speed, i.e., making zero-shot predictions within a few milliseconds. We believe that Sundial's pioneering generative forecasting capability can improve model reliability in real-world decision-making. Code is available at: https://github.com/thuml/Sundial.

Section: Introduction
Time series forecasting has fascinated people for thousands of years. Although people have been able to determine the time using instruments like sundials in 3000 BC, time series forecasting is intrinsically non-deterministic (Box et al., 2015). Therefore, generating a variety of probable predictions is crucial for decision-making. The growing demand has facilitated numerous statistical approaches over the past decades (Hyndman, 2018;Box, 2013), which provide highprofile theories and probabilistic tools for making reliable schedules. Recent advancements bring the boom of deftly designed models that automatically learn intricate dynamics and correlations from raw data (Oreshkin et al., 2019;Nie et al., 2022;Zhang & Yan, 2023;Liu et al., 2023a). Despite the impressive performance, deep models necessitate taskspecific training on sufficient in-distribution data. Motivated by advances in large models (Bommasani et al., 2021), pretrained time series foundation models have shown promising capabilities in out-of-distribution tasks (Das et al., 2023b;Liu et al., 2024b;Woo et al., 2024;Ansari et al., 2024).
Current research on time series foundation models has converged on building unified, scalable, and out-of-the-box forecasters, exhibiting zero-shot performance close to or sometimes surpassing supervised methods (Aksu et al., 2024). Notably, Transformers (Radford et al., 2018) are currently the de facto architecture of these models. While pre-trained Transformers with an inherent generative ability have facilitated great success in language, image, and video generation (Ramesh et al., 2021;OpenAI, 2023;Liu et al., 2024c), most time series foundation models are not "generative" or, more specifically, probabilistic forecasters, thereby limiting reliability in decision-making. Although parametric densities specified with prior distributions (Wen et al., 2017;Woo et al., 2024) can be adopted to address uncertainty in time series forecasting, they can reduce the capacity of distributions learned by pre-trained models, especially on time series modality characterized by high heterogeneity. To learn arbitrarily intricate distributions without mode collapse, language modeling (Bengio et al., 2000) that learns the categorical distribution via cross-entropy loss inspires subsequent works (Gruver et al., 2023;Ansari et al., 2024), which treat time series as a foreign language using discrete tokenization. Still, discrepancies between continuous-valued time series and discrete language tokens can lead to out-of-vocabulary issues and coarse-grained prediction intervals.
As shown in Figure 1, Sundial is presented as the first family of generative models among time series foundation models. As foundation models intend to learn complicated distributions from extensive datasets and facilitate transferability across agnostic downstream datasets, we do not specify any prior parametric densities, such as unimodal and multimodal Gaussian mixtures. Instead, we delve into generative modeling to tame Transformers as native, flexible, and scalable time series foundation models. By comparing to denoising diffusion models (Li et al., 2024), we opt for a simple yet effective flow-matching framework (Lipman et al., 2022), which provides notable efficiency and sample quality (Tong et al., 2023). We propose TimeFlow Loss, a parameterized training objective (Zhang et al., 2018) for autoregressive models to learn and sample from each token's predictive distribution. Optimizing models in the original continuousvalued domain, TimeFlow Loss facilitates patch-level generation and enables fast inference, which is naturally compatible with the time series modality.
In addition to TimeFlow, we enhance the Transformer with minimal but critical adaptations. We develop feasible patch tokenization for arbitrary-length input time series. We adopt RoPE (Su et al., 2024), Pre-LN (Xiong et al., 2020), FlashAttention (Dao et al., 2022), and KV Cache (Pope et al., 2023), which are crucial but generally neglected in the development of time series foundation models. Besides, we pre-train our models by multi-patch prediction to reduce autoregression steps. We realize a rapid generation of multiple samples by reusing a shared lookback representation. Beyond facilitating scalable pre-training, these adaptations help real-time long-context inference and long-term generation.
To validate the scaling law of time series foundation models, we collect and curate TimeBench with an unprecedented volume of a trillion time points. We present Sundial as a family of highly capable foundation models, which achieve state-ofthe-art on three large-scale and best-recognized benchmarks, including Time-Series-Library (TSLib) (Wu et al., 2022), GIFT-Eval (Aksu et al., 2024), and FEV (Ansari et al., 2024).
Our contributions lie in these aspects:
• We propose TimeFlow Loss to predict next-patch's distribution, allowing Transformers to be trained without discrete tokenization and make probable predictions.
• We present Sundial, a family of scalable and efficient time series foundation models built upon our enhanced Transformer and pre-trained on a trillion time points.
• Experimentally, Sundial achieves state-of-the-art zeroshot performance on point and probabilistic forecasting benchmarks, including TSLib, GIFT-Eval, and FEV, indicating a promising generative approach for the future improvement of time series foundation models.
this section cite: ['b16', 'b5', 'b34', 'b32', 'b59', 'b4', 'b30', 'b55', 'b1', 'b0', 'b38', 'b39', 'b33', 'b54', 'b55', 'b3', 'b13', 'b1', 'b21', 'b24', 'b51', 'b59', 'b49', 'b58', 'b7', 'b37', 'b57', 'b0', 'b1']

Section: Related Work

this section cite: []

Section: Time Series Forecasting
Forecasting is essential for decision-making. Advancements in deep learning for time series include theory-inspired deep modules (Wu et al., 2021;Liu et al., 2023b;Wu et al., 2022), architecture-oriented adaptations (Bai et al., 2018;Salinas et al., 2020;Lim et al., 2021), and time series preprocessing (Kim et al., 2021;Nie et al., 2022). Deep models learn the dataset-level distribution and benefit from strong generalization and model capacity. Statistical methods conduct case-by-case fitting on input series, achieving notable performance on small data (Ke et al., 2017;Hyndman, 2018).
One of the efforts towards more capable forecasters focuses on the foundation models (Bommasani et al., 2021), which address data-scarce scenarios by pre-training. More capable models support zero-shot forecasting, making inferences as fast as statistical methods and possessing large model capacity as deep models. Another aspect is to address uncertainty in time series forecasting. There is a growing research emphasis on probabilistic forecasting (Woo et al., 2024;Ansari et al., 2024). While parametric densities can be adopted as training objectives of probabilistic forecasting, they can be too specific to meet the heterogeneity of large-scale datasets, resulting in mode collapse in representation learning and over-smooth predictions (Figure 14-15). In this work, we introduce generative time series foundation models, which naturally address the uncertainty in forecasting.
this section cite: ['b56', 'b57', 'b3', 'b43', 'b23', 'b18', 'b32', 'b17', 'b16', 'b4', 'b55', 'b1']

Section: Time Series Foundation Models
Recent research has concentrated on building versatile large time series models (Liang et al., 2024). With the advances made in large language models, Transformer has become the dominant architecture. Several works adapt Transformers to address the unique 2D-dimensionality and heterogeneity of time series (Woo et al., 2024;Liu et al., 2024a). Specifically, our work delves into tokenization and optimization.
Models such as TimesFM (Das et al., 2023b), Timer (Liu et al., 2024a;b), and Time-MoE (Shi et al., 2024b) embed continuous values and fit unimodal distributions via MSE or quantile loss (Wen et al., 2017). However, prior loss may result in mode collapse because predictive distributions are highly divergent across different domains. Besides, these models cannot provide the confidence level of predictions, limiting reliability for decision-making. Based on continuous tokenization, Moirai (Woo et al., 2024) presents a probabilistic model learning a mixture of distributions, but this prior can still fail to accommodate complex distributions. Inspired by language modeling, Chronos (Ansari et al., 2024) discretizes series via bucket quantization, learning more flexible categorical distributions by cross-entropy. Still, discrete tokenization is applied at each time point, which can lead to long contexts. Also, the final performance can be sensitive to quantization techniques. Unlike before, we tame Transformers as native time series foundation models, learning flexible distributions without discrete tokenization.
this section cite: ['b22', 'b55', 'b54', 'b55', 'b1']

Section: Generative Modeling for Time Series
By addressing complicated distributions during pre-training, generative modeling has become a focal point in the development of various foundation models (Zhao et al., 2023;Liu et al., 2024c). While this direction for time series mostly focused on time series generation (Tashiro et al., 2021) and task-specific forecasters (Rasul et al., 2021;Shen & Kwok, 2023;Kollovieh et al., 2024), generative modeling for time series foundation models is hardly explored. With the comparable flexibility in distribution learning as language modeling, diffusion denoising (Sohl-Dickstein et al., 2015) and flow-matching (Lipman et al., 2022) have gained increasing prevalence in continuous-valued modalities (Lipman et al., 2024). Compared with diffusion denoising models, flowmatching provides a simple yet efficient framework. With fewer steps involved in the forward and reverse processes, large models based on flow-matching have shown superior performance in image generation (Esser et al., 2024).
Despite the connection in value continuity, generating images and future time series are fundamentally different tasks due to the autoregressive property of forecasting. Our proposed TimeFlow Loss is designed for autoregressive models to conduct conditional generation, which is a parameterized loss function (Zhang et al., 2018) for arbitrary distributions and enhances representation learning of foundation models.
this section cite: ['b60', 'b50', 'b41', 'b44', 'b20', 'b48', 'b24', 'b25', 'b10', 'b59']

Section: Preliminaries

this section cite: []

Section: Flow-Matching
The goal of generative modeling is to learn the underlying probability distribution that generates the data. The framework of flow-matching transforms a sample x 0 ∼ p 0 drawn from a source distribution into a sample x 1 ∼ p 1 drawn from a target distribution. The transformation is continuous in time. For d-dimensional distributions, it is defined by a time-dependent velocity field u t : [0, 1] × R d → R d , which is the solution of the ordinary differential equation (ODE):
d dt ψ t (x) = u t ψ t (x) and ψ 0 (x) = x.
The velocity field u t determines a flow ψ t . For all t ∈ [0, 1], ψ t generates the probability path p t that interpolates p 0 and p 1 , i.e., x t = ψ t (x 0 ) ∼ p t for x 0 ∼ p 0 . The implementation of flow-matching is to train a network u θ t parametrized by θ to fit the velocity field u t , which is a regression-based task formulated as the Flow-Matching objective:
L FM (θ) = E t,xt u θ t (x t ) -u t (x t ) 2 .
Furthermore, Lipman et al. (2022) proved the equivalence of optimizing the Conditional Flow-Matching objective:
L CFM (θ) = E t,xt,x1 u θ t (x t ) -u t (x t |x 1 ) 2 .
Leveraging the conditional optimal-transport (linear) path and a source Gaussian, the objective can be formulated as:
L Gauss CFM (θ) = E t,ϵ,x1 u θ t (x t ) -(x 1 -x 0 ) 2 . (1
)
where t ∼ U[0, 1], x 0 ∼ N (0, 1) and x t = tx 1 + (1 -t)ϵ.
Consequently, we can train a generative network on given samples from the target distribution, and generate new samples by applying a push-forward process on samples drawn from a simple source Gaussian distribution:
x t+∆t -x t = u θ t (x t )∆t, x 0 ∼ N (0, I), t ∈ [0, 1]. (2)
this section cite: ['b24']

Section: Generative Models for Probabilistic Forecasting
Given a historical observation x 1:t = {x 1 , ..., x t }, the target of time series forecasting is to predict future time series x t+1:t+f = {x t+1 , . . . , x t+f }. The task can be generally formulated as p(x t+1:t+f |h t ), where h t = f ϕ (x 1:t ) is the learned representation from a deep model f ϕ . In probabilistic forecasting, explicit optimization objectives are utilized to predict the statistics of future series, e.g., MSE or quantile loss, which have specified p as a prior distribution. While using one parametric density generally fits well on a small amount of data, it can be the major bottleneck for scaling time series foundation models. Inspired by the success of large generative models (Rombach et al., 2022;OpenAI, 2023;Esser et al., 2024), we introduce generative modeling to realize probabilistic forecasting:
p(x t+1:t+f |h t ) = g θ f ϕ (x 1:t ) .(3)
g θ is a small trainable generative network conditioned on the learned representations of f ϕ , which is jointly optimized   with f ϕ . While the generative model automatically fits the target distribution, it can sample raw predictions and calculating their statistics for probabilistic forecasting. The aim is conceptually related to conformal prediction (Vovk et al., 2005) but models uncertainty beyond prediction intervals.
E q f B Y z Y v 8 S D R d J 2 E + D 1 l N c u m z Z i y Y E 7 g + a 7 i j S x V v 3 D C R 8 C i 8 l u O Y d Q J n E P I + 9 x x J V L s d O H L o 9 i f D a Z d 3 C 0 W r Z O l l z g M 7 A 0 V k q x I V X t B G D x E 8 p A j A E E I S 9 u E g o a c F G x Z i 4 j q Y E C c I c R 1 n m C J P 2 p S y G G U 4 x I 7 o O 6 B d K 2 N D 2 i v P R K s 9 O s W n V 5 D S x C F p I s o T h N V p p o 6 n 2 l m x v 3 l P t K e 6 2 5 j + b u Y V E C s x J P Y v 3 S z z v z p V i 0 Q f 5 7 o G T j X F m l H V e Z l L q r u i b m 5 + q U q S Q 0 y c w j 2 K C 8 K e V s 7 6 b G p N o m t X v X V 0 / E 1 n K l b t v S w 3 x b u 6 J Q 3 Y / j n O e V A / L t m n J b t 6 U i x f Z K P O Y R 8 H O K J 5 n q G M K 1 R Q I + 8 Y j 3 j C s 1 E 1 b o 0 7 4 / 4 z 1 V j I N H v 4 t o y H D 9 O U l G 8 = < / l a t e x i t > h i < l a t e x i t s h a 1 _ b a s e 6 4 = " 7 2 c Q L K E U G b U y k f J i B A 1 V 1 A 9 M R S c = " > A A A C 1 3 i c j V H L S s N A F D 2 N r 1 p f t S 7 d B I u g m 5 K I q M u i G 5 c K 1 l Z 8 l E k 6 1 c G 8 m E z
i i R N q M s T h m M 2 E v 6 n t P u O G c j 2 m v P 1 K h 9 O i W g V 5 L S x j J p Y s q T h P V p t o l n x l m z v 3 n 3 j a e + W 4 / + X u 4 V E q t w Q e x f u m H m f 3 W 6 F o U u t k w N g m p K D K O r 8 3 O X z H R F 3 9 z + V J U i h 4 Q 4 j T s U l 4 R 9 o x z 2 2 T a a 1 N S u e 8 t M / N V k a l b v / T w 3 w 5 u + J Q 3 Y / T 7 O n + B w r e Z u 1 N z 9 9 W p 9 O x 9 1 E Y t Y w g r N c x N 1 7 G I P D f K + x g M e 8 W Q d W T f W r X X 3 k W o V c s 0 C v i z r / h 2 V /
this section cite: ['b42', 'b33', 'b10', 'b52']

Section: Approach
In this work, we conduct a univariate pre-training paradigm, which adopts the S3 format proposed by Liu et al. (2024b) to address multivariate data. To mitigate value range discrepancy, we conduct normalization on time series individually per variable. Afterwards, we sample varying-length training samples with the maximum context length of 2880. As a foundation model, Sundial is required to predict on out-ofdistribution series with varied lengths during inference.
this section cite: ['b30']

Section: Sundial
As shown in Figure 2, the Sundial models consist of three parts: (1) time series tokenization, including a context-level re-normalization and a patch embedding that addresses anylength time series, (2) a Transformer backbone that learns the per-token representation of time series, and (3) Time-Flow Loss, a parameterized loss function to model the pertoken distribution and generate raw series during inference. Intuitively, Sundial can be regarded as an ARMA (Auto-Regression and Moving-Average) deep model, i.e., Transformer learns token representations autoregressively. Conditioned on the lookback representations, TimeFlow transforms random noises into non-deterministic predictions.
this section cite: []

Section: TIME SERIES TOKENIZATION
Re-Normalization We adopt stationarization (Liu et al., 2022), a non-parametric two-stage instance normalization conducted within each sample, which is initially proposed to mitigate non-stationarity of time series. Here, it helps to ad-dress temporal distribution shift and outlier ranges in input series, improving generalizability for zero-shot forecasting.
Patch Embedding Given a univariate time series X = {x 1 , . . . , x T }, it is divided into patches x i = x 1+(i-1)P :iP with the length of P . To address non-divisible length, we pad the input at the beginning and use a binary mask m i ∈ R P for each patch to indicate the padded position. It will lead to N = ⌈T /P ⌉ such input tokens. Subsequently, we use a shared MLP : R 2P → R D to embed all patch tokens:
h i = PatchEmbed Concat(x i , m i ) ,(4)
where h i ∈ R D and D is the dimension of token embedding. Unlike point-level quantization (Ansari et al., 2024), we reserve original values without discrete quantization. It also reduces the context length (in token) of the Transformer.
this section cite: ['b26', 'b1']

Section: TRANSFORMER BACKBONE
Given N token embeddings {h i }, we adopt several crucial adaptations on a decoder-only Transformer to obtain pertoken representations aggregated from all previous tokens. First, we adapt Pre-LN (Xiong et al., 2020) to improve pretraining stability. Second, we leverage a causal self-attention mechanism with RoPE (Su et al., 2024) that introduces the position information of patch tokens. It can be formulated as follows (the layer index is omitted for simplicity):
A ij = h ⊤ i W q R Θ,i-j W ⊤ k h j , Attention(H) = Softmax Mask(A) √ d HW v ,(5)
where W q , W k , W v ∈ R D×d project token embeddings H = {h i } into d-dimensional queries, keys, and values. R Θ,t ∈ R d×d is the rotary matrix with rotation degree (t•Θ). Lastly, we implement FlashAttention (Dao et al., 2022) and KV Cache (Pope et al., 2023), since these enhancements for deployment are increasingly emphasized in large foundation models (Shoeybi et al., 2019;Rasley et al., 2020).
this section cite: ['b58', 'b49', 'b7', 'b37', 'b47', 'b40']

Section: TIMEFLOW LOSS
Given representations {h i } extracted by the last layer of the Transformer, we aim to generate length-F predictions y i = x 1+iP,F +iP at each position i via our autoregressive model. Motivated by the empirical observation that a larger patch size improves the performance in decoder-only Transformers (Das et al., 2023b) while a small patch size can be more flexible to accommodate data of different frequencies, we adopt multi-patch predictions (F > P ) for pre-training, which also reduces the steps of autoregressive inference.
Based on Equations 1 and 3, we formulate a new generative forecasting conditioned on a sequential representation h i :
L(θ, h i ) = E t,ϵ,yi u θ t y (t) i |h i -y i -y (0) i 2 . (6)
where y i ∈ R F is the groundtruth value and
y (0) i is a d- dimensional Gaussian noise, t is sampled from U[0, 1], and y (t) i = ty i + (1 -t)y (0) i
is constructed by the conditional optimal-transport path. It is important to note that the conditional representation h i differs from the conditional path and the conditional source distribution. Instead, h i is a condition of position i, also a time-invariant condition of the whole flow-matching process t ∈ [0, 1]. Technically, we implement the flow-matching network by a small MLP:
u θ t y (t) i |h i = FM-Net y (t) i , t, h i . (7
)
The training process involves sampling the noised y (t)
i , and jointly input it with t. The condition h i is integrated into the flow-matching network via AdaLN (Peebles & Xie, 2023). TimeFlow Loss for autoregressive models is formulated as:
L TimeFlow = N i=1 FM-Net y (t) i , t, h i -y i -y (0) i 2 . (8)
Inference Based on Equation 2, the push-forward process conditioned on a learned representation h i is formulated as
y (t+∆t) i = y (t) i + u θ t y (t) i |h i ∆t. (9
)
Technically, we adopt a K-step uniform trajectory, and set ∆t = 1/K. The sampling is done via starting from an initial Gaussian noise and advancing with the velocity generated by the trained FM-Net iteratively, as shown in Algorithm 1.
This procedure generates a predicted sample y i at position i. To calibrate probabilistic forecasting results during inference, we repeat this procedure using different initial noises and estimate statistics such as the median and quantiles from a set of generated predictions. We implement an efficient repeated-sampling in the TimeFlow module. The condition (representation) of lookback series is shared and reused for different initial noises, thereby reducing the overhead of repeated forwarding in the Transformer backbone.
Algorithm 1 TimeFlow Loss: Sampling Require: condition h i ∈ R D , path steps K. 1: Sample initial noise y i ∼ N (0, I).
2: ∆t = 1/K 3: for k in {0, 1 . . . , K -1} do 4: for y i ← y i + FM-Net y i , k∆t, h i ∆t 5: end for 6: Return: y i
this section cite: ['b36']

Section: TimeBench
We collected and curated TimeBench, which comprises over a trillion time points from various sources, as shown in Figure 3. Several datasets originate from research teams (Woo et al., 2024;Ansari et al., 2024;Liu et al., 2024a;b). While most datasets are collected from real-world records, a small portion (0.05%) is generated synthetically to enhance pattern diversity, following KernelSynth proposed by Ansari et al. (2024). We also leverage substantial meteorological data (Hersbach et al., 2020) because of the predictability of weather systems. Data of different frequencies encompasses common and comprehensive temporal dynamics.
9.11% 4.65% 1.02% 0.56% 22.29% 0.05% 12.50% 3.10% 39.35% 5.62% 1.31% 0.44% Chronos ECG Finance IoT LOTSA Synthetic ERA5 3h ERA5 12h ERA5 Daily ERA5 Weekly ERA5 Quarterly TimeBench (1 Trillion Time Points) ERA5 Monthly ERA5 Quarterly  4.
this section cite: ['b55', 'b1', 'b1', 'b14']

Section: Experiments
We evaluate Sundial on best-recognized zero-shot forecasting benchmarks (Section 5.1) and investigate the scaling behavior of Sundial (Section 5.2). We compare TimeFlow with other training objectives (Section 5.3). We delve into test-time calibration of generative forecasters (Section 5.4).
We conduct model adaptation of Sundial, i.e., instruction tuning (Section 5.5) and provide in-depth ablation studies to evaluate our modular enhancement (Section 5.6).
this section cite: []

Section: Time Series Forecasting
In this section, we focus on zero-shot forecasting, we compare Sundial with advanced time series foundation models on various benchmarks, including (1) point forecasting: we Table 1. Zero-shot forecasting results of time series foundation models on long-term forecasting datasets (Time-Series-Library) (Wu et al., 2022). Corresponding prediction lengths include {96, 192, 336, 720}. A lower MSE or MAE indicates a better prediction. Averaged results of four prediction lengths are reported here. 1 st Count represents the number of wins achieved by a model under all prediction lengths and datasets. Results of baseline models are officially reported by Shi et al. (2024b). Datasets in pre-training are not evaluated on corresponding models, which are denoted by the dash (-). Full results under all prediction lengths are provided in Table 9.
this section cite: ['b57']

Section: Models
SundialSmall SundialBase SundialLarge Time-MoEBase Time-MoELarge Time-MoEUltra Timer-XL MoiraiBase MoiraiLarge ChronosBase ChronosLarge TimesFM lowing Ansari et al. (2024), we calculate the median and quantiles using a set of raw predictions of Sundial. While several baseline models have been pre-trained by the consistent objective function for probabilistic evaluation, e.g., quantile loss for WQL, Sundial calculates these statistics for evaluation without any prior knowledge.
(Ours) (Ours) (Ours) (2024b) (2024b) (2024b) (2024a
this section cite: ['b1']

Section: GIFT-Eval
Aggregated results are presented in Table 2.
The benchmark evaluates performance from 23 datasets and 13 baseline models, encompassing statistical methods, task-specific models, and time series foundation models. Among supervised models and advanced foundation models, Sundial attains the first place in MASE and second place in CRPS on all unseen datasets. While the top PatchTST (Nie et al., 2022) is exhaustively trained and tweaked on each dataset, the zero-shot performance of Sundial highlights its simplicity and robustness on this comprehensive benchmark.
this section cite: ['b32']

Section: FEV Leaderboard
We evaluate our Sundial on the open leaderboard established by AutoGluon (Ansari et al., 2024), which includes 27 datasets for probabilistic forecasting. As shown in Figure 4, the zero-shot forecasting performance of Sundial exceeds 70% statistical methods and deep models that are superwisedly trained in distribution. While Sundial is ranked as the second zero-shot pre-trained models after Chronos, Sundial realizes 35× inference speedup as shown in Figure 5. Based on patch-wise tokenization and multi- patch prediction, our inference speed is near to N-BEATS.
Besides, we provide qualitative showcases in Appendix D. TimeFlow can generate highly eventful and coherent temporal patterns with input series. Beyond the mean or quantiles, our model enables the estimation of arbitrary statistics by sampling directly from the predictive distribution.
this section cite: ['b1']

Section: Scalability
From Table 1, the larger Sundial model consistently achieves better performance with the scaling of parameters. Beyond downstream performance, we delve into the utilization of model capacity. Figure 6 shows training curves of different sizes. Compared to Sundial (Small), the large version leads to 15.38% reduction in the converged training loss, exhibiting promising model capacity of generative forecasters.
this section cite: []

Section: TimeFlow Loss
Based on the flow-matching framework, TimeFlow Loss allows autoregressive models to learn and generate flexible distributions while enhancing representation learning. To validate the effectiveness of this design, we implement two alternatives: (1) an MLP network and MSE Loss and (2) a parameterized training objective based on the denoising diffusion procedure (Li et al., 2024). We adopt the same parameterized network and Transformer backbone and pretrain them on TimeBench. Since the converged training loss is not comparable across different objective functions, we compare zero-shot performance in Table 3. Despite allowing for sampling predictions, performance using diffusion-based objective is notably inferior to TimeFlow Loss.
In addition to zero-shot performance, we provide showcases We also provide a probablistic metric CRPS to compare different objectives in Table 7, which validate that the predictive distribution modeled by TimeFlow is more coherent and diverse than counterpart training objectives. It benefits downstream tasks by generating multiple plausible predictions, conveys various future possibilities and enhances the reliability of decision-making.
this section cite: ['b21']

Section: Test-Time Calibration
Generative modeling facilitates the flexibility to calibrate the final prediction during inference. Based on the medianbased forecasting strategy, i.e., starting from multiple noise of a standard Gaussian and calculating the median of raw predictions, there are two configurations to calibrate final predictions: (1) the number of samples to calculate statistics and (2) sampling steps K used for flow-matching. Figure 7 shows the results using different configurations.
The top two figures conform to the central limit theorem. Generating more samples leads to more calibrated estimation of prediction and confidence interval. The bottom two figures indicate that using fine-grained steps during the pushforward process can leads to more precise predictions.
The trade-off between inference time and performance reveals the potential of test-time calibration, which does not require retraining models. The generative capability of Sundial provides flexibility for various use cases requiring different levels of uncertainty. In our experiments, sampling 20 predictions with each generated by 50 steps consumes nearly one second on a CPU, which is notably more efficient than tuning deep models or statistical methods. Advanced strategies of sampling and post-processing of raw prediction leave interesting directions for future exploration.
this section cite: []

Section: Model Adaptation
Inspired by the prevalence of instruction tuning (Wei et al., 2021) that adapts foundation models on a collection of tasks. We fine-tune pre-trained Sundial (Base) on the FEV leaderboard, including short-term tasks with different prediction lengths. Our model is tuned once on all aggregated datasets. We evaluate the performance on unseen test splits (Figure 8). We observe that the performance can be further improved compared to zero-shot forecasting. Furthermore, training from scratch on aggregated datasets results in inferior performance, implying knowledge transfer in pre-trained models.
Figure 8. Performance on the FEV leaderboard, including (1) training Sundial from scratch on all datasets from the FEV leaderboard, (2) zero-shot forecasting using pre-trained Sundial, and (3) finetuning once on all datasets from the FEV leaderboard.
this section cite: ['b53']

Section: Ablation Study
We conducted several ablation studies that provide insights into the enhancement made to Sundial's architecture. We evaluate the overall zero-shot performance on TSLib, which covers six different datasets and four prediction lengths.
MSE MAE 0.26 0.28 0.30 0.32 0.34 0.36 0.38 0.40 TSLib Zero-Shot (Avg.) w/o RoPE RoPE MSE MAE 0.28 0.30 0.32 0.34 0.36 0.38 0.40 TSLib Zero-Shot (Avg.) Pre-LN (15k iter) Pre-LN (30k iter) Post-LN (15k iter) Post-LN (30k iter) Time Memory 1.20 1.22 1.24 1.26 1.28 1.30 Training Speed (s/iter) w/o FlashAttn FlashAttn 28 30 32 34 36 38
Memory Footprint (GB) Time 0.5 0.6 0.7 0.8 0.9
1.0 1.1 1.2 1.3 Inference Speed (s/iter) w/o KV Cache KV Cache (a) RoPE (b) LayerNorm (c) FlashAttention (d) KV Cache Figure 9
. Ablation studies with respect to architectural enhancements. We report the averaged results of TSLib datasets (Wu et al., 2022) from four prediction lengths {96, 192, 336, 720} and all six datasets. The context length is set to 2880 and the patch length is 16.
RoPE Prior research (Liu et al., 2024a) observed that the introduction of RoPE (Su et al., 2024) yields better results in supervised forecasting tasks. As shown in Figure 9 (a), RoPE can also improve zero-shot forecasting, presenting a general enhancement for time series foundation models.
Layer Normalization Pre-LN (Baevski & Auli, 2018) is widely adopted in large language models (Touvron et al., 2023) due to the training stability. As depicted in Figure 9 (b), training with Pre-LN for more iterations yield better performance. In contrast, training with Post-LN, which is the predominant choice in supervised models, may adversely affect downstream results.
FlashAttention and KV Cache We make it to leverage FlashAttention (Dao et al., 2022) and KV Cache to reduce the computational costs. As shown in Figure 9 (c) and (d), they notably reduce 14.8% memory footprint and 43.6% inference time without affecting performance.
this section cite: ['b57', 'b49', 'b2', 'b51', 'b7']

Section: Conclusion
In this work, we collect and curate TimeBench, a trillionscale time series dataset for building time series foundation models, which can benefit the research community. Towards time series foundation models, we delve into tokenization and optimization, presenting contributions in two aspects. First, we demonstrate that continuous tokenization, such as patch tokens, can be more effective and efficient for the time series modality, and generative modeling presents a native approach for learning on continuous-valued time series.
Second, we propose a novel training objective to accommodate heterogeneous time series distribution. It endows autoregressive models with an inherent capability to sample from non-categorical distribution. Our pre-trained Sundial models make substantial advances on best-recognized forecasting leaderboards. We hope this work can inspire future paradigms for pre-training time series foundation models and enhance their applicability to real-world applications.
this section cite: []

Section: References
Ref_id:b0 Title: Gift-eval: A benchmark for general time series forecasting model evaluation Year: (2024)
Ref_id:b1 Title: Learning the language of time series Year: (2024)
Ref_id:b2 Title: Adaptive input representations for neural language modeling Year: (2018)
Ref_id:b3 Title: An empirical evaluation of generic convolutional and recurrent networks for sequence modeling Year: (2000)
Ref_id:b4 Title: On the opportunities and risks of foundation models Year: (2021)
Ref_id:b5 Title: Box and jenkins: time series analysis, forecasting and control Year: (2013)
Ref_id:b6 Title: Time series analysis: forecasting and control Year: (2015)
Ref_id:b7 Title: Flashattention: Fast and memory-efficient exact attention with io-awareness Year: (2022)
Ref_id:b8 Title: Longterm forecasting with tide: Time-series dense encoder Year: (2023)
Ref_id:b9 Title: A decoderonly foundation model for time-series forecasting Year: (2023)
Ref_id:b10 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b11 Title: Physiobank, physiotoolkit, and physionet: components of a new research resource for complex physiologic signals Year: (2000)
Ref_id:b12 Title: Moment: A family of open time-series foundation models Year: (2024)
Ref_id:b13 Title: Large language models are zero-shot time series forecasters Year: (2023)
Ref_id:b14 Title: Large language models are zero-shot time series forecasters Year: (2020)
Ref_id:b15 Title: The tabular foundation model tabpfn outperforms specialized time series forecasting models based on simple features Year: (2025)
Ref_id:b16 Title: Forecasting: principles and practice Year: (2018)
Ref_id:b17 Title: A highly efficient gradient boosting decision tree. Advances in neural information processing systems Year: (2017)
Ref_id:b18 Title: Reversible instance normalization for accurate time-series forecasting against distribution shift Year: (2021)
Ref_id:b19 Title: A method for stochastic optimization Year: (2014)
Ref_id:b20 Title: Flow matching with gaussian process priors for probabilistic time series forecasting Year: (2024)
Ref_id:b21 Title: Autoregressive image generation without vector quantization Year: (2024)
Ref_id:b22 Title: Foundation models for time series analysis: A tutorial and survey Year: (2024)
Ref_id:b23 Title: Temporal fusion transformers for interpretable multi-horizon time series forecasting Year: (2021)
Ref_id:b24 Title: Flow matching for generative modeling Year: (2022)
Ref_id:b25 Title: Flow matching guide and code Year: (2024)
Ref_id:b26 Title: Non-stationary transformers: Exploring the stationarity in time series forecasting Year: (2022)
Ref_id:b27 Title: itransformer: Inverted transformers are effective for time series forecasting Year: (2023)
Ref_id:b28 Title: Learning nonstationary time series dynamics with koopman predictors Year: (2023)
Ref_id:b29 Title: Timerxl: Long-context transformers for unified time series forecasting Year: (2024)
Ref_id:b30 Title: Timer: Generative pre-trained transformers are large time series models Year: (2024)
Ref_id:b31 Title: Era5-land: A stateof-the-art global reanalysis dataset for land applications Year: (2021)
Ref_id:b32 Title: A time series is worth 64 words: Long-term forecasting with transformers Year: (2022)
Ref_id:b33 Title: Gpt-4 technical report Year: (2023)
Ref_id:b34 Title: Neural basis expansion analysis for interpretable time series forecasting Year: (2019)
Ref_id:b35 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b36 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b37 Title: Efficiently scaling transformer inference Year: (2023)
Ref_id:b38 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b39 Title: Zero-shot text-toimage generation Year: (2021)
Ref_id:b40 Title: Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters Year: (2020)
Ref_id:b41 Title: Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting Year: (2021)
Ref_id:b42 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b43 Title: Probabilistic forecasting with autoregressive recurrent networks Year: (2020)
Ref_id:b44 Title: Non-autoregressive conditional diffusion models for time series prediction Year: (2023)
Ref_id:b45 Title: Scaling law for time series forecasting Year: (2024)
Ref_id:b46 Title: Time-moe: Billion-scale time series foundation models with mixture of experts Year: (2024)
Ref_id:b47 Title: Megatron-lm: Training multibillion parameter language models using model parallelism Year: (2019)
Ref_id:b48 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b49 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b50 Title: Conditional score-based diffusion models for probabilistic time series imputation Year: (2021)
Ref_id:b51 Title: Improving and generalizing flow-based generative models with minibatch optimal transport Year: (2023)
Ref_id:b52 Title: Algorithmic learning in a random world Year: (2005)
Ref_id:b53 Title: Finetuned language models are zero-shot learners Year: (2021)
Ref_id:b54 Title: A multi-horizon quantile recurrent forecaster Year: (2017)
Ref_id:b55 Title: Unified training of universal time series forecasting transformers Year: (2024)
Ref_id:b56 Title: Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting Year: (2021)
Ref_id:b57 Title: Temporal 2d-variation modeling for general time series analysis Year: (2022)
Ref_id:b58 Title: On layer normalization in the transformer architecture Year: (2020)
Ref_id:b59 Title: Transformer utilizing cross-dimension dependency for multivariate time series forecasting Year: (2018)
Ref_id:b60 Title: A lower MSE or MAE indicates a better prediction. Averaged results of four prediction lengths are reported here. 1 st Count represents the number of wins achieved by a model under all prediction lengths and datasets. Results of baseline models are officially reported by Shi et al. (2024b). Datasets for pre-training are not evaluated on corresponding models Year: (2022)
Ref_id:b61 Title:  Year: (2023)
Ref_id:b62 Title:  Year: ()
Ref_id:b63 Title:  Year: ()
Ref_id:b64 Title: ETTh2 (Pred-96) ETTh Year: ()
Ref_id:b65 Title: ETTm1 (Pred-96) ETTm1 Year: ()
Ref_id:b66 Title: ETTm2 (Pred-96) ETTm Year: ()
Ref_id:b67 Title:  Year: ()
Ref_id:b68 Title: Showcases of zero-shot predictions from Sundial (Base) on long-term forecasting datasets Year: (2022)
