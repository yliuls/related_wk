Title: K 2 VAE: A Koopman-Kalman Enhanced Variational AutoEncoder for Probabilistic Time Series Forecasting
Abstract: Probabilistic Time Series Forecasting (PTSF) plays a crucial role in decision-making across various fields, including economics, energy, and transportation. Most existing methods excell at shortterm forecasting, while overlooking the hurdles of Long-term Probabilistic Time Series Forecasting (LPTSF). As the forecast horizon extends, the inherent nonlinear dynamics have a significant adverse effect on prediction accuracy, and make generative models inefficient by increasing the cost of each iteration. To overcome these limitations, we introduce K 2 VAE, an efficient VAE-based generative model that leverages a KoopmanNet to transform nonlinear time series into a linear dynamical system, and devises a KalmanNet to refine predictions and model uncertainty in such linear system, which reduces error accumulation in long-term forecasting. Extensive experiments demonstrate that K 2 VAE outperforms state-of-the-art methods in both short-and long-term PTSF, providing a more efficient and accurate solution.

Section: Introduction
In recent years, time series analysis has seen remarkable progress, with key tasks such as anomaly detection (Wang et al., 2023;Liu & Paparrizos, 2024;Miao et al., 2025;Hu et al., 2024;Wu et al., 2025c), classification (Yao et al., 2024;Campos et al., 2023), and imputation (Gao et al., 2025;Wang et al., 2024a;c;Yu et al., 2025a), among others (Wang et al., 2024b;Miao et al., 2024a;Liu et al., 2025a;Huang et al., 2023;Yao et al., 2023), gaining attention. Among these, Probabilistic Time Series Forecasting (PTSF) is a crucial and widely studied task. By quantifying the stochastic temporal evolutions of multiple continuous variables, it Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). We compare three native probabilistic forecasting models including GRU MAF, TimeGrad, and CSDI with three point forecasting models equipped with distributional heads including FITS, PatchTST, and iTransformer on ETTh1. Longer forecasting horizons lead to rapid collapse of the CRPS metric (lower is better) on probabilistic forecasting models, even worse than point forecasting models.
provides significant support for decision-making in various fields such as economics (Sezer et al., 2020;Huang et al., 2022b), traffic (Wu et al., 2024b;2025d;Pan et al., 2023a;Cirstea et al., 2022b;Fang et al., 2021;Yang et al., 2021;2022), energy (Wang et al., 2024d;Guo et al., 2015;Sun et al., 2022), and AIOps (Lin et al., 2024a;Campos et al., 2022;Chen et al., 2023;Pan et al., 2023c;Lin et al., 2025). In these practical applications, an urgent need is to extend the prediction time to the distant future, known as Longterm Probabilistic Time Series Forecasting (LPTSF), which is highly meaningful for long-term planning and early warning. Most existing methods excell at short-term problem settings, such as predicting up to 48 steps or fewer (Rasul et al., 2021a;Kollovieh et al., 2023;Rasul et al., 2021b), while directly applying these methods to long-term forecasting tasks often results in poor performance-see Figure 1.
However, probabilistic forecasting faces numerous challenges in long-term forecasting tasks. First, the inherent nonlinearity of time series challenges probabilistic models in modeling dynamic evolution. Due to factors such as non-stationarity and complex interdependencies between variables, time series typically exhibit nonlinear characteristics, complicating the construction of probabilistic models. Specifically, the nonlinearity makes it difficult for these models to derive a simple equation that precisely describes the state transition process. As a result, the uncertainties within the models are also hard to quantify, particularly in long-term forecasting tasks. Second, as the forecasting horizon extends, the accuracy and efficiency become major bottlenecks. Longer foreacasting horizons lead to more intricate target distributions, which causes remarkable error accumulation. It also makes the diffusion-based (Kollovieh et al., 2023;Rasul et al., 2021a) or flow-based models (Rasul et al., 2021b) difficult to find a clear probabilistic transition path and inefficient to perform each iteration, which results in more computational consumption but worse performance.
Since the nonlinearity in time series leads to the dynamic evolution of complex patterns, probabilistic models struggle to effectively capture these changes and accurately model their evolution. To tackle this thorny issue, Koopman Theory (Koopman, 1931) provides a linearization approach to transform the nonlinear time series into the space of measurement function, which is a theoretically infinitedimeansional space characterizing all measurements of the dynamical system at each moment, and the transition process of these measurements can be captured by a linear Koopman Operator (Lan & Mezić, 2013). On the other hand, in order to accurately and efficiently model the process uncertainty and mitigate the error accumulation phenomenon in long-term forecasting, the Kalman Filter (Welch, 1995) provides a solution, which fuses observations from multiple sensors to extract the Kalman gains, to refine the prediction and process uncertainty. This inspires us to transform the probabilistic time series forecasting into modeling the process uncertainty of a linear dynamical system in the space of the measurement function.
In this study, we propose K 2 VAE, a generative probabilistic forecasting model tailored for LPTSF-see Figure 2. First, to handle the nonlinearity and capture the underlying dynamics in time series, we patchify the time series into tokens and model them through the KoopmanNet. The KoopmanNet provides a way to simulate the Koopman Theory, which transforms the nonlinear time series into latent measurements, and fit the Koopman Operator to construct a "biased" linear dynamical system easy to describe and model. Second, to achieve accurate long-term forecasting performance, we design a KalmanNet in a data-driven manner based on the principle of Kalman Filter.
Through integrating the residual nonlinear information as control inputs, while treating the biased linear dynamical system as the observation, the KalmanNet predicts and updates to model and refine the uncertainty with Kalman gain. This effectively mitigates the error accumulation of the linear system and helps construct the variational distribution in the space of the measurement function with clear semantics. Compared to diffusion-based models or flow-based models with longer generation processes, which cause more computational consumption and memory overhead, K 2 VAE adopts a VAE-based structure composed of lightweight but effective KoopmanNet and KalmanNet, which contributes to fast one-step generation and lower memory occupation. The contributions are summarized as follows: • To address PTSF, we propose an efficient framework called K 2 VAE. It transforms nonlinear time series into a linear dynamical system. Through predicting and refining the process uncertainty of the system, K 2 VAE demonstrates strong generative capability and excells in both the short-and long-term probabilistic forecasting.
• To distengle the complex nonlinearity in the time series, we design a KoopmanNet to fully exploit the underlying linear dynamical characteristics in the space of measurement function, simplify the modeling, and thus contributing to high model efficiency.
• To mitigate the error accumulation in LPTSF, we devise a KalmanNet to model, and refine the prediction and uncertainty iteratively.
• Comprehensive experiments on both short-and longterm PTSF show that K 2 VAE outperforms stateof-the-art baselines. Additionally, all datasets and code are avaliable at https://github.com/  decisionintelligence/K2VAE.
this section cite: ['b70', 'b41', 'b47', 'b93', 'b4', 'b18', 'b92', 'b64', 'b16', 'b90', 'b20', 'b3', 'b8', 'b37', 'b22', 'b22', 'b23', 'b24', 'b80']

Section: Preliminaries
Koopman Theory. Koopman Theory (Koopman, 1931;Lan & Mezić, 2013) is a widely used mathematical tool for dynamic system analysis, providing a way to linearize the nonlinear systems. For nonlinear system x k+1 = f (x k ), where x k denotes system state and f is a nonlinear function, it assumes that the system's state can be mapped into the space of measurement function ψ, where it can be modeled by an infinite-dimensional linear Koopman Operator K:
ψ(x k+1 ) = ψ(f (x k )) = K • ψ(x k )(1)
Koopman Theory helps understand the underlying dynamics of complex nonlinear systems and serves as a powerful tool to linearize them for ease of process.
Kalman Filter. Kalman Filter (Welch, 1995;Simon, 2001) is a recursive algorithm used for estimating the state of a linear dynamic system. It works in two steps: first, it predicts the current state x k and uncertainty covariance matrix P k based on the system's state transition equation; then, it updates the estimation by incorporating the difference between the measurement and prediction, known as Kalman gain K k . The Kalman Filter effectively fuses information from multiple sensors to enhance estimation accuracy while modeling the uncertainty of the system.
VAE for Probabilistic Time Series Forecasting. PTSF can be treated as a conditional generative task, i.e., generating
forecasting horizon Ŷ = [x T +1 , xT +2 , • • • , xL ] ∈ R N ×L given context series X = [x 1 , x 2 , • • • , x T ] ∈ R N ×T
, where N denotes the number of variables, T denotes the context length, and L denotes the forecasting horizon. The objective is to model the conditional distribution P(Y |X) and sample from it to obtain Ŷ . When using Variational Au-toEncoder (Higgins et al., 2017;Pu et al., 2016), the loglikehood objective is optimized through the Evidence Lower Bound (2) which is obtained by Jensen Inequality:
L ELBO = -E[log P(Y |Z, X)] + D KL (Q(Z|X)||P(Z|X)) (2)
In our proposed K 2 VAE, we meticulously construct the variational distribution Q(Z|X), aligning it with the uncertainty of the dynamical system. This endows the latent space in K 2 VAE with clear semantics, enhancing its generative capabilities in PTSF.
this section cite: ['b23', 'b24', 'b80', 'b65', 'b21', 'b53']

Section: Methodology

this section cite: []

Section: K 2 VAE Architecture
As demonstrated in Figure 3, K 2 VAE consists of four main components: Input Token Embedding, KoopmanNet, KalmanNet, and Decoder. The KoopmanNet and Kalman-Net consitute the Encoder of K 2 VAE. To facilitate comprehension, we present the Data Flow-see Figure 2.
Overall, K 2 VAE employs a meticulously designed pipeline to model the time series at the perspective of dynamic system. First, the Input Token Embedding module patchifys the time series into tokens. Then the KoopmanNet projects them into the space of measurement function, where the inherent nonlinearity and intricate joint distribution between variables are reconsidered for ease. Sequentially, the Koopman Operator is fitted and iterates over the first token to delineate a linear system. Obviously, the perfect measurement function which constructs an absolute linear system is the ideal objective, which means the series generated by the Koopman Operator is biased. We then design the KalmanNet to refine such biased linear system and model the uncertainty by outputting the covariance matrix of multi-dimensional state vector, which assigns the variational posterior distribution Q(Z|X) in the space of measurement function with clear semantics. The Decoder works as the inverse measurement function ψ -1 to map the samples from Q(Z|X) to the original space, which also serves as the decoder of VAE and models the target distribution P(Y |Z, X) of the forecasting horizon.
this section cite: []

Section: INPUT TOKEN EMBEDDING
Since Triformer (Cirstea et al., 2022a) first proposes the Patching technique, existing works (Nie et al., 2023;Wu et al., 2025c) demonstrate that considering a patch as the "token" retains most semantic information and helps establish meaningful state transition procedure for autoregressive models. Our proposed K 2 VAE also works like an autoregressive dynamic system to model the state transition procedure. Different from those Channel-Independent models which divides patches for each channel and projects them independently, we consider multivariate patches as tokens to implicitly model the cross-variable interaction during state transition. We divide the context series X = [x 1 , x 2 , • • • , x T ] ∈ R N ×T into non-overlapping patches:
X P = x P 1 , x P 2 , • • • , x P n ∈ R N ×s×n ,(3)
where s = T /n denotes the patch size, n denotes the patch number, and x P i ∈ R N ×s denotes a patch. Then X P are embeded into high-dimensional hidden space:
X P ′ = Projection(Flatten(X P )),(4)
where patches are first flattened into R (N ×s)×n and then mapped into embeddings X P ′ ∈ R d×n through a linear projection to fuse the variable information.
this section cite: ['b48']

Section: K 2 VAE ENCODER
Linearizing with the KoopmanNet. Since there exists variable-wise periodic misalignment or temporal nonstationarity in realistic multivariate time series, yielding non-linearity, K 2 VAE applies Koopman Theory (Koopman, 1931) to construct the measurement function to project the system states into measurements which can be modeled as a linear system. Practically, we use a learnable MLP-based network to serve as the measurement function ψ:
X P * = ψ(X P ′ ) = x P * 1 , x P * 2 , • • • , x P * n ,(5)
where X P * ∈ R d×n denotes the projected tokens in the measurement space. To capture the transition rule, we utilize the one-step eDMD (Schmid, 2010;Liu et al., 2023) over X P * to efficiently find the best fitted K loc :
X P * back = x P * 1 , x P * 2 , • • • , x P * n-1 ,(6)
X P * f ore = x P * 2 , x P * 3 , • • • , x P * n ,(7)
K loc = X P * f ore (X P * back ) † ,(8)
where (X P * back ) † denotes the Moore-Penrose inverse of X P * back . K loc effectively captures the local transition rule in the space of current measurement function. However, when ψ is underfitted, the low quality of the space may cause numerical instability or guide the model to converge in a wrong direction. To mitigate this issue and capture the global-shared dynamics, we introduce a learnable part K glo . Then we delineate the system through the Koopman
Operator K = K loc + K glo : XC = xC 1 , xC 2 , • • • , xC n ,(9)
XH = xH 1 , xH 2 , • • • , xH m , (10
) xC i = (K) i-1 x P * 1 , xH i = (K) i+n-1 x P * 1 ,(11)
where XC ∈ R d×n denotes the reconstruction context generated by Koopman Operator K ∈ R d×d and XH ∈ R d×m is the predicted horizon, m = L/s means that predicting L steps in the original space is equivalent to predicting m steps in the space of measurement function.
Modeling the Uncertainty with the KalmanNet. Since we adopt a data-driven paradigm to model the measurement function ψ and Koopman Operator K, it exists bias between the generated XC and X P * during optimization, known as a biased linear system. Inspired by Kalman Filter (Welch, 1995;Simon, 2001) which is born to refine such biased linear sytem, we devise a KalmanNet to model and refine the uncertainty adaptively, aligning it with the variational distribution Q(Z|X) in the latent measurement space. Specifically, we first fully reuse the nonlinear residual through the Integrator based on an Encoder-Only Vanilla Transformer (Vaswani et al., 2017):
X Res = X P * -XC ,(12)
U = Integrator(X Res ) = [u 1 , u 2 , • • • , u m ] ,(13)
where U ∈ R d×m denotes the output integrated by the Integrator. We then construct the Process Model of KalmanNet, which describes the state transition process:
z k = Az k-1 + Bu k + w k ,(14)
z 0 = x P * n ,(15)
where A ∈ R d×d is the state transition matrix, B ∈ R d×d is the control input matrix, and w k ∼ N (0, Q) is the process noise and Q is its covariance matrix. Sequentially, we construct the Observation Model:
o k = Hz k + v k ,(16)
where H ∈ R d×d is the observation matrix and we treat the prediction XH as the prior observation in Update Step (20). v k ∼ N (0, R) is the observation noise and R is its covariance matrix. Our goal is to reuse the information from the nonlinear residual, and integrate it into the linear system constructed by KoopmanNet, thus obtaining a more accurate linear system and modeling the uncertainty. In the KalmanNet, all the matrices are learnable. Additionally, we initialize the covariance matrices Q and R as identity matrices and use lower triangular matrices L Q and L R to keep the positive definiteness: Q = L Q L T Q and R = L R L T R .
this section cite: ['b23', 'b63', 'b42', 'b80', 'b65', 'b69']

Section: Then we conduct the Prediction Step and Update Step iteratively, the Prediction Step can be formulated as:
ẑk = Az k-1 + Bu k ,(17)
Pk = AP k-1 A T + Q,(18)
where ẑk is the predicted state and Pk is the predicted covariance matrix of the process uncertainty. Then the Update Step measures the weight between observation and prediction through Kalman gain K k to refine the system:
K k = Pk H T (H Pk H T + R) -1 ,(19)
z k = ẑk + K k (x H k -H ẑk ),(20)
P k = (I -K k H) Pk ,(21)
where z k and P k is the refined state vector and covariance matrix. We then obtain the refined predictions Z = [z 1 , z 2 , • • • , z m ] and covariance matrices of each token P = [P 1 , P 2 , • • • , P m ], which describes the temporal process uncertainty in the dynamical system. We show that the process also obeys the basic assumptions of Koopman Theory in Section 3.2. To fully utilize the ability of the Integrator, we make a skip connection:
Z ′ = Z + U (22
)
During the training process, the model leverages the Integrator to integrate nonlinear information and gradually adjust the topological structure of the measurement space. Optimized by L Rec (27), the deviation of the linear system constructed by the KoopmanNet gradually decreases, causing U → 0. This facilitates a linear dynamical system in the measurement space and gradually reduces dependence on the Integrator.
this section cite: []

Section: K 2 VAE DECODER
After obtaining the prediction Z ′ , and the covariance matrix P of process uncertainty, the variational distribution is formulated as Q(Z|X) = N (Z ′ , P). During training, we conduct reparameterization sampling from it to keep the ensure the propagation of the gradient. Finally, we utilize the Decoder to map the samples back to the original space and model the P(Y |Z) with an isotropic Gaussian distribution. Specifically, the Decoder consists of two same MLP structures as the inverse of the Koopman Encoder ψ, we formalize them as ψ -1 µ and ψ -1 σ :
Z sample = Resample(Q(Z|X)),(23)
µ = ψ -1 µ (Z sample ), σ = ψ -1 σ (Z sample ),(24)
X Rec = ψ -1 µ ( XC ),(25)
so that the P(Y |Z) = N (µ, σ) is modeled. We also map back the Koopman reconstruction XC from the measurement space to optimize the L Rec (27), which helps measurement function ψ to build a linear system.
this section cite: []

Section: OVERALL LEARNING OBJECTIVE
The overall learning objective is weightsumed by L ELBO and L Rec :
L ELBO = -E[log P(Y |Z, X)]+ D KL (Q(Z|X)||P(Z|X)),(26)
L Rec = ||X -X Rec || 2 2 ,(27)
where the L ELBO ensures the fundmental mechanism of K 2 VAE. The prior distribution is P(Z|X) = N (0, I), where we hope the linear system in measurement space converge to a stable state. L Rec facilitates the linearization of the measurement space.
this section cite: []

Section: Theoretical Analysis

this section cite: []

Section: THE STABILITY OF KALMANNET
Since the proposed KalmanNet works in a data-driven manner, the floating-point operation error may cause the covariance matrix P losing positive definiteness, which often occurs in the step (21). To mitigate this, we utilize a numerically stable form for this step.
Theorem 3.1. The positive-definiteness of covariance matrix P k during the update step P k = (I -K k H k ) Pk can be retained through a numerically stable form:
P k = 1 2 (P k + P T k ),(28)
P dual k = (I -K k H k ) Pk (I -K k H k ) T + K k R k K T k ,(29)
where (28) ensures the symmetry, (29) stabilizes the positivedefiniteness by decomposing the formula into the sum of two positive definite terms, which better ensures positive definiteness during floating operation.
3.2.2. THE CONVERGENCE OF K 2 VAE Since K 2 VAE models a linear dynamical system in the measurement space where the Koopman Operator serves as the state transition equation, we hope that the convergence state of the KalmanNet does not violate the assumptions of Koopman Theory. In K 2 VAE, we meticulously design the KalmanNet by making it gradually converge to the Koopman Operator in the forecasting horizon. Theorem 3.2. When U → 0, the state transition equation of the KalmanNet in K 2 VAE gradually converges to the Koopman Operator.
We provide the proof of Theorem 3.1-3.2 in Appendix A. Table 2. Comparison on short-term probabilistic forecasting scenarios across eight real-world datasets. Lower CRPS or NMAE values indicate better predictions. The means and standard errors are based on 5 independent runs of retraining and evaluation. Red: the best, Blue: the 2nd best.
this section cite: []

Section: Model Metric Exchange-S Solar-S Electricity-S Traffic-S ETTh1-S ETTh2-S ETTm1-S ETTm2

this section cite: []

Section: Experiments
In this section, we provide empirical results to show the strong performance of K 2 VAE against state-of-art baselines on both short-and long-term probabilistic forecasting tasks.
We also analyze the model efficiency and the key parameters of K 2 VAE as the proof of architectural superiority.
this section cite: []

Section: Experimental Setup
Datasets. We conduct experiments on 8 datasets of shortterm forecasting and 9 datasets of long-term forecasting based on ProbTS (Zhang et al., 2024a), a comprehen-sive benchmark used to evaluate probabilistic forecasting tasks. Specifically, we use the datasets ETTh1-S, ETTh2-S, ETTm1-S, ETTm2-S, Electricity-S, Solar-S, Traffic-S, and Exchange-S for short-term forecasting, of which the context length T is equivalent to forecasting horizon L with T = L = 30 for Exchange-S and T = L = 24 for the others. For long-term forecasting, we use the datasets ETTh1-L, ETTh2-L, ETTm1-L, ETTm2-L, Electricity-L, Traffic-L, Exchange-L, Weather-L, and ILI-L with forecasting horizon L ∈ {24, 36, 48, 60} for ILI-L and L ∈ {96, 192, 336, 720} for the others. Note that we fix the context length of all the models with T = 36 for ILI-L and T = 96 for the others to ensure a fair comparison. Details are shown in Table 1.
Baselines. We compare K 2 VAE with 11 strong baselines, including 4 point forecasting models: FITS (Xu et al., 2024), PatchTST (Nie et al., 2023), iTransformer (Liu et al., 2024), and Koopa (Liu et al., 2023), as well as 7 generative models: TSDiff (Kollovieh et al., 2023), D 3 VAE (Li et al., 2022), GRU NVP, GRU MAF, Trans MAF (Rasul et al., 2021b), TimeGrad (Rasul et al., 2021a), and CSDI (Tashiro et al., 2021), in both short-term and long-term probabilistic forecasting scenarios. The point forecasting models are equipped with gaussian heads to predict the distributions. Detailed descriptions of these models can be found in Appendix C.2.
this section cite: ['b89', 'b48', 'b42', 'b22', 'b28', 'b68']

Section: Evaluation Metrics.
We use two commonly-used metrics CPRS (Continuous Ranked Probability Score) and NMAE (Normalized Mean Absolute Error) to evaluate the probabilistic forecasts. Detailed descriptions of these metrics can be found in Appendix C.3.
this section cite: []

Section: Main Results
Comprehensive probabilistic forecasting results are listed in Table 2 and Table 3 with the best in red and the second in blue. We have the following observations:
1) K 2 VAE outperforms state-of-the-art baselines, showing notable improvements in predictive performance. In short-term scenarios, it achieves a 7.3% reduction in CRPS and 14.5% reduction in NMAE compared to the second-best baseline, CSDI. In long-term scenarios, it surpasses PatchTST with improvements of 20.9% and 19.9%.
this section cite: []

Section: 2) K 2 VAE shows significant advantage on nonstationary time seris datasets such as Exchange-S and Exchange-L.
There exists distribution drift phenomenon in these datasets, which causes non-linearity and hinders the prediction and uncertainty modeling. Though diffusion-based and flowbased models are theoretically capable of fitting any distributions, they struggle to construct explict probability transition paths to reach such complex destinations. While K 2 VAE simplifys this difficulty through modeling the time series in a linear dynamical system, where the uncertainty is more explicit and easier to be modeled.
3) K 2 VAE also shows stable and strong performance with respect to the varying forecasting horizons-see Table 9 and Table 10 in Appendix C.5. The performance of most baselines drops significantly as the forecasting horizon extends, while K 2 VAE maintains superiority. One potential reason is that K 2 VAE utilizes the KoopmanNet to effectively handle the inherent nonlinearity in long-term forecasting. Another reason is that the KalmanNet mitigates the error accumulation by integrating diverse information.
this section cite: []

Section: Ablation Studies

this section cite: []

Section: VARIANTS OF KOOPMAN OPERATOR
We design the local Koopman Operator K loc obtained by one-step eDMD and a learnable part K glo . As one-step eDMD estimates the K loc through matrix calculation, which relys on the local quality of the space of measurement function. Once the initilization leads to an ill-conditioned topological structure, the one-step eDMD suffers from the numerical calculation error or guides the model to converge in a wrong direction, which often occurs in long-term probabilistic forecasting scenarios and hinders the performance. To enhance the robustness, we adopt a global learnable part K glo to mitigate this phenomenon while capturing the global-shared dynamics. As shown in Table 4, mixed Koopman Operator K = K loc + K glo demonstrates better performance on both short-and long-term tasks while providing roubustness to avoid calculation error if K loc fails. Complete experimental results are provided in Table 12 in Appendix C.6.
this section cite: []

Section: CONNECTIONS IN KALMANNET
We adopt an Integrator to assist the KalmanNet for faster convergence, which potentially helps tune the topological structure of the space of measurement function into the linear dynamical system. Specifically, the Integrator integrates the non-linear residual into the control input of KalmanNet, which is proved not to affect the prior of Koopman Theory in Section 3.2. We also make a skip connection between Integrator and the KalmanNet for the final prediction, this constraints Integrator predicting residuals from residuals. Since the space of the measurement space is optimized to converge to a linear system, the Integrator serves as an assistant and gradually stop helping the model. We showcase the different variants in Table 5, to which only some of the features mentioned above are applied. We observe that our adopted Mixed variant outperforms others, because it provides gains for the KalmanNet, which integrates non-linear information for adaption, and provides constraints for the Integrator, which makes full use of it without destroying the assumptions of Koopman Theory. The variant "w/o skip connection" completely depends on the linear fitting ability of the KalmanNet, which is hard to disentangle the nonlinear components in the early stages of training, thus potentially hindering the modeling of process uncertainty. On the other hand, the variant "w/o control input" gives too little support for KalmanNet to adaptively refine the prediction and process uncertainty. Complete experimental results are provided in Table 13 in Appendix C.5.
this section cite: []

Section: ABLATIONS OF KOOPMANNET & KALMANNET
As the most important modules, the KoopmanNet and KalmanNet jointly contribute to state-of-the-art performance of K 2 VAE. To evaluate their impactment, we conduct ablation studies and the results are shown in Table 6.
It is observed that both the KoopmanNet and KalmanNet show indispensability in probabilistic forecasting. Since the KoopmanNet ensures the linearization of the modeling, it shows greater impact in forecasting performance. Another reason is that KalmanNet does not excell at non-linear modeling because it is based on the linear kalman filter.
this section cite: []

Section: Model Efficiency
We evaluate the model efficiency from three aspects: probabilistic forecasting performance (CRPS), inference time (sec/sample), and max gpu memory (GB). Figure 4 showcases a common scenario on Electricity-L (96-96), which reflects the overall relative relationships on above-mentioned three aspects. K 2 VAE achieves best forecasting performance while occupying the minimum gpu memory and having the fastest inference speed. One reason is that K 2 VAE applies KoopmanNet and KalmanNet, composed of several lightweight MLPs or linear layers, to efficiently build the variational distribution as process uncertainty, thus enhancing generation capability of K 2 VAE. Another reason is that K 2 VAE utilizes the VAE architecture and obeys the one-step-generation paradigm, while diffusion-based or flow-based models have longer probabilistic transition paths, which produces more intermediate results and consumes longer duration. More evidence of model efficiency is provided in Table 11 in Appendix C.5.
this section cite: []

Section: Related Works

this section cite: []

Section: Probabilistic Time Series Forecasting
Probabilistic forecasting aims to provide the predictive distribution of the target variable. With the rapid development of deep learning, new methods are continually emerging. DeepAR (Salinas et al., 2020) uses recurrent neural networks (RNNs) to model the transitions of hidden states and generates a Gaussian distribution for predictions. Following the autoregressive paradigm, DeepState (Rangapuram et al., 2018) and DSSMF (Li et al., 2019) combine state space models with deep learning to improve forecasting accuracy. MANF (Feng et al., 2024) and ProTran (Tang & Matteson, 2021) introduced attention-based methods that enhance the model's ability to capture long-range dependencies, further improving forecasting accuracy. Diffusion models, such as those proposed by TimeGrad (Rasul et al., 2021a), TSDiff (Kollovieh et al., 2023), and CSDI (Tashiro et al., 2021), approach the forecasting task as a denoising process, excelling in handling high-dimensional data. Another approach involves using more complex distribution forms, such as normalizing flows (Rasul et al., 2021b), to further enhance forecasting performance. Compared with RNNbased or State Space models, K 2 VAE also autoregressively models the time series in a linear dynamical system, but mitigates the error accumulation through KalmanNet. Compared with generative diffusion-based or flow-based models, K 2 VAE adopts the VAE structure and follows single-stepgeneration principle, achieves faster inference speed, lower memory occupation, and better performance.
this section cite: ['b62', 'b59', 'b27', 'b17', 'b67', 'b22', 'b68']

Section: VAE for Time Series
Variational Autoencoders (VAEs) (Kingma & Welling, 2014) have found wide applicability across various time series tasks. In time series generation, VAEs synthesize time series by encoding the data into a lower-dimensional latent space and then decoding it to recreate similar sequences, which helps preserve the statistical properties of the original data, making VAEs valuable for data augmentation (Li et al., 2023a;Desai et al., 2021). In time series imputation, VAEs recover missing values by learning the underlying latent structure of the data (Boquet et al., 2019;Li et al., 2021). By capturing temporal dependencies and relationships, they help restore incomplete time series with high accuracy. In time series anomaly detection, VAEs are used to learn the expected patterns within time series data and flag deviations that indicate anomalous behavior (Huang et al., 2022a;Wang et al., 2024f). In time series forecasting, TimeVAE (Desai et al., 2021) and D 3 VAE (Li et al., 2022) are tailored for short-term probabilistic foercasting tasks. Koopa (Liu et al., 2023), as a strong baseline based on Koopman Theory, is tailored for long-term deterministic forecasting by adopting multi-scale MLP structures, which also falls short in probabilistic forecasting. Compared to these methods, K 2 VAE is tailored for LPTSF, which considers the inherent nonlinearity of time series through a KoopmanNet and tackles the error accumulation through a KalmanNet, thus enhancing the ability to predict long-term future distributions. This facilitates better decision-making in dynamic and uncertain environments.
this section cite: ['b15', 'b0', 'b26', 'b15', 'b28', 'b42']

Section: Conclusion
In this work, we propose a VAE-based probabilistic forecasting model called K 2 VAE to solve PTSF. By leveraging the KoopmanNet, K 2 VAE transforms nonlinear time series into a linear dynamical system, which allows for a more effective representation of state transitions and the inherent process uncertainties. Furthermore, the KalmanNet provides a solution to model the uncertainty in the linear dynamical system, mitigating the error accumuation in long-term forecasting tasks. Through comprehensive experiments, we demonstrate that K 2 VAE not only outperforms existing state-of-the-art methods in both short-and long-term probabilistic forecasting tasks, but also achieves fascinating model efficiency.
In the future, we hope to continuously study the one-step generation paradigm in time series probabilistic modeling to further improve the model performance and efficiency. Another primary direction is to pionner the exploration of foundation probabilistic time series forecasting models, which can work effectively in zero-shot scenarios.
this section cite: []

Section: References
Ref_id:b0 Title: Missing data in traffic estimation: A variational autoencoder imputation method Year: (2019)
Ref_id:b1 Title: Distribution of residual autocorrelations in autoregressive-integrated moving average time series models Year: (1970)
Ref_id:b2 Title: Random forests Year: (2001)
Ref_id:b3 Title: Unsupervised time series outlier detection with diversity-driven convolutional ensembles Year: (2022)
Ref_id:b4 Title: Lightweight time series classification with adaptive ensemble distillation Year: (2023)
Ref_id:b5 Title: Xgboost: A scalable tree boosting system Year: (2016)
Ref_id:b6 Title: Gim: A million-scale benchmark for generative image manipulation detection and localization Year: (2024)
Ref_id:b7 Title: AimTS: Augmented series and image contrastive learning for time series classification Year: (2025)
Ref_id:b8 Title: Monotonic neural ordinary differential equation: Timeseries forecasting for cumulative data Year: (2023)
Ref_id:b9 Title: Triangular, variable-specific attentions for long sequence multivariate time series forecasting Year: (2022)
Ref_id:b10 Title: Towards spatio-temporal aware traffic time series forecasting Year: (2022)
Ref_id:b11 Title: Correlation-aware cross-modal attention network for fashion compatibility modeling in ugc systems Year: (2024)
Ref_id:b12 Title: Real-time localization and bimodal point pattern analysis of palms using uav imagery Year: (2024)
Ref_id:b13 Title: Detection and geographic localization of natural objects in the wild: A case study on palms Year: (2025)
Ref_id:b14 Title: Periodicity decoupling framework for long-term series forecasting Year: (2024)
Ref_id:b15 Title: Timevae: A variational auto-encoder for multivariate time series generation Year: (2021)
Ref_id:b16 Title: MDTP: A multi-source deep traffic prediction framework over spatio-temporal trajectory data Year: (2021)
Ref_id:b17 Title: Multi-scale attention flow for probabilistic time series forecasting Year: (2024)
Ref_id:b18 Title: Ssdts: Exploring the potential of linear state space models for diffusion models in time series imputation Year: (2025)
Ref_id:b19 Title: Monash time series forecasting archive Year: (2021)
Ref_id:b20 Title: Ecomark 2.0: empowering eco-routing with vehicular environmental models and actual vehicle fuel consumption data Year: (2015)
Ref_id:b21 Title: betavae: Learning basic visual concepts with a constrained variational framework Year: (2017)
Ref_id:b22 Title: Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting Year: (2023)
Ref_id:b23 Title: Hamiltonian systems and transformation in hilbert space Year: (1931)
Ref_id:b24 Title: Linearization in the large of nonlinear systems and koopman operator spectrum Year: (2013)
Ref_id:b25 Title: Causal recurrent variational autoencoder for medical time series generation Year: (2023)
Ref_id:b26 Title: Variational auto-encoders based on the shift correction for imputation of specific missing in multivariate time series Year: (2021)
Ref_id:b27 Title: Learning interpretable deep state space model for probabilistic time series forecasting Year: (2019)
Ref_id:b28 Title: Generative time series forecasting with diffusion, denoise, and disentanglement Year: (2022)
Ref_id:b29 Title: Daanet: Dual attention aggregating network for salient object detection Year: ()
Ref_id:b30 Title: CP2M: Clustered-Patch-Mixed Mosaic Augmentation for Aerial Image Segmentation Year: (2025)
Ref_id:b31 Title: Dual Dynamic U-Net for Highly-Efficient Cloud Segmentation Year: (2025)
Ref_id:b32 Title: TSMF-Bench: Comprehensive and unified benchmarking of foundation models for time series forecasting Year: (2025)
Ref_id:b33 Title: Segment recurrent neural network for long-term time series forecasting Year: (2023)
Ref_id:b34 Title: Cocv: A compression algorithm for time-series data with continuous constant values in iot-based monitoring systems Year: (2024)
Ref_id:b35 Title: Modeling long-term time series forecasting with 1k parameters Year: (2024)
Ref_id:b36 Title: Cyclenet: Enhancing time series forecasting through modeling periodic patterns Year: ()
Ref_id:b37 Title: Benchmarking and revisiting time series forecasting methods in cloud workload prediction Year: (2025)
Ref_id:b38 Title: Towards llm-empowered multivariate time series forecasting via cross-modality alignment Year: (2025)
Ref_id:b39 Title: Calf: Aligning llms for time series forecasting via cross-modal fine-tuning Year: (2025)
Ref_id:b40 Title: Non-stationarity matters for long-term time series forecasting Year: (2025)
Ref_id:b41 Title: The elephant in the room: Towards a reliable time-series anomaly detection benchmark Year: (2024)
Ref_id:b42 Title: Learning nonstationary time series dynamics with koopman predictors Year: (2023)
Ref_id:b43 Title: itransformer: Inverted transformers are effective for time series forecasting Year: (2024)
Ref_id:b44 Title: Scoring rules for continuous probability distributions Year: (1976)
Ref_id:b45 Title: Less is more: Efficient time series dataset condensation via two-fold modal matching Year: (2024)
Ref_id:b46 Title: A unified replay-based continuous learning framework for spatio-temporal prediction on streaming data Year: (2024)
Ref_id:b47 Title: A parameter-efficient federated framework for streaming time series anomaly detection via lightweight adaptation Year: (2025)
Ref_id:b48 Title: A time series is worth 64 words: Long-term forecasting with transformers Year: (2023)
Ref_id:b49 Title: Ising-traffic: Using ising machine learning to predict traffic congestion under uncertainty Year: (2023)
Ref_id:b50 Title: Magicscaler: Uncertainty-aware, predictive autoscaling Year: (2023)
Ref_id:b51 Title: Magicscaler: Uncertainty-aware, predictive autoscaling Year: (2023)
Ref_id:b52 Title: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b53 Title: Variational autoencoder for deep learning of images, labels and captions Year: (2016)
Ref_id:b54 Title: TFB: towards comprehensive and fair benchmarking of time series forecasting methods Year: (2024)
Ref_id:b55 Title: A comprehensive survey of deep learning for multivariate time series forecasting Year: (2025)
Ref_id:b56 Title: Time series forecasting made easy Year: (2025)
Ref_id:b57 Title: TAB: Unified benchmarking of time series anomaly detection methods Year: ()
Ref_id:b58 Title: DUET: Dual clustering enhanced multivariate time series forecasting Year: (2025)
Ref_id:b59 Title: Deep state space models for time series forecasting Year: (2018)
Ref_id:b60 Title: Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting Year: (2021)
Ref_id:b61 Title: Multivariate probabilistic time series forecasting via conditioned normalizing flows Year: ()
Ref_id:b62 Title: Probabilistic forecasting with autoregressive recurrent networks Year: (2020)
Ref_id:b63 Title: Dynamic mode decomposition of numerical and experimental data Year: (2010)
Ref_id:b64 Title: Financial time series forecasting with deep learning : A systematic literature review Year: (2005)
Ref_id:b65 Title: Embedded systems programming Year: (2001)
Ref_id:b66 Title: Solar wind speed prediction via graph attention network Year: ()
Ref_id:b67 Title: Probabilistic transformer for time series analysis Year: (2021)
Ref_id:b68 Title: CSDI: conditional score-based diffusion models for probabilistic time series imputation Year: (2021)
Ref_id:b69 Title: Attention is all you need Year: (2017)
Ref_id:b70 Title: Drift doesn't matter: Dynamic decomposition with diffusion reconstruction for unstable multivariate time series anomaly detection Year: (2023)
Ref_id:b71 Title: Entire space counterfactual learning for reliable content recommendations Year: (2024)
Ref_id:b72 Title: Entire space counterfactual learning for reliable content recommendations Year: (2024)
Ref_id:b73 Title: Spot-i: Similarity preserved optimal transport for industrial iot data imputation Year: (2024)
Ref_id:b74 Title: An accurate and interpretable framework for trustworthy process monitoring Year: (2024)
Ref_id:b75 Title: Debiased recommendation via wasserstein causal balancing Year: (2025)
Ref_id:b76 Title: Enhancing code llms with reinforcement learning in code generation Year: (2024)
Ref_id:b77 Title: Lightgts: A lightweight general time series forecasting model Year: (2025)
Ref_id:b78 Title: Towards a general time series forecasting model with unified representation and adaptive transfer Year: (2025)
Ref_id:b79 Title: Revisiting vae for unsupervised time series anomaly detection: A frequency perspective Year: (2024)
Ref_id:b80 Title: An introduction to the kalman filter Year: (1995)
Ref_id:b81 Title: Timesnet: Temporal 2d-variation modeling for general time series analysis Year: ()
Ref_id:b82 Title: Prompt categories cluster for weakly supervised semantic segmentation Year: (2024)
Ref_id:b83 Title: Generative prompt controlled diffusion for weakly supervised semantic segmentation Year: (2025)
Ref_id:b84 Title: Image fusion for cross-domain sequential recommendation Year: (2025)
Ref_id:b85 Title: AutoCTS+: Joint neural architecture and hyperparameter search for correlated time series forecasting Year: (2023)
Ref_id:b86 Title: AutoCTS++: zero-shot joint neural architecture and hyperparameter search for correlated time series forecasting Year: (2024)
Ref_id:b87 Title: CATCH: Channel-aware multivariate time series anomaly detection via frequency patching Year: ()
Ref_id:b88 Title: Fully automated correlated time series forecasting in minutes Year: (2025)
Ref_id:b89 Title: FITS: modeling time series with 10k parameters Year: (2024)
Ref_id:b90 Title: Unsupervised path representation learning with curriculum negative sampling Year: (2021)
Ref_id:b91 Title: Context-aware path ranking in road networks Year: (2022)
Ref_id:b92 Title: Simplets: An efficient and universal model selection framework for time series forecasting Year: (2023)
Ref_id:b93 Title: Tsec: An efficient and effective framework for time series classification Year: (2024)
Ref_id:b94 Title: Score: Story coherence and retrieval enhancement for ai narratives Year: (2025)
Ref_id:b95 Title: Dsformer: A double sampling transformer for multivariate time series long-term prediction Year: (2023)
Ref_id:b96 Title: Ginar: An end-to-end multivariate time series forecasting model suitable for variable missing Year: (2024)
Ref_id:b97 Title: Ginar+: A robust end-to-end framework for multivariate time series forecasting with missing values Year: (2025)
Ref_id:b98 Title: Ich-scnet: Intracerebral hemorrhage segmentation and prognosis classification network using clip-guided sam mechanism Year: (2024)
Ref_id:b99 Title: Ichpro: Intracerebral hemorrhage prognosis classification via joint-attention fusion-based 3d cross-modal network Year: (2024)
Ref_id:b100 Title: Ich-prnet: a crossmodal intracerebral haemorrhage prognostic prediction method using joint-attention interaction mechanism Year: (2025)
Ref_id:b101 Title: Are transformers effective for time series forecasting Year: (2023)
Ref_id:b102 Title: ProbTS: Benchmarking point and distributional forecasting across diverse prediction horizons Year: ()
Ref_id:b103 Title: Can mllms guide weakly-supervised temporal action localization tasks Year: (2024)
Ref_id:b104 Title: Distilling semantic priors from sam to efficient image restoration models Year: (2024)
Ref_id:b105 Title: Adapting sam to image manipulation detection by cross-view automated prompt learning Year: (2025)
Ref_id:b106 Title: Rethinking pseudo-label guided learning for weakly supervised temporal action localization from the perspective of noise correction Year: (2025)
Ref_id:b107 Title: Multiple time series forecasting with dynamic graph modeling Year: (2023)
Ref_id:b108 Title: Informer: Beyond efficient transformer for long sequence time-series forecasting Year: (2021)
