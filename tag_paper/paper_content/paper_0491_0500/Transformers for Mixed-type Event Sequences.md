Title: Transformers for Mixed-type Event Sequences
Abstract: Event sequences appear widely in domains such as medicine, finance, and remote sensing, yet modeling them is challenging due to their heterogeneity: sequences often contain multiple event types with diverse structures-for example, electronic health records that mix discrete events like medical procedures with continuous lab measurements. Existing approaches either tokenize all entries, violating natural inductive biases, or ignore parts of the data to enforce a consistent structure. In this work, we propose a simple yet powerful Marked Temporal Point Process (MTPP) framework for modeling event sequences with flexible structure, using a single unified model. Our approach employs a single autoregressive transformer with discrete and continuous prediction heads, capable of modeling variable-length, mixed-type event sequences. The continuous head leverages an expressive normalizing flow to model continuous event attributes, avoiding the numerical integration required for inter-event times in most competing methods. Empirically, our model excels on both discrete-only and mixed-type sequences, improving prediction quality and enabling interpretable uncertainty quantification. We make our code public at https://github.com/czi-ai/FlexTPP.

Section: Introduction
Time series are a ubiquitous data modality, arising in a wide range of domains such as electronic health records [Wornow et al., 2023], high-frequency trading [Bacry et al., 2015], click stream prediction [Gündüz and Özsu, 2003], hardware and access logs in cybersecurity [Fortino et al., 2023], and earthquake monitoring in remote sensing [Ogata, 1998]. Particularly common are time series with irregular time intervals between adjacent events (e.g., lab test times), where each event is associated with metadata known as marks (e.g., test results). Such time series are known as Marked Temporal Point Processes (MTPPs) [Daley and Vere-Jones, 2008].
Despite their broad applicability, existing MTPP models typically handle only simplified forms of realworld event data. Prior work focuses on discrete marks such as event types [Mei and Eisner, 2017] or sets of items [Chang et al., 2024], or on continuous marks such as spatio-temporal features [Chen et al., 2021, Dong et al., 2024]. In contrast, many real datasets include mixed-type metadata that combines discrete attributes (e.g., diagnosis codes, transaction types) with structured continuous values (e.g., measurements, durations). The length and structure of this metadata also vary across events-for instance, due to different medical test panels or variable-length action logs in cybersecurity.
Ignoring these heterogeneities leads to a mismatch between model assumptions and real data. Recently, Event Stream GPT [McDermott et al., 2023] modeled heterogeneous event structures, but it treats continuous marks with unimodal Gaussians, limiting expressivity and autoregressive sequence 39th Conference on Neural Information Processing Systems (NeurIPS 2025).  generation. To address this, we propose a streamlined MTPP architecture that leverages expressive distributions for all variable types, yielding a unified and powerful joint probabilistic model. As illustrated in Figure 1, our approach converts each event sequence into a single mixed-type vector and models it with a Transformer-based autoregressive network. Discrete event types act as "headers" that determine variable event structure, and separate continuous and discrete prediction heads capture heterogeneous mark attributes.
this section cite: ['b46', 'b2', 'b18', 'b14', 'b7', 'b29', 'b3', 'b5', 'b8', 'b27']

Section: Mixed-type Autoregressive Model Conditional Event Sequence
With this more expressive mark space, we can fully exploit conditioning MTPPs on external input. While prior work considered limited forms of conditioning-such as static covariates-with minimal impact on prediction [Šeputis et al., 2022, Verheugd et al., 2020, Isik et al., 2023], we treat conditioning as central, addressing problems that cannot be solved without it. This broader perspective allows us to use conditional MTPPs as a flexible probabilistic framework for structural mixed-type regression.
Together, our contributions are as follows:
• We propose FLEXTPP, a flexible Transformer-based MTPP framework that supports variable-length, mixed-type marks, encompassing both discrete and continuous event attributes (section 4.1). Beyond standard generative modeling, we treat MTPPs as a structured prediction framework, conditioning event sequences on auxiliary input to solve regression and prediction tasks (section 4.2).
• FLEXTPP is intensity-free, avoiding the numerical integration required by competing intensity-based approaches. Empirically, it also outperforms both intensity-free and intensitybased methods on the discrete-only EasyTPP benchmark [Xue et al., 2024] (section 5.1).
• We demonstrate how FLEXTPP's flexibility can be leveraged in practice: modeling heterogeneous events in electronic health records, performing event-series annotation with uncertainty quantification and extracting event dependency structure (sections 5.2 to 5.4).
Together, we propose a simple and versatile framework to model heterogeneous event sequences.
this section cite: ['b59', 'b44', 'b22', 'b50']

Section: Related Work
There is extensive literature on modeling marked temporal point processes (MTPPs); we summarize our work in relation to existing methods in Table 1, and give a more detailed discussion below.
The predominant line of work models MTPPs with so-called Hawkes processes [Hawkes, 1971], parameterizing an intensity function that captures the rate at which the next event will occur. Several neural realizations have been proposed, first using recurrent neural networks [Du et al., 2016, Mei andEisner, 2017] and later Transformers [Zhang et al., 2020, Zuo et al., 2020, Yang et al., 2021], Table 1: Overview of related MTPP methods in the literature. To the best of our knowledge, we are the first to model expressive distributions to events with mixed data types and varying dimensions. In addition, we consider conditional MTPPs, where the point process depends on auxiliary input. [*] collects [Du et al., 2016, Mei and Eisner, 2017, Zhang et al., 2020, Zuo et al., 2020, Yang et al., 2021, Zhuzhel et al., 2024, Chang et al., 2025, Gao et al., 2024, Xu and Zha, 2017, Liu and Quan, 2024, Isik et al., 2023].
this section cite: ['b21', 'b29', 'b53', 'b58', 'b52', 'b9', 'b29', 'b58', 'b52', 'b57', 'b15', 'b47', 'b25', 'b22']

Section: Data types Mark dimension Conditional Discrete Continuous Multiple Variable
Classical Hawkes Process [Hawkes, 1971] ✓ [Omi et al., 2019, Shchur et al., 2020] ✓ ✗ ✗ ✗ ✗ Conditional event generators [Dong et al., 2024] ✗ ✓ ✓ ✗ ✗ Set-valued MTPPs [Chang et al., 2024
✗ ✗ ✗ ✗ Neural Hawkes Process [*] ✓ ✗ ✗ ✗ ✓ Instensity-Free TPP
] ✓ ✗ ✓ ✓ ✗ Neural spatio-temporal process [Chen et al., 2021] ✗ ✓ ✓ ✗ ✗ Event Stream GPT [McDermott et al., 2023] ✓ ✓/ ✗ ✓ ✓ ✓ Ours ✓ ✓ ✓ ✓ ✓
convolutional networks [Zhuzhel et al., 2024], and state-space models [Chang et al., 2025, Gao et al., 2024]. Other notable directions model the next-time distribution instead of intensities [Omi et al., 2019, Shchur et al., 2020], consider mixtures of series [Xu and Zha, 2017], or leverage the inductive biases of pretrained language models Liu and Quan [2024]. While the aforementioned methods only consider MTPPs where each event is associated with one discrete mark (the event type), other kinds of marks have also been considered. Chen et al. [2021] model spatio-temporal point processes using Neural ODEs, where marks are purely continuous. Similarly, Dong et al.
[2024] model multi-dimensional continuous marks using joint generative models such as diffusion models. Chang et al. [2024] model sets of discrete marks. Similar to our approach, McDermott et al.
[2023] propose a system to learn heterogeneous event structures, but they do not learn an expressive model for continuous event data, limiting performance. We generalize the previous work into one probabilistic framework that jointly handles variable-length and mixed-type mark spaces.
Some work has considered conditioning MTPPs on external input such as for predicting failures of water pipes [Verheugd et al., 2020] or in medical context [Šeputis et al., 2022[Šeputis et al., , Isik et al., 2023]]. Our Transformer-based architecture allows for natural conditioning, which we exploit both for feeding covariates and as the input to regression tasks.
In the context of the related work, our model is Transformer-based, intensity-free, and allows for variable-length and mixed-type marks.
In generative modeling, autoregressively predicting variable-length sequences is at the core of language models such as GPT [Mikolov et al., 2010, Radford et al., 2018]. Mixed data types are often integrated in the input but cannot be generated, for example, in vision-understanding models such as Flamingo [Alayrac et al., 2022]. Jointly modeling mixed modalities has been achieved by discretizing (tokenizing) continuous values or fusing other generative models and language models. We refer to [Xu et al., 2023] for a comprehensive overview. Similar to our modeling approach, [Fakoor et al., 2020] propose and [Strauss and Oliva, 2021] perform joint modeling of discrete and continuous data with a transformer backbone, inspired by autoregressive models for continuous data [Germain et al., 2015]. However, neither of these works exploit the full flexibility of modeling variable-length, mixed-type data.
this section cite: ['b21', 'b33', 'b37', 'b8', 'b3', 'b57', 'b15', 'b33', 'b37', 'b47', 'b25', 'b5', 'b3', 'b44', 'b59', 'b30', 'b35', 'b0', 'b48', 'b12', 'b40', 'b16']

Section: Background: Marked Temporal Point Process
A natural way to describe the distribution of event sequences over time is through a Marked Temporal Point Process (MTPP). An MTPP is a stochastic process describing a sequence of events {(t i , m i )} T i=1 , where each event is characterized by its occurrence time t i ∈ R + and an associated mark m i ∈ M. While M traditionally describes a single discrete or continuous variable (see section 2 and table 1), we generalize it in section 4.1 to mixed discrete-continuous spaces. Event sequences are inherently stochastic: both the timing and the mark of the next event are inherently uncertain. Their joint distribution can be factorized using the chain rule as:
p({(t i , m i )} T i=1 ) = T i=1 p(t i , m i | H ti ),(1)
where H ti = {(t j , m j ) | t j < t i } denotes the history of events up to t i , capturing both temporal and mark dependencies.
Ignoring the marks for a minute (we will re-introduce them in section 4.1), a key design choice is how to model the one-dimensional distribution of the next event time p(t i | H ti ).
First, intensity-based methods model the time distribution via a learned intensity function λ(t i | H ti ), representing the instantaneous rate of events at a time t [Daley and Vere-Jones, 2008]. Formally, the intensity function λ(t | H t ) defines the infinitesimal expected event rate, linking the point process to its conditional likelihood via integration over time:
p(t i | H ti ) = λ(t i | H ti , [m i ]) exp - ti ti-1 λ(t i | H ti ) dt .(2)
Since the intensity function can be any positive function, this formulation yields highly flexible time distributions. However, the numerical integration can be unstable and be expensive both at training (likelihood evaluation) and inference (sampling), depending on the modeling of the intensity function.
Second, intensity-free methods avoid numerical integration by directly modeling event times using parametric distributions p φ (t i | H ti ) conditioned on the history. Common choices are mixture models or one-dimensional normalizing flows, with parameters learned as a function of past events [Shchur et al., 2020]. This allows efficient evaluation and sampling.
Our proposed framework in section 4 is intensity-free. Outperforming previous models in section 5.1, it overcomes concerns about expressivity of intensity-free models [Chang et al., 2025].
this section cite: ['b7', 'b37']

Section: FlexTPP

this section cite: []

Section: Variable-length, mixed-discrete-continuous events
Real-world events often carry heterogeneous information: some attributes are discrete (e.g., event type), others are continuous (e.g., laboratory test results such as glucose levels), and their number may vary depending on the event. To model such data, we introduce a unified mixed-modality framework that extends MTPPs to variable-length, mixed-type marks.
We are given events i that consist of an arrival time t i and an associated mark m i that may contain a variable number of discrete and continuous components. Instead of treating these parts separately, we flatten the entire event sequence into a single sequence of scalar values X = (X 1 , . . . , X L ), together with a corresponding type vector D = (D 1 , . . . , D L ) indicating whether each entry is continuous or discrete.
D (X 1 , . . . , X L ) = L l=1 p D l (X l | X <l ),
where each conditional distribution matches the data type:
p D l (X l | X <l ) = Cat(X l | X <l ) if D l = disc, p density (X l | X <l ) if D l = cont .(4)
Here, Cat is a categorical distribution and p density a continuous density.
Algorithm 1 Sampling from Mixed-Type Autoregressive Model
1: Input: Lengths N (m type ) and types d(m type ) for each event type m type = 1, . . . , M . 2: Initialize empty sequence X = (), i = 1, l = 1. 3: while true do 4: Sample time X l = t i ∼ p density (X l |X 1 , . . . , X l-1 ). 5: Sample event type X l+1 = m type i ∼ Cat(X l+1 |X 1 , . . . , X l ). 6: if m type i = EOS then 7: break. 8: end if 9:
for j = 1, . . . , N (m type i ) do 10:
X l+j+1 = m ij add ∼ Cat(X l+j+1 |X 1 , . . . , X l+j ) if d(m type i ) j = disc p density (X l+j+1 |X 1 , . . . , X l+j ) else.
11: end for 12: Update indices: l = l + 2 + |G mi1 | and i = i + 1. 13: end while 14: return (t i , m type i , m add i ) i
The above approach is a unified autoregressive model of variable-length, mixed-type MTPPs with parameters θ. We train it by minimizing the negative loglikelihood of a dataset D of time series:
min θ - (X,D)∈D log p D,θ (X).(5)
Algorithm 1 shows how to sample from this model. Each event begins by sampling a continuous value for the arrival time, followed by an event type that determines the structure of the remaining entries. Sampling a special end-of-sequence token EOS terminates the sampling.
this section cite: []

Section: Conditional Marked Temporal Point Processes
The general-purpose structure of our extension of MTPPs to variable data types makes them a natural choice for the output format in supervised prediction tasks. For example, we will later formulate the detection of events in a time series as a conditional marked time point process (see section 5.3). Similarly, for health records, feeding demographic information can increase modeling accuracy (see section 5.2). To this end, we modify the setup in section 4.1 to include conditional input C, so that the likelihood of each sequence is conditioned on C:
p D (X|C) = L l=1 p D l (X l |C, X 1 , . . . , X l-1
). Now that we have generalized MTPPs to handle heterogeneous mark data and condition on auxiliary information, we next instantiate this modeling approach with a Transformer architecture.
this section cite: []

Section: Flexible Marked Temporal Point Process (FLEXTPP)
We implement the the conditional likelihood p D l (X l | X <l ) in eq. ( 4) with an autoregressive model. The idea is to map the history X 1...,i-1 to a fixed-dimensional representation:
ϕ l = ϕ(D l ; C, X 1 . . . X l-1 ) ∈ R dm ,(6)
which is then used to parameterize the conditional distribution of X l . As illustrated in Figure 2, we implement the autoregressive computation with a Transformer [Vaswani et al., 2017]. This architecture captures long-range dependencies between events [Zhang et al., 2020, Yang et al., 2021, Zuo et al., 2020], and lets the size of intermediate representations grow naturally with the dimensionality of the marks.
For predicting discrete dimensions, we map ϕ l into a categorical distribution as in language modeling,
Cat(x; ϕ l ) = [softmax(φ disc (ϕ l ))] x ,(7)
where we turn ϕ i into logits φ disc (ϕ l ) with a small fully-connected neural network.
For the continuous distributions, we map ϕ i to the parameters φ cont (ϕ i ) of a one-dimensional normalizing flow, as is common for continuous autoregressive models [Germain et al., 2015]. We use a rational-quadratic spline z = f φcont (x) [Durkan et al., 2019]. The flow defines a one-dimensional (conditional) density via the change of variables equation:
p density (x; ϕ l ) = N (z = f φcont(ϕ l ) (x); m = 0, σ = 1)|f ′ φ (x)|.(8)
Note that for simplicity, we use eq. ( 8) both for the time and continuous mark dimensions ("intensityfree", see section 3). We call the above model Flexible Marked Temporal Point Process FLEXTPP when no condition is present, and we refer to FLEXTPP-C whenever it is conditioned.
this section cite: ['b43', 'b53', 'b52', 'b58', 'b16', 'b10']

Section: Experiments
We evaluate our generalized MTPP framework in practice. We first confirm that FLEXTPP(-C) reliably works with discrete mark spaces on the EasyTPP benchmark [Xue et al., 2024] in section 5.1. We then demonstrate in section 5.2 how our generalized mark structure improves predicting medical procedures in electronic health records extracted from the EHRSHOT dataset [Wornow et al., 2023].
Finally, we propose to use MTPPs to make structured predictions in annotating time series in section 5.3. This MTPP heavily relies on its condition and can predict several types of annotations together with their structurally different properties.
For the experiments that concern generalized mark spaces, we compare the following versions of our model:
Conditional + discrete event type mark: p(t 1 , m type 1 , ¨m add 1 , . . . , t T , m type T , ¨m add T |C),(9)
Unconditional + full mark (FLEXTPP):
p(t 1 , m type 1 , m add 1 , . . . , t T , m type T , m add T |C), (10
) Conditional + full mark (FLEXTPP-C): p(t 1 , m type 1 , m add 1 , . . . , t T , m type T , m add T |C),(11)
The first model in eq. ( 9) only models time and a discrete event type as a mark, corresponding to how the bulk of the literature on MTPPs would model the data. We give these models access to the condition to allow for a fair comparison to the most general variant, FLEXTPP-C. The second model in eq. ( 10), in contrast, does not have access to the condition and only models the marginal marked time point process. Here, we measure how much the condition helps in making a prediction. Finally, our model in eq. ( 11) accepts a condition and jointly models all discrete and continuous marks.
We evaluate all models on a held-out test set in terms of negative log-likelihood (NLL), capturing the generative quality and the uncertainty in making predictions.
To compare our conditional model to the unconditional variant, we compute the negative logarithms of eqs. ( 10) and ( 11). Intuitively, a lower value means that the conditional model can make use of the condition to more accurately model the MTPP.
Comparing these likelihoods to the ones of the model that only captures discrete event types is not directly possible, as eq. ( 9) does not model the additional mark dimensions. We therefore also report Table 2: Our model sets a new SOTA in four out of five datasets in terms of Negative Log-Likelihoods (lower is better, ↓) on EasyTPP Datasets [Xue et al., 2024], notably outperforming both intensity-free TPP [Shchur et al., 2020] and intensity-based methods. Marks are single discrete event type. Baseline values are due to [Chang et al., 2025].
Model Amazon Retweet Taxi Taobao StackOverflow RMTPP [Du et al., 2016] 2.136 (0.003) 7.098 (0.217) -0.346 (0.002) -1.003 (0.004) 2.480 (0.019) NHP [Mei and Eisner, 2017] -0.129 (0.012) 6.348 (0.000) -0.514 (0.004) -1.157 (0.004) 2.241 (0.002) SAHP [Zhang et al., 2020] 2.074 (0.029) 6.708 (0.029) -0.298 (0.057) 1.646 (0.083) 2.341 (0.058) THP [Zuo et al., 2020] 2.096 (0.002) 6.659 (0.007) -0.372 (0.002) 1.712 (0.011) 2.338 (0.014) AttNHP [Yang et al., 2021] -0.484 (0.077) 6.499 (0.028) -0.493 (0.009) -1.259 (0.022) 2.194 (0.016) IFTPP [Shchur et al., 2020] -0.496 (0.002) 10.344 (0.016) -0.453 (0.002) -1.318 (0.017) 2.233 (0.009) MHP [Gao et al., 2024] -0.496 (0.002) 10.344 (0.016) -0.453 (0.002) -1.318 (0.017) 2.233 (0.009) S2P2 [Chang et al., 2025] -0.781 (0.011) 6.365 (0.003) -0.522 (0.004) -1.304 (0.039) 2.163 (0.009) FLEXTPP (Ours) -0.633 (0.039) 5.646 (0.070) -0.763 (0.005) -1.402 (0.013) 2.133 (0.004)
the negative log-likelihood of the time and the event type mark dimensions under each model:
- T i=1 log p(t i , m type i |[C], m type 1..i-1 , [m add 1..i-1 ]).(12)
We pass in the additional mark dimensions and the condition only if the model version accepts them.
A smaller loss means that the model can better estimate the arrival time and type of the next event when it has seen all marks of the previous event instead of just arrival times and marks. For example, in the clinical settings, it measures whether the next medical procedure prediction improves if we know the continuous lab results.
We give all details to replicate our experiments in appendix A.
this section cite: ['b50', 'b46', 'b50', 'b37', 'b9', 'b29', 'b53', 'b58', 'b52', 'b37', 'b15']

Section: EasyTPP Datasets
The EasyTPP benchmark [Xue et al., 2024] collects five datasets to compare models fitting MTPPs.
All datasets have a discrete mark space modeling event types only, see appendix A.1. Table 2 shows that our intensity-free, Transformer-based MTPP outperforms both intensity-based and previous intensity-free methods.
This is an important data point in the MTPP modeling space, as it shows that intensity-free methods can perform well, all while avoiding numerical integration at training and inference time. In the next sections, we generalize the mark space beyond discrete marks.
this section cite: ['b50']

Section: Patient Record Data
As a first generalized experiment, we model a subset of EHR data from the EHRSHOT benchmark [Wornow et al., 2023]. Unlike standard EHR setups that capture only discrete events like procedures [Chang et al., 2025], our construction also includes continuous lab results, enabling a richer, mixed-modality representation of patient trajectories. This structure improves next-event prediction and enhances uncertainty quantification by integrating diverse clinical signals. See appendix A.2 for details.
To compile the dataset, we subset the patients diagnosed with 20 different diseases such as type 2 diabetes, dyspnea, atrial fibrillation, etc., to create disease-specific longitudinal analyses. For each disease, we capture the most used Current Procedural Terminology (CPT-4) codes that denote medical services and the most common procedures as discrete events, and model disease-related lab results as continuous events. Additionally, demographic data are incorporated as conditional inputs, enabling a better modeling of patient-specific dynamics.
Table 3 demonstrates the effectiveness of our proposed method. Besides the baseline in eq. ( 9) ("Procedures") which only models procedures as discrete events, we also compare FLEXTPP and FLEXTPP-C to another baseline ("+ Lab Tests") which additionally measures discrete lab test types but without their corresponding continuous results. By jointly modeling continuous lab results alongside discrete procedure types and conditioning on demographic information, our FLEXTPP-C approach achieves the best performance on both measures of negative log-likelihood ("Full" and
Table 3: By incorporating continuous lab results via eq. ( 11), our flexible models outperform baselines in predicting time and type of medical procedures: An MTPP with discrete marks would either not model lab events at all [Chang et al., 2025] ("Procedures"), or model that a lab test exists, but ignore its value ("+ Lab Tests"). Conditioning on demographic covariates (our FLEXTPP-C) always improves prediction quality for procedures and in the majority of cases for all events. Standard deviations in tables 9 and 10.
Time + Discrete Procedure NLL (↓) Full NLL (↓) Dataset Procedures + Lab Tests ESGPT ESGPT-C FLEXTPP FLEXTPP-C ESGPT ESGPT-C FLEXTPP FLEXTPP-C Type 2 diabetes mellitus 0.691 0.202 0.186 0.182 0.147 0.143 0.754 0.758 0.641 0.638 Transplanted kidney 0.755 0.170 0.199 0.192 0.134 0.130 0.859 0.863 0.670 0.668 Transplanted lung 0.912 0.089 0.078 0.075 0.068 0.063 0.851 0.857 0.581 0.546 Dyspnea 0.681 0.162 0.162 0.155 0.121 0.117 0.560 0.549 0.463 0.438 Atrial fibrillation 0.715 0.081 0.075 0.071 0.062 0.059 0.388 0.376 0.204 0.192 Cardiac transplant disorder 0.887 0.033 0.034 0.033 0.029 0.027 0.552 0.594 0.408 0.393 End-stage renal disease 0.697 0.166 0.169 0.164 0.128 0.117 0.757 0.753 0.580 0.578 Transplanted heart 0.869 0.041 0.033 0.032 0.029 0.028 0.736 0.701 0.546 0.521 Congestive heart failure 0.704 0.134 0.130 0.128 0.103 0.100 0.846 0.861 0.525 0.509 Chronic pain 0.735 0.099 0.095 0.093 0.076 0.073 0.765 0.743 0.162 0.165 Neoplasm of female breast 0.716 0.043 0.040 0.038 0.032 0.030 0.608 0.608 0.516 0.461 Obstructive sleep apnea 0.695 0.093 0.090 0.089 0.061 0.059 0.710 0.676 0.195 0.187 Diabetes with complication 0.718 0.176 0.164 0.160 0.133 0.129 0.694 0.703 0.579 0.582 Anemia 0.655 0.164 0.165 0.158 0.127 0.126 0.322 0.296 0.238 0.245 Coronary artery disease 0.692 0.144 0.138 0.134 0.108 0.105 0.966 0.986 0.636 0.625 Hypothyroidism 0.735 0.224 0.212 0.208 0.166 0.159 0.666 0.664 0.524 0.528 Acute myeloid leukemia 0.729 0.182 0.160 0.154 0.124 0.117 0.727 0.735 0.567 0.561 Depressive disorder 0.683 0.136 0.135 0.129 0.108 0.105 0.813 0.826 0.608 0.545 Transplanted liver 0.799 0.183 0.192 0.186 0.144 0.142 0.907 0.902 0.727 0.725 Acute kidney injury 0.645 0.201 0.191 0.185 0.144 0.139 0.706 0.711 0.621 0.605
"Time + Discrete Procedure"). ESGPT and ESGPT-C (McDermott et al. [2023]) are also implemented as additional baselines, where continuous lab results and arrival times are modeled by Gaussian and Log-Normal mixture heads, respectively, instead of the normalizing flow head used in our proposed methods. These results validate the advantage of incorporating continuous event types, patientspecific conditioning, and the expressiveness in modeling continuous attributes using normalizing flow in multimodal health record modeling.
this section cite: ['b46']

Section: Time Series Annotation
Finding eventful subsequences in time series data is crucial in many real-world applications. For example, in healthcare, timely and accurate identification of critical events can significantly impact diagnosis and treatment, and in audio annotation, where a continuous time series is transformed into an interpretable description.
A traditional approach is to have a classifier p(m type |t, C) predict the presence of an event type m type at a given time t in an input sequence C [Zhao et al., 2017]. However, fig. 3 (top right) illustrates how this is fundamentally limited when several events are present at the same time: It expresses uncertainty in the event class instead of predicting both classes to be present. Another restriction is that classifiers are often evaluated on small windows to save compute for long input time series.
We lift these restrictions by directly predicting the target sequence of events as a MTPP under our flexible framework FLEXTPP-C. This allows for parallel events, and compute is naturally bounded by the number of events. In this setting, each event is characterized by a start timestamp (continuous), a type label (discrete), a duration (continuous), and additional event data (mixed length, mixed type). Because our framework is fully probabilistic, we get uncertainties for all these quantities.
We evaluate our model on a set of synthetic input time series. Our dataset is based on a noisy harmonic oscillator, a common model in biology, physics, acoustic and mechanical systems. It is given by the following system of stochastic differential equations:
dx = vdt + σ x dw x , dv = (-γv -ω 2 x + f (t))dt + σ v dw v .(13)
For each time series in our dataset, we first choose base values for the constants ω, γ, σ x , σ v . We then randomly sample a list of events that alter the dynamics over some amount of time by changing one or several of the constants, or add an external force f (t). See appendix A.3 for details.
Figure 3 shows a typical example time series with annotated events. Our FLEXTPP-C predicts event times and properties, together with uncertainty intervals. Note how an MTPP with discrete marks only captures the start times and type of events. Table 4 evaluates eqs. ( 9) and ( 11) to compare negative log-likelihoods. We also train a sliding window classifier [Zhao et al., 2017] with a finite
Input a) Input time series 0% 50% 100% c) Sliding window classifier External force Base frequency change Position noise No event E.f. B.f.c. P.n. d) Only discrete marks Time E.f. B.f.c. P.n. Omega factor: 0.89 Omega factor: 1.13 Position noise add: 1.74 Force magnitude: 1.36 Force shape: 0 Periodicity: 0.09 b) Ground truth annotations Time E.f. B.f.c. P.n. Omega factor: 0.89 ± 0.01 Omega factor: 1.10 ± 0.08 Position noise add: 1.74 ± 0.32 Force magnitude: 0.89 ± 0.28 Force shape: 0 (38%), 1 (33%) Periodicity: 0.06 ± 0.03 e) FlexTPP (ours) Table 4: We formulate probabilistic time series annotation as a conditional MTPP. Our FLEXTPP-C achieves the best AUC ROC scores by jointly sampling events, event types and durations. FLEXTPP without condition naturally achieves random AUC ROC (0.5), but yields a useful baseline for negative log-likelihoods (NLL). A conditional MTPP with only discrete marks does not predict event durations, but achieves decent AUC ROC when assuming the average duration for each event. As an additional baseline, we compare to a CNN-based time series classifier [Zhao et al., 2017] with limited window size. Standard deviations estimated from five runs, note that the type NLL is conditioned on correct event duration. Time + Type NLL in bits/event, Full NLL in bits/dim.
NLL (↓) AUC ROC (↑) Model Time + Type NLL Full NLL External force Damping change Base frequency Position noise No Event Mean Single-class FLEXTPP 0.250 (0.005) -0.010 (0.003) 0.5 (0.0) 0.5 (0.0) 0.5 (0.0) 0.5 (0.0) 0.5 (0.0) 0.5 (0.0) Only Discrete Marks -0.474 (0.006) ✗ ✗ ✗ ✗ ✗ ✗ ✗ + avg duration ✗ ✗ 0.96 (0.01) 0.63 (0.02) 0.990 (0.003) 0.98 (0.01) 0.91 (0.01) 0.90 (0.01) FLEXTPP-C -0.867 (0.005) -0.930 (0.007) 0.990 (0.01) 0.66 (0.02) 0.998 (0.001) 0.997 (0.003) 0.93 (0.01) 0.92 (0.01) Sliding Window Classifier ✗ ✗ 0.86 (0.06) 0.50 (0.01) 0.96 (0.04) 0.87 (0.06) 0.89 (0.04) 0.82 (0.02)
window size. We evaluate all methods using the area under the receiver-operator curve (AUC ROC) [Hanley and McNeil, 1982], a common metric to identify how well predicted events overlap with the ground truth [Schmidl et al., 2022]. FLEXTPP-C annotates most event types almost perfectly, and outperforms the random baseline significantly on the difficult "damping change" event type, which is hard to detect by construction. Naively, discrete mark MTPPs cannot be evaluated with AUC ROC since they do not predict event durations, so we evaluate this metric using the average event duration.
this section cite: ['b54', 'b54', 'b54', 'b19', 'b36']

Section: Extracting Event Dependencies
How do events depend on one another under a learned model? Some neural Hawkes process variants provide interpretability through structures their explicit triggering kernels q(t i , m i , t, m), so that λ(t i , m i |H ti ) = (t,m)∈Ht i q(t i , m i , t, m) [Isik et al., 2023, Zhu et al., 2022]. Such kernels directly encode pairwise influence structures, providing global access to the modeled dependency structure.
Our framework offers a complementary, local interpretability mechanism. To derive it analogously, we can solve eq. ( 2) for the intensity function λ(t i |H ti ) given a time density p(t i |H ti ):
λ(t i , m i |H ti ) = p(t i , m i |H ti ) 1 - t ti-1 p(s, m i |H ti )ds(14)
Here, we have reintroduced marks m i .
We can extract the local triggering kernel via eq. ( 14) by evaluating λ(t 2 , m 2 | (t 1 , m 1 )). To demonstrate this, fig. 7 in appendix A.4 replicates a synthetic setup from Isik et al. [2023]. We show that our model can recover an influence curve matching the ground truth with RMSE 0.01, indicating that our learned event dependencies are both accurate and recoverable.
These approaches exhibit a broader tradeoff between global interpretability and modeling flexibility. Kernel-based models provide a concise, fully inspectable description of event interactions, but their rigidity can misrepresent complex real-world dynamics. For instance, some domains exhibit statedependent effects-such as advertising "fatigue," where repeated exposures first increase and later decrease response probability-that cannot be captured well by monotone additive influence kernels.
In such cases, the model is not expressive enough for the true dependency structure, and the resulting explanations appear structured but do not reflect the true data-generating process.
Our model instead supports local interpretability: one can ask how a specific event or subset of events affects intensities or subsequent event sequences, yielding contextual, fine-grained explanations. This aligns with complex event sequences, where influences are often situational rather than universal.
this section cite: ['b22', 'b56', 'b22']

Section: Conclusion
In this work, we propose a simple yet powerful method to model more general Marked Time Point Processes -ones with mixed-type (discrete and continuous) marks and auxiliary contextual information. Our intensity-free modeling approach treats MTPP as one joint sequence consisting of auxiliary information (if available), event arrival times, and marks; we then model the sequence autoregressively using a single Transformer with appropriate output heads. We observe that incorporating additional information generally improves modeling performance; this is expected from an information-theoretic perspective -conditioning reduces entropy, which is the minimum achievable negative log-likelihood [Cover and Thomas, 2006]. From a modeling perspective, we observe that complex intensity functions may not be necessary -intensity-free modeling with good one-dimensional density estimators may be enough to model temporal processes well.
Our generalized MTPP framework also highlights how MTPPs can be used as a versatile prediction tool for annotating input time series. Crucially, MTPPs naturally model the joint probability of each prediction, so that each sampled annotation from the model is a consistent explanation of the input. The formulation also naturally allows for overlapping events. We envision future work to extend this paradigm for annotating spatio-temporal sequences to enhance uncertainty quantification in the annotation of time series, such as audio, video, as well as scientific measurements.
Broader Impact. Modeling critical data, such as in the context of medicine, can cause harm through wrong or wrongly interpreted predictions, such as those arising from biases in the training data and distribution shifts. On the positive side, modeling additional variables and incorporating context can increase prediction accuracy and enable novel applications.
this section cite: ['b6']

Section: Limitations
Limitations of autoregressive modeling. Our approach inherits the limitations of autoregressive modeling. In particular, when scaling to high-dimensional marks such as images, alternative generative modeling approaches may be more suitable [Chang et al., 2025]. Another direction is to jointly learn a representation of the marks that is better suited for downstream modeling [Tschannen et al., 2024]. Similarly, if the marks follow special structure such as special geometry or topology, autoregressive models cannot be applied faithfully and generic methods such as [Sorrenson et al., 2024] can be used to model these dimensions.
Alternative data types. In our work, we consider discrete and continuous data. Other mark modalities, such as sets or ordered discrete variables, could be studied in the future.
this section cite: ['b42', 'b39']

Section: Limitations of Transformer backbone.
The compute of Transformers scale quadratically with the length L of the underlying sequence: O(L 2 ). This complexity can make them unsuitable for modeling very long sequences. However by construction, our framework is compatible with other autoregressive models such as recurrent neural networks [Elman, 1990], linear-attention Transformers [Katharopoulos et al., 2020], or state-space models [Gu et al., 2021] that have better length scaling.
this section cite: ['b11', 'b24', 'b17']

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We explain our generalized framework and back up our claims with experiments.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: See limitations in section 7.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA] Justification: No theoretical results.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We give all details in appendix A.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11. Safeguards Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: We only release synthetic data.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: We give the references as required.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes] Justification: We will release our code and synthetic dataset upon publication under a permissive license.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets ( if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: No such research performed. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: Not applicable. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Depending on the type, we sample the event properties via table 13. The ramp time avoids sudden changes in parameters, so that the events are harder to detect.
The events replace the parameters in eq. ( 13) with the following dynamic coefficients:
σ x (t) = σ x,0 + E: m type E =D α(t; E)σ x,E ,(15)
ω(t) = w 0 E: m type E =C (1 + α(t; E)ω E ),(16)
γ(t) = γ 0 E: m type E =B (1 + α(t; E)γ E ),(17)
f (t) = E: m type E =A α(t; E)βs E t - 2π T t .(18)
Here, the sums/products run over all events E of the corresponding type m type E . The function α(t; E) turns off the event outside of its domain and interpolates over the ramp time at the beginning and end
this section cite: []

Section: Class probabilities per timestamp
To compare the different modeling approaches, we evaluate the class probabilities p(y(t)|time series) at each time stamp given the time series. To get these predictions about the class distribution at time t from the MTPP models, we approximate the trained models p(y(t)|time series) by averaging event occurrence over 100 sampled event annotations. For the MTPP with only discrete marks, we again use the average event duration for all predicted events. Figures 4 to 6 visualize the resulting probabilities for the three approaches. One can see that all three methods mostly agree at which time some event starts. However, the discrete-marks MTPP is biased when multiple events occur since it does not predict the duration of the events. Also, it learns heavier distributions for the events.
The classifier is less well calibrated in terms of regards of which class it expects. We think that this is due to a fundamental limitation of the formulation of annotation as classification: By construction, a classifier-based approach cannot differentiate between being unsure about which event class to predict and there being several events present. For example, let's assume there are two event types A and B and the classifier predicts the following values: p(event A at t|window) = 1/2, p(event B at t|window) = 1/2, p(no event at t|window) = 0.
(30) This result could be caused by (a) the model is certain that both events A and B being present, or (b) there is some event, but it is unclear what type of event (A or B) there is. We think that this ambiguity in the task representation leads to an overall high uncertainty in the predictions of the classifier. Table 4 quantify the performance of the models the macro-averaged Receiver Operating Characteristic -Area Under the Curve (ROC AUC) [Hanley and McNeil, 1982] on the per-timestamp y(t) vectors. This metric is computed by adopting a one-vs-rest strategy: for each class, a binary ROC AUC score is calculated by treating the current class as the positive class and all others as negative. The final score is then obtained by averaging the individual AUCs across all classes:
Macro AUC = 1 m m k=1 AUC k .(31)
Here, each AU C k represents the ROC AUC score for class k = 1, . . . , m + 1 (compare appendix A.3.4).
Each ROC curve is insensitive to a badly calibrated model because it is based on the ranking of predicted scores, not their absolute values. Therefore, the CNN Classifier still yields useful ROC AUCs despite the notably worse performance in predicting p(y(t)|time series) in figs. 4 to 6.
this section cite: ['b19']

Section: A.4 Extracting triggering kernel
For this experiment, we sample N = 100, 000 event sequences from the following intensity function:
λ(t|H t ) = µ + ti<t t-ti<β α sin π(t -t i ) β .(32)
We choose µ = 0.1, α = 0.2, β = 1 and sample T = 4 events per sequence.
We then train a FLEXTPP with hyperparameters given in table 17.
Figure 7 shows the above kernel and the kernel extracted from our model. To get it from our model, we average over t 1 = 0.1, 0.2, . . . , 10.0 and evaluate the intensity function from the learned probabilities via eq. ( 14) as a function of t 2 = t 1 + 0.033, . . . , t 1 + 10.0.
this section cite: []

Section: B Details on Architecture
Figure 1 in the main text visualizes our architecture. Figure 8 visualizes the architecture with more details and shows the sampling according to algorithm 1. The multi-head attention block is causal, meaning that token i can only attend to tokens 1, . . . , i -1. The transformer works with a token dimension of d E = n head d K and the feed-forward networks in each transformer block have a hidden dimension of d ff .
this section cite: []

Section: 
Answer: [Yes] Justification: See https://github.com/czi-ai/FlexTPP. Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental setting/details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We give all details in appendix A.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We give all details in appendix A.
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
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We give all details in appendix A.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We have read and comply with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10. Broader impacts Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [Yes] Justification: We provide a statement in the conclusion in section 6. Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to Justification: LLMs are not a core method in this research.
Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Flamingo: a visual language model for few-shot learning Year: (2022)
Ref_id:b1 Title: Framework for Easily Invertible Architectures (FrEIA) Year: (2018)
Ref_id:b2 Title: Hawkes processes in finance Year: (2015)
Ref_id:b3 Title: Probabilistic modeling for sequences of sets in continuous-time Year: (2024-05)
Ref_id:b4 Title: Deep continuous-time state-space models for marked event sequences Year: ()
Ref_id:b5 Title: Neural spatio-temporal point processes Year: (2021)
Ref_id:b6 Title: Elements of Information Theory Year: (2006)
Ref_id:b7 Title: An Introduction to the Theory of Point Processes. Probability and Its Applications Year: (2008)
Ref_id:b8 Title: Conditional generative modeling for highdimensional marked temporal point processes Year: (2024)
Ref_id:b9 Title: Recurrent Marked Temporal Point Processes: Embedding Event History to Vector Year: (2016-08)
Ref_id:b10 Title: Neural Spline Flows Year: (2019)
Ref_id:b11 Title: Finding structure in time Year: (1990)
Ref_id:b12 Title: TraDE: Transformers for density estimation Year: (2020)
Ref_id:b13 Title: Falcon and The PyTorch Lightning team Year: (2019-03)
Ref_id:b14 Title: Identification and prediction of attacks to industrial control systems using temporal point processes Year: (2023-05)
Ref_id:b15 Title:  Year: (2024-07)
Ref_id:b16 Title: MADE: Masked autoencoder for distribution estimation Year: (2015-07)
Ref_id:b17 Title: Efficiently modeling long sequences with structured state spaces Year: (2021)
Ref_id:b18 Title: A web page prediction model based on click-stream tree representation of user behavior Year: (2003)
Ref_id:b19 Title: The meaning and use of the area under a receiver operating characteristic (ROC) curve Year: (1982)
Ref_id:b20 Title: Array programming with NumPy Year: (2020)
Ref_id:b21 Title: Spectra of some self-exciting and mutually exciting point processes Year: (1971)
Ref_id:b22 Title: Hawkes process with flexible triggering kernels Year: (2023)
Ref_id:b23 Title: Numerical integration of Ito or Stratonovich SDEs Year: (2015)
Ref_id:b24 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b25 Title: TPP-LLM: Modeling Temporal Point Processes by Efficiently Fine-Tuning Large Language Models Year: (2024-10)
Ref_id:b26 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b27 Title: Event stream GPT: A data pre-processing and modeling library for generative, pre-trained transformers over continuoustime sequences of complex events Year: (2023)
Ref_id:b28 Title: Data Structures for Statistical Computing in Python Year: (2010)
Ref_id:b29 Title: The neural hawkes process: A neurally self-modulating multivariate point process Year: (2017)
Ref_id:b30 Title: Recurrent neural network based language model Year: (2010)
Ref_id:b31 Title: Justifying recommendations using distantly-labeled reviews and fine-grained aspects Year: (2019)
Ref_id:b32 Title:  Year: (1998-06)
Ref_id:b33 Title: Fully neural network based model for general temporal point processes Year: (2019)
Ref_id:b34 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b35 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b36 Title: Anomaly detection in time series: a comprehensive evaluation Year: (2022)
Ref_id:b37 Title: Intensity-free learning of temporal point processes Year: (2020)
Ref_id:b38 Title: Super-convergence: Very fast training of neural networks using large learning rates. In Artificial intelligence and machine learning for multi-domain operations applications Year: (2019)
Ref_id:b39 Title: Learning distributions on manifolds with free-form flows Year: (2024)
Ref_id:b40 Title: Arbitrary conditional distributions with energy Year: (2021)
Ref_id:b41 Title: The pandas development team. pandas-dev/pandas: Pandas Year: (2020-02)
Ref_id:b42 Title: JetFormer: An Autoregressive Generative Model of Raw Images and Text Year: (2024-11)
Ref_id:b43 Title: Attention is all you need Year: (2017)
Ref_id:b44 Title: Predicting Water Pipe Failures with a Recurrent Neural Hawkes Process Model Year: (2020-10)
Ref_id:b45 Title: FOILing nyc's taxi trip data Year: (2014)
Ref_id:b46 Title: EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models Year: (2023-12)
Ref_id:b47 Title: A dirichlet mixture model of hawkes processes for event sequence clustering Year: (2017)
Ref_id:b48 Title: Multimodal Learning with Transformers: A Survey Year: (2023-05)
Ref_id:b49 Title: HYPRO: a hybridly normalized probabilistic model for long-horizon prediction of event sequences Year: (2022)
Ref_id:b50 Title: EasyTPP: Towards open benchmarking temporal point processes Year: (2024)
Ref_id:b51 Title: Hydra -A framework for elegantly configuring complex applications Year: (2019)
Ref_id:b52 Title: Transformer embeddings of irregularly spaced events and their participants Year: (2021)
Ref_id:b53 Title: Self-attentive Hawkes process Year: (2020-07)
Ref_id:b54 Title: Convolutional neural networks for time series classification Year: (2017)
Ref_id:b55 Title: Seismic: A self-exciting point process model for predicting tweet popularity Year: (2015)
Ref_id:b56 Title: Neural spectral marked point processes Year: (2022)
Ref_id:b57 Title: Continuous-time convolutions model of event sequences Year: (2024-09)
Ref_id:b58 Title: Transformer Hawkes process Year: (2020-07)
Ref_id:b59 Title: Challenges and opportunities in applying neural temporal point processes to large scale industry data Year: (2022-08)
