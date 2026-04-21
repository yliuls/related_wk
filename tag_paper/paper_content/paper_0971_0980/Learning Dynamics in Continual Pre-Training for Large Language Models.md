Title: Learning Dynamics in Continual Pre-Training for Large Language Models
Abstract: Continual Pre-Training (CPT) is a popular and effective method for applying strong foundation models to specific downstream tasks. In this work, we explore the learning dynamics throughout the CPT process for large language models. We specifically focus on how general and downstream domain performance evolves at each training step, with performance measured by validation losses. We observe that the CPT loss curve fundamentally characterizes a transition from an initial pre-training trajectory to a new, domainspecific one, conceptualized as a shift between two hidden loss curves. This transition can be described by decoupling the effects of distribution shift and learning rate annealing. We derive a CPT scaling law that combines these two factors, enabling the prediction of loss at any (continual) training step and across various learning rate schedules. Our formulation presents a comprehensive understanding of several critical factors in CPT, including loss potential, peak learning rate, training steps, and replay ratio. Moreover, our approach can be adapted to optimize training hyper-parameters for different CPT goals, such as balancing general and domain-specific performance. Extensive experiments demonstrate that our scaling law holds across various CPT datasets and hyper-parameters.

Section: Introduction
In recent years, large language models (LLMs) have exhibited versatile abilities and garnered significant academic and industrial attention (Dubey et al., 2024;OpenAI, 2023).
Continual Pre-Training (CPT) of LLMs aims to enhance their abilities in specific downstream domains (e.g. coding, finance, math) while mitigating the substantial costs associated with re-training (Chen et al., 2023a;C ¸agatay Yıldız et al., 2024;Ibrahim et al., 2024).
CPT primarily involves a trade-off between performance on general and downstream domains. It is widely observed that improvements on downstream tasks may come at the expense of degrading performance on general domain tasks, a phenomenon known as catastrophic forgetting (French, 1999;Gupta et al., 2023). Recently, some scaling laws have been proposed for CPT scenarios. For example, Hernandez et al. (2021b) and Barnett (2024) discovered a law describing how data transfer effectiveness scales with fine-tuning dataset size and model size. Que et al. (2024) and Gu et al. (2024) proposed a law to find the optimal replay ratio to balance general and downstream performances.
However, very few studies have attempted to quantitatively describe the learning dynamics of CPT, particularly how performance varies on general and downstream domains throughout the CPT process. We have two primary research questions (RQs): (1) Can we derive an accurate law describing the influence of as many variables as possible on the final CPT performance? (2) Can we trace the performance of LLMs throughout the entire CPT process, rather than only the final performance? Studying the first RQ will help researchers investigate various factors that affect CPT performance and facilitate hyper-parameters optimization through prediction; studying the second RQ will help the community understand the learning dynamics of LLMs at each step of the CPT process, providing deeper insights and theoretical guidance for subsequent CPT research.
Following previous works (Gupta et al., 2023;Ibrahim et al., 2024;Que et al., 2024), we trace performance changes using validation losses on corresponding domains. We find that the CPT loss curve acts as a transfer curve and can be described by decoupling the effects of distribution shift and learning rate (LR) annealing. Specifically, the distribution shift between the pre-training (PT) and CPT data leads to a deviation in the loss curve, while LR annealing results in a loss decrease in both the PT and CPT phases. By analyzing various loss curves, we discover a CPT scaling law that integrates these two factors, enabling accurate prediction of  losses throughout the entire CPT phase.
Our proposed scaling law provides a comprehensive model of how key variables affect the training dynamics of CPT, such as loss potential (defined in section 3.3), peak LR, training steps, and replay ratio. We demonstrate how these variables jointly affect model performance at each CPT step, and how to optimize these hyper-parameters for better CPT performance. By applying our scaling law, several valuable conclusions emerge. For example: (1) PT models with higher loss potential can better adapt to downstream domains in CPT;
(2) The performance degradation on the PT domain during the CPT phase is inevitable if the turning length is infinitely large, which implies that the PT model is adequately trained or the distribution shift between the PT and CPT data is very large; (3) For specific CPT goals, like balancing performance between the PT and CPT domains, or optimizing out-of-domain performance, our scaling law can predict the optimal training hyper-parameters such as the loss potential, peak LR, and PT dataset replay ratio.
this section cite: ['b8', 'b23', 'b10', 'b14', 'b1', 'b36', 'b12', 'b14', 'b23', 'b36']

Section: Pilot Observation

this section cite: []

Section: Task Formulation
We investigate the dynamics of performance in both general and downstream domains during the CPT process. Following previous works (Ibrahim et al., 2024;Que et al., 2024;Gu et al., 2024;Hernandez et al., 2021a), we assess model performance by examining the validation loss on the PT dataset D pt and the CPT dataset D cpt .
this section cite: ['b23', 'b36', 'b12']

Section: Experimental Setup.
Our main experiments employ LLaMA-like models (Dubey et al., 2024) with 106M to 1.7B non-embedding parameters. We use FineWeb (Penedo et al., 2024) as D pt and Knowledge-Pile (Fei et al., 2024) as D cpt . We leverage different LRS in the PT and CPT phases (see Fig. 1). More details are provided in Appendix B.
this section cite: ['b8', 'b35', 'b9']

Section: Observation.
As observed in previous studies (Ibrahim et al., 2024;Gupta et al., 2023), during the CPT process, the D pt validation loss tends to increase (Fig. 1b and Fig. 1e), whereas the D cpt validation loss decreases (Fig. 1c and Fig. 1f). Moreover, in both PT and CPT phases, the loss curve is significantly influenced by the LRS. For example, the loss decreases rapidly when the LR anneals, which is observed in our prior work (Tissue et al., 2024).
this section cite: ['b23', 'b14']

Section: CPT Transfer Loss Curve
To enhance our understanding of the CPT training dynamics, we train two additional loss curves: the hidden PT curve trained on D pt and the hidden PT curve trained on D cpt .
Hidden PT Curve trained on D pt . This curve represents the loss when the model is consistently pre-trained using D pt with the same LRS as used in the CPT phase.
Hidden PT Curve trained on D cpt . This curve depicts the loss when the model is trained from scratch on D cpt , while adhering to the same training setups (such as LRS) as those applied in the PT and CPT phases.
0 10000 20000 30000 40000 50000 60000 Step 2.8 3.0 3.2 3.4 3.6 D pt Validation Loss Distribution Shift Pre-training Hidden Pre-training CPT in 10K Steps CPT in 20K Steps CPT in 30K Steps CPT Step (t) 0.0 0.1 0.2 0.3 Distribution Shift Power: 0.368(1 (0.008t + 1) 0.173 ) R 2 =0.994 Exp: 0.195(1 e 0.001t ) R 2 =0.926 (a) Dpt (FineWeb) validation loss shift. 0 10000 20000 30000 40000 50000 60000 Step 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 D cpt Validation Loss Distribution Shift Pre-training Hidden Pre-training CPT in 10K Steps CPT in 20K Steps CPT in 30K Steps CPT Step (t) 0.0 0.2 0.4 0.6 Distribution Shift Power: 0.702(1 (1.345t + 1) 0.131 ) R 2 =0.985 Exp: 0.485(1 e 0.007t ) R 2 =0.801  Transfer Curve. As shown in Fig. 1, the CPT loss curve acts as a transfer curve between these two hidden PT curves; i.e., the CPT loss deviates from the hidden PT curve trained on D pt and converges towards the hidden PT curve trained on D cpt . The discrepancy between the transfer loss curve and the hidden PT curve trained on D pt is called distribution shift. As the number of CPT steps approaches infinity, the CPT loss is expected to converge to the hidden PT curve trained on D cpt .
Finding 1. The process of CPT is how the loss curve transitions from the hidden PT curve trained on D pt to the hidden PT curve trained on D cpt .
this section cite: []

Section: Continual Learning Dynamics Law
We quantitatively analyze the transfer curve by modeling the effects of LR annealing and distribution shift.
this section cite: []

Section: LR Annealing
Without data transfer, the CPT loss curve would follow the trajectory of the hidden PT curve trained on D pt . Tissue et al. (2024) introduced a scaling law to describe the loss dynamics at each step t as affected by LR annealing:
L(t) = L 0 + A • S -α 1 -C • S 2 ,(1)
where the forward area S 1 = t i=1 η i is the summed LR, and the annealing area -k is a term affected by LR annealing. L 0 , A, C, α are constant positive parameters to be fitted. λ = 0.999 is a hyper-parameter related to the momentum term.
S 2 = t i=1 i k=1 (η k-1 -η k ) • λ i
The loss in the CPT process without distribution shift (denoted as L base (t)) follows this law, i.e.,
L base (t) = L 0 +A•(S pt 1 +S cpt 1 ) -α -C •(S pt 2 +S cpt 2 ), (2
)
where t denotes the CPT step, and S pt 1 (S pt 2 ) and S cpt 1 (S cpt 2 ) are the forward (annealing) areas at the PT and CPT stages, respectively.
this section cite: []

Section: Distribution Shift Term
The distribution shift term describes the deviations from the hidden PT curve trained on D pt . This shift reflects the distributional distance between D pt and D cpt . Many studies (Ibrahim et al., 2024;Wang et al., 2024;Parmar et al., 2024) have highlighted the impact of LRS at the CPT stage, implying that this shift should be also affected by the LRS. We first analyze the form of the distribution shift term with a constant LR to isolate the effects of LRS, then we incorporate the forward area into the equation to accurately describe the distribution shift term for different LRS.
this section cite: ['b23', 'b33']

Section: Constant LRS.
We first use a constant LR in both PT and CPT phases. To study the relationship between distribution shift and the PT model state, we continually pre-train the model starting from different transfer points. As shown in Fig. 2, these distribution shift terms tend to overlap regardless of the transfer starting point. This overlap suggests that the distribution shift term is independent of transfer starting points or PT model checkpoints.
We compare to fit the distribution shift term using exponential and power-law forms, and find the best fit to be ∆L(t) = B • (1 -(E • t + 1) -β ). We do not adopt the simple power-law form ∆L(t) = B • t -β to ensure that ∆L(0) = 0. We leverage this equation to fit the transfer loss curve of both D pt and D cpt validation sets, as shown in Fig. 2.
Other LRS. When considering the effect of LRS, we find that the LR values, i.e., the forward area in Eq. 1, significantly affects the distribution shift term. The smaller forward area in the CPT results in a smaller distribution
0 10000 20000 30000 40000 50000 60000 Step 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Learning Rate ×10 4 Pre-Training LRS Continual Pre-Training LRS (a) WSD PT and CPT LRS. 10000 20000 30000 40000 50000 60000 Step 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 Dpt Validation Loss L = 3.057 + 0.490(S pt 1 + S cpt 1 ) 0.497 + 0.366(1 (28S cpt 1 + 1) 0.189 ) 0.299S pt 2 0.298S cpt 2 Peak Point Distribution Shift + LR Re-Warmup WSD Pre-Training Truth Loss WSD Continual Pre-Training Truth Loss WSD Pre-Training Fitted Loss WSD Continual Pre-Training Fitted Loss (b) Dpt (FineWeb) Validation Loss. 10000 20000 30000 40000 50000 60000 Step 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 Dcpt Validation Loss L = 3.024 + 0.412(S pt 1 + S cpt 1 ) 0.677 0.666(1 (5662S cpt 1 + 1) 0.146 ) 0.278S pt 2 0.324S cpt 2 Distribution Shift + LR Re-Warmup WSD Pre-Training Truth Loss WSD Continual Pre-Training Truth Loss WSD Pre-Training Fitted Loss WSD Continual Pre-Training Fitted Loss (c) Dcpt (Knowledge Pile) Validation Loss. 0 10000 20000 30000 40000 50000 60000 Step 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Learning Rate ×10 4 Pre-Training LRS Continual Pre-Training LRS (d) Cosine PT and CPT LRS. 10000 20000 30000 40000 50000 60000 Step 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 Dpt Validation Loss L = 3.057 + 0.490(S pt 1 + S cpt 1 ) 0.497 + 0.366(1 (28S cpt 1 + 1) 0.189 ) 0.299S pt 2 0.298S cpt 2 Peak Point Distribution Shift + LR Re-Warmup Cosine Pre-Training Truth Loss Cosine Continual Pre-Training Truth Loss Cosine Pre-Training Fitted Loss Cosine Continual Pre-Training Fitted Loss (e) Dpt (FineWeb) Validation Loss. 10000 20000 30000 40000 50000 60000 Step 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 Dcpt Validation Loss L = 3.024 + 0.412(S pt 1 + S cpt 1 ) 0.677 0.666(1 (5662S cpt 1 + 1) 0.146 ) 0.278S pt 2 0.324S cpt 2 Distribution Shift + LR Re-Warmup Cosine Pre-Training Truth Loss Cosine Continual Pre-Training Truth Loss Cosine Pre-Training Fitted Loss Cosine Continual Pre-Training Fitted Loss  shift, as shown in different transfer curves in Fig. 1b vs. 1e (or Fig. 1c vs. 1f). Hence, following Tissue et al. (2024), we replace the training steps t with the forward area S cpt 1 in the CPT phase:
∆L(t) = B • (1 -(1 + E • S cpt 1 ) -β ),(3)
which instead adopts S cpt 1 to represent the training amount in CPT stage, considering the impact of LR values.
this section cite: []

Section: Final Transfer Curve
We combine the effect of LR annealing (Eq. 2) and distribution shift (Eq. 3) to get the equation for the CPT loss:
L(t) = L base (t) + ∆L(t) = L 0 + A • S pt 1 + S cpt 1 -α -C 1 • S pt 2 -C 2 • S cpt2
Scaling law with LR annealing
+ B • 1 -1 + E • S cpt 1 -β Power-law distribution shift(4)
We adopt different coefficients C 1 and C 2 for S pt 2 and S cpt 2 because the distributions of D pt and D cpt are different, and thus result in different annealing effects.
Our equation can predict the loss at any step with any LRS during both the PT and CPT phases. We conduct experiments utilizing the widely adopted WSD (Hu et al., 2024) and cosine (Loshchilov & Hutter, 2016) LRS in the PT and CPT phases (see Fig. 3a and Fig. 3d). We use Eq. 4 to fit all loss curves on the D pt and D cpt validation sets.
As illustrated in the middle and right panels of Fig. 3, our equation successfully captures the trends in loss variations across different LRS throughout the training process. We also use the fitted equation to predict loss curves of other LRS, and the prediction accurately matches the observation (see Fig. 10). Furthermore, the batch size and sequence length may change in the CPT phase. However, our scaling law equation remains adaptable to these hyper-parameter changes, as demonstrated in Appendix F.
Finding 2. The CPT loss curve can be decomposed into a hidden PT curve trained on D pt and a distribution shift term. The hidden PT curve trained on D pt is formalized as a scaling law with LR annealing, whereas the distribution shift term is independent of transfer starting points and adheres to a power-law form.
Transfer Loss Surface. To better understand our formulation, we follow Tissue et al. (2024) to view the loss surface of LLMs as a slide-like transition between surfaces in Fig. 4. The CPT process transitions from one surface to another following a power-law form. A larger distributional distance between D pt and D cpt leads to a steeper slope of the transfer surface, and thus a sharper increase in the D pt loss.
When the LR anneals, the amplitude of the oscillation on the loss surface decreases, and thus the loss also decreases.
In the annealing view, we term the "height" of the current model state as its loss potential. We use this concept to capture the potential for future loss drop via LR annealing. Quantitatively, we can define loss potential as the ratio of the final annealed LR of the PT phase to the peak learning rate in the PT phase.
this section cite: ['b20', 'b28']

Section: Extension to Model Size and Replay Ratio
We attempt to incorporate model size N into our CPT scaling law by analyzing the effect of N on both the LR annealing and distribution shift term. Our experiments show that the distribution shift terms remains unchanged across different model sizes when other settings are fixed (see details in Appendix E). Therefore, we can directly follow Tissue et al. (2024) to integrate an N -related term and use our scaling law to fit and predict CPT loss curves for different model sizes. More discussion is provided in Appendix E.
We also integrate the replay ratio into our scaling law since replaying some data from D pt is a common practice in CPT. Our experiments show that the replay ratio influences the distribution shift term in an exponential manner. By adding a single replay ratio related term, our scaling law can predict the entire training dynamic for different replay ratios, while previous studies (Que et al., 2024) can only predict the final loss. More details are given in Appendix H.
this section cite: ['b36']

Section: Factor Analyses and Applications
In this section, we analyze various factors for CPT and apply our scaling law to provide insights into these factors.
this section cite: []

Section: Loss Potential
Most PT models are trained by annealing to a minimum LR for lower PT losses. However, the optimal PT model for CPT is not necessarily a fully annealed model. We use the concept of loss potential introduced in section 3.3 to describe the degree of annealing for PT models. Specifically, a PT model trained without annealing has a high loss potential, while a PT model that anneals to a zero LR value has a low loss potential. We investigate the impact of loss potential on CPT under two different experimental settings: without or with LR re-warmup.
W/o Re-warmup. In this setting, we set the initial LR for CPT as the final LR in PT and linearly anneal the LR for CPT to zero. We conduct experiments using PT models with different loss potentials (Fig. 5a). As shown in Fig. 5b, models with higher loss potential achieve lower final losses on D cpt . This observation matches the prediction made by our CPT scaling law (Fig. 5c). We also utilize our equation to predict the final loss across various CPT steps, confirming that this trend persists in different settings.
With Re-warmup. A common practice for CPT is to linearly re-warmup the LR from zero to a certain value, such as 10% of the peak LR in PT, before annealing it to zero (Fig. 5d). As shown in Fig. 5e and Fig. 5f, models with high loss potential consistently achieve lower final losses.
We can use our CPT scaling law (Eq. 4) to analyze the impact of loss potentials. Specifically, as the annealing coefficient C 2 > C 1 often holds for D cpt , then allocating a larger annealing area in the CPT phase, i.e., a larger S cpt 2 , facilitates a lower loss. Moreover, models with higher loss potential have larger forward areas S pt 1 and S cpt 1 , which further contribute to a lower loss. Therefore, PT models with high loss potential usually lead to lower D cpt loss. This conclusion is also validated in previous works (Wang et al., 2024).
Finding 3. PT models with higher loss potential consistently achieve lower D cpt validation losses. Hence, we advocate that when releasing opensource models, it is beneficial to release a high loss potential version to facilitate downstream tasks.
this section cite: []

Section: Replay Ratio
The distributional distance between D pt and D cpt significantly influences the distribution shift term in Eq. 4. As shown in Fig. 6a, a more distinct D cpt , Pile of Law (Henderson* et al., 2022), leads to a sharper transfer curve than a more similar D cpt (Knowledge Pile (Fei et al., 2024)).
In CPT, it is a common practice to mix D pt into D cpt based on a certain replay ratio to mitigate the increase of validation loss on D pt . The replay ratio plays a critical role in adjusting the distributional distance between D pt and D cpt since D cpt
PT Total Step 0.0 0.5 1.0 1.5 2.0 Learning Rate ×10 4 10% Loss Potential 30% Loss Potential 50% Loss Potential 100% Loss Potential PT End (a) CPT with different loss potentials (w/o re-warmup setting). 10000 12000 14000 16000 18000 20000 CPT Step 2.56 2.58 2.60 2.62 2.64 2.66 2.68 2.70 Dcpt Validation Loss 10% Loss Potenial Truth Loss 30% Loss Potenial Truth Loss 50% Loss Potenial Truth Loss 100% Loss Potenial Truth Loss (b) Dcpt true loss vs. CPT step for different loss potentials (w/o re-warmup setting). 20 40 60 80 100 Loss Potential % 2.52 2.54 2.56 2.58 2.60 2.62 2.64 2.66 Dcpt Validation Loss 20K CPT Steps 30K CPT Steps 40K CPT Steps 50K CPT Steps (c) Dcpt predicted loss vs. loss potentials for different CPT steps (w/o re-warmup setting). PT Total Step 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Learning Rate ×10 4 0% Loss Potential 10% Loss Potential 50% Loss Potential 100% Loss Potential PT End (d) CPT with different loss potentials (w/ re-warmup). 10000 12000 14000 16000 18000 20000 CPT Step 2.63 2.64 2.65 2.66 2.67 Dcpt Validation Loss 0% Loss Potenial Truth Loss 10% Loss Potenial Truth Loss 50% Loss Potenial Truth Loss 100% Loss Potenial Truth Loss (e) Dcpt true loss vs. CPT step for different loss potentials (w/ re-warmup setting) . 0 20 40 60 80 100 Loss Potential % 2.59 2.60 2.61 2.62 2.63 2.64 2.65 2.66 2.67 Dcpt Validation Loss 20K CPT Steps 30K CPT Steps 40K CPT Steps 50K CPT Steps  is modified to approach D ptfoot_4 . Results in Fig. 6b and Fig. 6c indicate that higher replay ratios lead to smaller distribution shifts and thus effectively decelerate the deviation from D pt . Quantitatively, we find that the replay ratio influences the distribution shift term based on an exponential form, which is elaborated in Appendix H.
this section cite: ['b16', 'b9']

Section: Peak LR
In real scenarios, choosing an appropriate peak LR for rewarmup is important for CPT. Different peak LRs affect the D pt and D cpt validation loss. We leverage Eq. 4 to predict the final loss of different peak LRs. Specifically, we assume the PT model is trained using the WSD LRS. As shown in Fig. 7a and Fig. 7b, a high peak LR in the CPT phase accelerates the decrease of the D cpt validation loss while leading to an increase of the D pt validation loss.
this section cite: []

Section: CPT Training Steps
The number of CPT training steps is also an important hyperparameter. A general observation is that more training steps lead to lower D cpt validation loss. However, the D pt validation loss may exhibit three different patterns based on the state of the PT model and the distributional distance between D pt and D cpt : (1) a continuous rise; (2) an initial rise followed by a decline that does not return to the original loss value; or (3) an initial rise followed by a decline that goes below the original loss value.
As shown in Fig. 7c, we define the critical point (indicated by the blue dashed line) as the convergence value of the D pt loss on the hidden PT curve trained on D cpt . When CPT occurs before this critical point, the D pt loss will first rise and then decline. The final loss may or may not be lower than the original loss. The minimum training steps required to return to the initial loss value are designated as the turning length. Conversely, if CPT occurs after the critical point, achieving a lower D pt loss than the initial value becomes unattainable, regardless of how many steps we train.
Finding 4. Inadequate pre-training or weak distribution shift can result in lower D pt loss values after sufficient CPT steps compared to the PT model. Otherwise, we are unlikely to achieve a lower D pt loss than the PT model, regardless of how many CPT steps we train. In this situation, more training often leads to degraded general performance.
this section cite: []

Section: Balance Between D pt and D cpt Loss
Validation losses on D pt and D cpt typically exhibit a tradeoff in the CPT process. Balancing these losses is critical for optimizing the overall performance of the model during CPT. We define the increase in D pt loss as ∆L Dpt and the
10000 20000 30000 40000 50000 60000 Step 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 FineWeb Loss Weak Distribution Shift Strong Distribution Shift DCPT dataset = Knowledge Pile DCPT dataset = Pile of Law (a) The difference in distribution shift for different Dcpt datasets. 10000 20000 30000 40000 50000 60000 Step 3.20 3.25 3.30 3.35 3.40 3.45 3.50 3.55 FineWeb Loss Strong Distribution Shift Weak Distribution Shift KP:Fineweb=1:0 (0% Replay) KP:Fineweb=2:1 (33% Replay) KP:Fineweb=1:1 (50% Replay) KP:Fineweb=1:2 (67% Replay) KP:Fineweb=0:1 (100% Replay) (b) The distribution shift on the Dpt validation set for different replay ratios. 10000 20000 30000 40000 50000 60000 Step 2.6 2.8 3.0 3.2 3.4 Knowledge Pile Loss Strong Distribution Shift Weak Distribution Shift KP:Fineweb=1:0 (0% Replay) KP:Fineweb=2:1 (33% Replay) KP:Fineweb=1:1 (50% Replay) KP:Fineweb=1:2 (67% Replay) KP:Fineweb=0:1 (100% Replay) (c) The distribution shift on the Dcpt validation set for different replay ratios. Figure 6. We compare the distribution shift for different distributional distances between the Dcpt and Dpt datasets. Additionally, we examine the impact of different replay ratios on the distribution shifts within both the Dcpt and Dpt validation sets. 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Max learning rate 1e 4 3.30 3.31 3.32 3.33 3.34 3.35 3.36 3.37 Dpt Validation Loss 20K CPT Steps 40K CPT Steps 60K CPT Steps 80K CPT Steps (a) Dpt predicted loss vs. peak LRs for different CPT steps. 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Max learning rate 1e 4 2.52 2.54 2.56 2.58 2.60 2.62 2.64 2.66 Dcpt Validation Loss 20K CPT Steps 40K CPT Steps 60K CPT Steps 80K CPT Steps  decrease in D cpt loss as ∆L Dcpt . To balance the loss of D pt and D cpt validation sets, we assign normalized balance coefficient to different validation sets:
min S cpt 1 ,S cpt 2 λ 1 ∆L Dpt + λ 2 ∆L Dcpt s.t. λ 1 + λ 2 = 1 (5
)
where λ 1 and λ 2 are coefficients that should be set based on our prior knowledge of the relative importance of general and downstream performance.
this section cite: []

Section: Optimal Hyper-Parameters
Given the different coefficients λ 1 and λ 2 , there exist some optimal CPT hyper-parameters.
Loss Potential. Fig. 8a shows the optimal loss potential for different values of λ 1 . It can be observed that a small λ 1 corresponds to a large optimal loss potential. This makes sense since a small λ 1 means that the final loss is dominated by the D cpt loss, and thus it is necessary to reserve sufficient loss potential for downstream domains.
Peak LR. We can also predict the optimal peak LR in the CPT process when λ 1 is given (Fig. 8b). A larger λ 1 suggests a preference for minimizing the increase in D pt loss, thereby necessitating a lower peak LR.
this section cite: []

Section: Replay Ratio.
Based on our scaling law with replay ratio Eq. 8, we can determine the optimal replay ratio for each λ 1 (Fig. 8c). The same distribution line (dashed line) in Fig. 8c indicates that the optimal replay ratio should be the same as the target weight λ 1 if we initialize the CPT model randomly rather than from a pre-trained model. Instead, in practice, the optimal replay ratio shifts because the PT model has already been trained on D pt , which causes the curve to deviate and exhibit a wave pattern.
this section cite: []

Section: CPT Training Steps.
As shown in Fig. 13, we can get different turning lengths for different values of λ 1 . When λ 1 is small, the D cpt loss predominates the composite loss λ 1 L Dpt + λ 2 L Dcpt , which consistently remains below the initial value. Conversely, with a moderate λ 1 , there exists a specific step that makes the composite loss equals the inital loss. For a large λ 1 , the composite loss is always higher than the initial loss, which means that CPT is not suitable any more in this situation.
this section cite: []

Section: Out-of-Domain Validation Set
Note that our CPT scaling law is designated to predict losses on D pt and D cpt validation sets, while it is not directly applicable to the out-of-domain
(OOD) validation set D ood . Inspired by previous works (Ye et al., 2024; Zhang et al., 2025) that the OOD validation loss can be represented as a 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1 0 20 40 60 80 100 Optimal Loss Potential % Strong Distribution Shift Moderate Distribution Shift Weak Distribution Shift (a) The optimal loss potential. 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Optimal Max LR ×10 4 Strong Distribution Shift Moderate Distribution Shift Weak Distribution Shift (b) The optimal peak LR. 0.0 0.2 0.4 0.6 0.8 1.0 1 0.0 0.2 0.4 0.6 0.8 1.0 Optimal Replay Ratio 20K CPT Steps 30K CPT Steps 40K CPT Steps Same Distribution (c) The optimal replay ratio. Figure 8. Optimizing hyper-parameters for CPT based on different coefficients to balance general and downstream performance. Strong, moderate, weak distribution shift in (a) and (b) denote different CPT datasets as Pile of Law, Knowledge Pile, and a mixture of 67% FineWeb and 33% Knowledge Pile, respectively. The "same distribution" in (c) represents a reference line where the target weight (λ1) is the same as replay ratio. See Appendix I for more details. 42500 45000 47500 50000 52500 55000 57500 60000 Step 2.8 3.0 3.2 3.4 3.6 3.8 Out-of-Domain Loss C4 = 0.152Dcpt + 0.867Dpt SlimPajama = 0.385Dcpt + 0.622Dpt Stories = -0.516Dcpt + 1.409Dpt C4 = 0.152Dcpt + 0.867Dpt SlimPajama = 0.385Dcpt + 0.622Dpt Stories = -0.516Dcpt + 1.409Dpt C4 = 0.152Dcpt + 0.867Dpt SlimPajama = 0.385Dcpt + 0.622Dpt Stories = -0.516Dcpt + 1.409Dpt C4 Loss SlimPajama Loss Stories Loss (a) D ood dataset truth and predicted loss similar to Dpt. 42500 45000 47500 50000 52500 55000 57500 60000 Step 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 Out-of-Domain Loss Stackexchange = 1.853Dcpt -0.546Dpt Arxiv = 1.631Dcpt -0.609Dpt Books = 0.475Dcpt + 0.611Dpt Stackexchange = 1.853Dcpt -0.546Dpt Arxiv = 1.631Dcpt -0.609Dpt Books = 0.475Dcpt + 0.611Dpt Stackexchange = 1.853Dcpt -0.546Dpt Arxiv = 1.631Dcpt -0.609Dpt Books = 0.475Dcpt + 0.611Dpt Stackexchange Loss Arxiv Loss Books Loss (b) D ood dataset truth and predicted loss similar to Dcpt. linear combination of losses on several base domains, we hypothesize that the loss on D ood can be represented by a linear combination based on D pt and D cpt validation losses:
L D ood = λ ′ 1 L Dpt + λ ′ 2 L Dcpt(6)
We verify this hypothesis and calculate λ ′ 1 and λ ′ 2 for several example OOD datasets in Appendix K. Note that the coefficients λ ′ 1 and λ ′ 2 are related only to datasets and not to other training hyper-parameters.
Loss Prediction of D ood . The D ood validation loss does not adhere to the formulation described in Eq. 4. However, by calculating and specifying the coefficients λ ′ 1 and λ ′ 2 , it becomes feasible to predict the D ood loss curve using a linear combination of the D pt and D cpt loss curves. It is interesting that this problem reduces to the balance between D pt and D cpt loss (Eq. 5). The optimal hyper-parameters such as LR and replay ratio for this setting have been adequately discussed in the previous section.
As shown in Fig. 9, we first calculate the coefficients in Eq. 6 for several OOD datasets and then predict the corresponding validation losses. The almost perfect prediction suggests that our approach are quite effective and practical in real scenarios. Moreover, the calculated coefficient represents the "similarity" between OOD datasets and D pt or D cpt . As Fig. 9 shows, there are two kinds of OOD datasets: (1) D pt -like one (larger λ ′ 1 ) with loss curve upward, and (2) D cpt -like one (larger λ ′ 2 ) with loss curve downward.
Finding 5. There exists an optimal loss potential, peak LR and replay ratio designated to balance D pt and D cpt losses. Besides, the turning lengths vary depending on the different balance weights. Predicting L D ood is equivalent to balancing D pt and D cpt losses by utilization of linear combination tricks.
this section cite: []

Section: Open-Source PT Models
For the majority of LLM communities, the PT models we use are usually not trained by ourselves, but from opensource models. Most training details are not reported for those open-source PT models, i.e., the distribution of D pt , the loss potential, and the PT training hyper-parameters are usually unknown. This inhibits the direct application of our CPT scaling law. To solve this issue, we propose the following methods to make our scaling law become applicable again.
(a) Firstly, for the unknown PT dataset distribution, some methods based on probing (Hayase et al., 2024) have been proposed. Instead, we simply utilize an open-source Common Crawl dataset as a proxy D pt to approximate the distribution of D pt . (b) Secondly, when fitting our scaling law, we regard some variables as unknown parameters to fit. For example, we treat S pt 1 as a parameter that requires fitting to be close to the undisclosed real S pt 1 . (c) Thirdly, as most open-source PT models anneal to a minimal LR to get a better performance nowadays, we assume all open-source models anneal their LR to zero when calculating S cpt 2 . Refer to Appendix G for more details.
To verify our solutions for open-source PT models, we continually pre-train LLaMA3.2-1B (Dubey et al., 2024) and select the RedPajama (Weber et al., 2024) dataset as an proxy D pt . As Fig. 18 in Appendix G shows, the almost perfect fitting and prediction for CPT loss curve of LLaMA3.2-1B suggests the effectiveness of our proposed methods. Moreover, this result also indicates that our scaling law can be easily extended to CPT scenarios with unknown PT model information, demonstrating the superiority of our scaling law to capture the learning dynamics of CPT.
this section cite: ['b15', 'b8']

Section: Discussion
Laws Formulation. The formulation of S 2 in Eq. 1 can have other forms. For example, S 2 could also be a multipower form (Luo et al., 2025), which is proposed following the work of Tissue et al. (2024). We adopt the equation form in Eq. 1 because it has fewer parameters and it works more effectively in practice. We also compare some format variates including adding a LR-weighted coefficient and adding a power term to S 2 (see more details in Appendix J). The experiments show that all formats lead to similar results while our formulation has superiority in simplicity (i.e. fewer parameters).
this section cite: ['b30']

Section: References
Ref_id:b0 Title: Botorch: A framework for efficient monte-carlo bayesian optimization Year: (2020)
Ref_id:b1 Title: An empirical study of scaling laws for transfer Year: (2024)
Ref_id:b2 Title: Random search for hyperparameter optimization Year: (2012)
Ref_id:b3 Title: Lifelong language pretraining with distribution-specialized experts Year: ()
Ref_id:b4 Title: Meditron-70b: Scaling medical pretraining for large language models Year: (2023)
Ref_id:b5 Title: Saullm-7b: A pioneering large language model for law Year: (2024)
Ref_id:b6 Title: Deepseekcoder-v2: Breaking the barrier of closed-source models in code intelligence Year: (2024)
Ref_id:b7 Title: Open language models for south-east asia Year: (2024)
Ref_id:b8 Title: The llama 3 herd of models Year: (2024)
Ref_id:b9 Title: Query of cc: Unearthing large scale domainspecific knowledge from public corpora Year: (2024)
Ref_id:b10 Title: Catastrophic forgetting in connectionist networks Year: (1999)
Ref_id:b11 Title: The Pile: An 800gb dataset of diverse text for language modeling Year: (2020)
Ref_id:b12 Title: CMR scaling law: Predicting critical mixture ratios for continual pre-training of language models Year: (2024-11)
Ref_id:b13 Title: URL Year: ()
Ref_id:b14 Title: Continual pre-training of large language models: How to (re)warm your model? Year: (2023)
Ref_id:b15 Title: Data mixture inference: What do bpe tokenizers reveal about their training data? Year: (2024)
Ref_id:b16 Title: Pile of law: Learning responsible data filtering from the law and a 256gb open-source legal dataset Year: (2022)
Ref_id:b17 Title: Scaling laws for transfer Year: (2021)
Ref_id:b18 Title: Scaling laws for transfer Year: (2021)
Ref_id:b19 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b20 Title: Unveiling the potential of small language models with scalable training strategies Year: (2024)
Ref_id:b21 Title: Robust Estimation of a Location Parameter Year: (1964)
Ref_id:b22 Title: -coder technical report Year: (2024)
Ref_id:b23 Title: Simple and scalable strategies to continually pre-train large language models Year: (2024)
Ref_id:b24 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b25 Title: A method for stochastic optimization Year: (2015)
Ref_id:b26 Title: Continual evaluation for lifelong learning: Identifying the stability gap Year: (2023)
Ref_id:b27 Title: Crafting papers on machine learning Year: (2000)
Ref_id:b28 Title: Stochastic gradient descent with warm restarts. International Conference on Learning Representations Year: (2016)
Ref_id:b29 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b30 Title: A multi-power law for loss curve prediction across learning rate schedules Year: (2025)
Ref_id:b31 Title: Updating quasi newton matrices with limited storage Year: (1980-07)
Ref_id:b32 Title: OpenAI. Gpt-4 technical report Year: (2023)
Ref_id:b33 Title: don't retrain: A recipe for continued pretraining of language models Year: (2024)
Ref_id:b34 Title: Openwebmath: An open dataset of high-quality mathematical web text Year: (2023)
Ref_id:b35 Title: The fineweb datasets: Decanting the web for the finest text data at scale Year: (2024)
Ref_id:b36 Title: D-CPT law: Domain-specific continual pre-training scaling law for large language models Year: (2024)
Ref_id:b37 Title: Continual learning of large language models: A comprehensive survey Year: (2024)
Ref_id:b38 Title: Practical bayesian optimization of machine learning algorithms Year: (2012)
Ref_id:b39 Title: SlimPajama: A 627B token cleaned and deduplicated version of RedPajama Year: (2023-06)
