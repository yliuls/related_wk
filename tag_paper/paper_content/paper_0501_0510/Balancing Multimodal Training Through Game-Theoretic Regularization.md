Title: Balancing Multimodal Training Through Game-Theoretic Regularization
Abstract: Multimodal learning holds promise for richer information extraction by capturing dependencies across data sources. Yet, current training methods often underperform due to modality competition, a phenomenon where modalities contend for training resources leaving some underoptimized. This raises a pivotal question: how can we address training imbalances, ensure adequate optimization across all modalities, and achieve consistent performance improvements as we transition from unimodal to multimodal data? This paper proposes the Multimodal Competition Regularizer (MCR), inspired by a mutual information (MI) decomposition designed to prevent the adverse effects of competition in multimodal training. Our key contributions are: 1) A game-theoretic framework that adaptively balances modality contributions by encouraging each to maximize its informative role in the final prediction 2) Refining lower and upper bounds for each MI term to enhance the extraction of both taskrelevant unique and shared information across modalities. 3) Proposing latent space permutations for conditional MI estimation, significantly improving computational efficiency. MCR outperforms all previously suggested training strategies and simple baseline, clearly demonstrating that training modalities jointly leads to important performance gains on both synthetic and large real-world datasets. We release our code and models at https://github.com/kkontras/MCR.

Section: Introduction
Exploiting multimodal data has made significant progress, with advances in generalizable representations and larger datasets enabling solutions to previously unattainable tasks [28,33,30,39,45,44,46,52,55,66]. However, studies indicate that jointly trained multimodal data is often utilized suboptimally, underperforming compared to ensembles of unimodal models, jointly trained modalities, or even the best single modality [56,64]. The expectation that adding a new modality should improve performance, assuming independent errors and above-chance predictive power [16], is frequently contradicted in practice.
Huang et al. [20] attribute this issue to modality competition, where one modality quickly minimizes training error, misdirecting and suppressing the learning of others. To counteract this effect, monitoring each modality's contribution during training and applying corrective measures is crucial. To this The shared task-relevant information (S) is defined as I(X 1 ; X 2 ) -I(X 1 ; X 2 | Y ). (Right) Accuracy on a synthetic dataset designed to induce multimodal competition. We vary the ratio of unique information from modality 1 (U 1 ) to shared information (S), while keeping the contribution of modality 2 (U 2 ) constant. As the imbalance increases (moving right on the x-axis), the performance of most methods drops. The standard Joint Training (Singleloss) approach shows a steep decline, highlighting its vulnerability to modality competition where one modality dominates and suppresses the other. In contrast, our method, MCR, demonstrates greater robustness by maintaining the highest accuracy and exhibiting the slowest performance degradation. See Section 4.1 for more details.
end, several balancing strategies have been proposed [5,6,9,10,21,27,29,42,43,56,61,64,57,59,69,19,58]. Some ignore a modality's contribution beyond its independent unimodal performance, while others address this by measuring output differences under input perturbation, but at the cost of increased sensitivity to these perturbations and significant computational overhead. Moreover, it is crucial to examine whether enhancing one modality's influence on the output does not come at the expense of others, as this could undermine overall performance.
Given these challenges, how can we efficiently regularize multimodal competition to ensure balanced and effective learning across modalities?
This paper introduces a loss function encouraging the exploration of task-relevant information across modalities, the MULTIMODAL COMPETITION REGULARIZER (MCR). The approach incorporates the following key contributions: 1. MI Bounds: We decompose joint mutual information into task-relevant shared and unique components, using refined lower and upper bounds to promote informative signals and suppress noise. 2. Game-Theoretic Modality Balancing: We frame modality interaction through a game-theoretic framework, allowing each modality to adjust its contribution throughout training. 3. Efficient CMI Estimation: We introduce latent-space perturbations for low-cost conditional MI estimation, avoiding repeated full-model passes. We extensively evaluate MCR on synthetic datasets and several established real-world multimodal benchmarks, including action recognition on AVE [51] and UCF [47], emotion recognition on CREMA-D [4], human sentiment on CMU-MOSI [65], human emotions on CMU-MOSEI [67], and egocentric action recognition on Something-Something [14]. Our results demonstrate that MCR outperforms all previous methods and simple baselines across various datasets and models, improving multimodal supervised training. families are defined as, unimodal models for m = [1, .., M ] modalities: F um : f um (X m ; θ m , θ cm ) = f cm (f m (X m ; θ m ) ; θ cm ) , (1) and for multimodal models: F : f (X; θ) = f c ([f 1 (X 1 ; θ 1 ) , .., f M (X M ; θ M )] ; θ c ) .
(2) For simplicity, we continue our analysis with M = 2, focusing on models with two modalities.
this section cite: ['b27', 'b32', 'b29', 'b38', 'b44', 'b43', 'b45', 'b51', 'b54', 'b65', 'b55', 'b63', 'b15', 'b19', 'b4', 'b5', 'b8', 'b9', 'b20', 'b26', 'b28', 'b41', 'b42', 'b55', 'b60', 'b63', 'b56', 'b58', 'b68', 'b18', 'b57', 'b50', 'b46', 'b3', 'b64', 'b66', 'b13']

Section: The limitation of supervised multimodal training
In supervised learning, the goal is to learn representations Z 1 = f 1 (X 1 ; θ 1 ) and Z 2 = f 2 (X 2 ; θ 2 ) when fused via f c ([Z 1 , Z 2 ]), yield accurate predictions. This is achieved by minimizing the task loss or, equivalently, by maximizing the MI between the fused representation and the target: arg max Z1:=f1(X1;θ1), Z2:=f2(X2;θ2)
I(f c ([Z 1 , Z 2 ]); Y t ).(3)
During training, models often over-rely on the stronger or more accessible modality, limiting the contribution of others. This leads to mutual information being dominated by one modality, e.g., I(f c ([Z 1 , Z 2 ]); Y ) ≈ I(Z 1 ; Y ) with I(Z 2 ; Y | Z 1 ) ≈ 0, indicating that Z 2 adds little once Z 1 is learned. See Appendix A.1 for an illustrative experiment and Appendix A.2 for a formal definition of the resulting generalization gap.
This kind of imbalance is well-known in single-modality learning, where dominant features can overshadow others, harming generalization. Regularization techniques like l 1 / l 2 penalties and dropout promote balanced feature use [40,48], but their adaptation to multimodal settings is nontrivial. For example, applying modality-specific dropout [62] offers limited benefits [42]. The core challenge remains: how to effectively regulate interaction and competition between modalities.
this section cite: ['b39', 'b47', 'b61', 'b41']

Section: Related Work
Prior research has explored various strategies for multimodal learning, ranging from simple unimodal and ensemble-based approaches to more sophisticated methods for balancing modality contributions. Unimodal training optimizes each modality separately, while ensemble methods combine unimodal predictions without additional training. Joint training optimizes all modalities under a single-loss objective but does not explicitly ensure sufficient training for each modality. To address this, Multi-Loss [54] introduces additional unimodal task losses, and MMCosine [63] equalizes modality influence by standardizing features and weights. Pre-trained unimodal encoders are often used, either with frozen weights (Uni-Pre Frozen) or fine-tuned jointly (Uni-Pre Finetuned). Other adaptive strategies include MSLR [64], which adjusts learning rates based on unimodal validation performance, OGM [42], which modulates gradients by comparing unimodal performance across modalities, and MLB [27], which combines unimodal task losses and modulates gradients from both unimodal and multimodal objectives, MMPareto [57] that mitigates gradient conflicts between modalities by equalizing the contribution of unimodal and multimodal gradient and D&R [59] suggest a new strategy where modalities that overfit get their part of the network partially reweighted with the initial weights of the training.
Most of these methods assume distributional independence and measure modality contributions through unimodal performance, which can be a limited indicator, missing cases where modality correlation is crucial. Other approaches estimate influence based on prediction differences after perturbations [29,21,10]. AGM [29] uses zero-masking Shapley values directly optimizing them as unimodal predictors, Wei et al. [58] use a permutation-based Shapley values and resampling of the training set to affect the training, while other methods address similar problems by introducing perturbations such as Gaussian noise [10] or task-specific augmentations [21,32]. However, perturbation-based approaches increase the network's sensitivity to the chosen perturbations and hinder scalability due to their higher computational demands.
A line of work keeps unimodal training as the primary strategy. MLA [69] uses a shared task head and dynamic weighted summation during validation, while ReconBoost [19] alternates unimodal updates with agreement and diversity regularization before finetuning the ensemble. However, these approaches avoid multimodal training in the earlier steps to mitigate conflicts, yet overlook the potential benefits of direct multimodal interactions in those steps.
1: Video Encoder 2: Audio Encoder pair s 𝑋 1 Ζ 2 Ζ 1 ෨ 𝑌 1 ෨ 𝑍 1 ෨ 𝑍 2 𝑋 2 [ ෨ 𝑍 1 , 𝑍 2 ] 𝑍 = [𝑍 1 , 𝑍 2 ] [𝑍 1 , ෨ 𝑍 2 ] 𝑌 ෨ 𝑌 2 L 𝐶𝑜𝑛 L 𝑇𝑎𝑠𝑘 L 𝑀𝐼𝑃𝐷1 = -𝐽𝑆𝐷(𝑌, ෨ 𝑌 1 ) L 𝑀𝐼𝑃𝐷2 = -𝐽𝑆𝐷(𝑌, ෨ 𝑌 2 ) Reconstruction Network L 𝐶𝐸𝐵 (𝑍, መ 𝑍) 𝑌 መ 𝑍 L 𝑀𝐶𝑅 = L 𝑀𝐼𝑃𝐷 + L 𝐶𝑜𝑛 + L 𝐶𝐸𝐵 Multimodal Competition Regularizer Fusion Network Key Contribution: Augmentations in the latent space L 𝑀𝐼𝑃𝐷 = L 𝑀𝐼𝑃𝐷 1 + L 𝑀𝐼𝑃𝐷 2
this section cite: ['b53', 'b62', 'b63', 'b41', 'b26', 'b56', 'b58', 'b28', 'b20', 'b9', 'b28', 'b57', 'b9', 'b20', 'b31', 'b68', 'b18']

Section: Multimodal Competition Regularizer
Multimodal competition arises when a model trained on multiple modalities prioritizes one, leading to over-reliance and reducing the contribution of others. This imbalance limits the model's ability to fully utilize all available information. In this section, we introduce L MCR , a set of loss components designed to address multimodal competition. Each component of the loss is motivated by the following MI decomposition:
I(X 1 ; X 2 ; Y ) = I(X 1 ; Y | X 2 ) + I(X 2 ; Y | X 1 )
Task-Relevant Unique Information of each modality∼LMIPD
+ I(X 1 ; X 2 ) Shared Information LCon -I(X 1 ; X 2 | Y ) Task-Irrelevant Shared Information∼LCEB . (4
)
This decomposition is illustrated by the Venn diagram in Figure 1. The CMIs I(X 1 ; Y | X 2 ) and I(X 2 ; Y | X 1 ) capture modality-specific information for predicting the target. Maximizing them with the Mutual Information Perturbed Difference (MIPD) loss, L MIPD , which assesses each modality's contribution via output variations under input perturbations (elaborated in Sec. 3.3) and encourages the extraction of modality-specific, task-relevant features. The third term, I(X 1 ; X 2 ), quantifies shared information between modalities. Maximizing it with a contrastive loss, L Con , aligns representations and leverages their shared information effectively [22,41,46]. The final term, I(X 1 ; X 2 | Y ), represents task-irrelevant shared information. Penalizing it with the conditional entropy bottleneck (CEB) [8] and the corresponding loss L CEB to filter out irrelevant information, focusing the model on features relevant to the downstream task. Each term has a corresponding loss, as illustrated in Figure 2, forming the regularizer with three key losses:
L MCR = L MIPD + L Con + L CEB(5)
this section cite: ['b21', 'b40', 'b45', 'b7']

Section: Approximating MI Terms
I(X 1 ; Y | X 2 ) :
To approximate each CMI and capture the unique contribution of each modality, the MIPD serves as a surrogate function, measuring how input perturbations affect the model's output. By comparing predictions with and without these perturbations, MIPD estimates how much information each modality provides. If a modality is crucial, altering its input should significantly change the output, revealing its importance.
this section cite: []

Section: Estimating the CMI directly through
I(X 1 ; Y | X 2 ) = H(Y | X 2 ) -H(Y | X 1 , X 2 )
is typically intractable. Instead we use the MIPD as a lower bound, defined as:
MIPD(X 1 ; Y | X 2 ) = I(X 1 ; Y | X 2 ) -I( X1 ; Y | X 2 ) ≤ I(X 1 ; Y | X 2 ),(6)
𝑌, ෨ 𝑌 2 L 𝑀𝐼𝑃𝐷2 = -𝐽𝑆𝐷(𝑌, ෨ 𝑌 2 ) 𝑍 1 𝑍 2 * 𝑘 𝑘 = ቐ 1 0 -1 1: Video Encoder 2: Audio Encoder Fusion Network ෨ 𝑍 2 permute Gradient Game Strategies Collaborative (Video Encoder: ↑ Importance 2 ) Independent Greedy (Video Encoder: ↓ Importance 2 ) ↓ L 𝑀𝐼𝑃𝐷2 = ↑ Importance 2 𝑋 1 𝑋 2 Key Contribution: Video Encoder benefits itself by reducing ∇ 𝜃1 L 𝑀𝐼𝑃𝐷2 i.e. greedy strategy. ∇ 𝜃2 L 𝑀𝐼𝑃𝐷2 𝑘 • ∇ 𝜃1 L 𝑀𝐼𝑃𝐷2 where the perturbed version of modality X 1 is denoted as X1 . Interpreting MI via entropy, each CMI can be expressed as the difference of the log probabilities with and without the perturbations:
MIPD(X 1 ; Y | X 2 ) = H(Y | X 2 , X1 ) -H(Y | X 2 , X 1 ) = E y∼p(y) x1,x2∼p(x1,x2,y) -E x1∼p(x1) [log p(y | x 2 , x1 )] + log p(y | x 2 , x 1 ) . (7
)
Instead of this log-likelihood ratio, we use the symmetrically bounded Jensen-Shannon divergence (JSD) [34] to prevent training instabilities, leading to the following:
L MIPD1 = -MIPD(X 1 ; Y | X 2 ) = -E y∼p(y) x1,x2∼p(x1,x2,y) x1∼p(x1) [JSD(p(y | x 2 , x 1 ), p(y | x 2 , x1 ))] .(8)
Similarly, L MIPD2 can be computed symmetrically. I(X 1 ; X 2 ) : The next MI term measures how much information the two modalities share, capturing the common patterns between the modalities and aligning the representations of these shared aspects.
We exploit the available label information employing the supervised contrastive loss L Con [25]:
L Con = E x1,y∼p(x1,y) x + 2 ∼p(x2|y) x - 2 ∼p(x2|¬y
) 2 log ψ(x 1 , x + 2 ) k ψ(x 1 , x - 2 k ) ,(9)
where ψ is the critic function, which, in our case, is the exponential dot product. Minimizing the L Con , maximizes a lower bound on both the MI between the two modalities and the CMI terms:
I(X 2 ; Y |X 1 ) + I(X 1 ; Y |X 2 ) + 2I(X 2 ; X 1 ) ≥ log N -L Opt Con .(10)
As N increases, the bound becomes tighter, while the bound is not affected by the number of positive samples (same class datapoints). More details are provided in Appendix B.
I(X 1 ; X 2 | Y) :
The final term captures irrelevant shared information between modalities, and minimizing an upper bound on this ensures the model retains only task-relevant content. For this purpose, we exploit the idea of Conditional Entropy Bottleneck (CEB) L CEB [7], targeting superfluous information in multimodal representations via a reconstruction loss. A small reconstruction head, h : Y ; θ h → Z = (Z 1 , Z 2 ), predicts back the latent space, effectively filtering out irrelevant content:
L CEB = E x1,x2,y∼p(x1,x2,y) ∥[f 1 (x 1 ), f 2 (x 2 )] -h(y; θ h )∥ 2 (11
)
The exact derivation of this loss term can be found in Appendix A.6. Penalizing irrelevant information has been shown to enhance calibration and robustness [8], but it must be carefully evaluated, as it can introduce constraints that may hinder overall performance.
this section cite: ['b33', 'b24', 'b6', 'b7']

Section: The Game of Multimodal Fusion
We adopt a game-theoretic approach to balance the terms of the proposed L MIPD . The key idea is that increasing one modality's importance (e.g., via MIPD 1 ) can inherently reduce the other's (e.g., MIPD 2 ). Thus, an underutilized encoder i (with parameters θ i ) can boost its relevance both by minimizing L MIPDi and by maximizing L MIPD¬i . This twofold strategy helps prevent suppression of weaker modalities. We frame L MIPD as a game where each encoder (player) selects a strategy, minimize, maximize, or ignore. Figure 3 illustrates how the video modality, via a hyperparameter k, can choose to assist, ignore, or diminish the audio modality. Each encoder applies this logic selectively as formalized below:
∇ θ1 L MIPD = λ M (∇ θ1 L MIPD1 + k ∇ θ1 L MIPD2 ) ,(12)
∇ θ2 L MIPD = λ M (∇ θ2 L MIPD2 + k ∇ θ2 L MIPD1 ) . 3 (13
)
where λ M is a Lagrange multiplier, and k ∈ {-1, 0, 1} sets the modality's strategy:
• Collaborative (k = 1): All modalities work together to increase each other's contributions. The L MIPD terms are applied across all parameters, resulting in min θ L MIPD .
• Independent (k = 0): Each modality focuses on maximizing its own contribution by optimizing solely its respective L MIPD term, leading to min θi L MIPDi .
• Greedy (k = -1): Each modality seeks to maximize its own contribution by: 1) minimizing its own L MIPD term, and 2) maximizing the L MIPD terms of other modalities, resulting in a min-max game,
min θi max θ¬i L MIPDi 4 .
Following the results in Appendix A.8, we adopt the greedy strategy as default, as it showed the most consistent performance in our setting.
this section cite: []

Section: Perturbations
To assess the importance of modality X 1 , we define L MIPD1 , which captures changes in the model's output when X 1 is perturbed (i.e., { X1 , X 2 } vs. {X 1 , X 2 }). Instead of traditional input-space perturbations, which can be computationally expensive and task-dependent, we apply a within-batch permutation σ e ∼ Uniform(P) in the latent space, yielding X1 = σ e (X 1 ). This approach avoids extra forward passes and reduces computational and memory overhead. Further analysis of this technique and comparisons with prior methods are provided in Appendix A.10 and A.14.
The complete algorithm is presented in Algorithm 1, with an extension of L MCR to M modalities described in Appendix A.7. In Appendix A.9, we analyze various combinations of loss components, revealing that penalizing task-irrelevant information benefits models with extensive SSL pretraining but proves detrimental for those without it.
this section cite: []

Section: Experiments

this section cite: []

Section: Synthetic Dataset
We create a scenario where mutual information varies, showcasing modality competition. While various factors can contribute to such a phenomenon, we focus on modality informativeness imbalance to motivate our approach.
this section cite: []

Section: Data:
We generate task-irrelevant information for each modality by sampling N 1 , N 2 ∼ N (0, I) and the 5-class label Y t from a uniform distribution Y t ∼ Uniform(5). Each modality is converted into a high-dimensional vector using fixed transformations, similar to Liang et al. [32]. We relate both modalities to the label through a linear relationship: X 1 = N 1 + Y t and X 2 = N 2 + Y t . Data points are distributed in such a way that either both modalities contain label information (Shared Information) or only one of the modalities (Unique Information). In cases where only one modality contains label information, the other modality is defined as X 1 = N 1 and X 2 = N 2 respectively. We Algorithm 1 Multimodal Training with MCR Input: Training dataset D with modalities X 1 , X 2 , . . . , X M , labels Y t , multimodal model f ∈ F, initialized unimodal encoders θ i , reconstruction model h, λ uni , λ M Lagrangian coefficients: 1: for each batch (X 1 , .., X M , Y t ) of each epoch do 2:
Compute L task (f (X 1 , .., X M ), Y t ) and
L uni task = λ uni M m=1 L task (f u m (X m ), Y t ) 3: Extract (Z 1 , .., Z M ) from f (X 1 , .., X M ) 4:
Assess the L Con with Eq. 9 and L CEB with Eq. 11 5:
Sample σ e permutations and compute permuted pairs on the latent space Z Results: Figure 1 shows the performance on synthetic data, comparing our method (MCR) with several baselines. As the shared information S among the modalities decreases, and the unique information of one modality U 1 increases while the U 2 remains constant, we observe a performance drop for all methods. MCR maintains the highest accuracy across all combinations, demonstrating the slowest decline and highlighting its robustness to such imbalance.
this section cite: ['b31', 'b4']

Section: Real-World Datasets
Datasets: We explore several real-world datasets, primarily with video, optical flow, audio, and text modalities, that either exhibit significant imbalance among modalities or serve as standard multimodal benchmarks. Detailed descriptions are provided in Appendix A.3, with brief summaries below: 1. CREMA-D [4]: An emotion recognition dataset with 91 actors expressing 6 distinct emotions. 2. AVE [51]: A collection of videos with temporally aligned audio-visual events across 28 categories. 3. UCF [47]: An action recognition dataset of real-life YouTube videos. 4. CMU-MOSEI [67]: Multimodal sentiment analysis dataset with 23k monologue clips. 5. CMU-MOSI [65]: Multimodal sentiment analysis dataset with over 2k YouTube video clips. 6. Something-Something (V2) [14]: 220k clips of individuals performing 174 hand actions.
this section cite: ['b3', 'b50', 'b46', 'b66', 'b64', 'b13']

Section: Models:
We employ a variety of models and backbone encoders to examine the behavior of both smaller-scale models trained from scratch and larger, more complex models pretrained with selfsupervised learning (SSL). This combination demonstrates that our method is effective in both limited data scenarios without pretraining and in cases with ample data where the goal is fine-tuning. We utilize ResNet-18 and small-scale Transformers (from thousand to 20M parameters) alongside stateof-the-art models such as Swin-TF [35] and Conformer [13], incorporating backbone encoders like Wav2Vec2, HuBERT, and ViViT, resulting in model sizes approaching 200M parameters. Detailed model configurations for each dataset are provided in Appendix A.4 and experimental details in Appendix A.5. These choices aim to bridge the gap between theoretical work and practical application.
Results: Table 1 reports accuracy comparisons across baseline methods on our evaluation datasets. We highlight two key observations: a) Most prior methods, including recent multimodal approaches, fail to outperform simpler alternatives such as Ensembles or unimodal encoders (either frozen or finetuned). While some of these methods partially address modality competition, they often fall short of effectively leveraging multimodal data. b) MCR is the only method that consistently surpasses all baselines across datasets, model architectures, number of modalities, and task types (classification and regression), with an exception on AVE with Conformer model. This consistent advantage highlights MCR's ability to balance Table 1: Performance comparison of MCR against prior multimodal training methods across six datasets. MCR consistently achieves top results across various modality combinations (V-A, V-T, V-A-T, V-OF) and model architectures. MOSI and MOSEI use three modalities and are trained as regression tasks (converted to binary accuracy); the rest are classification tasks. All baselines were rerun under our evaluation protocol to ensure fair comparison and address data leakage issues identified in previous evaluation setups.
CREMA-D AVE UCF MOSI MOSEI Sth-Sth ResNet Conformer ResNet Conformer ResNet Transformer Swin-TF Method V-A V-A V-A V-A V-A V-T V-A-T V-T V-A-T V-OF Unimodals V
: 55.4±3.0 A: 60.6±2.3 V: 69.4±2.0 A: 76.0±2.6 V: 45.7±1.6 A: 62.6±0.9 V: 75.5±1.2 A: 76.5±2.4 V: 38.5±0.9 A: 30.3±1.5 V: 54.1±3.7 A: 53.7±0.6 T: 72.1±3.3 V: 64.8±0.2 A: 64.4±0.2 T: 78.9±1.7 V: 61.4±0.2 OF: 50.8±0.1 Ensemble 71.7±2.2 84.6±1.0 70.5±0.2 88.4±2.2 52.8±0.5 70.5±2.1 67.2±1.5 78.4±0.7 77.2±0.6 64.6±0.2 Joint Training 62.6±5.8 74.6±2.2 66.7±1.5 82.2±0.8 47.7±1.5 73.0±1.3 73.6±1.3 80.5±0.2 80.8±0.3 57.5±0.1 Multi-Loss 69.2±1.8 82.6±0.9 70.1±0.9 86.3±1.1 51.1±1.8 72.1±0.4 73.6±2.9 80.0±0.7 80.2±0.5 61.5±0.1 Uni-Pre Frozen 72.4±1.8 85.0±1.8 72.2±0.3 87.2±2.4 53.0±0.9 73.3±1.8 72.7±1.6 79.9±0.5 79.8±0.3 64.0±0.2 Uni-Pre Finetuned 73.3±1.8 82.4±2.0 72.5±1.3 86.5±0.8 53.5±1.3 73.1±2.3 73.7±0.7 80.3±0.4 80.3±0.2 62.1±0.2 MSLR [64] 56.5±2.4 77.1±2.4 67.3±2.2 81.0±1.4 50.9±3.9 × × × × -MMCosine [63] 59.3±1.5 74.0±0.3 65.0±1.4 83.8±0.8 47.3±4.1 × × × × -OGM [42] 65.6±3.8 82.4±1.0 67.3±0.6 79.7±1.4 51.8±1.9 73.9±1.1 ⊗ 79.7±0.6 ⊗ 57.8±0.5 AGM [29] 69.3±1.4 78.5±1.6 68.4±1.1 85.3±0.5 51.0±1.6 74.0±1.3 73.9±1.9 79.3±0.4 80.2±0.3 56.6±0.4 MLB [27] 71.9±2.2 85.2±0.9 71.6±0.2 86.7±0.3 52.2±1.7 72.4±1.7 74.2±1.7 80.1±0.5 80.5±0.4 61.6±0.2 ReconBoost [19] 69.0±2.4 84.8±1.8 68.4±1.7 86.1±0.7 50.2±4.0 × × × × 56.1±0.3 MMPareto [57] 69.0±2.5 83.8±0.8 73.0±1.3 87.4±1.3 51.4±2.2 73.4±1.0 73.7±0.6 79.3±0.6 79.5±0.8 59.2±0.5 D&R [59] 70.6±1.3 85.0±0.4 72.3±1.5 91.0±0.7 49.3±1.0 × × × × 61.7±0.2 MCR 76.1±1.6 85.7±0.2 73.4±0.0 88.8±1.0 55.2±1.8 75.2±1.7 76.5±1.4 80.8±0.4 81.1±0.4 65.0±0.1
× method not applicable to regression tasks; ⊗ method not applicable to trimodal inputs; result not reported.
modality contributions during training under diverse settings, positioning it as a strong and generalizable approach for multimodal learning.
this section cite: ['b34', 'b12']

Section: Analysis of Multimodal Error
Methodology: To understand how MCR improves over existing methods, we perform a post-hoc error analysis by categorizing each sample based on the correctness of the unimodal predictions. Specifically, we consider four groups: (1) both unimodal models are correct, (2) only the first is correct, (3) only the second is correct, and (4) both are incorrect. This breakdown allows us to compare how different multimodal methods behave across these categories and identify whether gains arise from selective reliance on unimodal cues or from synergistic integration.
Results: The error analysis of Figure 4 reveals key strengths and limitations of the proposed method. MCR consistently outperforms other methods in routing information favouring both modalities in the cases that only one of the modalities is correct maintaining competitive performance on both of them and in the case that all modalities correctly predict the label. This demonstrates MCR's ability to effectively route information to the appropriate modality, in line with its design choice to model and control training via this modality-independent, task-relevant information through the mutual information terms.
However, MCR does not exhibit significant gains in capturing synergetic information in the datapoints that all unimodal models fail, underperforming relative to AGM and MLB. This suggests MCR excels in routing decisions but may be less effective at leveraging synergies across modalities when all individual models falter. This trend holds across other datasets (see Appendix A.11), with the exception of MOSI, where MCR improves synergy. Our initial assumption that concurrent modality training would foster synergy thus did not hold in practice.
this section cite: []

Section: Discussion
This paper examines the challenge of modality competition in multimodal learning, where certain modalities dominate the training process, resulting in suboptimal performance. We introduce the Multimodal Competition Regularizer (MCR), a novel approach inspired by information theory, which frames multimodal learning as a game where each modality competes to maximize its contribution to the final output. MCR efficiently computes lower and upper bounds to optimize both unique and shared task-relevant information for each modality. Our extensive experiments show that MCR consistently outperforms existing methods and simple baselines on both synthetic and real-world datasets, providing a more balanced and effective multimodal learning framework. MCR paves the way for fulfilling the long-standing promise of multimodal fusion methods to achieve performance that surpasses the combined results of unimodal training.
We explored different game strategies and observed that directly encouraging competition between modalities in the overall objective function positively impacts performance, as detailed in Appendix A.8. Future work could investigate more refined strategies to enable individualized and adaptive decisions for each modality to unlock greater performance gains.
Lastly, we conduct a post-hoc error analysis found both in Section 4.3 and Appendix A.11, examining overlaps between the errors of multimodal models and their unimodal counterparts. The results show that MCR excels at routing decisions to the correct unimodal information but does not promote synergetic behavior accordingly, compared to previous methods. Our initial assumption that the simultaneous progress of unimodal encoders during training would naturally enhance synergy was not supported in practice, highlighting the need for future work to promote this behavior explicitly. Finally, this analysis highlights the potential for performance improvements through enhanced multimodal training, motivating further exploration in this area.
this section cite: []

Section: References
Ref_id:b0 Title: Sorting out lipschitz function approximation Year: (2019)
Ref_id:b1 Title: Vivit: A video vision transformer Year: (2021)
Ref_id:b2 Title: wav2vec 2.0: A framework for self-supervised learning of speech representations Year: (2020)
Ref_id:b3 Title: Crema-d: Crowd-sourced emotional multimodal actors dataset Year: (2014)
Ref_id:b4 Title: On uni-modal feature learning in supervised multi-modal learning Year: (2023)
Ref_id:b5 Title: Pmr: Prototypical modal rebalance for multimodal learning Year: (2023)
Ref_id:b6 Title: The conditional entropy bottleneck Year: (2020)
Ref_id:b7 Title: Ceb improves model robustness Year: (2020)
Ref_id:b8 Title: Modality-specific learning rate control for multimodal classification Year: (2019-11-26)
Ref_id:b9 Title: Removing bias in multi-modal classifiers: Regularization by maximizing functional entropies Year: (2020)
Ref_id:b10 Title: Audio set: An ontology and human-labeled dataset for audio events Year: (2017)
Ref_id:b11 Title: Versatile audiovisual learning for handling single and multi modalities in emotion regression and classification tasks Year: (2023-05)
Ref_id:b12 Title: Versatile audio-visual learning for handling single and multi modalities in emotion regression and classification tasks Year: (2023)
Ref_id:b13 Title: The" something something" video database for learning and evaluating visual common sense Year: (2017)
Ref_id:b14 Title: Conformer: Convolution-augmented transformer for speech recognition Year: (2020)
Ref_id:b15 Title: Neural network ensembles Year: (1990)
Ref_id:b16 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b17 Title: Self-supervised speech representation learning by masked prediction of hidden units Year: (2021)
Ref_id:b18 Title: Reconboost: boosting can achieve modality reconcilement Year: (2024)
Ref_id:b19 Title: Modality competition: What makes joint training of multi-modal network fail in deep learning?(provably) Year: (2022)
Ref_id:b20 Title: Increasing visual awareness in multimodal neural machine translation from an information theoretic perspective Year: (2022)
Ref_id:b21 Title: Scaling up visual and vision-language representation learning with noisy text supervision Year: (2021)
Ref_id:b22 Title: Fairness-regulated dense subgraph discovery Year: (2024)
Ref_id:b23 Title: The kinetics human action video dataset Year: (2017)
Ref_id:b24 Title: Supervised contrastive learning Year: (2020)
Ref_id:b25 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b26 Title: Improving multimodal learning with multi-loss gradient modulation Year: (2024)
Ref_id:b27 Title: Core-sleep: A multimodal fusion framework for time series robust to imperfect modalities Year: (2024)
Ref_id:b28 Title: Boosting multi-modal model performance with adaptive gradient modulation Year: (2023)
Ref_id:b29 Title: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation Year: (2022)
Ref_id:b30 Title: Multibench: Multiscale benchmarks for multimodal representation learning Year: (2021)
Ref_id:b31 Title: Factorized contrastive learning: Going beyond multi-view redundancy Year: (2024)
Ref_id:b32 Title: Foundations & trends in multimodal machine learning: Principles, challenges, and open questions Year: (2024)
Ref_id:b33 Title: Divergence measures based on the shannon entropy Year: (1991)
Ref_id:b34 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b35 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b36 Title: A unified approach to interpreting model predictions Year: (2017)
Ref_id:b37 Title: Probabilistic machine learning: an introduction Year: (2022)
Ref_id:b38 Title: Attention bottlenecks for multimodal fusion Year: (2021)
Ref_id:b39 Title: Feature selection, l 1 vs. l 2 regularization, and rotational invariance Year: (2004)
Ref_id:b40 Title: Representation learning with contrastive predictive coding Year: (2018)
Ref_id:b41 Title: Balanced multimodal learning via on-the-fly gradient modulation Year: (2022)
Ref_id:b42 Title: Xsleepnet: Multi-view sequential model for automatic sleep staging Year: (2021)
Ref_id:b43 Title: Revisiting spatio-temporal layouts for compositional action recognition Year: (2021)
Ref_id:b44 Title: Multimodal distillation for egocentric action recognition Year: (2023)
Ref_id:b45 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b46 Title: Ucf101: A dataset of 101 human actions classes from videos in the wild Year: (2012)
Ref_id:b47 Title: Dropout: a simple way to prevent neural networks from overfitting Year: (2014)
Ref_id:b48 Title: Intriguing properties of neural networks Year: (2013)
Ref_id:b49 Title: Rethinking model scaling for convolutional neural networks Year: (2019)
Ref_id:b50 Title: Audio-visual event localization in unconstrained videos Year: (2018)
Ref_id:b51 Title: Multimodal transformer for unaligned multimodal language sequences Year: (2019)
Ref_id:b52 Title: Attention is all you need Year: (2017)
Ref_id:b53 Title: Centralnet: a multilayer approach for multimodal fusion Year: (2018)
Ref_id:b54 Title: Adversarial cross-modal retrieval Year: (2017)
Ref_id:b55 Title: What makes training multi-modal classification networks hard? Year: (2020)
Ref_id:b56 Title: Mmpareto: boosting multimodal learning with innocent unimodal assistance Year: (2024)
Ref_id:b57 Title: Enhancing multimodal cooperation via samplelevel modality valuation Year: (2024)
Ref_id:b58 Title: Diagnosing and re-learning for balanced multimodal learning Year: (2024)
Ref_id:b59 Title: Transformers: State-of-the-art natural language processing Year: (2020-10)
Ref_id:b60 Title: Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks Year: (2022)
Ref_id:b61 Title: Audiovisual slowfast networks for video recognition Year: (2020)
Ref_id:b62 Title: Mmcosine: Multi-modal cosine loss towards balanced audio-visual fine-grained learning Year: (2023)
Ref_id:b63 Title: Modality-specific learning rates for effective multimodal additive latefusion Year: (2022)
Ref_id:b64 Title: Mosi: multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos Year: (2016)
Ref_id:b65 Title: Tensor fusion network for multimodal sentiment analysis Year: (2017)
Ref_id:b66 Title: Multimodal language analysis in the wild: Cmu-mosei dataset and interpretable dynamic fusion graph Year: (2018)
Ref_id:b67 Title: Joint face detection and alignment using multitask cascaded convolutional networks Year: (2016)
Ref_id:b68 Title: Multimodal representation learning by alternating unimodal adaptation Year: (2024)
