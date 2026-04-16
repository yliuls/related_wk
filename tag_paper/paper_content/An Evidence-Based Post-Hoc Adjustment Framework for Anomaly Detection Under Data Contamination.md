Title: An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination
Abstract: Unsupervised anomaly detection (AD) methods typically assume clean training data, yet real-world datasets often contain undetected or mislabeled anomalies, leading to significant performance degradation. Existing solutions require access to the training pipelines, data or prior knowledge of the proportions of anomalies in the data, limiting their real-world applicability. To address this challenge, we propose EPHAD, a simple yet effective test-time adaptation framework that updates the outputs of AD models trained on contaminated datasets using evidence gathered at test time. Our approach integrates the prior knowledge captured by the AD model trained on contaminated datasets with evidence derived from multimodal foundation models like Contrastive Language-Image Pre-training (CLIP), classical AD methods like the Local Outlier Factor or domain-specific knowledge. We illustrate the intuition behind EPHAD using a synthetic toy example and validate its effectiveness through comprehensive experiments across eight visual AD datasets, twenty-six tabular AD datasets, and a real-world industrial AD dataset. Additionally, we conduct an ablation study to analyse hyperparameter influence and robustness to varying contamination levels, demonstrating the versatility and robustness of EPHAD across diverse AD models and evidence pairs. To ensure reproducibility, our code is publicly available 2 .

Section: Introduction
Anomaly detection (AD) is the basis of many critical applications, including cybersecurity (Xiao et al., 2024;Li et al., 2023a), healthcare (Bijlani et al., 2024;Huang et al., 2024), and industrial maintenance (Schwarz et al., 2025;Patra et al., 2024). By enabling the identification of abnormalities, potential threats, or critical system failures, AD contributes to the robustness and safety of real-world systems. Despite its significance, AD remains a challenging task due to the inherent difficulty in characterising anomalous behaviours and the lack of prior knowledge about anomalous samples (Ruff et al., 2021). Consequently, AD is commonly approached as an unsupervised representation learning problem without access to labelled anomalies (Batzner et al., 2024;You et al., 2022).
A standard approach in unsupervised AD involves training a model to learn a "compact" representation of the normal samples from a training dataset under the assumption that the training data is "clean", i.e. contains only normal samples (Ruff et al., 2021). Then, anomalies are identified as deviations from this learned normality. One-class (OC) classification methods (Ruff et al., 2018;Tax and Duin, 2004) learn a decision boundary that encompasses all the normal samples. In contrast, density-based methods (Gudovskiy et al., 2022;Yu et al., 2021) learn the probability distribution of normal samples. Furthermore, memory bank-based approaches (Roth et al., 2022) store the features corresponding to normal samples in a memory bank. However, real-world datasets are often contaminated with undetected anomalies (Das et al., 2025;Hien et al., 2024;Qiu et al., 2022). For example, a dataset collected for industrial maintenance may already include unnoticed defects. This leads to biased AD models that struggle to reliably distinguish between normal and anomalous instances.
We consider the more realistic setting where the training data may be contaminated with anomalies. Existing approaches to handle contamination in the unsupervised setting primarily follow two strategies. The first employs an auxiliary OC classifier to filter out suspected anomalies (Yoon et al., 2022;Jiang et al., 2022), while the second modifies the training pipeline to enhance robustness against contamination (Qiu et al., 2022;Eduardo et al., 2020). Although effective, these methods rely on prior knowledge of the proportion of anomalies in the training data, i.e. the contamination ratio, which is typically unknown. Also, such methods are often computationally expensive. In the semi-supervised setting, methods leverage additional labelled datasets containing normal and anomalous samples (Hien et al., 2024;Ruff et al., 2020). However, their effectiveness diminishes when the anomalous instances encountered during training do not replicate real-world anomalies (Perini et al., 2025).
In this work, we aim to mitigate the possible adverse effects of data contamination on the performance of unsupervised AD models (Bouman et al., 2024). Specifically, we address the challenging setting in which training pipelines, data, or prior knowledge of the proportions of anomalies cannot be accessed. This scenario reflects the growing trend of deploying proprietary AD models in real-world applications, where access to internal model components is often restricted. Even when fine-tuning is permitted, it is not only computationally intensive but also unreliable due to the absence of guaranteed clean training data, as anomalies are inherently unknown a priori. This setup aligns with preparationagnostic test-time adaptation (TTA) methods (Karmanov et al., 2024;Zhang et al., 2023;Xiao and Snoek, 2024) , which remain largely unexplored in the context of AD. To address this gap, we introduce the Evidence-based Post-Hoc Adjustment Framework for Anomaly Detection (EPHAD), a simple yet effective method that adjusts the outputs of a pretrained AD model post-hoc, using evidence collected at test time.
Notably, we establish conceptual links between EPHAD and recent advances in test-time alignment for generative models (Mudgal et al., 2024;Li et al., 2024;Korbak et al., 2022), underscoring its broader significance. EPHAD is flexible and can incorporate various forms of evidence, including foundation models like Contrastive Language-Image Pre-training (CLIP) (Zhou et al., 2024;Jeong et al., 2023), classical AD methods such as Local Outlier Factor (LOF) (Breunig et al., 2000), and domain-specific knowledge. Our core contributions are summarised below:
• We introduce EPHAD, a simple yet effective TTA framework for unsupervised AD models trained on contaminated datasets. Unlike existing approaches, it requires no access to training pipelines, data or prior knowledge of the proportions of anomalies in the data, making it highly practical for real-world deployments.
• EPHAD performs TTA by combining the prior knowledge captured by the AD model trained on the contaminated dataset and an evidence gathered at test-time. This principled formulation allows for conceptual connections to recent test-time alignment techniques in generative modelling.
• We illustrate the intuition behind EPHAD using a carefully designed toy example. Furthermore, extensive experiments across eight visual AD, twenty-six tabular AD datasets, and a real-world industrial AD dataset demonstrate the effectiveness of EPHAD across diverse unsupervised AD models, evidence pairs.
2 Related work Unsupervised AD. Over the years, numerous approaches have been developed for unsupervised AD, which can be broadly categorised into four main families: one-class classifiers (OCCs), feature embedding-based, density-based, and reconstruction-based methods. OCCs aim to learn a decision boundary that encapsulates all normal samples. Classical OCC approaches employ shallow models such as support vector-based methods that learn a maximum-margin hyperplane (Schölkopf et al., 2001) or a hypersphere (Tax and Duin, 1999). To mitigate the limitations of manual feature engineering and extend to high-dimensional data, deep learning-based variants like DeepSVDD (Ruff et al., 2018) have been introduced. Feature embedding-based methods, on the other hand, leverage pre-trained deep models to extract representations of input data. These representations are then either stored in a memory bank (Roth et al., 2022;Lee et al., 2022) or used to train a student-teacher network (Zhang et al., 2024;Batzner et al., 2024;Patra and Ben Taieb, 2024). Density-based methods detect anomalies by estimating the probability distribution of normal samples, assuming that anomalies reside in low-density regions. While early methods include KDE (Kim and Scott, 2012), more recent deep-learning-based variants include DAGMM (Zong et al., 2018), CFLOW (Gudovskiy et al., 2022), and FastFlow (Yu et al., 2021). Lastly, reconstruction-based approaches learn to map normal samples into a lower-dimensional bottleneck and reconstruct them. The inability to accurately reconstruct samples during inference serves as a detection criterion. For a more comprehensive survey, we refer readers to Liu et al. (2024) and Ruff et al. (2021).
this section cite: ['b3', 'b15', 'b49', 'b45', 'b1', 'b62', 'b45', 'b46', 'b54', 'b11', 'b63', 'b44', 'b7', 'b14', 'b41', 'b61', 'b20', 'b41', 'b10', 'b14', 'b47', 'b36', 'b4', 'b21', 'b66', 'b60', 'b32', 'b27', 'b23', 'b18', 'b5', 'b50', 'b53', 'b46', 'b44', 'b25', 'b65', 'b1', 'b33', 'b22', 'b69', 'b11', 'b63', 'b31', 'b45']

Section: Data contamination.
Handling dataset contamination in AD typically assumes a low proportion of anomalies, allowing methods to prioritise normal instances (inlier priority) (Wang et al., 2019). However, in practice, this assumption is difficult to ensure since anomalies are often unknown. To mitigate contamination, Yoon et al. (2022) proposed a data refinement approach using an ensemble of one-class classifiers (OCCs) to filter suspected anomalies and create a cleaner dataset. While effective, this method incurs high computational costs and discards anomalies rather than leveraging them for improved generalisation via Outlier Exposure (Hendrycks et al., 2019). To address this, Qiu et al. (2022) introduced Latent Outlier Exposure (LOE), which iteratively assigns anomaly scores and infers labels using block coordinate descent while incorporating the contamination ratio to prevent degenerate solutions. However, estimating the contamination ratio remains a challenge. Perini et al. (2022) tackled this by leveraging an auxiliary dataset with a known contamination ratio, assuming domain similarity. Alternatively, Perini et al. (2023) fits a Dirichlet Process Gaussian Mixture Model to anomaly scores, though this approach lacks a closed-form solution. Despite these advancements, existing methods introduce computational overhead and are often impractical for modern pre-trained proprietary models, limiting their real-world applicability.
this section cite: ['b58', 'b61', 'b13', 'b41', 'b38', 'b35']

Section: Background
Let X ∈ X and Y ∈ Y denote a pair of random variables following a joint probability distribution P X,Y over the space X × Y, where X ⊆ R d and Y := {-1, +1}. Here, Y = +1 corresponds to the normal class, while Y = -1 represents the anomalous class.
The conditional distribution of normal samples is P X|Y =+1 denoted as P + X with PDF f + X . Likewise, the conditional distribution of anomalous samples is P X|Y =-1 denoted as P - X , with PDF f - X . The training dataset D + train := {x i } m i=1 contains only normal samples (uncontaminated) i.e., x i iid ∼ P + X . We denote the test dataset as D test := {(x i , y i )} n i=1 which contains both normal and anomalous samples i.e, (x i , y i ) iid ∼ P X,Y .
this section cite: []

Section: Density-based anomaly detection.
An anomaly can be defined as "an observation that deviates significantly from some concept of normality" (Ruff et al., 2021). This definition comprises two key aspects: the concept of normality and the significant deviation from it, which can be formalised using a probabilistic framework. The concept of normality is defined as the probability distribution of normal samples P + X . To formalise this further, we adopt the concentration assumption (Steinwart et al., 2005), which posits that although the data space X is unbounded, the high-density regions of P + X are bounded and concentrated. In contrast, P - X is assumed to be non-concentrated (Schölkopf and Smola, 2002), and is often approximated by a uniform distribution over X (Tax, 2001). Given the PDF f + X associated with P + X , which we refer to as inlier density, a data point x ∈ X is identified as an anomaly if it deviates substantially from this concept of normality, i.e., if it resides in a low-probability region under P + X . However, since f + X is typically unknown in practice, density-based methods approximate it using a density estimator.
Score-based anomaly detection. Density estimation poses significant challenges, particularly in high-dimensional spaces or when data is sparse, and often incurs substantial computational cost. Fortunately, in the context of anomaly detection, the goal is typically not to recover the exact data likelihood but rather to establish a ranking of data points based on their degree of normality. This motivates an alternative strategy: learning an anomaly score function s out (x) : X → R , which directly assigns an anomaly score to a data point x ∈ X , thereby quantifying its degree of anomalousness (Ruff et al., 2021). To complement this, the inlier score function is defined as s in (x) = -s out (x) , capturing the degree of normality, where higher values indicate that x is normal. For AD, first, we train a model to learn the anomaly score function s + out (x) using D + train . Then, we define the anomaly detector as
g λs (x) = +1, if s + out (x) ≤ λ s -1, if s + out (x) > λ s(1)
where λ s ≥ 0 is a pre-determined threshold (Perini et al., 2023(Perini et al., , 2022)). The density-based AD method can also be interpreted as a specific case of the score-based AD methods where the anomaly score s + out (x) = -ϕ(f + X (x)) and ϕ(•) is an order-preserving transformation chosen to be the logarithm. Data contamination. For training the AD method, a common assumption is that the training dataset D + train consists solely of i.i.d. samples from the normal data distribution P + X , without anomalies. However, this assumption is rarely satisfied in practice, since anomalies are typically unknown a priori. As a result, the training dataset is often contaminated with undetected anomalies. A more realistic assumption is that the dataset D ± train := {x i } m i=1 contains both normal and anomalous samples drawn from a mixture distribution P ± X with PDF f ± X (Huber and Ronchetti, 2011;Huber, 1992). Letting ϵ = P(Y = -1) denote the contamination factor, P ± X can be written as
P ± X = ϵ P - X + (1 -ϵ) P + X .(2)
As ϵ increases, the model trained on D ± train becomes biased towards the anomalous regions, reducing its ability to separate normal from anomalous samples (Qiu et al., 2022;Yoon et al., 2022). The existing literature examining the impact of contamination on unsupervised AD methods (Jiang et al., 2022;Qiu et al., 2022;Hien et al., 2024;Perini et al., 2023Perini et al., , 2022) ) typically considers contamination levels ranging from 0% to 20%. Additionally, an analysis of 57 datasets spanning Natural Language Processing and Computer Vision in ADBench (Han et al., 2022) [Appendix B.2, Figure B1] revealed that nearly 70% of the datasets exhibit anomaly ratios below 10%, with a median of 5%.
this section cite: ['b45', 'b51', 'b48', 'b52', 'b45', 'b35', 'b38', 'b17', 'b16', 'b41', 'b61', 'b20', 'b41', 'b14', 'b35', 'b38', 'b12']

Section: EPHAD: An evidence-based post-hoc adjustment framework
We consider the realistic scenario in which an AD model has already been trained on a possibly contaminated dataset D ± train . Instead of retraining the model, our goal is to adapt its test-time predictions to mitigate the impact of contamination. To this end, we introduce a novel Evidence-based Post-Hoc Adjustment Framework for Anomaly Detection (EPHAD), that corrects model predictions using an evidence function at test-time. The evidence function T (x) : X → R assigns higher values to samples deemed more likely to be normal and can incorporate domain-specific knowledge. Thus, EPHAD aligns with preparation-agnostic TTA methods (Xiao and Snoek, 2024).
For density-based AD (refer to Section 3), anomalies are identified as samples lying in the low-density regions under the distribution of normal samples P + X . However, due to data contamination, the trained model estimates the PDF f ± X of the contaminated distribution P ± X , as defined in (2), rather than the inlier PDF f + X . Given an evidence function T (x), EPHAD computes a revised PDF f ± X using exponential tilting as:
f ± X (x) = f ± X (x) exp(T (x)/β) Z β X ,(3)
where exp(T (x)/β) is the evidence scaled by a temperature parameter β ∈ R and Z β X = X f ± X (x) exp(T (x)/β) dx is the normalising constant. This formulation upweights normal samples according to the evidence while maintaining consistency with the model's original density. Proposition 4.1 provides a condition under which the revised PDF f ± X is closer to the inlinear PDF of normal samples f + X than the contaminated PDF f ± X , in terms of Kullback-Leibler (KL) divergence. Proposition 4.1. Let f + X , f ± X , and f ± X be PDFs over the same domain X . Then the KL divergence between f + X and f ± X is strictly less than the divergence between f + X and
f ± X iff E x∼P + X log exp(T (x)/β) Z β X > 0. (4
)
The proof is provided in Appendix A.1. Consequently, when condition (4) holds, the revised density f ± X (x) assigns higher relative likelihoods to true inliers, leading to improved separation between normal and anomalous samples. Hence, we expect EPHAD to yield better anomaly detection performance than the unadjusted model f ± X (x), provided that a suitable detection threshold is used. Moreover, it can be shown that (3) is the optimal solution to the KL-regularised objective
J KL ( f ± X ) := E x∼ f ± X [T (x)] -β KL( f ± X ∥f ± X ).(5)
This objective balances two competing goals: aligning the adjusted PDF with the evidence (first term) and maintaining fidelity to the original PDF (second term). The temperature parameter β controls this trade-off, recovering the evidence-driven solution as β → 0 and reverting to the original model as β → ∞. For the proof of (5), see Korbak et al. (2022). This interpretation also highlights a close connection to well-established TTA approaches used in generative models (Korbak et al., 2022;Mudgal et al., 2024;Li et al., 2024), where the model is viewed as an RL policy fine-tuned with a reward function encoding evidence or alignment criteria. In this view, EPHAD performs a KL-regularised shift of the contaminated density f ± X toward regions favored by the evidence function T (x) while preserving consistency with f ± X through the KL term.
this section cite: ['b60', 'b23', 'b23', 'b32', 'b27']

Section: Extension to score-based anomaly detection
Since estimating explicit densities is often infeasible in high-dimensional spaces, most modern AD methods rely on scores rather than PDFs. Recall that the inlier score function is an order-preserving transformation of the inlier PDF, i.e., s + in (x) = ϕ(f + X (x)), where ϕ(•) is a monotonic transformation such as the logarithm. When trained on contaminated data D ± train , the model learns a contaminated inlier score
s ± in (x) = ϕ(f ± X (x)).
Although ϕ is typically unknown and possibly non-invertible, the sample ranking induced by s ± in (x) is identical to that induced by f ± X (x)). Following energy-based model (EBM) formulations (LeCun et al., 2006), we can define the associated contaminated PDF as
f ± X (x) = exp(s ± in (x)) Z e X ,(6)
where
Z e X = X exp(s ± in (x))
is the normalising constant. Applying exponential tilting to f ± X (x) as in (3), we obtain:
f ± X (x) = f ± X (x) exp(T (x)/β) Z β X = exp(s ± in (x)) exp(T (x)/β) Z β X Z e X .(7)
Under Proposition 4.1, when condition (4) holds, the revised f ± X is closer to the true inlier density f + X in KL divergence than the unadjusted f ± X . Because AD depends only on the relative ordering of samples, the normalization constants in (7) can be ignored. The exponential mapping is strictly monotonic, so ranking and decision regions are preserved. Consequently, we can write
f ± X (x) ∝ exp(s ± in (x) + T (x)/β) := š± in (x),(8)
where we define š± in as the revised inlier score. The anomaly detector in (1) can thus be redefined as
g λs (x) = +1, if š± in (x) ≥ λ s , -1, otherwise. (9
)
This extension allows EPHAD to operate directly on score-based AD models, enabling post-hoc correction of models trained on contaminated datasets without requiring retraining or access to the original training procedure. In all subsequent experiments, we adopt this score-based formulation of EPHAD, reflecting the dominance of score-based methods in modern anomaly detection practice.
this section cite: ['b24']

Section: An illustrative example
To illustrate the effect of EPHAD, we use a toy dataset inspired by Qiu et al. (2022). The dataset is generated using a two-dimensional mixture model comprising three Gaussian components: c 1 := N (µ 1 , Σ 1 ), c 2 := N (µ 2 , Σ 2 ), c 3 := N (µ 3 , Σ 3 ). Here, each component follows a Gaussian distribution N (µ, Σ) with mean µ and covariance Σ. Normal samples are drawn from f + X = c 1 ,
Ground truth 0 .2 0.2 0. 4 0. 4 0.6 0 .8 Blind 0 .2 0.2 0.2 0. 4 0 .6 0 .8 Refine 0 .2 0.2 0. 2 0.4 0 .6 0.8 EPHAD (Ours) 0. 2 0 .4 0.6 0.8 0.0 0.2 0.4 0.6 0.8 1.0 Normal Anomaly
Figure 1: DeepSVDD trained on 2D synthetic contaminated training data with different configurations: (I) Supervised AD with ground truth labels for reference, (ii)"Blind" considering all samples as normal, (iii) "Refine" filtering out a fraction of the anomalies, and (iv) EPHAD updating the "Blind" anomaly detector using evidence computed on the samples available at test-time.
with µ 1 = [1, 1] T and Σ 1 = 0.07 I 2 . Anomalous samples are drawn from a mixture distribution f - X := 0.5c 2 + 0.5c 3 where µ 2 = [-0.25, 2.5] T , µ 3 = [-1, 0.5] T and Σ 2 = Σ 3 = 0.03 I 2 . The extended implementation details is provided in Appendix B.2. Using this setting, we create a contaminated dataset consisting 100 data points. We compare the baseline DeepSVDD (Ruff et al., 2018) across three configurations as illustrated in Figure 1: (i) "Blind", (ii) "Refine", and (iii) with EPHAD. We refer to the baseline model that treats all samples as normal as "Blind", while "Refine" denotes a model that iteratively filters out suspected anomalies during training. As an evidence function in EPHAD, LOF (Breunig et al., 2000) is computed on test samples at test time. The results in Figure 1 demonstrate that the "Blind" configuration mistakenly considers all anomalies as normal. The "Refine" configuration improves performance by filtering out a subset of anomalies. Finally, EPHAD establishes a clearer boundary around normal samples.
this section cite: ['b41', 'b46', 'b5']

Section: Determining the temperature parameter β
As previously discussed, EPHAD has only a single hyperparameter, β, which controls the trade-off between reliance on the original AD model and the evidence function T (x). A straightforward approach to selecting β would involve evaluating the AD performance of the prior and T (x) individually on a validation set and choosing β accordingly. However, this strategy introduces additional computational overhead at test time and requires access to a labelled validation set of sufficient size to ensure reliable performance estimation -conditions often impractical in real-world deployments. To address this limitation, we propose an adaptive extension of our approach, termed EPHAD-Ada, that determines the optimal β in an unsupervised manner using only test data at test time. This adaptation is inspired by the principle of Entropy Minimisation (EM) (Press et al., 2024), a widely-used technique in test-time adaptation (Xiao and Snoek, 2024). Motivated by the observation from Wang et al. (2021) that models tend to be more accurate when predictions are made with high confidence, we apply it to compute the hyperparameter β. Specifically, the computation of β depends on the entropy of the inlier probabilities derived from the scores of both the original model and the evidence function.
Computing inlier probability from the output scores. For an output score s ∈ R, the class label given the score can be modelled as a conditional random variable Y | S = s. Following this, the inlier probability can be expressed as
p Y =+1 (s) := P(Y = +1 | S = s) = P(S > s) = 1 -p s ,(10)
where p s := P(S ≤ s). Since p s is unknown in practice, we follow the approach of Perini et al. (2021) and treat it as a random variable P s with a prior distribution Beta(1, 1), corresponding to a uniform prior over [0,1]. Given that the label Y ∈ {+1, -1}, we model the conditional distribution Y | S = s as a Bernoulli random variable. To estimate p s , we draw samples s ′ ∼ S by first sampling x ∼ X and then computing the corresponding anomaly score s ′ . We record a success (b = 1) if s ′ ≤ s, and a failure (b = 0) otherwise. Repeating this procedure n times yields t successes and n -t failures. Then, according to Theorem 2 in Perini et al. (2021), the posterior distribution of P s given the observed binary outcomes b 1 , . . . , b n is Beta(1 + t, 1 + n -t). We estimate p s using the posterior mean of P s as
p s := E[P s ] = 1 + t 2 + n .(11)
In practice, the posterior is inferred from test samples, so the sample size n is constrained by the number of available test points. Finally, combining Equations ( 10) and (11), we obtain the estimated inlier probability for a data point x with anomaly score s as
p Y =+1 (s) = 1 -p s = 1 - 1 + t 2 + n .(12)
Finally, using (12), we compute the inlier probabilities p o Y =+1 (x) := p Y =+1 (s ± in (x)) and p e Y =+1 (x) := p Y =+1 (T (x)) from the scores of the original model and the evidence function, respectively.
Computing the value of the hyperparameter β. We define the empirical entropy of the binary predictive PMF p Y (x) as
H(p Y ) = - x∈Dtest [p Y =+1 (x) log p Y =+1 (x) + p Y =-1 (x) log p Y =-1 (x)] .(13)
The adaptive temperature parameter is then defined as
β ada = H(p e Y ) H(p o Y ) + δ ,(14)
where δ > 0 is a small constant introduced to ensure numerical stability. A low H(p o Y ) indicates that the original AD model produces confident (low-entropy) predictions, suggesting that a higher value of β should be used to place greater trust in this model. Conversely, a lower H(p e Y ) implies higher confidence in the evidence function, motivating a smaller β. Through this formulation, EPHAD-Ada enables unsupervised, test-time determination of β, thereby improving practicality and eliminating the need for labelled validation data.
this section cite: ['b39', 'b60', 'b57', 'b37', 'b37']

Section: Experiments
We evaluate the effectiveness of EPHAD for unsupervised AD across a range of datasets, including visual AD datasets (Section 5.1), tabular AD datasets (Section 5.2), and an industrial AD use case (Appendix C.2). To systematically investigate the impact of contamination at different levels in a rigorous and reproducible way, we introduce controlled contamination into the data, adhering to the experimental design employed in several prior studies (Jiang et al., 2022;Wang et al., 2025;Zhou and Wu, 2024). The evidence functions employed in the experiments are computed in an unsupervised manner without utilising ground-truth labels in the test set D test , mitigating the risk of overfitting. Unless stated otherwise, we use a contamination factor of ϵ = 0.1 and a parameter β = 0.5. An ablation study on different values of ϵ and β is presented in Section 5.3. For image and tabular datasets, we evaluate performance using the AUROC. Following prior work (Roth et al., 2022;Gudovskiy et al., 2022), AUROC is averaged across all categories for each dataset.
this section cite: ['b20', 'b55', 'b67', 'b44', 'b11']

Section: Experiments on visual AD datasets
Benchmark datasets. We assess the effectiveness of EPHAD in both sensory and semantic anomaly detection. Sensory AD focuses on detecting physical defects or imperfections, such as a broken capsule or a cut in a carpet, while semantic AD identifies anomalies belonging to a different semantic class-for instance, treating cats as normal and any other animal as anomalous. For sensory AD in industrial contexts, we evaluate performance using four well-established benchmark datasets: MVTecAD (Bergmann et al., 2019), MPDD (Jezek et al., 2021), ViSA (Zou et al., 2022), and RealIAD (Wang et al., 2024). For semantic AD, we utilise four commonly used datasets, including CIFAR-10, Fashion-MNIST, MNIST, and SVHN. Following the one-vs-rest protocol (Qiu et al., 2022), we construct k AD tasks per dataset, where k corresponds to the number of classes. For MVTecAD, ViSA, MPDD and RealIAD, we adopt the "overlap" setting, introducing ϵ% contamination into the training set by randomly selecting anomalous samples from the test set while retaining them in the test set Jiang et al. (2022). For the remaining datasets, we follow the "non-overlapping" setting, excluding anomalous samples used for contamination simulation from the test set. Our implementation is based on the public codebase from Jiang et al. (2022). Additional details are provided in Appendix B.1.
Baseline AD methods. We evaluate the performance of several state-of-the-art unsupervised anomaly detection methods, including PatchCore (Roth et al., 2022), PaDim (Defard et al., 2021), CFLOW  (Gudovskiy et al., 2022), FastFLOW (Yu et al., 2021), DRAEM (Zavrtanik et al., 2021), Reverse Distillation (RD) (Deng and Li, 2022), and ULSAD (Patra and Ben Taieb, 2024), both with and without the integration of EPHAD. Implementations for all methods, except ULSAD, are based on the Anomalib library (Akcay et al., 2022), while ULSAD is implemented using its official public code. Since, to the best of our knowledge, no existing AD method with contaminated data offers post-hoc adaptation in the same manner as EPHAD, our primary objective is to demonstrate the effectiveness of EPHAD by comparing its relative performance against the AD model and the evidence function alone. We also provide comparative analyses with three existing frameworks "Refine" (Yoon et al., 2022), Latent Outlier Exposure (LOE) (Qiu et al., 2022), and SoftPatch (Jiang et al., 2022) in Appendix C.3.
this section cite: ['b2', 'b19', 'b70', 'b56', 'b41', 'b20', 'b20', 'b44', 'b8', 'b11', 'b63', 'b64', 'b9', 'b33', 'b0', 'b61', 'b41', 'b20']

Section: Evidence function.
For the experiments, we employ Contrastive Language-Image Pre-training (CLIP) (Radford et al., 2021) as the evidence function for image-based datasets, following the anomaly detection approach as in WinCLIP (Jeong et al., 2023). We use CLIP as the evidence function T (x) in EPHAD. We start by defining two lists of textual prompt templates, T N = {n 1 , • • • , n k } and T A = {a 1 , • • • , a k }, corresponding to normal and anomalous classes, respectively. These templates are dataset-dependent, reflecting subjectivity (e.g., "missing wire" as anomalous for cables). For each label, compute the mean of text embeddings t N and t A . Finally, given an input image x, the evidence T (x) at test-time is computed as:
T (x) := exp (⟨e i (x), t N ⟩/γ) exp (⟨e i (x), t N ⟩/γ) + exp (⟨e i (x), t A ⟩/γ)
.
Additional implementation details are provided in Appendix B.3.1.
While CLIP has been previously applied as a standalone zero-shot anomaly detector, our methodology leverages it differently: we employ CLIP not as a complete detection system, but as an auxiliary source of evidence integrated into a more general and flexible framework. Importantly, EPHAD is not limited to foundation models such as CLIP; it can seamlessly incorporate domain-specific knowledge as well (see Section C.2), thereby broadening its applicability across diverse domains.
this section cite: ['b43', 'b18']

Section: Results.
In our experiments, as we adopt CLIP in the same manner as WinCLIP (Jeong et al., 2023), the baseline CLIP results reported here directly correspond to the standalone performance of WinCLIP. In Table 1, we observe that while zero-shot AD using CLIP performs well on real-world image datasets such as CIFAR10 and FMNIST, its effectiveness declines on domain-specific datasets like MVTec, MPDD, and ViSA, where existing AD methods, such as ULSAD, achieve superior performance. However, when these AD methods are used within the EPHAD framework with CLIP as an evidence function in a post-hoc manner, their performance improves in most cases. Notably, even when CLIP-based AD alone does not achieve the best results, as seen in SVHN, incorporating it within EPHAD still leads to significant improvements. For instance, CFLOW, PaDiM, and RD exhibit enhanced performance after using EPHAD, surpassing both CLIP and the standalone AD methods. This highlights the effectiveness of EPHAD in refining anomaly scores for better AD performance. In some cases, such as ULSAD on SVHN, we observe a decline in performance when integrating EPHAD compared to the standalone AD method. This typically occurs when the AD method substantially outperforms the evidence function. In such scenarios, overly relying on the evidence can diminish overall performance. To mitigate this effect, careful tuning of β enables the framework to adapt effectively to different datasets, AD methods, and evidence functions. A detailed analysis of the impact of varying β values is presented in Section 5.3.
Using the adaptive variant, EPHAD-Ada, we observe further improvements in certain settings, such as with PatchCore and DREAM on the RealIAD dataset. Interestingly, in cases where the default value of β = 0.5 led to decreased performance (e.g., ULSAD on SVHN or MPDD), EPHAD-Ada manages to overcome the problem, highlighting its effectiveness. Nevertheless, while EPHAD-Ada offers an unsupervised mechanism for determining β, its performance is often comparable to, or slightly below, that of EPHAD with the default value for β.
this section cite: ['b18']

Section: Experiments on tabular AD datasets
Benchmark datasets. We evaluate our proposed framework on 26 classical benchmark datasets from ADBench (Han et al., 2022). The classical datasets include datasets from different domains such as healthcare (e.g., annthyroid, breastw), astronautics (e.g. satellite), and finance (fraud). Following Qiu et al. (2022), we preprocess, split the dataset into the train and test sets and simulate contamination using synthetic anomalies created by adding zero-mean Gaussian noise with a large variance to the anomalous sample from the test set.
Baseline AD methods. We compare EPHAD against IFOREST (Liu et al., 2012), LOF (Breunig et al., 2000), DeepSVDD (Ruff et al., 2018), ECOD (Li et al., 2023b) and COPOD (Li et al., 2020) using ADBench (Han et al., 2022).
Evidence function. We use the output of Local Outlier Factor (LOF) (Breunig et al., 2000) and Isolation Forest (IForest) (Liu et al., 2012). Additional details provided in the Appendix B.3.2.
Results. The experimental results for a subset of the 26 benchmarking datasets are presented in Table 2, with the extended version provided in Appendix C.1. We observe that most AD methods benefit from our post-hoc adjustment framework EPHAD, often achieving performance improvements that surpass both the evidence function and the AD method in isolation. For example, COPOD, when updated with LOF as the evidence function on cover, glass and pendigits datasets, shows this behaviour. Additionally, as seen in the image-based experiments, performance degradation in certain cases arises when the framework places excessive emphasis on an evidence function that is substantially weaker than the AD method. However, as previously discussed, this limitation can be mitigated by appropriately tuning β. Similar to the results in the image-based experiments, we  observe improvements when using the adaptive variant EPHAD-Ada. In some scenarios, we also observe that EPHAD-Ada avoids the performance drop observed with EPHAD, such as with LOF on the ionosphere dataset and with DeepSVDD on the pendigits dataset. Nonetheless, the performance in most cases is similar to EPHAD with default value β, suggesting the need for further exploration.
this section cite: ['b12', 'b41', 'b30', 'b5', 'b46', 'b29', 'b28', 'b12', 'b5', 'b30']

Section: Ablation study
In this section, we first analyse the sensitivity of EPHAD to various contamination ratios. Then, we investigate the effect of the temperature β on AD performance.
this section cite: []

Section: Effect of varying contamination ratio.
Here, we evaluate the sensitivity of our proposed framework by varying the contamination ratio {0%, 5%, 10%, 15%}. The results are summarised in the Figure 2a. Applying EPHAD results in improvements across all contamination ratios for most of the AD methods. Furthermore, in the presence of a strong evidence function, such as CLIP, we can observe that the performance becomes almost constant even as the contamination ratio increases from 5% to 15%. An extended version is provided in Figure 3.
Effect of temperature parameter β. We also analyse the performance of the EPHAD by varying the temperature parameter β. In Figure 2b, we can see how β allows for controlling the trade-off between the prior AD method and the evidence. As discussed earlier, we observe that setting β ≈ 0 results in full reliance on T (x), while with increasing β, T (x) is disregarded and it defaults to the prior. Additionally, EPHAD-Ada achieves performance comparable to the best configuration of EPHAD across the explored range of β, highlighting its effectiveness. An extended version is provided in Figure 4.
this section cite: []

Section: Conclusion
Limitations and future work. While existing AD methods can serve as domain-agnostic evidence functions within EPHAD, the full potential of our framework is best realised by designing evidence functions that incorporate domain-specific knowledge. Exploring the interplay between datasets, AD methods, and evidence functions remains an open direction for future work. Another limitation concerns the parameter β, which has a significant influence on overall performance, as demonstrated in our experiments. Although we introduced an unsupervised strategy for estimating β in EPHAD-Ada, this approach does not always lead to performance improvements. We hypothesize that this may stem from uncalibrated inlier probability. Future work should thus investigate more reliable approaches for inferring β based on the anomaly scores and the underlying distributions of normal and anomalous samples in the test set. Finally, integrating explainability techniques into EPHAD represents an interesting direction for future research, as it could provide deeper insights for real-world applications.
Concluding remarks. Unsupervised AD methods typically assume anomaly-free training data, yet real-world datasets often contain undetected or mislabeled anomalies, leading to significant performance degradation. Existing approaches to address contamination often require access to model parameters, training data, or the training pipeline, limiting their practicality in real-world deployments. In this work, we introduce EPHAD, a simple, post-hoc adjustment framework that refines the outputs of any AD method trained on contaminated data by incorporating evidence collected at test-time. Extensive experiments demonstrate the effectiveness of EPHAD across diverse sources of evidence, multiple AD methods, and various datasets. Additionally, ablation studies analyse the impact of hyperparameters and varying contamination levels, highlighting the robustness of EPHAD.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The contributions are summarised in the introduction and also discussed in the abstract.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We have discussed limitations of this work in the Conclusion section.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: For each theoretical result, we provide all necessary details and the proof in the Appendix.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We have used all publicly available datasets, and our code can be found here.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
Answer: [Yes]
Justification: We have used all publicly available datasets, and our code can be found here.
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

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: We have provided key experimental details in the main paper and extended information in the Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: We provide standard errors for the extended tables provided in the Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes]
Justification: We have provided information on the computer resources in the Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: We ensured that our work adheres to the code of ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] Justification: [NA] Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: [NA] Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes]
Justification: We have used publicly available datasets, and for code, we have used opensource GitHub repositories after citing them in the paper.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes] Justification: Our anonymized code is accessible from here. We have also shared the details for setting up the environment and running the code.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable).
You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: [NA]
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According
to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: [NA] Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [No] Justification: LLM is solely used for grammar check and formatting purposes. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: A Proofs

this section cite: []

Section: A.1 Proof of Proposition 4.1
Proof. From (2), we have x). Additionally, from (3), we have
f ± X (x) = ϵf - X (x) + (1 -ϵ)f + X(
f ± X (x) = f ± X (x) exp(T (x)/β) Z β X Then, D KL (f + X ∥ f ± X ) = E x∼P + X log f + X (x) f ± X (x) = E x∼P + X log f + X (x) -log f ± X (x) = E x∼P + X log f + X (x) -log f ± X (x) exp(T (x)/β) Z β X = E x∼P + X log f + X (x) -log f ± X (x) - T (x) β + log Z β X = D KL (f + X ∥f ± X ) -E x∼P + X T (x) β -log Z β X = D KL (f + X ∥f ± X ) -E x∼P + X log exp(T (x)/β) Z β X .
We aim to increase the alignment between f + X and f ± X . Since the KL divergence is non-negative, if the expectation term is positive, we obtain
D KL (f + X ∥ f ± X ) ≤ D KL (f + X ∥f ± X )
. Therefore, the following condition should hold:
E x∼P + X log exp(T (x)/β) Z β X ≥ 0.
this section cite: []

Section: B Additional implementation details B.1 Benchmark datasets
For sensory AD in industrial settings, we use three widely recognised benchmark datasets. MVTecAD (Bergmann et al., 2019) comprises images from 15 categories (10 objects and 5 textures) with 3629 normal training images and 1258 anomalous and 467 normal test images, each containing pixel-level annotations of defects. MPDD (Jezek et al., 2021) targets metal part defects under varying conditions, offering 888 training images and test datasets consisting of 176 normal and 282 anomalous images across 6 metal part categories. ViSA (Zou et al., 2022) provides 10821 high-resolution images (9621 normal and 1200 anomalous) spanning 12 categories, capturing a range of anomalies such as scratches, cracks, missing parts, and misplacements. Each defect type is represented by 15-20 images, and some images feature multiple defects. RealIAD (Wang et al., 2024) is a large-scale industrial AD dataset comprising ∼ 150k images across 30 categories and having various types of defects such as scratches, dirt and missing parts. For experiments with RealIAD, we use the training split with 10% contamination and the test split provided by the authors. For the semantic datasets, using the one-vs-rest protocol, we create k AD tasks for each dataset, where k is the number of classes. In each task, one class is designated as normal, while the remaining classes are treated as anomalous. Across both sensory and semantic AD, the training datasets consist of a mixture of normal samples and a fraction ϵ of anomalous samples, reflecting realistic contamination scenarios.
this section cite: ['b2', 'b19', 'b70', 'b56']

Section: B.2 Details of the experiment using synthetic Data
The synthetic dataset is generated using a 2D Gaussian mixture model with three components. Normal samples are drawn from f + X (x) := N ([1, 1] T , 0.07I 2 ), while anomalous samples are sampled from f - X (x) := N ([-0.25, 2.5] T , 0.03I 2 ) + N ([-1, 0.5] T , 0.03I 2 ). For the experiments, we use DeepSVDD with a one-layer radial basis function (RBF) network. The hidden layer comprises three neurons, with their centres fixed at the mean of each Gaussian component, while the scales are optimized during training. The RBF network outputs a 1D scalar obtained as a linear combination of the outputs from the hidden layer. The centre is initialized randomly and made trainable, with an added bias term in the final layer. Although these modifications are not recommended by Ruff et al. (2018) to avoid collapse to a trivial solution, Qiu et al. (2022) observed that these changes enhance model flexibility and convergence. Following this, we train DeepSVDD using the Adam optimizer with a learning rate of 0.01, 200 epochs, and a mini-batch size of 25.
this section cite: ['b46', 'b41']

Section: B.3 Computing evidence functions
EPHAD relies on an evidence function T (x), computed during test-time, to refine anomaly scores by assigning higher values to samples from P + X than those from P - X . In this section, we introduce domainagnostic evidence functions applicable to image (Section B.3.1) and tabular datasets (Section B.3.2). While these functions are commonly used as standalone methods for anomaly detection, their role as evidence functions is novel and complementary to our framework. By operating in a transductive setting, they refine the outputs of an AD model initially trained in an inductive setting. Moreover, as shown in Section 5, using these evidence functions solely as anomaly scores does not always yield strong AD performance. However, when integrated into EPHAD, they significantly enhance the performance of a pre-trained model. Finally, the choice of an T (x) is not restricted to AD methods and can be adapted to incorporate domain-specific knowledge for improved effectiveness.
this section cite: []

Section: B.3.1 Evidence for visual datasets
For the evidence function in image-based AD, we propose using Contrastive Language-Image Pretraining (CLIP) (Radford et al., 2021), a robust large-scale framework that learns joint vision-language representations from web-collected image-text pairs. While CLIP has been explored in prior work as a zero-shot AD method (Jeong et al., 2023;Zhou et al., 2024), its performance varies across different datasets. Although CLIP excels in detecting anomalies in real-world image datasets such as CIFAR10, it faces significant challenges when applied to domain-specific datasets, particularly those used for industrial inspection, like MVTec. This limitation stems from the lack of domain-specific knowledge in CLIP's pre-training. In this section, we describe how CLIP is integrated into EPHAD as an evidence function T (x), leveraging its strengths while mitigating its limitations in specialized domains.
Given a dataset D := {(x j , t j )} n j=1 , CLIP trains an image encoder e i and a text encoder e t using contrastive learning (Chen et al., 2020), maximizing the cosine similarity between e i (x j ) and e t (t j ) for all (x j , t j ) ∈ D. For an input image x, CLIP performs zero-shot classification (Radford et al., 2021) by computing a k-way categorical distribution over a set of candidate class texts
C = {c 1 , . . . , c k } p(c = c j | x; c ∈ C) := exp (⟨e i (x), e t (c j )⟩/γ) s∈C exp (⟨e i (x), e t (s)⟩/γ) ,
where ⟨•, •⟩ denotes the cosine similarity, and γ is a temperature parameter that controls the sharpness of the distribution. Pairing class labels c ∈ C with prompt templates (e.g., a photo of a [c]) improves classification accuracy, and aggregating embeddings from multiple prompt variations (e.g., a cropped photo of a [c]) further enhances performance.
Building on Jeong et al. (2023), we use CLIP as evidence function T (x) in EPHAD. We start by defining two lists of textual prompt templates,
T N = {n 1 , • • • , n k } and T A = {a 1 , • • • , a k }, corresponding
to normal and anomalous classes, respectively. The list of prompts is provided in Table 3. These templates are dataset-dependent, reflecting subjectivity (e.g., "missing wire" as anomalous for cables).
For each label, we generate two lists of prompts for normal and anomalous cases using T N and T A and compute the mean of text embeddings t N and t A . Finally, given an input image x, the evidence damaged "c" a photo of the number "c" a photo of something flawless "c" "c" with flaw perfect "c" "c" with defect unblemished "c" "c" with damage "c" without flaw "c" without defect "c" without damage T (x) during test-time is computed as:
T (x) := exp (⟨e i (x), t N ⟩/γ) exp (⟨e i (x), t N ⟩/γ) + exp (⟨e i (x), t A ⟩/γ)
.
One potential concern when using pre-trained models like CLIP is the overlap between their training data and the test samples encountered in downstream tasks. Such overlap could challenge the assumption that test-time statistics are based solely on test data. However, Radford et al. (2021) provides an extensive analysis of this issue and shows that excluding all overlapping samples from CLIP's pre-training corpus leads to only a negligible performance drop. This result suggests that CLIP's effectiveness stems primarily from its generalisation ability rather than memorisation. Accordingly, our experiments emphasise this generalisation property, ensuring that the use of CLIP within our framework remains valid.
this section cite: ['b43', 'b18', 'b6', 'b43', 'b18', 'b43']

Section: B.3.2 Evidence for tabular datasets
For tabular datasets, we use the output of two classical unsupervised AD methods as evidence functions T (x), namely, Local Outlier Factor (LOF) (Breunig et al., 2000) and Isolation Forest (IForest) (Liu et al., 2012).
this section cite: ['b5', 'b30']

Section: Local Outlier Factor.
To detect anomalies, the local density of a point is compared to that of its k-nearest neighbours. Specifically, given a dataset D := {x j } n j=1 , the k-distance of a point x, denoted as k-distance(x), is defined as the distance from x to its k-th nearest neighbor.
Based on this, the k-distance neighborhood of x, denoted as N k (x), consists of all points whose distance from x is at most k-distance(x). Additionally, the reachability distance of x from a neighbor x i is computed as reach-dist k (x, x i ) = max{k-distance(x), d(x, x i )}, where d(x, x i ) represents the distance between x and x i .
Then, local reachability density (LRD) of x is computed as
LRD k (x) = xi∈N k (x) reach-dist k (x, x i ) |N k (x)| -1 .
Finally, the LOF-based evidence is computed as
T (x) = - xi∈N k (x) LRD k (xi) LRD k (x) |N k (x)| .
Isolation Forest. Anomalies are identified by recursively partitioning the data using a tree-based method, where features and split values are selected randomly. IForest operates under the assumption that anomalies are more susceptible to isolation due to their sparsity and distinctiveness in the feature space. Given D, IForest constructs multiple isolation trees (ITrees), where each data point x is assigned a depth representing the number of splits required to isolate it, referred to as the path length. Specifically, the evidence function T (x) is computed as: n) , where h(x) is the path length of x, i.e., the number of edges traversed from the root node to the leaf node where x is isolated in an ITree. E(h(x)) is the expected path length, i.e., the average path length across multiple ITrees, and c(n) is the average path length of an unsuccessful search.
T (x) = -2 -E(h(x)) c(
this section cite: []

Section: B.4 Experimental setup
For training the base AD methods, we use open-source Anomalib and ADBench libraries for experiments with image and tabular datasets, respectively. Our decision to rely on these public libraries was intentional, ensuring transparency and facilitating unbiased comparisons. For the training of each base AD model, we used a single NVIDIA A100 GPU. Then, we run inference using EPHAD on CPU.
this section cite: []

Section: C Extended results

this section cite: []

Section: C.1 Additional experiments on tabular Datasets
Table 4, 5, 6, and 7 summarise the results on a larger set of tabular datasets from ADBench. Each experiment is repeated with three seeds. We can observe that in most cases AD methods benefit from our post-hoc adjustment framework EPHAD, often achieving performance improvements that surpass both the evidence function and the AD method in isolation.    For the industrial setting, we utilise the simulated dataset introduced by Patra et al. (2024), which is generated by training a variational autoencoder on real-world data collected from an operational CSP plant. The dataset consists of thermal images of solar panels captured using infrared (IR) cameras, distinguishing it from the semantic and sensory anomaly datasets, as the images lack semantic structure and do not depict specific objects.
Baseline AD method. We evaluate the performance of the forecasting-based anomaly detection method ForecastAD, as proposed by the original authors, both with and without the integration of EPHAD. All experiments are conducted using the original implementation provided by the authors.
Rule-based evidence. Foundation models, such as CLIP, which were previously used in our experiments on image datasets, are not applicable in specialised applications, such as detecting anomalous behaviour in solar power plants, due to the lack of semantic content in thermal images. This makes zero-shot methods like WinCLIP and AnoCLIP inapplicable. In contrast, while EPHAD can in-corporate evidence from foundation models like CLIP, it also allows the seamless integration of domain-specific knowledge. To compute evidence, we utilise two of the four rules proposed by Patra et al. (2024) that indicate normal operational behaviour of the CSP plant. The first rule (R1) is based on the difference between consecutive images. Under normal conditions, the plant's temperature is expected to remain relatively stable; therefore, substantial deviations from one image to the next suggest potential anomalies. To quantify this, pixel-wise squared differences are computed between every pair of consecutive images, and the 95th percentile of these differences is extracted as the representative evidence for each pair. The second rule (R2) involves the difference from the average daily temperature. Here, samples with average temperatures significantly diverging from the typical daily average could indicate anomalous behaviour. For this, the mean temperature of each day is first determined, and then the absolute difference between each image's average temperature and that day's mean is computed to serve as the evidence. Results. The results presented in Table 8 underscore the effectiveness and adaptability of our approach. Under a 10% contamination setting, the baseline method ForecastAD experiences a performance drop of approximately 5%. However, by incorporating domain-specific rules R1 and R2 as sources of evidence using EPHAD and further using EPHAD-Ada, the performance nearly matches that on the clean dataset. It emphasises the value of leveraging structured, context-aware evidence to enhance the detection of anomalies. Importantly, foundation models like CLIP are unsuitable in this context due to the lack of semantic content in thermal imagery, rendering zero-shot approaches such as WinCLIP (Jeong et al., 2023) and AnoCLIP (Zhou et al., 2024) ineffective. EPHAD addresses this limitation by providing a flexible framework that integrates both powerful foundation models, where applicable, and domain-specific knowledge when necessary. This versatility enables EPHAD to deliver robust performance across diverse real-world anomaly detection tasks while maintaining efficiency and ease of deployment.
this section cite: ['b18']

Section: C.3 Comparison against LOE and SoftPatch
To ensure a comprehensive evaluation, we compare the performance of our proposed post-hoc framework against SoftPatch (Jiang et al., 2022) and both variants of LOE (Qiu et al., 2022). However, it is important to note that, unlike our approach, both SoftPatch and LOE modify the training process to account for contamination, making it inapplicable to pre-trained networks without access to the training dataset and pipeline, which is our main focus. First, for comparison with LOE, we conduct experiments using the Neural Transformation Learning-based (NTL) AD method (Qiu et al., 2021) and evaluate it under four configurations: "Blind", "Refine", LOE-Hard and LOE-Soft. Additionally, we follow the same setup as LOE by extracting image features using pre-trained ResNet152 and WideResNet50 for semantic and sensory datasets, respectively, which are then used to train NTL. The results, summarised in Table 9, show that given a good evidence function, i.e. the performance of the evidence is better than the "Blind" configuration, our simple test-time framework outperforms LOE. Results on MVTec, CIFAR10, FMIST, and SVHN are examples of this behaviour. Also, on the ViSA dataset, the performance improves over the "Blind" and "Refine" configurations. In the converse situations where the performance of the evidence is lower than the "Blind" configuration, we observe a reduction in performance which can be accounted for by putting more emphasis on the AD model by adjusting β. Now, we compare it against SoftPatch, an approach built upon PatchCore (Roth et al., 2022). SoftPatch enhances PatchCore by incorporating traditional anomaly detection (AD) techniques to refine the memory bank, specifically by identifying and re-weighting patches based on their outlier scores during training. While this strategy improves performance, it introduces a strong dependency on the choice of AD method and increases the computational burden of the training pipeline. For a fair comparison, we adopt the Local Outlier Factor (LOF) as the AD method, as it has been empirically found to be the most effective for SoftPatch. As shown in Table 10, our method, EPHAD, achieves competitive results despite being a fully post-hoc approach that requires no modification to the training process. Crucially, while SoftPatch is tailored for memory-bank-based methods, EPHAD is inherently model-agnostic and can be seamlessly applied to any combination of a pre-trained model and an evidence function. This versatility highlights EPHAD's broad applicability and practical utility across a diverse range of settings.
this section cite: ['b20', 'b41', 'b42', 'b44']

Section: C.4 Ablation on ϵ and β
Extended ablation on ϵ and β can be found in Figure 3, 4. We can make similar conclusions as discussed above in Section 5.3.
this section cite: []

Section: C.5 Effect of test set size n
The performance of our proposed framework, EPHAD, is influenced by both the pre-trained AD method and the evidence function. While the pre-trained AD method is affected only by the training data, for the evidence function, we evaluated two scenarios: (1) When using foundation models such as CLIP, the evidence function remains independent of the test sample distribution. (2) When employing traditional AD methods like Isolation Forest or Local Outlier Factor, the evidence function relies on the local density of test samples, meaning that an insufficient number of test samples could lead to less informative evidence which can be accounted for in EPHAD by adjusting the temperature parameter β. In Figure 5, we analyse the impact of varying the proportion of anomalies in the test set, which exhibits consistent improvements across all tested settings.
this section cite: []

Section: References
Ref_id:b0 Title: Anomalib: A Deep Learning Library for Anomaly Detection Year: (2022)
Ref_id:b1 Title: Efficientad: Accurate visual anomaly detection at millisecond-level latencies Year: (2024)
Ref_id:b2 Title: MVTEC ad-A comprehensive real-world dataset for unsupervised anomaly detection Year: (2019)
Ref_id:b3 Title: Explainable anomaly detection in sensor-based remote healthcare monitoring with adaptive temporal contrast Year: (2024)
Ref_id:b4 Title: Unsupervised anomaly detection algorithms on real-world data: how many do we need Year: (2024)
Ref_id:b5 Title: Lof: identifying density-based local outliers Year: (2000)
Ref_id:b6 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b7 Title: Adaptive deviation learning for visual anomaly detection with data contamination Year: (2025)
Ref_id:b8 Title: Padim: A patch distribution modeling framework for anomaly detection and localization Year: (2021)
Ref_id:b9 Title: Anomaly detection via reverse distillation from one-class embedding Year: (2022)
Ref_id:b10 Title: Robust variational autoencoders for outlier detection and repair of mixed-type data Year: (2020)
Ref_id:b11 Title: Cflow-ad: Real-time unsupervised anomaly detection with localization via conditional normalizing flows Year: (2022)
Ref_id:b12 Title: Adbench: Anomaly detection benchmark Year: (2022)
Ref_id:b13 Title: Deep anomaly detection with outlier exposure Year: (2019)
Ref_id:b14 Title: Anomaly detection with semi-supervised classification based on risk estimators Year: (2024)
Ref_id:b15 Title: Adapting visual-language models for generalizable anomaly detection in medical images Year: (2024)
Ref_id:b16 Title: Robust estimation of a location parameter Year: (1992)
Ref_id:b17 Title: Robust statistics Year: (2011)
Ref_id:b18 Title: Winclip: Zero-/fewshot anomaly classification and segmentation Year: (2023)
Ref_id:b19 Title: Deep learning-based defect detection of metal parts: evaluating current methods in complex conditions Year: (2021)
Ref_id:b20 Title: Softpatch: Unsupervised anomaly detection with noisy data Year: (2022)
Ref_id:b21 Title: Efficient test-time adaptation of vision-language models Year: (2024)
Ref_id:b22 Title: Robust kernel density estimation Year: (2012)
Ref_id:b23 Title: RL with KL penalties is better viewed as Bayesian inference Year: (2022)
Ref_id:b24 Title: A tutorial on energy-based learning Year: (2006)
Ref_id:b25 Title: Cfa: Coupled-hypersphere-based feature adaptation for target-oriented anomaly localization Year: (2022)
Ref_id:b26 Title: Interpreting unsupervised anomaly detection in security via rule extraction Year: (2023)
Ref_id:b27 Title: Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding Year: (2024)
Ref_id:b28 Title: COPOD: copula-based outlier detection Year: (2020)
Ref_id:b29 Title: Ecod: Unsupervised outlier detection using empirical cumulative distribution functions Year: (2023)
Ref_id:b30 Title: Isolation-based anomaly detection Year: (2012)
Ref_id:b31 Title: Deep industrial image anomaly detection: A survey Year: (2024)
Ref_id:b32 Title: Controlled decoding from language models Year: (2024)
Ref_id:b33 Title: Revisiting deep feature reconstruction for logical and structural industrial anomaly detection Year: (2024)
Ref_id:b34 Title: Detecting abnormal operations in concentrated solar power plants from irregular sequences of thermal images Year: (2024)
Ref_id:b35 Title: Estimating the contamination factor's distribution in unsupervised anomaly detection Year: (2023)
Ref_id:b36 Title: Uncertainty-aware evaluation of auxiliary anomalies with the expected anomaly posterior Year: (2025)
Ref_id:b37 Title: Quantifying the confidence of anomaly detectors in their example-wise predictions Year: (2021)
Ref_id:b38 Title: Transferring the Contamination Factor between Anomaly Detection Domains by Shape Similarity Year: (2022)
Ref_id:b39 Title: The entropy enigma: Success and failure of entropy minimization Year: (2024)
Ref_id:b40 Title:  Year: ()
Ref_id:b41 Title: Latent outlier exposure for anomaly detection with contaminated data Year: (2022)
Ref_id:b42 Title: Neural transformation learning for deep anomaly detection beyond images Year: (2021)
Ref_id:b43 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b44 Title: Towards total recall in industrial anomaly detection Year: (2022)
Ref_id:b45 Title: A unifying review of deep and shallow anomaly detection Year: (2021)
Ref_id:b46 Title: Deep one-class classification Year: (2018)
Ref_id:b47 Title: Deep semi-supervised anomaly detection Year: (2020)
Ref_id:b48 Title: Learning with kernels: support vector machines, regularization, optimization, and beyond Year: (2002)
Ref_id:b49 Title: Data augmentation in predictive maintenance applicable to hydrogen combustion engines: a review Year: (2025)
Ref_id:b50 Title: Estimating the support of a high-dimensional distribution Year: (2001)
Ref_id:b51 Title: A classification framework for anomaly detection Year: (2005)
Ref_id:b52 Title: One-class classification Year: (2001)
Ref_id:b53 Title: Support vector domain description Year: (1999)
Ref_id:b54 Title: Support Vector Data Description Year: (2004)
Ref_id:b55 Title: Softpatch+: Fully unsupervised anomaly classification and segmentation Year: (2025)
Ref_id:b56 Title: Real-iad: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection Year: (2024)
Ref_id:b57 Title: Tent: Fully test-time adaptation by entropy minimization Year: (2021)
Ref_id:b58 Title: Effective end-toend unsupervised outlier detection via inlier priority of discriminative network Year: (2019)
Ref_id:b59 Title: Make your home safe: Time-aware unsupervised user behavior anomaly detection in smart homes via loss-guided mask Year: (2024)
Ref_id:b60 Title: Beyond model adaptation at test time: A survey Year: (2024)
Ref_id:b61 Title: Self-supervise, refine, repeat: Improving unsupervised anomaly detection Year: (2022)
Ref_id:b62 Title: A unified model for multi-class anomaly detection Year: (2022)
Ref_id:b63 Title: FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows Year: (2021)
Ref_id:b64 Title: Draem -a discriminatively trained reconstruction embedding for surface anomaly detection Year: (2021)
Ref_id:b65 Title: Contextual affinity distillation for image anomaly detection Year: (2024)
Ref_id:b66 Title: AdaNPC: Exploring non-parametric classifier for test-time adaptation Year: (2023)
Ref_id:b67 Title: Outlier-probability-based feature adaptation for robust unsupervised anomaly detection on contaminated training data Year: (2024)
Ref_id:b68 Title: AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection Year: (2024)
Ref_id:b69 Title: Deep autoencoding gaussian mixture model for unsupervised anomaly detection Year: (2018)
Ref_id:b70 Title: Spot-the-difference self-supervised pre-training for anomaly detection and segmentation Year: (2022)
