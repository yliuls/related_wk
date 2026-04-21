Title: MaxSup: Overcoming Representation Collapse in Label Smoothing
Abstract: Label Smoothing (LS) is widely adopted to reduce overconfidence in neural network predictions and improve generalization. Despite these benefits, recent studies reveal two critical issues with LS. First, LS induces overconfidence in misclassified samples. Second, it compacts feature representations into overly tight clusters, diluting intra-class diversity, although the precise cause of this phenomenon remained elusive. In this paper, we analytically decompose the LS-induced loss, exposing two key terms: (i) a regularization term that dampens overconfidence only when the prediction is correct, and (ii) an error-amplification term that arises under misclassifications. This latter term compels the network to reinforce incorrect predictions with undue certainty, exacerbating representation collapse. To address these shortcomings, we propose Max Suppression (MaxSup), which applies uniform regularization to both correct and incorrect predictions by penalizing the top-1 logit rather than the ground-truth logit. Through extensive feature-space analyses, we show that MaxSup restores intra-class variation and sharpens inter-class boundaries. Experiments on large-scale image classification and multiple downstream tasks confirm that MaxSup is a more robust alternative to LS. 4 * Equal contribution.

Section: Introduction
Multi-class classification [19,26] typically relies on one-hot labels, which implicitly treat different classes as mutually orthogonal. In practice, however, classes often share low-level features [31,44] or exhibit high-level semantic similarities [3,24,42], rendering the one-hot assumption overly restrictive. Such a mismatch can yield over-confident classifiers and ultimately degrade generalization [9].
To moderate overconfidence, Szegedy et al. [34] introduced Label Smoothing (LS), which combines a uniform distribution with the hard ground-truth label, thereby reducing the model's certainty in the primary class. LS has since become prevalent in image recognition [10,22,36,47] and neural machine translation [1,6], often boosting accuracy and calibration [23]. Yet subsequent work indicates that LS can overly compress features into tight clusters [15,28,41], hindering intra-class variability and transferability [5]. In parallel, Zhu et al. [48] found that LS paradoxically fosters overconfidence in misclassified samples, though the precise mechanism behind this remains uncertain.
In this paper, we reveal that LS's training objective inherently contains an error amplification term. This term pushes the network to reinforce incorrect predictions with exaggerated certainty, yielding highly confident misclassifications and further compressing feature clusters (Section 3.1, Table 1). Building on Zhu et al. [48], we characterize "overconfidence" in terms of the model's top-1 prediction, rather than through conventional calibration metrics. Through our analysis, we further show that punishing the ground-truth logit during misclassification reduces intra-class variation (Table 2), a phenomenon corroborated by Grad-CAM visualizations (Figure 2).
To overcome these shortcomings, we introduce Max Suppression (MaxSup), a method that retains the beneficial regularization effect of LS while eliminating its error amplification. Rather than penalizing the ground-truth logit, MaxSup focuses on the model's top-1 logit, ensuring a consistent regularization signal regardless of whether the current prediction is correct or misclassified. By preserving the ground-truth logit in misclassifications, MaxSup sustains richer intra-class variability and sharpens inter-class boundaries. As visualized in Figure 1, this approach mitigates the feature collapse and attention drift often induced by LS, ultimately leading to more robust representations. Through comprehensive experiments in both image classification (Section 4.2) and semantic segmentation (Section 4.3), we show that MaxSup not only alleviates severe intra-class collapse but also consistently boosts top-1 accuracy and robustly enhances downstream transfer performance (Section 4.1).
Our contributions are summarized as follows:
• We perform a logit-level analysis of Label Smoothing, revealing how the error amplification term inflates misclassification confidence and compresses features.
• We propose Max Suppression (MaxSup), removing detrimental error amplification while preserving LS's beneficial regularization. As shown in extensive ablations, MaxSup alleviates intra-class collapse and yields consistent accuracy gains.
• We demonstrate superior performance across tasks and architectures, including ResNet, MobileNetV2, and DeiT-S, where MaxSup significantly boosts accuracy on ImageNet and consistently delivers stronger representations for downstream tasks such as semantic segmentation and robust transfer learning.
this section cite: ['b18', 'b25', 'b30', 'b43', 'b2', 'b23', 'b41', 'b8', 'b33', 'b9', 'b21', 'b35', 'b46', 'b0', 'b5', 'b22', 'b14', 'b27', 'b40', 'b4', 'b47', 'b47']

Section: Related Work
We first outline mainstream regularization techniques in deep learning, then survey recent advances in Label Smoothing (LS), and finally clarify how our MaxSup diverges from prior variants.
this section cite: []

Section: Regularization
Regularization techniques aim to improve the generalization of deep neural networks by constraining model complexity. Classical methods like ℓ 2 [18] and ℓ 1 [49] impose direct penalties on large or sparse weights, while Dropout [32] randomly deactivates neurons to discourage over-adaptation. In the realm of loss-based strategies, Label Smoothing (LS) [34] redistributes a fraction of the label probability mass away from the ground-truth class, thereby improving accuracy and calibration [23].
Variants such as Online Label Smoothing (OLS) [45] and Zipf Label Smoothing (Zipf-LS) [21] refine LS by dynamically adjusting the smoothed labels based on a model's evolving predictions. However, they do not fully address the fundamental issue that emerges when the ground-truth logit is not the highest one (see Section 3.1, Table 1). Other loss-based regularizers focus on alternative aspects of the predictive distribution. Confidence Penalty [25] penalizes the model's confidence directly, while Logit Penalty [4] minimizes the global ℓ 2 -norm of logits, a technique reported to enhance class separation [15]. Despite these benefits, Logit Penalty can inadvertently shrink intra-class variation, thereby hampering transfer learning (see Section 4.1). Unlike the aforementioned methods, MaxSup enforces regularization by penalizing only the top-1 logit z max rather than the ground-truth logit z gt . In LS-based approaches, suppressing z gt for misclassified samples can worsen errors, whereas MaxSup applies a uniform penalty regardless of whether the model's prediction is correct. Consequently, MaxSup avoids the error amplification effect, retains richer intra-class diversity (see Table 2), and achieves robust transfer performance across diverse datasets and model families (see Table 3).
this section cite: ['b17', 'b48', 'b31', 'b33', 'b22', 'b44', 'b20', 'b24', 'b3', 'b14']

Section: Studies on Label Smoothing
Label Smoothing has also been studied extensively under knowledge distillation. For instance, Yuan et al. [43] observed that LS can approximate the effect of a teacher-student framework, while Shen et al. [30] investigated its role in such pipelines more systematically. Additionally, Chandrasegaran et al. [2] demonstrated that a low-temperature, LS-trained teacher can notably improve distillation outcomes. Concurrently, Kornblith et al. [15] showed that LS tightens intra-class clusters in the feature space, diminishing transfer performance. From a Neural Collapse perspective [46,8], LS nudges the model toward rigid feature clusters, as evidenced by the reduced feature variability measured in Xu and Liu [41]. Our goal is to overcome LS's inherent error amplification effect. Rather than adjusting how the smoothed label distribution is constructed (as in OLS or Zipf-LS), MaxSup directly penalizes the highest logit z max . This design ensures consistent regularization even if z gt is not the top logit, thereby avoiding the degradation in performance typical of misclassified samples under LS (see Section 3.2). Moreover, MaxSup integrates seamlessly into standard training pipelines, introducing negligible computational overhead beyond substituting the LS term.
this section cite: ['b42', 'b29', 'b1', 'b14', 'b45', 'b7', 'b40']

Section: Max Suppression Regularization (MaxSup)
We first partition the training objective into two components: the standard Cross-Entropy (CE) loss and a regularization term introduced by Label Smoothing (LS). By expressing LS in terms of logits (Theorem 3.3), we isolate two key factors: a regularization term that controls overconfidence and an error amplification term that enlarges the gap between the ground-truth logit z gt and any higher logits (Theorem 3.4, Equation ( 5)), ultimately degrading performance. To address these issues, we propose Max Suppression Regularization (MaxSup), which applies the penalty to the largest logit z max rather than z gt (Equation (8), Section 3.2). This shift delivers consistent regularization for both correct and incorrect predictions, preserves intra-class variation, and bolsters inter-class separability. Consequently, MaxSup mitigates the representation collapse found in LS, attains superior ImageNet-1K accuracy (Table 1), and improves transferability (Table 2, Table 3). The following sections elaborate on MaxSup's formulation and integration into the training pipeline.
this section cite: []

Section: Revisiting Label Smoothing
Label Smoothing (LS) is a regularization technique designed to reduce overconfidence by softening the target distribution. Rather than assigning probability 1 to the ground-truth class and 0 to all others, LS redistributes a fraction α of the probability uniformly across all classes: Definition 3.1. For a standard classification task with K classes, Label Smoothing (LS) converts a one-hot label y ∈ R K into a softened target label s ∈ R K :
s k = (1 -α)y k + α K ,(1)
where y k = 1 {k=gt} denotes the ground-truth class. The smoothing factor α ∈ [0, 1] reduces the confidence assigned to the ground-truth class and distributes α K to other classes uniformly, thereby mitigating overfitting, enhancing robustness, and promoting better generalization.
To clarify the effect of LS on model training, we first decompose the Cross-Entropy (CE) loss into a standard CE term and an additional LS-induced regularization term: Lemma 3.2. Decomposition of Cross-Entropy Loss with Soft Labels.
H(s, q) = H(y, q) + L LS ,(2)
where
L LS = α H 1 K , q -H(y, q) .(3)
Where, q is the predicted probability vector, H(•) denotes the Cross-Entropy, and 1 K is the uniform distribution introduced by LS. This shows that LS adds a regularization term, L LS , which smooths the output distribution and helps to reduce overfitting. (See Section A for a formal proof.) Building on Theorem 3.2, we next explicitly express L LS at the logit level for further analysis. Theorem 3.3. Logit-Level Formulation of Label Smoothing Loss.
L LS = α z gt - 1 K K k=1 z k ,(4)
where z gt is the logit corresponding to the ground-truth class, and 1 K K k=1 z k is the average logit. Thus, LS penalizes the gap between z gt and the average logit, encouraging a more balanced output distribution and reducing overconfidence. (See Section B for the proof.)
The behavior of L LS differs depending on whether z gt is already the maximum logit. Specifically, depending on whether the prediction is correct (z gt = z max ) or incorrect (z gt ̸ = z max ), we can decompose L LS into two parts: Corollary 3.4. Decomposition of Label Smoothing Loss.
L LS = α K zm<zgt z gt -z m Regularization + α K zn>zgt z gt -z n Error amplification ,(5)
where M and N are the numbers of logits below and above z gt , respectively (M + N = K -1).
Note that the error amplification term vanishes when z gt = z max .
1. Regularization: Penalizes the gap between z gt and any smaller logits, thereby moderating overconfidence.
2. Error amplification: Penalizes the gap between z gt and larger logits, inadvertently increasing overconfidence in incorrect predictions.
Although LS aims to combat overfitting by reducing prediction confidence, its error amplification component can be detrimental for misclassified samples, as it widens the gap between the ground-truth logit z gt and the incorrect top logit. Concretely:
1. Correct Predictions (z gt = z max ): The error amplification term is zero, and the regularization term effectively reduces overconfidence by shrinking the gap between z gt and any smaller logits. 2. Incorrect Predictions (z gt ̸ = z max ): LS introduces two potential issues:
• Error amplification: Increases the gap between z gt and larger logits, reinforcing overconfidence in incorrect predictions.
• Inconsistent Regularization: The regularization term lowers z gt yet does not penalize z max , which further impairs learning. These issues with LS on misclassified samples have also been systematically observed in prior work [39]. By precisely disentangling these two components (regularization vs. error amplification), we can design a more targeted and effective solution.
this section cite: ['b38']

Section: Ablation Study on LS Components.
To isolate the effects of each component in LS, we carefully perform a detailed and systematic ablation study on ImageNet-1K using a DeiT-Small model [36] without Mixup or CutMix. As indicated in Table 1, the performance gains from LS stem solely from the regularization term, whereas the error amplification term degrades accuracy. In contrast, our MaxSup omits the error amplification component and leverages only the beneficial regularization, thereby boosting accuracy beyond that of standard LS. Specifically, Table 1 shows that LS's overall improvement can be attributed exclusively to its regularization contribution; the error amplification term consistently reduces accuracy (e.g., to 73.63% or 73.69%). Disabling only the error amplification while retaining the regularization yields a slight but measurable improvement (75.98% vs. 75.91%). By fully removing error amplification and faithfully preserving the helpful aspects of LS, our MaxSup achieves 76.12% accuracy, clearly and consistently outperforming LS. This result underscores that MaxSup directly tackles LS's fundamental shortcoming by maintaining a consistent and meaningful regularization signal-even when the top-1 prediction is incorrect.
this section cite: ['b35']

Section: Max Suppression Regularization
Building on our analysis in Section 3.1, we find that Label Smoothing (LS) not only impacts correctly classified samples but also influences misclassifications in unintended and harmful ways. Specifically, LS suffers from two main limitations: inconsistent regularization and error amplification. As illustrated in Table 1, LS penalizes the ground-truth logit z gt even in misclassified examples, needlessly widening the gap between z gt and the erroneous top-1 logit. To resolve these critical shortcomings, we propose Max Suppression Regularization (MaxSup), which explicitly penalizes the largest logit z max rather than z gt . This key design choice ensures uniform regularization across both correct and misclassified samples, effectively eliminating the error-amplification issue in LS (Table 1) and preserving the ground-truth logit's integrity for more stable, robust learning.
this section cite: []

Section: Definition 3.5. Max Suppression Regularization
We define the Cross-Entropy loss with MaxSup as follows:
H(s, q)
CE with Soft Labels = H(y, q) CE with Hard Labels
+ L MaxSup Max Suppression Loss ,(6)
where
L MaxSup = α H 1 K , q -H(y ′ , q) ,(7)
and
y ′ k = 1 k=arg max(q)
, so that y ′ k = 1 identifies the model's top-1 prediction and y ′ k = 0 otherwise. Here, H 1 K , q encourages a uniform output distribution to mitigate overconfidence, while H(y ′ , q) penalizes the current top-1 logit. By shifting the penalty from z gt (the ground-truth logit) to z max (the highest logit), MaxSup avoids unduly suppressing z gt when the model misclassifies, thus overcoming Label Smoothing's principal shortcoming.
this section cite: []

Section: Logit-Level Formulation of MaxSup.
Building on the logit-level perspective introduced for LS in Section 3.1, we can express L MaxSup as:
L MaxSup = α z max -1 K K k=1 z k ,(8)
where z max = max k {z k } is the largest (top-1) logit, and 1 K K k=1 z k is the mean logit. Unlike LS, which penalizes the ground-truth logit z gt and may worsen errors in misclassified samples, MaxSup shifts the highest logit uniformly, thus providing consistent regularization for both correct and incorrect predictions. As shown in Table 1, this approach eliminates LS's error-amplification issue while preserving the intended overconfidence suppression.
Comparison with Label Smoothing. MaxSup fundamentally differs from LS in handling correct and incorrect predictions. When z gt = z max , both LS and MaxSup similarly reduce overconfidence. However, when z gt ̸ = z max , LS shrinks z gt , widening the gap with the incorrect logit, whereas MaxSup penalizes z max , preserving z gt from undue suppression. As illustrated in Figure 2, this helps the model recover from mistakes more effectively and avoid reinforcing incorrect predictions.
this section cite: []

Section: Gradient Analysis.
To understand MaxSup's optimization dynamics, we compute its gradients with respect to each logit z k . Specifically,
∂L MaxSup ∂z k = α 1 -1 K , if k = arg max(q), -α K , otherwise.(9)
Thus, the top-1 logit z max is reduced by α 1 -1 K , while all other logits slightly increase by α K . In misclassified cases, the ground-truth logit z gt is spared from penalization, avoiding the erroramplification issue seen in LS. For completeness, Appendix A provides the full gradient derivation. While [39] conducted a related gradient analysis of the training loss, it focuses specifically on the setting of selective classification, and examines a posthoc logit normalization technique to mitigate confidence calibration issues. However, this approach addresses only the overconfidence problem of label smoothing (LS), without tackling representation collapse. Moreover, our work presents a logit-level reformulation of LS that provides a deeper theoretical understanding of why LS amplifies errors.
Behavior Across Different Samples. MaxSup applies a dynamic penalty based on the model's current predictions. For high-confidence, correctly classified examples, it behaves similarly to LS by reducing overconfidence, effectively mitigating overfitting. In contrast, for misclassified or uncertain samples, MaxSup aggressively suppresses the incorrect top-1 logit, further safeguarding the groundtruth logit z gt . This selective strategy preserves a faithful and reliable representation of the true class while actively discouraging error propagation. As shown in Section 4.2 and Table 5, this promotes more robust decision boundaries and leads to stronger generalization.
this section cite: ['b38']

Section: Theoretical Insights and Practical Benefits.
MaxSup provides both theoretical and practical advantages over LS. Whereas LS applies a uniform penalty to the ground-truth logit regardless of correctness, MaxSup penalizes only the most confident logit z max . This dynamic adjustment robustly prevents error accumulation in misclassifications, ensuring more stable convergence. As a result, MaxSup generalizes better and achieves strong performance on challenging datasets. Moreover, as shown in Section 4.1, MaxSup preserves greater intra-class diversity, substantially improving transfer learning (Table 3) and yielding more interpretable activation maps (Figure 2).
this section cite: []

Section: Experiments
We begin by examining how MaxSup improves feature representations, then evaluate it on large-scale image classification and semantic segmentation tasks. Finally, we visualize class activation maps to illustrate the practical benefits of MaxSup.
this section cite: []

Section: Analysis of MaxSup's Learning Benefits
Having established how MaxSup addresses Label Smoothing's (LS) principal shortcomings (Section 3.1), we now demonstrate its impact on inter-class separability and intra-class variation-two properties essential for accurate classification and effective transfer learning. As noted in Section 3.1, Label Smoothing (LS) primarily curbs overconfidence for correctly classified samples but inadvertently triggers error amplification in misclassifications. This uneven penalization can overly compress intra-class feature representations. By contrast, Max-Sup uniformly penalizes the top-1 logit, whether the prediction is correct or incorrect, thereby eliminating LS's erroramplification effect and preserving finer distinctions within each class.
this section cite: []

Section: Intra-Class Variation and Transferability
Table 2 compares intra-class variation ( dwithin ) and inter-class separability (R 2 ) [15] for ResNet-50 trained on ImageNet-1K. Although all investigated regularizers decrease dwithin relative to a baseline, MaxSup yields the smallest reduction, indicating a stronger retention of subtle within-class diversity-widely associated with enhanced generalization and improved transfer performance.
These benefits are further underscored by the linear-probe transfer accuracy on CIFAR-10 (Table 3). While LS and Logit Penalty each boost ImageNet accuracy, both degrade transfer accuracy, likely by suppressing informative and transferable features. By contrast, MaxSup preserves near-baseline performance, implying that it maintains rich discriminative information crucial for downstream tasks. For extended evaluations on diverse datasets, see Table 12 in the appendix. As detailed in Section 3, both Label Smoothing (LS) variants and MaxSup impose penalties directly at the logit level, aligning with the perspective that various regularizers influence a model's representational capacity via distinct logit constraints [15]. Within this family of techniques, Logit Penalty and MaxSup both address the maximum logit, yet diverge fundamentally in their specific methods of regularization.
this section cite: ['b14', 'b14']

Section: Connection to Logit Penalty
Logit Penalty minimizes the ℓ 2 -norm of the entire logit vector, inducing a global contraction that can improve class separation but also reduce intra-class diversity, potentially hindering downstream transfer. By contrast, MaxSup focuses exclusively on the top-1 logit, gently nudging it closer to the mean logit. Because only the highest-confidence prediction is penalized, MaxSup avoids the uniform shrinkage observed in Logit Penalty, preserving richer intra-class variation-a property essential for robust transfer. Further insights into this behavior can be found in Section L, where logit-value histograms illustrate how each method affects the logit distribution.
this section cite: []

Section: Evaluation on ImageNet Classification
Next, we compare MaxSup to standard Label Smoothing (LS) and various LS extensions on the large-scale ImageNet-1K dataset.
this section cite: []

Section: Experiment Setup
Model Training Configurations.We evaluate both convolutional (ResNet [10], MobileNetV2 [27]) and transformer (DeiT-Small [36]) architectures on ImageNet [17]. For the ResNet Series, we train for 200 epochs using stochastic gradient descent (SGD) with momentum0.9, weight decay of 1 × 10 -4 , and a batch size of 2048. The initial learning rate is 0.85 and is annealed via a cosine schedule. 5 We also test ResNet variants on CIFAR-100 with a conventional setup: an initial learning rate of 0.1 (reduced fivefold at epochs 60, 120, and 160), training for 200 epochs with batch size 128 and weight decay 5 × 10 -4 . For DeiT-Small, we use the official codebase [36], training from scratch without knowledge distillation to isolate MaxSup's contribution. CutMix and Mixup are disabled to ensure the model optimization objective remains unchanged.
Hyperparameters for Compared Methods.We compare Max Suppression Regularization against a range of LS extensions, including Zipf Label Smoothing [21] and Online Label Smoothing [45].
Where official implementations exist, we adopt them directly; otherwise, we follow the methodological details provided in each respective paper. Except for any method-specific hyperparameters, all other core training settings remain identical to the baselines. Furthermore, both MaxSup and standard LS employ a linearly increasing α-scheduler for improved training stability (see Section F). This ensures a fair comparison under consistent and reproducible training protocols.
this section cite: ['b9', 'b26', 'b35', 'b16', 'b35', 'b20', 'b44']

Section: Experiment Results

this section cite: []

Section: ConvNet Comparison.
Table4 shows results for MaxSup alongside various label-smoothing and self-distillation methods on both ImageNet and CIFAR-100 benchmarks. Across all convolutional architectures tested, MaxSup consistently delivers the highest top-1 accuracy among label-smoothing approaches. By contrast, OLS [45] and Zipf-LS [21] exhibit less stable gains, suggesting their effectiveness may heavily hinge on specific training protocols.
To reproduce OLS and Zipf-LS, we apply the authors' official codebases and hyperparameters but do not replicate their complete training recipes (e.g., OLS trains for 250 epochs with a step-scheduled learning rate of 0.1, and Zipf-LS uses 100 epochs with distinct hyperparameters). Even under these modified settings, MaxSup remains robust, highlighting its effectiveness across a variety of training schedules-unlike the more schedule-sensitive improvements noted for OLS and Zipf-LS. DeiT Comparison. Table 5 summarizes performance for DeiT-Small on ImageNet across various regularization strategies. Notably, MaxSup attains a top-1 accuracy of 76.49%, surpassing standard Label Smoothing by 0.41%. In contrast, LS variants such as Zipf-LS and OLS offer only minor gains over LS, implying that their heavy reliance on data augmentation may limit their applicability to vision transformers. By outperforming both LS and its variants without additional data manipulations, MaxSup demonstrates robust feature enhancement. These findings underscore MaxSup's adaptability to different architectures and emphasize its utility in scenarios where conventional label-smoothing methods yield limited benefits. Fine-Grained Classification. Beyond largescale benchmarks like ImageNet, we further evaluate MaxSup on two fine-grained visual recognition tasks: CUB-200-2011 [37] and Stanford Cars [16]. These datasets pose unique challenges due to subtle inter-class differences, which often expose the limitations of standard regularization approaches. As shown in Table 6, MaxSup achieves the best performance across both datasets, surpassing LS and its recent variants. This demonstrates that MaxSup encourages the model to learn more discriminative and semantically rich representations that better capture fine-grained attributes, such as textures and part-level details. The consistent improvements on these benchmarks further validate MaxSup's capacity to generalize across different visual domains and its potential to enhance robustness in recognition scenarios where nuanced feature understanding is critical.
Long-Tailed Classification. To assess the effectiveness of MaxSup under data imbalance, we performed experiments on the CIFAR-10-LT dataset with imbalance ratios of 50 and 100, following the experimental settings described in [35]. The corresponding results are summarized in Table 7.
The evaluation compares three setups: Focal Loss, Focal Loss + LS, and Focal Loss + MaxSup. Across all imbalance ratios and splits (val/test), MaxSup consistently outperforms both the baseline and LS in overall accuracy, which jointly reflects the many-shot, medium-shot, and low-shot (minor class) performance. For example, at an imbalance ratio of 50 on the test split, MaxSup achieves 81.4% accuracy, outperforming Focal Loss (76.8%) by 4.6 percentage points, and LS (80.5%) by Table 8: Comparison of MaxSup, Label Smoothing (LS), and standard Cross Entropy (CE) on CIFAR-10-C. Lower is better. Values show mean(std) across three setups. Metric MaxSup LS CE Error (Corr) 0.362(0.055) 0.359(0.064) 0.354(0.015) NLL (Corr) 1.770(0.103) 1.476(0.111) 1.819(0.158) ECE (Corr) 0.145(0.003) 0.158(0.015) 0.260(0.015)
this section cite: ['b44', 'b20', 'b36', 'b15', 'b34']

Section: Corrupted Image Classification
To evaluate the effectiveness of MaxSup on out-ofdistribution (OOD) settings, we also conducted experiments on CIFAR10-C benchmark [12] shown in Table 8 following settings in [11]. Table 8 reports the performance of MaxSup and Label Smoothing (LS) on this benchmark using ResNet-50 as the backbone. Specifically, LS yields a better NLL (1.5730 vs. 1.8431), implying more confident probabilistic predictions. However, MaxSup achieves a better ECE (0.1479 vs. 0.1741), indicating better calibration of the predicted confidence scores. These results validate that MaxSup remains effective on OOD datasets, achieving performance comparable to LS across all three metrics.
this section cite: ['b11', 'b10']

Section: Ablation on the Weight Schedule.
We also systematically investigate how different α scheduling strategies impact MaxSup's performance. Empirical results indicate that MaxSup consistently maintains high accuracy across a wide range of schedules, further underscoring its robustness against hyperparameter changes. For additional details and discussions, refer to Section F. We further investigate MaxSup's applicability to downstream tasks by evaluating its performance on semantic segmentation using the widely adopted MMSegmentation framework. 6 Specifically, we adopt the Uper-Net [40] architecture with a DeiT-Small backbone, trained on ADE20K. Models pretrained on ImageNet-1K with either MaxSup or Label Smoothing are then fine-tuned under the same cross-entropy objective (Section 4.2.2).
this section cite: ['b39']

Section: Evaluation on Semantic Segmentation
Table 9 shows that initializing with MaxSup-pretrained weights yields an mIoU of 42.8%, surpassing the 42.4% achieved by Label Smoothing. This improvement indicates that MaxSup fosters more discriminative feature representations conducive to dense prediction tasks. By more effectively capturing class boundaries and within-class variability, MaxSup promotes stronger segmentation results, underscoring its potential to deliver features that are both transferable and highly robust.
this section cite: []

Section: Visualization via Class Activation Maps
Figure 2: Grad-CAM [29] visualizations for DeiT-Small models under three training setups: MaxSup (2nd row), Label Smoothing (3rd row), and a baseline (4th row). The first row shows the original images. Compared to Label Smoothing, MaxSup more effectively filters out non-target regions and highlights essential features of the target class, reducing instances where the model partially or entirely focuses on irrelevant areas.
To better understand how MaxSup fundamentally differs from Label Smoothing (LS) in guiding model decisions, we employ Gradientweighted Class Activation Mapping (Grad-CAM) [29], which highlights regions most influential for each prediction.
We evaluate DeiT-Small under three training setups: MaxSup (second row), LS (third row), and a baseline with standard cross-entropy (fourth row). As illustrated in Figure 2, MaxSup-trained models more effectively suppress background distractions than LS, which often fixates on unrelated objects-such as poles in "Bird," tubes in "Goldfish," and caps in "House Finch." This behavior reflects LS's error-enhancement mechanism, which can misdirect attention.
Moreover, MaxSup retains a wider spectrum of salient features, as exemplified in the Shark" and Monkey" images, where LS-trained models often omit crucial semantic details (e.g., fins, tails, or facial contours). These findings align with our analysis in Section I, clearly demonstrating that MaxSup preserves richer intra-class information. Consequently, MaxSup-trained models produce more accurate and consistent predictions by effectively leveraging fine-grained object cues. Further quantitative Grad-CAM overlay metrics (e.g., precision and recall for target regions) confirm that MaxSup yields more focused and comprehensive activation maps, further underscoring its overall efficacy.
this section cite: ['b28', 'b28']

Section: Conclusion
We examined the shortcomings of Label Smoothing (LS) and introduced Max Suppression Regularization (MaxSup) as a targeted and practical remedy. Our analysis shows that LS can unintentionally heighten overconfidence in misclassified samples by failing to sufficiently penalize incorrect top-1 logits. In contrast, MaxSup uniformly penalizes the highest logit, regardless of prediction correctness, thereby effectively eliminating LS's error amplification. Extensive experiments demonstrate that MaxSup not only improves accuracy but also preserves richer intra-class variation and enforces sharper inter-class boundaries, leading to more nuanced and transferable feature representations and superior transfer performance. Moreover, class activation maps confirm that MaxSup better attends to salient object regions, reducing focus on irrelevant background elements.
Limitations. Prior work [23] notes that LS-trained teachers may degrade knowledge distillation [13,14], and Guo et al. [8] suggests LS accelerates convergence via improved conditioning. Examining MaxSup's potential role in distillation and its overall impact on training dynamics would clarify these underlying effects. Recent studies [33,7] also show that ℓ 2 regularization biases final-layer features toward low-rank solutions, raising interesting questions about whether MaxSup behaves similarly.
Impact. In practical applications, MaxSup shows strong promise for systems demanding robust generalization and efficient transfer, and we have not observed any additional adverse effects or trade-offs. By offering researchers and practitioners both a clearer understanding of LS's limitations and a straightforward, computationally light, and easily integrable method to overcome them, MaxSup may help guide the development of more reliable and interpretable deep learning models.
this section cite: ['b22', 'b12', 'b13', 'b7', 'b32', 'b6']

Section: References
Ref_id:b0 Title: Steering large language models for machine translation with finetuning and in-context learning Year: (2023)
Ref_id:b1 Title: Revisiting label smoothing and knowledge distillation compatibility: What was missing? Year: (2022)
Ref_id:b2 Title: Hsva: Hierarchical semantic-visual adaptation for zero-shot learning Year: (2021)
Ref_id:b3 Title: Deconstructing the regularization of batchnorm Year: ()
Ref_id:b4 Title: Rethinking supervised pre-training for better downstream transferring Year: (2021)
Ref_id:b5 Title: Towards a better understanding of label smoothing in neural machine translation Year: (2020)
Ref_id:b6 Title: The persistence of neural collapse despite low-rank bias: An analytic perspective through unconstrained features Year: (2024)
Ref_id:b7 Title: Cross entropy versus label smoothing: A neural collapse perspective Year: (2024)
Ref_id:b8 Title: Online knowledge distillation via collaborative learning Year: (2020)
Ref_id:b9 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b10 Title: Robust classification by coupling data mollification with label smoothing Year: (2025)
Ref_id:b11 Title: Benchmarking neural network robustness to common corruptions and perturbations Year: (2019)
Ref_id:b12 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b13 Title: Distilling causal effect of data in class-incremental learning Year: ()
Ref_id:b14 Title: Why do better loss functions lead to less transferable features? Year: (2021)
Ref_id:b15 Title: 3d object representations for finegrained categorization Year: (2013)
Ref_id:b16 Title: Imagenet classification with deep convolutional neural networks Year: (2012)
Ref_id:b17 Title: A simple weight decay can improve generalization Year: (1991)
Ref_id:b18 Title: The mnist database of handwritten digits Year: (1998)
Ref_id:b19 Title: Adaptive label smoothing with selfknowledge in natural language generation Year: (2022)
Ref_id:b20 Title: Efficient one pass self-distillation with zipf's label smoothing Year: (2022)
Ref_id:b21 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b22 Title: When does label smoothing help? Year: (2019)
Ref_id:b23 Title: Chils: Zero-shot image classification with hierarchical label sets Year: (2023)
Ref_id:b24 Title: Regularizing neural networks by penalizing confident output distributions Year: (2017)
Ref_id:b25 Title: Imagenet large scale visual recognition challenge Year: (2015)
Ref_id:b26 Title: Mobilenetv2: Inverted residuals and linear bottlenecks Year: (2018)
Ref_id:b27 Title: No reason for no supervision: Improved generalization in supervised models Year: (2022)
Ref_id:b28 Title: Grad-cam: Visual explanations from deep networks via gradientbased localization Year: (2019-10)
Ref_id:b29 Title: Is label smoothing truly incompatible with knowledge distillation: An empirical study Year: (2021)
Ref_id:b30 Title: A survey of hierarchical classification across different application domains Year: (2011)
Ref_id:b31 Title: Dropout: A simple way to prevent neural networks from overfitting Year: (2014)
Ref_id:b32 Title: Neural collapse versus low-rank bias: Is deep neural collapse really optimal? arXiv preprint Year: (2024)
Ref_id:b33 Title: Rethinking the inception architecture for computer vision Year: (2016)
Ref_id:b34 Title: Long-tailed classification by keeping the good and removing the bad momentum causal effect Year: (2020)
Ref_id:b35 Title: Training data-efficient image transformers & distillation through attention Year: (2021)
Ref_id:b36 Title: The caltech-ucsd birds-200 Year: (2011)
Ref_id:b37 Title: Mitigating neural network overconfidence with logit normalization Year: (2022)
Ref_id:b38 Title: Understanding why label smoothing degrades selective classification and how to fix it Year: (2024)
Ref_id:b39 Title: Unified perceptual parsing for scene understanding Year: (2018)
Ref_id:b40 Title: Quantifying the variability collapse of neural networks Year: (2023)
Ref_id:b41 Title: Exploring hierarchical graph representation for large-scale zero-shot image classification Year: (2022)
Ref_id:b42 Title: Revisiting knowledge distillation via label smoothing regularization Year: (2020)
Ref_id:b43 Title: Visualizing and understanding convolutional networks Year: (2014)
Ref_id:b44 Title: Delving deep into label smoothing Year: (2021)
Ref_id:b45 Title: Are all losses created equal: A neural collapse perspective Year: (2022)
Ref_id:b46 Title: Sp-vit: Learning 2d spatial priors for vision transformers Year: (2022)
Ref_id:b47 Title: Rethinking confidence calibration for failure prediction Year: (2022)
Ref_id:b48 Title: Regularization and variable selection via the elastic net Year: (2005)
