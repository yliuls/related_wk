Title: Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion
Abstract: Multimodal learning (MML) is significantly constrained by modality imbalance, leading to suboptimal performance in practice. While existing approaches primarily focus on balancing the learning of different modalities to address this issue, they fundamentally overlook the inherent disproportion in model classification ability, which serves as the primary cause of this phenomenon. In this paper, we propose a novel multimodal learning approach to dynamically balance the classification ability of weak and strong modalities by incorporating the principle of boosting. Concretely, we first propose a sustained boosting algorithm in multimodal learning by simultaneously optimizing the classification and residual errors. Subsequently, we introduce an adaptive classifier assignment strategy to dynamically facilitate the classification performance of the weak modality. Furthermore, we theoretically analyze the convergence property of the cross-modal gap function, ensuring the effectiveness of the proposed boosting scheme. To this end, the classification ability of strong and weak modalities is expected to be balanced, thereby mitigating the imbalance issue. Empirical experiments on widely used datasets reveal the superiority of our method through comparison with various state-of-the-art (SOTA) multimodal learning baselines. The source code is available at https://github.  com/njustkmg/NeurIPS25-AUG.

Section: Introduction
In recent years, multimodal learning [33,48,43,47,21] has received growing attention for its ability to effectively integrate heterogeneous information. As extra information from multimodal data can be utilized, multimodal learning is expected to achieve better performance compared with unimodal approaches. However, contrary to expectations, multimodal learning has been surprisingly shown to underperform compared to unimodal ones in certain scenarios [42,35,46]. The root of this problem lies in the existence of the modality imbalance [42]. Concretely, different modalities in a joint-training paradigm typically converge at different speeds [35,45]. The fasterconverging modality, i.e., strong modality [49], tends to achieve higher performance, while the weak modality performs poorly. Subsequently, this disproportion in classification ability often leads to modality imbalance [42], ultimately resulting in lower performance.
Researchers have explored the modality imbalance issue from various perspectives in multimodal learning [42,35,52]. Given the inconsistent learning progress between strong and weak modalities, a natural idea [42,35,28,50] is to manually intervene in their learning processes to achieve rebalancing. Another type of method is to bridge the information gap between modality training phases and enhance the interaction between different modalities during training. To be specific, impressive works [52,11] such as MLA [52], ReconBoost [21] and DI-MML [11] focus on bridging the learning gap of different modalities through injecting the optimization information between modalities.
Although the above methods can rebalance multimodal learning, they focus more on balancing the learning process while failing to enhance the classification ability explicitly. Compared to weaker modalities, stronger modalities typically yield more robust classifiers due to their more sufficient information [49]. Is there a way to directly improve the performance of weak classifiers to balance the classification performance between strong and weak modalities? A natural choice is boosting [13,14], which utilizes the ensemble technique to enhance the ability of the weak classifier. We conduct a toy experiment to illustrate this idea on CREMAD dataset [4], where the classifier of weak modality is enhanced by the gradient boosting [14]. The results in Figure 1 present the comparison among naive MML, a model learning adjustment-based MML approach (G-Blend [42]), and gradient boostingbased MML (MML w/ GB). For MML w/ GB, we apply the gradient boosting algorithm to further improve the trained video model using naive MML, while keeping the audio model fixed. We can find that the classification gap between video and audio modalities of naive MML and G-Blend is relatively large. More importantly, for MML w/ GB, the accuracy of audio modality remains unchanged, but the accuracy of video is greatly improved, leading to the improvement of overall accuracy. This demonstrates the feasibility and effectiveness of using boosting to balance the classification ability of strong and weak modalities in mitigating modality imbalance.
Naive MML G-Blend MML w/ GB Ours 0.2 0.4 0.6 0.8 65.07% 64.65% 72.94% 85.15%
this section cite: ['b32', 'b47', 'b42', 'b46', 'b20', 'b41', 'b34', 'b45', 'b41', 'b34', 'b44', 'b48', 'b41', 'b41', 'b34', 'b51', 'b41', 'b34', 'b27', 'b49', 'b51', 'b10', 'b51', 'b20', 'b10', 'b48', 'b12', 'b13', 'b3', 'b13', 'b41']

Section: Accuracy
Video; Audio; Multi Figure 1: Comparison with naive MML, gradient boosting based MML (MML w/ GB), G-Blend [42], and Ours on CREMAD dataset. We find that enhancing the classification performance of the weak modality narrows the performance gap between the two modalities and improves overall performance.
According to the aforementioned observations, in this paper, we propose a novel multimodal learning approach by designing a sustained boosting algorithm to facilitate the classification ability of the weaker modality. Concretely, we first propose a sustained boosting algorithm by jointly optimizing the classification loss and residual error, aiming to enhance the classification performance of the weak modality. The algorithm takes features extracted by the encoder as input and provides predictions for the given data. Then, we employ the confident score [35] to monitor the learning status during joint training, and further propose an adaptive classifier assignment (ACA) strategy to adjust the classifier of weak modality. To this end, we can enhance the classification ability for the weak modality, thereby rebalancing the classification ability of strong and weak modalities. Meanwhile, we theoretically show that, under the boosting algorithm framework, the gap between different losses is provably convergent. In Figure 1, we present the classification enhancement results of our method (Ours). We can find that the performance of our method outperforms that of MML w/ GB thanks to the sustained boosting and adaptive classifier assignment strategy. Furthermore, it is worth mentioning that ReconBoost [21] also employs the gradient boosting algorithm for multimodal learning. However, unlike our approach, ReconBoost uses gradient boosting to iteratively learn complementary information across modalities. Our main contributions are outlined as follows:
• A novel sustained boosting algorithm in MML is proposed. This algorithm aims to simultaneously minimize the classification and residual errors to facilitate the classification ability of the weak modality.
• A novel adaptive classifier assignment strategy is proposed to dynamically enhance the classification ability of weak modality based on the learning status, thus rebalancing the classification ability of all modalities.
• We theoretically analyze the impact of the boosting algorithm on the loss gap between different modalities and prove its convergence.
• Experiments reveal that our approach can outperform SOTA baselines to achieve the best performance by a large margin on widely used datasets.
2 Related Work
this section cite: ['b41', 'b34', 'b20']

Section: Rebalanced Multimodal Learning
The goal of multimodal learning [48,24,33,39,22] is to fuse the multimodal information from diverse sensors. Compared to unimodal methods, multimodal learning can mine data information from different perspectives, thus the performance of multimodal learning should be better [30,38,20,16]. However, due to heterogeneity of multimodal data, multimodal learning often encounters imbalance problems [42,23] in practice, leading to performance degeneration of multimodal learning.
Early pioneering works [42,35,12,17] focus more on adaptively adjusting the learning procedure for different modalities. Representative approaches in this category employ different learning strategies, e.g., gradient modulation [35,28] and learning rate adjustment [50], to rebalance the learning of weak and strong modalities. Other approaches including MLA [52], ReconBoost [21] and IGM [25] take a different path, focusing on enhancing the interaction between modalities to address the modality imbalance problem. For example, MLA [52] designs an alternating algorithm to train different modalities iteratively. During the training phase, the interaction is enhanced by transferring the learning information between different modalities. ReconBoost [21] balances modality learning by leveraging gradient boosting to capture information from other modalities during interactive learning. IGM [25] employs a flat gradient modification strategy to enhance the interactive multimodal learning.
The aforementioned methods focus on rebalancing the learning process for weak and strong modalities while failing to explicitly facilitate the classification ability of the weak modality. In this paper, we aim to address the modality imbalance issue from facilitating the classification ability of weak modality and rebalancing the classification ability of weak and strong modalities.
this section cite: ['b47', 'b23', 'b32', 'b38', 'b21', 'b29', 'b37', 'b19', 'b15', 'b41', 'b22', 'b41', 'b34', 'b11', 'b16', 'b34', 'b27', 'b49', 'b51', 'b20', 'b24', 'b51', 'b20', 'b24']

Section: Boosting Method
Boosting algorithm [13,14,29,8,37] is one of the most important algorithms in ensemble learning. The core idea of boosting is to integrate multiple learners to create a strong learner. Adaboost [13], one of the earliest boosting algorithms, adjusts the weights of incorrectly classified data points, giving more attention to the harder-to-classify examples in each iteration. Gradient boosting [14], on the other hand, builds models in a stage-wise fashion, minimizing a loss function through gradient descent. It iteratively refines the overall model by fitting the negative gradient of the loss function [32] with respect to the model's predictions.
The key advantage of boosting lies in its ability to improve model accuracy without requiring complex individual models. Therefore, boosting becomes the natural choice for improving the performance of weak classifiers.
this section cite: ['b12', 'b13', 'b28', 'b7', 'b36', 'b12', 'b13', 'b31']

Section: Methodology

this section cite: []

Section: Multimodal Learning
For simplicity, we use two modalities, i.e., audio and video, for illustration. It is worth mentioning that our method can be easily adapted to cases with more than two modalities.
Assume that we have N data points, each of which has audio and video modalities. Without loss of generality, we use X = {(x a i , x v i )} N i=1 to denote the multimodal data, where x a i and x v i denote the i-th data point of audio and video, respectively. In addition, we are also given a category labels set Y = { y i | y i ∈ {0, 1} K } N i=1 , where K denotes the number of category labels. Given the above training information X and Y , the goal of multimodal learning is to train a model to fuse the multimodal information and predict its category label as accurately as possible.
For the sake of simplicity, we use superscript o to indicate the module corresponding to a specific modality in this section, where o ∈ {a, v}. With the rapid growth of deep learning, representative MML approaches [33,42,12,28] have adopted deep neural network (DNN) for multimodal learning. Following these methods, we also utilize DNN to construct our models. Specifically, we use ϕ o (•) to denote encoders. Then the features can be calculated by u o = ϕ o (x o ; θ o ), where θ o denotes the encoder parameters. Then, the prediction of given data can be calculated by a classifier ψ o (•): p o = ψ o (u o ; Θ o ), where Θ o denotes the parameters of the classifier. Based on p o and its ground-truth, the objective function can be written as:
L CE (X o , Y ) = 1 N N i=1 ℓ(p o i , y i ) = - 1 N N i=1 y ⊤ i log(p o i ),(1)
where Φ o ≜ {θ o , Θ o } denotes the parameters to be learned, X o ≜ {x o i } N i=1 and ℓ(•) denotes the cross entropy loss.
this section cite: ['b32', 'b41', 'b11', 'b27']

Section: Sustained Boosting
By training the model for each modality based on objective function (1), we can obtain multiple individual classifiers. Due to the existence of strong and weak modalities [49], these classifiers exhibit different classification abilities. Hence, we can employ boosting technique [14] to improve the classification ability of weak modality.
Concretely, assuming the classification performance of the o-modality requires improvement, we first apply the gradient boosting algorithm to train n classifiers for the o-modality. Since feature extraction focuses on common patterns, we set the encoders of all classifiers to be shared. Then the j-th classifier can be defined as:
Φ o t ≜ {θ o , Θ o t }, t ∈ {1, • • • , n}.
In practice, we adopt multiple fully-connected layers and nonlinear activation rectified linear unit (ReLU) [1] to construct our classification module. This module called the configurable classifier, is relatively independent and can be adjusted based on the classification ability. Furthermore, we adopt the shared head structure commonly used in MML [52,7,25] to strengthen the interaction between weak and strong modalities during training.
Inspired by gradient boosting [14], the classification ability can be facilitated through minimizing the residual error introduced by previous classifiers. Concretely, when we learn t-th classifier, the residual labels are defined as:
ŷo it = y i -λ t-1 j=1 y i ⊙ p o ij ,
where λ ∈ [0, 1] is used to soften hard labels [40], ⊙ denotes the element-wise product, and we utilize y i to mask non ground-truth labels to ensure the non-negativity of residual labels. Then the objective function can be defined as follows:
ϵ(x o i , y i , t) = ℓ p o it , ŷo it ,(2)
where p o it denotes the prediction obtained by t-th classifier for i-th data point. Since we utilize a shared encoder, the encoder will be updated when training the t-th classifier. Therefore, other classifiers must be updated simultaneously to prevent performance degradation. The corresponding objective can be formed as:
ϵ all (x o i , y i , t) = ℓ   p o it + t-1 j=1 p o ij , y i   = ℓ   t j=1 p o ij , y i   .(3)
Meanwhile, we have to ensure the first t -1 classifiers are well-trained. Hence, we define the following objective for t -1 classifiers:
ϵ pre (x o i , y i , t) = ℓ t-1 j=1 p o ij , y i .(4)
By combining (2), (3), and (4), the objective can be defined as:
L(x o i , y i , t) = ϵ(x o i , y i , t) + ϵ all (x o i , y i , t) + ϵ pre (x o i , y i , t).(5)
Unlike traditional gradient boosting [14], our method sustainedly minimizes classification and residual errors by optimizing (5). The loss function of sustained boosting can be formed as:
L SUB (X o , Y , n o ; Φ o ) = 1 N N i=1 L(x o i , y i , n o ),(6)
where n o denotes the number of classifier for o-modality.
Algorithm 1 Learning algorithm of our proposed method. Require: Training data X, category labels Y . Ensure: The learned DNN models for all modalities. 1: INIT Initialize the number of classifier n a = 1, n v = 1. Initialize iteration t = 1. Initialize DNN parameters Φ a t and Φ v t . 2: for t = 1 → #iterations do 3: Sample a mini-batch X t = {(x a i , x v i )} n b i=1 ; ▷ Learn MML models. 4: ∀x a i , x v i ∈ X t , calculate features u a i and u v i ; 5: Calculate predictions {p a ij } n a j=1 and {p v ij } n v j=1 . 6: Calculate loss in (6) based on predictions; 7: Update DNN parameters Φ a t and Φ v t based on SGD; 8:
if mod(t, t N ) = 0 then ▷ Adaptive Classification assignment strategy.
9: Calculate confident score {s a t , s v t } based on predictions; 10: if s a tσs v t > τ then 11: Add a classifier for audio modality; 12: n a = n a + 1; 13: else if s a tσs v t < τ then 14: Add a classifier for video modality; 15:
n v = n v + 1; 16:
end if 17:
end if 18: end for
this section cite: ['b48', 'b13', 'b0', 'b51', 'b6', 'b24', 'b13', 'b39', 'b13', 'b4']

Section: Adaptive Classifier Assignment
Thus far, we have defined a configurable classifier module and designed a sustained boosting in multimodal learning to enhance the classification performance of weak modality. However, recent studies [35] have shown that differences between modalities evolve dynamically due to imbalance issues in multimodal learning. This implies the need to design a strategy for enhancing classification ability that adapts to dynamic changes. Hence, we propose an adaptive classifier assignment strategy to adjust the number of the weak classifier. For simplicity, we redefine the modality classifiers as:
Φ o t ≜ {θ o , Θ o t }, t ∈ {1, • • • , n o },
where n o is the parameter to be updated. Then, we utilize confident score to monitor the learning status. At t-th iteration, confident score can be calculated by:
∀o ∈ {a, v}, s o t = 1 N N i=1 y ⊤ i n o j=1 p o ij .
The confident score reflects the classification ability of the models. Hence, if s a tσs v t > τ , we assign a new configurable classifier for video modality at this iteration, where σ ≥ 1 is the coefficient. τ is the dead zone for fault tolerance. On the contrary, we also assign a new configurable classifier for audio modality if s a tσs v t < τ . Our algorithm is summarized in Algorithm (1). In practice, we perform adaptive classification assignment strategy to determine if we need to adjust the classification ability every t N iterations.
this section cite: ['b34']

Section: Theoretical Analysis
The sustained boosting algorithm is introduced to reduce the loss gap between different modalities. In this section, we theoretically analyze its effect on minimizing this gap. We first define the gap function as follows:
G(Φ) = L a (Φ a ) -L v (Φ v ),(7)
where Φ a and Φ v respectively denote the parameters of audio and video modality, L a and L v denote the overall loss function L SUB for audio and video, respectively.Without loss of generality, we assume that L a > L v ; the case where L a < L v can be analyzed analogously.
We derive the following conclusions:
Theorem 1 (Convergence of Gap Loss, Informal) Under some assumptions for the loss function and the effectiveness of sustained boosting algorithm, we have:
G(Φ(T )) ≤ 1 1 + ν 2 κ 2 2Laβ 2 T G(Φ(0)) G(Φ(0))(8)
where ν, κ, L a and β are constant.
The results in Theorem 1 indicate that, by employing the gradient boosting algorithm, the loss gap between modalities converges at a rate of O(1/T ). The proof can be found in the appendix.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
Dataset: We carry out the experiments on six extensive multimodal datasets, i.e., CREMAD [4], KSounds [2], NVGesture [31], VGGSound [6], Twitter [51], and Sarcasm [3] datasets. The CREMAD, KSounds, and VGGSound datasets consist of audio and video modalities. NVGesture dataset contains three modalities, i.e., RGB, optical flow (OF), and Depth. Twitter and Sarcasm datasets consist of image and text modalities.
To ensure a fair comparison, we strictly follow the data partitioning strategy adopted by representative methods [42,35,52,21]. Specifically, the CREMAD dataset contains 7,442 clips, which are divided into training set with 6,698 samples and testing set with 744 samples. For KSounds dataset, which contains 19,000 video clips, is divided into training set with 15,000 clips, validation set with 1,900 clips, and testing set with 1,900 clips. VGGSound dataset includes 168,618 videos for training and validation, and 13,954 videos for testing. The NVGesture dataset is divided into 1,050 samples for training and 482 samples for testing. Twitter dataset is divided into training set with 3,197 pairs, validation set with 1,122 pairs and testing set with 1,037 pairs. Sarcasm dataset includes 19,816 pairs for the training set, 2,410 pairs for the validation set, and 2,409 pairs for the testing set. More details are provided in the appendix.
Baselines: We select two categories of methods for comparison, i.e., traditional multimodal fusion methods and rebalanced multimodal learning methods. In detail, traditional multimodal fusion methods include concatenation fusion (Concat), affine transformation fusion (Affine) [36], multi-layers lstm fusion (ML-LSTM) [34], prediction summation fusion (Sum) [48], and prediction weighting fusion (Weight) [48]. Rebalanced multimodal learning methods include MSES [15], G-blend [42], MSLR [50], OGM [35], PMR [12], AGM [28], MMParato [44], SMV [43], MLA [52], DI-MML [11], LFM [47], ReconBoost [21].
Evaluation Protocols: Following the setting of MLA [52] and ReconBoost [21], we adopt accuracy, mean average precision (MAP) and MacroF1 as evaluation metrics. The accuracy measures the proportion of correct predictions of total predictions. MAP returns the average precision of all samples. And MacroF1 calculates the average F1 across all categories.
Implementation Details: Following OGM [35], we employ ResNet18 [19] as the backbone to encode audio and video for CREMAD, KSounds and VGGSound datasets. All the parameters of the backbone are randomly initialized. For NVGesture dataset, we employ the I3D [5] as unimodal branch following the setting of [45]. We initialize the encoder with the pre-trained model trained on ImageNet. For the architecture of the configurable classifier, we explore a two-layer network, which can be denoted as "Layer1(D × 256) → ReLU → Layer2 (256 × K)". Here, D denotes the output dimensions of encoders, "Layer1"/"Layer2" are fully connected layer, and "ReLU" denotes the ReLU [1] activation layer. Furthermore, the Layer2 is utilized as shared head for all modalities as described in Section 3. Both Layer1 and Layer2 are randomly initialized. In addition, all hyper-parameters are selected by using the cross-validation strategy. Specifically, we use stochastic gradient descent (SGD) as the optimizer with a momentum of 0.9 and weight decay of 1 × 10 -4 . The initial learning rate is set to be 1 × 10 -2 for CREMAD, KSounds, VGGSound , and NVGesture datasets. During training, the learning rate is progressively reduced by a factor of ten upon observing loss saturates. The batch size is set to be 64 for CREMAD and KSounds datasets, 16 for VGGSound dataset, and 2 for NVGesture dataset. We set the iteration t N for checking whether to assign the classifier to 20 epochs for CREMAD, 5 for Twitter, 1 for Sarcasm, and 10 for VGGSound, KSounds, NVGesture datasets. For all datasets, we search λ in {0.1, 0.2, 0.33, 0.5, 1.0}. For all datasets, σ and τ are set to be 1.0 and 0.01, respectively. For Twitter and Sarcasm dataset, following [51,3], we adopt BERT [9] as the text encoder and ResNet50 [19] as the image encoder. We use Adam [26] as the optimizer, with an initial learning rate of 2 × 10 -5 . The batch size is set to 32 for Twitter and Sarcasm datasets. The other parameter settings are the same as audio-video datasets. For comparison methods, the source codes of all baselines are kindly provided by their authors. For fair comparison, all baselines also adopt the same backbone and initialization strategy for the experiment. All experiments are conducted on an NVIDIA GeForce RTX 4090 and all models are implemented with pytorch.
this section cite: ['b3', 'b1', 'b30', 'b5', 'b50', 'b2', 'b41', 'b34', 'b51', 'b20', 'b35', 'b33', 'b47', 'b47', 'b14', 'b41', 'b49', 'b34', 'b11', 'b27', 'b43', 'b42', 'b51', 'b10', 'b46', 'b20', 'b51', 'b20', 'b34', 'b18', 'b4', 'b44', 'b0', 'b50', 'b2', 'b8', 'b18', 'b25']

Section: Main Results

this section cite: []

Section: Classification Performance Comparison:
The classification results on all datasets are reported in Table 1, where "-" denotes that corresponding methods cannot applied to the dataset with more than two modalities. And Unimodal-1/2/3 is used to denote the results based on unimodal. Unimodal-1/2 respectively denote the video/audio for CREMAD and KSounds, and text/image for Twitter and Sarcasm. Unimodal-1/2/3 denotes the RGB/OF/Depth modality for NVGesture dataset, respectively. Furthermore, the results with a gray background indicate that the performance based on multimodal learning is inferior to that of the best unimodal approach. We can draw the following observations:
(1). Compared with unimodal baselines, traditional multimodal fusion methods and rebalanced multimodal learning methods can achieve better performance in almost all cases; (2). Our method can outperform existing SOTA baselines to achieve the best accuracy in all cases for multimodal situations; (2). The accuracy on NVGesture dataset demonstrates that our method can extend to the case with more than two modalities and achieve the best performance.
this section cite: ['b1']

Section: Sensitivity to Hyperparameter
We explore the influence of σ and λ on CREMAD and KSounds datasets in this section. More results can be found in appendix. 0.1 0.2 0.33 0.5 1 0 0.2 0.4 0.6 0.8 1 Video@CREMAD Audio@CREMAD Multi@CREMAD Video@KSounds Audio@KSounds Multi@KSounds    2. We can find that our method is not sensitive to threshold σ in a large range.
this section cite: []

Section: Sensitivity to Smoothing Factor λ:
We explore the influence of smoothing factor λ on CREMAD and KSounds datasets. The accuracy with different λ ∈ [0.1, 1] is reported in Figure 2. We can find that our method is not sensitive to hyper-parameter smoothing factor λ in a large range.
this section cite: []

Section: Ablation Study
We investigate the effectiveness of our method by analyzing the influence of the key components of our objectives in Equation ( 2), (3), and (4), respectively denoted as ϵ, ϵ o , and ϵ p . The accuracy results on CREMAD and KSounds datasets are reported in Table 2. From Table 2, we can find that: (1). Both objectives in Equation ( 2), (3), and (4) can boost multimodal performance; (2). While the unimodal performance of the method using all objectives may not always reach the highest level, it achieves a more balanced classification performance across modalities. More results are reported in appendix.
We further investigate the impact of residual learning on classification performance by comparing the performance of all t classifiers with that of the first t -1 classifiers during the training. The results are presented in Figure 3, where the former accuracy is denoted as "Full Prediction" and the latter is denoted as "Prediction of t -1 CLS". In Figure 3, we also present the number of the video classifier. We observe that the number of classifiers for the video modality has increased, and the performance of all t classifiers is generally superior to that of the first t -1 classifiers. This performance gain arises from our learning of the residual objective.
this section cite: ['b0', 'b1']

Section: Further Analysis

this section cite: []

Section: Impact of Weak Classifier Assignment Strategy:
We conduct an experiment to study the influence of adaptive classifier assignment strategy. Specifically, we design a fixed classifier assignment strategy for comparison. This approach allocates n (fix) classifiers for weak modality during the init stage. And we no longer dynamically adjust the number of classifiers during training for weak modality.
The results on CREMAD dataset are reported in Table 3, where n (fix) is set to be 10 and 12. The results in Table 3 demonstrate that our proposed adaptive classifier assignment strategy can boost performance compared with fixed classifier strategy. This is because our method dynamically adjusts modality classification performance in response to modality imbalance during training.
this section cite: []

Section: Visualization Results:
We further study the property of embeddings through visualization. Specifically, we illustrate the t-SNE [41] results on CREMAD dataset for naive multimodal learning (naive (e). ReconBoost@Video. Video (f). Ours@Video. 0 1 2 3 4 0.4 0.5 0.6 0.7 0.8 0.9 Accuracy Naive PMR AGM MLA ReconBoost Ours Figure 5: Training time (hrs).  MML), ReconBoost [21], and our method in Figure 4. From Figure 4, we can find that: (1). Compared to naive MML, our method and ReconBoost can learn more discriminative multimodal features, as both approaches enhance the weak modality using information from the strong modality; (2). Compared to ReconBoost, our method demonstrates significantly superior classification performance on the video modality, with several distinct categories highlighted by circle markers in Figure 4 (f). This improvement is primarily attributed to our explicit enhancement of the classification capabilities of the weaker modality.
Training Overhead: Considering that our method introduces additional computational overhead due to the use of the sustained boosting strategy, we compare its training cost with that of competitive SOTA baselines, including Naive, PMR, AGM, MLA, and ReconBoost, through empirical experiments under the same setting. The results are shown in Figure 5. It can be observed that our method achieves the best accuracy while maintaining competitive training time.
Convergence of Gap Loss: To further validate the convergence of the gap function, we conducted experiments on the CREMAD dataset. The change of gap function G(Φ) during training process is reported in Figure 6, where we also report the unimodal loss and multimodal loss. It can be observed that as training progresses, the gap function gradually converges. Moreover, we report the accuracy changes during training in Figure 7. From Figure 7, it can be seen that the accuracy exhibits a similar convergence trend.
this section cite: ['b40', 'b20', 'b0']

Section: Impact of the Model Capacity:
Since our method utilizes multiple designed classifiers for weak modality, it essentially utilizes more model parameters than baselines. Does our method benefit solely from having more model parameters? We conduct an experiment to verify this. Specifically, we Then we run the baselines and our method on CREMAD dataset. The results are shown in Table 4.
From Table 4, we can find that our approach improves accuracy by nearly 20% with only an additional 1M parameters compared to naive MML with the same network architecture. Our method also outperforms the naive MML baseline with larger backbone. Furthermore, we observe an interesting and counterintuitive phenomenon. That is, the method with ResNet34 is worse than that with ResNet18. The reason behind this may be that the ResNet34-based method is more difficult to converge.
this section cite: []

Section: Stability under Modality Missing:
Our method is adaptable to scenarios involving missing modalities. We comprehensively evaluate its performance under test-time missing modality conditions. Specifically, test-time missing refers to cases where the modalities are complete during training but missing during the testing phase. The experiments are conducted on CREMAD dataset with different missing rate [52]. Naive MML, ReconBoost [21], and MLA [52] are selected as baselines for comparison, where MLA introduces specifically designed algorithms to address the corresponding challenges in modality missing. We report the results with missing rate 20% and 50% in Table 5. We can find that as the modality missing rate increases, the performance of all methods declines. Nevertheless, our method consistently achieves the best performance under all missing rates, demonstrating the effectiveness of our method in scenario with missing modality.
this section cite: ['b51', 'b20', 'b51']

Section: Conclusion
To address the modality imbalance issue, we propose a novel multimodal learning approach by designing a sustained boosting algorithm to dynamically enhance the classification ability of weak modality. Concretely, we first propose a sustained boosting algorithm for multimodal learning by minimizing the classification and residual errors simultaneously. Then, we propose an adaptive classifier assignment strategy to dynamically facilitate the classification ability of weak modality. The effectiveness of the proposed boosting algorithm is theoretically guaranteed by analyzing the convergence properties of the cross-modal gap function. To this end, the classification ability can be rebalanced adaptively during the training procedure. Experiments on widely used datasets reveal that our proposed method can achieve state-of-the-art performance compared with various baselines by a large margin.
Limitations: Our proposed method mainly focuses on the classifier of each modality. For early fusion MML, our method can extend to balance the strong, weak, and fusion classification abilities. In addition, our theoretical analysis only examines the effect of the boosting algorithm on the convergence of the cross-modal gap function. In the full iterative framework, the overall convergence behavior and its influence on the model's learning capability warrant further investigation. We leave a more comprehensive investigation as future work.
this section cite: []

Section: References
Ref_id:b0 Title: Deep learning using rectified linear units (relu) Year: (2018)
Ref_id:b1 Title: Look, listen and learn Year: (2017)
Ref_id:b2 Title: Multi-modal sarcasm detection in twitter with hierarchical fusion model Year: (2019)
Ref_id:b3 Title: CREMA-D: crowd-sourced emotional multimodal actors dataset Year: (2014)
Ref_id:b4 Title: Quo vadis, action recognition? A new model and the kinetics dataset Year: (2017)
Ref_id:b5 Title: Vggsound: A large-scale audio-visual dataset Year: (2020)
Ref_id:b6 Title: Deep multiview clustering by contrasting cluster assignments Year: (2023)
Ref_id:b7 Title: Deep boosting Year: (2014)
Ref_id:b8 Title: BERT: pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b9 Title: What makes training multi-modal classification networks hard? Year: (2020)
Ref_id:b10 Title: Detached and interactive multimodal learning Year: (2024)
Ref_id:b11 Title: PMR: prototypical modal rebalance for multimodal learning Year: (2023)
Ref_id:b12 Title: A decision-theoretic generalization of on-line learning and an application to boosting Year: (1995)
Ref_id:b13 Title: Greedy function approximation: a gradient boosting machine Year: (2001)
Ref_id:b14 Title: Modality-specific learning rate control for multimodal classification Year: (2019)
Ref_id:b15 Title: Listen to look: Action recognition by previewing audio Year: (2020)
Ref_id:b16 Title: Improving zero-shot generalization and robustness of multi-modal models Year: (2023)
Ref_id:b17 Title: Auxformer: Robust approach to audiovisual emotion recognition Year: (2022)
Ref_id:b18 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b19 Title: Temporal multimodal learning in audiovisual speech recognition Year: (2016)
Ref_id:b20 Title: Reconboost: Boosting can achieve modality reconcilement Year: (2024)
Ref_id:b21 Title: What makes multi-modal learning better than single (provably) Year: (2021)
Ref_id:b22 Title: Modality competition: What makes joint training of multi-modal network fail in deep learning? (provably) Year: (2022)
Ref_id:b23 Title: Scaling up visual and vision-language representation learning with noisy text supervision Year: (2021)
Ref_id:b24 Title: Interactive multimodal learning via flat gradient modification Year: ()
Ref_id:b25 Title: Adam: A method for stochastic optimization Year: (2015)
Ref_id:b26 Title: Balancing multimodal training through game-theoretic regularization Year: (2025)
Ref_id:b27 Title: Boosting multi-modal model performance with adaptive gradient modulation Year: (2023)
Ref_id:b28 Title: Facial expression recognition via a boosted deep belief network Year: (2014)
Ref_id:b29 Title: Progressive modality reinforcement for human multimodal emotion recognition from unaligned multimodal sequences Year: (2021)
Ref_id:b30 Title: Online detection and classification of dynamic hand gestures with recurrent 3d convolutional neural networks Year: (2016)
Ref_id:b31 Title: Gradient boosting machines, a tutorial Year: (2013)
Ref_id:b32 Title: Multimodal deep learning Year: (2011)
Ref_id:b33 Title: Multi-modal feature fusion based on multi-layers LSTM for video emotion recognition Year: (2021)
Ref_id:b34 Title: Balanced multimodal learning via on-the-fly gradient modulation Year: (2022)
Ref_id:b35 Title: Film: Visual reasoning with a general conditioning layer Year: (2018)
Ref_id:b36 Title: Catboost: unbiased boosting with categorical features Year: (2018)
Ref_id:b37 Title: Two-stream convolutional networks for action recognition in videos Year: (2014)
Ref_id:b38 Title: FLAVA: A foundational language and vision alignment model Year: (2022)
Ref_id:b39 Title: Rethinking the inception architecture for computer vision Year: (2016)
Ref_id:b40 Title: Visualizing data using t-sne Year: (2008)
Ref_id:b41 Title: What makes training multi-modal classification networks hard? Year: (2020)
Ref_id:b42 Title: Enhancing multimodal cooperation via sample-level modality valuation Year: (2024)
Ref_id:b43 Title: Mmpareto: Boosting multimodal learning with innocent unimodal assistance Year: (2024)
Ref_id:b44 Title: Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks Year: (2022)
Ref_id:b45 Title: Learning to rebalance multi-modal optimization by adaptively masking subnetworks Year: (2025)
Ref_id:b46 Title: Facilitating multimodal classification via dynamically learning modality gap Year: (2024)
Ref_id:b47 Title: Comprehensive semi-supervised multi-modal learning Year: (2019)
Ref_id:b48 Title: Auxiliary information regularized machine for multiple modality feature learning Year: (2015)
Ref_id:b49 Title: Modality-specific learning rates for effective multimodal additive late-fusion Year: (2022)
Ref_id:b50 Title: Adapting BERT for target-oriented multimodal sentiment classification Year: (2019)
Ref_id:b51 Title: Multimodal representation learning by alternating unimodal adaptation Year: (2024)
