Title: Trusted Multi-View Classification with Expert Knowledge Constraints
Abstract: Trusted multi-view classification (TMVC) based on the Dempster-Shafer theory has gained significant recognition for its reliability in safetycritical applications. However, existing methods predominantly focus on providing confidence levels for decision outcomes without explaining the reasoning behind these decisions. Moreover, the reliance on first-order statistical magnitudes of belief masses often inadequately capture the intrinsic uncertainty within the evidence. To address these limitations, we propose a novel framework termed Trusted Multi-view Classification Constrained with Expert Knowledge (TM-CEK). TMCEK integrates expert knowledge to enhance feature-level interpretability and introduces a distribution-aware subjective opinion mechanism to derive more reliable and realistic confidence estimates. The theoretical superiority of the proposed uncertainty measure over conventional approaches is rigorously established. Extensive experiments conducted on three multiview datasets for sleep stage classification demonstrate that TMCEK achieves state-of-the-art performance while offering interpretability at both the feature and decision levels. These results position TMCEK as a robust and interpretable solution for MVC in safety-critical domains. The code is available at https://github.com/  jie019/TMCEK_ICML2025.

Section: Introduction
Sleep disorder is related to lots of diseases like insomnia, narcolepsy, obstructive sleep apnea syndrome (OSA) (Jahrami et al., 2022). Sleep stage classification (SSC) is the primary diagnostic tool for sleep disorder (Guillot et al., 2020;Wulff et al., 2010). Multi-view learning (MVL) is a powerful paradigm that leverages diverse data representations to improve model performance and robustness (Zhang et al., 2020;Liang et al., 2022;Wei et al., 2025;Yuan et al., 2025), making it particularly valuable in SSC that is mainly based on multi-view polysomnography signals such as EEG, EOG and EMG (Phan et al., 2020). By leveraging data from multiple perspectives, MVL not only enhances diagnostic accuracy but also offers better generalization across diverse patient populations.
While MVL addresses many challenges in SSC, trustworthiness and interpretability remain critical concerns (Wang et al., 2023;Zou et al., 2023). Trusted multi-view learning aims to improve the reliability of predictions by incorporating uncertainty estimation (Han et al., 2023). However, two significant limitations persist: (1) feature-level opacity: Current trusted learning methods often function as black boxes at the feature level, failing to clarify which features are critical and how they contribute to the decision-making process. The lack of transparency reduces trust and interpretability for clinicians and patients. (2) Inaccurate confidence estimates at decision level: Existing methods primarily rely on the quantity of evidences for uncertainty estimation, without considering their distribution. As a result, confidence estimates may deviate significantly from expected values, particularly in scenarios involving ambiguous or conflicting data, undermining their practical utility.
In this paper, we propose a novel trusted multi-view learning framework that combines interpretability at both the feature and decision levels. Specifically, we firstly utilize Gabor functions in the initial layers of the model to embed expert knowledge into feature extraction, enabling explicit representation of critical features. This approach allows for better understanding of which features contribute to classification decisions, improving transparency and reliance. Second, we improve uncertainty estimation by introducing the distribution of evidence as an additional factor, moving beyond traditional reliance on the quantity of evidence. By capturing both the magnitude and distribution of evidence, our method provides more reliable confidence estimates, especially in scenarios with ambiguous or conflicting data. These innovations enhance the interpretability, reliability, and robustness of multi-view learning in high-stakes applications like automated sleep stage classification.
Our main contributions are summarized as follows: (1) We propose a novel trusted multi-view learning framework that enhances feature-level interpretability by embedding expert knowledge, enabling explicit identification of critical features contributing to classification decisions. (2) We introduce an improved uncertainty estimation mechanism by incorporating the distribution of evidence, providing more reliable and realistic confidence estimates. (3) Due to the rich expert knowledge available in the sleep domain, we instantiate our method on the sleep datasets, and the experiments show that our approach improves both interpretability and reliability, while also outperforming baseline methods in terms of accuracy.
this section cite: ['b11', 'b8', 'b37', 'b41', 'b16', 'b35', 'b40', 'b26', 'b33', 'b47', 'b9']

Section: Related Work

this section cite: []

Section: Trusted Multi-View Learning
Multi-view learning has become a powerful approach for integrating complementary information from multiple data representations, leading to more robust and accurate models across a range of applications (Liang et al., 2024;Wen et al., 2024;Zhang et al., 2024). However, these methods often fail to capture the uncertainty of their predictions. Evidential deep learning (EDL) (Sensoy et al., 2018) uses subjective logic-based approaches to avoid sampling by explicitly modeling uncertainty, making them computationally efficient. Recent work extends EDL into multi-view learning (Liang et al., 2025). A notable approach, trusted multi-view classification (TMC) (Han et al., 2023), employs Dempster's combination rule (Jøsang & Hankin, 2012), assigning lower weights to views with high uncertainty, thereby prioritizing more reliable views. Building on this foundation, several aggregation methods have advanced uncertainty handling by refining how views are integrated (Liu et al., 2023;Zhang et al., 2023). These methods commonly exhibit the property that adding another opinion reduces overall uncertainty. Since data of different views are might not aligned, RCML (Xu et al., 2024) proposes a new aggregation method to ensure that integrating conflicting views appropriately raises uncertainty. Despite these advances, current methods exhibit notable limitations: feature-level opacity and Inaccurate confidence estimates at decision level. In light of these shortcomings, this work proposes a novel framework to address these gaps by enhancing interpretability at the feature level and introducing a more comprehensive uncertainty estimation method that moves beyond evidence magnitude alone.
this section cite: ['b17', 'b36', 'b42', 'b29', 'b18', 'b9', 'b13', 'b20', 'b44', 'b39']

Section: Automatic Sleep Stage Classification
Existing work on automatic sleep staging can be categorized three main categories based on the types of signal input representation of the network. The first uses raw one-dimensional (1D) signals directly as input to capture sequential features by one-dimensional convolutional neural networks (1D CNNs) (Chambon et al., 2017), recurrent neural networks (RNNs) (Dong et al., 2016) and attention mechanism (Phan et al., 2018b). The second converts raw signals into two-dimensional (2D) spectrograms using techniques such as the continuous wavelet transform (Kuo et al., 2022) or short-time Fourier transform (STFT) (Guillot et al., 2019). Two-dimensional convolutional neural networks (2D CNNs) are then used to process these spectrograms, which capture essential frequency characteristics linked to each sleep stage. The third combines both temporal and time-frequency representations which employs a dual-stream architecture. In this structure, each branch of the model processes a different view. Then integrating the outputs from each view to produce a fusion output by concatenate operation. XSleepNet (Phan et al., 2020) uses the outputs of three branches to compute losses. SleepPrintNet (Jia et al., 2020) only uses the fusion output to compute losses. Compared with the above methods that focuses on classification performance, our method focuses on trustworthiness and interpretability besides classification performance.
this section cite: ['b1', 'b3', 'b14', 'b7', 'b26', 'b12']

Section: The Proposed Method
In this study, we propose a method that integrates both timedomain and time-frequency (T-F) domain representations of biosignals to establish a robust multi-view learning framework. We design two subnetworks to independently process the two types of input signals and adopt late fusion at the decision level, where the classification outputs of the two subnetworks are combined to produce the final prediction. The overall architecture of the model is illustrated in Fig. 1.
this section cite: []

Section: The Interpretation of Feature Level
Filter parametrization has become a frequently-used technique for designing interpretability deep neural networks (Xie et al., 2023;Zheng et al., 2024). The Gabor function is often used as filters in signal processing tasks (Chang & Morgan, 2014). To explicitly examine the features that contribute to decision-making, we follow the strategy (Niknazar & Mednick, 2024) to parameterize Gabor function, and use them to fit specific signal patterns that align with expert knowledge such as slow oscillations (∼1 Hz), alpha (8-13 Hz), theta (3-7 Hz), spindles (15-18 Hz bursts of activity in a spindle shape), K-complexes (large biphasic waves). Specifically, some Gabor functions are first embedded into the first convolutional layer as its kernels K G , and then a convolution operation is performed between the input signal X t and K G . The outputs of the convolution operation reflect the correlation between both of them. At the end of the training, the kernel K G is optimized to be pattern which facilitates decision-making. The process can be mathematically expressed as follows:
K G (T ) = e -( T -u σ ) 2 cos (2πf T ) ,(1)
Output = K G * X t ,(2)
where u is center of the kernel function (temporal or spatial location), σ is controls the width of the Gaussian envelope, f is frequency of the oscillations in the cosine part of the kernel, T is the time variable used in evaluating the kernel (-1s ≤ T ≤ 1s), and * is convolution operator.
Feature mapping F 1 and Classifier CL 1 . In order to transform a raw signal X t into a high-level feature x t , the feature mapping F 1 : X t → x t is realized by convolutional neural networks (CNNs). The output of the Gabor convolutional layer (GCL) represents the magnitude of the specific waveforms (Gabor kernels K G ) across the time. Then the resulting Gabor features are used as input to the one-dimensional CNNs to further extract discriminative features. Then flatten and randomly discard some of the processed features to form epoch-wise (30s) feature vector
x t =F 1 (K G * X t ) ∈ R 3072 .
In the end, through three fully connected layers, we get the probability distribution evidence e 1 = CL 1 (x t ) ∈ R 5 in the time domain.
Feature mapping F 2 and Classifier CL 2 . Simultaneously, the EEG raw signals X t in the time domain is transformed into the time-frequency (T-F) domain using the Short-Time Fourier Transform (STFT), yielding time-frequency representation X tf ∈ R (2×T ×F ) where F is frequency bins and T is time steps. Same as F 1 , the feature mapping F 2 : X tf → x tf is also realized by convolutional neural networks (CNNs). The processed frequency domain features are reshaped into one-dimensional feature vector by adaptive average pooling. Then the vector x tf = F 2 (X tf ) ∈ R 512 is passed to three fully connected layers, we get the probability distribution evidence e 2 = CL 2 (x tf ) ∈ R 5 in the time-frequency domain. Details on the network architecture are provided in the Appendix A.3.
this section cite: ['b38', 'b45', 'b2', 'b23']

Section: The Interpretation of Decision Level
In the previous process, we learn view-specific evidence by F 1 , CL 1 , F 2 and CL 2 , which could be termed as the amount of support the classification collected from data. Then the view-specific distributions of the class probabilities are modeled by Dirichlet distribution, parameterized with view-specific evidence. From the distributions, we can construct mass consisting of the belief quality of each category and the overall uncertainty. We also combine a conflicting mass aggregation strategy based on trusted fusion to reduce decision conflicts caused by view-specific F 1 and F 2 .
this section cite: []

Section: View-Specific Evidencial Deep Learning.
In decision layer, it is essential to ensure accurate and trustworthy predictions. Traditional methods like softmax layers often overestimate confidence, particularly in incorrect predictions (Wang et al., 2021). Due to their reliance on single-point probability estimates, this limits their ability to capture true model confidence and risk. To overcome the limitation, evidential deep learning (EDL) which is based on evidence theory under the framework of subjective logic (SL) has been introduced (Sensoy et al., 2018). Evidence e here refers to the information extracted from the input data that supports the classification decision, and it is used to derive a belief mass b i to each class label and an overall uncertainty mass u to the whole frame based on the evidence theory. For the vth view, then the belief mass b v j and the uncertainty
u v are computed as: b v j = e v j S v , u v = K S v , where S v = K j=1 (e v j + 1) = K j=1 α v j is the Dirichlet strength.
The core of this framework is that the more evidence there is, the higher the quality of belief in a category, and the more confident the model is in predicting that category. Conversely, when there is less evidence, the overall uncertainty increases. The mean of the corresponding Dirichlet distribution pv for the class probability pv j is computed as
pv j = α v j S v . In subjective opinion, the uncertainty mass is defined as u v = K S v = K K j=1 (e v j +1
) , which implies that u depends solely on the aggregate sum of evidences e. So, it is unsensitive to distribution of evidences e. The problem, which is called as Evidence Distribution-unaware Problem, can be illustrated using the below example. Example 3.2. Given an input x, we feed it into a network to obtain evidence e normal =[4,1,1,1,0]. From this, we compute α normal =[5,2,2,2,1] and uncertainty u normal =5/12. Now, if we add noise to x, the evidence e noisy =[2,2,2,1,0], leading to α noisy =[3,3,3,2,1] and u noisy = 5/12. Normally, as the distribution becomes more concentrated, the uncertainty should increase. Interestingly, after adding noise, the uncertainty u remains constant, which is counterintuitive and clearly unreasonable. We present this problem in Fig. 2, the complete presentation is in Fig. concentration of e. A larger d indicates a more concentrated evidence distribution, which corresponds to greater uncertainty. And then distribution-aware subjective opinion is defined as DM = [b 1 , b 2 , • • • , b K , d, u], we redefine the calculation of b v j and u v as follows:
b k = e k S , u = Kd S , d = (1 + Gini(e)) 2(3)
where
S = K j=1 (e j + d), Gini(e) = 1 - K k=1 p 2 k is Gini coefficient where p k = e k K j=1 ej is probability of class k.
In contrast to conventional subjective opinion frameworks, our approach determines the uncertainty measure u through a dual consideration of both the cumulative evidence sum and its distribution characteristics, represented by e. This enhanced formulation is formally characterized by the theoretical analysis in Subsection 3.3, which demonstrates its superior sensitivity in uncertainty quantification compared to existing methods.
this section cite: ['b32', 'b29']

Section: Evidential Multi-View Fusion via Distributed Mass Aggregation. After obtaining V independent sets of probability masses assignments {DM
v } V 1 , where DM v = b v j K j=1 , d v , u v under each view, we next need to com- bine them to obtain a joint mass DM = {b j } K j=1 , d, u .
Misalignment of multi-view data in feature mapping can cause conflicts. We would diminish their impact in the fusion stage.
The joint mass DM = {b j } K j=1 , d, u is calculated from the two sets of masses DM 1 = b 1 j K j=1 , d 1 , u 1 and DM 2 = b 2 j K j=1 , d 2 , u 2 in the following manner: DM 1♢2 = DM 1 ♢DM 2 = (b 1♢2 , u 1♢2 , d 1♢2 ), (4) b 1♢2 j = b 1 j u 2 + b 2 j u 1 u 1 + u 2 , u 1♢2 = 2u 1 u 2 u 1 + u 2 , d 1♢2 = 2d 1 d 2 d 1 + d 2 . (5
)
The averaging belief fusion can be computed simply by
d 2 e 1 k +d 1 e 2 k d 1 +d 2
in Appendix A.2. We can fusion the final joint mass DM from different views with the following rule:
DM = DM 1 ♢DM 2 ♢ • • • ♢DM v .(6)
Based on the above combination rule, we can obtain the estimated multi-view joint evidence e and the corresponding parameters of joint Dirichlet distribution α to produce the final probability of each class and the overall uncertainty.
Loss Function. For instance
{X v i } V 1 , e v i = CL v (F v (X v i ))
represent the evidence vector predicted by the network for the classification. To ensure that the network outputs nonnegative values, we need to replace the softmax layer of the traditional neural network based classifier with the activation function layer (ReLU). Different from the typical crossentropy loss used in traditional neural networks below:
L ce = - K j=1 y ij log p ij ,(7)
where p ij is the predicted probability of the ith sample for class j. For our model, we can get the parameter α i of the Dirichlet distribution by α i = e i + d. Based on Eq. ( 7), we have the adjusted cross-entropy loss using evidence-based approach:
L ace (α i ) =   K j=1 -y ij log p ij   1 B(α i ) K j=1 p αij -1 ij dp i = K j=1 y ij (ψ(S i ) -ψ(α ij )) ,(13)
where ψ(•) is the digamma function. The above loss function does not guarantee that the evidence generated by the incorrect labels is lower. To address this issue, we can introduce an additional term in the loss function, namely the Kullback-Leibler (KL) divergence:
L KL (α i ) = KL [D(p i | αi ) ∥ D(p i |1)](8)
= log   Γ K j=1 αij Γ(K) K j=1 Γ(α ij )   + K j=1 (α ij -1)   ψ(α ij ) -ψ   K j=1 αij     ,
where D(p i |1) is the uniform Dirichlet distribution, αi = y i + (1 -y i ) ⊙ α i is the Dirichlet parameters after removal of the non-misleading evidence from predicted parameters α i for the i-th instance, and Γ(•) is the gamma function.
Therefore, given the Dirichlet distribution with parameter α i for the i-th instance, the loss is:
L acc (α i ) = L ace (α i ) + λ t L KL (α i ),(9)
where λ t = min(1.0, t/T ) ∈ [0, 1] is the annealing coefficient, t is the index of the current training epoch, and T is the annealing step. By gradually increasing the influence of KL divergence in loss, premature convergence of misclassified instances to uniform distribution can be avoided.
In order to ensure the consistency of results between different mass during training, minimizing the conflicts between mass was adopted. The consistency loss for the instance
{x v i } V
v=1 is calculated as (Xu et al., 2024):
L con1 = V m=1 V n̸ =m ( K j=1 |p m j -p n j | • (1 -u m ) • (1 -u n ) 2 • (V -1) ), L con2 = 1 V (V -1) V -1 m=1 V n=m+1 e m • e n ∥e m ∥∥e n ∥ , L con = ζL con1 + ηL con2 .(10)
To sum up, the overall loss function for a specific instance
{X v i } V
v=1 can be calculated as:
L = L acc (α i ) + β V v=1 L acc (α v i ) + γL con .(11)
this section cite: ['b39']

Section: Theoretical Analysis
To demonstrate the superiority of the distribution-aware subjective opinion framework, we conduct a comprehensive theoretical analysis. This examination reveals several key advantages: (1) enhanced modeling capability for uncertainty quantification through explicit distribution consideration, (2) the relation between distribution-aware subjective opinion aggregation and evidence aggregation, and (3) aggregation properties. The theoretical framework establishes a rigorous mathematical foundation that not only justifies its practical effectiveness but also provides insights into its relationship with conventional subjective logic approaches. Proposition 3.4. Given two evidences e
1 = [e 1 1 , e 1 2 , • • • , e 1 K ] and e 2 = [e 2 1 , e 2 2 , • • • , e 2 K ]. If K j e 1 j = K j e 2 j and d(e 1 ) ≤ d(e 2 ), then u 1 ≤ u 2 .
Proposition 3.4 establishes that our modified opinion framework properly captures the uncertainty quantification of evidence through its dispersion characteristics. The complete mathematical proof is provided in Appendix A.1. Proposition 3.5. The distribution-aware subjective opinion aggregation operation DM 1♢2 = DM 1 ♢DM 2 is mathematically equivalent to the weighted evidence pooling:
e 1♢2 = d 2 e 1 + d 1 e 2 d 1 + d 2 .(12)
Proposition 3.5 demonstrates that the proposed distributionaware aggregation mechanism can be effectively implemented through a dispersion-weighted evidence pooling scheme. The detailed proof is available in Appendix A.2. Proposition 3.6.
Let DM 1 = [b 1 1 , b 1 2 , • • • , b 1 K , d 1 , u 1 ] and DM 2 = [b 2 1 , b 2 2 , • • • , b 2 K , d 2 , u 2
] represent distributionaware subjective opinions from two distinct views, with u 1 < u 2 . The aggregation process exhibits the following properties:
• When DM 1 is aggregated into DM 2 , the resulting uncertainty mass decreases: u 2 new < u 2 ;
• When DM 2 is aggregated into DM 1 , the resulting uncertainty mass increases: Proposition 3.6 reveals that the proposed aggregation method naturally accounts for potential conflicts between different opinions through its uncertainty-aware fusion mechanism. Based on Eq. 5, its proof is obvious.
u 1 new > u 1 . W N1
4. Experiments
4.1. Experimental Setups Datasets. In this experiments, we use three public datasets including Sleep-EDF 20, Sleep-EDF 78 and Sleep Heart Health Study (SHHS) as shown in Appendix A.4. For each dataset, we use a single EEG channel for various models in our experiments.
this section cite: []

Section: Compared Methods.
We compared our model with the several representative methods on three datasets including DeepSleepNet (Supratak et al., 2017), ARNN+SVM (Phan et al., 2018b), SleepEEGNet (Mousavi et al., 2019), ResNetLSTM (Sun et al., 2018), MultiTaskCNN (Phan et al., 2018a), DFSC (Liu et al., 2018), ResAtten (Qu et al., 2020), AttnSleep (Eldele et al., 2021) and MISC (Niknazar & Mednick, 2024). For a detailed description of these methods, please refer to Appendix A.5.
this section cite: ['b31', 'b22', 'b30', 'b19', 'b27', 'b4', 'b23']

Section: Evaluation Metrics and Implementation Details.
To evaluate the performance of the proposed method of sleep stage scoring. We used accuracy (Acc), macro F1-score (MF1), and Cohen's kappa (Kappa). Among these measures, F1score showed the performance of the method with respect to each sleep stage separately. The experimental setups are detailed in Appendix A.6
this section cite: []

Section: Experimental Results
In the section, we conduct the following experiments to evaluate our model from three aspects: performance comparison, confusion matrix analysis and hypnogram visualization.
this section cite: []

Section: Performance Comparison.
To verify the superiority of the proposed sleep scoring system, we compare the three evaluation metrics of our model with the baseline on two (3) Compared to DFSC, our superior performance may be attributed to fusion method leverages preliminary feature interaction and post-fusion, ensuring deeper integration of complementary information.
this section cite: []

Section: Confusion Matrix Analysis.
The classification results of the confusion matrix are shown in Fig. 3. From the results, it can be observed that TMCEK has the best classification results on W and N3 stages. The classification performance of the N1 stage is the worst among all sleep stages. The reasons of the results may be that the N1 stage, as a transitional state between wakefulness and deeper sleep stages like N2, exhibits overlapping features with both. For example, EEG characteristics of N1, such as low-amplitude theta waves and occasional alpha waves, are also present in wakefulness and N2, leading to blurred boundaries. This feature ambiguity makes it difficult for traditional feature extraction methods and deep learning models to distinguish N1 from other stages, particularly W and N2. In addition, the N1 stage is typically underrepresented in sleep staging datasets: N1 accounts for approximately 5% of a night's sleep, much lower than other stages like N2 or REM. This imbalance biases classification models towards dominant stages, thereby degrading the performance for the N1 stage.
this section cite: []

Section: Hypnogram Visualization.
In Fig. 4 we present an original manually scored hypnogram and its corresponding estimated sleep hypnogram using the trained single-epoch and multi-epoch networks for one fold on the Sleep-EDF 20. Its score is approximately equal to the mean score across the entire dataset. From Fig. 4, it can be observed that there are many misclassified points in the single epoch network output. Using the multi-epoch network to model transition rules between epochs can eliminate partial misclassified points and increased the classification performance significantly.
this section cite: []

Section: Interpretation
To increase the system's interpretability and reliability, we introduce the Gabor kernel at the first convolution layer in time domain. In addition, we can know whether the decision is credible by uncertainty estimation. Next, we analyze these two aspects respectively.
this section cite: []

Section: INTERPRETABILITY AT FEATURE-LEVEL
In terms of interpretability, the trainable Gabor kernels in the first layer are used to learn meaningful waveform patterns that are directly associated with sleep stages. These kernels are adjusted during training to capture representative time-frequency structures within the EEG signals. The outputs of this Gabor layer reflect how prominently each learned waveform appears in the input, effectively highlighting characteristic features relevant to sleep staging. The calculation of the overall qualitative impact of each kernel waveform is given in Appendix A.7.
In 1Hz), deta waves(1 to 4Hz), theta waves(3 to 8Hz). Optimized Gabor kernels 8 and 17 is similar to SW and deta waves, kernels 5, 24 and 25 are fitted to theta waves.
Fig. 9 in Appendix A.8 displays the importance of the corresponding Gabor waveforms in the overall sleep staging process and different sleep stages of the single-epoch network. The results in this figure are compatible to the experts' knowledge and the sleep scoring manuals. For example, Gabor kernels 8 and 17 which represent SW and deta waves have highest impact in stage S2 and SWS. On the other hand, the results in Fig. 9 show that some Gabor kernels are not important because the training process could not optimize them or they had redundant information, and other optimized kernels produced enough information for decision making. To improve kernel optimization, our subsequent work considers to apply diversity regularization for explicitly penalizing similarity among kernels and encouraging each to capture distinct patterns.
this section cite: []

Section: UNCERTAINTY ESTIMATION AT DECISION LEVEL
To further evaluate the estimated uncertainty, we visualize the distribution of normal and noisy data on the Sleep-EDF 20 dataset in Fig. 11 in Appendix A.8. To construct noisy test sets, we introduce Gaussian noise with standard deviation σ = 10, 30, 50, 100 to of the test instances. The experimental results are presented in Fig. 6. The results reveal that, when the noise intensity is low (σ = 10), the distribution curve of the noisy instances closely aligns with that of the normal instances. However, as the noise intensity increases, the uncertainty of the noisy instances also increases. This finding indicates that the estimated uncertainty is cor- related with the quality of the instances, thereby validating the capability of our method in uncertainty estimation.
this section cite: []

Section: Robustness
The robustness of the model can be improved through trusted learning. In order to verify the robustness improvement brought by trusted learning, we select the first fold (the first subject is used as the test set, and the others are used as the training set) for robustness testing on Sleep-EDF 20. We compare the evaluation indicators of the model trained with trusted learning and the model without trusted learning by adding the Gaussian noise with different levels of standard deviations (δ) to time domain view on the test set. From Fig. 7, it can be observed that models using trusted learning are more robust than those without it, which highlights the importance of the reliability of decision results.
this section cite: []

Section: Comparison with Trusted Multi-view Methods
To validate the effectiveness of the proposed trustworthy multi-view learning method, we conducted experiments on several benchmark multi-view datasets, including the Hand- Written (HW) and Scene15, CUB and PIE datasets (details in Appendix A.4). We compare our methods with EDL (Sensoy et al., 2018), DCCAE (Wang et al., 2015), CALM (Zhou et al., 2023), ETMC (Han et al., 2023), RCML (Xu et al., 2024) and CCML (Liu et al., 2024). Among them, EDL, ETMC, RCML, and CCML are four widely trusted multi-view methods. Detailed descriptions of the compared methods and implementation specifics are provided in Appendix A.5 and A.6. From Table 2, one can get that TM-CEK achieves the best classification accuracy across all datasets. To further demonstrate the superiority of TMCEK, we select a sample from Scene15 and show its classification performance and uncertainty estimate before and after adding noise. The result is illustrated in Fig. 12. From Fig. 12, it can be observed that (1) RCML misclassifies the noisy sample, whereas our method correctly classifies it. (2) RCML shows a decrease in uncertainty after noise injection due to an overall increase in the amount of evidence. In contrast, TMCEK provides a more accurate uncertainty estimation. This improvement stems from our method's novel dual consideration of both the cumulative evidence and its distributional properties during uncertainty quantification.
this section cite: ['b29', 'b34', 'b46', 'b9', 'b39', 'b21']

Section: Conclusion
In this paper, we have presented a novel trusted multi-view classification framework constrained with expert knowledge to address critical challenges in trusted multi-view learning. TMCEK effectively integrates expert knowledge by embedding Gabor kernels into the feature extraction module, thereby achieving interpretability at the feature level. This design improves the transparency of the decision-making process, making it more understandable and trustworthy for clinicians and patients. We further improved the reliability of uncertainty estimation by introducing a novel approach that considers not only the quantity but also the distribution of evidence, enhancing trustworthiness in high-stakes medical applications. We theoretically proved that it can enable more precise uncertainty estimation. Furthermore, experimental results on multiple datasets validated the effectiveness of TMCEK, confirming its superior performance in terms of accuracy, interpretability and reliability for multiview classification tasks. In the future, we focus on the embedded strategies of the expert knowledge from other domain into TMCEK.
this section cite: []

Section: References
Ref_id:b0 Title: Gradient-Based Attribution Methods Year: ()
Ref_id:b1 Title: A deep learning architecture for temporal sleep stage classification using multivariate and multimodal time series Year: (2017)
Ref_id:b2 Title: Robust cnn-based speech recognition with gabor filter kernels Year: (2014)
Ref_id:b3 Title: Mixed neural network approach for temporal sleep stage classification Year: (2016)
Ref_id:b4 Title: An attention-based deep learning approach for sleep stage classification with single-channel eeg Year: (2021)
Ref_id:b5 Title: Cardiorespiratory sleep stage detection using conditional random fields Year: (2017)
Ref_id:b6 Title: Physiobank, physiotoolkit, and physionet Year: (2000)
Ref_id:b7 Title: Dreem open datasets: Multi-scored sleep datasets to compare human and automated sleep staging Year: (2019)
Ref_id:b8 Title: Dreem open datasets: Multi-scored sleep datasets to compare human and automated sleep staging Year: (2020)
Ref_id:b9 Title: Trusted multiview classification with dynamic evidential fusion Year: (2023)
Ref_id:b10 Title: The american academy of sleep medicine (aasm) manual for the scoring of sleep and associated events: Rules, terminology and technical specifications Year: (2007)
Ref_id:b11 Title: Sleep disturbances during the covid-19 pandemic: A systematic review, meta-analysis, and meta-regression Year: (2022)
Ref_id:b12 Title: Sleepprintnet: A multivariate multimodal neural network based on physiological time-series for automatic sleep staging Year: (2020)
Ref_id:b13 Title: Interpretation and fusion of hyper opinions in subjective logic Year: (2012)
Ref_id:b14 Title: Towards precision sleep medicine: Self-attention gan as an innovative data augmentation technique for developing personalized automatic sleep scoring classification Year: (2022)
Ref_id:b15 Title: Crafting papers on machine learning Year: (2000)
Ref_id:b16 Title: AF: An association-based fusion method for multi-modal classification Year: (2022)
Ref_id:b17 Title: DC-NAS: Divide-and-conquer neural architecture search for multimodal classification Year: (2024)
Ref_id:b18 Title: Trusted multi-view classification via evolutionary multi-view fusion Year: (2025)
Ref_id:b19 Title: Diffuse to fuse eeg spectra -intrinsic geometry of sleep dynamics for classification Year: (2018)
Ref_id:b20 Title: Safe multi-view deep classification Year: (2023)
Ref_id:b21 Title: Dynamic evidence decoupling for trusted multi-view learning Year: (2024)
Ref_id:b22 Title: Automated sleep stage scoring with sequence to sequence deep learning approach Year: (2019)
Ref_id:b23 Title: A multi-level interpretable sleep stage scoring system by infusing experts' knowledge into a deep network architecture Year: (2024)
Ref_id:b24 Title: Joint classification and prediction cnn framework for automatic sleep stage classification Year: (2018)
Ref_id:b25 Title: Automatic sleep stage classification using singlechannel eeg: Learning sequential features with attentionbased recurrent neural networks Year: (2018)
Ref_id:b26 Title: Xsleepnet: Multi-view sequential model for automatic sleep staging Year: (2020)
Ref_id:b27 Title: A residual based attention model for eeg based sleep staging Year: (2020)
Ref_id:b28 Title: The sleep heart health study: design, rationale, and methods Year: (1997)
Ref_id:b29 Title: Evidential deep learning to quantify classification uncertainty Year: (2018)
Ref_id:b30 Title: Deep convolutional network method for automatic sleep stage classification based on neurophysiological signals Year: (2018)
Ref_id:b31 Title: Deepsleepnet: A model for automatic sleep stage scoring based on raw single-channel eeg Year: (1998)
Ref_id:b32 Title: Rethinking calibration of deep neural networks: Do not be afraid of overconfidence Year: (2021)
Ref_id:b33 Title: Uncertainty-inspired open set learning for retinal anomaly identification Year: (2023)
Ref_id:b34 Title: On deep multi-view representation learning Year: (2015)
Ref_id:b35 Title: On-the-fly modulation for balanced multimodal learning Year: (2025)
Ref_id:b36 Title: Discriminative regression with adaptive graph diffusion Year: (2024)
Ref_id:b37 Title: Sleep and circadian rhythm disruption in psychiatric and neurodegenerative disease Year: (2010)
Ref_id:b38 Title: Fourier series expansion based filter parametrization for equivariant convolutions Year: (2023)
Ref_id:b39 Title: Reliable conflictive multi-view learning Year: (2024)
Ref_id:b40 Title: Prototype matching learning for incomplete multiview clustering Year: (2025)
Ref_id:b41 Title: Generalized latent multi-view subspace clustering Year: (2020)
Ref_id:b42 Title: Efficient multi-view unsupervised feature selection with adaptive structure learning and inference Year: (2024)
Ref_id:b43 Title: The national sleep research resource: towards a sleep data commons Year: (2018)
Ref_id:b44 Title: Provable dynamic fusion for low-quality multimodal data Year: (2023)
Ref_id:b45 Title: Centrosymmetric constrained convolutional neural networks Year: (2024)
Ref_id:b46 Title: Calm: An enhanced encoding and confidence evaluating framework for trustworthy multi-view learning Year: (2023)
Ref_id:b47 Title: Reliable multimodality eye disease screening via mixture of student's t distributions Year: (2023)
