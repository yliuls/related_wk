Title: Unbiased Prototype Consistency Learning for Multi-Modal and Multi-Task Object Re-Identification
Abstract: In object re-identification (ReID) task, both cross-modal and multi-modal retrieval methods have achieved notable progress. However, existing approaches are designed for specific modality and category (person or vehicle) retrieval task, lacking generalizability to others. Acquiring multiple task-specific models would result in wasteful allocation of both training and deployment resources. To address the practical requirements for unified retrieval, we introduce Multi-Modal and Multi-Task object ReID (M 3 T-ReID). The M 3 T-ReID task aims to utilize a unified model to simultaneously achieve retrieval tasks across different modalities and different categories. Specifically, to tackle the challenges of modality distibution divergence and category semantics discrepancy posed in M 3 T-ReID, we design a novel Unbiased Prototype Consistency Learning (UPCL) framework, which consists of two main modules: Unbiased Prototypes-guided Modality Enhancement (UPME) and Cluster Prototype Consistency Regularization (CPCR). UPME leverages modalityunbiased prototypes to simultaneously enhance cross-modal shared features and multi-modal fused features. Additionally, CPCR regulates discriminative semantics learning with category-consistent information through prototypes clustering. Under the collaborative operation of these two modules, our model can simultaneously learn robust cross-modal shared feature and multi-modal fused feature spaces, while also exhibiting strong category-discriminative capabilities. Extensive experiments on multi-modal datasets RGBNT201 and RGBNT100 demonstrates our UPCL framework showcasing exceptional performance for M 3 T-ReID. The code is available at https://github.com/ZhouZhongao/UPCL.

Section: Introduction
Object re-identification (ReID) [75,49,65,45,18,60,3,9,75,73,69,30,4] leverages computer vision techniques to identify specific objects (such as persons or vehicles) in videos and still images. ReID technology has been widely applied in intelligent video surveillance, public security, and other related fields. Traditional ReID predominantly focuses on single-modal scenario, where both the query and gallery consist of RGB images. However, RGB cameras are highly sensitive to illumination variations, making it difficult to accurately capture target information under lowlight or overexposed conditions. To address the above challenges, Near-Infrared (NI) and Thermal Infrared (TI) modalities have been introduced into ReID tasks, enabling robust imaging in challenging environments [72,15,29,55,17,71,6,63]. Depending on the retrieval scenarios, the existing ReID methods can be broadly categorized into cross-modal ReID [58,53,36,70,41,19,2] and multi-modal ReID [47,50,68,67]. Specifically, cross-modal ReID focuses on retrieval between two different modalities (e.g.,NI-RGB, TI-RGB), whereas multi-modal ReID utilizes RGB, NI, and TI fusion to achieve feature matching. We introduce modality-unbiased prototypes and cluster-derived category-consistent prototypes to enhance the model's comprehensive retrieval capability from both modality and category perspectives.
As illustrated in Figure 1, although existing cross-modal and multi-modal approaches have achieved remarkable results, they still suffer from two significant limitations: 1) Real-world surveillance environments present extreme complexity, where targets may appear in scenarios captured by either single-modality cameras or aligned multi-modal imaging systems. While cross-modal ReID primarily focuses on learning a shared cross-modal feature space, multi-modal ReID emphasizes the effective fusion of different modalities to obtain more robust fused features. The cross-modal and multi-modal approaches follow different optimization directions, thereby resulting in distribution divergence. Therefore, existing ReID models cannot simultaneously handle both cross-modal and multi-modal retrieval.
2) The retrieval tasks are confined to a specific category, necessitating separate model training for either person or vehicle ReID tasks. In practical scenarios such as criminal investigations, surveillance systems require the capability to simultaneously retrieve both suspects and vehicles. Due to the discrepancy in semantics between diverse categories, existing ReID methods lack the generalization capability to perform unified retrieval across categories. While training dedicated retrieval models for distinct modalities and categories may serve as a feasible solution, this approach inevitably leads to substantial redundancy in both training and deployment resources.
To meet the real-world demands for retrieval across diverse modalities and categories, we propose the Multi-Modal and Multi-Task object ReID (M 3 T-ReID). The M 3 T-ReID task aims to achieve a unified model for simultaneous retrieval across multiple modalities and diverse categories. However, achieving high-performance of M 3 T-ReID introduces several challenges. Firstly, since cross-modal and multi-modal retrieval models optimize fundamentally different objectives, this leads to challenge I : How to jointly learn both a robust cross-modal shared feature space and an effective multi-modal fusion feature space. Secondly, discriminative features vary significantly across object categories. For instance, person ReID primarily focuses on attributes like pose and clothing while vehicle ReID emphasizes vehicle type and color, which raises the challenge II : How to enable a model to simultaneously learn category-specific discriminative representations for heterogeneous objects.
To address the aforementioned chanllenges in M 3 T-ReID, we propose Unbiased Prototype Consistency Learning framework (UPCL) which comprises two key modules: Unbiased Prototypes-guided Modality Enhancement (UPME) and Cluster Prototype Consistency Regularization (CPCR). For challenge I, UPME enhances both cross-modal shared features and multi-modal fused features through modality-unbiased prototypes, thereby bridging the discrepancy across heterogeneous modalities and simultaneously improving robustness of the overall feature space. For challenge II, CPCR derives category-consistent features through prototypes clustering, thereby regulating the model to stably acquire category-specific discriminative semantics from diverse categories.
The main contributions of this paper can be summarized as follows:
• To address the practical demands for retrieval of diverse modalities and categories, we propose a novel Multi-Modal and Multi-Task object ReID (M 3 T-ReID).
• To address the challenges in M 3 T-ReID, we propose UPCL which comprises two main components: UPME and CPCR. UPME leverages modality-unbiased prototypes to simultaneously enhance cross-modal shared features and multi-modal fused features, and CPCR regulates the learning of discriminative semantics across diverse object categories through prototypes clustering.
• Extensive experiments on the public multi-modal ReID benchmarks RGBNT201 and RGBNT100
have verified the advantage of our methods, achieving significantly higher accuracy compared to existing counterparts in both cross-modal and multi-modal retrieval scenarios.
2 Related Work
this section cite: ['b76', 'b49', 'b66', 'b45', 'b17', 'b61', 'b2', 'b8', 'b76', 'b74', 'b70', 'b29', 'b3', 'b73', 'b14', 'b28', 'b55', 'b16', 'b72', 'b5', 'b64', 'b59', 'b53', 'b35', 'b71', 'b41', 'b18', 'b1', 'b47', 'b50', 'b69', 'b68']

Section: Cross-modal and Multi-modal Re-identification
Cross-modal re-identification [23,14,20,5,8,62,31,59,11] aims to retrieve target RGB images across heterogeneous modalities. The cross-modal retrieval capability of a model primarily depends on the robustness of itscross-modal shared feature space. Wu et al. [52] utilize a zero-padding onestream network with grayscale inputs to learn the shared feature between RGB andNI images. Ye et al. [61] introduce a Channel Augmentation (CA) mechanism that mitigates the modality gap by generating color-irrelevant person representations. Liu et al. [23] propose the Memory-Augmented Unidirectional Metric (MAUM) method to enhance the cross-modality correlation by utilizing two unidirectional metrics. Liang et al. [20] make an early attempt at unsupervised cross-modal ReID with a two-stage framework. Yang et al. [57] design an Augmented Dual-Contrastive Aggregation (ADCA) learning framework for Unsupervised Learning Visible-Infrared Person ReID.
Multi-modal re-identification [47,15] jointly leverages complementary information from multiple modalities to extract more robust fused features, thereby improving retrieval accuracy. Zheng et al. [72] propose PFNet which hierarchically fuses RGB,NI, and TI features to obtain more robust representations. Wang et al. [49] design a Cross-Modal Interacting Module to enhance modality-specific information during feature fusion. Wang et al. [50] utilize the relationship among heterogeneous modalities to fine-tune the network prior to inference, thereby improving generalization to unseen data. Zhang et al. [68] propose a general PromptMA framework, which employs learnable prompts to aggregate modalities and bridge the modality distribution gap. Wang et al. [46] introduce the token permutation to enhance inter-modal interaction and facilitate multi-spectral feature alignment. Zhang et al. [67] propose EDITOR framework, which obtains spatial-frequency masks to refine multimodal features. Wang et al. [48] adaptively balances decoupled features using a mixture-of-experts mechanism to produce more robust multi-modal representations.
Due to the divergent optimization objectives between cross-modal and multi-modal, a single model cannot be effectively applied to both retrieval scenarios simultaneously. Furthermore, significant semantic discrepancies exist among objects of different categories, models trained on one category cannot achieve generalized to retrieval of other categories. To address diverse retrieval requirements in real-world scenarios, we propose the UPCL framework, which trains a unified model capable of performing retrieval across multiple modalities and object categories.
this section cite: ['b22', 'b13', 'b19', 'b4', 'b7', 'b63', 'b30', 'b60', 'b10', 'b52', 'b62', 'b22', 'b19', 'b58', 'b47', 'b14', 'b73', 'b49', 'b50', 'b69', 'b46', 'b68', 'b48']

Section: Prototypes Learning
Prototype is calculated as the mean feature of the instances belonging to the same ID [38]. Due to its simplicity and scalability, prototype plays an indispensable role across various domains, such as few-shot learning [56,38,13,27,42] , unsupervised learning [54,7,16,53],incremental learning [74,76,66,43,37], and federated learning [21,12,26,22,28,25,40,64,34,35]. In object ReID research [24,57,58,53], prototypes and other feature-centric concepts have also demonstrated significant effectiveness. Luo et al. [24] introduce Center Loss to enhance the feature similarity of same-ID samples in the model. Yang et al. [57] assigns pseudo-labels to unannotated data through clustering and dynamically update the centers of samples corresponding to each pseudo-ID as supervisory signals for network optimization.
Current applications of prototypes or similar concepts primarily leverage their statistical properties at ID-level to enhance feature compactness within the same category, thereby improving network robustness. In this work, we innovatively utilize modality-unbiased prototypes from a modality- We dynamically update the multi-modal prototype memory for each ID during every training iteration. UPME aggregates modality-unbiased prototypes via Equation ( 9), leveraging their identity-consistent information to strengthen both cross-modal and multi-modal representation learning. CPCR further utilizes category-consistent prototypes clustered from modality prototypes, exploiting their category-consistent semantics to regularize the model and achieve more discriminative category-wise decision boundaries.
consistency perspective and employ clustering strategies to obtain category-level (rather than identitylevel) discriminative features. This approach significantly enhances the unified ReID model's capability to perform robust retrieval across multiple categories and modalities.
this section cite: ['b38', 'b57', 'b38', 'b12', 'b26', 'b42', 'b54', 'b6', 'b15', 'b53', 'b75', 'b77', 'b67', 'b43', 'b37', 'b20', 'b11', 'b25', 'b21', 'b27', 'b24', 'b40', 'b65', 'b33', 'b34', 'b23', 'b58', 'b59', 'b53', 'b23', 'b58']

Section: Method
In this section, we present the Unbiased Prototype Consistency Learning framework (UPCL) which consists of two main modules. The Unbiased Prototypes-guided Modality Enhancement (UPME) module leverages modality-unbiased prototypes to simultaneously enhance both cross-modal shared features and multi-modal fused features, thereby improving the model's performance across different retrieval modes. The Cluster Prototype Consistency Regularization (CPCR) module utilizes modalityunbiased prototypes via clustering to derive category-consistent prototypes, which are then utilized to regulate the model's discriminative semantic learning process for different categories. The overview of UPCL is illustrated in Figure 2 and the details are discussed in the following subsections.
this section cite: []

Section: Overall Architecture
Our method utilizes the pretrained CLIP [32] model as the visual encoder which is shared with RGB, NI and TI modalities. Specifically, for the i-th multi-modal instance V i = {V rgb i , V ni i , V ti i }, the images of three modalities are cropped into equal-sized patches and then mapped to embedding vectors with fixed dimensions. Next, we feed these embedding vectors into the visual encoder to obtain their corresponding class token f m i ∈ R D and patch tokens p m i ∈ R Np×D , where m ∈ {rgb, ni, ti}, N p denotes the number of patch tokens and D is the embedding dimension.
Consistent with existing multi-modal methods [72,49,50,46], we employ the label smoothing cross-entropy loss L ce [39] and triplet loss L tri [10] to supervise the learning of visual encoder:
L g = L ce + L tri .(1)
To further encourage cross-modal feature alignment, we introduce an additional cross-modal alignment loss function:
L (m→n) (i) = -log exp (⟨f m i , f n i ⟩ /τ ) B j=1 exp f m i , f n j /τ ,(2)
L (n→m) (i) = -log exp (⟨f n i , f m i ⟩ /τ ) B j=1 exp f n i , f m j /τ ,(3)
where ⟨•, •⟩ represents the cosine similarity function. f m i and f n i denote the features of i-th instance with different modalities m, n ∈ {rgb, ni, ti}. B is the batch size, and τ is the temperature parameter. Then the cross-modal loss function between modalities m and n can be formulated as:
L m↔n = 1 2B B i=1 [L (m→n) (i) + L (n→m) (i)].(4)
Building upon this, we derive the cross-modal loss function L cross :
L cross = L rgb↔ni + L rgb↔ti + L ni↔ti .(5)
Finally, we obtain the base objective L b of the framework :
L b = L g + L cross . (6
)
this section cite: ['b31', 'b73', 'b49', 'b50', 'b46', 'b39', 'b9']

Section: Unbiased Prototypes-guided Modality Enhancement
To enable the model to achieve high performance in both cross-modal and multi-modal retrieval, it is essential to design optimization strategies that maintain directional consistency between these two paradigms. Inspired by the successful application of prototypes in other domains, we leverage the identity-consistent information encapsulated in prototypes to guide the feature learning. We compare two commonly used prototypes (global prototypes and modality-specific prototypes) in multi-modal learning alongside our newly proposed unbiased prototype in Figure 3.
this section cite: []

Section: Global Prototypes.
The features of all samples across three modalities under the same ID are fused to obtain the global prototype:
P global c = 1 ∥I(c)∥ i∈I(c) L f used (f rgb i , f ni i , f ti i ),(7)
where I(c) denotes the indices of all instances with identity c. L f used is an MLP designed to fuse the features from the three modalities, producing a global feature in R D .
this section cite: []

Section: Modality-Specific Prototypes.
We compute modality-specific prototypes for identity c :
P m c = 1 ∥I(c)∥ i∈I(c) f m i , m ∈ {rgb, ni, ti}.(8)
Then this yields a set of prototypes P c = {P rgb c , P ni c , P ti c } for three modalities . Modality-Unbiased Prototypes. Global prototypes tend to incorporate semantic features biased toward the dominant modality, thereby deviating from identity-consistent semantics. Conversely, modality-specific prototypes primarily focus on intra-modal identity information, which inherently carries modality-specific bias relative to identity consistency. Therefore, in order to optimize the features towards identity-consistent semantics, we comprehensively derive a modality-unbiased prototype from P c = {P rgb c , P ni c , P ti c }:
U c = (P rgb c + P ni c + P ti c )/3.(9)
As illustrated in Figure 3, compared with global prototypes and modality-specific prototypes, the modality-unbiased prototypes mitigate both inter-modal and intra-modal distribution discrepancies, thereby thereby capturing more robust identity representations.
Under the guidance of modality-unbiased prototypes, we enhance features of modality m by:
L m U P M E = - 1 B B i=1 log exp (⟨f m i , U ci ⟩ /τ ) B j=1 exp f m i , U cj /τ ,(10)
where c i represents the identity of the i-th sample. Finally, we formulate the Unbiased Prototypesguided Modality Enhancement loss through summation:
L U P M E = L rgb U P M E + L ni U P M E + L ti U P M E .(11)
By enhancing semantic consistency across multi-modal features, the discriminative capability for identities in any modality is improved, thereby boosting the cross-modal retrieval performance. Simultaneously, the approach reduces inter-modal divergence and improves fusion efficiency, yielding more robust multi-modal representations.
this section cite: []

Section: Cluster Prototype Consistency Regularization
Current ReID methods typically train dedicated models for specific object categories (e.g., persons or vehicles). However, since discriminative semantic features vary significantly across different object categories, models trained on specific category lack generalization capability. To develop a unified model capable of retrieving diverse object categories, we need to enable the model to accurately capture category-consistent semantic information. As detailed in Section 3.2, the modality-unbiased prototypes inherently encapsulate identity-consistent semantic information that is strongly correlated with sample categories. This intrinsic correlation motivates us to explore a novel approach for extracting category-consistent semantics directly from these prototypes.
Assuming that there are N p and N v identities belonging to persons and vehicles, respectively, we utilize
U p = {U c } Np c=1 and U v = {U c } Np+Nv c=Np+1
to denote the sets of modality-unbiased prototypes. To obtain category-consistent information, we aim to fully integrate prototypes from all identities within a specific category. While the identities of any two prototypes in U p or U h are distinct, they may share highly similar semantic characteristics (e.g. males wearing short sleeves). Such semantic similarity naturally forms clusters among these identities, where the dominant clusters contribute more representative information within U p or U v .
To better capture discriminative semantic features of specific categories, we propose to utilize cluster-level statistics instead of identity-level information. The acquisition of reliable cluster-level statistics fundamentally depends on achieving proper clustering of U p and U v . Unlike conventional clustering algorithms K-means [1] and HAC [51], FINCH [33] operates in a completely parameterfree manner, automatically determining the optimal number of clusters based on the inherent similarity relationships among all prototypes. Therefore, we employ FINCH to cluster P u,p and P u,v , obtaining the sets of multiple clustered prototypes C p and C v :
U p = {U c } Np c=1 Cluster -----→ C p = {C p l } Lp l=1 ,(12)
U v = {U c } Nv c=Np+1 Cluster -----→ C v = {C v l } Lv l=1 ,(13)
where L p and L v denote the number of elements in C p and C v . The l-th cluster prototypes for person and vehicle are denoted as C u l and C u l , which are calculated by averaging all modality-unbiased prototypes in l-th cluster. Then we obtain the category-consistent prototype through:
Cate p = 1 L p Lp l=1 C p l ,(14)
Cate v = 1 L v Lv l=1 C v l .(15)
Then, we introduce the CPCR loss function with Cate p and Cate v for specific category:
L p CP CR = 1 ∥I p ∥ m B p i∈I p log exp (⟨f m i , Cate p ⟩ /τ ) exp (⟨f m i , Cate p ⟩ /τ ) + exp (⟨f m i , Cate v ⟩ /τ ) , (16
)
L v CP CR = 1 ∥I v ∥ m B v i∈I v log exp (⟨f m i , Cate v ⟩ /τ ) exp (⟨f m i , Cate p ⟩ /τ ) + exp (⟨f m i , Cate v ⟩ /τ ) ,(17)
where I p and I v denote the index sets of all person-class and vehicle-class instances within a batch, respectively. It is natural to derive the total CPCR loss:
L CP CR = L p CP CR + L v CP CR .(18)
Under the regularization of category-consistent prototypes, the model effectively learns discriminative semantic features that represent object categories, thereby achieving robust performance across diverse retrieval tasks involving different object categories.
Finally, we integrate all components to formulate a comprehensive optimization objective:
L total = L b + αL U P M E + βL CP CR , (19
)
where α and β are the the hyper-parameters to balance the contributions of each loss function.
this section cite: ['b0', 'b51', 'b32']

Section: Experiments

this section cite: []

Section: Experimental Settings Datasets and Evaluation Protocols.
To evaluate the performance of our UPCL, we combined RGBNT201 [72] and RGBNT100 [15] to construct a multi-modal dataset with diverse object categories. Specifically, RGBNT201 is the first multi-modal person ReID dataset, where each pedestrian ID contains RGB, NI and TI modalities. RGBNT100 contains the same modalities as RGBNT201, but collects vehicle images instead. We evaluate the performance of our proposed UPCL with Rank-k matching accuracy, mean Average Precision (mAP) which are the commonly utilized metrics in object ReID tasks. The evaluation protocol encompasses six cross-modal testing scenarios (R → N, N → R, R → T, T → R, N → T, and T → N) along with one multi-modal testing configuration (RNT → RNT).
Implementation Details. The implementation platform is Pytorch with a NVIDIA 3090 GPU. We utilize the pre-trained CLIP as the visual encoder. Images of all modalities are resized to 256×128.
For data augmentation, we apply random horizontal flipping, cropping and erasing. The batch size is set to 64, sampling 8 images per identity. The training process is conducted with the Adam optimizer for 50 epochs, and the initial learning rate is set to 3.5e-4. We select 0.03 as the temperature parameter τ . The hyper-parameters α and β are set as 2.0 and 0.5 respectively. During the testing phase, cross-modal retrieval directly computes similarity using features from two modalities, while multi-modal retrieval concatenates features from three modalities for matching.
this section cite: ['b73', 'b14']

Section: Comparison with State-of-the-art Methods
We present a comprehensive comparison of UPCL with state-of-the-art methods as outlined in Table 1. To thoroughly evaluate the model's holistic performance on six cross-modal and one multimodal testing scenarios, we employ the harmonic mean of task-specific metrics as the aggregated performance measure, given that the harmonic mean places greater emphasis on smaller values compared to the arithmetic mean.
As demonstrated in Table 1, on the person-category dataset (RGBNT201) , UPCL significantly outperforms all competing methods across all six cross-modal scenarios, which strongly validates its cross-modal matching capability. For multi-modal retrieval results, compared with PromptMA which contains module specifically designed for multi-modal fusion matching, our approach trails by merely 0.8 percentage points in mAP. Regarding the arithmetic mean of all seven metrics, UPCL achieves substantial superiority over other methods, surpassing the second-best approach by 5.43 and 9.02 percentage points on these two metrics, respectively. For the vehicle-category dataset (RGBNT100) , our UPCL demonstrates consistent superiority across every scenario. Across both category-specific datasets (person and vehicle) under all seven evaluation protocols, our method achieves remarkable performance, which conclusively demonstrates its effectiveness for the M 3 T-ReID task.
this section cite: []

Section: Ablation Studies
To thoroughly validate the effectiveness of our proposed UPME and CPCR modules, we conducted extensive ablation experiments on both RGBNT201 and RGBNT100 datasets. The experimental protocol involves incrementally integrating our proposed modules into the baseline model without introducing any extra modifications, enabling direct performance comparison through controlled ablation studies. As illustrated in Table 2, all reported metrics are presented as the harmonic mean of performance across seven evaluation scenarios.
this section cite: []

Section: Effect of UPME.
By leveraging the identity-consistent information captured in modality-unbiased prototypes, the UPME module enhances feature representations across different modalities, thereby boosting cross-modal and multi-modal representation capabilities. The experimental results demonstrate that our UPME module achieves mAP improvements of 3.20% and 2.19% on RGBNT201 and RGBNT100 respectively, compared to the baseline. When aggregated across both datasets, the module delivers average gains of 2.41% mAP and 3.37% rank-1 accuracy.
this section cite: []

Section: Effect of CPCR.
The CPCR module clusters modality-unbiased prototypes from UPME to derive category-consistent prototypes, then exploits their category-discriminative semantics to guide model optimization. After incorporating CPCR, the model achieves a 4.10% mAP and 5.99% rank-1 improvement on RGBNT201, while attaining 0.73% mAP and 2.91% rank-1 gains on RGBNT100.  Visualization. To intuitively demonstrate the effectiveness of UPME and CPCR, we visualize the t-SNE [44] feature distribution of several identities in Figure 4. Specifically, the three subplots on the left visualize the embedding space of multi-modal fused features. Under the effects of UPME and CPCR, the fused features of the same ID progressively converge, while the discriminability between different identities becomes increasingly pronounced. The cross-modal features of identical IDs in the right figure exhibit consistent variation trends. Therefore, the t-SNE visualization clearly indicates that integrating UPME and CPCR enhances the intra-identity consistency and inter-identity discriminability of both fused and cross-modal features.
Table 3: Comparison between specifc and unified model. Each model is trained on a combined dataset consisting of RGBNT201 and RGBNT100, and evaluated separately. mAP(%) is reported. RGBNT201 RGBNT100 Methods Specific Unified Specific Unified HTT [50] 69.0 9.16 75.7 32.21 TOP-ReID [46] 72.3 63.74 81.2 71.47 PromptMA [68] 78.4 65.71 85.3 71.07 EDITOR [67] 66.5 56.03 79.8 74.25 DeMo [48] 79.0 64.35 86.2 79.12
Effects of Diverse Categories . As discussed earlier, different categories of retrieval targets contain distinct discriminative semantics, making it challenging for a model to maintain high performance across retrieval tasks involving diverse categories. To validate this claim, Table 3 presents a comparison between category-specific model and a unified model.
The category-specific model is trained on a single-category dataset, either RGBNT201 or RGBNT100, while the unified model is trained on a multi-category dataset formed by combining RGBNT201 and RGBNT100. It can be clearly observed that, for the same method, the unified model exhibits a significant performance drop compared to its corresponding category-specific model. This suggests that mixing multiple categories in training introduces substantial interference, hindering the model's ability to simultaneously learn discriminative semantics for all categories. These findings strongly support our hypothesis that multi-category training can negatively impact model optimization.
this section cite: ['b44']

Section: Cross-domain Retrieval Evaluation
In conventional Re-ID evaluations, both training and testing sets are typically collected under the same scene conditions. As a result, the learned feature space may be constrained to a specific environment, making the evaluation results insufficient to reflect the generalization capability required in real-world applications. To assess the cross-domain generalization ability of our model, we train it on a combined dataset of RGBNT201 and RGBNT100, and evaluate it on the unseen MSVR310 dataset.
The comparison results are presented in Table 4. All models show significantly lower detection accuracy on the MSVR310 dataset compared to RGBNT201 and RGBNT100, indicating that crossdomain generalization remains highly challenging for existing Re-ID models. In particular, methods  31 6.43 3.87 3.21 2.23 0.85 1.84 1.52 2.30 1.02 2.54 2.71 14.91 25.55 2.94 1.89  PromptMA [68] 11.92 14.72 10.71 13.37 2.43 1.02 2.64 1.52 2.82 2.53 3.70 3.21 23.14 31.98 4.28 2.78  EDITOR [67] 2.16 1.18 1.53 0.52 1.59 0.51 1.49 0.63 1.45 0.85 1.35 1.02 17.05 30.96 1.70 0.82 DeMo [48] 2.12 0.34 1.73 0.17 1.87 1.35 1.77 1.02 1.66 1.02 1.76 0.51 15.89 26.90 2.07 0.52 UPCL (Ours) 13. 89 20.30 14.09 20.14 9.43 10.15 9.03 9.14 9.57 11.68 9.35 11.17 24.60 38.07 11.44 13.77such as HTT, TOP-ReID, EDITOR, and DeMo exhibit severe performance degradation under both cross-modal and multi-modal testing scenarios on the unseen MSVR310 dataset. In contrast, the UPCL and PromptMA methods achieve considerably better results, with UPCL consistently outperforming PromptMA across various detection settings. These experimental results provide strong evidence that our model possesses excellent cross-domain generalization capability. The superior cross-domain retrieval performance of UPCL further validates our method from another perspective-not only does it enhance the robustness of the fused feature space across modalities, but it also strengthens the model's ability to discriminate semantic features across different categories.
this section cite: []

Section: Limitations
Although our method has demonstrated promising performance on existing multi-modal ReID datasets, it still exhibits several notable limitations. The current multi-modal ReID datasets are relatively scarce, mainly restricted to the pedestrian and vehicle domains. Considering the practical demands of real-world applications, it is crucial to explore how incorporating a broader range of categories affects retrieval performance. In addition, our unified model may slightly underperform task-specific models in a very limited number of test scenarios. Although this is reasonable, this also indicates that our approach still has potential for further enhancement. Therefore, our future work will focus on extending UPCL to more generalized multi-modal and multi-task retrieval scenarios.
this section cite: []

Section: Conclusion
In this paper, we introduce a novel M 3 T-ReID task to address the practical demands of retrieval across diverse modalities and categories. To tackle the challenges in M 3 T-ReID, we propose the Unbiased Prototype Consistency Learning framework (UPCL) which consists of two main modules UPME and CPCR. Specifically, UPME mitigates the divergence between cross-modal shared spaces and multimodal fusion distributions by leveraging identity-consistent information from modality-unbiased prototypes, thereby enhancing both cross-modal and multi-modal representations. Meanwhile, CPCR reduces semantic discrepancies across categories by clustering modality-unbiased prototypes to obtain category-consistent prototypes with discriminative semantics. Extensive experiments on multiple datasets validate the superiority and effectiveness of our method, demonstrating its robustness and generalization ability in diverse retrieval scenarios.
this section cite: []

Section: References
Ref_id:b0 Title: Gated multimodal units for information fusion Year: (2017)
Ref_id:b1 Title: Towards modality-agnostic person re-identification with descriptive query Year: (2023)
Ref_id:b2 Title: Meta batchinstance normalization for generalizable person re-identification Year: (2021)
Ref_id:b3 Title: Learning continual compatible representation for re-indexing free lifelong person re-identification Year: (2024)
Ref_id:b4 Title: Cross-modality person re-identification with generative adversarial training Year: (2018)
Ref_id:b5 Title: Person re-identification method based on color attack and joint defence Year: (2022)
Ref_id:b6 Title: Hcsc: Hierarchical contrastive selective coding Year: (2022)
Ref_id:b7 Title: Cross-modality person re-identification via modality confusion and center aggregation Year: (2021)
Ref_id:b8 Title: Transreid: Transformerbased object re-identification Year: (2021)
Ref_id:b9 Title: defense of the triplet loss for person re-identification Year: (2017)
Ref_id:b10 Title: Empowering visible-infrared person re-identification with large foundation models Year: (2024)
Ref_id:b11 Title: Rethinking federated learning with domain shift: A prototype view Year: (2023)
Ref_id:b12 Title: Prototypical priors: From improving classification to zero-shot learning Year: (2015)
Ref_id:b13 Title: Cross-modal implicit relation reasoning and aligning for text-to-image person retrieval Year: (2023)
Ref_id:b14 Title: Multi-spectral vehicle re-identification: A challenge Year: (2020)
Ref_id:b15 Title: Prototypical contrastive learning of unsupervised representations Year: (2020)
Ref_id:b16 Title: Exemplar-free lifelong person reidentification via prompt-guided adaptive knowledge consolidation Year: (2024)
Ref_id:b17 Title: Dc-former: Diverse and compact transformer for person re-identification Year: (2023)
Ref_id:b18 Title: Robust duality learning for unsupervised visible-infrared person re-identification Year: (2025)
Ref_id:b19 Title: Homogeneous-toheterogeneous: Unsupervised learning for rgb-infrared person re-identification Year: (2021)
Ref_id:b20 Title: Hyperfed: hyperbolic prototypes exploration with consistent aggregation for non-iid data in federated learning Year: (2023)
Ref_id:b21 Title: Prototypebased layered federated cross-modal hashing Year: (2023)
Ref_id:b22 Title: Learning memoryaugmented unidirectional metrics for cross-modality person re-identification Year: (2022)
Ref_id:b23 Title: Bag of tricks and a strong baseline for deep person re-identification Year: (2019)
Ref_id:b24 Title: No fear of heterogeneity: Classifier calibration for federated learning with non-iid data Year: (2021)
Ref_id:b25 Title: A prototype-based knowledge distillation framework for heterogeneous federated learning Year: (2023)
Ref_id:b26 Title: Elise Van der Pol, and Cees Snoek. Hyperspherical prototype networks Year: (2019)
Ref_id:b27 Title: Fedproc: Prototypical contrastive federated learning on non-iid data Year: (2023)
Ref_id:b28 Title: Person recognition system based on a combination of body images from visible light and thermal cameras Year: (2017)
Ref_id:b29 Title: Identity-clothing similarity modeling for unsupervised clothing change person re-identification Year: (2025)
Ref_id:b30 Title: Noisycorrespondence learning for text-to-image person re-identification Year: (2024)
Ref_id:b31 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b32 Title: Efficient parameter-free clustering using first neighbor relations Year: (2019)
Ref_id:b33 Title: Label-free backdoor attacks in vertical federated learning Year: (2025)
Ref_id:b34 Title: Build yourself before collaboration: Vertical federated learning with limited aligned samples Year: (2025)
Ref_id:b35 Title: Multi-memory matching for unsupervised visible Year: ()
Ref_id:b36 Title: ECCV Year: (2024)
Ref_id:b37 Title: Prototype reminiscence and augmented asymmetric knowledge aggregation for non-exemplar class-incremental learning Year: (2023)
Ref_id:b38 Title: Prototypical networks for few-shot learning Year: (2017)
Ref_id:b39 Title: Rethinking the inception architecture for computer vision Year: (2016)
Ref_id:b40 Title: Fedproto: Federated prototype learning across heterogeneous clients Year: (2022)
Ref_id:b41 Title: Relieving universal label noise for unsupervised visible-infrared person re-identification by inferring from neighbors Year: (2025)
Ref_id:b42 Title: Rethinking few-shot image classification: a good embedding is all you need? Year: (2020)
Ref_id:b43 Title: Bring evanescent representations to life in lifelong class incremental learning Year: (2022)
Ref_id:b44 Title: Visualizing data using t-sne Year: (2008)
Ref_id:b45 Title: Pose-guided feature disentangling for occluded person re-identification based on transformer Year: (2022)
Ref_id:b46 Title: Top-reid: Multi-spectral object re-identification with token permutation Year: (2024)
Ref_id:b47 Title: Mambapro: Multi-modal object re-identification with mamba aggregation and synergistic prompt Year: (2025)
Ref_id:b48 Title: Decoupled feature-based mixture of experts for multi-modal object re-identification Year: (2025)
Ref_id:b49 Title: Ran He, and Jin Tang. Interact, embed, and enlarge: Boosting modality-specific representations for multi-modal person re-identification Year: (2022)
Ref_id:b50 Title: Heterogeneous test-time training for multi-modal person re-identification Year: (2024)
Ref_id:b51 Title: Hierarchical grouping to optimize an objective function Year: (1963)
Ref_id:b52 Title: Rgb-infrared cross-modality person re-identification Year: (2017)
Ref_id:b53 Title: Unsupervised visible-infrared person re-identification via progressive graph matching and alternate learning Year: (2023)
Ref_id:b54 Title: Unsupervised feature learning via non-parametric instance discrimination Year: (2018)
Ref_id:b55 Title: Distribution rehearsing via adaptive style kernel learning for exemplar-free lifelong person re-identification Year: ()
Ref_id:b56 Title: AAAI Year: (2025)
Ref_id:b57 Title: Attribute prototype network for zero-shot learning Year: (2020)
Ref_id:b58 Title: Augmented dual-contrastive aggregation learning for unsupervised visible-infrared person re-identification Year: (2022)
Ref_id:b59 Title: Translation, association and augmentation: Learning cross-modality re-identification from single-modality annotation Year: (2023)
Ref_id:b60 Title: Instancelevel heterogeneous domain adaptation for limited-labeled sketch-to-photo retrieval Year: (2020)
Ref_id:b61 Title: Large-scale training data search for object reidentification Year: (2023)
Ref_id:b62 Title: Channel augmented joint learning for visible-infrared recognition Year: (2021-10)
Ref_id:b63 Title: Deep learning for person re-identification: A survey and outlook Year: (2021)
Ref_id:b64 Title: Transformer for object re-identification: A survey Year: (2025)
Ref_id:b65 Title: Vertical federated learning for effectiveness, security, applicability: A survey Year: (2025)
Ref_id:b66 Title: Modality unifying network for visible-infrared person re-identification Year: (2023)
Ref_id:b67 Title: Semantic drift compensation for class-incremental learning Year: (2020)
Ref_id:b68 Title: Magic tokens: Select diverse tokens for multi-modal object re-identification Year: (2024)
Ref_id:b69 Title: Prompt-based modality alignment for effective multi-modal object re-identification Year: (2025)
Ref_id:b70 Title: Refining pseudo labels with clustering consensus over generations for unsupervised object re-identification Year: (2021)
Ref_id:b71 Title: Frequency domain nuances mining for visible-infrared person re-identification Year: (2025)
Ref_id:b72 Title: Adaptive middle modality alignment learning for visible-infrared person re-identification Year: (2025)
Ref_id:b73 Title: Robust multi-modality person re-identification Year: (2021)
Ref_id:b74 Title: Adaptive sparse pairwise loss for object re-identification Year: (2023)
Ref_id:b75 Title: Prototype augmentation and self-supervision for incremental learning Year: (2021)
Ref_id:b76 Title: Dual cross-attention learning for fine-grained visual categorization and object re-identification Year: (2022)
Ref_id:b77 Title: Self-sustaining representation expansion for non-exemplar class-incremental learning Year: (2022)
