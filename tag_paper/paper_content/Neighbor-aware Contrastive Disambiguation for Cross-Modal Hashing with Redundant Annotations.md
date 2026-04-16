Title: Neighbor-aware Contrastive Disambiguation for Cross-Modal Hashing with Redundant Annotations
Abstract: Cross-modal hashing aims to efficiently retrieve information across different modalities by mapping data into compact hash codes. However, most existing methods assume access to fully accurate supervision, which rarely holds in real-world scenarios. In fact, annotations are often redundant, i.e., each sample is associated with a set of candidate labels that includes both ground-truth labels and redundant noisy labels. Treating all annotated labels as equally valid introduces two critical issues:(1) the sparse presence of true labels within the label set is not explicitly addressed, leading to overfitting on redundant noisy annotations; (2) redundant noisy labels induce spurious similarities that distort semantic alignment across modalities and degrade the quality of the hash space. To address these challenges, we propose that effective cross-modal hashing requires explicitly identifying and leveraging the true label subset within all candidate annotations. Based on this insight, we present Neighbor-aware Contrastive Disambiguation (NACD), a novel framework designed for robust learning under redundant supervision. NACD consists of two key components. The first, Neighbor-aware Confidence Reconstruction (NACR), refines label confidence by aggregating information from cross-modal neighbors to distinguish true labels from redundant noisy ones. The second, Class-aware Robust Contrastive Hashing (CRCH), constructs reliable positive and negative pairs based on label confidence scores, thereby significantly enhancing robustness against noisy supervision. Moreover, to effectively reduce the quantization error, we incorporate a quantization loss that enforces binary constraints on the learned hash representations. Extensive experiments conducted on three large-scale multimodal benchmarks demonstrate that our method consistently outperforms state-of-the-art approaches, thereby establishing a new standard for cross-modal hashing with redundant annotations. Code is available at https://github.com/Rose-bud/NACD.

Section: Introduction
With the explosion of large-scale and diverse data on the Internet [1][2][3][4][5][6][7], efficiently retrieving semantically relevant data across modalities has become increasingly important [8][9][10][11][12][13][14][15][16]. For large-scale datasets, cross-modal hashing (CMH) offers an effective solution by encoding heterogeneous data into compact binary hash codes, enabling high retrieval efficiency and low storage cost. The core challenge of CMH lies in effectively leveraging available supervision while minimizing semantic discrepancies between different modalities.
Existing CMH methods can be broadly categorized into unsupervised and supervised approaches based on whether the label information is available. The unsupervised CMH methods [17][18][19][20][21][22] learn hash functions by exploring the intrinsic structure and similarity of the data without access to labels. For instance, CIRH [21] jointly preserves the multimodal correlation and identity semantics into binary hash codes based on a heterogeneous graph network. Moreover, UCCH [22] proposes a contrastive learning-based unsupervised CMH method with a momentum optimizer and crossmodal ranking learning loss to improve performance. However, the lack of supervision limits their ability to learn semantically discriminative representations. In contrast, supervised methods [23][24][25][26][27][28][29][30][31] leverage label information to learn more discriminative hash codes to improve retrieval performance. Within a probabilistic modality alignment framework, MIAN [27] investigates the preservation of asymmetric similarities both within and between modalities, thereby fully utilizing multi-level semantic information throughout the entire database. RSHNL [31] designs a Robust Self-paced Hashing mechanism that mitigates the misleading effects of noisy labels on the model by simulating the human cognitive process. While these supervised methods have achieved satisfactory retrieval performance by leveraging label information, they implicitly rely on two assumptions: (1) all labels in the training data are accurate; (2) noisy annotations are simulated by replacing correct labels with incorrect ones. However, in real-world applications, data annotations are often redundant, i.e., each instance is labeled with a candidate label set that includes both true and spurious labels. We refer to this setting as redundant annotations. As shown in Fig. 1, among the annotations of the anchor sample pair, only "Sea, Plant, Beach, Cloud" are correct labels, while the others are additional noisy labels. Such redundancy can severely distort semantic similarity estimation and hinder effective hash learning by introducing spurious correlations across modalities. Crucially, existing methods fail to explicitly distinguish true labels within the candidate label set, leading to degraded performance under redundant noisy supervision. Although partial multi-label learning (PML) [32][33][34][35] provides a potential solution for redundant annotations, most PML methods assume a shared feature space and overlook cross-modal semantic divergence. In contrast, CMH must address both accurate label disambiguation and modality-robust contrastive pair construction, which remains a rarely explored challenge.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b26', 'b30', 'b31', 'b32', 'b33', 'b34']

Section: Anchor Neighbor
A serene beach scene with footprints in the sand, calm waters, and plants under a partly cloudy sky. In this paper, we focus on the practical scenario of redundant annotations and propose a novel framework, Neighbor-aware Contrastive Disambiguation (NACD), for robust cross-modal hashing. NACD benefits from an efficient Neighbor-aware Confidence Reconstruction (NACR) module and a novel Class-aware Robust Contrastive Hashing (CRCH) module. Specifically, NACR integrates sample label confidence with its cross-modal neighborhood label confidence to identify the groundtruth labels within the entire annotations. CRCH dynamically constructs positive and negative sample pairs based on class confidence thresholds, reducing erroneous associations caused by misleading labels and significantly enhancing the model's robustness to redundant supervision. Moreover, to effectively reduce the quantization error, we employ an effective quantization loss that enforces binary constraints on the learned hash representations. The main contributions of the proposed NACD are as follows:
this section cite: []

Section: Redundant Annotations
Sea
• We first focus on the practical scenario of redundant annotations in cross-modal hashing and propose a novel Neighbor-aware Contrastive Disambiguation (NACD) framework that addresses the challenges caused by redundant noisy supervision.
• To achieve accurate label disambiguation, an efficient Neighbor-aware Confidence Reconstruction (NACR) module is presented, which integrates the label confidence of the anchor sample pair with that aggregated from its cross-modal neighbors to identify the ground-truth labels within the entire annotations.
• We design an innovative Class-aware Robust Contrastive Hashing (CRCH) module which dynamically constructs positive and negative sample pairs based on class-wise confidence thresholds, reducing erroneous associations caused by redundant noisy labels and significantly enhancing the model's robustness to incorrect supervision.
• Comprehensive experiments on three multimodal datasets, i.e., MIRFlickr-25k, NUS-WIDE, and MS-COCO, demonstrate that NACD consistently outperforms state-of-the-art CMH methods across various redundancy levels.
2 Related Work
this section cite: []

Section: Cross-Modal Hashing
Cross-modal hashing aims to retrieve semantically relevant data across different modalities within a shared Hamming space. The key challenge lies in bridging the modality gap. To address this challenge, numerous approaches have been proposed, which can be broadly divided into two categories: unsupervised CMH methods and supervised CMH methods. More specifically, unsupervised CMH methods [19][20][21][22] learn modality-specific transformations by maximizing cross-modal correlations without label supervision. For example, UCCH [22] integrates contrastive learning into unsupervised CMH to enhance retrieval performance and robustness. However, these unsupervised methods suffer from limited performance due to the absence of explicit supervision. Supervised CMH methods [29][30][31]36] are typically based on two assumptions: (1) all labels in the training data are accurate; (2) noisy annotations are simulated by replacing correct labels with incorrect ones. For example, HCCH [36] introduces a coarse-to-fine hierarchical hashing strategy to effectively utilize hierarchical features and accurate labels across modalities. RSHNL [31] proposes a robust self-paced hashing mechanism that emulates human cognition, thereby reducing the negative impact of noisy labels and improving model performance.
However, in real-world applications, multimodal annotations often contain "redundant annotations". Therefore, this paper focuses on a largely unexplored yet challenging problem: cross-modal hashing with redundant annotations.
this section cite: ['b18', 'b19', 'b20', 'b21', 'b21', 'b28', 'b29', 'b30', 'b35', 'b35', 'b30']

Section: Learning with Redundant Annotations
Partial multi-label learning (PML) trains models using redundantly annotated data, where each instance is associated with a candidate label set containing both true and redundant noisy labels.
The key challenge in PML lies in filtering out incorrect labels and identifying reliable ones, thereby recovering the true label distribution for supervision. To achieve this, a number of methods have been developed. These methods can be broadly categorized into smoothness assumption-based, lowrank constraint-based, and sparsity regularization-based approaches. Smoothness assumption-based approaches [35,37,38] are based on the assumption that neighboring samples in the feature space are more likely to have similar labels. The low-rank constraint-based approaches [39][40][41] leverage the lowrank property to achieve disambiguation. The sparsity regularization-based approaches [32,33,40] impose sparsity on the candidate label set, effectively suppressing noisy labels and facilitating disambiguation.
In contrast to the aforementioned PML methods, our approach integrates the label confidence of an anchor sample pair with that aggregated from its cross-modal neighbors to identify true labels within all annotations. Furthermore, a dynamically updated class-wise threshold enables more accurate and adaptive label disambiguation.
Text Input A serene beach scene with footprints in the sand, calm waters, and plants under a partly cloudy sky. Text Backbone … … … … … (1,-1,1) (1,1,-1) (-1,1,1) Label Similarity 1.0 0.3 0.6 … 0 0.6 0.8 0.3 1.0 0 … 0 0.6 0.4 0.6 0 1.0 … 0.5 0 0.3 … … … … … … … 0.3 0 0.5 … 1.0 0.9 0 0.4 0.6 0 … 0.9 1.0 0.2 0.8 0.6 0 … 0 0.2 1.0 Image Input Image Backbone … … … … … negative itself positive Sea Plant Road Beach … Label Confidence 0.9 √ 0.8 √ 0.1 × 0.6 √ … Sea Plant Road Beach … Class Threshold 0.7 0.5 0.4 0.5 … Class-aware Robust Contrastive Hashing Neighbor-aware Confidence Reconstruction ... Anchor ... Neighbor 1 Reconstructed Confidence ... ... Neighbor C Hash Code Feature Similarity anchor neighbor other ... Classifier Classifier 𝐿 𝑐𝑟𝑐ℎ 𝐿 𝑛𝑎𝑐𝑟 𝐿 𝑐𝑟𝑐ℎ 𝐿 𝑛𝑎𝑐𝑟 𝐿 𝑞𝑢𝑎𝑛𝑡 𝐿 𝑞𝑢𝑎𝑛𝑡 Figure 2: The pipeline of the proposed framework NACD for cross-modal hashing with redundant annotations. NACR refines label confidence by aggregating information from cross-modal neighbors to distinguish true labels from redundant noisy ones. Meanwhile, CRCH constructs reliable positive and negative pairs based on the learned label confidence, which significantly improves robustness against noisy supervision.
this section cite: ['b34', 'b36', 'b37', 'b38', 'b39', 'b40', 'b31', 'b32', 'b39']

Section: Proposed Approach

this section cite: []

Section: Problem Definition
For ease of presentation, we first give some definitions of cross-modal hashing of redundant annotations. Suppose the input space is denoted as X , and the label space as Y = {1, 2, ..., K}, where K indicates the total number of classes. Denote x m j 2 m=1 , y j N j=1 as the training set with N sample pairs, where x m j represents the j-th instance from the m-th modality (m = 1 for image and m = 2 for text). y j ∈ {0, 1} K denotes the candidate label vector of the j-th sample pair, which encodes both ground-truth and redundant noisy labels. The k-th element of y j equals 1 if the corresponding sample pair is annotated as class k. Here, p j is the label confidence vector of the candidate label vector y j . The hash codes are denoted as b m j N j=1 ∈ {-1, 1}
L , where L is the hash code length.
CMH leverages hash functions to map data from different modalities into a common Hamming space, enabling efficient similarity search across modalities through compact binary codes. Let the hash functions be f m , where m ∈ {1, 2}. Due to the NP-hard problem in binary optimization, we calculate the hash representations by h m j = tanh(f m (x m j )), m ∈ {1, 2} in the training process. Thus, the final binary hash codes are obtained by applying the sign function: b m j = sign(h m j ), m ∈ {1, 2}. Additionally, a linear classifier with a sigmoid activation function g(•) is employed to obtain the probability distribution z m j = g(h m j ), where m ∈ {1, 2}.
this section cite: []

Section: Neighbor-aware Confidence Reconstruction
Under redundant supervision, confidence estimates obtained from individual samples may be unreliable. Meanwhile, semantically similar samples across modalities often share the same true labels. Motivated by this observation, we present a confidence-mixture (CM) strategy to reconstruct label confidence by balancing neighborhood consensus and self-prediction.
First, assume that the prediction probability of the j-th target sample x m j 2 m=1 is z j , under the supervision of redundant annotations y j . We define the confidence of each sample's prediction by the model as:
p j ← γp j + (1 -γ) 1 2 2 m=1 z m j • y j ,(1)
where • denotes Hadamard product, and γ is a momentum parameter that decays from 0.95 to 0.8 during training.
Since model predictions may be unreliable under redundant supervision, and similar samples tend to share the same true labels, we exploit the predictions of cross-modal neighbors to refine the confidence estimation. Given the C nearest neighbors N j of the anchor sample pair x j , the neighbor-aggregated confidence q j is calculated as:
q j = c∈Nj s jc p c c ′ ∈Nj s jc ′ , where s jc = 1 2 s 1 jc + s 2 jc ,(2)
where s jc denotes the fused similarity between the anchor j and its neighbor c, obtained by averaging the similarities from the image and text modalities. To mitigate the issue that inaccurate estimates during training may hinder model optimization, we reconstruct the anchor confidence p j by mixing it with the neighbor-aggregated confidence q j . The reconstructed confidence is computed as:
p j ← λq j + (1 -λ)p j ,(3)
where λ is a mixture coefficient that balances neighborhood consensus and self-prediction. After obtaining reconstructed confidence by Eq. ( 3), the disambiguation loss L nacr can be formulated as:
L nacr = - 1 2N 2 m=1 N j=1 K k=1 p jk log z m jk + (1 -p jk ) log 1 -z m jk ,(4)
where z m jk denotes the predicted probability that the j-th sample belongs to the k-th class under modality m, and p jk represents the reconstructed confidence of the j-th sample pair for class k.
Empirically, incorporating neighborhood consensus into label confidence estimation improves disambiguation accuracy, especially with abundant redundant noisy annotations. It enables the model to construct semantically faithful contrastive pairs, thus significantly enhancing the model's robustness.
this section cite: []

Section: Class-aware Robust Contrastive Hashing
Although NACR reconstructs label confidence effectively, it does not explicitly incorporate classlevel information when constructing robust positive and negative pairs. Therefore, we propose a Class-aware Robust Contrastive Hashing (CRCH) module to adaptively build reliable pseudo-labels thereby enhancing the stability of contrastive optimization.
First, we establish a class-wise threshold for each class based on the reconstructed label confidence. The threshold t k for the k-th class is calculated as follows:
t k = 1 N k N j=1 p jk ,(5)
where p jk denotes the reconstructed label confidence of the j-th sample pair on class k, and N k indicates the number of samples for which p jk > 0 among all N samples. This class-specific averaging adaptively captures the distributional characteristics of each class rather than relying on a fixed threshold. To build class-wise supervision, we derive the pseudo-label ŷj = [ŷ j1 , . . . , ŷjK ] by comparing each reconstructed confidence p jk with its corresponding class-wise threshold t k :
ŷjk = 1, if p jk ≥ t k 0, otherwise ,(6)
where ŷjk ∈ {0, 1} indicates whether class k is considered positive for the j-th sample pair. For a mini-batch containing n sample pairs, we compute a label similarity matrix T ∈ [0, 1] n×n based on the intersection-over-union (IoU) between pseudo-label vectors:
T ij = ŷi ∩ ŷj ŷi ∪ ŷj ,(7)
where T ij measures the semantic similarity between the i-th and j-th sample pair based on their pseudo-labels. A higher T ij indicates stronger semantic consistency, while T ij = 0 means that the sample pairs are semantically disjoint. Accordingly, positive and negative pairs are determined by the indicator functions I[T ij > 0] and I[T ij = 0], respectively. However, directly relying on T ij for pair construction can be unreliable due to redundant noisy annotations. To mitigate this issue, we follow [22] and introduce a margin-based thresholding mechanism that adaptively adjusts cross-modal similarities matrix S to reduce the impact of negative pairs with overly high similarities. Specifically, the adjusted similarity matrix N * ij is defined as:
N * ij = S * ij , if S * ij ≥ S ii -δ S * ij -ξ, otherwise ,(8)
where * ∈ {12, 21} denotes image-to-text and text-to-image retrieval directions.
S 12 ij = h 1 i • h 2 j , S 21 ij = h 2 i •h 1 j .
The margin parameter δ distinguishes hard and easy negative pairs, while the shift parameter ξ suppresses the influence of overly easy negatives. The loss of CRCH is defined as:
L crch = 1 n 2 n i=1 n j̸ =i I[T ij = 0] • * exp N * ij + I[T ij > 0] • exp(-S ij -T ij ) - 1 n n i=1 S ii ,(9)
where L crch consists of three components: (1) I[T ij = 0] • * exp(N * ij ) penalizes all negative cross-modal sample pairs, especially those that are easily confusable. (2) I[T ij > 0] • exp(-S ij -T ij ) treats cross-modal sample pairs with non-zero label similarity T ij as positive pairs and encourages greater semantic alignment. The higher the label similarity, the stronger the penalty imposed for insufficient feature similarity S ij . (3) -1 n n i=1 S ii encourages consistency within each sample pair by increasing the similarity between the image and text representations.
The CRCH module dynamically constructs positive and negative sample pairs based on class-wise confidence thresholds, enabling the model to effectively capture nuanced relationships between samples. This design significantly reduces erroneous associations caused by redundant noisy labels and enhances the model's robustness to incorrect supervision.
this section cite: ['b21']

Section: Optimization
Benefiting from the NACR and CRCH modules, our model effectively learns discriminative hash codes within the Hamming space. However, the discrepancy between continuous representations and discrete binary codes inevitably leads to quantization errors, which can substantially degrade retrieval performance in CMH. To address this issue, we present an effective quantization loss as follows:
L quant = 1 n • L 2 m=1 n j=1 L l=1 h m jl -sign(h m jl ) ,(10)
where h m jl denotes the l-th element of the hash representation h m j . This loss penalizes the deviation between continuous hash values and their corresponding binary codes sign(h m jl ), thereby encouraging each element to approach ±1 and effectively reducing the quantization error, as demonstrated in our ablation studies. Thus, the final loss function of NACD can be defined as:
L = L nacr + αL crch + βL quant ,(11)
where α and β are hyperparameters that balance the contributions of L nacr , L crch , and L quant . Additional optimization details of NACD are provided in Appendix Sec. A.
this section cite: []

Section: Experiments 4.1 Experimental Settings
To evaluate the effectiveness of NACD, we conducted experiments on three benchmark datasets: MIRFlickr-25k (Flickr) [42], NUS-WIDE (NUS) [43], and MS-COCO (COCO) [44]. For all methods, we evaluate the Mean Average Precision (MAP) on both Image-to-Text (I2T) and Text-to-Image (T2I) retrieval tasks. Note that all MAP scores are computed over the entire retrieval set (i.e., MAP@ALL). Additionally, precision-recall curves under the hash lookup protocol are employed to visually assess the performance of CMH. To distinguish between different levels of redundant annotations, we define a redundant rate, which represents the ratio between the number of redundant noisy labels and the number of ground-truth labels in the entire annotations. Experiments were conducted with hash code lengths of 32, 64, and 128 bits, under redundant rates of 1.0, 1.5, 2.0, and 2.5. Additional details about the datasets are provided in Appendix Sec. B.
this section cite: ['b41', 'b42', 'b43']

Section: Implementation Details
In the proposed NACD, the image modality adopts the VGG19 model [45], pre-trained on ImageNet, as its CNN backbone. For text processing, the pre-trained Doc2Vec model [46] is employed as the backbone. For cross-modal shared representation learning, the image and text modalities employ three and two hidden layers, respectively. Each fully connected (FC) layer is succeeded by a ReLU activation layer, except for the final layer, which uses a tanh function. Each hidden layer contains 8,192 units, followed by an output layer of dimension L representing the shared embedding space. The model is trained using the RMSprop optimizer [47], with an initial learning rate of 1e -5 and a maximum of 100 epochs. The parameters δ and ξ in Eq. ( 8) are set to 0.2 and 1.0, respectively. Additionally, we employ a batch size n of 128. The model is evaluated every 20 epochs, with the first 10 epochs serving as a warm-up phase during which the CM strategy and class-wise threshold update are disabled. The number of neighbors C is set to 20 to ensure accurate neighborhood information.
Our NACD is implemented using the PyTorch framework [48] and all experiments are carried out with 4 NVIDIA V100 GPUs.
this section cite: ['b44', 'b45', 'b46', 'b47']

Section: Comparison with State-of-the-Arts
In this work, we compare our NACD against 11 state-of-the-art CMH methods, including five unsupervised methods: DJSRH [18], DGCPN [20], PIP [49], CIRH [21], and UCCH [22]; and six supervised methods: CMMQ [50], MIAN [27], LtCMH [28], DHRL [29], NRCH [30], and RSHNL [31]. The average MAP scores for the I2T and T2I tasks are reported in Table 1. Additionally, the experiment results with 8 and 16 bits can be found in Appendix Sec. C.1. Fig. 3 presents precisionrecall curves on three datasets for a hash code length of 128 bits and a redundant rate of 2.5. Based on these results, we make the following observations:
• As the hash code length increases, the performance of almost all methods improves, since longer codes contain more discriminative information in the Hamming space.
• As the redundant rate increases, the performance of all supervised CMH methods deteriorates, since the progressively redundant noisy supervision misleads model training. In contrast, NACD maintains stable and superior performance by effectively extracting correct supervision from redundant annotations. Meanwhile, unsupervised CMH methods remain unaffected as they do not rely on label information.
• From Table 1, we can see that the proposed NACD consistently outperforms all competing methods across all settings. For instance, when the hash code length is 128 and the redundant rate is 2.5, NACD exceeds the second-best methods by 3.3%, 1.7%, and 2.6% on the Flickr, NUS, and COCO datasets, respectively.
• As illustrated in Fig. 3, the area under the precision-recall curves indicates that NACD consistently outperforms all other state-of-the-art methods in both I2T and T2I tasks, demonstrating its stable and superior performance.
this section cite: ['b17', 'b19', 'b48', 'b20', 'b21', 'b49', 'b26', 'b27', 'b28', 'b29', 'b30']

Section: Ablation Study
To verify the effectiveness of each component in NACD, we conducted extensive ablation studies on the NUS and COCO datasets with a hash code length of 128 bits across various redundant rates. We compared the full NACD with six ablated variants
: (1) only L nacr ; (2) only L crch ; (3) NACD without the CM strategy; (4) NACD without the class-wise threshold t k fixed at 0.5; (5) NACD without L quant ; (6) NACD without the warm-up phase; and (7) the full NACD. As shown in Table 2, NACR, CRCH, and L quant all effectively enhance the performance of NACD. Additionally, the warm-up strategy, the CM strategy, and dynamic class-wise threshold updating are all crucial for achieving optimal model performance. 0.0 0.2 0.4 0.6 0.8 1.0 Recall 0.5 0.6 0.7 0.8 0.9 precision (a) Flickr (I2T) 0.0 0.2 0.4 0.6 0.8 1.0 Recall 0.3 0.4 0.5 0.6 0.7 0.8 precision (b) NUS (I2T) 0.0 0.2 0.4 0.6 0.8 1.0 Recall 0.4 0.5 0.6 0.7 0.8 0.9 precision NACD DJSRH DGCPN PIP CIRH UCCH CMMQ MIAN LtCMH DHRL NRCH RSHNL (c) COCO (I2T) 0.0 0.2 0.4 0.6 0.8 1.0 Recall 0.55 0.60 0.65 0.70 0.75 0.80 0.85 precision (d) Flickr (T2I) 0.0 0.2 0.4 0.6 0.8 1.0 Recall 0.3 0.4 0.5 0.6 0.7 0.8 precision (e) NUS (T2I) 0.0 0.2 0.4 0.6 0.8 1.0 Recall 0.0 0.2 0.4 0.6 0.8 precision NACD DJSRH DGCPN PIP CIRH UCCH CMMQ MIAN LtCMH DHRL NRCH RSHNL (f) COCO (T2I)
Figure 3: The precision-recall curves on three datasets. Note that the hash code length is 128bits and the redundant rate is 2.5.
this section cite: []

Section: Parameter Analysis
To evaluate the impact of the coefficient λ in Eq. ( 3), as well as α and β in Eq. ( 11), we conducted extensive experiments on the COCO dataset with a hash code length of 128 bits under various redundant rates. As shown in Fig. 4, λ yields optimal results within the range of [0.01, 0.04], demonstrating the effectiveness of NACR. Moreover, the model achieves superior performance when α lies within [0.1, 0.5] and β within [0.5, 1.5], further validating the effectiveness of both CRCH and the quantization loss L quant . Additional parameter analyses on the Flick and NUS datasets are provided in Appendix Sec. C.
2. 0 0.01 0.02 0.03 0.04 0.05 0.08 1 0.65 0.66 0.67 0.68 0.69 0.70 MAP Redundant Rate:1.0 Redundant Rate:1.5 Redundant Rate:2.0 Redundant Rate:2.5 (a) λ on COCO. 0.05 0.1 0.3 0.5 1 1.5 2 3 0.50 0.55 0.60 0.65 0.70 MAP Redundant Rate:1.0 Redundant Rate:1.5 Redundant Rate:2.0 Redundant Rate:2.5 (b) α on COCO. 0 0.1 0.3 0.5 1 1.5 3 5 0.65 0.66 0.67 0.68 0.69 0.70 MAP Redundant Rate:1.0 Redundant Rate:1.5 Redundant Rate:2.0 Redundant Rate:2.5 (c) β on COCO.
this section cite: []

Section: Model Analysis

this section cite: []

Section: Robustness Analysis.
To intuitively demonstrate the robustness of NACD, we compared it with two of its variants and the NRCH method. Their average MAP scores on the COCO dataset were plotted under the settings of a hash code length of 128 bits and redundant rates of 2.0 and 2.5. Specifically, NACD-1 denotes the variant without the CM strategy, while NACD-2 represents the variant in which the class-wise threshold t k is fixed at 0.5, meaning that no dynamic threshold update is performed.  As shown in Fig. 5(a,b): (1) Both NACD-1 and NACD-2 tend to overfit redundant noise, indicating that the CM strategy and class-wise threshold updating are crucial for enhancing the robustness of NACD. (2) Although NRCH exhibits certain robustness under redundant annotations, it still lags significantly behind NACD. This gap widens as noise levels increase, further underscoring NACD's robustness in challenging scenarios.
this section cite: []

Section: Disambiguation Analysis.
In Fig. 5(c), we analyze the model's ability to disambiguate redundant annotations by computing the average pseudo-label length for all sample pairs in the COCO dataset.
Here, the pseudo-label length for the j-th sample pair is defined as the number of positive entries in ŷj , reflecting how many classes the model predicts as positive. As shown in the figure, when the redundant rate decreases, the average pseudo-label length gradually approaches the average number of ground-truth labels in the COCO dataset (2.76). This indicates that NACD can accurately recover the true label subset from redundant annotations, confirming its strong disambiguation capability.
0 20 40 60 80 100 Epoch 0.45 0.50 0.55 0.60 0.65 0.70 MAP NACD NACD-1 NACD-2 NRCH (a) redundant rate of 2.0. 0 20 40 60 80 100 Epoch 0.45 0.50 0.55 0.60 0.65 0.70 MAP NACD NACD-1 NACD-2 NRCH (b) redundant rate of 2.5. 0 20 40 60 80 100 Epoch 2 3 4 5 6 7 8 9
Average Length
Redundant Rate:1.0 Redundant Rate:1.5 Redundant Rate:2.0 Redundant Rate:2.5 { warmup (c) average label length.
Figure 5: The robustness study and disambiguation study results with 32 bits on the COCO dataset.
this section cite: []

Section: Conclusion
In this paper, we propose a novel Neighbor-aware Contrastive Disambiguation (NACD) framework to address the challenge of redundant annotations in cross-modal hashing. NACD comprises two key modules: Neighbor-aware Confidence Reconstruction (NACR) and Class-aware Robust Contrastive Hashing (CRCH). Specifically, NACR reconstructs label confidence by aggregating information from cross-modal neighbors, thereby distinguishing true labels from ambiguous ones. Meanwhile, CRCH constructs reliable positive and negative pairs based on label confidence, substantially enhancing robustness under noisy supervision. Moreover, a quantization loss is incorporated to reduce the quantization error and enforce binary constraints on the learned hash representations. Extensive experiments on three large-scale multimodal benchmarks demonstrate that NACD consistently outperforms state-of-the-art approaches, showing strong robustness and stable performance for cross-modal hashing with redundant annotations.
this section cite: []

Section: Limitation.
Although our proposed NACD demonstrates strong performance, there are still some limitations that need to be addressed. In this paper, the "redundant annotations" we study in crossmodal hashing do not account for the inherent similarity between labels. NACD may struggle when noisy labels are highly similar to the ground-truth, such as when the correct label is "car" and the noisy label is "truck". Additionally, we only conduct extensive experiments on image and text modalities to demonstrate NACD's effectiveness. In the future, additional modalities need to be considered to verify the generalization ability of NACD. We encourage further research to better understand and mitigate the limitations and risks of cross-modal hashing with redundant annotations.
this section cite: []

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: This paper proposes a novel Neighbor-aware Contrastive Disambiguation (NACD) method for cross-modal hashing with redundant annotations. The effectiveness of this method is thoroughly demonstrated through extensive experiments, which precisely align with the main claims made at the beginning. Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The limitations are discussed in the conclusion section.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We provide code and instructions for using the data and code in the supplementary material.
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
Answer: [Yes] Justification: We briefly describe the key information of the experimental setting/details in the main paper and provide full details in the appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [No] Justification: The experimental data and comparative experiments in this paper are both very extensive. In addition to the large number of retrieval results, this paper also requires a significant amount of space to analyze the model's performance and influencing factors, which makes the presentation of error bars quite challenging. To enhance the reliability of the experimental evaluation, we conducted multiple experiments and presented the average results. However, this also greatly increased the resources and time consumed in our training.
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
Answer: [Yes] Justification: We detail the computation resources used in our experiments in the appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: We ensure that the research in the paper fully complies with the Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] Justification: Although our paper addresses the redundant annotations problem, which is related to the robustness of models, it does not have a direct association with societal aspects.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes]
Justification: We will release our code, which is well documented, along with a link to the codebase repository.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
this section cite: []

Section: Institutional review board (IRB) approvals or equivalent for research with human subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
this section cite: []

Section: 
Answer: [NA] Justification: This is not a theoretical paper.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: The architecture of NACD is fully described in the main paper. The experimental setup and implementation details are disclosed in the supplementary sections.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: This paper merely addresses the Redundant Annotations problem in crossmodal hashing and does not involve these risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: We used open datasets and correctly referenced the papers.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The LLM is used only for writing. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: References
Ref_id:b0 Title: Graph regularized and feature aware matrix factorization for robust incomplete multi-view clustering Year: (2023)
Ref_id:b1 Title: Metavg: A meta-learning framework for visual grounding Year: (2023)
Ref_id:b2 Title: Self-weighted contrastive fusion for deep multi-view clustering Year: (2024)
Ref_id:b3 Title: Graph and text multi-modal representation learning with momentum distillation on electronic health records. Knowledge-Based Systems Year: (2024)
Ref_id:b4 Title: Partial multiview incomplete multilabel learning via uncertainty-driven reliable dynamic fusion Year: (2025)
Ref_id:b5 Title: Reliable representation learning for incomplete multi-view missing multi-label classification Year: (2025)
Ref_id:b6 Title: Variational graph generator for multiview graph clustering Year: (2025)
Ref_id:b7 Title: Rono: robust discriminative learning with noisy labels for 2d-3d cross-modal retrieval Year: (2023)
Ref_id:b8 Title: Dida: Disambiguated domain alignment for cross-domain retrieval with partial labels Year: (2024)
Ref_id:b9 Title: Robust unsupervised multimodal learning with noisy pseudo labels Year: (2024)
Ref_id:b10 Title: Dica: Disambiguated contrastive alignment for cross-modal retrieval with partial labels Year: (2025)
Ref_id:b11 Title: Robust duality learning for unsupervised visible-infrared person re-identification Year: (2025)
Ref_id:b12 Title: Multi-granularity confidence learning for unsupervised text-to-image person re-identification with incomplete modality Year: (2025)
Ref_id:b13 Title: Pointcloud-text matching: Benchmark dataset and baseline Year: (2025)
Ref_id:b14 Title: Roda: Robust domain alignment for cross-domain retrieval against label noise Year: (2025)
Ref_id:b15 Title: She: Streaming-media hashing retrieval Year: (2025)
Ref_id:b16 Title: Unsupervised generative adversarial cross-modal hashing Year: (2018)
Ref_id:b17 Title: Deep joint-semantics reconstructing hashing for large-scale unsupervised cross-modal retrieval Year: (2019)
Ref_id:b18 Title: Deep semantic-alignment hashing for unsupervised cross-modal retrieval Year: (2020)
Ref_id:b19 Title: Deep graph-neighbor coherence preserving network for unsupervised cross-modal hashing Year: (2021)
Ref_id:b20 Title: Work together: Correlation-identity reconstruction hashing for unsupervised cross-modal retrieval Year: (2022)
Ref_id:b21 Title: Unsupervised contrastive cross-modal hashing Year: (2022)
Ref_id:b22 Title: Deep cross-modal hashing Year: (2017)
Ref_id:b23 Title: Attention-aware deep adversarial hashing for crossmodal retrieval Year: (2018)
Ref_id:b24 Title: Multi-task consistencypreserving adversarial hashing for cross-modal retrieval Year: (2020)
Ref_id:b25 Title: Deep cross-modal hashing with hashing functions and unified hash codes jointly learning Year: (2020)
Ref_id:b26 Title: Modality-invariant asymmetric networks for cross-modal hashing Year: (2023)
Ref_id:b27 Title: Long-tail cross modal hashing Year: (2023)
Ref_id:b28 Title: Deep cross-modal hashing with ranking learning for noisy labels Year: (2024)
Ref_id:b29 Title: Robust contrastive cross-modal hashing with noisy labels Year: (2024)
Ref_id:b30 Title: Robust self-paced hashing for cross-modal retrieval with noisy labels Year: (2025)
Ref_id:b31 Title: Global-local label correlation for partial multi-label learning Year: (2022)
Ref_id:b32 Title: Partial multi-label learning with noisy label identification Year: (2022)
Ref_id:b33 Title: Partial multi-label learning with probabilistic graphical disambiguation Year: (2023)
Ref_id:b34 Title: Partial multilabel learning based on near-far neighborhood label enhancement and nonlinear guidance Year: (2024)
Ref_id:b35 Title: Hierarchical consensus hashing for cross-modal retrieval Year: (2023)
Ref_id:b36 Title: Partial multi-label learning Year: (2018-06)
Ref_id:b37 Title: Partial multi-label learning via probabilistic graph matching mechanism Year: (2020)
Ref_id:b38 Title: Feature-induced partial multi-label learning Year: (2018-11)
Ref_id:b39 Title: Partial multi-label learning by low-rank and sparse decomposition Year: (2019)
Ref_id:b40 Title: Prior knowledge regularized self-representation model for partial multilabel learning Year: (2023-03)
Ref_id:b41 Title: The mir flickr retrieval evaluation Year: (2008)
Ref_id:b42 Title: Nuswide: a real-world web image database from national university of singapore Year: (2009)
Ref_id:b43 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b44 Title: Very deep convolutional networks for large-scale image recognition Year: (2014)
Ref_id:b45 Title: An empirical evaluation of doc2vec with practical insights into document embedding generation Year: (2016)
Ref_id:b46 Title: Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude Year: (2012)
Ref_id:b47 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b48 Title: Privacy protection in deep multi-modal retrieval Year: (2021)
Ref_id:b49 Title: Mutual quantization for crossmodal search with noisy labels Year: (2022)
