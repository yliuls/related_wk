Title: Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection
Abstract: Out-of-distribution (OOD) detection attempts to distinguish outlier samples to prevent models trained on the in-distribution (ID) dataset from producing unavailable outputs. Most OOD detection methods require many ID samples for training, which seriously limits their real-world applications. To this end, we target a challenging setting: few-shot OOD detection, where only a few labeled ID samples are available. Therefore, few-shot OOD detection is much more challenging than the traditional OOD detection setting. Previous few-shot OOD detection works ignore the distinct diversity between different classes. In this paper, we propose a novel network: Adaptive Multi-prompt Contrastive Network (AMCN), which adapts the ID-OOD separation boundary by learning inter-and intra-class distribution. To compensate for the absence of OOD and scarcity of ID image samples, we leverage CLIP, connecting text with images, engineering learnable ID and OOD textual prompts. Specifically, we first generate adaptive prompts (learnable ID prompts, label-fixed OOD prompts and label-adaptive OOD prompts). Then, we generate an adaptive class boundary for each class by introducing a class-wise threshold. Finally, we propose a prompt-guided ID-OOD separation module to control the margin between ID and OOD prompts. Experimental results show that AMCN outperforms other state-of-the-art works.

Section: Introduction
Deep neural networks (DNNs) receive more and more attention due to their wide machine learning applications, such 1 Energy Research Institute @ NTU, Interdisciplinary Graduate Programme, Nanyang Technological University, Singapore 2 College of Computing and Data Science, Nanyang Technological University, Singapore 3 CNRS and CNRS@CREATE, IPAL IRL 2955, France and Singapore. Correspondence to: Xiang Fang <xiang003@e.ntu.edu.sg>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
as image classification (Bai et al., 2024;Gu et al., 2024;Li et al., 2024a). Unfortunately, most of DNNs refer to a closed-set assumption that all the test samples are seen during training and no outliers are observed during inference. In fact, there are many unseen test samples (i.e., outliers) in real-world applications, such as autonomous driving (Zendel et al., 2022;Vyas et al., 2018;Lu et al., 2023). These DNN methods still mistakenly classify each outlier into a seen class. The wrong classification of outliers will result in irrecoverable losses in some safety-critical scenarios. To solve the above problem, the out-of-distribution detection (OOD detection) task (Gautam et al., 2023;Zhang et al., 2024a;Hendrycks & Gimpel, 2016;Sun & Li, 2022;Fort et al., 2021;Liu et al., 2020) is proposed to accurately detect outliers in OOD classes and correctly classify samples from in-distribution (ID) classes during testing. Therefore, OOD detection has attracted increasing attention and various OOD detection models have been proposed for various safety-critical scenarios (Kirchheim et al., 2024;Abrecht et al., 2024;Kaur et al., 2024;Wu et al., 2024). Most OOD detection works (Shen et al., 2024;Regmi et al., 2024a;Xue et al., 2024;Regmi et al., 2024b;Zhang et al., 2024b) refer to a fully-supervised assumption that samples of all types (e.g., all races of cats) of an ID class (e.g., cat) in all situations are accessible during training, which is unrealistic. Further, many OOD samples are also usually required during training.
Few-shot OOD detection is posed to first train the designed model on a few samples in each ID class and then conduct OOD detection on the whole test set. Such a setting limits the performance of standard OOD detection methods. Existing few-shot OOD detection task meets the following challenges: 1) Most few-shot methods (Jeong & Kim, 2020;Dionelis et al., 2022;Mehta et al., 2024;Zhan et al., 2022;Zhu et al., 2024) are sensitive to the background of the image. When they train the designed model on a few number of images of each class, it might lead to the understanding bias, resulting in the wrong OOD detection results. For example, "dog" (ID) class and "wolf" (OOD) class share many visual characteristics. When a dog appears on the grassland, the designed model might misidentify it as a wolf. Besides, some similar classes have various backgrounds, which might lead to wrong OOD detection reasoning results. For instance, in a real-world training dataset, cat images are mainly indoors, while dog images are mostly outdoors. Thus, the
Few-shot learning Test set ID OOD ... ... (a) Traditional OOD detection vs few-shot OOD detection Tabby (Less diversity) ... Ox (More diversity) ... Distribution Distribution Different diversity (b) Effect of different shots (c) Distinct diversity for different classes K-shot ID images ... Statistics ... ... ... Statistics μ, σ μ, σ λ λ Statistics μ, σ λ Pull Push λ Statistics μ, σ LFOPs LAOPs ID Labels Dog Cat .
.. W1 W2 W3 ... yi ... LIPs M1 M2 M3 ... oi H1 H2 H3 ... o'i Tree Car ... OOD Labels ... ... P c =λµc +(1-λ)µc Intra-class DN Inter-class DN Cosine similarity LIP: learnable ID prompt LFOP: label-fixed OOD prompt DN: distribution normalization LAOP: label-adaptive OOD prompt Prototypes µ: mean, σ: standard deviation Any image Distribution matching ID (cat) Cosine similarity Train Test (d) Framework of our proposed method dog images often contain more complex backgrounds than the cat images. Besides, different classes have various levels of diversity, which makes it difficult to accurately learn the class boundaries with only a few samples. 2) In the few-shot setting, as the class number increases, the model performance always decreases since the ID-OOD boundary becomes more complex. Thus, the model has to learn more subtle differences between classes with only a few examples. With more classes, there is a greater likelihood of overlap between class features, making it harder for the model to distinguish between them accurately. 3) Few-shot learning inherently suffers from accessing limited samples, which exacerbates the issue of lacking representative examples to generalize well across more classes. Also, the limited samples might lead to overfitting, seriously limiting the model performance.
To address the above challenges, we design a novel network for the challenging few-shot OOD detection task. 1) To compensate for the absence of OOD and scarcity of ID image samples, we leverage CLIP (Radford et al., 2021), connecting text with images, engineering learnable ID and OOD textual prompts. we first generate three kinds of adaptive prompts (learnable ID prompts, label-fixed OOD prompts and label-adaptive OOD prompts). Then, we construct the corresponding prototypes, which effectively reduce the neg-ative impact of the background. 2) As for the second challenge, a prompt-guided OOD detection module is designed to learn an explicit margin between ID and OOD prompts for precise ID-OOD boundary. 3) About the third challenge, we ingeniously introduce two carefully-designed losses to understand the ID image features: (a) To ensure that the labeladaptive OOD prompts have no similar semantics with ID prompts, we design an OOD alignment loss based on labelfixed OOD prompts and label-adaptive OOD prompts. (b) We utilize the weighted cross-entropy loss with ID images, ID prompts and OOD prompts for multi-prompt contrastive learning.
In summary, our main contributions include:
• We target the challenging multi-diversity few-shot OOD detection task, which randomly utilizes a certain number of images with different diversity from each class for training, and conduct OOD detection on the whole testing dataset. Unlike previous works that only learn ID prompts for training, we construct ID and OOD prompts for each class to fully understand the images.
• We propose a novel AMCN for the challenging fewshot OOD detection task. Three carefully-designed
ID Labels Label Query image CLIP encoder CLIP encoder Distribution learning Dog Cat ... W1 W2 W3 ... yi M1 M2 M3 ... oi H1 H2 H3 ... o'i LIP LFOP LAOP Distribution learning CLIP encoder CLIP encoder ... ... ... Distribution learning ID CLIP features LIP CLIP features MOP CLIP features LOP CLIP features CLIP feature Prototypes ID alignment L I 1 +L I 2 OOD alignment L 3 ID-OOD Separation L 2 Tree Car ... OOD Labels Learnable Frozen Cosine similarity µ, σ DN: distribution normalization Inter-class DN L I 2 Intra-class DN L I 1 P=λµ +(1-λ)σ P: class-wise threshold µ: mean, σ: standard deviation Test L 4 L C (a) (b) (c) Prompts K-shot ID images All losses (L C L I 1 L I 2 L 2 L 3 ) ID or OOD Cosine similarity modules are utilized in AMCN to address three challenges.
• Extensive experiments show that our proposed AMCN can significantly outperform existing state-of-the-art works in the few-shot OOD detection task.
this section cite: ['b1', 'b12', 'b73', 'b60', 'b38', 'b11', 'b13', 'b9', 'b36', 'b24', 'b0', 'b22', 'b63', 'b54', 'b66', 'b19', 'b40', 'b74', 'b81', 'b49']

Section: Related Works

this section cite: []

Section: Out-of-distribution Detection
As a challenging machine learning task, the out-ofdistribution (OOD) detection task aims to detect test samples from distributions that do not overlap with the training distribution. Previous OOD detection methods (Liang et al., 2018b;Liu et al., 2020;Sun et al., 2021;Lee et al., 2018b;Mohseni et al., 2020;Vyas et al., 2018;Yu & Aizawa, 2019;Zaeemzadeh et al., 2021;Hsu et al., 2020;Ming et al., 2023;Jiang et al., 2024) can be divided into four types: classification-based methods (Hendrycks & Gimpel, 2016;Liang et al., 2018b;Lee et al., 2018c;a), density-based methods (Kirichenko et al., 2020;Serrà et al., 2019), distancebased methods (Techapanurak et al., 2020;Lee et al., 2018b) and reconstruction-based methods (Zhou, 2022;Yang et al., 2022). Although previous works have achieved decent suc-cess, most of them require all the samples for training. Besides, they ignore the different levels of diversity between different classes. Different from these OOD detection methods, we aim at a more challenging task: few-shot OOD detection with multi-diversity distribution.
this section cite: ['b36', 'b56', 'b28', 'b45', 'b60', 'b71', 'b72', 'b15', 'b42', 'b20', 'b13', 'b25', 'b53', 'b59', 'b28', 'b80', 'b68']

Section: Prompt Learning
Prompt learning (Gao et al., 2024;Park et al., 2024;Kim et al., 2024) is an emerging area in natural language processing that leverages prompts or instructions to guide pretrained language models like GPT (Brown et al., 2020), BERT (Devlin, 2018), T5 (Raffel et al., 2020), etc., to perform various downstream tasks. Prompt learning approaches (Liu et al., 2023;Pouramini & Faili, 2024;Xing et al., 2024) aim to bridge the gap between pre-training and fine-tuning by providing models with task-specific context or guidance in the form of natural language prompts. However, most prompt learning works (Pouramini & Faili, 2024;Xing et al., 2024) refer to the closed-set assumption, and cannot be directly utilized into challenging few-shot OOD detection task.
this section cite: ['b10', 'b47', 'b23', 'b2', 'b7', 'b50', 'b35', 'b48', 'b65', 'b48', 'b65']

Section: Few-shot Learning
Few-shot learning (Hu et al., 2024;Zhang et al., 2025;Hu et al., 2025;Wang et al., 2025) is a branch of machine learning that focuses on building models capable of learning new concepts with only a few examples. Unlike traditional machine learning models that require large amounts of labeled data to generalize well, few-shot learning aims to make the most out of limited data, mimicking the human ability to learn from only a few examples. Since only a few samples can be used during few-shot training, we often obtain limited knowledge from these training samples. Obviously, our targeted few-shot OOD detection is more challenging than traditional OOD detection.
this section cite: ['b17', 'b76', 'b16', 'b62']

Section: Methodology
Problem definition. For the K-shot OOD detection task, it aims to use only K labeled ID images from each class for model training, and to test on the complete test set for OOD detection. For the training process, we denote the training set as:
D id = {(x i , y i )|i ∈ {1, ..., N }, y i ∈ {1, ..., C}}
with N labeled images from C ID classes. We denote D ood = (x ood , y ood ) as the OOD dataset, where x ood is the input OOD image, and y ood ∈ Y ood := C + 1, ..., C + O denotes the OOD label, where O is the OOD label number. Please note that the OOD labels are unknown during training, and they have no overlap of classes with ID labels, i.e,., Y ood ∩ Y id = ϕ. Please note that the OOD data D ood is inaccessible during training.
Pipeline. We present our pipeline in Figure 2. Firstly, we utilize the pretrained CLIP encoder (Radford et al., 2021) to extract the image features. Then, we generate adaptive prompts for ID classification. Specifically, we combine P learnable ID prefixes and the label name to generate the learnable ID prompts (LIPs). Also, we generate S labelfixed OOD prompts (LFOPs) by introducing OOD labels from other datasets that disjoint with the ID label set. Since the introduced OOD labels are often limited, we explore Z label-adaptive OOD prompts (LAOPs) for each ID prompt. Besides, we align the image features and ID prompt features by a prompt-guided contrastive loss for ID classification. Moreover, we learn the different distributions of all the classes for adaptive ID alignment. Finally, a prompt-guided OOD detection module is designed to control the explicit margin between ID and OOD prompts for OOD detection.
this section cite: ['b49']

Section: Adaptive Prompt Generation for ID Classification
In real-world applications, we only access ID samples during training. Therefore, it is unrealistic to directly obtain the OOD prompt for the future OOD detection task. Given a sentence, we can obtain different sentences with various semantics by changing the prefix of the sentence or replacing the class label. For the text prompt for class y i , we follow the popular predefined templates: "a photo of a [y i ]", where [y i ] denotes the corresponding class name. Thus, we can design the learnable ID prompt as follows:
f i lip = [W 1 ][W 2 ] . . . [W N IP ][y i ],(1)
where N IP denotes the length of the ID prefix and [W i ] denotes the i-th text token learned from the CLIP network. By Eq. ( 1), we can construct the learnable ID prompt.
Similarly, we generate OOD prompts from the OOD labels in other large-scale datasets, which can provide partial knowledge about OOD samples (Yang et al., 2024;Liu et al., 2021;Cao et al., 2024). In real-world applications, we can obtain partial knowledge about OOD samples. For example, when we treat the CIFAR-10 dataset (Krizhevsky, 2009) as ID, we can use "chair" in the CIFAR-100 dataset (Krizhevsky, 2009) as OOD since CIFAR-10 has the "chair" class. In this way, we want to generate two types of labeladaptive OOD prompts: label-fixed OOD prompts and labeladaptive OOD prompts. The label-fixed OOD prompt will introduce some human knowledge to assist our model for OOD detection. As for the label-adaptive OOD prompt, we let it learn prefixes and labels by itself. To obtain the OOD prompt based on the ID prompt and the OOD labels, we utilize a similar process to generate the adaptive OOD prompts:
f i lf op = [M 1 ] . . . [M N lf op ][o i ], f i laop = [H 1 ] . . . [H N laop ][o ′ i ],(2)
where [o i ] is the OOD label from other datasets (D ood ) that disjoint with D id ; [o ′ i ] is the learnable label, which is initialized by [o i ]; N laop denotes the length of label-adaptive OOD prefix, f i lf op and f i laop denote the label-fixed OOD prompt and label-adaptive OOD prompt, respectively. For f i lf op , its prefix is learnable and its label is fixed. As for f i laop , both its prefix and its label are learnable. Based on f i lf op , we can handle various prefix structures; by f i laop , we are able to explore different latent OOD labels. To ensure that the prompt feature dimensions are consistent, we make all the text encoders share the parameter weights. Besides, we align ID prompts with corresponding ID images, and push OOD prompts away from ID images. During training, since more negative prompts (OOD prompts) will help us conduct better multi-prompt contrastive learning for guidance, we utilize all OOD prompts to compare with ID images. For convenience, we introduce a similarity function S(a, b) for two inputs (a and b) as follows:
S(a, b) = exp[1/σ • cos(a, b)],(3)
where cos(•, •) denotes the cosine similarity; σ is a temperature parameter. Therefore, we introduce the following
Tabby (Less diversity) Ox (More diversity) Tabby Ox Class boundary ... ... Different diversity T-SNE visualization prompt learning loss for ID classification:
LC= E f i x -log τ1 f i x S(f i x , f i lip ) τ1 f i x S(f i x , f i lip )+(1-τ1) f i op ∈Fo S(f i x , f i op ) ,(4)
where
f i x denotes ID image feature; f i lip = ave(f 1 lip , f 2 lip , ..., f P lip )
is the prototype of ID prompt features for the i-th image, where ave(•) denotes the average pooling;
F o = {ave(f i op )|f i op ∈ {f i lf op } N lf op i=1 ∪ {f i laop } N laop i=1
} is a set of OOD prompt features. Based on L 1 , we can train our ID classifier by ID prompts and OOD prompts.
this section cite: ['b67', 'b34', 'b4', 'b26', 'b26']

Section: Prompt-based Multi-diversity Distribution Learning for Adaptive ID Alignment
In real-world applications, different classes indeed exhibit varying degrees of sample diversity. This diversity, which reflects how varied the images within a class are, can be influenced by multiple factors, including the nature of the class, its semantic breadth, and the challenges of collecting representative samples.
In fact, there is a distribution gap between unseen ID samples (i.e., not selected in K samples) and OOD samples. Since these unseen ID samples is not used for training, previous OOD detection methods might treat these unseen ID samples as OOD samples, which will lead to incorrect detection results. In addition, these methods utilize the same threshold for all classes, seriously limiting their performance in real-world complex applications. Motivated by the effectiveness of normal distribution, we try to learn the threshold for each ID class.
For any class c, we name these samples with y ̸ = c as pseudo-OOD samples, where c ∈ {1, ..., C} is the corresponding class of f i x ,. If a sample is the pseudo-OOD sample for all the classes, it is a real OOD sample. For a dataset with 2 classes ("cat" and "dog"), a "dog" sample is the pseudo-OOD sample for the "cat" class, while a "tree" sample is the real OOD sample for both "cat" and "dog" classes. In this section, we design an adaptive distribution extraction module to fully learn the distribution of each class based on only a few samples and fine-tune the prediction output of any test sample.
Learning distribution. Since the mean and the standard deviation are two significant metrics to learn the distribution, we calculate them in each class. Given K ID training samples {x i } K i=1 ∈ D id in the c-th class, we estimate the mean µ in and standard deviation σ in as follows:
µ c = K i=1 S c (x i ) K , σ c = K i=1 (S c (x i ) -µ c ) 2 K -1 ,(5)
where S c (x i ) is a class distribution score, which is defined as follows:
S c (x i ) = exp (o c (x i )) τ 0 + M pse c ,(6)
where τ 0 is a parameter adjustable as need, o c (x i ) denotes the logit output of sample x i in class c, and M pse c ∈ R denotes the initial pseudo-OOD distribution. To adapt the pseudo-OOD distribution M pse c , we first use the OOD filter to predict OOD samples, and then conduct a momentum update of M pse c during inference. Based on the above process, M pse c is updated as the mean of distribution for the predicted OOD samples. For the pseudo-OOD distribution M pse c , we first initialize its entries based on the mean distribution of the pseudo-OOD samples. Then, we update these entries by an online fashion module.
As shown in Figure 3, different classes have distinct diversity. Therefore, a diversity-guided decision boundary is required for each class during classification. We propose the following novel P-score as the class-wise threshold for diversity-guided decision boundary:
P c = λ • µ c + (1 -λ) • σ c ,(7)
where λ is a parameter to balance the weight of mean and standard deviation. Based on S c (x i ) and P c , we can obtain:
x i belongs to pseudo-OOD, S c (x i ) > P c , class c, S c (x i ) ≤ P c .(8)
Based on (8), if a sample x i is pseudo-OOD for all the classes, it will be detected as real OOD. If any sample x i is detected as pseudo-OOD sample, we can update the distribution M pse c as follows:
M pse c (t) = M pse c (t -1), S c (x i ) > P c , exp(oc(xi))+O•M pse c (t-1) O+1 , S c (x i ) ≤ P c ,(9)
where t denotes the t-th iteration during training and O denotes the number of predicted OOD samples. We only keep O and current M pse c unchanged during inference.
Previous OOD detection works utilize common global distribution loss to learn the vanilla pseudo-OOD distribution M pse c . Besides, we have to manually fine-tune the sensitive hyperparameters on distribution margins in complex datasets under the challenging few-shot setting, which might result in an sensitive OOD filter and then destroy the distribution learning. To this end, we aim to explore intra-class distribution and inter-class distribution.
this section cite: []

Section: Intra-class distribution normalization.
To fully learn the intra-class distribution of ID samples for better classification, we independently normalize the distribution for each class by the following loss:
L 1 I = C c=1 (E (f i x ,c) [(max(0, ϵ1 - B i=1 τ1 f i x S(f i x , f i lip ) Mi )) 2 ] + E f i op ∈Fo [(max(0, B i=1 (1-τ1) f i op ∈Fo S(f i x , f i op ) Mi -ϵ2)) 2 ]),(10)
where ϵ 1 and ϵ 2 are two hyper-parameters, B denotes the batch size, and M i is defined as:
Mi = τ1 f i x S(f i x , f i lip )+(1-τ1) f i op ∈Fo S(f i x , f i op ), (11)
where τ 1 is a parameter adjustable as need, Based on L 1 I , we can balance the the sum of distribution score on all ID samples for each ID class within a batch.
Inter-class distribution normalization. Similar to intraclass distribution normalization, we balance the distributions of all the classes by the following loss:
L 2 I = E (f i x ,c) [(max(0, ϵ3 - C c=1 τ1 f i x S(f i x , f i lip ) Mi )) 2 ](12)
+ E f i op ∈Fo [(max(0, C c=1 (1-τ1) f i op ∈Fo S(f i x , f i op ) Mi -ϵ4)) 2 ],
where ϵ 3 and ϵ 4 are two hyper-parameters.
By integrating L C , L 1 I and L 2 I , we can obtain the final ID classification loss as follows:
L 1 = L C + L 1 I + L 2 I .(13)
this section cite: []

Section: Prompt-guided OOD Detection
The realistic OOD detection networks always face the following challenges: 1) we rarely obtain OOD images.
2) The OOD prompts (label-fixed and label-adaptive) in Eq. ( 4) are treated equally, and only treat ID image features to generate negative samples for multi-prompt contrastive learning, which might lead to an unclear margin between ID and OOD prompt and wrong OOD detection results. We observe that many OOD prompts can help us understand the ID images. For example, "a photo of a cat" contains the ID semantics (cat), and we can change the label to obtain an OOD prompt "a photo of a chair", which corresponds to the OOD semantics. To update these OOD prompts, we initialize their embeddings and introduce a weighted OOD alignment loss. Then, we integrate ID prompts and OOD prompts by a multi-prompt contrastive learning strategy for OOD detection. Remark 3.1. In our proposed AMCN, all final features are projected onto the unit hyper-sphere for cross-modal matching.
Therefore, we propose a novel prompt-guided ID-OOD separation module to generate an explicit margin between ID and OOD prompt features. Thus, we introduce the following prompt-guided ID-OOD separation loss:
L2= E f i x -min 0, e( f i x ∥f i x ∥2 , f i op ∥ f i op∥2
)-e(
f i x ∥f i x ∥2 , f i lip ∥ f i lip∥2
) ,
where e(•, •) denotes the euclidean distance. To mine the latent OOD information, we conduct weighted average on all OOD prompt features to generate the final OOD prototype f i op :
f i op = 1 S + Z [ S i=1 ave(f i lf op ) + Z i=1 ave(f i laop )]. (15
)
Similarly, we normalize the features in L 2 and set the margin to 0. Unlike L 1 , L 2 attempts to generate a larger margin between ID samples and the OOD prototype than between ID samples and the ID prototype, allowing our model to correctly distinguish ID and OOD prototypes. To keep the label-adaptive OOD prompts away from the ID prompts, we conduct the following weighted OOD alignment based on label-fixed OOD prompts and label-adaptive OOD prompts:
L 3 = i τ 2 • f i laop ∥ f i laop ∥ 2 - (1 -τ 2 ) • f i lf op ∥ f i lf op ∥ 2 2 2 ,(16)
where || • || 2 is the L2-norm; τ 2 is a weight parameter; for the i-th image, f i lf op and f i laop are respectively the feature prototypes of label-fixed OOD prompts and label-adaptive OOD prompts.
Similar to the ID classification loss L 1 , we design the multiprompt contrastive learning loss for OOD detection:
L4 = E f i op    -log τ3 f i op S(f i x , f i op ) (1-τ3) f i lip S(f i x , f i lip )+τ3 f i op S(f i x , f i op )    , (17
)
where τ 3 is a weight parameter. Based on L 4 , we can minimize the similarity between ID prompts and OOD prompts for clearer ID-OOD separation.
The overall loss function with balanced hyperparameters (α 1 , α 2 and α 3 ) for training is as follows:
L = L 1 + α 1 L 2 + α 2 L 3 + α 3 L 4 ,(18)
where α 1 , α 2 and α 3 are parameters to balance the significance between different losses.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
Datasets. For fair comparison, we follow MOS (Huang & Li, 2021) and MCM (Ming et al., 2022) to utilize ImageNet-1k (Deng et al., 2009) as ID set and a subset of iNaturalist (Horn et al., 2018), PLACES (Zhou et al., 2018), TEXTURE (Cimpoi et al., 2014) and SUN (Xiao et al., 2010) as OOD set. For each OOD set, the classes are not overlapping with the ID set. Also, we follow (Huang & Li, 2021) to randomly select these OOD data from the classes disjointing from ImageNet-1k (Deng et al., 2009). Please refer to (Huang & Li, 2021;Miyai et al., 2023;Ming et al., 2022) for more dataset details and public dataset split.
Implementation details. Following (Miyai et al., 2023), we adopt CLIP-ViT-B/16 (Radford et al., 2021) as the pretrained model for OOD prompt learning. For the few-shot setting (Ye et al., 2020), following previous works (Miyai et al., 2023;Ye et al., 2020), we try different shots (1, 2, 4, 8, 16). For the parameters, we set α 1 = 0.4, α 2 = 0.2, α 3 = 0.8, θ = 0.8, τ = 1.0, γ = 0.7, P = 1, S = 50, Z = 50. We set AdamW (Loshchilov & Hutter, 2019) as the optimizer, the learning rate of 0.003, the batch size as 64, the token length as 16 and the training epoch as 100. Codes are available in Github.
Evaluation metrics. Following (Miyai et al., 2023;Bukhsh & Saeed, 2023;Nakamura et al., 2024;Kahya et al., 2024), we employ three popular evaluation metrics: FPR95, AU-ROC and ACC. FPR95 means the false positive rate of OOD samples when the true positive rate of ID samples is at 95%. AUROC measures the area under the receiver operating characteristic curve.
this section cite: ['b18', 'b41', 'b6', 'b14', 'b78', 'b5', 'b64', 'b18', 'b6', 'b18', 'b43', 'b41', 'b43', 'b49', 'b69', 'b43', 'b69', 'b37', 'b43', 'b3', 'b46', 'b21']

Section: Comparison With State-of-the-arts Compared methods.
To comprehensively analyze the performance of our model, we compare our model with four types of state-of-the-art OOD detection models: fullysupervised (Full), zero-shot, one-shot and eight-shot, where fully-supervised methods and zero-shot methods are baselines for performance comparison. The following opensource methods are selected for performance comparison. 1) Fully-supervised: ODIN (Liang et al., 2018a), ViM (Wang et al., 2022), KNN (Sun et al., 2022), NPOS (Tao et al., 2023). 2) Zero-shot: MCM (Ming et al., 2022), SeTAR (Li et al., 2024b) with MCM Score and GL-MCM (Miyai et al., 2025). 3) One-shot and eight-shot: CoOp (Zhou et al., 2022) and LoCoOp (Miyai et al., 2023), SCT (Yu et al., 2024).
Experimental results. As shown in Table 1 and Figure 0 0.2 0.4 0.6 0.8 1 Value 0 20 40 60 80 100 FPR95 1 2 3 0 0.2 0.4 0.6 0.8 1 Value 50 60 70 80 90 100 AUROC 1 2 3 Figure 5. Parameter analysis on SUN.
4, we compare our proposed method with state-of-the-art methods, where our proposed method achieves the best performance in all the cases. In particular, with the Texture dataset as the OOD set in terms of "FPR95", our method outperforms state-of-the-art method SCT by 9.71% under the one-shot setting. The significant performance improvement is mainly because our method can first construct adaptive ID and OOD prompts by only limited labeled ID images, and then conduct prompt-based ID-OOD separation. On the SUN dataset, compared with SCT in terms of "AUROC", our method improves the OOD detection performance by 0.27% under the one-shot setting and by 1.12% under the eight-shot setting. The main reason is that our method can effectively learn the distribution of each class to reduce the negative impact of multi-diversity distribution on SUN. In Figure 4, we compare two representative few-shot OOD detection works (LoCoOp and CoOp) under different fewshot settings. Obviously, our method outperforms LoCoOp and CoOp in all the cases by a large margin. In many cases, LoCoOp and CoOp even perform worse than zeroshot GL-MCM, while our method significantly outperforms GL-MCM. The core reason is that LoCoOp and CoOp are misled by the various background information in the images. Different from LoCoOp and CoOp, we can learn the proper prototypes for each input (ID image, learnable ID prompt, label-fixed OOD prompt and label-adaptive OOD prompt) based on the limited images to handle various backgrounds.
Visualization results. To qualitatively investigate the effectiveness of our method, we report a representative example.
As shown in Figure 6, our method achieves better performance in both ID classification and OOD detection, which further shows the effectiveness of our method.    for the challenging few-shot OOD detection task. As the core module, "Adaptive Prompt Generation" can generate adaptive prompts to fully understand the images and labels by three adaptive prompts (LIP, LFOP and LAOP) for ID classification. In Texture, different classes have distinct diversities, which makes many few-shot OOD detection methods difficult to learn the correct distribution for each class. Fortunately, "Prompt-based Multi-diversity Distribution Learning" can effectively learn the intra-class distribution and inter-class distribution, which reduces the negative impact of distinct diversity of different classes. Based on three adaptive prompts, "Prompt-guided OOD detection" can correctly detect OOD samples in Texture.
Importance of different prompts. In the "Adaptive Prompt Generation" module, we design three adaptive prompts (LIP, LFOP and LAOP) for each labeled ID sample. To show the importance of different prompts, we conduct an ablation  Effect of the prompt number. We further conduct an ablation study to analyze the impact of the negative prompt numbers (LFOP number S and LAOP number Z) on iNaturalist. As shown in Figure 7, we can observe that, with the increase of K, the variation of the performance follows a general trend, i.e., rises at first and then starts to decline.
The optimal LFOP number S is 50 and the optimal LAOP number Z is 50. Thus, we set S = Z = 50 in this paper.
Influence of adaptive threshold. In our "Prompt-based Multi-diversity Distribution Learning" module, we design an adaptive threshold P c for each class. To assess the performance of the adaptive threshold P c , we change the threshold to obtain the ablation models. Table 4 illustrates the performance comparison for different models. Obviously, our full model obtains the best results since our module can generate an adaptive threshold, which illustrates the effectiveness of our adaptive threshold to learn both inter-class distribution and inter-class distribution on multi-diversity dataset iNaturalist.
Parameter analysis. We introduce some losses to supervise the training process. To evaluate the significance of these losses, we conduct experiments on the SUN dataset under the eight-shot setting. Figure 5 presents the ablation study on the hyper-parameters (α 1 , α 2 , α 3 ). We can find that, their performance only varies in a small range, indicating that our model is insensitive to these parameters. Our model can achieve the best performance with α 1 = 0.4, α 2 = 0.2, α 3 = 0.8. Therefore, we utilize the above parameter setting in our all experiments.
this section cite: ['b61', 'b58', 'b41', 'b44', 'b79', 'b43', 'b70']

Section: Conclusion
In this paper, we target a challenging machine learning task: few-shot OOD detection, which only uses a few labeled ID images from each class to train the designed model and to test the complete test set for OOD detection. To address it, we propose a novel method, AMCN, which generates adaptive prompts by the given label set to learn the distribution of each class for adaptive OOD detection. Experimental results on multiple challenging benchmarks demonstrate the effectiveness of our proposed method.
this section cite: []

Section: References
Ref_id:b0 Title: Deep learning safety concerns in automated driving perception Year: (2024)
Ref_id:b1 Title: Id-like prompt learning for few-shot out-of-distribution detection Year: (2024)
Ref_id:b2 Title: Language models are few-shot learners Year: (2020)
Ref_id:b3 Title: On out-of-distribution detection for audio with deep nearest neighbors Year: (2023)
Ref_id:b4 Title: Envisioning outlier exposure by large language models for out-of-distribution detection Year: (2024)
Ref_id:b5 Title: Describing textures in the wild Year: (2014)
Ref_id:b6 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b7 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b8 Title: Frob: Few-shot robust model for joint classification and outof-distribution detection Year: ()
Ref_id:b9 Title: Exploring the limits of out-of-distribution detection Year: (2021)
Ref_id:b10 Title: Label alignment for multi-modal prompt learning Year: (2024)
Ref_id:b11 Title: Unsupervised out-of-distribution detection using few indistribution samples Year: (2023)
Ref_id:b12 Title: Detecting industrial anomalies using large vision-language models Year: (2024)
Ref_id:b13 Title: A baseline for detecting misclassified and out-of-distribution examples in neural networks Year: (2016)
Ref_id:b14 Title: The inaturalist species classification and detection dataset Year: (2018)
Ref_id:b15 Title: Generalized odin: Detecting out-of-distribution image without learning from out-of-distribution data Year: (2020)
Ref_id:b16 Title: Progressive learning strategy for few-shot class-incremental learning Year: (2025)
Ref_id:b17 Title: Understanding few-shot learning: Measuring task relatedness and adaptation difficulty via attributes Year: (2024)
Ref_id:b18 Title: MOS: towards scaling out-ofdistribution detection for large semantic space Year: (2021)
Ref_id:b19 Title: Ood-maml: Meta-learning for few-shot out-of-distribution detection and classification Year: (2020)
Ref_id:b20 Title: Negative label guided ood detection with pretrained vision-language models Year: (2024)
Ref_id:b21 Title: Human activity classification and out-of-distribution detection with short-range FMCW radar Year: (2024)
Ref_id:b22 Title: Outof-distribution detection in dependent data for cyberphysical systems with conformal guarantees Year: (2024)
Ref_id:b23 Title: Adding attributes to prompt learning for vision-language models Year: (2024)
Ref_id:b24 Title: Out-ofdistribution detection with logical reasoning Year: (2024)
Ref_id:b25 Title: Why normalizing flows fail to detect out-of-distribution data Year: (2020)
Ref_id:b26 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b27 Title: Training confidencecalibrated classifiers for detecting out-of-distribution samples Year: ()
Ref_id:b28 Title: A simple unified framework for detecting out-of-distribution samples and adversarial attacks Year: (2018)
Ref_id:b29 Title: Hierarchical novelty detection for visual object recognition Year: (2018)
Ref_id:b30 Title: Promptad: Learning prompts with only normal samples for few-shot anomaly detection Year: (2024)
Ref_id:b31 Title: Out-ofdistribution detection with selective low-rank approximation Year: ()
Ref_id:b32 Title: Enhancing the reliability of out-of-distribution image detection in neural networks Year: ()
Ref_id:b33 Title: Enhancing the reliability of out-of-distribution image detection in neural networks Year: (2018)
Ref_id:b34 Title: Towards out-of-distribution generalization: A survey Year: (2021)
Ref_id:b35 Title: Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing Year: (2023)
Ref_id:b36 Title: Energy-based out-of-distribution detection Year: (2020)
Ref_id:b37 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b38 Title: Uncertainty-aware optimal transport for semantically coherent out-of-distribution detection Year: (2023)
Ref_id:b39 Title: Learning with mixture of prototypes for out-ofdistribution detection Year: (2024)
Ref_id:b40 Title: Out-of-distribution detection and classification in few-shot settings Year: (2024)
Ref_id:b41 Title: Delving into out-of-distribution detection with vision-language representations Year: (2022)
Ref_id:b42 Title: How to exploit hyperspherical embeddings for out-of-distribution detection? Year: (2023)
Ref_id:b43 Title: Fewshot out-of-distribution detection via prompt learning Year: (2023)
Ref_id:b44 Title: Gl-mcm: Global and local maximum concept matching for zero-shot outof-distribution detection Year: (2025)
Ref_id:b45 Title: Selfsupervised learning for generalizable out-of-distribution detection Year: (2020)
Ref_id:b46 Title: Pseudo-outlier synthesis using q-gaussian distributions for out-of-distribution detection Year: (2024)
Ref_id:b47 Title: Prompt learning via metaregularization Year: (2024)
Ref_id:b48 Title: Matching tasks to objectives: Fine-tuning and prompt-tuning strategies for encoderdecoder pre-trained language models Year: (2024)
Ref_id:b49 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b50 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b51 Title: T2fnorm: Train-time feature normalization for ood detection in image classification Year: (2024)
Ref_id:b52 Title: Reweightood: Loss reweighting for distance-based ood detection Year: (2024)
Ref_id:b53 Title: Input complexity and out-ofdistribution detection with likelihood-based generative models Year: (2019)
Ref_id:b54 Title: Optimizing ood detection in molecular graphs: A novel approach with diffusion models Year: (2024)
Ref_id:b55 Title: Dice: Leveraging sparsification for out-ofdistribution detection Year: (2022)
Ref_id:b56 Title: React: Out-of-distribution detection with rectified activations Year: (2021)
Ref_id:b57 Title: Out-of-distribution detection with deep nearest neighbors Year: (2022)
Ref_id:b58 Title: Non-parametric outlier synthesis Year: (2023)
Ref_id:b59 Title: Hyperparameter-free out-of-distribution detection using cosine similarity Year: (2020)
Ref_id:b60 Title: Out-of-distribution detection using an ensemble of self supervised leave-out classifiers Year: (2018)
Ref_id:b61 Title: Out-ofdistribution with virtual-logit matching Year: (2022)
Ref_id:b62 Title: Integrated image-text augmentation for few-shot learning in vision-language models Year: (2025)
Ref_id:b63 Title: Any-to-any multimodal LLM Year: (2024)
Ref_id:b64 Title: SUN database: Large-scale scene recognition from abbey to zoo Year: (2010)
Ref_id:b65 Title: A survey of efficient fine-tuning methods for vision-language models-prompt and adapter Year: (2024)
Ref_id:b66 Title: Enhancing the power of ood detection via sample-aware model selection Year: (2024)
Ref_id:b67 Title: Generalized out-ofdistribution detection: A survey Year: (2024)
Ref_id:b68 Title: Out-of-distribution detection with semantic mismatch under masking Year: (2022)
Ref_id:b69 Title: Few-shot learning via embedding adaptation with set-to-set functions Year: (2020)
Ref_id:b70 Title: Self-calibrated tuning of vision-language models for out-of-distribution detection Year: (2024)
Ref_id:b71 Title: Unsupervised out-of-distribution detection by maximum classifier discrepancy Year: (2019)
Ref_id:b72 Title: Out-of-distribution detection using union of 1-dimensional subspaces Year: (2021)
Ref_id:b73 Title: Unifying panoptic segmentation for autonomous driving Year: (2022)
Ref_id:b74 Title: A closer look at few-shot out-of-distribution intent detection Year: (2022)
Ref_id:b75 Title: EPA: neural collapse inspired robust out-of-distribution detector Year: (2024)
Ref_id:b76 Title: Few-shot class-incremental learning for classification and object detection: A survey Year: (2025)
Ref_id:b77 Title: Lapt: Label-driven automated prompt tuning for ood detection with vision-language models Year: (2024)
Ref_id:b78 Title: Places: A 10 million image database for scene recognition Year: (2018)
Ref_id:b79 Title: Learning to prompt for vision-language models Year: (2022)
Ref_id:b80 Title: Rethinking reconstruction autoencoder-based outof-distribution detection Year: (2022)
Ref_id:b81 Title: Croft: Robust fine-tuning with concurrent optimization for ood generalization and open-set ood detection Year: (2024)
Ref_id:b82 Title: Challenges in diverse classes. High-diversity classes pose challenges for machine learning models, as they must generalize across a wide range of appearances and contexts. Conversely, low-diversity classes might lead to models that perform well on training data but lack robustness to real Year: ()
