Title: PASS: Private Attributes Protection with Stochastic Data Substitution
Abstract: The growing Machine Learning (ML) services require extensive collections of user data, which may inadvertently include people's private information irrelevant to the services. Various studies have been proposed to protect private attributes by removing them from the data while maintaining the utilities of the data for downstream tasks. Nevertheless, as we theoretically and empirically show in the paper, these methods reveal severe vulnerability because of a common weakness rooted in their adversarial training based strategies. To overcome this limitation, we propose a novel approach, PASS, designed to stochastically substitute the original sample with another one according to certain probabilities, which is trained with a novel loss function soundly derived from information-theoretic objective defined for utilitypreserving private attributes protection. The comprehensive evaluation of PASS on various datasets of different modalities, including facial images, human activity sensory signals, and voice recording datasets, substantiates PASS's effectiveness and generalizability.

Section: Introduction
The expansion of modern Machine Learning (ML) services has seamlessly improved the convenience of daily lives, where the service providers oftentimes gather data from users and then utilize advanced models to cater to users' needs (Secinaro et al., 2021;Ahmed et al., 2022). However, the data collected frequently includes private information that users may be reluctant to disclose (Boulemtafes et al., 2020;Kumar et al., 2023). For instance, a human voice recognition service needs to collect speaker's voice Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
to extract the content (Kheddar et al., 2024), which would inadvertently contain private attributes such as the speaker's gender or accent, imposing the risk of privacy leakage when an adversary tries to eavesdrop on the voice. Therefore, there has been a lasting interest in developing a data obfuscation module that can be inserted into the data sharing or ML service pipelines to protect the private attributes by suppressing or removing them from the data, while maintaining the utility of the data for downstream tasks.
Various methods are proposed towards this goal. Roy & Boddeti (2019); Bertran et al. (2019); Wu et al. (2020) focused on suppressing private attributes while preserving explicitly annotated useful attributes. On the other hand, Huang et al. (2018); Malekzadeh et al. (2019); Dave et al. (2022) extended their researches to managing unannotated general features for broader applicability. Furthermore, Chen et al. (2024) summarized and satisified SUIFT, 5 desirable properties of utility-preserving private attributes protection methods. Notably, these state-of-the-art methods are adversarial training based, where their obfuscation module is trained to prevent an adversarial private attributes classifier from making correct inferences.
However, as widely discussed in the neighboring fields of adversarial robustness (Carlini & Wagner, 2017;Athalye et al., 2018;Ilyas et al., 2019;Carlini et al., 2019), generated image/vedio detection (Yu et al., 2019;Wang et al., 2020;Masood et al., 2023), membership inference attacks (Carlini et al., 2022) and gradient inversion in federated learning (Geiping et al., 2020;Huang et al., 2021), a common weakness with adversarial training based methods is that, although the defender can tolerate the jointly-trained adversary, it may be vulnerable to slightly stronger or unseen adversaries. In the private attributes protection context, as we theoretically and empirically demonstrate in Section 3.2, this weakness can indeed cause significant negative impact on the state-of-the-art methods, reducing their practicality in real-world deployment.
To overcome the above weakness, we present a novel method for private attributes protection, called Private Attributes protection with Stochatsic data Substitution (or PASS), which avoids the adversarial training strategy altogether. Specifically, we propose to substitute each input sample of the data sharing system or ML service pipeline with another sample according to a stochastic data substitu- We suppress "sex" as a private attribute, and preserve "Eyeglasses" and "Smiling" as useful attributes. Apart from these attributes, we also preserve general features in facial images, such as "Black hair" and "Young", which are not explicitly annotated in the dataset, but are useful for potential downstream applications. PASS stochastically substitutes the original sample with another sample such that the private attribute cannot be accurately inferred from the substituted sample. On the contrary, the useful attributes and general features are still inferable from the substituted sample.
tion algorithm. This algorithm is parameterized by a neural network and is trained with our novel loss function derived step-by-step from an information-theoretic objective defined for utility-preserving private attributes protection. Benefiting from the theoretical basis, PASS has clear operational boundaries for entangled attributes and can trade-off between privacy and utility controllably. An illustrative use case of PASS is provided in Figure 1.
In summary, our paper's contributions are threefold: 1) We propose PASS, a stochastic data substitution based method that overcomes the common weakness of state-of-the-art private attributes protection methods; 2) We demonstrate theoretically that PASS is rigorously rooted in information theory with desirable properties; and 3) We extensively evaluate PASS on facial images, human activity sensory signals, and human voice recordings to show its broad applicability.
this section cite: ['b49', 'b1', 'b7', 'b32', 'b31', 'b45', 'b6', 'b57', 'b25', 'b39', 'b17', 'b15', 'b9', 'b4', 'b27', 'b10', 'b60', 'b54', 'b40', 'b11', 'b22', 'b26']

Section: Related Works Utility-preserving Private Attributes Protection.
Various studies have been proposed on this topic recently. At-triGuard (Jia & Gong, 2018) proposed to defend against attribute inference attacks using evasion attack on the adversarial classifier. PPDAR (Wu et al., 2020), GAP (Huang et al., 2018) and MaSS (Chen et al., 2024) proposed to minimize private information leakage by making the adversarial classifier unable to make correct predictions. In comparison, ALR (Bertran et al., 2019) and BDQ (Kumawat & Nagahara, 2022) proposed to minimize private information leakage by making the adversarial classifier uncertain about its predictions. Maxent-ARL (Roy & Boddeti, 2019) and MSDA (Malekzadeh et al., 2019) adopted a combination of both above strategies. Different from above, SPAct (Dave et al., 2022) proposed to train the obfuscation model to maximize a contrastive learning loss adversarially with its feature extractor. All of the above methods, except for GAP, proposed to preserve annotated useful attributes by ensuring their predictability. Generalizing the problem, GAP, MSDA, MaSS, and SPAct (Dave et al., 2022) proposed to manage unannotated general features. These works are adversarial training based, resulting in a common weakness elaborated in Section 3.2, motivating the design of PASS.
The field of fairness shares similar problem formulation and designing techniques with private attributes protection (Edwards & Storkey, 2016;Madras et al., 2018;Sarhan et al., 2020;Caton & Haas, 2024). Nevertheless, their training losses are typically derived from fairness metrics, as opposed to the privacy objectives, making them uncomparable with private attributes protection methods.
this section cite: ['b28', 'b57', 'b25', 'b15', 'b6', 'b33', 'b45', 'b39', 'b17', 'b17', 'b20', 'b36', 'b47', 'b12']

Section: Local Differential Privacy (LDP) And Randomized Response.
Differential Privacy requires the randomized mechanism to produce similar distributions when applied to any two neighboring datasets (Dwork et al., 2006), which is typically achieved with addictive noises (Dwork et al., 2014;Abadi et al., 2016;Zhang et al., 2018;Sun et al., 2020). Similarly, Local Differential Privacy casts the same requirement on any two samples in a dataset (Kasiviswanathan et al., 2011;Yang et al., 2023;Arachchige et al., 2019), instead of two neighboring datasets. Local Differential Privacy is mainly achieved with Randomized Response (Warner, 1965;Wang et al., 2016;Chaudhuri & Mukerjee, 2020), which typically involves randomly switching the class of the sensitive attribute when collected. As shown in Appendix B, under certain assumptions, PASS can be viewed as a LDP mechanism-specifically, an extension of randomized response, that is distinguished by its utility preservation requirements and applicability on high-dimensional space.
this section cite: ['b18', 'b19', 'b0', 'b61', 'b52', 'b30', 'b59', 'b3', 'b56', 'b55', 'b13']

Section: k-Anonymity, l-Diversity and t-Closeness (k-l-t privacy).
PASS can also be viewed as an extension to the k-l-t privacy in high-dimensional spaces. k-anonymity aims at thwarting membership inference by ensuring that at least k samples share the same identifiable attributes after obfuscation, and consequently undistinguishable (Samarati & Sweeney, 1998;Qu et al., 2017;Song et al., 2019), while PASS aims at thwarting private attribute inference by ensuring that multiple samples with different private attributes are substituted by each other, and consequently also undistinguishable. ldiversity and t-closeness extended k-anonymity, requiring that these k-samples (equivalent group) have diverse private attributes (Li et al., 2006), preferably with similar distribution as the entire dataset (Sei et al., 2017;Rajendran et al., 2017;Majeed & Lee, 2020), preventing the attacker from guessing the private attribute from obfuscated sample accurately. Similarly, PASS also encourages the private attribute guessed from the substituted sample to have a similar distribution as in the entire dataset.
this section cite: ['b46', 'b42', 'b51', 'b34', 'b50', 'b43', 'b37']

Section: Problem Definition and Motivation
We aim to build a data obfuscation algorithm that could be inserted into a data sharing or processing pipeline to remove certain private attributes from an input sample while preserving its useful attributes as well as general features for downstream tasks. To ensure practicality and soundness, we adhere to the restrictive SUIFT requirements as recently proposed in Chen et al. (2024), as elaborated below.
Sensitivity suppression and Utility preservation are the most basic requirements, suggesting that our method should ensure that private attributes are no longer predictable after obfuscation while the specified useful attributes are maintained. For example, when sharing human voices, the user of our method may remove the speaker's gender information from the audio clip, while preserving the spoken content.
Invariance of sample space dictating that the obfuscated data should remain in the same space as the original data, ensuring the seamless insertion of our method into existing pipelines and data re-usability, which is adopted in most of the recent works (Bertran et al., 2019;Dave et al., 2022;Chen et al., 2024;Malekzadeh et al., 2019).
Feature management without annotation requires preserving the unspecified general features in the data. In the human voice example, general features may include the speaker's accent and age, etc., which are not explicitly specified or labeled in the dataset. This requirement ensures broader usage of our method in the real world, where the downstream tasks are oftentimes unknown, plentiful, and constantly evolving (Huang et al., 2018;Dave et al., 2022;Chen et al., 2024).
Finally, Theoretical basis is also required for the entire proposed framework to enhance soundness and correctness.
this section cite: ['b15', 'b6', 'b17', 'b15', 'b39', 'b25', 'b17', 'b15']

Section: Information-theoretic Problem Definition for Private Attributes Protection
Following the high-level requirements, we formulate our problem into an information-theoretic framework for indepth understanding. For presentation clarity, we use the following notation convention in our paper: uppercase letters (e.g., X, S) denote random variables, and their corresponding lowercase letters (e.g., x, s) the realization of random variables. We use P (•) to denote probability distributions (e.g., P (X)), among which we use P data (•) to indicate that this distribution is purely determined by a dataset and can be readily calculated, and P θ (•) to indicate that this distribution is parameterized by θ and can be calculated readily (e.g., by forward propagation of a neural network). Calligraphic letters (e.g., D) denote datasets.
We consider a multi-attribute dataset with training split D train and test split D test , which can be seen as both drawn from the underlying data distribution P data (X, S, U ), where X is the high-dimensional original input data, S = {S 1 , S 2 , . . . , S M } denotes a set of M user-chosen private attributes annotated on X, and U = {U 1 , U 2 , . . . , U N } denotes a set of N user-chosen useful attributes annotated on X. For example, in the AudioMNIST dataset (Becker et al., 2018), X can denote the high dimensional audio clips, and the user can choose S = {"gender"} as a private attribute to remove, and choose U = {"spoken digit"} as a useful attribute to preserve.
Following the assumptions made in Bertran et al. (2019), we assume that all attributes in S, U follow finite categorical distributions, to ensure a finite mutual information between X and each attribute. We also realistically assume that each attribute is deterministic when X is given, namely P (S i |X) and P (U j |X) are degenerate distributions.
With above definitions and assumptions, we can now formally describe our goal in the information theory framework as to find the optimal data obfuscation algorithm, denoted as P θ (X ′ |X), by solving the following optimization problem
min P θ (X ′ |X) L = M i=1 I(X ′ ; S i )-λ N j=1 I(X ′ ; U j )-µI(X ′ ; X),(1)
where the random variable X ′ denotes the obfuscated data, and our obfuscation algorithm P θ (X ′ |X) is parameterized by θ. I(•, •) denotes Shannon mutual information, and λ and µ are two hyperparameters used to trade-off privacy protection and utility preservation. This optimization objective tries to simultaneously minimize the information leaked for S i in X ′ to remove private attributes, maximize the information of U j in X ′ to preserve useful attributes, and maximize the information of original data X in X ′ to preserve the general features of the data. To summarize the relationship of random variables U, S, X, and X ′ , we illustrate their probabilistic model in Figure 2. The information-theoretic optimization objective of Equation 1 is similar to the objectives used in Bertran et al. (2019); Malekzadeh et al. (2019); Chen et al. (2024). It also closely resembles the optimization objective in Privacy Funnel or Information Bottleneck literature with different focuses (Makhdoumi et al., 2014;Tishby et al., 2000;Alemi et al., 2017;Hjelm et al., 2019).
this section cite: ['b5', 'b6', 'b6', 'b39', 'b15', 'b38', 'b53', 'b2', 'b23']

Section: The Vulnerability of Existing Private Attributes Protection Methods
State-of-the-art utility-preserving private attributes protection methods have demonstrated satisfactory performance in their respective papers in recent years. These methods are based on adversarial training, where the protector trains an adversarial classifier trying to correctly infer the private attribute S i from obfuscated data X ′ , and jointly trains the obfuscation algorithm P θ (X ′ |X) to prevent the adversarial classifier from making correct inferences.
Nevertheless, as widely discussed in the neighboring fields of adversarial robustness (Carlini & Wagner, 2017;Athalye et al., 2018;Ilyas et al., 2019;Carlini et al., 2019), generated image/video detection (Yu et al., 2019;Wang et al., 2020;Masood et al., 2023), membership inference attacks (Carlini et al., 2022) and gradient inversion in federated learning (Geiping et al., 2020;Huang et al., 2021), adversarial training based methods share a common and critical weakness. That is, although their trained defender can safely defend against the jointly-trained adversary, they are vulnerable to potentially stronger or unseen adversaries, which can be obtained by certain tailored adaptive attacking strategies, or simply using longer training time, more computation power, larger datasets, etc.
Despite the existence of many advanced attacking strategies in these neighboring fields listed above, we empirically find out that in the context of private attributes protection, an even simpler and more realistic attacking method can effectively break the state-of-the-art methods, which is formally described below. Given an obfuscation algorithm P θ (X ′ |X), the attacker may repetitively get an original sample and its associated attributes (x, s, u) from P data (X, S, U ); feeds the original sample x into the obfuscation algorithm to get the corresponding obfuscated sample x ′ ∼ P θ (X ′ |X = x); and thus collect a dataset of (x, s, u, x ′ ) tuples. Then, this collected dataset is utilized to train a new adversarial classifier in a supervised manner.
In the rest of this paper, we call this attacking method the Probing Attack. The Probing Attack is realistic in that it does not have assumptions on the training protocol or model structures of the obfuscation algorithm, and can be applied to either deterministic or stochastic obfuscation algorithms.
We conduct a motivational experiment on the Motion Sense dataset to reveal Probing Attack's negative impact on existing methods, where we suppress "gender" and "ID" as private attributes and preserve "activity" as a useful attribute. We use the Normalized Accuracy Gain (NAG) proposed in Chen et al. (2024) as the metric, which is generally proportional to the accuracy of a classifier trained on X ′ for each attribute. The obfuscation is considered better when NAG is lower for private attributes and when NAG is higher for useful attributes. For each private attribute, we calculate the NAG using both 1) the protector's adversarial classifier trained jointly with the obfuscation algorithm, and 2) the attacker's new adversarial classifier trained with a moderate Probing Attack setup after the obfuscation algorithm is deployed. The detailed descriptions of the NAG's definition and experimental settings can be found in Section 5.
We can observe from the results shown in Table 1 that, for private attributes, all methods achieved low NAG with the protector's adversarial classifier but significantly higher NAG with the attacker's adversarial classifier. These experiment results substantiate existing works' severe vulnerability to the simple Probing Attack method, indicating their impracticality in challenging real-world deployment.
To further examine adversarial training based works' vulnerability, we present an information-theoretic interpretation of this phenomenon in Appendix C.
this section cite: ['b9', 'b4', 'b27', 'b10', 'b60', 'b54', 'b40', 'b11', 'b22', 'b26', 'b15']

Section: Our Method: PASS
To build a data obfuscation model that is robust to Probing Attack, we propose PASS, Private Attributes protection with Stochastic data Substitution, which abandons adversarial training but adopts a novel idea of stochastic data substitution. Specifically, we propose first to draw a subset from the training dataset D train , which is denoted as the substitution dataset D substitute . Then, for each input original sample x, instead of transforming x into an obfuscated sample, we propose to strategically replace x with a sample x ′ in the substitution dataset D substitute according to our designed substitution probability, such that the attacker can not correctly infer the private attributes of the original sample x from the substituted sample x ′ , but can still infer the useful attributes and some general features of x from x ′ .
We propose to parameterize our substitution probability with a neural network, and then train it with our novel loss function, which is rigorously derived from our informationtheoretic objective Equation 1. An overview of PASS is shown in Figure 3, and the design details will be elaborated in the next sections.
this section cite: []

Section: Neural Network-based Stochastic Data Substitution
To unify the discussions of PASS and existing private attributes protection methods into the same informationtheoretic framework, we reuse the notation P θ (X ′ |X) to denote substitution probability, where
P θ (X ′ = x ′ |X = x)
denotes the probability of substituting the original sample x with the substitute sample x ′ ∈ D substitute , and θ denotes all the parameters in PASS.
To calculate P θ (X ′ |X), we propose to first input each original input sample x into a neural network to calculate an embedding, denoted as f (x). Then, we obtain a learnable embedding g(x ′ ) for each substitute sample x ′ ∈ D substitute .
Next, we calculate P θ (X ′ |X) based on the cosine similarity between each pair of embeddings f (x) and g(x ′ ) as
P θ (X ′ = x ′ |X = x) = e cos(f (x),g(x ′ ))/τ x ′′ ∈Dsubstitute e cos(f (x),g(x ′′ ))/τ ,(2)
where cos(•, •) is the cosine similarity, τ is the temperature hyper-parameter which is set to 0.01 for all of our experiments. Similar designs can be found widely in contrastive learning literature (Oord et al., 2018;Chen et al., 2020).
this section cite: ['b41', 'b14']

Section: Loss Function Derivation
Ideally, to achieve our goal of utility-preserving private attributes protection, P θ (X ′ |X) should be trained to minimize our information-theoretic optimization objective L defined in Equation 1. Unfortunately, L cannot be accurately esti-mated for each mini-batch during training. Therefore, we propose a novel, fully differentiable loss function L to train P θ (X ′ |X), which is derived step-by-step from L and is theoretically valid in mini-batched based training.
In the rest of this section, we will focus on the formulation of L. For a deeper understanding, please refer to Appendix D for detailed discussions on 1) the theoretical reason why L cannot be accurately estimated, 2) the step-by-step derivation of L and proof of its theoretical validity, and 3) an intuitive explanation of L with a running example.
L can be decomposed into several loss terms as
L = M i=1 LSi -λ N j=1 LUj -µ LX ,(3)
where LSi , LUj and LX are derived from I(X ′ ; S i ), I(X ′ ; U j ), and I(X ′ ; X) respectively, and are responsible for protecting each private attribute S i , preserving each useful attribute U j and preserving general features, respectively. λ and µ are the same trade-off hyperparameters used in L.
We will focus on each of them below.
this section cite: []

Section: Private Attributes Protection.
To remove the information of each private attribute S i from X ′ , we minimize I(X ′ ; S i ) by minimizing LSi , which is derived to be
LSi = -E Pdata(Si) [H(X ′ |S i )] ,(4)
where
H(•|•) denotes conditional Shannon entropy. The conditional distribution P (X ′ |S i ) is calculated as P (X ′ |S i ) = E Pdata(X|Si) [P θ (X ′ |X)] ,(5)
where P data (X|S i ) can be interpreted as all x with each class of private attribute S i in the dataset. Taking expectation over P data (X|S i ) is estimated by averaging over all x with S i in a mini-batch.
this section cite: []

Section: Useful Attributes Preservation.
Extending the notations in Section 3.1, we denote the data distribution of D substitute as P data (X ′ , S ′ , U ′ ), where U ′ = {U ′ 1 , U ′ 2 , . . . , U ′ N } are called substitute useful attributes, which represents the useful attributes directly annotated on X ′ .
To preserve the useful attributes U j , we propose to encourage the substitute useful attributes U ′ j to be similar to the original useful attribute U j . For the AudioMNIST example where we choose "spoken digit" as the useful attribute, for original audio with the spoken digit "1", we try to encourage the substitute audio also to have the spoken digit "1". To achieve this, we propose to minimize LUj defined as:
LUj = log |U j | E Pdata(X,Uj ) -log P (U ′ j = U j |X)) , (6
)
where
U j denotes the support of U j , |•| denotes the cardinal- ity. log |U j | is a coefficient to adjust LUj , so that the scale of LUj matches I(X ′ ; U j ). Taking expectation over P data (X)
is estimated by averaging over a mini-batch. P (U ′ j |X) denotes the expected U ′ j of X, which can be calculated as:
P (U ′ j |X) = E P θ (X ′ |X) P data (U ′ j |X ′ ) .(7)
As shown in Appendix D.2, LUj can also be rigorously derived from I(X ′ ; U j ).
this section cite: []

Section: General Feature Preservation.
To preserve general features, we maximize I(X ′ ; X) by minimizing the conditional entropy of P θ (X ′ |X), which can be written as minimizing the loss term LX :
LX = E Pdata(X) [H(X ′ |X)] ,(8)
where taking expectation over P data (X) is also estimated by averaging over a mini-batch.
Next, we will show that the expectation of L over minibatches is proportional to an upperbound of L defined in Equation 1, suggesting that minimizing L can lead to the minimization of L, confirming its theoretical validity.
Theorem 4.1. For L and L defined in Equation 1 and Equation 3 respectively, for µ ≤ N , we can have
E L + C ≥ L,(9)
where the expectation is taken over all mini-batches, C is a constant defined as
C = (M -µ) log(|D substitute |) -λ N j=1 H(U j ) + λ. (10
)
Please refer to Appendix D.2 for detailed proof. The constant C is independent of model's parameters θ, and can be calculated before training to estimate L during training.
this section cite: []

Section: Training and Inference Procedures
Our design follows a standard two-phase machine learning workflow, consisting of training and inference. During training, we compute P θ (X ′ |X) using Equation 2for a mini-batch of samples from D train , then calculate our loss function according to Equation 3 and update the parameters θ via backpropagation. We summarize the pseudo-code for training in 1.
Once training is complete, inference proceeds by freezing the model parameters θ. For each unseen test-set sample x ∈ D test , we compute the substitution probability P θ (X ′ = x ′ |X = x) and draw a x ′ accordingly to substitute x. The inference pseudo-code is detailed in Algorithm 2.
this section cite: []

Section: Algorithm 1 PASS Training Pseudo-code
Require: training dataset D train , substitution dataset D substitute , model parameters θ Ensure: trained model parameters θ 1: while not reached max number of epochs do
2: sample a mini-batch from D train 3: compute f (x) for x in mini-batch 4: compute g(x ′ ) for x ′ ∈ D substitute 5: compute P θ (X ′ = x ′ |X = x) using f (x) and g(x ′ ) (Equation 2) 6: compute L using P θ (X ′ = x ′ |X = x) (Equation 3) 7: update θ with ∂ L ∂θ 8: end while 9: return θ Algorithm 2 PASS Inference Pseudo-code Require: original sample x, substitution dataset D substitute , model parameters θ Ensure: substituted sample x ′ 1: compute f (x) 2: compute g(x ′ ) for x ′ ∈ D substitute 3: compute P θ (X ′ = x ′ |X = x) using f (x) and g(x ′ ) (Equation 2) 4: draw x ′ ∈ D substitute according to P θ (X ′ = x ′ |X = x) 5: return x ′
PASS has lower training and inference computational overhead compared with state-of-the-art methods, because its P θ (X ′ |X) is parameterized as an embedding extraction followed by a cosine similarity, whereas the state-of-the-art methods typically parameterize P θ (X ′ |X) as an end-toend data reconstruction neural network (Ronneberger et al., 2015;Cao et al., 2022), which may require more computation.
this section cite: ['b44', 'b8']

Section: Theoretical Analysis for Entangled Attributes
Intuitively, when the useful attributes and private attributes are highly correlated or entangled, we need to sacrifice the utility of useful attributes to an extent to achieve private attributes protection. To theoretically describe this phenomenon in the information theory framework, and to provide an estimation of the extent of the utility sacrifice, we present the following theorem, revealing that both I(X ′ ; U j ) and I(X ′ ; X) are bounded by I(X ′ ; S i ) and cannot be arbitrarily large. Theorem 4.2. For any i ∈ {1, 2, . . . , M }, any J ⊆ {1, 2, . . . , N }, and U = {U j |j ∈ J }, we can have
j∈J I(X ′ ; U j ) ≤ I(X ′ ; S i ) + H(U|S i ) + C(U), (11)
where C(•) denotes the total correlation, H(•|•) denotes conditional Shannon entropy. We can also have
I(X ′ ; X) ≤ I(X ′ ; S i ) + H(X|S i ).(12)
Please refer to Appendix A for detailed proof. This theorem can be viewed as an extension of the analysis in Chen et al. (2024). The terms H(U|S i ), C(U), and H(X|S i ) can all be calculated before training to estimate the extent of utility sacrifice, and to track the training progress.
this section cite: ['b15']

Section: Theoretical Interpretation within Local Differential Privacy Framework
As elaborated in Appendix B, under certain assumptions, our information-theoretic problem formulation can also be interpreted within the framework of local differential privacy (LDP) (Yang et al., 2023). In this context, PASS can be viewed as an LDP mechanism, more specifically, an extension of the classical randomized response methods. The distinctiveness of PASS from classical methods lies in its capability to operate over high-dimensional data domains, its explicit focus on utility preservation, and its foundation in information-theoretic principles.
this section cite: ['b59']

Section: Experiments

this section cite: []

Section: Experimental Setup
Datasets. We thoroughly evaluated PASS on three multiattribute benchmark datasets, each representing a different application of a different modality. These datasets include AudioMNIST (Becker et al., 2018), containing recordings of human voices; Motion Sense (Malekzadeh et al., 2019), consisting of human activity sensory signals; and CelebA (Liu et al., 2015), containing facial images. More detailed descriptions of datasets can be found in Appendix E.1.
Baselines. We primarily compare PASS against six stateof-the-art baseline methods: ALR (Bertran et al., 2019), GAP (Huang et al., 2018), MSDA (Malekzadeh et al., 2019), BDQ (Kumawat & Nagahara, 2022), PPDAR (Wu et al., 2020), and MaSS (Chen et al., 2024), whose detailed descriptions are presented in Section 2.
this section cite: ['b5', 'b39', 'b35', 'b6', 'b25', 'b39', 'b33', 'b57', 'b15']

Section: Evaluation Metrics and Default Probing Attack's Setting.
To measure the performance of the obfuscation on imbalanced datasets, we adopt the metric Normalized Accuracy Gain (NAG) proposed by Chen et al. (2024), which can ensure that all attributes are measured on the same scale. The NAG for private attribute S i is defined as
NAG(S i ) = max 0, Acc(S i ) -Acc guessing (S i ) Acc no suppr. (S i ) -Acc guessing (S i ) ,(13)
where Acc(S i ) is the accuracy of a classifier trained on the obfuscated data and the ground truth label of attribute S i . Acc guessing (S i ) is the accuracy of the majority classifier for S i , serving as a lower bound of the Acc(S i ). And Acc no suppr. (S i ) is the accuracy of a classifier trained on the original data and the ground truth label of attribute S i , serving as an upper bound of Acc(S i ).
A critical difference between our evaluation protocol and baselines lies in that, we use the attacker's adversarial classifier trained with the Probing Attack to calculate NAG and thus measure the obfuscation performance, enhancing the practicality and realisticity of the evaluation. In the default Probing Attack's setting, the attacker has the API of the trained obfuscation model P θ (X ′ |X) and its training dataset. The attacker adopts a medium-sized neural network based adversarial classifier and trains it on the obfuscated data using the Probing Attack.
To evaluate whether our method and baselines can preserve the data's general features, we hide some attributes during training and only reveal them during evaluation to verify if they can be preserved. We call these attributes hidden useful attributes and denote them as F k for k ∈ {1, . . . , K}.
NAG for each useful attribute NAG(U j ) and each hidden useful attribute NAG(F k ) is defined and calculated in the same way as NAG(S i ). Higher NAG suggests that this attribute's information is well preserved in X ′ . Therefore, the performance of an obfuscation method is considered better when each NAG(S i ) is lower and when each NAG(U j ) and each NAG(F k ) is higher.
To facilitate a fair comparison between different methods, we propose a novel scalar metric, mean Normalized Accuracy Gain (mNAG), to measure the trade-off between private attributes protection and useful attributes preservation in a comprehensive way. mNAG is defined as the average of the NAG for useful attributes (including hidden useful attributes), minus the average of the NAG for private attributes, Table 2. Comparison of the NAG between PASS and baselines on AudioMNIST. We suppress "gender" as a private attribute, while preserving "digit" as a useful attribute. We take "accent", "age", and "ID" as hidden useful attributes to evaluate general feature preservation. which can be formally written as
mNAG = 1 M + K ( M j=1 NAG(U j ) + K k=1 NAG(F k )) - 1 N N i=1 NAG(S i ).(14)
We report the NAG and mNAG in the main paper, while the corresponding accuracy can be found in Appendix F.
Hyperparameters. Unless otherwise specified, we set λ = N M and µ = 0.2N throughout our experiments to balance private attributes protection, useful attributes preservation, and general feature preservation. The substitute dataset is constructed by randomly sampling 4096 data points from the training dataset. All the experiments in this paper are conducted with three random seeds and then aggregated. Other hyperparameters for neural network structures, training configurations, and datasets can be found in Appendix E.
this section cite: ['b15']

Section: Evaluation on Human Voice Recording
We begin with the task of removing gender information from human voice recordings on AudioMNIST dataset, where we suppress "gender", and preserve "digit". We take "accent", "age", "ID" as hidden useful attributes to evaluate general feature preservation. As shown in Table 2, PASS exhibited 0 NAG on "gender" and a significantly higher mNAG than all baselines, which strongly substantiated PASS's capability on utility-preserving private attributes protection, and PASS's tolerance to the Probing Attack.
Furthermore, we conduct an ablation study to verify if PASS is robust to different combinations of useful and private attributes. The combinations and their corresponding results are presented in Table 3, demonstrating that PASS can consistently achieve a high mNAG for all combinations.
When useful attributes and private attributes are highly entangled (e.g., when we suppress "gender", "accent","age", but preserve "ID"), PASS manages to find a satisfactory compromise between privacy and utility automatically.
We also conduct four more ablation studies on AudioM-NIST by varying 1) the coefficient λ, 2) the coefficient µ, 3) the number of samples in the substitute dataset |D substitute |, and 4) the distribution of the substitute dataset. As shown in Appendix F.1, PASS consistently maintains high performance across all different settings, showing robustness to hyperparameter changes.
this section cite: []

Section: Evaluation on Human Activity Sensory Data
In our subsequent study, we apply PASS to the task of anonymized activity recognition on Motion Sense dataset, where we suppress "gender" and "ID" attributes while preserving the "activity" attribute. On "activity", in addition to the standard NAG, we also report the NAG calculated with an un-finetuned classifier, which is only pre-trained on original data X, without finetuning on X ′ . The results, presented in Table 4, show that PASS achieves NAG of 0 for private attributes and a much higher mNAG than all baselines, which substantiated the effectiveness of PASS. Besides, PASS also achieved the highest NAG on "activity" with un-finetuned classifier, which shows that PASS can be plugged into an existing ML pipeline to protect private attributes without necessarily altering the downstream models. We visualize the results of data substitution in this experiment using Table 5. Comparison of the NAG between PASS and baselines on CelebA. We suppress "Male" as a private attribute, while preserving "Smiling" and "Young" as useful attributes, and we take "Attractive," "Mouth Slightly Open," and "High Cheekbones" as hidden useful attributes to evaluate general feature preservation.
Method NAG (%) mNAG (%) (↑) Male (↓) Smiling (↑) Young (↑) Attractive (↑) Mouth Slightly Open (↑) High Cheekbones (↑) ADV 99.9±0.1 98.8±0.1 97.0±0.9 94.6±0.4 99.1±0.1 97.0±0.5 -2.6±0.2 GAP 83.0±1.1 75.9±1.3 45.4±3.0 77.6±1.1 61.1±2.1 75.6±0.7 -15.9±2.3 MSDA 91.6±0.7 99.8±0.2 92.4±2.4 89.9±1.0 91.8±0.8 95.7±1.1 2.3±0.8 BDQ 99.7±0.1 98.8±0.2 96.3±0.8 94.1±0.6 98.9±0.4 97.0±0.3 -2.7±0.2 PPDAR 99.7±0.1 98.9±0.3 97.2±1.2 94.4±0.6 99.0±0.1 97.0±0.4 -2.4±0.3 MaSS 96.9±0.1 97.2±0.2 86.2±1.4 90.6±0.3 97.6±0.2 94.6±0.4 -3.7±0.4 PASS 4.9±0.5 98.3±0.1 78.6±0.8 58.1±2.8 67.0±0.8 86.7±0.3 72.9±0.2 confusion matrices, as shown in Appendix F.2.
We also experimented to show that PASS can safely tolerate the Probing Attack when the attacker has more data than the protector. Detailed results and analysis are presented in Appendix F.2.
this section cite: []

Section: Evaluation on Facial Images
We then extend the application of PASS to removing gender information from facial images on CelebA dataset, where we aim to suppress the "Male" as private attribute while preserving "Smiling", "Young" as useful attributes, and evaluating the general features preservation by taking "Attractive", "Mouth Slightly Open", "High Cheekbones" as hidden useful attributes. As displayed in Table 5, PASS achieved a near 0 NAG on "Male", and a much higher mNAG than all the baselines, which highlights PASS's efficacy on images.
We further compare PASS with four additional DP-based baselines: Laplace Mechanism (Additive Noise) (Dwork et al., 2006), DPPix (Fan, 2018), Snow (John et al., 2020), and DP-Image (Xue et al., 2021). As shown in Table 17 in Appendix F.3, these methods show limited performance because they primarily aim to prevent membership inference, which differs from our goal of protecting specific private attributes while preserving utility.
In addition, we conduct an ablation study showing PASS's robustness on different combinations of useful and private attributes on CelebA. And we also experiment to reveal PASS's superiority over the baseline SPAct (Dave et al., 2022). Please refer to Appendix F.3 for results and analyses.
this section cite: ['b18', 'b21', 'b29', 'b58', 'b17']

Section: Conclusion
In this paper, we theoretically and empirically demonstrate that the common weakness of adversarial training has a noticeable negative impact on state-of-the-art private attributes protection methods. To address this, we propose PASS, a stochastic data substitution based method rooted rigorously in information theory, that overcomes the above weakness. The evaluation of PASS on three datasets substantiates PASS's broad applicability in various applications.
this section cite: []

Section: References
Ref_id:b0 Title: Deep learning with differential privacy Year: (2016)
Ref_id:b1 Title: Artificial intelligence and machine learning in finance: A bibliometric review Year: (2022)
Ref_id:b2 Title: Deep variational information bottleneck Year: (2017)
Ref_id:b3 Title: Local differential privacy for deep learning Year: (2019)
Ref_id:b4 Title: Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples Year: (2018)
Ref_id:b5 Title: Interpreting and explaining deep neural networks for classification of audio signals Year: (2018)
Ref_id:b6 Title: Adversarially learned representations for information obfuscation and inference Year: (2019)
Ref_id:b7 Title: A review of privacy-preserving techniques for deep learning Year: (2020)
Ref_id:b8 Title: Swin-unet: Unet-like pure transformer for medical image segmentation Year: (2022)
Ref_id:b9 Title: Towards evaluating the robustness of neural networks Year: (2017)
Ref_id:b10 Title: On evaluating adversarial robustness Year: (2019)
Ref_id:b11 Title: Membership inference attacks from first principles Year: (2022)
Ref_id:b12 Title: Fairness in machine learning: A survey Year: (2024)
Ref_id:b13 Title: Randomized response: Theory and techniques Year: (2020)
Ref_id:b14 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b15 Title: Mass: Multi-attribute selective suppression for utility-preserving data transformation from an information-theoretic perspective Year: (2024)
Ref_id:b16 Title: Elements of information theory Year: (1999)
Ref_id:b17 Title: Self-supervised privacy preservation for action recognition Year: (2022)
Ref_id:b18 Title: Calibrating noise to sensitivity in private data analysis Year: (2006)
Ref_id:b19 Title: The algorithmic foundations of differential privacy Year: (2014)
Ref_id:b20 Title: Censoring representations with an adversary Year: (2016)
Ref_id:b21 Title: Image pixelization with differential privacy Year: (2018)
Ref_id:b22 Title: Inverting gradients-how easy is it to break privacy in federated learning? Advances in neural information processing systems Year: (2020)
Ref_id:b23 Title: Learning deep representations by mutual information estimation and maximization Year: (2019)
Ref_id:b24 Title: Selfsupervised speech representation learning by masked prediction of hidden units Year: (2021)
Ref_id:b25 Title: Generative adversarial privacy Year: (2018)
Ref_id:b26 Title: Evaluating gradient inversion attacks and defenses in federated learning Year: (2021)
Ref_id:b27 Title: Adversarial examples are not bugs, they are features Year: (2019)
Ref_id:b28 Title: A practical defense against attribute inference attacks via adversarial machine learning Year: (2018)
Ref_id:b29 Title: Let it snow: Adding pixel noise to protect the user's identity Year: (2020)
Ref_id:b30 Title: What can we learn privately? Year: (2011)
Ref_id:b31 Title: Automatic speech recognition using advanced deep learning approaches: A survey Year: (2024)
Ref_id:b32 Title: Artificial intelligence in healthcare: review, ethics, trust challenges & future research directions Year: (2023)
Ref_id:b33 Title: Privacy-preserving action recognition via motion difference quantization Year: (2022)
Ref_id:b34 Title: Privacy beyond k-anonymity and l-diversity Year: (2006)
Ref_id:b35 Title: Loshchilov, I. and Hutter, F. Decoupled weight decay regularization Year: (2015-12)
Ref_id:b36 Title: Learning adversarially fair and transferable representations Year: (2018)
Ref_id:b37 Title: Anonymization techniques for privacy preserving data publishing: A comprehensive survey Year: (2020)
Ref_id:b38 Title: From the information bottleneck to the privacy funnel Year: (2014)
Ref_id:b39 Title: Mobile sensor data anonymization Year: (2019)
Ref_id:b40 Title: Deepfakes generation and detection: Stateof-the-art, open challenges, countermeasures, and way forward Year: (2023)
Ref_id:b41 Title: Representation learning with contrastive predictive coding Year: (2018)
Ref_id:b42 Title: Big data set privacy preserving through sensitive attribute-based grouping Year: (2017)
Ref_id:b43 Title: A study on k-anonymity, l-diversity, and t-closeness techniques Year: (2017)
Ref_id:b44 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b45 Title: Mitigating information leakage in image representations: A maximum entropy approach Year: (2019)
Ref_id:b46 Title: Protecting privacy when disclosing information: k-anonymity and its enforcement through generalization and suppression Year: (1998)
Ref_id:b47 Title: Fairness by learning orthogonal disentangled representations Year: (2020)
Ref_id:b48 Title: Facenet: A unified embedding for face recognition and clustering Year: (2015)
Ref_id:b49 Title: The role of artificial intelligence in healthcare: a structured literature review Year: (2021)
Ref_id:b50 Title: Anonymization of sensitive quasi-identifiers for ldiversity and t-closeness Year: (2017)
Ref_id:b51 Title: A new method of privacy protection: random k-anonymous Year: (2019)
Ref_id:b52 Title: Human action image generation with differential privacy Year: (2020)
Ref_id:b53 Title: The information bottleneck method Year: (2000)
Ref_id:b54 Title: Cnn-generated images are surprisingly easy to spot... for now Year: (2020)
Ref_id:b55 Title: Using randomized response for differential privacy preserving data collection Year: (2016)
Ref_id:b56 Title: Randomized response: A survey technique for eliminating evasive answer bias Year: (1965)
Ref_id:b57 Title: Privacypreserving deep action recognition: An adversarial learning framework and a new dataset Year: (2020)
Ref_id:b58 Title: Dp-image: Differential privacy for image data in feature space Year: (2021)
Ref_id:b59 Title: Local differential privacy and its applications: A comprehensive survey. Computer Standards & Interfaces Year: (2023)
Ref_id:b60 Title: Attributing fake images to gans: Learning and analyzing gan fingerprints Year: (2019)
Ref_id:b61 Title: Differentially private releasing via deep generative model (technical report) Year: (2018)
