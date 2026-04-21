Title: Long-Tailed Recognition via Information-Preservable Two-Stage Learning
Abstract: The imbalance (or long-tail) is the nature of many real-world data distributions, which often induces the undesirable bias of deep classification models toward frequent classes, resulting in poor performance for tail classes. In this paper, we propose a novel two-stage learning approach to mitigate such a majority-biased tendency while preserving valuable information within datasets. Specifically, the first stage proposes a new representation learning technique from the information theory perspective. This approach is theoretically equivalent to minimizing intraclass distance, yielding an effective and well-separated feature space. The second stage develops a novel sampling strategy that selects mathematically informative instances, able to rectify majority-biased decision boundaries without compromising a model's overall performance. As a result, our approach achieves state-of-the-art performance across various long-tailed benchmark datasets. Our code is available at https://github.com/fudong03/BNS_IPDPP.

Section: Introduction
Class imbalance naturally arises in real-world scenarios, spanning across such wide applications as online transactions, medical diagnoses, social networks spam detection, among many others. Such data in real scenarios usually follows a long-tailed distribution, i.e., a few head classes dominate the entire dataset while some tail classes account for only a small portion. When encountered with such long-tailed data, deep neural networks (DNNs) suffer from majority-biased decision boundaries [31,53,24,57,13,41,16,42], undesirably favoring frequent classes and leading to poor performance in tail classes. Misclassifying tail classes may yield catastrophic consequences, e.g., failing to identify lung cancer from millions of biomedical images may result in fatalities. To this end, building functional classifiers with unbiased decision boundaries when tackling the long-tailed data is critical but remains open.
So far, the mainstream strategies addressing long-tailed recognition primarily fall into two categories: one-stage learning and two-stage learning methods. One-stage learning approaches include: i) reweighting strategies [38,10,3,9,47], which prevent dominant head classes from overwhelming the training process by reducing the weight of loss functions for majority samples; and ii) sampling techniques [4,21,58,25,15], which create balanced training subsets either by downsampling majority samples or by synthesizing minority samples. However, such strategies struggle to achieve effective performance across both head and tail classes due to their limited representation learning capabilities. On the other hand, two-stage learning methods [28,51,11,62,30,39,34,35,27] decouple the training process into "representation learning" and "classification" stages. The former stage focuses on learning effective and generalizable feature spaces, while the latter stage aims to rectify majority-biased decision boundaries caused by highly skewed data distributions.
this section cite: ['b30', 'b52', 'b23', 'b56', 'b12', 'b40', 'b15', 'b41', 'b37', 'b9', 'b2', 'b8', 'b46', 'b3', 'b20', 'b57', 'b24', 'b14', 'b27', 'b50', 'b10', 'b61', 'b29', 'b38', 'b33', 'b34', 'b26']

Section: Preliminary
Mutual Information (MI) refers to a measure of the mutual dependence between two random variables. It quantifies the amount of information that knowing one random variable reduces the uncertainty of the other variable. Given two jointly discrete random variables X and Y, for their mutual information, we have:
M I(X, Y) = x∈X y∈Y pXY(x, y) log pXY(x, y) pX(x)pY(y) .(1)
Here, p XY (x, y) represents the joint probability of X and Y, while p X (x) and p Y (y) are the marginal probabilities of X and Y, respectively.
Information Content (IC), also known as Shannon information or self-information, measures the degree of "surprise" associated with a particular outcome. Given a ground set X, for any event x ∈ X with probability p(x), the information content I(•) is defined as follows:
I(x) = -log [p(x)] .(2)
By definition, information content has three key properties: i) an event with 100% probability yields no information; ii) less probable events are more surprising and contain more information; and iii) given a set of independent events, its total self-information equals the sum of each event's individual self-information, i.e., I(X) = x∈X I(x).
4 Our Approaches
this section cite: []

Section: Problem Statement
Consider a set of N samples for training, i.e., X = {(x i , y i )} N i=1 , where data point x i is labeled with y i . Suppose the training set has C classes, i.e., y ∈ {1, 2, . . . , C}, and let N c (c = 1, 2, • • • , C) be the number of training data for the c-th class. Without loss of generality, we consider all classes to be sorted in the decreasing order, i.e., N c ≥ N c+1 . Naturally, ∀N c , we have N c ≥ N C . Here, we consider a long-tailed setting, i.e., N 1 ≫ N C , indicating that head and tail classes are highly skewed.
Deep neural networks (DNNs) struggle with such long-tailed data, where they perform poorly on tail classes. This can be attributed to two primary factors. First, when using conventional representation learning methods, the inherent "label bias" in long-tailed data leads to poor feature spaces. Second, majority class samples tend to dominate the training process, resulting in biased decision boundaries that unfairly favor the head classes. These two factors call for the development of innovative solutions in both representation learning and classification stages.
this section cite: []

Section: Stage 1: Representation Learning via Balanced Negative Sampling
Our first stage aims to learn an effective and well-separated feature space for long-tailed recognition. We resort to maximizing mutual information between instances sharing the same label-a process mathematically equivalent to minimizing intra-class distances.
Our Objective. Given any image of the ground set x ∈ X, we employ data augmentation modules to transform it into two different views of the same sample, denoted as x i ∈ X Q and x j ∈ X V . Let f θ (•) be a DNN parameterized by θ, used for encoding feature representations. Here, we regard Q(X Q ; θ) and V (X V ; θ) as feature spaces corresponding to X Q and X V , respectively. In this work, our goal is to learn high-quality feature representations for imbalanced classification by maximizing the mutual information between two feature spaces: arg max θ M I(Q(XQ; θ), V (XV ; θ)).
(
In the rest of the paper, we abbreviate Q(X Q ; θ) and V (X V ; θ) as Q and V , respectively. Then, given any images of x i ∈ X Q and x j ∈ X V , their representations can be expressed respectively as q i ∈ Q and v j ∈ V , where q i = f θ (x i ) and v j = f θ (x j ).
this section cite: []

Section: CL-based Representation Learning.
However, directly maximizing the mutual information between two representation spaces Q and V is computationally intractable. Worse still, prior studies [60,39] have demonstrated that long-tailed data inherently causes "label bias", resulting in poor representations for tail classes. In this work, we reformulate Eq. ( 3) within the framework of contrastive learning (CL), enabling the efficient optimization of our objective. Meanwhile, employing CL-based techniques [5,23,29,37,27] can also effectively mitigate the "label bias" issue, as highlighted in prior studies [60,39].
Specifically, our approach leverages a binary classifier to distinguish the target image from noise samples, drawing inspiration from Noise Contrastive Estimation (NCE) [18]. Given an anchor image x i ∈ X Q , we have its corresponding target image x + j,i ∈ X V . Here, x i and x + j,i are two augmented versions of the same image. Meanwhile, we sample n noise images (having different labels with x i ) from the dataset X V , denoted as {x - j } n j=1 . Regarding their representations, we have one positive pair (q i , v + j,i ) and n negative pairs {(q i , v - j )} n j=1 . Therefore, there is a 1 n+1 chance to pick the positive pair and a n n+1 chance to pick the negative pair. Let p(•) be the joint probability of Q and V , and g(•) represent a binary classifier, where its output d = 1 and d = 0 denote the positive and negative pair, respectively. Then, we have:
g(qi, vj | d) = 1 n+1 p(qi, v + j,i ), d = 1 n n+1 p(qi, v - j ), d = 0 . (4
)
Considering the positive pair only, we arrive at:
g(qi, vj | d = 1) = p(qi, v + j,i ) p(qi, v + j,i ) + n × p(qi, v - j ) .(5)
We assume that the distributions between different classes are independent. Then, for any negative pair (q i , v - j ), we have p(q i , v - j ) = p(q i )p(v - j ). As such, we can rewrite Eq. ( 5) as follows:
g(qi, vj | d = 1) = p(qi, v + j,i ) p(qi, v + j,i ) + n × p(qi)p(v - j ) .(6)
Taking the logarithm of Eq. ( 6) and rearranging the terms (see Appendix A.1 for detailed derivation), we obtain:
log g(qi, vj | d = 1) ≤ log p(qi, v + j,i ) p(qi)p(v - j ) -log n.(7)
Taking the expectation of p(q i , v + j,i ) on both sides, we have:
E p(q i ,v + j,i ) log p(qi, v + j,i ) p(qi)p(v - j ) ≥ E p(q i ,v + j,i ) log g(qi, vj | d = 1) + log n.(8)
Combining Eq. ( 1), Eq. ( 3), and Eq. ( 8), we have:
M I (Q, V ) maximize MI ≥ E p(q i ,v + j,i ) log g(qi, vj | d = 1) maximize lower bound + log n.(9)
Here, log n is a constant, indicating that maximizing the lower bound in Eq. ( 9) is equivalent to maximizing the mutual information between the two feature spaces.
this section cite: ['b59', 'b38', 'b4', 'b22', 'b28', 'b36', 'b26', 'b59', 'b38', 'b17']

Section: Balanced Negative Sampling (BNS).
Inspired by prior studies [18,48,52], we train a logistic regression classifier to maximize the lower bound in Eq. ( 9). However, log g(q i , v j | d = 1) is computationally intractable. To address this issue, we approximate the classifier's output with sigmoid function σ(•), expressed as below:
g(qi, vj | d) = σ( q ⊤ i v j τ ), d = 1 σ(- q ⊤ i v j τ ), d = 0 . (10
)
Here, τ is the temperature parameter that controls the sharpness of the similarity scores. As such, our NS-based contrastive learning, designed for learning high-quality representations, is formulated as follows:
LNS = -log σ( q ⊤ i v + j,i τ ) + n j=1 log σ(- q ⊤ i v - j τ ) .(11)
Our NS-based contrastive learning, i.e., Eq. ( 11), can mitigate "label bias" inherent to long-tailed data, yielding a higher quality of representations. However, it is ineffective in learning a well-separated representation space. This is because positive pairs of head classes dominate the representation space.
We then propose a novel Balanced Negative Sampling (BNS) to learn an effective and well-separated representation space. That is, for a given anchor image x i ∈ X Q , we sample an additional set of m images from X Q that share the same label as x i , denoted as {x k } m k=1 . Let q k ∈ Q + i,m denote representations for the addition set of images. Then, we have m + 1 positive pairs {(q * , v + j,i ) | q * ∈ {q i } ∪ Q + i,m } and n(m + 1) negative pairs {(q * , v - j ) | q * ∈ {q i } ∪ Q + i,m and j = 1, 2, . . . , n}. Mathematically, our BNS technique can be expressed as:
LBNS = - 1 m + 1    q * ∈{q i } ∪ Q + i,m log σ( q ⊤ * v + j,i τ ) + q * ∈{q i } ∪ Q + i,m n j=1 log σ(- q ⊤ * v - j τ )    .(12)
In practice, m is set to a small value due to the limited number of samples in minority classes. Eq. ( 12) effectively enhances the quality of feature representations by maximizing the mutual information shown in Eq. ( 3). This maximization of mutual information directly corresponds to minimizing intra-class distances, as stated next. Q and X c V , respectively. Then, any pair of q c i ∈ Q c and v c j ∈ V c is a positive pair. Let M I(•) and D(•) respectively denote the mutual information and a distance metric, we have:
max M I(Q c , V c ) ∝ min D(Q c , V c ),(13)
where D(Q c , V c ) can be considered as the intra-class distance because they have the same label.
The proof of Theorem 4.1 is deferred to Appendix A.2.
Instance-Level and Class-Level Semantics. An effective representation space must capture two key aspects: instance-level semantics to ensure high-quality feature representations and class-level semantics to achieve well-separated feature spaces. To understand how our BNS technique achieves this, we decompose Eq. ( 12) as follows:
LBNS = - 1 m + 1                log σ( q ⊤ i v + j,i τ ) + n j=1
log σ(-
q ⊤ i v - j τ ) instance-level + q k ∈Q + i,m   log σ( q ⊤ k v + j,i τ ) + n j=1
log σ(-
q ⊤ k v - j τ )   class-level                . (14
)
According to Theorem 4.1, our BNS approach naturally minimizes distances at both the instance and class levels. Specifically, the pair (q i , v + j,i ) originates from the same instance, while the pairs (q k , v + j,i ) are from the same class. This dual-level minimization naturally encourages the emergence of both instance-level and class-level semantics within the representation space, leading to improved representation quality and better separation of feature spaces.
this section cite: ['b17', 'b47', 'b51']

Section: Stage 2: Information-Preservable Determinantal Point Process
Next, we propose a new sampling solution, namely Information-Preservable Determinantal Point Process (IP-DPP), aiming to rectify majority-biased classification decision boundaries while maintaining the model's overall performance. Specifically, our approach builds on the Determinantal Point Process (DPP) [33], a stochastic process that captures global negative correlations, as outlined below. Definition 4.2 (Determinantal Point Process). Given a ground set X with N items, a point process P in this ground set is a distribution over discrete and finite subsets of X. Let K ∈ R N ×N be a real, symmetric marginal kernel matrix indexed by the elements of X. A point process P is called a DPP only if, for every random subset Y ⊆ X drawn according to P, we have:
P(Y) = det(K Y ),(15)
where K Y = [K ij ] i,j∈Y is the principle submatrix of K, indexed by elements of Y.
Since P is a probability measurement, i.e., 0 ≤ P ≤ 1, the marginal kernel matrix K must satisfy specific structural properties, as stated next.
this section cite: ['b32']

Section: Remark 4.3 (Properties of Marginal Kernel Matrix).
The marginal kernel matrix K for a DPP must satisfy: i) K is a positive semidefinite matrix; and ii) All eigenvalues of K are bounded in the interval [0, 1], i.e., 0 ⪯ K ⪯ I.
However, it is very difficult to construct a DPP through the marginal kernel matrix K in real longtailed settings. In this work, we follow the prior study [33] by constructing a DPP based on the L-ensemble framework [2]. Specifically, consider an image x i ∈ X with ground truth label y i , let p(i) = p ϕ (y i |x i ) denote the probability of correctly predicting y i given x i , where the classifier is parameterized by ϕ. For any two distinct elements i, j ∈ X, let p(i, j) denote the joint probability of correctly classifying both elements. Assuming independence between classifications, we have p(i, j) = p(i)p(j). Let S be a N × N matrix, where each element S i,j is defined as follows:
Si,j = p(i)p(j) N , i ̸ = j 1 -k̸ =j p(k)p(j) N , i = j . (16
)
As such, S is a symmetric stochastic matrix where each row (or column) sums to 1. The symmetric stochastic matrix S is positive semi-definite, and all its eigenvalues are bounded in [0, 1], as stated in Lemmas 4.4 and 4.5, respectively. S ∈ R N ×N is a symmetric stochastic matrix, where each row (or column) sums to 1. Then, we have:
v ⊤ Sv ≥ 0, ∀v ∈ R N .(17)
In other words, S is positive semi-definite. Lemma 4.5. (Bounds on Eigenvalues) Let {λ i } N i=1 be the eigenvalues of the symmetric stochastic matrix S ∈ R N ×N , we have:
0 ≤ λi ≤ 1, ∀λi.(18)
The proofs of Lemmata 4.4 and 4.5 are deferred to Appendix A.3 and Appendix A.4, respectively.
As such, the symmetric stochastic matrix S satisfies the two properties stated in Remark 4.3. Hence, it can be used to construct a DPP through L-ensemble, expressed as below:
P S (Y) = det(SY) det(S + I) , (19
)
where I is an N × N identity matrix. Since P S (Y) is a probability measurement, it needs to be bounded in [0, 1]. The DPP defined in Eq. ( 19) is valid, i.e., 0 ≤ P S (Y) ≤ 1, as outlined below. Theorem 4.6. (Bounded Determinant Probability Measurement) Let X be a ground set with N items and S ∈ R N ×N denote a symmetric stochastic matrix, indexed by elements in X. Here, S is positive semi-definite and satisfies 0 ⪯ S ⪯ I, where I is the N × N identity matrix. Let S Y denote the principal submatrix of S corresponding to Y, for any subset Y ⊆ X, the following holds:
0 ≤ det(S Y ) det(S + I) ≤ 1.(20)
In other words, P S (Y) = det(SY) det(S+I) defines a valid probability measurement.
The proof of Theorem 4.6 is deferred to Appendix A.5.
this section cite: ['b32', 'b1']

Section: Information-Preservable Property.
In the long-tailed settings, our DPP method defined in Eq. ( 19) can effectively preserve valuable information, as discussed next. Remark 4.7 (Information-Preserving Sampling Principle). Let I(x) = -log[p(y|x)] denote the information content of item x relevant to its correct classification. P S (Y ∪ {x}) denotes the probability that item x is sampled by our DPP approach, which prioritizes sampling elements with higher information content, as expressed by:
P S (Y ∪ {x}) ∝ I(x) .(21)
Here, we use a simple example to illustrate how Remark 4.7 holds. Let A = {i, j} be a subset of X sampled by our DPP approach. For the given ground set, det(S + I) is a constant. Then, we arrive at (see Appendix A.6 for details):
P S (A) = det(S A ) det(S + I) ∝ det(S A ) = 1 -p(i) • p(j).(22)
Algorithm 1 IP-DPP 1: Input: a ground set X = {xi} N i=1 , its symmetric stochastic matrix S, and sample size k 2: Initialize: standard basis vectors {ei} N i=1 and pairs of orthonormal eigenvalues and eigenvectors {(λi, vi)} N i=1 for S 3: V ← ∅ 4:
for i = 1, 2, • • • , N do 5: if u ∼ U (0, 1) < λ i λ i +1 then 6: V ← V ∪ {vi} 7:
k ← k -1 8: end if 9: if k = 0 then 10: break 11: end if 12: end for 13:
Y ← ∅ 14: while |V | > 0 do 15: for i = 1, 2, • • • , N do 16: p(i) ← 1 |V | v∈V (v ⊤ ei) 2 17:
end for 18:
i * ← arg max i p(i) 19: Y ← Y ∪ {xi * } 20: V ← V ⊥ // Update V
to an orthonormal basis for the subspace orthogonal to ei * 21: end while 22: Return: a subset Y Therefore, we obtain P S ({i, j}) ∝ -p(i) • p(j). In this work, we have p(i) = p(y i |x i ), implying that images less likely to be correctly classified are more likely to be sampled by our DPP approach. According to information content (see Eq. ( 2) for details), we have I(
x i ) = -log[p(y i |x i )]. Thus, P S ({x i }) ∝ I(x i ).
this section cite: []

Section: Balanced Sample Size.
To effectively rectify biased decision boundaries, the cardinality of sampled subsets must be carefully balanced. A subset with a large sample size risks preserving the original imbalance, whereas an overly small subset may lead to significant information loss. Given a ground set with N items, we theoretically demonstrate that that the expected sample size of a DPP defined in Eq. ( 19) is N (1-ln 2). Due to the page limit, the details of this theorem are deferred to Appendix A.7.
This reduction to roughly one-third of the original size is inadequate for balancing the class priors in a highly imbalanced setting. To address this issue, we propose Information-Preservable Determinantal Point Process (IP-DPP) to sample balanced subsets by selecting a fixed cardinality k instances from each majority class, as defined below:
P k S (Y) = det(SY) |Y ′ | det(S Y ′ ) .(23)
As such, our IP-DPP approach can effectively sample balanced subsets to rectify decision boundaries while preserving valuable information.
Effective Sampling Strategy. However, directly applying Eq. ( 23) for sampling entails significant computational costs. Drawing inspiration from prior studies [32,33], we devise a novel and computationally efficient sampling strategy for our IP-DPP method. Specifically, given the symmetric stochastic matrix S, its spectral decomposition yields orthonormal eigenvectors {v i } N i=1 with corresponding eigenvalues {λ i } N i=1 , such that:
S = N i=1 λiviv ⊤ i .(24)
Let e i ∈ R N denote the i-th standard basis vector, which contains a single 1 in its i-th entry and 0's elsewhere. U (0, 1) is the standard uniform distribution. Then, Algorithm 1 outlines an efficient sampling strategy for our IP-DPP approach.
this section cite: ['b31', 'b32']

Section: Experimental Setup
Datasets. We conduct experiments on four artificially induced or real-world long-tailed datasets: i) CIFAR-10-LT and ii) CIFAR-100-LT: we follow the setting in [3] by sampling long-tailed datasets respectively from the original CIFAR-10 and CIFAR-100 datasets; iii) ImageNet-LT [43]: a truncated version of ImageNet [12] with a total of 1, 000 classes; and iv) iNaturalist 2018 [26]: a naturally long-tailed dataset containing 8, 142 species around the world. The imbalanced factor (IF) for CIFAR-10-LT and CIFAR-100-LT, if not specified, is set to 100 (i.e., Nmax Nmin = 100). Compared Approaches. We compare our approach to nine state-of-the-arts for long-tailed recognition: Focal Loss [38], LDAM Loss [3], τ -norm [30], RIDE [59], KCL [34], TSC [35], SBCL [27], OTmix [15], and DisA [14].
Metrics. We evaluate long-tailed recognition performance using four metrics: many-shot, mediumshot, few-shot, and overall accuracies. Many-shot, medium-shot, and few-shot assess model performance in head, medium, and tail classes, respectively. All results are averaged over 5 trials.
Additional experimental settings, including thresholds for defining the above metrics and hyperparameters, are provided in Appendix B.1 to conserve space.
this section cite: ['b2', 'b42', 'b11', 'b25', 'b37', 'b2', 'b29', 'b58', 'b33', 'b34', 'b26', 'b14', 'b13']

Section: Comparisons to State-of-the-Arts
Small-Scale Datasets. We first conduct experiments on two small-scale long-tailed datasets, i.e., CIFAR-10-LT and CIFAR-100-LT, to compare our approach with nine counterparts mentioned in Section 5.1. Table 1 presents comparative results. On CIFAR-10-LT, our approach achieves the best overall accuracy of 76.4%, outperforming all counterparts by 2.6% at least. This performance improvement can be attributed to two key aspects. First, our BNS approach effectively captures both instance-level and class-level semantics, facilitating the learning of high-quality representations and the creation of well-separated feature spaces, respectively. Second, our IP-DPP method addresses biased decision boundaries by sampling relatively balanced subsets while preserving valuable information. This can mitigate the majority-biased tendency while maintaining the model's overall performance. On the other hand, although our approach lags behind prior studies in many-shot accuracy, these methods consistently struggle with biased decision boundaries. They prioritize performance on head classes, resulting in significantly diminished accuracy for medium and tail classes. For instance, while our method falls short of OTmix by 5.9% in many-shot accuracy, it surpasses OTmix with significantly higher medium-shot and few-shot accuracies, improving by 8.5% and 19.9%, respectively. These results demonstrate that our approach effectively mitigates biased decision boundaries while maintaining the model's overall performance.
We observe similar trends on CIFAR-100-LT. First, our approach achieves the highest overall accuracy of 52.4%, surpassing prior state-of-the-art, i.e., DisA, by a notable margin of 3.2%. Second, while our approach lags behind OTmix by 10.7% in many-shot accuracy, it achieves improvements of 11.7% in medium-shot accuracy and 12.8% in few-shot accuracy, as well as an improvement of 4.3% in overall accuracy. These results further confirm that our approach effectively mitigates majority-biased tendencies while preserving the model's overall performance. Large-Scale Datasets. Next, we conduct experiments on ImageNet-LT and iNaturalist 2018 to assess the effectiveness of our approach on large-scale, long-tailed datasets. Table 2 provides the comprehensive results. We make two key observations. First, our approach achieves the highest accuracies on both ImageNet-LT (i.e., 51.7%) and iNaturalist 2018 ( i.e., 74.0%), outperforming all competing methods. For instance, on ImageNet-LT, our approach surpasses DisA, the best baseline method, by 2.3%. Similarly, on iNaturalist 2018, it outperforms SBCL, the best counterpart, by 3.6%. These results highlight the strong generalizability of our method to large-scale, long-tailed datasets. Moreover, while our approach lags behind some baseline methods in many-shot accuracy, it achieves the highest medium-shot and few-shot accuracies across both datasets. This is because prior methods result in majority-biased decision boundaries, which disproportionately favor head classes.
this section cite: []

Section: Evaluation on Representation Learning
Quantitative Evaluation. Next, we quantitatively evaluate the performance of our BNS method for representation learning. Specifically, we compare it against three state-of-the-art contrastive learning methods for long-tailed recognition: KCL, TSC, and SBCL. To assess the quality of the learned feature representations, we use linear probing accuracy, which involves fine-tuning a linear classifier on a pre-trained feature extractor with frozen weights.
Figures 1a and 1b illustrate linear probing accuracies on CIFAR-10-LT and CIFAR-100-LT, respectively. On CIFAR-10-LT, our approach achieves the best overall accuracy of 68.2% (see the pink bar), outperforming KCL, TSC, and SBCL by 7.2%, 4.4%, and 3.5%, respectively.
KCL TSC SBCL Ours Methods 0 20 40 60 80 Accuracy (%) 82.1 84.8 80.1 69.4 55.5 47.7 68.1 66.0 29.4 39.7 36.6 67.6 61.0 63.8 64.7 68.2 Many-shot Medium-shot Few-shot Overall This is because maximizing the mutual information expressed in Eq. ( 3) is equivalent to minimizing the intra-class distance, which, in turn, enhances the quality of feature representations. Existing methods for long-tailed data often exhibit an undesirable bias toward head classes, leading to poor performance on tail classes, as evidenced by significant disparities between many-shot and few-shot accuracies: 52.7% for KCL, 45.1% for TSC, and 43.5% for SBCL. In contrast, our BNS method enjoys unbiased representation space, exhibiting similar performance on many-shot and few-shot accuracies (i.e., 69.4% vs. 67.6%). Similarly, our approach achieves the highest linear probing accuracy of 47.1% on CIFAR-100-LT, surpassing KCL, TSC, and SBCL by 6.7%, 5.7%, and 2.9%, respectively. Furthermore, our approach effectively mitigates the majority-biased tendency. For instance, compared to SBCL, our method achieves substantial improvements of 11.5% in medium-shot accuracy and 6.1% in few-shot accuracy, with only a modest 8.5% reduction in many-shot accuracy.
this section cite: []

Section: Qualitative Evaluation.
To gain a deeper understanding of our BNS method's role in representation learning, we utilize t-SNE [56] to visualize the feature spaces learned by SBCL and our BNS approach. Figure 2 illustrates the feature spaces for the CIFAR-10 training and test sets. The training representation space learned by SBCL exhibits poor separation for medium and tail classes (see Figure 2a), leading to overlapping boundaries among these classes in the test representation space  (see Figure 2b). In contrast, our approach achieves improved separation for medium and tail classes in the training representation space (see Figure 2c), leading to clear and well-defined boundaries for these classes in the test representation space (see Figure 2d). This improvement stems from our method's ability to capture class-level semantics, which promotes the development of well-separated representation spaces. The differences in decision boundaries between SBCL and our approach elucidate why SBCL achieves poor linear probing accuracies on medium and tail classes, whereas our approach demonstrates superior performance on these classes (see Figures 1a and 1b for details).
this section cite: ['b55']

Section: Performance Results under Various Imbalanced Factors
This section presents performance results under various imbalance factors (IF). Here, we consider five IF values ranging from 10 to 200. Table 3 shows comparative results on CIFAR-10-LT and CIFAR-100-LT, where we compare our approach with TSC, SBCL, OTmix, and DisA. On both datasets, our approach achieves the highest overall accuracies across all IF values. For instance, when the IF is set to 200, our method achieves an accuracy of 73.5% on CIFAR-10-LT and 46.7% on CIFAR-100-LT. Moreover, our method demonstrates greater robustness to large imbalance factors. For example, when the IF value increases from 10 to 200, our approach experiences a performance drop of 10.2% (i.e., 83.7% vs. 73.5%) on CIFAR-10-LT, whereas baseline methods suffer larger decreases, with at least a 15.2% drop (see DisA, 82.4% vs. 67.2%).
Additional experimental results are provided in Appendices B.2-B.6. These include adaptability of our approach, applicability across different model architectures, evaluation of computational overhead, ablation studies, and hyperparameter sensitivity.
this section cite: []

Section: Conclusion
This work has addressed the challenging long-tailed data classification problem, by proposing a novel information-preservable two-stage learning approach. Our key contributions include: i) Balanced Negative Sampling (BNS), a new representation learning strategy that effectively captures both instance-level and class-level semantics, facilitating the creation of high-quality feature representations and well-separated feature spaces; and ii) Information-Preservable Determinantal Point Process (IP-DPP), a novel sampling technique designed to select mathematically informative instances, effectively rectifying majority-biased decision boundaries while maintaining the model's overall performance. As such, our approach achieves state-of-the-art performance across various long-tailed datasets by preserving the valuable information within the entire dataset, allowing it to consistently and decisively outperform its counterparts.
this section cite: []

Section: References
Ref_id:b0 Title: Beit: BERT pre-training of image transformers Year: ()
Ref_id:b1 Title: Eynard-mehta theorem, schur process, and their pfaffian analogs Year: (2005)
Ref_id:b2 Title: Learning imbalanced datasets with label-distribution-aware margin loss Year: (2019)
Ref_id:b3 Title: SMOTE: synthetic minority over-sampling technique Year: (2002)
Ref_id:b4 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b5 Title: Big self-supervised models are strong semi-supervised learners Year: (2020)
Ref_id:b6 Title: An empirical study of training self-supervised vision transformers Year: (2021)
Ref_id:b7 Title: ELECTRA: pre-training text encoders as discriminators rather than generators Year: (2020)
Ref_id:b8 Title: Parametric contrastive learning Year: (2021)
Ref_id:b9 Title: Class-balanced loss based on effective number of samples Year: (2019)
Ref_id:b10 Title: Large scale fine-grained categorization and domain-specific transfer learning Year: (2018)
Ref_id:b11 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b12 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: ()
Ref_id:b13 Title: Distribution alignment optimization through neural collapse for long-tailed classification Year: ()
Ref_id:b14 Title: Enhancing minority classes by mixing: An adaptative optimal transport approach for long-tailed classification Year: ()
Ref_id:b15 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b16 Title: Discriminative sample generation for deep imbalanced learning Year: (2019)
Ref_id:b17 Title: Noise-contrastive estimation: A new estimation principle for unnormalized statistical models Year: (2010)
Ref_id:b18 Title: Borderline-smote: A new over-sampling method in imbalanced data sets learning Year: (2005)
Ref_id:b19 Title: ADASYN: adaptive synthetic sampling approach for imbalanced learning Year: (2008)
Ref_id:b20 Title: Learning from imbalanced data Year: (2009)
Ref_id:b21 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b22 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b23 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b24 Title: Interpretable minority synthesis for imbalanced classification Year: (2021)
Ref_id:b25 Title: The inaturalist species classification and detection dataset Year: (2018)
Ref_id:b26 Title: Subclass-balancing contrastive learning for long-tailed recognition Year: ()
Ref_id:b27 Title: Learning deep representation for imbalanced classification Year: (2016)
Ref_id:b28 Title: Exploring balanced feature spaces for representation learning Year: ()
Ref_id:b29 Title: Decoupling representation and classifier for long-tailed recognition Year: (2020)
Ref_id:b30 Title: Imagenet classification with deep convolutional neural networks Year: (2012)
Ref_id:b31 Title: k-dpps: Fixed-size determinantal point processes Year: (2011)
Ref_id:b32 Title: Determinantal point processes for machine learning Year: (2012)
Ref_id:b33 Title: Targeted supervised contrastive learning for long-tailed recognition Year: ()
Ref_id:b34 Title: Targeted supervised contrastive learning for long-tailed recognition Year: ()
Ref_id:b35 Title: Scaling language-image pre-training via masking Year: (2023)
Ref_id:b36 Title: Mmst-vit: Climate change-aware crop yield prediction via multi-modal spatial-temporal vision transformer Year: (2023)
Ref_id:b37 Title: Focal loss for dense object detection Year: (2017)
Ref_id:b38 Title: Self-supervised learning is more robust to dataset imbalance Year: ()
Ref_id:b39 Title: Exploratory undersampling for class-imbalance learning Year: (2009)
Ref_id:b40 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b41 Title: Kolmogorov-arnold networks Year: (2024)
Ref_id:b42 Title: Large-scale long-tailed recognition in an open world Year: (2019)
Ref_id:b43 Title: SGDR: stochastic gradient descent with warm restarts Year: (2017)
Ref_id:b44 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b45 Title: Comparing clusterings by the variation of information Year: (2003)
Ref_id:b46 Title: Long-tail learning via logit adjustment Year: (2021)
Ref_id:b47 Title: Distributed representations of words and phrases and their compositionality Year: (2013)
Ref_id:b48 Title: Generative adversarial minority oversampling Year: (2019)
Ref_id:b49 Title: Elastic-infogan: Unsupervised disentangled representation learning in class-imbalanced data Year: (2020)
Ref_id:b50 Title: Factors in finetuning deep model for object detection with long-tail distribution Year: (2016)
Ref_id:b51 Title: MIM4DD: mutual information maximization for dataset distillation Year: ()
Ref_id:b52 Title: Very deep convolutional networks for large-scale image recognition Year: (2015)
Ref_id:b53 Title: Ecgn: A cluster-aware approach to graph neural networks for imbalanced classification Year: (2024)
Ref_id:b54 Title: Training data-efficient image transformers & distillation through attention Year: (2021)
Ref_id:b55 Title: Visualizing data using t-sne Year: (2008)
Ref_id:b56 Title: Attention is all you need Year: (2017)
Ref_id:b57 Title: Deep generative model for robust imbalance classification Year: (2020)
Ref_id:b58 Title: Long-tailed recognition by routing diverse distribution-aware experts Year: ()
Ref_id:b59 Title: Rethinking the value of labels for improving class-imbalanced learning Year: (2020)
Ref_id:b60 Title: A novel model for imbalanced data classification Year: (2020)
Ref_id:b61 Title: BBN: bilateral-branch network with cumulative learning for long-tailed visual recognition Year: (2020)
