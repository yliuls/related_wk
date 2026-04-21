Title: AnaCP: Toward Upper-Bound Continual Learning via Analytic Contrastive Projection
Abstract: This paper studies the problem of class-incremental learning (CIL), a core setting within continual learning where a model learns a sequence of tasks, each containing a distinct set of classes. Traditional CIL methods, which do not leverage pretrained models (PTMs), suffer from catastrophic forgetting (CF) due to the need to incrementally learn both feature representations and the classifier. The integration of PTMs into CIL has recently led to efficient approaches that treat the PTM as a fixed feature extractor combined with analytic classifiers, achieving state-ofthe-art performance. However, they still face a major limitation: the inability to continually adapt feature representations to best suit the CIL tasks, leading to suboptimal performance. To address this, we propose AnaCP (Analytic Contrastive Projection), a novel method that preserves the efficiency of analytic classifiers while enabling incremental feature adaptation without gradient-based training, thereby eliminating the CF caused by gradient updates. Our experiments show that AnaCP not only outperforms existing baselines but also achieves the accuracy level of joint training, which is regarded as the upper bound of CIL.

Section: Introduction
Continual learning (CL) learns a sequence of tasks incrementally, where each task introduces a set of new classes [1]. The key challenge in CL is catastrophic forgetting (CF) [2], a phenomenon where learning new tasks degrades the performance on previously learned ones. Among CL settings, class-incremental learning (CIL) [3] is particularly challenging as it needs to build a unified classifier to recognize all encountered classes, without task-id information at test time. This paper focuses on the CIL setting. Early CIL approaches typically trained models from scratch using regularization, experience replay, or architectural modifications to mitigate CF [4,5]. However, they fall significantly short of joint training accuracy, which trains a model with the data from all tasks simultaneously, representing the upper bound for CIL performance [6].
Recent CIL methods increasingly exploit pre-trained models (PTMs) to improve accuracy by leveraging their strong feature representations [7,8,9]. Existing PTM-based approaches for CIL can be broadly divided into three main categories: (i) fine-tuning the PTM, either fully or through lightweight adapter [10], (ii) optimizing learnable prompts while keeping the PTM parameters frozen [11], and (iii) using the PTM as a fixed feature extractor paired with an analytic classifier, such as the nearest class mean (NCM) method or more advanced variants [12,13,14]. We use the term analytic to describe methods with a closed-form solution that requires no gradient-based training. The first two groups require task-specific training, which is slow and prone to CF. The third group is highly efficient and immune to CF because it avoids gradient-based updates.
Analytic approaches often achieve state-of-the-art performance in CIL, outperforming methods that fine-tune the PTM or learn prompts [15,14]. However, a major limitation of the analytic methods is their inability to adapt the PTM features to suit the downstream tasks, leading to suboptimal performance. Prior works have proposed to fine-tune the PTM on the first task, called first session adaptation (FSA) [16,13]. However, the PTM must remain frozen afterward to maintain the integrity of analytic learning; otherwise, their closed-form solutions will not work. This paper introduces AnaCP (Analytic Contrastive Projection), a novel method that enables analytic approaches to also perform feature adaptation. Our analytic method is related to extreme learning machines (ELMs) [17,18], as outlined in Section 3. AnaCP adapts features using a contrastive projection layer, inspired by contrastive learning [19,20], which draws samples from the same class closer while pushing different classes apart to enhance separation. However, unlike standard contrastive learning, which requires gradient-based training for each task and suffers from CF, AnaCP achieves a similar effect analytically. With the adapted features, an NCM classifier can be easily constructed (see Section 4.3). However, we found that building another ELM classifier by sampling feature representations from distributions of previous tasks yields better performance, while remaining fully analytic. To summarize, this work makes the following contributions:
1. We propose a novel analytic approach that enables continual adaptation of feature representations across tasks, addressing the long-standing limitation of fixed representations in prior analytic methods.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b13', 'b15', 'b12', 'b16', 'b17', 'b18', 'b19']

Section: 2.
Our method avoids CF entirely, as it requires no gradient-based training and instead relies on closed-form updates.
3. This approach is highly efficient, requiring no trainable parameters and incurring negligible computational overhead.
4. Through extensive experiments, we show that when paired with a strong PTM, our method achieves accuracy comparable to the joint training upper bound. This is particularly notable, as the inability to match joint training remains a key barrier to the practical adoption of CL.
this section cite: []

Section: Related work
Many CIL approaches have been proposed to deal with CF. These methods can be categorized into three main groups: (i) Regularization methods, which mitigate CF by penalizing updates to parameters that are important for old tasks [21,22,23,24], or by aligning the current model with those of earlier ones using knowledge distillation [25,26]. (ii) Experience replay methods, which retain a subset of past task samples in a buffer and use them in the new task training [27,28,29,30,31].A variation of this is pseudo-replay, where synthetic samples or feature representations are generated to simulate past data [32,33,34,35]. Our method also replays feature representations in its final step, but they are sampled from each class's distribution represented by its mean and a shared covariance. (iii) architecture-based methods, which expand the network for each task [36,37,38], or isolate parameters through learning task-specific sub-networks via masking or orthogonal projections [39,40,41,42,43], and then use a task-id prediction method at test time [6,44,45,46].
Early approaches to CIL learned both feature representations and classification parameters for each new task without using PTMs, suffering from serious CF. Recent methods address this limitation by leveraging PTMs, which substantially improve CIL performance [7,12,47,13,6,14], primarily following three strategies: (i) fine-tuning the PTM, either by directly updating its parameters (e.g., SLCA [10]) or by adding adapters [6,48,49]. (ii) learning prompts, where the PTM remains frozen and trainable prompts are introduced to condition its representations (e.g., L2P [11], DualPrompt [47], CODA-Prompt [50], as well as other variants [51,52,53]). (iii) treating the PTM as a frozen feature extractor with an analytic classifier [13,54,14].
Our method aligns with the third strategy, which relies on closed-form solutions. The simplest method is NCM [55,56], which computes a mean vector for each class and assigns each test sample to the class with the nearest mean. Another method is SLDA [57], which takes advantage of linear discriminant analysis (LDA) [58]. KLDA [14] applies an RBF kernel to expand the feature space before using LDA for classification. The regularized least squares used in ACIL [12] and GACL [54] is also within this paradigm. Analytic CIL methods can be enhanced by adopting the FSA strategy, where the PTM is fine-tuned on the first task and then kept frozen for subsequent tasks. APER [56] adopts this strategy with an NCM classifier. RanPac [13] has a similar approach but applies a random projection [18] to the feature space, improving class separation. LoRanPAC [59] enables RanPAC to use a much higher dimensional random projection. FeCAM [15] uses a Mahalanobis distance classifier with normalized covariance matrices.
We follow the analytic CIL paradigm by using PTMs as frozen feature extractors. However, unlike previous methods, we introduce a contrastive projection layer to adapt the features for each task, which enables AnaCP to improve the accuracy while preserving the computational efficiency.
this section cite: ['b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b5', 'b43', 'b44', 'b45', 'b6', 'b11', 'b46', 'b12', 'b5', 'b13', 'b9', 'b5', 'b47', 'b48', 'b10', 'b46', 'b49', 'b50', 'b51', 'b52', 'b12', 'b53', 'b13', 'b54', 'b55', 'b56', 'b57', 'b13', 'b11', 'b53', 'b55', 'b12', 'b17', 'b58', 'b14']

Section: Background
Class-Incremental Learning: CIL seeks to sequentially learn a series of tasks, each introducing a new set of classes, while prohibiting access to data from previously encountered tasks. Specifically, each task t is defined by a training dataset D t = {(x
(i) t , y (i) t )} nt i=1
, where
x (i) t
∈ X t represents an input sample, and y Ridge Regression: Ridge regression is a closed-form method for training a linear classifier on fixed features using one-hot encoded targets [60]. Given a feature matrix X ∈ R N ×d from the PTM and a one-hot label matrix Y ∈ R N ×C , the objective is to minimize the squared error with L 2 regularization:
min W ∥XW -Y∥ 2 + λ∥W∥ 2 ,(1)
where λ controls the strength of the regularization. This objective penalizes deviations from the target labels while encouraging smaller weights to prevent overfitting. The solution is obtained by setting the gradient for W to zero, yielding the closed-form optimal weights:
W = (X ⊤ X + λI) -1 X ⊤ Y,(2)
often called a ridge classifier or regularized least squares. Here, X ⊤ X is called the Gram matrix (G) and X ⊤ Y is the cross matrix (H). This closed-form solution is attractive because it forgoes iterative training and directly computes a global optimum.
Analytic CIL: When tasks arrive sequentially, we can compute the ridge regression solution without revisiting past data [17,12]. Let (X t , Y t ) denote the data from task t. Rather than storing all data, we can incrementally update the Gram and cross matrices as follows:
G t = G t-1 + X ⊤ t X t , H t = H t-1 + X ⊤ t Y t ,(3)
where G t and H t are the gram and cross matrices after task t, respectively. Here, we slightly abuse the notation for H t by allowing the number of classes to increase over time. To maintain dimensional consistency, both H t-1 and the one-hot labels in Y t must be zero-padded. This allows us to compute the updated solution:
W t = (G t + λI) -1 H t (4
)
Random Projection and Extreme Learning Machines: To improve the feature representations, the original d-dimensional features can be projected into a higher-dimensional space of dimension D before learning the analytic classifier in the projected space. [18] proposes to project the features via a random nonlinear layer:
Z = ϕ(XR),(5)
where R ∈ R d×D is a fixed matrix with random values and ϕ(•) is a nonlinear activation, for which we use a GELU function. The classifier is learned analytically via ridge regression:
W = (Z ⊤ Z + λI) -1 Z ⊤ Y (6
)
This solution can be updated incrementally as before (Eqs. 3 and 4). The random projection acts like a kernel function, expanding the feature space and introducing nonlinear interactions among input features, which enhances class separability. and their enhancement after contrastive projection using DINO-v2 as the PTM. Random features are generally more separable than input features due to their higher dimensionality, even though this is not easily observable in the 2D t-SNE map.
this section cite: ['b59', 'b16', 'b11', 'b17']

Section: Methodology
Existing analytic CIL methods operate on fixed feature representations, allowing the classifier to be updated analytically without revisiting prior data. Freezing the PTM is essential because modifying it during training would alter the feature representations, undermining previous analytic updates and leading to CF. However, this rigid strategy also limits the capability to adapt feature representations to suit the downstream CIL tasks.
To overcome this limitation, we propose a novel analytic approach that also adapts feature representations for the continual learning tasks. The pipeline of our approach, illustrated in Figure 1 (left), begins with the PTM generating feature representations, termed the input layer. These features are transformed via a random projection (RP) layer into a higher-dimensional space [18] and then passed through the contrastive projection (CP) layer, which enhances the class separability. The CP layer draws inspiration from contrastive learning, where the samples from the same class (positives) are drawn closer, while samples from different classes (negatives) are pushed apart, typically using a contrastive loss. However, applying this directly to CIL would lead to CF due to iterative gradient updates. We achieve similar effects via an analytic CP layer that can be incrementally updated as new tasks arrive. The impact of this CP layer on feature distribution is depicted in Figure 1 (right). Finally, a classifier layer makes the final predictions based on these adapted features.
this section cite: ['b17']

Section: Positive Alignment via Prototype Regression
We begin with the ELM formulation, where ridge regression is employed to map the features from the RP layer to the target vectors, as previously defined:
W = (Z ⊤ Z + λI) -1 Z ⊤ T,(7)
where Z ∈ R N ×D represents the random feature obtained from the RP layer, and T contains target vectors for inputs. When T is one-hot encoded for classification, the cross matrix Z ⊤ T reduces to a matrix where each column is the sum of random features belonging to a given class. This can be factorized as:
H = Z ⊤ T = MN (8
)
Here, M ∈ R D×C contains the class prototypes m c , defined as the means of the random features of class c, and N = diag(n 1 , . . . , n C ) holds the number of samples per class. Both M and N can be computed incrementally as each task arrives.
This formulation can be extended to arbitrary targets. Instead of using one-hot vectors, we can assign each class a target prototype, representing its ideal position in the projected space. These target prototypes can have any dimensionality; however, to preserve the geometric structure of the representations, we set their dimension to match that of the input layer, d. Let P ∈ R C×d denote the target prototype matrix, where each row p c specifies the desired projection for class c. For instance, p c can be set as the original class mean of the input layer, pulling the feature representations of class c toward their respective mean. In this case, the cross matrix becomes:
H = Z ⊤ T = c i∈c z i p ⊤ c = c m c n c p ⊤ c = MNP (9
)
This decomposition enables efficient incremental updates by maintaining class prototypes M computed based on the features from the random projection layer, sample counts N, and target prototypes P. The learned projection W maps samples toward their corresponding target prototypes, effectively pulling intra-class samples together and enhancing positive alignment.
this section cite: []

Section: Negative Repulsion via Target-Prototype Separation
Aligning input samples to their respective class means (using class means as target prototypes) improves intra-class compactness but does not guarantee adequate inter-class separation. In particular, some class means may lie close to each other in the feature space, making them hard to distinguish.
To address this, we refine the target prototypes by shifting the class means to enhance the separation between classes.
Let C = (µ 1 , . . . , µ C ) ∈ R d×C denote the matrix of class means at the input layer. We normalize these means by whitening them with respect to the feature covariance matrix Σ (shared by all classes). This adjustment is essential because a large variance in some dimensions can give a false impression of class separability. For example, in a low-variance dimension even a small separation is meaningful, whereas in high-variance dimensions, a much larger separation is needed to achieve the same effect. Whitening corrects for this imbalance by accounting for the variance of each feature dimension. The covariance is computed incrementally as [15]:
Σ t = N t-1 N t Σ t-1 + 1 N t c∈Ct i∈c (x i -µ c )(x i -µ c ) ⊤ ,(10)
where N t is the total number of samples seen up to task t and C t is the set of classes introduced at task t. We can then transform the class means as:
Ĉ = Σ -1/2 C,(11)
yielding whitened means with unit variance in all directions. We then perform singular value decomposition (SVD) on the whitened class means:
Ĉ = USV ⊤ ,(12)
where U ∈ R d×C is an orthonormal basis of the input space, V ∈ R C×C defines the class means subspace, and S ∈ R C×C is a diagonal matrix containing the singular values.
Our objective is to enhance the separation between class means, quantified by the sum of their pairwise cosine similarities. In principle, maximum separation would be achieved if the means were arranged uniformly on the surface of a hypersphere. However, this strict configuration would significantly alter the original data geometry, distorting the feature representations. To maintain the structure while increasing class separation, we introduce an adjustment to the singular values in S, which perturbs the class means in directions that reduce their pairwise cosine similarities:
S = S + αDiag{δ 1 , . . . , δ C }, (13
)
where α is a scaling factor, and δ i is defined in the following Lemma. Lemma 4.1. Denote w 1 , . . . , w C ∈ R C as arbitrary vectors, and e 1 , . . . , e C ∈ R C as a set of orthogonal bases of R C . Denote ⟨x, y⟩ = x ⊤ y as the inner product and θ(x, y) = x ⊤ y ||x||•||y|| as the cosine similarity. There exist α > 0 and
δ i =                1, if j̸ =i 1 ||w j || (2 • 1 ⟨wi,wj ⟩≥0 ) • ⟨e i , w j ⟩ • ||w i || 2 -⟨e i , w i ⟩ • |⟨w i , w j ⟩| < 0, 0, if j̸ =i 1 ||w j || (2 • 1 ⟨wi,wj ⟩≥0 ) • ⟨e i , w j ⟩ • ||w i || 2 -⟨e i , w i ⟩ • |⟨w i , w j ⟩| = 0, -1, else, s.t. i j̸ =i |θ(w i + αδ i e i , w j + αδ j e j )| ≤ i j̸ =i |θ(w i , w j )|(14)
See the proof in Appendix A. In Eq. 12, we denote SV ⊤ = (w 1 , . . . , w C ) and V ⊤ = (e 1 , . . . , e C ). Each w i in the subspace SV ⊤ is perturbed by αδ i e i , effectively shifting it in an orthogonal direction with magnitude α. By Lemma 4.1, there exist coefficients δ 1 , . . . , δ C and a scalar α > 0 such that Eq. 14 holds. The sign of each δ i determines whether the shift occurs in the direction of e i or its opposite. This condition arises from the proof of Lemma 4.1, where we analyze the derivative of the function f i (α), defined as the sum of cosine similarities involving class i. By selecting δ i accordingly, we ensure that the slope of f i (α) at α = 0 is negative, which guarantees that a small positive α decreases the cosine similarity. The choice of α is also important: if α is too small, the shift becomes negligible, whereas if it is too large, the reduction in cosine similarity may no longer be valid. In practice, we set α = 1 in our experiments.
Defining ∆ = Diag{δ 1 , . . . , δ C }, the lemma guarantees that the average cosine similarity among the columns of (S + α∆)V ⊤ is smaller than that of SV ⊤ . Since the columns of U ∈ R d×C form an orthogonal basis in R d , the transformations preserve inner products and norms:
⟨Uw i , Uw j ⟩ = ⟨w i , w j ⟩, ||Uw i || = ||w i ||, ||Uw j || = ||w j ||(15)
Thus, the cosine similarity between transformed vectors remains the same, i.e., θ(Uw i , Uw j ) = θ(w i , w j ). Therefore, the average cosine similarity among the columns of U(S + α∆)V ⊤ is also reduced compared to the original matrix Ĉ = USV ⊤ .
Note that separability is measured as the sum of cosine similarities, but the class means are whitened. This procedure is closely related to Mahalanobis distance, which has been shown to outperform cosine similarity in related contexts [15]. By whitening, we effectively assess separation using Mahalanobis distance; after increasing the separation, we de-whiten the features to restore their original scale:
Ĉ = U SV ⊤ , C = ĈΣ 1/2 (16
)
The separated class means are then used as target prototypes for the CP layer by setting P in Eq. 9 to C, aligning samples with their respective classes while enhancing the separation between classes.
this section cite: ['b14', 'b14']

Section: Classifier
The CP layer is an analytic module that adapts feature representations across tasks without modifying the underlying PTM. To increase its capacity, we extend it with multiple random projections. Since each RP is initialized independently, we learn a corresponding CP for each RP, all sharing the same set of target prototypes. Specifically, we define H such heads, where each CP head W (h) maps the random features to the same target prototype space. Given an input x, each head produces a projection and the output is obtained by averaging across heads:
u (h) = ϕ(xR (h) )W (h) , u = 1 H H h=1 u (h)(17)
This aggregated representation u is then classified using NCM, which assigns it to the nearest target prototype in P. As shown in Table 3 (row 3), AnaCP with an NCM classifier already outperforms all baselines. However, NCM is often surpassed by more expressive classifiers. To further improve performance, we place an ELM classifier after the averaged CP representation.
Unlike PTM features and their random projections, which remain fixed, the CP outputs evolve as new tasks are introduced due to the incremental computation of the CP layer. Directly updating the ELM classifier incrementally (Eq. 3) may therefore lead to CF. To mitigate this, we employ a pseudoreplay strategy. We model the PTM features (input layer) as a multivariate Gaussian distribution parameterized by the class means and a shared covariance matrix, from which we generate pseudoreplay samples for training. This approach resembles that of [10], which also trains the classifier on generated features. However, whereas [10] maintains a separate covariance matrix per class (leading to memory cost that grows with the number of classes), we use a single shared covariance matrix across all classes. This design significantly reduces memory usage while yielding comparable performance. Specifically, we reuse the class means and shared covariance matrix from Eq. 10 for feature generation. The generated samples are then processed by the RP and CP layers to form the classifier input, upon which the ELM is trained using Eq. 6.
5 Experiments
this section cite: ['b9', 'b9']

Section: Experimental Setup
Datasets: We conduct experiments on five publicly available datasets: CIFAR100 (100 classes) [61], ImageNet-R (200 classes) [62], CUB (200 classes) [63], TinyImageNet (200 classes) [64], and Cars (196 classes) [65]. For all datasets, we use the official train and test splits. Each dataset is split into 10 disjoint tasks by shuffling classes, and experiments are repeated with three different random seeds to account for variability in class-task assignments.
Baselines: We compare AnaCP with a range of baselines, including prompt-learning methods (L2P [11], DualPrompt [47], and CODA-Prompt [50]), fine-tuning method (SLCA [10]), and analytic methods that treat the PTM as a frozen feature extractor. The analytic baselines include SimpleCIL [55], SLDA [57], KLDA [14], GACL [54], APER [56], FeCAM [15], and RanPac [13], with some incorporating FSA. For a complete performance perspective, we also include results from joint linear probe, where a linear softmax classifier is trained on the PTM features using gradient descent with all tasks' data, and joint fine-tuning, which directly fine-tunes the PTM on the entire dataset and serves as the upper bound for CIL performance.
Implementation Details: For the PTMs, we use DINO-v2 [66] and MoCo-v3 [67], both of which are self-supervised PTMs. This choice helps prevent information leakage, as supervised PTMs are exposed to class labels during training, some of which may reappear in CIL, leading to unintended information leakage. MoCo-v3 is pre-trained on ImageNet-1k -a commonly used but relatively small dataset for pre-training in prior studies. However, because ImageNet-1k (or even ImageNet-21k) is considered limited for real-world applications, we also utilize DINO-v2, a more powerful model pre-trained on the substantially larger LVD-142M dataset. For joint fine-tuning, we employ LoRA adapters [68] instead of full fine-tuning, as we found this approach to be more accurate.
The prompt learning baselines are taken from the [50] repository, while SLCA, FeCAM, and RanPac are evaluated using their original implementations. All remaining analytic baselines are incorporated into a unified experimental setup, following their official implementations to ensure fair comparison. 1We also adopt FSA following [56] before applying AnaCP to the frozen PTM as it helps improve accuracy. We use RP dimension D = 5000, number of CP heads H = 3, and number of generated feature representations per class for the classifier R = 100 as the default configuration; ablations on other variants are included. The regularization parameter λ in Eq. 6 for ELM is set to 10 2 , and the coefficient α for class mean repulsion is set to 1. These values were optimized on a validation set derived from the CIFAR100 training set and are consistently used across all datasets and both PTMs.
All experiments are conducted on a single NVIDIA A100 GPU with 80GB VRAM.
this section cite: ['b60', 'b61', 'b62', 'b63', 'b64', 'b10', 'b46', 'b49', 'b9', 'b54', 'b56', 'b13', 'b53', 'b55', 'b14', 'b12', 'b65', 'b66', 'b67', 'b49', 'b55']

Section: Evaluation Metric:
We evaluate model performance using two primary metrics: Last Accuracy (A Last ), the accuracy after completing all tasks, and Average Incremental Accuracy (A Avg ), calculated as A Avg = 1 T T t=1 A t , where A t denotes the accuracy at the end of task t. Additionally, we measure running time and memory efficiency to assess the practicality of the methods. We also define relative error reduction as A I -A0 100-A0 × 100, where A 0 is the baseline accuracy and A I is the improved accuracy.
this section cite: []

Section: Comparison with Baselines
Table 1 presents the main results of our experiments, with all methods using DINO-v2 as the PTM.
AnaCP consistently outperforms all baselines across the evaluated datasets, achieving an absolute improvement in last accuracy ranging from 1.03% to 3.17% across different datasets, corresponding to a relative error reduction between 4.8% and 31.4%. We can observe that using replay features with a shared covariance (AnaCP) achieves similar results to class-specific covariances (AnaCP -Σ y ) while significantly reducing the number of parameters. AnaCP also surpasses the joint linear probe on all datasets and achieves comparable accuracy to joint fine-tuning, which represents the Table 2: Comparison of AnaCP with baselines using MoCo-v3 as the PTM. The prompt learning baselines are excluded here due to their much lower accuracy, which is also the case in Table 1.
upper bound for CIL. Notably, on two datasets, AnaCP even exceeds this upper bound, while on the remaining datasets, the largest accuracy gap is only around 2%. These results highlight that AnaCP's feature adaptation is highly effective, even without directly training the PTM on each task.
We also evaluate AnaCP using MoCo-v3 as the PTM, as presented in Table 2. AnaCP outperforms the baselines by a larger margin with MoCo-v3 than with DINO-v2, as the weaker features of MoCo-v3 make the CP layer more impactful. In this case, the gap to joint fine-tuning increases to 6%, which is expected since MoCo-v3 is a weaker PTM and benefits more from full fine-tuning. However, given that both PTMs have the same architecture (number of layers and hidden size), there is no reason to prefer a weaker PTM in real-world applications.
this section cite: []

Section: Analysis of Catastrophic Forgetting
The CP layer in our method is inherently immune to CF, as it operates on a frozen PTM and updates its statistics (prototypes and the Gram matrix) incrementally without overwriting prior information.
In contrast, the final ELM classifier relies on pseudo-replay, which in principle could introduce some forgetting. In practice, however, we found this effect to be minimal. To verify this, we evaluate under the task-incremental learning (TIL) setting, where the task identity is provided at inference time. Specifically, we present the accuracy matrix A[t][i], where each entry denotes the accuracy on task i after training on the first t tasks of CIFAR-100 (10-task split) using DINO-v2 as the backbone.
We adopt the TIL setting for this analysis because it offers a clearer view of forgetting. In CIL, accuracy inevitably decreases as more tasks are introduced, not necessarily due to forgetting, but because the classification problem becomes harder with a growing number of classes. By contrast, TIL keeps the classification difficulty fixed, since each task has a constant number of classes and the task ID is known at inference. Under this setup, forgetting would appear as a decline in accuracy on earlier tasks. However, as shown in Figure 2, the accuracy for task 1 (first column) remains nearly constant across all steps. The same holds for other tasks, demonstrating that our method effectively preserves knowledge from prior tasks.
row CLS CP NR H D R CIFAR100 ImageNet-R CUB TinyImageNet Cars 1 NCM ✗ ----88.13±0.21 81.60±0.08 86.95±0.36 83.63±0.48 69.51±0.87 2 NCM ✓ ✗ 3 5000 -91.31±0.03 83.29±0.45 89.52±0.10 85.89±0.26 86.25±0.25 3 NCM ✓ ✓ 3 5000 -91.86±0.05 86.01±0.11 89.88±0.32 87.02±0.27 89.48±0.29 4 ELM ✗ --5000 -91.12±0.10 85.14±0.26 89.75±0.19 86.48±0.12 89.32±0.20 5 ELM ✓ ✗ 3 5000 100 91.87±0.02 85.73±0.19 90.39±0.14 86.69±0.28 90.24±0.20 6 ELM ✓ ✓ 3 5000 100 92.15±0.09 86.60±0.04 90.48±0.15 87.71±0.29 90.65±0.24 7 ELM ✓ ✓ 3 5000 20 92.14±0.06 86.39±0.04 90.43±0.18 87.29±0.32 90.45±0.23 8 ELM ✓ ✓ 3 5000 50 92.13±0.04 86.47±0.05 90.46±0.10 87.49±0.32 90.57±0.16 9 ELM ✓ ✓ 3 5000 100 92.15±0.09 86.60±0.04 90.48±0.15 87.71±0.29 90.65±0.24 10 ELM ✓ ✓ 1 5000 100 91.99±0.01 85.72±0.24 90.36±0.13 87.12±0.24 90.35±0.24 11 ELM ✓ ✓ 3 5000 100 92.15±0.09 86.60±0.04 90.48±0.15 87.71±0.29 90.65±0.24 12 ELM ✓ ✓ 5 5000 100 92.20±0.04 86.62±0.10 90.51±0.05 87.76±0.29 90.68±0.18 13 ELM ✓ ✓ 3 1000 100 90.64±0.10 84.01±0.29 89.56±0.15 85.48±0.17 88.23±0.21 14 ELM ✓ ✓ 3 2000 100 91.35±0.08 85.15±0.19 89.97±0.18 86.24±0.22 89.93±0.10 15 ELM ✓ ✓ 3 5000 100 92.15±0.09 86.60±0.04 90.48±0.15 87.71±0.29 90.65±0.24 16 ELM ✓ ✓ 3 10000 100 92.75±0.10 86.86±0.10 90.60±0.10 88.11±0.24 90.73±0.16
Table 3: Ablation studies on AnaCP with DINO-v2 as the PTM. All reported values are A last . CLS indicates the method employed in the classifier (Section 4.3). CP indicates whether the proposed contrastive projection is applied. NR (negative repulsion) shows whether class mean separation is improved through Lemma 4.1; when not used, the original class means serve as target prototypes. The table also includes variations in the RP dimension (D), the number of heads in the CP layer (H), and the number of feature representations generated per class (R).
this section cite: []

Section: Ablation Studies
We conduct a series of ablations to evaluate the impact of key components, as shown in Table 3. Rows 1-3 examine the effect of the CP layer. Removing CP and directly applying an NCM classifier to PTM features results in a significant drop in absolute accuracy between 2.93% to 19.97% (row 1 vs. row 3). This highlights the importance of CP in enhancing feature representations, as illustrated in Figure 1. The effect of negative repulsion (NR), which increases the separation between class means (Section 4.2) is also evident; disabling NR and relying only on positive alignment with the original class means as target prototypes leads to an accuracy reduction from 0.38% to 3.23% (row 2 vs. row 3). This confirms that NR is a critical component for improving class separability.
The role of classifier is explored in Rows 4-6, where NCM is replaced by ELM. We find that using an ELM classifier on average improves the performance by 0.66% (row 3 vs. row 6). Although this requires generating feature representations as pseudo-replay for training the ELM classifier, as explained in Section 4.3. An arbitrary number of feature representations can be sampled from the Gaussian distribution with negligible computational overhead, as detailed in the running time analysis. We observe that increasing the number of generated feature representations R improves performance on some datasets (Rows 7-9). The effects of various numbers of CP heads H (Rows 10-12) and RP dimensions D (Rows 13-16) are also explored. Generally, increasing these values improves accuracy, albeit at the cost of additional parameters. For the main results, we have used D = 5000 and H = 3. We note that even with H = 1, AnaCP outperforms all baselines.
this section cite: []

Section: Memory and Running Time Efficiency
AnaCP needs to save C class means, each of size d, and a shared covariance matrix of size d × d for feature whitening and pseudo-replay feature generation at the input layer (Figure 1). While a separate covariance matrix can be maintained for each class, this would result in a significant increase in memory as the number of classes grows. As shown in Table 1, using a shared covariance matrix does not compromise accuracy. Each CP head maintains the following: (1) an RP matrix of size d × D, (2) a D × D Gram matrix, (3) C class prototypes (or means) in the random feature space, each of size D, and (4) a projection matrix W of size d × D. We have H such heads in total. The target prototypes do not need to be stored, as they can be derived from the class means. The classifier layer also maintains an RP matrix of size d × D and W of size C × D. Unlike the CP layer, this layer doesn't need to store the Gram matrix for incremental updates, as it is computed from generated feature representations after each task.
For a typical configuration with C = 200, d = 768, D = 5000, and H = 3, the total number of stored parameters amounts to approximately 106.6 million. The majority of these parameters belong to the D × D Gram matrices, which does not increase with the number of classes. Also, none of these parameters are trainable, making them suitable for storage in reduced-precision formats such as float8. This would further reduce the total memory usage to approximately 102 MiB.
While AnaCP may have a higher parameter count compared to some other methods, the design is balanced with exceptional computational efficiency. For example, CIFAR100 with the DINO-v2 PTM requires only 9 minutes and 18 seconds for training. 6 minutes and 5 seconds of this time are dedicated to the FSA, and 3 minutes and 8 seconds are spent passing inputs through the PTM to extract features. The core operations of AnaCP, including updating the CP layer and training the classifier with generated features, collectively take only 5 seconds. Thus, despite the larger parameter count, these operations are extremely fast. For comparison, a baseline such as CODA-Prompt that requires task-specific training takes 3 hours and 47 minutes to complete training on CIFAR100, and joint fine-tuning requires 1 hour and 58 minutes.
this section cite: []

Section: Discussion
Our results support the proposition that a strong PTM is the key to CL, and that representation-level forgetting is not the primary limitation. Even simple CIL strategies, when paired with a fixed PTM, can achieve near-optimal performance. This view aligns with neuroscience evidence that the human brain maintains stable representations, despite minor drift over time [69,70]. Cortical activity also preserves a stable similarity structure that supports consistent perception and behavior, even as neural responses gradually shift [71,70]. The human brain can thus be viewed as an innate PTM refined through evolution, which provides a stable foundation for continual adaptation. Similarly, large-scale pre-training in machine learning resembles this biological evolution and development, producing reusable representations that enable CL without any forgetting. For further discussion, see [72].
this section cite: ['b68', 'b69', 'b70', 'b69', 'b71']

Section: Conclusion
CIL poses a significant challenge in continual learning. Recent analytic methods with PTMs have shown promise. However, since they cannot learn or adapt features obtained from PTMs to suit specific CIL tasks, their performance is still suboptimal. This paper proposed a novel and principled method, called AnaCP, to adapt features extracted from PTMs analytically, which offers a highly efficient and accurate solution for CIL. Empirical evaluations demonstrate that AnaCP outperforms strong baselines, but more importantly, if paired with a strong PTM, its accuracy can be comparable to the joint training upper bound.
Limitations: When using a strong PTM like DINO-v2, AnaCP achieves accuracy on par with the joint fine-tuning upper bound. However, with a weaker PTM such as MoCo-v3, AnaCP cannot match joint fine-tuning accuracy due to the critical role of PTM features played in our method. While strong PTMs are readily available today, improving AnaCP's performance with weaker PTMs remains a meaningful goal. Additionally, our work currently focuses CIL. We believe AnaCP can be extended to TIL and domain-incremental learning (DIL) scenarios with appropriate adaptations.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: Limitations is discussed in the conclusion.
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
Answer: [Yes] Justification: The only novel theoretical formula included in this paper is lemma 4.1, which is proven in the appendix.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We provided all information needed to reproduce the results and the code in included in the supplementary materials.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes]
Justification: The code is included in the supplementary materials.
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
Answer: [Yes] Justification: We use the official data splits, and all the hyperparameters used are reported.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: The standard deviation of the accuracies are reported.
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
Answer: [Yes] Justification: The computer resources and the execution time of the experiments are reported.
this section cite: []

Section: A Technical Appendices
Lemma A.1 (Lemma 4.1). Denote w 1 , . . . , w C ∈ R C as arbitrary vectors, and e 1 , . . . , e C ∈ R C as a set of orthogonal bases of R C . Denote ⟨x, y⟩ = x ⊤ y as the inner product and θ(x, y) = x ⊤ y ||x||•||y||
as the cosine similarity. There exist α > 0 and
δ i =                1, if j̸ =i 1 ||w j || (2 • 1 ⟨wi,wj ⟩≥0 ) • ⟨e i , w j ⟩ • ||w i || 2 -⟨e i , w i ⟩ • |⟨w i , w j ⟩| < 0, 0, if j̸ =i 1 ||w j || (2 • 1 ⟨wi,wj ⟩≥0 ) • ⟨e i , w j ⟩ • ||w i || 2 -⟨e i , w i ⟩ • |⟨w i , w j ⟩| = 0, -1, else, s.t. i j̸ =i |θ(w i + αδ i e i , w j + αδ j e j )| ≤ i j̸ =i |θ(w i , w j )|.(18)
Proof. Let
f i (α) = j̸ =i |⟨w j , w i + αδ i e i ⟩| ||w j || • ||w i + αδ i e i || .
For simplicity, denote
p i,j (α) = |⟨w i + αδ i e i , w j + αδ j e j ⟩| = |⟨w i , w j ⟩ + αδ i ⟨e i , w j ⟩| and q i (α) = ||w i + αδ i e i || = ⟨w i + αδ i e i , w i + αδ i e i ⟩.
Therefore, we have
f i (α) = j̸ =i 1 ||w j || • p i,j(
α) q i (α) . Since d|x| dx = 2 • 1 x≥0 -1, we have p ′ i,j (α)| α=0 = (2 • 1 ⟨wi,wj ⟩≥0 ) • δ i ⟨e i , w j ⟩. Since ⟨w i + αδ i e i , w i + αδ i e i ⟩ = ⟨w i , w i ⟩ + 2δ i ⟨e i , w i ⟩ + α 2 , we have q ′ i (α)| α=0 = 1 2 2δ i ⟨e i , w i ⟩ + 2α ⟨w i + αδ i e i , w i + αδ i e i ⟩ | α=0 = δ i ⟨e i , w i ⟩ ||w i || Therefore, we have
f ′ i (α)| α=0 = j̸ =i 1 ||w j || • p ′ i,j (α)q i (α) -p i,j (α)q ′ i (α) q 2 i (α) | α=0 = j̸ =i 1 ||w j ||||w i || 2 (2 • 1 ⟨wi,wj ⟩≥0 -1) • δ i ⟨e i , w j ⟩ • ||w i || -|⟨w i , w j ⟩| • δ i ⟨e i , w i ⟩ ||w i || = δ i ||w i || 3 j̸ =i 1 ||w j || (2 • 1 ⟨wi,wj ⟩≥0 -1) • ⟨e i , w j ⟩ • ||w i || 2 -⟨e i , w i ⟩ • |⟨w i , w j ⟩| ,
which shows that δ i 's defined above guarantee
f ′ i (α)| α=0 ≤ 0. Let g j (α; α) = i̸ =j |⟨w j + αδ j e j , w i + αδ i e i ⟩| ||w j + αδ j e j || • ||w i + αδ i e i || .
Since f ′ i (α) is a continuous function, we could take a sufficiently small α s.t. f ′ i (α ′ ) has the same sign when α ′ ∈ [0, α]. Then for α ′ ∈ [0, α], the δ i 's defined above also guarantee g ′ j (α; α ′ )| α=0 ≤ 0. Choosing α = α ′ gives the result.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The abstract and introduction match our claims and the paper's scope.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The research complies with NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This paper studies a standard machine learning problem with no direct societal impact.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
this section cite: []

Section: Answer: [NA]
Justification: There is no high risk misuse.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: All the datasets and models are properly cited.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
this section cite: []

Section: New assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] Justification: Paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not include human research experiments. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not include human research experiments. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: LLMs were only used for editing without any impact on the core methodology. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Lifelong machine learning Year: (2018)
Ref_id:b1 Title: Catastrophic interference in connectionist networks: The sequential learning problem Year: (1989)
Ref_id:b2 Title: Three scenarios for continual learning Year: (2019)
Ref_id:b3 Title: Continual learning of natural language processing tasks: A survey Year: (2022)
Ref_id:b4 Title: A comprehensive survey of continual learning: theory, method and application Year: (2024)
Ref_id:b5 Title: Class incremental learning via likelihood ratio based task prediction Year: (2024)
Ref_id:b6 Title: Achieving forgetting prevention and knowledge transfer in continual learning Year: (2021)
Ref_id:b7 Title: Learnability and algorithm for continual learning Year: (2023)
Ref_id:b8 Title: Recent advances of foundation language models-based continual learning: a survey Year: (2024)
Ref_id:b9 Title: Slca: Slow learner with classifier alignment for continual learning on a pre-trained model Year: (2023)
Ref_id:b10 Title: Learning to prompt for continual learning Year: (2022)
Ref_id:b11 Title: Acil: Analytic class-incremental learning with absolute memorization and privacy protection Year: (2022)
Ref_id:b12 Title: Ranpac: Random projections and pre-trained models for continual learning Year: (2023)
Ref_id:b13 Title: Continual learning using a kernel-based method over foundation models Year: (2025)
Ref_id:b14 Title: Exploiting the heterogeneity of class distributions in exemplar-free continual learning Year: (2023)
Ref_id:b15 Title: First session adaptation: A strong replay-free baseline for class-incremental learning Year: (2023)
Ref_id:b16 Title: A fast and accurate online sequential learning algorithm for feedforward networks Year: (2006)
Ref_id:b17 Title: Extreme learning machine for regression and multiclass classification Year: (2011)
Ref_id:b18 Title: Supervised contrastive learning Year: (2020)
Ref_id:b19 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b20 Title: Overcoming catastrophic forgetting in neural networks Year: (2017)
Ref_id:b21 Title: Continual learning through synaptic intelligence Year: (2017)
Ref_id:b22 Title: Continual learning for sentence representations using conceptors Year: (2019)
Ref_id:b23 Title: Overcoming catastrophic forgetting during domain adaptation of seq2seq language generation Year: (2022)
Ref_id:b24 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b25 Title: Dark experience for general continual learning: a strong, simple baseline Year: (2020)
Ref_id:b26 Title: Online continual learning with maximal interfered retrieval Year: (2019)
Ref_id:b27 Title: Continual learning in lowrank orthogonal subspaces Year: (2020)
Ref_id:b28 Title: Online class-incremental continual learning with adversarial shapley value Year: (2021)
Ref_id:b29 Title: Lifelong intent detection via multi-strategy rebalancing Year: (2021)
Ref_id:b30 Title: Efficient lifelong pre-training for emerging data Year: (2022)
Ref_id:b31 Title: Generative feature replay for class-incremental learning Year: (2020)
Ref_id:b32 Title: Brain-inspired replay for continual learning with artificial neural networks Year: (2020)
Ref_id:b33 Title: Self-sustaining representation expansion for non-exemplar class-incremental learning Year: (2022)
Ref_id:b34 Title: Continual pre-training of language models for math problem understanding with syntax-aware memory network Year: (2022)
Ref_id:b35 Title: Beef: Bi-compatible class-incremental learning via energy-based expansion and fusion Year: (2022)
Ref_id:b36 Title: Der: Dynamically expandable representation for class incremental learning Year: (2021)
Ref_id:b37 Title: Lifelong sequence generation with dynamic module expansion and adaptation Year: (2023)
Ref_id:b38 Title: Overcoming catastrophic forgetting with hard attention to the task Year: (2018)
Ref_id:b39 Title: Demix layers: Disentangling domains for modular language modeling Year: (2022)
Ref_id:b40 Title: Continual learning for task-oriented dialogue system with iterative network pruning, expanding and masking Year: ()
Ref_id:b41 Title: Supermasks in superposition Year: (2020)
Ref_id:b42 Title: Continual learning for multilingual neural machine translation via dual importance-based model division Year: (2023)
Ref_id:b43 Title: A theoretical study on solving continual learning Year: (2022)
Ref_id:b44 Title: Rehearsalfree modular and compositional continual learning for language models Year: (2024)
Ref_id:b45 Title: itaml: An incremental task-agnostic meta-learning approach Year: (2020)
Ref_id:b46 Title: Dualprompt: Complementary prompting for rehearsal-free continual learning Year: (2022)
Ref_id:b47 Title: Inflora: Interference-free low-rank adaptation for continual learning Year: (2024)
Ref_id:b48 Title: Mos: Model surgery for pre-trained model-based class-incremental learning Year: (2025)
Ref_id:b49 Title: Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning Year: (2023)
Ref_id:b50 Title: Hierarchical decomposition of prompt-based continual learning: Rethinking obscured sub-optimality Year: (2023)
Ref_id:b51 Title: Generating instance-level prompts for rehearsal-free continual learning Year: (2023)
Ref_id:b52 Title: Convolutional prompting meets language models for continual learning Year: (2024)
Ref_id:b53 Title: GACL: Exemplar-free generalized analytic continual learning Year: (2024)
Ref_id:b54 Title: A simple baseline that questions the use of pretrained-models in continual learning Year: (2022)
Ref_id:b55 Title: Revisiting class-incremental learning with pre-trained models: Generalizability and adaptivity are all you need Year: (2024)
Ref_id:b56 Title: Lifelong machine learning with deep streaming linear discriminant analysis Year: (2020)
Ref_id:b57 Title: Incremental linear discriminant analysis for classification of data streams Year: (2005)
Ref_id:b58 Title: Loranpac: Low-rank random features and pre-trained models for bridging theory and practice in continual learning Year: (2025)
Ref_id:b59 Title: Theory of ridge regression estimation with applications Year: (2019)
Ref_id:b60 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b61 Title: The many faces of robustness: A critical analysis of out-of-distribution generalization Year: (2021)
Ref_id:b62 Title: The caltech-ucsd birds-200 Year: (2011)
Ref_id:b63 Title: Tiny imagenet visual recognition challenge Year: (2015)
Ref_id:b64 Title: Chen Change Loy, and Xiaoou Tang. A large-scale car dataset for fine-grained categorization and verification Year: (2015)
Ref_id:b65 Title: Dinov2: Learning robust visual features without supervision Year: (2023)
Ref_id:b66 Title: An empirical study of training self-supervised vision transformers Year: (2021)
Ref_id:b67 Title: Lora: Low-rank adaptation of large language models Year: (2021)
Ref_id:b68 Title: Persistence of neuronal representations through time and damage in the hippocampus Year: (2019)
Ref_id:b69 Title: Representations in human primary visual cortex drift over time Year: ()
Ref_id:b70 Title: Face-selective neurons maintain consistent visual responses across months Year: (2014)
Ref_id:b71 Title: Achieving upper bound accuracy of joint training in continual learning Year: (2025)
