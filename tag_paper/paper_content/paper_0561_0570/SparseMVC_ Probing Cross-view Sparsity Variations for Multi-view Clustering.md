Title: SparseMVC: Probing Cross-view Sparsity Variations for Multi-view Clustering
Abstract: Existing multi-view clustering methods employ various strategies to address datalevel sparsity and view-level dynamic fusion. However, we identify a critical yet overlooked issue: varying sparsity across views. Cross-view sparsity variations lead to encoding discrepancies, heightening sample-level semantic heterogeneity and making view-level dynamic weighting inappropriate. To tackle these challenges, we propose Adaptive Sparse Autoencoders for Multi-View Clustering (SparseMVC), a framework with three key modules. Initially, the sparse autoencoder probes the sparsity of each view and adaptively adjusts encoding formats via an entropymatching loss term, mitigating cross-view inconsistencies. Subsequently, the correlation-informed sample reweighting module employs attention mechanisms to assign weights by capturing correlations between early-fused global and viewspecific features, reducing encoding discrepancies and balancing contributions. Furthermore, the cross-view distribution alignment module aligns feature distributions during the late fusion stage, accommodating datasets with an arbitrary number of views. Extensive experiments demonstrate that SparseMVC achieves state-of-theart clustering performance. Our framework advances the field by extending sparsity handling from the data-level to view-level and mitigating the adverse effects of encoding discrepancies through sample-level dynamic weighting. The source code is publicly available at https://github.com/cleste-pome/SparseMVC.

Section: Introduction
Multi-view learning has emerged as a powerful paradigm for leveraging complementary information across multiple perspectives, significantly improving the performance of unsupervised learning tasks such as clustering [1,2,3,4,5]. At the same time, the sparsity of multi-view data has become a pivotal focus of research, with numerous studies proposing solutions from perspectives such as activation functions [6], tensor decomposition [7,8], and variational autoencoders [9]. Nevertheless, while prior methods focus on designing advanced approaches to address data sparsity, they often overlook a fundamental aspect-the potential variations in sparsity across different views. Given that multi-view data consists of multiple views originating from distinct sources, and sparsity is a pervasive characteristic in multi-view data [10,11,12,13,14]. Building upon the factual observations, a natural question arises: "Does there exist a phenomenon of varying sparsity across views?" To quantify cross-view sparsity variations, we define the sparsity ratio s v for the v-th view:
s v = 1 N • F N j=1 F i=1 I[x v i,j = 0],(1)
where N refers to the number of samples, F refers to the feature dimension, x v i,j represents the i-th feature of the j-th sample in the v-th view, and the indicator function I takes the value of one if x v i,j equals zero, and zero otherwise. Zero-valued features x v i,j suggests missing dimensions or data collection errors. Our statistical and computational analysis addresses the question posed earlier and reveals that sparsity variations across views not only exist, but are widely prevalent in diverse multi-view data, as illustrated in Fig. 1.
Figure 1: Sparsity ratios across views in multi-view datasets. TOP BOX PLOT illustrates the sparsity ratio distribution, which shows the median (orange line), interquartile range (box), and any outliers (points outside the whiskers). BOT-TOM BAR PLOT presents the sparsity ratios for each view within each dataset. To provide a more comprehensive situation of sparsity variations, additional datasets are included: Digits [15], LandUse-21 [16], RGB-D [17]. GSE [18] and STL-10 [19]. The cross-view sparsity ratios have been processed to improve visualization using the sigmoid function.
The disparity in view sparsity presents a multifaceted challenge: highly sparse views, lacking sufficient informative content, are prone to underfitting, whereas less sparse views, often burdened with redundant or irrelevant features, are susceptible to overfitting. Applying a uniform encoder architecture or regularization strategy across such heterogeneous views compromises representational consistency and limits the model's capacity to extract complementary cross-view information. To resolve this, we adopt an adaptive and sparsity-aware encoding strategy tailored to individual views. This requires rethinking the autoencoder design to accommodate view heterogeneity and structural disparities. Our solution is to design an autoencoder capable of adaptively adjusting its constraints based on the sparsity ratio of each view, allowing its encoding form to evolve accordingly.
Due to varying sparsity across views for the same sample, encoders can introduce semantic and representational inconsistencies, affecting subsequent stages [20,21]. As a result, it becomes essential to dynamically assess the contribution of each view based on the features extracted from different types of encoders. To address this, we design a correlation-informed reweighting module that assigns sample-wise weights based on the correlations between global and local features, thereby balancing view contributions and discrepancies. The early fusion strategy integrates multi-view data through feature concatenation. Since our reweighting module uses the global latent representation encoded from early fusion to guide the subsequent dynamic weighted early fusion of local representations, we chose an early fusion approach that preserves the original feature values as much as possible.
In late fusion, we introduce a cross-view distribution alignment module to align feature distributions across views, enabling robust integration and supporting datasets with arbitrary view numbers while balancing global consistency and view-specific diversity. These three core components together form the SparseMVC framework, addressing sparsity inconsistencies, encoding discrepancies, and semantic heterogeneity in a step-by-step and interdependent design. As presented in Table 1, we evaluate our method across a range of datasets exhibiting extreme sparsity disparities. For instance, in ALOI-100, the sparsity ratio spans from as low as 0.0001 to as high as 0.6644, revealing a substantial imbalance in information density across views. To effectively address such heterogeneity, our framework incorporates dynamically adaptive autoencoders that adjust both the encoding process and sparsity-aware regularization in accordance with each view's sparsity profile. Unlike existing methods, which generally overlook the impact of cross-view sparsity variation, our targeted design enables SparseMVC to achieve consistently superior performance under severe disparity conditions. Moreover, it maintains strong competitiveness even in datasets with nearuniform sparsity variations, such as LGG and Synthetic3D, demonstrating both its responsiveness to sparsity imbalance and its robustness in more homogeneous scenarios.
To the best of our knowledge, this is the first work to explicitly identify, analyze, and define the problem of cross-view sparsity variations in multi-view data, and to propose a dedicated framework SparseMVC that offers a targeted and principled solution.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20']

Section: Related Work
For multi-view fusion, early methods [22,23] assumed equal importance of views, ignoring view heterogeneity. Hence, dynamic weighting approaches have emerged, with attention-based methods [24,25] leading, alongside loss optimization [26,27], kernel techniques [28,29], and subspace methods [30]. However, uniform view-level weighting fails to address intra-view variability, highlighting the need for sample-level dynamic weighting. Trust-based methods [31,32,33,34] excel in supervised scenarios. For multi-view clustering, sample-adaptive fusion [35] has been proposed using Laplacian matrix divergence. In contrast, our framework dynamically computes sample weights via correlation calculation without additional loss, while adopting a decoupled design with independent global and view-specific autoencoders. More details can be found in Appendix A.2.
Sparse representation effectively captures essential features in sparse data by enforcing sparsity constraints [36] but struggles to handle the non-linear structures common in multi-view datasets [37]. Autoencoders excel at learning non-linear latent features [38,39], yet their lack of sparsity enforcement limits their adaptability to varying sparsity rates across views. Sparse autoencoders combine the strengths of both sparse representation and standard autoencoders by incorporating sparsity constraints into the hidden layers [40], which have become a research focus in multi-view learning [9,41].
this section cite: ['b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b8', 'b40']

Section: Method
Correlation-
this section cite: []

Section: Informed Sample Reweighting
View-1: Text
this section cite: []

Section: View-n: Image
A beautifully crafted upright piano with a closed lid.
this section cite: []

Section: Sparse Encoders Sparse Decoders
Cross-view This section sequentially introduces the three key submodules of SparseMVC, as illustrated in Fig. 2. It begins by utilizing sparse autoencoders with adaptive constraints, which dynamically adjust the coding strategy based on the probed s v , to generate latent features (Z), making the reconstructed features ( X) approximate the original input features (X). Subsequently, the correlation between the early-fused global features ( Z) and view-specific features ({Z v } n v=1 ) guides the computation of sample-level weights ({W v } n v=1 ) via the attention mechanism within the correlation-informed sample reweighting module. Finally, the cross-view distribution alignment module enhances clustering performance by setting the late-fused global features Y as the anchor latent representation, and then simultaneously aligning the multi-view feature distribution between Y and each view-specific compressed feature ({Y v } n v=1 ). The algorithm of the framework can be found in Appendix A.1.
this section cite: []

Section: Sparse Autoencoder with Adaptive Constraints
To handle varying view sparsity rates, we propose the sparse autoencoder with adaptive constraints (SAA), extending traditional sparse autoencoders. SAA employs an adaptive loss function that integrates reconstruction and sparsity-aware entropy-matching as distinct constraints, wherein the adjustment is dynamically guided by view sparsity ratios formulated as prior knowledge.
The reconstruction loss, typically measured by mean squared error (MSE), quantifies the difference between the reconstructed output xv j and the input x v j for the j-th sample:
L v recon = 1 N N j=1 xv j -x v j 2 , (2
)
where N is the number of samples (batch size) for view v. Motivated by the widespread presence of sparsity variations in different views, our aim is to design a function that is positively correlated with s v , allowing adaptive adjustments to both the encoder type and the strength of the sparsity constraints. We scale by f (s v ) outside the loss rather than tuning ρ in Eq. ( 4), please refer to Appendix B.2. The design of the adaptive weighting factor f (s v ) follows a ReLU-like approach, adjusting the strength of L v entropy based on the probed input dataset sparsity s v for view v:
f (s v ) = 0, if s v ≤ θ, sv-θ 1-θ , if s v > θ,(3)
where the default value of θ is 0.01. Selecting the threshold θ for the sparsity ratio s v is based on the actual cross-view sparsity distribution of each dataset. For most datasets, the sparsity ratios exhibit a skewed distribution, with a significant concentration of both high and low values, as shown in Fig. 1 and Table 1. A θ value of 0.01 effectively captures these low-sparsity views. To enforce sparsity constraints, the entropy-matching loss is defined using Kullback-Leibler (KL) [42] divergence, encouraging the activations ĥv k in the hidden layer to align with a target sparsity level ρ. The entropy-matching loss and the sparsity loss derived from it are formulated as follows:
L v sparse = f (s v ) • L v entropy = f (s v ) • H k=1 (ρ log ρ ĥv k + (1 -ρ) log 1 -ρ 1 -ĥv k ),(4)
where H refers to the number of units in the hidden layers of each sparse encoder, and ρ is the target sparsity level. ĥv k represents the average activation of the k-th hidden unit for view v, which is clamped to lie strictly within the open interval from zero to one. Following [40], ρ is set to 0.05, which is a well-validated choice that balances sparsity and the learning capacity of the autoencoder, allowing it to effectively capture key features in the data while avoiding overfitting to irrelevant features. The average activation is computed by:
ĥv k = 1 N N j=1 σ W v k x v j + b v k , (5
)
Adaptive Constrains
this section cite: ['b41', 'b39']

Section: Reconstruction Loss

this section cite: []

Section: Sparse Representation Loss

this section cite: []

Section: Sparse Autoencoders View 1~V Sparsity

this section cite: []

Section: Local View

this section cite: []

Section: Global

this section cite: []

Section: View

this section cite: []

Section: Local View
Figure 3: Sparse autoencoder with adaptive constraints.
where W v k is the weight matrix, b v k is the bias term for the k-th hidden unit in the v-th view, x v j is the input feature, and σ(•) denotes the ReLU activation function. When s v ≤ θ, L v entropy is deactivated (f (s v ) = 0) and the sparse autoencoder degenerates into a standard autoencoder. Conversely, for input views where s v > θ, f (s v ) exhibits a linear increase with s v , ensuring that the sparsity constraint becomes more prominent for highly sparse inputs. Based on the above, the loss function for SAA is then formulated as follows:
L SAA = V v=1 L v recon + L v sparse .(6)
As shown in Fig. 3, this piecewise linear design dynamically aligns the sparsity constraint with the input sparsity level and customizes the encoding strategy for each view.
this section cite: []

Section: Correlation-Informed Sample Reweighting
While SAA balances reconstruction and sparsity, it introduces sample-specific encoding inconsistencies due to differences in sparsity. Additionally, while networks adjust weights through layer updates, this weighting alone cannot address the lack of communication between autoencoders. On the above bases, we drew inspiration from the concept of multi-head attention and designed the correlationinformed sample reweighting module (CSR), which leverages correlations between the early-fused global features ( Z) and the view-specific local features (Z v ), then computes sample-specific weights. This cascading design is devised to achieve two objectives: mitigating the encoding inconsistencies introduced by SAA and leveraging globally fused features, which preserve the relatively high-fidelity patterns of the original data structure, to supervise the computation of view correlations.
CSR adopts a simplified structure inspired by multi-head attention, which captures similar effects while deviating from the standard formulation. Initially, CSR takes Z ∈ R N ×F and Z v ∈ R N ×F as input, and projects them into the query, key, and value spaces through parallel linear transformations:
Q = ZW Q , K v = Z v W K , V v = Z v W V ,(7)
where W Q , W K , W V ∈ R F ×F are the learnable weight matrices, generating the query matrix Q ∈ R N ×F , which encapsulates global semantic information, and the key matrix K v ∈ R N ×F , which capture view-specific features for each view. Our primary objective is to quantify inter-view relationships by evaluating attention scores between queries and keys, without generating new feature representations, thus omitting the value matrix V v . To compute the correlation between Z and Z v , we define the correlation score C v ∈ R N for the v-th view based on Einstein summation convention as:
C v = F f =1 Q f • (K v f ) T √ F ,(8)
where √ F is the scaling factor equals to the square root of the dimension of the key vector. The correlation scores are normalized via the softmax function to produce sample-specific W v ∈ R N as:
W v = exp(C v ) V v=1 exp(C v ) ,(9)
that dynamically adjust the contribution of each corresponding sample in respective autoencoders.
this section cite: []

Section: Cross-view Distribution Alignment
Aligning features across views is a fundamental challenge in multi-view learning, as it is crucial for leveraging the complementary information provided by diverse views. The cross-view distribution Alignment module (CDA) addresses this issue by performing contrastive learning between the latefused global features (Y ) and the compressed features of individual views (Y v ), ensuring effective alignment of multi-view features within a unified and shared latent space.
To mitigate the risk of dimensional collapse during alignment, potentially caused by an excessively large latent space, we introduce a compression layer before feeding the encoded view-specific features into the CDA. More details are provided in Appendix B.1. Specifically, Y v ∈ R N ×F is obtained from Z v via the compression layer. In parallel, the global features Y ∈ R N ×F are as follows:
Y = F V v=1 W v Z v ,(10)
where fusion function F represents the late fusion layers and ensures that the transformed dimension matches Y v . The similarity matrix S v between Y and Y v is defined as:
S v = Y • (Y v ) T τ ,(11)
with τ denoting a temperature parameter that scales the similarity values. The sample pairs position indices in S v are defined as p and q. Positive pairs, which correspond to the same samples across views , are represented by the diagonal elements of S v , denoted as S p,p v . Negative pairs, which involve different samples across views, are identified using a mask matrix M v ∈ R N ×N , where M p,q v = 1 if p ̸ = q, and M p,q v = 0 otherwise. The contrastive loss for each sample is:
L p,v con = -log exp(S p,p v ) N q=1 exp(S p,q v ) • M p,q v ,(12)
where exp(S p,p ) quantifies the similarity of positive pairs, and the denominator aggregates the exponential similarities of all pairs, weighted by the mask matrix M v . The overall CDA loss across all views is obtained by summing the individual losses for each view and averaging over all samples:  For contrastive learning, when samples of the same class are clustered in one view, the attraction exerted by positive pairs propagates to other views. In contrast, although the distinction between samples does not necessarily imply class disparity, repulsion among negative pairs enables view with greater discriminative power to transmit class separations to other views. This results in a mechanism that transforms both the alignment and misalignment information at the cross-view sample level into an objective, aiming to minimize intra-class while maximizing inter-class distances.
L CDA = V v=1 1 N N p=1 L p,v con . (13
)
Regarding the role of CDA, the global view serves as an anchor, which is compared in parallel against each local view. To minimize the overall contrastive loss, the sample distribution of the global view is concurrently attracted toward all local views, thereby encouraging the features of each sample to converge more tightly in the latent space. From the perspective of the entire latent space, this process effectively facilitates overall distribution alignment. The rationale behind utilizing local views for distribution alignment with the global view to enhance class separability lies in the ability to leverage easily classified samples from one view to improve the distinguishability of harder-to-classify samples in another. The reasoning above, together with Fig. 4, illustrate the working principle of the CDA: utilizing contrastive learning as a tool, through the process of aligning the distribution of sample features across views, enhancing the differentiation of difficult-to-classify samples in one view by leveraging easy-to-classify samples in another, and ultimately achieving the goal of optimal alignment of features in the shared latent space.
this section cite: []

Section: The Overall Loss Function of SparseMVC
The total loss L total comprises the adaptive sparse autoencoder loss L SAA in Eq. ( 6), preserving data fidelity and enforcing structured sparsity via L recon and L entropy , and the cross-view alignment loss L CDA in Eq. ( 13), ensuring consistent clustering across views:
L total = V v=1 L v recon + f (s v ) • L v entropy + λ CR • L CDA ,(14)
where λ CR is the constraint ratio coefficient that controls the trade-off between L SAA and L CDA .
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Settings
Compared Methods. Our proposed method is compared against the following 12 state-of-theart multi-view clustering methods based on deep learning. DSMVC [43], COMPLETER [44], DCP [45], CVCL [46] and SCMVC [47] focus on dynamic contrastive learning; MFLVC [48], GCFAgg [3], DealMVC [49] combine feature fusion and consistency; DSMVC [43] and SDMVC [50] enhance consistency through discriminative learning; CPSPAN [51] and MVCAN [52] apply proxy supervision and prototype alignment.
this section cite: ['b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b2', 'b48', 'b42', 'b49', 'b50', 'b51']

Section: Benchmark Datasets.
The selected datasets span diverse domains: Image datasets include MSRCV1 [53] focusing on objects and scenes, Dermatology [54] on medical images, Out-Scene [55] on natural scenes, and ALOI-100 [56] on object recognition. Image-text datasets include Wikipediafoot_0 , which provides website crossmodal data. Omics datasets include LGG [57] focusing on brain tumor genomics and BRCA [58] on breast cancer genomics. Synthetic3d [59] supports 3D object modeling and recognition. Detailed properties of datasets are listed in Table 2.
Evaluation Metrics. Accuracy (ACC) evaluates alignment with ground truth, normalized mutual information (NMI) measures shared information, purity (PUR) assesses cluster homogeneity. Adjusted Rand index (ARI), measuring clustering similarity, is partially utilized in experiments. For all metrics, higher values indicate better performance.
this section cite: ['b52', 'b53', 'b54', 'b55', 'b56', 'b57', 'b58']

Section: Implementation Details
All experiments were conducted using Python 3.8.15 and PyTorch 1.13.1+cu116 on a Windows PC equipped with an AMD Ryzen 9 5900HX CPU, 32GB RAM, and an Nvidia RTX 3080 GPU (16GB). Models were trained using the Adam optimizer [60], a learning rate of 0.003, and a fixed seed of 50, with batch size equal to the dataset's sample count. Pre-training was performed uniformly for 300 epochs, while alignment training was conducted for 300 epochs for datasets with less than 2500 samples and 1000 epochs for larger datasets. For clustering, k-means [61] was applied with the number of clusters equal to the dataset categories and 100 initializations. During pre-training, global features Z v derived from early fusion were used, while alignment training used late fusion features Y . Metrics were calculated as the average of 10 runs in the final epoch, with no fine-tuning performed for specific datasets. To ensure fairness, the hyperparameters for the comparison methods were determined based on either the default global settings or the configuration of the first dataset.
this section cite: ['b59', 'b60']

Section: Comparative Results Analysis
Table 3 and 4 summarize the comparative results, leading to the following conclusions:
(1) Our method achieves state-of-the-art performance across eight diverse multi-view datasets, along with larger-scale datasets as shown in Table 8. These results validate the versatility of our approach and highlight its potential for a wide range of downstream tasks. In comparison, other approaches such as SCMVC, MVCAN, and CPSPAN achieved relatively good results on specific datasets but failed to maintain an advantage due to their limited generalizability across other datasets.
(2) Our method demonstrates stability in clustering performance, as evaluation metrics oscillate upward within a small range and stabilize with increasing epochs, showcasing robust results. In contrast, methods like SURE and DealMVC on ALOI-100 or CVCL and CPSPAN on Wikipedia fail to stabilize, with metrics either degrading significantly after peaking or fluctuating dramatically without consistent improvement. During the early stages of contrastive training when the embedding space's distribution remains uneven, potentially causing abrupt gradient fluctuations. To alleviate the instability, we adopt a dynamic fusion strategy in Sec. 3.2 and a pre-training approach in which only the autoencoder is trained initially.
(3) Although our model is specifically designed to address the challenge: across-view sparsity variations, it also achieves superior performance on dense datasets, such as BRCA, thereby demonstrating its adaptability and broad applicability. This is attributed to the inherent flexibility of our designed sparse autoencoder, which adaptively transitions into a conventional autoencoder when confronted with dense data, thereby prioritizing the reconstruction objective with greater emphasis.
(4) Compared to recent state-of-the-art methods, our approach demonstrates superior feature representation performance by producing clearer boundaries and more compact clusters, as shown in Fig. 5. Its most notable advantage is the ability to disentangle intra-class clusters while preserving inter-class separability, which leads to better scalability and robustness when handling data with large cross-view distribution disparities. By analyzing the training curve in Fig. 6, we observe the following key points: (1) The evaluation metrics generally exhibit an oscillatory increase followed by stabilization, with this stability being maintained as the number of training epochs progresses. This observation underscores the model's convergence and its robustness in maintaining stable clustering performance despite optimization challenges. (2) During the early stages of the alignment training phase, the evaluation metrics exhibit a brief dip, which is quickly followed by a recovery and subsequent stabilization at a higher level. The fluctuations observed in the evaluation metrics can be attributed to L CDA in Eq. ( 13), which necessitates the initialization of all network parameters except the autoencoder. (3) Despite significant fluctuations in the loss for small datasets, intriguingly, these fluctuations do not propagate to the evaluation metrics, suggesting that our approach effectively mitigates the influence of instability in the optimization algorithm on the clustering structure.
this section cite: []

Section: Convergence Analysis

this section cite: []

Section: Ablation Study
Loss Function To ensure the rigor of ablation experiment, we selected three datasets with significant variations in view sparsity as shown in Fig. 1, ensuring the functionality of the SAA. We assessed the effectiveness of individual losses in the total loss Eq. ( 14) of SparseMVC, as presented in Table 5. Specifically, we use L recon as the baseline and find that adding either L entropy or L CDA improves performance. L entropy yields substantial improvements on ALOI-100 and Dermatology, which have stronger variations in view sparsity. These improvements highlight the effectiveness of adaptive encoding and cross-view distribution alignment as robust constraints that contribute positively to the overall model training process. Moreover, when all losses are activated simultaneously, the model achieves optimal performance, suggesting that L entropy and L CDA complement each other synergistically. Notably, their integration does not introduce any mutual interference, further underlining the coherence and compatibility of these objectives in driving superior learning outcomes.
Components Our approach focuses on view-level structural sparsity, specifically the sparsity variation across views within the same multi-view data. This differs from data-level sparsity methods, which typically apply uniform sparse encoding to all views without explicitly considering the heterogeneity of inter-view sparsity. To further validate the effectiveness of the proposed SAA module, we extend the ablation study in Table 5 by introducing two additional comparative settings: (i) uniformly sparse encoding applied to all views, which mimics methods designed for data-level sparsity; and (ii) adaptive encoding tailored to each view. On top of this, we also ablate the CSR module, which reweights the local features during the late fusion stage. Regardless of whether the CSR module is applied, the results in Table 6 show that the proposed SAA, which leverages adaptive autoencoders, remains effective and consistently achieves superior performance. Results further confirm that the effectiveness arises from the synergy between adaptive encoding and sample reweighting, rather than from the use of sparse autoencoders alone. Collectively, these findings confirm that our method is robust to varying sparsity across views and that the SAA and CSR modules function synergistically rather than independently.
this section cite: []

Section: Parameter Sensitivity Analysis
We selected the temperature parameter τ in L CDA and the constraint ratio coefficient λ CR , the ratio of L SAA to L CDA , as the two parameters for analysis. Both coefficients were set to gradually increase from 0.1 to 1.9 with a step size of 0.3. In Fig. 7, we can discover that the accuracy initially increases and then decreases as τ increases, remaining relatively stable within a range around 1.0. The influence of λ CR on clustering performance is comparatively minor, with a negligible impact when τ is within the range of 0.4 to 1.0. In light of Sec. 3.3, L CDA incorporates the smoothing property of the logarithmic function, which diminishes the direct effect of λ CR adjustments on the gradient. In contrast, changes to τ significantly influence clusters separability and alignment performance by modulating the nonlinear response of the softmax function. A smaller τ enhances class separability, while a larger τ emphasizes global consistency. Therefore, we set the default values of τ and λ CR to 1.0 in the loss function.
Our method is largely insensitive to hyperparameter changes. To begin with, significant performance degradation only occurs when the temperature coefficient τ ≤ 0.4 or ≥ 1.6. The extremely small value of 0.1 is chosen to probe the lower bound of performance degradation and is rarely used in practical applications. Furthermore, the performance fluctuation mainly occurs along the τ -axis, whereas it remains relatively insensitive to changes in the constraint ratio coefficient λ CR . In practice, it is the relative weighting between loss terms that is more commonly adjusted. In addition, the accuracy-axis was intentionally truncated to better highlight the differences, accentuating the visual disparity. Finally, regarding the concern that hyperparameters are not easy to tune in practice, our method maintains stable performance even under noticeable loss fluctuations, as illustrated in Fig. 6.
this section cite: []

Section: Conclusion
This paper highlights a frequently overlooked issue in deep multi-view learning: varying sparsity ratios across views. Therefore, we systematically define, quantify, and analyze cross-view sparsity variation as a fundamental characteristic of multi-view data. Our entire framework, SparseMVC, is designed to handle view-level sparsity variations with a complete data-driven and tightly integrated architecture. To tackle sparsity variation, we propose an adaptive encoding strategy that uses the sparsity ratio of each view as prior knowledge, enabling the encoder to switch between standard and sparse forms with appropriate constraint strengths. Additionally, we introduce a series of interdependent mechanisms to mitigate the side effects of representational divergence caused by nonuniform encoding. Specifically, a correlation-guided fusion strategy leverages global-to-local feature relationships from the early stages to guide the weighting of local features in late fusion. Moreover, a distribution alignment module structurally constrains the fused representations, enhancing cross-view complementarity in the final stage. Comprehensive experiments and detailed dissections of each module validate the efficacy of SparseMVC. We hope this work inspires greater attention to the intrinsic characteristics of data and to the design of architectures driven by data.
this section cite: []

Section: References
Ref_id:b0 Title: Align then fusion: Generalized large-scale multi-view clustering with anchor matching correspondences Year: (2022)
Ref_id:b1 Title: Inclusivity induced adaptive graph learning for multi-view clustering Year: (2023)
Ref_id:b2 Title: Gcfagg: Global and cross-view feature aggregation for multi-view clustering Year: (2023)
Ref_id:b3 Title: Contrastive and view-interaction structure learning for multi-view clustering Year: (2024)
Ref_id:b4 Title: Dual alignment feature embedding network for multi-omics data clustering Year: (2025)
Ref_id:b5 Title: Learning deep sparse regularizers with applications to multi-view clustering and semi-supervised classification Year: (2021)
Ref_id:b6 Title: Consensus graph learning for multi-view clustering Year: (2021)
Ref_id:b7 Title: Enhanced tensor low-rank and sparse representation recovery for incomplete multi-view clustering Year: (2023)
Ref_id:b8 Title: Sc-vae: Sparse coding-based variational autoencoder with learned ista Year: (2024)
Ref_id:b9 Title: Convex sparse spectral clustering: Single-view to multi-view Year: (2016)
Ref_id:b10 Title: Localized sparse incomplete multi-view clustering Year: (2022)
Ref_id:b11 Title: Sparse multi-view hand-object reconstruction for unseen environments Year: (2024)
Ref_id:b12 Title: Cancer-drug response prediction via feature aggregation and association graph learning Year: (2025)
Ref_id:b13 Title: Don't just chase Year: (2025)
Ref_id:b14 Title: The mnist database of handwritten digit images for machine learning research Year: (2012)
Ref_id:b15 Title: Bag-of-visual-words and spatial extensions for land-use classification Year: (2010)
Ref_id:b16 Title: What are you talking about? text-to-image coreference Year: (2014)
Ref_id:b17 Title: Gse: a comprehensive database system for the representation, retrieval, and analysis of microarray data Year: (2008)
Ref_id:b18 Title: An analysis of single-layer networks in unsupervised feature learning Year: (2011)
Ref_id:b19 Title: A survey of multi-view representation learning Year: (2018)
Ref_id:b20 Title: Multi-view graph learning by joint modeling of consistency and inconsistency Year: (2022)
Ref_id:b21 Title: Combining labeled and unlabeled data with co-training Year: (1998)
Ref_id:b22 Title: Multi-view clustering via canonical correlation analysis Year: (2009)
Ref_id:b23 Title: An attentionbased collaboration framework for multi-view network representation learning Year: (2017)
Ref_id:b24 Title: Multi-view graph convolutional networks with attention mechanism Year: (2022)
Ref_id:b25 Title: Generative partial multi-view clustering with adaptive fusion and cycle consistency Year: (2021)
Ref_id:b26 Title: Efficient and effective one-step multiview clustering Year: (2023)
Ref_id:b27 Title: Auto-weighted multi-view clustering via kernelized graph learning Year: (2019)
Ref_id:b28 Title: Multiple kernel clustering with adaptive multi-scale partition selection Year: (2024)
Ref_id:b29 Title: Multi-view subspace clustering via partition fusion Year: (2021)
Ref_id:b30 Title: Multimodal dynamics: Dynamical fusion for trustworthy multimodal classification Year: (2022)
Ref_id:b31 Title: Dpnet: Dynamic poly-attention network for trustworthy multi-modal classification Year: (2023)
Ref_id:b32 Title: Hierarchical attention learning for multimodal classification Year: (2023)
Ref_id:b33 Title: Trusted mamba contrastive network for multi-view clustering Year: (2025)
Ref_id:b34 Title: Sample-level weights learning for multi-view clustering on spectral rotation Year: (2023)
Ref_id:b35 Title: Sparse coding with an overcomplete basis set: A strategy employed by v1? Year: (1997)
Ref_id:b36 Title: Linear spatial pyramid matching using sparse coding for image classification Year: (2009)
Ref_id:b37 Title: Autoencoders, minimum description length and helmholtz free energy Year: (1993)
Ref_id:b38 Title: Reducing the dimensionality of data with neural networks Year: (2006)
Ref_id:b39 Title: Sparse autoencoder. CS294A Lecture Notes Year: (2011)
Ref_id:b40 Title: Structured autoencoders for subspace clustering Year: (2018)
Ref_id:b41 Title: On information and sufficiency Year: (1951)
Ref_id:b42 Title: Deep safe multi-view clustering: Reducing the risk of clustering performance degradation caused by view increase Year: (2022)
Ref_id:b43 Title: Completer: Incomplete multi-view clustering via contrastive prediction Year: (2021)
Ref_id:b44 Title: Dual contrastive prediction for incomplete multi-view representation learning Year: (2022)
Ref_id:b45 Title: Deep multiview clustering by contrasting cluster assignments Year: (2023)
Ref_id:b46 Title: Self-weighted contrastive fusion for deep multi-view clustering Year: (2024)
Ref_id:b47 Title: Multi-level feature learning for contrastive multi-view clustering Year: (2022)
Ref_id:b48 Title: Dealmvc: Dual contrastive calibration for multi-view clustering Year: (2023)
Ref_id:b49 Title: Self-supervised discriminative feature learning for deep multi-view clustering Year: (2023)
Ref_id:b50 Title: Deep incomplete multi-view clustering with cross-view partial sample and prototype alignment Year: (2023)
Ref_id:b51 Title: Investigating and mitigating the side effects of noisy views for self-supervised clustering algorithms in practical multi-view scenarios Year: (2024)
Ref_id:b52 Title: Locus: Learning object classes with unsupervised segmentation Year: (2005)
Ref_id:b53 Title: Learning differential diagnosis of erythemato-squamous diseases using voting feature intervals Year: (1998)
Ref_id:b54 Title: Modeling the shape of the scene: A holistic representation of the spatial envelope Year: (2001)
Ref_id:b55 Title: The amsterdam library of object images Year: (2005)
Ref_id:b56 Title: Cancer Genome Atlas Research Network. Comprehensive, integrative genomic analysis of diffuse lower-grade gliomas Year: (2015)
Ref_id:b57 Title: Tcga-network, comprehensive molecular portraits of human breast tumours Year: (2012)
Ref_id:b58 Title: Co-regularized multi-view spectral clustering Year: (2011)
Ref_id:b59 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b60 Title: Some methods for classification and analysis of multivariate observations Year: (1967)
Ref_id:b61 Title: Robust multi-view clustering with incomplete information Year: (2022)
Ref_id:b62 Title: Spgmvc: Multiview clustering via partitioning the signed prototype graph Year: (2024)
Ref_id:b63 Title: Multi-view graph convolutional networks with attention mechanism Year: (2022)
Ref_id:b64 Title: Evaluate then cooperate: Shapley-based view cooperation enhancement for multi-view clustering Year: (2024)
Ref_id:b65 Title: Trusted multi-view classification Year: (2021)
Ref_id:b66 Title: Look twice before you answer: Memoryspace visual retracing for hallucination mitigation in multimodal large language models Year: ()
Ref_id:b67 Title: Sparse clustering algorithm based on multi-domain dimensionality reduction autoencoder Year: (2024)
Ref_id:b68 Title: Multi-view k-means clustering with adaptive sparse memberships and weight allocation Year: (2020)
Ref_id:b69 Title: Multi-view clustering via pairwise sparse subspace representation Year: (2015)
Ref_id:b70 Title: Multiview spectral clustering via sparse graph learning Year: (2020)
Ref_id:b71 Title: Augmented sparse representation for incomplete multiview clustering Year: (2022)
Ref_id:b72 Title: A mathematical theory of communication Year: (1948)
Ref_id:b73 Title: On feature decorrelation in self-supervised learning Year: (2021)
Ref_id:b74 Title: Attribute-based classification for zero-shot visual object categorization Year: (2013)
Ref_id:b75 Title: Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy Year: (2005)
Ref_id:b76 Title: Optimally sparse representation in general (nonorthogonal) dictionaries via ℓ1 minimization Year: (2003)
Ref_id:b77 Title: Compressed sensing Year: (2006)
