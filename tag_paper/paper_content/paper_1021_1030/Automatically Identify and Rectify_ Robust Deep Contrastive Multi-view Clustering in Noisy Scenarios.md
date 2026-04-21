Title: Automatically Identify and Rectify: Robust Deep Contrastive Multi-view Clustering in Noisy Scenarios
Abstract: Leveraging the powerful representation learning capabilities, deep multi-view clustering methods have demonstrated reliable performance by effectively integrating multi-source information from diverse views in recent years. Most existing methods rely on the assumption of clean views. However, noise is pervasive in real-world scenarios, leading to a significant degradation in performance. To tackle this problem, we propose a novel multi-view clustering framework for the automatic identification and rectification of noisy data, termed AIRMVC. Specifically, we reformulate noisy identification as an anomaly identification problem using GMM. We then design a hybrid rectification strategy to mitigate the adverse effects of noisy data based on the identification results. Furthermore, we introduce a noise-robust contrastive mechanism to generate reliable representations. Additionally, we provide a theoretical proof demonstrating that these representations can discard noisy information, thereby improving the performance of downstream tasks. Extensive experiments on six benchmark datasets demonstrate that AIRMVC outperforms state-of-the-art algorithms in terms of robustness in noisy scenarios. The code of AIRMVC are available at https://github.com/xihongyang1999/AIRMVC on Github.

Section: 
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). In the diagram, the areas marked with red exclamation points indicate instances where sensor failures or malfunctions at specific moments lead to data corruption. Compared to other views, these instances are considered noisy data.
this section cite: []

Section: Introduction
In real-world scenarios, multi-source information data is prevalent. To effectively handle such data, Multi-View Clustering (MVC) has emerged as a powerful unsupervised method. In recent years, MVC has gained significant attention and has become a prominent focus of research. Existing MVC methods can be broadly categorized into two main groups. The first group comprises traditional approaches, including Multiple Kernel Clustering (MKC) (Liu et al., 2021), Non-negative Matrix Factorization (NMF) (Wen et al., 2018), subspace clustering (Zhou et al., 2019), and graph-based clustering (Liu et al., 2022a). With the advancement of deep neural networks, deep multi-view clustering algorithms (Yang et al., 2023b;Sun et al., 2024), representing the second category of MVC methods, have garnered significant attention from researchers.
While existing MVC algorithms have demonstrated notable clustering performance, they predominantly rely on the assumption that the features from all views are clean. These approaches generally adopt a standard workflow: representations are first extracted via an encoder, followed by a feature fusion strategy, and then utilized for downstream clustering tasks. The integration of clean views enables the exploitation of complementary information from different perspectives of the same sample, resulting in enhanced performance compared to single-view clustering methods. This complementary information across views is instrumental in revealing the underlying cluster structures within the data. How can the complementary information across multiple views be effectively captured? Contrastive learning presents a paradigm that utilizes self-supervised techniques to learn cross-view consistency, ensuring coherent predictions across diverse views (Lu et al., 2024;Yang et al., 2023b;Xu et al., 2022b). Alternatively, self-training explores consistency by generating a unified cluster partition, thereby facilitating the discovery of complementary information (Xu et al., 2022a;Wang et al., 2021).
Although the aforementioned methods have demonstrated promising results, we identified notable limitations when applied to real-world scenarios involving noisy data. Fig. 1 illustrates an example of noisy data in a multi-view setting, where three views are presented, and random noise exists within some of them. Moreover, we observed a significant performance degradation in existing methods under such noisy conditions (Tab. 4). The noisy data not only fail to contribute positively during multi-view feature fusion but also disrupt the underlying cluster structures. Moreover, the erroneous influence of noisy data introduces considerable bias in the optimization process, thereby diminishing the advantages of complementary information across views. Consequently, the performance of these methods may even fall below that of single-view models. Detailed experimental evidence supporting this claim is presented in Section. 5.3. Recently, some noisy-based deep multi-view clustering methods have been proposed to alleviate the problem. RMCNC (Sun et al., 2024) designed a noise-tolerance contrastive loss to mitigate the impact for noisy correspondence. MVCAN (Xu et al., 2024) employed un-shared network structure and designed a two-level optimization for multi-view clustering. Although a large improvement has been made, the exploration of noisy data in those methods remains focused on enhancing feature robustness. However, they have yet to develop dedicated frameworks for the identification and rectification of noisy data.
To automatically identify and rectify the noisy data in multiview scenario, we propose a novel deep contrastive multiview clustering framework, termed AIRMVC. Specifically, we reformulate noise identification as an anomaly identification problem and introduce Gaussian Mixture Model (GMM) to address this problem. By substituting the latent variable of GMM with the soft prediction, we enable a dynamic update of GMM parameters. Based on the identification results, we introduce a hybrid rectification strategy that employs an interpolation mechanism to alleviate the ad-verse effects of noisy data, thereby enhancing the robustness of the correction process. In this way, the adverse effects of noisy data could be mitigated. Furthermore, by carefully refining the soft predicted distributions, we design a noiserobust contrastive mechanism to enhance the discriminative capacity of the learned representations. Furthermore, we conduct a theoretical investigation of the contrastive mechanism to validate the stability and robustness of the learned representations. Extensive experiments conducted on six benchmark datasets demonstrate the effectiveness and robustness of our proposed method. The key contributions of our paper are summarized as follows:
• We reformulate the noise identification as an anomaly identification problem, solving it by GMM. Based on the results, we propose a hybrid rectification strategy to automatically correct the noisy data.
• A noise-robust contrastive mechanism is proposed to generate more reliable representations. Besides, we theoretically prove that the generated representations could discard noisy information to benefit the downstream task.
• We conduct extensive experiments on six benchmark datasets to verify the effectiveness and robustness of AIRMVC.
this section cite: ['b21', 'b38', 'b63', 'b27', 'b24', 'b37', 'b27', 'b41']

Section: Preliminary & Problem Definition
In this paper, we focus on the multi-view clustering task with noisy inputs, as noise is a common occurrence in multisource information inputs in real-world scenarios. To address this challenge, we aim to enhance the robustness of the network by automatically identifying and rectifying noisy data in an unsupervised multi-view setting. For simplicity, we provide the following symbolic definitions. Given a dataset {x v } V v=1 with V views and N samples, we define E v i and y v i as the extracted representations and the soft predictions of class probabilities for each sample, respectively. The encoder network and decoder network are denoted as
F v (x v ; Θ v ) and G v (E v ; Φ v ), respectively. Additional nota- tions are summarized in Tab. 5 in the Appendix.
Most existing multi-view clustering (MVC) methods operate under the assumption that the input multi-view data is both complete and consistent. In such idealized scenarios, the primary learning paradigm in MVC involves learning representations for V independent views, followed by the application of various fusion strategies to effectively extract and utilize shared information across the views. Although these models demonstrate strong performance under ideal conditions, we observe a significant decline in robustness when low-quality, noisy views are introduced, leading to substantial performance degradation. However, noise is a pervasive issue in real-world scenarios. To further illustrate this problem, we conduct toy experiments involving clustering tasks on BBCSport and WebKB datasets under different scenarios, including noisy multi-view, clean single-view, and clean multi-view conditions. From the experimental results are presented in Fig. 3 and Tab. 4, it can be observed that when confronted with noisy data, the model's performance degrades drastically, even performing worse than single-view clustering. Therefore, it is critical to mitigate the adverse effects of noisy data in multi-view clustering tasks. In the following sections, we detail our proposed strategies for the automatic identification and rectification of noise.
this section cite: []

Section: Methodology
To mitigate the negative impact of noisy data, we propose a robust Automatic Identification and Rectification deep contrastive Multi-View Clustering framework with noisy view, which termed AIRMVC. The overall framework of AIRMVC is presented in Fig. 2. Specifically, we begin with providing the detailed descriptions of our AIRMVC, including three components, i.e., noisy identification, hybrid rectification, and the noise-robust contrastive mechanism. Then, we define the objective functions to optimize the whole model. Finally, we present the theoretical analysis of our designed noise-robust contrastive component.
this section cite: []

Section: Noisy Identification
For a given multi-view dataset {x v } V v=1 , we utilize an encoder network to generate the representations, expressed as
E v = F v (x v ; Θ v ).
For each representation, we first process it using a multi-layer perceptron (MLP) layer as the projector. Subsequently, we model its distribution using a Gaussian Mixture Model (GMM). This process can be described as follows:
p(E) = K k=1 p(E, q = k) = K k=1 p(q = k)N (E|µ k , σ k ),(1)
where q ∈ {1, 2, . . . , K} represents the discrete latent variables. µ k and σ k denote the mean and variance, respectively. Assuming that q follows a uniform distribution, the probability of q = k could be expressed as p(q = k) = 1 K . Based on this, the posterior probability of assigning x i to the k-th cluster can be calculated as:
χ ik = p(q i = k|x i ) ∝ N (x i |µ k , σ k ).
(2)
The discrete variables q can correspond directly to the labels in an ideal condition. The entire process can be optimized and solved using the Expectation-Maximization (EM) algorithm. However, in real-world scenarios, noisy data inputs are pervasive and unavoidable, leading to biases in the optimization process. Thus, the identification and rectification of noisy data are critically important. In unsupervised scenarios, identifying noise remains a challenging task. To address this issue, we establish a connection between the latent variable q and the model prediction y in an unsupervised manner. Specifically, we replace the assignment p(q i = k|x i ) with the soft prediction of the network p(y i = k|x i ). The parameters of the GMM can then be computed as follows:
µ k = Norm i p(y i = k|x i )E i i p(y i = k|x i ) , σ k = i p(y i = k|x i )(E i -µ k )(E i -µ k ) T i p(y i = k|x i ) ,(3)
where Norm means the ℓ 2 normalization. Based on Eq. ( 3), we reformulate Eq. ( 2) by considering the intra-cluster distance. Since we implement ℓ 2 normalization, we have
(E -µ k ) T (E -µ k ) = 2 -2E T µ k , the process of Eq. (2)
could be expressed by:
χ ik = p(q i = k|x i ) = exp -(E i -µ k ) T (E i -µ k ) /2σ k k exp -(E i -µ k ) T (E i -µ k ) /2σ k = exp E T i µ k /σ k k exp E T i µ k /σ k .(4)
In this way, we obtain the soft prediction of a sample belonging to the k-th cluster (µ k ). Furthermore, we combine the model's predictions with the representation distribution in an unsupervised manner and update them using the GMM. By formulating the GMM to model the distribution of representations and soft predictions, we transform the noisy identification problem into an anomaly identification problem. In a multi-view setting, if a sample is clean, its soft predictions and cluster assignments should remain consistent across the different views. For a given sample x i , it is classified as either clean (normal data) or noisy (anomalous data). Based on the above analysis, we provide the conditional probability to determine the likelihood of x i containing clean information, which can be calculated as:
χ y=q|i = p(y i = q i |x i ) = exp E T i µ qi /σ qi k exp E T i µ k /σ k .(5)
After that, we introduce a two-component GMM to automatically identify the clean probability of a given sample, presented as:
p(χ y=q|i ) = p(χ y=q|i , a = 1) φi + p(χ y=q|i , a = 0) 1-φi(6)
where a = 1 represents the cluster of clean samples with a higher mean value, while a = 0 corresponds to the cluster with a lower mean value. Consequently, φ i can be interpreted as the probability of a sample x i being clean, whereas 1 -φ i denotes the probability of x i being noisy. Using Eq. ( 6), we can calculate the posterior probability that determines whether a sample x i is clean, which is presented in Fig. 2(a).
this section cite: []

Section: Hybrid Rectification Strategy Design
In Section 3.1, we estimate the probability φ v i of each sample being classified as clean in v-th view. Following this, we propose a hybrid rectification strategy to address noisy samples. The details of the strategy are illustrated in Fig. 2(b). Specifically, we utilize a combination of predicted soft distributions to perform noise rectification:
y v i = h(E v i ), m v i = φ v i × y v i + (1-φ v i ) × y 1 i , v = {2, . . . , V },(7)
where m v i represents the mixed soft prediction, and h(•) denotes the classifier head. We utilize a multilayer perceptron (MLP) with a softmax function as the backbone for h(•). In this paper, we focus on the unsupervised clustering task, where obtaining reliable supervisory information is particularly challenging. Following previous noise-robust MVC methods (Huang et al., 2020;Yang et al., 2023a;Sun et al., 2024;Yang et al., 2021), we assume the first view to be the clean view. Therefore, in Eq. ( 6), the mixed soft prediction is predominantly influenced by the clean samples. Conversely, as φ i approaches 0, the mixed soft prediction is adjusted based on the prediction of the first view.
With the mixed soft prediction, we could provide the following rectification loss:
L rs = 1 V -1 V v=2   - N i=1 K j=1 m v ij log y v ij   , (8
)
where y v represents the v-th soft prediction and v ∈ {2, . . . , V }. In this manner, the noisy data could be rectified by the cross-entropy loss. A more detailed experimental analysis of the hybrid rectification strategy is presented in Section 5.3.
this section cite: ['b13', 'b27', 'b42']

Section: Noise-Robust Contrastive Mechanism
Contrastive learning (Yang et al., 2024b) has been proven to be an effective technique for enhancing representation robustness. In this subsection, we propose a noise-robust contrastive mechanism to further mitigate the impact of noisy data in multi-view clustering tasks. In unsupervised multi-view noisy scenarios, constructing positive and negative sample pairs based solely on indices may result in incorrect pairings (Sun et al., 2024). Ensuring the reliability of sample pair construction is a significant challenge. To improve the accuracy of pair construction, we incorporate soft predictions as an additional validation criterion. Similar to previous methods, the first step is to calculate the similarity between samples across views:
s(E m i , E n j ) = sim(E m i , E n j ) ||E m i || 2 ||E n j || 2 ,(9)
where sim(•) denotes the similarity function, e.g., cosine similarity. Next, we present the noise-robust contrastive loss for i-th and j-th sample in m-th and n-th view as:
ℓ mn = I{(y m i ) T (y n j ) ≥ τ } log(1 -s(E m i , E n j )) + log(1 -s(E m j , E n i )) ,(10)
where τ is the confidence threshold used to control the construction of sample pairs for contrastive learning by selecting similar samples. As illustrated in Fig. 2(c), green denotes correctly constructed sample pairs, while red indicates incorrect ones. Our proposed strategy significantly improves the accuracy of sample pair construction, thereby enhancing the overall reliability of the contrastive learning process. Then, the robust contrastive loss for all cross-view is defined by:
L con = 1 V (V -1) V m=1 V n=1 ℓ mn , m ̸ = n.(11)
this section cite: ['b27']

Section: Objective Function
Our proposed AIRMVC is primarily optimized using three objective functions: reconstruction loss L rec , rectification loss L rs , and robust-noisy contrastive loss L con . To be specific, L rec denotes the reconstruction procedure with autoencoder network to learn the representations E in latent space. The process could be formulated as:
L rec = V v=1 N i=1 ∥x v i -G v (F v (x v i ; Θ v ); Φ v )∥ 2 2 , (12
)
where G v and F v are the decoder and encoder network for v-th view, respectively. In summary, the overall objective function of AIRMVC is:
L = L rec + α • L rs + β • L con ,(13)
where α and β are the trade-off hyper-parameters. By optimizing the overall objective function, AIRMVC could automatically identify and rectify the noisy data in unsupervised multi-view clustering task. Due to the space limited, we present the training algorithm in Alg. 1 in Appendix.
this section cite: []

Section: Theoretical Analysis
In this subsection, we examine the rationale behind our proposed noise-robust contrastive mechanism from a theoretical perspective. For clarity, we provide the following definitions of symbols.
Let x and x + denote the input sample and its positive sample in our noise-robust contrastive mechanism. y and y ′ represent the clean and noisy soft prediction, respectively. We consider E * as the representations by maximizing the mutual information between E and x+, i.e., E * = argmax E I(E, x + ). Besides, we define I(x; y|x + ) ≤ ϑ and I(x; y ′ |x + ) > η. For input sample x, the clean soft prediction y, and the noisy soft prediction y ′ , we have: Theorem 4.1. The representations E * retain clean information and discard noisy information, which can be presented as: I(x; y) -ϑ ≤ I(E * ; y) ≤ I(x; y), I(E * ; y ′ ) ≤ I(x; y ′ ) -η + ϑ.
Remark: For the input samples x, the corresponding positive samples x + and prediction y i , ϑ could be interpreted as the relatively small information gain contributed by the positive samples x + . For the fixed x and its positive samples x + , the information gain for the class prediction y is limited. In contrast, x + contributes greater information gain to the noisy prediction y ′ , which we regard as η. By calculating the mutual information I(x i ; y i ), Theorem. 14 demonstrates that the representations E * learned by the contrastive mechanism could discard the noisy information while preserving the clean information. The proof is provided in Section A.2.
this section cite: []

Section: Experiments
In this section, we perform a series of experiments to evaluate the effectiveness and advantages of our proposed method. Specifically, we aim to address the following research questions (RQs): RQ1: How does AIRMVC compare with other leading deep multi-view clustering techniques in terms of performance? RQ2: What art the impacts of the components of AIRMVC to enhance multi-view clustering results? RQ3: What clustering structures are identified by AIR-MVC? RQ4: What is the effect of hyper-parameters on the efficacy of AIRMVC? Evaluation Metrics: To provide a thorough evaluation of the model's clustering performance, we utilize three widely recognized metrics: Accuracy (ACC), Normalized Mutual Information (NMI), and Purity (PUR). To ensure a fair comparison, all methods are assessed across 10 independent runs, and the average results are reported.
Comparison algorithms: To showcase the broad applicability and superior performance of the proposed AIRMVC, we compare it against 11 state-of-the-art deep multi-view clustering methods. These baselines are categorized into two groups: classical deep multi-view clustering methods (CoMVC (Trosten et al., 2021), SiMVC (Trosten et al., 2021), MFLVC (Xu et al., 2022b), DealMVC (Yang et al., 2023b), SURE (Yang et al., 2023a), CANDY (Guo et al.), DIVIDE (Lu et al., 2024), TGM-MVC (Wang et al., 2024a), and SCE-MVC (Wang et al., 2024b)) and noise-resilient deep multi-view clustering methods (RMCNC (Sun et al., 2024) and MVCAN (Xu et al., 2024)). Detailed information on these baselines is provided in Section. A.5 of the Appendix.
Implement Details: In this study, all experiments are con-ducted on the PyTorch (Imambi et al., 2021) platform using an NVIDIA A6000 GPU. For fair comparison, we reproduce the results of all baselines using the original source codes and configurations provided by their authors. Our proposed AIRMVC is trained using the Adam optimizer (Kingma & Ba, 2014) with its default settings. To enhance the discriminative power of the network and obtain reliable soft predictions in an unsupervised setting, we pre-train the model with an auto-encoder module for 100 epochs. The trade-off hyperparameters α and β are consistently set to 1.0. The maximal training epoch is set to 400 for all datasets. A detailed overview of the hyperparameter settings is provided in Tab. 7 in the Appendix.
this section cite: ['b29', 'b29', 'b24', 'b27', 'b41', 'b15', 'b16']

Section: Comparison Experiments (RQ1)
In this subsection, we conduct comprehensive experiments to demonstrate the superior and effectiveness of our designed AIRMVC. To be specific, we compare with 11 stateof-the-art baselines with different noise ratio. The experimental results are presented in Tab. 2, 3, 6, 8, 9. We highlight the optimal results in bold red, and the sub-optimal results with blue. Due to the space limited, we present Tab. 6, 8, 9 in Section. A.3 of Appendix. From the results, we could observe the following observations:
1) AIRMVC significantly outperforms state-of-the-art baselines across most metrics and datasets. To be specific, compared to the best multi-view clustering algorithm, AIRMVC achieves remarkable improvements by 2.39%, 9.64%, and 1.46% w.r.t. ACC, NMI and PUR in BBCSport dataset with 10% noise. We analyze the reason is that AIRMVC could  automatically identify and rectify the noisy data, leading to a robust learning procedure.
2) When noise is present in multi-view data, the performance of deep multi-view clustering models tends to become unstable. The clustering performance of the model exhibits a downward trend as the noise proportion increases. We attribute this decline to the distortion of the underlying clustering structure caused by the presence of noisy data.
3) Compared with the classical deep multi-view clustering methods, e.g., TGM-MVC (Wang et al., 2024a), our AIR-MVC could achieve better performance. These methods lack specifically designed strategies to mitigate the adverse effects of noise, which consequently leads to a decline in performance.
4) Although noise deep multi-view clustering methods could obtain promising performance, e.g., MVCAN (Xu et al., 2024), the impact of noise has not been fully eliminated in complex noisy scenario. The noisy identification and rectification strategies and noise-robust contrastive mechanism in AIRMVC could identify and rectify the noisy data. Thus, achieving better performance. In summary, AIRMVC enables robust training under varying levels of noise. Moreover, the above observations and experimental results further validate the effectiveness and generalization capability of AIRMVC.
this section cite: ['b41']

Section: Ablation Study (RQ2)
In this subsection, we conduct experiments to investigate the effectiveness of the components in AIRMVC. Due to the space limited, we present the experimental results with 30% noise ratio in Section. A.3.2 of Appendix.
Effectiveness of design modules: Here, we implement ablation studies to verify the designed modules, including the noisy identification and rectification strategy and noiserobust contrastive mechanism. In this subsection, we denote "(w/o) D&R", "(w/o) Con", and "(w/o) D&R&Con" represent the model to remove noisy identification and rectification strategy, noise-robust contrastive mechanism, and both modules combined, respectively. Besides, we leverage an autoencoder network as the backbone for "(w/o) D&R&Con" to derive representations for the downstream clustering task.
We conduct experiments with three metrics on six datasets.
The experimental results are demonstrated in Fig. 4. From those results, we could conclude the following observations:
1) When any component of our designed model is removed, the performance of the model degrades. This indicates that BBCSport Caltech101 STL10 Reuters 20 Epoch 80 Epoch 140 Epoch 200 Epoch Figure 5. Visualization of the representations during the training process on UCI-digit dataset.
each part of our design contributes to the overall clustering performance.
2) The removal of noisy identification and rectification strategy ("(w/o) D&R") results in a more substantial performance degradation compared to the removal of noise-robust contrastive mechanism ("(w/o) Con"), highlighting the relative importance of noisy identification and rectification strategy to the model's performance.
this section cite: []

Section: Effectiveness of rectification strategy:
In AIRMVC, we rectify the noisy data with two steps. Specifically, we first employ two-component GMM to identify the noisy data. Then, we rectify the noisy data based on identification result. To illustrate the adverse effects of noise, we present experimental evidence in Tab. 4 using the WebKB dataset, demonstrating a significant performance degradation in noisy scenarios. This underscores the necessity of designing a strategy specifically targeted at noise identification and rectification.To further validate the effectiveness of our noise identification and rectification strategy, we conduct experiments on the BBCSport dataset with a 10% noise ratio. For a comprehensive evaluation, we define four different experimental setups. For simplicity, we use "Noisy Data" to represent the direct fusion of noisy multiview features, "Single View" to represent the results from a single noisy view, "Directly Rectify" to represent directly using cross-entropy with the first view to rectify, and "Ours" to represent our proposed strategy. The experimental results are presented in Fig. 3. From these results, we can derive the following observations:
1) Directly fusing noisy multi-view data results in the worst clustering performance. We attribute this phenomenon to the lack of any noisy rectification mechanism, allowing the noisy views to mutually amplify their adverse effects. Furthermore, the fusion process exacerbates the impact of noise. Without rectification, the clustering performance of multi-view data is even worse than that of a single view.
2) Compared to the directly rectification with the first view, our proposed AIRMVC could achieve better performance. This phenomenon can be attributed to the tendency of directly using the correction results from the first view to homogenize features across different views. In contrast, our noisy identification and rectification strategy effectively filters out noisy data from each view while retaining complementary information that is beneficial for downstream tasks.
this section cite: []

Section: Visualization Analysis (RQ3)
To visualize the latent representation space and assess the effectiveness of AIRMVC, we employ t-SNE (Van der Maaten & Hinton, 2008) as the visualization tool in our experiments. Specifically, we extract representations from the UCI-digit dataset. The results, as shown in Fig. 5, reveal the following observations. As training progresses, the cluster structures within the UCI-digit dataset become increasingly distinct. By the 200th epoch, well-defined and distinguishable cluster structures have emerged. This observation demonstrates the capability of AIRMVC to effectively uncover the latent cluster structures within feature representations.
this section cite: ['b32']

Section: Hyper-parameter Analysis (RQ4)
In this subsection, we conduct experiments to analysis the sensitivity of the hyper-parameters in this paper.
Sensitive analysis of trade-off hyper-parameter α and β: BBCSport-ACC BBCSport-NMI WebKB-ACC WebKB-NMI Reuters-NMI Caltech101-NMI STL10-PUR UCI-digit-NMI Figure 7. Sensitivity Analysis for α and β with ACC, NMI, and PUR on BBCSport, WebKB, Reuters, UCI-digits, Caltech101 and STL10 datasets with 10% noisy ratio.
To further examine the impact of the parameters α and β on our model, we performed sensitive experiments on six datasets with 10% noise ratio. Specifically, we analyze parameter values within the range of {0.01, 0.1, 1.0, 10}. According to the results presented in Fig. 7. We could find the following observations. 1) When α and β approach extreme values, the model's clustering performance deteriorates significantly. This decline can be attributed to the disruption of the balance in the loss function. We observe that the model achieves optimal clustering performance when the values of α and β are set to 1.0. Thus, we set the parameters to 1.0.
2) Compared to changes in β, variations in α result in more pronounced changes in the model's performance. This indicates that the noisy identification and rectification module contributes more significantly to enhancing performance.
this section cite: []

Section: Sensitive analysis of threshold τ :
We conduct experiments to evaluate the influence of the threshold parameter τ . We varied the value of τ within the range of {0.2, 0.4, 0.6, 0.8}. The results are demonstrated in Fig. 6. We have the following observation. As the threshold increases, the clustering performance of the model improves progressively. We attribute this to the fact that higher thresholds reduce the number of incorrect contrastive learning sample pairs.
this section cite: []

Section: Conclusion
In this work, we present a novel deep contrastive multi-view clustering network to automatically identify and rectify the noisy data (AIRMVC). In AIRMVC, we first consider the noisy identification as an anomaly identification problem. Then, based on the identification results, we introduce a hybrid rectification strategy to alleviate the adverse impact of noisy data. After that, we design a noise-robust contrastive mechanism to obtain more discriminative representations. Moreover, we theoretical proof that the learned representations could discard the noisy information. Comprehensive experiments conducted on six benchmark datasets validate the effectiveness and robustness of AIRMVC.
this section cite: []

Section: References
Ref_id:b0 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b1 Title: Elements of information theory Year: (1999)
Ref_id:b2 Title: Efficient and adaptive recommendation unlearning: A guided filtering framework to erase outdated preferences Year: (2025)
Ref_id:b3 Title: Data augmentation as free lunch: Exploring the test-time augmentation for sequential recommendation Year: (2025)
Ref_id:b4 Title: Iterative deep structural graph contrast clustering for multiview raw data Year: (2023)
Ref_id:b5 Title: Cross-view topology based consistent and complementary information for deep multi-view clustering Year: (2023-10)
Ref_id:b6 Title: Robust contrastive multi-view clustering against dual noisy correspondence Year: ()
Ref_id:b7 Title: Contrastive multi-view representation learning on graphs Year: (2020)
Ref_id:b8 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b9 Title: Training products of experts by minimizing contrastive divergence Year: (2002)
Ref_id:b10 Title: Learning deep representations by mutual information estimation and maximization Year: (2018)
Ref_id:b11 Title: Exploring the role of node diversity in directed graph representation learning Year: (2024)
Ref_id:b12 Title: On which nodes does gcn fail? enhancing gcn from the node perspective Year: (2024)
Ref_id:b13 Title: Partially view-aligned clustering Year: (2020)
Ref_id:b14 Title: Estimation of non-normalized statistical models by score matching Year: (2005)
Ref_id:b15 Title: Programming with TensorFlow: solution for edge computing applications Year: (2021)
Ref_id:b16 Title: A method for stochastic optimization Year: (2014)
Ref_id:b17 Title: Cross-view graph matching guided anchor alignment for incomplete multi-view clustering Year: (2023)
Ref_id:b18 Title: Incomplete multi-view clustering with paired and balanced dynamic anchor learning Year: (2025)
Ref_id:b19 Title: Consensus graph learning for multi-view clustering Year: (2021)
Ref_id:b20 Title: Efficient one-pass multi-view subspace clustering with consensus anchors Year: (2022)
Ref_id:b21 Title: One pass late fusion multiview clustering Year: (2021)
Ref_id:b22 Title: Deep graph clustering via dual correlation reduction Year: (2022)
Ref_id:b23 Title: Simple contrastive graph clustering Year: (2023)
Ref_id:b24 Title: Decoupled contrastive multi-view clustering with high-order random walks Year: (2024)
Ref_id:b25 Title: Revisiting selfsupervised heterogeneous graph learning from spectral clustering perspective Year: (2024)
Ref_id:b26 Title: Hg-adapter: Improving pre-trained heterogeneous graph neural networks with dual adapters Year: (2025)
Ref_id:b27 Title: Robust multi-view clustering with noisy correspondence Year: (2024)
Ref_id:b28 Title: Contrastive multiview coding Year: (2020)
Ref_id:b29 Title: Reconsidering representation alignment for multi-view clustering Year: (2021)
Ref_id:b30 Title: Self-supervised learning from a multi-view perspective Year: (2020)
Ref_id:b31 Title: Representation learning with contrastive predictive coding Year: (2018)
Ref_id:b32 Title: Visualizing data using t-sne Year: (2008)
Ref_id:b33 Title: Continual multi-view clustering Year: (2022)
Ref_id:b34 Title: Fast continual multi-view clustering with incomplete views Year: (2024)
Ref_id:b35 Title: View gap matters: Cross-view topology and information decoupling for multi-view clustering Year: (2024)
Ref_id:b36 Title: Evaluate then cooperate: Shapley-based view cooperation enhancement for multi-view clustering Year: (2024)
Ref_id:b37 Title: Generative partial multi-view clustering with adaptive fusion and cycle consistency Year: (2021)
Ref_id:b38 Title: Incomplete multiview clustering via graph regularized matrix factorization Year: (2018)
Ref_id:b39 Title: Self-supervised discriminative feature learning for deep multi-view clustering Year: (2022)
Ref_id:b40 Title: Multi-level feature learning for contrastive multi-view clustering Year: (2022)
Ref_id:b41 Title: Investigating and mitigating the side effects of noisy views for self-supervised clustering algorithms in practical multi-view scenarios Year: (2024)
Ref_id:b42 Title: Partially view-aligned representation learning with noiserobust contrastive loss Year: (2021)
Ref_id:b43 Title: Robust multi-view clustering with incomplete information Year: (2023)
Ref_id:b44 Title: Interpolation-based contrastive learning for few-label semi-supervised learning Year: (2022)
Ref_id:b45 Title: Dealmvc: Dual contrastive calibration for multi-view clustering Year: (2023)
Ref_id:b46 Title: Cluster-guided contrastive graph clustering network Year: (2023)
Ref_id:b47 Title: Convert: Contrastive graph clustering with reliable augmentation Year: (2023)
Ref_id:b48 Title: Hyperbolic contrastive learning for cross-domain recommendation Year: (2024)
Ref_id:b49 Title: Graphlearner: Graph node clustering with fully learnable augmentation Year: (2024)
Ref_id:b50 Title: Mixed graph contrastive network for semi-supervised node classification Year: (2024)
Ref_id:b51 Title: Darec: A disentangled alignment framework for large language model and recommender system Year: (2025)
Ref_id:b52 Title: Dual test-time training for out-ofdistribution recommender system Year: (2025)
Ref_id:b53 Title: Apgl4sr: A generic framework with adaptive and personalized global collaborative information in sequential recommendation Year: (2023)
Ref_id:b54 Title: Dataset regeneration for sequential recommendation Year: (2024)
Ref_id:b55 Title: Gzoo: Black-box node injection attack on graph neural networks via zeroth-order optimization Year: (2024)
Ref_id:b56 Title: Defending against backdoor attacks on graph neural networks via discrepancy learning. Network and Distributed System Security Symposium Year: (2025)
Ref_id:b57 Title: Sparse low-rank multi-view subspace clustering with consensus anchors and unified bipartite graph Year: (2023)
Ref_id:b58 Title: How to construct corresponding anchors for incomplete multiview clustering Year: (2023)
Ref_id:b59 Title: Towards resource-friendly, extensible and stable incomplete multi-view clustering Year: (2024)
Ref_id:b60 Title: Crossdomain recommendation via user interest alignment Year: (2023)
Ref_id:b61 Title: Cross-domain recommendation via progressive structural alignment Year: (2023)
Ref_id:b62 Title: Asymmetric double-winged multi-view clustering network for exploring diverse and consistent information Year: (2024)
Ref_id:b63 Title: Multiple kernel clustering with neighbor-kernel subspace segmentation Year: (2019)
