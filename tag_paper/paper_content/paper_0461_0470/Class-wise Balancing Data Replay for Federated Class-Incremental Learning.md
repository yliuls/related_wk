Title: Class-wise Balancing Data Replay for Federated Class-Incremental Learning
Abstract: Federated Class Incremental Learning (FCIL) aims to collaboratively process continuously increasing incoming tasks across multiple clients. Among various approaches, data replay has become a promising solution, which can alleviate forgetting by reintroducing representative samples from previous tasks. However, their performance is typically limited by class imbalance, both within the replay buffer due to limited global awareness and between replayed and newly arrived classes. To address this issue, we propose a class-wise balancing data replay method for FCIL (FedCBDR), which employs a global coordination mechanism for class-level memory construction and reweights the learning objective to alleviate the aforementioned imbalances. Specifically, FedCBDR has two key components: 1) the global-perspective data replay module reconstructs global representations of prior task in a privacy-preserving manner, which then guides a class-aware and importance-sensitive sampling strategy to achieve balanced replay; 2) Subsequently, to handle class imbalance across tasks, the task-aware temperature scaling module adaptively adjusts the temperature of logits at both class and instance levels based on task dynamics, which reduces the model's overconfidence in majority classes while enhancing its sensitivity to minority classes. Experimental results verified that FedCBDR achieves balanced class-wise sampling under heterogeneous data distributions and improves generalization under task imbalance between earlier and recent tasks, yielding a 2%-15% Top-1 accuracy improvement over six stateof-the-art methods.

Section: Introduction
Federated learning (FL) is a distributed machine learning paradigm that enables collaborative training of a shared global model across multiple data sources [1,2,3,4,5]. It periodically performs parameter-level interaction between clients and the server instead of gathering clients' data, which can enhance data privacy while leveraging the diversity of distributed data sources to build a more generalized global model [6,7,8,9,10,11]. This mechanism makes it widely applicable to various fields [12,13,14,15,16]. Building upon this foundation, Federated Class-Incremental Learning (FCIL) extends FL by introducing dynamic data streams where clients sequentially encounter different task classes under non-independent and identically distributed data [17,18,19,20,21]. However, this amplifies the inherent complexities of FL, as the global model must integrate heterogeneous and Figure 1: Motivation of the FedCBDR. Traditional data replay strategies typically focus on local information and, due to the lack of global awareness, often result in imbalanced class distributions during replay. FedCBDR aims to explore global information in a privacy-preserving manner and leverage it for sampling, which can alleviate the class imbalance problem. evolving knowledge from clients while mitigating catastrophic forgetting, despite having no or only limited access to historical data [22,23,24].
To address the challenge of catastrophic forgetting in FCIL, data replay has emerged as a promising strategy for retaining knowledge from previous tasks. Existing replay-based methods can be broadly categorized into two types: generative-based replay and exemplar-based replay. The former leverages generative models to synthesize representative samples from historical tasks [23,25,26]. Its core idea is to learn the data distribution of previous tasks and internalize knowledge in the form of model parameters, enabling the indirect reconstruction of prior knowledge through sample generation when needed [19,27]. However, they often overlook the computational cost of training generative models and are inherently constrained by the quality and fidelity of the synthesized data [23,28,29]. In contrast, exemplar-based replay methods directly store real samples from previous tasks, avoiding the complexity of generative processes while leveraging high-quality raw data to ensure robust retention of prior knowledge [17,28,30,31]. These methods rely on a limited set of historical samples to maintain the decision boundaries of previously learned task classes. However, due to the lack of a global perspective on data distribution across clients, these methods are prone to class-level imbalance in replayed samples, which undermines the model's ability to retain prior knowledge [30,31].
To address these issues, this paper proposes a class-wise balancing data replay method for FCIL, termed FedCBDR, which incorporates the global signal to regulate class-balanced memory construction, aiming to achieve distribution-aware replay and mitigate the challenges posed by non-IID client data, as illustrated in Figure 1. Specifically, FedCBDR comprises two primary modules: 1) the global-perspective data replay (GDR) module reconstructs a privacy-preserving pseudo global representation of historical tasks by leveraging feature space decomposition, which enables effective cross-client knowledge integration while preserving essential attributes information. Furthermore, it introduces a principled importance-driven selection mechanism that enables class-balanced replay, guided by a globally-informed understanding of data distribution; 2) the task-aware temperature scaling (TTS) module introduces a multi-level dynamic confidence calibration strategy that combines task-level temperature adjustment with instance-level weighting. By modulating the sharpness of the softmax distribution, it balances the predictive confidence between majority and minority classes, enhancing the model's robustness to class imbalance between historical and current task samples.
Extensive experiments were conducted on three datasets with different levels of heterogeneity, including performance comparisons, ablation studies, in-depth analysis, and case studies. The results demonstrate that FedCBDR effectively balances the number of replayed samples across classes and alleviates the long-tail problem. Compared to six state-of-the-art existing methods, FedCBDR achieves a 2%-15% Top-1 accuracy improvement.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b22', 'b24', 'b25', 'b18', 'b26', 'b22', 'b27', 'b28', 'b16', 'b27', 'b29', 'b30', 'b29', 'b30']

Section: Related Work

this section cite: []

Section: Exemplar-based Replay Methods
In FCIL, exemplar-based replay methods aim to mitigate catastrophic forgetting by storing and replaying a subset of samples from previous tasks. They typically maintain a small exemplar buffer on each client, which is used during training alongside new task data to preserve knowledge of previously learned classes [17,30,31,32,33,34]. For example, GLFC alleviates forgetting in FCIL by leveraging local exemplar buffers for rehearsal, while introducing class-aware gradient compensation and prototype-guided global coordination to jointly address local and global forgetting [17]. Moreover, Re-Fed introduces a Personalized Informative Model to strategically identify and replay task-relevant local samples, enhancing the efficiency of buffer usage and further reducing forgetting in heterogeneous client environments [30]. However, the lack of global insight in local sample selection often results in class imbalance, while the long-tailed distribution between replayed and current data is frequently overlooked, degrading the effectiveness of data replay [30,31].
this section cite: ['b16', 'b29', 'b30', 'b31', 'b32', 'b33', 'b16', 'b29', 'b29', 'b30']

Section: Generative-based Replay Methods
Generative replay methods aim to reconstruct the samples of past tasks through techniques such as generative modeling [19,23,27,35,36], which enables the model to revisit historical knowledge to mitigate catastrophic forgetting. Following this line of thought, TARGET generates pseudo features through a globally pre-trained encoder and performs knowledge distillation by aligning the current model's predictions with those of a frozen global model [27]; LANDER utilizes pre-trained semantic text embeddings as anchors to synthesize meaningful pseudo samples, and distills knowledge by aligning the model's predictions with class prototypes derived from textual descriptions [19]. However, these methods are typically limited by the high computational cost of training generative models and the suboptimal performance caused by low-fidelity pseudo samples [19,27].
this section cite: ['b18', 'b22', 'b26', 'b34', 'b35', 'b26', 'b18', 'b18', 'b26']

Section: Knowledge Distillation-based Methods
Knowledge distillation-based methods generally follow two paradigms. The first focus om aligning the output predictions of the current model with those of previous models, which aims to preserve task-specific decision boundaries [37,38,39,40,41,42,43,44,45]. The second estimates the importance of model parameters for previously learned tasks and performs regularization to prevent forgetting [46,47]. Both approaches avoid storing raw data but are prone to knowledge degradation over time, especially as the number of tasks increases [37,46].
this section cite: ['b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b36', 'b45']

Section: Preliminaries
We consider a federated class-incremental learning (FCIL) setting, where a central server aims to collaboratively train a global model with the assistance of K distributed clients. Each client k receives a sequence of classification tasks {D (1)
k , D (2) k , . . . , D (t)
k }, where each task introduces a disjoint set of new classes. Upon the arrival of task t, the global model parameters θ t are optimized to minimize the average loss over the union of all samples seen so far, i.e.,
D t = t s=1 K k=1 D (s) k , by solving min θ 1 |D t | t s=1 K k=1 N (s) k i=1 L(f k (x (s) k,i ; θ), y (s) k,i ).
In replay-based methods, each client maintains a memory buffer with a fixed budget of M samples. When task t arrives, the client selects up to N representative samples from each of the previous tasks {1, . . . , t -1}, subject to the total memory constraint. The resulting memory set is denoted by B
(t-1) k = t-1 s=1 {(x (s) k,i , y (s) k,i )} N
i=1 , where N is the number of samples stored per task and
B (t-1) k satisfies |B (t-1) k | ≤ M . The local training set on client k then becomes D (t) k,train = D (t) k ∪ B (t-1) k
, combining current and replayed samples. Based on these local datasets, the server updates the global model by minimizing the aggregated loss: min θ K k=1 (x,y)∈D (t) k,train L(f k (x; θ), y).
this section cite: []

Section: Class-wise Balancing Data Replay for Federated Class-Incremental Learning
This section presents an effective active data selection method for FCIL, which aims to explore global data distribution to balance class-wise sampling. Moreover, it leverages temperature scaling to adjust the logits, which can alleviate the imbalance between samples from previously learned and newly introduced tasks. Figure 2 and Algorithm 1 illustrates the framework of the proposed FedCBDR.
this section cite: []

Section: Global-perspective Data Replay (GDR)
Due to privacy constraints, traditional data replay strategies typically rely on local data distributions. However, the absence of global information often leads to class imbalance in the replay buffer. To address this, the GDR module aggregates local informative features into a global pseudo feature set, enabling exploration of the global distribution without exposing raw data.
Inspired by Singular Value Decomposition (SVD) [48,49], we first generate a set of random orthogonal matrices: a client-specific matrix
P (i) k ∈ R |D (i) k |×|D (i) k |
for each client k and task i, and a globally shared matrix Q (i) ∈ R d×d , where d denotes the dimension of the feature. Each client encrypts its local feature matrix X (i)
k = M g (D (i) k ) via Inverse Singular Value Decomposition (ISVD): X (i) ′ k = P (i) k X (i) k Q (i) ,(1)
and uploads the encrypted matrix
X (i) ′ k
to the server, where M g (•) is the feature extractor of the global model. The server then aggregates all encrypted matrices into a global matrix X (i) ′ :
X (i) ′ = concat{X (i) ′ k | k = 1, . . . , K},(2)
and performs SVD as follows:
X (i) ′ = U (i) ′ Σ (i) ′ V (i) ′ ⊤ ,(3)
where
U (i) ′ ∈ R n×n , Σ (i) ′ ∈ R n×d
, and V (i) ′ ∈ R d×d , with n denoting the total number of samples from all clients. Next, the server extracts a submatrix of the left singular vectors for each client k by:
U (i) k = I k (U (i) ′ ) ∈ R |D (i) k |×n ,(4)
where I k (•) denotes a row selection function that returns the indices corresponding to client k's samples. To quantify the importance of local samples within the global latent space, client k computes a leverage score [50,51,52] for j-th sample of task i as:
τ i,j k = ∥e ⊤ i,j U (i) k ∥ 2 2 ,(5)
where e i,j denotes the j-th standard basis vector in task i. Notably, a higher leverage score indicates that the sample has a larger projection in the low-dimensional latent space, suggesting that it contributes more significantly to the global structure and is more representative. Moreover, clients send their leverage scores to the server, which aggregates them into a global vector τ i = concat{τ i,j k |k = 1, ..., K; j = 1, ..., n i k } and normalizes it to obtain a sampling distribution:
p i,j k = τ i,j k n i k j ′ =1 τ i,j ′ k .(6)
Algorithm 1 FEDCBDR 1: Initialize: R: number of communication rounds; K: number of clients; t: number of tasks; θ g : global model parameters; B pre k : replay buffer for historical tasks on client k; D s k : local data of task s on client k. 2: for each task s = 1 to t do 3:
for each communication round r = 1 to R do 4:
for each client k = 1 to K do 5: Initialize local model parameters: θ k ← θ g 6: if s == 1 then 7: Sample a mini-batch ζ from D (1)
k , and update θ k using Eq. 8. k ∪ B pre k , and update θ k using Eq. 9.
11:
Compute pseudo-features based on Eq. 1, and upload them to the server.
12: end if 13: end for 14: if r < R then 15: Aggregate local model parameters across clients. 16: else 17:
Aggregate model parameters and pseudo-features from all clients using Eq. 2.
18:
Perform Global Sampling based on Eqs. 3-6, and send the selected sample IDs back to the corresponding clients.
19:
end if 20:
end for 21: end for 22: // Global Sampling Procedure 23: Form the global feature pool X (i) by aggregating all pseudo-features via Eq. ( 2). 24: Perform singular value decomposition (SVD) using Eq. 3 to extract key attributes. 25: Compute leverage scores for each client's samples using Eqs. 4-5, and normalize globally using Eq. 6. 26: Perform sampling and adjust the probabilities of the selected samples accordingly.
Subsequently, we perform i.i.d. sampling based on the distribution p = {p i,j k |k = 1, ..., K; j = 1, ..., n i k }. Once a sample x is selected, its sampling weight is adjusted to 1 √ ns•px e x , where n s denotes the number of selected samples and p x is the original sampling probability of x, e x is the standard basis vector of x. This adjustment ensures unbiased estimation during aggregation. Following the sampling procedure, the server communicates the selected sample indices to their respective clients, where the corresponding data points are subsequently marked for further use.
this section cite: ['b47', 'b48', 'b49', 'b50', 'b51']

Section: Task-aware Temperature Scaling (TTS)
Due to limited replay budgets, samples from previous tasks are often much fewer than those from the current task, leading to class imbalance and poor retention of past knowledge. To mitigate this, the TTS module dynamically adjusts sample temperature and weight based on task order, enhancing the contribution of tail-class samples during optimization.
Specifically, we use a lower temperature to sharpen logits for samples from earlier tasks. Furthermore, to further amplify the optimization effect of tail-class samples during training, we also leverage a re-weighted cross-entropy loss, i.e.,
L TTS = 1 Nold Nold i=1 ω old • CE y i , Softmax Concat z old i τ old , z new i τ new + 1 N new Nnew j=1 ω new • CE y j , Softmax Concat z old j τ old , z new j τ new (7
)
where N old and N new denote the number of samples from the previous and newly arrived task, respectively; y i and y j are the ground-truth labels; z old i and z new i denote the logits corresponding to old classes and new classes, respectively; τ old and τ new are the temperature scaling factors for previous and newly arrived task samples; ω old and ω new are the corresponding sample weights; CE(•) denotes
this section cite: []

Section: Training Strategy
The training strategy consists of two stages to progressively address the evolving challenges in federated class-incremental learning. Algorithm 1 presents the pipeline of the FedCBDR.
Stage 1: Initial Task Optimization. In the first task, client k learns from local data using the standard cross-entropy loss, i.e.,
min θ k 1 N N i=1 CE(y i , Softmax(f θ k (x i ))),(8)
Stage 2: Class-Incremental Optimization. As new tasks arrive and class imbalance emerges between previous and current tasks in client k, we employ L T T S to mitigate the imbalance, i.e.,
minθ k 1 Nold Nold i=1 ωold • CE yi, Softmax Concat f old θk (xi) τold , f new θk (xi) τnew + 1 Nnew Nnew j=1 ωnew • CE yj, Softmax Concat f old θk (xj) τold , f new θk (xj) τnew(9)
where x i is the input sample, y i is the corresponding ground-truth, f old θ k (x) and f new θ k (x) represent the outputs of the model corresponding to old and new classes, respectively. Softmax(•) converts the logits into a probability distribution.
this section cite: []

Section: Experiments

this section cite: []

Section: Experiment Settings
Datasets. Following existing studies [27,30], we conducted all experiments on three commonly used datasets, including CIFAR10 [53,54], CIFAR100 [53,54] and TinyImageNet [55] to validate the effectiveness of the FedCBDR. We simulate heterogeneous data distributions across clients using the Dirichlet distribution with parameters β = {0.1, 0.5, 1.0}, where smaller values of β correspond to higher level of data heterogeneity. The statistical details are presented in the Table 1.
Evaluation Metric. Following prior studies [19,56,57,58], we adopt Top-1 Accuracy as the evaluation metric, defined as Accuracy = N correct /N total , where N correct and N total denote the number of correct predictions and the total number of samples, respectively.
this section cite: ['b26', 'b29', 'b52', 'b53', 'b52', 'b53', 'b54', 'b18', 'b55', 'b56', 'b57']

Section: Implementation Details.
In the experiments, the number of clients is fixed at K = 5, with each client running local epochs E = 2 per round, using a batch size B = 128. For all datasets, we adopt ResNet-18 as the backbone, with the classifier's output dimension dynamically updated as tasks progress and conduct T = 100 communication rounds per task. The SGD optimizer is employed with a learning rate of 0.01 and a weight decay of 1 × 10 -5 . The number of stored samples per task varies by dataset and split setting: for CIFAR10, 450 samples are stored under 3-task splits and 300 under 5-task splits; for CIFAR100, 1,000 samples are used for 5-task splits and 500 for 10-task splits; for TinyImageNet, 2,000 samples are stored for 10-task splits and 1,000 for 20-task splits. For the temperature and weighted parameters, we select τ old ∈ {0.8, 0.9} and w old ∈ {1.1, 1.2, 1.3, 1.4} for previous tasks, while τ new ∈ {1.1, 1.2} and w new ∈ {0.7, 0.8, 0.9} are used for newly arrived tasks. Moreover, the hyperparameters of baselines are tuned based on their original papers for fair comparison. And training on each client is performed using an NVIDIA RTX 3090 GPU (24 GB).
Table 2: Performance comparison between FedCBDR and baselines across three datasets under varying levels of heterogeneity (β). CIFAR10 is divided into 3 tasks, CIFAR100 into 5 tasks, and TinyImageNet into 10 tasks. All methods were executed under three different random seeds, and both the mean and standard deviation of the results are reported. The best results are bolded.
this section cite: []

Section: Performance Comparison
To evaluate the effectiveness of the proposed FedCBDR, we compare it with six representative baseline methods: Finetune [19], FedEWC [46], FedLwF [37], TARGET [27], LANDER [19], and Re-Fed [30]. As reported in Table 2 and Table 3, the results can be summarized as follows:
• FedCBDR achieves the highest Top-1 accuracy in most cases across the three datasets under varying levels of heterogeneity and task splits. The only suboptimal result occurs on CIFAR100 with 5 tasks and β = 1.0, where FedCBDR (52.06%) performs slightly worse than LANDER (52.77%). This demonstrates the adaptability and robustness of the proposed FedCBDR across complex settings.
• Despite LANDER attains the best performance on CIFAR100 under the 5-task and β = 1.0 setting, it demands the generation of more than 10,000 samples per task, and the overhead of training its data generator surpasses that of the federated model, raising concerns about its scalability.
• Knowledge distillation-based methods like FedLwF perform well on simpler tasks (CIFAR10) by using pretrained knowledge to guide local models. However, their performance drops on more complex or heterogeneous tasks due to limited adaptability to local variations.
• Given an equal memory budget, class-balanced sampling (FedCBDR) consistently achieves superior performance compared to class-imbalanced strategy (Re-Fed), as it ensures more equitable representation across categories and effectively mitigates class-level forgetting in FCIL scenarios.
this section cite: ['b18', 'b45', 'b36', 'b26', 'b18', 'b29']

Section: Ablation Study
In this section, we conducted an ablation study to investigate the contributions of key modules, including the Global-perspective Active Data Replay (GDR) module and the Task-aware Temperature Scaling (TTS) module. Table 4 presents the results, which can be summarized as follows:
Table 4: Ablation results under different levels of data heterogeneity and task splitting settings. "3/5/10" denotes CIFAR10 with 3 tasks, CIFAR100 with 5 tasks, and TinyImageNet with 10 tasks; "5/10/20" represents 5, 10, and 20 tasks respectively.
Task Splitting Method CIFAR10 CIFAR100 TinyImageNet β=0.5 β=1.0 β=0.1 β=0.5 β=1.0 β=0.1 β=0.5 β=1.0 3/5/10 Finetune 38.71 ±3.7 40.49 ±3.0 15.17 ±2.2 16.75 ±2.6 17.15 ±1.3 6.06 ±0.9 6.00 ±0.8 6.40 ±0.5 +GDR 62.13 ±2.1 63.81 ±1.9 45.28 ±1.5 47.66 ±0.9 51.47 ±1.7 17.24 ±0.6 17.89 ±0.5 18.04 ±0.4 +TTS 41.34 ±2.3 42.55 ±2.2 17.32 ±0.5 17.14 ±0.4 19.32 ±0.5 6.67 ±0.2 6.92 ±0.3 7.27 ±0.4 +GDR+TTS 64.11 ±1.2 65.20 ±1.9 46.40 ±1.6 49.76 ±2.7 52.06 ±1.5 18.37 ±1.1 18.86 ±0.9 18.78 ±0.9 5/10/20 Finetune 19.78 ±2.3 23.34 ±2.8 7.22 ±1.1 9.39 ±0.7 9.64 ±0.5 3.40 ±0.4 3.73 ±0.5 3.95 ±0.3 +GDR 59.34 ±3.1 63.20 ±2.6 44.04 ±1.3 46.33 ±0.5 46.50 ±0.8 11.44 ±0.3 13.85 ±0.5 14.51 ±0.6 +TTS 22.43 ±2.4 25.81 ±2.1 8.31 ±0.2 10.21 ±0.3 10.33 ±0.4 3.78 ±0.5 4.04 ±0.4 4.16 ±0.3 +GDR+TTS 61.18 ±1.3 65.42 ±1.8 45.11 ±1.2 46.51 ±1.6 47.79 ±1.4 12.58 ±0.4 14.47 ±0.7 15.69 ±0.6
• Incorporating the GDR module substantially improves performance across all cases, particularly under high data heterogeneity (β = 0.1), demonstrating its effectiveness in alleviating catastrophic forgetting even with a limited number of replay samples in federated class-incremental learning.
• Using the TTS module alone leads to consistent improvements over Finetune, highlighting its effectiveness in addressing intra-client class imbalance through temperature scaling. This contribution to better generalization is particularly evident under the more challenging "5/10/20" task splitting scenario.
• The integration of both modules results in the best overall performance, consistently achieving the highest Top-1 accuracy across various datasets and heterogeneity levels. This stems from their complementary strengths: the GDR module mitigates inter-task forgetting, while the TTS module alleviates both intra-and inter-client class imbalance. This section investigates the performance of FedCBDR and the baselines in incremental cases on three datasets. Figure 3 presents the average accuracy of all methods on both current and previous tasks. Notably, FedCBDR consistently outperforms other baseline methods across all task splits, with its accuracy curves remaining higher throughout the incremental process. Furthermore, FedCBDR exhibits a slower performance degradation as the number of tasks increases, indicating stronger resistance to catastrophic forgetting. In addition, it maintains significantly higher accuracy on later tasks, especially in challenging settings such as CIFAR100 and TinyImageNet with 10 tasks, highlighting its ability to balance knowledge retention and adaptation to new classes.
this section cite: []

Section: Performance Evaluation of FedCBDR under Incremental Tasks

this section cite: []

Section: Quantitative Analysis of Replay Buffer Size on Test Accuracy
In this section, we evaluate the performance of Re-Fed and FedCBDR under different buffer size M settings, and additionally include LANDER, which generates 10,240 synthetic samples for each task. As shown in Table 5, FedCBDR exhibits more significant performance advantages over Re-Fed under limited memory settings, and even surpasses LANDER, which relies on a large-scale generative replay buffer. Furthermore, as the buffer size increases, FedCBDR demonstrates more stable and significant performance improvements. This indicates that the method can effectively leverage larger replay buffers for continuous optimization. However, Re-Fed exhibits noticeable performance fluctuations under small and medium buffer settings. In particular, its accuracy is significantly lower than that of FedCBDR on CIFAR100 with M = 500 and TinyImageNet with M = 2000, indicating its limited ability to mitigate inter-class interference and retain knowledge from previous tasks. These findings validate that, under the same buffer budget, a balanced sampling distribution is more effective than an imbalanced one in alleviating forgetting and improving overall model performance. Figure 4 gives a sensitivity analysis of FedCBDR with respect to temperature and sample weighting hyperparameters. Overall, temperature scaling and sample re-weighting help mitigate class imbalance, but model performance varies considerably with different hyperparameter settings. The model achieves better overall performance when ω old = 1.1, ω new = 0.9, τ old = 0.9, and τ new = 1.1. This is because slightly higher weight and temperature for previous-task samples help retain old knowledge, while lower weight and higher temperature for newly arrived samples reduce overfitting and improve adaptation. However, inappropriate hyperparameter choices may harm performance. For instance, a large τ new (e.g., 2.0) leads to overly smooth predictions, reducing discrimination among newly arrived classes. These results emphasize the need for proper tuning to ensure balanced learning.  This section presents case studies comparing prediction confidence and attention focus using Grad-CAM [59,60,61,62,63] visualizations. As shown in Figure 6(a-c), in the absence of data replay, the model struggles to correctly classify samples from previous tasks and fails to attend to the relevant target regions. The incorporation of data replay in FedCBDR alleviates this issue by correcting predictions and guiding attention back to semantically important areas. Despite partially mitigating forgetting, data replay alone may still lead to misclassification or low-confidence predictions for tail classes with limited samples. The integration of temperature scaling (T) and sample re-weighting (W) in the TTS module enables the model to better distinguish confusing classes through temperature adjustment, improving tail class accuracy and enhancing prediction stability, as depicted in Figure 6(d-f). These findings demonstrate the crucial role of the collaboration between both modules in mitigating knowledge forgetting during incremental learning.
this section cite: ['b58', 'b59', 'b60', 'b61', 'b62']

Section: Sensitivity Analysis of

this section cite: []

Section: Comparison of Per-Class Sample Distributions in the Replay Buffer

this section cite: []

Section: Conclusions and Future Work
To address the challenge of inter-class imbalance in replay-based federated class-incremental learning, we propose FedCBDR that combines class-balanced sampling with loss adjustment to better exploit the global data distribution and enhance the contribution of tail-class samples to model optimization. Specifically, it uses SVD to decouple and reconstruct local data, aggregates local information in a privacy-preserving manner, and explores i.i.d. sampling within the aggregated distribution. In addition, it applies task-aware temperature scaling and sample re-weighting to mitigate the long-tail problem. Experimental results show that FedCBDR effectively reduces inter-class sampling imbalance and significantly improves final performance.
Despite the impressive performance of FedCBDR, there remain several directions worth exploring to address its limitations. Specifically, we plan to investigate lightweight sampling strategies to reduce feature transmission costs in FedCBDR, and to develop more robust post-sampling balancing methods that mitigate class imbalance with less sensitivity to hyperparameters [64,65,66,67]. Moreover, extending FedCBDR to more complex scenarios [68,69,70,71,72] is a promising direction.
this section cite: ['b63', 'b64', 'b65', 'b66', 'b67', 'b68', 'b69', 'b70', 'b71']

Section: References
Ref_id:b0 Title: FedSSA: Semantic Similarity-based Aggregation for Efficient Model-Heterogeneous Personalized Federated Learning Year: (2024)
Ref_id:b1 Title: Fedmut: Generalized federated learning via stochastic mutation Year: (2024)
Ref_id:b2 Title: Fedcda: Federated learning with cross-rounds divergence-aware aggregation Year: (2024)
Ref_id:b3 Title: Federated multi-view clustering via tensor factorization Year: (2024)
Ref_id:b4 Title: Less is more: Federated graph learning with alleviating topology heterogeneity from a causal perspective Year: ()
Ref_id:b5 Title: Privacy-preserving vertical federated learning with tensor decomposition for data missing features Year: (2025)
Ref_id:b6 Title: Cross-silo prototypical calibration for federated learning with non-iid data Year: (2023)
Ref_id:b7 Title: Learn the global prompt in the low-rank tensor space for heterogeneous federated learning Year: (2025)
Ref_id:b8 Title: Fedcross: Towards accurate federated learning via multi-model cross-aggregation Year: (2024)
Ref_id:b9 Title: Feddse: Distribution-aware sub-model extraction for federated learning over resource-constrained devices Year: (2024)
Ref_id:b10 Title: Fedgh: Heterogeneous federated learning with generalized global header Year: (2023)
Ref_id:b11 Title: Tta-feddg: Leveraging test-time adaptation to address federated domain generalization Year: (2025)
Ref_id:b12 Title: Scalable federated one-step multi-view clustering with tensorized regularization Year: (2025)
Ref_id:b13 Title: Attentive modeling and distillation for out-ofdistribution generalization of federated learning Year: (2024)
Ref_id:b14 Title: Unlearning through knowledge overwriting: Reversible federated unlearning via selective sparse adapter Year: (2025)
Ref_id:b15 Title: Flee: A hierarchical federated learning framework for distributed deep neural network over cloud, edge, and end device Year: (2022)
Ref_id:b16 Title: Federated classincremental learning Year: (2022)
Ref_id:b17 Title: Federated class-incremental learning via weighted aggregation and distillation Year: (2025)
Ref_id:b18 Title: Text-enhanced datafree approach for federated class-incremental learning Year: (2024)
Ref_id:b19 Title: Federated continual learning via knowledge fusion: A survey Year: (2024)
Ref_id:b20 Title: Beyond federated prototype learning: Learnable semantic anchors with hyperspherical contrast for domain-skewed data Year: (2025)
Ref_id:b21 Title: Federated classincremental learning with dynamic feature extractor fusion Year: (2024)
Ref_id:b22 Title: General federated class-incremental learning with lightweight generative replay Year: (2024)
Ref_id:b23 Title: Fedprok: Trustworthy federated class-incremental learning via prototypical feature knowledge transfer Year: (2024)
Ref_id:b24 Title: A datafree approach to mitigate catastrophic forgetting in federated class incremental learning for vision tasks Year: (2023)
Ref_id:b25 Title: Data-free federated class incremental learning with diffusion-based generative memory Year: (2024)
Ref_id:b26 Title: Target: Federated class-continual learning via exemplar-free distillation Year: (2023)
Ref_id:b27 Title: Diffusion-driven data replay: A novel approach to combat forgetting in federated class continual learning Year: (2024)
Ref_id:b28 Title: Federated class incremental learning: A pseudo feature based approach without exemplars Year: (2024)
Ref_id:b29 Title: Towards efficient replay in federated incremental learning Year: (2024)
Ref_id:b30 Title: Re-fed+: A better replay strategy for federated incremental learning Year: (2025)
Ref_id:b31 Title: icarl: Incremental classifier and representation learning Year: (2001)
Ref_id:b32 Title: No one left behind: Real-world federated class-incremental learning Year: (2024)
Ref_id:b33 Title: Sr-fdil: Synergistic replay for federated domain-incremental learning Year: (2024)
Ref_id:b34 Title: Overcoming catastrophic forgetting in federated class-incremental learning via federated global twin generator Year: (2024)
Ref_id:b35 Title: Better generative replay for continual federated learning Year: (2023)
Ref_id:b36 Title: Learning without forgetting Year: (2017)
Ref_id:b37 Title: Fl-clip: Bridging plasticity and stability in pre-trained federated class-incremental learning models Year: (2024)
Ref_id:b38 Title: Self-reinforcing prototype evolution with dual-knowledge cooperation for semi-supervised lifelong person re-identification Year: (2025)
Ref_id:b39 Title: Fedrcil: Federated knowledge distillation for representation based contrastive incremental learning Year: (2023)
Ref_id:b40 Title: Knowledge efficient federated continual learning for industrial edge systems Year: (2025)
Ref_id:b41 Title: Sacfl: Self-adaptive federated continual learning for resource-constrained end devices Year: (2025)
Ref_id:b42 Title: Distribution-aware knowledge prototyping for non-exemplar lifelong person re-identification Year: (2024)
Ref_id:b43 Title: Federated incremental semantic segmentation Year: (2023-06)
Ref_id:b44 Title: Long short-term knowledge decomposition and consolidation for lifelong person re-identification Year: (2025)
Ref_id:b45 Title: Overcoming catastrophic forgetting in neural networks Year: (2017)
Ref_id:b46 Title: Overcoming spatialtemporal catastrophic forgetting for federated class-incremental learning Year: (2024)
Ref_id:b47 Title: Efficient decentralized federated singular vector decomposition Year: (2024)
Ref_id:b48 Title: A numerical des perspective on unfolded linearized admm networks for inverse problems Year: (2022)
Ref_id:b49 Title: Fast approximation of matrix coherence and statistical leverage Year: (2012)
Ref_id:b50 Title: Efficient collaborative crowdsourcing Year: (2016)
Ref_id:b51 Title: Dynamic witness selection for trustworthy distributed cooperative sensing in cognitive radio networks Year: (2011)
Ref_id:b52 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b53 Title: Clustering-based curriculum construction for sample-balanced federated learning Year: (2022)
Ref_id:b54 Title: Tiny imagenet visual recognition challenge Year: (2015)
Ref_id:b55 Title: Flexfl: Heterogeneous federated learning via apoz-guided flexible pruning in uncertain scenarios Year: (2024)
Ref_id:b56 Title: Fednlr: Federated learning with neuron-wise learning rates Year: (2024)
Ref_id:b57 Title: Cross-training with multi-view knowledge fusion for heterogenous federated learning Year: (2024)
Ref_id:b58 Title: Cross-silo feature space alignment for federated learning on clients with imbalanced data Year: (2025)
Ref_id:b59 Title: Improving global generalization and local personalization for federated learning Year: (2025)
Ref_id:b60 Title: Federated deconfounding and debiasing learning for out-of-distribution generalization Year: (2025)
Ref_id:b61 Title: Causal inference over visual-semantic-aligned graph for image classification Year: (2025)
Ref_id:b62 Title: Adversarial cross-modal retrieval Year: (2017)
Ref_id:b63 Title: Class-level structural relation modeling and smoothing for visual representation learning Year: (2023)
Ref_id:b64 Title: A novel density-based outlier detection method using key attributes Year: (2022)
Ref_id:b65 Title: Empowering vision transformers with multi-scale causal intervention for long-tailed image classification Year: (2025)
Ref_id:b66 Title: Cross-modal learning using privileged information for long-tailed image classification Year: (2024)
Ref_id:b67 Title: Yuheng Jia, and Xin Geng. Inaccurate label distribution learning Year: (2024)
Ref_id:b68 Title: Exploiting multi-label correlation in label distribution learning Year: ()
Ref_id:b69 Title: Learning with twin noisy labels for visible-infrared person re-identification Year: (2022)
Ref_id:b70 Title: Learning real facial concepts for independent deepfake detection Year: (2025)
Ref_id:b71 Title: Causal inference for out-of-distribution recognition via sample balancing Year: (2024)
