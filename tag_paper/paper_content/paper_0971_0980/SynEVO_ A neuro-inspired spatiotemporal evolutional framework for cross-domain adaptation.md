Title: SynEVO: A neuro-inspired spatiotemporal evolutional framework for cross-domain adaptation
Abstract: Discovering regularities from spatiotemporal systems can benefit various scientific and social planning. Current spatiotemporal learners usually train an independent model from a specific source data that leads to limited transferability among sources, where even correlated tasks requires new design and training. The key towards increasing cross-domain knowledge is to enable collective intelligence and model evolution. In this paper, inspired by neuroscience theories, we theoretically derive the increased information boundary via learning cross-domain collective intelligence and propose a Synaptic EVOlutional spatiotemporal network, SynEVO, where SynEVO breaks the model independence and enables cross-domain knowledge to be shared and aggregated. Specifically, we first re-order the sample groups to imitate the human curriculum learning, and devise two complementary learners, elastic common container and task-independent extractor to allow model growth and task-wise commonality and personality disentanglement. Then an adaptive dynamic coupler with a new difference metric determines whether the new sample group should be incorporated into common container to achieve model evolution under various domains. Experiments show that SynEVO improves the generalization capacity by at most 42% under cross-domain scenarios and SynEVO provides a paradigm of NeuroAI for knowledge transfer and adaptation. Code available at https://github.com/Rodger-Lau/SynEVO.

Section: Introduction
Spatiotemporal learning aims to predict future urban evolution, which facilitates urban management and socioeconomic planning. Recently, diverse spatiotemporal forecasting solutions have well resolved data sparsity (Zhou et al., 2020), temporal shifts (Zhou et al., 2023), as well as unseen area inferences (Feng et al., 2024). As usual practices, almost all spatiotemporal learners train independent models from specific sources where both models and data are isolated. With the growing of available urban sensors and the diversity of data sources under the rapid urban expansion, the task-specific data-driven learning is inevitably faced with the increased costs of repetitive model designs and computational resources. To this end, an evolvable dataadaptive learner that accommodates cross-domain transferability and adaptivity is highly required to facilitate sustainable urban computing. There have been a number of efforts that try to improve generalization of spatiotemporal learning, where it can be classified as two folds, i.e., countering shifts on the same source domain and across different source domains. On the same source, the initial step is the continuous spatiotemporal learning implemented with experience reply (Chen et al., 2021), and spatiotemporal outof-distribution issue (OOD) is raised by capturing causal invariance for confronting temporal shifts (Zhou et al., 2023). After that, a series of models take environments as indicators to guide generalization (Xia et al., 2024;Wang et al., 2024a;Yuan et al., 2023). However, even for different kinds of sources in a same system, these independent models fail to share common information for task transfer across domains. For generalization across sources, a prompt-empowered universal model for adapting various data sources (Yuan et al., 2024), and a task-level continuous spatiotemporal learner, which actively capture the stable commonality and fine-tune with individual personality for new tasks (Yi et al., 2024) are proposed. Even so, these models still suffer three critical issues for cross-domain transfer and data adaptive model evolution. 1) No theoretical guarantee for collective intelligence facilitating cross-domain transfer. 2) Not all tasks share common patterns, thus uniformly involving all tasks inevitably introduces noise. 3) These models are not elastic to actively evolve when data distribution changes.
Fortunately, with the progress of Neuro-Artificial Intelligence (NeuroAI), neural networks are designed to imitate knowledge acquiring for model generalization and cooperation (Wang et al., 2022;2024b), such as neuro-inspired continuous learning (Wang et al., 2023b)and complementarybased (Kumaran et al., 2016) neural architecture. Considering the similarity between the cross-domain transfer and the way human acquire new skills from prior knowledge, NeuroAI holds great potential to overcome effective crossdomain knowledge transfer. However, given various learning behaviors in human brain, how to couple neuroscience with spatiotemporal network tailored for effective transfer and evolution is still challenging as following aspects, 1) As the collective intelligence cannot be accomplished in an action, how to progressively learn tasks from different domains from easy to difficult, 2) How to imitate human learning process to disentangle commonality and ensure common container elastic with continually receiving information. 3) How to ensure the common-individual models to quickly accessible for few-shot generalization.
Actually, classical neuroscience theories and new advancement reveal that, 1) synapse is an important structure connecting neurons, which takes the role of message sharing and cooperation (Van de Ven et al., 2020;Kennedy, 2016;Benfenati, 2007), 2) neuron and synapse are activated by neurotransmitters and action potentials, which is analogous to gradients in artificial neural networks (ANNs), where larger gradient intensity implies larger inconsistency between solidified knowledge and new information (Gulledge et al., 2005;Hussain & Al Alili, 2017;Zenke et al., 2017). Inspired by observations, we theoretically analyze datadriven knowledge space within neural network can be increased with commonality via information entropy. To this end, we propose a Synaptic EVOlutional spatiotemporal network, SynEVO, to enable easy cross-domain transfer. Our SynEVO couples three neuro-inspired structures. First, to learn the tasks progressively, we devise a curriculumguided sample group re-ordering to monotone increasing learning difficulty via task-level gradient relations. Second, to disentangle cross-domain commonality, we borrow the cerebral neocortex and hippocampus structures in brain, and design complementary dual learners, an Elastic Common Container with growing capacity as the core synaptic function to receive cross-domain information for elastically collecting new regularities, and a Task independent Personality Extractor to characterize individual task features for adaptation. Finally, an adaptive dynamic coupler with a difference metric is devised to identify whether the new sample group should be incorporated into common container for enabling model evolution and increasing collective intelligence, which reduces the pollution of inappropriate data. The contributions are three-fold.
• Inspired by neuroscience, we theoretically analyze the increased generalization capacity with cross-domain intelligence and the facilitated task convergence from progressive learning, and a novel NeuroAI framework is proposed to tackle the cross-domain challenge and empower model evolution.
• By human-machine analogy, we introduce an STsynapse, and couple curriculum and complementary learning with synapse to realize the progressive learning and commonality disentanglement. The model evolution is achieved by elastic common container growing and adaptive sample incorporation.
• Extensive experiments show the collective intelligence increases the model generalization capacity under both source and temporal shifts by at 0.5% to 42%, including few-shot and zero-shot transfer, and empirically validate the convergency of progressive curriculum learning. The extremely reduced memory cost, i.e., only 21.75% memory cost against SOTA on iterative model training and evolution advances urban computing towards sustainable computing paradigm.
this section cite: ['b50', 'b51', 'b9', 'b6', 'b51', 'b39', 'b44', 'b45', 'b42', 'b34', 'b17', 'b31', 'b14', 'b2', 'b10', 'b13', 'b46']

Section: Related Works
Spatiotemporal learning and its generalization capacity.
Spatiotemporal learning has been investigated to enable convenient urban life (Zhang et al., 2017;2020;Ye et al., 2019;Zhou et al., 2020;Wu et al., 2019;Liu et al., 2024;2025a;b;Miao et al., 2024;2025). However, with urban expansion and economic development, it increases the concerns of distribution shifts as all previous learning models assume independent identical distribution. CauSTG devises causal spatiotemporal learning to explicitly address the temporal domain shifts (Zhou et al., 2023). Concurrently, continuous spatiotemporal learning becomes a prevalent topic for model update to counter temporal shifts (Wang et al., 2023a;Chen et al., 2021). Although prosperity, researchers find that data from different sources in a same system tend to share some common patterns. Some pioneering literature exploits unified multi-task spatiotemporal learning to accommodate diverse modalities where a prompt-empowered universal model is proposed (Yuan et al., 2024), while Yi, et,al. investigates the task-level continuous learning to iteratively capture the common and individual patterns (Yi et al., 2024). Even though, these models still ignore three important issues. 1) Fail to unify the source domain and temporal domain for transfer in a same learning architecture where commonality and individuality are well-decoupled.
2) In a same system, how to quantify relations among tasks and filtering the related ones to facilitate commonality is not clear. 3) Previous models fail to evolve with distribution changes to realize efficient cross-domain few-shot generalization. In contrast, we overcome the cross-domain adaptation through lens of NeuroAI by imitating human acquiring new knowledge and skills, to achieve evolutional spatiotemporal network.
this section cite: ['b48', 'b19', 'b41', 'b50', 'b38', 'b20', 'b14', 'b25', 'b51', 'b6', 'b45', 'b42']

Section: Neuro-inspired Artificial Intelligence (NeuroAI).
Neu-roAI is an emerging research topic which designs ANNs from the inspiration of neuroscience and the way how human acquire new knowledge (Luk & Christodoulou, 2024). ANN is originated from the biological neurons and researchers teach machine learners to learn like humans (Lindsay, 2020; Zhang & Zhang, 2018). The literature can be classified as two folds from macroscopic to micro-structures.
In a macro perspective, human brain usually progressively acquires the knowledge from easy to difficult, which inspires curriculum learning in machine intelligence (Wang et al., 2021;Blessing et al., 2024), and complementary learning scheme with hippocampus and neocortex structures (Kumaran et al., 2016;O'Reilly et al., 2014;Arani et al., 2022).
To provide feedback on AI model, reinforcement learning is proposed, and reinforcement learning from human feedback (RLHF) is incorporated into LLMs for thinking like human (Lee et al., 2023). On the micro aspect, brain neuron is activated by neurotransmitters and action potentials with a threshold for activation, where the message passing occurs when there is large potential difference (Zhang & Zhang, 2018). Analogously, gradients in ANN are similar to potential difference in brain neurons, and gradient can be viewed as the knowledge gap between new data and trained models, thus gradients can be exploited to interpret the relation between model and sample groups. Synapse is also an essential structure for bridging the message between neurons, where pioneering researches have demonstrated the potential of knowledge transfer by imitating synapse structure (Zenke et al., 2017;Hussain & Al Alili, 2017). Actually, unveiling the relations between brain structures and AI model transfer mechanism can advance the model evolution. Despite prosperity, on cross-domain transfer in spatiotemporal learning, how to investigate the specific mechanism that adapting to brain learning on transfer and generalization is still under-explored.
this section cite: ['b24', 'b49', 'b37', 'b5', 'b17', 'b28', 'b0', 'b18', 'b49', 'b46', 'b13']

Section: Preliminaries
Spatiotemporal Cross-Domain Observations. In an urban system, data can be collected from different sources. We can model diverse spatiotemporal learning tasks as spatial-temporal graph prediction, and the deterministic observations can be defined as (x j i ) c , which is an element in X = {X 1 , X 2 , ..., X C } ∈ R N ×T ×C . The (x j i ) c indicates the value on graph node j at timestamp i from c-th data source, where C represents the number of sources. As the data distribution can be shifted across temporal steps, then the task domain can be classified into both different temporal domains with changed distribution and source domains.
Neuro-Inspired Cross-Domain Learning. We define neuro-inspired cross-domain spatiotemporal learning model as an evolution model M, i.e.,
Y = M(X 1 , X 2 , . . . , X k ; θ M )(1)
where θ M denotes the learnable parameters of model M.
When the data from new domain X k+1 comes, we aim to quickly adapt M to M ′ , i.e.,
M ′ ← M(X k+1 , θ M ; θ M ′ ) (2
)
where θ M ′ denotes learnable parameters of updated M ′ . Proposition 3.1. Increased information with cross-domain learning.
Given spatiotemporal data observations from different sources {X 1 , X 2 , X 3 , ..., X k } and then there must be shared commonality among domain data patterns, i.e.,
∀i, j(1 ≤ i < j ≤ k), I(Xi; Xj) > 0(3)
then the well-learned information from the cross-domain learning model M is increased by continually receiving domain knowledge, i.e.,
Info(M(X1, ..., X k ; θM)) > Info(M(X1, ..., X k-1 ; θM)) > ... > Info(M(X1; θM)) (4
)
where Info is the information encapsulated in M.
The Prop.3.1 delivers that cross-domain learning with different data sources 'in harmony with diversity' can increase the learned information in model M and it can be proved in Appendix. A.
this section cite: []

Section: Methodology

this section cite: []

Section: Framework Overview
SynEVO constructs a neural synaptic spatiotemporal network to share and transfer cross-domain knowledge for generalization and adaptation. As illustrated in Fig. 1, our synaptic neural structure consists of three components, curriculum-inspired sample group re-ordering to determine the learning order of sequential samples from easy to difficult, complementary dual common-individual learners including an Elastic Common Container and task-independent personality extractor 1 to disentangle the task commonality and personality and an adaptive dynamic coupler to aggregate the dual learners so as to adapt shared patterns and preserve the individuality of tasks. It is noted that the Spatiotemporal learner backbone is implemented by Graph-WaveNet (GWN) (Wu et al., 2019).
this section cite: ['b38']

Section: Curriculum Guided Task Reordering
Learning from easy to difficult is a common practice for human acquiring knowledge and skills, which is named as curriculum learning (Bengio et al., 2009), e.g., the teaching process in our class also follows inculcating knowledge from basic to improved ones. To capture the commonality of data patterns in different domains, we propose curriculum guided task re-ordering. From an optimization perspective, directly confronting a complex task can lead the model to be caught in poor local optimums with exploding gradients. In contrast, by starting from a simple task and gradually increasing the difficulty, the model can be effectively guided to converge in the direction of global optimum, which enables better exploration in parameter space.
Gradients can characterize the consistency degree between training model and new sample groups, thus the gradients are exploited to indicate the difficulty of adapting models to samples. We then exploit the gradient to compute the adaptation between new feeding samples and the training model, and determine the learning order via imitating the curriculum learning process. Specifically, we apply the backward of loss to compute the gradient. For each input sample group X c from c-th domain, we initial a trainable model M c (X c ; θ Mc ) and train M c until the loss function of it converges. Then we backward the final loss to compute the gradients of θ M as {∇ 1 , ∇ 2 , . . . , ∇ n }, where ∇ i denotes the gradient of the i-th layer of θ M and n denotes the the number of the layers of θ M . After that, compute the sum of squares of the gradients by,
sum c = n i=1 ||∇ i || 2 2 (5
)
where ||∇ i || 2 2 denotes the square of the L2 norm of ∇ i . Then, we concatenate the gradient to denote the overall 1 As observations in source and temporal domains are organized into sample groups, which can be viewed as various tasks, here we interchangeably utilize sample group and tasks in our main text.
consistency between data and model by,
cat c = [∇ (expand) 1 ||∇ (expand) 2 || . . . ||∇ (expand) n ] (6)
where ∇ (expand) i denotes the expanded tensor of ∇ i and || denotes the concatenation of tensors.
With obtaining all cat c for input data X c , we identify the minimum value among them by min = arg min c sum c , which is considered as the compared bench. Next, to reorder other sample groups, we compute the vector difference between cat c and the bench one cat min by,
d c = cat c ⊖ cat min (7
)
where ⊖ is the element-wise minus for vectors. After that, we can reorder the input sample groups {X 1 , X 2 , . . . , X k } based on the length of d c , i.e., l(d c ) in ascending order and get the ordered sequence S = {X c1 , X c2 , . . . , X c k }, where k is the number of sample groups. Therefore, we can capture the inner relation between sample groups by gradients, which avoids the isolation of information, and allows the learning process from easy to difficult.
this section cite: ['b3']

Section: Complementary Dual Common-individual Learners
Inspired by complementary functions in brain memory, we construct a dual common individual learners to respectively accommodate two major knowledge based on three insights, 1) Complementary memory where neocortex remembers long-term and stable skills while hippocampus acquires new and quick knowledge. 2) More neurons are activated with knowledge increasing. 3) Distinguished patterns makes long-standing memory. Overall, our design inherits the complementary learning scheme into respective common container and personality pattern extractor. The common container is devised accounting for the core synaptic function to receive cross-domain common information with elastically increasing collective intelligence, and a task independent personality extractor is to characterize individual task features for quick adaptation. They cooperate with each other for generalized cross-domain learning.
this section cite: []

Section: ELASTIC COMMON CONTAINER
Deep learning models iteratively trained with new samples can automatically fuse patterns across all samples. Based on above analysis, the commonality should be expanded when the acquired knowledge is increasing with iteratively feeding into new samples. With well-organized task sequences, we are expected to mimic such knowledge expansion in brains for neural networks. In detail, we borrow a couple of simple yet effective strategies in deep learning to empower the model with elastic property. Dropout and L2 Regularization with weight decay control the overall complexity of model that potentially avoid over-fitting by the number of active neurons. For Dropout, every neuron can be set as zero (dropout) with probability p and each weight decay coefficient weight λ for L2 controls the importance of L2 item. The smaller probability p and smaller weight decay λ, the model is more active. Despite promising, how to quantitatively control the activeness of neurons by probability p and weight decay λ is still unclear. To address such quantitation challenge, we introduce Lemma. 4.1 from neuroscience (Gulledge et al., 2005).
Lemma 4.1. The probability of presynaptic neurotransmitter release can be described by a propagation model (Bertram et al., 1996;Schneggenburger & Neher, 2000),
P r = P 0 (1 -e -τ ) (8
)
where P r denotes the probability of neurotransmitter release, P 0 denotes the basic release probability, τ is the successive activeness difference between a pre-synaptic neuron and after-synaptic neuron.
Lemma. 4.1 suggests that the neurotransmitter can induce the activeness of neurons and we can exploit the such electric potential difference to mimic the process. Fortunately, earlier we have discussed that gradient can indicate the consistency between model and new samples, thus we take the second-order difference, the successive gradient variation |d| based on Eq. 7 as the propagation degree τ .
For dropout, we first define the matrix of all learnable parameter in model M as M , and provide the following definition of activated model parameters.
Definition 4.2. Activated model parameters. For parameter matrix [M ], its activeness matrix is defined as,
[A] x,y = 0 if [M ] x,y is dropped out, 1 otherwise.(9)
where [A] x,y , [M ] x,y denotes the element of matrix [A], [M ] on position (x, y) respectively. Based on the activeness matrix, the activated model parameters can be updated as, M = M ⊙A, where ⊙ is the Hadamard product.
The larger number of non-zero elements in [A] indicates more model activeness and capture more information with decreased probability p.
With above Lemma. 4.1, we further model the dropout factor p c as,
p c (d c ) = p 0 (1 -e l(dc)-dmax )(0 < p 0 ≤ 1) (10
)
where p c denotes the dropout factor for domain c, p 0 is a hyperparameter, d c denotes the vector difference for domain c against cat min and d max is the maximum length of all gradient vectors d c .
Similarly, for the weight decay coefficient of L2 regularization that controls the model in the optimizer, we can re-write the dynamic update of weight decay according the variation of gradient as below,
λ c (d c ) = λ 0 (1 -e l(dc)-dmax )(0 < λ 0 < 1) (11
)
In Eq. 11, λ c decreases as l(d c ) increases, which realize the elastic growth of the model and improves the generalization of the model. To this end, we consider that controlling the dropout value p c and weight decay coefficient λ c with the vector d c can efficiently realize the gradual release of model activeness, which has been shown in Fig. 2. With our synapse structure, common patterns are iteratively enhanced.
As new domain arrives, the overall knowledge boundary and learned parameter space of our common container are expanding. Then can do quick and light-weight adaptation from the existing knowledge space to new domain.
this section cite: ['b10', 'b4', 'b29']

Section: TASK INDEPENDENT PERSONALITY EXTRACTOR
Besides commonality, the personalized pattern of each task especially new task is also vital for domain adaptation. Formally, it is expected to derive an additional personality extractor g, which transforms the input X c to the output E c , i.e.,
E c = g(X c ; W g ) = W g X c .
Here, we define a new criterion D(A, B) to measure the difference between tensors A and B, i.e.,
D(A, B) = (A ⊖ B) 2(12)
Inspired by distinguishing patterns in memory for activating the neuron activeness, we explore the contrastive learning (Khosla et al., 2020;Hadsell et al., 2006) to implement such personality extractor,
R(E i , E j ; W g ) = ŷD(E i , E j ) +(1 -ŷ)max(0, m -D(E i , E j ))(13)
where R(E i , E j ) refers to contrastive objective function between two representations E i , E j . The domain indicator ŷ = 1 if X i and X j come from the same domain while ŷ = 0 indicates two samples from different domains, and m controls the minimum distance between X i and X j from different domains. Since the personality extractor g updates to approach the minimization objective, the representations E i and E j from the same domain become closer and the representations E i and E j from different domains are away from each other.
Therefore, the commonality and personality can be disentangled with the contrastive learning objective. As samples of a new domain comes, the personality extractor is capable of extracting the representation of the new domain, i.e., E c = g(X c ; W g ), and the commonality patterns are learned by the common container for commonality growth, which significantly enhances the ability of the model to comprehend both previous and new knowledge.
this section cite: ['b15', 'b12']

Section: Adaptive Dynamic Coupler
To achieve cross-domain adaptation while maintaining both personality and commonality, we construct an adaptive dynamic coupler to aggregate the commonality and personality.
When the sample group from a new domain X k+1 comes, the task independent personality extractor first extracts the representation of
X k+1 by E k+1 = g(X k+1 ; W g ).
this section cite: []

Section: Then we preserve a list G which contains the distance between the representation of the new domain and the trained domains based on Eq. 12.
To be specific, G is defined as, G = {D 1 , D 2 , . . . , D k }, where k is the number of trained domains in the elastic common container and D i denotes D(E k+1 , E i ). Let D min be the minimum D i in G and if 0 < D min < κ (where κ is a threshold which is a hyperparameter), it indicates that the new domain shares potential commonality with the trained domains. Then we can put it into the common container to compute the gradient and dynamically adjust the dropout factor p k+1 and the weight decay coefficient λ k+1 . At last, the common container trains the new domain based upon the adjusted factors and thus absorb the knowledge from the new domain to realize elastic growth. If D min ≥ κ, which indicates that the new domain almost shares no commonality with trained domains, then we re-instantiate personality extractor by initializing learnable parameters with previous extractor for a quick adaptation. The comparison of representations is implemented by a gate structure h,
h(Dmin, κ) = 1 if 0 < Dmin < κ 0 otherwise.(14)
To conclude, the overall learning objective of our model M' to the input X k+1 can be defined as,
Loss(θ M ′ ) = h(Dmin, κ)(L(M ′ (X k+1 , θM; θ M ′ ), Y k+1 ) +λ k+1 ||θ M ′ || 2 2 ) +(1 -h(Dmin, κ))L(M ′ (X k+1 , θinit; θ M ′ ), Y k+1 )(15)
where Y k+1 denotes the target value of M ′ (X k+1 ), λ k+1 is computed based on Eq. 10, ||θ M ′ || 2 2 denotes the square of L2 norm of θ M ′ and θ init denotes the new random initial learnable parameters of M ′ .
this section cite: []

Section: Experiment

this section cite: []

Section: Datasets
We collect and process four datasets for our experiments: 1) NYC (NewYorkCity, 2016): Include three months of traffic data consisting of four source domains which are CrowdIn, CrowdOut, TaxiPick and TaxiDrop collected from Manhattan in New York City. 2) CHI (CHICAGO, 2023): Consist of three source domains of traffic status, which are Risk, TaxiPick and TaxiDrop collected in the second half of 2023 from Chicago. 3) SIP: Includes three months of traffic data consisting of two source domains which are Flow and Speed collected from Suzhou Industrial Park. 4) SD (Liu et al., 2023): Include traffic flow data collected from San Diego in 2019.
this section cite: ['b23']

Section: Evaluation Metrics and Baselines
We apply three evaluation metrics in our experiments, which are mean absolute error (MAE), root mean square error (RMSE) and mean absolute percentage error (MAPE). We exploit seven prevalent baselines for evaluations, including STGNNs (STGCN (Yu et al., 2017), STGODE (Fang et al., 2021), GWN (Wu et al., 2019)), RNN-based models (AGCRN (Bai et al., 2020)) and attention-based models (STTN (Xu et al., 2020), ASTGCN (Guo et al., 2019), CMuST (Yi et al., 2024)).
this section cite: ['b43', 'b8', 'b38', 'b1', 'b40', 'b11', 'b42']

Section: Implementation Details
We split the datasets into training, validation and testing sets with the ratio of 7:1:2. Datasets NYC, CHI and SIP are utilized for validation on cross-source and cross-temporal domain tasks, while SD with one attribute but large time span is for cross-temporal domain evaluation. For the first three datasets, on cross-domain evaluation, we leave Tax-iPick, TaxiDrop and Speed as the evaluation domain on respective NYC/CHI/SIP sets, and let other data sources to be trained iteratively. For their cross-temporal domain evaluation, we divide one day into four equal periods, and leave the last period of a day on all source domains for evaluation. Regarding cross-temporal domain validation on SD, we divide one day into six equal periods and leave the last period for evaluation of temporal domain adaptation. We run each baseline three times and report the averaged results to reduce the influence of randomness issue. For curriculumguided task reordering, Adam optimizer (Kingma, 2014) is applied with initialized learning rate of 0.01 and weight decay of 0.001 for the initial learnable model M c . For complementary dual learners, we use the mean square error (MSE) as the criterion D of the personality extractor. For
this section cite: ['b16']

Section: Ablation Study
In order to uncover the significance of each module to the success of SynEVO, we perform an ablation study on crosstemporal domain adaptation via removing each module on the four datasets. The ablated variants are as follows. 1) SynEVO-REO: Remove the module of curriculum-guided sample group reordering. 2) SynEVO-Ela: Remove the dynamic adjustment of dropout factor p and weight decay coefficient λ with a static value, e.g., p = 0.1, λ = 0.001.
this section cite: []

Section: 3) SynEVO-PE: Without relation comparison on individual and commonality, any domain even unrelated sample groups can be fed into the model without a judgment gate.
Tab.4 shows the results of ablation studies. In general, removing abovementioned modules leads to the consistent performance drops. When removing the module of elastic growth, the performance drops the most by 44.2% on MAE, indicating the most important structure of elastic com- mon container for model evolution and adaptation. More specifically, when removing sample re-ordering, the performance drops by about 18.1%, while personality extractor is discarded, the performance drops by about 19.1%, which shows the personality and re-instantiate mechanism is also critical for ensuring model robustness and consistency.
this section cite: []

Section: Detailed Analysis
Uncovered sample group sequences for curriculum learning. In curriculum guided task reordering, the reordered the input group samples are illustrated in Fig. 3(a) based on the gradients. In our experiments, we find that domains from the same source are not necessarily next to each other in the ordered sequence S, which verifies that SynEVO has successfully uncovered hidden correlation information between domains. Moreover, it is demonstrated that reordering our input sample groups can learn certain commonality, empirically providing evidence for the effectiveness of cross-domain learning.
Observed quick adaptation via loss behavior. In elastic common container, the sample groups are periodically fed into models. We let SynEVO elastically grow to absorb the commonality by iteratively feeding sample groups, and visualize the training loss of two consecutive learning cycles on SD in Fig. 3(b). The blue curve denotes the first training cycle while the red one denotes the second. These two cycles share the same training data and order. Every mutation in the curve represents the input of a new domain. Obviously, the loss of the second cycle is much lower than the former one and the loss drops quickly after new domain input, which shows the quick adaptation by constructing learning tasks in a cycled and elastic manner.
this section cite: []

Section: Effective zero-shot adaptation.
To further evaluate the adaptation performance of SynEVO, we conduct zero-shot cross-temporal domain adaptation, i.e., testing without training, and make comparisons with the backbone GWN, where results are shown in Tab.5. Obviously, SynEVO outperforms better than GWN by averagely 29.8% on the four datasets, and it numerically proves that SynEVO has captured the hidden commonality of various input data so that it can achieve effective and even superior performance on zero-shot tasks.
this section cite: []

Section: More empirical analysis on task reordering and design of commonality extraction.
We supplement the experi-   Results are shown in Tab.6. The reversed order falls into inferior performances, which emphasizes the significance of reordering the tasks from easy to difficult in our design. Moreover, hard-to-easy performances are better than random ordering of SynEVO-REO, which may be attributed to common relations between neighboring tasks as they are ordered even the reverse one. Based on the experimental results, we can conclude our commonality learner and the setting of p are reasonable and empirically justified.
this section cite: []

Section: Hyperparameter Sensitivity Analysis
We varied Dropout p 0 from {0.1, 0.3, 0.5, 0.7, 1}, weight decay coefficient λ 0 from {0.01, 0.03, 0.05, 0.07, 0.1}, and distance threshold κ from {1 × 10 3 , 1 × 10 4 , 1 × 10 5 , 1 × 10 6 }. Results shown in Fig. 4-Fig. 7 indicate that the optimal settings are κ = 1 × 10 3 on all datasets, p 0 = 0.5, λ 0 = 0.05 on NYC and SIP, p 0 = 1, λ 0 = 0.1 on CHI and p 0 = 0.7, λ 0 = 0.07 on SD. For κ, if the threshold is extremely small, it means no new domain is allowed in, which will violate the principle of 'harmony with diversity for collective intelligence' then we can observe that with increasing and more relaxed condition for fusion, more noise will be introduced to reduce the performances. Thus the trade-off between commonality and individual feature extraction should be obtained during model design.
this section cite: []

Section: Conclusion
In this paper, we propose a novel NeuroAI framework SynEVO to enable cross-domain spatiotemporal learning for few-shot domain adaptation. From neuroscience theories, a curriculum-guided sample group re-ordering, a couple of complementary dual learners which includes an elastic common container, and a task independent personality extractor are proposed to capture commonality in an elastic manner. The adaptive dynamic coupler determines whether the new feeding samples can be aggregated into SynEVO for achieving model evolution. Extensive experiments on both cross-source and cross-temporal domains validate the 0.5% to 42% improvements against baselines. For future work, we plan to mine the more inner mechanism of human brain to facilitate the generalization of general AI models. Moreover, our model can be generally nested within other neural networks. In other areas, it is applicable to utilize our evolvable 'data-model' collaboration to decouple the invariant and variable patterns, and reconstruct the OOD distribution with new patterns.
this section cite: []

Section: References
Ref_id:b0 Title: Learning fast, learning slow: A general continual learning method based on complementary learning system Year: (2022)
Ref_id:b1 Title: Adaptive graph convolutional recurrent network for traffic forecasting Year: (2020)
Ref_id:b2 Title: Synaptic plasticity and the neurobiology of learning and memory Year: (2007)
Ref_id:b3 Title: Curriculum learning Year: (2009)
Ref_id:b4 Title: Singledomain/bound calcium hypothesis of transmitter release and facilitation Year: (1996)
Ref_id:b5 Title: Information maximizing curriculum: A curriculum-based approach for learning versatile skills Year: (2024)
Ref_id:b6 Title: Trafficstream: A streaming traffic flow forecasting framework based on graph neural networks and continual learning Year: (2021)
Ref_id:b7 Title:  Year: (2023)
Ref_id:b8 Title: Spatial-temporal graph ODE networks for traffic flow forecasting Year: (2021)
Ref_id:b9 Title: Spatio-temporal field neural networks for air quality inference Year: (2024)
Ref_id:b10 Title: Synaptic integration in dendritic trees Year: (2005)
Ref_id:b11 Title: Attention based spatial-temporal graph convolutional networks for traffic flow forecasting Year: (2019)
Ref_id:b12 Title: Dimensionality reduction by learning an invariant mapping Year: (2006)
Ref_id:b13 Title: A pruning approach to optimize synaptic connections and select relevant input parameters for neural network modelling of solar radiation Year: (2017)
Ref_id:b14 Title: Synaptic signaling in learning and memory Year: (2016)
Ref_id:b15 Title: Supervised contrastive learning Year: (2020)
Ref_id:b16 Title: A method for stochastic optimization Year: (2014)
Ref_id:b17 Title: What learning systems do intelligent agents need? complementary learning systems theory updated Year: (2016)
Ref_id:b18 Title: Scaling reinforcement learning from human feedback with ai feedback Year: (2023)
Ref_id:b19 Title: Attention in psychology, neuroscience, and machine learning Year: (2020)
Ref_id:b20 Title: Spatial-temporal large language model for traffic prediction Year: (2024)
Ref_id:b21 Title: Efficient multivariate time series forecasting via calibrated language models with privileged knowledge distillation Year: (2025)
Ref_id:b22 Title: TimeCMA: Towards llm-empowered multivariate time series forecasting via cross-modality alignment Year: (2025)
Ref_id:b23 Title: Largest: A benchmark dataset for large-scale traffic forecasting Year: (2023)
Ref_id:b24 Title: Cognitive neuroscience and education Year: (2024)
Ref_id:b25 Title: Less is more: Efficient time series dataset condensation via two-fold modal matching Year: (2024)
Ref_id:b26 Title: A parameter-efficient federated framework for streaming time series anomaly detection via lightweight adaptation Year: (2025)
Ref_id:b27 Title: Nyc dataset. Website Year: (2016)
Ref_id:b28 Title: Complementary learning systems Year: (2014)
Ref_id:b29 Title: Intracellular calcium dependence of transmitter release rates at a fast central synapse Year: (2000)
Ref_id:b30 Title: The information bottleneck method Year: (2000)
Ref_id:b31 Title: Brain-inspired replay for continual learning with artificial neural networks Year: (2020)
Ref_id:b32 Title: Pattern expansion and consolidation on evolving graphs for continual traffic prediction Year: (2023)
Ref_id:b33 Title: Stone: A spatio-temporal ood learning framework kills both spatial and temporal shifts Year: (2024)
Ref_id:b34 Title: Coscl: Cooperation of small continual learners is stronger than a big one Year: (2022)
Ref_id:b35 Title: Incorporating neuro-inspired adaptability for continual learning in artificial intelligence Year: (2023)
Ref_id:b36 Title: A comprehensive survey of continual learning: theory, method and application Year: (2024)
Ref_id:b37 Title: A survey on curriculum learning Year: (2021)
Ref_id:b38 Title: Graph wavenet for deep spatial-temporal graph modeling Year: (2019)
Ref_id:b39 Title: Deciphering spatio-temporal graph forecasting: A causal lens and treatment Year: (2024)
Ref_id:b40 Title: Spatial-temporal transformer networks for traffic flow forecasting Year: (2020)
Ref_id:b41 Title: Co-prediction of multiple transportation demands based on deep spatio-temporal neural network Year: (2019)
Ref_id:b42 Title: Get rid of isolation: A continuous multi-task spatio-temporal learning framework Year: (2024)
Ref_id:b43 Title: Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting Year: (2017)
Ref_id:b44 Title: Environment-aware dynamic graph learning for out-of-distribution generalization Year: (2023)
Ref_id:b45 Title: Unist: A prompt-empowered universal model for urban spatiotemporal prediction Year: (2024)
Ref_id:b46 Title: Continual learning through synaptic intelligence Year: (2017)
Ref_id:b47 Title: Taxi demand prediction using parallel multi-task learning model Year: (2020)
Ref_id:b48 Title: Deep spatio-temporal residual networks for citywide crowd flows prediction Year: (2017)
Ref_id:b49 Title: Artificial neural network. Multivariate time series analysis in climate and environmental research Year: (2018)
Ref_id:b50 Title: Riskoracle: A minute-level citywide traffic accident forecasting framework Year: (2020)
Ref_id:b51 Title: Maintaining the status quo: Capturing invariant relations for ood spatiotemporal learning Year: (2023)
