Title: An Analysis for Reasoning Bias of Language Models with Small Initialization
Abstract: Transformer-based Large Language Models (LLMs) have revolutionized Natural Language Processing by demonstrating exceptional performance across diverse tasks. This study investigates the impact of the parameter initialization scale on the training behavior and task preferences of LLMs. We discover that smaller initialization scales encourage models to favor reasoning tasks, whereas larger initialization scales lead to a preference for memorization tasks. We validate this reasoning bias via real datasets and meticulously designed anchor functions. Further analysis of initial training dynamics suggests that specific model components, particularly the embedding space and self-attention mechanisms, play pivotal roles in shaping these learning biases. We provide a theoretical framework from the perspective of model training dynamics to explain these phenomena. Additionally, experiments on realworld language tasks corroborate our theoretical insights. This work enhances our understanding of how initialization strategies influence LLM performance on reasoning tasks and offers valuable guidelines for training models.

Section: Introduction
With the rapid advancement of deep learning technologies, Large Language Models have achieved remarkable success in the field of Natural Language Processing (NLP). These models have demonstrated exceptional capabilities across a  wide range of tasks, from text generation to complex reasoning (Wei et al., 2022a;Achiam et al., 2023;Liu et al., 2024). Reasoning, in particular, is a critical ability for LLMs. A number of studies have focused on improving the reasoning ability of these models through data-driven approaches, such as RHO-1 (Lin et al., 2024) and Phi-4 (Abdin et al., 2024). However, there remains an ongoing debate as to whether LLMs genuinely learn the underlying logical rules or merely mimic patterns observed in the data (Marcus, 2003;Smolensky et al., 2022).
An alternative approach to enhancing the reasoning ability of LLMs focuses on the model architecture and its training process. In one such study examining the use of Transformers to model compositional functions, it was observed that the scale of model parameter initialization significantly impacts the model's reasoning behavior (Zhang et al., 2024a;2025). Specifically, smaller initialization scales bias the model toward fitting the data by learning primitive-level functions and compositional rules, whereas larger initialization scales tend to encourage memorization of input-output mappings. A qualitative rationale for this phenomenon has been proposed: with a small initialization, a well-documented effect known as neuron condensation emerges during training (Xu et al., 2025;Luo et al., 2021;Zhou et al., 2022;Zhang et al., 2022;Zhang & Xu, 2023;Zhang et al., 2023;Zhang & Xu, 2024). This phenomenon suggests that neurons within the same layer tend to behave similarly, promoting data fitting with the least possible complexity. To achieve a low-complexity result, the model must learn a minimal set of rules leading to capture the intrinsic primitive functions and compositional rules. However, this rationale does not reveal a critical question: how the optimization process, together with the Transformer structure, can achieve reasoning solutions with small initialization?
In this work, we identify a reasoning bias during the training of neural networks that learn natural language when initialized with small parameter scales. To illustrate this phenomenon, we employ a GPT-2 model (Radford et al., 2019) to train on a mixed dataset comprising two types of language data with distinct levels of reasoning complexity, within a single next-token prediction training framework. The first dataset, PrOntoQA (Saparov & He, 2023), consists of question-answering examples that include chains of thought, which explicitly describe the reasoning necessary to answer the questions correctly. The second dataset, TinyStories (Eldan & Li, 2023), is a synthetic corpus of short stories containing only words typically understood by children aged 3 to 4 years. As shown in Figure 1, the training loss for PrOntoQA decreases significantly faster than for TinyStories, suggesting that the model encounters and learns the reasoning patterns more readily.
We uncover a key mechanism whereby reasoning tasks are learned earlier during training because the tokens associated with these tasks become more differentiated in the embedding space at an early stage of the training process. We validate this mechanism using both synthetic data and real-world datasets. Furthermore, we provide a theoretical explanation for the evolution of token embeddings, which depends on the distribution of sample labels. Since each token is encoded as a one-hot vector, its embedding is adjusted based on the loss associated with the labels of that token. Consequently, different label distributions can lead to distinct learning behaviors for token embeddings. For memory tasks, the labels associated with each token are typically random and lack explicit structure, which results in similar distributions for different memory token labels. As a result, the embeddings for memory tokens are difficult to differentiate in the early stages of training. In contrast, reasoning tokens often exhibit distinct label distributions, leading to more differentiated embedding vectors for these tokens. These insights are elaborated through a simplified model using a multi-layer perceptron (MLP) and embedding structure, followed by an analysis of a Transformer model.
The primary contribution of this research lies in uncovering the impact of the parameter initialization scale on the training behavior and task preferences of LLMs. By combining theoretical analysis with empirical evidence, we enhance the understanding of LLM training dynamics and provide new insights for optimizing model initialization strategies.
this section cite: ['b1', 'b22', 'b21', 'b0', 'b26', 'b34', 'b45', 'b25', 'b57', 'b50', 'b29', 'b32', 'b13']

Section: Preliminary

this section cite: []

Section: Synthetic Composition Task
To study the task bias during the training, we use the concept of anchor function (Zhang et al., 2024b) to construct a dataset that contains tasks of different reasoning complexities. We consider all tokens belonging to positive integers. A set of tokens are designated as anchors, denoted as A := {a ∈ N + |α min ≤ a ≤ α max }, where each anchor represents an addition/randomness operation in this work. Another set of tokens are designated as keys, denoted as Z := {z ∈ N + |ζ min ≤ z ≤ ζ max } with the assumption that Z ∩ A = ∅. For convenience, we denote
N Z = ζ max -ζ min + 1 and N A = α max -α min + 1.
This section introduces two types of sequence mappings. The first step involves constructing a sequence of positive integers with length L, represented as:
X (q,L) = {X|X = [z 1 , • • • , z p , a p+1 , • • • , a p+q , z p+q+1 , • • • , z L ] , z i ∈ Z, a i ∈ A} .(1)
We define q as the number of anchors in the sequence, and p as the index of the element immediately preceding the first anchor element a p+1 in the sequence.
For a given sequence X ∈ X (q,
L) , we define its key-anchor combination as (z p , a p+1 , • • • , a p+q ), which is denoted concisely as pair (z p , a), and other keys are regarded as noise in this input sequence. The anchor set A is divided into two subsets, i.e., reasoning anchor set A rsn and memory anchor set A mem , where A = A rsn ∪ A mem and A rsn ∩ A mem = ∅.
Reasoning mapping. For any X with a p+i ∈ A rsn , i = 1, • • • , q, we define the following mapping as a reasoning mapping
F rsn (X) = z p + q i=1 a p+i .
this section cite: []

Section: Memory mapping.
For any key-anchor pair (z p , a), where each element in a belongs to A mem , we randomly sample a number y (zp,a) from Z as the memory mapping label of any sequence X containing (z p , a), i.e. F mem (X) = y (zp,a) , ∀X contains (z p , a) .
A detailed example is provided in Figure 2. It's noted that the key-anchor pair may occur at any position within the sequence. The label is independent of both the noise tokens and the position of the key-anchor pair within the sequence but is determined solely by the value of the key-anchor pair.
this section cite: []

Section: Dataset Setup
In this study, we denote a data pair as (X, y), where X represents the input sequence and y corresponds to its associated label. We define y as the one-hot encoded representation of y for convenience. For memory mapping, all data are contained within the training set D mem , and no test set is employed, as the generalization is not considered in this framework.
this section cite: []

Section: Model Architecture
We give the formulation of the embedding space and self-attention module here for notation convenience. Let d vob , d m , d k denote the vocabulary size, embedding space dimension, and query-key-value projection dimension, respectively. For any token s, denote its one-hot vector by e s ∈ R 1×d vob . The embedding vector of s is w emb,s = e s W emb where W emb ∈ R d vob ×dm is the embedding matrix. Additionally, the self-attention operator Attn on any embedding sequence X ∈ R L×dm is defined as:
Attn (X) = g mask XW Q W KT X T √ d k ,(2)
O = Attn (X) XW V W O ,(3)
where g (•) is the softmax operator and T means the matrix transposition. W Q , W K , W V ∈ R dm×d k are the query, key and value projection matrices, respectively. W O ∈ R d k ×dm represents the output projection matrix. The detailed expression of multilayer Transformer models can be found in Appendix A.1.
this section cite: []

Section: Parameter Initialization
Given any trainable parameter matrix W ∈ R d1×d2 , where d 1 and d 2 denote the input and output dimensions, respectively, its elements are initialized according to a normal distribution:
W i,j ∼ N 0, d -γ 1 2 ,
where γ is the initialization rate. Specifically, the initialization scale decreases as γ increases. Note that γ = 0.5 is commonly used in many default initialization methods, such as LeCun initialization (LeCun et al., 1998) and He initialization (He et al., 2015). As the network width towards infinity (Luo et al., 2021;Zhou et al., 2022), the training of the network with γ > 0.5 exhibits significant non-linear characteristics, i.e., condensation. Therefore, initialization scales with γ > 0.5 are generally considered small.
this section cite: ['b19', 'b15', 'b25', 'b57']

Section: Result
In this section, we present empirical evidence of a reasoning bias during the training of Transformers with small initialization by utilizing composition tasks. To further explore this phenomenon, we introduce a simplified model consisting of an embedding layer and a multi-layer perceptron, which reproduces the reasoning bias and enables theoretical analysis. A key mechanism underlying this bias is that the training behavior of each token's embedding depends on the label distribution of the samples containing that token. For reasoning anchors, the label distributions typically exhibit greater variability compared to memory anchors, leading to the more rapid differentiation of their embeddings early in training. Additionally, we extend our analysis to the Transformer architecture to demonstrate the generalizability of this effect. For each observation mentioned in the following, we provide a similar analysis with larger initialization scales in Appendix C.
this section cite: []

Section: Reasoning Bias in Transformer with Composite Anchor Functions
In our experiment, we set that q = 2, L = 9. The dataset is constructed with the following configurations:
Z = {21, • • • , 120}, A mem = {1, • • • , 10}, A rsn = {11, • • • , 20}
and M = {(11, 13), (13, 11)}. The dataset contains 200000 data pairs, ensuring an equal number of samples for each anchor combination. The loss function employed is Cross Entropy and the optimization algorithm used is AdamW. The model architecture comprises a decoderonly Transformer structure with 2 layers and a single attention head. We train the model under different initialization scales with γ = 0.3, 0.5, 0.8 utilizing the last token prediction method. Additional details about the experimental setup can be found in Appendix A.
To investigate the impact on training behavior under varying initialization scales, we analyze the dynamics of loss and prediction accuracy on D mem , D rsn,train and D rsn,test . As illustrated in Figure 3A, for γ = 0.3, the losses on D mem and D rsn,train decrease at nearly identical rates, while the loss on D rsn,test remains effectively unchanged. This observation suggests that the model primarily memorizes the training data in this setting. In contrast, when γ = 0.8, the losses on D rsn,train and D rsn,test decrease significantly faster than the loss on D mem . This behavior indicates a shift towards a reasoning bias in the model. These findings reveal that the model's learning bias is influenced by the initialization scale: as the initialization scale decreases, the model exhibits a progressively stronger reasoning bias.
this section cite: []

Section: Simplified Model: Phenomena and Analysis
To further investigate the underlying cause of the reasoning bias under a small initialization scale, we begin by employing a two-layer fully connected network to address a particular task, where p ≡ 1 and L = q + 1. The network structure is defined as follows:
Definition 1. Given that W (1) ∈ R dm×d f , W (2) ∈ R d f ×d vob , and σ as the activation function. Given any sequence X ∈ X (q,q+1) , we define the Embedding-MLP
model (Emb-MLP) G θ as G θ (X) := σ s∈X w emb,s W (1) W (2) .
Comparing with a large initialization scale (γ = 0.3), a noticeable reasoning bias can still be observed in Figure 3B for a small initialization scale (γ = 0.8).
Embedding space exhibits distinct patterns To investigate the causes of the reasoning bias under small initialization for such a simplified model, it's critical to understand the structure of the embedding space. Figure 4A depicts the cosine similarity matrices for embeddings of memory anchors and reasoning anchors at epochs 50 and 900. The results reveal that the cosine similarity between reasoning anchors s i , s j decreases with the increase of |s i -s j |, suggesting that reasoning anchors quickly establish a hierarchical structure within the embedding space. In contrast, the memory anchors exhibit consistently high similarity and alignment, leading to a lack of differentiation among them. Nevertheless, given that the model needs to learn more primitive-level mappings for memory mapping than reasoning mapping, the embedding space associated with memory mapping should, in principle, exhibit greater complexity and variability. This phenomenon reveals that the primary challenge preventing the model from effectively learning memory mapping could highly possibly lie in its difficulty in identifying and differentiating between individual anchors. d vob 1 and P s j -1 d vob 1 for any reasoning anchor si, sj, exhibiting a similar structure to the embedding space of reasoning anchors observed in A.
this section cite: []

Section: Target distribution shapes the embedding This phenomenon can be interpreted through the training dynamics.
To facilitate our analysis, we give the following assumption (Chen et al., 2024):
Assumption 1. The activation function σ ∈ C 2 (R), and there exists a universal constant C L > 0 such that its first and second derivatives satisfy
||σ ′ (•)|| ∞ ≤ C L , ||σ ′′ (•)|| ∞ ≤ C L . Moreover, σ(0) = 0, σ ′ (0) = 1.
For any token s, let X s,i , y s,i ns i=1 denote all input sequences containing s and corresponding labels, where n s means the appearance times of s. As the initialization scale decreases, with Assumption 1, we have
σ ′ x∈X i w emb,x W (1) = 1, softmax G θ X s,i = 1 d vob 1,
where 1 ∈ R 1×d vob means the vector with all elements equal to 1. Then the gradient flow of w emb,s could be approximated by the limit formulation, i.e.,
dw emb,s dt = 1 n ns i=1 y s,i - 1 d vob 1 W (2)T W (1)T , (4
)
where n represents the count of all tokens. We consider n → ∞ to obtain the asymptotic form of the following gradient flow.
Proposition 1. For any token s, denote Y s as a random variable, which takes values randomly from the label of any input sequence that contains token s. In the limit n → ∞, we define P s with its i-th element as the probability of Y s = i, i.e., P s i = P (Y s = i). Assume the ratio of the token s in the whole dataset r s := ns n remains constant, then (4) can be approximated as:
dw emb,s dt = r s P s - 1 d vob 1 W (2)T W (1)T . (5
)
The proof is provided in Appendix B.1. Proposition 1 demonstrates that for any token s, its embedding vector is dominated by the distribution of Y s which indicates that it's significant to discuss the distribution Y s for different tokens. Firstly, we define the following random variables (U for discrete uniform distribution):
Z ∼ U (Z) , A j ∼ U (A rsn ) , j = 1, 2, • • • , q. (6
)
Then we have the following results:
P (Y s = i | s ∈ A mem ) = 1 N Z δ i∈Z ,(7)
and
P (Y s = i | s ∈ A rsn ) =P   Z + q-1 j=1 A j = i -s | s ∈ A rsn   .(8)
Equation ( 7) reveals that the information to different memory anchors is identical such that the embedding space of memory anchors exhibits a high similarity. However, (8) demonstrates that the distribution of Y s exhibits shifts in the mean values that depend on the specific s for any s ∈ A rsn .
Figure 4B and 4C visualize the distribution of P s -1 d vob 1 and the resulting cosine similarity among different reasoning anchors s, suggesting that the labels' distributions play a critical role in establishing the embedding structure of reasoning anchors during the early stages of training, facilitating the differentiation among the embeddings associated with different reasoning anchors. The detailed formulations can be found in Appendix B.2.
this section cite: ['b5']

Section: Transformer with General Task
In the previous section, we investigate the key mechanisms driving the learning bias of Emb-MLP and analyze the dynamics of its embedding space. However, when applied to a general sequence containing some degree of noise, i.e., L > q + 1, we find the MLP model fails to perform effectively due to its inability to extract critical tokens z p , a p+1 , and a p+2 . In contrast, Transformer architectures overcome this limitation through self-attention mechanisms, which can identify the key and anchor elements and propagate their information.
In the following section, we conduct an in-depth analysis of the Transformer's characteristics and processing mechanisms under a small initialization scale. Specifically, we investigate whether the embedding space exhibits similar phenomena to those observed in Emb-MLP and assess how the model captures critical information from the input sequence.
Embedding space. The embedding space of the Transformer exhibits a phenomenon similar to that observed in the Emb-MLP. Figure 5A illustrates the cosine similarity among different anchors' embedding vectors, revealing distinct patterns for reasoning and memory tasks. Reasoning anchors display a hierarchical structure, the further distance, the smaller similarity, suggesting a clearer organization within the embedding space. In contrast, memory anchors exhibit high similarity and alignment. Additionally, we apply Principal Component Analysis (PCA) to the entire embedding space to examine its structural properties. The results in Figure 5B reveal a strong inherent numerical order. This structure is particularly advantageous for reasoning tasks, as it supports the model's capacity to generalize based on the underlying numerical relationships.
this section cite: []

Section: First attention module.
As illustrated in Figure 6, the first attention matrix approaches the behavior of an average operator when initialized with small scales, such that (Attn (X) V ) i = 1 i j≤i V j , where V = XW V . Consequently, each token aggregates information from all preceding tokens. Additionally, the largest singular value of W V is significantly larger than the remaining singular values, and its corresponding singular vector is aligned closely with the embedding vectors of reasoning anchors, but nearly orthogonal to those of memory anchors. These phenomena suggest reasoning anchors are prominently captured by W V and subsequently propagated to all subsequent tokens in the sequence via the average operation. However, the memory anchors are not distinctly identified, indicating that the model faces challenges in capturing significant information from a memory sequence effectively. More analysis of W V can be found in Appendix B.8.
this section cite: []

Section: Second attention module.
The second attention module functions to extract the key, and propagate its information to the final position in the sequence. This is facilitated through the use of position embeddings. Since this mechanism is applicable to both memory and reasoning tasks, we provide a detailed explanation in Appendix D.
this section cite: []

Section: Theoretical analysis.
Based on the observations from the experiments, we extract the sketch component of the model, which is crucial to its learning preferences, and analyze the underlying mechanisms for its occurrence. We define the following one-layer Transformer model:
Definition 2 (One-layer Transformer). Let d f ∈ N + denotes the hidden layer of the feedforward neural network (FNN). For any X ∈ X (q,L) , denote Attn W emb,X by A, then we define f θ : X (q,L) → R dm as follows:
f θ (X) =σ((A L,: W emb,X W V W O + W emb,X L,: )W f 1 )W f 2 +A L,: W emb,X W V W O + W emb,X L,: , (9
) where W f 1 ∈ R dm×d f , W f 2 ∈ R d f ×dm
are the feedforward layer projection matrices. The subscript L, : in A L,: and W emb,X L,:
denotes the L-th row.
Definition 2 is introduced to facilitate the theoretical analysis, excluding the Layer Normalization and the final projection operator, as they do not impact our results (see Appendix F).
As we observed, with a small initialization scale, the attention operator A can be interpreted as an average operator. Specifically, we have Lemma 1. For any ε ∈ (0, 1], there exists C > 0 such that for any γ > C, the elements of A at initialization, denoted by A i,j , satisfy A i,j -1 i ≤ ε for any i ≤ j with probability at least 1 -ε.
Denote that W f = W f 1 W f 2 , W V O = W V W O
and W = W f,T + I W V O,T + I , where the identity matrix I comes from the resnet. Using techniques similar to those employed in the previous section, we derive the gradient flow of w emb,s under small initialization scales as follows:
Proposition 2. For any s ∈ A mem , let n, γ → ∞, with Assumption 1 we have the following result:
dw emb,s dt = r s L δ Z N Z - 1 d m 1 W . (10
)
Proposition 3. For any s ∈ A rsn , let n, γ → ∞, with Assumption 1 we have the following result:
dw emb,s dt = r s L P s - 1 d m 1 W ,(11)
where the i-th element of P s is defined as
P s i = P Z + q-1 j=1 A j = i -s | s ∈ A rsn .
Furthermore, we utilize the normal distribution to approximate the distribution of Y s , s ∈ A rsn and give an approximation of w emb,s to describe the overall structure and internal relationships within the embedding space observed in real-world training scenarios.
Theorem 1. let n → ∞, define the learning rate η and assume that L -q = O(1), rsη L = O(1) and
||w emb || ∞ ≤ O (d -γ m ) at initialization. We propose the approximation of w emb,s , s ∈ A rsn by wemb,s j = C 1 C 2 e - (j-s) 2 2σ P - 1 d m + ε,(12)
where C 1 , C 2 , σ P are constants depending on r s , η, L, q and ε ∼ N (0, (d -γ m ) 2 ). Then we have the following result
sup i,j wemb,sj , wemb,si -w emb,sj , w emb,si ≤ O d 1-γ m q -1 2 + d -γ m ,(13)
where (•, •) denotes the inner production.
Additionally, for any key z ∈ Z and A mem , we could have a similar result. To validate our theory analysis, we set the detailed formulation of reasoning anchors and keys as wemb,s i = e -(i-s) 2 12 -1 d m + ε, s ∈ A rsn , wemb,s i = e -(i-s) 2 12 + ε, s ∈ Z.
(14)
Figure 5C exhibits the cosine similarity among the wemb,s for any s ∈ A rsn (top) and compare cos w emb,15 , w emb,sj in real training process with the theoretical approximation cos wemb,15 , wemb,sj (bottom, a complete comparison is provided in Appendix B.7). Figure 5D presents the PCA projection of w emb,s for s ∈ A rsn and Z, respectively. These visualizations exhibit a strong alignment with the experimental observations, thereby substantiating the validity of our analysis. The proofs of our theoretical results are provided in Appendix B.4, B.5, and B.6.
this section cite: []

Section: Real Language Tasks
For the experiment in Figure 1, we also conduct comparisons with initialization scales γ = 0.3 and 0.5. To quantify the reasoning bias of the model, we define the following metric:
∆L := L Tinystories -L PrOntoQA L PrOntoQA ,
where L Tinystories and L PrOntoQA denote the loss on TinyStories and PrOntoQA, respectively. As γ increases, ∆L exhibits an upward trend, indicating a growing bias for reasoning task, which is depicted in Figure 7A. To further validate our analysis, we examine the embedding space of the GPT-2 model during the early stages of training which we train with a small initialization scale. Figure 7B reveals that the embeddings of tokens in PrOntoQA are significantly more distinguishable from each other compared to the tokens in TinyStories. The average cosine similarity among the PrOntoQA is 0.123 while 0.531 among the TinyStories. These results provide strong support for our analysis, highlighting the impact of embedding distinguishability on training preference. Previous sections reveal that under small initialization, the distribution of labels plays a pivotal role in shaping the embedding space of tokens and regulating the model's training dynamics. To more intuitively demonstrate the impact of the label distribution of each token on its output, we designed four groups of memory mappings. The label ranges of the four groups are set to {30, • • • , 29 + 20 × i}, i = 1, 2, 3, 4. The right picture of Figure 8 illustrates the distribution of the model's outputs for each group during the early stages of training. Notably, it can be observed that even at this initial stage, when the model's accuracy is still relatively low, its outputs do not exceed the range of the label distributions. This highlights the critical influence of label distributions on the token structure, which in turn significantly impacts the model's outputs. Additionally, we compare two memory tasks with differing label distributions. In the first task, denoted F mem,1 , for any keyanchor pair (z p , a), the label y (zp,a) is randomly sampled from Z. In the second task F mem,2 , y (zp,a) is randomly sampled from z p -p+q i=p+1 a i , • • • , z p + p+q i=p+1 a i . While both tasks are clearly memory tasks, the label distributions in F mem,2 vary depending on the anchor. As shown in the left picture of Figure 8, the learning rate for F mem,2 is significantly faster than that for F mem,1 . This observation underscores the crucial role that label distribution plays in the model's learning process.
this section cite: []

Section: Effect of Label's Distribution

this section cite: []

Section: Related Works
Recent advancements in large language models have shown remarkable capabilities, often surpassing human-level performance in many tasks (Fu et al., 2022;Wei et al., 2022a). However, despite their strong performance in many aspects (Srivastava et al., 2022), LLMs face challenges in handling complex reasoning tasks (Csordás et al., 2021;Dziri et al., 2024;Hupkes et al., 2018;Lepori et al., 2023;Okawa et al., 2023;Yun et al., 2022;Wang et al., 2024d;Csordás et al., 2022). For example, Ramesh et al. (Ramesh et al., 2023) show that Transformers trained on a synthetic benchmark struggle when tasked with combining multiple reasoning steps. Similarly, Liu et al. (Liu et al., 2022) suggest that shallow Transformers tend to learn shortcuts during training, which limits their ability to perform well in more complex reasoning scenarios. Several strategies have been proposed to address these challenges, such as encouraging the generation of explicit reasoning steps in a single output (Wei et al., 2022b) or using LLMs to iteratively produce reasoning steps (Creswell et al., 2022;Creswell & Shanahan, 2022). Despite these efforts, achieving reliability remains a significant challenge. Additionally, some studies have explored the internal mechanisms of language models to enhance their performance (Wang et al., 2024b;c;2023), but they often do not address the impact of training dynamics on the model's final behavior. To better understand these models' behaviors and inner workings, Zhang et al. (Zhang et al., 2024b) introduced anchor functions as a tool for probing Transformer behavior. Building on this framework, our research investigates how different initialization scales influence model reasoning bias and internal mechanisms from a perspective of training dynamics.
The fitting ability and generalization of deep learning models are critical in understanding deep learning (Breiman, 1995;Zhang et al., 2016), and the initialization of neural network parameters plays a crucial role in determining the network's fitting results (Arora et al., 2019;Chizat & Bach, 2018;Zhang et al., 2019b;E et al., 2020;Jacot et al., 2018;Mei et al., 2018;Rotskoff & Vanden-Eijnden, 2018;Sirignano & Spiliopoulos, 2020;Williams et al., 2019). Luo et al. (Luo et al., 2021) and Zhou et al. (Zhou et al., 2022) primarily identify the linear and condensed regimes in wide ReLU neural networks. In the condensed regime, neuron weights within the same layer tend to become similar. A body of research indicates that condensed networks often exhibit strong generalization capabilities (Zhang et al., 2022;Zhang & Xu, 2023;Zhang et al., 2023;Zhang & Xu, 2024). See (Xu et al., 2025) for an overview of condensation. In our study, we demonstrate that with small initialization values, the parameters of the embedding layer can reach a low-rank state rather than a condensed state. This means that while embeddings of different tokens become linearly dependent, they are not identical. This distinction allows low-rank models to effectively capture essential patterns and generalize well without the stringent alignment required by condensation, which is particularly important for applications such as word embedding matrices where distinct representations for different tokens are necessary. Recent investigations have also explored how initialization affects the training dynamics of LLMs (Huang et al., 2020;Liu et al., 2020;Trockman & Kolter, 2023;Wang et al., 2024a;Zhang et al., 2019a;Zhu et al., 2021). These studies mainly examine how the scale of initialization influences the stability of the training process and is vital for ensuring efficient and effective training of LLMs. In our work, we observe that different initialization schemes result in varying speeds of convergence for memorization tasks versus reasoning tasks and provide a theoretical rationale for this behavior.
this section cite: ['b14', 'b35', 'b9', 'b11', 'b17', 'b20', 'b28', 'b46', 'b10', 'b30', 'b23', 'b3', 'b48', 'b2', 'b6', 'b12', 'b18', 'b27', 'b31', 'b33', 'b44', 'b25', 'b57', 'b50', 'b45', 'b16', 'b24', 'b36', 'b58']

Section: Conclusions
In this paper, we investigate the underlying mechanism of which small initialization scales promote a reasoning preference in language models. Our findings suggest that the label distribution of tokens plays a pivotal role in shaping the embedding space, thereby influencing the learning dynamics and task complexity. Our result can be readily extended to the next-token prediction training and obtain similar results. This perspective is supported through a combination of experimental observations and theoretical analysis, providing a deeper understanding of how initialization strategies impact task-specific behavior in language models. under small initialization scales, as well as the training behavior of individual modules within the model architecture. These findings not only contribute to understanding the model behavior and training mechanisms, but also help with optimizing model initialization strategies and designing novel algorithms to enhance the reasoning capabilities of language models.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2024)
Ref_id:b1 Title: Gpt-4 technical report Year: (2023)
Ref_id:b2 Title: On exact computation with an infinitely wide neural net Year: (2019)
Ref_id:b3 Title: Reflections after refereeing papers for nips Year: (1995)
Ref_id:b4 Title: Polynomial coefficients and distribution of the sum of discrete uniform variables Year: (2007)
Ref_id:b5 Title: Phase diagram of initial condensation for two-layer neural networks Year: (2024)
Ref_id:b6 Title: On the Global Convergence of Gradient Descent for Over-parameterized Models using Optimal Transport Year: (2018)
Ref_id:b7 Title: Faithful reasoning using large language models Year: (2022)
Ref_id:b8 Title: Selectioninference: Exploiting large language models for interpretable logical reasoning Year: (2022)
Ref_id:b9 Title: The neural data router: Adaptive control flow in transformers improves systematic generalization Year: (2021)
Ref_id:b10 Title: Ctl++: Evaluating generalization on never-seen compositional patterns of known functions, and compatibility of neural representations Year: (2022)
Ref_id:b11 Title: Faith and fate: Limits of transformers on compositionality Year: (2024)
Ref_id:b12 Title: A comparative analysis of optimization and generalization properties of two-layer neural network and random feature models under gradient descent dynamics Year: (2020)
Ref_id:b13 Title: Tinystories: How small can language models be and still speak coherent english? Year: (2023)
Ref_id:b14 Title: How does gpt obtain its ability? tracing emergent abilities of language models to their sources Year: (2022)
Ref_id:b15 Title: Delving deep into rectifiers: Surpassing human-level performance on imagenet classification Year: (2015)
Ref_id:b16 Title: Improving transformer optimization through better initialization Year: (2020)
Ref_id:b17 Title: Learning compositionally through attentive guidance Year: (2018)
Ref_id:b18 Title: Neural Tangent Kernel: Convergence and Generalization in Neural Networks Year: (2018)
Ref_id:b19 Title:  Year: (1998)
Ref_id:b20 Title: Break it down: Evidence for structural compositionality in neural networks Year: (2023)
Ref_id:b21 Title: Not all tokens are what you need for pretraining Year: (2024)
Ref_id:b22 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b23 Title: Transformers learn shortcuts to automata Year: (2022)
Ref_id:b24 Title: Understanding the difficulty of training transformers Year: (2020)
Ref_id:b25 Title: Phase diagram for two-layer relu neural networks at infinite-width limit Year: (2021)
Ref_id:b26 Title: The algebraic mind: Integrating connectionism and cognitive science Year: (2003)
Ref_id:b27 Title: A mean field view of the landscape of two-layer neural networks Year: (2018)
Ref_id:b28 Title: Compositional abilities emerge multiplicatively: Exploring diffusion models on a synthetic task Year: (2023)
Ref_id:b29 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b30 Title: How capable can a transformer become? a study on synthetic, interpretable tasks Year: (2023)
Ref_id:b31 Title: Parameters as interacting particles: long time convergence and asymptotic error scaling of neural networks Year: (2018)
Ref_id:b32 Title: Language models are greedy reasoners: A systematic formal analysis of chain-of-thought Year: (2023)
Ref_id:b33 Title: Mean field analysis of neural networks: A central limit theorem. Stochastic Processes and their Applications Year: (2020)
Ref_id:b34 Title: Neurocompositional computing: From the central paradox of cognition to a new generation of ai systems Year: (2022)
Ref_id:b35 Title: Beyond the imitation game: Quantifying and extrapolating the capabilities of language models Year: (2022)
Ref_id:b36 Title: Mimetic initialization of selfattention layers Year: (2023)
Ref_id:b37 Title: Deepnet: Scaling transformers to 1,000 layers Year: (2024)
Ref_id:b38 Title: Label words are anchors: An information flow perspective for understanding in-context learning Year: (2023)
Ref_id:b39 Title: Improving generalization and convergence by enhancing implicit regularization Year: (2024)
Ref_id:b40 Title: Understanding the expressive power and mechanisms of transformer for sequence modeling Year: (2024)
Ref_id:b41 Title: Towards understanding how transformer perform multistep reasoning with matching operation Year: (2024)
Ref_id:b42 Title: Emergent abilities of large language models Year: (2022)
Ref_id:b43 Title: Chain of thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b44 Title: Gradient dynamics of shallow univariate relu networks Year: (2019)
Ref_id:b45 Title: An overview of condensation phenomenon in deep learning Year: (2025)
Ref_id:b46 Title: Do visionlanguage pretrained models learn composable primitive concepts? arXiv preprint Year: (2022)
Ref_id:b47 Title: Improving deep transformer with depth-scaled initialization and merged attention Year: (2019)
Ref_id:b48 Title: Understanding deep learning requires rethinking generalization Year: (2016)
Ref_id:b49 Title: A type of generalization error induced by initialization in deep neural networks Year: (2019)
Ref_id:b50 Title: Linear stability hypothesis and rank stratification for nonlinear models Year: (2022)
Ref_id:b51 Title: Loss spike in training neural networks Year: (2023)
Ref_id:b52 Title: Implicit regularization of dropout Year: (2024)
Ref_id:b53 Title: Stochastic modified equations and dynamics of dropout algorithm Year: (2023)
Ref_id:b54 Title: Initialization is critical to whether transformers fit composite functions by reasoning or memorizing Year: (2024)
Ref_id:b55 Title: Anchor function: a type of benchmark functions for studying language models Year: (2024)
Ref_id:b56 Title: Complexity control facilitates reasoning-based compositional generalization in transformers Year: (2025)
Ref_id:b57 Title: Empirical phase diagram for three-layer neural networks with infinite width Year: (2022)
Ref_id:b58 Title: Gradinit: Learning to initialize neural networks for stable and efficient training Year: (2021)
