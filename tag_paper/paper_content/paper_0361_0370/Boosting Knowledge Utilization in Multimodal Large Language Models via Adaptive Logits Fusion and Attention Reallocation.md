Title: Boosting Knowledge Utilization in Multimodal Large Language Models via Adaptive Logits Fusion and Attention Reallocation
Abstract: Despite their recent progress, Multimodal Large Language Models (MLLMs) often struggle in knowledge-intensive tasks due to the limited and outdated parametric knowledge acquired during training. Multimodal Retrieval Augmented Generation addresses this issue by retrieving contextual knowledge from external databases, thereby enhancing MLLMs with expanded knowledge sources. However, existing MLLMs often fail to fully leverage the retrieved contextual knowledge for response generation. We examine representative MLLMs and identify two major causes, namely, attention bias toward different tokens and knowledge conflicts between parametric and contextual knowledge. To this end, we design Adaptive Logits Fusion and Attention Reallocation (ALFAR), a training-free and plugand-play approach that improves MLLM responses by maximizing the utility of the retrieved knowledge. Specifically, ALFAR tackles the challenges from two perspectives. First, it alleviates attention bias by adaptively shifting attention from visual tokens to relevant context tokens according to query-context relevance. Second, it decouples and weights parametric and contextual knowledge at output logits, mitigating conflicts between the two types of knowledge. As a plug-and-play method, ALFAR achieves superior performance across diverse datasets without requiring additional training or external tools. Extensive experiments over multiple MLLMs and benchmarks show that ALFAR consistently outperforms the state-of-the-art by large margins. Our code and data are available at https://github.com/Lackel/ALFAR.

Section: Introduction
Building upon powerful Large Language Models (LLMs) [1,2,3,4,5,6,7,8], Multimodal Large Language Models (MLLMs) [9,10,11,12,13,14,15,16,17] have achieved impressive performance over a wide range of vision-centric tasks such as image captioning [18,19,20], visual question answering [21,22], etc. Nevertheless, MLLMs often struggle to handle knowledge-intensive visionlanguage tasks [23,24], primarily due to the limited and outdated parametric knowledge acquired during training [25,26]. Multimodal Retrieval Augmented Generation (MRAG) [27,28,29], a prevalent approach that attempts to resolve this issue, retrieves contextual knowledge from external data to empower MLLMs for accurate response generation. However, the way of exploiting the contextual knowledge remains under-explored, undermining the effectiveness of MRAG.
We examined representative MLLMs with MRAG and found that while MRAG can improve MLLM performance when high-quality contextual knowledge is retrieved (as shown in Fig. 1), MLLMs often fail to make full use of the retrieved knowledge, even when ground-truth knowledge is available. We identify two primary causes, namely, attention bias among visual and context tokens and conflicts between MLLMs' parametric knowledge and retrieved contextual knowledge. For the attention bias, MLLMs tend to allocate more attention to image tokens over context tokens, especially in shallow layers that are critical for knowledge extraction and exchange [30]. Since images often do not provide sufficient information for knowledge-intensive questions [23,31], the attention bias hinders the effective utilization of contextual knowledge and leads to inaccurate MLLM responses. In addition, MLLMs allocate attention uniformly across context tokens without prioritization, which dilutes the contributions of query-relevant knowledge and tends to introduce inaccurate MLLM responses.
65.47 52.69 38.03 37.20 71.13 58.20 43.10 41.83 30.00 35.00 40.00 45.00 50.00 55.00 60.00 65.00 70.00 75.00 Ground-Truth Knowledge 1st Retrieved Knowledge 2nd Retrieved Knowledge 3rd Retrieved Knowledge
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b22', 'b30']

Section: VQA Accuracy

this section cite: []

Section: Contextual Knowledge for Inference
Vanilla LLaVA LLaVA + MRAG LLaVA + ALFAR
this section cite: []

Section: 39.15
Figure 1: VQA accuracy of LLaVA-1.5 [14] on the multi-choice InfoSeek dataset [32] with respect to the quality of contextual knowledge. ALFAR fully exploits the retrieved knowledge, consistently improving MRAG performance regardless of knowledge quality.
Knowledge conflicts typically arise from the discrepancy between contextual and parametric knowledge. We observe that MLLMs tend to rely excessively on their parametric knowledge even when accurate contextual knowledge is present, leading to under-utilization of contextual knowledge and counterfactual responses. Such a phenomenon is well aligned with observations in previous LLM studies [33,34] and findings in psychology research [35,36], both underscoring a clear preference toward intrinsic instead of retrieved knowledge. On the other end, the preference for the parametric knowledge does help when the contextual knowledge is unreliable [28,37,38]. This can be observed in Fig. 1, where low-quality contextual knowledge significantly degrades performance. Therefore, striking a balance between parametric and contextual knowledge while leveraging their complementary strengths is critical for generating accurate responses.
In this work, we propose Adaptive Logits Fusion and Attention Reallocation (ALFAR), a trainingfree and plug-and-play approach that enables effective utilization of MRAG-retrieved contextual knowledge for accurate MLLM responses. ALFAR addresses attention bias and knowledge conflicts by dynamically adjusting attention allocation and balancing the parametric and contextual knowledge, respectively. Specifically, ALFAR adaptively shifts attention from image tokens to relevant context tokens based on retrieval scores and query-context relevance, enabling MLLMs to focus on more pertinent information. In addition, ALFAR decouples parametric and contextual knowledge at output logits and weights them according to the attention distribution, enabling a balanced and synergistic integration of the two types of knowledge. Extensive experiments across multiple representative MLLMs and benchmarks demonstrate ALFAR's superior and broad applicability without involving additional training or external tools.
The contributions of this work can be summarized in three major aspects. First, we dive deeply into knowledge utilization in MLLMs, identifying attention bias and knowledge conflicts as two key factors that impede the effective utilization of the retrieved knowledge. These findings provide valuable insights for advancing knowledge utilization in MLLMs. Second, we design ALFAR, a trainingfree and plug-and-play approach that reallocates attention and balances parametric and contextual knowledge effectively. Third, Extensive experiments over multiple generative and discriminative benchmarks validate ALFAR's effectiveness and versatility, demonstrating its superior performance and broad applicability across various multimodal tasks.
2 Related Work
this section cite: ['b13', 'b31', 'b32', 'b33', 'b34', 'b35', 'b27', 'b36', 'b37']

Section: Multimodal Large Language Models
The rapid advancements in Large Language Models (LLMs) [1,2,3,4,5,6,7,8,39] have greatly propelled the development of Multimodal Large Language Models (MLLMs) [9,10,11,13,14,15,16,40]. To align visual and textual modalities, prior studies explore different approaches such as visual encoders with linear projectors [14,41,40], Q-former [18,10], and Perceivers [42], which transform image patches into visual tokens that are compatible with LLMs. Most MLLMs conduct training in two stages, namely, pre-training for feature alignment and instruction-based fine-tuning [14,43,10], enabling impressive performance across diverse multimodal tasks [44,45,46,47]. Despite these advancements, MLLMs often face challenges in knowledge-intensive tasks [23,31], due to the limitations of their parametric knowledge acquired during training.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b38', 'b8', 'b9', 'b10', 'b12', 'b13', 'b14', 'b15', 'b39', 'b13', 'b40', 'b39', 'b17', 'b9', 'b41', 'b13', 'b42', 'b9', 'b43', 'b44', 'b45', 'b46', 'b22', 'b30']

Section: Multimodal Retrieval Augmented Generation
Inspired by the concept of Retrieval Augmented Generation (RAG) for LLMs [48,49], Multimodal Retrieval Augmented Generation (MRAG) has been widely explored for enhancing MLLMs with more comprehensive and up-to-date knowledge [27,28,29,50]. MRAG retrieves relevant knowledge from a multimodal database and incorporates the retrieved knowledge as the context of the input. For instance, Wiki-LLaVA [27] broadens the knowledge scope of LLaVA [14] by incorporating the retrieved Wikipedia articles in training. EchoSight [29] leverages a fine-tuned Q-Former [18] to filter retrieved knowledge and enhance retrieval recall. ReflectiVA [28] and MR 2 AG [50] introduce a trainable reflection mechanism to assess the necessity of retrieval and the relevance of retrieved knowledge. In the LLM domain, several training-free methods have been proposed to better utilize the retrieved knowledge to enhance generation quality. For instance, CAD [26] employs contrastive decoding [51] to increase the faithfulness of generation. Moreover, AdaCAD [52], Entropy [53], and COIECD [54] extend CAD [26] by introducing JS divergence, entropy, and information constraints, respectively. Despite the improved faithfulness toward the retrieved context, these methods struggle to balance parametric and contextual knowledge [55], resulting in sub-optimal performance when the contextual knowledge is noisy.
this section cite: ['b47', 'b48', 'b26', 'b27', 'b28', 'b49', 'b26', 'b13', 'b28', 'b17', 'b27', 'b49', 'b25', 'b50', 'b51', 'b52', 'b53', 'b25', 'b54']

Section: Preliminary and Motivation

this section cite: []

Section: Multimodal Retrieval Augmented Generation
Given a textual query q and a query image I, an MLLM M θ parameterized by θ is expected to generate a reliable answer y. To enrich MLLMs with external knowledge, MRAG employs a multimodal retriever R ϕ to fetch relevant knowledge from a multimodal knowledge base C = {( I i , c i )} M i=1 , where I i and c i represent an image and its corresponding textual knowledge, respectively. The retriever R ϕ measures the similarity between the query pair (q, I) and a multimodal knowledge pair ( I, c) based on the cosine similarity between their image embeddings:
α = R ϕ (I) • R ϕ ( I) ||R ϕ (I)|| • ||R ϕ ( I)||(1)
The textual knowledge c with the highest retrieval similarity α is selected as the input context for MLLMs. Consequently, the output distributions of the MLLM with MRAG at the time step t are: p(y t ) ∼ softmax(M θ (y t |q, I, c, y <t ))
(2) where y <t represents the sequence of generated tokens before the time step t.
this section cite: []

Section: Self-attention Mechanism in MLLMs
0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1 2 3 4 5 6 7 8 9 10
this section cite: []

Section: Attention Distribution
Layer Index
this section cite: []

Section: Image tokens Context tokens
Figure 2: Proportions of attention weights that are assigned to image and context tokens at different shallow layers of LLaVA-1.5 [14].
MLLMs generate responses auto-regressively using Transformer blocks [56]. Specifically, the input image, query, and context tokens are concatenated and projected into three distinct vectors: the query vector Q, the key vector K, and the value vector V, through three linear layers, W q , W k , and W v . The self-attention mechanism computes the relevance of each token to other tokens as follows:
A = Q • K ⊤ √ d + M(3)
where A ∈ R n×n is the attention weight matrix, M is a causal mask, d is the feature dimension, and n is the number of input tokens. The output O can then be calculated by:
O = softmax(A) • V(4)
The attention weight matrix A reflects the importance of different input tokens in generating new tokens. This property makes it a valuable tool to analyze the contribution of different types of tokens to the generated responses.
this section cite: ['b13', 'b55']

Section: Attention Bias in MRAG
To generate the response token y n+1 , MLLMs perform self-attention over all input tokens which correspond to the n-th row of the attention weight matrix A. We analyze the contributions of the input image and context using an importance score, defined as the total attention weights assigned to these tokens. For the input image, the importance score at layer i is calculated by:
S i (I) = j∈I A i nj 1 .
Similarly, for the input context, the importance score at layer i is determined by: S i (c) = j∈c A i nj . As illustrated in Fig. 2, MLLMs tend to allocate more attention to image tokens than context tokens, particularly in shallow layers that are pivotal for extracting and exchanging information from distinct tokens [30]. This issue affects the effective utilization of contextual knowledge since images often do not capture sufficient information for knowledge-intensive questions [23,31]. Moreover, MLLMs assign attention uniformly across different parts of the context without highlighting queryrelevant segments. Such indiscriminate distribution increases the distraction of irrelevant knowledge, ultimately leading to inaccurate or misleading responses for MLLMs. After knowledge retrieval, MLLMs integrate the retrieved contextual knowledge with their internal parametric knowledge to generate responses. However, similar to LLMs [33,34,52], MLLMs often encounter knowledge conflicts due to the discrepancy between the two types of knowledge, affecting the effectiveness of the model in various practical tasks. Worse still, MLLMs tend to prioritize their parametric knowledge even when perfect contextual knowledge is provided, leading to under-utilization of contextual knowledge and factually inconsistent answers. By assuming the accessibility of the ground-truth contextual knowledge, we evaluate the conflict rate of the two types of knowledge and the resultant performance degradation on the multi-choice InfoSeek [32] and ViQuAE [24,32] datasets with LLaVA-1.5 [14]. As shown in Tab. 1, around half of the samples exhibit knowledge conflicts, which lead to an up to 30% performance drop, highlighting the necessity of mitigating such conflicts to enhance MLLM performance on knowledge-intensive tasks.
this section cite: ['b29', 'b22', 'b30', 'b32', 'b33', 'b51', 'b31', 'b23', 'b31', 'b13']

Section: Knowledge Conflicts in MRAG

this section cite: []

Section: Method
The proposed framework consists of two branches for effective handling of parametric and contextual knowledge as illustrated in Fig. 4. Within the contextual branch, we design an attention reallocation mechanism that tackles the attention bias and improves the utilization of contextual knowledge by adaptively adjusting model attention toward relevant context tokens based on query-context relevance (Sec. 4.1). In addition, the network fuses the parametric and contextual knowledge adaptively in the output logits, mitigating knowledge conflicts under the guidance of the model attention that dynamically captures the relative importance of the two types of knowledge (Sec. 4.2).
this section cite: []

Section: Attention Reallocation
As analyzed in Sec. 3.3, the attention bias results from two major factors, namely, attention preference toward image tokens and uniform attention to context tokens. We address the attention preference
The Eiffel Tower is in Paris, which was designed by Gustave Eiffel and completed constrcuted in 1889. Database MRAG image context T o k e n i z e r V i s i o n E n c o d e r When was the building in the image constructed? query P r o j e c t o r image token query token context token Tokenization 1889 Decoding M L P Large Language Model M L P Attention Matrix Regular Attn. Blk. softmax Attn. Mat. softmax Modified Attn. Blk.
this section cite: []

Section: Attention Reallocation Adaptive Logits Fusion

this section cite: []

Section: Figure 4: The overview of Adaptive Logits Fusion and Attention Reallocation (ALFAR).
by adaptively adjusting model attention as illustrated in Fig. 3, based on the retrieval similarity α in Eq. 1 that reflects the reliability of the retrieved context. Specifically, less attention is allocated to image tokens if the context is more reliable with a high retrieval similarity. The attention reallocation can be formulated as follows:
Âni = (1 -β) • A ni , s.t. i ∈ S I(5)
where β = k • α is the scaled retrieval similarity with a scaling factor k. Â is the modified attention weight matrix, n is the number of all input tokens and S I is the index set of image tokens. A ni corresponds to the attention weight in the n-th row and i-th column of A.
☉ element-wise multiplication sum up sum up reallocation coefficient original weight reallocated weight Attention Weight Matrix Attention Reallocation Image attention weight Query attention weight Context attention weight Scaled retrieval similarity Logits fusion weights Scaled query-context relevance Causal mask Decrease attention Increase attention In addition, we introduce a query-context relevance score to mitigate the uniform attention to all context tokens and allow MLLMs to focus more on query-relevant context. We derive the relevance score from attention weights assigned to query tokens by j-th context token as follows:
ωj = k∈Sq A jk l∈Sc k∈Sq A lk , s.t. j ∈ Sc (6
)
where S q and S c are index sets of query and context tokens, respectively. With the relevance scores, we adaptively increase MLLMs' attention to context tokens:
Ânj = (1 + γ j ) • A nj , s.t. j ∈ S c (7
)
where γ j = k • ω j is the scaled query-context relevance with a scaling factor k. After attention reallocation, we apply softmax to redistribute the attention as in Eq. 4 to compute the output hidden states. This repeats auto-regressively for each subsequent token prediction.
this section cite: []

Section: Adaptive Knowledge Fusion
As analyzed in Sec. 3.4, parametric knowledge could hinder the utilization of contextual knowledge, leading to inaccurate responses. However, parametric knowledge brings benefits when the retrieved context is unreliable. Therefore, striking a balance between parametric and contextual knowledge based on their reliability is essential for generating accurate and reliable responses for MLLMs. Nevertheless, parametric knowledge is implicitly embedded and the two types of knowledge are entangled during inference, making it hard to explicitly represent and utilize them separately. We address this issue by disentangling the two types of knowledge and fuse them at output logits. Specifically, we represent parametric knowledge at step t by using the output logits that have only query and image as inputs: Similarly, we represent contextual knowledge by using the output logits that have context as additional inputs and perform attention reallocation to better utilize the context:
logit p = M θ (y t |q, I, y <t )(8)
logit c = Mθ (y t |q, I, c, y <t ) (9
)
where Mθ is the MLLM with attention reallocation. The reliability of the parametric and contextual knowledge can thus be measured by the attention weights that are assigned to the image and context tokens capturing the correlation among tokens [59,60,61]: Finally, based on the adaptive weights, the two types of knowledge are fused dynamically at each decoding step t:
λ t v = i∈S I A ti , λ t c = j∈Sc A tj (10
)
p(yt) ∼ softmax (1 + λ t c λ t v ) logitc -(1 - λ t v λ t c ) logitp (11
)
5 Experiment
this section cite: ['b58', 'b59', 'b60']

Section: Experimental Settings
Datasets. We conduct experiments over three types of knowledge-intensive datasets: (1) Freeform generative datasets including Human [23], a high-quality info-seeking dataset curated and verified by experts, and INFOSEEK wiki [23] which encompasses diverse entities from Wikidata. For INFOSEEK wiki , we adopt its Validation set for evaluations to be aligned with prior studies [27,28].
(2) Multi-choice discriminative datasets including Infoseek [23,32] and ViQuAE [24,32] which are both multi-choice knowledge-intensive datasets that are collected for assessing cross-modality knowledge conflicts as described in [32]. (3) Knowledge-based datasets including OK-VQA [64], AOK-VQA [65] and Encyclopedic VQA (E-VQA) [31], which are widely adopted for evaluations of tasks that require commonsense knowledge. Additional details about the datasets and knowledge bases are listed in the Appendix A5.
MLLM baselines and SOTA methods. We perform evaluations by using four representative MLLMs as backbones: LLaVA-1.5 (7B and 13B) [14], InstructBLIP (7B and 13B) [10], Shikra (7B) [40], MiniGPT-4 (7B) [16], LLaVA-Next (7B) [62], and Qwen2.5-VL (3B) [63]. For benchmarking, we select several SOTA training-free decoding methods that aim to mitigate knowledge conflicts in LLMs: Contrastive Decoding (CD) [51], Adaptive Context-Aware Decoding (AdaCAD) [52], Entropy-based decoding (Entropy) [53], Context-Aware Decoding (CAD) [26] and COntextual Information-Entropy Constraint Decoding (COIECD) [54]. In addition, we also benchmark with two representative hallucination mitigation methods, including Visual Contrastive Decoding (VCD) [58] and Assembly of Global and Local Attention (AGLA) [57].
Implementation details. For knowledge retrieval, we employ the vision encoder of CLIP-ViT-L/14-336 [66] as the retriever and append the first retrieved knowledge to the prompt as context. The scaled factor k is set to 0.4 to avoid excessive adjustment. For the knowledge base, we use Wikipedia dumps provided by [23] and select items with associated images for retrieval. Multinomial sampling serves as the decoding strategy. We denote MLLM inference with retrieved knowledge as Regular and without retrieved knowledge as Parametric. We follow prior studies [26,53,58] and adopt adaptive plausibility constraints [51] for fair comparisons. All experiments are conducted on four NVIDIA RTX 3090 GPUs. All compared methods are reproduced by us according to their released codes or original papers.
this section cite: ['b22', 'b22', 'b26', 'b27', 'b22', 'b31', 'b23', 'b31', 'b31', 'b63', 'b64', 'b30', 'b13', 'b9', 'b39', 'b15', 'b61', 'b62', 'b50', 'b51', 'b52', 'b25', 'b53', 'b57', 'b56', 'b65', 'b22', 'b25', 'b52', 'b57', 'b50']

Section: Experimental Results
Experiments on free-form datasets. Tab. 2 shows experimental results of four representative MLLMs [14,10,40,16] over two free-form generative knowledge-intensive datasets [23]. We can see that the proposed ALFAR consistently outperforms the Regular decoding strategy by substantial margins (averaged around 2.5% in overall accuracy) across all MLLMs and datasets. Additionally, ALFAR surpasses state-of-the-art decoding methods as well, demonstrating its effectiveness in the better utilization of contextual knowledge.
Experiments on multi-choice datasets. Tab. 3 presents experimental results of six MLLMs [14,10,40,16,62,63] over two multi-choice discriminative datasets [32,24]. Notably, ALFAR achieves an average improvement of 6.6% over Regular decoding and consistently surpasses state-of-the-art decoding strategies by substantial margins, underscoring its effectiveness in diverse tasks. Moreover, we observe that LLaVA-1.5 [14] demonstrates a stronger instruction-following capability compared to other models, enabling it to produce more correctly formatted outputs.
this section cite: ['b13', 'b9', 'b39', 'b15', 'b22', 'b13', 'b9', 'b39', 'b15', 'b61', 'b62', 'b31', 'b23', 'b13']

Section: Experiments on knowledge-based datasets.
In addition to entity knowledge-based datasets [32,24], we conduct experiments on commonsense knowledge-based datasets, OK-VQA [64], AOK-VQA [65] and Encyclopedic VQA (E-VQA) [31] with LLaVA-1.5 [14]. As shown in Tab. 4, ALFAR surpasses Regular decoding by 15.2% and consistently outperforms state-of-the-art decoding strategies, underscoring its effectiveness in addressing a broader range of knowledge-intensive tasks. We conduct ablation studies on both multi-choice and commonsense knowledge-based datasets [32,65] to assess the effectiveness of each design in the proposed ALFAR model with LLaVA-1.5 [14].
this section cite: ['b31', 'b23', 'b63', 'b64', 'b30', 'b13', 'b31', 'b64', 'b13']

Section: Discussion

this section cite: []

Section: References
Ref_id:b0 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b1 Title: Ul2: Unifying language learning paradigms Year: (2022)
Ref_id:b2 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b3 Title: Language models are few-shot learners Year: (2020)
Ref_id:b4 Title: Scaling language modeling with pathways Year: (2022)
Ref_id:b5 Title: Stanford alpaca: An instruction-following llama model Year: (2023)
Ref_id:b6 Title: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality Year: (2023-03)
Ref_id:b7 Title: Qwen technical report Year: (2023)
Ref_id:b8 Title: Qwen-vl: A frontier large vision-language model with versatile abilities Year: (2023)
Ref_id:b9 Title: Instructblip: Towards general-purpose vision-language models with instruction tuning Year: (2023)
Ref_id:b10 Title: Multimodal-gpt: A vision and language model for dialogue with humans Year: (2023)
Ref_id:b11 Title: Empowering multimodal llms with external tools: A comprehensive survey Year: (2025)
Ref_id:b12 Title: Otter: A multi-modal model with in-context instruction tuning Year: (2023)
Ref_id:b13 Title: Visual instruction tuning Year: (2023)
Ref_id:b14 Title: mplug-owl: Modularization empowers large language models with multimodality Year: (2023)
Ref_id:b15 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2023)
Ref_id:b16 Title: Mm-llms: Recent advances in multimodal large language models Year: (2024)
Ref_id:b17 Title: Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b18 Title: A comprehensive literature review on image captioning methods and metrics based on deep learning technique Year: (2024)
Ref_id:b19 Title: A review of multimodal explainable artificial intelligence: Past, present and future Year: (2024)
Ref_id:b20 Title: Knowledge acquisition disentanglement for knowledgebased visual question answering with large language models Year: (2024)
Ref_id:b21 Title: Visual question answering instruction: Unlocking multimodal large language model to domain-specific visual multitasks Year: (2024)
Ref_id:b22 Title: Can pre-trained vision and language models answer visual information-seeking questions? Year: (2023)
Ref_id:b23 Title: Viquae, a dataset for knowledge-based visual question answering about named entities Year: (2022)
Ref_id:b24 Title: Retrieval augmented language model pre-training Year: (2020)
Ref_id:b25 Title: Trusting your evidence: Hallucinate less with context-aware decoding Year: (2024)
Ref_id:b26 Title: Wiki-llava: Hierarchical retrieval-augmented generation for multimodal llms Year: (2024)
Ref_id:b27 Title: Augmenting multimodal llms with self-reflective tokens for knowledge-based visual question answering Year: (2024)
Ref_id:b28 Title: Echosight: Advancing visual-language models with wiki knowledge Year: (2024)
Ref_id:b29 Title: Attend first, consolidate later: On the importance of attention in different llm layers Year: (2024)
Ref_id:b30 Title: Encyclopedic vqa: Visual questions about detailed properties of fine-grained categories Year: (2023)
Ref_id:b31 Title: Unraveling cross-modality knowledge conflicts in large vision-language models Year: (2024)
Ref_id:b32 Title: Deciphering the interplay of parametric and nonparametric memory in retrieval-augmented language models Year: (2024)
Ref_id:b33 Title: Astute rag: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models Year: (2024)
Ref_id:b34 Title: Unskilled and unaware of it: how difficulties in recognizing one's own incompetence lead to inflated self-assessments Year: (1999)
Ref_id:b35 Title: The confidence-competence gap in large language models: A cognitive study Year: (2023)
Ref_id:b36 Title: Preflmr: Scaling up fine-grained late-interaction multi-modal retrievers Year: (2024)
Ref_id:b37 Title: Uniir: Training and benchmarking universal multimodal information retrievers Year: (2025)
Ref_id:b38 Title: Chatgpt outperforms crowd-workers for text-annotation tasks Year: (2023)
Ref_id:b39 Title: Shikra: Unleashing multimodal llm's referential dialogue magic Year: (2023)
Ref_id:b40 Title: Improved baselines with visual instruction tuning Year: (2023)
Ref_id:b41 Title: What matters when building vision-language models? arXiv preprint Year: (2024)
Ref_id:b42 Title: Blip: Bootstrapping languageimage pre-training for unified vision-language understanding and generation Year: (2022)
Ref_id:b43 Title: Seeing clearly by layer two: Enhancing attention heads to alleviate hallucination in lvlms Year: (2024)
Ref_id:b44 Title: Dreamsalon: A staged diffusion framework for preserving identity-context in editable face generation Year: (2024)
Ref_id:b45 Title: From redundancy to relevance: Enhancing explainability in multimodal large language models. arXiv e-prints Year: (2024)
Ref_id:b46 Title: Mmrel: A relation understanding dataset and benchmark in the mllm era Year: (2024)
Ref_id:b47 Title: Retrieval-augmented generation for large language models: A survey Year: (2023)
Ref_id:b48 Title: Retrieval-augmented generation for knowledge-intensive nlp tasks Year: (2020)
Ref_id:b49 Title: mr 2 ag: Multimodal retrieval-reflection-augmented generation for knowledge-based vqa Year: (2024)
Ref_id:b50 Title: Contrastive decoding: Open-ended text generation as optimization Year: (2023)
Ref_id:b51 Title: Elias Stengel-Eskin, and Mohit Bansal. Adacad: Adaptively decoding to balance conflicts between contextual and parametric knowledge Year: (2024)
Ref_id:b52 Title: Entropy-based decoding for retrieval-augmented large language models Year: (2024)
Ref_id:b53 Title: Discerning and resolving knowledge conflicts through adaptive decoding with contextual informationentropy constraint Year: (2024)
Ref_id:b54 Title: Dynamic attention-guided context decoding for mitigating context faithfulness hallucinations in large language models Year: (2025)
Ref_id:b55 Title: Attention is all you need Year: (2017)
Ref_id:b56 Title: Agla: Mitigating object hallucinations in large vision-language models with assembly of global and local attention Year: (2024)
Ref_id:b57 Title: Mitigating object hallucinations in large vision-language models through visual contrastive decoding Year: (2023)
Ref_id:b58 Title: Accelerating multimodel large language models by searching optimal vision token reduction Year: (2024)
Ref_id:b59 Title: Transformer interpretability beyond attention visualization Year: (2021)
Ref_id:b60 Title: Llm knows what you are looking for before generation Year: (2024)
Ref_id:b61 Title: Improved baselines with visual instruction tuning Year: (2023)
Ref_id:b62 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b63 Title: Ok-vqa: A visual question answering benchmark requiring external knowledge Year: (2019)
Ref_id:b64 Title: A-okvqa: A benchmark for visual question answering using world knowledge Year: (2022)
Ref_id:b65 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b66 Title: The curious case of neural text degeneration Year: (2019)
Ref_id:b67 Title: Hierarchical neural story generation Year: (2018)
Ref_id:b68 Title: Some remarks on greedy algorithms Year: (1996)
Ref_id:b69 Title: A learning algorithm for boltzmann machines Year: (1985)
Ref_id:b70 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b71 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b72 Title: The inaturalist species classification and detection dataset Year: (2018)
Ref_id:b73 Title: Intervening anchor token: Decoding strategy in alleviating hallucinations for MLLMs Year: (2025)
Ref_id:b74 Title: Lost in the middle: How language models use long contexts Year: (2023)
