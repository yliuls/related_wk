Title: Retrieval-Augmented Perception: High-Resolution Image Perception Meets Visual RAG
Abstract: High-resolution (HR) image perception remains a key challenge in multimodal large language models (MLLMs). To drive progress beyond the limits of heuristic methods, this paper advances HR perception capabilities of MLLMs by harnessing cutting-edge long-context techniques such as retrieval-augmented generation (RAG). Towards this end, this paper presents the first study exploring the use of RAG to address HR perception challenges. Specifically, we propose Retrieval-Augmented Perception (RAP), a training-free framework that retrieves and fuses relevant image crops while preserving spatial context using the proposed Spatial-Awareness Layout. To accommodate different tasks, the proposed Retrieved-Exploration Search (RE-Search) dynamically selects the optimal number of crops based on model confidence and retrieval scores. Experimental results on HR benchmarks demonstrate the significant effectiveness of RAP, with LLaVA-v1.5-13B achieving a 43% improvement on V * Bench and 19% on HR-Bench. Code is available at https://github.com/DreamMr/RAP.

Section: Introduction
Multimodal large language models (MLLMs) have achieved remarkable progress in vision-language understanding, reasoning, and interaction, leveraging visual signals to process and interpret visual information (Yin et al., 2023). Current Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
this section cite: ['b44']

Section: High-Resolution image
Retrieval Layout MLLM Query: Where is the small stone cairn located relative to the waterfall? MLLMs (Liu et al., 2024a;Bai et al., 2023;Liu et al., 2024b;Wang et al., 2023;Abdin et al., 2024) typically process images at a fixed resolution (e.g., 448×448). While this design streamlines the computational pipeline, it introduces significant challenges, such as shape distortion and blurring when handling high-resolution (HR) images. These distortions notably impair the performance of MLLMs, especially in tasks that involve analysing real-world images with varying resolutions, such as visual grounding and optical character recognition that demand fine-grained visual details (Zhang et al., 2024a;Tian et al., 2022;2023;Wang et al., 2024).
In response to this dilemma, emerging research on enhancing the HR image perceptual capabilities of MLLMs has gained increasing attention. Existing approaches can be broadly categorised into three groups: (1) cropping-based methods (Chen et al., 2024c;Liu et al., 2024b;Li et al., 2024c), (2) HR visual encoder methods (Luo et al., 2024;Ge et al., 2024;Lu et al., 2024), and (3) search-based meth-ods (Wu & Xie, 2024;Wang et al., 2025;Shen et al., 2024). Despite notable progress, both cropping-based and HR visual encoder methods still require downsampling HR images to mitigate excessively long visual token sequences, resulting in substantial loss of fine-grained details. Although search-based methods avoid downsampling, they face several limitations. These methods follow a top-down search from high to low resolution; however, at the initial stage, models struggle to accurately perceive small objects, often resulting in erroneous search paths. Furthermore, searchbased approaches rely on hierarchical, layer-by-layer retrieval, preventing parallel processing and rendering them inefficient and cumbersome for deployment.
These limitations prompt our rethinking of the fundamental challenge in HR perception. Ideally, effective HR perception requires an MLLM with robust long-context capabilities-for instance, processing an 8K HR image with ViT-L/14 (Dosovitskiy et al., 2021) generates approximately ∼300K visual tokens. This raises the question of whether the key to HR perception lies in enhancing the long-context capacity of MLLMs, rather than relying on existing heuristic approaches, particularly in light of recent encouraging advancements in long-context techniques for general LLMs.
In particular, retrieval-augmented generation (RAG) has proven highly effective in recent long-context LLMs, by retrieving crucial fragments and reducing the impact of irrelevant information (Jin et al., 2024). Motivated by this, this paper poses a largely overlooked question: Is it possible to directly enhance the long-context capability of MLLMs using RAG, as in general LLMs, to overcome the limitations of existing HR perception methods?
However, exploring this research question presents significant challenges, as images, unlike text, are two-dimensional (excluding the channel dimension) and are characterised by width and height. As a pilot study, we begin by focusing on two key aspects: the layout of retrieved image crops and the impact of the number of retrieved crops on performance. This leads to the following specific challenges: 1) How should the retrieved image crops be organised? Furthermore, the number of retrieved key fragments plays a critical role in RAG performance (Jin et al., 2024), prompting our second research question: 2) How does the number of retrieved image crops influence the final performance? Building on insights from these two questions, we further pose a third research question: 3) How can RAG systems be designed to enhance MLLM perception of HR images?
To address the 1 st challenge, we conduct a series of experiments using the HR-Bench (Wang et al., 2025) to investigate the effects of different layout strategies. We evaluate state-of-the-art (SOTA) MLLMs (Liu et al., 2024a;b) across various layout configurations. Specifically, we compare three strategies: 1) arranging the retrieved image crops in their original order, 2) ordering them in descending order based on retrieval scores (Jin et al., 2024), and 3) preserving the relative positional relationships among the retrieved crops. Our empirical results suggest that maintaining the relative positional relationships of the image crops significantly enhances HR perception, particularly for tasks that depend on spatial relationships.
In response to the 2 nd question, this paper investigates the impact of the number of retrieved image crops. Our findings reveal that the optimal number of retrieved crops depends on the task type. For single-instance perception tasks, a small number of crops suffices for significant performance improvements, whereas too many crops degrade performance due to the high image resolution. In contrast, for crossinstance perception tasks, fewer crops result in information loss and reduced performance, while more crops help preserve essential details and minimise performance degradation. However, an excessive number of crops still harms performance due to challenges from overly high resolution.
In tackling the 3 rd question, we integrate the insights gained from the previous investigations to design a new framework, which we term Retrieval-Augmented Perception (RAP). As illustrated in Figure 1(a), RAP processes high-resolution images by retrieving image crops relevant to the query through VisRAG (Yu et al., 2024). We propose a simple yet efficient layout method, termed as Spatial-Awareness Layout, which preserves the original relative spatial relationships among the image crops. To determine the optimal number of retrieved image crops, we introduce a novel scheme termed as RE-Search (Retrieved-Exploration Search), which adaptively adjusts the number of crops based on the model's confidence in the sufficiency of the retrieved information.
In particular, VisRAG is first used to compute the similarity scores between each image crop and the query. We then retain the top K crops with the highest similarity scores, ensuring their relative spatial relationships are preserved through the Spatial-Awareness Layout. To determine the optimal K, we construct a RE-Tree, where each node represents a new image synthesized by retaining different proportions of the image crops. The search process within this tree is guided by both the retrieved similarity scores and the model's confidence in whether the image offers sufficient information to answer the query.
Our contribution is thereby the first investigation into using visual RAG to enhance HR image perception in MLLMs. This is accomplished by a novel RAP, a training-free framework that comprises Spatial-Awareness Layout to preserve the positions of image crops and RE-Search to adaptively select the optimal number of retained crops. Experiments demonstrate that RAP consistently delivers significant improvements, with an average accuracy increase of 24% on HR image benchmarks and even general MLLM tasks.
this section cite: ['b1', 'b36', 'b0', 'b31', 'b37', 'b25', 'b7', 'b24', 'b40', 'b38', 'b29', 'b6', 'b13', 'b13', 'b38', 'b13', 'b46']

Section: Related Work
MLLMs consist of a Visual Encoder (Dosovitskiy et al., 2021;Radford et al., 2021) for extracting visual features and a LLM (Touvron et al., 2023a;b) for decoding text, both initialized from pretrained models. A multimodal Connector (e.g., MLP) links the vision and language modalities. To align the resolution used during visual encoder pretraining (e.g., 336 × 336 in LLaVA), images are typically resized, which can distort and blur HR images. To address this, existing approaches fall into three categories: 1) croppingbased methods, 2) HR visual encoder methods, and 3) search-based methods.
Cropping-based methods. Representative cropping-based methods for HR MLLMs (Chen et al., 2024a;Zhang et al., 2024b;Liu et al., 2024c), such as LLaVA-v1.6 (Liu et al., 2024b) and LLaVA-ov (Li et al., 2024a), segment images into multiple image crops. Each image crop is independently encoded using ViT (Dosovitskiy et al., 2021) and subsequently concatenated for LLM processing.
this section cite: ['b6', 'b28', 'b6']

Section: HR Visual Encoder.
High-resolution image understanding can be enhanced by incorporating HR visual encoders without substantially increasing the number of visual tokens. For instance, Vary (Wei et al., 2023) and Deepseek-VL (Lu et al., 2024) adopt the SAM (Kirillov et al., 2023) to improve the performance of MLLMs on HR images. MiniGemini-HD (Li et al., 2024b), LLaVA-HR (Luo et al., 2024), and ConvLLaVA (Ge et al., 2024) utilize ConvNeXt (Liu et al., 2022), employing techniques such as cross-attention or adapter to extract visual features.
Search-based Methods. Search-based methods organize images into a tree structure to extract query-relevant regions through a top-down approach. DC 2 (Wang et al., 2025) leverages visual memory to store objects and coordinates, retrieving crops to generate text and reduce detail loss. Zoom Eye (Shen et al., 2024) employs a tree search algorithm to directly identify and extract relevant crops from HR images. Wu & Xie (2024) propose SEAL, a meta-architecture that actively reasons and retrieves essential visual information.
Multimodality RAG. Multimodal RAG tasks include matching images to text and retrieving text-image pairs to answer questions (Chang et al., 2022;Han et al., 2017;Xia et al., 2024a;b). Yu et al. (2024) propose Vision-based Retrieval-augmented Generation to effectively utilize and retain data in multimodal documents.
Existing methods enhance MLLMs' ability to perceive HR images, but processing extremely HR images (e.g., 8K) remains challenging. Inspired by RAG's success in handling long contexts for LLMs, this paper for the first time explores its use to improve MLLMs' HR image perception.
this section cite: ['b39', 'b24', 'b14', 'b25', 'b7', 'b22', 'b38', 'b29', 'b40', 'b2', 'b9', 'b46']

Section: Pilot Study
In this section, we conduct a systematic investigation into the challenges associated with employing RAG to enhance the perceptual capabilities of MLLMs, motivating the design of the proposed RAP framework in Sect. 4.
this section cite: []

Section: Preliminary
In this section, we introduce the pipeline for applying RAG to MLLMs for the perception of HR images. Given an HR image, we divide it into an image crop set, denoted as V = {v 1 , ..., v n }, where n is the number of image crops. Inspired by Yu et al. (2024), the query and image crops are independently encoded as text and images within the VLM, yielding a sequence of hidden states. Subsequently, the similarity scores between the query embedding and the image crop embeddings are computed. The similarity score s(q, V ) is calculated by the cosine similarity of the query and image crop embeddings:
s(q, V ) = (1 - q • V T ||q|| • ||V || ) • 1 2 .(1)
Finally, the top K image crops are selected based on the s(q, V ) to facilitate the MLLM's perception of HR images.
In the following sections, we systematically analyze the impact of retrieved image crop layouts and quantities on HR-Bench, which consists of HR-Bench 8K and HR-Bench 4K. HR-Bench 8K, with 8K-resolution images from DIV8K (Gu et al., 2019) and the Internet, includes Fine-grained Single-instance Perception (FSP) and Finegrained Cross-instance Perception (FCP) tasks. Cropping 8K images around relevant objects produces HR-Bench 4K.
this section cite: ['b46', 'b8']

Section: Impact of the Layout of Retrieved Image Crops
This subsection investigates the relationship between the layout of retrieved image crops and the performance of MLLMs in the RAG system.
Experimental setting. We compare three layout strategies: 1) Sort according to the retrieval scores in descending order; 2) After selecting the top K image crops, arrange them in the order in which the image crops appear; 3) Maintain the relative positional relationships of the image crops. We conduct experiments on HR-Bench using LLaVA-v1.6-7B.
Observations. As shown in Table 1, retrieving key image crops through RAG significantly improves performance on the FSP task but results in a noticeable performance drop on the FCP task. Furthermore, maintaining the relative positions between each image crop achieves a better performance balance between the FSP and FCP tasks.
Insights. Maintaining the relative positional relationships between retrieved image crops is essential, particularly for
Table 1. The effect of different layout strategies. While all three strategies improve fine-grained perception, only strategy 3) excels in FCP tasks by preserving positions, achieving superior performance compared to other strategies. HR-Bench 4K HR-Bench 8K FSP FCP Avg. FSP FCP Avg. Baseline 49.0 46.8 47.9 37.2 44.2 40.8 + 1) 74.8 38.3 56.5 56.8 26.8 41.8 + 2) 72.3 38.5 55.4 61.3 25.5 43.4 + 3) 74.0 41.5 57.8 59.5 30.0 44.8
tasks requiring spatial awareness.
this section cite: []

Section: Impact of the Number of Retrieved Image Crops
This subsection investigates the relationship between the number of retrieved image crops and the performance of MLLMs in HR image perception.
this section cite: []

Section: Experimental setting.
We analyze the relationship between performance (i.e., accuracy) and the number of the retrieved image crops, using the LLaVA-v1.5 and LLaVA-v1.6.
Observations. As shown in Figure 2, we visualize the relationship between the number of retrieved image crops (i.e., K) and performance. As K increases, more image crops are introduced, providing additional visual information that enhances performance on FCP tasks. However, this also raises the image resolution, increasing the likelihood of the model generating incorrect answers. Conversely, smaller K retains only essential visual information, improving performance on FSP tasks but sacrificing significant visual details, which causes a notable performance decline on FCP tasks.
Insights. Different types of tasks require different numbers of retrieved image crops K. For FSP tasks, smaller K improves results, but larger K reduces performance by increasing resolution. Conversely, for FCP tasks, larger K preserves visual information and outperforms smaller K.
this section cite: []

Section: Proposed Retrieval-Augmented Perception
4.1. Method Overview Driven by the aforementioned insights in Sect. 3, we propose a novel framework -Retrieval-Augmented Perception (RAP). The design principle of RAP is to retrieve key image crops to replace the original HR image, preserving essential visual information while reducing resolution to improve MLLM perception of HR images. To achieve this, we divide the image into various crops, calculate similarity scores (Eq. 1) with the query, and select the top K image crops to synthesize a new image V ′ . We design a Spatial-Awareness Layout algorithm to maintain the relative positional relation-LLaVA-v1.6-13B FSP Task FCP Task LLaVA-v1.5-7B LLaVA-v1.5-13B LLaVA-v1.6-7B ships between the image crops. To adaptively select K, we propose Retrieved-Exploration Search (RE-Search), which determines K based on the model's confidence in V ′ and its similarity to the query. The Spatial-Awareness Layout and RE-Search are presented in the subsequent sections.
this section cite: []

Section: Spatial-Awareness Layout
In Sect. 3.2, we find that maintaing the positional relationship between image crops is essential. Thus, we propose a simple and efficient method, termed Spatial-Awareness Layout. We denote M ∈ {0, 1} R×C as a binary matrix of size R × C, where R and C represent the number of rows and columns of image crops V , respectively. The M i,j = 1 indicates an image crop to be preserved and M i,j = 0 indicates the image crops to be removed. We seek to construct a compressed matrix M ′ by removing any row or column of M that is entirely zero. Formally, we define two index sets:
R ′ = {i|∃ j s.t.M i,j = 1}, C ′ = {j|∃ i s.t.M i,j = 1}. (2)
The compressed matrix M ′ ∈ {0, 1} Nr×Nc , with N r = |R ′ | and N c = |C ′ |, is then constructed according to: M ′ ĩ, j = M i,j , where i = R ′ [ ĩ] and j = C ′ [ j]. This guarantees that M ′ retains all rows and columns of M containing at least one entry equal to 1, effectively discarding rows and columns composed entirely of zeros. Moreover, an mapping function Φ : {0, ..., N r -1}×{0, ..., N c -1} → {0, ..., R-1} × {0, ..., C -1} is defined as Φ( ĩ, j) = (R ′ [ ĩ], C ′ [ j]), thereby enabling each coordinate ( ĩ, j) in the compressed matrix M ′ to be mapped back to its original position (i, j) in M . Finally, we initializes an blank image V ′ and iterates over the mapping Φ, where each pair ( ĩ, j) is mapped to (i, j). For each mapping, V [i][j] is assigned to the corresponding V ′ [ ĩ] of Spatial-Awareness Layout is shown in Algorithm 1.
this section cite: []

Section: Retrieved-Exploration Search
In Sect. 3.3, we find that different types of tasks significantly influence the choice of K. Here, we utilize a search algorithm to obtain the optimal K. For search algorithm, we consider two primary factors: Efficiency, ensuring high efficiency for optimal user experience, and Robustness, guaranteeing consistent results across multiple runs in image perception tasks. Existing tree-search methods (Wang et al., 2025) require visiting all nodes, leading to low efficiency.
With the development of O1, many recent works (Yao et al., 2024;Zhao et al.) employ Monte Carlo Tree Search (MCTS) to find the optimal reasoning path. However, MCTS relies on random sampling, resulting in a lower robustness. A * search algorithm uses a heuristic function to intelligently guide its exploration. This heuristic allows A * to prioritize promising paths, significantly accelerating the search process. Furthermore, A * explores the nodes in the same order and find the same optimal path, ensures high robustness. However, effectively defining the state representation and designing an appropriate heuristic function for A * is a non-trivial challenge.
Building upon the strengths of A * , we introduce Retrieved-Exploration Search (RE-Search). In the following parts, we will elucidate the RE-Tree, a novel structure that elegantly represents the search states within RE-Search, and the REward function, which serves as the guiding heuristic for this innovative approach.
RE-Tree Representation. Inspired by Wang et al. (2025); Shen et al. (2024), we model the HR image as a tree. Unlike existing search-based methods, we represent distinct nodes at the same layer by preserving different K image crops. This enables the model to perceive lower-resolution images from the begining, mitigating the risk of the MLLM converging to suboptimal solutions. We denote P = {p 1 , ..., p n } as the retention ratio. For instance, for the first child node n 1 , we retain the top N ′ × p 1 image crops. The N ′ represents the number of image crops for the current image. To obtain a complete image for calculating the REward function, we employ Spatial-Awareness Layout to assemble the individual image crops into a complete image V ′ .
this section cite: ['b38', 'b43', 'b51', 'b38', 'b29']

Section: REward Function.
A * search is a best-first search algorithm that prioritizes nodes with the lowest combined cost, calculated as the sum of the actual cost g(t s ) from the start node t 0 to t s and the estimated cost h(t s ) to the goal. In our RE-Search, the path from t 0 to t s is represented as the
Algorithm 1 Spatial-Awareness Layout function SpatialLayout(V, M ) R ′ ← {i ∃ j s.t.M i,j = 1} C ′ = {j ∃ i s.t.M i,j = 1} N r ← |R ′ |, N c ← |C ′ | Construct a binary matrixM ′ ∈ {0, 1} Nr×Nc for ĩ = 1 → N r -1 do for j = 0 → N c -1 do i ← R ′ [ ĩ], j ← C ′ [ j] M ′ ĩ, j ← M i,j end for end for Initialize a blank image V ′ for ĩ = 0 → N r -1 do for j = 0 → N c -1 do i ← R ′ [ ĩ], j ← C ′ [ j] if M ′ ĩ, j = 1 then V ′ [ ĩ, j] ← V [i, j] end if end for end for return V ′ end function
progression from the original HR image to the currently retained top-K image crops. We use the similarity score between these K image crops and the query as g(t s ):
g(t s ) = 1 n n i=1 s(q, v i ),(3)
where n represents the number of image crops, and v i represents the i-th image crops for current image V . Inspired by Shen et al. (2024), we use the model's confidence in whether the current image V can answer the given query as the cost from t s to the goal:
h(t s ) = 1 -P θ ("Yes"|p h (q), V ),(4)
where P θ represents the MLLM and p h (•) represents the prompt (e.g., "Question: {q}. Could you answer the question based on the available visual information? Answer Yes or No.") used to query the MLLM for calculating the confidence that the answer is "Yes". We utilize the model's confidence to estimate the cost from the current to the target node, analogous to the heuristic function in the A * algorithm. A lower h(•) indicates a higher likelihood of containing essential information, warranting prioritized exploration.
Since MLLM cannot accurately perceive the HR image at the beginning, the h(t s ) provided at shallow depths of the tree is unreliable. As the tree depth increases and the image resolution gradually decreases, the model becomes more confident in determining whether the current image can answer the query. Therefore, we assign a lower weight to h(t s ) at the beginning and gradually increase its weight as the tree depth grows. Mathematically, the cost function f (t) can be written as:
f (t s ) = (1 -w) • g(t s ) + w • h(t s ),(5)
w = (1 -b) • (1 - 1 d ) 2 + b,(6)
where b is a bias value, set here at 0.2 and d denotes the depth of the image tree.
this section cite: ['b29']

Section: Algorithmic Workflow
In this section, we introduce how to use our RAP to perceive HR image. Given a HR image I, we first divide the HR image into various image crops V , with the size of each image crop not exceeding the predefined resolution of the retriever's image encoder. Subsequently, we utilize VisRAG (Yu et al., 2024) to compute the cosine similarity between the query and image crops. We use RE-Search to search the optimal K image crops and using the Spatial-Awareness Layout to synthesize the image V f , which replaces the original HR image V as input to the MLLM. We denote c as the answering confidence which is calculated by Eq. 4. When c exceeds a predefined threshold τ , the search terminates. We set τ = 0.6 throughout the paper. The implementation of RAP is shown in Appendix A.
this section cite: ['b46']

Section: Experiments
In this section, we evaluate our RAP on HR benchmarks and a general MLLM benchmark. Further experimental results, including the influence of inference computation scale and the effect of the hyperparameter, are provided in Appendix B. Case studies are illustrated in Figures 8∼9 in the Appendix C.
this section cite: []

Section: Results on HR Benchmark
Benchmarks. We evaluate our RAP on two HR benchmarks: V * Bench and HR-Bench. V * Bench, derived from SA-1B (Kirillov et al., 2023), averages a resolution of 2246 × 1582. More details about HR-Bench can be found in Sect. 3.1.
Main Results. As shown in Table 2, compared to the baseline MLLM, the performance of nearly all models significantly improved with our RAP, demonstrating the modelagnostic trait of RAP. We find that our RAP can bring significant improvements in both FSP and FCP tasks. Our RAP brings a maximum of 21.0% and 21.7% accuracy improvement on HR-Bench 4K and HR-Bench 8K respectively. Additionally, for tasks requiring spatial reasoning capabilities, RAP demonstrates significant improvements compared to the baseline (e.g., +39.5% accuracy on V * Bench using LLaVA-v1.5-7B). The results show that our method has a clear advantage with HR images.
this section cite: ['b14']

Section: Results on General Multimodal Benchmark
Benchmark. We conduct additional evaluations of RAP using the MME-RealWorld (Zhang et al., 2024c), a manually curated benchmark designed for partical, real-world scenarios. This benchmark encompasses five primary categories and 43 sub-class tasks. Due to space constraints, we present results for 9 sub-tasks that exhibit notable performance variations with RAP.
Main Results. As shown in Table 3, RAP improves the performance of LLaVA-v1.5-13B on most sub-tasks, especially on MO/Orientation (+7.3%), AD/Intention (+6.0%), and OCR/license (+10.3%). However, we observe that tasks involving Diagram and Table types do not exhibit significant improvements and, in some cases, even performance degradation. We find that this due to the reliance of such data on the model's spatial awareness and reasoning capabilities, which are inherent limitations of current MLLMs.
this section cite: []

Section: Ablation Study
To better understand the role of each module in our RAP, we conduct ablation study on HR-Bench 8K using LLaVA-v1.5-7B. As shown in Table 4, we first use VisRAG to retrieve key image crops, replacing the original HR images, resulting in an average improvement of 4.5% accuracy compared to the baseline. However, we find a significant improvement in the FSP task, but there is a noticeable performance drop in the FCP task. By incorporating the Spatial-Awareness Layout, the relative positional relationships between image crops are preserved, leading to an improvement in accuracy on the FCP task compared to +VisRAG. Finally, we utilize RE-Search to determine the optimal K for different samples, resulting in significant improvements in both the FSP and FCP tasks, with an average improvement of 21.7% accuracy compared to the baseline.
this section cite: []

Section: Performance and Efficiency
Efficiency concerns regarding RAP may arise among researchers. To address this,
Table 5 presents a comparative analysis of throughput and accuracy against SOTA searchbased methods (e.g., DC 2 and Zoom Eye). RAP achieves superior efficiency and performance by directly computing the relevance between image crops and the query, eliminating the need for hierarchical image partitioning, thereby significantly accelerating the search process. More comparison results with search-based methods can be found in Appendix B.5.
this section cite: []

Section: Alternative to Model Logit Confidence
In RE-Search, we use the logit as a measure of the model's confidence that the current image V can answer the given query. However, for some closed-source models, it is not possible to access the model's output logits. To tackle Table 5. Evaluation of performance and inference efficiency. We analyze the correlation between throughput (samples per minute) and accuracy of LLaVA-v1.5-13B enhanced with our RAP, comparing it agains search-based methods on HR-Bench 4K.
this section cite: []

Section: Method
Throughput↑ Accuracy↑ DC 2 (Wang et al., 2025) 2.1 51.5 Zoom Eye (Shen et al., 2024) 3.3 58.0 RAP 4.2 60.1 this issue, we explore a simple alternative approach using generation-based confidence scores. Specifically, we design a scoring prompt that asks the model to evaluate whether the given image contains sufficient information to answer the question. The constructed scoring prompt can be found in Appendix A.4.
We conduct experiments on HR-Bench using LLaVA-ov-0.5B. The experimental results are shown in Table 6. Although the generation-based confidence score performs worse than the logit-based confidence score, it still shows a clear improvement over the baseline (achieved an average improvement of 7.4%), demonstrating that generation-based confidence scores through the model can also lead to significant performance gains.
To further investigate the extent to which the generationbased confidence score can replace the logit-based confidence score, we calculate the cosine similarity between the two scores for the same images. As shown in the Figure 4, a high cosine similarity score of 0.97 between the  two types of confidence scores, indicating a remarkable degree of alignment between their distributions. Interestingly, generation-based confidence scores tend to be consistently higher than their logit-based counterparts. Nonetheless, RAP utilizing generation-based confidence scores continues to deliver substantial improvements.
this section cite: ['b38', 'b29']

Section: Effect of Crop Size
To investigate the impact of crop size, we perform experiments on HR-Bench 8K using LLaVA-ov-0.5B. As shown in Table 7, we find that while variations in crop size result in relatively minor differences, all configurations of our RAP yield substantial performance gains over the baseline. To explore the impact of retrieval quality on RAP performance, we conduct experiments on HR-Bench 8K using LLaVA-ov-0.5B with SigLIP and VisRAG. Due to the limited text input length of SigLIP, we utilize MLLM to extract noun phrases from the query to compute relevance with image crops.
As shown in Table 8, we evaluate the retrieval quality of SigLIP (Zhai et al., 2023) and VisRAG (Yu et al., 2024), finding that VisRAG achieves superior retrieval performance. Notably, our RAP significantly enhances performance even with the relatively weaker retriever (SigLIP); for instance, it delivers a 9.1% overall enhancement on HR-Bench 8K. Reviewing the design principles of RAP: Retrieve image crops related to the query to reduce the image resolution input to the MLLM, thereby enabling the MLLM to perceive images more accurately. To explore the underlying mechanism of RAP, we perform experiments that help address the following questions:
1) Is it truly necessary to retrieve image crops relevant to the query? we compare randomly retained image crops with query-relevant image crops using LLaVA-v1.5-7B on HR-Bench 8K. As shown in Table 9, we randomly retained K = 4 and half of the image crops, comparing them with K image crops retrieved through VisRAG that are relevant to the query. The results indicate that retaining query-relevant image crops is necessary.
2) Can RAP accurately select an appropriate K? To an-  swer this question, we visualize the distribution of the number of retrieved image crops (K) for LLaVA-v1.5-7B w/ RAP on HR-Bench 8K. As shown in Figure 5(a), our RAP effectively reduces the number of image crops, resulting a +21.7% accuracy improvement. Additionally, for the FSP task, the K selected by our RAP is smaller, while for the FCP task, it is widely distributed across the range corresponding to larger K (e.g., K ≥ 60). The experiment results demonstrate that our RAP can provide accurate K, thereby effectively reducing the image resolution.
this section cite: ['b47', 'b46']

Section: Conclusion
In this paper, we propose a novel training-free framework Retrieval-Augmented Perception (RAP) to enhance HR image understanding in MLLMs. We empirically demonstrated the effectiveness and universality of RAP on several widely used MLLM benchmarks. From the results, we mainly conclude that: (1) Retrieving image crops relevant to the query can result in significant improvements; (2) Maintaining the relative spatial relationships of the retrieved image crops is essential, particularly for tasks that rely on positional information; (3) The number of image crops that need to be retained varies across different task types. In our future work, we will explore more token compression techniques to further enhance HR perception and efficiency.
this section cite: []

Section: References
Ref_id:b0 Title: Phi-3 technical report: A highly capable language model locally on your phone Year: (2024)
Ref_id:b1 Title: Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond Year: (2023)
Ref_id:b2 Title: Multihop and multimodal qa Year: (2022)
Ref_id:b3 Title: Multi-resolution zoom supercharges large visual-language model Year: (2024)
Ref_id:b4 Title: Are we on the right way for evaluating large vision-language models? arXiv preprint, 2024b Year: ()
Ref_id:b5 Title: How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites Year: (2024)
Ref_id:b6 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b7 Title: Convllava: Hierarchical backbones as visual encoder for large multimodal models Year: (2024)
Ref_id:b8 Title: Div8k: Diverse 8k resolution image dataset Year: (2019)
Ref_id:b9 Title: Automatic spatially-aware fashion concept discovery Year: (2017)
Ref_id:b10 Title: Ai2d-rst: a multimodal corpus of 1000 primary school science diagrams Year: (2021)
Ref_id:b11 Title: Cogvlm2: Visual language models for image and video understanding Year: (2024)
Ref_id:b12 Title: Gpt-4o system card Year: (2024)
Ref_id:b13 Title: Long-context llms meet rag: Overcoming challenges for long inputs in rag Year: (2024)
Ref_id:b14 Title: Segment anything. In ICCV Year: (2023)
Ref_id:b15 Title: Llavaonevision: Easy visual task transfer Year: (2024)
Ref_id:b16 Title: Mining the potential of multi-modality vision language models Year: (2024)
Ref_id:b17 Title: Monkey: Image resolution and text label are important things for large multi-modal models Year: ()
Ref_id:b18 Title: CVPR Year: (2024)
Ref_id:b19 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b20 Title: Llava-next: Improved reasoning, ocr, and world knowledge, 2024b Year: ()
Ref_id:b21 Title: Infimmhd: A leap forward in high-resolution multimodal understanding Year: (2024)
Ref_id:b22 Title: A convnet for the 2020s Year: (2022)
Ref_id:b23 Title: Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution Year: (2024)
Ref_id:b24 Title: Deepseek-vl: Towards real-world vision-language understanding Year: (2024)
Ref_id:b25 Title: Feast your eyes: Mixture-of-resolution adaptation for multimodal large language models Year: (2024)
Ref_id:b26 Title: A benchmark for question answering about charts with visual and logical reasoning Year: (2022)
Ref_id:b27 Title: Docvqa: A dataset for vqa on document images Year: ()
Ref_id:b28 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b29 Title: Enhancing multimodal llms with human-like zooming capabilities through tree-based image exploration Year: (2024)
Ref_id:b30 Title: Towards vqa models that can read Year: (2019)
Ref_id:b31 Title: Emotion-aware multimodal pre-training for image-grounded emotional response generation Year: (2022)
Ref_id:b32 Title: A multi-view metalearning approach for multi-modal response generation Year: (2023)
Ref_id:b33 Title: Cambrian-1: A fully open, vision-centric exploration of multimodal llms Year: (2024)
Ref_id:b34 Title: Llama: Open and efficient foundation language models Year: (2023)
Ref_id:b35 Title: Llama 2: Open foundation and finetuned chat models Year: (2023)
Ref_id:b36 Title: Cogvlm: Visual expert for pretrained language models Year: (2023)
Ref_id:b37 Title: Wisdom: Improving multimodal sentiment analysis by fusing contextual world knowledge Year: (2024)
Ref_id:b38 Title: Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models Year: (2025)
Ref_id:b39 Title: Vary: Scaling up the vision vocabulary for large vision-language models Year: (2023)
Ref_id:b40 Title: Guided visual search as a core mechanism in multimodal llms Year: (2024)
Ref_id:b41 Title: Mmed-rag: Versatile multimodal rag system for medical vision language models Year: (2024)
Ref_id:b42 Title: Rule: Reliable multimodal rag for factuality in medical vision language models Year: (2024)
Ref_id:b43 Title: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search Year: (2024)
Ref_id:b44 Title: A survey on multimodal large language models Year: (2023)
Ref_id:b45 Title: Open foundation models by 01 Year: (2024)
Ref_id:b46 Title: Visrag: Visionbased retrieval-augmented generation on multi-modality documents Year: (2024)
Ref_id:b47 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b48 Title: Llava-uhd v2: an mllm integrating high-resolution feature pyramid via hierarchical window transformer Year: (2024)
Ref_id:b49 Title: Beyond llava-hd: Diving into highresolution large multimodal models Year: (2024)
Ref_id:b50 Title: Mmerealworld: Could your multimodal llm challenge highresolution real Year: (2024)
Ref_id:b51 Title: Towards open reasoning models for open-ended solutions, 2024a Year: ()
