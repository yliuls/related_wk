Title: Breaking the Batch Barrier (B3) of Contrastive Learning via Smart Batch Mining
Abstract: Contrastive learning (CL) is a prevalent technique for training embedding models, which pulls semantically similar examples (positives) closer in the representation space while pushing dissimilar ones (negatives) further apart. A key source of negatives are "in-batch" examples, i.e., positives from other examples in the batch. Effectiveness of such models is hence strongly influenced by the size and quality of training batches. In this work, we propose Breaking the Batch Barrier (B3), a novel batch construction strategy designed to curate high-quality batches for CL. Our approach begins by using a pretrained teacher embedding model to rank all examples in the dataset, from which a sparse similarity graph is constructed. A community detection algorithm is then applied to this graph to identify clusters of examples that serve as strong negatives for one another. The clusters are then used to construct batches that are rich in in-batch negatives. Empirical results on the MMEB multimodal embedding benchmark (36 tasks) demonstrate that our method sets a new state of the art, outperforming previous best methods by +1.3 and +2.9 points at the 7B and 2B model scales, respectively. Notably, models trained with B3 surpass existing state-of-the-art results even with a batch size as small as 64, which is 4-16× smaller than that required by other methods. Moreover, experiments show that B3 generalizes well across domains and tasks, maintaining strong performance even when trained with considerably weaker teachers.

Section: Introduction
Contrastive Learning (CL) has emerged as the dominant approach for training embedding models [5,2,8]. It typically operates on data in the form of (query, positive) pairs [29], where the objective is to minimize the distance between the query's representation and that of its positive counterpart. To foster more discriminative representations, CL also incorporates negative examples -instances which, for a given query, are not its designated positive -by increasing their representational distance from the query. In practice, other examples within the same batch serve as readily available in-batch negatives [9]. Furthermore, to significantly enhance the learning signal, "hard negatives" are often explicitly mined and integrated into the training process [5,11]. These hard negatives are particularly challenging examples that share superficial features with the query or its positive, making them easily confusable with the true positive, yet are semantically distinct.
Recent text-only embedding models, such as NV-Embed [11] and SFR-Embedding [18], have achieved state-of-the-art results through the use of hard negative mining from the training dataset. These methods typically employ a pretrained teacher model to rank potential negatives relative to a given query, subsequently selecting a small set (e.g., up to ten) of top-ranked candidates as hard negatives. While incorporating multiple hard negatives can enhance model performance, it also substantially increases training cost and duration (e.g. using ten hard negatives is approximately 4x training time compared to using one hard negative). This performance-cost trade-off, while manageable for text-only models, becomes computationally prohibitive in multimodal contexts. In such settings, positive examples often include high-resolution images, significantly amplifying the processing overhead associated with each additional hard negative [30]. Moreover, contrastive learning relies on large datasets, which are already hard to fully use in multimodal training [14,24]. Adding many hard negatives makes this even more demanding, limiting scalability.
Consequently, many recent efforts in multimodal embedding learning have largely avoided explicit hard negative mining across the entire training dataset. For instance, VLM2Vec [8] repurposes data from diverse tasks -such as retrieval, classification, and VQA -for contrastive training but omits dedicated hard negative sampling. MegaPairs [38] and mmE5 [1] focus on generating synthetic contrastive pairs, prioritizing data augmentation over mining challenging negatives. Other methods, including LLaVE [10] and UniLM [6], do incorporate negative sampling strategies, but restrict this to resampling examples within the current batch as negatives. While these approaches have proven effective to varying degrees, they often do not exploit the potentially stronger negative signals residing outside the immediate batch, thereby overlooking a rich source of contrastive supervision. This work addresses the aforementioned gap by constructing batches from the full dataset where examples within each batch serve as strong negatives for one another. This approach aims to eliminate the need for separately mined hard negatives, significantly reducing computational cost. Our proposed batch mining method, B3, leverages teacher-based ranking but revolutionizes batch construction. Unlike prior methods that add individually mined hard negatives to IID sampled batches, B3 uses graph-based community detection to cluster examples which are mutually strong negatives of each other. Batches are then strategically formed by sampling from these cohesive communities ensuring strong in-batch negatives. This facilitates strong training signals without the need for extra negatives.
Our contributions are summarized as follows:
• We introduce B3, a novel batch mining strategy that sets a new state-of-the-art on the MMEB benchmark, surpassing strong baselines by 1.3 and 2.9 points at the 7B and 2B scales resp.
• We provide theoretical justification for the design of B3.
• We conduct a comprehensive ablation study to evaluate the individual components of B3 and highlight the significant impact of the batch mining module.
• B3 achieves sota performance at 2B scale despite training at a much smaller batch size of 64.
• B3 beats random batching baseline with strong hard negatives using only half the compute.
• B3 works effectively with weak teachers and generalizes across domains.
this section cite: ['b4', 'b1', 'b7', 'b28', 'b8', 'b4', 'b10', 'b10', 'b17', 'b29', 'b13', 'b23', 'b7', 'b37', 'b0', 'b9', 'b5']

Section: Related Work
This section reviews prior work on improving negative sampling for contrastive learning, strategies for batch mining, and recent advances in multimodal embedding methods.
this section cite: []

Section: Mining Better Negatives
E5 [31] leveraged GPT-4 to construct contrastive learning (CL) datasets with diverse-length (anchor, positive, negative) tuples, which were then used to fine-tune the Mistral 7B model. SumCSE [28] used compositional transformations to create negatives. Gecko [12] employed a large LLM to generate queries from existing passages and subsequently relabeled positives and negatives for these queries. In contrast, our approach treats positives from other examples as potential negatives for a given query.
Orthogonal to this, several recent studies, including SFR-Embedding [18,17], NV-Embed [11] and NV-Retriever [20], incorporate diverse annotated datasets -including clustering and classification -in addition to retrieval data, achieving notable gains on MTEB. A common technique in these methods involves using a pretrained teacher model to rank all positive examples from other queries in the dataset relative to a given query, selecting the top-ranked instances as hard negatives. Drawing inspiration from these studies, our approach also employs a teacher model to assess relationships between examples. However, instead of directly selecting and adding hard negatives, we leverage this ranking information to construct high-quality batches where the constituent examples inherently serve as strong negatives for one another during contrastive training.
this section cite: ['b30', 'b27', 'b11', 'b17', 'b16', 'b10', 'b19']

Section: Batch Mining for Contrastive Learning
Several studies have explored improving contrastive models by optimizing the construction of training batches. NGAME [4] forms batches by clustering data points and merging clusters, targeting extreme multi-label classification tasks. GCBS [25] formulates batch mining as a matrix bandwidth minimization problem on adjacency graphs derived from intermediate model checkpoints. BatchSampler [33] employs random walks over adjacency graphs to sample informative batches. However, these methods do not account for the impact of false negatives or the detrimental effects of excessive number of hard negatives on the batch construction process, particularly in large-scale datasets. More recently, Morris and Rush [21] employed clustering to construct batches containing stronger negatives; however, their approach is limited to text-only benchmarks and smaller-scale models.
this section cite: ['b3', 'b24', 'b32', 'b20']

Section: Improving Multimodal Embeddings
CLIP Based: CLIP [24] trained separate encoders for images and captions using 400M (image, caption) pairs. DreamLip [37] built on this by introducing additional losses to align subcaption embeddings with image patch embeddings. MagicLens [36] utilized naturally occurring image pairs and generated text describing their differences, using these as contrastive instructions for training multimodal embeddings. However, these LIP-style models employed separate encoders for image and text. In contrast, recent studies showed that vision-language models (VLMs) with early fusion of image and text features achieved better performance on multimodal embedding tasks.
VLM-based methods: E5-V [7] employs EOL-style prompts (e.g., "Text/Image means in one word:") on text-only data to train a VLM for embeddings. VLM2Vec [8] is trained on pairwise data curated from diverse tasks-including retrieval, classification, visual question answering (VQA), and grounding-but does not incorporate hard negatives during training. MegaPairs [38], similar to MagicLens, leverages a vision-language model to generate questions linking semantically related image pairs. LLaVE [10] introduces a weighting mechanism over in-batch negatives, assigning higher weights to negatives with larger logits relative to a query, thereby prioritizing them during contrastive training. mmE5 [1] synthesizes multimodal triplets (query, positive, negative) involving diverse image-text combinations for training, drawing inspiration from E5. UniLM [6] enhances embedding quality by distilling knowledge from text-only models and resampling in-batch negatives as hard negatives. Despite their innovations, these MLLM-based approaches do not fully exploit the potential of strong negatives that can be systematically mined from the training dataset.
this section cite: ['b23', 'b36', 'b35', 'b6', 'b7', 'b37', 'b9', 'b0', 'b5']

Section: Methodology
Our proposed methodology comprises three key components: (1) a novel batch selection mechanism, which forms the core of our approach; (2) an optional negative sampling strategy tailored to these curated batches; and (3) enhanced prompting techniques to guide the embedding model.
this section cite: []

Section: Batch Selection
In contrastive learning, performance is critically influenced by both batch size and the quality of negatives within each batch. Addressing this, our work introduces a novel batch mining strategy that leverages teacher-model rankings to strategically compose batches. The aim is to ensure that the examples within each batch inherently act as strong negatives for each other, thus providing rich contrastive signals efficiently. The following subsections will elaborate on the derivation and components of this methodology.
this section cite: []

Section: Batch Mining Algorithm
Our algorithm starts by using a trained teacher embedding model to rank all positives in the training set w.r.t a given query, x i . Let R ∈ N N ×N be the rank matrix, where each row R i = R i,: is a permutation of the set {1, 2, . . . , N }. For each row i, the entries represent the ordered ranks of the N examples, sorted from highest to lowest relevance (or similarity) of the positive y i with respect to query x i according to the teacher model. The highest scoring targets for each query when used as in-batch negatives adversely affect performance due to the presence of False Negatives (targets which are semantically equivalent/closer to golden target for the query and cannot be used as a negative). To filter out these false negatives, we use a simple rank based thresholding as proposed in NV-Retriever [20].We exclude the top p ranks and only use the next m ranks to sample in the batch mining procedure.
Our final algorithm uses only the specified columns, i.e., S = R[:, p : p + m]. This submatrix represents the adjacency structure of a sparse directed graph. In constructing the final undirected graph S, we retain only the bidirectional edges. As can be seen, the edges in the graph denote preference for same batch (strong in-batch negatives in our case). We now apply METIS community algorithm [16] to identify clusters of size K from this graph. METIS is a minimum cut algorithm -maximizing the number of edges within communities and minimizing the edges across across communities. Given the smaller value of m, METIS is fairly fast and approximately only requires O(n) runtime.
Given the clusters of size K, we collect |B|/K random such clusters to construct a batch. Note that contrastive learning needs high batch size |B| [2,5]. It is also simple enough to derive the same from Eq. 2, that higher batch sizes minimize this difference. The algorithm is depicted in Fig. 1.
this section cite: ['b19', 'b15', 'b1', 'b4']

Section: Theoretical Justification of the Proposed Algorithm
Taking inspiration from Sachidananda et al. [25], we use the global contrastive loss as the reference to derive a theoretical justification for our algorithm. The global InfoNCE loss is defined below, with the summation taken over all examples in the dataset. The temperature term is omitted for brevity.
L Global = N i=1 -log exp(x T i y i ) N j=1 exp(x T i y j )(1)
where x i is a query and y i is the corresponding labeled target, N is the size of the dataset. Our goal is to build a batch B that can approximate the global loss on the batch level InfoNCE loss as follows.
min B L Global -L B(2)
where
L B = N i=1 -log exp(x T i y i ) j∈Bi exp(x T i y j )(3)
Theorem 1. The difference between global and batch loss terms is upper-bounded as follows:
L Global -L B ≤ N i=1 log N K H K i H K Bi,i(4)
where H K i and H K Bi,i denote the sum of the top K exponent terms in the denominator for the global and batch loss components, respectively, for each query x i .
The proof of the theorem is presented in §E. Since the bound holds for all values of K, we henceforth let K represent the number of top in-batch elements that encompass all strong negatives for query x i within the batch B i . The two mains things that we note from this bound: (1) A higher value of H K Bi,i makes the bound tighter; and (2) A higher value of K makes the bound tighter.
this section cite: ['b24']

Section: Theorem 2 ([26]).
Loss of an embedding model on downstream tasks is bounded as follows:
L sup ( f ) ≤ αL ̸ = un (f ) + βs(f ) + η Gen M (5
)
where s(f ) is the a measure of intra-class covariance and β is a term that increases with more strong negatives for a query.
Refer to [26] for a formal proof. Both intra-class covariance and β increase with a higher number of strong negatives, i.e. K. This theorem calls for avoiding picking very large values of K. The claim is also empirically corroborated by SFR-Embedding [19], where using high number of hard negatives resulted in inferior performance. Given the opposing trends for K observed in both theorems, it is most appropriate to select K empirically. Note that K is limited by the batch size |B|.
The bound in Eq: 4 should be optimized for all stages during training. Effectively, given a K, H K Bi,i needs to be simultaneously maximized ∀i,
B i i.e. max B1,..Bi..,B N/|B| N i=1 H K Bi,i(6)
Algorithm proposed in §3.1.1 does optimize for this. Each cluster is set to be size K. Note that the number of edges within each cluster are maximized. This maximizes the H K Bi,i term as edges denote preference for strong negatives. Our algorithm jointly optimizes batch composition to ensure that groups of mutually preferred negative examples are co-located within the same batch. It is important to note that one could, in principle, increase H K Bi,i by directly adding hard negatives while retaining random batches; However, this approach would be extremely computationally expensive.
this section cite: ['b25', 'b18']

Section: Unified Hard-Negative Mining (Optional)
While the mining algorithm in §3.1 is expected to build batches whose examples are strong in-batch negatives of each other, it might turn out that some examples are just not possible to put together with its preferred rank examples due to the collective optimization. Additional mined hard negatives might be important for such cases. In this work, we introduce an enhancement over the approach of randomly sampling h additional negatives per query from S, as proposed in SFR-Embedding.
Note that each additional hard-negative is contrasted against all queries in the batch. Hence, rather than naively using S[i, :] to sample hard negatives for query x i , we come up with an aggregated strategy to account for preferences of all queries in the batch mined in §3.1.1. A unified probability distribution Pr(j) is created such that Pr(j) ∝ Count(S B , j). Count(S B , j) is the number of times item j appeared in the rows of S corresponding to the items/queries in batch B. This unified distribution is used to sample h hard negatives per examples for all examples in the batch.
this section cite: []

Section: Improved Representation Prompts
Recent work on multi-modal embeddings has utilized instructions to encode queries, but has typically represented positive examples directly, without using instruction prompts [8,10]. This mismatch introduces inconsistencies in how different types of positives are represented. For example, a positive such as "aeroplane" in ImageNet classification requires a fundamentally different representation compared to a detailed image caption from MSCOCO, such as "Skateboarder performing a stunt in mid air." Incorporating instruction prompts for positives is also expected to help decouple stylistically diverse tasks during training. Hence, we design positive target prompts for all datasets in §I.
this section cite: ['b7', 'b9']

Section: Overall Methodology
We introduce two variants of our methodology: B3 and B3++. B3 serves as the core approach, integrating the components from §3.1 and §3.3, and delivers strong performance across benchmarks. B3++ builds on B3 by additionally incorporating hard negatives as described in §3.2, offering further gains when computational resources allow. While B3++ provides enhanced performance, B3 remains highly effective and is well-suited for resource-constrained settings.
this section cite: []

Section: Experiments
Training Set We train our models on the MMEB [8] training set. We train for 2000 steps (~2 epochs) with a batch size of 1024 unless specified.
this section cite: ['b7']

Section: Evaluation Set
We evaluate our methods on the MMEB [8] benchmark. MMEB benchmark contains 36 distinct tasks spanning four diverse categories -Retrieval (12), Classification (10), Visual Question Answering (10), Grounding (4). It contains 20 In-Domain and 16 Out-of-Domain tasks. Evaluation metric for all datasets is accuracy. Average accuracy of all datasets is reported.
this section cite: ['b7']

Section: Methods Compared:
We primarily evaluate the proposed methods-B3++ and B3-across various batch sizes. As a baseline, we also include Random Batches, which employs random batch selection. Additionally, we compare all methods with publicly accessible training methodology prior to the submission deadline.
Implementation: We test our methodology using two different VLMs -Qwen2-VL-7B-Instruct, InternVL3-8B with both 7B and 2B variants. We use m = 100 following SFR-Embedding and NV-Retriever. For p, we tuned this value on heldout portions of the train set. We used p = 30 for retrieval and grounding tasks and p = 70 for VQA tasks. For classification tasks, we just filter out the golden label from the rank list. We use 5 hard-negatives h = 5 for B3++ and no hard negatives in B3. All models in this work are trained using LoRA with a rank of 8. Unless mentioned, models are trained for 2k steps with peak learning rate of 1e-4 and warmup of 10%. Temperature used was 0.02.
this section cite: []

Section: Teacher Models:
To effectively capture the gains from our batch mining, we use teacher models -with the same scale, trained on the same data, without hard negatives. In our analysis, we used VLM2Vec (Qwen2-2B) to train 2B student models, and VLM2Vec (Qwen2-7B) to train 7B student models. The batch mining was performed at a task level for each of the 20 MMEB training datasets.
this section cite: []

Section: Main Results
Table 1 presents the performance of our models. At both the 2B and 7B model scales, our proposed methodology outperforms existing approaches. Specifically, B3++ (Qwen2-2B) surpasses the next best model by a substantial margin of 2.9 points, while B3++ (Qwen2-7B) achieves a notable improvement of 1.3 points, averaged across 36 tasks. While other models have significant data and modeling level differences, a more natural baseline for comparison is our teacher model, VLMVec. B3++ beats VLM2Vec by 6.2 points (Qwen2-7B) and 8.8 points (Qwen2-2B). 47.8 10.9 52.3 53.3 39.3 40.2 39.7 UniIR (BLIP FF ) [32] 42.1 15.0 60.1 62.2 44.7 40.4 42.8 UniIR (CLIP SF ) [32] 44. 3 16.2 61.8 65.3 47.1 41.7 44.7 MagicLens [36] 38.8 8.3 35.4 26.0 31.0 23.7 27.8 ∼2B VLM Models (Trained on MMEB) VLM2Vec (Qwen2-2B) [8] 65.4 59.0 49.4 73.4 0.0 0.0 59.3 UniME (Phi-3.5-V) [6] 64.5 54.8 55.9 81.8 68.2 52.7 64.2 LLaVE (Aquila-VL-2B) [10] 65.2 62.1 60.2 84.9 69.4 59.8 65.2 B3 (InternVL3-2B) (Ours) 69.0 62.6 64.0 86.9 73.5 60.8 67.8 B3++ (Qwen2-2B) (Ours) 70.9 67.0 61.2 79.9 72.1 63.1 68.1 (+2.9)
this section cite: ['b31', 'b31', 'b2']

Section: >7B VLM Models (Trained on MMEB)
VLM2Vec (Qwen2-7B) [8] 69.9 62.6 57.8 81.7 72.2 57.8 65.8 MMRet (Llava-Next-7B) [38] 69.9 56 57.4 83.6 68 59.1 64.1 mmE5 (Llama-3.2-11B) [1] 70.9 67.6 62.8 89.7 72.3 66.7 69.8 LLaVE (Llava-OV-7B) [10] 70.9 65.7 65.4 91.9 75.0 64.4 70.3 UniME (LLaVA-OneVision-7B) [6] 70.5 66.8 66.6 90.9 74.6 65. These results demonstrate the effectiveness of our approach across model sizes. The performance improvements are consistent across both in-domain and out-of-domain tasks. B3++ performs exceptionally well on retrieval, which is one of the most important applications of embedding models. The other VLM-based approaches evaluated rely primarily on either synthetic data generation or modeling innovations. In contrast, our method, B3, is a batch mining strategy and is therefore complementary to these techniques, making it amenable to integration with them.
this section cite: ['b7', 'b37', 'b0', 'b9', 'b5']

Section: Effect of Batch Size |B|
The core strength of B3 lies in its effective batch mining strategy. To assess the quality of the mined batches, we compare the performance of B3 and Random Batches across a range of batch sizes, starting from 32. The results, presented in Fig. 2, correspond to models trained for two epochs. At smaller batch sizes, B3 outperforms Random Batches by a substantial margin, achieving improvements of over 14 points. Even at a larger batch size of 1024, the gains remain notable, 85.5 95.9 62.8 77.6 98.1 98.0 exceeding 2.5 points. These results indicate that B3 consistently enhances performance across all batch sizes, with particularly pronounced benefits at smaller scales.
For additional context, we include the performance of the current 2B state-of-the-art model, LLaVE, trained with its default configuration, as a horizontal reference line in the plot. Remarkably, B3 surpasses LLaVE, the current 2B state-of-the-art model, even at a batch size as small as 64, further underscoring the effectiveness of its batch mining strategy. Enabling effective training with smaller batches facilitates model development on limited hardware resources and allows scaling to substantially larger models in high-capacity environments.
this section cite: []

Section: Dissecting B3++
Table 2 shows the results for individual components of the proposed B3++ methodology. B3 exhibits performance only slightly below that of B3++. Both our models outperform Random Batches baselines. B3 consuming the exact same compute as Random Batches beats it by 2.5 points. B3 beats the Random Batches + (w/ 5 hn from S) which samples negatives from matrix S, by 0.7 points despite using half the compute. B3 is both effective and efficient.
this section cite: []

Section: Short and Long Caption Retrieval
Following UniME [6], we perform zero-shot evaluation of B3++ on short (Flickr [23], COCO [15]) and long (Urban1k [35]) image caption retrieval. As shown in Table 3, B3++ (Qwen2-2B) surpasses UniME (7B), and B3++ (Qwen2-7B) achieves the best overall performance. Consistent with the substantial retrieval gains shown in Table 1, B3++ outperforms all other baselines.
this section cite: ['b5', 'b22', 'b14', 'b34']

Section: Ablations and Analysis

this section cite: []

Section: Effect of K
As discussed in §3.1.2, excessively large values of K may be suboptimal. In Table 1, we selected the value of K using a held-out subset of the training data. We now provide empirical evidence to support this choice on the test set. As shown in the ablation results, increasing K initially improves performance; however, beyond a certain threshold, very high values of K lead to a decline in performance. More details on this in §F. We examine the impact of limiting the maximum input resolution, using Qwen2-2B as the backbone. Images exceeding the specified resolution are down-sampled accordingly. Given a patch size of 28 × 28, Qwen2-2B produces approximately 600 tokens at a resolution of 700 and 1200 tokens at a resolution of 1000. Both experiments are conducted with a batch size of |B| = 1024. Increasing the number of tokens per image consistently leads to improved performance.
this section cite: []

Section: Strength of Teacher
In our main results, we employed a teacher model of the same scale as the student-trained on the same dataset and without hard negatives i.e. VLM2Vec, to isolate the contribution of our batch mining approach. In this ablation, we replace the teacher with other weaker and stronger models. In B3, the teacher's sole role is to spot potential strong negatives and group them in the same batch. We simply take ranks from the teacher to build batches with strong in-batch negatives-and even with as few as 8 such negatives (in Table 4), performance remains solid. Critically, the teacher's task is limited: it only needs to pick out some strong negatives, not all of them.
When trained with a weaker teacher such as CLIP-whose performance on MMEB is markedly lower than B3 continues to deliver competitive outcomes in Table 5. These results indicate that B3's effectiveness is not contingent on a highly capable teacher. Conversely, when trained with a stronger teacher, VLM2Vec(7B), the performance gains are minimal. We hypothesize that this is because the VLM2Vec(2B) teacher already provides sufficiently strong negatives, leaving little additional benefit from the larger VLM2Vec(7B) model.
this section cite: []

Section: Comparison with other Batch Selection techniques, Hyperparameters
In this subsection, we compare B3 with alternative batch selection strategies and analyze the impact of its key hyper-parameters. For baseline comparison, we consider Grit Batch Mining (GBM), based on GritLM [22], which selects random targets from the same task as a batch selection mechanism. We also evaluate variants of B3 by varying its two primary hyper-parameters, p and m. As shown in Fig. 3, all configurations of B3 outperform GBM. The results, averaged over 36 datasets, are statistically meaningful. Smaller values of p can introduce false negatives, leading to a performance drop. This is evident in Fig. 3 especially at smaller batch sizes.
this section cite: ['b21']

Section: Generalizability to other domains?
We report the performance of B3, trained on the same NLI data as SimCSE (both with and without annotated hard negatives (HN)), evaluated on the STS benchmark. The corresponding SimCSE models (with/without HN) serve as the teacher models. All models (teacher and student) in the following table are 300M Roberta-Large. B3 demonstrates notable improvements over SimCSE on the STS benchmark, validating our batch mining approach for text-only tasks. Note that even starting with HN in the teacher, we can still see improvements. In future work, we plan to extend this to larger benchmarks like MMEB. The entire B3 methodology operates as an offline preprocessing step over the training dataset. The teacher model's scoring of all training examples is fully parallelizable, and generating the rank list requires only O(n log n) time. Subsequently, applying the METIS algorithm on the sparse graph S incurs a linear O(n) runtime. The resulting mined batches can be efficiently stored on disk in a structured format. Integrating B3 into contrastive training pipelines requires minimal modifications-training simply involves sampling from the preprocessed batches. Looking ahead, we aim to extend the B3 framework to both text-only scenarios and broader multimodal settings.
this section cite: []

Section: Conclusion
We propose B3, an effective batch mining technique that leverages the entire training dataset to form batches composed of mutually strong negatives, to improve contrastive learning. B3++, a variant of B3 using hard-negatives achieves state-of-the-art results on the MMEB embedding benchmark comprising 36 diverse embedding tasks. B3 is complementary with existing methods centered on data synthesis and model architecture, and can be seamlessly integrated on top of them to further enhance performance. Through extensive experiments, we demonstrate that B3 consistently outperforms existing batch mining strategies across a wide range of batch sizes. Notably, B3++ surpasses the current 2B state-of-the-art model even with a batch size of just 64. Furthermore, B3 outperforms a random batch baseline augmented with five hard negatives, despite not using any hard negatives and requiring only half the training time. B3++ also outperforms other methods on image caption retrieval datasets, further showcasing its versatility and strength.
this section cite: []

Section: References
Ref_id:b0 Title: mme5: Improving multimodal multilingual embeddings via high-quality synthetic data Year: (2025)
Ref_id:b1 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b2 Title: Reproducible scaling laws for contrastive language-image learning Year: (2023)
Ref_id:b3 Title: Ngame: Negative mining-aware mini-batching for extreme classification Year: (2023)
Ref_id:b4 Title: Simple contrastive learning of sentence embeddings Year: (2021)
Ref_id:b5 Title: Breaking the modality barrier: Universal embedding learning with multimodal llms Year: (2025)
Ref_id:b6 Title: Deqing Wang, and Fuzhen Zhuang. E5-v: Universal embeddings with multimodal large language models Year: (2024)
Ref_id:b7 Title: Vlm2vec: Training vision-language models for massive multimodal embedding tasks Year: (2024)
Ref_id:b8 Title: Dense passage retrieval for open-domain question answering Year: (2020)
Ref_id:b9 Title: Llave: Large language and vision embedding models with hardness-weighted contrastive learning Year: (2025)
Ref_id:b10 Title: Nv-embed: Improved techniques for training llms as generalist embedding models Year: (2024)
Ref_id:b11 Title: Versatile text embeddings distilled from large language models Year: (2024)
Ref_id:b12 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b13 Title: Towards general text embeddings with multi-stage contrastive learning Year: (2023)
Ref_id:b14 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b15 Title: An empirical comparison of the summarization power of graph clustering methods Year: (2015)
Ref_id:b16 Title: Sfr-embedding-2: Advanced text embedding with multi-stage training Year: (2024)
Ref_id:b17 Title: Sfr-embedding-mistral: Enhance text retrieval with transfer learning Year: (2024)
Ref_id:b18 Title: Sfrembedding-mistral: enhance text retrieval with transfer learning Year: (2024)
Ref_id:b19 Title: Nv-retriever: Improving text embedding models with effective hard-negative mining Year: (2024)
Ref_id:b20 Title:  Year: (2024)
Ref_id:b21 Title: Generative representational instruction tuning Year: (2024)
Ref_id:b22 Title: Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models Year: (2015)
Ref_id:b23 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b24 Title: Global selection of contrastive batches via optimization on sample permutations Year: (2023)
Ref_id:b25 Title: A theoretical analysis of contrastive unsupervised representation learning Year: (2019)
Ref_id:b26 Title: Eva-clip-18b: Scaling clip to 18 billion parameters Year: (2024)
Ref_id:b27 Title: Sumcse: Summary as a transformation for contrastive learning Year: (2024)
Ref_id:b28 Title: What makes for good views for contrastive learning? Advances in neural information processing systems Year: (2020)
Ref_id:b29 Title: Fastvlm: Efficient vision encoding for vision language models Year: (2024)
Ref_id:b30 Title: Improving text embeddings with large language models Year: (2023)
Ref_id:b31 Title: Uniir: Training and benchmarking universal multimodal information retrievers Year: (2024)
Ref_id:b32 Title: Batchsampler: Sampling mini-batches for contrastive learning in vision, language, and graphs Year: (2023)
Ref_id:b33 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b34 Title: Long-clip: Unlocking the long-text capability of clip Year: (2024)
Ref_id:b35 Title: Magiclens: Self-supervised image retrieval with open-ended instructions Year: (2024)
Ref_id:b36 Title: Dreamlip: Language-image pre-training with long captions Year: (2024)
Ref_id:b37 Title: Megapairs: Massive data synthesis for universal multimodal retrieval Year: (2024)
