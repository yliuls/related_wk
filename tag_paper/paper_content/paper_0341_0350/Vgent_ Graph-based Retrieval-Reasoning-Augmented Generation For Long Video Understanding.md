Title: Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding
Abstract: Understanding and reasoning over long videos pose significant challenges for large video language models (LVLMs) due to the difficulty in processing intensive video tokens beyond context window and retaining long-term sequential information. Retrieval-Augmented Generation (RAG) has demonstrated effectiveness in processing long context for Large Language Models (LLMs); however, applying RAG to long video faces challenges such as disrupted temporal dependencies and inclusion of irrelevant information that can hinder accurate reasoning. To address these limitations, we propose Vgent, a novel graph-based retrieval-reasoning-augmented generation framework to enhance LVLMs for long video understanding. Our approach introduces two key innovations: (i) It represents videos by structured graphs with semantic relationships across video clips preserved to improve retrieval effectiveness. (ii) It introduces an intermediate reasoning step to mitigate the reasoning limitation of LVLMs, which leverages structured verification to reduce retrieval noise and facilitate the explicit aggregation of relevant information across clips, resulting in more accurate and context-aware responses. We comprehensively evaluate our framework with various open-source LVLMs on three long-video understanding benchmarks. Our approach yielded an overall performance improvement of 3.0% ∼ 5.4% over base models on MLVU, and outperformed state-of-the-art video RAG methods by 8.6%. Our code is publicly available at https://xiaoqian-shen.github.io/Vgent.

Section: Introduction
Multi-Modal Large Language Models (MLLMs) [6,22,30,39,40,58] have demonstrated remarkable visual understanding and reasoning capabilities, paving the way for advancements in video tasks. Recently, numerous studies have showcased impressive progress in building Large Video Language Models (LVLMs) for video understanding [9,7,24,35,55]. Moreover, long-video understanding is particularly crucial for applications in web content, life-logging, and streaming media, where intricate narratives and evolving contexts span extended durations.
However, processing and reasoning over long-context videos remain a formidable challenge for existing LVLMs, as representing video frames requires an extensive number of tokens-for example, a 30-minute video can exceed 200K tokens [4,22], beyond most models' context limits. To handle longer videos, existing methods resort to sparse frame sampling [55,22] or token compression [43], but these approaches inevitably lead to visual information loss, weakening fine-grained temporal understanding and coherent reasoning.
Recent studies [1,2,12,34,52,53] utilize Retrieval-Augmented Generation (RAG) [21] to enhance long-form video understanding by retrieving relevant information. However, they encounter certain limitations. First, some works segment lengthy videos into shorter clips, treating each as an individual document for retrieval [2], which disrupts the continuity of entities and temporal dependencies, 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
0 1 2 3 N Clip1 Clip2 Clip3 Clip4
...
this section cite: ['b5', 'b21', 'b29', 'b38', 'b39', 'b57', 'b8', 'b6', 'b23', 'b34', 'b54', 'b3', 'b21', 'b54', 'b21', 'b42', 'b0', 'b1', 'b11', 'b33', 'b51', 'b52', 'b20', 'b1']

Section: ClipN

this section cite: []

Section: Large Video Language Model

this section cite: []

Section: Query

this section cite: []

Section: Video Graph Construction

this section cite: []

Section: Reasoning
Filtered Video Clips ...
this section cite: []

Section: Graph-based Retrieval Structured Refinement
Where was the plate before I putting a bread slice on it?
A. cabinet B. refrigerator C. sink D. floor
this section cite: []

Section: Retrieval-Reasoning Augmented Generation
0 2 3 5 6 2 5 6 0 2 3 5 6
The plate was initially in the sink, then possibly moved to the cabinet before the bread was placed on it. Each video clip is represented as a node within a graph, interconnected through shared entities. This graph representation enables effective retrieval of relevant clips based on node connections, followed by an intermediate reasoning step to refine retrievals and aggregate over multimodal context for accurate generation.
leading to retrieval inaccuracies. Second, some methods [47,34] rely on proprietary LLMs like GPT-4 [37] for multi-turn interactions, planning, and reasoning, making them costly and less flexible. Lastly, several approaches [47,33] extract information from sparse key frames, neglecting temporal coherence in long videos.
this section cite: ['b46', 'b33', 'b36', 'b46', 'b32']

Section: Graph representation for enhanced retrieval:
To address the limitation of existing RAG methods, we propose a structured graph-based representation, where video clips are modeled as nodes interconnected by recurring subjects or scenes. This graph representation not only enables effectively retrieving nodes associated with specific entities, but also facilitates capturing temporal dependencies spanning over lengthy videos. Another advantage is that the graph construction is performed offline and is query-independent. Once the graph is built, it can be reused for multiple questions on the same video, allowing retrieval to operate directly on the graph without reprocessing the video. However, feeding all retrieved clips into LLMs can cause information overload, where key details are diluted by irrelevant content [14]. In videos, the issue is further amplified, as each frame consumes hundreds of tokens, with irrelevant information overshadowing the critical content.
this section cite: ['b13']

Section: Structured post-retrieval reasoning:
To address the aforementioned issue and fully harness the benefits from our GraphRAG, we introduce the structured reasoning step in the post-retrieval stage. Instead of generating answers directly from the retrieved clips, it decomposes the question and systematically verifies the relevance of each clip. As shown in Figure 1, this process refines the retrieved set by identifying clips that mention critical elements-such as the plate, sink, cabinet, and bread-and then aggregates information across them for temporal reasoning (e.g., the plate moving from the sink to the cabinet). This process mitigates noise, facilitates information aggregation across refined clips, and thus creates a more reliable pathway for producing accurate responses.
We evaluate our framework upon seven different LVLMs with sizes ranging from 2B to 7B across three long-video benchmarks: MLVU [57], VideoMME [13] and LongVideoBench [50]. Experimental results demonstrate that our framework consistently improves the performance of existing LVLMs by 3.0%-5.4%. We further show that our framework surpasses existing RAG-based video understanding works by 8.6%.
this section cite: ['b56', 'b12', 'b49']

Section: Contribution.
Our contribution is summarized as follows:
• We developed a novel graph-based RAG framework for long-video understanding, where video clips are represented as nodes within a graph, interconnected through shared entities, thereby preserving semantic relationships and temporal dependencies across clips, facilitating more effective retrieval.
• We propose structured reasoning to tackle the limited reasoning ability of LVLMs, which can be distracted by hard negative retrieved samples. Our approach introduces an intermediate reasoning step for retrieval verification and aggregates information across verified clips to enhance generation accuracy.
• Our graph-based retrieval-reasoning-augmented framework demonstrates 3.0%-5.4% improvements over various LVLMs ranging from 2B to 7B and surpasses existing RAG-based video understanding works by 8.6% in long-video understanding tasks.
2 Related Work
this section cite: []

Section: Large Video Language Models
Multimodal large language models (MLLMs) [6,58,31,30,45] have demonstrated remarkable progress in vision-language tasks. Recent advancements have further extended their capabilities to video understanding tasks [2,9,24,25,28,32]. Large Video Language Models (LVLMs) process videos by extracting and encoding frames, and then rearranging them into final video representations. Some approaches [24,25,9] leverage the Q-Former module from BLIP-2 [23] to integrate visual and textual features, while others [2,28,32] directly concatenate frame features. However, these models struggle with processing hour-long videos in a single pass as the number of video tokens exceeds their training context size. To address these limitations, most existing works train on sparsely sampled frames no matter how long the video is [24,2,9,55,22], while others try to handle long videos by token pooling [35,27,44], token compression [43], or memory aggregation [19]. However, they struggle to effectively capture and reason about temporal dependencies spanning hour-long videos.
this section cite: ['b5', 'b57', 'b30', 'b29', 'b44', 'b1', 'b8', 'b23', 'b24', 'b27', 'b31', 'b23', 'b24', 'b8', 'b22', 'b1', 'b27', 'b31', 'b23', 'b1', 'b8', 'b54', 'b21', 'b34', 'b26', 'b43', 'b42', 'b18']

Section: Agent-based Video Understanding
A dominant trend in long-context video question-answering involves equipping Large Language Models (LLMs) with tools that heavily rely on proprietary models to process queries and handle video clips. MM-VID [29] uses a video-to-script generation with GPT-4V [37] to transcribe multimodal elements into a long textual script. VideoAgent [47] integrates diverse foundation models through a unified memory architecture. DrVideo [34], VideoTree [49], VideoAgent [12] and OmAgent [53] dynamically invoke tools to enhance query processing and accuracy. Such methods consequently suffer from high operational costs and a critical reliance on external, closed-source systems, limiting their adaptability. In contrast, our work targets the development of a self-contained pipeline designed for flexible deployment with open-source LVLMs.
this section cite: ['b28', 'b36', 'b46', 'b33', 'b48', 'b11', 'b52']

Section: Video Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by retrieving relevant information to improve long context memory, factual accuracy, and reduce hallucinations. The process involves three stages: (i) indexing, which organizes raw data into a knowledge base;
(ii) retrieval, which searches for relevant information based on user queries; and (iii) generation, where the model takes the retrieved context to generate the final response. Recent advancements of RAG in LLM mainly follow two directions, i.e. chunk-based methods [15,5] and graph based methods [11,17,26,18], both of which have been applied in video understanding tasks. Goldfish [3] chunks long videos into shorter clips, processes each clip independently, and retrieves the most relevant clip in response to user queries. Wang et al. [48] applied graph structures for action recognition in short clips and Hussein et al. [20], Luo et al. [33] employ scene graphs for video understanding. However, constructing graphs for long videos and effectively retrieving information from the noisy and complex graph remains a challenge. Only recently, a concurrent work [42] constructs graphs for long-context video understanding, but they heavily relies on external proprietary LLM for graph construction, while graph-based video RAG with open-sourced LVLM itself remains unexplored, which our work aims to address.
this section cite: ['b14', 'b4', 'b10', 'b16', 'b25', 'b17', 'b2', 'b47', 'b19', 'b32', 'b41']

Section: Method
We introduce a novel, training-free framework, Vgent, for long-context video understanding. Unlike conventional Retrieval-Augmented Generation (RAG), our pipeline proposes a graph-based retrieval-reasoning-augmented generation paradigm, specifically designed to address complex video scenarios with improved contextual comprehension and structured reasoning. As illustrated in Figure 2, our proposed pipeline contains four stages: (1) Offline video graph construction (Section 3.1): Builds a video graph offline by extracting knowledge from long videos. (2) Graph-based
this section cite: []

Section: Offline Graph Construction
Video Clip Entities ( )
this section cite: []

Section: LVLM
Question: Please identify the option that corresponds to the order of events as they occur in the video. [Zumba, clean and jerk, milking cow, playing trombone]
this section cite: []

Section: LVLM

this section cite: []

Section: Query Keywords Extraction Strutured Query Refinement
Answer: milking cow --> zumba --> clean and jerk --> playing trombone 4 3 5 11 13
this section cite: []

Section: 🤔 Subtitles
[{"entity": "", "description": ""}, ... {"entity": "", "description": ""}]
Graph-based Clip Retrieval Clip Q1 Q2 Q3 Q4 3 4 5 11 13 Refined Clips 4 3 11 13
Milking cow is shown first, followed by zumba, then clean, and finally playing trombone. retrieval (Section 3.2): Retrieves relevant video clips from graph based on the user query. (3) Structured Reasoning (Section 3.3): Refines the retrieved clips using structured queries and aggregates information across the filtered clips. (4) Multimodal Augmented Generation (Section 3.4): Combines refined clips and intermediate reasoning results to generate the final response.
this section cite: []

Section: Information Aggregation

this section cite: []

Section: Video Graph Construction
To better capture the complex relationships and dependencies in long-context videos, we propose a graph-based representation to store video content and enhance semantic connections. Specifically, given a video V with F frames, we first partition it into a sequence of short video clips {V 1 , V 2 , . . . , V ⌈ F K ⌉ }, where each video clip V i consists of K frames. We then dynamically construct the graph by a series of structured steps, as detailed below.
this section cite: []

Section: Visual Entity Extraction.
For each video clip, we leverage the LVLM to extract the key semantic entities (i.e., the primary subjects, actions, or scenes) from both the spoken content (subtitles) C i and video clip V i .
this section cite: []

Section: {(e
i 1 , t i 1 ), (e i 2 , t i 2 ), ...} ← LVLM (C i , V i ),(1)
where the set of entity is denoted as E i = {e 1 i , e 2 i , . . . } and its corresponding description set is denoted as T i = {t 1  i , t 2 i , . . . }. In this step, the LVLM captures subjects, actions, and scene dynamics, seamlessly linking visual entities with spoken content to extract meaningful knowledge. Please refer to Appendix B.1 for illustrative examples.
this section cite: []

Section: Graph Construction.
Based on extracted information, we construct a video knowledge graph G = (V, E), where V denotes the nodes set representing video clips, and edges in E represents the connectivity between nodes. Additionally, we define a global set of unique prototype entities U = {u ∈ E i , i = 1, . . . , ⌈ F K ⌉} that spans across all nodes. As more video clips are processed, we will dynamically add newly extracted unique entity u to the set or link it to an existing entity. We define t u as the description of each entity u ∈ U.
this section cite: []

Section: Entity Merging and Node Connection.
Since LVLMs process video clips independently, it is essential to identify and unify semantically equivalent entities across clips. Given a newly extracted entity-description pair (e j i , t j i ) from video clip V i , we determine whether it belongs to an existing entity in the global entity set U. Specifically, we compute the similarity score between the textual descriptions t i j and descriptions of entities in U based on their respective text embeddings. If the similarity score > τ , the entity e i j is considered semantically equivalent to an existing entity and these two are merged into a single entity representation. Otherwise, e i j is treated as a distinct entity and added to U. This process is formulated as follows:
s * = max u∈U sim(t i j , t), u * = arg max u∈U sim(t i j , t u ), e i j → u * , if s * ≥ τ U ← U ∪ {e i j }, otherwise(2)
Once entity is merged, we then build edges from the node v i associated with the video clip V i to all the nodes that have the same entity u * , denoted as
V (u * ) . E ← E ∪ {(v i , v) | v ∈ V (u * ) } (3
)
As new video clips are processed, the graph is dynamically updated such that nodes containing the same entity are connected, which preserves semantic relationships and contextual dependencies. This forms a structured representation that facilitates effective video retrieval in subsequent processing stages.
this section cite: []

Section: Graph-based Retrieval
Keywords Extraction. Direct retrieval based on the original query may not provide sufficient context, especially when reasoning across multiple temporal clips is required. To address this, we extract keywords from the query for effective retrieval. Specifically, we prompt the LVLM to identify key semantic elements, denoted as K, from the query Q. The detailed prompt is provided in Appendix B.2.
Graph-based Clip Retrieval. Next, we leverage these extracted keywords for graph-based retrieval. Specifically, for each keyword k ∈ K and each entity u ∈ U, we compute a similarity score sim(k, t u ) to determine whether the entity matches the keyword. If sim(k, t u ) > θ, we include all nodes associated with entity u as the target retrieval node set R:
R = u∈U ,k∈K {v ∈ V | u ∈ U(v), sim(k, t u ) > θ}(4)
After obtaining the retrieval node set R, we refine the results by re-ranking the nodes based on the similarity between the query's keywords and the extracted information of each node, including entities, corresponding textual descriptions, and subtitles if available. Finally, we select the Top-N nodes with the highest average similarity scores across all associated information of each video clip.
this section cite: []

Section: Structured Reasoning
Feeding all relevant clips directly into LLMs can lead to information overload, diluting the focus on key details with irrelevant content [14]. Our empirical analysis also reveals that in roughly 40% of failure cases, the correct clip is successfully retrieved, yet the model still generates incorrect responses-even though it can answer correctly when provided with that clip alone. We then introduce structured reasoning in the post-retrieval stage that refines the retrieved clips and aggregates useful information towards final generation.
this section cite: ['b13']

Section: Structured Query Refinement.
We introduce the divide-and-conquer strategy to refine the retrieval through structured query verification. Specifically, we prompt the LVLM to generate structured subqueries, denoted as Q, based on the original query Q and extracted keywords K. These subqueries focus on verifying the presence of relevant entities or quantifying their occurrences, whose answers are expected to be binary (yes/no) or numerical value. Please refer to Appendix B.3 for the detailed prompt and Figure 3 for an example of generated subqueries.
After generating the subqueries, we process the Top-N retrieved video clips using the LVLM, producing either binary (yes/no) or numerical responses for each subquery. As shown in Figure 2, this structured verification systematically assesses the relevance of each clip to the original query, filtering out irrelevant clips that were wrongly retrieved based on semantic embedding similarity. Denoting 1 to yes and 0 to no in binary questions, this refined clip set R ′ can be formulated as:
R ′ = {v i ∈ R | ∃q j ∈ Q, f (v i , q j ) > 0}(5)
where f (v i , q j ) denotes the response of retrieved clip v i to subquery q j . We keep at most r clips after refinement. This refinement step ensures that only video clips satisfying the structured queries are retained, effectively eliminating hard negatives from the initial retrieval.
Information Aggregation. As shown in Figure 2, we then let LVLM aggregate and summarize all useful information from structured queries and their corresponding results for each video clip, providing an enriched auxiliary context that enhances the final inference.
this section cite: []

Section: Multimodal Augmented Generation.
We incorporate both the intermediate reasoning results and the filtered video clips as multimodal context inputs to the LVLM for the final response. This enriched input allows the model to leverage both structured reasoning and relevant visual information, enabling it to generate a more accurate and contextually grounded final response to the original question.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setups
Baselines. We apply our framework Vgent on open-sourced LVLMs including InternVL2.5 [8], Qwen2 [46], Qwen2.5-VL [4], LongVU [43] and LLaVA-Video [56] as base video understanding model. We further compare Vgent against state-of-the-art RAG baselines as follows: NaïveRAG [3], Video-RAG [33], and proprietary LLM-based methods including VideoAgent [47], LLoVi [52], DrVideo [34] and VideoTree [49]. More details can be found in Appendix A.3.
this section cite: ['b7', 'b45', 'b3', 'b42', 'b55', 'b32', 'b46', 'b51', 'b33', 'b48']

Section: Benchmarks.
We evaluate the performances of each model across three long-video benchmarks. Video-MME [13] is a widely used benchmark designed to evaluate LVLMs' capability to process detailed, real-world videos. It comprises three subsets categorized by video length, ranging from 11 seconds to 1 hour. MLVU [57] is a long-video understanding benchmark with videos ranging from 3 minutes to 2 hours, with an average length of about 12 minutes. LongVideoBench (LVB) [50] focuses on referred reasoning tasks that require models to analyze long frame sequences. These questions depend on extensive temporal context and cannot be effectively addressed using a single frame or a small set of sparsely sampled frames.
this section cite: ['b12', 'b56', 'b49']

Section: Implementation Details.
During the offline video graph construction, we sample videos at 1.0 FPS, segmenting the long video into clips, each containing K = 64 frames. We use the BAAI/bge-large-en-v1.5 [51] embedding for similarity calculation. The entity merging threshold is set to τ = 0.7. In the online retrieval stage, we use BAAI/bge-large-en-v1.5 to retrieve the top N = 20 clips based on extracted keywords (maximum to 20 to discard low-relevance, with a similarity threshold θ = 0.5). After structured query refinement, we retain a maximum of r = 5 clips. Thresholds are set as the same for all three benchmarks, with hyper-parameter selection details provided in the supplementary. For MLVU [57], we extract spoken content using openai/whisper-large, while for VideoMME [13] and LongVideoBench [50], we use the provided subtitles from benchmark.
All experiments are conducted on A100 80G GPUs.
this section cite: ['b50', 'b56', 'b12', 'b49']

Section: Main Results

this section cite: []

Section: Comparison with LVLMs.
In Table 1 and  an accuracy of 70.4%, surpassing its larger 7B counterpart and improving the base model by 4.2%. This result underscores the effectiveness of our approach in bridging the performance gap between small models and their larger counterparts. At the category level (Table 6), our framework notably improves Count and Order tasks, which demand event-level understanding and multi-clips reasoning.
In Table 1, we showcase the results of Vgent on the VideoMME [13] benchmark, where it consistently outperforms base models across all video lengths, achieving an average performance gain of 4.2%. Notably, our framework excels in long-video scenarios, surpassing the best baseline by 5.4%. These findings highlight the strength of our structured graph-based retrieval and reasoning approach, demonstrating its ability to enhance long-video comprehension by effectively capturing cross-segment dependencies and refining information retrieval for improved reasoning and final response generation.
Comparison with SoTA RAG Methods. In Table 2, we provide a comprehensive comparison of Vgent against state-of-the-art RAG methods on MLVU [57] and VideoMME [13] benchmarks.
(1) Our framework consistently outperforms the RAG baseline, Video-RAG [33], across three different LVLM base models. Unlike Video-RAG [33], which relies on CLIP [41]-based keyframe selection and external tools such as object detection and OCR for frame-level information extraction, Vgent eliminates these dependencies by leveraging LVLMs themselves for graph construction, verification, and intermediate reasoning. This structured approach significantly enhances retrieval precision and reasoning accuracy, leading to more reliable final responses.
(2) Our framework also surpasses proprietary RAG-based methods for long-video understanding. Compared to closed-source API-dependent methods which heavily rely on closed-source APIs, our framework is more flexible and effective solution for long-video understanding.
this section cite: ['b12', 'b56', 'b12', 'b32', 'b32', 'b40']

Section: Ablation Studies
NaïveRAG vs GraphRAG. As shown in Table 3, integrating GraphRAG yields an average improvement of 2.9% over NaïveRAG, with a particularly notable 4.1% gain on MLVU [57]. This is because NaïveRAG's difficulty in handling complex queries that requires temporal reasoning across multiple clips, as it treats each video clip as an independent document. In contrast, our GraphRAG effectively preserves semantic relationships between clips, enabling more accurate retrieval and reasoning. By structuring video content into a graph representation, our approach addresses retrieval inconsistencies inherent in traditional RAG methods.
However, the improvement remains marginal compared to the base models. Upon checking failure cases in MLVU, we observe that in 44% of the failures, the correct clip is actually present within the model's retrieved set, which indicates that while the retrieval was successful, irrelevant retrievals still distract the model, hindering accurate responses. Consequently, a post-retrieval stage is necessary to amplify the potential of our GraphRAG by refining the retrieved nodes and improving reasoning towards more precise answers.
this section cite: ['b56']

Section: Structured Reasoning (SR).
By refining retrieved nodes through intermediate reasoning with structured queries, we achieve an additional 2.6% improvement on MLVU [57] and 1.6% on VideoMME [13], resulting in an overall 3.4% average gain over the base model. This interme-diate reasoning step decomposes complex queries into targeted sub-questions and generates binary or numerical answers. These structured response are then used to systematically filter out irrelevant clips and aggregate relevant information across clips, guiding the model toward the correct final answer. Our findings also indicate that the final improvement is contingent upon Graph-based RAG. Specifically, if SR is applied to NaïveRAG, the inherent inaccuracy of NaïveRAG's retrievals restricts the potential for significant improvement through refinement alone.
this section cite: ['b56', 'b12']

Section: Number of retrieval r
We conduct an ablation study to examine the impact of the number of retrieved clips after structured query refinement. Table 4 presents both the overall performance and results across several MLVU [57] subcategories. Among these, Count and Order are two tasks that heavily require reasoning across multiple video clips. Count involves identifying the number of events or actions throughout an entire video, while Order requires the model to arrange multiple events in chronological sequence. r represents the maximum number of video clips retained after refinement. Our findings indicate that increasing the number of retrieved clips consistently improves performance, particularly for tasks demanding multi-clip reasoning, with the highest performance observed at r = 5.
this section cite: ['b56']

Section: Inference Time Analysis
We analyze the computational trade-offs and report the processing times in Table 5 for the APIbased method VideoAgent [47], Video-RAG [33] as well as our framework built on Qwen2.5VL [4]. VideoAgent [47] leverages a proprietary LLM (GPT-4 [36]) to iteratively perform self-reflection for frame selection and aggregating key information from the video. Video-RAG [33] relies on querydependent key frame selection and per-frame object detection, introducing online computational overhead. In contrast, our framework can offline constructs a query-independent graph from the video, which takes 20.13 seconds. Once the graph is built, the online retrieval, reasoning and generation process requires only 3.93 seconds per minute-video.
Our offline graph construction further improves efficiency in multi-question scenarios (e.g., three questions per video in VideoMME [13]). Unlike query-dependent methods that reprocess the entire
Question: Did I open the laptop? A. Maybe B. No C. I don't know D. Yes Q1: Is there a laptop shown in the video? Q2: Is someone visible interacting with the laptop in the video? Q3: Does the video show the laptop opened? Reasoning A laptop is shown in the video as not opened, and someone is interacting with it. Then, the laptop is shown opened. Graph-based Clip Retrieval [5,6,7,9,11] Answer: B. No Answer: D. Yes 0 1 2 3 4 5 6 7 8 9 10 11 laptop watch ... ... monitor bed Clip 4 Clip 5 Clip 6 Clip 7 Clip 8 Clip 9 Clip 10 Clip 11 ... ... ... Video Graph Construction Strutured Queries GraphRAG w/o reasoning GraphRAG with reasoning video for each question, our approach constructs the graph once, allowing the model to retrieve relevant clips based on entity descriptions-without the need to rewatch the entire video. As a result, our method achieves a 1.73× speedup over Video-RAG [33] when performing inference on VideoMME [13].
this section cite: ['b46', 'b32', 'b3', 'b46', 'b35', 'b32', 'b12', 'b32', 'b12']

Section: Qualitative Examples
We show a qualitative example in Figure 3, 5 and 6. Our graph construction effectively connects relevant video clips through shared entities. In Figure 3 the graph-based retrieval system can identify relevant nodes that contains a laptop, with Clip 6 providing crucial evidence to answer the query. However, the model incorrectly responded "No" to the question "Did I open the laptop?", presumably due to hard negatives from multiple clips featuring a opened laptop, hallucinating the model to overlook the closed laptop and the action of opening it.
In contrast, with an intermediate reasoning step, we validate each retrieved node with structured subqueries (e.g., "Is there a laptop open?" "Is someone interacting with the laptop?"). This verified information is aggregated to form an enhanced reasoning chain, allowing the model to correctly infer that the laptop was opened, overcoming the distraction from hard negatives.
this section cite: []

Section: Conclusion
In this work, we introduced a novel graph-based Retrieval-Augmented Generation (RAG) framework designed for long-video understanding. Our approach represents video clips as nodes in a graph and leverages entities to maintain semantic relationships, thereby enhancing retrieval effectiveness. To address retrieval noise, we proposed a structured query refinement strategy that systematically filters out irrelevant clips, ensuring a more precise selection of relevant video content. Additionally, we introduced an intermediate reasoning step that summarizes the response to the structured query, using the filtered retrieved clips as multimodal context to significantly improve the accuracy of the final answer generation. Our framework outperforms state-of-the-art video RAG methods by 8.6%, demonstrating its effectiveness in enhancing long-video understanding tasks. This work paves the way for more accurate and context-aware long-form video retrieval and reasoning systems.
this section cite: []

Section: References
Ref_id:b0 Title: Md Yusuf Sarwar Uddin, and Srimat Chakradhar. irag: Advancing rag for videos with an incremental approach Year: (2024)
Ref_id:b1 Title: Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens Year: (2024)
Ref_id:b2 Title: Goldfish: Vision-language understanding of arbitrarily long videos Year: (2024)
Ref_id:b3 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b4 Title: Rq-rag: Learning to refine queries for retrieval augmented generation Year: (2024)
Ref_id:b5 Title: Minigpt-v2: large language model as a unified interface for vision-language multi-task learning Year: (2023)
Ref_id:b6 Title: Sharegpt4video: Improving video understanding and generation with better captions Year: (2024)
Ref_id:b7 Title: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling Year: (2024)
Ref_id:b8 Title: Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms Year: (2024)
Ref_id:b9 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b10 Title: From local to global: A graph rag approach to query-focused summarization Year: (2024)
Ref_id:b11 Title: Videoagent: A memoryaugmented multimodal agent for video understanding Year: (2024)
Ref_id:b12 Title: Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis Year: (2024)
Ref_id:b13 Title: Parameter-efficient visual instruction model Year: (2023)
Ref_id:b14 Title: Retrieval-augmented generation for large language models: A survey Year: (2023)
Ref_id:b15 Title: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context Year: (2024)
Ref_id:b16 Title: Simple and fast retrievalaugmented generation Year: (2024)
Ref_id:b17 Title: Retrieval-augmented generation with graphs (graphrag) Year: (2024)
Ref_id:b18 Title: Ma-lmm: Memory-augmented large multimodal model for long-term video understanding Year: (2024)
Ref_id:b19 Title: Videograph: Recognizing minuteslong human activities in videos Year: (2019)
Ref_id:b20 Title: Retrieval-augmented generation for knowledgeintensive nlp tasks Year: (2020)
Ref_id:b21 Title: Llava-onevision: Easy visual task transfer Year: (2024)
Ref_id:b22 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b23 Title: Videochat: Chat-centric video understanding Year: (2023)
Ref_id:b24 Title: Mvbench: A comprehensive multi-modal video understanding benchmark Year: (2024)
Ref_id:b25 Title: Simple is effective: The roles of graphs and large language models in knowledge-graph-based retrieval-augmented generation Year: (2024)
Ref_id:b26 Title: Llama-vid: An image is worth 2 tokens in large language models Year: (2023)
Ref_id:b27 Title: Video-llava: Learning united visual representation by alignment before projection Year: (2023)
Ref_id:b28 Title: Mm-vid: Advancing video understanding with gpt-4v (ision) Year: (2023)
Ref_id:b29 Title: Llava-next: Improved reasoning, ocr, and world knowledge Year: (2024)
Ref_id:b30 Title: Visual instruction tuning Year: (2024)
Ref_id:b31 Title: Video assistant with large language model enhanced ability Year: (2023)
Ref_id:b32 Title: Video-rag: Visually-aligned retrieval-augmented long video comprehension Year: (2024)
Ref_id:b33 Title: Document retrieval based long video understanding Year: (2024)
Ref_id:b34 Title: Video-chatgpt: Towards detailed video understanding via large vision and language models Year: (2023)
Ref_id:b35 Title:  Year: (2023)
Ref_id:b36 Title: Gpt-4v(ision) system card Year: (2023)
Ref_id:b37 Title: Gpt-4o system card Year: (2024)
Ref_id:b38 Title: Internvl2: Better than the best-expanding performance boundaries of open-source multimodal models with the progressive scaling strategy Year: (2024)
Ref_id:b39 Title:  Year: (2024)
Ref_id:b40 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b41 Title: Videorag: Retrievalaugmented generation with extreme long-context videos Year: (2025)
Ref_id:b42 Title: Spatiotemporal adaptive compression for long video-language understanding Year: (2024)
Ref_id:b43 Title: From dense token to sparse memory for long video understanding Year: (2023)
Ref_id:b44 Title: Cambrian-1: A fully open, vision-centric exploration of multimodal llms Year: (2024)
Ref_id:b45 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b46 Title: Long-form video understanding with large language model as agent Year: (2024)
Ref_id:b47 Title: Supervoxel attention graphs for long-range video modeling Year: (2021)
Ref_id:b48 Title: Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos Year: (2024)
Ref_id:b49 Title: Longvideobench: A benchmark for long-context interleaved video-language understanding Year: (2025)
Ref_id:b50 Title: C-pack: Packaged resources to advance general chinese embedding Year: (2023)
Ref_id:b51 Title: A simple llm framework for long-range video question-answering Year: (2023)
Ref_id:b52 Title: Omagent: A multi-modal agent framework for complex video understanding with task divide-and-conquer Year: (2024)
Ref_id:b53 Title: Long context transfer from language to vision Year: (2024)
Ref_id:b54 Title: Llava-next: A strong zero-shot video understanding model Year: (2024)
Ref_id:b55 Title: Video instruction tuning with synthetic data Year: (2024)
Ref_id:b56 Title: Mlvu: A comprehensive benchmark for multi-task long video understanding Year: (2024)
Ref_id:b57 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2023)
