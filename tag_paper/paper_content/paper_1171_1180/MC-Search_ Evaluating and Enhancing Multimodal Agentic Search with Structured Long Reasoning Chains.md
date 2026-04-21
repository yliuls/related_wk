Title: MC-SEARCH: EVALUATING AND ENHANCING MULTI-MODAL AGENTIC SEARCH WITH STRUCTURED LONG REASONING CHAINS
Abstract: With the increasing demand for step-wise, cross-modal, and knowledge-grounded reasoning, multimodal large language models (MLLMs) are evolving beyond the traditional fixed retrieve-then-generate paradigm toward more sophisticated agentic multimodal retrieval-augmented generation (MM-RAG). Existing benchmarks, however, mainly focus on simplified QA with short retrieval chains, leaving adaptive planning and multimodal reasoning underexplored. We present MC-SEARCH, the first benchmark for agentic MM-RAG with long, step-wise annotated reasoning chains spanning five representative reasoning structures. Each example specifies sub-questions, retrieval modalities, supporting facts, and intermediate answers, with fidelity ensured by HAVE (Hop-wise Attribution and Verification of Evidence), resulting in 3,333 high-quality examples averaging 3.7 hops. Beyond answer accuracy, MC-SEARCH introduces new process-level metrics for reasoning quality, stepwise retrieval and planning accuracy. By developing a unified agentic MM-RAG pipeline, we benchmark six leading MLLMs and reveal systematic issues such as over-and under-retrieval and modality-misaligned planning. Finally, we introduce SEARCH-ALIGN, a process-supervised fine-tuning framework leveraging verified reasoning chains, showing that our data not only enables faithful evaluation but also improves planning and retrieval fidelity in open-source MLLMs.

Section: INTRODUCTION
Multimodal large language models (MLLMs) have rapidly advanced in their reasoning abilities (Wang et al., 2025b;Huang et al., 2025b;Li et al., 2024), moving beyond text-only inputs to interleaved cross-modal contexts. To ensure factuality and robustness, Retrieval-Augmented Generation (RAG) has emerged as a key paradigm for grounding model outputs in external textual evidence (Gao et al., 2023;Zhao et al., 2024;Fan et al., 2024). Building on this, Multimodal RAG (MM-RAG) equips MLLMs with visual seeking and extends knowledge grounding from text-only evidence to multimodal contexts, thereby facilitating cross-modal reasoning for knowledge-intensive queries.
Q: What is the name of the bus operator depicted in the image, its establishment year, the start of its mainline service to San Francisco and what is the city where its headquarters are located? <Thought> The question asks for the name … <Sub-Question> What is the name of the bus operator depicted in the image? <Question> What was the historical significance of this acquisition for the artist regarding American museums?
Retrieved golden supporting text.... <Answer> It was Tanner's first work to be purchased by an American museum.
<End> Golden Answer: The painting depicts the Annunciation scene, is held by the Philadelphia Museum of Art, was acquired on April 5, 1899 for $1,750, and it was Tanner's first work to be purchased by an American museum.
c Reasoning ediction Golden Reasoning Chain Q: What is the name of the bus operator depicted in the image, its establishment year, the start of its mainline service to San Francisco and what is the city where its headquarters are located?
<Thought> The question asks for the name of the bus operator, its establishment year, the start of its mainline service…… <Sub-Question> What is the name of the bus operator depicted in the image?
<Search> Local Image Retrieval with Input Image </Search> Retrieved content omitted.... <Thought> I need to gather more information about SamTrans, specifically its establishment year and the start of its mainline service to San Francisco. <Sub-Question> What is the city where SamTrans is headquartered? Retrieved content omitted.... <Question> What is the name of the bus operator visible on the vehicle in the image? Retrieved golden supporting image…… <Answer> SamTrans. <Question> In what year was the district for SamTrans established? Retrieved golden supporting text…… <Answer> 1976. <Question> How many years after its establishment did SamTrans begin mainline bus service to San Francisco? Retrieved golden supporting text.... <Answer> One year later. <Question> Where are the headquarters of SamTrans located? Retrieved golden supporting text.... <Answer> San Carlos, California. <End> Golden Answer: The bus operator is SamTrans. It was established in 1976 and began mainline bus service to San Francisco one year later. Its headquarters are located in San Carlos, California.
this section cite: ['b19', 'b41', 'b7']

Section: Agentic Reasoning Prediction
Golden Reasoning Chain Q: What is the name of the bus operator depicted in the image, its establishment year, the start of its mainline service to San Francisco and what is the city where its headquarters are located?
<Thought> The bus operator's name is "SamTrans", as indicated in the retrieved image caption. Now I need to find information…… <Sub-Question> What is the establishment year of SamTrans and when did it start….. <End> Final Answer: SamTrans is headquartered in San Carlos, California, but I cannot provide its establishment year or the start of its mainline service to San Francisco.
<End> Final Answer: The bus operator is SamTrans. It was established in 1976 … one year later. Its headquarters is in San Carlos, California.
Conversation continues for two more turns… In practice, queries for MLLMs are often ambiguous and complex (Figure 1). They go far beyond the shallow cases solvable with fixed minimal retrieval and instead require multi-step, cross-modal, and knowledge-intensive reasoning. As MLLMs advance toward long reasoning models (Wang et al., 2024), they are increasingly positioned as backbones for supporting such problem-solving needs. Realizing this potential, however, demands richer agentic behaviors (Li et al., 2025a;b), such as iterative task decomposition, adaptive crossmodal retrieval, and multimodal evidence integration. This shift calls for moving beyond the fixed retrieve-then-generate paradigm of classic RAG toward multimodal agentic search-enhanced reasoning (multimodal agentic RAG) (Wang et al., 2025a;Li et al., 2025a;Huang et al., 2025a;Jin et al., 2025;Zheng et al., 2025b;Chen et al., 2025). Accordingly, new benchmarks and evaluation pipelines are needed to faithfully capture and evaluate these capabilities.
Table 1: Left: Comparison of existing multimodal retrieval-augmented QA datasets. Right: Distribution of the five reasoning topologies in MC-SEARCH, with outer rings illustrating hop diversity (2-5 hops).
this section cite: ['b37', 'b5']

Section: Benchmark Knowledge Modality
Typical Hops Long Chain (≥ 4 Hops)
Stepwise Annotation Reasoning Topology OK-VQA (Marino et al., 2019) Text 1 Hop ✗ ✗ ✗ ViQuAE (Lerner et al., 2022) Text 1 Hop ✗ ✗ ✗ WebQA (Chang et al., 2022) Text/Caption ≤2 Hops ✗ ✗ ✗ InfoSeek (Chen et al., 2023) Text 1 Hop ✗ ✗ ✗ MMSearch (Jiang et al., 2024) Text/Image 1 Hop ✗ ✗ ✗ Dyn-VQA (Li et al., 2025b) Text/Image ≤2 Hops ✗ ✗ ✗ MRAG (Hu et al., 2025) Image 1 Hop ✗ ✗ ✗ M 2 RAG (Liu et al., 2025c) Text/Image
1 Hop ✗ ✗ ✗ MC-SEARCH Text/Image ≥4 Hops ✓ ✓ ✓ Im ag e-In it ia te d Ch ai n Tex t-O nly Ch ain Par alle l Ima ge-T ext For k M ul ti -I m ag es Fo rk Text-In itiated Chain 3 Ho p 4 Hop 5 H o p 2 H o p 3 Ho p 4 Ho p 5 H o p 3 H op 4 Hop 5 H o p 3 H o p 4 H op 5 Ho p 2 Hop 3 Hop 4 Hop
Although a number of recent benchmarks (Hu et al., 2025;Jiang et al., 2024;Liu et al., 2025c;Li et al., 2025b) have provided valuable evaluations of MM-RAG, they primarily focus on straightforward visual evidence: seeking questions with only 1-2 retrieval steps, thus falling short of examining the adaptive search-enhanced reasoning behaviors required in practice. Specifically, current datasets remain limited in three key respects: (i) most adopt simple question-answer formats with a fixed retrieve-then-generate pipeline that collapses multimodal evidence into a primary textual channel (Hu et al., 2025); (ii) they restrict evaluation to short 1-2 hop retrievals without long, adaptive reasoning trajectories (Li et al., 2025b); and (iii) they lack stepwise annotations and explicit reasoning topologies that clarify the roles of different modalities in the reasoning process (Hu et al., 2025;Liu et al., 2025b;Li et al., 2025b). As summarized in Table 1 (Left), these limitations make it difficult to determine whether MLLMs can truly perform long, structured reasoning over multimodal evidence.
To fill this gap, we present MC-SEARCH, the first benchmark for Multimodal agentic RAG with long, structured Chains of Search-enhanced reasoning. Each example is paired with a golden step-wise trajectory specifying the sub-question sequence, retrieval modality, supporting fact, and intermediate answer, enabling fine-grained evaluation, laying the foundation for process reward modeling.
MC-SEARCH is designed to be long, diverse, and non-redundant. It covers five representative multi-hop reasoning structures as visualized in Figure 2, including Text-Only Chain, Image-Initiated Chain, Text-Initiated Chain, Parallel Image-Text Fork, and Multi-Image Fork, capturing both serial and parallel reasoning patterns across modalities. To guarantee that each hop is necessary and structurally meaningful, we introduce HAVE (Hop-wise Attribution and Verification of Evidence), which filters spurious or redundant steps, resulting in 3,333 high-quality annotated examples with an average chain length of 3.7 hops, surpassing existing benchmarks (Hu et al., 2025;Liu et al., 2025b;Li et al., 2025b).
Table 1 (Right) reports the chain-length distribution across reasoning topologies. Beyond answer-level accuracy, in MC-SEARCH, we propose three process-level evaluation metrics: (i) LLM-as-a-Judge for open-ended reasoning quality, (ii) Structure-Aware per Step Hit Rate for per-step retrieval fidelity, and (iii) Rollout Deviation to quantify execution drift. To evaluate the abilities, we further develop a unified agentic MM-RAG pipeline for fair benchmarking and extensive experiments with six leading MLLMs, including both proprietary and open-source models.
Our analysis reveals key weaknesses such as over-retrieval, under-retrieval, and modality-specific planning errors, underscoring the need for better adaptive planning and retrieval over multimodal evidence.
In addition to serving as an evaluation resource, our data also provides training signals for improving model capabilities. To this end, we present SEARCH-ALIGN, a conversation-level alignment framework built on HAVE-verified reasoning chains. It applies supervised fine-tuning beyond final answers, using step-wise trajectories to provide process-level supervision that strengthens open-source models' ability to plan and retrieve across modalities.
Our main contributions are summarized as follows:
• Benchmark: We present MC-SEARCH, the first benchmark for agentic multimodal RAG with long, step-wise annotated reasoning chains spanning five representative structures, verified by the HAVE procedure for necessity and non-redundancy.
• Metrics: We propose new process-level metrics that move beyond answer accuracy to precisely attribute reasoning quality, per-step retrieval and planning fidelity.
• Evaluation: We develop a unified agentic MM-RAG pipeline and conduct extensive benchmarking of six leading MLLMs, revealing systematic issues such as over-and under-retrieval, modalitymisaligned planning, and eight characteristic error types.
this section cite: ['b27', 'b18', 'b4', 'b6', 'b14', 'b11', 'b11', 'b14', 'b11', 'b11', 'b11']

Section: MC-Search Benchmark Construction Multimodal Agentic RAG Pipeline
Multimodal Large Reasoning Model (e.g.,
this section cite: []

Section: GPT-4o)
Thinking …
Sub-Query: e.g.，What biblical scene featuring Archangel Gabriel is depicted in the image? Action: e.g.，[Image Search with Text Query] Image Search with Image Text Search Multimodal KB Reason in Retrieved Content Sub-Answer: e.g.， The Annunciation. Iterate Original Question: What biblical scene featuring Archangel Gabriel is depicted in the image, where's this painting currently held, when and for how much was it acquired for the collection, and what historical significance did this acquisition represent for the artist regarding American museums? Input Multimodal Large Language Models Grouped Metadata Instruction -Context Grounded -Accurate and Coherent -Requirement of the Reasoning topology Example Q: Main Question A: Final Answer Reasoning Chain: [ (Sub-question, Fact_ID, Sub-Answer)…]
this section cite: []

Section: ~3K Samples

this section cite: []

Section: …
Image-Initiated Chain Text-Initiated Chain
this section cite: []

Section: …
Text-Only Chain
this section cite: []

Section: …
Parallel Image-Text Fork
this section cite: []

Section: …
Multi-Images Fork
this section cite: []

Section: …
Data across Five Reasoning Topologies
this section cite: []

Section: Data Filtering by HAVE: Hop-Wise Attribution and Verification

this section cite: []

Section: Navigation Ability Examination

this section cite: []

Section: Context Utility Examination
Redundant Hop Shrinkage Redundant Hop?
Final Answer: The painting depicts the Annunciation scene, is held by the Philadelphia Museum of Art, was acquired on April 5, 1899 for $1,750, and it was Tanner's first work to be purchased by an American museum.
this section cite: []

Section: Output

this section cite: []

Section: Golden Chain
Predicted Chain • Method: We introduce SEARCH-ALIGN, which leverages verified reasoning chains for processsupervised fine-tuning of MLLMs, demonstrating the effectiveness of our data for training and improving planning and retrieval fidelity in agentic MM-RAG.
this section cite: []

Section: MC-SEARCH BENCHMARK
We propose MC-SEARCH, a large-scale benchmark for evaluating agentic multimodal RAG. In contrast to prior benchmarks that focus on shallow VQA tasks, MC-SEARCH provides long, step-wise verified reasoning chains across diverse reasoning topologies, enabling both fine-grained processlevel analysis and model alignment. In this section, we detail the benchmark construction, dataset composition, and process-level evaluation metrics, with the overall workflow illustrated in Figure 2.
this section cite: []

Section: BENCHMARK CONSTRUCTION AND COMPOSITION

this section cite: []

Section: Search-Enhanced Reasoning Topologies.
To capture the diversity of real-world agentic MM-RAG workflows, we design five distinct reasoning topologies that characterize how multimodal knowledge interacts and depends on each other within a long search-enhanced reasoning process. Each topology is represented as a reasoning graph (or reasoning chain) G(Q, A), associated with an overall question Q and its final answer A. The graph consists of sub-questions q t , retrieval modalities m t ∈ {text, image}, retrieved evidence r t , and intermediate answers a t derived from the evidence at hop t (Figure 2). We denote by R the retrieval function that, given a sub-question q t and modality m t , returns the corresponding evidence r t . Formally,
G(Q, A) = {(q t , m t , r t , a t )} T t=1 , r t = R(q t , m t ), A = f ({a t } T t=1 ),(1)
where f (•) denotes the reasoning procedure that aggregates intermediate answers into the final answer A for the overall question Q.
Concretely, the five reasoning graphs are defined as: (i) Image-Initiated Chain, where reasoning is anchored in image retrieval at the first step and subsequently builds on textual retrieval, applies to both image-containing and text-only queries; (ii) Text-Initiated Chain, where reasoning begins with text retrieval of factual knowledge and later incorporates image retrieval for visual validation; (iii) Parallel Image-Text Fork, where image and text retrieval evolve concurrently without dependency across specific hops, so that interchanging the retrieval of such hop order does not alter the final answer, requiring cross-modal coordination; (iv) Multi-Images Fork, where multiple images must be retrieved for visual comparison before turning to textual evidence for factual support; (v) Text-Only Chain, serving as a baseline where reasoning proceeds entirely through text. Data examples for each reasoning graph are provided in Appendix C.
this section cite: []

Section: Data Generation.
We construct the MC-SEARCH dataset by collecting multimodal knowledge from Wikipedia and organizing it into coherent clusters of text and images, typically drawn from the same or closely related pages, following Chang et al. (2022). From these clusters, we curate salient text facts and associated images as the basis for multi-hop QA generation, supplemented with illustrative examples aligned with each reasoning topology. From the curated pools, we then generate questions at scale while ensuring balanced coverage across the five reasoning structures. Specifically, we prompt Gemini-2.5-Flash to select one topology and synthesize a structured multi-hop question together with its answer and a coherent reasoning chain grounded in supporting facts. This procedure yields approximately 21k examples aligned with the five search-enhanced reasoning topologies. Additional construction details and prompt templates are provided in Appendix N.
this section cite: ['b4']

Section: Data Filtering.
A key challenge in constructing a high-quality dataset is that MLLMs often generate long reasoning chains with hallucinated steps, that is, plausible but ungrounded hops not supported by evidence, and redundant steps that do not contribute to the final answer. To address this, we propose HAVE (Hop-wise Attribution and Verification of Evidence), a data filtering mechanism that determines whether each hop in the reasoning chain is both necessary and non-redundant.
Let C = {r 1 , r 2 , . . . , r T } denote the set of retrieved evidence in each hop of a reasoning graph. We define the context utility of step r t as the drop in answer accuracy (measured by the F1 score between golden and generated answers) when r t is removed:
Util(t) = F1(C) -F1(C \ r t ).(2)
A step is identified as necessary if Util(t) exceeds a threshold. However, some steps may not directly affect answer accuracy but still serve to connect sub-questions and guide downstream reasoning. For example, in a two-hop chain, the first step may identify the building in the input image as "Christ Church Cathedral," while the second step queries factual knowledge about this entity. Although the first step has low context utility on its own, its extracted entity is essential for the second hop, making it indispensable for the reasoning process. To capture such cases, we further assess the navigational role of a step by checking whether entities in its intermediate answer a t appear in downstream sub-questions:
Nav(t) = 1, if Ent(a t ) ∩ Ent(q t+1:T ) ̸ = ∅, 0, otherwise,(3)
where Ent(•) denotes the set of entities extracted from a sub-question or an intermediate answer. If a step has Util(t) below the threshold and Nav(t) = 0, it is marked as redundant. We apply HAVE at scale by first filtering the generated dataset to remove samples with more than two redundant hops in their reasoning chains using the open-source Qwen2.5-VL-7B, yielding 4.8k candidate chains. To further mitigate model-specific bias, we then perform hop shrinkage and sample verification with Gemini-2.5-Pro, while simultaneously extracting key knowledge entities from the golden answers to facilitate future answer evaluation. Finally, we conduct a redundancy check using HAVE with the data-generation model Gemini-2.5-Flash, and yielding 3,333 high quality samples with all necessary hops and coherent reasoning trajectories. The final dataset composition and statistics are reported in Table 2 and Appendix E.
Moreover, to ensure the reliability of subsequent step-wise retrieval accuracy evaluation, we enforce that the reasoning graph for each question is unique. To this end, we use Gemini-2.5-Flash for automatic verification, detecting and filtering confounding knowledge pieces in the knowledge base so that no unused text fact or image within the same topic cluster can independently resolve any sub-question. This provides a solid foundation for step-wise evaluation of agentic MM-RAG.
this section cite: []

Section: Quality Verification.
To ensure dataset quality in terms of reasoning coherence, grounding, and QA accuracy, we employ Gemini-2.5-Pro to evaluate each chain on a 1-5 scale along four dimensions: factual correctness, step necessity, clarity, and multimodal alignment (Appendix F). The dataset achieves an overall average score of 4.87, and low-scoring samples are further refined through targeted answer editing, yielding a high-quality benchmark for search-enhanced multimodal reasoning.
this section cite: []

Section: EVALUATION PROTOCOL
To evaluate multimodal search-enhanced reasoning, we believe that measuring only final answer accuracy is insufficient, especially for long reasoning trajectories where errors in retrieval or planning may not be reflected in the final output. To enable a more fine-grained assessment, we introduce metrics that capture both answer correctness and the fidelity of step-wise retrieval and planning.
Answer Accuracy. We use four complementary metrics: (i) F1, the standard token-level overlap score between the generated final answer and the golden answer; (ii) ∆F1, which measures the performance gain brought by agentic MM-RAG beyond the model's parametric knowledge, defined as the difference between performance with and without retrieval:
∆F1 = F1 -F1 w/o R ; (iii) Golden F1
, an upper bound obtained by supplying the model with gold reasoning chains and golden retrieval content; and (iv) LLM-as-a-Judge (LJ), where a strong reasoning model (Gemini-2.5-Pro) evaluates predicted reasoning chains against the gold ones along four dimensions: answer accuracy, reasoning coherence, key knowledge entity coverage, and step alignment (prompt in Appendix P).
Chain alignment. We evaluate how well the predicted reasoning graph Ĝ aligns with the golden reasoning graph G. Each reasoning graph is a sequence of retrieval-augmented reasoning steps indexed by t, where r t ∈ G denotes the evidence retrieved at golden step t, and rt ′ ∈ Ĝ denotes the evidence retrieved at predicted step t ′ .
To assess step-level retrieval fidelity, we define the (i) Hit per Step (HPS) metric as the fraction of golden steps whose evidence is successfully recovered in the predicted graph:
HPS( Ĝ, G) = 1 |G| { (t, t ′ ) | r t ∈ G, rt ′ ∈ Ĝ, rt ′ = r t } ,(4)
where |G| is the number of golden steps and | • | on the right is the number of matched steps. Each "hit" corresponds to a predicted step that exactly matches the evidence of a golden step, with duplicates counted only once. Beyond exact matching, we further align Ĝ and G through maximum-weight bipartite matching, where nodes correspond to steps in the two graphs and edge weights reflect evidence similarity between predicted and golden steps. This alignment enables fine-grained step-wise comparison and better handles cases involving parallel reasoning steps.
To capture structural fidelity, we measure the step-length gap between the predicted and golden reasoning graphs. A large gap indicates under-or over-retrieval, while a small value suggests closer alignment in reasoning complexity. We refer to this metric as (ii) Rollout Deviation (RD):
RD( Ĝ, G) = | Ĝ| -|G| .(5)
this section cite: []

Section: AGENTIC MM-RAG PIPELINE AND PROCESS-LEVEL ALIGNMENT
To systematically evaluate and improve MLLMs in search-enhanced reasoning, we introduce a unified agentic MM-RAG pipeline for adaptive planning, retrieval for multimodal evidence, and reasoning over evidence, together with SEARCH-ALIGN, a process-level alignment framework that leverages verified reasoning chains to strengthen open-source models.
this section cite: []

Section: AGENTIC MM-RAG PIPELINE
As illustrated in Figure 2 (Right), our pipeline models multimodal search-enhanced reasoning as an iterative process with three modules: (i) sub-query and action generation, (ii) evidence acquisition, and (iii) iterative reasoning and synthesis. At each iteration, the agent formulates a sub-goal, selects a modality-aware retrieval action, integrates the retrieved evidence, and decides whether to continue searching or terminate with a final answer. This design supports not only effective reasoning but also fine-grained, step-level evaluation for agentic RAG.
(i) Sub-query and Action Generation. Given a complex question, the MLLM first predicts the next sub-goal and generates a focused sub-query (steps 1-2 in Figure 2). It then adaptively chooses one of three retrieval actions, corresponding to different ways of accessing multimodal evidence: (a) text search with a text query, (b) image search with a text query, or (c) image search with an input image.
(ii) Evidence Acquisition. The selected action is executed over our local multimodal knowledge base. The sub-query is embedded by a multimodal encoder for dense retrieval in the corresponding modality, returning textual or visual evidence. For each sub-query, we keep the top-1 evidence by query-answer similarity (step 3) and let the MLLM generate a sub-answer (steps 4-5) after reasoning. Both the chosen modality and retrieved evidence are logged for process evaluation.
this section cite: []

Section: (iii) Iterative Reasoning and Synthesis.
The sub-answer and its evidence are fed back into the model to guide the next planning step (step 6), forming a loop of planning, retrieval, and reasoning. The agent dynamically checks whether the accumulated evidence is sufficient; if so, it outputs a final response (step 7). Otherwise, it continues with another sub-query. This iterative design exposes the full reasoning trajectory, enabling modality-aware execution and precise chain-level evaluation.
this section cite: []

Section: SEARCH-ALIGN: PROCESS-LEVEL ALIGNMENT
Beyond evaluation, our dataset also serves as a resource for improving model capabilities. We introduce SEARCH-ALIGN, a conversation-level alignment framework for open-source MLLMs built on HAVE-verified reasoning chains. Unlike conventional SFT that supervises only final answers, SEARCH-ALIGN provides process-level supervision by leveraging step-wise annotated trajectories with sub-questions, retrieval actions, evidence, and intermediate answers.
Training Data Construction. Each reasoning graph is augmented with explicit reasoning thoughts generated by Gemini-2.5-Flash, which explain how to ground reasoning in evidence and connect adjacent hops. This converts step-wise annotations into coherent dialogue traces: the assistant poses sub-questions and reasons over evidence, while the user executes retrieval actions and returns results.
this section cite: []

Section: Supervised Fine-Tuning (SFT).
We then fine-tune MLLMs on these conversation-style traces, enabling them to learn not only to produce correct answers but also to plan, choose retrieval modalities, and integrate evidence across steps. This process-level alignment provides a richer training signal that better prepares models for long-horizon multimodal reasoning.
this section cite: []

Section: EXPERIMENTS
In this section, we evaluate leading MLLMs on the MC-SEARCH benchmark under our unified agentic MM-RAG pipeline, focusing on five research questions (RQs): RQ1: How do MLLMs perform across reasoning structures, and does SEARCH-ALIGN improve open-source models? RQ2: How does reasoning chain length affect reasoning performance? RQ3: What are the effects of overand under-retrieval? RQ4: Do models show inherent modality preferences? RQ5: What are the typical error types in multimodal agentic reasoning?
this section cite: []

Section: EXPERIMENTAL SETUP
We evaluate the following MLLMs using the agentic MM-RAG pipeline introduced in Section 3.1:
• Closed-source models: GPT-4o-Mini (Achiam et al., 2023), Gemini-2.5-Flash (Team et al., 2023), Gemini-2.5-Pro (Team et al., 2023), and Claude-3.7-Sonnet (Anthropic, 2024), representing the strongest commercially available reasoning-oriented MLLMs.
• Open-source models: InternVL3.5-8B (Wang et al., 2025b) and Qwen2.5-VL-7B (Bai et al., 2025), two state-of-the-art vision-language models. We also evaluate their performance after using SEARCH-ALIGN framework.
Evaluation protocol. All models are evaluated under identical conditions. We use the same embeding model for multimodal retrieval (Zhou et al., 2024), prompts (Appendix O), and decoding parameters across models. Each retrieval step is restricted to the top-1 result from the local multimodal knowledge base, and the same maximum number of reasoning iterations is enforced. Performance is reported using the six metrics in Section 2.2: answer accuracy (F1, ∆F1, Golden F1, and LLM-as-a-Judge) and chain-level retrieval and planning fidelity (Hit per Step and Rollout Deviation). Additional details on evaluation and the training of SEARCH-ALIGN are provided in Appendix I. In the Appendix D.1, we additionally report fixed-step RAG baselines (1-hop and 2-hop) for the backbone models to complement the main results. Backbone Performance and Search-Align Gains. As shown in Table 3, proprietary models such as Gemini-2.5-Pro achieve the highest overall accuracy across most reasoning topologies. Among open-source models, InternVL3.5-8B is the stronger backbone, while Qwen2.5-VL-7B consistently lags behind. With SEARCH-ALIGN, however, both backbones see substantial gains: InternVL improves by about +2.8 F1 and +12.0 HPS on average while reducing RD by 0.6, and QwenVL achieves even larger boosts of +13.7 F1 and +16.0 HPS with a -3.1 drop in RD. After alignment, Qwen2.5-VL-7B nearly matches Gemini-2.5-Pro on text-centric chains, and InternVL shows stable improvements across all topologies. These results indicate that a major weakness of open-source models lies in retrieval planning and step-level reasoning alignment rather than basic perceptual or semantic understanding, and that these capabilities can be effectively improved through training on high-quality search-enhanced reasoning data. Over-Retrieval Under-Retrieval Figure 4: Model ∆F1 vs. ∆ Step (difference in length between generated and golden reasoning chains), where larger positive ∆steps indicate more over-retrieval.
Topology-Specific Challenges. Across reasoning topologies, we observe Parallel Image-Text Fork is the most challenging, where all backbones reach their lowest F1 and HPS. The difficulty comes from the need to plan and cover both text and image branches simultaneously, so missing either branch directly lowers HPS, and under the top-1 retrieval constraint models often fail to capture the full evidence set. Aligning parallel evidence without strict order further increases the risk of incomplete or inconsistent reasoning. Multi-Images Fork is also difficult, mainly due to the challenge of aligning multiple visual evidences with corresponding textual facts, where models often retrieve the right images but misattribute or fail to integrate them coherently. In contrast, Text-Initiated and Text-Only Chains are relatively easier, as they rely primarily on text retrieval and linear reasoning, which align with model strengths. Image-Initiated Chains are also tractable: once the initial visual step is correct, subsequent textual validation is linear and more stable than multi-branch reasoning.
this section cite: ['b0', 'b32', 'b32', 'b2', 'b3', 'b45']

Section: ANALYSIS
RQ2: Performance vs. Chain Length. Figure 3 reports model F1 across samples with different chain lengths, showing a consistent drop in performance as the chain length increases. Models remain relatively robust on short reasoning chains (1-3 hops), where performance differences across backbones are moderate, with the exception of Qwen2.5-VL-7B, which lags behind consistently even at early hops. When the chain length extends to 4 or 5 hops, however, all models experience a sharp degradation, reflecting the compounding difficulty of sustaining long-horizon reasoning.
Among closed-source systems, Gemini-2.5-Pro is the most robust, sustaining the highest performance at longer hops. Gemini-2.5-Flash and InternVL3.5-8B degrade gradually, while GPT-4o-Mini and Claude-3.7-Sonnet drop sharply beyond 3 hops. Qwen2.5-VL-7B remains weakest across all lengths. Overall, the main challenge lies in extended multi-step chains, where compounding retrieval errors and unstable planning reduce accuracy.
this section cite: []

Section: RQ3: Over-and Under-Retrieval Analysis.
Figure 4 shows ∆F1 across different ∆Steps, where negative values correspond to under-retrieval and positive values indicate over-retrieval. The results reveal that both extremes are harmful, though in different ways. When models severely under-retrieve (∆Step < -2), performance drops sharply as essential evidence is skipped, creating gaps that later reasoning cannot recover. Conversely, moderate over-retrieval (∆Step = 1-2) often improves accuracy: Gemini-2.5-Pro, Gemini-2.5-Flash, and InternVL3.5-8B achieve their highest ∆F1 here, suggesting that one or two extra turns can help recover from imperfect planning.
However, when over-retrieval is excessive (∆Step ≥ 4), noise and irrelevant context lead to sharp drops across all models. Gemini-2.5-Pro is most resilient, Gemini-2.5-Flash and InternVL3.5-8B remain moderately stable, while Claude-3.7-Sonnet and GPT-4o-Mini are highly vulnerable. This underscores a central tension between capturing sufficient evidence and avoiding over-retrieval.
this section cite: []

Section: RQ4: Modality Coverage Analysis.
As shown in Table 4, which reports step-level retrieval coverage (covered vs. gold steps) for both image and text modalities under queries with and without image input, both models maintain strong text coverage (>78%) across all settings, but image coverage is far weaker and highly dependent on explicit image inputs. For example, Gemini-2.5-Pro drops from 87.35% with image mentions to 29.50% without, while InternVL3.5-8B falls from 63.84% to 0.66%. These findings expose a pronounced modality gap, as models tend to default to text retrieval and lose visual grounding in the absence of explicit image cues.  Mid-frequency errors (e.g., modality mismatch, spurious steps) reflect unstable planning, while structural errors (order/dependency mistakes, multi-hop failures) are rarer but, given limited retrieval, remain non-trivial for future study. Evidence misinterpretation is infrequent, suggesting models can handle retrieved context; major failures arise earlier, during planning and evidence acquisition. Overall, these patterns, together with our HPS and over/under-retrieval analyses, identify retrieval fidelity (modality-aware retrieval, sufficient planning/hop coverage, and calibrated stopping) as the central bottleneck for reliable chain-level reasoning.
this section cite: []

Section: RELATED WORK
We review existing work on MM-RAG benchmarks and Agentic RAG, with full details in Appendix B.
Multimodal RAG Benchmarks. MM-RAG extends retrieval-augmented generation by incorporating image evidence for cross-modal reasoning (Xia et al., 2024;Chang et al., 2022). Existing benchmarks are limited: most use fixed retrieve-then-generate pipelines (Marino et al., 2019;Hu et al., 2025), reduce images to captions (Lerner et al., 2022;Chen et al., 2023), restrict reasoning to 1-2 hops (Li et al., 2025b;Liu et al., 2025c), and lack step-wise annotations, making it hard to assess multimodal search-enhanced reasoning. We introduce MC-SEARCH, with HAVE-verified long reasoning trajectories across diverse topologies, enabling fine-grained attribution and analysis.
Agentic RAG. Recent work reframes retrieval as sequential decision-making, where agentic RAG systems decompose queries, adaptively retrieve, and synthesize knowledge (Wang et al., 2025a;Li et al., 2025a;Zheng et al., 2025b). While effective, most remain text-only and focus on multihop QA or scientific domains (Trivedi et al., 2022;Rein et al., 2024). Our benchmark fills this gap by extending agentic RAG to multimodal settings with diverse, and step-wise verified reasoning chains.
this section cite: ['b38', 'b4', 'b27', 'b11', 'b18', 'b6', 'b33', 'b29']

Section: CONCLUSION
We present MC-SEARCH, a benchmark for structured, step-wise multimodal retrieval-augmented reasoning, encompassing five diverse reasoning mechanisms. With fine-grained annotations, hop-wise attribution and verification, and new chain-level metrics, MC-SEARCH enables thorough diagnosis of retrieval planning and reasoning quality. Our analyses highlight the importance of adaptive, modalityaware search strategies in agentic MM-RAG systems. Looking ahead, we envision MC-SEARCH as a foundation for advancing multimodal agents and fostering principled evaluation standards for agentic reasoning. we will broaden evaluations to stronger reasoning models and extend the benchmark to additional domains such as science and mathematics.
Appendix Contents A Use of Large Language Models B Related Work B.1 Multimodal RAG Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . B.2 Agentic Search-Enhanced Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . B.3 Structure-Aware Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C Data Example C.1 Image-Initiated Chain . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.2 Text-Initiated Chain . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.3 Text Chain . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.4 Parallel Visual-Textual Fork . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.5 Multi-Images Fork . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . D Baseline RAG Performance D.1 Single-Step RAG Baseline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . D.2 Traditional Multi-modal RAG Baseline . . . . . . . . . . . . . . . . . . . . . . . . . E Dataset Statistics F Dataset Quality Verification G Cross-Model and Human Consistency Analysis G.1 Cross-Model LLM-as-Judge Consistency . . . . . . . . . . . . . . . . . . . . . . G.2 Human and LLM-as-Judge Agreement . . . . . . . . . . . . . . . . . . . . . . . . G.3 Cross-Model Ground-Truth Consistency . . . . . . . . . . . . . . . . . . . . . . . H Fine-Grained Improvement Analysis of SEARCH-ALIGN I SEARCH-ALIGN Fine-tuning Details J Soft HPS Evaluation with Thresholded Semantic Matching K Evaluation under Top-k Retrieval K.1 Under-and Over-Retrieval under Retrieval@K . . . . . . . . . . . . . . . . . . . K.2 SEARCH-ALIGN under Retrieval@K . . . . . . . . . . . . . . . . . . . . . . . . L Agentic MM-RAG Case Study L.1 Success Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . L.2 Failure Case . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . M Theoretical Justification of Topology Designs M.1 Problem Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . M.2 Proof by Exhaustion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . M.3 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . N Prompt for Data Construction N.1 Image-Initiated Chain . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . N.2 Parallel Image-Text Fork . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . N.3 Text-Initiated Chain . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . N.4 Multi-Images Fork . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . N.5 Text Chain . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O Prompt Template for Agentic MM-RAG Pipeline P Prompt Template for LLM-as-Judge
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Evaluating the robustness of retrieval-augmented generation to adversarial evidence in the health domain Year: (2025)
Ref_id:b2 Title: Introducing the next generation of claude Year: (2024)
Ref_id:b3 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b4 Title: Webqa: Multihop and multimodal qa Year: (2022)
Ref_id:b5 Title: Learning to reason with search for llms via reinforcement learning Year: (2025)
Ref_id:b6 Title: Can pre-trained vision and language models answer visual information-seeking questions Year: (2023)
Ref_id:b7 Title: A survey on RAG meeting llms: Towards retrieval-augmented large language models Year: (2024)
Ref_id:b8 Title: Scalable multi-hop relational reasoning for knowledge-aware question answering Year: (2020)
Ref_id:b9 Title: Ragbench: Explainable benchmark for retrievalaugmented generation systems Year: (2024)
Ref_id:b10 Title: Retrieval-augmented generation for large language models: A survey Year: ()
Ref_id:b11 Title: MRAG-bench: Vision-centric evaluation for retrieval-augmented multimodal models Year: (2025)
Ref_id:b12 Title: Rag-rl: Advancing retrieval-augmented generation via rl and curriculum learning Year: (2025)
Ref_id:b13 Title: Vision-r1: Incentivizing reasoning capability in multimodal large language models Year: (2025)
Ref_id:b14 Title: Benchmarking the potential of large models as multi-modal search engines Year: (2024)
Ref_id:b15 Title: Search-r1: Training llms to reason and leverage search engines with reinforcement learning Year: (2025)
Ref_id:b16 Title: Robust multi model rag pipeline for documents containing text, table & images Year: (2024)
Ref_id:b17 Title: Agent-g: An agentic framework for graph retrieval augmented generation Year: ()
Ref_id:b18 Title: Viquae, a dataset for knowledge-based visual question answering about named entities Year: (2022)
Ref_id:b19 Title: Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models Year: (2024)
Ref_id:b20 Title: Search-o1: Agentic search-enhanced large reasoning models Year: (2025)
Ref_id:b21 Title: Benchmarking multimodal retrieval augmented generation with dynamic VQA dataset and self-adaptive planning agent Year: ()
Ref_id:b22 Title: Retrieval augmented visual question answering with outside knowledge Year: (2022)
Ref_id:b23 Title: Alexandru Coca, and Bill Byrne. Fine-grained lateinteraction multi-modal retrieval for retrieval augmented visual question answering Year: (2023)
Ref_id:b24 Title: Symagent: A neural-symbolic self-learning agent framework for complex reasoning over knowledge graphs Year: (2025)
Ref_id:b25 Title: Benchmarking retrieval-augmented generation in multi-modal contexts Year: (2025)
Ref_id:b26 Title: Benchmarking retrieval-augmented generation in multi-modal contexts Year: (2025)
Ref_id:b27 Title: Ok-vqa: A visual question answering benchmark requiring external knowledge Year: (2019)
Ref_id:b28 Title: Chain-of-action: Faithful and multimodal question answering through large language models Year: (2024)
Ref_id:b29 Title: Gpqa: A graduate-level google-proof q&a benchmark Year: (2024)
Ref_id:b30 Title: Gear: Graph-enhanced agent for retrieval-augmented generation Year: (2024)
Ref_id:b31 Title: Agentic retrieval-augmented generation: A survey on agentic rag Year: (2025)
Ref_id:b32 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b33 Title: musique: Multihop questions via single-hop question composition Year: (2022)
Ref_id:b34 Title: Chainof-retrieval augmented generation Year: (2025)
Ref_id:b35 Title: Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency Year: (2025)
Ref_id:b36 Title: Deep inductive logic reasoning for multi-hop reading comprehension Year: (2022)
Ref_id:b37 Title: Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning Year: (2024)
Ref_id:b38 Title: Mmed-rag: Versatile multimodal rag system for medical vision language models Year: (2024)
Ref_id:b39 Title: Hotpotqa: A dataset for diverse, explainable multi-hop question answering Year: (2018)
Ref_id:b40 Title: Learning to retrieve and reason on knowledge graph through active self-reflection Year: (2025)
Ref_id:b41 Title: Retrieval-augmented generation for ai-generated content: A survey Year: (2024)
Ref_id:b42 Title: Bin Ren, Danda Paudel, Nicu Sebe, Luc Van Gool, and Xuming Hu. Retrieval augmented generation and understanding in vision: A survey and new outlook Year: (2025)
Ref_id:b43 Title: Unified efficient fine-tuning of 100+ language models Year: (2024)
Ref_id:b44 Title: Scaling deep research via reinforcement learning in real-world environments Year: (2025)
Ref_id:b45 Title: Megapairs: Massive data synthesis for universal multimodal retrieval Year: (2024)
