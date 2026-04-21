Title: KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment
Abstract: Maintaining comprehensive and up-to-date knowledge graphs (KGs) is critical for modern AI systems, but manual curation struggles to scale with the rapid growth of scientific literature. This paper presents KARMA, a novel framework employing multi-agent large language models (LLMs) to automate KG enrichment through structured analysis of unstructured text. Our approach employs nine collaborative agents, spanning entity discovery, relation extraction, schema alignment, and conflict resolution that iteratively parse documents, verify extracted knowledge, and integrate it into existing graph structures while adhering to domain-specific schema. Experiments on 1,200 PubMed articles from three different domains demonstrate the effectiveness of KARMA in knowledge graph enrichment, with the identification of up to 38,230 new entities while achieving 83.1% LLM-verified correctness and reducing conflict edges by 18.6% through multi-layer assessments.

Section: Introduction
Knowledge graphs (KGs) are essential for structuring and reasoning over complex information across diverse fields [9,11,18]. By encoding entities and their relationships in machine-readable formats, widely adopted KGs such as Wikidata [26] and DBpedia [13] have become foundational to both industry and academic research. Yet, the exponential growth of scientific literature, with over 7 million articles published annually [2], exposes a significant bottleneck: the widening gap between unstructured knowledge in texts and its structured representation in KGs.
The challenge of enriching KGs becomes even more apparent in fields with complex and specialized terminology, such as healthcare, finance, or autonomous systems. Traditional approaches to KG enrichment, such as manual curation, are reliable but unsustainable at scale. Automated methods based on conventional natural language processing (NLP) techniques often struggle to handle domainspecific terminology and context-dependent relationships found in scientific and technical texts [22]. Moreover, extracting and integrating knowledge into existing KGs requires robust mechanisms for schema alignment, consistency, and conflict resolution [5]. In high-stakes applications, the costs of inaccuracies in these systems can be severe.
Recent advances in large language models (LLMs) [7,1,15] have demonstrated remarkable improvements in contextual understanding and reasoning [28]. Building on these advances, the research community has increasingly explored multi-agent systems, where several specialized agents work in concert to tackle complex tasks [8]. These systems harness the strengths of individual agents, each optimized for a particular subtask, and enable cross-agent verification and iterative refinement of outputs. Such multi-agent frameworks have shown promise in areas ranging from decision-making to structured data extraction [6,19], offering robustness through redundancy and collaboration. However,
this section cite: ['b8', 'b10', 'b17', 'b25', 'b12', 'b21', 'b4', 'b6', 'b14', 'b27', 'b7', 'b5', 'b18']

Section: Article

this section cite: []

Section: Knowledge

this section cite: []

Section: KG Enrichment
Multi-Agent Multi-Agent directly applying these systems to KG enrichment remains challenging due to issues like domain adaptation, systematic verification requirements [10], and the complexity of integrating outputs into heterogeneous knowledge structures.
In this paper, we propose KARMA, a novel multi-agent framework that harnesses LLMs through a collaborative system of specialized agents (Figure 1). Each agent focuses on distinct tasks in the KG enrichment pipeline. Our framework offers three key innovations. First, the multi-agent architecture enables cross-agent verification, enhancing the reliability of extracted knowledge. For instance, Relationship Extraction Agents validate candidate entities against Schema Alignment outputs, while Conflict Resolution Agents resolve contradictions through LLM-based debate mechanisms. Second, domain-adaptive prompting strategies allow the system to handle specialized contexts while preserving accuracy. Third, the modular design ensures extensibility and supports dynamic updates as new entities or relationships emerge. Through proof-of-concept experiments on datasets from three distinct domains, we demonstrate that KARMA can efficiently extract high-quality knowledge from unstructured texts, substantially enriching existing knowledge graphs with both precision and scalability.
2 Related Work
this section cite: ['b9']

Section: Knowledge Graph Construction
The quest to transform unstructured text into structured knowledge has evolved through three generations of technical paradigms. First-generation systems (1990s-2010s) like WordNet [21] and ConceptNet [16] relied on hand-crafted rules and shallow linguistic patterns, achieving high precision at the cost of limited recall and domain specificity. The neural revolution (2010s-2022) introduced learned representations through architectures like BioBERT [12] and SapBERT [17], which achieved improvements on biomedical NER through domain-adaptive pretraining. However, these methods require expensive supervised tuning (3-5k labeled examples per relation type [30]) and fail to generalize beyond predefined schema, which is a critical limitation when processing novel scientific discoveries. The current LLM-powered generation (2022-present) attempts to overcome schema rigidity through instruction tuning [24,31]. This progression reveals an unresolved tension: neural methods scale better than rules but require supervision, while LLMs enable open schema learning at the cost of verification mechanisms. LLMs have shown promise in open-domain KG construction through their inherent reasoning capabilities. However, these approaches exhibit critical limitations:
(1) Hallucination during extracting complex relationships [20], (2) Inability to maintain schema consistency across documents [29], and (3) Quadratic computational costs when processing full-text articles [23].
this section cite: ['b20', 'b15', 'b11', 'b16', 'b29', 'b23', 'b30', 'b19', 'b28', 'b22']

Section: Multi-Agent Systems
Early multi-agent systems focused on distributing subtasks across specialized modules, such as separate agents for named entity recognition and relation extraction [3]. These systems relied on predefined pipelines and handcrafted coordination rules, limiting adaptability to new domains.
Recent advances in LLMs have enabled more dynamic architectures and rediscovered multi-agent collaboration as a mechanism for enhancing LLM reliability [25,19]. Building on classic blackboard architectures, contemporary systems like AutoGen [28] show that task decomposition with specialized agents reduces hallucination compared to monolithic models. For knowledge graph construction, [14] demonstrated that task decomposition across specialized agents (e.g., entity linker, relation validator)
Introduction ......
this section cite: ['b24', 'b18', 'b27', 'b13']

Section: Results

this section cite: []

Section: Summary
This segment describes the use of IL-6 inhibition in ……
this section cite: []

Section: Summary
This segment describes the use of IL-6 inhibition in …… 0.85 improves schema alignment on Wikidata benchmarks. maintaining linear time complexity relative to input text length.
KARMA synthesizes insights from these research threads while introducing key innovations: (1) a modular, multi-agent architecture that allows for specialized handling of complex tasks in knowledge graph enrichment, (2) domain-adaptive prompting strategies that enable more accurate extraction across diverse scientific fields, (3) LLM-based verification mechanisms that mitigate issues such as hallucination and schema inconsistency.
this section cite: []

Section: Methodology
In this section, we introduce KARMA, a hierarchical multi-agent system (see Figure 2) that leverages specialized LLMs to perform end-to-end KG enrichment. Our approach decomposes the overall task into modular sub-tasks, ranging from document ingestion to final KG integration, each handled by an independent LLM-based agent. We first present a formal problem formulation and then detail the design and mathematical foundations of each agent within the pipeline.
this section cite: []

Section: Problem Formulation
Let G = (V, E) denote an existing KG, where V is the set of entities (e.g., genes, diseases, drugs) and E the set of directed edges representing relationships. Each relationship is defined as a triplet t = (e h , r, e t ) with e h , e t ∈ V and r specifying the relation type (e.g., treats, causes). We are provided with a corpus of unstructured publications P = p 1 , . . . , p n . The objective is to automatically extract novel triplets t / ∈ E from each document p i and integrate them into G to form an augmented graph G new .
Gnew = G ∪ n i=1 Ki, where Ki = Extract(pi),(1)
where Extract(p i ) is the set of valid triplets obtained from publication p i . To maintain consistency and accuracy, each candidate triplet is evaluated by an LLM-based verifier prior to integration.
this section cite: []

Section: System Overview
KARMA comprises multiple LLM-based agents operating in parallel under the orchestration of a Central Controller. Each agent uses specialized prompts, hyper-parameters, and domain knowledge to optimize its performance. In KARMA, we define a set of agents (B):
• Ingestion Agents (IA): Retrieve and normalize input documents (B.3).
• Reader Agents (RA): Parse and segment relevant text sections (B.4).
• Summarizer Agents (SA): Condense relevant sections into shorter domain-specific summaries (B.5).
• Entity Extraction Agents (EEA): Identify and normalize topic-related entities (B.6).
• Relationship Extraction Agents (REA): Infer relationships between entities (B.7).
• Schema Alignment Agents (SAA): Align entities and relations to KG schemas (B.8).
• Conflict Resolution Agents (CRA): Detect and resolve logical inconsistencies with existing knowledge (B.9).
• Evaluator Agents (EA): Aggregate multiple verification signals and decide on final integration (B.10,B.11,B.12).
this section cite: []

Section: Ingestion Agents (IA)
The Ingestion Agents are LLM-based modules specialized in document retrieval, format normalization, and metadata extraction. Let p i be a raw publication. IA includes:
IA(pi) = normalize(pi), metadata(pi) ,(2)
where normalize(p i ) uses an LLM prompt P ingest to handle complexities like OCR errors, or structural inconsistencies. The output is a standardized textual representation plus key metadata (journal, date, authors, etc.). This representation is then placed into a data queue for Reader Agents.
this section cite: []

Section: Reader Agents (RA)
Reader Agents parse normalized text into coherent segments (abstract, methods, results, ect.) and filter out irrelevant content. Let p ′ i be the normalized document. RA splits
p ′ i into {s 1 , s 2 , . . . , s mi }. Each segment s j is assigned a relevance score R(s j ) by: R(sj) = LLM reader sj, G ,(3)
where LLM reader is prompted with domain-specific instructions to assess the segment's biomedical significance relative to the current KG G. RA discards segments if R(s j ) < δ, where δ is a domaincalibrated threshold. Surviving segments are passed along to Summarizer Agents.
this section cite: []

Section: Summarizer Agents (SA)
To reduce computational overhead, each RA segment s j is condensed by Summarizer Agents into a concise representation u j . Formally, we define:
uj = LLMsumm sj, Psumm ,(4)
where P summ is a prompt for LLM to retain critical entities, relations, and domain-specific terms. This summarization ensures Entity Extraction Agents and Relationship Extraction Agents receive textual inputs that are both high-signal and low-noise.
this section cite: []

Section: Entity Extraction Agents (EEA)
LLM-Based NER. Each summary u j is routed to an LLM-based NER pipeline that identifies mentions of topic-related entities. Define:
E(uj) = LLME uj, PE ⊙ DE,(5)
where LLM E is an specialized entity-extraction LLM with prompt P E , and ⊙ D E indicates a dictionary/ontology-based filtering. This step filters out false positives and normalizes entity mentions to canonical forms (e.g., mapping "acetylsalicylic acid" to "Aspirin").
this section cite: []

Section: Entity Normalization.
Let e be a raw entity mentioned from E(u j ). We map e to a normalized entity ê ∈ V by minimizing a distance function in a joint embedding space:
ê = arg min v∈V d ϕ(e), ψ(v) ,(6)
where ϕ maps textual mentions to embeddings (using, e.g., a BERT-based model), and ψ maps known KG entities to the same embedding space. The distance metric d(•, •) can be cosine distance or a domain-specific measure. Any entity with min v∈V d(ϕ(e), ψ(v)) > ρ is flagged as new and added to the set of candidate vertices V + .
this section cite: []

Section: Relationship Extraction Agents (REA)
After entity normalization, each pair (ê i , êj ) within summary u j is fed to an LLM-based classifier:
p(r | êi, êj, uj) = LLMR êi, êj, uj, PR ,(7)
where p(r|•) is the probability distribution over possible relationships r ∈ {r 1 , . . . , r K }. The prompt P R instructs the LLM to focus on domain relationship candidates. We select any relationship r for which p(r|ê i , êj ) ≥ θ r and form a triplet (ê i , r, êj ). In certain passages, more than one relationship can be implied. We allow multi-label predictions by setting an indicator variable:
I(r) = I{p(r | êi, êj) ≥ θr},(8)
Hence, R(u j ) is the set of triplets (ê i , r, êj ) such that I(r) = 1.
this section cite: []

Section: Schema Alignment Agents (SAA)
If a new entity v ∈ V + or a new relation r does not match existing KG types, the Schema Alignment Agent performs a domain-specific classification. For entities, the SAA solves:
τ * = arg max τ ∈T LLMSAA v, τ, P align ,(9)
where T is the set of valid entity types (Disease, Drug, Gene, etc.), and LLM SAA estimates the probability that v belongs to type τ . A similar approach is used for mapping new relation r to known KG relation types. If no suitable match exists, the SAA flags v or r as candidate additions for review.
this section cite: []

Section: Conflict Resolution Agents (CRA)
New triplets can contradict previously established relationships. Let t = (ê h , r, êt ) be a newly extracted triplet, and let t ′ = (ê h , r ′ , êt ) be a conflicting triplet in G if r is logically incompatible with r ′ . We define:
conflict(t, G) = 1, if ∃ t ′ that contradicts t, 0, otherwise. (10
)
The CRA uses an LLM-based debate prompt:
LLMCRA t, t ′ → {Agree, Contradict},(11)
If LLM CRA yields Contradict, t is then discarded or queued for manual expert review, depending on the system's confidence.
this section cite: []

Section: Evaluator Agents (EA)
Finally, the Evaluator Agents aggregate multiple verification signals and compute global confidence C(t), clarity Cl(t), and relevance R(t) for each triplet t.
Confidence:
C(t) = σ αivi(t) ,(12)
Clarity:
Cl(t) = σ βjcj(t) , (13
) Relevance: R(t) = σ γ k r k (t) ,(14)
where σ(x) = 1 1+e -x and {α i , β j , γ k } reflect the trustworthiness of each verification source, and v i , c j , r k are verification signals for confidence, clarity, and relevance respectively. We finalize t for integration using the mean score:
integrate(t) = 1, if C(t)+Cl(t)+R(t) 3 ≥ Θ 0, otherwise.(15)
Altogether, this multi-agent pipeline, fully powered by specialized LLMs in each stage, enables robust, scalable, and accurate enrichment of large-scale KG. Future extensions can easily incorporate new domain ontologies, additional specialized agents, or updated LLM prompts as tasks continues to evolve.
this section cite: []

Section: Experimental Setup
This section presents a comprehensive proof-of-concept evaluation settings of the proposed KARMA framework. Unlike conventional NLP tasks that rely on a gold-standard dataset of biomedical entities and relationships, our evaluation adopts a multi-faceted approach. We integrate LLM-based verification with specialized graph-level metrics to assess the quality of the generated knowledge graph. The evaluation spans genomics, proteomics, and metabolomics, showcasing KARMA's adaptability across diverse biomedical domains.
this section cite: []

Section: Data Collection
We curate scientific publications from PubMed [27] across three primary domains: the Genomics Corpus, which includes 720 papers focused on gene variants, regulatory elements, and sequencing studies; the Proteomics Corpus, comprising 360 papers related to protein structures, functions, and protein-interaction networks; and the Metabolomics Corpus, containing 120 papers discussing metabolic pathways, metabolite profiling, and clinical applications. All articles are stored in PDF format and processed by the Ingestion Agent within KARMA.
this section cite: ['b26']

Section: LLM Backbones
We evaluate three general-purpose LLMs as the backbone for KARMA's multi-agent knowledge graph enrichment pipeline using their APIs.
this section cite: []

Section: GLM-4 [7]: An open-source 9B-parameter model, achieving 72.4 on the MMLU NLP benchmark.
GPT-4o [1]: A proprietary multimodal model optimized through RLHF. It has demonstrated strong adaptability in scientific knowledge extraction and concept grounding [4].
DeepSeek-v3
this section cite: []

Section: Metrics
To evaluate the enriched knowledge graph (KG) in the absence of a gold-standard reference, we employ a multi-faceted evaluation framework that assesses structural integrity, correctness, and practical utility. This framework comprises three categories of metrics: core metrics, graph statistics, and quality indicators. Together, these metrics provide comprehensive insights into the quality and usability of newly added triples and the overall augmented KG.
Core Metrics focus on the properties of newly added triples using structural and LLM-based indicators. The Average Confidence (M ↑ Con ) measures the mean confidence scores across all new triples, reflecting their reliability. The Average Clarity (M ↑ Cla ) computes the mean clarity scores, indicating how unambiguous or direct each relation is. The Average Relevance (M ↑ Rel ) captures the
M ↑ Con M ↑ Cla M ↑ Rel ∆ ↑ Cov ∆ ↓ Con R ↑ CR R ↑ LC C ↑ QA R ↑ HE Genomics Single-Agent NA NA NA4384
1.083 NA 0.493 0.472 0.320 GLM-4 0.729 0.804 0.716 4969 1.131 0.238 0.623 0.589 0.445 GPT-4o 0.843 0.744 0.640 9795 1.265 0.148 0.880 0.569 0.510 DeepSeek-v3 0.846 0.754 0.667 38230 1.765 0.186 0.831 0.612 0.625 Proteomics Single-Agent NA NA NA 5002 1.150 NA 0.638 0.572 0.415 GLM-4 0.731 0.752 0.609 6832 1.173 0.214 0.720 0.617 0.500 GPT-4o 0.823 0.797 0.613 7008 1.191 0.160 0.740 0.612 0.550 DeepSeek-v3 0.845 0.825 0.682 11936 1.468 0.151 0.772 0.613 0.575 Metabolomics Single-Agent NA NA NA 485 1.077 NA 0.527 0.450 0.455 GLM-4 0.701 0.790 0.762 703 1.159 0.188 0.617 0.449 0.485 GPT-4o 0.802 0.730 0.726 773 1.143 0.147 0.683 0.482 0.535 DeepSeek-v3 0.790 0.746 0.767 1752 1.811 0.132 0.668 0.493 0.580
mean relevance scores, assessing the domain significance of the triples. These metrics collectively evaluate the intrinsic quality of the added knowledge.
this section cite: []

Section: Graph Statistics quantify the structural properties of the augmented KG. The Coverage Gain (∆ ↑
Cov ) measures the number of newly introduced entities not previously in the KG, reflecting its expanded scope. The Connectivity Gain (∆ ↑ Con ) calculates the net increase in node degrees (summed over existing entities), indicating enhanced interconnectedness.
Quality Indicators assess reliability and usability through multiple lenses. The Conflict Ratio (R ↓ CR ) represents the fraction of newly extracted edges removed by the ConflictResolutionAgent due to internal or external contradictions. The LLM-based Correctness (R ↑ LC ) is determined by a hold-out LLM judging each new triple ((head, r, tail)) as likely correct, uncertain, or likely incorrect, with R LC = #(likely correct) #(all new triples) . The Question-Answer Coherence (C ↑ QA ) evaluates the fraction of plausible KG-derived answers for a curated set of domain-specific questions answerable via KG traversal. Finally, the Human Evaluation Score (R ↑ HE ) scaled from 0 to 1, gauges the quality of triple extractions based on assessments by two human experts, offering a comprehensive measure of the knowledge graph's accuracy and utility.
These complementary metrics provide insights into the structural integrity, internal consistency, correctness, and practical utility of the enriched knowledge graph.
this section cite: []

Section: Results

this section cite: []

Section: Overall Evaluation
Our comprehensive evaluation (Table 1, with examples in Appendix C.1,C.2,C.3) demonstrates that KARMA significantly extends domain-specific knowledge graphs through its multi-agent architecture. (4) Evaluating knowledge and resolving conflicts automatically can enhance the quality of the extracted knowledge graph, improving LLM-based accuracy by 4.6%-14.4%. 5.2 Domain-Level Observations. Genomics: Scale Meets Precision (C.1) The genomics domain (720 papers) exhibits the most pronounced model differentiation. DeepSeek-v3 achieves ∆ Cov = 38, 230 while maintaining a competitive correctness score R LC = 0.831, only 5.6% below GPT-4o's peak. This suggests that MoE architectures can balance recall and precision in large-scale extraction. Proteomics: Balanced Optimization (C.2) With 360 papers, proteomics reveals balanced gains: DeepSeek-v3 leads in both core metrics (M Con = 0.845) and structural gains (∆ Con = 1.468), while GLM-4 achieves peak QA coherence (C QA = 0.617). The 19.1% higher ∆ Cov for DeepSeek-v3 versus GPT-4o indicates greater sensitivity to protein interaction nuances.
Metabolomics: Specialization Pays Off (C.3) Despite the smallest corpus (120 papers), GLM-4 delivers superior clarity (M cla = 0.790) and GPT-4o excels in correctness (R LC = 0.683). However, DeepSeek-v3's ∆ Con = 1, 752 is 127% higher than GPT-4o, demonstrates unique capability to extrapolate metabolic pathways from limited data.
this section cite: []

Section: Analysis of LLM Backbones
Our comparison reveals strengths of different backbones: DeepSeek-v3 drives unparalleled coverage gains, outpacing GPT-4o by 3.9× in genomics and 2.3× in metabolomics while maintaining competitive correctness (R LC = 0.831 vs GPT-4o's 0.880 in genomics). This contrasts with GPT-4o's precision-first profile, where it achieves peak R LC scores (0.880 genomics, 0.740 proteomics) but yields 41% lower connectivity gains than DeepSeek-v3, reflecting underutilized implicit relationships. GLM-4, though smaller (10B parameters), demonstrates domain-specific prowess: its biomedical tuning delivers best-in-class metabolomics clarity (M Cla = 0.762) and proteomics QA coherence (C QA = 0.617), while its conflict ratio (R CR = 0.188) remains competitive despite lower parameter count. The tradeoffs (DeepSeek-v3's coverage balance for correctness, GPT-4o's precision sacrifice for completeness, GLM-4's niche adaptation) underscore why KARMA's multi-agent framework strategically decouples extraction, validation, and can utilize the strengths of each backbone. Different backbones also lead to variations in the distribution of key evaluation metrics (Figure C.1,C.2,C.3).
this section cite: []

Section: Cost Analysis
The evaluation of computational costs (Figure 3) demonstrates distinct trade-offs in token usage and processing time across different domains. The variations in article lengths and information density naturally lead to differences in token consumption and processing times. Notably, genomics shows higher completion token distributions (mean = 550.64, std = 232.92), explaining KARMA's higher ∆ Cov in this domain. Meanwhile, proteomics exhibits broader processing time distributions (mean = 96.58, std = 46.90), which correlates with its stronger performance in knowledge quality metrics (R LC and C QA ), suggesting that longer processing times contribute to more thorough relationship analysis and validation.
this section cite: []

Section: Ablation Study
To better quantify the contributions of each specialized agent in KARMA, we conduct an ablation study (Table 2) by systematically removing or replacing selected agents and measure the resulting performance across the three domains. Specifically, we evaluate:
• KARMA-Full: All agents active, including Summarizer, Conflict Resolution, and Evaluator modules.
• w/o Summarizer: Bypasses the Summarizer Agents, passing all text directly from Reader Agents to Entity and Relationship Extraction.
• w/o Conflict Resolution: Disables the Conflict Resolution Agent, allowing potentially contradictory edges into the final graph.
• w/o Evaluator: Omits the final confidence, clarity, and relevance evaluation and aggregation, integrating relationships without filtering.
We conduct these ablations using the same LLM backbone (DeepSeek-v3 in our experiments) for consistency. Table 2 summarizes the impact on evaluation metrics (R LC , C QA ) for each domain.
The ablation study highlights the importance of each agent in KARMA's performance. Removing the Summarizer Agent produce much more entities and triples, but reduces accuracy (C QA drop 22.9% (0.612 → 0.472) in genomics) and coherence (R LC drop 18.2% (0.772 → 0.632) in proteomics), as unfiltered text introduces noise. Disabling the Conflict Resolution Agent significantly lowers correctness (C QA drop 4.9% (0.831 → 0.790) in genomics), especially in resolving contradictions like conflicting gene-disease associations. Omitting the Evaluator Agents has the most impact on usability, as unfiltered, low-confidence edges degrade answer quality (R LC drop 9.7% (0.668 → 0.603) in metabolomics). Across all domains, conflict resolution proves critical for maintaining logical consistency, while summarization and evaluation ensure focused extraction and high-quality integration. This demonstrates that KARMA's multi-agent design is essential for balancing accuracy, consistency, and usability in KG enrichment.
this section cite: []

Section: Conclusion
We introduce KARMA, a multi-agent LLM framework designed to tackle the challenge of scalable knowledge graph enrichment from scientific literature. By decomposing the extraction process into specialized agents for entity discovery, relationship validation, and conflict resolution, KARMA ensures adaptive and accurate knowledge integration. Its modular design reduces the impact of conflicting edges through multi-layered assessments and cross-agent verification. Experimental results across genomics, proteomics, and metabolomics demonstrate that multi-agent collaboration can overcome the limitations of single-agent approaches, particularly in domains that require complex semantic understanding and adherence to structured schemas.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Growth rates of modern science: a latent piecewise growth curve approach to model publication numbers from established and new literature databases Year: (2021)
Ref_id:b2 Title: Multi-agent systems for natural language processing Year: (1998)
Ref_id:b3 Title: Structured information extraction from scientific text with large language models Year: (2024)
Ref_id:b4 Title: Ontology matching Year: (2007)
Ref_id:b5 Title: Magentic-one: A generalist multi-agent system for solving complex tasks Year: (2024)
Ref_id:b6 Title: A family of large language models from glm-130b to glm-4 all tools Year: (2024)
Ref_id:b7 Title: Large language model based multi-agents: A survey of progress and challenges Year: (2024)
Ref_id:b8 Title: Knowledge graphs Year: (2021)
Ref_id:b9 Title: Ai safety via debate Year: (2018)
Ref_id:b10 Title: A survey on knowledge graphs: Representation, acquisition, and applications Year: (2021)
Ref_id:b11 Title: Biobert: a pre-trained biomedical language representation model for biomedical text mining Year: (2020)
Ref_id:b12 Title: The contentious nature of soil organic matter Year: (2015)
Ref_id:b13 Title: Encouraging divergent thinking in large language models through multiagent debate Year: (2023)
Ref_id:b14 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b15 Title: Conceptnet-a practical commonsense reasoning tool-kit Year: (2004)
Ref_id:b16 Title: Self-supervised learning: Generative or contrastive Year: (2021)
Ref_id:b17 Title: Biomedical knowledge graph: A survey of domains, tasks, and real-world applications Year: (2025)
Ref_id:b18 Title: Clinicalrag: Enhancing clinical decision support through heterogeneous knowledge retrieval Year: (2024)
Ref_id:b19 Title: Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models Year: (2023)
Ref_id:b20 Title: Wordnet: a lexical database for english Year: (1995)
Ref_id:b21 Title: Information extraction from scientific articles: a survey Year: (2018)
Ref_id:b22 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b23 Title: Unifying large language models and knowledge graphs: A roadmap Year: (2024)
Ref_id:b24 Title: Multi-agent collaboration: Harnessing the power of intelligent llm agents Year: (2023)
Ref_id:b25 Title: Wikidata: a free collaborative knowledgebase Year: (2014)
Ref_id:b26 Title: Pubmed 2.0. Medical reference services quarterly Year: (2020)
Ref_id:b27 Title: Autogen: Enabling next-gen llm applications via multi-agent conversation framework Year: (2023)
Ref_id:b28 Title: Consistent and efficient long document understanding Year: (2023)
Ref_id:b29 Title: Biokg: a comprehensive, large-scale biomedical knowledge graph for ai-powered, data-driven biomedical research Year: (2023)
Ref_id:b30 Title: Llms for knowledge graph construction and reasoning: Recent capabilities and future opportunities Year: (2024)
Ref_id:b31 Title: Extract all available metadata: Title, Authors, Date Year: ()
Ref_id:b32 Title: Convert the text to ASCII or minimal LaTeX Year: ()
Ref_id:b33 Title:  Year: ()
Ref_id:b34 Title: } B.11 Evaluator Agent (EA) Prompt for Clarity Title: EA_Prompt_clarity Role Description: You are the Evaluator Agent responsible for assessing the clarity of each candidate triplet. After the initial extraction, some triplets may contain ambiguous terminology or uncertain references. Your job is to assign a clarity score Cl(t) to each triplet and decide whether it is sufficiently clear to be integrated into the Knowledge Graph (KG) Year: ()
Ref_id:b35 Title: Partial clarity metrics Year: ()
Ref_id:b36 Title: A note on whether the triplet's terms or relation are ambiguous Year: ()
Ref_id:b37 Title: A JSON array "final_triplets Year: ()
Ref_id:b38 Title: Downweight or penalize triplets tagged as having ambiguous or unclear terms. LLM Prompt Template (Illustrative Example): [System Role: EvaluatorAgent] Given an array of triplets with partial clarity metrics [c1, c2, ...] and any notes on ambiguity, compute a final clarity score Year: ()
Ref_id:b39 Title: Some triplets may be factually correct but not pertinent to the KG's domain or scope. Your duty is to compute a relevance score R(t) for each triplet and decide if it should be included in the KG Year: ()
Ref_id:b40 Title: Partial relevance scores Year: (2002)
Ref_id:b41 Title: Metadata or tags indicating alignment with the KG's domain Year: ()
Ref_id:b42 Title: Output: A JSON array "final_triplets Year: ()
Ref_id:b43 Title: Aggregation Formula: -Combine the partial relevance scores via an average or logistic function. -Penalize triplets flagged as outside of the domain or referencing unknown entities Year: ()
Ref_id:b44 Title: Given an array of triplets with partial relevance scores [r1, r2, ...] and domain tags, compute a final relevance score Year: ()
