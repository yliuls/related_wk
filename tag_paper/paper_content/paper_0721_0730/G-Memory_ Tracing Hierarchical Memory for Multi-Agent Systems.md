Title: G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems
Abstract: Large language model (LLM)-powered multi-agent systems (MAS) have demonstrated cognitive and execution capabilities that far exceed those of single LLM agents, yet their capacity for self-evolution remains hampered by underdeveloped memory architectures. Upon close inspection, we are alarmed to discover that prevailing MAS memory mechanisms (1) are overly simplistic, completely disregarding the nuanced inter-agent collaboration trajectories, and (2) lack crosstrial and agent-specific customization, in stark contrast to the expressive memory developed for single agents. To bridge this gap, we introduce G-Memory, a hierarchical, agentic memory system for MAS inspired by organizational memory theory [1], which manages the lengthy MAS interaction via a three-tier graph hierarchy: insight, query, and interaction graphs. Upon receiving a new user query, G-Memory performs bi-directional memory traversal to retrieve both high-level, generalizable insights that enable the system to leverage cross-trial knowledge, and fine-grained, condensed interaction trajectories that compactly encode prior collaboration experiences. Upon task execution, the entire hierarchy evolves by assimilating new collaborative trajectories, nurturing the progressive evolution of agent teams. Extensive experiments across five benchmarks, three LLM backbones, and three popular MAS frameworks demonstrate that G-Memory improves success rates in embodied action and accuracy in knowledge QA by up to 20.89% and 10.12%, respectively, without any modifications to the original frameworks. Our codes are available at https://github.com/bingreeky/GMemory.

Section: Introduction
As Large Language Models (LLMs) continue to redefine the frontier of artificial intelligence, LLMdriven agents have exhibited unprecedented prowess in perception [2,3,4,5], planning [6,7,8], reasoning [9,10], and action [11,12], which have catalyzed remarkable progress across diverse downstream domains, including code generation [13,14], data analysis [15], embodied tasks [16] and autonomous driving [3,17,18]. Building upon the impressive competencies of single agents, LLMbased Multi-Agent Systems (MAS) have been demonstrated to push the boundaries of single model capacity [19,20,21]. Similar to collective intelligence arising from human social collaboration [22,23,24], MAS orchestrates multiple agents [25,26,27], whether through cooperation [28,29,30,31] or competition [32,33,34], to transcend the cognitive and specialized limitations of solitary agents.
this section cite: ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b2', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33']

Section: Self-Evolving Agents.
What especially characterizes LLM agents is their self-evolving capacity, i.e., the ability to continuously adapt and improve through interactions with the environment, as seen in prior works where such adaptability has led to two-to three-fold quantitative improvements [35]. The central driving force behind such self-evolving nature is memory mechanism of agents [36,37], which parallels human abilities to accumulate knowledge, process past experiences, and retrieve
this section cite: ['b34', 'b35', 'b36']

Section: Insight Graph

this section cite: []

Section: Query Graph Interaction Graph
Verify the state of objects before and after action...
this section cite: []

Section: Insight

this section cite: []

Section: Query
Check whether there is an apple in the garden Status Execution success! Query Check whether there are more than two oranges on the table of the kitchen? Status Execution fail! Unconvincing! You should .. I will reclaim the order more clearly... I found 2 apples, 1 orange. OK, my task is to check the garden... Insight Check all possible locations before reaching an conclusion on ... The answer now is YES! CoT (5.8e5 tokens) ComplexCoT (6.1e5 tokens) AutoGen (4.3e6 tokens) DyLAN (7.9e6 tokens) Token cost on ALFWorld benchmark Performance (%) Token cost (1e+6) relevant information. Previous successful memory mechanism designs, including both inside-trial memory (i.e., context retained within solving one single query) and cross-trial memory (i.e., experience accumulated across multiple tasks) [38], have empowered agents to excel in diverse applications such as personalized chat [36,39,40], recommendation [41], embodied action [42,16], and social simulation [19,43,44], enabling them to evolve into experiential learners that effectively leverage past experiences and world knowledge.
Self-Evolving MAS. However, such self-evolving capacity remains largely absent in multi-agent systems. Most existing MAS are still constrained by manually defined workflows, such as the Standard Operating Procedures (SOP) in MetaGPT [21] and ChatDev [45], or rely on pre-defined communication topologies in MacNet [46] and AgentPrune [30]. More recent automated MASs, such as GPTSwarm [47], ADAS [48], AFlow [49], and MaAS [50] have made it to automatically optimize inter-agent topologies or prompts, which, nevertheless, ultimately yield giant and cumbersome MAS architectures, lacking the agility to self-adjust with accumulated collaboration experience.
this section cite: ['b37', 'b35', 'b38', 'b39', 'b40', 'b41', 'b18', 'b42', 'b43', 'b20', 'b44', 'b45', 'b29', 'b46', 'b47', 'b48', 'b49']

Section: Memory for MAS.
The absence of the aforementioned self-evolving capacity is, in fact, rooted in the lack of memory mechanisms specifically tailored for MAS. One may challenge this claim from two perspectives: ❶ Do existing MASs lack memory mechanisms altogether? Not entirely. Classical MAS frameworks such as MetaGPT, ChatDev, and Exchange-of-Thought [51] incorporate memory-related designs. However, these are often limited to inside-trial memory [51], while cross-trial memory, if present, remains rudimentary-typically involving the transmission of overly condensed artifacts (e.g., final solutions or execution results) [21,45,46], and failing to enable meaningful learning from collaborative experience. ❷ Why not directly transfer existing single-agent memory mechanisms to MAS? Unfortunately, such a transfer is far from straightforward. The inherent nature of MAS, i.e., multi-turn orchestration across multiple agents [26,27], leads to substantially longer task-solving trajectories compared to single-agent settings (up to 10× more tokens, as demonstrated by Figure 1 (Left)). This poses a significant challenge to traditional retrieval-based memory designs [36,37,16], as naive feeding of the entire long-context trajectory without proper abstraction from a collaborative perspective offers little benefit. Given the aforementioned challenges, a natural question arises:
How can we design a memory mechanism capable of storing, retrieving, and managing the lengthy interaction history of multi-agent systems, such that agent teams can benefit from concise and instructive experience and insights?
The Present Work: G-Memory. In response to the above question, we introduce a :
Graph-based Agentic ::::::: Memory Mechanism for LLM-based Multi-Agent Systems, dubbed G-Memory, which manages the complex and lengthy interaction history of MAS through a three-tier hierarchical graph structure:
✱ Insight Graph, which abstracts generalizable insights from historical experience; ✱ Query Graph, which encodes meta-information of task queries and their connectivity; ✱ Interaction Graph, which stores fine-grained textual communication logs among agents.
Figure 1 (Right) visualizes these structures, and their formal definitions are placed in Section 3. When a new query arrives, G-Memory efficiently retrieves relevant query records by leveraging the topology of the query graph, and then traverses upward (i.e., query→insight graph) to extract associated highlevel insights and downward (i.e., query→interaction graph) to identify core interaction subgraphs that are most pertinent to the task at hand, thereby mitigating information overload. Based on the retrieved memory, G-Memory offers actionable guidance to the MAS, e.g., division of labor, task decomposition, and lessons from past failures. Upon the completion of a task, all three levels of the memory hierarchy are updated in an agentic manner, with newly distilled insights, enriched query records, detailed MAS trajectories, and their level of detailed associations. Through this refinement, G-Memory functions as a plug-and-play module that can be seamlessly embedded into mainstream MAS frameworks, empowering evolving inter-agent collaboration and collective intelligence.
Our contributions are summarized as follows:
❶ Bottleneck Identification. We conduct a thorough review of existing multi-agent systems and identify a fundamental bottleneck in their self-evolving capabilities, which is largely attributed to the oversimplified memory architectures. ❷ Practical Solution. We propose G-Memory, a hierarchical agentic memory architecture for MAS, which models complex and prolonged inter-agent collaboration through a three-tier structure comprising insight, query, and interaction graphs. ❸ Experimental Evaluation. Extensive experiments across five benchmarks show that G-Memory is (I) high-performing, improving state-of-the-art MAS by up to 20.89% and 10.12% on embodied action and knowledge QA tasks, respectively; and (II) resource-friendly, maintaining comparable or even lower token usage than mainstream memory designs.
this section cite: ['b50', 'b50', 'b20', 'b44', 'b45', 'b25', 'b26', 'b35', 'b36']

Section: Related Works
Single-Agent Memory. Memory serves as a primary driving force for agents to accumulate experiences and explore the world through interactions with the environment [52,53,54,55]. It plays a critical role in both task-solving and social simulation LLM agents, and this work primarily focuses on the former. Early research on agent memory was confined to simple inside-trial memory, mainly addressing limitations posed by the LLM context window in chatbot applications, including MemoryBank [36], ChatDB [39], MemoChat [40], and MemGPT [37], which typically adopt retrievalaugmented generation (RAG)-style, similarity-based chunk retrieval. Subsequent developments have progressed toward more cognitively inspired memory architectures, including (1) memory scope extended to cross-trial memory like ExpeL [42] and Synapse [56]; (2) application domains broadened to include computer control [56], embodied action [57], scientific discovery [58], coding and reasoning [59]; and (3) management techniques evolved from coarse-grained textual similarity toward more sophisticated abstraction and summarization of acquired knowledge and experiences [19], as seen in A-Mem [60], Mem0 [61] and MemInsight [62]. More discussions are in Appendix D.
this section cite: ['b51', 'b52', 'b53', 'b54', 'b35', 'b38', 'b39', 'b36', 'b41', 'b55', 'b55', 'b56', 'b57', 'b58', 'b18', 'b59', 'b60', 'b61']

Section: Memory in Multi-agent System.
However, the memory mechanisms tailored for MAS remain markedly underexplored. Some representative frameworks, such as LLM-Debate [20,33] and Mixture-of-Agent [63], omit memory components altogether. Others merely adopt simplistic insidetrial memory schemes [46,51]. Even in frameworks that attempt cross-trial memory [45], the memory is merely compressed as the final outcome artifacts, overlooking the nuanced agent interactions. Collectively, there is a pressing need for a principled memory architecture that can capture, organize, and retrieve the inherently intricate task-solving processes unique to MAS [38].
this section cite: ['b19', 'b32', 'b62', 'b45', 'b50', 'b44', 'b37']

Section: LLM-based Multi-Agent Systems.
Our work focuses on task-solving MAS, which, unlike their single-agent counterparts, often lack the capacity for continual evolution through interaction with the environment [64,65]. Early frameworks such as AutoGen [13], CAMEL [24], and AgentVerse [66] rely entirely on pre-defined workflows. More recent efforts [67,68,49,48,69,31] introduce a degree of adaptivity by generating dynamic MAS in response to environmental feedback. However, such evolution is often one-shot: for example, AFlow [49] employs Monte Carlo Tree Search to construct a complex MAS tailored to a specific task domain, which yet lacks the capacity to evolve with increasing task exposure or transfer across domains [50,70]. From this perspective, constructing MAS with genuine self-evolving capabilities remains an open and challenging research frontier.
this section cite: ['b63', 'b64', 'b12', 'b23', 'b65', 'b66', 'b67', 'b48', 'b47', 'b68', 'b30', 'b48', 'b49', 'b69']

Section: Preliminary
In this section, we establish the notation and formalize key concepts of multi-agent systems and G-Memory's hierarchical memory architecture.
this section cite: []

Section: Multi-agent System Formalization.
Consider a multi-agent framework represented by a directed graph G = (V, E), where |V| = N is the number of agents and E ⊆ V×V defines their communication channels. Each node C i ∈ V corresponds to an individual agent described by the quadruple:
C i = (Base i , Role i , Mem i , Plugin i ),(1)
where Base i denotes the underlying large language model instance, Role i specifies the agent's designated role or persona, Mem i encapsulates its memory state, including past interactions or external knowledge stores, and Plugin i is the set of auxiliary tools (e.g., web-search engine).
Upon receiving a user query Q, the system evolves through T synchronous communication epochs. At each epoch t, we derive a topological ordering π = [π 1 , . . . , π N ] of the nodes such that if there is an edge from π j to π k , then j < k, which guarantees that every agent processes its inputs only after all its predecessors have acted. For each agent C i in π, its output at iteration t is computed as:
r (t) i = C i P (t) sys , Q, {r (t) j : C j ∈ N -(C i )} ,
where: r (t)
i denotes the response generated by C i (which may include reasoning steps, intermediate analyses, or final proposals), P (t) sys comprises global instructions (including each agent's R i ), N -(C i ) is the set of in-neighbors of C i , whose outputs serve as contextual inputs. After all agents have acted, a global aggregation operator A fuses the collection of responses into an interim solution a (t) :
a (t) = A(r (t) 1 , . . . , r (t) N ).
Common implementations for A include majority voting schemes [47], hierarchical summarization via dedicated aggregator agents [13,30], or simply adopting the final agent's output as the answer [46]. These epochs iterate for t = {1, . . . , T } until either a preset limit is reached or an early-stopping criterion is met [71], producing the final response a (T ) to the query Q.
this section cite: ['b46', 'b12', 'b29', 'b45', 'b70']

Section: Memory Architecture.
Our proposed G-Memory orchestrates and manages the memory of multiagent systems via the following three hierarchical graph structures:
[✱] Interaction Graph (Utterance Graph). For query Q, let G (Q) inter = (U (Q) , E (Q)
u ) denote its interaction trajectory, where (i) nodes U (Q) = {u i } represent atomic utterances, with each u i ≜ (A i , m i ) containing A i ∈ V (speaking agent), and m i (textual content), (ii
) Edges E (Q) u ⊆ U (Q) × U (Q) follow temporal relationships: (u j , u k ) ∈ E (Q) u
⇐⇒ u j is transmitted to and inspires u k .
[✱] Query Graph. The query graph, storing previously tackled queries and metadata, is as follows:
G query = (Q, E q ) = Q i , Ψ i , G (Qi) inter |Q| i=1 , E q ,(2)
where
Q = {q i } is the node set, node q i ≜ (Q i , Ψ i , G(Qi)
inter ) is composed of the original query Q i , task status Ψ i ∈ {Failed, Resolved}, and its associated interaction graph G (Qi) inter . The edges E q ⊆ Q × Q encode semantic relationships between queries. The query graph enables retrieval beyond coarse metrics such as embedding similarity, with its meticulous topology.
[✱] Insight Graph. The highest-level insight graph is featured as follows:
G insight = (I, E i ) = ⟨κ k , Ω k ι k ⟩ |I| k=1 , E i ,(3)
where the node set I = {ι k } represents distilled insights, each node ι k is composed of the insight content κ k and the set of supporting queries Ω k ⊆ Q. The edges E i ⊆ I × I × Q forming hyper-connections where (ι m , ι n , q j ) indicates insight ι m contextualizes ι n through query q j .
this section cite: []

Section: G-Memory
This section outlines the management workflow of G-Memory, as illustrated in Figure 2. Specifically, upon the arrival of a new query Q, G-Memory first conducts coarse-grained retrieval to identify pertinent trajectory records (▷ Section 4.1). It then performs bi-directional hierarchical memory traversal: upward to retrieve collective cognitive insights, and downward to distill concrete procedural trajectories (▷ Section 4.2). After the memory-augmented MAS completes the query execution, the hierarchical memory architecture is jointly updated based on environmental feedback, thereby achieving the institutionalization of group knowledge (▷ Section 4.3).
this section cite: []

Section: Query

this section cite: []

Section: Similaritybased Retrival
1.
this section cite: []

Section: Core Path Extraction
Are Deodato and Alejandro both film directors?
this section cite: []

Section: Insight

this section cite: []

Section: Iteration

this section cite: []

Section: Multi-agent System
🤔 Query/Task
Topic: Diffculty: Embodied Topic: Diffculty: Embodied You are in the middle of a room. Looking quickly around you, you see a cabinet 6, a cabinet 5, ... Your task is to: put a clean egg in microwave. Your task is to find a butterfly egg in the outside. Move it to the green box in the bathroom. Are both Lygodium or Maxillaria a genus of orchids? Topic: Diffculty: Web search
this section cite: []

Section: Trajectory Condensation

this section cite: []

Section: Collab. Experience
This history follows a chainstyle, collaboration strategy...
this section cite: []

Section: Failure Lessons
Take care when summarizing the final result, DO NOT ...
this section cite: []

Section: Distilled Insights
Insight 1: Clearly identify key entities and their roles, use specific names and titles Insight 2: Doublecheck the relevance of the search results
this section cite: []

Section: from from and
Ensure the search terms are precise and directly related to the specific entity or institution ...
this section cite: []

Section: Memory Augmentation Output Solution

this section cite: []

Section: Environment Feedback
Execution: Success Token cost: 3,345 Tool calls: 3 ...
this section cite: []

Section: Update Interaction Update Insights
Since both are confirmed as film directors from their respective countries, the answer is Yes.
this section cite: []

Section: Interaction Graphs
Query Graph Interactions CEO agent: assigning tasks... Thinker agent: OK, I will... Executor agent: ...
this section cite: []

Section: Insight Graph

this section cite: []

Section: Memory Augmentation
Black font: Notations
this section cite: []

Section: Symbols
Red font: Operations
: Forward Process : Insight node : Query node : Agent utterance node : Edges btween nodes Topic: Diffculty: Game b1 is on b2., b2 is on b6., b3 is on b7., b5 is on b3.
this section cite: []

Section: Coarse-grained Memory Retrieval
As a plug-in designed for seamless integration into mainstream MAS, G-Memory is triggered when the MAS G encounters a new user query Q. As emphasized in organizational memory theory [1], efficient knowledge retrieval typically begins with broadly relevant schemas prior to more fine-grained access. Following this principle, G-Memory first performs a coarse-grained similarity-based retrieval over the query graph G query to efficiently obtain a sketched set of queries Q S :
Q S = arg top-k qi∈Q s.t. |Q S |=k v(Q) • v(q i ) |v(Q)| |v(q i )| ,(4)
where v(•) maps queries into fixed-length embeddings using models such as MiniLM [72]. While Equation (4) retrieves semantically similar historical queries, the similarity may be only superficial or noisy. Therefore, G-Memory further enlarges the relevant set via hop expansion on the query graph:
QS = Q S ∪ Q k ∈ Q | ∃Q j ∈ Q S , Q k ∈ N + (Q j ) ∪ N -(Q j ) ,(5)
where QS is augmented with the 1-hop neighbors of Q S on the query graph G query , and N + (•) and N -(•) denote the out-neighborhood and in-neighborhood of node Q j , respectively. However, it is suboptimal to directly feed these relevant records as input akin to certain single-agent memory systems [40,37]. On one hand, the excessive context length may overwhelm the LLM; on the other hand, agents in MAS play distinct roles and should be assigned specialized memory tailored to their functions. To address this, the next section introduces a bi-directional processing scheme in G-Memory that operates over both abstract and fine-grained memory levels.
this section cite: ['b0', 'b71', 'b39', 'b36']

Section: Bi-directional Memory Traversal
Subsequent to identifying the expanded set of relevant query nodes QS within G query , G-Memory executes a bi-directional memory traversal to furnish multi-granularity memory support. Specifically, G-Memory first performs an upward traversal (G query → G insight ), retrieving insight nodes that may provide high-level guidance for the current task:
I S = Π Q→I ( QS ), Π Q→I (S q ) ≜ {ι k ∈ I | Ω k ∩ S q ̸ = ∅} ,(6)
where Π Q→I is a query-to-insight projector that identifies all the insight nodes whose supporting query sets intersect with the input query set, and the retrieved insights I S encapsulate distilled, generalized knowledge potentially relevant for orienting the MAS G's strategic approach to Q.
Beyond generalized insights, the fine-grained textual interaction history of the MAS is equally valuable, as it reveals the underlying reasoning patterns that led to successful or failed collaborations [67,73,74]. To utilize these concisely, in the downward traversal (G query → G interaction ), G-Memory employs an LLM-facilitated graph sparsifier S LLM (•, •) to extract the core subgraph that encapsulates essential inter-agent collaboration:
{ ĜQi inter } |M | i=1 = S LLM (G (Qj ) inter , Q) | q j ∈ argtop-M {q ′ k ∈ QS } s.t. |•|=M R LLM (Q, q ′ k ) ,(7)
where R LLM (Q, q j ) rates the relevancy of historical queries w.r.t. Q, and the sparsifier
S LLM (G (Qj ) inter , Q) constructs a sparsified graph Ĝ(Qj) inter = ( Û(Qj) , Ê(Qj) u ) from the original G (Qj )
inter by identifying and retaining dialogue elements. Please refer to Appendix C for their implementations.
Upon completing the bi-directional traversal, we obtain both generalizable insights (I S ) and detailed collaborative trajectories ({ ĜQi inter }
this section cite: ['b66', 'b72', 'b73']

Section: |M |
i=1 ). G-Memory then proceeds to provide specialized memory support for each agent C ∈ V within the MAS G.
Mem i ← Φ I S , { ĜQi inter } |M | i=1 ; Role i , Q , ∀C i = (Base i , Role i , Mem i , Plugin i ) ∈ V,(8)
where the operator Φ(•; •) evaluates the utility and relevance of each insight ι k ∈ I S and sparsified interaction graph Ĝ(Qj) inter concerning the agent's specific role Role i and the task Q (see Appendix C). Based on this evaluation, Φ intializes each agent's internal memory state Mem i with filtered insights, interaction snippets, summaries thereof, equipping it with pertinent historical context before it participates in the subsequent reasoning epochs of the MAS. It is worth noting that G-Memory is invoked at the onset of solving query Q in our implementation. However, practitioners may flexibly configure more fine-grained invocation strategies, such as at the beginning of each MAS dialogue round or selectively for specific agents, based on their needs.
this section cite: []

Section: Hierarchy Memory Update
After completing memory augmentation for each agent, the system G is executed as outlined in Section 3, yielding a final solution a (T ) and receiving environmental feedback, including execution status Ψ i ∈ {Failed, Resolved}, token usage, and other performance metrics. Subsequently, G-Memory updates its hierarchical memory architecture to incorporate this new query. At the interaction level, G-Memory traces each agent's utterances to construct the interaction graph G (Q) inter , which is then stored. At the query level, a new query node is instantiated and added to the query graph Q query :
q new ← (Q, Ψ, G (Q) inter ), N conn ← Q R ∪ ι k ∈I S Ω k , E new ← {(q n , q new ) | q n ∈ N conn }, G next query ← (Q ∪ {q new }, E q ∪ E new ),(9)
where edges are established between q new and (ii) the set Q R containing the top-M relevant historical queries identified in Equation (7), and (ii) the set of queries ι k ∈Iret Ω k that support the insights I S utilized for solving Q. G next query denotes the updated query graph. Finally, at the insight level, G-Memory integrates the learning from the completed query Q into the insight graph G insight = (I, E i ). First, possible new insights summarizing the experience are generated and structurally linked via a summarization function J (•, •) (see prompt in Appendix C) as follows:
ι new = (J (G (Q) inter , Ψ), {q new }), E i, new ← {(ι k , ι new , q new ) | ι k ∈ I S } G ′ insight ← (I ∪ {ι new }, E i ∪ E i, new )(10)
where edges are added to connect the previously utilized insights which inspires the completion of Q in Equation (6). Afterward, the supporting query sets (Ω k ) for the utilized insights (I S ) are updated to include q new , reflecting their relevance to this successful (or failed) application:
I next ← (I \ I ret ) ∪ {(κ k , Ω k ∪ {q new }) | ι k = (κ k , Ω k ) ∈ I ret } ∪ {ι new } G next insight ← (I next , E i ∪ E i, new ),(11)
where the final node set I next incorporates the new insight and the updated versions of the utilized insights, and the resulting graph G next insight thus encapsulates the integrated knowledge. This continuous update cycle across all hierarchical levels enables G-Memory to learn and adaptively refine its collective memory based on ongoing experience.
this section cite: ['b6', 'b5']

Section: Experiment
In this section, we conduct extensive experiments to answer: (RQ1) How does G-Memory perform compared to existing single/multi-agent memory architectures? (RQ2) Does G-Memory incur excessive resource overhead? (RQ3) How sensitive is G-Memory to its key components and parameters?
this section cite: []

Section: Experiment Setup Datasets and Benchmarks.
To thoroughly evaluate the effectiveness of G-Memory, we adopt five widely-adopted benchmarks across three domains: (1) Knowledge reasoning, including Hot-potQA [75] and FEVER [76];
(2) Embodied action, including ALFWorld [77] and SciWorld [78];
(3) Game, namely PDDL [79]. Details on these benchmarks are in Appendix A.1.
Baselines. We select four representative single-agent memory baselines, including non-memory, Voyager [16], MemoryBank [36], and Generative Agents [19], as well as three multi-agent memory implementations from MetaGPT [21], ChatDev [45], and MacNet [46], denoted as MetaGPT-M, ChatDev-M, and MacNet-M, respectively. Details are in Appendix A.2. MAS and LLM Backbones. We select three representative multi-agent frameworks to integrate with G-Memory and the baselines, including AutoGen Parameter Configurations. We implement the embedding function v(•) in Equation (4) with ALL-MINILM-L6-V2 [80]. The number of the most relevant interaction graphs M in Equation ( 7) is set among {2, 3, 4, 5}, and the number of relevant queries k in Equation ( 4) is set among {1, 2}. The detailed ablation study on hyper-parameters is placed in Section 5.4.
this section cite: ['b74', 'b75', 'b76', 'b77', 'b78', 'b35', 'b18', 'b20', 'b44', 'b45', 'b79']

Section: Main Results (RQ1)
Tables 1, 2 and 3 comprehensively report the performance of different memory architectures across three LLM backbones and three MAS frameworks. We summarize the key observations as follows: Takeaway ➊: G-Memory consistently improves performance across all task domains and MAS frameworks. As shown in Table 2, when integrated with AutoGen and MacNet (powered by Qwen-2.5-7b), G-Memory surpasses the best-performing single-/multi-agent memory baselines by an average of 6.8% and 5.5%, respectively. With the more capable Qwen-2.5-14b, the improvement is even more pronounced: in Table 3, G-Memory boosts MacNet's performance on ALFWorld from 58.21% to 79.10%, achieving a substantial 20.89% gain.
Takeaway ➋: Multi-agent systems demand specialized memory designs. A thorough examination of existing baselines reveals a surprising insight: most memory mechanisms fail to consistently benefit MAS settings. In Table 2, baselines such as Voyager and MemoryBank degrade AutoGen's performance on PDDL by as much as 4.17% and 1.34%, respectively. We attribute this to the inability of these methods to provide agent role-specific memory support, which is essential in the PDDL strategic game tasks, where effective division of labor is critical to success. Even MAS-oriented designs, such as ChatDev-M, result in a 2.32% performance drop when applied to MacNet+SciWorld. We attribute this to ChatDev-M's narrow memory scope-storing only the execution results of past queries, which provides limited utility in embodied action environments. These findings highlight the necessity of G-Memory's core characteristics: role-specific memory cues, abstracted high-level insights, and trajectory condensation-all of which are critical for effective memory in MAS.
this section cite: []

Section: Cost Analysis (RQ2)
To evaluate the efficiency of G-Memory in terms of token consumption, we visualize the performance versus token cost trade-off across various settings, as shown in Figures 3 and 7. Our findings are: Takeaway ➌: G-Memory achieves high-performing collective memory without excessive token consumption. As depicted in Figure 3, G-Memory consistently delivers the highest performance improvement (10.32% ↑ over no-memory setting on PDDL+AutoGen) while maintaining a modest increase in token consumption (only 1.4 × 10 6 ). In contrast, MetaGPT-M incurred an additional 2.2 × 10 6 tokens for a mere 4.07% gain. This clearly demonstrates the token-efficiency of G-Memory.
this section cite: []

Section: Framework Analysis (RQ3) Sensitivity Analysis.
Regarding the hop expansion, as shown in Figure 4a, 1-hop expansion consistently yields the best or near-best performance across tasks, with peak accuracies of 85.82% (ALFWorld), 55.24% (PDDL) in AutoGen. In contrast, 2-hop and 3-hop settings often degrade performance, e.g., PDDL drops to 49.79% (2-hop). This suggests that excessive hop expansion may introduce irrelevant insights during memory upward traversal, impairing task-specific reasoning.
Similarly, Figure 4b shows that the optimal k is among {1, 2}. Larger k values (e.g., k=5) can significantly degrade the system performance, e.g., 7.71% ↓ on ALFWorld+AutoGen and 2.5% ↓ on FEVER+DyLAN, indicating that retrieving more queries may introduce task-irrelevant noise. Collectively, we employ 1-hop expansion and k ∈ {1, 2} throughout the experiments.
this section cite: []

Section: Ablation Study.
Figure 4c presents an ablation of G-Memory by isolating the impact of the highlevel insight module (I S in Equation ( 6)) and fine-grained interactions ({ ĜQi inter }
this section cite: []

Section: |M |
i=1 in Equation ( 7)). As shown, removing either part leads to a consistent performance drop. When only fine-grained interactions are enabled, the average scores drop by 4.47% ↓ for AutoGen and 3.82% ↓ for DyLAN   ALFWorld + AutoGen Query put a clean cloth in countertop
this section cite: []

Section: AutoGen Team
: Ensure all required items are accessible, clean them, and return them to their designated storage locations or the specified location after use.
this section cite: []

Section: Fine-grained Trajectory

this section cite: []

Section: High-level Insights
Solver agent Ground agent Action agent For : After cleaning an item, ensure it is placed in the designated storage location immediately to avoid confusion or loss.
this section cite: []

Section: For
Task: put a clean egg in microwave.
this section cite: []

Section: Compressed Traj:
Go to Fridge & Take Egg Execution Success ... Go to microwave Clean first! HotpotQA + DyLAN Query DyLAN Team Fine-grained Trajectory High-level Insights Task: Compressed Traj: Question: Are both Lygodium or Maxillaria a genus of orchids? verify that the search results are not mistakenly referring to similar entities with similar names or unrelated information. Avoid mistakenly referring Are Ruggero Deodato from Italy, and Mexican Alejandro Springall, both film directors?" Search for Deodato Identify Deodato Identify Warning! Re-search for Deodato Passed ... PDDL + MacNet Query MacNet Team Fine-grained Trajectory High-level Insights Task: Compressed Traj: The goal is to satisfy the following conditions: b2 is on b3., b3 is on b1. Unstack b2 from b3 Check b1 and b3 Unstack b3 from b1 ... b1 is on b2., b2 is on b6., b3 is on b7., b5 is on b3., b6 is on b5., b7 is on b4 edge agent For : Ensure that blocks are clear and in the correct positions before attempting to stack them on another block, because this prevents invalid actions and ensures the blocks are placed correctly. Check b3 and b2 Stack b2 on b3 compared to the full method. Conversely, enabling only insights leads to smaller drops of 3.95% and 3.39%. This indicates that while both components are contributive, interactions offer a slightly greater impact, likely due to their preserving more fine-grained, dialogue-level contextual grounding.
this section cite: []

Section: Case Study
Figure 5 illustrates concrete memory cues provided by G-Memory across diverse tasks. For example, in the ALFWorld+AutoGen setting, given the task query "put a clean cloth in countertop", G-Memory successfully retrieves a highly analogous historical query, "put a clean egg in microwave"-both requiring the object to be in a clean state. Alongside this, G-Memory surfaces a critical trajectory segment where the solver agent attempts to place the egg in the microwave before cleaning, prompting the ground agent to intervene. This collaborative trajectory offers actionable guidance for the current task. Moreover, the high-level insights retrieved by G-Memory prove equally valuable for task execution. In the context of HotpotQA's web search task, G-Memory retrieves an insight warning against "mistakenly referring", which helps prevent agents from incorrectly answering based on similarly named individuals. Overall, G-Memory provides effective multi-level memory support across varied domains, including embodied action, knowledge reasoning, and game environments.
this section cite: []

Section: Conclusion & Limitation
In this paper, we conduct a thorough examination of existing memory architectures designed for multi-agent systems (MAS) and identify that their overly simplified designs fundamentally hinder the systems' capacity for self-evolution. To bridge this gap, we propose G-Memory, a hierarchical memory framework that organizes the complex and extended interaction trajectories of MAS into a three-tier graph hierarchy: the insight, query, and interaction graphs. G-Memory provides each agent with customized and hierarchical memory cues, ranging from abstract, generalizable insights to fine-grained, task-critical collaborative segments, and dynamically evolves its knowledge base across episodes. Extensive experiments demonstrate that G-Memory can be seamlessly integrated into state-of-the-art MAS frameworks, significantly enhancing their self-evolution capability, e.g., up to 20.89% ↑ improvement on embodied action tasks. Limitations: Although G-Memory has been evaluated across three domains and five benchmarks, further validation on more diverse tasks (e.g., medical QA) would strengthen its soundness, which we leave for future work.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (1991)
Ref_id:b1 Title: Palm-e: An embodied multimodal language model Year: (2023)
Ref_id:b2 Title: Omnidrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning Year: (2024)
Ref_id:b3 Title: Steve-eye: Equipping llm-based embodied agents with visual perception in open worlds Year: (2023)
Ref_id:b4 Title: Editable scene simulation for autonomous driving via collaborative llm-agents Year: (2024)
Ref_id:b5 Title: Knowledge-augmented planning for llm-based agents Year: (2024)
Ref_id:b6 Title: Plan-and-act: Improving planning of agents for long-horizon tasks Year: (2025)
Ref_id:b7 Title: Ruiming Tang, and Enhong Chen. Understanding the planning of llm agents: A survey Year: (2024)
Ref_id:b8 Title: Agent q: Advanced reasoning and learning for autonomous ai agents Year: (2024)
Ref_id:b9 Title: The landscape of emerging ai agent architectures for reasoning, planning, and tool calling: A survey Year: (2024)
Ref_id:b10 Title: Embodied agent interface: Benchmarking llms for embodied decision making Year: (2024)
Ref_id:b11 Title: Embodied multi-modal agent trained by an llm from a parallel textworld Year: (2024)
Ref_id:b12 Title: Autogen: Enabling next-gen llm applications via multi-agent conversation framework Year: (2023-08-01)
Ref_id:b13 Title: Deepseek-coder: When the large language model meets programming -the rise of code intelligence Year: (2024)
Ref_id:b14 Title: Data interpreter: An llm agent for data science Year: (2024)
Ref_id:b15 Title: An Open-Ended Embodied Agent with Large Language Models. arXiv e-prints Year: (2023-05)
Ref_id:b16 Title: Driving with llms: Fusing object-level vector modality for explainable autonomous driving Year: (2024)
Ref_id:b17 Title: Optimizing autonomous driving for safety: A human-centric approach with llm-enhanced rlhf Year: (2024)
Ref_id:b18 Title: Generative agents: Interactive simulacra of human behavior Year: (2023-04-01)
Ref_id:b19 Title: Improving factuality and reasoning in language models through multiagent debate Year: (2023)
Ref_id:b20 Title: Metagpt: Meta programming for multi-agent collaborative framework Year: (2023-08-01)
Ref_id:b21 Title: Society of mind. Simon and Schuster Year: (1988)
Ref_id:b22 Title: Examining the society of mind Year: (2003)
Ref_id:b23 Title: CAMEL: communicative agents for "mind" exploration of large language model society Year: (2023)
Ref_id:b24 Title: Unleashing cognitive synergy in large language models: A task-solving agent through multi-persona selfcollaboration Year: (2023-07-01)
Ref_id:b25 Title: Large language model based multi-agents: A survey of progress and challenges Year: (2024)
Ref_id:b26 Title: Reasoning capacity in multi-agent systems: Limitations, challenges and human-centered solutions Year: (2024)
Ref_id:b27 Title: Mrinmaya Sachan, and Rada Mihalcea. Cooperate or collapse: Emergence of sustainability behaviors in a society of llm agents Year: (2024)
Ref_id:b28 Title: Discovering causality for efficient cooperation in multi-agent environments Year: (2023)
Ref_id:b29 Title: Cut the crap: An economical communication pipeline for llm-based multi-agent systems Year: (2024)
Ref_id:b30 Title: Learning to route llms for multi-agent systems Year: (2025)
Ref_id:b31 Title: Understanding the competition behaviors in large language model-based agents Year: (2023)
Ref_id:b32 Title: Encouraging divergent thinking in large language models through multi-agent debate Year: (2023)
Ref_id:b33 Title: Battleagentbench: A benchmark for evaluating cooperation and competition capabilities of language models in multi-agent systems Year: (2024)
Ref_id:b34 Title: Progressive-hint prompting improves reasoning in large language models Year: (2023-04-01)
Ref_id:b35 Title: Memorybank: Enhancing large language models with long-term memory Year: (2024)
Ref_id:b36 Title: Towards llms as operating systems Year: (2023)
Ref_id:b37 Title: A survey on the memory mechanism of large language model based agents Year: (2024)
Ref_id:b38 Title: Augmenting llms with databases as their symbolic memory Year: (2023)
Ref_id:b39 Title: Memochat: Tuning llms to use memos for consistent long-range open-domain conversation Year: (2023)
Ref_id:b40 Title: Large language model powered agent for recommendation Year: (2023)
Ref_id:b41 Title: Expel: Llm agents are experiential learners Year: (2024)
Ref_id:b42 Title: Simulating interactions of human behaviors for llm-based task-oriented coordination via collaborative generative agents Year: (2023)
Ref_id:b43 Title: S3: Social-network simulation system with large language model-empowered agents Year: (2023)
Ref_id:b44 Title: Communicative agents for software development Year: (2023-07-01)
Ref_id:b45 Title: Scaling large-language-model-based multi-agent collaboration Year: (2024)
Ref_id:b46 Title: Gptswarm: Language agents as optimizable graphs Year: (2024)
Ref_id:b47 Title: Automated design of agentic systems Year: (2024)
Ref_id:b48 Title:  Year: (2024-10)
Ref_id:b49 Title: Multi-agent architecture search via agentic supernet Year: (2025)
Ref_id:b50 Title: Exchange-of-thought: Enhancing large language model capabilities through cross-model communication Year: (2023)
Ref_id:b51 Title: A survey on large language model based autonomous agents Year: (2024)
Ref_id:b52 Title: Xipeng Qiu, Xuanjing Huan, and Tao Gui. The rise and potential of large language model based agents: A survey Year: (2023)
Ref_id:b53 Title: Large language models empowered agent-based modeling and simulation: A survey and perspectives Year: (2023)
Ref_id:b54 Title: A survey on llm-based multi-agent systems: workflow, infrastructure, and challenges Year: (2024)
Ref_id:b55 Title: Synapse: Trajectory-as-exemplar prompting with memory for computer control Year: (2023)
Ref_id:b56 Title: Ghost in the minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory Year: (2023)
Ref_id:b57 Title: Self-updating library in large language models improves chemical reasoning Year: (2025)
Ref_id:b58 Title: Reflexion: an autonomous agent with dynamic memory and self-reflection Year: (2023)
Ref_id:b59 Title: A-mem: Agentic memory for llm agents Year: (2025)
Ref_id:b60 Title: Mem0: Building production-ready ai agents with scalable long-term memory Year: (2025)
Ref_id:b61 Title: Meminsight: Autonomous memory augmentation for llm agents Year: (2025)
Ref_id:b62 Title: Mixture-of-agents enhances large language model capabilities Year: (2024)
Ref_id:b63 Title: Symbolic learning enables self-evolving agents Year: (2024)
Ref_id:b64 Title: Self-evolving agents with reflective and memory-augmented abilities Year: (2024)
Ref_id:b65 Title: Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents Year: (2023)
Ref_id:b66 Title: Self-evolving multi-agent collaboration networks for software development Year: (2024)
Ref_id:b67 Title: Architecting multi-agent communication topologies via graph neural networks Year: (2024)
Ref_id:b68 Title: Evoagent: Towards automatic multi-agent generation via evolutionary algorithms Year: (2024)
Ref_id:b69 Title: Evoflow: Evolving diverse agentic workflows on the fly Year: (2025)
Ref_id:b70 Title: Dynamic llm-agent network: An llm-agent collaboration framework with agent team optimization Year: (2023)
Ref_id:b71 Title: Microsoft academic graph: When experts are not enough Year: (2020)
Ref_id:b72 Title: Sirius: Self-improving multiagent systems via bootstrapped reasoning Year: (2025)
Ref_id:b73 Title: Reso: A reward-driven selforganizing llm-based multi-agent system for reasoning tasks Year: (2025)
Ref_id:b74 Title: Hotpotqa: A dataset for diverse, explainable multi-hop question answering Year: (2018)
Ref_id:b75 Title: Fever: a large-scale dataset for fact extraction and verification Year: (2018)
Ref_id:b76 Title: Alfworld: Aligning text and embodied environments for interactive learning Year: (2020)
Ref_id:b77 Title: Is your agent smarter than a 5th grader? arXiv preprint Year: (2022)
Ref_id:b78 Title: Agentboard: An analytical evaluation board of multi-turn llm agents Year: (2024)
Ref_id:b79 Title: Deep self-attention distillation for task-agnostic compression of pre-trained transformers Year: (2020)
Ref_id:b80 Title:  Year: ()
Ref_id:b81 Title:  Year: (2024)
Ref_id:b82 Title:  Year: ()
Ref_id:b83 Title:  Year: ()
Ref_id:b84 Title:  Year: ()
Ref_id:b85 Title:  Year: ()
Ref_id:b86 Title: Performance (%) and latency (s) comparison of different memory mechanisms on AutoGen and DyLAN frameworks, along with ALFWorld and SciWorld benchmarks Year: ()
Ref_id:b87 Title:  Year: ()
