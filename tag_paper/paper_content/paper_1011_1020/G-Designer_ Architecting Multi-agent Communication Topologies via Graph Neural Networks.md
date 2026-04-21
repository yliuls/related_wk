Title: G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks
Abstract: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring highquality solution? In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: (1) high-performing, achieving superior results on MMLU with accuracy at 84.50% and on HumanEval with pass@1 at 89.90%; (2) taskadaptive, architecting communication protocols tailored to task difficulty, reducing token consumption by up to 95.33% on HumanEval; and (3) adversarially robust, defending against agent adversarial attacks with merely 0.3% accuracy drop. The code is available at https://github.  com/yanweiyue/GDesigner.

Section: Introduction
An LLM-based agent, which integrates the language generation capabilities of LLMs with decision-making and action-execution functionalities (Richards & et al., 2023;Nakajima, 2023;Reworkd, 2023), has exhibited impressive performance across a wide range of tasks, from reasoning (Yao et al., 2023b) and code generation (Shinn et al., 2023) to even more complex applications like video gaming (Wang et al., 2023) and autonomous driving (Jin et al., 2023). Even more exciting, researchers have discovered that combining multiple LLM-based agents-whether implicitly or explicitly-into a team can outperform individual agents when tackling complex tasks (Du et al., 2023;Liang et al., 2023;Wang et al., 2023b;Jiang et al., 2023;Shinn et al., 2023;Zheng et al., 2023;Wu et al., 2023), demonstrating a form of collaborative intelligence reminiscent of human teamwork in multi-agent systems (Zhang et al., 2023b). This emergence of human-esque collective intelligence is fundamentally driven by the design of their topology, i.e., how multi-agents are connected, and how they transmit, exchange, and assimilate information reciprocally.
In practice, prior research has extensively explored how multiple instances of LLMs, referred to as agents (Wang et al., 2024;Xi et al., 2023;Gao et al., 2023;Cheng et al., 2024), should be structured and organized to converse, collaborate, debate, or even compete. Various topological designs have been investigated, such as chain (Wei et al., 2022;Hong et al., 2023), tree (Yao et al., 2023a; Wu et al., 2023), star (Wu et al., 2023), complete graphs (Qian et al., 2024), random graphs (Qian et al., 2024), optimizable graphs (Zhuge et al., 2024;Zhang et al., 2024), and LLM-based networks (Hao et al., 2023;Liu et al., 2023). These elaborately designed communication topologies have demonstrated remarkable performance with minimal human supervision, bridging the gap between individual and collective intelligence. Faced with numerous structures available, an inquisitive practitioner might ask: how should I select or design a topology that best suits my task at hand?
The question posed above is non-trivial and, at times, perplexing. A piece of experimental evidence is presented in Figure 2, where we evaluated the performance of different multi-agent structures on the MMLU dataset (Hendrycks et al., 2021), a collection of multiple-choice questions across various subjects. The results reveal that even within the same dataset, the suitability of different communication topologies varies. ❶ Simpler Case: in the simpler "High School Biology" subset, the chain structure performs comparably to the complex GPTSwarm, while consuming significantly fewer tokens (0.5k versus 7.8k). In this case, the chain structure is clearly a more economical choice. ❷ Harder Case: However, for the more challenging "College Mathematics" subset, GPTSwarm outperforms the chain structure by 8.75%, primarily attributed to its intricate topology and prompt optimization. In summary, practitioners often find it challenging to effortlessly identify the most efficient and complexity-adaptive multi-agent topology for a given task.
In light of this dilemma, we propose the LLM-based ::
Multi- The formal definition of MACP is provided in Section 3.3.
To design a communication topology that ideally adheres to the MACP principles, we propose an effective, adaptive, and robust LLM-powered multi-agent communication graph designer, termed G-Designer. Technically, G-Designer first architects a multi-agent graph, where each agent, along with its specific properties (e.g., profile (Li et al., 2023a), external API tools (Zhuang et al., 2023), or knowledge base (Chen et al., 2024a)), is represented as a node, and communication between agents forms the edges. G-Designer employs a variational graph auto-encoder to encode the nodes (agents) along with task-specific information, and to decode the resulting collaboration network between agents. This input-dependent paradigm allows G-Designer to design task-adaptive, high-performing communication topology, which is, at the same time, assured of efficiency and robustness with sparsity regularization. Unlike previous LLM-based multi-agent topology designs, which rely on a static structure for all queries/tasks, G-Designer adaptively crafts customized topologies for different domains and tasks, serving as a fully autonomous and flexible assistant for multi-agent system establishment and deployments.
Our contribution can be summarized as follows:
❶ Protocol Proposal. We propose the first communication protocol tailored for LLM-powered multi-agent systems, MACP, which comprehensively regulates multiagent topology design across three dimensions: performance, adaptability, and robustness, and incisively highlights the shortcomings of existing designs. ❷ Practical Solution. We present G-Designer, an effective, adaptive, and robust designer of LLM-powered multiagent communication graphs. By leveraging a variational graph auto-encoder to construct and process the multi-agent network, G-Designer decodes task-adaptive and high-performing agent communication, which is also equipped with strong robustness against agent-rooted adversarial attacks via dynamic topology adjustment. ❸ Experimental Validation. Extensive experiments across six benchmarks show that G-Designer is: (1) highperforming, surpassing state-of-the-art topologies by 0.20% ∼ 4.10% on MMLU and HumanEval; (2) taskadaptive, dynamically adjusting topology complexity with task awareness, outperforming state-of-the-art methods on MMLU with a cost of merely 1.5e + 5 compared to their 2.6e + 6, reducing token consumption by up to 92.24%; and (3) adversarially robust, defending against agent adversarial attacks with merely 0.3% accuracy drop. 2. Related Works LLM-agent Collaboration Recent research has explored various multi-agent communication topologies, including:
(1) Non-interactive, where agents operate independently without inter-agent communication, as employed in systems like LATM (Zhang et al., 2023a) and LLM-Debate (Du et al., 2023); (2) Chain, where agents are arranged in a sequential structure, each receiving the output from its predecessor and passing information to its successor, utilized by ChatDev (Qian et al., 2023), MetaGPT (Hong et al., 2023), and L2MAC (Holt et al., 2024); (3) Star, where a central administrative agent (often referred to as a commander, teacher, etc.) directs subordinate agents, seen in AutoGen (Wu et al., 2023), SecurityBot (Yan et al., 2024), and MiniGrid (Zhou et al., 2023); (4) Tree, where a root agent hierarchically manages multiple child agents, as in SoA (Ishibashi & Nishimura, 2024); and (5) Graph, encompassing complete graphs (Qian et al., 2024;Zhuge et al., 2024) and random graphs (Qian et al., 2024), among others.
Multi-agents as Graphs Graphs, as a fundamental data structure for organizing and representing relationships between entities (Zhang & Chartrand, 2006), are widely adopted in the pre-LLM era as a powerful tool to facilitate effective communication in multi-agent reinforcement learning (MARL) (Pesce & Montana, 2023;Hu et al., 2024;Liu et al., 2022). With the rise of LLMs and the proliferation of LLM-based agents (Chen et al., 2023a;Cohen et al., 2023;Hua et al., 2023), researchers have similarly recognized that interactions among multiple agents can naturally be modeled from a graph-based perspective (Chen et al., 2023b;Zhuge et al., 2024;Qian et al., 2024;Liu et al., 2023). Early attempts are implicit, like ChatEval (Chan et al., 2023), Au-toGen (Wu et al., 2023), and DSPy (Khattab et al., 2023). More recent practices including ChatLLM (Hao et al., 2023), DyLAN (Liu et al., 2023), GPTSwarm (Zhuge et al., 2024), and MacNet (Qian et al., 2024), have explicitly represented the organization of multiple agents as a graph. However, all these attempts, whether predefined or iteratively optimized, remain input-independent. Consequently, they fail to be task-aware and adaptively design topologies that suit the complexity of the specific task.
this section cite: ['b40', 'b33', 'b39', 'b21', 'b10', 'b28', 'b20', 'b64', 'b53', 'b47', 'b54', 'b12', 'b7', 'b51', 'b16', 'b56', 'b53', 'b37', 'b37', 'b67', 'b59', 'b13', 'b31', 'b66', 'b10', 'b36', 'b16', 'b15', 'b53', 'b55', 'b65', 'b19', 'b37', 'b67', 'b37', 'b62', 'b35', 'b17', 'b30', 'b9', 'b18', 'b67', 'b37', 'b31', 'b1', 'b53', 'b22', 'b13', 'b31', 'b67', 'b37']

Section: Formalization
This section establishes the notation, formalizes key concepts from a topology perspective, and formally defines our proposed multi-agent communication protocol.
this section cite: []

Section: Topology Structure
We model the multi-agent system as a directed graph G = (V, E), where V = {v 1 , . . . , v N } represents the set of nodes (with N = |V|) and E denotes the set of edges. Each node v i ∈ V corresponds to an agent, formalized as:
v i = {Base i , Role i , State i , Plugin i },(1)
where each agent v i is composed of four key elements:
(1) Base i , the language model instance powering v i ;
(2) Role i , the agent's pre-assigned role or function;
(3) State i , representing the agent's accumulated knowledge and interaction history; and (4) Plugin i , a set of external tools or plugins available to v i , such as web searchers (Ma et al., 2023), code compilers (Richards & et al., 2023;Wu et al., 2023;Hong et al., 2023;Bouzenia et al., 2024;Ishibashi & Nishimura, 2024), or file readers (Zhuge et al., 2024;Richards & et al., 2023). Each LLM-based agent v i receives prompt P and generates response R i :
R i = v i (P) = v i (P sys , P usr ),(2)
where P sys = {Role i , State i } represents the system prompt encompassing its role and state, and P usr denotes the user prompt, which possibly includes the given tasks, responses/instructions from other agents and externally retrieved knowledge.
The connectivity of G can also be characterized by a (nonsymmetric) adjacency matrix A ∈ {0, 1} N ×N , where
A[i, j] = 1 if e ij = (v i , v j ) ∈ E, otherwise 0. Each edge e ij ∈ E represents the flow of information from v i to v j .
this section cite: ['b32', 'b40', 'b53', 'b16', 'b0', 'b19', 'b67', 'b40']

Section: Communication Pipeline
Given a query/problem Q, the multi-agent system engages in K rounds of interactive utterances, which collaboratively drive the agents toward producing the final solution a (K) based on their cumulative dialogue exchanges. At the beginning of the t-th dialogue round, a mapping function ϕ is applied to determine the execution index for each agent:
ϕ : G -→ σ, σ = [v σ1 , v σ2 , • • • , v σ N ], s. t. ∀i > j, v σi / ∈ N in (v σj ),(3)
where σ is the execution sequence of agents, N in (v σ(j) ) denotes the in-neighborhood of v σ(j) , and the constraint ensures that an agent v σ(i) can only execute after any agent v σ(j) from which it receives information. Once the execution order is determined, each agent proceeds to perform input-output operations sequentially:
R (t) i = v i (P (t) sys , P (t) usr ), P (t) usr = {Q, ∪ vj ∈Nin(vi) R (t) j } (4) where R(t)
i represents the output of v i , which could be a rationale, an answer, or a partial solution, depending on the specific context. The output R
(t) i is generated based on the system prompt P (t)
sys and the context prompt, consisting of the query Q and messages from other agents. At the end of each dialogue, an aggregation function is adopted to generate the answer/solution a (t) :
a (t) ← Aggregate(R (t) 1 , R (t) 2 , • • • , R (t) N ).
(5)
The implementation of the Aggregate function is flexible, with possible options including majority voting (Chen et al., 2024b;Zhuge et al., 2024;Li et al., 2024), aggregating all agents' responses and delegating one agent to provide the final answer (Wu et al., 2023;Jiang et al., 2023;Liu et al., 2023;Zhang et al., 2024), or simply using the output of
this section cite: ['b67', 'b26', 'b53', 'b20', 'b31', 'b59']

Section: Materials Construct Design Optimize

this section cite: []

Section: 🤔 Query/Problem
An electric motor has a label on it that reads: Input: 120V AC, 1.0 Amps, 60 Hz
-Efficiency -75%. At what constant speed can the motor lift up a 6 kg mass? 🤖 Agent (node) set 🧰 Tool/Plugin set 🖥 Python complier 📁 File reader 🦜 Text-to-speech translator 🖼 Image question answerer ... ... (other external tools) Chief Programmer Mathematician Physicist Instantiated by gpt-4/Llama 3.1/... 🎭 Role/Profile pool Task-specific Agent representation Math Analyst Programmer Phycist Chief Scientist Node Encoder Profile Assignment Sentence Bert; MiniLM; ... task Multi-agent Network Virtual node Multi-agent Network: Anchor structure is either user-defined or generated by LLMs as an anchor 0.69 0.28 0.62 0.81 0.93 0.16 0 .9 2 0 .8 1 0 .4 6 Encode: Decode: 0.70 0 .3 5 0.53 Graph Auto-encoder Encoder Decoder Communication Graph Anchor Regularize Sparse Regularzie 🧮 Mathematical calculator generated by gpt-4 Teacher Math Analyst Physicist ... Economist 🧮 🖥 📁 Optimization Config Number of dialogues: Execution order: All the calculations are correct.The key relationships used, such as the power equation , are mathematically solid. Let's break down the problem step by step: The input power 𝑃 𝑖𝑛 is calculated as: ... Iteration `python iv = 120 ic = 1.0 eff = 0.75 eff mass = 6 ... ```F or Physicist, provide a problemsolving analysis; For Mathematician, please check its correctness; For Programmer, write Python codes for verification Iteration round Workflow Ouput solution Solution to task Based on the discussions, the answer is a speed of approximately 1.53 m/s at which the motor can lift 6 kg mass. Input query ▶ Construct graph ▶ Encode & Decode ▶ Iterative dialogue the last agent R (t)
σ N (Qian et al., 2024). Through K rounds of utterances, either predefined (Qian et al., 2024) or determined by an early-stopping mechanism (Liu et al., 2023), the overall system G produces the final answer a (K) for Q.
this section cite: ['b37', 'b37', 'b31']

Section: MACP Protocol
We give the formal definition of MACP Protocol as follows: Definition 3.1 (Multi-agent Communication Protocol). Given an LLM-based multi-agent system G = (V, E), we establish the following objective as optimization principle:
min G∈G -u G(Q) +β1 • ||G|| + β2 • Ĝ( Q) -G(Q) , (6)
where G represents the feasible parameter space of G, u(•) is the utility evaluator, ||G|| measures the computational and communication overhead of the entire graph, and Q and Ĝ denote the query description and the multi-agent system after adversarial perturbation, respectively. The first term in Equation ( 6) corresponds to high performance, aiming to maximize the utility of the system's output; the second term addresses task-adaptiveness, seeking to minimize system complexity to reduce power consumption and economic cost; and the third term focuses on robustness, constraining the deviation of system output under adversarial attacks.
this section cite: []

Section: G-Designer
Figure 3 illustrates how G-Designer adaptively designs communication topologies for any given query. Specifi-cally, the process begins with a few "raw materials": the input query Q, the agent set V, the profile pool, and the toolset. In the Construct stage, G-Designer leverages a node encoder to construct a multi-agent network along with a task-specific virtual node. In the Design stage, a graph auto-encoder is employed to decode the communication graph topology G com , which is leveraged for multi-round inter-agent collaboration in the Optimize stage.
this section cite: []

Section: Multi-agent Network Construction
Given an input query Q and a set of LLM-agents V, G-Designer aims to design a task-adaptive and effective communication topology G com . We begin by assigning each agent a unique role and profile, as previous research (Wang et al., 2023b) has shown that assigning distinct personas or roles to LLM-based agents can enhance cognitive synergy. Based on these roles, different external tools are allocated to the agents (e.g., Mathematica for a math analyst, Python compiler for a programmer). Thus, we successfully initialize each agent v i as {Base i , Role i , State i , Plugin i }, as defined in Equation (1).
We proceed to construct a structured multi-agent network as input to G-Designer, represented as G = (X agent , A), where X agent ∈ R N ×D is the node (agent) feature matrix and A ∈ R N ×N represents the connectivity matrix. For the feature matrix X agent , we employ a node encoder to transform each agent's unique profile into a fixed-length embedding representation:
xi ← NodeEncoder (T (Basei), Rolei, T (Plugin i )) , (7)
where T (•) extracts the textual description of the agent's LLM backbone and its assigned plugins, and NodeEncoder can be realized using small and lightweight text embedding models (Reimers, 2019). After encoding the individual agents, we aim to ensure that the multi-agent network incorporates information related to the query Q, as this query-dependent approach enables G-Designer to be task-aware and adaptive. To this end, we introduce an additional task-specific virtual global node v task , which is bidirectionally connected to all agent nodes, enabling a global "storage sink" and facilitating smoother information flow among agents (Shirzad et al., 2023;Tan et al., 2023;Rosenbluth et al., 2024). This task node is encoded by the NodeEncoder as follows: x task ← NodeEncoder(Q).
After obtaining the agent node features X agent = [x 1 , x 2 , . . . , x N ] ⊤ and the task-specific embedding x task , we provide a simple anchor topology A anchor ∈ {0, 1} N ×N , which serves as a starting point for G-Designer's topology design process. For instance, given a code generation task with three agents: manager/programmer/code reviewer, the anchor topology could be configured as a chain structure, i.e., "manager → programmer → reviewer", reflecting the typical workflow of code completion. The anchor topology, being either user-defined or automatically generated by LLMs, is often simple and sub-optimalfoot_0 . However, it provides a foundational reference and prior knowledge for G-Designer's subsequent optimization process. We incorporate the taskspecific vertex v task and its corresponding edges and obtain Ãanchor ∈ {0, 1} (N +1)×(N +1) . Consequently, we establish a task-specific multi-agent network G:
G = X agent x ⊤ task , Ãanchor = ( Ṽ, Ẽ) = V ∪ {v task }, E ∪ { ←→ (v i , v task )|v i ∈ V)} ,(8)
where Xagent x ⊤ task can also be jointly denoted as X.
this section cite: ['b38', 'b44', 'b45', 'b41']

Section: Designing Communication Topology
Building upon the task-specific multi-agent network G, G-Designer seeks to establish a more fine-grained and precise communication topology G com . Drawing inspiration from the variational graph auto-encoder (VGAE) framework (Kipf & Welling, 2016;Zhao & Zhang, 2024), G-Designer employs a VGAE-based encoder-decoder f v to generate the multi-agent interaction topology:
Gcom = fv( G; Θv) = p(Gcom | H)q(H | X, Ãanchor ), (9
)
where f v is the encoder-decoder architecture with parameters Θ v , q(•) is the encoder module, p(•) is the decoder module. The encoder utilizes posterior probabilities to encode the node embeddings into low-dimensional latent vector representations H agent , which can be formulated as:
q(H agent | X, Ãanchor ) = N i=1 q(h i | X, Ãanchor ), q(h i | X, Ãanchor ) = N (h i | µ i , diag(σ 2 i )),(10)
where µ = GNN µ ( X, Ãanchor ; Θ µ ) is the matrix of mean vectors µ i ; similarly log(σ) = GNN σ ( X, Ãanchor ; Θ σ ).
The choice of GNN backbone can be customized as needed; here, we utilize a simple two-layer GCN (Kipf & Welling, 2017). h i , µ i , and σ i denote the i-th column of H, µ, and σ, respectively. The encoder q(•) is parameterized by Θ e = {Θ µ , Θ σ }. Following the encoding phase, the decoder employs the latent representations to generate a comprehensive blueprint for multi-agent communication.
More specifically, the decoder q(•) = q c • q s first constructs a parameterized, sketched graph S, which is then refined into the final multi-agent communication topology:
p(G com | H agent ) = S p c (G com | S)p s (S | H agent ) dS.
(11) At the first step, p s (•) constructs the sketched adjacency matrix S from the latent representation H agent :
ps(S | Hagent) = N i=1 N j=1 ps(Sij | hi, hj, h task ; Θ d ), (12)
whose detailed derivation is as follows:
p s (S ij = 1 | h i , h j , h task ) = g(h i , h j , h task ), = Sigmoid((log(ϵ) -log(1 -ϵ) + ϖ ij )/τ ),(13)
where ϖ = FFN d ([h i , h j , h task ]) with FFN d parameterized by Θ d , ϵ ∼ Uniform(0, 1), and τ denotes the temperature coefficient. When τ approaches zero, Equation (13) essentially return the Bernouli sampling result for S ij . The resulting matrix S ∈ [0, 1] N ×N represents a densely-connected, non-negative graph distribution, indicating an overly complex and resource-intensive pair-wise communication structure, which is not yet suitable for guiding multi-agent collaboration. To align with G-Designer's objectives of task adaptiveness and minimizing costs, we apply a refinement decoder p c (•) to refine the sketched S into a compact, sparse, and highly informative communication graph, instantiated by a regularization objective:
p c : arg max S∈S 1/2||S -ZWZ ⊤ || 2 F + ζ||W|| * + 1/2||A anchor -ZWZ ⊤ || 2 F , s. t. S = ZWZ ⊤ ,(14)
where Z ∈ R N ×r is the top-r columns of left singular matrix S, ζ is a coefficient hyperparameter, W ∈ R r×r is an optimizable weight matrix, || • || F denotes the Frobenius norm and ||W|| * = i λ i where λ i is the i-th singular value of W. S ∈ R N ×N is the desired sparse topology, which is decomposed as ZWZ ⊤ . In Equation ( 14), the first and second terms are jointly denoted as anchor regularization, which encourage the learned S to maintain similarity with both the original S and the anchor topology. The third term, denoted as sparsity regularization, though appearing to minimize the nuclear norm of W, essentially sparsifies S, since || S|| * = ||W|| * holds due to Z ⊤ Z = I r×r . Therefore, Equation ( 14) achieves two key goals: (1) producing a sparse, refined communication topology, and (2) constraining the design to remain grounded in practical intuition. The resulting communication can be represented as follows:
Gcom = (V, Ecom), Ecom = {(i, j) | Sij ̸ = 0 ∧ (i, j) ∈ E}).(15)
At this stage, we have successfully distilled a lightweight and informative collaboration network G com from the sketched task-specific network G, which is now ready to guide inter-agent message passing in the following process.
this section cite: ['b23', 'b24']

Section: Optimizing G-Designer
Upon obtaining G com , the multi-agent utterances and dialogues can proceed as usual using G com , as detailed in Section 3.2. After K rounds of interaction, the agents converge to a final solution a (K) = G com (Q). We then give the following optimization objective:
arg min Θe,Θ d E Θe,Θ d ∼Ω u G com (Q) ,(16)
where Θ e and Θ d are the parameters of the encoder q(•) and decoder p(•), respectively, Ω is the parameter space and E(•) denotes the mathematical expectation. Equation ( 16) aims to maximize the utility of the generated solution, but it is inherently intractable and non-differentiable, as u(•) often depends on external API calls (Li et al., 2023b;Hendrycks et al., 2021). To address this, following standard approaches in multi-agent structure design (Zhuge et al., 2024;Zhang et al., 2024), we apply policy gradient (Williams, 1992) to approximate and optimize Equation ( 16):
∇ΘEΘ∼Ω u Gcom(Q) ≈ 1 M M k=1 u(a (K) m )∇Θ(P (G k )), (17
)
where Θ = {Θ e , Θ d }, {G k } M m=1 are indepently samples from G com , and {a
(K) m } M
m=1 are the corresponding output. P (G k ) calculates the probability of G k being sampled, which can be expressed as
P (G k ) = N i=1 N j=1 Sij .
Through iterative optimization guided by Equations ( 14) and ( 16) over a limited set of queries as the "training set", G-Designer efficiently develops task-awareness and the ca-pability to strategically design the agent network, achieving truly task-customized multi-agent topology design.
this section cite: ['b67', 'b59', 'b52']

Section: Optimization configuration
The overall training objective of our method is formulated as L G-Designer = L utility + L anchor + L sparse , where L utility represents the optimization target from Equation ( 16), L anchor corresponds to the first and third terms in Equation ( 14), and L sparse is the second term. Given a benchmark {Q i } D i=1 consisting of B queries, G-Designer begins by optimizing with a small subset of B ′ queries and fixes the learned parameters for testing on the remaining (B -B ′ ) queries. The whole algorithm workflow of G-Designer is depicted in Algorithm 1.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup Datasets and Metrics
We evaluate G-Designer on three categories of datasets: ■ General Reasoning: MMLU (Hendrycks et al., 2021); ■ Mathematical Reasoning: GSM8K (Cobbe et al., 2021), MultiArith (Roy & Roth, 2016), SVAMP (Patel et al., 2021), and AQuA (Ling et al., 2017); ■ Code: HumanEval (Chen et al., 2021). We include the dataset statistics in Table 4.
Baselines For single-agent approaches, we select COT (Wei et al., 2022), ComplexCoT (Fu et al., 2022), Self-Consistency (Wang et al., 2023a), and PHP (Zheng et al., 2023). For multi-agent topologies, we select Chain, Star, and Tree (formally defined in (Qian et al., 2024)), Complete Graph and Random Graph , AutoGen (Wu et al., 2023), MetaGPT (Hong et al., 2023), LLM-Debate (Du et al., 2023), LLM-Blender (Jiang et al., 2023), DyLAN (Liu et al., 2023), and GPTSwarm (Zhuge et al., 2024).
Implementation Details We access the GPT via the Ope-nAI API, and mainly test on gpt-4-1106-preview (gpt-4) and gpt-3.5-turbo-0125 (gpt-3.5). We set temperature to 0 for the single execution and single agent baselines and 1 for multi-agent methods. We set a summarizer agent to aggregate the dialogue history and produce the final solution a (K) , with K = 3 across all experiments. The NodeEncoder(•) is implemented using all-MiniLM-L6-v2 (Wang et al., 2020), with the embedding dimension set to D = 384. The anchor topology A anchor is predefined as a simple chain structure. The sampling times M are set as 10, and τ = 1e -2 and ζ = 1e -1 are set for all experiments. We provide explicit agent profiling for multi-agent methods, following the classical configurations in LLM-MA systems (Liu et al., 2023;Zhuge et al., 2024;Yin et al., 2023), and use gpt-4 to generate agent profile pools. For all benchmarks, we merely use B ′ ∈ {40, 80} queries for optimization.
this section cite: ['b8', 'b42', 'b34', 'b29', 'b5', 'b51', 'b11', 'b64', 'b37', 'b53', 'b16', 'b10', 'b20', 'b31', 'b67', 'b48', 'b31', 'b67', 'b58']

Section: Main Results
In this section, we conduct extensive experiments across six benchmarks to verify that G-Designer is:  while the complete graph and GPTSwarm incur an overwhelming token cost at 20 agents (5.6 ∼ 30.3M tokens), G-Designer achieves superior performance with merely 6.11% of GPTSwarm's prompt token consumption, surpassing it by 2.44% ↑. These results decisively demonstrate the scalability and potential of G-Designer in advancing large-scale autonomous multi-agent systems.
this section cite: []

Section: Token-economical (inference)
A key benefit of G-Designer's adaptivity is that it prevents the use of overly complex structures for simple tasks, thus minimizing
&KDLQ 6WDU 7UHH &RPSOHWH *UDSK 5DQGRP *UDSK '\/$1 *376ZDUP $XWR*HQ *'HVLJQHU EHIRUHDWWDFN DIWHUDWWDFN Figure 5. We compare the accuracy (%) of various multi-agent frameworks before and after prompt attacks on MMLU.
unnecessary communication costs-in the case of LLM-MA, reducing token consumption. Figure 4 It illustrates the differences in prompt token consumption between G-Designer and several representative multi-agent designs. We observe that simpler topologies, such as complete graphs and random graphs, consume fewer tokens but show significantly weaker performance. More complex communication structures, like GPTSwarm and DyLAN, achieve superior performance, albeit at the cost of excessive token consumption. For instance, DyLAN's cost on GSM8K is 2.82× that of the random graph, reaching a substantial 2.2e + 7. In contrast, G-Designer elegantly balances both efficiency and task performance, achieving the highest performance across all four benchmarks while maintaining the lowest token cost. For example, on SVAMP, G-Designer surpasses DyLAN by 4% while using only 23.7% of DyLAN's token cost.
this section cite: []

Section: Resource-efficient (training)
We validate G-Designer's training process is resource-friendly from three dimensions: GPU cost, token cost, and wall-clock time. Table 5 showcases that training G-Designer with up to 1000 agents requires less than 4GB of memory. Table 2 unveils that G-Designer not only attains the highest accuracy but also exhibits superior token efficiency and reduced wall-clock time compared to existing baselines, underscoring its effectiveness in multi-agent collaboration.
this section cite: []

Section: Robustness Analysis
Following (Zhuge et al., 2024), we simulate a system prompt attack on one of the five agents. As seen in Figure 5, many trivial structures, such as chain or complete graph, experience significant performance degradation under partial system attacks, with drops as high as 11.0%. Among more sophisticated structures, GPTSwarm, benefiting from its specialized node optimization mechanism, only suffers a minor 0.3% accuracy decline. However, other methods fare less well, with DyLAN and AutoGen showing accuracy drops of 6.2% and 9.9%, respectively. Remarkably, G-Designer demonstrates exceptional robustness against adversarial attacks, maintaining nearly identical performance pre-and post-attack. This resilience can be attributed to its agent encoding capability, which, during optimization, can detect malicious inputs and prune the corresponding edges.
this section cite: ['b67']

Section: Framework Analysis Ablation Study.
We report results for two variants of ourmethod:
(1) w/o SR, which removes the sparsity regularization in Equation ( 14), (2) w/o Anchor, which excludes the anchor structure A anchor , (3) w/o NodeEncoder, removing node encoder in Equation ( 7), and (4) w/o v task in Equation ( 8). As shown in Table 3, removing the task virtual node disrupts G-Designer's task-adaptiveness, leading to the most significant performance drop. The removal of A anchor consistently leads to performance degradation, while the absence of sparsity regularization makes the system more vulnerable to adversarial attacks.
Discussion on anchor topology. Given that G-Designer is initialized with the anchor topology A anchor introduced in Section 4.1, one may question whether the performance gains of G-Designer primarily stem from A anchor itself. In response, we emphasize that the anchor topology corresponds to the simple Chain structure in Table 1, where G-Designer achieves substantial improvements over it, specifically 9.50% ↑ on GSM8K and 8.44% ↑ on SVAMP. Thus, we assert that the superior performance of G-Designer is predominantly attributed to its adaptive topology design rather than the anchor topology itself.
this section cite: []

Section: Conclusion
In this paper, we first present the LLM-based Multi-agent Communication Protocol (MACP), which aims to provide insightful guidance for designing complex multi-agent systems. Furthermore, we propose an effective, adaptive, and robust LLM-powered multi-agent communication graph designer, termed G-Designer, to facilitate the automated design of collaborative AI systems. G-Designer is highly task-aware, dynamically crafting compact and robust communication topologies based on the complexity of the task at hand. We hope that G-Designer will inspire future research on the emergence of self-organizing and self-evolving collective intelligence.
A. Algorithm Workflow
G com = (V, E com ), E com = {(i, j) | Sij ̸ = 0 ∧ (i, j) ∈ E}) / * Guide multi-agent system collaboration * / for iteration t in {1, 2, • • • , K} do for node i in ϕ(G com ) do Agent v i generates R (t) i ← v i (P(t)
sys , P
usr ), P
usr = {Q, ∪ vj ∈Nin(vi) R (t) j } end / * Aggregate solution * / a (t) ← Aggregate(R (t) 1 , R (t) 2 , • • • , R (t) N ) end / * Update G-Designer parameters * / Θ d+1 ← Θ d -α∇ Θ d L G-Designer end B.(t)
this section cite: []

Section: References
Ref_id:b0 Title: Repairagent: An autonomous, llm-based agent for program repair Year: (2024)
Ref_id:b1 Title: Towards Better LLM-based Evaluators through Multi-Agent Debate Year: (2023-08)
Ref_id:b2 Title: Multi-agent collaborative framework for game development Year: (2023)
Ref_id:b3 Title: Benchmarking large language models in retrieval-augmented generation Year: (2024)
Ref_id:b4 Title: Are more llm calls all you need? towards scaling laws of compound inference systems Year: (2024)
Ref_id:b5 Title:  Year: (2021-07-01)
Ref_id:b6 Title: Facilitating multi-agent collaboration and exploring emergent behaviors in agents Year: (2023)
Ref_id:b7 Title: Exploring large language model based intelligent agents: Definitions, methods, and prospects Year: (2024)
Ref_id:b8 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b9 Title: Lm vs lm: Detecting factual errors via cross examination Year: (2023)
Ref_id:b10 Title: Improving factuality and reasoning in language models through multiagent debate Year: (2023)
Ref_id:b11 Title: Complexity-based prompting for multi-step reasoning Year: (2022)
Ref_id:b12 Title: Large language models empowered agent-based modeling and simulation: A survey and perspectives Year: (2023)
Ref_id:b13 Title: Chatllm network: More brains, more intelligence Year: (2023-04-01)
Ref_id:b14 Title: Measuring massive multitask language understanding Year: ()
Ref_id:b15 Title: L2mac: Large language model automatic computer for extensive code generation Year: (2024)
Ref_id:b16 Title: Meta programming for multi-agent collaborative framework Year: (2023-08-01)
Ref_id:b17 Title: Learning multiagent communication from graph modeling perspective Year: (2024)
Ref_id:b18 Title: Large language model-based multi-agent simulation of world wars Year: (2023)
Ref_id:b19 Title: Self-organized agents: A llm multi-agent framework toward ultra large-scale code generation and optimization Year: (2024)
Ref_id:b20 Title: LLM-blender: Ensembling large language models with pairwise ranking and generative fusion Year: (2023-07)
Ref_id:b21 Title: Designing generative driver agent simulation framework in urban contexts based on large language model Year: (2023)
Ref_id:b22 Title: Compiling declarative language model calls into self-improving pipelines Year: (2023)
Ref_id:b23 Title: Variational graph auto-encoders Year: (2016)
Ref_id:b24 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b25 Title: CAMEL: communicative agents for "mind" exploration of large language model society Year: ()
Ref_id:b26 Title: More agents is all you need Year: (2024)
Ref_id:b27 Title: Api-bank: A comprehensive benchmark for tool-augmented llms Year: (2023)
Ref_id:b28 Title: Encouraging divergent thinking in large language models through multi-agent debate Year: (2023)
Ref_id:b29 Title: Program induction by rationale generation: Learning to solve and explain algebraic word problems Year: (2017)
Ref_id:b30 Title: Temporal dynamic weighted graph convolution for multi-agent reinforcement learning Year: (2022)
Ref_id:b31 Title: Dynamic llmagent network: An llm-agent collaboration framework with agent team optimization Year: (2023)
Ref_id:b32 Title: Llm agent with state-space exploration for web navigation Year: (2023)
Ref_id:b33 Title:  Year: (2023)
Ref_id:b34 Title: Are nlp models really able to solve simple math word problems? arXiv preprint Year: (2021)
Ref_id:b35 Title: Learning multi-agent coordination through connectivity-driven communication Year: (2023)
Ref_id:b36 Title: Communicative agents for software development Year: (2023-07-01)
Ref_id:b37 Title: Scaling largelanguage-model-based multi-agent collaboration Year: (2024)
Ref_id:b38 Title: Sentence embeddings using siamese bert-networks Year: (2019)
Ref_id:b39 Title:  Year: (2023)
Ref_id:b40 Title: Auto-gpt: An autonomous gpt-4 experiment Year: (2023)
Ref_id:b41 Title: Distinguished in uniform: Self attention vs Year: (2024)
Ref_id:b42 Title: Solving general arithmetic word problems Year: (2016)
Ref_id:b43 Title: Reflexion: an autonomous agent with dynamic memory and selfreflection Year: ()
Ref_id:b44 Title: Exphormer: Sparse transformers for graphs Year: (2023)
Ref_id:b45 Title: Virtual node tuning for few-shot node classification Year: (2023)
Ref_id:b46 Title: An Open-Ended Embodied Agent with Large Language Models. arXiv e-prints, art Year: (2023-05)
Ref_id:b47 Title: A survey on large language model based autonomous agents Year: (2024)
Ref_id:b48 Title: Deep self-attention distillation for task-agnostic compression of pre-trained transformers Year: (2020)
Ref_id:b49 Title: Selfconsistency improves chain of thought reasoning in language models Year: (2023)
Ref_id:b50 Title: Unleashing cognitive synergy in large language models: A task-solving agent through multi-persona selfcollaboration Year: (2023-07-01)
Ref_id:b51 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022-01-01)
Ref_id:b52 Title: Simple statistical gradient-following algorithms for connectionist reinforcement learning Year: (1992)
Ref_id:b53 Title: Autogen: Enabling next-gen llm applications via multi-agent conversation framework Year: (2023-08-01)
Ref_id:b54 Title: The rise and potential of large language model based agents: A survey Year: (2023)
Ref_id:b55 Title: Depending on yourself when you should: Mentoring llm with rl agents to become the master in cybersecurity games Year: (2024)
Ref_id:b56 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2023-05-01)
Ref_id:b57 Title: React: Synergizing reasoning and acting in language models Year: (2023)
Ref_id:b58 Title: Exchange-of-thought: Enhancing large language model capabilities through cross-model communication Year: (2023)
Ref_id:b59 Title: Cut the crap: An economical communication pipeline for llm-based multiagent systems Year: (2024)
Ref_id:b60 Title: Exploring collaboration mechanisms for llm agents: A social psychology view Year: (2023)
Ref_id:b61 Title: Exploring collaboration mechanisms for llm agents: A social psychology view Year: (2023)
Ref_id:b62 Title: Introduction to graph theory Year: (2006)
Ref_id:b63 Title: Causality-inspired spatial-temporal explanations for dynamic graph neural networks Year: (2024)
Ref_id:b64 Title: Progressivehint prompting improves reasoning in large language models Year: (2023-04-01)
Ref_id:b65 Title: Large language model as a policy teacher for training reinforcement learning agents Year: (2023)
Ref_id:b66 Title: Efficient action space navigation in large language models with a* search Year: (2023)
Ref_id:b67 Title: Language agents as optimizable graphs Year: (2024)
