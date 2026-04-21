Title: Multi-agent Architecture Search via Agentic Supernet
Abstract: Large Language Model (LLM)-empowered multiagent systems extend the cognitive boundaries of individual agents through disciplined collaboration and interaction, while constructing these systems often requires labor-intensive manual designs. Despite the availability of methods to automate the design of agentic workflows, they typically seek to identify a static, complex, onesize-fits-all system, which, however, fails to dynamically allocate inference resources based on the difficulty and domain of each query. To address this challenge, we shift away from the pursuit of a monolithic agentic system, instead optimizing the agentic supernet, a probabilistic and continuous distribution of agentic architectures. We introduce MaAS, an automated framework that samples query-dependent agentic systems from the supernet, delivering high-quality solutions and tailored resource allocation (e.g., LLM calls, tool calls, token cost). Comprehensive evaluation across six benchmarks demonstrates that MaAS (I) requires only 6 ∼ 45% of the inference costs of existing handcrafted or automated multi-agent systems, (II) surpasses them by 0.54% ∼ 16.89%, and (III) enjoys superior cross-dataset and cross-LLM-backbone transferability. The code is available at https:  //github.com/bingreeky/MaAS.

Section: Introduction
Large Language Model (LLM)-based agents (Richards & et al., 2023;Nakajima, 2023;Reworkd, 2023) have made remarkable strides in a spectrum of domains, such as question answering (Zhu et al., 2024a), data analysis (Hong et al., 2024;Li et al., 2024), code generation (Shinn et al., 2023), web navigation (Deng et al., 2024), and data synthesis (Butt et al., 2024), by equipping LLMs with highlevel features, including persona (Wang et al., 2023b;Chen et al., 2024), tools (Shen et al., 2024;Richards & et al., 2023), planning (Qiao et al., 2024;Wu et al., 2024;He et al., 2023), and memory (Zhong et al., 2024;Hatalis et al., 2023;Packer et al., 2023). Building upon the success of single agents, researchers have demonstrated that combining multiple agents, either cooperatively (Zhuge et al., 2024) or competitively (Zhao et al., 2023), can surpass the cognitive and intellectual capabilities of individuals (Du et al., 2023;Liang et al., 2023;Wang et al., 2023b;Jiang et al., 2023;Wu et al., 2023;Zhang et al., 2024a), showcasing the collective intelligence in a society of LLM-agents (Piatti et al., 2024).
Early multi-agent systems, such as CAMEL (Li et al., 2023), AutoGen (Wu et al., 2023), and MetaGPT (Hong et al., 2023), while delivering specialized capacity, often heavily rely on manual configurations, including prompt engineering, agent profiling, and inter-agent communication pipelines (Qian et al., 2024). This dependency significantly limits the rapid adaptation of multi-agent systems to diverse domains and application scenarios (Tang et al., 2023;Zhang et al., 2024c). More recently, the research community has shifted toward automating multi-agent system design. For instance, DsPy (Khattab et al., 2023) and Evo-Prompting (Guo et al., 2023) automate prompt optimization, GPTSwarm (Zhuge et al., 2024) and G-Designer (Zhang et al., 2024b) optimize inter-agent communication, and EvoAgent (Yuan et al., 2024) and AutoAgents (Chen et al., 2023a) self-evolve agent profiling. Nevertheless, they typically focus on automating specific aspects of the system. Subsequently, ADAS (Hu et al., 2024a), AgentSqure (Shang et al., 2024), and AFlow (Zhang et al., 2024c) broaden the design search space. These state-of-the-art (SOTA) methods optimize a single, complex (multi-)agent workflow for a given dataset via different search paradigms, e.g., heuristic search (Hu et al., 2024a), Monte Carlo tree search (Zhang et al., 2024c), and evolution (Shang et al., 2024), surpassing the performance of manually designed systems.
Although the paradigm of searching for a one-size-fitsall multi-agent system appears sufficient to optimize performance-related metrics such as accuracy and pass@k, its performance is largely constrained on resource-related
this section cite: ['b45', 'b37', 'b44', 'b22', 'b30', 'b10', 'b3', 'b6', 'b50', 'b45', 'b41', 'b63', 'b19', 'b74', 'b18', 'b39', 'b78', 'b73', 'b11', 'b31', 'b27', 'b62', 'b29', 'b62', 'b21', 'b40', 'b52', 'b28', 'b16', 'b78', 'b68', 'b48', 'b48']

Section: Building Blocks
CoT Reflexion Debate ReAct Evaluator-optimizer I/O Debate Web search Exit Agentic Supernet I/O + ReAct Tool if-esle Task solution I/O+Reflexion+Debate Task feedback solution Complicated system feedback Task Tools if-esle generator executor Reflexion output critique CoT ReAct Tools execution
this section cite: []

Section: …… ……

this section cite: []

Section: Highschool Physics
How much work is required to charge a 10 µF capacitor to a potential difference of 100 V ?
this section cite: []

Section: Complex Coding
Design an online review website like yelp.com. Implement user authentication and authorization. Create product pages.
Simple arithmetic metrics, such as token cost, LLM calls, and inference latency (Dilemma 1). Specifically, contemporary methods tend to optimize for a complex and resource-intensive agentic system, often involving dozens of LLM API calls and external tool usage (Liu et al., 2023). However, this is far from an optimal solution: for example, in mathematical benchmarks (Hendrycks et al., 2021), Ph.D.-level abstract algebra may indeed require complicated, token-heavy systems, while simple elementary-level arithmetic works well with a single zero-shot I/O. This paradigm becomes even more problematic when applied to benchmarks across multiple task domains (Dilemma 2): for instance, in the GAIA benchmark (Mialon et al., 2023), there is no single system that is optimal for both file reading and web searching tasks, leaving practitioners with no alternative but to split the benchmark and optimize separately (Zhuge et al., 2024). These dilemmas unveil that, the paradigm of automatically optimizing a single multi-agent architecture fails to meet the dynamic and evolving demands of agentic deployment.
To address the above challenges, we propose Multi-agent Architecture Search (MaAS), which, instead of searching for a plausible (possibly non-existent) optimal solution, generates a distribution of multi-agent systems. Technically, we model the optimization of MaAS on the agentic supernet, a probabilistic, continuous agentic architecture distribution that encompasses a vast number of possible multi-agent candidates. The agentic supernet can be seen as a cascaded multi-layer workflow, including ❶ multiple agentic operators (e.g., CoT (Wei et al., 2022), Multi-agent Debate (Du et al., 2023), ReAct (Yao et al., 2023)), as well as ❷ the parameterized probability distributions of operators across layers. During training, MaAS leverages a controller network to sample multi-agent architectures conditioned on input queries. The distribution parameters and operators are jointly updated based on environmental feedback, with the former's gradients approximated via Monte Carlo sampling and the latter's via textual gradient estimation.
During inference, for different queries, MaAS samples a suitable multi-agent system delivering satisfactory resolution and appropriate inference resources, thereby achieving task-customized collective intelligence.
We conduct comprehensive evaluations on seven widely adopted benchmarks, covering diverse use cases in code generation (HumanEval, MBPP), mathematical reasoning (GSM8K, MATH, SVAMP), and diverse tool usage (GAIA). Empirical results demonstrate that MaAS is ❶ high-performing, surpassing existing handcrafted or automated multi-agent systems by 0.54% ∼ 16.89%; ❷ tokeneconomical, outperforming the SOTA baseline AFlow on the MATH benchmark with 15% of the training cost and 25% of the inference cost; ❸ transferable across datasets and LLM-backbones; ❹ inductive, demonstrating strong generalizability to unseen agentic operators.
Briefly put, our key contributions are summarized as follows:
• Paradigm Reformulation: We introduce the concept of agentic supernet, a probabilistic, continuous agentic architecture distribution, which transforms the paradigm of optimizing a single optimal multi-agent system into optimizing the distribution of multiple architectures.
• Practical Solution: We propose MaAS, an agentic supernet-based framework that automatically evolves powerful multi-agent systems and adaptively allocates high-performing and resource-efficient solutions for user queries with varied difficulty, domain and features.
• Experimental Evaluation: Extensive evaluations on six benchmarks demonstrate that our framework discovers novel agentic systems with 0.54% ∼ 16.89% higher performance, significantly lower training/inference costs, transferability across benchmarks and LLMs, and superior inductive capacity.
Controller Agentic Supernet I/O + ReAct Tool if-else Task solution feedback Complicated system Task Tool if-else generator executor E n v ir o n m en t fe ed b a ck Textual gradient -For Debate operator: add transition prompt "..." -For Evaluator-optimizer operator: update gating function -... gradient agent Varied benchmark Topic: Diffculty: Probability Topic: Diffculty: File summarize A board game spinner is divided into three parts labeled , and . The probability of the spinner landing on is ... What's the last line of rhyme on the headstone visible in the background of the photo of the oldest flavor's headstone ... According to github, when was Regression added to numpy.polynomial ... Topic: Diffculty: Web navigation Conditioned sampling
this section cite: ['b34', 'b20', 'b36', 'b78', 'b59', 'b11', 'b67']

Section: Related Work LLM-Agents and Agentic Systems.
Building on the success of single agents (Shen et al., 2024;Zhu et al., 2024b;Zhong et al., 2024), studies have shown that grouping multiple LLM-based agents into multi-agent systems (MAS) can substantially enhance individual model capabilities (Wang et al., 2024a), as demonstrated in early attempts such as AutoGen (Wu et al., 2023), LLM-Debate (Du et al., 2023), and AgentVerse (Chen et al., 2023b). However, they heavily relied on manually crafted designs, which constrained the adaptability and flexibility of agents in addressing unforeseen challenges (He et al., 2023;Chen et al., 2023b). As a result, automated agentic system design has gained increasing attention in the academic community.
this section cite: ['b50', 'b74', 'b62', 'b11', 'b85', 'b19', 'b85']

Section: Automating Agentic Systems.
Efforts to automate the design of agent-based systems can be broadly classified into the following categories: (I) Prompt Optimization, such as PromptBreeder (Fernando et al., 2023), DsPy (Khattab et al., 2023), and EvoPrompt (Guo et al., 2023); (II) Inter-agent Communication, which focuses on orchestrating interactions between agents, including GPTSwarm (Zhuge et al., 2024), DyLAN (Liu et al., 2023), EvoMAC (Hu et al., 2024b), AgentPrune (Zhang et al., 2024a) and G-Designer (Zhang et al., 2024b); and (III) Agent Profiling, represented by AgentVerse (Chen et al., 2023b), EvoAgent (Yuan et al., 2024), and AutoAgents (Chen et al., 2023a). Further, ADAS (Hu et al., 2024a) and AgentSquare (Shang et al., 2024) provide more comprehensive automation for single-agent design, while AFlow (Zhang et al., 2024c) achieves multi-agent workflow automation using Monte Carlo tree search (MCTS). However, these high-performing methods still follow the paradigm of searching for a single final system, whereas MaAS searches for distribution of architectures with lower average inference costs (LLM calls, token cost, etc.).
AutoML. Automating the design of agentic systems is an emerging topic, yet the history of AutoML (He et al., 2021) provides clear precedents. Notably, the progression of agentic automation mirrors that of neural architecture search (NAS) (Ren et al., 2021). Core NAS techniques, such as reinforcement learning (Zoph, 2016), evolutionary algorithms (Liu et al., 2021), Bayesian optimization (BO) (White et al., 2021), and MCTS (Wang et al., 2021), have inspired analogous approaches in agentic automation, from policy gradient in (Zhuge et al., 2024) to evolutionary search in (Yuan et al., 2024), BO in (Shang et al., 2024), and MCTS in (Zhang et al., 2024c). In NAS, however, these black-box methods were eventually eclipsed by efficient supernet training (White et al., 2023), culminating in seminal works like DARTS (Liu et al., 2018) and SNAS (Xie et al., 2018). Inspired by this, we introduce the first MAS searching framework leveraging an agentic supernet, posing new paradigms and challenges for agentic automation.
this section cite: ['b14', 'b28', 'b16', 'b78', 'b34', 'b85', 'b68', 'b48', 'b19', 'b43', 'b79', 'b33', 'b60', 'b53', 'b78', 'b68', 'b48', 'b61', 'b32', 'b64']

Section: Methodology
Figure 2 illustrates the overall workflow of our method. MaAS takes diverse and varying difficulty queries as input and leverages a controller to sample a subnetwork from the agentic supernet for each query, corresponding to a customized multi-agent system. After the sampled system executes the query, MaAS receives environment feedback and jointly optimizes the supernet's parameterized distribution and agentic operators. In the following sections, Section 3.1 formally defines the search space and optimization objective of MaAS, Section 3.2 details how the controller querydependently samples multi-agent structures, and Section 3.3 details the optimization of MaAS.
this section cite: []

Section: Preliminary Search Space.
We first define the basic unit of MaAS's search space, namely the agentic operator as follows:
Definition 3.1 (Agentic Operator). An agentic operator O is a composite LLM-agent invocation process that involves multiple LLM calls and tool usage:
O = {{M i } m i=1 , P, {T i } n i=1 }, M i ∈ M, P ∈ P, T i ∈ T,(1)
where M and M correspond to LLM backbones and the set of available LLMs, respectively. Similarly, P and T represent prompts and tools. m and n denote the number of LLM-agents and tools invoked in the operator, respectively.
Most existing single/multi-agent workflows can be viewed as agentic operators: CoT (Wei et al., 2022) can be considered one with m = 1 and n = 0, denoted as O CoT ; Self-RAG (Asai et al., 2023) similarly involves m = 1 agent allocation, but is equipped with n = 1 retrieval engine, denoted as O SRAG ; Multi-agent debate (Du et al., 2023) involves multiple LLM-agent, multi-turn calls, denoted as O Debate . The feasible set of agentic operators is denoted as O, and we discuss the initialization of O in Section 4.1 and Appendix B.1. We define a multi-agent system as:
G = {V, E}, V ⊂ O, E ∈ V × V,(2)
where V is the set of selected operators in G and E denotes their connectivity. G is constrained as a direct acyclic graph (DAG). Finally, we define the agentic supernet: Definition 3.2 (Agentic Supernet). The agentic supernet is denoted as
A = {π, O} = {{π ℓ (O)} O∈O } L ℓ=1 , where: π ℓ (O) = p(O | A 1:ℓ-1 ), O ∈ O, A 1:ℓ-1 = {{π k (O)} O∈O } ℓ-1 k=1 ,(3)
where π ℓ (O) represents the probability of operator O present at layer ℓ, conditioned on the preceding layers A 1:ℓ-1 . The supernet induces a joint distribution over all possible multi-layer operator configurations:
p(G) = L ℓ=1 O∈O π ℓ (O) I O∈V ℓ ,(4)
where I O∈V ℓ is the indicator function for the inclusion of O in the set of active operators V ℓ at layer ℓ.
Problem Formulation. Given a benchmark D comprising multiple queries q and their corresponding oracle answers/solutions a, the objective of MaAS is not to identify a single optimal agentic system like previous practices (Zhang et al., 2024c;Zhuge et al., 2024), but to optimize a conditional probability distribution as follows:
max P(G|q) E (q,a)∼D, G∼P(G|q) U (G; q, a)-λ•C(G; q) , s.t. G ⊂ A (5
)
where P(G|q) is a distribution that generates querydependent agentic architectures. U (•) and C(•) represent the utiulity/performance and cost of G for query q, respectively, and λ is a trade-off parameter.
this section cite: ['b59', 'b0', 'b11', 'b78']

Section: Agentic Architecture Sampling
The core of MaAS lies in tailoring a customized multi-agent system for each user query, which may vary in difficulty and domain, to deliver a satisfactory solution:
p(a|q, π, O) = e(a|G) Q ϕ (G|q, π, O) dG,(6)
where Q ϕ represents the controller network, which takes the query q, the parameterized distribution π, and the available operators O, and outputs the sampled agentic architecture G. Q ϕ is parameterized by ϕ, and e(•|•) denotes producing solution via executing G. we implement Q ϕ as follows:
Q ϕ (G|q, π, O) = L ℓ=1 π ℓ (V ℓ |q, {V h } ℓ-1 h=1 ),(7)
where V h denotes the selected operators at layer h. The selection of V ℓ is conditionally dependent on the query q and the operators from the previous layers. However, not all queries require execution across L layers. As discussed in Section 1, many questions can be resolved with a simple zero-shot I/O (Zhang et al., 2024b), rendering L layers unnecessarily redundant. To address this, we introduce an early-exit operator, denoted as O exit . During sampling, if O exit is encountered, the process exits early:
Q ϕ (G|q, π, O) = L ℓ=1 π ℓ (V ℓ |q, {V h } ℓ-1 h=1 ) • I Oexit / ∈V ℓ + I Oexit∈V ℓ • δ ℓ -ℓ exit ,(8)
where ℓ exit denotes the layer at which O exit appears, and δ(•) is the Kronecker delta function. We implement the sampling process π ϕ with a Mixture-of-Expert (MoE)-style network (Shazeer et al., 2017;Huang et al., 2024):
π ℓ : q → V ℓ , V ℓ = {O ℓ1 , O ℓ2 , • • • , O ℓt }, t = arg min k∈{1,••• ,|O|} j<k S ↓ k > thres,(9)
where S ↓ = sort(S, desc), and
S ∈ R |O| = [S 1 , • • • , S |O| ]
represents the activation scores of all feasible operators w.r.t. q. Note that thres is a threshold value that governs operator activation. Operators are activated sequentially, starting from the one with the highest score, and the process continues until the cumulative score exceeds thres. This ensures that the number of selected operators per layer is query-dependent, allowing MaAS to dynamically allocate resources based on task complexity. S is given by:
S i = FFN(v(q)∥ O∈V1 v(O)∥ • • • ∥ O∈V ℓ-1 v(O)),
where v(•) denotes the embedding function using lightweight models like MiniLM (Wang et al., 2020) and Sentence-Bert (Reimers, 2019), and ∥ represents concatenation. The detailed implementation of v(•) is placed in Appendix B.2.
Upon completing the sequential sampling procedure in MaAS, a task-specific multi-agent system G is generated and executed to produce the answer a. In the next section, we elucidate the process of updating the agentic supernet based on environmental feedback.
this section cite: ['b49', 'b26', 'b55', 'b42']

Section: Cost-constrained Supernet Optimization
We present the optimization objective of MaAS as follows:
min π,O E (q,a)∼D,G∼Q ϕ [-p(a|q, π, O) + λ • C(G; q)] (10
)
where C(•) evaluates the cost of multi-agent systems, represented by token cost, and λ is the trade-off parameter. The term p(a|q, π, O) in Equation ( 10) corresponds to Equation ( 6), where the calculation of e(a|G) often involves external tools or API-based LLM calls, rendering it nondifferentiable. Therefore, we employ an empirical Bayes Monte Carlo procedure (Carlin & Louis, 2000;Yan et al., 2021) to estimate the gradient w.r.t the distribution π:
∇ π L ≈ 1 K (q,a)∈D K k=1 m k ∇ π p (G k ) , m k = p(a|q, G k ) i p(a|q, G i ) -λ • C(G k ; q) i C(G i ; q) ,(11)
where m k denotes the cost-aware importance weights of the agentic architecture. Intuitively, the distribution π is updated to favor multi-agent systems that generate highquality solutions with minimal token cost.
Sampled MAS Task feedback Ensemble Text Textual Gradient Operator Gradient add a debater LLM to debate operator Temperature Gradient lower the ensemble LLM's temperature for stability Prompt Gradient add few-shot <example> to the refine process However, the gradient w.r.t operators ∇ O L cannot be computed similarly. As shown in Equation ( 1), operators include black-box tool usage and natural language prompts, making numerical gradient updates infeasible. To address this, we utilize agent-based textual gradient (Hao et al., 2023;Liu et al., 2023;Hu et al., 2024b;Zhou et al., 2024) to approximate the backpropagation for agentic operators, as visualized in Figure 3 and formalized as follows:
∇ O L = T P ⊕ T T ⊕ T N , T x ∈ T, x ∈ {P, T , N } (12
)
where T P , T T , T N represent agent-generated gradient analyses in textual format, corresponding to updates of the prompt, model temperature, and operator node structure
* / for layer ℓ ← 1 to L do V ℓ ← π ϕ (V ℓ |q, {V h } ℓ-1 h=1 ); ▷ Eq. 9 if ℓ = L or O exit ∈ V ℓ then
break// Exit when reaching maximal sampling depth or encountering the early-exit operator
Obtain G ← ⟨V 1 , • • • , V ℓ ⟩
for query q; ▷ Eq. 8 / * Execute sampled MAS * / Execute G and obtain ã ← e(a|G); ▷ Eq. 6 / * Self-evolve agentic supernet * / Compute loss w.r.t. π, ∇ π L; ▷ Eq. 11 Estimate loss w.r.t O via textual gradient; ▷ Eq. 12 Update π and O accordingly; ▷ Eq. 10 (such as merging, splitting, altering, etc.), respectively. See prompts in Appendix B.3. In this way, the core components of the agentic supernet, namely the agentic operators and their connectivity, are jointly updated, enabling the fully automated evolution of multi-agent systems. We summarize the notations in Table 5, and the algorithm in Algorithm 1. Baselines. We compare MaAS with three series of agentic baselines: (1) single agent execution methods, including CoT (Wei et al., 2022), ComplexCoT (Fu et al., 2022), Self-Consistency (Wang et al., 2023a); (2) hand-craft multiagent systems, including MultiPersona (Wang et al., 2023b), LLM-Debate (Du et al., 2023), LLM-Blender (Jiang et al., 2023), DyLAN (Liu et al., 2023), AgentVerse (Chen et al., 2023b) and MacNet (Qian et al., 2024); (3) (partially or fully) autonomous multi-agent systems, including GPTSwarm (Zhuge et al., 2024), AutoAgents (Chen et al.,  2023a), ADAS (Hu et al., 2024a), AgentSquare (Shang et al., 2024) and AFlow (Zhang et al., 2024c). More details on baseline setups are provided in Appendix C.2.
Implementation details. We leverage both close-source LLM (gpt-4o-mini-0718 (OpenAI, 2024)) and opensource LLM (Qwen-2.5-72b-instruct (Yang et al., 2024) and llama-3.1-70b (Dubey et al., 2024)). All models are accessed via APIs with the temperature set to 1. We set the number of layers as L = 4, the cost penalty coefficient λ as λ ∈ {1e -3, 5e -3, 1e -2}, and the sampling times K = 4. thres = 0.3 for Equation ( 9).
this section cite: ['b4', 'b65', 'b17', 'b34', 'b75', 'b59', 'b15', 'b11', 'b27', 'b34', 'b85', 'b40', 'b78', 'b48', 'b66', 'b12']

Section: Performance Analysis
We compare MaAS with 14 baselines on the GSM8K, MATH, MultiArith, HumanEval, and MBPP benchmarks in Table 1, and with 10 baselines on GAIA in Table 2. The following observations can be made:
Obs.❶ MaAS achieves optimal performance across all task domains. The multi-agent system optimized by MaAS outperforms manually designed methods by an average of 3.90 ∼ 6.40% and existing automated methods by 2.07 ∼ 8.26%. Overall, as for mathematical reasoning and code generation, MaAS achieves an average best score of 83.59%, demonstrating its versatility and superiority.
Table 2 shows a comparison of MaAS with automated systems and three additional baselines, including Auto-GPT (Richards & et al., 2023), TapeAgent (Bahdanau et al., 2024), and Sibyl (Wang et al., 2024b) on the GAIA benchmark. GAIA encompasses tasks from various domains such as web browsing, file reading, and multimodal understanding, making it challenging to pursue a single optimal multiagent system for all tasks. Thus, the modest improvements of AFlow and ADAS over vanilla LLMs (only 3.35% ↑ and 2.04% ↑ on average) are understandable. In contrast, MaAS can adaptively sample customized agentic systems for different domains, achieving 18.38% and 17.61% improvements on Level 1 and 2 tasks, respectively.
this section cite: ['b45', 'b2']

Section: Cost Analysis
To answer RQ2, we demonstrate that MaAS is both training/inference cost-efficient from the following three dimensions: (1) token cost, (2) API cost, and (3) wall-clock time, as shown in Table 3 and Figure 4. We observe:
Obs.❷ MaAS's optimization is resource-friendly. As shown in Figure 4 (Training Tokens), among the various optimization-oriented agentic workflows, MaAS achieves the highest accuracy with the least training token consumption. While AFlow's accuracy is comparable to that of MaAS, its training cost reaches 22.50$, which is 6.8× that of MaAS (merely 3.38$). Additionally, existing agentic  automation pipelines are relatively time-consuming, with DyLAN taking 508 minutes and GPTSwarm taking 129 minutes. In contrast, the optimization wall-clock time of MaAS requires only 53 minutes.
Obs.❸ Agentic supernet enjoys superior token economy during inference. As shown in Figure 4 (Inference API Cost), MaAS achieves the highest accuracy with an API cost of 0.42$, demonstrating its high performance and token economy. Although AgentSquare's API cost is slightly lower than MaAS's, this is due to its limitation to a singleagent search, which severely restricts its performance (resulting in a 4% drop compared to MaAS). Table 3 further highlights that MaAS has the lowest prompt/completion token consumption, the lowest API cost, and the shortest wall-clock time during inference. These advantages can be attributed to the agentic supernet's ability to dynamically allocate resources based on the difficulty of the query.
this section cite: []

Section: Easy query
Query: How many positive integers less than 103 have an odd number of positive divisors?
this section cite: []

Section: Case Study
In this section, we explore and visualize the intrinsic mechanisms of the agentic supernet. Figure 5 showcases the probability distributions of the agentic supernet when faced with different queries, and Figure 6 presents the multi-agent systems designed by MaAS for queries from the MATH, GAIA, and HumanEval benchmarks. We have:
Obs.❹ MaAS learns to query-aware early exit from the reasoning process. As shown in Figure 5, when faced with the easy queries (a) and (b), MaAS exits multi-agent architecture sampling at the second layer with probabilities of 0.37 and 0.47, respectively, selecting the early-exit operator. Notably, query (b) chose two agentic operators at the first
this section cite: []

Section: CoT Prompting
def triangle_area(a, h):
""" Given length of a side and high return area for a triangle. >>> triangle_area(5, 3) 7.5 """
this section cite: []

Section: Query Workflow
According to wikipedia, how many Asian countries still have a monarchy and access to the sea in 2021? The number of layers 𝐿 Cost penalty λ Population size 𝑁 Sampling times 𝐾 Figure 7. Parameter sensitivity analysis of MaAS. The unit of cost per query (right) and performance (left) is 10 -3 • $ and pass@1 (%), respectively.
layer: direct I/O and ReAct, demonstrating MaAS's ability to dynamically allocate different operators at each layer (corresponding to Equation ( 9)). For the more challenging queries (c) and (d), MaAS sampled additional layers, further proving its ability to customize the multi-agent system based on query awareness. This is also visualized by Figure 8, in which the probability of O exit becomes increasingly high with the supernet depth increases.
this section cite: []

Section: Framework Analysis Sensitivity Analysis
We analyze the sensitivity of MaAS to three core parameters: the number of layers in the agentic supernet L, the cost penalty coefficient λ in Equation (10), and the sampling count K in Equation (11). The results are presented in Figure 7. For the parameter L, we observe a significant performance improvement as L increases from 2 to 4 (89.5% → 92.8%). However, further increases yield only marginal performance gains while incurring higher per-query inference costs. Considering both performance and cost, we select L = 4. For the parameter λ, we find that larger values lead MaAS to favor more cost-efficient solutions, albeit with some performance degradation. For the parameter K, we note that performance is suboptimal with highest variance when K = 2. Increasing K to 4 effectively achieves a satisfactory low-variance estimation.
this section cite: ['b88', 'b89']

Section: Ablation Study
We perform an ablation study on three key components of MaAS: (1) w/o ∇ O L, removing the textual gradient in Equation ( 12); (2) w/o O exit , removing the early-exit operator in Equation ( 8); and
(3) w/o C(•), eliminating the cost constraint in Equation ( 10). We observe from Table 4 that removing the textual gradient causes the largest performance drop, as it disables MaAS's self-evolving capability. Removing O exit and C(•) results in little impact on performance, but it weakens MaAS's query-dependent nature and unnecessarily increases the inference cost.
this section cite: []

Section: Transferability Analysis.
We evaluate whether the agentic supernet of MaAS is (1) model-agnostic and (2) generalizable across datasets, with results presented in Tables 7 and 8. As shown, the agentic supernet optimized by MaAS transfers well to models such as Qwen-2.5-70b, with 4.98% ∼ 5.50% ↑ in performance, while also demonstrating strong cross-dataset generalization.
this section cite: []

Section: Inductive Analysis.
To evaluate whether MaAS possesses inductive capabilities, i.e., the ability to generalize to unseen agentic operators, we select the Debate (Du et al., 2023) operator as a holdout. We then compare the operator distribution of MaAS during inference with and without Debate in Figures 8 and 9. The results demonstrate that MaAS can still reasonably activate and utilize the unseen operator at an appropriate proportion.
this section cite: ['b11']

Section: Conclusion
In this paper, we for the first time shift the paradigm of automated multi-agent system design from seeking a (possibly non-existent) single optimal system to optimizing a probabilistic, continuous distribution of agentic architectures, termed the agentic supernet. Building on this concept, we propose MaAS, which dynamically samples multi-agent systems that deliver satisfactory performance and token efficiency for user queries across different domains and varying levels of difficulty. We believe that MaAS paves the way toward fully automated, self-organizing, and self-evolving collective intelligence.
A. Notations An L-layer probabilistic agentic supernet, consisting of a distribution π and a set of feasible operators O. π
The distribution associated with the agentic supernet. U (G; q, a)
The utility evaluator of G with respect to query q and answer a. C(G; q, a)
The cost evaluator of G with respect to query q and answer a. Q ϕ
The controller network parameterized by ϕ. e(a∥G)
Execution of G to produce the answer a. V ℓ
The selected operators at layer ℓ of the agentic supernet A. O exit
The early-exit operator. v(•)
The text embedding function.
∇ π L
The gradient of the loss L with respect to the distribution π. ∇ O L
The textual gradient of the loss L with respect to the operators O.
this section cite: []

Section: References
Ref_id:b0 Title: Selfrag: Learning to retrieve, generate, and critique through self-reflection Year: (2023)
Ref_id:b1 Title: Program synthesis with large language models Year: (2021)
Ref_id:b2 Title: Tapeagents: a holistic framework for agent development and optimization Year: (2024)
Ref_id:b3 Title: Automated benchmark creation with agent interaction Year: (2024)
Ref_id:b4 Title: Empirical bayes: Past, present and future Year: (2000)
Ref_id:b5 Title: Autoagents: A framework for automatic agent generation Year: (2023)
Ref_id:b6 Title: From persona to personalization: A survey on role-playing language agents Year: (2024)
Ref_id:b7 Title:  Year: (2021-07-01)
Ref_id:b8 Title: Facilitating multi-agent collaboration and exploring emergent behaviors in agents Year: (2023)
Ref_id:b9 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b10 Title: Mind2web: Towards a generalist agent for the web Year: (2024)
Ref_id:b11 Title: Improving factuality and reasoning in language models through multiagent debate Year: (2023)
Ref_id:b12 Title: The llama 3 herd of models Year: (2024)
Ref_id:b13 Title: Graphrouter: A graph-based router for llm selections Year: (2024)
Ref_id:b14 Title: Promptbreeder: Self-referential self-improvement via prompt evolution Year: (2023)
Ref_id:b15 Title: Complexity-based prompting for multi-step reasoning Year: (2022)
Ref_id:b16 Title: Connecting large language models with evolutionary algorithms yields powerful prompt optimizers Year: (2023)
Ref_id:b17 Title: Chatllm network: More brains, more intelligence Year: (2023-04-01)
Ref_id:b18 Title: Memory matters: The need to improve long-term memory in llm-agents Year: (2023)
Ref_id:b19 Title: A multi-agent collaborative framework with roleplaying and iterative feedback for causality explanation generation Year: (2021)
Ref_id:b20 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b21 Title: Meta programming for multi-agent collaborative framework Year: (2023-08-01)
Ref_id:b22 Title: Data interpreter: An llm agent for data science Year: (2024)
Ref_id:b23 Title: Automated design of agentic systems Year: (2024)
Ref_id:b24 Title: Self-evolving multi-agent collaboration networks for software development Year: (2024)
Ref_id:b25 Title: Multi-agent-based code generation with iterative testing and optimisation Year: (2023)
Ref_id:b26 Title: Harder tasks need more experts Year: (2024)
Ref_id:b27 Title: LLM-blender: Ensembling large language models with pairwise ranking and generative fusion Year: (2023-07)
Ref_id:b28 Title: Compiling declarative language model calls into self-improving pipelines Year: (2023)
Ref_id:b29 Title: CAMEL: communicative agents for "mind" exploration of large language model society Year: (2023)
Ref_id:b30 Title: Autokaggle: A multi-agent framework for autonomous data science competitions Year: (2024)
Ref_id:b31 Title: Encouraging divergent thinking in large language models through multi-agent debate Year: (2023)
Ref_id:b32 Title: Differentiable architecture search Year: (2018)
Ref_id:b33 Title: A survey on evolutionary neural architecture search Year: (2021)
Ref_id:b34 Title: Dynamic llmagent network: An llm-agent collaboration framework with agent team optimization Year: (2023)
Ref_id:b35 Title: Self-refine: Iterative refinement with selffeedback Year: (2023)
Ref_id:b36 Title: Gaia: a benchmark for general ai assistants Year: (2023)
Ref_id:b37 Title:  Year: (2023)
Ref_id:b38 Title: Gpt-4o mini: Advancing cost-efficient intelligence Year: (2024)
Ref_id:b39 Title: Cooperate or collapse: Emergence of sustainability behaviors in a society of llm agents Year: (2023)
Ref_id:b40 Title: Scaling largelanguage-model-based multi-agent collaboration Year: (2024)
Ref_id:b41 Title: Automatic agent learning from scratch via self-planning Year: (2024)
Ref_id:b42 Title: Sentence embeddings using siamese bert-networks Year: (2019)
Ref_id:b43 Title: A comprehensive survey of neural architecture search: Challenges and solutions Year: (2021)
Ref_id:b44 Title:  Year: (2023)
Ref_id:b45 Title: Auto-gpt: An autonomous gpt-4 experiment Year: (2023)
Ref_id:b46 Title: Solving general arithmetic word problems Year: (2016)
Ref_id:b47 Title: An architecture search framework for inference-time techniques Year: (2024)
Ref_id:b48 Title: Automatic llm agent search in modular design space Year: (2024)
Ref_id:b49 Title: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer Year: (2017)
Ref_id:b50 Title: Solving ai tasks with chatgpt and its friends in hugging face Year: (2024)
Ref_id:b51 Title: Reflexion: an autonomous agent with dynamic memory and selfreflection Year: ()
Ref_id:b52 Title: Verifai: verified generative ai Year: (2023)
Ref_id:b53 Title: Sampleefficient neural architecture search by learning actions for monte carlo tree search Year: (2021)
Ref_id:b54 Title: A survey on large language model based autonomous agents Year: ()
Ref_id:b55 Title: Deep self-attention distillation for task-agnostic compression of pre-trained transformers Year: (2020)
Ref_id:b56 Title: Selfconsistency improves chain of thought reasoning in language models Year: (2023)
Ref_id:b57 Title: Simple yet effective agent framework for complex real-world reasoning Year: (2024)
Ref_id:b58 Title: Unleashing cognitive synergy in large language models: A task-solving agent through multi-persona selfcollaboration Year: (2023-07-01)
Ref_id:b59 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022-01-01)
Ref_id:b60 Title: Bananas: Bayesian optimization with neural architectures for neural architecture search Year: (2021)
Ref_id:b61 Title: Neural architecture search: Insights from 1000 papers Year: (2023)
Ref_id:b62 Title: Autogen: Enabling next-gen llm applications via multi-agent conversation framework Year: (2023-08-01)
Ref_id:b63 Title: Can graph learning improve planning in llm-based agents? Year: (2024)
Ref_id:b64 Title: Snas: stochastic neural architecture search Year: (2018)
Ref_id:b65 Title: Fp-nas: Fast probabilistic neural architecture search Year: (2021)
Ref_id:b66 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b67 Title: React: Synergizing reasoning and acting in language models Year: (2023)
Ref_id:b68 Title: Towards automatic multi-agent generation via evolutionary algorithms Year: (2024)
Ref_id:b69 Title: Cut the crap: An economical communication pipeline for llm-based multiagent systems Year: (2024)
Ref_id:b70 Title: Architecting multiagent communication topologies via graph neural networks Year: (2024)
Ref_id:b71 Title: Automating agentic workflow generation Year: (2024)
Ref_id:b72 Title: Automatic chain of thought prompting in large language models Year: (2022)
Ref_id:b73 Title: Understanding the competition behaviors in large language model-based agents Year: (2023)
Ref_id:b74 Title: Memorybank: Enhancing large language models with long-term memory Year: (2024)
Ref_id:b75 Title: Symbolic learning enables self-evolving agents Year: (2024)
Ref_id:b76 Title: Autotqa: Towards autonomous tabular question answering through multi-agent large language models Year: (2024)
Ref_id:b77 Title: Knowledge-augmented planning for llm-based agents Year: (2024)
Ref_id:b78 Title: Language agents as optimizable graphs Year: (2024)
Ref_id:b79 Title: Neural architecture search with reinforcement learning Year: (2016)
Ref_id:b80 Title: We follow the official implementation Year: ()
Ref_id:b81 Title: Self-consistency. To enhance robustness, we aggregate five CoT-generated solutions Year: ()
Ref_id:b82 Title: We instantiate five LLM-agents, each assigned a distinct role, which participate in up to two rounds of debate, after which the final decision is determined via majority voting Year: ()
Ref_id:b83 Title: We choose two gpt-4o-mini, one Qwen-2.5-72b, and one llama-3.1-70b to empower LLM-Blender Year: (2023)
Ref_id:b84 Title: We directly utilize the implementation from Year: (2023)
Ref_id:b85 Title: The experimental setup follows the original implementation from Year: (2023)
Ref_id:b86 Title: For MacNet (Qian et al., 2024), we adopt the "MacNet-MESH" variant Year: ()
Ref_id:b87 Title: The method is implemented in accordance with the original settings described in Year: (2024)
Ref_id:b88 Title: We adhere to the official configuration specified in Year: (2023)
Ref_id:b89 Title: The implementation details are directly inherited from Year: ()
Ref_id:b90 Title: We utilize the modular search framework introduced in Year: (2024)
Ref_id:b91 Title: AFlow operates with both gpt-4o-mini and claude-3.5-sonnet. To maintain fairness under homogeneous conditions, we restrict AFlow to gpt-4o-mini and set MAX ITERATION=20 Year: (2024)
Ref_id:b92 Title: Results We have visualized the evolution of operator sampling trends as the sampling count increases Year: ()
