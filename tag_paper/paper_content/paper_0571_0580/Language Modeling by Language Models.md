Title: Language Modeling by Language Models
Abstract: Can we leverage LLMs to model the process of discovering novel language model architectures? Inspired by real research, we propose a multi-agent LLM approach that simulates the conventional stages of research, from ideation and literature search (proposal stage) to design implementation (code generation), generative pre-training, and downstream evaluation (verification). Using ideas from scaling laws, our system Genesys employs a Ladder of Scales approach; new designs are proposed, adversarially reviewed, implemented, and selectively verified at increasingly larger model scales (14M∼350M parameters) with a narrowing budget (the number of models we can train at each scale). To help make discovery efficient and factorizable, Genesys uses a novel genetic programming backbone, which we show has empirical advantages over commonly used direct prompt generation workflows (e.g., ∼86% percentage point improvement in successful design generation, a key bottleneck). We report experiments involving 1,162 newly discovered designs (1,062 fully verified through pre-training) and find the best designs to be highly competitive with known architectures (e.g., outperform GPT2, Mamba2, etc., on 6/9 common benchmarks). We couple these results with comprehensive system-level ablations and formal results, which give broader insights into the design of effective autonomous LLM-driven discovery systems.

Section: Introduction
Automated scientific discovery (ASD) (Langley, 1987;Wang et al., 2023), which aims to simulate all aspects of the conventional research process -from ideation/system design to experiment execution, has the promise of changing the way that research is performed by making it more accessible, efficient, and less error-prone. However, while many new large language model (LLM)-driven ASD systems have been recently proposed, including AI Scientist (Lu et al., 2024a;Yamada et al., 2025) and others (Liu et al., 2024b;Jansen et al., 2025b;Schmidgall et al., 2025b), much of this work focuses on open-ended research with unclear goals and where discoveries are hard to verify. This motivates the development of new tasks that address foundational challenges in ASD, tasks that are broad in scope and address impactful research problems, but that have clear goals and criteria for success.
In this paper, we focus on discovery in machine learning and ask: Can we model the process of discovering novel language model architectures that improve on the standard transformer architecture? While transformers (Vaswani et al., 2017) remain the de facto standard architecture for language models, research into alternative architectures (Dao & Gu, 2024;Sun et al., 2024Sun et al., , 2023;;Peng et al., 2024) and transformer variants (Tay et al., 2022b) remains an active and important area of research with connections to the mature field of neural architecture search (NAS) (Elsken et al., 2019). In contrast to open-ended research tasks, architecture research involves a clear goal (i.e., producing an executable architecture design) and offers many metrics for evaluating success. It also introduces Figure 1: Can we discover novel language model architectures? A high-level illustration of our approach, consisting of a discovery environment (Left), or LMADE, that provides knowledge access (Knowledge Engine) and automated evaluation (Verification Engine). Right: Genesys, a LLM-driven agent system that proposes, implements, then verifies new designs using design and verifier agents (see algorithmic workflow, far right) and feedback from LMADE. new challenges for ASD, including requiring deep literature understanding, careful management of resources (e.g., pretraining compute), and the need to write code in an unbounded design space.
Our approach is shown in Figure 1 and factors into a discovery environment that provides the foundational tools for ASD and a discovery system that produces discovery artifacts using feedback from the environment. Our Language Model Architecture Discovery Environment (LMADE) specifically consists of two core resources, a general-purpose knowledge engine that provides access to the academic literature and a verification engine that provides tools for performing model pretraining and evaluation. Our system Genesys then consists of LLM-driven designer agents that propose new research ideas and produce executable architecture designs, and verifier agents that select designs and perform on-the-fly generative pre-training. At the core of Genesys is an evolution tree that stores seed designs and new discovery artifacts. These artifacts are implemented using a special code construct called a generalized autoregressive block (GAB) (Figure 3) that is capable of expressing a wide range of neural architecture types and factorizable into discrete tree representations that allow us to employ efficient genetic programming (GP)-style optimization.
We performed large-scale discovery experiments that resulted in 1,062 new architecture designs fully verified through pre-training (at the 14M-350M parameter scales). To make verification feasible, we employ a Ladder-of-Scales approach where new designs are verified on increasingly larger model scales with a controlled budget, closely following the methodology used in research on small LMs (Lu et al., 2024b;Hu et al., 2024). To our knowledge, our work constitutes the largest ASD experiment of its kind, involving >1 billion tokens, 2.76M lines of code, and 86K agent interactions. 2We find that our system produces highly competitive designs, e.g., ones that outperform comparable transformer and mamba2 models (Dao & Gu, 2024) in 6 / 9 common downstream tasks. These results are significant and show the feasibility of LLM-driven discovery for competitive ML research. Through systematic ablations, we also find that our system leads to more stable discovery (e.g., measurable improvements in the fitness of new designs over time) and effective code generation (e.g., ∼86% percentage point improvement in successful design generation), which give broader insight into how to effectively build large-scale discovery systems.
2 Related work AI in Scientific Discovery AI approaches to ASD have recently proliferated, notably in biomedical science (Jumper et al., 2021;Cheng et al., 2023;Wong et al., 2024), material science (Park et al.,Figure 3: What are we trying to discover? 1 ⃝ visualizes standard autoregressive LMs and the blocks that our system aims to discover (implemented via the Pytorch modules in 2 ⃝ and 4 ⃝ with function type (X,Z) → (X,Z)). 5
⃝ shows an implemented block for the GPT and its factorization into a tree 3 ⃝ that shows the units in that block (e.g., multi-head attention implemented in 6 ⃝).
2024; Merchant et al., 2023), and other areas (Chen et al., 2024;Nearing et al., 2024). As discussed above, recent attempts at fully end-to-end research via LLM-driven systems, such as AI Scientist (Lu et al., 2024a;Yamada et al., 2025), AgentLab (Schmidgall et al., 2025a), CodeScientist (Jansen et al., 2025a), and AIGS (Liu et al., 2024a), have focused on open-ended research tasks with unclear goals and evaluation protocols. In contrast, we focus on the challenging task of neural architecture discovery, which offers a clear objective yet involves many new challenges for ASD. Language Model Architectures Our work relates to research on efficient transformer variants (Xiao et al., 2024;Ye et al., 2024), and alternative architectures, such as state-space models (Gu et al., 2022;Dao & Gu, 2024), modern RNNs (Peng et al., 2024;Beck et al., 2024;Feng et al., 2024), and testtime training (Sun et al., 2024;Behrouz et al., 2024). Since our system aims to perform autonomous research in this area, much of this related work is modeled directly and stored in a reference library shown in Figure 2 that serves as the background work used for system ideation.
Neural Architecture Search (NAS) Lastly, we take inspiration from the NAS literature (Chitty-Venkata et al., 2022;White et al., 2023;Elsken et al., 2019;Chen et al., 2023) which has the same aim of discovering improved architectures. Unlike this work, which traditionally searches fixed operation spaces (e.g., attention heads, convolution kernels), we aim for a broader space of operations and architectures and, importantly, attempt to model the broader scientific discovery process. We follow many approaches in NAS that employ genetic programming techniques (GP) (Koza, 1994) and more recent approaches that mix GP with LLMs (Hemberg et al., 2024;Romera-Paredes et al., 2024).
this section cite: ['b33', 'b60', 'b67', 'b57', 'b13', 'b53', 'b52', 'b15', 'b25', 'b13', 'b28', 'b10', 'b62', 'b39', 'b9', 'b40', 'b67', 'b65', 'b68', 'b13', 'b3', 'b16', 'b53', 'b4', 'b11', 'b61', 'b15', 'b8', 'b32', 'b24', 'b47']

Section: Language Model Architecture Discovery
As illustrated in Fig. 3 1 ⃝, standard LMs work by embedding input, then applying N layer or block transformations over that input to produce a final representation (e.g., one that can be used for next token prediction as in autoregressive LMs). Central to any layer/block is a block design, concretely a piece of code B LM , that dictates how information flows through a network. Our goal is to jointly discover novel autoregressive block designs B LM while also modeling the broader research process associated with producing B LM . In this section, we define this problem formally ( § 3.1) and introduce our Language Model Architecture Discovery Environment (LMADE) ( § 3.2) that provides the foundational tools used for discovery and for evaluating block designs B LM .
this section cite: []

Section: Problem Definition and Goals
We define architecture discovery as a program search problem that involves finding an optimal program BLM (in the space of valid programs B LM ) that maximizes some fitness function : B LM → R. We can define this formally as:
BLM = argmax B LM ∈B LM F(B LM ) , with F(B LM ) = 1 M • K M i=1 K j=1 Perf(B LM , D i , S j ) (1)
where, following standard practice, F is defined as the average empirical performance Perf of B LM on a set of M downstream tasks {D 1 , ..., D M } across K different model scales {S 1 , ..., S K } (i.e., model parameter sizes). Operationally, a valid program will be any syntactically correct instantiation of the GABBase module in Figure 3 2 ⃝ that involves (via the implementation of _forward) a differentiable, causal transformation of input tensor X ∈ R batch_size×seq_len×emb_dim to an output tensor of the same dimension and type, along with the other semantic constraints (see Table 6).
The role of LMADE is to provide input and feedback I to a separate discovery system that produces new block designs, as well as to provide all other tools needed for checking the validity of designs, verifying them through experiments, and computing F. We consider these components next, followed by a discussion of our discovery system, Genesys, in § 4.
this section cite: []

Section: Language Model Architecture Discovery Environment (LMADE)
LMADE consists of two core utilities, a knowledge engine and a verification engine that provide signal and feedback I to a discovery system. The Knowledge Engine (KE) provides information from the academic literature that is needed to produce new research ideas. It specifically includes a manually curated reference library (Fig. 2) of 297 LM papers (stored in a searchable vector store) coupled with code, as well as tools for querying ArXiv, Semantic Scholar (Kinney et al., 2023), and the web via services such as Perplexity.ai. More details are provided in § B. 1.3. The Verification Engine (VE) then provides tools for verifying the correctness of designs and executing experiments. In the former case, the VE uses a general code construct, called a Generalized Autoregressive Block (GAB) (operationalized by the GABBase class in Figure 3, see code templates in App B.3) to represent all architecture designs B LM and uses the structure of this module to check the syntactic correctness of each design. Semantically, the VE also includes a Symbolic checker that performs static (AST-based) and runtime (PyTorch-based) code analysis to check for differentiability, causality, numerical stability, and the efficiency of a code design as detailed in Table 6 (further details in § B. 1.2). Finally, VE can perform design verification by automating pretraining on a filtered SmolLM corpus (Allal et al., 2025) and evaluation on 29 selected LM-Eval benchmarks (Gao et al., 2024). Standard pretraining protocols (Biderman et al., 2023) are applied (see § B for more details).
this section cite: ['b31', 'b2', 'b6']

Section: Genesys: Genetic Discovery System
Using resources from LMADE, our system Genesys employs a genetic programming (GP)-style optimization to discover new designs. Importantly, this relies on a factorized representation of code designs and an evolution tree described in § 4.1. Genesys then includes two core sets of agents: LLM-driven designers ( § 4.2) that select past designs from the evolution tree, propose unit-wise modifications to those designs based on background research, then implement the proposed designs and add them to the evolution tree. Verifiers ( § 4.3) select designs from the evolution tree and verify them through budget-aware pre-training. We consider each component in turn and provide various technical justifications for our design decisions that we further formalize in Appendix A.
this section cite: []

Section: Evolution Tree and Design Factorization
Figure 4: The evolution tree in Genesys (left), where nodes denote block designs and contain each design's executable code, a GAU tree representation of the code, design traces, and empirical performance metrics.
In order to apply GP-optimization, Genesys factorizes each block program B LM into a discrete tree representation called a generalized autoregressive unit (GAU) tree. For example, Figure 3 3 ⃝ shows a GAU tree for the transformer block, where each unit, or GAU, corresponds to a portion of the executable code implementation (see example in Fig. 16). This factorization forms the basis of an evolutionary tree (Figure 4) that stores new designs with these details and other artifacts. Importantly, this provides an interpretable representation of the discovery search space, where each block in the tree can be compared to another via their atomic units.
Based on this tree representation, standard GP operations (Koza, 1994) such as mutation (i.e., modifying a unit of a design) and crossover (i.e., merging the units of multiple designs) can be applied (see examples in Figure 14). In contrast to traditional GP, however, we use a relaxed form of GP that does not rely on a fixed inventory of mutation operators but instead uses an LLM to generate new code units, similar to Hemberg et al. (2024); Romera-Paredes et al. (2024). These units are implemented using the same GAB class construct described above, which allows for a consistent set of syntactic and semantic checks on the validity of each unit implementation.
Fail False ? Fail Input Output ? Fully implemented or not Save Checkpoint Proposer Reviewer Planner Coder Observer & SC Ideation and research Model implementation
Figure 5: How do we model the ideation and design stages in LM discovery? A high-level illustration of the agent subsystems, including a pair of proposer-reviewer LLM (consisting of a proposer and reviewer agent) that draft and score research ideas (left) and a hybrid network of planner-coder-observer agents (right) that produce code (consisting of a planner agent, a coder agent, and observer agent coupled with a symbolic checker (SC) tool that performs static and execution-based code analysis).
Design factorization: formal considerations As we discuss later ( § 4.2), by representing each code artifact A as a GAU tree, consisting of a sequence of GAUs A = I 1 , ...I N (each implemented as a GABBase module in Figure 3 2 ⃝), we can use such a factorization to not only perform GP-style optimization and efficient validity checking, but also devise efficient algorithms for block generation. While such representations are useful for understanding the discovery space, one natural question is: Does such a factorization adequately capture the full design space, or does it oversimplify the problem in some limiting way? As noted in Figure 3 using torchtyping- (Kidger, 2021) and Python-style type annotations, blocks and their units are naturally expressible as compositions of functions of type (X,Z) → (X,Z) (or more generally Σ → Σ). Through further formalization of the language underlying these structures, in A.2 we show formally that any composition of Σ → Σ functions guarantees a decomposition of the resulting code into the kinds of GAU tree representations we use.
this section cite: ['b32', 'b24', 'b47', 'b30']

Section: Model Designers
As shown in Figure 5, we break the design process into two stages, a proposal stage and an implementation stage. Further algorithmic details are provided in § B.2 with prompts in § F.
this section cite: []

Section: Proposal stage
The proposal stage starts by selecting a past design or pair of designs from the evolution tree (using the strategy in § 4.3) along with background references from the reference library, which queries the knowledge engine in LMADE. Based on this input, an LLM proposal agent comes up with a novel research idea involving a modification of the selected design(s) and writes a research proposal with high-level details of that idea and its implementation. Modifications are limited to either mutating a particular unit in the selected design, mixing units if multiple designs are selected (crossover), or designing a block from scratch (i.e., a special case of mutating the root Figure 6: How are new design ideas generated and vetted? An illustration of our proposer-reviewer agent architecture using real example design artifacts (right). First, a proposer agent uses parent designs (GAU tree, proposal, verification reports) from the evolutionary tree and selected references (code, text chunks, metadata) from the reference library to generate a research proposal, which is then adversarially reviewed and scored by a reviewer agent before proceeding to implementation.
Figure 7: The abstract code generation process in Genesys, where individual units in a code's unit tree (GAU tree) are modified and implemented piece-by-piece. Here shows the implementation of a mutation in 1 ⃝ involving the marked units (B, D) (all white units are protected). A new root F 2 ⃝ replaces B and consists of sub-units G and H, which each get implemented in turn (i.e., transformed into executable code) 3 ⃝-5 ⃝ until the new design is fully functional.
unit in a tree). Then, a separate LLM reviewer agent reviews and scores this proposal in a way analogous to an adversarial peer-reviewer and compares against past proposals to ensure novelty ( § B. 1.5). An illustration provided in Fig. 6. This loop continues until the proposal is accepted and the score assigned by the reviewer exceeds a certain threshold.
this section cite: []

Section: Implementation stage
The accepted research proposals are translated to executable designs B LM in this stage. Given that proposals involve unit-wise modifications to existing designs and their GAU trees, this allows for the step-by-step recursive generation shown in Fig. 7 (see also Alg. 3). This builds up a block program by incrementally constructing the GAU tree, which implicitly performs the factorization online. It maintains an Unimplemented list, initialized with the root of the editing subtree or a new tree. In each step: 1) A LLM planner agent selects an unimplemented GAU, and provides a plan for its implementation; 2) A LLM coder agent generates the Python code, potentially decomposing the GAU by declaring new children (via special statements), which will be added to Unimplemented with placeholder implementations; 3) Implementation is validated by a symbolic checker (verifying GAU/GAB compliance for the current unit and the entire tree) and a LLM observer that assesses code quality, proposal adherence, and novelty against prior/sibling implementations, then rates it (threshold: 3/5). If both checks pass, the GAU is accepted; otherwise, the tree state reverts for a retry. The implementation finishes when the Unimplemented is empty.
Unit-based code generation: algorithmic advantages One motivation for unit-based code generation is that a direct prompting approach often fails to produce useful and valid code (i.e., code that not only improves on past designs but also satisfies the constraints in Table 6). Such a direct approach, which is familiar to many code generation systems, is illustrated in Fig. 8(A) and involves presenting a model with input I and, on failure to produce a valid/useful output A, retries (e.g., with a modified I) until success. This difficulty can be understood formally: given the probability p of generating a valid/useful artifact A, the expected number of (i.i.d) calls to the model is E[calls] = 1 p , which will be prohibitive for most complex discovery problems with small p. In contrast to direct prompting, our approach (Fig. 8(B) generates unit-by-unit A = I 0 , ..., I N , where each successful unit I j is frozen in place before the next unit is generated. This operationalizes a Viterbi-style search (Viterbi, 1967), which we show from the first principles in § A.1 exponentially reduces the expected number of model calls. This explains the results in Table 3, and highlights the importance of a factorized search space.
this section cite: ['b58']

Section: Verifiers and Efficient Evolution
Distributed approach To allow for efficient exploration, Genesys runs the designer and verifier agents in parallel as in Romera-Paredes et al. (2024) (Fig. 1 Right), both of which communicate through the evolutionary tree. The evolutionary tree is initially populated with several state-of-the-art architecture designs, including the transformer/GPT (Biderman et al., 2023), Mamba2 (Dao & Gu, 2024), RetNet (Sun et al., 2023), RWKV6 (Peng et al., 2024), and TTT (Sun et al., 2024). Designer nodes continuously select parents (per the strategy below), query LMADE for references, and task the designer agent ( § 4.2) with generating new designs. Concurrently, verifier nodes select designs/scales and run verification in the LMADE Verification Engine whenever available. Further analysis of optimal worker ratios is provided in § E.
Figure 9: How do we perform efficient verification? Our Ladder-of-scales strategy involves starting small (training 1,000 14M parameter models on 0.7B tokens; bottom) and allocating progressively fewer trials for larger scales (training 5 350M parameter models on 50B tokens; top).
this section cite: ['b6', 'b13', 'b52', 'b53']

Section: Design selection
To effectively allocate resources, designers and verifiers select designs from the evolutionary tree by balancing exploitation (i.e., refining promising designs) and exploration (i.e., investigating diverse options). Designs in the evolutionary tree are assessed along two dimensions: fitness F (i.e., aggregate downstream task performance) and confidence (i.e., number of model scales where verification was performed). Designs are then categorized into four quadrants (see Fig-
this section cite: []

Section: Budget management
Verifying every design at each scale is prohibitively expensive. Inspired by scaling laws (Kaplan et al., 2020;Tay et al., 2022a) -which suggest that performance correlates across scales -and the methodology commonly employed for small LMs (Hu et al., 2024), we implement the Ladder of Scales strategy shown in Fig. 9 where several trials are performed at small parameter/token sizes, then scaled with decreasingly fewer trials. Formally, a total verification budget B m = {β 0 , ..., β N S } across N S + 1 scales (e.g., 14M-350M parameters) is structured pyramidally: more trials β i at smaller scales, with β i+1 ≈ sr i • β i (sr i < 1 is the target inter-scale selection ratio). Higher-scale budgets are released gradually to ensure fairness and to prevent early depletion. A dynamic allocatable budget B a = {α 0 , ..., α N S } is initialized with α 0 = 1 and α i>0 = 0. Budgets are replenished at the lowest scale upon use, and higher-scale budgets α i+1 are released when used lower-scale budgets β i exceed 1/sr i . The verifier node selects designs per the above strategy, verifying at the lowest unverified scale i with an available budget α i > 0 (see Alg. 5).
this section cite: ['b29', 'b25']

Section: Experiments
We empirically test the core components and overall effectiveness of our Genesys system. We aim to demonstrate the advantages of our approach and the potential of our system to perform advanced LM research. Our investigation is structured around three key research questions: RQ1 ( § 5.1): Does our GP approach lead to a better and more stable optimization process, one where each system component positively impacts the evolution process? RQ2 ( § 5.2): Does our unit-based code generation approach enhance design generation quality and efficiency compared to direct prompting methods, and how much do the individual model implementation agents (Fig. 5) affect performance? RQ3 ( § 5.3): Does our system ultimately discover architecture designs that are competitive with standard architectures? Detailed setups are provided in § C, with additional results in § D.  2024)). These variations allow us to address the following question: How much does literature understanding, verification feedback, and access to acquired knowledge contribute to successful discovery?
this section cite: []

Section: Evolutionary Experiments
We sample 300 designs for all configurations and 500 and 1000 designs for selected setups due to computational constraints. Evolutionary progress was evaluated using the following population fitness metrics over time (population size S P = 50, step size k s = 25): the End (∆) and peak (∆ max ) fitness improvement. Volatility (ν), or the standard deviation (std) of generational differences. We also measure the Sharpe Ratio (SR), or the risk-adjusted improvement computed as the mean of generational differences divided by their std and the Maximum Drawdown (M DD) that measures the maximal fitness decrement, which indicates stability. We note that these last metrics originate from financial economics Sharpe (1994); Gu et al. (2020b) with M DD and ν being the complement Pl. Coder Obs. SC UG Valid Attempts Costs LFC Full ✓ ✓ ✓ ✓ ✓ 92% 2.6 (±1.1) 15.0 (±18.5) 181 (±44) No UG ✓ ✓ ✓ ✓ 73% 3.0 (±1.7) 7.9 (±7.1) 75 (±29) No Pl. ✓ ✓ ✓ ✓ 91% 2.6 (±1.1) 16.0 (±20.8) 218 (±69) No Ob. ✓ ✓ ✓ ✓ 89% 2.6 (±1.1) 12.1 (±20.1) 211 (±67) No SC ✓ ✓ ✓ ✓ 30% 2.4 (±1.0) 2.9 (±4.7) 167 (±33) Direct ✓ 6% 1.1 (±0.2) 0.3 (±0.3) 49 (±15)
Table 3: Code quality vs. model designer variants w/wo the planner (Pl.), Coder, Observer (Ob.), Symbolic Checker (SC), and Unit-based Generation (UG), and a "Direct" prompting strategy. Valid reports the % of valid code generated, Attempts is the avg. number of generation attempts (at most 5 times), Costs is the average token cost, and LFC is the average Lines of Function-body Code. of SR. Recently, SR has been used in reinforcement learning to measure risk-return balance over time, which also suits our evolutionary search process and the exploration-exploitation trade-off.
The results are reported in Table 1. For the initial 300 designs, our full system performed the best by having the highest fitness improvements (∆ = 4.10%, ∆ max = 4.16%) and superior stability with the highest SR = 0.69 and lowest M DD = -0.38%. w/o Lit. reduced ∆ by 0.73% and SR to 0.567, underscoring the importance of literature guidance for stable progress. Base showed negligible improvement, while w/ Mem. boosted ∆ (to 2.81%) and SR (to 0.196), confirming the value of experience. For our extended runs (500 & 1000 Designs), we see similar advantages over w/o Exp. with doubled ∆ max and quadrupled SR, highlighting the role of experimental feedback as selection signals. Full continued to improve up to 1000 designs, reaching a peak fitness of 0.633 and showing signs of convergence (Fig. 10 Right). Interestingly, its evolutionary tree (Fig. 35) displays "hubness", which is analogous to the long-tail distribution of paper citations (Wu et al., 2009).
Full w/o Exp. w/o Lit. Base w/ Mem. Err 8.61% 27.31% 7.67% 21.09% 23.70%
Table 2: The error rate (%) during the design verification and evaluation stages under different system ablations.
Besides impacting fitness and stability, in Table 2, we show how the removal of components can result in increased errors during the verification process. For example, removing experiments led to code with a ∼19% higher error rate compared to our full system, showing how design evaluations can help avoid downstream errors later in discovery.
this section cite: ['b51', 'b63']

Section: Designer Agent Analysis
As mentioned earlier, a key bottleneck in our discovery system is generating valid code that satisfies the conditions in Table 6. To measure the effectiveness of our full system with its different design agents and symbolic checker (see again Fig. 5), we directly evaluated the implementation abilities of our designer agents using 100 proposals from our full evolution run as test cases. We systematically compared against variants of our system that removed the code planner (No. Pl), the observer agent (No Ob.), the unit-based generation strategy (No UG), and the semantic checker (No SC). Finally, we compared against the Direct prompting approach discussed in Figure 8. We measured the rate of successful implementations passing all checkers (Valid (%)), the average attempts (Attempts), and token costs (Costs) and Lines of Function-body Code (LFC) as a proxy for code complexity/quality.
Table 3 presents the results. Removing UG (No UG) significantly degraded 20.7% in the valid rate and 58.6% in LFC compared to the Full agent, highlighting the benefit of the structured, unit-by-unit approach for generating complex and valid code. Disabling the symbolic checker (No SC) drastic reduced the valid rate by 67.4%, which confirms its importance. Ablating the Planner (No Pl.) or Observer (No Ob.) results in minimal quantitative impact, yet both play a crucial qualitative role, such as guiding the implementation order and assessing novelty/quality. Moreover, the Full agent's code complexity (avg. LFC 181) was comparable to the human-written reference library (avg. LFC 220), suggesting more realistically complex designs. Simpler setups (Direct, No UG) often produce trivial outputs (e.g., basic ConvNets) with a significantly lower magnitude in LFC (~50).
Few vs. Many Samples: formal considerations One of the design principles underlying our agent system is that designers should produce few but deliberate and interpretable designs, much like in everyday research. This is in contrast to traditional GP approaches where a vast number of simple trials are routinely performed (e.g., see Real et al. (2020); Chen et al. (2024)). This raises the natural question: It is more effective to design more complex, and ultimately more expensive, agent systems that are more deliberate (e.g., our system with planners/observers/coders), or to rely on simpler, more cost-effective agent systems that can perform more trials?
In Appendix A.3 we provide a formal argument that attempts to justify the former and link this decision with our Viterbi-style search. Blimp Wnli RTE WG CoLA SST2 WSC IS Mrpc Avg. Random 69.75 43.66 52.71 48.78 50.00 49.08 49.82 50.03 31.62 49.49 GPT 92.70 60.56 62.80 52.17 53.24 54.13 56.76 55.31 68.38 61.78 Mamba2 83.22 63.38 63.88 51.22 55.94 56.58 57.12 53.85 67.89 61.45 RWKV7 88.76 61.97 60.21 49.80 54.25 55.32 54.57 57.00 68.38 61.14 RetNet 85.16 61.97 61.35 50.51 56.29 55.43 56.03 54.95 56.37 59.78 TTT 86.13 63.38 55.23 50.75 55.55 56.35 54.93 55.31 59.80 59.71 VQH 94.37 59.15 59.91 50.28 54.25 53.56 53.83 49.45 56.62 59.05 HMamba 83.74 64.79 61.35 53.59 54.69 57.04 56.40 54.58 59.31 60.61 Geogate 90.95 59.15 61.35 52.72 54.25 55.32 58.96 54.95 68.63 61.81 Hippovq 87.96 50.70 59.91 50.28 54.25 55.73 53.83 55.68 69.88 59.80 SRN 80.83 65.52 59.55 50.75 54.45 52.98 56.03 54.95 61.03 59.57
Table 4: How good are our discovered designs? A comparison of our seed models (top) vs. our discovered models (350M scale, 50B training tokens) based on end-task benchmark accuracy (%).
Bold/underline/italics denote the best/second/worst.
this section cite: ['b46', 'b9']

Section: Discovered Model Evaluation
To measure the overall performance of our new designs, we perform a standard zero-shot end-task evaluation that compares the top 5 designs discovered using the "Full" Genesys system against the five human seed designs shown in Figure 35 (GPT2, TTT, Mamba2, RWKV, and RetNet). We specifically evaluated these designs on the scales of 125M and 350M parameters, trained on 25B and 50B tokens, respectively. Following standard protocols Groeneveld et al. (2024), tasks were selected based on their informativeness on smaller scales (see § E.3.1 and Table 16 for details).
How good are the discovered designs? Tables 4 and 14 ( § D.1) show the results of our evaluation.
Our discovered designs outperformed/matched baselines on 7/9 (125M) and 6/9 (350M) benchmarks, with superior averages. Although no single model dominated all tasks, consistent with many other studies on small LM development (Fourrier et al., 2024)), the discovered designs consistently performed competitively with the state-of-the-art human baselines. This shows the feasibility of using LLMs to automate human-level LM research, at least at the functional level. We see scaling experiments, as well as analysis on the intelligibility of the AI designs, as promising future work.
this section cite: ['b20', 'b17']

Section: Discussion, Limitations & Conclusion
We introduce Genesys, an autonomous system for discovering novel LM designs, featuring a novel unit-based design agent and cost-effective distributed evolution. We also present LMADE, a resource environment to support further research in this field. Current limitations include integrating efficiencyfocused innovations, such as FlashAttention (Dao, 2024), hindered by complex hardware-specific evaluations, and the constraints of billion-parameter-level discovery due to limited computational resources. Future work will aim to enhance the agent's learning from feedback, possibly via reinforcement learning, as well as to develop a more adaptive design selection strategy. Our largescale experiments yielded 1,062 novel LM architectures (14M-350M parameters), fully verified with pretraining. This is, to our knowledge, the largest automated LM discovery experiment. Genesys produced highly competitive designs; some outperformed human baselines such as the GPT and Mamba2 models in common downstream tasks. These results show the feasibility and lay the groundwork for autonomous evolutionary systems in scientifically complex and costly domains.
NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We present our experiments in § 5 and § D, theoretical results in § A.1, we also show extensive analysis in § E. Guidelines: • The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: ['b12']

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Discussed in § 6. Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: We provide proofs in § A.1.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We provide full implementation details in § B and full experiment details in § C. Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We provide full experiment details in § C, and we will release all our code and discovered models upon acceptance.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental setting/details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We provide full implementation details in § B and full experiment details in § C. Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [No] Justification: Error bars are not reported because it would be too computationally expensive for the evolution experiments. But we report the std for agent experiment (Table 3).
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: We provide details in § C.1.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: We have reviewed and conducted the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: We have an impact statement section following the main text.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA]
Justification: The paper does not pose such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: We cited all assets or provided the URLs.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [Yes] Justification: We introduced LMADE in § 3 and provide full details in § B.1.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets ( if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [Yes] Justification: Our paper uses LLM agent to sample LM designs. Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described. E.1.1 Analysis of Design Sessions . . . . . . . . . . . . . . . . . . . . . . . . . E.1.2 Analysis of Evolutionary Tree . . . . . . . . . . . . . . . . . . . . . . . . E.2 Analysis of Agents . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . E.2.1 Analysis of Foundation Models . . . . . . . . . . . . . . . . . . . . . . . E.2.2 Analysis of Implementation Errors . . . . . . . . . . . . . . . . . . . . . . E.3 Analysis of Discovered Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . E.3.1 Analysis of Verification Process . . . . . . . . . . . . . . . . . . . . . . . E.3.2 Analysis of Model Units . . . . . . . . . . . . . . . . . . . . . . . . . . . E.3.3 Unit-Performance Relation . . . . . . . . . . . . . . . . . . . . . . . . . . E.4 Analysis of System and Performance . . . . . . . . . . . . . . . . . . . . . . . . . E.4.1 Training Time Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . E.4.2 Optimal Pipeline Throughput and the V-D Ratio . . . . . . . . . . . . . . F Prompts F.1 Proposer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.1.1 System and GP Background . . . . . . . . . . . . . . . . . . . . . . . . . F.1.2 Search and Refinement . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.2 Reviewer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.2.1 System Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.2.2 Final Review . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.3 Planner . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.3.1 System Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.3.2 Plan and Unit Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.4 Coder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.4.1 System Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.4.2 Implementation and Debugging . . . . . . . . . . . . . . . . . . . . . . . F.5 Observer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.5.1 System Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.5.2 Observation Feedback . .
. . . . . . . . . . . . . . . . . . . . . . . . . . G Qualitative Examples G.1 Example GAU Trees . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . G.1.1 Five Evaluated Designs . . . . . . . . . . . . . . . . . . . . . . . . . . . . G.1.2 Complicated Designs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . G.2 Example Design Artifact . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
this section cite: []

Section: A Formal Analysis of System Design and Proofs
In this section, we provide further formal analysis of the technical and algorithmic points made in § 4. In § A.1-A.1.2 we discuss the properties of our unit-by-unit Viterbi generation strategy and its advantages over a direct prompting approach. In § A.2 we formalize the structure of block programs B LM and use this structure to justify our GAB tree factorization. Finally, in § A.3.2 we justify our decision to optimize for few-vs-many samples in our GP approach and link this with properties of our Viterbi search strategy.
this section cite: []

Section: A.1 Viterbi-style Search (VS) Proofs
To compare different prompting strategies, we analyze the expected number of attempts (and then token costs) of a single-shot direct-prompting (direct) approach versus a Viterbi-style Search (VS) or our unit-by-unit generation approach. The argument is based on straightforward properties of the geometric distribution, but yields an exponential advantage (Prop. 1) for VS when a design artifact must satisfy multiple constraints.
this section cite: []

Section: A.1.1 Direct vs. Viterbi Approach: Basic Analysis
Setup Let A be the set of possible final artifacts (e.g. fully implemented LM architectures). We consider:
• A direct approach that tries to generate an artifact A in one shot. If the artifact is invalid (e.g., doesn't pass the checker in Table 6), we discard and try again from scratch.
• A VS approach that factorizes the generation into N sequential sub-decisions, each retried upon failure only locally (i.e. we "checkpoint" partial successes).
The following result relates to the expected number of model calls for the direct approach.
Lemma 1 (Single-Shot Expected Calls). Suppose the probability of success (i.e., generating a valid output that satisfies some target constraints) in one single-shot generation is p valid ∈ (0, 1).
Then the expected number of calls until success (denoted as E[calls direct ]) is E[calls direct ] = 1 p valid , assuming each call is an i.i.d. Bernoulli(p valid ) trial.
Proof. This follows directly from the geometric distribution: the probability we succeed on the k-th attempt is (1 -p valid ) k-1 p valid , and the expected number of attempts is 1/p valid .
In many scenarios, success requires N sub-components to be correct simultaneously (e.g., N generated code units all being correct), each with probability p k . Then
p valid ≈ N k=1 p k =⇒ E[calls direct ] ≈ 1 N k=1 p k .
this section cite: []

Section: Viterbi-style Unit-based Factorization
In the VS approach, we imagine the creation of an artifact as N steps:
I 0 → I 1 → • • • → I N = A,
where each step (I k-1 → I k ) succeeds with probability p k . Failures at step k do not discard previously completed steps; we simply revert to I k-1 and retry. For this approach, the following holds:
Lemma 2 (Expected Calls: VS). If step k has a probability p k of success on each attempt, the expected total number of calls to complete all N steps is
E[calls VS ] = N k=1 1 p k .
this section cite: []

Section: Proof.
Step k follows a geometric distribution with success probability p k . Hence, its expected trials are 1/p k . Summing over k = 1, . . . , N yields N k=1 1/p k .
From these two facts, the following follows regarding how the VS approach requires exponentially fewer model calls over the direct approach.
Proposition 1 (VS vs. Direct: Exponential Gain). If p valid ≈ N k=1 p k , then E[calls direct ] ≈ 1 N k=1 p k , E[calls VS ] = N k=1 1 p k .
In typical cases where
N k=1 p k ≪ p j for each j, we have N k=1 1/p k ≪ 1/ N k=1 p k , indicating a potential exponential improvement for VS.
Then the following corollary follows straightforwardly.
Corollary 2 (Identical Steps Case). If p k = p for all k, then E[calls direct ] = 1 p N , E[calls VS ] = N p .
As N grows, N/p 1/p N = N p N -1 → 0 (exponentially), showing that the advantage of VS grows dramatically with larger N .
The exponential gain of VS also explains why high-quality samples outweigh vast low-quality onesit may take exponentially more samples to reach the same optimal point in a VS sample.
this section cite: []

Section: A.1.2 Refined Analysis: Token Costs with Growing History
Given that each model call incurs token costs, an exponential improvement in the number of calls or steps k means that the token costs may reduce exponentially. In this section, we quantify such token costs in terms of the prompting and history tokens that need to be processed during each try. Let:
• H k : the number of "history" input tokens at step k.
• δ k : any additional instructions or new tokens in step k.
• O k : the number of output tokens generated by step k.
• c i , c o : cost coefficients for input and output tokens, respectively.
this section cite: []

Section: Then the cost of a single attempt of Step k is:
Cost
k = c i (H k + δ k ) + c o O k .
Under geometric retries, the expected attempts at step k are 1/p k and the following holds.
Lemma 3 (Expected Token Cost in VS). The expected total cost to complete all N steps in VS is
E[Cost VS ] = N k=1 1 p k c i (H k + δ k ) + c o O k .
Proof. At each step k, we expect 1/p k attempts. Each attempt incurs Cost k . Summing over k completes the proof.
Comparison to Single-Shot In a single-shot approach, the probability of success is N k=1 p k , so the expected number of attempts is 1/ p k . Each try regenerates the entire artifact. Let its cost be Cost
full = c i (. . . ) + c o (. . . ). Then E[Cost direct ] = 1 N k=1 p k × Cost full .
Thus, even accounting for partial history {H k } in VS, one can still have:
N k=1 Cost k p k ≪ Cost full N k=1 p k
, especially when N k=1 p k is very small. This confirms that the exponential improvement result extends to token-cost models, not just the raw count of attempts.
this section cite: []

Section: Conclusion (VS)
Our Viterbi-style search can yield an exponential reduction in expected attempts (and potentially in token cost) compared to single-shot approaches, particularly when many subdecisions need to be correct simultaneously. This underpins the efficiency of Genesys's stepwise Planner-Coder-Observer pipeline, where partial successes are preserved.
this section cite: []

Section: References
Ref_id:b0 Title: Phi-3 technical report: A highly capable language model locally on your phone Year: (2024)
Ref_id:b1 Title: Smollm -blazingly fast and remarkably powerful. Hugging Face Blogs Year: (2024)
Ref_id:b2 Title: Smollm2: When smol goes big-data-centric training of a small language model Year: (2025)
Ref_id:b3 Title: Extended long short-term memory Year: (2024)
Ref_id:b4 Title: Learning to memorize at test time Year: (2024)
Ref_id:b5 Title: Hierarchical state space models for continuous sequence-to-sequence modeling Year: (2024)
Ref_id:b6 Title: Pythia: A suite for analyzing large language models across training and scaling Year: (2023)
Ref_id:b7 Title: Language models are few-shot learners Year: (2020)
Ref_id:b8 Title: Evoprompting: Language models for code-level neural architecture search Year: (2023)
Ref_id:b9 Title: Symbolic discovery of optimization algorithms Year: (2024)
Ref_id:b10 Title: Accurate proteome-wide missense variant effect prediction with alphamissense Year: (2023)
Ref_id:b11 Title: Neural architecture search for transformers: A survey Year: (2022)
Ref_id:b12 Title: Flashattention-2: Faster attention with better parallelism and work partitioning Year: (2024)
Ref_id:b13 Title: Transformers are ssms: Generalized models and efficient algorithms through structured state space duality Year: (2024)
Ref_id:b14 Title: Tinystories: How small can language models be and still speak coherent english Year: (2023)
Ref_id:b15 Title: Neural architecture search: A survey Year: (2019)
Ref_id:b16 Title: Were rnns all we needed? arXiv preprint Year: (2024)
Ref_id:b17 Title: Open llm leaderboard v2 Year: (2024)
Ref_id:b18 Title: The pile: An 800gb dataset of diverse text for language modeling Year: (2020)
Ref_id:b19 Title: A framework for few-shot language model evaluation Year: ()
Ref_id:b20 Title: Accelerating the Science of Language Models. Proceedings of ACL Year: (2024)
Ref_id:b21 Title: Hippo: Recurrent memory with optimal polynomial projections Year: (2020)
Ref_id:b22 Title: Efficiently modeling long sequences with structured state spaces Year: ()
Ref_id:b23 Title: Empirical asset pricing via machine learning Year: (2020)
Ref_id:b24 Title: Evolving code with a large language model Year: (2024)
Ref_id:b25 Title: Unveiling the potential of small language models with scalable training strategies Year: (2024)
Ref_id:b26 Title: Codescientist: End-to-end semi-automated scientific discovery with code-based experimentation Year: (2025)
Ref_id:b27 Title: Codescientist: End-to-end semi-automated scientific discovery with code-based experimentation Year: (2025)
Ref_id:b28 Title: Highly accurate protein structure prediction with alphafold Year: (2021)
Ref_id:b29 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b30 Title: Type annotations and runtime type checking of tensor shapes (and dtypes Year: (2021)
Ref_id:b31 Title: The semantic scholar open data platform Year: (2023)
Ref_id:b32 Title: Genetic programming as a means for programming computers by natural selection Year: (1994)
Ref_id:b33 Title: Scientific discovery: Computational explorations of the creative processes Year: (1987)
Ref_id:b34 Title: Evolving deeper llm thinking Year: (2025)
Ref_id:b35 Title: Generating science from ai-powered automated falsification Year: (2024)
Ref_id:b36 Title: Generating science from ai-powered automated falsification Year: (2024)
Ref_id:b37 Title: The ai scientist: Towards fully automated open-ended scientific discovery Year: (2024)
Ref_id:b38 Title: Small language models: Survey, measurements, and insights Year: (2024)
Ref_id:b39 Title: Scaling deep learning for materials discovery Year: (2023)
Ref_id:b40 Title: Global prediction of extreme floods in ungauged watersheds Year: (2024)
Ref_id:b41 Title: A generative artificial intelligence framework based on a molecular diffusion model for the design of metal-organic frameworks for carbon capture Year: (2024)
Ref_id:b42 Title: An open dataset of high-quality mathematical web text Year: (2024)
Ref_id:b43 Title: The fineweb datasets: Decanting the web for the finest text data at scale Year: (2024)
Ref_id:b44 Title: Eagle and finch: RWKV with matrix-valued states and dynamic recurrence Year: (2024)
Ref_id:b45 Title: Hierarchically gated recurrent neural network for sequence modeling Year: (2023)
Ref_id:b46 Title: Automl-zero: Evolving machine learning algorithms from scratch Year: (2020)
Ref_id:b47 Title: Mathematical discoveries from program search with large language models Year: (2024)
Ref_id:b48 Title: Analysing mathematical reasoning abilities of neural models Year: (2019)
Ref_id:b49 Title: Agent laboratory: Using llm agents as research assistants Year: (2025)
Ref_id:b50 Title: Agent laboratory: Using llm agents as research assistants Year: (2025)
Ref_id:b51 Title: The sharpe ratio Year: (1994)
Ref_id:b52 Title: Retentive network: A successor to transformer for large language models Year: (2023)
Ref_id:b53 Title: Rnns with expressive hidden states Year: (2024)
Ref_id:b54 Title: Rethinking optimization and architecture for tiny language models Year: (2024)
Ref_id:b55 Title: Scaling laws vs model architectures: How does inductive bias influence scaling? arXiv preprint Year: (2022)
Ref_id:b56 Title: Efficient transformers: A survey Year: (2022)
Ref_id:b57 Title: Attention is all you need Year: (2017)
Ref_id:b58 Title: Error bounds for convolutional codes and an asymptotically optimum decoding algorithm Year: (1967)
Ref_id:b59 Title: Efficient large language models: A survey Year: (2024)
Ref_id:b60 Title: Scientific discovery in the age of artificial intelligence Year: (2023)
Ref_id:b61 Title: Neural architecture search: Insights from 1000 papers Year: (2023)
Ref_id:b62 Title: Discovery of a structural class of antibiotics with explainable deep learning Year: (2024)
Ref_id:b63 Title: Research and the long tail: A large-scale citation analysis Year: (2009)
Ref_id:b64 Title: Neural language of thought models Year: (2024)
Ref_id:b65 Title: Improving transformers with dynamically composable multi-head attention Year: (2024)
Ref_id:b66 Title: Efficient streaming language models with attention sinks Year: (2023)
Ref_id:b67 Title: The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search Year: (2025)
Ref_id:b68 Title:  Year: (2024)
Ref_id:b69 Title: *Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence* Year: (2024)
Ref_id:b70 Title: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* Year: (2023)
Ref_id:b71 Title: *Pyramid Vector Quantization for Efficient Encoding and Decoding* Year: (2018)
Ref_id:b72 Title: *Channel-Relaxed Vector Quantization for Memory Compression in Language Models* Year: (2024)
Ref_id:b73 Title: *DenseMamba: State Space Models with Dense Hidden Connection for Efficient Large Language Models* Year: (2024)
