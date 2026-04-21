Title: Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation
Abstract: Autoregressive Large Language Models (AR-LLMs) frequently exhibit implicit parallelism in sequential generation. Inspired by this, we introduce Multiverse, a new generative model enabling natively parallel generation. Multiverse internalizes a MapReduce paradigm, generating automatically through three stages: (i) a Map stage for adaptive task decomposition, (ii) a Process stage for parallel subtask execution, and (iii) a Reduce stage for lossless result synthesis. Next, we build a real-world Multiverse reasoning model with co-design of data, algorithm, and system, enabling rapid and seamless transfer from frontier AR-LLMs. For data creation, we develop Multiverse Curator, an automated LLM-assisted pipeline that transforms sequential reasoning chains into structured training data, avoiding costly human annotations. Algorithmically, we design Multiverse Attention to separate parallel reasoning steps while keeping compatibility with causal attention for efficient training. Systematically, we implement Multiverse Engine to support parallel inference. It features a dedicated interpreter that dynamically switches between sequential and parallel generation, triggered directly by the model. After a 3-hour fine-tuning with 1K examples, our Multiverse-32B stands as the only opensourced non-AR model achieving performance on par with leading AR-LLMs of the same scale, evidenced by AIME24 & 25 scores of 54% and 46%, respectively. Moreover, our budget control experiments show that Multiverse-32B exhibits superior scaling, outperforming AR-LLMs by 1.87% on average using the same context length. Such scaling further leads to practical efficiency gain, achieving up to 2× speedup across varying batch sizes. We have open-sourced the entire Multiverse ecosystem, including data, model weights, serving system, supporting tools, as well as data curation prompts and detailed training and evaluation recipes.

Section: Introduction
"In an infinite multiverse, everything that can happen does happen-somewhere."
Test-time scaling has advanced Large Language Models (LLMs) by increasing the generation length [20,16] and depth [12], closely reflecting human cognition. However, empowered by modern hardware like GPUs, ideal LLMs can surpass humans by scaling a third dimension: width, which allows parallel task-solving. Realizing this potential requires LLMs to "smartly" parallelize and merge their generation, following the classic MapReduce paradigm [8]: splitting into subtasks, processing them independently in parallel, and merging their results. Such philosophy has a long history in computer science [26,1] while driving fundamental progress in other fields including manufacturing [19], agriculture [28], and finance [7]. This shift from sequential to parallel task-solving unlocks economies of scale: reducing the time per unit and keeping near-constant overall latency as task complexity grows, thereby offering a promising path towards artificial superintelligence (ASI).
this section cite: ['b19', 'b15', 'b11', 'b7', 'b25', 'b0', 'b18', 'b27', 'b6']

Section: Autoregressive Model Diffusion Model
Analysis: prove an equation
Subtask 1: simplify LHS Split Conclusion: LHS = RHS Merge Subtask 2: simplify RHS
this section cite: []

Section: Multiverse Model
Generation following sequential logic Reduce phase losslessly merge results
this section cite: []

Section: Sequential blocks
Parallel blocks Map phase: adaptive task split Reduce phase: lossless result merge Despite this potential, current LLMs are limited by the inherently sequential nature of autoregressive (AR) generation. While non-AR architectures, such as diffusion models [37,49] and consistency models [22], along with their hybrid semi-AR variants [2,29], natively enable parallel generation, they incur substantial computational waste. Their rigid, brute-force parallelism ignores logical dependencies, partly due to a lack of real-world training data to supervise when and how parallel generation should occur. Another stream of research [48,6,45,32] leverages external tools to parallelize or merge tasks heuristically, leading to the loss of internal states, like the intermediate reasoning steps, during communication with external modules. Although our concurrent work [21,36] allows internal communication, they introduce inconsistencies between training and inference, limiting their effectiveness to short sequences with shallow parallelism. These challenges raise a question: How to design a modeling framework for LLMs that can (i) adaptively split and merge tasks, (ii) losslessly preserve internal states, and (iii) generally apply to diverse parallelism patterns?
Due to the dominance of AR-LLMs, we start to answer it by revealing numerous intrinsic parallelism in their sequential outputs. Specifically, we analyze the long Chain-of-Thought (CoT) trajectories from the s1K-1.1 dataset [27]. Among them, over 98% exhibit parallelizable branches, despite being trained only for sequential generation. These branches, as shown in Figure 2, fall into collective and selective ones that appear frequently within individual CoT trajectories, either consecutively or recursively, covering a wide range of scenarios. However, our prompting and probing tests verify that AR-LLMs cannot actively enforce or discern such parallelism. These findings motivate the design of a new modeling framework that can be bootstrapped directly from pre-trained AR-LLMs, which further requires us to address three practical limitations: (i) Data: Real-world CoT trajectories lack explicit parallel structure. (ii) Algorithm: Transformers with causal attention are limited to sequential generation. (iii) System: Inference engines for AR-LLMs cannot support practical parallel generation.
To achieve these, we introduce Multiverse, a generative modeling framework built on the MapReduce paradigm that dynamically adjusts its parallelism during generation. It internalizes a three-stage pipeline: a sequential Map stage performs adaptive task decomposition; a parallel Process stage allows independent subtask execution; and a sequential Reduce stage ensures lossless result synthesis. Moreover, the pipeline can invoke itself recursively, enabling optimal time complexity with unlimited resources. We theoretically prove this optimality on a synthetic NP-hard SAT problem, demonstrating that Multiverse is the only framework that achieves a linear-time solution. Based on this, we co-design our data, algorithm, and system, providing a general solution to building a real-world Multiverse model for complex reasoning tasks, offering a smooth and rapid transition from pre-trained AR-LLMs.
this section cite: ['b36', 'b48', 'b21', 'b1', 'b28', 'b47', 'b5', 'b44', 'b31', 'b20', 'b35', 'b26']

Section: Data Curation.
In Section 5.1, we develop Multiverse Curator, an automated LLM-assisted pipeline that transforms sequential reasoning chains into parallel structures via five steps: (i) parsing the sequential chain into a summary tree; (ii) identifying parallelizable nodes within the summary tree;
(iii) reformatting the summary into a parallel generation structure; (iv) refilling original reasoning steps into this structure; and (v) adding Map & Reduce stages while rewriting Process stage. Moreover, content and grammar checks are performed to flag low-quality data for regeneration, avoiding costly manual filtration and annotation. In practice, this process results in Multiverse-1K, a dataset of 1,000 high-quality structured training samples for advancing LLM reasoning.
Algorithm Design. In Section 5.2, we design Multiverse Attention to enable parallel generation while maintaining training efficiency. This is achieved by modifying attention masks and position indices to strictly separate independent reasoning branches in attention calculation. Due to its which can be trained in parallel, similar to causal attention. This design also excels in data efficiency: since these changes are minor, pre-trained AR models can be rapidly transferred from causal attention to Multiverse attention using only a few thousand examples.
System Implementation. In Section 5.3, we implement Multiverse Engine featuring a specialized interpreter to support MapReduce in execution. By interpreting control tags generated by the Multiverse model itself, our engine can dynamically switch between sequential and parallel generation with near-zero overhead, yielding a flexible workflow. This process includes two stages: (i) Sequential → Parallel: mapping subtasks to separate branches for parallel execution with prefix sharing, and (ii) Parallel → Sequential: reducing Key-Value (KV) states from all branches back into one sequence.
The integration of these modules enables highly efficient training and inference of Multiverse models. Specifically, we develop Multiverse-32B by supervised fine-tuning (SFT) Qwen-2.5-32B-Instruct with only 1K examples, which takes only 3 hours. Empirically, Multiverse-32B achieves significant performance improvement, outperforming the base model by 23.6%, with AIME24 and AIME25 scores of 53.8% and 45.8%, respectively. These results are comparable to AR-LLMs, confirming that Multiverse does not compromise model performance. Furthermore, Multiverse-32B exhibits more efficient test-time scaling, yielding an average improvement of 1.87% within fixed latency constraints. This efficiency stems from its parallel generation capabilities, leading to up to 2× wall-clock speedup per generated token while keeping effective scaling across variable batch sizes range from 1 to 128.
this section cite: []

Section: Related Work
Test-time Scaling. Prior work has shown that optimizing AR-LLMs to generate longer outputs improves their reasoning abilities. This is evident in frontier reasoning models built with reinforcement learning (RL) [31,9,30,15,42], and also validated through supervised fine-tuning (SFT) on smaller models with a few distilled examples [27,47]. However, this length scaling greatly increases latency due to the sequential nature of AR generation. Other methods like depth scaling [12,49] suffer from the same issue, while width scaling [5,32] requires external information to split or merge generations.
this section cite: ['b30', 'b8', 'b29', 'b14', 'b41', 'b26', 'b46', 'b11', 'b48', 'b4', 'b31']

Section: Internal Parallel Generation.
Recent work has increasingly explored other models to replace the commonly used AR models, thereby enabling parallel generation. Among them, discrete diffusion models [37,38,3,23,40], including masked and absorbed variants, are gaining growing attention. To narrow their gap with AR models, efforts have been made on methods like hybrid AR-diffusion generation [2,10] and training/test-time scaling [29,49,46]. However, [11] has theoretically shown that these approaches cannot reduce the number of sequential generating or sampling steps, as they brute-force parallelize token generation without adhering to inherent relations. Similarly, other work explores continuous diffusion models [4] and consistency models [22]. Among these open-sourced, non-AR models, a common issue is their current inability to scale to complex reasoning tasks, such as AIME [24]. While our concurrent work [21,36] begins to explore the use of customized attention masks for parallel generation, their design are not general or adaptive, limiting their effectiveness to shallow, non-nested parallelism. In contrast, Multiverse offers a more efficient and scalable approach to enable internal parallel generation, which is generally applicable to diverse parallelism patterns.
this section cite: ['b36', 'b37', 'b2', 'b22', 'b39', 'b1', 'b9', 'b28', 'b48', 'b45', 'b10', 'b3', 'b21', 'b23', 'b20', 'b35']

Section: External Parallel Generation.
In another line of research, several approaches leverage external tools or models to enable parallel generation [45,32,41,5,48]. However, these methods generally leverage heuristic rules and external tools to parallelize or merge their generation. For instance, Best-of-N [5] and self-consistency [41] use a brute-force approach by parallelizing generation at the beginning of generation. Other methods like Monte Carlo tree search (MCTS) [48] and Tree of Thoughts (ToT) [45] offer more fine-grained parallelism, yet they are still fundamentally guided by heuristics and depend on an external verifier. While recent work [32] enables more adaptive parallel generation, it suffers from significant information loss when parallelizing and merging its generation, as it requires inter-model communication when switching between sequential and parallel generation, during which short text summaries rather than complete KV states can be shared between models.
this section cite: ['b44', 'b31', 'b40', 'b4', 'b47', 'b4', 'b40', 'b47', 'b44', 'b31']

Section: Long CoT Generation: Sequential or Parallel in Logic?
In this section, we present several key observations of intrinsic parallelism within AR-LLMs. First, in Section 3.1, we examine the long CoT trajectories generated by such models, verifying the common existence of intrinsic parallelism. Subsequently, Section 3.2 details probing and prompting tests, showing that AR-LLMs are unable to explicitly enforce or discern this parallelism during generation.
this section cite: []

Section: LLMs can Implicitly Generate Parallelizable Branches.
We start by analyzing the long CoT trajectories of current AR-LLMs using the s1K-1.1 dataset [27], including Deepseek R1 [16] and Gemini 2.0 Flash Thinking [13], aiming to answer the following question: Does the logic of sequentially generated text truly depend on all content that precedes it?  Table 1: Parallelizable branches commonly exist in long CoT trajectories generated by AR-LLMs. Perexample existence ratio (R%) and frequency (F) of different types are measured in the format R|F. Collective Branch Selective Branch Total Case Study Subtask Execution Other Path Exploration Other Deepseek R1 28.0 | 2.00 47.3 | 3.38 3.2 | 0.23 19.4 | 1.39 1.1 | 0.07 99.0 | 7.07 Gemini 2.0 Flash 39.4 | 2.82 45.0 | 3.22 2.9 | 0.21 8.3 | 0.60 1.4 | 0.09 97.0 | 6.94
Surprisingly, we observe distinct cases, termed "parallelizable branches", where multiple generation steps can be executed concurrently, rather than strictly awaiting the completion of prior text generation. These cases highlight the inherent parallelism in AR-LLMs. Figure 2 exemplifies these branches, which are categorized as collective and selective, that can be combined consecutively or recursively.
Collective Branches. This involves independent steps whose outputs are subsequently merged into the final result. Such scenarios often result from splitting a complex task into subtasks that can be processed concurrently. Examples include studying different cases and analyzing individual events.
Selective Branches. This refers to cases where numerous paths are considered, but not all contribute to the final output. Examples include exploring diverse solutions or examining competing hypotheses.
Table 1 further details the occurrence ratios and frequencies for different types in s1K-1.1, where over 98% of examples involve parallelizable branches. Among them, collective branches (e.g., case study and subtask execution) are dominant, accounting for 79%, with selective branches like path exploration comprising the other 19%. Moreover, these branches appear frequently, averaging 7 times per example, which are arranged into combinations of consecutive and recursive parallel structures.
this section cite: ['b26', 'b15', 'b12']

Section: LLMs cannot Explicitly Structure Parallelizable Branches.
Next, we examine the behavior of AR-LLMs from both token and hidden spaces, revealing their current inability to explicitly generate or identify such parallelizable branches based on our structures.
0 1 2 0 50 100 0 1 2 3 4 5 6 7 8 9 10 11 12 13 1415+ 0 10 20 0 1 2 # Explicit 0 50 100 0 1 2 3 4 5 6 7 8 9 10 11 12 13 1415+ # Implicit 0 10 20 Deepseek R1 Gemini 2.5 Pro Percentage (%) (a) Prompting Test: Comparing Explicit and Implicit Structure Counts. QWQ-32B R1-32B R1-70B 30 35 40 45 50 55 60 Accuracy (%) 48% 57% 53% (b) Probing Test: Classifier Acc. Prompting Test. We first prompt Deepseek R1 and Gemini 2.5 Pro using the same questions, with a detailed description of parallel structures that includes both types and their combinations. As shown in Figure 3a, a significant 90% disparity emerges between explicit occurrence and implicit existence of these structures, highlighting that current LLMs are unable to perform explicit parallel generation.
Probing Test. Next, we delve deeper into the hidden space of AR-LLMs, conducting probing test to confirm whether they can discern intrinsic parallelism. Specifically, we label the tokens before each parallel block as positive examples and treat all other tokens as negative. Final-layer representations of these tokens are extracted using DeepSeek-R1-Distill-Qwen-32B & 70B [9] and QWQ-32B [34].
A two-layer MLP classifier is trained to predict whether a token initiates a parallel structure. However, the classifier's low test accuracy in Figure 3b suggests that AR-LLMs do not truly understand such parallelism but generate these structures unconsciously based on patterns from the pre-training corpus.
4 Designing Multiverse for Natively Parallel Generative Modeling.
With all findings in Section 3, we present Multiverse, a new generative modeling framework built on the MapReduce paradigm that explicitly decides how to parallelize and merge the generation process.
this section cite: ['b8', 'b33']

Section: Preliminaries.
Language Modeling aims to learn the joint probability distribution over sequences of words or tokens. Given a finite vocabulary V of tokens, and a sequence of L tokens denoted by x 1:L = (x 1 , x 2 , . . . , x L ), a language model estimates the joint probability P (x 1 , x 2 , . . . , x L ) of the sequence.
Autoregressive Modeling involves representing a sequence from left to right, where the probability of each token x t is conditioned on all previously tokens in the sequence (i.e., x 1:t-1 ). Consequently, the joint probability of the entire sequence x 1:L is factorized as a product of conditional probabilities:
P (x 1:L |θ AR ) = P (x 1 , x 2 , . . . , x L |θ AR ) = L t=1 P (x t |x 1 , . . . , x t-1 ; θ AR )
where θ AR denotes model parameters. AR models offer high accuracy but exhibit poor parallelism.
this section cite: []

Section: Multiverse Modeling.
Our modeling framework, Multiverse, advances beyond AR by eliminating redundant sequential dependencies between independent sequences, enabling adaptive and lossless parallel generative modeling. Therefore, our Multiverse must "smartly" decide when to start and end parallel generation.
To realize this, we adopt a MapReduce structure internalizing three stages, as illustrated in Figure 4.
x2 x1,1 x1,2 x1,3 x1,4 x1,5 x2,1 x2,2 x2,3 x2,4 x2,5 x3 x4 x 1 Process Stage (execution in parallel) Map Stage Reduce Stage x 1 x 2 x s x 1,s x 1,1 x 1,2 x 1,3 x 1,4 x 1,5 x 1,e x 2,s x 2,1 x 2,2 x 2,3 x 2,4 x 2,5 x 2,e x e x 3 x 4 Map Stage. The pipeline begins by generating a task decomposition plan, denoted as x s . Each subtask in x s is then mapped to an independent prefix sequence, modeled as P (x 1,s |x s ) and P (x 2,s |x s ). Process Stage. Next, it performs parallel modeling for each branch independently, conditioned on its own prefix. This enables the concurrent generation of diverse branches, like: P (x 1,1:6 |x [1:3,s] , x 1,s ) and P (x 2,1:6 |x 2,s , x [1:3,s] ). Each branch should end if a specific suffix (i.e, x 1,e or x 2,e ) is generated. Reduce Stage. After completing all branches, Multiverse shift back to sequential generation that conditioned on all preceding tokens, which is modeled as P (x e,[3:4] |x 1,[s,1:6,e] , x 2,[s,1:6,e] , x [1:3,s]
).
The integration of this three-stage pipeline enables Multiverse to: (i) adaptively decide when and how to parallelize generation during the Map stage; and (ii) retain informational completeness by ensuring every branch remains fully accessible throughout the Reduce stage and beyond. Notably, Multiverse naturally generalizes to both recursive and consecutive compositions of multiple MapReduce blocks.
this section cite: []

Section: Structured Generation Flow.
<Conclusion> The two inequalities are each responsible for an interval: (0°, alpha] for the first inequality and [beta, 90°) for the second. Note that [sqrt5 -1]/2 multiplied by [1 + sqrt5]/2 = (5 -1)/4 = 4/4=1. This is useful because it shows that alpha and beta are complementary angles in the sense that tan alpha * tan beta=1, so tan beta=1/tan alpha, which implies that beta=90°-alpha. </Conclusion> </Parallel>
… Let's think in parallel! <Parallel> <Goal> Consider two cases. <Outline> 1.failure condition sin2x + sinxcosx ≤ cos2x </Outline> <Outline> 2.failure condition cos2x + sinxcosx ≤ sin2x </Outline> </Goal> <Path> 1. Let's rewrite this as: sin2x -cos2x + sinxcosx ≤ 0 Hmm, perhaps factoring? … Let me denote t = tanx, then we have t2 + t <=1 … Therefore, the inequality sin2x + sinxcosx <= cos2x holds when x is in (0°, alpha], where alpha=arctan([sqrt5 -1]/2). </Path> <Path> 2. Let's handle the other inequality: cos2x + sinxcosx <= sin2x. … Perhaps use division by cos2x (since cosx >0). Divide by cos2x:1 -tan2x + tanx <=0 … Let that angle be beta = arctan([1 + sqrt5]/2 )≈58.3°. x is in [ beta, 90°). </Path> Map Phase Process Phase Reduce Phase To enable automatic and interpretable control over the generation flow, Multiverse further employs a structured set of specialized control tags that explicitly define each MapReduce block. These tags, such as <Parallel>, delineate the boundaries of such blocks and coordinate all three internal stages. An example is provided in Figure 5.
The process begins with the Map stage, initiated by the <Goal> tag. This tag defines the overall objective, which is then broken down into subtasks using nested and indexed <Outline> tags. After goal specification (signaled by </Goal>), the Process stage starts. In this stage, each subtask is independently mapped and processed within a <Path> block in parallel, matched by its index, constituting the Process stage. Once all paths have finished (signaled by </Path>), the <Conclusion> tag triggers the Reduce stage that merges the results from these paths into a final coherent output ended with </Conclusion> tags.
this section cite: []

Section: Building a Real-world Multiverse Reasoning Model.
To deploy Multiverse in real-world scenarios, we present a comprehensive suite consisting of Multiverse Curator as the data generator, Multiverse Attention as the model architecture, and Multiverse Engine as the serving system. This suite enables a seamless and rapid transition from leading AR models to Multiverse models. In particular, we apply this suite in complex reasoning tasks, leading to a Multiverse model that exhibits strong reasoning capabilities using remarkably low training cost.
this section cite: []

Section: Data Curation: Multiverse 1K.
To address the absence of MapReduce structures in existing sequential reasoning data, we introduce Multiverse-1K. While these long CoT trajactories often inherently contains such structures, explicitly generating them is difficult, as detailed in Section 3. Thus, we develop an automated LLM-assisted pipeline that transforms sequential reasoning chains into parallel MapReduce structures. This convert is guided by a five-stage prompting protocol powered by Gemini 2.5 Pro [14] as shown in Figure 6a.
this section cite: ['b13']

Section: Step 2.2 Step 2.3

this section cite: []

Section: Step 3 Step 4 Step 5

this section cite: []

Section: [Step 2.1-2.2, Step 2.3]: Collective Branches
Inherent Parallelism in Original Reasoning Chain Generating a Summary Tree. First, we iteratively decompose and outline the original reasoning chain into a two-level tree structure. In the first round, the entire reasoning chain is broken down into multiple steps. In the second round, each step is examined by the LLM for further decomposition into substeps. Each resulting step or substep will be labeled and outlined with a concise description.
this section cite: []

Section: Identify Parallel Nodes
Identifying Parallel Groups. Second, we instruct the LLM to analyze each reasoning step, identifying which steps or groups of steps can be executed in parallel without violating logical dependencies.
Reformating into Parallel Structures. Third, the summary tree is converted into a parallel structure based on the previous analysis. To explicitly signal parallel execution, parallelizable steps or step groups are enclosed within the control tags <Parallel> and </Parallel>, forming a parallel block.
Refilling Original Details. Fourth,we prompt the LLM to repopulate the detailed content for each step and substep. This is achieved by retrieving and copying the related original reasoning trajectories.
this section cite: []

Section: Adding MapReduce Structures.
Finally, we further convert the parallel structures into MapReduce structures as defined in Section 4.3. For each parallel block, the LLM generates both the Map and Reduce stages by outlining the specific goals and results for each individual path. Moreover, all paths are rewritten to avoid words implying sequential relations (e.g., similarly) and to prevent including or referencing content from other paths, thereby ensuring each path's completeness and independence.
To further refine our data, two supplementary validation stages are incorporated. After the fourth stage, a content check will filter out data if its edit distance ratio is above 0.2. Next, after the fifth stage, a grammar check will confirm strict adherence to our MapReduce structures. Data failing either case will be iteratively regenerated through our pipeline until both standards are met. The application of this automated pipeline to the s1K-1.1 dataset has yielded Multiverse 1K, a new dataset consisting of 1,000 high-quality, structured reasoning trajectories across a range of math and science problems.
this section cite: []

Section: Algorithm Design: Multiverse Attention.
Next, we introduce Multiverse Attention to replace the causal attention [39] in AR-LLMs. Causal attention computes the i-th token's output with query q i , and keys k j , values v j from positions j ≤ i:
a ij = Softmax (q ⊤ i + P (i)) • (k j + P (j)) + M ij ,(1)
where M ij = 0, j ≤ i -∞, otherwise is the causal mask, and P (i) is the embedding for the i-th position.
However, this formulation poses challenges for parallel generation, as the computation of later paths depend on both (i) the key-value (KV) pairs and (ii) the positional indices produced by earlier paths.
To address this issue, we modify both the attention masks and position indices following APE [44], as illustrated in Figure 6b. In Multiverse Attention, each path within the same Process block starts from an identical position and executes independently without accessing others. During the Reduce stage, all paths converge to the same position, which is set to the maximum position reached by any path to ensure non-negative relative distances, while accommodating variable-length paths.
Moreover, its structural similarity to causal attention brings two key efficiency benefits: (i) Hardware Efficiency: it can preserve training parallelism by using customized attention kernel in FlexAttention [43], and (ii) Data Efficiency: it can be rapidly adapted via post-training on a few samples.
this section cite: ['b38', 'b43', 'b42']

Section: System Implementation: Multiverse Engine.
To enable truly parallel generation in practical deployments, we introduce Multiverse Engine, an extension of existing inference engines designed for AR models. Specifically, we employ SGLang [50] due to its support for continuous batching and radix attention. These features allow dynamic batch scheduling and flexible KV-cache reuse, two scenarios frequently occur in the Map and Reduce stages.
The Map stage is automatically triggered when a <Parallel> token is generated. Next, the scheduler counts the number of <Outline> encountered to decide the degree of parallelism until reaching </Goal>. Based on this count, the engine creates multiple paths executed in parallel as distinct samples within the same batch. Leveraging radix attention, these paths share the prefix KV cache from the current context. Each path is identified and initiated with "<Path> i" according to its order i in the <Outline> list. After prefilling, all paths are added to the decoding queue for parallel generation. When a path finishes, either by reaching </Path> or the maximum length, it enters a "zombie" state that releases all resources and waits for the completion of other paths before continuing.
The Reduce stage begins once all processing paths have completed. In this stage, the engine merges the KV states from all paths along with the preceding context to form a new sequence. Thanks to the flexible memory layout of the radix cache, indices of KV cache can be seamlessly merged without any padding, thereby avoiding both physical data copying and subsequent redundant computation. The token <Conclusion>, prefixed by this combined KV cache, is then added to the prefilling queue. Once finished, the task is moved to the decoding queue to resume generation along the new sequence.
this section cite: ['b49']

Section: Experiments
We evaluate the effectiveness and efficiency of Multiverse in real-world reasoning tasks. Specifically,
• In Section 6.2, Multiverse-32B achieves substantial gains over the Qwen2.5 model by 23.6% after training on Multiverse-1K, and matches or exceeds the accuracies of AR-LLMs on reasoning tasks.
• In Section 6.3, Multiverse-32B scales better than AR-LLMs when using the same generation length.
this section cite: []

Section: Setup.
Training. We created Multiverse-32B by performing SFT on the Qwen2.5-32B-Instruct model [33], integrating our Multiverse Attention. The training data consisted of a combination of Multiverse 1K prompted with "Think step by step and in parallel", and the original sequential data appended by "Think step by step", using a mixture ratio increased from 0:1 (all original data) to 1:0 (all our data) across eight epochs. Our fine-tuning took 3 hours on 8 NVIDIA B200 GPUs with PyTorch FSDP.
Evaluation. Following common practice in assessing reasoning models, we measure Multiverse-32B on four tasks, including AIME24 [24], AIME25 [25], MATH500 [18], and GPQA Diamond [35]. LightEval [17] is employed as the evaluation toolkit, powered by our SGLang [50]-based Multiverse Engine. We test our model under two prompting conditions: with and without the phrase "in parallel".
this section cite: ['b32', 'b23', 'b24', 'b17', 'b34', 'b16', 'b49']

Section: Baselines.
We compare our model with the Qwen2.5 model and an Autoregressive-32B trained using the same data, but without any control tags or the Map and Reduce stages. In addition to pass@1, we report the degree of parallelism (# Parallel) as the ratio between generated token counts and lengths.
this section cite: []

Section: Real-world Reasoning Performance
In Table 2, we report the performance of Multiverse-32B on complex reasoning tasks with 32K contexts, showing improvements of 36%, 32%, 12%, and 15% over the Qwen2.5-32B-Instruct model across the respective benchmarks after fine-tuning. Notably, Multiverse-32B matches or even surpasses the performance of autoregressive models, as demonstrated by its comparison with Autoregressive-32B. For reference, we also include the results of the s1.1-32B model trained on the sequential CoT data from which Multiverse-1K is derived. The comparable performance between these models confirms that our data curation pipeline successfully preserves the original data quality.
We also evaluate Multiverse-32B-Zero, a variant prompted without the "think in parallel" instruction. While both versions exhibit similar performance, Multiverse-32B achieves greater parallelism. This parallelism, measured as the ratio of generated tokens to generation length, aligns with our training strategy, suggesting the potential for controllably switching between AR and Multiverse generation. Moreover, the reduced parallelism on AIME tasks indicates that the model exhibits less parallelism during longer generation, partly due to the scarcity of data exceeding 16K tokens in Multiverse-1K.
this section cite: []

Section: Scaling Performance
To highlight the benefits of parallel generation, we conduct budget control experiments on GPQA-Diamond and MATH500 using the same context length (i.e., approximately equal generation time). We vary the context length from 1K to 4K tokens, during which accuracy increases sharply. As shown in Figure 7, while longer contexts improved performance for both models, Multiverse-32B generates more tokens within the same context length. This parallel scaling yielded performance gains of 2.23% on the GPQA-Diamond (with # parallel = 1.17) and 1.51% on the MATH500 (with # parallel = 1.15).
this section cite: []

Section: Efficiency Analysis
Having demonstrated Multiverse-32B's strong scalability and overall performance, we now further analyze the practical efficiency of Multiverse, showing the potential unlocked through parallel scaling. First, we investigate the relationship between the degree of parallelism and latency per token across various generation lengths (8K, 16K, and 32K), using a batch size of one. The resulting data points, illustrated in Figure 8a, demonstrate that Multiverse enhances generation efficiency by increasing the degree of parallelism. Furthermore, we fit the sampled data points into three inverse curves, one for each. These curves highlight the potential of Multiverse to further reduce latency by encouraging parallelism. Specifically, we identify three key regions based on the sample distribution, demarcated by red lines. The first, encompassing parallelism degrees from 1.0 to 1.3, represents the majority of data points and reflects real-world scenarios, yielding an average speedup of 18.5%. Furthermore, examples show that higher parallelization is achievable, offering acceleration up to 2.1×. The final region, characterized by extended lines, indicates the promising potential for further improvements.
Next, we show the speedup achieved by Multiverse-32B with varying degrees of parallelism across different batch sizes, while keeping a fixed 4K output length. The results in Figure 8b indicate that inference remains memory-bound as the batch size increases from 1 to 128. Therefore, the speedup of Multiverse is influenced by the degree of parallelism in multiple scenarios, showcasing its scalability.
1.0 1.5 2.0 2.5 3.0 # Parallel 7.5 10.0 12.5 15.0 17.5 Latency per token (ms) y = 15.43/x + 0.60 y = 15.43/x + 1.06 y = 15.38/x + 2.14 8K: sampled data 16K: sampled data 32K: sampled data
this section cite: []

Section: Conclusion
This work proposes Multiverse, a natively parallel generative model based on a MapReduce paradigm that internalizes three stages: (i) a Map stage for adaptive task decomposition, (ii) a Process stage for parallel subtask execution, and (iii) a Reduce stage for lossless result synthesis. To build a real-world Multiverse reasoning model, we co-design our data, algorithm, and system, enabling a seamless and rapid transfer from AR-LLMs. After fine-tuning on Multiverse-1K, our Multiverse-32B achieves performance comparable to AR-LLMs on real-world reasoning tasks, while achieving better performance using the same context length due to parallel scaling. Additionally, such parallel generation also results in an up to 2× efficiency gain across varying batch sizes, based on the degrees of parallelism. We hope Multiverse can be considered a successor to Autoregression for generative modeling. For the Limitations and Broader Impacts, we will discuss them in detail in our Appendix.
this section cite: []

Section: A Limitations
While Multiverse provides a general framework for generative modeling, its application to diverse data and task types beyond LLM reasoning remains underexplored. Moreover, as Multiverse-32B was trained solely using Supervised Fine-Tuning (SFT), a key direction in future research is to integrate Reinforcement Learning (RL) into training to explore and encourage more parallelism, which in turn would require a more robust Multiverse engine.
this section cite: []

Section: B Broader Impacts
Multiverse significantly boosts GPU utilization by enabling massive parallel generation. This modeling framework is particularly beneficial for small-batch and long-context inference scenarios, leading to substantial reductions in latency and corresponding energy consumption. Furthermore, Multiverse enables economies of scale for difficult but parallelizable tasks, decreasing the time per task unit while maintaining near-constant overall latency, even as task complexity increases. This remarkable scalability showcases its potential to address extremely complex tasks in practice that were previously intractable, offering a promising path towards artificial superintelligence (ASI).
this section cite: []

Section: C Prompt of Multiverse Curator
In this section, we release our complete five-stage prompting protocol to create Multiverse-1K, powered by the Gemini 2.5 Pro model. This protocol is engineered to transform any sequential CoT data into Multiverse data.
This protocol starts with a multi-round conversation with the LLM (Stages 1-3) to convert an original reasoning chain into a parallel-structured summary. In Stage 4, both this summary and the original reasoning trajectory are fed to the LLM to repopulate each summarized step with its complete, original details. A content checker then immediately assesses these refilled steps. If the editor distance (e.g., Levenshtein distance between the original trajectory
(s ori ) and its rewritten version (s gen ), denoted as d(s ori , s gen )) is too high, that step is re-generated. To normalize this, a relative editor distance is calculated to decide if a threshold r is exceeded (set to 0.2 in practice): Relative Editor Distance = d(s ori , s gen ) max(length(s ori ), length(s gen ))
Next, in Stage 5, we transform the output from Stage 4 into a MapReduce-structured reasoning trajectory by inserting the Map and Reduce phases that are generated by Gemini 2.5 Pro. To ensure the structural validity of the data, we perform a grammar check using a customized XML interpreter, which filters out invalid entries and extracts the outermost MapReduce blocks in the remaining valid ones. Finally, each path is rewritten separately to produce fully independent reasoning paths. The prompts used in the entire protocol are as follows:
this section cite: []

Section: STAGE 1: Generating a Summary Tree

this section cite: []

Section: Main-Step Extraction
Analyze the given reasoning chain (for a math or coding problem) and pull out every major step. Ignore substeps-only list the top-level insights or actions.
Output format S1: [First major step] S2: [Second major step] S3: [Third major step] . . . SX: [Description of step X] . . .
this section cite: []

Section: Guidelines
• Label each top-level step consecutively ('S1', 'S2', 'S3', . . . ).
• Please capture the entire thought process presented in the reasoning chain, and do not skip any step that includes but not is limited to:
1. Initial problem understanding and analysis 2. All exploration paths (both successful and unsuccessful) 3. Case studies, checks, or tests performed 4. Any "aha" or correction (re-evaluation or re-thinking) moments 5. The final reasoning that yields the solution
• Keep each item concise yet descriptive.
• Do not include any sub-numbering (no 'S2.1', etc.).
• Explicitly split multiple cases or scenarios into different steps. Each case should be allocated an independent step.
this section cite: []

Section: Substep Extraction
Given the output including all main step from a reasoning chain, break it down into all its internal substeps only if it can be meaningfully subdivided into smaller thought units.
Output format S1: [Description of step 1] S2: [Description of step 2] S2.1 [Description of step 2.1] S2.2 [Description of step 2.2] . . . S2.10 [Description of step 2.10] S3: [Description of step 3] S4: [Description of step 4] . . . S10: [Description of step 10] . . .
this section cite: []

Section: Guidelines
• Use the same parent index ('x') as the main step (e.g. if breaking down 'S2', label 'S2.1', 'S2.2', . . . ).
• Capture the entire thought process presented in the reasoning chain, and do not skip any substep that includes but is not limited to: 1. Initial problem understanding and analysis 2. All exploration paths (both successful and unsuccessful) 3. Case studies, checks, or tests performed 4. Any "aha" or correction (re-evaluation or re-thinking) moments 5. The final reasoning that yields the solution • Do not introduce deeper nesting larger than 2 (e.g. 'S2.1.1' is not allowed).
• Explicitly split multiple cases or scenarios into different substeps. Each case should be allocated an independent substep.
this section cite: []

Section: STAGE 2: Identifying Parallel Groups

this section cite: []

Section: Parallelizing Main Steps
Using only the main steps (S1, S2, . . . ) you extracted in Stage 1, identify all steps or contiguous step groups that can be executed in parallel without violating logical dependencies, and rewrite the plan as a structured parallel execution outline.
this section cite: []

Section: Identify Parallel Groups
• Find sets of adjacent main steps with no dependencies among them.
• Label groups P1, P2, . . . and list their step ranges (e.g. [S1+S2, S3], [S4]).
this section cite: []

Section: Rewrite into a Parallel Execution Plan
• Preserve each step's original wording as much as possible.
Output Format:
Parallel
groups: P1: [S1+S2, S3] P2: [S4] ... Parallel execution plan: P1[parallel reason: ...]: S1+S2: [text of S1 + text of S2] S3: [text of S3] P2[parallel reason: ...]: S4: [text of S4] ... Guidelines • Coverage: Include every step exactly once, either alone or inside a parallel group.
• Contiguous Blocks: Combine only adjacent steps into blocks; do not combine non-adjacent steps.
• Strict Parallelism Only: Build a dependency graph: draw an edge from step A to B if B uses A's output. A group P_i may include steps (or blocks) only if there are no edges between them. Treat conditional branches as independent tasks.
• Contiguous Grouping Only: Each parallel group must cover a continuous sequence of steps. Do not parallelize non-adjacent steps.
• Conciseness: Keep each bullet short and stick closely to the original text.
this section cite: []

Section: Parallelizing Substeps
Using only the substeps (S2.1, S2.2, ...) you extracted in Stage 1, identify all substeps or contiguous substep groups can be executed in parallel without violating logical dependencies, and rewrite the plan as a structured parallel execution outline.
this section cite: []

Section: Identify Parallel Groups
• Find sets of adjacent main steps with no dependencies among them.
• Label groups P1, P2, . . . and list their step ranges (e.g.
[S2.1+S2.2, S2.3], [S3.1]). 2. Rewrite into a Parallel Execution Plan • Preserve each step's original wording as much as possible. Output Format: Parallel groups: P1: [S2.1+S2.2, S2.3] P2: [S2.4] P2: [S3.1] ... Parallel execution plan: P1[parallel reason: ...]: S2.1+S2.2: [text of S2.1 + text of S2.2] S2.3: [text of S2.3] P2[parallel reason: ...]: S3.1: [text of S3.1] ... Guidelines • Coverage: Include every substep exactly once, either alone or inside a parallel group.
• Contiguous Blocks: Combine only adjacent substeps into blocks; do not combine non-adjacent substeps.
• Strict Parallelism Only: Build an explicit dependency graph in your analysis: draw an edge from substep A to substep B if B uses A's output or insight. A group Pi may include steps (or contiguous blocks) only if there are no edges between any two steps. In conditional logic, treat the if branch and else branch as independent tasks and parallelize them even though their outputs cannot both occur at runtime.
• Contiguous Grouping Only: Each parallel group must cover a continuous sequence of steps or blocks. In other words, you may only parallelize adjacent substeps. The occurrence of substeps in parallel groups must follow their original order. For example, P1: [S2.2, S3.1] is not allowed.
• Conciseness: Keep each bullet short and stick closely to the original text.
this section cite: []

Section: STAGE 3: Reformating into Parallel Structures

this section cite: []

Section: Get Structured Summary
Please summarize the conversation above by extracting the reasoning steps and substeps in Stage 1 as a tree structure with explicit parallelism annotations following Stage 2.
this section cite: []

Section: Output
Format O1: [Brief summary of top-level step S1] <parallel>[parallel reason: ...] O1.1: [Summary of substep S1.1] O1.2: [Summary of substep S1.2] . . . </parallel> <parallel>[parallel reason: ...] O2: [Brief description of top-level step S2] <parallel>[parallel reason: ...] O2.1: [Summary of substep S2.1 + Summary of substep S2.2] O2.2: [Summary of substep S2.3] . . . </parallel> O3: [Brief description of top-level step S3] </parallel> O4: [Brief description of top-level step S4] . . .
this section cite: []

Section: Guidelines
• Max depth of nested <parallel> is 2. Do not nest <parallel> tags more deeply than two levels.
• Max depth of nested numbering is 2. Only use Ox and Ox.y; do not introduce deeper numbering like Ox.y.z.
• Sequential subpaths stay unexpanded. If a node's children are purely sequential, list them normally without any <parallel> wrapper.
• Tag parallel blocks.
Wrap only genuinely parallelizable sibling steps in a <parallel>. . . </parallel> block, and include a parallel-reason annotation.
• Concise summaries. Each step and substep should be described briefly and clearly.
• Avoid over-splitting. If most children are sequential and only a pair can run in parallel, either leave the group un-split or tag only the truly parallel pair.
• Group parallelizable sets. You may combine several independent paths into one <parallel> block when they share no dependencies.
this section cite: []

Section: STAGE 4: Refilling Original Details

this section cite: []

Section: Refill the Full, Detailed Reasoning Trajectories into the Structured Summary
You will receive an outline that may be incomplete but includes <parallel> tags indicating parallel structures. It contain summaries for several steps and substeps. You will also receive the corresponding original text, where sentences implicitly or explicitly map to hierarchical prefixes (e.g., O1, O1.1, O2) in sequence. Your task is to process the original reasoning chain sequentially to update the outline: replace existing summaries or insert new steps as needed, while preserving the original <parallel> tag structure.
Guidelines:
• Initialize Structure Start with the structure provided by the input outline, including its text/summaries and all <parallel> tags in their original locations.
• Read Sentences Sequentially: Process each sentence of the original text one by one, in the exact order they appear.
• Process Each Sentence:
1. Determine the hierarchical prefix associated with this sentence (e.g., O1, O1.1, O2).
2. Check if a step or substep with this prefix already exists in the outline.
this section cite: []

Section: If it exists:
Replace its current summary with the full original sentence. 4. If it does not exist: Insert a new step/substep at the correct hierarchical position (e.g., S1.1 under S1, S2 after S1), using the full original sentence as its content and matching the outline's indentation.
• Preserve <parallel> Tags: Keep every existing <parallel> and </parallel> tag exactly where it was in the input outline. Do not add, remove, or relocate any tags.
• Ensure Correct Output Formatting:
-Maintain proper hierarchical indentation for all steps and substeps.
-Each entry must be on its own line, beginning with its prefix (e.g., O1:, O1.1:), followed by the full original sentence.
• Maintain Completeness: Verify that every sentence from the original reasoning chain has been processed and appears in the updated outline. Do not omit or merge any sentences.
this section cite: []

Section: STAGE 5: Adding MapReduce Structures & Rewriting All Paths

this section cite: []

Section: Filling Detailed Goal and Conclusion Based on the New Reasoning Trajectory
Based on the generated reasoning chain, your task is to transform it according to the following rules:
Output Format [Full reasoning copied from the reasoning chain for the first top-level path] [Full reasoning copied from the reasoning chain for the second top-level path] ... Let's think in parallel. <Parallel> <Goal> Path: [brief, self-contained description of case A] Path: [brief, self-contained description of case B] ... </Goal> <Path> [Introductory reasoning for case A] Let's think in parallel. <Parallel> <Goal> Path: [brief, self-contained description of case A.1] Path: [brief, self-contained description of case A.2] </Goal> <Path> [Full detailed reasoning for case A.1, rewritten clearly and independently] </Path> <Path> [Full detailed reasoning for case A.2, rewritten clearly and independently] </Path> <Conclusion> [Your concise summary of outcomes from A.1 and A.2] </Conclusion> </Parallel> </Path> <Path> [Full detailed reasoning for case B, rewritten clearly and independently] </Path> ... <Conclusion> [Your concise summary of outcomes from A and B] </Conclusion> </Parallel> [Full detailed reasoning for any remaining paths]
this section cite: []

Section: References
Ref_id:b0 Title: The Design and Analysis of Computer Algorithms Year: (1974)
Ref_id:b1 Title: Block diffusion: Interpolating between autoregressive and diffusion language models Year: (2025)
Ref_id:b2 Title: Structured denoising diffusion models in discrete state-spaces Year: (2021)
Ref_id:b3 Title: Large concept models: Language modeling in a sentence representation space Year: (2024)
Ref_id:b4 Title: Large language monkeys: Scaling inference compute with repeated sampling Year: (2024)
Ref_id:b5 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b6 Title: Mapreduce: simplified data processing on large clusters Year: (2008)
Ref_id:b7 Title: Mapreduce: simplified data processing on large clusters Year: (2004)
Ref_id:b8 Title:  Year: (2025)
Ref_id:b9 Title: Unifying autoregressive and diffusion-based sequence generation Year: (2025)
Ref_id:b10 Title: Theoretical benefit and limitation of diffusion language model Year: (2025)
Ref_id:b11 Title: Scaling up test-time compute with latent reasoning: A recurrent depth approach Year: (2025)
Ref_id:b12 Title: Gemini 2.0 flash thinking mode (gemini-2.0-flash-thinking Year: (2024)
Ref_id:b13 Title:  Year: (2025-03)
Ref_id:b14 Title: Gemini 2.5: Our most intelligent ai model Year: (2025-03)
Ref_id:b15 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b16 Title: Lighteval: A lightweight framework for llm evaluation Year: (2023)
Ref_id:b17 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b18 Title: From the American system to mass production Year: (1984)
Ref_id:b19 Title: Openai o1 system card Year: (2024)
Ref_id:b20 Title: Learning to keep a promise: Scaling language model decoding parallelism with learned asynchronous decoding Year: (2025)
Ref_id:b21 Title: Cllms: Consistency large language models Year: (2024)
Ref_id:b22 Title: Discrete diffusion modeling by estimating the ratios of the data distribution Year: (2023)
Ref_id:b23 Title: American Invitational Mathematics Examination 2024 Year: (2024)
Ref_id:b24 Title: American Invitational Mathematics Examination 2025 Year: (2025)
Ref_id:b25 Title: Recursive functions of symbolic expressions and their computation by machine, part i Year: (1960)
Ref_id:b26 Title: Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling Year: (2025)
Ref_id:b27 Title: Smallholders, householders. The ENVIRONMENT in anthropology: A reader in ecology, culture, and sustainable living Year: (1993)
Ref_id:b28 Title: Large language diffusion models Year: (2025)
Ref_id:b29 Title: Introducing openai o3 and o4-mini Year: (2025-04)
Ref_id:b30 Title: Openai o1 system card Year: (2024)
Ref_id:b31 Title: Learning adaptive parallel reasoning with language models Year: (2025)
Ref_id:b32 Title: Qwen2: A family of open-source language models by alibaba cloud Year: (2024)
Ref_id:b33 Title: Qwq-32b: Embracing the power of reinforcement learning Year: (2025-03)
Ref_id:b34 Title: Gpqa: A graduate-level google-proof q&a benchmark Year: (2024)
Ref_id:b35 Title: Hogwild! inference: Parallel llm generation via concurrent attention Year: (2025)
Ref_id:b36 Title: Simple and effective masked diffusion language models Year: (2024)
Ref_id:b37 Title: Simplified and generalized masked diffusion for discrete data Year: (2024)
Ref_id:b38 Title: Attention is all you need Year: (2017)
Ref_id:b39 Title: Remasking discrete diffusion models with inference-time scaling Year: (2025)
Ref_id:b40 Title: Self-consistency improves chain of thought reasoning in language models Year: (2022)
Ref_id:b41 Title: Grok 3 beta -the age of reasoning agents Year: (2025-02)
Ref_id:b42 Title: Long thoughts with short memory Year: (2025)
Ref_id:b43 Title: Ape: Faster and longer context-augmented generation via adaptive parallel encoding Year: (2025)
Ref_id:b44 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2023)
Ref_id:b45 Title:  Year: (2025)
Ref_id:b46 Title: Less is more for reasoning Year: (2025)
Ref_id:b47 Title: Accessing gpt-4 level mathematical olympiad solutions via monte carlo tree self-refine with llama-3 8b Year: (2024)
Ref_id:b48 Title: Qinqing Zheng, and Aditya Grover. d1: Scaling reasoning in diffusion large language models via reinforcement learning Year: (2025)
Ref_id:b49 Title: Efficiently programming large language models using sglang Year: (2023)
