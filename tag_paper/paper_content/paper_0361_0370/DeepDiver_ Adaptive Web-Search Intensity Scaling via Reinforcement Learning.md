Title: DeepDiver: Adaptive Web-Search Intensity Scaling via Reinforcement Learning
Abstract: Information seeking demands iterative evidence gathering and reflective reasoning, yet large language models (LLMs) still struggle with it in open-web question answering. Existing prompting and supervised fine-tuning (SFT) methods remain fixed by prompt rules or training corpora, and are usually benchmarked only on wellstructured wiki sources, limiting real-world adaptability. We introduce WebPuzzle, a 24k-sample training and 275-sample test benchmark that evaluates information seeking on the live internet, across both wiki and open-domain queries. Leveraging 7k WebPuzzle instances, we develop DeepDiver, a reinforcement-learning (RL) framework that cultivates Search Intensity Scaling (SIS)-an emergent ability to escalate search frequency and depth instead of settling on overconfident, underevidenced answers. With SIS, Qwen2.5-7B-Instruct and Pangu-7B-Reasoner attain performance on real-web tasks comparable to the 671B-parameter DeepSeek-R1. We detail DeepDiver's curriculum from cold-start SFT to a well designed RL procedure, and show that its seeking policy generalized from closed-ended queries to open-ended generation such as long-form writing. Our results advance adaptive information seeking in LLMs and provide a rigorous benchmark for future work.Recently, reinforcement learning (RL) [26,13] has been applied to enhance inference-time reasoning in LLMs, enabling iterative refinement and exploration of reasoning [6,31,19]. Several studies

Section: Introduction
Information seeking [34] is a fundamental cognitive skill that involves iterative evidence gathering, reflective reasoning, and the resolution of conflicting information. Despite significant advancements in artificial intelligence, LLMs continue to struggle with replicating such information-seeking behaviors. Knowledge-intensive question answering, a central challenge for LLMs, requires a robust capability for information seeking. Current models often fail to determine when and what information to seek, verify the relevance of evidence, and reason effectively over noisy or conflicting contexts.
Iterative Retrieval-Augmented Generation (RAG) [15] frameworks have been proposed to address these challenges by alternating between retrieval and reasoning. Existing approaches generally fall into two categories: prompting-based and task-specific supervised fine-tuning (SFT). Promptingbased methods leverage predefined rules or in-context learning (ICL) [3], forcing the LLM to follow a specific pipeline to complete complex tasks [11,23,32,42,16]. However, the fixed reasoning flow provided in the prompts limits their adaptability to complex, dynamic problems. In contrast, supervised fine-tuning methods train models to improve retrieval and reasoning capabilities [2,41], generally yielding better performance. However, these methods often internalize inference patterns tied to the training corpus, restricting generalization to more dynamic or unseen situations.
have integrated RL into iterative RAG frameworks, encouraging models to explore diverse reasoning paths and rewarding accurate outcomes [12,25,4,45]. However, these works predominantly train and evaluate their methods on well-structured datasets such as HotpotQA [39], which are based on corpora like Wikipedia. In such settings, many tasks can be effectively solved using the LLMs' internal knowledge, and the introduced search environments are "clean," containing minimal noise or conflicting information. In contrast, real-world search environments are inherently more complex-characterized by noisy, inconsistent, and unreliable sources. This discrepancy limits the generalizability of the reported "incentivized search capabilities" to more realistic, open-ended information-seeking scenarios.
To investigate RL-guided LLM behaviors in more realistic, open-domain scenarios, we introduce WebPuzzle, a dataset designed to evaluate information-seeking capabilities in real-world search environments. WebPuzzle contains 24k training samples and 275 human-annotated test examples, covering tasks solvable with Wikipedia content as well as broader open-domain queries extracted from open-web environment. Even Wikipedia subset are rigorously validated to require external retrieval, ensuring a realistic assessment of LLMs' search behaviors. Along with WebPuzzle, we introduce DeepDiver, an RL-driven search and reasoning framework trained on this dataset. DeepDiver interacts with real-world search engines, continuously refining and denoising retrieved documents to provide accurate answers. A key innovation of DeepDiver is the emergent capability of search intensity scaling (SIS), which dynamically scales up the search frequency and depth as information demands increase. This enables LLMs to tackle more complex, information-intensive problems under open-web environment. Together, WebPuzzle and DeepDiver provide a comprehensive framework for developing and examining information seeking ability of LLMs, offering a promising approach for knowledge-intensive problem solving.
Through systematic empirical analysis, we identify critical factors that influence model behavior, including the search intensity, the training environment, and the generalization capabilities. Our analysis reveals several key insights: (1) DeepDiver exhibits exceptional information-seeking ability via adaptive SIS, where the depth and frequency of searching proportional to both problem difficulty and the model's performance. (2) Compared to the "clean" Wiki-based environment, WebPuzzle and real-world search settings better support complex reasoning beahviours, guiding LLMs to actively supplement evidence, resolve conflicts, verify content, and reflect for self-correction. (3) RL training significantly enhances the generalization capability of LLMs, enabling the transition from closed-ended to open-ended problems. In conclusion, our method underscores the potential of reinforcement learning to foster emergent adaptive search behaviors-specifically, search intensity scaling-in LLMs. This significantly enhances their ability to perform adaptive, verifiable, and scalable information seeking, providing a promising direction for future advancements in knowledgeintensive problem solving.
this section cite: ['b33', 'b14', 'b10', 'b22', 'b31', 'b41', 'b15', 'b40', 'b11', 'b24', 'b44', 'b38']

Section: Preliminaries

this section cite: []

Section: Iterative RAG
We formulate the iterative Retrieval-Augmented Generation (RAG) framework for question answering. Given a question q, the model iteratively performs reasoning and retrieval to produce an answer.
At each iteration t ∈ {1, 2, . . . , T }, the model maintains a reasoning history H t-1 = {q, (r 1 , s 1 , d 1 ), . . . , (r t-1 , s t-1 , d t-1 )}, where r i represents the intermediate CoT generated at round i, s i denotes search queries, and d i denotes retrieved documents from web search.
At round t, conditioned on history H t-1 , the model first generates intermediate reasoning r t ∼ p(r t | H t-1 ) to analyze the current status. Then, based on the reasoning r t , the model selects one of two actions: (1) Search: generate additional queries s t ∼ p(s t | H t-1 , r t ) and retrieve supporting documents d t = Retrieval(s t ); (2) Answer: finalize the answer a ∼ p(a | H t-1 , r t ) to question q. This iterative reasoning-and-retrieval process continues until the model chooses the answer action, resulting in a final answer that is well supported by retrieved evidence and explicit reasoning steps.
this section cite: []

Section: Information Seeking Behaviour
We define information seeking behaviour within iterative RAG frameworks as a structured decisionmaking process: at each iteration the model adopts specific strategies to resolve uncertainties, improve evidence quality, and enhance the overall reliability of answers. Formally, at iteration t, conditioned on the reasoning history H t-1 and current intermediate reasoning r t , the model exhibits several strategies to guide its search and reasoning processes.
Inspired by the findings of Gandhi et al. [7], we categorize these strategies into four types of information seeking behaviours: (1) Evidence Gathering & Supplements, where the model actively seeks to fill identified knowledge gaps by formulating targeted queries s t and retrieving supporting documents d t , formally represented as (s t , d t ) ∼ p(s t , d t | H t-1 , r t ), where d t = Retrieval(s t ). This strategy is exemplified by traditional question-answering datasets such as 2Wiki [9], HotpotQA [39], and FRAMES [14]; (2) Conflict Resolution, where the model reasons about inconsistencies and evaluates competing claims when retrieved information contains contradictions; (3) Verification & Denoising, where the model cross-checks facts and isolates trustworthy information from noisy or irrelevant retrieved content; and (4) Reflection & Correction, where the model periodically re-assesses its reasoning trajectory, revisits earlier assumptions, and explicitly corrects previous reasoning steps for iterative refinement. The latter three behaviours can be represented generally as generating reasoning steps r t ∼ p(r t | H t-1 , d t ), where the specific conditions (e.g. presence of contradiction, noise, or previous mistakes) differ according to each behaviour.
Existing works adopting the wiki-based datasets, limiting their scope to structured and well-organized knowledge bases, and thus predominantly emphasize "Evidence Gathering & Supplements". To prove this observation, we show an detailed analysis in Appendix A.2. In contrast, our proposed WebPuzzle and the real-world searching environment explicitly necessitates employing all four behaviours, thereby reflecting a more comprehensive and realistic scenario for real-world problem-solving with web-searching. More details about WebPuzzle will be included in section 3.1.
this section cite: ['b8', 'b38', 'b13']

Section: Method
In this section, we discuss the details of our approach. We begin by introducing WebPuzzle, a dataset designed to address real-world reasoning and search challenges. Next, we describe DeepDiver, a reinforcement learning-based training framework aimed at enhancing LLMs with robust capabilities introduced in section 2.2.
this section cite: []

Section: WebPuzzle
Unlike existing open-domain QA datasets based on Wikipedia [39,21,14] where LLMs often perform well using only internal knowledge, we introduce WebPuzzle, a dataset designed to evaluate LLMs' ability to locate and reason over noisy, scattered information on the open web. Figure 2 illustrates our data synthesis and curation processes.
this section cite: ['b38', 'b20', 'b13']

Section: Candidate Data Generation
We collect candidate data from Wiki-corpus and real-user queries with retrieved webpages from our deployed smart assistant service. Our generation involves two
Open Domain Webpages Wiki Pages Entity Selection Context Hiding Riddle Generation Cross-Page Info Synthesis QA Generation Cross-Page QA (10k) Open Riddle (7.5k) Wiki Riddle (6k) WebPuzzle Full Set DeepDiver Training Set WebPuzzle (24k) Pass@k-based Difficulty Tagging Outliers (7.5k) N/A (1k) WebPuzzle (7k) Easy (8k) Med. (5.5k) Hard (3k) pass@k = 0 Easy (200) Med. (3.4k) Hard (2.4k) Sampling for DeepDiver approaches: (1) Cross-page question generation, where an LLM extracts facts from web pages to generate "inverted" questions, answers and checklists [29,14]-applied only to open web pages as Wiki-corpus tends to produce overly-simple questions; and (2) Riddle creation, where the LLM selects distinctive entity attributes and applies obfuscation or generalization to create challenging problems, with original entities as labels. Examples appear in Appendix D.1. More quality assurance protocols are shown in Appendix D.3.
this section cite: ['b28', 'b13']

Section: Difficulty Assessment
To ensure stable RL training with consistent reward signals, we tag each problem's difficulty level, enabling a data mixture strategy that prevents all-zero rewards which could lead to training collapse. For each problem, we test DeepSeek-R1 four times, using the number of correct answers to determine difficulty. The formal definition appears in Appendix D.4, with the statistics of the dataset presented in Table 7 and tagging workflow in Appendix E.6.
this section cite: []

Section: Test Set Annotation
Unlike the training set which used LLM labeling, our test set was manually annotated by 5 human experts using an open-web search engine. From 500 seed samples, experts followed the principles in Appendix D.2 to ensure meaningful evaluation of LLMs' informationseeking behaviors. Through iterative annotation, we finalized 275 samples for testing.
this section cite: []

Section: DeepDiver
Building upon the WebPuzzle, we showcase its efficacy within a RL framework designed to explore the information-seeking behavior of LLMs. In this section, we present our method, DeepDiver. DeepDiver ultilize the procedure of cold-start supervised fine-tuning (SFT) followed by reinforcement learning (RL), while incorporates a carefully designed reward assignment and scheduling mechanism to maintain stable RL training.
this section cite: []

Section: Initialization of Reasoning and Searching
To equip DeepDiver with essential reasoning and searching capabilities for WebPuzzle, we implement a cold-start supervised fine-tuning process using diverse data: 2,000 WebPuzzle samples across difficulty levels, 300 real-user questions from our deployed smart assistant, 2,200 general reasoning problems from recent studies [10,17,35,37,43], and 1,000 real-user queries concatenated with retrieved documents. This dataset distills responses from DeepSeek-R1, establishing DeepDiver's foundational abilities to iteratively search and reasoning over retrieved documents. The distillation prompt configuration is detailed in Appendix E.6.
GRPO With Iterative RAG After SFT, we enhance DeepDiver by extending GRPO [24] with iterative RAG. As shown in Figure 3, the model iteratively performs reasoning and searching until reaching an acceptable answer, following the pipeline in Section 2.1. We apply a loss mask to distinguish model-generated from externally retrieved tokens, with GRPO updating parameters based solely on model-generated content.
Extra Search Call Rewards Beyond standard format and accuracy rewards in GRPO, we introduce an extra reward to encourage search engine use for complex problems. When no search-free rollouts solve a problem but at least one search-enabled rollout succeeds, we assign an additional reward of Loose and Strict Rewards Our reward function employs LLM-based graders in a two-stage training approach that transitions from loose to strict grading: the loose grader assigns scores from 1 to 10 (scores ≥ 6 yield 1.0 reward), particularly benefiting early training as shown in Section 5.4. The strict grader conducts three evaluation rounds, requiring at least 2 of 3 positive judgments. Both grader definitions appear in Appendix E.2.
this section cite: ['b9', 'b16', 'b34', 'b36', 'b42', 'b23']

Section: Experiments

this section cite: []

Section: Setup
Data Mixture and Selection Due to computational constraints and capability limits of the 7B model, we train DeepDiver on a carefully selected mixture of 7k WebPuzzle samples rather than the full dataset. We evenly split these into 2k samples for cold-start SFT (Section 3.2) and 5k for RL training. This mixture strategy balances computational efficiency and model effectiveness. Detailed statistics appear in Table 8.
this section cite: []

Section: Benchmark Datasets and Baseline Models
We evaluate performance using closed-ended Chinese benchmarks including C-simpleQA-500 [33,8], FRAMES-zh-230 [14], BamBoogle-zh-71 [21], and our proposed WebPuzzle (detailed in Appendix E.4). For trainable baselines, we use Qwen2.5-7B-Instruct [30] and Pangu-7B-Reasoner [28] as backbone models. Training-free baselines include QwQ-32B [31], GPT-4o [20] and DeepSeek-R1 [6]. We evaluate methods including Prompted without Web Search, Prompted with Iterative RAG, and R1-Distillation (detailed in Appendix E.5). Our evaluation uses the strict grader from Section 3.2, which considers both reference answers and checklists for more robust assessment than conventional LLM-as-a-judge [44]
this section cite: ['b32', 'b7', 'b13', 'b20', 'b29', 'b27', 'b30', 'b19', 'b43']

Section: What is the Relationship between Search Intensity and Performance?
Search intensity is strongly correlated with performance improvements, as increases in search frequency and depth during the RL phase consistently lead to better outcomes. Figure 4 illustrates this relationship during the training phase, showing a clear trend: as search engine calls increase, so do training rewards. For testing results, despite SFT's progress compared with the prompting methods,
:HE3X]]OH )5$0(6]K %DP%RRJOH $FFXUDF\ 'HHS'LYHUYV'HHS6HHN53URPSWLQJ :HE3X]]OH )5$0(6]K %DP%RRJOH 'HHS'LYHUYV4Z43URPSWLQJ :HE3X]]OH )5$0(6]K %DP%RRJOH 'HHS'LYHUYV*37R3URPSWLQJ 7RRO&DOO5RXQG 'HHS'LYHURXUV $FFXUDF\ 'HHS6HHN53URPSWLQJ$FFXUDF\ 4Z43URPSWLQJ$FFXUDF\ *37R3URPSWLQJ$FFXUDF\ 'HHS'LYHURXUV 7RRO&DOO5RXQG 'HHS6HHN53URPSWLQJ7RRO&DOO5RXQG 4Z43URPSWLQJ7RRO&DOO5RXQG *37R3URPSWLQJ7RRO&DOO5RXQG Figure 5: The comparison after removing cases answered correctly through internal knowledge. the model faces a performance bottleneck to adapt to more challenging problems, still lagging behind off-the-shelf APIs with large margin. In contrast, our RL-based DeepDiver-Qwen2.5-7B promotes higher search intensity with an average of 2.51 search and reasoning rounds, substantially higher than the SFT model's 1.75. Similar gains appear in DeepDiver-Pangu-7B, where increased search rounds (1.84 → 2.89) correspond to performance improvements (30.3 → 38.1). This searching intensity scaling enables models to explore and verify more relevant information, enhancing their ability to tackle complex problems. 4.4 Can DeepDiver Generalize from Open-web Training to OOD Wiki-based Problem? Training with WebPuzzle, DeepDiver demonstrates strong generalization capabilities and performance improvements on Wiki-based problems. DeepDiver shows impressive generalization on Wikibased benchmarks despite not being specifically trained for these tasks. Both DeepDiver-Qwen2.5-7B and DeepDiver-Pangu-7B significantly outperform their distilled variants and demonstrate substantial improvements over cold-start models. While DeepSeek-R1 performs well on Wiki-based problems without web search, it shows modest gains when combined with iterative RAG pipeline. This suggests DeepSeek-R1 may have already internalized the necessary knowledge for Wiki-based problems, highlighting the importance of our proposed WebPuzzle benchmark. We further investigate this hypothesis through isolated tests on information seeking and verification in section 5.1.
this section cite: []

Section: Analysis
This section focuses on the Qwen2.5-7B-Instruct model, a simpler model comparing with the Pangu-7B-Reasoner. We analyze several key aspects, including isolated evaluations of information-seeking behavior, comparisons with concurrent related work, the design of the reward function, and the model's generalization to open-ended problems. Additional analyses-such as the relationship between search intensity and problem difficulty, statistics of information-seeking behavior across different training and testing environments, comparisons between human and DeepDiver performance, and detailed case studies-are provided in Appendix A.
this section cite: []

Section: Isolation Testing of Information-Seeking
While DeepDiver lags behind models such as QwQ and DeepSeek-R1 on certain datasets in Section 4, our primary focus is investigating information-seeking behavior rather than knowledge memorization. This raises a question: When isolating evaluation to focus purely on information seeking ability, how does DeepDiver compare to strong baselines?
Setup We conduct pairwise comparisons between DeepDiver and each baseline. For each pair, we perform k = 3 tests without web search to evaluate whether problems can be solved using internal knowledge alone. We calculate the pass@k rate to filter out problems solvable by both models, then analyze accuracy on the remaining problems with the iterative RAG pipeline.
this section cite: []

Section: Results
DeepDiver exhibits exceptional information-seeking capabilities, comparable to all baselines on problems that cannot be solved by internal knowledge alone. While our 7B DeepDiver initially trails behind 671B baselines in full-set tests, results shift when isolating information-seeking behavior. As Figure 5 shows, on problems challenging even for larger models, DeepDiver demonstrates competitive performance across all benchmarks. Notably, it outperforms DeepSeek-R1 across all domains, with a 5.1-point lead on WebPuzzle. This suggests our 7B model's limitations in full-dataset performance stem primarily from its smaller size limiting internal knowledge. However, when tackling problems requiring external information search and verification, DeepDiver's information-seeking capability demonstrates strength in addressing real-world open-web problems.
this section cite: []

Section: Comparisons with Wiki-based Methods
To highlight wiki-based training environments' limitations, we compare DeepDiver with prior wikibased methods. Despite being trained entirely in Chinese, we evaluate DeepDiver on English benchmarks with English search engines to demonstrate its robustness and generalizability.
this section cite: []

Section: Open-Web Problems Wiki-based Problems
WebPuzzle-en BamBoogle FRAMES HotpotQA
R1-Searcher [25] 13.7 (1.9) 46.7 (2.0) 25.3 (1.9) 57.9 (2.3) DeepResearcher [45] 15.0 (7.5) 53.9 (7.1) 33.6 (7.2) 56.6 (4.4) DeepDiver-Qwen 26.1 (14.7) 56.8 (9.1) 32.0 (14.2) 58.4 (10.4)
Table 2: The comparison results with relevant works on the English evaluation dataset using English search engine environment. The number in () indicates the average number of search queries invoked.
this section cite: []

Section: Setup
We use R1-Searcher [25] and DeepResearcher [45] as baselines-both trained in English using Wiki-based corpora. Search engine settings appear in Appendix E.8. For evaluation, we translate WebPuzzle into English via Qwen Max [30], use the full Bamboogle dataset [21] (125 examples), and randomly sample 300 examples from FRAMES [14] and HotpotQA [39]. For fairness, we report accuracy based on the average judgment across all methods.
this section cite: ['b24', 'b44', 'b29', 'b20', 'b13', 'b38']

Section: Results
Despite the language gap, DeepDiver-trained in a real-world Chinese internet setting using WebPuzzle queries-outperforms Wiki-based baselines on most tasks, underscoring the strength of SIS. As shown in Table 2, DeepDiver significantly outperforms DeepResearchers on WebPuzzle-en with 11.1 point leads, while maintaining strong results on Wiki-based datasets despite no English training for information-seeking. We attribute this to DeepDiver's use of SIS, which enables intensive information retrieval and verification rather than relying on limited internal knowledge or languagespecific constraints. R1-Searcher and DeepResearcher make significantly fewer search calls than DeepDiver due to their "cleaner" and more constrained training environments, leading to poorer real-world performance when facing the noise and complexity of open information-seeking tasks. For additional results, including individual judge assessments, see Table 9.
this section cite: []

Section: Emergence of the SIS
A natural concern is that SIS could be an artifact of reward shaping rather than a genuine behavior that emerges from training in a real web environment.
To address this concern, we analyze whether the extra search-call reward introduced in Section E.3 consistently encourages the model to prefer search over no-search when both solve the task. Setup Theoretically, recall the bonus is only awarded when at least one search-enabled rollout succeeds and no search-free rollout succeeds for the same prompt group, i.e., it should not reward search when both search and no-search solve the task. Consequently, during training, we tracked the frequency of the extra bonus. Every 10 steps (448 trajectories per step), we counted how often the bonus fired and compared this against the evolving search intensity (average number of tool-use rounds per query).
Results SIS is not merely reward-shaped but an emergent behavior developed during RL training, our reward design serves as a transient early scaffold (not persistent incentive) and differs fundamentally from existing works that explicitly encourages the tool calls. As shown in Figure 6, the
this section cite: []

Section: VWHSORRVH VWHSORRVH VWHSVWULFW

this section cite: []

Section: $FFXUDF\
:HE3X]]OHDFFXUDF\ )5$0(6]KDFFXUDF\ %DP%RRJOH]KDFFXUDF\ :HE3X]]OHWRROFDOOURXQG )5$0(6]KWRROFDOOURXQG %DP%RRJOH]KWRROFDOOURXQG
this section cite: []

Section: 7RRO&DOO5RXQG
Figure 7: Training results with different reward functions show that a looser reward function stabilizes initial RL training, while a stricter reward function helps overcome later bottlenecks.
special search reward exhibits a clear phasing-out trend: its trigger percentage drops from 4.5% (198 occurrences) in steps 0-9 to 0.1% (6 occurrences) in steps 70-80. Two key observations confirm its transient role: (1) Even in the earliest training phase, only a small fraction of trajectories (4.5%) received the reward, ruling out "over-rewarding"; (2) After step 30, the trigger percentage remains ≤ 1.1%, indicating the reward is essentially inactive in mid-to-late training. More importantly, Figure 4 shows that tool-use rounds grow sharply during steps 80-120-well after the auxiliary reward fades. This confirms SIS is not a reward-driven "phenomenon" but an emergent behavior: the model proactively leverages external tools to compensate for internal knowledge limitations, even without direct incentives.
this section cite: []

Section: Tolerance of the Reward Function
During DeepDiver-Qwen7B's RL training, we observed a reward plateau after approximately 80 optimization steps. We investigated potential factors including learning rate scheduler, exploration diversity, environmental instability, and gradient issues, but found no obvious problems. We therefore focused on the reward function design as a potential cause of the performance plateau.
Setup Starting from checkpoints obtained after 80 optimization steps, we compared DeepDiver's performance under continued training with two different reward functions: the loose and strict rewards introduced in Section 3.2. Both guided continued training from steps 80 to 120. We evaluated performance on WebPuzzle test sets, analyzing accuracy and search intensity trends.
Results A looser reward function stabilizes the initial training phase of RL, while a stricter reward function helps overcome bottlenecks in the later stages. Our results show that a looser reward function stabilizes early RL training, but continuing with it doesn't always lead to improvements. As Figure 7 shows, when transitioning from loose rewards (first 80 steps) to stricter rewards, we observed a nearly 9-point performance increase on WebPuzzle (from 29.1 to 37.6), compared to almost no improvement when continuing with loose rewards. On FRAMES-zh-230, continued training with loose rewards caused a sharp 7-point performance drop, while the stricter reward function continued driving performance upward.
this section cite: []

Section: Generalization to Open-ended Problems
DeepDiver is trained exclusively on closed-ended WebPuzzle problems, adaptively scaling search intensity based on complexity. We investigate whether these capabilities can generalize to open-ended tasks like long-form writing.
this section cite: []

Section: Setup
We evaluate DeepDiver on ProxyQA [27] against R1-Distilled baselines. Since DeepDiver generates Chinese responses, we translate all ProxyQA meta-questions and sub-questions for evaluation. Testing prompt and evaluator configuration follow the original study. We analyze generalization benefits gained through RL training compared to distillation.
Results RL training significantly enhances the generalization capability of LLMs, enabling transition from closed-ended to open-ended problems and demonstrating strong adaptability to long-form writing tasks. As shown in Table 6, our RL-guided DeepDiver achieves 32.72%, outperforming the R1-distilled model by 9.47 percentage points. This suggests RL training enables more effective information seeking and validation in open-web environments, resulting in more comprehensive responses. Additionally, DeepDiver's response length and search queries are substantially higher than the distilled model's, providing evidence that search intensity scaling encourages active information acquisition for more comprehensive answers.
this section cite: ['b26']

Section: Related Work
Prompting-based strategies-including in-context learning [3] and retrieval-augmented chain-ofthought pipelines [11,32,23]-enable zero-or few-shot question answering, yet their fixed templates rarely adapt retrieval depth to unforeseen information gaps. Supervised fine-tuning (SFT) improves the synergy between retrieval and generation [2,41] but can overfit corpus-specific inference patterns, hindering transfer to noisy settings. Reinforcement-learning (RL) methods let LLMs decide when and what to search, achieving state-of-the-art results on curated benchmarks such as HotpotQA [12,25,4,45,39], yet they remain evaluated mostly in "clean" Wikipedia-style environments. Beyond these directions, tool-augmented agents that interleave reasoning with web search [18,40,38] similarly demonstrate promise but still rely on limited test beds, underscoring the need for benchmarks that reflect real-world, noisy information-seeking scenarios. Additional introduction of the related works are shown in Appendix B.
this section cite: ['b10', 'b31', 'b22', 'b40', 'b11', 'b24', 'b44', 'b38', 'b17', 'b39', 'b37']

Section: Conclusion
We conducted a comprehensive investigation into various aspects of information-seeking behavior in LLMs for solving real-world, knowledge-intensive problems.
Our findings indicate that an RL-driven framework, when combined with open-web search engines, enables LLMs to scale search intensity and adapt to tasks of varying difficulty levels. We introduced WebPuzzle, a large-scale dataset designed specifically for developing and testing LLMs' information-seeking behavior, and DeepDiver, a 7B parameter LLM powered by WebPuzzle, which demonstrates competitive performance when compared to the 671B DeepSeek-R1 model on knowledge-intensive tasks. Additionally, we explored key factors influencing RL training and the behavior of LLMs. Our work empowers LLMs to spontaneously adapt their seeking behavior, contributing to advancements in the field and providing extensive insights into the information-seeking capabilities of LLMs in real-world tasks. Appendix A Further Analysis A.1 Search Intensity vs. Difficulty To showcase DeepDiver's ability to dynamically adjust search intensity based on problem complexity, we examine the relationship between search intensity and accuracy across fractions with varying difficulty levels in the WebPuzzle. This analysis follows the experimental setup outlined in Section 4 and compares our proposed DeepDiver with the DeepSeek-R1 baseline. Methods WebPuzzle Cross-Page QA-130 Open&Wiki Reddle-145 Easy&Medium-96 Hard&Outliers-179 DeepSeek-R1 (w/o search) 32.6 (0.00) 32.9 (0.00) 53.5 (0.00) 21.6 (0.00) DeepSeek-R1 Iterative RAG 43.8 (1.31) 31.0 (1.64) 61.1 (1.30) 24.2 (1.58) Qwen7b-Ins-R1-Distill 37.2 (1.49) 23.2 (1.99) 45.2 (1.62) 21.6 (1.83) DeepDiver (Ours) 47.4 (2.35) 28.8 (2.65) 55.6 (2.34) 27.9 (2.60)
Table 3: The performance of different subsets in WebPuzzle. The number in () indicates the average number of search call rounds on the subset. Results DeepDiver demonstrates significant benefits from adaptive search intensity scaling, where search intensity is proportional to both problem difficulty and the LLM's performance. As shown in table 3, across all difficulty levels, both DeepDiver and the baseline models show an increasing number of search call rounds as problem complexity rises. However, DeepDiver consistently consumes more search calls, which translates into better performance. In particular, when compared to DeepSeek-R1, DeepDiver outperforms it in the hard and outlier fractions by a large margin. Specifically, DeepDiver achieves a notable 3.7-point performance leading, driven by an average of 2.6 search rounds compared to DeepSeek-R1's 1.59. This demonstrates that DeepDiver, empowered by open-web environment and reinforcement learning, exhibits superior performance on more complex problems.
An interesting observation arises when examining the performance of DeepSeek-R1 on the Wiki Riddle fraction. Although equipped with the iterative RAG pipeline, DeepSeek-R1 experiences a 1.9-point performance drop (from 32.9 to 31.0). We hypothesize that this decline is due to knowledge conflicts between the pre-trained internalized Wiki corpus and the real-world open-web environment, which introduces confusion and hallucination that hinders the model's ability to correctly answer the question. These results further validate the effectiveness of the WebPuzzle and DeepDiver.
this section cite: []

Section: References
Ref_id:b0 Title: Introducing the model context protocol Year: (2024-11)
Ref_id:b1 Title: Self-rag: Learning to retrieve, generate, and critique through self-reflection Year: (2023)
Ref_id:b2 Title: Language models are few-shot learners Year: (2020)
Ref_id:b3 Title: Research: Learning to reason with search for llms via reinforcement learning Year: (2025)
Ref_id:b4 Title: Do not think that much for 2+3=? on the overthinking of o1-like llms Year: (2025)
Ref_id:b5 Title:  Year: (2025)
Ref_id:b6 Title: Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars Year: (2025)
Ref_id:b7 Title: Chinese simpleqa: A chinese factuality evaluation for large language models Year: (2024)
Ref_id:b8 Title: Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps Year: (2020)
Ref_id:b9 Title: Enhancing llm reasoning with reward-guided tree search Year: (2024)
Ref_id:b10 Title: Active retrieval augmented generation Year: (2023)
Ref_id:b11 Title: Search-r1: Training llms to reason and leverage search engines with reinforcement learning Year: (2025)
Ref_id:b12 Title: Reinforcement learning: A survey Year: (1996)
Ref_id:b13 Title: Fact, fetch, and reason: A unified evaluation of retrieval-augmented generation Year: (2024)
Ref_id:b14 Title: Retrieval-augmented generation for knowledge-intensive nlp tasks Year: (2021)
Ref_id:b15 Title: Search-o1: Agentic search-enhanced large reasoning models Year: (2025)
Ref_id:b16 Title: Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems Year: (2024)
Ref_id:b17 Title: Webgpt: Browser-assisted question-answering with human feedback Year: (2022)
Ref_id:b18 Title: Learning to reason with llms Year: (2024)
Ref_id:b19 Title: Hello gpt Year: (2024)
Ref_id:b20 Title: Measuring and narrowing the compositionality gap in language models Year: (2023)
Ref_id:b21 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b22 Title: Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy Year: (2023)
Ref_id:b23 Title: Deepseekmath: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b24 Title: R1-searcher: Incentivizing the search capability in llms via reinforcement learning Year: (2025)
Ref_id:b25 Title: Reinforcement Learning: An Introduction Year: (2018)
Ref_id:b26 Title: Proxyqa: An alternative framework for evaluating long-form text generation with large language models Year: (2024)
Ref_id:b27 Title: Rethinking optimization and architecture for tiny language models Year: (2024)
Ref_id:b28 Title: Multihop-rag: Benchmarking retrieval-augmented generation for multi-hop queries Year: (2024)
Ref_id:b29 Title: Qwen2.5: A party of foundation models Year: (2024-09)
Ref_id:b30 Title: Qwq-32b: Embracing the power of reinforcement learning Year: (2025-03)
Ref_id:b31 Title: Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions Year: (2023)
Ref_id:b32 Title: Measuring short-form factuality in large language models Year: (2024)
Ref_id:b33 Title: Models in information behaviour research Year: (1999)
Ref_id:b34 Title: Measuring the values of chinese large language models from safety to responsibility Year: (2023)
Ref_id:b35 Title: Qwen2 technical report Year: (2024)
Ref_id:b36 Title:  Year: (2023)
Ref_id:b37 Title: Gpt4tools: Teaching large language model to use tools via self-instruction Year: (2023)
Ref_id:b38 Title: Hotpotqa: A dataset for diverse, explainable multi-hop question answering Year: (2018)
Ref_id:b39 Title: React: Synergizing reasoning and acting in language models Year: (2023)
Ref_id:b40 Title: Auto-rag: Autonomous retrieval-augmented generation for large language models Year: (2024)
Ref_id:b41 Title: Inference scaling for long-context retrieval augmented generation Year: (2025)
Ref_id:b42 Title: Chinese open instruction generalist: A preliminary release Year: (2023)
Ref_id:b43 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b44 Title: Reasoning and Search Rounds of ProxyQA -DeepDiver Proxy Question: 你能详细介绍一下世界上最辣的辣椒吗，包括它的起源、培育过程以及导致其极高辣度的因素？ Response Year: (2025)
Ref_id:b45 Title: 世界最辣的辣椒"，首先需要明确具体是哪种辣椒，因为可能有很多种被称作"最辣"的，比如印度 的Naga Viper、阿富汗的Bihun辣椒等，但需要确认权威来源。 Year: ()
Ref_id:b46 Title:  Year: ()
Ref_id:b47 Title:  Year: ()
Ref_id:b48 Title:  Year: ()
Ref_id:b49 Title: ROUND 1: [世界最辣的辣椒具体品种, Naga Viper辣椒起源, 辣椒素含量与辣度关系, 辣椒育种技术最辣品种 Year: ()
Ref_id:b50 Title: 269万SHU，而卡罗莱纳死神为220万，Naga Viper为135.9万，需明确当前世 界最辣品种。 2. 需要补充X辣椒的培育细节，包括杂交品种及技术手段。 Year: ()
Ref_id:b51 Title:  Year: ()
Ref_id:b52 Title:  Year: ()
Ref_id:b53 Title:  Year: ()
Ref_id:b54 Title:  Year: (2023)
Ref_id:b55 Title: ROUND 2: [2023年最辣辣椒吉尼斯, X辣椒培育技术, 辣椒素分布与辣度关系, 辣椒育种抗病性, Naga Viper 起源地 Year: ()
Ref_id:b56 Title: 年吉尼斯认证的X辣椒辣度为269万SHU，超过卡罗莱纳死神的220万，成为新的世界之最， 但需确认2024年是否有新纪录。 Year: (2023)
Ref_id:b57 Title: 提到Naga Viper由英国培育，但2023年X辣椒认证后，需重新评估其地位。 Year: ()
Ref_id:b58 Title:  Year: ()
Ref_id:b59 Title:  Year: ()
Ref_id:b60 Title:  Year: ()
