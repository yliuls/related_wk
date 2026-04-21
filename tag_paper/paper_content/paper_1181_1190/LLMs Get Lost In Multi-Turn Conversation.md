Title: LLMS GET LOST IN MULTI-TURN CONVERSATION
Abstract: Large Language Models (LLMs) are conversational interfaces. As such, LLMs have the potential to assist their users not only when they can fully specify the task at hand, but also to help them define, explore, and refine what they need through multi-turn conversational exchange. Although analysis of LLM conversation logs has confirmed that underspecification occurs frequently in user instructions, LLM evaluation has predominantly focused on the single-turn, fully-specified instruction setting. In this work, we perform large-scale simulation experiments to compare LLM performance in single-and multi-turn settings. Our experiments confirm that all the top open-and closed-weight LLMs we test exhibit significantly lower performance in multi-turn conversations than single-turn, with an average drop of 39% across six generation tasks. Analysis of 200,000+ simulated conversations decomposes the performance degradation into two components: a minor loss in aptitude and a significant increase in unreliability. We find that LLMs often make assumptions in early turns and prematurely attempt to generate final solutions, on which they overly rely. In simpler terms, we discover that when LLMs take a wrong turn in a conversation, they get lost and do not recover. Multi-turn UnderspecifiedLower Aptitude (-15%) Very High Unreliability (+112%)

Section: 
User Please generate X. I need [Requirement 1], [Requirement 2], also [Requirement 3]. LLM I'm trying to implement X. Do you mean X' ? Answer Attempt One more thing, can you include [Requirement 2]? Absolutely, here it is: def function(y, x): [...] Wrong Answer Single-turn Fully-Specified High Aptitude Low Unreliability Sure thing! def solution(x, y): [...]
this section cite: []

Section: INTRODUCTION
Today's large language models (LLMs) function as conversational interfaces (e.g., ChatGPT, Gemini, Claude), enabling users to interact with the LLM through multiple conversation turns. Such interaction promises to help users not only when they know what they need (i.e., they can fully specify their requirements in an instruction), but also when they don't. In such cases, users might start with an underspecified instruction and further clarify their needs through turn interactions. Though studies of LLM conversation logs have confirmed that underspecification in user instructions is prevalent (Herlihy et al., 2024), LLM systems are typically evaluated in single-turn, fully-specified settings.
Even though a growing body of work proposes to evaluate LLMs in a multi-turn fashion, we identify in our review (Section 2) that most prior work treats the conversation as episodic: conversation turns might relate to each other, but the conversation can effectively be decomposed as an array of subtasks that can be evaluated in isolation. We argue that episodic tasks move away from what is prevalent in human conversation: underspecification (Zipf, 1949;Herlihy et al., 2024).
In this work, we close this gap by creating a simulation environment for multi-turn underspecified conversations -sharded simulation -that leverages existing instructions from high-quality singleturn benchmarks. At a high level, the sharding process we propose transforms existing single-turn instructions into sharded instructions, a set of smaller instructions that jointly deliver the same information as the original instruction. Sharded simulation then ensures that each turn of conversation reveals at most one shard of information per conversation turn, enforcing that the instruction is gradually revealed through the conversation.
On the set of tasks that we experimented on, we observed that models engaged in multi-turn underspecified conversations achieved an average performance of 65%-a 25-point drop from singleturn performances of 90% when they receive the entire instruction at the beginning of the conversation. Notably, we observe this drop in performance even in two-turn conversations, and across all LLMs we test, from small open-weights (LLama3.1-8B-Instruct) to state-of-the-art (Gemini 2.5 Pro).
Furthermore, we decompose the performance degradation into two components: (1) loss in aptitude, and (2) increase in unreliability. We find that in single-turn, LLMs with higher aptitude tend to be more reliable (e.g., GPT-4.1). On the other hand, all LLMs exhibit very high unreliability in multi-turn settings, regardless of aptitude. We refer to this as the lost in conversation phenomenon: when LLMs take a wrong turn in multi-turn conversation, they get lost and do not recover.
We investigate several explanations for this effect and show that the LLMs tend to (1) generate overly verbose responses, leading them to (2) propose final solutions prematurely in conversation, (3) make incorrect assumptions about underspecified details, and (4) rely too heavily on previous (incorrect) answer attempts.
Our findings highlight a gap between how LLMs are used in practice and how the models are being evaluated. Ubiquitous performance degradation over multi-turn interactions is likely a reason for low uptake of AI systems (Southworth et al., 2023;Brauner et al., 2023;Horowitz et al., 2024), particularly with novice users who are less skilled at providing complete, detailed instructions from the onset of conversation (Zamfirescu-Pereira et al., 2023;Knoth et al., 2024). We provide actionable recommendations based on small-scale experiments and make a concrete call-to-action to LLM builders, urging them to prioritize multi-turn reliability in conjunction with aptitude.
this section cite: ['b23', 'b85', 'b23', 'b70', 'b2', 'b24', 'b81']

Section: BACKGROUND AND RELATED WORK
Previous-generation language models (e.g., BART (Lewis et al., 2019), GPT-2 (Radford et al., 2019), or T5 (Raffel et al., 2020)) were designed for single-turn tasks, and so do the evaluation. Conversational agents were instead built as modular systems using language models as a component (Konrád et al., 2021), and were evaluated through human protocols (Deriu et al. 2021;Murakhovs' ka et al. 2023, inter alia).
While a series of work on multi-turn evaluation emerged since the rise of ChatGPT (Zheng et al., 2023b;Kwan et al., 2024), we argue that such works frame conversation as episodic: every turn is a self-contained subtask that can be graded in isolation. We show that such a design overestimates LLM capability. In fact, when the model must fuse dispersed clues across turns, performance drops sharply and consistently (Appendix G.3). Indeed, real users often issue underspecified instructions both in human-AI communication (Herlihy et al., 2024) and in natural human communication-named as "principle of least effort" (Zipf, 1949). We place underspecification at the center of our study. 1A second limitation of prior work on multi-turn episodic evaluation is the mismatch in task granularity: multi-turn subtasks differ from the single-turn counterpart, which prevents direct comparison. We run both settings on a common set of tasks, allowing a precise observation of performance degradation from single to multi-turn. Crucially, we focus on open-ended generation-the predominant real-world use case in code and natural-language tasks (Zheng et al., 2023a;Handa et al., 2025)-rather than short-form generation or classification.
this section cite: ['b40', 'b62', 'b63', 'b32', 'b15', 'b33', 'b23', 'b85', 'b22']

Section: Generate Response

this section cite: []

Section: Failed answer
Non-answer Successful answer Scaling multi-turn experimentation requires simulating a user. Possible options range from templates (Choi et al., 2018;Reddy et al., 2019;Laban et al., 2023;Deng et al., 2024), LLM (Poelitz & McKenna, 2025;Li et al., 2024;Chang et al., 2025;Liang et al., 2024), fixed annotations (Finch et al., 2023;Chang et al., 2025), or real users (Ram et al., 2018;Laban et al., 2021;Chiang et al., 2024), each trading realism for cost and scalability. We adopt an LLM-based simulator that balances diversity and control, treating it strictly as a probe of LLM behavior, not user behavior (due to the approximation on the user end). In addition, our simulation is simplistic and idealized, and the degradations we report likely underestimate those in genuine human-AI conversations (Section 8).
this section cite: ['b10', 'b65', 'b35', 'b14', 'b60', 'b41', 'b5', 'b43', 'b18', 'b5', 'b64', 'b34', 'b9']

Section: SIMULATING UNDERSPECIFIED, MULTI-TURN CONVERSATION
We develop a simulation environment that repurposes existing tasks from single-turn benchmarks. First, we apply a sharding process to transform original fully-specified instructions into sharded instructions. Second, we implement a sharding simulation environment that carries out a multi-turn conversation based on a sharded instruction.
this section cite: []

Section: SHARDING PROCESS
To construct high-quality sharded instructions from the original fully-specified instructions, we define a set of properties that a sharded instruction must satisfy, such as information preservation, clear initial intent, and order insensitivity. We refer to Appendix B for a precise and mathematical definition of shards and their properties. Based on these properties, we develop a semi-automatic sharding process to scale the creation of sharded instructions, which involves (1) segmentation (2) rephrasing (3) verification (4) manual inspection. 2 At a high level, a sharded instruction is composed of a set of shards, each introducing a single element from the original instruction. Taken jointly, the set of shards reflects the same information provided in the fully-specified instruction, with the information explicitly divided across shards. During manual inspection, we reviewed every sharded instruction, merging, splitting, or reordering shards to enhance the naturalness of the sharded sample, and editing phrasing to ensure that each shard represents a natural unit of information a user might reasonably provide in a single turn and not an adversarial representation of the original instruction. This process ensures that the experiments we carried out used sharded instructions that adhered to the properties we defined. Example pairs of fully-specified and sharded instructions are shown in Figure 4.
this section cite: []

Section: SIMULATING SHARDED CONVERSATIONS
Figure 2 outlines our multi-turn simulator for sharded tasks. We implement a loop with three LLMbacked roles. The assistant is the model under test, the user owns the full sharded instruction and chooses which shard to disclose each turn, and the system tags and grades assistant replies.
At turn 1 the user reveals Shard 1, the assistant responds freely, and the system maps that reply to one of seven strategies-clarification, refusal, hedging, interrogation, discussion, missing, or answer attempt-following Herlihy et al. (2024). If a reply contains an answer attempt, we extract the answer span (e.g., a number or code block) to shield scoring from surrounding text, then pass it to a task-specific evaluator. Subsequent turns repeat: the user may expose one additional shard, the assistant replies, and any answer attempt is scored. A conversation's final score is the maximum over all per-turn scores: in a conversation with N shards, the assistant gets up to N answer attempts, and is credited for the best one. From that perspective, the assistant has some advantage in the sharded setting over single-turn simulations as they only permit at most one answer attempt. The conversation ends when an answer is deemed correct or no shards remain.
Choosing the next shard is non-trivial because assistants often ask shard-specific follow-ups. We therefore instantiate the user simulator with an LLM (GPT-4o-mini), giving it the entire instruction and conversation history so it can select and lightly rephrase the shard that best fits the exchange, for instance, responding to a clarification question with the relevant shard rather than revealing shards in a fixed or random order. Before the first turn, the assistant receives only minimal context (e.g., a tool list) as a system prompt; it is never told that the conversation will be underspecified or multi-turn, enabling us to measure default behavior.
The strategy classifier and answer extractor are also prompt-based GPT-4o-mini modules. The classifier's primary role in the simulation loop is to detect answer attempt turns, which trigger answer extraction and scoring; it does not directly influence the user simulator's shard selection. To ensure the validity of the framework, we manually reviewed several hundred conversations and found that errors in any component occurred in fewer than 5% of cases, and less than 2% of cases where the assistant was disfavored. 3 We thus regard the simulator as sufficiently accurate for our experiments.
Appendix K provides an example simulated sharded conversation.
this section cite: ['b23']

Section: SIMULATION TYPES k turn

this section cite: []

Section: Concat Sharded
Recap Snowball We simulate five types of conversation, each with a different pace of information:
FULLY-SPECIFIED (short-form: FULL) Single-turn task: the original instruction appears in turn 1. This serves as the baseline.
SHARDED Multi-turn task: the sharded instruction is revealed across turns as described above. Core setting for underspecification.
CONCAT All shards are concatenated into one bullet-point prompt in a single turn. This setting removes underspecification (like FULL) while retaining sharding rephrasing; a control to ensure the performance drop in SHARDED is due to multi-turn underspecification, not due to rephrasing.
RECAP runs a SHARDED conversation then adds one recap turn that restates every shard, giving the model a last, "fully-specified" chance. This setting tests whether a simple agent-style recap mitigates SHARDED degradation.
SNOWBALL At each turn, the user reveals the next shard and repeats all prior shards, "snowballing" the context. This setting evaluates whether continual reminders reduce the memory burden over long, multi-turn interactions.
this section cite: []

Section: EXPERIMENT

this section cite: []

Section: TASK SELECTION
We construct sharded instructions for six tasks that we use in a large-scale simulation experiment. For each task, we select instructions from one or two high-quality single-turn benchmarks and apply a semi-automatic sharding process (outlined in Appendix C). For each task, we prepare 90-120 sharded instructions. We select popular and diverse generation tasks across programming and nonprogramming use cases. Figure 4 provides an example of an original and sharded instruction for each task, which we introduce below: Solve this problem: Josh decides to try flipping a house. He buys a house for $80k and then puts in $50k in repairs. This increased the value of the house by 150%. How much profit did he make?
Write me a function below_zero to find out if account is ever <0 Input's a list of ints that are transactions.
this section cite: []

Section: [Example 1]
Balance is 0 at the start.
Return True if balance's ever <0, o/w return False [Example 2] Write the Python function def below_zero(ops): """ You're given a list of deposits & withdrawals on a bank account that starts with balance of 0. Detect if at any point the balance < 0, if so return True, otherwise False. >>> [2 example uses] """ I'm giving you a table, please write a sentence describing it. [Table HTML] Actually focus on these highlighted cells: [Highlighted Table HTML] It came from a page about the 2000 Americas Cricket Cup The exact page is [URL] Write a Table caption: [Highlighted Table HTML] The table comes from [URL] about the 2000 Americas Cricket Cup. I've highlighted some cells. I need a summary of 12 documents, on query: [QUERY] I'll give the docs as I get them, consider all of them. Docs 1-2: [Documents 1-2] Just got four more. Docs 3-6: [Documents 3-6] Here's a new batch. Docs 7-10: [Documents 7-10] I've got two more. Docs 11-12: [Documents 11-12] Write a Summary: About the following 12 documents, on the following query: [QUERY] Documents: [Documents 1-12] Functional Accuracy HumanEval & LiveCodeBench Functional Accuracy Spider Exact Match Berkeley Function Calling Leaderboard Exact Match GSM8K BLEU ToTTo Coverage & Citation Summary of a Haystack For each task, an illustrative fully-specified instruction and its sharded counterpart. We sharded 90 to 120 instructions based on existing datasets (Instruction Origin), and re-purposing evaluation.
Code The assistant writes a Python function that implements the instruction. The original instructions are sourced from HumanEval (Chen et al., 2021) and LiveCodeBench (Jain et al., 2024).
this section cite: ['b8', 'b27']

Section: Database
The assistant generates an SQL query given a database's schema and a user query in natural language (Text-to-SQL). The original instructions and databases are sourced from the popular Spider dataset (Yu et al., 2018).
Actions The assistant generates API calls that satisfy a user request given a set of API schemas. We source API schemas and user instructions from Berkeley Function Calling Leaderboard (BFCL) (Yan et al., 2024), a standard benchmark for measuring function calling capabilities.
this section cite: ['b80', 'b79']

Section: Math
The assistant numerically solves an elementary math word problem. Problems are sourced from the GSM8K dataset (Cobbe et al., 2021).
Data-to-text The assistant produces a caption describing a table and several elements of related metadata. The ToTTo dataset (Parikh et al., 2020) is used to prepare sharded instructions.
this section cite: ['b12']

Section: Summary
The assistant generates a summary with citations given a user query and a set of (around twenty) documents. We re-purpose the instructions from Summary of a Haystack (Laban et al., 2024). The instruction in this task requires long-context understanding, a skill models are known to struggle with (Huang et al., 2023;Karpinska et al., 2024;Kim et al., 2024a).
For each task, we reuse the metrics used in the original benchmarks to assess instance-level correctness. We measure Code and Database with the functional accuracy, Actions and Math with semantic equivalence to the reference answer, all of which render a binary correctness. Data-to-Text and Summary are refinement tasks, which get scored on a range (0-100): BLEU (Papineni et al., 2002) for Data-to-Text and a custom LLM-as-a-judge metric ("Joint Score") for Summary. We map binary accuracy to the range of 0-100 (0 = failure, 100 = success) so that all tasks produce scores on a common scale, facilitating aggregation.
this section cite: ['b36', 'b25', 'b28', 'b55']

Section: .2 SIMULATION METRICS
LLMs employ a stochastic process to generate text. When setting the generation parameters to their default (e.g., temperature = 1.0), they generate many distinct responses for a fixed conversation state. We leverage this property to conduct repeated simulations for a given instruction and quantify the variations that occur. Suppose the i-th simulation yields a score S i ∈ [0, 100] from a task-specific evaluator. By running N simulations for an instruction and obtaining the set of scores S = {S i } N i=1 , we define three metrics: averaged performance P , aptitude A 90 , and unreliability U 90 10 :
P = N i=1 S i N, A 90 = percentile 90 (S), U 90 10 = percentile 90 (S) -percentile 10 (S).
Averaged performance P is an unbiased estimate of a model's mean score on an instruction in a given simulation type. Aptitude A 90 is an estimate of a model's 90th percentile score on a given instruction, i.e. a best-case metric that estimates scores obtained in the top 10% of simulations conducted. Unreliability U 90 10 is an interpercentile range estimate measuring the gap between the 90th and 10th percentile estimates, giving a sense of level of degradation that occurs in response quality due to stochasticity in the LLM. 5   Each metric is computed on a per-instruction basis and can be averaged across a corpus of instructions to obtain corpus-level metrics.
For the rest of the paper, we refer to reliability and unreliability interchangeably, with reliability defined as R 90 10 = 100 -U 90 10 . We also simplify the notations to A for aptitude and U for unreliability, though the metrics can be generalized to other percentile thresholds (e.g., A 80 or U 95 5 ).
this section cite: []

Section: SIMULATION SCALE AND PARAMETERS
In the main simulation experiment, we simulate conversations across three types: FULL, CONCAT, and SHARDED on around 100 instances for each of the six tasks. We experiment with 15 LLMs, running N = 10 simulations for each pair of model and simulation type, totaling more than 200,000 simulated conversations. All simulations are conducted with a default temperature of T = 1, however, we performed a supplementary experiment (Section G.2) that explores the effect of temperature on aptitude and reliability. Although simulating ten conversations for each (LLM, instance, simulation type) increases experimental costs ten-fold, it allows us to not only measure averaged performance (P ) more accurately, but also study aptitude and reliability of LLM systems in depth in Section 5.2. All tasks in our experiments require fewer than 20k tokens of total context, with the summarization task requiring the most and all others typically fitting within 8k tokens. Two models (Phi-4, OLMo-2-13B) could not accommodate the summarization task's context length and were excluded from it (indicated by dashes in Table 1). We emphasize that the intent of these experiments is to study multi-turn behavior at regular context lengths, not to stress-test long-context capabilities.
Model FULL CONCAT SHARDED Overall / / 3.1-8B 27.4 64.1 82.9 13.7 63.9 7.6 21.2 47.7 83.0 15.7 62.6 6.5 21.7 25.9 45.5 13.3 37.4 3.4 91.6 62.5 OLMo2 18.8 54.8 56.1 17.2 80.0 -16.3 40.5 49.8 14.3 80.1 -14.4 22.4 13.8 9.0 46.3 -86.5 50.5 3-Haiku 44.8 85.0 83.5 29.8 73.9 11.6 36.3 76.5 80.2 30.1 76.1 9.2 31.5 31.8 55.9 18.6 47.1 1.6 91.6 52.4 4o-mini 75.9 89.3 94.1 35.9 88.1 14.9 66.7 90.7 92.2 31.2 88.0 12.5 50.3 40.2 52.4 19.8 58.7 7.2 93.0 56.2 3.3-70B 72.0 91.1 95.0 34.1 91.7 15.8 52.7 87.9 97.0 32.0 91.8 14.7 51.6 35.4 71.0 22.4 61.5 10.5 93.2 64.2 Phi-4 53.2 87.6 82.7 23.9 89.2 -48.4 79.6 76.0 28.6 90.4 -39.1 33.1 34.1 23.2 52.5 -99.0 61.7 CMD-A 72.0 91.9 98.5 27.7 94.5 24.3 61.6 86.1 98.4 33.2 91.9 21.3 44.9 33.6 72.0 27.9 66.0 4.9 97.3 60.4 4-Scout 73.9 92.7 98.0 35.2 96.3 13.7 60.3 81.5 98.3 28.2 92.9 13.7 46.4 27.1 69.9 26.1 67.0 12.3 91.0 66.1 o3 86.4 92.0 89.8 40.2 81.6 30.7 87.2 83.3 91.5 39.4 80.0 30.4 53.0 35.4 60.2 21.7 63.1 26.5 98.1 64.1 3.7-Sonnet 78.0 93.9 95.4 45.6 85.4 29.3 76.2 81.5 96.0 53.3 87.2 28.9 65.6 34.9 33.3 35.1 70.0 23.6 100.4 65.9 R1 99.4 92.1 97.0 27.0 95.5 26.1 97.1 89.9 97.0 36.7 92.9 24.4 70.9 31.5 47.5 20.0 67.3 17.2 103.6 60.8 4o 88.4 93.6 96.1 42.1 93.8 23.9 82.9 91.7 97.1 32.2 91.9 23.9 61.3 42.3 65.0 20.5 67.9 10.6 94.5 57.9 2.5-Flash 97.0 96.3 88.4 51.2 90.6 29.1 92.5 95.5 89.2 51.9 88.4 29.4 68.3 51.3 42.6 31.0 66.1 26.1 99.3 65.8 4.1 96.6 93.0 94.7 54.6 91.7 26.5 88.7 86.5 98.5 54.4 89.7 26.8 72.6 46.0 62.9 28.6 70.7 13.3 97.9 61.8 2.5-Pro 97.4 97.3 97.8 54.8 90.2 31.2 95.7 94.9 98.1 56.9 89.3 31.8 68.1 43.8 36.3 46.2 64.3 24.9 100.1 64.5 Table 1: Averaged Performance (P ) of LLMs on six tasks ( Code, Database, Actions, Data-to-text, Math, and Summary). For each task, conversations are simulated in three settings: FULL, CONCAT, and SHARDED. Models are sorted in ascending order of average FULL scores across tasks. Background color indicates the level of degradation from the FULL setting. The last two columns average the performance drops from the CONCAT and SHARDED compared to the FULL in percentages across the six tasks. A= 95 U=65 30 A= 80 U=40 40 A= 65 40 A= 95 70 A= 95 Performance 100% 0% 50% Loss in aptitude A= 95 70 Performance Loss in reliability 70 Performance Loss in aptitude & reliability U=25 A= Aptitude U= Unreliability 100% 0% 50% 100% 0% 50% U=25 U=25 U=25 (a) Visualizing Aptitude and Unreliability. Ll a m a 3 .1 -8 B -I n st O LM o 2 -1 3 B C la u d e3 -H ai ku G PT -4 o-m in i Ll am a3 .3 -7 0B -In st Ph i-4 Co m m an d-A Ll am a4 -S co ut o3 Cl au de 3. 7-So nn et De ep se ek -R 1 GP T-4 o Ge mi ni-2.5 -Fl as h GP T-4 .1 Ge mi ni-2.5 -Pr o Full 49% 65 16 47% 67 20 29% 68 39 20% 75 55 14% 73 58 39% 81 42 17% 76 59 13% 74 61 21% 79 58 21% 80 60 15% 78 63 22% 80 59 19% 82 63 14% 82 68 13% 83 70 Concat 50% 63 13 45% 63 18 29% 65 36 22% 73 50 14% 69 55 48% 82 34 20% 74 54 15% 69 54 25% 80 54 23% 80 57 18% 80 62 26% 79 53 22% 83 61 19% 81 62 15% 83 68 Sharded 56% 59 3 48% 50 2 45% 54 9 49% 62 13 47% 65 18 63% 70 7 44% 62 19 48% 65 17 50% 68 18 48% 66 18 51% 65 14 48% 66 18 55% 74 19 47% 71 24 50% 71 20 (b) Observed Model Degradations 1 2 3 4 5 6 7 8 Performance 19% 100 81 49% 87 38 46% 91 45 65% 91 26 65% 94 29 62% 87 26 68% 90 23 71% 90 19 GPT-4o 1 2 3 4 5 6 7 8 Number of shards Performance 32% 90 58 45% 68 23 65% 77 13 58% 74 16 53% 65 13 59% 68 10 56% 65 10 56% 69 13 GPT-4o-mini (c) Gradual Sharding Results
this section cite: []

Section: RESULTS

this section cite: []

Section: AVERAGE PERFORMANCE FINDINGS
Table 1 summarizes results from the simulation. At a high level, every model sees its performance degrade on every task when comparing FULL and SHARDED performance, with an average degradation of -39%. We name this phenomenon Lost in Conversation: models that achieve stellar (90%+) performance in the lab-like setting of fully-specified, single-turn conversation struggle on the exact same tasks in more realistic, underspecified, and multi-turn conversations.
Published as a conference paper at ICLR 2026
In comparison, models perform roughly equivalently in the CONCAT setting, with CONCAT performance averaging 95.1% of the FULL counterpart. This implies that the loss in performance for SHARDED is not explained by potential loss of information in sharded instructions, as such a loss would be reflected in lower CONCAT performance. We observe that smaller models (Llama3.1-8B-Instruct, OLMo-2-13B, Claude 3 Haiku) have more pronounced CONCAT degradations (86-92), and interpret this as indicating that smaller models struggle to generalize as well as larger models: benign rephrasing affects performance more than for larger, more robust models. This lack of robustness to paraphrasing can be observed visually in Table 1: CONCAT degradation (red background) is more pronounced in the top rows (weaker models) than the bottom rows (stronger models).
The last column of the Table ( / ) aggregates performance degradation across the six tasks, summarizing the magnitude of the Lost in Conversation effect for each model. Surprisingly, more performant models (Claude 3.7 Sonnet, Gemini 2.5, GPT-4.1) get equally lost in conversation compared to smaller models (Llama3.1-8B-Instruct, Phi-4), with average degradations of 30-40%. This is in part due to metric definitions. Since smaller models achieve lower absolute scores in FULL, they have less scope for degradation than the better models. In short, no matter how strong an LLM's single-turn performance is, we observe large performance degradations in the multi-turn setting.
When looking at the task-specific breakdown, some models see more muted degradations in certain tasks. For instance, Command-A sees the least degradation on Actions, while Claude 3.7 Sonnet and GPT-4.1 conserve performance well on Code, and Gemini 2.5 Pro in Data-to-Text. This finding indicates that the multi-turn capabilities of models are not uniform across domains and validates the importance of benchmarking models across a wide variety of tasks to investigate model capabilities.
Additional test-time compute (reasoning tokens) does not help models navigate multi-turn underspecification, as the two reasoning models (o3, Deepseek-R1) deteriorate in similar ways to non-reasoning models. This result confirms that additional test-time compute does not, on its own, allow models to strategize over multi-turn conversation. The analysis we conduct identifies a potential root cause: reasoning models tend to generate lengthier responses (on avg. 33% longer than non-reasoning LLMs). As we find in Appendix F, longer assistant responses tend to contain more assumptions, which can confuse the model about user requirements vs. its own prior responses.
this section cite: []

Section: APTITUDE VS. RELIABILITY ANALYSIS
Results presented in Table 1 present averaged performance degradation (P ). We now report on the aptitude and reliability analysis based on metrics A and U . Figure 5b visually summarizes the results of the reliability analysis we conducted on the 15 LLMs included in our simulation experiment. First, looking at the two single-turn settings, we see that models that are more able (higher A) tend to be more reliable (lower U). For instance, the two most able models (GPT-4.1 and Gemini 2.5 Pro) achieve the lowest unreliability. At the lower end, the two models with the lowest aptitude (Llama3.1-8B-Instruct and OLMo-2-13B) are also the most unreliable. In summary, in single-turn settings, models with higher aptitude tend to be more reliable. This fact is known in the community, with arguments made that better models require less prompt engineering, as they are more robust to variations in inputs and outputs (Li et al., 2023).
The sharded setting paints a different picture. Aptitude degrades in a non-significant way between the full and sharded settings, with an average drop of 16%. On the other hand, unreliability skyrockets with an average increase of 112% (more than doubling). More interestingly, though better models tend to have slightly higher multi-turn aptitude, all models tend to have similar levels of unreliability.
In other words, in multi-turn, underspecified settings, all models we test exhibit very high unreliability, with performance degrading 50 percent points on average between the best and worst simulated run for a fixed instruction. This refines our definition of the lost in conversation phenomenon: when comparing single-and multi-turn settings, we find that large performance degradations (P ) are due in large part to increased model unreliability (U), rather than a loss in aptitude (A).
Appendix F explores potential root causes for models getting lost in conversations. We identify four specific causes: (1) LLMs prematurely propose full answer attempts, making assumptions about problem specifications that lead to confusion ( §F.1), (2) they overly rely on previous (incorrect) answer attempts leading to lengthier "bloated" answers ( §F.2), (3) LLMs overly adjust their answers based on the first and last turn of conversation, evidenced by a loss-of-middle-turns phenomenon ( §F.3), and (4) they produce overly verbose answers, which likely introduces assumptions that detract attention from user utterances ( §F.4).
this section cite: ['b42']

Section: GRADUAL SHARDING EXPERIMENT
Revealing minimal amounts of information in each turn (Section 3.2) can seem unrealistic and adversarial. To explore the relationship between the granularity of sharding and the severity of the effect, we propose the gradual sharding experiment.
In the gradual sharding experiment, we selected 31 instructions from our original experiment across multiple tasks and expanded each sharded instruction into seven variants, with the shard-set size growing from 2 to 8 shards. The instruction selection and sharding process are detailed in Appendix L. The process ensured that at each shard set size (from 1 to 8), task complexity is fixed, and the only modified factor is sharding granularity.
We ran simulations for the gradual sharding experiments with two models (GPT-4o and GPT-4omini), with results summarized in Figure 5c. We find that both models get lost in conversation (a minor degradation in aptitude and a large increase in unreliability) with two-shard instructions and beyond. In other words, the gradual sharding experiment indicates that any conversation that involves underspecification and occurs in two or more turns leads to models getting lost in conversation. For users, the granularity at which information is specified does not majorly impact reliability: providing all the information at once (1-shard) is the only effective method to improve reliability.
We note a design decision that affects interpretation: because we selected only instructions with exactly 8 shards, the resulting instructions are inherently more complex than the corpus average (3 to 5 shards). Consequently, even the 2-shard variants in the gradual sharding experiment represent more complex instructions, which may explain why the large drop at N =2 is not followed by a proportionally steeper decline as N grows to 8. A complementary experiment using instructions with varying instruction complexity counts would likely yield a more gradual degradation curve, but would confound granularity with task complexity.
this section cite: []

Section: IMPLICATIONS SUMMARY
System and Agent Builders (Appendix G.1) Modern LLM applications often rely on agent frameworks like LangChain and Autogen to orchestrate problem decomposition, retrieval, and tool use, raising the question of whether LLMs need native multi-turn capabilities at all. We simulate two agent-style interactions (RECAP and SNOWBALL) that repeatedly inject past user instructions to mitigate the performance drop seen in underspecified multi-turn scenarios. In RECAP, once all shards have been revealed, a final corrective turn informs the assistant that its prior solutions were incorrect, restates all shards as a consolidated bullet list, and asks it to try once more. Both strategies improve over SHARDED but fall short of FULL or CONCAT performance. SNOWBALL offers the more achievable improvement in practice (15-20%) through cumulative repetition. The findings show that offloading memory to agent frameworks is not sufficient; LLMs should natively support multi-turn interaction. We also investigate the effect of altering the system prompt, providing an explicit hint to the assistant that the conversation is likely to be multi-turn and underspecified, and found that such a system prompt hint leads to modest gains in performance (+1% across tasks), but does not effectively help the model avoid getting lost in conversation.
this section cite: []

Section: LLM Builders (Appendix G.2).
While the community has focused on improving LLM aptitude, our findings emphasize the importance of model reliability, especially in multi-turn settings. We conducted a temperature ablation experiment (setting T = 1.0, 0.5, 0.0) and found that reliability does improve at lower temperatures in single-turn conversations, but not in SHARDED multi-turn ones. Even at T = 0.0, multi-turn unreliability remains high (30%) due to cascading effects of early stochastic variation. In short, lowering temperature does not mitigate unreliability in multi-turn contexts. We urge LLM builders to jointly optimize for aptitude and reliability, developing models that: (1) maintain similar aptitude in single-and multi-turn settings, (2) demonstrate low unreliability (U 90 10 < 15) in multi-turn, and (3) achieve this performance at standard temperature (T = 1).
this section cite: []

Section: NLP Practitioners (Appendix G.3).
Our proposed sharding procedure is semi-automated but still requires manual effort (3 hours per 100 samples) to ensure quality. We hypothesize that tasks most vulnerable to multi-turn degradation share three properties: (1) they are generative (not extractive), ( 2) sufficiently complex such that sharding would yield 3+ shards per instruction, and (3) are non-episodic tasks, with each new shard requiring the modification of the entire solution. For applicable tasks, we encourage researchers to release sharded variants alongside fully specified datasets.
this section cite: []

Section: Conversational System Users (Appendix G.4).
Users should be aware of LLMs' reliability limitations, particularly in multi-turn settings. We offer two practical recommendations. First, "if time allows, try again"-starting a new conversation with the same information often yields better outcomes than persisting with a model that has become lost in conversation. Second, "consolidate before retrying"-since LLMs struggle with information dispersed across multiple turns, consolidating requirements into a single instruction improves both aptitude and reliability. Users can achieve this by asking the LLM to consolidate all user turns into an instruction used in a new conversation. These strategies remain cumbersome workarounds rather than principled solutions, highlighting the need for LLMs that can reliably handle multi-turn conversations without requiring such interventions.
Appendix G shares additional perspective for each of the audiences listed above.
this section cite: []

Section: CONCLUSION
In this work, we conduct a large-scale simulation of single-and multi-turn conversations with LLMs, and find that on a fixed set of tasks, LLM performance degrades significantly in multi-turn, underspecified settings. LLMs get lost in conversation, which materializes as a significant decrease in reliability as models struggle to maintain context across turns, make premature assumptions, and over-rely on their previous responses. Additional experiments reveal that known remediations that work for simpler settings (agent-like concatenation or decreasing temperature) are ineffective in multi-turn settings, and we call on LLM builders to prioritize the reliability of models in multi-turn settings.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2024)
Ref_id:b1 Title: From single to multi: How llms hallucinate in multi-document summarization Year: (2024)
Ref_id:b2 Title: What does the public think about artificial intelligence?-a criticality map to understand bias in the public perception of ai Year: (2023)
Ref_id:b3 Title: Art or artifice? large language models and the false promise of creativity Year: (2024)
Ref_id:b4 Title: Ai-slop to ai-polish? aligning language models through edit-based writing rewards and test-time computation Year: (2025)
Ref_id:b5 Title: From static benchmarks to human-ai evaluation Year: (2025)
Ref_id:b6 Title:  Year: (2022-10)
Ref_id:b7 Title: Nebula: A discourse aware minecraft builder Year: (2024)
Ref_id:b8 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b9 Title: Chatbot arena: An open platform for evaluating llms by human preference Year: (2024)
Ref_id:b10 Title: Question answering in context Year: (2018)
Ref_id:b11 Title: Decontextualization: Making sentences stand-alone Year: (2021)
Ref_id:b12 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b13 Title: Raphaël Avalos, et al. Command a: An enterprise-ready large language model Year: (2025)
Ref_id:b14 Title: On the multi-turn instruction following for conversational web agents Year: (2024)
Ref_id:b15 Title: Survey on evaluation methods for dialogue systems Year: (2021)
Ref_id:b16 Title: Simulatorarena: Are user simulators reliable proxies for multi-turn evaluation of ai assistants? Year: (2025)
Ref_id:b17 Title: Ambiguity, accessibility, and a division of labor for communicative success Year: (2008)
Ref_id:b18 Title: Don't forget your abc's: Evaluating the state-ofthe-art in chat-oriented dialogue systems Year: (2023)
Ref_id:b19 Title: Semantic underspecification in language processing Year: (2009)
Ref_id:b20 Title: The llama 3 herd of models Year: (2024)
Ref_id:b21 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b22 Title: Which economic tasks are performed with ai? evidence from millions of claude conversations Year: (2025)
Ref_id:b23 Title: On overcoming miscalibrated conversational priors in llm-based chatbots Year: (2024)
Ref_id:b24 Title: Adopting ai: how familiarity breeds both trust and contempt Year: (2024)
Ref_id:b25 Title: Embrace divergence for richer insights: A multi-document summarization benchmark and a case study on summarizing diverse information from news articles Year: (2023)
Ref_id:b26 Title: Gpt-4o system card Year: (2024)
Ref_id:b27 Title: Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code Year: (2024)
Ref_id:b28 Title: One thousand and one pairs: A" novel" challenge for long-context language models Year: (2024)
Ref_id:b29 Title: Fables: Evaluating faithfulness and content selection in book-length summarization Year: (2024)
Ref_id:b30 Title: Beyond prompts: Learning from human communication for enhanced ai intent alignment Year: (2024)
Ref_id:b31 Title: Ai literacy and its implications for prompt engineering strategies Year: ()
Ref_id:b32 Title: Lenka Hỳlová, and Jan Šedivỳ. Alquist 4.0: Towards social intelligence using generative models and dialogue personalization Year: (2021)
Ref_id:b33 Title: Mt-eval: A multi-turn capabilities evaluation benchmark for large language models Year: (2024)
Ref_id:b34 Title: What's the latest? a question-driven news chatbot Year: (2021)
Ref_id:b35 Title: Are you sure? challenging llms leads to performance drops in the flipflop experiment Year: (2023)
Ref_id:b36 Title: Summary of a haystack: A challenge to long-context llms and rag systems Year: (2024)
Ref_id:b37 Title: An intensional parametric semantics for vague quantifiers Year: (2000)
Ref_id:b38 Title: One vs. many: Comprehending accurate information from multiple erroneous and inconsistent ai generations Year: (2024)
Ref_id:b39 Title: Spider 2.0: Evaluating language models on real-world enterprise text-to-sql workflows Year: (2024)
Ref_id:b40 Title: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension Year: (2019)
Ref_id:b41 Title: Iqa-eval: Automatic evaluation of humanmodel interactive question answering Year: (2024)
Ref_id:b42 Title: Instruction-following evaluation through verbalizer manipulation Year: (2023)
Ref_id:b43 Title: Benchmarking mathematical reasoning and instruction following in multi-turn interactions Year: (2024)
Ref_id:b44 Title: We're afraid language models aren't modeling ambiguity Year: (2023)
Ref_id:b45 Title: Lost in the middle: How language models use long contexts Year: (2024)
Ref_id:b46 Title: Revisiting the gold standard: Grounding summarization evaluation with robust human evaluation Year: (2022)
Ref_id:b47 Title: A comparison of approaches to document-level machine translation Year: (2021)
Ref_id:b48 Title: Contextualized evaluations: Taking the guesswork out of language model evaluations Year: (2024)
Ref_id:b49 Title: Goal alignment in llm-based user simulators for conversational ai Year: (2025)
Ref_id:b50 Title: Salespeople vs salesbot: Exploring the role of educational value in conversational recommender systems Year: (2023)
Ref_id:b51 Title: Artificial intelligence (ai) trust framework and maturity model: Applying an entropy lens to improve security, privacy, and ethical ai Year: (2023)
Ref_id:b52 Title: Flipping the dialogue: Training and evaluating user language models Year: (2025)
Ref_id:b53 Title: 2 olmo 2 furious Year: (2024)
Ref_id:b54 Title: OpenAI o3 and o4-mini System Card -openai Year: (2025)
Ref_id:b55 Title: Bleu: a method for automatic evaluation of machine translation Year: (2002)
Ref_id:b56 Title: Totto: A controlled table-to-text generation dataset Year: (2020)
Ref_id:b57 Title: When does in-context learning fall short and why? a study on specification-heavy tasks Year: (2023)
Ref_id:b58 Title: Dealing with semantic underspecification in multimodal nlp Year: (2023)
Ref_id:b59 Title: Humanity's last exam Year: (2025)
Ref_id:b60 Title: Synthetic clarification and correction dialogues about datacentric tasks-a teacher-student approach Year: (2025)
Ref_id:b61 Title: Escaping the sentence-level paradigm in machine translation Year: (2023)
Ref_id:b62 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b63 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b64 Title: Conversational ai: The science behind the alexa prize Year: (2018)
Ref_id:b65 Title: Coqa: A conversational question answering challenge Year: (2019)
Ref_id:b66 Title: Learning to make mistakes: Modeling incorrect student thinking and key errors Year: (2025)
Ref_id:b67 Title: Conversational user-ai intervention: A study on prompt rewriting for improved llm response generation Year: (2025)
Ref_id:b68 Title: Analysing concatenation approaches to document-level nmt in two different domains Year: (2019-11)
Ref_id:b69 Title: Navigating rifts in human-llm grounding: Study and benchmark Year: (2025)
Ref_id:b70 Title: Developing a model for ai across the curriculum: Transforming the higher education landscape via innovation in ai literacy Year: (2023)
Ref_id:b71 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b72 Title: Interactive ai alignment: specification, process, and evaluation alignment Year: (2023)
Ref_id:b73 Title: Search engines in an ai era: The false promise of factual and verifiable source-cited responses Year: (2024)
Ref_id:b74 Title: Akhila Yerukola, Maarten Sap, and Graham Neubig. Interactive agents to overcome ambiguity in software engineering Year: (2025)
Ref_id:b75 Title: Design principles for generative ai applications Year: (2024)
Ref_id:b76 Title: as an ai language model, i cannot": Investigating llm denials of user requests Year: (2024)
Ref_id:b77 Title: Do pre-trained language models detect and understand semantic underspecification? ask the dust! ArXiv Year: (2024)
Ref_id:b78 Title: Autogen: Enabling next-gen llm applications via multi-agent conversation Year: (2023)
Ref_id:b79 Title: Berkeley function calling leaderboard Year: (2024)
Ref_id:b80 Title: Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task Year: (2018)
Ref_id:b81 Title: Why johnny can't prompt: how non-ai experts try (and fail) to design llm prompts Year: (2023)
Ref_id:b82 Title: Lmsys-chat-1m: A large-scale real-world llm conversation dataset Year: (2023)
Ref_id:b83 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b84 Title: Semantic evaluation for text-to-sql with distilled test suites Year: (2020)
Ref_id:b85 Title: Human behavior and the principle of least effort: An introduction to human eoclogy Year: (1949)
Ref_id:b86 Title: Doublebracketed terms are placeholders that get replaced with the actual data. Other tasks share the same outline with different exemplars and rules to enforce stable outputs. Segmentation You are a given a fully specified instruction, and your task is to segment the instruction into a units of information that each reveal a single piece of information of the instruction. You must output a list of segments in the following JSON format: [ {"segment Year: ()
Ref_id:b87 Title: If you have a compound expression (X and Y), you should split it into two segments. Each segment should represent a unit of information Year: (2014)
Ref_id:b88 Title: Example 1: Instruction: What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015? Shards: {"initial_segment": "stadiums", "initial_shard": "I'm looking for active stadiums Year: ()
Ref_id:b89 Title: Response strategy categorization You are reviewing a multi-turn conversation between a user and an assistant, and are given the last turn of the conversation Year: ()
Ref_id:b90 Title: You must classify the response of the assistant according to the response type: -`answer_attempt`: The response contains a complete answer attempt to the user's question (not templated or hypothetical) Year: ()
Ref_id:b91 Title: The response is short (less than 100 words) and contains a single question addressed to the user that directly inquires about an aspect of the user's query. A clarification turn cannot be long (see `discussion`), cannot contain a vague question (see `discussion`) Year: ()
Ref_id:b92 Title: The response contains multiple questions addressed to the user, sometimes organized in a list or bullet-points Year: ()
Ref_id:b93 Title: The response discusses the question in detail, without providing a final answer, asking a specific clarification question, or a refusal to answer. The response may or may not contain a vague question Year: ()
Ref_id:b94 Title: refuse`: The response contains an explicit or implicit refusal to answer the user's question without a follow-up question or a request Year: ()
Ref_id:b95 Title: You must output your answer in the following JSON format: {"response_type": "refuse|missing|answer_attempt|hedge\\ |clarification|interrogation|discussion"} Rules: -The assistant giving a hint at how an answer could look like is not a final answer. You should only select `answer_attempt`if the conversation could end at this stage with the user having an entirely final answer to the problem they've formulated Year: ()
Ref_id:b96 Title: Conversation's last turn Year: ()
