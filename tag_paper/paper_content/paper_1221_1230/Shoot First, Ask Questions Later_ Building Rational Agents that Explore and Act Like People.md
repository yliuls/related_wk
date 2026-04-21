Title: SHOOT FIRST, ASK QUESTIONS LATER? BUILDING RATIONAL AGENTS THAT EXPLORE AND ACT LIKE PEOPLE
Abstract: Many emerging applications of AI-from scientific discovery to medical diagnosis-require agents to seek information strategically: forming hypotheses, asking targeted questions, and making decisions under uncertainty. In high-stakes settings with limited resources, do language models (LMs) behave like rational agents? Drawing on insights from human cognition, we develop methods to evaluate and enhance agentic information-seeking. First, we introduce a decisionoriented dialogue task called Collaborative Battleship, in which a Captain must balance exploration (asking questions) and action (taking shots), while a Spotter must supply accurate, contextually-grounded answers. Compared to human players (N=42), we find that many LM agents struggle to ask informative questions, produce accurate answers, and identify high-utility actions. To address these gaps, we develop novel Monte Carlo inference strategies for LMs inspired by Bayesian Experimental Design (BED). For Spotter agents, our approach boosts accuracy by up to 14.7% absolute over LM-only baselines; for Captain agents, it raises expected information gain (EIG) by up to 0.227 bits (94.2% of the achievable noise ceiling). Combined, these components yield sharper targeting (+0.303-0.374 F1), and enable weaker LMs, such as Llama-4-Scout, to outperform both humans (8% → 82% win rate) and frontier models (0% → 67% win rate vs. GPT-5) at ≈1% of GPT-5's cost. We replicate these findings on Guess Who?, where our methods significantly boost accuracy (+28.3-42.4 p.p.), demonstrating their general applicability for building information-seeking agents.gabegrand.github.io/battleship

Section: INTRODUCTION
Language models (LMs) are rapidly evolving from chat-based assistants into fully-fledged agents that interact with the world. Some of the most exciting applications of agents-conducting scientific experiments, conjecturing new mathematical theorems, or discovering novel drugs (Gottweis et al., 2025;Lu et al., 2024;Poesia et al., 2024;Schmidgall et al., 2025)-involve seeking "hits" in combinatorially vast hypothesis spaces. Traditional accounts of information-seeking assume agents are capable of various rational, probabilistic inferences; e.g., inferring belief states, reasoning about uncertainty, and navigating explore/exploit tradeoffs (Anderson, 1990;Auer et al., 2002;Lindley, 1956;MacKay, 1992;Sutton & Barto, 2018). To what extent can current LMs, which are typically optimized to answer users' queries (Ouyang et al., 2022;Rafailov et al., 2023), instead ask good questions for themselves? And what strategies can we use to improve their information-seeking abilities at inference time?
In this work, we aim to both evaluate and improve the ability of frontier models to ask goal-directed questions and take actions in a dynamic environment. Our setting is an adaptation of the classic board game Battleship where players may ask natural language questions to gain information about hidden ships. We further extend this paradigm, which was originally developed to study human question-asking (Rothe et al., 2017;2018;2019), into a two-player dialogue and decision-making task. We conduct experiments with both human-human and agent-agent pairings, comparing the strategies that LMs employ against both human behavior and idealized resource rational strategies that combine LMs with Bayesian inference techniques.
Figure 1: Illustration of our Collaborative Battleship game. On each turn, the Captain must choose whether to gather information (ask a question) or take action (shoot at a tile). The Spotter sees the full board, but can only provide yes/no answers. Each role requires well-defined forms of internal reasoning (thought bubbles), which we implement as Monte Carlo inference over an approximate hypothesis space. This framework allows us to compare both humans and LM agents against idealized Bayesian strategies in a controlled setting.
Our Battleship task tests several distinct cognitive capabilities: (1) Asking informative questions that effectively reduce uncertainty; (2) Providing accurate answers that are grounded in both the current observation state and the dialogue context; (3) Taking strategic actions that leverage available information; (4) Navigating explore/exploit tradeoffs in order to balance information-gathering with goal-directed behavior. The minimalistic environment, which shares elements of other challenging text-and grid-based evaluations (Chollet et al., 2024;Guertler et al., 2025;Jansen et al., 2024;Ke et al., 2024;Wang et al., 2022;Yao et al., 2025), provides an ideal testbed for studying Bayesian experimental design (BED; Chaloner & Verdinelli, 1995;Lindley, 1956;Rainforth et al., 2023) in complex state spaces. In particular, questions in Battleship are directly translatable to Python programs; executing these against a sampled "hypothesis space" of game states to compute their expected information gain (EIG) provides a robust way to compare the utility of human and modelgenerated questions.
As the foundation of our study, we collect 126 full human-human game trajectories (N=42 participants), capturing both dialogue and actions. Our BATTLESHIPQA dataset provides two complementary evaluation settings: SpotterQA, which tests grounded answering on 931 gold yes/no questions with expert annotations for answers and various question features, and CaptainQA, which tests full strategic gameplay under limited questions and shots.
Evaluating current LMs on these benchmarks reveals a wide spectrum of capability. Weaker models like Llama-4-Scout only marginally exceed random baselines on both SpotterQA (62.2% accuracy) and CaptainQA (68.8% win rate vs. random firing), while frontier reasoning models like GPT-5 match or exceed average human performance. On the answering side, we find that Python code generation substantially improves grounding: across 15 LMs, it boosts SpotterQA accuracy by 14.7% over direct answering and chain-of-thought baselines, with especially large gains for models such as GPT-4.1 (75.2% → 90.9%) and Claude 4 Opus (86.8% → 94.4%). On the question-asking side, we introduce a simple Bayesian sampling method that substantially improves question quality, raising mean per-question expected information gain (EIG) on CaptainQA by up to 0.227 bits-94.2% of the information-theoretic ceiling. We also observe that many models tend to ask redundant questions that have zero information gain (e.g., Llama-4-Scout: 18.5% of questions; GPT-4o: 14.6%); our method effectively eliminates these cases.
In total, we introduce three rational strategies for question-asking (Bayes-Q), move selection (Bayes-M), and decision-making (Bayes-D). When combined, these strategies yield substantial improvements in overall game performance, as measured by targeting accuracy (+0.397 F1 for Llama-4-Scout, +0.332 F1 for GPT-4o). Remarkably, these Bayesian enhancements enable even weak LMs to reach superhuman-level performance, with win rates of 81-82% against humans and 67% against GPT-5, all while maintaining substantial cost savings (99.7× for Llama-4-Scout and 2.8× for GPT-4o relative to GPT-5). Finally, we replicate our experiments on the Guess Who? task from TextArena (Guertler et al., 2025) and observe similarly significant gains (+42.4 p.p. for Llama-4-Scout and +28.3 p.p. for GPT-4o), demonstrating that our approach generalizes beyond the Battleship domain.
In sum, our work provides both practical and theoretical contributions. Concretely, we introduce a reusable evaluation harness for studying agentic information-seeking, and curate a novel multimodal dataset, BATTLESHIPQA, that captures rich pragmatic phenomena in grounded dialogue. Conceptually, we formalize several Bayesian-inspired inference-time strategies that can be applied to other discovery settings to build rational information-seeking agents.
this section cite: ['b17', 'b41', 'b58', 'b67', 'b3', 'b4', 'b40', 'b42', 'b71', 'b54', 'b60', 'b63', 'b75', 'b11', 'b20', 'b31', 'b32', 'b74', 'b79', 'b7', 'b40', 'b61', 'b20']

Section: THE BATTLESHIP GAME
Our environment draws inspiration from the cognitive science literature, where Battleship-like tasks have previously been utilized to study human information-seeking behavior. In prior work, singleplayer participants viewed partially-revealed game boards and decided what tiles to reveal (Gureckis & Markant, 2009;Markant & Gureckis, 2012;2014) or what questions to ask (Rothe et al., 2017;2018;2019). Here, we adapt this paradigm to study both humans and language model agents. Our Collaborative Battleship game (Fig. 1) is played by two players: a partially-informed Captain who must balance exploration (asking questions) and exploitation (taking shots), and a fully-informed Spotter who must provide accurate answers that are grounded in both the game state and the ongoing dialogue. Further details about the game rules, interface, and data collection are provided in §A.
Our work extends the Battleship paradigm along several key dimensions. (1) Full multi-turn games: Prior work was limited to static "snapshots" of game states; here, we simulate full game trajectories.
(2) Dialogue setting: To enable multi-turn play, we introduce a second player (the Spotter), whose role is necessary to provide real-time answers to the Captain's questions. (3) Python programs: In prior work, questions were represented as programs in a hand-engineered domain-specific language (DSL), with translation performed either manually (Rothe et al., 2017) or using LMs (Grand et al., 2024). Here, we represent questions with Python programs, which are both more expressive and easier for LMs to generate. (4) Information bottleneck: In prior work, players could in principle ask questions like, "What are the coordinates of all the ships?". To prevent players from asking "game-breaking" questions in order to achieve monetary bonuses ( §4.1), we deliberately restrict the Spotter to "Yes" or "No" answers. While less open-ended, such information bottlenecks-also found in games like Twenty Questions, Guess Who, Mastermind, and the original Battleshipensure strategic balance and enable the study of explore/exploit decision-making.
this section cite: ['b21', 'b63', 'b63', 'b18']

Section: FORMAL FRAMEWORK: BAYESIAN EXPERIMENTAL DESIGN
We cast question selection in our Battleship variant as Bayesian Experimental Design (BED): on each turn we choose to explore-by asking a yes/no question to gain information-or to act-by firing at a hidden tile.
• The hidden board is a random variable S ∈ S. The observed partial board x induces the feasible set S ⊢x = { s ∈ S : s is consistent with x }.
• The belief at the start of turn t given history H 1:t is π t (s) = Pr(S = s | x, H 1:t ). Intuitively, this belief places weight on every board that could still be true given what we have seen so far.
• A natural-language question q translates to a deterministic function f q : S → {0, 1}. The true (noise-free) answer is A q := f q (S) ∈ {0, 1}. For the question asked at turn t, we write A t := f qt (S).
this section cite: []

Section: Observation Model (Noisy Spotter)
In our setting, the Spotter plays the role of an observation model. Assuming an oracle Spotter, the predictive probability that the true answer will be "yes" is simply
p t := Pr(A t = 1 | x, H 1:t ) = s∈S ⊢x π t (s) 1{f qt (s) = 1} .(1)
We expect that the Spotter (human or AI) may occasionally make mistakes, either because they misread the board, misinterpreted the question, or miscommunicated the answer. Accordingly, we model the Spotter (and language-to-code translation) as a binary symmetric channel BSC(ε) with flip probability ε ∈ [0, 1 2 ], yielding a noisy answer A t ∈ {0, 1}. Bayesian Belief Update. We start with a basic model of the Captain as a Bayesian ideal observer. After asking q t , the Captain updates their belief by reweighting each candidate board s ∈ S ⊢x by how compatible it is with the observed answer ãt ∈ {0, 1} under the noise model:
π t+1 (s) ∝ π t (s) (1 -ε) 1{ã t = f qt (s)} + ε 1{ã t ̸ = f qt (s)} .(2)
Intuition. Boards that would have produced the observed answer (e.g., f qt (s) = ãt ) get boosted by a factor 1 -ε, while boards that would have said the opposite retain some weight ε because mistakes are possible. This gently shifts mass toward boards consistent with what we saw without discarding alternatives outright when ε > 0.
this section cite: []

Section: Sequential Monte Carlo (SMC) Approximation.
Exact sums over S ⊢x are typically intractable; we therefore maintain a weighted particle approximation {(s j , w
j )} N j=1 which is updated via sequential Monte Carlo (SMC; Doucet et al., 2001) with per-turn resampling; see §B for details.
this section cite: ['b14']

Section: Expected Information Gain.
To decide which of many possible candidate questions q t to ask, we adopt a standard information-theoretic approach (MacKay, 1992;Shannon, 1948) that considers the expected information the Captain will gain about the board from its (noisy) answer:
EIG ε (q t | x, H 1:t ) := I(S; A t | x, H 1:t )(3)
For a BSC(ε), this admits a closed form in terms of the binary entropy H b :
EIG ε (q t | x, H 1:t ) = H b ε + (1 -2ε) p t -H b (ε), p t as in (1) (4
)
Intuition. The first term H b (ε + (1 -2ε)p t ) is the uncertainty in the noisy answer we expect to hear; the second term H b (ε) is the uncertainty injected by the channel itself. Their difference is how much uncertainty about the board we expect to remove by asking this question. This quantity is maximized when p t ≈ 1 2 , i.e., when under our current belief the question is about as likely to return "yes" as "no".
this section cite: ['b42', 'b68']

Section: BAYESIAN STRATEGIES FOR EXPLORATION AND ACTION
We now describe three strategies that leverage the formal framework above to (i) ask questions that maximize expected information gain, (ii) select actions that maximize hit probability, and (iii) decide between asking a question or taking an action at each turn. These strategies are not necessarily globally optimal, but rather resource rational in the sense that they use the current belief π t to maximize expected utility at each turn.
this section cite: []

Section: Asking questions to maximize information gain (Q Bayes ).
To maximize expected information gain, a simple strategy is to sample a set of candidate questions Q (e.g., from an LM) and select the one with highest EIG ε :
q ⋆ t ∈ arg max q∈Q EIG ε (q | x, H 1:t ),(5)
Selecting moves to maximize hit probability (M Bayes ). For a candidate tile u (restricted to unrevealed locations), define the probability of a hit under the current belief and the corresponding myopic maximum a posteriori (MAP) action:
u ⋆ t ∈ arg max u unrevealed p hit t u | x, H 1:t , p hit t u | x, H 1:t := s∈S ⊢x π t (s)1{u contains ship in s} (6)
this section cite: []

Section: Making rational decisions via one-step lookahead (D Bayes ).
There are many possible strategies for deciding whether to ask a question or take a shot; here, we describe a simple planning-based approach that uses a discounted one-step lookahead to estimate the value of asking a question vs. taking a shot.
this section cite: []

Section: Expected post-question hit probability.
For a candidate question q t , we can estimate the next-turn MAP hit probability by marginalizing over possible answers ã
p hit t+1 u | x, H 1:t , q t := ã∈{0,1} Pr( A t = ã | x, H 1:t , q) p hit t+1 u | x, H 1:t , ã .(7)
Here Pr( A t = 1 | •) = ε + (1 -2ε) p t with p t as in (1), and the inner hit probability is evaluated under the updated belief given ã. Let γ ∈ [0, 1] discount the value of information acquired this turn.
The strategy proceeds as follows:
1. Select a potential question q ⋆ t by maximizing EIG via Eq. ( 5).
2. Compute the pre-and post-question MAP moves u ⋆ t and u ⋆ t+1 and corresponding hit probabilities as in Eqs. (6) and (7). 3. If γ p hit t+1 (q ⋆ t | x, H 1:t ) > p hit t u ⋆ t | x, H 1:t , ask q ⋆ t ; otherwise, act (shoot) at u ⋆ t .
Intuition. When γ = 1, the policy asks a question iff the expected one-step improvement in the nextturn MAP hit probability exceeds the current MAP chance; smaller γ prefers acting sooner. While optimizing over longer horizons is possible, the problem of belief-space planning is PSPACE-hard (Papadimitriou & Tsitsiklis, 1987); in practice one-step lookahead is simple and effective.
this section cite: ['b55']

Section: EXPERIMENTS
We present our experimental evaluation in three parts. First, in §4.1, we describe our human behavioral study and dataset, BATTLESHIPQA, in order to build an empirical account of human-like information seeking. Next, we present a series of experiments that evaluate the ability of LMs to answer and ask questions, leveraging BATTLESHIPQA in two distinct ways. In §4.2, we focus on grounded question-answering, evaluating the performance of LMs in answering questions asked by humans. Finally, in §4.3, we focus on question-asking, evaluating the ability of LMs to ask informative questions and make strategic moves in the full game setting.
this section cite: []

Section: EVALUATING HUMAN INFORMATION-SEEKING BEHAVIOR
We conducted a two-player, synchronous behavioral study in which participants played Collaborative Battleship (described in §2). Participants (N=42) were recruited from Prolific and randomly partnered with another participant. Each pair alternated between the Captain and Spotter roles over 6 games, with a total time commitment of 48-60 minutes and average earnings of $12.03-$14.62/hr (including bonuses based on targeting score). In total, we collected data for 18 pre-sampled, 8×8 game boards, allocated evenly across pairs. Further details about the experimental design, implementation, and compensation are provided in §A.2.
Analysis of human data. We manually annotated the human data in order to (1) establish a set of gold labels for evaluating model performance and (2) categorize the types of questions being asked. First, we annotated all questions with a "gold answer" label and filtered out questions for which there was inter-annotator disagreement. After this process, we obtained a gold dataset of 931 questions, establishing a human accuracy baseline of 92.5%.
Next, we categorized questions into two groups based on whether they could be answered solely from the true board S (simple) or required additional context from the game history H 1:t (complex). Within the "complex" category, we futher annotated for various linguistic and pragmatic phenomena, such as discourse-dependence (e.g., "Is it longer than 3 tiles?" where "it" references a specific ship), state-dependence (e.g., "Should I keep firing up?"), vagueness (e.g., "Is there a ship near the center?") and ambiguity (i.e., multiple possible interpretations). Further definitions and details are given in §A.3; illustrated examples are provided in §A.4.
this section cite: []

Section: SPOTTERQA: MODELING QUESTION-ANSWERING
We begin our modeling experiments with a focused evaluation of models' performance in the Spotter role, since providing accurate answers is a crucial signal of models' ability to reason about the game state. In particular, we are interested in two answering strategies that we hypothesize should improve 200 400 600 800 1000 200 400 600 800 1000 Time (s) 0.0 0.2 0.4 0.6 0.8 1.0 Completion (hits / total) ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? pair_00 pair_07 pair_08 pair_09 pair_10 pair_12 pair_18 (b) Timeline of multiple human-human games of Battleship for a single board (B02). Each ? represents a question asked; colors indicate different players. Figure 2: Human results highlights. (a) Asking questions correlates with performance (ρ = 0.684, p < 0.002); (b) Different players demonstrate widely varying explore/exploit strategies; some alternate between asking questions and making moves, while others focus on information-gathering before taking any actions. 0.5 0.6 0.7 0.8 0.9 1.0 Accuracy claude-sonnet-4 claude-opus-4 gemini-2.5-flash llama-3.1-70b-instruct llama-3.1-405b-instruct llama-4-maverick llama-4-scout gpt-4o-mini gpt-4o gpt-4.1-nano gpt-4.1-mini gpt-4.1 o3 o4-mini gpt-5 (a) By LM Base CoT Code CoT + Code 0.5 0.6 0.7 0.8 0.9 1.0 Accuracy Human Performance (b) By Spotter Strategy Simple Complex 0.5 0.6 0.7 0.8 0.9 1.0 Accuracy (c) By Question Complexity Base LM CoT Code CoT + Code Human Figure 3: SpotterQA key results. (a) Accuracy varies significantly across the 15 LMs tested. (b) Across LMs, code generation improves accuracy over direct answering and CoT baselines. (c) Whereas human accuracy remains roughly consistent, LMs degrade significantly on "complex" questions that require context. Dashed lines indicate mean human accuracy (µ = 92.5%) w/r/t gold answer labels. Error bars indicate 95% CIs.
performance: chain-of-thought reasoning (CoT; Kojima et al., 2022;Nye et al., 2021;Wei et al., 2022) and language-to-code translation (Code; Austin et al., 2021;Wong et al., 2023, see §D.1.1 for examples of synthesized programs). We evaluate these two strategies alongside a combined CoT + Code condition and a direct-answer Base LM condition.
Evaluation Details Each item in the SpotterQA benchmark is a tuple (q t , A t , x t , H 1:t ) containing a question, answer, observed board, and history from the human experiment. Boards are represented as Numpy arrays (prompting details in §C.3). 1 Since questions are independent given the history, sequences can be split up for efficient parallel processing. Accordingly, we are able to evaluate a broad (but non-exhaustive) set of frontier LMs from Anthropic (Anthropic, 2025), Google (Comanici & Team, 2025), Meta (Grattafiori et al., 2024;Meta AI, 2025), and OpenAI (OpenAI, 2024;2025a;b;c). We query all models via the OpenRouter API with default parameters.
this section cite: ['b34', 'b50', 'b76', 'b5', 'b77', 'b19', 'b48', 'b68']

Section: RESULTS
Fig. 3 summarizes the results of our SpotterQA evaluation; full details are provided in §D.1. We highlight three key findings:
Question-answering abilities vary widely across models. As seen in Fig. 3a, accuracy varies widely across the 15 LMs tested, ranging from near-chance (52.5%; GPT-4o-mini) to 92.8% (o3 mini). Notably, several models (o3, o4-mini, and GPT-5) meet or exceed mean human performance (92.5%). (See Table 3 for full results.)
Code generation consistently improves answering accuracy. Across models (Fig. 3b), we find that code improves SpotterQA by 13.2% absolute accuracy points over Base; combining code generation with chain-of-thought (CoT + Code) yields even greater 14.7% gains. A two-sided Mann-Whitney U test confirms that these differences are significant (p < 0.001; see Fig. 17). Importantly, for many models, code generation substantially closes the gap between LMs and human performance: for instance, Claude 4 Opus improves from 86.8% (Base) to 94.4% (CoT + Code), and GPT-4.1 improves from 75.2% (Base) to 90.9% (CoT + Code). These results demonstrate that code generation helps LMs to produce more accurate, grounded answers.
LMs struggle with context-dependent questions. As shown in Fig. 3c, whereas human accuracy remains roughly consistent across simple (92.8%) and complex (91.9%) questions, LMs degrade significantly on complex questions (as defined in §4.1). For instance, GPT-4o's accuracy drops from 72.8% on simple questions to 60.4% on complex ones; similarly, Llama-4-Scout drops from 68.0% to 54.0%. Code generation partially mitigates this gap, but even the best model (o3) still falls short of human performance on complex questions (87.4% vs. 91.9%). These results suggest that current LMs still struggle with pragmatic reasoning and context-dependent meanings.
this section cite: []

Section: CAPTAINQA: MODELING QUESTION-ASKING
In this section, we evaluate the ability of LMs to play the Captain role in the full game setting. We compare a variety of Captain strategies that differ in how they select moves, ask questions, and decide whether to ask or shoot (as formalized in §3.1). A summary is provided in Table 1.
Captain Decision (D) Question (Q) Move (M) Random Move - i, j ∼ Unif({1, . . . , 8}) Greedy Move - arg max i,j πt(• | x) LM pLM(• | x, H1:t) pLM(• | x, H1:t) pLM(• | x, H1:t) + Bayes-Q • • • arg max q∈Q EIGε(q | x, H1:t) • • • + Bayes-M • • • pLM(• | x, H1:t) arg max i,j πt(• | x, H1:t) + Bayes-QM • • • QBayes MBayes + Bayes-QMD p hit (πt) > γp hit (πt+1 | qt, At) • • • • • •
Table 1: Summary of Captain strategies. Random and Greedy are move-only baselines; LM is a pure language model, which the Bayes strategies build upon. Triple-dots indicates inheritance from the row above.
Experimental Details We evaluate each Captain strategy over 54 games (the 18 pre-sampled boards from the human experiment; §4.1, with 3 random seeds per board). Each game enforces the same constraints as the human setting: a maximum of 15 questions and 40 moves. To control for answer quality, we fix the Spotter to GPT-5 (CoT + Code) for all strategies and set ε = 0.1, calibrated from GPT-5's SpotterQA accuracy ( §4.2). Because each turn involves multiple API queries which must be made sequentially, we restrict CaptainQA evaluation to three representative LMs: Llama-4-Scout (small, non-reasoning), GPT-4o (large, non-reasoning), and GPT-5 (large, reasoning-capable). We evaluate every combination of LM and Captain strategy, with the exception of Bayes-QMD, which we were not able to evaluate with GPT-5 due to cost (see Table 5 for a breakdown). For D Bayes , we set γ = 0.95 to encode a minor discount for future action. For strategies that use LMs, the prompt includes the current board state and the full game history (see §C.2 for prompting details).
this section cite: []

Section: Captain Evaluation Metrics
We evaluate Captains using five primary metrics that capture different aspects of performance:
• Targeting Score (F1): Overall metric of ship-sinking performance, balancing both precision and recall. This metric treats the board as a binary classification task, where ship tiles correspond to the positive class and water tiles correspond to the negative class.
• Move Count: The average number of shots taken per game; equivalently, the average game length (max 40 moves; questions do not count towards the move count).
• Questions Asked: The average number of questions asked per game (max 15).
• Win Rate: A pairwise metric based on board-matched performance in simulated head-to-head play. For each pair of Captains on the same board, the winner is the one that sinks all ships in the fewest moves, with tiebreaking based on targeting score (F1). Since Captains do not play each other directly, this comparison is computed post-hoc over all pairs of games, averaging over boards.
• EIG: The average expected information gain of questions asked. We use ε = 0.1, meaning that the maximum possible EIG is 1 -H b (0.1) = 0.531 bits.
this section cite: []

Section: RESULTS
Fig. 4 summarizes the results of our CaptainQA evaluation; full details are provided in §D.2. We highlight four key findings:
Incorporating Bayesian strategies brings weaker models up to super-human performance. As seen in Fig. 4a, the LM-only strategy shows a wide range of performance, with Llama-4-Scout achieving only 0.367 F1, while GPT-5 reaches 0.716. However, adding Bayesian question, move, and decision strategies (+Bayes-QMD) significantly improves performance across all models (e.g., Llama-4-Scout jumps 0.367 → 0.764 F1, GPT-4o 0.450 → 0.782 F1). Remarkably, with the full Bayesian model, both Llama-4-Scout and GPT-4o outperform both humans (0.82-0.83 win rate vs. humans) as well as the strongest LM (0.67 win rate vs. GPT-5), suggesting that Bayesian strategies can compensate for weaker LM capabilities (see win rates in Fig. 18). The LMs equipped with this full Bayesian model also outperform GPT-5 at a fraction of its cost (≈1% of GPT-5's cost for Llama-4-Scout, and ≈35% for GPT-4o, see Table 5) Notably, GPT-5 itself does not significantly benefit from Bayesian question or move selection, indicating that it may already be employing effective versions of these strategies internally. Inference scaling yields more informative questions for all models. As shown in Fig. 4b, EIG scales with the number of candidate questions sampled under the Q Bayes strategy. Across models, we see EIG improvements of up to 0.227 bits per question (up to 94.2% of the info-theoretic ceiling). Additionally, Q Bayes significantly reduces the proportion of redundant questions (EIG = 0; Table 4) asked by Llama-4-Scout (18.5% → 0.2%) and GPT-4o (14.6% → 1.2%). (We find that both humans and GPT-5 rarely ask redundant questions.) These findings illustrate how LMs-including models like GPT-5 that already perform extensive test-time reasoning-can further benefit from inference scaling techniques designed to improve question quality.
Asking high-EIG questions is not sufficient to guarantee strong game performance. While Q Bayes consistently improves EIG, this does not consistently translate to improved game performance. For instance, both Llama-4-Scout and GPT-4o see significant EIG gains with Q Bayes , but their targeting scores improve only marginally (+0.021-0.026 F1). This suggests that weaker models may not be able to effectively leverage information into accurate moves. In contrast, M Bayes , which explicitly marginalizes over the implications of each question to compute π t , provides a more reliable mechanism for doing so.
Skilled players ask some questions first, but not all. As seen in Fig. 4c, both humans and GPT-5 tend to ask several questions early in the game, but also reserve some questions for later turns. In contrast, weaker players (e.g., Llama-4-Scout) tend to front-load all 15 questions-this myopic behavior is mitigated by introducing one-step lookahead (D Bayes ), which leads to a more balanced approach. Interestingly, stronger players (humans, GPT-5) ask fewer questions overall (8.0-8.2 vs. 14.1-14.9), indicating that they both gain more information per question (higher EIG; Table 4) and make more effective use of that information in their moves (higher F1; Fig. 4a).
this section cite: []

Section: GENERALIZED INFORMATION-SEEKING GAMES
L M + B a y e s -Q + B a y e s -M + B a y e s -Q M 0.00 0.25 0.50 0.75 1.00
this section cite: []

Section: Success Rate
Llama-4-Scout GPT-4o
Figure 5: Guess Who? Our Bayesian strategies yield consistent improvements, replicating the core findings from Battleship.
We extend our Bayesian strategies to a general family of information-seeking games from TextArena (Guertler et al., 2025). Here, we present results from the board game "Guess Who? " (see §E for full details). This game provides a distinct testbed from Battleship, as it involves richer, object-relational semantics and requires more complex reasoning about entities and attributes (e.g., age, clothing, facial hair, etc.). As shown in Fig. 5, both GPT-4o and Llama-4-Scout's success rates improve significantly with QM Bayes (GPT-4o: 0.617 → 0.900; Llama-4-Scout: 0.300 → 0.724). These results suggest our framework successfully generalizes to information-seeking environments with combinatorial hypothesis spaces.
this section cite: ['b20']

Section: RELATED WORK, DISCUSSION, AND CONCLUSION
Expressing queries and hypotheses with programs. Several notable prior works use languageto-code to support efficient probabilistic inferences (Ellis, 2023;Li et al., 2024;Piriyakulkij et al., 2024;Wang et al., 2024;Wong et al., 2023). Our work integrates hypothesis-driven reasoning with other recent approaches for planning and rational agent behavior (Chiu et al., 2023;Curtis et al., 2025;Ying et al., 2024;2025).
Eliciting user preferences and clarifying ambiguity. Asking informative questions is critical for many user-facing applications of LMs; recent approaches consider structured prompting (Li et al., 2023), entropy reduction (GX-Chen et al., 2025;Hu et al., 2024;Mazzaccara et al., 2024;Piriyakulkij et al., 2023;Rao & Daumé III, 2018;Yu et al., 2020;Zhang & Choi, 2023), Bayesian inference (Handa et al., 2024;Qiu et al., 2025), and constraint satisfaction (Li et al., 2025).
Human information-seeking and resource rationality. Our work is broadly informed by "resource rational" accounts that describe how human behavior is shaped by cognitive constraints (Anderson, 1990;Chater & Oaksford, 1999;Icard, 2025;Leider et al., 2025;Lieder & Griffiths, 2020). In information-seeking tasks, both children and adults greedy heuristics (Markant et al., 2016;Meder et al., 2019;Ruggeri et al., 2016), consider only a few hypotheses at a time (Klayman & Ha, 1989;Vul et al., 2014), and prefer queries that yield easily interpretable information (Cheyette et al., 2023). These ideas motivate our modeling approach, which emphasizes greedy, sample-based strategies over exact planning and inference.
this section cite: ['b15', 'b38', 'b56', 'b73', 'b77', 'b10', 'b13', 'b80', 'b48', 'b36', 'b22', 'b28', 'b46', 'b57', 'b62', 'b82', 'b83', 'b24', 'b59', 'b37', 'b3', 'b8', 'b30', 'b35', 'b39', 'b45', 'b47', 'b66', 'b33', 'b72', 'b9']

Section: LIMITATIONS AND FUTURE WORK
Collaborative Battleship gives rise to many rich pragmatic behaviors. While discussed in § § A.3 and A.4, in general, these are not explicitly modeled; incorporating techniques based on the rational speech acts (RSA) framework (Frank & Goodman, 2012;Hawkins et al., 2017;2023) could yield agents capable of more sophisticated pragmatic reasoning. In particular, people are highly sensitive to the reliability of information (Harris & Corriveau, 2011;Sperber et al., 2010), as reflected in interactions between human players (e.g., Figs. 14 and 15). In place of a fixed ε, a more robust approach would be to infer ε to account for the differences in reliability across individual Spotters. Analogously, in scientific settings, agents may need to adapt to different levels of aleatoric uncertainty (Hüllermeier & Waegeman, 2021).
Our Bayesian strategies rely on the ability to efficiently draw conditional samples s ∼ p(s | x, H) from a generative "world model." While implementable by hand in a domain like Battleship, in more general settings, we may wish to learn a generative model of world states represented as code (e.g., via model synthesis architectures (MSA); Wong et al., 2023;2025) or images; (e.g., via VAEs or diffusion; Alonso et al., 2024;Ha & Schmidhuber, 2018). Finally, building agents that collaborate effectively with people is increasingly important (Boiko et al., 2023;Noti et al., 2025); Collaborative Battleship provides an ideal setting for studying human-agent interactions.
this section cite: ['b16', 'b26', 'b15', 'b25', 'b69', 'b29', 'b77', 'b48', 'b2', 'b23', 'b6', 'b49']

Section: CONCLUSION
In this work, we presented a comparative evaluation of human and agent information-seeking. We introduced a cognitively-inspired Collaborative Battleship task designed to replicate the core components of Bayesian Experimental Design (BED), including forming hypotheses about latent variables, asking and answering questions, and leveraging context-dependent information to inform actions. Our behavioral study provides a rich, multimodal dataset of human interactions, BATTLESHIPQA, enabling systematic comparison with model performance.
Our results highlight both progress and gaps: while smaller models like Llama-4-Scout and GPT-4o struggle to explore and act coherently, larger reasoning models like GPT-5 approach or even surpass human-level performance at steep resource costs. Notably, humans themselves are not Bayesoptimal reasoners; in general they do not ask maximally-informative questions or make exhaustive use of all available resources. Nevertheless, people are remarkably good at asking useful questions, providing contextually-grounded answers, and making well-informed guesses. Prioritizing strategies that are resource rational is therefore essential if we want to build agents that scale beyond benchmarks to collaborate with people on real-world problems.
this section cite: []

Section: References
Ref_id:b0 Title: The Surprising Effectiveness of Test-Time Training for Few-Shot Learning Year: (2024)
Ref_id:b1 Title: Empirica: a virtual lab for high-throughput macro-level experiments Year: (2021)
Ref_id:b2 Title: Diffusion for world modeling: Visual details matter in atari Year: (2024)
Ref_id:b3 Title: The adaptive character of thought. Lawrence Erlbaum Year: (1990)
Ref_id:b4 Title: Finite-time analysis of the multiarmed bandit problem Year: (2002)
Ref_id:b5 Title: Program Synthesis with Large Language Models Year: (2021)
Ref_id:b6 Title: Autonomous chemical research with large language models Year: (2023)
Ref_id:b7 Title: Bayesian Experimental Design: A Review Year: (1995)
Ref_id:b8 Title: Ten years of the rational analysis of cognition Year: (1999)
Ref_id:b9 Title: People seek easily interpretable information Year: (2023)
Ref_id:b10 Title: Symbolic planning and code generation for grounded dialogue Year: (2023)
Ref_id:b11 Title: ARC Prize Year: (2024)
Ref_id:b12 Title: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities Year: (2025)
Ref_id:b13 Title: LLM-Guided Probabilistic Program Induction for POMDP Model Estimation Year: (2025)
Ref_id:b14 Title: Sequential Monte Carlo Methods in Practice Year: (2001)
Ref_id:b15 Title: Human-like few-shot learning via bayesian reasoning over natural language Year: (2023)
Ref_id:b16 Title: Predicting Pragmatic Reasoning in Language Games Year: (2012)
Ref_id:b17 Title: Alan Karthikesalingam, and Vivek Natarajan. Towards an AI co-scientist Year: (2025)
Ref_id:b18 Title: Loose LIPS Sink Ships: Asking Questions in Battleship with Language-Informed Program Sampling Year: (2024)
Ref_id:b19 Title: The llama 3 herd of models Year: (2024)
Ref_id:b20 Title:  Year: (2025)
Ref_id:b21 Title: Active Learning Strategies in a Spatial Concept Learning Game Year: (2009)
Ref_id:b22 Title: Language Agents Mirror Human Causal Reasoning Biases. How Can We Help Them Think Like Scientists? Year: (2025)
Ref_id:b23 Title: World Models Year: (2018)
Ref_id:b24 Title: Bayesian Preference Elicitation with Language Models Year: (2024)
Ref_id:b25 Title: Young children's selective trust in informants. Philosophical Year: (1567)
Ref_id:b26 Title: Convention-formation in iterated reference games Year: (2017)
Ref_id:b27 Title: From partners to populations: A hierarchical Bayesian account of coordination and convention Year: (2023)
Ref_id:b28 Title: Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models Year: (2024)
Ref_id:b29 Title: Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods Year: (2021)
Ref_id:b30 Title: Resource Rationality Year: (2025)
Ref_id:b31 Title: Discoveryworld: A virtual environment for developing and evaluating automated scientific discovery agents Year: (2024)
Ref_id:b32 Title: Can foundation models actively gather information in interactive environments to test hypotheses? Year: (2024)
Ref_id:b33 Title: Hypothesis testing in rule discovery: Strategy, structure, and content Year: (1989)
Ref_id:b34 Title: Large language models are zero-shot reasoners Year: (2022-12-09)
Ref_id:b35 Title: The Rational Use of Cognitive Resources Year: (2025)
Ref_id:b36 Title: Eliciting Human Preferences with Language Models Year: (2023)
Ref_id:b37 Title: QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks? ArXiv preprint Year: (2025)
Ref_id:b38 Title:  Year: (2024)
Ref_id:b39 Title: Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources Year: (2020)
Ref_id:b40 Title: On a measure of the information provided by an experiment Year: (1956)
Ref_id:b41 Title: The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery Year: (2024)
Ref_id:b42 Title: Information-based objective functions for active data selection Year: (1992)
Ref_id:b43 Title: The meaning of meaning: a study of the influence of language upon thought and of the science of symbolism Year: (1927)
Ref_id:b44 Title: A preference for the unpredictable over the informative during self-directed learning Year: (2014)
Ref_id:b45 Title: Self-directed learning favors local, rather than global, uncertainty Year: (2016)
Ref_id:b46 Title: Learning to Ask Informative Questions: Enhancing LLMs with Preference Optimization and Expected Information Gain Year: (2024)
Ref_id:b47 Title: Stepwise versus globally optimal search in children and adults Year: (2019)
Ref_id:b48 Title: The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation Year: (2025)
Ref_id:b49 Title: AI-Assisted Decision Making with Human Learning Year: (2025)
Ref_id:b50 Title: Show Your Work: Scratchpads for Intermediate Computation with Language Models Year: (2021)
Ref_id:b51 Title: GPT-4o system card Year: (2024)
Ref_id:b52 Title: Introducing GPT-4.1 in the API Year: (2025)
Ref_id:b53 Title: GPT-5 system card Year: (2025)
Ref_id:b54 Title: Training language models to follow instructions with human feedback Year: (2022-12-09)
Ref_id:b55 Title: The complexity of markov decision processes Year: (1987)
Ref_id:b56 Title: Doing experiments and revising rules with natural language and probabilistic reasoning Year: (2024)
Ref_id:b57 Title: Active Preference Inference using Language Models and Probabilistic Reasoning Year: (2009)
Ref_id:b58 Title: Learning formal mathematics from intrinsic motivation Year: (2024)
Ref_id:b59 Title: Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models Year: (2025)
Ref_id:b60 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b61 Title: Modern Bayesian Experimental Design Year: (2023)
Ref_id:b62 Title: Learning to ask good questions: Ranking clarification questions using neural expected value of perfect information Year: (2018)
Ref_id:b63 Title: Question asking as program generation Year: (2017-12-04)
Ref_id:b64 Title: Do People Ask Good Questions? Year: (2018)
Ref_id:b65 Title: Asking goal-oriented questions and learning from answers Year: (2019)
Ref_id:b66 Title: Sources of developmental change in the efficiency of information search Year: (2016)
Ref_id:b67 Title: Agent Laboratory: Using LLM Agents as Research Assistants Year: (2025)
Ref_id:b68 Title: A mathematical theory of communication. The Bell System Technical Year: (1948)
Ref_id:b69 Title: Epistemic Vigilance Year: (2010)
Ref_id:b70 Title:  Year: ()
Ref_id:b71 Title: Reinforcement learning: An introduction Year: (2018)
Ref_id:b72 Title: One and done? Optimal decisions from very few samples Year: (2014)
Ref_id:b73 Title: Hypothesis search: Inductive reasoning with language models Year: (2024)
Ref_id:b74 Title: ScienceWorld: Is your agent smarter than a 5th grader? Year: (2022)
Ref_id:b75 Title: emnlp-main Year: ()
Ref_id:b76 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022-12-09)
Ref_id:b77 Title: From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought Year: (2023)
Ref_id:b78 Title: Modeling Open-World Cognition as On-Demand Synthesis of Probabilistic Models Year: (2025)
Ref_id:b79 Title: SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially? Year: (2025)
Ref_id:b80 Title: Grounding Language about Belief in a Bayesian Theory-of-Mind Year: (2024)
Ref_id:b81 Title: Language-Informed Synthesis of Rational Agent Models for Grounded Theory-of-Mind Reasoning On-The-Fly Year: (2025)
Ref_id:b82 Title: Interactive classification by asking informative questions Year: (2020)
Ref_id:b83 Title: Clarify When Necessary: Resolving Ambiguity Through Interaction with LMs Year: (2023)
