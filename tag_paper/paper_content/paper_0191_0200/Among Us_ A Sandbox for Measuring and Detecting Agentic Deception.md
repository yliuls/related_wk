Title: Among Us: A Sandbox for Measuring and Detecting Agentic Deception
Abstract: Prior studies on deception in language-based AI agents typically assess whether the agent produces a false statement about a topic, or makes a binary choice prompted by a goal, rather than allowing open-ended deceptive behavior to emerge in pursuit of a longer-term goal. To fix this, we introduce Among Us 1 , a sandbox social deception game where LLM-agents exhibit long-term, open-ended deception as a consequence of the game objectives. While most benchmarks saturate quickly, Among Us can be expected to last much longer, because it is a multi-player game far from equilibrium. Using the sandbox, we evaluate 18 proprietary and open-weight LLMs and uncover a general trend: models trained with RL are comparatively much better at producing deception than detecting it. We evaluate the effectiveness of methods to detect lying and deception: logistic regression on the activations and sparse autoencoders (SAEs). We find that probes trained on a dataset of "pretend you're a dishonest model: . . . " generalize extremely well out-of-distribution, consistently obtaining AUROCs over 95% even when evaluated just on the deceptive statement, without the chain of thought. We also find two SAE features that work well at deception detection but are unable to steer the model to lie less. We hope our open-sourced sandbox, game logs, and probes serve to anticipate and mitigate deceptive behavior and capabilities in language-based agents.

Section: Introduction
As large language models (LLMs) become increasingly capable at conversation [1,2] and reasoning ability [3,4], their adoption, especially as agents that can interact with the world, comes with significant risks. An important risk is deception [5], where an agent intentionally misleads others to achieve its own goals. Thus, being able to understand, evaluate, and control deception in AI agents is crucial for their safe and trusted deployment. Prior work has shown models to act deceptively or "scheme" under specific conditions [6] or when asked to do something harmful intentionally such as inserting hidden backdoors in code [7], an ability that is seen to persists through safety training.
However, a major hurdle in studying agentic deception is the lack of suitable sandboxes and model organisms that elicit open-ended, long-term deception. To address this problem, in this work, we introduce "Among Us" as a rich and controlled sandbox to elicit open-ended deception from LLMagents. Based on the Elo rating system used in games like chess [8], we use deception Elo and detection Elo as an unbounded metric for measuring the capability to deceive or detect deception. We show that agents at the frontier are relatively more capable at producing than detecting deception. Finally, we show that interpretability techniques such as activation monitoring (using linear probes) and Sparse Autoencoders (SAEs) help us detect deception out-of-distribution with high accuracy. Among Us is a social deduction and deception game where players assume the roles of crewmates or impostors. Crewmates aim to complete tasks and identify and vote out impostors, while impostors seek to eliminate crewmates without being detected. Chi et al. [9] build AmongAgents, a text-based game logic for Among Us for LLMs, and show that GPT-4 can play the game well with various archetypal deception personalities.
In contrast, we study goal-directed, open-ended deception, i.e., situations where the agent figures out that it should act deceptively and comes up with its own ways to do so. Thus, unlike Chi et al. [9], we provide only the game rules to the LLMs, with no examples, ideas, or personalities. We show that models can craft human-style deceptive ideas on their own (e.g. Figure 1).
Lying and deception are closely related but distinct concepts. Jones and Bergen [10] define a taxonomy of various deceptive and misleading behaviors, and in this work, we define lying as the act of intentionally saying a factually incorrect statement, and deception to be a (true or false) statement or action aimed to mislead and gain advantage (as per the agent's objective). Diverting the discussion to other (factually true) things is an example of deception without lying, and hallucination [11] is an example of lying without deception. Since both behaviors are important for aligning models in different ways, we study both separately in this work.
• We introduce agents playing "Among Us" as a sandbox for eliciting and studying open-ended strategic deception, and introduce "Deception Elo" as an unbounded metric for measuring it.
• Using our elo metrics for deception and detection, we show that RL-trained models are not better at detecting deception than older, RLHF-only models (Figure 5).
• We train linear probes on the activations of open-weight models on various datasets to evaluate their effectiveness in detecting lying and deception out-of-distribution (OOD) data. We find probes are extremely effective at detecting deception, with AUROCs in the 95%-99% range (Section 4).
• We evaluate Sparse Autoencoders (SAEs) features correlated with deception for detection and steering in this out-of-distribution setting. Detection works well, but steering does not (Section 4.3).
Our open-source sandbox environment, along with full logs of 400 complete game rollouts, 2054 multiple-model game summaries, and linear probe weights, are available here.
this section cite: ['b0', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b9', 'b10', 'b11']

Section: Design and Relevance of the Sandbox Environment
In this section, we describe the sandbox in detail and motivate its use as a proxy for human-agent interactions in terms of tractability, richness, and unbounded capability. The reader should consult Chi et al. [9] for more details on the game dynamics and rules. Note the high win-rates and high deception capability in frontier reasoning/"thinking" models (compare with Figure 5).
this section cite: ['b9']

Section: The Game State S t and Observation Space O i t
The game state at time t is defined by S t = (P t , L t , G t ), where P i t is the information about a player i (their role, assigned tasks, progress, cooldown, etc.), L t is the current location of all players, and G t has the game metrics such as the number of steps remaining, players alive, voting results, etc.
The observation space O i t of each player i includes the phase information (see 2.3), the map connections and vents, the player's last k observations and actions, and the tasks (or fake tasks) assigned to them. See Appendix B for an example of the entire observation space provided to the player.
this section cite: []

Section: The Action Space A i
Each player i has an action space A i (during the task phase; see 2.3) that depends on their role:
A i = A C ∪ A S , if i is a Crewmate A I ∪ A S , if i is an Impostor
where A S (the set of common/shared actions) is {Move, Speak, Vote, Report, Call Meeting, Check Security Camera}, A C (Crewmate Actions) is {Complete Task}, and A I (Impostor Actions) are {Kill, Fake Task, Vent}. At each time step, each player looks at their observation and picks an action from their list of available actions. The actions taken by the players update the game state S t .
this section cite: []

Section: Game Phases and Termination
The game starts with randomly initialized players in the task phase and alternates between:
• Task Phase: In this phase, crewmates complete tasks while impostors blend in by performing fake tasks or killing players. It starts again after a meeting if the game does not end.
• Meeting Phase: This phase is triggered by a reported body or an emergency meeting (during the task phase). After 3 rounds of discussion, the players vote to eliminate a suspect.
Each game instance begins with 5 crewmates and 2 impostors, and players take actions sequentially in each phase. Crewmates win if all tasks are completed (impostor tasks do not count) or all impostors are eliminated before the time limit, and impostors win if the number of crewmates is less than the number of impostors or a time limit of 50 steps is reached. This protocol, with (S t , O i t , A i , R t = win/lose) and state transitions, can be modeled as a Markov Decision Process (MDP) for each player.
this section cite: []

Section: Relevance to Real-world Deception Capability
Several reasons make Among Us a good sandbox for studying agentic deception:
Tractability: Chi et al. [9] find that GPT-4, without fine-tuning, can understand and play the game when prompted to be deceptive and with in-context examples (ICE). We show that open-weight models matching GPT-4 in performance can do so too even without fine-tuning and with just rules of the game (no ICE). This makes it a great setting to study strategic deception. We highlight some instances of deceptive behavior in Llama-3.3-70b-Instruct in Figure 1.
Richness (agent-human interaction): Among Us provides a setting for studying agentic behaviors for which we currently lack a good proxy. It encapsulates key agent-human interactions that we care about, including thoughts (using a scratchpad), memory, speech (during discussions), and actions (tasks, voting, and killing). Deceptive behavior emerges in impostors naturally from the game rules, without explicit prompting, training, or backdooring (such as in [7]).
Impostors can act innocently at first to gain trust and later exploit it, making the game a valuable setting to study sycophancy, power-seeking, and deceptive alignment in the future. The performance of crewmates reveals how well LLMs can detect deception and outperform similarly-sized evaluators, thus helping assess the feasibility of AI control [12]. The game inherently involves "out-numbering" and "out-voting" opponents, a relevant phenomenon as AI agents become more prevalent online (see gradual disempowerment [13]). Additionally, it allows for fine-tuning, SAE-based interventions, steering, and direct reward signals for superhuman deceptive capability in a controlled setup.
Measurement of Deceptive Capability: While static benchmarks [14] become obsolete due to increasing capabilities or Goodharting [15], multi-agent games impose no pre-defined ceiling on sophistication and scheming. As long as we maintain an asymmetry between evaluation and generation, we can continue to study progressively advanced levels of deception.
In our sandbox, limits on agent capability arise due to partial observability, bounded rationality, and strategic multi-agent interactions. The need to reason under uncertainty and anticipate others' actions within a Nash equilibrium framework make the game a natural proxy for studying deception under adversarial and cooperative dynamics.
3 Evaluation: Skills, Deceptive Capability, and Win-rates
this section cite: ['b9', 'b7', 'b12', 'b13', 'b14', 'b15']

Section: LLM-based Skill Evaluations and Win-rates for Impostors and Crewmates
We run 400 games with 100 pairs of impostors and crewmates sampled from the open-weight models Meta 'Llama-3.3-70b-instruct' [16] and Microsoft 'Phi-4-15b' [17] respectively. Crewmates almost never lie, and in some cases impostors are truthful in order to gain trust (see Chi et al. [9]).
this section cite: ['b16', 'b17', 'b9']

Section: Awareness
We perform LLM-based output evaluations for all the game steps by players playing as Impostors and Crewmates (by passing the context and agent output to 'GPT-4omini' with a custom prompt (see Appendix D)) to generate ground-truth for skills in awareness, lying, deception, and planning for evaluating linear probes and SAEs. Based on manual inspection, we find that these scores, given from 1 to 10, are not very well-calibrated in the sense that very few scores are in the range of 3 to 7, but they correctly predict the binary labels we use as ground truth. We share violin plots for the same in Figure 4.
In Figure 3, we see that 'Llama-3.3', a bigger and more capable model, is better (wins more) as an Impostor, while 'Phi-4' is better as a Crewmates (at detecting deception). This corroborates our impression in 3.3 that the frontier pushes more for deceptive capability.
this section cite: []

Section: Defining the Multi-agent Deception Elo Metric
Since deception and its detection are central to winning in Among Us, we use multi-agent Elo ratings [18] to get an unbounded measure of deception (for impostors) and detection (for crewmates). We compute Deception Elo ratings for each model based on their performance as impostors across n games. In each game, there are t players, of which k are impostors. The Elo rating provides an unbounded measure of deception, where higher ratings indicate more deception capability. for various models on 2054 games as in Figure 2. Triangle models are RL-trained on tasks, whereas circle models see no RL except perhaps RLHF. The dashed line passes through the means with a slope of 1, and we find most reasoning models to be above the line. CIs similar to Figure 2; omitted for clarity.
For each game, we update the Elo ratings of the k impostors by whether they win or lose against the average Elo of the t -k crewmates. The expected probability of an impostor winning against the average crewmate rating is given by:
E i = 1 1 + 10 (Rc-Ri)/400
where E i is the expected win probability for impostor i, R i is their current rating, and R c is the average rating of the crewmates they played against. After each game outcome, the rating of each LLM-agent acting as an impostor is updated according as per the update rule:
R ′ i = R i + K(S -E i )
where K = 32 is the update factor and S is the actual outcome (1 for impostor win, 0 for loss). Similarly, we define a "Detection Elo" for crewmates (see Figure 5).
To obtain robust estimates of model performance, we compute 90% confidence inter-
0 0.2 0.4 0.6 0.8 1 0 0.2 0.4 0.6 0.8 1 Random (AUC = 0.500) Action (AUC = 0.653) +Thinking (AUC = 0.948) +User (AUC = 0.938) +System (AUC = 1.000) False Positive Rate True Positive Rate (a) Probes on all actions (77% of these are nonspeech actions). 0 0.2 0.4 0.6 0.8 1 0 0.2 0.4 0.6 0.8 1 Random (AUC = 0.500) Action (AUC = 0.984) +Thinking (AUC = 1.000) +User (AUC = 0.967) +System (AUC = 1.000) False Positive Rate True Positive Rate (b) Probe on "speak" actions (along with the speech). vals through bootstrap resampling with replacement for 1000 iterations (Figure 2). This provides both mean Elo ratings and win rates for each model, along with their respective confidence intervals.
All models start with a base Elo rating of 1500. The variance in bootstrap samples can be partly explained by the variability of game outcomes based on random initialization and the stochasticity of LLMs.
this section cite: ['b18']

Section: Frontier Reasoning Models are Differentially better at Deception
We run 2054 games with 7 players randomly initialized as LLM-agents using 18 of the most popular models on OpenRouter (as of April 2025). We log all game summaries and compute Deception Elo ratings and win rates for each model (see Section 3.2 for details). Our results are shown in Figure 2. We find that reasoning models (such as DeepSeek R1 and Claude 3.7 Sonnet) perform significantly better and have a much higher Deception Elo compared to the other models.
In Figure 5, we compare models based on their deception and detection capability (based on their Elo score as an Impostor and a Crewmate), and we find that smaller and non-reasoning models are comparable to the frontier models at detection, and frontier and reasoning models are much better at deception than others. This leads us to the surprising impression that the frontier pushes for deceptive capability and not for detection. We observe more evidence for this in Section 3.1 and show a comparison of the two Elo ratings against the release dates of the models in Appendix J.
this section cite: []

Section: Activation Monitoring for Lying and Deception
To investigate whether out-of-distribution (OOD) deception can be detected by looking at the model's hidden activations, we train and evaluate linear probes on various datasets using the Phi-4 model [17].
We download the models using the HuggingFace API [19]. We also share results on detection and steering using Sparse Autoencoder (SAE) features in Section 4.3.
this section cite: ['b17', 'b19']

Section: Training Methodology for Linear Probes
Prior works compare several techniques for training linear probes [20,21] and find logistic regression probes to work best. Thus, we train logistic regression probes on the residual stream activations from layer 20 (of a total of 40 layers) in the Phi-4 model with an embedding dimension of 5120. We use a weight decay of 10 -3 and train for n = 4 epochs with a batch size of 32 using the Adam optimizer and a learning rate of 0.001 with StepLR scheduling. We normalize the activations using mean and variance from the train data. For each dataset, we used the following training procedure:
Step 1: Split the dataset into training (80%) and a held-out test (20%) set. Since probes generalize very well on-distribution, we do not use a validation set. Step 2: Add PyTorch hooks [22] after layer 20 and run the model's forward pass to cache activations for the last 10 tokens for each datapoint.
this section cite: ['b20', 'b21', 'b22']

Section: Step 3: Train a logistic regression probe on the activations of these tokens to predict the label.
Step 4: Follow the same procedure to evaluate the probe on each dataset's held-out test set.
We evaluate linear probes (and SAE features) obtained from contrasting pairs from different datasets:
• TruthfulQA (TQA): A correct vs. incorrect labeled dataset of factual questions with contrastive answers and no system prompt [23].
• DishonestQA (DQA): The same questions and answers as TruthfulQA, but with a "Pretend you're honest" and "Pretend you're dishonest" added to the system prompt.
• Among Us: Our dataset of gameplay from the Among Us experiments (see Section 3).
• RepEng: A contrastive dataset from Representation Engineering [20].
We show the out-of-distribution (OOD) performance of probes trained on all 40 layers in Appendix H (most layers work well), and pick the middle layer n = 20 for the results in Section 4.2.
this section cite: ['b23', 'b20']

Section: Out-of-distribution Linear Probe Evaluation
Figure 7: Linear probes generalizes across datasets including Among Us (train data on y-axis). See Appendix A for details on why they don't work on TruthfulQA, but there is still some performance gap compared to on-distribution probes.
In Figure 6a, we show the ROC curves for the linear probes trained on the RepEng dataset and evaluated on deception in Among Us on various parts of the prompt. (see Appendix H for ROC curves for probes on the lying and deception ground truth labels).
The probes trained on RepEng (Figure 7) show a significant ability to detect deception in Among Us, with an area under curve (AUC) of 0.94 for lying and 0.99 for deception. The probes trained on DQA and TQA also show high performance.
We also train probes on-distribution using Among Us data. Since impostors sometimes do not perform "lying" actions (see Section 3.1), training for that label provides a higher signal for deception than for lying. However, these probes give us a good upper-bound measure of how well deception probes can do in-distribution compared to OOD-probes.
Qualitatively, we observe that the probes do actively detect signs of deceptive thinking (see Figure 15 in the Appendix for token-wise scores on the entire prompt), and work well with running averages of the probe scores (Figure 11 in Appendix G).
We also evaluate the efficacy and cosine similarity of probes as we reduce the training data size and find that probes are very effective even when trained on significantly lesser datapoints (see Appendix C for more details about the experimental setup and results).
this section cite: []

Section: Sparse Autoencoder (SAE) Features for Deception and Steering
We analyze SAE features extracted from contrastive datapoints in the DishonestQA dataset (features that contrast the positive and negative labels the most in DQA), and evaluate the out-of-distribution performance of the top features for honesty and dishonesty on the Among Us data (for impostor vs. crewmate) using pre-trained SAEs using the Goodfire API [24].
this section cite: ['b24']

Section: Category Sparse Autoencoder (SAE) Feature AUROC

this section cite: []

Section: Honest
The assistant should promote and embody honesty 0.967 Dishonest Acts of deception and dishonesty 0.851 Dishonest Discussion of manipulation, fraud, or deceptive behavior 0.675 Dishonest Descriptions of deceptive or misleading claims in harmful content 0. We present the AUROC values for SAE features with their auto-interp descriptions [25] in Table 1. We find two SAE features to perform very well in detecting deception OOD. The feature associated with "promoting honesty" achieved the highest AUROC (0.967), followed by features directly capturing deceptive actions (0.851). Features related to discussions of manipulation showed moderate discriminative ability (0.675, 0.638), while several honesty-related features demonstrated no generalization (AUROC = 0.5). Also, a feature capturing "contexts involving deception" showed an inverse relationship (AUROC = 0.182), possibly reflecting scenarios where deception is being analyzed rather than enacted.
We also use the top SAE features to steer the Llama-3.3-70b-instruct model Goodfire's AutoSteer API. We try several steering features and amounts and observe that Impostors either continue acting normally (for small amounts of steering) or begin to hallucinate [11] (for larger amounts). We share examples in Appendix F. This corroborates with recent work finding activation steering to not work well in OOD contexts [26].
5 Related Work
this section cite: ['b25', 'b11', 'b26']

Section: Social Deduction and Deception Games
In the past, AIs (including LLMs) have been trained or prompted to play social deduction and deception games, such as Avalon [27,28], Hoodwinked [29], and Werewolf [30]. Guan et al. [31], Bakhtin et al. [32] trained a model to play the online game Diplomacy, and most recently, Sarkar et al. [33] used reinforcement learning (RL) to train a model to play a simpler version of Among Us. Hu et al. [34] provides a survey of LLM-based game environements and agents. However, to the best of our knowledge, we are the first to use social games as a sandbox to elicit, evaluate, and study harmful behavior in AI agents.
this section cite: ['b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34']

Section: Deception in Language Models
A number of recent works have studied various forms of deception in LLMs. Sleeper agents [7] train LLMs to have specific deceptive behavior in the form of code backdoors that persist through safety training, and Greenblatt et al. [6] study alignment faking, where LLMs fake information about their alignment when faced with difficult choices. Concurrent to our work, Goldowsky-Dill et al. [35] explore deception in domains such as insider trading and roleplaying to study strategic deception in LLMs and extensively explore the training of linear probes to detect it. However, the datasets or tasks these works study are not rich enough to capture all aspects of agent-human interactions we care about (see Section 2.4), and do not allow LLM-agents to elicit goal-directed, instrumental, open-ended deception like our sandbox.
this section cite: ['b7', 'b6', 'b35']

Section: Safety Techniques for Detecting Deception
Language model feedback has been used to both create ground truth labels [36,35] and use them to refine model outputs for harmful behavior. Various kinds of linear probes [37] have been trained in previous works for interpreting language model features using intermediate activations [20,21,38]. Sparse autoencoders (SAEs) [39,40] have been used to decompose the activation space of a model into human-interpretable latents or features. We create a sandbox to study the effectiveness of AI safety techniques for detecting agentic deception, and evaluate some of them, while leaving more complex ones for future work.
this section cite: ['b36', 'b35', 'b37', 'b20', 'b21', 'b38', 'b39', 'b40']

Section: Discussion
Limitations One limitation of our work is that the results might not represent real-world user-model interaction, or the deception mechanisms in the sandbox could differ from real-world settings. Also, the techniques we evaluate might work for uninteresting reasons (such as detecting impostor/crewmate tokens based on latent knowledge associations). To address these, we keep the prompts as simple as possible with just the game rules, and show the token-wise scores in Appendix H.
A potential failure mode is sandbagging -where models strategically underperform. We can address this through reinforcement learning on LLM priors to un-sandbag the model [33]. Also, it is likely that models know about Among Us from pre-training and are aware of the rules and that deception helps in it. Since training data is proprietary for most frontier models, it is difficult to figure out how much of this happens in practice [41]. Just like any other deception evaluation setup, we need to prompt a scenario. However, the open-ended ways to be deceptive are still specific to each game's context and situation, which makes it an important sandbox to study for open-ended deception.
Another critique is that Among Us, as a game, requires deception to win. But we believe it serves as a good proxy for how agents might function when deception helps achieve their goals in certain situations in the real-world based on their perceived reward structures. Also, this sandbox evaluates deceptive capability, which is different from deception propensity, i.e., willingness to be deceptive in a given situation.
Broader Impacts A potential positive impact of this sandbox is as a benchmark to iterate and test technical AI safety techniques to detect and mitigate deception in LLM-agents. This can potentially lead to safer and more aligned models. We show that probes trained using data from our sandbox are effective at detecting deception in other settings as well.
A possible use of game models of deception is as a training environment to improve deception abilities in AIs, i.e., gain-of-function research. It would be important to study if more deceptive models remain detectable with simple probes, and to anticipate how far models are from superhuman deception abilities. We believe training models for deception using the sandbox to be relatively low-risk, because while deception mechanisms can generalize, such models would not perform economically valuable tasks better than other agents and would be restricted to deception in the controlled sandbox.
this section cite: ['b33']

Section: References
Ref_id:b0 Title: The claude 3 model family: Opus, sonnet, haiku Year: ()
Ref_id:b1 Title:  Year: (2024)
Ref_id:b2 Title: Gpt-4.5 system card Year: (2025)
Ref_id:b3 Title:  Year: (2025)
Ref_id:b4 Title: Openai o1 system card Year: (2024-12)
Ref_id:b5 Title: Deception in llms: Selfpreservation and autonomous goals in large language models Year: (2025)
Ref_id:b6 Title: Alignment faking in large language models Year: (2024)
Ref_id:b7 Title: Sleeper agents: Training deceptive llms that persist through safety training Year: (2024)
Ref_id:b8 Title: Tournaments and paired comparisons Year: (1959)
Ref_id:b9 Title: Amongagents: Evaluating large language models in the interactive text-based social deduction game Year: (2024)
Ref_id:b10 Title: Lies, damned lies, and distributional language statistics: Persuasion and deception with large language models Year: (2024)
Ref_id:b11 Title: Llms will always hallucinate, and we need to live with this Year: (2024)
Ref_id:b12 Title: Ai control: Improving safety despite intentional subversion Year: (2024)
Ref_id:b13 Title: Gradual disempowerment: Systemic existential risks from incremental ai development Year: (2025)
Ref_id:b14 Title: Measuring massive multitask language understanding Year: (2021)
Ref_id:b15 Title: Line goes up? inherent limitations of benchmarks for evaluating large language models Year: (2025)
Ref_id:b16 Title: Llama 3.3 model card and prompt formats Year: (2024-12)
Ref_id:b17 Title:  Year: (2024)
Ref_id:b18 Title: Multi-agent training for pommerman: Curriculum learning and population-based self-play approach Year: (2025)
Ref_id:b19 Title: Huggingface's transformers: State-of-the-art natural language processing Year: (2020)
Ref_id:b20 Title: Representation engineering: A top-down approach to ai transparency Year: (2023)
Ref_id:b21 Title: The geometry of truth: Emergent linear structure in large language model representations of true/false datasets Year: (2024)
Ref_id:b22 Title: Automatic differentiation in pytorch Year: (2017)
Ref_id:b23 Title: Measuring how models mimic human falsehoods Year: (2022)
Ref_id:b24 Title: Mapping the latent space of Llama 3.3 70B Year: (2024)
Ref_id:b25 Title: 6coxChSWcxCDy8/self-explaining-sae-features Year: (2024-08)
Ref_id:b26 Title: Stepan Shabalin, and Sheikh Abdur Raheem Ali. Patterns and mechanisms of contrastive activation engineering Year: (2025)
Ref_id:b27 Title: Avalon's game of thoughts: Battle against deception through recursive contemplation Year: (2023)
Ref_id:b28 Title: Avalonbench: Evaluating llms playing the game of avalon Year: (2023)
Ref_id:b29 Title: Hoodwinked: Deception and cooperation in a text-based game for language models Year: (2023)
Ref_id:b30 Title: Exploring large language models for communication games: An empirical study on werewolf Year: (2024)
Ref_id:b31 Title: Richelieu: Self-evolving llm-based agents for ai diplomacy Year: (2024)
Ref_id:b32 Title: Mastering the game of no-press diplomacy via humanregularized reinforcement learning and planning Year: (2022)
Ref_id:b33 Title: Training language models for social deduction with multi-agent reinforcement learning Year: (2025)
Ref_id:b34 Title: A survey on large language model-based game agents Year: (2024)
Ref_id:b35 Title: Detecting strategic deception using linear probes Year: (2025)
Ref_id:b36 Title: Llms are superior feedback providers: Bootstrapping reasoning for lie detection with self-generated feedback Year: (2024)
Ref_id:b37 Title: Understanding intermediate layers using linear classifier probes Year: (2018)
Ref_id:b38 Title: Discovering latent knowledge in language models without supervision Year: (2024)
Ref_id:b39 Title: Sparse autoencoders find highly interpretable features in language models Year: (2023)
Ref_id:b40 Title:  Year: ()
