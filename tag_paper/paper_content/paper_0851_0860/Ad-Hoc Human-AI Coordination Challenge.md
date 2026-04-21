Title: Ad-Hoc Human-AI Coordination Challenge
Abstract: Achieving seamless coordination between AI agents and humans is crucial for real-world applications, yet it remains a significant open challenge. Hanabi is a cooperative card game featuring imperfect information, constrained communication, theory of mind requirements, and coordinated action -making it an ideal testbed for human-AI coordination. However, its use for human-AI interaction has been limited by the challenges of human evaluation. In this work, we introduce the Ad-Hoc Human-AI Coordination Challenge (AH2AC2) to overcome the constraints of costly and difficult-to-reproduce human evaluations. We develop human proxy agents on a large-scale human dataset that serve as robust, cheap, and reproducible human-like evaluation partners in AH2AC2. To encourage the development of data-efficient methods, we opensource a dataset of 3,079 games, deliberately limiting the amount of available human gameplay data. We present baseline results for both two-and three-player Hanabi scenarios. To ensure fair evaluation, we host the proxy agents through a controlled evaluation system rather than releasing them publicly. The code is available at https://github.com/FLAIROx/ah2ac2.

Section: Introduction
Human-AI interaction is rapidly advancing due to significant AI progress and its growing integration into daily life (Rawas, 2024). Effective coordination between humans and AI in complex settings becomes crucial as AI agents become more sophisticated, capable, and prevalent. This interaction spans a wide range of domains, from collaborative decision-making in healthcare (Asan et al., 2020) to shared control in autonomous vehicles (Bansal et al., 2018) and robotics (Haarnoja et al., 2018;Ahn et al., 2024) as well as advanced digital assistants (Gemini Team, 2024; Bai et al., 2022). The ultimate goal is to create AI agents that are not limited to solving problems independently but can also work effectively with humans to complete tasks in human-compatible ways (Russell, 2019;Carroll et al., 2019).
Traditional approaches to training AI agents in simulation frequently employ self-play (SP) where agents train under a joint policy that controls the strategies of all players (Samuel, 1959;Tesauro, 1994). While SP has been successful in competitive games like chess and Go (Silver et al., 2016), in cooperative settings this approach can lead to agents that overfit to specific strategies, limiting their ability to generalise to novel partners (Carroll et al., 2019). In human-AI coordination scenarios, forming rigid conventions is particularly problematic, as doing so can pose safety risks where humans are unable to adapt appropriately (Bard et al., 2019).
Despite the growing importance of human-AI coordination, the field lacks standardised benchmarks that accurately reflect the complexities of interacting with humans in complex, partially observable settings. Existing evaluation methods often rely on closed datasets and proprietary proxy agents (Bakhtin et al., 2022;Jacob et al., 2022), which hinder reproducibility and broad-based progress. Without accessible and robust benchmarks, it is challenging to measure advancements consistently. Furthermore, the scarcity of opensource human-AI coordination datasets limits the ability of researchers to develop and test innovative, data-efficient algorithms that are essential for real-world applications where large-scale human data may not be readily available.
Hanabi is an established, fully cooperative benchmark environment that involves imperfect information, limited communication, theory of mind, and the necessity for coordination among different players to achieve a shared goal (Bard et al., 2019). These characteristics make Hanabi a solid testbed for evaluating human-AI coordination. While previous research in Hanabi has used held-out sets of human data and human proxy agents to evaluate human-AI coordination (Hu et al., 2022;2021;Lupu et al., 2021), these datasets and proxy agents have thus far remained closed-source and un-available to the wider research community. To address these problems, we introduce the Ad-Hoc Human-AI Coordination Challenge (AH2AC2) as a standardised way to evaluate human-AI coordination in Hanabi. Specifically, we develop human proxy agents through a combination of behavioural cloning (BC) and regularised reinforcement learning (RL). We first train the BC component on a large-scale dataset of human gameplay, comprising 101,096 two-player games and 46,525 three-player games from the hanab.live community. We then refine the BC policy using Independent Proximal Policy Optimisation (IPPO) (de Witt et al., 2020; Schulman et al., 2017) with a regularisation term that encourages adherence to human-style play (Bakhtin et al., 2022;Hu et al., 2022;Cornelisse & Vinitsky, 2024). Our empirical evaluation shows that these human proxy agents outperform pure imitation learning while maintaining human-like behaviour. These human proxy agents serve as robust, cheap and reproducible evaluation partners in AH2AC2.
To ensure AH2AC2 evaluation integrity and to prevent overfitting by the community, we withhold public access to the human proxy agents and their large-scale training dataset, and only open-source a limited dataset for both two-and three-player settings. This also encourages research on methods that are data-efficient with respect to human data. We also provide various baselines: first, zero-shot coordination methods such as Off-Belief Learning (OBL) (Hu et al., 2021), which operate without human data; second, datadependent approaches such as best response to behavioural cloning policy (BR-BC) (Carroll et al., 2019); third, the first evaluation of Fictitious Co-Play (FCP) (Strouse et al., 2021) in Hanabi, which serves as a baseline for population-based methods; fourth, we develop a DeepSeek-R1 (DeepSeek-AI et al., 2025) Hanabi agent, to provide insights into the current capabilities of Large Language Models (LLMs) for human-AI coordination.
In summary, our key contributions are:
• The first open-source Hanabi human gameplay dataset, containing 1,858 two-player and 1,221 three-player games.
• High-performing human proxy agents for both two-player and three-player Hanabi settings, using BC on a closedsource large-scale human play dataset (over 100k games) combined with regularised RL.
• Our evaluation protocol, where we host proxy agents behind an API, paired with a public leaderboard to track research progress. API access requires the pre-registration of experiments, a gold standard for empirical research.
• A comprehensive set of baselines for both two-and threeplayer settings, spanning zero-shot coordination, datadependent, and population-based methods, alongside the DeepSeek-R1 Hanabi agent, which specifically provides an assessment of LLM performance in human-AI coor-dination; these baselines collectively highlight the significant difficulty current methods face in building humancompatible agents for partially observable environments.
this section cite: ['b29', 'b1', 'b4', 'b14', 'b0', 'b2', 'b30', 'b7', 'b32', 'b37', 'b34', 'b7', 'b5', 'b3', 'b19', 'b5', 'b18', 'b25', 'b33', 'b3', 'b18', 'b9', 'b17', 'b7', 'b36']

Section: Background
Dec-POMDP We consider a decentralised partially observable Markov decision process (Dec-POMDP) (Oliehoek, 2012), defined as a 9-tuple (S, n, {A i } n i=1 , {O i } n i=1 , T , R, {U i } n i=1 , H, γ). S is the finite state space and n is the number of agents. A i and O i are the local action and observation spaces for agent i, and A := × n j=1 A i , O := × n i=1 O i are the joint action and observation spaces. The transition function T : S × A × S → [0, 1] defines the probability of transitioning to state s t+1 when taking joint action a t = (a 1 t , ..., a n t ) in state s t . The agents receive the reward r t+1 = R(s t+1 , a t ), and agent i receives the local observation o i t+1 with probability U i (s t+1 , a t , o i t+1 ). γ ∈ [0, 1] is the discount factor, and H is the horizon. i.e. s H is always a terminal state.
The local action-observation history (AOH) of player i is defined as τ i t = (a i 0 , o i 1 . . . , o i t-1 , a i t-1 , o i t ), and the joint AOH is defined as τ = (τ 1 , ..., τ n ). Each player i selects a local action a i t according to a local policy π i (a i t |τ i t ). The joint policy π = (π 1 , ..., π n ) then selects joint action a t with probability π(a t |τ t ) = n j=1 π i (a i t |τ i t ). Given a joint policy π, the expected return is defined as J(π) = E π H-1 t=0 γ t r t+1 .
this section cite: []

Section: Zero-shot Coordination and Ad-Hoc Teamplay
In many real-world scenarios, AI agents must coordinate with unseen partners, including humans. Traditional cooperative multi-agent RL uses SP training, where agents train together to maximize the expected return (Bard et al., 2019;Carroll et al., 2019). However, SP training often leads to specialized communication protocols that fail with independently trained agents, including humans. Thus, training in SP is not a solution to the challenges of human-AI coordination (Carroll et al., 2019;Strouse et al., 2021). Zero-shot coordination (ZSC) addresses this by training agents to collaborate effectively with new partners which were trained with the same algorithm (Hu et al., 2020;Treutlein et al., 2021).
Ad-hoc teamplay assesses an agent's ability to cooperate with unfamiliar teammates at test time (Stone et al., 2010). Like ZSC, agents are evaluated with partners they have not trained with, but unlike ZSC, teammates may use different training algorithms. In this work, we focus on ad-hoc teamplay involving human and human-like agents.
Hanabi Hanabi is a cooperative card game where players can see the cards in each other's hands, but not in their own, and thus rely on others to give interpretable hints on which cards to play or discard. It is designed for 2-5 players, but we restrict ourselves to two-and three-player settings in this work due to the availability of human gameplay data for these configurations. The maximum score in the game is 25, but it cannot be achieved for every shuffling of the deck. If the team of players makes three mistakes in total, the game ends with a score of 0. For more details on Hanabi we refer to Appendix A.1.
this section cite: ['b5', 'b7', 'b7', 'b36', 'b16', 'b38', 'b35']

Section: Related Work
Hanabi was introduced as a benchmark encompassing both SP and ad-hoc teamplay (Bard et al., 2019), but significant progress has been primarily confined to SP, with methods like SPARTA (Lerer et al., 2019) achieving near-perfect 24.61/25 points on average. However, agents trained in SP often rely on specialised conventions, leading to poor generalisation when paired with novel teammates. Consequently, ad-hoc team play, especially with humans, presents a more demanding and unsolved challenge.
A central issue when evaluating agent abilities for ad-hoc coordination is the selection of test-time partners. One approach to tackle this issue is to ensure the diversity of policies, thereby minimising the possibility of favouring any specific agent. For example, Cui et al. (2023) introduce ADVERSITY that aims to produce highly skilled and reasonable policies that play according to diverse conventions. Another approach is to focus on a set of policies that hold intrinsic value, with the most natural choice being human policies. Coordinating with humans can be seen as a specialised form of ad-hoc teamplay, where the set of test policies comprises human strategies, which are inherently valuable due to their real-world relevance.
Ad-hoc human-AI coordination in Hanabi is explored in many previous works, with each presenting a different methodology for acquiring human proxy agents and evaluating human-AI coordination capabilities (Hu et al., 2021; 2020; Cui et al., 2023). However, there is no standard approach for ad-hoc human-AI coordination evaluation in the existing literature. Therefore, our work addresses this gap and proposes the AH2AC2.
Recent works have empirically shown that augmenting imitation learning methods with regularised RL creates stronger and more reliable policies. Multiple works have shown that regularised RL leads to policies that are more compatible with existing social conventions of the human reference group (Jacob et al., 2022;Bakhtin et al., 2022;Hu et al., 2022). Recently, (Cornelisse & Vinitsky, 2024) extended these works to the driving setting, where the authors show that data-driven regularisation leads to human-compatible policies. Our work builds upon these findings to build strong and human-like human proxy agents that enable AH2AC2.
Nekoei et al. (2023) urge the multi-agent RL community to address few-shot adaptation alongside zero-shot coordination (ZSC). They show that current ZSC algorithms struggle to adapt to new partners. To connect AH2AC2 with few-shot coordination, we introduce data-limited settings. Unlike traditional few-shot scenarios, our approach involves offline, one-sided adaptation: human proxy agents maintain fixed behaviour during testing, and we use a small sample of human play data for training. This emphasizes the challenge of coordinating with human-like agents, where only the AI agent adapts due to the constraints of human behaviour (Stone et al., 2010).
this section cite: ['b5', 'b22', 'b19', 'b3', 'b18', 'b35']

Section: Ad-Hoc Human-AI Coordination Challenge (AH2AC2)
This section outlines our proposed challenge, which consists of two key evaluation regimes: (a) evaluation with a set of human proxy agents. This requires participants to develop human-compatible agents based on a small provided dataset of human gameplay, and (b) a human action prediction task on an unseen, closed-source set of games.
this section cite: []

Section: Methodology Overview
We collected a comprehensive dataset from the hanab.live platform, comprising of 101,096 two-player games and 46,525 three-player games. All games adhere to H-group conventions, a set of hand-crafted strategies used by Hanabi players on hanab.live. It is important to note that H-group Conventions are not a single strategy. Instead, it is helpful to think of H-Group Conventions as a collection of different strategies and techniques that players learn and combine. Players often mix and adapt these strategies within a single game, depending on the players' strength. Therefore, the dataset itself naturally contains a variety of playstyles. Therefore, the use of H-Group Conventions as a foundation does not overly constrain the strategic variety of our human proxies. Details about data composition and splits are provided in Appendices A.5 and A.4.
As a part of the AH2AC2, we open source 3,079 games from the large-scale dataset -1,858 two-player and 1,221 threeplayer games. Participants are allowed to use these opensourced games when tackling the AH2AC2. Key statistics of the open-sourced dataset are summarised in Table 1.
Using the entire dataset we develop human-proxy agents that act as standard and cheap test partners for ad-hoc human-AI coordination evaluation. To prevent overfitting, we host the human proxies behind an API instead of releasing them publicly. Participants have to pre-register an evaluation which gives them access to to 1,000 evaluation games with our human proxies. This controlled access ensures consistency across submission and pre-registration of experiments is the gold standard for empirical science. The candidate agent's performance is evaluated based on the mean and median scores achieved across 1,000 games with human proxies and will be published on our leaderboard.
In the second (optional) part of the challenge, we also assess the agent's ability to predict human actions in an unseen set of human-played games. We evaluate performance using the teacher-forced cross-entropy loss.
this section cite: []

Section: Evaluation Protocol

this section cite: []

Section: PART 1 OF AH2AC2: COORDINATION WITH HUMAN PROXIES
We develop four human proxy agents for evaluating adhoc human-AI coordination: two for two-player Hanabi and two for three-player.
These agents are trained using Human-Data-Regularised IPPO (HDR-IPPO), a procedure combining BC and regularised IPPO (de Witt et al., 2020). First, BC policies are trained on a large-scale dataset of human gameplay -101,096 two-player games and 46,525 three-player games. Because BC alone struggles to generalise to unseen game states (Carroll et al., 2019; Hu et al., 2022; Bakhtin et al., 2022; Cornelisse & Vinitsky, 2024), we then refine them through regularised SP using IPPO. The regularisation ensures the final policies remain close to human play styles. We provide further details regarding PPO and IPPO in Appendix A.2.
When training human proxy agents using HDR-IPPO, we learn a parameterised local policy, denoted as π HP θ . As a first step, we train a local BC policy π BC θ , which, given Hanabi's discrete action space, translates into a classification task: the features are local AOHs τ i t , and the labels are ground truth local actions a i t+1 given. To capture the sequential nature of the actions and observations, we model π BC θ through an LSTM-based architecture (Hochreiter & Schmidhuber, 1997). Critically, fixed neural parameters do not imply static behaviour; our proxies condition on the history of the game, which includes partner actions. When the proxy agents see an unexpected action or observation, from that action-observation history onward, they will account for the fact that the other agent is using a different convention. We train the BC model through supervised learning, minimising the standard cross-entropy loss between the predicted action distribution and the ground truth human actions.
At the end of each training epoch, we compute the average SP score of π BC θ over 5000 games and store the parameters θ ′ , that yield the highest average SP score. During each of those SP evaluations, each agent, at every timestep, selects the local action with the highest predicted probability according to the local policy, i.e. a i t = arg max a π BC θ (a i |τ i t ). In the second step of the HDR-IPPO method we leverage the baseline BC policy, π BC θ ′ , to guide the training of a more robust policy, π HP θ . First, we initialize the weights of π HP θ to θ ′ . Next, to encourage the final policy to remain close to the human-like behaviour exhibited by π BC θ ′ , we introduce the KL (Kullback & Leibler, 1951)  Our human proxies serve as standardised, robust and cheap partners for human-AI coordination evaluation. In the context of AH2AC2, human proxies play a pivotal role in benchmarking AI performance against human-like behaviour. This setup not only facilitates the assessment of AI adaptability and robustness but also ensures that evaluations are scalable and reproducible.
Additionally, we present an ablation study to examine the impact of the HDR-IPPO KL regularisation term in Appendix A.8. This analysis explores the effects of varying regularisation strength on the learned policies, offering insights into the role of this component. We defer this discussion to the appendix to maintain focus on the properties of the developed human proxy agents within the main text.
this section cite: ['b15', 'b21']

Section: PART 2 OF AH2AC2: ACTION PREDICTION CHALLENGE
Beyond the primary human-AI ad-hoc coordination challenge, we introduce an action prediction task. Although human-compatible play does not strictly ensure accurate action prediction, successfully predicting human actions further demonstrates human-like behaviour. The action prediction dataset consists of a held-out portion of the human data used to train the human proxies.
We evaluate performance using the teacher-forced crossentropy loss since we aim to quantify the difference between the predicted action distribution and the true human actions. Lower cross-entropy loss indicates better alignment with human decision-making. Participating agents receive a local action-observation history and must predict the action taken by the human player at each timestep.
this section cite: []

Section: Evaluation API and Leaderboard
To facilitate participation and ensure fair evaluation, we host the human proxy agents and provide access through a dedicated evaluation API.
We have established a dedicated website for the AH2AC2 challenge at https://ah2ac2.com/. To initiate the evaluation process, participants must fill out a form to register for access to the evaluation phase. Once registered, we provide participants with a private key, which grants restricted access to our human proxies. This key allows a one-time evaluation run, strictly limited to 1,000 games. Once the evaluation with human proxies is finished, participants get limited access to the test dataset through the API we provide. Upon completion, results are published on the challenge leaderboard. Furthermore, our evaluation API allows interaction with the proxy agents while restricting access to global game state information, enforcing the partial observability inherent to Hanabi.
this section cite: []

Section: Analysing and Validating the Human-Proxies
This section is organised as follows. First, we evaluate the self-play performance of our human proxy agents, demonstrating significant improvements compared to the BC policies. Second, we validate the human-likeness of our human proxy (HP) agents. This is done through cross-play experiments with BC policies, by an evaluation of the action prediction performance of the HP policies on held-out datasets of human gameplay, and with an analysis of behavioural metrics.
this section cite: []

Section: Self-Play Scores of Human Proxy Agents
Table 2 presents the SP scores of our final human proxy agents and their improvement over the initial BC policies. The performance gains from regularised RL are particularly pronounced in the three-player setting, where the BC policies trained on limited data frequently lose all their lives, resulting in a large proportion of zero-score games and overall bad performance. For instance, in a three-player SP evaluation, BC agents scored zero in 70.92% of games. With the help of regularised RL, human proxies score zero points only on 0.27% of the games. This highlights the robustness achieved through regularised RL. Moreover, we observe a larger number of perfect-score games for all agents, compared to BC counterparts.
Furthermore, Figure 2a and 2b illustrate the cross-play results for two-player and three-player setting human proxy agents. We observe consistent scores across different pairings, suggesting that the agents have converged to compatible strategies despite variations in their architectures and regularisation strengths.
this section cite: []

Section: Validating Human-Likeness of Human Proxy Agents

this section cite: []

Section: Cross-Play between Human Proxies and BC Policies.
Behavioural cloning (BC) policies are closely aligned with human conventions but lack generalization capabilities. However, we anticipate that when BC policies are paired with human proxies, the resulting cross-play scores will be substantially higher than the BC policies' SP scores. This  expectation is confirmed in Figures 2a and 2b. By comparing median and mean scores, we observe that agents either coordinate exceptionally well or fail completely. These results suggest that HDR-IPPO agents have not only learned to play effectively amongst each other but have also retained strategies employed by BC agents (i.e. agents trained solely on human data).
In the three-player setting, mean scores are significantly lower than median scores, especially when two BC policies team with one human proxy. This suggests many zeroscore games and highlights the brittleness of BC policies in unfamiliar scenarios. Despite this, median scores remain high despite BC policies' poor single-player performance.
When two BC policies are paired with a stronger human proxy, median scores increase substantially, reaching 17 or higher in all configurations.
These cross-play experiments provide evidence that human proxy agents have successfully learned to play the game at a high level while maintaining the ability to interact effectively with agents that utilize strategies derived purely from human demonstrations.
this section cite: []

Section: Action Prediction Performance of Human Proxies.
Next, we evaluate the action prediction performance of human proxies using a test set that was excluded from the training of the BC agents. The results, reported in Table 3, demonstrate that the human proxies achieve similar accuracy and loss metrics on the test sets as the BC policies. In many Hanabi game scenarios, multiple human-like actions are possible. We thus report the Top-10% and Top-20% accuracies, which represent the probability that the ground-truth action is among the top 10%, 20% of most likely actions under the human proxy policies. This metric corresponds to top-2, top-4, and top-3, top-6 accuracies for two-player and three-players, respectively.
this section cite: []

Section: Behaviour Analysis of Human Proxies.
We assess the behaviour against a large-scale human dataset using two metrics from (Canaan et al., 2020): IPP and Communicativeness. IPP (Information per Played Card) measures the information an agent has about each card it plays: for every card played, we track which attributes (colour and/or rank) are known (0, 1, or 2), average these values across all played cards, and normalize by dividing by 2 to obtain a score between 0 and 1. Communicativeness measures the proportion of turns where an agent gives a hint when a hint token is available, quantifying how often an agent chooses to communicate. These metrics were computed from 50,000 human-proxy trajectories in SP and the exten-sive human play dataset. As shown in Table 4, both metrics are nearly identical across human proxies and the human dataset, indicating similar behaviour and strategy. For additional experiments and results, please refer to A.5, A.8 and A.6. We conducted an ablation study by varying the strength of the regularisation term to examine its impact on the final policy. Our findings indicate that initializing training from a BC policy without regularisation leads to weak and/or human-incompatible policies. Finally, we provide further behavioural analysis of our human proxies to illustrate their performance and interactions.
this section cite: ['b6']

Section: Results of Baseline Methods on AH2AC2
In this section, we evaluate several baselines in the AH2AC2 challenge, informed by previous research (Hu et  We use this agent only in the two-player setting since we do not have access to three-player weights. OP (Hu et al., 2020): A ZSC method that prevents agents from learning equivalent but mutually incompatible policies across independent training runs. OP accomplishes this by enforcing the equivariance of the policies under the symmetries of the Dec-POMDP, which must be provided as an input of the algorithm. FCP For each of the BC, BR-BC, and HDR-IPPO baselines, we train with three different random seeds. The best agent, based on cross-entropy loss on the validation set, is selected for evaluation. Notably, we observe minimal performance variance across different seeds on the validation set (see Appendix A.3). The IPPO baseline, intended to showcase the limitations of SP in ad-hoc coordination, is trained with a single seed. Finally, we use pre-trained weights and respective hyperparameters for OBL (Hu et al., 2021).
The training population for the FCP agent comprises 36 random seeds. Since Hanabi presents a greater challenge than Overcooked, the environment in which FCP was initially introduced, we employ four checkpoints for each agent. Consequently, the training process for a single FCP agent utilizes 144 checkpoints. Due to the computational demands involved, we train the FCP agent with a single seed.
For DeepSeek-R1, we evaluate its capability using two prompting variants: one providing the LLM solely with the current game state in natural language, and another that additionally includes a description of H-conventions. Further details on these prompts are available in Appendix A.10.
Table 5 shows the initial AH2AC2 leaderboard. OBL (L4) achieves the highest performance without using any human data. In two-player settings, BR-BC achieves the highest score, while HDR-IPPO leads in three-player settings (although OBL lacks pre-trained weights for three-players, inhibiting its evaluation in this setting). The BC and IPPO baselines perform poorly in both settings, as expected.
These results highlight OBL's effectiveness, surpassing BC, BR-BC, and HDR-IPPO without relying on human data. Methods like OP fail to achieve successful human-AI coordination, and current approaches that leverage limited human data underperform compared to state-of-the-art ZSC algorithms like OBL. This reveals a research gap; existing methods cannot effectively integrate small human datasets to enhance coordination. Additionally, our evaluation of FCP in Hanabi shows it struggles in complex, partially observable environments, suggesting that population-based methods may not provide robust coordination capabilities.
There is thus a need for new techniques that efficiently utilise limited human data to improve human-AI teamwork. Although DeepSeek-R1 demonstrates foundational capability, significant improvements are still needed. In the two-player setting, even when prompted with H-conventions, it significantly underperforms compared to OBL. While providing H-conventions substantially improved its score over the basic prompt (5.43 vs 9.91), this approach still falls considerably short of OBL. In the three-player setting, it shows relatively stronger performance, outperforming all other baselines. Our results suggest that while current LLMs, even without fine-tuning, exhibit some inherent coordination capabilities, they do not yet achieve the desired level of efficacy on AH2AC2. Also, it is crucial to note that, due to resource and time constraints, our evaluation of DeepSeek-R1 was conducted on only 100 evaluation games in contrast to the 1000 games used for all other methods, and we leave the action prediction challenge for future work.
this section cite: ['b17']

Section: Conclusion
We introduced the Ad-Hoc Human-AI Coordination Challenge (AH2AC2), evaluating human-AI ad-hoc teamplay in the context of the cooperative card game Hanabi. By leveraging a large-scale human play dataset to generate human-like agent policies, we provide a meaningful evaluation framework for assessing AI agents' ability to coordinate with human partners. We released a comprehensive set of baselines, including agents trained with and without human data, to provide a reference point for evaluating novel approaches.
Our results highlight the inherent challenge of human-AI coordination. We believe that AH2AC2 represents a significant step forward in the field of human-AI coordination. By providing a standardised evaluation protocol, human proxy agents, and a diverse set of baselines, we aim to foster further research and development. We invite researchers and practitioners alike to participate, pushing the boundaries of what's possible in human-AI collaboration.
Several open challenges and questions remain. We highlight some promising directions for future research:
• Theoretical Analysis of HDR-IPPO: While our experiments and previous works provide strong empirical evidence for the effectiveness of regularised RL in generating human-like agents, a deeper theoretical understanding of the methodology is crucial.
• Generalisation of the benchmark We currently only cover 2 and 3 players as well as "standard" Hanabi, ignoring the large set of possible variations of the game that are created e.g. by the "rainbow cards". Extending the benchmark to cover those scenarios would be a great way to measure the generalisation ability of agentic systems.
• Direct Human-AI Play with Human Proxy Agents:
The ultimate validation of our human proxy agents requires direct human-AI play. Future work should involve conducting play experiments with human participants, comparing their experiences and performance when playing with human proxies versus playing with other humans.
• Comprehensive Evaluation and Advancement of Agentic LLMs: Our preliminary evaluation of DeepSeek-R1, though limited by resource constraints to fewer games than other baselines, provides initial insights into LLM ca- 18.80 19 7.53 † Not constrained by game limits, acts as a golden standard. BR-BC* is trained with a BC policy trained on the entire dataset. We report average performance over two human proxies. pabilities for human-AI coordination. Future work should conduct a more extensive evaluation of LLMs and explore more sophisticated methods. Excitingly, when paired with our human proxies, Hanabi becomes an excellent benchmark for assessing theory of mind in LLMs and their ability to cooperate with humans in complex, partially observable tasks.
Table 6. Cross-entropy loss on the test set for BC, BR-BC and HDR-IPPO agents in the two-player setting. For human proxies we report results over two available agents. For the rest, we report results over three different seeds. Even though agents are trained with different seeds, they achieve almost identical results. We report loss ± SE. to prevent adaptation or overfitting during model development; these serve exclusively for final performance evaluation. The participants of the challenge are not required to use the same data split as defined here, but are encouraged to do so.
this section cite: []

Section: References
Ref_id:b0 Title: Embodied foundation models for large scale orchestration of robotic agents Year: (2024)
Ref_id:b1 Title: Artificial Intelligence and Human Trust in Healthcare: Focus on Clinicians Year: (2020-06)
Ref_id:b2 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b3 Title: Mastering the Game of No-Press Diplomacy via Human-Regularized Reinforcement Learning and Planning Year: (2022)
Ref_id:b4 Title: Learning to Drive by Imitating the Best and Synthesizing the Worst Year: (2018)
Ref_id:b5 Title: The Hanabi Challenge: A New Frontier for AI Research Year: (2019)
Ref_id:b6 Title: Generating and adapting to diverse ad-hoc cooperation agents in hanabi Year: (2020)
Ref_id:b7 Title: On the Utility of Learning about Humans for Human-AI Coordination Year: (2019)
Ref_id:b8 Title: Learning phrase representations using RNN encoder-decoder for statistical machine translation Year: (2014)
Ref_id:b9 Title: Human-compatible driving partners through data-regularized self-play reinforcement learning Year: (2024)
Ref_id:b10 Title: Adversarial Diversity in Hanabi Year: (2023)
Ref_id:b11 Title: Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? Year: (2020)
Ref_id:b12 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b13 Title: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024 Year: ()
Ref_id:b14 Title: Soft Actor-Critic Algorithms and Applications Year: (2018)
Ref_id:b15 Title: Long Short-Term Memory Year: (1997-11)
Ref_id:b16 Title: Other-Play" for Zero-Shot Coordination Year: (2020)
Ref_id:b17 Title: Off-Belief Learning Year: (2021)
Ref_id:b18 Title: Human-AI Coordination via Human-Regularized Search and Learning Year: (2022)
Ref_id:b19 Title: Modeling strong and human-like gameplay with KL-regularized search Year: (2022-07)
Ref_id:b20 Title: A Method for Stochastic Optimization Year: (2014)
Ref_id:b21 Title: On Information and Sufficiency Year: (1951)
Ref_id:b22 Title: Improving Policies via Search in Cooperative Partially Observable Games Year: (2019)
Ref_id:b23 Title: Retrieval-augmented generation for knowledgeintensive nlp tasks. Advances in neural information processing systems Year: (2020)
Ref_id:b24 Title: Lost in the middle: How language models use long contexts Year: (2023)
Ref_id:b25 Title: Trajectory Diversity for Zero-Shot Coordination Year: (2021-05)
Ref_id:b26 Title: Towards Few-shot Coordination: Revisiting Ad-hoc Teamplay Challenge In the Game of Hanabi Year: (2023)
Ref_id:b27 Title:  Year: ()
Ref_id:b28 Title:  Year: (2012)
Ref_id:b29 Title: AI: the future of humanity Year: (2024-03)
Ref_id:b30 Title: Human compatible: artificial intelligence and the problem of control. Viking, New York? Year: (2019)
Ref_id:b31 Title: Multi-Agent RL Environments in JAX Year: (2023)
Ref_id:b32 Title: Some studies in machine learning using the game of checkers Year: (1959)
Ref_id:b33 Title: Proximal Policy Optimization Algorithms Year: (2017)
Ref_id:b34 Title: Mastering the game of Go with deep neural networks and tree search Year: (2016-01)
Ref_id:b35 Title: Ad hoc autonomous agent teams: collaboration without pre-coordination Year: (2010-07)
Ref_id:b36 Title: Collaborating with Humans without Human Data Year: (2021)
Ref_id:b37 Title: Td-gammon, a self-teaching backgammon program, achieves master-level play Year: (1994)
Ref_id:b38 Title: A new formalism, method and open issues for zero-shot coordination Year: (2021)
Ref_id:b39 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b40 Title: Turn 3: Success (Actor 1 plays card slot 2, knowing it's a B1, which follows the Year: ()
Ref_id:b41 Title: Success Year: ()
Ref_id:b42 Title: Turn 6: Failure (Actor 0 should have 2-saved the Red 2 on chop) Year: ()
Ref_id:b43 Title: Success (Actor 0 plays slot 0 Year: ()
Ref_id:b44 Title: Success (Actor 1 discards chop on slot 3) Year: ()
Ref_id:b45 Title: Success (Fix clued the duplicate cards) Year: ()
Ref_id:b46 Title: Success (Plays known playable card Year: ()
Ref_id:b47 Title: Success Year: ()
Ref_id:b48 Title: Success (Plays the Red 1) Year: ()
Ref_id:b49 Title: Success (Discards chop) Year: ()
Ref_id:b50 Title: Success Year: ()
Ref_id:b51 Title: Success (Plays Yellow 3) Year: ()
Ref_id:b52 Title:  Year: ()
Ref_id:b53 Title: Success (Discards chop) Year: ()
Ref_id:b54 Title: Failure (Doesn't understand chop-focus, giving play clue to wrong card Year: ()
Ref_id:b55 Title: Failure (The focus of the last clue was the chop Year: ()
Ref_id:b56 Title: Success Year: ()
Ref_id:b57 Title: Success (Plays known playable Red 3) Year: ()
Ref_id:b58 Title: Success (Discards chop Year: ()
Ref_id:b59 Title: Success (Gives play clue to White Year: ()
Ref_id:b60 Title: Success (Plays White Year: ()
Ref_id:b61 Title: Success Year: ()
Ref_id:b62 Title: Success (Plays the Red 4) Year: ()
Ref_id:b63 Title: Success (Play clue on Blue Year: ()
Ref_id:b64 Title: Success (Plays White 3) Year: ()
Ref_id:b65 Title:  Year: ()
Ref_id:b66 Title: Success Year: ()
Ref_id:b67 Title: Success (Plays White 4) Year: ()
Ref_id:b68 Title: Success Year: ()
Ref_id:b69 Title: Success (Discards known trash Red 1) Year: ()
Ref_id:b70 Title: Failure (Should have played its Blue 3 because of the play clue, instead gave a 5 hint off chop Year: ()
Ref_id:b71 Title: Success (Discards chop) Year: ()
Ref_id:b72 Title: Failure (Should have played its Blue 3, instead gave a 2 hint Year: ()
Ref_id:b73 Title: Success (Discards chop) Year: ()
Ref_id:b74 Title: Failure Year: ()
Ref_id:b75 Title: Success Year: ()
Ref_id:b76 Title: Success (Plays Blue 3) Year: ()
Ref_id:b77 Title: Success (Discards chop) Year: ()
Ref_id:b78 Title: Success (Plays Blue 4) Year: ()
Ref_id:b79 Title: Success (Discards chop) Year: ()
Ref_id:b80 Title: Success Year: ()
Ref_id:b81 Title: Success Year: ()
Ref_id:b82 Title: Success Year: ()
Ref_id:b83 Title: Success (Plays Green 1) Year: ()
Ref_id:b84 Title: Success (Discards chop) Year: ()
Ref_id:b85 Title: Success (Play clue on Yellow 4) Year: ()
Ref_id:b86 Title: Success (Plays Yellow 4) Year: ()
Ref_id:b87 Title: Success (Plays Green Year: ()
Ref_id:b88 Title: Success (Reveals Green 4 identity Year: ()
Ref_id:b89 Title: Success (Plays Yellow 5) Year: ()
Ref_id:b90 Title: Success (Discards chop) Year: ()
Ref_id:b91 Title: Success (5 save on Green 5) Year: ()
Ref_id:b92 Title: Success (Stalling, hinting 1s) Year: ()
Ref_id:b93 Title: Failure (Hinting Green is seen as a play clue on Green 1 Year: ()
Ref_id:b94 Title: Success (Plays Green 1 Year: ()
Ref_id:b95 Title: Success Year: ()
Ref_id:b96 Title: Success Year: ()
Ref_id:b97 Title: Success (Hints White 5) Year: ()
Ref_id:b98 Title: Success Year: ()
Ref_id:b99 Title: In summary, in this game, the human proxy followed H-group conventions for 88% of the moves and used various strategies while playing the game Year: ()
