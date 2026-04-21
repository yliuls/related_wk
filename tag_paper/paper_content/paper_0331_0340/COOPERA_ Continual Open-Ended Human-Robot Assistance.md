Title: COOPERA: Continual Open-Ended Human-Robot Assistance
Abstract: To understand and collaborate with humans, robots must account for individual human traits, habits, and activities over time. However, most robotic assistants lack these abilities, as they primarily focus on predefined tasks in structured environments and lack a human model to learn from. This work introduces COOPERA, a novel framework for COntinual, OPen-Ended human-Robot Assistance, where simulated humans, driven by psychological traits and long-term intentions, interact with robots in complex environments. By integrating continuous human feedback, our framework, for the first time, enables the study of long-term, openended human-robot collaboration (HRC) in different collaborative tasks across various time-scales. Within COOPERA, we introduce a benchmark and an approach to personalize the robot's collaborative actions by learning human traits and context-dependent intents. Experiments validate the extent to which our simulated humans reflect realistic human behaviors and demonstrate the value of inferring and personalizing to human intents for open-ended and long-term HRC.

Section: Introduction
A long-standing goal in robotics is to develop agents that can effectively assist humans in their daily lives by adapting to their preferences and habits. In order to do this, a robot agent must be able to not only learn to interact in environments with humans in a given moment, but also reason about the human across long periods of time, adapting its behavior to provide better assistance. For example, such an agent should be able to fetch a cup of coffee while also understanding that someone may prefer it cooler in the morning but stronger in the afternoon, heating up water accordingly.
Over recent years, several works have made significant advances in developing agents that can assist humans in household tasks [52,50,75,65,47], using simulation environments to study human-robot collaboration (HRC) in a safe and scalable manner. However, most of these works focus on episodic settings, where a robot is evaluated over a set of short collaboration scenarios with tasks specified in advance. These settings are very different from real-world scenarios, where humans have preferences and long-term goals that guide their behaviors, needing different types of assistance at different times.
To advance robot agents that can assist and adapt to humans, we propose COOPERA, a novel framework for COntinual, OPen-Ended human-Robot Assistance in complex household environments (Fig. 1). At its core, COOPERA features a human model with preferences that supports long-term interactions, a feedback mechanism, and benchmarks and metrics to evaluate if robots can assist humans in long-term tasks and reason about their preferences effectively.
To model realistic humans, we simulate humans using an LLM with detailed human traits and habits, retrieved environment information, and intention history, enabling behaviors that exhibit three characteristics. 1) Dynamic intention-driven: Humans act based on intentions that vary over time Figure 1: Continual human-robot collaboration for open-ended tasks over multiple days. Our framework COOPERA entails an approach to simulate traits-driven humans with long-term, whole-day behaviors within robot simulation platform, enabling the first study of long-term, open-ended human-robot collaboration. We also introduce a benchmark and a method for the robot to personalize collaboration in such continual, open-ended settings by learning human traits and context-dependent intents over time.
(e.g., setting the dinner table at 6 pm, then watching TV at 7 pm). 2) Open-ended and environmentconditioned: Rather than following predefined tasks, humans generate spontaneous intentions based on the environment, available objects, and time of day. 3) Traits-driven: Psychological traits and habits shape human behavior, resulting in diverse routines even within similar environments (e.g., one person starts their day reading, while another prefers cleaning).
As the human interacts in the environment, we need a way to provide feedback to the robot so it can improve over time. We structure our framework into two stages which happen on each day of interaction. At the beginning of the day, the robot observes and collaborates with the human, assisting in inferred tasks. At the day's end, the human communicates with the robot and provides feedback to help improve the robot's collaboration success rate for subsequent days.
COOPERA presents unique challenges that are often overlooked in existing HRC benchmarks. First, robot agents need to reason not only about the environment state but also a given human's behavior for effective assistance. Rather than learning a behavior that assists all possible humans, they need to adapt to each person's preferences and traits. Second, humans in our framework perform different tasks depending on the time of the day or the day of the week (e.g., they may only do exercise once). Thus, effective agents need to reason about time as a cue for how to best assist the human.
To explore this challenging framework, we provide a benchmark and propose a method that tracks human preference profiles and uses VLMs and classifiers to predict and score human goals based on time, observed behavior, and profile, suggesting actions to achieve these intentions. This enables the robot to learn and mimic human behavior by capturing the underlying correlations between human traits, temporal dependencies, and their corresponding intentions and tasks. We compare our method against several baselines, evaluating the robot's collaboration performance over multiple days in different collaborative tasks and across diverse humans and scenes. Furthermore, we conduct extensive experiments to assess how well our simulated humans reflect real human behavior, particularly their ability to exhibit distinct, trait-driven patterns aligned with human profiles. In summary, our main contributions are threefold:
• We present COOPERA, a novel HRC framework for continual, open-ended collaboration with humans who exhibit individual traits across long horizons.
• We develop a method to simulate humans with long-term behavior models driven by individual traits and habits.
• Within this framework, we introduce a benchmark and an approach that enables increasingly adaptive and personalized collaboration with humans over multiple days.
this section cite: ['b51', 'b74', 'b64', 'b46']

Section: Related Work
Human-Robot Collaboration. Prior HRC work has largely focused on controlled lab settings [19,10,44,59], where collaborative tasks are shared by both the human and the robot or narrowly defined.
More recent research has expanded to complex household environments, requiring robots to infer human intentions from a single demonstration [50,20,64] or in an online fashion [51]. Subsequent works explore human intention inference using data from images [39] or simplified environments (e.g., 2D worlds) [5,55,70,68], progressing to simulated real-world environments [20] and leveraging recent advances in VLMs. However, these approaches typically rely on predefined, closed-form representations of human intentions and tasks [14,32,2,12,71], and often ignore realistic human behavior. Furthermore, collaboration is usually limited to fixed episodes with predefined task set. In contrast, our work considers open-ended and continual HRC, where humans spontaneously propose their actions based on environmental factors, and the collaboration persists across days.
this section cite: ['b18', 'b9', 'b43', 'b58', 'b19', 'b63', 'b50', 'b38', 'b54', 'b69', 'b67', 'b19', 'b13', 'b31', 'b11', 'b70']

Section: Human Simulation.
Most embodied AI works [6,3,16,17,60] assume that environmental changes are solely driven by a single robot [52]. Due to the challenges of real-human experiments (e.g., safety, scalability, cost), recent research has integrated deformable humans with plausible motion and appearance into robot simulation platforms [49,52], enabling the study of safe and scalable HRC. However, these simulated humans focus only on motion feasibility, lacking the complexity and variability of real human behavior. Another research direction simulates humans with psychological traits and social interactions [76,43,46,26,64], but remains language-based and does not involve environmental interaction. In contrast, we simulate humans driven by psychological traits and habits, whose behavior is long-term and capable of interacting with their environment.
LLMs for Human Task Inference. One line of research treats human intentions as direct inputs and investigates how LLMs can interpret open-ended natural language instructions to generate structured robot plans [25,24,23,73,63,38]. These works use techniques such as 3D scene graphs [56,36,8] to semantically ground high-level goals and decompose them into actionable subgoals, or incorporate human feedback [58,33,9] to quantify uncertainty and enable skill acquisition through interaction. In contrast, COOPERA takes a step further by aiming to let LLMs/VLMs infer personalized task plans, adapting to specific human traits and habits rather than general commonsense knowledge.
this section cite: ['b15', 'b16', 'b59', 'b51', 'b48', 'b51', 'b75', 'b42', 'b45', 'b25', 'b63', 'b24', 'b23', 'b22', 'b72', 'b62', 'b37', 'b55', 'b35', 'b57', 'b32', 'b8']

Section: COOPERA: Continual, Open-Ended HRC Framework
Our goal is to enable the study of continual HRC in open-ended tasks. To that end, we investigate how a robotic agent can become more effective in assisting humans by learning from their behavior. Central to COOPERA are LLM-powered simulated humans driven by traits and long-term intentions that the robot can reason for effective collaboration, and a human feedback mechanism for improving collaboration over time. We first outline our framework and problem setup, detailing the collaboration settings we explore (Fig. 2). Then, we describe our approach of simulating humans driven by traits with long-term behaviors (Fig. 3). Finally, we propose a method to tackle our framework (Fig. 4).
this section cite: []

Section: Overview

this section cite: []

Section: Feedback

this section cite: []

Section: Optimization Alg. acts

this section cite: []

Section: After Each Day

this section cite: []

Section: Assistive Tasks
First Task Demo observes predicts helps
this section cite: []

Section: Human Model

this section cite: []

Section: LLM

this section cite: []

Section: Human Profile

this section cite: []

Section: Continual (Multiple-Day), Open-Ended HRC Framework
Figure 2: COOPERA: Continual, open-ended humanrobot collaboration framework. The LLM-powered human proposes whole-day intentions and tasks, executed in the environment. As the robot observes the human actions, it predicts a set of tasks to assist them. After each day, the human provides feedback to the robot, enabling the robot to improve for subsequent days.
In order to investigate HRC in a safe and reproducible manner, we consider a simulated human agent that interacts in a 3D household environment to achieve a set of high-level goals. These goals vary throughout the day and are driven by human traits and habits, as well as by the activities the human has done before. The robot's goal is to assist the human in those tasks, without receiving explicit commands about the goal they should help with, or information about the human's traits. Both human and robot have full knowledge of the environment. Each day is represented as 12 one-hour intervals covering the time from 9 am to 9 pm (the rest is treated as being asleep). At the beginning of each hour, the human proposes a high-level intention (e.g., leisure) and decomposes it into a sequence of tasks and executes them in the environment (e.g., watch TV on the sofa). As the human interacts in the environment, the robot has to infer the human's goals and provide assistance. At the end of each day, the human provides feedback on the robot's help, which is then used to improve the robot's collaboration success in subsequent days.
Problem Setup. We define two types of collaboration with increasing difficulty and openness. Collaboration type 1 is an open-ended variant of the Watch-and-Help challenge [50], where one intention (e.g., set up dinner table) is decomposed into 3 pick-and-place tasks (i.e., picking an object and placing it on a static object). For each intention, the robot is given a video of the human
"I'm an organized and responsible individual... OCEAN: [4.65, 3.4, 4.2, 4.95, 1.55]" Traits, Habits, Conversations & Psychometric Data Intention Proposal Rooms in Household Scene Time Time, Intention & Task History Tasks Proposal Object-Room Mapping Execution Memory Retrieval (recency, relevance) Search Proposed Intention Reflect 3D Info Usage Profile Extension Reflect Human Alignment & Dependence Motion Dataset Proposed Motion Search "Time: 9 am Intention: Organize and clean the living room." "Task 1: Dust the area around the white sideboard using a duster. Motion: dusting ..."
this section cite: []

Section: LLM LLM LLM
Figure 3: Human Simulation Pipeline. We seed the human-LLM with an extended profile. At each time of day, the human proposes an intention and decomposes it into tasks, aligning with profile traits and temporal dependence on intention/task history. LLM inputs are optimized with Memory Retrieval and Search, and robustness is enhanced via two rounds of Reflexion. This pipeline generates continuous, whole-day intentions and tasks executed in the environment with expressive whole-body motion. See Appendices C and F for details.
performing the first task and its textual description, and must infer and assist with the remaining tasks based on objects available in the scene. Collaboration type 2 is more challenging and moves beyond pick-and-place: each intention (e.g., morning hygiene) is decomposed into 5 tasks involving free-form human motion while interacting with static objects (e.g., the human brushes teeth at the mirror). Unlike type 1, the robot is unconstrained by the scene and may propose any object it deems helpful (e.g., the robot offers toothbrush). It receives only the first task video with no textual guidance.
Evaluation Settings. We define four progressively challenging settings. 1) Same human, same scene: The robot collaborates with the same human in the same scene over 5 consecutive days (5 days, 1 scene). 2) Same human, different scenes: The robot collaborates with the same human across 5 different scenes, with a new scene each day (5 days, 5 scenes). 3) Different humans, same scene:
The robot collaborates with different humans in the same scene, rotating among Human 1, 2, and 3, each for one day, repeating this cycle three times in the same scene (9 days, 1 scene). 4) Different humans, different scenes: The robot collaborates with different humans across multiple scenes, rotating through Human 1, 2, and 3 in the first scene, then repeating this sequence in the second and third scenes (9 days, 3 scenes). In 3) & 4), we explore if knowledge gained from interacting with different humans improves future collaboration, despite fewer interaction days per human.
this section cite: []

Section: Simulating Humans
We aim to model humans who interact in the environment over long periods of time, act driven by their goals, preferences, and context, and who can react and provide feedback as a robot assists them.
To achieve this, we propose a hierarchical model that combines LLMs and 3D human motion to simulate long-term, realistic human behaviors in indoor environments. First, our model generates a description of human traits describing their preferences and habits. Based on these traits, the environment, and the history of human actions, the model then generates a sequence of tasks for the human to perform. For every task, we use the environment information to generate human motions and interactions, providing a realistic demonstration of each task. Fig. 3 shows an overview of our design. Next, we describe in more detail each of the human simulation components.
this section cite: []

Section: Generating Human Traits.
We use LLMs to generate personality traits that determine the human long-term behaviors. For this, we sample conversations from the Synthetic Human Dataset [26], containing dialogues between different humans, and prompt an LLM to generate a description of the human based on the conversation, inferring attributes such as their job, preferences or common activities. Inspired by [67,77,48], we also prompt the LLM to generate a vector measuring Big-5 human personality traits [18] (openness, conscientiousness, extroversion, agreeableness, and neuroticism), allowing us to measure the diversity across generated humans and how well the robot agents can infer the human's personality from their interactions.
Whole-Day Intentions and Tasks. Given human traits, we generate long-term human behaviors, with tasks featuring temporal dependences within a day and diversity across days. Temporal Dependence: Given 3D environment information, we use an LLM to propose intentions for different times of day (e.g., 9 am: clean the living room). Next, we prompt the LLM to decompose the intentions into a sequence of inter-dependent tasks (e.g., dust the area around the white board, clean the counter). The LLM also receives the human's intention and task history from previous hours, and is explicitly prompted to consider their inter-dependency. Varying Distribution: While humans with specific traits follow general routines, their daily behavior varies daily (i.e., Monday 9 am for cleaning, Tuesday 9
this section cite: ['b25', 'b66', 'b76', 'b47', 'b17']

Section: Inferred Intentions

this section cite: []

Section: Feedback

this section cite: []

Section: Inferred Tasks Actions Actions

this section cite: []

Section: Inferred Human Profile

this section cite: []

Section: Traits Intentions
Habits Time Tasks
this section cite: []

Section: Multi-Day Collaboration History
optimizes I want to... I can help. Time: 3 pm Intention: I want to explore creative ideas for enhancing bedroom decor. Actions: 1. Sit on the bedside table holding a small plant to imagine placements. 2. Jot down idea with a notebook. 3. Hold a candle considering how ambient lighting enhances setup. LLM CLS LLM Classifiers Inferred Traits & Data Time + Intention/Task Retrieved History [ 0 1 ] After Each Day Human: Indoor gardener, plant lover. Robot: Helpful assistant. Time: 3 pm Inferred Intention: I can help with arranging plants for a inviting space. Tasks for Assistance: 1. Offer plants for inspection, checking for dry soil or health issues. 2. Offer magnifying glass for inspection. 3. Place a scented candle near plant clusters to add warmth.
this section cite: []

Section: LLM CLS

this section cite: []

Section: scores

this section cite: []

Section: VLM

this section cite: []

Section: LLM CLS

this section cite: []

Section: scores

this section cite: []

Section: VLM

this section cite: []

Section: Human Model

this section cite: []

Section: LLM

this section cite: []

Section: Frozen Finetuned
Figure 4: Our approach for human assistance. We decouple robot task inference into intention and task inference. By chaining VLM and classifier, the robot selects tasks aligned with the human's traits and temporal context. It maintains a human profile inferred from collaboration history, which, combined with feedback, optimizes the robot-VLM via prompting and the classifiers via supervised learning. See Appendices D and G for details.
am for exercise). To model this, we reset the intention and task history at the start of each new day, setting a high temperature for the human-LLM to encourage diversity across days.
Expressive Whole-Body Motion. We simulate human agents physically using expressive 3D whole-body motions during task execution [31,21,42] by chaining motion sequences for each task. For pick-and-place tasks, the sequence includes walking, reaching and picking, walking again, then reaching and placing. For tasks involving free-form motion (e.g., sitting on a sofa), the human-LLM describes a free-form human motion that matches each task, using examples from our human motion dataset. The resulting sequence combines walking with the selected free-form motion.
this section cite: ['b30', 'b20', 'b41']

Section: Optimizing Long-Context Inputs.
Our progressive prompting chain provides the human-LLM with substantial information at each stage, especially during the task proposal stage, where the 3D environment may contain hundreds of objects and the motion dataset includes thousands of data points. Additionally, as the day progresses, the intention and task history grows long (e.g., from 9 am to 9 pm, 13 intention sentences and dozens of task descriptions accumulate). Since LLMs struggle with long-context inputs [30,35,69], we introduce Search and Memory Retrieval mechanisms. Search: Given a query text and a list of texts, we return the top-K most relevant items based on semantic similarity. Memory Retrieval: We use recency and relevance scores to retrieve the top-K memories. Recency decays over time with a decay factor λ from the current time, and relevance is calculated by semantic similarity, similar to search. The final retrieval score is the product of both [46].
this section cite: ['b29', 'b34', 'b68', 'b45']

Section: Self-Corrections.
Given the complexity of our progressive prompting process and the LLM responses, even state-of-the-art models can make mistakes. Therefore, during the most complex task proposal stage, we perform two rounds of Reflexion [61,74,72] to identify and correct errors related to human traits, temporal dependencies, and object use within the 3D environment.
this section cite: ['b60', 'b73', 'b71']

Section: Instantiating COOPERA with an Assistive Agent
To study COOPERA, we propose an approach (Fig. 4) for continual HRC, enabling the robot to learn correlations between human intentions, tasks, traits, and temporal dependencies at each time of day.
At any given time, a human's intentions/tasks can be viewed as meta-intentions/meta-tasks, encompassing a range of possible options due to the diversity of human behavior across days. Our solution decouples task inference into two stages: first inferring intentions, then identifying specific tasks. We capture the correct sets by chaining VLM to imagine multiple possible intentions/tasks and classifiers to score and filter them. Given observation (frames uniformly extracted from a video V = [f 1 , . . . , f N ]) of the human's first task, the robot-VLM generates an intention superset. For each positively classified intention by the intention classifier, the robot-VLM infers a set of possible tasks, forming a task superset. The task classifier then identifies the tasks most suitable for collaboration.
We optimize the robot-VLM through prompting and the binary classifiers via supervised learning. Using human feedback from the end-of-day discussion, the robot keeps tracks of a human profile by prompting robot-VLM to infer and summarize the human's traits, habits, and psychometric data. This human profile, along with the retrieved history of intentions and tasks, is incorporated into the robot-VLM prompts and provided as input to the classifiers in the subsequent times and days. The robot-VLM and classifiers are optimized per day. Please see Fig. 4 for the input data format.
this section cite: []

Section: Experiments and Analysis
Within COOPERA, we first examine 1) if the central component, the simulated human model, reflects real human behavior and to what extent. 2) We then introduce the benchmark setup (baselines, evaluation metrics) and explore if our proposed approach leads to more personalized robot assistance over multiple days compared to baselines. 3) Subsequently, we analyze the real-world applicability of our framework. 4) Finally, we evaluate the effectiveness of each module through ablation studies.
this section cite: []

Section: Framework Implementation
Environment and Scene. We use Habitat 3.0 [52] as the robot simulation platform and HSSD [28] as the 3D environment, which includes 18,656 static objects across diverse scenes in style and size. Since the original HSSD includes only static objects, we develop a systematic approach to create dynamic scenes by making small objects from specific categories (e.g., decor, kitchenware) movable. We also sample 20 dynamic objects from the YCB Dataset [7] and place them in contextually appropriate locations (e.g., a mug on a bedside table) using Habitat's built-in tools. Dynamic scenes are initialized at the start of each episode. Across days, Habitat tracks object locations as the human and robot interact with the environment, allowing them to maintain updated environment knowledge. We select 5 scenes with varying of rooms (4-11), static objects (51-140), and dynamic objects . All scenes provide enough space for the human and Fetch robot [15] to navigate. Please see Appendix B for more details.
this section cite: ['b51', 'b27', 'b14']

Section: Human Dataset.
For modeling unique humans, we use the SPC: Synthetic-Persona-Chat Dataset [26], a fully synthetic dataset that includes hundreds of short user profiles along with their conversations and compute psychometric data (details in Appendix C). We use Motion-X [31] and AMASS [41] as the human motion dataset. We generate 10 human profiles.
this section cite: ['b25', 'b30', 'b40']

Section: Training and Inference.
We use open-source models for interpretability and benchmarking value. For simulating humans, we use Llama-3.1-8B [13] with temperature 0.7. For search and memory retrieval, we use MiniLM-L6-v2 [66] with a decay factor λ = 0.95, retrieving the top 3 intentions and top 5 tasks. For the assistive agent, we use Llama-3.2-11B [13] as the robot-VLM. Classifiers are finetuned on Mistral-7B-Instruct-v0.2 [27] using LoRA [22] in instructional format to output binary yes/no. We train on 3 NVIDIA A10 GPUs (24GB RAM). Please see Appendix D for more details.
this section cite: ['b12', 'b65', 'b12', 'b26', 'b21']

Section: Analysis of Human Simulation
Distinct Simulated Humans. We examine if simulated humans with distinct Big-5 traits exhibit machine-identifiable features. Each of 10 humans is placed in 5 scenes, living 20 days per scene. We aggregate daily intentions and tasks into one data point per human. Two 10-way BERT-largeuncased classifiers [11] are finetuned-one for intentions (10 epochs), one for tasks (20 epochs) with train-test split 0.8:0.2, learning rate 5e-6, and tested on an unseen scene. As shown in Table . 1, task classification is harder than intention classification, as intentions align more with human traits, but tasks (e.g., drinking water) may correspond to multiple intentions (e.g., leisure, exercise).
Diverse Simulated Humans. We assess diversity by standard deviation (SD) of Big-5 traits (1-5 scale) across 10 simulated humans. We compute per-trait SD and take average. From Table . 1, the high SD exceeds the typical 0.7-0.9 range in real-world distributions [62], validating diversity.
Human Traits and Psychometrics Coherence. In our main approach, the robot-VLM infers Big-5 traits throughout the day based on the human's intention and task history. Using the final scores at the end of collaboration, we assess coherence with ground truth via Pearson correlation [48,4], and introduce a one-step mismatch for comparison. The significant drop in correlation for mismatched pairs (Table. 1) confirms alignment between inferred traits and psychometric data, demonstrating the LLM's ability to interpret human psychology from behavior. 9 am Traits / Habits Intentions {openness: 4.2, conscientiousness: 3.0, extroversion: 1.8, agreeableness: 2.9, neuroticism: 2.9} 10 am 11 am 4 pm 5 pm 6 pm
this section cite: ['b10', 'b61', 'b47']

Section: Tasks / Actions

this section cite: []

Section: am
Playful activity in the living room.
Relaxing bath while listening to a quirky podcast.
Engage in a new creative project in the bedroom.
Imaginative storytelling session in the living room.
Cooking activity in the kitchen.
Dinner in the living room while watching a quirky cartoon.
1. Sit on the bed, read and brainstorming.
this section cite: []

Section: 2.
Draw inspirations from the artwork.
this section cite: []

Section: … … … … … …
Task Sequence Temporal Dependence in Human Behavior. We study how current-hour intentions depend on prior hours via a next-sentence prediction task: given three earlier intentions (e.g., 9-11 am), the 12 pm intention is used as the positive example, with other-time intentions as negatives. We evaluate on 10 simulated humans, each living for 20 days in 5 scenes. BERT-large-uncased [11] is trained for 20 epochs (learning rate 5e-6). Results in Table . 1 confirm temporal dependence.
this section cite: ['b10']

Section: User Studies.
We conduct two user studies (25 participants each) to assess: 1) Whether real humans can identify the same simulated human across days and scenes, and 2) Whether real humans can distinguish simulated humans with varying traits and Big-5 scores. For 1), we sample fullday intentions and tasks of 10 simulated humans across 2 days and 2 scenes (4 samples/human), then construct 10 multiple-choice questions showing a human profile and three behavior options (1 correct, 2 distractors). For 2), we sample full-day behaviors from 10 simulated humans with distinct traits, and ask participants to match trait descriptions to the corresponding full-day intentions and actions. Results in Table . 1 show higher accuracy in identifying the same simulated human than in distinguishing between different simulated humans. This discrepancy likely arises because multiplechoice tasks provide explicit answer options, reducing ambiguity, whereas trait-based matching requires deeper reasoning about personality-behavior relationships, making it more challenging. Alignment with Real-Human Behavior. We study how simulated human intentions align with real-human intentions.
We recruit six participants who provide personality traits and psychometric data, and record their daily intentions over five days. Using these traits, we prompt the LLM to generate simulated intentions for the same time span. To assess alignment, we aggregate both sets into single paragraphs (removing time formatting like "9am: ..." to avoid inflated structural similarity) and compute semantic similarity using SBERT (allmpnet-base-v2) [57] and OpenAI embeddings (text-embedding-3-small) [1]. We compare against: 1) prompting without human profile (generic) and 2) mismatched LLM-human intention pairs (mismatched). From Table 2, both baselines yield moderate similarity (∼0.5), as sentence encoders assign partial similarity to structurally similar content. In contrast, aligned pairs achieve much higher scores, indicating strong alignment between simulated and real human intentions. SBERT slightly outperforms OpenAI embeddings, likely due to its sentence-level training objective.
this section cite: ['b56']

Section: Qualitative Results.
We present examples of full-day intentions and tasks proposed by a human with specific human traits and psychometric data in Fig. 5.
this section cite: []

Section: Analysis of Continual, Open-Ended HRC
Since COOPERA involves long-term, open-ended tasks that requires the robot reasoning over human traits and temporal context, we construct baselines using standard LLM/VLM-based approaches adapted for task inference. These baselines reflect commonly used paradigms in open-world robot planning [8,29].
this section cite: ['b28']

Section: Baselines. 1) Direct Prompting:
The robot proposes a single intention from visual input and decomposes it into tasks. The robot-VLM is optimized solely via prompting with retrieved intention/task history. 2) Direct Finetuning: The robot brain is finetuned to directly output a single intention and decompose it into tasks. 3) Oracle: The robot is given the ground-truth human intention and decomposes it into tasks. 4) Random: Intention and task classifiers are removed; all proposed intentions and tasks are accepted without validation. 5) Intention Agnostic: The robot directly predicts and filters tasks without first inferring intentions. 6) Human & Context Agnostic: The classifiers do not learn the correlation between human traits and intentions/tasks or the temporal dependence between previous and current intentions/tasks. They only learn the relationship between the current time and the intentions/tasks.
Evaluation Metrics. We assess the assistive agent's performance using F1-based success rate across three methods, ensuring a comprehensive evaluation from simulation to real-world perspectives, incorporating prior HRC approaches [51,8]. 1) Predicate-based: Tasks are executed and evaluated by predicate functions with success based on object class matches rather than instance matches, following Watch-and-Help
this section cite: ['b50']

Section: Analysis of Assistive Performance.
We analyze two aspects: 1) Within-day improvement-does the robot's collaboration success increase throughout the day by learning temporal dependencies between human intentions and actions? 2) Across-day improvement-does collaboration become more successful and personalized over multiple days, using end-of-day feedback? From Fig. 6 (a), our method achieves the highest withinday improvement. In contrast, prompting, random, and oracle exhibit little to no improvement, or even decline. We hypothesize that these methods do not benefit from learning human intentions, which are highly correlated with human traits and temporal context. This finding aligns with our human classification experiment in Section 4.2. From Fig. 6 (b), our method shows the strongest improvement across days, second to oracle. The minimal gain in prompting and finetuning highlights the challenge of varying human behavior across days, as these methods tend to establish a 1-to-1 mapping between time and human intentions/tasks. Prompting relies heavily on collaboration history, while finetuning prioritizes the highest probability training data, limiting adaptability to varying human behaviors.
this section cite: []

Section: Out-of-Domain Generalization.
We study 1) Scene generalization: can the robot personalize collaboration with a human in an unseen scene after interacting in other scenes? 2) Human generalization: can the robot collaborate effectively with a new human after training with others? For 1), we use models finetuned from setting 2 on four scenes and evaluate on a fifth, unseen scene with the same human. The baseline is the robot's average performance during its initial interaction in the unseen scene, without finetuning on the previous scenes, averaged over 10 humans.
For 2), we use models finetuned from setting 3 on three humans and evaluate on collaboration with a fourth unseen human. The baseline is the robot's unadapted performance when first interacting with the new human, without finetuning on the previous three, averaged over 5 new humans. Result in Table . 3 show that generalizing to a new human is much harder than to a new scene. This is likely due to greater variability in human behaviors. While the robot can learn shared patterns across humans, effective collaboration with a new human requires adaptation to fine-grained, person-specific traits.
Qualitative Results. We show how the robot improves collaboration within a day by inferring more correct tasks for assistance, along with a visualization of HRC at a specific time in Fig. 7.
At 9 am, the human wants to engage in a leisure activity in the living room. Collaboration is successful. The human sits on the sofa. The human opens the TV. Based on the human practicing relaxing yoga at 9 am on the previous day and current observation, the robot infers her intention is leisure. The human approves the assistance and picks up the tea. The robot fetches a cup of tea from the kitchen for relaxation.
The robot puts the tea on the table. The human enjoys the tea. a. Human proposes intention and task. b. Robot infers intention and task. c. Human judges collaboration. Leisure in living room 9 am 12 pm 3 pm 6 pm 9 pm Meditate in bedroom Lunch prep in kitchen Reflect & create in bedroom Enjoy relaxing evening Read comics Enjoy tea Human Robot Read cookbook Fill pot Wash hand with soap Sip a drink Offer cozy pillow Video game with gamepad Blanket for comfort Offer pen Lighter for candle Notebook for poem
this section cite: []

Section: Analysis of Real-World Applicability
The ultimate goal of COOPERA is to develop robot agents that assist real humans by adapting to their preferences over long-term. Yet, fully real-world experiments pose ethical, safety, and cost issues (requiring a real human and physical robot to interact in a household over multiple days). To validate applicability of COOPERA under real-world conditions, we take three complementary approaches. Please see Appendix E for detailed results, qualitative examples, and additional analysis. Also, in open-ended settings with large state spaces, LLMs/VLMs serve as effective reasoning modules due to their generalization abilities [8,29].
Collaborating with Offline Real Humans. Real humans exhibit greater behavioral dynamics and emergent decisions due to temporary factors (e.g., plans, mood, weather). We recruit six participants who provide personality traits and psychometric data, and record their daily intentions over five days. An LLM decomposes these into tasks in HSSD scenes, where the robot collaborates across all settings. Results in Table 4 show performance comparable to simulated humans (Table . 6 row 4), despite increased dynamics.
Human-in-the-Loop. Six real humans replace the LLM and collaborate with the assistive agent. They are shown retrieved object and motion sets and select which to interact with based on their intention. For ease of implementation and usability, participants input responses as text rather than using a keyboard to control the simulated agent. Results in Table 4 indicate that real human collaborators do not make the task more difficult and can, in some cases, lead to higher success rates compared to offline or simulated humans.
this section cite: ['b28']

Section: Ablation Studies
Human Simulation. 1) Removing human profile extension: To explore whether our method yields the most distinct simulated humans, we remove simulated conversations and profile extension, prompting LLM only with the original short trait paragraph. We evaluate by finetuning and measuring Assistive Agent. 1) Removing human traits inference. We examine the importance of learning human traits by removing robot's inference of human traits or Big-5 scores from past intentions and tasks, preventing classifiers from learning their correlation. 2) Removing temporal context learning: We assess the impact of learning temporal dependence between human intentions and tasks by preventing classifiers from using past intentions/tasks when predicting the current one. 3) Changing the robot brain backbone: We replace the robot-VLM from Llama-3.2-11B [13] to LLaVA-1.6-Mistral-7B [34]. Using different VLMs for human and robot reduces alignment and tests robustness. From Table 6, removing trait inference significantly reduces success rate in settings 3 and 4 involving multiple humans, as the robot struggles to distinguish them. The learning of time-based context benefits all settings. Despite smaller model size, the robot still achieves reasonable success.
this section cite: ['b12', 'b33']

Section: Conclusion
We introduce COOPERA, a framework for continual, open-ended HRC. We propose a human model to generate long-term human behaviors driven by personality traits, a benchmark, and a method to assist humans under by predicting their long-term intentions. Our framework opens up exciting directions for future work, such as using communication to better infer human traits or build agents that can perform proactive assistance (e.g. arranging a house before the start of the day based on preferences). We hope that this work can promote future research on building agents that can work over long time horizons and adapt to human preferences.
Limitations. Despite compelling HRC performance, COOPERA currently focuses on single-human settings, leaving multi-human collaboration for future work. While evaluations are primarily conducted in simulation, we validate sim-to-real transfer through real-human routines and interactions.
Our method uses skill primitives compatible with standard robot platforms (e.g., Fetch), making it readily transferable to real hardware. Justification: We did not have theoretical assumptions. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: References
Ref_id:b0 Title: New embedding models and api updates Year: (2024)
Ref_id:b1 Title: Autonomous agents modelling other agents: A comprehensive survey and open problems Year: (2018)
Ref_id:b2 Title: Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments Year: (2018)
Ref_id:b3 Title: Predicting the big 5 personality traits from digital footprints on social media: A meta-analysis Year: (2018)
Ref_id:b4 Title: Rational quantitative attribution of beliefs, desires and percepts in human mentalizing Year: (2017)
Ref_id:b5 Title: Objectnav revisited: On evaluation of embodied agents navigating to objects Year: (2020)
Ref_id:b6 Title: The YCB object and model set: Towards common benchmarks for manipulation research Year: (2015)
Ref_id:b7 Title: Partnr: A benchmark for planning and reasoning in embodied multi-agent tasks Year: (2025)
Ref_id:b8 Title: On extending direct preference optimization to accommodate ties Year: (2024)
Ref_id:b9 Title: Socially intelligent robots: dimensions of human-robot interaction. Philosophical Transactions of Year: (2007)
Ref_id:b10 Title: BERT: pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b11 Title: Ave: Assistance via empowerment Year: (2020)
Ref_id:b12 Title: The llama 3 herd of models Year: (2024)
Ref_id:b13 Title: A decision-theoretic model of assistance Year: (2014)
Ref_id:b14 Title:  Year: (2020)
Ref_id:b15 Title: Threedworld: A platform for interactive multi-modal physical simulation Year: (2021)
Ref_id:b16 Title: Navigating to objects in the real world Year: (2023)
Ref_id:b17 Title: An alternative "description of personality": the big-five factor structure Year: (1990-12)
Ref_id:b18 Title: Human-robot interaction: A survey Year: (2007)
Ref_id:b19 Title: GG-LLM: geometrically grounding large language models for zero-shot human activity forecasting in human-aware task planning Year: (2024)
Ref_id:b20 Title: Generating diverse and natural 3d human motions from text Year: (2022)
Ref_id:b21 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b22 Title: Language models as zero-shot planners: Extracting actionable knowledge for embodied agents Year: (2022)
Ref_id:b23 Title: Inner monologue: Embodied reasoning through planning with language models Year: (2022)
Ref_id:b24 Title: Do as I can, not as I say: Grounding language in robotic affordances Year: (2022)
Ref_id:b25 Title: Faithful persona-based conversational dataset generation with large language models Year: (2024)
Ref_id:b26 Title: Mistral 7b Year: (2023)
Ref_id:b27 Title: Habitat synthetic scenes dataset (HSSD-200): an analysis of 3d scene scale and realism tradeoffs for objectgoal navigation Year: (2024)
Ref_id:b28 Title: Embodied agent interface: Benchmarking llms for embodied decision making Year: (2024)
Ref_id:b29 Title: Long-context llms struggle with long in-context learning Year: (2024)
Ref_id:b30 Title: Motion-x: A large-scale 3d expressive whole-body human motion dataset Year: (2023)
Ref_id:b31 Title: Goal inference improves objective and perceived performance in human-robot collaboration Year: (2016)
Ref_id:b32 Title: Enhancing the llm-based robot manipulation through human-robot collaboration Year: (2024)
Ref_id:b33 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b34 Title: Lost in the middle: How language models use long contexts Year: (2024)
Ref_id:b35 Title: DELTA: decomposed efficient long-term robot task planning using large language models Year: (2024)
Ref_id:b36 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b37 Title: Kitchenvla: Iterative vision-language corrections for robotic execution of human tasks Year: (2025)
Ref_id:b38 Title: Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors Year: (2024)
Ref_id:b39 Title: Gradient-less federated gradient boosting tree with learnable learning rates Year: (2023)
Ref_id:b40 Title: AMASS: archive of motion capture as surface shapes Year: (2019-11-02)
Ref_id:b41 Title: Generating continual human motion in diverse 3d scenes Year: (2024)
Ref_id:b42 Title: Agentsense: Benchmarking social intelligence of language agents through interactive scenarios Year: (2024)
Ref_id:b43 Title: Efficient model learning from joint-action demonstrations for human-robot collaborative tasks Year: (2015)
Ref_id:b44 Title: The big five personality test Year: ()
Ref_id:b45 Title: Generative agents: Interactive simulacra of human behavior Year: (2023)
Ref_id:b46 Title: Robot navigation in constrained pedestrian environments using reinforcement learning Year: (2021)
Ref_id:b47 Title: Large language models can infer psychological dispositions of social media users Year: (2023)
Ref_id:b48 Title: Virtualhome: Simulating household activities via programs Year: (2018)
Ref_id:b49 Title: Watch-and-help: A challenge for social perception and human-ai collaboration Year: (2021)
Ref_id:b50 Title: NOPA: neurallyguided online probabilistic assistance for building socially intelligent home assistants Year: (2023)
Ref_id:b51 Title: Habitat 3.0: A co-habitat for humans, avatars, and robots Year: (2024)
Ref_id:b52 Title: Efficient vertical federated learning with secure aggregation Year: (2023)
Ref_id:b53 Title: vfedsec: Efficient secure aggregation for vertical federated learning via secure layer Year: (2023)
Ref_id:b54 Title: Machine theory of mind Year: (2018)
Ref_id:b55 Title: Grounding large language models using 3d scene graphs for scalable robot task planning Year: (2023)
Ref_id:b56 Title: Sentence-bert: Sentence embeddings using siamese bertnetworks Year: (2019)
Ref_id:b57 Title: Robots that ask for help: Uncertainty alignment for large language model planners Year: (2023)
Ref_id:b58 Title: Learning physical collaborative robot behaviors from human demonstrations Year: (2016)
Ref_id:b59 Title: Vint: A foundation model for visual navigation Year: (2023)
Ref_id:b60 Title: Reflexion: language agents with verbal reinforcement learning Year: (2023)
Ref_id:b61 Title: The next big five inventory (BFI-2): Developing and assessing a hierarchical model with 15 facets to enhance bandwidth, fidelity, and predictive power Year: (2017)
Ref_id:b62 Title: Large language models as generalizable policies for embodied tasks Year: (2024)
Ref_id:b63 Title: Infer human's intentions before following natural language instructions Year: (2024)
Ref_id:b64 Title: Co-gail: Learning diverse strategies for human-robot collaboration Year: (2022)
Ref_id:b65 Title: Deep self-attention distillation for task-agnostic compression of pre-trained transformers Year: (2020)
Ref_id:b66 Title: Persuasion for good: Towards a personalized persuasive dialogue system for social good Year: (2019)
Ref_id:b67 Title: Too many cooks: Bayesian inference for coordinating multi-agent collaboration Year: (2021)
Ref_id:b68 Title: Effective long-context scaling of foundation models Year: (2024)
Ref_id:b69 Title: Sparse and complete latent organization for geospatial semantic segmentation Year: (2022)
Ref_id:b70 Title: Touch and go: Learning from human-collected vision and touch Year: (2022)
Ref_id:b71 Title: Direct preference optimization for neural machine translation with minimum Bayes risk decoding Year: (2024)
Ref_id:b72 Title: React: Synergizing reasoning and acting in language models Year: (2023)
Ref_id:b73 Title: Textgrad: Automatic "differentiation" via text Year: (2024)
Ref_id:b74 Title: Building cooperative embodied agents modularly with large language models Year: (2023)
Ref_id:b75 Title: DIALOGPT : Large-scale generative pre-training for conversational response generation Year: (2020)
Ref_id:b76 Title: SOTOPIA: interactive evaluation for social intelligence in language agents Year: (2024)
Ref_id:b77 Title: Prompt Details of the Simulated Human We show exact prompts for LLMs in simulated human behavior. Human Profile Summary and Extension Year: ()
Ref_id:b78 Title: Human 2 profile Year: ()
Ref_id:b79 Title: Instruction: Summarize and reasonably expand Human 1's profile into a first-person self-introduction based on their initial profile and past conversations with other humans. Provide a detailed description covering, if presented, the person's job, hobbies, daily activities, food preferences, social life, physical activity, entertainment preferences, travel habits, personal values, goals and aspirations, stress and coping mechanisms, technological use, cultural interests, health and wellness Year: ()
Ref_id:b80 Title: Input: 1. Human profile. Instruction: Infer Big Five personality traits (scale 1-5, float) based on the provided human profile. Write in the following format: {'openness': a, 'conscientiousness': b, 'extroversion': c, 'agreeableness': d Year: ()
Ref_id:b81 Title: In the questions below, for each statement 1-50 mark how much you agree with on the scale 1-5, where 1=disagree, 2=slightly disagree Year: ()
Ref_id:b82 Title: Feel little concern for others Year: ()
Ref_id:b83 Title: Am full of ideas Year: ()
Ref_id:b84 Title: A list of rooms in the house Year: ()
Ref_id:b85 Title: Your Big Five scores (scale 1-5) and human profile Year: ()
Ref_id:b86 Title: Most relevant human intentions proposed at previous times (ignore if empty-this means it's the first intention of the day) Year: ()
Ref_id:b87 Title: Most relevant human tasks proposed at previous times.ids (ignore if empty-this means it's the first task of the day) Year: ()
Ref_id:b88 Title: Intention must align with your Big 5 scores, reflect all aspects of the profile, and be diverse yet reasonable based on the house layout and available objects Year: ()
Ref_id:b89 Title: Intention must be high-level and either human-centric (e.g., hygiene, sport, leisure) or roomcentric (e.g., clean, organize, set-up). Do not mention specific objects Year: ()
Ref_id:b90 Title: Intention must have temporal dependence but be non-repetitive with the previous intentions and tasks Year: ()
Ref_id:b91 Title: Intention: basic descriptions. Reason_human: detailed descriptions of why it follows your Big 5 scores and profile. Reason_intentions: detailed descriptions of why it has temporal dependence with the previous Year: ()
Ref_id:b92 Title: Reason_tasks: detailed descriptions of why it has temporal dependence with the previous, relevant tasks at Year: ()
Ref_id:b93 Title: Human Task Proposal. Input: 1. The proposed intention at current time Year: ()
Ref_id:b94 Title: A dict mapping rigid, static objects to their IDs and rooms Year: ()
Ref_id:b95 Title: Your Big Five scores (scale 1-5) Year: ()
Ref_id:b96 Title: Most relevant human intentions proposed at previous times (ignore if empty-this means it's the first intention of the day) Year: ()
Ref_id:b97 Title: You are a human living in the house. Instructions: 1. Break down the intention into 5 tasks for collaboration with a robot. 2. Task types: -Type 1: Creative, reasonable free-form human motion interacting or approaching a fixed, static object (static objects cannot be moved) with an object in hand provided by the robot (e.g., sit on sofa with TV remote control in hand Year: ()
Ref_id:b98 Title: For interacting with fixed, static objects, use only objects from the given static object dict Year: ()
Ref_id:b99 Title: Both interacting and inhand objects must be specified Year: ()
Ref_id:b100 Title: Tasks should be continuous and logical, and align with your Big 5 scores and profile Year: ()
Ref_id:b101 Title: Tasks must have temporal dependence with the intentions and tasks at previous times Year: ()
Ref_id:b102 Title: Free-form motion should be diverse. Examples: sampled_motion_list. Feel free to propose others Year: ()
Ref_id:b103 Title: Do not output anything else: Time: xxx am/pm Intention: basic descriptions. Tasks: 1. Thought: detailed descriptions of the task Year: ()
Ref_id:b104 Title: A dict mapping rigid, static objects to their IDs and rooms Year: ()
Ref_id:b105 Title: Your Big Five scores and human profile Year: ()
Ref_id:b106 Title: Most relevant human intentions proposed at previous times (if empty, ignore it-this means it's the first intention of the day) Year: ()
Ref_id:b107 Title: Your task is to check if the temporal dependence and human profile are strictly followed in each task, and revise to make better if necessary Year: ()
Ref_id:b108 Title: Tasks should be continuous and logical, and align with your Big 5 scores and profile Year: ()
Ref_id:b109 Title: Tasks must have temporal dependence with the previous intentions and tasks, with detailed explanation mentioning previous intentions and tasks explicitly Year: ()
Ref_id:b110 Title: For interacting with fixed, static objects, use only objects from the given static object dict. For objects in hand, a robot will provide them. Write in the following format. Do not output anything else: Time: xxx am/pm Intention: basic descriptions. Reflect Each Task: 1. no mistake or change made Year: ()
Ref_id:b111 Title: Revised Tasks Year: ()
Ref_id:b112 Title: Reason_human: why it aligns with your Big 5 scores and profile. Reason_intentions: how it depends on previous, relevant intentions at Year: ()
Ref_id:b113 Title: Your task is to check if the instructions are strictly followed in each task, and revise to make better if necessary. Instructions: 1. Break down the intention into 5 tasks for collaboration with a robot. 2. Task types: -Type 1: Creative, reasonable free-form human motion interacting or approaching a fixed, static object (static objects cannot be moved) with an object in hand provided by the robot (e.g., sit on sofa with TV remote control in hand Year: ()
Ref_id:b114 Title: For interacting with fixed, static objects, use only objects from the given static object dict (exact name) Year: ()
Ref_id:b115 Title: Free-form motion should be diverse. Examples: sampled_motion_list. Feel free to propose others Year: ()
Ref_id:b116 Title: Write in the following format. Do not output anything else: Time: xxx am/pm Intention: basic descriptions. Reflect Each Task: 1. no mistake or change made Year: ()
Ref_id:b117 Title: Revised Tasks Year: ()
Ref_id:b118 Title: Reason_human: why it aligns with your Big 5 scores and profile. Reason_intentions: how it depends on previous, relevant intentions at Year: ()
Ref_id:b119 Title: Current human tasks Year: ()
Ref_id:b120 Title: You are the human. Decide if the robot's assistance align with your needs Year: ()
Ref_id:b121 Title: Assess if each robot task supports the human tasks and intention. The robot's task doesn't need to be an exact match but should be relevant in purpose, context, or object categories. Use common reasoning to decide if it helps meet your needs Year: ()
Ref_id:b122 Title: Consider each robot thought and object individually against the human tasks. Approve it if it meets any one of the human tasks; sequence does not matter Year: ()
Ref_id:b123 Title: Be fair in your judgment-avoid being too generous or too harsh Year: ()
Ref_id:b124 Title: Ensure items are in a list. Write in the following format. Do not output anything else: Tasks: [yes, no, ...] Reasons_tasks: 1. ... G Prompt Details of the Assistive Agent We show exact prompts for VLMs in building the assistive agent. Intention Discovery. Input: 1. Sequence of images showing human motion from your and human's perspectives Year: ()
Ref_id:b125 Title: Current time Year: ()
Ref_id:b126 Title: Inferred Big Five personality scores (ignore if empty-this means it's your first collaboration with this human) Year: ()
Ref_id:b127 Title: Inferred human profile (ignore if empty-this means it's your first collaboration with this human) Year: ()
Ref_id:b128 Title: Most relevant human intentions discovered at previous times (ignore if empty-this means it's the first intention of the day) Year: ()
Ref_id:b129 Title: Most relevant human tasks discovered at previous times.ids (ignore if empty-this means it's the first task of the day) Year: ()
Ref_id:b130 Title: Map the observed human motion to 5 possible high-level intentions at the current time (without mentioning the specific motion) Year: ()
Ref_id:b131 Title: Intention must align with human Big 5 scores and reflect all aspects of the profile, and be diverse yet reasonable based on the house layout and available objects Year: ()
Ref_id:b132 Title: Intention must be high-level and either human-centric (e.g., hygiene, sport, leisure) or roomcentric (e.g., clean, organize, set-up). Do not mention specific objects Year: ()
Ref_id:b133 Title: Intention must have temporal dependence but be non-repetitive with the intentions and tasks at previous times in the input. Write in the following format. Do not output anything else: Time: xxx am/pm Intention 1: basic descriptions. Reason_human: detailed descriptions of why it follows the Big 5 scores and profile. Reason_intentions: detailed descriptions of why it has temporal dependence with the previous Year: ()
Ref_id:b134 Title: Reason_tasks: detailed descriptions of why it has temporal dependence with the previous, relevant tasks at Year: ()
Ref_id:b135 Title: Reason_vis: detailed descriptions with respect to the visual cues Year: ()
Ref_id:b136 Title: A dict mapping rigid, static furnitures to their IDs and rooms Year: ()
Ref_id:b137 Title: Inferred Big Five personality scores (ignore if empty-this means it's your first collaboration with this human) Year: ()
Ref_id:b138 Title: Most relevant human intentions discovered at previous times (ignore if empty-this means it's the first intention of the day) Year: ()
Ref_id:b139 Title: Most relevant human tasks discovered at previous times.ids (ignore if empty-this means it's the first task of the day) Year: ()
Ref_id:b140 Title: Task type: For each human task, provide one small, handable object from a magical box. Furnitures in the dict are for room understanding and cannot be used Year: ()
Ref_id:b141 Title: Tasks should be continuous and logical, and align with your Big 5 scores and profile Year: ()
Ref_id:b142 Title: Write in the following format. Do not output anything else: Time: xxx am/pm Intention: basic descriptions. Tasks: 1. Thought: detailed descriptions of the task Year: ()
Ref_id:b143 Title: Input: 1. Human intentions at previous times (ignore if empty-this means it's your first inference) Year: ()
Ref_id:b144 Title: Human tasks at previous times.ids Year: ()
Ref_id:b145 Title: Human profile (ignore if empty-this means it's your first inference) Year: ()
Ref_id:b146 Title: Inferring Big Five personality traits (scale 1-5, float) based on the provided intentions and task Year: ()
Ref_id:b147 Title: Revise the existing human profile if necessary. Write in the following format. Do not output anything else: Scores: {'openness': a, 'conscientiousness': b, 'extroversion': c, 'agreeableness': d Year: ()
