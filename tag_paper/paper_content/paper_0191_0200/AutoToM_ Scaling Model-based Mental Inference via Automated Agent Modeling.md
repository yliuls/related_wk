Title: AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling
Abstract: Theory of Mind (ToM), the ability to understand people's minds based on their behavior, is key to developing socially intelligent agents. Current approaches to ToM reasoning either rely on prompting Large Language Models (LLMs), which are prone to systematic errors, or use handcrafted, rigid agent models for model-based inference, which are more robust but fail to generalize across domains. In this work, we introduce AutoToM , an automated agent modeling method for scalable, robust, and interpretable mental inference. Given a ToM problem, AutoToM first proposes an initial agent model and then performs automated Bayesian inverse planning based on this model, leveraging an LLM backend. Guided by inference uncertainty, it iteratively refines the model by introducing additional mental variables and/or incorporating more timesteps in the context. Across five diverse benchmarks, AutoToM outperforms existing ToM methods and even large reasoning models. Additionally, we show that AutoToM can produce human-like confidence estimates and enable online mental inference for embodied decision-making.

Section: Introduction
To successfully engage in rich and complex social interactions such as cooperation, communication, and social learning, humans must adequately understand one another's mental states (e.g., goals, beliefs, desires). This ability is termed Theory of Mind (ToM) [49]. Prior works have demonstrated that like human interactions, Theory of Mind is also crucial for the success of human-AI interactions [7,14,28]. To safely and productively interact with humans in an open-ended manner, AI systems need to interpret humans' mental states from observed human behavior [5,45,44,31,33,53,51,21].
There are two primary approaches to developing machine Theory of Mind in recent works. First, with the rapid progress of large language models (LLMs), there has been an increasing interest in directly applying LLMs to reason about people's mental states with prompting strategies such as perspective-taking [48,37,22], change-tracking [18], and temporal-spatial reasoning [17]. However, even with these advanced prompting techniques, state-of-the-art LLMs still make systematic errors in complex scenarios [20]. Second, cognitive studies have demonstrated that model-based inference, in particular, Bayesian inverse planning (BIP), can reverse engineer human-like theory of Mind reasoning [4,43,3,52]. BIP relies on Bayesian Theory of Mind (BToM) models [3] to approximate rational agent behaviors. Inspired by this, recent works have proposed to combine BIP and LLMs to ⋮ ⋮ ⋮ ⋮ new model final result model utility s t-1 s t o t-1 o t b t-1 b t a t-1 a t ⋮ ⋮ ⋮ ⋮ Initial 𝑃 𝑣 " # 𝑋 #!:# ∝ -𝑃(𝑣 " # , 𝑉 %" #!:# , 𝑋 #!:# ) & "# $!:$
this section cite: ['b48', 'b6', 'b13', 'b27', 'b4', 'b44', 'b43', 'b30', 'b32', 'b52', 'b50', 'b20', 'b47', 'b36', 'b21', 'b17', 'b16', 'b19', 'b3', 'b42', 'b2', 'b51', 'b2']

Section: Hypothesis Sampling
Figure 1: An overview of AutoToM . X ts:t are observable variables, V ts:t are latent mental variables, and q is the query (in this case, a mental variable v t i ∈ V t ). t s : t denotes timesteps from t s to t in the context that are considered for inference. Variables s t , o t , b t , a t , g t represent state, observation, belief, action, and goal, respectively, with solid arrows indicating dependencies defined in the models. Given a question, we extract the observable variables (information extraction) and propose an initial agent model. This is followed by automated Bayesian inverse planning and iterative model adjustment. When the model utility is high enough, we will produce the final answer based on the inference result.
achieve scalable yet robust model-based ToM inference [20,39]. While these methods significantly outperform LLMs in specific domains, they typically require manual specification of agent models, including necessary mental variables (e.g., goals, beliefs) for answering a given ToM question. Therefore, they lack the required generalizability for open-ended Theory of Mind.
In this work, we aim to develop a fully automated model-based Theory of Mind method. That is a unified method that can be applied to robustly infer any given mental variable in any domain. Achieving this aim requires addressing two critical questions: (1) How can we ensure that our approach is flexible enough to adapt across contexts, robust enough to model diverse human behaviors, and scalable enough to tackle increasingly complex scenarios? (2) How can we avoid manual model specifications and instead automate agent modeling for model-based mental inference?
To address these challenges, we introduce AutoToM , a general framework for model-based Theory of Mind. It automates every aspect of Bayesian inverse planning, including the proposal and adjustment of model structures, the identification of relevant timesteps, the generation of hypotheses, and the execution of Bayesian inference. It is designed to operate in any context, infer any mental state, reason about any number of agents, and support any order of recursive reasoning, which represents our vision of an open-ended and robust machine Theory of Mind.
Figure 1 provides an overview of AutoToM , which consists of two main components: First, Automated Bayesian Inverse Planning conducts Bayesian inference based on any given agent model (in the form of a Bayesian network) using an LLM as a computational backend. Unlike prior works that leverages LLMs for Bayesian inverse planning, it has no assumptions about model structure or variable representations. Second, Automated Agent Model Discovery iteratively constructs and adjusts an agent model most suitable a given ToM inference problem, eliminating the need for manual model specifications typically required by prior works on model-based ToM inference.
Our main contributions include: (1) a unified formulation of model-based ToM inference; (2) the first approach of automated agent model discovery, AutoToM, for scalable model-based ToM; and (3) a systematic evaluation of AutoToM on multiple ToM benchmarks, cognitive studies, and embodied assistance tasks. The results show that AutoToM outperforms state-of-the-art LLMs and large reasoning models, establishing a scalable, robust, and interpretable framework for machine ToM.
this ability: SimToM [48] encourages LLMs to adopt perspective-taking, PercepToM [22] improves perception-to-belief inference by extracting relevant contextual information, and Huang et al. [18] employ an LLM as a world model to track environmental changes and refine prompts. Explicit symbolic frameworks also contribute: TimeToM [17] constructs a temporal reasoning framework to support inference, SymbolicToM [37] uses graphical representations to track characters' beliefs, and thought-tracing [24] traces multiple hypotheses over time. However, these approaches still exhibit systematic errors in handling long contexts, complex behaviors, and recursive reasoning scenarios. Among these works, thought-tracing is closely related to ours, as it also maintains hypotheses of mental variables. Compared to thought-tracing [24], AutoToM performs explicit agent modeling: it constructs Bayesian networks over mental variables and their causal dependencies, rather than tracking only the queried mental variables. This yields higher robustness to wording or superficial story changes (e.g., no need for wording changes in AutoToM ), and improves interpretability, as errors can be analyzed through the model structure. Moreover, AutoToM adaptively minimizes inference complexity by expanding models only when beneficial, preventing under-/over-modeling and improving efficiency on tasks with longer contexts , more agents, and deeper recursion. By contrast, thought-tracing reweights hypotheses without adjusting model structure or temporal depth.
this section cite: ['b19', 'b38', 'b47', 'b21', 'b17', 'b16', 'b36', 'b23', 'b23']

Section: Model-based Theory of Mind inference.
Model-based Theory of Mind inference, particularly Bayesian inverse planning (BIP) [4,43,3,52], explicitly constructs representations of agents' mental states and models how these mental states guide behavior through probabilistic agent models. These methods can reverse engineer human ToM inference in simple domains [e.g., 3,29,40]. Recent works combine BIP with LLMs to improve ToM inference in more realistic settings [20,39]. However, they require manual specification of the agent models as well as rigid, domain-specific implementations of Bayesian inference, limiting their adaptability to open-ended scenarios. To overcome this, we propose AutoToM , a method for automated agent modeling and mental inference across diverse domains.
this section cite: ['b3', 'b42', 'b2', 'b51', 'b2', 'b28', 'b39', 'b19', 'b38']

Section: Automated Modeling with LLMs.
There has been an increasing interest in integrating LLMs with inductive reasoning and probabilistic inference for automated modeling. Piriyakulkij et al. [32] combine LLMs with Sequential Monte Carlo to perform probabilistic inference about underlying rules. Qiu et al. [34] further enhance LLM-based inductive reasoning by iteratively proposing, selecting, and refining textual hypotheses of rules. Li et al. [27] employ LLMs to construct, critique, and refine statistical models represented as probabilistic programs for data modeling. Wang et al. [46] prompt LLMs to generate natural language hypotheses that are then implemented as verifiable programs for inductive reasoning. Hypothetical minds [6] leverage LLMs to propose and evaluate agent strategies for multi-agent planning, but do not specifically infer individual mental variables.
Our method also aims to achieve automated modeling with LLMs. Unlike prior works, we propose a novel automated model discovery approach for Bayesian inverse planning, where the objective is to confidently infer any mental variable given any context by constructing a suitable agent model.
this section cite: ['b31', 'b33', 'b26', 'b45', 'b5']

Section: AutoToM

this section cite: []

Section: Preliminaries: A Unified Formulation of Model-based ToM
Bayesian Inverse Planning (BIP) is a computational framework for model-based ToM inference [4]. It assumes that the agent acts rationally according to a generative agent model [3], which specifies how internal variables lead to observable actions in a Bayesian network (e.g., the example models on the bottom panels in Figure 2a. Using inverse inference, BIP inverts this generative process to assess what latent mental variables can lead to observed agent behavior. This probabilistic inference reasons about how agents make decisions, serving as a robust solution to ToM challenges.
There have been different instantiations of BIP in prior works [e.g., 4,43,30,19]. Here we formally define BIP in a unified manner. We denote the observable variables at time t describing the environment and an agent's behaviors as X t = {x t i } i∈N X , where N X is the set of observable variables and x t i is a particular variable (state, action, or utterance) at t. We can extract the values of these observable variables from the context provided in a ToM problem. We denote an agent's latent mental variables at time t as V t = {v t i } i∈N V , where N V is the set of mental variables and v t i is a particular mental variable (e.g., goal, desire, belief) at t. BIP formulates an agent model as a Bayesian network that defines P (V t , X t ), which indicates how the mental variables drive an agent's behavior. Given Story: Mei is a pearl diver in a small coastal village in Japan. Mei wants to find a perfect pearl to give to her grandmother for her birthday. Mei spots an oyster at the bottom of the sea that looks to be the right size and age to contain a pearl. Mei believes that the oyster she spotted contains a pearl. A curious octopus opens the oyster, revealing that there is no pearl inside, and then swims away. Mei dives down to collect the oyster. Question: Does Mei believe the oyster she spotted contains a pearl or that it is empty? Story: The milk is on the table. Sally exited the room. Anne transferred the milk onto the box. Alex exited the room, then Anne exited the room. Outside the room, the three interacted with each other -Alex lied to all: The milk is in the fridge! Sally secretly told Anne: The milk is on the table! Question: Where does Alex think Sally thinks Anne thinks the milk is? BigToM ToMi MMToM-QA MuMA-ToM Hi-ToM Model (a) AutoToM constructs appropriate agent models tailored to different scenarios (b) AutoToM produces human-like confidence estimates as observed in cognitive studies (c) AutoToM enables online mental inference to support embodied decision-making Model Model Model Model Human AutoToM 👤 🤖 Which one is the agent's goal object? (Baker et al. (2009)) Which type of food does the agent want? Which type of food truck does the agent believe is in the corner? (Baker et al. (2017)) Human
this section cite: ['b3', 'b2', 'b3', 'b42', 'b29', 'b18']

Section: 👤

this section cite: []

Section: AutoToM

this section cite: []

Section: 🤖
None.
Alice opens the fridge.
this section cite: []

Section: Goal inferred by AutoToM
Alice puts a plate on the table.
this section cite: []

Section: Observed Main's actions

this section cite: []

Section: Helper's action
Alice closes the fridge without taking anything.
Hypo 1: Put a salmon in the oven. (prob = 0.34) Hypo 2: Put an apple on the plate. (prob = 0.16) Hypo 1: Put a wine glass in the dishwasher. (prob = 0.32) Hypo 2: Put a plate and a fork on the table. (prob = 0.15) Hypo 1: Put a plate and a fork on the table. (prob = 0.81) Hypo 2: Put a plate and a wine glass on the table. (prob = 0.12) ⋮ None. Get the fork from the fridge and put it on the table. this model, BIP infers the latent mental variables for the current step t:
P (V t |X t ) = P (V t , X t )/ V P (V, X t ) ∝ P (V t , X t ).(1)
In many real-world scenarios, past observations (such as actions taken at the previous steps) are often valuable for inferring the mental variables at the current step. Suppose the context from step t s to step t is relevant for the current mental variable inference, then the inference becomes:
P (V ts:t |X ts:t ) ∝ P (V ts:t , X ts:t ).(2)
In a ToM problem, there is a query concerning a specific target variable q to be inferred. We can answer the query via P (q|X ts:t ). Typically, the query asks about a latent mental variable q = v t i ∈ V t , the posterior probability is obtained by marginalizing over other latent variables V ts:
t -i which is the subset of V ts:t excluding v t i : P (v t i |X ts:t ) ∝ V ts :t -i P (v t i , V ts:t -i , X ts:t ). (3) This can also be extended to predicting a future observable variable q = x t+1 i : P (x t+1 i |X ts:t ) ∝ V ts:t P (V ts:t , x t+1 i , X ts:t ).
To conduct BIP in different scenarios, we must formulate the mental variables and their causal relationships with agent behavior using suitable agent models. Each model M is uniquely defined by
s t o t b t a t g ⋮ ⋮ ⋮ ⋮ b t 1: Jennifer thinks that the plate is not inside the fridge b t 2: Jennifer thinks that the plate is inside the fridge. o t 1: Jennifer sees that the plate is inside the fridge. o t 2: Jennifer sees that the plate is not inside the fridge. o t 3: Jennifer has no clues about the plate's location. 𝑃 𝑎 ! 𝑏" ! , 𝑔) = 0.2 𝑃 𝑎 ! 𝑏# ! , 𝑔) = 0.3 𝑃 𝑜" ! 𝑠 ! ) = 0.01 low likelihood → hypothesis reduction 𝑃 𝑏" ! 𝑏" !$" , 𝑜# ! ) = 0.4 𝑃 𝑏" ! 𝑏# !$" , 𝑜% ! ) = 0.5 ⋮ 𝑃 𝑏" ! 𝑠 ! , 𝑎 ! ) = 0.2 𝑃 𝑏# ! 𝑠 ! , 𝑎 ! ) = 0.8 Answer: b t 2 → 𝑃 𝑣& ! 𝑋 !!:! ∝ 6 𝑃(𝑣& ! , 𝑉 $& !!:! , 𝑋 !!:! ) ( "# $!:$ Final probability: Bayesian Inference Hypothesis Sampling (a) Automated Bayesian inverse planning. s t-1 s t b t-1 b t a t-1 a t s t-1 s t a t-1 a t g g Before After s t-1 s t a t-1 a t g s t-1 s t a t-1 a t g s t-2 a t-2 Before After Variable Adjustment Timestep Adjustment (b) Model adjustments. the observable variables and the latent mental variables, i.e., M = (V ts:t , X ts:t ). Let s t ∈ S be the state at time t, and a t ∈ A be the action taken by the agent at time t. The current state and action determines the next state s t+1 . When the agent has an explicit goal g ∈ G, this setup constitutes a Markov Decision Process (MDP). If the agent only has a partial observation of the state, the model becomes a Partially Observable Markov Decision Process (POMDP) [23]. In POMDP, the agent receives a partial observation o t of the true state s t , maintains a belief b t over the possible states, and selects its action a t based on this belief and goal. When there is high-order recursive reasoning between two agents (i and j), we can adopt an Interactive POMDP (I-POMDP) [12], where the belief of state at level l > 0 for agent i will become the belief of interactive state is t = (s, b j,l-1 , g j ), where b j,l-1 is the belief of agent j at the lower level l -1 and g j is agent j's goal.
this section cite: ['b22', 'b11']

Section: Overview of AutoToM
As shown in Figure 1, AutoToM aims to construct a suitable agent model for Bayesian inverse planning to confidently infer any target variable. There are several key challenges in achieving this: First, different ToM inference problems require different agent models (as illustrated in Figure 2a). Second, our method must determine which timesteps in the context are relevant. Third, there is no predefined hypothesis space for each variable, and each space could be infinite. Last, to infer mental variables in any context, we must flexibly represent them without manual specifications.
AutoToM addresses these challenges in the two key components: (1) automated Bayesian inverse planning (Section 3.3), which conducts BIP given a specified agent model, and (2) automated agent model discovery (Section 3.4), which proposes and adjusts the agent model based on the question and the inference results. These two components form a self-improvement loop to iteratively update the agent model and the corresponding inference result. More details are provided in Appendix A.
this section cite: []

Section: Automated Bayesian Inverse Planning
Given an agent model, M , including the necessary latent mental variables V ts:t and the observable variables X ts:t , we integrate LLMs as the computational backend to implement every aspect of the Bayesian inverse planning. In particular, the hypothesis sampling module suggests a small set of possible values of latent variables. The Bayesian inference module then computes the posterior distribution of the target variable in the query based on Eqn. (3) or Eqn.( 4).
this section cite: []

Section: Hypothesis Sampling.
Conventional BIP assumes a manually defined hypothesis space and representation for each latent mental variable. Our hypothesis sampling module instead leverages an LLM to propose only a small set of quality hypotheses for each latent variable in V ts:t . This is akin to amortized inference [35,19]. To ensure that the sampled hypotheses are relevant to the ToM inference, we guide the sampling process with both the question and the observable variables X ts:t . To remove spurious hypotheses generated by the LLM, we further apply hypothesis reduction to eliminate unlikely hypotheses and reduce the hypothesis space. Unlikely hypotheses are identified by evaluating the local conditionals. For instance, we discard observation hypotheses with low likelihood conditioned on the state as shown in Figure 3a.
this section cite: ['b34', 'b18']

Section: Bayesian Inference.
As shown in Figure 3a, we estimate each local conditional in P (V ts:t , X ts:t ) using an LLM. After marginalizing the joint distribution over non-target latent variables via explicit calculation, we then produce the posterior probabilities of the target variable, i.e., Eqn. (3). This also applies to predicting a future observable variable, i.e., Eqn. (4).
Our automated BIP greatly generalizes prior methods that combine BIP and LLMs, such as BIP-ALM [20] and LIMP [39]. Specifically, prior methods assume a fixed model structure defined for a specific ToM problem and require handcrafted, domain-specific representations for physical and mental states.
They also cannot propose hypotheses for non-target latent variables. For instance, to infer an agent's goal, BIP-ALM conducts a manual belief update while LIMP has no explicit belief update at all. In contrast, AutoToM can conduct any ToM inference based on any agent model structure and consider multiple non-target latent variables simultaneously. Additionally, unlike prior methods, our Bayesian inference can work with arbitrary levels of recursion for high-order ToM inference.
this section cite: ['b19', 'b38']

Section: Automated Agent Model Discovery
Prior works on model-based ToM inference rely on manually designed agent models, limiting their applicability to domain-specific scenarios. In contrast, the Automated Model Discovery component automatically proposes a model and dynamically adjusts it to ensure both the effectiveness of the model-confidently inferring agents' mental states-and the efficiency of the inference by minimizing model complexity. To achieve this, we formulate the utility of a model M = (V ts:t , X ts:t ) used for answering a given query q as U (M, q) = R(M, q) -C(M ),
where R(M, q) assesses the model's confidence in answering the query, and C(M ) is its computational cost. In this work, the reward is defined as R(M, q) = -H(P (q|X ts:t )), where P (q|X ts:t ) is the probability distribution of the target variable based on Eqn. (3) or Eqn. (4), and H(•) is its entropy. This is designed to decrease the uncertainty in the inference. To minimize the compute needed for the inference, we define the cost of the model as C(M ) = α|M |, where |M | denotes the model's complexity, measured by the number of latent mental variables, and α > 0 is a weighting factor. The cost increases with complexity, encouraging parsimonious models with lower compute.
There are three modules for Automated Model Discovery:
Information Extraction. This module extracts the values of observable variables X 1:t from the context, including states (s t ), actions (a t ), and utterances (u t ), organized along a timeline (the number of timesteps is determined by the number of actions and utterances). When there are multiple agents, we identify whose mental state the question is asking about (i.e., the target agent), and then construct the timesteps based on the target agent's actions and/or utterances. The extraction is performed once using an LLM and used for model proposal and Bayesian inverse planning.
this section cite: []

Section: Initial Model Proposal.
We employ an LLM to propose an initial agent model based on X 1:t and the query. This initial model has minimal complexity, containing only the essential mental variables needed to answer the question. This initial proposal also assesses the level of recursive reasoning necessary for higher-order ToM inference. Note that we always begin with only considering the last timestep in context, i.e., t s = t. Following this model, we conduct automated Bayesian inverse planning, as described in Section 3.3. If the model utility exceeds a threshold U min , we accept the inference result as the final answer. Otherwise, we use the model utility to guide model adjustments.
this section cite: []

Section: Model Adjustment.
We iteratively adjust the proposed model to maximize the utility by considering two types of model adjustments: variable adjustment (Figure 3b) and timestep adjustment (Figure 3b):
Variable Adjustment. We refine the model structure at a specific timestep by iteratively introducing new, relevant latent variables into the model to address uncertainty in the inference. These variables include goal, belief, observation, and interactive state as summarized in Table 4 in Appendix A. This follows the typical causal structures introduced in prior decision-making models [e.g., 23,3,43,12]. Such restricted variable adjustment helps reduce the model space and ensures the proposed models can explain human behavior. For each adjustment, we compute the updated model utility and accept the modification that offers the biggest increase in utility. This iterative process continues until no further significant improvements are possible. Note that our method can still propose diverse models beyond standard MDP, POMDP, and I-POMDP, even with this restricted model adjustment. Appendix A.5 provides more details on the model space. Timestep Adjustment. If model utility remains low and no significant improvement can be achieved via variable adjustment within the current timesteps t s : t, we incorporate an additional step, t s -1, to enhance context for inference. Upon adding a timestep, we first apply the initial model structure and then adjust variables accordingly.
We iterate the variable and timestep adjustments until either the model utility exceeds the desired threshold or no further meaningful improvement is possible.
this section cite: ['b22', 'b2', 'b42', 'b11']

Section: Experiments

this section cite: []

Section: Experiment 1: Evaluation on ToM Benchmarks
Setting. We evaluated our method on multiple Theory of Mind benchmarks, including ToMi [26], BigToM [11], MMToM-QA [20], MuMA-ToM [39], and Hi-ToM [15]. The diversity and complexity of these benchmarks pose significant reasoning challenges. For instance, MMToM-QA and MuMA-ToM incorporate both vision and language inputs, while MuMA-ToM and Hi-ToM require higher-order inference. Additionally, MMToM-QA features exceptionally long contexts, and BigToM presents open-ended scenarios.
We compared AutoToM against state-of-the-art baselines:
• LLMs: Llama 3.1 70B [9], GPT-4o [1], Gemini 2.0 Flash and Gemini 2.0 Pro [41];
• ToM Prompting for LLMs: SymbolicToM [37] and SimToM [48];
• Large Reasoning Models: DeepSeek-R1 [13], Gemini 2.0 Flash Thinking, and o3-mini-high;
• Model-based Inference: BIP-ALM [20] and LIMP [39].
We use GPT-4o as the LLM backend for AutoToM and all ToM prompting and model-based inference baselines to ensure a fair comparison. For multimodal benchmarks, MMToM-QA and MuMA-ToM, we adopt the information fusion methods proposed by Jin et al. [20] and Shi et al. [39] to fuse information from visual and text inputs, respectively. The fused information is in text form. We ensure that all methods use the same fused information as their input.
Results. The main results are summarized in Table 1. AutoToM demonstrates the strongest overall performance among all methods, including large reasoning models. Specifically, it outperforms its LLM backend, GPT-4o, by a large margin. This is because AutoToM is more robust for inferring mental states given long contexts with complex environments and agent behavior. It is also more adept at recursive reasoning, which is key to higher-order inference. Compared to prior model-based methods, it exhibits superior generalization across different domains. This is enabled by our agent model discovery and the automated BIP.
① Information Extraction 𝑃(𝑋 !:# ) Adjusted Model ② Initial Model Proposal s 0 s 1 o 0 o 1 b 0 b 1 a 0 a 1 s 0 s 1 o 0 o 1 b 0 b 1 ③ Bayesian Inference 𝑃(𝑏 ! ! = almond | 𝑋 ":! ) = 0.5 𝑃(𝑏 $ ! = oat | 𝑋 ":! ) = 0.5 𝑃 𝑏 ! |𝑠 ! , 𝑎 ! ∝ / % ! ,' " 𝑃 𝑏 ! 𝑏 " , 𝑜 ! 𝑃 𝑜 ! |𝑠 ! 𝑃(𝑏 " ) 𝑃(𝑏 ! ! = almond | 𝑋 ":! ) = 0.1 𝑃(𝑏 $ ! = oat | 𝑋 ":! ) = 0.9 𝑃 𝑏 ! |𝑠 ! , 𝑎 ! ∝ / % ! ,' " ,( 𝑷(𝒂 𝟏 |𝒃 𝟏 , 𝒈)𝑃 𝑏 ! 𝑏 " , 𝑜 ! 𝑃 𝑜 ! |𝑠 ! 𝑃(𝑏 " )𝑷(𝒈) ⑤ Bayesian Inference Model Utility high ✅ (Confident) a 0 a 1 g state t
The milk pitcher contains no milk. The milk pitcher contains almond milk.
this section cite: ['b25', 'b10', 'b19', 'b38', 'b14', 'b8', 'b0', 'b40', 'b36', 'b47', 'b12', 'b19', 'b38', 'b19', 'b38']

Section: action t
Noor makes the cappuccino using the milk in the pitcher. Noor grabs a milk pitcher and fills it with oat milk.
this section cite: []

Section: Model Utility low ❌ (Uncertain)

this section cite: []

Section: ④ Model Adjustment
Story: Noor is working as a barista at a busy coffee shop. Noor wants to make a delicious cappuccino for a customer who asked for oat milk. Noor grabs a milk pitcher and fills it with oat milk. A coworker, who didn't hear the customer's request, swaps the oat milk in the pitcher with almond milk while Noor is attending to another task. Noor makes the cappuccino using the milk in the pitcher.   [11]. We show the results from each key model step. It demonstrates how AutoToM adjusts the agent model to increase inference confidence. Detailed procedures of Bayesian inference for both the proposed and adjusted models are provided in Appendix C.5.
We also compared the performance of AutoToM with large reasoning models across different conditions, summarized over all benchmarks. These include question types, the context length, the number of agents, and the level of recursion. As shown in Figure 4, AutoToM demonstrates robust scalability and exhibits a much lower degree of volatility under different conditions than large reasoning models. We provide additional results and evaluations in Appendix C.2 and C.3.
We further report the token cost and inference time comparison on MMToM-QA in Appendix C.1. AutoToM achieves higher reasoning performance with comparable or lower computational cost, highlighting its efficiency and scalability.
this section cite: ['b10']

Section: Ablation Study.
We evaluated the following variants of AutoToM for an ablation study: no hypothesis reduction (w/o hypo. reduction); always using POMDP (w/ POMDP); always using the initial model proposal without variable adjustment (w/o variable adj.); only considering the last timestep (w/ last
Table 2: Performance comparison on MMToM-QA. LLM indicates the model itself; AutoToM represents our method with the corresponding model as the backend.
this section cite: []

Section: LLM AutoToM
GPT-4o 44.0 83.0
Qwen3-235b-a22b-2507 45.0 67.5
DeepSeek-chat-v3-0324 34.8 71.1
Gemini-2.5-Flash (thinking disabled) 44.7 71.7 timestep); and considering all timesteps without timestep adjustment (w/ all timesteps). The results in Figure 6 show that the full AutoToM method constructs a suitable agent model, enabling rich ToM inferences while reducing compute. In particular, key model components, including hypothesis reduction, variable adjustment, and timestep adjustment, optimize efficiency without sacrificing performance. Full ablation results are provided in Appendix C.4. Sensitivity to LLM Backends. To test AutoToM 's performance sensitivity to LLM backends, we conducted additional experiments using alternative models. Note that we used the same prompt for each backend LLM. Specifically, we replace the GPT-4o backend with Qwen3-235B (open-sourced), DeepSeek-V3 (open-sourced), and Gemini-2.5-flash (thinking disabled) on the most challenging MMToM-QA benchmark. Notably, AutoToM with any LLM as the backend outperforms the corresponding LLM performance by a large margin (Table 2). Crucially, we achieve this without extra prompt engineering.
this section cite: []

Section: Statistical Reliability.
To assess result stability, we additionally ran multiple trials on the most challenging benchmark, MMToM-QA. Across three different random seeds, AutoToM achieved a mean accuracy of 82.56% with a standard error of 0.45%, which is consistent with the 83.00% reported in Table 1. Similarly, o3-minihigh achieved a mean accuracy of 65.94% with a standard error of 0.59%. These results indicate that the evaluation is stable across runs, and our conclusions remain robust.
this section cite: []

Section: Experiment 2: Evaluation on Classic Cognitive Studies
Setting. AutoToM produces posterior distributions over the hypothesis space, offering uncertainty estimates. This allows us to compare the model uncertainties with human judgments. We adapted two well-known cognitive studies on human ToM: online goal inference in [4] and desire and belief inferences in the food truck scenarios [3]. As shown in Figure 2b, in each study, participants were shown agent behavior in a 2D gridworld and asked to judge the agent's goal in [4] and desires and beliefs in [3]. A capable model needs to sequentially update multiple hypotheses with varying degrees of confidence that closely resemble human judgment.
In this experiment, we generated captions for the frames in both tasks and evaluated AutoToM on all available types of scenarios, using the posterior probabilities from AutoToM as its confidence. For baseline, we asked GPT-4o and o3-mini-high to produce confidence scores for each hypothesis in all trials, given the same captions. Implementation details are provided in Appendix D.2.
Results. We computed the correlation between model responses and human judgments reported in the original studies. As shown in Table 3, AutoToM aligns well with human confidence judgments on all three tasks. In particular, AutoToM demonstrates a substantially higher correlation with humans than GPT-4o and o3-mini-high in more complex tasks with a partially observable environment. The results indicate that AutoToM is able to produce nuanced confidence estimates that closely mirror human inference patterns in different environments. We provide additional results in Appendix D.1.  Setting. As recent cognitive studies have suggested, humans routinely utilize ToM to improve our decision making in multi-agent settings [47,16]. To evaluate whether AutoToM can help improve multi-agent decision making, we further evaluated it in an embodied assistance benchmark, Online Watch-And-Help (O-WAH) [33], where a helper agent must simultaneously observe a main agent's actions, infer its goal, and assist it to reach the inferred goal faster in realistic household environments. In these tasks, a ToM model must update its inference of the main agent's goal based on the latest observations in an online manner. Given the goal inference at each step, we adopted the uncertainty-aware helping planner proposed in [33] to generate helping actions accordingly. There are 4 task categories (setting the table, putting groceries in the fridge, preparing a simple meal, washing dishes). We evaluated each method across 20 episodes, with 5 episodes in each task category. To reduce variance, the results are reported as the average over 3 runs per episode.
this section cite: ['b3', 'b2', 'b3', 'b2', 'b46', 'b15', 'b32', 'b32']

Section: Experiment 3: Embodied Assistance
As shown in Figure 2c, we applied AutoToM to online goal inference. Specifically, AutoToM constructs an agent model at each step and maintains the goal hypotheses and corresponding probabilities using Sequential Monte Carlo (SMC) [8]. We also paired the same planner with two baseline goal inference methods: Random Goal (i.e., randomly sampling a goal) and GPT-4o for online goal inference. We did not evaluate any large reasoning models due to their slow inference speed (more than 1 minute per timestep), which makes it impractical for online embodied assistance tasks.
Results. As shown in Figure 7, the Random Goal baseline achieves a 6.3% speedup, but with high variance and negative speedup in 50% of the episodes. GPT-4o achieves a similar but more stable speedup of 6.8%. In contrast, AutoToM achieves the highest speedup of 27.7%, significantly outperforming all baselines. This is because AutoToM can produce more accurate uncertainty estimation of goal hypotheses based on observed actions, which is key to generating robust and useful helping plans. Additional details are provided in Appendix E.
this section cite: ['b7']

Section: Conclusion
We have proposed AutoToM , a novel framework for scalable model-based Theory of Mind. Given any ToM inference problem, AutoToM can automatically construct a suitable agent model and conduct automated Bayesian inverse planning with an LLM backend. Our experimental results have demonstrated that AutoToM can answer different Theory of Mind questions in diverse scenarios, significantly outperforming baselines. We have also shown that AutoToM can produce human-like confidence estimation about mental inferences in classic cognitive studies, and conduct online goal inference for enhancing embodied assistance in complex household scenarios. AutoToM suggests a promising direction toward cognitively grounded ToM modeling that is scalable and robust.
Limitations and Future Work. AutoToM currently requires a separate process to first fuse information from different modalities into text before inference. In the future, we intend to investigate a natively supported multimodal capacity. Additionally, model adjustments may sometimes fail to recognize the relevance of certain mental variables, resulting in an insufficient model. In the future, we intend to further improve the robustness of AutoToM while reducing its inference cost by exploring the possibility of implicit model proposal and Bayesian inference.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report Year: ()
Ref_id:b1 Title: Textual time travel: A temporally informed approach to theory of mind Year: (2021)
Ref_id:b2 Title: Rational quantitative attribution of beliefs, desires and percepts in human mentalizing Year: (2017)
Ref_id:b3 Title: Action understanding as inverse planning Year: (2009)
Ref_id:b4 Title: Stylepredict: Machine theory of mind for human driver behavior from trajectories Year: (2020)
Ref_id:b5 Title: Hypothetical minds: Scaffolding theory of mind for multi-agent tasks with large language models Year: (2024)
Ref_id:b6 Title: Socially intelligent robots: dimensions of human-robot interaction Year: (1480)
Ref_id:b7 Title: Sequential monte carlo samplers Year: (2006)
Ref_id:b8 Title: Angela Fan, and 1 others. 2024. The llama 3 herd of models Year: ()
Ref_id:b9 Title: Somi-tom: Evaluating multi-perspective theory of mind in embodied social interactions Year: (2025)
Ref_id:b10 Title: Understanding social reasoning in language models with language models Year: (2024)
Ref_id:b11 Title: A framework for sequential planning in multi-agent settings Year: (2005)
Ref_id:b12 Title: Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: ()
Ref_id:b13 Title: Cooperative inverse reinforcement learning Year: (2016)
Ref_id:b14 Title: Hi-tom: A benchmark for evaluating higher-order theory of mind reasoning in large language models Year: (2023)
Ref_id:b15 Title: Planning with theory of mind Year: (2022)
Ref_id:b16 Title: Timetom: Temporal space is the key to unlocking the door of large language models' theory-of-mind Year: (2024)
Ref_id:b17 Title: A notion of complexity for theory of mind via discrete world models Year: (2024)
Ref_id:b18 Title: Neural amortized inference for nested multi-agent reasoning Year: (2024)
Ref_id:b19 Title: Mmtom-qa: Multimodal theory of mind question answering Year: (2024)
Ref_id:b20 Title: The era of real-world human interaction: Rl from user conversations Year: (2025)
Ref_id:b21 Title: Perceptions to beliefs: Exploring precursory inferences for theory of mind in large language models Year: (2024)
Ref_id:b22 Title: Planning and acting in partially observable stochastic domains Year: (1998)
Ref_id:b23 Title: Hypothesis-driven theory-of-mind reasoning for large language models Year: (2025)
Ref_id:b24 Title: Fantom: A benchmark for stress-testing machine theory of mind in interactions Year: (2023)
Ref_id:b25 Title: Revisiting the evaluation of theory of mind through question answering Year: (2019)
Ref_id:b26 Title: Automated statistical model discovery with language models Year: (2024)
Ref_id:b27 Title: Goal inference improves objective and perceived performance in human-robot collaboration Year: (2018)
Ref_id:b28 Title: Phase: Physically-grounded abstract social events for machine social perception Year: (2021)
Ref_id:b29 Title: Computational models of emotion inference in theory of mind: A review and roadmap Year: (2019)
Ref_id:b30 Title: Proactive robot assistance via spatio-temporal object modeling Year: (2022)
Ref_id:b31 Title: Doing experiments and revising rules with natural language and probabilistic reasoning Year: (2024)
Ref_id:b32 Title: Nopa: Neurallyguided online probabilistic assistance for building socially intelligent home assistants Year: (2023)
Ref_id:b33 Title: Nouha Dziri, and 1 others. 2023. Phenomenal yet puzzling: Testing inductive reasoning capabilities of language models with hypothesis refinement Year: ()
Ref_id:b34 Title: Deep amortized inference for probabilistic programs Year: (2016)
Ref_id:b35 Title: Neural theory-of-mind? on the limits of social intelligence in large lms Year: (2022)
Ref_id:b36 Title: Minding language models'(lack of) theory of mind: A plug-and-play multi-character belief tracker Year: (2023)
Ref_id:b37 Title: Clever hans or neural theory of mind? stress testing social reasoning in large language models Year: (2023)
Ref_id:b38 Title: Muma-tom: Multi-modal multi-agent theory of mind Year: (2024)
Ref_id:b39 Title: Agent: A benchmark for core psychological reasoning Year: (2021)
Ref_id:b40 Title: Katie Millican, and 1 others. 2023. Gemini: a family of highly capable multimodal models Year: ()
Ref_id:b41 Title: Large language models fail on trivial alterations to theory-of-mind tasks Year: (2023)
Ref_id:b42 Title: Help or hinder: Bayesian models of social goal inference Year: (2009)
Ref_id:b43 Title: Handmethat: Human-robot communication in physical and social environments Year: (2022)
Ref_id:b44 Title: Towards mutual theory of mind in human-ai interaction: How language reflects what students perceive about a virtual teaching assistant Year: (2021)
Ref_id:b45 Title: Hypothesis search: Inductive reasoning with language models Year: (2023)
Ref_id:b46 Title: Altruistic helping in human infants and young chimpanzees Year: (2006)
Ref_id:b47 Title: Think twice: Perspective-taking improves large language models' theory-of-mind capabilities Year: (2023)
Ref_id:b48 Title: Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children's understanding of deception Year: (1983)
Ref_id:b49 Title: Opentom: A comprehensive benchmark for evaluating theory-of-mind reasoning capabilities of large language models Year: (2024)
Ref_id:b50 Title: GOMA: Proactive embodied cooperative communication via goal-oriented mental alignment Year: (2024)
Ref_id:b51 Title: Online bayesian goal inference for boundedly rational planning agents Year: (2020)
Ref_id:b52 Title: Pragmatic instruction following and goal assistance via cooperative language-guided inverse planning Year: (2024)
